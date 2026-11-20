"""O multiverso repousa em disco — item 58, 28/08/2026.

**Escolha feita sobre medição:** gravar o **estado final e a conclusão**, e não célula a
célula. Os números que decidiram estão na tabela acima de `MultiversoRepository` em
`persistence/repositories.py`; o que este módulo verifica é que a gravação **acontece**, que ela
diz a verdade sobre o que aconteceu, e que uma execução que o programa não terminou fica marcada
como tal em vez de dizer `executando` para sempre.

**SQLite em arquivo temporário e não Postgres**, para isto ser portão de commit e não teste que só
roda com serviço em pé. O que se mede aqui é o comportamento do repositório; o que difere
entre dialetos é como o `ALTER` chega ao banco, e disso trata
`tests/integration/test_migrations.py`.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import Engine, create_engine, text

from refino.desktop.banco import migra
from refino.persistence.repositories import (
    EstabilidadeParaGravar,
    MultiversoRepository,
)

AMOSTRAS: list[dict[str, Any]] = [
    {"name": "QC001", "injection_order": 1, "sample_type": "qc", "grupo": None, "batch": "A"},
    {"name": "S001", "injection_order": 2, "sample_type": "sample", "grupo": "ATLL", "batch": "A"},
]

ABERTURA: dict[str, Any] = {
    "estudo": "estudo-de-teste",
    "mode": "neg_c18",
    "checksum_entrada": "a" * 64,
    "n_samples": 2,
    "n_features": 10,
    "amostras": AMOSTRAS,
    "n_specs": 96,
}


@pytest.fixture
def motor(tmp_path: Path) -> Iterator[Engine]:
    url = f"sqlite:///{(tmp_path / 'refino.db').as_posix()}"
    migra(url)
    eng = create_engine(url)
    yield eng
    eng.dispose()


@pytest.fixture
def repo(motor: Engine) -> MultiversoRepository:
    return MultiversoRepository(motor)


def _estabilidade(n: int) -> list[EstabilidadeParaGravar]:
    """`n` medidas com frações que **DIVERGEM**, e a divergência é o ponto.

    Foi medido que nos dados reais **todas** as frações valem zero — 24 de 24 pares nos quatro
    modos. Um fixture que reproduzisse isso tornaria vácua qualquer asserção sobre **ordem**:
    ascendente e descendente dariam a mesma tabela, e uma implementação que ignorasse a ordenação
    passaria. A pergunta — *a minha planta cai sobre dado em que as duas coisas que ela
    separa diferem?* — vale também para o insumo.
    """
    return [
        EstabilidadeParaGravar(
            feature=f"f{i:03d}",
            frac_significant=i / n,
            n_cells=96,
            median_qvalue=None if i % 2 else 0.01 * i,
            median_rank=float(i),
        )
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# A abertura, e o que ela deixa em disco
# ---------------------------------------------------------------------------


def test_a_abertura_grava_uma_linha_executando_sem_fim(repo: MultiversoRepository) -> None:
    """Uma linha, `executando`, `terminado_em` vazio, e `n_feitas` em zero."""
    repo.abre_multiverso(identificador="mv-1", **ABERTURA)

    linha = repo.lista_multiversos()

    assert len(linha) == 1, f"a abertura devia gravar exactamente uma linha, e gravou {len(linha)}"
    (m,) = linha
    assert m.identificador == "mv-1"
    assert m.estado == "executando"
    assert m.terminou is False
    assert m.terminado_em is None
    assert (m.n_feitas, m.n_specs) == (0, 96)
    assert (m.estudo, m.mode) == ("estudo-de-teste", "neg_c18")
    assert m.n_features == 0, "não há conclusão antes do fecho"


def test_a_mesma_entrada_duas_vezes_reusa_a_matriz_e_grava_duas_execucoes(
    repo: MultiversoRepository, motor: Engine
) -> None:
    """**O 500 medido em 26/08/2026, na porta do multiverso.**

    `uq_matrices_origem` diz que `(study_id, mode, checksum_entrada)` é única, e disparar duas
    grades sobre os mesmos arquivos é a coisa mais natural que o profissional faz. A função
    partilhada `_garante_estudo_e_matriz` é o que impede a segunda de violar a restrição — e este
    teste é o que impede a partilha de ser desfeita sem que nada acuse.
    """
    repo.abre_multiverso(identificador="mv-1", **ABERTURA)
    repo.abre_multiverso(identificador="mv-2", **ABERTURA)

    with motor.connect() as c:
        matrizes = c.execute(text("SELECT count(*) FROM matrices")).scalar_one()
        amostras = c.execute(text("SELECT count(*) FROM samples")).scalar_one()

    assert matrizes == 1, f"a mesma entrada gerou {matrizes} matrizes; devia reusar a primeira"
    assert amostras == len(AMOSTRAS), (
        f"as amostras foram reinseridas sobre a matriz reaproveitada: {amostras}"
    )
    assert {m.identificador for m in repo.lista_multiversos()} == {"mv-1", "mv-2"}


def test_o_identificador_repetido_e_recusado_pelo_banco(repo: MultiversoRepository) -> None:
    """`uq_multiverse_runs_identificador` — duas aberturas com o mesmo UUID é defeito, não dado."""
    repo.abre_multiverso(identificador="mv-1", **ABERTURA)

    with pytest.raises(Exception, match=r"(?i)unique"):
        repo.abre_multiverso(identificador="mv-1", **ABERTURA)


# ---------------------------------------------------------------------------
# O fecho, nos quatro desfechos
# ---------------------------------------------------------------------------


def test_o_fecho_concluido_grava_o_estado_e_a_conclusao(repo: MultiversoRepository) -> None:
    """O caminho feliz: `concluido`, `n_feitas == n_specs`, e a tabela de estabilidade em disco."""
    repo.abre_multiverso(identificador="mv-1", grupos=("ATLL", "HAM"), **ABERTURA)

    gravou = repo.fecha_multiverso(
        "mv-1", estado="concluido", n_feitas=96, estabilidade=_estabilidade(5)
    )

    assert gravou is True
    conclusao = repo.le_multiverso("mv-1")
    assert conclusao is not None
    assert conclusao.execucao.estado == "concluido"
    assert conclusao.execucao.terminou is True
    assert conclusao.execucao.terminado_em is not None
    assert conclusao.execucao.n_feitas == 96
    assert conclusao.execucao.grupos == ("ATLL", "HAM")
    assert len(conclusao.estabilidade) == 5


def test_a_conclusao_volta_do_disco_com_os_sete_campos(repo: MultiversoRepository) -> None:
    """**Bijeção campo a campo, e é ela que impede a gravação de perder metade em silêncio.**

    Um `INSERT` que omitisse `rank_iqr` deixaria a coluna nula, a leitura devolveria `None`, e nada
    reprovaria — o histórico mostraria menos do que a tela mostrou no dia, e pareceria uma
    propriedade dos dados.
    """
    repo.abre_multiverso(identificador="mv-1", **ABERTURA)
    medida = EstabilidadeParaGravar(
        feature="2.25_810.7739m/z",
        frac_significant=0.75,
        n_cells=4320,
        median_qvalue=0.031,
        median_rank=12.5,
        rank_iqr=3.25,
        sign_consistency=0.9,
    )

    repo.fecha_multiverso("mv-1", estado="concluido", n_feitas=96, estabilidade=[medida])

    conclusao = repo.le_multiverso("mv-1")
    assert conclusao is not None
    assert conclusao.estabilidade == (medida,), (
        "a medida não voltou do disco idêntica à que entrou; algum campo se perdeu no caminho"
    )


def test_a_ordem_da_conclusao_e_fracao_decrescente_com_o_nome_a_desempatar(
    repo: MultiversoRepository,
) -> None:
    """A mesma ordem de `ordem_de_estabilidade`, com o desempate.

    O banco não promete ordem sem `ORDER BY`. Sem esta asserção, duas leituras da mesma linha
    podiam devolver tabelas em ordens diferentes — igualdade que `==` não veria.
    """
    repo.abre_multiverso(identificador="mv-1", **ABERTURA)
    # **TODAS as frações iguais — o regime medido nos dados reais.**
    #
    # A primeira versão deste teste tinha uma fração diferente (`mm` a 0,9), e por isso media
    # apenas a metade `frac_significant.desc()`. Medido: com o desempate REMOVIDO do `ORDER BY`,
    # ele continuava a passar — porque `uq_stability_feature` é `(multiverse_run_id, feature_name)`
    # e o SQLite varre por esse índice, entregando as linhas já em ordem de nome. O plano confirma:
    # `SEARCH feature_stability USING INDEX sqlite_autoindex_feature_stability_1`.
    #
    # > **A ordenação acidental do índice tornava vácua exactamente a metade que garante a ordem
    # > na camada de leitura** — e é ela que carrega a ordem
    # > INTEIRA quando todas as frações são zero, que é o caso real: foram medidas zero features
    # > sobreviventes ao FDR em 24 de 24 pares nos quatro modos.
    #
    # Com as frações iguais, a única coisa que pode ordenar é o desempate. E em Postgres — que é o
    # banco do compose — nem a ordenação acidental existe.
    empatadas = [
        EstabilidadeParaGravar(feature="zz", frac_significant=0.5, n_cells=2),
        EstabilidadeParaGravar(feature="mm", frac_significant=0.5, n_cells=2),
        EstabilidadeParaGravar(feature="aa", frac_significant=0.5, n_cells=2),
    ]

    repo.fecha_multiverso("mv-1", estado="concluido", n_feitas=96, estabilidade=empatadas)

    conclusao = repo.le_multiverso("mv-1")
    assert conclusao is not None
    assert [m.feature for m in conclusao.estabilidade] == ["aa", "mm", "zz"], (
        "com as frações todas iguais, a ordem É o desempate por nome — e ela saiu diferente"
    )


def test_a_fracao_decrescente_vence_o_desempate(repo: MultiversoRepository) -> None:
    """A outra metade do `ORDER BY`, e esta É falsificável no SQLite.

    O par é o conteúdo: um afirma o desempate por nome, o outro que a fração manda quando ela
    difere. Removido o `ORDER BY` inteiro, **este** reprova.
    """
    repo.abre_multiverso(identificador="mv-1", **ABERTURA)
    # `zz` tem a MAIOR fração e o nome que ordenaria por último: os dois critérios discordam.
    medidas = [
        EstabilidadeParaGravar(feature="aa", frac_significant=0.1, n_cells=2),
        EstabilidadeParaGravar(feature="zz", frac_significant=0.9, n_cells=2),
    ]

    repo.fecha_multiverso("mv-1", estado="concluido", n_feitas=96, estabilidade=medidas)

    conclusao = repo.le_multiverso("mv-1")
    assert conclusao is not None
    assert [m.feature for m in conclusao.estabilidade] == ["zz", "aa"]


def test_a_leitura_pede_as_duas_chaves_de_ordenacao_ao_banco(
    repo: MultiversoRepository, motor: Engine
) -> None:
    """**O desempate por nome NÃO é observável no SQLite, e isto é medido e declarado.**

    ## Por que o teste comportamental acima não basta

    `uq_stability_feature` é `(multiverse_run_id, feature_name)`, e a leitura filtra por
    `multiverse_run_id` — o prefixo exacto daquele índice. O SQLite varre por ele e entrega as
    linhas **já em ordem de nome**, que é a mesma ordem que o desempate pede. Medido: removendo
    `feature_stability.c.feature_name` do `ORDER BY`, o teste acima **continua a passar**.

    > **Ordenação acidental do índice é indistinguível de ordenação pedida** — e é ela que carrega
    > a ordem INTEIRA no regime real, porque foram medidas zero features sobreviventes ao FDR em 24
    > de
    > 24 pares nos quatro modos: todas as frações valem zero, e só o desempate ordena.
    >
    > **Em Postgres — que é o banco do compose — a coincidência não existe**, e a ausência do
    > desempate produziria ordens diferentes entre leituras da mesma linha. É a reprodutibilidade
    > a depender de um acidente de plano.

    ## O que este teste mede, e o que ele NÃO mede

    Ele intercepta o SQL que o SQLAlchemy **de facto envia** e afirma que o `ORDER BY` nomeia as
    duas chaves. **É um proxy, e vai declarado como tal**: mede o que o banco é *pedido* a
    fazer, não o que ele devolve. O que justifica o proxy é que a propriedade **não é observável**
    neste dialeto — e um portão que só medisse o observável certificaria uma ordem que só existe
    por acidente.

    O que ele não mede: que o banco **obedeça** ao `ORDER BY`. Isso é responsabilidade do SGBD, e
    testá-lo seria testar a biblioteca.
    """
    from sqlalchemy import event

    repo.abre_multiverso(identificador="mv-1", **ABERTURA)
    repo.fecha_multiverso("mv-1", estado="concluido", n_feitas=96, estabilidade=_estabilidade(3))

    consultas: list[str] = []

    def anota(_conn: object, _cur: object, sql: str, *_resto: object) -> None:
        consultas.append(sql)

    event.listen(motor, "before_cursor_execute", anota)
    try:
        repo.le_multiverso("mv-1")
    finally:
        event.remove(motor, "before_cursor_execute", anota)

    assert consultas, "âncora: nenhuma consulta foi interceptada"
    ordenadas = [s for s in consultas if "ORDER BY" in s.upper() and "feature_stability" in s]
    assert ordenadas, (
        f"nenhuma das {len(consultas)} consultas de `le_multiverso` ordena `feature_stability`"
    )
    (sql,) = ordenadas
    depois_do_order = sql.upper().split("ORDER BY", 1)[1]
    assert "FRAC_SIGNIFICANT" in depois_do_order, f"o `ORDER BY` não pede a fração: {sql}"
    assert "FEATURE_NAME" in depois_do_order, (
        "o `ORDER BY` não pede o DESEMPATE por nome. No SQLite isso não muda o resultado — o "
        f"índice já entrega em ordem de nome —, e em Postgres muda: {sql}"
    )


@pytest.mark.parametrize("estado", ["falhou", "cancelado"])
def test_um_fecho_sem_conclusao_existe_e_tem_estabilidade_vazia(
    repo: MultiversoRepository, estado: str
) -> None:
    """**`None` é "não conheço"; execução sem conclusão devolve tabela vazia.**

    As duas pedem mensagens opostas ao usuário — *"não achei"* contra *"achei, e ela parou antes de
    produzir conclusão"* —, e colapsá-las seria um erro na célula que a interface lê para
    decidir o que desenhar.
    """
    repo.abre_multiverso(identificador="mv-1", **ABERTURA)

    repo.fecha_multiverso(
        "mv-1",
        estado=estado,
        n_feitas=7,
        celula_que_falhou="celula#7" if estado == "falhou" else None,
        erro="ValueError: coisa" if estado == "falhou" else None,
    )

    conclusao = repo.le_multiverso("mv-1")
    assert conclusao is not None, "a execução existe; `None` diria que ela não existe"
    assert conclusao.estabilidade == ()
    assert conclusao.execucao.estado == estado
    assert conclusao.execucao.n_feitas == 7, (
        "`n_feitas` é o que separa «até onde chegou» de «quanto se pediu»"
    )
    assert conclusao.execucao.n_specs == 96


def test_fechar_execucao_que_nao_existe_devolve_falso_e_nao_levanta(
    repo: MultiversoRepository,
) -> None:
    """O caminho em que a abertura não pôde gravar — banco em pé só depois do disparo.

    Levantar aqui transformaria uma execução **bem-sucedida** em falha por causa da persistência
    que a rota já declarou não ter.
    """
    assert repo.fecha_multiverso("nunca-aberta", estado="concluido", n_feitas=1) is False


def test_fechar_para_executando_e_recusado_com_o_motivo(repo: MultiversoRepository) -> None:
    """Estado não-terminal no fecho deixaria a linha indistinguível de uma órfã."""
    repo.abre_multiverso(identificador="mv-1", **ABERTURA)

    with pytest.raises(ValueError, match="não é um estado terminal"):
        repo.fecha_multiverso("mv-1", estado="executando", n_feitas=1)


def test_estado_fora_do_vocabulario_e_recusado_antes_do_banco(repo: MultiversoRepository) -> None:
    """A guarda de Python é a que dá mensagem de domínio; o `CHECK` é a que vive no servidor."""
    repo.abre_multiverso(identificador="mv-1", **ABERTURA)

    with pytest.raises(ValueError, match=r"estado.*inválido"):
        repo.fecha_multiverso("mv-1", estado="terminou_bem", n_feitas=1)


def test_o_check_do_banco_recusa_o_mesmo_valor_por_baixo_do_python(
    repo: MultiversoRepository, motor: Engine
) -> None:
    """**A metade que vive no servidor, exercitada por SQL cru.**

    Sem esta linha, a guarda de Python seria a única — e um `INSERT` por qualquer outro caminho
    escreveria o rótulo inválido, que voltaria da leitura e cairia num ramo razoável da interface.
    """
    repo.abre_multiverso(identificador="mv-1", **ABERTURA)

    with pytest.raises(Exception, match=r"(?i)check|constraint"), motor.begin() as c:
        c.execute(text("UPDATE multiverse_runs SET estado = 'terminou_bem'"))


# ---------------------------------------------------------------------------
# As órfãs — a metade que torna a abertura honesta
# ---------------------------------------------------------------------------


def test_uma_execucao_aberta_e_nunca_fechada_vira_interrompido(repo: MultiversoRepository) -> None:
    """**A resposta à objeção que a docstring de `RegistroDeProgresso` levanta.**

    Sem isto, a linha diria `executando` para sempre: o contador preso, com a agravante de
    sobreviver a reinícios. Com isto, ela diz *"o programa fechou no meio"*, que é um resultado.
    """
    repo.abre_multiverso(identificador="mv-viva", **ABERTURA)

    fechadas = repo.fecha_orfas()

    assert fechadas == ("mv-viva",), (
        "devolve os identificadores e não a contagem: quem chama tem de poder dizer QUAIS"
    )
    (m,) = repo.lista_multiversos()
    assert m.estado == "interrompido"
    assert m.terminou is True, "interrompido é um fim; lê-lo como «a correr» é o defeito de AR36"
    assert m.terminado_em is not None


def test_fechar_orfas_nao_toca_no_que_ja_terminou(repo: MultiversoRepository) -> None:
    """**A outra direção, e é ela que impede o mecanismo de reescrever desfecho real.**

    Um `UPDATE` sem a cláusula `estado == 'executando'` marcaria toda a tabela como interrompida a
    cada arranque — e a suíte ficaria verde no teste de cima, porque ele só olha para a órfã.
    """
    repo.abre_multiverso(identificador="mv-ok", **ABERTURA)
    repo.abre_multiverso(identificador="mv-mau", **{**ABERTURA, "checksum_entrada": "b" * 64})
    repo.fecha_multiverso("mv-ok", estado="concluido", n_feitas=96)
    repo.fecha_multiverso("mv-mau", estado="falhou", n_feitas=3, erro="X")

    fechadas = repo.fecha_orfas()

    assert fechadas == (), f"não havia órfã e o mecanismo fechou {fechadas}"
    estados = {m.identificador: m.estado for m in repo.lista_multiversos()}
    assert estados == {"mv-ok": "concluido", "mv-mau": "falhou"}


def test_fechar_orfas_com_a_tabela_vazia_nao_levanta(repo: MultiversoRepository) -> None:
    """O arranque de um banco novo passa por aqui em toda abertura do programa."""
    assert repo.fecha_orfas() == ()


# ---------------------------------------------------------------------------
# A listagem
# ---------------------------------------------------------------------------


def test_a_listagem_vem_da_mais_recente_para_a_mais_antiga(repo: MultiversoRepository) -> None:
    """A ordem que a tabela do histórico desenha, e ela é do repositório e não da interface."""
    base = datetime(2026, 8, 28, 12, 0, tzinfo=UTC)
    for i in range(3):
        repo.abre_multiverso(
            identificador=f"mv-{i}",
            iniciado_em=base + timedelta(minutes=i),
            **{**ABERTURA, "checksum_entrada": f"{i}" * 64},
        )

    assert [m.identificador for m in repo.lista_multiversos()] == ["mv-2", "mv-1", "mv-0"]


def test_a_listagem_respeita_o_limite(repo: MultiversoRepository) -> None:
    """Consulta sem teto é a que fica lenta muito depois de quem a escreveu ter saído."""
    for i in range(5):
        repo.abre_multiverso(
            identificador=f"mv-{i}", **{**ABERTURA, "checksum_entrada": f"{i}" * 64}
        )

    assert len(repo.lista_multiversos(limite=2)) == 2
    assert len(repo.lista_multiversos()) == 5


def test_a_listagem_conta_as_features_sem_as_carregar(repo: MultiversoRepository) -> None:
    """`n_features` vem por subconsulta — 6.035 linhas por execução no `pos_hilic`."""
    repo.abre_multiverso(identificador="com", **ABERTURA)
    repo.abre_multiverso(identificador="sem", **{**ABERTURA, "checksum_entrada": "c" * 64})
    repo.fecha_multiverso("com", estado="concluido", n_feitas=96, estabilidade=_estabilidade(7))
    repo.fecha_multiverso("sem", estado="cancelado", n_feitas=4)

    contagens = {m.identificador: m.n_features for m in repo.lista_multiversos()}

    assert contagens == {"com": 7, "sem": 0}


def test_ler_identificador_desconhecido_devolve_nulo(repo: MultiversoRepository) -> None:
    assert repo.le_multiverso("nunca-existiu") is None


# ---------------------------------------------------------------------------
# A PODA EM DISCO — a terceira pergunta que o item 58 exigia antes do código
# ---------------------------------------------------------------------------


def test_a_poda_apaga_as_terminadas_mais_antigas_alem_do_teto(repo: MultiversoRepository) -> None:
    """O teto existe, e a ordem de poda é a mais antiga primeiro.

    O item 58 escrevia: *"a poda por contagem que existe hoje (teto de 32) tem de ganhar
    equivalente em disco, ou o banco cresce sem teto"*.
    """
    base = datetime(2026, 8, 28, 12, 0, tzinfo=UTC)
    for i in range(5):
        repo.abre_multiverso(
            identificador=f"mv-{i}",
            iniciado_em=base + timedelta(minutes=i),
            limite=1000,
            **{**ABERTURA, "checksum_entrada": f"{i}" * 64},
        )
        repo.fecha_multiverso(f"mv-{i}", estado="concluido", n_feitas=96)

    esquecidas = repo.esquece_alem_de(2)

    assert set(esquecidas) == {"mv-0", "mv-1", "mv-2"}, (
        f"a poda levou {esquecidas}; devia levar as três mais ANTIGAS"
    )
    assert [m.identificador for m in repo.lista_multiversos()] == ["mv-4", "mv-3"]


def test_a_poda_nao_toca_no_que_esta_a_correr(repo: MultiversoRepository) -> None:
    """**A exclusão é requisito e não zelo, e este é o par causal do teste de cima.**

    Podar uma execução em curso apagaria a linha que `fecha_multiverso` vai atualizar minutos
    depois: ele devolveria `False`, e um trabalho que correu bem desapareceria do histórico sem
    nada acusar. Sem esta asserção, uma poda que ignorasse o estado passaria no teste anterior.
    """
    base = datetime(2026, 8, 28, 12, 0, tzinfo=UTC)
    for i in range(3):
        repo.abre_multiverso(
            identificador=f"mv-{i}",
            iniciado_em=base + timedelta(minutes=i),
            limite=1000,
            **{**ABERTURA, "checksum_entrada": f"{i}" * 64},
        )
    # A mais ANTIGA fica a correr; as duas mais novas terminam.
    repo.fecha_multiverso("mv-1", estado="concluido", n_feitas=96)
    repo.fecha_multiverso("mv-2", estado="concluido", n_feitas=96)

    esquecidas = repo.esquece_alem_de(1)

    # **A ÂNCORA vem primeiro, e sem ela a asserção de ausência é satisfeita por vacuidade.**
    # Medido: com `esquece_alem_de` reduzida a `return ()`, a asserção de baixo passava.
    assert esquecidas == ("mv-1",), (
        f"a poda devia levar exactamente a terminada excedente, e levou {esquecidas}"
    )
    assert "mv-0" not in esquecidas, (
        "a poda levou a execução em curso, que é a mais antiga: um teto que ordena por data e "
        "ignora o estado apaga exactamente o trabalho que ainda está a ser feito"
    )
    assert repo.fecha_multiverso("mv-0", estado="concluido", n_feitas=96) is True, (
        "a linha da execução em curso sumiu: o fecho já não a encontra"
    )


def test_a_poda_apaga_tambem_as_linhas_de_estabilidade(
    repo: MultiversoRepository, motor: Engine
) -> None:
    """**Sem esta metade, a poda pareceria funcionar e o banco continuaria a crescer.**

    A chave estrangeira declara `ondelete="CASCADE"` — e em SQLite ela **não é aplicada** sem
    `PRAGMA foreign_keys=ON`, que este projeto não liga em lado nenhum. A contagem de execuções, que
    é o que se mede, ficaria certa; as conclusões ficariam órfãs no arquivo. Uma conclusão do
    `neg_c18` são 2.136 linhas.
    """
    repo.abre_multiverso(identificador="mv-velha", limite=1000, **ABERTURA)
    repo.fecha_multiverso(
        "mv-velha", estado="concluido", n_feitas=96, estabilidade=_estabilidade(4)
    )

    def linhas_de_estabilidade() -> int:
        with motor.connect() as c:
            return int(c.execute(text("SELECT count(*) FROM feature_stability")).scalar_one())

    assert linhas_de_estabilidade() == 4, "âncora: a conclusão não foi gravada"

    repo.esquece_alem_de(0)

    assert repo.lista_multiversos() == ()
    assert linhas_de_estabilidade() == 0, (
        "a execução sumiu e a conclusão dela ficou: a poda mede uma coisa e apaga outra"
    )


def test_a_abertura_poda_sozinha_sem_ninguem_a_chamar(repo: MultiversoRepository) -> None:
    """A poda corre na abertura, que é a única altura em que o histórico cresce.

    **Um mecanismo que só corre nesse instante não tem de existir em lado nenhum** — é o molde de
    `esquece_terminados_alem_de` no registro em memória.
    """
    base = datetime(2026, 8, 28, 12, 0, tzinfo=UTC)
    for i in range(3):
        repo.abre_multiverso(
            identificador=f"mv-{i}",
            iniciado_em=base + timedelta(minutes=i),
            limite=2,
            **{**ABERTURA, "checksum_entrada": f"{i}" * 64},
        )
        repo.fecha_multiverso(f"mv-{i}", estado="concluido", n_feitas=96)

    # **A âncora é IGUALDADE e não `<=`, e apertá-la mediu a semântica do teto.**
    #
    # A primeira versão dizia `<= 3` depois de exactamente três aberturas — verdadeiro para
    # qualquer resultado, inclusive para uma poda que nunca corresse. Apertada para `== 2` ela
    # **reprovou**, e o certo era o código:
    #
    # > **O teto conta as TERMINADAS, e a que acabou de abrir está `executando`.** Na k-ésima
    # > abertura há `k-1` terminadas, então com `limite=2` a terceira abertura não tem o que podar.
    # > O estado estacionário é `limite + 1` linhas: o teto de terminadas, mais a que corre.
    #
    # É a exclusão que `esquece_alem_de` declara — podar a que está a correr apagaria a linha que
    # o fecho vai actualizar.
    assert len(repo.lista_multiversos()) == 3, (
        f"três aberturas e `limite=2`: a terceira não tem o que podar, porque só duas terminaram. "
        f"{[m.identificador for m in repo.lista_multiversos()]}"
    )
    repo.abre_multiverso(
        identificador="mv-nova", limite=2, **{**ABERTURA, "checksum_entrada": "9" * 64}
    )

    vivos = [m.identificador for m in repo.lista_multiversos()]
    assert "mv-nova" in vivos
    assert "mv-0" not in vivos, f"a abertura não podou nada: {vivos}"
