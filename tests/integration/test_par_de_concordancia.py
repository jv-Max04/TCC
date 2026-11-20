"""O PAR de concordância, lido do banco.

**A asserção é sobre o que ficou GRAVADO, não sobre o retorno de `grava_execucao`.** É a
distinção que o critério afirma: o que se promete é que a concordância entre a máquina e o
especialista é **reconstruível por consulta**, e uma asserção sobre o valor que a função
acabou de compor provaria que a composição é consistente consigo mesma e nada sobre o banco.
O instrumento tem de medir onde se pensa que mede, e é por
isso que todo teste daqui passa por `classifica_concordancia`, que lê `decisions`.

O banco é um arquivo SQLite migrado pelo alembic, o mesmo motor que a aplicação usa. O
`CHECK` de `apresentacao` é a metade da guarda que vive no banco, e é exercitado aqui.

## O que A11 consertou, e por que o estado do meio é o que prova o desenho

`grava_execucao` escrevia **uma** linha em `decisions` por execução. Gravar a escolha do
humano **apagava a proposta da máquina** — e concordância é relação entre **dois** valores.
O insumo que o "Requisito de interação" da especificação promete era inconstruível por desenho.

O reparo não exigiu migração de chave: `decisions` **não** tem unicidade em
`(matrix_id, diagnosis_id)`, então o par cabe.

**E o gatilho da segunda linha é o controle ter sido EXERCIDO, não a spec diferir**.
Sob "quando difere", `sobrescreveu_concordando` e `nao_sobrescreveu` produziam uma linha só —
e são respectivamente **concordância medida** e **ausência de sinal**.

## Os QUATRO desfechos, e o quarto é o que um desenho de três apagaria

| Estado | Linhas | O que significa |
|---|---|---|
| `nao_sobrescreveu` | uma, `origem` da política | aceitou sem tocar no controle |
| `sobrescreveu_concordando` | duas, `spec_json` **igual** | abriu, avaliou, e manteve |
| `sobrescreveu_discordando` | duas, `spec_json` diferente | abriu, avaliou, e mudou |
| `sem_proposta` | uma, `origem="manual"` | a máquina nunca propôs — uso legítimo da CLI |

*«Uso legítimo da CLI»* não descrevia caminho nenhum — o único chamador de
`grava_execucao` é `POST /execucoes` —, e a rota passou a alcançar o quarto estado com o campo
`sem_proposta`, que a janela e a tela web mandam quando nenhuma sugestão da máquina foi seguida.

Mais **três** ramos que levantam. Sete desfechos, e a regra diz que *"dois"* é o piso e não o
número: mecanismo que classifica em N desfechos tem N direções, e a que fica inerte é sempre
a mesma. Os sete estão exercitados abaixo, começando pelo `else`.
"""

import json
from collections.abc import Iterator

import pytest
from sqlalchemy import Engine, text

from refino.domain.diagnosis import DefectProfile
from refino.persistence.repositories import (
    ConcordanciaIndeterminadaError,
    RunRepository,
    SobrescritaExercida,
)
from tests.integration.test_migrations import _alembic, engine  # noqa: F401 - fixture reexportada

pytestmark = pytest.mark.integration

PERFIL = DefectProfile(
    defect_type="erratic",
    confidence=0.61,
    suggested_batches=(0, 0, 1),
    changepoints=(45,),
    qc_rsd_median=0.3,
    qc_rsd_p75=0.4,
    zero_fraction=0.158,
    missing_fraction=0.0,
    pc1_injection_correlation=0.29,
    qc_dispersion_ratio=0.93,
    n_features=6035,
    n_samples=68,
    n_qcs=7,
)

# A proposta da máquina e a escolha do humano, como texto. Duas constantes e não uma com
# substituição, porque a igualdade e a diferença de `spec_json` são o que separa os dois
# estados de duas linhas — e construir uma da outra deixaria a distinção depender de uma
# operação de texto em vez de ser dado do teste.
SPEC_PROPOSTA = '{"correction": "serrf", "n_batches": 1}'
SPEC_ESCOLHIDA = '{"correction": "loess", "n_batches": 1}'

RAZAO_MAQUINA = (
    "Recomendado `serrf` por política constante, com abstenção de 0,00. A margem do oráculo "
    "NÃO foi medida para esta matriz."
)


@pytest.fixture
def esquema(engine: Engine) -> Iterator[Engine]:  # noqa: F811 - a fixture vem do outro módulo
    """Esquema em `head` na entrada, **e em `base` na saída aconteça o que acontecer**.

    **A limpeza é teardown de fixture e não statement final, e isso é conserto de defeito
    real.** A primeira versão destes testes fazia `_alembic("downgrade", "base")` na última
    linha de cada função — e a última linha de uma função que **reprova** nunca executa. Uma
    reprovação deixava as nove tabelas com dado dentro, e os testes seguintes mediam um banco
    sujo: dois falharam por `uq_studies_nome` duplicado e por contar 27 linhas onde esperavam
    zero, **nenhum dos dois pelo motivo que afirmavam**.

    Reprovar pelo motivo errado, na superfície do arnês: o
    teste de limpeza acusava, e o que ele acusava não era o defeito que ele descreve.

    E a descida confere o **código de saída**: `downgrade` que falha em silêncio deixaria o
    banco em `head` e o próximo `upgrade` seria no-op, com tudo passando sobre estado herdado.
    """
    assert _alembic(engine, "upgrade", "head").returncode == 0
    try:
        yield engine
    finally:
        assert _alembic(engine, "downgrade", "base").returncode == 0, (
            "o `downgrade` reprovou. O banco fica em `head` e o próximo `upgrade` é no-op — "
            "os testes seguintes medem estado herdado, e é como o `[TO-10.5]` descreve a perda "
            "silenciosa"
        )


@pytest.fixture
def repo(esquema: Engine) -> RunRepository:
    return RunRepository(esquema)


def _grava(
    repo: RunRepository,
    *,
    checksum: str = "a" * 64,
    **kw: object,
) -> object:
    """Uma execução completa. `checksum` varia para escapar de `uq_matrices_origem`."""
    base: dict[str, object] = {
        "estudo": "ATLL/HAM-TSP",
        "mode": "pos_hilic",
        "checksum_entrada": checksum,
        "n_samples": 68,
        "n_features": 6035,
        "amostras": [{"name": "QC001", "injection_order": 1, "sample_type": "qc", "grupo": "QC"}],
        "perfil": PERFIL,
        "spec_json": SPEC_PROPOSTA,
        "seed": 42,
        "rationale": RAZAO_MAQUINA,
        "origem": "modelo",
        "metrics_json": '{"qc_rsd_median": 0.3}',
        "apresentacao": "sem_apresentacao",
    }
    base.update(kw)
    return repo.grava_execucao(**base)  # type: ignore[arg-type]


def _linhas(repo: RunRepository, matrix_id: int) -> list[dict[str, object]]:
    """`decisions` daquela matriz, LIDAS DO BANCO, na ordem de gravação."""
    with repo._engine.connect() as c:
        return [
            dict(r._mapping)
            for r in c.execute(
                text(
                    "SELECT id, origem, spec_json, apresentacao, campos_sobrescritos, rationale "
                    "FROM decisions WHERE matrix_id = :m ORDER BY id"
                ),
                {"m": matrix_id},
            )
        ]


def _sobrescreve(
    repo: RunRepository,
    *,
    spec_escolhida: str,
    campos: tuple[str, ...],
    checksum: str,
    **kw: object,
) -> object:
    """Uma execução em que o controle FOI exercido."""
    return _grava(
        repo,
        checksum=checksum,
        spec_json=spec_escolhida,
        origem="manual",
        rationale=f"O pesquisador sobrescreveu {list(campos)}.",
        sobrescrita=SobrescritaExercida(
            spec_proposta_json=SPEC_PROPOSTA,
            rationale_proposta=RAZAO_MAQUINA,
            origem_proposta="modelo",
            campos=campos,
        ),
        **kw,
    )


# ---------------------------------------------------------------------------
# Os quatro estados, um caso cada, TODOS lidos do banco
# ---------------------------------------------------------------------------


def test_to_12_3_nao_sobrescreveu_grava_uma_linha(repo: RunRepository) -> None:
    """**A direção que impede o par de ser gravado sempre.**

    Sem ela, "concordância" contaria como discordância todo caso em que o humano concordou —
    porque um par gravado incondicionalmente tornaria a existência da segunda linha vácua.
    """
    g = _grava(repo)

    linhas = _linhas(repo, g.matrix_id)  # type: ignore[attr-defined]
    assert len(linhas) == 1, f"aceitar sem tocar no controle gravou {len(linhas)} linhas"
    assert linhas[0]["origem"] == "modelo"
    assert linhas[0]["campos_sobrescritos"] == "[]", "a máquina não exerceu controle nenhum"
    assert g.decision_proposta_id is None  # type: ignore[attr-defined]

    assert (
        repo.classifica_concordancia(g.matrix_id, g.diagnosis_id)  # type: ignore[attr-defined]
        == "nao_sobrescreveu"
    )


def test_to_12_3_sobrescreveu_concordando_grava_duas_com_spec_igual(repo: RunRepository) -> None:
    """**O teste que prova o desenho, porque era ele que o desenho antigo apagava.**

    O humano abriu o controle, avaliou, e manteve o valor. Sob a regra "escreve a segunda
    linha quando a spec difere", este caso produzia **uma** linha — indistinguível de nunca
    ter aberto o controle. Um é concordância medida; o outro é ausência de sinal.
    """
    g = _sobrescreve(repo, spec_escolhida=SPEC_PROPOSTA, campos=("correction",), checksum="b" * 64)

    linhas = _linhas(repo, g.matrix_id)  # type: ignore[attr-defined]
    assert len(linhas) == 2, (
        f"o controle foi exercido e o banco tem {len(linhas)} linha(s). O gatilho é o controle "
        "ter sido exercido, NÃO a spec diferir (KTD-2b)"
    )
    assert linhas[0]["spec_json"] == linhas[1]["spec_json"], "este caso é o de spec IGUAL"
    assert [linha["origem"] for linha in linhas] == ["modelo", "manual"]

    assert (
        repo.classifica_concordancia(g.matrix_id, g.diagnosis_id)  # type: ignore[attr-defined]
        == "sobrescreveu_concordando"
    )


def test_to_12_3_sobrescreveu_discordando_grava_duas_com_spec_diferente(
    repo: RunRepository,
) -> None:
    """Abriu, avaliou, e mudou. As duas linhas, e a proposta da máquina PRESERVADA."""
    g = _sobrescreve(repo, spec_escolhida=SPEC_ESCOLHIDA, campos=("correction",), checksum="c" * 64)

    linhas = _linhas(repo, g.matrix_id)  # type: ignore[attr-defined]
    assert len(linhas) == 2
    assert linhas[0]["spec_json"] == SPEC_PROPOSTA, (
        "a proposta da máquina foi APAGADA. Era o defeito de A11: gravar a escolha do humano "
        "sobrescrevia a linha única, e concordância é relação entre dois valores"
    )
    assert linhas[1]["spec_json"] == SPEC_ESCOLHIDA

    assert (
        repo.classifica_concordancia(g.matrix_id, g.diagnosis_id)  # type: ignore[attr-defined]
        == "sobrescreveu_discordando"
    )


def test_to_12_3_sem_proposta_e_o_quarto_estado_e_nao_e_nao_sobrescreveu(
    repo: RunRepository,
) -> None:
    """**O quarto desfecho, e um desenho de três o apagaria**.

    Pipeline executado com parâmetros inteiramente explícitos, sem máquina nenhuma — uso
    legítimo da CLI. Colapsá-lo em `nao_sobrescreveu` contaria execução puramente manual como
    aceitação tácita de uma proposta que **nunca existiu**.
    """
    g = _grava(repo, checksum="d" * 64, origem="manual", rationale="parâmetros explícitos")

    assert len(_linhas(repo, g.matrix_id)) == 1  # type: ignore[attr-defined]
    assert (
        repo.classifica_concordancia(g.matrix_id, g.diagnosis_id)  # type: ignore[attr-defined]
        == "sem_proposta"
    )


def test_os_quatro_estados_sao_distinguiveis_por_consulta(repo: RunRepository) -> None:
    """A asserção que o critério pede: os quatro são separáveis lendo o banco.

    Quatro execuções numa suíte só, e os quatro rótulos distintos. Testes separados provam
    cada estado; este prova que nenhum par deles colide — que é o que "distinguível"
    significa, e o que um desenho colapsado reprovaria aqui e não nos anteriores.
    """
    gravadas = {
        "nao_sobrescreveu": _grava(repo, checksum="e" * 64),
        "sem_proposta": _grava(repo, checksum="f" * 64, origem="manual", rationale="explícito"),
        "sobrescreveu_concordando": _sobrescreve(
            repo, spec_escolhida=SPEC_PROPOSTA, campos=("scaling",), checksum="0" * 64
        ),
        "sobrescreveu_discordando": _sobrescreve(
            repo, spec_escolhida=SPEC_ESCOLHIDA, campos=("scaling",), checksum="1" * 64
        ),
    }

    medidos = {
        esperado: repo.classifica_concordancia(g.matrix_id, g.diagnosis_id)  # type: ignore[attr-defined]
        for esperado, g in gravadas.items()
    }

    assert medidos == dict.fromkeys(gravadas, None) | medidos  # forma, para o diagnóstico
    assert all(esperado == medido for esperado, medido in medidos.items()), medidos
    assert len(set(medidos.values())) == 4, f"dois estados colidiram: {medidos}"


# ---------------------------------------------------------------------------
# O eixo, e não só o fato de algo ter mudado
# ---------------------------------------------------------------------------


def test_o_par_guarda_qual_eixo_mudou_e_nao_so_que_algo_mudou(repo: RunRepository) -> None:
    """Sobrescrita de UM parâmetro é registrada como tal.

    **É o sinal mais informativo que o conjunto pode ter:** quem discorda de uma escolha
    específica diz mais que quem aceita tudo ou refaz tudo, e a discordância localizada
    **nomeia o eixo**.
    """
    g = _sobrescreve(repo, spec_escolhida=SPEC_ESCOLHIDA, campos=("correction",), checksum="2" * 64)

    linhas = _linhas(repo, g.matrix_id)  # type: ignore[attr-defined]
    assert json.loads(str(linhas[1]["campos_sobrescritos"])) == ["correction"]
    assert json.loads(str(linhas[0]["campos_sobrescritos"])) == [], (
        "a linha da máquina não exerceu controle, e é o vetor vazio que a distingue da do humano"
    )


def test_o_eixo_e_recuperavel_no_estado_do_meio_onde_o_diff_e_vazio(repo: RunRepository) -> None:
    """**A razão de `campos_sobrescritos` ser coluna e não derivação.**

    Em `sobrescreveu_discordando` as chaves que diferem entre os dois `spec_json` são os
    eixos. Em `sobrescreveu_concordando` o diff é **vazio** — e sem a coluna o estado do meio
    seria detectável mas **anônimo**, quando é justamente ele que carrega concordância medida.
    """
    g = _sobrescreve(repo, spec_escolhida=SPEC_PROPOSTA, campos=("imputation",), checksum="3" * 64)

    linhas = _linhas(repo, g.matrix_id)  # type: ignore[attr-defined]
    assert linhas[0]["spec_json"] == linhas[1]["spec_json"], "o diff é vazio neste estado"
    assert json.loads(str(linhas[1]["campos_sobrescritos"])) == ["imputation"], (
        "o eixo não é recuperável do diff aqui, então tem de estar gravado"
    )


def test_o_conjunto_de_eixos_e_gravado_ordenado(repo: RunRepository) -> None:
    """Duas interfaces que renderizem os controles em ordem diferente gravam o mesmo texto.

    Sem a ordenação, o mesmo fato produziria `campos_sobrescritos` diferentes, e a consulta de
    concordância passaria a depender da ordem de renderização.
    """
    g = _sobrescreve(
        repo,
        spec_escolhida=SPEC_ESCOLHIDA,
        campos=("scaling", "correction", "imputation"),
        checksum="4" * 64,
    )

    linhas = _linhas(repo, g.matrix_id)  # type: ignore[attr-defined]
    assert json.loads(str(linhas[1]["campos_sobrescritos"])) == [
        "correction",
        "imputation",
        "scaling",
    ]


# ---------------------------------------------------------------------------
# A apresentação da justificativa, DOIS casos
# ---------------------------------------------------------------------------


def test_to_12_5_exibida_e_nao_exibida_produzem_valores_diferentes_no_banco(
    repo: RunRepository,
) -> None:
    """Sem os dois casos o campo é decorativo.

    *"Aceitou a proposta"* cobre dois casos **opostos**: viu a justificativa e concordou, e
    nunca a abriu e seguiu. O primeiro é concordância, o segundo é indiferença, e um conjunto
    de treino que os misture aprende que aceitação tácita vale como endosso.
    """
    exibida = _grava(repo, checksum="5" * 64, apresentacao="exibida")
    nao = _grava(repo, checksum="6" * 64, apresentacao="nao_exibida")

    valor_exibida = _linhas(repo, exibida.matrix_id)[0]["apresentacao"]  # type: ignore[attr-defined]
    valor_nao = _linhas(repo, nao.matrix_id)[0]["apresentacao"]  # type: ignore[attr-defined]

    assert valor_exibida == "exibida"
    assert valor_nao == "nao_exibida"
    assert valor_exibida != valor_nao, "o campo não distingue concordância de indiferença"


def test_a_apresentacao_vale_para_as_duas_linhas_do_par(repo: RunRepository) -> None:
    """É um fato da interação, e repeti-la é o que permite filtrar o conjunto sem junção.

    `WHERE origem='manual' AND apresentacao='exibida'` é o filtro do conjunto de treino, e ele
    só existe sem junção se a linha do humano também carregar o valor.
    """
    g = _sobrescreve(
        repo,
        spec_escolhida=SPEC_ESCOLHIDA,
        campos=("correction",),
        checksum="7" * 64,
        apresentacao="exibida",
    )

    assert [linha["apresentacao"] for linha in _linhas(repo, g.matrix_id)] == [  # type: ignore[attr-defined]
        "exibida",
        "exibida",
    ]


def test_apresentacao_fora_do_vocabulario_e_recusada_na_fronteira(repo: RunRepository) -> None:
    """`validate_apresentacao` levanta antes do banco, com mensagem de domínio.

    O `CHECK` do servidor é a outra metade; esta é a que dá mensagem em vez de
    `IntegrityError` cru.
    """
    with pytest.raises(ValueError, match=r"apresentacao. invalida"):
        _grava(repo, checksum="8" * 64, apresentacao="lida")


def _uma_matriz(motor: Engine, checksum: str) -> None:
    """Um `study` e uma `matrix`, para as escritas diretas em `decisions`.

    O nome do estudo carrega o `checksum` porque `studies.nome` é único: dois testes que
    inserissem o mesmo nome reprovariam por colisão de chave e **não** pelo que afirmam — reprovar
    pelo motivo errado, e foi como duas destas funções reprovaram na primeira execução.
    """
    with motor.begin() as c:
        c.execute(
            text("INSERT INTO studies (nome, criado_em) VALUES (:n, CURRENT_TIMESTAMP)"),
            {"n": f"estudo-{checksum}"},
        )
        c.execute(
            text(
                "INSERT INTO matrices (study_id, mode, n_samples, n_features, checksum_entrada) "
                "SELECT id, 'pos_hilic', 68, 6035, :c FROM studies WHERE nome = :n"
            ),
            {"c": checksum, "n": f"estudo-{checksum}"},
        )


def test_o_check_do_banco_recusa_apresentacao_fora_do_vocabulario(esquema: Engine) -> None:
    """A metade que vive no BANCO, exercitada por escrita direta.

    Sem esta asserção, a guarda de `apresentacao` existiria só em Python — e um caminho de
    escrita que não passasse por `validate_apresentacao` a contornaria em silêncio.
    """
    from sqlalchemy.exc import IntegrityError

    _uma_matriz(esquema, "zz")
    with pytest.raises(IntegrityError), esquema.begin() as c:
        c.execute(
            text(
                "INSERT INTO decisions (matrix_id, spec_json, rationale, origem, criado_em, "
                "apresentacao, campos_sobrescritos) "
                "SELECT id, '{}', 'x', 'modelo', CURRENT_TIMESTAMP, 'lida', '[]' FROM matrices"
            )
        )


def test_o_par_causal_as_tres_apresentacoes_do_vocabulario_entram(esquema: Engine) -> None:
    """**Visto passar sem a planta**: o `CHECK` deixa passar o que deve.

    Sem esta direção, um `CHECK` que recusasse tudo passaria no teste acima — portão que
    **sempre** reprova.
    """
    _uma_matriz(esquema, "yy")
    with esquema.begin() as c:
        for valor in ("exibida", "nao_exibida", "sem_apresentacao"):
            c.execute(
                text(
                    "INSERT INTO decisions (matrix_id, spec_json, rationale, origem, criado_em, "
                    "apresentacao, campos_sobrescritos) "
                    "SELECT id, '{}', 'x', 'modelo', CURRENT_TIMESTAMP, :a, '[]' FROM matrices"
                ),
                {"a": valor},
            )
    with esquema.connect() as c:
        assert c.execute(text("SELECT count(*) FROM decisions")).scalar_one() == 3


# ---------------------------------------------------------------------------
# `SUPPORTED_OVERRIDES` é enumerado e FECHADO
# ---------------------------------------------------------------------------


def test_to_12_4_sobrescrever_campo_fora_da_lista_e_recusado_e_nada_e_gravado(
    repo: RunRepository,
) -> None:
    """Campo fora de `SUPPORTED_OVERRIDES` levanta, e a recusa não deixa linha para trás.

    **Ignorar o campo desconhecido seria o modo de falha insidioso:** gravaria par de
    concordância sobre um eixo que a enumeração não declara, e o conjunto teria linha cujo
    significado ninguém pode reconstruir.
    """
    with pytest.raises(ValueError, match="nao sobrescreviv"):
        _sobrescreve(repo, spec_escolhida=SPEC_ESCOLHIDA, campos=("seed",), checksum="9" * 64)

    with repo._engine.connect() as c:
        assert c.execute(text("SELECT count(*) FROM decisions")).scalar_one() == 0


def test_sobrescrita_com_conjunto_de_eixos_vazio_e_recusada(repo: RunRepository) -> None:
    """Exercer controle sobre nenhum eixo não é fato possível.

    Admiti-lo gravaria um par cujo eixo ninguém sabe nomear — e o par sem eixo é
    indistinguível, na consulta, de um par cuja coluna nunca foi preenchida.
    """
    with pytest.raises(ValueError, match="`campos` vazio"):
        SobrescritaExercida(
            spec_proposta_json=SPEC_PROPOSTA,
            rationale_proposta=RAZAO_MAQUINA,
            origem_proposta="modelo",
            campos=(),
        )


def test_a_proposta_da_maquina_nao_pode_ter_origem_manual(repo: RunRepository) -> None:
    """Um par com as duas linhas `manual` não tem proposta com que concordar."""
    with pytest.raises(ValueError, match="origem_proposta"):
        SobrescritaExercida(
            spec_proposta_json=SPEC_PROPOSTA,
            rationale_proposta=RAZAO_MAQUINA,
            origem_proposta="manual",
            campos=("correction",),
        )


def test_sobrescrita_exercida_com_origem_da_escolha_nao_manual_e_recusada(
    repo: RunRepository,
) -> None:
    """A linha B é a escolha do humano, e a origem dela é `manual` por definição.

    Gravar de outro modo tornaria o par **inclassificável** pela consulta: sem exactamente uma
    linha `manual` não há como saber qual é a proposta e qual é a escolha.
    """
    with pytest.raises(ValueError, match=r"sobrescrita exercida e .origem"):
        _grava(
            repo,
            checksum="A" * 64,
            origem="modelo",
            sobrescrita=SobrescritaExercida(
                spec_proposta_json=SPEC_PROPOSTA,
                rationale_proposta=RAZAO_MAQUINA,
                origem_proposta="modelo",
                campos=("correction",),
            ),
        )


# ---------------------------------------------------------------------------
# A transação, e os TRÊS ramos que levantam
# ---------------------------------------------------------------------------


def test_falha_ao_gravar_a_segunda_linha_desfaz_a_primeira(repo: RunRepository) -> None:
    """Sem isto o banco fica com proposta órfã, e o conjunto teria linha inclassificável.

    **A planta entra no BANCO e a chamada é a de produção**, e essa ordem é conserto de um
    defeito meu. A primeira versão deste teste escrevia as duas linhas à mão dentro de um
    `engine.begin()` do próprio teste e afirmava que o `rollback` as removia — o que exercita
    a semântica transacional do **banco** e não `grava_execucao`. O instrumento
    estava correto e media o objeto errado.

    A planta é um gatilho em `decisions` que recusa `origem='manual'`. O SQLite não acrescenta
    `CHECK` a tabela existente, e o gatilho com `RAISE(ABORT)` dá o mesmo erro de restrição.
    A linha A já está escrita quando a linha B reprova, então só o `rollback` **de dentro de
    `grava_execucao`** deixa a tabela vazia. Plantar na fronteira em Python não serviria:
    `validate_origem` levantaria **antes** de a transação abrir, e nada teria sido escrito.
    """
    from sqlalchemy.exc import IntegrityError

    with repo._engine.begin() as c:
        c.execute(
            text(
                "CREATE TRIGGER planta_recusa_manual BEFORE INSERT ON decisions "
                "WHEN NEW.origem = 'manual' "
                "BEGIN SELECT RAISE(ABORT, 'planta: origem manual recusada'); END"
            )
        )

    with pytest.raises(IntegrityError):
        _sobrescreve(repo, spec_escolhida=SPEC_ESCOLHIDA, campos=("correction",), checksum="D" * 64)

    with repo._engine.connect() as c:
        assert c.execute(text("SELECT count(*) FROM decisions")).scalar_one() == 0, (
            "a proposta ficou órfã: a linha B reprovou e a linha A sobreviveu. O par tem de "
            "ser atômico — proposta sem escolha é linha que ninguém sabe classificar"
        )


def test_o_par_causal_da_transacao_sem_a_planta_grava_as_duas(repo: RunRepository) -> None:
    """**Visto passar com a planta REMOVIDA**, e é negativo EXTERNO por construção.

    O teste acima constrói estado no banco e invoca o caminho de produção — segunda linha da
    taxonomia —, então ele pode ficar inerte por duas vias: a invocação não alcançar a
    planta, ou o desfecho ser o mesmo com e sem ela. Esta função é a segunda execução do par
    causal: mesma chamada, sem o gatilho, e o desfecho **divergindo**.
    """
    g = _sobrescreve(repo, spec_escolhida=SPEC_ESCOLHIDA, campos=("correction",), checksum="D" * 64)

    assert len(_linhas(repo, g.matrix_id)) == 2, (  # type: ignore[attr-defined]
        "sem a planta o par tem de ser gravado. Se aqui também der zero, o teste acima não "
        "mede o `rollback` — mede uma escrita que nunca funciona"
    )


def test_o_ramo_else_zero_linhas_levanta(repo: RunRepository) -> None:
    """**Começa pelo `else`**: ele dispara por AUSÊNCIA e não há o que plantar nele.

    Ausência de linha não é ausência de sobrescrita — é execução que não gravou. Devolvê-la
    como estado a faria entrar na contagem do conjunto.
    """
    with pytest.raises(ConcordanciaIndeterminadaError, match="nenhuma decisão gravada"):
        repo.classifica_concordancia(99999, 99999)


def test_duas_linhas_sem_exactamente_uma_manual_levanta(repo: RunRepository) -> None:
    """Duas propostas e nenhuma escolha: não há como saber qual linha é qual."""
    g = _grava(repo, checksum="B" * 64)
    with repo._engine.begin() as c:
        c.execute(
            text(
                "INSERT INTO decisions (matrix_id, diagnosis_id, spec_json, rationale, origem, "
                "criado_em, apresentacao, campos_sobrescritos) "
                "VALUES (:m, :d, '{}', 'x', 'modelo', CURRENT_TIMESTAMP, 'exibida', '[]')"
            ),
            {"m": g.matrix_id, "d": g.diagnosis_id},  # type: ignore[attr-defined]
        )

    with pytest.raises(ConcordanciaIndeterminadaError, match=r"0 com .origem"):
        repo.classifica_concordancia(g.matrix_id, g.diagnosis_id)  # type: ignore[attr-defined]


def test_mais_de_duas_linhas_levanta(repo: RunRepository) -> None:
    """O par tem duas. Três é escrita defeituosa, e não um quinto estado."""
    g = _sobrescreve(repo, spec_escolhida=SPEC_ESCOLHIDA, campos=("correction",), checksum="C" * 64)
    with repo._engine.begin() as c:
        c.execute(
            text(
                "INSERT INTO decisions (matrix_id, diagnosis_id, spec_json, rationale, origem, "
                "criado_em, apresentacao, campos_sobrescritos) "
                "VALUES (:m, :d, '{}', 'x', 'manual', CURRENT_TIMESTAMP, 'exibida', "
                "'[\"scaling\"]')"
            ),
            {"m": g.matrix_id, "d": g.diagnosis_id},  # type: ignore[attr-defined]
        )

    with pytest.raises(ConcordanciaIndeterminadaError, match="3 decisões"):
        repo.classifica_concordancia(g.matrix_id, g.diagnosis_id)  # type: ignore[attr-defined]
