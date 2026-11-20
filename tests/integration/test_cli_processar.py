"""O pipeline completo na CLI, e o log como ENTRADA.

**A ida e volta é o teste, não a inspeção de campos.** O critério diz "realimentar o log
produz saída idêntica", e a forma de verificar isso é executar, ler o log, reexecutar a
partir dele e comparar **bit a bit**. Conferir que o log tem os campos certos verificaria a
serialização; comparar as duas saídas verifica a propriedade.

**E as duas medições a registrar vivem aqui**. Os critérios 5.1 e 5.2 deixaram de ser
portão porque citavam números que a especificação não contém: eram medição real com procedência
falsa. Agora medem e registram, **sem tolerância e sem valor esperado** —
o que se verifica é que a medição aconteceu, com variante e estimador declarados ao lado.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from refino.cli import main
from refino.domain.spec import PipelineSpec
from refino.io.readers import load_matrix
from refino.preprocessing.pipeline import CANONICAL_ORDER, chain_is_consistent, process_matrix
from tests.conftest import measurements_path, sample_list_path

pytestmark = pytest.mark.integration

MODOS = ("pos_hilic", "pos_c18", "neg_hilic", "neg_c18")


def _argv(modo: str, saida: Path, log: Path, **extra: str) -> list[str]:
    argv = [
        "processar",
        "--measurements",
        str(measurements_path(modo)),
        "--sample-list",
        str(sample_list_path(modo)),
        "--mode",
        modo,
        "--output",
        str(saida),
        "--log",
        str(log),
        "--rsd-cutoff",
        "0.30",
        "--imputation",
        "one_fifth_min",
        "--sample-normalization",
        "sum",
        "--transformation",
        "log10",
        "--scaling",
        "auto",
    ]
    for chave, valor in extra.items():
        argv += [f"--{chave.replace('_', '-')}", valor]
    return argv


# ---------------------------------------------------------------------------
# A ida e volta
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("modo", MODOS)
def test_ca_5_9_o_log_realimentado_produz_saida_identica(modo: str, tmp_path: Path) -> None:
    """Esta é a propriedade inteira: o log é ENTRADA e não relatório.

    Comparação **bit a bit** dos dois arquivos de saída. Comparar valores com tolerância
    deixaria passar não-determinismo pequeno, e é justamente o não-determinismo pequeno que
    torna um log irreproduzível sem que ninguém note.
    """
    primeira = tmp_path / "primeira.csv"
    log = tmp_path / "execucao.json"
    assert main(_argv(modo, primeira, log)) == 0

    segunda = tmp_path / "segunda.csv"
    assert main(["processar", "--from-log", str(log), "--output", str(segunda)]) == 0

    assert primeira.read_bytes() == segunda.read_bytes(), (
        f"{modo}: a reexecução a partir do log deu saída diferente; o log não é entrada"
    )


def test_o_log_carrega_contagem_por_etapa_na_ordem_canonica(tmp_path: Path) -> None:
    """A contagem por etapa é o que hoje se perde, e a ORDEM é parte dela.

    **Duas asserções contra a MESMA lista escrita à mão, e a segunda é nova em 04/09/2026.** A
    primeira afirma o que o programa produz; a segunda afirma que a constante que o projeto
    exporta como ordem canônica **é** essa ordem. Até aqui `CANONICAL_ORDER` não tinha leitor
    nenhum — nem módulo, nem teste — e declarava seis etapas contra as sete que corriam, com
    `impute` de fora. **Sao OITO desde 07/09/2026**, com o filtro por posto da paridade com o
    MetaboAnalyst — e foi este pino que obrigou a lista a crescer junto com a constante.

    **O pino continua a ser a lista escrita aqui, e não a constante**, porque comparar o log
    contra ela e mais nada faria os dois deslizarem juntos: quem reordenasse a constante veria o
    teste segui-la. É a distinção entre derivar e pinar, e aqui as duas coisas são
    afirmadas contra um terceiro que não se move sozinho.
    """
    saida = tmp_path / "saida.csv"
    log = tmp_path / "log.json"
    assert main(_argv("neg_c18", saida, log)) == 0

    esperada = [
        "filter_by_qc_rsd",
        "filter_by_ranking",
        # **O segundo passe por posto — 10/09/2026.** A lista e escrita a mao DE PROPOSITO, e a
        # docstring acima diz por que: derivar da constante faria os dois deslizarem juntos.
        # Cresce de oito para nove entradas, e a constante cresce no mesmo commit.
        "filter_by_ranking_2",
        "filter_near_zero_variance",
        "drop_by_group_missingness",
        "impute",
        "normalize_samples",
        "transform",
        "scale_features",
    ]
    conteudo = json.loads(log.read_text(encoding="utf-8"))
    assert conteudo["stage_order"] == esperada
    assert list(CANONICAL_ORDER) == esperada, (
        "`CANONICAL_ORDER` é o que o projeto exporta como a ordem canônica, e não bate com a que "
        "o pipeline executa. Ela declarou SEIS etapas — sem `impute` — até 04/09/2026, sem leitor "
        "nenhum que a acusasse"
    )
    assert conteudo["chain_is_consistent"] is True
    for etapa, dados in conteudo["stages"].items():
        assert dados["n_features_out"] <= dados["n_features_in"], etapa


def test_a_cadeia_de_contagens_encadeia(tmp_path: Path) -> None:
    """Saída de cada etapa é a entrada da seguinte.

    É o que impede uma etapa de reportar sobre matriz que não é a que passou por ela — e sem
    ele a contagem por etapa poderia estar inteira e descrever outra execução.
    """
    matriz = load_matrix(measurements_path("neg_c18"), sample_list_path("neg_c18"), "neg_c18")
    spec = PipelineSpec(
        correction="none",
        n_batches=1,
        rsd_cutoff=0.30,
        ranking_filter="none",
        ranking_keep_fraction=None,
        ranking_filter_2="none",
        ranking_keep_fraction_2=None,
        rsd_variant="outlier_removed",
        missing_threshold=0.8,
        imputation="one_fifth_min",
        sample_normalization="sum",
        normalization_reference=None,
        normalization_factors=None,
        transformation="log10",
        scaling="auto",
    )
    processada, resultados = process_matrix(matriz, spec)
    assert chain_is_consistent(resultados)
    assert resultados[-1].n_features_out == processada.n_features


# ---------------------------------------------------------------------------
# As duas medições a registrar
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("modo", "corte", "retidas_esperadas", "indefinido_esperado"),
    [("pos_hilic", 0.30, 758, 234), ("pos_c18", 0.20, 550, 60)],
)
def test_ca_5_1_e_5_2_retencao_exata_com_a_variante_declarada(
    modo: str, corte: float, retidas_esperadas: int, indefinido_esperado: int
) -> None:
    """Contagem **exata**, sem tolerância, variante declarada.

    **Voltaram a ser portão porque a procedência foi estabelecida**, e o que a
    estabeleceu não foi remedição: foi **corroboração independente**. As contagens de RSD
    indefinido sob `outlier_removed` dão 234 e 60, que são exatamente os números que a especificação
    **tem** — duas quantidades convergindo para a mesma variante por caminhos diferentes.

    **Sem tolerância porque não há o que absorver.** O filtro sobre matriz sem correção é
    determinístico. O ±5 anterior não cobria a diferença de 266 contra `plain`, que é o que a
    variante errada produz — ele encobria a ausência do rótulo, não ruído.
    """
    from refino.preprocessing.filtering import ACIMA_DO_CORTE, INDEFINIDO, filter_by_qc_rsd

    matriz = load_matrix(measurements_path(modo), sample_list_path(modo), modo)
    _, _, resultado = filter_by_qc_rsd(matriz, corte, variant="outlier_removed")

    assert resultado.n_features_out == retidas_esperadas, (
        f"{modo} com corte {corte} e `outlier_removed` retém {resultado.n_features_out} e o "
        f"critério pede exatamente {retidas_esperadas}. Divergência aqui é ACHADO "
        "— a contagem é determinística e não tem tolerância."
    )
    assert resultado.discarded_by_reason[INDEFINIDO] == indefinido_esperado, (
        "a contagem de indefinido é a corroboração independente da variante, e é um número "
        "que a §4 contém"
    )
    assert resultado.labels["rsd_variant"] == "outlier_removed"
    assert resultado.labels["rsd_estimator"] == "in_sample"
    assert set(resultado.discarded_by_reason) == {ACIMA_DO_CORTE, INDEFINIDO}


@pytest.mark.parametrize(
    ("modo", "corte", "retidas_outlier_removed"),
    [("pos_hilic", 0.30, 758), ("pos_c18", 0.20, 550)],
)
def test_a_variante_e_parte_do_criterio_e_nao_detalhe(
    modo: str, corte: float, retidas_outlier_removed: int
) -> None:
    """**O par causal do rótulo no critério**, e é ele que justifica a variante ser declarada.

    Com `plain` as contagens são 492 e 481. Um critério que não declarasse a variante
    passaria ou reprovaria conforme quem medisse — que é exatamente o defeito que a versão
    anterior tinha, e o que a tolerância de ±5 encobria sem cobrir.
    """
    from refino.preprocessing.filtering import filter_by_qc_rsd

    matriz = load_matrix(measurements_path(modo), sample_list_path(modo), modo)
    _, _, com_outlier_removed = filter_by_qc_rsd(matriz, corte, variant="outlier_removed")
    _, _, com_plain = filter_by_qc_rsd(matriz, corte, variant="plain")

    assert com_outlier_removed.n_features_out == retidas_outlier_removed
    assert com_plain.n_features_out != com_outlier_removed.n_features_out, (
        "as duas variantes deram a mesma contagem; o rótulo do critério seria decoração"
    )
    # E as duas medições ficam registradas, uma célula por arquivo.
    for resultado, variante in ((com_outlier_removed, "outlier_removed"), (com_plain, "plain")):
        destino = Path("out") / f"ca_5_1_5_2_retencao_{modo}_{corte:g}_{variante}.json"
        destino.parent.mkdir(parents=True, exist_ok=True)
        destino.write_text(
            json.dumps(
                {
                    "modo": modo,
                    "corte": corte,
                    "variante": variante,
                    "estimador": "in_sample",
                    "n_features_in": resultado.n_features_in,
                    "retidas": resultado.n_features_out,
                    "descartadas": dict(resultado.discarded_by_reason),
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )


def test_a_politica_de_rsd_indefinido_e_visivel_na_medicao() -> None:
    """As duas contagens separadas, sobre dado real, e o indefinido é o que a especificação nomeia.

    Nos arquivos sem correção o indefinido atinge 234 features no `pos_hilic` e 60 no
    `pos_c18` — números que a especificação **tem** —, e o teste afirma que a contagem é
    estritamente positiva em vez de fixar o valor: fixar o transformaria em portão sobre número cuja
    variante o critério não declarava.
    """
    from refino.preprocessing.filtering import INDEFINIDO, filter_by_qc_rsd

    matriz = load_matrix(measurements_path("pos_hilic"), sample_list_path("pos_hilic"), "pos_hilic")
    _, _, resultado = filter_by_qc_rsd(matriz, 0.30, variant="outlier_removed")
    assert resultado.discarded_by_reason[INDEFINIDO] > 0


# ---------------------------------------------------------------------------
# A fronteira dupla: CLI e log
# ---------------------------------------------------------------------------


def test_rotulo_invalido_e_recusado_antes_de_carregar_a_matriz(tmp_path: Path) -> None:
    """Recusar depois desperdiça a leitura de alguns MB para rejeitar o que já era inválido."""
    saida = tmp_path / "saida.csv"
    log = tmp_path / "log.json"
    argv = _argv("neg_c18", saida, log)
    argv[argv.index("--transformation") + 1] = "log"  # próximo de `log10`, o erro realista
    assert main(argv) == 2
    assert not saida.exists(), "a matriz não devia ter sido escrita"


def test_a_combinacao_proibida_e_recusada_pela_cli(tmp_path: Path) -> None:
    """`imputation="none"` com normalização, recusado na construção da especificação.

    Propagar ausência produziria matriz que **parece** processada e não é, e o multiverso teria
    especificações comparando relatórios sobre matrizes parcialmente indefinidas.
    """
    saida = tmp_path / "saida.csv"
    log = tmp_path / "log.json"
    argv = _argv("neg_c18", saida, log)
    argv[argv.index("--imputation") + 1] = "none"
    assert main(argv) == 2
    assert not saida.exists()


def test_none_na_imputacao_com_tudo_none_e_permitido(tmp_path: Path) -> None:
    """O caminho que deixa a matriz como está — e é a referência que o multiverso usa."""
    saida = tmp_path / "saida.csv"
    log = tmp_path / "log.json"
    argv = _argv("neg_c18", saida, log)
    for flag, valor in (
        ("--imputation", "none"),
        ("--sample-normalization", "none"),
        ("--transformation", "none"),
        ("--scaling", "none"),
    ):
        argv[argv.index(flag) + 1] = valor
    assert main(argv) == 0
    assert saida.exists()


def test_log_com_campo_ausente_e_recusado(tmp_path: Path) -> None:
    """**A segunda fronteira**, e a que se esquece: o log é dado vindo de disco.

    Preencher com padrão faria a reexecução divergir da original com o log parecendo
    íntegro — o modo de falha que não levanta.
    """
    saida = tmp_path / "primeira.csv"
    log = tmp_path / "log.json"
    assert main(_argv("neg_c18", saida, log)) == 0

    conteudo = json.loads(log.read_text(encoding="utf-8"))
    del conteudo["spec"]["scaling"]
    mutilado = tmp_path / "mutilado.json"
    mutilado.write_text(json.dumps(conteudo), encoding="utf-8")

    assert (
        main(["processar", "--from-log", str(mutilado), "--output", str(tmp_path / "x.csv")]) == 2
    )


def test_log_com_rotulo_invalido_e_recusado(tmp_path: Path) -> None:
    """Rótulo editado à mão no log passa pela mesma guarda que o da linha de comando."""
    saida = tmp_path / "primeira.csv"
    log = tmp_path / "log.json"
    assert main(_argv("neg_c18", saida, log)) == 0

    conteudo = json.loads(log.read_text(encoding="utf-8"))
    conteudo["spec"]["scaling"] = "minmax"
    mutilado = tmp_path / "mutilado.json"
    mutilado.write_text(json.dumps(conteudo), encoding="utf-8")

    assert (
        main(["processar", "--from-log", str(mutilado), "--output", str(tmp_path / "x.csv")]) == 2
    )


def test_argumento_obrigatorio_ausente_e_recusado(tmp_path: Path) -> None:
    """Sem `--from-log`, os quatro obrigatórios são exigidos com a lista COMPLETA."""
    assert main(["processar", "--mode", "neg_c18"]) == 2


def test_a_matriz_processada_nao_tem_nao_finito(tmp_path: Path) -> None:
    """De ponta a ponta: a cascata inteira sobre dado real não produz não-finito."""
    matriz = load_matrix(measurements_path("neg_c18"), sample_list_path("neg_c18"), "neg_c18")
    spec = PipelineSpec(
        correction="none",
        n_batches=1,
        rsd_cutoff=0.30,
        ranking_filter="none",
        ranking_keep_fraction=None,
        ranking_filter_2="none",
        ranking_keep_fraction_2=None,
        rsd_variant="outlier_removed",
        missing_threshold=0.8,
        imputation="one_fifth_min",
        sample_normalization="sum",
        normalization_reference=None,
        normalization_factors=None,
        transformation="log10",
        scaling="auto",
    )
    processada, _ = process_matrix(matriz, spec)
    assert np.isfinite(np.asarray(processada.values)).all()


def test_o_filtro_por_posto_alcanca_o_pipeline_e_nao_e_ignorado(tmp_path: Path) -> None:
    """O par causal de ponta a ponta da etapa nova, e ele fecha um defeito concreto.

    **Entre o `Literal` existir e a etapa correr há uma janela em que o eixo é aceito e
    silenciosamente ignorado** — o usuário pede `--ranking-filter iqr`, a guarda aprova, e
    nada filtra. É a mesma classe do `else` cego que `impute` tinha: a política pedida não é
    a que corre, e o desfecho é indistinguível de sucesso.

    As duas direções, sobre um modo real: com `none` a etapa reporta entrada igual a saída;
    com `iqr` ela descarta. Sem a segunda metade, a primeira passaria com a etapa ausente.
    """
    contagens = {}
    for filtro in ("none", "iqr"):
        saida = tmp_path / f"{filtro}.csv"
        log = tmp_path / f"{filtro}.json"
        assert main(_argv("neg_c18", saida, log, ranking_filter=filtro)) == 0
        etapa = json.loads(log.read_text(encoding="utf-8"))["stages"]["filter_by_ranking"]
        contagens[filtro] = (etapa["n_features_in"], etapa["n_features_out"])

    entrada_none, saida_none = contagens["none"]
    entrada_iqr, saida_iqr = contagens["iqr"]

    assert saida_none == entrada_none, (
        f"com `none` a etapa descartou {entrada_none - saida_none} features; ela tem de ser "
        "saltada, senão quem não escolher muda de resultado"
    )
    assert saida_iqr < entrada_iqr, (
        "com `iqr` a etapa não descartou nada: o eixo chega ao pipeline e não age, que é o "
        "defeito de rótulo aceito e ignorado"
    )
    assert entrada_none == entrada_iqr, (
        "as duas execuções entraram na etapa com contagens diferentes: a comparação não é "
        "sobre a mesma população e não separa nada"
    )


def test_a_ida_e_volta_corre_com_o_valor_que_EXERCITA_a_imputacao(  # noqa: N802
    tmp_path: Path,
) -> None:
    """O log realimentado com um valor que muda MAQUINARIA, e não apenas um número.

    A ida e volta acima corre com `one_fifth_min`, que substitui ausência por uma constante
    derivada da coluna. **`knn` percorre outro caminho inteiro**: distância entre amostras,
    vizinhança, média ponderada. Um registro que reproduza a primeira não prova nada sobre a
    segunda, e é a segunda que tem estado interno para não reproduzir.

    O eixo é a imputação de propósito: é o único a que a imputação acrescenta valor e que ela
    **não** altera, então este teste e o da imputação não se cruzam.
    """
    primeira = tmp_path / "primeira.csv"
    log = tmp_path / "knn.json"
    assert main(_argv("neg_c18", primeira, log, imputation="knn")) == 0

    registro = json.loads(log.read_text(encoding="utf-8"))
    assert registro["spec"]["imputation"] == "knn", (
        "o registro não gravou a imputação pedida; a reexecução abaixo cairia noutro método "
        "e a comparação bit a bit não separaria nada"
    )

    segunda = tmp_path / "segunda.csv"
    assert main(["processar", "--from-log", str(log), "--output", str(segunda)]) == 0
    assert primeira.read_bytes() == segunda.read_bytes(), (
        "a reexecução com `knn` deu saída diferente: o registro não carrega o que aquele "
        "caminho precisa para se repetir"
    )


def test_par_causal_a_chave_trocada_na_escrita_QUEBRA_a_reexecucao(  # noqa: N802
    tmp_path: Path,
) -> None:
    """A planta que prova que o teste acima não passa por coincidência.

    **A planta cai sobre dado em que as duas coisas diferem**: `knn` e o método
    plantado têm de produzir matrizes distintas, e isso é afirmado ANTES de a planta ser
    lida — senão a reexecução bateria bit a bit com a chave trocada, e o teste de cima
    estaria a passar por vacuidade.
    """
    com_knn = tmp_path / "knn.csv"
    log = tmp_path / "knn.json"
    assert main(_argv("neg_c18", com_knn, log, imputation="knn")) == 0

    com_outro = tmp_path / "outro.csv"
    assert main(_argv("neg_c18", com_outro, tmp_path / "outro.json", imputation="median")) == 0
    assert com_knn.read_bytes() != com_outro.read_bytes(), (
        "`knn` e `median` deram a mesma matriz neste modo; a planta abaixo não separaria as "
        "duas coisas e o par causal seria decorativo"
    )

    mutilado = json.loads(log.read_text(encoding="utf-8"))
    mutilado["spec"]["imputation"] = "median"
    caminho = tmp_path / "mutilado.json"
    caminho.write_text(json.dumps(mutilado), encoding="utf-8")

    reexecutada = tmp_path / "reexecutada.csv"
    assert main(["processar", "--from-log", str(caminho), "--output", str(reexecutada)]) == 0
    assert reexecutada.read_bytes() != com_knn.read_bytes(), (
        "com a imputação trocada no registro a reexecução deu a MESMA saída: a leitura não "
        "está a usar a chave, e a ida e volta acima é verde por não exercitar nada"
    )


# ---------------------------------------------------------------------------
# O critério em CADA eixo novo
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("rotulo", "eixos"),
    [
        (
            "filtro em dois eixos",
            {
                "ranking_filter": "iqr",
                "ranking_keep_fraction": "0.60",
                "ranking_filter_2": "mean",
                "ranking_keep_fraction_2": "0.50",
            },
        ),
        (
            "feature de referência",
            {
                "sample_normalization": "feature_reference",
                "normalization_reference": "__PRIMEIRA__",
            },
        ),
    ],
)
def test_a_reexecucao_bate_com_valor_NAO_PADRAO_em_cada_eixo_novo(  # noqa: N802
    tmp_path: Path, rotulo: str, eixos: dict[str, str]
) -> None:
    """O critério em cada eixo novo, e o portão do registro **não** o cobre.

    ## O que faltava, e a distinção é fina

    `tests/unit/test_cli_log.py` prova que cada campo do contrato **viaja**: trocar o valor no
    registro troca o campo reconstruído. Isso é transporte.

    O critério afirma outra coisa — *realimentar o log produz saída idêntica* —, e pede-a
    **com valor não padrão em cada eixo novo**. A comparação byte a byte corria com `imputation`
    e com `supplied_factors`. Os dois eixos do filtro por posto e os dois métodos de referência
    nunca a tinham exercitado.

    **O valor decide quanto do caminho corre, aqui na cláusula de aceitação e não no arnês:** o
    valor natural — `none` nos dois filtros, `None` na referência — é o que executa menos do
    caminho, e a ida e volta ficava verde sobre um salto.

    ## A âncora de não-vacuidade vem ANTES, e é o que separa isto de tautologia

    Duas execuções com o mesmo eixo em `none` também bateriam byte a byte. O teste afirma
    primeiro que o eixo **muda a saída** — comparando com a execução sem ele — e só então que a
    reexecução a reproduz. Sem essa metade, ele passaria sobre um pipeline que ignorasse os
    campos por inteiro.
    """
    modo = "neg_c18"

    # A referência tem de existir na matriz, e o nome sai da matriz em vez de ser inventado.
    concreto = dict(eixos)
    if concreto.get("normalization_reference") == "__PRIMEIRA__":
        matriz = load_matrix(measurements_path(modo), sample_list_path(modo), mode=modo)
        concreto["normalization_reference"] = matriz.feature_names[0]

    sem_eixo = tmp_path / "sem.csv"
    assert main(_argv(modo, sem_eixo, tmp_path / "sem.json")) == 0

    primeira = tmp_path / "primeira.csv"
    log = tmp_path / "com.json"
    assert main(_argv(modo, primeira, log, **concreto)) == 0

    assert primeira.read_bytes() != sem_eixo.read_bytes(), (
        f"o eixo «{rotulo}» com valor não padrão deu a MESMA saída que sem ele. A reexecução "
        "abaixo bateria por o pipeline ignorar os campos, e este teste seria decorativo"
    )

    segunda = tmp_path / "segunda.csv"
    assert main(["processar", "--from-log", str(log), "--output", str(segunda)]) == 0
    assert primeira.read_bytes() == segunda.read_bytes(), (
        f"a reexecução a partir do registro divergiu no eixo «{rotulo}»: o log não carrega o "
        "que aquele caminho precisa para se repetir"
    )


# ---------------------------------------------------------------------------
# O critério 5, sobre a POPULAÇÃO que ele nomeia
# ---------------------------------------------------------------------------


def test_criterio_5_de_d410_o_encadeamento_sobre_o_NEG_C18() -> None:  # noqa: N802
    """A âncora pré-registrada, na população que o pré-registro nomeia.

    ## Por que ela existe ao lado da irmã sintética

    `tests/unit/test_filtering.py::test_a_ancora_do_encadeamento_separa_as_TRES_causas` mede a
    mesma propriedade e é a melhor construída das duas: ela separa três causas antes de
    concluir, sobre uma fixture desenhada para pôr variância e abundância em oposição.

    **O que ela não faz é medir sobre o `neg_c18`**, e o critério 5 nomeia-o:

    > *"Sobre o `neg_c18`: IQR a 40% depois de abundância a 0% retém exactamente o mesmo
    > conjunto que IQR a 40% sozinho; com abundância a 10% o conjunto é estritamente contido
    > no anterior"*

    Trocar a população declarada por outra **sem o declarar** é rótulo certo sobre a
    quantidade errada, na forma de escopo —, e a regra 1 do README diz o mesmo pelo outro lado:
    contagem sobre arquivo real testa-se sobre ele, porque uma fixture que reproduz a propriedade
    está a testar a fixture.

    A propriedade é algébrica e vale em qualquer matriz; o que o dado real acrescenta é que
    ela vale **com as populações, as ausências e os empates que o laboratório produz** — e
    empate de posto é exactamente onde uma implementação de corte por posto se parte.

    ## As duas metades, e a primeira é a que pode ser trivial

    A metade de `0%` é identidade e passaria mesmo que o segundo passe não corresse. Por isso
    a metade de `10%` vem com a asserção de que ela **remove alguma coisa**: sem isso, a
    contenção estrita seria verdadeira por vacuidade.
    """
    from refino.preprocessing.filtering import filter_by_ranking

    modo = "neg_c18"
    matriz = load_matrix(measurements_path(modo), sample_list_path(modo), mode=modo)

    so_iqr, _, _ = filter_by_ranking(matriz, "iqr", keep_fraction=0.40)

    # Metade 1: abundância a 0% de REMOÇÃO — isto é, retendo 100% — é identidade.
    encadeado_neutro, _, _ = filter_by_ranking(so_iqr, "mean", keep_fraction=1.0)
    assert set(encadeado_neutro.feature_names) == set(so_iqr.feature_names), (
        "com o segundo passe a reter 100% o conjunto mudou. A causa não é o encadeamento — é "
        "a política de indefinido a descartar feature sobre a qual a estatística não se "
        "calcula, e isso contaminaria a contenção medida abaixo"
    )

    # Metade 2: abundância a 10% de remoção — retendo 90% — é subconjunto ESTRITO.
    encadeado, _, _ = filter_by_ranking(so_iqr, "mean", keep_fraction=0.90)
    removidas = set(so_iqr.feature_names) - set(encadeado.feature_names)
    assert removidas, (
        "o segundo passe não removeu nada no `neg_c18`, e a contenção estrita abaixo seria "
        "trivialmente verdadeira"
    )
    assert set(encadeado.feature_names) < set(so_iqr.feature_names), (
        "o conjunto encadeado não está contido no da variância sozinha: a ordem das etapas "
        "está trocada, e o segundo passe está a ver features que o primeiro descartou"
    )
