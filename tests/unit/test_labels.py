"""Vocabulário do domínio: guarda em runtime e paridade com os registros.

Dois grupos de teste, e o segundo é o que dá dente ao primeiro.

A guarda (`validate_label`) é a de runtime: `Literal` é análise
estática, e dado vindo de arquivo, de CLI ou do `metrics_json` nunca passou pelo
mypy.

A paridade fecha o modo de falha que se **cria** ao separar nome de
implementação: método novo no módulo sem o valor correspondente no `Literal`, ou
o inverso. E a lista de isenção — tipos cujo registro nasce em fase futura —
precisa do teste que impede que ela vire a saída de emergência que esvazia a
verificação, exatamente como `GLOBALLY_ENABLED` precisou.
"""

import importlib
import re
from pathlib import Path

import pytest

from refino.domain.labels import VOCABULARY, InvalidLabelError, validate_label

# Os dez tipos que moram em `domain`. Declarado aqui e não derivado
# de `VOCABULARY`: se o registro perder um tipo, comparar `VOCABULARY` consigo
# mesmo não acusaria nada.
EXPECTED_KINDS = frozenset(
    {
        "Correction",
        "RsdVariant",
        "Imputation",
        "SampleNormalization",
        "Transformation",
        "Scaling",
        "DefectType",
        "PreTransform",
        "MeasurementPoint",
        "RsdEstimator",
        "NullDenominatorPolicy",
        "UncorrectedCause",
        "Orientation",
        # Acrescentados com o gerador sintético. `InjectableDefect` é subconjunto próprio de
        # `DefectType`
        # e a relação é afirmada abaixo — os dois conjuntos existirem sem essa amarra é o
        # que produziria matriz gerada com tipo que `diagnose` nunca devolve.
        "InjectableDefect",
        "ImitatedMode",
        # Acrescentado na paridade com o MetaboAnalyst: a redução de features por posto,
        # que é a etapa que o pipeline não tinha.
        "RankingFilter",
        # Os cinco de `AnalysisSpec`. São parâmetros do que se
        # CALCULA da matriz processada, e por isso vivem ao lado de `PipelineSpec` e não
        # dentro dele: a impressão digital daquele é a chave de cache do multiverso.
        "PostHoc",
        "ClusterMetric",
        "ClusterLinkage",
        "CorrelationMethod",
        "ClassifierKind",
        # O teste do contraste, escolhido na tela. Também de `AnalysisSpec`, pelo mesmo
        # motivo dos cinco acima: trocar o teste não move uma célula da matriz processada.
        "UnivariateTest",
        # Os dois da predição de vias. `IonizationMode` é o que separa os quatro modos
        # do estudo em duas polaridades; `NeutralMassPolicy` decide o destino da feature
        # que o Progenesis já deconvoluiu.
        "IonizationMode",
        "NeutralMassPolicy",
        # Quantas formas ionicas procurar por composto. Nasceu da comparacao contra o
        # MetaboAnalystR: e o parametro que separa as duas implementacoes.
        "AdductSearch",
        # As recusas do roteiro R das vias: texto lido da saida de um processo.
        "ViasRecusa",
    }
)

# Registro de implementações por tipo: onde procurar as chaves que têm de casar
# com `get_args`. Nasceu vazio e foi preenchido por `RsdVariant` e por
# `PreTransform` — o movimento de SEM_REGISTRO para cá é forçado por
# `test_isencao_nao_contem_tipo_cujo_registro_ja_existe`, que reprova no instante
# em que o registro passa a existir.
PARITY_CASES: dict[str, tuple[str, str]] = {
    "RsdVariant": ("refino.metrics.rsd", "VARIANT_IMPLEMENTATIONS"),
    "PreTransform": ("refino.metrics.structure", "PRETRANSFORM_IMPLEMENTATIONS"),
    # Nasce em PARITY_CASES e não em SEM_REGISTRO: os dois valores têm
    # implementação desde que o `Literal` passou a existir.
    "Orientation": ("refino.io.readers", "ORIENTATION_HANDLERS"),
    # Entrou em PARITY_CASES no mesmo commit que criou o registro: o gatilho
    # (`test_isencao_nao_contem_tipo_cujo_registro_ja_existe`) reprova no instante em que
    # `RANKING_IMPLEMENTATIONS` passa a existir, e foi ele que forcou este movimento.
    "RankingFilter": ("refino.preprocessing.filtering", "RANKING_IMPLEMENTATIONS"),
    # Os dois do agrupamento entraram aqui no mesmo commit que criou `statistics/clustering.py`,
    # e **foi o gatilho que os moveu, não a memória de quem escreveu**: assim que os dois
    # registos passaram a existir, `test_isencao_nao_contem_tipo_cujo_registro_ja_existe` reprovou
    # com o nome de cada um.
    "ClusterMetric": ("refino.statistics.clustering", "CLUSTER_METRIC_IMPLEMENTATIONS"),
    "ClusterLinkage": ("refino.statistics.clustering", "CLUSTER_LINKAGE_IMPLEMENTATIONS"),
    # **O da correlação entrou aqui pelo MESMO gatilho, e a evidência é a mensagem dele.** Assim que
    # `refino.statistics.correlacao.CORRELATION_IMPLEMENTATIONS` passou a existir,
    # `test_isencao_nao_contem_tipo_cujo_registro_ja_existe` reprovou nomeando `CorrelationMethod`
    # — o mecanismo a mover a isenção em vez de alguém se lembrar dela.
    "CorrelationMethod": ("refino.statistics.correlacao", "CORRELATION_IMPLEMENTATIONS"),
    # **Nasce em PARITY_CASES, como `Orientation`: o registro nasceu no mesmo commit do `Literal`.**
    # O despacho é a tabela que leva cada escolha ao `parametric` de `univariate_test`, e
    # uma escolha sem entrada nela levantaria `KeyError` na rota depois da correção.
    "UnivariateTest": ("refino.domain.analysis", "PARAMETRIC_BY_UNIVARIATE_TEST"),
}

# Tipos sem registro de implementações, com a FASE de cada um. A fase é
# obrigatória: isenção sem gatilho de revisão é a lista que ninguém revisa.
SEM_REGISTRO: dict[str, str] = {
    "Correction": "Fase 3 — correction/base.py e os três corretores",
    "Imputation": "Fase 5 — preprocessing/missing.py",
    "SampleNormalization": "Fase 5 — preprocessing/normalization.py",
    "Transformation": "Fase 5 — preprocessing/normalization.py",
    "Scaling": "Fase 5 — preprocessing/normalization.py",
    "DefectType": "Fase 4 — diagnosis/profile.py",
    "MeasurementPoint": "sem registro por desenho: rótulo de contexto, não despacho",
    "RsdEstimator": "Fase 3 — os dois estimadores fora da amostra",
    # Os dois da correção fecham com os corretores, e o motivo é o mesmo de
    # `Correction`: a paridade exige as chaves IGUAIS a `get_args`, e as causas do
    # SERRF só existem quando o SERRF existe. Ver o docstring de
    # `correction/base.py`.
    "NullDenominatorPolicy": "Fase 3, U7 — correction/loess.py e os dois modos",
    # **Os cinco de `AnalysisSpec` nascem sem registro, e a fase de cada um é a unidade que o
    # cria.** O gatilho é o mesmo dos de cima: o instante em que o registro nomeado em
    # `REGISTROS_FUTUROS` passar a existir, `test_isencao_nao_contem_tipo_cujo_registro_ja_existe`
    # reprova e força a mudança para `PARITY_CASES` — sem ninguém ter de se lembrar.
    "PostHoc": "bloco das análises, U5 — statistics/anova.py e os dois post-hoc",
    "ClassifierKind": "bloco das análises, U15 — statistics/classificadores.py",
    "UncorrectedCause": "Fase 3, U10 — as cinco causas só existem com os dois corretores",
    # Os dois do gerador sintético são rótulos de CONTEXTO e não de despacho, como
    # `MeasurementPoint`: `InjectableDefect` seleciona o ramo de degradação dentro de
    # `generate`, e a paridade dele com `DefectType` é afirmada diretamente em
    # `test_injectable_defect_e_subconjunto_proprio_de_defect_type` — que é amarra mais
    # forte que um registro de implementações, porque compara os dois conjuntos.
    "InjectableDefect": "sem registro por desenho: a paridade é contra DefectType",
    "ImitatedMode": "sem registro por desenho: rótulo de referente, não despacho",
    # Os dois da predição de vias, e os dois são isentos por DESENHO e não por fase.
    # `IonizationMode` é rótulo de referente, como `ImitatedMode`: diz sobre qual
    # polaridade o dado foi medido, e não seleciona implementação nossa — quem despacha
    # nele é a lista de adutos da biblioteca de terceiro, que não é registro que
    # possamos manter em paridade.
    "IonizationMode": "sem registro por desenho: rótulo de referente, não despacho",
    # `NeutralMassPolicy` seleciona ramo dentro de `prediz_vias`, como `InjectableDefect`
    # dentro de `generate`. Um registro de dois membros para um `if` de duas pernas seria
    # cerimônia; o que amarra os valores é `test_politica_de_massa_neutra_muda_a_conta`,
    # que afirma que os dois produzem recortes DIFERENTES sobre o mesmo dado.
    "NeutralMassPolicy": "sem registro por desenho: dois ramos, amarrados por teste de par causal",
    # `AdductSearch` indexa a tabela de adutos do mummichog, que e registro de TERCEIRO:
    # nao ha chaves nossas com que bijetar. O que o amarra e
    # `test_a_busca_ampla_procura_mais_formas_ionicas`, que afirma a desigualdade medida.
    "AdductSearch": "sem registro por desenho: indexa tabela de terceiro, amarrado por teste",
    # `ViasRecusa` tem registro, mas ele e o vetor `DESFECHOS` de um roteiro R, que nao se importa:
    # `tests/test_vias_entrypoint_contrato.py` le o roteiro e afirma a bijecao com este Literal.
    "ViasRecusa": "sem registro por desenho: o registro e o roteiro R, amarrado por teste",
}

# Onde cada tipo isento terá seu registro. Alimenta o gatilho de revisão: assim
# que o atributo existir, a isenção correspondente reprova.
REGISTROS_FUTUROS: dict[str, tuple[str, str]] = {
    "RsdVariant": ("refino.metrics.rsd", "VARIANT_IMPLEMENTATIONS"),
    "PreTransform": ("refino.metrics.structure", "PRETRANSFORM_IMPLEMENTATIONS"),
    "Orientation": ("refino.io.readers", "ORIENTATION_HANDLERS"),
    "Correction": ("refino.correction.base", "CORRECTOR_IMPLEMENTATIONS"),
    "Imputation": ("refino.preprocessing.missing", "IMPUTATION_IMPLEMENTATIONS"),
    "SampleNormalization": (
        "refino.preprocessing.normalization",
        "SAMPLE_NORMALIZATION_IMPLEMENTATIONS",
    ),
    "Transformation": ("refino.preprocessing.normalization", "TRANSFORMATION_IMPLEMENTATIONS"),
    "Scaling": ("refino.preprocessing.normalization", "SCALING_IMPLEMENTATIONS"),
    "DefectType": ("refino.diagnosis.profile", "DEFECT_TYPE_IMPLEMENTATIONS"),
    "RsdEstimator": ("refino.metrics.crossval", "ESTIMATOR_IMPLEMENTATIONS"),
    "NullDenominatorPolicy": ("refino.correction.loess", "NULL_DENOMINATOR_IMPLEMENTATIONS"),
    "UncorrectedCause": ("refino.correction.base", "CAUSE_EMITTERS"),
    "RankingFilter": ("refino.preprocessing.filtering", "RANKING_IMPLEMENTATIONS"),
    "PostHoc": ("refino.statistics.anova", "POST_HOC_IMPLEMENTATIONS"),
    "ClusterMetric": ("refino.statistics.clustering", "CLUSTER_METRIC_IMPLEMENTATIONS"),
    "ClusterLinkage": ("refino.statistics.clustering", "CLUSTER_LINKAGE_IMPLEMENTATIONS"),
    "CorrelationMethod": ("refino.statistics.correlacao", "CORRELATION_IMPLEMENTATIONS"),
    "ClassifierKind": ("refino.statistics.classificadores", "CLASSIFIER_IMPLEMENTATIONS"),
}


def _registry_keys(module_name: str, attr: str) -> set[str] | None:
    """Chaves do registro, ou `None` quando ele ainda não existe."""
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError:  # pragma: no cover - os módulos existem; guarda para renomeação
        return None
    registry = getattr(module, attr, None)
    if registry is None:
        return None
    return set(registry)


# ---------------------------------------------------------------------------
# O registro de vocabulário
# ---------------------------------------------------------------------------


def test_vocabulary_cobre_exatamente_os_tipos_declarados() -> None:
    assert set(VOCABULARY) == EXPECTED_KINDS, (
        "VOCABULARY divergiu dos tipos que a §5.1 manda morar em domain. "
        f"Sobrando: {sorted(set(VOCABULARY) - EXPECTED_KINDS)}; "
        f"faltando: {sorted(EXPECTED_KINDS - set(VOCABULARY))}"
    )


def test_nenhum_tipo_do_vocabulario_esta_vazio() -> None:
    vazios = [kind for kind, valores in VOCABULARY.items() if not valores]
    assert not vazios, f"tipo sem valores é `Literal` mal formado: {vazios}"


@pytest.mark.parametrize("kind", sorted(EXPECTED_KINDS))
def test_todo_valor_declarado_e_aceito_pela_guarda(kind: str) -> None:
    for valor in VOCABULARY[kind]:
        validate_label(kind, valor)


# ---------------------------------------------------------------------------
# A guarda em runtime
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kind", "valor"),
    [
        ("RsdVariant", "outlier removed"),  # espaço em vez de sublinhado
        ("RsdVariant", "plain "),
        ("RsdVariant", "Plain"),  # caixa
        ("RsdVariant", ""),
        ("PreTransform", "log10"),  # existe em Transformation, não aqui
        ("Scaling", "pareto_scaling"),
        ("Correction", "SERRF"),  # caixa: o vocabulário é minúsculo
        ("MeasurementPoint", "post-correction"),  # hífen em vez de sublinhado
        ("ViasRecusa", "biblioteca ausente"),  # espaço em vez de sublinhado
        ("ViasRecusa", "ok"),  # sucesso nao e recusa
    ],
)
def test_guarda_reprova_valor_invalido(kind: str, valor: str) -> None:
    with pytest.raises(InvalidLabelError):
        validate_label(kind, valor)


def test_guarda_reprova_valor_de_outro_tipo() -> None:
    """`"pareto"` é `Scaling` legítimo e `RsdVariant` inválido.

    É o modo de falha concreto: o valor existe no vocabulário do projeto,
    então quem lê o código não desconfia — e ele cai no ramo de outra política.
    """
    validate_label("Scaling", "pareto")
    with pytest.raises(InvalidLabelError):
        validate_label("RsdVariant", "pareto")


def test_guarda_reprova_valor_nao_textual() -> None:
    for valor in (None, 0, 1.5, ["plain"]):
        with pytest.raises(InvalidLabelError):
            validate_label("RsdVariant", valor)


def test_kind_desconhecido_e_defeito_de_programacao_nao_de_dado() -> None:
    """`KeyError`, não `InvalidLabelError`.

    Confundir os dois faria a fronteira reportar "dado inválido" para o que é
    erro de digitação no nome do tipo — e quem ingere trataria como problema do
    arquivo do laboratório.
    """
    with pytest.raises(KeyError):
        validate_label("RsdVarient", "plain")
    with pytest.raises(InvalidLabelError):
        validate_label("RsdVariant", "plain2")


def test_mensagem_de_erro_lista_os_valores_aceitos() -> None:
    """Mensagem sem os aceitos obriga quem ingere a ler o código-fonte."""
    with pytest.raises(InvalidLabelError, match=r"outlier_removed"):
        validate_label("RsdVariant", "outlier-removed")


# ---------------------------------------------------------------------------
# Paridade entre nome e implementação
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", sorted(PARITY_CASES))
def test_paridade_entre_literal_e_registro(kind: str) -> None:
    module_name, attr = PARITY_CASES[kind]
    chaves = _registry_keys(module_name, attr)
    assert chaves is not None, (
        f"{kind} está em PARITY_CASES mas {module_name}.{attr} não existe. "
        "Mova para SEM_REGISTRO com a fase, ou crie o registro."
    )
    declarados = set(VOCABULARY[kind])
    assert declarados == chaves, (
        f"deriva em {kind}: declarado sem implementação "
        f"{sorted(declarados - chaves)}; implementado sem nome no Literal "
        f"{sorted(chaves - declarados)}"
    )


def test_todo_tipo_esta_coberto_por_paridade_ou_isencao() -> None:
    """Tipo novo sem nenhum dos dois reprova — é o teste que fecha a regra.

    Sem ele, acrescentar um `Literal` a `labels.py` e esquecer de ligá-lo a um
    registro passaria calado, e a paridade cobriria um subconjunto arbitrário.
    """
    coberto = set(PARITY_CASES) | set(SEM_REGISTRO)
    assert coberto == EXPECTED_KINDS, (
        f"sem paridade nem isenção: {sorted(EXPECTED_KINDS - coberto)}; "
        f"em duas listas ou inexistente: {sorted(coberto - EXPECTED_KINDS)}"
    )


def test_isencao_e_paridade_sao_disjuntas() -> None:
    sobreposto = set(PARITY_CASES) & set(SEM_REGISTRO)
    assert not sobreposto, f"tipo isento E com paridade: {sorted(sobreposto)}"


def test_isencao_nao_contem_tipo_cujo_registro_ja_existe() -> None:
    """O gatilho de revisão mecanizado.

    Isenção sem gatilho é a lista que sobrevive à razão dela. Quando alguém criar
    `VARIANT_IMPLEMENTATIONS`, este teste reprova até que a isenção de
    `RsdVariant` saia e a paridade passe a valer — sem ninguém precisar lembrar.
    """
    existentes = {
        kind: f"{module}.{attr}"
        for kind, (module, attr) in REGISTROS_FUTUROS.items()
        if kind in SEM_REGISTRO and _registry_keys(module, attr) is not None
    }
    assert not existentes, (
        "o registro destes tipos já existe e a isenção continua lá: "
        f"{existentes}. Remova de SEM_REGISTRO e acrescente a PARITY_CASES."
    )


def test_todo_tipo_isento_com_registro_previsto_tem_endereco() -> None:
    """Isenção cujo registro futuro não tem endereço não pode ser gatilhada.

    `MeasurementPoint` é a exceção declarada: rótulo de contexto, sem despacho,
    e por isso não tem registro previsto nem nunca terá.
    """
    sem_endereco = [
        kind
        for kind in SEM_REGISTRO
        if kind not in REGISTROS_FUTUROS and "desenho" not in SEM_REGISTRO[kind]
    ]
    assert not sem_endereco, (
        f"isento sem endereço de registro futuro: {sem_endereco}. "
        "Sem endereço, o gatilho de revisão não dispara nunca."
    )


def test_toda_isencao_nomeia_a_fase() -> None:
    """Toda isenção nomeia QUANDO expira, ou por que nunca expira.

    **O oráculo passou a aceitar unidade além de fase, e
    a mudança é do CONTRATO e não um afrouxamento.** Ele exigia a substring `Fase` porque, até
    aqui, toda isenção era de fase — e o que a forma codificava era a propriedade *"a isenção
    nomeia o momento em que se revê"*. Os cinco rótulos de `AnalysisSpec` nascem dentro de um
    **bloco**, não de uma fase, e escrever «Fase» neles seria falso para agradar a um `regex`.

    Um marcador de unidade é gatilho **mais forte** e não mais fraco:
    nomeia a unidade e o módulo (`statistics/anova.py`), contra um número de fase que nomeia
    semanas. E o gatilho
    mecânico continua a ser `test_isencao_nao_contem_tipo_cujo_registro_ja_existe`, que reprova no
    instante em que o registro nomeado em `REGISTROS_FUTUROS` existir — esse não se mexeu.
    """
    sem_gatilho = [
        kind
        for kind, motivo in SEM_REGISTRO.items()
        if not motivo
        or not (
            "Fase" in motivo or "desenho" in motivo or re.search(r"\bU\d+\b", motivo) is not None
        )
    ]
    assert not sem_gatilho, (
        f"isenção sem fase, unidade nomeada nem justificativa de desenho: {sem_gatilho}. "
        "É o que a torna revisável em vez de permanente."
    )


# ---------------------------------------------------------------------------
# A direção dos imports, e é o motivo de os tipos morarem em `domain`
# ---------------------------------------------------------------------------


def test_domain_nao_importa_as_camadas_de_fora() -> None:
    """`domain` é a camada mais interna: a seta só aponta para fora.

    Se `domain` importasse `metrics`, o argumento inteiro da paridade desapareceria —
    os `Literal` poderiam morar ao lado das implementações.
    """
    proibidos = ("refino.metrics", "refino.correction", "refino.preprocessing", "refino.io")
    dominio = Path(__file__).resolve().parents[2] / "src" / "refino" / "domain"
    ofensas: list[str] = []
    for arquivo in sorted(dominio.glob("*.py")):
        texto = arquivo.read_text(encoding="utf-8")
        ofensas.extend(
            f"{arquivo.name} importa {pacote}" for pacote in proibidos if pacote in texto
        )
    assert not ofensas, f"inversão de camada da §5.1: {ofensas}"


def test_o_detector_de_inversao_de_camada_consegue_falhar() -> None:
    """Prova que o teste acima não passa por varrer o diretório errado."""
    dominio = Path(__file__).resolve().parents[2] / "src" / "refino" / "domain"
    arquivos = sorted(dominio.glob("*.py"))
    assert len(arquivos) >= 3, f"varredura vazia ou rasa: {[a.name for a in arquivos]}"
    assert any(a.name == "labels.py" for a in arquivos)
    assert any("refino.domain" in a.read_text(encoding="utf-8") for a in arquivos), (
        "nenhum arquivo do pacote menciona refino.domain — o detector de "
        "substring não está lendo o que pensa que lê"
    )


def test_acessor_de_registro_ausente_devolve_none() -> None:
    """A coleta não pode morrer por causa de módulo de fase futura."""
    assert _registry_keys("refino.preprocessing.normalization", "SCALING_IMPLEMENTATIONS") is None
    assert _registry_keys("refino.domain.labels", "VOCABULARY") is not None


# ---------------------------------------------------------------------------
# A relação entre `InjectableDefect` e `DefectType`
# ---------------------------------------------------------------------------


def test_injectable_defect_e_subconjunto_proprio_de_defect_type() -> None:
    """Os quatro injetáveis contra os cinco classificáveis, nas duas direções.

    **Por que a amarra existe.** O gerador sintético injeta quatro tipos; `diagnose` classifica
    cinco, porque `unassessable` é ausência de medida e não há como injetá-la. Dois conjuntos
    para a mesma pergunta derivam — e a divergência apareceria como matriz gerada com tipo que o
    classificador nunca devolve, ou pior, como tipo classificável que o gerador nunca exercita.
    """
    injetaveis = set(VOCABULARY["InjectableDefect"])
    classificaveis = set(VOCABULARY["DefectType"])

    assert injetaveis < classificaveis, (
        f"InjectableDefect não é subconjunto próprio de DefectType. Injetáveis: "
        f"{sorted(injetaveis)}; classificáveis: {sorted(classificaveis)}"
    )
    assert classificaveis - injetaveis == {"unassessable"}, (
        "a diferença entre os dois conjuntos deixou de ser exatamente {'unassessable'}: "
        f"{sorted(classificaveis - injetaveis)}. Tipo classificável que o gerador não injeta é "
        "ramo que o [CA-7.4] nunca exercita, e tipo injetável que diagnose não devolve é matriz "
        "que ele não pode acertar"
    )


def test_imitated_mode_sao_os_quatro_modos_do_estudo() -> None:
    """Os quatro modos como rótulo fechado, e a contagem é observação e não invariante.

    Um quinto modo entra no `Literal` **e** ganha as próprias propriedades medidas
    — o rótulo fechado é o que força as duas coisas juntas em vez de deixar a segunda
    para depois.
    """
    assert set(VOCABULARY["ImitatedMode"]) == {"pos_hilic", "pos_c18", "neg_hilic", "neg_c18"}


# ---------------------------------------------------------------------------
# Os cinco eixos alinhados ao MetaboAnalyst
# ---------------------------------------------------------------------------

# O que cada valor nosso corresponde do lado do MetaboAnalyst. Declarado como dado e
# nao como prosa porque e o que o teste abaixo compara: `docs/metodos-metaboanalyst.md`
# cita a fonte, e este mapa e a metade executavel da mesma afirmacao.
#
# **A contagem e explicita e nao derivada do `Literal`.** Comparar `get_args` consigo
# mesmo concordaria sobre um valor a menos, que e exatamente o modo de falha que este
# teste existe para pegar — o mesmo argumento da linha 34 de
# `tests/test_metaboanalyst_spec_citation.py`.
EIXOS_ALINHADOS: dict[str, dict[str, str]] = {
    "Imputation": {
        "one_fifth_min": "ReplaceMin — um quinto do minimo positivo",
        "knn": "KNN (sample-wise) — o vizinho e outra AMOSTRA",
        "knn_features": "KNN (feature-wise) — o vizinho e outra FEATURE",
        "median": "median",
        "mean": "mean",
        "col_min": "min — o minimo da coluna, SEM o quinto",
        "exclude": "exclude — remove a feature com qualquer ausente",
        "missforest": "sem contraparte: nosso, e fica",
        "none": "sem contraparte: nosso, e fica",
    },
    "SampleNormalization": {
        "none": "None",
        "sum": "SumNorm",
        "median": "MedianNorm",
        "pqn_group": "GroupPQN — o grupo de referencia e o dos QCs",
        "sample_reference": "SamplePQN — a amostra de referencia e nomeada pelo usuario",
        "feature_reference": "CompNorm — a feature de referencia e nomeada pelo usuario",
        "supplied_factors": "SpecNorm — o fator por amostra vem de fora do equipamento",
        "quantile": "QuantileNorm",
    },
    "Transformation": {
        "none": "None",
        "log10": "LogNorm",
        "log2": "Log2Norm",
        "sqrt": "SrNorm",
        "cbrt": "CrNorm",
    },
    "Scaling": {
        "none": "None",
        "mean_center": "MeanCenter",
        "auto": "AutoNorm",
        "pareto": "ParetoNorm",
        "range": "RangeNorm",
        "vast": "sem contraparte: nosso, e fica",
    },
    "RankingFilter": {
        "none": "None",
        "iqr": "iqr",
        "sd": "sd",
        "mad": "mad",
        "rsd": "rsd",
        "nrsd": "nrsd",
        "mean": "mean",
        "median": "median",
    },
}


@pytest.mark.parametrize("kind", sorted(EIXOS_ALINHADOS))
def test_o_eixo_alinhado_tem_exatamente_os_valores_declarados(kind: str) -> None:
    """Valor que some do `Literal` reprova aqui, e e a unica coisa que o pega.

    A paridade generica compara `VOCABULARY` com `get_args` do mesmo `Literal` — ela
    concorda consigo mesma quando um valor desaparece. Este mapa e a lista escrita a mao
    contra a qual a remocao tem de reprovar, e cada chave carrega a contraparte do
    MetaboAnalyst para que a afirmacao de paridade tenha referente conferivel.
    """
    declarados = set(VOCABULARY[kind])
    esperados = set(EIXOS_ALINHADOS[kind])
    assert declarados == esperados, (
        f"o eixo {kind} divergiu do alinhamento declarado: "
        f"no Literal e nao no mapa {sorted(declarados - esperados)}; "
        f"no mapa e nao no Literal {sorted(esperados - declarados)}. "
        "Mudar um eixo alinhado exige mexer aqui E em `docs/metodos-metaboanalyst.md`."
    )


def test_o_mapa_de_alinhamento_nao_esta_vazio() -> None:
    """Ancora de nao-vacuidade do teste acima.

    `set() == set()` passa. Sem esta ancora, um mapa esvaziado por acidente faria os
    cinco parametrizados passarem sobre nada — asserção satisfeita por vacuidade.
    """
    assert len(EIXOS_ALINHADOS) == 5
    for kind, mapa in EIXOS_ALINHADOS.items():
        assert len(mapa) >= 5, f"o eixo {kind} tem {len(mapa)} valores no mapa; esperava >= 5"


def test_pqn_sem_sufixo_nao_e_mais_vocabulario() -> None:
    """O rotulo antigo sai do `Literal`, e isso e o que o rename significa.

    Deixa-lo aceito pela guarda faria a substituicao ser so cosmetica: os dois nomes
    conviveriam e a escrita poderia emitir qualquer um. A compatibilidade de LEITURA
    vive em `PipelineSpec`, na fronteira, e nao no vocabulario.
    """
    with pytest.raises(InvalidLabelError):
        validate_label("SampleNormalization", "pqn")


# ---------------------------------------------------------------------------
# A fronteira: `PipelineSpec` traduz o rótulo antigo e valida o eixo novo
# ---------------------------------------------------------------------------


def _spec(**troca: object) -> object:
    from refino.domain.spec import PipelineSpec

    campos: dict[str, object] = {
        "correction": "none",
        "n_batches": 1,
        "rsd_cutoff": None,
        "ranking_filter": "none",
        "ranking_keep_fraction": None,
        "ranking_filter_2": "none",
        "ranking_keep_fraction_2": None,
        "rsd_variant": "outlier_removed",
        "missing_threshold": 0.8,
        "imputation": "one_fifth_min",
        "sample_normalization": "none",
        "normalization_reference": None,
        "normalization_factors": None,
        "transformation": "none",
        "scaling": "none",
    }
    campos.update(troca)
    return PipelineSpec(**campos)  # type: ignore[arg-type]


def test_o_rotulo_pqn_antigo_e_lido_como_pqn_group() -> None:
    """Execução gravada antes do rename continua a ser lida — e a tradução é só na LEITURA.

    `refino.db` e os `--log` escritos antes da paridade carregam `pqn`. Sem esta tradução a
    guarda os recusaria, e o desfecho seria «dado inválido» para dado que este projeto mesmo
    escreveu. Que o campo guarde sempre o nome novo é a outra metade: sem ela os dois rótulos
    conviveriam e a escrita poderia emitir qualquer um.
    """
    assert _spec(sample_normalization="pqn").sample_normalization == "pqn_group"  # type: ignore[attr-defined]
    assert _spec(sample_normalization="pqn_group").sample_normalization == "pqn_group"  # type: ignore[attr-defined]


def test_a_traducao_nao_alcanca_rotulo_inventado() -> None:
    """A compatibilidade é de UM rótulo e não uma porta aberta.

    Sem esta âncora, um `except` largo ou um `.get` com padrão faria qualquer rótulo
    desconhecido virar `pqn_group` em silêncio — que é a classe de defeito que a tradução
    existe para não criar.
    """
    with pytest.raises(InvalidLabelError):
        _spec(sample_normalization="pqn_sample")
    with pytest.raises(InvalidLabelError):
        _spec(sample_normalization="PQN")


def test_o_eixo_de_filtro_e_validado_na_fronteira() -> None:
    """O eixo novo passa pela mesma guarda que os seis antigos."""
    for valido in VOCABULARY["RankingFilter"]:
        _spec(ranking_filter=valido)
    with pytest.raises(InvalidLabelError, match="RankingFilter"):
        _spec(ranking_filter="interquartile")
