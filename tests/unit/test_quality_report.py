"""`QualityReport` e `build_quality_report`.

Três mecanismos, cada um com os testes que provam que **conseguem** reprovar:

- o acoplamento dos rótulos de contexto, nas duas combinações contraditórias;
- a codificação de três estados de `computed`, nos três estados e nas duas
  violações da bicondicional;
- o espaço de features, incluindo o caso que só o digest pega — mesma contagem,
  conjunto diferente.
"""

import numpy as np
import pytest

from refino.domain.labels import InvalidLabelError
from refino.domain.matrix import RefinoMatrix
from refino.domain.metadata import SampleMetadata
from refino.domain.spec import (
    FAMILIAS_DO_PLSDA,
    OPTIONAL_METRIC_FIELDS,
    REQUIRED_METRIC_FIELDS,
    QualityReport,
)
from refino.metrics.report import build_quality_report
from refino.metrics.structure import retained_column_digest


def _matrix(values: np.ndarray, tipos: tuple[str, ...]) -> RefinoMatrix:
    samples = tuple(
        SampleMetadata(
            name=f"QC{i + 1}" if tipo == "qc" else f"amostra{i + 1}",
            injection_order=i + 1,
            sample_type=tipo,  # type: ignore[arg-type]
            group="QC" if tipo == "qc" else "NoHTLV",
        )
        for i, tipo in enumerate(tipos)
    )
    return RefinoMatrix(
        values=np.asarray(values, dtype=np.float64),
        feature_names=tuple(f"f{j + 1}" for j in range(values.shape[1])),
        samples=samples,
        mode="teste",
    )


def _drifting(n_samples: int = 12, n_features: int = 6, seed: int = 11) -> RefinoMatrix:
    """Mesmo modelo: nível por feature, deriva, ruído. Verdade conhecida."""
    rng = np.random.default_rng(seed)
    level = rng.uniform(1e4, 1e6, size=(1, n_features))
    drift = np.linspace(1.0, 3.0, n_samples)[:, None]
    noise = 1.0 + 0.05 * rng.standard_normal(size=(n_samples, n_features))
    tipos = tuple("qc" if i % 4 == 0 else "sample" for i in range(n_samples))
    return _matrix(level * drift * noise, tipos)


def _report(**overrides: object) -> QualityReport:
    """Relatório válido mínimo, com os campos obrigatórios em `computed`."""
    base: dict[str, object] = {
        "measurement_point": "post_correction",
        "rsd_variant": "outlier_removed",
        "rsd_estimator": "in_sample",
        "structural_pretransform": "log10p1",
        "computed": frozenset(REQUIRED_METRIC_FIELDS),
        "qc_rsd_median": 0.14,
        "qc_rsd_mean": 0.19,
        "frac_features_under_20": 0.62,
        "frac_features_under_30": 0.81,
        "n_features_rsd_undefined": 3,
        # **97 e não um número qualquer, e a coerência é com a linha acima.** As duas
        # contagens saem do MESMO vetor de RSD em `build_quality_report`: as definidas são
        # `finitas.size` e as indefinidas são o complemento, então uma base com 3 indefinidas
        # descreve um estudo de 100 features. Um valor arbitrário aqui daria uma fixture que
        # nenhum produtor sabe produzir, que é exactamente o defeito que esta base já tinha.
        "n_features_rsd_defined": 97,
        "qc_dispersion_ratio": 0.93,
        "pc1_injection_correlation_abs": 0.48,
        "structural_features_used": 100,
        "structural_features_dropped_zero_variance": 2,
        "structural_features_dropped_non_finite": 5,
        "structural_column_digest": "a" * 64,
        "qc_dispersion_ratio_spread": 1.3e-15,
        "pc1_injection_correlation_spread": 0.0,
        "stability_replicates": 32,
        "stability_seed": 0,
    }
    base.update(overrides)
    return QualityReport(**base)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Os quatro rótulos e o acoplamento
# ---------------------------------------------------------------------------


def test_relatorio_valido_da_fase_2() -> None:
    relatorio = _report()
    assert relatorio.rsd_estimator == "in_sample"
    assert relatorio.qc_rsd_cv_median is None
    assert relatorio.qc_rsd_r_cv_median is None
    assert not relatorio.was_computed("qc_rsd_cv_median")
    assert not relatorio.was_computed("qc_rsd_r_cv_median")


@pytest.mark.parametrize(
    ("point", "pretransform"),
    [("post_correction", "none"), ("post_pipeline", "log10p1")],
)
def test_as_duas_combinacoes_contraditorias_de_rotulo_sao_recusadas(
    point: str, pretransform: str
) -> None:
    """`log10` duplo e `log10` nenhum produzem número sem levantar.

    São exatamente as duas que a construção tem de recusar, e são duas e não uma —
    testar só uma deixaria metade do acoplamento sem verificação.
    """
    with pytest.raises(ValueError, match="exige structural_pretransform"):
        _report(measurement_point=point, structural_pretransform=pretransform)


@pytest.mark.parametrize(
    ("campo", "valor"),
    [
        ("measurement_point", "durante"),
        ("rsd_variant", "pareto"),
        ("rsd_estimator", "bootstrap"),
        ("structural_pretransform", "log10"),
    ],
)
def test_rotulo_invalido_levanta_pela_guarda_da_u1(campo: str, valor: str) -> None:
    """Na fronteira de volta: o `metrics_json` desserializado nunca viu o mypy."""
    with pytest.raises(InvalidLabelError):
        _report(**{campo: valor})


def test_as_duas_combinacoes_validas_sao_aceitas() -> None:
    """O outro lado: guarda que recusa tudo é indistinguível de tipo inconstruível."""
    assert _report(measurement_point="post_correction", structural_pretransform="log10p1")
    assert _report(measurement_point="post_pipeline", structural_pretransform="none")


# ---------------------------------------------------------------------------
# A codificação de três estados
# ---------------------------------------------------------------------------


def test_estado_nao_calculada() -> None:
    relatorio = _report()
    for nome in OPTIONAL_METRIC_FIELDS:
        assert getattr(relatorio, nome) is None
        assert not relatorio.was_computed(nome)


def test_estado_calculada_e_definida() -> None:
    """**O exemplo mudou de `plsda_q2` para `qc_rsd_cv_median` em 27/08/2026.**

    A propriedade medida é a — os três estados de um campo opcional — e ela não mudou. O
    que mudou é que `plsda_q2` deixou de ser um exemplo **livre**: a guarda da regra 6 do README
    exige que ele viaje com o `p` e com a procedência, então usá-lo aqui passou a medir aquela
    guarda em vez desta propriedade. `qc_rsd_cv_median` é opcional pelo mesmo mecanismo e não tem
    par.
    """
    relatorio = _report(
        computed=frozenset(REQUIRED_METRIC_FIELDS | {"qc_rsd_cv_median"}),
        qc_rsd_cv_median=0.24,
    )
    assert relatorio.was_computed("qc_rsd_cv_median")
    assert relatorio.qc_rsd_cv_median == 0.24


def test_estado_calculada_e_indefinida_e_nan_e_aceito() -> None:
    """`nan` num campo cujo nome está em `computed` é o terceiro estado, legítimo.

    Mesmo motivo do teste acima para a troca de exemplo.
    """
    relatorio = _report(
        computed=frozenset(REQUIRED_METRIC_FIELDS | {"qc_rsd_cv_median"}),
        qc_rsd_cv_median=float("nan"),
    )
    assert relatorio.was_computed("qc_rsd_cv_median")
    assert np.isnan(relatorio.qc_rsd_cv_median)


def test_nan_em_campo_obrigatorio_e_aceito() -> None:
    """Obrigatório indefinido é `nan`, nunca `None` — é o que torna o par distinguível.

    > **A FIXTURE MUDOU em 15/09/2026, e a mudança é ACHADO e não obstáculo.** Ela fazia
    > `_report(qc_rsd_median=nan, frac_features_under_20=nan)` sobre a base, cujo denominador é
    > positivo — e com a bicondicional de `_check_denominador_do_rsd` esse relatório passou a ser
    > **impossível de construir**. Está certo que passe: uma fração `nan` sobre features
    > **definidas** é um estado que `_fraction_under` não sabe produzir, porque ela só devolve
    > `nan` quando o vetor é vazio. A fixture descrevia um relatório irreal.
    >
    > **Actualizá-la não é afrouxar nada** (regra 2 do README): não há tolerância movida nem número
    > esperado ajustado — o que muda é a fixture passar a descrever um estado **alcançável**. E é
    > a melhor evidência possível de que a guarda não é decorativa: ela achou um defeito de
    > fixture na primeira corrida, sem ninguém plantar nada.
    """
    relatorio = _report(
        qc_rsd_median=float("nan"),
        frac_features_under_20=float("nan"),
        frac_features_under_30=float("nan"),
        n_features_rsd_defined=0,
    )
    assert np.isnan(relatorio.qc_rsd_median)
    assert relatorio.was_computed("qc_rsd_median")


def test_bicondicional_nome_em_computed_com_campo_none_reprova() -> None:
    with pytest.raises(ValueError, match="está em `computed` e vale `None`"):
        _report(computed=frozenset(REQUIRED_METRIC_FIELDS | {"plsda_q2"}))


def test_bicondicional_campo_preenchido_fora_de_computed_reprova() -> None:
    with pytest.raises(ValueError, match="não está em `computed`"):
        _report(plsda_q2=0.24)


def test_campo_obrigatorio_ausente_de_computed_reprova() -> None:
    with pytest.raises(ValueError, match="campos obrigatórios ausentes"):
        _report(computed=frozenset(REQUIRED_METRIC_FIELDS - {"qc_rsd_median"}))


def test_computed_com_nome_que_nao_e_campo_reprova() -> None:
    """Pertinência sobre nome inexistente responde sobre nada."""
    with pytest.raises(ValueError, match="campos que não existem"):
        _report(computed=frozenset(REQUIRED_METRIC_FIELDS | {"qc_rsd_mediana"}))


def test_was_computed_com_nome_desconhecido_levanta_key_error() -> None:
    """`KeyError` e não `False`: perguntar por campo que não existe é defeito de código."""
    with pytest.raises(KeyError):
        _report().was_computed("inventado")


def test_particao_dos_campos_de_metrica_e_disjunta_e_completa() -> None:
    """Obrigatório e opcional não se sobrepõem, e todo nome é campo de verdade.

    Mesma família do teste de exaustividade da partição de códigos de validação:
    um campo que caísse nos dois conjuntos teria duas regras contraditórias na
    construção, e um que caísse em nenhum não teria regra alguma.
    """
    from dataclasses import fields

    nomes = {f.name for f in fields(QualityReport)}
    assert not REQUIRED_METRIC_FIELDS & OPTIONAL_METRIC_FIELDS
    assert nomes >= REQUIRED_METRIC_FIELDS
    assert nomes >= OPTIONAL_METRIC_FIELDS


# ---------------------------------------------------------------------------
# O espaço de features
# ---------------------------------------------------------------------------


def test_digest_vazio_reprova() -> None:
    with pytest.raises(ValueError, match="structural_column_digest vazio"):
        _report(structural_column_digest="")


@pytest.mark.parametrize(
    "campo",
    [
        "structural_features_used",
        "structural_features_dropped_zero_variance",
        "structural_features_dropped_non_finite",
        "n_features_rsd_undefined",
        # **Entrou com o campo.** Esta lista ESPELHA à mão a tupla de
        # `spec.py::_check_feature_accounting`, e contagem nova que não entre nas duas cai em
        # silêncio: a guarda existiria e nada afirmaria que ela corre sobre este campo.
        "n_features_rsd_defined",
        "n_features_retained",
    ],
)
def test_contagem_negativa_reprova(campo: str) -> None:
    with pytest.raises(ValueError, match="não pode ser negativo"):
        _report(**{campo: -1})


def test_fracao_fora_de_zero_um_reprova() -> None:
    """O erro de unidade que já foi cometido uma vez."""
    with pytest.raises(ValueError, match="é fração e recebi"):
        _report(frac_features_under_20=62.0)


def test_comparable_with_exige_os_quatro_rotulos_e_o_mesmo_espaco() -> None:
    a = _report()
    assert a.comparable_with(_report())
    assert not a.comparable_with(_report(rsd_variant="plain"))
    assert not a.comparable_with(
        _report(measurement_point="post_pipeline", structural_pretransform="none")
    )
    assert not a.comparable_with(_report(rsd_estimator="r_cv"))


def test_mesma_contagem_com_conjunto_diferente_nao_e_comparavel() -> None:
    """**O caso que só o digest pega**, e a razão de ele existir.

    Duas especificações descartam 7 colunas cada uma, então as três contagens
    batem — e descartam colunas **diferentes**. Comparar `qc_dispersion_ratio`
    entre as duas é comparar medições em espaços distintos, e sem o digest nada
    distingue isso de duas medições no mesmo espaço.
    """
    a = _report(structural_column_digest="a" * 64)
    b = _report(structural_column_digest="b" * 64)
    assert a.structural_features_used == b.structural_features_used
    assert a.structural_features_dropped_non_finite == b.structural_features_dropped_non_finite
    assert not a.comparable_with(b)


def test_digest_distingue_conjuntos_de_mesmo_tamanho() -> None:
    """A propriedade do digest, medida em vez de suposta."""
    nomes = ("f1", "f2", "f3", "f4")
    primeiro = retained_column_digest(nomes, np.array([True, True, False, False]))
    segundo = retained_column_digest(nomes, np.array([False, False, True, True]))
    terceiro = retained_column_digest(nomes, np.array([True, True, False, False]))
    assert primeiro != segundo, "conjuntos diferentes de mesmo tamanho colidiram"
    assert primeiro == terceiro, "o digest tem de ser determinístico"


def test_digest_depende_do_nome_e_nao_so_da_mascara() -> None:
    """Se fosse sobre a máscara, duas matrizes de features diferentes colidiriam."""
    mascara = np.array([True, False, True])
    assert retained_column_digest(("a", "b", "c"), mascara) != retained_column_digest(
        ("x", "y", "z"), mascara
    )


def test_digest_exige_mascara_do_mesmo_tamanho_que_os_nomes() -> None:
    """`strict=True` no `zip`: máscara de tamanho errado é defeito, não recorte."""
    with pytest.raises(ValueError):
        retained_column_digest(("a", "b", "c"), np.array([True, False]))


# ---------------------------------------------------------------------------
# Imutabilidade
# ---------------------------------------------------------------------------


def test_relatorio_e_imutavel() -> None:
    relatorio = _report()
    with pytest.raises(AttributeError):
        relatorio.qc_rsd_median = 0.0  # type: ignore[misc]
    with pytest.raises(AttributeError):
        relatorio.computed = frozenset()  # type: ignore[misc]


# ---------------------------------------------------------------------------
# `build_quality_report`
# ---------------------------------------------------------------------------


def test_build_da_fase_2_declara_que_nao_calculou_os_dois_de_fora_da_amostra() -> None:
    relatorio = build_quality_report(
        _drifting(),
        measurement_point="post_correction",
        structural_pretransform="log10p1",
    )
    assert relatorio.rsd_estimator == "in_sample"
    assert relatorio.qc_rsd_cv_median is None
    assert relatorio.qc_rsd_r_cv_median is None
    assert "qc_rsd_cv_median" not in relatorio.computed
    assert relatorio.computed >= REQUIRED_METRIC_FIELDS


def test_build_com_os_dois_vetores_poe_os_nomes_em_computed() -> None:
    """A porta que a correção usa: vetores, nunca um corretor."""
    relatorio = build_quality_report(
        _drifting(),
        measurement_point="post_correction",
        structural_pretransform="log10p1",
        rsd_estimator="leave_one_qc_out",
        qc_rsd_cv=np.array([0.10, 0.20, 0.30]),
        qc_rsd_r_cv=np.array([0.40, np.nan, 0.60]),
    )
    assert relatorio.was_computed("qc_rsd_cv_median")
    assert relatorio.was_computed("qc_rsd_r_cv_median")
    assert relatorio.qc_rsd_cv_median == pytest.approx(0.20)
    # A mediana ignora o indefinido, como em toda agregação de RSD.
    assert relatorio.qc_rsd_r_cv_median == pytest.approx(0.50)


def test_build_nao_aceita_corretor_e_nem_conhece_correction() -> None:
    """Verificação estática de que `metrics/` não importa `correction/`."""
    import ast
    from pathlib import Path

    fonte = (
        Path(__file__).resolve().parent.parent.parent / "src" / "refino" / "metrics" / "report.py"
    ).read_text(encoding="utf-8")
    for node in ast.walk(ast.parse(fonte)):
        if isinstance(node, ast.ImportFrom):
            assert not (node.module or "").startswith("refino.correction"), (
                "metrics/report.py importou correction/, quebrando a inversão de dependência"
            )


def test_build_propaga_as_contagens_estruturais_e_o_digest() -> None:
    """O multiverso precisa dos três números e do digest, não da razão sozinha."""
    matrix = _drifting(n_features=6)
    valores = matrix.values.copy()
    valores[:, 1] = 5.0  # variância zero
    valores[2, 4] = np.inf  # não finito
    sujo = RefinoMatrix(
        values=valores,
        feature_names=matrix.feature_names,
        samples=matrix.samples,
        mode=matrix.mode,
    )
    relatorio = build_quality_report(
        sujo, measurement_point="post_correction", structural_pretransform="log10p1"
    )
    assert relatorio.structural_features_dropped_zero_variance == 1
    assert relatorio.structural_features_dropped_non_finite == 1
    assert relatorio.structural_features_used == 4
    assert len(relatorio.structural_column_digest) == 64


def test_build_com_feature_de_qcs_todos_zero_reduz_o_denominador() -> None:
    """A política de indefinido, e a fração difere da do denominador cheio.

    A segunda metade é a que tem dente: sem ela, uma implementação que usasse o
    denominador cheio passaria — as duas convenções só divergem quando há
    indefinido, e é justamente esse caso que o teste constrói.
    """
    tipos = ("qc", "sample", "sample", "qc", "sample", "qc")
    rng = np.random.default_rng(5)
    valores = rng.uniform(1e4, 1e6, size=(len(tipos), 4))
    # Três features com QCs APERTADOS: RSD bem abaixo de 0,20, para que a fração
    # não seja zero nas duas convenções — com fração zero o teste não distinguiria
    # Nada, que é o mesmo defeito num denominador.
    for i, tipo in enumerate(tipos):
        if tipo == "qc":
            valores[i, 0] = 1.0e5 * (1 + 0.01 * i)
            valores[i, 1] = 2.0e5 * (1 + 0.01 * i)
            valores[i, 3] = 3.0e5 * (1 + 0.01 * i)
            # E uma feature com todos os QCs zero: RSD indefinido por média zero.
            valores[i, 2] = 0.0

    relatorio = build_quality_report(
        _matrix(valores, tipos),
        measurement_point="post_correction",
        structural_pretransform="log10p1",
    )
    assert relatorio.n_features_rsd_undefined == 1

    from refino.metrics.rsd import qc_rsd

    rsd_values = qc_rsd(_matrix(valores, tipos), variant="plain")
    definidas = rsd_values[np.isfinite(rsd_values)]
    denominador_cheio = float(np.mean(np.nan_to_num(rsd_values, nan=1e9) < 0.20))
    assert relatorio.frac_features_under_20 == pytest.approx(float(np.mean(definidas < 0.20)))
    assert relatorio.frac_features_under_20 != pytest.approx(denominador_cheio), (
        "as duas convenções de denominador coincidem neste fixture, então ele não "
        "distingue as implementações e o teste seria decorativo"
    )

    # **O denominador passou a viajar como CAMPO.** Ele é o tamanho do vetor sobre que a
    # fração acima correu, e a asserção compara-o contra a contagem medida do outro lado, não
    # contra um literal: um literal fixaria o tamanho deste fixture em vez da propriedade.
    assert relatorio.n_features_rsd_defined == definidas.size
    # **E a âncora que o separa da contagem vizinha:** definidas e indefinidas medem o MESMO
    # vetor, e neste fixture elas DIFEREM — 3 contra 1. Sem isto, uma implementação que
    # devolvesse `n_features_rsd_undefined` no campo novo passaria.
    assert relatorio.n_features_rsd_defined != relatorio.n_features_rsd_undefined, (
        "as duas contagens coincidem neste fixture, e o teste não distinguiria uma da outra"
    )


def test_build_com_todas_as_features_indefinidas_da_nan_e_nao_levanta() -> None:
    """`nan` é "calculada e indefinida"; `None` diria que ninguém tentou."""
    tipos = ("qc", "sample", "sample", "qc", "sample", "qc")
    valores = np.zeros((len(tipos), 3))
    # As amostras variam para que o PCA tenha estrutura; os QCs são todos zero.
    for i, tipo in enumerate(tipos):
        if tipo != "qc":
            valores[i] = np.array([1e5, 2e5, 3e5]) * (1 + 0.1 * i)

    relatorio = build_quality_report(
        _matrix(valores, tipos),
        measurement_point="post_correction",
        structural_pretransform="log10p1",
    )
    assert np.isnan(relatorio.qc_rsd_median)
    assert np.isnan(relatorio.frac_features_under_20)
    assert relatorio.was_computed("qc_rsd_median")
    assert relatorio.n_features_rsd_undefined == 3
    # **O outro lado da bicondicional, e ele é PRODUZIDO e não plantado:** com todas as
    # features indefinidas, `_fraction_under` devolve `nan` e `finitas.size` é zero. É o estado
    # que a guarda aceita — e o único em que a fração pode ser `nan`.
    assert relatorio.n_features_rsd_defined == 0


def test_build_exige_os_dois_rotulos_de_contexto() -> None:
    """Nenhum dos dois tem padrão, mesmo sendo acoplados."""
    with pytest.raises(TypeError):
        build_quality_report(_drifting())  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        build_quality_report(_drifting(), measurement_point="post_correction")  # type: ignore[call-arg]


def test_build_recusa_a_combinacao_contraditoria() -> None:
    """O acoplamento vale pelo caminho de construção também, não só no dataclass."""
    with pytest.raises(ValueError, match="exige structural_pretransform"):
        build_quality_report(
            _drifting(),
            measurement_point="post_pipeline",
            structural_pretransform="log10p1",
        )


def test_build_respeita_pontos_de_corte_customizados() -> None:
    matrix = _drifting()
    frouxo = build_quality_report(
        matrix,
        measurement_point="post_correction",
        structural_pretransform="log10p1",
        rsd_cutoffs=(0.9, 0.95),
    )
    apertado = build_quality_report(
        matrix,
        measurement_point="post_correction",
        structural_pretransform="log10p1",
        rsd_cutoffs=(0.001, 0.002),
    )
    assert frouxo.frac_features_under_20 > apertado.frac_features_under_20


def test_none_em_campo_obrigatorio_reprova_na_fronteira() -> None:
    """`None` num obrigatório é dado inválido, não indefinido.

    O caminho existe porque o `metrics_json` volta do banco sem passar pelo
    mypy: `float` no dataclass é análise estática, e um `null` no JSON chega aqui
    como `None`. Sem a guarda ele viraria comparação com `nan` três camadas acima.
    """
    with pytest.raises(ValueError, match="é obrigatório e não pode ser `None`"):
        _report(qc_rsd_median=None)


# ---------------------------------------------------------------------------
# Estabilidade no relatório
# ---------------------------------------------------------------------------


def test_structurally_assessable_espelha_o_predicado_das_metricas() -> None:
    """A etapa seguinte lê o relatório, não a `StructuralMetrics`.

    Reimplementar a pergunta ali criaria duas definições de "avaliável" que
    derivam — a mesma razão tirar o vocabulário de quem implementa.
    """
    assert _report().structurally_assessable(max_spread=0.01)
    assert not _report(qc_dispersion_ratio=float("nan")).structurally_assessable(max_spread=0.01)
    assert not _report(pc1_injection_correlation_spread=0.85).structurally_assessable(
        max_spread=0.01
    )
    assert not _report(qc_dispersion_ratio_spread=float("inf")).structurally_assessable(
        max_spread=0.01
    )


def test_metrica_indefinida_nao_vira_avaliavel_afrouxando_a_tolerancia() -> None:
    """`nan` é ausência de medida, não instabilidade grande."""
    relatorio = _report(pc1_injection_correlation_abs=float("nan"))
    for tolerancia in (0.0, 1.0, 1e9):
        assert not relatorio.structurally_assessable(max_spread=tolerancia)


def test_structurally_assessable_exige_tolerancia_nomeada_e_nao_negativa() -> None:
    with pytest.raises(TypeError):
        _report().structurally_assessable(0.01)  # type: ignore[misc]
    with pytest.raises(ValueError, match="max_spread não pode ser negativo"):
        _report().structurally_assessable(max_spread=-0.5)


def test_amplitude_negativa_reprova() -> None:
    with pytest.raises(ValueError, match="é amplitude e não pode ser negativa"):
        _report(qc_dispersion_ratio_spread=-1e-9)


def test_zero_replicas_reprova() -> None:
    """Amplitude de lista vazia é `nan`, indistinguível de métrica indefinida."""
    with pytest.raises(ValueError, match="stability_replicates deve ser >= 1"):
        _report(stability_replicates=0)


def test_igualdade_de_relatorio_e_falsa_quando_algum_campo_e_nan() -> None:
    """**Armadilha para o multiverso**, afirmada aqui para que ela não a descubra sozinha.

    `==` de dataclass compara campo a campo, e `nan != nan` por IEEE 754. Dois
    relatórios **idênticos** cujas métricas sejam indefinidas comparam como
    diferentes. Qualquer deduplicação, cache ou "já vi este relatório" baseado em
    `==` dá falso negativo exatamente nas matrizes mal-condicionadas, que são as que
    mais aparecem repetidas no multiverso.

    `comparable_with` não sofre disso porque compara rótulos e digest, não valores —
    e é essa a comparação que o multiverso deve usar para decidir comensurabilidade.
    """
    a = _report(qc_rsd_median=float("nan"))
    b = _report(qc_rsd_median=float("nan"))
    assert a != b, "se isto passar a ser igual, a armadilha desapareceu e o aviso pode sair"
    assert a.comparable_with(b), "comparable_with não deve depender de valor"


def test_build_propaga_estabilidade_e_semente() -> None:
    relatorio = build_quality_report(
        _drifting(),
        measurement_point="post_correction",
        structural_pretransform="log10p1",
        stability_replicates=8,
        stability_seed=3,
    )
    assert relatorio.stability_replicates == 8
    assert relatorio.stability_seed == 3
    assert relatorio.qc_dispersion_ratio_spread < 1e-9
    assert relatorio.structurally_assessable(max_spread=0.01)


# ---------------------------------------------------------------------------
# A contagem de features significativas deixa de ser SEM CAMINHO
# ---------------------------------------------------------------------------


def test_build_sem_a_contagem_de_fdr_deixa_o_campo_fora_de_computed() -> None:
    """A âncora do par: sem o argumento, o estado continua *não calculada*.

    **E isso NÃO é o mesmo que «sem caminho»**: o primeiro diz *"esta execução não trouxe
    valor"*, o segundo diz *"nenhuma rota calcula isto"*. Ligar o caminho move a métrica de um
    estado para o outro, não para «sempre preenchida».
    """
    relatorio = build_quality_report(
        _drifting(),
        measurement_point="post_pipeline",
        structural_pretransform="none",
    )
    assert relatorio.n_significant_fdr is None
    assert "n_significant_fdr" not in relatorio.computed


def test_build_com_a_contagem_de_fdr_poe_o_nome_em_computed() -> None:
    """O outro lado do par. A contagem entra por argumento, como o par do PLS-DA.

    **Quem conta é o chamador, e não este construtor**, pelo mesmo motivo do PLS-DA: o teste
    univariado precisa do par de grupos, e `build_quality_report` não é o sítio que os conhece.
    """
    relatorio = build_quality_report(
        _drifting(),
        measurement_point="post_pipeline",
        structural_pretransform="none",
        n_significant_fdr=7,
    )
    assert relatorio.n_significant_fdr == 7
    assert "n_significant_fdr" in relatorio.computed


def test_a_contagem_de_fdr_igual_a_zero_e_calculada_e_nao_ausente() -> None:
    """**O caso que importa neste corpus: zero calculado não é ausência.**

    Medido no `pos_hilic` sobre 96 células do multiverso: `n_significant_fdr` deu **zero em todas
    as 96**, porque nenhuma feature sobrevive ao FDR — o menor `q` é 0,9976. Então zero é o valor
    esperado aqui, e um construtor que tratasse `0` como ausente — por `if n_significant_fdr:` em
    vez de `is not None` — apagaria a única resposta que este estudo dá, e a tela mostraria *"não
    calculada"* sobre um número que foi calculado.
    """
    relatorio = build_quality_report(
        _drifting(),
        measurement_point="post_pipeline",
        structural_pretransform="none",
        n_significant_fdr=0,
    )
    assert relatorio.n_significant_fdr == 0
    assert "n_significant_fdr" in relatorio.computed, (
        "zero foi tratado como ausente: a contagem de FDR é 0 neste corpus, e apagá-la deixaria a "
        "tela a dizer «não calculada» sobre a resposta que a ferramenta de facto deu"
    )


# ---------------------------------------------------------------------------
# A bicondicional do denominador
# ---------------------------------------------------------------------------


def test_d452_a_bicondicional_do_denominador_aceita_os_dois_estados_coerentes() -> None:
    """Os DOIS desfechos que a guarda deixa passar, e são os dois que o produtor sabe produzir.

    **Par causal em processo**: os mesmos campos, uma vez com fração definida e denominador
    positivo, outra com fração `nan` e denominador zero. Sem o primeiro, o teste da reprovação
    abaixo seria satisfeito por uma guarda que recusasse tudo.
    """
    definido = _report()
    assert definido.n_features_rsd_defined == 97
    assert not np.isnan(definido.frac_features_under_20)

    vazio = _report(
        frac_features_under_20=float("nan"),
        frac_features_under_30=float("nan"),
        n_features_rsd_defined=0,
    )
    assert np.isnan(vazio.frac_features_under_20)
    assert vazio.n_features_rsd_defined == 0


@pytest.mark.parametrize("campo", ["frac_features_under_20", "frac_features_under_30"])
@pytest.mark.parametrize(
    ("valor", "definidas", "sentido"),
    [
        (float("nan"), 97, "fração indefinida sobre features DEFINIDAS"),
        (0.62, 0, "fração definida sobre denominador ZERO"),
    ],
)
def test_d452_a_bicondicional_reprova_nos_dois_sentidos(
    campo: str, valor: float, definidas: int, sentido: str
) -> None:
    """Os dois estados que `_fraction_under` NÃO sabe produzir.

    A guarda é `nan ⟺ denominador zero`, e uma bicondicional tem **duas** direções de violação.
    Testar só uma deixaria metade inerte, e é a metade que não se manifesta como erro: um
    relatório que afirma 62% abaixo do corte **e** zero features com RSD definido viaja para a
    tela sem nada a acusar.

    **As DUAS frações**, porque partilham o denominador por construção: `_fraction_under` corre
    duas vezes sobre o mesmo vetor, e guardar só a de 20% deixaria `(nan, 0,81)` construtível.
    """
    outra = (
        "frac_features_under_30"
        if campo == "frac_features_under_20"
        else ("frac_features_under_20")
    )
    # A irmã acompanha o denominador, para que a única incoerência seja a que se planta.
    coerente = float("nan") if definidas == 0 else 0.62
    with pytest.raises(ValueError, match="se e somente se"):
        _report(**{campo: valor, outra: coerente, "n_features_rsd_defined": definidas})


def test_d452_a_planta_cai_sobre_dado_em_que_as_duas_coisas_diferem() -> None:
    """A âncora: sem ela o par acima passaria por coincidência.

    **As duas quantidades que a guarda liga são calculadas aqui, e afirma-se que divergem** — a
    base tem denominador 97 e fração definida, então o estado plantado acima é de facto outro
    estado, e não o mesmo escrito de outra maneira.
    """
    base = _report()
    assert base.n_features_rsd_defined > 0
    assert not np.isnan(base.frac_features_under_20)
    # E o estado plantado é alcançável pela construção normal — o teste acima não recusa tudo.
    assert (
        _report(
            frac_features_under_20=float("nan"),
            frac_features_under_30=float("nan"),
            n_features_rsd_defined=0,
        ).n_features_rsd_defined
        == 0
    )


# ---------------------------------------------------------------------------
# A bicondicional sobre as FAMÍLIAS do PLS-DA — as duas direções
# ---------------------------------------------------------------------------
#
# **A metade `membro de `computed` e campo a `None`` estava inerte para o multiclasse, e a causa
# era um retorno antecipado.** `_check_colecoes_do_plsda_multiclasse` abria com
# `if self.plsda_multiclass_q2 is None: return` e só depois perguntava por `computed` — e o único
# estado em que aquela direção pode ser violada é exactamente o que o retorno saltava.
#
# **Medido em 17/09/2026, antes do conserto:** `computed` a nomear `plsda_multiclass_q2` e
# `plsda_multiclass_permutation_p`, com os dois campos a `None`, **CONSTRUIU**; o binário
# equivalente **RECUSOU**. A mesma bicondicional com desfechos opostos conforme a família.
#
# **Ninguém plantava esta direção**, e é por isso que o defeito sobreviveu à unidade que escreveu a
# tabela `FAMILIAS_DO_PLSDA`: os testes da família nova plantam meio par, procedência em falta,
# coleções desalinhadas e valor fora de `computed` — os quatro estados em que o campo TEM valor.

_MULTICLASSE_VALIDO: dict[str, object] = {
    "plsda_multiclass_q2": (0.31, 0.12, -0.04, 0.22),
    "plsda_multiclass_permutation_p": (0.0476, 0.2381, 0.7143, 0.0952),
    "plsda_multiclass_groups": ("ATLL", "Asymptomatic", "HAMTSP", "NoHTLV"),
    "plsda_multiclass_n_components": 2,
    "plsda_multiclass_cv_folds": 7,
    "plsda_multiclass_n_permutations": 20,
    "plsda_multiclass_seed": 42,
}
_COMPUTED_DO_MULTICLASSE = frozenset({"plsda_multiclass_q2", "plsda_multiclass_permutation_p"})


@pytest.mark.parametrize(
    "nome",
    sorted({nome for medida, p, _ in FAMILIAS_DO_PLSDA for nome in (medida, p)}),
)
def test_d44_toda_familia_do_plsda_recusa_nome_em_computed_com_campo_none(nome: str) -> None:
    """**O cenário que ninguém plantava, e é ele que prova que a metade nova é mecanismo.**

    Planta: o nome em `computed` e o campo a `None` — *"calculei, e não há valor"* sobre uma
    medição que nunca aconteceu. Antes do conserto isto construía para os dois nomes do
    multiclasse e levantava para os dois do binário.

    **A parametrização é DERIVADA de `FAMILIAS_DO_PLSDA` e não escrita à mão**, que é a metade
    que faltava: uma família nova herda as duas direções em vez de as repetir. Escrever os quatro
    nomes aqui daria um teste que fica verde no dia em que a quinta linha da tabela nascer sem
    guarda — exactamente o que aconteceu à direção oposta.

    **Nenhum valor é passado de propósito.** O estado só existe quando o campo é `None`, e é essa
    a razão de o retorno antecipado o ter escondido.
    """
    with pytest.raises(ValueError, match="está em `computed` e vale `None`"):
        _report(computed=frozenset(REQUIRED_METRIC_FIELDS) | {nome})


def test_d44_o_par_multiclasse_com_valor_fora_de_computed_continua_recusado() -> None:
    """A direção que já existia, afirmada aqui porque a cláusula MUDOU de função.

    Ela vivia em `_check_colecoes_do_plsda_multiclasse` e passou a
    `_check_bicondicional_das_familias_do_plsda`; modificar mecanismo com evidência **invalida a
    evidência**, e o comportamento que a modificação declara preservar é reafirmado por
    teste e não por leitura do diff.
    """
    with pytest.raises(ValueError, match="não está em `computed`"):
        _report(**_MULTICLASSE_VALIDO)


def test_d44_o_multiclasse_declarado_por_inteiro_e_ACEITO() -> None:  # noqa: N802
    """**A âncora.** Sem ela, as duas recusas acima seriam satisfeitas por uma guarda que recusasse
    todo relatório multiclasse — o portão que sempre reprova.

    É também a direção *"deixar de acusar quando não há o que acusar"*: com a planta removida — o
    nome em `computed` **e** o campo com valor — a construção passa.
    """
    relatorio = _report(
        computed=frozenset(REQUIRED_METRIC_FIELDS) | _COMPUTED_DO_MULTICLASSE,
        **_MULTICLASSE_VALIDO,
    )
    assert relatorio.plsda_multiclass_q2 == (0.31, 0.12, -0.04, 0.22)
    assert relatorio.was_computed("plsda_multiclass_permutation_p")


def test_d291_a_planta_do_computed_cai_sobre_dado_em_que_os_desfechos_DIVERGEM() -> None:  # noqa: N802
    """**O par causal, e aqui ele é indispensável e não cerimónia.**

    Antes do conserto os dois lados desta planta davam o MESMO desfecho: com os campos a `None`, o
    relatório construía-se estivessem os nomes em `computed` ou não. Um par causal cujos dois lados
    concordam não prova nada — e era essa a condição do repositório.

    As duas construções diferem **só** pela pertinência plantada, e afirma-se que divergem.
    """
    limpo = _report(computed=frozenset(REQUIRED_METRIC_FIELDS))
    assert limpo.plsda_multiclass_q2 is None
    assert not limpo.was_computed("plsda_multiclass_q2"), (
        "o lado sem planta já nomeia o campo em `computed`: a planta não distingue estado nenhum"
    )

    with pytest.raises(ValueError):
        _report(computed=frozenset(REQUIRED_METRIC_FIELDS) | {"plsda_multiclass_q2"})


def test_d44_as_duas_familias_do_plsda_respondem_IGUAL_a_mesma_planta() -> None:  # noqa: N802
    """**A assimetria era o defeito, então a simetria é o oráculo** — e não a mensagem de cada uma.

    A mesma planta — os dois nomes de uma família em `computed`, os dois campos a `None` — sobre o
    binário e sobre o multiclasse. Antes do conserto, `ValueError` de um lado e construção do
    outro. É a metade de DIREÇÃO: a bicondicional é lida por dois
    mecanismos, e satisfazer um deles deixava o outro a passar.
    """
    desfechos: dict[str, str] = {}
    for medida, pvalue, _procedencia in FAMILIAS_DO_PLSDA:
        try:
            _report(computed=frozenset(REQUIRED_METRIC_FIELDS) | {medida, pvalue})
        except ValueError:
            desfechos[medida] = "recusou"
        else:
            desfechos[medida] = "construiu"

    assert set(desfechos.values()) == {"recusou"}, (
        f"famílias com desfechos diferentes para a mesma planta: {desfechos}. A regra 6 do README "
        "não pode valer numa família e não na outra"
    )
