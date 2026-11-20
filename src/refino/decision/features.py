"""O vetor de características que a política recebe, e a lista do que ela não pode ver.

A política existe para recomendar um corretor sem rodar a medição fora da amostra dos três, que é
cara. Por isso o vetor só usa o que `diagnose` produz da matriz sozinha. Se entrasse algo que só
existe depois da medição (a margem entre o melhor e o segundo, o rótulo), a resposta já seria
conhecida e a política não serviria para nada. `OBSERVAVEIS` é a lista fechada do que entra e
`PROIBIDAS` a do que não pode entrar. Há teste para as duas, para que um campo novo no
`DefectProfile` não entre em silêncio por nenhum dos lados.
"""

from collections.abc import Sequence

import numpy as np

from refino.diagnosis.policy import LIMIARES_INICIAIS, RSD_MEDIANA_MAX_POR_MODO, DiagnosisPolicy
from refino.diagnosis.profile import diagnose
from refino.domain.diagnosis import DefectProfile
from refino.domain.matrix import RefinoMatrix

__all__ = [
    "FEATURE_NAMES",
    "OBSERVAVEIS",
    "POLITICA_DE_DIAGNOSTICO_DO_CORPUS",
    "PROIBIDAS",
    "feature_matrix",
    "feature_vector",
    "perfil_para_a_politica",
]

# A política de diagnóstico com que o corpus de treino foi diagnosticado. É a do aplicativo:
# `LIMIARES_INICIAIS` mais o limiar de RSD por modo, a mesma que a rota do diagnóstico usa por
# padrão. A confiança e o tipo de defeito do vetor dependem dela, então a árvore tem de receber o
# perfil calculado com a política do treino. Se o corpus for regerado com outra, é aqui que muda.
# `tests/test_perfil_da_politica.py` confere a constante contra o corpus.
POLITICA_DE_DIAGNOSTICO_DO_CORPUS = DiagnosisPolicy(
    **LIMIARES_INICIAIS, rsd_mediana_max_por_modo=RSD_MEDIANA_MAX_POR_MODO
)

# O que a política pode ver: campos que `diagnose` produz da matriz sozinha, sem corrigir nada.
OBSERVAVEIS: tuple[str, ...] = (
    "qc_rsd_median",
    "qc_rsd_p75",
    "zero_fraction",
    "missing_fraction",
    "pc1_injection_correlation",
    "qc_dispersion_ratio",
    "n_features",
    "n_samples",
    "n_qcs",
)

# O que a política não pode ver, porque só existe depois da medição dos corretores. A lista é
# explícita, e não o complemento de `OBSERVAVEIS`: um campo novo no `DefectProfile` fica fora das
# duas e o teste reprova.
PROIBIDAS: tuple[str, ...] = (
    "oracle_margin",
    "oracle_label",
    "leave_one_qc_out_rsd",
    "best_corrector",
    "scores",
)

# `defect_type` entra como indicadoras, em ordem fixa. Na ordem de aparição, um corpus embaralhado
# poria as colunas em posições diferentes sob o mesmo nome.
TIPOS: tuple[str, ...] = ("gradual_drift", "step_change", "erratic", "clean", "unassessable")

FEATURE_NAMES: tuple[str, ...] = (
    *OBSERVAVEIS,
    "diag_confidence",
    "n_suggested_batches",
    "n_changepoints",
    *(f"is_{t}" for t in TIPOS),
)


def feature_vector(profile: DefectProfile) -> np.ndarray:
    """O vetor de características observáveis, na ordem de `FEATURE_NAMES`.

    Não lê `raw_features`, para que nada produzido depois da medição chegue à política.
    `n_features` e `n_samples` entram em log10: variam por um fator de 3 entre os modos, e essa
    diferença é de escala, não de sinal.
    """
    valores: list[float] = []
    for nome in OBSERVAVEIS:
        bruto = float(getattr(profile, nome))
        valores.append(float(np.log10(max(bruto, 1.0))) if nome.startswith("n_") else bruto)
    valores.append(float(profile.confidence))
    # `suggested_batches` é um rótulo por amostra, e não uma lista de lotes: o número de lotes é
    # o de rótulos distintos, e `len()` daria o número de amostras.
    valores.append(float(len(set(profile.suggested_batches))))
    valores.append(float(len(profile.changepoints)))
    valores.extend(1.0 if profile.defect_type == t else 0.0 for t in TIPOS)
    return np.asarray(valores, dtype=np.float64)


def feature_matrix(profiles: Sequence[DefectProfile]) -> np.ndarray:
    """Matriz `(n_perfis, len(FEATURE_NAMES))`.

    Sem perfis, devolve zero linhas com a largura certa, porque quem consome confere `X.shape[1]`.
    """
    if not profiles:
        return np.empty((0, len(FEATURE_NAMES)), dtype=np.float64)
    return np.vstack([feature_vector(p) for p in profiles])


def perfil_para_a_politica(matrix: RefinoMatrix) -> DefectProfile:
    """O perfil que a política aprendida recebe: o diagnóstico com a política do corpus.

    É uma função, e não a constante passada em cada chamada, porque são três os pontos que servem
    a política (a recomendação, o palpite da medição e a delegação) e todos têm de usar a mesma.
    """
    return diagnose(matrix, policy=POLITICA_DE_DIAGNOSTICO_DO_CORPUS)
