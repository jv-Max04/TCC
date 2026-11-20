"""A floresta do SERRF com a contagem do `ranger` — a opção que o código traz.

**O que este arquivo prova, e o que não.** Prova que a opção existe, que é a MESMA floresta que a
Mediu-se (`scripts/floresta_ranger_emulada.py`), que ela conserta o mecanismo — a raiz
divide com sete QCs —, que é determinística, e que é o padrão hoje, com a floresta antiga
viva pelo nome. **Não** prova que ela corrige melhor: isso pede a comparação contra a verdade e
contra o R.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import joblib
import numpy as np
import pytest
from sklearn.ensemble import RandomForestRegressor

from refino.correction._floresta_ranger import FlorestaRanger
from refino.correction._serrf_steps import (
    RANGER_MAX_FEATURES,
    RANGER_MIN_SAMPLES_SPLIT,
    RANGER_N_ESTIMATORS,
)
from refino.correction.serrf import FLORESTA_PADRAO, SerrfCorrector
from refino.domain.matrix import RefinoMatrix
from refino.domain.metadata import SampleMetadata

RAIZ = Path(__file__).resolve().parents[2]
N_QC = 7


def _script_medido() -> type:
    spec = importlib.util.spec_from_file_location(
        "floresta_ranger_emulada", RAIZ / "scripts" / "floresta_ranger_emulada.py"
    )
    assert spec is not None and spec.loader is not None
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    classe: type = modulo.FlorestaRangerEmulada
    return classe


def _treino(seed: int = 5) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sete linhas de treino e dez preditores — o regime do passo 4 com sete QCs."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((N_QC, 10))
    y = x[:, 0] * 2.0 + rng.standard_normal(N_QC) * 0.5
    teste = rng.standard_normal((25, 10))
    return x, y, teste


def _matriz(n_samples: int = 30, n_features: int = 14, seed: int = 3) -> RefinoMatrix:
    """Sete QCs espalhados e deriva leve, como `_realistic` de `test_serrf_corrector.py`."""
    rng = np.random.default_rng(seed)
    qc_at = tuple(round(i) for i in np.linspace(0, n_samples - 1, N_QC))
    drift = 1.0 + 0.004 * np.arange(n_samples)
    base = rng.uniform(1e3, 1e5, size=n_features)
    values = (
        base[None, :] * drift[:, None] * (1.0 + 0.06 * rng.standard_normal((n_samples, n_features)))
    )
    samples = tuple(
        SampleMetadata(
            name=f"S{i + 1}",
            injection_order=i + 1,
            sample_type="qc" if i in qc_at else "sample",
            group="QC" if i in qc_at else "NoHTLV",
            batch="A",
        )
        for i in range(n_samples)
    )
    return RefinoMatrix(
        values=values,
        feature_names=tuple(f"F{i + 1}" for i in range(n_features)),
        samples=samples,
        mode="tiny",
    )


def _profundidade(matriz: RefinoMatrix) -> float:
    nota = matriz.provenance[-1]
    campo = next(p for p in nota.split(",") if p.startswith("mean_tree_depth="))
    return float(campo.split("=", 1)[1].rstrip(")"))


def test_a_floresta_do_codigo_e_a_do_script_medido_bit_a_bit() -> None:
    """**É o que faz os números da YO3 e valerem para o código.**

    Os dois consomem o gerador na mesma ordem — índices, depois a semente de cada árvore —, e uma
    troca de ordem mudaria todas as árvores sem levantar erro nenhum.
    """
    x, y, teste = _treino()
    emulada = _script_medido()
    parametros = {
        "n_estimators": 60,
        "max_features": RANGER_MAX_FEATURES,
        "min_samples_split": RANGER_MIN_SAMPLES_SPLIT,
        "random_state": 1234,
    }
    nossa = FlorestaRanger(**parametros).fit(x, y)  # type: ignore[arg-type]
    medida = emulada(**parametros).fit(x, y)

    assert np.array_equal(nossa.predict(teste), medida.predict(teste))
    assert [a.get_depth() for a in nossa.estimators_] == [a.get_depth() for a in medida.estimators_]


def test_com_sete_linhas_a_raiz_divide_na_ranger_e_trava_na_do_scikit() -> None:
    """O mecanismo, nas DUAS direções: o `scikit-learn` deixa tocos, a contagem do `ranger` não.

    Sem o lado do `scikit-learn`, «nenhuma árvore sem divisão» passaria também com dados em que
    nenhuma floresta trava — e não provaria que a contagem é o que muda.
    """
    x, y, _ = _treino()
    ranger = FlorestaRanger(
        n_estimators=RANGER_N_ESTIMATORS,
        max_features=RANGER_MAX_FEATURES,
        min_samples_split=RANGER_MIN_SAMPLES_SPLIT,
        random_state=7,
    ).fit(x, y)
    scikit = RandomForestRegressor(
        n_estimators=RANGER_N_ESTIMATORS,
        max_features=RANGER_MAX_FEATURES,
        min_samples_split=RANGER_MIN_SAMPLES_SPLIT,
        random_state=7,
    ).fit(x, y)

    tocos_ranger = sum(a.get_depth() == 0 for a in ranger.estimators_)
    tocos_scikit = sum(a.get_depth() == 0 for a in scikit.estimators_)
    assert len(ranger.estimators_) == RANGER_N_ESTIMATORS
    assert tocos_ranger == 0, f"{tocos_ranger} árvores sem divisão com a contagem do ranger"
    assert tocos_scikit >= RANGER_N_ESTIMATORS // 5, (
        f"só {tocos_scikit} tocos no scikit-learn: o dado do teste não reproduz o mecanismo de "
        "AR160 (~41%), e o lado de cima passaria por vacuidade"
    )


def test_predict_antes_de_fit_e_recusado() -> None:
    floresta = FlorestaRanger(
        n_estimators=3, max_features="sqrt", min_samples_split=5, random_state=1
    )
    with pytest.raises(RuntimeError, match="antes de fit"):
        floresta.predict(np.zeros((2, 3)))


def test_o_padrao_e_o_ranger_e_pedir_none_e_pedir_o_padrao() -> None:
    """O padrão é o `ranger` hoje, e `None` resolve para a constante do módulo."""
    assert FLORESTA_PADRAO == "ranger"
    matriz = _matriz()
    padrao = SerrfCorrector().fit_transform(matriz, seed=42)
    explicito = SerrfCorrector(floresta="ranger").fit_transform(matriz, seed=42)
    assert SerrfCorrector().floresta == "ranger"
    assert np.array_equal(padrao.matrix.values, explicito.matrix.values)
    assert padrao.matrix.provenance == explicito.matrix.provenance


def test_a_opcao_ranger_e_viva_e_deterministica() -> None:
    """Par causal: a mesma matriz e a mesma semente, e só a floresta muda.

    Determinística — duas execuções iguais — e VIVA: os valores mudam e a profundidade média sobe,
    porque a raiz deixa de travar. Sem a segunda metade, uma opção que não chegasse à floresta
    passaria. **O `ranger` é o padrão**, então o outro lado do par é a floresta antiga
    pedida pelo nome — que é também a prova de que ela continua viva no código.
    """
    matriz = _matriz()
    a = SerrfCorrector(floresta="ranger").fit_transform(matriz, seed=42)
    b = SerrfCorrector(floresta="ranger").fit_transform(matriz, seed=42)
    antiga = SerrfCorrector(floresta="scikit_learn").fit_transform(matriz, seed=42)

    assert np.array_equal(a.matrix.values, b.matrix.values)
    assert a.matrix.provenance == b.matrix.provenance
    assert "floresta=ranger" in a.matrix.provenance[-1]
    assert "floresta=scikit_learn" in antiga.matrix.provenance[-1]
    assert not np.array_equal(a.matrix.values, antiga.matrix.values), (
        "a opção ranger devolveu a mesma matriz que a floresta antiga: ela não chegou à floresta"
    )
    assert _profundidade(a.matrix) > _profundidade(antiga.matrix)


def test_a_opcao_ranger_e_bit_identica_serial_e_paralela() -> None:
    """O mesmo par causal de `test_forest_speedup.py`, para a floresta nova."""
    if joblib.effective_n_jobs(-1) <= 1:
        pytest.skip(
            "paralelismo indisponível: `effective_n_jobs(-1)` devolve 1 — com "
            "`LOKY_MAX_CPU_COUNT=1` este par causal compararia 1 contra 1 e passaria por vacuidade"
        )
    matriz = _matriz()
    serial = SerrfCorrector(n_jobs=1, floresta="ranger").fit_transform(matriz, seed=42)
    paralela = SerrfCorrector(n_jobs=-1, floresta="ranger").fit_transform(matriz, seed=42)
    assert np.array_equal(serial.matrix.values, paralela.matrix.values)
    assert serial.matrix.provenance == paralela.matrix.provenance


def test_floresta_desconhecida_e_recusada_na_construcao() -> None:
    """A guarda em runtime: o valor errado não cai no `else` e corre a outra floresta."""
    with pytest.raises(ValueError, match="floresta desconhecida"):
        SerrfCorrector(floresta="Ranger")  # type: ignore[arg-type]
    assert SerrfCorrector(floresta="ranger").floresta == "ranger"


def test_a_floresta_separa_a_identidade_de_reuso() -> None:
    """A floresta muda valores, então duas florestas não podem partilhar a chave de reuso.

    **E o padrão pedido sem nome tem a chave do `ranger` **: reuso entre uma execução
    que pediu o padrão e outra que pediu o `ranger` pelo nome é legítimo, porque é a mesma conta.
    """
    ranger = SerrfCorrector(floresta="ranger").identidade()
    assert ranger != SerrfCorrector(floresta="scikit_learn").identidade()
    assert "floresta=ranger" in ranger
    assert SerrfCorrector().identidade() == ranger
