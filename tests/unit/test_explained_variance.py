"""`explained_variance_ratio` - o campo que o diagnóstico acrescentou a `StructuralMetrics`.

**Comportamento novo recebe as duas direções**. A modificação declara preservar
tudo o mais de `structural_metrics`, e isso é reafirmado pelos 120 testes de estrutura e
PCA que já existiam; o que este arquivo cobre é só o campo novo.

A derivação não usa os valores singulares, e por isso ela **pode** estar errada de um
jeito que ainda produz números plausíveis - fração entre 0 e 1, decrescente. Os testes
abaixo checam a identidade algébrica que a sustenta, e não só a plausibilidade.
"""

import numpy as np
import pytest

from refino.metrics.structure import structural_metrics
from tests.conftest import measurements_path, sample_list_path

MODOS = ("pos_hilic", "pos_c18", "neg_hilic", "neg_c18")


def _carrega(modo: str):  # type: ignore[no-untyped-def]
    from refino.io.readers import load_matrix

    return load_matrix(measurements_path(modo), sample_list_path(modo), modo)


def test_a_identidade_algebrica_da_fracao_explicada() -> None:
    """**A checagem que não é de plausibilidade: a soma sobre TODOS os componentes é 1.**

    Com `k = n_samples` componentes, a soma das frações tem de ser exactamente 1, porque
    a matriz autoescalonada tem rank no máximo `n_samples` e os componentes esgotam a
    variância. É a identidade que a derivação usa - variância total igual a
    `n_samples * n_features_used` -, e ela falha se o denominador estiver errado.

    Fração entre 0 e 1 e ordem decrescente passariam com o denominador errado por
    qualquer fator constante; esta não.
    """
    gerador = np.random.default_rng(42)
    valores = gerador.normal(100.0, 20.0, size=(12, 40))
    from refino.domain.matrix import RefinoMatrix
    from refino.domain.metadata import SampleMetadata

    amostras = tuple(
        SampleMetadata(
            name=f"S{i}",
            injection_order=i + 1,
            sample_type="qc" if i % 4 == 0 else "sample",
            group="QC" if i % 4 == 0 else "G",
        )
        for i in range(12)
    )
    matriz = RefinoMatrix(
        values=valores,
        feature_names=tuple(f"f{j}" for j in range(40)),
        samples=amostras,
        mode="sintetico",
    )

    metricas = structural_metrics(matriz, pretransform="none", n_components=12)
    soma = sum(metricas.explained_variance_ratio)
    assert soma == pytest.approx(1.0, abs=1e-10), (
        f"a soma sobre todos os componentes deu {soma}, e tem de ser 1; o denominador "
        "da fração explicada está errado"
    )


def test_o_comprimento_acompanha_n_components() -> None:
    """Tupla e não dois escalares: o número de componentes é parâmetro."""
    matriz = _carrega("neg_c18")
    for k in (1, 2, 3):
        metricas = structural_metrics(matriz, pretransform="log10p1", n_components=k)
        assert len(metricas.explained_variance_ratio) == k


@pytest.mark.parametrize("modo", MODOS)
def test_a_fracao_e_decrescente_e_somavel_nos_quatro_modos(modo: str) -> None:
    """PC1 explica ao menos tanto quanto PC2, e a soma dos dois não passa de 1."""
    metricas = structural_metrics(_carrega(modo), pretransform="log10p1", n_components=2)
    pc1, pc2 = metricas.explained_variance_ratio

    assert 0.0 < pc1 <= 1.0
    assert 0.0 < pc2 <= pc1, "PC2 não pode explicar mais que PC1"
    assert pc1 + pc2 <= 1.0 + 1e-12


def test_matriz_sem_estrutura_espalha_a_variancia() -> None:
    """**A outra direção: ruído puro NÃO concentra variância em PC1.**

    Sem este teste, uma implementação que devolvesse sempre um número alto para PC1
    passaria nos quatro modos reais - onde há estrutura de verdade - e ninguém veria.
    Ruído independente sobre 12 injeções e 40 features espalha, e o teste afirma o
    teto.
    """
    from refino.domain.matrix import RefinoMatrix
    from refino.domain.metadata import SampleMetadata

    gerador = np.random.default_rng(7)
    amostras = tuple(
        SampleMetadata(
            name=f"S{i}",
            injection_order=i + 1,
            sample_type="qc" if i % 4 == 0 else "sample",
            group="QC" if i % 4 == 0 else "G",
        )
        for i in range(12)
    )
    matriz = RefinoMatrix(
        values=gerador.normal(100.0, 20.0, size=(12, 40)),
        feature_names=tuple(f"f{j}" for j in range(40)),
        samples=amostras,
        mode="sintetico",
    )
    pc1 = structural_metrics(matriz, pretransform="none", n_components=2).explained_variance_ratio[
        0
    ]
    assert pc1 < 0.5, f"ruído puro concentrou {pc1:.3f} da variância em PC1"
