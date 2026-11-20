"""Variância explicada contra `sklearn`, e o par que separa as duas PCAs.

**O teste que mais importa aqui não é o critério; é o par.** O critério compara com o
`sklearn` e passaria com qualquer conta correta. O que quase deu defeito foi outra coisa: reusar
`metrics/structure.py::_explained_variance_ratio`, cujo atalho `total = n_samples * n_features`
vale **só** sob autoescalonamento com `ddof=0`.

Dois testes afirmam a relação verdadeira: as duas contas **concordam** sobre entrada
autoescalonada, e **divergem** fora dela. Sem a segunda metade, alguém unifica as duas em nome
de remover duplicação e produz número errado em silêncio sob `scaling="none"`.
"""

import numpy as np
import pytest
import sklearn.decomposition

from refino.domain.matrix import RefinoMatrix
from refino.domain.metadata import SampleMetadata
from refino.metrics.structure import _explained_variance_ratio as atalho_da_fase_2
from refino.statistics.pca import pca

pytestmark = pytest.mark.property

TOLERANCIA = 1e-10


def _matriz(valores: np.ndarray) -> RefinoMatrix:
    n = valores.shape[0]
    amostras = tuple(
        SampleMetadata(
            name=f"S{i}",
            injection_order=i + 1,
            sample_type="qc" if i < 2 else "sample",
            group="QC" if i < 2 else "G",
        )
        for i in range(n)
    )
    return RefinoMatrix(
        values=valores,
        feature_names=tuple(f"f{j}" for j in range(valores.shape[1])),
        samples=amostras,
        mode="sintetico",
    )


def _autoescalona(valores: np.ndarray) -> np.ndarray:
    centrada = valores - valores.mean(axis=0, keepdims=True)
    return centrada / valores.std(axis=0, ddof=0, keepdims=True)


def test_ca_6_5_variancia_explicada_bate_com_sklearn() -> None:
    """Dentro de 1e-10, sobre entradas de escalonamentos diferentes.

    Varre quatro entradas de propósito — crua, autoescalonada, centrada só, e com amplitude
    muito diferente entre colunas. A conta geral tem de valer nas quatro; o atalho
    valeria só na segunda.
    """
    gerador = np.random.default_rng(7)
    crua = gerador.lognormal(10.0, 1.5, size=(20, 8))
    entradas = {
        "crua": crua,
        "autoescalonada": _autoescalona(crua),
        "centrada": crua - crua.mean(axis=0, keepdims=True),
        "amplitude desigual": crua * np.array([1.0, 1e3, 1.0, 1e-3, 1.0, 10.0, 1.0, 1.0]),
    }

    for nome, valores in entradas.items():
        nosso = pca(_matriz(valores), n_components=5)
        deles = sklearn.decomposition.PCA(n_components=5).fit(valores)
        erro = float(
            np.max(
                np.abs(np.array(nosso.explained_variance_ratio) - deles.explained_variance_ratio_)
            )
        )
        assert erro <= TOLERANCIA, (
            f"entrada {nome!r}: erro {erro:.3e} contra `sklearn.decomposition.PCA` excede "
            f"{TOLERANCIA:.0e}"
        )


def test_a_conta_da_fase_2_concorda_sobre_entrada_autoescalonada() -> None:
    """Metade 1 do par: sob autoescalonamento com `ddof=0`, as duas coincidem.

    É o que torna o atalho correto **lá**, e o que faz a coexistência ser justificada
    em vez de duplicação por descuido.
    """
    gerador = np.random.default_rng(11)
    valores = _autoescalona(gerador.lognormal(9.0, 1.0, size=(24, 6)))
    matriz = _matriz(valores)

    nosso = pca(matriz, n_components=3)
    do_atalho = atalho_da_fase_2(nosso.scores, matriz, nosso.n_features_used)

    assert np.allclose(np.array(nosso.explained_variance_ratio), np.array(do_atalho), atol=1e-12)


def test_a_conta_da_fase_2_diverge_fora_do_autoescalonamento() -> None:
    """**Metade 2, e é a que impede a unificação.** Sem autoescalonar, o atalho erra.

    Se este teste passasse a falhar, o atalho teria se tornado geral e as duas contas seriam
    de fato a mesma — o que seria mudança de comportamento em `metrics/structure.py`, não
    conveniência. Enquanto ele reprova, `pca` **não pode** delegar para lá.
    """
    gerador = np.random.default_rng(13)
    # Entrada crua: colunas com variância muito diferente de 1, que é a condição do atalho.
    valores = gerador.lognormal(10.0, 1.5, size=(20, 6))
    matriz = _matriz(valores)

    nosso = pca(matriz, n_components=3)
    do_atalho = atalho_da_fase_2(nosso.scores, matriz, nosso.n_features_used)

    assert not np.allclose(
        np.array(nosso.explained_variance_ratio), np.array(do_atalho), atol=1e-6
    ), (
        "o atalho da Fase 2 coincidiu com a conta geral sobre entrada NÃO autoescalonada; "
        "então a condição que a docstring dele declara deixou de morder, e o teste de "
        "concordância acima virou vácuo"
    )


def test_a_soma_sobre_todos_os_componentes_e_um() -> None:
    """Propriedade da fração: com o rank completo, a soma fecha em 1."""
    gerador = np.random.default_rng(17)
    valores = gerador.normal(0.0, 1.0, size=(12, 5))
    resultado = pca(_matriz(valores), n_components=5)
    assert sum(resultado.explained_variance_ratio) == pytest.approx(1.0, abs=1e-12)


def test_ca_6_7_orientacao_canonica() -> None:
    """Em `pca`: `X` e `-X` dão loadings de mesmo sinal."""
    gerador = np.random.default_rng(19)
    valores = gerador.normal(0.0, 1.0, size=(16, 6))

    direto = pca(_matriz(valores), n_components=3)
    negado = pca(_matriz(-valores), n_components=3)

    assert np.allclose(np.sign(direto.loadings), np.sign(negado.loadings))


def test_coluna_de_variancia_zero_e_descartada_contada_e_digerida() -> None:
    """As três contagens, e o digest que distingue conjunto de contagem."""
    gerador = np.random.default_rng(23)
    valores = gerador.normal(0.0, 1.0, size=(10, 5))
    valores[:, 2] = 4.2  # constante

    resultado = pca(_matriz(valores), n_components=3)

    assert resultado.n_features_used == 4
    assert resultado.n_features_dropped_zero_variance == 1
    assert resultado.n_features_dropped_non_finite == 0

    # Mesmo NÚMERO de colunas retidas, conjunto diferente: o digest tem de diferir.
    outros = gerador.normal(0.0, 1.0, size=(10, 5))
    outros[:, 3] = 4.2
    assert pca(_matriz(outros), n_components=3).column_digest != resultado.column_digest


def test_coluna_nao_finita_e_descartada_e_contada() -> None:
    """Não finito sai da conta em vez de propagar `nan` para os componentes."""
    gerador = np.random.default_rng(29)
    valores = gerador.normal(0.0, 1.0, size=(10, 4))
    valores[3, 1] = np.nan

    resultado = pca(_matriz(valores), n_components=3)

    assert resultado.n_features_dropped_non_finite == 1
    assert resultado.n_features_used == 3
    assert np.isfinite(resultado.scores).all()


def test_n_components_acima_do_rank_levanta_nomeando_as_duas_contagens() -> None:
    gerador = np.random.default_rng(31)
    with pytest.raises(ValueError, match="só admite"):
        pca(_matriz(gerador.normal(0.0, 1.0, size=(4, 3))), n_components=5)


def test_matriz_sem_feature_utilizavel_levanta() -> None:
    """Levanta em vez de produzir componentes de nada com aparência de medida."""
    with pytest.raises(ValueError, match="nenhuma das"):
        pca(_matriz(np.ones((5, 3))), n_components=1)


def test_determinismo_bit_a_bit() -> None:
    gerador = np.random.default_rng(37)
    matriz = _matriz(gerador.normal(0.0, 1.0, size=(14, 6)))
    a, b = pca(matriz, n_components=4), pca(matriz, n_components=4)
    assert np.array_equal(a.scores, b.scores)
    assert a.explained_variance_ratio == b.explained_variance_ratio
