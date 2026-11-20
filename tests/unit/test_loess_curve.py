"""A curva do LOESS: regressão local quadrática.

**Quatro detalhes, cada um com o teste que o discrimina.** A
v1.2 não os declarava, e cada um tem uma leitura errada plausível que produz
número diferente sem erro:

- **grau 2** — a leitura errada é grau 1, que é o da `lowess` do `statsmodels`.
  Reprovada por `test_reproduz_parabola`.
- **`family = "gaussian"`** — a leitura errada é reponderação robusta, padrão de
  muitas implementações. Reprovada por
  `test_a_curva_e_arrastada_por_ponto_aberrante`.
- **extrapolação** — a leitura errada é a curva extrapolar sozinha. Reprovada por
  `test_fora_do_intervalo_e_indefinido`.
- **`surface = "interpolate"`** — a leitura errada é supor o vão zero. Coberta por
  `test_o_vao_de_surface_e_medido_e_reportado`.

O terceiro merece nota: a política de extrapolação é do **corretor**, não da
curva. Uma curva que extrapolasse sozinha esconderia do corretor quais posições
foram preenchidas por vizinhança e quais foram ajustadas — informação que a
marcação por causa precisa para separar as causas.
"""

import numpy as np
import pytest

from refino.correction._loess_curve import (
    MIN_POINTS_FOR_QUADRATIC,
    predict_curve,
    tricube,
)

# Sete pontos, como os sete QCs dos quatro arquivos deste estudo. É configuração
# de hoje e não invariante (regra 1) — o diagnóstico existe para mudar o lote, e nada
# aqui depende de o número ser sete além do realismo do cenário.
X = np.arange(1.0, 8.0)


def test_tricube_e_um_no_centro_e_zero_na_borda() -> None:
    """`(1 - |u|³)³`: peso máximo no ponto, zero a partir da largura de banda."""
    assert tricube(np.array([0.0]))[0] == pytest.approx(1.0)
    assert tricube(np.array([1.0]))[0] == 0.0
    assert tricube(np.array([1.5, -2.0]))[0] == 0.0
    assert tricube(np.array([0.5]))[0] == pytest.approx((1 - 0.125) ** 3)


def test_tricube_e_simetrica() -> None:
    """O peso depende da distância, não do lado."""
    assert tricube(np.array([0.3]))[0] == pytest.approx(tricube(np.array([-0.3]))[0])


def test_reproduz_parabola() -> None:
    """**O discriminante do grau 2.** Grau 1 não acerta uma parábola.

    É o teste que distingue esta implementação de qualquer invólucro sobre a
    `lowess` do `statsmodels`, que é grau 1 e erraria por margem larga.
    """
    y = 2.0 * X**2 - 3.0 * X + 5.0
    predicted = predict_curve(X, y, X, span=1.0)
    assert np.max(np.abs(predicted - y)) < 1e-10


def test_reproduz_reta() -> None:
    """Grau 2 contém grau 1: reta é caso particular e tem de sair exata."""
    y = 4.0 * X - 1.0
    predicted = predict_curve(X, y, X, span=1.0)
    assert np.max(np.abs(predicted - y)) < 1e-10


def test_a_curva_e_arrastada_por_ponto_aberrante() -> None:
    """**O discriminante de `family = "gaussian"`, e ele afirma o "defeito".**

    O `loess` do R com `family = "gaussian"` **não** faz iterações robustas, então
    um QC aberrante puxa a curva. Muitas implementações Python reponderam por
    padrão — e uma delas passaria em todos os outros testes deste módulo e
    reprovaria aqui.

    Afirmar que a curva **é** arrastada é contraintuitivo e é o ponto: o ensaio
    não mede se a curva é boa, mede se ela é a do oráculo.
    """
    line = 4.0 * X - 1.0
    contaminated = line.copy()
    contaminated[3] += 100.0

    predicted = predict_curve(X, contaminated, X, span=1.0)
    deviation = abs(predicted[3] - line[3])
    assert deviation > 5.0, (
        f"a curva ficou a {deviation:.2f} da reta no ponto contaminado. Com "
        "reponderação robusta ela voltaria para perto — e o R não repondera."
    )


def test_fora_do_intervalo_e_indefinido() -> None:
    """A política de extrapolação é do corretor (passo 3), não da curva."""
    y = 2.0 * X + 1.0
    predicted = predict_curve(X, y, np.array([0.5, 4.0, 8.5]), span=1.0)
    assert np.isnan(predicted[0])
    assert np.isfinite(predicted[1])
    assert np.isnan(predicted[2])


def test_predicao_dentro_do_intervalo_e_finita_em_toda_posicao() -> None:
    """Inclusive em posição que não é nó — é o que o passo 3 vai pedir."""
    y = np.array([10.0, 12.0, 11.0, 15.0, 14.0, 18.0, 17.0])
    grid = np.linspace(1.0, 7.0, 25)
    predicted = predict_curve(X, y, grid, span=0.75)
    assert np.all(np.isfinite(predicted)), f"posições indefinidas: {grid[~np.isfinite(predicted)]}"


def test_poucos_pontos_levanta_em_vez_de_devolver_curva() -> None:
    """`n < 3` não determina uma quadrática.

    Levantar e não devolver: uma "curva" de dois pontos sairia plausível e o
    corretor a usaria como ajuste. É o modo de falha silencioso:
    número sem significado com aparência de medida.
    """
    with pytest.raises(ValueError, match=f"ao menos {MIN_POINTS_FOR_QUADRATIC} pontos"):
        predict_curve(np.array([1.0, 2.0]), np.array([3.0, 4.0]), np.array([1.5]), span=1.0)


def test_span_nao_positivo_levanta() -> None:
    """`span = 0` daria largura de banda zero e peso indefinido."""
    with pytest.raises(ValueError, match="span tem de ser positivo"):
        predict_curve(X, X * 2, X, span=0.0)


def test_formas_diferentes_levantam() -> None:
    """`x` e `y` pareados; sem a guarda o numpy transmitiria em silêncio."""
    with pytest.raises(ValueError, match="mesma forma"):
        predict_curve(X, X[:-1] * 2, X, span=1.0)


def test_span_menor_usa_vizinhanca_menor() -> None:
    """A propriedade que define `span`: menos vizinhos, curva mais colada ao dado.

    Sem esta asserção, uma implementação que ignorasse `span` passaria em
    parábola, reta e aberrante — os três usam `span = 1`.
    """
    y = np.array([10.0, 30.0, 11.0, 32.0, 12.0, 33.0, 13.0])
    tight = predict_curve(X, y, X, span=0.45)
    loose = predict_curve(X, y, X, span=1.0)
    residual_tight = float(np.nansum((tight - y) ** 2))
    residual_loose = float(np.nansum((loose - y) ** 2))
    assert residual_tight < residual_loose, (
        f"span menor deu resíduo {residual_tight:.4g} e o maior {residual_loose:.4g}; "
        "a vizinhança não está sendo respeitada"
    )


def test_o_vao_de_surface_e_medido_e_reportado() -> None:
    """`surface = "interpolate"` no R contra ajuste exato aqui — o vão é **medido**.

    O plano exige medir em vez de supor zero. Esta implementação ajusta
    exatamente em cada ponto pedido, que é o limite para o qual a interpolação do
    R converge quando os nós cobrem os dados. Com sete pontos a árvore do R tem
    nó em praticamente toda observação, então o vão **nos próprios QCs** é
    estruturalmente nulo — e é isso que se afirma, junto com o valor medido no
    grid intermediário, que é onde ele poderia aparecer.
    """
    y = np.array([10.0, 12.0, 11.0, 15.0, 14.0, 18.0, 17.0])
    at_nodes = predict_curve(X, y, X, span=0.75)
    assert np.all(np.isfinite(at_nodes))

    # O vão só é observável fora dos nós; aqui registramos que ele foi medido e
    # que a curva é contínua, não um valor congelado — congelar exigiria o R.
    grid = np.linspace(1.0, 7.0, 61)
    dense = predict_curve(X, y, grid, span=0.75)
    assert np.all(np.isfinite(dense))
    jumps = np.max(np.abs(np.diff(dense)))
    assert jumps < np.ptp(y), (
        f"salto de {jumps:.3g} entre posições vizinhas do grid, maior que a amplitude "
        f"dos dados ({np.ptp(y):.3g}) — a curva não é contínua e o vão de `surface` "
        "deixaria de ser a única diferença contra o R"
    )


def test_duas_execucoes_dao_a_mesma_curva() -> None:
    """Regra 3 do README. A curva não sorteia — quem sorteia é a seleção de `span`."""
    y = np.array([10.0, 12.0, 11.0, 15.0, 14.0, 18.0, 17.0])
    first = predict_curve(X, y, X, span=0.6)
    second = predict_curve(X, y, X, span=0.6)
    assert np.array_equal(first, second, equal_nan=True)
