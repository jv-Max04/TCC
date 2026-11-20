"""A tubulação dos portões derivados, exercitada com oráculo trivial.

**O que este módulo verifica não é um número, é o encanamento.** A concordância com o oráculo
é avaliada depois, com onze unidades construídas em cima. Se o
maquinário do portão estiver errado lá, a reprovação é **ambígua** entre "o porte
diverge" e "o portão está errado" — duas conclusões que pedem investigações
opostas, e nenhum dado disponível naquele momento as separa.

**Oráculo trivial para maquinário não trivial.** O corretor `none` é oráculo
**exato**: nos quatro modos, a matriz Python e a golden
`normalized by - none.csv` são idênticas célula a célula, com diferença máxima
**zero** sobre 174.896 células no `pos_c18`. Então:

- os dois vetores de RSD são **bit-idênticos**, e `d` fica limitado pela
  auto-divergência do vetor contra si mesmo;
- a mediana de RSD dos dois lados tem de bater **exatamente**.

O que se exercita no caminho é caro e frágil — ler o fixture congelado, conferir
o `sha256`, calcular o vetor de RSD dos dois lados com a **mesma** régua, reduzir
os pares , aplicar a tolerância. A resposta é conhecida
de antemão, então divergência aqui só pode ser do portão.

**Não fecha a concordância do SERRF com o oráculo.** Fecha a
pergunta anterior: o encanamento calcula o que diz calcular.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from refino.correction.none import NoneCorrector
from refino.metrics.agreement import (
    spearman_divergence,
    tolerance_from_dispersions,
)
from refino.metrics.rsd import rsd
from tests.conftest import matrix_from_normalized

pytestmark = pytest.mark.golden

REPO = Path(__file__).resolve().parent.parent.parent
FIXTURE = REPO / "data" / "fixtures" / "r_serrf_divergence.json"
MODES = ("pos_hilic", "pos_c18", "neg_hilic", "neg_c18")

# A régua nomeada nos dois lados do portão. Trocar qualquer um dos dois muda o
# número e inverte ranking (achado 4.3) — por isso viajam com a medição.
RSD_VARIANT = "outlier_removed"


@pytest.fixture(scope="module")
def preregistration() -> dict[str, Any]:
    if not FIXTURE.is_file():
        pytest.skip(
            "r_serrf_divergence.json ausente; a fixture é versionada, restaure-a do repositório"
        )
    return json.loads(FIXTURE.read_text(encoding="utf-8"))  # type: ignore[no-any-return]


def _qc_rsd_vector(matrix: Any) -> np.ndarray:
    """RSD por feature sobre as linhas de QC, in-sample, `outlier_removed`."""
    values = matrix.values[matrix.qc_mask]
    result: np.ndarray = rsd(values, axis=0, variant=RSD_VARIANT)
    return result


# ---------------------------------------------------------------------------
# A tubulação inteira, com resposta conhecida
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", MODES)
def test_os_vetores_de_rsd_do_none_sao_bit_identicos_ao_do_r(mode: str) -> None:
    """**O oráculo trivial.** `none` é exato, então os dois vetores são idênticos.

    O nome diz `bit-idênticos` e não `divergência zero` de propósito: a primeira
    versão afirmava `d == 0` e reprovou no `pos_hilic` com 1,11e-16 — que é a
    aritmética do `spearmanr` sobre 5.801 pontos, não a tubulação. Nome que
    afirma mais do que o teste verifica engana na etiqueta do próprio teste.

    Qualquer diferença nos vetores significa que algum elo do caminho — leitura
    da golden, máscara de QC, variante do RSD — está errado, e é isso que o SERRF
    não conseguiria distinguir de divergência do porte.
    """
    golden = matrix_from_normalized(mode, "none")
    ours = NoneCorrector().fit_transform(golden).matrix

    theirs_rsd = _qc_rsd_vector(golden)
    ours_rsd = _qc_rsd_vector(ours)

    assert np.array_equal(np.isfinite(theirs_rsd), np.isfinite(ours_rsd)), (
        f"{mode}: RSD indefinido em features diferentes — a política de indefinido "
        "difere entre os dois lados, e o `rho` passaria a comparar subconjuntos"
    )

    # **A afirmação exata é sobre os VETORES, não sobre `d`.** `none` é oráculo
    # exato, então os dois vetores de RSD são bit-idênticos — e isso é igualdade
    # de verdade, verificada nos quatro modos.
    assert np.array_equal(ours_rsd, theirs_rsd, equal_nan=True), (
        f"{mode}: os vetores de RSD diferem. `none` não muda valor, então isto é "
        "defeito da tubulação do portão, não do porte."
    )

    divergence = spearman_divergence(ours_rsd, theirs_rsd)
    assert divergence.n_used > 0, f"{mode}: nenhuma feature com RSD definido nos dois lados"

    # **E o resíduo de `d` é limitado pela AUTO-divergência, não por tolerância
    # escolhida.** `1 - spearmanr(a, a)` não é exatamente zero para `n` grande:
    # medido, 1,11e-16 no `pos_hilic` com 5.801 pontos e 0 nos outros três. É a
    # aritmética do `scipy`, e comparar contra ela é a referência derivada que a
    # regra exige — um limiar declarado aqui mediria a soma, não o portão.
    assert divergence.is_at_floor, (
        f"{mode}: d(Python_none, R_none) = {divergence.d:.3e} excede o piso "
        f"{divergence.floor:.3e} do mesmo vetor contra si mesmo, sobre "
        f"{divergence.n_used} features. Com vetores bit-idênticos isso é "
        "impossível por aritmética — é defeito da redução."
    )


@pytest.mark.parametrize("mode", MODES)
def test_a_mediana_do_none_bate_exatamente(mode: str) -> None:
    """O outro lado do portão: a mediana, e também por igualdade.

    Sem esta metade, um portão que calculasse `rho` certo e mediana errada passaria
    — e a mediana é o que o portão compara.
    """
    golden = matrix_from_normalized(mode, "none")
    ours = NoneCorrector().fit_transform(golden).matrix

    theirs = _qc_rsd_vector(golden)
    mine = _qc_rsd_vector(ours)
    theirs_median = float(np.median(theirs[np.isfinite(theirs)]))
    my_median = float(np.median(mine[np.isfinite(mine)]))
    assert my_median == theirs_median, (
        f"{mode}: mediana {my_median} contra {theirs_median} — `none` não muda valor"
    )


# ---------------------------------------------------------------------------
# A aritmética do portão, sem corretor
# ---------------------------------------------------------------------------


def test_a_tolerancia_e_soma_quadratica_e_nao_aritmetica() -> None:
    """A soma aritmética infla **41,4%** quando os dois desvios são parecidos.

    É o teste que reprova a fórmula da v1.4, e a diferença é exatamente a folga
    onde um defeito de porte se esconderia sem reprovar.
    """
    quadratic = tolerance_from_dispersions(1.0, 0.2, 0.2)
    arithmetic = 1.0 * (0.2 + 0.2)
    assert quadratic == pytest.approx(0.2 * np.sqrt(2))
    assert arithmetic / quadratic == pytest.approx(np.sqrt(2), rel=1e-12)
    assert (arithmetic / quadratic - 1.0) == pytest.approx(0.4142, abs=1e-4)


def test_a_tolerancia_cresce_com_k_e_com_cada_desvio() -> None:
    """Monotonicidade nos três argumentos — o mínimo que a fórmula tem de ter."""
    base = tolerance_from_dispersions(3.0, 0.1, 0.2)
    assert tolerance_from_dispersions(4.0, 0.1, 0.2) > base
    assert tolerance_from_dispersions(3.0, 0.3, 0.2) > base
    assert tolerance_from_dispersions(3.0, 0.1, 0.5) > base


def test_o_k_vem_do_fixture_congelado_e_nao_de_constante_local(
    preregistration: dict[str, Any],
) -> None:
    """`k` é pré-registrado, e ler de outro lugar dissolveria o pré-registro.

    O `sha256` do arquivo entrou no tronco antes de existir corretor Python; um
    `k` lido de constante no código do portão tornaria isso decorativo.
    """
    registered = preregistration["preregistration"]
    assert registered["k_ca_3_1"] == 3.0
    assert registered["k_ca_3_2"] == 3.0
    assert "sqrt" in registered["tolerance_formula"]


@pytest.mark.parametrize("mode", MODES)
def test_o_denominador_do_ca_3_1_e_positivo_e_congelado(
    preregistration: dict[str, Any], mode: str
) -> None:
    """A razão do portão precisa de denominador não nulo, e ele vem do fixture.

    Denominador zero tornaria a tolerância derivada **indefinida** em vez de
    estrita — o desfecho que o teste do SERRF em `test_r_reproducibility.py`
    protege pela outra ponta.
    """
    entry = next(m for m in preregistration["modes"] if m["mode"] == mode)
    denominator = entry["serrf"]["d_r1_r2_for_ca_3_1"]
    assert denominator > 0.0, f"{mode}: denominador do [CA-3.1] é {denominator}"
    budget = preregistration["preregistration"]["k_ca_3_1"] * denominator
    assert budget > denominator, "o orçamento tem de ser mais frouxo que o denominador"


def test_o_portao_reprova_quando_excedido() -> None:
    """**O teste que prova que o portão consegue reprovar**.

    Sem ele a aritmética estaria verificada e o **uso** dela não: um portão que
    calculasse a tolerância certa e comparasse na direção errada passaria em tudo
    o mais deste módulo.
    """
    tolerance = tolerance_from_dispersions(3.0, 0.1, 0.1)

    within = 0.4
    beyond = 0.5
    assert within <= tolerance, "o caso que deve passar não passa; o cenário perdeu o dente"
    assert not (beyond <= tolerance), "o caso que deve reprovar passou"
