"""`refino corrigir` nos quatro modos reais, critério **parcial**.

**O que este arquivo pode cobrar, e é menos do que o nome do critério sugere.** O
O critério pede que o CSV da CLI tenha RSD mediano nos QCs dentro da **tolerância
derivada do critério** — que não existe ainda. Então o que se verifica aqui é
outra coisa, e nomeá-la evita que "verde" signifique mais do que foi medido:

> **a CLI não introduz erro próprio.** O número que ela escreve é o que a
> biblioteca calcula, a menos do round-trip textual.

Se a correção está **certa** é o critério para o `batch_loess` — golden por
feature, com folga derivada — e os critérios de concordância para o SERRF. Nenhum dos
dois é este arquivo. Um teste de CLI que afirmasse "o RSD caiu, logo funciona"
mediria o LOESS pelo estimador que ele zera por construção (achado 4.3).

`--method serrf` não aparece aqui e a ausência é por desenho: ele é **recusado na
fronteira** por enquanto, e o teste dessa recusa é unitário. Não há `skip` por método
indisponível, porque não há o que pular — não existe execução para comparar. `skip`
descreveria uma comparação impedida por dado ausente, que não é o caso.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from refino.cli import EXIT_OK, main, median_qc_rsd
from refino.correction.loess import LoessCorrector
from refino.correction.none import NoneCorrector
from refino.io.readers import load_matrix, read_toserrf
from tests.conftest import measurements_path, sample_list_path

pytestmark = pytest.mark.integration

MODES = ("pos_hilic", "neg_hilic", "pos_c18", "neg_c18")

# Round-trip textual da especificação. `write_toserrf` grava 15 dígitos significativos, e
# esta é a tolerância que o formato declara — não é folga sobre quantidade
# estocástica, e não pode ser afrouxada para acomodar divergência de correção.
ROUND_TRIP_RTOL = 1e-9


def _run(
    mode: str, method: str, tmp_path: Path, measurements: Path, sample_list: Path
) -> dict[str, object]:
    output = tmp_path / f"{mode}_{method}.csv"
    log = tmp_path / f"{mode}_{method}.json"
    code = main(
        [
            "corrigir",
            "--measurements",
            str(measurements),
            "--sample-list",
            str(sample_list),
            "--mode",
            mode,
            "--method",
            method,
            "--output",
            str(output),
            "--log",
            str(log),
        ]
    )
    assert code == EXIT_OK, f"{mode}/{method} não terminou em 0"
    payload = json.loads(log.read_text(encoding="utf-8"))
    payload["_output"] = output
    return payload


@pytest.mark.parametrize("mode", MODES)
def test_ca_3_9_parcial_none_reproduz_a_biblioteca(mode: str, tmp_path: Path) -> None:
    """`none` nos quatro modos: o CSV escrito tem o RSD que a biblioteca calcula.

    Com `none` a comparação é a mais estreita possível — o corretor não muda valor
    nenhum, então qualquer diferença é da CLI: leitura, orientação, ordenação por
    injeção ou escrita. É o teste que isola a fronteira de tudo o mais.
    """
    measurements, samples = measurements_path(mode), sample_list_path(mode)
    payload = _run(mode, "none", tmp_path, measurements, samples)

    matrix = load_matrix(measurements, samples, mode)
    esperado = median_qc_rsd(NoneCorrector().fit_transform(matrix).matrix, "outlier_removed")

    doc = read_toserrf(payload["_output"])
    assert doc.values.shape == (matrix.n_features, matrix.n_samples)
    assert np.allclose(doc.values, matrix.values.T, rtol=ROUND_TRIP_RTOL, atol=0, equal_nan=True)
    assert payload["median_qc_rsd_pct"]["after"] == pytest.approx(esperado, rel=ROUND_TRIP_RTOL)


@pytest.mark.slow
@pytest.mark.parametrize("mode", MODES)
def test_ca_3_9_parcial_loess_reproduz_a_biblioteca(mode: str, tmp_path: Path) -> None:
    """`loess` nos quatro modos, e a comparação é contra a biblioteca, não contra o R.

    Marcado `slow` porque corrige as 6.035 features do `pos_hilic` de verdade — o
    orçamento é do alvo `test-all`, não do CI de commit.

    **O RSD reportado NÃO é evidência de que a correção é boa.** Ele é in-sample, e
    o LOESS o zera por construção; a asserção é de igualdade contra a própria
    biblioteca, que é o que a fronteira pode responder.
    """
    measurements, samples = measurements_path(mode), sample_list_path(mode)
    payload = _run(mode, "loess", tmp_path, measurements, samples)

    matrix = load_matrix(measurements, samples, mode)
    result = LoessCorrector().fit_transform(matrix, seed=42)
    esperado = median_qc_rsd(result.matrix, "outlier_removed")

    assert payload["median_qc_rsd_pct"]["after"] == pytest.approx(esperado, rel=ROUND_TRIP_RTOL)
    assert payload["uncorrected"]["digest"] == result.uncorrected_digest()
    assert payload["uncorrected"]["by_cause"] == {
        causa: contagem for causa, contagem in result.counts_by_cause()
    }

    doc = read_toserrf(payload["_output"])
    finito = np.isfinite(doc.values) & np.isfinite(result.matrix.values.T)
    assert np.allclose(
        doc.values[finito], result.matrix.values.T[finito], rtol=ROUND_TRIP_RTOL, atol=0
    )


@pytest.mark.parametrize("mode", ("pos_hilic",))
def test_a_orientacao_padrao_le_o_arquivo_do_laboratorio_sem_transpor(
    mode: str, tmp_path: Path
) -> None:
    """No dado real: o padrão é a orientação que o Progenesis QI exporta.

    Sem esta asserção, `--orientation` poderia ter padrão trocado e todos os testes
    unitários continuariam passando — eles usam fixtures que o projeto escreveu, na
    orientação que o projeto escolheu.
    """
    measurements, samples = measurements_path(mode), sample_list_path(mode)
    payload = _run(mode, "none", tmp_path, measurements, samples)
    assert payload["spec"]["orientation"] == "features-in-rows"
    assert payload["shape"]["n_qc"] == 7  # observação sobre estes quatro arquivos
    assert payload["shape"]["n_samples"] == 68
