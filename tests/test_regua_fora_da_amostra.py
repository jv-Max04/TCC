"""O cruzamento que confere `data/fixtures/regua_fora_da_amostra.json`.

**Por que existe.** Aquele fixture custou 56 execuções do `r_service` — os dois
corretores, sete QCs, quatro modos — e nada o lia. Fixture de medição cara sem leitor é
a condição que faz um número envelhecer sem sintoma.

**O oráculo é o `none`, e ele NÃO é escolha de conveniência.** `none` não ajusta modelo,
então não tem como se beneficiar de ser avaliado nos próprios pontos: in-sample e fora
da amostra **coincidem nele por construção**. Logo a coluna `none` deste fixture tem de
bater com a de `insample_rsd_medians.json`, que é gerado por outro caminho de código, a
partir de outro arquivo, por outra transcrição da fórmula.

Duas medições independentes que têm de dar o mesmo número no único caso em que a
igualdade é obrigatória — e é isso que faz o portão pegar fixture corrompido, remedido
com estimador errado, ou com os modos trocados entre si.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[1]
REGUA = RAIZ / "data" / "fixtures" / "regua_fora_da_amostra.json"
INSAMPLE = RAIZ / "data" / "fixtures" / "insample_rsd_medians.json"
MODOS = ("pos_hilic", "neg_hilic", "pos_c18", "neg_c18")

# O `none` do fora-da-amostra é lido da âncora do arnês; o in-sample vem do gerador
# independente. Duas casas decimais: a âncora é publicada com quatro e o in-sample com
# treze, e o que se afirma é a IGUALDADE da medida, não a do arredondamento.
TOLERANCIA_PP = 0.01


# O que o `skip` diz quando a fixture falta. A régua veio do R e não se regenera neste
# repositório; a mediana in-sample tem gerador local.
COMO_OBTER: dict[str, str] = {
    REGUA.name: "a fixture é versionada, restaure-a do repositório",
    INSAMPLE.name: "gere por scripts/gen_insample_medians.py",
}


def _carrega(caminho: Path) -> dict:
    """Lê uma fixture de oráculo; se ela falta, pula dizendo como obtê-la.

    As duas frases casam o regime `frozen_fixture_absent` de `tests/skip_policy.py`.
    """
    if not caminho.exists():  # pragma: no cover - o resolvedor do conftest cobre a ausência
        pytest.skip(f"{caminho.name} ausente; {COMO_OBTER[caminho.name]}")
    return json.loads(caminho.read_text(encoding="utf-8"))


def _le_registro(caminho: Path) -> dict:
    """Lê um REGISTRO DE EXPERIMENTO versionado. **Não pula, e a diferença é de natureza.**

    `serrf_min_samples_split.json` é rastreado no git e não tem gerador — nenhum script do
    repositório o escreve, e a procedência dele não está registrada. Então a ausência dele não é
    condição externa: é **checkout quebrado**, e `skip` ali seria a forma que a política de `skip`
    ensina a ler como legítima escondendo uma quebra real.

    Deixar cair no `FileNotFoundError` é o desfecho certo — alto, e nomeando o arquivo.
    """
    return json.loads(caminho.read_text(encoding="utf-8"))


def test_o_fixture_traz_os_quatro_modos() -> None:
    """Asserção de PRESENÇA antes de qualquer comparação: dicionário vazio passaria vácuo."""
    modos = _carrega(REGUA)["modos"]
    assert set(modos) == set(MODOS), f"esperava os quatro modos, veio {sorted(modos)}"
    for modo, celula in modos.items():
        for campo in ("none_ancora_pct", "r_serrf_pct", "r_loess_pct", "nosso_serrf_pct"):
            assert celula[campo] > 0, f"{modo}.{campo} não é positivo"


@pytest.mark.parametrize("modo", MODOS)
def test_o_none_bate_com_a_medicao_in_sample(modo: str) -> None:
    """`none` é identidade: as duas réguas coincidem nele, e nas outras não têm de coincidir."""
    ancora = _carrega(REGUA)["modos"][modo]["none_ancora_pct"]
    celulas = _carrega(INSAMPLE)["cells"]
    escolhida = [c for c in celulas if c["mode"] == modo and c["method"] == "none"]
    assert len(escolhida) == 1, f"esperava uma célula none/{modo}, vieram {len(escolhida)}"
    esperado = escolhida[0]["outlier_removed"]["median_rsd_pct"]
    assert abs(ancora - esperado) <= TOLERANCIA_PP, (
        f"{modo}: o `none` fora da amostra dá {ancora:.4f}% e o in-sample dá {esperado:.4f}%. "
        "Eles TÊM de coincidir — `none` não ajusta modelo. Divergir significa que um dos dois "
        "fixtures foi remedido com outro estimador, outra variante, ou com os modos trocados."
    )


def test_o_veredito_do_experimento_da_floresta_esta_registrado() -> None:
    """O fixture da floresta carrega a REFUTAÇÃO, e não só os números que a motivaram."""
    dados = _le_registro(RAIZ / "data" / "fixtures" / "serrf_min_samples_split.json")
    assert "NAO ADOTAR" in dados["veredito"], "o veredito deixou de ser a refutação"
    pareado = dados["3_a_refutacao_contra_a_verdade"]["pareado_2_menos_5"]
    assert pareado["fracao_piorada"]["media"] > 0, "a fração piorada deixou de subir"
    assert pareado["fracao_piorada"]["todas_no_mesmo_sentido"] is True
    assert pareado["erro"]["todas_no_mesmo_sentido"] is False, (
        "o erro contra a verdade era INCONSISTENTE entre sementes, e é isso que sustenta "
        "'não há ganho'. Se passou a ser consistente, a leitura de AR160 muda."
    )
