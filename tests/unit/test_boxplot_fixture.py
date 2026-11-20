"""Composição e procedência do oráculo do R.

Este módulo não testa `boxplot_outlier_mask`: testa o **fixture**. A distinção
importa porque o critério de paridade entrou defeituoso justamente por confiar que "20
vetores tirados dos dados reais" cobririam o caso que interessa. Os vetores
óbvios são os de 7 QCs — todos ímpares, o caso em que dobradiça e quartil
`type = 7` são a mesma expressão. Um fixture sem `n` par deixa o critério
decorativo, e nenhum teste do critério acusaria isso.

Por isso a composição do fixture é ela mesma verificada, e a verificação que
mais importa é `test_fixture_contem_caso_que_discrimina_as_duas_familias`.
"""

import json
from pathlib import Path
from typing import Any

import pytest

from tests.conftest import REPO, read_manifest

FIXTURE = REPO / "data" / "fixtures" / "boxplot_stats_r.json"

MIN_VETORES = 20
MIN_PARES = 5
N_OBRIGATORIOS = frozenset({2, 5, 7, 68})
COEFS_OBRIGATORIOS = frozenset({1.5, 3.0})


@pytest.fixture(scope="module")
def fixture_json() -> dict[str, Any]:
    if not FIXTURE.is_file():  # pragma: no cover - versionado; guarda para checkout parcial
        pytest.skip(
            f"{FIXTURE.relative_to(REPO)} ausente; a fixture é versionada, "
            "restaure-a do repositório"
        )
    data: dict[str, Any] = json.loads(FIXTURE.read_text(encoding="utf-8"))
    return data


@pytest.fixture(scope="module")
def vetores(fixture_json: dict[str, Any]) -> list[dict[str, Any]]:
    vs: list[dict[str, Any]] = fixture_json["vectors"]
    return vs


# ---------------------------------------------------------------------------
# Composição — o que o critério de paridade exige do conjunto
# ---------------------------------------------------------------------------


def test_fixture_tem_vetores_suficientes(vetores: list[dict[str, Any]]) -> None:
    assert len(vetores) >= MIN_VETORES, f"{len(vetores)} vetores; o critério pede {MIN_VETORES}"


def test_fixture_tem_comprimento_par_em_quantidade(vetores: list[dict[str, Any]]) -> None:
    pares = [v for v in vetores if v["n"] % 2 == 0]
    assert len(pares) >= MIN_PARES, (
        f"{len(pares)} vetores de n par; o critério pede {MIN_PARES}. "
        "Sem n par o fixture cai inteiro no caso em que as duas famílias de "
        "quartil coincidem por identidade algébrica."
    )


def test_fixture_cobre_os_comprimentos_que_o_pipeline_produz(vetores: list[dict[str, Any]]) -> None:
    """7 QCs, 68 injeções, e os 5 e 2 que a partição dos QCs produz."""
    presentes = {v["n"] for v in vetores}
    faltando = N_OBRIGATORIOS - presentes
    assert not faltando, f"comprimentos ausentes do fixture: {sorted(faltando)}"


def test_fixture_cobre_os_dois_coef_em_uso(vetores: list[dict[str, Any]]) -> None:
    presentes = {float(v["coef"]) for v in vetores}
    faltando = COEFS_OBRIGATORIOS - presentes
    assert not faltando, f"coef ausentes: {sorted(faltando)}"


def test_fixture_cobre_os_quatro_modos(vetores: list[dict[str, Any]]) -> None:
    modos = {v["mode"] for v in vetores}
    assert modos == {"pos_hilic", "pos_c18", "neg_hilic", "neg_c18"}, modos


def test_fixture_contem_caso_que_discrimina_as_duas_familias(
    vetores: list[dict[str, Any]],
) -> None:
    """**O teste que dá dente à paridade com o R.**

    O fixture grava o conjunto de outliers das duas famílias. Se nenhum vetor
    tiver conjuntos diferentes, uma implementação por `numpy.percentile` passa em
    tudo — e o critério que existe para pegar exatamente esse defeito não pega
    nada. Sem esta asserção, "o fixture cobre n par" é afirmação sobre o gerador
    que ninguém verifica.
    """
    divergentes = [v for v in vetores if v["out_indices_1based"] != v["type7_out_indices_1based"]]
    assert divergentes, (
        "nenhum vetor do fixture distingue dobradiça de Tukey de quartil type 7. "
        "O [CA-2.7] seria decorativo: reveja a regra de seleção do gerador."
    )
    assert all(v["n"] % 2 == 0 for v in divergentes), (
        "vetor de n ímpar com conjuntos divergentes é impossível por identidade "
        f"algébrica — o gerador ou o fixture está errado: "
        f"{[v['id'] for v in divergentes if v['n'] % 2]}"
    )


def test_vetores_de_n_impar_nunca_divergem(vetores: list[dict[str, Any]]) -> None:
    """A outra metade da identidade: para `n` ímpar as duas famílias coincidem.

    É invariante de desenho — posição `(n+3)/4` nas duas —, não observação sobre
    estes arquivos, e por isso é afirmada como invariante.
    """
    for v in vetores:
        if v["n"] % 2 == 1:
            assert v["out_indices_1based"] == v["type7_out_indices_1based"], (
                f"{v['id']} tem n={v['n']} ímpar e conjuntos diferentes"
            )


def test_vetores_de_no_maximo_dois_pontos_nao_marcam_nada(vetores: list[dict[str, Any]]) -> None:
    """Com `n <= 2` as dobradiças são os extremos: nenhum ponto fica fora."""
    for v in vetores:
        if v["n_finite"] <= 2:
            assert v["out_indices_1based"] == [], f"{v['id']} marcou outlier com n<=2"


def test_todo_vetor_tem_os_campos_que_o_teste_de_paridade_consome(
    vetores: list[dict[str, Any]],
) -> None:
    obrigatorios = {
        "id",
        "mode",
        "origin",
        "feature",
        "n",
        "n_finite",
        "coef",
        "values",
        "fivenum",
        "out_indices_1based",
        "type7_out_indices_1based",
    }
    for v in vetores:
        faltando = obrigatorios - set(v)
        assert not faltando, f"{v.get('id', '?')} sem os campos {sorted(faltando)}"
        assert len(v["values"]) == v["n"]
        assert len(v["fivenum"]) == 5


def test_ids_sao_unicos(vetores: list[dict[str, Any]]) -> None:
    ids = [v["id"] for v in vetores]
    assert len(ids) == len(set(ids)), "id repetido torna a mensagem de falha ambígua"


# ---------------------------------------------------------------------------
# Procedência — "mudou o R ou mudou nosso código?"
# ---------------------------------------------------------------------------


def test_procedencia_esta_completa(fixture_json: dict[str, Any]) -> None:
    prov = fixture_json["provenance"]
    for campo in ("r_version", "image", "image_digest", "command", "generated_at", "sources"):
        assert prov.get(campo), f"procedência sem {campo}"
    assert prov["r_version"].startswith("R version"), prov["r_version"]
    assert "sha256:" in prov["image_digest"] or prov["image_digest"].startswith("sha256"), (
        f"digest sem forma de digest: {prov['image_digest']!r}"
    )
    assert "seed" in prov, (
        "a chave `seed` tem de existir mesmo nula: ausência é indistinguível de "
        "esquecimento, e `null` afirma que o processo é determinístico"
    )


def test_procedencia_declara_a_regra_de_selecao(fixture_json: dict[str, Any]) -> None:
    """Sem a regra escrita, ninguém sabe por que estas features e não outras."""
    regra = fixture_json["provenance"]["selection_rule"]
    assert "impar" in regra or "ímpar" in regra
    assert "PAR" in regra or "par" in regra


def test_sha256_das_fontes_casa_com_o_manifesto_vigente(fixture_json: dict[str, Any]) -> None:
    """Expectativa obsoleta reprova com "regenere", não compara em silêncio.

    Sem isto, um `measurements` regerado pelo laboratório deixaria o fixture
    descrevendo dados que não existem mais, e a paridade continuaria verde.
    """
    manifesto = read_manifest()
    divergentes = {
        caminho: (sha, manifesto.get(caminho))
        for caminho, sha in fixture_json["provenance"]["sources"].items()
        if manifesto.get(caminho) != sha
    }
    assert not divergentes, (
        "as fontes do fixture divergem do manifesto de integridade: "
        f"{divergentes}. A fixture descreve dados que não são os do manifesto."
    )


def test_caminho_do_fixture_e_o_que_a_especificacao_declara() -> None:
    assert FIXTURE == REPO / "data" / "fixtures" / "boxplot_stats_r.json"
    assert Path(FIXTURE).is_file()
