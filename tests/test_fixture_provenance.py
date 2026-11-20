"""Procedência de fixture verificada mecanicamente.

**O defeito que originou.** `loess_golden_r.json` registrava `"seed": 42`. A amostra
de 400 features havia sido sorteada com semente **1**; 42 era a do corretor. Um
campo, duas quantidades — e a que ficou registrada não era a que o campo, lido de
fora, promete.

**Por que isso é pior que omitir.** Omissão se anuncia: quem for regenerar vê que
falta e pergunta. O campo errado **responde** a pergunta, com o número de outra
coisa. Quem regenerasse a partir da procedência obteria outra amostra, veria os
números não fecharem, e concluiria **que o fixture não reproduz** — diagnóstico
apontando para o artefato quando o defeito estava no registro..

**Duas guardas, e as duas são estreitas de propósito.** Uma proíbe o campo `seed`
sem qualificação; a outra exige que fixture com sidecar declare o que **não**
reproduz. Nenhuma verifica que a procedência está *certa* — isso não é mecanizável.
Verificam que ela é **inequívoca**, que é o que faltava.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
FIXTURES = REPO / "data" / "fixtures"

# Palavras que, num campo vizinho, tornam `seed: null` inequívoco. O par
# `null` + explicação distingue "não se aplica" de "ninguém registrou"; sozinho,
# `null` é os dois.
_DETERMINISM_WORDS = ("determinist", "nenhuma semente", "sem sorteio", "nao ha sorteio")


def ambiguous_seed_fields(payload: dict[str, Any]) -> list[str]:
    """Caminhos em que um campo `seed` nu aparece sem qualificação.

    `seed` nu é o defeito: `sample_seed` e `corrector_seed` nomeiam a quantidade, e
    `n_python_seeds` é contagem e não semente. O único `seed` aceitável é `null`
    acompanhado de nota que diga por quê — e é assim que os dois geradores
    determinísticos do projeto já o registravam, o que torna o contraste medido e
    não hipotético.
    """
    out: list[str] = []

    def walk(node: Any, path: str) -> None:
        if isinstance(node, dict):
            if "seed" in node:
                value = node["seed"]
                neighbours = " ".join(str(v).lower() for v in node.values())
                explained = any(word in neighbours for word in _DETERMINISM_WORDS)
                if value is not None or not explained:
                    out.append(f"{path}.seed = {value!r}")
            for key, value in node.items():
                walk(value, f"{path}.{key}")
        elif isinstance(node, list):
            for index, value in enumerate(node[:4]):
                walk(value, f"{path}[{index}]")

    walk(payload, "")
    return out


def sidecar_declaration_problems(payload: dict[str, Any], fixture: Path) -> list[str]:
    """Fixture com sidecar tem de declarar o que **não** reproduz, e nomear existente.

    Lista vazia **explícita** é resposta válida e ausência não é: ausência não
    distingue "tudo reproduz" de "ninguém verificou", e as duas pedem ação oposta.
    E os nomes declarados são conferidos contra o `.npz`, senão um
    array renomeado deixaria a declaração falando de coisa que não existe — a
    isenção que sobrevive ao que isentava.
    """
    provenance = payload.get("provenance", {})
    sidecar = provenance.get("sidecar")
    if not sidecar:
        return []

    declaration = provenance.get("reproducibility")
    if not isinstance(declaration, dict):
        return [f"{fixture.name}: tem sidecar `{sidecar}` e nenhum campo `reproducibility`"]
    for key in ("reproducible_arrays", "non_reproducible_arrays"):
        if not isinstance(declaration.get(key), list):
            return [f"{fixture.name}: `reproducibility` sem `{key}` como lista"]
    good = list(declaration["reproducible_arrays"])
    bad = list(declaration["non_reproducible_arrays"])

    problems: list[str] = []
    if len(str(declaration.get("measured", ""))) < 40:
        problems.append(f"{fixture.name}: `reproducibility` sem o campo `measured` escrito")

    overlap = set(good) & set(bad)
    if overlap:
        problems.append(f"{fixture.name}: {sorted(overlap)} declarado nas duas listas")

    path = fixture.parent / str(sidecar)
    if path.is_file():
        with np.load(path) as data:
            names = set(data.files)
        # **Exaustividade, e é ela que fecha o limbo.** Array que não está em
        # nenhuma das duas listas não é "provavelmente reproduzível": é array sobre
        # o qual ninguém disse nada, e é o estado em que o `step1_after` passou uma
        # unidade inteira. Sem esta asserção a declaração cobre o que havia no dia
        # em que foi escrita, que é a lista virando retrato.
        for name in sorted(names - set(good) - set(bad)):
            problems.append(
                f"{fixture.name}: `{name}` está em {sidecar} e em nenhuma das duas listas"
            )
        for name in sorted((set(good) | set(bad)) - names):
            problems.append(f"{fixture.name}: declara `{name}`, que não está em {sidecar}")
    return problems


def _fixture_payloads() -> list[tuple[Path, dict[str, Any]]]:
    out: list[tuple[Path, dict[str, Any]]] = []
    for path in sorted(FIXTURES.glob("*.json")):
        out.append((path, json.loads(path.read_text(encoding="utf-8"))))
    return out


def test_ha_fixture_json_para_varrer() -> None:
    """Varredura sobre zero arquivos passa vacuamente.

    O portão de cobertura passou com zero statements. Esta asserção é a
    versão barata da mesma defesa: se o `glob` deixar de casar, a suíte diz que a
    procedência está sã.
    """
    assert len(_fixture_payloads()) >= 4


@pytest.mark.parametrize("fixture", [p.name for p, _ in _fixture_payloads()])
def test_nenhum_fixture_usa_campo_seed_ambiguo(fixture: str) -> None:
    payload = json.loads((FIXTURES / fixture).read_text(encoding="utf-8"))
    ambiguos = ambiguous_seed_fields(payload)
    assert not ambiguos, (
        f"{fixture} usa `seed` sem dizer de quê: {ambiguos}. Nomeie a quantidade — "
        "`sample_seed`, `corrector_seed` — ou registre `null` com a nota que diz por "
        "que não há sorteio."
    )


@pytest.mark.parametrize("fixture", [p.name for p, _ in _fixture_payloads()])
def test_todo_fixture_com_sidecar_declara_o_que_nao_reproduz(fixture: str) -> None:
    path = FIXTURES / fixture
    problemas = sidecar_declaration_problems(json.loads(path.read_text(encoding="utf-8")), path)
    assert not problemas, (
        f"{problemas}. Sidecar sem declaração de reprodutibilidade faz um `--check` "
        "futuro comparar tudo e reprovar sempre (AR35), e faz quem vê um array mudar "
        "concluir que o fixture não reproduz — quando o irreprodutível é o oráculo."
    )


@pytest.mark.parametrize(
    ("payload", "acusa"),
    [
        # O defeito real, em pessoa.
        ({"provenance": {"seed": 42}}, True),
        # `null` sem explicação: continua sendo dois significados num campo.
        ({"provenance": {"seed": None}}, True),
        # `null` com a explicação — a forma que os dois geradores determinísticos usam.
        ({"provenance": {"seed": None, "note": "fivenum e deterministico"}}, False),
        # Nomeados: nenhum é `seed` nu.
        ({"provenance": {"sample_seed": 1, "corrector_seed": 42}}, False),
        ({"preregistration": {"n_python_seeds": 20}}, False),
        # Aninhado, para provar que a varredura desce.
        ({"a": {"b": [{"seed": 7}]}}, True),
    ],
)
def test_a_guarda_de_semente_acusa_o_ambiguo_e_so_ele(payload: dict[str, Any], acusa: bool) -> None:
    """A violação plantada, pela **mesma** função (taxonomia em processo).

    Os quatro casos que **não** podem acusar carregam metade do valor: guarda que
    reprova todo `seed` obrigaria a remover o campo, que é a informação.
    """
    assert bool(ambiguous_seed_fields(payload)) is acusa


@pytest.mark.parametrize(
    ("provenance", "acusa"),
    [
        ({"sidecar": "loess_golden_r.npz"}, True),
        ({"sidecar": "loess_golden_r.npz", "reproducibility": {}}, True),
        # Declara só um lado: os doze arrays ficam em limbo.
        (
            {
                "sidecar": "loess_golden_r.npz",
                "reproducibility": {"non_reproducible_arrays": [], "measured": "x" * 41},
            },
            True,
        ),
        # Nomeia array que não existe no sidecar.
        (
            {
                "sidecar": "loess_golden_r.npz",
                "reproducibility": {
                    "reproducible_arrays": [
                        "floor_neg_c18",
                        "floor_neg_hilic",
                        "floor_pos_c18",
                        "floor_pos_hilic",
                        "golden_uncorrected_neg_c18",
                        "golden_uncorrected_neg_hilic",
                        "golden_uncorrected_pos_c18",
                        "golden_uncorrected_pos_hilic",
                        "sample_neg_c18",
                        "sample_neg_hilic",
                        "sample_pos_c18",
                        "sample_pos_hilic",
                    ],
                    "non_reproducible_arrays": ["nao_existe"],
                    "measured": "x" * 41,
                },
            },
            True,
        ),
        # Um array em falta na partição — o limbo com onze dos doze declarados.
        (
            {
                "sidecar": "loess_golden_r.npz",
                "reproducibility": {
                    "reproducible_arrays": [
                        "floor_neg_c18",
                        "floor_neg_hilic",
                        "floor_pos_c18",
                        "floor_pos_hilic",
                        "golden_uncorrected_neg_c18",
                        "golden_uncorrected_neg_hilic",
                        "golden_uncorrected_pos_c18",
                        "golden_uncorrected_pos_hilic",
                        "sample_neg_c18",
                        "sample_neg_hilic",
                        "sample_pos_c18",
                    ],
                    "non_reproducible_arrays": [],
                    "measured": "x" * 41,
                },
            },
            True,
        ),
        # O mesmo array nas duas listas.
        (
            {
                "sidecar": "loess_golden_r.npz",
                "reproducibility": {
                    "reproducible_arrays": [
                        "floor_neg_c18",
                        "floor_neg_hilic",
                        "floor_pos_c18",
                        "floor_pos_hilic",
                        "golden_uncorrected_neg_c18",
                        "golden_uncorrected_neg_hilic",
                        "golden_uncorrected_pos_c18",
                        "golden_uncorrected_pos_hilic",
                        "sample_neg_c18",
                        "sample_neg_hilic",
                        "sample_pos_c18",
                        "sample_pos_hilic",
                    ],
                    "non_reproducible_arrays": ["floor_pos_hilic"],
                    "measured": "x" * 41,
                },
            },
            True,
        ),
        # A partição exaustiva e disjunta: o único caso que passa.
        (
            {
                "sidecar": "loess_golden_r.npz",
                "reproducibility": {
                    "reproducible_arrays": [
                        "floor_neg_c18",
                        "floor_neg_hilic",
                        "floor_pos_c18",
                        "floor_pos_hilic",
                        "golden_uncorrected_neg_c18",
                        "golden_uncorrected_neg_hilic",
                        "golden_uncorrected_pos_c18",
                        "golden_uncorrected_pos_hilic",
                        "sample_neg_c18",
                        "sample_neg_hilic",
                        "sample_pos_c18",
                        "sample_pos_hilic",
                    ],
                    "non_reproducible_arrays": [],
                    "measured": "x" * 41,
                },
            },
            False,
        ),
        # Sem sidecar não há o que declarar.
        ({"command": "scripts/gen_insample_medians.py"}, False),
    ],
)
def test_a_guarda_de_sidecar_acusa_declaracao_ausente_ou_orfa(
    provenance: dict[str, Any], acusa: bool
) -> None:
    """Inclui o caso do array **renomeado**, que é o que faz a declaração envelhecer.

    Uma declaração que nomeia array inexistente é isenção que sobreviveu ao que
    isentava — o mesmo modo de falha que a segunda direção da política de `skip`
    existe para pegar.
    """
    problemas = sidecar_declaration_problems(
        {"provenance": provenance}, FIXTURES / "loess_golden_r.json"
    )
    assert bool(problemas) is acusa
