"""O Palpite pela verdade — o artefato, e o portão que o impede de envelhecer em silêncio.

**O que ele é.** A mesma árvore de profundidade 3 do Palpite, com as mesmas características e a
mesma semente, treinada pelo corretor que mais recupera o sinal verdadeiro nas matrizes sintéticas,
e não pelo de menor RSD dos QCs. O corpus é `data/fixtures/corpus_pela_verdade_88.jsonl`: as 77
matrizes decidíveis das 88 do plano da P2, com a floresta do SERRF que conta como o `ranger`
(um script de medição o monta e confere as quatro âncoras dele).

**Os portões são os do Palpite pelo RSD (`tests/test_politica_aprendida.py`), aplicados a este
artefato:** forma e procedência, concordância recomendação a recomendação com a árvore ajustada na
hora, e o par causal que prova que a concordância consegue reprovar. O leitor do corpus é o de lá,
importado e não copiado — a cláusula da irmã.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from refino.decision.artefato import PoliticaDoArtefato
from refino.decision.features import FEATURE_NAMES
from refino.decision.model import LearnedPolicy
from tests.test_politica_aprendida import SEMENTE, _carrega_corpus

REPO = Path(__file__).resolve().parents[1]
CORPUS = REPO / "data" / "fixtures" / "corpus_pela_verdade_88.jsonl"
ARTEFATO = REPO / "src" / "refino" / "decision" / "politica_pela_verdade.json"

# As 88 matrizes do plano da P2 têm 11 empates a 1% pela verdade com a floresta do R (4 nas 40 de
# treino, 7 nas 48 da YO1), e empate não treina. O número é AFIRMADO: comparação sobre corpus vazio
# passaria por vacuidade.
N_ESPERADO = 77


@pytest.fixture(scope="module")
def corpus() -> tuple[list[Any], list[str], list[float]]:
    assert CORPUS.exists(), f"corpus ausente: {CORPUS}. Ele é versionado — a ausência é quebra"
    perfis, rotulos, margens = _carrega_corpus(CORPUS)
    assert len(perfis) == N_ESPERADO, f"o corpus tem {len(perfis)} perfis, esperava {N_ESPERADO}"
    return perfis, rotulos, margens


@pytest.fixture(scope="module")
def ajustada(corpus: tuple[list[Any], list[str], list[float]]) -> LearnedPolicy:
    perfis, rotulos, margens = corpus
    return LearnedPolicy(seed=SEMENTE).fit(perfis, rotulos, margens)  # type: ignore[arg-type]


def test_o_artefato_declara_a_ordem_e_a_procedencia() -> None:
    dados = json.loads(ARTEFATO.read_text(encoding="utf-8"))
    assert tuple(dados["caracteristicas"]) == FEATURE_NAMES, "a ordem diverge: REGERE o artefato"
    procedencia = dados["procedencia"]
    assert procedencia["semente"] == SEMENTE
    assert procedencia["n_corpus"] == N_ESPERADO
    assert procedencia["corpus"] == CORPUS.name


def test_o_corpus_e_rotulado_pela_verdade_e_nao_pelo_rsd() -> None:
    """O que separa este corpus do do Palpite pelo RSD: cada linha traz os erros contra a verdade.

    E o rótulo é o de menor erro — conferido aqui linha a linha, para que um corpus montado com o
    rótulo errado (o do RSD, por exemplo) não passe pelo portão de concordância, que só compara a
    árvore com ela mesma.
    """
    linhas = [json.loads(x) for x in CORPUS.read_text(encoding="utf-8").splitlines() if x.strip()]
    assert linhas, "âncora: o corpus está vazio"
    for r in linhas:
        erros = r["erros"]
        assert set(erros) == {"none", "loess", "serrf"}
        assert r["rotulo"] == min(erros, key=lambda c: erros[c]), r
        segundo = sorted(erros.values())[1]
        assert r["margem"] == pytest.approx((segundo - erros[r["rotulo"]]) / erros[r["rotulo"]])


def test_o_artefato_reproduz_a_politica_ajustada_em_todos_os_perfis(
    corpus: tuple[list[Any], list[str], list[float]], ajustada: LearnedPolicy
) -> None:
    perfis, _, _ = corpus
    do_artefato = PoliticaDoArtefato.de_dict(json.loads(ARTEFATO.read_text(encoding="utf-8")))
    assert perfis, "âncora: o corpus está vazio e a comparação abaixo seria vácua"
    divergentes = [
        i
        for i, p in enumerate(perfis)
        if (do_artefato.recommend(p).correction, do_artefato.recommend(p).rationale)
        != (ajustada.recommend(p).correction, ajustada.recommend(p).rationale)
    ]
    assert not divergentes, (
        f"{len(divergentes)} de {len(perfis)} perfis divergem: REGERE com "
        "`python scripts/gera_politica_aprendida.py --corpus data/fixtures/corpus_pela_verdade_88"
        ".jsonl --destino src/refino/decision/politica_pela_verdade.json`"
    )


def test_um_limiar_adulterado_faz_a_concordancia_reprovar(
    corpus: tuple[list[Any], list[str], list[float]], ajustada: LearnedPolicy
) -> None:
    """O par causal: a planta cai na raiz, que toda entrada avalia, e a raiz é nó de decisão."""
    dados = json.loads(ARTEFATO.read_text(encoding="utf-8"))
    raiz = dados["corretor"]["nos"][0]
    assert raiz.get("folha") is not True, "âncora: a raiz é folha e a planta não separaria nada"
    dados["corretor"]["nos"][0]["limiar"] = float(raiz["limiar"]) + 1_000_000.0
    adulterada = PoliticaDoArtefato.de_dict(dados)
    perfis, _, _ = corpus
    assert any(
        adulterada.recommend(p).correction != ajustada.recommend(p).correction for p in perfis
    ), "adulterar a raiz não mudou recomendação nenhuma: o portão não consegue reprovar"


def test_os_dois_palpites_discordam_em_algum_perfil(
    corpus: tuple[list[Any], list[str], list[float]],
) -> None:
    """Âncora do valor da segunda opinião: se os dois concordassem sempre, ela não diria nada.

    Afirmado sobre os perfis do corpus pela verdade, que é onde os dois foram comparados.
    """
    perfis, _, _ = corpus
    pela_verdade = PoliticaDoArtefato.de_dict(json.loads(ARTEFATO.read_text(encoding="utf-8")))
    pelo_rsd = PoliticaDoArtefato.de_dict(
        json.loads(
            (REPO / "src" / "refino" / "decision" / "politica_aprendida.json").read_text(
                encoding="utf-8"
            )
        )
    )
    assert any(
        pela_verdade.recommend(p).correction != pelo_rsd.recommend(p).correction for p in perfis
    )


def _jsons_da_decisao_fora_do_spec(spec: str) -> list[str]:
    """Os artefatos JSON de `refino/decision` que o `datas=` do congelado não leva."""
    return [
        p.name
        for p in sorted((REPO / "src" / "refino" / "decision").glob("*.json"))
        if f'"decision" / "{p.name}"' not in spec
    ]


def test_o_congelado_leva_todo_artefato_da_decisao_no_datas() -> None:
    """JSON não é módulo: sem a entrada no `datas=`, a recomendação dá 503 só no executável.

    **Derivado da pasta, e não pinado nos dois nomes** — o próximo artefato nasce coberto. A
    âncora é que os dois Palpites estão na pasta: com a pasta vazia a lista abaixo passaria vazia.
    """
    spec = (REPO / "refino-desktop.spec").read_text(encoding="utf-8")
    nomes = {p.name for p in (REPO / "src" / "refino" / "decision").glob("*.json")}
    assert {"politica_aprendida.json", "politica_pela_verdade.json"} <= nomes, sorted(nomes)
    assert _jsons_da_decisao_fora_do_spec(spec) == []
    # par causal: sem a linha do Palpite pela verdade, o predicado acusa exatamente ele
    plantado = spec.replace(
        '"decision" / "politica_pela_verdade.json"', '"decision" / "outro.json"'
    )
    assert _jsons_da_decisao_fora_do_spec(plantado) == ["politica_pela_verdade.json"]


def test_d578_a_segunda_opiniao_esta_em_stand_by_e_o_interruptor_e_lido_na_chamada(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """O portão de estado: a segunda opinião nasce DESLIGADA, e religá-la é uma troca só.

    **O estado é deliberado** — só o Palpite por RSD, com a outra opinião em stand-by —, e este
    teste reprova quem a religar por descuido. **O par
    causal é sobre a leitura na chamada:** trocado o valor do módulo, a função devolve o novo, e é
    isso que deixa os testes das superfícies correrem as duas posições num processo só. O artefato
    continua a abrir: em stand-by, ele não pode ter apodrecido.
    """
    from refino.io import politica

    assert politica.SEGUNDA_OPINIAO_ATIVA is False
    assert politica.segunda_opiniao_ativa() is False
    monkeypatch.setattr(politica, "SEGUNDA_OPINIAO_ATIVA", True)
    assert politica.segunda_opiniao_ativa() is True
    assert politica.carrega_politica_pela_verdade().recommend is not None
