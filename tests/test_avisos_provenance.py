"""Os três avisos do relatório: a fixture cobre a lista canônica, com os dois textos de cada um.

A comparação é contra `AVISOS`, a lista canônica, e nunca contra o que o relatório produziu.
"""

import json
import shutil
from pathlib import Path

import pytest

from refino.reporting import avisos as modulo
from refino.reporting.avisos import (
    AVISOS,
    FIXTURE,
    carrega_avisos,
    defeitos_de_procedencia,
)
from refino.reporting.model import (
    ITENS_OBRIGATORIOS,
    Afirmacao,
    ItemApendice,
    ItemCorpo,
    Relatorio,
    defeitos_do_relatorio,
)

# Um trecho do corpo e um do apêndice de cada aviso, escritos à mão. Se viessem da fixture, o
# teste compararia a fixture com ela mesma.
TRECHOS: dict[str, tuple[str, str]] = {
    "dv18": (
        "Comparação com resultado obtido no SERRF do R não vai bater no dígito",
        "O porte é fiel nos onze passos",
    ),
    "dv22": (
        "6 de 20 propriedades delas divergem das reais além da tolerância declarada",
        "Duas correções estruturais levaram o vão de 13 para 6 propriedades",
    ),
    "m8": (
        "use como ponto de partida e confira o corretor no seu modo, não como veredito",
        "Leave-one-mode-out sobre 40 matrizes sintéticas com 7 QCs",
    ),
}

METRICAS = {"qc_rsd_median": 1.4381, "qc_dispersion_ratio": 0.9302}


def _relatorio_com(dados: dict[str, dict[str, str]]) -> Relatorio:
    """Um relatório completo cujos avisos vêm de `dados`."""
    return Relatorio(
        corpo=tuple(
            ItemCorpo(
                numero=c.numero,
                titulo=c.titulo,
                afirmacoes=(Afirmacao(f"conteúdo do item {c.numero}", "apx-bloco"),),
            )
            for c in ITENS_OBRIGATORIOS
        ),
        apendice=(ItemApendice("apx-bloco", "Bloco reexecutável", "refino processar"),),
        metricas=METRICAS,
        metricas_computadas=frozenset(METRICAS),
        avisos_corpo={a.id: dados[a.id]["consequencia"] for a in AVISOS},
        avisos_apendice={a.id: dados[a.id]["procedencia"] for a in AVISOS},
    )


def test_a_fixture_existe_e_cobre_os_tres_avisos_canonicos() -> None:
    dados = carrega_avisos()
    assert set(dados) == {a.id for a in AVISOS}


def test_cada_aviso_tem_os_dois_enquadramentos_nao_vazios() -> None:
    """O corpo diz a consequência e o apêndice diz a procedência, em campos separados."""
    dados = carrega_avisos()
    for aid, r in dados.items():
        assert r["consequencia"].strip(), f"{aid} sem consequência"
        assert r["procedencia"].strip(), f"{aid} sem procedência"
        assert r["consequencia"] != r["procedencia"], (
            f"{aid}: os dois enquadramentos são o mesmo texto"
        )


def test_o_relatorio_leva_os_tres_avisos_com_o_texto_de_sempre() -> None:
    """O relatório montado com a fixture traz os três avisos, no corpo e no apêndice."""
    rel = _relatorio_com(carrega_avisos())

    assert defeitos_do_relatorio(rel) == []
    assert set(rel.avisos_corpo) == set(rel.avisos_apendice) == set(TRECHOS)
    for aid, (no_corpo, no_apendice) in TRECHOS.items():
        assert no_corpo in rel.avisos_corpo[aid]
        assert no_apendice in rel.avisos_apendice[aid]


def test_cada_aviso_traz_so_os_dois_textos() -> None:
    """A fixture não carrega mais nada além do que o relatório mostra."""
    dados = json.loads(FIXTURE.read_text(encoding="utf-8"))
    for aid, registro in dados.items():
        assert set(registro) == {"consequencia", "procedencia"}, aid


def test_os_avisos_carregam_sem_documento_nenhum_no_disco(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Com a raiz apontada para uma pasta que só tem a fixture, carregar e conferir funcionam."""
    copia = tmp_path / "data" / "fixtures" / FIXTURE.name
    copia.parent.mkdir(parents=True)
    shutil.copyfile(FIXTURE, copia)
    monkeypatch.setattr(modulo, "REPO", tmp_path)
    monkeypatch.setattr(modulo, "FIXTURE", copia)
    assert not (tmp_path / "docs").exists()

    dados = modulo.carrega_avisos()

    assert set(dados) == {a.id for a in AVISOS}
    assert modulo.defeitos_de_procedencia(dados) == []
    assert defeitos_do_relatorio(_relatorio_com(dados)) == []


def test_a_arvore_limpa_nao_acusa_defeito_de_procedencia() -> None:
    """O lado positivo dos negativos abaixo: sem defeito plantado, nada é acusado."""
    assert defeitos_de_procedencia(carrega_avisos()) == []


def test_negativo_aviso_canonico_ausente_da_fixture_acusa() -> None:
    dados = carrega_avisos()
    del dados["dv18"]
    d = defeitos_de_procedencia(dados)
    assert any("canônicos ausentes" in x and "dv18" in x for x in d)


def test_negativo_aviso_fora_da_lista_canonica_acusa() -> None:
    dados = carrega_avisos()
    dados["outro"] = dict(dados["m8"])
    d = defeitos_de_procedencia(dados)
    assert len(d) == 1
    assert "fora de `AVISOS`" in d[0] and "outro" in d[0]


def test_negativo_enquadramento_vazio_acusa_nomeando_o_campo() -> None:
    dados = carrega_avisos()
    dados["dv22"] = {**dados["dv22"], "procedencia": "   "}
    d = defeitos_de_procedencia(dados)
    assert any("`procedencia` vazio" in x for x in d)


def test_fixture_ausente_levanta_nomeando_o_arquivo(tmp_path: Path) -> None:
    """Devolver vazio seria indistinguível de não haver aviso a mostrar."""
    with pytest.raises(FileNotFoundError, match=r"nao_existe\.json ausente"):
        carrega_avisos(tmp_path / "nao_existe.json")
