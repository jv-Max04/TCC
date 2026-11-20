"""Os cinco documentos que VIAJAM afirmam o regime CERTO do executável.

O congelado corre o SERRF em paralelo, e cada documento que vai dentro da pasta entregue tem de
trazer a medição desse regime. Número caduco tem todas as marcas de confiabilidade — foi medido,
tem população, tem data — e envelhece em silêncio. **Quem o lê é o usuário**, na máquina dele, sem
nada com que o desmentir.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.conftest import REPO


def _linhas(caminho: Path) -> list[tuple[int, str]]:
    return list(enumerate(caminho.read_text(encoding="utf-8").splitlines(), start=1))


def _paragrafos(caminho: Path) -> list[list[tuple[int, str]]]:
    """As linhas agrupadas em PARÁGRAFOS — corridas maximais de linhas não vazias.

    **A busca é por parágrafo, e não por linha:** estes documentos quebram por volta dos 90
    caracteres, e uma medição partida entre duas linhas não casaria numa busca linha a linha.

    **A linha em branco é a fronteira nos dois formatos deste conjunto**: o `.txt` separa blocos
    por linha vazia, e os quatro `.html` também — nenhum deles é HTML minificado. É por isso que a
    regra é a mesma para os cinco, e não uma por extensão.
    """
    grupos: list[list[tuple[int, str]]] = []
    atual: list[tuple[int, str]] = []
    for numero, linha in _linhas(caminho):
        if linha.strip():
            atual.append((numero, linha))
        elif atual:
            grupos.append(atual)
            atual = []
    if atual:
        grupos.append(atual)
    return grupos


@pytest.mark.parametrize(
    ("documento", "numero_novo"),
    (
        (REPO / "docs" / "GUIA-do-aplicativo.html", "82,3 s"),
        (REPO / "docs" / "LEIA-ME-aplicativo.txt", "82,3 s"),
        (REPO / "docs" / "EXPLICAR-cada-tela-e-cada-campo.html", "82,3 s"),
        (REPO / "docs" / "FAQ-perguntas-frequentes.html", "vários núcleos"),
        (REPO / "docs" / "ENTENDER-o-sistema.html", "82,3 s"),
    ),
    ids=lambda x: x.name if isinstance(x, Path) else x,
)
def test_cada_documento_tem_a_medicao_NOVA_e_ela_nao_esta_marcada(  # noqa: N802
    documento: Path, numero_novo: str
) -> None:
    """O documento tem de dizer o que vale HOJE, com a medição do regime paralelo."""
    novas = [
        paragrafo[0][0]
        for paragrafo in _paragrafos(documento)
        if numero_novo in " ".join(linha.strip() for _, linha in paragrafo)
    ]
    assert novas, f"{documento.name} não traz `{numero_novo}`: a medição nova não entrou"
