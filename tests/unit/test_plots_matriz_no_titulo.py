"""A matriz no título, além do rodapé.

`recorte.aplica` acrescenta «Matriz: …» ao título geral, quando há, e senão ao do primeiro eixo.
O oráculo da frase é o rodapé que a mesma chamada desenha: os dois saem do campo `matriz` do
`Recorte`, e `confere_a_matriz_do_titulo` afirma a igualdade — que todos os testes de figura passam
a cobrar por `confere_rotulos`.
"""

from __future__ import annotations

from matplotlib.figure import Figure

from refino.plots.recorte import (
    PREFIXO_DA_MATRIZ_NO_TITULO,
    Recorte,
    aplica,
    poe_a_matriz_no_titulo,
)
from tests.arnes_de_figura import confere_a_matriz_do_titulo, sem_a_matriz

RECORTE = Recorte(3, 10, "sem amostragem", "uma — «corrigida», do corretor «serrf»", "sem teste")
LINHA = f"{PREFIXO_DA_MATRIZ_NO_TITULO}{RECORTE.matriz}"


def _figura(*, geral: str | None, titulos: tuple[str, ...]) -> Figure:
    figura = Figure(figsize=(8, 5))
    eixos = figura.subplots(1, len(titulos)) if len(titulos) > 1 else [figura.add_subplot()]
    for eixo, titulo in zip(eixos, titulos, strict=True):
        eixo.plot([0, 1], [0, 1])
        eixo.set_title(titulo)
    if geral is not None:
        figura.suptitle(geral)
    return figura


def test_sem_titulo_geral_a_linha_vai_para_o_titulo_do_primeiro_eixo() -> None:
    figura = _figura(geral=None, titulos=("Correlação entre features",))
    aplica(figura, RECORTE)
    titulo = figura.axes[0].get_title()
    assert " ".join(titulo.split()).endswith(" ".join(LINHA.split()))
    assert sem_a_matriz(titulo) == "Correlação entre features"


def test_com_titulo_geral_a_linha_vai_para_ele_e_os_eixos_ficam_como_estavam() -> None:
    figura = _figura(geral="Painel da correção", titulos=("antes", "depois"))
    aplica(figura, RECORTE)
    assert LINHA in figura.get_suptitle()
    assert [e.get_title() for e in figura.axes] == ["antes", "depois"]


def test_a_segunda_chamada_nao_repete_a_linha_e_figura_vazia_nao_tem_onde_a_por() -> None:
    figura = _figura(geral=None, titulos=("t",))
    poe_a_matriz_no_titulo(figura, RECORTE)
    poe_a_matriz_no_titulo(figura, RECORTE)
    assert figura.axes[0].get_title().count(PREFIXO_DA_MATRIZ_NO_TITULO) == 1
    assert poe_a_matriz_no_titulo(Figure(), RECORTE) is None


def test_a_conferencia_contra_o_rodape_acusa_o_titulo_sem_a_linha() -> None:
    """Par causal: a mesma figura, com a linha e sem ela; só a segunda é acusada."""
    figura = _figura(geral=None, titulos=("Correlação entre features",))
    aplica(figura, RECORTE)
    assert confere_a_matriz_do_titulo(figura) == []
    eixo = figura.axes[0]
    eixo.set_title(sem_a_matriz(eixo.get_title()))
    assert confere_a_matriz_do_titulo(figura) != [], (
        "a planta nao aterrou: o titulo ficou sem a linha"
    )
