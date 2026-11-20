"""A tela de arranque: quem escreve nela, quem a fecha, e a coordenada que dois arquivos partilham.

## Por que este arquivo existe

`_fecha_a_tela_de_arranque` nasceu em 08/09/2026 e ficou **sem teste nenhum** até 08/09/2026 — foi
escrito, embarcado e enviado sem nunca ter sido visto produzir os dois desfechos. Construir a
verificação dá a sensação de tê-la, e o passo seguinte não tem gatilho próprio. `_diz_na_tela`
nasce aqui já com as duas direções.

## As três direções, e a terceira é a que ninguém exercita

As duas óbvias são *o módulo existe* e *o módulo não existe*. A terceira é **o módulo existe e
levanta** — e é ela que justifica o `except` largo. Sem a exercitar, o `except` é uma afirmação
sobre um caminho que ninguém percorreu.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any

import pytest

from refino.desktop import arranque

if TYPE_CHECKING:  # pragma: no cover - só para o mypy
    from collections.abc import Iterator

REPO = Path(__file__).resolve().parents[2]


def _falso_pyi_splash(registro: list[tuple[str, Any]], *, levanta: bool = False) -> ModuleType:
    """Um `pyi_splash` de mentira que ANOTA o que lhe pediram.

    **Anota em vez de só existir**, porque a pergunta do par causal não é *"a função devolveu?"* —
    é *"a chamada chegou ao outro lado, com o argumento certo?"*. Um duplo que aceite tudo e não
    registre nada passa nas duas direções e não separa nenhuma.
    """
    modulo = ModuleType("pyi_splash")

    def update_text(msg: str) -> None:
        registro.append(("update_text", msg))
        if levanta:
            raise RuntimeError("a tela morreu entre a conferência e a escrita")

    def close() -> None:
        registro.append(("close", None))
        if levanta:
            raise RuntimeError("a tela morreu entre a conferência e o fecho")

    modulo.update_text = update_text  # type: ignore[attr-defined]
    modulo.close = close  # type: ignore[attr-defined]
    return modulo


@pytest.fixture
def sem_pyi_splash() -> Iterator[None]:
    """Garante a ausência REAL do módulo, e repõe o que houvesse.

    Fora do congelado ele já não existe — mas afirmar isso seria depender do ambiente, e é
    exactamente a classe de asserção que passa por acidente.
    """
    guardado = sys.modules.pop("pyi_splash", None)
    try:
        yield
    finally:
        if guardado is not None:
            sys.modules["pyi_splash"] = guardado


# =============================================================================
# O PAR CAUSAL: com o módulo e sem ele, os desfechos DIVERGEM
# =============================================================================


def test_com_a_tela_presente_a_etapa_chega_ao_outro_lado(monkeypatch: pytest.MonkeyPatch) -> None:
    """DIREÇÃO (a): há tela, e o texto pedido chega tal e qual."""
    registro: list[tuple[str, Any]] = []
    monkeypatch.setitem(sys.modules, "pyi_splash", _falso_pyi_splash(registro))

    arranque._diz_na_tela("a subir o servidor local... (3 de 4)")

    assert registro == [("update_text", "a subir o servidor local... (3 de 4)")], (
        "a etapa não chegou ao `pyi_splash`, ou chegou alterada"
    )


def test_sem_a_tela_a_etapa_nao_chega_a_lado_nenhum_e_nao_levanta(sem_pyi_splash: None) -> None:
    """DIREÇÃO (b): não há tela — e o desfecho é silêncio, não erro.

    **É este o desfecho que o par causal precisa de ver DIVERGIR** do de cima. Sem ele, um
    `_diz_na_tela` que nunca fizesse nada passaria na direção (a) por outro motivo qualquer.
    """
    assert "pyi_splash" not in sys.modules

    arranque._diz_na_tela("a preparar o banco de dados... (2 de 4)")

    assert "pyi_splash" not in sys.modules, "a função não pode CRIAR o módulo que não existia"


def test_com_a_tela_presente_o_fecho_chega_ao_outro_lado(monkeypatch: pytest.MonkeyPatch) -> None:
    """Com o `pyi_splash` presente, o fecho chama o `close` dele uma vez."""
    registro: list[tuple[str, Any]] = []
    monkeypatch.setitem(sys.modules, "pyi_splash", _falso_pyi_splash(registro))

    arranque._fecha_a_tela_de_arranque()

    assert registro == [("close", None)]


def test_sem_a_tela_o_fecho_nao_levanta(sem_pyi_splash: None) -> None:
    """A ausência é o caso NORMAL: CLI, suíte, e congelado sem `Splash(...)`."""
    arranque._fecha_a_tela_de_arranque()


# =============================================================================
# A TERCEIRA direção: o módulo existe E LEVANTA — o ramo que o `except` largo cobre
# =============================================================================


@pytest.mark.parametrize(
    ("chamada", "argumento"),
    [(arranque._diz_na_tela, "seja o que for"), (arranque._fecha_a_tela_de_arranque, None)],
    ids=["diz", "fecha"],
)
def test_uma_tela_que_levanta_nao_derruba_o_arranque(
    monkeypatch: pytest.MonkeyPatch, chamada: Any, argumento: str | None
) -> None:
    """O desfecho pior é o programa não abrir, e é este ramo que o impede.

    **A planta cai sobre dado em que as duas coisas diferem:** o duplo REGISTA antes de levantar,
    então a asserção separa *"a chamada aconteceu e a excepção foi engolida"* de *"a chamada nunca
    aconteceu"* — que produziriam o mesmo silêncio se só se afirmasse a ausência de erro.
    """
    registro: list[tuple[str, Any]] = []
    monkeypatch.setitem(sys.modules, "pyi_splash", _falso_pyi_splash(registro, levanta=True))

    if argumento is None:
        chamada()
    else:
        chamada(argumento)

    assert len(registro) == 1, "a chamada não chegou ao outro lado — o teste passaria por vacuidade"


# =============================================================================
# O PORTÃO DA COORDENADA: dois arquivos têm de dizer o mesmo par
# =============================================================================


def test_a_posicao_da_etapa_e_a_mesma_no_gerador_e_no_spec() -> None:
    """`gera_splash.py` reserva a faixa e o `.spec` escreve nela — e nada os liga.

    **O modo de falha é silencioso nas duas direções:** com `text_pos` maior que a altura, a etapa
    é desenhada fora da imagem; com `text_pos` menor, ela cai por cima da prosa fixa. Nos dois
    casos o congelamento passa, o PNG é válido, e o único sintoma seria alguém OLHAR para a tela
    de arranque durante os segundos em que ela existe.
    """
    gerador = (REPO / "scripts" / "gera_splash.py").read_text(encoding="utf-8")
    spec = (REPO / "refino-desktop.spec").read_text(encoding="utf-8")

    achados = [linha for linha in gerador.splitlines() if linha.startswith("ETAPA_X, ETAPA_Y = ")]
    assert len(achados) == 1, f"esperava UMA definição de ETAPA_X/ETAPA_Y, achei {len(achados)}"
    x, y = (int(n.strip()) for n in achados[0].split("=", 1)[1].split(","))

    assert spec.count("    text_pos=") == 1, "esperava UM `text_pos` no `.spec`"
    assert f"    text_pos=({x}, {y})," in spec, (
        f"o gerador reserva a etapa em ({x}, {y}) e o `.spec` não diz o mesmo — "
        "a etapa sai por cima da prosa fixa ou fora da imagem, e nada levanta"
    )


def test_as_quatro_etapas_estao_numeradas_de_1_a_4_e_sem_buracos() -> None:
    """A numeração é promessa ao utilizador: «(3 de 4)» diz que faltam duas.

    Um quinto ponto acrescentado sem mexer nos outros produziria `(1 de 4)` duas vezes, ou um
    `(5 de 4)`. O oráculo é a PROPRIEDADE — a sequência completa — e não a forma de uma instância.
    """
    fonte = (REPO / "src" / "refino" / "desktop" / "arranque.py").read_text(encoding="utf-8")
    import re

    marcas = re.findall(r'_diz_na_tela\("[^"]*\((\d+) de (\d+)\)"\)', fonte)
    assert marcas, "nenhuma etapa numerada encontrada — o portão estaria a passar por vacuidade"

    totais = {int(t) for _, t in marcas}
    assert totais == {len(marcas)}, (
        f"{len(marcas)} etapas, mas elas dizem «de {totais}» — o total tem de bater com a contagem"
    )
    assert sorted(int(n) for n, _ in marcas) == list(range(1, len(marcas) + 1)), (
        "a numeração das etapas tem buraco ou repetição"
    )
