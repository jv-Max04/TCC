"""A tela nunca se pede por `QWidget.screen()` neste projeto.

**O defeito que isto guarda, e ele é da biblioteca e não do nosso cálculo.** No PySide6 6.11 o
objeto Python do `QScreen` que `QWidget.screen()` devolve nasce **filho** do widget que chamou
(heurística de valor de retorno do Shiboken). O `QScreen` é global da aplicação e continua vivo;
quando o widget morre, o Shiboken invalida o objeto Python mesmo assim — e ele é o MESMO objeto que
`QApplication.primaryScreen()` devolve —, e a consulta seguinte à tela corrompe o heap. Medido em
25/09/2026: `0xc0000374` em `_dimensiona` e em `dpi_da_tela`, 9 quebras em 20 no reprodutor de 123
testes com `.screen()`, e 0 em 20 sem ele.

## Os testes

1. **O portão**: nenhuma chamada `.screen(...)` — com ou sem argumento, porque `QWidget.screen(w)`
   também invalida — em `src/` nem em `tests/`. Lido por AST, com a âncora de não-vacuidade e o par
   causal das duas linhas de `b1139da`, verbatim.
2. **O perigo existe nesta versão, e a busca estática não o tem**: duas sondas, cada uma no seu
   processo. Plantar o perigo aqui dentro invalidaria o objeto global da tela para o resto da
   sessão, e a segunda sonda no mesmo processo leria esse estado. **Se uma versão futura do PySide
   deixar de invalidar**, a primeira reprova e diz que a guarda pode ter sobrado.
3. **`dpi_da_tela` lê a métrica do próprio widget**, e não da tela: é o conserto do segundo sítio.
"""

from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Final

import pytest

from tests.qt_guard import garante_plataforma_offscreen, membros_ausentes, sonda_o_ambiente

_FALTAM = membros_ausentes(sonda_o_ambiente())
if _FALTAM:  # pragma: no cover — a decisão é exercitada em `tests/test_qt_guard.py`, com planta
    pytest.skip(
        f"grupo opcional `desktop` ausente: falta {', '.join(_FALTAM)}. "
        'Instale com `uv pip install -e ".[desktop]"` para exercitar os testes de janela.',
        allow_module_level=True,
    )

garante_plataforma_offscreen()

from refino.desktop.janela.figuras import dpi_da_tela  # noqa: E402

if TYPE_CHECKING:
    from pytestqt.qtbot import QtBot

REPO: Final = Path(__file__).resolve().parents[1]
VARRIDOS: Final = (REPO / "src", REPO / "tests")


def chamadas_de_screen(fonte: str) -> list[int]:
    """As linhas com `<algo>.screen(...)`, com ou sem argumento."""
    return [
        no.lineno
        for no in ast.walk(ast.parse(fonte))
        if isinstance(no, ast.Call)
        and isinstance(no.func, ast.Attribute)
        and no.func.attr == "screen"
    ]


def test_ninguem_pede_a_tela_por_instancia() -> None:
    arquivos = sorted(p for raiz in VARRIDOS for p in raiz.rglob("*.py"))
    assert len(arquivos) >= 200, (
        f"âncora: só {len(arquivos)} arquivos em src/ e tests/; a leitura falhou"
    )
    achadas = {
        str(p.relative_to(REPO)): linhas
        for p in arquivos
        if (linhas := chamadas_de_screen(p.read_text(encoding="utf-8")))
    }
    assert not achadas, (
        f"`.screen(...)` em {achadas}: no PySide6 o objeto da tela vira filho do widget e é "
        "invalidado quando ele morre. Use `QApplication.screenAt(...)`, "
        "`QApplication.primaryScreen()` ou a métrica do próprio widget (`logicalDpiX`)."
    )


def test_o_portao_ACUSA_as_chamadas_que_existiam() -> None:  # noqa: N802
    """A planta são as duas linhas de `b1139da`, verbatim, e a forma com argumento."""
    assert chamadas_de_screen("tela = self.screen()\n") == [1]
    assert chamadas_de_screen("tela = widget.screen()\n") == [1]
    assert chamadas_de_screen("tela = QWidget.screen(widget)\n") == [1]
    assert chamadas_de_screen("tela = QApplication.primaryScreen()\n") == []


def _sonda(codigo: str) -> str:
    feito = subprocess.run(
        [sys.executable, "-c", codigo],
        capture_output=True,
        text=True,
        timeout=120,
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
        check=False,
    )
    assert feito.returncode == 0, f"a sonda não correu: {feito.stderr[-800:]}"
    return feito.stdout.strip()


_POR_INSTANCIA: Final = """
import shiboken6
from PySide6.QtWidgets import QApplication, QWidget
app = QApplication([])
w = QWidget()
tela = w.screen()
shiboken6.delete(w)
print(shiboken6.isValid(tela))
"""

_ESTATICA: Final = """
import shiboken6
from PySide6.QtWidgets import QApplication, QWidget
app = QApplication([])
w = QWidget()
tela = QApplication.screenAt(w.mapToGlobal(w.rect().center())) or QApplication.primaryScreen()
dpi = (w.logicalDpiX() + w.logicalDpiY()) / 2
shiboken6.delete(w)
print(shiboken6.isValid(tela), dpi > 0)
"""


def test_a_tela_pedida_por_instancia_morre_com_o_widget() -> None:
    assert _sonda(_POR_INSTANCIA) == "False", (
        "o objeto da tela sobreviveu ao widget que o pediu: esta versão do PySide deixou de "
        "invalidá-lo, e a guarda pode ter sobrado."
    )


def test_a_tela_buscada_de_forma_estatica_sobrevive_ao_widget() -> None:
    assert _sonda(_ESTATICA) == "True True"


def test_dpi_da_tela_e_a_metrica_do_proprio_widget(qtbot: QtBot) -> None:
    from PySide6.QtWidgets import QWidget

    widget = QWidget()
    qtbot.addWidget(widget)
    esperado = (widget.logicalDpiX() + widget.logicalDpiY()) / 2
    assert esperado > 0, "âncora: a métrica do widget não é positiva"
    assert dpi_da_tela(widget) == esperado
