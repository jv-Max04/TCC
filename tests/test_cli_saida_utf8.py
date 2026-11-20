"""A CLI imprime prosa acentuada, e numa saída `cp932` isso derrubava o `exportar`.

O caso de teste de sistema CT-CLI-001 registrou o defeito em 13/09/2026: `refino exportar` gravava
o arquivo inteiro e saía com código 1, porque a linha *«correção aplicada»* não se codifica em
`cp932`, que é a página de código da saída em tubo deste Windows. O teste de byte-identidade
entre CLI e API não o via porque procurava o ponto de entrada no caminho do Linux e **pulava**.

**O par causal é em processo:** a mesma saída `cp932`, com e sem a guarda. Com ela o comando sai com
zero e a linha acentuada chega ao fluxo; sem ela, o mesmo comando levanta `UnicodeEncodeError`.
"""

from __future__ import annotations

import io
import sys
from typing import TYPE_CHECKING

import pytest

from refino import cli
from tests.conftest import measurements_path, sample_list_path

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.integration

MODO = "neg_hilic"


def _saida_cp932(monkeypatch: pytest.MonkeyPatch) -> io.TextIOWrapper:
    """Um `stdout` em `cp932` e `errors="strict"`, como o tubo desta máquina."""
    fluxo = io.TextIOWrapper(io.BytesIO(), encoding="cp932", errors="strict")
    monkeypatch.setattr(sys, "stdout", fluxo)
    return fluxo


def _argv(destino: Path) -> list[str]:
    return [
        "exportar",
        "--measurements",
        str(measurements_path(MODO)),
        "--sample-list",
        str(sample_list_path(MODO)),
        "--mode",
        MODO,
        "--output",
        str(destino),
    ]


def test_exportar_sai_com_zero_numa_saida_cp932(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fluxo = _saida_cp932(monkeypatch)
    destino = tmp_path / "exportado.csv"

    assert cli.main(_argv(destino)) == cli.EXIT_OK
    assert destino.is_file(), "âncora: o exportar não gravou o arquivo"

    fluxo.flush()
    texto = fluxo.buffer.getvalue().decode("utf-8")  # type: ignore[attr-defined]
    assert "correção aplicada" in texto, (
        f"a linha que derrubava o comando não chegou ao fluxo: {texto[-300:]!r}"
    )


def test_sem_a_guarda_a_mesma_saida_derruba_o_exportar(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A planta: a guarda desligada. Se isto deixar de levantar, a saída deixou de ser `cp932`."""
    _saida_cp932(monkeypatch)
    monkeypatch.setattr(cli, "garante_utf8", lambda *_a, **_k: ())

    with pytest.raises(UnicodeEncodeError):
        cli.main(_argv(tmp_path / "exportado.csv"))
