"""O `--help` de cada subcomando da CLI sai 0 e mostra o texto.

O argparse formata a ajuda com `%`, e um `%` literal sem escape — «as 60% de cima» — fazia
`refino processar --help` levantar `TypeError` e sair 1. Os subcomandos são lidos do próprio
analisador, e não de uma lista à mão, para que o próximo subcomando entre sem ninguém o acrescentar.
"""

from __future__ import annotations

import argparse

import pytest

from refino import cli


def _subcomandos() -> list[str]:
    analisador = cli.build_parser()
    acoes = [a for a in analisador._actions if isinstance(a, argparse._SubParsersAction)]
    assert acoes, "âncora: o analisador da CLI não tem subcomandos"
    return sorted(acoes[0].choices)


def test_ha_subcomandos() -> None:
    # A âncora contra a vacuidade do teste parametrizado abaixo.
    assert len(_subcomandos()) >= 3


@pytest.mark.parametrize("subcomando", _subcomandos())
def test_a_ajuda_de_cada_subcomando_sai_zero(
    subcomando: str, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as saida:
        cli.build_parser().parse_args([subcomando, "--help"])
    assert saida.value.code == 0
    assert subcomando in capsys.readouterr().out


def test_nenhuma_ajuda_da_cli_diz_multiverso(capsys: pytest.CaptureFixture[str]) -> None:
    textos = []
    for argumentos in [[], *[[sub] for sub in _subcomandos()]]:
        with pytest.raises(SystemExit):
            cli.build_parser().parse_args([*argumentos, "--help"])
        textos.append(capsys.readouterr().out)
    assert all(len(t) > 100 for t in textos), "âncora: alguma ajuda saiu vazia"
    assert [t for t in textos if "ultivers" in t.lower()] == []
