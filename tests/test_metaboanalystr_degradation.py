"""O subcomando `exportar` da CLI: escreve o arquivo do MetaboAnalyst e confere o formato.

A CLI não pede enriquecimento de vias. Quem enriquece é a janela, pelo R portátil.
"""

import argparse
from pathlib import Path

import pytest

from refino import cli
from refino.cli import main
from refino.io.validators import check_metaboanalyst_conformance
from tests.conftest import measurements_path, sample_list_path


def _cli_args(tmp_path: Path) -> list[str]:
    return [
        "exportar",
        "--measurements",
        str(measurements_path("neg_c18")),
        "--sample-list",
        str(sample_list_path("neg_c18")),
        "--mode",
        "neg_c18",
        "--output",
        str(tmp_path / "saida.csv"),
    ]


def test_a_exportacao_completa_e_sai_zero(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """A exportação escreve o arquivo, confere a conformidade e sai com código 0."""
    assert main(_cli_args(tmp_path)) == 0

    output = capsys.readouterr().out
    assert "exportado:" in output
    assert "conformidade ao formato publicado: OK" in output
    assert "enriquecimento" not in output
    assert (tmp_path / "saida.csv").is_file()
    assert check_metaboanalyst_conformance(tmp_path / "saida.csv") == []


def test_a_cli_reporta_as_tres_causas_de_remocao(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    """As três causas chegam ao usuário separadas, e não como um total."""
    assert main(_cli_args(tmp_path)) == 0

    output = capsys.readouterr().out
    assert "QCs todos zero:" in output
    assert "remoção de outliers apagou os QCs:" in output
    assert "RSD acima do corte:" in output


def test_enriquecer_e_opcao_desconhecida(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """A opção saiu da CLI: o analisador a recusa e nada é escrito."""
    with pytest.raises(SystemExit) as saida:
        main([*_cli_args(tmp_path), "--enriquecer"])

    assert saida.value.code == 2
    erro = capsys.readouterr().err
    assert "unrecognized arguments: --enriquecer" in erro
    assert not (tmp_path / "saida.csv").exists()


def _ajuda_de_exportar(capsys: pytest.CaptureFixture) -> str:
    with pytest.raises(SystemExit) as saida:
        cli.build_parser().parse_args(["exportar", "--help"])
    assert saida.value.code == 0
    return capsys.readouterr().out


def test_a_ajuda_de_exportar_nao_cita_conteiner_nem_enriquecimento(
    capsys: pytest.CaptureFixture,
) -> None:
    """A ajuda descreve só o que o subcomando faz: corrigir, exportar e conferir."""
    ajuda = _ajuda_de_exportar(capsys)
    assert "--measurements" in ajuda, "âncora: a ajuda de exportar veio vazia"

    minuscula = ajuda.lower()
    for termo in ("metaboanalystr", "contêiner", "container", "docker", "enriquec", "serviço r"):
        assert termo not in minuscula, f"a ajuda de exportar ainda cita {termo!r}"


def test_a_linha_de_exportar_na_ajuda_geral_nao_cita_enriquecimento() -> None:
    """A lista de subcomandos da ajuda geral mostra o resumo de exportar sem enriquecimento."""
    analisador = cli.build_parser()
    acoes = [a for a in analisador._actions if isinstance(a, argparse._SubParsersAction)]
    resumos = {a.dest: a.help or "" for a in acoes[0]._choices_actions}
    assert "exportar" in resumos, "âncora: exportar não está entre os subcomandos"
    assert "enriquec" not in resumos["exportar"].lower()
    assert "MetaboAnalyst" in resumos["exportar"]
