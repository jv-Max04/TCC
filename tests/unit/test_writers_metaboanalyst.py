"""O escritor conforme do formato do MetaboAnalyst.

Os requisitos `[F-1]` a `[F-6]` estão em `docs/formato-metaboanalyst.md`, com
procedência e data de acesso. Este módulo exercita o **escritor**; a
paridade entre as constantes do código e a citação é de
`tests/test_metaboanalyst_conformance.py`.

**O que este módulo NÃO cobre, e a leitura natural sugeriria que sim:** ele não
verifica política de corte. `max_features` é do critério e — aqui a
matriz cabe sempre, e `kept` é toda a população que entrou.
"""

import csv
import math
from pathlib import Path

import numpy as np
import pytest

from refino.domain.matrix import RefinoMatrix, SampleMetadata
from refino.io.readers import load_matrix
from refino.io.writers import (
    CONFORMING_NAME,
    MetaboAnalystExport,
    _sanitize_name,
    write_metaboanalyst,
)
from refino.preprocessing.normalization import scale_features, transform
from tests.conftest import measurements_path, sample_list_path

MODES = ("pos_hilic", "pos_c18", "neg_hilic", "neg_c18")


def _read(path: Path) -> list[list[str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.reader(handle))


def _samples(n: int) -> tuple[SampleMetadata, ...]:
    return tuple(
        SampleMetadata(
            name=f"S{i + 1}",
            injection_order=i + 1,
            sample_type="qc" if i % 2 == 0 else "sample",
            group="QC" if i % 2 == 0 else "NoHTLV",
        )
        for i in range(n)
    )


def _matrix(n_samples: int = 4, n_features: int = 3) -> RefinoMatrix:
    values = np.arange(1, n_samples * n_features + 1, dtype=np.float64)
    return RefinoMatrix(
        values=values.reshape(n_samples, n_features),
        feature_names=tuple(f"F{i + 1}" for i in range(n_features)),
        samples=_samples(n_samples),
        mode="tiny",
    )


# ---------------------------------------------------------------------------
# Sanitização de nome — `[F-4]`
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("original", "expected"),
    [
        ("2.25_810.7739m/z", "2_25_810_7739m_z"),
        ("HTLV_HILIC_PLASMA_MSe_POS_QC1", "HTLV_HILIC_PLASMA_MSe_POS_QC1"),
        ("Asymptomatic", "Asymptomatic"),
        ("a-b", "a_b"),
        ("café", "caf_"),
    ],
)
def test_sanitiza_para_o_alfabeto_do_formato(original: str, expected: str) -> None:
    """`[F-4]` admite letras inglesas, sublinhado e número, e nada mais."""
    assert _sanitize_name(original) == expected
    assert CONFORMING_NAME.match(_sanitize_name(original))


def test_nome_conforme_atravessa_intacto() -> None:
    """A direção que impede a sanitização de ser destrutiva por atacado.

    Sem esta, um sanitizador que trocasse tudo por sublinhado passaria o teste
    acima — o arquivo ficaria conforme e ilegível, e nada acusaria.
    """
    assert _sanitize_name("HTLV_POS_1_ID33") == "HTLV_POS_1_ID33"


# ---------------------------------------------------------------------------
# Estrutura do arquivo — `[F-5]`
# ---------------------------------------------------------------------------


def test_amostras_em_linhas_com_rotulo_na_segunda_coluna(tmp_path: Path) -> None:
    """`[F-5]`: "The class label is in the second column"."""
    matrix = _matrix(n_samples=4, n_features=3)
    destination = tmp_path / "export.csv"

    write_metaboanalyst(matrix, destination)
    rows = _read(destination)

    assert len(rows) == 5, "uma linha de cabeçalho e uma por amostra"
    assert rows[0][0] == "Sample"
    assert rows[0][1] == "Label"
    for row, sample in zip(rows[1:], matrix.samples, strict=True):
        assert row[1] == _sanitize_name(sample.group), (
            "o rótulo de classe não está na segunda coluna"
        )


def test_toda_celula_de_nome_do_arquivo_e_conforme(tmp_path: Path) -> None:
    matrix = _matrix(n_samples=3, n_features=4)
    destination = tmp_path / "export.csv"

    write_metaboanalyst(matrix, destination)
    rows = _read(destination)

    for name in rows[0][2:]:
        assert CONFORMING_NAME.match(name), f"nome de feature não conforme no arquivo: {name!r}"
    for row in rows[1:]:
        assert CONFORMING_NAME.match(row[0]), f"nome de amostra não conforme: {row[0]!r}"


# ---------------------------------------------------------------------------
# Zero e ausente — `[F-2]`
# ---------------------------------------------------------------------------


def test_zero_sai_como_celula_vazia(tmp_path: Path) -> None:
    matrix = _matrix(n_samples=2, n_features=2)
    values = matrix.values.copy()
    values[0, 0] = 0.0
    matrix = matrix.with_values(values, "teste")
    destination = tmp_path / "export.csv"

    report = write_metaboanalyst(matrix, destination)
    rows = _read(destination)

    assert rows[1][2] == "", "zero deveria sair vazio, que é a convenção do `[F-2]` para ausente"
    assert report.zeros_written_as_empty == 1


def test_ausente_sai_como_celula_vazia(tmp_path: Path) -> None:
    """`imputation="none"` chega até aqui, e `nan` é ausente pelo mesmo caminho."""
    matrix = _matrix(n_samples=2, n_features=2)
    values = matrix.values.copy()
    values[1, 1] = math.nan
    matrix = matrix.with_values(values, "teste")
    destination = tmp_path / "export.csv"

    report = write_metaboanalyst(matrix, destination)
    rows = _read(destination)

    assert rows[2][3] == ""
    assert report.zeros_written_as_empty == 1


def test_valor_positivo_nao_e_convertido(tmp_path: Path) -> None:
    """A direção que impede o formatador de esvaziar tudo.

    Sem ela, um formatador que devolvesse sempre `""` passaria os dois testes
    acima e produziria arquivo vazio de dado — o portão que nunca acusa, e
    indistinguível de sucesso na contagem de linhas.
    """
    matrix = _matrix(n_samples=2, n_features=2)
    values = np.full_like(matrix.values, 123.5)
    matrix = matrix.with_values(values, "teste")
    destination = tmp_path / "export.csv"

    report = write_metaboanalyst(matrix, destination)
    rows = _read(destination)

    assert report.zeros_written_as_empty == 0
    for row in rows[1:]:
        for cell in row[2:]:
            assert cell == "123.5"


def test_nenhuma_celula_escrita_tem_espaco(tmp_path: Path) -> None:
    """`[F-3]`: "there should not be spaces between numbers".

    A cláusula final do requisito é o que o torna perigoso: `1 600` é lido como
    `1`, **sem erro**. O modo de falha declarado pelo próprio terceiro é
    silencioso.
    """
    matrix = _matrix(n_samples=3, n_features=3)
    destination = tmp_path / "export.csv"

    write_metaboanalyst(matrix, destination)

    for row in _read(destination):
        for cell in row:
            assert " " not in cell, f"célula com espaço: {cell!r}"


def test_valor_negativo_levanta(tmp_path: Path) -> None:
    """`[F-2]` pede valores positivos, e matriz transformada por `log` tem negativos.

    Levantar em vez de escrever: arquivo silenciosamente não conforme é o desfecho
    que o critério existe para impedir, e ele não apareceria como erro.
    """
    matrix = _matrix(n_samples=2, n_features=2)
    values = matrix.values.copy()
    values[0, 0] = -1.0
    matrix = matrix.with_values(values, "teste")

    with pytest.raises(ValueError, match="negativos"):
        write_metaboanalyst(matrix, tmp_path / "export.csv")


# A alternativa que toda recusa por negativo oferece, como ela aparece na frase. É a mesma para os
# três desfechos, e é ela que torna a recusa acionável: o MetaboAnalyst aplica as duas operações.
ALTERNATIVA = "transformação e o escalonamento em `none`"


def test_negativo_do_escalonamento_nomeia_o_eixo_e_a_alternativa(tmp_path: Path) -> None:
    """A matriz vem de `scale_features` DE VERDADE, e não de uma proveniência escrita à mão.

    A frase lê a proveniência da matriz para saber qual eixo gerou os negativos; uma proveniência
    montada no teste afirmaria a leitura que eu imagino do produtor, e não a que ele grava — e um
    rename em `normalization.py` deixaria o escritor no ramo genérico sem nada reprovar.
    """
    escalonada = scale_features(_matrix(n_samples=4, n_features=3), "auto")
    assert (escalonada.values < 0).any(), "âncora: o escalonamento não gerou negativo"

    with pytest.raises(ValueError) as erro:
        write_metaboanalyst(escalonada, tmp_path / "export.csv")

    frase = str(erro.value)
    assert "escalonamento `auto`" in frase, frase
    assert "sempre" in frase, "a frase não diz que o escalonamento gera negativo por construção"
    assert ALTERNATIVA in frase and "MetaboAnalyst" in frase, frase


def test_negativo_do_log_nomeia_a_transformacao_e_a_alternativa(tmp_path: Path) -> None:
    """O `log` gera negativo onde a intensidade é menor que 1, e só aí — a frase diz isso."""
    pequena = _matrix(n_samples=4, n_features=3)
    pequena = pequena.with_values(np.asarray(pequena.values) / 100.0, "teste: abaixo de 1")
    transformada = transform(pequena, "log10")
    assert (transformada.values < 0).any(), "âncora: o `log10` não gerou negativo"

    with pytest.raises(ValueError) as erro:
        write_metaboanalyst(transformada, tmp_path / "export.csv")

    frase = str(erro.value)
    assert "transformação `log10`" in frase, frase
    assert "escalonamento `" not in frase, "a frase acusa um escalonamento que não correu"
    assert ALTERNATIVA in frase, frase


def test_negativo_sem_escalonamento_nem_log_nao_inventa_causa(tmp_path: Path) -> None:
    """Sem nenhum dos dois eixos na proveniência, a frase não os acusa nem oferece desligá-los.

    É o ramo `else`, que dispara por AUSÊNCIA: o negativo já estava no arquivo, ou veio da correção.
    Acusar o `log` aqui, como a frase anterior fazia sempre, mandaria procurar uma transformação
    que não foi pedida; e «transformação e escalonamento em `none`» seria oferecer o que já está.
    """
    matrix = _matrix(n_samples=2, n_features=2)
    values = matrix.values.copy()
    values[0, 0] = -1.0
    matrix = matrix.with_values(values, "teste")

    with pytest.raises(ValueError) as erro:
        write_metaboanalyst(matrix, tmp_path / "export.csv")

    frase = str(erro.value)
    assert "transformação `" not in frase and "escalonamento `" not in frase, frase
    assert ALTERNATIVA not in frase, "a frase oferece desligar eixos que não correram"
    assert "arquivo de entrada" in frase and "por conta própria" in frase, frase


# ---------------------------------------------------------------------------
# Guardas de construção do relatório
# ---------------------------------------------------------------------------


def test_nome_repetido_no_relatorio_levanta() -> None:
    """Unicidade é verificada e não suposta.

    Zero colisão é observação sobre os quatro arquivos do laboratório, e a
    ferramenta existe para receber dado que não é esse.
    """
    with pytest.raises(ValueError, match="nome repetido"):
        MetaboAnalystExport(kept=("a", "a"), zeros_written_as_empty=0, renamed=0)


def test_contagem_negativa_levanta() -> None:
    with pytest.raises(ValueError, match="negativas"):
        MetaboAnalystExport(kept=("a",), zeros_written_as_empty=-1, renamed=0)


# ---------------------------------------------------------------------------
# Os quatro modos reais — regra 1 do README, e a contagem é sobre ELES
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", MODES)
def test_os_modos_reais_exportam_com_todo_nome_conforme(mode: str, tmp_path: Path) -> None:
    """A medição é sobre estes arquivos, então é neles que se afirma.

    Contagem observada nos arquivos reais é testada **sobre eles**, nunca sobre
    matriz de factory: uma factory que reproduza o número está testando a factory.
    """
    matrix = load_matrix(measurements_path(mode), sample_list_path(mode), mode)
    destination = tmp_path / f"{mode}.csv"

    report = write_metaboanalyst(matrix, destination)
    rows = _read(destination)

    assert len(report.kept) == min(matrix.n_features, 5000), (
        "sem excesso a exportação é íntegra; com excesso mantém 5.000 (U3, `[TO-11.1]`)"
    )
    assert len(rows) == matrix.values.shape[0] + 1
    assert len(rows[0]) == len(report.kept) + 2, "cabeçalho: duas colunas de metadado e as mantidas"
    for name in rows[0][2:]:
        assert CONFORMING_NAME.match(name)
    for row in rows[1:]:
        assert CONFORMING_NAME.match(row[0])
        assert CONFORMING_NAME.match(row[1])


def test_no_pos_hilic_todas_as_features_sao_renomeadas(tmp_path: Path) -> None:
    """O achado: **6.035 de 6.035** violam o `[F-4]`.

    O número vai afirmado sobre o arquivo real e nomeando a população, porque é
    observação e não invariante — matriz nova tem outra contagem, e a asserção
    ficaria falsa por motivo que não é defeito.

    **A medição é sobre `feature_names`, e `renamed` é sobre o ARQUIVO.** As duas
    populações são diferentes : o `pos_hilic` excede 5.000, então o arquivo
    tem 5.000 nomes e a matriz tem 6.035. Afirmar 6.035 em `renamed` seria
    o número certo sobre a população errada.
    """
    matrix = load_matrix(measurements_path("pos_hilic"), sample_list_path("pos_hilic"), "pos_hilic")

    report = write_metaboanalyst(matrix, tmp_path / "pos_hilic.csv")

    assert matrix.n_features == 6035, "a população do `pos_hilic` na ingestão mudou; remedir"
    violating = [f for f in matrix.feature_names if not CONFORMING_NAME.match(f)]
    assert len(violating) == 6035, "as 6.035 features do `pos_hilic` violam o `[F-4]`"
    assert report.renamed == len(report.kept) == 5000, (
        "`renamed` é sobre o arquivo escrito, e o arquivo tem as 5.000 mantidas"
    )
    assert len(set(_sanitize_name(f) for f in matrix.feature_names)) == 6035, (
        "sanitizar deixou de dar 6.035 nomes distintos; a ausência de colisão é "
        "observação sobre este arquivo e a guarda de unicidade é o que a protege"
    )
