"""O leitor da tabela de identificação do Progenesis.

As tabelas são feitas à mão, aqui, com nomes públicos de compostos e features no formato `RT_mz`
(a tabela real do laboratório não vira teste). O formato real da exportação é
desconhecido (Q1), e por isso o que se testa é o contrato do leitor — só as colunas escolhidas
sobrevivem, e cada recusa diz onde —, e não um cabeçalho presumido.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from refino.io.identificacao import (
    IdentificacaoMalformadaError,
    le_cabecalho,
    le_identificacao,
    sugere_colunas,
)
from refino.io.readers import load_matrix
from refino.io.writers import write_toserrf
from tests.conftest import measurements_path, sample_list_path, toserrf_path

MODO = "pos_hilic"
CABECALHO = ["Compound", "Description", "Accepted", "Score", "Amostra_01", "Amostra_02"]
LINHAS = [
    ["2.25_810.7739m/z", "Glycine", "1", "51.2", "1000", "1200"],
    ["3.10_146.0459m/z", "L-Glutamic acid", "1", "48.0", "830", "790"],
    ["3.10_146.0459m/z", "D-Glutamic acid", "0", "40.1", "830", "790"],
    ["4.02_118.0863m/z", "L-Valine", "", "39.9", "15", "17"],
]


def _escreve(caminho: Path, linhas: list[list[str]], *, sep: str = ",", cod: str = "utf-8") -> Path:
    texto = "\n".join(sep.join(c for c in ln) for ln in linhas) + "\n"
    caminho.write_bytes(texto.encode(cod))
    return caminho


def _le(caminho: Path, aceito: str | None = "Accepted") -> object:
    return le_identificacao(
        caminho, coluna_feature="Compound", coluna_composto="Description", coluna_aceito=aceito
    )


def test_ponto_e_virgula_latin1_e_bom_le_igual_a_virgula_utf8(tmp_path: Path) -> None:
    linhas = [CABECALHO, *LINHAS, ["5.00_100.0m/z", "Ácido cítrico", "1", "1", "1", "1"]]
    a = _escreve(tmp_path / "a.csv", linhas)
    b = _escreve(tmp_path / "b.csv", linhas, sep=";", cod="latin-1")
    c = _escreve(tmp_path / "c.csv", linhas, cod="utf-8-sig")
    ta, tb, tc = _le(a), _le(b), _le(c)
    assert ta.linhas == tb.linhas == tc.linhas  # type: ignore[attr-defined]
    assert tc.cabecalho[0] == "Compound", "o BOM grudou no nome da primeira coluna"  # type: ignore[attr-defined]
    assert ta.linhas[-1].composto == "Ácido cítrico"  # type: ignore[attr-defined]


def test_so_as_colunas_escolhidas_sobrevivem(tmp_path: Path) -> None:
    t = _le(_escreve(tmp_path / "t.csv", [CABECALHO, *LINHAS]))
    linha = t.linhas[0]  # type: ignore[attr-defined]
    assert set(vars(linha)) == {"feature", "composto", "aceito"}
    assert "1000" not in repr(t.linhas), "uma abundância por amostra sobreviveu à leitura"  # type: ignore[attr-defined]


def test_coluna_ausente_e_nomeada(tmp_path: Path) -> None:
    caminho = _escreve(tmp_path / "t.csv", [CABECALHO, *LINHAS])
    with pytest.raises(IdentificacaoMalformadaError, match="«Nome do composto»"):
        le_identificacao(caminho, coluna_feature="Compound", coluna_composto="Nome do composto")


def test_arquivo_vazio_e_so_cabecalho_dao_erros_distintos(tmp_path: Path) -> None:
    vazio = tmp_path / "vazio.csv"
    vazio.write_bytes(b"")
    with pytest.raises(IdentificacaoMalformadaError, match="vazio"):
        _le(vazio)
    so_cabecalho = _escreve(tmp_path / "cab.csv", [CABECALHO])
    with pytest.raises(IdentificacaoMalformadaError, match="só há o cabeçalho"):
        _le(so_cabecalho)


@pytest.mark.parametrize("bruto", [b"\n\n", b",,\n,,\n"], ids=["so_quebras", "so_separadores"])
def test_arquivo_nao_vazio_so_de_linhas_em_branco_e_vazio(tmp_path: Path, bruto: bytes) -> None:
    """O segundo caminho de «o arquivo está vazio»: bytes que existem e nenhuma linha com conteúdo.

    O teste de cima chega à mensagem pelo arquivo de zero bytes, que para antes do `csv`. Este passa
    pelo `csv` e pelo filtro de linhas em branco, e a recusa sai do cabeçalho que não veio — pelas
    duas funções públicas, a do cabeçalho e a da leitura.
    """
    caminho = tmp_path / "branco.csv"
    caminho.write_bytes(bruto)
    assert caminho.stat().st_size > 0, "âncora: a planta não é o arquivo de zero bytes"
    with pytest.raises(IdentificacaoMalformadaError, match=r"branco\.csv: o arquivo está vazio"):
        le_cabecalho(caminho)
    with pytest.raises(IdentificacaoMalformadaError, match=r"branco\.csv: o arquivo está vazio"):
        _le(caminho, aceito=None)


def test_nenhuma_linha_com_feature_e_composto_e_recusa_propria(tmp_path: Path) -> None:
    """O `else` dos dados: há linhas, e nenhuma tem as duas colunas escolhidas preenchidas.

    As três linhas plantadas têm conteúdo noutra coluna, então não são as linhas em branco que o
    leitor descarta antes — sem esta recusa, a leitura devolveria uma tabela com zero
    identificações. Par causal na mesma função: a mesma tabela com UMA linha completa
    é lida, e só ela sobrevive.
    """
    incompletas = [
        ["", "Glycine", "1", "51.2", "1000", "1200"],
        ["3.10_146.0459m/z", "", "1", "48.0", "830", "790"],
        ["  ", "   ", "0", "40.1", "830", "790"],
    ]
    caminho = _escreve(tmp_path / "t.csv", [CABECALHO, *incompletas])
    with pytest.raises(
        IdentificacaoMalformadaError,
        match=r"t\.csv: nenhuma linha tem feature e composto preenchidos",
    ):
        _le(caminho)
    com_uma = _escreve(tmp_path / "t1.csv", [CABECALHO, *incompletas, LINHAS[0]])
    t = _le(com_uma)
    assert [(ln.feature, ln.composto) for ln in t.linhas] == [  # type: ignore[attr-defined]
        ("2.25_810.7739m/z", "Glycine")
    ]


@pytest.mark.parametrize("nome", ['"Gly\ncine"', '"Gly\tcine"'])
def test_nome_com_quebra_de_linha_ou_tabulacao_e_recusado(tmp_path: Path, nome: str) -> None:
    caminho = tmp_path / "t.csv"
    caminho.write_text(
        "Compound,Description\n2.25_810.7739m/z," + nome + "\n", encoding="utf-8", newline=""
    )
    with pytest.raises(IdentificacaoMalformadaError, match="caractere de controle"):
        _le(caminho, aceito=None)


@pytest.mark.parametrize(
    "linha_suja",
    [
        # 0x92 é a aspa tipográfica do Excel em página de código do Windows; o `sniffer` cai para
        # latin-1 e ela chega como U+0092, que é C1 — o que o R recusa e o leitor deixava passar.
        b"2.25_810.7739m/z,2\x92-Deoxyguanosine",
        b"2.25\x85_810.7739m/z,Glycine",
        # Os dois separadores Unicode que o `[[:cntrl:]]` do R também casa, em UTF-8 válido.
        "2.25_810.7739m/z,Gly\u2028cine".encode(),
        "2.25_810.7739m/z,Gly\u2029cine".encode(),
    ],
)
def test_caractere_de_controle_que_o_r_recusa_e_acusado_aqui_com_a_linha(
    tmp_path: Path, linha_suja: bytes
) -> None:
    """Achado 13: o R recusa `[[:cntrl:]]` Unicode; a guarda daqui tem de recusar o mesmo conjunto.

    Par causal no mesmo teste: o arquivo idêntico, com o caractere trocado por pontuação que o R
    aceita (aspa tipográfica, ponto de exclamação invertido, `é`), é lido — e a recusa que a guarda
    estreita deixava escapar nomeia a linha.
    """
    sujo = tmp_path / "sujo.csv"
    sujo.write_bytes(b"Compound,Description\n" + linha_suja + b"\n")
    with pytest.raises(
        IdentificacaoMalformadaError, match=r"linha 2: o nome tem caractere de controle"
    ):
        _le(sujo, aceito=None)
    for aceitavel in ("\u2019", "\u00a1", "\u00e9"):
        limpo = tmp_path / f"limpo_{ord(aceitavel):04x}.csv"
        limpo.write_bytes(f"Compound,Description\n2.25_810.7739m/z,Gly{aceitavel}cine\n".encode())
        t = _le(limpo, aceito=None)
        assert t.linhas[0].composto == f"Gly{aceitavel}cine"  # type: ignore[attr-defined]


def test_campo_maior_que_o_limite_do_csv_e_recusa_propria_e_nao_csv_error(tmp_path: Path) -> None:
    """Par causal: a mesma tabela com o campo dentro do limite é lida; acima dele, é recusa nomeada.

    `csv.field_size_limit()` é o caminho de `csv.Error` que um arquivo real alcança — uma célula do
    Progenesis maior que 131072 caracteres —, e sem a guarda ele escapava cru pelo slot da tela
    (a irmã `io/fatores.py` tem o mesmo teste).
    """
    limite = csv.field_size_limit()
    assert limite > 10, "âncora: o limite do `csv` é maior que o campo curto do par"
    curto = tmp_path / "curto.csv"
    curto.write_text("Compound,Description\n1.0_100.0m/z," + "A" * 10 + "\n", encoding="utf-8")
    assert _le(curto, aceito=None).linhas[0].composto == "A" * 10  # type: ignore[attr-defined]
    longo = tmp_path / "longo.csv"
    longo.write_text(
        "Compound,Description\n1.0_100.0m/z," + "A" * (limite + 10) + "\n", encoding="utf-8"
    )
    with pytest.raises(IdentificacaoMalformadaError, match="o CSV não pôde ser lido"):
        _le(longo, aceito=None)


def test_feature_repetida_fica_como_candidatos_e_nao_e_sobrescrita(tmp_path: Path) -> None:
    t = _le(_escreve(tmp_path / "t.csv", [CABECALHO, *LINHAS]))
    candidatos = [ln for ln in t.linhas if ln.feature == "3.10_146.0459m/z"]  # type: ignore[attr-defined]
    assert [(c.composto, c.aceito) for c in candidatos] == [
        ("L-Glutamic acid", True),
        ("D-Glutamic acid", False),
    ]


@pytest.mark.parametrize("produtor", ["refino", "tools.R"])
def test_matriz_toserrf_e_reconhecida_e_nao_vira_cabecalho(tmp_path: Path, produtor: str) -> None:
    """A matriz corrigida, carregada no lugar da tabela, oferecia «NA, batch, A, A…».

    Os dois produtores reais do formato: o `write_toserrf` do refino — o que grava
    `saidas\\1-matrizes-corrigidas`, e foi um desses o carregado — e o `tools.R`. Antes da guarda
    as duas leituras abaixo eram aceitas: o cabeçalho voltava com «A» repetido, e confirmar
    `batch`/`A` devolvia 6.038 linhas no `pos_hilic` (medido em 08/10/2026).
    """
    if produtor == "refino":
        caminho = tmp_path / "corrigida.csv"
        write_toserrf(load_matrix(measurements_path(MODO), sample_list_path(MODO), MODO), caminho)
    else:
        caminho = toserrf_path(MODO)
    with pytest.raises(IdentificacaoMalformadaError, match="formato do SERRF"):
        le_cabecalho(caminho)
    with pytest.raises(IdentificacaoMalformadaError, match="formato do SERRF"):
        le_identificacao(caminho, coluna_feature="batch", coluna_composto="A")


def test_coluna_escolhida_repetida_e_recusada_e_a_repeticao_fora_da_escolha_nao(
    tmp_path: Path,
) -> None:
    """`index` lia a primeira das repetidas em silêncio. A repetição em si continua aceita."""
    cabecalho = ["Compound", "Description", "Amostra_01", "Amostra_01"]
    caminho = _escreve(tmp_path / "t.csv", [cabecalho, ["2.25_810.7739m/z", "Glycine", "10", "9"]])
    t = le_identificacao(caminho, coluna_feature="Compound", coluna_composto="Description")
    assert [(ln.feature, ln.composto) for ln in t.linhas] == [("2.25_810.7739m/z", "Glycine")]
    with pytest.raises(IdentificacaoMalformadaError, match="«Amostra_01» aparece 2 vezes"):
        le_identificacao(caminho, coluna_feature="Compound", coluna_composto="Amostra_01")


def test_espaco_nas_pontas_sai_e_aceito_sem_coluna_e_none(tmp_path: Path) -> None:
    linhas = [["Compound", "Description"], ["  2.25_810.7739m/z ", " Glycine  "]]
    t = _le(_escreve(tmp_path / "t.csv", linhas), aceito=None)
    assert t.linhas[0].feature == "2.25_810.7739m/z"  # type: ignore[attr-defined]
    assert t.linhas[0].composto == "Glycine"  # type: ignore[attr-defined]
    assert t.linhas[0].aceito is None  # type: ignore[attr-defined]


def test_a_sugestao_so_preenche_e_acha_as_colunas_comuns(tmp_path: Path) -> None:
    assert le_cabecalho(_escreve(tmp_path / "t.csv", [CABECALHO, *LINHAS])) == tuple(CABECALHO)
    assert sugere_colunas(CABECALHO) == {
        "feature": "Compound",
        "composto": "Description",
        "aceito": "Accepted",
    }
    assert sugere_colunas(["x", "y"]) == {"feature": None, "composto": None, "aceito": None}


def _grande(caminho: Path, *, byte_invalido: bool) -> Path:
    """Um CSV UTF-8 maior que a amostra do `sniffer` (64 KiB), com um `0xE9` cru no fim ou não."""
    corpo = "Compound,Description,Accepted\n" + "".join(
        f"{i}.00_{100 + i}.0000m/z,Glycine,1\n" for i in range(4000)
    )
    bruto = corpo.encode("utf-8")
    assert len(bruto) > 65536, "âncora: o byte inválido tem de cair depois da amostra"
    if byte_invalido:
        bruto += b"9.99_999.9999m/z,Caf\xe9,1\n"
    caminho.write_bytes(bruto)
    return caminho


def test_byte_invalido_depois_da_amostra_e_recusa_propria_e_nao_unicodeerror(
    tmp_path: Path,
) -> None:
    """Par causal: o mesmo arquivo grande, com e sem o byte; a irmã de `io/fatores.py`."""
    limpo = _grande(tmp_path / "limpo.csv", byte_invalido=False)
    assert len(_le(limpo, aceito=None).linhas) == 4000  # type: ignore[attr-defined]
    sujo = _grande(tmp_path / "sujo.csv", byte_invalido=True)
    with pytest.raises(IdentificacaoMalformadaError, match="não está em utf-8"):
        _le(sujo, aceito=None)


def test_o_cabecalho_le_so_a_primeira_linha(tmp_path: Path) -> None:
    """O cabeçalho para na primeira linha: o byte inválido do fim não é alcançado aqui."""
    sujo = _grande(tmp_path / "sujo.csv", byte_invalido=True)
    assert le_cabecalho(sujo) == ("Compound", "Description", "Accepted")
