"""Nenhum arquivo versionado traz amostra real do estudo — nem dentro de um exemplo «sintético».

Os nomes de amostra reais usados como exemplo foram trocados por sintéticos, e se conferiu à
mão que não sobrou nome nem par real (e a conferência não viu os nomes em duas planilhas nem os
pares em dois testes). A troca manteve, de propósito, «o mesmo grupo»: o exemplo `ID99` ficou
com o NÚMERO DA AMOSTRA de uma amostra real — o número que precede o ID no nome — e o grupo dela,
e a verificação da release de 02/10/2026 achou isso dentro do executável publicado, que diz o
grupo daquela amostra. Uma conferência à mão não protege a próxima edição; este portão
protege.

**As três formas que ele recusa**, lidas dos quatro `Sample_List`:

- o nome completo de uma amostra de paciente;
- o par `ID<n>` e grupo real, em qualquer caixa, com zero à esquerda, colado por `_` ou separado
  por espaço, vírgula, aspas ou dois-pontos — a forma de um dicionário `{"ID<n>": "<grupo>"}`;
- o número de uma amostra real seguido do grupo REAL dela, com qualquer prefixo e qualquer ID — a
  forma que um exemplo sintético reproduz quando troca só o número do paciente.

O número da amostra NÃO é a ordem de injeção: ele vai de 1 a 61 e bate com a coluna `ID` do
`Sample_List` em 0 das 244 amostras. E o par número-grupo é lido dos quatro arquivos como
conjunto, sem supor que eles concordem.

**O que ele lê**: todo arquivo versionado que decodifica como UTF-8, e por dentro os documentos do
Office (`.xlsx`, `.docx`, `.pptx`), que são zip de XML — duas planilhas versionadas traziam nomes
completos de paciente e o portão, que pulava o que não era texto, ficou verde sobre elas
.

**O que fica de fora, declarado:** os dados de referência (`serrf/`, `data/reference/`), que SÃO o
estudo; as *fixtures* feitas à mão `tiny_*` e `corrupt_*`, que se decidiu manter, e o esquema
`TINY_<n>_ID<n>_<grupo>` delas onde aparece colado no texto; e os binários de `BINARIOS_PULADOS`,
cujo conteúdo não é texto — imagens, o PDF, as fontes TTF da interface e os arquivos de
dados serializados. Extensão indecodificável fora dessa lista reprova, em vez de ser pulada. As
mensagens de reprovação dão arquivo, linha e forma — nunca o nome, que é justamente o que não pode
sair.
E o portão salta, pelo regime `reference_data_absent`, num clone sem os `Sample_List`, que é o que
um clone novo encontra desde que `serrf/` saiu do git. Ali ele não protege edição
nenhuma.
"""

from __future__ import annotations

import html
import io
import re
import subprocess
import zipfile
from pathlib import Path

import pytest

from tests.conftest import sample_list_path

REPO = Path(__file__).resolve().parent.parent
MODOS = ("pos_hilic", "pos_c18", "neg_hilic", "neg_c18")
FORA = ("serrf/", "data/reference/", "data/fixtures/tiny_", "data/fixtures/corrupt_")
NOME_DE_PACIENTE = re.compile(r"(.+)_(\d+)_(ID\d+)_([A-Za-z]+)")
OOXML = frozenset({".xlsx", ".docx", ".pptx"})
# Medido em 02/10/2026 sobre os versionados: 18 `.png`, 5 `.rds`, 5 `.ser`, 2 `.npz`, 1 `.ico` e
# 1 `.pdf`. Nenhum é texto; o PDF não tem extrator na venv, e a revisão o leu pelos CMaps.
BINARIOS_PULADOS = frozenset({".png", ".ico", ".npz", ".rds", ".ser", ".pdf", ".ttf"})

PAR = re.compile(r"(?i)(?<![A-Za-z0-9])ID[\s_\-.:#]*0*(\d+)[\s_\-.,;:|\"'=]{1,6}([A-Za-z]+)")
NUMERO = re.compile(r"(?i)(?<![A-Za-z0-9])(?<!tiny_)0*(\d+)[_\-\s]+ID[^_\s\-]*[_\-\s]+([A-Za-z]+)")


def _amostras() -> list[tuple[int, int, str]]:
    """(número da amostra no nome, ID, grupo) de cada paciente dos quatro `Sample_List`."""
    saida = []
    for modo in MODOS:
        texto = sample_list_path(modo).read_text(encoding="utf-8-sig")
        for celula in re.split(r"[,;\t\r\n]", texto):
            m = NOME_DE_PACIENTE.fullmatch(celula.strip().strip('"'))
            if m:
                saida.append((int(m.group(2)), int(m.group(3)[2:]), m.group(4)))
    return saida


def vazamentos(texto: str, amostras: list[tuple[int, int, str]]) -> list[tuple[int, str]]:
    """(linha, forma) de cada amostra real em `texto`. Ver a docstring do módulo."""
    pares = {(ident, grupo.lower()) for _, ident, grupo in amostras}
    numeros = {(numero, grupo.lower()) for numero, _, grupo in amostras}
    achados = []
    for i, linha in enumerate(texto.splitlines(), 1):
        for m in NUMERO.finditer(linha):
            if (int(m.group(1)), m.group(2).lower()) in numeros:
                achados.append((i, "número da amostra real com o grupo real"))
        for m in PAR.finditer(linha):
            colado_ao_tiny = re.search(r"(?i)tiny_\d+_$", linha[: m.start()])
            if not colado_ao_tiny and (int(m.group(1)), m.group(2).lower()) in pares:
                achados.append((i, "par real ID_grupo"))
    return achados


def _texto_do_ooxml(bruto: bytes) -> list[tuple[str, str]]:
    """(parte, texto) de cada XML de um documento do Office, com as marcas tiradas de DUAS formas.

    Tirar as marcas sem separador junta as corridas de um `.docx`, que partem uma palavra em
    várias; com quebra de linha, separa as células de uma planilha, que juntas criariam palavras
    que não existem. Uma ocorrência que só uma das duas leituras acha conta igual.
    """
    saida = []
    with zipfile.ZipFile(io.BytesIO(bruto)) as z:
        for parte in z.namelist():
            if parte.endswith((".xml", ".rels")):
                xml = z.read(parte).decode("utf-8", errors="replace")
                for junta in ("\n", ""):
                    saida.append((parte, html.unescape(re.sub(r"<[^>]+>", junta, xml))))
    return saida


def _versionados() -> list[Path]:
    # `core.quotepath=off` e `-z`: sem eles, caminho com acento chega entre aspas e escapado, não
    # existe no disco, e a leitura falharia em silêncio.
    saida = subprocess.run(
        ["git", "-C", str(REPO), "-c", "core.quotepath=off", "ls-files", "-z"],
        capture_output=True,
        check=True,
    ).stdout.decode("utf-8")
    return [REPO / rel for rel in saida.split("\0") if rel and not rel.startswith(FORA)]


def ocorrencias(
    arquivo: Path, amostras: list[tuple[int, int, str]], raiz: Path = REPO
) -> tuple[list[str], str]:
    """As ocorrências de um arquivo e COMO ele foi lido: «texto», «ooxml» ou «pulado»."""
    rel = arquivo.relative_to(raiz).as_posix()
    sufixo = arquivo.suffix.lower()
    if sufixo in OOXML:
        vistas = {
            f"{rel} (parte {parte}) — {forma}"
            for parte, texto in _texto_do_ooxml(arquivo.read_bytes())
            for _, forma in vazamentos(texto, amostras)
        }
        return sorted(vistas), "ooxml"
    try:
        texto = arquivo.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        if sufixo in BINARIOS_PULADOS:
            return [], "pulado"
        return [f"{rel} — não decodifica e a extensão não está em BINARIOS_PULADOS"], "pulado"
    return [f"{rel}:{linha} — {forma}" for linha, forma in vazamentos(texto, amostras)], "texto"


def test_nenhum_arquivo_versionado_traz_amostra_real() -> None:
    amostras = _amostras()
    # A âncora vem antes: sem amostras lidas, a busca abaixo não acharia nada por vacuidade.
    assert len(amostras) == 244, f"âncora: são 244 amostras de paciente; lidas {len(amostras)}"
    arquivos = _versionados()
    # 826 listados e 786 lidos como texto em 02/10/2026; a âncora só pede lista não vazia.
    assert len(arquivos) > 500, f"âncora: só {len(arquivos)} arquivos versionados listados"
    lidos = {"texto": 0, "ooxml": 0, "pulado": 0}
    falhas = []
    for arquivo in arquivos:
        achadas, como = ocorrencias(arquivo, amostras)
        lidos[como] += 1
        falhas.extend(achadas)
    assert lidos["texto"] > 500, f"âncora: só {lidos['texto']} arquivos de texto lidos"
    assert not falhas, (
        f"{len(falhas)} ocorrência(s) de amostra real em arquivo versionado — troque por sintético "
        "com número de amostra e ID que não existem no estudo: " + "; ".join(falhas)
    )


def test_o_portao_acha_as_formas_e_nao_acha_o_sintetico() -> None:
    """O par causal, sobre texto montado em memória com as próprias amostras — nada é gravado."""
    amostras = _amostras()
    assert amostras, "âncora: há amostras"
    numero, ident, grupo = next(a for a in amostras if a[1] >= 10)
    prefixo = "HTLV_HILIC_PLASMA_MSe_POS"
    outro_grupo = next(g for _, _, g in amostras if g.lower() != grupo.lower())
    num, par = "número da amostra real com o grupo real", "par real ID_grupo"
    casos = {
        "nome completo": (f"x {prefixo}_{numero}_ID{ident}_{grupo} y", [num, par]),
        "numero e grupo, outro ID": (f"x {prefixo}_{numero}_ID99_{grupo} y", [num]),
        "numero e grupo, ID de marcador": (f"x {prefixo}_{numero}_ID<n>_{grupo} y", [num]),
        "numero e grupo, sem prefixo": (f'x "{numero}_ID99_{grupo}" y', [num]),
        "numero e grupo, em minusculas e com hifen": (
            f"x {prefixo.lower()}-{numero}-id99-{grupo.lower()} y",
            [num],
        ),
        "par sozinho": (f"x ID{ident}_{grupo} y", [par]),
        "par com zero a esquerda": (f"x ID0{ident}_{grupo} y", [par]),
        "par em minusculas": (f"x id{ident}_{grupo.lower()} y", [par]),
        "par como dicionario": (f'x {{"ID{ident}": "{grupo}"}} y', [par]),
        "par separado por espaco": (f"x paciente ID{ident} {grupo} y", [par]),
        "numero real, outro grupo": (f"x {prefixo}_{numero}_ID99_{outro_grupo} y", []),
        "numero inexistente, mesmo grupo": (f"x {prefixo}_0_ID99_{grupo} y", []),
        "esquema TINY": (f"x TINY_{numero}_ID{ident}_{grupo} y", []),
        "ID acima do estudo": (f"x ID{ident + 100}_{grupo} y", []),
    }
    for nome, (texto, esperado) in casos.items():
        assert [f for _, f in vazamentos(texto, amostras)] == esperado, nome


def test_o_portao_le_por_dentro_do_ooxml_e_reprova_o_binario_desconhecido(tmp_path: Path) -> None:
    """O par causal da leitura por dentro, sobre zips montados em memória.

    Uma planilha com o nome completo numa célula, e um documento com o mesmo nome partido em duas
    corridas, são acusados; a mesma planilha com o número inexistente não é. E um arquivo que não
    decodifica, com extensão fora de `BINARIOS_PULADOS`, reprova em vez de ser pulado.
    """
    amostras = _amostras()
    numero, ident, grupo = amostras[0]
    real = f"HTLV_C18_PLASMA_MSe_NEG_{numero}_ID{ident}_{grupo}"
    sintetico = f"HTLV_C18_PLASMA_MSe_NEG_0_ID99_{grupo}"
    metade = len(real) // 2

    def zipa(nome: str, parte: str, xml: str) -> Path:
        alvo = tmp_path / nome
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as z:
            z.writestr(parte, xml)
        alvo.write_bytes(buffer.getvalue())
        return alvo

    def acusa(alvo: Path) -> list[str]:
        return ocorrencias(alvo, amostras, tmp_path)[0]

    celula = '<c t="inlineStr"><is><t>{}</t></is></c><c><v>1</v></c>'
    corridas = "<w:p><w:r><w:t>{}</w:t></w:r><w:r><w:t>{}</w:t></w:r></w:p>"
    assert acusa(zipa("a.xlsx", "xl/worksheets/sheet1.xml", celula.format(real)))
    assert acusa(zipa("b.docx", "word/document.xml", corridas.format(real[:metade], real[metade:])))
    assert not acusa(zipa("c.xlsx", "xl/worksheets/sheet1.xml", celula.format(sintetico)))
    estranho = tmp_path / "d.bin"
    estranho.write_bytes(b"\xff\xfe\x00\x81")
    assert acusa(estranho) == ["d.bin — não decodifica e a extensão não está em BINARIOS_PULADOS"]
    pulado = tmp_path / "e.png"
    pulado.write_bytes(b"\xff\xfe\x00\x81")
    assert acusa(pulado) == []


@pytest.mark.parametrize("modo", MODOS)
def test_cada_sample_list_tem_61_amostras_de_paciente(modo: str) -> None:
    texto = sample_list_path(modo).read_text(encoding="utf-8-sig")
    n = sum(
        1
        for celula in re.split(r"[,;\t\r\n]", texto)
        if NOME_DE_PACIENTE.fullmatch(celula.strip().strip('"'))
    )
    assert n == 61, f"{modo}: {n} amostras de paciente, e são 61 por arquivo"
