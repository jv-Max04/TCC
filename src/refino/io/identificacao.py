"""Lê a tabela de identificação exportada do Progenesis, a entrada do enriquecimento.

O laboratório identifica compostos no Progenesis, depois da estatística, e exporta uma tabela em que
cada linha liga uma feature (`RT_mz`) a um nome de composto. Este módulo lê essa tabela e devolve
só as colunas que o usuário confirmou. Abundâncias por amostra, pontuações e as demais colunas não
sobrevivem à leitura: o que entra no enriquecimento é a ligação entre feature e composto.

O formato da exportação não é conhecido de antemão, então a leitura não presume nomes de coluna: o
usuário escolhe, e `sugere_colunas` só pré-preenche a escolha. Uma coluna de identificador HMDB ou
KEGG não é lida; o mapeamento adiante é por nome.
"""

from __future__ import annotations

import csv
import re
from collections.abc import Generator, Iterator, Sequence
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from refino.io.sniffer import sniff_dialect
from refino.io.writers import METADATA_ROWS

__all__ = [
    "IdentificacaoMalformadaError",
    "LinhaDeIdentificacao",
    "TabelaDeIdentificacao",
    "le_cabecalho",
    "le_identificacao",
    "sugere_colunas",
]

# O mesmo conjunto que o R recusa com `[[:cntrl:]]` no `r_service/vias_entrypoint.R`. No R
# portátil a classe casa U+0009, U+001F, U+007F, U+0080, U+0092, U+009F, U+2028 e U+2029, e não
# casa U+00A0; as faixas C0 e C1 (U+0080 a U+009F) do padrão abaixo são extensão desses pontos.
#
# A faixa C1 importa: um CSV salvo pelo Excel em página de código do Windows e decodificado como
# latin-1 pelo `sniffer` traz uma aspa tipográfica (0x92) como U+0092.
_CONTROLE: Final = re.compile(r"[\x00-\x1f\x7f-\x9f\u2028\u2029]")
_VERDADEIROS: Final = frozenset({"1", "true", "yes", "sim", "x", "aceito", "accepted", "y", "s"})


class IdentificacaoMalformadaError(ValueError):
    """A tabela não pode ser lida como o usuário a descreveu; a mensagem diz onde."""


@dataclass(frozen=True)
class LinhaDeIdentificacao:
    feature: str
    composto: str
    # `None` quando não há coluna de aceito; senão, se a linha é o candidato marcado como aceito.
    aceito: bool | None


@dataclass(frozen=True)
class TabelaDeIdentificacao:
    origem: str
    # O cabeçalho inteiro do arquivo, e não só as colunas escolhidas: a guarda do envio externo
    # recusa carga que repita um nome de coluna, e é daqui que ela o tira.
    cabecalho: tuple[str, ...]
    linhas: tuple[LinhaDeIdentificacao, ...]


def _linhas_csv(caminho: Path) -> Generator[list[str], None, None]:
    """As linhas não vazias do CSV, uma a uma, sem carregar o arquivo inteiro.

    A exportação do Progenesis traz a abundância por amostra e por isso é larga, e o cabeçalho
    precisa de uma linha só. O `sniffer` olha apenas uma amostra do início, então um byte inválido
    mais adiante é apanhado aqui e vira `IdentificacaoMalformadaError`.
    """
    if not caminho.is_file() or caminho.stat().st_size == 0:
        raise IdentificacaoMalformadaError(f"{caminho.name}: o arquivo está vazio")
    dialeto = sniff_dialect(caminho)
    # O `sniffer` decodifica em `utf-8`, e o Excel escreve BOM: sem o `-sig`, o BOM vira parte do
    # nome da primeira coluna e ela deixa de casar com a escolha do usuário.
    codificacao = "utf-8-sig" if dialeto.encoding == "utf-8" else dialeto.encoding
    try:
        with caminho.open(encoding=codificacao, newline="") as arquivo:
            for linha in csv.reader(arquivo, delimiter=dialeto.delimiter):
                if any(c.strip() for c in linha):
                    yield linha
    except UnicodeDecodeError as erro:
        raise IdentificacaoMalformadaError(
            f"{caminho.name}: o arquivo não está em {codificacao}: {erro}"
        ) from erro
    except csv.Error as erro:
        raise IdentificacaoMalformadaError(
            f"{caminho.name}: o CSV não pôde ser lido: {erro}"
        ) from erro


def _cabecalho(linhas: Iterator[list[str]], caminho: Path) -> tuple[str, ...]:
    primeira = next(linhas, None)
    if primeira is None:
        raise IdentificacaoMalformadaError(f"{caminho.name}: o arquivo está vazio")
    cabecalho = tuple(c.strip() for c in primeira)
    # A matriz no formato `toserrf` abre com a linha dos lotes, `NA,batch,A,A,…`. Lida como
    # cabeçalho, ela ofereceria rótulos de lote como colunas, e qualquer par escolhido devolveria
    # rótulos de amostra como identificações. O formato é reconhecido pela segunda célula, que é
    # onde o escritor põe o nome da linha.
    if len(cabecalho) >= 2 and cabecalho[1].casefold() == METADATA_ROWS[0]:
        raise IdentificacaoMalformadaError(
            f"{caminho.name}: isto é uma matriz de medições no formato do SERRF (a primeira linha "
            "é a dos lotes), e não a tabela de identificação. Aqui entra a tabela exportada do "
            "Progenesis que liga cada feature a um nome de composto."
        )
    return cabecalho


def le_cabecalho(caminho: Path) -> tuple[str, ...]:
    """O cabeçalho, para a tela oferecer a escolha de colunas. Lê só até a primeira linha."""
    with closing(_linhas_csv(caminho)) as linhas:
        return _cabecalho(linhas, caminho)


def sugere_colunas(cabecalho: Sequence[str]) -> dict[str, str | None]:
    """Sugere as colunas `feature`, `composto` e `aceito` a partir do cabeçalho.

    A busca é por palavras que aparecem em exportações de identificação; a tela mostra a sugestão
    e o usuário confirma. Coluna não achada fica `None`.
    """

    def acha(*padroes: str) -> str | None:
        for padrao in padroes:
            for nome in cabecalho:
                if re.search(padrao, nome, re.IGNORECASE):
                    return nome
        return None

    feature = acha(r"^compound$", r"feature", r"rt[_ ]?mz", r"^id$")
    composto = acha(
        r"^description$", r"accepted description", r"compound name", r"^name$", r"nome", r"descri"
    )
    aceito = acha(r"^accepted$", r"aceito", r"is accepted")
    return {"feature": feature, "composto": composto, "aceito": aceito}


def _celula(linha: Sequence[str], i: int) -> str:
    return linha[i] if i < len(linha) else ""


def _aceito(valor: str) -> bool:
    return valor.strip().casefold() in _VERDADEIROS


def le_identificacao(
    caminho: Path,
    *,
    coluna_feature: str,
    coluna_composto: str,
    coluna_aceito: str | None = None,
) -> TabelaDeIdentificacao:
    """Lê só as colunas confirmadas. Levanta `IdentificacaoMalformadaError` dizendo onde."""
    with closing(_linhas_csv(caminho)) as linhas:
        return _le_corpo(caminho, linhas, coluna_feature, coluna_composto, coluna_aceito)


def _le_corpo(
    caminho: Path,
    linhas: Iterator[list[str]],
    coluna_feature: str,
    coluna_composto: str,
    coluna_aceito: str | None,
) -> TabelaDeIdentificacao:
    cabecalho = _cabecalho(linhas, caminho)
    escolhidas = [coluna_feature, coluna_composto] + ([coluna_aceito] if coluna_aceito else [])
    for nome in escolhidas:
        if nome not in cabecalho:
            raise IdentificacaoMalformadaError(
                f"{caminho.name}: o cabeçalho não tem a coluna «{nome}»"
            )
        # `index` devolveria a primeira das repetidas sem dizer nada. A repetição em si não é
        # recusada, porque a exportação pode repetir nomes de amostra. O que não se lê é uma
        # coluna escolhida que não se sabe qual é.
        vezes = cabecalho.count(nome)
        if vezes > 1:
            raise IdentificacaoMalformadaError(
                f"{caminho.name}: a coluna «{nome}» aparece {vezes} vezes no cabeçalho, e não há "
                "como saber qual delas foi escolhida"
            )
    i_feature = cabecalho.index(coluna_feature)
    i_composto = cabecalho.index(coluna_composto)
    i_aceito = cabecalho.index(coluna_aceito) if coluna_aceito else None
    saida: list[LinhaDeIdentificacao] = []
    houve_linha = False
    for numero, linha in enumerate(linhas, start=2):
        houve_linha = True
        feature = _celula(linha, i_feature).strip()
        composto = _celula(linha, i_composto).strip()
        if not feature or not composto:
            continue
        # O nome vai ao R num arquivo de uma linha por nome: quebra de linha ou tabulação o
        # dividiria em dois, e qualquer outro caractere de controle o R recusa (`_CONTROLE`).
        # Acusar aqui permite dizer a linha.
        if _CONTROLE.search(composto) or _CONTROLE.search(feature):
            raise IdentificacaoMalformadaError(
                f"{caminho.name}, linha {numero}: o nome tem caractere de controle "
                "(quebra de linha, tabulação ou outro; um arquivo salvo em página de código do "
                "Windows pode trazer aspas e travessões que o leitor não decodificou)"
            )
        aceito = _aceito(_celula(linha, i_aceito)) if i_aceito is not None else None
        saida.append(LinhaDeIdentificacao(feature, composto, aceito))
    if not houve_linha:
        raise IdentificacaoMalformadaError(
            f"{caminho.name}: só há o cabeçalho, nenhuma linha de identificação"
        )
    if not saida:
        raise IdentificacaoMalformadaError(
            f"{caminho.name}: nenhuma linha tem feature e composto preenchidos"
        )
    return TabelaDeIdentificacao(caminho.name, cabecalho, tuple(saida))
