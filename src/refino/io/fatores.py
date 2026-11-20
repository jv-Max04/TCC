"""Leitura do CSV de fatores de normalização.

`cli.py` e `api/rotas.py` são adaptadores finos sobre este módulo: cada um decide de onde vêm os
bytes e em que traduz o erro. Aqui trata-se só a leitura: decodificação, cabeçalho e forma das
linhas. A guarda de valor (fator positivo, finito, sem nomes repetidos) fica em
`PipelineSpec.__post_init__`, porque vale também para o registro reexecutado, que não passa por
este leitor.

O delimitador é o da primeira linha útil. O Excel em português salva CSV com `;` entre as colunas
e vírgula decimal: com `;`, a vírgula é decimal; com `,`, o ponto é decimal. Duas leituras erradas
dariam fator positivo e finito, que a guarda de valor deixaria passar, e são recusadas: com `;`,
um fator com ponto de milhar (`1.050`) seria lido como 1,05; com `,`, um fator com vírgula decimal
sem aspas (`A,1,5`) viraria três colunas, e o leitor ficaria com `1`.
"""

from __future__ import annotations

import csv
import re
from io import StringIO
from typing import Final

__all__ = [
    "FatoresMalformadosError",
    "le_fatores_de_bytes",
    "le_fatores_de_texto",
]


class FatoresMalformadosError(ValueError):
    """O CSV de fatores não tem a forma esperada.

    Classe própria porque as duas fronteiras a traduzem de formas diferentes: a CLI para
    `UsageError`, que vira código de saída 2, e a API para `HTTPException` 422.
    """


def le_fatores_de_bytes(bruto: bytes, *, origem: str) -> tuple[tuple[str, float], ...]:
    """Decodifica e analisa. `origem` é o rótulo que aparece nas mensagens.

    `utf-8-sig` e não `utf-8`: o Excel escreve BOM, e o laboratório exporta do Excel. Sem o
    `-sig` o BOM vira parte do nome da primeira amostra, que depois não casa com nenhuma linha
    da matriz.
    """
    try:
        texto = bruto.decode("utf-8-sig")
    except UnicodeDecodeError as erro:
        raise FatoresMalformadosError(
            f"{origem}: o arquivo de fatores não é UTF-8: {erro}"
        ) from erro
    return le_fatores_de_texto(texto, origem=origem)


def le_fatores_de_texto(texto: str, *, origem: str) -> tuple[tuple[str, float], ...]:
    """Lê duas colunas, nome da amostra e fator, com cabeçalho opcional.

    `csv.Error` não deriva de `ValueError`, então é apanhado aqui e convertido. Sem isso um campo
    acima de `csv.field_size_limit()` escaparia de quem trata `ValueError`.

    O cabeçalho é a primeira linha com conteúdo cujo segundo campo não é número, e não
    necessariamente a primeira linha: o Excel produz linha vazia no início ao exportar uma
    seleção. O número de linha nas mensagens é o físico, que é o que se vê no editor.

    As mensagens nomeiam o delimitador lido, para que quem salvou com o separador do outro idioma
    entenda por que o arquivo que tem duas colunas no Excel tem uma aqui.
    """
    pares: list[tuple[str, float]] = []
    primeira_util = True
    delimitador = _delimitador(texto)
    try:
        linhas = list(csv.reader(StringIO(texto), delimiter=delimitador))
    except csv.Error as erro:
        raise FatoresMalformadosError(
            f"{origem}: o arquivo de fatores não é um CSV legível: {erro}"
        ) from erro

    for numero, campos in enumerate(linhas, start=1):
        if not campos or not campos[0].strip():
            continue
        if len(campos) < 2:
            raise FatoresMalformadosError(
                f"{origem}:{numero}: esperava duas colunas — nome e fator, separadas por "
                f"{_NOME_DO_DELIMITADOR[delimitador]} — e achei {len(campos)}."
            )
        nome = campos[0].strip()
        valor = campos[1].strip()
        if (
            delimitador == ","
            and len(campos) > 2
            and _DIGITOS.fullmatch(valor)
            and (_DIGITOS.fullmatch(campos[2].strip()))
        ):
            raise FatoresMalformadosError(
                f"{origem}:{numero}: a linha de {nome!r} tem {valor!r} e {campos[2].strip()!r} "
                "em colunas separadas por vírgula. Se o fator usa vírgula decimal, separe as "
                "colunas com ponto e vírgula (;), como salva o Excel em português, ou escreva o "
                "fator com ponto decimal."
            )
        if delimitador == ";" and _MILHAR.fullmatch(valor):
            raise FatoresMalformadosError(_recusa_do_ponto(origem, numero, nome, valor))
        lido = _numero(valor, delimitador)
        if primeira_util and lido is None:
            primeira_util = False
            continue  # cabeçalho: a primeira linha útil cujo segundo campo não é número
        primeira_util = False
        if lido is None:
            raise FatoresMalformadosError(
                f"{origem}:{numero}: o fator de {nome!r} é {campos[1]!r}, que não é número."
            )
        pares.append((nome, lido))

    if not pares:
        raise FatoresMalformadosError(
            f"{origem}: nenhum fator lido. Um mapa vazio não normaliza nada."
        )
    return tuple(pares)


# O ponto de milhar do Excel em português: um a três dígitos sem zero à esquerda, grupos de três
# depois do ponto, e talvez a vírgula decimal. `0.050` não casa: é decimal com ponto. O sinal é
# aceito porque sem ele `+1.050` passaria como 1,05, a leitura mil vezes menor que esta guarda
# recusa.
_MILHAR: Final = re.compile(r"[+-]?[1-9]\d{0,2}(?:\.\d{3})+(?:,\d*)?")
_DIGITOS: Final = re.compile(r"[+-]?\d+")
_NOME_DO_DELIMITADOR: Final = {",": "vírgula", ";": "ponto e vírgula"}


def _recusa_do_ponto(origem: str, numero: int, nome: str, valor: str) -> str:
    """A mensagem da recusa de `1.125` num arquivo com `;`, com as duas leituras quando há duas.

    Num arquivo com `;` e ponto decimal o leitor aceita `1.5` e recusa `1.125`. Mandar só tirar o
    ponto faria quem usa ponto decimal gravar um fator mil vezes maior, positivo e finito, que a
    guarda de valor deixa passar. O leitor não sabe qual das duas o arquivo quer dizer, então a
    mensagem dá as duas.

    Só há leitura decimal com um ponto e sem vírgula: em `1.234.567` e em `1.050,5` o ponto só
    pode ser de milhar.
    """
    milhar = valor.replace(".", "")
    if valor.count(".") != 1 or "," in valor:
        return (
            f"{origem}:{numero}: o fator de {nome!r} é {valor!r}, com ponto de milhar. Num "
            "arquivo separado por ponto e vírgula a vírgula é a decimal, e o ponto não tem "
            f"leitura segura: escreva o número sem o separador de milhar ({milhar})."
        )
    decimal = valor.replace(".", ",")
    return (
        f"{origem}:{numero}: o fator de {nome!r} é {valor!r}, e o ponto tem duas leituras num "
        "arquivo separado por ponto e vírgula, onde a vírgula é a decimal: escreva "
        f"{milhar} se o ponto separa milhares, ou {decimal} se ele é a vírgula decimal."
    )


def _delimitador(texto: str) -> str:
    """O da primeira linha útil: `;` se ela o tiver, senão `,`."""
    for linha in texto.splitlines():
        if linha.strip():
            return ";" if ";" in linha else ","
    return ","


def _numero(texto: str, delimitador: str) -> float | None:
    """O fator, ou `None` se o campo não for número. Com `;`, a vírgula é a decimal."""
    bruto = texto.strip()
    if delimitador == ";":
        bruto = bruto.replace(",", ".")
    try:
        return float(bruto)
    except ValueError:
        return None
