"""Poe `<meta charset="utf-8">` nos documentos HTML de entrega que nao o declaram.

## Por que isto existe

Quatro documentos de entrega nasceram **sem `<!doctype html>`**, de proposito: eles tem dois
destinos — publicados dentro do esqueleto de outra pagina, e abertos por duplo clique a partir
do disco. Sem `<!doctype>` tambem nao ha `<head>`, e por tabela nao havia charset nenhum. Os
arquivos sao UTF-8 e **nada o dizia**.

No primeiro destino o esqueleto do publicador declara. No segundo, o browser ADIVINHA — e
adivinha `windows-1252`. Medido ao abrir o arquivo por duplo clique:
`GUIA-do-aplicativo.html` renderiza `Como usar o programa, do comeÃ§o ao arquivo salvo`.

## O ORACULO E PRESENCA DO VALOR, e a primeira versao errou DUAS vezes

**Primeiro erro — posicao.** A especificacao do HTML define o *prescan* sobre os primeiros 1024
bytes, e a primeira versao fez disso o criterio. Medido: o Chrome honra a tag no byte **3002**;
so a **ausencia** quebra. Cinco paginas-sonda servidas com `?v=` para afastar o cache.

**Segundo erro — o VALOR.** A versao seguinte passou a procurar
o literal `<meta charset`, que termina **antes do `=`**. Entao `<meta charset="windows-1252">`
num arquivo UTF-8 passava as duas metades do portao — e renderiza **pior** que a ausencia,
porque uma declaracao explicita vence o palpite do browser.

> O oraculo agora exige o **valor**: `<meta charset="utf-8">` ou `utf8`, com espaco e aspas
> tolerados. `tests/test_charset_dos_documentos.py` le a mesma expressao daqui, para que as
> duas nao possam divergir.

## E ele RECUSA declarar o que nao mediu

Escrever `utf-8` num arquivo que nao decodifica como UTF-8 produz um arquivo que **mente** —
pior que o estado inicial. A primeira versao classificava a codificacao, imprimia a
classificacao e escrevia na mesma. Agora a classificacao decide.

**A tag entra DEPOIS do `<!doctype html>` quando ele existe.** Tres dos oito documentos de
`docs/` comecam com o doctype no byte 0; po-la antes empurraria o doctype para dentro do corpo
e levaria a pagina a *quirks mode*.
"""

from __future__ import annotations

import argparse
import pathlib
import re
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]

TAG = b'<meta charset="utf-8">\n'

DECLARA_UTF8 = re.compile(rb"""<meta\s+charset\s*=\s*["']?\s*utf-?8""", re.I)
"""A declaracao de UTF-8, com o VALOR e nao so o prefixo do atributo.

Tolera espaco a volta do `=` e aspas de qualquer tipo ou nenhumas, porque as tres formas sao
HTML valido; nao tolera outro valor, que era o buraco da versao anterior.
"""

_DOCTYPE = re.compile(rb"^\s*<!doctype[^>]*>[\r\n]*", re.I)
"""O `<!doctype>` inicial, quando existe — a tag tem de entrar DEPOIS dele."""


def declara(bruto: bytes) -> int:
    """O byte em que a declaracao de UTF-8 comeca, ou -1.

    **Devolve a posicao e nao um booleano de proposito**: quem conferir
    quer saber *onde*, porque foi uma teoria sobre a posicao que produziu o primeiro portao
    errado. Um `True`/`False` teria escondido justamente o numero que desfez a teoria.
    """
    achado = DECLARA_UTF8.search(bruto)
    return achado.start() if achado else -1


def com_a_tag(bruto: bytes) -> bytes:
    """Os bytes com a declaracao inserida no lugar certo.

    Depois do `<!doctype>` quando ele existe; no byte zero quando nao existe. Funcao separada
    para que o teste possa exercitar as duas formas sem tocar em disco.
    """
    if inicio := _DOCTYPE.match(bruto):
        return bruto[: inicio.end()] + TAG + bruto[inicio.end() :]
    return TAG + bruto


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="declara UTF-8 nos HTML de entrega")
    ap.add_argument("alvos", nargs="+", type=pathlib.Path)
    ap.add_argument("--confere", action="store_true", help="so reporta, nao escreve")
    args = ap.parse_args(argv)

    faltam = 0
    recusados = 0
    for alvo in args.alvos:
        bruto = alvo.read_bytes()
        try:
            bruto.decode("utf-8")
            e_utf8 = True
        except UnicodeDecodeError:
            e_utf8 = False

        posicao = declara(bruto)
        if posicao >= 0:
            print(f"  declara     {alvo.name:<44s} (byte {posicao})")
            continue

        faltam += 1
        if args.confere:
            print(f"  SEM CHARSET {alvo.name:<44s} ({'utf-8' if e_utf8 else 'NAO E UTF-8'})")
            continue

        # **RECUSA em vez de mentir.** Declarar `utf-8` num arquivo que nao decodifica como
        # UTF-8 deixa-o pior que estava: a declaracao explicita vence o palpite do browser.
        if not e_utf8:
            recusados += 1
            print(f"  RECUSADO    {alvo.name:<44s} (nao e UTF-8 — converta o arquivo primeiro)")
            continue

        # Escrita ATOMICA: um `write_bytes` interrompido a meio corromperia um documento que o
        # usuario recebe. O temporario nasce ao lado, para `replace()` ficar no mesmo volume.
        temporario = alvo.with_name(alvo.name + ".parcial")
        try:
            temporario.write_bytes(com_a_tag(bruto))
            temporario.replace(alvo)
        except BaseException:
            temporario.unlink(missing_ok=True)
            raise
        print(f"  ESCRITO     {alvo.name:<44s}")

    print(f"\n{len(args.alvos)} conferidos, {faltam} sem declaracao, {recusados} recusados")
    # **Recusa reprova em QUALQUER modo.** A versao anterior so sabia reprovar com `--confere`,
    # entao um arquivo que nao era UTF-8 saia com 0 e ninguem via.
    return 1 if (recusados or (faltam and args.confere)) else 0


if __name__ == "__main__":
    raise SystemExit(main())
