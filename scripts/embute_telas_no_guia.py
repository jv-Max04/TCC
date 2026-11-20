#!/usr/bin/env python3
"""Embute as capturas de `out/telas/` nas sete figuras do `docs/GUIA-do-aplicativo.html`.

## Por que existe

**O passo 0 do roteiro de entrega estava pela metade, e a metade que faltava era manual.**
`scripts/captura_telas_do_guia.py` produz os PNG e diz, em prosa, *«ABRA AS N E OLHE antes de
embutir»* — e depois disso alguém colava o base64 à mão. Um passo manual num roteiro de cinco
etapas é o que fica por fazer, e o desfecho é o GUIA a mostrar uma janela que já não existe, com
três pares de botões que já tinham virado um.

## O mapeamento é DECLARADO e conferido, e não derivado da ordem por acaso

As sete figuras não são sete capturas distintas: **duas aparecem duas vezes**, porque as abas
fundiram — o próprio GUIA o declara. Derivar o mapeamento da ordem dos arquivos poria a captura
errada na figura errada em silêncio, que é pior que a captura velha: ela ao menos foi verdade um
dia.

Por isso a tabela abaixo é escrita, o número de figuras é **afirmado** antes de qualquer troca, e
uma divergência **para** em vez de adivinhar.

## O que ele NÃO faz, declarado

Não captura — quem o faz é o script irmão, e a separação é deliberada: a captura abre uma janela
nativa e leva minutos; esta troca é textual e roda em segundos. E **não olha para as imagens**:
olhar é do humano, e o script irmão imprime o pedido.

## Uso

    .venv/Scripts/python.exe scripts/embute_telas_no_guia.py [--telas out/telas] [--conferir]

`--conferir` não escreve: diz se o GUIA já está em dia com as capturas do disco, e é o modo que o
roteiro de entrega usa para decidir se o passo 0 precisa de correr.
"""

from __future__ import annotations

import argparse
import base64
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]

REPO = Path(__file__).resolve().parent.parent
GUIA = REPO / "docs" / "GUIA-do-aplicativo.html"

# **A ordem é a do DOCUMENTO, e a tabela nomeia o que cada figura mostra.** As duas repetições
# estão declaradas no próprio GUIA: *"duas telas aparecem duas vezes: «Preparar o estudo» ilustra
# os passos 1 e 2, e «Correção» ilustra os passos 3 e 6."* A figura do passo 6 é a aba «Correção»
# porque é lá que o download do resultado mora (o «Resultado» dela).
#
# As capturas das abas sem figura NÃO entram, e a ausência é declarada no GUIA: *"As abas sem
# captura são «Normalização», «Análise estatística», «Multiverso (beta)» e «Vias metabólicas
# (beta)»"*. Um mapeamento que as incluísse mudaria o documento sem ninguém ter decidido.
#
# **OS NOMES SÃO O ÍNDICE DA ABA.** O script irmão grava `f"{indice}-{aba.nome_da_captura}.png"`,
# então uma aba que entra ou sai antes de outra renomeia a captura desta, e a tabela tem de
# acompanhar. Uma tabela desatualizada para com «falta N-….png» em vez de embutir a captura errada.
# As abas de etapa são nomeadas pela etapa, porque não têm painel; a última aba chama-se «Métodos e
# limites», e o nome do arquivo continua a ser o da captura, `recorte`.
FIGURAS: tuple[tuple[str, str], ...] = (
    ("0-arquivos-leitura.png", "a aba «Preparar o estudo», no passo 1"),
    ("0-arquivos-leitura.png", "a MESMA, no passo 2 — as duas metades são uma aba só"),
    ("1-etapa-1.png", "a aba «Correção», no passo 3"),
    ("1-etapa-1.png", "a MESMA, no passo 6 — o «Resultado» dela baixa o arquivo"),
    ("5-qualidade.png", "a aba «Qualidade do resultado» depois da execução"),
    ("6-graficos.png", "a aba «Gráficos», com a figura desenhada do dado real"),
    ("8-recorte.png", "a declaração de escopo, na aba «Métodos e limites»"),
)

PADRAO = re.compile(r'src="data:image/png;base64,([A-Za-z0-9+/=]+)"')


def main() -> int:
    analisador = argparse.ArgumentParser(description=__doc__)
    analisador.add_argument("--telas", default="out/telas")
    analisador.add_argument("--conferir", action="store_true")
    argumentos = analisador.parse_args()

    telas = REPO / argumentos.telas
    texto = GUIA.read_text(encoding="utf-8")
    achadas = PADRAO.findall(texto)

    # **A âncora de não-vacuidade, e ela vem ANTES de qualquer troca.** Um `regex` que
    # deixasse de casar daria lista vazia, e uma substituição sobre lista vazia é um sucesso que
    # não fez nada.
    if len(achadas) != len(FIGURAS):
        print(
            f"FALHA: o GUIA tem {len(achadas)} imagens embutidas e a tabela declara "
            f"{len(FIGURAS)}. O documento mudou de forma, e adivinhar o mapeamento poria a "
            "captura errada na figura errada em silêncio",
            file=sys.stderr,
        )
        return 1

    novas: list[str] = []
    for nome, _o_que_mostra in FIGURAS:
        caminho = telas / nome
        if not caminho.is_file():
            print(
                f"FALHA: falta {caminho}. Corra `scripts/captura_telas_do_guia.py`",
                file=sys.stderr,
            )
            return 1
        novas.append(base64.b64encode(caminho.read_bytes()).decode("ascii"))

    iguais = sum(1 for antiga, nova in zip(achadas, novas, strict=True) if antiga == nova)
    print(f"figuras: {len(FIGURAS)}  já em dia: {iguais}  a trocar: {len(FIGURAS) - iguais}")
    for (nome, o_que), antiga, nova in zip(FIGURAS, achadas, novas, strict=True):
        marca = "=" if antiga == nova else "TROCA"
        print(f"  {marca:<5} {nome:<28} {o_que}")

    if argumentos.conferir:
        if iguais == len(FIGURAS):
            print("\no GUIA está em dia com as capturas do disco")
            return 0
        print("\no GUIA está DESACTUALIZADO: corra sem `--conferir`", file=sys.stderr)
        return 1

    if iguais == len(FIGURAS):
        print("\nnada a fazer")
        return 0

    # A troca é posicional e de uma passagem: o `regex` casa na ordem do documento, e o iterador
    # devolve a substituição correspondente.
    fila = iter(novas)
    novo_texto = PADRAO.sub(lambda _: f'src="data:image/png;base64,{next(fila)}"', texto)
    GUIA.write_text(novo_texto, encoding="utf-8")
    print(f"\n{len(FIGURAS) - iguais} figura(s) trocada(s) em {GUIA.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
