#!/usr/bin/env python3
"""Desenha a tela de arranque que o carregador do PyInstaller mostra — 08/09/2026.

## Por que ela existe, e o que ela NAO conserta

**O arranque a frio custa 84,1 s numa maquina que nunca viu estes bytes**, medido em 08/09/2026 em
tres copias independentes da pasta entregue. Ler os 400 MB custa 3 s; o resto e o antivirus a
varrer arquivo a arquivo na primeira execucao. Devolver o PYZ ao aplicativo ataca a
causa; **esta tela nao reduz um segundo**.

O que ela tira e o dano: durante esse tempo **nada aparece na tela**, e o LEIA-ME tem um paragrafo
inteiro a pedir que ninguem feche nem clique duas vezes. Um pedido em prosa, num arquivo que a
pessoa ainda nao abriu, contra uma area de trabalho que nao mudou.

## Por que o carregador e nao o Qt

A tela do PyInstaller e desenhada pelo **bootloader**, em C, antes de o Python arrancar. Uma
`QSplashScreen` precisaria de o PySide6 ja estar importado — e sao 92 MB e 145 arquivos, dentro
exactamente da janela que se quer cobrir.

## O texto e curto de proposito

Tres linhas: o que e, que a primeira vez demora, e que nao se deve fechar. Quem le isto esta a
olhar para um retangulo ha um minuto, e o que precisa de saber e que o computador nao travou.

## Uso

    .venv/Scripts/python.exe scripts/gera_splash.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]

REPO = Path(__file__).resolve().parent.parent
ALVO = REPO / "src" / "refino" / "desktop" / "recursos" / "splash.png"

# **560 e nao 480, e o numero veio de OLHAR.** Com 480 a ultima linha saiu cortada em
# «abre em segund…» — e o script tinha acabado de imprimir o pedido para eu olhar. A guarda
# abaixo mede a linha mais larga e reprova, para a proxima edicao do texto nao repetir isto.
LARGURA, ALTURA = 560, 236

# **A ALTURA subiu de 200 para 236 em 08/09/2026, e a faixa nova tem dono.** `ETAPA_Y` e a linha
# em que o carregador escreve o texto que MUDA — `pyi_splash.update_text` —, e o `.spec` tem de
# apontar `text_pos` para exactamente este par. Se os dois divergirem, o texto sai por cima da
# prosa fixa ou fora da imagem, e o desfecho e um PNG valido: «existe e tem pixels» nao distingue.
ETAPA_X, ETAPA_Y = 32, 196
FUNDO = (28, 30, 34)
TINTA = (232, 234, 238)
TINTA2 = (150, 155, 165)
REALCE = (108, 168, 255)


def _fonte(tamanho: int, negrito: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """A fonte do sistema, com recuo declarado.

    **O recuo para a fonte embutida do Pillow nao e silencioso**: ela desenha, mas pequena e feia.
    Se acontecer, a imagem sai visivelmente diferente, e olhar para ela antes de embutir e
    quando isso aparece.
    """
    for nome in ("segoeui.ttf", "arial.ttf", "DejaVuSans.ttf"):
        caminho = Path("C:/Windows/Fonts") / nome
        if caminho.is_file():
            return ImageFont.truetype(str(caminho), tamanho)
    return ImageFont.load_default()


def main() -> int:
    imagem = Image.new("RGB", (LARGURA, ALTURA), FUNDO)
    desenho = ImageDraw.Draw(imagem)

    # A moldura fina: separa a tela do fundo da area de trabalho, que pode ser de qualquer cor.
    desenho.rectangle([0, 0, LARGURA - 1, ALTURA - 1], outline=(60, 64, 70))
    desenho.rectangle([0, 0, LARGURA - 1, 3], fill=REALCE)

    icone = REPO / "src" / "refino" / "desktop" / "recursos" / "refino.ico"
    if icone.is_file():
        with Image.open(icone) as ico:
            ico.load()
            miniatura = ico.convert("RGBA").resize((56, 56), Image.LANCZOS)
            imagem.paste(miniatura, (32, 40), miniatura)

    desenho.text((108, 44), "refino", font=_fonte(30, True), fill=TINTA)
    desenho.text(
        (108, 84), "refino quimiométrico de metabolômica LC-MS", font=_fonte(13), fill=TINTA2
    )

    desenho.text(
        (32, 130),
        "A PRIMEIRA abertura demora — o Windows está a verificar os arquivos.",
        font=_fonte(13),
        fill=TINTA,
    )
    desenho.text(
        (32, 152),
        "Não feche e não clique de novo. Da segunda vez em diante abre em segundos.",
        font=_fonte(13),
        fill=TINTA2,
    )

    # **A régua que separa a prosa FIXA do texto que MUDA.** Sem ela as duas leem-se como um
    # parágrafo só, e a linha que se move parece erro de renderização em vez de progresso.
    desenho.line([(32, 178), (LARGURA - 32, 178)], fill=(60, 64, 70))

    # **A GUARDA DE TRANSBORDO, e ela nasceu de uma linha cortada.** O `textlength` mede a linha
    # desenhada; a margem direita e a mesma da esquerda. Sem isto, editar o texto corta-o outra vez
    # e o desfecho e um arquivo PNG valido — «existe e tem pixels» nao distingue.
    linhas = [
        (32, "A PRIMEIRA abertura demora — o Windows está a verificar os arquivos.", 13),
        (32, "Não feche e não clique de novo. Da segunda vez em diante abre em segundos.", 13),
        (108, "refino quimiométrico de metabolômica LC-MS", 13),
    ]
    transbordo = [
        (texto, round(x + desenho.textlength(texto, font=_fonte(t)) - (LARGURA - 32)))
        for x, texto, t in linhas
        if x + desenho.textlength(texto, font=_fonte(t)) > LARGURA - 32
    ]
    if transbordo:
        for texto, sobra in transbordo:
            print(f"FALHA: «{texto[:50]}…» transborda {sobra} px", file=sys.stderr)
        return 1

    # **A FAIXA DA ETAPA FICA VAZIA NO PNG, e isso e deliberado.** Quem a escreve e o
    # `pyi_splash.update_text`, do lado do Python; desenha-la aqui poria texto morto por baixo do
    # vivo. A guarda abaixo afirma que ela EXISTE — que ha altura reservada por baixo da regua —,
    # porque uma imagem sem essa folga aceita o `text_pos` do `.spec` sem se queixar e desenha a
    # etapa fora da tela.
    if ALTURA - ETAPA_Y < 16:
        print(f"FALHA: sobram {ALTURA - ETAPA_Y}px por baixo de ETAPA_Y={ETAPA_Y}", file=sys.stderr)
        return 1

    ALVO.parent.mkdir(parents=True, exist_ok=True)
    imagem.save(ALVO, "PNG")
    tamanho = f"{ALVO.stat().st_size:,}".replace(",", ".")
    print(f"{ALVO.relative_to(REPO)}  {tamanho} bytes  {LARGURA}x{ALTURA}")
    print(f"a etapa e escrita em ({ETAPA_X}, {ETAPA_Y}); o `.spec` diz o mesmo par em `text_pos`")
    print("\nABRA E OLHE antes de congelar: se o texto sair pequeno ou serrilhado, o Pillow caiu")
    print("para a fonte embutida e a tela vai assim na entrega.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
