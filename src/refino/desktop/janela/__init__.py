"""A janela Qt do aplicativo de mesa.

Nenhum pacote do núcleo importa Qt; fora deste subpacote, só o import tardio em
`arranque._corre_a_janela` o toca, para o grupo `desktop` continuar opcional. A dependência aponta
numa direção só: a janela importa `refino.interface.cliente`, e o cliente não sabe que a janela
existe.
Cada painel tem o seu arquivo, e `principal.py` os compõe.
"""
