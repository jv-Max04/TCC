"""A espera pelo Palpite medido, para os arneses que conduzem a janela de verdade.

## Por que existe

Antes, o botão «Palpite» corria a árvore, que respondia em 0,4 s, e o roteiro de captura
`captura_telas_do_guia.py` dava-lhe poucos segundos de laço. O Palpite passou a ser medido: os três
corretores em 200 *features*, de 52,7 a 59,7 s por modo num Ryzen de 12 núcleos. **E a janela trava
toda correção enquanto ele corre** (`PALPITE_EM_CURSO`). Com a espera antiga, a captura sairia com o
Palpite a «medir» e a correção seguinte RECUSADA — com `EXIT=0`, porque o roteiro não conferia o que
a tela mostrava. O arnês diria que capturou a janela, e a janela que ele capturou não é a que se
queria mostrar.

## Os três desfechos, e por que são três

| Desfecho | Quando | O que o arnês faz |
|---|---|---|
| `CHEGOU` | o Palpite saiu das esperas da janela e ficou **em dia** | segue |
| `FALHOU` | a espera acabou sem Palpite em dia, ou nunca começou | para, com a frase da tela |
| `TETO` | o relógio passou do teto com o Palpite ainda a medir | para, com a frase da tela |

`FALHOU` e `TETO` levam à mesma ação, e por isso o arnês os trata igual; ficam separados porque a
causa é outra — a rota respondeu mal, ou a máquina é lenta — e quem lê o relato decide coisas
diferentes com cada um. **Um caso cai no balde errado, e vai declarado**: uma
leitura de progresso morta fora de `envia`, por exceção que não é `Exception`, passa por `_falha`,
que não encerra a espera — a chave fica nas esperas, e o desfecho é `TETO` e não `FALHOU`. É raro,
porque `envia` converte toda `Exception` em resposta.

## O que ele lê da janela, e um atributo é privado

`esperas_em_curso()` é público e é o que os testes da janela usam. **«Em dia» é `_palpite_em_dia`,
privado**: é o que `_palpite_chegou` marca só com `200` e `correction_proposta` no corpo, e não há
leitura pública dele. Ler o atributo é escolha deste arnês, e ela **falha fechada**: se o nome
mudar, `getattr` devolve `None`, o Palpite nunca fica «em dia», e o arnês para com `FALHOU` — nunca
segue sobre uma tela errada.

**Pré-condição:** a janela nunca mediu um Palpite antes desta chamada. O roteiro abre uma janela
nova; um Palpite anterior em dia faria `CHEGOU` antes do tempo.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any, Final

CHEGOU: Final = "chegou"
FALHOU: Final = "falhou"
TETO: Final = "teto"

# O teto da espera. 52,7 a 59,7 s por modo com 12 núcleos; com 4, perto de 3 min. 900 s cobre
# uma máquina lenta sem pendurar o roteiro, e o `neg_c18` do arnês é o menor dos quatro.
TETO_DO_PALPITE_S: Final = 900.0
# A espera tem de COMEÇAR — o `202` chegar e a janela agendar a releitura — dentro disto. O disparo
# lê os dois arquivos do `neg_c18` e responde em menos de 1 s; 60 s é folga larga, e sem este teto
# um disparo recusado esperaria os 900 s inteiros para dizer o que já se sabia.
TETO_DO_INICIO_S: Final = 60.0
# O passo do laço, em milissegundos: a janela relê o progresso de segundo em segundo.
PASSO_MS: Final = 500


def espera_o_palpite(
    janela: Any,
    bombeia: Callable[[int], None],
    *,
    chave: str,
    teto_s: float = TETO_DO_PALPITE_S,
    teto_do_inicio_s: float = TETO_DO_INICIO_S,
    relogio: Callable[[], float] = time.monotonic,
) -> str:
    """Gira o laço de eventos até o Palpite chegar. Devolve `CHEGOU`, `FALHOU` ou `TETO`.

    `bombeia(ms)` gira o laço da aplicação; `chave` é `CHAVE_DO_PALPITE` da janela, recebida e não
    importada para que o teste deste arquivo não precise do Qt. `relogio` é injetável pelo mesmo
    motivo.
    """
    inicio = relogio()
    viu_a_espera = False
    while True:
        em_curso = chave in janela.esperas_em_curso()
        em_dia = getattr(janela, "_palpite_em_dia", None) is not None
        viu_a_espera = viu_a_espera or em_curso
        if em_dia and not em_curso:
            return CHEGOU
        if viu_a_espera and not em_curso:
            return FALHOU
        decorrido = relogio() - inicio
        if not viu_a_espera and decorrido > teto_do_inicio_s:
            return FALHOU
        if decorrido > teto_s:
            return TETO
        bombeia(PASSO_MS)


def frase_do_palpite(janela: Any, chave: str) -> str:
    """O que a linha do Palpite diz na tela — para o relato, e para quem transcreve a captura."""
    rotulo = janela.leitura.rotulos.get(chave)
    return "" if rotulo is None else str(rotulo.text())
