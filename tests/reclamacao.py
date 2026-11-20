"""Disparar e reclamar, num auxiliar só.

**Por que existe.** Trocou-se `200` com o artefato no corpo por `202` com identificador nas
quatro rotas que correm corretor, e **seis** arquivos de teste falam com elas. Sem um auxiliar, cada
um repetiria o laço de consulta — e seis cópias do mesmo laço divergem no dia em que o contrato
mudar outra vez.

## A ÂNCORA está aqui dentro, e é ela que impede o auxiliar de esconder a regressão

Um auxiliar que devolvesse *"a resposta final, seja ela qual for"* aceitaria em silêncio uma rota
que voltasse a ser síncrona: o `200` com os bytes chegaria no disparo, as asserções de
byte-identidade continuariam verdes, e **a propriedade que a unidade existe para dar teria
desaparecido sem nada acusar**. É a forma mais convidativa do defeito — o mecanismo que passa a
medir menos e continua verde.

> **Disparo com desfecho de sucesso tem de ser `202`.** Qualquer outro `2xx` reprova aqui, com a
> mensagem a dizer que a rota deixou de ser assíncrona.

**O que NÃO é asserção:** desfecho de recusa. `422` do vocabulário, `409`, `503` do banco continuam
a acontecer no disparo por desenho (a fronteira não se moveu), e o auxiliar os devolve tal como
vieram — é o que permite aos testes de recusa usarem o mesmo caminho dos de sucesso.
"""

from __future__ import annotations

import threading
import time
from typing import Any

from fastapi import status

# Intervalo entre consultas. **Curto e não calibrado:** o que se espera é milissegundos de `none` e
# não os 209,9 s do SERRF, e o teto de tempo — não o de voltas — é o que decide a desistência.
INTERVALO_S = 0.05

# Teto de espera por trabalho, em segundos de parede. **600 e não 60**, porque os dois casos `slow`
# medem 86 s por ajuste do SERRF no `neg_c18` e um teto de minuto os faria
# reprovar por relógio em vez de por defeito — reprovação pelo motivo errado.
TETO_S = 600.0


def reclama(
    cliente: Any,
    rota: str,
    *,
    files: dict[str, Any],
    data: dict[str, Any],
    teto_s: float = TETO_S,
) -> Any:
    """Dispara, espera o trabalho terminar, e devolve a resposta da RECLAMAÇÃO.

    Quando o disparo não é sucesso — `422`, `409`, `503` — devolve o **disparo**, que é onde a
    recusa acontece. Assim o mesmo auxiliar serve o caminho feliz e o de recusa, e nenhum teste
    precisa de saber qual dos dois vai receber.
    """
    disparo = cliente.post(rota, files=files, data=data)
    if not 200 <= disparo.status_code < 300:
        return disparo

    assert disparo.status_code == status.HTTP_202_ACCEPTED, (
        f"{rota} respondeu {disparo.status_code} e não 202: a rota voltou a ser SÍNCRONA, e este "
        "auxiliar teria escondido a regressão devolvendo o corpo do disparo como se fosse o "
        "resultado reclamado (U10, R30)"
    )
    identificador = disparo.json()["identificador"]
    return espera(cliente, identificador, teto_s=teto_s)


def espera(cliente: Any, identificador: str, *, teto_s: float = TETO_S) -> Any:
    """Consulta `GET /trabalhos/{id}` até sair de `202`, e devolve a resposta.

    Levanta ao esgotar o teto em vez de devolver o último `202`: um teste que recebesse o progresso
    onde espera o resultado reprovaria com uma mensagem sobre o corpo errado, e a causa real —
    trabalho que não termina — ficaria escondida atrás dela.
    """
    limite = time.monotonic() + teto_s
    while True:
        resposta = cliente.get(f"/trabalhos/{identificador}")
        if resposta.status_code != status.HTTP_202_ACCEPTED:
            return resposta
        if time.monotonic() >= limite:
            raise AssertionError(
                f"o trabalho {identificador!r} não terminou em {teto_s:.0f} s de parede; "
                f"a última leitura foi {resposta.json()}"
            )
        threading.Event().wait(INTERVALO_S)
