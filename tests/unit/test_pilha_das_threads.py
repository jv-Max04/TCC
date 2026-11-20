"""A pilha das threads de trabalho continua posta.

**Este portão existe porque a alternativa é silêncio.** O que a linha do `conftest.py` compra é a
suíte completa **terminar** neste clone; sem ela o desfecho não é reprovação, é
`Windows fatal exception: access violation` com `EXIT=139` — e um processo morto a meio não deixa
relatório que alguém compare com a linha de base.

**A quebra foi medida nas duas direções, e a segunda célula é a que fecha a atribuição:** com o
trabalho da sessão GUARDADO no `stash`, os mesmos dois arquivos quebram na mesma. O defeito está
em `main`, e o par causal — mesma árvore, muda só a pilha — está no bloco do `conftest.py`.
"""

import threading

# **8 MB e não 64.** O portão defende a ORDEM DE GRANDEZA, não a escolha: quem quiser baixar de 64
# para 16 não deve ser reprovado por isso, e quem apagar a linha deve. O valor escolhido e o
# argumento inteiro vivem no `conftest.py`, que é onde a medição está.
PISO_DA_PILHA_BYTES = 8 * 1024 * 1024


def test_d384_a_pilha_das_threads_de_trabalho_continua_ampliada() -> None:
    """O `conftest.py` pôs a pilha, e ela é maior que o padrão da plataforma.

    **A âncora de não-vacuidade é o `!= 0`**, e ela não é decorativa: `threading.stack_size()`
    devolve **zero** quando ninguém a definiu — *"use o padrão da plataforma"* —, e zero é
    exactamente o estado em que a suíte quebra. Sem a âncora, a asserção do piso passaria a ser
    lida como satisfeita por um valor que significa o contrário do que ela quer.
    """
    posta = threading.stack_size()

    assert posta != 0, (
        "`threading.stack_size()` devolve 0 — ninguém a definiu, e a plataforma usa o padrão. "
        "É o estado em que a suíte completa morre com `access violation`. A linha saiu do "
        "`tests/conftest.py`?"
    )
    assert posta >= PISO_DA_PILHA_BYTES, (
        f"a pilha das threads caiu para {posta / 1024 / 1024:.1f} MB, abaixo do piso de "
        f"{PISO_DA_PILHA_BYTES / 1024 / 1024:.0f} MB. O bloco do `conftest.py` traz a medição: "
        f"com a pilha padrão, `tests/test_desktop_janela.py` mais `test_desktop_qualidade.py` "
        f"quebram com `EXIT=139`"
    )
