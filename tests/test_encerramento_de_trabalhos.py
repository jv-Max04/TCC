"""O encerramento de trabalhos vivos, exercitado nos TRÊS desfechos.

## Por que este arquivo existe

`encerra_trabalhos_vivos` nasceu em 06/09/2026 como corpo de uma fixture de teardown, contra um
vazamento de thread: `_despacha` lança `threading.Thread(..., daemon=True)` e ninguém a junta, então
um teste que dispara rota longa e confere só o `202` deixa o corretor a correr por dentro dos testes
seguintes — e o SERRF custa 209,9 s no `pos_hilic`.

**Ele passou uma sessão inteira sem nunca correr.** Medido com sonda em 07/09/2026:
`test_api_execucoes.py` e `test_cancelamento_de_trabalhos.py` inteiros, e a lista de threads vivas
esteve **sempre vazia** — nem o cancelamento, nem a junção, nem o `esquece`, nem o aviso alguma vez
dispararam. E a sessão que o escreveu chegou a atribuir-lhe um conserto que ele não fez.

> **A causa não foi descuido: era inalcançável.** Uma fixture de teardown não é chamável, e um teste
> não afirma sobre a própria teardown. É por isso que o conserto foi **extrair a função**, o
> mesmo idioma que `tests/unit/test_identidade_do_corretor.py` levou no mesmo dia.

**Três desfechos, e «dois» não seria resposta:** não há trabalho vivo; há e ele para; há e
ele **não** para. O terceiro é o que produz o aviso, e sem ele um trabalho que ignore o cancelamento
seria indistinguível de não haver trabalho nenhum.
"""

import threading
import warnings
from typing import Any

import pytest

from conftest import encerra_trabalhos_vivos


class _TrabalhoFalso(threading.Thread):
    """Uma thread com o NOME de um trabalho, que para quando o registro pede — ou não para.

    **Não usa a API**, e a escolha vai declarada: disparar `POST /matrizes/corrigir` com `serrf`
    daria uma thread verdadeira e custaria 209,9 s por caso. O que se mede aqui é o
    **encerrador** — se ele acha a thread pelo nome, pede o cancelamento pelo identificador certo,
    espera, e limpa o registro. Para isso a thread precisa de ter o nome e de consultar a bandeira,
    e é exactamente isso que ela faz.

    **A ponte é a mesma que a produção usa:** `TRABALHOS.quer_cancelar(identificador)`. Um duplo que
    parasse por outro caminho mediria o arnês.
    """

    def __init__(self, modulo: Any, identificador: str, *, obedece: bool) -> None:
        super().__init__(name=f"{modulo.PREFIXO_DA_THREAD_DE_TRABALHO}{identificador}", daemon=True)
        self._modulo, self._identificador, self._obedece = modulo, identificador, obedece
        self.entrou = threading.Event()
        self.solta = threading.Event()

    def run(self) -> None:
        self.entrou.set()
        while not self.solta.is_set():
            if self._obedece and self._modulo.TRABALHOS.quer_cancelar(self._identificador):
                return
            self.solta.wait(0.01)


@pytest.fixture
def rotas() -> Any:
    """O módulo das rotas. Importado aqui e não no topo, pelo motivo do `conftest`."""
    from refino.api import rotas as modulo

    return modulo


def test_d386_sem_trabalho_vivo_o_encerrador_nao_faz_nada(rotas: Any) -> None:
    """O ramo `else`, e ele vem primeiro **de propósito**.

    O ramo que dispara por **ausência** de condição é o mais fácil de ficar
    inerte sem ninguém notar, porque não há o que lhe plantar. Aqui ele é o estado normal da suíte
    — foi o único que a sonda de 07/09/2026 observou em dois arquivos inteiros.
    """
    vivas_antes = [
        t.name
        for t in threading.enumerate()
        if t.name.startswith(rotas.PREFIXO_DA_THREAD_DE_TRABALHO) and t.is_alive()
    ]
    assert not vivas_antes, (
        f"âncora: já há trabalho vivo e este caso não mede o vazio: {vivas_antes}"
    )

    assert encerra_trabalhos_vivos(rotas) == (), "achou trabalho onde não há nenhum"


def test_d386_o_encerrador_cancela_junta_e_esquece_o_trabalho_vivo(rotas: Any) -> None:
    """O desfecho central: há trabalho vivo, e depois não há — nem no processo, nem no registro.

    **As três asserções são três coisas diferentes**, e colapsá-las esconderia a que falha:
    o encerrador **relatou** o identificador; a **thread** parou; e o **registro** já não o
    conhece. A terceira é a que impede o cadáver reclamável — um trabalho cancelado que ficasse no
    registro faria o teste seguinte com a mesma entrada receber `reclamado=True`, e nenhuma thread
    nasceria para o mover.
    """
    identificador = rotas.TRABALHOS.abre(2)
    trabalho = _TrabalhoFalso(rotas, identificador, obedece=True)
    trabalho.start()
    assert trabalho.entrou.wait(timeout=5), "âncora: a thread falsa nunca arrancou"
    assert rotas.TRABALHOS.le(identificador) is not None, (
        "âncora: o registro não conhece o trabalho"
    )

    try:
        tratados = encerra_trabalhos_vivos(rotas, teto_s=5.0)
    finally:
        trabalho.solta.set()
        trabalho.join(timeout=5)

    assert tratados == (identificador,), f"o encerrador não relatou o trabalho vivo: {tratados}"
    assert not trabalho.is_alive(), "a thread continua viva depois do encerramento"
    assert rotas.TRABALHOS.le(identificador) is None, (
        "o trabalho ficou no registro depois de encerrado, e um trabalho no registro é "
        "RECLAMÁVEL: o próximo disparo da mesma entrada não lançaria thread nenhuma"
    )


def test_d386_o_trabalho_que_ignora_o_cancelamento_produz_aviso(rotas: Any) -> None:
    """O terceiro desfecho, e é o que o silêncio esconderia.

    Uma thread que não olha a bandeira sobrevive à junção. **Sem o aviso ela seria indistinguível
    de não haver thread nenhuma** — o encerrador devolveria o identificador na mesma, e o próximo
    teste herdaria o trabalho a correr sem nada o dizer.

    **O teto é de fração de segundo e vem por parâmetro**, que é o motivo de `teto_s` existir: com
    os 10 s de produção este caso custaria dez segundos para medir uma linha.
    """
    identificador = rotas.TRABALHOS.abre(2)
    trabalho = _TrabalhoFalso(rotas, identificador, obedece=False)
    trabalho.start()
    assert trabalho.entrou.wait(timeout=5), "âncora: a thread falsa nunca arrancou"

    try:
        with warnings.catch_warnings(record=True) as apanhados:
            warnings.simplefilter("always")
            tratados = encerra_trabalhos_vivos(rotas, teto_s=0.2)
    finally:
        trabalho.solta.set()
        trabalho.join(timeout=5)

    assert tratados == (identificador,), f"o encerrador não relatou o trabalho: {tratados}"
    assert len(apanhados) == 1, (
        f"esperava UM aviso sobre a thread que não parou, vieram {len(apanhados)}: "
        f"{[str(a.message) for a in apanhados]}"
    )
    assert identificador in str(apanhados[0].message), (
        f"o aviso não nomeia o trabalho que ficou a correr: {apanhados[0].message}"
    )


def test_d386_o_par_causal_o_aviso_nao_sai_quando_o_trabalho_obedece(rotas: Any) -> None:
    """A outra metade do par: um encerrador que avisasse sempre passaria no caso de cima.

    É a assimetria — ver o mecanismo acusar não é o mesmo que vê-lo **deixar de acusar**
    quando não há o que acusar.
    """
    identificador = rotas.TRABALHOS.abre(2)
    trabalho = _TrabalhoFalso(rotas, identificador, obedece=True)
    trabalho.start()
    assert trabalho.entrou.wait(timeout=5), "âncora: a thread falsa nunca arrancou"

    try:
        with warnings.catch_warnings(record=True) as apanhados:
            warnings.simplefilter("always")
            encerra_trabalhos_vivos(rotas, teto_s=5.0)
    finally:
        trabalho.solta.set()
        trabalho.join(timeout=5)

    assert not apanhados, (
        f"avisou sobre um trabalho que PAROU dentro do teto: "
        f"{[str(a.message) for a in apanhados]}. Um aviso que sai sempre não distingue nada"
    )
