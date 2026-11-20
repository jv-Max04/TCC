"""Dois trabalhos em vários núcleos com `n_jobs` efetivos diferentes, no mesmo processo.

## O defeito, medido antes do conserto

O executor reutilizável do `loky` é UM por processo. Quando um `Parallel` pede outro número de
workers que o do executor em uso, `get_reusable_executor` o muda **com a trava global do `loky` na
mão**: encerra-o com `shutdown(wait=True)` quando o `env` dos workers muda (`n_jobs` 2 contra 3 num
Linux de 4 núcleos), ou redimensiona-o com `_resize`, à espera de que as tarefas pendentes acabem,
quando o `env` é o mesmo (3 contra 4). Nos dois caminhos, se o outro trabalho ainda tem lotes a
despachar, o callback de conclusão dele, na thread de gestão do `loky`, chama `submit` para o lote
seguinte — e `submit` pede a mesma trava. **O processo trava**, e não só os dois trabalhos: todo
`Parallel` seguinte que peça o executor espera a mesma trava. Um de `n_jobs` efetivo 1 não o pede, e
corre.

**A condição é o outro ainda DESPACHAR.** Com o gerador dele já esgotado, não há `submit` no
callback, e o segundo espera as tarefas dele e termina, e isso foi medido. A âncora do voo dos
testes do filho é por isso NECESSÁRIA e
não suficiente: ela diz que o A ainda não terminou, e não que ainda despacha. Aqui o LOESS de 600
features em 3 núcleos despacha durante quase todo o tempo dele, e numa máquina mais rápida a janela
pode encolher; sem a vez, o B esperaria então os lotes do A, e o teste passaria sem ter exercitado a
trava.

O conserto é a vez no pool (`domain/pool.py`): um trabalho cujo `n_jobs` efetivo difere do que está
em uso espera a vez antes de pedir o executor, e trabalhos com o mesmo `n_jobs` continuam a correr
juntos.

## Por que os testes de produto correm num processo FILHO

Um teste que reproduzisse o defeito no processo do `pytest` deixaria, ao reprovar, a trava global do
`loky` presa numa thread: a sessão inteira penduraria no `Parallel` seguinte, sem dizer qual teste a
prendeu. No filho, o vigia despeja as pilhas e sai, e o teste reprova com o despejo na mensagem.
**É a forma externa da tabela**, e as duas guardas dela vão no próprio teste: o filho diz
de que árvore importou o `refino` — e o teste confere que é esta —, e diz que o trabalho A estava em
voo quando o B começou.

## Os desfechos da vez, e os cinco estão aqui

| Desfecho | Teste (`test_d678_...`) |
|---|---|
| entra logo: o mesmo `n_jobs` efetivo, ou um núcleo | `o_mesmo_n_efetivo_...`, `um_nucleo_...` |
| espera a vez: outro `n_jobs` em uso | `outro_n_jobs_espera_o_primeiro_sair` |
| espera atrás de quem chegou antes | `a_fila_e_por_ordem_de_chegada` |
| parado enquanto espera: levanta com o sítio e sai da fila | `parar_quem_espera_...` |
| o corpo levanta dentro: a vez é devolvida | `quem_levanta_...`, `o_recuo_para_a_serie_...` |

**E o SERRF, que ficou de fora, pede-a pela classe que o produto constrói.** Os dois
testes do filho no fim deste arquivo correm o SERRF de `build_corrector` ao lado de um LOESS com
outro `n_jobs`, nas duas ordens, e o mesmo `n_jobs` ao lado; os de cada chamador e o portão dos
construtores estão em `tests/unit/test_serrf_na_vez.py`.

**O quinto faltou na primeira versão, e era o que falhava em silêncio:** uma planta que tirava o
`finally` da devolução passava nos testes do arquivo e nos do recuo do LOESS e da régua — o
trabalho recuado terminava certo, e a vez ficava presa para todo outro `n_jobs` do processo.
"""

from __future__ import annotations

import contextlib
import json
import os
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

import refino
from refino.domain.cancellation import ExecutionCancelledError

# A importação de `vez_no_pool` é feita dentro de cada teste, e não aqui: os testes de produto, que
# correm num processo filho, têm de poder ser colhidos e reprovar pelo motivo deles — a trava — na
# árvore anterior ao conserto, onde o nome não existe.


def _vez() -> Callable[..., Any]:
    from refino.domain.pool import vez_no_pool

    return vez_no_pool


# **Nenhum teste aqui afirma a LATÊNCIA da parada de quem espera, e é de propósito.** Os 2 s que
# Escreveram como critério da parada são tempo de parede, e uma comparação deles sob
# carga reprova sem separar defeito de máquina ocupada — é o que o funil de
# `tests/orcamento.py` existe para guardar. O que
# se afirma é a ordem, que é determinística: a parada acontece com o outro AINDA dentro da vez. O
# número foi medido do pedido à exceção, com a bandeira lida a cada décimo de segundo.


def _segura(n_jobs: int, entrou: threading.Event, solta: threading.Event) -> threading.Thread:
    """Uma thread que toma a vez com `n_jobs` e só a devolve quando `solta` é erguido."""
    vez = _vez()  # aqui, e não na thread: um nome que falta reprova o teste, e não a thread

    def corre() -> None:
        with vez(n_jobs, None, "teste: quem segura"):
            entrou.set()
            solta.wait(30)

    linha = threading.Thread(target=corre, daemon=True)
    linha.start()
    assert entrou.wait(10), "âncora: quem segura nunca entrou"
    return linha


def _pede(
    n_jobs: int, cancelled: Any = None, where: str = "teste: quem pede"
) -> tuple[threading.Thread, threading.Event, threading.Event, dict[str, Any]]:
    """Uma thread que pede a vez com `n_jobs`; devolve-a com o «entrou», o «solta» e o desfecho."""
    entrou, solta = threading.Event(), threading.Event()
    desfecho: dict[str, Any] = {}
    vez = _vez()

    def corre() -> None:
        try:
            with vez(n_jobs, cancelled, where):
                desfecho["entrou_em"] = time.perf_counter()
                entrou.set()
                solta.wait(30)
        except ExecutionCancelledError as erro:
            desfecho["parada"] = erro

    linha = threading.Thread(target=corre, daemon=True)
    linha.start()
    return linha, entrou, solta, desfecho


def test_d678_o_mesmo_n_efetivo_entra_ao_mesmo_tempo() -> None:
    """Dois pedidos com o mesmo `n_jobs` ficam DENTRO ao mesmo tempo: a vez não serializa o pool.

    E o `n_jobs` comparado é o efetivo: `-1` e `cpu_count()` são o mesmo executor do `loky`, e a
    janela pede `-1` enquanto a API aceita o número. Se a vez os separasse, dois trabalhos que hoje
    correm juntos passariam a esperar um pelo outro.
    """
    from joblib import cpu_count

    solta = threading.Event()
    primeiro = _segura(-1, threading.Event(), solta)
    linha, entrou, solta_2, _ = _pede(cpu_count())
    try:
        assert entrou.wait(5), "o segundo pedido com o mesmo n_jobs efetivo esperou o primeiro"
    finally:
        solta_2.set()
        solta.set()
        linha.join(10)
        primeiro.join(10)


def test_d678_um_nucleo_nao_espera() -> None:
    """`n_jobs` efetivo 1 entra com o pool ocupado: o `joblib` corre-o em série, sem executor."""
    solta = threading.Event()
    primeiro = _segura(3, threading.Event(), solta)
    linha, entrou, solta_1, _ = _pede(1)
    try:
        assert entrou.wait(5), "o pedido de um núcleo esperou a vez, e não usa o executor"
    finally:
        solta_1.set()
        solta.set()
        linha.join(10)
        primeiro.join(10)


def test_d678_outro_n_jobs_espera_o_primeiro_sair() -> None:
    """Com 3 em uso, o pedido de 2 não entra; quando o de 3 sai, ele entra.

    **As duas metades no mesmo teste, e é isso que o torna par causal:** a primeira asserção sozinha
    passaria com uma vez que nunca deixasse ninguém entrar; a segunda sozinha, com uma que deixasse
    todos.
    """
    solta = threading.Event()
    primeiro = _segura(3, threading.Event(), solta)
    linha, entrou, solta_2, desfecho = _pede(2)
    try:
        assert not entrou.wait(0.5), "o pedido de 2 entrou com o pool em uso com 3"
        saiu_em = time.perf_counter()
        solta.set()
        primeiro.join(10)
        assert entrou.wait(5), "o primeiro saiu, e o pedido de 2 continuou à espera"
        assert desfecho["entrou_em"] >= saiu_em
    finally:
        solta_2.set()
        solta.set()
        linha.join(10)


def test_d678_a_fila_e_por_ordem_de_chegada() -> None:
    """Com 3 em uso e um pedido de 2 à espera, um pedido novo de 3 espera atrás do de 2.

    Sem a fila, uma sucessão de trabalhos com o `n_jobs` em uso faria o de 2 esperar para sempre.
    """
    solta = threading.Event()
    primeiro = _segura(3, threading.Event(), solta)
    linha_2, entrou_2, solta_2, desfecho_2 = _pede(2)
    assert not entrou_2.wait(0.3), "âncora: o pedido de 2 entrou com o pool em uso com 3"
    linha_3, entrou_3, solta_3, desfecho_3 = _pede(3)
    try:
        assert not entrou_3.wait(0.5), (
            "o pedido de 3 entrou à frente do de 2, que chegou antes e estava à espera"
        )
        solta.set()
        primeiro.join(10)
        assert entrou_2.wait(5), "o primeiro saiu, e o pedido de 2 continuou à espera"
        assert not entrou_3.wait(0.3), "o pedido de 3 entrou com o de 2 dentro"
        solta_2.set()
        linha_2.join(10)
        assert entrou_3.wait(5), "o de 2 saiu, e o pedido de 3 continuou à espera"
        assert desfecho_2["entrou_em"] < desfecho_3["entrou_em"]
    finally:
        solta.set()
        solta_2.set()
        solta_3.set()
        linha_2.join(10)
        linha_3.join(10)


def test_d678_parar_quem_espera_levanta_com_o_sitio_e_libera_a_fila() -> None:
    """O pedido parado enquanto espera levanta sem esperar o outro sair, e deixa de estar na fila.

    **A segunda metade é a que um conserto ingénuo perde:** um pedido que levantasse sem sair da
    fila deixaria o bilhete dele à frente, e um pedido seguinte com o `n_jobs` em uso esperaria por
    ele para sempre — a trava trocada por outra.
    """
    erguida: dict[str, float] = {}

    def bandeira() -> bool:
        return "t" in erguida

    solta = threading.Event()
    primeiro = _segura(3, threading.Event(), solta)
    linha, entrou, solta_2, desfecho = _pede(2, bandeira, "teste: espera da vez no pool")
    try:
        assert not entrou.wait(0.3), "âncora: o pedido de 2 entrou com o pool em uso com 3"
        erguida["t"] = time.perf_counter()
        linha.join(10)
        assert "parada" in desfecho, "o pedido parado continuou à espera, ou entrou"
        assert desfecho["parada"].where == "teste: espera da vez no pool"
        assert primeiro.is_alive(), "âncora: quem segurava a vez saiu antes da parada"
        linha_3, entrou_3, solta_3, _ = _pede(3)
        try:
            assert entrou_3.wait(5), "o pedido parado ficou na fila e prendeu o seguinte"
        finally:
            solta_3.set()
            linha_3.join(10)
    finally:
        solta.set()
        solta_2.set()
        primeiro.join(10)


def test_d678_quem_levanta_dentro_da_vez_a_devolve() -> None:
    """O corpo que levanta DENTRO da vez devolve-a: um pedido com outro `n_jobs` entra logo depois.

    **É o quinto desfecho, e o que falha em silêncio.** O LOESS, a régua e o jackknife apanham a
    morte do pool FORA do `with` e refazem em série: o trabalho continua vivo, e uma vez que não
    fosse devolvida quando o `Parallel` levanta deixaria todo outro `n_jobs` do processo à espera
    para sempre, sem erro em lado nenhum. As duas metades no mesmo teste: o pedido de 2 espera
    enquanto o corpo está dentro — senão a vez não teria sido tomada e o teste não provaria nada —,
    e entra depois de o corpo levantar.

    **O pedido de 2 é parado no fim, entre ou não.** Com a vez presa, ele ficaria à frente da fila
    para sempre, e todo teste seguinte deste processo esperaria atrás dele: medido com a planta
    que tira o `finally`, a sessão pendurou no primeiro despacho sem bandeira. Parado, ele sai da
    fila, e a reprovação fica neste teste.
    """
    vez = _vez()
    pedido: dict[str, Any] = {}
    sai = threading.Event()
    try:
        with pytest.raises(RuntimeError, match="planta"), vez(3, None, "teste: quem levanta"):
            pedido["pede"] = _pede(2, sai.is_set, "teste: o pedido de 2")
            assert not pedido["pede"][1].wait(0.3), (
                "âncora: o pedido de 2 entrou com a vez em uso com 3"
            )
            raise RuntimeError("planta: o corpo do Parallel levantou")
        _, entrou, _, _ = pedido["pede"]
        assert entrou.wait(5), "o corpo levantou dentro da vez, e ela ficou presa com ele"
    finally:
        sai.set()
        if "pede" in pedido:
            linha, _, solta, _ = pedido["pede"]
            solta.set()
            linha.join(10)


# ---------------------------------------------------------------------------
# A vez nos quatro despachos — o que liga o mecanismo ao produto
# ---------------------------------------------------------------------------


def _matriz(n_features: int, n_amostras: int = 68) -> Any:
    """Sintética: 68 amostras, 7 QCs nas injeções 1, 12, 23, 34, 45, 56 e 68, lote único."""
    import numpy as np

    from refino.domain.matrix import RefinoMatrix
    from refino.domain.metadata import SampleMetadata

    qcs = (1, 12, 23, 34, 45, 56, 68)
    rng = np.random.default_rng(7)
    ordem = np.arange(1, n_amostras + 1)
    base = rng.lognormal(mean=9.0, sigma=1.5, size=n_features)
    fase = rng.uniform(0, 2 * np.pi, size=n_features)
    deriva = 1.0 + 0.25 * np.sin(2 * np.pi * ordem[:, None] / 68.0 + fase[None, :])
    ruido = rng.lognormal(mean=0.0, sigma=0.12, size=(n_amostras, n_features))
    return RefinoMatrix(
        values=base[None, :] * deriva * ruido,
        feature_names=tuple(f"f{j}" for j in range(n_features)),
        samples=tuple(
            SampleMetadata(
                name=f"s{i:02d}",
                injection_order=int(i),
                sample_type="qc" if i in qcs else "sample",
                group="qc" if i in qcs else "caso",
                batch="A",
            )
            for i in ordem
        ),
        mode="synthetic",
    )


def _despacho_loess(cancelled: Any) -> Any:
    from refino.correction.loess import LoessCorrector

    return LoessCorrector(n_jobs=2).fit_transform(_matriz(4), seed=42, cancelled=cancelled)


def _despacho_da_regua(cancelled: Any) -> Any:
    from refino.correction.none import NoneCorrector
    from refino.metrics.crossval import leave_one_qc_out_rsd

    return leave_one_qc_out_rsd(_matriz(4), NoneCorrector(), n_jobs=2, cancelled=cancelled)


def _despacho_do_jackknife(cancelled: Any) -> Any:
    from refino.correction.none import NoneCorrector
    from refino.metrics.crossval import pair_out_of_sample_values

    return pair_out_of_sample_values(_matriz(4), NoneCorrector(), n_jobs=2, cancelled=cancelled)


def _despacho_do_executor(cancelled: Any) -> Any:
    from refino.multiverse.executor import run_multiverse
    from refino.multiverse.grid import GridDefinition

    grade = GridDefinition(
        corrections=("none",),
        batch_options=(1,),
        rsd_cutoffs=(None,),
        missing_thresholds=(0.8,),
        imputations=("median",),
        sample_norms=("none",),
        transformations=("none",),
        scalings=("none", "auto"),
    )
    return run_multiverse(_matriz(4), grade, n_jobs=2, cancelled=cancelled)


@pytest.mark.parametrize(
    ("despacho", "sitio"),
    [
        (_despacho_loess, "loess: espera da vez no pool"),
        (_despacho_da_regua, "leave-one-qc-out: espera da vez no pool"),
        (_despacho_do_jackknife, "jackknife: espera da vez no pool"),
        (_despacho_do_executor, "executor: espera da vez no pool"),
    ],
    ids=["loess", "regua-fora-da-amostra", "jackknife", "executor-do-multiverso"],
)
def test_d678_cada_despacho_espera_a_vez_com_outro_n_jobs_em_uso(despacho: Any, sitio: str) -> None:
    """Com 3 em uso, o despacho de 2 núcleos espera a vez — e, parado ali, levanta com o sítio dela.

    **O sítio é a prova de que ele esperou, e de que esperou ANTES do `Parallel`:** a parada no
    despacho levanta com outro nome (`... despacho de ...`), e uma parada lá significaria que o
    trabalho pediu o executor com outro em uso — o caminho da trava. Quem segura a vez aqui é o
    teste, e não um `Parallel`, então nenhum executor é tocado: o teste é barato e não pode travar.
    """
    erguida: dict[str, float] = {}
    leituras = [0]

    def bandeira() -> bool:
        leituras[0] += 1
        return "t" in erguida

    solta = threading.Event()
    primeiro = _segura(3, threading.Event(), solta)
    desfecho: dict[str, Any] = {}

    def corre() -> None:
        try:
            despacho(bandeira)
            desfecho["terminou"] = True
        except ExecutionCancelledError as erro:
            desfecho["parada"] = erro

    linha = threading.Thread(target=corre, daemon=True)
    linha.start()
    try:
        prazo = time.perf_counter() + 10
        while leituras[0] < 3 and time.perf_counter() < prazo:
            time.sleep(0.02)
        assert leituras[0] >= 3, "âncora: o despacho nunca chegou a ler a bandeira à espera"
        assert "terminou" not in desfecho, "o despacho terminou com outro n_jobs em uso"
        erguida["t"] = time.perf_counter()
        linha.join(10)
        assert "parada" in desfecho, f"o despacho não parou: {desfecho}"
        assert desfecho["parada"].where == sitio
        assert primeiro.is_alive(), "âncora: quem segurava a vez saiu antes da parada"
    finally:
        solta.set()
        primeiro.join(10)
        linha.join(10)


def _recuo_do_loess(n_jobs: int, cancelled: Any) -> bytes:
    from refino.correction.loess import LoessCorrector

    corrigida = LoessCorrector(n_jobs=n_jobs).fit_transform(
        _matriz(4), seed=42, cancelled=cancelled
    )
    return bytes(corrigida.matrix.values.tobytes())


def _recuo_da_regua(n_jobs: int, cancelled: Any) -> bytes:
    from refino.correction.none import NoneCorrector
    from refino.metrics.crossval import leave_one_qc_out_rsd

    valores = leave_one_qc_out_rsd(_matriz(4), NoneCorrector(), n_jobs=n_jobs, cancelled=cancelled)
    return bytes(valores.tobytes())


def _recuo_do_jackknife(n_jobs: int, cancelled: Any) -> bytes:
    from refino.correction.none import NoneCorrector
    from refino.metrics.crossval import pair_out_of_sample_values

    pares = pair_out_of_sample_values(
        _matriz(4), NoneCorrector(), n_jobs=n_jobs, cancelled=cancelled
    )
    return b"".join(i.tobytes() + j.tobytes() for _, (i, j) in sorted(pares.items()))


@pytest.mark.parametrize(
    ("modulo", "recuo"),
    [
        ("refino.correction.loess", _recuo_do_loess),
        ("refino.metrics.crossval", _recuo_da_regua),
        ("refino.metrics.crossval", _recuo_do_jackknife),
    ],
    ids=["loess", "regua-fora-da-amostra", "jackknife"],
)
def test_d678_o_recuo_para_a_serie_devolve_a_vez(
    modulo: str, recuo: Callable[[int, Any], bytes], monkeypatch: pytest.MonkeyPatch
) -> None:
    """O pool morre a meio com a vez tomada, o despacho refaz em série — e a vez fica LIVRE.

    São os três despachos que apanham `MORTE_DO_POOL` fora do `with` e seguem vivos; o executor do
    multiverso não recua, e a exceção dele sobe, que é o caso de
    `quem_levanta_dentro_da_vez_a_devolve`. A planta é o `Parallel` que morre depois de consumir uma
    tarefa, a dos testes do recuo; dentro dele, um pedido de 2 tem de ESPERAR — a âncora de que o
    despacho de 3 tomou a vez —, e depois do recuo tem de entrar.

    **O despacho de 3 corre com um prazo na bandeira**, e o pedido de 2 é parado no fim: com a vez
    presa por um teste anterior, um despacho sem bandeira esperaria para sempre, e a sessão
    penduraria em vez de reprovar. O prazo não é latência afirmada — é o limite de uma espera que,
    sem defeito, não acontece.
    """
    import importlib

    from joblib.externals.loky.process_executor import TerminatedWorkerError

    pedido: dict[str, Any] = {}
    sai = threading.Event()
    limite = time.perf_counter() + 30

    def prazo() -> bool:
        return time.perf_counter() > limite

    class _PoolQueMorre:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            del args, kwargs

        def __call__(self, tarefas: Any) -> Any:
            next(iter(tarefas))
            pedido["pede"] = _pede(2, sai.is_set, "teste: o pedido de 2")
            pedido["esperou"] = not pedido["pede"][1].wait(0.3)
            raise TerminatedWorkerError("planta: o worker morreu a meio")

    serie = recuo(1, None)
    monkeypatch.setattr(importlib.import_module(modulo), "Parallel", _PoolQueMorre)
    try:
        recuada = recuo(3, prazo)
        assert "pede" in pedido, "âncora: o despacho de 3 não chegou ao Parallel"
        assert pedido["esperou"], "âncora: o pedido de 2 entrou com o despacho de 3 no Parallel"
        assert recuada == serie, "o recuo não devolveu a saída da série"
        assert pedido["pede"][1].wait(5), "o despacho recuou para a série, e a vez ficou presa"
    finally:
        sai.set()
        if "pede" in pedido:
            linha, _, solta, _ = pedido["pede"]
            solta.set()
            linha.join(10)


# ---------------------------------------------------------------------------
# O produto, num processo filho: os dois trabalhos terminam, e com os bytes da série
# ---------------------------------------------------------------------------

_VIGIA_DO_FILHO_S = 45.0

_FILHO = r"""
import faulthandler, hashlib, json, os, sys, threading, time

import numpy as np

import refino
from refino.domain.cancellation import ExecutionCancelledError
from refino.domain.matrix import RefinoMatrix
from refino.domain.metadata import SampleMetadata

QCS = (1, 12, 23, 34, 45, 56, 68)


def matriz(n):
    rng = np.random.default_rng(7)
    ordem = np.arange(1, 69)
    base = rng.lognormal(mean=9.0, sigma=1.5, size=n)
    fase = rng.uniform(0, 2 * np.pi, size=n)
    deriva = 1.0 + 0.25 * np.sin(2 * np.pi * ordem[:, None] / 68.0 + fase[None, :])
    ruido = rng.lognormal(mean=0.0, sigma=0.12, size=(68, n))
    return RefinoMatrix(
        values=base[None, :] * deriva * ruido,
        feature_names=tuple(f"f{j}" for j in range(n)),
        samples=tuple(
            SampleMetadata(
                name=f"s{i:02d}", injection_order=int(i),
                sample_type="qc" if i in QCS else "sample",
                group="qc" if i in QCS else "caso", batch="A",
            )
            for i in ordem
        ),
        mode="synthetic",
    )


def corre(spec, cancelled):
    tipo, n_jobs, n = spec.split(":")
    n_jobs, m = int(n_jobs), matriz(int(n))
    if tipo == "loess":
        from refino.correction.loess import LoessCorrector
        r = LoessCorrector(n_jobs=n_jobs).fit_transform(m, seed=42, cancelled=cancelled)
        return np.ascontiguousarray(r.matrix.values).tobytes()
    if tipo == "regua":
        from refino.correction.loess import LoessCorrector
        from refino.metrics.crossval import leave_one_qc_out_rsd
        v = leave_one_qc_out_rsd(m, LoessCorrector(n_jobs=1), n_jobs=n_jobs, cancelled=cancelled)
        return np.ascontiguousarray(v).tobytes()
    if tipo == "serrf":
        from refino.multiverse.executor import build_corrector
        r = build_corrector("serrf", n_jobs).fit_transform(m, seed=42, cancelled=cancelled)
        return np.ascontiguousarray(r.matrix.values).tobytes()
    from refino.multiverse.executor import run_multiverse
    from refino.multiverse.grid import GridDefinition
    grade = GridDefinition(
        corrections=("none",), batch_options=(1,), rsd_cutoffs=(None, 0.30),
        missing_thresholds=(0.8,), imputations=("median",), sample_norms=("none",),
        transformations=("none", "log10", "sqrt", "cbrt"),
        scalings=("none", "mean_center", "auto", "pareto"),
    )
    run = run_multiverse(m, grade, n_jobs=n_jobs, cancelled=cancelled)
    return b"".join(np.ascontiguousarray(c.matrix.values).tobytes() for c in run.cells)


spec_a, spec_b, vigia, contra_a_serie = sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4]
t0 = time.perf_counter()
reg = {"A": {}, "B": {}}
a_em_voo = threading.Event()
leituras = [0]
# O SERRF lê a bandeira uma vez por feature no passo 1 e uma no içamento antes de despachar: a
# âncora do voo dele é a terceira leitura do despacho. A do LOESS é a terceira, já no despacho.
tipo_a, _jobs_a, n_a = spec_a.split(":")
limiar_do_voo = int(n_a) + 4 if tipo_a == "serrf" else 3


def bandeira_a():
    leituras[0] += 1
    if leituras[0] >= limiar_do_voo:
        a_em_voo.set()
    return False


def trabalho(nome, spec, bandeira):
    reg[nome]["inicio"] = time.perf_counter() - t0
    try:
        reg[nome]["bytes"] = corre(spec, bandeira)
    except BaseException as exc:
        reg[nome]["erro"] = f"{type(exc).__name__}: {exc}"
    reg[nome]["fim"] = time.perf_counter() - t0


ta = threading.Thread(target=trabalho, args=("A", spec_a, bandeira_a), daemon=True)
ta.start()
a_em_voo.wait(vigia)
time.sleep(0.3)
em_voo = "fim" not in reg["A"]
tb = threading.Thread(target=trabalho, args=("B", spec_b, None), daemon=True)
tb.start()
ta.join(vigia)
tb.join(max(0.0, vigia - (time.perf_counter() - t0)))
if ta.is_alive() or tb.is_alive():
    print(json.dumps({"refino": refino.__file__, "travou": True}), flush=True)
    faulthandler.dump_traceback(file=sys.stderr, all_threads=True)
    sys.stderr.flush()
    os._exit(3)
saida = {"refino": refino.__file__, "travou": False, "a_em_voo": em_voo}
for nome, spec in (("A", spec_a), ("B", spec_b)):
    r = reg[nome]
    saida[nome] = {k: v for k, v in r.items() if k != "bytes"}
    if "bytes" in r and nome in contra_a_serie:
        tipo, _n_jobs, n = spec.split(":")
        r["serie"] = corre(f"{tipo}:1:{n}", None)
        saida[nome]["igual_a_serie"] = r["bytes"] == r["serie"]
print(json.dumps(saida), flush=True)
"""


def _corre_o_filho(spec_a: str, spec_b: str, contra_a_serie: str, pasta: Path) -> dict[str, Any]:
    """Corre o par no filho, com o `refino` DESTA árvore, e devolve o desfecho que ele imprimiu.

    `contra_a_serie` diz quais dos dois — `A`, `B` ou `AB` — o filho refaz em série para comparar os
    bytes. A série do A, o LOESS de 600 features, custa perto de 5 s, e basta num caso.

    **A saída do filho vai para arquivos, e não para tubos — medido na árvore anterior ao
    conserto.**
    Os workers do `loky` de um filho que travou e saiu por `os._exit` herdam os descritores dele e
    ficam vivos: com tubos, `communicate` esperava o fim deles até o próprio tempo limite — cada
    caso levou perto de 3 min, o limite, e não os 45 s do vigia —, e o teste não podia reprovar com
    as pilhas. `wait` espera só o filho; a sessão própria e o grupo morto no fim levam os órfãos.
    """
    src = Path(refino.__file__).resolve().parents[1]
    caminhos = [str(src), *filter(None, os.environ.get("PYTHONPATH", "").split(os.pathsep))]
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(caminhos)}
    saida_txt, erros_txt = pasta / "filho.out", pasta / "filho.err"
    with saida_txt.open("w", encoding="utf-8") as out, erros_txt.open("w", encoding="utf-8") as err:
        proc = subprocess.Popen(
            [sys.executable, "-c", _FILHO, spec_a, spec_b, str(_VIGIA_DO_FILHO_S), contra_a_serie],
            stdout=out,
            stderr=err,
            env=env,
            start_new_session=True,
        )
        try:
            proc.wait(timeout=_VIGIA_DO_FILHO_S * 4)
        finally:
            if hasattr(os, "killpg"):
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(proc.pid, signal.SIGKILL)
            if proc.poll() is None:
                proc.kill()
                proc.wait(10)
    saida = saida_txt.read_text(encoding="utf-8", errors="replace")
    erros = erros_txt.read_text(encoding="utf-8", errors="replace")
    linhas = [linha for linha in saida.splitlines() if linha.startswith("{")]
    assert linhas, f"o filho não imprimiu desfecho (EXIT={proc.returncode}):\n{erros[-4000:]}"
    desfecho: dict[str, Any] = json.loads(linhas[-1])
    assert Path(desfecho["refino"]).resolve() == Path(refino.__file__).resolve(), (
        f"âncora: o filho importou o refino de outra árvore: {desfecho['refino']}"
    )
    assert not desfecho["travou"], (
        f"o processo travou com {spec_a} e {spec_b} em duas threads (o vigia de "
        f"{_VIGIA_DO_FILHO_S:.0f} s disparou). Pilhas:\n{erros[-6000:]}"
    )
    assert proc.returncode == 0, erros[-4000:]
    return desfecho


@pytest.mark.parametrize(
    ("spec_b", "contra_a_serie"),
    [("loess:2:100", "AB"), ("regua:2:30", "B"), ("executor:2:200", "B")],
    ids=["loess", "regua-fora-da-amostra", "executor-do-multiverso"],
)
def test_d678_dois_trabalhos_com_n_jobs_diferentes_terminam_os_dois(
    spec_b: str, contra_a_serie: str, tmp_path: Path
) -> None:
    """O LOESS em 3 núcleos em voo e outro trabalho em 2 começado ao lado: os dois terminam.

    Na árvore anterior ao conserto, o filho trava nos três casos e o vigia o derruba. **A âncora do
    voo vem antes da asserção sobre os dois:** se o A já tivesse terminado quando o B começou, os
    dois terminariam com qualquer desenho, e o teste não provaria nada. Ela é necessária e não
    suficiente — a trava pede o A ainda a DESPACHAR, e não só em voo (o cabeçalho deste arquivo);
    na árvore anterior ao conserto os três casos travaram aqui, e é isso que diz que a janela existe
    nesta máquina.
    """
    desfecho = _corre_o_filho("loess:3:600", spec_b, contra_a_serie, tmp_path)

    assert desfecho["a_em_voo"], "âncora: o A já tinha terminado quando o B começou"
    for nome in ("A", "B"):
        assert "erro" not in desfecho[nome], f"{nome}: {desfecho[nome]['erro']}"
    for nome in contra_a_serie:
        assert desfecho[nome]["igual_a_serie"], f"{nome}: a saída não é a da série bit a bit"


def test_d678_dois_trabalhos_com_o_mesmo_n_jobs_continuam_simultaneos(tmp_path: Path) -> None:
    """O LOESS em 3 núcleos em voo e um LOESS curto em 3 ao lado: o curto termina ANTES do longo.

    É o comportamento que a vez declara preservar, afirmado e não lido no diff: trabalhos com
    o mesmo `n_jobs` partilham o executor ao mesmo tempo. Se a vez os serializasse, o curto só
    começaria depois de o longo acabar, e terminaria depois dele.
    """
    desfecho = _corre_o_filho("loess:3:1200", "loess:3:20", "B", tmp_path)

    assert desfecho["a_em_voo"], "âncora: o A já tinha terminado quando o B começou"
    for nome in ("A", "B"):
        assert "erro" not in desfecho[nome], f"{nome}: {desfecho[nome]['erro']}"
    assert desfecho["B"]["igual_a_serie"], "B: a saída não é a da série bit a bit"
    assert desfecho["B"]["fim"] < desfecho["A"]["fim"], (
        f"o curto terminou em {desfecho['B']['fim']:.2f} s, depois do longo "
        f"({desfecho['A']['fim']:.2f} s): os dois não correram juntos"
    )


# ---------------------------------------------------------------------------
# O SERRF, que pede a vez pela classe que o produto constrói
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("spec_a", "spec_b"),
    [("loess:3:600", "serrf:2:8"), ("serrf:3:60", "loess:2:100")],
    ids=["o-serrf-chega-depois", "o-serrf-chega-antes"],
)
def test_d681_o_serrf_e_outro_n_jobs_terminam_os_dois_nas_duas_ordens(
    spec_a: str, spec_b: str, tmp_path: Path
) -> None:
    """O SERRF do executor ao lado do LOESS com outro `n_jobs`, nas duas ordens: terminam os dois.

    **As duas ordens, porque o defeito tinha as duas:** antes do conserto a vez não via o SERRF, e
    tanto o SERRF que chegava com o LOESS a despachar como o LOESS que chegava com o SERRF a
    despachar pediam o executor com outro `n_jobs` em uso. Na árvore anterior ao conserto o filho
    trava nos dois casos. O SERRF é o que `build_corrector` do executor monta, e não o
    `SerrfCorrector` solto: é a classe do produto que tem de pedir a vez. A âncora do voo do SERRF
    cai no despacho dele, depois do passo 1 e do içamento (`limiar_do_voo`, no filho).
    """
    desfecho = _corre_o_filho(spec_a, spec_b, "B", tmp_path)

    assert desfecho["a_em_voo"], "âncora: o A já tinha terminado quando o B começou"
    for nome in ("A", "B"):
        assert "erro" not in desfecho[nome], f"{nome}: {desfecho[nome]['erro']}"
    assert desfecho["B"]["igual_a_serie"], "B: a saída não é a da série bit a bit"


def test_d681_o_serrf_e_o_mesmo_n_jobs_continuam_simultaneos(tmp_path: Path) -> None:
    """O LOESS em 3 núcleos em voo e o SERRF curto em 3 ao lado: o SERRF termina ANTES do LOESS.

    A vez em volta do SERRF cobre o `fit_transform` inteiro, e não só o `Parallel` dele; com o mesmo
    `n_jobs` efetivo ela continua a deixar entrar junto. É o comportamento que o conserto declara
    preservar, afirmado e não lido no diff: se a vez os serializasse, o SERRF só começaria
    depois de o LOESS acabar, e terminaria depois dele.
    """
    desfecho = _corre_o_filho("loess:3:1200", "serrf:3:8", "B", tmp_path)

    assert desfecho["a_em_voo"], "âncora: o A já tinha terminado quando o B começou"
    for nome in ("A", "B"):
        assert "erro" not in desfecho[nome], f"{nome}: {desfecho[nome]['erro']}"
    assert desfecho["B"]["igual_a_serie"], "B: a saída não é a da série bit a bit"
    assert desfecho["B"]["fim"] < desfecho["A"]["fim"], (
        f"o SERRF terminou em {desfecho['B']['fim']:.2f} s, depois do LOESS "
        f"({desfecho['A']['fim']:.2f} s): os dois não correram juntos"
    )
