"""O SERRF na vez do pool, pelos construtores do produto.

## O que se protege

A vez foi posta no pool nos despachos do LOESS, da régua, do jackknife e das folhas do executor, e
deixou o SERRF de fora, porque `correction/serrf.py` é caminho medido de dois portões de
procedência.
O SERRF com outro `n_jobs` que o de um despacho ainda a submeter lotes travava o processo, nas duas
ordens. **O conserto é por construção**: todo construtor do produto monta `SerrfNaVez`, que põe o
`fit_transform` inteiro dentro da vez; os testes do processo filho, com os bytes da série, estão em
`tests/unit/test_vez_no_pool.py`, ao lado dos.

| O que se perde | Teste (`test_d681_...`) |
|---|---|
| um chamador deixa de pedir a vez | `o_serrf_de_cada_chamador_espera_a_vez_...` |
| um construtor novo monta o `SerrfCorrector` à mão | `nenhum_construtor_do_produto_...`, e o par |
| a vez muda a conta | `a_vez_nao_muda_um_byte_da_saida` |

**Os chamadores são os do produto, e não o corretor solto:** o executor, a medição dos corretores e
o seu intervalo, as duas fábricas das rotas e a da CLI. A régua e os pares do jackknife em série
chamam o `fit_transform` do corretor que recebem, em `metrics/`, e é o caso da medição que diz se o
SERRF que lhes chega pede a vez.
"""

from __future__ import annotations

import ast
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import refino
from refino.desktop import paralelismo
from refino.desktop.paralelismo import PARALELO
from refino.domain.cancellation import ExecutionCancelledError
from refino.domain.matrix import RefinoMatrix
from refino.domain.metadata import SampleMetadata

# O sítio da espera é literal aqui, e não importado de `correction/serrf_na_vez.py`: os testes têm
# de poder ser colhidos e reprovar pelo motivo deles na árvore anterior ao conserto, onde o módulo
# não existe.
_SITIO = "serrf: espera da vez no pool"

_QCS = (1, 12, 23, 34, 45, 56, 68)


def _matriz(n_features: int = 4, semente: int = 7) -> RefinoMatrix:
    """A matriz sintética: 68 amostras, 7 QCs, lote único."""
    rng = np.random.default_rng(semente)
    ordem = np.arange(1, 69)
    base = rng.lognormal(mean=9.0, sigma=1.5, size=n_features)
    fase = rng.uniform(0, 2 * np.pi, size=n_features)
    deriva = 1.0 + 0.25 * np.sin(2 * np.pi * ordem[:, None] / 68.0 + fase[None, :])
    ruido = rng.lognormal(mean=0.0, sigma=0.12, size=(68, n_features))
    return RefinoMatrix(
        values=base[None, :] * deriva * ruido,
        feature_names=tuple(f"f{j}" for j in range(n_features)),
        samples=tuple(
            SampleMetadata(
                name=f"s{i:02d}",
                injection_order=int(i),
                sample_type="qc" if i in _QCS else "sample",
                group="qc" if i in _QCS else "caso",
                batch="A",
            )
            for i in ordem
        ),
        mode="synthetic",
    )


@pytest.fixture
def regime_paralelo() -> Iterator[None]:
    """A capacidade do processo em `PARALELO`, e devolvida no fim — ela é global."""
    anterior = paralelismo._REGIME
    paralelismo._REGIME = PARALELO
    yield
    paralelismo._REGIME = anterior


def _efetivo(n_jobs: int) -> int:
    """O `n_jobs` efetivo de `n_jobs` pela conta da vez, para o teste segurar OUTRO."""
    from refino.domain.pool import _n_efetivo

    return _n_efetivo(n_jobs)


def _pelo_executor(cancelled: Any) -> Any:
    from refino.multiverse.executor import run_multiverse
    from refino.multiverse.grid import GridDefinition

    grade = GridDefinition(
        corrections=("serrf",),
        batch_options=(1,),
        rsd_cutoffs=(None,),
        missing_thresholds=(0.8,),
        imputations=("median",),
        sample_norms=("none",),
        transformations=("none",),
        scalings=("none",),
    )
    return run_multiverse(_matriz(), grade, n_jobs=1, corrector_jobs=2, cancelled=cancelled)


def _pela_medicao(cancelled: Any) -> Any:
    from refino.decision.labeling import LabelingOracle

    oraculo = LabelingOracle(prefixes=(("serrf", 1),), n_jobs=2)
    return oraculo.label(_matriz(), cancelled=cancelled)


def _pelo_intervalo_da_medicao(cancelled: Any) -> Any:
    from refino.decision.labeling import LabelingOracle

    oraculo = LabelingOracle(prefixes=(("serrf", 1),), n_jobs=2)
    return oraculo.label_com_intervalos(_matriz(), cancelled=cancelled)


def _pela_rota_de_correcao(cancelled: Any) -> Any:
    from refino.api.rotas import _corretor_ou_422, _corrige_reusando

    corretor = _corretor_ou_422("serrf", "r_faithful", paralelo=True)
    # Uma semente própria da matriz: a memória de correções é do processo, e a chave de uma
    # correção que outro teste tivesse completado devolveria o guardado sem chamar o corretor.
    return _corrige_reusando(
        corretor, _matriz(semente=11), batches=None, seed=42, cancelled=cancelled
    )


def _pela_rota_das_execucoes(cancelled: Any) -> Any:
    from refino.api.rotas import _corretor_da_spec, _corrige_reusando

    corretor = _corretor_da_spec("serrf", paralelo=True)
    return _corrige_reusando(
        corretor, _matriz(semente=13), batches=None, seed=42, cancelled=cancelled
    )


def _pela_cli(cancelled: Any) -> Any:
    from refino.cli import build_corrector

    corretor = build_corrector("serrf", "r_faithful")
    return corretor.fit_transform(_matriz(), seed=42, cancelled=cancelled)


@pytest.mark.parametrize(
    ("chamador", "n_jobs_do_serrf"),
    [
        (_pelo_executor, 2),
        (_pela_medicao, 2),
        (_pelo_intervalo_da_medicao, 2),
        (_pela_rota_de_correcao, PARALELO),
        (_pela_rota_das_execucoes, PARALELO),
        (_pela_cli, -1),
    ],
    ids=["executor", "medicao", "intervalo-da-medicao", "rota-de-correcao", "execucoes", "cli"],
)
def test_d681_o_serrf_de_cada_chamador_espera_a_vez_com_outro_n_jobs_em_uso(
    chamador: Callable[[Any], Any], n_jobs_do_serrf: int, regime_paralelo: object
) -> None:
    """Com outro `n_jobs` em uso, o SERRF de cada chamador espera a vez — e, parado ali, levanta
    com o sítio dela.

    **O sítio é a prova de que esperou ANTES do passo 1:** uma parada no laço do passo 1 ou no
    despacho levanta com outro nome, e significaria que o SERRF correu com outro `n_jobs` em uso —
    o caminho da trava, quando o outro é um `Parallel`. Quem segura a vez aqui é o teste, e nenhum
    executor do `loky` é tocado enquanto ele a segura: o teste não pode travar.

    O `n_jobs` que se segura é o efetivo do SERRF mais um, e não um número fixo: `-1` vale o
    número de núcleos desta máquina.
    """
    from refino.domain.pool import vez_no_pool

    n_do_serrf = _efetivo(n_jobs_do_serrf)
    assert n_do_serrf > 1, "âncora: com um núcleo o SERRF corre em série e não pede a vez"

    entrou, solta = threading.Event(), threading.Event()

    def segura() -> None:
        with vez_no_pool(n_do_serrf + 1, None, "teste: quem segura"):
            entrou.set()
            solta.wait(30)

    quem_segura = threading.Thread(target=segura, daemon=True)
    quem_segura.start()
    assert entrou.wait(10), "âncora: quem segura nunca entrou"

    erguida: dict[str, float] = {}
    leituras = [0]

    def bandeira() -> bool:
        leituras[0] += 1
        return "t" in erguida

    desfecho: dict[str, Any] = {}

    def corre() -> None:
        try:
            chamador(bandeira)
            desfecho["terminou"] = True
        except ExecutionCancelledError as erro:
            desfecho["parada"] = erro

    linha = threading.Thread(target=corre, daemon=True)
    linha.start()
    try:
        prazo = time.perf_counter() + 10
        while leituras[0] < 3 and time.perf_counter() < prazo:
            time.sleep(0.02)
        assert leituras[0] >= 3, "âncora: o chamador nunca chegou a ler a bandeira"
        assert "terminou" not in desfecho, "o SERRF terminou com outro n_jobs em uso"
        erguida["t"] = time.perf_counter()
        linha.join(60)
        assert not linha.is_alive(), "o chamador não parou em 60 s"
        assert "parada" in desfecho, f"o chamador não parou: {desfecho}"
        assert desfecho["parada"].where == _SITIO, (
            f"o SERRF parou em {desfecho['parada'].where!r}, e não à espera da vez: correu com "
            "outro n_jobs em uso"
        )
        assert quem_segura.is_alive(), "âncora: quem segurava a vez saiu antes da parada"
    finally:
        solta.set()
        quem_segura.join(10)
        linha.join(60)


def test_d681_a_vez_nao_muda_um_byte_da_saida() -> None:
    """O SERRF na vez, em dois núcleos, devolve os bytes do `SerrfCorrector` em série.

    É a afirmação de que a subclasse só decide QUANDO o ajuste começa, afirmada e não lida no diff:
    a vez está fora do caminho medido dos portões de procedência, e se ela mudasse a conta os
    dois portões continuariam verdes sobre um número que já não é o do produto. A identidade vai
    junto, porque é ela que a memória de correções usa como chave.
    """
    from refino.correction.serrf import SerrfCorrector
    from refino.correction.serrf_na_vez import SerrfNaVez

    matriz = _matriz(n_features=6)
    serie = SerrfCorrector(n_jobs=1).fit_transform(matriz, seed=42)
    na_vez = SerrfNaVez(n_jobs=2).fit_transform(matriz, seed=42)

    assert na_vez.matrix.values.tobytes() == serie.matrix.values.tobytes()
    assert na_vez.uncorrected == serie.uncorrected
    assert SerrfNaVez(n_jobs=2).identidade() == SerrfCorrector(n_jobs=1).identidade()
    assert SerrfNaVez.name == SerrfCorrector.name == "serrf"


# ---------------------------------------------------------------------------
# O portão: nenhum construtor do produto monta o `SerrfCorrector` à mão
# ---------------------------------------------------------------------------

_ISENTOS = (
    Path("correction") / "serrf.py",
    Path("correction") / "serrf_na_vez.py",
)


def _usos_como_valor(fonte: str) -> list[int]:
    """As linhas em que o nome `SerrfCorrector` é usado como VALOR — chamado, guardado, passado.

    Anotação não conta: `-> NoneCorrector | LoessCorrector | SerrfCorrector` diz o tipo do que se
    devolve, e a subclasse é desse tipo. Importação também não, porque não é uso; mas o NOME que ela
    dá conta — `from ... import SerrfCorrector as S` faz de `S(n_jobs=-1)` a mesma construção. O
    segundo argumento de `isinstance` e de `issubclass` também não conta: pergunta pelo tipo, e o
    `SerrfNaVez` responde que sim.

    **O que fica é a sintaxe em que o nome PODE virar um objeto construído sem a vez**, e não só a
    construção: `SerrfCorrector(n_jobs=-1)`, a classe numa tabela que alguém instancia depois, ou um
    `partial` dela — e também um `cast(SerrfCorrector, ...)` ou um alias de tipo por atribuição,
    `Z = list[SerrfCorrector]`, que não constroem nada e reprovam. Essa direção é ruidosa: reprova e
    diz a linha. **A silenciosa fica no escopo negativo**: um `getattr` sobre o módulo, um
    `type(corretor)(n_jobs=...)` ou uma importação dinâmica escapam do portão.
    """
    arvore = ast.parse(fonte)
    nomes = {"SerrfCorrector"}
    for no in ast.walk(arvore):
        if isinstance(no, ast.ImportFrom):
            nomes.update(
                apelido.asname
                for apelido in no.names
                if apelido.name == "SerrfCorrector" and apelido.asname
            )
    em_anotacao: set[int] = set()
    for no in ast.walk(arvore):
        if (
            isinstance(no, ast.Call)
            and isinstance(no.func, ast.Name)
            and no.func.id in {"isinstance", "issubclass"}
            and len(no.args) == 2
        ):
            em_anotacao.update(id(filho) for filho in ast.walk(no.args[1]))
        anotacoes: list[ast.expr | None] = []
        if isinstance(no, ast.FunctionDef | ast.AsyncFunctionDef):
            argumentos = no.args
            todos = [*argumentos.posonlyargs, *argumentos.args, *argumentos.kwonlyargs]
            todos += [a for a in (argumentos.vararg, argumentos.kwarg) if a is not None]
            anotacoes += [a.annotation for a in todos]
            anotacoes.append(no.returns)
        elif isinstance(no, ast.AnnAssign):
            anotacoes.append(no.annotation)
        for anotacao in anotacoes:
            if anotacao is not None:
                em_anotacao.update(id(filho) for filho in ast.walk(anotacao))
    linhas = []
    for no in ast.walk(arvore):
        if id(no) in em_anotacao:
            continue
        if (isinstance(no, ast.Name) and no.id in nomes) or (
            isinstance(no, ast.Attribute) and no.attr == "SerrfCorrector"
        ):
            linhas.append(no.lineno)
    return sorted(linhas)


def test_d681_nenhum_construtor_do_produto_monta_o_serrf_sem_a_vez() -> None:
    """Em `src/refino/`, fora de `serrf.py` e de `serrf_na_vez.py`, o `SerrfCorrector` só aparece
    em anotação.

    **Por que um portão, e não só os testes de comportamento:** eles cobrem os seis chamadores que
    havia quando foram escritos. Um construtor novo — uma rota, um script do produto, uma tabela —
    que montasse o `SerrfCorrector` à mão voltaria a travar o processo ao lado de outro `n_jobs`,
    sem nada a reprovar. **Contado antes de ser escrito**, na árvore anterior ao conserto: seis usos
    como valor — a tabela `AVAILABLE_METHODS` e os construtores da CLI, das duas fábricas das rotas,
    da rotulagem e do executor —, e é esse o conjunto que o conserto mudou.
    """
    pacote = Path(refino.__file__).resolve().parent
    arquivos = sorted(pacote.rglob("*.py"))
    assert len(arquivos) > 50, f"âncora: o portão leu só {len(arquivos)} arquivos de {pacote}"
    acusados = [
        f"{caminho.relative_to(pacote)}:{linha}"
        for caminho in arquivos
        if caminho.relative_to(pacote) not in _ISENTOS
        for linha in _usos_como_valor(caminho.read_text(encoding="utf-8"))
    ]
    assert not acusados, (
        "o `SerrfCorrector` é usado como valor fora da subclasse que pede a vez; monte "
        f"`SerrfNaVez`, de `correction/serrf_na_vez.py`: {acusados}"
    )


def test_d681_o_portao_acusa_o_construtor_plantado_e_poupa_a_anotacao() -> None:
    """O par plantado do portão: a chamada, a tabela, o atributo e o apelido acusam; a anotação e o
    `isinstance` não.

    Sem a metade da anotação, um portão que acusasse todo `SerrfCorrector` passaria aqui e
    reprovaria a árvore inteira por causa das assinaturas; sem a outra, um portão que não acusasse
    nada passaria no teste de cima por vacuidade.
    """
    plantado = (
        "from refino.correction import serrf\n"
        "from refino.correction.serrf import SerrfCorrector\n"
        "def fabrica(n: int) -> SerrfCorrector:\n"
        "    return SerrfCorrector(n_jobs=n)\n"
        "TABELA = {'serrf': SerrfCorrector}\n"
        "outro = serrf.SerrfCorrector(n_jobs=-1)\n"
        "x: SerrfCorrector | None = None\n"
    )
    assert _usos_como_valor(plantado) == [4, 5, 6]
    com_apelido = (
        "from refino.correction.serrf import SerrfCorrector as S\ncorretor = S(n_jobs=-1)\n"
    )
    assert _usos_como_valor(com_apelido) == [2], "o apelido da importação escapou do portão"
    so_anotacao = (
        "from refino.correction.serrf import SerrfCorrector\n"
        "def fabrica(c: SerrfCorrector, *a: SerrfCorrector) -> SerrfCorrector:\n"
        "    return c\n"
        "e_dele = isinstance(object(), SerrfCorrector)\n"
        "e_sub = issubclass(type(object()), (SerrfCorrector, int))\n"
    )
    assert _usos_como_valor(so_anotacao) == []
