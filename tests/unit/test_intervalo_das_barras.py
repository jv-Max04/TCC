"""O intervalo por barra do painel da correção: jackknife-t sobre os QCs, com reajuste.

A barra de cada corretor é a mediana, sobre as features, do RSD dos QCs fora da amostra
(`leave_one_qc_out_rsd`). O replicado `j` do jackknife é essa régua refeita sem o QC `j`: cada um
dos outros QCs `i` é retido, e o corretor ajusta-se com `i` e `j` remarcados como `sample` (a
convenção da régua). **Um ajuste serve aos dois QCs do par**, então o intervalo custa `n(n-1)/2`
ajustes a mais e a medição inteira `n(n+1)/2`: a razão é `(n+1)/2`.

**As matrizes daqui são construídas à mão, e não são dado do laboratório (regra 1 do README).** O
que estes testes afirmam é aritmética e contrato — quantos ajustes, que linha de que ajuste, a
ordem, a semente, a recusa —, e nada disso depende de os números serem os de um estudo real. A
cobertura do intervalo NÃO é afirmada aqui, nem em lado nenhum: o gerador do projeto precisa da
matriz real como fonte, e ela não está nesta máquina.
"""

from __future__ import annotations

import warnings
from dataclasses import replace
from typing import Any

import numpy as np
import pytest
from scipy import stats

from refino.correction.loess import LoessCorrector
from refino.correction.none import NoneCorrector
from refino.decision import labeling
from refino.decision.labeling import LabelingOracle, build_corrector_for_labeling
from refino.domain.cancellation import CancelCheck, ExecutionCancelledError
from refino.domain.correction import CorrectionResult
from refino.domain.labels import InvalidLabelError
from refino.domain.matrix import RefinoMatrix
from refino.domain.metadata import SampleMetadata
from refino.metrics import crossval
from refino.metrics.crossval import (
    JACKKNIFE_LEVEL,
    MIN_QC_PER_BATCH_FOR_JACKKNIFE,
    held_out_pairs,
    held_out_rows,
    jackknife_cost_ratio,
    jackknife_qc_minimum,
    leave_one_qc_out_rsd,
    leave_one_qc_out_rsd_with_jackknife,
    out_of_sample_values,
    pair_out_of_sample_values,
    remark_as_sample,
    remark_pair_as_sample,
)
from refino.metrics.rsd import rsd

VARIANTE = "outlier_removed"


def matriz_com_deriva(
    n_qc: int = 7,
    n_amostras: int = 34,
    n_features: int = 8,
    semente: int = 5,
    lotes_dos_qcs: tuple[str, ...] | None = None,
) -> RefinoMatrix:
    """Deriva suave ao longo da injeção e ruído multiplicativo — o LOESS tem o que corrigir.

    Os QCs espalhados pela ordem de injeção, como nos arquivos do estudo; o lote de cada amostra é
    o do QC mais próximo à esquerda quando `lotes_dos_qcs` é dado, para os blocos serem contíguos.
    """
    rng = np.random.default_rng(semente)
    posicoes = tuple(round(i) for i in np.linspace(0, n_amostras - 1, n_qc))
    ordem = np.arange(n_amostras, dtype=np.float64)
    nivel = rng.uniform(1e3, 1e5, size=n_features)
    deriva = 1.0 + 0.4 * np.sin(ordem[:, None] / 9.0 + np.arange(n_features)[None, :])
    ruido = rng.lognormal(0.0, 0.08, size=(n_amostras, n_features))
    valores = nivel[None, :] * deriva * ruido

    indice = {linha: k for k, linha in enumerate(posicoes)}
    amostras = []
    lote_corrente = "A"
    for i in range(n_amostras):
        if lotes_dos_qcs is not None and i in indice:
            lote_corrente = lotes_dos_qcs[indice[i]]
        amostras.append(
            SampleMetadata(
                name=f"s{i:03d}",
                injection_order=i + 1,
                sample_type="qc" if i in indice else "sample",
                group="qc" if i in indice else "case",
                batch=lote_corrente,
            )
        )
    return RefinoMatrix(
        values=valores,
        feature_names=tuple(f"f{j}" for j in range(n_features)),
        samples=tuple(amostras),
        mode="synthetic",
    )


class _Contador:
    """Envolve um corretor e conta as chamadas a `fit_transform` — no processo do chamador."""

    def __init__(self, interno: Any) -> None:
        self.interno = interno
        self.name = interno.name
        self.n_jobs = getattr(interno, "n_jobs", 1)
        # Repassado: o estimador lê o mínimo de treino do corretor por atributo, e um envoltório
        # que o escondesse mediria o piso da régua em vez do do corretor que envolve.
        if hasattr(interno, "min_train_qc_per_batch"):
            self.min_train_qc_per_batch = interno.min_train_qc_per_batch
        self.chamadas = 0

    def fit_transform(self, matrix: RefinoMatrix, **kwargs: Any) -> CorrectionResult:
        self.chamadas += 1
        return self.interno.fit_transform(matrix, **kwargs)  # type: ignore[no-any-return]


class _RuidoPelaSemente:
    """Corretor de teste cuja saída DEPENDE da semente: multiplica por ruído sorteado dela.

    Existe para o par causal do determinismo: com o LOESS, que não sorteia nesta matriz, trocar a
    semente não mudaria nada, e a asserção «outra semente dá outro intervalo» passaria por vácuo.
    Módulo-nível porque o `loky` o serializa.
    """

    name = "ruido-pela-semente"

    def fit_transform(
        self,
        matrix: RefinoMatrix,
        *,
        batches: np.ndarray | None = None,
        seed: int = 42,
        target_features: frozenset[str] | None = None,
        cancelled: Any = None,
    ) -> CorrectionResult:
        del batches, target_features, cancelled
        rng = np.random.default_rng([seed, int(matrix.qc_mask.sum())])
        fator = 1.0 + 0.05 * rng.standard_normal(matrix.values.shape)
        return CorrectionResult(
            matrix=matrix.with_values(matrix.values * fator, "correction:ruido"), uncorrected={}
        )


def _depois_de(n: int) -> CancelCheck:
    contador = {"n": 0}

    def bandeira() -> bool:
        contador["n"] += 1
        return contador["n"] > n

    return bandeira


def _mediana(linhas: np.ndarray) -> float:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return float(np.nanmedian(rsd(linhas, axis=0, variant=VARIANTE)))


def _leituras_da_regua(matriz: RefinoMatrix, n_jobs: int) -> int:
    """Quantas vezes a régua de base consulta a bandeira — medido, e não contado à mão.

    É o que deixa os testes da parada pôr a primeira resposta «sim» exactamente no primeiro ajuste
    de par, sem pinar um número que muda no dia em que o corretor ganhar uma checagem.
    """
    leituras = {"n": 0}

    def conta() -> bool:
        leituras["n"] += 1
        return False

    leave_one_qc_out_rsd(matriz, NoneCorrector(), variant=VARIANTE, n_jobs=n_jobs, cancelled=conta)
    return leituras["n"]


def _ep(replicados: list[float]) -> float:
    r = np.asarray(replicados, dtype=np.float64)
    return float(np.sqrt((r.size - 1) / r.size * np.sum((r - r.mean()) ** 2)))


def _matriz_do_serrf() -> RefinoMatrix:
    """A menor matriz à mão em que o SERRF real CORRIGE: onze features e quatro QCs.

    **Com as oito features da matriz padrão o SERRF não corrige nada** — medido: os 28 ajustes do
    intervalo voltam com as oito features marcadas `empty_predictor_intersection`, porque o passo 3
    procura `N_CORRELATED = 10` parceiras, e o erro-padrão sai o do `none`, idêntico. Um teste
    sobre ela passaria o SERRF pelo caminho sem que ele ajustasse coisa nenhuma. Com onze o
    ajuste inteiro não marca feature nenhuma; quatro QCs porque cada ajuste custa ~2,6 s aqui.
    """
    return matriz_com_deriva(n_qc=4, n_amostras=16, n_features=11)


_CORRETORES_QUE_AJUSTAM = [
    pytest.param(LoessCorrector(n_jobs=1), matriz_com_deriva(), id="loess"),
    pytest.param(
        build_corrector_for_labeling("serrf", n_jobs=1),
        _matriz_do_serrf(),
        id="serrf",
        # **`slow` pelo custo, medido:** ~2,6 s por ajuste do SERRF nesta matriz, num Linux de 4
        # núcleos partilhados, e os dois testes que o recebem fazem 14 e 22 ajustes.
        marks=pytest.mark.slow,
    ),
]


# ---------------------------------------------------------------------------
# A conta dos ajustes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_qc", [4, 7, 11])
def test_sem_o_intervalo_a_regua_faz_n_ajustes_e_com_ele_n_n_mais_1_sobre_2(n_qc: int) -> None:
    """**Quem não pede não paga nada:** `n` ajustes sem o intervalo, `n(n+1)/2` com ele.

    E a razão que a janela diz ao lado da caixa, `jackknife_cost_ratio`, é a desta contagem e não
    uma fórmula escrita à parte: `(n+1)/2` é 2,5 com quatro QCs, 4 com sete, 6 com onze.
    """
    matriz = matriz_com_deriva(n_qc=n_qc, n_amostras=4 * n_qc)
    sem = _Contador(NoneCorrector())
    leave_one_qc_out_rsd(matriz, sem, variant=VARIANTE)  # type: ignore[arg-type]
    assert sem.chamadas == n_qc, f"a régua sem o intervalo fez {sem.chamadas} ajustes, e não {n_qc}"

    com = _Contador(NoneCorrector())
    _vetor, intervalo = leave_one_qc_out_rsd_with_jackknife(matriz, com, variant=VARIANTE)  # type: ignore[arg-type]
    esperados = n_qc * (n_qc + 1) // 2
    assert intervalo.reason is None, intervalo.reason
    assert com.chamadas == esperados, (
        f"com o intervalo foram {com.chamadas} ajustes, e a conta é n(n+1)/2 = {esperados}: "
        "o par de QCs deixou de ser um ajuste só, ou a régua de base correu duas vezes"
    )
    assert jackknife_cost_ratio(n_qc) == com.chamadas / sem.chamadas == (n_qc + 1) / 2


def test_os_pares_sao_os_n_n_menos_1_sobre_2_sem_repeticao_e_em_ordem() -> None:
    """Cada par de QCs uma vez, `(menor, maior)`, na ordem das linhas — a determinística."""
    matriz = matriz_com_deriva()
    qcs = [int(r) for r in held_out_rows(matriz)]
    pares = held_out_pairs(matriz)
    assert len(pares) == len(qcs) * (len(qcs) - 1) // 2
    assert len(set(pares)) == len(pares)
    assert all(a < b and a in qcs and b in qcs for a, b in pares)
    assert list(pares) == sorted(pares)


# ---------------------------------------------------------------------------
# A barra é a da régua, e o intervalo contém-na
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("corretor", "matriz"), _CORRETORES_QUE_AJUSTAM)
def test_o_intervalo_contem_a_estimativa_e_a_barra_e_a_da_regua_bit_a_bit(
    corretor: Any, matriz: RefinoMatrix
) -> None:
    """O vetor por feature é o de `leave_one_qc_out_rsd`, bit a bit, e o intervalo cerca a mediana.

    A largura é `2 t EP`, com `t` de Student com `n - 1` graus de liberdade ao nível declarado; a
    âncora de não-vacuidade é o erro-padrão positivo — com ele zero, «contém» seria trivial.

    **E o centro é a BARRA, e não a média dos replicados** (o jackknife-t centrado na estimativa,
    para o traço cercar o número que o produto mostra). `lower < estimativa < upper` e a
    largura `2 t EP` são satisfeitos também pelo intervalo centrado na média — foi a planta que o
    mostrou, verde —, então o centro é afirmado à parte, com a âncora de que a média dos
    replicados se afasta da barra neste dado.
    """
    da_regua = leave_one_qc_out_rsd(matriz, corretor, variant=VARIANTE)
    vetor, intervalo = leave_one_qc_out_rsd_with_jackknife(matriz, corretor, variant=VARIANTE)

    assert vetor.tobytes() == da_regua.tobytes(), "a barra deixou de ser a da régua do produto"
    assert (
        vetor.tobytes() != leave_one_qc_out_rsd(matriz, NoneCorrector(), variant=VARIANTE).tobytes()
    ), "âncora: o corretor não ajustou nada nesta matriz, e o teste seria o do `none`"
    assert intervalo.reason is None, intervalo.reason
    assert intervalo.standard_error is not None and intervalo.standard_error > 0.0, (
        "âncora: erro-padrão zero, e a contenção abaixo seria trivial"
    )
    assert intervalo.lower is not None and intervalo.upper is not None
    assert intervalo.lower < intervalo.estimate < intervalo.upper
    assert intervalo.estimate == float(np.nanmedian(da_regua))
    t = float(stats.t.ppf(0.5 + JACKKNIFE_LEVEL / 2.0, intervalo.n_qc - 1))
    assert intervalo.upper - intervalo.lower == pytest.approx(
        2.0 * t * intervalo.standard_error, rel=1e-12
    )
    media = float(np.mean(intervalo.replicates))
    assert abs(media - intervalo.estimate) > 1e-6 * abs(intervalo.estimate), (
        "âncora: a média dos replicados coincide com a barra neste dado, e o centro não se separa"
    )
    assert (intervalo.lower + intervalo.upper) / 2.0 == pytest.approx(
        intervalo.estimate, rel=1e-12
    ), "o intervalo deixou de ser centrado na barra"
    n_qc = int(matriz.qc_mask.sum())
    assert intervalo.level == JACKKNIFE_LEVEL and intervalo.n_qc == n_qc
    assert len(intervalo.replicates) == n_qc


def test_o_erro_padrao_e_o_do_jackknife_sobre_os_replicados() -> None:
    """`sqrt((n-1)/n soma (t_j - media)^2)`, recalculado aqui a partir dos replicados devolvidos."""
    matriz = matriz_com_deriva()
    _vetor, intervalo = leave_one_qc_out_rsd_with_jackknife(
        matriz, LoessCorrector(n_jobs=1), variant=VARIANTE
    )
    assert intervalo.standard_error == pytest.approx(_ep(list(intervalo.replicates)), rel=1e-12)


def test_um_nivel_menor_da_um_intervalo_mais_estreito_e_fora_de_0_1_recusa() -> None:
    matriz = matriz_com_deriva()
    corretor = NoneCorrector()
    _v, a95 = leave_one_qc_out_rsd_with_jackknife(matriz, corretor, variant=VARIANTE)
    _v, a80 = leave_one_qc_out_rsd_with_jackknife(matriz, corretor, variant=VARIANTE, level=0.80)
    assert a95.upper is not None and a95.lower is not None
    assert a80.upper is not None and a80.lower is not None
    assert a80.upper - a80.lower < a95.upper - a95.lower
    for nivel in (0.0, 1.0, 1.5, float("nan")):
        with pytest.raises(ValueError, match="nível"):
            leave_one_qc_out_rsd_with_jackknife(matriz, corretor, variant=VARIANTE, level=nivel)


def test_a_variante_invalida_levanta_na_fronteira() -> None:
    """A variante é `Literal`, e um valor de fora não cai num `else` razoável."""
    with pytest.raises(InvalidLabelError):
        leave_one_qc_out_rsd_with_jackknife(matriz_com_deriva(), NoneCorrector(), variant="robusta")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Um ajuste serve aos dois QCs do par
# ---------------------------------------------------------------------------


def test_um_ajuste_serve_aos_dois_qcs_do_par_e_a_ordem_de_remarcar_nao_importa() -> None:
    """**A identidade que faz o custo ser `n(n-1)/2` e não `n(n-1)`.** Remarcar `i` e depois `j`,
    ou `j` e depois `i`, dá a mesma matriz de tipos, e o LOESS — determinístico, que não lê o
    `provenance` — devolve as mesmas linhas, bit a bit.
    """
    matriz = matriz_com_deriva()
    corretor = LoessCorrector(n_jobs=1)
    i, j = (int(r) for r in held_out_rows(matriz)[:2])
    direto = remark_pair_as_sample(matriz, (i, j))
    invertido = remark_as_sample(remark_as_sample(matriz, j), i)
    a = corretor.fit_transform(direto, seed=42).matrix.values
    b = corretor.fit_transform(invertido, seed=42).matrix.values
    assert a[[i, j], :].tobytes() == b[[i, j], :].tobytes()
    assert direto.qc_mask.sum() == matriz.qc_mask.sum() - 2


@pytest.mark.parametrize(
    ("corretor", "matriz"),
    [
        *_CORRETORES_QUE_AJUSTAM,
        pytest.param(_RuidoPelaSemente(), matriz_com_deriva(), id="ruido-pela-semente"),
    ],
)
def test_os_replicados_sao_os_da_conta_por_extenso_com_n_n_menos_1_ajustes(
    corretor: Any, matriz: RefinoMatrix
) -> None:
    """**A identidade, no que o estimador DEVOLVE.** A conta por extenso faz um ajuste por par
    ORDENADO `(j, i)` — remarca `j`, depois `i`, e lê a linha de `i` —, `n(n-1)` ajustes; a do
    produto faz um por par e lê as duas linhas dele. Os replicados têm de ser os mesmos, bit a bit.

    **Semente 7 dos dois lados, e não o padrão.** Com o LOESS desta matriz a semente não muda nada,
    e uma semente outra nos ajustes de par — o replicado deixaria de ser «a régua refeita sem `j`»
    e somaria o ruído do sorteio ao erro-padrão — passava verde: foi a planta `seed + 1` nos pares
    que o mostrou. O corretor que sorteia pela semente, e o SERRF, são os que a apanham.
    """
    qcs = [int(r) for r in held_out_rows(matriz)]
    _v, intervalo = leave_one_qc_out_rsd_with_jackknife(matriz, corretor, variant=VARIANTE, seed=7)

    por_extenso = []
    for j in qcs:
        sem_j = remark_as_sample(matriz, j)
        linhas = [
            corretor.fit_transform(remark_as_sample(sem_j, i), seed=7).matrix.values[i, :]
            for i in qcs
            if i != j
        ]
        por_extenso.append(_mediana(np.vstack(linhas)))
    assert np.array(intervalo.replicates).tobytes() == np.array(por_extenso).tobytes()
    _v, do_none = leave_one_qc_out_rsd_with_jackknife(
        matriz, NoneCorrector(), variant=VARIANTE, seed=7
    )
    assert intervalo.replicates != do_none.replicates, (
        "âncora: os replicados são os do `none`, e o corretor não ajustou nada nesta matriz"
    )


def test_a_planta_que_le_a_linha_errada_do_par_muda_os_replicados(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """O teste acima consegue reprovar: ler a linha do OUTRO QC do par muda os replicados.

    A planta é no passo nomeado `_row_of_pair`, e a asserção corre pelo estimador de verdade — a
    forma «em processo». A âncora é que as duas leituras diferem neste dado.
    """
    matriz = matriz_com_deriva()
    corretor = LoessCorrector(n_jobs=1)
    _v, limpo = leave_one_qc_out_rsd_with_jackknife(matriz, corretor, variant=VARIANTE)
    original = crossval._row_of_pair

    def trocada(pares: Any, i: int, j: int) -> np.ndarray:
        return original(pares, j, i)  # type: ignore[no-any-return]

    monkeypatch.setattr(crossval, "_row_of_pair", trocada)
    _v, plantado = leave_one_qc_out_rsd_with_jackknife(matriz, corretor, variant=VARIANTE)
    assert plantado.replicates != limpo.replicates


# ---------------------------------------------------------------------------
# A âncora do arnês: com `none`, reajustar não muda nada
# ---------------------------------------------------------------------------


def _sem_reajuste(matriz: RefinoMatrix, corretor: Any) -> list[float]:
    """O jackknife SEM reajuste: as `n` linhas de base, uma de fora por vez — zero ajustes."""
    base = out_of_sample_values(matriz, corretor, seed=42)
    n = base.shape[0]
    return [_mediana(base[[k for k in range(n) if k != j], :]) for j in range(n)]


def test_com_none_o_jackknife_com_reajuste_e_o_sem_reajuste_coincidem_por_construcao() -> None:
    """**A âncora do arnês.** O `none` não treina nada: a linha de `i` é a mesma em qualquer
    ajuste, e os dois jackknifes têm de dar os mesmos replicados — bit a bit. Se divergissem, o
    arnês estaria a ler outra linha ou outro par, e não o reajuste.
    """
    matriz = matriz_com_deriva()
    _v, intervalo = leave_one_qc_out_rsd_with_jackknife(matriz, NoneCorrector(), variant=VARIANTE)
    sem = _sem_reajuste(matriz, NoneCorrector())
    assert np.array(intervalo.replicates).tobytes() == np.array(sem).tobytes()
    assert intervalo.standard_error == _ep(sem)


def test_com_o_loess_o_sem_reajuste_mede_outra_coisa() -> None:
    """A planta da âncora: com um corretor que treina, o atalho sem reajuste dá OUTROS replicados.

    É o que impede a âncora de cima de passar por um estimador que ignorasse o reajuste: no `none`
    as duas contas coincidem, e no LOESS têm de divergir.
    """
    matriz = matriz_com_deriva()
    corretor = LoessCorrector(n_jobs=1)
    _v, intervalo = leave_one_qc_out_rsd_with_jackknife(matriz, corretor, variant=VARIANTE)
    assert list(intervalo.replicates) != _sem_reajuste(matriz, corretor)


# ---------------------------------------------------------------------------
# Determinismo, e vários núcleos iguais à série
# ---------------------------------------------------------------------------


def test_a_mesma_semente_da_o_mesmo_intervalo_e_outra_semente_da_outro() -> None:
    """Regra 3 do README, com o par causal: o corretor de teste sorteia pela semente."""
    matriz = matriz_com_deriva()
    corretor = _RuidoPelaSemente()
    a = leave_one_qc_out_rsd_with_jackknife(matriz, corretor, variant=VARIANTE, seed=7)  # type: ignore[arg-type]
    b = leave_one_qc_out_rsd_with_jackknife(matriz, corretor, variant=VARIANTE, seed=7)  # type: ignore[arg-type]
    c = leave_one_qc_out_rsd_with_jackknife(matriz, corretor, variant=VARIANTE, seed=8)  # type: ignore[arg-type]
    assert a[0].tobytes() == b[0].tobytes() and a[1] == b[1]
    assert a[1].replicates != c[1].replicates


@pytest.mark.parametrize("corretor", [LoessCorrector(n_jobs=1), _RuidoPelaSemente()])
def test_dois_nucleos_dao_o_intervalo_da_serie_bit_a_bit(corretor: Any) -> None:
    """Como na régua, aqui para os pares: os ajustes voltam na ordem dos pares."""
    matriz = matriz_com_deriva()
    # **Semente 7, e não o padrão:** com o 42 dos dois lados, um braço paralelo que ignorasse a
    # semente pedida e usasse o padrão daria o mesmo número — o valor que não exercita. Foi a
    # planta «semente fixa nos pares» que o mostrou, verde com o padrão.
    serie = leave_one_qc_out_rsd_with_jackknife(
        matriz, corretor, variant=VARIANTE, n_jobs=1, seed=7
    )
    paralelo = leave_one_qc_out_rsd_with_jackknife(
        matriz, corretor, variant=VARIANTE, n_jobs=2, seed=7
    )
    assert serie[1].reason is None
    assert paralelo[0].tobytes() == serie[0].tobytes()
    assert np.array(paralelo[1].replicates).tobytes() == np.array(serie[1].replicates).tobytes()
    assert paralelo[1] == serie[1]


def test_os_dois_niveis_de_paralelismo_juntos_sao_recusados() -> None:
    with pytest.raises(ValueError, match="dois níveis"):
        leave_one_qc_out_rsd_with_jackknife(
            matriz_com_deriva(), LoessCorrector(n_jobs=2), variant=VARIANTE, n_jobs=2
        )


# ---------------------------------------------------------------------------
# A parada: o despacho para de produzir, e nada levanta dentro do `Parallel`
# ---------------------------------------------------------------------------


def test_em_serie_a_parada_e_atendida_entre_os_ajustes_de_par() -> None:
    """As consultas da régua de base, mais a de antes dos pares, respondem «não»; a seguinte é já a
    de um par, e é ela que levanta — com o sítio dos pares, e não o da régua."""
    matriz = matriz_com_deriva()
    base = _leituras_da_regua(matriz, n_jobs=1)
    with pytest.raises(ExecutionCancelledError) as capturado:
        leave_one_qc_out_rsd_with_jackknife(
            matriz, NoneCorrector(), variant=VARIANTE, cancelled=_depois_de(base + 1)
        )
    assert capturado.value.where == "jackknife: antes do ajuste de um par de QCs"


def test_em_varios_nucleos_a_parada_e_no_despacho_dos_pares() -> None:
    """Com dois núcleos a bandeira é lida no despacho, no pai (`CancellableDispatch`): o
    despacho para de produzir, o `Parallel` acaba os que já saíram, e a exceção vem depois."""
    matriz = matriz_com_deriva()
    base = _leituras_da_regua(matriz, n_jobs=2)
    with pytest.raises(ExecutionCancelledError) as capturado:
        leave_one_qc_out_rsd_with_jackknife(
            matriz, NoneCorrector(), variant=VARIANTE, n_jobs=2, cancelled=_depois_de(base + 1)
        )
    assert capturado.value.where == "jackknife: despacho de um par de QCs"


class _PoolQueMorre:
    """Um `Parallel` cujo pool morre depois de consumir uma tarefa — a planta do recuo do LOESS."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs

    def __call__(self, tarefas: Any) -> Any:
        from joblib.externals.loky.process_executor import TerminatedWorkerError

        next(iter(tarefas))
        raise TerminatedWorkerError("planta: o worker morreu a meio")


class _PoolQueFalha:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs

    def __call__(self, tarefas: Any) -> Any:
        next(iter(tarefas))
        raise RuntimeError("planta: defeito que não é a morte do pool")


def test_o_pool_que_morre_nos_pares_recua_para_a_serie_com_a_mesma_saida(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """O recuo de `pair_out_of_sample_values`: o pool que morre a meio refaz os pares em série, com
    a saída da série bit a bit, e deixa registro. Com o corretor que sorteia, semente 7."""
    import logging

    matriz = matriz_com_deriva()
    corretor = _RuidoPelaSemente()
    serie = pair_out_of_sample_values(matriz, corretor, seed=7, n_jobs=1)  # type: ignore[arg-type]
    monkeypatch.setattr(crossval, "Parallel", _PoolQueMorre)
    with caplog.at_level(logging.WARNING, logger="refino.metrics.crossval"):
        recuada = pair_out_of_sample_values(matriz, corretor, seed=7, n_jobs=2)  # type: ignore[arg-type]

    assert list(recuada) == list(serie) == list(held_out_pairs(matriz))
    for par in serie:
        assert recuada[par][0].tobytes() == serie[par][0].tobytes()
        assert recuada[par][1].tobytes() == serie[par][1].tobytes()
    assert any("SERIE" in r.getMessage() and "par" in r.getMessage() for r in caplog.records), (
        "o recuo para série aconteceu sem deixar registro"
    )


def test_nos_pares_so_a_morte_do_pool_recua_e_outra_falha_sobe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A outra direção do recuo: um `except` largo esconderia defeito como lentidão."""
    monkeypatch.setattr(crossval, "Parallel", _PoolQueFalha)
    with pytest.raises(RuntimeError, match="planta"):
        pair_out_of_sample_values(matriz_com_deriva(), _RuidoPelaSemente(), seed=7, n_jobs=2)  # type: ignore[arg-type]


def test_sem_bandeira_levantada_os_dois_nucleos_medem_tudo() -> None:
    matriz = matriz_com_deriva()
    _v, intervalo = leave_one_qc_out_rsd_with_jackknife(
        matriz, NoneCorrector(), variant=VARIANTE, n_jobs=2, cancelled=lambda: False
    )
    assert intervalo.reason is None and len(intervalo.replicates) == 7


# ---------------------------------------------------------------------------
# Sem intervalo: o motivo, e não um número
# ---------------------------------------------------------------------------


def test_com_tres_qcs_ha_barra_e_nao_ha_intervalo_e_nenhum_par_e_ajustado() -> None:
    """Cada replicado tira um QC e refaz a régua, que precisa de três por lote: o intervalo
    exige quatro. Com três há barra — a régua de hoje aceita-os — e o intervalo vem com o motivo,
    sem número, e sem gastar um ajuste de par sequer.
    """
    matriz = matriz_com_deriva(n_qc=3, n_amostras=15)
    contador = _Contador(NoneCorrector())
    vetor, intervalo = leave_one_qc_out_rsd_with_jackknife(matriz, contador, variant=VARIANTE)  # type: ignore[arg-type]
    assert np.isfinite(np.nanmedian(vetor)), "âncora: a barra existe"
    assert contador.chamadas == 3, f"{contador.chamadas} ajustes: algum par foi ajustado à toa"
    assert intervalo.reason is not None and str(MIN_QC_PER_BATCH_FOR_JACKKNIFE) in intervalo.reason
    assert intervalo.lower is None and intervalo.upper is None and intervalo.standard_error is None
    assert intervalo.replicates == ()


def test_por_lote_o_lote_de_tres_qcs_tira_o_intervalo_e_o_de_quatro_nao() -> None:
    """A guarda é POR LOTE, como a da régua: 4 + 3 não tem intervalo; 4 + 4 tem — com o `none`,
    que não declara mínimo de treino. Com o LOESS o mínimo é cinco, no teste seguinte."""
    de_tres = matriz_com_deriva(n_qc=7, lotes_dos_qcs=("A",) * 4 + ("B",) * 3)
    _v, sem = leave_one_qc_out_rsd_with_jackknife(de_tres, NoneCorrector(), variant=VARIANTE)
    assert sem.reason is not None and "B=3" in sem.reason, sem.reason

    de_quatro = matriz_com_deriva(n_qc=8, lotes_dos_qcs=("A",) * 4 + ("B",) * 4)
    _v, com = leave_one_qc_out_rsd_with_jackknife(de_quatro, NoneCorrector(), variant=VARIANTE)
    assert com.reason is None and len(com.replicates) == 8


def test_com_o_loess_um_lote_de_quatro_qcs_da_motivo_porque_o_par_treinaria_com_dois() -> None:
    """**O mínimo é o do CORRETOR, e não o piso da régua, quando o corretor exige mais.**

    Um ajuste de par treina com dois QCs a menos no lote. O LOESS recusa o lote com menos de três
    (`MIN_QC_FOR_FIT`) e devolve a feature sem correção; num lote de quatro, o par de dentro dele
    treina com dois. Com o piso de quatro para todos, o intervalo saía SEM motivo — 12 dos 28
    ajustes de par em `fit_failed`, os replicados de dado não corrigido em volta de uma barra
    corrigida. Agora o mínimo do LOESS é cinco, o motivo o diz, e nenhum par é ajustado.

    **A âncora vem primeiro:** o par de dentro de um lote de quatro falha DE FACTO em todas
    as features neste dado, e o de 5 + 5 dá intervalo — sem ela, a guarda podia estar a recusar um
    caso que o corretor aguenta.
    """
    de_quatro = matriz_com_deriva(
        n_qc=8, n_amostras=40, n_features=9, semente=11, lotes_dos_qcs=("A",) * 4 + ("B",) * 4
    )
    a, b = (int(r) for r in held_out_rows(de_quatro)[:2])
    par = LoessCorrector(n_jobs=1).fit_transform(remark_pair_as_sample(de_quatro, (a, b)), seed=42)
    assert par.uncorrected.get("fit_failed") == frozenset(de_quatro.feature_names), (
        "âncora: o par de dentro do lote de quatro já não falha no LOESS, e a guarda recusaria "
        "à toa"
    )
    assert jackknife_qc_minimum(LoessCorrector()) == 5

    contador = _Contador(LoessCorrector(n_jobs=1))
    vetor, intervalo = leave_one_qc_out_rsd_with_jackknife(de_quatro, contador, variant=VARIANTE)  # type: ignore[arg-type]
    assert np.isfinite(np.nanmedian(vetor)), (
        "âncora: a barra existe — o lote de quatro treina com três"
    )
    assert contador.chamadas == 8, f"{contador.chamadas} ajustes: algum par foi ajustado à toa"
    assert intervalo.reason is not None and intervalo.standard_error is None, intervalo
    assert "5" in intervalo.reason and "A=4" in intervalo.reason and "loess" in intervalo.reason

    de_cinco = matriz_com_deriva(
        n_qc=10, n_amostras=50, n_features=9, semente=11, lotes_dos_qcs=("A",) * 5 + ("B",) * 5
    )
    _v, com = leave_one_qc_out_rsd_with_jackknife(
        de_cinco, LoessCorrector(n_jobs=1), variant=VARIANTE
    )
    assert com.reason is None and len(com.replicates) == 10


def test_o_minimo_do_intervalo_e_o_piso_ou_o_do_corretor_mais_dois() -> None:
    """`jackknife_qc_minimum`: o piso da régua (quatro) para quem não declara, ou declara menos; o
    mínimo de treino declarado mais dois para quem exige mais."""

    class _Declara:
        name = "declara"

        def __init__(self, minimo: int) -> None:
            self.min_train_qc_per_batch = minimo

    assert jackknife_qc_minimum(NoneCorrector()) == MIN_QC_PER_BATCH_FOR_JACKKNIFE == 4
    assert jackknife_qc_minimum(build_corrector_for_labeling("serrf", n_jobs=1)) == 4
    assert jackknife_qc_minimum(LoessCorrector()) == 5
    assert jackknife_qc_minimum(_Declara(1)) == 4  # type: ignore[arg-type]
    assert jackknife_qc_minimum(_Declara(6)) == 8  # type: ignore[arg-type]


def test_barra_indefinida_da_motivo_e_nao_intervalo() -> None:
    """QCs todos zero: o RSD é indefinido em toda feature, a barra é `nan`, e o intervalo diz por
    quê — não vira zero, não vira `nan` num campo numérico."""
    matriz = matriz_com_deriva()
    valores = np.array(matriz.values, copy=True)
    valores[matriz.qc_mask, :] = 0.0
    zerada = replace(matriz, values=valores)
    vetor, intervalo = leave_one_qc_out_rsd_with_jackknife(
        zerada, NoneCorrector(), variant=VARIANTE
    )
    assert np.all(np.isnan(vetor))
    assert intervalo.reason is not None and "indefinid" in intervalo.reason
    assert intervalo.lower is None and intervalo.standard_error is None


def test_matriz_sem_qc_nao_ajusta_e_diz_por_que() -> None:
    matriz = matriz_com_deriva()
    sem_qc = replace(
        matriz,
        samples=tuple(replace(a, sample_type="sample", group="case") for a in matriz.samples),
    )
    contador = _Contador(NoneCorrector())
    vetor, intervalo = leave_one_qc_out_rsd_with_jackknife(sem_qc, contador, variant=VARIANTE)  # type: ignore[arg-type]
    assert contador.chamadas == 0 and np.all(np.isnan(vetor))
    assert intervalo.reason is not None and intervalo.n_qc == 0


def test_o_intervalo_sem_motivo_tem_os_tres_numeros_e_com_motivo_nenhum() -> None:
    """A forma é fechada nas duas direções: o construtor recusa as misturas."""
    with pytest.raises(ValueError, match="motivo"):
        crossval.JackknifeInterval(estimate=0.1, level=0.95, n_qc=7)
    with pytest.raises(ValueError, match="motivo"):
        crossval.JackknifeInterval(
            estimate=0.1, level=0.95, n_qc=7, lower=0.0, upper=0.2, standard_error=0.05, reason="x"
        )


# ---------------------------------------------------------------------------
# Amostra de features: os replicados correm sobre as MESMAS features que a barra
# ---------------------------------------------------------------------------


def test_com_amostra_de_features_o_intervalo_e_sobre_as_mesmas_da_barra() -> None:
    """Os replicados correm sobre as features sorteadas para a barra, e não sobre a matriz inteira.

    **A âncora vem antes, e a primeira escrita deste teste precisou dela:** com quatro
    features sorteadas de dez, a mediana das quatro coincidia com a das dez no primeiro replicado —
    0,25796 nas duas —, e a planta que fazia os pares sobre a matriz inteira passava. Agora a conta
    compara os SETE replicados, e a âncora afirma que as duas populações dão replicados diferentes
    neste dado.
    """
    matriz = matriz_com_deriva(n_features=10)
    vetor, intervalo = leave_one_qc_out_rsd_with_jackknife(
        matriz, NoneCorrector(), variant=VARIANTE, n_sampled_features=4
    )
    da_regua = leave_one_qc_out_rsd(matriz, NoneCorrector(), variant=VARIANTE, n_sampled_features=4)
    assert vetor.tobytes() == da_regua.tobytes()
    assert int(np.isfinite(vetor).sum()) == 4
    assert intervalo.reason is None
    qcs = [int(r) for r in held_out_rows(matriz)]
    colunas = np.isfinite(vetor)
    valores = np.asarray(matriz.values)
    sorteadas = [_mediana(valores[[i for i in qcs if i != j], :][:, colunas]) for j in qcs]
    todas = [_mediana(valores[[i for i in qcs if i != j], :]) for j in qcs]
    assert sorteadas != todas, "âncora: as duas populações dão os mesmos replicados neste dado"
    assert list(intervalo.replicates) == sorteadas


# ---------------------------------------------------------------------------
# O oráculo: `label` não muda, e `label_com_intervalos` traz um por prefixo
# ---------------------------------------------------------------------------


def _oraculo_leve(monkeypatch: pytest.MonkeyPatch) -> dict[str, _Contador]:
    """Os três prefixos do oráculo servidos por corretores leves e contados — sem SERRF.

    `none` é ele mesmo; `loess` e `serrf` viram o LOESS em série, que treina e custa pouco nesta
    matriz. O que se afirma é a fiação do oráculo, e não o SERRF.
    """
    contadores: dict[str, _Contador] = {}

    def constroi(nome: str, *, n_jobs: int = -1) -> Any:
        del n_jobs
        contador = _Contador(NoneCorrector() if nome == "none" else LoessCorrector(n_jobs=1))
        contadores[nome] = contador
        return contador

    monkeypatch.setattr(labeling, "build_corrector_for_labeling", constroi)
    return contadores


def test_label_com_intervalos_da_o_mesmo_rotulo_que_label_e_um_intervalo_por_prefixo(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """**A caixa não muda a eleição.** O rótulo com o intervalo é o de `label`, bit a bit nos
    escores, e a estimativa de cada intervalo é o escore daquela barra."""
    contadores = _oraculo_leve(monkeypatch)
    matriz = matriz_com_deriva()
    oraculo = LabelingOracle(n_jobs=1)

    sem = oraculo.label(matriz)
    contagem_sem = {nome: c.chamadas for nome, c in contadores.items()}
    com, intervalos = oraculo.label_com_intervalos(matriz)
    contagem_com = {nome: c.chamadas for nome, c in contadores.items()}

    assert com.prefix == sem.prefix and com.margin == sem.margin
    assert {k: np.float64(v).tobytes() for k, v in com.scores.items()} == {
        k: np.float64(v).tobytes() for k, v in sem.scores.items()
    }
    assert set(intervalos) == set(oraculo.prefixes)
    for prefixo, intervalo in intervalos.items():
        assert intervalo.estimate == com.scores[prefixo]
        assert intervalo.reason is None, intervalo.reason
    assert contagem_sem == {"none": 7, "loess": 7, "serrf": 7}
    assert contagem_com == {"none": 28, "loess": 28, "serrf": 28}


def test_label_com_intervalos_avisa_o_gancho_uma_vez_por_prefixo(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _oraculo_leve(monkeypatch)
    avisados: list[Any] = []
    LabelingOracle(n_jobs=1).label_com_intervalos(
        matriz_com_deriva(), depois_de_cada=avisados.append
    )
    assert avisados == [("none", 1), ("loess", 1), ("serrf", 1)]


def test_label_com_intervalos_passa_a_bandeira_ao_estimador(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _oraculo_leve(monkeypatch)
    with pytest.raises(ExecutionCancelledError, match=r"jackknife|leave-one-qc-out"):
        LabelingOracle(n_jobs=1).label_com_intervalos(matriz_com_deriva(), cancelled=lambda: True)
