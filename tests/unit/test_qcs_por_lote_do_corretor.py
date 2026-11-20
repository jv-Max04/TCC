"""O corretor cujo lote menor não deixa QCs de treino fica SEM NÚMERO e COM MOTIVO.

Quando um lote tem menos QCs do que o corretor precisa para o ajuste fora da amostra, a barra desse
corretor diz o motivo em vez de um número, como o intervalo já faz.

**O defeito, medido aqui na âncora:** a régua retém um QC e ajusta com os outros, e aceita lotes de
três (dois no treino). O LOESS não ajusta com menos de três (`MIN_QC_FOR_FIT`): numa matriz à
mão com lotes 3 + 4, **3 dos 7 ajustes de base voltam `fit_failed` em todas as features**, e a barra
do LOESS misturava valores corrigidos e não corrigidos sem o dizer.

**O mínimo é o que o corretor DECLARA** — `min_train_qc_per_batch`, o atributo criado para o
intervalo, lido por `getattr` —, e quem não declara fica no piso. O `none` e o SERRF não
declaram, e por isso nunca são recusados por esta guarda; o `none` não ajusta nada.

**As matrizes são construídas à mão, e não são dado do laboratório (regra 1 do README).** O que se
afirma é o contrato — quem é recusado, com que motivo, antes de quantos ajustes, e o que o resto da
cadeia faz com a recusa —, e nada disso depende de os números serem os de um estudo.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from refino.correction.loess import MIN_QC_FOR_FIT, LoessCorrector
from refino.correction.none import NoneCorrector
from refino.decision import labeling
from refino.decision import palpite as modulo_do_palpite
from refino.decision.labeling import Label, LabelingOracle, build_corrector_for_labeling
from refino.decision.palpite import decide_o_palpite, mede_o_palpite
from refino.metrics.crossval import (
    MIN_TRAIN_QC_PER_BATCH,
    BatchQcShortfallError,
    DegenerateCrossValidationError,
    held_out_rows,
    jackknife_qc_minimum,
    leave_one_qc_out_rsd,
    leave_one_qc_out_rsd_with_jackknife,
    out_of_sample_qc_minimum,
    remark_as_sample,
)
from tests.unit.test_intervalo_das_barras import _Contador, matriz_com_deriva

VARIANTE = "outlier_removed"

# Os QCs de um lote de três e de outro de quatro — e o mesmo com quatro e quatro, o par causal.
LOTES_3_E_4 = ("A",) * 3 + ("B",) * 4
LOTES_4_E_3 = ("A",) * 4 + ("B",) * 3
LOTES_4_E_4 = ("A",) * 4 + ("B",) * 4


def _matriz(lotes: tuple[str, ...]) -> Any:
    """Cinco injeções por QC, nove features, a semente da âncora."""
    return matriz_com_deriva(
        n_qc=len(lotes), n_amostras=5 * len(lotes), n_features=9, semente=11, lotes_dos_qcs=lotes
    )


def _vetor_de_lotes(matriz: Any) -> np.ndarray:
    """O `batch` das amostras como vetor, para o oráculo — o mesmo que a régua lê."""
    return np.asarray([amostra.batch for amostra in matriz.samples])


# ---------------------------------------------------------------------------
# 1. O núcleo: o mínimo por corretor, e a recusa antes de qualquer ajuste
# ---------------------------------------------------------------------------


def test_o_minimo_da_regua_e_o_piso_ou_o_do_corretor_mais_um() -> None:
    """`out_of_sample_qc_minimum`: um retido e o treino. O piso (três) para quem não declara
    ou declara menos; o mínimo de treino declarado mais um para quem exige mais. E o do intervalo é
    sempre este mais um — o par tira um QC a mais."""

    class _Declara:
        name = "declara"

        def __init__(self, minimo: int) -> None:
            self.min_train_qc_per_batch = minimo

    assert out_of_sample_qc_minimum(NoneCorrector()) == MIN_TRAIN_QC_PER_BATCH + 1 == 3
    assert out_of_sample_qc_minimum(build_corrector_for_labeling("serrf", n_jobs=1)) == 3
    assert out_of_sample_qc_minimum(LoessCorrector()) == MIN_QC_FOR_FIT + 1 == 4
    assert out_of_sample_qc_minimum(_Declara(1)) == 3  # type: ignore[arg-type]
    assert out_of_sample_qc_minimum(_Declara(6)) == 7  # type: ignore[arg-type]
    for corretor in (NoneCorrector(), LoessCorrector(), _Declara(1), _Declara(6)):
        assert jackknife_qc_minimum(corretor) == out_of_sample_qc_minimum(corretor) + 1  # type: ignore[arg-type]


@pytest.mark.parametrize("lotes", [LOTES_3_E_4, LOTES_4_E_3], ids=["3+4", "4+3"])
def test_com_o_loess_um_lote_de_tres_qcs_recusa_antes_de_ajustar(lotes: tuple[str, ...]) -> None:
    """**A âncora vem primeiro:** o ajuste que retém um QC do lote de três falha DE FACTO em
    todas as features neste dado — sem ela, a guarda podia estar a recusar o que o corretor aguenta.
    Depois a régua: levanta a recusa tipada, com o lote, o mínimo e o corretor no motivo, e sem um
    ajuste sequer."""
    matriz = _matriz(lotes)
    curto = "A" if lotes == LOTES_3_E_4 else "B"
    retidos_do_curto = [int(r) for r in held_out_rows(matriz) if matriz.samples[r].batch == curto]
    assert len(retidos_do_curto) == 3
    for linha in retidos_do_curto:
        ajuste = LoessCorrector(n_jobs=1).fit_transform(remark_as_sample(matriz, linha), seed=42)
        assert ajuste.uncorrected.get("fit_failed") == frozenset(matriz.feature_names), (
            "âncora: o ajuste que treina com dois QCs no lote já não falha no LOESS, e a guarda "
            "recusaria à toa"
        )

    for estimador in (leave_one_qc_out_rsd, leave_one_qc_out_rsd_with_jackknife):
        contador = _Contador(LoessCorrector(n_jobs=1))
        with pytest.raises(BatchQcShortfallError) as recusa:
            estimador(matriz, contador, variant=VARIANTE)  # type: ignore[operator]
        assert contador.chamadas == 0, f"{estimador.__name__}: {contador.chamadas} ajustes à toa"
        assert recusa.value.corrector == "loess" and recusa.value.minimum == 4
        assert recusa.value.short == {curto: 3}
        motivo = recusa.value.reason
        assert f"{curto}=3 QCs" in motivo and "menos de 4" in motivo and "loess" in motivo, motivo


def test_com_lotes_de_quatro_o_loess_mede_e_o_none_mede_com_tres() -> None:
    """O par causal da recusa: o mesmo LOESS com 4 + 4 mede, sem `fit_failed` em ajuste nenhum — a
    recusa é causada pelo lote de três, e não pelo corretor. E o `none`, que não declara mínimo,
    mede com o lote de três: a guarda é do corretor."""
    de_quatro = _matriz(LOTES_4_E_4)
    vetor = leave_one_qc_out_rsd(de_quatro, LoessCorrector(n_jobs=1), variant=VARIANTE)
    assert np.isfinite(np.nanmedian(vetor))

    contador = _Contador(NoneCorrector())
    vetor = leave_one_qc_out_rsd(_matriz(LOTES_3_E_4), contador, variant=VARIANTE)  # type: ignore[arg-type]
    assert np.isfinite(np.nanmedian(vetor)) and contador.chamadas == 7


def test_abaixo_do_piso_a_recusa_e_a_de_d66_para_todos_e_nao_a_do_corretor() -> None:
    """Com um lote de dois, nem o `none` mede — a guarda vale para todos e levanta antes, com
    a classe-mãe e a frase dela. A recusa por corretor é subclasse (quem já apanha a recusa da
    régua continua a apanhá-la) e só dispara acima do piso."""
    matriz = _matriz(("A",) * 2 + ("B",) * 5)
    for corretor in (NoneCorrector(), LoessCorrector(n_jobs=1)):
        with pytest.raises(DegenerateCrossValidationError, match="3 QCs por lote") as recusa:
            leave_one_qc_out_rsd(matriz, corretor, variant=VARIANTE)
        assert type(recusa.value) is DegenerateCrossValidationError, type(recusa.value)
    assert issubclass(BatchQcShortfallError, DegenerateCrossValidationError)


# ---------------------------------------------------------------------------
# 2. O oráculo: sem número, com motivo, fora da eleição — e os outros medidos
# ---------------------------------------------------------------------------


@pytest.fixture
def oraculo_leve(monkeypatch: pytest.MonkeyPatch) -> dict[str, list[_Contador]]:
    """`none` ele mesmo; `loess` o LOESS em série; e `serrf` o `none` sob outro nome, contado.

    O SERRF de verdade custa minutos e nenhuma asserção daqui depende da floresta: o que importa é
    que ele NÃO declara mínimo de treino, como o SERRF real — e é isso que o `none` faz.
    """
    construidos: dict[str, list[_Contador]] = {"none": [], "loess": [], "serrf": []}

    def constroi(nome: str, *, n_jobs: int = -1) -> Any:
        del n_jobs
        interno = LoessCorrector(n_jobs=1) if nome == "loess" else NoneCorrector()
        contador = _Contador(interno)
        contador.name = nome
        construidos[nome].append(contador)
        return contador

    monkeypatch.setattr(labeling, "build_corrector_for_labeling", constroi)
    return construidos


def _chamadas(construidos: dict[str, list[_Contador]]) -> dict[str, int]:
    return {nome: sum(c.chamadas for c in lista) for nome, lista in construidos.items()}


def test_o_oraculo_recusa_o_loess_com_motivo_e_elege_entre_os_medidos(
    oraculo_leve: dict[str, list[_Contador]],
) -> None:
    """O LOESS sai sem número e com o motivo da régua; o `none` e o «SERRF» são medidos; a eleição
    e a margem são só entre os medidos. E o par causal: com 4 + 4 o LOESS volta a ter número."""
    matriz = _matriz(LOTES_3_E_4)
    prefixos = (("none", 2), ("loess", 2), ("serrf", 2))
    rotulo = LabelingOracle(prefixes=prefixos).label(matriz, batches=_vetor_de_lotes(matriz))

    assert set(rotulo.refusals) == {("loess", 2)}, rotulo.refusals
    assert "A=3 QCs" in rotulo.refusals[("loess", 2)], rotulo.refusals
    assert math.isnan(rotulo.scores[("loess", 2)])
    assert np.isfinite(rotulo.scores[("none", 2)]) and np.isfinite(rotulo.scores[("serrf", 2)])
    assert rotulo.correction != "loess" and rotulo.assessable
    assert _chamadas(oraculo_leve) == {"none": 7, "loess": 0, "serrf": 7}

    de_quatro = _matriz(LOTES_4_E_4)
    medido = LabelingOracle(prefixes=prefixos).label(de_quatro, batches=_vetor_de_lotes(de_quatro))
    assert medido.refusals == {} and np.isfinite(medido.scores[("loess", 2)])


def test_o_none_nunca_e_recusado_por_esta_guarda(oraculo_leve: dict[str, list[_Contador]]) -> None:
    """Com o lote de três, que é o mínimo do piso, o `none` mede — e a recusa do `none` só existe
    pela guarda, que levanta para a medição inteira, como antes."""
    matriz = _matriz(LOTES_4_E_3)
    rotulo = LabelingOracle(prefixes=(("none", 2), ("loess", 2))).label(
        matriz, batches=_vetor_de_lotes(matriz)
    )
    assert ("none", 2) not in rotulo.refusals and np.isfinite(rotulo.scores[("none", 2)])
    assert set(rotulo.refusals) == {("loess", 2)}


def test_recusado_e_degenerado_sao_distinguiveis_na_contagem(
    oraculo_leve: dict[str, list[_Contador]],
) -> None:
    """**Duas ausências distintas.** As duas ausências de número pedem a mesma ação — não entrar na
    eleição — e são fatos diferentes: o degenerado (QCs todos zero, RSD indefinido em toda feature)
    é `nan` SEM motivo; o recusado é `nan` COM motivo. Contadas, dão números diferentes."""
    matriz = _matriz(LOTES_3_E_4)
    valores = np.asarray(matriz.values, dtype=np.float64).copy()
    valores[matriz.qc_mask] = 0.0
    zerada = matriz.with_values(valores, "teste:qcs-zerados")
    rotulo = LabelingOracle(prefixes=(("none", 2), ("loess", 2), ("serrf", 2))).label(
        zerada, batches=_vetor_de_lotes(zerada)
    )
    sem_numero = {p for p, v in rotulo.scores.items() if not np.isfinite(v)}
    assert sem_numero == {("none", 2), ("loess", 2), ("serrf", 2)}, "âncora: nenhum deu número"
    recusados = set(rotulo.refusals)
    degenerados = sem_numero - recusados
    assert recusados == {("loess", 2)} and degenerados == {("none", 2), ("serrf", 2)}
    assert not rotulo.assessable


def test_com_o_intervalo_o_prefixo_recusado_traz_o_motivo_da_barra(
    oraculo_leve: dict[str, list[_Contador]],
) -> None:
    """Sem barra não há intervalo, e o motivo do intervalo é o da barra — sem número e sem um
    ajuste de par. O gancho do progresso é avisado também pelo prefixo recusado: a etapa é o
    corretor, e um corretor recusado também terminou."""
    matriz = _matriz(LOTES_3_E_4)
    avisados: list[Any] = []
    rotulo, intervalos = LabelingOracle(
        prefixes=(("none", 2), ("loess", 2), ("serrf", 2))
    ).label_com_intervalos(matriz, batches=_vetor_de_lotes(matriz), depois_de_cada=avisados.append)

    assert avisados == [("none", 2), ("loess", 2), ("serrf", 2)]
    do_loess = intervalos[("loess", 2)]
    assert do_loess.reason == rotulo.refusals[("loess", 2)]
    assert math.isnan(do_loess.estimate) and do_loess.standard_error is None
    assert do_loess.n_qc == 7
    assert _chamadas(oraculo_leve)["loess"] == 0
    # O `none` com o lote de três tem barra e não tem intervalo — a guarda do piso.
    assert intervalos[("none", 2)].reason is not None and "4" in str(intervalos[("none", 2)].reason)


# ---------------------------------------------------------------------------
# 3. O Palpite medido, e por ele a CLI `--auto medida` e a delegação
# ---------------------------------------------------------------------------

MOTIVO = (
    "menos de 4 QCs num lote (A=3 QCs): cada ajuste fora da amostra treina com um a menos, e o "
    "loess não ajusta com menos de 3"
)


def test_o_palpite_elege_entre_os_medidos_e_a_justificativa_diz_o_motivo() -> None:
    """O LOESS recusado não entra na eleição nem na margem; a justificativa — que vai ao registro da
    decisão, ao log da CLI e à delegação — diz que ele não foi medido e por quê."""
    palpite = decide_o_palpite(
        {"none": 0.30, "loess": math.nan, "serrf": 0.20},
        n_features_medidas=200,
        recusas={"loess": MOTIVO},
    )
    assert palpite.correction == "serrf" and palpite.segundo == "none"
    assert palpite.margem == pytest.approx(0.5)
    assert palpite.recusas == {"loess": MOTIVO}
    assert f"o loess não foi medido: {MOTIVO}" in palpite.rationale, palpite.rationale
    recomendacao = palpite.como_recomendacao()
    assert MOTIVO in recomendacao.rationale


def test_na_evidencia_o_recusado_fica_de_fora_e_o_degenerado_fica_nan() -> None:
    """**A mesma distinção na evidência**, que é `dict[str, float]` e vai ao log: sem número por
    recusa é AUSÊNCIA da chave — nunca houve medição —, e o RSD indefinido continua `nan`, que é o
    que a medição deu."""
    recusado = decide_o_palpite(
        {"none": 0.30, "loess": math.nan, "serrf": 0.20},
        n_features_medidas=200,
        recusas={"loess": MOTIVO},
    )
    degenerado = decide_o_palpite(
        {"none": 0.30, "loess": math.nan, "serrf": 0.20}, n_features_medidas=200
    )
    assert "rsd_loess" not in recusado.evidencia
    assert "rsd_loess" in degenerado.evidencia and math.isnan(degenerado.evidencia["rsd_loess"])
    assert recusado.evidencia["rsd_none"] == 0.30 == degenerado.evidencia["rsd_none"]
    assert "não foi medido" not in degenerado.rationale


def test_nada_avaliavel_com_um_recusado_nao_diz_tres_corretores() -> None:
    """Sem escore finito, a justificativa dizia «nos três corretores» — falso quando um não foi
    medido. Ela passa a nomear os medidos e o motivo do que não foi."""
    palpite = decide_o_palpite(
        {"none": math.nan, "loess": math.nan, "serrf": math.nan},
        n_features_medidas=200,
        recusas={"loess": MOTIVO},
    )
    assert not palpite.avaliavel
    assert "três" not in palpite.rationale, palpite.rationale
    assert "none, serrf" in palpite.rationale and MOTIVO in palpite.rationale


def test_um_numero_que_venha_com_a_recusa_nao_entra_na_justificativa() -> None:
    """`decide_o_palpite` aceita um escore finito para o recusado e o deixa fora da eleição — «o
    motivo vence». A justificativa listava o número dele entre os escores e logo depois dizia que
    ele não foi medido: o `loess` com o MENOR número e outro vencedor, na mesma frase. Sem a recusa,
    o mesmo número entra, elege, e é citado — e é isso que separa a planta da ausência dela."""
    escores = {"none": 0.30, "loess": 0.10, "serrf": 0.20}
    recusado = decide_o_palpite(escores, n_features_medidas=200, recusas={"loess": MOTIVO})
    medido = decide_o_palpite(escores, n_features_medidas=200)
    assert medido.correction == "loess" and "loess 0,1000" in medido.rationale, medido.rationale
    assert recusado.correction == "serrf"
    assert "0,1000" not in recusado.rationale, recusado.rationale
    assert "none 0,3000, serrf 0,2000" in recusado.rationale, recusado.rationale
    assert f"o loess não foi medido: {MOTIVO}" in recusado.rationale


def test_um_numero_que_venha_com_a_recusa_nao_entra_na_tabela_da_prosa() -> None:
    """A irmã da justificativa, na prosa da medição: a tabela listava todo escore finito, e
    com um número junto da recusa ela punha o `loess` à frente do vencedor e dizia, no fim, que ele
    não foi medido. O painel da correção já o tirava (`fila_do_fio`); a prosa diz o mesmo."""
    from refino.interface.fluxo import prosa_da_medicao

    campos: dict[str, Any] = {
        "vencedor": "serrf",
        "scores": {"none": 0.30, "loess": 0.10, "serrf": 0.20},
        "margem": 0.5,
        "avaliavel": True,
        "abstem": False,
        "n_features_medidas": 200,
    }
    sem = prosa_da_medicao(**campos)
    com = prosa_da_medicao(**campos, recusas={"loess": MOTIVO})
    assert "`loess` 0,1000" in sem, sem
    assert "0,1000" not in com, com
    assert "`serrf` 0,2000 · `none` 0,3000" in com, com
    assert f"O `loess` não foi medido: {MOTIVO}" in com, com


def test_mede_o_palpite_leva_a_recusa_do_oraculo_ao_palpite(monkeypatch: Any) -> None:
    """A recusa sai do oráculo e chega ao Palpite pelo nome do corretor — o prefixo `(c, n)` é
    detalhe do oráculo, e o Palpite fala em corretores."""

    def label(self: LabelingOracle, matriz: Any, **_: Any) -> Label:
        return Label(
            ("serrf", 2),
            {("none", 2): 0.30, ("loess", 2): math.nan, ("serrf", 2): 0.20},
            "leave_one_qc_out_rsd",
            "outlier_removed",
            refusals={("loess", 2): MOTIVO},
        )

    monkeypatch.setattr(modulo_do_palpite.LabelingOracle, "label", label)
    matriz = _matriz(LOTES_3_E_4)
    palpite = mede_o_palpite(matriz, batches=_vetor_de_lotes(matriz), n_jobs=1)
    assert palpite.recusas == {"loess": MOTIVO}
    assert palpite.correction == "serrf"


def test_a_cli_auto_medida_elege_sem_o_recusado_e_o_log_leva_o_motivo(
    oraculo_leve: dict[str, list[_Contador]],
) -> None:
    """`--auto medida` passa por `cli.recomenda`, que mede o Palpite; a justificativa e a evidência
    da recomendação são as que vão ao log da CLI e ao registro da decisão. O LOESS recusado não é
    ajustado, não é eleito, e o motivo está na justificativa e fora da evidência."""
    from refino.cli import AUTO_MEDIDA, recomenda

    matriz = _matriz(LOTES_3_E_4)
    recomendacao = recomenda(matriz, AUTO_MEDIDA, batches=_vetor_de_lotes(matriz), n_jobs=1)
    assert recomendacao.correction != "loess" and recomendacao.assessable
    assert "o loess não foi medido: menos de 4 QCs num lote (A=3 QCs)" in recomendacao.rationale
    assert "rsd_loess" not in recomendacao.evidence and "rsd_none" in recomendacao.evidence
    assert _chamadas(oraculo_leve)["loess"] == 0


# ---------------------------------------------------------------------------
# 4. A prosa da medição na janela
# ---------------------------------------------------------------------------


def test_a_prosa_da_medicao_diz_quem_nao_foi_medido_e_por_que() -> None:
    """A tabela de escores da janela lista só os medidos; sem esta frase, o corretor recusado
    desapareceria da prosa sem explicação. Sem `recusas`, a prosa é a de antes."""
    from refino.interface.fluxo import prosa_da_medicao

    campos: dict[str, Any] = {
        "vencedor": "serrf",
        "scores": {"none": 0.30, "loess": None, "serrf": 0.20},
        "margem": 0.5,
        "avaliavel": True,
        "abstem": False,
        "n_features_medidas": None,
    }
    sem = prosa_da_medicao(**campos)
    com = prosa_da_medicao(**campos, recusas={"loess": MOTIVO})
    assert "não foi medido" not in sem
    assert com.startswith(sem) and f"O `loess` não foi medido: {MOTIVO}" in com, com
    assert prosa_da_medicao(**campos, recusas=None) == sem
