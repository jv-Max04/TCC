"""O `LabelingOracle` — e o PAR CAUSAL, que é o que distingue oráculo de constante.

**Um oráculo que sempre devolve o mesmo rótulo passa em metade dos testes que alguém escreveria.**
Se a suíte só verificar "sobre uma matriz com deriva ele devolve `loess`", um oráculo que devolve
`loess` sempre fica verde. O par exige as duas direções sobre matrizes construídas para terem
resposta oposta — e ele é em processo, então não consegue ficar inerte.

**E o alvo desta fase já mostrou que a direção "não corrigir" é a majoritária.** Medido nos quatro
arquivos reais, o ótimo fora da amostra é `none`, `none`, `loess`, `none`: a constante que
responde sempre `none` acerta três dos quatro. É exatamente o caso em que o teste de uma
direção só seria verde e vazio.
"""

from dataclasses import replace

import numpy as np
import pytest

from refino.correction.loess import LoessCorrector
from refino.decision.labeling import Label, LabelingOracle, build_corrector_for_labeling
from refino.domain.matrix import RefinoMatrix
from refino.domain.metadata import SampleMetadata
from refino.metrics.crossval import DegenerateCrossValidationError, leave_one_qc_out_rsd

pytestmark = pytest.mark.property

# `serrf` fica de fora do padrao dos testes: sete reajustes de floresta custam minutos, e nenhuma
# asserticao deste arquivo depende dele. O oraculo com os tres prefixos e exercitado
# sobre o corpus, marcado `slow`.
PREFIXOS = (("none", 1), ("loess", 1))


def _matriz(
    *, deriva: float, n_amostras: int = 30, n_features: int = 12, semente: int = 7
) -> RefinoMatrix:
    """Matriz com QCs periódicos e deriva multiplicativa monótona de magnitude `deriva`.

    `deriva=0` dá QCs estáveis, onde corrigir só pode introduzir variação — o LOESS ajusta curva a
    ruído. `deriva` grande dá tendência que o LOESS remove. **São as duas direções do par.**
    """
    gerador = np.random.default_rng(semente)
    base = gerador.lognormal(10.0, 0.3, size=n_features)
    valores = np.tile(base, (n_amostras, 1)) * gerador.lognormal(
        0.0, 0.05, size=(n_amostras, n_features)
    )
    ordem = np.arange(n_amostras) / max(n_amostras - 1, 1)
    valores *= (1.0 + deriva * ordem)[:, None]

    eh_qc = np.arange(n_amostras) % 4 == 0
    amostras = tuple(
        SampleMetadata(
            name=f"{'QC' if eh_qc[i] else 'S'}{i:03d}",
            injection_order=i + 1,
            sample_type="qc" if eh_qc[i] else "sample",
            group="QC" if eh_qc[i] else ("A" if i % 2 else "B"),
            batch="A",
        )
        for i in range(n_amostras)
    )
    return RefinoMatrix(
        values=valores,
        feature_names=tuple(f"f{j:03d}" for j in range(n_features)),
        samples=amostras,
        mode="sintetico",
    )


# ---------------------------------------------------------------------------
# O par causal
# ---------------------------------------------------------------------------


def test_o_oraculo_elege_corretores_diferentes_em_matrizes_opostas() -> None:
    """**O par causal, e sem ele o oráculo é indistinguível de uma constante.**

    Duas matrizes construídas para terem resposta oposta, o mesmo oráculo, dois rótulos. Se as duas
    metades dessem o mesmo rótulo, o teste que exercita só uma delas ficaria verde para sempre.
    """
    oraculo = LabelingOracle(prefixes=PREFIXOS)
    estavel = oraculo.label(_matriz(deriva=0.0))
    com_deriva = oraculo.label(_matriz(deriva=6.0))

    assert estavel.correction == "none", (
        f"sem deriva, corrigir só pode piorar — o LOESS ajusta curva a ruído. Deu "
        f"{estavel.correction!r} com scores {estavel.scores}"
    )
    assert com_deriva.correction == "loess", (
        f"com deriva monótona forte, o LOESS a remove e o RSD fora da amostra cai. Deu "
        f"{com_deriva.correction!r} com scores {com_deriva.scores}"
    )
    assert estavel.correction != com_deriva.correction, (
        "as duas matrizes deram o MESMO rótulo. O oráculo pode estar devolvendo constante, e "
        "qualquer teste de uma direção só seria verde e vazio"
    )


def test_o_rotulo_carrega_o_score_de_todos_os_prefixos_e_nao_so_o_vencedor() -> None:
    """Sem os scores não há `margin`, e "venceu por 10x" fica igual a "venceu no 5o decimal".

    Valor de retorno que não distingue os dois casos apaga a informação que separa
    ações opostas — decidir com confiança e abster-se.
    """
    rotulo = LabelingOracle(prefixes=PREFIXOS).label(_matriz(deriva=0.0))
    assert set(rotulo.scores) == set(PREFIXOS)
    assert rotulo.scores[rotulo.prefix] == min(rotulo.scores.values())
    assert rotulo.estimator == "leave_one_qc_out_rsd"
    assert rotulo.variant == "outlier_removed"


def test_a_margem_e_relativa_e_e_nan_com_um_prefixo_so() -> None:
    """`nan` e não zero: "não há segundo" não é "a margem é zero"."""
    um_so = LabelingOracle(prefixes=(("none", 1),)).label(_matriz(deriva=0.0))
    assert np.isnan(um_so.margin), "com um prefixo não há margem a medir"

    dois = LabelingOracle(prefixes=PREFIXOS).label(_matriz(deriva=6.0))
    esperado = sorted(dois.scores.values())
    assert dois.margin == pytest.approx((esperado[1] - esperado[0]) / abs(esperado[0]))
    assert dois.margin > 0.0


def test_margem_grande_e_margem_pequena_sao_distinguiveis() -> None:
    """A propriedade que faz `margin` valer a pena, exercitada e não declarada.

    Se a margem não separasse o regime claro do regime ambíguo, a abstenção não teria como definir
    "vencedor claro".
    """
    oraculo = LabelingOracle(prefixes=PREFIXOS)
    claro = oraculo.label(_matriz(deriva=6.0)).margin
    ambiguo = oraculo.label(_matriz(deriva=0.05)).margin
    assert claro > ambiguo, (
        f"margem com deriva forte ({claro:.4f}) não é maior que com deriva quase nula "
        f"({ambiguo:.4f}) — a margem não está medindo o que o nome diz"
    )


# ---------------------------------------------------------------------------
# As guardas
# ---------------------------------------------------------------------------


def test_o_oraculo_e_deterministico() -> None:
    """Duas chamadas, o mesmo rótulo e os mesmos scores — bit a bit."""
    a = LabelingOracle(prefixes=PREFIXOS).label(_matriz(deriva=1.0))
    b = LabelingOracle(prefixes=PREFIXOS).label(_matriz(deriva=1.0))
    assert a.prefix == b.prefix
    assert a.scores == b.scores


def test_prefixo_vazio_e_prefixo_repetido_sao_recusados_na_construcao() -> None:
    with pytest.raises(ValueError, match="ao menos um prefixo"):
        LabelingOracle(prefixes=())
    with pytest.raises(ValueError, match="repetidos"):
        LabelingOracle(prefixes=(("none", 1), ("none", 1)))


def test_lote_multiplo_levanta_em_vez_de_devolver_numero_errado() -> None:
    """**Levanta e não cai num `else` razoável**, que é o modo de falha insidioso.

    Sem o vetor de lote em `batches`, aceitar `n_batches=2` e ignorá-lo devolveria o rótulo de lote
    único sob a etiqueta de dois lotes — número certo para a pergunta errada, sem erro. Com o vetor,
    a medição por lote existe — ver os testes abaixo.
    """
    with pytest.raises(NotImplementedError, match="lotes múltiplos"):
        LabelingOracle(prefixes=(("loess", 2),)).label(_matriz(deriva=1.0))


# ---------------------------------------------------------------------------
# O vetor de lote chega ao oráculo
# ---------------------------------------------------------------------------

# Dois lotes de 16 injeções, com os QCs em 1, 5, ..., 29: quatro em cada lote, o mínimo que a guarda
# da régua deixa medir com um retido e mais um de folga.
LOTES_4_E_4 = (np.arange(32) >= 16).astype(np.int64)


def _com_degrau(degrau: float) -> RefinoMatrix:
    """`_matriz` sem deriva e com um degrau multiplicativo na metade da corrida."""
    base = _matriz(deriva=0.0, n_amostras=32)
    valores = np.asarray(base.values, dtype=np.float64).copy()
    valores[16:] *= 1.0 + degrau
    return RefinoMatrix(
        values=valores, feature_names=base.feature_names, samples=base.samples, mode=base.mode
    )


def test_o_vetor_de_lote_CHEGA_ao_corretor() -> None:  # noqa: N802
    """O par causal: com o vetor, o LOESS mede outra coisa; o `none`, que não lê lote, não.

    E o valor do LOESS com o vetor é o mesmo que a régua dá sobre a matriz com os lotes gravados nas
    amostras — o outro caminho como oráculo: a igualdade é a asserção, e não um número.
    """
    matriz = _com_degrau(1.0)
    sem = LabelingOracle(prefixes=PREFIXOS).label(matriz)
    com = LabelingOracle(prefixes=(("none", 2), ("loess", 2))).label(matriz, batches=LOTES_4_E_4)

    assert np.isfinite(com.scores[("loess", 2)]), "âncora: a medição com lotes deu número"
    assert com.scores[("none", 2)] == sem.scores[("none", 1)], "o `none` não lê o lote"
    assert com.scores[("loess", 2)] != sem.scores[("loess", 1)], (
        "o LOESS corrige por lote: com o vetor, o número tem de mudar"
    )

    gravada = replace(
        matriz,
        samples=tuple(
            replace(amostra, batch=str(lote))
            for amostra, lote in zip(matriz.samples, LOTES_4_E_4.tolist(), strict=True)
        ),
    )
    direto = leave_one_qc_out_rsd(gravada, LoessCorrector(), variant="outlier_removed")
    assert com.scores[("loess", 2)] == float(np.nanmedian(direto))
    assert set(matriz.batches) == {"A"}, "a matriz de entrada não pode mudar (regra 7 do README)"


def test_lote_com_DOIS_qcs_levanta_e_nao_vira_numero() -> None:  # noqa: N802
    """A guarda da régua vale com lotes: reter um de dois QCs treinaria sobre um ponto só."""
    so_os_dois_ultimos = (np.arange(32) >= 24).astype(np.int64)  # QCs 25 e 29 no segundo lote
    with pytest.raises(DegenerateCrossValidationError, match="3 QCs por lote"):
        LabelingOracle(prefixes=(("loess", 2),)).label(_com_degrau(1.0), batches=so_os_dois_ultimos)


def test_vetor_de_lote_que_nao_casa_com_o_prefixo_levanta() -> None:
    """Prefixo de três lotes com vetor de dois é a pergunta errada, e não um número."""
    with pytest.raises(ValueError, match="declara 3 lote"):
        LabelingOracle(prefixes=(("loess", 3),)).label(_com_degrau(1.0), batches=LOTES_4_E_4)
    with pytest.raises(ValueError, match="alinhado a `samples`"):
        LabelingOracle(prefixes=(("loess", 2),)).label(_com_degrau(1.0), batches=LOTES_4_E_4[:10])


def test_matriz_sem_rsd_definido_e_marcada_nao_avaliavel_e_nao_levanta() -> None:
    """`assessable` separa "não medido" de "medido e empatou", na superfície do rótulo.

    Uma matriz de QCs todos zero não tem RSD em prefixo nenhum. Levantar perderia a matriz; devolver
    um rótulo sem marca faria a camada de decisão treinar sobre um alvo que não foi eleito.
    """
    matriz = _matriz(deriva=0.0)
    valores = np.asarray(matriz.values, dtype=np.float64).copy()
    valores[matriz.qc_mask] = 0.0
    zerada = RefinoMatrix(
        values=valores,
        feature_names=matriz.feature_names,
        samples=matriz.samples,
        mode=matriz.mode,
    )
    rotulo = LabelingOracle(prefixes=PREFIXOS).label(zerada)
    assert isinstance(rotulo, Label)
    assert not rotulo.assessable
    assert np.isnan(rotulo.margin)


def test_o_oraculo_nao_satisfaz_o_protocolo_das_politicas() -> None:
    """**A separação é de tipo e não de comentário**, e é o mecanismo que impede a comparação com o
    oráculo.

    O oráculo vê o desfecho da matriz que rotula. Se ele tivesse `recommend`, alguém o poria na
    tabela ao lado das políticas e a comparação voltaria — que é exactamente o
    critério que foi retirado.
    """
    assert not hasattr(LabelingOracle, "recommend")
    assert not hasattr(LabelingOracle, "name")


def test_o_corretor_da_rotulagem_usa_n_jobs_menos_um_e_o_do_executor_nao() -> None:
    """**O par que separa as duas decisões de paralelismo, e ele é medido.**

    O executor do multiverso dá `n_jobs=1` ao SERRF porque paraleliza as folhas; a rotulagem não tem
    folhas e o único paralelismo é o de dentro do corretor. Medido em 12 núcleos no `neg_hilic`:
    6,03 min por ajuste serial contra 1,10 min paralelo, **5,5x**.

    Sem este teste, alguém "unifica" as duas fábricas num refactor razoável e o corpus passa de
    17 h para 90 h sem nada acusar — o número muda, nenhum teste reprova.
    """
    from refino.multiverse.executor import build_corrector

    assert build_corrector_for_labeling("serrf").n_jobs == -1
    assert build_corrector("serrf").n_jobs == 1, (
        "o executor da Fase 8 precisa de `n_jobs=1` no corretor — ele paraleliza as folhas, e os "
        "dois níveis juntos produzem sobrescrição (AR34)"
    )
    for nome in ("none", "loess"):
        assert type(build_corrector_for_labeling(nome)) is type(build_corrector(nome))

    # **O LOESS entra no par pelo mesmo argumento:** a rotulagem dá-lhe os núcleos, e o
    # executor, que paraleliza as folhas, deixa-o em um.
    assert build_corrector_for_labeling("loess").n_jobs == -1
    assert build_corrector("loess").n_jobs == 1, (
        "o executor da Fase 8 paraleliza as folhas: o LOESS dele tem de ficar em 1 (AR34)"
    )

    with pytest.raises(ValueError, match="desconhecido"):
        build_corrector_for_labeling("inexistente")  # type: ignore[arg-type]


def test_a_semente_do_oraculo_chega_a_regua(monkeypatch: pytest.MonkeyPatch) -> None:
    """`LabelingOracle.seed` vai a `leave_one_qc_out_rsd`, com 42 por padrão.

    Espião em processo sobre a régua, nas duas direções: o padrão e uma semente trocada.
    """
    import refino.decision.labeling as labeling

    sementes: list[int] = []

    def regua(matrix: RefinoMatrix, corrector: object, **kwargs: object) -> np.ndarray:
        sementes.append(int(kwargs["seed"]))  # type: ignore[call-overload]
        return np.full(matrix.n_features, 0.1)

    monkeypatch.setattr(labeling, "leave_one_qc_out_rsd", regua)
    matriz = _matriz(deriva=0.0)
    LabelingOracle(prefixes=PREFIXOS).label(matriz)
    LabelingOracle(prefixes=PREFIXOS, seed=7).label(matriz)
    assert sementes == [42, 42, 7, 7]


# ---------------------------------------------------------------------------
# Vários núcleos: onde a rotulagem os põe
# ---------------------------------------------------------------------------


def test_a_rotulagem_da_os_mesmos_escores_com_um_e_com_dois_nucleos() -> None:
    """A medição fora da amostra das telas, com um e com dois núcleos: os escores bit a bit.

    `none` e `loess`, como o resto deste arquivo: o par do SERRF é o de `test_forest_speedup.py`.
    """
    matriz = _matriz(deriva=6.0)
    um = LabelingOracle(prefixes=PREFIXOS, n_jobs=1).label(matriz)
    dois = LabelingOracle(prefixes=PREFIXOS, n_jobs=2).label(matriz)

    escores_um = np.array([um.scores[p] for p in PREFIXOS])
    assert np.isfinite(escores_um).all(), f"âncora: escore não finito em {um.scores}"
    escores_dois = np.array([dois.scores[p] for p in PREFIXOS])
    assert escores_dois.tobytes() == escores_um.tobytes(), (
        f"os escores mudaram com o número de núcleos: {um.scores} contra {dois.scores}"
    )
    assert um.prefix == dois.prefix


def test_a_rotulagem_poe_os_nucleos_dentro_do_corretor_e_num_nivel_so(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Os núcleos vão para dentro do SERRF e do LOESS, e a régua fica em série — explícita.

    Espião sobre a régua: ele grava, por corretor, o `n_jobs` do corretor e o da régua. Com um
    núcleo, tudo em série; com dois, o SERRF e o LOESS em dois e a régua em um — nunca os dois
    níveis juntos, que abririam um pool por worker. O `none` não tem laço a dividir.
    """
    import refino.decision.labeling as labeling

    vistos: dict[str, tuple[int, int]] = {}

    def regua(matrix: RefinoMatrix, corrector: object, **kwargs: object) -> np.ndarray:
        nome = str(getattr(corrector, "name", "?"))
        vistos[nome] = (int(getattr(corrector, "n_jobs", 1)), int(kwargs["n_jobs"]))  # type: ignore[call-overload]
        return np.full(matrix.n_features, 0.1)

    monkeypatch.setattr(labeling, "leave_one_qc_out_rsd", regua)
    tres = (("none", 1), ("loess", 1), ("serrf", 1))
    matriz = _matriz(deriva=0.0)

    LabelingOracle(prefixes=tres, n_jobs=1).label(matriz)
    assert vistos == {"none": (1, 1), "loess": (1, 1), "serrf": (1, 1)}, vistos

    LabelingOracle(prefixes=tres, n_jobs=2).label(matriz)
    assert vistos == {"none": (1, 1), "loess": (2, 1), "serrf": (2, 1)}, vistos
