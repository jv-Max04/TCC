"""A curva do LOESS fora do núcleo, e as curvas refeitas sem cada QC.

**Duas propriedades, e cada uma tem o OUTRO CAMINHO como oráculo.** As duas quantidades já
são contadas por um caminho do produto, então a asserção é a igualdade entre os dois, e não um
valor:

| Propriedade | O outro caminho |
|---|---|
| a curva devolvida é a que a correção usou | `fit_transform`, que divide por ela |
| a predição de cada QC sem ele implica o valor da régua | `out_of_sample_values` |

**A igualdade é bit a bit, e a população dela vai declarada:** a divisão do passo 5 só deixa ver a
curva onde a feature é finita e diferente de zero — nas outras posições o passo 11 devolve zero ou
ausente qualquer que seja a curva. É lá que os testes contam as posições comparadas.
"""

from collections.abc import Callable

import numpy as np
import pytest

from refino.correction.base import restore_zeros_and_missing
from refino.correction.loess import FittedCurve, LoessCorrector, curve_sensitivity
from refino.domain.matrix import RefinoMatrix
from refino.domain.metadata import SampleMetadata
from refino.metrics.crossval import out_of_sample_values
from tests.conftest import matrix_from_normalized

# As sete injeções de QC dos quatro arquivos deste estudo, em posição de linha (base zero). São
# OBSERVAÇÃO sobre aqueles arquivos e não invariante: aqui servem de forma plausível para
# a matriz sintética, e é por serem os extremos da corrida que as curvas sem o primeiro e sem o
# último QC têm trecho não ajustado.
QCS_DO_ESTUDO = (0, 11, 22, 33, 44, 55, 67)
N_AMOSTRAS = 68


def _divide_pela_curva(coluna: np.ndarray, curva: np.ndarray) -> np.ndarray:
    """Os passos 5 e 11 sobre uma curva dada: a conta que `fit_transform` faz depois da curva."""
    saida = LoessCorrector()._normalize(coluna, curva).values
    return restore_zeros_and_missing(coluna[:, None], saida[:, None]).values[:, 0]


def _sintetica(
    *,
    qcs: tuple[int, ...] = QCS_DO_ESTUDO,
    lotes: tuple[str, ...] | None = None,
    n_features: int = 3,
    seed: int = 7,
) -> RefinoMatrix:
    """Deriva suave conhecida, com ruído pequeno: o LOESS ajusta, e as curvas sem um QC diferem."""
    rng = np.random.default_rng(seed)
    tempo = np.arange(1, N_AMOSTRAS + 1, dtype=np.float64)
    base = rng.uniform(500.0, 5000.0, size=n_features)
    deriva = 1.0 + 0.004 * tempo[:, None] + 0.05 * np.sin(tempo[:, None] / 9.0)
    valores = base[None, :] * deriva * rng.normal(1.0, 0.02, size=(N_AMOSTRAS, n_features))
    amostras = tuple(
        SampleMetadata(
            name=f"S{i + 1}",
            injection_order=i + 1,
            sample_type="qc" if i in qcs else "sample",
            group="QC" if i in qcs else "Estudo",
            batch=lotes[i] if lotes else "A",
        )
        for i in range(N_AMOSTRAS)
    )
    return RefinoMatrix(
        values=valores,
        feature_names=tuple(f"F{j + 1}" for j in range(n_features)),
        samples=amostras,
        mode="tiny",
    )


def _sintetica_com_ramos(*, seed: int, n_features: int = 36) -> RefinoMatrix:
    """Seis formas de feature, em rodízio, escolhidas para a correção percorrer TODOS os ramos.

    **Para quê, e por que não substitui o recorte real (regra 1 do README).** Que a curva devolvida
    é a que a correção usou decorre do DESENHO — `fitted_curve` e `fit_transform` correm o mesmo
    `_trace_feature` — e não é contagem observada nos arquivos do laboratório; então ela se afirma
    também onde eles não estão, e o teste sobre o `pos_hilic` continua a ser o do dado real. O que
    se exige desta matriz é só que as duas coisas que a comparação separa DIFIRAM nela: que
    haja curva com sorteio do passo 4, trecho preenchido pelo passo 3, feature sem curva, zero e
    ausente — as âncoras dos testes contam cada um.

    | Forma | O ramo que ela alcança |
    |---|---|
    | deriva suave | a curva ajustada em toda a corrida |
    | QCs perto de zero e oscilantes | curva negativa entre eles: sorteio, ou mais de um quinto |
    | zeros espalhados | posições que o passo 11 apaga |
    | ausentes, às vezes num QC | QC fora do ajuste, e trecho preenchido pelo tempo mais próximo |
    | um QC aberrante | o descarte do passo 1 |
    | tendência forte em valores pequenos | curva negativa na ponta da corrida |
    """
    rng = np.random.default_rng(seed)
    tempo = np.arange(1, N_AMOSTRAS + 1, dtype=np.float64)
    qcs = list(QCS_DO_ESTUDO)
    colunas = []
    for j in range(n_features):
        base = rng.uniform(50.0, 5000.0)
        forma = j % 6
        if forma == 0:
            deriva = 1 + 0.004 * tempo + 0.05 * np.sin(tempo / 9)
            coluna = base * deriva * rng.normal(1.0, 0.03, N_AMOSTRAS)
        elif forma == 1:
            coluna = np.abs(rng.normal(0.0, 1.0, N_AMOSTRAS)) * rng.uniform(0.5, 3.0)
            coluna[qcs] = np.abs(rng.normal(0.0, 2.0, len(qcs))) * (rng.random(len(qcs)) < 0.6)
        elif forma == 2:
            coluna = base * rng.lognormal(0.0, 0.3, N_AMOSTRAS)
            coluna[rng.random(N_AMOSTRAS) < 0.25] = 0.0
        elif forma == 3:
            coluna = base * rng.lognormal(0.0, 0.2, N_AMOSTRAS)
            coluna[rng.random(N_AMOSTRAS) < 0.1] = np.nan
            if rng.random() < 0.5:
                coluna[qcs[int(rng.integers(len(qcs)))]] = np.nan
        elif forma == 4:
            coluna = base * rng.lognormal(0.0, 0.1, N_AMOSTRAS)
            coluna[qcs[int(rng.integers(len(qcs)))]] *= 20.0
        else:
            coluna = np.maximum(0.0, 30.0 - 0.6 * tempo + rng.normal(0.0, 3.0, N_AMOSTRAS))
        colunas.append(coluna)
    base_da_matriz = _sintetica(n_features=1)
    return RefinoMatrix(
        values=np.column_stack(colunas),
        feature_names=tuple(f"F{j + 1}" for j in range(n_features)),
        samples=base_da_matriz.samples,
        mode="tiny",
    )


def _recorte(n: int, semente: int) -> RefinoMatrix:
    """`n` features do `pos_hilic` real, sorteadas com semente fixa e na ordem da matriz."""
    cheia = matrix_from_normalized("pos_hilic", "none")
    escolhidas = np.sort(np.random.default_rng(semente).choice(cheia.n_features, n, replace=False))
    return RefinoMatrix(
        values=np.array(cheia.values[:, escolhidas], copy=True),
        feature_names=tuple(cheia.feature_names[j] for j in escolhidas),
        samples=cheia.samples,
        mode=cheia.mode,
    )


@pytest.fixture(scope="module")
def recorte_real() -> RefinoMatrix:
    """80 features do `pos_hilic`. **Medido antes de escrever as âncoras:** ver o teste da curva."""
    return _recorte(80, 0)


# ---------------------------------------------------------------------------
# A curva que corrigiu é a curva devolvida
# ---------------------------------------------------------------------------


def test_a_curva_devolvida_e_a_que_a_correcao_usou_bit_a_bit(recorte_real: RefinoMatrix) -> None:
    """Dividir pela curva devolvida reproduz a saída de `fit_transform`, coluna a coluna, em bytes.

    **Medido em 06/11/2026, sobre as 80 features, antes de escrever as âncoras:** 59 com curva e
    21 marcadas `negative_fitted_curve`; das 59, 28 têm posições sorteadas no passo 4 e 12 têm
    trechos preenchidos pelo passo 3, e as posições visíveis à comparação são 3.574. A curva que
    corrigiu inclui o sorteio, e é por isso que a âncora do sorteio vem junto: sem ela, um `drawn`
    sempre vazio passaria.
    """
    corretor = LoessCorrector()
    resultado = corretor.fit_transform(recorte_real, seed=42)
    corrigida = np.asarray(resultado.matrix.values, dtype=np.float64)

    comparadas = posicoes = com_sorteio = marcadas = 0
    for j, nome in enumerate(recorte_real.feature_names):
        curva = corretor.fitted_curve(recorte_real, nome, seed=42)
        coluna = np.asarray(recorte_real.values[:, j], dtype=np.float64)
        if curva.cause is not None:
            assert nome in resultado.uncorrected[curva.cause], (nome, curva.cause)
            assert np.isnan(curva.curve).all() and not curva.used_qc.any(), nome
            assert curva.reason, f"{nome}: sem curva e sem o porquê"
            marcadas += 1
            continue
        assert not any(nome in f for f in resultado.uncorrected.values()), nome
        assert _divide_pela_curva(coluna, curva.curve).tobytes() == corrigida[:, j].tobytes(), (
            f"{nome}: dividir pela curva devolvida não reproduz a correção — a curva da figura não "
            "seria a que corrigiu o sinal"
        )
        comparadas += 1
        posicoes += int(np.sum(np.isfinite(coluna) & (coluna != 0.0)))
        com_sorteio += int(curva.drawn.any())

    # As âncoras ANTES de concluir nada: a comparação acima não pode ter corrido sobre nada.
    assert comparadas >= 50 and posicoes >= 1_500, (comparadas, posicoes)
    assert marcadas >= 1, "nenhuma feature sem curva: o ramo da causa não foi exercitado"
    assert com_sorteio >= 1, "nenhuma curva com passo 4: `drawn` pode estar sempre vazio"


def test_a_comparacao_acusa_uma_curva_diferente_da_que_corrigiu(
    recorte_real: RefinoMatrix,
) -> None:
    """A planta: a curva mexida numa posição deixa de reproduzir a correção.

    A posição é uma em que a feature é finita e diferente de zero — fora dela o passo 11 apaga a
    curva, e a planta cairia sobre dado em que as duas coisas não diferem.
    """
    corretor = LoessCorrector()
    corrigida = np.asarray(corretor.fit_transform(recorte_real, seed=42).matrix.values)
    for j, nome in enumerate(recorte_real.feature_names):
        curva = corretor.fitted_curve(recorte_real, nome, seed=42)
        coluna = np.asarray(recorte_real.values[:, j], dtype=np.float64)
        visiveis = np.flatnonzero(np.isfinite(coluna) & (coluna != 0.0))
        if curva.cause is None and visiveis.size:
            break
    else:  # pragma: no cover - o recorte tem dezenas de features com curva
        pytest.fail("nenhuma feature com curva e posição visível no recorte")

    assert _divide_pela_curva(coluna, curva.curve).tobytes() == corrigida[:, j].tobytes()
    plantada = curva.curve.copy()
    plantada[visiveis[0]] *= 1.0 + 1e-9
    assert _divide_pela_curva(coluna, plantada).tobytes() != corrigida[:, j].tobytes(), (
        "a comparação não separa a curva que corrigiu de uma curva mexida: ela é decorativa"
    )


def _confronta_com_a_correcao(
    matriz: RefinoMatrix,
    seed: int,
    planta: Callable[[FittedCurve, np.ndarray], np.ndarray] | None = None,
) -> tuple[list[str], dict[str, int]]:
    """As features em que dividir pela curva devolvida NÃO reproduz `fit_transform`, e as contagens.

    `planta` troca a curva antes da divisão — é por onde o negativo entra, e o caminho da
    comparação é o mesmo nas duas direções.
    """
    corretor = LoessCorrector()
    resultado = corretor.fit_transform(matriz, seed=seed)
    corrigida = np.asarray(resultado.matrix.values, dtype=np.float64)
    divergencias: list[str] = []
    conta = {"comparadas": 0, "posicoes": 0, "com_sorteio": 0, "com_preenchida": 0, "marcadas": 0}
    for j, nome in enumerate(matriz.feature_names):
        curva = corretor.fitted_curve(matriz, nome, seed=seed)
        coluna = np.asarray(matriz.values[:, j], dtype=np.float64)
        if curva.cause is not None:
            if nome not in resultado.uncorrected.get(curva.cause, frozenset()):
                divergencias.append(f"{nome}: sem curva por {curva.cause} e a correção não marcou")
            conta["marcadas"] += 1
            continue
        divisor = curva.curve if planta is None else planta(curva, coluna)
        if _divide_pela_curva(coluna, divisor).tobytes() != corrigida[:, j].tobytes():
            divergencias.append(nome)
        conta["comparadas"] += 1
        conta["posicoes"] += int(np.sum(np.isfinite(coluna) & (coluna != 0.0)))
        conta["com_sorteio"] += int(curva.drawn.any())
        conta["com_preenchida"] += int(curva.filled.any())
    return divergencias, conta


def test_na_sintetica_a_curva_devolvida_e_a_que_a_correcao_usou_bit_a_bit() -> None:
    """O par do teste do `pos_hilic`, sobre a matriz que corre onde os arquivos não estão.

    **Medido em 06/11/2026 sobre `_sintetica_com_ramos(seed=1)`, antes de escrever as âncoras:** das
    36 features, 34 com curva e 2 marcadas `negative_fitted_curve`; das 34, 12 com sorteio do passo
    4 e 4 com trecho preenchido pelo passo 3, e 2.057 posições visíveis à comparação. As âncoras
    ficam abaixo das quatro contagens e exigem cada ramo — sem a do sorteio, um `drawn` sempre vazio
    e uma curva sem o passo 4 passariam juntos.
    """
    divergencias, conta = _confronta_com_a_correcao(_sintetica_com_ramos(seed=1), 42)
    assert conta["comparadas"] >= 30 and conta["posicoes"] >= 1_800, conta
    assert conta["marcadas"] >= 1, f"nenhuma feature sem curva: {conta}"
    assert conta["com_sorteio"] >= 6 and conta["com_preenchida"] >= 2, conta
    assert divergencias == [], divergencias[:5]


def test_na_sintetica_a_comparacao_acusa_a_curva_sem_o_sorteio_e_so_ela() -> None:
    """A planta cai onde as duas coisas diferem: as posições sorteadas no passo 4.

    A curva plantada troca o sinal das posições que o passo 4 sorteou — negativas, como o ajuste as
    tinha antes do sorteio. Ela só difere da que corrigiu nas features com sorteio numa posição que
    o passo 11 não apaga, então o confronto tem de acusar **exactamente** essas: a partição limpa é
    o que separa uma comparação que vê o sorteio de uma que acusa tudo.
    """
    matriz = _sintetica_com_ramos(seed=1)
    corretor = LoessCorrector()

    def visiveis(coluna: np.ndarray) -> np.ndarray:
        return np.isfinite(coluna) & (coluna != 0.0)

    def sem_o_sorteio(curva: FittedCurve, coluna: np.ndarray) -> np.ndarray:
        plantada = curva.curve.copy()
        plantada[curva.drawn] = -plantada[curva.drawn]
        return plantada

    esperadas = sorted(
        nome
        for j, nome in enumerate(matriz.feature_names)
        if (curva := corretor.fitted_curve(matriz, nome, seed=42)).cause is None
        and bool(np.any(curva.drawn & visiveis(np.asarray(matriz.values[:, j]))))
    )
    assert esperadas, "nenhuma posição sorteada visível: a planta cairia onde nada difere"
    assert _confronta_com_a_correcao(matriz, 42)[0] == [], "a árvore limpa já acusava"
    acusadas, _ = _confronta_com_a_correcao(matriz, 42, planta=sem_o_sorteio)
    assert sorted(acusadas) == esperadas, (acusadas, esperadas)


def test_na_sintetica_used_qc_marca_os_qcs_que_entraram_no_ajuste_e_so_eles() -> None:
    """`used_qc` é o QC que o passo 1 deixou no ajuste: finito e não aberrante no lote dele.

    **Duas âncoras por CONSTRUÇÃO, e elas não leem o código.** Na forma 4 de
    `_sintetica_com_ramos` um QC é multiplicado por 20, entre valores com 10% de dispersão
    log-normal; na forma 3 um QC pode ser ausente. Os dois ficam fora do ajuste em toda feature
    com curva, qualquer que seja o resto do descarte. **E a definição inteira** — finito e fora da
    caixa de Tukey com `coef = 1.5` sobre os QCs do lote — é a do `remove_outlier()` do `tools.R`,
    que `loess.py` transcreve; o teste a refaz por `boxplot_outlier_mask` para apanhar um
    `used_qc` que marque todo QC, ou nenhum.

    **Medido em 07/11/2026 sobre `_sintetica_com_ramos(seed=1)`, antes de escrever as âncoras:**
    das 34 features com curva, 14 têm algum QC fora do ajuste; das 6 da forma 4, as 6 têm curva e
    o QC multiplicado fora; 3 das 6 da forma 3 têm um QC ausente, e o ausente fica fora nas três.
    """
    from refino.correction.loess import OUTLIER_COEF
    from refino.metrics.rsd import boxplot_outlier_mask

    matriz = _sintetica_com_ramos(seed=1)
    corretor = LoessCorrector()
    qc = matriz.qc_mask
    linhas_qc = np.flatnonzero(qc)
    com_algum_fora = aberrantes = ausentes = 0
    for j, nome in enumerate(matriz.feature_names):
        curva = corretor.fitted_curve(matriz, nome, seed=42)
        coluna = np.asarray(matriz.values[:, j], dtype=np.float64)
        assert not curva.used_qc[~qc].any(), f"{nome}: amostra marcada como QC do ajuste"
        if curva.cause is not None:
            assert not curva.used_qc.any(), f"{nome}: sem curva e com QC no ajuste"
            continue
        valores_qc = coluna[linhas_qc]
        esperado = ~boxplot_outlier_mask(valores_qc, coef=OUTLIER_COEF) & np.isfinite(valores_qc)
        assert curva.used_qc[linhas_qc].tolist() == esperado.tolist(), nome
        com_algum_fora += int(not curva.used_qc[linhas_qc].all())
        if j % 6 == 4:
            multiplicado = linhas_qc[int(np.argmax(valores_qc / np.nanmedian(valores_qc)))]
            assert not curva.used_qc[multiplicado], f"{nome}: o QC multiplicado entrou no ajuste"
            aberrantes += 1
        if j % 6 == 3 and not np.isfinite(valores_qc).all():
            assert not curva.used_qc[linhas_qc[~np.isfinite(valores_qc)]].any(), nome
            ausentes += 1

    # As âncoras: sem elas, uma matriz em que nenhum QC sai do ajuste passaria.
    assert com_algum_fora >= 10, com_algum_fora
    assert aberrantes >= 5 and ausentes >= 2, (aberrantes, ausentes)


def test_na_sintetica_a_predicao_sem_cada_qc_implica_o_valor_que_a_regua_agrega() -> None:
    """O par do teste do `pos_hilic` para as curvas sem um QC, sobre a matriz sintética.

    **Medido em 06/11/2026 sobre `_sintetica_com_ramos(seed=1)`:** 252 pares de feature e QC
    retido, 224 com predição e 28 sem, todos `negative_fitted_curve`. As âncoras exigem os dois.
    """
    divergencias, com_predicao, sem_predicao = _confronta_com_a_regua(
        _sintetica_com_ramos(seed=1), 42
    )
    assert com_predicao >= 200, com_predicao
    assert sem_predicao >= 10, "o ramo sem predição não correu"
    assert divergencias == [], divergencias[:5]


def test_na_sintetica_o_confronto_acusa_a_curva_que_viu_o_qc(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A planta do vazamento sobre a matriz sintética: sem a remarcação, o confronto acusa."""
    import refino.correction.loess as modulo

    def sem_remarcar(matriz: RefinoMatrix) -> tuple[tuple[int, RefinoMatrix], ...]:
        return tuple((int(linha), matriz) for linha in np.flatnonzero(matriz.qc_mask))

    matriz = _sintetica_com_ramos(seed=1, n_features=12)
    assert _confronta_com_a_regua(matriz, 42)[0] == [], "a árvore limpa já acusava"
    monkeypatch.setattr(modulo, "held_out_matrices", sem_remarcar)
    divergencias, com_predicao, _ = _confronta_com_a_regua(matriz, 42)
    assert com_predicao >= 1, "a planta não deixou curva nenhuma: o confronto não teria o que ver"
    assert divergencias, "o confronto não separa a curva que viu o QC da que não o viu"


def test_a_curva_com_lotes_e_a_do_vetor_de_lote_e_a_das_amostras() -> None:
    """Os dois caminhos do lote dão a mesma curva: o vetor, que a correção recebe, e as amostras.

    A correção da janela passa os lotes por `batches`; a régua lê-os de `samples`. A figura lê-os
    das amostras, e só pode fazê-lo porque o corretor usa a partição e não o rótulo.
    """
    lotes = tuple("A" if i < 34 else "B" for i in range(N_AMOSTRAS))
    qcs = (0, 11, 22, 33, 34, 45, 56, 67)
    nas_amostras = _sintetica(qcs=qcs, lotes=lotes)
    sem_lote = _sintetica(qcs=qcs)
    vetor = np.array([1 if i < 34 else 2 for i in range(N_AMOSTRAS)])
    corretor = LoessCorrector()
    for nome in nas_amostras.feature_names:
        pelas_amostras = corretor.fitted_curve(nas_amostras, nome, seed=3)
        pelo_vetor = corretor.fitted_curve(sem_lote, nome, batches=vetor, seed=3)
        assert pelas_amostras.cause is None, pelas_amostras.reason
        assert pelas_amostras.curve.tobytes() == pelo_vetor.curve.tobytes(), nome
        # E o lote muda de facto a curva: sem esta âncora, os dois caminhos podiam ignorá-lo juntos.
        corrida_unica = corretor.fitted_curve(sem_lote, nome, seed=3)
        assert corrida_unica.curve.tobytes() != pelo_vetor.curve.tobytes(), nome


def test_a_feature_que_a_matriz_nao_tem_levanta() -> None:
    with pytest.raises(ValueError, match="não existe na matriz"):
        LoessCorrector().fitted_curve(_sintetica(), "F99", seed=1)


# ---------------------------------------------------------------------------
# As curvas sem cada QC são as da régua
# ---------------------------------------------------------------------------


def _confronta_com_a_regua(matriz: RefinoMatrix, seed: int) -> tuple[list[str], int, int]:
    """Os pares (feature, QC retido) em que a figura e a régua discordam, e as duas contagens.

    Devolve a lista em vez de afirmar, e é o que torna a planta honesta: o teste que planta
    o vazamento percorre exactamente este caminho.
    """
    fora = out_of_sample_values(matriz, LoessCorrector(), seed=seed)
    divergencias: list[str] = []
    com_predicao = sem_predicao = 0
    for j, nome in enumerate(matriz.feature_names):
        sensibilidade = curve_sensitivity(matriz, nome, seed=seed)
        coluna = np.asarray(matriz.values[:, j], dtype=np.float64)
        assert len(sensibilidade.held_out) == fora.shape[0], nome
        for i, (linha, curva) in enumerate(
            zip(sensibilidade.held_out_rows, sensibilidade.held_out, strict=True)
        ):
            if curva.cause is None:
                implicado = _divide_pela_curva(coluna, curva.curve)[linha]
                com_predicao += 1
            else:
                implicado = coluna[linha]
                sem_predicao += 1
            if np.float64(implicado).tobytes() != np.float64(fora[i, j]).tobytes():
                divergencias.append(f"{nome}@{linha}: figura {implicado!r}, régua {fora[i, j]!r}")
    return divergencias, com_predicao, sem_predicao


def test_a_predicao_sem_cada_qc_implica_o_valor_que_a_regua_agrega() -> None:
    """Para cada QC retido, a curva que não o viu dá o valor fora da amostra que a régua agrega.

    O oráculo é `out_of_sample_values`, o laço de `leave_one_qc_out_rsd`, sobre a MESMA matriz e a
    mesma semente. Quando o ajuste sem um QC falha, a feature volta sem correção, e o valor da régua
    é o medido — a figura diz «sem predição», e o teste confere que é esse o número agregado.

    **Medido em 06/11/2026, sobre 30 features do `pos_hilic` — 210 pares de feature e QC
    retido:** 156 com predição e 54 sem, todos `negative_fitted_curve`. As âncoras abaixo ficam
    abaixo das duas contagens, e exigem os dois ramos.
    """
    divergencias, com_predicao, sem_predicao = _confronta_com_a_regua(_recorte(30, 1), 42)
    assert com_predicao >= 100, com_predicao
    assert sem_predicao >= 1, "nenhum ajuste sem um QC falhou: o ramo sem predição não correu"
    assert divergencias == [], divergencias[:5]


def test_o_confronto_acusa_a_curva_que_viu_o_qc(monkeypatch: pytest.MonkeyPatch) -> None:
    """A planta do vazamento: sem a remarcação, a curva vê o QC e o confronto acusa.

    A planta troca os passos da régua por uma versão que devolve a matriz SEM remarcar o retido —
    o vazamento que o teste do espelho planta no estimador. As curvas passam a ser a que corrigiu,
    e o
    valor implicado vira o de dentro da amostra.
    """
    import refino.correction.loess as modulo

    def sem_remarcar(matriz: RefinoMatrix) -> tuple[tuple[int, RefinoMatrix], ...]:
        return tuple((int(linha), matriz) for linha in np.flatnonzero(matriz.qc_mask))

    matriz = _recorte(10, 1)
    assert _confronta_com_a_regua(matriz, 42)[0] == [], "a árvore limpa já acusava"
    monkeypatch.setattr(modulo, "held_out_matrices", sem_remarcar)
    divergencias, com_predicao, _ = _confronta_com_a_regua(matriz, 42)
    assert com_predicao >= 1, "a planta não deixou curva nenhuma: o confronto não teria o que ver"
    assert divergencias, "o confronto não separa a curva que viu o QC da que não o viu"


def test_lote_de_tres_qcs_corrige_e_cada_curva_sem_um_deles_falha_como_na_regua() -> None:
    """Com três QCs no lote a curva existe, e sem um deles sobram dois: o ajuste falha.

    É o mínimo de `loess.py` (`MIN_QC_FOR_FIT`) visto pelos dois lados. A guarda da régua deixa
    passar — dois de treino é o mínimo dela —, e o corretor recusa ajustar dois pontos, então a
    feature volta sem correção e o valor que a régua agrega é o medido.
    """
    lotes = tuple("A" if i < 40 else "B" for i in range(N_AMOSTRAS))
    qcs = (0, 13, 26, 39, 45, 56, 67)
    matriz = _sintetica(qcs=qcs, lotes=lotes, n_features=1)
    sensibilidade = curve_sensitivity(matriz, "F1", seed=5)

    assert sensibilidade.fitted.cause is None, sensibilidade.fitted.reason
    assert sensibilidade.degenerate is None
    fora = out_of_sample_values(matriz, LoessCorrector(), seed=5)
    coluna = matriz.values[:, 0]
    for i, (linha, curva) in enumerate(
        zip(sensibilidade.held_out_rows, sensibilidade.held_out, strict=True)
    ):
        do_lote_b = matriz.samples[linha].batch == "B"
        assert (curva.cause is not None) == do_lote_b, (linha, curva.reason)
        if do_lote_b:
            assert "exige 3" in curva.reason, curva.reason
            assert fora[i, 0] == coluna[linha], "a régua agrega o valor medido, sem correção"


def test_lote_com_menos_de_tres_qcs_nao_tem_curva_e_diz_por_que() -> None:
    """Dois QCs no lote: sem curva, com o motivo, e a régua recusa — o mesmo desfecho dos dois."""
    lotes = tuple("A" if i < 50 else "B" for i in range(N_AMOSTRAS))
    qcs = (0, 11, 22, 33, 44, 55, 67)
    matriz = _sintetica(qcs=qcs, lotes=lotes, n_features=1)
    sensibilidade = curve_sensitivity(matriz, "F1", seed=5)

    assert sensibilidade.fitted.cause == "fit_failed"
    assert sensibilidade.fitted.spans == (), "sem curva não há span eleito"
    assert "o lote B tem 2 QC(s)" in sensibilidade.fitted.reason
    assert "exige 3" in sensibilidade.fitted.reason
    assert np.isnan(sensibilidade.fitted.curve).all()
    assert sensibilidade.degenerate is not None and "leave_one_qc_out" in sensibilidade.degenerate
    assert sensibilidade.held_out == () and sensibilidade.n_curves == 0
    assert np.isnan(sensibilidade.lower).all() and np.isnan(sensibilidade.upper).all()


def test_o_envelope_e_o_minimo_e_o_maximo_das_curvas_e_marca_o_que_elas_nao_ajustaram() -> None:
    """Sete QCs nos extremos da corrida: as curvas sem o primeiro e sem o último não alcançam a
    borda, e o envelope marca esses trechos.

    **«Esses trechos» e não «exactamente esses»**, e a primeira versão deste teste afirmava o
    segundo e reprovou: no interior também há posição que uma curva sem um QC não ajusta. Com o
    vizinho de um QC retirado, o `span` estreito pode deixar menos de três QCs com peso numa
    posição, a predição sai indefinida e o passo 3 põe ali a do tempo mais próximo — é a mesma
    marca, pelo mesmo motivo. O que o teste afirma é a definição — a união do que cada curva não
    ajustou — e que as duas bordas estão nela.
    """
    matriz = _sintetica(n_features=1)
    sensibilidade = curve_sensitivity(matriz, "F1", seed=11)

    assert sensibilidade.fitted.cause is None, sensibilidade.fitted.reason
    assert len(sensibilidade.fitted.spans) == 1, "um lote, um span eleito"
    assert sensibilidade.n_curves == 7, [c.reason for c in sensibilidade.held_out]
    pilha = np.vstack([c.curve for c in sensibilidade.held_out])
    assert sensibilidade.lower.tobytes() == pilha.min(axis=0).tobytes()
    assert sensibilidade.upper.tobytes() == pilha.max(axis=0).tobytes()
    # Tem largura: sem um QC a curva muda. Sem esta âncora um envelope de largura zero passaria.
    assert float(np.max(sensibilidade.upper - sensibilidade.lower)) > 0.0

    uniao = np.any(np.vstack([c.filled | c.drawn for c in sensibilidade.held_out]), axis=0)
    assert sensibilidade.not_fitted.tolist() == uniao.tolist()
    ordens = matriz.injection_orders
    antes_do_segundo = ordens < ordens[QCS_DO_ESTUDO[1]]
    depois_do_penultimo = ordens > ordens[QCS_DO_ESTUDO[-2]]
    sem_o_primeiro, sem_o_ultimo = sensibilidade.held_out[0], sensibilidade.held_out[-1]
    assert sem_o_primeiro.filled[antes_do_segundo].all(), "sem o 1.º QC a curva chega à borda"
    assert sem_o_ultimo.filled[depois_do_penultimo].all(), "sem o último QC a curva chega à borda"
    assert sensibilidade.not_fitted[antes_do_segundo | depois_do_penultimo].all()
    # E a curva que corrigiu, com os sete, foi ajustada em toda a corrida.
    assert not sensibilidade.fitted.filled.any()


def test_matriz_sem_qc_nao_tem_curva_nem_curvas_sem_um_qc_e_nao_e_recusa_da_regua() -> None:
    """Sem QC nenhum: a correção não ajusta, não há QC a reter, e a régua não recusa — ela não mede.

    `degenerate` fica `None` porque a guarda dos três QCs por lote não conta lote sem QC; é o que
    separa «a régua recusou» de «não havia QC», que a figura diz de formas diferentes.
    """
    from refino.metrics.crossval import held_out_matrices

    matriz = _sintetica(qcs=(), n_features=1)
    assert not matriz.qc_mask.any(), "âncora: a matriz tem QC"
    assert held_out_matrices(matriz) == ()
    sensibilidade = curve_sensitivity(matriz, "F1", seed=5)
    assert sensibilidade.fitted.cause == "fit_failed", sensibilidade.fitted.reason
    assert sensibilidade.degenerate is None
    assert sensibilidade.held_out == () and sensibilidade.n_curves == 0
    assert np.isnan(sensibilidade.lower).all() and np.isnan(sensibilidade.upper).all()
    assert not sensibilidade.not_fitted.any()


def test_duas_chamadas_com_a_mesma_semente_dao_a_mesma_sensibilidade() -> None:
    """Regra 3 do README: a mesma entrada e a mesma semente dão a mesma curva e o mesmo envelope."""
    matriz = _sintetica(n_features=1)
    a = curve_sensitivity(matriz, "F1", seed=11)
    b = curve_sensitivity(matriz, "F1", seed=11)
    assert a.fitted.curve.tobytes() == b.fitted.curve.tobytes()
    assert a.lower.tobytes() == b.lower.tobytes() and a.upper.tobytes() == b.upper.tobytes()
