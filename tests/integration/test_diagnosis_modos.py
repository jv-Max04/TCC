"""Critérios 4.1 a 4.6 sobre os quatro arquivos reais, mais o negativo.

**O negativo é o teste que esta suíte tem de ter e quase não teria.** Numa
cascata, a regra que decide esconde as que decidiriam depois: afirmar que o `neg_c18`
sai `clean` confirma a cascata inteira e **não** distingue "a regra 1 acertou" de "a
regra 2 classificaria degrau e foi interceptada". A pergunta é

> se a regra que casou não casasse, o que este caso viraria?

e ela é respondida **pela política pública** - apertando `rsd_mediana_max` até a regra 1
não poder casar -, e não por remendo no interior. Remendar seria testar o remendo.
"""

import numpy as np
import pytest

from refino.diagnosis.batches import propose_batches
from refino.diagnosis.changepoint import detect_changepoints
from refino.diagnosis.policy import DiagnosisPolicy
from refino.diagnosis.profile import classify, diagnose, extract_features
from refino.io.readers import load_matrix
from refino.metrics.structure import structural_metrics
from tests.conftest import measurements_path, sample_list_path

pytestmark = pytest.mark.integration

MODOS = ("pos_hilic", "pos_c18", "neg_hilic", "neg_c18")

# Os valores iniciais da especificação. `max_spread` e `trajectory_movement_min` são os que
# a especificação deixa para quem tem os dados; aqui entram largos, porque o que estes
# testes verificam são as regras 1 a 3 e não a calibração da regra 0.
POLITICA = DiagnosisPolicy(
    max_spread=0.10,
    rsd_mediana_max=0.20,
    pc1_injection_correlation_max=0.30,
    step_sustain_min=2.0,
    qc_monotonicity_min=0.487,
    trajectory_movement_min=1e-9,
)

# Critérios 4.1 a 4.3. São OBSERVAÇÃO sobre os quatro arquivos do laboratório e não
# invariante do método: um quinto modo recebe as mesmas regras e tem sua própria
# classe medida.
CLASSE_ESPERADA = {
    "neg_hilic": ("step_change",),
    "neg_c18": ("clean", "gradual_drift"),
    "pos_hilic": ("erratic",),
    "pos_c18": ("erratic",),
}


@pytest.fixture(scope="module")
def matrizes() -> dict[str, object]:
    """As quatro matrizes reais, carregadas uma vez."""
    return {
        modo: load_matrix(measurements_path(modo), sample_list_path(modo), modo) for modo in MODOS
    }


@pytest.mark.parametrize("modo", MODOS)
def test_ca_4_1_a_4_3_a_classe_dos_quatro_modos(modo: str, matrizes: dict) -> None:
    """Critérios 4.1 a 4.3 - e a REGRA que casou vai na mensagem.

    Reportar a regra junto é o que torna a reprovação diagnóstica: "saiu `erratic`" não
    diz se as regras 2 e 3 falharam por pouco ou por muito, e "regra 4 com confiança
    0,03" diz.
    """
    resultado = classify(matrizes[modo], policy=POLITICA)
    assert resultado.defect_type in CLASSE_ESPERADA[modo], (
        f"{modo}: saiu {resultado.defect_type!r} pela regra {resultado.rule} com confiança "
        f"{resultado.confidence:.3f}; esperava um de {CLASSE_ESPERADA[modo]}"
    )


def test_a_regra_que_casa_em_cada_modo_e_a_esperada(matrizes: dict) -> None:
    """Não basta a classe: a regra que a produziu é parte do que se verifica.

    Se o `neg_hilic` saísse `step_change` pela regra 4 - impossível hoje, e é o ponto -,
    a classe estaria certa e a cascata errada.
    """
    esperado = {"neg_hilic": 2, "neg_c18": 1, "pos_hilic": 4, "pos_c18": 4}
    medido = {modo: classify(matrizes[modo], policy=POLITICA).rule for modo in MODOS}
    assert medido == esperado


def test_ar54_o_neg_c18_sem_a_regra_1_sai_gradual_drift(matrizes: dict) -> None:
    """**O negativo, e ele é a razão de esta suíte não ser decorativa.**

    O `neg_c18` casa a regra 1 e sai `clean`. Se não casasse, o que ele viraria? Antes
    da emenda, `step_change` - degrau num modo sem degrau -, porque a razão de
    saltos dele é 3,33 e o limiar da regra 2 era 3. O que o interceptava era a regra 1
    disparar antes, e ela só disparava porque `rsd_mediana` era lido em fração: consertar
    a unidade teria **quebrado** uma classificação certa.

    Com a regra 2 testando nível sustentado, os dois caminhos levam a resposta que o
    O critério aceita, e nenhum depende do outro. É isso que se afirma aqui.

    A regra 1 é desligada **pela política pública** - `rsd_mediana_max` abaixo de
    qualquer RSD real -, o que exercita o mesmo caminho que o portão percorre.
    """
    sem_regra_1 = DiagnosisPolicy(
        max_spread=POLITICA.max_spread,
        rsd_mediana_max=1e-9,
        pc1_injection_correlation_max=1e-9,
        step_sustain_min=POLITICA.step_sustain_min,
        qc_monotonicity_min=POLITICA.qc_monotonicity_min,
        trajectory_movement_min=POLITICA.trajectory_movement_min,
    )
    resultado = classify(matrizes["neg_c18"], policy=sem_regra_1)

    assert resultado.rule != 1, "a regra 1 ainda casou; o negativo não está exercitando nada"
    assert resultado.defect_type == "gradual_drift", (
        f"com a regra 1 desligada o `neg_c18` saiu {resultado.defect_type!r} pela regra "
        f"{resultado.rule}. Se saiu `step_change`, a regra 2 voltou a classificar degrau "
        "num modo sem degrau - que é exactamente o defeito de AR54."
    )
    assert resultado.rule == 3


@pytest.mark.parametrize("modo", MODOS)
def test_ca_4_6_o_vetor_nao_tem_nan_nos_quatro_modos(modo: str, matrizes: dict) -> None:
    """Todas as chaves da tabela, sem `NaN`, nos quatro modos reais."""
    vetor = extract_features(matrizes[modo])
    com_nan = sorted(chave for chave, valor in vetor.items() if not np.isfinite(valor))
    assert not com_nan, f"{modo}: chaves não finitas no vetor: {com_nan}"


def test_ca_4_6_o_conjunto_de_chaves_e_o_declarado(matrizes: dict) -> None:
    """O CONJUNTO, e não a contagem: contagem igual não é conjunto igual.

    Enumerar aqui é deliberado. Se uma chave for renomeada, este teste reprova com o
    nome antigo e o novo lado a lado - e é o multiverso que consome o vetor por nome.
    """
    esperado = {
        # Dimensão
        "n_samples",
        "n_features",
        "n_qcs",
        "razao_p_n",
        # Esparsidade - sem `fracao_ausentes`, porque zero É o ausente neste domínio
        "fracao_zeros",
        "frac_features_sem_zero",
        "frac_features_mais_80_zeros",
        # RSD, toda chave com a unidade no nome
        "rsd_mediana_fracao",
        "rsd_media_fracao",
        "rsd_p25_fracao",
        "rsd_p75_fracao",
        "rsd_p90_fracao",
        "frac_sob_20",
        "frac_sob_30",
        "assimetria_fisher",
        "n_features_rsd_definido",
        # Trajetória de QC
        "salto_mediano",
        "salto_maximo",
        "razao_maior_salto_sobre_mediana_dos_demais",
        "monotonicidade_liquido_bruto",
        "movimento_total",
        "rho_posicao_qc_pc1",
        "amplitude_total",
        "salto_sobre_amplitude",
        # Estrutura
        "pc1_injection_correlation",
        "qc_dispersion_ratio",
        "variancia_explicada_pc1",
        "variancia_explicada_pc2",
        # Intensidade
        "log_intensidade_mediana",
        "log_intensidade_iqr",
        "razao_p99_p50",
    }
    assert set(extract_features(matrizes["neg_c18"])) == esperado


def test_o_rsd_do_vetor_esta_em_fracao_e_nao_em_percentual(matrizes: dict) -> None:
    """A guarda do lado do PRODUTOR, e ela é independente da do consumidor.

    `DiagnosisPolicy` recusa limiar acima de 1; este teste afirma que o valor produzido
    é fração. As duas metades são necessárias: guarda só no limiar deixaria passar um
    vetor que devolvesse percentual, e aí a regra 1 nunca casaria.
    """
    for modo in MODOS:
        vetor = extract_features(matrizes[modo])
        for chave in (chave for chave in vetor if chave.endswith("_fracao")):
            assert 0.0 <= vetor[chave] < 100.0, (
                f"{modo}/{chave} = {vetor[chave]}: valor compatível com percentual, não fração"
            )
        assert vetor["rsd_mediana_fracao"] < 2.0, (
            f"{modo}: mediana de RSD {vetor['rsd_mediana_fracao']} parece percentual"
        )


def test_a_politica_de_indefinido_reporta_a_contagem(matrizes: dict) -> None:
    """A contagem de definidas viaja com a estatística.

    Nos dados reais há features de RSD indefinido em todos os modos, então a contagem tem
    de ser estritamente menor que `n_features`. Se fossem iguais, ou a política mudou ou
    a variante mudou, e as duas coisas mudam os números.
    """
    for modo in MODOS:
        vetor = extract_features(matrizes[modo])
        assert 0 < vetor["n_features_rsd_definido"] < vetor["n_features"]


def test_ca_4_4_o_corte_do_neg_hilic_cai_na_janela_45_56(matrizes: dict) -> None:
    """Sobre o arquivo real - OBSERVAÇÃO sobre um arquivo."""
    matriz = matrizes["neg_hilic"]
    metricas = structural_metrics(matriz, pretransform="log10p1")
    trajetoria = np.asarray(metricas.qc_pc1_trajectory, dtype=np.float64)
    cortes = detect_changepoints(trajetoria, matriz.injection_orders[matriz.qc_mask])

    assert cortes, "o degrau do `neg_hilic` tem de produzir ao menos um corte"
    assert any(45 <= corte <= 56 for corte in cortes), (
        f"nenhum corte na janela 45-56; achei {cortes}"
    )


def test_ca_4_5_o_neg_hilic_da_exatamente_dois_lotes(matrizes: dict) -> None:
    """Dois lotes, com as DUAS âncoras.

    O critério pede "fronteira no ponto detectado", o que afirma acordo entre duas
    funções nossas e passaria junto se as duas errassem. O teste ancora **também** na
    janela medida da especificação - entre as injeções 45 e 56 -, que é evidência sobre o dado e
    não sobre a nossa concordância interna.
    """
    matriz = matrizes["neg_hilic"]
    metricas = structural_metrics(matriz, pretransform="log10p1")
    trajetoria = np.asarray(metricas.qc_pc1_trajectory, dtype=np.float64)
    cortes = detect_changepoints(trajetoria, matriz.injection_orders[matriz.qc_mask])
    lotes = propose_batches(matriz, cortes)

    distintos = sorted(set(lotes.tolist()))
    assert len(distintos) == 2, f"esperava dois lotes, achei {distintos}"
    assert len(cortes) == 1
    assert 45 <= cortes[0] <= 56, "a fronteira tem de cair na janela que a §4.2 mede"
    assert lotes.size == matriz.n_samples


def test_os_lotes_alinham_com_cada_amostra_pela_ordem_dela(matrizes: dict) -> None:
    """O rótulo da posição `i` corresponde a `matrix.samples[i]`, pela ordem DELA.

    **E este teste tem menos dente do que parece, o que está declarado de propósito.**
    A primeira versão trazia uma asserção antivacuidade afirmando que as colunas não
    estão em ordem de injeção - o que é verdade do **arquivo** e falso da **matriz
    carregada**: a especificação declara `samples` ordenado por `injection_order`, e a ingestão
    ordena. Medido: `injection_orders` do `neg_hilic` é `1..68`.

    Então "alinhado com `samples`" e "alinhado com a ordem de injeção ordenada"
    **coincidem** para qualquer `RefinoMatrix` válida, e nenhum teste sobre dado real
    pode separá-los. O risco é fechado a montante, na ingestão, e quem
    o verifica é a invariante posicional de lá.

    O que resta aqui é verificável e vale: o rótulo depende da ordem de injeção **da
    amostra** e não da posição dela. A versão com ordens não contíguas, que separa as
    duas, está em `tests/unit/test_propose_batches.py`.
    """
    matriz = matrizes["neg_hilic"]
    metricas = structural_metrics(matriz, pretransform="log10p1")
    trajetoria = np.asarray(metricas.qc_pc1_trajectory, dtype=np.float64)
    cortes = detect_changepoints(trajetoria, matriz.injection_orders[matriz.qc_mask])
    lotes = propose_batches(matriz, cortes)

    fronteira = cortes[0]
    for posicao, amostra in enumerate(matriz.samples):
        esperado = 0 if amostra.injection_order < fronteira else 1
        assert lotes[posicao] == esperado, (
            f"amostra {amostra.name!r} na posição {posicao} tem ordem "
            f"{amostra.injection_order} e recebeu lote {lotes[posicao]}"
        )


def test_diagnose_devolve_perfil_completo_e_nao_altera_a_matriz(matrizes: dict) -> None:
    """Regra 7 do README - imutabilidade - mais o contrato do `DefectProfile`."""
    matriz = matrizes["neg_hilic"]
    antes = np.array(matriz.values, copy=True)

    perfil = diagnose(matriz, policy=POLITICA)

    assert np.array_equal(np.asarray(matriz.values), antes), "`diagnose` alterou a matriz"
    assert perfil.defect_type == "step_change"
    assert perfil.n_qcs == 7
    assert perfil.n_samples == matriz.n_samples
    assert perfil.n_features == matriz.n_features
    assert len(perfil.changepoints) == 1
    assert len(perfil.suggested_batches) == matriz.n_samples
    assert perfil.raw_features["rsd_mediana_fracao"] == perfil.qc_rsd_median
    # `missing_fraction` conta célula NÃO FINITA, que é quantidade diferente de zero.
    assert perfil.missing_fraction == 0.0
    assert perfil.zero_fraction > 0.0


def test_diagnose_e_determinista(matrizes: dict) -> None:
    """Regra 3 do README - duas execuções com a mesma entrada dão o mesmo perfil."""
    primeiro = diagnose(matrizes["pos_c18"], policy=POLITICA)
    segundo = diagnose(matrizes["pos_c18"], policy=POLITICA)
    assert primeiro == segundo


@pytest.mark.slow
@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason=(
        "[CA-4.7] exige acurácia acima de 85% em 200 matrizes sintéticas com defeito "
        "conhecido, e é a MESMA quantidade que o [CA-7.4] mede: este teste corre a medição "
        "dele, e reprova enquanto ele reprovar. `strict=True` é deliberado: `xfail` não "
        "estrito passa em silêncio quando começa a passar, e `XPASS` aqui é o sinal de que "
        "a acurácia chegou - o desfecho que manda remover este marcador em vez de afrouxá-lo. "
        "`raises=AssertionError` porque só a asserção da acurácia é a falha esperada: "
        "qualquer outra exceção, como um ImportError, reprova em vez de ser absorvida pelo "
        "marcador sem que o critério meça nada."
    ),
)
def test_ca_4_7_acuracia_sobre_matrizes_sinteticas(matrizes: dict) -> None:
    """Inerte por dependência declarada, e o `xfail` é o mecanismo.

    **Não é `skip`, e a diferença importa.** `skipped` é o desfecho que a política de
    `skip` deste projeto ensina a ler como legítimo, então um `skip` aqui **nunca
    reprovaria** - é a família, dezoito testes de golden pulando por citar
    gerador inexistente. `xfail(strict=True)` reprova quando passa, o que é a única
    direção em que este teste tem informação hoje.

    O critério ficou onde a especificação o pôs. A auditoria registrou que
    ele é defeito de alocação - critério verificável só com maquinário de fase
    posterior, cujo remédio estabelecido é realocar com decisão registrada - e a realocação **não**
    foi aprovada. Está declarado como pendência aberta no plano da fase.
    """
    from tests.integration.test_gerador_sintetico import acuracia_do_diagnose_em_200

    acertos, confusao = acuracia_do_diagnose_em_200(matrizes)
    assert acertos / 200 > 0.85, (
        f"[CA-4.7] acurácia {acertos}/200 = {acertos / 200:.1%}, abaixo de 85%; confusão: "
        f"{dict(confusao)}. É a medição do [CA-7.4], e a causa medida está na razão dele"
    )
