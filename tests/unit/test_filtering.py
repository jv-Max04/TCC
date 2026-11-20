"""`filter_by_qc_rsd` e `filter_near_zero_variance` — as duas primeiras etapas.

**O que mais importa aqui é a PARTIÇÃO EXAUSTIVA das três contagens.** Acima do corte,
indefinido e retidas têm de somar `n_features`, e os dois motivos de descarte não podem
colapsar num só: instabilidade **medida** e **ausência** de medida levam a ações opostas — a
primeira sugere corrigir melhor, a segunda sugere olhar o dado.

E o `filter_near_zero_variance` é porte de semântica de terceiro — `caret::nearZeroVar` —,
então **cada metade da condição conjunta é testada separadamente**. Testar só o efeito
combinado deixaria passar uma implementação que ignorasse uma das duas.
"""

import numpy as np
import pytest

from refino.domain.labels import InvalidLabelError
from refino.domain.matrix import RefinoMatrix
from refino.domain.metadata import SampleMetadata
from refino.preprocessing.filtering import (
    ACIMA_DO_CORTE,
    INDEFINIDO,
    filter_by_qc_rsd,
    filter_by_ranking,
    filter_near_zero_variance,
    fracao_retida_por_tamanho,
)

N_QC = 4


def _matriz(colunas: dict[str, list[float]], n_qc: int = N_QC) -> RefinoMatrix:
    """Matriz com as colunas dadas; as primeiras `n_qc` linhas são QC."""
    nomes = tuple(colunas)
    valores = np.array([colunas[nome] for nome in nomes], dtype=np.float64).T
    n = valores.shape[0]
    amostras = tuple(
        SampleMetadata(
            name=f"S{i}",
            injection_order=i + 1,
            sample_type="qc" if i < n_qc else "sample",
            group="QC" if i < n_qc else "G",
        )
        for i in range(n)
    )
    return RefinoMatrix(values=valores, feature_names=nomes, samples=amostras, mode="sintetico")


# QCs nas quatro primeiras linhas; amostras depois. `estavel` tem RSD baixo, `instavel`
# alto, e `sem_medida` tem QCs todos zero — RSD indefinido por divisão por zero.
FIXTURE = {
    "estavel": [100.0, 101.0, 99.0, 100.0, 50.0, 60.0],
    "instavel": [10.0, 200.0, 5.0, 300.0, 50.0, 60.0],
    "sem_medida": [0.0, 0.0, 0.0, 0.0, 50.0, 60.0],
}


# ---------------------------------------------------------------------------
# A partição exaustiva, que é a propriedade vinculante
# ---------------------------------------------------------------------------


def test_a_particao_das_tres_contagens_e_exaustiva() -> None:
    """Retidas + acima do corte + indefinido = `n_features`, sem sobra."""
    matriz = _matriz(FIXTURE)
    _, manter, resultado = filter_by_qc_rsd(matriz, 0.30, variant="plain")

    assert resultado.n_features_in == 3
    assert int(manter.sum()) == resultado.n_features_out
    assert (
        resultado.n_features_out
        + resultado.discarded_by_reason[ACIMA_DO_CORTE]
        + resultado.discarded_by_reason[INDEFINIDO]
        == 3
    )


def test_indefinido_reprova_e_e_contado_no_motivo_proprio() -> None:
    """**Indefinido reprova, e o motivo separado é a metade que se perde.**

    Feature com QCs todos zero tem RSD indefinido e **reprova** — feature sobre a qual nada
    se sabe não passa filtro de qualidade. E ela entra em `rsd_indefinido`, **não** em
    `rsd_acima_do_corte`: colapsar os dois tornaria indistinguíveis em contagem duas coisas
    que levam a ações opostas.
    """
    matriz = _matriz(FIXTURE)
    _, manter, resultado = filter_by_qc_rsd(matriz, 0.30, variant="plain")

    indice = matriz.feature_names.index("sem_medida")
    assert not manter[indice], "feature de RSD indefinido tem de REPROVAR"
    assert resultado.discarded_by_reason[INDEFINIDO] == 1
    assert resultado.discarded_by_reason[ACIMA_DO_CORTE] == 1, (
        "a `instavel` tem de estar no motivo dela, não somada com a indefinida"
    )


def test_a_feature_estavel_fica_e_a_instavel_sai() -> None:
    """O comportamento nominal, com RSD conhecido por construção."""
    matriz = _matriz(FIXTURE)
    filtrada, manter, _ = filter_by_qc_rsd(matriz, 0.30, variant="plain")

    assert manter[matriz.feature_names.index("estavel")]
    assert not manter[matriz.feature_names.index("instavel")]
    assert filtrada.feature_names == ("estavel",)


def test_nenhuma_feature_reprovada_da_mascara_toda_verdadeira() -> None:
    """O caso que prova que o filtro não descarta por acidente.

    Sem ele, um filtro que descartasse sempre passaria em todos os testes acima — eles só
    afirmam que as reprovadas saem.
    """
    matriz = _matriz({"a": [10.0, 10.0, 10.0, 10.0, 5.0, 6.0]})
    _, manter, resultado = filter_by_qc_rsd(matriz, 0.30, variant="plain")

    assert manter.all()
    assert resultado.n_discarded == 0
    assert sum(resultado.discarded_by_reason.values()) == 0


# ---------------------------------------------------------------------------
# Os rótulos
# ---------------------------------------------------------------------------


def test_a_variante_nao_tem_padrao() -> None:
    """`variant` é somente-nomeado e sem padrão — escolher em silêncio é o defeito."""
    matriz = _matriz(FIXTURE)
    with pytest.raises(TypeError, match="variant"):
        filter_by_qc_rsd(matriz, 0.30)  # type: ignore[call-arg]


def test_a_variante_muda_o_resultado() -> None:
    """**O par causal do rótulo: se as duas variantes dessem o mesmo, o rótulo seria decoração.**

    A `outlier_removed` remove outliers dos QCs antes de calcular, então uma feature cujo
    RSD alto vem de **um** QC aberrante passa a ser retida. O fixture é construído para
    isso, e a construção dele foi corrigida uma vez — ver o comentário abaixo.
    """
    # SETE QCs, seis iguais e um distante. Com seis valores idênticos a dobradiça
    # inferior e a superior coincidem, o IQR é ZERO, e a cerca de Tukey fica no próprio
    # valor — então o sétimo é outlier. Com quatro QCs a cerca é larga o bastante para
    # absorvê-lo, e foi assim que a primeira versão deste fixture não separou nada.
    coluna = [100.0] * 6 + [1000.0] + [50.0, 60.0]
    matriz = _matriz({"um_outlier": coluna}, n_qc=7)
    _, manter_plain, _ = filter_by_qc_rsd(matriz, 0.30, variant="plain")
    _, manter_sem_outlier, _ = filter_by_qc_rsd(matriz, 0.30, variant="outlier_removed")

    assert not manter_plain[0], "com `plain` o outlier infla o RSD e a feature reprova"
    assert manter_sem_outlier[0], "com `outlier_removed` ela é retida — o rótulo é consultado"


def test_variante_invalida_levanta() -> None:
    """`Literal` é análise estática; a guarda em runtime é a fronteira."""
    matriz = _matriz(FIXTURE)
    with pytest.raises(InvalidLabelError, match="RsdVariant"):
        filter_by_qc_rsd(matriz, 0.30, variant="plane")  # type: ignore[arg-type]


def test_estimador_nao_implementado_levanta_em_vez_de_medir_in_sample() -> None:
    """Rótulo aceito e ignorado daria número diferente do pedido, com o nome errado no log."""
    matriz = _matriz(FIXTURE)
    for estimador in ("leave_one_qc_out", "r_cv"):
        with pytest.raises(NotImplementedError, match=estimador):
            filter_by_qc_rsd(matriz, 0.30, variant="plain", estimator=estimador)  # type: ignore[arg-type]


@pytest.mark.parametrize("corte", [1.5, 30.0, 0.0, -0.1])
def test_corte_fora_de_zero_um_levanta_nomeando_a_confusao(corte: float) -> None:
    """A guarda de unidade nesta etapa, e `30.0` é o corte de 30% em percentual.

    Sem ela, `cutoff=30` retém tudo — todo RSD em fração é menor que 30 — e o filtro vira
    identidade em silêncio, que é o modo de falha que não levanta.
    """
    matriz = _matriz(FIXTURE)
    with pytest.raises(ValueError, match="FRAÇÃO"):
        filter_by_qc_rsd(matriz, corte, variant="plain")


def test_os_rotulos_viajam_no_stage_result() -> None:
    """Quantidade sem rótulo não é comparável entre execuções."""
    matriz = _matriz(FIXTURE)
    _, _, resultado = filter_by_qc_rsd(matriz, 0.30, variant="outlier_removed")
    assert resultado.labels["rsd_variant"] == "outlier_removed"
    assert resultado.labels["rsd_estimator"] == "in_sample"
    assert resultado.labels["cutoff"] == "0.3"


def test_a_matriz_de_entrada_nao_e_alterada() -> None:
    """Regra 7 do README — imutabilidade, e ela alcança o array."""
    matriz = _matriz(FIXTURE)
    antes = np.array(matriz.values, copy=True)
    filter_by_qc_rsd(matriz, 0.30, variant="plain")
    assert np.array_equal(np.asarray(matriz.values), antes)


# ---------------------------------------------------------------------------
# `nearZeroVar`: cada metade da condição conjunta, separada
# ---------------------------------------------------------------------------


def test_coluna_constante_sai() -> None:
    """Degenerado por construção: não há segundo valor mais comum."""
    matriz = _matriz({"const": [7.0] * 6, "ok": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]})
    _, manter, resultado = filter_near_zero_variance(matriz)
    assert not manter[0]
    assert manter[1]
    assert resultado.n_features_out == 1


def test_sai_quando_as_duas_condicoes_valem() -> None:
    """Razão de frequência alta **e** poucos valores distintos: o caso do `caret`.

    Vinte amostras, dezenove iguais e uma diferente: razão 19/1 = 19, que não excede
    `freq_cut` de 19 — então é preciso um pouco mais desequilíbrio. Com 21 amostras e vinte
    iguais a razão é 20 e os distintos são 2/21 = 9,5%, abaixo de 10.
    """
    coluna = [1.0] * 20 + [2.0]
    matriz = _matriz({"quase_const": coluna}, n_qc=4)
    _, manter, _ = filter_near_zero_variance(matriz)
    assert not manter[0], "razão 20 > 19 e 9,5% < 10: as duas condições valem, tem de sair"


def test_fica_quando_so_a_razao_de_frequencia_vale() -> None:
    """**A primeira metade isolada.** Um valor dominante, mas muitos distintos.

    Sem esta asserção, uma implementação que ignorasse `unique_cut` passaria — ela
    descartaria esta coluna e nenhum outro teste a pegaria.
    """
    coluna = [1.0] * 20 + [float(i) for i in range(2, 12)]
    matriz = _matriz({"dominante_mas_variada": coluna}, n_qc=4)
    _, manter, _ = filter_near_zero_variance(matriz)
    distintos_pct = 100.0 * 11 / len(coluna)
    assert distintos_pct >= 10.0, "o fixture tem de violar SÓ a razão, não os distintos"
    assert manter[0], "a condição é CONJUNTA: só a razão não basta para descartar"


def test_fica_quando_so_os_distintos_valem() -> None:
    """**A segunda metade isolada.** Poucos valores distintos, bem distribuídos.

    Sem esta asserção, uma implementação que ignorasse `freq_cut` passaria.
    """
    coluna = [1.0] * 11 + [2.0] * 10
    matriz = _matriz({"dois_valores_equilibrados": coluna}, n_qc=4)
    _, manter, _ = filter_near_zero_variance(matriz)
    assert 100.0 * 2 / len(coluna) < 10.0, "o fixture tem de violar SÓ os distintos"
    assert 11 / 10 <= 19.0, "e a razão de frequência NÃO pode ser violada"
    assert manter[0], "a condição é CONJUNTA: só os distintos não bastam"


def test_os_parametros_viajam_no_stage_result() -> None:
    matriz = _matriz({"a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]})
    _, _, resultado = filter_near_zero_variance(matriz, freq_cut=5.0, unique_cut=20.0)
    assert resultado.labels == {"freq_cut": "5", "unique_cut": "20"}


# ---------------------------------------------------------------------------
# O filtro por posto — a etapa que a paridade com o MetaboAnalyst acrescentou
# ---------------------------------------------------------------------------


def _quatro_colunas_com_variabilidade_conhecida() -> RefinoMatrix:
    """Quatro colunas cuja ORDEM de variabilidade é a mesma em qualquer estatística.

    `a` varia muito, `d` é constante, e `b` e `c` ficam no meio. Escolhida assim de
    propósito: se a ordem dependesse da estatística, um teste que passasse com `iqr` não
    diria nada sobre `sd`, e a parametrização seria falsa cobertura.
    """
    return _matriz(
        {
            "a": [1.0, 50.0, 2.0, 80.0, 3.0, 99.0],
            "b": [10.0, 20.0, 11.0, 22.0, 12.0, 24.0],
            "c": [5.0, 6.0, 5.0, 6.0, 5.0, 6.0],
            "d": [7.0, 7.0, 7.0, 7.0, 7.0, 7.0],
        }
    )


def test_none_salta_a_etapa_e_devolve_a_matriz_inteira() -> None:
    """A metade do par causal que garante que quem não escolher não muda de resultado."""
    matriz = _quatro_colunas_com_variabilidade_conhecida()
    filtrada, manter, resultado = filter_by_ranking(matriz, "none")
    assert filtrada.feature_names == matriz.feature_names
    assert bool(manter.all())
    assert resultado.n_features_in == resultado.n_features_out == 4
    assert resultado.labels["saltada"] == "ranking_filter=none"


# **As cinco de DISPERSAO.** `mean` e `median` ordenam por intensidade e nao por
# variabilidade, entao o fixture acima nao os separa — eles tem teste proprio abaixo.
# Deixa-los fora desta lista e deliberado; deixa-los sem teste NENHUM foi o defeito que a
# revisao apanhou: sete implementacoes registadas e cinco exercitadas.
@pytest.mark.parametrize("metodo", ["iqr", "sd", "mad", "rsd", "nrsd"])
def test_o_posto_mantem_as_mais_variaveis(metodo: str) -> None:
    """A outra metade: com método e fração explícitos, sobram as que devem sobrar.

    Metade de quatro é duas, e as duas mais variáveis são `a` e `b` em todas as cinco
    estatísticas de dispersão — é a propriedade do fixture, afirmada abaixo.
    """
    matriz = _quatro_colunas_com_variabilidade_conhecida()
    filtrada, _, resultado = filter_by_ranking(matriz, metodo, keep_fraction=0.5)
    assert filtrada.feature_names == ("a", "b"), (
        f"`{metodo}` manteve {filtrada.feature_names}, esperava ('a', 'b')"
    )
    assert resultado.n_features_out == 2
    assert resultado.labels["method"] == metodo


def test_a_coluna_constante_da_estatistica_indefinida_no_nrsd() -> None:
    """Indefinido REPROVA, e conta numa causa própria.

    `d` é constante e não nula: `mad` dá zero e `nrsd` dá `0/7 = 0`, que é definido. A
    indefinição precisa de mediana **zero**, e é essa a coluna que este teste constrói —
    escolher a coluna errada faria a asserção passar sem exercitar o ramo.
    """
    matriz = _matriz({"viva": [1.0, 9.0, 2.0, 8.0, 3.0, 7.0], "mediana_zero": [0.0] * 6})
    _, _, resultado = filter_by_ranking(matriz, "nrsd", keep_fraction=1.0)
    assert resultado.discarded_by_reason["estatistica_indefinida"] == 1
    assert resultado.discarded_by_reason["abaixo_do_posto"] == 0


@pytest.mark.parametrize(
    ("n_features", "esperado"),
    [(249, 0.95), (250, 0.90), (500, 0.90), (501, 0.75), (1000, 0.75), (1001, 0.60)],
)
def test_a_fracao_por_tamanho_afirma_as_quatro_fronteiras_dos_dois_lados(
    n_features: int, esperado: float
) -> None:
    """Transcrição de números, e é onde ela erra sem nada levantar.

    Trocar `<` por `<=` numa das fronteiras muda a fração de UMA matriz — a de exatamente
    250, 500 ou 1.000 features — e de mais nenhuma. Afirmar só um lado de cada degrau
    deixaria a troca passar.
    """
    assert fracao_retida_por_tamanho(n_features) == esperado


def test_ao_menos_uma_feature_sobrevive_a_fracao_pequena() -> None:
    """Fração que trunca para zero produziria matriz vazia por arredondamento.

    `process_matrix` traduziria isso em `EmptyMatrixError` sobre um corte que o usuário não
    pediu tão agressivo — o defeito apareceria como a etapa a levantar, não como a fração a
    estar errada.
    """
    matriz = _quatro_colunas_com_variabilidade_conhecida()
    filtrada, _, _ = filter_by_ranking(matriz, "sd", keep_fraction=0.01)
    assert filtrada.n_features == 1


def test_fracao_fora_da_faixa_levanta() -> None:
    """Fração em percentual retém tudo e a etapa vira identidade em silêncio."""
    matriz = _quatro_colunas_com_variabilidade_conhecida()
    for invalida in (0.0, -0.5, 50.0):
        with pytest.raises(ValueError, match="FRAÇÃO"):
            filter_by_ranking(matriz, "sd", keep_fraction=invalida)


def test_metodo_invalido_levanta_na_fronteira() -> None:
    """`Literal` é análise estática; a guarda em runtime é a fronteira."""
    with pytest.raises(InvalidLabelError, match="RankingFilter"):
        filter_by_ranking(_quatro_colunas_com_variabilidade_conhecida(), "interquartile")  # type: ignore[arg-type]


def test_o_filtro_por_posto_nao_altera_a_matriz_de_entrada() -> None:
    """Regra 7 do README — a imutabilidade alcança o array.

    **O nome tinha de ser próprio, e o genérico já estava tomado.** Escrito como
    `test_a_matriz_de_entrada_nao_e_alterada`, ele redefinia o teste homônimo do filtro de
    RSD — que passava a NÃO correr, em silêncio, com a suíte igualmente verde. Foi o `F811`
    do `ruff` que o apanhou, e nenhuma execução o teria feito.
    """
    matriz = _quatro_colunas_com_variabilidade_conhecida()
    antes = np.array(matriz.values, copy=True)
    filter_by_ranking(matriz, "sd", keep_fraction=0.5)
    assert np.array_equal(np.asarray(matriz.values), antes)
    assert matriz.n_features == 4


@pytest.mark.parametrize("metodo", ["mean", "median"])
def test_o_posto_por_intensidade_mantem_as_mais_altas(metodo: str) -> None:
    """As duas estatísticas de tendência central, que a parametrização de dispersão não cobre.

    Elas ordenam por **intensidade** e não por variabilidade, e o fixture de dispersão não as
    separa — nele a coluna mais variável é também a de maior média. Este fixture inverte isso
    de propósito: `alta_estavel` tem a maior média e a menor dispersão, então um teste que
    passasse com `sd` no lugar de `mean` reprovaria aqui.
    """
    matriz = _matriz(
        {
            "alta_estavel": [100.0, 101.0, 100.0, 101.0, 100.0, 101.0],
            "baixa_variavel": [1.0, 40.0, 2.0, 39.0, 1.0, 41.0],
        }
    )
    filtrada, _, _ = filter_by_ranking(matriz, metodo, keep_fraction=0.5)
    assert filtrada.feature_names == ("alta_estavel",), (
        f"`{metodo}` manteve {filtrada.feature_names}: ele tem de ordenar por intensidade, e "
        "esta matriz foi construída para que a mais intensa NÃO seja a mais variável"
    )

    por_dispersao, _, _ = filter_by_ranking(matriz, "sd", keep_fraction=0.5)
    assert por_dispersao.feature_names == ("baixa_variavel",), (
        "âncora: o fixture não separa intensidade de dispersão, e o teste acima passaria com "
        "a implementação errada"
    )


def test_a_estatistica_ignora_zeros_porque_zero_e_ausente() -> None:
    """A definição de ausente deste domínio, aplicada onde ela mudou o número.

    Zero **é** ausência aqui — o equipamento deixa a célula vazia quando não detecta. A
    primeira versão mascarava só o não finito, então uma coluna com dois valores altos e
    quatro zeros pontuava como se os zeros fossem medida, e as estatísticas ordenavam as
    features por quanta **não-detecção** elas têm.

    A planta separa: `com_zeros` tem os mesmos dois valores presentes que `densa_baixa`, e
    difere só nos zeros. Sob a definição certa as duas médias são iguais; sob a antiga a de
    `com_zeros` cairia a um terço.
    """
    matriz = _matriz(
        {
            "com_zeros": [30.0, 30.0, 0.0, 0.0, 0.0, 0.0],
            "densa_baixa": [10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
        }
    )
    filtrada, _, resultado = filter_by_ranking(matriz, "mean", keep_fraction=0.5)
    assert filtrada.feature_names == ("com_zeros",), (
        "a média contou os zeros como medida: `com_zeros` tem média 30 sobre as células "
        "PRESENTES e 10 se os zeros entrarem, e a segunda leitura empata com `densa_baixa`"
    )
    assert resultado.labels["n_avaliaveis"] == "2", (
        "a população em que a fração agiu não foi reportada, e sem ela `keep_fraction` no log "
        "não diz sobre quantas features ela agiu"
    )


# ---------------------------------------------------------------------------
# `exclude` no PIPELINE — o cenário que o plano exigia e que faltava
# ---------------------------------------------------------------------------


def _spec_com(imputacao: str, **troca: object) -> object:
    from refino.domain.spec import PipelineSpec

    campos: dict[str, object] = {
        "correction": "none",
        "n_batches": 1,
        "rsd_cutoff": None,
        "ranking_filter": "none",
        "ranking_keep_fraction": None,
        "ranking_filter_2": "none",
        "ranking_keep_fraction_2": None,
        "rsd_variant": "outlier_removed",
        "missing_threshold": None,
        "imputation": imputacao,
        "sample_normalization": "none",
        "normalization_reference": None,
        "normalization_factors": None,
        "transformation": "none",
        "scaling": "none",
    }
    campos.update(troca)
    return PipelineSpec(**campos)  # type: ignore[arg-type]


def test_exclude_que_esvazia_a_matriz_levanta_nomeando_a_etapa_e_a_causa() -> None:
    """O par causal do único método de imputação que muda a FORMA.

    **Este cenário estava no plano desta unidade e não foi implementado**, e a revisão o
    apanhou. Ele importa por duas razões: `_exige_features` foi acrescentado depois do
    `impute` exactamente para este caso, e a mensagem dele nomeia o motivo do descarte — que
    ficaria como parêntese vazio se o `discarded_by_reason` não fosse preenchido.
    """
    from refino.preprocessing.pipeline import EmptyMatrixError, process_matrix

    toda_com_ausente = _matriz(
        {"a": [1.0, 2.0, 3.0, 4.0, 5.0, 0.0], "b": [0.0, 2.0, 3.0, 4.0, 5.0, 6.0]}
    )
    with pytest.raises(EmptyMatrixError, match="impute") as erro:
        process_matrix(toda_com_ausente, _spec_com("exclude"))  # type: ignore[arg-type]
    assert "ausente_na_feature=2" in str(erro.value), (
        f"a mensagem não nomeia a causa do descarte: {erro.value}. Sem ela o erro diz "
        "«descartou TODAS as 2 features ()» — parêntese vazio no lugar do motivo"
    )


def test_exclude_que_nao_esvazia_reporta_a_contagem_e_segue() -> None:
    """A outra direção: sem ela, a de cima passaria com `exclude` a descartar tudo sempre."""
    from refino.preprocessing.pipeline import process_matrix

    uma_completa = _matriz(
        {"completa": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], "com_zero": [1.0, 2.0, 3.0, 4.0, 5.0, 0.0]}
    )
    saida, etapas = process_matrix(uma_completa, _spec_com("exclude"))  # type: ignore[arg-type]
    assert saida.feature_names == ("completa",)

    do_impute = next(e for e in etapas if e.stage == "impute")
    assert (do_impute.n_features_in, do_impute.n_features_out) == (2, 1)
    assert do_impute.discarded_by_reason == {"ausente_na_feature": 1}


def test_os_outros_metodos_nao_declaram_causa_de_descarte() -> None:
    """Âncora contra a contagem virar ruído: sete dos oito preservam a forma.

    Sem ela, preencher `discarded_by_reason` incondicionalmente passaria — e todo relatório
    de pipeline ganharia uma causa com valor zero, que se lê como «descartou nada por este
    motivo» em vez de «este motivo não se aplica».
    """
    from refino.preprocessing.pipeline import process_matrix

    matriz = _matriz({"a": [1.0, 2.0, 3.0, 4.0, 5.0, 0.0], "b": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]})
    _, etapas = process_matrix(matriz, _spec_com("one_fifth_min"))  # type: ignore[arg-type]
    do_impute = next(e for e in etapas if e.stage == "impute")
    assert do_impute.discarded_by_reason == {}
    assert do_impute.n_features_in == do_impute.n_features_out == 2


# ---------------------------------------------------------------------------
# O FILTRO EM DOIS EIXOS, e a âncora separa TRÊS causas antes do veredito
# ---------------------------------------------------------------------------


def _spec_de_posto(**troca: object) -> object:
    """Uma especificação com os dois passes de posto configuráveis."""
    from refino.domain.spec import PipelineSpec

    campos: dict[str, object] = {
        "correction": "none",
        "n_batches": 1,
        "rsd_cutoff": None,
        "ranking_filter": "none",
        "ranking_keep_fraction": None,
        "ranking_filter_2": "none",
        "ranking_keep_fraction_2": None,
        "rsd_variant": "outlier_removed",
        "missing_threshold": None,
        "imputation": "one_fifth_min",
        "sample_normalization": "none",
        "normalization_reference": None,
        "normalization_factors": None,
        "transformation": "none",
        "scaling": "none",
    }
    campos.update(troca)
    return PipelineSpec(**campos)  # type: ignore[arg-type]


def _matriz_para_posto(n_features: int = 40, semente: int = 20260910) -> RefinoMatrix:
    """Matriz em que variância e abundância ordenam as features DIFERENTE.

    **É a planta que separa**: se as duas estatísticas dessem o mesmo posto, encadear os
    dois passes seria indistinguível de correr o primeiro duas vezes, e a âncora abaixo passaria
    sem que nada do encadeamento fosse exercitado. A construção põe as duas em oposição — as
    features de maior nível são as de menor dispersão relativa — e o teste afirma isso antes de
    medir o resto.
    """
    rng = np.random.default_rng(semente)
    nivel = np.linspace(100.0, 10_000.0, n_features)
    # Dispersão que CAI com o nível: quem tem mais abundância varia menos.
    dispersao = np.linspace(0.45, 0.02, n_features)
    valores = np.abs(nivel * (1.0 + dispersao * rng.normal(size=(12, n_features))))
    amostras = tuple(
        SampleMetadata(
            name=f"S{i}",
            injection_order=i + 1,
            sample_type="qc" if i < 4 else "sample",
            group="QC" if i < 4 else "G",
        )
        for i in range(valores.shape[0])
    )
    return RefinoMatrix(
        values=valores,
        feature_names=tuple(f"f{j}" for j in range(n_features)),
        samples=amostras,
        mode="sintetico",
    )


def test_ancora_variancia_e_abundancia_ordenam_DIFERENTE_neste_fixture() -> None:  # noqa: N802
    """Sem esta âncora, o encadeamento seria testado sobre dado que não o distingue.

    Duas estatísticas que dessem o mesmo posto fariam o segundo passe descartar exactamente as
    que o primeiro já tinha descartado, e o teste da interseção abaixo passaria por construção.
    """
    matriz = _matriz_para_posto()
    so_variancia, _, _ = filter_by_ranking(matriz, "sd", keep_fraction=0.5)
    so_abundancia, _, _ = filter_by_ranking(matriz, "mean", keep_fraction=0.5)

    das_duas = set(so_variancia.feature_names) & set(so_abundancia.feature_names)
    assert das_duas != set(so_variancia.feature_names), (
        "as duas estatísticas retiveram exactamente o mesmo conjunto; sobre este fixture o "
        "encadeamento é indistinguível de correr o primeiro passe duas vezes"
    )


def test_a_ancora_do_encadeamento_separa_as_TRES_causas() -> None:  # noqa: N802
    """A âncora do encadeamento, e ela mede TRÊS coisas antes de concluir.

    O plano exige esta ordem porque *"o conjunto encadeado é subconjunto do da variância
    sozinha"* é verdade por três razões diferentes, e só uma delas é o encadeamento:

    1. **100% de retenção tem de ser identidade em CONTAGEM.** Se não for, a diferença vem de
       `ESTATISTICA_INDEFINIDA` — feature sobre a qual a estatística não se calcula reprova, e
       isso não é o corte por posto. Sem esta medida, o subconjunto abaixo seria explicado pela
       política de indefinido e atribuído ao encadeamento.
    2. **Quantas o segundo passe remove, e quantas dessas estavam no topo do primeiro.** É o que
       distingue *"o segundo cortou"* de *"o segundo cortou o que o primeiro já ia cortar"*.
    3. **Só então** o subconjunto — que passa a ser informação e não tautologia.
    """
    matriz = _matriz_para_posto()

    # (1) O braço de abundância a 100% não muda a CONTAGEM.
    inteiro, _, resultado_inteiro = filter_by_ranking(matriz, "mean", keep_fraction=1.0)
    assert resultado_inteiro.n_features_out == resultado_inteiro.n_features_in, (
        f"com fração retida de 100% a etapa descartou "
        f"{resultado_inteiro.n_features_in - resultado_inteiro.n_features_out} features. A "
        "identidade não vale, e a causa é a política de indefinido — não a ordem dos passes"
    )
    assert set(inteiro.feature_names) == set(matriz.feature_names)

    # (2) Quanto o segundo passe de facto remove, e de onde.
    so_variancia, _, _ = filter_by_ranking(matriz, "sd", keep_fraction=0.6)
    encadeado, _, _ = filter_by_ranking(so_variancia, "mean", keep_fraction=0.5)
    removidas = set(so_variancia.feature_names) - set(encadeado.feature_names)
    assert removidas, (
        "o segundo passe não removeu nada: o encadeamento é identidade e o subconjunto abaixo "
        "seria trivialmente verdadeiro"
    )

    # (3) Agora o subconjunto é informação.
    assert set(encadeado.feature_names) < set(so_variancia.feature_names)


def test_a_populacao_do_segundo_passe_e_MENOR_e_o_rotulo_diz_qual_e(  # noqa: N802
) -> None:
    """A fração do segundo passe age sobre o que o primeiro deixou, e o rótulo carrega a base.

    **Sem `n_avaliaveis` no rótulo, `keep_fraction=0,5` no registro não diz sobre quantas.** É o
    escopo aplicado a um par de etapas: o mesmo número, `0,5`, corta populações
    diferentes nos dois passes, e só a base torna a contagem de saída conferível.
    """
    matriz = _matriz_para_posto()
    primeiro, _, r1 = filter_by_ranking(matriz, "sd", keep_fraction=0.6)
    _, _, r2 = filter_by_ranking(primeiro, "mean", keep_fraction=0.5, stage="filter_by_ranking_2")

    base_1 = int(r1.labels["n_avaliaveis"])
    base_2 = int(r2.labels["n_avaliaveis"])
    assert base_2 < base_1, (
        f"o segundo passe reportou base {base_2} e o primeiro {base_1}: ele não está a operar "
        "sobre a população que o primeiro deixou"
    )
    assert r1.n_features_out == round(0.6 * base_1)
    assert r2.n_features_out == round(0.5 * base_2)


def test_os_dois_passes_aparecem_SEPARADOS_no_resumo_por_etapa() -> None:  # noqa: N802
    """Medido de ponta a ponta: dois nomes de etapa, duas entradas no resumo."""
    from refino.preprocessing.pipeline import process_matrix, stage_counts

    matriz = _matriz_para_posto()
    spec = _spec_de_posto(
        ranking_filter="sd",
        ranking_keep_fraction=0.6,
        ranking_filter_2="mean",
        ranking_keep_fraction_2=0.5,
    )
    _, resultados = process_matrix(matriz, spec)  # type: ignore[arg-type]
    resumo = stage_counts(resultados)

    assert "filter_by_ranking" in resumo and "filter_by_ranking_2" in resumo, (
        f"o resumo por etapa tem {sorted(resumo)} e os dois passes de posto têm de aparecer "
        "separados; um nome partilhado faz o segundo sobrescrever o primeiro"
    )
    assert (
        resumo["filter_by_ranking"]["n_features_out"]
        > resumo["filter_by_ranking_2"]["n_features_out"]
    )


def test_par_causal_o_nome_de_etapa_PARTILHADO_apaga_uma_entrada() -> None:  # noqa: N802
    """A planta, e é ela que prova que o nome distinto não é decoração.

    Com os dois passes a reportarem o mesmo nome, o resumo — que é um dicionário — perde uma
    entrada, **enquanto a lista de resultados continua com as duas**. É por isso que a
    verificação de consistência da cadeia não apanha isto: ela percorre a lista.
    """
    from refino.preprocessing.pipeline import stage_counts

    matriz = _matriz_para_posto()
    primeiro, _, r1 = filter_by_ranking(matriz, "sd", keep_fraction=0.6)
    _, _, r2_partilhado = filter_by_ranking(primeiro, "mean", keep_fraction=0.5)
    _, _, r2_distinto = filter_by_ranking(
        primeiro, "mean", keep_fraction=0.5, stage="filter_by_ranking_2"
    )

    assert len(stage_counts([r1, r2_partilhado])) == 1, (
        "com o nome partilhado o resumo devia colapsar as duas etapas numa; se não colapsa, "
        "esta planta não separa nada e KTD8 não descreve o mecanismo"
    )
    assert len(stage_counts([r1, r2_distinto])) == 2


def test_a_fracao_fora_do_intervalo_e_recusada_na_FRONTEIRA() -> None:  # noqa: N802
    """A guarda vive no contrato, e ela só é correta sob fração RETIDA.

    Sob a convenção de fração **cortada**, `(0, 1]` recusaria `0` — que é o caso de controle,
    *"cortar 0% é identidade"* — e admitiria `1`, que filtraria tudo. A
    conversão está escrita em `PipelineSpec`: 40%, 0% e 10% cortados são 60%, 100% e 90% retidos.
    """
    import pytest as _pytest

    for valor in (0.0, -0.1, 1.5, 60.0):
        with _pytest.raises(ValueError, match="FRACAO"):
            _spec_de_posto(ranking_filter="sd", ranking_keep_fraction=valor)
        with _pytest.raises(ValueError, match="FRACAO"):
            _spec_de_posto(ranking_filter_2="mean", ranking_keep_fraction_2=valor)

    # E o caso de controle do critério 5 é ACEITE, porque em fração retida ele é `1.0`.
    _spec_de_posto(ranking_filter="mean", ranking_keep_fraction=1.0)


def test_sem_fracao_a_regra_por_TAMANHO_decide_e_o_resultado_e_o_de_hoje() -> None:  # noqa: N802
    """`None` devolve a etapa ao que ela sempre fez — é o que protege quem não escolheu."""
    matriz = _matriz_para_posto()
    sem_fracao, _, r_sem = filter_by_ranking(matriz, "sd")
    pela_regra = fracao_retida_por_tamanho(int(r_sem.labels["n_avaliaveis"]))
    com_a_mesma, _, r_com = filter_by_ranking(matriz, "sd", keep_fraction=pela_regra)

    assert sem_fracao.feature_names == com_a_mesma.feature_names
    assert r_sem.n_features_out == r_com.n_features_out
