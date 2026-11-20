"""A correlação entre features e a busca por padrão: o coeficiente, o indefinido e a ordem.

## O que este módulo afirma, e o que ele deliberadamente NÃO afirma

**Nenhuma asserção sobre um valor de correlação «realista».** O que se afirma são os extremos que
a definição do coeficiente obriga — `r = 1` entre uma feature e ela própria, `r = -1` entre ela e o
seu negativo — e a **divergência** entre os três métodos sobre o dado em que eles têm de divergir.
Um teste que comparasse `0,73` contra `0,73` estaria a afirmar a implementação do `numpy`, não a
nossa.

## A planta que separa Pearson de Spearman cai sobre dado MONOTÔNICO NÃO LINEAR

*"Pearson e Spearman divergem"* é falso sobre dado linear: ali os dois valem exactamente 1, e o
teste passaria com **qualquer** um dos dois no lugar do outro — um despacho que ignorasse o método
pedido ficaria verde. A planta que separa é `y = x³` sobre `x` crescente: Spearman e Kendall valem
1 por serem sobre postos, e Pearson mede menos. O teste afirma as duas metades, e é a segunda que
faz a primeira significar alguma coisa.

## Indefinido não é zero, e a asserção é de AUSÊNCIA e de PRESENÇA ao mesmo tempo

Uma feature constante não tem correlação definida com nada. A escolha é excluí-la da matriz e
**contá-la** — nunca escrever `0,0` numa célula, que se leria como *"medi e não há associação"*.
Afirmar só que a célula não existe seria asserção de ausência: o teste afirma, na mesma
função e antes, que a matriz devolvida **tem** população.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest

from refino.domain.analysis import AnalysisSpec
from refino.domain.cancellation import CancelCheck, ExecutionCancelledError
from refino.domain.labels import CorrelationMethod
from refino.domain.matrix import RefinoMatrix
from refino.domain.metadata import SampleMetadata
from refino.plots.series import ordem_canonica
from refino.statistics.correlacao import (
    CORRELATION_IMPLEMENTATIONS,
    SEPARADOR_DO_PERFIL,
    CorrelationResult,
    PatternSearchResult,
    ReferenciaDesconhecidaError,
    feature_correlation,
    ordem_dos_grupos,
    pattern_search,
)

# ---------------------------------------------------------------------------
# As fixturas sintéticas
# ---------------------------------------------------------------------------


def matriz_de(
    valores: np.ndarray,
    *,
    grupos: Sequence[str] | None = None,
    tipos: Sequence[str] | None = None,
    nomes: Sequence[str] | None = None,
) -> RefinoMatrix:
    """Uma `RefinoMatrix` sintética a partir de um array, com grupos e tipos opcionais por linha."""
    array = np.ascontiguousarray(valores, dtype=np.float64)
    array.flags.writeable = False
    n_amostras, n_features = array.shape
    nomes_de_grupo = list(grupos) if grupos is not None else ["G"] * n_amostras
    tipos_de_amostra = list(tipos) if tipos is not None else ["sample"] * n_amostras
    amostras = tuple(
        SampleMetadata(
            name=f"s{i:03d}",
            injection_order=i + 1,
            sample_type=tipos_de_amostra[i],  # type: ignore[arg-type]
            group=nomes_de_grupo[i],
        )
        for i in range(n_amostras)
    )
    return RefinoMatrix(
        values=array,
        feature_names=tuple(nomes)
        if nomes is not None
        else tuple(f"f{j:03d}" for j in range(n_features)),
        samples=amostras,
        mode="sintetico",
    )


def com_gemea_e_oposta(*, n: int = 12, seed: int = 20260918) -> RefinoMatrix:
    """Três features: uma base, a sua cópia EXACTA, e o seu negativo.

    É a fixtura dos dois extremos que a definição obriga: `r(base, gemea) = 1` e
    `r(base, oposta) = -1`, nos três métodos.
    """
    base = np.random.default_rng(seed).normal(0.0, 1.0, n)
    return matriz_de(
        np.column_stack([base, base.copy(), -base]),
        nomes=("base", "gemea", "oposta"),
    )


def monotonica_nao_linear(*, n: int = 12) -> RefinoMatrix:
    """Duas features ligadas por `y = x³` — monotônica e NÃO linear.

    A planta cai onde os métodos diferem: sobre dado linear os três métodos coincidem e o teste
    seria vácuo.
    """
    x = np.linspace(1.0, 4.0, n)
    return matriz_de(np.column_stack([x, x**3]), nomes=("x", "x_ao_cubo"))


def com_constante(*, n: int = 10, seed: int = 20260918) -> RefinoMatrix:
    """Duas features com variação e UMA constante — a que não tem correlação definida."""
    rng = np.random.default_rng(seed)
    return matriz_de(
        np.column_stack([rng.normal(size=n), rng.normal(size=n), np.full(n, 7.0)]),
        nomes=("a", "b", "parada"),
    )


def quatro_grupos(*, por_grupo: int = 4, seed: int = 20260918) -> RefinoMatrix:
    """Quatro grupos clínicos, com uma feature que segue a ORDEM CANÔNICA e uma que não segue.

    `degrau` vale 0, 1, 2, 3 na ordem canônica dos grupos; `ruido` é sorteio. É a fixtura da busca
    por padrão: contra o perfil `0,1,2,3` a primeira tem de sair no topo com `r = 1`.
    """
    nomes_de_grupo = ["ATLL", "Asymptomatic", "HAMTSP", "NoHTLV"]
    grupos = [g for g in nomes_de_grupo for _ in range(por_grupo)]
    canonica = ordem_canonica(grupos)
    degrau = np.array([float(canonica.index(g)) for g in grupos])
    rng = np.random.default_rng(seed)
    return matriz_de(
        np.column_stack([degrau, rng.normal(size=len(grupos)), -degrau]),
        grupos=grupos,
        nomes=("degrau", "ruido", "degrau_invertido"),
    )


# ---------------------------------------------------------------------------
# 1. O registro de implementações, e a paridade com o `Literal`
# ---------------------------------------------------------------------------


def test_o_registro_cobre_exatamente_o_vocabulario() -> None:
    """Método novo sem nome, ou nome sem método."""
    from typing import get_args

    assert set(CORRELATION_IMPLEMENTATIONS) == set(get_args(CorrelationMethod))


# ---------------------------------------------------------------------------
# 2. Os dois extremos que a definição obriga
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("metodo", ["pearson", "spearman", "kendall"])
def test_a_gemea_da_um_e_a_oposta_da_menos_um(metodo: str) -> None:
    """`r = 1` com a cópia exacta, `r = -1` com o negativo. Vale nos três métodos."""
    resultado = feature_correlation(
        com_gemea_e_oposta(),
        AnalysisSpec(correlation_method=metodo),  # type: ignore[arg-type]
    )
    assert resultado.features == ("base", "gemea", "oposta")
    indice = {nome: i for i, nome in enumerate(resultado.features)}
    matriz = resultado.matrix
    assert matriz[indice["base"], indice["gemea"]] == pytest.approx(1.0, abs=1e-12)
    assert matriz[indice["base"], indice["oposta"]] == pytest.approx(-1.0, abs=1e-12)
    assert np.allclose(np.diag(matriz), 1.0, atol=1e-12), "a diagonal de uma correlação é 1"
    assert np.allclose(matriz, matriz.T, atol=1e-12), "a matriz de correlação é simétrica"


def test_a_matriz_devolvida_e_IMUTAVEL() -> None:  # noqa: N802
    """Regra 7 do README aplicada ao resultado: quem recebe não pode mutar o que outros vão ler."""
    resultado = feature_correlation(com_gemea_e_oposta())
    with pytest.raises(ValueError, match="read-only"):
        resultado.matrix[0, 0] = 99.0


def test_a_matriz_de_ENTRADA_nao_e_tocada() -> None:  # noqa: N802
    """Regra 7 do README: nenhuma operação altera a matriz de entrada."""
    matriz = com_gemea_e_oposta()
    antes = matriz.values.copy()
    feature_correlation(matriz)
    assert np.array_equal(matriz.values, antes)


# ---------------------------------------------------------------------------
# 3. A planta que separa os métodos — e a âncora de que ela separa
# ---------------------------------------------------------------------------


def test_pearson_e_spearman_DIVERGEM_sobre_monotonica_nao_linear() -> None:  # noqa: N802
    """A planta: sobre dado LINEAR os dois valem 1 e o teste seria vácuo.

    A âncora está na primeira asserção — sobre `y = x³` o coeficiente de posto é exactamente 1 e o
    de Pearson não é. Sem ela, um despacho que ignorasse `correlation_method` ficaria verde.
    """
    matriz = monotonica_nao_linear()
    por_metodo = {
        metodo: feature_correlation(
            matriz,
            AnalysisSpec(correlation_method=metodo),  # type: ignore[arg-type]
        ).matrix[0, 1]
        for metodo in ("pearson", "spearman", "kendall")
    }
    assert por_metodo["spearman"] == pytest.approx(1.0, abs=1e-12)
    assert por_metodo["kendall"] == pytest.approx(1.0, abs=1e-12)
    assert por_metodo["pearson"] < 0.99, (
        f"Pearson mediu {por_metodo['pearson']} sobre `y = x³`: a planta não separa nada, e um "
        "despacho que ignorasse o método pedido passaria"
    )


def test_o_LINEAR_e_a_anti_planta_e_esta_declarado() -> None:  # noqa: N802
    """O par causal da planta acima: sobre dado linear os três coincidem.

    Não é redundância — é o que prova que a divergência medida vem da **forma** do dado e não de
    uma implementação errada de um dos três.
    """
    x = np.linspace(1.0, 4.0, 12)
    linear = matriz_de(np.column_stack([x, 3.0 * x + 5.0]), nomes=("x", "linear_em_x"))
    valores = [
        feature_correlation(
            linear,
            AnalysisSpec(correlation_method=metodo),  # type: ignore[arg-type]
        ).matrix[0, 1]
        for metodo in ("pearson", "spearman", "kendall")
    ]
    assert all(v == pytest.approx(1.0, abs=1e-12) for v in valores), valores


# ---------------------------------------------------------------------------
# 4. Indefinido é CONTADO e declarado, nunca zero
# ---------------------------------------------------------------------------


def test_a_feature_constante_e_contada_e_NAO_vira_zero() -> None:  # noqa: N802
    """Dois casos que não colapsam, na escala de uma célula: `0,0` leria-se como *"medi e não há
    associação"*.

    **A asserção de não-vacuidade vai na MESMA função e ANTES**: sem ela, um resultado que
    descartasse tudo passaria — `"parada" not in features` é verdade sobre um resultado vazio.
    """
    resultado = feature_correlation(com_constante())

    assert resultado.n_features_used == 2, "âncora: sobraram features para correlacionar"
    assert resultado.matrix.shape == (2, 2)
    assert resultado.features == ("a", "b")

    assert "parada" not in resultado.features
    assert resultado.n_features_dropped_constant == 1
    assert resultado.n_features_dropped_non_finite == 0
    assert resultado.n_features_total == 3


def test_a_coluna_NAO_FINITA_e_contada_a_parte() -> None:  # noqa: N802
    """Dois motivos e não um: constante é fato do dado, não finito é fato do processamento."""
    valores = np.column_stack(
        [
            np.linspace(0.0, 1.0, 8),
            np.linspace(1.0, 3.0, 8),
            np.array([np.nan] + [1.0] * 7),
        ]
    )
    resultado = feature_correlation(matriz_de(valores, nomes=("a", "b", "rota")))
    assert resultado.n_features_used == 2, "âncora: sobraram features"
    assert resultado.n_features_dropped_non_finite == 1
    assert resultado.n_features_dropped_constant == 0


def test_matriz_sem_feature_utilizavel_LEVANTA_em_vez_de_devolver_vazio() -> None:  # noqa: N802
    """Resultado vazio seria indistinguível de *"não há associação nenhuma"*."""
    constante = matriz_de(np.full((6, 3), 2.0))
    with pytest.raises(ValueError, match="utilizáve"):
        feature_correlation(constante)


# ---------------------------------------------------------------------------
# 5. A população é a de ESTUDO, e isso é declarado
# ---------------------------------------------------------------------------


def test_os_QC_ficam_de_FORA_da_populacao() -> None:  # noqa: N802
    """Sete réplicas da mesma mistura encolheriam a variação medida, com o nome da matriz inteira.

    **É um par causal:** a mesma matriz com e sem as linhas de QC dá a **mesma** correlação, o que
    só é verdade se as de QC não entrarem na conta.
    """
    base = np.random.default_rng(7).normal(size=(10, 3))
    so_estudo = matriz_de(base, nomes=("a", "b", "c"))
    com_qc = matriz_de(
        np.vstack([base, np.full((4, 3), 5.0)]),
        tipos=["sample"] * 10 + ["qc"] * 4,
        nomes=("a", "b", "c"),
    )
    assert feature_correlation(com_qc).n_samples_used == 10
    assert np.allclose(
        feature_correlation(com_qc).matrix,
        feature_correlation(so_estudo).matrix,
        atol=1e-12,
    )


def test_matriz_sem_amostra_de_estudo_LEVANTA() -> None:  # noqa: N802
    """Uma matriz só de QC não tem população para a pergunta que esta análise faz."""
    so_qc = matriz_de(
        np.random.default_rng(3).normal(size=(6, 3)),
        tipos=["qc"] * 6,
    )
    with pytest.raises(ValueError, match="amostra"):
        feature_correlation(so_qc)


# ---------------------------------------------------------------------------
# 6. A seleção de colunas é de quem CHAMA, e o recorte dela é declarado
# ---------------------------------------------------------------------------


def test_colunas_recorta_a_populacao_e_o_denominador_FICA() -> None:  # noqa: N802
    """Quantas features entram é decisão de quem chama (KD2); o total continua a viajar."""
    matriz = matriz_de(np.random.default_rng(11).normal(size=(9, 6)))
    resultado = feature_correlation(matriz, colunas=(0, 2, 4))
    assert resultado.features == ("f000", "f002", "f004")
    assert resultado.matrix.shape == (3, 3)
    assert resultado.n_features_used == 3
    assert resultado.n_features_total == 6, (
        "o denominador do rodapé é a matriz inteira; recortá-lo com o numerador apagaria o "
        "«de quantas»"
    )


def test_colunas_repetidas_ou_fora_da_matriz_LEVANTAM() -> None:  # noqa: N802
    """Índice repetido desenharia a mesma feature duas vezes, com a figura bem formada."""
    matriz = matriz_de(np.random.default_rng(11).normal(size=(9, 6)))
    with pytest.raises(ValueError, match="repet"):
        feature_correlation(matriz, colunas=(0, 0, 1))
    with pytest.raises(ValueError, match="fora"):
        feature_correlation(matriz, colunas=(0, 99))


# ---------------------------------------------------------------------------
# 7. A busca por padrão — OQ2, a ordem assumida
# ---------------------------------------------------------------------------


def test_a_busca_contra_uma_FEATURE_devolve_essa_feature_no_topo() -> None:  # noqa: N802
    """Correlação de uma feature consigo própria é 1, e 1 é o máximo: ela é o topo por definição."""
    resultado = pattern_search(
        quatro_grupos(),
        AnalysisSpec(pattern_reference="degrau"),
    )
    assert resultado.features, "âncora: a busca devolveu alguma feature"
    assert resultado.features[0] == "degrau"
    assert resultado.coefficient[0] == pytest.approx(1.0, abs=1e-12)
    assert resultado.reference_kind == "feature"
    assert resultado.reference == "degrau"


def test_a_busca_ordena_por_coeficiente_DESCENDENTE_e_a_oposta_fica_no_fim() -> None:  # noqa: N802
    """Descendente e não por módulo: *"sobe junto"* e *"desce quando sobe"* são achados opostos.

    Colapsá-los pelo valor absoluto seria juntar dois casos numa ordenação — as duas features
    apareceriam juntas no topo, e o pesquisador leria a segunda como a primeira.
    """
    resultado = pattern_search(quatro_grupos(), AnalysisSpec(pattern_reference="degrau"))
    assert len(resultado.features) == 3, "âncora: as três features estão na lista"
    assert resultado.features[-1] == "degrau_invertido"
    assert resultado.coefficient[-1] == pytest.approx(-1.0, abs=1e-12)
    assert list(resultado.coefficient) == sorted(resultado.coefficient, reverse=True)


def test_a_busca_contra_um_PERFIL_por_grupo_usa_a_ordem_canonica() -> None:  # noqa: N802
    """O perfil é lido na ordem canônica dos grupos, e o resultado DECLARA qual ordem assumiu.

    É a resposta a OQ2: com quatro grupos nominais sem ordem natural, a busca só tem significado se
    a ordem for declarada — e a figura escreve-a.
    """
    matriz = quatro_grupos()
    resultado = pattern_search(
        matriz,
        AnalysisSpec(pattern_reference=SEPARADOR_DO_PERFIL.join(("0", "1", "2", "3"))),
    )
    assert resultado.reference_kind == "profile"
    assert resultado.group_order == ("ATLL", "Asymptomatic", "HAMTSP", "NoHTLV")
    assert resultado.profile == (0.0, 1.0, 2.0, 3.0)
    assert resultado.features[0] == "degrau"
    assert resultado.coefficient[0] == pytest.approx(1.0, abs=1e-12)


def test_a_ordem_canonica_do_NUCLEO_e_a_do_DESENHO() -> None:  # noqa: N802
    """A irmã declarada: duas ordens canônicas divergentes trocariam perfil e cor.

    `plots.series.ordem_canonica` decide a **cor** de cada grupo; `ordem_dos_grupos` decide a que
    grupo cada número do perfil pertence. Se as duas divergissem, a figura pintaria uma coorte e
    declararia outra ordem — com a imagem perfeitamente bem formada.
    """
    coortes = (
        ["ATLL", "Asymptomatic", "HAMTSP", "NoHTLV"],
        ["b", "a", "B", "A"],
        ["só um"],
        ["z", "z", "a"],
    )
    for coorte in coortes:
        assert ordem_dos_grupos(coorte) == ordem_canonica(coorte), coorte


def test_referencia_INEXISTENTE_recusa_nomeando_as_disponiveis() -> None:  # noqa: N802
    """A mensagem tem de dizer o que existe — senão o usuário adivinha o nome."""
    with pytest.raises(ReferenciaDesconhecidaError) as erro:
        pattern_search(quatro_grupos(), AnalysisSpec(pattern_reference="nao_existe"))
    mensagem = str(erro.value)
    assert "nao_existe" in mensagem
    assert "degrau" in mensagem, "a recusa não nomeia nenhuma feature disponível"
    assert "4" in mensagem, "a recusa não diz quantos números um perfil precisaria de ter"


def test_perfil_com_o_NUMERO_ERRADO_de_grupos_recusa() -> None:  # noqa: N802
    """Três números para quatro grupos: o alinhamento seria por posição, em silêncio."""
    with pytest.raises(ReferenciaDesconhecidaError, match="4"):
        pattern_search(
            quatro_grupos(),
            AnalysisSpec(pattern_reference=SEPARADOR_DO_PERFIL.join(("0", "1", "2"))),
        )


def test_perfil_CONSTANTE_recusa_em_vez_de_devolver_tudo_indefinido() -> None:  # noqa: N802
    """Um perfil sem variação não tem correlação definida com nada: a figura sairia toda vazia."""
    with pytest.raises(ReferenciaDesconhecidaError, match="constante"):
        pattern_search(
            quatro_grupos(),
            AnalysisSpec(pattern_reference=SEPARADOR_DO_PERFIL.join(("2", "2", "2", "2"))),
        )


def test_a_busca_sem_referencia_LEVANTA_em_vez_de_devolver_vazio() -> None:  # noqa: N802
    """`pattern_reference=None` significa *"não peça"*, e quem chama decide — não esta função."""
    with pytest.raises(ValueError, match="pattern_reference"):
        pattern_search(quatro_grupos(), AnalysisSpec(pattern_reference=None))


def test_a_busca_conta_as_features_que_saiu_da_conta() -> None:
    """A contagem de indefinidas viaja: é ela que a figura declara (R12b)."""
    matriz = matriz_de(
        np.column_stack(
            [
                np.linspace(0.0, 3.0, 8),
                np.full(8, 1.0),
                np.linspace(3.0, 0.0, 8),
            ]
        ),
        grupos=["A", "A", "A", "A", "B", "B", "B", "B"],
        nomes=("sobe", "parada", "desce"),
    )
    resultado = pattern_search(matriz, AnalysisSpec(pattern_reference="sobe"))
    assert resultado.n_features_used == 2, "âncora: sobraram features"
    assert resultado.n_features_dropped_constant == 1
    assert "parada" not in resultado.features


# ---------------------------------------------------------------------------
# 8. O cancelamento, e o par causal
# ---------------------------------------------------------------------------


def _cancela_sempre() -> CancelCheck:
    return lambda: True


def test_o_cancelamento_levanta_com_o_ROTULO_do_ponto() -> None:  # noqa: N802
    """A análise longa recebe o verificador na assinatura do DOMÍNIO."""
    with pytest.raises(ExecutionCancelledError) as erro:
        feature_correlation(com_gemea_e_oposta(), cancelled=_cancela_sempre())
    assert "correlação" in str(erro.value)

    with pytest.raises(ExecutionCancelledError) as erro_da_busca:
        pattern_search(
            quatro_grupos(),
            AnalysisSpec(pattern_reference="degrau"),
            cancelled=_cancela_sempre(),
        )
    assert "padrão" in str(erro_da_busca.value)


def test_sem_cancelamento_pedido_o_resultado_e_o_MESMO() -> None:  # noqa: N802
    """A outra direção do par causal: o parâmetro novo não moveu número nenhum."""
    matriz = com_gemea_e_oposta()
    sem = feature_correlation(matriz)
    com = feature_correlation(matriz, cancelled=lambda: False)
    assert np.array_equal(sem.matrix, com.matrix)
    assert sem.column_digest == com.column_digest


# ---------------------------------------------------------------------------
# 9. Determinismo — regra 3 do README
# ---------------------------------------------------------------------------


def test_duas_execucoes_produzem_saida_IDENTICA() -> None:  # noqa: N802
    """Não há passo estocástico nesta unidade, e a âncora é o que impede que passe a haver."""
    matriz = quatro_grupos()
    primeira = feature_correlation(matriz)
    segunda = feature_correlation(matriz)
    assert np.array_equal(primeira.matrix, segunda.matrix)
    assert primeira.column_digest == segunda.column_digest

    uma = pattern_search(matriz, AnalysisSpec(pattern_reference="degrau"))
    outra = pattern_search(matriz, AnalysisSpec(pattern_reference="degrau"))
    assert uma.features == outra.features
    assert uma.coefficient == outra.coefficient


def test_o_tipo_do_resultado_e_o_declarado() -> None:
    """Âncora de forma: os dois resultados são congelados, como `PipelineSpec` e `AnalysisSpec`."""
    assert isinstance(feature_correlation(com_gemea_e_oposta()), CorrelationResult)
    assert isinstance(
        pattern_search(quatro_grupos(), AnalysisSpec(pattern_reference="degrau")),
        PatternSearchResult,
    )
    with pytest.raises(AttributeError, match="cannot assign"):
        feature_correlation(com_gemea_e_oposta()).method = "spearman"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# 10. Os caminhos de ERRO que sobram — é onde o teste importa
# ---------------------------------------------------------------------------


def test_perfil_com_pedaco_NAO_NUMERICO_recusa() -> None:  # noqa: N802
    """`"a,b,c,d"` tem o número certo de pedaços e nenhum é número."""
    with pytest.raises(ReferenciaDesconhecidaError, match="perfil por grupo"):
        pattern_search(quatro_grupos(), AnalysisSpec(pattern_reference="a,b,c,d"))


def test_perfil_com_valor_NAO_FINITO_recusa() -> None:  # noqa: N802
    """`inf` converte por `float()` e não é um nível de nada.

    **Sem esta guarda o `inf` atravessaria**: `float("inf")` não levanta, `ptp` de um vetor com
    `inf` é `inf` — que não é zero —, e a correlação sairia toda `nan` com a figura bem formada.
    """
    with pytest.raises(ReferenciaDesconhecidaError, match="perfil por grupo"):
        pattern_search(quatro_grupos(), AnalysisSpec(pattern_reference="0,1,2,inf"))


def test_referencia_que_e_feature_CONSTANTE_recusa() -> None:  # noqa: N802
    """A feature existe, e a correlação dela com qualquer outra é `0/0`.

    **É outro fato que «nome inexistente», e a mensagem é outra:** aqui o nome está certo e o que
    falta é variação. Colapsar os dois mandaria o profissional procurar um erro de digitação que
    não existe.
    """
    matriz = matriz_de(
        np.column_stack([np.linspace(0.0, 1.0, 8), np.full(8, 3.0)]),
        nomes=("varia", "parada"),
    )
    with pytest.raises(ReferenciaDesconhecidaError, match="constante"):
        pattern_search(matriz, AnalysisSpec(pattern_reference="parada"))


def test_busca_sobre_matriz_SEM_FEATURE_UTILIZAVEL_levanta() -> None:  # noqa: N802
    """A irmã da guarda do mapa, e ela precisa de cenário próprio.

    A referência é procurada **depois** de as colunas utilizáveis serem escolhidas, então uma matriz
    inteiramente constante cai nesta guarda e não na da referência — e as duas mensagens dizem
    coisas diferentes.
    """
    with pytest.raises(ValueError, match="utilizáve"):
        pattern_search(matriz_de(np.full((6, 3), 2.0)), AnalysisSpec(pattern_reference="f000"))
