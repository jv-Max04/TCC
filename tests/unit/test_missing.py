"""`drop_by_group_missingness` e as oito imputações.

**O quantificador é o que se testa.** O critério pede que feature presente em 100% de um
grupo e ausente em 100% de outro **fique**, e o par que dá dente é ele contra a feature
ausente em todos os grupos: "em todos" e "em algum" dão conjuntos diferentes, e um teste com
um grupo só não os distingue.

**E os fixtures que provam separação são verificados quanto a separar antes de serem usados**
Nas duas imputações estocásticas, o par causal da semente só é par se sementes
diferentes derem resultados diferentes — e isso é propriedade da matriz, não da função:
matriz sem estrutura pode dar o mesmo com qualquer semente.
"""

import numpy as np
import pytest

from refino.domain.labels import InvalidLabelError
from refino.domain.matrix import RefinoMatrix
from refino.domain.metadata import SampleMetadata
from refino.preprocessing.missing import (
    AUSENTE_EM_TODOS_OS_GRUPOS,
    drop_by_group_missingness,
    impute,
    is_missing,
)

METODOS_QUE_IMPUTAM = ("one_fifth_min", "col_min", "median", "mean", "knn", "missforest")
# **Só o `missforest` consulta a semente.** `KNNImputer` não tem aleatoriedade e não aceita
# `random_state`: a vizinhança é determinada por distância, e a média dos vizinhos é uma
# função fechada. Então o determinismo dele é INCONDICIONAL — mais forte que "determinístico
# dada a semente" —, e afirmar que sementes diferentes o mudam seria afirmar algo falso.
#
# Foi descoberto pelo próprio par causal: o teste reprovou para `knn` e a causa não
# era defeito, era a premissa de que os dois métodos são estocásticos.
CONSULTAM_A_SEMENTE = ("missforest",)


def _matriz(colunas: dict[str, list[float]], grupos: list[str]) -> RefinoMatrix:
    """Matriz com grupos explícitos; a primeira linha é QC para satisfazer o domínio."""
    nomes = tuple(colunas)
    valores = np.array([colunas[nome] for nome in nomes], dtype=np.float64).T
    amostras = tuple(
        SampleMetadata(
            name=f"S{i}",
            injection_order=i + 1,
            sample_type="qc" if i == 0 else "sample",
            group=grupo,
        )
        for i, grupo in enumerate(grupos)
    )
    return RefinoMatrix(values=valores, feature_names=nomes, samples=amostras, mode="sintetico")


# Dois grupos de três amostras. A primeira linha é QC e pertence ao grupo "QC", então os
# grupos efetivos são três — declarado porque a contagem entra no `StageResult`.
GRUPOS = ["QC", "A", "A", "B", "B", "B"]


# ---------------------------------------------------------------------------
# O quantificador universal
# ---------------------------------------------------------------------------


def test_ca_5_7_feature_presente_em_um_grupo_e_ausente_noutro_fica() -> None:
    """Ausência específica de grupo é informação biológica.

    Metabólito presente numa condição clínica e ausente noutra é o sinal que o estudo
    procura, e descartá-lo apagaria justamente ele.
    """
    matriz = _matriz(
        {"especifica": [10.0, 20.0, 30.0, 0.0, 0.0, 0.0]},
        GRUPOS,
    )
    _, manter, resultado = drop_by_group_missingness(matriz, threshold=0.8)
    assert manter[0], "feature bem medida em ao menos um grupo tem de FICAR"
    assert resultado.n_discarded == 0


def test_feature_ausente_em_todos_os_grupos_sai() -> None:
    """A outra metade do par: ausência difusa não é informação de grupo."""
    matriz = _matriz({"difusa": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]}, GRUPOS)
    _, manter, resultado = drop_by_group_missingness(matriz, threshold=0.8)
    assert not manter[0]
    assert resultado.discarded_by_reason[AUSENTE_EM_TODOS_OS_GRUPOS] == 1


def test_o_quantificador_e_em_todos_e_nao_em_algum() -> None:
    """**O teste que distingue as duas leituras, e é o defeito que o critério existe para pegar.**

    A feature está ausente em dois dos três grupos e presente no terceiro. Sob "em todos" ela
    **fica**; sob "em algum" ela sairia. Um fixture com um grupo só não separaria as duas
    leituras — o insumo tem de conter a diferença que o teste afirma.
    """
    matriz = _matriz({"dois_de_tres": [0.0, 0.0, 0.0, 10.0, 20.0, 30.0]}, GRUPOS)
    ausente = is_missing(np.asarray(matriz.values))
    grupos = np.asarray(matriz.groups)
    por_grupo = {g: float(ausente[grupos == g].mean()) for g in ("QC", "A", "B")}
    assert por_grupo["QC"] > 0.8 and por_grupo["A"] > 0.8, "o fixture tem de exceder em dois"
    assert por_grupo["B"] <= 0.8, "e NÃO exceder no terceiro — senão não separa as leituras"

    _, manter, _ = drop_by_group_missingness(matriz, threshold=0.8)
    assert manter[0], "sob 'em todos' ela fica; se saiu, o quantificador virou 'em algum'"


def test_o_limiar_e_estritamente_maior() -> None:
    """Fração **igual** ao limiar não excede. A borda é fixada em vez de implícita."""
    # Grupo "B" com três amostras: duas ausentes dá 2/3 = 0,667.
    matriz = _matriz({"na_borda": [0.0, 0.0, 0.0, 0.0, 0.0, 10.0]}, GRUPOS)
    _, manter_igual, _ = drop_by_group_missingness(matriz, threshold=2 / 3)
    _, manter_abaixo, _ = drop_by_group_missingness(matriz, threshold=0.5)
    assert manter_igual[0], "fração igual ao limiar NÃO excede"
    assert not manter_abaixo[0], "acima do limiar, exclui"


@pytest.mark.parametrize("limiar", [1.5, 80.0, 0.0, -0.1])
def test_limiar_fora_de_zero_um_levanta(limiar: float) -> None:
    """`80.0` é o limiar de 80% em percentual, e sob ele nada nunca excede."""
    matriz = _matriz({"a": [0.0] * 6}, GRUPOS)
    with pytest.raises(ValueError, match="FRAÇÃO"):
        drop_by_group_missingness(matriz, threshold=limiar)


def test_a_contagem_de_grupos_viaja_no_stage_result() -> None:
    matriz = _matriz({"a": [1.0] * 6}, GRUPOS)
    _, _, resultado = drop_by_group_missingness(matriz)
    assert resultado.labels["n_grupos"] == "3"


# ---------------------------------------------------------------------------
# Ausência: zero E não finito
# ---------------------------------------------------------------------------


def test_ausencia_cobre_zero_e_nao_finito() -> None:
    """Contar só uma das duas deixaria metade da ausência invisível ao limiar."""
    valores = np.array([[0.0, 1.0, np.nan, np.inf, -np.inf]])
    assert is_missing(valores).tolist() == [[True, False, True, True, True]]


# ---------------------------------------------------------------------------
# As cinco imputações
# ---------------------------------------------------------------------------


def _com_ausencia() -> RefinoMatrix:
    """Matriz com ausência em posições conhecidas e estrutura entre features.

    A estrutura importa: `knn` e `missforest` aprendem de correlação entre colunas, e sobre
    matriz sem estrutura eles convergem para o centro e ficam indistinguíveis da mediana — o
    que tornaria o teste da semente vácuo.
    """
    gerador = np.random.default_rng(11)
    base = gerador.normal(100.0, 10.0, size=(12, 6))
    base[:, 1] = base[:, 0] * 2.0 + gerador.normal(0.0, 1.0, size=12)
    base[:, 3] = base[:, 2] * -1.5 + gerador.normal(0.0, 1.0, size=12)
    base = np.abs(base) + 1.0
    base[2, 0] = 0.0
    base[5, 1] = 0.0
    base[7, 3] = 0.0
    grupos = ["QC"] + ["A"] * 5 + ["B"] * 6
    nomes = tuple(f"f{j}" for j in range(6))
    amostras = tuple(
        SampleMetadata(
            name=f"S{i}",
            injection_order=i + 1,
            sample_type="qc" if i == 0 else "sample",
            group=g,
        )
        for i, g in enumerate(grupos)
    )
    return RefinoMatrix(values=base, feature_names=nomes, samples=amostras, mode="sintetico")


@pytest.mark.parametrize("metodo", METODOS_QUE_IMPUTAM)
def test_ca_5_8_imputacao_e_determinista_dada_a_semente(metodo: str) -> None:
    """Duas execuções com a mesma semente dão matriz bit-idêntica."""
    matriz = _com_ausencia()
    primeira = impute(matriz, metodo, seed=42)  # type: ignore[arg-type]
    segunda = impute(matriz, metodo, seed=42)  # type: ignore[arg-type]
    assert np.array_equal(np.asarray(primeira.values), np.asarray(segunda.values))


@pytest.mark.parametrize("metodo", CONSULTAM_A_SEMENTE)
def test_o_par_causal_da_semente_e_verificado_antes_de_afirmado(metodo: str) -> None:
    """**O par causal, com o fixture verificado quanto a separar.**

    Determinismo passaria com a semente **ignorada**. O que prova que ela é consultada é
    sementes diferentes darem resultados diferentes — e isso é propriedade da matriz: sobre
    matriz sem estrutura os dois métodos convergem para o centro e dão o mesmo com qualquer
    semente, e o par deixaria de ser par.

    O fixture tem duas colunas correlacionadas por construção, e a asserção sobre a
    correlação vem **antes** da asserção sobre a semente.
    """
    matriz = _com_ausencia()
    valores = np.asarray(matriz.values)
    # A correlação é medida sobre as linhas em que NENHUMA das duas colunas está ausente.
    # Medi-la sobre a matriz inteira dá 0,13 — os zeros injetados são dois pontos extremos
    # em doze linhas e colapsam o coeficiente. A estrutura que `knn` e `missforest`
    # exploram está entre os valores PRESENTES, e é ela que precisa existir.
    ausente = is_missing(valores)
    completas = ~(ausente[:, 0] | ausente[:, 1])
    correlacao = abs(float(np.corrcoef(valores[completas, 0], valores[completas, 1])[0, 1]))
    assert correlacao > 0.8, (
        f"o fixture tem correlação {correlacao:.2f} entre f0 e f1 nas linhas completas, e "
        "precisa de estrutura para que a semente possa mudar o resultado; sem isso este par "
        "causal não é causal (AR57)"
    )

    uma = impute(matriz, metodo, seed=42)  # type: ignore[arg-type]
    outra = impute(matriz, metodo, seed=1234)  # type: ignore[arg-type]
    assert not np.array_equal(np.asarray(uma.values), np.asarray(outra.values)), (
        f"{metodo}: sementes diferentes deram matriz idêntica; a semente não é consultada"
    )


def test_one_fifth_min_usa_um_quinto_do_menor_positivo() -> None:
    """O valor imputado é exactamente um quinto do menor positivo da feature."""
    matriz = _matriz({"f": [0.0, 50.0, 10.0, 20.0, 30.0, 40.0]}, GRUPOS)
    resultado = impute(matriz, "one_fifth_min")
    assert float(np.asarray(resultado.values)[0, 0]) == pytest.approx(10.0 / 5.0)


def test_median_usa_a_mediana_das_presentes() -> None:
    """A ausência não entra no cálculo da mediana — senão o zero a puxaria para baixo."""
    matriz = _matriz({"f": [0.0, 10.0, 20.0, 30.0, 40.0, 50.0]}, GRUPOS)
    resultado = impute(matriz, "median")
    assert float(np.asarray(resultado.values)[0, 0]) == pytest.approx(30.0)


@pytest.mark.parametrize("metodo", METODOS_QUE_IMPUTAM)
def test_nenhuma_ausencia_sobra_depois_de_imputar(metodo: str) -> None:
    """Os quatro que prometem imputar não deixam ausência — é o que os separa de `none`."""
    resultado = impute(_com_ausencia(), metodo, seed=42)  # type: ignore[arg-type]
    assert not is_missing(np.asarray(resultado.values)).any()


def test_none_preserva_a_ausencia_contada() -> None:
    """`none` é opção legítima, e o que ela deixa passar é ausência."""
    matriz = _com_ausencia()
    antes = int(is_missing(np.asarray(matriz.values)).sum())
    resultado = impute(matriz, "none")
    assert int(is_missing(np.asarray(resultado.values)).sum()) == antes
    assert antes > 0, "o fixture tem de ter ausência, senão o teste é vácuo"


def test_feature_inteiramente_ausente_nao_vira_nan_silencioso() -> None:
    """**O caso de borda que decide entre declarado e acidental.**

    Coluna sem valor presente nenhum não tem de onde derivar o preenchimento. `nan` ali faria
    a matriz sair da imputação **com ausência dentro**, que é o que `none` promete e os outros
    não — e o consumidor não teria como distinguir.
    """
    matriz = _matriz({"vazia": [0.0] * 6, "ok": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]}, GRUPOS)
    for metodo in ("one_fifth_min", "median"):
        resultado = impute(matriz, metodo)  # type: ignore[arg-type]
        coluna = np.asarray(resultado.values)[:, 0]
        assert np.isfinite(coluna).all(), f"{metodo} deixou não-finito numa coluna vazia"


def test_matriz_sem_ausencia_passa_intacta() -> None:
    """Sem nada a imputar, a matriz sai igual — e o `provenance` diz que nada foi feito."""
    matriz = _matriz({"f": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]}, GRUPOS)
    resultado = impute(matriz, "median")
    assert np.array_equal(np.asarray(resultado.values), np.asarray(matriz.values))
    assert "nada-ausente" in resultado.provenance[-1]


def test_metodo_invalido_levanta() -> None:
    """`Literal` é análise estática; a guarda em runtime é a fronteira.

    **O exemplo era `"mean"` até a paridade com o MetaboAnalyst, e ele passou a ser
    VÁLIDO.** O teste continuou verde por um instante porque `impute` terminava num `else`
    que caía no `missforest` — a guarda aceitava e a política era outra. Trocar o exemplo
    por um valor que nunca será vocabulário é o que impede o teste de caducar de novo pelo
    mesmo caminho.
    """
    with pytest.raises(InvalidLabelError, match="Imputation"):
        impute(_com_ausencia(), "media-aritmetica")  # type: ignore[arg-type]


def test_a_matriz_de_entrada_nao_e_alterada() -> None:
    """Regra 7 do README — a imutabilidade alcança o array."""
    matriz = _com_ausencia()
    antes = np.array(matriz.values, copy=True)
    impute(matriz, "one_fifth_min")
    assert np.array_equal(np.asarray(matriz.values), antes)


def test_knn_e_deterministico_independente_da_semente() -> None:
    """**A outra metade do achado: `knn` não consulta a semente, e isso é declarado.**

    `KNNImputer` não tem aleatoriedade — a vizinhança vem de distância e o valor é a média
    dos vizinhos, uma função fechada da matriz. Não aceita `random_state`.

    O critério pede "determinístico dada a semente", e o `knn` o satisfaz **trivialmente**:
    ele é determinístico sem semente. Afirmar isso explicitamente é o que impede a próxima
    pessoa de escrever o par causal da semente para ele e concluir que há defeito — foi
    exactamente o que aconteceu ao escrever este arquivo.
    """
    matriz = _com_ausencia()
    uma = impute(matriz, "knn", seed=42)
    outra = impute(matriz, "knn", seed=1234)
    assert np.array_equal(np.asarray(uma.values), np.asarray(outra.values)), (
        "o `knn` passou a depender da semente; ou o `sklearn` mudou, ou a implementação "
        "introduziu aleatoriedade que não estava lá"
    )


# ---------------------------------------------------------------------------
# Os três métodos que a paridade com o MetaboAnalyst acrescentou
# ---------------------------------------------------------------------------


def _uma_coluna_com_ausente() -> RefinoMatrix:
    """Coluna cujos presentes são 2, 4, 6, 8 e 10 e cuja sexta célula é ausente.

    Escolhida para que **média e mediana coincidam** seja falso: presentes 2,4,6,8,10 dão
    média 6 e mediana 6 — coincidem. Então a segunda coluna quebra a simetria de propósito,
    com presentes 1,1,1,1,10: média 2,8 e mediana 1. Sem ela o teste de `mean` passaria
    igualmente com a implementação de `median`, e não separaria nada.
    """
    return _matriz(
        {
            "simetrica": [2.0, 4.0, 6.0, 8.0, 10.0, 0.0],
            "assimetrica": [1.0, 1.0, 1.0, 1.0, 10.0, 0.0],
        },
        GRUPOS,
    )


def test_mean_usa_a_media_e_nao_a_mediana() -> None:
    """A planta cai sobre coluna em que as duas DIFEREM — senão o teste é vácuo.

    Na coluna simétrica média e mediana valem 6 e o teste passaria com qualquer das duas
    implementações; é a assimétrica que separa, com média 2,8 contra mediana 1.
    """
    saida = np.asarray(impute(_uma_coluna_com_ausente(), "mean").values)
    assert saida[5, 0] == pytest.approx(6.0)
    assert saida[5, 1] == pytest.approx(2.8)

    mediana = np.asarray(impute(_uma_coluna_com_ausente(), "median").values)
    assert mediana[5, 1] == pytest.approx(1.0)
    assert saida[5, 1] != pytest.approx(mediana[5, 1]), (
        "média e mediana deram o mesmo nesta coluna: o fixture não separa os dois métodos "
        "e o teste passaria com a implementação errada"
    )


def test_col_min_e_cinco_vezes_o_one_fifth_min() -> None:
    """Os dois medem a mesma coluna e diferem por um fator, e é isso que o nome promete.

    `ReplaceMin` do MetaboAnalyst é o nosso `one_fifth_min`; o `min` deles é o `col_min`.
    Um teste que só afirmasse "col_min é o menor positivo" não pegaria a troca dos dois.
    """
    matriz = _uma_coluna_com_ausente()
    menor = np.asarray(impute(matriz, "col_min").values)
    quinto = np.asarray(impute(matriz, "one_fifth_min").values)

    assert menor[5, 0] == pytest.approx(2.0)
    assert menor[5, 1] == pytest.approx(1.0)
    assert quinto[5, 0] == pytest.approx(2.0 / 5.0)
    assert quinto[5, 1] == pytest.approx(1.0 / 5.0)
    assert menor[5, 0] == pytest.approx(5.0 * quinto[5, 0])


def test_col_min_sem_positivo_na_coluna_da_zero() -> None:
    """A política de coluna degenerada é a mesma dos dois, porque o auxiliar é o mesmo."""
    matriz = _matriz({"tudo_zero": [0.0] * 6, "normal": [1.0, 2.0, 3.0, 4.0, 5.0, 0.0]}, GRUPOS)
    saida = np.asarray(impute(matriz, "col_min").values)
    assert np.all(np.isfinite(saida)), "coluna sem positivo produziu não-finito"
    assert saida[0, 0] == pytest.approx(0.0)


def test_exclude_remove_a_feature_com_qualquer_ausente() -> None:
    """É o único método que muda a FORMA, e o teste afirma as duas metades.

    A feature completa fica com os valores intactos; a incompleta some do conjunto de nomes.
    Afirmar só a contagem deixaria passar a remoção da coluna errada.
    """
    matriz = _matriz(
        {
            "completa": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "com_um_zero": [1.0, 2.0, 3.0, 4.0, 5.0, 0.0],
        },
        GRUPOS,
    )
    saida = impute(matriz, "exclude")
    assert saida.feature_names == ("completa",)
    assert saida.n_features == 1
    np.testing.assert_allclose(np.asarray(saida.values)[:, 0], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])


def test_exclude_nao_altera_a_matriz_de_entrada() -> None:
    """Regra 7 do README — e `exclude` é o caminho que constrói matriz nova por outro código."""
    matriz = _matriz({"a": [1.0, 2.0, 3.0, 4.0, 5.0, 0.0]}, GRUPOS)
    antes = np.array(matriz.values, copy=True)
    impute(matriz, "exclude")
    assert np.array_equal(np.asarray(matriz.values), antes)
    assert matriz.n_features == 1


def test_os_tres_metodos_novos_estao_no_vocabulario_e_tem_ramo() -> None:
    """Ancora contra o `else` que engolia valor desconhecido.

    Até a paridade, `impute` terminava num `else` que caía no `missforest`: um valor novo no
    `Literal` passava pela guarda e rodava OUTRA política, em silêncio. Este teste afirma que
    os novos produzem resultado DIFERENTE do `missforest` sobre a mesma matriz — que é o que
    o `else` não permitiria distinguir.

    **Compara a matriz INTEIRA e não uma célula, e a diferença é medida.** A primeira versão
    olhava `[5, 1]`, onde `missforest` devolve 1,0000 e `col_min` também — coincidência do
    fixture, não defeito de despacho. Uma âncora presa a uma célula afirma sobre a célula que
    o autor escolheu; sobre a matriz, ela afirma sobre o método.
    """
    matriz = _uma_coluna_com_ausente()
    floresta = np.asarray(impute(matriz, "missforest").values)
    for metodo in ("mean", "col_min"):
        obtido = np.asarray(impute(matriz, metodo).values)
        assert not np.allclose(obtido, floresta), (
            f"`{metodo}` produziu a MESMA matriz que `missforest`: o despacho pode estar a "
            "cair no ramo errado, que é exatamente o defeito que o `else` cego permitia"
        )
