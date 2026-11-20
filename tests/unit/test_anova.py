"""`one_way_anova` — a ANOVA de N grupos, os dois post-hoc, e a correção em duas etapas.

**A âncora deste arquivo é fechada e não vem da implementação**. Com
`df1 = 2` a sobrevivência de F tem forma analítica —

    P(F > f) = (1 + 2f/df2)^(-df2/2)

— então três grupos de três observações dão `F = 27` e `p = 10⁻³` **exactos**, calculáveis à mão
antes de existir função nenhuma. É o que separa *"a função devolve número"* de *"a função devolve
o número certo"*.

**A planta do Fisher é o item mais consequente**, e ela exige dado HETEROCEDÁSTICO: sobre grupos
de variância igual o quadrado médio agrupado e o quadrado médio por par **coincidem**, e o teste
seria vácuo. `test_a_planta_do_fisher_separa_as_duas_implementacoes` calcula as duas quantidades e
afirma que divergem **antes** de olhar o resultado da implementação.
"""

import ast
import itertools
import pathlib
import warnings
from dataclasses import fields
from typing import get_args

import numpy as np
import pytest
from scipy.stats import f_oneway, kruskal, mannwhitneyu, rankdata, tukey_hsd
from scipy.stats import norm as normal_padrao
from scipy.stats import t as t_de_student

from refino.domain.cancellation import ExecutionCancelledError
from refino.domain.labels import VOCABULARY, InvalidLabelError, PostHoc, validate_label
from refino.domain.matrix import RefinoMatrix
from refino.domain.metadata import SampleMetadata, SampleType
from refino.domain.statistics import DunnComparison, PostHocComparison, PostHocMethod
from refino.statistics.anova import FDR_ALPHA, kruskal_wallis, one_way_anova
from refino.statistics.univariate import MIN_AMOSTRAS_POR_GRUPO, benjamini_hochberg

# ---------------------------------------------------------------------------
# Construção de matriz — as três colunas que todo cenário varia
# ---------------------------------------------------------------------------


def _matriz(
    valores: np.ndarray | list[list[float]],
    grupos: list[str],
    tipos: list[SampleType] | None = None,
) -> RefinoMatrix:
    """Matriz sintética com um rótulo clínico e um `sample_type` por linha.

    `tipos` existe para os cenários de população: uma amostra `validate` com rótulo de grupo é a
    planta de que a ANOVA deriva a população da máscara e não do rótulo.
    """
    arranjo = np.asarray(valores, dtype=np.float64)
    reais: list[SampleType] = tipos if tipos is not None else ["sample"] * len(grupos)
    amostras = tuple(
        SampleMetadata(name=f"S{i}", injection_order=i + 1, sample_type=tipo, group=grupo)
        for i, (grupo, tipo) in enumerate(zip(grupos, reais, strict=True))
    )
    return RefinoMatrix(
        values=arranjo,
        feature_names=tuple(f"f{j}" for j in range(arranjo.shape[1])),
        samples=amostras,
        mode="sintetico",
    )


def _coluna(*blocos: list[float]) -> tuple[np.ndarray, list[str]]:
    """Uma feature só, com um bloco de valores por grupo — `A`, `B`, `C`, ... nessa ordem."""
    rotulos = [chr(ord("A") + i) for i, bloco in enumerate(blocos) for _ in bloco]
    valores = np.array([[v] for bloco in blocos for v in bloco], dtype=np.float64)
    return valores, rotulos


# Os três blocos da âncora fechada, e os da planta heterocedástica. Ficam no topo porque são
# citados por mais de um cenário, e repeti-los faria duas fontes do mesmo número.
ANCORA_A = [1.0, 2.0, 3.0]
ANCORA_B = [4.0, 5.0, 6.0]
ANCORA_C = [7.0, 8.0, 9.0]

HETERO_A = [1.0, 2.0, 3.0]  # média 2, soma de quadrados 2
HETERO_B = [4.0, 5.0, 6.0]  # média 5, soma de quadrados 2
# média 50, soma de quadrados 200 — é o que quebra a homocedasticidade
HETERO_C = [40.0, 50.0, 60.0]


# ---------------------------------------------------------------------------
# A âncora à mão — corre antes de qualquer leitura de resultado
# ---------------------------------------------------------------------------


def test_a_ancora_a_mao_de_tres_grupos_com_verdade_conhecida() -> None:
    """`F = 27` e `p = 10⁻³` EXACTOS, obtidos sem a implementação.

    Com `A = [1,2,3]`, `B = [4,5,6]`, `C = [7,8,9]`: médias 2, 5 e 8, média geral 5.

    * entre grupos: `3·((2-5)² + 0 + (8-5)²) = 54`, com 2 graus → quadrado médio 27
    * dentro:       `2 + 2 + 2 = 6`, com 6 graus → quadrado médio 1
    * `F = 27/1 = 27`

    E com `df1 = 2` a sobrevivência fecha: `(1 + 2·27/6)^(-3) = 10^(-3)`.
    """
    valores, grupos = _coluna(ANCORA_A, ANCORA_B, ANCORA_C)
    resultado = one_way_anova(_matriz(valores, grupos))

    assert resultado.statistic[0] == pytest.approx(27.0, abs=1e-12)
    assert resultado.pvalue[0] == pytest.approx(1e-3, abs=1e-12)
    assert resultado.groups == ("A", "B", "C")
    assert resultado.n_per_group == (3, 3, 3)


def test_a_segunda_ancora_fechada_com_grupos_de_variancia_diferente() -> None:
    """`F = 49/34` e `p = (102/151)³`, também fechados — a planta do Fisher tem F conferido.

    `A=[1,2,3]`, `B=[4,5,6]`, `C=[0,10,20]`: dentro dá `2+2+200 = 204` com 6 graus (34), entre dá
    `98` com 2 graus (49). `F = 49/34`, e `(1 + 2F/6)^(-3) = (102/151)³`.
    """
    valores, grupos = _coluna([1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [0.0, 10.0, 20.0])
    resultado = one_way_anova(_matriz(valores, grupos))

    assert resultado.statistic[0] == pytest.approx(49.0 / 34.0, rel=1e-12)
    assert resultado.pvalue[0] == pytest.approx((102.0 / 151.0) ** 3, rel=1e-12)


# ---------------------------------------------------------------------------
# Os desfechos da ANOVA
# ---------------------------------------------------------------------------


def test_tres_grupos_com_medias_iguais_nao_produzem_sobrevivente() -> None:
    """F próximo de zero, p alto, e a rota do post-hoc diz `no_survivors` — não `nothing_testable`.

    A distinção é o motivo de a rota ter três estados: *"medi e nada sobreviveu"* não
    é *"não havia o que medir"*, e as duas produzem figura vazia.
    """
    gerador = np.random.default_rng(20260913)
    valores = gerador.normal(10.0, 1.0, size=(18, 6))
    grupos = ["A"] * 6 + ["B"] * 6 + ["C"] * 6
    resultado = one_way_anova(_matriz(valores, grupos))

    assert np.all(resultado.pvalue > 0.05), f"p-valores inesperadamente baixos: {resultado.pvalue}"
    assert resultado.counts.survivors == 0
    assert resultado.post_hoc.route == "no_survivors"
    assert resultado.comparisons == ()


def test_um_grupo_deslocado_da_f_alto_so_na_deslocada_e_o_post_hoc_aponta_o_par() -> None:
    """O par causal do efeito: a feature deslocada sobrevive, as outras não, e o par certo sai.

    Com `C` deslocado, as comparações `A-C` e `B-C` são as significativas e `A-B` não. Sem esta
    asserção, um post-hoc que devolvesse todos os pares seria indistinguível de um que acertasse.
    """
    gerador = np.random.default_rng(7)
    valores = gerador.normal(10.0, 0.5, size=(18, 4))
    valores[12:, 0] += 8.0
    grupos = ["A"] * 6 + ["B"] * 6 + ["C"] * 6
    resultado = one_way_anova(_matriz(valores, grupos))

    assert resultado.counts.survivors == 1
    assert bool(resultado.survived[0]) is True
    assert not resultado.survived[1:].any()
    assert resultado.post_hoc.route == "ran"

    por_par = {(c.group_a, c.group_b): c for c in resultado.comparisons}
    assert set(por_par) == {("A", "B"), ("A", "C"), ("B", "C")}
    assert all(c.feature == "f0" for c in resultado.comparisons)
    assert por_par[("A", "C")].pvalue < 0.05
    assert por_par[("B", "C")].pvalue < 0.05
    assert por_par[("A", "B")].pvalue > 0.05
    # A diferença é `média de a` menos `média de b`, e `C` está acima — o sinal tem de o dizer.
    assert por_par[("A", "C")].difference < 0.0


def test_o_parametro_post_hoc_e_consultado() -> None:
    """Par causal do rótulo: Tukey e Fisher dão p diferentes sobre a mesma entrada.

    Sem esta asserção o parâmetro poderia ser ignorado e a suíte inteira passaria.
    """
    valores, grupos = _coluna(HETERO_A, HETERO_B, HETERO_C)
    matriz = _matriz(valores, grupos)

    com_tukey = one_way_anova(matriz, post_hoc="tukey")
    com_fisher = one_way_anova(matriz, post_hoc="fisher")

    assert com_tukey.post_hoc.method == "tukey"
    assert com_fisher.post_hoc.method == "fisher"
    p_tukey = [c.pvalue for c in com_tukey.comparisons]
    p_fisher = [c.pvalue for c in com_fisher.comparisons]
    assert len(p_tukey) == len(p_fisher) == 3
    assert not np.allclose(p_tukey, p_fisher), (
        "Tukey e Fisher deram os mesmos p; então o rótulo não está a ser consultado, ou a "
        "entrada não os distingue"
    )
    # A ANOVA não depende do post-hoc: trocar o rótulo não pode mover F, p nem q.
    assert np.array_equal(com_tukey.statistic, com_fisher.statistic)
    assert np.array_equal(com_tukey.qvalue_fdr, com_fisher.qvalue_fdr)


def test_o_rotulo_do_post_hoc_e_validado_na_fronteira() -> None:
    """O modo de falha insidioso é o `else` razoável, e o código tem um.

    A bifurcação é `if post_hoc == "tukey": ... else: fisher`. Sem a guarda, `"bonferroni"` correria
    Fisher em silêncio e o resultado declararia o nome pedido.
    """
    valores, grupos = _coluna(ANCORA_A, ANCORA_B, ANCORA_C)
    with pytest.raises(InvalidLabelError, match="PostHoc"):
        one_way_anova(_matriz(valores, grupos), post_hoc="bonferroni")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# A planta que separa as duas implementações de Fisher
# ---------------------------------------------------------------------------


def _fisher_agrupado(a: np.ndarray, b: np.ndarray, todos: list[np.ndarray]) -> tuple[float, float]:
    """`t` e `p` com o quadrado médio residual da ANOVA INTEIRA — a implementação certa."""
    soma_quadrados = float(sum(((g - g.mean()) ** 2).sum() for g in todos))
    graus = int(sum(g.size for g in todos) - len(todos))
    quadrado_medio = soma_quadrados / graus
    erro = float(np.sqrt(quadrado_medio * (1.0 / a.size + 1.0 / b.size)))
    t = float(a.mean() - b.mean()) / erro
    return t, float(2.0 * t_de_student.sf(abs(t), graus))


def _fisher_por_par(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """`t` e `p` com a variância recalculada SÓ do par — a implementação ingênua."""
    graus = a.size + b.size - 2
    variancia = (((a - a.mean()) ** 2).sum() + ((b - b.mean()) ** 2).sum()) / graus
    erro = float(np.sqrt(variancia * (1.0 / a.size + 1.0 / b.size)))
    t = float(a.mean() - b.mean()) / erro
    return t, float(2.0 * t_de_student.sf(abs(t), graus))


def test_sobre_dado_homocedastico_o_quadrado_medio_das_duas_implementacoes_coincide() -> None:
    """A prova de que a planta precisa de heterocedasticidade — senão ela não separa o defeito.

    Com `A`, `B` e `C` de variância idêntica o quadrado médio agrupado **é** o do par — vale 1 nos
    dois — e o `t` sai igual ao último dígito. Um teste que afirmasse o `t` sobre este dado passaria
    com a implementação errada.

    **O `p` ainda difere aqui, e a causa é OUTRA: os graus de liberdade**, 6 contra 4. Isto foi
    medido ao escrever o cenário e contradiz a primeira redação dele, que dizia que as duas
    coincidiam. A asserção abaixo isola a causa: com o `t` do par avaliado a 6 graus, os dois `p`
    coincidem ao último dígito — logo a variância é a mesma e só a contagem de graus difere.

    E o que fica é o argumento da planta: **sobre este dado o veredito a 5% é o MESMO**, então o
    defeito do quadrado médio por par não é detectável aqui. Só a heterocedasticidade o separa.
    """
    a, b, c = (np.array(bloco) for bloco in (ANCORA_A, ANCORA_B, ANCORA_C))
    t_agrupado, p_agrupado = _fisher_agrupado(a, b, [a, b, c])
    t_por_par, p_por_par = _fisher_por_par(a, b)

    assert t_agrupado == pytest.approx(t_por_par, rel=1e-12)
    assert p_agrupado != pytest.approx(p_por_par, rel=1e-3)
    assert 2.0 * t_de_student.sf(abs(t_por_par), 6) == pytest.approx(p_agrupado, rel=1e-12)
    assert (p_agrupado < 0.05) == (p_por_par < 0.05)


def test_a_planta_do_fisher_separa_as_duas_implementacoes() -> None:
    """Fisher usa o quadrado médio AGRUPADO da ANOVA inteira, não o recalculado por par.

    A ordem é o conteúdo: as duas quantidades são calculadas e a divergência é AFIRMADA antes de
    o resultado da implementação ser lido. Com `C` de variância cem vezes maior, o agrupado vale
    34 e o do par vale 1 — e os dois dão vereditos opostos a 5%.
    """
    a, b, c = (np.array(bloco) for bloco in (HETERO_A, HETERO_B, HETERO_C))
    t_agrupado, p_agrupado = _fisher_agrupado(a, b, [a, b, c])
    t_por_par, p_por_par = _fisher_por_par(a, b)

    # A planta cai sobre dado em que as duas coisas DIFEREM — afirmado, não suposto.
    assert abs(t_agrupado) < abs(t_por_par) / 4.0, (
        f"as duas implementações não divergem neste dado (t {t_agrupado} contra {t_por_par}); "
        "a planta seria vácua"
    )
    assert p_agrupado > 0.05 > p_por_par, (
        f"os vereditos coincidem (p {p_agrupado} contra {p_por_par}); a planta seria fraca"
    )

    valores, grupos = _coluna(HETERO_A, HETERO_B, HETERO_C)
    resultado = one_way_anova(_matriz(valores, grupos), post_hoc="fisher")
    assert resultado.counts.survivors == 1, "a feature tinha de sobreviver para haver post-hoc"

    par = next(c for c in resultado.comparisons if (c.group_a, c.group_b) == ("A", "B"))
    assert par.pvalue == pytest.approx(p_agrupado, rel=1e-12)
    assert par.pvalue != pytest.approx(p_por_par, rel=1e-3)
    assert par.difference == pytest.approx(a.mean() - b.mean(), rel=1e-12)


def test_o_tukey_e_o_do_intervalo_estudentizado_do_scipy() -> None:
    """O oráculo do Tukey é `scipy.stats.tukey_hsd` sobre os mesmos blocos, par a par."""
    valores, grupos = _coluna(HETERO_A, HETERO_B, HETERO_C)
    resultado = one_way_anova(_matriz(valores, grupos), post_hoc="tukey")

    blocos = [np.array(b) for b in (HETERO_A, HETERO_B, HETERO_C)]
    oraculo = tukey_hsd(*blocos)
    indices = {"A": 0, "B": 1, "C": 2}
    for comparacao in resultado.comparisons:
        i, j = indices[comparacao.group_a], indices[comparacao.group_b]
        assert comparacao.pvalue == pytest.approx(float(oraculo.pvalue[i][j]), rel=1e-12)
        assert comparacao.difference == pytest.approx(float(oraculo.statistic[i][j]), rel=1e-12)


# ---------------------------------------------------------------------------
# A correção em duas etapas
# ---------------------------------------------------------------------------


def test_a_correcao_de_multiplos_testes_nao_corre_duas_vezes() -> None:
    """O `q` da ANOVA e o `p` do post-hoc são grandezas distintas, e a segunda é CRUA.

    O par: os p do post-hoc batem com o t agrupado calculado aqui, e **não** batem com o que
    sairia de lhes aplicar Benjamini-Hochberg. Sem a segunda metade, uma implementação que
    ajustasse duas vezes passaria.
    """
    gerador = np.random.default_rng(4242)
    valores = gerador.normal(10.0, 0.4, size=(18, 3))
    valores[12:, 0] += 6.0
    valores[12:, 1] += 5.0
    grupos = ["A"] * 6 + ["B"] * 6 + ["C"] * 6
    resultado = one_way_anova(_matriz(valores, grupos), post_hoc="fisher")

    assert resultado.counts.survivors >= 2, "o cenário precisa de mais de uma sobrevivente"
    crus = np.array([c.pvalue for c in resultado.comparisons], dtype=np.float64)
    assert crus.size >= 6

    ajustados = benjamini_hochberg(crus)
    assert not np.allclose(crus, ajustados), (
        "os p do post-hoc coincidem com a versão ajustada por BH; o FDR pode estar a correr duas "
        "vezes, ou o cenário não os distingue"
    )

    # E a âncora positiva: cada p do post-hoc é o t agrupado da feature, cru.
    colunas = {nome: i for i, nome in enumerate(resultado.feature)}
    blocos = {"A": slice(0, 6), "B": slice(6, 12), "C": slice(12, 18)}
    for comparacao in resultado.comparisons:
        j = colunas[comparacao.feature]
        todos = [valores[fatia, j] for fatia in blocos.values()]
        _, esperado = _fisher_agrupado(
            valores[blocos[comparacao.group_a], j], valores[blocos[comparacao.group_b], j], todos
        )
        assert comparacao.pvalue == pytest.approx(esperado, rel=1e-12)


def test_o_post_hoc_so_corre_nas_features_que_sobreviveram_ao_fdr() -> None:
    """O filtro é a primeira etapa, e o post-hoc a segunda — nunca sobre todas as features."""
    gerador = np.random.default_rng(99)
    valores = gerador.normal(10.0, 0.4, size=(18, 5))
    valores[12:, 2] += 7.0
    grupos = ["A"] * 6 + ["B"] * 6 + ["C"] * 6
    resultado = one_way_anova(_matriz(valores, grupos))

    sobreviventes = {
        nome for nome, viva in zip(resultado.feature, resultado.survived, strict=True) if viva
    }
    assert sobreviventes == {"f2"}
    assert {c.feature for c in resultado.comparisons} == sobreviventes
    assert resultado.counts.survivors == len(sobreviventes)


def test_o_q_e_o_de_benjamini_hochberg_sobre_o_vetor_inteiro_de_p() -> None:
    """A primeira etapa corre sobre TODAS as features, não sobre um subconjunto pré-filtrado."""
    gerador = np.random.default_rng(31)
    valores = gerador.normal(10.0, 1.0, size=(18, 9))
    valores[12:, 0] += 6.0
    grupos = ["A"] * 6 + ["B"] * 6 + ["C"] * 6
    resultado = one_way_anova(_matriz(valores, grupos))

    esperado = benjamini_hochberg(resultado.pvalue)
    assert np.allclose(resultado.qvalue_fdr, esperado, equal_nan=True)


# ---------------------------------------------------------------------------
# Recusas e indefinidos — nada descartado em silêncio
# ---------------------------------------------------------------------------


def test_grupo_com_uma_amostra_so_e_recusado_nomeando_o_grupo_e_a_contagem() -> None:
    valores, grupos = _coluna(ANCORA_A, ANCORA_B, [7.0])
    with pytest.raises(ValueError) as erro:
        one_way_anova(_matriz(valores, grupos))
    mensagem = str(erro.value)
    assert "'C'" in mensagem
    assert "1" in mensagem


def test_corte_de_fdr_fora_da_faixa_e_recusado() -> None:
    """A guarda do alfa tem as duas direções — e zero é a que produz figura vazia por construção."""
    valores, grupos = _coluna(ANCORA_A, ANCORA_B, ANCORA_C)
    matriz = _matriz(valores, grupos)
    for invalido in (0.0, -0.1, 1.5):
        with pytest.raises(ValueError, match="fdr_alpha"):
            one_way_anova(matriz, fdr_alpha=invalido)
    # 1,0 é o limite superior ACEITE: todo `q` definido sobrevive, e isso é pedido legítimo.
    assert one_way_anova(matriz, fdr_alpha=1.0).counts.survivors == 1


def test_menos_de_dois_grupos_e_recusado_nomeando_o_que_ha() -> None:
    valores, grupos = _coluna(ANCORA_A + ANCORA_B)
    with pytest.raises(ValueError) as erro:
        one_way_anova(_matriz(valores, grupos))
    mensagem = str(erro.value)
    assert "ao menos 2 grupos" in mensagem
    # Nomear o que HÁ é a metade acionável: quem tem quatro rótulos e vê um só precisa de saber
    # que a máscara comeu os outros três.
    assert "['A']" in mensagem


def test_feature_toda_constante_tem_f_indefinido_contado_e_declarado() -> None:
    """`0/0` é ausência de medida, não F baixo — e ela é CONTADA, nunca filtrada."""
    valores = np.array([[5.0, v] for v in [1, 2, 3, 4, 5, 6, 7, 8, 9]], dtype=np.float64)
    grupos = ["A"] * 3 + ["B"] * 3 + ["C"] * 3
    resultado = one_way_anova(_matriz(valores, grupos))

    assert np.isnan(resultado.statistic[0])
    assert np.isnan(resultado.pvalue[0])
    assert np.isnan(resultado.qvalue_fdr[0])
    assert resultado.counts.constant == 1
    assert resultado.counts.insufficient_groups == 0
    assert resultado.counts.statistic_undefined == 1
    assert resultado.counts.n_features == 2
    # A feature constante continua na tabela, na posição dela.
    assert resultado.feature == ("f0", "f1")


def test_feature_constante_dentro_de_cada_grupo_e_p_zero_exacto_e_nao_indefinido() -> None:
    """`F = +inf` e `p = 0` são medida EXACTA, e a contagem os separa do indefinido.

    É o irmão de `pvalue_zero` em `UnivariateCounts`: `0` entra no ajuste de BH e conta no `n`
    dele. Chamar-lhe indefinido apagaria a distinção.

    **E o `q = 0` que daqui sai é o SOBREVIVENTE MAIS FORTE que existe — a asserção sobre
    `survivors` é a âncora do lado do núcleo (17/09/2026).** O que a `-log10` da figura não desenha
    é o **ponto**, e até esta data a figura perdia também a **contagem**: `plots/anova.py` derivava
    `sobreviventes` das features desenháveis e publicava `0` enquanto este campo dizia `1`. A
    separação entre classificar e desenhar vive lá, e
    `test_a_contagem_da_figura_BATE_com_AnovaCounts_survivors_do_nucleo` cruza as duas pontas.
    """
    valores, grupos = _coluna([1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [9.0, 9.0, 9.0])
    resultado = one_way_anova(_matriz(valores, grupos))

    assert np.isposinf(resultado.statistic[0])
    assert resultado.pvalue[0] == 0.0
    assert resultado.qvalue_fdr[0] == 0.0, "o BH leva o `p = 0` a `q = 0`, e é isso que sobrevive"
    assert resultado.counts.pvalue_zero == 1
    assert resultado.counts.survivors == 1, (
        "`q = 0` passa qualquer corte: contá-lo é o que separa separação exacta de ausência de "
        f"medida — {resultado.counts}"
    )
    assert resultado.counts.statistic_undefined == 0
    assert resultado.counts.constant == 0

    # **E o post-hoc CORRE sobre ela — `q = 0` sobrevive a qualquer corte.** É o ramo degenerado,
    # em que o quadrado médio residual é zero: `tukey_hsd` dividiria por zero ali. O desfecho vai
    # afirmado nos dois sentidos, porque é a diferença entre separação exacta e ausência de medida.
    por_par = {(c.group_a, c.group_b): c.pvalue for c in resultado.comparisons}
    assert por_par[("A", "C")] == 0.0
    assert por_par[("B", "C")] == 0.0
    assert np.isnan(por_par[("A", "B")]), (
        "`A` e `B` são idênticos e a variância residual é zero: é `0/0`, e não separação com p=0"
    )


def test_grupo_vazio_por_feature_nao_e_descartado_em_silencio() -> None:
    """A ANOVA corre por feature, e um grupo indefinido numa coluna muda a população DELA.

    A feature não é testada — a F seria sobre uma população diferente da que o cabeçalho declara,
    rótulo certo sobre a quantidade errada — e a exclusão fica contada e localizável por
    `n_usable_per_group`.
    """
    valores = np.array(
        [
            [1.0, 1.0],
            [2.0, 2.0],
            [3.0, 3.0],
            [4.0, 4.0],
            [5.0, 5.0],
            [6.0, 6.0],
            [np.nan, 7.0],
            [np.nan, 8.0],
            [np.nan, 9.0],
        ],
        dtype=np.float64,
    )
    grupos = ["A"] * 3 + ["B"] * 3 + ["C"] * 3
    resultado = one_way_anova(_matriz(valores, grupos))

    assert np.isnan(resultado.statistic[0])
    assert resultado.counts.insufficient_groups == 1
    assert resultado.counts.constant == 0
    assert list(resultado.n_groups_used) == [2, 3]
    assert list(resultado.n_usable_per_group[0]) == [3, 3, 0]
    assert list(resultado.n_usable_per_group[1]) == [3, 3, 3]
    # A outra coluna continua a ser a âncora fechada — a exclusão é por feature e não global.
    assert resultado.statistic[1] == pytest.approx(27.0, abs=1e-12)


def test_grupo_com_uma_observacao_util_numa_feature_conta_como_insuficiente() -> None:
    """O limiar por feature é o MESMO do global: dois por grupo. Uma observação não é contraste."""
    valores = np.array(
        [[1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0], [np.nan], [np.nan]], dtype=np.float64
    )
    grupos = ["A"] * 3 + ["B"] * 3 + ["C"] * 3
    resultado = one_way_anova(_matriz(valores, grupos))

    assert np.isnan(resultado.statistic[0])
    assert resultado.counts.insufficient_groups == 1
    assert list(resultado.n_usable_per_group[0]) == [3, 3, 1]


def test_a_particao_do_indefinido_e_exaustiva() -> None:
    """`statistic_undefined == constant + insufficient_groups` — o que falta de uma partição vira
    residual, e residual lê-se como pronto."""
    valores = np.array(
        [
            [5.0, 1.0, 1.0],
            [5.0, 2.0, 2.0],
            [5.0, 3.0, 3.0],
            [5.0, 4.0, 4.0],
            [5.0, 5.0, 5.0],
            [5.0, 6.0, 6.0],
            [5.0, np.nan, 7.0],
            [5.0, np.nan, 8.0],
            [5.0, np.nan, 9.0],
        ],
        dtype=np.float64,
    )
    grupos = ["A"] * 3 + ["B"] * 3 + ["C"] * 3
    contagens = one_way_anova(_matriz(valores, grupos)).counts

    assert contagens.constant == 1
    assert contagens.insufficient_groups == 1
    assert contagens.statistic_undefined == contagens.constant + contagens.insufficient_groups
    assert contagens.pvalue_undefined == contagens.statistic_undefined
    assert contagens.qvalue_undefined == contagens.pvalue_undefined


def test_quando_nenhuma_feature_e_testavel_a_rota_diz_nothing_testable() -> None:
    """*"Não havia o que medir"* não é *"medi e nada sobreviveu"*."""
    valores = np.full((9, 2), 5.0)
    grupos = ["A"] * 3 + ["B"] * 3 + ["C"] * 3
    resultado = one_way_anova(_matriz(valores, grupos))

    assert resultado.post_hoc.route == "nothing_testable"
    assert resultado.post_hoc.method == "tukey"
    assert resultado.comparisons == ()
    assert resultado.counts.survivors == 0


# ---------------------------------------------------------------------------
# A população — máscara e não rótulo
# ---------------------------------------------------------------------------


def test_amostra_de_validacao_com_rotulo_de_grupo_nao_entra_na_populacao() -> None:
    """A população é `sample_mask`, e não «tudo o que não é QC».

    `grupos_contrastaveis` em `api/rotas.py` filtra **apenas** `qc_mask` — conferido na fonte em
    13/09/2026 —, então uma amostra `validate` com rótulo clínico entraria por ali. Isto é
    invariante de DESENHO e não observação sobre o corpus: decorre de `SampleType` ter
    três valores, e é por isso que vive no teste e não numa contagem.
    """
    valores_puros, grupos_puros = _coluna(ANCORA_A, ANCORA_B, ANCORA_C)
    limpa = one_way_anova(_matriz(valores_puros, grupos_puros))

    # As mesmas nove linhas, mais uma `validate` no grupo `A` e um QC — as duas com valor extremo.
    valores = np.vstack([valores_puros, [[1000.0]], [[-1000.0]]])
    grupos = [*grupos_puros, "A", "QC"]
    tipos: list[SampleType] = [*(["sample"] * 9), "validate", "qc"]
    suja = one_way_anova(_matriz(valores, grupos, tipos))

    assert suja.groups == ("A", "B", "C")
    assert suja.n_per_group == (3, 3, 3)
    assert suja.statistic[0] == pytest.approx(limpa.statistic[0], rel=1e-12)
    assert suja.pvalue[0] == pytest.approx(limpa.pvalue[0], rel=1e-12)
    assert list(suja.n_usable_per_group[0]) == [3, 3, 3]


def test_a_populacao_declarada_bate_com_a_contagem_por_feature_quando_nao_ha_indefinido() -> None:
    """`n_per_group` é o cabeçalho e `n_usable_per_group` é o que a feature usou. Sem `nan` os dois
    coincidem — e é a divergência entre eles que o cenário do grupo vazio existe para mostrar."""
    valores, grupos = _coluna(ANCORA_A, ANCORA_B, ANCORA_C)
    resultado = one_way_anova(_matriz(valores, grupos))
    assert list(resultado.n_usable_per_group[0]) == list(resultado.n_per_group)


def test_a_procedencia_da_matriz_viaja_com_os_numeros() -> None:
    valores, grupos = _coluna(ANCORA_A, ANCORA_B, ANCORA_C)
    matriz = _matriz(valores, grupos).with_provenance(
        "teste: procedência inventada não, esta é real"
    )
    resultado = one_way_anova(matriz)
    assert resultado.provenance == matriz.provenance


# ---------------------------------------------------------------------------
# Cancelamento e determinismo
# ---------------------------------------------------------------------------


def _matriz_com_duas_sobreviventes() -> RefinoMatrix:
    gerador = np.random.default_rng(1234)
    valores = gerador.normal(10.0, 0.4, size=(18, 3))
    valores[12:, 0] += 6.0
    valores[12:, 1] += 5.0
    return _matriz(valores, ["A"] * 6 + ["B"] * 6 + ["C"] * 6)


def test_cancelamento_no_meio_do_post_hoc_levanta_com_o_rotulo_do_ponto() -> None:
    """O laço do post-hoc é o ponto de parada — a ANOVA é vetorizada e não tem laço.

    A bandeira só diz «sim» na SEGUNDA pergunta: cancelar na primeira não provaria que a checagem
    está dentro do laço, apenas que existe antes dele.
    """
    matriz = _matriz_com_duas_sobreviventes()
    completo = one_way_anova(matriz)
    assert completo.counts.survivors >= 2

    chamadas = {"n": 0}

    def bandeira() -> bool:
        chamadas["n"] += 1
        return chamadas["n"] >= 2

    with pytest.raises(ExecutionCancelledError) as erro:
        one_way_anova(matriz, cancelled=bandeira)
    assert "anova" in erro.value.where
    assert chamadas["n"] >= 2, "a checagem não correu duas vezes; não está dentro do laço"


def test_sem_cancelamento_o_resultado_e_identico() -> None:
    """Âncora de que o parâmetro novo não moveu número nenhum."""
    matriz = _matriz_com_duas_sobreviventes()
    sem = one_way_anova(matriz)
    com = one_way_anova(matriz, cancelled=lambda: False)

    assert np.array_equal(sem.statistic, com.statistic, equal_nan=True)
    assert np.array_equal(sem.qvalue_fdr, com.qvalue_fdr, equal_nan=True)
    assert sem.comparisons == com.comparisons
    assert sem.counts == com.counts


def test_determinismo() -> None:
    """Mesma entrada, saída idêntica — regra 3 do README. Não há estocasticidade neste caminho."""
    matriz = _matriz_com_duas_sobreviventes()
    primeira = one_way_anova(matriz, post_hoc="fisher")
    segunda = one_way_anova(matriz, post_hoc="fisher")

    assert np.array_equal(primeira.statistic, segunda.statistic, equal_nan=True)
    assert np.array_equal(primeira.pvalue, segunda.pvalue, equal_nan=True)
    assert np.array_equal(primeira.qvalue_fdr, segunda.qvalue_fdr, equal_nan=True)
    assert primeira.comparisons == segunda.comparisons


# ---------------------------------------------------------------------------
# O corte, e o oráculo do próprio F
# ---------------------------------------------------------------------------


def test_o_corte_do_fdr_e_consultado() -> None:
    """Par causal do alfa: com o corte em zero nada sobrevive, com um corte largo sobrevive."""
    gerador = np.random.default_rng(555)
    valores = gerador.normal(10.0, 1.0, size=(18, 4))
    valores[12:, 0] += 3.0
    matriz = _matriz(valores, ["A"] * 6 + ["B"] * 6 + ["C"] * 6)

    apertado = one_way_anova(matriz, fdr_alpha=1e-12)
    largo = one_way_anova(matriz, fdr_alpha=0.5)

    assert apertado.counts.survivors < largo.counts.survivors
    assert apertado.fdr_alpha == 1e-12
    assert largo.fdr_alpha == 0.5
    assert one_way_anova(matriz).fdr_alpha == FDR_ALPHA


def test_o_f_bate_com_o_oraculo_do_scipy_feature_a_feature() -> None:
    """A contagem de acertos classifica o defeito, então a asserção é sobre ela.

    A população é as doze features, e o número que se lê primeiro é `12 de 12` — não o maior erro.
    """
    gerador = np.random.default_rng(2026)
    valores = gerador.normal(10.0, 1.5, size=(21, 12))
    valores[14:, :3] += 4.0
    grupos = ["A"] * 7 + ["B"] * 7 + ["C"] * 7
    resultado = one_way_anova(_matriz(valores, grupos))

    esperado = f_oneway(valores[:7], valores[7:14], valores[14:], axis=0)
    acertos = int(np.isclose(resultado.statistic, esperado.statistic, rtol=1e-12).sum())
    assert acertos == 12, f"{acertos} de 12 features batem com o oráculo"
    assert np.allclose(resultado.pvalue, esperado.pvalue, rtol=1e-12)


# ---------------------------------------------------------------------------
# As duas fontes que não podem discordar
# ---------------------------------------------------------------------------


def test_o_minimo_por_grupo_e_o_mesmo_do_teste_pareado() -> None:
    """Uma fonte e não duas, mecanizada.

    `AnovaResult.min_per_group` é campo e não constante de domínio exactamente para que o valor
    entre de `statistics/univariate.py`, que já o declara com o argumento. Se alguém acrescentar
    uma segunda constante com o mesmo papel, é aqui que a divergência aparece.
    """
    valores, grupos = _coluna(ANCORA_A, ANCORA_B, ANCORA_C)
    assert one_way_anova(_matriz(valores, grupos)).min_per_group == MIN_AMOSTRAS_POR_GRUPO


def test_a_contagem_de_sobreviventes_bate_com_a_mascara_derivada() -> None:
    """`counts.survivors` é a contagem de `survived`, e a docstring do domínio promete-o."""
    resultado = one_way_anova(_matriz_com_duas_sobreviventes())
    assert resultado.counts.survivors == int(resultado.survived.sum())
    assert resultado.counts.survivors == len({c.feature for c in resultado.comparisons})


def test_n_groups_used_e_derivado_de_n_usable_per_group() -> None:
    """A propriedade não é uma segunda medição: é `n_usable_per_group` lido com o limiar."""
    valores = np.array(
        [[1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [np.nan], [np.nan], [np.nan]], dtype=np.float64
    )
    resultado = one_way_anova(_matriz(valores, ["A"] * 3 + ["B"] * 3 + ["C"] * 3))
    esperado = (resultado.n_usable_per_group >= resultado.min_per_group).sum(axis=1)
    assert list(resultado.n_groups_used) == list(esperado)
    assert list(resultado.n_groups_used) == [2]


# ---------------------------------------------------------------------------
# A irmã do corte de FDR — a guarda que se põe aqui e não fecha
# ---------------------------------------------------------------------------
#
# **O corte de `q` vive hoje em QUATRO sítios, com o mesmo valor**, e nenhum mecanismo os liga:
# `statistics/anova.py`, `api/rotas.py`, e os três padrões de `multiverse/stability.py`. O
# comentário da fronteira já declara a obrigação — *"usar outro aqui faria a tela e a grade
# contarem quantidades diferentes com o mesmo nome"* — e a obrigação não tinha quem a conferisse.
#
# **Unificar num dono só é da unidade do FIO**, e não desta: o núcleo não pode importar da
# fronteira (regra 8 do README), então o dono tem de ser um módulo de `statistics/` e `rotas.py`
# passa a importá-lo. Mexer em `rotas.py` aqui alargaria esta unidade para além do que ela é.
#
# **O que esta unidade PODE fazer é barato e é mecanismo em vez de lembrança:** a guarda que acusa
# no instante em que os quatro deixarem de concordar. É a guarda das cópias irmãs na forma mínima —
# não impede a divergência, mas torna-a impossível de passar despercebida.


def test_o_corte_de_fdr_concorda_entre_o_nucleo_a_fronteira_e_o_multiverso() -> None:
    """Os quatro sítios que cortam `q` têm de dizer o mesmo número.

    Sem esta guarda, mudar um deles deixa a tela e a grade a contar «features significativas» com
    populações diferentes sob o mesmo nome, e nada levanta.
    """
    import inspect

    from refino.api.rotas import FDR_ALPHA as DA_FRONTEIRA
    from refino.multiverse import stability
    from refino.statistics.anova import FDR_ALPHA as DO_NUCLEO

    do_multiverso = {
        f"{nome}(alpha=)": inspect.signature(getattr(stability, nome)).parameters["alpha"].default
        # **Os nomes vêm de LEITURA e não de memória.** A primeira versão escreveu
        # `perfil_de_estabilidade` e `melhores_por_metrica`, que não existem — e a âncora de
        # não-vacuidade abaixo apanhou-o na primeira execução, em segundos, porque a
        # âncora estava na mesma função.
        for nome in ("axis_decomposition", "best_by_metric", "feature_stability")
        if hasattr(stability, nome)
        and "alpha" in inspect.signature(getattr(stability, nome)).parameters
    }
    # Âncora de não-vacuidade NA MESMA FUNÇÃO: um dicionário vazio faria a comparação abaixo
    # passar sobre nada, que é a asserção de ausência satisfeita por vacuidade.
    assert do_multiverso, (
        "nenhuma função de `multiverse/stability.py` expôs um padrão `alpha` — ou elas mudaram de "
        "nome, ou a introspecção falhou. Em qualquer dos casos esta guarda deixou de guardar"
    )

    divergentes = {nome: valor for nome, valor in do_multiverso.items() if valor != DO_NUCLEO}
    assert DA_FRONTEIRA == DO_NUCLEO and not divergentes, (
        f"o corte de FDR divergiu. Núcleo (`statistics/anova.py`): {DO_NUCLEO}; fronteira "
        f"(`api/rotas.py`): {DA_FRONTEIRA}; multiverso: {do_multiverso}. Enquanto os quatro não "
        f"tiverem um dono só — trabalho da unidade do fio —, esta guarda é o que impede a "
        f"divergência de passar despercebida"
    )


# ---------------------------------------------------------------------------
# Kruskal-Wallis com post-hoc de Dunn. A metade SEM oráculo
# ---------------------------------------------------------------------------
#
# **A âncora deste bloco também é fechada, e pelo mesmo motivo da do topo.** Com `k = 3` grupos a
# distribuição nula do `H` é `qui-quadrado` com 2 graus, cuja sobrevivência é `exp(-x/2)` — então
# um caso pequeno tem `H` e `p` calculáveis à mão, em frações exactas, antes de existir função
# nenhuma. Foi o que se fez, e só depois o resultado da implementação foi lido.
#
# **A planta dos empates é o item mais consequente**, e ela exige dado com empates: sobre valores
# todos distintos o fator `1 - Σ(t³-t)/(N³-N)` vale **1**, as duas implementações coincidem ao
# último dígito e o teste seria vácuo. `test_a_planta_dos_empates_separa_a_correcao_da_ausencia`
# calcula as duas quantidades e afirma que divergem **antes** de olhar o resultado.

# Os três blocos da âncora fechada. Oito das nove observações estão empatadas.
EMPATE_A = [1.0, 1.0, 2.0]
EMPATE_B = [2.0, 3.0, 3.0]
EMPATE_C = [4.0, 4.0, 5.0]

# A planta: METADE das dezoito observações no mesmo valor, que é o que 15,8% de zeros produz em
# escala. Escolhida por busca exaustiva sobre as partições de `{1..9}` em três grupos de seis com
# nove zeros — 576 arranjos dão vereditos opostos a 5%, e este é o de maior folga dos dois lados.
PLANTA_A = [1.0, 2.0, 3.0, 5.0, 7.0, 9.0]
PLANTA_B = [0.0, 0.0, 0.0, 0.0, 6.0, 8.0]
PLANTA_C = [0.0, 0.0, 0.0, 0.0, 0.0, 4.0]


def _h_a_mao(blocos: list[list[float]]) -> tuple[float, float, float]:
    """`H` sem correção, o fator de empates, e `H` com correção — dos postos, sem a implementação.

    A definição, e nada mais: `H = 12/(N(N+1)) · Σ Rᵢ²/nᵢ - 3(N+1)`, e o fator é
    `1 - Σ(t³-t)/(N³-N)` sobre os tamanhos dos blocos de empate. É esta função que faz de cada
    cenário um par causal — ela produz as **duas** quantidades que a planta tem de separar.
    """
    todos = np.concatenate([np.asarray(bloco, dtype=np.float64) for bloco in blocos])
    n = int(todos.size)
    postos = rankdata(todos)
    corte, somas = 0, []
    for bloco in blocos:
        somas.append(float(postos[corte : corte + len(bloco)].sum()))
        corte += len(bloco)

    sem = 12.0 / (n * (n + 1.0)) * sum(
        s * s / len(b) for s, b in zip(somas, blocos, strict=True)
    ) - 3.0 * (n + 1.0)
    contagens = np.unique(todos, return_counts=True)[1].astype(np.float64)
    fator = 1.0 - float((contagens**3 - contagens).sum()) / (n**3 - n)
    return sem, fator, sem / fator


def _q_do_dunn(comparacao: PostHocComparison | DunnComparison) -> float:
    """O `q` de uma comparação, recusando a que não o tem — o estreitamento que o tipo exige.

    `AnovaResult.comparisons` é uma união, e ler `qvalue_fdr` dela sem estreitar é o que o mypy
    recusa no `src/`. Aqui a asserção faz o mesmo papel em runtime: se o caminho não paramétrico
    algum dia devolver `PostHocComparison`, o teste diz isso em vez de levantar `AttributeError`
    trinta linhas adiante.
    """
    assert isinstance(comparacao, DunnComparison), (
        f"esperava uma comparação de Dunn e recebi {type(comparacao).__name__}"
    )
    return comparacao.qvalue_fdr


def _dunn_a_mao(blocos: list[list[float]]) -> dict[tuple[int, int], float]:
    """O `p` de Dunn par a par, da definição, sem a implementação.

    `z = (R̄ᵢ - R̄ⱼ) / sqrt([N(N+1)/12 - Σ(t³-t)/(12(N-1))]·(1/nᵢ + 1/nⱼ))`, bilateral pela normal.
    """
    todos = np.concatenate([np.asarray(bloco, dtype=np.float64) for bloco in blocos])
    n = int(todos.size)
    postos = rankdata(todos)
    corte, medias = 0, []
    for bloco in blocos:
        medias.append(float(postos[corte : corte + len(bloco)].mean()))
        corte += len(bloco)
    contagens = np.unique(todos, return_counts=True)[1].astype(np.float64)
    empates = float((contagens**3 - contagens).sum())
    variancia = n * (n + 1.0) / 12.0 - empates / (12.0 * (n - 1.0))

    saida: dict[tuple[int, int], float] = {}
    for i, j in itertools.combinations(range(len(blocos)), 2):
        erro = np.sqrt(variancia * (1.0 / len(blocos[i]) + 1.0 / len(blocos[j])))
        z = (medias[i] - medias[j]) / erro
        saida[(i, j)] = float(2.0 * normal_padrao.sf(abs(z)))
    return saida


# ---------------------------------------------------------------------------
# A âncora à mão — corre antes de qualquer leitura de resultado
# ---------------------------------------------------------------------------


def test_a_ancora_a_mao_do_h_com_correcao_de_empates() -> None:
    """`H = 614/87` e `p = exp(-307/87)` EXACTOS, obtidos sem a implementação.

    Com `A = [1,1,2]`, `B = [2,3,3]`, `C = [4,4,5]` os nove postos são
    `1½, 1½, 3½, 3½, 5½, 5½, 7½, 7½, 9` e as somas por grupo são `6½`, `14½` e `24` — que somam
    45, como `N(N+1)/2` exige.

    * `H` sem correção: `12/90 · (6,5² + 14,5² + 24²)/3 - 30 = 307/45`
    * empates: quatro pares, `Σ(t³-t) = 4·6 = 24`; `N³-N = 720`; fator `= 1 - 24/720 = 29/30`
    * `H` com correção: `(307/45)·(30/29) = 614/87`

    E com `k = 3` a sobrevivência fecha: `P(χ²₂ > x) = exp(-x/2)`, logo `p = exp(-307/87)`.

    **As duas quantidades já divergem nesta âncora** — `307/45 = 6,822` contra `614/87 = 7,057` —,
    então ela sozinha distingue a implementação que corrige da que não corrige. A planta abaixo vai
    mais longe e separa os vereditos.
    """
    sem, fator, com = _h_a_mao([EMPATE_A, EMPATE_B, EMPATE_C])
    assert sem == pytest.approx(307.0 / 45.0, rel=1e-12)
    assert fator == pytest.approx(29.0 / 30.0, rel=1e-12)
    assert com == pytest.approx(614.0 / 87.0, rel=1e-12)

    valores, grupos = _coluna(EMPATE_A, EMPATE_B, EMPATE_C)
    resultado = kruskal_wallis(_matriz(valores, grupos))

    assert resultado.statistic[0] == pytest.approx(614.0 / 87.0, rel=1e-12)
    assert resultado.pvalue[0] == pytest.approx(float(np.exp(-307.0 / 87.0)), rel=1e-12)
    # E NÃO é o valor sem correção — a âncora afirma as duas direções.
    assert resultado.statistic[0] != pytest.approx(307.0 / 45.0, rel=1e-3)
    assert resultado.method == "kruskal_wallis"
    assert resultado.groups == ("A", "B", "C")
    assert resultado.n_per_group == (3, 3, 3)


def test_a_planta_dos_empates_separa_a_correcao_da_ausencia_dela() -> None:
    """Com metade das observações no mesmo valor, os vereditos a 5% são OPOSTOS.

    A ordem é o conteúdo: as duas quantidades são calculadas e a divergência é **afirmada** antes
    de o resultado da implementação ser lido — se elas coincidissem neste dado, o teste passaria
    com uma implementação sem correção nenhuma e seria vácuo.

    Os números, todos fechados: postos `5` para os nove zeros e `10..18` para os nove positivos,
    somas `81`, `52` e `38`; `H` sem correção `= 962/171`, fator `= 283/323`, `H` com correção
    `= 16354/2547`. A 5%: `0,06003` contra `0,04034`.
    """
    blocos = [PLANTA_A, PLANTA_B, PLANTA_C]
    todos = np.concatenate([np.asarray(b) for b in blocos])
    # A planta é o que ela diz ser: metade das observações no mesmo valor.
    assert int((todos == 0.0).sum()) * 2 == todos.size == 18

    sem, fator, com = _h_a_mao(blocos)
    p_sem, p_com = float(np.exp(-sem / 2.0)), float(np.exp(-com / 2.0))

    assert fator == pytest.approx(283.0 / 323.0, rel=1e-12), "o fator de empates mudou de valor"
    assert fator < 0.9, f"fator {fator}: os empates não pesam neste dado, e a planta seria vácua"
    assert p_sem > 0.05 > p_com, (
        f"os vereditos coincidem (p sem correção {p_sem}, com correção {p_com}); a planta não "
        "separaria as duas implementações"
    )

    valores, grupos = _coluna(*blocos)
    resultado = kruskal_wallis(_matriz(valores, grupos))

    assert resultado.statistic[0] == pytest.approx(16354.0 / 2547.0, rel=1e-12)
    assert resultado.pvalue[0] == pytest.approx(p_com, rel=1e-12)
    assert resultado.pvalue[0] != pytest.approx(p_sem, rel=1e-3)
    # E o veredito da implementação é o do lado corrigido: a feature sobrevive ao corte de 5%.
    assert resultado.counts.survivors == 1
    assert resultado.post_hoc.route == "ran"


# ---------------------------------------------------------------------------
# O post-hoc de Dunn — e o oráculo EXTERNO que a identidade algébrica dá
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("rotulo", "a", "b"),
    [
        ("com empates", [0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0], [0.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]),
        ("sem empates", [1.0, 2.0, 3.0, 4.0, 5.0], [6.0, 7.0, 8.0, 9.0, 10.0]),
    ],
)
def test_o_dunn_com_dois_grupos_bate_com_mann_whitney_assintotico(
    rotulo: str, a: list[float], b: list[float]
) -> None:
    """O oráculo EXTERNO do Dunn, e ele existe por identidade algébrica — não por semelhança.

    Não há Dunn no `scipy`. Mas com `k = 2` o `z` de Dunn com correção de empates **é** o `z`
    assintótico de Mann-Whitney sem correção de continuidade: `R̄₁ - R̄₂ = (N/n₁n₂)(U - n₁n₂/2)` e
    o termo de empates de um é o do outro. Logo `scipy.stats.mannwhitneyu` mede exactamente esta
    quantidade sem saber o que é Dunn — que é o que distingue um oráculo de uma segunda cópia da
    minha fórmula.

    Os dois casos existem porque a identidade tem de valer **com** empates, que é onde o termo de
    correção deixa de ser zero: sem eles o caso não exercita o que interessa.
    """
    valores, grupos = _coluna(a, b)
    resultado = kruskal_wallis(_matriz(valores, grupos))
    assert resultado.counts.survivors == 1, (
        f"caso {rotulo!r}: sem sobrevivente o post-hoc não corre e o teste seria vácuo"
    )
    assert len(resultado.comparisons) == 1

    oraculo = mannwhitneyu(a, b, method="asymptotic", use_continuity=False, alternative="two-sided")
    comparacao = resultado.comparisons[0]
    assert isinstance(comparacao, DunnComparison)
    assert comparacao.pvalue == pytest.approx(float(oraculo.pvalue), rel=1e-12)
    # Com UMA comparação na família, Benjamini-Hochberg é a identidade — e dizê-lo é o que
    # impede este cenário de ser lido como prova de que o `q` está certo. Quem prova isso é
    # `test_o_q_do_dunn_e_de_outra_familia_que_o_q_entre_features`.
    assert comparacao.qvalue_fdr == pytest.approx(comparacao.pvalue, rel=1e-12)


def test_um_grupo_deslocado_e_o_par_certo_sobrevive_a_correcao() -> None:
    """O par causal do efeito com QUATRO grupos, que é a população deste estudo.

    Com `D` deslocado, as três comparações que o envolvem sobrevivem ao `q` de 5% e as três que
    não o envolvem não sobrevivem. Sem esta asserção, um post-hoc que devolvesse todos os pares
    como significativos seria indistinguível de um que acertasse.

    > **ESCOPO, e ele foi MEDIDO e não suposto.** Neste dado o conjunto `{q < 0,05}` é o mesmo
    > com e sem o ajuste de Benjamini-Hochberg — `0,01276` e `0,02553` caem do mesmo lado do
    > corte. A primeira versão deste teste parava aqui, e a planta que copia o `p` cru para o
    > campo do `q` **passou por ela**: o nome dizia «sobrevive à correção» e a asserção não a
    > tocava. A última asserção é o conserto — ela afirma que o `q` e o `p` divergem, que é o que
    > separa as duas implementações mesmo quando o veredito coincide.
    """
    gerador = np.random.default_rng(11)
    valores = gerador.normal(10.0, 0.5, size=(24, 4))
    valores[18:, 0] += 8.0
    grupos = ["A"] * 6 + ["B"] * 6 + ["C"] * 6 + ["D"] * 6
    resultado = kruskal_wallis(_matriz(valores, grupos))

    assert resultado.counts.survivors == 1
    assert resultado.post_hoc.route == "ran"
    assert {c.feature for c in resultado.comparisons} == {"f0"}
    assert len(resultado.comparisons) == 6

    significativos = {(c.group_a, c.group_b) for c in resultado.comparisons if _q_do_dunn(c) < 0.05}
    assert significativos == {("A", "D"), ("B", "D"), ("C", "D")}
    # `D` está acima, então a diferença de postos médios `a - b` tem de ser negativa em todos.
    for comparacao in resultado.comparisons:
        assert isinstance(comparacao, DunnComparison)
        if comparacao.group_b == "D":
            assert comparacao.mean_rank_difference < 0.0

    # **E o `q` não é o `p` copiado** — a asserção que a versão anterior não tinha, e sem a qual
    # este cenário não distingue o campo ajustado do campo cru.
    crus = [c.pvalue for c in resultado.comparisons]
    ajustados = [_q_do_dunn(c) for c in resultado.comparisons]
    assert np.allclose(ajustados, benjamini_hochberg(np.array(crus)), rtol=1e-12)
    assert not np.allclose(crus, ajustados, rtol=1e-6), (
        f"o `q` coincide com o `p` cru ({crus} contra {ajustados}); o ajuste da família do "
        "post-hoc não está a correr"
    )


def test_todos_os_grupos_identicos_nenhuma_comparacao_sobrevive() -> None:
    """Sem efeito não há sobrevivente, e a rota diz `no_survivors` — não `nothing_testable`.

    A distinção vale igual no caminho não paramétrico: *"medi e nada sobreviveu"* não
    é *"não havia o que medir"*, e as duas produzem a mesma figura vazia.
    """
    gerador = np.random.default_rng(20260914)
    valores = gerador.normal(10.0, 1.0, size=(18, 6))
    grupos = ["A"] * 6 + ["B"] * 6 + ["C"] * 6
    resultado = kruskal_wallis(_matriz(valores, grupos))

    assert resultado.pvalue.size == 6, "sem features não há o que `np.all` afirme"
    assert np.all(resultado.pvalue > 0.05), f"p-valores inesperadamente baixos: {resultado.pvalue}"
    assert resultado.counts.survivors == 0
    assert resultado.post_hoc.route == "no_survivors"
    assert resultado.post_hoc.method == "dunn"
    assert resultado.comparisons == ()


# ---------------------------------------------------------------------------
# As DUAS famílias de Benjamini-Hochberg
# ---------------------------------------------------------------------------


def test_o_q_do_dunn_e_de_outra_familia_que_o_q_entre_features() -> None:
    """O `q` do Dunn é BH sobre as comparações DAQUELA feature, e o `p` continua cru.

    As duas metades, e a segunda é a que impede o duplo ajuste de passar: o `p` de cada comparação
    bate com o `z` de Dunn calculado aqui, **cru**; e o `q` é o que Benjamini-Hochberg faz sobre
    os `k(k-1)/2` p-valores da feature — não sobre o vetor de `p` entre features, que já correu e
    vive noutro campo.
    """
    valores, grupos = _coluna(PLANTA_A, PLANTA_B, PLANTA_C)
    resultado = kruskal_wallis(_matriz(valores, grupos))
    assert len(resultado.comparisons) == 3

    esperado = _dunn_a_mao([PLANTA_A, PLANTA_B, PLANTA_C])
    indice = {"A": 0, "B": 1, "C": 2}
    crus = []
    for comparacao in resultado.comparisons:
        par = (indice[comparacao.group_a], indice[comparacao.group_b])
        assert comparacao.pvalue == pytest.approx(esperado[par], rel=1e-12)
        crus.append(comparacao.pvalue)

    ajustados = benjamini_hochberg(np.array(crus, dtype=np.float64))
    obtidos = [_q_do_dunn(c) for c in resultado.comparisons]
    assert np.allclose(obtidos, ajustados, rtol=1e-12)
    # E as duas famílias DIVERGEM neste dado: se `q` e `p` coincidissem, o cenário não separaria
    # a implementação que ajusta da que copia o `p` para o campo do `q`.
    assert not np.allclose(crus, ajustados, rtol=1e-6), (
        f"o `q` do Dunn coincide com o `p` cru ({crus} contra {list(ajustados)}); ou o ajuste não "
        "corre, ou este cenário não o distingue"
    )
    # O `q` ENTRE features é outro número, sobre outra população — uma feature só.
    assert resultado.qvalue_fdr[0] == pytest.approx(resultado.pvalue[0], rel=1e-12)


def test_o_tipo_do_post_hoc_e_o_mecanismo_que_impede_o_duplo_ajuste() -> None:
    """Quem devolve `PostHocComparison` continua sem onde escrever um `q`.

    A guarda da ANOVA era a **ausência** de campo. O caminho não paramétrico precisa de um `q` — de
    outra família —, e a resposta foi um tipo próprio, não um campo opcional no tipo antigo:
    assim a ANOVA continua impedida por construção, e não por disciplina.

    A segunda metade é o efeito: `difference` e `mean_rank_difference` são grandezas diferentes e
    **não partilham nome**, então nenhum consumidor pode ler uma julgando ler a outra.
    """
    campos_do_par = {campo.name for campo in fields(PostHocComparison)}
    campos_de_dunn = {campo.name for campo in fields(DunnComparison)}

    assert "qvalue_fdr" not in campos_do_par
    assert "qvalue_fdr" in campos_de_dunn
    assert "difference" in campos_do_par and "difference" not in campos_de_dunn
    assert "mean_rank_difference" in campos_de_dunn and "mean_rank_difference" not in campos_do_par
    # E não é subclasse: herdar teria posto os dois efeitos sob o mesmo nome.
    assert not issubclass(DunnComparison, PostHocComparison)

    valores, grupos = _coluna(ANCORA_A, ANCORA_B, ANCORA_C)
    matriz = _matriz(valores, grupos)
    do_parametrico = one_way_anova(matriz).comparisons
    do_nao_parametrico = kruskal_wallis(matriz).comparisons
    # **Não-vacuidade NA MESMA FUNÇÃO e ANTES**, porque `all(...)` sobre sequência vazia é `True`
    # Sem estas duas linhas, um caminho que devolvesse zero comparações satisfaria as
    # duas asserções abaixo e o teste diria que os tipos estão certos sem ter visto nenhum.
    assert len(do_parametrico) == 3
    assert len(do_nao_parametrico) == 3
    assert all(type(c) is PostHocComparison for c in do_parametrico)
    assert all(type(c) is DunnComparison for c in do_nao_parametrico)


# ---------------------------------------------------------------------------
# Os rótulos — e o que eles existem para desambiguar
# ---------------------------------------------------------------------------


def test_o_rotulo_de_metodo_distingue_anova_de_kruskal_wallis() -> None:
    """`statistic` carrega `F` ou `H`, e **nada além do rótulo** os separa.

    A asserção é sobre a ambiguidade, e não só sobre o rótulo: os dois vetores têm a mesma forma e
    o mesmo `dtype`, então nenhuma introspecção do array diz qual é qual. Quem escrever «F» numa
    legenda a partir do nome do campo escreve errado metade das vezes, e é `method` que fecha isso.
    """
    gerador = np.random.default_rng(7)
    valores = gerador.normal(10.0, 0.5, size=(18, 4))
    valores[12:, 0] += 8.0
    matriz = _matriz(valores, ["A"] * 6 + ["B"] * 6 + ["C"] * 6)

    parametrico = one_way_anova(matriz)
    nao_parametrico = kruskal_wallis(matriz)

    assert parametrico.method == "anova"
    assert nao_parametrico.method == "kruskal_wallis"
    # Indistinguíveis por forma e tipo — é por isso que o rótulo é obrigatório e não decorativo.
    assert parametrico.statistic.shape == nao_parametrico.statistic.shape
    assert parametrico.statistic.dtype == nao_parametrico.statistic.dtype
    assert parametrico.statistic[0] != pytest.approx(nao_parametrico.statistic[0], rel=1e-6)
    assert parametrico.post_hoc.method == "tukey"
    assert nao_parametrico.post_hoc.method == "dunn"


def test_o_post_hoc_escolhivel_e_subconjunto_proprio_do_post_hoc_declarado() -> None:
    """`PostHoc` ⊂ `PostHocMethod`, nas duas direções — e a diferença é exactamente `dunn`.

    Sem esta guarda os dois conjuntos derivam, e a divergência apareceria como um post-hoc que a
    análise deixa escolher e que o resultado nunca sabe declarar — ou o contrário. É a mesma forma
    do par `InjectableDefect` / `DefectType`, que `tests/unit/test_labels.py` já afirma.

    **A direção que importa é a segunda:** `dunn` **não** pode entrar em `AnalysisSpec.post_hoc`,
    porque ali ele seria oferecido ao lado da ANOVA paramétrica, que não tem Dunn nenhum.
    """
    escolhiveis = set(get_args(PostHoc))
    declaraveis = set(get_args(PostHocMethod))

    assert escolhiveis < declaraveis, "o que se escolhe tem de ser sempre declarável"
    assert declaraveis - escolhiveis == {"dunn"}
    assert "dunn" not in escolhiveis
    # E o vocabulário com guarda em runtime continua a ser só o escolhível.
    assert set(VOCABULARY["PostHoc"]) == escolhiveis
    with pytest.raises(InvalidLabelError, match="PostHoc"):
        validate_label("PostHoc", "dunn")


# ---------------------------------------------------------------------------
# Recusas, indefinidos e a população
# ---------------------------------------------------------------------------


def test_grupo_com_uma_amostra_so_e_recusado_nomeando_o_teste_que_recusou() -> None:
    """A recusa é partilhada com a ANOVA; a justificativa **não é**, e a mensagem tem de o dizer.

    *"O quadrado médio residual fica sem contribuição dela"* é verdade da ANOVA e falsa aqui — o
    Kruskal-Wallis não calcula quadrado médio nenhum. Mensagem que explica a recusa com o
    mecanismo do outro teste é rótulo errado à mão, na superfície que o usuário lê.
    """
    valores, grupos = _coluna(EMPATE_A, EMPATE_B, [7.0])
    with pytest.raises(ValueError) as erro:
        kruskal_wallis(_matriz(valores, grupos))
    mensagem = str(erro.value)
    assert "'C'" in mensagem
    assert "Kruskal-Wallis" in mensagem
    assert "quadrado médio residual" not in mensagem

    # E a irmã continua a dizer o que dizia — a mudança preservou a mensagem.
    with pytest.raises(ValueError) as da_anova:
        one_way_anova(_matriz(valores, grupos))
    assert "a ANOVA exige ao menos" in str(da_anova.value)
    assert "quadrado médio residual" in str(da_anova.value)


def test_menos_de_dois_grupos_e_recusado_nomeando_o_teste_e_o_que_ha() -> None:
    valores, grupos = _coluna(EMPATE_A + EMPATE_B)
    with pytest.raises(ValueError) as erro:
        kruskal_wallis(_matriz(valores, grupos))
    mensagem = str(erro.value)
    assert "o Kruskal-Wallis precisa de ao menos 2 grupos" in mensagem
    assert "['A']" in mensagem


def test_feature_toda_constante_tem_h_indefinido_contado_e_a_particao_fecha() -> None:
    """`0/0` é ausência de medida também sobre postos — e ela é CONTADA, nunca filtrada.

    Com todas as observações empatadas, `Σ(t³-t) = N³-N`, o fator de empates é **zero**, e o `H`
    fica indefinido. É o mesmo fato que a irmã paramétrica reporta como quadrado médio entre
    grupos e residual ambos nulos, e conta no mesmo campo.

    A segunda metade é a partição: `statistic_undefined == constant + insufficient_groups`.
    O que falta
    de uma partição vira residual, e residual lê-se como pronto.
    """
    valores = np.array(
        [
            [5.0, 1.0, 1.0],
            [5.0, 2.0, 2.0],
            [5.0, 3.0, 3.0],
            [5.0, 4.0, 4.0],
            [5.0, 5.0, 5.0],
            [5.0, 6.0, 6.0],
            [5.0, np.nan, 7.0],
            [5.0, np.nan, 8.0],
            [5.0, np.nan, 9.0],
        ],
        dtype=np.float64,
    )
    grupos = ["A"] * 3 + ["B"] * 3 + ["C"] * 3
    resultado = kruskal_wallis(_matriz(valores, grupos))
    contagens = resultado.counts

    assert np.isnan(resultado.statistic[0]) and np.isnan(resultado.pvalue[0])
    assert np.isnan(resultado.statistic[1])
    assert contagens.constant == 1
    assert contagens.insufficient_groups == 1
    assert contagens.statistic_undefined == contagens.constant + contagens.insufficient_groups
    assert contagens.qvalue_undefined == contagens.pvalue_undefined == contagens.statistic_undefined
    assert list(resultado.n_usable_per_group[1]) == [3, 3, 0]
    # A terceira coluna é medida e continua na tabela, na posição dela.
    assert np.isfinite(resultado.statistic[2])
    assert resultado.feature == ("f0", "f1", "f2")


def test_nenhum_aviso_dispara_e_o_pre_filtro_e_a_causa() -> None:
    """O par causal do pré-filtro: o `scipy` avisa sobre a coluna constante, e esta função não.

    **As duas metades são necessárias.** «Não avisou» sozinho passaria numa árvore onde nada
    avisasse nunca; o que torna a asserção acionável é a segunda linha, que mostra o aviso a
    disparar sobre **o mesmo dado**, sem o pré-filtro. É a diferença entre ver o mecanismo passar
    e ver que havia o que ele impediu.

    E o motivo de isto ser teste e não nota: a resposta natural a um `RuntimeWarning` é um filtro
    de avisos, e um filtro passa a esconder também os que ainda não se viu.
    """
    valores = np.array(
        [[5.0, 1.0], [5.0, 2.0], [5.0, 3.0], [5.0, 4.0], [5.0, 5.0], [5.0, 9.0]],
        dtype=np.float64,
    )
    grupos = ["A"] * 2 + ["B"] * 2 + ["C"] * 2

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        resultado = kruskal_wallis(_matriz(valores, grupos))
    assert resultado.counts.constant == 1

    # A planta REMOVIDA: os mesmos blocos direto no `scipy`, sem o pré-filtro.
    with pytest.warns(RuntimeWarning):
        kruskal(valores[0:2], valores[2:4], valores[4:6], axis=0)


def test_o_nao_finito_sai_da_ordenacao_e_nao_so_da_contagem() -> None:
    """`+inf` tratado como `nan`: a população medida é a população declarada.

    O par causal, e ele separa duas implementações que dão as duas números finitos: `utilizaveis`
    conta por `isfinite`, mas `nan_policy="omit"` só descarta `nan` — um `+inf` deixado no array
    entraria na ordenação, onde `rankdata` o põe no topo sem se queixar. A matriz com `inf` e a
    matriz com `nan` no mesmo lugar têm de dar o **mesmo** resultado; e passar o `inf` cru ao
    `scipy` dá outro, o que é o que torna esta asserção não vácua.
    """
    valores = np.array([[1.0], [2.0], [3.0], [4.0], [5.0], [np.inf], [7.0], [8.0], [9.0]])
    grupos = ["A"] * 3 + ["B"] * 3 + ["C"] * 3
    com_inf = kruskal_wallis(_matriz(valores, grupos))

    trocado = valores.copy()
    trocado[5, 0] = np.nan
    com_nan = kruskal_wallis(_matriz(trocado, grupos))

    assert list(com_inf.n_usable_per_group[0]) == [3, 2, 3]
    assert com_inf.statistic[0] == pytest.approx(com_nan.statistic[0], rel=1e-12)
    assert com_inf.pvalue[0] == pytest.approx(com_nan.pvalue[0], rel=1e-12)

    # A planta: o `inf` CRU mede outra coisa — e aqui os vereditos a 5% até divergem.
    cru = kruskal(valores[0:3, 0], valores[3:6, 0], valores[6:9, 0])
    assert float(cru.statistic) != pytest.approx(float(com_inf.statistic[0]), rel=1e-6), (
        "o `inf` cru dá o mesmo número que o tratado; neste dado a planta não separaria as duas "
        "implementações"
    )


def test_a_populacao_e_a_mascara_tambem_no_caminho_nao_parametrico() -> None:
    """Uma amostra `validate` com rótulo clínico não entra — e um QC também não.

    Invariante de desenho e não observação: decorre de `SampleType` ter três valores. A
    guarda é partilhada com a ANOVA, e o teste existe porque partilhar código não é o mesmo que
    ter a propriedade — quem trocasse `sample_mask` por «tudo o que não é QC» quebraria só um dos
    dois caminhos, e só um dos dois testes.
    """
    valores_puros, grupos_puros = _coluna(EMPATE_A, EMPATE_B, EMPATE_C)
    limpa = kruskal_wallis(_matriz(valores_puros, grupos_puros))

    valores = np.vstack([valores_puros, [[1000.0]], [[-1000.0]]])
    grupos = [*grupos_puros, "A", "QC"]
    tipos: list[SampleType] = [*(["sample"] * 9), "validate", "qc"]
    suja = kruskal_wallis(_matriz(valores, grupos, tipos))

    assert suja.groups == ("A", "B", "C")
    assert suja.n_per_group == (3, 3, 3)
    assert suja.statistic[0] == pytest.approx(limpa.statistic[0], rel=1e-12)
    assert list(suja.n_usable_per_group[0]) == [3, 3, 3]


# ---------------------------------------------------------------------------
# O corte, o cancelamento e o determinismo
# ---------------------------------------------------------------------------


def test_o_corte_de_fdr_do_kruskal_wallis_e_consultado_e_guardado() -> None:
    """As duas direções da guarda, e o par causal do corte."""
    valores, grupos = _coluna(PLANTA_A, PLANTA_B, PLANTA_C)
    matriz = _matriz(valores, grupos)

    for invalido in (0.0, -0.1, 1.5):
        with pytest.raises(ValueError, match="fdr_alpha"):
            kruskal_wallis(matriz, fdr_alpha=invalido)

    assert kruskal_wallis(matriz, fdr_alpha=1e-12).counts.survivors == 0
    assert kruskal_wallis(matriz, fdr_alpha=0.5).counts.survivors == 1
    assert kruskal_wallis(matriz).fdr_alpha == FDR_ALPHA
    # O corte de 5% é o que separa este dado: a planta foi escolhida para isso.
    assert kruskal_wallis(matriz, fdr_alpha=0.05).counts.survivors == 1


def _matriz_kw_com_duas_sobreviventes() -> RefinoMatrix:
    gerador = np.random.default_rng(1234)
    valores = gerador.normal(10.0, 0.4, size=(18, 3))
    valores[12:, 0] += 6.0
    valores[12:, 1] += 5.0
    return _matriz(valores, ["A"] * 6 + ["B"] * 6 + ["C"] * 6)


def test_cancelamento_no_meio_do_post_hoc_de_dunn_levanta_com_o_rotulo_do_ponto() -> None:
    """O laço do Dunn é o ponto de parada — a estatística é vetorizada e não tem laço.

    A bandeira só diz «sim» na SEGUNDA pergunta: cancelar na primeira não provaria que a checagem
    está dentro do laço, apenas que existe antes dele.
    """
    matriz = _matriz_kw_com_duas_sobreviventes()
    completo = kruskal_wallis(matriz)
    assert completo.counts.survivors >= 2

    chamadas = {"n": 0}

    def bandeira() -> bool:
        chamadas["n"] += 1
        return chamadas["n"] >= 2

    with pytest.raises(ExecutionCancelledError) as erro:
        kruskal_wallis(matriz, cancelled=bandeira)
    assert "kruskal_wallis" in erro.value.where
    assert chamadas["n"] >= 2, "a checagem não correu duas vezes; não está dentro do laço"


def test_sem_cancelamento_o_resultado_do_kruskal_wallis_e_identico() -> None:
    """Âncora de que o parâmetro não moveu número nenhum."""
    matriz = _matriz_kw_com_duas_sobreviventes()
    sem = kruskal_wallis(matriz)
    com = kruskal_wallis(matriz, cancelled=lambda: False)

    assert np.array_equal(sem.statistic, com.statistic, equal_nan=True)
    assert np.array_equal(sem.qvalue_fdr, com.qvalue_fdr, equal_nan=True)
    assert sem.comparisons == com.comparisons
    assert sem.counts == com.counts


def test_determinismo_do_kruskal_wallis() -> None:
    """Mesma entrada, saída idêntica — regra 3 do README. Não há estocasticidade neste caminho."""
    matriz = _matriz_kw_com_duas_sobreviventes()
    primeira = kruskal_wallis(matriz)
    segunda = kruskal_wallis(matriz)

    assert np.array_equal(primeira.statistic, segunda.statistic, equal_nan=True)
    assert np.array_equal(primeira.pvalue, segunda.pvalue, equal_nan=True)
    assert np.array_equal(primeira.qvalue_fdr, segunda.qvalue_fdr, equal_nan=True)
    assert primeira.comparisons == segunda.comparisons


def test_o_post_hoc_de_dunn_so_corre_nas_features_que_sobreviveram_ao_fdr() -> None:
    """O filtro é a primeira etapa e o Dunn a segunda — nunca sobre todas as features."""
    gerador = np.random.default_rng(99)
    valores = gerador.normal(10.0, 0.4, size=(18, 5))
    valores[12:, 2] += 7.0
    resultado = kruskal_wallis(_matriz(valores, ["A"] * 6 + ["B"] * 6 + ["C"] * 6))

    sobreviventes = {
        nome for nome, viva in zip(resultado.feature, resultado.survived, strict=True) if viva
    }
    assert sobreviventes == {"f2"}
    assert {c.feature for c in resultado.comparisons} == sobreviventes
    assert resultado.counts.survivors == len(sobreviventes)
    assert resultado.qvalue_fdr[2] == pytest.approx(
        benjamini_hochberg(resultado.pvalue)[2], rel=1e-12
    )


def test_no_kruskal_wallis_tudo_constante_tambem_da_nothing_testable() -> None:
    """Todas constantes: o denominador do BH era zero, e a figura vazia tem de o poder dizer."""
    valores = np.full((9, 2), 5.0)
    resultado = kruskal_wallis(_matriz(valores, ["A"] * 3 + ["B"] * 3 + ["C"] * 3))

    assert resultado.post_hoc.route == "nothing_testable"
    assert resultado.post_hoc.method == "dunn"
    assert resultado.comparisons == ()
    assert resultado.counts.constant == 2
    assert resultado.counts.survivors == 0


def test_kruskal_sem_nenhuma_coluna_testavel_nao_avisa_sobre_fatia_toda_nan() -> None:
    """A guarda de `_colunas_constantes` quando NADA é testável — o único ramo sem teste.

    **Não é caso de borda: é o que 15,8% de zeros no `pos_hilic` produzem** numa feature em que a
    remoção de outliers apagou os QCs não nulos. Cada grupo aqui tem três linhas — passa a guarda
    de tamanho de grupo — e **uma só observação finita por feature**, abaixo de
    `MIN_AMOSTRAS_POR_GRUPO`. Nenhuma coluna é testável.

    **ESTE TESTE COBRE O RAMO E NÃO PROVA QUE ELE É NECESSÁRIO, e a distinção é medida.** O par
    causal foi corrido: removendo o retorno antecipado de `_colunas_constantes`, este teste continua
    **verde**. A razão está lá — com nada testável, `empilhado` tem zero colunas e `np.nanmin` sobre
    `(n, 0)` devolve `[]` sem aviso (medido em `numpy` 2.5.1). O atalho é comportamentalmente
    neutro.

    Fica escrito porque a leitura natural de um teste que cobre uma guarda é *«a guarda está
    provada»*, e aqui ela não está — o que está afirmado é o **desfecho** quando nenhuma coluna é
    testável. O `simplefilter("error")` fica na mesma: ele é o que faria este teste acusar no dia em
    que a seleção deixasse passar uma coluna toda-`nan`, que é o defeito que importa.
    """
    valores = np.full((9, 2), np.nan)
    # uma finita por grupo e por feature: abaixo do mínimo, em todas as colunas
    for linha in (0, 3, 6):
        valores[linha, :] = [1.0, 2.0]

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        resultado = kruskal_wallis(_matriz(valores, ["A"] * 3 + ["B"] * 3 + ["C"] * 3))

    # Não-vacuidade ANTES das asserções de ausência: sem isto, um resultado com zero
    # features satisfaria as três seguintes por vacuidade.
    assert resultado.counts.n_features == 2
    assert len(resultado.feature) == 2

    assert resultado.counts.insufficient_groups == 2
    assert resultado.counts.constant == 0
    assert resultado.counts.survivors == 0
    assert resultado.post_hoc.route == "nothing_testable"
    assert resultado.comparisons == ()


# ---------------------------------------------------------------------------------------
# O QUE ESTA UNIDADE NÃO PÔDE FECHAR, e o portão que o guarda.
# ---------------------------------------------------------------------------------------


def test_anova_parametric_ainda_nao_escolhe_nada_e_isto_e_um_portao() -> None:
    """`AnalysisSpec.anova_parametric` existe e NINGUÉM o lê — declarado, não suposto.

    `kruskal_wallis` fica ao lado de `one_way_anova`, e a bandeira que escolhe
    entre os dois é `AnalysisSpec.anova_parametric`. **Ligá-la é da unidade do FIO**,
    que é quem monta a chamada a partir do contrato — o núcleo não pode importar da rota.

    Até lá a bandeira é uma caixa que não faz nada, e é exactamente o modo de falha
    da bandeira de cancelamento *«chamada 2.184 vezes na suíte»* com zero levantamentos.
    A diferença entre lá e aqui é este portão.

    **Ele reprova no dia em que o campo ganhar consumidor**, e a resposta certa nesse dia é
    APAGÁ-LO junto com o commit que liga a bandeira — não afrouxá-lo. Portão que sobrevive ao
    seu referente é verificação que já não verifica nada.
    """
    raiz = pathlib.Path(__file__).resolve().parents[2] / "src" / "refino"
    # Os DOIS excluídos são os que a propriedade não alcança: `anova.py` define as funções e
    # `analysis.py` DECLARA o campo. A primeira versão deste portão excluía só `anova.py` e
    # reprovava na árvore limpa, a acusar a declaração — o oráculo estava a apanhar o sítio
    # onde a coisa nasce em vez de onde ela é consumida.
    # **O oráculo exclui por CAMINHO e não por nome de arquivo**, e a diferença importa: um
    # `ui/analysis.py` ou `plots/anova.py` futuro ficaria isento em silêncio se a comparação
    # fosse por basename. O portão nasceu assim e a revisão de processo apanhou-o.
    declaram = {"statistics/anova.py", "domain/analysis.py"}
    modulos = [m for m in raiz.rglob("*.py") if m.relative_to(raiz).as_posix() not in declaram]

    # As três âncoras de não-vacuidade, e vão AQUI e não num teste vizinho: sem elas
    # as asserções de ausência abaixo passam com a varredura vazia e o campo apagado.
    assert len(modulos) > 50, f"a varredura só achou {len(modulos)} módulos — ela não está a ler"
    assert "anova_parametric" in (raiz / "domain" / "analysis.py").read_text(encoding="utf-8")
    fonte_da_anova = (raiz / "statistics" / "anova.py").read_text(encoding="utf-8")
    assert "def one_way_anova(" in fonte_da_anova
    assert "def kruskal_wallis(" in fonte_da_anova

    # **O oráculo é a ÁRVORE SINTÁTICA e não o texto, e a diferença custou uma unidade.** A
    # primeira versão perguntava `"anova_parametric" in fonte`, o que é a FORMA e não a
    # propriedade: a unidade do fio escreveu o identificador numa DOCSTRING a explicar por que não
    # o ligou,
    # e o portão acusou-a de *"ganhou consumidor em `api/rotas.py`"*. Ela contornou reescrevendo
    # a prosa — e um portão que obriga a evitar uma palavra em comentário está a dizer que a
    # forma do código é contrato, quando não é.
    #
    # A propriedade é **leitura de atributo ou ligação de nome**, e prosa não a produz.
    def consome(fonte: str, alvo: str) -> bool:
        try:
            arvore = ast.parse(fonte)
        except SyntaxError:  # pragma: no cover - nenhum módulo de `src/` é insintático
            return False
        for no in ast.walk(arvore):
            if isinstance(no, ast.Attribute) and no.attr == alvo:
                return True
            if isinstance(no, ast.Name) and no.id == alvo:
                return True
        return False

    le_a_bandeira = [
        m.relative_to(raiz).as_posix()
        for m in modulos
        if consome(m.read_text(encoding="utf-8"), "anova_parametric")
    ]
    assert le_a_bandeira == [], (
        "`anova_parametric` ganhou consumidor em " + ", ".join(le_a_bandeira) + ". "
        "Se foi a unidade do fio a ligar a bandeira, APAGUE este teste no mesmo commit."
    )

    escolhe_o_teste = [
        m.relative_to(raiz).as_posix()
        for m in modulos
        if consome(m.read_text(encoding="utf-8"), "kruskal_wallis")
    ]
    assert escolhe_o_teste == [], (
        "`kruskal_wallis` ganhou chamador de produção em " + ", ".join(escolhe_o_teste) + ". "
        "Ligar os dois caminhos é da unidade do fio, e este teste sai no mesmo commit."
    )
