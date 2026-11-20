"""A normalidade por variável e por grupo, depois da correção e da normalização.

**Toda matriz aqui é CONSTRUÍDA, e dizê-lo é requisito (regra 1 do README).** O que estes testes
afirmam é propriedade da função — o tamanho do teste sob a normal, o poder contra a log-normal, a
invariância afim, o empate da imputação, as ausências tipadas, o QC fora dos grupos —, e não
contagem observada nos quatro arquivos do laboratório: essa vai para a decisão, medida sobre eles.

**Os critérios de desfecho foram escritos ANTES de medir com a função de produção**, e
vivem nas constantes abaixo com a origem de cada um. A exploração que os precedeu chamou o
`scipy.stats.shapiro` direto, fora do repositório; nenhum número dela entra aqui como esperado.
"""

from __future__ import annotations

import csv
import io
import warnings
from typing import Any, get_args

import numpy as np
import pytest
from scipy.stats import shapiro

from refino.domain.analysis import (
    PADRAO_DA_ANALISE,
    PARAMETRIC_BY_UNIVARIATE_TEST,
    AnalysisSpec,
)
from refino.domain.labels import InvalidLabelError, UnivariateTest
from refino.domain.matrix import RefinoMatrix
from refino.domain.metadata import SampleMetadata
from refino.statistics.normalidade import (
    FDR_LEVEL,
    REJECTION_LEVEL,
    STOREY_LAMBDA,
    NormalityResult,
    NormalityUnassessable,
    normality_by_group,
)
from refino.statistics.univariate import benjamini_hochberg

pytestmark = pytest.mark.property

SEMENTE = 20261106

# **A banda binomial do tamanho do teste, escrita antes de medir.** Com 1.000 variáveis normais
# num grupo, o número que rejeita a 0,05 é Binomial(1.000; 0,05): três erros-padrão para cada lado
# de 5% dão 2,93% a 7,07%. É a banda da hipótese nula, e não uma tolerância escolhida.
VARIAVEIS = 1_000
BANDA = 3.0 * np.sqrt(0.05 * 0.95 / VARIAVEIS)

# **«Rejeita muito mais», escrito antes de medir:** pelo menos dez vezes o nível. O dossiê do
# plano mediu 0,740 com 13 amostras e 0,901 com 18 para a log-normal com desvio 1 no log; com 15
# o número fica entre os dois. Metade das variáveis é o piso que separa sem depender do valor.
PISO_DA_LOGNORMAL = 0.50

# **O empate da imputação, escrito antes de medir.** O dossiê mediu 0,347 de rejeição com os 3
# menores de 18 valores normais trocados por uma constante abaixo do mínimo, e 0,054 sem a troca.
# O piso é 0,20 — quatro vezes o nível — e o teto da versão limpa é a banda da nula.
PISO_DO_EMPATE = 0.20


def _matriz(
    grupos: list[str],
    valores: np.ndarray,
    *,
    tipos: list[str] | None = None,
) -> RefinoMatrix:
    tipos = tipos or ["sample"] * len(grupos)
    amostras = tuple(
        SampleMetadata(name=f"S{i}", injection_order=i + 1, sample_type=tipo, group=grupo)  # type: ignore[arg-type]
        for i, (grupo, tipo) in enumerate(zip(grupos, tipos, strict=True))
    )
    return RefinoMatrix(
        values=np.asarray(valores, dtype=np.float64),
        feature_names=tuple(f"f{j}" for j in range(np.asarray(valores).shape[1])),
        samples=amostras,
        mode="sintetico",
    )


def _nenhuma_imputada(matriz: RefinoMatrix) -> np.ndarray:
    return np.zeros(np.asarray(matriz.values).shape, dtype=bool)


def _grupo(resultado: NormalityResult, nome: str) -> Any:
    por_nome = {g.group: g for g in resultado.groups}
    assert nome in por_nome, f"o grupo {nome!r} não está no resultado: {sorted(por_nome)}"
    return por_nome[nome]


def _com_qcs(blocos: list[tuple[str, np.ndarray]], rng: np.random.Generator) -> RefinoMatrix:
    """Os blocos de estudo empilhados, e SETE QCs no fim, com valores que o teste rejeitaria.

    **Os QCs estão aqui para que a planta do QC-como-grupo caia sobre dado em que as duas coisas
    diferem:** um grupo a mais seria contado, e com valores log-normais ele rejeitaria.
    """
    largura = blocos[0][1].shape[1]
    qcs = np.exp(rng.normal(size=(7, largura)))
    grupos = [nome for nome, bloco in blocos for _ in range(bloco.shape[0])] + ["QC"] * 7
    tipos = ["sample"] * (len(grupos) - 7) + ["qc"] * 7
    return _matriz(grupos, np.vstack([b for _, b in blocos] + [qcs]), tipos=tipos)


# ---------------------------------------------------------------------------
# O tamanho e o poder — o par causal da distribuição
# ---------------------------------------------------------------------------


def test_sob_a_normal_a_fracao_fica_na_banda_binomial_e_a_lognormal_rejeita_muito_mais() -> None:
    """1.000 variáveis gaussianas em dois grupos de 15: a fração que rejeita fica na banda de 5%.

    **E o par causal, sobre os MESMOS tamanhos:** as mesmas 1.000 variáveis, exponenciadas — a
    log-normal com desvio 1 no log —, rejeitam muito mais. Sem a segunda metade, um teste que
    devolvesse `p = 0,5` para tudo passaria na primeira com fração zero… e reprovaria; um que
    devolvesse `p` uniforme passaria nas duas metades da primeira e cairia na segunda.
    """
    rng = np.random.default_rng(SEMENTE)
    normais = rng.normal(size=(30, VARIAVEIS))
    matriz = _com_qcs([("A", normais[:15]), ("B", normais[15:])], rng)

    resultado = normality_by_group(matriz, _nenhuma_imputada(matriz))
    for nome in ("A", "B"):
        grupo = _grupo(resultado, nome)
        assert grupo.n_tested == VARIAVEIS, (nome, grupo.n_tested)
        assert grupo.fraction_rejected is not None
        assert abs(grupo.fraction_rejected - 0.05) <= BANDA, (
            f"grupo {nome}: {grupo.fraction_rejected:.4f} rejeitam sob a normal, fora da banda "
            f"binomial de 5% ± {BANDA:.4f}"
        )

    lognormais = np.exp(normais)
    matriz_log = _com_qcs([("A", lognormais[:15]), ("B", lognormais[15:])], rng)
    resultado_log = normality_by_group(matriz_log, _nenhuma_imputada(matriz_log))
    for nome in ("A", "B"):
        grupo = _grupo(resultado_log, nome)
        assert grupo.fraction_rejected is not None
        assert grupo.fraction_rejected >= PISO_DA_LOGNORMAL, (
            f"grupo {nome}: a log-normal rejeita {grupo.fraction_rejected:.3f}, abaixo do piso "
            f"{PISO_DA_LOGNORMAL} — o teste não separa a log-normal da normal"
        )


def test_o_w_e_o_p_sao_os_do_scipy_coluna_a_coluna_com_n_diferentes() -> None:
    """O oráculo é a chamada por coluna do `scipy`, sobre os valores finitos de cada uma.

    **A vetorização agrupa as colunas pelo número de valores finitos**, e é aí que ela pode errar:
    uma coluna com dois não finitos tem de ser testada sobre os 16 que sobram, e não sobre 18 com
    `nan` dentro. As colunas abaixo têm 18, 17, 16 e 3 valores.
    """
    rng = np.random.default_rng(SEMENTE + 1)
    valores = rng.normal(size=(18, 6))
    valores[0, 1] = np.nan
    valores[[2, 5], 2] = np.nan
    valores[3:, 3] = np.inf
    valores[0, 4] = -np.inf
    matriz = _matriz(["A"] * 18, valores)

    grupo = _grupo(normality_by_group(matriz, _nenhuma_imputada(matriz)), "A")
    assert grupo.n_values.tolist() == [18, 17, 16, 3, 17, 18]
    for j in range(6):
        finitos = valores[np.isfinite(valores[:, j]), j]
        oraculo = shapiro(finitos)
        assert grupo.statistic[j] == pytest.approx(float(oraculo.statistic), abs=1e-12), j
        assert grupo.pvalue[j] == pytest.approx(float(oraculo.pvalue), abs=1e-12), j


def test_o_w_nao_muda_por_transformacao_afim_e_muda_pela_que_nao_e() -> None:
    """A mesma variável multiplicada e somada por constantes dá o mesmo W — e o mesmo `p`.

    **É o que torna a tabela independente do escalonamento**, que é afim por variável em todos os
    métodos que o pipeline oferece. O negativo é uma transformação que NÃO é afim — o
    exponencial —, e ele tem de mover o W: sem ele, um teste que devolvesse sempre o mesmo número
    passaria na igualdade.
    """
    rng = np.random.default_rng(SEMENTE + 2)
    x = rng.normal(size=18)
    colunas = [x, 5.0 * x + 3.0, -2.0 * x - 0.5, 1e-4 * x, 1e6 * x - 3e6, np.exp(x)]
    matriz = _matriz(["A"] * 18, np.column_stack(colunas))

    grupo = _grupo(normality_by_group(matriz, _nenhuma_imputada(matriz)), "A")
    for j in range(1, 5):
        assert grupo.statistic[j] == pytest.approx(grupo.statistic[0], abs=1e-12), j
        assert grupo.pvalue[j] == pytest.approx(grupo.pvalue[0], abs=1e-12), j
    assert abs(grupo.statistic[5] - grupo.statistic[0]) > 1e-3, (
        "o exponencial não moveu o W: a igualdade acima não separaria nada"
    )


# ---------------------------------------------------------------------------
# O empate da imputação — contado à parte
# ---------------------------------------------------------------------------


def test_o_empate_da_imputacao_eleva_a_rejeicao_e_sai_na_contagem_a_parte() -> None:
    """Os 3 menores de 18 valores normais trocados por uma constante abaixo do mínimo.

    **É o que `one_fifth_min` faz a uma variável com três ausentes no grupo:** três valores iguais,
    abaixo de todos os medidos. O Shapiro-Wilk rejeita isso sozinho, e a rejeição é do empate e
    não da forma — é outro motivo, e por isso fica fora do resumo.

    **O par causal tem duas metades, e as duas caem sobre dado em que as coisas diferem:**
    a rejeição com a troca contra a mesma variável sem ela; e a máscara de imputadas marcada contra
    a mesma matriz sem a máscara — sem ela, as plantadas entram no resumo e o número dele sobe.
    """
    rng = np.random.default_rng(SEMENTE + 3)
    limpas = rng.normal(size=(18, 2 * VARIAVEIS))
    plantadas = limpas.copy()
    imputadas = np.zeros(plantadas.shape, dtype=bool)
    # Metade das variáveis recebe a planta; a outra metade fica limpa, na mesma matriz.
    for j in range(VARIAVEIS):
        tres_menores = np.argsort(plantadas[:, j])[:3]
        plantadas[tres_menores, j] = plantadas[:, j].min() - 1.0
        imputadas[tres_menores, j] = True

    matriz_limpa = _matriz(["A"] * 18, limpas)
    limpo = _grupo(normality_by_group(matriz_limpa, _nenhuma_imputada(matriz_limpa)), "A")
    matriz = _matriz(["A"] * 18, plantadas)
    marcado = _grupo(normality_by_group(matriz, imputadas), "A")
    sem_mascara = _grupo(normality_by_group(matriz, _nenhuma_imputada(matriz)), "A")

    rejeicao_limpa = float(np.mean(limpo.pvalue[:VARIAVEIS] < REJECTION_LEVEL))
    rejeicao_plantada = float(np.mean(marcado.pvalue[:VARIAVEIS] < REJECTION_LEVEL))
    assert rejeicao_limpa <= 0.05 + BANDA, rejeicao_limpa
    assert rejeicao_plantada >= PISO_DO_EMPATE, (
        f"a planta do empate rejeita {rejeicao_plantada:.3f}, abaixo do piso {PISO_DO_EMPATE}: "
        "o empate não derruba o teste, e a contagem à parte não teria o que separar"
    )

    # As plantadas saem do resumo e entram na contagem à parte, e só elas.
    assert marcado.has_imputed[:VARIAVEIS].all() and not marcado.has_imputed[VARIAVEIS:].any()
    assert marcado.n_with_imputed == VARIAVEIS
    assert marcado.n_with_imputed_rejected == int(
        np.sum(marcado.pvalue[:VARIAVEIS] < REJECTION_LEVEL)
    )
    assert marcado.n_tested == VARIAVEIS
    assert marcado.n_rejected == int(np.sum(marcado.pvalue[VARIAVEIS:] < REJECTION_LEVEL))
    # O q fica definido só na população do resumo.
    assert np.isnan(marcado.qvalue[:VARIAVEIS]).all()
    assert np.isfinite(marcado.qvalue[VARIAVEIS:]).all()

    # A outra metade do par: sem a máscara, as plantadas entram no resumo e o número dele sobe.
    assert sem_mascara.n_with_imputed == 0 and sem_mascara.n_tested == 2 * VARIAVEIS
    assert sem_mascara.n_rejected > marcado.n_rejected, (
        "sem a máscara o resumo não ganhou rejeições: a separação não está a separar nada"
    )


# ---------------------------------------------------------------------------
# As ausências tipadas
# ---------------------------------------------------------------------------


def test_grupo_com_duas_amostras_e_variavel_constante_sao_nao_avaliaveis_com_motivo() -> None:
    """Grupo de 2: todas «não avaliável», com o motivo; constante num grupo também.

    **E nunca `p` não finito sem motivo, nas duas direções.** A âncora é o próprio `scipy`: sobre a
    coluna constante ele devolve `nan` sem dizer porquê, que é o desfecho que o motivo tipado
    existe para não ter.
    """
    rng = np.random.default_rng(SEMENTE + 4)
    valores = rng.normal(size=(20, 4))
    valores[2:, 1] = 7.25  # constante no grupo de 18
    matriz = _matriz(["pequeno"] * 2 + ["grande"] * 18, valores)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        p_do_scipy = float(shapiro(valores[2:, 1]).pvalue)
    assert np.isnan(p_do_scipy), (
        "âncora: o scipy deixou de devolver `nan` para a coluna constante, e a ausência tipada "
        "deixaria de ser a única coisa entre a tela e um `p` sem motivo"
    )

    resultado = normality_by_group(matriz, _nenhuma_imputada(matriz))
    pequeno = _grupo(resultado, "pequeno")
    grande = _grupo(resultado, "grande")

    assert pequeno.n_samples == 2
    assert pequeno.unassessable == ("fewer_than_three_values",) * 4
    assert pequeno.n_fewer_than_three == 4 and pequeno.n_tested == 0
    assert pequeno.fraction_rejected is None and pequeno.min_fraction_non_normal is None
    assert grande.unassessable[1] == "constant_in_group"
    assert grande.n_constant == 1 and grande.n_tested == 3

    vocabulario = set(get_args(NormalityUnassessable))
    for grupo in (pequeno, grande):
        for j, motivo in enumerate(grupo.unassessable):
            assert (motivo is None) == bool(np.isfinite(grupo.pvalue[j])), (grupo.group, j, motivo)
            assert (motivo is None) == bool(np.isfinite(grupo.statistic[j])), (grupo.group, j)
            assert motivo is None or motivo in vocabulario, motivo


def test_a_particao_de_cada_grupo_soma_as_variaveis_e_o_resumo_sai_dos_vetores() -> None:
    """Não avaliável, com imputado e testada são uma partição; o resumo é contado dos vetores.

    **É o outro caminho como oráculo:** cada número do resumo é recontado aqui a partir dos
    vetores por variável, com o `benjamini_hochberg` da casa e a fórmula de Storey escrita à mão.
    """
    rng = np.random.default_rng(SEMENTE + 5)
    valores = np.exp(rng.normal(size=(16, 300)))
    valores[:, 0] = 3.0  # constante
    valores[:13, 1] = np.nan  # só 3 finitos
    valores[:14, 2] = np.nan  # só 2 finitos
    imputadas = np.zeros(valores.shape, dtype=bool)
    imputadas[0, 10:40] = True
    matriz = _matriz(["A"] * 16, valores)

    grupo = _grupo(normality_by_group(matriz, imputadas), "A")
    assert grupo.n_fewer_than_three == 1 and grupo.n_constant == 1
    assert (
        grupo.n_fewer_than_three + grupo.n_constant + grupo.n_with_imputed + grupo.n_tested
        == matriz.n_features
    )

    testadas = np.array([m is None for m in grupo.unassessable]) & ~grupo.has_imputed
    p = grupo.pvalue[testadas]
    # 300 variáveis, menos a constante, menos a de dois valores, menos as 30 imputadas; a de três
    # valores finitos é avaliável e fica.
    assert grupo.n_with_imputed == 30
    assert p.size == grupo.n_tested == 300 - 1 - 1 - 30
    assert grupo.n_rejected == int(np.sum(p < REJECTION_LEVEL))
    assert grupo.fraction_rejected == pytest.approx(grupo.n_rejected / grupo.n_tested)
    q = benjamini_hochberg(np.where(testadas, grupo.pvalue, np.nan))
    np.testing.assert_array_equal(np.isnan(q), np.isnan(grupo.qvalue))
    np.testing.assert_allclose(grupo.qvalue[testadas], q[testadas], rtol=0, atol=0)
    assert grupo.n_rejected_fdr == int(np.sum(q[testadas] < FDR_LEVEL))
    pi0 = min(1.0, float(np.sum(p > STOREY_LAMBDA)) / (p.size * (1.0 - STOREY_LAMBDA)))
    assert grupo.min_fraction_non_normal == pytest.approx(1.0 - pi0)


def test_a_mascara_de_imputadas_com_outra_forma_e_recusada() -> None:
    """A máscara marca as células desta matriz uma a uma; com outra forma, marcaria outras."""
    matriz = _matriz(["A"] * 5, np.ones((5, 3)) + np.arange(15).reshape(5, 3))
    with pytest.raises(ValueError, match="forma"):
        normality_by_group(matriz, np.zeros((5, 2), dtype=bool))


# ---------------------------------------------------------------------------
# O QC não é grupo
# ---------------------------------------------------------------------------


def test_o_qc_nao_entra_como_grupo_nem_pelo_nome_nem_pelo_tipo() -> None:
    """Os grupos saem de `sample_mask`, e não de `groups` — o QC e o `validate` ficam de fora.

    **Duas plantas e as duas caem sobre dado em que as coisas diferem.** O QC tem sete
    valores por variável, então como grupo ele seria testado e apareceria; e o pool chama-se
    `Pool` e não `QC`, então excluir pelo nome o deixaria entrar. A amostra `validate` tem o rótulo
    de um grupo de estudo e não pode engordar o `n` dele.
    """
    rng = np.random.default_rng(SEMENTE + 6)
    estudo = rng.normal(size=(12, 40))
    pool = np.exp(rng.normal(size=(7, 40)))
    validacao = rng.normal(size=(1, 40))
    grupos = ["A"] * 6 + ["B"] * 6 + ["Pool"] * 7 + ["A"]
    tipos = ["sample"] * 12 + ["qc"] * 7 + ["validate"]
    matriz = _matriz(grupos, np.vstack([estudo, pool, validacao]), tipos=tipos)

    assert "Pool" in set(matriz.groups), "âncora: a planta do pool não aterrou na matriz"
    resultado = normality_by_group(matriz, _nenhuma_imputada(matriz))
    assert tuple(g.group for g in resultado.groups) == ("A", "B"), [
        g.group for g in resultado.groups
    ]
    assert _grupo(resultado, "A").n_samples == 6, "a amostra `validate` entrou no grupo A"


# ---------------------------------------------------------------------------
# O resumo nunca apresenta a fração que não rejeita como fração normal
# ---------------------------------------------------------------------------


def test_com_t_de_3_gl_o_resumo_nao_apresenta_maioria_normal_e_o_pelo_menos_e_positivo() -> None:
    """Todas as variáveis t com 3 graus de liberdade, grupos de 13 e de 18 — nenhuma é normal.

    **O π̂₀ de Storey sai ALTO mesmo assim**, porque o teste tem pouco poder com estes tamanhos: a
    revisão do plano mediu 0,73 com 13 amostras. É por isso que ele nunca vai para a tela como
    «fração normal». O que vai é `1 - π̂₀`, como piso do que NÃO segue a normal, e ele tem de ser
    positivo aqui. O texto é conferido junto: ele diz «pelo menos» e não diz quantas são normais.
    """
    from refino.interface.fluxo import prosa_dos_grupos_da_normalidade

    rng = np.random.default_rng(SEMENTE + 7)
    t13 = rng.standard_t(3, size=(13, 2 * VARIAVEIS))
    t18 = rng.standard_t(3, size=(18, 2 * VARIAVEIS))
    matriz = _matriz(["G13"] * 13 + ["G18"] * 18, np.vstack([t13, t18]))

    resultado = normality_by_group(matriz, _nenhuma_imputada(matriz))
    for nome in ("G13", "G18"):
        grupo = _grupo(resultado, nome)
        assert grupo.min_fraction_non_normal is not None
        assert grupo.min_fraction_non_normal > 0.0, (nome, grupo.min_fraction_non_normal)
        # Nenhuma é normal, e o resumo também não pode ler-se como «a maioria é normal».
        assert grupo.fraction_rejected is not None and grupo.fraction_rejected < 0.5

    from refino.api.rotas import _resumo_da_normalidade

    fio = _resumo_da_normalidade(resultado).model_dump()
    texto = prosa_dos_grupos_da_normalidade(
        fio["grupos"],
        alfa=fio["alfa"],
        nivel_do_fdr=fio["nivel_do_fdr"],
        lambda_de_storey=fio["lambda_de_storey"],
    )
    assert "pelo menos" in texto.casefold(), texto
    proibidas = ("são normais", "seguem a normal", "é normal", "maioria normal")
    for frase in texto.split("\n"):
        for proibida in proibidas:
            # «não seguem a normal» é a leitura certa; a proibida é a afirmativa.
            assert proibida not in frase.replace("não seguem a normal", ""), frase


# ---------------------------------------------------------------------------
# A escolha do teste do contraste — a guarda da fronteira
# ---------------------------------------------------------------------------


def test_a_escolha_do_teste_tem_tres_valores_o_padrao_e_o_automatico_e_o_resto_e_recusado() -> None:
    """«Automático», «Welch», «Mann-Whitney» — e o `parametric` de cada um.

    **O padrão é o automático**, que é a decisão única: `parametric=None`. Um valor
    fora dos três é recusado na construção, e não cai num `else` que se comportaria como um dos
    outros dois — o modo de falha insidioso.
    """
    assert PADRAO_DA_ANALISE.univariate_test == "automatic"
    assert set(PARAMETRIC_BY_UNIVARIATE_TEST) == set(get_args(UnivariateTest))
    assert PARAMETRIC_BY_UNIVARIATE_TEST == {
        "automatic": None,
        "welch": True,
        "mann_whitney": False,
    }
    for valido in get_args(UnivariateTest):
        assert AnalysisSpec(univariate_test=valido).univariate_test == valido
    for invalido in ("t", "Welch", "", None, True):
        with pytest.raises(InvalidLabelError):
            AnalysisSpec(univariate_test=invalido)  # type: ignore[arg-type]


def test_a_familia_da_normalidade_vem_ligada_e_conta_como_pedido() -> None:
    """Ligada por padrão, pelo critério de KD11 — custa milissegundos —, e pedida quando ligada."""
    assert PADRAO_DA_ANALISE.normality is True
    so_ela = AnalysisSpec(
        anova=False, correlation=False, permanova=False, normality=True, classifiers=()
    )
    assert so_ela.pediu_alguma_coisa()
    nenhuma = AnalysisSpec(
        anova=False, correlation=False, permanova=False, normality=False, classifiers=()
    )
    assert not nenhuma.pediu_alguma_coisa()


# ---------------------------------------------------------------------------
# A tabela por variável em CSV, e o texto do resumo — o que viaja até a pessoa
# ---------------------------------------------------------------------------


def _fio_de_exemplo() -> dict[str, Any]:
    """Um resultado real da função, passado pelo esquema da resposta — o que a janela recebe."""
    from refino.api.rotas import _resumo_da_normalidade

    rng = np.random.default_rng(SEMENTE + 8)
    valores = np.exp(rng.normal(size=(9, 5)))
    valores[:, 3] = 2.0  # constante em todos os grupos
    imputadas = np.zeros(valores.shape, dtype=bool)
    imputadas[0, 4] = True
    matriz = _matriz(["A"] * 4 + ["B"] * 2 + ["QC"] * 3, valores, tipos=["sample"] * 6 + ["qc"] * 3)
    return _resumo_da_normalidade(normality_by_group(matriz, imputadas)).model_dump()


def test_a_tabela_em_csv_tem_uma_linha_por_variavel_e_grupo_e_vazio_onde_nao_ha_valor() -> None:
    """O cabeçalho, uma linha por (grupo, variável), e célula vazia onde o número não existe.

    **Lida de volta pelo módulo `csv`**, que é o que uma planilha faz: uma vírgula a mais numa
    linha deslocaria todas as colunas dela, e só a releitura o mostra.
    """
    from refino.interface.fluxo import COLUNAS_DA_TABELA_DA_NORMALIDADE, csv_da_normalidade

    fio = _fio_de_exemplo()
    texto = csv_da_normalidade(fio["features"], fio["grupos"])
    linhas = list(csv.reader(io.StringIO(texto)))
    assert tuple(linhas[0]) == COLUNAS_DA_TABELA_DA_NORMALIDADE
    corpo = linhas[1:]
    assert len(corpo) == 2 * 5, len(corpo)
    por_chave = {(linha[0], linha[1]): dict(zip(linhas[0], linha, strict=True)) for linha in corpo}
    assert {grupo for grupo, _ in por_chave} == {"A", "B"}, "o QC entrou na tabela"

    # B tem 2 amostras: todas sem W, p e q, com o motivo.
    for j in range(5):
        linha = por_chave[("B", f"f{j}")]
        assert linha["w"] == linha["p"] == linha["q"] == "", linha
        assert linha["nao_avaliavel"] == "fewer_than_three_values", linha
    # A constante em A: motivo próprio; a imputada em A: W e p, sem q.
    assert por_chave[("A", "f3")]["nao_avaliavel"] == "constant_in_group"
    imputada = por_chave[("A", "f4")]
    assert imputada["valor_imputado"] == "sim" and imputada["p"] != "" and imputada["q"] == ""
    testada = por_chave[("A", "f0")]
    assert testada["valor_imputado"] == "não" and testada["q"] != "" and testada["n"] == "4"
    grupo_a = next(g for g in fio["grupos"] if g["grupo"] == "A")
    assert float(testada["w"]) == grupo_a["w"][0], "o W escrito não é o do fio"


def test_o_texto_do_resumo_tem_um_paragrafo_por_grupo_com_os_numeros_do_fio() -> None:
    """Os números do fio chegam ao texto: as contagens, a fração e os 5% esperados ao lado.

    **O fio é o do esquema real**, e não um dicionário escrito à mão: uma chave lida pelo texto que
    o esquema não declarasse cairia num valor de recurso, e este teste é quem o apanha.
    """
    from refino.interface.fluxo import prosa_dos_grupos_da_normalidade

    fio = _fio_de_exemplo()
    texto = prosa_dos_grupos_da_normalidade(
        fio["grupos"],
        alfa=fio["alfa"],
        nivel_do_fdr=fio["nivel_do_fdr"],
        lambda_de_storey=fio["lambda_de_storey"],
    )
    paragrafos = [p for p in texto.split("\n\n") if p.strip()]
    assert len(paragrafos) == 2, paragrafos
    assert paragrafos[0].startswith("**A**") and paragrafos[1].startswith("**B**"), paragrafos
    assert "4 amostras" in paragrafos[0] and "os 5% que o acaso" in paragrafos[0], paragrafos[0]
    assert "1 constante no grupo" in paragrafos[0], paragrafos[0]
    assert "1 variável com valor imputado" in paragrafos[0], paragrafos[0]
    assert "2 amostras" in paragrafos[1] and "5 com menos de 3 valores" in paragrafos[1], (
        paragrafos[1]
    )
    assert "não declarado" not in texto, texto
    # A concordância de máquina — «0 rejeitam», «1 constantes» — não chega à tela.
    for errada in (" 0 rejeitam", " 1 rejeitam", "1 constantes", " 0 ficam"):
        assert errada not in texto, (errada, texto)
