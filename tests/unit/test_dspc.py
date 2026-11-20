"""A rede de correlação parcial esparsa: a correção de viés, o `p` por aresta e o vazio.

## O teste que carrega a unidade é o PAR CAUSAL da correção

R8 diz que sem o passo de correção o resultado não é o método. O risco que o plano nomeia é que
**o atalho não produz erro**: uma rede estimada só pelo *graphical lasso* sai igualmente desenhável,
e a única coisa que falta é a inferência. Nada reprovaria.

`test_o_PAR_CAUSAL_da_correcao_de_vies_diverge` é o que fecha isso, e ele está construído para cair
sobre dado em que as duas coisas **diferem**:

1. afirma **antes de comparar** que a penalização pôs zeros exactos fora da diagonal — se não
   pusesse, não haveria o que a correção desenviesar e o par seria vácuo;
2. calcula o `p` pelos dois caminhos sobre a **mesma** precisão e a **mesma** covariância;
3. afirma que no caminho sem correção aqueles pares dão `p = 1` **exacto** — ausência de inferência,
   não inferência fraca — e que no caminho com correção dão `p < 1`.

A diferença não é de magnitude, é de **existência**: é por isso que a planta separa.

## As duas âncoras da correção são EXACTAS, e nenhuma delas precisa de oráculo

`2Θ - ΘΣΘ` é uma linha, e uma linha transcrita errada — sinal trocado, fator dois ausente — produz
uma rede perfeitamente bem formada. As duas âncoras são as da docstring do módulo: o **ponto fixo**
(com `Θ = Σ⁻¹` a correção é nula) e o **valor à mão** (`Θ = I`, `Σ = [[1, ½], [½, 1]]`). A primeira
sozinha passaria com `T = Θ`, que é a correção ausente; a segunda sozinha passaria com fórmulas que
coincidem naquele ponto. **Juntas não passam com nenhuma das duas.**

## Vazio é resultado, e a asserção dele é de AUSÊNCIA

*"Nenhuma aresta sobrevive"* é satisfeito por uma rede com zero pares testados — matriz vazia,
recorte que descartou tudo, laço que não correu. Em toda função que afirma ausência, a asserção de
não-vacuidade está **na mesma função e antes**: afirma-se que houve pares testados e que a
rede tem features, e só então que nenhuma aresta passou o corte.

O mesmo vale ao contrário em `test_a_contagem_de_arestas_vem_de_DOIS_caminhos`: `n_edges` e
`len(edges)` podem ambos ser zero, e uma igualdade entre dois zeros não afirma nada sobre o laço.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest

from refino.domain.analysis import AnalysisSpec
from refino.domain.cancellation import CancelCheck, ExecutionCancelledError
from refino.domain.matrix import RefinoMatrix
from refino.domain.metadata import SampleMetadata
from refino.statistics.clustering import TETO_DE_FEATURES_PARA_COMPUTAR
from refino.statistics.dspc import (
    DOBRAS_DA_VALIDACAO_CRUZADA,
    MIN_AMOSTRAS_PARA_DSPC,
    DspcResult,
    RedeAcimaDoTetoError,
    RedeSemPopulacaoError,
    desenviesa_precisao,
    dspc_network,
    p_por_aresta,
    precisao_esparsa,
)

# ---------------------------------------------------------------------------
# As fixturas sintéticas
# ---------------------------------------------------------------------------


def matriz_de(
    valores: np.ndarray,
    *,
    tipos: Sequence[str] | None = None,
    nomes: Sequence[str] | None = None,
) -> RefinoMatrix:
    """Uma `RefinoMatrix` sintética a partir de um array. Mesma forma da de `test_correlacao`."""
    array = np.ascontiguousarray(valores, dtype=np.float64)
    array.flags.writeable = False
    n_amostras, n_features = array.shape
    tipos_de_amostra = list(tipos) if tipos is not None else ["sample"] * n_amostras
    amostras = tuple(
        SampleMetadata(
            name=f"s{i:03d}",
            injection_order=i + 1,
            sample_type=tipos_de_amostra[i],  # type: ignore[arg-type]
            group="G",
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


def precisao_em_cadeia(k: int, fora: float = -0.45) -> np.ndarray:
    """A precisão de uma cadeia AR(1): `f000-f001-f002-…`, e nada mais.

    **A estrutura é escolhida para as arestas verdadeiras serem NOMEÁVEIS.** Uma precisão aleatória
    esparsa daria uma verdade que o teste teria de recalcular do próprio gerador; uma cadeia dá
    `k-1` pares que se escrevem à mão, e é contra esses que a recuperação é medida.
    """
    theta = np.eye(k, dtype=np.float64)
    for i in range(k - 1):
        theta[i, i + 1] = theta[i + 1, i] = fora
    return theta


def amostra_de(theta: np.ndarray, *, n: int, seed: int) -> np.ndarray:
    """`n` observações da normal multivariada cuja precisão é `theta`."""
    gerador = np.random.default_rng(seed)
    return np.asarray(
        gerador.multivariate_normal(np.zeros(theta.shape[0]), np.linalg.inv(theta), size=n),
        dtype=np.float64,
    )


def padronizada(x: np.ndarray) -> np.ndarray:
    """Centrada e escalada, **por uma conta escrita aqui e não pelo helper do módulo**.

    O módulo tem `_padroniza`; usá-lo aqui faria o teste da recuperação depender da mesma conta que
    o caminho sob teste usa, e um erro de escala passaria pelos dois lados.
    """
    centrada = x - x.mean(axis=0)
    return np.asarray(centrada / centrada.std(axis=0, ddof=0), dtype=np.float64)


CADEIA_DE_OITO = {"k": 8, "n": 120, "seed": 20260918}
"""O caso de trabalho da maioria dos testes. **Pequeno de propósito.**

`k = 8` dá 28 pares e `n = 120` dá poder para a assintótica — o ajuste custa uma fração de segundo,
e família cara fica fora do caminho de `make check`.
"""


def arestas_da_cadeia(k: int) -> set[tuple[str, str]]:
    """Os pares que a cadeia tem por construção, na ordem canônica dos nomes."""
    return {(f"f{i:03d}", f"f{i + 1:03d}") for i in range(k - 1)}


def rede_da_cadeia(alpha: float = 0.05) -> DspcResult:
    """A rede sobre a cadeia de oito. Nasce em cada teste que a pede — o resultado é congelado."""
    x = amostra_de(precisao_em_cadeia(CADEIA_DE_OITO["k"]), n=CADEIA_DE_OITO["n"], seed=20260918)
    return dspc_network(matriz_de(x), AnalysisSpec(dspc=True, dspc_alpha=alpha))


# ---------------------------------------------------------------------------
# 1. A correção de viés — as duas âncoras exactas
# ---------------------------------------------------------------------------


def test_a_correcao_e_NULA_no_ponto_fixo() -> None:  # noqa: N802
    """Com `Θ = Σ⁻¹` exacto, `2Θ - ΘΣΘ = Θ`. **É o que distingue correção de transformação.**

    Uma transformação arbitrária mexeria também no estimador que já não tem viés. Esta âncora é
    exacta na aritmética real e o `float64` a cumpre ao epsilon de máquina.
    """
    gerador = np.random.default_rng(3)
    base = gerador.normal(size=(40, 6))
    sigma = np.asarray(base.T @ base / 40.0, dtype=np.float64)
    theta = np.linalg.inv(sigma)

    corrigida = desenviesa_precisao(theta, sigma)

    # A planta aterrou: `sigma` é de facto invertível e `theta` não é trivial.
    assert np.isfinite(theta).all()
    assert float(np.abs(theta - np.eye(6)).max()) > 1e-3
    assert float(np.abs(corrigida - theta).max()) < 1e-9


def test_a_correcao_bate_o_valor_calculado_A_MAO() -> None:  # noqa: N802
    """`Θ = I`, `Σ = [[1, ½], [½, 1]]` ⇒ `T = 2I - Σ`. Fixa o SINAL e o fator.

    O ponto fixo sozinho não fixa nenhum dos dois: `T = Θ` o satisfaz, e `T = Θ` é a correção
    ausente. Este caso é o que reprova uma transcrição com o sinal trocado ou sem o `2`.
    """
    theta = np.eye(2)
    sigma = np.array([[1.0, 0.5], [0.5, 1.0]])

    corrigida = desenviesa_precisao(theta, sigma)

    esperada = np.array([[1.0, -0.5], [-0.5, 1.0]])
    assert np.allclose(corrigida, esperada, atol=1e-12)
    # E a diferença para `Θ` é real: sem ela a âncora acima já passaria.
    assert float(np.abs(corrigida - theta).max()) == pytest.approx(0.5)


def test_a_correcao_recusa_formas_que_nao_sao_o_par() -> None:
    """`2Θ - ΘΣΘ` exige duas matrizes quadradas da mesma ordem."""
    with pytest.raises(ValueError, match="mesma matriz quadrada"):
        desenviesa_precisao(np.eye(3), np.eye(4))


# ---------------------------------------------------------------------------
# 2. O PAR CAUSAL — a unidade inteira depende dele
# ---------------------------------------------------------------------------


def test_o_PAR_CAUSAL_da_correcao_de_vies_diverge() -> None:  # noqa: N802
    """Com e sem a correção, sobre a MESMA entrada, o `p` por aresta diverge — e como diverge.

    **O desfecho não é «os números mudam um pouco»**, é que sem a correção não há número nenhum:
    todo par que a penalização zerou sai com `p = 1` exacto. Ver o cabeçalho deste módulo.
    """
    x = padronizada(amostra_de(precisao_em_cadeia(8), n=120, seed=20260918))
    theta, sigma, _lambda = precisao_esparsa(x)

    fora_da_diagonal = ~np.eye(8, dtype=bool)
    zerados = fora_da_diagonal & (theta == 0.0)
    # **A planta aterrou?** Sem zeros exactos não há esparsidade a desenviesar, e o par seria
    # vácuo — a comparação passaria por coincidência de magnitudes. Isto é afirmado ANTES de ler
    # qualquer desfecho.
    assert int(zerados.sum()) > 0, "a penalização não zerou nada: não há o que a correção separe"

    corrigida = desenviesa_precisao(theta, sigma)
    _z_com, p_com = p_por_aresta(corrigida, theta, 120)
    # **O caminho SEM a correção**: `Θ̂` entra onde `T` entraria. É o atalho que R8 proíbe.
    _z_sem, p_sem = p_por_aresta(theta, theta, 120)

    acima = np.triu(np.ones((8, 8), dtype=bool), k=1)
    zerados_acima = acima & zerados
    assert int(zerados_acima.sum()) == 12
    assert int(acima.sum()) == 28

    # **A quantidade que diverge é quantos pares TÊM estimativa**, e ela é exacta: 16 contra 28.
    com_estimativa_sem = int(np.count_nonzero(acima & (theta != 0.0)))
    com_estimativa_com = int(np.count_nonzero(acima & (corrigida != 0.0)))
    assert com_estimativa_sem == 16
    assert com_estimativa_com == 28
    # Cada um dos doze que a penalização apagou volta a ter estimativa. Nenhum sobra.
    assert np.all(corrigida[zerados_acima] != 0.0)

    # Sem a correção: ausência de inferência, e é `1` EXACTO e não «alto».
    assert np.all(p_sem[zerados_acima] == 1.0)
    # Com a correção, dez dos doze ficam abaixo de 1. **Dez e não doze, e o número é medido:** em
    # dois pares a estimativa desenviesada é pequena a ponto de `2·Φ(-|z|)` arredondar para 1,0 em
    # `float64`. Afirmar «todos» aqui seria um critério que o `float64` não sustenta.
    assert int(np.count_nonzero(p_com[zerados_acima] < 1.0)) == 10


def test_a_contagem_de_arestas_a_5_POR_CENTO_nao_separa_os_dois_caminhos() -> None:  # noqa: N802
    """**O achado que torna o atalho invisível, medido:** a 5% os dois caminhos dão 7 arestas.

    Este teste afirma uma COINCIDÊNCIA, e é de propósito. A leitura natural — *"se a correção
    importa, ela tem de mudar a contagem de arestas significativas"* — é **falsa** neste dado: os
    sete pares verdadeiros já são significativos pelos dois caminhos, e os doze que a penalização
    zerou continuam longe do corte depois de desenviesados.

    Então a contagem de arestas **não é detector** do passo ausente, e é exactamente isso que o
    plano nomeia quando diz que *"a rede sai igualmente bonita sem a correção"*. Quem procurasse o
    atalho por aí não o acharia. O detector é o teste acima, sobre **quantos pares têm estimativa**.

    Se um dia esta igualdade deixar de valer, o teste reprova — e a resposta certa é reescrever
    esta docstring, não afrouxar a asserção: o achado é sobre este dado, não é invariante.
    """
    x = padronizada(amostra_de(precisao_em_cadeia(8), n=120, seed=20260918))
    theta, sigma, _lambda = precisao_esparsa(x)
    acima = np.triu(np.ones((8, 8), dtype=bool), k=1)

    _z_com, p_com = p_por_aresta(desenviesa_precisao(theta, sigma), theta, 120)
    _z_sem, p_sem = p_por_aresta(theta, theta, 120)

    com_correcao = int(np.count_nonzero(acima & (p_com < 0.05)))
    sem_correcao = int(np.count_nonzero(acima & (p_sem < 0.05)))
    assert int(acima.sum()) == 28
    assert com_correcao == 7
    assert sem_correcao == 7


# ---------------------------------------------------------------------------
# 3. A rede sobre estrutura conhecida
# ---------------------------------------------------------------------------


def test_a_rede_recupera_as_arestas_verdadeiras_da_cadeia() -> None:
    """Precisão em cadeia: a rede acha os `k-1` pares vizinhos e não inventa outros."""
    resultado = rede_da_cadeia()

    achadas = {(a.feature_a, a.feature_b) for a in resultado.edges}
    verdadeiras = arestas_da_cadeia(8)
    assert resultado.n_pairs_tested == 28
    assert len(resultado.features) == 8
    assert achadas == verdadeiras


def test_o_SINAL_da_correlacao_parcial_e_o_simetrico_da_precisao() -> None:  # noqa: N802
    """Precisão fora da diagonal **negativa** ⇒ correlação parcial **positiva**.

    Um sinal trocado aqui desenharia a rede inteira com ativação e inibição invertidas, e nada mais
    no cálculo o acusaria: os `p` são de `|z|` e não mudariam.
    """
    resultado = rede_da_cadeia()

    assert resultado.edges, "sem aresta não há sinal a conferir"
    # A cadeia tem `-0,45` fora da diagonal, logo toda aresta verdadeira é positiva.
    assert all(a.partial_correlation > 0.0 for a in resultado.edges)
    # E a coerência entre as duas quantidades que partilham o numerador.
    assert all(np.sign(a.partial_correlation) == np.sign(-a.statistic) for a in resultado.edges)


def test_variaveis_INDEPENDENTES_nao_deixam_aresta_sobreviver() -> None:  # noqa: N802
    """Precisão identidade: nenhuma aresta passa o corte. **Vazio é resultado** (R12b).

    A asserção é de ausência, então a de não-vacuidade vem antes e na mesma função: sem
    afirmar que houve pares testados e features retidas, «nenhuma aresta» seria satisfeito por uma
    rede que não correu.
    """
    x = amostra_de(np.eye(10), n=120, seed=4242)
    resultado = dspc_network(matriz_de(x), AnalysisSpec(dspc=True))

    # Não-vacuidade, ANTES: a rede de facto correu sobre uma população.
    assert resultado.n_features_used == 10
    assert resultado.n_pairs_tested == 45
    assert np.isfinite(resultado.pvalue[np.triu_indices(10, k=1)]).all()

    assert resultado.n_edges == 0
    assert resultado.edges == ()


def test_a_contagem_de_arestas_vem_de_DOIS_caminhos() -> None:  # noqa: N802
    """`n_edges` sai de `count_nonzero` e `len(edges)` de um laço. Os dois têm de concordar.

    **A não-vacuidade está na mesma função e antes da igualdade**: dois zeros são iguais, e
    uma igualdade entre eles não afirma que o laço percorreu par nenhum.
    """
    resultado = rede_da_cadeia()

    assert resultado.n_edges > 0
    assert len(resultado.edges) > 0
    assert resultado.n_edges == len(resultado.edges)
    # E o laço leu as células certas: cada aresta corresponde a um par abaixo do alfa.
    for aresta in resultado.edges:
        i = resultado.features.index(aresta.feature_a)
        j = resultado.features.index(aresta.feature_b)
        assert resultado.pvalue[i, j] == pytest.approx(aresta.pvalue)
        assert aresta.pvalue < resultado.alpha


def test_o_alfa_governa_quantas_arestas_sobrevivem() -> None:
    """O corte é no `p` cru contra `dspc_alpha`, e mover o alfa move a contagem na direção certa."""
    apertado = rede_da_cadeia(alpha=1e-6)
    frouxo = rede_da_cadeia(alpha=0.5)

    assert frouxo.n_edges > apertado.n_edges
    assert apertado.alpha == 1e-6


def test_o_q_de_BH_viaja_em_cada_aresta_e_NAO_e_o_corte() -> None:  # noqa: N802
    """`qvalue_fdr` está presente, é `>= p`, e não é o critério de sobrevivência.

    A segunda metade é a que importa: se o corte fosse no `q`, uma aresta com `p < alpha <= q` não
    estaria na lista. O teste afirma que ao menos uma aresta sobreviveu com `q >= alpha`, ou
    declara que o caso não ocorreu — nunca passa por vacuidade.
    """
    resultado = rede_da_cadeia(alpha=0.2)

    assert resultado.edges
    for aresta in resultado.edges:
        assert np.isfinite(aresta.qvalue_fdr)
        assert aresta.qvalue_fdr >= aresta.pvalue - 1e-12
    # A diagonal do `q` não é aresta e não pode ser contada.
    assert np.isnan(np.diag(resultado.pvalue)).all()


def test_a_diagonal_nao_e_ARESTA_em_nenhuma_das_tres_matrizes() -> None:  # noqa: N802
    """`nan` e não `0` na diagonal — um zero ali entraria em qualquer varredura de `p < alpha`.

    É a asserção de ausência vista do lado da **presença** espúria: `k` auto-arestas
    perfeitamente significativas, com a rede bem formada.
    """
    resultado = rede_da_cadeia()

    assert len(resultado.features) == 8
    for matriz in (resultado.pvalue, resultado.partial_correlation, resultado.statistic):
        assert np.isnan(np.diag(matriz)).all()
    assert all(a.feature_a != a.feature_b for a in resultado.edges)


# ---------------------------------------------------------------------------
# 4. Determinismo, imutabilidade e recusas
# ---------------------------------------------------------------------------


def test_duas_execucoes_com_a_MESMA_entrada_dao_saida_identica() -> None:  # noqa: N802
    """Regra 3 do README. Não há semente aqui porque não há sorteio — e é isto que o afirma."""
    primeira = rede_da_cadeia()
    segunda = rede_da_cadeia()

    assert primeira.penalty == segunda.penalty
    assert primeira.edges == segunda.edges
    assert np.array_equal(primeira.pvalue, segunda.pvalue, equal_nan=True)


def test_as_matrizes_do_resultado_sao_IMUTAVEIS() -> None:  # noqa: N802
    """Regra 7 do README na escala do resultado: resultado que se pode mutar depois de lido deixa de
    ser o que a figura declara."""
    resultado = rede_da_cadeia()

    for matriz in (resultado.pvalue, resultado.partial_correlation, resultado.statistic):
        assert not matriz.flags.writeable
        with pytest.raises(ValueError, match="read-only"):
            matriz[0, 1] = 0.0


def test_acima_do_teto_a_recusa_e_TIPADA() -> None:  # noqa: N802
    """As duas direções do teto, na mesma função: acima recusa, abaixo corre.

    **O teto de produção é `None` e portanto não recusa nada** — é o herdado do agrupamento. Este
    teste
    passa um teto explícito, que é o que torna o mecanismo exercitável: sem ele a guarda estaria
    construída e nunca vista a produzir desfecho nenhum.
    """
    x = amostra_de(precisao_em_cadeia(8), n=120, seed=20260918)
    matriz = matriz_de(x)

    with pytest.raises(RedeAcimaDoTetoError, match="8 features e o teto de cálculo é 5"):
        dspc_network(matriz, AnalysisSpec(dspc=True), teto=5)

    # A outra direção: com o teto acima do pedido, a mesma chamada produz rede.
    abaixo = dspc_network(matriz, AnalysisSpec(dspc=True), teto=8)
    assert abaixo.n_features_used == 8


def test_o_teto_de_producao_e_o_HERDADO_de_U10() -> None:  # noqa: N802
    """O padrão do argumento é a constante do agrupamento, e não um número novo.

    Hoje ela vale `None`, então a guarda de produção está **inerte** — e dizê-lo em teste é o que
    impede alguém de ler a existência da recusa como proteção efetiva.
    """
    assert TETO_DE_FEATURES_PARA_COMPUTAR is None
    x = amostra_de(precisao_em_cadeia(6), n=60, seed=11)
    # Sem `teto=`, herda `None` e não recusa.
    assert dspc_network(matriz_de(x), AnalysisSpec(dspc=True)).n_features_used == 6


def test_populacao_pequena_demais_recusa_com_tipo_proprio() -> None:
    """Menos amostras que dobras: a validação cruzada não particiona. Recusa no vocabulário do
    domínio, e não um erro de biblioteca no meio do ajuste."""
    assert MIN_AMOSTRAS_PARA_DSPC == DOBRAS_DA_VALIDACAO_CRUZADA
    x = amostra_de(precisao_em_cadeia(4), n=MIN_AMOSTRAS_PARA_DSPC - 1, seed=5)

    with pytest.raises(RedeSemPopulacaoError, match="amostras de estudo"):
        dspc_network(matriz_de(x), AnalysisSpec(dspc=True))


def test_os_QC_ficam_FORA_da_populacao() -> None:  # noqa: N802
    """`sample_mask` — e a contagem publicada é a das amostras de estudo, não a das linhas."""
    x = amostra_de(precisao_em_cadeia(6), n=70, seed=77)
    tipos = ["qc"] * 10 + ["sample"] * 60

    resultado = dspc_network(matriz_de(x, tipos=tipos), AnalysisSpec(dspc=True))

    assert resultado.n_samples_used == 60


def test_coluna_constante_sai_da_rede_e_e_CONTADA() -> None:  # noqa: N802
    """Uma constante não tem correlação parcial definida com nada: sai, e a contagem viaja."""
    x = amostra_de(precisao_em_cadeia(6), n=60, seed=9)
    com_constante = np.column_stack([x, np.full(60, 3.5)])

    resultado = dspc_network(matriz_de(com_constante), AnalysisSpec(dspc=True))

    assert resultado.n_features_total == 7
    assert resultado.n_features_used == 6
    assert resultado.n_features_dropped_constant == 1
    assert "f006" not in resultado.features


def test_uma_feature_so_nao_e_rede() -> None:
    """Com uma coluna utilizável não há par, e devolver uma rede vazia seria indistinguível de
    «não há aresta»."""
    x = np.column_stack([np.random.default_rng(2).normal(size=60), np.full(60, 1.0)])

    with pytest.raises(RedeSemPopulacaoError, match="ao menos duas colunas"):
        dspc_network(matriz_de(x), AnalysisSpec(dspc=True))


def test_recorte_invalido_recusa_com_a_causa_nomeada() -> None:
    """Índice repetido e índice fora do intervalo são dois defeitos e duas mensagens."""
    matriz = matriz_de(amostra_de(precisao_em_cadeia(6), n=60, seed=13))

    with pytest.raises(ValueError, match="repetido"):
        dspc_network(matriz, AnalysisSpec(dspc=True), colunas=[0, 1, 1])
    with pytest.raises(ValueError, match="fora da matriz"):
        dspc_network(matriz, AnalysisSpec(dspc=True), colunas=[0, 99])


def test_o_recorte_governa_quais_features_entram() -> None:
    """`colunas` é parâmetro de chamada, e a rede sai sobre exactamente o recorte pedido."""
    matriz = matriz_de(amostra_de(precisao_em_cadeia(8), n=120, seed=20260918))

    resultado = dspc_network(matriz, AnalysisSpec(dspc=True), colunas=[0, 1, 2])

    assert resultado.features == ("f000", "f001", "f002")
    assert resultado.n_features_total == 8
    assert resultado.n_pairs_tested == 3


# ---------------------------------------------------------------------------
# 5. Cancelamento
# ---------------------------------------------------------------------------


def _cancela_na_chamada(quando: int) -> CancelCheck:
    """Um verificador que diz «sim» a partir da `quando`-ésima pergunta.

    **Contado e não constante `True`**, porque um `True` constante levantaria no primeiro ponto de
    parada e não distinguiria os pontos posteriores — a rede pararia sempre no mesmo lugar, e o
    teste não saberia se os outros pontos existem.
    """
    estado = {"n": 0}

    def verifica() -> bool:
        estado["n"] += 1
        return estado["n"] >= quando

    return verifica


def test_o_cancelamento_para_a_rede_em_cada_ponto_declarado() -> None:
    """Os pontos de parada existem e **cada um** levanta. R18 na assinatura do domínio.

    A direção positiva vem antes: com o verificador a dizer sempre «não», a mesma chamada produz
    rede. Sem ela, um `raise` incondicional passaria neste teste.
    """
    matriz = matriz_de(amostra_de(precisao_em_cadeia(8), n=120, seed=20260918))

    # Direção que NÃO cancela — a planta removida.
    intacta = dspc_network(matriz, AnalysisSpec(dspc=True), cancelled=lambda: False)
    assert intacta.n_edges > 0

    vistos: set[str] = set()
    for quando in (1, 2, 3, 4):
        with pytest.raises(ExecutionCancelledError) as erro:
            dspc_network(matriz, AnalysisSpec(dspc=True), cancelled=_cancela_na_chamada(quando))
        vistos.add(str(erro.value))

    # Quatro perguntas atingiram quatro pontos distintos: o mecanismo não é um `raise` no topo.
    assert len(vistos) == 4
    assert any("ajuste esparso" in onde for onde in vistos)
    assert any("aresta da linha" in onde for onde in vistos)


def test_sem_verificador_a_rede_corre_igual() -> None:
    """`cancelled=None` é o caminho de produção da CLI, e não muda o resultado."""
    matriz = matriz_de(amostra_de(precisao_em_cadeia(6), n=60, seed=31))

    com = dspc_network(matriz, AnalysisSpec(dspc=True), cancelled=lambda: False)
    sem = dspc_network(matriz, AnalysisSpec(dspc=True), cancelled=None)

    assert com.edges == sem.edges


# ---------------------------------------------------------------------------
# 6. O `p` por aresta — a conta, isolada
# ---------------------------------------------------------------------------


def test_p_por_aresta_recusa_n_nao_positivo() -> None:
    """`√n` no numerador: sem amostra não há estatística."""
    with pytest.raises(ValueError, match="`n_amostras`"):
        p_por_aresta(np.eye(2), np.eye(2), 0)


def test_p_por_aresta_cresce_com_n_para_o_MESMO_efeito() -> None:  # noqa: N802
    """A mesma estimativa com mais amostras dá `p` menor. É o `√n`, e é o que a assintótica promete.

    Sem esta âncora, um `n` esquecido no numerador produziria `p` que não responde ao tamanho da
    amostra — e nenhum dos outros testes o notaria, porque todos correm com um `n` só.
    """
    corrigida = np.array([[1.0, 0.2], [0.2, 1.0]])
    precisao = np.array([[1.0, 0.0], [0.0, 1.0]])

    _z30, p30 = p_por_aresta(corrigida, precisao, 30)
    _z300, p300 = p_por_aresta(corrigida, precisao, 300)

    assert p30[0, 1] > p300[0, 1]

    # **A âncora é o quantil de LIVRO, e não uma conta minha.** Com `Θ = I` a variância é 1, então
    # `z = √n · T`. Escolhendo `T = 1,959964/√n`, o `z` é o quantil bilateral de 5% e o `p` tem de
    # sair `0,05` — o único número desta família que se confere sem calculadora.
    #
    # A primeira versão deste teste comparava contra `Φ(1,0954) = 0,863246` escrito de cabeça, e
    # **reprovou**: o valor certo é `0,863339`. Foi o número à mão — um valor
    # tabelado de memória, que ninguém confere porque parece evidência.
    no_quantil = np.array([[1.0, 1.959964 / np.sqrt(100.0)], [1.959964 / np.sqrt(100.0), 1.0]])
    _z, p_no_quantil = p_por_aresta(no_quantil, precisao, 100)
    assert p_no_quantil[0, 1] == pytest.approx(0.05, abs=1e-7)
