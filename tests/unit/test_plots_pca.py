"""As quatro figuras de PCA, e o par causal que fixa QUAL das duas PCAs as alimenta.

**Não é `test_pca.py`, que já existe em `tests/unit/` e é da órfã.** Os dois nomes descrevem coisas
diferentes e a coincidência seria a pior possível: um arquivo chamado `test_pca.py` a testar a PCA
que **não** corre, ao lado de outro a testar as figuras da que corre.

## O que este arquivo prova, e por que a primeira seção é a mais importante

O instrumento pode ler o referente certo e a **instância** errada dele. Aqui há
duas PCAs, as duas devolvem `(n_amostras, 2)` de `float`, e uma figura construída da errada é uma
figura bem formada com o número errado. A primeira seção é o par causal que fixa qual é qual:

1. as duas **divergem** sobre a mesma matriz real — sem isto, tudo o resto passa por coincidência;
2. o que a figura desenha **é** o da que corre, e **não** é o da órfã;
3. a órfã continua sem chamador de produção, e o dia em que ganhar um obriga a decidir.

As três vivem no mesmo arquivo de propósito: um teste vizinho que ligue as pontas pode ser
removido sem que nada ligue os dois.
"""

from __future__ import annotations

import ast
import math
import re
from pathlib import Path

import numpy as np
import pytest
from matplotlib.axes import Axes
from matplotlib.colors import to_hex
from matplotlib.figure import Figure
from scipy.stats import chi2
from scipy.stats import f as distribuicao_f

from refino.diagnosis.profile import PRETRANSFORM
from refino.domain.matrix import RefinoMatrix
from refino.domain.metadata import SampleMetadata
from refino.metrics.structure import structural_metrics, structural_pca
from refino.plots import pca
from refino.plots.base import MatrizVaziaError, cores_do_ciclo, nova_figura
from refino.plots.catalogo import CONSTRUTORES, Vinculo
from refino.plots.pca import (
    ALFA_DA_ELIPSE,
    COLUNAS_DO_MOTIVO,
    DIMENSAO_DA_ELIPSE,
    MINIMO_PARA_ELIPSE,
    NOTA_MAIOR_QUE_O_QUADRO,
    NOTA_SEM_REGIAO,
    PERMANOVA_AUSENTE,
    TITULO_DAS_CARGAS,
    TITULO_DO_BIPLOT,
    TITULO_DO_SCREE,
    TITULO_DOS_SCORES,
    Cargas,
    Elipse,
    ElipseAusente,
    Escores,
    PcaDesalinhadaError,
    Permanova,
    RegioesDosGrupos,
    TipoDeAmostra,
    cargas_da_pca,
    elipse_das_amostras,
    elipse_de_hotelling,
    escores_da_pca,
    figura_das_cargas,
    figura_do_biplot,
    figura_do_scree,
    figura_dos_scores,
    legenda_das_setas,
    permanova_do_fio,
    recorte_das_cargas,
    recorte_do_biplot,
    recorte_do_scree,
    recorte_dos_scores,
    regiao_dos_scores,
    regioes_dos_grupos,
    rotulo_da_regiao_do_grupo,
    rotulo_do_componente,
    setas_do_biplot,
)
from refino.plots.recorte import RecorteInvalidoError, aplica, decimal, milhar
from refino.plots.series import (
    CORES_DOS_GRUPOS,
    SIMBOLOS_DOS_GRUPOS,
    EstiloDoGrupo,
    EstilosDosGrupos,
    GrupoInvalidoError,
    estilos_dos_grupos,
    rotulo_do_grupo,
)
from refino.reporting.model import ITENS_OBRIGATORIOS
from tests.arnes_de_figura import (
    confere_numeros,
    confere_rotulos,
    confere_series,
    confere_vista,
    planta_corte_de_eixo,
    planta_eixos_trocados,
    planta_serie_removida,
    planta_sinal_invertido,
    series,
)
from tests.conftest import matrix_from_toserrf

REPO = Path(__file__).resolve().parents[2]

MODO = "pos_hilic"


# ---------------------------------------------------------------------------
# As fixturas: o corpus real, e uma matriz sintética de estrutura conhecida
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def matriz_real() -> RefinoMatrix:
    """O `pos_hilic` do laboratório. **Escopo de módulo: a carga custa 0,35 s e a PCA 0,03 s.**"""
    return matrix_from_toserrf(MODO)


def matriz_sintetica(
    *,
    n_amostras: int = 12,
    n_features: int = 6,
    com_constante: bool = False,
    com_validacao: bool = False,
    seed: int = 20260911,
) -> RefinoMatrix:
    """Uma matriz de estrutura **conhecida**: um eixo domina, e o resto é ruído.

    A primeira feature carrega um sinal forte e as outras ruído pequeno, então PC1 tem de explicar
    a maior parte da variância — que é o oráculo do caminho feliz. Os valores são positivos porque
    a pré-transformação é `log10(x + 1)`.
    """
    # **O sinal é PARTILHADO por todas as features, e a primeira versão desta fixtura não era.**
    # Ela punha o sinal numa coluna só, e o passo 4 autoescalona: depois disso a coluna forte fica
    # com variância 1 como as outras, e PC1 medido deu **0,3815** — que é o que seis colunas de
    # ruído dão com doze linhas. A estrutura tem de estar na CORRELAÇÃO entre colunas, que é a
    # única coisa que o autoescalonamento não apaga.
    rng = np.random.default_rng(seed)
    latente = np.linspace(-1.0, 1.0, n_amostras)
    pesos = rng.uniform(0.5, 1.5, n_features)
    valores = (
        100.0
        + 30.0 * np.outer(latente, pesos)
        + rng.standard_normal((n_amostras, n_features)) * 0.5
    )
    if com_constante:
        valores[:, -1] = 7.0

    tipos = ["qc" if i % 4 == 0 else "sample" for i in range(n_amostras)]
    if com_validacao:
        tipos[1] = "validate"
    amostras = tuple(
        SampleMetadata(
            name=f"a{i:02d}", injection_order=i + 1, sample_type=tipos[i], group=f"g{i % 2}"
        )
        for i in range(n_amostras)
    )
    return RefinoMatrix(
        values=valores,
        feature_names=tuple(f"f{j:02d}" for j in range(n_features)),
        samples=amostras,
        mode=MODO,
    )


def rodape_de(figura: Figure) -> str:
    """O texto do rodapé com o espaço em branco COLAPSADO — e isto não é conveniência.

    `recorte.aplica` quebra a frase em linhas pela largura útil da figura, então uma citação de
    meia dúzia de palavras atravessa a quebra e `in` devolve `False` sobre texto verdadeiro. É um
    falso negativo dentro de um teste: a primeira versão de
    `test_o_rodape_do_biplot_MAPEIA_numero_para_feature` reprovou por isto, com o rodapé correcto.
    """
    return " ".join(figura.texts[0].get_text().split())


def figuras_de(matriz: RefinoMatrix, *, n_componentes: int = 4, n_setas: int = 3) -> dict:
    """As quatro figuras e as populações que as produziram, de uma execução da PCA que CORRE."""
    p = structural_pca(matriz, pretransform=PRETRANSFORM, n_components=n_componentes)
    escores = escores_da_pca(
        pc1=p.scores[:, 0], pc2=p.scores[:, 1], tipos=[s.sample_type for s in matriz.samples]
    )
    cargas = cargas_da_pca(
        pc1=p.loadings[:, 0], pc2=p.loadings[:, 1], features=list(p.feature_names_used)
    )
    mascara = escores.mascara(TipoDeAmostra.AMOSTRA)
    regiao = elipse_de_hotelling(escores.x[mascara], escores.y[mascara])
    return {
        "pca": p,
        "escores": escores,
        "cargas": cargas,
        "regiao": regiao,
        "scree": figura_do_scree(
            variancia_explicada=p.explained_variance_ratio,
            recorte=recorte_do_scree(
                n_features_usadas=p.n_features_used, n_features_totais=matriz.n_features
            ),
        ),
        "scores": figura_dos_scores(
            escores=escores,
            variancia_explicada=p.explained_variance_ratio,
            regiao=regiao,
            recorte=recorte_dos_scores(
                n_features_usadas=p.n_features_used,
                n_features_totais=matriz.n_features,
                regiao=regiao,
            ),
        ),
        "cargas_fig": figura_das_cargas(
            cargas=cargas,
            variancia_explicada=p.explained_variance_ratio,
            recorte=recorte_das_cargas(cargas=cargas, n_features_totais=matriz.n_features),
        ),
        "biplot": figura_do_biplot(
            escores=escores,
            cargas=cargas,
            n_setas=n_setas,
            variancia_explicada=p.explained_variance_ratio,
            recorte=recorte_do_biplot(
                escores=escores,
                cargas=cargas,
                n_setas=n_setas,
                n_features_totais=matriz.n_features,
            ),
        ),
    }


# ---------------------------------------------------------------------------
# 1. O par causal das DUAS PCAs
# ---------------------------------------------------------------------------


def test_ktd7_as_duas_pcas_DIVERGEM_sobre_a_mesma_matriz(matriz_real: RefinoMatrix) -> None:  # noqa: N802
    """**A âncora de separação: a planta cai sobre dado em que as duas coisas DIFEREM.**

    Sem esta afirmação, o teste seguinte — *"a figura desenha o da que corre"* — passaria por
    coincidência se as duas PCAs concordassem, e trocar a fonte não seria detectável. Aqui a
    divergência é medida antes de qualquer figura existir.

    **O que se afirma é a ORDEM DE GRANDEZA e a DIREÇÃO, e não um número exacto.** Números exactos
    de SVD sobre 6.035 colunas não são o alvo: o alvo é que as duas não sejam a mesma coisa. As
    duas metades juntas — escala e correlação — é que fecham, porque cada uma sozinha tem
    explicação inocente (escala é normalização; correlação baixa num componente é ruído).
    """
    from refino.statistics.pca import pca as pca_orfa

    corre = structural_pca(matriz_real, pretransform=PRETRANSFORM, n_components=2)
    orfa = pca_orfa(matriz_real, n_components=2)

    assert corre.scores.shape == orfa.scores.shape, (
        "as duas têm de devolver a mesma FORMA — é isso que torna a troca indetectável a olho, e "
        "portanto é isso que este par causal existe para cobrir"
    )

    escala_corre = float(np.abs(corre.scores).max())
    escala_orfa = float(np.abs(orfa.scores).max())
    assert escala_orfa > 1e4 * escala_corre, (
        f"a que corre chega a {escala_corre:.3g} e a órfã a {escala_orfa:.3g}: sem o "
        "autoescalonamento, a órfã fica em unidades de intensidade bruta. Se estes números "
        "convergirem, é porque uma das duas mudou de pré-tratamento"
    )

    pc2 = float(np.corrcoef(corre.scores[:, 1], orfa.scores[:, 1])[0, 1])
    assert abs(pc2) < 0.5, (
        f"o PC2 das duas correlaciona {pc2:.4f}: elas deixaram de ser decomposições diferentes, e "
        "a figura passa a não distinguir de qual saiu"
    )


def test_ktd7_a_figura_desenha_a_pca_que_CORRE_e_nao_a_orfa(matriz_real: RefinoMatrix) -> None:  # noqa: N802
    """A outra metade do par: o que está DESENHADO é o da que corre, ponto a ponto.

    **A asserção é sobre o dado do artista** (classe (a) do arnês) e não sobre o que o construtor
    recebeu: um construtor que ignorasse a entrada e recalculasse passaria na segunda e reprovaria
    nesta.
    """
    from refino.statistics.pca import pca as pca_orfa

    tudo = figuras_de(matriz_real, n_componentes=2)
    corre = tudo["pca"]
    orfa = pca_orfa(matriz_real, n_components=2)

    desenhados = np.concatenate([s.x for s in series(tudo["scores"].axes[0]) if s.x.size == 7])
    dos_qcs = corre.scores[matriz_real.qc_mask, 0]
    assert np.allclose(np.sort(desenhados), np.sort(dos_qcs)), (
        "o x da série de QC não é o PC1 de `structural_pca`. Ou o construtor recalculou, ou a "
        "fonte mudou"
    )
    assert not np.allclose(np.sort(desenhados), np.sort(orfa.scores[matriz_real.qc_mask, 0])), (
        "o que está desenhado bate com a PCA ÓRFÃ: o referente certo, a instância errada"
    )


def test_ktd7_a_pca_orfa_continua_SEM_CHAMADOR_DE_PRODUCAO() -> None:  # noqa: N802
    """O destino da órfã, mecanizado — porque *atenção não é defesa*.

    **A órfã fica registada como órfã, com o motivo, e não ganha chamador nesta unidade.** Ela é o
    entregável para a matriz **já processada**, e nenhuma rota
    expõe PCA da processada. Dar-lhe um chamador agora seria escolher **de qual matriz** as figuras
    falam, e elas falam da ingerida — a que `diagnose` lê e a que a aba de qualidade mostra.

    **Por que um portão e não um comentário.** É da medição *"zero chamadores de
    produção"* que sai a escolha da fonte. Um comentário caduca em silêncio;
    este portão reprova no dia em que a medição deixar de valer, e a mensagem diz o que
    decidir — não *"desfaça"*.
    """
    achados = importadores_da_orfa(REPO / "src")
    assert achados == [], (
        f"`refino.statistics.pca` ganhou chamador de produção: {achados}. Isso não é erro — é a "
        "condição que KTD7 media. Decida de qual matriz as figuras de PCA falam "
        "e actualize este portão. Ver o cabeçalho de `plots/pca.py`"
    )


def importadores_da_orfa(raiz: Path) -> list[str]:
    """Os módulos sob `raiz` que importam `refino.statistics.pca`, por AST.

    **Por AST e não por `grep`, para este portão não ler a instância errada:** `grep` acharia a
    string dentro de uma docstring — e as docstrings deste projeto citam `statistics/pca.py` várias
    vezes, incluindo a do módulo que este teste protege. O que interessa é o `import`.
    """
    achados: list[str] = []
    for caminho in sorted(raiz.rglob("*.py")):
        arvore = ast.parse(caminho.read_text(encoding="utf-8"))
        for no in ast.walk(arvore):
            if (isinstance(no, ast.ImportFrom) and no.module == "refino.statistics.pca") or (
                isinstance(no, ast.Import)
                and any(a.name == "refino.statistics.pca" for a in no.names)
            ):
                achados.append(f"{caminho.relative_to(raiz)}:{no.lineno}")
    return achados


def test_o_portao_da_orfa_ACUSA_a_planta(tmp_path: Path) -> None:  # noqa: N802
    """Par causal do portão acima, em processo: a planta é a árvore que o extractor recebe.

    **As duas direções**, e a de baixo é a que impede o portão de acusar sempre: um módulo que só
    **mencione** a órfã numa docstring — que é o caso de `plots/pca.py` — não é chamador.
    """
    (tmp_path / "limpo.py").write_text(
        '"""Fala de `refino.statistics.pca` na prosa, e não a importa."""\n', encoding="utf-8"
    )
    assert importadores_da_orfa(tmp_path) == [], "o portão acusa uma menção em docstring"

    (tmp_path / "sujo.py").write_text("from refino.statistics.pca import pca\n", encoding="utf-8")
    assert [a.split(":")[0] for a in importadores_da_orfa(tmp_path)] == ["sujo.py"]


# ---------------------------------------------------------------------------
# 2. A elipse de Hotelling T² — o instrumento validado contra resposta conhecida
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", [7, 13, 30])
def test_ktd11_a_cobertura_da_elipse_bate_o_nominal_e_a_qui_quadrado_NAO(n: int) -> None:  # noqa: N802
    """**A âncora barata: alimentar o instrumento com algo cuja resposta já se sabe.**

    A população é normal bivariada com covariância **não diagonal** — a diagonal deixaria a elipse
    alinhada aos eixos, e um erro de rotação passaria. Para cada réplica: estimar média e
    covariância de `n` pontos, sortear **um ponto novo**, e ver se ele cai dentro.

    **As duas direções na mesma função**: a de Hotelling acerta o nominal, e a de
    qui-quadrado **não** acerta. Sem a segunda, este teste diria *"a nossa elipse cobre 95%"* sem
    dizer nada sobre o que ela melhora.

    A tolerância de 0,01 sobre 6.000 réplicas é folgada: o erro padrão de uma proporção de 0,95 com
    6.000 amostras é 0,0028, então 0,01 é 3,5 erros padrão. Semente fixa (regra 3 do README).

    > **Três `n`, e os dois novos são os pequenos.** Esta é a função que o painel da
    > correção passou a desenhar em cada lado, e é com `n` pequeno que a aproximação de qui-quadrado
    > erra mais: sete é o número de QCs dos quatro arquivos deste estudo, e treze está na faixa de
    > 13 a 18 em que o dossiê do plano a mediu cobrindo 0,88 a 0,90. Medido com esta semente,
    > Hotelling cobre **0,9558 / 0,9545 / 0,9512** com n = 7 / 13 / 30, e qui-quadrado **0,8025 /
    > 0,8815 / 0,9222** — contra a cobertura exacta de qui-quadrado sob normalidade, **0,7919 /
    > 0,8769 / 0,9221**. A planta é a região de qui-quadrado sob o MESMO critério de 0,01, e ela
    > reprova nos três; com n = 7 fica a 0,1475 do nominal, quase quinze vezes a tolerância.
    >
    > **E o arnês mede o que diz medir:** a cobertura exacta do raio de qui-quadrado é
    > conhecida — `F(c²_qui / k; p, n - p)`, com `k` o fator de predição — e a simulação tem de
    > a reproduzir. É o outro caminho como oráculo: sem ele, uma simulação que errasse a
    > região de qui-quadrado tornaria a planta vácua, e a primeira metade passaria sozinha.
    """
    rng = np.random.default_rng(20260911)
    sigma = np.array([[4.0, 1.5], [1.5, 1.0]])
    cholesky = np.linalg.cholesky(sigma)
    mu = np.array([3.0, -2.0])
    replicas = 6000
    p = DIMENSAO_DA_ELIPSE
    nominal = 1.0 - ALFA_DA_ELIPSE

    dentro_hotelling = 0
    dentro_qui = 0
    corte_qui = float(chi2.ppf(nominal, p))
    for _ in range(replicas):
        pontos = mu + rng.standard_normal((n, p)) @ cholesky.T
        regiao = elipse_de_hotelling(pontos[:, 0], pontos[:, 1])
        assert isinstance(regiao, Elipse)
        novo = mu + rng.standard_normal(p) @ cholesky.T
        desvio = novo - np.asarray(regiao.centro)
        d2 = float(desvio @ np.linalg.inv(np.cov(pontos, rowvar=False)) @ desvio)
        dentro_hotelling += d2 <= regiao.c2
        dentro_qui += d2 <= corte_qui

    cobertura = dentro_hotelling / replicas
    assert abs(cobertura - nominal) < 0.01, (
        f"a elipse de Hotelling cobre {cobertura:.4f} com n = {n} e devia cobrir {nominal:.2f}. "
        "O quantil está errado"
    )
    cobertura_qui = dentro_qui / replicas
    assert abs(cobertura_qui - nominal) >= 0.01, (
        f"a região de qui-quadrado cobriu {cobertura_qui:.4f} com n = {n} e passa no critério em "
        "que a de Hotelling passa: a planta não reprova, e este teste não separa as duas"
    )
    assert cobertura_qui < 0.94, (
        f"a aproximação de qui-quadrado cobriu {cobertura_qui:.4f} com n = {n}, e devia "
        "SUBESTIMAR a região. Se ela também acerta, o argumento de KTD11 não vale para este n e a "
        "escolha precisa de outra justificativa"
    )
    fator_de_predicao = p * (n + 1) * (n - 1) / (n * (n - p))
    exacta_qui = float(distribuicao_f.cdf(corte_qui / fator_de_predicao, p, n - p))
    assert abs(cobertura_qui - exacta_qui) < 0.02, (
        f"a simulação mediu {cobertura_qui:.4f} para a região de qui-quadrado com n = {n}, e a "
        f"cobertura exacta é {exacta_qui:.4f}: o arnês não mede a região que diz medir"
    )


def test_o_quantil_da_elipse_e_o_de_PREDICAO_e_nao_o_da_media() -> None:  # noqa: N802
    """O `c²` conferido contra a fórmula, derivada aqui e não copiada do construtor (classe (d))."""
    rng = np.random.default_rng(7)
    pontos = rng.standard_normal((25, 2))
    regiao = elipse_de_hotelling(pontos[:, 0], pontos[:, 1])
    assert isinstance(regiao, Elipse)

    n, p = 25, 2
    predicao = p * (n + 1) * (n - 1) / (n * (n - p)) * float(distribuicao_f.ppf(0.95, p, n - p))
    da_media = p * (n - 1) / (n - p) * float(distribuicao_f.ppf(0.95, p, n - p)) / n
    assert regiao.c2 == pytest.approx(predicao)
    assert regiao.c2 > da_media * 10.0, (
        "o quantil caiu na região da MÉDIA, que é uma mancha à volta do centroide — num gráfico de "
        "dispersão ela faria quase todo ponto parecer discrepante"
    )


def test_o_contorno_da_elipse_satisfaz_a_forma_quadratica() -> None:
    """Todo vértice desenhado tem `d² == c²`. **É a ponte entre a álgebra e o DESENHO.**

    Um erro de rotação — trocar os autovetores, ou tomar o ângulo do autovalor menor — não muda o
    `c²` nem os semieixos, e nenhuma asserção sobre eles o veria. Esta vê: um contorno rodado
    errado deixa de satisfazer a forma quadrática.
    """
    rng = np.random.default_rng(11)
    pontos = rng.standard_normal((40, 2)) @ np.array([[2.0, 0.0], [1.4, 0.6]])
    regiao = elipse_de_hotelling(pontos[:, 0], pontos[:, 1])
    assert isinstance(regiao, Elipse)

    inversa = np.linalg.inv(np.cov(pontos, rowvar=False))
    x, y = regiao.contorno()
    desvios = np.column_stack((x, y)) - np.asarray(regiao.centro)
    d2 = np.einsum("ij,jk,ik->i", desvios, inversa, desvios)
    assert np.allclose(d2, regiao.c2), (
        f"os vértices variam entre {d2.min():.6f} e {d2.max():.6f} contra c² = {regiao.c2:.6f}: o "
        "contorno não é a curva de nível que o quantil define"
    )
    assert x[0] == pytest.approx(x[-1]) and y[0] == pytest.approx(y[-1]), (
        "o polígono não fecha — uma fenda no contorno lê-se como um ponto que escapa"
    )


@pytest.mark.parametrize(
    ("pontos", "trecho"),
    [
        (np.array([[0.0, 0.0], [1.0, 1.0]]), "são precisos ao menos"),
        (np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]), "singular"),
    ],
    ids=["poucos-pontos", "covariancia-singular"],
)
def test_a_elipse_declara_os_DOIS_motivos_de_ausencia(pontos: np.ndarray, trecho: str) -> None:  # noqa: N802
    """Na escala de um retorno, *"poucos pontos"* e *"colineares"* são fatos distintos.

    Os dois têm conserto diferente — o primeiro é o desenho do estudo, o segundo é o dado — e um
    `None` para os dois tornaria as duas causas indistinguíveis na frase que a figura imprime.
    """
    regiao = elipse_de_hotelling(pontos[:, 0], pontos[:, 1])
    assert isinstance(regiao, ElipseAusente)
    assert trecho in regiao.motivo, regiao.motivo


def test_a_elipse_ignora_ponto_nao_finito_em_vez_de_o_propagar() -> None:
    """`nan` numa coordenada não pode virar `nan` na covariância: a elipse sumiria inteira."""
    rng = np.random.default_rng(3)
    pontos = rng.standard_normal((20, 2))
    com_buraco = np.vstack([pontos, [np.nan, 1.0], [2.0, np.inf]])
    regiao = elipse_de_hotelling(com_buraco[:, 0], com_buraco[:, 1])
    assert isinstance(regiao, Elipse)
    assert regiao.n == 20, f"a elipse contou {regiao.n} pontos e só 20 são finitos"


def test_o_alfa_fora_do_intervalo_e_recusa() -> None:
    rng = np.random.default_rng(5)
    pontos = rng.standard_normal((20, 2))
    with pytest.raises(ValueError, match="alfa"):
        elipse_de_hotelling(pontos[:, 0], pontos[:, 1], alfa=1.5)


# ---------------------------------------------------------------------------
# 3. O caminho feliz das quatro, com estrutura conhecida
# ---------------------------------------------------------------------------


def test_a_variancia_de_pc1_bate_a_estrutura_PLANTADA() -> None:  # noqa: N802
    """Caminho feliz com oráculo: numa matriz em que um eixo domina, PC1 tem de o capturar.

    **O número esperado é derivado da construção**, e não do construtor: a matriz tem uma feature
    com sinal forte e as outras com ruído, então PC1 explica a maior parte — e o *scree* declara
    essa fração na imagem.
    """
    tudo = figuras_de(matriz_sintetica())
    fracoes = tudo["pca"].explained_variance_ratio
    assert fracoes[0] > 0.8, f"PC1 explica {fracoes[0]:.4f} numa matriz com um eixo dominante"
    assert sum(fracoes) <= 1.0 + 1e-9
    assert fracoes[0] > fracoes[1], "o scree tem de sair decrescente — o SVD ordena por s"


def test_o_scree_desenha_a_individual_e_a_ACUMULADA(matriz_real: RefinoMatrix) -> None:  # noqa: N802
    """As duas séries, e a acumulada é a soma corrida da individual — derivada pelo teste."""
    tudo = figuras_de(matriz_real, n_componentes=5)
    fracoes = np.asarray(tudo["pca"].explained_variance_ratio)
    eixo = tudo["scree"].axes[0]
    componentes = np.arange(1, fracoes.size + 1, dtype=float)
    assert (
        confere_series(
            eixo,
            {
                "variância do componente": (componentes, fracoes),
                "variância acumulada": (componentes, np.cumsum(fracoes)),
            },
        )
        == []
    )
    assert confere_vista(eixo) == []


@pytest.mark.parametrize(
    "chave",
    ["scree", "scores", "cargas_fig", "biplot"],
    ids=["scree", "scores", "cargas", "biplot"],
)
def test_as_quatro_passam_nas_classes_de_vista_e_de_rotulo(
    matriz_real: RefinoMatrix, chave: str
) -> None:
    """Classe (b) e classe (c) nas quatro: nenhum eixo sem rótulo, nenhuma série fora da vista."""
    tudo = figuras_de(matriz_real, n_componentes=4)
    eixo = tudo[chave].axes[0]
    assert confere_vista(eixo) == [], chave
    assert eixo.get_xlabel() and eixo.get_ylabel() and eixo.get_title(), chave
    legenda = eixo.get_legend()
    assert legenda is not None and legenda.get_texts(), f"{chave} sem legenda"


# ---------------------------------------------------------------------------
# 4. O QC como série própria — a leitura que a fonte primária regista
# ---------------------------------------------------------------------------


def test_a_serie_de_QC_e_ENDERECAVEL_e_a_contagem_bate_a_mascara(matriz_real: RefinoMatrix) -> None:  # noqa: N802
    """**O único cenário desta unidade sobre o corpus, e o que ele afirma é `[14:23]`.**

    *"Espera-se o QC agrupado no meio da nuvem de amostras."* Para essa leitura ser possível a
    figura tem de ter o QC como série **própria** — não como cor dentro de um artista único —, e a
    legenda tem de dizer quantos são. As três metades: a série existe com o rótulo certo, o dado
    dela é o dos QCs de `qc_mask`, e a contagem na legenda bate.

    **A contagem esperada vem de `qc_mask` e não de um literal**: sete QCs é observação
    sobre estes quatro arquivos.
    """
    tudo = figuras_de(matriz_real, n_componentes=2)
    n_qc = int(matriz_real.qc_mask.sum())
    n_amostra = int(matriz_real.sample_mask.sum())
    n_validate = int(matriz_real.validate_mask.sum())
    assert n_qc and n_amostra, "âncora: sem QC ou sem amostra as asserções abaixo seriam vácuas"

    eixo = tudo["scores"].axes[0]
    por_rotulo = {s.rotulo: s for s in series(eixo)}
    rotulo_qc = f"QC — {milhar(n_qc)} amostras"
    assert rotulo_qc in por_rotulo, (
        f"a série de QC não é endereçável por rótulo próprio; há {sorted(por_rotulo)}"
    )

    scores = tudo["pca"].scores
    assert np.allclose(np.sort(por_rotulo[rotulo_qc].x), np.sort(scores[matriz_real.qc_mask, 0]))
    assert np.allclose(np.sort(por_rotulo[rotulo_qc].y), np.sort(scores[matriz_real.qc_mask, 1]))

    legenda = tuple(t.get_text() for t in eixo.get_legend().get_texts())
    assert rotulo_qc in legenda, legenda
    assert f"amostra de estudo — {milhar(n_amostra)} amostras" in legenda, legenda

    # **A ausência É a informação**, e é por isso que a asserção é de ausência com âncora ao lado:
    # nestes quatro arquivos não há amostra `validate` — observação, não invariante.
    assert n_validate == 0, (
        "este corpus ganhou amostras `validate`; a asserção abaixo mudou de sentido"
    )
    assert not any("validação" in t for t in legenda), (
        f"a legenda promete uma série de validação e não há nenhuma amostra `validate`: {legenda}"
    )


def test_a_serie_de_validacao_APARECE_quando_ha_populacao() -> None:  # noqa: N802
    """O par causal da asserção de ausência acima: com `validate` no dado, a série entra.

    Sem esta metade, *"a legenda não promete validação"* passaria também num construtor que nunca
    soubesse desenhá-la — que é a asserção de ausência satisfeita por vacuidade.
    """
    tudo = figuras_de(matriz_sintetica(com_validacao=True))
    legenda = tuple(t.get_text() for t in tudo["scores"].axes[0].get_legend().get_texts())
    assert any("amostra de validação — 1 amostra" in t for t in legenda), legenda


def test_o_biplot_tambem_separa_as_tres_series(matriz_real: RefinoMatrix) -> None:
    """A mesma leitura no *biplot*: o QC continua identificável ao lado das setas."""
    tudo = figuras_de(matriz_real, n_componentes=2, n_setas=5)
    rotulos = {s.rotulo for s in series(tudo["biplot"].axes[0])}
    assert f"QC — {milhar(int(matriz_real.qc_mask.sum()))} amostras" in rotulos, sorted(rotulos)


def test_o_tipo_de_amostra_DESCONHECIDO_e_recusa_e_nao_recuo() -> None:  # noqa: N802
    """A guarda de `Literal` na fronteira.

    Um `sampleType` inválido que caísse num `else` razoável tiraria a amostra das três séries: a
    legenda contaria menos do que a matriz tem, e nada levantaria.
    """
    with pytest.raises(PcaDesalinhadaError, match="não é um dos três"):
        escores_da_pca(pc1=[1.0, 2.0], pc2=[3.0, 4.0], tipos=["qc", "controle"])


@pytest.mark.parametrize(
    ("argumentos", "trecho"),
    [
        ({"pc1": [], "pc2": [], "tipos": []}, "não veio score nenhum"),
        ({"pc1": [1.0, 2.0], "pc2": [1.0], "tipos": ["qc", "qc"]}, "de PC1 para"),
        ({"pc1": [1.0, 2.0], "pc2": [1.0, 2.0], "tipos": ["qc"]}, "tipo(s) de amostra"),
    ],
    ids=["vazio", "pc-desalinhado", "tipos-desalinhados"],
)
def test_os_escores_recusam_entrada_desalinhada(argumentos: dict, trecho: str) -> None:
    with pytest.raises((MatrizVaziaError, PcaDesalinhadaError), match=re.escape(trecho)):
        escores_da_pca(**argumentos)


@pytest.mark.parametrize(
    ("argumentos", "trecho"),
    [
        ({"pc1": [], "pc2": [], "features": []}, "não veio carga nenhuma"),
        ({"pc1": [0.1, 0.2], "pc2": [0.3], "features": ["f0", "f1"]}, "carga(s) de PC1 para"),
        ({"pc1": [0.1, 0.2], "pc2": [0.3, 0.4], "features": ["f0"]}, "nome"),
    ],
    ids=["vazio", "pc-desalinhado", "nomes-desalinhados"],
)
def test_as_cargas_recusam_as_MESMAS_tres_entradas_que_os_escores(  # noqa: N802
    argumentos: dict, trecho: str
) -> None:
    """As três guardas de `cargas_da_pca`, e não só a dos nomes.

    `cargas_da_pca` tem exactamente as mesmas três guardas de fronteira que a irmã
    `escores_da_pca` — vazio, comprimentos de PC1/PC2, comprimento dos nomes —, e a irmã tinha as
    três cobertas por `test_os_escores_recusam_entrada_desalinhada` enquanto esta tinha uma. Um
    erro de um por um em qualquer das duas primeiras passava a suíte inteira sem nada acusar.

    O terceiro caso é o original e fica com a redação dele: o nome errado num ponto é a
    identificação de R2 a devolver o vizinho — sem nada levantar.
    """
    with pytest.raises((MatrizVaziaError, PcaDesalinhadaError), match=re.escape(trecho)):
        cargas_da_pca(**argumentos)


# ---------------------------------------------------------------------------
# 5. As quatro plantas do arnês
# ---------------------------------------------------------------------------


def test_planta_serie_removida_nos_scores(matriz_real: RefinoMatrix) -> None:
    tudo = figuras_de(matriz_real, n_componentes=2)
    eixo = tudo["scores"].axes[0]
    esperadas = {s.rotulo: (s.x, s.y) for s in series(eixo)}
    assert confere_series(eixo, esperadas) == [], "a figura limpa já reprova — a planta não separa"

    apagada = planta_serie_removida(eixo)
    achados = confere_series(eixo, esperadas)
    assert achados and apagada in achados[0], achados


def test_planta_eixos_trocados_nas_cargas(matriz_real: RefinoMatrix) -> None:
    tudo = figuras_de(matriz_real, n_componentes=2)
    eixo = tudo["cargas_fig"].axes[0]
    esperadas = {s.rotulo: (s.x, s.y) for s in series(eixo)}
    assert confere_series(eixo, esperadas) == []

    planta_eixos_trocados(eixo)
    assert confere_series(eixo, esperadas) != []


def test_planta_sinal_invertido_no_scree(matriz_real: RefinoMatrix) -> None:
    tudo = figuras_de(matriz_real, n_componentes=5)
    eixo = tudo["scree"].axes[0]
    esperadas = {s.rotulo: (s.x, s.y) for s in series(eixo)}
    planta_sinal_invertido(eixo)
    assert confere_series(eixo, esperadas) != []


def test_planta_corte_de_eixo_no_biplot(matriz_real: RefinoMatrix) -> None:
    """A planta que as três de dado não alcançam: a tela em branco com o dado certo."""
    tudo = figuras_de(matriz_real, n_componentes=2, n_setas=4)
    eixo = tudo["biplot"].axes[0]
    assert confere_vista(eixo) == []

    planta_corte_de_eixo(eixo)
    achados = confere_vista(eixo)
    assert achados and "A figura está em branco" in " ".join(achados), achados


# ---------------------------------------------------------------------------
# 6. Classe (d): o número que a figura DECLARA
# ---------------------------------------------------------------------------


def test_o_rodape_declara_a_populacao_derivada_de_FONTE_INDEPENDENTE(  # noqa: N802
    matriz_real: RefinoMatrix,
) -> None:
    """Os dois denominadores derivados do `structural_metrics`, e não do construtor.

    **É a classe (d) com a fonte independente que ela exige.** `structural_metrics` é outro caminho
    para as mesmas contagens — e, sendo o que a aba de qualidade usa, ele responde também a R8b: a
    figura declara o mesmo número que a tela mostra.
    """
    tudo = figuras_de(matriz_real, n_componentes=2)
    metricas = structural_metrics(matriz_real, pretransform=PRETRANSFORM)
    esperados = [milhar(metricas.n_features_used), milhar(matriz_real.n_features)]
    for nome in ("scree", "scores", "cargas_fig"):
        rodape = rodape_de(tudo[nome])
        for numero in esperados:
            assert numero in rodape, f"{nome}: {numero!r} não está no rodapé"


def test_o_scree_declara_a_soma_que_ele_desenha(matriz_real: RefinoMatrix) -> None:
    """O total do título é a soma das frações — recalculada pelo teste."""
    tudo = figuras_de(matriz_real, n_componentes=5)
    total = sum(tudo["pca"].explained_variance_ratio)
    esperado = f"{total * 100:.1f}".replace(".", ",")
    assert esperado in tudo["scree"].axes[0].get_title(), tudo["scree"].axes[0].get_title()


def test_o_rodape_dos_scores_declara_o_METODO_e_o_n_da_elipse(matriz_real: RefinoMatrix) -> None:  # noqa: N802
    """R5b: o método e o `n`. Sem o `n`, a elipse de 7 pontos e a de 61 leem-se igual."""
    tudo = figuras_de(matriz_real, n_componentes=2)
    rodape = rodape_de(tudo["scores"])
    regiao = tudo["regiao"]
    assert isinstance(regiao, Elipse)
    assert "Hotelling T²" in rodape
    assert milhar(regiao.n) in rodape
    assert regiao.n == int(matriz_real.sample_mask.sum()), (
        "a elipse foi calculada sobre outra população que não as amostras de estudo"
    )
    assert "qui-quadrado" in rodape, "a divergência declarada contra o MetaboAnalyst saiu do rodapé"


def test_o_rotulo_do_eixo_DECLARA_o_sinal_arbitrario(matriz_real: RefinoMatrix) -> None:  # noqa: N802
    """O sinal arbitrário no sítio em que é lido — e este módulo declara em vez de corrigir.

    Corrigir o sinal aqui moveria `qc_pc1_trajectory`, que `diagnose` já usa e que a janela já
    desenha. A segunda metade afirma que a orientação **não** foi aplicada: se alguém a aplicasse,
    os scores da figura deixariam de ser os de `structural_pca`, e o teste das duas PCAs reprovaria.
    """
    tudo = figuras_de(matriz_real, n_componentes=2)
    for chave in ("scores", "cargas_fig", "biplot"):
        eixo = tudo[chave].axes[0]
        assert "sinal arbitrário" in eixo.get_xlabel(), chave
        assert "sinal arbitrário" in eixo.get_ylabel(), chave


def test_o_rotulo_do_componente_quebra_em_duas_linhas() -> None:
    """A quebra veio de a primeira renderização cortar o rótulo do eixo y contra a borda."""
    rotulo = rotulo_do_componente(0, 0.4723, prefixo="carga em ")
    linhas = rotulo.split("\n")
    assert len(linhas) == 2, rotulo
    assert max(len(linha) for linha in linhas) <= 42, (
        f"a linha mais longa tem {max(len(x) for x in linhas)} caracteres — acima do que cabe no "
        "eixo y de uma figura de 4,5 polegadas com rodapé"
    )
    assert "47,2%" in rotulo and "sinal arbitrário" in rotulo


def test_o_rotulo_do_componente_sem_fracao_diz_que_NAO_foi_declarada() -> None:  # noqa: N802
    """`nan` não vira `0,0%`: *"não medi"* e *"medi e deu zero"* são fatos diferentes."""
    assert "não declarada" in rotulo_do_componente(2, float("nan"))


# ---------------------------------------------------------------------------
# 7. As bordas que o plano nomeia
# ---------------------------------------------------------------------------


def test_feature_de_variancia_ZERO_e_contada_e_declarada() -> None:  # noqa: N802
    """A borda de R6: a coluna que o passo 3 descarta aparece no rodapé, e não some.

    **O par causal está na mesma função:** a mesma matriz sem a coluna constante declara descarte
    nenhum. Sem essa metade, *"a contagem aparece"* passaria por uma frase que aparece sempre.
    """
    com = figuras_de(matriz_sintetica(com_constante=True))
    assert com["pca"].n_features_dropped_zero_variance == 1, com["pca"]
    rodape = rodape_de(com["scree"])
    assert "1 descartada" in rodape or "as 1 descartadas" in rodape, rodape

    sem = figuras_de(matriz_sintetica(com_constante=False))
    assert sem["pca"].n_features_dropped_zero_variance == 0
    assert "nenhuma foi descartada" in rodape_de(sem["scree"])


def test_duas_amostras_so_NAO_produzem_elipse_e_a_figura_o_DIZ() -> None:  # noqa: N802
    """A borda do plano: *"dois grupos com uma amostra cada — a elipse não se calcula, e o diz"*.

    **Texto no lugar da elipse, nunca espaço em branco** — a convenção de R4b e de
    `ItemCorpo.vazio_declarado`: região ausente sem explicação lê-se como região esquecida.
    """
    matriz = matriz_sintetica(n_amostras=4, n_features=5)
    tudo = figuras_de(matriz, n_componentes=2)
    # Três amostras de estudo e um QC: acima do mínimo. Forçamos a borda com a máscara nua.
    poucos = np.array([1.0, 2.0])
    regiao = elipse_de_hotelling(poucos, poucos)
    assert isinstance(regiao, ElipseAusente)

    figura = figura_dos_scores(
        escores=tudo["escores"],
        variancia_explicada=tudo["pca"].explained_variance_ratio,
        regiao=regiao,
        recorte=recorte_dos_scores(
            n_features_usadas=tudo["pca"].n_features_used,
            n_features_totais=matriz.n_features,
            regiao=regiao,
        ),
    )
    eixo = figura.axes[0]
    assert any("sem elipse" in t.get_text() for t in eixo.texts), (
        "a elipse não existe e a figura não o diz — espaço em branco lê-se como figura quebrada"
    )
    assert "sem elipse" in rodape_de(figura), "o rodapé também tem de o declarar (R5b)"
    assert not any("Hotelling" in s.rotulo for s in series(eixo)), (
        "há uma série de elipse desenhada e a região é ausente"
    )


def test_menos_componentes_que_os_pedidos_e_RECUSA_TIPADA() -> None:  # noqa: N802
    """A borda de erro do plano: pedir mais componentes do que a matriz admite."""
    matriz = matriz_sintetica(n_amostras=4, n_features=3)
    with pytest.raises(ValueError, match="componentes"):
        structural_pca(matriz, pretransform=PRETRANSFORM, n_components=20)


def test_o_scree_sem_componente_nenhum_e_recusa_por_populacao_vazia() -> None:
    with pytest.raises(MatrizVaziaError, match="ao menos um componente"):
        figura_do_scree(
            variancia_explicada=[],
            recorte=recorte_do_scree(n_features_usadas=1, n_features_totais=1),
        )


def test_o_recorte_sem_denominador_e_MatrizVaziaError_e_nao_RecorteInvalido() -> None:  # noqa: N802
    """A escolha de excepção é sobre QUEM a apanha: só `MatrizVaziaError` vira texto no painel."""
    with pytest.raises(MatrizVaziaError, match="de quantas features"):
        recorte_do_scree(n_features_usadas=0, n_features_totais=0)
    assert not issubclass(MatrizVaziaError, RecorteInvalidoError)


# ---------------------------------------------------------------------------
# 8. As setas do biplot: determinismo e legenda
# ---------------------------------------------------------------------------


def test_as_setas_sao_as_de_maior_MODULO_e_o_empate_resolve_pelo_nome() -> None:  # noqa: N802
    """Regra 3 do README: duas execuções com a mesma entrada dão a mesma figura.

    **A planta cai sobre dado em que as duas coisas diferem**: há um empate de módulo
    plantado, e sem o desempate por nome a ordem dependeria da ordem de entrada.
    """
    cargas = Cargas(
        x=np.array([0.1, 0.9, 0.9, 0.5]),
        y=np.array([0.0, 0.0, 0.0, 0.0]),
        features=("zz", "mm", "aa", "kk"),
    )
    modulos = np.hypot(cargas.x, cargas.y)
    assert modulos[1] == modulos[2], "âncora: sem empate, o desempate por nome não é exercitado"

    escolhidas = [cargas.features[i] for i in setas_do_biplot(cargas, 3)]
    assert escolhidas == ["aa", "mm", "kk"], escolhidas
    assert escolhidas == [cargas.features[i] for i in setas_do_biplot(cargas, 3)]


def test_pedir_mais_setas_do_que_ha_features_devolve_todas() -> None:
    cargas = Cargas(x=np.array([0.1, 0.2]), y=np.array([0.0, 0.0]), features=("a", "b"))
    assert setas_do_biplot(cargas, 10).size == 2


def test_zero_setas_e_recusa() -> None:
    cargas = Cargas(x=np.array([0.1]), y=np.array([0.0]), features=("a",))
    with pytest.raises(ValueError, match="n_setas"):
        setas_do_biplot(cargas, 0)


def test_a_carga_nao_finita_nao_vira_seta() -> None:
    cargas = Cargas(
        x=np.array([np.nan, 0.2, 0.9]), y=np.array([0.1, 0.0, np.inf]), features=("a", "b", "c")
    )
    assert [cargas.features[i] for i in setas_do_biplot(cargas, 3)] == ["b"]


def test_o_rodape_do_biplot_MAPEIA_numero_para_feature(matriz_real: RefinoMatrix) -> None:  # noqa: N802
    """A correspondência que substituiu o nome na ponta da seta — ver `recorte_do_biplot`.

    Sem ela, os números desenhados não nomeiam nada e R2 fica por satisfazer nesta figura.
    """
    tudo = figuras_de(matriz_real, n_componentes=2, n_setas=4)
    indices = setas_do_biplot(tudo["cargas"], 4)
    rodape = rodape_de(tudo["biplot"])
    for posicao, indice in enumerate(indices):
        nome = tudo["cargas"].features[int(indice)]
        assert f"{posicao + 1} = {nome}" in rodape, (
            f"a seta {posicao + 1} não é mapeada para {nome!r} no rodapé"
        )
    assert legenda_das_setas(tudo["cargas"], indices).count("=") == 4


def test_o_fator_das_setas_e_DECLARADO_no_rodape(matriz_real: RefinoMatrix) -> None:  # noqa: N802
    """R5: sem o fator, o comprimento de uma seta lê-se em unidades de score — e não é."""
    tudo = figuras_de(matriz_real, n_componentes=2, n_setas=3)
    rodape = rodape_de(tudo["biplot"])
    assert "multiplicadas por" in rodape and "NUNCA com os pontos" in rodape, rodape


# ---------------------------------------------------------------------------
# 9. O que estas quatro figuras NÃO constroem — o item 6 do relatório
# ---------------------------------------------------------------------------


def test_as_quatro_bijetam_contra_o_CATALOGO_e_nunca_contra_o_relatorio() -> None:  # noqa: N802
    """**Nenhuma destas quatro é o item 6, e este portão impede que se acredite o contrário.**

    O item 6 é *"Gráfico de PCA antes e depois, com QCs destacados"*. A metade dos **QCs
    destacados** está feita — é a série própria do gráfico de scores —, e é justamente por isso que
    alguém pode ler o item como pronto a partir daqui. A metade que falta a estas quatro é *"antes e
    depois"*: **duas** matrizes, e elas descrevem **uma**.

    Registá-las com vínculo de relatório declararia construído o que ninguém construiu, e nada
    acusaria a divergência (KD2).

    > **A ASSERÇÃO DO FIM FOI INVERTIDA em 17/09/2026.** Até aqui
    > ela dizia *"ninguém registou `item-6`"*, e era verdade: o painel «Before»/«After» não existia.
    > Ele foi construído em `plots/correcao.py` e registado com aquela chave, e este teste foi
    > **visto reprovar** com *"alguém registou um construtor para o item 6"* antes de a inversão
    > ser escrita. O que ela afirma agora é mais forte do que a negação: o item 6 tem **exactamente
    > um** construtor, e ele **não** é de `pca` — que é a propriedade que a versão anterior protegia
    > enquanto não havia nenhum.
    """
    do_pca = [c for c in CONSTRUTORES if c.modulo == "pca"]
    assert len(do_pca) == 4, [c.funcao for c in do_pca]
    assert {c.funcao for c in do_pca} == {
        "figura_do_scree",
        "figura_dos_scores",
        "figura_das_cargas",
        "figura_do_biplot",
    }
    assert all(c.vinculo is Vinculo.CATALOGO for c in do_pca), do_pca
    assert {c.chave for c in do_pca} == {"PCA"}, (
        "as quatro servem a MESMA entrada do `[G-1]`: o `PCA` do MetaboAnalyst é uma entrada com "
        "seis abas, e a bijeção de R1 corre sobre conjuntos de chaves"
    )

    item6 = next(i for i in ITENS_OBRIGATORIOS if i.numero == 6)
    assert "antes e depois" in item6.titulo, item6.titulo
    chave = f"item-{item6.numero}"
    donos = [c for c in CONSTRUTORES if c.chave == chave]
    # **Contar e não perguntar se existe.** `chave in {...}` devolve o desfecho
    # de sucesso com duas entradas iguais, e duas fontes para o mesmo desenho é exactamente o que
    # se recusa — e o que nada mais acusaria, porque o relatório compara número de item.
    assert len(donos) == 1, (
        f"o item 6 — {item6.titulo!r} — tem {len(donos)}: {[c.funcao for c in donos]}"
    )
    assert donos[0].vinculo is Vinculo.RELATORIO, donos[0]
    assert donos[0].modulo != "pca", (
        f"o construtor do item 6 é de `pca`: {donos[0].funcao}. As quatro figuras deste módulo "
        "descrevem UMA matriz, e o item pede duas"
    )


# ---------------------------------------------------------------------------
# 10. `structural_pca` é o MESMO caminho de `structural_metrics`
# ---------------------------------------------------------------------------


def test_a_variancia_explicada_NAO_depende_de_quantos_componentes_se_pede(  # noqa: N802
    matriz_real: RefinoMatrix,
) -> None:
    """Se dependesse, o *scree* de dez publicaria frações de outra decomposição que a aba de dois.

    É a suposição que justifica a figura de *scree* pedir mais componentes que o diagnóstico — e
    suposição que justifica desenho é suposição que se afirma.
    """
    dois = structural_pca(matriz_real, pretransform=PRETRANSFORM, n_components=2)
    dez = structural_pca(matriz_real, pretransform=PRETRANSFORM, n_components=10)
    assert np.allclose(dois.explained_variance_ratio, dez.explained_variance_ratio[:2])
    assert np.array_equal(dois.scores, dez.scores[:, :2])
    assert dois.retained_column_digest == dez.retained_column_digest


def test_structural_metrics_e_structural_pca_dao_o_MESMO_numero(matriz_real: RefinoMatrix) -> None:  # noqa: N802
    """A delegação medida, e não afirmada.

    `_pca_scores_with_drops` passou a delegar a `structural_pca` nesta unidade. O comportamento que
    a modificação declara preservar é reafirmado por teste, nunca por leitura do diff.
    """
    metricas = structural_metrics(matriz_real, pretransform=PRETRANSFORM)
    completa = structural_pca(matriz_real, pretransform=PRETRANSFORM, n_components=2)
    assert metricas.n_features_used == completa.n_features_used
    assert metricas.retained_column_digest == completa.retained_column_digest
    assert np.allclose(metricas.explained_variance_ratio, completa.explained_variance_ratio)
    assert np.allclose(metricas.qc_pc1_trajectory, completa.scores[matriz_real.qc_mask, 0]), (
        "a trajetória que a janela já desenha deixou de ser a primeira coluna destes scores"
    )


def test_as_cargas_estao_alinhadas_as_features_RETIDAS(matriz_real: RefinoMatrix) -> None:  # noqa: N802
    """Nomes das retidas, e não `matrix.feature_names` — senão cada ponto leva o nome de outra."""
    com_constante = matriz_sintetica(com_constante=True)
    p = structural_pca(com_constante, pretransform=PRETRANSFORM, n_components=2)
    assert p.loadings.shape[0] == len(p.feature_names_used) == p.n_features_used
    assert p.n_features_used == com_constante.n_features - 1
    assert com_constante.feature_names[-1] not in p.feature_names_used, (
        "a coluna constante, que o passo 3 descartou, continua no vetor de nomes"
    )


def test_a_reconstrucao_dos_scores_pelas_cargas_fecha() -> None:
    """`scores == escalada @ cargas`: prova que os dois lados vêm do MESMO SVD.

    **É o que impede alguém de servir cargas de uma decomposição e scores de outra** — e as duas
    ficariam bem formadas, com as formas certas.
    """
    matriz = matriz_sintetica(n_amostras=10, n_features=5)
    p = structural_pca(matriz, pretransform=PRETRANSFORM, n_components=3)
    valores = np.log10(np.asarray(matriz.values) + 1.0)
    escalada = (valores - valores.mean(axis=0)) / valores.std(axis=0)
    assert np.allclose(escalada @ p.loadings, p.scores, atol=1e-9)


def test_a_orientacao_canonica_NAO_e_aplicada_por_este_produtor() -> None:  # noqa: N802
    """Medido em vez de afirmado: o sinal é o que o SVD deu, e o consumidor DECLARA.

    Se alguém aplicasse `orient_canonically` aqui, `qc_pc1_trajectory` mudaria de sinal e a janela
    passaria a desenhar a trajetória espelhada — sem nada levantar. A convenção canônica é *"o
    elemento de maior módulo do loading é positivo"*, e este teste afirma que ela **não** vale aqui
    em ao menos um componente.
    """
    matriz = matriz_sintetica(n_amostras=14, n_features=8, seed=4)
    p = structural_pca(matriz, pretransform=PRETRANSFORM, n_components=4)
    maiores = [
        float(p.loadings[int(np.argmax(np.abs(p.loadings[:, k]))), k])
        for k in range(p.n_components)
    ]
    assert any(v < 0.0 for v in maiores), (
        f"todos os componentes já saem na orientação canônica ({maiores}). Ou o SVD calhou, ou "
        "alguém aplicou `orient_canonically` no produtor — e nesse caso `qc_pc1_trajectory` mudou"
    )


def test_o_sinal_dos_scores_acompanha_o_das_cargas() -> None:
    """Sinal arbitrário, sim; **incoerente entre os dois lados, não**.

    `scores @ loadings.T` reconstrói a matriz. Virar um lado só mudaria a reconstrução, e nenhum
    teste que olhasse um dos dois isoladamente o veria — é o argumento de `orient_canonically`.
    """
    matriz = matriz_sintetica(n_amostras=10, n_features=5, seed=9)
    p = structural_pca(matriz, pretransform=PRETRANSFORM, n_components=2)
    valores = np.log10(np.asarray(matriz.values) + 1.0)
    escalada = (valores - valores.mean(axis=0)) / valores.std(axis=0)
    reconstruida = p.scores @ p.loadings.T
    assert math.isclose(
        float(np.sum(escalada * reconstruida)), float(np.sum(reconstruida**2)), rel_tol=1e-9
    )


# ---------------------------------------------------------------------------
# 11. Os títulos, para que uma renomeação não passe em silêncio
# ---------------------------------------------------------------------------


def test_os_titulos_das_quatro_sao_os_declarados(matriz_real: RefinoMatrix) -> None:
    tudo = figuras_de(matriz_real, n_componentes=3, n_setas=3)
    for chave, titulo in (
        ("scree", TITULO_DO_SCREE),
        ("scores", TITULO_DOS_SCORES),
        ("cargas_fig", TITULO_DAS_CARGAS),
        ("biplot", TITULO_DO_BIPLOT),
    ):
        assert tudo[chave].axes[0].get_title().startswith(titulo), chave


def test_o_minimo_para_elipse_decorre_da_DIMENSAO_e_nao_e_literal() -> None:  # noqa: N802
    """`n >= p + 1` é condição de existência: com `n = p` a covariância é singular."""
    assert MINIMO_PARA_ELIPSE == DIMENSAO_DA_ELIPSE + 1


def test_confere_rotulos_e_confere_numeros_correm_ao_menos_uma_vez(
    matriz_real: RefinoMatrix,
) -> None:
    """Âncora: as duas funções do arnês que os testes acima não chamam pelo nome.

    Sem esta função, `confere_rotulos` e `confere_numeros` estariam importados e nunca corridos —
    e um import sem uso é a lista de isenção com outro nome.
    """
    tudo = figuras_de(matriz_real, n_componentes=2)
    eixo = tudo["scree"].axes[0]
    assert (
        confere_rotulos(
            eixo,
            x="componente principal",
            y="fração da variância total",
            titulo=eixo.get_title(),
            legenda=("variância do componente", "variância acumulada"),
        )
        == []
    )
    achados = confere_numeros(tudo["scree"], ["6.035"])
    assert achados, "a figura declara mais números que este, e o arnês tem de o dizer"


# ---------------------------------------------------------------------------
# 7. O par PERMANOVA / dispersão no rodapé dos scores (R7, R12b)
# ---------------------------------------------------------------------------
#
# **O par é um número desta figura e não uma figura nova**: ele não tem entrada própria no
# catálogo `[G-1]`, então nada aqui bijeta contra `CONSTRUTORES`. O que estas funções guardam é o
# rodapé — que é a única parte da imagem que outra pessoa pode desmentir com fonte independente.

PAR_DO_ESTUDO = Permanova(
    estatistica=0.96631,
    pvalue=0.432,
    estatistica_da_dispersao=1.2443,
    pvalue_da_dispersao=0.285,
    permutacoes=999,
    semente=42,
    grupos=("ATLL", "Asymptomatic", "HAMTSP", "NoHTLV"),
    n_por_grupo=(13, 18, 14, 16),
)
"""As quatro contagens dos `measurements_*.csv` do laboratório, com números de `p` plausíveis.

**As contagens são OBSERVAÇÃO sobre aqueles quatro arquivos e não invariante**: o que
elas exercitam aqui é o comprimento da frase da população, que é o que empurra o rodapé contra o
teto — e é por isso que a população do estudo real é a certa para o exercitar."""

# A altura em que o par causal da recusa SEPARA, e a janela foi medida em 18/09/2026: com o par a
# recusa dispara abaixo de 3,9 polegadas, e sem ele só abaixo de 3,2. `3,5` é o meio da janela.
#
# **Um número medido e não escolhido:** no tamanho padrão de 4,5 polegadas o rodapé COM o par
# reserva 39,1% da altura contra o teto de 45% — cabe, com 5,9 pontos percentuais de folga, que é
# menos de duas linhas. Encolher a figura é o que torna a recusa observável sem inventar texto.
ALTURA_QUE_SEPARA = 3.5


def _rodape_com(par: Permanova | None, *, regiao: Elipse | ElipseAusente) -> str:
    return " ".join(
        recorte_dos_scores(
            n_features_usadas=5942, n_features_totais=6035, regiao=regiao, permanova=par
        )
        .texto()
        .split()
    )


def test_u13_o_par_de_numeros_ENTRA_no_rodape_da_figura_montada(  # noqa: N802
    matriz_real: RefinoMatrix,
) -> None:
    """Os dois `p`, a pseudo-F, os sorteios, a semente e a população, no texto da figura.

    **A asserção é sobre a figura MONTADA e não sobre o `Recorte`**, porque é a figura que o
    profissional vê: um construtor que recebesse o recorte e não o aplicasse passaria numa
    asserção sobre o objeto e reprovaria nesta. `rodape_de` colapsa o espaço em branco porque
    a frase atravessa a quebra de linha, e `in` sobre o texto cru dá falso negativo.
    """
    p = structural_pca(matriz_real, pretransform=PRETRANSFORM, n_components=2)
    escores = escores_da_pca(
        pc1=p.scores[:, 0], pc2=p.scores[:, 1], tipos=[s.sample_type for s in matriz_real.samples]
    )
    mascara = escores.mascara(TipoDeAmostra.AMOSTRA)
    regiao = elipse_de_hotelling(escores.x[mascara], escores.y[mascara])
    figura = figura_dos_scores(
        escores=escores,
        variancia_explicada=p.explained_variance_ratio,
        regiao=regiao,
        recorte=recorte_dos_scores(
            n_features_usadas=p.n_features_used,
            n_features_totais=matriz_real.n_features,
            regiao=regiao,
            permanova=PAR_DO_ESTUDO,
        ),
    )

    texto = rodape_de(figura)
    for esperado in (
        "PERMANOVA nos componentes",
        "999 permutações",
        "semente 42",
        "pseudo-F = 0,9663",
        "p = 0,432",
        # O segundo par, e é ele que tem de estar na MESMA frase.
        "dispersão, sempre junto",
        "F = 1,244",
        "p = 0,285",
        # A população do TESTE, que não é a da nuvem desenhada.
        "61 amostras de estudo",
        "ATLL 13",
        "NoHTLV 16",
        "QC e validação fora",
        # R12b: o desfecho sem sinal é rotulado como achado válido.
        "achado válido e não ausência de medida",
    ):
        assert esperado in texto, f"o rodapé não declara {esperado!r}. Rodapé: {texto}"


def test_u13_sem_o_par_o_rodape_DECLARA_a_ausencia_em_vez_de_a_esconder() -> None:  # noqa: N802
    """Par causal do acréscimo: os dois rodapés divergem, e o de `None` não finge resultado.

    **`None` é *"não calculado"* e nunca *"calculado e não separou"***. A segunda é `p`
    alto, e é o desfecho **normal** deste corpus; colapsá-las poria a figura a afirmar que o teste
    correu quando a matriz não tem grupos que o sustentem.

    **A asserção negativa vem acompanhada da positiva**, porque *"não diz pseudo-F"* sobre um texto
    vazio seria satisfeito por vacuidade.
    """
    regiao = ElipseAusente("sem elipse nesta planta")
    com = _rodape_com(PAR_DO_ESTUDO, regiao=regiao)
    sem = _rodape_com(None, regiao=regiao)

    assert com != sem, "acrescentar o par não mudou o rodapé: o argumento está a ser descartado"
    assert PERMANOVA_AUSENTE in sem, f"a ausência não é declarada. Rodapé: {sem}"
    assert "PERMANOVA não calculada" in sem
    assert "pseudo-F" not in sem, f"o rodapé sem o par publica uma estatística. Rodapé: {sem}"
    assert "pseudo-F" in com, "a âncora positiva falhou: nem com o par há estatística no rodapé"


def test_u13_o_rodape_com_o_par_RECUSA_numa_figura_baixa_e_sem_ele_NAO() -> None:  # noqa: N802
    """**O desfecho OPOSTO ao de sucesso, e sem ele este acréscimo seria uma aposta.**

    O rodapé dos scores é o mais cheio deste pacote — pré-tratamento, elipse, ressalva de sinal — e
    `recorte.aplica` recusa acima de `RESERVA_MAXIMA` da altura. Dois números a mais são exactamente
    o tipo de acréscimo que atravessa esse teto, e o plano previu-o.

    **É um par causal e não uma asserção de recusa:** as duas execuções diferem **só** pelo par, na
    mesma figura e no mesmo tamanho. Sem a metade de cima, o teste passaria por a figura ser
    pequena, e não por o par a ter empurrado — o instrumento a acusar o alvo errado.

    **No tamanho padrão o par CABE, e isso foi medido** — 39,1% contra o teto de 45%. O que este
    teste guarda é a recusa continuar a existir: se alguém subir `RESERVA_MAXIMA` para acomodar
    texto, é aqui que se vê.
    """
    regiao = elipse_de_hotelling(
        np.random.default_rng(0).normal(size=61), np.random.default_rng(1).normal(size=61)
    )
    argumentos = {"n_features_usadas": 5942, "n_features_totais": 6035, "regiao": regiao}

    baixa_sem = nova_figura(altura=ALTURA_QUE_SEPARA)
    baixa_sem.add_subplot(1, 1, 1)
    aplica(baixa_sem, recorte_dos_scores(**argumentos, permanova=None))

    baixa_com = nova_figura(altura=ALTURA_QUE_SEPARA)
    baixa_com.add_subplot(1, 1, 1)
    with pytest.raises(RecorteInvalidoError, match="rodapé quebrou"):
        aplica(baixa_com, recorte_dos_scores(**argumentos, permanova=PAR_DO_ESTUDO))


def test_u13_permanova_do_fio_aceita_o_par_inteiro_e_recusa_METADE() -> None:  # noqa: N802
    """Guarda de fronteira, nas duas direções — e a de baixo é a que importa.

    **O modo de falha insidioso é o que não levanta:** um corpo com `pvalue` e sem
    `pvalue_da_dispersao` — servidor a meio de migração, cliente a montar o dicionário à mão — não
    é um erro de tipo em nenhum ponto do caminho. Ele publicaria metade do par, que é exactamente
    o que se proíbe, com a figura bem formada.
    """
    inteiro = {
        "estatistica": 1.5,
        "pvalue": 0.02,
        "estatistica_da_dispersao": 0.8,
        "pvalue_da_dispersao": 0.61,
        "permutacoes": 999,
        "semente": 42,
        "grupos": ["a", "b"],
        "n_por_grupo": [10, 12],
    }
    lido = permanova_do_fio(inteiro)
    assert lido is not None
    assert (lido.pvalue, lido.pvalue_da_dispersao) == (0.02, 0.61)
    assert lido.n_por_grupo == (10, 12)

    for chave in ("estatistica", "pvalue", "estatistica_da_dispersao", "pvalue_da_dispersao"):
        metade = {k: v for k, v in inteiro.items() if k != chave}
        assert permanova_do_fio(metade) is None, (
            f"o corpo sem {chave!r} produziu um par. Metade do par publicada é o que KTD9 proíbe"
        )

    assert permanova_do_fio(None) is None
    assert permanova_do_fio("não é objeto") is None
    assert permanova_do_fio({}) is None, (
        "o dicionário VAZIO produziu um par: a guarda enumera as chaves em vez de as varrer, "
        "porque `all()` sobre conjunto vazio é `True`"
    )


def test_u13_permanova_do_fio_recusa_populacao_DESALINHADA() -> None:  # noqa: N802
    """Três nomes para duas contagens poria um `n` que não é o do teste no rodapé."""
    base = {
        "estatistica": 1.5,
        "pvalue": 0.02,
        "estatistica_da_dispersao": 0.8,
        "pvalue_da_dispersao": 0.61,
        "permutacoes": 999,
        "semente": 42,
    }
    assert permanova_do_fio({**base, "grupos": ["a", "b", "c"], "n_por_grupo": [10, 12]}) is None
    assert permanova_do_fio({**base, "grupos": [], "n_por_grupo": []}) is None
    assert permanova_do_fio({**base, "grupos": ["a", "b"], "n_por_grupo": [10, 12]}) is not None


def test_u13_o_par_do_rodape_e_o_da_pca_que_CORRE_e_nao_o_da_orfa(  # noqa: N802
    matriz_real: RefinoMatrix,
) -> None:
    """O número novo: o referente certo, e a instância certa dele.

    As duas PCAs deste repositório dão decomposições diferentes — a seção 1 deste arquivo mede-o —,
    então há **dois** pares de números possíveis para esta figura, os dois bem formados. Este teste
    afirma que o que vai para o rodapé é o da que corre, **e que os dois divergem**: sem a segunda
    metade, a asserção passaria por coincidência e trocar a fonte não seria detectável.

    **A planta cai sobre dado em que as duas coisas diferem**, e é a estatística que se
    compara e não o `p` — o `p` é granulado em `1/(N+1)` e duas quantidades diferentes podem cair
    na mesma casa, o que tornaria a divergência indetectável.
    """
    from refino.statistics.pca import pca as pca_orfa
    from refino.statistics.permanova import permanova as calcula

    estudo = matriz_real.sample_mask
    rotulos = [g for g, entra in zip(matriz_real.groups, estudo.tolist(), strict=True) if entra]
    # **Asserção e não `pytest.skip`, e a escolha é de contrato.** A lista de regimes de `skip` é
    # parte do contrato dos testes, e um regime novo para uma condição que **não ocorre**
    # alargaria-a sem referente — o portão de `tests/skip_policy.py` apanhou exactamente isso na
    # primeira versão desta função. O caso de o arquivo faltar já tem regime: quem o emite é
    # `matrix_from_toserrf`, no resolvedor do `conftest`.
    #
    # **Quatro grupos é OBSERVAÇÃO sobre os `measurements_*.csv` do laboratório e não invariante**:
    # `Asymptomatic` 18, `NoHTLV` 16, `HAMTSP` 14 e `ATLL` 13. O que se afirma aqui é o
    # piso de que este teste precisa, sobre aqueles arquivos.
    assert len(set(rotulos)) >= 2, (
        f"os `measurements_{MODO}.csv` do laboratório declaram {sorted(set(rotulos))} como grupos "
        "de estudo, e a PERMANOVA precisa de dois. O corpus mudou, e este teste com ele"
    )

    corre = structural_pca(matriz_real, pretransform=PRETRANSFORM, n_components=2)
    da_que_corre = calcula(corre.scores[estudo], rotulos, permutations=99, seed=42, cancelled=None)
    da_orfa = calcula(
        pca_orfa(matriz_real, n_components=2).scores[estudo],
        rotulos,
        permutations=99,
        seed=42,
        cancelled=None,
    )

    assert da_que_corre.statistic != pytest.approx(da_orfa.statistic, rel=1e-3), (
        f"as duas PCAs dão a MESMA pseudo-F — {da_que_corre.statistic} e {da_orfa.statistic}: a "
        "planta não separa, e este teste passaria com a fonte trocada"
    )

    texto = _rodape_com(
        Permanova(
            estatistica=da_que_corre.statistic,
            pvalue=da_que_corre.pvalue,
            estatistica_da_dispersao=da_que_corre.dispersion_statistic,
            pvalue_da_dispersao=da_que_corre.dispersion_pvalue,
            permutacoes=da_que_corre.permutations,
            semente=da_que_corre.seed,
            grupos=da_que_corre.groups,
            n_por_grupo=da_que_corre.n_per_group,
        ),
        regiao=ElipseAusente("sem elipse nesta planta"),
    )
    assert f"pseudo-F = {decimal(da_que_corre.statistic)}" in texto, texto
    assert f"pseudo-F = {decimal(da_orfa.statistic)}" not in texto, (
        "o rodapé carrega a pseudo-F da PCA ÓRFÃ — o referente certo, a instância errada"
    )


# ---------------------------------------------------------------------------
# A PCA POR GRUPO — uma região por grupo, recortada ao quadro dos pontos
# ---------------------------------------------------------------------------
#
# **A forma é recortar e anotar n.** Com grupos de 3 a 5 amostras a região exata fica enorme —
# raio de 32,6 desvios com n = 3 —, então ela é recortada ao quadro dos PONTOS, cada grupo leva o
# n na legenda, e nenhum grupo some. **O QC não é grupo:** o grupo de cada amostra sai de
# `sample_mask` no servidor, e a fronteira daqui recusa um QC que chegue com grupo.

# As injeções da fixtura, em (tipo, grupo, PC1, PC2). **Geometria escolhida e não sorteada:** o
# grupo C tem DUAS amostras, abaixo do mínimo; o B tem quatro, e a região dele (c² = 71,25) passa do
# quadro; o A tem seis. **A ordem mistura os tipos de propósito** — uma máscara que dependesse de
# blocos contíguos passaria numa fixtura em blocos.
_INJECOES_AGRUPADAS = (
    ("qc", None, 0.2, 0.1),
    ("sample", "A", -4.0, 0.5),
    ("sample", "B", 2.5, 1.5),
    ("sample", "A", -3.2, -0.4),
    ("sample", "C", 0.5, -3.0),
    ("sample", "B", 3.4, 0.4),
    ("qc", None, -0.3, 0.4),
    ("sample", "A", -2.5, 0.8),
    ("validate", None, 1.0, 2.5),
    ("sample", "B", 3.9, 1.8),
    ("sample", "A", -3.6, -1.0),
    ("sample", "C", -0.4, -2.6),
    ("sample", "A", -2.2, 0.1),
    ("sample", "B", 2.8, 0.9),
    ("qc", None, 0.1, -0.2),
    ("sample", "A", -3.0, 1.2),
)

# **D tem TRÊS amostras espalhadas pela nuvem inteira**, e com n = 3 o c² é 1.064: a região dele
# contém o quadro todo, e o contorno não passa por dentro dele em ponto nenhum. **E tem cinco
# amostras apertadas**, e a região dele cabe — é o par que separa a nota da ausência dela.
_INJECOES_COM_GRUPO_LARGO = (
    ("sample", "D", -3.0, -2.0),
    ("sample", "E", 1.0, 1.1),
    ("qc", None, 0.0, 0.2),
    ("sample", "D", 3.0, -2.0),
    ("sample", "E", 1.2, 0.9),
    ("sample", "E", 0.9, 1.0),
    ("sample", "D", 0.0, 3.0),
    ("sample", "E", 1.1, 1.2),
    ("qc", None, 0.2, -0.1),
    ("sample", "E", 1.05, 0.95),
)


def _escores_de(injecoes: tuple[tuple[str, str | None, float, float], ...]) -> Escores:
    """Os escores agrupados de uma fixtura, pela mesma fronteira que o fio atravessa."""
    return escores_da_pca(
        pc1=[x for _, _, x, _ in injecoes],
        pc2=[y for _, _, _, y in injecoes],
        tipos=[tipo for tipo, _, _, _ in injecoes],
        grupos=[grupo for _, grupo, _, _ in injecoes],
    )


def _figura_agrupada(
    escores: Escores,
    *,
    estilos: EstilosDosGrupos | None = None,
    permanova: Permanova | None = None,
) -> Figure:
    """A figura como a janela a monta: a região sai de `regiao_dos_scores`, e vai às DUAS pontas."""
    regiao = regiao_dos_scores(escores)
    return figura_dos_scores(
        escores=escores,
        variancia_explicada=[0.41, 0.17],
        regiao=regiao,
        recorte=recorte_dos_scores(
            n_features_usadas=40, n_features_totais=42, regiao=regiao, permanova=permanova
        ),
        estilos=estilos,
    )


def _legenda_de(eixo: Axes) -> tuple[str, ...]:
    """Os textos da legenda, na ordem em que ela os mostra."""
    return tuple(t.get_text() for t in eixo.get_legend().get_texts())


def _dentro_pela_forma_quadratica(
    escores: Escores, grupo: str, regiao: Elipse, px: float, py: float
) -> bool:
    """`(p - xbar)' S^-1 (p - xbar) <= c²`, calculado AQUI a partir dos pontos do grupo.

    **O outro caminho como oráculo:** a figura decide a nota pela geometria da `Elipse`
    (centro, semieixos, ângulo); este teste decide pela forma quadrática sobre a covariância
    amostral. Um erro de rotação ou de semieixo na figura faria os dois discordarem.
    """
    mascara = escores.mascara_do_grupo(grupo)
    pontos = np.column_stack((escores.x[mascara], escores.y[mascara]))
    desvio = np.array([px, py]) - pontos.mean(axis=0)
    forma = float(desvio @ np.linalg.inv(np.cov(pontos, rowvar=False)) @ desvio)
    return forma <= regiao.c2


def test_por_grupo_uma_regiao_por_grupo_com_pontos_e_o_grupo_pequeno_SEM_ela_com_o_motivo() -> None:  # noqa: N802
    """O cenário do plano: uma região por grupo que a sustenta, e o grupo de 2 sem ela, DITO.

    **Cada região é a de `elipse_de_hotelling` sobre os pontos DAQUELE grupo**, vértice a vértice —
    uma região calculada sobre a nuvem inteira, ou com o QC dentro, desenharia outro contorno. **E
    nenhum grupo some:** o C não tem região, mas tem a série, e a entrada dele na legenda traz a
    contagem e o motivo.
    """
    escores = _escores_de(_INJECOES_AGRUPADAS)
    regioes = regiao_dos_scores(escores)
    assert isinstance(regioes, RegioesDosGrupos)
    assert regioes.grupos == ("A", "B", "C")

    eixo = _figura_agrupada(escores).axes[0]
    por_rotulo = {s.rotulo: s for s in series(eixo)}
    for grupo in ("A", "B"):
        mascara = escores.mascara_do_grupo(grupo)
        esperada = elipse_de_hotelling(escores.x[mascara], escores.y[mascara])
        assert isinstance(esperada, Elipse), grupo
        x, y = esperada.contorno()
        desenhada = por_rotulo[rotulo_da_regiao_do_grupo(grupo)]
        assert np.allclose(desenhada.x, x) and np.allclose(desenhada.y, y), grupo

    assert rotulo_da_regiao_do_grupo("C") not in por_rotulo
    assert isinstance(regioes.de("C").regiao, ElipseAusente)
    assert regioes.de("C").n == 2

    legenda = _legenda_de(eixo)
    for rotulo in (
        "A — 6 amostras",
        "B — 4 amostras",
        f"C — 2 amostras\n({NOTA_SEM_REGIAO}: menos de {MINIMO_PARA_ELIPSE})",
        "QC — 3 amostras",
        "amostra de validação — 1 amostra",
    ):
        assert rotulo in legenda, legenda
    assert por_rotulo["C — 2 amostras"].x.size == 2, "o grupo sem região sumiu da figura"


def test_por_grupo_o_QC_NAO_vira_grupo_nem_entra_numa_regiao() -> None:  # noqa: N802
    """O segundo cenário do plano, nas duas pontas: o desenho e a fronteira.

    **No desenho**, os pontos das séries de grupo são exactamente as amostras de estudo — nenhum QC
    entre eles —, e o QC continua série própria, por cima. **Na fronteira**, um QC que chegue com
    nome de grupo é recusado: aceitá-lo como grupo produziria o defeito do QC contrastado
    como se fosse um grupo clínico, e ignorá-lo em silêncio esconderia um corpo montado errado.
    """
    escores = _escores_de(_INJECOES_AGRUPADAS)
    regioes = regioes_dos_grupos(escores)
    assert regioes.grupos == ("A", "B", "C"), regioes.grupos
    assert {r.n for r in regioes.por_grupo} == {6, 4, 2}

    eixo = _figura_agrupada(escores).axes[0]
    por_rotulo = {s.rotulo: s for s in series(eixo)}
    qc = escores.mascara(TipoDeAmostra.QC)
    assert int(qc.sum()) == 3, "âncora: a fixtura tem três QCs, e a asserção abaixo seria vácua"
    assert np.allclose(np.sort(por_rotulo["QC — 3 amostras"].x), np.sort(escores.x[qc]))

    dos_grupos = np.concatenate(
        [por_rotulo[rotulo_do_grupo(g, n)].x for g, n in (("A", 6), ("B", 4), ("C", 2))]
    )
    estudo = escores.mascara(TipoDeAmostra.AMOSTRA)
    assert np.allclose(np.sort(dos_grupos), np.sort(escores.x[estudo]))

    tipos = [tipo for tipo, _, _, _ in _INJECOES_AGRUPADAS]
    plantado: list[str | None] = [grupo for _, grupo, _, _ in _INJECOES_AGRUPADAS]
    plantado[tipos.index("qc")] = "QC"
    with pytest.raises(GrupoInvalidoError, match="QC"):
        escores_da_pca(pc1=list(escores.x), pc2=list(escores.y), tipos=tipos, grupos=plantado)


@pytest.mark.parametrize(
    ("posicao", "valor"),
    [(1, None), (1, 7), (8, "A")],
    ids=["estudo-sem-grupo", "estudo-grupo-que-nao-e-texto", "validacao-com-grupo"],
)
def test_por_grupo_a_fronteira_recusa_o_grupo_que_NAO_casa_com_o_tipo(  # noqa: N802
    posicao: int, valor: str | int | None
) -> None:
    """A guarda de fronteira no vetor de grupos: um corpo que o servidor não monta é recusado.

    Uma amostra de estudo que caísse num `else` razoável sairia
    das séries de grupo e da região, e a legenda contaria menos do que a matriz tem. **Os três casos
    são de CORPO, e nenhum é fato do dado:** `rotas._resumo_da_pca` serve sempre o `group` da
    amostra de estudo, que é `str`, e `null` fora de `sample_mask`. O nome de grupo EM BRANCO é
    outro caso, porque a ingestão o aceita — ver o teste seguinte.
    """
    grupos: list[str | None] = [grupo for _, grupo, _, _ in _INJECOES_AGRUPADAS]
    grupos[posicao] = valor
    with pytest.raises(GrupoInvalidoError):
        escores_da_pca(
            pc1=[x for _, _, x, _ in _INJECOES_AGRUPADAS],
            pc2=[y for _, _, _, y in _INJECOES_AGRUPADAS],
            tipos=[tipo for tipo, _, _, _ in _INJECOES_AGRUPADAS],
            grupos=grupos,
        )


@pytest.mark.parametrize("em_branco", ["", "   "], ids=["vazio", "so-espacos"])
def test_por_grupo_o_nome_EM_BRANCO_da_a_figura_por_tipo_e_o_titulo_DIZ_porque(  # noqa: N802
    em_branco: str,
) -> None:
    """Uma amostra de estudo com o grupo em branco: a figura é a por tipo, e o título conta-a.

    **O nome em branco é fato do dado e não corpo montado errado**, e é isso que o separa dos casos
    do teste anterior: a ingestão aceita a célula `LABEL` vazia (`SampleMetadata.group` é `str` sem
    guarda, e `_check_group_labels` não emite código), e `rotas._resumo_da_pca` serve-a como `""`.
    **Recusar aqui retirava da tela uma figura que já se desenhava** com o mesmo dado; pintar por
    grupo com a amostra fora das séries faria a legenda contar menos do que a matriz tem; e pô-la
    num grupo *"(sem grupo)"* inventaria uma população. A figura volta à forma por tipo, com a
    região única sobre TODAS as amostras de estudo, e o título diz quantas chegaram sem nome.

    **O par:** o mesmo corpo com o nome preenchido pinta por grupo e o título não fala de grupo.
    """
    grupos: list[str | None] = [grupo for _, grupo, _, _ in _INJECOES_AGRUPADAS]
    argumentos = {
        "pc1": [x for _, _, x, _ in _INJECOES_AGRUPADAS],
        "pc2": [y for _, _, _, y in _INJECOES_AGRUPADAS],
        "tipos": [tipo for tipo, _, _, _ in _INJECOES_AGRUPADAS],
    }
    inteiro = escores_da_pca(**argumentos, grupos=grupos)
    assert inteiro.grupos is not None and inteiro.grupos_em_branco == 0
    assert isinstance(regiao_dos_scores(inteiro), RegioesDosGrupos)
    titulo_inteiro = _figura_agrupada(inteiro).axes[0].get_title()
    assert "grupo" not in titulo_inteiro, titulo_inteiro

    grupos[1] = em_branco
    escores = escores_da_pca(**argumentos, grupos=grupos)
    assert escores.grupos is None, "o nome em branco pintou por grupo"
    assert escores.grupos_em_branco == 1
    regiao = regiao_dos_scores(escores)
    assert regiao == elipse_das_amostras(escores), "a região não é a única"

    eixo = _figura_agrupada(escores).axes[0]
    estudo = int(escores.mascara(TipoDeAmostra.AMOSTRA).sum())
    rotulos = {s.rotulo for s in series(eixo)}
    assert f"amostra de estudo — {estudo} amostras" in rotulos, sorted(rotulos)
    titulo = " ".join(eixo.get_title().split())
    assert "1 amostra de estudo com o grupo em branco" in titulo, titulo

    grupos[0] = "QC"
    with pytest.raises(GrupoInvalidoError, match="QC"):
        escores_da_pca(**argumentos, grupos=grupos)


@pytest.mark.parametrize("grupos", [["A"], "A", {"A": 1}], ids=["curto", "texto", "dicionario"])
def test_por_grupo_o_vetor_de_grupos_DESALINHADO_e_recusa(grupos: object) -> None:  # noqa: N802
    """Um vetor de grupos que não descreve as mesmas injeções é recusado, e não recortado."""
    with pytest.raises(PcaDesalinhadaError, match="grupo"):
        escores_da_pca(pc1=[0.0, 1.0], pc2=[1.0, 0.0], tipos=["sample", "qc"], grupos=grupos)


def test_por_grupo_a_regiao_e_RECORTADA_ao_quadro_dos_PONTOS() -> None:  # noqa: N802
    """A metade «recortar» da escolha: o quadro é o dos pontos, e a região não o alarga.

    **O oráculo é um eixo com SÓ os pontos**, desenhado aqui: os limites da figura têm de ser os
    dele. **A âncora vem junto:** a região de B passa do quadro, então uma figura que
    deixasse a região entrar no autoescalonamento teria outros limites, e a igualdade separa os
    dois casos.
    E a classe (c) do arnês acusa exactamente as regiões — que é o recorte, visto de fora.
    """
    escores = _escores_de(_INJECOES_AGRUPADAS)
    eixo = _figura_agrupada(escores).axes[0]

    so_os_pontos = nova_figura().add_subplot(1, 1, 1)
    so_os_pontos.plot(escores.x, escores.y, marker="o", linestyle="none")
    assert eixo.get_xlim() == pytest.approx(so_os_pontos.get_xlim())
    assert eixo.get_ylim() == pytest.approx(so_os_pontos.get_ylim())

    (x0, x1), (y0, y1) = eixo.get_xlim(), eixo.get_ylim()
    regiao_b = {s.rotulo: s for s in series(eixo)}[rotulo_da_regiao_do_grupo("B")]
    fora = (regiao_b.x < x0) | (regiao_b.x > x1) | (regiao_b.y < y0) | (regiao_b.y > y1)
    assert fora.any(), "âncora: a região de B cabe no quadro, e o recorte não foi exercitado"

    achados = confere_vista(eixo)
    assert achados, "âncora: nenhuma região passa do quadro, e a asserção abaixo seria vácua"
    assert all(rotulo_da_regiao_do_grupo("") in achado for achado in achados), achados


def test_por_grupo_a_regiao_MAIOR_que_o_quadro_e_DITA_e_a_que_cabe_NAO() -> None:  # noqa: N802
    """Recortada, a região de um grupo pequeno pode conter o quadro inteiro — e então não aparece.

    **Região que não aparece sem explicação lê-se como região esquecida** — a regra que
    `_desenha_a_elipse` já segue para a ausência —: a entrada do grupo na legenda diz que a região
    dele é maior que o quadro. O par é o grupo E, cuja região cabe e cuja entrada não o diz.
    """
    escores = _escores_de(_INJECOES_COM_GRUPO_LARGO)
    regioes = regioes_dos_grupos(escores)
    larga, apertada = regioes.de("D").regiao, regioes.de("E").regiao
    assert isinstance(larga, Elipse) and isinstance(apertada, Elipse)

    eixo = _figura_agrupada(escores).axes[0]
    (x0, x1), (y0, y1) = eixo.get_xlim(), eixo.get_ylim()
    cantos = ((x0, y0), (x0, y1), (x1, y0), (x1, y1))
    assert all(_dentro_pela_forma_quadratica(escores, "D", larga, *c) for c in cantos), (
        "âncora: o quadro não cabe na região de D, e a nota não foi exercitada"
    )
    assert not all(_dentro_pela_forma_quadratica(escores, "E", apertada, *c) for c in cantos)

    legenda = _legenda_de(eixo)
    assert f"D — 3 amostras\n({NOTA_MAIOR_QUE_O_QUADRO})" in legenda, legenda
    assert "E — 5 amostras" in legenda, legenda


def test_por_grupo_o_motivo_vai_na_LEGENDA_e_NADA_se_escreve_dentro_do_quadro() -> None:  # noqa: N802
    """Nenhuma nota dentro da área de dado da figura por grupo — o motivo vive na legenda.

    **Veio de olhar a figura montada.** A primeira versão escrevia o motivo num texto no canto
    de cima do eixo, como `_desenha_a_elipse` faz com a ausência da região única; com grupos de 3, 4
    e 5 e um de 2, em 50 cenários sintéticos, a nota aparecia nos 50 e tapava ponto em 48, no
    tamanho de construção. O quadro é o dos pontos, com a margem do autoescalonamento, então todo
    canto do eixo tem ponto perto — a razão por que a legenda já vive fora da área de dado
    (`legenda_fora`).

    **As duas fixturas exercitam os dois motivos**, e a âncora é que cada uma tem um: sem ela, um
    quadro vazio de texto passaria por vacuidade numa figura que nunca teve nota a escrever.
    """
    for injecoes, motivo in (
        (_INJECOES_AGRUPADAS, NOTA_SEM_REGIAO),
        (_INJECOES_COM_GRUPO_LARGO, NOTA_MAIOR_QUE_O_QUADRO),
    ):
        eixo = _figura_agrupada(_escores_de(injecoes)).axes[0]
        assert any(f"({motivo}" in rotulo for rotulo in _legenda_de(eixo)), (
            f"âncora: nenhuma entrada diz {motivo!r}, e o quadro vazio de texto seria vácuo"
        )
        assert not eixo.texts, [t.get_text() for t in eixo.texts]


def test_por_grupo_o_motivo_vai_na_SEGUNDA_linha_e_o_eixo_NAO_perde_largura(  # noqa: N802
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """O parêntese do motivo abre uma linha nova na entrada, e a legenda não alarga por ele.

    **Medido, e é o que decidiu a forma:** com o motivo na mesma linha, a entrada mais
    longa passava a ser a do grupo anotado, e em 50 cenários sintéticos de grupos de 3, 4 e 5 e um
    de 2 o eixo encolhia de 372 para 263 px de largura (medianas) no tamanho de construção — 29%
    da área de dado entregue à legenda.

    **A população em que a largura se mantém, e ela vai dita:** a de uma legenda que já
    tem uma entrada pelo menos tão longa quanto a linha do motivo — aqui, *"amostra de validação —
    1 amostra"*, de 32 caracteres, contra os 24 de *"(sem região: menos de 3)"*. Numa legenda só
    de entradas curtas, a linha do motivo passa a ser a mais longa e alarga-a; o que se garante
    então é o teto da linha, que é o teste seguinte.

    **O par causal é a mesma figura, com e sem os motivos** — `_desenha_as_regioes` desenha o
    mesmo e devolve nada —, e as larguras dos dois eixos coincidem. **A âncora vem junto:**
    a primeira figura tem entrada anotada e a segunda não, senão a igualdade não separaria nada.
    """
    _larguras_com_e_sem_motivo(_escores_de(_INJECOES_AGRUPADAS), monkeypatch)


def _larguras_com_e_sem_motivo(escores: Escores, monkeypatch: pytest.MonkeyPatch) -> None:
    """O par causal das larguras: a figura com os motivos, e com `_desenha_as_regioes` mudo."""
    com_motivo = _figura_agrupada(escores)
    assert any("(" in t for t in _legenda_de(com_motivo.axes[0])), "âncora: nada anotado"

    desenha = pca._desenha_as_regioes
    with monkeypatch.context() as remendo:
        remendo.setattr(pca, "_desenha_as_regioes", lambda *a: (desenha(*a), {})[1])
        sem_motivo = _figura_agrupada(escores)
    assert not any("(" in t for t in _legenda_de(sem_motivo.axes[0])), "a planta não aterrou"

    com_motivo.canvas.draw()
    sem_motivo.canvas.draw()
    largura = sem_motivo.axes[0].get_window_extent().width
    assert com_motivo.axes[0].get_window_extent().width == pytest.approx(largura, abs=1.0)


def test_por_grupo_o_motivo_LONGO_e_quebrado_e_o_eixo_NAO_perde_largura(  # noqa: N802
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A covariância singular: o motivo tem 66 caracteres, e quebrado não passa do teto da linha.

    **Medido contra a forma anterior:** com o grupo colinear, o motivo
    *"(sem região: a covariância destes pontos é singular ou não finita)"* ia inteiro na segunda
    linha, e a largura do eixo caía de 378,3 para 216,2 px no tamanho de construção — mais do que a
    forma «na mesma linha» que tinha sido rejeitada por custar 29%. **Agora o motivo quebra em
    linhas de até `COLUNAS_DO_MOTIVO`** — o comprimento do mais longo dos dois motivos curtos —, e
    nenhum motivo custa mais largura do que *"(região maior que o quadro)"*.

    **As duas metades:** toda linha de motivo cabe no teto, e a largura com e sem os motivos é a
    mesma nesta fixtura, que tem a entrada da validação mais longa que o teto.
    """
    injecoes = (
        *_INJECOES_AGRUPADAS,
        ("sample", "G", 0.0, 0.0),
        ("sample", "G", 1.0, 1.0),
        ("sample", "G", 2.0, 2.0),
    )
    escores = _escores_de(injecoes)
    ausente = regioes_dos_grupos(escores).de("G").regiao
    assert isinstance(ausente, ElipseAusente), "âncora: os três pontos sustentam região"
    assert len(f"({NOTA_SEM_REGIAO}: {ausente.motivo.split(':', 1)[0]})") > COLUNAS_DO_MOTIVO, (
        "âncora: o motivo cabe numa linha, e a quebra não foi exercitada"
    )

    legenda = _legenda_de(_figura_agrupada(escores).axes[0])
    do_g = next(t for t in legenda if t.startswith("G — 3 amostras"))
    motivo = do_g.split("\n")[1:]
    assert len(motivo) > 1, do_g
    assert all(len(linha) <= COLUNAS_DO_MOTIVO for linha in motivo), motivo
    assert " ".join(motivo) == f"({NOTA_SEM_REGIAO}: {ausente.motivo.split(':', 1)[0]})", motivo

    _larguras_com_e_sem_motivo(escores, monkeypatch)


def test_por_grupo_o_motivo_na_legenda_NAO_adivinha_a_causa() -> None:  # noqa: N802
    """Três pontos COLINEARES: a região não existe, e não é por faltar ponto.

    **O ramo do motivo que não é o do `n`.** A entrada escreve a cabeça do motivo que
    `elipse_de_hotelling` deu, e não *"menos de 3"* — que seria falso com três pontos — nem uma
    causa adivinhada. É a regra de `correcao._frase_sem_regiao`.
    """
    injecoes = (
        *_INJECOES_AGRUPADAS,
        ("sample", "G", 0.0, 0.0),
        ("sample", "G", 1.0, 1.0),
        ("sample", "G", 2.0, 2.0),
    )
    escores = _escores_de(injecoes)
    ausente = regioes_dos_grupos(escores).de("G").regiao
    assert isinstance(ausente, ElipseAusente), "âncora: os três pontos sustentam região"
    # Comparado sem as quebras: a cabeça deste motivo passa de `COLUNAS_DO_MOTIVO` e sai em várias
    # linhas — o teste da largura é que afirma a forma da quebra.
    legenda = [" ".join(t.split()) for t in _legenda_de(_figura_agrupada(escores).axes[0])]
    esperada = f"G — 3 amostras ({NOTA_SEM_REGIAO}: {ausente.motivo.split(':', 1)[0]})"
    assert esperada in legenda, legenda
    assert not any(t.startswith("G —") and "menos de" in t for t in legenda), legenda


def test_por_grupo_o_grupo_SEM_ponto_finito_NAO_some_da_legenda() -> None:  # noqa: N802
    """Um grupo cujas amostras têm todas o score não finito não tem série — e tem entrada.

    **É o caso que só a legenda pode dizer:** sem ponto, o grupo não tem marcador no quadro nem
    região, e sem a entrada ele sumiria da figura sem nada a acusar. O `n` é o da população da
    região, as amostras com as duas coordenadas finitas, e por isso é zero.
    """
    injecoes = (*_INJECOES_AGRUPADAS, ("sample", "F", float("nan"), 1.0))
    escores = _escores_de(injecoes)
    assert "F" in escores.grupos_de_estudo and not escores.mascara_do_grupo("F").any(), (
        "âncora: o grupo F tem ponto desenhável, e o caso não foi exercitado"
    )
    eixo = _figura_agrupada(escores).axes[0]
    assert f"F — 0 amostras\n({NOTA_SEM_REGIAO}: menos de {MINIMO_PARA_ELIPSE})" in _legenda_de(
        eixo
    ), _legenda_de(eixo)
    assert not any(str(linha.get_label()).startswith("F —") for linha in eixo.lines)


@pytest.mark.parametrize("com_estilos", [False, True], ids=["ciclo", "estilos-da-coorte"])
def test_por_grupo_cada_serie_tem_um_PAR_proprio_e_a_regiao_a_cor_do_grupo(  # noqa: N802
    com_estilos: bool,
) -> None:
    """O estilo escolhido aplicado, e o QC e a validação com um par que nenhum grupo usa.

    **Sem estilo, as cores do ciclo**, como em todo construtor que pinta por grupo sem escolha;
    com ele, a cor e o símbolo da escolha. **Nos dois casos nenhuma série partilha o par com
    outra** — o QC com a cor de um grupo leria-se como mais uma amostra desse grupo, que é a
    pergunta inteira da figura. E a região de cada grupo é tracejada **na cor dele**.
    """
    escolha = {"B": EstiloDoGrupo(cor="rosa", simbolo="quadrado")}
    estilos = estilos_dos_grupos(("A", "B", "C"), escolha) if com_estilos else None
    eixo = _figura_agrupada(_escores_de(_INJECOES_AGRUPADAS), estilos=estilos).axes[0]
    linhas = {str(linha.get_label()): linha for linha in eixo.lines}

    serie_b, regiao_b = linhas["B — 4 amostras"], linhas[rotulo_da_regiao_do_grupo("B")]
    assert regiao_b.get_color() == serie_b.get_color()
    assert regiao_b.get_linestyle() == "--"
    if com_estilos:
        assert serie_b.get_color() == CORES_DOS_GRUPOS["rosa"]
        assert serie_b.get_marker() == SIMBOLOS_DOS_GRUPOS["quadrado"]

    pontos = {
        rotulo: (linha.get_color(), linha.get_marker())
        for rotulo, linha in linhas.items()
        if not rotulo.startswith(rotulo_da_regiao_do_grupo(""))
    }
    assert len(pontos) == 5, sorted(pontos)
    assert len(set(pontos.values())) == len(pontos), pontos

    # **Por baixo de TODOS os pontos, o QC incluído.** A região desenha-se depois das séries — só
    # então o quadro existe para a recortar —, e sem a ordem explícita o tracejado ficaria por cima.
    regioes = [linha for rotulo, linha in linhas.items() if rotulo not in pontos]
    assert regioes, "âncora: nenhuma região desenhada, e a asserção abaixo seria vácua"
    acima = min(linhas[rotulo].get_zorder() for rotulo in pontos)
    assert all(regiao.get_zorder() < acima for regiao in regioes), [
        (str(r.get_label()), r.get_zorder()) for r in regioes
    ]


def test_por_grupo_sem_estilo_e_o_ciclo_DANDO_A_VOLTA_o_QC_NAO_repete_par_de_grupo() -> None:  # noqa: N802
    """Nove grupos sem estilo: o ciclo tem dez cores, e o QC sairia com o par do primeiro grupo.

    Sem a tabela dos grupos — a composição ainda não chegou à
    janela, ou a coorte passa dos pares de estilo —, o grupo `i` leva a cor `i` do ciclo, e a
    validação e o QC as duas seguintes. Com nove grupos o QC é a 11.ª cor de um ciclo de dez, que é
    a do primeiro grupo, e com o mesmo círculo: o QC leria-se como mais uma amostra dele. **Entre os
    grupos o par repete-se a partir da 11.ª** — é a forma de `series.series_por_grupo` sem estilo,
    que esta figura não muda —; o que não se repete é o par do QC e o da validação.

    **A âncora:** o ciclo vigente tem menos cores do que os nove grupos e as duas séries de tipo,
    senão a volta não acontece e a asserção não separa nada.
    """
    n_cores = len({to_hex(c) for c in cores_do_ciclo(11)})
    assert n_cores < 11, f"âncora: o ciclo tem {n_cores} cores, e não dá a volta com 11 séries"
    injecoes: tuple[tuple[str, str | None, float, float], ...] = (
        *(("sample", f"G{i}", float(i), float(i % 3)) for i in range(9)),
        ("qc", None, 4.0, 1.0),
        ("validate", None, 2.0, 2.0),
    )
    eixo = _figura_agrupada(_escores_de(injecoes)).axes[0]
    pares = {
        str(linha.get_label()): (to_hex(linha.get_color()), linha.get_marker())
        for linha in eixo.lines
    }
    dos_grupos = {v for rotulo, v in pares.items() if rotulo.startswith("G")}
    assert len(dos_grupos) == 9, pares
    do_qc, da_validacao = pares["QC — 1 amostra"], pares["amostra de validação — 1 amostra"]
    assert do_qc not in dos_grupos, (do_qc, pares)
    assert da_validacao not in dos_grupos, (da_validacao, pares)
    assert do_qc != da_validacao, pares


def test_contem_concorda_com_a_FORMA_QUADRATICA_em_regioes_RODADAS_e_perto_da_borda() -> None:  # noqa: N802
    """`Elipse.contem` contra `(p - xbar)' S^-1 (p - xbar) <= c²`, calculado dos pontos.

    O único teste que usava este oráculo corria sobre uma fixtura de
    covariância DIAGONAL — ângulo de 0 ou 90 graus — e com os cantos do quadro longe da borda, e
    duas plantas passavam por ele: a rotação com o sinal trocado e os semieixos trocados. **Aqui as
    regiões são rodadas e alongadas**, e os pontos ficam a 3% da borda, para dentro e para fora,
    em 16 direções: a borda é a da forma quadrática, `xbar + u·sqrt(c² / (u' S^-1 u))`, que não
    passa pelos semieixos nem pelo ângulo da `Elipse`.

    **Semente explícita (regra 3 do README)**, e a âncora é que as regiões não são círculos nem
    estão alinhadas aos eixos — senão as duas plantas voltam a passar.
    """
    gerador = np.random.default_rng(20261107)
    angulos = np.linspace(0.0, 2.0 * math.pi, 16, endpoint=False)
    rodadas = 0
    for _ in range(40):
        n = int(gerador.integers(3, 9))
        fator = gerador.normal(size=(2, 2))
        pontos = gerador.normal(size=(n, 2)) @ fator + gerador.normal(scale=3.0, size=2)
        regiao = elipse_de_hotelling(pontos[:, 0], pontos[:, 1])
        assert isinstance(regiao, Elipse)
        a, b = regiao.semieixos
        rodadas += int(a > 1.2 * b and abs(math.sin(2.0 * math.radians(regiao.angulo_graus))) > 0.3)

        xbar = pontos.mean(axis=0)
        inversa = np.linalg.inv(np.cov(pontos, rowvar=False))
        for angulo in angulos:
            u = np.array([math.cos(angulo), math.sin(angulo)])
            borda = math.sqrt(regiao.c2 / float(u @ inversa @ u))
            dentro, fora = xbar + 0.97 * borda * u, xbar + 1.03 * borda * u
            assert regiao.contem(*dentro), (regiao, angulo)
            assert not regiao.contem(*fora), (regiao, angulo)
    assert rodadas >= 10, f"âncora: só {rodadas} das 40 regiões são rodadas e alongadas"


def test_mascara_do_grupo_deixa_o_QC_de_FORA_mesmo_com_nome_de_grupo_no_vetor() -> None:  # noqa: N802
    """A defesa por construção de `mascara_do_grupo`, exercitada sem passar pela fronteira.

    `escores_da_pca` já recusa um QC com grupo, então a única forma de a ver agir é montar os
    `Escores` à mão: um vetor de grupos com `"A"` também na posição do QC. **A âncora** é que o
    vetor de grupos, sozinho, põe o QC no grupo — senão a asserção não separa a máscara do tipo de
    nada.
    """
    escores = Escores(
        x=np.array([0.0, 1.0, 2.0]),
        y=np.array([0.0, 1.0, 2.0]),
        tipos=(TipoDeAmostra.QC, TipoDeAmostra.AMOSTRA, TipoDeAmostra.AMOSTRA),
        grupos=("A", "A", "B"),
    )
    assert escores.grupos is not None and escores.grupos[0] == "A", "âncora: o QC não tem nome"
    assert escores.mascara_do_grupo("A").tolist() == [False, True, False]


def test_por_grupo_regioes_de_OUTROS_grupos_sao_recusadas_pela_figura() -> None:  # noqa: N802
    """A guarda de `_desenha_por_grupo` na metade em que os dois vetores TÊM grupo, e não os mesmos.

    **As regiões de um subconjunto dos grupos** — calculadas sobre escores sem o C — e a figura dos
    escores inteiros: sem a guarda, o C desenhava-se como série e não entrava na legenda, e a figura
    ficava bem formada. **O par:** as regiões dos mesmos escores desenham.
    """
    escores = _escores_de(_INJECOES_AGRUPADAS)
    sem_c = _escores_de(tuple(i for i in _INJECOES_AGRUPADAS if i[1] != "C"))
    regioes = regioes_dos_grupos(sem_c)
    assert regioes.grupos == ("A", "B") and escores.grupos_de_estudo == ("A", "B", "C")

    def desenha(regiao: RegioesDosGrupos) -> Figure:
        return figura_dos_scores(
            escores=escores,
            variancia_explicada=[0.41, 0.17],
            regiao=regiao,
            recorte=recorte_dos_scores(n_features_usadas=40, n_features_totais=42, regiao=regiao),
        )

    assert desenha(regioes_dos_grupos(escores)).axes
    with pytest.raises(ValueError, match="as regiões são dos grupos"):
        desenha(regioes)


def test_por_grupo_o_rodape_DIZ_quando_nenhum_grupo_tem_regiao_ou_nao_ha_grupo() -> None:  # noqa: N802
    """Os dois ramos do rodapé sem faixa de c²: não há c² a dizer, e a frase diz porquê.

    **Os dois são fato do dado:** todos os grupos abaixo do mínimo, e um vetor de grupos sem
    nenhuma amostra de estudo — só QC e validação. Sem eles, a faixa sairia de `min` sobre uma lista
    vazia, e a figura inteira seria recusada por um `ValueError` sem frase.
    """
    pequenos = _escores_de(
        (
            ("sample", "A", 0.0, 0.0),
            ("sample", "A", 1.0, 0.5),
            ("sample", "B", 2.0, 1.0),
            ("qc", None, 1.0, 1.0),
        )
    )
    so_tipos = _escores_de((("qc", None, 0.0, 0.0), ("validate", None, 1.0, 1.0)))
    for escores, frase in (
        (pequenos, "nenhum grupo de estudo sustenta a região"),
        (so_tipos, "não há amostra de estudo com grupo"),
    ):
        regioes = regioes_dos_grupos(escores)
        assert not any(isinstance(r.regiao, Elipse) for r in regioes.por_grupo)
        assert frase in " ".join(rodape_de(_figura_agrupada(escores)).split())


def test_por_grupo_o_rodape_diz_UMA_POR_GRUPO_e_NAO_e_mais_longo_que_o_da_regiao_unica() -> None:  # noqa: N802
    """O método no rodapé, com a faixa de c², e sem uma linha a mais que o da região única no
    padrão.

    **A segunda metade é a que sustenta a redação.** O rodapé dos scores é o mais cheio do pacote e
    `aplica` recusa a figura acima de `RESERVA_MAXIMA` no tamanho de construção; um c² por grupo e
    os grupos sem região por extenso cresceriam uma oração por grupo. **O par é a mesma figura, com
    o mesmo par da PERMANOVA, nas duas formas**, e a por grupo não pode ter linha a mais.
    """
    escores = _escores_de(_INJECOES_AGRUPADAS)
    regioes = regioes_dos_grupos(escores)
    texto = " ".join(
        recorte_dos_scores(
            n_features_usadas=40, n_features_totais=42, regiao=regioes, permanova=PAR_DO_ESTUDO
        )
        .texto()
        .split()
    )
    c2s = [r.regiao.c2 for r in regioes.por_grupo if isinstance(r.regiao, Elipse)]
    assert len(c2s) == 2, "âncora: a fixtura tem dois grupos com região"
    assert "uma por grupo" in texto, texto
    assert "recortadas ao quadro" in texto, texto
    assert f"c² de {decimal(min(c2s))} a {decimal(max(c2s))}" in texto, texto
    assert "qui-quadrado" in texto, "a divergência declarada contra o MetaboAnalyst saiu do rodapé"

    agrupada = _figura_agrupada(escores, permanova=PAR_DO_ESTUDO)
    assert rodape_de(agrupada) == texto

    sem_grupo = escores_da_pca(pc1=escores.x, pc2=escores.y, tipos=[t.value for t in escores.tipos])
    unica = regiao_dos_scores(sem_grupo)
    assert isinstance(unica, Elipse), "âncora: a região única não existe, e o par não compara nada"
    de_d420 = figura_dos_scores(
        escores=sem_grupo,
        variancia_explicada=[0.41, 0.17],
        regiao=unica,
        recorte=recorte_dos_scores(
            n_features_usadas=40, n_features_totais=42, regiao=unica, permanova=PAR_DO_ESTUDO
        ),
    )
    linhas = {
        nome: figura.texts[0].get_text().count("\n") + 1
        for nome, figura in (("por grupo", agrupada), ("região única", de_d420))
    }
    assert linhas["por grupo"] <= linhas["região única"], linhas


def test_por_grupo_regioes_de_grupo_sobre_escores_SEM_grupo_e_erro_de_quem_chama() -> None:  # noqa: N802
    """Sem o vetor de grupos, a região é a única — e pedir as de grupo é erro de programa.

    **O par:** `regiao_dos_scores` sobre os mesmos pontos sem grupo devolve a região de
    `elipse_das_amostras`, a de antes; e uma figura que recebesse regiões por grupo sobre escores
    sem grupo desenharia regiões de uma população que ela não mostra.
    """
    sem_grupo = escores_da_pca(
        pc1=[x for _, _, x, _ in _INJECOES_AGRUPADAS],
        pc2=[y for _, _, _, y in _INJECOES_AGRUPADAS],
        tipos=[tipo for tipo, _, _, _ in _INJECOES_AGRUPADAS],
    )
    assert sem_grupo.grupos is None
    assert regiao_dos_scores(sem_grupo) == elipse_das_amostras(sem_grupo)
    with pytest.raises(ValueError, match="grupo"):
        regioes_dos_grupos(sem_grupo)

    regioes = regioes_dos_grupos(_escores_de(_INJECOES_AGRUPADAS))
    with pytest.raises(ValueError, match="grupo"):
        figura_dos_scores(
            escores=sem_grupo,
            variancia_explicada=[0.41, 0.17],
            regiao=regioes,
            recorte=recorte_dos_scores(n_features_usadas=40, n_features_totais=42, regiao=regioes),
        )


def test_por_grupo_no_corpus_real_cada_grupo_tem_regiao_e_o_QC_fica_DE_FORA(  # noqa: N802
    matriz_real: RefinoMatrix,
) -> None:
    """O `pos_hilic` do laboratório, com o grupo de cada amostra tirado de `sample_mask`.

    **A âncora vai primeiro:** o vetor `groups` da matriz traz, nas posições de QC, um
    rótulo que não é de grupo de estudo — sem ele no dado, *"o QC não vira grupo"* passaria por
    vacuidade. As contagens esperadas vêm de `sample_mask` e de `qc_mask`, e não de literais:
    quatro grupos e sete QCs são observação sobre estes arquivos.
    """
    estudo = matriz_real.sample_mask.tolist()
    pares = list(zip(matriz_real.groups, estudo, strict=True))
    fora_do_estudo = {g for g, e in pares if not e}
    do_estudo = sorted({g for g, e in pares if e})
    assert fora_do_estudo - set(do_estudo), (
        "âncora: nenhuma injeção fora do estudo traz rótulo próprio, e a asserção do QC seria vácua"
    )

    p = structural_pca(matriz_real, pretransform=PRETRANSFORM, n_components=2)
    escores = escores_da_pca(
        pc1=p.scores[:, 0],
        pc2=p.scores[:, 1],
        tipos=[s.sample_type for s in matriz_real.samples],
        grupos=[g if e else None for g, e in pares],
    )
    regioes = regioes_dos_grupos(escores)
    assert list(regioes.grupos) == do_estudo
    assert not set(regioes.grupos) & fora_do_estudo
    for regiao in regioes.por_grupo:
        assert isinstance(regiao.regiao, Elipse), regiao.grupo
        assert regiao.n == sum(1 for g, e in pares if e and g == regiao.grupo), regiao.grupo

    agrupada = _figura_agrupada(escores, permanova=PAR_DO_ESTUDO)
    legenda = _legenda_de(agrupada.axes[0])
    n_validacao = int(matriz_real.validate_mask.sum())
    assert len(legenda) == len(do_estudo) + 1 + (1 if n_validacao else 0), legenda
    assert f"QC — {milhar(int(matriz_real.qc_mask.sum()))} amostras" in legenda, legenda
    # A entrada é o rótulo do grupo, com o motivo entre parênteses quando a região dele não se vê.
    # **O teste aceita as duas formas e não afirma qual sai em que grupo:** esta forma
    # do teste nunca correu contra os dados.
    for regiao in regioes.por_grupo:
        rotulo = rotulo_do_grupo(regiao.grupo, regiao.n)
        assert any(t in (rotulo, f"{rotulo}\n({NOTA_MAIOR_QUE_O_QUADRO})") for t in legenda), (
            legenda
        )
    assert not agrupada.axes[0].texts, [t.get_text() for t in agrupada.axes[0].texts]

    # **O rodapé, sobre a população em que a redação foi medida:** no tamanho de
    # construção, com o par do estudo, o da figura por grupo não tem linha a mais que o da região
    # única.
    sem_grupo = escores_da_pca(pc1=escores.x, pc2=escores.y, tipos=[t.value for t in escores.tipos])
    unica = regiao_dos_scores(sem_grupo)
    de_d420 = figura_dos_scores(
        escores=sem_grupo,
        variancia_explicada=[0.41, 0.17],
        regiao=unica,
        recorte=recorte_dos_scores(
            n_features_usadas=40, n_features_totais=42, regiao=unica, permanova=PAR_DO_ESTUDO
        ),
    )
    assert agrupada.texts[0].get_text().count("\n") <= de_d420.texts[0].get_text().count("\n")
