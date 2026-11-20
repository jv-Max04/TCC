"""A varredura um-fator-por-vez, e o que ela achou sobre a grade do multiverso.

**O que esta suíte fixa não é a medição, é a REGRA que a medição descobriu.** A varredura
mediu `d(A,B) = ||A - B||_F / ||B||_F` entre a matriz de cada braço e a da linha de base
sobre os quatro modos reais, com quatro desfechos declarados antes de rodar. Dois
dispararam, e são estes:

| Desfecho | O que disparou | O que este arquivo afirma |
|---|---|---|
| 3 — um eixo domina | `scaling`, máxima 113x a menor | a dominância, e a **causa** dela |
| 4 — opções idênticas | `log10` ≡ `log2` nos quatro modos | a regra fechada, nas duas direções |

**O desfecho 4 é o achado que justifica a unidade, e ele tem forma fechada.** `log2` é
`log10` dividido por `log10(2)` — fator **constante**. `auto` centra e divide pelo desvio,
o que remove exatamente qualquer fator constante; `range` e `vast` também são invariantes a
ele. Então `log10` e `log2` são o **mesmo ponto** da grade sob três dos cinco
escalonamentos, e pontos distintos sob os outros dois — com distância que a álgebra prevê
e que não depende do dado:

    scaling='none'   →  d = |1/log10(2) - 1|       = 2,3219
    scaling='pareto' →  d = |1/sqrt(log10(2)) - 1| = 0,8226

Os dois números saíram **idênticos nos quatro modos**, a quinze dígitos, que é a assinatura
de propriedade que só depende da constante. É por isso que o teste afirma a **derivação** e
não a medição: número derivado não é número congelado, e a regra 2 do README não tem
o que absorver aqui.

**A consequência para o multiverso é sobre ALARGAR a grade, não sobre encolher a de hoje.** A grade
padrão é `GridDefinition().size()` células e `log2` **não está** no eixo das
transformações. **Este parágrafo dizia `1.296` células e `("none", "log10")` até
08/08/2026, e os DOIS estavam caducos**: medido, são **4.320** e
`("none", "log10", "sqrt", "cbrt")`. O segundo é o mais instrutivo — a lista literal de um eixo
caduca **em silêncio** quando o eixo cresce, e nada aqui dependia dela para o argumento.

Então a regra não economiza execução hoje; ela (a) justifica a omissão
de `log2`, que sob `auto` seria célula duplicada de `log10`, e (b) diz o preço de acrescentá-lo:
9 pares `(transformation, scaling)` dos quais **8** distintos, um terço do acréscimo em
duplicata. `range` e `vast` são invariantes pelo mesmo argumento, então alargar o eixo do
escalonamento multiplica a duplicata em vez de diluí-la.

A primeira redação deste arquivo dizia *"625 → 550, 12% menos"*. O 625 era 5⁴ — aritmética sobre
o espaço de opções em vez do número que o critério declara, e o número estava à mão.

**O desfecho 3 disparou e a causa NÃO é a que o número sugere** — e é aqui que o rótulo
certo sobre a quantidade errada morde o próprio instrumento. `d` não é invariante a reescala global
do braço, então "esta etapa move mais" e "esta etapa muda de unidade" são indistinguíveis nele. A
decomposição `r = ||A||/||B||` e `cos = <A,B>/(||A|| ||B||)` separa os dois, e o resultado inverte a
leitura: `cos(vast, auto)` é **negativo** — de -0,87 a -0,93 —, porque sob a ordem canônica
`sum` → `log10` os valores são log de frações e a **média por feature é negativa**, então o
fator `média/desvio` do `vast` **inverte o sinal** e amplifica por |média|/desvio. E o braço
estruturalmente mais distante não é o `vast`: é `scaling='none'`, com `cos` entre 0,05 e
0,23 e `d` de apenas 1,8 a 4,8. **A medida e a estrutura elegem opções diferentes**, e um
teste que só afirmasse `d` esconderia isso.

**O defeito de escopo que esta suíte impede de voltar.** A primeira varredura declarou "100
features amostradas por modo" — o que descreve a **entrada** da cascata de filtros, não a
população medida. A cascata deixava 8, 77, 9 e 71, e sobre 8 features seis pares de
imputação apareceram como "idênticos" quando a causa era não haver célula ausente para
imputar naquele recorte. Refeita na matriz cheia, com o sorteio **depois** da cascata, os
seis pares desapareceram e os quatro estruturais ficaram. Daí o quarto teste: ele afirma
que no ponto da imputação **há o que imputar** nos quatro modos, e reprova se alguém
encolher a população para onde a etapa não tem efeito.
"""

import math

import numpy as np
import pytest

from refino.correction.base import select_features
from refino.domain.matrix import RefinoMatrix
from refino.domain.spec import PipelineSpec
from refino.io.readers import load_matrix
from refino.metrics.crossval import sampled_feature_names
from refino.multiverse.grid import GridDefinition
from refino.preprocessing.filtering import filter_by_qc_rsd, filter_near_zero_variance
from refino.preprocessing.missing import drop_by_group_missingness, is_missing
from refino.preprocessing.pipeline import process_matrix
from tests.conftest import measurements_path, sample_list_path

pytestmark = [pytest.mark.integration, pytest.mark.slow]

MODOS = ("pos_hilic", "pos_c18", "neg_hilic", "neg_c18")

# A linha de base da varredura: a especificação padrão do laboratório.
BASE = {
    "imputation": "one_fifth_min",
    "sample_normalization": "sum",
    "normalization_reference": None,
    "normalization_factors": None,
    "transformation": "log10",
    "scaling": "auto",
}
CORTE_RSD, LIMITE_AUSENTES, SEMENTE = 0.30, 0.8, 42

# Os escalonamentos invariantes a fator constante multiplicativo, e os que não são. A
# partição é DERIVADA e está no teste como derivação: `auto` divide pelo desvio, `range`
# pela amplitude e `vast` é `auto` vezes um quociente de duas quantidades que escalam
# juntas — os três anulam o fator. `none` não divide por nada e `pareto` divide pela
# **raiz** do desvio, que anula só metade do fator.
INVARIANTES = ("auto", "range", "vast")
SENSIVEIS = ("none", "pareto")

# `d` abaixo de 1e-12 é "a mesma matriz": a coincidência é exata em aritmética real e o
# resíduo medido foi de 8e-16 a 6e-15, quinze ordens de grandeza abaixo do limiar.
IDENTICAS = 1e-12


def _spec(**trocas: object) -> PipelineSpec:
    return PipelineSpec(
        correction="none",
        n_batches=1,
        rsd_cutoff=CORTE_RSD,
        ranking_filter="none",
        ranking_keep_fraction=None,
        ranking_filter_2="none",
        ranking_keep_fraction_2=None,
        rsd_variant="outlier_removed",
        missing_threshold=LIMITE_AUSENTES,
        seed=SEMENTE,
        **{**BASE, **trocas},  # type: ignore[arg-type]
    )


def _d(a: np.ndarray, b: np.ndarray) -> float:
    """A medida da varredura, escolhida antes de qualquer resultado."""
    x, y = np.asarray(a).ravel(), np.asarray(b).ravel()
    norma = float(np.linalg.norm(y))
    return float(np.linalg.norm(x - y) / norma) if norma > 0 else float("nan")


@pytest.fixture(scope="module")
def matrizes() -> dict[str, RefinoMatrix]:
    """As quatro matrizes reais, carregadas uma vez para toda a suíte."""
    return {m: load_matrix(measurements_path(m), sample_list_path(m), m) for m in MODOS}


@pytest.fixture(scope="module")
def retidas(matrizes: dict[str, RefinoMatrix]) -> dict[str, RefinoMatrix]:
    """A matriz no ponto em que a imputação roda — depois dos três filtros.

    Existe como fixture porque é a **população** da varredura, e a primeira versão da
    medição errou justamente ao confundi-la com a entrada da cascata.
    """
    saida = {}
    for modo, matriz in matrizes.items():
        atual, _, _ = filter_by_qc_rsd(matriz, CORTE_RSD, variant="outlier_removed")
        atual, _, _ = filter_near_zero_variance(atual)
        atual, _, _ = drop_by_group_missingness(atual, LIMITE_AUSENTES)
        saida[modo] = atual
    return saida


# ---------------------------------------------------------------------------
# Desfecho 4: a regra fechada, e as duas direções dela
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("modo", MODOS)
@pytest.mark.parametrize("escalonamento", INVARIANTES)
def test_log10_e_log2_sao_o_mesmo_ponto_sob_escalonamento_invariante(
    matrizes: dict[str, RefinoMatrix], modo: str, escalonamento: str
) -> None:
    """A metade que COLAPSA: sob `auto`, `range` e `vast` as duas transformações coincidem.

    Se este teste passar a reprovar, alguma das três invariâncias se perdeu,
    e o multiverso passaria a pagar célula duplicada ao incluir as duas bases de log — que é o que
    a regra existe para dizer. Mudança de comportamento numa etapa, não ruído.
    """
    pedido = {"scaling": escalonamento}
    esquerda, _ = process_matrix(matrizes[modo], _spec(transformation="log10", **pedido))
    direita, _ = process_matrix(matrizes[modo], _spec(transformation="log2", **pedido))

    distancia = _d(direita.values, esquerda.values)
    assert distancia < IDENTICAS, (
        f"{modo}/{escalonamento}: log10 e log2 divergiram em {distancia:.3e}; sob "
        f"escalonamento invariante a fator constante os dois têm de ser o MESMO ponto da "
        f"grade, e é isso que justifica `log2` ficar fora da grade padrão de "
        f"{GridDefinition().size()}"
    )


@pytest.mark.parametrize("modo", MODOS)
def test_log10_e_log2_divergem_pelo_fator_que_a_algebra_preve(
    matrizes: dict[str, RefinoMatrix], modo: str
) -> None:
    """A metade que SEPARA — o par causal, e a razão de o colapso não ser vácuo.

    Sem esta direção, o teste acima passaria também se `transform` ignorasse a base do
    logaritmo, ou se `log2` fosse um apelido de `log10`. O que a distingue é que aqui os
    dois **têm** de divergir, por um fator que a álgebra dá de antemão:

        `none`   → o braço é `log10/c`, logo d = |1/c - 1| com c = log10(2)
        `pareto` → o braço escala por `sqrt(1/c)`, logo d = |1/sqrt(c) - 1|

    Valor **derivado** e não medido, e é o que o torna afirmável sem congelar medição: a
    tolerância é de arredondamento, não de concordância.
    """
    c = math.log10(2.0)
    previsto = {"none": abs(1.0 / c - 1.0), "pareto": abs(1.0 / math.sqrt(c) - 1.0)}

    for escalonamento in SENSIVEIS:
        esquerda, _ = process_matrix(
            matrizes[modo], _spec(transformation="log10", scaling=escalonamento)
        )
        direita, _ = process_matrix(
            matrizes[modo], _spec(transformation="log2", scaling=escalonamento)
        )
        medido = _d(direita.values, esquerda.values)
        assert medido == pytest.approx(previsto[escalonamento], rel=1e-9), (
            f"{modo}/{escalonamento}: d medido {medido:.6f} contra {previsto[escalonamento]:.6f} "
            f"previsto pela álgebra do fator log10(2). Divergência aqui não é tolerância "
            f"apertada: é a forma estabilizada do log ou o escalonamento mudando de política"
        )


# ---------------------------------------------------------------------------
# Desfecho 3: a dominância, e a causa que o número esconde
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("modo", MODOS)
def test_o_escalonamento_domina_a_grade_sob_a_medida_pre_registrada(
    matrizes: dict[str, RefinoMatrix], modo: str
) -> None:
    """O desfecho 3, afirmado como ESTRUTURA do resultado e não como número medido.

    O que se afirma é a ordenação — a máxima do `scaling` acima de 10x a mediana das outras
    três etapas —, que é o limiar declarado antes da medição. Congelar 113,1x seria fixar
    um número que depende da amostra; a ordenação é o que muda como o multiverso amostra.
    """
    maximas: dict[str, float] = {}
    # Só as quatro etapas da varredura. A referência e os fatores da normalização
    # são parâmetros sem opções a varrer, e ficam no `BASE`; quando entraram também aqui, como
    # `None`, o laço passou a levantar `TypeError` nos quatro modos antes de medir.
    opcoes = {
        "imputation": ("one_fifth_min", "knn", "median"),
        "sample_normalization": ("none", "sum", "median", "pqn_group", "quantile"),
        "transformation": ("none", "log10", "log2", "sqrt", "cbrt"),
        "scaling": ("none", "auto", "pareto", "range", "vast"),
    }
    base, _ = process_matrix(matrizes[modo], _spec())
    for etapa, valores in opcoes.items():
        distancias = []
        for opcao in valores:
            braco, _ = process_matrix(matrizes[modo], _spec(**{etapa: opcao}))
            distancias.append(_d(braco.values, base.values))
        maximas[etapa] = max(distancias)

    outras = float(np.median([v for e, v in maximas.items() if e != "scaling"]))
    assert maximas["scaling"] >= 10.0 * outras, (
        f"{modo}: a máxima do `scaling` ({maximas['scaling']:.4f}) deixou de dominar a "
        f"mediana das outras ({outras:.4f}); o desfecho 3 da U7 é o que justifica a Fase 8 "
        f"amostrar o eixo do escalonamento antes dos outros três"
    )


@pytest.mark.parametrize("modo", MODOS)
def test_a_dominancia_do_vast_e_inversao_de_sinal_e_nao_reordenacao(
    matrizes: dict[str, RefinoMatrix], modo: str
) -> None:
    """A causa que `d` sozinho não mostra, e a razão de a decomposição existir.

    Duas afirmações, e a segunda é a mecânica da primeira:

    1. `cos(vast, auto) < 0` — os dois braços apontam em direções **opostas**.
    2. a maioria das features tem média negativa sob a ordem canônica, e é isso que faz o
       fator `média/desvio` do `vast` ser negativo.

    Sem a segunda, a primeira seria observação sem mecanismo, e a explicação plausível
    passaria pela mesma porta da evidência.
    """
    base, _ = process_matrix(matrizes[modo], _spec())
    braco, _ = process_matrix(matrizes[modo], _spec(scaling="vast"))
    a, b = np.asarray(braco.values).ravel(), np.asarray(base.values).ravel()
    cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))

    assert cos < 0.0, (
        f"{modo}: cos(vast, auto) = {cos:.4f}, não negativo. A dominância do `vast` sob `d` "
        f"seria então reordenação e não inversão de sinal, e a leitura do desfecho 3 muda"
    )

    # A mecânica: o sinal do fator do `vast` é o sinal da média da feature, e sob
    # `sum` → `log10` os valores são log de frações, todas menores que 1. A média que
    # decide é a de ANTES do escalonamento — em `base` ela é ~0 por construção do `auto`,
    # então medi-la ali responderia outra pergunta.
    sem_escalonar, _ = process_matrix(matrizes[modo], _spec(scaling="none"))
    medias = np.asarray(sem_escalonar.values).mean(axis=0)
    negativas = int((medias < 0).sum())
    assert negativas > medias.size / 2, (
        f"{modo}: só {negativas} de {medias.size} features com média negativa antes do "
        f"escalonamento. O sinal do fator do `vast` é o da média, e sem maioria negativa a "
        f"inversão medida não teria a causa que este teste atribui"
    )


# ---------------------------------------------------------------------------
# O defeito de escopo, virado em portão
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("modo", MODOS)
def test_no_ponto_da_imputacao_ha_o_que_imputar(
    retidas: dict[str, RefinoMatrix], modo: str
) -> None:
    """O portão que impede o defeito de escopo da primeira varredura de voltar.

    A imputação só é comparável entre métodos se a população tiver célula ausente. Sobre a
    subamostra de 100 features **sorteada antes da cascata**, o `pos_c18` retinha 77
    features e **zero** ausentes, e as quatro imputações saíram idênticas — o que se lê
    como "a etapa é inerte" e era "o recorte não tem o que imputar".

    As contagens são OBSERVAÇÃO sobre os quatro arquivos, então o que se afirma é
    a existência, não o número: um quinto modo tem a sua própria contagem.
    """
    ausentes = int(is_missing(np.asarray(retidas[modo].values)).sum())
    assert ausentes > 0, (
        f"{modo}: zero células ausentes no ponto da imputação, sobre "
        f"{retidas[modo].n_features} features retidas. Uma varredura de imputação sobre "
        f"esta população mediria a ausência de dado a imputar e reportaria a etapa como "
        f"inerte — que é exatamente o artefato que a U7 achou e corrigiu"
    )


@pytest.mark.parametrize("modo", MODOS)
def test_as_imputacoes_diferem_quando_ha_o_que_imputar(
    matrizes: dict[str, RefinoMatrix], modo: str
) -> None:
    """A outra metade do par: com população adequada, os métodos **não** coincidem.

    E o contraste é construído, não suposto — a mesma comparação sobre a subamostra
    sorteada **antes** da cascata é o que produziu a coincidência. Este teste afirma que a
    diferença existe na matriz cheia; a coincidência da subamostra é o registro.
    """
    base, _ = process_matrix(matrizes[modo], _spec())
    for metodo in ("knn", "median"):
        braco, _ = process_matrix(matrizes[modo], _spec(imputation=metodo))
        distancia = _d(braco.values, base.values)
        assert distancia > IDENTICAS, (
            f"{modo}: {metodo} deu matriz idêntica a one_fifth_min sobre a matriz cheia. "
            f"Ou a população perdeu as células ausentes, ou `impute` deixou de distinguir "
            f"os métodos — as duas mudam o que a Fase 8 pode concluir sobre o eixo"
        )


def test_a_subamostra_pre_cascata_e_o_que_enganou_e_esta_documentado(
    matrizes: dict[str, RefinoMatrix],
) -> None:
    """O par causal do defeito de escopo: a coincidência reproduzida, de propósito.

    Um teste que só afirmasse "há o que imputar na matriz cheia" não distinguiria "o
    defeito era o recorte" de "o defeito nunca existiu". Este reproduz o recorte original —
    100 features sorteadas **antes** dos filtros, semente 42 — e afirma que sobre ele o
    `pos_c18` fica sem célula ausente. É a evidência de que a causa atribuída é a
    causa, e não uma explicação plausível para um artefato real.
    """
    nomes = sampled_feature_names(matrizes["pos_c18"], 100, seed=SEMENTE)
    assert nomes is not None
    subamostra = select_features(matrizes["pos_c18"], nomes)

    atual, _, _ = filter_by_qc_rsd(subamostra, CORTE_RSD, variant="outlier_removed")
    atual, _, _ = filter_near_zero_variance(atual)
    atual, _, _ = drop_by_group_missingness(atual, LIMITE_AUSENTES)

    assert int(is_missing(np.asarray(atual.values)).sum()) == 0, (
        "o recorte de 100 features sorteado ANTES da cascata deixou de ser o caso "
        "degenerado que enganou a primeira varredura; se ele passou a ter células "
        "ausentes, o par causal perdeu a metade que atribui a causa"
    )
