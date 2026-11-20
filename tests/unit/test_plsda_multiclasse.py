"""`plsda_multiclass` — o um-contra-resto, e a regra 6 do README na classe nova.

## A ÂNCORA corre primeiro, e é ela que torna o resto legível

As entradas de um arnês novo se validam contra âncora conhecida **antes** de ler o
resultado; depois, a verificação vira defesa de um número que já se quer acreditar. O
oráculo exato aqui é `statistics.plsda`: com **dois** grupos o um-contra-resto é a mesma
matemática, e a âncora afirma desvio **zero** — não «próximo».

## Escopo negativo deste arquivo

**Nenhum teste corre sobre os arquivos do laboratório.** O custo foi medido no `pos_hilic` real —
61 amostras de estudo x 6.035 features, quatro grupos — em **77,3 s** para 1000 permutações, e isso
não é orçamento de teste de unidade. O que estes testes afirmam são propriedades da função sobre
matrizes com verdade conhecida por construção; o caminho com dado real fica para quem o ligar.

**O `nan` de `_vip` não é reafirmado aqui.** A função é **importada** de `statistics.plsda` e o par
causal dela vive em `tests/unit/test_plots_plsda.py::
test_o_vip_do_nucleo_devolve_NAN_e_nao_zeros_quando_a_soma_de_quadrados_e_zero`. O que este arquivo
verifica é a **propagação**: que a ausência chega ao resultado como ausência e contada.

**A matriz com amostra `validate` é SINTÉTICA, e a declaração é requisito (regra 1 do README).**
Medido em 17/09/2026 sobre os quatro `measurements_*.csv` do laboratório: 68 linhas, 61 `sample`, 7
`qc` e **zero** `validate` nos quatro modos. A amostra `validate` de rótulo clínico **não existe no
corpus** — fabricá-la sobre os arquivos reais e apresentá-la como observação seria o que a
regra 1 do README proíbe, e por isso ela é construída à mão e dita como construída. O que os testes
abaixo afirmam é a propriedade de desenho, que não depende do corpus de hoje.

> **E a matriz constante que o plano sugeria como planta NÃO alcança esse ramo** — medido:
> `PLSRegression` sobre `X` inteiramente constante nem ajusta, levanta `ValueError: slice 0 has a
> NaN entry` dentro de `scipy.linalg.pinv`. É a mesma medição que a docstring daquele teste já
> regista, e vale igual aqui.
"""

import dataclasses

import numpy as np
import pytest

from refino.domain.cancellation import ExecutionCancelledError
from refino.domain.matrix import RefinoMatrix
from refino.domain.metadata import SampleMetadata
from refino.statistics import plsda_multiclasse
from refino.statistics.plsda import plsda
from refino.statistics.plsda_multiclasse import (
    MulticlassPLSDAResult,
    OneVsRestModel,
    plsda_multiclass,
)

pytestmark = pytest.mark.synthetic

# O desenho do bloco sintético, DECLARADO e não calibrado contra o desfecho. É o de
# `scripts/mede_q2_multiclasse.py`, reduzido no número de features porque aqui a pergunta é de
# propriedade e não de arquitetura.
N_POR_GRUPO = 15
N_FEATURES = 200
N_FEATURES_DE_SINAL = 10
DELTA = 1.5


def _matriz(valores: np.ndarray, grupos: list[str], tipos: list[str] | None = None) -> RefinoMatrix:
    tipos = tipos if tipos is not None else ["sample"] * len(grupos)
    amostras = tuple(
        SampleMetadata(name=f"S{i:03d}", injection_order=i + 1, sample_type=t, group=g)  # type: ignore[arg-type]
        for i, (g, t) in enumerate(zip(grupos, tipos, strict=True))
    )
    return RefinoMatrix(
        values=np.ascontiguousarray(valores, dtype=np.float64),
        feature_names=tuple(f"f{j:04d}" for j in range(valores.shape[1])),
        samples=amostras,
        mode="sintetico",
    )


def _com_sinal(semente: int, grupos: tuple[str, ...]) -> RefinoMatrix:
    """Ruído padrão com deslocamento conhecido em blocos **disjuntos** de features por grupo.

    Blocos disjuntos, e não um eixo único: um eixo único seria uma estrutura latente de baixa
    dimensão, e o desempenho mediria o desenho da matriz em vez do modelo. É o desenho.
    """
    gerador = np.random.default_rng([semente, 7])
    rotulos = [g for g in grupos for _ in range(N_POR_GRUPO)]
    valores = gerador.normal(size=(len(rotulos), N_FEATURES))
    vetor = np.array(rotulos)
    for k, grupo in enumerate(grupos):
        inicio = k * N_FEATURES_DE_SINAL
        valores[vetor == grupo, inicio : inicio + N_FEATURES_DE_SINAL] += DELTA
    return _matriz(valores, rotulos)


def _sem_sinal(semente: int, grupos: tuple[str, ...]) -> RefinoMatrix:
    gerador = np.random.default_rng([semente, 7])
    rotulos = [g for g in grupos for _ in range(N_POR_GRUPO)]
    return _matriz(gerador.normal(size=(len(rotulos), N_FEATURES)), rotulos)


QUATRO = ("A", "B", "C", "D")


# ---------------------------------------------------------------------------
# A ÂNCORA — corre antes de qualquer leitura do resultado multiclasse
# ---------------------------------------------------------------------------


def test_ancora_com_dois_grupos_o_um_contra_resto_reproduz_o_caminho_binario() -> None:
    """**Desvio ZERO contra o oráculo exato**, e nas quatro quantidades determinísticas.

    Com dois grupos, o alvo do modelo `B` é literalmente o `y` que `statistics.plsda` monta, e a
    estratificação pelo vetor de classe é a mesma partição que ele usa. Medido sobre **30 amostras
    (15 por grupo) x 200 features**, sinal de `+1,5 sd` em 10 features, semente 101: `q2`, `r2y` e
    `vip` saem **bit a bit** iguais.

    **O `p` fica FORA desta âncora de propósito.** Ele coincidiu na medição — os dois valem
    `1/21`, o **mínimo possível** com 20 permutações —, e afirmar igualdade ali seria concluir
    sobre a quantidade errada: o que a coincidência mostra é que nenhuma permutação supera um `Q²`
    de 0,716, não que os dois fluxos de semente sejam o mesmo. São streams diferentes por
    construção (o multiclasse deriva de três níveis, o binário de dois). Número
    disponível é a fonte mais comum de número errado.
    """
    matriz = _com_sinal(101, ("A", "B"))

    binario = plsda(matriz, ("A", "B"), n_components=2, cv_folds=7, n_permutations=20, seed=42)
    multi = plsda_multiclass(matriz, n_components=2, cv_folds=7, n_permutations=20, seed=42)

    assert multi.groups == ("A", "B")
    assert len(multi.models) == 2, "a âncora precisa dos dois modelos para ser não vácua"
    for modelo in multi.models:
        assert modelo.q2 == binario.q2, (
            f"o modelo `{modelo.group} vs resto` deu q2={modelo.q2!r} contra {binario.q2!r} do "
            "caminho binário, que com dois grupos é oráculo EXATO"
        )
        assert modelo.r2y == binario.r2y
        assert np.array_equal(modelo.vip, binario.vip), "o VIP divergiu do oráculo exato"


def test_ancora_o_agregado_nao_esconde_divergencia_porque_nao_existe_agregado() -> None:
    """**A segunda cláusula da âncora, e aqui ela é satisfeita por construção.**

    Na medição a segunda cláusula existia porque comparar só a **média** deixaria duas
    divergências simétricas de ±0,05 esconderem-se uma na outra. Este módulo não tem média: `q2` é
    uma tupla e nada a colapsa, então a planta que aquela cláusula precisava de detectar não é
    representável.

    **A asserção é de PRESENÇA e não de ausência**: afirma que os campos são coleções do
    tamanho certo, e não que «não existe agregado» — esta última passaria sobre um resultado vazio.
    """
    multi = plsda_multiclass(_com_sinal(202, QUATRO), n_permutations=20, seed=42)

    assert isinstance(multi.q2, tuple) and len(multi.q2) == 4
    assert isinstance(multi.permutation_pvalue, tuple) and len(multi.permutation_pvalue) == 4
    assert multi.q2 == tuple(m.q2 for m in multi.models), "a tupla desalinhou dos modelos"

    campos = {c.name for c in dataclasses.fields(MulticlassPLSDAResult)}
    assert campos == {
        "models",
        "groups",
        "n_components",
        "cv_folds",
        "n_permutations",
        "seed",
    }, f"o conjunto de campos mudou: {sorted(campos)}"


# ---------------------------------------------------------------------------
# A REGRA 6 DO README na origem — por introspecção, como no caminho binário
# ---------------------------------------------------------------------------


def test_o_modelo_nao_pode_ser_construido_sem_a_medida_e_sem_o_p() -> None:
    """**Os sete campos obrigatórios, e a asserção é sobre a DECLARAÇÃO.**

    *"Não pode existir caminho de código que retorne scores de PLS-DA sem `q2` e
    `permutation_pvalue`"*. Dataclass congelado sem valor padrão não admite construção parcial, e é
    isso que torna a regra mecanismo em vez de cuidado. Um padrão em qualquer dos sete abriria
    a construção que a regra proíbe.
    """
    campos = dataclasses.fields(OneVsRestModel)
    nomes = {c.name for c in campos}
    assert nomes == {
        "group",
        "scores",
        "loadings",
        "vip",
        "r2y",
        "q2",
        "permutation_pvalue",
    }, f"o conjunto de campos mudou: {sorted(nomes)}"

    for campo in campos:
        assert campo.default is dataclasses.MISSING, (
            f"{campo.name} ganhou valor padrão; com padrão em `q2` ou `permutation_pvalue` passa a "
            "existir construção válida de scores sem validação, que é a regra 6 do README"
        )
        assert campo.default_factory is dataclasses.MISSING, f"{campo.name} ganhou factory"

    with pytest.raises(TypeError):
        OneVsRestModel(  # type: ignore[call-arg]
            group="A",
            scores=np.zeros((2, 2)),
            loadings=np.zeros((3, 2)),
            vip=np.zeros(3),
            r2y=0.5,
            q2=0.4,
        )


def test_o_resultado_nao_carrega_score_nenhum_e_e_isso_que_fecha_a_ponta() -> None:
    """**Onde não há score, não há meio par possível.**

    `MulticlassPLSDAResult` não tem `scores` nem `loadings`: quem quiser um score tem de passar por
    `OneVsRestModel`, que traz a validação ao lado. Sem esta asserção, um campo `scores`
    acrescentado ao contentor amanhã abriria exactamente o caminho que a regra 6 do README proíbe, e
    nada o acusaria.
    """
    campos = {c.name for c in dataclasses.fields(MulticlassPLSDAResult)}
    assert campos, "âncora: o dataclass não tem campo nenhum e a asserção abaixo seria vácua"
    assert not (campos & {"scores", "loadings", "vip"}), (
        f"o contentor passou a carregar {sorted(campos & {'scores', 'loadings', 'vip'})} sem o par "
        "ao lado — é a regra 6 do README reaberta na classe nova"
    )


@pytest.mark.parametrize(
    ("modelos", "pedaco"),
    [
        (1, "abaixo de 2"),
        (0, "abaixo de 2"),
    ],
)
def test_um_resultado_com_menos_de_dois_modelos_e_recusado(modelos: int, pedaco: str) -> None:
    """Um modelo só seria lido como «multiclasse» e é o caminho binário com outro nome."""
    um = OneVsRestModel(
        group="A",
        scores=np.zeros((2, 2)),
        loadings=np.zeros((3, 2)),
        vip=np.zeros(3),
        r2y=0.5,
        q2=0.4,
        permutation_pvalue=0.1,
    )
    with pytest.raises(ValueError, match=pedaco):
        MulticlassPLSDAResult(
            models=tuple([um] * modelos),
            groups=tuple(["A"] * modelos),
            n_components=2,
            cv_folds=7,
            n_permutations=20,
            seed=42,
        )


def test_modelos_fora_da_ordem_dos_grupos_sao_recusados() -> None:
    """**A ordem é o que alinha `q2` com `groups`**, e desalinhá-la troca o rótulo de cada número.

    Na forma mecânica: o número continua certo e passa a responder por outra classe.
    """

    def _modelo(nome: str, q2: float) -> OneVsRestModel:
        return OneVsRestModel(
            group=nome,
            scores=np.zeros((2, 2)),
            loadings=np.zeros((3, 2)),
            vip=np.zeros(3),
            r2y=0.5,
            q2=q2,
            permutation_pvalue=0.1,
        )

    comum = {"n_components": 2, "cv_folds": 7, "n_permutations": 20, "seed": 42}
    alinhado = MulticlassPLSDAResult(
        models=(_modelo("A", 0.1), _modelo("B", 0.9)), groups=("A", "B"), **comum
    )
    assert alinhado.q2 == (0.1, 0.9), "âncora: o caminho alinhado tem de construir"

    with pytest.raises(ValueError, match="ordem"):
        MulticlassPLSDAResult(
            models=(_modelo("A", 0.1), _modelo("B", 0.9)), groups=("B", "A"), **comum
        )


# ---------------------------------------------------------------------------
# O par de verdade conhecida — é ele que impede o gráfico de scores de enganar
# ---------------------------------------------------------------------------


def test_par_causal_quatro_grupos_com_sinal_contra_quatro_sem_sinal() -> None:
    """**Sem este par, uma implementação quebrada pareceria conservadora.**

    A literatura regista que o gráfico de scores de PLS-DA separa nuvens em dado **totalmente
    aleatório**: é o falso positivo que a regra 6 do README existe para expor. As duas pontas correm
    sobre o MESMO desenho — 60 amostras (15 por grupo) x 200 features, `n_components=2`,
    `cv_folds=7`, 20 permutações — e diferem só pelo deslocamento de `+1,5 sd`.

    **Medido, e são as duas populações declaradas:**

    | Ponta | `Q²` por modelo | `p` por modelo |
    |---|---|---|
    | com sinal, 202 | 0,7607 / 0,6252 / 0,6804 / 0,7799 | os quatro em `1/21` |
    | sem sinal, 303 | -0,3044 / -0,5127 / -0,0508 / -0,3127 | 0,571 / 0,810 / 0,190 / 0,667 |

    `1/21` é o **mínimo possível** com 20 permutações, pela convenção `(k+1)/(n+1)`: nenhuma
    permutação supera o observado. A asserção é sobre o **sinal** do `Q²` e sobre a ordenação dos
    `p`, e não sobre os decimais — congelar os decimais fixaria a versão do `scikit-learn` em vez
    da propriedade.
    """
    com = plsda_multiclass(_com_sinal(202, QUATRO), n_permutations=20, seed=42)
    sem = plsda_multiclass(_sem_sinal(303, QUATRO), n_permutations=20, seed=42)

    assert len(com.q2) == len(sem.q2) == 4, "âncora: as duas pontas têm de ter quatro modelos"
    assert all(v > 0.5 for v in com.q2), f"com sinal, algum Q² ficou abaixo de 0,5: {com.q2}"
    assert all(v < 0.0 for v in sem.q2), f"sem sinal, algum Q² ficou acima de zero: {sem.q2}"
    assert max(com.permutation_pvalue) < min(sem.permutation_pvalue), (
        f"os p não separam as duas pontas: com sinal {com.permutation_pvalue}, sem sinal "
        f"{sem.permutation_pvalue}"
    )
    assert min(sem.permutation_pvalue) > 1.0 / 21.0, (
        "sem sinal, algum p ficou no mínimo possível — a ponta negativa não estaria a medir nada"
    )


def test_grupos_desbalanceados_o_resultado_nao_e_dominado_pela_classe_maior() -> None:
    """**Nas proporções reais do estudo, e o sinal está na classe MENOR.**

    As contagens são as dos quatro arquivos do laboratório — `Asymptomatic` 18, `NoHTLV` 16,
    `HAMTSP` 14, `ATLL` 13, num total de **61 amostras de estudo** — e o deslocamento de `+2,0 sd`
    em 10 features entra **só** no `ATLL`, que é 13 de 61, ou 21,3% da população.

    **Medido, semente 404, 200 features, 20 permutações:** o modelo do `ATLL` dá `Q² = 0,6952` com
    `p = 1/21`, e os outros três ficam em -0,314, -0,046 e -0,206. A medida de desempenho é por
    modelo, então a classe menor aparece **sem** ser diluída — que é o que um agregado único faria.

    > **A amplitude aqui é 1,0092**, muito maior que os 0,0865 do bloco equilibrado. Ela é
    > **reportada e não classificada**: o desbalanço com sinal numa classe só é exactamente a
    > situação em que o conjunto É média de coisas diferentes, e o número diz isso.
    """
    contagens = {"ATLL": 13, "Asymptomatic": 18, "HAMTSP": 14, "NoHTLV": 16}
    rotulos = [g for g, n in contagens.items() for _ in range(n)]
    gerador = np.random.default_rng([404, 7])
    valores = gerador.normal(size=(len(rotulos), N_FEATURES))
    valores[np.array(rotulos) == "ATLL", 0:10] += 2.0

    multi = plsda_multiclass(_matriz(valores, rotulos), n_permutations=20, seed=42)

    assert multi.groups == ("ATLL", "Asymptomatic", "HAMTSP", "NoHTLV")
    por_grupo = dict(zip(multi.groups, multi.q2, strict=True))
    assert por_grupo["ATLL"] > 0.5, (
        f"a classe com sinal é 13 de 61 e o modelo dela deu Q²={por_grupo['ATLL']:.4f}: o "
        "desempenho está a ser dominado pelas classes maiores"
    )
    assert all(v < 0.0 for g, v in por_grupo.items() if g != "ATLL"), (
        f"uma classe sem sinal deu Q² positivo: {por_grupo}"
    )
    assert multi.q2_spread > 0.5, f"a amplitude não denuncia o desbalanço: {multi.q2_spread}"


# ---------------------------------------------------------------------------
# Determinismo — e o que a semente MOVE e o que ela não move
# ---------------------------------------------------------------------------


def test_a_mesma_semente_reproduz_e_sementes_diferentes_movem_SO_o_p() -> None:  # noqa: N802
    """**Duas metades, e a segunda é a que ninguém escreve.**

    A primeira é a regra 3 do README: mesma entrada e mesma semente, saída idêntica. A segunda é que
    sementes diferentes mudam **o `p` e nada mais** — o `Q²` sai da validação cruzada com
    `shuffle=False` e não consome aleatoriedade nenhuma. Um `Q²` que mudasse com a semente seria
    prova de que a partição está a ser sorteada, que é outra quantidade.

    **A planta cai sobre dado em que as duas coisas DIFEREM, e isso foi MEDIDO antes**: na
    matriz com sinal os quatro `p` ficam todos em `1/21`, o mínimo possível, e sementes diferentes
    dariam o mesmo valor — o par causal seria vácuo. Sobre a matriz **sem sinal**, semente 42 dá
    `(0,571, 0,810, 0,190, 0,667)` e semente 7 dá `(0,571, 0,810, 0,190, 0,619)`: divergem no
    quarto modelo.
    """
    matriz = _sem_sinal(303, QUATRO)
    primeira = plsda_multiclass(matriz, n_permutations=20, seed=42)
    repetida = plsda_multiclass(matriz, n_permutations=20, seed=42)
    outra = plsda_multiclass(matriz, n_permutations=20, seed=7)

    assert primeira.q2 == repetida.q2
    assert primeira.permutation_pvalue == repetida.permutation_pvalue
    for a, b in zip(primeira.models, repetida.models, strict=True):
        assert np.array_equal(a.scores, b.scores) and np.array_equal(a.vip, b.vip)

    assert outra.q2 == primeira.q2, (
        f"o Q² mudou com a semente: {primeira.q2} contra {outra.q2}. A validação cruzada é "
        "`shuffle=False` e não consome aleatoriedade — se mudou, a partição está a ser sorteada"
    )
    assert outra.permutation_pvalue != primeira.permutation_pvalue, (
        f"sementes diferentes deram o MESMO p: {primeira.permutation_pvalue}. Ou o fluxo derivado "
        "não está a ser usado, ou esta matriz não separa os dois desfechos"
    )


def test_cada_modelo_tem_fluxo_PROPRIO_e_nao_a_mesma_sequencia_de_embaralhamentos() -> None:  # noqa: N802
    """**Três níveis no fluxo — `(seed, modelo, permutação)` — e não dois.**

    Com `(seed, permutação)` só, os K modelos consumiriam a **mesma** sequência de embaralhamentos.
    A planta que torna isso visível é **dois grupos sem sinal**, e ela foi escolhida porque nela as
    duas coisas DIFEREM por construção: com dois grupos o alvo de `B` é `1 -` o alvo de `A`,
    e o `Q²` sai **idêntico** nos dois modelos — a reflexão não muda PRESS nem TSS. Então, sob fluxo
    partilhado, os dois modelos veriam a mesma sequência de `Q²` embaralhados contra o mesmo `Q²`
    observado, e os dois `p` seriam **forçosamente iguais**.

    **Medido, semente 909, 30 amostras x 200 features, 20 permutações:** `Q²` igual nos dois
    modelos — `-0,10527946760668616` — e `p` de **0,3333** contra **0,6190**. A igualdade do `Q²` é
    a âncora: sem ela, `p` diferentes provariam apenas que os alvos são diferentes.

    > **A versão anterior deste teste comparava dois `np.random.default_rng` DIRETAMENTE, e não
    > tocava o módulo.** Era uma tautologia sobre o numpy: o arnês de par causal plantou
    > `default_rng([seed, indice])` no lugar de `default_rng([seed, indice_do_modelo, indice])` e
    > os 19 testes ficaram verdes.
    """
    gerador = np.random.default_rng([909, 7])
    matriz = _matriz(gerador.normal(size=(2 * N_POR_GRUPO, N_FEATURES)), ["A"] * 15 + ["B"] * 15)

    multi = plsda_multiclass(matriz, n_permutations=20, seed=42)

    assert multi.q2[0] == multi.q2[1], (
        f"âncora: com dois grupos o Q² tem de ser idêntico nos dois modelos e saiu {multi.q2}. Sem "
        "isso, p diferentes não diriam nada sobre o fluxo"
    )
    assert multi.permutation_pvalue[0] != multi.permutation_pvalue[1], (
        f"os dois modelos deram o MESMO p ({multi.permutation_pvalue}) sobre Q² idêntico: só o "
        "fluxo os distingue, e ele está a ser partilhado"
    )


def test_o_p_e_reproduzido_A_MAO_e_e_isso_que_fixa_O_QUE_se_embaralha() -> None:  # noqa: N802
    """**Reproduzir o oráculo à mão sobre um caso cuja resposta já se sabe**.

    Duas coisas ficam fixadas de uma vez, e nenhuma delas é observável de fora do resultado:

    1. **o fluxo é `(seed, índice do modelo, índice da permutação)`**, com os três níveis;
    2. **o que se embaralha é o vetor de CLASSE**, e alvo *e* estratos saem os dois do vetor
       embaralhado. Embaralhar só o alvo deixaria a estratificação a ver o rótulo **verdadeiro**, e
       a partição de validação continuaria a carregar a informação que a nula diz não existir.

    A recomputação usa o `_q2` do próprio módulo — a fórmula não é o assunto aqui —, e refaz só o
    **protocolo**. O protocolo ERRADO é recomputado ao lado, e a asserção de que os dois divergem
    vem **antes** da comparação: sem ela, este teste passaria sobre um módulo que estratificasse
    pelo rótulo verdadeiro.

    **Medido, semente 909, 30 amostras x 200 features, `n = 30`, modelo 0:** os dois protocolos dão
    um `Q²` diferente em **30 de 30** permutações, com diferença máxima de 0,36, e o número de
    permutações que superam o observado é **11** contra **10** — logo `p` de `12/31` contra `11/31`.

    > **Com `n = 5` os dois protocolos davam o MESMO p**, e a primeira versão deste teste ficou
    > verde sob a planta. Trinta permutações custam ~0,5 s e é o que separa os dois.
    """
    gerador = np.random.default_rng([909, 7])
    rotulos = ["A"] * 15 + ["B"] * 15
    matriz = _matriz(gerador.normal(size=(30, N_FEATURES)), rotulos)
    x = np.asarray(matriz.values, dtype=np.float64)
    vetor = np.array(rotulos)

    q2_observado = plsda_multiclass(matriz, n_permutations=1, seed=42).models[0].q2

    n = 30
    superaram = 0
    superaram_pelo_protocolo_errado = 0
    for indice in range(n):
        embaralhado = np.random.default_rng([42, 0, indice]).permutation(vetor)
        certo = plsda_multiclasse._q2(
            x, plsda_multiclasse._alvo(embaralhado, "A"), embaralhado, n_components=2, cv_folds=7
        )
        superaram += int(certo >= q2_observado)

        # O protocolo ERRADO: embaralha o alvo e estratifica pelo rótulo VERDADEIRO.
        alvo_embaralhado = np.random.default_rng([42, 0, indice]).permutation(
            plsda_multiclasse._alvo(vetor, "A")
        )
        errado = plsda_multiclasse._q2(x, alvo_embaralhado, vetor, n_components=2, cv_folds=7)
        superaram_pelo_protocolo_errado += int(errado >= q2_observado)

    assert superaram != superaram_pelo_protocolo_errado, (
        f"os dois protocolos deram a MESMA contagem ({superaram}): esta matriz não os separa, e a "
        "asserção abaixo passaria sobre qualquer um dos dois"
    )
    assert 0 < superaram < n, (
        f"{superaram} de {n} permutações superaram: com 0 ou com {n} o p seria o extremo e a "
        "asserção abaixo não distinguiria protocolo nenhum"
    )
    esperado = (superaram + 1) / (n + 1)
    obtido = plsda_multiclasse._permutation_pvalue(
        x,
        vetor,
        "A",
        q2_observado,
        indice_do_modelo=0,
        n_components=2,
        cv_folds=7,
        n=n,
        seed=42,
    )
    assert obtido == esperado, (
        f"o p do módulo deu {obtido!r} e a recomputação à mão deu {esperado!r}: ou o fluxo tem "
        "outra derivação, ou o que se embaralha não é o vetor de classe"
    )


def test_o_menor_p_possivel_e_um_sobre_n_mais_um_e_NUNCA_zero() -> None:  # noqa: N802
    """**A convenção `(k+1)/(n+1)`, e o que ela impede é uma afirmação sobre o mundo.**

    Com `k/n` puro, um resultado que nenhuma permutação supera dá `p = 0` — impossibilidade
    afirmada a partir de 20 sorteios. O `+1` conta o observado entre os sorteios, que é o que ele é
    sob a nula, e o menor valor possível passa a ser `1/(n+1)`: uma afirmação sobre a **resolução**
    do teste em vez de sobre o mundo.

    **Medido:** sobre os quatro grupos com sinal (semente 202, `+1,5 sd`, 20 permutações) os quatro
    `p` batem **exactamente** em `1/21`. É o regime em que a diferença entre as duas convenções é
    máxima — e é por isso que a planta cai aqui e não na matriz sem sinal, onde as duas
    convenções dão números próximos e a asserção seria de magnitude.
    """
    n = 20
    multi = plsda_multiclass(_com_sinal(202, QUATRO), n_permutations=n, seed=42)

    assert len(multi.permutation_pvalue) == 4, "âncora: quatro modelos, senão o `all` é vácuo"
    for grupo, p in zip(multi.groups, multi.permutation_pvalue, strict=True):
        assert p == 1.0 / (n + 1), (
            f"`{grupo}` deu p={p!r} e nenhuma permutação supera um Q² destes: o piso tem de ser "
            f"1/(n+1) = {1.0 / (n + 1)!r}. Um p de 0,0 afirmaria impossibilidade a partir de {n} "
            "sorteios"
        )
        assert p > 0.0


# ---------------------------------------------------------------------------
# As guardas de fronteira
# ---------------------------------------------------------------------------


def test_o_qc_fica_fora_pela_MASCARA_e_nao_pelo_nome() -> None:  # noqa: N802
    """**O QC é a régua e não uma classe**, e a exclusão é por `sample_mask`.

    Comparar contra a string `"QC"` casaria por acidente de escrita e falharia num arquivo que a
    escrevesse de outra forma.

    **A planta cai sobre dado em que as duas coisas DIFEREM**: as sete amostras de controlo
    têm `sample_type="qc"` e o rótulo de grupo **`"Pool"`**. Uma exclusão pelo nome deixaria-as
    entrar — são sete, e `cv_folds=7` aceita-as —, e o resultado viria com **quatro** classes. Com
    o rótulo `"QC"` as duas implementações concordariam, e o teste passaria por coincidência.

    **As duas metades**: os grupos biológicos estão lá, e o controlo não. Só a segunda
    passaria sobre uma lista vazia.
    """
    rotulos = ["Pool"] * 7 + ["A"] * 8 + ["B"] * 8 + ["C"] * 8
    tipos = ["qc"] * 7 + ["sample"] * 24
    gerador = np.random.default_rng([505, 7])
    matriz = _matriz(gerador.normal(size=(31, 40)), rotulos, tipos)

    multi = plsda_multiclass(matriz, cv_folds=7, n_permutations=3, seed=42)

    assert multi.groups == ("A", "B", "C"), f"os grupos biológicos não vieram: {multi.groups}"
    assert "Pool" not in multi.groups, (
        "a mistura de controlo foi modelada como classe: a exclusão está a olhar o nome e não a "
        "máscara"
    )


# As contagens abaixo são o desenho do estudo real reduzido — sete QCs, e o `ATLL` como a classe
# que recebe as linhas `validate` —, e a matriz é **SINTÉTICA e declarada como tal**: nenhum dos
# quatro arquivos do laboratório tem amostra `validate` (medido, 0 de 68 nos quatro modos).
N_QC_DO_ESTUDO = 7


def _com_validate(tipo_das_duas: str) -> RefinoMatrix:
    """8 `ATLL` e 8 `NoHTLV` de estudo, DUAS linhas de rótulo `ATLL`, e 7 controles de rótulo `QC`.

    `tipo_das_duas` é a **única** coisa que o par causal move: valores da mesma semente, mesmos
    rótulos, mesma ordem de injeção. Com `"validate"` as duas linhas não são amostras de estudo;
    com `"sample"` são.

    **Sete controles, e o número é requisito e não decoração.** A guarda de `cv_folds` exige ao
    menos 7 membros por classe, então com menos controles o pedido `groups=("QC", ...)` seria
    recusado pelo **tamanho** e a recusa não distinguiria as duas guardas.
    """
    rotulos = ["ATLL"] * 8 + ["NoHTLV"] * 8 + ["ATLL"] * 2 + ["QC"] * N_QC_DO_ESTUDO
    tipos = ["sample"] * 16 + [tipo_das_duas] * 2 + ["qc"] * N_QC_DO_ESTUDO
    gerador = np.random.default_rng([1212, 7])
    return _matriz(gerador.normal(size=(len(rotulos), 40)), rotulos, tipos)


def test_a_amostra_validate_de_rotulo_clinico_NAO_entra_no_modelo() -> None:  # noqa: N802
    """**`sample_mask` não é o complemento de `qc_mask`**, e é a seleção que tem de o saber.

    Há três valores em `SampleType`. Uma amostra `validate` existe para medir a correção **fora**
    da amostra de treino, e o rótulo clínico dela é o do paciente de onde veio — então ele coincide
    com uma classe de estudo por construção, e não por acidente. Deixá-la entrar contamina a
    população de um modelo que existe para separar grupos clínicos com material que não é dela.

    **A planta cai sobre dado em que as duas seleções DIFEREM, e isso é afirmado ANTES**:
    neste fixture a seleção por rótulo dá **18** linhas e a por máscara dá **16**. Com as duas
    iguais o teste passaria por coincidência.

    **O par causal é o TIPO e nada mais:** as mesmas duas linhas, os mesmos valores e o mesmo
    rótulo, tipadas `validate` numa ponta e `sample` na outra. **Medido em 17/09/2026, antes do
    conserto, as duas pontas davam 18 linhas e `Q²` idêntico** — que é `sample_type` a não ser lido
    de todo. Depois: 16 contra 18, e `Q²` de `0,0625` contra `-0,0266`.
    """
    matriz = _com_validate("validate")
    por_linha = np.array(matriz.groups)
    estudo = np.asarray(matriz.sample_mask)
    nomes = tuple(sorted(set(por_linha[estudo].tolist())))
    por_rotulo = int(np.isin(por_linha, nomes).sum())
    por_mascara = int((np.isin(por_linha, nomes) & estudo).sum())
    assert (por_rotulo, por_mascara) == (18, 16), (
        f"a planta não separa: por rótulo entram {por_rotulo} linhas e por máscara {por_mascara}. "
        "Com as duas iguais, a asserção abaixo passaria sobre qualquer das duas seleções"
    )

    multi = plsda_multiclass(matriz, cv_folds=7, n_permutations=3, seed=42)

    assert multi.groups == ("ATLL", "NoHTLV"), f"os grupos de estudo não vieram: {multi.groups}"
    for modelo in multi.models:
        assert modelo.scores.shape[0] == por_mascara, (
            f"o modelo `{modelo.group} vs resto` foi ajustado sobre {modelo.scores.shape[0]} "
            f"linhas e as amostras de estudo são {por_mascara}: as duas `validate` de rótulo "
            "`ATLL` entraram, e a seleção está a olhar o rótulo e não a máscara"
        )

    como_estudo = plsda_multiclass(_com_validate("sample"), cv_folds=7, n_permutations=3, seed=42)
    assert como_estudo.models[0].scores.shape[0] == por_rotulo, (
        "âncora do par causal: com as duas linhas tipadas `sample` elas TÊM de entrar, senão a "
        "divergência abaixo não seria atribuível ao tipo"
    )
    assert multi.q2 != como_estudo.q2, (
        f"tipar as duas linhas como `validate` ou como `sample` deu o MESMO Q² ({multi.q2}): "
        "`sample_type` não está a ser lido de todo"
    )


def test_o_rotulo_dos_QC_pedido_como_CLASSE_e_recusado() -> None:  # noqa: N802
    """**O QC é a régua, e pedi-lo como classe tem de ser recusado — também em `groups=`.**

    O caminho `groups=None` já o deixava fora. O caminho declarado não: a guarda de pertinência
    lia os rótulos de **todas** as linhas, então `groups=("QC", "ATLL")` era aceito e devolvia `Q²`
    e `p` para um contraste entre a mistura de controle e um diagnóstico — número que não significa
    nada, sobre a quantidade que a ferramenta usa como régua de tudo o resto.

    **Medido no `pos_hilic` REAL em 17/09/2026, antes do conserto:** aceito, 7 controles contra 13
    `ATLL`, com `Q² = -0,0120` e `p = 0,0476`. É a mesma coincidência que `api/rotas.py` já
    registou — os QCs são exactamente 7, e `cv_folds=7` aceita-os.

    **A âncora vem primeiro e é ela que impede o teste de passar por recusar tudo:** um par de
    classes de estudo TEM de ser aceito sobre o mesmo fixture. E a contagem de controles é afirmada
    porque com menos de `cv_folds` a recusa viria da guarda de **tamanho**, e este teste não
    distinguiria as duas.
    """
    matriz = _com_validate("validate")

    aceito = plsda_multiclass(
        matriz, groups=("ATLL", "NoHTLV"), cv_folds=7, n_permutations=3, seed=42
    )
    assert aceito.groups == ("ATLL", "NoHTLV"), "âncora: um par de classes de estudo é legítimo"

    n_controles = int(np.asarray(matriz.qc_mask).sum())
    assert n_controles >= 7, (
        f"há {n_controles} controles e `cv_folds=7`: a recusa abaixo viria do TAMANHO e não da "
        "máscara, e o teste passaria pelo motivo errado"
    )

    with pytest.raises(ValueError) as erro:
        plsda_multiclass(matriz, groups=("QC", "ATLL"), cv_folds=7, n_permutations=3, seed=42)
    texto = str(erro.value)
    assert "'QC'" in texto and "não está entre as amostras de estudo" in texto, texto
    assert "não existe nesta matriz" not in texto, (
        f"a mensagem afirma que o rótulo não existe, e ele existe — é o rótulo dos controles "
        f"desta matriz. Quem a lê vai procurar um erro de escrita que não há: {texto}"
    )


def test_grupo_menor_que_cv_folds_e_recusado_com_AS_CONTAGENS_na_mensagem() -> None:  # noqa: N802
    """**A mensagem nomeia todas as contagens**, e é isso que a torna acionável.

    Quem vê *"o grupo X é pequeno"* sem as outras contagens não sabe se o problema é aquela classe
    ou o desenho do estudo. A planta: três grupos de 8 com `cv_folds=7` constrói, e o mesmo dado com
    um grupo de 5 reprova — o par causal é sobre o mesmo desenho menos uma condição.
    """
    gerador = np.random.default_rng([606, 7])
    grande = ["A"] * 8 + ["B"] * 8 + ["C"] * 8
    ok = _matriz(gerador.normal(size=(24, 40)), grande)
    assert plsda_multiclass(ok, cv_folds=7, n_permutations=3).groups == ("A", "B", "C")

    pequeno = ["A"] * 8 + ["B"] * 8 + ["C"] * 5
    ruim = _matriz(gerador.normal(size=(21, 40)), pequeno)
    with pytest.raises(ValueError) as erro:
        plsda_multiclass(ruim, cv_folds=7, n_permutations=3)
    texto = str(erro.value)
    assert "'C'" in texto and "cv_folds=7" in texto
    for esperado in ("'A': 8", "'B': 8", "'C': 5"):
        assert esperado in texto, f"a mensagem não traz {esperado}: {texto}"


def test_menos_de_dois_grupos_manda_para_o_caminho_binario() -> None:
    """Um grupo só não é contraste, e a mensagem diz **onde** está a resposta."""
    gerador = np.random.default_rng([707, 7])
    matriz = _matriz(gerador.normal(size=(16, 40)), ["A"] * 16)

    with pytest.raises(ValueError, match="ao menos 2"):
        plsda_multiclass(matriz, cv_folds=7, n_permutations=3)


@pytest.mark.parametrize(
    ("grupos", "pedaco"),
    [
        (("A", "A", "B"), "repetido"),
        (("A", "Inexistente"), "não está entre as amostras de estudo"),
    ],
)
def test_conjunto_de_grupos_invalido_e_recusado(grupos: tuple[str, ...], pedaco: str) -> None:
    """Grupo repetido e grupo inexistente. Nenhum dos dois é conjunto de classes.

    O segundo fragmento mudou em 17/09/2026: a guarda passou a ler os rótulos das **amostras de
    estudo**, e a mensagem diz isso em vez de *"não existe nesta matriz"* — que é falso quando o
    nome pedido é o rótulo dos controles.
    """
    gerador = np.random.default_rng([808, 7])
    matriz = _matriz(gerador.normal(size=(24, 40)), ["A"] * 8 + ["B"] * 8 + ["C"] * 8)

    with pytest.raises(ValueError, match=pedaco):
        plsda_multiclass(matriz, groups=grupos, cv_folds=7, n_permutations=3)


def test_o_cancelamento_interrompe_o_laco_das_permutacoes() -> None:
    """**O laço que mais custa.** 1000 permutações sobre o `pos_hilic` real custam 77,3 s,
    e são K laços destes: sem ponto de parada, a janela promete cancelamento e não entrega.

    **As duas direções:** com a bandeira sempre falsa a função devolve; com a bandeira a virar
    verdadeira ela levanta. Sem a primeira, uma função que levantasse sempre passaria.

    **A asserção é sobre QUAL ponto de parada disparou, e não sobre ter disparado.** Há dois — um
    por modelo e um por permutação —, e o arnês de par causal mediu que remover o do laço das
    permutações deixa o teste verde se ele só exigir a exceção: com quatro modelos, a bandeira que
    vira na quarta consulta é apanhada pelo ponto de parada do **modelo**, e a mensagem é a única
    coisa que os separa: o mecanismo certo sobre o alvo errado.
    """
    matriz = _matriz(
        np.random.default_rng([909, 7]).normal(size=(30, N_FEATURES)), ["A"] * 15 + ["B"] * 15
    )
    assert plsda_multiclass(matriz, n_permutations=5, seed=42, cancelled=lambda: False)

    chamadas = {"n": 0}

    def vira_verdadeira() -> bool:
        chamadas["n"] += 1
        return chamadas["n"] > 3

    with pytest.raises(ExecutionCancelledError, match="laco das permutacoes"):
        plsda_multiclass(matriz, n_permutations=50, seed=42, cancelled=vira_verdadeira)
    assert chamadas["n"] == 4, (
        f"a bandeira foi consultada {chamadas['n']} vezes: ou o laço não a lê, ou lê depois de já "
        "ter feito o trabalho que ela existe para evitar"
    )


# ---------------------------------------------------------------------------
# A importância indefinida — AUSENTE e contada, nunca zero
# ---------------------------------------------------------------------------


def test_importancia_indefinida_chega_como_AUSENTE_e_contada() -> None:  # noqa: N802
    """**A propagação, e não a fórmula.** O `nan` de `_vip` tem par causal na origem
    (`tests/unit/test_plots_plsda.py`); o que se afirma aqui é que ele **chega** e é contado.

    *"Nenhuma feature importa"* e *"não foi possível calcular"* pedem ações opostas: a primeira é
    uma medida, a segunda é a ausência dela. Um vetor de zeros faria as duas chegarem idênticas a
    quem conta, e a figura desenharia barras de altura zero para importância que ninguém calculou.

    **Par causal sobre o mesmo campo:** um VIP finito conta zero indefinidas, e o mesmo vetor com
    duas entradas `nan` conta duas.
    """

    def _modelo(vip: np.ndarray) -> OneVsRestModel:
        return OneVsRestModel(
            group="A",
            scores=np.zeros((2, 2)),
            loadings=np.zeros((3, 2)),
            vip=vip,
            r2y=0.5,
            q2=0.4,
            permutation_pvalue=0.1,
        )

    finito = _modelo(np.array([1.6, 0.8, 1.1]))
    assert finito.n_vip_undefined == 0

    ausente = _modelo(np.array([1.6, np.nan, np.nan]))
    assert ausente.n_vip_undefined == 2, (
        f"contou {ausente.n_vip_undefined} indefinidas em {ausente.vip!r}"
    )
    assert not np.array_equal(np.nan_to_num(ausente.vip), ausente.vip), (
        "o vetor não tem não finitas: a contagem acima estaria a medir outra coisa"
    )


def test_o_vip_do_caminho_real_e_finito_e_nao_negativo() -> None:
    """A outra direção: sobre dado que ajusta, nenhuma feature fica sem importância.

    Sem ela, a contagem de indefinidas acima poderia estar a descrever o caso **normal**.
    """
    multi = plsda_multiclass(_com_sinal(202, QUATRO), n_permutations=3, seed=42)

    for modelo in multi.models:
        assert modelo.vip.shape == (N_FEATURES,)
        assert np.all(np.isfinite(modelo.vip)), f"`{modelo.group}` tem VIP não finito"
        assert np.all(modelo.vip >= 0.0), "o VIP é uma importância e não pode ser negativo"
        assert modelo.n_vip_undefined == 0
