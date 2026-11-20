"""PLS-DA de N grupos por um-contra-resto.

`Q²` não tem definição única com mais de dois grupos, e a escolha é de quantos modelos rodar.
Aqui são K modelos, um por classe contra o resto. Com dois componentes e dado sintético de
quatro grupos, um PLS2 único sobre a matriz indicadora dá `Q²` bem mais baixo. O par a par dá o
mais alto, mas responde pergunta mais fácil: cada modelo dele separa duas classes e não quatro.
`n_components` fica em 2, o padrão do caminho binário.

Não existe `Q²` agregado nem `p` agregado. Os K valores viajam como coleção: reduzi-los a um
apaga a amplitude entre modelos, e combinar K p-valores exige uma regra (mínimo com Bonferroni,
máximo, Fisher) que responde pergunta diferente conforme a escolha, 'alguma classe separa' ou
'todas separam'.

Nenhum caminho devolve scores sem `q2` e `permutation_pvalue`. Os scores vivem só em
`OneVsRestModel`, que é congelado e declara os dois como campos obrigatórios ao lado deles.
`MulticlassPLSDAResult` não carrega score nenhum, só a coleção de modelos e o regime.

Cada permutação recebe um gerador derivado de `(seed, índice do modelo, índice da permutação)`.
Uma semente global consumida em ordem tornaria a saída dependente da ordem de execução e do
paralelismo.

`StratifiedKFold` recebe os K rótulos de classe, e não o `0/1` que cada modelo constrói. Assim a
partição de validação é a mesma nos K modelos, e a diferença de `Q²` entre eles não mistura o
modelo com o sorteio de dobra.
"""

from dataclasses import dataclass

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import StratifiedKFold

from refino.domain.cancellation import CancelCheck, raise_if_cancelled
from refino.domain.matrix import RefinoMatrix
from refino.statistics.orientation import orient_canonically

# `_vip` é importado, e não copiado, para a fórmula e o ramo que devolve `nan` quando a soma de
# quadrados total é nula existirem num lugar só.
from refino.statistics.plsda import MIN_AMOSTRAS_POR_GRUPO, _vip

__all__ = ["MulticlassPLSDAResult", "OneVsRestModel", "plsda_multiclass"]

# Com menos de dois grupos não há multiclasse: é o caminho binário, ou nada.
MIN_GRUPOS = 2


@dataclass(frozen=True)
class OneVsRestModel:
    """Um modelo, uma classe contra o resto, com a validação ao lado dos scores.

    Os sete campos são obrigatórios e sem valor padrão, como em `PLSDAResult`: não há como
    construir `scores` sem `q2` e sem `permutation_pvalue`.

    `vip` é não negativo onde é definido e não recebe orientação, porque importância não tem
    sinal. Onde a soma de quadrados total do modelo é zero ele sai `nan`, nunca zeros: 'nenhuma
    feature importa' e 'não foi possível calcular' são respostas diferentes.
    """

    group: str
    scores: np.ndarray
    loadings: np.ndarray
    vip: np.ndarray
    r2y: float
    q2: float
    permutation_pvalue: float

    @property
    def n_vip_undefined(self) -> int:
        """Quantas features ficaram sem importância calculável.

        É o que permite a quem desenha dizer quantas features não têm VIP, em vez de desenhar
        barras de altura zero.
        """
        return int((~np.isfinite(self.vip)).sum())


@dataclass(frozen=True)
class MulticlassPLSDAResult:
    """Os K modelos e o regime em que foram medidos. Não carrega score nenhum.

    Não há agregado: `q2` e `permutation_pvalue` saem como tuplas alinhadas com `models`, pelo
    motivo dado no cabeçalho do módulo.

    Os seis campos são obrigatórios. Os quatro últimos dizem como o resultado foi obtido: sem o
    `n` de permutações, `p = 0,00498` passa por medida quando pode ser o mínimo possível, já que
    a convenção é `(k + 1)/(n + 1)` e a resolução do teste é `1/(n + 1)`.
    """

    models: tuple[OneVsRestModel, ...]
    groups: tuple[str, ...]
    n_components: int
    cv_folds: int
    n_permutations: int
    seed: int

    def __post_init__(self) -> None:
        """Confere que há ao menos dois modelos e que eles estão na ordem de `groups`.

        Um resultado com um modelo só seria o caminho binário com outro nome.
        """
        if len(self.models) < MIN_GRUPOS:
            raise ValueError(
                f"um-contra-resto com {len(self.models)} modelo(s): abaixo de {MIN_GRUPOS} não há "
                "multiclasse a modelar, e um resultado de um modelo só seria lido como se houvesse"
            )
        if len(self.models) != len(self.groups):
            raise ValueError(
                f"{len(self.models)} modelo(s) para {len(self.groups)} grupo(s): a coleção e os "
                "rótulos são lidos alinhados, e um desalinhamento troca o nome de cada Q²"
            )
        nomes = tuple(m.group for m in self.models)
        if nomes != self.groups:
            raise ValueError(
                f"os modelos estão na ordem {nomes} e os grupos em {self.groups}. A ordem é o que "
                "alinha `q2` com `groups`, e invertê-la troca o rótulo de cada número"
            )

    @property
    def q2(self) -> tuple[float, ...]:
        """Um `Q²` por modelo, na ordem de `groups`. Tupla e nunca escalar."""
        return tuple(m.q2 for m in self.models)

    @property
    def permutation_pvalue(self) -> tuple[float, ...]:
        """Um p por modelo, na ordem de `groups`. Alinhado com `q2` termo a termo."""
        return tuple(m.permutation_pvalue for m in self.models)

    @property
    def q2_spread(self) -> float:
        """Amplitude do `Q²` entre os modelos: o maior menos o menor, entre os finitos.

        É só reportada: não classifica nada e não tem limiar.
        """
        finitos = [v for v in self.q2 if np.isfinite(v)]
        return float(max(finitos) - min(finitos)) if finitos else float("nan")


def plsda_multiclass(
    matrix: RefinoMatrix,
    groups: tuple[str, ...] | None = None,
    n_components: int = 2,
    cv_folds: int = 7,
    n_permutations: int = 1000,
    seed: int = 42,
    cancelled: CancelCheck | None = None,
) -> MulticlassPLSDAResult:
    """Um modelo por classe contra o resto, cada um com `Q²` fora da amostra e p de permutação.

    `groups=None` usa todos os grupos das amostras de estudo, em ordem alfabética. O QC fica fora
    pela máscara e não pelo nome: ele é uma mistura de todas as amostras, injetada como régua
    técnica, e como classe daria um contraste com número e sem significado. As amostras
    `validate` também ficam fora, porque `sample_mask` não é o complemento de `qc_mask`:
    `SampleType` tem três valores.

    As duas exclusões valem também quando `groups` é declarado: a máscara governa a lista
    oferecida, a guarda de pertinência e a seleção das linhas. Pedir o rótulo dos QCs como classe
    é recusado. Ver `_matriz_e_rotulos`.

    `cv_folds=7` é o padrão do caminho binário. O número não vem dos QCs do estudo, que por
    coincidência também são 7: QC não entra em PLS-DA.

    `n_permutations` é 1000, e não os 200 do caminho binário, por causa do piso do p: o menor
    valor possível com N sorteios é `1/(N+1)`, e `p < 0,01` não se sustenta com 200. O padrão
    concorda com `AnalysisSpec.plsda_multiclass_permutations`.

    Custo: 1000 permutações com 61 amostras de estudo, 6.035 features e quatro grupos levam
    cerca de 77 s.
    """
    x, rotulos, nomes = _matriz_e_rotulos(matrix, groups, cv_folds)

    modelos = []
    for indice, grupo in enumerate(nomes):
        raise_if_cancelled(cancelled, f"plsda_multiclass: modelo {indice + 1} de {len(nomes)}")
        modelos.append(
            _um_contra_resto(
                x,
                rotulos,
                grupo,
                indice=indice,
                n_components=n_components,
                cv_folds=cv_folds,
                n_permutations=n_permutations,
                seed=seed,
                cancelled=cancelled,
            )
        )

    return MulticlassPLSDAResult(
        models=tuple(modelos),
        groups=nomes,
        n_components=n_components,
        cv_folds=cv_folds,
        n_permutations=n_permutations,
        seed=seed,
    )


def _matriz_e_rotulos(
    matrix: RefinoMatrix, groups: tuple[str, ...] | None, cv_folds: int
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    """As linhas dos grupos pedidos, o vetor de classe e os nomes em ordem estável.

    `sample_mask` é lida uma vez e governa as três pontas: a lista de nomes que `groups=None`
    oferece, a guarda de pertinência de `groups=(...)` e a seleção das linhas. O QC fica fora
    porque é a régua e não um grupo, e `validate` fica fora porque não é amostra de estudo.

    Levanta com a contagem de cada grupo quando `cv_folds` excede o menor: a estratificação é
    pelo vetor de classe, e uma classe com menos de `cv_folds` membros deixa alguma dobra sem
    representante dela.
    """
    por_linha = np.array(matrix.groups)
    estudo = np.asarray(matrix.sample_mask)
    if groups is None:
        nomes = tuple(sorted(set(por_linha[estudo].tolist())))
    else:
        nomes = tuple(groups)
        if len(set(nomes)) != len(nomes):
            raise ValueError(f"grupo repetido em {nomes}: cada classe entra uma vez no conjunto")
        # A mensagem não diz que o grupo não existe: o rótulo dos QCs existe na matriz, só não
        # é classe de estudo.
        disponiveis = set(por_linha[estudo].tolist())
        for nome in nomes:
            if nome not in disponiveis:
                raise ValueError(
                    f"grupo {nome!r} não está entre as amostras de estudo desta matriz. "
                    f"Disponíveis: {sorted(disponiveis)}"
                )

    if len(nomes) < MIN_GRUPOS:
        raise ValueError(
            f"{len(nomes)} grupo(s) para modelar: {list(nomes)}. Um-contra-resto exige ao menos "
            f"{MIN_GRUPOS}, e com dois o caminho binário `statistics.plsda` responde a mesma "
            "pergunta com um modelo só"
        )

    # O `& estudo` impede que uma amostra `validate` com o rótulo de uma das classes entre no
    # modelo e na contagem que a guarda de `cv_folds` confere.
    mascara = np.isin(por_linha, nomes) & estudo
    x = np.asarray(matrix.values, dtype=np.float64)[mascara]
    rotulos = por_linha[mascara]

    minimo = max(MIN_AMOSTRAS_POR_GRUPO, cv_folds)
    contagens = {nome: int((rotulos == nome).sum()) for nome in nomes}
    pequenos = sorted(nome for nome, n in contagens.items() if n < minimo)
    if pequenos:
        raise ValueError(
            f"grupo(s) {pequenos} com menos de {minimo} amostras e cv_folds={cv_folds}: "
            f"contagens {contagens}. Dobras estratificadas pelo vetor de classe exigem ao menos "
            "`cv_folds` membros de cada classe, senão alguma dobra fica sem uma delas e o PRESS "
            "mede outra coisa"
        )
    return x, rotulos, nomes


def _um_contra_resto(
    x: np.ndarray,
    rotulos: np.ndarray,
    grupo: str,
    *,
    indice: int,
    n_components: int,
    cv_folds: int,
    n_permutations: int,
    seed: int,
    cancelled: CancelCheck | None,
) -> OneVsRestModel:
    """Um modelo: `grupo` contra todos os outros, com os sete campos preenchidos de uma vez."""
    alvo = _alvo(rotulos, grupo)
    modelo = PLSRegression(n_components=n_components, scale=False).fit(x, alvo)
    escores, cargas = orient_canonically(
        np.asarray(modelo.x_scores_), np.asarray(modelo.x_loadings_)
    )

    predito = np.asarray(modelo.predict(x)).reshape(alvo.shape)
    tss = float(((alvo - alvo.mean(axis=0)) ** 2).sum())
    r2y = 1.0 - float(((alvo - predito) ** 2).sum()) / tss if tss > 0.0 else float("nan")

    q2 = _q2(x, alvo, rotulos, n_components=n_components, cv_folds=cv_folds)
    return OneVsRestModel(
        group=grupo,
        scores=escores,
        loadings=cargas,
        vip=_vip(modelo),
        r2y=r2y,
        q2=q2,
        permutation_pvalue=_permutation_pvalue(
            x,
            rotulos,
            grupo,
            q2,
            indice_do_modelo=indice,
            n_components=n_components,
            cv_folds=cv_folds,
            n=n_permutations,
            seed=seed,
            cancelled=cancelled,
        ),
    )


def _alvo(rotulos: np.ndarray, grupo: str) -> np.ndarray:
    """A coluna `0/1` do um-contra-resto, em `(n, 1)`.

    Bidimensional porque `PLSRegression.predict` devolve `(n, 1)`.
    """
    alvo: np.ndarray = (rotulos == grupo).astype(np.float64).reshape(-1, 1)
    return alvo


def _q2(
    x: np.ndarray,
    alvo: np.ndarray,
    estratos: np.ndarray,
    *,
    n_components: int,
    cv_folds: int,
) -> float:
    """`1 - PRESS/TSS`, com o PRESS somado sobre as predições fora da dobra.

    Generaliza `statistics.plsda._q2`, com `estratos` separado do alvo pelo motivo dado no
    cabeçalho do módulo. Com um alvo de uma coluna e `estratos = alvo`, as duas contas coincidem
    termo a termo.

    `shuffle=False` no `StratifiedKFold`, como no módulo binário: a ordem das linhas já é
    determinística.
    """
    press = 0.0
    for treino, retido in StratifiedKFold(n_splits=cv_folds).split(x, estratos):
        ajuste = PLSRegression(n_components=n_components, scale=False).fit(x[treino], alvo[treino])
        predito = np.asarray(ajuste.predict(x[retido])).reshape(alvo[retido].shape)
        press += float(((alvo[retido] - predito) ** 2).sum())

    tss = float(((alvo - alvo.mean(axis=0)) ** 2).sum())
    return 1.0 - press / tss if tss > 0.0 else float("nan")


def _permutation_pvalue(
    x: np.ndarray,
    rotulos: np.ndarray,
    grupo: str,
    q2_observado: float,
    *,
    indice_do_modelo: int,
    n_components: int,
    cv_folds: int,
    n: int,
    seed: int,
    cancelled: CancelCheck | None = None,
) -> float:
    """Fração de permutações cujo `Q²` iguala ou supera o observado, por `(k + 1) / (n + 1)`.

    O que se embaralha é o vetor de classe, e alvo e estratos saem os dois do vetor embaralhado.
    A nula é 'os rótulos de classe não têm relação com a matriz'; embaralhar só o alvo deixaria
    a estratificação ver o rótulo verdadeiro.

    O `+1` conta o observado entre os sorteios: o menor p possível é `1/(n+1)`, e não zero.

    Cada permutação tem gerador derivado de `(seed, índice do modelo, índice da permutação)`.
    Só com `(seed, índice)`, os K modelos consumiriam a mesma sequência de embaralhamentos e os
    K p-valores deixariam de ser sorteios independentes.
    """
    if not np.isfinite(q2_observado):
        return float("nan")

    superaram = 0
    for indice in range(n):
        # Ponto de cancelamento a cada permutação. Sem ele o multiclasse correria dezenas de
        # segundos sem parada.
        raise_if_cancelled(cancelled, "plsda_multiclass: laco das permutacoes")
        gerador = np.random.default_rng([seed, indice_do_modelo, indice])
        embaralhado = gerador.permutation(rotulos)
        alvo = _alvo(embaralhado, grupo)
        if _q2(x, alvo, embaralhado, n_components=n_components, cv_folds=cv_folds) >= q2_observado:
            superaram += 1
    return (superaram + 1) / (n + 1)
