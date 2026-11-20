"""Random Forest e SVM com validação cruzada estratificada e `p` de permutação.

O resultado nunca traz a medida de desempenho sem o `p` de permutação: `ClassifierResult` é
congelado e sem valor padrão, como `PLSDAResult`. Com muito mais features que amostras existe
sempre um hiperplano que separa os grupos, e a acurácia dentro da amostra não diz nada. O `p` de
permutação é o que mostra se o desempenho fora da amostra supera o de rótulos embaralhados.

Não há métrica única. O resultado traz acurácia balanceada, acurácia simples, F1 macro, o recall
por classe, a matriz de confusão e o nível do acaso. A que a figura destaca é a balanceada, a
média dos recalls por classe, porque os grupos são desbalanceados: um classificador que responda
sempre a classe maior tem acurácia simples acima de `1/K` e balanceada exatamente `1/K`,
qualquer que seja o desbalanço. Por isso `chance_level` é `1/K` por definição, e não medido.

No `p` de permutação o que se embaralha é o vetor de classe, que alimenta ao mesmo tempo o alvo
e a estratificação das dobras; embaralhar só o alvo deixaria a partição ver o rótulo verdadeiro.
A convenção é `(k + 1) / (n + 1)`, como no PLS-DA, e o menor valor possível é `1/(n+1)`.

Cada permutação recebe um gerador derivado de `(seed, identificador do classificador, índice da
permutação)`, e o estimador recebe `(seed, identificador)`. Só com `(seed, índice)`, os dois
classificadores consumiriam a mesma sequência de embaralhamentos.

Os estimadores nascem com `n_jobs=SERIE_DO_CLASSIFICADOR`, um trabalhador só. Sem pedido de mais
de um não há `loky` nem processo relançado, que no executável congelado abriria uma janela por
processo. `tests/unit/test_classificadores.py` mede `joblib.effective_n_jobs` sobre os
estimadores que este módulo constrói.

A população é a de `sample_mask`, lida uma vez: ela decide quais classes existem e quais linhas
entram. QC fica fora porque é a régua técnica e não um grupo clínico, e `validate` fica fora
porque `sample_mask` não é o complemento de `qc_mask` (`SampleType` tem três valores).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC

from refino.domain.analysis import PADRAO_DA_ANALISE, AnalysisSpec
from refino.domain.cancellation import CancelCheck, raise_if_cancelled
from refino.domain.labels import ClassifierKind
from refino.metrics.structure import column_usability, retained_column_digest
from refino.statistics.univariate import MIN_AMOSTRAS_POR_GRUPO

if TYPE_CHECKING:  # pragma: no cover - só para o mypy
    from sklearn.base import BaseEstimator

    from refino.domain.matrix import RefinoMatrix

__all__ = [
    "IDENTIFICADOR_DO_CLASSIFICADOR",
    "MIN_CLASSES",
    "NOME_DO_CLASSIFICADOR",
    "SERIE_DO_CLASSIFICADOR",
    "TIPO_DA_IMPORTANCIA",
    "ClassificacaoImpossivelError",
    "ClassifierResult",
    "classify",
    "estimador_do_classificador",
]

MIN_CLASSES: Final = 2
"""Com menos de duas classes não há contraste.

Com uma classe só, a acurácia é `1` por construção e o `p` de permutação é `1`.
"""

SERIE_DO_CLASSIFICADOR: Final = 1
"""O `n_jobs` dos estimadores deste módulo, sempre este valor.

`1` faz o `joblib` correr em linha, no mesmo processo, sem `loky` e sem relançar
`sys.executable`, que no executável congelado é o `refino.exe`. É constante para o teste de
paralelismo ler o mesmo valor que os construtores usam.
"""

# O número de árvores e o de permutações não são constantes deste módulo: são
# `AnalysisSpec.classifier_trees` e `AnalysisSpec.classifier_permutations`. Aqui ficam as
# constantes que não são escolha de quem analisa.

IDENTIFICADOR_DO_CLASSIFICADOR: Final[dict[ClassifierKind, int]] = {
    "random_forest": 21,
    "svm": 22,
}
"""O identificador de unidade que deriva o gerador aleatório de `(seed, unidade)`.

Os dois classificadores correm na mesma execução com a mesma `PipelineSpec.seed`. Com a semente
crua, os dois consumiriam a mesma sequência de embaralhamentos, e os dois p-valores deixariam de
ser sorteios independentes.

Os números são distintos dos de `clustering` (11 e 12). A derivação é por `SeedSequence`, e não
por soma, porque somar componentes colide.
"""

NOME_DO_CLASSIFICADOR: Final[dict[ClassifierKind, str]] = {
    "random_forest": "Random Forest",
    "svm": "SVM",
}
"""O nome que a figura escreve.

É o mesmo texto que o catálogo de figuras usa como chave. Uma tradução aqui daria dois nomes
para a mesma coisa em duas superfícies do produto.
"""

TIPO_DA_IMPORTANCIA: Final[dict[ClassifierKind, str]] = {
    "random_forest": "redução média de impureza (Gini), normalizada",
    "svm": "magnitude média do peso linear entre os modelos um-contra-um",
}
"""O que a importância é, por classificador.

As duas quantidades não são comparáveis: a da floresta é um resíduo da construção das árvores, e
a do SVM é o peso de um hiperplano. Por isso o tipo viaja ao lado do vetor, em vez de ficar
implícito no nome do campo.
"""


class ClassificacaoImpossivelError(ValueError):
    """Recusa: o que se pediu não cabe na população.

    A mensagem nomeia os dois números em conflito, para quem escolheu o número de dobras saber
    contra que limite bateu. É classe própria para quem chama distinguir um pedido não
    satisfazível, que volta ao usuário como 422, de outro erro de valor, que é defeito de
    programação.
    """


@dataclass(frozen=True)
class ClassifierResult:
    """O desempenho fora da amostra de um classificador, com o `p` de permutação ao lado.

    Todos os campos são obrigatórios e sem valor padrão, como em `PLSDAResult` e
    `OneVsRestModel`: não há como construir a medida de desempenho sem `permutation_pvalue`.

    Não há medida única. As cinco quantidades de desempenho viajam lado a lado e nenhum campo as
    resume, porque as duas acurácias respondem perguntas opostas sob desbalanço.

    `n_permutations`, `cv_folds` e `seed` dizem como o resultado foi obtido. Sem o `n`, um
    `p = 0,00498` passa por medida quando pode ser o mínimo possível, `1/(n+1)`. Sem as dobras,
    duas acurácias balanceadas não são a mesma quantidade. Sem a semente, o resultado não é
    reproduzível.
    """

    kind: ClassifierKind
    groups: tuple[str, ...]
    """As classes, em ordem alfabética. Alinhada com `class_counts`, `recall` e `confusion`."""

    class_counts: tuple[int, ...]
    """Quantas amostras de estudo em cada classe."""

    balanced_accuracy: float
    """A média dos recalls por classe, fora da amostra. É a medida que a figura destaca.

    O nível do acaso dela é `1/K` qualquer que seja o desbalanço: um classificador que responda
    sempre a classe maior fica exatamente no acaso.
    """

    accuracy: float
    """A fração de acertos, fora da amostra. Publicada ao lado da balanceada, nunca sozinha.

    Sob desbalanço ela premia quem responde sempre a classe maior.
    """

    macro_f1: float
    """F1 médio entre classes, sem ponderar por população."""

    recall: tuple[float, ...]
    """O recall de cada classe, na ordem de `groups`. A média deles é `balanced_accuracy`.

    É o que a figura desenha: a barra por classe separa 'acerta tudo' de 'acerta a classe grande
    e erra as pequenas', que a média sozinha não distingue.
    """

    confusion: tuple[tuple[int, ...], ...]
    """`confusion[i][j]`: amostras da classe `i` preditas como `j`, fora da amostra.

    Contagens inteiras, e não frações: a normalização por linha é derivável da contagem.
    """

    permutation_pvalue: float
    """`(k + 1) / (n + 1)` sobre a acurácia balanceada."""

    chance_level: float
    """`1/K`. Derivado e não medido: é o valor da acurácia balanceada quando a predição é
    independente do rótulo. A figura o desenha como régua."""

    importances: tuple[float, ...]
    """Uma importância por feature usada, alinhada com `feature_names`. Ver `importance_kind`."""

    importance_kind: str
    """O que a importância é, de `TIPO_DA_IMPORTANCIA`."""

    feature_names: tuple[str, ...]
    cv_folds: int
    n_permutations: int
    seed: int
    """A semente derivada de `(seed, unidade)` que produziu o resultado, e não a global."""

    n_samples: int
    n_features_used: int
    n_features_total: int
    column_digest: str
    """`sha256` dos nomes retidos, porque contagem igual não é conjunto igual."""

    @property
    def n_classes(self) -> int:
        return len(self.groups)

    @property
    def acima_do_acaso(self) -> bool:
        """Se a acurácia balanceada supera o nível do acaso. Descritivo e sem limiar.

        Não é veredito: quem diz se a separação se sustenta é o `p` de permutação. A figura usa
        isto para saber de que lado da régua desenhar o rótulo.
        """
        return bool(np.isfinite(self.balanced_accuracy)) and (
            self.balanced_accuracy > self.chance_level
        )


def estimador_do_classificador(
    kind: ClassifierKind, analise: AnalysisSpec, *, seed: int
) -> BaseEstimator:
    """O estimador que este módulo constrói, com um trabalhador e a semente derivada.

    É função para o teste de paralelismo construir o mesmo objeto que o cálculo constrói. Recebe
    `AnalysisSpec`, e não um número de árvores solto, porque a faixa é validada em
    `AnalysisSpec.__post_init__`.

    `class_weight="balanced"` nos dois é a contrapartida, no treino, da acurácia balanceada na
    medição. Sem ele, um SVM sobre classes desbalanceadas empurra a fronteira na direção da
    classe maior e o recall das pequenas cai.

    O SVM é linear por interpretabilidade: com muito mais features que amostras qualquer núcleo
    separa, e só o linear tem `coef_`, de onde sai a importância por feature.
    """
    if kind == "random_forest":
        return RandomForestClassifier(
            n_estimators=analise.classifier_trees,
            max_features="sqrt",
            class_weight="balanced",
            random_state=seed,
            n_jobs=SERIE_DO_CLASSIFICADOR,
        )
    # `kind` é `Literal` fechado de dois membros, e `classify` o valida na entrada: um terceiro
    # valor cairia aqui no SVM sem levantar.
    return SVC(
        kernel="linear",
        class_weight="balanced",
        random_state=seed,
    )


def _semente_derivada(seed: int, unidade: int) -> int:
    """Um inteiro de fluxo próprio para `(seed, unidade)`. Ver `IDENTIFICADOR_DO_CLASSIFICADOR`."""
    return int(np.random.default_rng([seed, unidade]).integers(2**31))


@dataclass(frozen=True)
class _Populacao:
    """As linhas de estudo, os rótulos e as colunas utilizáveis."""

    valores: np.ndarray
    rotulos: np.ndarray
    classes: tuple[str, ...]
    nomes: tuple[str, ...]
    usaveis: np.ndarray
    n_features_total: int


def _populacao_ou_recusa(matrix: RefinoMatrix, *, cv_folds: int) -> _Populacao:
    """As amostras de estudo, com as guardas que tornam a validação cruzada possível.

    A máscara é lida uma vez e decide quais classes existem e quais linhas entram. Comparar com
    a string `"QC"` falharia num arquivo que a escrevesse de outra forma, e a exclusão de
    `validate` decorre de `sample_mask` não ser o complemento de `qc_mask`.

    A guarda de `cv_folds` nomeia a contagem de cada classe. A estratificação exige ao menos
    `cv_folds` membros de cada classe; senão alguma dobra fica sem uma delas, e o recall dela
    sai medido sobre menos dobras que o das outras.
    """
    estudo = np.asarray(matrix.sample_mask)
    if not bool(estudo.any()):
        raise ClassificacaoImpossivelError(
            f"nenhuma das {matrix.n_samples} injeções é amostra de estudo: as amostras `qc` são a "
            "régua técnica e as `validate` não são coorte. Sem amostra de estudo não há classe a "
            "predizer."
        )

    por_linha = np.asarray(matrix.groups)[estudo]
    classes = tuple(sorted(set(por_linha.tolist())))
    if len(classes) < MIN_CLASSES:
        raise ClassificacaoImpossivelError(
            f"{len(classes)} classe(s) entre as amostras de estudo: {list(classes)}. Abaixo de "
            f"{MIN_CLASSES} não há contraste — a acurácia seria 1 por construção e o p de "
            "permutação seria 1, dois números bem formados sobre uma pergunta que ninguém fez."
        )

    minimo = max(MIN_AMOSTRAS_POR_GRUPO, cv_folds)
    contagens = {nome: int((por_linha == nome).sum()) for nome in classes}
    pequenas = sorted(nome for nome, n in contagens.items() if n < minimo)
    if pequenas:
        raise ClassificacaoImpossivelError(
            f"classe(s) {pequenas} com menos de {minimo} amostras e cv_folds={cv_folds}: "
            f"contagens {contagens}. Dobras estratificadas exigem ao menos `cv_folds` membros de "
            "cada classe, senão alguma dobra fica sem uma delas e o recall dela passa a ser medido "
            "sobre menos dobras que o das outras."
        )

    valores = np.asarray(matrix.values, dtype=np.float64)[estudo, :]
    usaveis, n_constantes, n_nao_finitas = column_usability(valores)
    if int(usaveis.sum()) == 0:
        raise ClassificacaoImpossivelError(
            f"nenhuma das {matrix.n_features} features é utilizável para classificar: "
            f"{n_constantes} são constantes nas amostras de estudo e {n_nao_finitas} têm valor não "
            "finito. Coluna constante não separa nada, e coluna não finita derruba o ajuste."
        )

    return _Populacao(
        valores=valores[:, usaveis],
        rotulos=por_linha,
        classes=classes,
        nomes=tuple(
            nome
            for nome, mantem in zip(matrix.feature_names, usaveis.tolist(), strict=True)
            if mantem
        ),
        usaveis=usaveis,
        n_features_total=matrix.n_features,
    )


@dataclass(frozen=True)
class _Desempenho:
    """O que uma validação cruzada produz.

    É um objeto, e não uma tupla, para as duas acurácias não trocarem de lugar por posição.
    """

    balanceada: float
    simples: float
    macro_f1: float
    recall: tuple[float, ...]
    confusao: tuple[tuple[int, ...], ...]


def _predicoes_fora_da_dobra(
    x: np.ndarray,
    rotulos: np.ndarray,
    *,
    kind: ClassifierKind,
    analise: AnalysisSpec,
    seed: int,
    cancelled: CancelCheck | None,
) -> np.ndarray:
    """A predição de cada amostra pelo modelo que não a viu. Uma por linha, alinhada.

    `shuffle=False` no `StratifiedKFold`, como nos módulos de PLS-DA: a ordem das linhas já é
    determinística.

    Com o vetor completo de predições fora da dobra há uma matriz de confusão só, sobre todas as
    amostras. Com médias por dobra haveria uma matriz por dobra, e a agregação seria mais uma
    escolha.
    """
    predito = np.empty(len(rotulos), dtype=object)
    for treino, retido in StratifiedKFold(n_splits=analise.classifier_folds).split(x, rotulos):
        raise_if_cancelled(cancelled, f"classificador {kind}: dobra da validação cruzada")
        modelo = estimador_do_classificador(kind, analise, seed=seed)
        modelo.fit(x[treino], rotulos[treino])
        predito[retido] = modelo.predict(x[retido])
    return predito


def _desempenho(
    verdadeiro: np.ndarray, predito: np.ndarray, classes: tuple[str, ...]
) -> _Desempenho:
    """As cinco quantidades, todas da mesma matriz de confusão.

    Saem de uma fonte só, e não de chamadas separadas de biblioteca: `balanced_accuracy_score`,
    `accuracy_score` e `f1_score` têm cada uma a sua política para a classe sem predição
    nenhuma.

    O recall da classe sem suporte é `nan`, e nunca zero: 'o modelo nunca acertou esta classe' e
    'esta classe não tinha amostra' são respostas diferentes. A guarda de `cv_folds` garante o
    suporte, então o ramo é defensivo.
    """
    indice = {nome: i for i, nome in enumerate(classes)}
    matriz = np.zeros((len(classes), len(classes)), dtype=np.int64)
    for real, chute in zip(verdadeiro.tolist(), predito.tolist(), strict=True):
        matriz[indice[str(real)], indice[str(chute)]] += 1

    suporte = matriz.sum(axis=1)
    acertos = np.diag(matriz).astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        recall = np.where(suporte > 0, acertos / np.maximum(suporte, 1), np.nan)
        preditos = matriz.sum(axis=0)
        precisao = np.where(preditos > 0, acertos / np.maximum(preditos, 1), np.nan)
        f1 = np.where(
            (recall + precisao) > 0.0,
            2.0 * recall * precisao / np.maximum(recall + precisao, 1e-300),
            0.0,
        )

    total = float(matriz.sum())
    return _Desempenho(
        balanceada=float(np.nanmean(recall)) if np.isfinite(recall).any() else float("nan"),
        simples=float(acertos.sum() / total) if total > 0.0 else float("nan"),
        macro_f1=float(np.nanmean(f1)) if np.isfinite(f1).any() else float("nan"),
        recall=tuple(float(v) for v in recall.tolist()),
        confusao=tuple(tuple(int(v) for v in linha) for linha in matriz.tolist()),
    )


def _importancias(
    x: np.ndarray,
    rotulos: np.ndarray,
    *,
    kind: ClassifierKind,
    analise: AnalysisSpec,
    seed: int,
) -> tuple[float, ...]:
    """A importância por feature do modelo ajustado sobre todas as amostras de estudo.

    É o modelo cheio, e não a média das dobras: a importância descreve o que o classificador
    usou, e nenhum número daqui entra em medida de desempenho.

    A do SVM é a média de `|coef_|` entre os modelos um-contra-um. Com `K` classes o `SVC`
    linear ajusta `K(K-1)/2` hiperplanos, e a média das magnitudes é uma quantidade sobre o
    conjunto deles. Ver `TIPO_DA_IMPORTANCIA`.
    """
    modelo = estimador_do_classificador(kind, analise, seed=seed)
    modelo.fit(x, rotulos)
    if kind == "random_forest":
        bruto = np.asarray(modelo.feature_importances_, dtype=np.float64)
    else:
        bruto = np.abs(np.asarray(modelo.coef_, dtype=np.float64)).mean(axis=0)
    return tuple(float(v) for v in bruto.tolist())


def _p_de_permutacao(
    x: np.ndarray,
    rotulos: np.ndarray,
    observado: float,
    *,
    kind: ClassifierKind,
    analise: AnalysisSpec,
    seed: int,
    classes: tuple[str, ...],
    cancelled: CancelCheck | None,
) -> float:
    """Fração de permutações cuja acurácia balanceada alcança a observada, por `(k + 1)/(n + 1)`.

    O que se embaralha é o vetor de classe, que alimenta ao mesmo tempo o alvo e a
    estratificação. A nula é 'os rótulos não têm relação com a matriz'.

    O estimador de cada permutação recebe a mesma semente do observado. O que varia entre os
    sorteios é só o rótulo; semente diferente a cada ajuste somaria a variação do estimador à
    nula.

    A comparação é `>=`, e não `>`: um sorteio que empata com o observado conta contra a
    separação. Contá-lo a favor enviesaria o `p` para baixo.
    """
    if not np.isfinite(observado):
        return float("nan")

    alcancaram = 0
    n = analise.classifier_permutations
    for indice in range(n):
        # Ponto de cancelamento a cada permutação. Sem ele o laço correria minutos sem parada.
        raise_if_cancelled(cancelled, f"classificador {kind}: laco das permutacoes")
        gerador = np.random.default_rng([seed, IDENTIFICADOR_DO_CLASSIFICADOR[kind], indice])
        embaralhado = gerador.permutation(rotulos)
        predito = _predicoes_fora_da_dobra(
            x, embaralhado, kind=kind, analise=analise, seed=seed, cancelled=cancelled
        )
        if _desempenho(embaralhado, predito, classes).balanceada >= observado:
            alcancaram += 1
    return (alcancaram + 1) / (n + 1)


def classify(
    matrix: RefinoMatrix,
    analise: AnalysisSpec = PADRAO_DA_ANALISE,
    *,
    kind: ClassifierKind,
    seed: int,
    cancelled: CancelCheck | None = None,
) -> ClassifierResult:
    """Classifica as amostras de estudo pelo grupo clínico, com validação cruzada e `p`.

    Recebe `AnalysisSpec`, e não parâmetros soltos, porque as faixas de `classifier_folds`,
    `classifier_permutations` e `classifier_trees` são validadas em
    `AnalysisSpec.__post_init__`.

    `seed` é obrigatório e sem valor padrão: toda operação estocástica recebe semente declarada.

    A matriz chega processada. Esta função não imputa, não normaliza e não transforma. As
    colunas constantes e não finitas saem, porque nem separam nem ajustam.

    A validação cruzada faz `(classifier_permutations + 1) x classifier_folds` ajustes, cada um
    de uma floresta de `classifier_trees` árvores ou de um SVM linear. Ver
    `AnalysisSpec.classifier_permutations` para o custo.

    Levanta `ClassificacaoImpossivelError` quando a população não sustenta o pedido: sem amostra
    de estudo, menos de duas classes, classe menor que o número de dobras, ou nenhuma feature
    utilizável. Levanta `InvalidLabelError` quando `kind` não pertence ao vocabulário.
    """
    # `estimador_do_classificador` tem um ramo final sem condição: sem esta guarda, um rótulo
    # inválido construiria o SVM e o resultado declararia o nome pedido.
    from refino.domain.labels import validate_label

    # Não há guarda do número de sorteios aqui: `AnalysisSpec.__post_init__` já recusa
    # `classifier_permutations` e `classifier_trees` abaixo de 1.
    validate_label("ClassifierKind", kind)

    populacao = _populacao_ou_recusa(matrix, cv_folds=analise.classifier_folds)
    semente = _semente_derivada(seed, IDENTIFICADOR_DO_CLASSIFICADOR[kind])

    raise_if_cancelled(cancelled, f"classificador {kind}: antes da validação cruzada")
    predito = _predicoes_fora_da_dobra(
        populacao.valores,
        populacao.rotulos,
        kind=kind,
        analise=analise,
        seed=semente,
        cancelled=cancelled,
    )
    desempenho = _desempenho(populacao.rotulos, predito, populacao.classes)

    return ClassifierResult(
        kind=kind,
        groups=populacao.classes,
        class_counts=tuple(int((populacao.rotulos == nome).sum()) for nome in populacao.classes),
        balanced_accuracy=desempenho.balanceada,
        accuracy=desempenho.simples,
        macro_f1=desempenho.macro_f1,
        recall=desempenho.recall,
        confusion=desempenho.confusao,
        permutation_pvalue=_p_de_permutacao(
            populacao.valores,
            populacao.rotulos,
            desempenho.balanceada,
            kind=kind,
            analise=analise,
            seed=semente,
            classes=populacao.classes,
            cancelled=cancelled,
        ),
        chance_level=1.0 / len(populacao.classes),
        importances=_importancias(
            populacao.valores, populacao.rotulos, kind=kind, analise=analise, seed=semente
        ),
        importance_kind=TIPO_DA_IMPORTANCIA[kind],
        feature_names=populacao.nomes,
        cv_folds=analise.classifier_folds,
        n_permutations=analise.classifier_permutations,
        seed=semente,
        n_samples=len(populacao.rotulos),
        n_features_used=len(populacao.nomes),
        n_features_total=populacao.n_features_total,
        column_digest=retained_column_digest(matrix.feature_names, populacao.usaveis),
    )
