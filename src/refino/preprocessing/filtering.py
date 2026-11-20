"""Filtros de features: por RSD dos QCs, por posto e por variância quase nula.

Feature cujo RSD nos QCs não é calculável reprova: sobre ela nada se sabe, e
deixá-la passar entregaria às etapas seguintes uma coluna de qualidade
desconhecida. As duas contagens de descarte saem separadas, porque acima do corte
é instabilidade medida e indefinido é ausência de medida.

Nos quatro arquivos de referência nenhuma feature com RSD indefinido passaria
pelo corte de qualquer forma, então reprovar ou não o indefinido dá a mesma
contagem ali. Isso é propriedade desses arquivos, e não vale para dado novo.

`variant` não tem padrão e `estimator` é declarado: RSD sem os dois rótulos não
é comparável.
"""

import warnings
from collections.abc import Callable
from dataclasses import replace

import numpy as np

from refino.domain.labels import RankingFilter, RsdEstimator, RsdVariant, validate_label
from refino.domain.matrix import RefinoMatrix
from refino.metrics.rsd import qc_rsd
from refino.preprocessing.missing import is_missing
from refino.preprocessing.stage import StageResult

__all__ = [
    "ABAIXO_DO_POSTO",
    "ACIMA_DO_CORTE",
    "ESTATISTICA_INDEFINIDA",
    "INDEFINIDO",
    "RANKING_IMPLEMENTATIONS",
    "filter_by_qc_rsd",
    "filter_by_ranking",
    "filter_near_zero_variance",
]

# Os dois motivos de descarte do filtro de RSD. São constantes porque entram no
# relatório e no `--log`.
ACIMA_DO_CORTE = "rsd_acima_do_corte"
INDEFINIDO = "rsd_indefinido"

# Os dois motivos do filtro por posto: abaixo do posto é variabilidade medida e
# baixa, indefinida é ausência de medida.
ABAIXO_DO_POSTO = "abaixo_do_posto"
ESTATISTICA_INDEFINIDA = "estatistica_indefinida"


def filter_by_qc_rsd(
    matrix: RefinoMatrix,
    cutoff: float,
    *,
    variant: RsdVariant,
    estimator: RsdEstimator = "in_sample",
) -> tuple[RefinoMatrix, np.ndarray, StageResult]:
    """Retém features cujo RSD nos QCs está abaixo do corte.

    Devolve `(matriz filtrada, máscara de mantidas, contagens por motivo)`.

    `cutoff` é fração, como a `rsd()` deste projeto devolve: 0,30 significa 30%.
    Corte acima de 1 é recusado, porque limiar em percentual sobre quantidade em
    fração nunca reprova nada.

    RSD indefinido reprova, e os dois motivos saem contados separadamente:
    instabilidade medida sugere corrigir melhor, ausência de medida sugere olhar
    o dado.

    Só o estimador `"in_sample"` é implementado; os outros dois exigem reajuste do
    corretor. Valor não implementado levanta em vez de medir in-sample, o que
    daria número diferente do pedido com o nome errado no `provenance`.
    """
    validate_label("RsdVariant", variant)
    validate_label("RsdEstimator", estimator)
    if estimator != "in_sample":
        raise NotImplementedError(
            f"`estimator={estimator!r}` não é implementado nesta etapa. "
            "`leave_one_qc_out` e `r_cv` exigem reajuste do corretor, que é medição "
            "e não filtro. Aceitar o rótulo e medir in-sample daria número diferente "
            "do pedido com o `provenance` registrando o nome errado."
        )
    if not 0.0 < cutoff <= 1.0:
        raise ValueError(
            f"`cutoff` tem de estar em (0, 1]; recebi {cutoff!r}. O RSD deste projeto é "
            "FRAÇÃO e não percentual - 0,30 significa 30% -, então corte acima de 1 nunca "
            "reprova feature nenhuma e o filtro vira identidade em silêncio."
        )

    valores = qc_rsd(matrix, variant=variant)
    definido = np.isfinite(valores)
    manter = definido & (valores < cutoff)

    resultado = StageResult(
        stage="filter_by_qc_rsd",
        n_features_in=matrix.n_features,
        n_features_out=int(manter.sum()),
        discarded_by_reason={
            ACIMA_DO_CORTE: int((definido & ~manter).sum()),
            INDEFINIDO: int((~definido).sum()),
        },
        labels={"rsd_variant": variant, "rsd_estimator": estimator, "cutoff": f"{cutoff:g}"},
    )
    filtrada = _subset(matrix, manter, f"filter_by_qc_rsd(cutoff={cutoff:g},variant={variant})")
    return filtrada, manter, resultado


def filter_near_zero_variance(
    matrix: RefinoMatrix,
    freq_cut: float = 95 / 5,
    unique_cut: float = 10.0,
) -> tuple[RefinoMatrix, np.ndarray, StageResult]:
    """Réplica da semântica de `caret::nearZeroVar`. Descarta feature quase constante.

    Uma coluna é quase degenerada quando as duas condições valem juntas:

    1. a razão entre a frequência do valor mais comum e a do segundo mais comum
       excede `freq_cut` (o padrão é `95/5`, ou seja, 19);
    2. o percentual de valores distintos sobre o número de amostras fica abaixo de
       `unique_cut` (o padrão é 10).

    A conjunção evita dois falsos positivos: coluna com poucos valores distintos
    mas bem distribuída, e coluna com um valor dominante e muitos outros
    distintos.

    Coluna constante não tem segundo valor mais comum: a razão é infinita por
    convenção, e ela é descartada, como no `caret`.
    """
    valores = np.asarray(matrix.values)
    n = matrix.n_samples
    manter = np.ones(matrix.n_features, dtype=bool)

    for coluna in range(matrix.n_features):
        contagens = np.unique(valores[:, coluna], return_counts=True)[1]
        if contagens.size == 1:
            manter[coluna] = False  # constante: degenerado por construção
            continue
        ordenadas = np.sort(contagens)[::-1]
        razao = float(ordenadas[0]) / float(ordenadas[1])
        distintos_pct = 100.0 * contagens.size / n
        manter[coluna] = not (razao > freq_cut and distintos_pct < unique_cut)

    resultado = StageResult(
        stage="filter_near_zero_variance",
        n_features_in=matrix.n_features,
        n_features_out=int(manter.sum()),
        labels={"freq_cut": f"{freq_cut:g}", "unique_cut": f"{unique_cut:g}"},
    )
    filtrada = _subset(matrix, manter, f"filter_near_zero_variance(freq_cut={freq_cut:g})")
    return filtrada, manter, resultado


def _subset(matrix: RefinoMatrix, manter: np.ndarray, operacao: str) -> RefinoMatrix:
    """Nova matriz com as colunas mantidas. A entrada não é alterada."""
    valores = np.asarray(matrix.values)[:, manter]
    nomes = tuple(nome for nome, fica in zip(matrix.feature_names, manter, strict=True) if fica)
    reduzida = replace(matrix, values=valores, feature_names=nomes)
    return reduzida.with_provenance(operacao)


# ---------------------------------------------------------------------------
# Filtro por posto, a etapa equivalente à do MetaboAnalyst
# ---------------------------------------------------------------------------


def _iqr(valores: np.ndarray) -> np.ndarray:
    """Amplitude interquartil por coluna."""
    return np.asarray(
        np.nanpercentile(valores, 75, axis=0) - np.nanpercentile(valores, 25, axis=0),
        dtype=np.float64,
    )


def _mad(valores: np.ndarray) -> np.ndarray:
    """Desvio absoluto mediano por coluna: `median(|x - median(x)|)`."""
    medianas = np.nanmedian(valores, axis=0, keepdims=True)
    return np.asarray(np.nanmedian(np.abs(valores - medianas), axis=0), dtype=np.float64)


def _sd(valores: np.ndarray) -> np.ndarray:
    return np.asarray(np.nanstd(valores, axis=0, ddof=1), dtype=np.float64)


def _rsd(valores: np.ndarray) -> np.ndarray:
    """`sd / mean`. Média zero dá indefinido, e indefinido reprova."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.asarray(_sd(valores) / np.nanmean(valores, axis=0), dtype=np.float64)


def _nrsd(valores: np.ndarray) -> np.ndarray:
    """`mad / median`, a versão não paramétrica. Mediana zero dá indefinido."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.asarray(_mad(valores) / np.nanmedian(valores, axis=0), dtype=np.float64)


# As implementações por rótulo. `none` mapeia para `None` porque salta a etapa
# inteira. Está no registro porque o teste de paridade compara as chaves com
# `get_args(RankingFilter)`.
RANKING_IMPLEMENTATIONS: dict[str, Callable[[np.ndarray], np.ndarray] | None] = {
    "none": None,
    "iqr": _iqr,
    "sd": _sd,
    "mad": _mad,
    "rsd": _rsd,
    "nrsd": _nrsd,
    "mean": lambda v: np.asarray(np.nanmean(v, axis=0), dtype=np.float64),
    "median": lambda v: np.asarray(np.nanmedian(v, axis=0), dtype=np.float64),
}


def fracao_retida_por_tamanho(n_features: int) -> float:
    """A fração que o MetaboAnalyst retém, em função do número de variáveis.

    Transcrita da fonte (`docs/metodos-metaboanalyst.md`, `[N-4]`): menos de 250
    remove 5%; de 250 a 500 remove 10%; de 500 a 1.000 remove 25%; acima de 1.000
    remove 40%. Exatamente 250 já cai na faixa de 10%; 500 e 1.000 ficam na faixa
    de baixo.
    """
    if n_features < 250:
        return 0.95
    if n_features <= 500:
        return 0.90
    if n_features <= 1000:
        return 0.75
    return 0.60


def filter_by_ranking(
    matrix: RefinoMatrix,
    method: RankingFilter,
    *,
    keep_fraction: float | None = None,
    stage: str = "filter_by_ranking",
    campo_do_contrato: str = "ranking_filter",
) -> tuple[RefinoMatrix, np.ndarray, StageResult]:
    """Reduz features por posto de uma estatística por coluna, como o MetaboAnalyst.

    Cada método ordena as features por uma estatística calculada sobre todas as
    amostras e mantém a fração superior. `none` salta a etapa.

    Não é o `filter_by_qc_rsd`: aquele mede o RSD dos QCs e reprova acima de um
    corte absoluto; este ordena por uma estatística sobre todas as amostras e
    corta por posto, inclusive quando o método é `rsd`.

    Estatística indefinida reprova, como no filtro de RSD, e as duas contagens de
    descarte saem separadas.

    `stage` é parâmetro porque a etapa corre duas vezes no pipeline, e o resumo
    por etapa é um dicionário indexado por esse nome: com o mesmo nome, o segundo
    passe sobrescreveria o primeiro.

    `campo_do_contrato` é o nome do campo que o usuário preencheu, usado no rótulo
    `saltada`. É separado de `stage` porque um é o nome interno da etapa e o outro
    é o do campo da especificação.
    """
    validate_label("RankingFilter", method)
    if keep_fraction is not None and not 0.0 < keep_fraction <= 1.0:
        raise ValueError(
            f"`keep_fraction` é FRAÇÃO e tem de estar em (0, 1]; recebi {keep_fraction!r}. "
            "Valor em percentual aqui retém tudo e a etapa vira identidade em silêncio."
        )

    if method == "none":
        manter = np.ones(matrix.n_features, dtype=bool)
        resultado = StageResult(
            stage=stage,
            n_features_in=matrix.n_features,
            n_features_out=matrix.n_features,
            labels={"saltada": f"{campo_do_contrato}=none"},
        )
        return matrix.with_provenance(f"{stage}(none)"), manter, resultado

    estatistica = RANKING_IMPLEMENTATIONS[method]
    assert estatistica is not None  # `none` já saiu acima; o registro fecha o vocabulário

    # O ausente é `is_missing`, e não só o não finito: zero é ausência de
    # detecção, e são 15,8% das células no `pos_hilic`. Contar os zeros como
    # medida faria `rsd`, `mean` e `median` ordenarem as features por quanta
    # não-detecção elas têm.
    valores = np.asarray(matrix.values, dtype=np.float64)
    presentes = np.where(is_missing(valores), np.nan, valores)
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        # Coluna inteiramente ausente dá fatia toda `nan`, e o aviso vem do módulo
        # `warnings`, que `errstate` não alcança. O desfecho `nan` é o correto:
        # ele reprova a coluna.
        warnings.filterwarnings("ignore", "All-NaN slice encountered", RuntimeWarning)
        warnings.filterwarnings("ignore", "Mean of empty slice", RuntimeWarning)
        warnings.filterwarnings("ignore", "Degrees of freedom <= 0 for slice", RuntimeWarning)
        pontuacao = estatistica(presentes)

    definida = np.isfinite(pontuacao)
    # A fração e a contagem saem da mesma população, as features com estatística
    # definida.
    avaliaveis = int(definida.sum())
    fracao = keep_fraction if keep_fraction is not None else fracao_retida_por_tamanho(avaliaveis)
    # Ao menos uma feature sobrevive: `round` pode dar zero numa matriz pequena
    # com fração baixa, e `process_matrix` levantaria `EmptyMatrixError` por um
    # corte que o usuário não pediu tão agressivo.
    quantas = max(1, round(fracao * avaliaveis))

    manter = np.zeros(matrix.n_features, dtype=bool)
    if definida.any():
        # Ordem decrescente de estatística, com desempate estável pelo índice,
        # para que o resultado seja reprodutível entre execuções.
        candidatas = np.flatnonzero(definida)
        ordem = candidatas[np.argsort(-pontuacao[candidatas], kind="stable")]
        manter[ordem[:quantas]] = True

    resultado = StageResult(
        stage=stage,
        n_features_in=matrix.n_features,
        n_features_out=int(manter.sum()),
        discarded_by_reason={
            ABAIXO_DO_POSTO: int((definida & ~manter).sum()),
            ESTATISTICA_INDEFINIDA: int((~definida).sum()),
        },
        labels={
            "method": method,
            "keep_fraction": f"{fracao:g}",
            # A população em que a fração foi aplicada, para que `keep_fraction`
            # no log diga sobre quantas features agiu.
            "n_avaliaveis": str(avaliaveis),
        },
    )
    filtrada = _subset(matrix, manter, f"{stage}({method},keep={fracao:g})")
    return filtrada, manter, resultado
