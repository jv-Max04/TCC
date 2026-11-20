"""Escrita do formato `toserrf`, da matriz normalizada e do formato do MetaboAnalyst.

O `toserrf` é posicional e não tem cabeçalho nomeado. São quatro linhas de
metadado (`batch`, `sampleType`, `time`, `label`) e depois uma linha por
feature, com as colunas em ordem de injeção. Dos artefatos de serialização do R,
as aspas em todos os campos e o expoente maiúsculo não são reproduzidos; o `NA`
do canto é, porque o leitor do R depende dele.

O formato do MetaboAnalyst é contrato externo. Os requisitos estão transcritos
em `docs/formato-metaboanalyst.md`, e `tests/test_metaboanalyst_conformance.py`
confere a paridade entre aquele documento e as constantes deste módulo. A
paridade é por teste porque ler `docs/` daqui faria a fronteira depender do
layout do repositório.
"""

import csv
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

import numpy as np

from refino.domain.labels import RsdVariant, validate_label
from refino.domain.matrix import RefinoMatrix
from refino.metrics.rsd import qc_rsd

# Dígitos significativos na escrita. Quinze é o que o R usa em `as.character`
# para doubles, e é mais do que os 10 da comparação com os arquivos do R:
# escrever com folga evita que ela falhe por arredondamento nosso.
SIGNIFICANT_DIGITS = 15

METADATA_ROWS = ("batch", "sampleType", "time", "label")


def _format(value: float) -> str:
    """Formata com 15 dígitos significativos, sem notação exponencial gratuita."""
    if value == 0:
        return "0"
    text = f"{value:.{SIGNIFICANT_DIGITS}g}"
    return text


def write_toserrf(matrix: RefinoMatrix, path: Path) -> None:
    """Escreve a matriz no formato exigido pelo motor SERRF.

    As colunas saem em ordem de injeção, que é a ordem das linhas da matriz e
    invariante de construção de `RefinoMatrix`.

    As três primeiras linhas abrem com `NA` literal, e ele é estrutura do formato:
    `serrf.read.data` (`tools.R:242-299`) deriva o layout do arquivo por
    `min(which(!is.na(...)))`, e com o canto vazio o leitor do R não abre o
    arquivo. `read_toserrf` não lê esse canto, então o round-trip é indiferente a
    ele.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\r\n")

        writer.writerow(["NA", "batch", *matrix.batches])
        writer.writerow(["NA", "sampleType", *(s.sample_type for s in matrix.samples)])
        writer.writerow(["NA", "time", *(str(o) for o in matrix.injection_orders)])
        writer.writerow(["No", "label", *matrix.sample_names])

        # Features em linhas, com índice sequencial 1-based na primeira coluna.
        for index, feature in enumerate(matrix.feature_names, start=1):
            column = matrix.values[:, index - 1]
            writer.writerow([str(index), feature, *(_format(v) for v in column)])


def write_normalizado(matrix: RefinoMatrix, path: Path) -> None:
    """Escreve a matriz corrigida na forma que o pipeline R produz.

    `write_toserrf` escreve a forma que alimenta o motor SERRF; esta escreve a que ele devolve,
    `normalized by - X.csv`. Ela não tem linhas de metadados nem coluna de índice: descarta lote,
    tipo de amostra e ordem de injeção, e abre em qualquer planilha.

    O formato segue o arquivo de referência do laboratório: cabeçalho `label,<nomes das amostras>`
    com o rótulo minúsculo, uma linha por feature (`<nome da feature>,<valores>`), terminador CRLF
    e nenhuma aspa. Os valores passam por `_format`, o mesmo do `toserrf`, para que as duas saídas
    não divirjam por formatadores diferentes.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as handle:
        # `QUOTE_MINIMAL`: aspas só onde a célula contém o separador, as aspas ou uma quebra. Com
        # os nomes do laboratório (`RT_mz`) e valores numéricos o arquivo sai sem nenhuma aspa,
        # como o de referência. Um nome com vírgula, como o IUPAC `1,3-Diaminopropane`, que a
        # ingestão aceita citado, sai citado em vez de levantar `csv.Error`.
        writer = csv.writer(handle, lineterminator="\r\n", quoting=csv.QUOTE_MINIMAL)

        writer.writerow(["label", *matrix.sample_names])
        for index, feature in enumerate(matrix.feature_names):
            writer.writerow([feature, *(_format(v) for v in matrix.values[:, index])])


# ---------------------------------------------------------------------------
# Formato do MetaboAnalyst
# ---------------------------------------------------------------------------

# `[F-4]` da citação: "must be unique and consist of a combination of common
# English letters, underscores and numbers". Os nomes de feature do laboratório,
# como `2.25_810.7739m/z`, têm ponto e barra, então sanitizar não é opcional.
CONFORMING_NAME = re.compile(r"^[A-Za-z0-9_]+$")

# O que substitui o caractere não conforme. Sublinhado porque `[F-4]` o admite.
NAME_REPLACEMENT = "_"

# `[F-5]`: com amostras em linhas, "The class label is in the second column".
# O export sai com amostras em linhas, que é a orientação interna.
SAMPLE_COLUMN_HEADER = "Sample"
CLASS_LABEL_HEADER = "Label"

# As três causas de remoção, como rótulo no arquivo das removidas. São os mesmos três baldes
# que `MetaboAnalystExport` separa em campos. A coluna da causa importa porque "nunca foi
# detectada" e "o RSD é alto" pedem ações opostas de quem lê.
#
# Sem acento e em `snake_case` porque o valor é lido por planilha e por `grep`, e um rótulo
# com cedilha atravessa mal a página de código da console do Windows.
CausaDeRemocao = Literal["qcs_todos_zero", "outliers_apagaram_qcs", "rsd_acima_do_corte"]
CAUSAS_DE_REMOCAO: Final[tuple[str, ...]] = (
    "qcs_todos_zero",
    "outliers_apagaram_qcs",
    "rsd_acima_do_corte",
)

# O cabeçalho do arquivo das removidas. `rsd_fracao` e não `rsd` porque o RSD aqui é fração
# (`0,30` é 30%), e uma coluna chamada `rsd` com `1,8342` convidaria a leitura "1,83%".
COLUNAS_DAS_REMOVIDAS: Final[tuple[str, ...]] = ("feature", "causa", "rsd_fracao")


@dataclass(frozen=True)
class FeatureRemovida:
    """Uma feature que saiu no corte, com a causa e, quando existe, o RSD medido.

    `rsd` é `None` nas duas causas de indefinição: elas saem porque o RSD não pode ser
    calculado, e gravar `0` faria uma feature nunca detectada aparecer como a mais estável
    do arquivo.
    """

    nome: str
    causa: CausaDeRemocao
    rsd: float | None


def _sanitize_name(name: str) -> str:
    """Torna um nome conforme ao `[F-4]`, substituindo o não conforme.

    Não é reversível. O relatório das removidas leva o nome original, que é o
    que o laboratório reconhece; o arquivo leva o sanitizado, que é o que o
    MetaboAnalyst aceita.
    """
    return re.sub(r"[^A-Za-z0-9_]", NAME_REPLACEMENT, name)


def _format_metaboanalyst(value: float) -> str:
    """Formata um valor para o `[F-2]`, com ausente e zero saindo vazios.

    `[F-2]` pede "only numeric and positive values (using empty or NA for missing
    values)". Zero não é positivo, e neste domínio zero é ausência de detecção:
    o equipamento não detectou o composto. Então zero sai como a convenção do
    formato para ausente, e a imputação dessas células passa a ser a do
    MetaboAnalyst. `MetaboAnalystExport` devolve a contagem das conversões.

    Ausente (`nan`) sai vazio pelo mesmo caminho.
    """
    if value == 0 or math.isnan(value):
        return ""
    return _format(value)


@dataclass(frozen=True)
class MetaboAnalystExport:
    """O que a exportação escreveu, converteu, renomeou e removeu, por causa.

    As três causas de remoção são campos separados, porque a contagem por motivo
    não se reconstrói por diferença:

    - `removed_all_zero_qcs`: QCs todos zero, feature nunca detectada.
    - `removed_outliers_erased`: a remoção de outliers apagou os QCs não nulos.
    - `removed_high_rsd`: instabilidade medida, acima do corte.

    Ausência de medida e medida ruim levam a ações diferentes: a primeira sugere
    olhar o dado, a segunda sugere corrigir melhor.

    `kept` e as três listas trazem os nomes originais, não os sanitizados.
    """

    kept: tuple[str, ...]
    zeros_written_as_empty: int
    renamed: int
    removed_all_zero_qcs: tuple[str, ...] = ()
    removed_outliers_erased: tuple[str, ...] = ()
    removed_high_rsd: tuple[str, ...] = ()
    # O RSD das removidas por RSD alto, alinhado posicionalmente com `removed_high_rsd`. É
    # array paralelo, e não tupla de pares, para não mudar o tipo que `removed` e
    # `removed_undefined` concatenam; o `__post_init__` confere o comprimento. Vazio significa
    # que não houve corte por RSD.
    rsd_of_removed_high_rsd: tuple[float, ...] = ()

    @property
    def detalhe_das_removidas(self) -> tuple[FeatureRemovida, ...]:
        """As removidas com causa e RSD, na mesma ordem de `removed`.

        É o que o arquivo das removidas escreve. A ordem é critério (as indefinidas
        primeiro, e as de RSD alto da pior para a melhor) e por isso é definida aqui, e não
        no escritor.
        """
        return (
            *(FeatureRemovida(nome, "qcs_todos_zero", None) for nome in self.removed_all_zero_qcs),
            *(
                FeatureRemovida(nome, "outliers_apagaram_qcs", None)
                for nome in self.removed_outliers_erased
            ),
            *(
                FeatureRemovida(nome, "rsd_acima_do_corte", rsd)
                for nome, rsd in zip(
                    self.removed_high_rsd, self.rsd_of_removed_high_rsd, strict=True
                )
            ),
        )

    @property
    def removed(self) -> tuple[str, ...]:
        """Todas as removidas, na ordem do critério.

        As indefinidas vêm primeiro, porque RSD indefinido significa que não se
        sabe nada sobre a estabilidade; depois as de RSD alto, da pior para a
        melhor. É derivada das três listas para não haver uma segunda fonte que
        possa divergir delas.
        """
        return (*self.removed_all_zero_qcs, *self.removed_outliers_erased, *self.removed_high_rsd)

    @property
    def removed_undefined(self) -> tuple[str, ...]:
        """As duas causas de indefinição juntas, quando a distinção não importa.

        Serve ao chamador que só quer saber quantas saíram por falta de medida.
        Não substitui os dois campos.
        """
        return (*self.removed_all_zero_qcs, *self.removed_outliers_erased)

    def __post_init__(self) -> None:
        """Guardas de construção: contagens, unicidade e partição disjunta.

        A unicidade é verificada porque nomes distintos podem sanitizar para o
        mesmo nome. Nos arquivos do laboratório isso não acontece, mas uma matriz
        nova pode colidir.

        A partição tem de ser disjunta: uma feature em duas causas ao mesmo tempo
        faria a soma das três não fechar com o total de removidas.
        """
        if len(self.rsd_of_removed_high_rsd) != len(self.removed_high_rsd):
            raise ValueError(
                f"{len(self.rsd_of_removed_high_rsd)} valores de RSD para "
                f"{len(self.removed_high_rsd)} features removidas por RSD alto. Os dois campos "
                "são posicionalmente alinhados, e comprimentos diferentes fariam "
                "`detalhe_das_removidas` casar cada nome com o RSD de outra feature — número "
                "certo no rótulo errado, na coluna que justifica o corte"
            )
        if self.zeros_written_as_empty < 0 or self.renamed < 0:
            raise ValueError(
                f"contagens negativas: zeros={self.zeros_written_as_empty}, "
                f"renamed={self.renamed}. Contagem negativa é defeito de cálculo, "
                "não estado possível do arquivo escrito."
            )
        if len(set(self.kept)) != len(self.kept):
            raise ValueError(
                "`kept` tem nome repetido, e o `[F-4]` exige nomes únicos. "
                "Repetição aqui significa que duas features do mesmo arquivo "
                "colidiram, e o arquivo escrito não é conforme."
            )
        buckets = (
            ("removed_all_zero_qcs", self.removed_all_zero_qcs),
            ("removed_outliers_erased", self.removed_outliers_erased),
            ("removed_high_rsd", self.removed_high_rsd),
            ("kept", self.kept),
        )
        seen: dict[str, str] = {}
        for label, names in buckets:
            for name in names:
                if name in seen:
                    raise ValueError(
                        f"a feature {name!r} está em {seen[name]!r} e em {label!r} ao "
                        "mesmo tempo. A partição por causa tem de ser disjunta: "
                        "classificação dupla faz a soma das causas não fechar com o "
                        "total, e o relatório reporta mais remoções do que houve."
                    )
                seen[name] = label


def _classify_features(
    matrix: RefinoMatrix, max_features: int, rsd_variant: RsdVariant
) -> tuple[list[int], list[int], list[int], list[int], list[float]]:
    """Reparte os índices de feature em mantidas e nas três causas de remoção.

    Devolve `(mantidas, causa A, causa B, causa C, RSD da causa C)`: as quatro
    primeiras como listas de índices de coluna, a quinta alinhada com a causa C.
    O RSD sai daqui para que o chamador não chame `qc_rsd` outra vez e arrisque
    uma segunda medição com outra variante.

    As duas causas de indefinição saem da diferença entre as variantes:

    - indefinida na variante pedida e em `plain`: os QCs são todos zero, e a
      média zero é a única via para indefinido quando não há infinito nem
      ausente. É a causa A, feature nunca detectada.
    - indefinida na variante pedida e definida em `plain`: a remoção de outliers
      apagou os QCs não nulos e deixou um vetor de zeros. É a causa B, feature
      detectada em um único QC.

    Sob `rsd_variant="plain"` a causa B é vazia, porque as duas máscaras
    coincidem.

    Empate de RSD é resolvido por índice original, com ordenação estável, para
    que o conjunto mantido seja determinístico.
    """
    validate_label("RsdVariant", rsd_variant)

    n_features = len(matrix.feature_names)

    # Sem excesso não há corte, e a lista de removidas é vazia, mesmo havendo
    # indefinidas. O export existe para caber no limite do formato; quem filtra
    # qualidade é `filter_by_qc_rsd`, que é etapa do pipeline e o usuário escolhe.
    if n_features <= max_features:
        return list(range(n_features)), [], [], [], []

    rsd_selected = qc_rsd(matrix, variant=rsd_variant)
    rsd_plain = qc_rsd(matrix, variant="plain")

    undefined_selected = np.isnan(rsd_selected)
    undefined_plain = np.isnan(rsd_plain)

    cause_a = np.flatnonzero(undefined_selected & undefined_plain).tolist()
    cause_b = np.flatnonzero(undefined_selected & ~undefined_plain).tolist()

    # As definidas, da menor RSD para a maior. `stable` mantém a ordem de coluna
    # entre empates.
    defined = np.flatnonzero(~undefined_selected)
    order = defined[np.argsort(rsd_selected[defined], kind="stable")]

    # Toda indefinida sai, e só depois se corta por RSD alto. O excedente que
    # resta define quantas definidas saem. Se as indefinidas sozinhas já excedem
    # o que precisa sair, o arquivo sai com menos de `max_features`, em vez de
    # completar a cota com feature de estabilidade desconhecida.
    surplus = n_features - max_features
    from_defined = surplus - len(cause_a) - len(cause_b)

    if from_defined <= 0:
        return sorted(order.tolist()), cause_a, cause_b, [], []

    keep = order[:-from_defined].tolist()
    # Removidas por RSD alto saem da pior para a melhor. `order` é crescente, então
    # a fatia final está da melhor para a pior e precisa inverter: a primeira linha
    # da lista de removidas tem de ser a mais instável.
    cause_c = order[-from_defined:].tolist()[::-1]
    # O RSD de cada uma, na mesma ordem. A fatia é indexada por `cause_c` em vez de repetir
    # o `[::-1]`, para que as duas listas não possam se desalinhar.
    rsd_cause_c = [float(rsd_selected[index]) for index in cause_c]

    # As mantidas voltam à ordem original da matriz: a ordem do arquivo não
    # carrega informação de qualidade.
    return sorted(keep), cause_a, cause_b, cause_c, rsd_cause_c


def _metodo_na_proveniencia(provenance: Sequence[str], operacao: str) -> str | None:
    """O método da última `operacao(...)` registrada na proveniência, ou `None` se não houver.

    Lê a nota que `preprocessing/normalization.py` grava, como `scale_features(auto,ddof=0)` ou
    `transform(log10)`, e devolve o primeiro argumento. Vale a última porque é a que produziu a
    matriz que chega aqui.
    """
    prefixo = f"{operacao}("
    for nota in reversed(provenance):
        if nota.startswith(prefixo):
            return nota[len(prefixo) :].split(",", 1)[0].rstrip(")")
    return None


def _frase_da_recusa_por_negativo(negativos: int, provenance: Sequence[str]) -> str:
    """A mensagem da recusa por valor negativo, com a causa e a alternativa.

    A causa sai da proveniência, que viaja com a matriz, e não de um argumento: a CLI e a API
    chegam aqui e recebem a mesma mensagem. São três desfechos:

    - escalonamento: todo método de `Scaling` menos `none` centra cada feature na média, então
      produz negativos sempre;
    - `log`: produz negativo onde a intensidade está abaixo de 1, e só aí;
    - nenhum dos dois na proveniência: os negativos vieram de antes.

    Nos dois primeiros a mensagem oferece exportar sem a transformação e o escalonamento, que
    o MetaboAnalyst aplica por conta própria (`docs/metodos-metaboanalyst.md`, `[N-6]` e
    `[N-7]`).
    """
    inicio = (
        f"{negativos} valores negativos na matriz, e o `[F-2]` do formato pede "
        '"only numeric and positive values". '
    )
    escalonamento = _metodo_na_proveniencia(provenance, "scale_features")
    transformacao = _metodo_na_proveniencia(provenance, "transform")
    log = transformacao if transformacao is not None and transformacao.startswith("log") else None
    alternativa = (
        "A alternativa é exportar o mesmo pedido com a transformação e o escalonamento em `none`: "
        "o MetaboAnalyst aplica os dois por conta própria, depois do upload, e o formato que ele "
        "recebe é de intensidade."
    )
    if escalonamento is not None and escalonamento != "none":
        tambem = f" A transformação `{log}` também pode produzi-los." if log else ""
        return (
            f"{inicio}A causa é o escalonamento `{escalonamento}`: ele centra cada feature na "
            f"média, e por isso produz negativos sempre, qualquer que seja o dado.{tambem} "
            f"{alternativa}"
        )
    if log is not None:
        return (
            f"{inicio}A causa provável é a transformação `{log}`: ela devolve negativo em toda "
            f"intensidade abaixo de 1. {alternativa}"
        )
    return (
        f"{inicio}Nem escalonamento nem `log` foram aplicados por esta ferramenta — a "
        "proveniência da matriz não os registra —, então os negativos vieram de antes: do "
        "arquivo de entrada, se ele foi transformado ou escalonado fora daqui, ou da correção. O "
        "formato pede a intensidade antes da transformação e do escalonamento, que o "
        "MetaboAnalyst aplica por conta própria."
    )


def write_metaboanalyst(
    matrix: RefinoMatrix,
    path: Path,
    max_features: int = 5000,
    *,
    rsd_variant: RsdVariant = "outlier_removed",
) -> MetaboAnalystExport:
    """Exporta no formato do MetaboAnalyst, cortando por RSD crescente.

    Quando `n_features > max_features`, mantém as `max_features` de menor RSD nos
    QCs e devolve as removidas separadas por causa, para registro no relatório.

    O arquivo tem amostras em linhas, rótulo de classe na segunda coluna
    (`[F-5]`), vírgula como separador (`[F-1]`), nome sanitizado (`[F-4]`), e zero
    e ausente como célula vazia (`[F-2]`).

    Dentro do corte, feature com RSD indefinido é descartada primeiro: não se sabe
    nada sobre a estabilidade dela, e manter o desconhecido no lugar do medido
    inverteria o critério. Sem excesso não há corte e a lista de removidas é
    vazia, mesmo havendo indefinidas; quem filtra qualidade é `filter_by_qc_rsd`.

    `rsd_variant` entra no critério porque muda o conjunto de indefinidas.

    Levanta em valor negativo (`[F-2]`) e em infinito: com `±Inf` num QC o `sd`
    vira `nan` por `inf - inf`, e a feature fica indefinida em `plain` sem ter os
    QCs todos zero, o que invalida a separação das duas causas de indefinição.
    """
    if max_features < 1:
        raise ValueError(
            f"max_features={max_features} não é exportável. O corte mantém as N de "
            "menor RSD, e N abaixo de 1 não descreve nenhum arquivo."
        )

    values = matrix.values
    negative = int(np.count_nonzero(values < 0))
    if negative:
        raise ValueError(_frase_da_recusa_por_negativo(negative, matrix.provenance))
    infinite = np.count_nonzero(np.isinf(values))
    if infinite:
        raise ValueError(
            f"{infinite} valores infinitos na matriz. A separação das duas causas de "
            "indefinição de RSD supõe ausência de infinito: com `±Inf` o `sd` "
            "vira `nan` por `inf - inf` e a feature fica indefinida em `plain` sem ter "
            "os QCs todos zero, o que faria a causa A absorver um mecanismo que não é "
            "o dela. Recusar é melhor que classificar errado em silêncio. "
            "A causa provável é corrigir com `batch_loess` sob "
            "`null_denominator=r_faithful`: onde a curva ajustada vale zero a divisão "
            "explode, e são 109, 552, 448 e 763 células medidas nos quatro "
            "modos. Use `null_denominator=uncorrected_feature`, que é o padrão de "
            "`refino exportar`: a feature volta sem correção e MARCADA, em vez de o "
            "arquivo inteiro não existir."
        )

    keep, cause_a, cause_b, cause_c, rsd_cause_c = _classify_features(
        matrix, max_features, rsd_variant
    )
    names = matrix.feature_names

    sanitized = [_sanitize_name(names[index]) for index in keep]
    renamed = sum(1 for index in keep if names[index] != _sanitize_name(names[index]))
    kept_values = values[:, keep]
    report = MetaboAnalystExport(
        kept=tuple(names[index] for index in keep),
        zeros_written_as_empty=int(
            np.count_nonzero(kept_values == 0) + np.count_nonzero(np.isnan(kept_values))
        ),
        renamed=renamed,
        removed_all_zero_qcs=tuple(names[index] for index in cause_a),
        removed_outliers_erased=tuple(names[index] for index in cause_b),
        removed_high_rsd=tuple(names[index] for index in cause_c),
        rsd_of_removed_high_rsd=tuple(rsd_cause_c),
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\r\n")
        writer.writerow([SAMPLE_COLUMN_HEADER, CLASS_LABEL_HEADER, *sanitized])
        for row, sample in enumerate(matrix.samples):
            writer.writerow(
                [
                    _sanitize_name(matrix.sample_names[row]),
                    _sanitize_name(sample.group),
                    *(_format_metaboanalyst(v) for v in kept_values[row, :]),
                ]
            )

    return report


def write_removidas(removidas: Sequence[FeatureRemovida], path: Path) -> None:
    """Escreve o arquivo das removidas, ao lado do CSV cortado: uma linha por feature.

    O registro do que saiu viaja com o artefato que sobe para o MetaboAnalyst, e existe
    exatamente quando ele existe. A CLI e a janela chamam esta mesma função, para que os dois
    arquivos saiam idênticos.

    `rsd_fracao` sai vazio nas duas causas de indefinição: célula vazia diz que não há medida,
    e um `0` ordenaria a feature nunca detectada como a mais estável do arquivo.

    Não há linha de comentário nem de resumo: o arquivo é para abrir em planilha, e prosa antes
    do cabeçalho quebra a leitura como CSV.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\r\n")
        writer.writerow(COLUNAS_DAS_REMOVIDAS)
        for removida in removidas:
            writer.writerow(
                [
                    removida.nome,
                    removida.causa,
                    "" if removida.rsd is None else _format(removida.rsd),
                ]
            )
