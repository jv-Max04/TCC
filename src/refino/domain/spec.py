"""`QualityReport` e `PipelineSpec`: o relatório de qualidade e a especificação de processamento.

A qualidade nunca se reduz a um único número sem pedido explícito: RSD e agrupamento de QC no PCA
discordam e elegem métodos diferentes. Além das métricas, o relatório carrega os rótulos de como
foi medido, o conjunto `computed` (que separa 'não calculei' de 'calculei e é indefinido') e o
espaço de features em que as métricas estruturais foram medidas. Sem o último, o multiverso
compararia relatórios medidos sobre colunas diferentes: o `batch_loess` produz não finitos e o
`none` não.
"""

import hashlib
import math
from collections.abc import Iterable
from dataclasses import dataclass, fields

from refino.domain.labels import (
    Correction,
    Imputation,
    MeasurementPoint,
    PreTransform,
    RankingFilter,
    RsdEstimator,
    RsdVariant,
    SampleNormalization,
    Scaling,
    Transformation,
    validate_label,
)
from refino.domain.lotes import confere_cortes

__all__ = [
    "FAMILIAS_DO_PLSDA",
    "MIN_GRUPOS_DO_PLSDA_MULTICLASSE",
    "OPTIONAL_METRIC_FIELDS",
    "REQUIRED_METRIC_FIELDS",
    "SUPPORTED_OVERRIDES",
    "PipelineSpec",
    "QualityReport",
    "validate_override_fields",
]

# Campos de métrica opcionais: podem não ter sido calculados. A bicondicional
# `nome ∉ computed ⟺ campo is None` é verificada na construção para cada um.
OPTIONAL_METRIC_FIELDS: frozenset[str] = frozenset(
    {
        "qc_rsd_cv_median",
        "qc_rsd_r_cv_median",
        "n_significant_fdr",
        "plsda_q2",
        "plsda_permutation_p",
    }
)

# Campos de métrica obrigatórios: sempre calculados, logo sempre em `computed`.
# Indefinido neles é `nan`, nunca `None`, para que 'calculada e indefinida' se
# distinga de 'não calculada'.
REQUIRED_METRIC_FIELDS: frozenset[str] = frozenset(
    {
        "qc_rsd_median",
        "qc_rsd_mean",
        "frac_features_under_20",
        "frac_features_under_30",
        "qc_dispersion_ratio",
        "pc1_injection_correlation_abs",
    }
)

# As famílias do PLS-DA que o relatório pode carregar. Cada entrada é
# `(campo da medida, campo do p, campos de procedência)`. `_check_plsda_completo` percorre a
# tabela, então uma família nova é uma linha aqui, e não uma cláusula a mais na guarda.
FAMILIAS_DO_PLSDA: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    (
        "plsda_q2",
        "plsda_permutation_p",
        ("plsda_groups", "plsda_cv_folds", "plsda_n_permutations", "plsda_seed"),
    ),
    (
        "plsda_multiclass_q2",
        "plsda_multiclass_permutation_p",
        (
            "plsda_multiclass_groups",
            "plsda_multiclass_cv_folds",
            "plsda_multiclass_n_permutations",
            "plsda_multiclass_seed",
        ),
    ),
)

# Abaixo de duas classes não há multiclasse. É o mesmo número que
# `statistics.plsda_multiclasse.MIN_GRUPOS` guarda do lado de quem calcula.
MIN_GRUPOS_DO_PLSDA_MULTICLASSE = 2

# Acoplamento entre onde se mediu e como se pré-transformou. As outras duas
# combinações são recusadas na construção: `log10` sobre matriz já transformada por
# `log10` é transformação dupla, e `none` sobre matriz de intensidade bruta é
# transformação nenhuma. Os dois erros produzem número sem levantar.
REQUIRED_PRETRANSFORM: dict[MeasurementPoint, PreTransform] = {
    "post_correction": "log10p1",
    "post_pipeline": "none",
}


@dataclass(frozen=True)
class QualityReport:
    """Avaliação multi-critério. Nunca reduzir a um único número sem pedido explícito."""

    # -- onde e como foi medido -------------------------------------------
    #
    # Sem estes rótulos os números abaixo não são comparáveis com nada. Voltam
    # desserializados do `metrics_json`, e por isso têm guarda em runtime.
    measurement_point: MeasurementPoint
    rsd_variant: RsdVariant
    rsd_estimator: RsdEstimator
    structural_pretransform: PreTransform

    # Quais métricas foram efetivamente calculadas. O consumidor pergunta
    # pertinência antes de valor.
    computed: frozenset[str]

    # -- RSD, in-sample ---------------------------------------------------
    qc_rsd_median: float
    qc_rsd_mean: float
    frac_features_under_20: float
    frac_features_under_30: float
    n_features_rsd_undefined: int

    # O denominador das quatro acima: o tamanho do vetor de features com RSD definido, sobre o
    # qual a mediana, a média e as duas frações correram (`metrics/report.py::_fraction_under`).
    #
    # Fica fora de `REQUIRED_METRIC_FIELDS` e de `OPTIONAL_METRIC_FIELDS`: não é uma métrica que
    # a tela mostra, é a população em que quatro delas foram medidas. Não tem padrão, para que
    # todo sítio de construção o declare em vez de herdar um zero.
    n_features_rsd_defined: int

    # -- estrutura --------------------------------------------------------
    qc_dispersion_ratio: float
    pc1_injection_correlation_abs: float

    # -- o espaço de features das métricas estruturais ---------------------
    #
    # O prefixo `structural_` distingue estas contagens de `n_features_retained`, abaixo, que é
    # a retenção do filtro de RSD, noutro ponto do pipeline.
    structural_features_used: int
    structural_features_dropped_zero_variance: int
    structural_features_dropped_non_finite: int
    structural_column_digest: str

    # -- estabilidade das duas métricas estruturais ------------------------
    #
    # As métricas não são funções contínuas da entrada, e o diagnóstico classifica `clean` por
    # `pc1_injection_correlation < 0,30`. O valor sozinho não sustenta a decisão: `nan` ou
    # amplitude grande sob perturbação de `eps x amplitude` significam `unassessable`, e não um
    # lado do limiar.
    #
    # A semente e o número de réplicas viajam porque a perturbação é estocástica: sem eles, duas
    # amplitudes diferentes não seriam comparáveis.
    qc_dispersion_ratio_spread: float
    pc1_injection_correlation_spread: float
    stability_replicates: int
    stability_seed: int

    # -- fora da amostra: `None` quando nenhum corretor rodou ---------------
    qc_rsd_cv_median: float | None = None
    qc_rsd_r_cv_median: float | None = None

    # -- etapas posteriores -------------------------------------------------
    #
    # `n_features_retained` é a contagem que sobra do filtro de RSD (`filter_by_qc_rsd`). Não
    # tem relação com `structural_features_used`, que é a seleção de colunas do PCA.
    n_features_retained: int = 0
    n_significant_fdr: int | None = None
    plsda_q2: float | None = None
    plsda_permutation_p: float | None = None

    # -- procedência do PLS-DA ---------------------------------------------
    #
    # Seis descritores, e nenhum é métrica: são a população e o regime em que o par `q2`/`p` foi
    # medido. Ficam fora de `REQUIRED_METRIC_FIELDS` e de `OPTIONAL_METRIC_FIELDS`. Um `q2` sem o
    # par de grupos não diz que pergunta respondeu: com quatro grupos há seis pares.
    plsda_groups: tuple[str, str] | None = None
    plsda_r2y: float | None = None
    plsda_n_components: int | None = None
    plsda_cv_folds: int | None = None
    plsda_n_permutations: int | None = None
    plsda_seed: int | None = None

    # -- o PLS-DA multiclasse, um-contra-resto ------------------------------
    #
    # São coleções, e não escalares: com um-contra-resto o desempenho são K medidas e K
    # p-valores, alinhados com `plsda_multiclass_groups`. Nada os agrega, porque uma média de K
    # `Q²` apagaria a amplitude entre modelos, e combinar K p-valores exige uma regra que muda a
    # pergunta respondida.
    #
    # Ficam fora de `OPTIONAL_METRIC_FIELDS`, que é o conjunto das métricas escalares que a tela
    # de qualidade exibe lado a lado e que o multiverso compara entre células. Por isso a
    # bicondicional com `computed` é afirmada para eles em `_check_plsda_completo`.
    plsda_multiclass_q2: tuple[float, ...] | None = None
    plsda_multiclass_permutation_p: tuple[float, ...] | None = None
    plsda_multiclass_groups: tuple[str, ...] | None = None
    plsda_multiclass_n_components: int | None = None
    plsda_multiclass_cv_folds: int | None = None
    plsda_multiclass_n_permutations: int | None = None
    plsda_multiclass_seed: int | None = None

    def __post_init__(self) -> None:
        validate_label("MeasurementPoint", self.measurement_point)
        validate_label("RsdVariant", self.rsd_variant)
        validate_label("RsdEstimator", self.rsd_estimator)
        validate_label("PreTransform", self.structural_pretransform)

        exigido = REQUIRED_PRETRANSFORM[self.measurement_point]
        if self.structural_pretransform != exigido:
            raise ValueError(
                f"medição em {self.measurement_point!r} exige "
                f"structural_pretransform={exigido!r} e recebi "
                f"{self.structural_pretransform!r}. Guardar os dois campos em vez "
                "de derivar um do outro é deliberado; esta guarda é o que impede a "
                "redundância de virar divergência."
            )

        self._check_computed_membership()
        self._check_feature_accounting()
        self._check_plsda_completo()

    def _check_plsda_completo(self) -> None:
        """Garante que o PLS-DA nunca é relatado sem Q² e p de permutação juntos.

        `PLSDAResult` e `OneVsRestModel` já garantem isso do lado de quem calcula, porque os
        campos são obrigatórios. Aqui os dois chegam como opcionais, e um chamador poderia
        preencher só um. Para cada família de `FAMILIAS_DO_PLSDA`:

        - a medida e o p de permutação viajam juntos ou nenhum. Um Q² sem o teste que diz se ele
          é ruído é o falso positivo que a ferramenta existe para expor;
        - par presente exige procedência: os grupos, as dobras, as permutações e a semente.

        Levanta `ValueError` porque o relatório é imutável: um estado inválido construído aqui
        seguiria para a tela e para o banco.
        """
        self._check_bicondicional_das_familias_do_plsda()

        for medida, pvalue, procedencia in FAMILIAS_DO_PLSDA:
            tem_medida = getattr(self, medida) is not None
            tem_p = getattr(self, pvalue) is not None
            if tem_medida != tem_p:
                raise ValueError(
                    f"PLS-DA com meio par: {medida}={getattr(self, medida)!r}, "
                    f"{pvalue}={getattr(self, pvalue)!r}. É proibido caminho que "
                    "produza um sem o outro — um Q² sem o p da permutação é o falso positivo que "
                    "a ferramenta existe para expor."
                )
            if not tem_medida:
                continue
            faltando = sorted(nome for nome in procedencia if getattr(self, nome) is None)
            if faltando:
                raise ValueError(
                    f"PLS-DA medido e sem procedência: {faltando} a `None`. Um Q² sem os grupos "
                    "não diz que pergunta respondeu — os dados reais têm quatro grupos —, e sem a "
                    "semente e as permutações o p não é reproduzível."
                )

        self._check_grupos_do_plsda_binario()
        self._check_colecoes_do_plsda_multiclasse()

    def _check_bicondicional_das_familias_do_plsda(self) -> None:
        """Para cada família do PLS-DA, o campo está em `computed` se e só se tem valor.

        `_check_computed_membership` só percorre `OPTIONAL_METRIC_FIELDS`, e as coleções do
        multiclasse ficam fora dele. Sem esta guarda, um campo multiclasse em `computed` e a
        `None` construiria, e a tela, que pergunta `computed` primeiro, escreveria que a métrica
        foi calculada e não tem valor.

        Corre antes da guarda do meio par, para que as duas famílias recebam as mensagens na
        mesma ordem. Os campos do par binário, que também estão em `OPTIONAL_METRIC_FIELDS`, são
        conferidos nos dois lugares de propósito: a guarda não depende daquele conjunto.
        """
        for medida, pvalue, _procedencia in FAMILIAS_DO_PLSDA:
            for nome in (medida, pvalue):
                valor = getattr(self, nome)
                membro = nome in self.computed
                if membro and valor is None:
                    raise ValueError(
                        f"{nome} está em `computed` e vale `None`. Calculada e indefinida se "
                        "codifica com `nan` — ou, numa coleção, com uma coleção de `nan` —, e "
                        "`None` é reservado para não calculada. A tela pergunta `computed` "
                        "primeiro e escreveria «a métrica foi calculada e não há valor» sobre "
                        "uma medição que nunca aconteceu."
                    )
                if not membro and valor is not None:
                    raise ValueError(
                        f"{nome} tem valor e não está em `computed`. Valor sem pertinência é "
                        "número que ninguém pode saber se vale — e as coleções do "
                        "multiclasse ficam fora de `OPTIONAL_METRIC_FIELDS`, então a "
                        "bicondicional é afirmada aqui."
                    )

    def _check_grupos_do_plsda_binario(self) -> None:
        """O par de grupos do caminho binário: dois nomes distintos e não vazios."""
        grupos = self.plsda_groups
        if grupos is not None and (len(grupos) != 2 or grupos[0] == grupos[1] or not all(grupos)):
            raise ValueError(
                f"`plsda_groups` tem de ser um par de nomes distintos e não vazios: {grupos!r}. "
                "Um par degenerado não é contraste, e o número medido sobre ele não tem sentido."
            )

    def _check_colecoes_do_plsda_multiclasse(self) -> None:
        """As três coleções do um-contra-resto descrevem os mesmos K modelos.

        Sem isto, `plsda_multiclass_q2[2]` poderia responder por outra classe que não
        `plsda_multiclass_groups[2]`. A bicondicional com `computed` é conferida antes, em
        `_check_bicondicional_das_familias_do_plsda`; aqui só se confere o alinhamento, que exige
        as coleções presentes.
        """
        medidas = self.plsda_multiclass_q2
        if medidas is None:
            return

        pvalues = self.plsda_multiclass_permutation_p
        grupos = self.plsda_multiclass_groups
        tamanhos = {
            "plsda_multiclass_q2": len(medidas),
            "plsda_multiclass_permutation_p": len(pvalues) if pvalues is not None else -1,
            "plsda_multiclass_groups": len(grupos) if grupos is not None else -1,
        }
        if len(set(tamanhos.values())) != 1:
            raise ValueError(
                f"o PLS-DA multiclasse desalinhou: {tamanhos}. As três coleções descrevem os "
                "mesmos K modelos, e um desalinhamento troca a classe de cada Q²."
            )
        if grupos is None or len(grupos) < MIN_GRUPOS_DO_PLSDA_MULTICLASSE:
            raise ValueError(
                f"`plsda_multiclass_groups` tem {grupos!r}: abaixo de "
                f"{MIN_GRUPOS_DO_PLSDA_MULTICLASSE} classes não há multiclasse, e um resultado de "
                "um modelo só seria lido como se houvesse."
            )
        if len(set(grupos)) != len(grupos) or not all(grupos):
            raise ValueError(
                f"`plsda_multiclass_groups` tem nome repetido ou vazio: {grupos!r}. Cada classe "
                "entra uma vez, e um nome vazio não identifica pergunta nenhuma."
            )

    def _check_computed_membership(self) -> None:
        """A bicondicional entre `computed` e o valor, nas duas direções, e a obrigatoriedade."""
        nomes = {f.name for f in fields(self)}
        desconhecidos = self.computed - nomes
        if desconhecidos:
            raise ValueError(
                f"`computed` nomeia campos que não existem: {sorted(desconhecidos)}. "
                "Nome de métrica só entra em `computed` se houver campo para ele — "
                "senão a pergunta de pertinência responde sobre nada."
            )

        faltando = REQUIRED_METRIC_FIELDS - self.computed
        if faltando:
            raise ValueError(
                f"campos obrigatórios ausentes de `computed`: {sorted(faltando)}. "
                "Métrica obrigatória é sempre calculada; indefinido nela é `nan`."
            )

        for nome in sorted(OPTIONAL_METRIC_FIELDS):
            valor = getattr(self, nome)
            membro = nome in self.computed
            if membro and valor is None:
                raise ValueError(
                    f"{nome} está em `computed` e vale `None`. "
                    "Calculada e indefinida se codifica com `nan`; `None` é reservado "
                    "para não calculada, e conflacionar os dois é o defeito que esta guarda fecha."
                )
            if not membro and valor is not None:
                raise ValueError(
                    f"{nome} vale {valor!r} e não está em `computed`. "
                    "Valor sem pertinência é número que ninguém pode saber se vale."
                )

        for nome in sorted(REQUIRED_METRIC_FIELDS):
            if getattr(self, nome) is None:
                # Alcançável, e por isso sem `pragma`: o `metrics_json` desserializado não
                # passa pelo mypy, e `None` num campo obrigatório vindo do banco é dado
                # inválido.
                raise ValueError(
                    f"{nome} é obrigatório e não pode ser `None`. "
                    "Métrica obrigatória indefinida se codifica com `nan`; `None` aqui "
                    "vem de fronteira de confiança e é dado inválido."
                )

    def _check_feature_accounting(self) -> None:
        """Confere as contagens, o digest, as réplicas, as amplitudes e as faixas das frações.

        O valor destes campos para a comparação entre relatórios é serem conferíveis.
        """
        for nome in (
            "structural_features_used",
            "structural_features_dropped_zero_variance",
            "structural_features_dropped_non_finite",
            "n_features_rsd_undefined",
            "n_features_rsd_defined",
            "n_features_retained",
        ):
            valor = getattr(self, nome)
            if valor < 0:
                raise ValueError(f"{nome} não pode ser negativo; recebi {valor}")

        if not self.structural_column_digest:
            raise ValueError(
                "structural_column_digest vazio. Ele é o que torna "
                "'as duas medições usaram o mesmo espaço?' pergunta exata em vez de "
                "inferência a partir de contagens iguais."
            )

        if self.stability_replicates < 1:
            raise ValueError(
                f"stability_replicates deve ser >= 1; recebi {self.stability_replicates}. "
                "Zero réplicas produziria amplitude vazia, que a `_spread` reportaria "
                "como `nan` — indistinguível de métrica indefinida."
            )
        for nome in ("qc_dispersion_ratio_spread", "pc1_injection_correlation_spread"):
            valor = getattr(self, nome)
            if not math.isnan(valor) and valor < 0.0:
                raise ValueError(f"{nome} é amplitude e não pode ser negativa; recebi {valor}")

        for nome in ("frac_features_under_20", "frac_features_under_30"):
            valor = getattr(self, nome)
            if not math.isnan(valor) and not 0.0 <= valor <= 1.0:
                raise ValueError(
                    f"{nome} é fração e recebi {valor}. "
                    "Percentual onde se espera fração é erro de unidade que este projeto já "
                    "cometeu uma vez."
                )

        self._check_denominador_do_rsd()

    def _check_denominador_do_rsd(self) -> None:
        """A fração é `nan` se e somente se nenhuma feature tem RSD definido.

        `metrics/report.py::_fraction_under` devolve `nan` exatamente quando o vetor de features
        com RSD definido é vazio, e `n_features_rsd_defined` é o tamanho desse vetor:

            frac_features_under_{20,30} é `nan`  ⟺  n_features_rsd_defined == 0

        Sem a guarda, um relatório poderia afirmar que 62% das features têm RSD abaixo de 20% e
        que zero features têm RSD definido. Vale para as duas frações, que partilham o
        denominador.
        """
        vazio = self.n_features_rsd_defined == 0
        for nome in ("frac_features_under_20", "frac_features_under_30"):
            indefinida = math.isnan(getattr(self, nome))
            if indefinida == vazio:
                continue
            raise ValueError(
                f"{nome}={getattr(self, nome)!r} e n_features_rsd_defined="
                f"{self.n_features_rsd_defined}. A fração é `nan` se e somente se o vetor de "
                "features com RSD definido é vazio — é o que `_fraction_under` faz, e o campo é "
                "o tamanho daquele vetor. Um dos dois está errado, e a tela mostraria os dois."
            )

    # -- consulta ----------------------------------------------------------

    def was_computed(self, name: str) -> bool:
        """Se a métrica foi calculada. É a pergunta a fazer antes de ler o valor."""
        if name not in {f.name for f in fields(self)}:
            raise KeyError(f"{name!r} não é campo de QualityReport")
        return name in self.computed

    def structurally_assessable(self, *, max_spread: float) -> bool:
        """Espelha `StructuralMetrics.is_assessable` sobre o relatório serializado.

        O diagnóstico lê o relatório, que atravessa o banco, e não a `StructuralMetrics`.
        Reimplementar a pergunta lá criaria duas definições de 'avaliável'. `max_spread` não tem
        padrão, como lá.
        """
        if max_spread < 0:
            raise ValueError(f"max_spread não pode ser negativo; recebi {max_spread}")
        if math.isnan(self.qc_dispersion_ratio) or math.isnan(self.pc1_injection_correlation_abs):
            return False
        return all(
            math.isfinite(a) and a <= max_spread
            for a in (self.qc_dispersion_ratio_spread, self.pc1_injection_correlation_spread)
        )

    def comparable_with(self, other: "QualityReport") -> bool:
        """`True` quando os dois relatórios podem ser comparados sem qualificação.

        Exige os quatro rótulos iguais e o mesmo espaço de features estruturais. O `batch_loess`
        produz não finitos e o `none` não, então dois relatórios do mesmo estudo podem ter medido
        `qc_dispersion_ratio` sobre conjuntos de colunas diferentes.

        Compara o digest, e não as contagens: duas especificações podem descartar 500 features
        cada uma e descartar features diferentes.
        """
        return (
            self.measurement_point == other.measurement_point
            and self.rsd_variant == other.rsd_variant
            and self.rsd_estimator == other.rsd_estimator
            and self.structural_pretransform == other.structural_pretransform
            and self.structural_column_digest == other.structural_column_digest
        )


@dataclass(frozen=True)
class PipelineSpec:
    """Uma especificação completa de processamento.

    Toda escolha é explícita, e só a semente e `batch_cuts` têm padrão: um padrão aqui seria uma
    decisão tomada por omissão e sem registro. O padrão `none` de `ranking_filter`, por exemplo,
    vive na CLI, na rota e nas telas, e não neste dataclass.

    A guarda de combinação fica na construção, e não no executor. `imputation="none"` deixa
    ausência, e propagar ausência por uma normalização produz uma matriz que parece processada e
    não é. Recusada na construção, a especificação inválida não existe, e a grade do multiverso
    não pode contê-la. Os rótulos passam por `validate_label` porque chegam da CLI e do `--log`,
    que é relido como entrada.

    O filtro por posto corre duas vezes, com o mesmo mecanismo (`ranking_filter` e
    `ranking_filter_2`). O MetaboAnalyst tem dois controles, variância e abundância, e
    `RankingFilter` cobre as estatísticas dos dois. Os campos são nomeados pela ordem porque nada
    impede escolher duas estatísticas de dispersão. As frações são as retidas, não as cortadas, e
    `None` devolve a etapa à regra por tamanho.

    `normalization_factors` guarda o mapa resolvido, e não o caminho do arquivo. Um caminho
    entraria em `fingerprint` como caminho: o arquivo poderia mudar sem mudar a chave de cache, e
    reexecutar o registro só reproduziria a saída enquanto o arquivo não mudasse. Quem lê o
    arquivo é a fronteira (CLI e telas), antes de construir a especificação. É uma tupla de pares,
    e não um `dict`, porque o dataclass é congelado e precisa ser hasheável, e porque a ordem fixa
    torna a serialização determinística.
    """

    correction: Correction
    n_batches: int
    rsd_cutoff: float | None
    ranking_filter: RankingFilter
    ranking_keep_fraction: float | None
    ranking_filter_2: RankingFilter
    ranking_keep_fraction_2: float | None
    rsd_variant: RsdVariant
    missing_threshold: float | None
    imputation: Imputation
    sample_normalization: SampleNormalization
    normalization_reference: str | None
    normalization_factors: tuple[tuple[str, float], ...] | None
    transformation: Transformation
    scaling: Scaling
    seed: int = 42
    # Os cortes de lote. Com `None`, os `n_batches` lotes são blocos de tamanho quase igual pela
    # ordem de injeção; com cortes, cada um é a ordem de injeção que abre um lote, como na
    # proposta do diagnóstico. O campo tem padrão para não quebrar as especificações já gravadas
    # (os `--log`, a grade do multiverso), que não o trazem.
    batch_cuts: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        """Valida os rótulos, confere as faixas e recusa as combinações proibidas."""
        # Leitura compatível do rótulo antigo, antes da guarda: `pqn` virou `pqn_group` (ver
        # `SampleNormalization` em `labels.py`), e execuções gravadas em `refino.db` e arquivos
        # de `--log` anteriores ainda carregam `pqn`. A tradução é só na leitura: o campo guarda
        # sempre o nome novo.
        # O `object` existe porque o mypy sabe que `pqn` saiu do `Literal` e marcaria a
        # comparação como impossível.
        rotulo_bruto: object = self.sample_normalization
        if rotulo_bruto == "pqn":
            object.__setattr__(self, "sample_normalization", "pqn_group")

        # Chamadas explícitas, e não um laço: um laço uniria os `Literal` numa união e o mypy
        # perderia a estreiteza de cada campo nas comparações abaixo.
        validate_label("Correction", self.correction)
        validate_label("RsdVariant", self.rsd_variant)
        validate_label("RankingFilter", self.ranking_filter)
        validate_label("RankingFilter", self.ranking_filter_2)
        validate_label("Imputation", self.imputation)
        validate_label("SampleNormalization", self.sample_normalization)
        validate_label("Transformation", self.transformation)
        validate_label("Scaling", self.scaling)

        if self.n_batches < 1:
            raise ValueError(f"`n_batches` tem de ser >= 1; recebi {self.n_batches!r}")
        if self.batch_cuts is not None:
            # A forma dos cortes é conferível aqui; se caem dentro da corrida, só a matriz diz
            # (`domain/lotes.py::lotes_declarados`). Tupla de `int` nativo, porque a
            # serialização é comparada por texto.
            object.__setattr__(self, "batch_cuts", confere_cortes(self.batch_cuts, self.n_batches))
        for nome, valor in (
            ("rsd_cutoff", self.rsd_cutoff),
            ("missing_threshold", self.missing_threshold),
            # As duas frações de posto são as retidas: 1,0 mantém tudo, e 0 filtraria tudo e é
            # recusado.
            ("ranking_keep_fraction", self.ranking_keep_fraction),
            ("ranking_keep_fraction_2", self.ranking_keep_fraction_2),
        ):
            if valor is not None and not 0.0 < valor <= 1.0:
                raise ValueError(
                    f"`{nome}` e FRACAO e tem de estar em (0, 1]; recebi {valor!r}. Valor em "
                    "percentual aqui nunca reprova nada e a etapa vira identidade em "
                    "silencio."
                )

        # A guarda da referência é de forma, não de existência: um nome vazio ou só com espaços
        # não é nome. Se o nome existe na matriz só a matriz responde, e isso é conferido onde
        # ela é conhecida.
        if self.normalization_reference is not None and not self.normalization_reference.strip():
            raise ValueError(
                "`normalization_reference` e um nome e recebi vazio ou so espacos. Para dizer "
                "«nao escolhi referencia» o valor e `None`; `''` seria um nome que nenhuma "
                "matriz tem, e a recusa viria da etapa em vez da fronteira."
            )

        # A guarda dos fatores é de forma e de valor, conhecíveis sem a matriz: nome vazio não é
        # nome, e fator não positivo não é fator. O negativo é o que importa: ele passa pela
        # divisão protegida (`_seguro`, em `preprocessing/normalization.py`) e inverte o sinal de
        # uma amostra inteira sem nada acusar. Se todas as linhas da matriz têm fator é conferido
        # onde a matriz é conhecida.
        if self.normalization_factors is not None:
            vistos: set[str] = set()
            for nome, fator in self.normalization_factors:
                if not nome or not nome.strip():
                    raise ValueError(
                        "`normalization_factors` tem entrada com nome vazio. Um mapa cuja chave "
                        "e `''` nao endereca linha nenhuma."
                    )
                if nome in vistos:
                    raise ValueError(
                        f"`normalization_factors` repete a amostra {nome!r}. O leitor da lista "
                        "de amostras sobrescreve duplicados em silencio, e aqui isso faria o "
                        "ultimo valor ganhar sem que ninguem soubesse qual foi usado."
                    )
                vistos.add(nome)
                if not math.isfinite(fator) or fator <= 0.0:
                    raise ValueError(
                        f"o fator da amostra {nome!r} e {fator!r}, e fator tem de ser positivo e "
                        "finito. Um NEGATIVO passa pela divisao protegida e inverte o sinal da "
                        "amostra inteira sem nada a acusar."
                    )

        # O insumo que o método exige é conferido aqui, sem a matriz: `sample_reference` sem nome
        # de referência não é uma especificação. Conferir só na normalização deixaria o erro
        # aparecer depois de o corretor correr. Como `ValueError`, as fronteiras já o traduzem
        # (`422` na rota, código 2 na CLI).
        #
        # A existência do nome na matriz não é conferida aqui: só tem resposta onde a matriz
        # está, e lá a recusa é `ReferenciaAusenteError`.
        if self.sample_normalization in ("sample_reference", "feature_reference"):
            if self.normalization_reference is None:
                raise ValueError(
                    f"`sample_normalization={self.sample_normalization!r}` divide cada amostra "
                    "por uma referencia nomeada, e `normalization_reference` e `None`. Diga qual "
                    "amostra (`sample_reference`) ou qual feature (`feature_reference`) serve de "
                    "denominador, ou escolha outra normalizacao."
                )
        elif self.sample_normalization == "supplied_factors" and self.normalization_factors is None:
            raise ValueError(
                "`sample_normalization='supplied_factors'` divide cada amostra pelo fator dela, "
                "e `normalization_factors` e `None`. Os fatores sao medidos FORA do equipamento "
                "— peso seco, volume, numero de celulas — e nao ha como os derivar da matriz."
            )

        if self.imputation == "none" and self.normalizes():
            raise ValueError(
                "`imputation='none'` nao pode ser combinada com normalizacao "
                f"(sample_normalization={self.sample_normalization!r}, "
                f"transformation={self.transformation!r}, scaling={self.scaling!r}). "
                "`none` deixa ausencia, e propagar ausencia por normalizacao produz matriz "
                "que PARECE processada e nao e: "
                "ausencia de decisao nao e decisao. Combine `imputation='none'` com as tres "
                "normalizacoes em `none`, que e o caminho que deixa a matriz como esta, ou "
                "escolha um metodo de imputacao."
            )

    def normalizes(self) -> bool:
        """Se alguma das três operações de normalização está ativa."""
        return (
            self.sample_normalization != "none"
            or self.transformation != "none"
            or self.scaling != "none"
        )

    def fingerprint(self) -> str:
        """Hash estável dos campos. É a chave de cache do multiverso.

        Calculado sobre os pares nome=valor dos campos, na ordem declarada.
        """
        partes = "|".join(
            f"{campo.name}={getattr(self, campo.name)!r}"
            for campo in fields(self)
            # `batch_cuts` só entra quando existe. Incluir `batch_cuts=None` mudaria a impressão
            # digital de toda especificação já gravada, e o cache do multiverso deixaria de
            # reconhecer o que guardou.
            if not (campo.name == "batch_cuts" and getattr(self, campo.name) is None)
        )
        return hashlib.sha256(partes.encode("utf-8")).hexdigest()[:16]


# Os eixos que o humano pode sobrescrever individualmente. Quem discorda de uma escolha
# específica nomeia o eixo, e é esse o sinal que o registro de concordância guarda.
#
# O conjunto é enumerado para que um campo não passe a ser sobrescrevível por acidente. `seed`
# fica fora: é configuração de determinismo, não escolha metodológica.
#
# Vive em `domain/` porque `api/` e `persistence/` precisam dele e nenhuma importa a outra.
SUPPORTED_OVERRIDES: frozenset[str] = frozenset(
    {
        "correction",
        "n_batches",
        "rsd_cutoff",
        # Escolha metodológica como as outras: decide se uma etapa inteira do pipeline corre e
        # qual estatística ordena as features.
        "ranking_filter",
        # Os três do segundo passe, pelo mesmo argumento: decidem se a etapa corre, qual
        # estatística ordena as features e quantas sobrevivem.
        "ranking_keep_fraction",
        "ranking_filter_2",
        "ranking_keep_fraction_2",
        "rsd_variant",
        "missing_threshold",
        "imputation",
        "sample_normalization",
        # A referência é escolha metodológica: trocar a amostra contra a qual tudo é normalizado
        # muda cada célula da matriz.
        "normalization_reference",
        # `normalization_factors` fica de fora. Sobrescrever é o humano discordar da máquina, e a
        # máquina nunca propõe fatores: eles são medidos fora do equipamento, pelo laboratório.
        # São insumo do estudo, como `measurements` e `sample_list`. A referência fica, porque é
        # escolha metodológica sobre dado que já está na matriz.
        "transformation",
        "scaling",
    }
)


def validate_override_fields(campos: Iterable[str]) -> tuple[str, ...]:
    """Valida o conjunto de eixos sobrescritos e devolve-os ordenados.

    Os nomes chegam do corpo de uma requisição HTTP ou de uma coluna do banco. Campo fora de
    `SUPPORTED_OVERRIDES` levanta em vez de ser ignorado, para não gravar concordância sobre um
    eixo não declarado.

    A ordenação é requisito: o conjunto vai para `campos_sobrescritos` como JSON, e os mesmos
    eixos em ordem diferente dariam textos diferentes para o mesmo fato.
    """
    pedidos = tuple(campos)
    desconhecidos = sorted(set(pedidos) - SUPPORTED_OVERRIDES)
    if desconhecidos:
        raise ValueError(
            f"campo(s) nao sobrescrevivel(is): {desconhecidos}. O conjunto e fechado em "
            f"{sorted(SUPPORTED_OVERRIDES)}. `seed` fica fora de "
            "proposito: trocar a semente e configuracao de determinismo, nao discordancia "
            "metodologica"
        )
    return tuple(sorted(set(pedidos)))
