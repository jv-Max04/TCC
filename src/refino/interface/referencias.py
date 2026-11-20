"""As referências dos métodos e o resumo metodológico do que produziu um arquivo.

Cada obra leva a sua conferência: 'conteúdo conferido' quando o que o texto lhe atribui foi lido,
'só metadados' quando só a referência foi conferida (Welch, Mann-Whitney e Shapiro-Wilk, R52 a
R54). Onde a obra não cobre o que a ferramenta faz, a ressalva vai junto: Wei et al. usam metade do
mínimo, e não um quinto. A lista é a dos métodos implementados, e método sem obra aparece como
'referência por conferir'.

O resumo diz só o que foi registrado: o corretor que a resposta registrou, a semente e os lotes do
pedido, as etapas do pré-processamento na ordem de `CANONICAL_ORDER`, as análises que a resposta
trouxe e as contagens dela. Etapa que o pedido não pediu não é mencionada. O módulo não conhece Qt;
o painel que o mostra é `desktop/janela/execucao.py`.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final, Literal, get_args

from refino.preprocessing.pipeline import CANONICAL_ORDER

__all__ = [
    "CONFERENCIAS",
    "LEGENDA_DA_CONFERENCIA",
    "LINHA_DA_ETAPA",
    "METODOS",
    "METODO_POR_CHAVE",
    "POLITICA_DO_DENOMINADOR_NULO_DA_ROTA",
    "POR_CONFERIR",
    "REFERENCIAS",
    "SEM_RESUMO",
    "SEM_RESUMO_DA_ETAPA",
    "ArquivoExportado",
    "Conferencia",
    "Metodo",
    "Referencia",
    "ResumoMetodologico",
    "chave_do_valor",
    "resumo_da_execucao",
    "resumo_da_exportacao",
    "sem_marcacao",
    "texto_das_referencias",
]

Conferencia = Literal["conteúdo conferido", "só metadados"]
CONFERENCIAS: Final[tuple[str, ...]] = get_args(Conferencia)

# O estado de um método sem obra na tabela: a tela não inventa citação para ele.
POR_CONFERIR: Final = "referência por conferir"

# As duas definições, uma por classe de conferência.
LEGENDA_DA_CONFERENCIA: Final[tuple[str, ...]] = (
    "**conteúdo conferido** — o que o texto atribui à obra foi lido no resumo ou no texto aberto;",
    "**só metadados** — a obra existe e a referência está certa, mas o conteúdo atribuído **não** "
    "foi lido; antes de afirmar algo sobre ela, conferir no texto integral.",
)
# A terceira classe é de método sem obra, e não uma conferência, e por isso não está na tupla acima.
LEGENDA_DO_POR_CONFERIR: Final = (
    f"**{POR_CONFERIR}** — não há obra conferida para o método, e a tela não sugere nenhuma."
)

# A frase de antes da primeira análise, na aba 'Qualidade do resultado', onde o resumo da execução
# fica junto das métricas que a mesma resposta preenche.
SEM_RESUMO: Final = (
    "O resumo aparece depois de a análise terminar, na aba «Análise estatística». Ele traz a "
    "correção da matriz que entrou, a semente, os lotes e as etapas que produziram o resultado, e "
    "as obras de cada método."
)
# A frase de antes do primeiro arquivo, no Resultado de cada etapa da janela: 1 é a Correção e 2 a
# Normalização. Cada uma diz o gesto da sua aba, porque o resumo ali descreve só os arquivos que a
# própria aba baixou.
SEM_RESUMO_DA_ETAPA: Final[dict[int, str]] = {
    1: (
        "O resumo aparece depois de baixar a matriz corrigida ou de «Gerar os três». Ele traz o "
        "corretor, a semente e os lotes que produziram cada arquivo, e as obras de cada método."
    ),
    2: (
        "O resumo aparece depois de baixar a matriz normalizada. Ele traz a correção da matriz que "
        "entrou, a semente, as etapas do pré-processamento que produziram o arquivo, e as obras de "
        "cada método."
    ),
}


@dataclass(frozen=True)
class Referencia:
    """Uma obra: o número, o texto ABNT e a conferência.

    `abnt` guarda a marcação de negrito (`**`); quem a mostra em texto simples passa por
    `sem_marcacao`.
    """

    numero: str
    abnt: str
    conferencia: Conferencia

    def __post_init__(self) -> None:
        """Uma conferência fora das duas classes levanta, em vez de cair num estilo neutro.

        O dado é escrito aqui, e não lido de fronteira, mas a guarda custa uma linha: uma terceira
        palavra, como 'conferido', passaria no `mypy` se alguém a anotasse com `str`, e seria lida
        na tela como uma das duas.
        """
        if self.conferencia not in CONFERENCIAS:
            raise ValueError(
                f"{self.numero}: conferência {self.conferencia!r} não é uma das duas classes, "
                f"{CONFERENCIAS}"
            )


@dataclass(frozen=True)
class Metodo:
    """Um método implementado, com as obras da tabela que o citam; sem obras, 'por conferir'.

    `chave` é `eixo:valor` para o que vem do vocabulário do domínio, e um nome próprio para o
    resto. `ressalvas` dizem onde uma das obras não cobre o que a ferramenta faz.
    """

    chave: str
    nome: str
    etapa: str
    referencias: tuple[str, ...] = ()
    ressalvas: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# As obras
# ---------------------------------------------------------------------------
#
# Só as que algum método da lista cita, e por isso a numeração tem lacunas. O teste reprova obra
# carregada que método nenhum cita.

_OBRAS: Final[tuple[tuple[str, str, Conferencia], ...]] = (
    (
        "R1",
        "LI, S.; PARK, Y.; DURAISINGHAM, S.; STROBEL, F. H.; KHAN, N.; SOLTOW, Q. A.; JONES, D. "
        "P.; PULENDRAN, B. Predicting network activity from high throughput metabolomics. **PLoS "
        "Computational Biology**, v. 9, n. 7, e1003123, 2013. DOI: 10.1371/journal.pcbi.1003123.",
        "conteúdo conferido",
    ),
    (
        "R2",
        "STEEGEN, S.; TUERLINCKX, F.; GELMAN, A.; VANPAEMEL, W. Increasing transparency through a "
        "multiverse analysis. **Perspectives on Psychological Science**, v. 11, n. 5, p. 702-712, "
        "2016. DOI: 10.1177/1745691616658637.",
        "conteúdo conferido",
    ),
    (
        "R6",
        "BROADHURST, D.; GOODACRE, R.; REINKE, S. N.; KULIGOWSKI, J.; WILSON, I. D.; LEWIS, M. "
        "R.; DUNN, W. B. Guidelines and considerations for the use of system suitability and "
        "quality control samples in mass spectrometry assays applied in untargeted clinical "
        "metabolomic studies. **Metabolomics**, v. 14, n. 6, art. 72, 2018. DOI: "
        "10.1007/s11306-018-1367-3.",
        "conteúdo conferido",
    ),
    (
        "R7",
        "CLEVELAND, W. S. Robust locally weighted regression and smoothing scatterplots. "
        "**Journal of the American Statistical Association**, v. 74, n. 368, p. 829-836, 1979. "
        "DOI: 10.1080/01621459.1979.10481038.",
        "conteúdo conferido",
    ),
    (
        "R8",
        "CLEVELAND, W. S.; DEVLIN, S. J. Locally weighted regression: an approach to regression "
        "analysis by local fitting. **Journal of the American Statistical Association**, v. 83, "
        "n. 403, p. 596-610, 1988. DOI: 10.1080/01621459.1988.10478639.",
        "conteúdo conferido",
    ),
    (
        "R9",
        "DUNN, W. B. et al. Procedures for large-scale metabolic profiling of serum and plasma "
        "using gas chromatography and liquid chromatography coupled to mass spectrometry. "
        "**Nature Protocols**, v. 6, n. 7, p. 1060-1083, 2011. DOI: 10.1038/nprot.2011.335.",
        "conteúdo conferido",
    ),
    (
        "R10",
        "FAN, S. et al. Systematic error removal using random forest for normalizing large-scale "
        "untargeted lipidomics data. **Analytical Chemistry**, v. 91, n. 5, p. 3590-3596, 2019. "
        "DOI: 10.1021/acs.analchem.8b05592.",
        "conteúdo conferido",
    ),
    (
        "R11",
        "WEI, R. et al. Missing value imputation approach for mass spectrometry-based "
        "metabolomics data. **Scientific Reports**, v. 8, n. 1, art. 663, 2018. DOI: "
        "10.1038/s41598-017-19120-0.",
        "conteúdo conferido",
    ),
    (
        "R26",
        "BREIMAN, L. Random forests. Machine Learning, v. 45, n. 1, p. 5-32, 2001. DOI: "
        "10.1023/A:1010933404324.",
        "conteúdo conferido",
    ),
    (
        "R27",
        "CORTES, C.; VAPNIK, V. Support-vector networks. Machine Learning, v. 20, n. 3, p. "
        "273-297, 1995. DOI: 10.1007/BF00994018.",
        "conteúdo conferido",
    ),
    (
        "R28",
        "LLOYD, S. Least squares quantization in PCM. IEEE Transactions on Information Theory, v. "
        "28, n. 2, p. 129-137, 1982. DOI: 10.1109/TIT.1982.1056489.",
        "só metadados",
    ),
    (
        "R29",
        "KOHONEN, T. Self-organized formation of topologically correct feature maps. Biological "
        "Cybernetics, v. 43, n. 1, p. 59-69, 1982. DOI: 10.1007/BF00337288.",
        "conteúdo conferido",
    ),
    (
        "R30",
        "WARD, J. H. Hierarchical grouping to optimize an objective function. Journal of the "
        "American Statistical Association, v. 58, n. 301, p. 236-244, 1963. DOI: "
        "10.1080/01621459.1963.10500845.",
        "só metadados",
    ),
    (
        "R31",
        "ANDERSON, M. J. A new method for non-parametric multivariate analysis of variance. "
        "Austral Ecology, v. 26, n. 1, p. 32-46, 2001. DOI: 10.1111/j.1442-9993.2001.01070.pp.x.",
        "conteúdo conferido",
    ),
    (
        "R32",
        "BASU, S.; DUREN, W.; EVANS, C. R.; BURANT, C. F.; MICHAILIDIS, G.; KARNOVSKY, A. Sparse "
        "network modeling and metscape-based visualization methods for the analysis of "
        "large-scale metabolomics data. Bioinformatics, v. 33, n. 10, p. 1545-1553, 2017. DOI: "
        "10.1093/bioinformatics/btx012.",
        "conteúdo conferido",
    ),
    (
        "R33",
        "JANKOVÁ, J.; VAN DE GEER, S. Confidence intervals for high-dimensional inverse "
        "covariance estimation. Electronic Journal of Statistics, v. 9, n. 1, p. 1205-1229, 2015. "
        "DOI: 10.1214/15-EJS1031.",
        "conteúdo conferido",
    ),
    (
        "R34",
        "TUKEY, J. W. Comparing individual means in the analysis of variance. Biometrics, v. 5, "
        "n. 2, p. 99-114, 1949. DOI: 10.2307/3001913.",
        "só metadados",
    ),
    (
        "R35",
        "FISHER, R. A. Statistical methods for research workers. Edinburgh: Oliver and Boyd, "
        "1925. 239 p.",
        "só metadados",
    ),
    (
        "R36",
        "KRUSKAL, W. H.; WALLIS, W. A. Use of ranks in one-criterion variance analysis. Journal "
        "of the American Statistical Association, v. 47, n. 260, p. 583-621, 1952. DOI: "
        "10.1080/01621459.1952.10483441.",
        "só metadados",
    ),
    (
        "R37",
        "DUNN, O. J. Multiple comparisons using rank sums. Technometrics, v. 6, n. 3, p. 241-252, "
        "1964. DOI: 10.1080/00401706.1964.10490181.",
        "só metadados",
    ),
    (
        "R38",
        "PEARSON, K. Mathematical contributions to the theory of evolution. III. Regression, "
        "heredity, and panmixia. Philosophical Transactions of the Royal Society of London. "
        "Series A, v. 187, p. 253-318, 1896. DOI: 10.1098/rsta.1896.0007.",
        "só metadados",
    ),
    (
        "R39",
        "SPEARMAN, C. The proof and measurement of association between two things. The American "
        "Journal of Psychology, v. 15, n. 1, p. 72-101, 1904. DOI: 10.2307/1412159.",
        "só metadados",
    ),
    (
        "R40",
        "KENDALL, M. G. A new measure of rank correlation. Biometrika, v. 30, n. 1-2, p. 81-93, "
        "1938. DOI: 10.1093/biomet/30.1-2.81.",
        "só metadados",
    ),
    (
        "R46",
        "TUKEY, John W. **Exploratory data analysis**. Reading, MA: Addison-Wesley, 1977. xvi, "
        "688 p. (Addison-Wesley series in behavioral science). ISBN 0-201-07616-0.",
        "só metadados",
    ),
    (
        "R47",
        "KUHN, Max. **caret**: classification and regression training. Versão 7.0-1. [S. l.]: The "
        "R Foundation, 2024. Pacote R. DOI: 10.32614/CRAN.package.caret. Disponível em: "
        "https://CRAN.R-project.org/package=caret. Acesso em: 25 set. 2026.",
        "conteúdo conferido",
    ),
    (
        "R48",
        "KUHN, Max. Building predictive models in R using the caret package. **Journal of "
        "Statistical Software**, v. 28, n. 5, p. 1-26, 2008. DOI: 10.18637/jss.v028.i05.",
        "conteúdo conferido",
    ),
    (
        "R49",
        "XIA, J.; MANDAL, R.; SINELNIKOV, I. V.; BROADHURST, D.; WISHART, D. S. MetaboAnalyst "
        "2.0—a comprehensive server for metabolomic data analysis. **Nucleic Acids Research**, v. "
        "40, n. W1, p. W127-W133, 2012. DOI: 10.1093/nar/gks374.",
        "conteúdo conferido",
    ),
    (
        "R50",
        "HACKSTADT, Amber J.; HESS, Ann M. Filtering for increased power for microarray data "
        "analysis. **BMC Bioinformatics**, v. 10, n. 1, art. 11, 2009. DOI: "
        "10.1186/1471-2105-10-11.",
        "conteúdo conferido",
    ),
    (
        "R51",
        "BENJAMINI, Yoav; HOCHBERG, Yosef. Controlling the false discovery rate: a practical and "
        "powerful approach to multiple testing. **Journal of the Royal Statistical Society Series "
        "B: Statistical Methodology**, v. 57, n. 1, p. 289-300, 1995. DOI: "
        "10.1111/j.2517-6161.1995.tb02031.x.",
        "conteúdo conferido",
    ),
    (
        "R52",
        "WELCH, B. L. The generalization of 'Student's' problem when several different population "
        "variances are involved. **Biometrika**, v. 34, n. 1-2, p. 28-35, 1947. DOI: "
        "10.1093/biomet/34.1-2.28.",
        "só metadados",
    ),
    (
        "R53",
        "MANN, H. B.; WHITNEY, D. R. On a test of whether one of two random variables is "
        "stochastically larger than the other. **The Annals of Mathematical Statistics**, v. 18, "
        "n. 1, p. 50-60, mar. 1947. DOI: 10.1214/aoms/1177730491.",
        "só metadados",
    ),
    (
        "R54",
        "SHAPIRO, S. S.; WILK, M. B. An analysis of variance test for normality (complete "
        "samples). **Biometrika**, v. 52, n. 3-4, p. 591-611, dez. 1965. DOI: "
        "10.1093/biomet/52.3-4.591.",
        "só metadados",
    ),
    (
        "R55",
        "WESTERHUIS, Johan A.; HOEFSLOOT, Huub C. J.; SMIT, Suzanne; VIS, Daniel J.; SMILDE, Age "
        "K.; VAN VELZEN, Ewoud J. J.; VAN DUIJNHOVEN, John P. M.; VAN DORSTEN, Ferdi A. "
        "Assessment of PLSDA cross validation. **Metabolomics**, v. 4, n. 1, p. 81-89, 2008. DOI: "
        "10.1007/s11306-007-0099-6.",
        "conteúdo conferido",
    ),
    (
        "R56",
        "PHIPSON, Belinda; SMYTH, Gordon K. Permutation p-values should never be zero: "
        "calculating exact p-values when permutations are randomly drawn. **Statistical "
        "Applications in Genetics and Molecular Biology**, v. 9, n. 1, art. 39, 2010. DOI: "
        "10.2202/1544-6115.1585.",
        "conteúdo conferido",
    ),
)

REFERENCIAS: Final[dict[str, Referencia]] = {
    numero: Referencia(numero, abnt, conferencia) for numero, abnt, conferencia in _OBRAS
}


# ---------------------------------------------------------------------------
# Os métodos implementados, e a ligação de cada um às obras
# ---------------------------------------------------------------------------
#
# A ligação: R7 e R8 são do LOESS e R9 do QC-RLSC; R10 é o artigo que o SERRF porta e R26 a
# Random Forest que ele usa; R11 a imputação por fração do mínimo; R46 o boxplot de Tukey; R47 e
# R48 o `nearZeroVar`; R49 e R50 o filtro por IQR; R28 a R40 uma referência por método das
# análises de N grupos, agrupamento, correlação, PERMANOVA e DSPC; R6 é o RSD de QC como régua. O
# LSD de Fisher e as ligações média, completa e simples ficam 'por conferir': R35 e R30 não os
# cobrem, e não há obra conferida para eles.

_CORRECAO: Final = "Correção do sinal"
_QUALIDADE: Final = "RSD dos QCs e filtros"
_IMPUTACAO: Final = "Imputação de ausentes"
_NORMALIZACAO: Final = "Normalização por amostra"
_TRANSFORMACAO: Final = "Transformação"
_ESCALONAMENTO: Final = "Escalonamento"
_UNIVARIADA: Final = "Análise univariada"
_N_GRUPOS: Final = "Análise de N grupos"
_CORRELACAO: Final = "Correlação"
_MULTIVARIADA: Final = "Análise multivariada"
_AGRUPAMENTO: Final = "Agrupamento"
_REDE_E_CLASSIFICACAO: Final = "Rede e classificação"
_BETA: Final = "Funcionalidades em beta"

_REDUCAO: Final = "Redução por posto"

METODOS: Final[tuple[Metodo, ...]] = (
    # --- Correção do sinal
    Metodo(
        "correction:loess",
        "LOESS ajustado aos QCs ao longo da ordem de injeção",
        _CORRECAO,
        ("R7", "R8", "R9"),
        (
            "Recomendação: citar 1979 como origem da regressão local ponderada e 1988 para o "
            "loess de grau 2 usado de fato, deixando explícito que a reponderação robusta de "
            "1979 NÃO é usada.",
        ),
    ),
    Metodo(
        "correction:serrf",
        "SERRF, correção por Random Forest treinada nos QCs",
        _CORRECAO,
        ("R10", "R26"),
        ("O título diz LIPIDÔMICA, e o estudo usa seis conjuntos de lipidômica.",),
    ),
    Metodo(
        "escolha_automatica",
        "Escolha automática do corretor pela medição do RSD dos QCs fora da amostra",
        _CORRECAO,
    ),
    Metodo(
        "diagnostico",
        "Diagnóstico da forma do erro sistemático (cascata de regras e ponto de mudança)",
        _CORRECAO,
    ),
    # --- RSD dos QCs e filtros
    Metodo(
        "rsd_variant:plain",
        "RSD dos QCs sem remoção de discrepantes (desvio-padrão sobre a média)",
        _QUALIDADE,
        ("R6",),
    ),
    Metodo(
        "rsd_variant:outlier_removed",
        "RSD dos QCs com remoção de discrepantes pelo critério de boxplot de Tukey",
        _QUALIDADE,
        ("R6", "R46"),
    ),
    Metodo("filtro:corte_por_rsd", "Corte por RSD dos QCs", _QUALIDADE, ("R6",)),
    Metodo(
        "filtro:variancia_quase_nula",
        "Filtro de variância quase nula (réplica do nearZeroVar do caret, razão 95/5 e 10% de "
        "valores distintos)",
        _QUALIDADE,
        ("R47", "R48"),
        (
            "os limiares do artigo de 2008 (menos de 20% de valores únicos e razão maior que 20) "
            "NÃO são os que o projeto usa (95/5 e 10%).",
        ),
    ),
    Metodo(
        "filtro:ausentes_por_grupo",
        "Descarte das features com fração de ausentes acima do corte em todos os grupos",
        _QUALIDADE,
    ),
    Metodo(
        "ranking_filter:iqr",
        f"{_REDUCAO} pelo intervalo interquartil (IQR)",
        _QUALIDADE,
        ("R49", "R50"),
        ("O artigo NÃO traz as frações por tamanho (5/10/25/40%).",),
    ),
    Metodo("ranking_filter:sd", f"{_REDUCAO} pelo desvio-padrão", _QUALIDADE),
    Metodo("ranking_filter:mad", f"{_REDUCAO} pelo desvio absoluto mediano (MAD)", _QUALIDADE),
    Metodo(
        "ranking_filter:rsd",
        f"{_REDUCAO} pelo desvio-padrão relativo em todas as amostras",
        _QUALIDADE,
    ),
    Metodo(
        "ranking_filter:nrsd",
        f"{_REDUCAO} pelo desvio relativo não paramétrico (MAD sobre a mediana)",
        _QUALIDADE,
    ),
    Metodo("ranking_filter:mean", f"{_REDUCAO} pela intensidade média", _QUALIDADE),
    Metodo("ranking_filter:median", f"{_REDUCAO} pela intensidade mediana", _QUALIDADE),
    # --- Imputação
    Metodo(
        "imputation:one_fifth_min",
        "Imputação por um quinto do menor valor positivo da feature",
        _IMPUTACAO,
        ("R11",),
        (
            "Serve para citar a fração do mínimo como abordagem usual para ausentes MNAR abaixo "
            "do limite de detecção, NÃO como a melhor.",
            "O fator do artigo é METADE, e não o UM QUINTO do laboratório.",
        ),
    ),
    Metodo("imputation:col_min", "Imputação pelo menor valor positivo da feature", _IMPUTACAO),
    Metodo("imputation:median", "Imputação pela mediana da feature", _IMPUTACAO),
    Metodo("imputation:mean", "Imputação pela média da feature", _IMPUTACAO),
    Metodo("imputation:knn", "Imputação por vizinhos mais próximos entre amostras", _IMPUTACAO),
    Metodo(
        "imputation:knn_features", "Imputação por vizinhos mais próximos entre features", _IMPUTACAO
    ),
    Metodo("imputation:missforest", "Imputação por missForest", _IMPUTACAO),
    Metodo("imputation:exclude", "Exclusão das features com algum valor ausente", _IMPUTACAO),
    # --- Normalização por amostra
    Metodo("sample_normalization:sum", "Normalização pela soma", _NORMALIZACAO),
    Metodo("sample_normalization:median", "Normalização pela mediana", _NORMALIZACAO),
    Metodo(
        "sample_normalization:pqn_group",
        "Normalização por quociente probabilístico (PQN) com referência nos QCs",
        _NORMALIZACAO,
    ),
    Metodo(
        "sample_normalization:sample_reference",
        "Normalização por quociente probabilístico (PQN) com uma amostra de referência",
        _NORMALIZACAO,
    ),
    Metodo(
        "sample_normalization:feature_reference",
        "Normalização por uma feature de referência",
        _NORMALIZACAO,
    ),
    Metodo(
        "sample_normalization:supplied_factors",
        "Normalização por fatores medidos fora do equipamento (peso, volume)",
        _NORMALIZACAO,
    ),
    Metodo("sample_normalization:quantile", "Normalização por quantis", _NORMALIZACAO),
    # --- Transformação
    Metodo(
        "transformation:log10", "Transformação logarítmica estabilizada, base 10", _TRANSFORMACAO
    ),
    Metodo("transformation:log2", "Transformação logarítmica estabilizada, base 2", _TRANSFORMACAO),
    Metodo("transformation:sqrt", "Transformação por raiz quadrada estabilizada", _TRANSFORMACAO),
    Metodo("transformation:cbrt", "Transformação por raiz cúbica", _TRANSFORMACAO),
    # --- Escalonamento
    Metodo("scaling:mean_center", "Centragem na média", _ESCALONAMENTO),
    Metodo(
        "scaling:auto",
        "Autoescalonamento (centrado e dividido pelo desvio-padrão)",
        _ESCALONAMENTO,
    ),
    Metodo(
        "scaling:pareto",
        "Escalonamento de Pareto (centrado e dividido pela raiz do desvio-padrão)",
        _ESCALONAMENTO,
    ),
    Metodo(
        "scaling:range",
        "Escalonamento por amplitude (centrado e dividido pela amplitude)",
        _ESCALONAMENTO,
    ),
    Metodo("scaling:vast", "Escalonamento VAST", _ESCALONAMENTO),
    # --- Análise univariada
    Metodo("univariada:welch", "Teste t de Welch", _UNIVARIADA, ("R52",)),
    Metodo("univariada:mann_whitney", "Teste U de Mann-Whitney", _UNIVARIADA, ("R53",)),
    Metodo(
        "normalidade:shapiro_wilk", "Teste de normalidade de Shapiro-Wilk", _UNIVARIADA, ("R54",)
    ),
    Metodo(
        "multiplos_testes:benjamini_hochberg",
        "Correção de Benjamini-Hochberg (FDR)",
        _UNIVARIADA,
        ("R51",),
        (
            "o artigo de 1995 enuncia uma regra de rejeição, e a forma de p-valor ajustado é "
            "posterior.",
        ),
    ),
    Metodo("fold_change", "Fold change pela razão das medianas", _UNIVARIADA),
    # --- Análise de N grupos
    Metodo("anova:anova", "ANOVA de uma via", _N_GRUPOS, ("R35",)),
    Metodo("anova:kruskal_wallis", "Teste de Kruskal-Wallis", _N_GRUPOS, ("R36",)),
    Metodo(
        "post_hoc:tukey",
        "Post-hoc de Tukey (HSD)",
        _N_GRUPOS,
        ("R34",),
        (
            "este artigo de 1949 é o procedimento de 'gaps' de Tukey.",
            "O HSD que o MetaboAnalyst e o scipy implementam se consolidou num relatório de "
            "1953, não publicado, que não foi buscado.",
        ),
    ),
    Metodo("post_hoc:fisher", "Post-hoc LSD de Fisher", _N_GRUPOS),
    Metodo("post_hoc:dunn", "Post-hoc de Dunn", _N_GRUPOS, ("R37",)),
    # --- Correlação
    Metodo("correlacao:pearson", "Correlação de Pearson", _CORRELACAO, ("R38",)),
    Metodo("correlacao:spearman", "Correlação de Spearman", _CORRELACAO, ("R39",)),
    Metodo("correlacao:kendall", "Correlação de Kendall", _CORRELACAO, ("R40",)),
    Metodo(
        "busca_por_padrao",
        "Busca por padrão: correlação de cada feature com um perfil",
        _CORRELACAO,
    ),
    # --- Análise multivariada
    Metodo("pca", "Análise de componentes principais (PCA)", _MULTIVARIADA),
    Metodo("regiao_de_hotelling", "Região de predição T² de Hotelling nas PCAs", _MULTIVARIADA),
    Metodo(
        "plsda",
        "PLS-DA com validação cruzada e teste de permutação",
        _MULTIVARIADA,
        ("R55",),
        ("O artigo defende validação cruzada de modelo em dupla camada.",),
    ),
    Metodo("permanova", "PERMANOVA", _MULTIVARIADA, ("R31",)),
    # --- Agrupamento
    Metodo("kmeans", "K-means, algoritmo de Lloyd", _AGRUPAMENTO, ("R28",)),
    Metodo("som", "Mapa auto-organizável (SOM)", _AGRUPAMENTO, ("R29",)),
    Metodo("agrupamento:ward", "Agrupamento hierárquico, ligação de Ward", _AGRUPAMENTO, ("R30",)),
    Metodo("agrupamento:average", "Agrupamento hierárquico, ligação média", _AGRUPAMENTO),
    Metodo("agrupamento:complete", "Agrupamento hierárquico, ligação completa", _AGRUPAMENTO),
    Metodo("agrupamento:single", "Agrupamento hierárquico, ligação simples", _AGRUPAMENTO),
    # --- Rede e classificação
    Metodo(
        "dspc", "Rede de correlação parcial esparsa (DSPC)", _REDE_E_CLASSIFICACAO, ("R32", "R33")
    ),
    Metodo(
        "classificador:random_forest",
        "Random Forest como classificador",
        _REDE_E_CLASSIFICACAO,
        ("R26",),
    ),
    Metodo(
        "classificador:svm",
        "SVM de kernel linear como classificador",
        _REDE_E_CLASSIFICACAO,
        ("R27",),
        (
            "a extensão multiclasse (um-contra-um ou um-contra-todos do scikit-learn) não está "
            "neste artigo.",
        ),
    ),
    # --- Em beta: no executável, fora da entrega ao laboratório
    Metodo("multiverso", "Teste de robustez (análise de multiverso)", _BETA, ("R2",)),
    Metodo(
        "vias:mummichog", "Predição de vias pelos picos, algoritmo do mummichog", _BETA, ("R1",)
    ),
    Metodo("vias:p_de_permutacao", "p de permutação das vias", _BETA, ("R56",)),
)

METODO_POR_CHAVE: Final[dict[str, Metodo]] = {metodo.chave: metodo for metodo in METODOS}

# O segundo passe da redução por posto é o MESMO método, sobre o que o primeiro deixou.
_EIXO_DO_METODO: Final[dict[str, str]] = {"ranking_filter_2": "ranking_filter"}


def chave_do_valor(eixo: str, valor: str) -> str:
    """A chave do método que um valor do vocabulário nomeia, como `imputation:knn`."""
    return f"{_EIXO_DO_METODO.get(eixo, eixo)}:{valor}"


def sem_marcacao(texto: str) -> str:
    """O texto sem o negrito da tabela (`**`), para a cópia em texto simples."""
    return texto.replace("**", "")


def _citacao(metodo: Metodo) -> str:
    """`R7 (conteúdo conferido), R8 (…)`, ou 'referência por conferir'."""
    if not metodo.referencias:
        return POR_CONFERIR
    return ", ".join(f"{rid} ({REFERENCIAS[rid].conferencia})" for rid in metodo.referencias)


def texto_das_referencias(metodos: Sequence[Metodo] = METODOS) -> str:
    """A seção da janela, em Markdown: a legenda, os métodos por etapa, e as obras com o texto ABNT.

    As obras vêm depois dos métodos, numa lista só, porque várias são citadas por mais de um
    método (R6 por três), e repetir o ABNT em cada um faria a seção crescer sem dizer mais.
    """
    partes = ["**Como ler a conferência**:", ""]
    partes += [f"- {definicao}" for definicao in (*LEGENDA_DA_CONFERENCIA, LEGENDA_DO_POR_CONFERIR)]
    etapa_corrente = None
    for metodo in metodos:
        if metodo.etapa != etapa_corrente:
            etapa_corrente = metodo.etapa
            partes += ["", f"**{metodo.etapa}**", ""]
        partes.append(f"- {metodo.nome} — {_citacao(metodo)}")
        partes += [f"    - Ressalva: «{ressalva}»" for ressalva in metodo.ressalvas]
    partes += ["", "**As obras**", ""]
    citadas = {rid for metodo in metodos for rid in metodo.referencias}
    partes += [
        f"- {referencia.numero} — {referencia.conferencia} — {referencia.abnt}"
        for referencia in REFERENCIAS.values()
        if referencia.numero in citadas
    ]
    return "\n".join(partes)


# ---------------------------------------------------------------------------
# O resumo metodológico
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ResumoMetodologico:
    """O texto para copiar, e as chaves dos métodos que ele cita, na ordem em que os cita.

    É por `metodos` que o teste afirma que cada método citado tem linha na lista, e ele é conferido
    contra o próprio texto: o nome de cada um aparece nele.
    """

    texto: str
    metodos: tuple[str, ...]


@dataclass(frozen=True)
class ArquivoExportado:
    """Um arquivo escrito em disco, o pedido que o produziu e o aceite da rota, se veio.

    `artefato` é a chave de `CHAVES_DE_ARTEFATO`: `corrigir`, `processar` ou `exportar`, a rota
    que produziu o arquivo, mesmo quando quem o pediu foi o Resultado de uma etapa da janela. O
    aceite (`TrabalhoAceito`) só importa sob escolha automática: é ele que diz qual corretor a
    política escolheu, e o arquivo de bytes não o diz.

    `o_que_e` é o que o arquivo é, dito por quem o pediu (por exemplo, a matriz da etapa de
    normalização sobre a matriz sem correção). Sem ele, o resumo diz o da rota.
    """

    nome: str
    artefato: str
    pedido: Mapping[str, Any]
    aceite: Mapping[str, Any] | None = None
    o_que_e: str | None = None


class _Citacoes:
    """Os métodos citados, na ordem em que o texto os cita, e a forma de os citar."""

    def __init__(self) -> None:
        self.chaves: list[str] = []

    def cita(self, chave: str, *, no_meio: bool = False) -> str:
        """O nome do método e as obras entre colchetes: `[R7, R8]` ou `[referência por conferir]`.

        `no_meio` põe a inicial em minúscula, para o nome caber no meio da frase. Sigla e nome com
        hífen, como LOESS e K-means, ficam como estão, porque a segunda letra deles não é minúscula.
        """
        metodo = METODO_POR_CHAVE[chave]
        if chave not in self.chaves:
            self.chaves.append(chave)
        nome = metodo.nome
        if no_meio and len(nome) > 1 and nome[1].islower():
            nome = nome[0].lower() + nome[1:]
        obras = ", ".join(metodo.referencias) if metodo.referencias else POR_CONFERIR
        return f"{nome} [{obras}]"


def _numero(valor: float | int) -> str:
    """`2158` → `2.158`, `0.6` → `0,6`: o número como a tela em português o escreve."""
    if isinstance(valor, int):
        return f"{valor:,}".replace(",", ".")
    return f"{valor:g}".replace(".", ",")


def _percentual(fracao: Any) -> str:
    return f"{_numero(round(float(fracao) * 100, 6))}%"


def _texto_dos_lotes(pedido: Mapping[str, Any]) -> str:
    """Os lotes do pedido, e os cortes quando o pedido os leva.

    Sem cortes, a correção parte a corrida em blocos iguais pela ordem de injeção, que é como a rota
    lê a falta do campo. Sem `n_batches`, a frase diz isso e não inventa o padrão da rota.
    """
    bruto = pedido.get("n_batches")
    if bruto is None:
        return "o pedido não diz o número de lotes"
    lotes = int(bruto)
    if lotes <= 1:
        return "em um lote só"
    cortes = pedido.get("batch_cuts")
    if cortes:
        lista = ", ".join(str(corte) for corte in json.loads(cortes))
        return f"em {lotes} lotes, cortados antes das injeções {lista}"
    return f"em {lotes} lotes de tamanho igual pela ordem de injeção"


def _correcao(
    c: _Citacoes, pedido: Mapping[str, Any], corretor: str | None, *, automatico: bool
) -> str:
    """O corretor aplicado e os lotes, ou 'nenhuma', e, sob escolha automática, quem escolheu.

    Com `none` os lotes não aparecem: sem correção, eles não mudam nada no arquivo.
    """
    if corretor == "none":
        frase = "nenhuma"
    elif corretor is None:
        frase = "a resposta não disse qual corretor"
    else:
        frase = f"{c.cita(chave_do_valor('correction', corretor))}, {_texto_dos_lotes(pedido)}"
    if automatico:
        frase += f" — {c.cita('escolha_automatica', no_meio=True)}"
    return frase


def _semente(pedido: Mapping[str, Any]) -> str:
    semente = pedido.get("seed")
    return "Semente: o pedido não a diz." if semente is None else f"Semente: {semente}."


# Uma linha por etapa de `CANONICAL_ORDER`, ou `None` quando o pedido não a pediu. O teste confere
# que as chaves são as etapas do pipeline: uma etapa nova sem linha reprova em vez de sumir.


def _corte_por_rsd(c: _Citacoes, p: Mapping[str, Any]) -> str | None:
    if p.get("rsd_cutoff") is None:
        return None
    variante = c.cita(chave_do_valor("rsd_variant", str(p.get("rsd_variant"))))
    return (
        f"{c.cita('filtro:corte_por_rsd')}: removidas as features com mais de "
        f"{_percentual(p.get('rsd_cutoff'))} pelo {variante}"
    )


def _reducao(c: _Citacoes, p: Mapping[str, Any], eixo: str, fracao: str, passe: str) -> str | None:
    valor = p.get(eixo)
    if not valor or valor == "none":
        return None
    mantida = p.get(fracao)
    quanto = (
        f"mantendo {_percentual(mantida)} das features"
        if mantida is not None
        else "mantendo a fração que a regra por tamanho da matriz define"
    )
    return f"{c.cita(chave_do_valor(eixo, str(valor)))}, {passe}, {quanto}"


def _primeiro_passe(c: _Citacoes, p: Mapping[str, Any]) -> str | None:
    return _reducao(c, p, "ranking_filter", "ranking_keep_fraction", "1º passe")


def _segundo_passe(c: _Citacoes, p: Mapping[str, Any]) -> str | None:
    return _reducao(c, p, "ranking_filter_2", "ranking_keep_fraction_2", "2º passe")


def _variancia_quase_nula(c: _Citacoes, _p: Mapping[str, Any]) -> str:
    # O pipeline aplica esta etapa sempre: ela não tem campo no pedido (`process_matrix_detailed`).
    return f"{c.cita('filtro:variancia_quase_nula')}, que o pipeline sempre aplica"


def _ausentes(c: _Citacoes, p: Mapping[str, Any]) -> str | None:
    if p.get("missing_threshold") is None:
        return None
    return (
        f"{c.cita('filtro:ausentes_por_grupo')}: corte de {_percentual(p.get('missing_threshold'))}"
    )


def _do_eixo(eixo: str) -> Callable[[_Citacoes, Mapping[str, Any]], str | None]:
    def linha(c: _Citacoes, p: Mapping[str, Any]) -> str | None:
        valor = p.get(eixo)
        if not valor or valor == "none":
            return None
        texto = c.cita(chave_do_valor(eixo, str(valor)))
        referencia = p.get("normalization_reference")
        if eixo == "sample_normalization" and referencia:
            texto += f", sobre «{referencia}»"
        return texto

    return linha


LINHA_DA_ETAPA: Final[dict[str, Callable[[_Citacoes, Mapping[str, Any]], str | None]]] = {
    "filter_by_qc_rsd": _corte_por_rsd,
    "filter_by_ranking": _primeiro_passe,
    "filter_by_ranking_2": _segundo_passe,
    "filter_near_zero_variance": _variancia_quase_nula,
    "drop_by_group_missingness": _ausentes,
    "impute": _do_eixo("imputation"),
    "normalize_samples": _do_eixo("sample_normalization"),
    "transform": _do_eixo("transformation"),
    "scale_features": _do_eixo("scaling"),
}


def _preprocessamento(c: _Citacoes, pedido: Mapping[str, Any]) -> list[str]:
    linhas = [LINHA_DA_ETAPA[etapa](c, pedido) for etapa in CANONICAL_ORDER]
    return [f"- {linha}." for linha in linhas if linha]


def _dicionario(valor: Any) -> Mapping[str, Any]:
    return valor if isinstance(valor, Mapping) else {}


def _correcao_multipla(c: _Citacoes, resumo: Mapping[str, Any]) -> str:
    """A correção de múltiplos testes que a resposta REGISTROU, e só ela é citada.

    A rota escreve o nome por extenso no campo `correcao`; uma correção que não fosse a de
    Benjamini-Hochberg sairia com o nome dela e sem obra, em vez de herdar a citação errada.
    """
    bruto = resumo.get("correcao")
    if bruto is None:
        return "sem correção de múltiplos testes registrada na resposta"
    texto = str(bruto)
    if "Benjamini-Hochberg" in texto:
        return f"pela {c.cita('multiplos_testes:benjamini_hochberg', no_meio=True)}"
    return f"por «{texto}»"


def _analises(c: _Citacoes, resposta: Mapping[str, Any]) -> list[str]:
    """As análises que a resposta trouxe; a que não veio não correu, e não é mencionada."""
    linhas: list[str] = []
    volcano = _dicionario(resposta.get("volcano"))
    if volcano:
        teste = c.cita(
            "univariada:welch" if volcano.get("parametrico") else "univariada:mann_whitney"
        )
        rota = volcano.get("rota_da_normalidade")
        if rota == "shapiro":
            shapiro = c.cita("normalidade:shapiro_wilk", no_meio=True)
            pvalor = volcano.get("shapiro_pvalue")
            sufixo = "" if pvalor is None else f", p = {_numero(float(f'{float(pvalor):.2g}'))}"
            como = f"escolhido pelo {shapiro} sobre os resíduos de todas as features{sufixo}"
        elif rota == "shapiro_unavailable":
            como = "porque não havia resíduos que bastassem para o teste de normalidade"
        else:
            como = "escolhido no pedido"
        linhas.append(
            f"- {teste} entre {volcano.get('grupo_a')} (n = {volcano.get('n_grupo_a')}) e "
            f"{volcano.get('grupo_b')} (n = {volcano.get('n_grupo_b')}), em "
            f"{_numero(int(volcano.get('n_features_testadas') or 0))} features, {como}; "
            f"q-valores {_correcao_multipla(c, volcano)}; "
            f"{c.cita('fold_change', no_meio=True)}."
        )
    # A normalidade por variável e por grupo. Grupos vazios querem dizer que a análise correu sem
    # amostra de estudo, e aí não há o que citar.
    normalidade = _dicionario(resposta.get("normalidade"))
    grupos_da_normalidade = normalidade.get("grupos") or ()
    if grupos_da_normalidade:
        linhas.append(
            f"- {c.cita('normalidade:shapiro_wilk')} por variável e por grupo de estudo, em "
            f"{len(grupos_da_normalidade)} grupos, com as variáveis que têm valor imputado no "
            "grupo contadas à parte."
        )
    anova = _dicionario(resposta.get("anova"))
    if anova:
        teste = c.cita(chave_do_valor("anova", str(anova.get("metodo"))))
        post_hoc = c.cita(
            chave_do_valor("post_hoc", str(anova.get("post_hoc_metodo"))), no_meio=True
        )
        grupos = len(anova.get("grupos") or ())
        alfa = anova.get("alfa_fdr")
        corte = "que sobrevivem" if alfa is None else f"com q abaixo de {_numero(float(alfa))}"
        linhas.append(
            f"- {teste} entre {grupos} grupos, com {post_hoc} nas features {corte}, "
            f"{_correcao_multipla(c, anova)}."
        )
    plsda = _dicionario(resposta.get("plsda"))
    if plsda:
        linhas.append(
            f"- {c.cita('plsda')} entre {plsda.get('grupo_positivo')} "
            f"(n = {plsda.get('n_positivo')}) e {plsda.get('grupo_negativo')} "
            f"(n = {plsda.get('n_negativo')}): {plsda.get('n_components')} componentes, "
            f"validação cruzada em {plsda.get('cv_folds')} partes, "
            f"{_numero(int(plsda.get('n_permutations') or 0))} permutações, "
            f"semente {plsda.get('seed')}."
        )
    correlacao = _dicionario(resposta.get("correlacao"))
    if correlacao:
        linhas.append(
            f"- {c.cita(chave_do_valor('correlacao', str(correlacao.get('metodo'))))} entre "
            f"features: {correlacao.get('amostragem')}."
        )
    busca = _dicionario(resposta.get("busca_por_padrao"))
    if busca:
        linhas.append(
            f"- {c.cita('busca_por_padrao')} «{busca.get('referencia')}», pela "
            f"{c.cita(chave_do_valor('correlacao', str(busca.get('metodo'))), no_meio=True)}."
        )
    kmeans = _dicionario(resposta.get("kmeans"))
    if kmeans:
        linhas.append(
            f"- {c.cita('kmeans')} com k = {kmeans.get('k')}, semente {kmeans.get('semente')}: "
            f"{kmeans.get('amostragem')}."
        )
    som = _dicionario(resposta.get("som"))
    if som:
        linhas.append(
            f"- {c.cita('som')} de {som.get('linhas')} por {som.get('colunas')} unidades, "
            f"{som.get('epocas')} épocas, semente {som.get('semente')}: {som.get('amostragem')}."
        )
    rede = _dicionario(resposta.get("rede"))
    if rede:
        linhas.append(
            f"- {c.cita('dspc')}, alfa {_numero(float(rede.get('alfa') or 0))}: "
            f"{rede.get('amostragem')}."
        )
    for campo in ("random_forest", "svm"):
        classificador = _dicionario(resposta.get(campo))
        if classificador:
            linhas.append(
                f"- {c.cita(chave_do_valor('classificador', str(classificador.get('kind'))))}, "
                f"validação cruzada em {classificador.get('cv_folds')} partes, "
                f"{_numero(int(classificador.get('n_permutations') or 0))} permutações, semente "
                f"{classificador.get('semente')}: {classificador.get('amostragem')}."
            )
    return linhas


def _fecho(c: _Citacoes) -> list[str]:
    """As obras citadas, pela ordem dos números e com a conferência, e as ressalvas delas."""
    citadas = sorted(
        {rid for chave in c.chaves for rid in METODO_POR_CHAVE[chave].referencias},
        key=lambda rid: int(rid[1:]),
    )
    linhas: list[str] = []
    if citadas:
        linhas += ["", "Referências citadas, com a conferência de cada uma:"]
        linhas += [
            f"[{rid}] {sem_marcacao(REFERENCIAS[rid].abnt)} — {REFERENCIAS[rid].conferencia}"
            for rid in citadas
        ]
    com_ressalva = [
        METODO_POR_CHAVE[chave] for chave in c.chaves if METODO_POR_CHAVE[chave].ressalvas
    ]
    if com_ressalva:
        linhas += ["", "Antes de citar, as ressalvas sobre estas obras:"]
        linhas += [
            f"- {metodo.nome}: " + " ".join(f"«{ressalva}»" for ressalva in metodo.ressalvas)
            for metodo in com_ressalva
        ]
    return linhas


def resumo_da_execucao(
    pedido: Mapping[str, Any], resposta: Mapping[str, Any]
) -> ResumoMetodologico:
    """O resumo de uma execução: o `data` de `POST /execucoes` e o corpo do `200` que a encerrou.

    O corretor é o que a resposta registrou (`pcas_da_correcao.corretor`), e não o campo do pedido:
    sob a escolha automática a rota troca o corretor, e o pedido diria o que a tela mostrava antes
    de a política decidir.
    """
    c = _Citacoes()
    registrado = _dicionario(resposta.get("pcas_da_correcao")).get("corretor")
    corretor = registrado or pedido.get("correction")
    correcao = _correcao(
        c,
        pedido,
        None if corretor is None else str(corretor),
        automatico=bool(pedido.get("auto")),
    )
    linhas = [
        f"Resumo metodológico da execução {resposta.get('run_id')} — estudo "
        f"«{pedido.get('estudo')}», modo {pedido.get('mode')}.",
        "",
        f"Correção do sinal: {correcao}.",
        _semente(pedido),
        "",
        "Pré-processamento, na ordem em que o pipeline o aplica:",
        *_preprocessamento(c, pedido),
    ]
    analises = _analises(c, resposta)
    if analises:
        linhas += ["", "Análises:", *analises]
    saida = resposta.get("n_features_saida")
    if saida is not None:
        linhas += ["", f"Resultado: {_numero(int(saida))} features na matriz processada."]
    linhas += _fecho(c)
    return ResumoMetodologico("\n".join(linhas), tuple(c.chaves))


# O que cada artefato é, sem afirmar correção: com `none` o arquivo da etapa de correção não foi
# corrigido, e 'matriz corrigida' seria falso.
_O_QUE_O_ARTEFATO_E: Final[dict[str, str]] = {
    "corrigir": "matriz da etapa de correção",
    "processar": "matriz pré-processada",
    "exportar": "matriz no formato do MetaboAnalyst",
}


# A política do denominador nulo do LOESS que cada rota de artefato usa quando o pedido não a
# manda, e a janela não a manda: `null_denominator` não está entre os campos de cada artefato em
# `janela/artefatos.py`. O padrão em `api/rotas.py::corrigir` e `::processar` é `r_faithful`, e em
# `::exportar` é `uncorrected_feature`, porque sob `r_faithful` o LOESS produz infinito nos quatro
# modos e o escritor do MetaboAnalyst recusa infinito. As prévias das etapas usam o padrão do
# `LoessCorrector`, que é `r_faithful`. `tests/test_referencias.py` confere contra as assinaturas.
POLITICA_DO_DENOMINADOR_NULO_DA_ROTA: Final[dict[str, str]] = {
    "corrigir": "r_faithful",
    "processar": "r_faithful",
    "exportar": "uncorrected_feature",
}
# A contagem das features sem correção não aparece: o arquivo volta como bytes, e nem o aceite nem
# o relatório do corte a trazem. O resumo diz a política, que decide o que acontece com elas, e diz
# que o número não veio.
#
# As duas frases tratam à parte a feature de mediana zero. Com a mediana da linha também zero, o
# divisor do passo 5 é `0/0`, a guarda (`divisor == 0.0`) não o vê, e a feature sai com células
# vazias onde havia medida; e o passo 11 devolve a zero as células que já eram zero. O que cada
# frase afirma é o que `tests/unit/test_loess_uncorrected.py` mede.
_FRASE_DA_POLITICA: Final[dict[str, str]] = {
    "r_faithful": (
        "Denominador nulo do LOESS pela política `r_faithful`, como no R: onde a curva ajustada "
        "vale zero, a divisão dá infinito nas células medidas, e as que já eram zero continuam "
        "zero. Se a mediana da feature também é zero, a célula medida sai vazia (NaN) em vez de "
        "infinita. É a política das prévias que as abas de etapa mostram."
    ),
    "uncorrected_feature": (
        "Denominador nulo do LOESS pela política `uncorrected_feature`: onde a curva ajustada vale "
        "zero e a mediana da feature é positiva, a feature sai sem correção, porque o formato do "
        "MetaboAnalyst recusa infinito. Se a mediana também é zero, a divisão dá vazio (NaN) e não "
        "infinito, a guarda não a alcança, e a feature sai com células vazias onde havia medida. "
        "As prévias que as abas de etapa mostram usam `r_faithful`, e nas features sem correção "
        "este arquivo difere delas; quantas são não volta com o arquivo."
    ),
}


def _denominador_nulo(arquivo: ArquivoExportado) -> str:
    """A política do denominador nulo que produziu o arquivo: a do pedido, ou o padrão da rota."""
    politica = arquivo.pedido.get("null_denominator") or (
        POLITICA_DO_DENOMINADOR_NULO_DA_ROTA.get(arquivo.artefato)
    )
    if politica is None:
        return "Denominador nulo do LOESS: a política que produziu este arquivo não é conhecida."
    return _FRASE_DA_POLITICA.get(
        str(politica), f"Denominador nulo do LOESS pela política `{politica}`."
    )


def _pede_preprocessamento(pedido: Mapping[str, Any]) -> bool:
    """O pedido liga alguma etapa do pré-processamento além do filtro de variância quase nula?

    É a pergunta de `preprocessing.pipeline.requests_preprocessing`, feita ao pedido pelas mesmas
    linhas que o resumo escreve: cada etapa tem linha exatamente quando o campo que a liga está no
    pedido. A exportação para o MetaboAnalyst só corre o pré-processamento, e o filtro de variância
    quase nula com ele, quando a resposta é sim; sem isto o resumo diria que o pipeline aplicou
    aquele filtro a um arquivo que nunca passou por ele. As citações vão a um rascunho, para a
    pergunta não citar nada.
    """
    rascunho = _Citacoes()
    return any(
        LINHA_DA_ETAPA[etapa](rascunho, pedido)
        for etapa in CANONICAL_ORDER
        if etapa != "filter_near_zero_variance"
    )


def _corretor_do_arquivo(arquivo: ArquivoExportado) -> str | None:
    """Sob escolha automática, o corretor vem do aceite; sem aceite, `None`, e a frase o diz."""
    if arquivo.pedido.get("auto"):
        delegacao = _dicionario(_dicionario(arquivo.aceite).get("delegacao"))
        escolhido = delegacao.get("correction")
        return None if escolhido is None else str(escolhido)
    return str(arquivo.pedido.get("method"))


def resumo_da_exportacao(arquivos: Sequence[ArquivoExportado]) -> ResumoMetodologico:
    """O resumo dos arquivos que um gesto de exportação escreveu: um, ou os três do lote.

    Um bloco por arquivo, cada um com o seu pedido: no lote os três diferem pelo corretor, e juntar
    o que é comum afirmaria uma igualdade que só o pedido de cada um pode confirmar.
    """
    if not arquivos:
        raise ValueError("o resumo de uma exportação pede ao menos um arquivo escrito")
    c = _Citacoes()
    modo = arquivos[0].pedido.get("mode")
    quantos = "um arquivo" if len(arquivos) == 1 else f"{len(arquivos)} arquivos"
    linhas = [f"Resumo metodológico da exportação — modo {modo}, {quantos}."]
    for arquivo in arquivos:
        pedido = arquivo.pedido
        o_que = arquivo.o_que_e or _O_QUE_O_ARTEFATO_E.get(arquivo.artefato, arquivo.artefato)
        corretor = _corretor_do_arquivo(arquivo)
        correcao = _correcao(c, pedido, corretor, automatico=bool(pedido.get("auto")))
        linhas += [
            "",
            f"{arquivo.nome} — {o_que}.",
            f"Correção do sinal: {correcao}.",
        ]
        # O denominador nulo só existe no LOESS, e cada rota o trata por uma política
        # (`POLITICA_DO_DENOMINADOR_NULO_DA_ROTA`). Dizê-lo separa o arquivo MetaboAnalyst da
        # prévia que a aba mostrou.
        if corretor == "loess":
            linhas.append(_denominador_nulo(arquivo))
        linhas.append(_semente(pedido))
        if arquivo.artefato == "processar" or (
            arquivo.artefato == "exportar" and _pede_preprocessamento(pedido)
        ):
            linhas += ["Pré-processamento, na ordem em que o pipeline o aplica:"]
            linhas += _preprocessamento(c, pedido)
        if arquivo.artefato == "exportar":
            teto = int(pedido.get("max_features") or 0)
            variante = c.cita(chave_do_valor("rsd_variant", str(pedido.get("rsd_variant"))))
            linhas.append(
                f"Formato do MetaboAnalyst: se a matriz passar de {_numero(teto)} features, ficam "
                f"as {_numero(teto)} de menor {variante}."
            )
    linhas += _fecho(c)
    return ResumoMetodologico("\n".join(linhas), tuple(c.chaves))
