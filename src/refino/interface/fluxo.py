"""Declaração pura do fluxo que a janela e a API usam.

Aqui ficam os passos do fluxo, os pontos de decisão automática, os controles de sobrescrita, a
tradução das métricas de qualidade, as equivalências com o MetaboAnalyst e a prosa das
estimativas de custo. O módulo não conhece Qt nem HTTP. As duas camadas leem daqui, então um
rótulo ou uma regra muda num lugar só, e testar a declaração é aritmética.
"""

import csv
import hashlib
import io
import math
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Final, Literal, get_args

from refino.diagnosis.batches import MIN_QC_PER_PROPOSED_BATCH
from refino.domain.analysis import PADRAO_DA_ANALISE, AnalysisSpec
from refino.domain.labels import (
    DefectType,
    Imputation,
    RankingFilter,
    SampleNormalization,
    Scaling,
    Transformation,
)
from refino.domain.spec import (
    OPTIONAL_METRIC_FIELDS,
    REQUIRED_METRIC_FIELDS,
    SUPPORTED_OVERRIDES,
)
from refino.metrics.crossval import jackknife_cost_ratio
from refino.multiverse.grid import GridDefinition

# Por onde a coisa declarada pode ser obtida. A tela nomeia coisas que vivem noutro lugar, e sem
# este eixo 'disponível' não diria onde.
# `nenhum` é valor de primeira classe: a coisa é declarada e não tem por onde ser obtida, como
# algumas métricas da tela de qualidade. Colapsá-lo em `None` misturaria 'não sei' com 'sei que
# não há'.
Caminho = Literal["api", "cli", "nenhum"]

# Em que seção o controle vive. Cada controle é desenhado num lugar só, e a declaração é a única
# coisa que sabe qual.
Secao = Literal["decisao", "execucao"]

# Em que unidade o número se lê: '0,5802' vira '58,0%', '1,8523' fica '1,852'. É campo fechado
# porque ler a unidade por substring da prosa formataria em silêncio a próxima métrica com a
# política errada. Contagem é valor à parte: `n_significant_fdr` é inteiro, e '0,000' para
# nenhuma feature significativa diria que o número foi arredondado.
FormatoDoValor = Literal["percentagem", "razao", "contagem"]

# Sobre que população o número foi calculado. `vetor_de_rsd` marca as quatro métricas que
# `build_quality_report` tira do mesmo vetor de RSD (mediana, média e as duas frações): o
# denominador é `n_features_rsd_defined`, e os rótulos são o `rsd_variant` e o `rsd_estimator` do
# relatório, sem os quais RSDs não se comparam.
# `nenhuma` afirma que a métrica não agrega aquele vetor. As duas de RSD fora da amostra ficam aqui
# de propósito: o `rsd_estimator` do relatório descreve o vetor principal, e colá-lo a um número
# medido por outro estimador daria o rótulo certo sobre a população errada.
PopulacaoDaMetrica = Literal["vetor_de_rsd", "nenhuma"]

# Sob que tema a métrica se lê. É um campo, e não uma lista em cada painel, porque uma partição
# escrita em dois lugares diverge no primeiro acréscimo sem sintoma nenhum.
# O quinto valor é um residual nomeado, com composição afirmada membro a membro (hoje só
# `n_significant_fdr`), e não um balde que absorve o próximo erro. `n_significant_fdr` fica fora
# dos quatro temas porque não é nota de qualidade: mais features significativas não indicam
# correção melhor, e agrupá-la com as outras diria o contrário por vizinhança.
# O vocabulário é fechado: uma métrica com tema fora destes cinco aparece marcada na tela, e quem
# a marca, em runtime, é `agrupa_metricas_da_qualidade`.
TemaDaMetrica = Literal[
    "dispersao_na_amostra",
    "dispersao_fora_da_amostra",
    "estrutura_no_pca",
    "validacao_do_plsda",
    "nao_e_nota_de_qualidade",
]

__all__ = [
    "CAMINHO_DA_METRICA",
    "CAUDA_DA_EXECUCAO_S",
    "CELULAS_POR_CORRETOR",
    "CHAVES_DE_ARTEFATO",
    "CHAVE_SEM_TEMA",
    "COLUNAS_DA_TABELA_DA_NORMALIDADE",
    "COMO_O_PALPITE_DECIDE",
    "CONTROLES_DE_SOBRESCRITA",
    "CORRETORES_DA_GRADE",
    "CORTE_SUGERIDO_NO_METABOANALYST",
    "CUSTO_DA_ANALISE_S",
    "CUSTO_DE_UM_AJUSTE_MS",
    "CUSTO_MEDIDO_POR_MODO",
    "CUSTO_POR_CORRETOR",
    "CUSTO_POR_FEATURE_MS",
    "CUSTO_POR_LOTE",
    "DENTRO_DO_FLUXO",
    "DICA_DA_FAMILIA",
    "DICA_DO_TESTE_UNIVARIADO",
    "EIXOS_DA_GRADE",
    "EIXOS_PROPOSTOS",
    "EQUIVALENCIA_NO_METABOANALYST",
    "FAMILIAS_DE_ANALISE",
    "FATOR_SERIAL_DO_SERRF",
    "FORMATADORES_DE_JUSTIFICATIVA",
    "GANHO_DAS_FOLHAS",
    "GANHO_DO_CORRETOR",
    "LIMITE_DA_NORMALIDADE",
    "METRICAS_DA_TELA_DE_QUALIDADE",
    "METRICAS_DE_VALIDACAO_DO_PLSDA",
    "METRICAS_SOB_PAR_DE_GRUPOS",
    "MODOS_SUGERIDOS",
    "NAO_CALCULADA",
    "NOMES_DO_TESTE_UNIVARIADO",
    "NOME_DA_TABELA_DA_NORMALIDADE",
    "NORMALIDADE_EM_REPOUSO",
    "NORMALIDADE_NAO_PEDIDA",
    "PARES_DE_VALIDACAO_OBRIGATORIA",
    "PONTOS_DE_DECISAO_AUTOMATICA",
    "PREFIXO_DE_LEITURA",
    "PREFIXO_DO_ENDERECO",
    "PROPOSTAS_POR_ROTA",
    "PROSA_DA_CONCORDANCIA",
    "PROSA_DA_VARIANTE_DE_RSD",
    "PROSA_DO_ESTIMADOR_DE_RSD",
    "ROTULOS_DE_DEFEITO",
    "ROTULO_DA_TABELA_DA_NORMALIDADE",
    "SEM_CAMINHO",
    "SEM_CAMINHO_PARA_PLSDA",
    "SEM_CONTRAPARTE_TRANSCRITA",
    "SEM_CORRETOR",
    "SEM_MEDICAO_DA_CLASSIFICACAO",
    "SEM_PLSDA",
    "SEM_VALOR_DEFINIDO",
    "SOB_PAR_DE_GRUPOS",
    "SOB_PAR_DE_GRUPOS_NA_JANELA",
    "TEMAS_DA_QUALIDADE",
    "TEMA_SEM_DECLARACAO",
    "TITULO_DA_NORMALIDADE",
    "TRADUCAO_DA_METRICA",
    "VALOR_DE_REPOUSO",
    "Caminho",
    "ControleDaTela",
    "FamiliaDeAnalise",
    "FormatoDoValor",
    "GrupoDeMetricas",
    "MetricaTraduzida",
    "NoMetaboAnalyst",
    "PassoDoFluxo",
    "PontoDeDecisao",
    "PopulacaoDaMetrica",
    "Secao",
    "TemaDaMetrica",
    "TemaDeQualidade",
    "agrupa_metricas_da_qualidade",
    "aviso_do_plsda",
    "cauda_da_execucao_s",
    "chave_da_entrada",
    "csv_da_normalidade",
    "eixos_do_ponto",
    "eixos_propostos_do_ponto",
    "eixos_so_do_humano_do_ponto",
    "endereco_da_metrica",
    "equivalencia_do_eixo",
    "estado_de_ausencia",
    "estimativa_da_execucao",
    "estimativa_da_medicao",
    "estimativa_de_custo",
    "etapas_das_analises",
    "familia_e_barata",
    "familia_pedida",
    "familias_pedidas",
    "familias_por_medir",
    "frase_das_recusas",
    "justificativa_em_prosa",
    "legenda_da_metrica",
    "partes_do_defeito",
    "procedencia_do_plsda",
    "prosa_da_concordancia",
    "prosa_da_confianca",
    "prosa_da_confianca_do_diagnostico",
    "prosa_da_equivalencia",
    "prosa_da_medicao",
    "prosa_da_recomendacao",
    "prosa_da_variante_de_rsd",
    "prosa_das_analises_pedidas",
    "prosa_das_etapas_pedidas",
    "prosa_de_duracao",
    "prosa_do_defeito",
    "prosa_do_estimador_de_rsd",
    "prosa_do_numero",
    "prosa_do_palpite_medido",
    "prosa_do_par",
    "prosa_dos_grupos_da_normalidade",
    "prosa_dos_lotes",
    "prosa_dos_rotulos_do_rsd",
    "rotulo_da_metrica",
    "segundos_da_medicao",
    "valor_da_metrica",
]


# Os eixos da grade que a interface deixa escolher: chave, rótulo, padrão e opções. Vivem aqui,
# e não em cada tela, para que haja uma cópia só. As opções saem de `get_args` do domínio, então um
# rótulo novo aparece sem ninguém lembrar de um segundo lugar.
EIXOS_DA_GRADE: Final[tuple[tuple[str, str, str, tuple[str, ...]], ...]] = (
    # A ordem da tupla é a ordem em que o pipeline aplica as etapas, e a redução por posto é a
    # segunda da ordem canônica: quem lê a tela de cima a baixo lê a sequência.
    ("ranking_filter", "Redução por posto — 1º passe", "none", get_args(RankingFilter)),
    # O segundo passe vem logo depois porque é a etapa seguinte, como no filtro em dois eixos do
    # MetaboAnalyst.
    ("ranking_filter_2", "Redução por posto — 2º passe", "none", get_args(RankingFilter)),
    ("imputation", "Imputação de ausentes", "one_fifth_min", get_args(Imputation)),
    ("sample_normalization", "Normalização por amostra", "none", get_args(SampleNormalization)),
    ("transformation", "Transformação", "none", get_args(Transformation)),
    ("scaling", "Escalonamento", "none", get_args(Scaling)),
)


# ==========================================================================
# A contraparte no MetaboAnalyst, no vocabulário de quem já usa a ferramenta
# ==========================================================================
#
# Quem abre a janela lê `supplied_factors` e não tem como saber que é o controle que no
# MetaboAnalyst se chama 'Sample-specific normalization'. Esta tabela dá o nome que o profissional
# já conhece.
#
# O rótulo é o da tela de `metaboanalyst.ca`, copiado letra a letra, e não o nome da função do
# `MetaboAnalystR` (`SpecNorm`, `ParetoNorm`): quem trabalha na bancada usa o site, e o rótulo só
# serve se for reconhecido.
#
# Cada `rotulo` é conferido por `tests/test_equivalencia_metaboanalyst.py` contra
# `docs/metodos-metaboanalyst.md`, que traz a fonte e a data de acesso; rótulo ausente de lá
# reprova. A conferência normaliza espaço em branco dos dois lados, porque a transcrição quebra
# linha no meio de frases e uma comparação crua daria falso negativo.
#
# Todo valor dos cinco vocabulários está em `EQUIVALENCIA_NO_METABOANALYST` ou em
# `SEM_CONTRAPARTE_TRANSCRITA`, nunca nos dois e nunca em nenhum. Um terceiro balde implícito,
# 'ainda não olhei', seria lido como pronto.


@dataclass(frozen=True)
class NoMetaboAnalyst:
    """O mesmo controle, como o MetaboAnalyst o chama na tela dele.

    `grupo` e `rotulo` são separados porque a tela deles agrupa, e o grupo é metade do sentido: o
    rótulo do `log2` é só `(base 2)`, que sob 'Data transformation' diz tudo. Concatená-los
    produziria uma string que não existe na tela deles.

    `conta` é `identica` quando a aritmética é a mesma e `difere` quando o método é o mesmo com
    outro parâmetro; aí `nota` diz qual. Quem compara número com número precisa saber, por
    exemplo, que o desvio deles divide por `n - 1` e o nosso por `n`.
    """

    tela: str
    grupo: str
    rotulo: str
    conta: Literal["identica", "difere"]
    nota: str = ""

    def __post_init__(self) -> None:
        """Recusa `difere` sem `nota`, que prometeria uma equivalência que não existe.

        A nota é opcional para `identica`, em que não há o que dizer, e obrigatória para `difere`,
        em que o silêncio é a afirmação errada.
        """
        if self.conta == "difere" and not self.nota.strip():
            raise ValueError(
                f"`{self.rotulo}` declara `conta='difere'` e não diz o que difere. Rótulo que "
                "aponta para o MetaboAnalyst sem declarar a divergência promete uma equivalência "
                "que a medição desmente — e quem compara número com número não tem como saber."
            )


_PROC_MISSING: Final = "ProcMissing.xhtml"
_NORMALIZATION: Final = "NormalizationView.xhtml"
_FILTER: Final = "FilterView.xhtml"

# A nota do KNN, escrita uma vez: as duas entradas partilham a divergência menos a palavra final, e
# duas redações do mesmo fato divergiriam na primeira edição. A nota dá a regra e não um exemplo
# numérico, que chegaria à tela sem a população declarada.
_NOTA_DO_KNN: Final = (
    "mesma ideia, vizinhança diferente: o k deles é **10 fixo**, e o nosso é a raiz do número de "
)

# A nota do `n - 1`, escrita uma vez: `auto` e `pareto` partilham exatamente a divergência.
_NOTA_DO_DESVIO: Final = (
    "o desvio deles divide por **n - 1** e o nosso por **n** — medido em **0,829%** de diferença "
    "com 61 amostras, e mantido por decisão"
)

# Os cinco eixos de vocabulário, com as chaves de `EIXOS_DA_GRADE`. É dicionário de dicionários
# porque `none`, `mean` e `median` valem em mais de um eixo, e uma tabela achatada poria o rótulo
# do `median` da imputação sob a redução por posto.
EQUIVALENCIA_NO_METABOANALYST: Final[dict[str, dict[str, NoMetaboAnalyst]]] = {
    # ------------------------------------------------------------------
    # Imputação: a tela `ProcMissing.xhtml`, dez botões de rádio em três grupos
    # ------------------------------------------------------------------
    "imputation": {
        "one_fifth_min": NoMetaboAnalyst(
            tela=_PROC_MISSING,
            grupo="Left-censored data estimation",
            rotulo="LoD (1/5 of the min. positive value)",
            conta="identica",
        ),
        "mean": NoMetaboAnalyst(
            tela=_PROC_MISSING,
            grupo="Univariate statistical methods",
            rotulo="mean",
            conta="identica",
        ),
        "median": NoMetaboAnalyst(
            tela=_PROC_MISSING,
            grupo="Univariate statistical methods",
            rotulo="median",
            conta="identica",
        ),
        # `col_min` e `one_fifth_min` medem a mesma coluna e diferem por fator 5: o `min` deles é
        # o nosso `col_min`, e o `LoD (1/5 ...)` deles é o nosso `one_fifth_min`. Trocá-los não
        # dá erro nenhum.
        "col_min": NoMetaboAnalyst(
            tela=_PROC_MISSING,
            grupo="Univariate statistical methods",
            rotulo="min",
            conta="identica",
        ),
        "knn": NoMetaboAnalyst(
            tela=_PROC_MISSING,
            grupo="Multivariate statistical methods",
            rotulo="KNN (sample-wise)",
            conta="difere",
            nota=_NOTA_DO_KNN + "**features**",
        ),
        "knn_features": NoMetaboAnalyst(
            tela=_PROC_MISSING,
            grupo="Multivariate statistical methods",
            rotulo="KNN (feature-wise)",
            conta="difere",
            nota=_NOTA_DO_KNN + "**amostras**",
        ),
        # Grupo vazio de propósito: o `missForest` não é um dos dez botões de rádio, aparece só no
        # texto da tela, e dar-lhe um dos três grupos seria inventar onde ele está.
        "missforest": NoMetaboAnalyst(
            tela=_PROC_MISSING,
            grupo="",
            rotulo="missForest",
            conta="difere",
            nota=(
                "eles declaram teto de **30.000 ausentes** para este método, pelo custo de "
                "computação; o nosso não tem teto declarado"
            ),
        ),
    },
    # ------------------------------------------------------------------
    # Normalização por amostra: `NormalizationView.xhtml`. A bijeção é completa:
    # os oito rótulos do grupo deles são os nossos oito valores, um a um.
    # ------------------------------------------------------------------
    "sample_normalization": {
        "none": NoMetaboAnalyst(
            tela=_NORMALIZATION,
            grupo="Sample normalization",
            rotulo="None",
            conta="identica",
        ),
        "supplied_factors": NoMetaboAnalyst(
            tela=_NORMALIZATION,
            grupo="Sample normalization",
            rotulo="Sample-specific normalization (i.e. weight, volume)",
            conta="identica",
        ),
        # A conta deles traz o fator 1000, `1000*x/sum(x)`, e a nossa também.
        "sum": NoMetaboAnalyst(
            tela=_NORMALIZATION,
            grupo="Sample normalization",
            rotulo="Normalization by sum",
            conta="identica",
        ),
        "median": NoMetaboAnalyst(
            tela=_NORMALIZATION,
            grupo="Sample normalization",
            rotulo="Normalization by median",
            conta="identica",
        ),
        "sample_reference": NoMetaboAnalyst(
            tela=_NORMALIZATION,
            grupo="Sample normalization",
            rotulo="Normalization by a reference sample (PQN)",
            conta="identica",
        ),
        # O nosso grupo de referência são os QCs, fixo; na tela deles escolhe-se o grupo. A conta
        # é a mesma e só falta o seletor, por isso `identica` e não `difere`.
        "pqn_group": NoMetaboAnalyst(
            tela=_NORMALIZATION,
            grupo="Sample normalization",
            rotulo="Normalization by a pooled sample from group (group PQN)",
            conta="identica",
        ),
        "feature_reference": NoMetaboAnalyst(
            tela=_NORMALIZATION,
            grupo="Sample normalization",
            rotulo="Normalization by reference feature",
            conta="identica",
        ),
        "quantile": NoMetaboAnalyst(
            tela=_NORMALIZATION,
            grupo="Sample normalization",
            rotulo="Quantile normalization (suggested only for > 1000 features)",
            conta="identica",
        ),
    },
    # ------------------------------------------------------------------
    # Transformação: mesma tela, grupo `Data transformation`
    # ------------------------------------------------------------------
    "transformation": {
        "log10": NoMetaboAnalyst(
            tela=_NORMALIZATION,
            grupo="Data transformation",
            rotulo="Log transformation (base 10)",
            conta="identica",
        ),
        # `(base 2)` sozinho é o que está na tela deles: a opção vem logo abaixo da de base 10 e
        # herda dela o começo da frase.
        "log2": NoMetaboAnalyst(
            tela=_NORMALIZATION,
            grupo="Data transformation",
            rotulo="(base 2)",
            conta="identica",
        ),
        "sqrt": NoMetaboAnalyst(
            tela=_NORMALIZATION,
            grupo="Data transformation",
            rotulo="Square root transformation (square root of data values)",
            conta="identica",
        ),
        "cbrt": NoMetaboAnalyst(
            tela=_NORMALIZATION,
            grupo="Data transformation",
            rotulo="Cube root transformation (cube root of data values)",
            conta="identica",
        ),
    },
    # ------------------------------------------------------------------
    # Escalonamento: mesma tela, grupo `Data scaling`
    # ------------------------------------------------------------------
    "scaling": {
        "mean_center": NoMetaboAnalyst(
            tela=_NORMALIZATION,
            grupo="Data scaling",
            rotulo="Mean centering (mean-centered only)",
            conta="identica",
        ),
        "auto": NoMetaboAnalyst(
            tela=_NORMALIZATION,
            grupo="Data scaling",
            rotulo=(
                "Auto scaling (mean-centered and divided by the standard deviation of each "
                "variable)"
            ),
            conta="difere",
            nota=_NOTA_DO_DESVIO,
        ),
        "pareto": NoMetaboAnalyst(
            tela=_NORMALIZATION,
            grupo="Data scaling",
            rotulo=(
                "Pareto scaling (mean-centered and divided by the square root of the standard "
                "deviation of each variable)"
            ),
            conta="difere",
            nota=_NOTA_DO_DESVIO,
        ),
        "range": NoMetaboAnalyst(
            tela=_NORMALIZATION,
            grupo="Data scaling",
            rotulo="Range scaling (mean-centered and divided by the range of each variable)",
            conta="identica",
        ),
    },
}

# Os dois eixos de redução por posto partilham vocabulário e contraparte. Na tela deles,
# 'Low-variance' e 'Low-abundance' são dois filtros em sequência, cada um com o seu percentil, e é
# por isso que temos dois eixos de redução e não um. O dicionário é um só para não haver duas
# cópias a divergir.
_REDUCAO_POR_POSTO: Final[dict[str, NoMetaboAnalyst]] = {
    # `Interquantile` é a grafia deles e fica, para a citação continuar conferível contra a tela;
    # a transcrição marca-a com `[SIC]`.
    "iqr": NoMetaboAnalyst(
        tela=_FILTER,
        grupo="Low-variance filter",
        rotulo="Interquantile range (IQR)",
        conta="identica",
    ),
    "sd": NoMetaboAnalyst(
        tela=_FILTER,
        grupo="Low-variance filter",
        rotulo="Standard deviation (SD)",
        conta="identica",
    ),
    "mad": NoMetaboAnalyst(
        tela=_FILTER,
        grupo="Low-variance filter",
        rotulo="Median absolute deviation (MAD)",
        conta="identica",
    ),
    "rsd": NoMetaboAnalyst(
        tela=_FILTER,
        grupo="Low-variance filter",
        rotulo="Relative standard deviation (RSD = SD/mean)",
        conta="identica",
    ),
    "nrsd": NoMetaboAnalyst(
        tela=_FILTER,
        grupo="Low-variance filter",
        rotulo="Non-parametric relative standard deviation (MAD/median)",
        conta="identica",
    ),
    "mean": NoMetaboAnalyst(
        tela=_FILTER,
        grupo="Low-abundance filter",
        rotulo="Mean intensity value",
        conta="identica",
    ),
    "median": NoMetaboAnalyst(
        tela=_FILTER,
        grupo="Low-abundance filter",
        rotulo="Median intensity value",
        conta="identica",
    ),
}

EQUIVALENCIA_NO_METABOANALYST["ranking_filter"] = _REDUCAO_POR_POSTO
EQUIVALENCIA_NO_METABOANALYST["ranking_filter_2"] = _REDUCAO_POR_POSTO


# O valor de repouso não ganha linha de equivalência na tela. Cinco dos seis eixos repousam em
# `none`, e cinco frases dizendo que 'não fazer nada' não tem tradução empurrariam para baixo o que
# a pessoa procura. É uma regra sobre o valor, e não um campo por entrada, porque o `none` da
# normalização por amostra tem contraparte transcrita ('None') e diria algo verdadeiro e inútil.
# Calar é decisão da tela: o motivo de cada ausência continua escrito abaixo, e a partição continua
# exaustiva.
VALOR_DE_REPOUSO: Final = "none"

# O outro lado da partição: por que cada valor não recebe nome do MetaboAnalyst na tela. As razões
# têm naturezas diferentes e pedem ações diferentes: 'não há o que traduzir', 'é nosso' e 'a
# transcrição não tem rótulo conferível'. Só a última se fecha com uma transcrição nova.
SEM_CONTRAPARTE_TRANSCRITA: Final[dict[str, dict[str, str]]] = {
    "imputation": {
        "none": "«não imputar» não precisa de tradução",
        # `preprocessing/missing.py` chama-lhe o `exclude` do MetaboAnalyst, mas a transcrição de
        # `ProcMissing.xhtml` não traz rótulo para ele: o descarte por ausentes vive noutro
        # controle da tela deles. Sem rótulo conferível, não se põe nome na tela.
        "exclude": (
            "o descarte existe lá, e a transcrição da tela não traz rótulo conferível para ele"
        ),
    },
    "sample_normalization": {},
    "transformation": {"none": "«não transformar» não precisa de tradução"},
    "scaling": {
        "none": "«não escalonar» não precisa de tradução",
        # `vast` é método de verdade, então fala: quem usa o MetaboAnalyst procura o equivalente e
        # precisa saber que não há. Silêncio aqui se leria como 'não carregou'.
        "vast": "é NOSSO; a transcrição declara que não tem contraparte lá",
    },
    "ranking_filter": {"none": "«não reduzir» não precisa de tradução"},
    "ranking_filter_2": {"none": "«não reduzir» não precisa de tradução"},
}


def equivalencia_do_eixo(eixo: str, valor: str) -> NoMetaboAnalyst | None:
    """A contraparte do valor escolhido, ou `None` quando não há uma transcrita.

    `None` significa só isso. O motivo está em `SEM_CONTRAPARTE_TRANSCRITA`, e o chamador decide
    se o mostra; a janela mostra, porque silêncio sob um rótulo que os vizinhos têm se leria como
    'não carregou'.
    """
    return EQUIVALENCIA_NO_METABOANALYST.get(eixo, {}).get(valor)


def prosa_da_equivalencia(eixo: str, valor: str) -> str:
    """A linha, em markdown, que a interface mostra sob o eixo.

    O texto nasce aqui e não é reescrito no ponto de uso, para haver uma redação só. O rótulo
    deles vai entre crases e em inglês: o que se quer é que a pessoa reconheça a string que já leu
    na tela do MetaboAnalyst, e uma tradução não seria reconhecível.

    Devolve string vazia para o `VALOR_DE_REPOUSO` e para eixo que não é de vocabulário.
    """
    if valor == VALOR_DE_REPOUSO:
        return ""
    equivalente = equivalencia_do_eixo(eixo, valor)
    if equivalente is None:
        motivo = SEM_CONTRAPARTE_TRANSCRITA.get(eixo, {}).get(valor, "")
        if not motivo:
            return ""
        return f"No MetaboAnalyst: sem equivalente — {motivo}."
    onde = f"*{equivalente.grupo}* → " if equivalente.grupo else ""
    linha = f"No MetaboAnalyst: {onde}`{equivalente.rotulo}`"
    if equivalente.conta == "identica":
        return f"{linha} — a conta é a mesma."
    return f"{linha} — **atenção:** {equivalente.nota}."


@dataclass(frozen=True)
class PassoDoFluxo:
    """Um passo que a interface executa, com a rota que o serve.

    Ligar cada passo a uma rota da aplicação é o que torna o fluxo executável, e um teste reprova
    passo cuja rota não existe.

    `titulo` é o nome do passo e o texto do botão de download; `rotulo_de_acao` é o do botão que
    dispara o cálculo, para que 'Baixar' apareça só onde há bytes. Fica vazio, exceto nos passos
    que produzem arquivo, os únicos com o par disparo e download.
    """

    chave: str
    titulo: str
    rota: str
    metodo: str
    rotulo_de_acao: str = ""


@dataclass(frozen=True)
class PontoDeDecisao:
    """Um ponto em que o sistema decide automaticamente.

    Em todo ponto de decisão automática a interface exibe a justificativa e oferece sobrescrita,
    por isso os dois campos são obrigatórios juntos. Sem controle, o usuário ficaria informado e
    sem ação; sem justificativa, com ação e sem base.
    """

    chave: str
    o_que_decide: str
    campo_da_justificativa: str
    eixos_sobrescreviveis: tuple[str, ...]


@dataclass(frozen=True)
class ControleDaTela:
    """Um controle que a tela expõe e o eixo de sobrescrita que ele move.

    Esta tabela, e não `SUPPORTED_OVERRIDES`, é a fonte dos eixos oferecidos para sobrescrita: um
    eixo que a especificação suporta e a tela não deixa mover gravaria par de concordância sobre
    algo que o sistema ignora. Um controle novo entra aqui e o eixo aparece no ponto sozinho.

    - `chave_do_widget`: a chave com que a tela o desenha e com que o teste o encontra.
    - `eixo`: o eixo de `SUPPORTED_OVERRIDES` que ele move.
    - `ponto`: a decisão automática de que ele é a sobrescrita.
    - `onde`: a seção que o desenha.

    `chave_do_widget` não é o nome do eixo: o seletor do corretor tem chave `correcao` e move
    `correction`. Derivar um do outro exigiria que os dois vocabulários coincidissem.
    """

    chave_do_widget: str
    eixo: str
    ponto: str
    onde: Secao


@dataclass(frozen=True)
class MetricaTraduzida:
    """O que uma das onze métricas da tela de qualidade diz, em português.

    O leitor conhece LC-MS e não este repositório: sabe o que é RSD, QC e deriva, mas não a nossa
    grafia `frac_features_under_20`. O nome técnico não é campo daqui porque cada interface o
    mostra ao lado; é com ele que o profissional fala conosco e acha a métrica no documento.

    - `rotulo`: como o profissional chama a quantidade.
    - `unidade`: em que escala o número vem.
    - `formato`: em que unidade o número se escreve na tela.
    - `populacao`: sobre que conjunto ele foi calculado.
    - `tema`: ao lado de que outras métricas ela se lê.
    - `sentido`: menor é melhor, maior é melhor, ou nenhum dos dois.
    - `ressalva`: o que faz o número ser mal lido, quando há.
    - `por_que_sem_caminho`: por que nenhuma rota o produz, emitido só quando é o caso.
    - `referencia`: a âncora da área, só onde a fonte foi lida.

    `formato`, `populacao` e `tema` não têm padrão: uma métrica nova não se constrói sem os
    declarar, e um padrão escolheria a política por omissão. `referencia` tem padrão vazio porque
    a ausência de âncora é o estado conservador, e a tela não inventa referência.

    `unidade` importa porque o RSD deste projeto é fração (0,25 = 25%). Ler 0,155 como pontos
    percentuais muda a cascata do diagnóstico e a classe atribuída a um modo inteiro.

    `por_que_sem_caminho` só sai quando `CAMINHO_DA_METRICA` diz que não há caminho. Um texto fixo
    continuaria a negar o caminho que uma rota ganhasse.
    """

    rotulo: str
    unidade: str
    formato: FormatoDoValor
    populacao: PopulacaoDaMetrica
    tema: TemaDaMetrica
    sentido: str
    ressalva: str = ""
    por_que_sem_caminho: str = ""
    referencia: str = ""


@dataclass(frozen=True)
class TemaDeQualidade:
    """O cabeçalho sob o qual um conjunto de métricas se lê.

    O título é o que o profissional lê e a chave é o que o código compara, então a redação muda
    sem tocar na partição. Todo tema tem `nota`, que diz o que o grupo tem em comum. Quem a exige é
    o residual: uma contagem que não é nota de qualidade, sob um cabeçalho de qualidade, seria lida
    como nota por vizinhança; e se só ele tivesse nota, pareceria a exceção.
    """

    titulo: str
    nota: str


@dataclass(frozen=True)
class GrupoDeMetricas:
    """Um tema já resolvido para desenho: cabeçalho, nota e os nomes que vão debaixo dele.

    As interfaces recebem isto pronto e não repartem nada, para que a partição não seja
    redigitada em cada painel. `chave` viaja junto do `titulo` para que os testes afirmem a
    partição e não a redação.
    """

    chave: str
    titulo: str
    nota: str
    metricas: tuple[str, ...]


# ---------------------------------------------------------------------------
# Os passos do fluxo
# ---------------------------------------------------------------------------

DENTRO_DO_FLUXO: Final[tuple[PassoDoFluxo, ...]] = (
    PassoDoFluxo("validacao", "Ingestão e problemas de validação", "/matrizes/validar", "POST"),
    PassoDoFluxo(
        "diagnostico",
        "Diagnóstico, com os pontos de mudança sugeridos",
        "/matrizes/diagnosticar",
        "POST",
    ),
    PassoDoFluxo(
        "recomendacao",
        "Recomendação, com a justificativa exibida",
        "/matrizes/recomendar",
        "POST",
    ),
    PassoDoFluxo("execucao", "Pipeline com parâmetros explícitos", "/execucoes", "POST"),
    PassoDoFluxo(
        "multiverso", "Teste de robustez, com progresso incremental", "/multiversos", "POST"
    ),
    PassoDoFluxo(
        "progresso", "Acompanhamento do teste de robustez", "/multiversos/{identificador}", "GET"
    ),
    # Os três passos que produzem arquivo. Declarados aqui, a tela não oferece operação que o
    # fluxo não nomeia, e o teste confere que a rota de cada um existe na aplicação. O primeiro
    # botão de cada um dispara o cálculo e devolve um identificador de trabalho, por isso o
    # `rotulo_de_acao` diz 'Gerar' e só o download diz 'Baixar'.
    PassoDoFluxo(
        "corrigir",
        "Baixar a matriz corrigida",
        "/matrizes/corrigir",
        "POST",
        rotulo_de_acao="Gerar a matriz corrigida",
    ),
    PassoDoFluxo(
        "processar",
        "Baixar a matriz processada",
        "/matrizes/processar",
        "POST",
        rotulo_de_acao="Gerar a matriz processada",
    ),
    PassoDoFluxo(
        "exportar",
        "Baixar no formato MetaboAnalyst",
        "/matrizes/exportar",
        "POST",
        rotulo_de_acao="Gerar o arquivo no formato MetaboAnalyst",
    ),
    # Do mesmo gênero que `progresso`: as operações longas devolvem `202` com identificador, e o
    # resultado é reclamado por esta rota, que também precisa estar no fluxo nomeado.
    PassoDoFluxo(
        "reclamacao", "Reclamar o resultado por identificador", "/trabalhos/{identificador}", "GET"
    ),
)

# As chaves dos passos que produzem arquivo. A janela tira os passos daqui, em vez de repetir
# chave, rota e rótulo numa segunda tabela que divergiria quando uma rota fosse renomeada.
CHAVES_DE_ARTEFATO: Final[tuple[str, ...]] = ("corrigir", "processar", "exportar")


# ---------------------------------------------------------------------------
# Os pontos de decisão automática
# ---------------------------------------------------------------------------
#
# A enumeração sai das rotas: cada chave é a de um passo de `DENTRO_DO_FLUXO`, e um teste reprova
# ponto que cite rota inexistente.
#
# São dois. Além da recomendação, o diagnóstico decide a classe do defeito e sugere as fronteiras
# de lote, e essa sugestão pede confirmação humana; tratá-la como informação a deixaria sem
# justificativa exibida.

# Os controles que a tela expõe. Dar controle a um eixo novo é uma linha aqui, e o eixo passa a ser
# oferecido no ponto sem mais nada.
#
# `missing_threshold` não tem padrão do servidor: `None` faz `process_matrix` saltar
# `drop_by_group_missingness`, como o corte de RSD.
#
# `quem_escolhe` não entra. Esta tabela liga controle a eixo de sobrescrita, e sobrescrever é o
# humano discordar da máquina; delegar é o oposto. O eixo que a delegação move é `correction`, que
# já tem o seletor `correcao`, e uma segunda linha ofereceria o mesmo eixo duas vezes e gravaria
# concordância sobre um gesto que não é discordância.
CONTROLES_DE_SOBRESCRITA: Final[tuple[ControleDaTela, ...]] = (
    ControleDaTela(
        chave_do_widget="correcao", eixo="correction", ponto="recomendacao", onde="execucao"
    ),
    ControleDaTela(
        chave_do_widget="rsd_cutoff", eixo="rsd_cutoff", ponto="recomendacao", onde="execucao"
    ),
    # O único controle do ponto do diagnóstico. Todo ponto exige justificativa e controle, e
    # `eixos_do_ponto` levanta sobre lista vazia para que um ponto sem controle não seja possível.
    ControleDaTela(
        chave_do_widget="n_batches", eixo="n_batches", ponto="diagnostico", onde="decisao"
    ),
    # Os eixos do pipeline têm `ponto` `recomendacao`, e não `diagnostico`: vivem na seção do
    # pipeline, ao lado do corretor e do corte de RSD, onde a proposta da máquina é comparada com a
    # escolha humana. Cada combo que a tela desenha tem a sua linha aqui, senão o eixo seria
    # desenhado e não oferecido para sobrescrita.
    ControleDaTela(
        chave_do_widget="eixo_ranking_filter",
        eixo="ranking_filter",
        ponto="recomendacao",
        onde="execucao",
    ),
    # Os três do filtro em dois eixos. As frações retidas entram pelo mesmo motivo de
    # `rsd_cutoff` e `missing_threshold`: são escolha metodológica, e discordar delas nomeia o eixo.
    ControleDaTela(
        chave_do_widget="ranking_keep_fraction",
        eixo="ranking_keep_fraction",
        ponto="recomendacao",
        onde="execucao",
    ),
    ControleDaTela(
        chave_do_widget="eixo_ranking_filter_2",
        eixo="ranking_filter_2",
        ponto="recomendacao",
        onde="execucao",
    ),
    ControleDaTela(
        chave_do_widget="ranking_keep_fraction_2",
        eixo="ranking_keep_fraction_2",
        ponto="recomendacao",
        onde="execucao",
    ),
    # A referência da normalização é escolha metodológica forte: trocar a amostra contra a qual
    # tudo é normalizado muda cada célula da matriz.
    ControleDaTela(
        chave_do_widget="normalization_reference",
        eixo="normalization_reference",
        ponto="recomendacao",
        onde="execucao",
    ),
    # Os fatores fornecidos não entram: sobrescrever é discordar da máquina, e ela nunca propõe
    # fatores, que são medidos fora do equipamento. O seletor de arquivo envia-os, mas não há
    # discordância a registrar.
    ControleDaTela(
        chave_do_widget="eixo_imputation",
        eixo="imputation",
        ponto="recomendacao",
        onde="execucao",
    ),
    ControleDaTela(
        chave_do_widget="eixo_sample_normalization",
        eixo="sample_normalization",
        ponto="recomendacao",
        onde="execucao",
    ),
    ControleDaTela(
        chave_do_widget="eixo_transformation",
        eixo="transformation",
        ponto="recomendacao",
        onde="execucao",
    ),
    ControleDaTela(
        chave_do_widget="eixo_scaling", eixo="scaling", ponto="recomendacao", onde="execucao"
    ),
    ControleDaTela(
        chave_do_widget="missing_threshold",
        eixo="missing_threshold",
        ponto="recomendacao",
        onde="execucao",
    ),
    ControleDaTela(
        chave_do_widget="rsd_variant", eixo="rsd_variant", ponto="recomendacao", onde="execucao"
    ),
)


# A tradução dos rótulos de defeito. `erratic` e `gradual_drift` são identificadores de código, e
# quem opera o equipamento não tem por que os conhecer.
#
# `unassessable` não é um quinto tipo de defeito: é a ausência de medida. Traduzi-lo como
# 'errático' ou 'instável' misturaria 'medi e está ruim' com 'não pude medir', que levam a ações
# opostas.
#
# `clean` também não é elogio. Diz que a cascata não achou padrão, o que é diferente de 'o dado
# está bom': com poucos QCs não se exclui deriva pequena.
#
# Um teste confere a bijeção contra `get_args(DefectType)` nas duas direções: tipo novo sem
# tradução reprova em vez de aparecer cru na tela, e tradução órfã também.
ROTULOS_DE_DEFEITO: Final[dict[str, str]] = {
    "gradual_drift": "deriva gradual — o sinal escorrega ao longo da corrida",
    "step_change": "degrau — o sinal muda de nível e fica no nível novo",
    "erratic": "errático — varia sem direção que a cascata reconheça",
    "clean": "sem padrão detectado — o que **não** é o mesmo que «o dado está bom»",
    "unassessable": (
        "**não avaliável** — a métrica estrutural ficou indefinida ou instável, então não houve "
        "medição. Não é um defeito: é a ausência de uma"
    ),
}


# ---------------------------------------------------------------------------
# O modo deduzido
# ---------------------------------------------------------------------------
#
# O modo não escolhe limiar nenhum: rotula as saídas e alimenta o aviso `MODE_CONTENT_MISMATCH`.
# Por isso o campo fica em 'Opções avançadas', preenchido por dedução, e a tela diz de onde o valor
# veio. Sem dedução o campo fica vazio, porque um padrão silencioso rotularia um estudo `neg_c18`
# como o primeiro modo da lista.
#
# As duas fontes, os nomes `measurements_{modo}.csv` e os tokens `_POS_`, `_NEG_`, `HILIC` e `C18`
# nos nomes de amostra, são costume dos arquivos deste laboratório. Por isso o que sai daqui é
# sugestão visível, que a pessoa troca.
ORIGEM_NO_NOME_DO_ARQUIVO: Final = "do nome do arquivo"
ORIGEM_NOS_NOMES_DAS_AMOSTRAS: Final = "dos nomes das amostras"
ORIGEM_ESCOLHIDA_A_MAO: Final = "escolhido à mão"

_MODO_NO_NOME: Final = re.compile(r"(?<![a-z])(pos|neg)_(hilic|c18)(?![a-z0-9])", re.IGNORECASE)


def modo_do_nome_do_arquivo(nome: str) -> str | None:
    """O modo que o nome do arquivo traz, como `measurements_pos_hilic.csv`, ou `None`."""
    achado = _MODO_NO_NOME.search(nome)
    if achado is None:
        return None
    return f"{achado.group(1).lower()}_{achado.group(2).lower()}"


def prosa_do_modo(modo: str, origem: str | None) -> str:
    """A linha que a tela mostra: o modo e de onde ele veio, ou o que falta."""
    if not modo:
        return (
            "Modo: não deduzido do nome do arquivo nem dos nomes das amostras. Escolha-o em "
            "«Opções avançadas»."
        )
    return f"Modo: {modo} ({origem})" if origem else f"Modo: {modo}"


# ---------------------------------------------------------------------------
# Como interpretar o comportamento do sinal analítico, no texto do laboratório
# ---------------------------------------------------------------------------
#
# O texto é do laboratório, transcrito só com correção tipográfica: as aspas da introdução saíram,
# os dois-pontos de 'Exemplo prático' viraram título de campo, e 'REFino' virou 'Refino', o nome
# que a janela usa.
#
# É constante própria para que `ROTULOS_DE_DEFEITO`, que alimenta a linha curta do diagnóstico
# (`prosa_do_defeito`), fique intacta: a curta diz o que o diagnóstico achou, e esta explica como
# ler a classe.
#
# Os campos de cada classe são os que o laboratório escreveu, na ordem dele: 'Sem padrão
# detectado' não tem exemplo, e 'Não avaliável' pergunta 'Por que isso pode acontecer?'. Um molde
# fixo inventaria texto que ele não escreveu ou apagaria o que escreveu.
#
# Um teste confere a bijeção com `DefectType` nas duas direções, como a de `ROTULOS_DE_DEFEITO`.
TITULO_DA_INTERPRETACAO_DO_SINAL: Final = "Como interpretar o comportamento do sinal analítico"

INTRODUCAO_DA_INTERPRETACAO_DO_SINAL: Final[tuple[str, ...]] = (
    "Durante uma análise por LC-MS, a intensidade do sinal de um mesmo composto pode variar ao "
    "longo da sequência de injeções, mesmo quando analisamos repetidamente uma mesma amostra de "
    "controle de qualidade (QC).",
    "O Refino utiliza as informações desses controles para investigar possíveis padrões de "
    "variação instrumental.",
    "Os diagnósticos apresentados abaixo ajudam a compreender como o sinal se comportou durante a "
    "corrida, mas não determinam, isoladamente, a qualidade dos dados ou a necessidade de "
    "correção.",
)


@dataclass(frozen=True)
class InterpretacaoDoSinal:
    """Uma classe do diagnóstico explicada como o laboratório a escreveu.

    `campos` é a sequência de pares `(pergunta, resposta)` na ordem do texto dele.
    """

    nome: str
    resumo: str
    campos: tuple[tuple[str, str], ...]


INTERPRETACAO_DO_SINAL: Final[dict[str, InterpretacaoDoSinal]] = {
    "gradual_drift": InterpretacaoDoSinal(
        nome="Deriva gradual",
        resumo="O sinal muda progressivamente",
        campos=(
            (
                "O que significa?",
                "A intensidade do sinal aumenta ou diminui gradualmente ao longo da sequência "
                "analítica.",
            ),
            (
                "Exemplo prático",
                "Imagine que a mesma amostra de QC seja analisada várias vezes. Nas primeiras "
                "injeções, um composto apresenta intensidade de 1.000; posteriormente, 900, 800 e "
                "700. Isso indica uma tendência de redução do sinal ao longo da corrida.",
            ),
            (
                "O que pode estar acontecendo?",
                "Esse comportamento pode estar relacionado a mudanças progressivas na resposta "
                "instrumental, como alterações na eficiência de ionização ou na sensibilidade do "
                "sistema.",
            ),
            (
                "O que fazer?",
                "Investigar a tendência temporal e avaliar se um método de correção de deriva pode "
                "reduzir essa variação sem comprometer os sinais biológicos.",
            ),
        ),
    ),
    "step_change": InterpretacaoDoSinal(
        nome="Degrau",
        resumo="O sinal muda repentinamente",
        campos=(
            (
                "O que significa?",
                "O sinal permanece relativamente estável durante parte da corrida, sofre uma "
                "mudança abrupta e passa a apresentar intensidades em outro nível.",
            ),
            (
                "Exemplo prático",
                "Uma feature apresenta intensidade próxima de 1.000 nos primeiros QCs. A partir de "
                "determinado momento, os valores passam para aproximadamente 600 e permanecem "
                "próximos desse novo nível.",
            ),
            (
                "O que pode estar acontecendo?",
                "Pode haver uma alteração instrumental ou operacional durante a sequência, como "
                "uma intervenção no equipamento, uma interrupção ou uma mudança nas condições "
                "analíticas.",
            ),
            (
                "O que fazer?",
                "Investigar o momento da mudança e verificar os registros da sequência analítica "
                "antes de aplicar uma correção.",
            ),
        ),
    ),
    "erratic": InterpretacaoDoSinal(
        nome="Errático",
        resumo="O sinal apresenta oscilações irregulares",
        campos=(
            (
                "O que significa?",
                "A intensidade varia de maneira irregular, sem apresentar uma tendência temporal "
                "bem definida.",
            ),
            (
                "Exemplo prático",
                "O mesmo composto apresenta intensidades de 1.000, 500, 1.200, 700 e 1.100 em "
                "injeções sucessivas de QC.",
            ),
            (
                "O que pode estar acontecendo?",
                "Esse comportamento pode indicar instabilidade analítica, problemas de integração "
                "cromatográfica, variações de ionização ou outros fatores técnicos.",
            ),
            (
                "O que fazer?",
                "Avaliar a variabilidade nos QCs, a qualidade dos picos cromatográficos e a "
                "presença de valores discrepantes. Uma correção de deriva temporal não "
                "necessariamente resolverá oscilações aleatórias.",
            ),
        ),
    ),
    "clean": InterpretacaoDoSinal(
        nome="Sem padrão detectado",
        resumo="Nenhuma tendência característica foi identificada",
        campos=(
            (
                "O que significa?",
                "O programa não identificou evidências suficientes para classificar a variação do "
                "sinal como gradual, abrupta ou errática.",
            ),
            (
                "Isso significa que os dados estão bons?",
                "Não necessariamente. A ausência de um padrão detectável não garante estabilidade, "
                "precisão ou qualidade analítica.",
            ),
            (
                "O que fazer?",
                "Avaliar outras métricas, especialmente o RSD dos controles de qualidade, a "
                "distribuição das intensidades e a presença de valores ausentes.",
            ),
        ),
    ),
    "unassessable": InterpretacaoDoSinal(
        nome="Não avaliável",
        resumo="Não foi possível realizar uma classificação confiável",
        campos=(
            (
                "O que significa?",
                "As informações disponíveis não permitiram calcular uma ou mais métricas "
                "necessárias para avaliar o comportamento do sinal.",
            ),
            (
                "Por que isso pode acontecer?",
                "Pode ocorrer quando existem dados insuficientes, muitos valores ausentes, "
                "intensidades inadequadas para determinados cálculos ou condições que tornam as "
                "métricas indefinidas.",
            ),
            (
                "O que fazer?",
                "Verificar a integridade dos arquivos, a identificação dos QCs e a disponibilidade "
                "de dados suficientes para realizar a avaliação.",
            ),
        ),
    ),
}


def classes_sem_interpretacao(
    mapa: dict[str, InterpretacaoDoSinal] | None = None,
) -> tuple[frozenset[str], frozenset[str]]:
    """As duas direções da bijeção com `DefectType`: `(sem texto, texto órfão)`.

    Devolve as diferenças em vez de um booleano porque é o nome da classe que diz a quem conserta o
    que falta; `mapa` entra por parâmetro para o teste plantar um mapa sem uma classe.
    """
    usado = INTERPRETACAO_DO_SINAL if mapa is None else mapa
    tipos = frozenset(get_args(DefectType))
    return tipos - frozenset(usado), frozenset(usado) - tipos


# ---------------------------------------------------------------------------
# A estimativa de duração da medição de corretor
# ---------------------------------------------------------------------------
#
# Custo do estimador leave-one-QC-out inteiro (os `n_qc` ajustes) por feature amostrada, em ms,
# medido com 12 núcleos lógicos, `variant="outlier_removed"`, 7 QCs e 68 amostras, no `neg_c18` e
# no `pos_hilic`.
#
# A estimativa multiplica estes valores pelo número de features, e duas propriedades medidas a
# sustentam:
# - o custo é linear em `n_sampled_features` acima de cerca de 200 features; abaixo disso o custo
#   fixo domina;
# - o custo por feature quase não depende do tamanho da matriz: o `pos_hilic` tem 2,8x mais
#   features que o `neg_c18`, e o LOESS custa menos por feature nele.
# O maior ponto medido tem 200 features amostradas e a estimativa vai até 6.035, então é
# extrapolação, e a segunda propriedade é o único apoio dela.
#
# Os valores vêm do `pos_hilic`, o modo maior. No `serrf`, que domina o total, ele é também o mais
# caro por feature, e a estimativa erra para cima, que é a preferência: estimativa curta faz o
# profissional achar que travou. No `loess` o valor fica até 17% abaixo do medido no `neg_c18`.
CUSTO_POR_FEATURE_MS: Final[dict[str, float]] = {
    "none": 0.08,
    "loess": 39.0,
    "serrf": 362.6,
}

# Quanto o `serrf` fica mais lento em série do que em paralelo. Sai de quatro pares série e
# paralelo, um por modo, medidos no mesmo executável e na mesma sessão. O fator vai de 5,45x a
# 5,74x entre os modos, e fica o maior, para a estimativa errar para cima; o custo em série que ele
# implica erra para cima nos quatro modos, entre 5% e 7%.
#
# O regime depende de duas coisas: a capacidade, `regime_do_processo()`, resolvida no arranque e
# que cai para série se o interpretador embarcado faltar; e a escolha, a caixa 'Usar vários
# núcleos'. A estimativa lê as duas juntas em `PainelDePipeline._serial_agora`.
#
# Só o `serrf` recebe o fator. O `none` não corre nada. O LOESS também lê a caixa, mas o custo dele
# nesta tabela é o da série e o fator dele no executável não foi medido; com a caixa marcada, a
# estimativa do LOESS é a da série, e erra para cima.
FATOR_SERIAL_DO_SERRF: Final = 5.74

# O custo de um ajuste, em ms por feature, para `POST /execucoes`. `CUSTO_POR_FEATURE_MS` mede o
# estimador leave-one-QC-out, com `n_qc` reajustes por corretor; a execução faz um ajuste de um
# corretor, e reusar aquele número diria horas para algo que custa minutos.
#
# Medido com um ajuste por modo, sobre a matriz inteira dos quatro arquivos do laboratório.
#
# O valor do `serrf` é do regime paralelo, e `FATOR_SERIAL_DO_SERRF` entra quando `serial`, como em
# `CUSTO_POR_FEATURE_MS`; guardar o valor já multiplicado obrigaria a desfazer a conta no regime de
# processos. O do `loess` é o da série e não recebe fator, pela razão escrita em
# `FATOR_SERIAL_DO_SERRF`.
CUSTO_DE_UM_AJUSTE_MS: Final[dict[str, float]] = {
    # Abaixo da resolução do relógio nos quatro modos. O valor é simbólico, para a soma não ter
    # ramo especial.
    "none": 0.001,
    # Dos três modos medidos com a máquina livre, que concordam em 4%.
    "loess": 6.8,
    # Medido no regime paralelo, dentro do executável, nos quatro modos; fica o maior, para errar
    # para cima. O custo em série sai daqui pelo fator, e não de uma segunda entrada que
    # divergiria numa remedição.
    "serrf": 39.7,
}

# A cauda fixa: o que a rota faz depois de corrigir e que corre sempre, o pipeline e o relatório
# de qualidade, sem nenhuma família de análise pedida. Mede de 0,98 s a 2,42 s entre os quatro
# modos e é constante, não proporcional ao tamanho. É dominada pelo PLS-DA com 200 permutações,
# que só corre com par de grupos declarado, então o valor é o teto e não o típico.
# O custo de cada família pedida está em `CUSTO_DA_ANALISE_S`, e não aqui. A gravação no banco
# também não está incluída, porque depende de haver banco.
CAUDA_DA_EXECUCAO_S: Final = 2.5


# ---------------------------------------------------------------------------
# As famílias de análise
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FamiliaDeAnalise:
    """Uma família de análise pedível: o campo que a liga, o rótulo da caixa e a etapa da barra.

    O custo não é campo. 'Barata' é `familia_e_barata`, que lê o padrão de `AnalysisSpec`: as
    famílias que custam segundos vêm ligadas e as que custam minutos, desligadas. Um campo aqui
    seria uma segunda fonte de verdade sobre a mesma escolha, e é o padrão que a rota obedece
    quando o cliente não manda nada.
    """

    campo: str
    """O campo de `AnalysisSpec` que esta família liga. Bijetado contra `pediu_alguma_coisa`."""

    rotulo: str
    """O texto da caixa de seleção, na língua de quem opera, nunca o nome do campo.

    É curto porque rótulos em frase alargavam a grade de caixas a ponto de a área de rolagem
    ganhar barra horizontal. O que explica cada família fica na dica, que não ocupa espaço.
    """

    etapa: str
    """O nome da etapa na barra de progresso, no vocabulário das quatro que já existem."""


# As baratas primeiro, que é a ordem em que a tela as desenha: as que já vêm marcadas ficam em
# cima, e o bloco se lê sem contar caixas.
FAMILIAS_DE_ANALISE: Final[tuple[FamiliaDeAnalise, ...]] = (
    FamiliaDeAnalise("anova", "ANOVA de N grupos", "anova"),
    FamiliaDeAnalise("correlation", "Correlação entre features", "correlacao"),
    FamiliaDeAnalise("permanova", "PERMANOVA nos componentes", "permanova"),
    FamiliaDeAnalise("normality", "Normalidade por grupo", "normalidade"),
    FamiliaDeAnalise("plsda_multiclass", "PLS-DA multiclasse", "plsda multiclasse"),
    FamiliaDeAnalise("clustering", "Agrupamento hierárquico", "agrupamento"),
    FamiliaDeAnalise("kmeans", "K-means", "kmeans"),
    FamiliaDeAnalise("som", "Mapa auto-organizável", "som"),
    FamiliaDeAnalise("dspc", "Rede de correlação parcial", "rede"),
    FamiliaDeAnalise("classifiers", "Classificador", "classificadores"),
)

# O que cada família faz, para a dica da caixa. Fica separada do rótulo porque o rótulo ocupa
# largura e a dica não. A chave é o `campo`, e um teste confere a bijeção contra a tabela acima:
# família sem dica apareceria na tela só com o rótulo curto.
DICA_DA_FAMILIA: Final[dict[str, str]] = {
    "anova": "ANOVA de uma via sobre TODOS os grupos, com post-hoc nas features que sobreviverem",
    "correlation": "mapa de correlação entre as features da matriz processada",
    "permanova": "testa separação entre grupos nos componentes da PCA, por permutação",
    "normality": (
        "Shapiro-Wilk por variável dentro de cada grupo, depois da correção e da normalização; "
        "o resumo sai em «Qualidade do resultado» e a tabela por variável em CSV"
    ),
    "plsda_multiclass": "PLS-DA sobre os quatro grupos de uma vez, com permutações",
    "clustering": "agrupamento hierárquico das amostras, com dendrograma",
    "kmeans": "agrupamento das amostras em k grupos",
    "som": "mapa auto-organizável das amostras",
    "dspc": "rede de correlação parcial esparsa entre features",
    "classifiers": "classificador com validação cruzada, para medir poder de separação",
}

# O custo medido de cada família, em segundos, pela chave `campo`. Está vazio de propósito: um
# número sem medição seria constante não derivada no rótulo que diz quanto demora. Enquanto a
# família não está aqui, `prosa_das_analises_pedidas` a declara não medida em vez de somá-la como
# zero.
#
# A normalidade fica fora embora medida (de 7 a 16 ms por modo, no cabeçalho de
# `statistics/normalidade.py`): somar milissegundos a uma cauda de 2,5 s não muda a estimativa, e o
# teste da cauda conta com a tabela vazia.
#
# A ANOVA tem medição e fica fora porque o custo dela não é um número. Com zero sobreviventes custa
# 16,8 ms; com o post-hoc `fisher` sobre 214 sobreviventes, 92,3 ms; com `tukey` sobre os mesmos
# 214, 28.486,6 ms. Um valor por família não distingue os casos, e somar 16,8 ms só estaria certo
# por acaso: zero sobreviventes é propriedade deste corpus, o padrão de `AnalysisSpec.post_hoc` é
# `tukey`, e no primeiro estudo com sobreviventes a estimativa erraria cerca de 28,5 s para baixo.
# Para incluí-la, o tipo desta tabela tem de mudar, porque o custo depende de `post_hoc` e do
# número de sobreviventes, que só se sabe depois de correr.
#
# A correlação fica fora pelo mesmo motivo. No `pos_hilic` processado, o mapa de 60 features mais a
# busca sobre todas custam 10,0 ms com `pearson`, o padrão, 32,8 ms com `spearman` e 1.368,2 ms
# com `kendall`. Aqui o padrão é o lado barato e o teto é 1,4 s, então a lacuna custa pouco.
CUSTO_DA_ANALISE_S: Final[dict[str, float]] = {}


def familia_e_barata(familia: FamiliaDeAnalise) -> bool:
    """Se esta família custa segundos, derivado do padrão de `AnalysisSpec`.

    As baratas vêm ligadas por padrão e as caras desligadas, então o padrão é a classificação de
    custo, e ler daqui evita uma segunda lista que divergiria dele.
    """
    return bool(getattr(PADRAO_DA_ANALISE, familia.campo))


def familia_pedida(analise: AnalysisSpec, familia: FamiliaDeAnalise) -> bool:
    """Se esta família foi pedida nesta análise.

    `bool()` e não `is True`, porque `classifiers` é uma tupla: vazia é 'nenhum' e com elementos é
    'estes'. Comparar com `True` deixaria a família mais cara sempre fora da conta.
    """
    return bool(getattr(analise, familia.campo))


def familias_pedidas(analise: AnalysisSpec) -> tuple[FamiliaDeAnalise, ...]:
    """As famílias pedidas, na ordem da tabela."""
    return tuple(f for f in FAMILIAS_DE_ANALISE if familia_pedida(analise, f))


def etapas_das_analises(analise: AnalysisSpec) -> tuple[str, ...]:
    """As etapas que a barra anuncia por causa das análises pedidas.

    Uma etapa por família, e não uma etapa 'análises', para a barra não ficar parada durante o
    conjunto inteiro. `api.rotas.corre_as_analises` itera sobre a mesma `familias_pedidas` e marca
    uma vez por volta, então cada etapa anunciada é marcada.
    """
    return tuple(f.etapa for f in familias_pedidas(analise))


def familias_por_medir(analise: AnalysisSpec) -> tuple[FamiliaDeAnalise, ...]:
    """As famílias pedidas cujo custo `CUSTO_DA_ANALISE_S` ainda não conhece."""
    return tuple(f for f in familias_pedidas(analise) if f.campo not in CUSTO_DA_ANALISE_S)


def cauda_da_execucao_s(analise: AnalysisSpec | None) -> float:
    """A cauda da execução em segundos, para esta análise.

    `None` quer dizer 'não me disseram' e devolve a cauda fixa. Não é o mesmo que `AnalysisSpec()`,
    que traz os padrões e já pede famílias; colapsar os dois cobraria de quem chama sem argumento
    uma escolha que nunca fez.

    Família sem custo medido soma zero aqui e é nomeada em `prosa_das_analises_pedidas`: um número
    inventado pareceria medida, e omiti-la calaria o que a estimativa não cobre.
    """
    if analise is None:
        return CAUDA_DA_EXECUCAO_S
    medidas = sum(
        CUSTO_DA_ANALISE_S[f.campo]
        for f in familias_pedidas(analise)
        if f.campo in CUSTO_DA_ANALISE_S
    )
    return CAUDA_DA_EXECUCAO_S + medidas


def prosa_das_etapas_pedidas(analise: AnalysisSpec) -> str:
    """Quantas etapas a barra ganha por causa das famílias pedidas, e quais.

    Fala de progresso, não de duração: quanto tempo a estimativa não cobre é com
    `prosa_das_analises_pedidas`, e os dois rótulos ficam vizinhos. Sem família pedida, a frase diz
    que o pipeline corre mesmo assim, senão se leria como 'a execução não faz nada'.

    Os nomes saem sem acento porque são os das etapas na barra (`correcao`, `pipeline`,
    `relatorio de qualidade`): escrever 'correlação' aqui e `correlacao` na barra daria dois nomes
    à mesma etapa. A caixa de seleção nomeia a família, e por isso vai acentuada.
    """
    etapas = etapas_das_analises(analise)
    if not etapas:
        return (
            "Nenhuma família pedida — a execução corrige e processa como sempre, e a barra "
            "anuncia só as etapas do pipeline."
        )
    plural = "etapa" if len(etapas) == 1 else "etapas"
    return f"A barra anuncia {len(etapas)} {plural} a mais: {', '.join(etapas)}."


def prosa_das_analises_pedidas(analise: AnalysisSpec | None) -> str:
    """O que as famílias pedidas acrescentam à estimativa, ou vazio quando não acrescentam nada.

    Vazio quando não há análise declarada ou nenhuma família foi pedida, e aí a cauda fixa cobre
    tudo. Senão, a frase dá o número de famílias e quantas estão na classe 'minutos', e avisa
    quando o custo de alguma ainda não está em `CUSTO_DA_ANALISE_S`.

    O que cresce é a contagem, que é derivada: marcar uma família cara muda o que a tela promete
    sem afirmar uma duração que ninguém mediu.
    """
    if analise is None:
        return ""
    pedidas = familias_pedidas(analise)
    if not pedidas:
        return ""
    caras = tuple(f for f in pedidas if not familia_e_barata(f))
    por_medir = familias_por_medir(analise)
    plural = "família" if len(pedidas) == 1 else "famílias"
    corpo = f" — mais {len(pedidas)} {plural} de análise"
    if caras:
        corpo += f", {len(caras)} na classe «minutos» ({', '.join(f.etapa for f in caras)})"
    if por_medir:
        # A frase diz 'não medido' e não 'desprezível': o custo dessas famílias não está na soma,
        # e quem decide se espera precisa saber disso.
        quantas = "delas" if len(por_medir) == len(pedidas) else f"de {len(por_medir)} delas"
        corpo += f", e o custo {quantas} ainda NÃO foi medido"
    return corpo


# Os corretores cujo custo cresce com o número de lotes. Só o SERRF: ele treina uma floresta por
# feature e por lote, com o mesmo número de árvores em cada uma, e dois lotes quase dobram o custo
# (1,89x no `neg_c18`). O LOESS ajusta uma curva sobre os QCs de cada lote, os mesmos pontos
# repartidos, e reparti-los não acrescenta trabalho.
# O fator aplicado é `n_lotes` e não 1,89: sobrestima em 6% no único par medido, e sobrestimar é o
# lado seguro, porque estimativa curta faz o profissional achar que travou.
CUSTO_POR_LOTE: Final[frozenset[str]] = frozenset({"serrf"})


def estimativa_da_execucao(
    n_features: int | None,
    corretor: str,
    *,
    serial: bool,
    n_lotes: int = 1,
    analise: AnalysisSpec | None = None,
) -> str:
    """Quanto 'Executar pipeline' deve custar, em prosa, para este corretor e esta matriz.

    Mede um ajuste de um corretor, mais a cauda; `estimativa_da_medicao` mede a medição de
    corretor, com `n_qc` reajustes de cada um dos três.

    Sem `n_features` sai a frase de ausência, e não um número: antes da validação não se sabe
    quantas features a matriz tem. Corretor que a tabela não conhece também dá a frase de
    ausência, porque uma estimativa sobre método não medido seria invenção.

    `analise` decide a cauda por `cauda_da_execucao_s`, e a frase nomeia as famílias ainda não
    medidas. `None` quer dizer 'não me disseram quais análises correm' e dá a cauda fixa;
    `AnalysisSpec()` traz os padrões, que já pedem famílias.
    """
    custo = CUSTO_DE_UM_AJUSTE_MS.get(corretor)
    if not n_features or n_features <= 0 or custo is None:
        # A frase nomeia a operação porque o rótulo vizinho, de `estimativa_da_medicao`, também
        # espera a validação, e com a mesma frase o profissional não saberia qual é qual. Diz
        # 'estimativa' e não 'palpite', que é o nome do botão que escolhe o corretor.
        return (
            "**estimativa de executar o pipeline:** aparece depois de «Validação» — é ela que diz "
            "quantas features a matriz tem"
        )
    fator = FATOR_SERIAL_DO_SERRF if (serial and corretor == "serrf") else 1.0
    lotes = max(1, n_lotes)
    fator_dos_lotes = float(lotes) if corretor in CUSTO_POR_LOTE else 1.0
    segundos = n_features * custo * fator * fator_dos_lotes / 1000 + cauda_da_execucao_s(analise)
    sobre = f"{n_features:,} features".replace(",", ".")
    if fator_dos_lotes > 1:
        sobre += f" em {lotes} lotes"
    # O nome do corretor vai no rótulo, que distingue as duas estimativas. A duração continua a ser
    # o primeiro numeral da frase, que é o que o teste lê por regex.
    return (
        (
            f"**estimativa de executar o pipeline com `{corretor}`:** "
            f"~{_prosa_de_duracao(segundos)} sobre {sobre}"
        )
        + (" — máquina de 12 núcleos lógicos; a sua pode diferir" if segundos > 60 else "")
        + (
            # As análises vão no fim, depois da ressalva da máquina, para 'a sua pode diferir'
            # qualificar o corretor e não o trabalho que o número não cobre.
            prosa_das_analises_pedidas(analise)
        )
    )


def _prosa_de_duracao(segundos: float) -> str:
    """Segundos em prosa, em minutos acima de 90 s: '1.400 s' ninguém converte de cabeça."""
    if segundos < 90:
        return f"{segundos:.0f} s"
    return f"{segundos / 60:.0f} min"


def segundos_da_medicao(
    n_features: int,
    *,
    serial: bool,
    intervalo: bool = False,
    n_qc: int | None = None,
) -> float:
    """Os segundos que a medição de corretor deve custar, o número de `estimativa_da_medicao`.

    Com o intervalo por barra e o número de QCs conhecido, multiplica pela razão da contagem de
    ajustes, `(n+1)/2` (`metrics.crossval.jackknife_cost_ratio`): `n(n+1)/2` ajustes por corretor
    contra `n`. Vale para os três corretores, `none` incluído, porque é contagem de ajustes. Sem o
    número de QCs, ou com menos de um, o fator não se aplica e a frase diz a conta.
    """
    por_feature_ms = sum(
        custo * (FATOR_SERIAL_DO_SERRF if serial and corretor == "serrf" else 1.0)
        for corretor, custo in CUSTO_POR_FEATURE_MS.items()
    )
    segundos = n_features * por_feature_ms / 1000.0
    if intervalo and n_qc is not None and n_qc >= 1:
        segundos *= jackknife_cost_ratio(n_qc)
    return segundos


def estimativa_da_medicao(
    n_features: int | None,
    *,
    serial: bool,
    intervalo: bool = False,
    n_qc: int | None = None,
) -> str:
    """Quanto a medição de corretor deve custar, em prosa, a partir do tamanho da entrada.

    `serial` distingue os dois regimes da mesma máquina: `False` é o `joblib` com processos, e
    `True` é um núcleo só, o da caixa 'Usar vários núcleos' desmarcada. No `serrf` a diferença é de
    cerca de 5,7x, e escondê-la faria a estimativa errar por horas no `pos_hilic`.

    Sem `n_features` sai a frase de ausência, e não um número: antes da validação não se sabe
    quantas features a matriz tem.

    A frase se declara estimativa e traz a base: a medição é de uma máquina de 12 núcleos lógicos,
    e outra pode diferir por um fator que ninguém mediu.

    `intervalo` é a caixa do intervalo por barra. Marcada, o número multiplica por `(n+1)/2` com os
    `n_qc` QCs da matriz validada, e a frase diz o fator; sem o número de QCs, diz a conta. O fator
    usa os QCs da matriz inteira: com um lote abaixo do mínimo do corretor
    (`metrics.crossval.jackknife_qc_minimum`), aquele corretor não tem intervalo, e a estimativa
    erra para cima.
    """
    if not n_features or n_features <= 0:
        # Nomeia a operação pelo mesmo motivo da frase de ausência de `estimativa_da_execucao`: os
        # dois rótulos são vizinhos, e cada um diz o nome do botão logo acima dele.
        return (
            "**estimativa de medir qual corretor vence:** aparece depois de «Validação» — é ela "
            "que diz quantas features a matriz tem"
        )
    segundos = segundos_da_medicao(n_features, serial=serial, intervalo=intervalo, n_qc=n_qc)
    # Separador de milhar à mão e não `:n`, que obedece à `locale` do processo e num Windows em
    # inglês devolve `2,184`.
    milhar = f"{n_features:,}".replace(",", ".")
    return (
        f"**estimativa de medir qual corretor vence: {prosa_de_duracao(segundos)}** para {milhar} "
        f"features — os três corretores, cada um reajustado uma vez por QC"
        + (_frase_do_intervalo_na_estimativa(n_qc) if intervalo else "")
        + ". Medido em máquina de 12 núcleos"
        # O executável paraleliza por padrão; a série é a da caixa 'Usar vários núcleos'
        # desmarcada, ou a de quando o interpretador embarcado falta.
        + (" e corrigido para o regime de UM núcleo, que é o da caixa desticada" if serial else "")
        # A faixa de trabalhos fica no topo da janela (`tema._monta_faixa_de_trabalhos`), e o nome
        # vai junto porque o lugar sozinho não a acha.
        + "; a sua pode diferir. A janela **não** desiste de esperar, e você pode parar o "
        "trabalho na faixa «Trabalhos em curso», no topo da janela."
    )


def _frase_do_intervalo_na_estimativa(n_qc: int | None) -> str:
    """O pedaço da estimativa que diz o custo do intervalo por barra."""
    if n_qc is None or n_qc < 1:
        return (
            "; com o intervalo em cada barra, multiplique por (n+1)/2, com n o número de QCs, "
            "que aparece depois de «Validação»"
        )
    fator = f"{jackknife_cost_ratio(n_qc):g}".replace(".", ",")
    return f", e com o intervalo em cada barra {fator} vezes isso, pelos {n_qc} QCs desta matriz"


def prosa_de_duracao(segundos: float) -> str:
    """Segundos em prosa, com a unidade que o número pede e sem precisão inventada.

    A precisão cai quando a magnitude sobe: '2.402 s' afirmaria quatro algarismos significativos
    sobre uma extrapolação, e 'cerca de 40 min' afirma o que se sabe.
    """
    if segundos < 90:
        return f"cerca de {round(segundos)} s"
    if segundos < 5_400:
        return f"cerca de {round(segundos / 60)} min"
    # Vírgula decimal, porque a prosa do produto é em português e `:.1f` produz `3.4`.
    return f"cerca de {segundos / 3600:.1f} h".replace(".", ",")


def partes_do_defeito(defect_type: str, rotulos: dict[str, str] | None = None) -> tuple[str, str]:
    """Um rótulo de `ROTULOS_DE_DEFEITO` partido em nome e significado, para a legenda da tela.

    Deriva da constante, e não de uma segunda tabela: um tipo novo entra em `ROTULOS_DE_DEFEITO`
    e aparece na legenda sem ninguém lembrar de outro lugar.

    O separador é o travessão com espaços, e esse formato é contrato da constante: ele separa o
    nome curto da explicação. Rótulo sem o separador levanta, porque devolver `("", texto)`
    desenharia uma linha com a primeira coluna vazia, que se lê como célula por preencher.

    `rotulos` é argumento com padrão para o teste plantar um rótulo sem travessão sem mutar a
    constante do módulo.
    """
    rotulo = (ROTULOS_DE_DEFEITO if rotulos is None else rotulos)[defect_type]
    nome, separador, significado = rotulo.partition(" — ")
    if not separador:
        raise ValueError(
            f"o rótulo de `{defect_type}` não tem o travessão que separa o nome do significado: "
            f"{rotulo!r}. A legenda da tela deriva as duas colunas dele, e sem o separador a "
            f"primeira sairia vazia — que se lê como célula por preencher"
        )
    return nome, significado


# O que a máquina de fato propõe: dois eixos, o corretor e o número de lotes. Vive aqui para que a
# travessia da janela e a frase da tela usem a mesma resposta.
# Os outros eixos de `CONTROLES_DE_SOBRESCRITA` não são propostos por rota nenhuma: a máquina não
# tem opinião sobre eles, e o que a tela oferece ali é escolha do profissional. Eixo sem proposta
# nunca entra em `campos_sobrescritos`, porque `/execucoes` recusa com `422` um eixo sobrescrito
# sem proposta com que comparar.
PROPOSTAS_POR_ROTA: Final[dict[str, tuple[str, str]]] = {
    "recomendacao": ("correction", "correction_proposta"),
    "diagnostico": ("n_batches", "lotes"),
}

EIXOS_PROPOSTOS: Final[frozenset[str]] = frozenset(
    eixo for eixo, _campo in PROPOSTAS_POR_ROTA.values()
)


def eixos_propostos_do_ponto(chave: str) -> tuple[str, ...]:
    """Os eixos daquele ponto que a máquina de fato propõe, na ordem da estrutura."""
    return tuple(
        controle.eixo
        for controle in CONTROLES_DE_SOBRESCRITA
        if controle.ponto == chave and controle.eixo in EIXOS_PROPOSTOS
    )


def eixos_so_do_humano_do_ponto(chave: str) -> tuple[str, ...]:
    """Os eixos daquele ponto que a tela oferece e que rota nenhuma propõe.

    É o complemento derivado, e não uma segunda lista: escrita à mão, a soma das duas divergiria de
    `CONTROLES_DE_SOBRESCRITA` quando um eixo entrasse, e o eixo novo sumiria da tela sem erro.
    """
    return tuple(
        controle.eixo
        for controle in CONTROLES_DE_SOBRESCRITA
        if controle.ponto == chave and controle.eixo not in EIXOS_PROPOSTOS
    )


def eixos_do_ponto(
    chave: str, controles: tuple[ControleDaTela, ...] = CONTROLES_DE_SOBRESCRITA
) -> tuple[str, ...]:
    """Os eixos que a tela deixa mover naquele ponto de decisão.

    Levanta em dois casos:
    - eixo fora de `SUPPORTED_OVERRIDES`: a API o recusaria com `422` na fronteira, e a tela leria
      a recusa como erro do profissional;
    - conjunto vazio: o ponto teria justificativa e nenhum controle.

    `controles` é argumento com padrão para o teste plantar um controle sem mexer na constante do
    módulo.
    """
    eixos = tuple(sorted({controle.eixo for controle in controles if controle.ponto == chave}))
    desconhecidos = [eixo for eixo in eixos if eixo not in SUPPORTED_OVERRIDES]
    if desconhecidos:
        raise ValueError(
            f"o ponto {chave!r} declara controle sobre {desconhecidos}, que `SUPPORTED_OVERRIDES` "
            f"não reconhece. O domínio é fechado em {sorted(SUPPORTED_OVERRIDES)}: a API o recusa "
            "com 422 na fronteira, e a tela leria a recusa como erro do usuário"
        )
    if not eixos:
        raise ValueError(
            f"o ponto {chave!r} não tem controle nenhum em `CONTROLES_DE_SOBRESCRITA`. Cada "
            "ponto de decisão automática precisa de justificativa E de controle, e "
            "justificativa sem controle deixa o usuário informado e sem ação"
        )
    return eixos


PONTOS_DE_DECISAO_AUTOMATICA: Final[tuple[PontoDeDecisao, ...]] = (
    PontoDeDecisao(
        chave="diagnostico",
        o_que_decide="a classe do defeito e as fronteiras de lote sugeridas",
        campo_da_justificativa="lotes",
        # O eixo que a confirmação humana das fronteiras de lote move é `n_batches`: confirmar a
        # sugestão ou recusá-la e escolher quantos lotes declarar.
        eixos_sobrescreviveis=eixos_do_ponto("diagnostico"),
    ),
    PontoDeDecisao(
        chave="recomendacao",
        o_que_decide="qual corretor aplicar",
        campo_da_justificativa="rationale",
        # Os eixos que a tela move, e não todos os que a especificação suporta: sobrescrita sobre
        # eixo sem controle seria ignorada pelo sistema.
        eixos_sobrescreviveis=eixos_do_ponto("recomendacao"),
    ),
)


# ---------------------------------------------------------------------------
# A identidade da entrada que produziu um resultado
# ---------------------------------------------------------------------------


def chave_da_entrada(*, modo: str, metodo: str | None, medidas: bytes, lista: bytes) -> str:
    """A identidade da entrada que produziu um resultado: modo, método e digest dos bytes.

    Todo resultado derivado é guardado sob esta chave e só aparece quando ela bate com a seleção
    corrente. Sem isso, um artefato produzido em `neg_c18` seria oferecido, depois de trocar o
    seletor, num arquivo chamado `pos_hilic_*.csv`, porque o nome usa o modo do instante do desenho.

    Os três eixos contam: o modo é rótulo e a API não o confere contra o conteúdo, o método muda o
    que a rota calcula, e os bytes são a entrada. `metodo=None` deixa o método fora da chave, para
    os resultados que não dependem dele e devem sobreviver à troca do corretor.

    A chave leva o `sha256` truncado em 16 hexadecimais, e não os bytes, para não duplicar megabytes
    por resultado; colisão acidental entre dois arquivos da mesma sessão é irrelevante.

    A função é pura e única porque duas implementações da mesma identidade divergem sem erro, e a
    divergência só aparece como artefato salvo com o rótulo errado.
    """
    digest = hashlib.sha256(medidas + b"|" + lista).hexdigest()[:16]
    if metodo is None:
        return f"{modo}|{digest}"
    return f"{modo}|{metodo}|{digest}"


# ---------------------------------------------------------------------------
# A tela de qualidade exibe o conjunto completo de métricas
# ---------------------------------------------------------------------------
#
# Derivado de `QualityReport`, e não redigitado: métrica nova no relatório entra aqui sozinha, e
# uma tela que a omitisse reprova no teste. Nenhum número sozinho resume a qualidade.
METRICAS_DA_TELA_DE_QUALIDADE: Final[frozenset[str]] = (
    REQUIRED_METRIC_FIELDS | OPTIONAL_METRIC_FIELDS
)


# ---------------------------------------------------------------------------
# PLS-DA nunca sem validação
# ---------------------------------------------------------------------------
#
# Nenhum score de PLS-DA chega à tela sem o Q² e o p de permutação ao lado. Noutro estudo do
# laboratório, o PLS-DA mostrou separação visual convincente com Q² de ~24%, provável falso
# positivo; uma tela que mostrasse a separação sem os dois números reproduziria esse erro no olho
# de quem lê. São duas famílias, o par binário e o um-contra-resto multiclasse, e uma família nova
# é uma linha aqui.
PARES_DE_VALIDACAO_OBRIGATORIA: Final[tuple[tuple[str, tuple[str, ...]], ...]] = (
    ("plsda_q2", ("plsda_q2", "plsda_permutation_p")),
    (
        "plsda_multiclass_q2",
        ("plsda_multiclass_q2", "plsda_multiclass_permutation_p"),
    ),
)


# Por onde cada métrica da tela é calculada, derivado e não escrito à mão.
#
# 'Não calculada' cobre dois fatos opostos: a rota calcula e esta execução não tem valor, ou
# nenhuma rota calcula a métrica. A rota chama `build_quality_report` sem `qc_rsd_cv` e sem
# `qc_rsd_r_cv`, então os campos obrigatórios têm caminho incondicional. O par do PLS-DA e a
# contagem de FDR têm caminho sob argumento: `POST /execucoes` os calcula quando o par de grupos é
# declarado, e dizer 'nenhuma rota calcula' sobre eles esconderia que bastam dois campos.
# `qc_rsd_cv_median` e `qc_rsd_r_cv_median` ficam sem caminho por custo: exigem refazer a correção
# uma vez por QC.
#
# No corpus de referência a contagem de FDR é zero; mostrar 0 é a resposta do estudo, e declarar a
# métrica indisponível diria que a ferramenta não sabe responder.
#
# Derivado de `PARES_DE_VALIDACAO_OBRIGATORIA` para que a lista do par e a tabela de caminhos não
# possam divergir.
METRICAS_DE_VALIDACAO_DO_PLSDA: Final[frozenset[str]] = frozenset(
    nome for score, exigidos in PARES_DE_VALIDACAO_OBRIGATORIA for nome in (score, *exigidos)
)

# A interseção com a tela. `PARES_DE_VALIDACAO_OBRIGATORIA` diz que nomes nunca aparecem sem o
# par ao lado; esta diz quais das onze métricas da tela são do PLS-DA. `plsda_multiclass_q2` está
# na primeira e não na segunda: não é escalar, e nenhuma rota o produz. Sem a interseção,
# `CAMINHO_DA_METRICA` diria `"api"` sobre ele, e a tela mandaria procurar um número que não existe.
METRICAS_DO_PLSDA: Final[frozenset[str]] = (
    METRICAS_DE_VALIDACAO_DO_PLSDA & METRICAS_DA_TELA_DE_QUALIDADE
)

# As métricas que a rota produz quando o par de grupos é declarado: o par do PLS-DA e a contagem
# de FDR (`POST /execucoes` chama `plsda` e `univariate_test`). A contagem de FDR entra pelo nome
# porque não pertence a par nenhum, e não há constante de onde derivá-la.
METRICAS_SOB_PAR_DE_GRUPOS: Final[frozenset[str]] = METRICAS_DO_PLSDA | {"n_significant_fdr"}

CAMINHO_DA_METRICA: Final[dict[str, Caminho]] = {
    nome: ("api" if nome in REQUIRED_METRIC_FIELDS | METRICAS_SOB_PAR_DE_GRUPOS else "nenhum")
    for nome in sorted(METRICAS_DA_TELA_DE_QUALIDADE)
}


# ---------------------------------------------------------------------------
# O vocabulário da tela de qualidade
# ---------------------------------------------------------------------------
#
# Os textos descrevem estados de medição, e por isso ficam aqui, junto da tabela de caminhos que
# decide qual deles vale; o painel da janela só decide como desenhá-los.

# Os estados de uma métrica sem valor. `SEM_CAMINHO`: nenhuma rota desta versão a calcula, e não
# há execução que a preencha. `NAO_CALCULADA`: há caminho, e esta execução não trouxe valor.
SEM_CAMINHO: Final[str] = "não disponível nesta versão"
NAO_CALCULADA: Final[str] = "não calculada"

# As três métricas que dependem do par de grupos dizem de que dependem e onde se escolhe: 'não
# calculada' sozinho lê-se como falha, e a rota as calcula desde que o par venha. `NAO_CALCULADA`
# continua a prefixar, para a frase manter o vocabulário dos outros estados. A frase nomeia a aba
# e os dois campos, porque nomear só um dos dois obriga a procurar o outro.
SOB_PAR_DE_GRUPOS: Final[str] = (
    f"{NAO_CALCULADA} — depende do par de grupos, e a escolha é sua: preencha «Grupo A do "
    "contraste» e «Grupo B do contraste», na aba «Análise estatística», e execute outra vez."
)
# A frase que a janela mostra no lugar de `SOB_PAR_DE_GRUPOS`: o mesmo fato e a mesma ação, sem a
# instrução para outra superfície. Ela aparece em vários cartões seguidos, e por isso é curta.
SOB_PAR_DE_GRUPOS_NA_JANELA: Final[str] = (
    f"{NAO_CALCULADA} — falta o par de grupos: preencha «Grupo A do contraste» e «Grupo B do "
    "contraste» em «Análise estatística» e execute outra vez."
)

# A frase que precede a razão de uma métrica sem caminho. Ela é a mesma para todas, e o que muda é
# o motivo; repeti-la em cada entrada da tabela criaria várias redações da mesma afirmação.
NENHUMA_ROTA_CALCULA: Final[str] = (
    "Nenhuma rota desta versão calcula esta métrica, e nenhuma execução a preenche:"
)

# ---------------------------------------------------------------------------
# A referência da área, só onde a fonte foi lida
# ---------------------------------------------------------------------------
#
# Cada métrica cai numa de três classes:
# - fonte lida, âncora consensual: a afirmação, com a fonte nomeada;
# - fonte lida, âncora com ressalva de escopo: a afirmação e a ressalva, sempre juntas;
# - sem âncora: nada, porque a tela não inventa referência.
#
# Sangster (2006) e Dunn (2011) só são conhecidos aqui de segunda mão, e por isso nenhum texto os
# cita. A terceira classe é a maioria. O limiar de 50% do D-ratio de Broadhurst não se transfere
# para `qc_dispersion_ratio`: aquele é por feature no espaço original, este é sobre os scores de
# `k` componentes. A correlação com a ordem de injeção tem inspeção recomendada e nenhum limiar
# publicado.

# A régua, primeira classe: a afirmação é que o RSD dos QCs é o instrumento de qualidade
# analítica, e não um corte.
REFERENCIA_DA_REGUA_DE_RSD: Final[str] = (
    "Referência da área: o RSD dos QCs é a régua consensual de qualidade analítica — "
    "Broadhurst et al. (2018), *Metabolomics*, DOI 10.1007/s11306-018-1367-3"
)

# O mesmo corte na tela do MetaboAnalyst. Broadhurst et al. (2018) continua a ser a âncora; esta é
# a confirmação que o profissional reconhece, do 'Low-repeatability filter' da tela de filtragem.
# É constante porque as duas referências abaixo a citam, e
# `tests/test_equivalencia_metaboanalyst.py` a compara com a transcrição.
CORTE_SUGERIDO_NO_METABOANALYST: Final = "suggested threshold: 20% for LC-MS and 30% for GC-MS"

# O corte de 20%, segunda classe: a afirmação e a ressalva saem juntas. O valor coincide com o
# publicado e não deriva dele. A ressalva é de escopo: nos quatro arquivos de referência, três dos
# quatro modos têm RSD mediano de QC acima de 20% (1,4381, 0,2415, 0,6718 e 0,1550), o que é
# observação sobre esses arquivos e não propriedade do método.
REFERENCIA_DO_CORTE_DE_20: Final[str] = (
    "Referência da área, COM RESSALVA DE ESCOPO: o corte de 20% **coincide** com o valor "
    "consensual e não deriva dele — Broadhurst et al. (2018), *Metabolomics*, "
    "DOI 10.1007/s11306-018-1367-3, que escreve «typically set to < 20% or < 30% depending on "
    "the sample type»; 20% é a mais estrita das duas formas. A ressalva: nos quatro arquivos de "
    "referência deste trabalho, três dos quatro modos têm RSD mediano de QC acima de 20%, então "
    "neste corpus o corte é exigente — e isso é observação sobre estes quatro arquivos, não "
    "propriedade do método. É também o corte que o MetaboAnalyst sugere no "
    "*Low-repeatability filter* da tela de filtragem: «" + CORTE_SUGERIDO_NO_METABOANALYST + "»"
)

# O corte de 30%, da mesma fonte, que nomeia as duas formas. Sem referência, a linha passaria por
# limiar inventado aqui. A ressalva é a inversa da do corte de 20: nos mesmos quatro arquivos, dois
# modos ficam abaixo de 30% (0,2415 e 0,1550) e dois acima (1,4381 e 0,6718).
REFERENCIA_DO_CORTE_DE_30: Final[str] = (
    "Referência da área, COM RESSALVA DE ESCOPO: o corte de 30% é a forma mais FROUXA do mesmo "
    "valor consensual — Broadhurst et al. (2018), *Metabolomics*, "
    "DOI 10.1007/s11306-018-1367-3, «typically set to < 20% or < 30% depending on the sample "
    "type». A ressalva: nos quatro arquivos de referência deste trabalho, dois dos quatro modos "
    "têm RSD mediano de QC abaixo de 30% e dois acima, então neste corpus o corte separa — e isso "
    "é observação sobre estes quatro arquivos, não propriedade do método. O MetaboAnalyst "
    "sugere as duas formas no mesmo controle — o *Low-repeatability filter* —, e reserva os "
    "30% ao GC-MS: «" + CORTE_SUGERIDO_NO_METABOANALYST + "»"
)

# ---------------------------------------------------------------------------
# Os temas: a lista de qualidade lê-se por assunto
# ---------------------------------------------------------------------------
#
# Quatro temas e um residual nomeado repartem as onze métricas da tela:
# - dispersão dentro da amostra: saem do mesmo vetor `rsd_values` e partilham o denominador;
# - dispersão fora da amostra: o `rsd_estimator` que as duas nomeiam define o tema;
# - estrutura no PCA: as duas métricas estruturais;
# - validação do PLS-DA: os nomes de `PARES_DE_VALIDACAO_OBRIGATORIA`;
# - residual: o campo `sentido` da métrica diz que ela não é nota de qualidade.
#
# O critério do segundo tema não é `CAMINHO_DA_METRICA == "nenhum"`. As duas métricas dele não têm
# caminho por custo, e no dia em que uma ganhar caminho um critério derivado dali mudaria o tema
# sem ninguém decidir.
#
# A ordem de declaração deste dicionário é a ordem dos cabeçalhos na tela, e a das métricas dentro
# de cada um é a de `TRADUCAO_DA_METRICA`. Nenhum teste fixa essa ordem, porque fixá-la exigiria
# uma segunda lista escrita à mão.
TEMAS_DA_QUALIDADE: Final[dict[TemaDaMetrica, TemaDeQualidade]] = {
    # A nota não diz que o LOESS zera o RSD dentro da amostra: isso vale para o `batch_loess` do R,
    # e não para o `loess` desta ferramenta. `tests/test_desktop_qualidade.py` reprova se a frase
    # voltar a um texto que chega à janela.
    "dispersao_na_amostra": TemaDeQualidade(
        titulo="Dispersão dos QCs — medida DENTRO da amostra",
        nota=(
            "As quatro saem do mesmo vetor de RSD e partilham o denominador, então comparam-se "
            "entre si. Medido dentro da amostra, o RSD favorece quem ajustou a curva através dos "
            "próprios QCs que ele mede — por isso não ordena corretores."
        ),
    ),
    "dispersao_fora_da_amostra": TemaDeQualidade(
        titulo="Dispersão dos QCs — medida FORA da amostra",
        nota=(
            "As mesmas quantidades, com um QC retido de cada vez: é a comparação honesta entre "
            "corretores, porque não premia quem ajusta a curva através dos QCs que a métrica mede."
        ),
    ),
    "estrutura_no_pca": TemaDeQualidade(
        titulo="Estrutura da matriz no PCA",
        nota=(
            "Não olham o RSD: perguntam se os QCs ficam apertados dentro da nuvem das amostras, e "
            "se a maior fonte de variação da matriz ainda é **quando** a amostra foi injetada."
        ),
    ),
    "validacao_do_plsda": TemaDeQualidade(
        titulo="Validação do PLS-DA",
        nota=(
            "O Q² nunca aparece sem o p de permutação. Separação visual convincente com Q² baixo é "
            "o falso positivo que esta ferramenta existe para expor, e é por isso que as duas se "
            "leem juntas."
        ),
    ),
    "nao_e_nota_de_qualidade": TemaDeQualidade(
        titulo="Contagem que acompanha — NÃO é nota de qualidade",
        nota=(
            "O que está aqui não mede qualidade de correção: mais features significativas não é "
            "sinal de correção melhor. Aparece nesta aba porque é onde se procura, e vem separada "
            "para que a vizinhança não afirme o contrário."
        ),
    ),
}

# O grupo das métricas sem tema declarado, ou sem tradução, aparece marcado como defeito da tela.
# Um grupo 'outras' absorveria o próximo erro em silêncio, e esconder a métrica tiraria da tela
# parte do conjunto de qualidade.
CHAVE_SEM_TEMA: Final[str] = "sem_tema_declarado"
TEMA_SEM_DECLARACAO: Final[TemaDeQualidade] = TemaDeQualidade(
    titulo="Métricas SEM TEMA DECLARADO — defeito desta tela, não do seu dado",
    nota=(
        "Estas métricas não declaram a que tema pertencem, ou não têm tradução nesta tela. Elas "
        "aparecem porque nenhuma métrica de qualidade pode ser escondida, e aparecem **marcadas** "
        "porque um grupo «outras» esconderia o defeito em vez de o mostrar. O valor ao lado "
        "continua correto."
    ),
)

# A tradução das métricas para quem lê a tela. A tabela diz o que se afirma sobre cada métrica; o
# painel decide como desenhar. A bijeção contra `METRICAS_DA_TELA_DE_QUALIDADE` é conferida nas
# duas direções em `tests/test_ui_fluxo.py`: métrica nova apareceria crua, e tradução órfã
# sobreviveria a uma métrica que saiu.
#
# A ordem de declaração é a ordem de leitura dentro de cada tema: `agrupa_metricas_da_qualidade`
# percorre a tabela na ordem em que ela está escrita. Nenhum teste fixa essa ordem.
TRADUCAO_DA_METRICA: Final[dict[str, MetricaTraduzida]] = {
    "qc_rsd_median": MetricaTraduzida(
        rotulo="RSD mediano dos QCs",
        unidade="fração do sinal, e 0,25 é 25%",
        formato="percentagem",
        populacao="vetor_de_rsd",
        tema="dispersao_na_amostra",
        referencia=REFERENCIA_DA_REGUA_DE_RSD,
        sentido="menor é melhor",
        ressalva=(
            "É a dispersão típica de uma feature entre os QCs, e a mediana é a leitura "
            "robusta: a média ao lado sobe com poucas features de RSD muito alto"
        ),
    ),
    "qc_rsd_mean": MetricaTraduzida(
        rotulo="RSD médio dos QCs",
        unidade="fração do sinal, e 0,25 é 25%",
        formato="percentagem",
        populacao="vetor_de_rsd",
        tema="dispersao_na_amostra",
        referencia=REFERENCIA_DA_REGUA_DE_RSD,
        sentido="menor é melhor",
        ressalva=(
            "Quando ela e a mediana divergem muito, quem está a falar é a cauda de features "
            "com RSD alto, e não o conjunto"
        ),
    ),
    "frac_features_under_20": MetricaTraduzida(
        rotulo="Features com RSD dos QCs abaixo de 20%",
        unidade="fração das features, e 0,80 é 80% delas",
        formato="percentagem",
        populacao="vetor_de_rsd",
        tema="dispersao_na_amostra",
        referencia=REFERENCIA_DO_CORTE_DE_20,
        sentido="maior é melhor",
        ressalva=(
            "O denominador são as features com RSD **definido**, e não todas — as que ficaram "
            "indefinidas são contadas à parte e não entram nem em cima nem em baixo"
        ),
    ),
    "frac_features_under_30": MetricaTraduzida(
        rotulo="Features com RSD dos QCs abaixo de 30%",
        unidade="fração das features, e 0,80 é 80% delas",
        formato="percentagem",
        populacao="vetor_de_rsd",
        tema="dispersao_na_amostra",
        referencia=REFERENCIA_DO_CORTE_DE_30,
        sentido="maior é melhor",
        ressalva=(
            "Mesmo denominador da linha de 20% — as features com RSD **definido** —, e é por "
            "isso que as duas são comparáveis entre si"
        ),
    ),
    "qc_dispersion_ratio": MetricaTraduzida(
        rotulo="Dispersão dos QCs dividida pela das amostras, no PCA",
        unidade="razão sem unidade, e 1 é dispersão igual à das amostras biológicas",
        formato="razao",
        populacao="nenhuma",
        tema="estrutura_no_pca",
        # Sem referência: o limiar do D-ratio de Broadhurst (`s_qc/s_amostra < 50%`) é por feature
        # no espaço original, e este número é sobre os scores de `k` componentes.
        sentido="menor é melhor",
        ressalva=(
            "Os QCs são a mesma mistura injetada várias vezes, então idealmente formam um ponto "
            "apertado dentro da nuvem das amostras, que variam de verdade"
        ),
    ),
    "pc1_injection_correlation_abs": MetricaTraduzida(
        rotulo="Correlação entre o primeiro componente do PCA e a ordem de injeção",
        unidade="correlação de Spearman em módulo, de 0 a 1",
        formato="razao",
        populacao="nenhuma",
        tema="estrutura_no_pca",
        # Sem referência: a inspeção é recomendada e não tem limiar publicado, e pôr um aqui seria
        # inventar o que a fonte não dá.
        sentido="menor é melhor",
        ressalva=(
            "Alto significa que a maior fonte de variação da matriz ainda é **quando** a amostra "
            "foi injetada, e não o que havia nela — é o defeito que a correção existe para desfazer"
        ),
    ),
    "qc_rsd_cv_median": MetricaTraduzida(
        rotulo="RSD mediano dos QCs medido fora da amostra, um QC retido de cada vez",
        unidade="fração do sinal, e 0,25 é 25%",
        formato="percentagem",
        # `nenhuma`, e não `vetor_de_rsd`: este número vem de outro estimador, e o
        # `rsd_estimator` do relatório descreve o vetor principal.
        populacao="nenhuma",
        tema="dispersao_fora_da_amostra",
        sentido="menor é melhor",
        ressalva=(
            "É a comparação honesta entre corretores: o RSD medido dentro da amostra premia quem "
            "ajusta a curva através dos próprios QCs que a métrica mede"
        ),
        # A razão da tela é a de quem lê, sem rota nem função. A de código: a rota chama
        # `build_quality_report` sem este vetor.
        por_que_sem_caminho=(
            "obtê-la exige refazer a correção uma vez para cada QC, e a execução desta tela não "
            "paga esse custo"
        ),
    ),
    "qc_rsd_r_cv_median": MetricaTraduzida(
        rotulo="RSD mediano dos QCs pela reamostragem do pipeline R, para compatibilidade",
        unidade="fração do sinal, e 0,25 é 25%",
        formato="percentagem",
        # Mesmo argumento da linha acima: estimador próprio, e não o do vetor principal.
        populacao="nenhuma",
        tema="dispersao_fora_da_amostra",
        sentido="menor é melhor",
        ressalva=(
            "Existe para reproduzir as colunas do `QC - RSD.csv` do R, é ruidoso por construção "
            "— dois QCs retidos por sorteio — e **não deve decidir nada**"
        ),
        por_que_sem_caminho=(
            "obtê-la exige refazer a correção uma vez para cada sorteio de QCs, e a execução desta "
            "tela não paga esse custo"
        ),
    ),
    "n_significant_fdr": MetricaTraduzida(
        rotulo="Features significativas depois da correção de FDR",
        unidade="contagem de features",
        # `contagem`, e não `razao`: `0,000` diria que o número foi arredondado de alguma coisa.
        formato="contagem",
        populacao="nenhuma",
        tema="nao_e_nota_de_qualidade",
        sentido="não é nota de qualidade: mais significativas não é sinal de correção melhor",
        # Quem a conta é `POST /execucoes`, com `univariate_test` sobre a matriz processada e o
        # corte `FDR_ALPHA = 0,05`, o mesmo de `multiverse/stability.py`.
        ressalva=(
            "Só aparece quando você declara os dois grupos a comparar, e **zero é um resultado, "
            "não uma falha**: nestes dados nenhuma feature sobrevive à correção de múltiplos "
            "testes — o menor q medido foi 0,9976 sobre 96 combinações do teste de robustez"
        ),
    ),
    "plsda_q2": MetricaTraduzida(
        rotulo="Q² do PLS-DA — quanto o modelo acerta em amostras que não viu",
        unidade="fração, e 0,24 é 24%",
        # `razao`, e não `percentagem`: a quimiometria reporta Q² como decimal, e o valor pode ser
        # negativo, caso em que `-24,0%` lê-se pior que `-0,240`.
        formato="razao",
        populacao="nenhuma",
        tema="validacao_do_plsda",
        sentido="maior é melhor",
        ressalva=(
            "É `1 - PRESS/TSS`, então valor NEGATIVO é possível e significa prever pior que a "
            "média. Noutro estudo do laboratório houve separação visual convincente com Q² de "
            "~24%, provável falso positivo — é o erro que esta ferramenta existe para evitar"
        ),
    ),
    "plsda_permutation_p": MetricaTraduzida(
        rotulo="p de permutação do PLS-DA",
        unidade="fração das permutações que superaram o Q² observado, de 0 a 1",
        # `razao`, porque valor-p não se escreve em percentagem.
        formato="razao",
        populacao="nenhuma",
        tema="validacao_do_plsda",
        sentido="quanto menor, menos provável que uma separação assim apareça por acaso",
        ressalva=(
            "Sai sempre ao lado do Q² e não o substitui: um p pequeno com Q² baixo diz que a "
            "separação é improvável por acaso **e** que o modelo não generaliza"
        ),
    ),
    # As duas do multiclasse não são das onze da tela, porque não são escalares. Entram porque
    # `PARES_DE_VALIDACAO_OBRIGATORIA` as pode pôr na seção do PLS-DA, e sem tradução sairiam em
    # inglês e marcadas.
    "plsda_multiclass_q2": MetricaTraduzida(
        rotulo="Q² do PLS-DA de todos os grupos — um valor por grupo",
        unidade="fração, **uma por grupo**: com quatro grupos são quatro números",
        # Os três campos seguintes repetem os de `plsda_q2`: é a mesma quantidade, num modelo
        # um-contra-resto, e só a cardinalidade muda. Outro `tema` partiria a seção em duas.
        formato="razao",
        populacao="nenhuma",
        tema="validacao_do_plsda",
        sentido="maior é melhor, e lê-se grupo a grupo",
        ressalva=(
            "Cada valor é de um modelo que separa **um** grupo de todos os outros juntos, e não "
            "os grupos entre si. Não há média: quatro valores muito diferentes significam que a "
            "ferramenta separa bem um grupo e mal outro, e é isso que a lista mostra"
        ),
    ),
    "plsda_multiclass_permutation_p": MetricaTraduzida(
        rotulo="p de permutação do PLS-DA de todos os grupos — um por grupo",
        unidade="fração de 0 a 1, **uma por grupo**, alinhada com a lista de Q²",
        # Os mesmos de `plsda_permutation_p`, pelo motivo da entrada acima.
        formato="razao",
        populacao="nenhuma",
        tema="validacao_do_plsda",
        sentido="quanto menor, menos provável que aquela separação apareça por acaso",
        ressalva=(
            "Não são combinados num p só: juntá-los exigiria escolher entre «algum grupo separa» "
            "e «todos separam», que são perguntas diferentes. Cada p vale para o seu grupo"
        ),
    ),
}


def agrupa_metricas_da_qualidade(
    traducao: dict[str, MetricaTraduzida] | None = None,
) -> tuple[GrupoDeMetricas, ...]:
    """As onze métricas repartidas por tema, prontas para desenho.

    A partição deriva do campo `tema`, e o painel só desenha o que esta função devolve. Toda métrica
    de `METRICAS_DA_TELA_DE_QUALIDADE` sai em exatamente um grupo: as de tema conhecido no grupo
    dele, e as outras (tema fora do vocabulário, ou sem tradução) no grupo marcado como defeito.
    `traducao` entra por argumento para o teste exercitar o grupo marcado sem quebrar a tabela real.

    Tema vazio não vira cabeçalho, porque um título sobre coisa nenhuma promete um número que não
    vem. Os cabeçalhos saem na ordem de `TEMAS_DA_QUALIDADE`, e as métricas na de
    `TRADUCAO_DA_METRICA`, que já é a ordem de leitura: a mediana antes da média, e as duas antes
    das frações que delas dependem.
    """
    tabela = TRADUCAO_DA_METRICA if traducao is None else traducao
    baldes: dict[str, list[str]] = {chave: [] for chave in TEMAS_DA_QUALIDADE}
    marcadas: list[str] = []
    for nome, traduzida in tabela.items():
        if nome not in METRICAS_DA_TELA_DE_QUALIDADE:
            # Tradução órfã: a bijeção reprova por ela, e desenhá-la aqui mostraria na tela uma
            # métrica que o relatório não produz.
            continue
        if traduzida.tema in baldes:
            baldes[traduzida.tema].append(nome)
        else:
            marcadas.append(nome)
    # As sem tradução entram no mesmo grupo marcado: para quem lê, o fato é o mesmo.
    marcadas.extend(sorted(METRICAS_DA_TELA_DE_QUALIDADE - set(tabela)))
    grupos = [
        GrupoDeMetricas(chave, tema.titulo, tema.nota, tuple(baldes[chave]))
        for chave, tema in TEMAS_DA_QUALIDADE.items()
        if baldes[chave]
    ]
    if marcadas:
        grupos.append(
            GrupoDeMetricas(
                CHAVE_SEM_TEMA,
                TEMA_SEM_DECLARACAO.titulo,
                TEMA_SEM_DECLARACAO.nota,
                tuple(marcadas),
            )
        )
    return tuple(grupos)


def estado_de_ausencia(nome: str, caminhos: dict[str, str] | None = None) -> str:
    """Qual dos três textos de ausência uma métrica sem valor merece.

    - `SEM_CAMINHO`: nenhuma rota desta versão a calcula, e não há o que fazer;
    - `SOB_PAR_DE_GRUPOS`: a rota calcula se o par de grupos vier, e basta preencher dois campos;
    - `NAO_CALCULADA`: há caminho e esta execução não trouxe valor; basta executar outra vez.

    A pergunta do caminho vem primeiro: uma métrica sob argumento e sem rota que a produza tem de
    sair como sem caminho, porque mandar preencher campos para um número que nada calcula é pior
    que não dizer nada. Hoje as classes são disjuntas, e a ordem decide se deixarem de ser.

    A pergunta anterior, se a métrica está em `computed`, fica com quem chama
    (`janela/qualidade.py::texto_da_metrica`). `caminhos` entra por argumento para o teste ver os
    dois desfechos da mesma métrica sem esperar que uma rota mude.
    """
    tabela = CAMINHO_DA_METRICA if caminhos is None else caminhos
    if tabela.get(nome) == "nenhum":
        return SEM_CAMINHO
    if nome in METRICAS_SOB_PAR_DE_GRUPOS:
        return SOB_PAR_DE_GRUPOS
    return NAO_CALCULADA


# ---------------------------------------------------------------------------
# O endereço da métrica: onde o nome técnico aparece
# ---------------------------------------------------------------------------
#
# Quem opera o espectrômetro conhece RSD, QC e deriva, e não a grafia `frac_features_under_20`
# deste código. O nome técnico sai da linha do valor e fica no fim da legenda, como endereço: quem
# tem o nome cru num CSV exportado ou numa resposta da API acha ali o que ele é. Fora da tela,
# `QualityReport`, o JSON, a API e a CLI continuam a usar o nome técnico.
PREFIXO_DO_ENDERECO: Final[str] = "No arquivo exportado e na API:"

# A abertura da legenda. Sem este rótulo, a primeira linha lê-se como continuação do valor que
# está acima dela.
PREFIXO_DE_LEITURA: Final[str] = "Como ler:"


def endereco_da_metrica(nome: str) -> str:
    """A última frase da legenda, a única que nomeia a métrica pelo nome técnico.

    Fica no fim, como nota de rodapé: quem veio ler o número não tropeça nela, e quem veio do CSV
    acha-a onde se procura referência.
    """
    return f"{PREFIXO_DO_ENDERECO} `{nome}`"


# ---------------------------------------------------------------------------
# Os dois rótulos do RSD, em português
# ---------------------------------------------------------------------------
#
# Um RSD só é comparável com outro quando os dois declaram `rsd_variant` e `rsd_estimator`, e os
# valores destes rótulos (`in_sample`, `outlier_removed`) também são grafia de código. A frase do
# estimador diz a circularidade, e não o nome: medido nos próprios QCs do ajuste, o RSD favorece
# quem ajusta a curva nos QCs.
#
# A bijeção contra `RsdVariant` e `RsdEstimator`, de `domain/labels.py`, é conferida em teste: um
# valor novo sem entrada aqui apareceria como outra política.
PROSA_DA_VARIANTE_DE_RSD: Final[dict[str, str]] = {
    "plain": "sem remoção de outliers",
    "outlier_removed": "com remoção de outliers",
}

PROSA_DO_ESTIMADOR_DE_RSD: Final[dict[str, str]] = {
    "in_sample": "medido nos próprios QCs do ajuste",
    "leave_one_qc_out": "medido com um QC retido de cada vez",
    "r_cv": "medido pela reamostragem de QCs do pipeline R",
}


def prosa_da_variante_de_rsd(valor: object) -> str:
    """Qual fórmula de RSD, em português.

    Valor desconhecido aparece marcado: devolver o nome cru deixaria um rótulo novo aparecer em
    inglês sem nada o denunciar.
    """
    if isinstance(valor, str) and valor in PROSA_DA_VARIANTE_DE_RSD:
        return PROSA_DA_VARIANTE_DE_RSD[valor]
    return f"{valor} (**variante de RSD sem tradução nesta tela**)"


def prosa_do_estimador_de_rsd(valor: object) -> str:
    """Como o RSD foi medido, em português. Valor desconhecido aparece marcado, como acima."""
    if isinstance(valor, str) and valor in PROSA_DO_ESTIMADOR_DE_RSD:
        return PROSA_DO_ESTIMADOR_DE_RSD[valor]
    return f"{valor} (**estimador de RSD sem tradução nesta tela**)"


def prosa_dos_rotulos_do_rsd(variante: object, estimador: object) -> str:
    """Os dois rótulos do RSD juntos, que é a única forma em que eles significam alguma coisa.

    A linha do valor e a linha de antes-e-depois mostram o mesmo par na mesma aba; uma composição
    só impede que as duas divirjam na grafia.
    """
    return f"{prosa_da_variante_de_rsd(variante)}, {prosa_do_estimador_de_rsd(estimador)}"


def rotulo_da_metrica(nome: str) -> str:
    """O nome da métrica na língua de quem lê, sem o nome técnico.

    O nome técnico vai no fim da legenda (`endereco_da_metrica`). Nome desconhecido aparece marcado,
    para a omissão ficar visível na tela; o teste de bijeção a pega antes disso.
    """
    traduzida = TRADUCAO_DA_METRICA.get(nome)
    if traduzida is None:
        return f"{nome} (**métrica sem tradução nesta tela**)"
    return traduzida.rotulo


def legenda_da_metrica(nome: str, caminhos: dict[str, str] | None = None) -> str:
    """A unidade, o sentido e a ressalva de uma métrica, e a razão quando não há valor possível.

    A unidade importa: o RSD deste projeto é fração, e ler 0,155 como pontos percentuais muda a
    conclusão. A frase de ausência deriva de `CAMINHO_DA_METRICA`, e não fica escrita na tabela,
    para acompanhar a rota quando uma métrica ganhar caminho. `caminhos` entra por argumento para o
    teste ver os dois desfechos da mesma métrica.

    As partes separam-se por `\\n\\n` porque `QTextDocument.setMarkdown` junta num bloco só as
    linhas separadas por `\\n` simples. `Como ler:` abre a primeira parte, e o nome técnico fecha a
    última (`endereco_da_metrica`).
    """
    traduzida = TRADUCAO_DA_METRICA.get(nome)
    if traduzida is None:
        # Métrica desconhecida: o nome cru, marcado, é a única coisa verdadeira que a tela sabe
        # dizer.
        return f"`{nome}` — **métrica sem tradução nesta tela**"
    partes = [f"{PREFIXO_DE_LEITURA} {traduzida.unidade}; {traduzida.sentido}."]
    if traduzida.ressalva:
        partes.append(f"{traduzida.ressalva}.")
    tabela = CAMINHO_DA_METRICA if caminhos is None else caminhos
    if tabela.get(nome) == "nenhum":
        partes.append(f"{NENHUMA_ROTA_CALCULA} {traduzida.por_que_sem_caminho}.")
    # A referência da área vai na legenda, e não na linha do valor, que é o que quem abre a aba veio
    # buscar. Sem referência, nada se escreve: repetir 'sem referência publicada' na maioria das
    # métricas ensinaria a saltar a linha.
    if traduzida.referencia:
        partes.append(f"{traduzida.referencia}.")
    partes.append(endereco_da_metrica(nome))
    return "\n\n".join(partes)


# A linha do PLS-DA quando a execução não declarou o par de grupos: a rota calcula o PLS-DA, e a
# ação é preencher os dois campos.
SEM_PLSDA: Final[str] = (
    "Esta execução não declarou o par de grupos, então não há PLS-DA a validar. Preencha os dois "
    "campos de grupo e execute outra vez: sem o par, um Q² não teria população declarada — os "
    "dados têm vários grupos, e cada par responde uma pergunta diferente."
)

# A linha do PLS-DA quando nenhuma rota o calcula, o que volta a acontecer se a chamada for
# desligada. Fica separada de `SEM_PLSDA` porque as duas pedem ações opostas: aqui não há nada a
# fazer do lado de quem opera.
SEM_CAMINHO_PARA_PLSDA: Final[str] = (
    "Nenhum caminho do produto calcula PLS-DA nesta versão: a guarda vale e não há score a "
    "guardar. Não há o que preencher — é limitação da ferramenta, não da sua execução."
)


# Quanto o R²Y tem de exceder o Q² para a tela acrescentar o alerta. Não é limiar de decisão: só
# liga uma frase de leitura. O 0,4 vem da ordem de grandeza do caso conhecido (Q² de ~24% com um
# R²Y perto de 1), e não de uma distribuição medida.
DISTANCIA_R2Y_Q2_SUSPEITA: Final = 0.4


def aviso_do_plsda(
    caminhos: dict[str, str] | None = None, relatorio: dict[str, Any] | None = None
) -> str:
    """A linha do PLS-DA, em três desfechos que pedem ações diferentes.

    - `SEM_CAMINHO_PARA_PLSDA`: nenhuma rota calcula o par, e não há o que fazer;
    - `SEM_PLSDA`: há caminho e a execução não pediu; basta preencher os dois campos;
    - a procedência: há caminho e o par veio, e o Q² lê-se ao lado do R²Y e do p.

    `caminhos` entra por argumento para o teste alcançar o primeiro desfecho, que em produção não
    acontece. A procedência existe porque um Q² sem o par de grupos não diz que pergunta respondeu.
    """
    # `METRICAS_DO_PLSDA`, e não os nomes de `PARES_DE_VALIDACAO_OBRIGATORIA`: os do multiclasse
    # não estão em `CAMINHO_DA_METRICA`, e com eles o `all` seria sempre falso.
    tabela = CAMINHO_DA_METRICA if caminhos is None else caminhos
    if all(tabela.get(n) == "nenhum" for n in sorted(METRICAS_DO_PLSDA)):
        return SEM_CAMINHO_PARA_PLSDA
    if relatorio is None or relatorio.get("plsda_q2") is None:
        return SEM_PLSDA
    return procedencia_do_plsda(relatorio)


def procedencia_do_plsda(relatorio: dict[str, Any]) -> str:
    """A linha que diz sobre que par de grupos e em que regime o PLS-DA foi medido.

    Sem ela o Q² viaja sem população: com quatro grupos há seis pares, e cada um é uma pergunta
    biológica diferente. O alerta de R²Y alto com Q² baixo descreve o falso positivo clássico, uma
    separação visual convincente que não generaliza; quem lê um R²Y de 0,9 sem ele conclui o oposto
    do que os números dizem.
    """
    grupos = relatorio.get("plsda_groups") or ("?", "?")
    par = " contra ".join(str(g) for g in grupos)
    r2y = relatorio.get("plsda_r2y")
    aviso = ""
    q2 = relatorio.get("plsda_q2")
    if (
        isinstance(r2y, (int, float))
        and isinstance(q2, (int, float))
        and float(r2y) - float(q2) > DISTANCIA_R2Y_Q2_SUSPEITA
    ):
        aviso = (
            " ATENÇÃO: o R²Y é muito maior que o Q², que é a assinatura de um modelo que "
            "descreve bem as amostras que viu e não generaliza — o falso positivo que esta "
            "validação existe para expor."
        )
    return (
        f"Contraste medido: {par}. R²Y = {_numero(r2y)}, "
        f"{relatorio.get('plsda_cv_folds')} dobras estratificadas, "
        f"{relatorio.get('plsda_n_permutations')} permutações, semente "
        f"{relatorio.get('plsda_seed')}.{aviso}"
    )


def _numero(valor: Any) -> str:
    """Um float em prosa curta, ou a palavra. `None` nunca vira string vazia."""
    if valor is None:
        return "não medido"
    if isinstance(valor, (int, float)):
        return f"{float(valor):.4g}"
    return str(valor)


def prosa_do_numero(valor: Any) -> str:
    """Um número da resposta em texto, com quatro casas, ou a razão de não haver número.

    `None` é o que a serialização produz para `nan` (pydantic converte não finito em `null`), e
    mostrar vazio diria que a métrica não separa quando ela não pôde ser medida. Acontece nas
    medianas do par antes/depois: `api/rotas.py` devolve `nan` quando nenhuma feature tem RSD
    definido nos dois lados, e isso é diferente de não ter pedido a comparação.
    """
    if valor is None:
        return "indefinido (não houve feature com RSD mensurável nos dois lados)"
    # Vírgula decimal, como os percentuais da mesma aba.
    return f"{float(valor):.4f}".replace(".", ",")


# A concordância com a proposta da máquina, em prosa. O vocabulário é `EstadoDeConcordancia`, de
# `persistence/repositories.py`, e a bijeção contra ele é conferida em teste. A chave crua vai ao
# lado, entre parênteses, porque é ela que o registro grava. `sem_proposta` cobre também a
# Correção feita sem 'Seguir sugestões do app' com o Palpite calculado: houve proposta, e ela não
# foi seguida.
PROSA_DA_CONCORDANCIA: Final[dict[str, str]] = {
    "nao_sobrescreveu": "a proposta da máquina foi seguida",
    "sobrescreveu_concordando": "a proposta foi revista e mantida",
    "sobrescreveu_discordando": "a proposta da máquina foi trocada",
    "sem_proposta": (
        "nenhuma proposta da máquina foi seguida: os parâmetros são os escolhidos na tela"
    ),
}


def prosa_da_concordancia(estado: str) -> str:
    """A concordância legível, com a chave crua entre parênteses; estado novo aparece marcado."""
    prosa = PROSA_DA_CONCORDANCIA.get(estado)
    if prosa is None:
        return f"{estado} (**estado sem prosa nesta tela**)"
    return f"{prosa} ({estado})"


# ---------------------------------------------------------------------------
# A linha do valor: o número na unidade que o profissional lê
# ---------------------------------------------------------------------------
#
# O formato fica aqui, e não no painel, porque `desktop/janela/qualidade.py` tem um teste por
# árvore sintática que proíbe ali toda aritmética e todo especificador de formato com `%`.
# `prosa_do_numero` continua a ser outra política, com quatro casas, para as duas medianas do
# painel de antes-e-depois; este formatador é o da linha de valor das onze métricas.

# Vírgula decimal: a prosa da tela escreve '0,25 é 25%', e um número com ponto ao lado obrigaria
# quem lê a traduzir.
_SEPARADOR_DECIMAL: Final[str] = ","

# O texto de uma métrica calculada cujo valor é indefinido. Sem ele o painel mostraria `None`, o
# `null` em que pydantic serializa o `nan`. Não leva travessão, porque o travessão separa o número
# da população na linha composta.
SEM_VALOR_DEFINIDO: Final[str] = "indefinido (a métrica foi calculada e não há valor)"


def _numero_na_unidade(valor: Any, formato: FormatoDoValor) -> str:
    """O número sozinho, na unidade que `formato` declara.

    As guardas são sobre `valor`, que atravessou JSON, e não sobre `formato`, que vem da tabela.
    `nan` e `None` dão o mesmo texto, porque o `None` daqui é o `nan` serializado; a distinção entre
    não calculada e calculada e indefinida é feita antes, por `computed`. `inf` não colapsa com
    eles: é um valor, e numa razão como `qc_dispersion_ratio` diz que a dispersão das amostras é
    zero.
    """
    if valor is None:
        return SEM_VALOR_DEFINIDO
    if isinstance(valor, bool) or not isinstance(valor, (int, float)):
        # O JSON pode trazer texto onde se espera número. Mostrar o que veio, marcado, é melhor
        # que levantar dentro de um slot de Qt.
        return f"{valor} (**valor não numérico**)"
    numero = float(valor)
    if math.isnan(numero):
        return SEM_VALOR_DEFINIDO
    if math.isinf(numero):
        return "infinito negativo" if numero < 0 else "infinito"
    if formato == "contagem":
        return f"{numero:.0f}"
    cru = f"{numero:.1%}" if formato == "percentagem" else f"{numero:.3f}"
    return cru.replace(".", _SEPARADOR_DECIMAL)


def _populacao_do_vetor_de_rsd(relatorio: dict[str, Any]) -> str:
    """A população e os dois rótulos das agregações do vetor de RSD.

    A aba mostra RSD de três populações (o relatório de qualidade, o painel de antes-e-depois e o
    diagnóstico), e cada linha de valor diz qual é a sua. O denominador é o campo
    `n_features_rsd_defined`, que é `finitas.size` em `_fraction_under`; somar outras contagens à
    mão, como `n_features_saida`, daria o nome certo sobre outra população. Com denominador zero a
    frase diz que nenhuma feature tem RSD definido, em vez de escrever '0 de 0'.
    """
    # Os dois rótulos saem juntos ou nenhum, porque um sozinho não torna o número comparável, e
    # sem guarda a tela escreveria 'variante None'. Sem crases: o rótulo do valor no painel não é
    # Markdown, e quem lê opera o espectrômetro, não o código.
    variante = relatorio.get("rsd_variant")
    estimador = relatorio.get("rsd_estimator")
    rotulos = (
        prosa_dos_rotulos_do_rsd(variante, estimador)
        if isinstance(variante, str) and isinstance(estimador, str)
        else "variante e estimador não declarados"
    )
    definidas = relatorio.get("n_features_rsd_defined")
    if isinstance(definidas, bool) or not isinstance(definidas, int):
        # Alcançável: um `metrics_json` gravado por uma versão anterior não tem o campo, e a tela
        # declara a ausência em vez de inventar a população.
        return f"população não declarada nesta resposta; {rotulos}"
    if definidas == 0:
        return f"nenhuma feature tem RSD definido; {rotulos}"
    # Concordância no singular: 'sobre as 1 features' faria desconfiar do número ao lado, e uma
    # matriz com uma única feature de RSD definido é possível.
    artigo, substantivo = ("a", "feature") if definidas == 1 else ("as", "features")
    return f"sobre {artigo} {definidas} {substantivo} com RSD definido; {rotulos}"


def valor_da_metrica(nome: str, relatorio: dict[str, Any]) -> str:
    """A linha do valor de uma métrica calculada: número, população e rótulos.

    Só formata. Quem decide entre os estados, perguntando `computed` primeiro, é
    `janela/qualidade.py::texto_da_metrica`; trazer a decisão para cá daria um segundo caminho para
    o mesmo estado. O nome técnico não entra: ele vai no fim da legenda. Nome sem tradução aparece
    marcado, como em `rotulo_da_metrica`.
    """
    traduzida = TRADUCAO_DA_METRICA.get(nome)
    if traduzida is None:
        return f"{relatorio.get(nome)} (**métrica sem tradução nesta tela**)"
    numero = _numero_na_unidade(relatorio.get(nome), traduzida.formato)
    if traduzida.populacao == "nenhuma":
        return numero
    return f"{numero} — {_populacao_do_vetor_de_rsd(relatorio)}"


# ---------------------------------------------------------------------------
# Os modos sugeridos, e a prosa da confiança e da recomendação
# ---------------------------------------------------------------------------

# Os modos deste laboratório, oferecidos como sugestão e não como domínio fechado: outro estudo
# pode ter outros modos. O núcleo trata o modo como rótulo aberto (`load_matrix` recebe
# `mode: str`), e o limiar calibrado é procurado pelo arquivo (`mapa_calibrado_para`), não pelo
# rótulo: um estudo novo que escolha `pos_hilic` recebe a reserva escalar. Quem oferece esta tupla
# deve aceitar também um valor livre.
MODOS_SUGERIDOS: Final[tuple[str, ...]] = ("pos_hilic", "pos_c18", "neg_hilic", "neg_c18")


def prosa_da_confianca(valor: float | int | None, *, do_que: str) -> str:
    """A confiança em percentagem, com o que ela mede ao lado.

    `0.87` sozinho não diz confiança em quê, nem em que escala; `do_que` dá o referente (da
    classificação, da proposta), e a percentagem lê-se sem instruções.
    """
    if valor is None:
        return f"confiança d{do_que} não informada"
    return f"confiança d{do_que}: **{float(valor) * 100:.0f}%**"


def prosa_da_recomendacao(
    *, avaliavel: object, abstem: object, confianca: float | int | None
) -> str:
    """A confiança da recomendação em três estados.

    - `avaliavel=False`: não houve margem a medir;
    - `abstem=True`: a margem foi medida e é pequena demais para eleger um;
    - os dois falsos: a confiança em percentagem, com o referente.

    Margem não medida e margem minúscula levam à mesma ação e a contagens diferentes, e por isso são
    campos separados no esquema. O ramo de `avaliavel` vem primeiro porque uma política que se
    abstém por não ter medido satisfaz os dois. Os campos entram um a um, e não como o objeto da
    resposta, para que a leitura aconteça no chamador, onde o teste de contrato a liga a
    `/matrizes/recomendar`.
    """
    if not avaliavel:
        return (
            "a confiança **não foi avaliável**: a margem até o segundo colocado não pôde ser "
            "medida, o que é diferente de ter sido medida e dado zero."
        )
    if abstem:
        return (
            "a política **abstém-se** — a margem até o segundo colocado foi medida e é pequena "
            "demais para eleger um."
        )
    return prosa_da_confianca(confianca, do_que="a proposta") + "."


def prosa_da_medicao(
    *,
    vencedor: object,
    scores: object,
    margem: float | int | None,
    avaliavel: object,
    abstem: object,
    n_features_medidas: int | None,
    recusas: object = None,
) -> str:
    """A recomendação medida, em prosa: o que a medição achou depois de correr.

    `prosa_da_recomendacao` diz o que a política palpita antes; esta diz o que a medição achou.
    Os `scores` não levam '%': `metrics/rsd.py` devolve `sd/mean`, uma fração, e `0,1399` escrito
    como `13,99%` teria a unidade errada. A `margem` é relativa por construção,
    `(segundo - melhor) / |melhor|`, e vai em percentagem com 'relativa' e 'do melhor' ao lado,
    para não se ler como pontos percentuais de RSD.

    `n_features_medidas` é o recorte em que a comparação correu, e `None` significa a população
    inteira. `recusas` traz, por corretor, o motivo de ele ficar sem número (o lote menor não lhe
    deixa QCs de treino); o recusado sai da tabela mesmo que um número viesse junto, como sai da
    barra do painel (`plots.correcao.fila_do_fio`), e é dito por último. Sem recusa, a prosa não
    muda.
    """
    recusados = frase_das_recusas(recusas)
    nomes = _nomes_recusados(recusas)
    if nomes and isinstance(scores, dict):
        scores = {nome: v for nome, v in scores.items() if nome not in nomes}
    return (
        _prosa_da_medicao_sem_recusas(
            vencedor=vencedor,
            scores=scores,
            margem=margem,
            avaliavel=avaliavel,
            abstem=abstem,
            n_features_medidas=n_features_medidas,
            # 'Indefinido em todos os corretores' seria falso com um que nem foi medido.
            todos="todos os corretores medidos" if recusados else "todos os corretores",
        )
        + recusados
    )


def frase_das_recusas(recusas: object) -> str:
    """` O `loess` não foi medido: <motivo>.` por corretor recusado, ou vazio.

    O campo vem do corpo de uma resposta, e o `mypy` não o viu. Só um dicionário de texto para texto
    não vazio vira frase; qualquer outra forma conta como ausência de recusa.
    """
    if not isinstance(recusas, dict):
        return ""
    return "".join(
        f" O `{nome}` não foi medido: {motivo.strip()}."
        for nome, motivo in recusas.items()
        if isinstance(nome, str) and isinstance(motivo, str) and motivo.strip()
    )


def _nomes_recusados(recusas: object) -> frozenset[str]:
    """Os corretores que `frase_das_recusas` nomearia, pela mesma guarda."""
    if not isinstance(recusas, dict):
        return frozenset()
    return frozenset(
        nome
        for nome, motivo in recusas.items()
        if isinstance(nome, str) and isinstance(motivo, str) and motivo.strip()
    )


def _prosa_da_medicao_sem_recusas(
    *,
    vencedor: object,
    scores: object,
    margem: float | int | None,
    avaliavel: object,
    abstem: object,
    n_features_medidas: int | None,
    todos: str,
) -> str:
    """A prosa da eleição entre os escores medidos, sem a frase das recusas."""
    valores = scores if isinstance(scores, dict) else {}
    if not avaliavel or not valores:
        return (
            f"a medição **não foi avaliável**: o RSD ficou indefinido em {todos}, "
            "então não houve comparação a fazer."
        )
    ordenados = sorted(
        ((nome, v) for nome, v in valores.items() if isinstance(v, int | float) and v == v),
        key=lambda par: par[1],
    )
    tabela = " · ".join(f"`{nome}` {valor:.4f}".replace(".", ",") for nome, valor in ordenados)
    onde = (
        "sobre a população inteira de features"
        if n_features_medidas is None
        else f"sobre **{n_features_medidas}** features amostradas"
    )
    cabeca = (
        f"**{vencedor}** é o menor RSD de QC fora da amostra, {onde}: {tabela}. "
        "Menor é melhor — os valores são frações, não percentagens."
    )
    if margem is None:
        return (
            cabeca + " A **margem** até o segundo colocado não pôde ser medida, o que é diferente "
            "de ter sido medida e dado zero."
        )
    fim = f" A **margem relativa** até o segundo é de **{float(margem) * 100:.1f}%** do melhor"
    fim = fim.replace(".", ",")
    if abstem:
        return (
            cabeca + fim + ", pequena demais para eleger um: a escolha entre os dois primeiros "
            "não é sustentada pela medição."
        )
    return cabeca + fim + "."


def prosa_do_par(palpite: object, palpite_acertou: object, vencedor: object) -> str:
    """O par palpite e medição: o Palpite, medido em 200 features, contra a medição de todas.

    Três estados: `palpite_acertou` é `None` quando não houve palpite, e contá-lo como `False`
    lançaria como erro uma execução em que o Palpite nem foi consultado.
    """
    if palpite is None or palpite_acertou is None:
        return (
            "Não houve palpite a comparar: o Palpite desta matriz e destes lotes não tinha sido "
            "medido quando esta medição começou."
        )
    if palpite_acertou:
        return f"O palpite era `{palpite}` e a medição de todas as features **confirmou**."
    return (
        f"O palpite era `{palpite}` e a medição de todas as features achou **{vencedor}** — o "
        "palpite **errou**: a amostra de 200 features apontou outro corretor."
    )


def prosa_do_palpite_medido(
    *,
    avaliavel: object,
    abstem: object,
    confianca: float | int | None,
    margem: float | int | None,
    limiar: float | int | None,
    recusas: object = None,
) -> str:
    """A confiança do Palpite, nos três estados de `prosa_da_recomendacao`.

    Aqui não há política que se abstenha: há uma medição numa amostra de 200 features, e o empate
    diz que a amostra não separa os dois primeiros e que a saída é medir todas. Os campos entram um
    a um pelo motivo de `prosa_da_recomendacao`. `recusas` só muda o estado não avaliável, que passa
    a dizer 'todos os corretores medidos'; quem chama acrescenta o motivo do recusado com
    `frase_das_recusas`.
    """
    if not avaliavel:
        onde = (
            "em todos os corretores medidos" if _nomes_recusados(recusas) else "nos três corretores"
        )
        return (
            f"a confiança **não foi avaliável**: o RSD dos QCs ficou indefinido {onde}, e não "
            "houve o que comparar."
        )
    if abstem:
        if margem is None or limiar is None:
            # Dois motivos dão margem nula, e a frase vale para os dois: não houve segundo com RSD
            # definido, ou o menor RSD é zero e a margem relativa não se define.
            return (
                "**a amostra não decide**: a margem até o segundo colocado não se define — "
                "não houve segundo corretor com RSD definido, ou o menor RSD é zero. Para decidir, "
                "meça todas as features com «Medir qual corretor vence»."
            )
        margem_txt = f"{float(margem) * 100:.1f}".replace(".", ",")
        limiar_txt = f"{float(limiar) * 100:.1f}".replace(".", ",")
        return (
            f"**empate na amostra**: a margem relativa até o segundo colocado, {margem_txt}% do "
            f"melhor, ficou abaixo de {limiar_txt}%. Para decidir, meça todas as features com "
            "«Medir qual corretor vence»."
        )
    return prosa_da_confianca(confianca, do_que="a proposta") + "."


# 'Como o Palpite decide': o texto tem de bastar sozinho. Diz o que se mede, como se compara, o
# que é empate e o que fazer com ele, e traz um exemplo medido com a população ao lado.
#
# Os números do exemplo são os escores do Palpite nos arquivos de referência, gravados em
# `data/fixtures/medicoes/jh/f2_limiar.json`; `tests/test_ui_fluxo.py` confere cada um contra esse
# arquivo, nas duas direções. O 16,7% e as 200 features são literais porque este módulo não importa
# `refino.decision`, que traria o `sklearn`; o mesmo teste os confere contra
# `LIMIAR_DE_EMPATE_DO_PALPITE` e `N_FEATURES_DO_PALPITE`. A frase do limiar não diz 'duas vezes o
# desvio-padrão': com os vetores atuais essa conta dá 16,8%, e o produto fixou 16,7%.
COMO_O_PALPITE_DECIDE: Final = (
    "**Como o Palpite decide**\n\n"
    "1. Sorteia 200 features do arquivo — as mesmas para o mesmo arquivo e a mesma semente.\n"
    "2. Corrige essas features de três formas: sem correção, com o LOESS e com o SERRF.\n"
    "3. Em cada feature, mede o RSD dos QCs **fora da amostra**: cada QC é corrigido por um ajuste "
    "que não o usou, como se fosse uma amostra comum, e o RSD sai desses valores, sem os que a "
    "regra do boxplot marca como aberrantes. É a variação que sobra nos QCs depois da correção: "
    "quanto menor, mais estável ficou a régua.\n"
    "4. Resume cada forma pela **mediana** desse RSD nas 200 features, e propõe a de menor "
    "mediana.\n"
    "5. **Empate.** A margem é quanto o segundo colocado fica acima do primeiro, em proporção do "
    "primeiro. Abaixo de 16,7%, as 200 features não separam os dois: o Palpite continua a propor o "
    "primeiro, e «Seguir sugestões do app», na aba «Correção», o aplica, mas o selo diz «Empate na "
    "amostra». Para "
    "decidir com todas as features, use «Medir qual corretor vence».\n\n"
    "O limiar de 16,7% não foi escolhido à mão: ele vem do ruído do próprio sorteio, isto é, do "
    "quanto a margem muda quando só muda quais 200 features são sorteadas, medido nos quatro "
    "arquivos de referência.\n\n"
    "**Um exemplo medido em 04/11/2026**, nos arquivos de referência deste estudo, com 7 QCs "
    "cada:\n\n"
    "- `pos_c18` (200 das 2.572 features): SERRF 14,9%, sem correção 24,0%, LOESS 31,3%. O "
    "segundo colocado fica 60,8% acima do SERRF, bem acima de 16,7%: o Palpite propõe o SERRF, sem "
    "empate.\n"
    "- `neg_hilic` (200 das 2.039 features): LOESS 62,6%, SERRF 67,2%, sem correção 73,0%. O SERRF "
    "fica 7,3% acima do LOESS, abaixo de 16,7%: o Palpite propõe o LOESS e avisa que é empate.\n\n"
    "Nos dois arquivos, a medição de todas as features apontou o mesmo corretor que o Palpite."
)


# A frase da confiança quando o defeito é `unassessable`. Fica separada do rótulo para o teste poder
# afirmar a ausência do número sem medir também o texto do rótulo.
SEM_MEDICAO_DA_CLASSIFICACAO: Final = (
    "a confiança da classificação **não foi medida**: a métrica estrutural ficou indefinida ou "
    "instável, então não houve classificação a que atribuir confiança."
)


def prosa_da_confianca_do_diagnostico(defect_type: object, confidence: float | int | None) -> str:
    """A confiança do diagnóstico em três estados.

    - `defect_type == "unassessable"`: não houve medição, e nenhuma percentagem aparece;
    - `confidence is None`: não informada;
    - o resto: a percentagem, com o referente ao lado.

    `unassessable` é o desfecho da ausência de medida, e o classificador devolve `confidence=0.0`
    para ele (`diagnosis/profile.py`). Mostrar '0%' ao lado de um rótulo que diz que não houve
    medição afirmaria duas coisas opostas sobre a mesma quantidade.
    """
    if defect_type == "unassessable":
        return SEM_MEDICAO_DA_CLASSIFICACAO
    return prosa_da_confianca(confidence, do_que="a classificação")


# ---------------------------------------------------------------------------
# A justificativa de cada ponto de decisão, em prosa
# ---------------------------------------------------------------------------


def prosa_dos_lotes(lotes: dict[str, Any]) -> str:
    """A sugestão de lotes em prosa: quantos lotes, a injeção da fronteira e os QCs de cada um.

    O dicionário inteiro traria `rotulos_por_amostra`, com uma entrada por amostra; o que decide é
    o número de lotes e a injeção da fronteira. A fronteira só cai em posição de QC:
    `detect_changepoints` recebe `matrix.injection_orders[matrix.qc_mask]` e devolve elementos
    desse vetor.
    """
    n = lotes.get("n_lotes")
    cortes = lotes.get("changepoints") or []
    por_lote = lotes.get("qcs_por_lote") or []
    detectados = lotes.get("cortes_detectados") or []
    por_lote_detectado = lotes.get("qcs_por_lote_detectados") or []
    # O total sai da soma de `qcs_por_lote`, e não de um número fixo de QCs, que varia por estudo.
    # Sem o campo, a frase diz 'os QCs da corrida' em vez de quebrar.
    total = sum(por_lote) if por_lote else None
    quantos = f"{total} QCs" if total is not None else "os QCs da corrida"
    if not cortes and detectados:
        # A detecção achou cortes, e a proposta não os adotou: eles deixariam um lote com QC de
        # menos para medir fora da amostra. A frase diz o que foi achado, por que não entrou e
        # como declarar os lotes à mão, porque a pessoa pode saber que a corrida parou ali.
        onde_detectado = ", ".join(str(c) for c in detectados)
        deixariam = (
            "deixariam lotes com " + ", ".join(str(q) for q in por_lote_detectado) + " QC(s)"
            if por_lote_detectado
            else "deixariam algum lote com QCs de menos"
        )
        return (
            f"**Lotes sugeridos:** {n} — a detecção achou ponto de mudança na injeção "
            f"{onde_detectado}, mas esses cortes {deixariam}, e medir um corretor fora da "
            f"amostra pede ao menos {MIN_QC_PER_PROPOSED_BATCH} QCs em cada lote: um retido e "
            f"{MIN_QC_PER_PROPOSED_BATCH - 1} para ajustar. Por isso a sugestão é um **lote "
            f"único**, com {quantos}, e o Palpite é medido nele. Se a corrida teve mesmo uma "
            "interrupção ali, os lotes podem ser declarados à mão na aba «Correção», sabendo "
            "que cada um terá poucos QCs."
        )
    if not cortes:
        return (
            f"**Lotes sugeridos:** {n} — sem ponto de mudança detectado, "
            f"e {quantos} ficam num lote único."
        )
    onde = ", ".join(str(c) for c in cortes)
    # Os QCs por lote dizem se a sugestão é utilizável: a correção por lote se apoia nos QCs daquele
    # lote, e um lote com um QC não tem dispersão a medir.
    reparticao = (
        " Cada lote fica com " + ", ".join(str(q) for q in por_lote) + " QC(s), nessa ordem."
        if por_lote
        else ""
    )
    return (
        f"**Lotes sugeridos:** {n}, com fronteira na injeção {onde}.{reparticao} "
        f"É **sugestão** e não fato: vem de detecção de ponto de mudança sobre {quantos}. "
        "A fronteira só pode cair **numa posição de QC**, porque a detecção vê apenas a "
        "trajetória dos QCs e devolve uma das ordens de injeção deles."
    )


def prosa_do_defeito(defect_type: Any) -> str:
    """A classe do defeito em português, pela tabela `ROTULOS_DE_DEFEITO`.

    `erratic` e `gradual_drift` são identificadores de código, e em inglês num painel em português
    leem-se como jargão de sistema. Valor desconhecido aparece marcado; o teste de bijeção contra
    `get_args(DefectType)` o pega antes de chegar à tela.
    """
    if not defect_type:
        return "(a classe do defeito ainda não foi obtida)"
    traduzido = ROTULOS_DE_DEFEITO.get(str(defect_type))
    if traduzido is None:
        return f"`{defect_type}` (**rótulo sem tradução nesta tela**)"
    return traduzido


# Quem põe em prosa a justificativa de cada ponto. Um mapa, e não uma cadeia de `if`: ponto novo
# com justificativa em objeto entra numa linha, e quem esquecer cai no ramo que avisa.
FORMATADORES_DE_JUSTIFICATIVA: Final[dict[str, Callable[[dict[str, Any]], str]]] = {
    # O ponto do diagnóstico tem `campo_da_justificativa="lotes"`, o objeto com
    # `rotulos_por_amostra`, uma entrada por amostra.
    "diagnostico": prosa_dos_lotes,
}


def justificativa_em_prosa(chave: str, justificativa: Any) -> str:
    """A justificativa de um ponto de decisão em texto legível.

    - ausente, vazia ou `{}`: 'ainda não foi obtida', o estado normal antes do clique;
    - `str`: passa intacta (o ponto da recomendação tem `rationale: str`);
    - objeto com formatador em `FORMATADORES_DE_JUSTIFICATIVA`: a prosa do formatador;
    - objeto sem formatador: um aviso que nomeia o ponto, e não o objeto despejado.

    O último ramo evita `str(justificativa)`: o objeto do diagnóstico tem uma entrada por amostra,
    e um ponto novo sem formatador despejaria tudo sob 'Por quê:' sem nada reprovar. O aviso é a
    saída em que o esquecimento aparece.
    """
    if not justificativa:
        return "(a justificativa ainda não foi obtida)"
    if isinstance(justificativa, str):
        return justificativa
    formatador = FORMATADORES_DE_JUSTIFICATIVA.get(chave)
    if formatador is not None and isinstance(justificativa, dict):
        return formatador(justificativa)
    return (
        f"(a justificativa de «{chave}» veio como objeto e esta tela não sabe pô-la em prosa — "
        "o dado existe na resposta da API, e o que falta é o formatador)"
    )


# ===========================================================================
# O custo da grade do multiverso
# ===========================================================================
#
# O bloco é puro, dados medidos e duas funções sem estado, e fica aqui, fora de
# `desktop/janela/multiverso.py`, para ser usado sem importar PySide6.

# A grade de referência, derivada de `GridDefinition()`, a grade padrão: um literal aqui
# divergiria em silêncio no dia em que um eixo mudasse.
_GRADE_DE_REFERENCIA: Final = GridDefinition()

CORRETORES_DA_GRADE: Final[tuple[str, ...]] = tuple(_GRADE_DE_REFERENCIA.corrections)
"""Os corretores que a grade padrão percorre, na ordem dela."""

CELULAS_POR_CORRETOR: Final[int] = _GRADE_DE_REFERENCIA.size() // len(CORRETORES_DA_GRADE)
"""Quantas células a grade tem por corretor, as folhas de um prefixo.

Derivado por divisão porque `corrections` é o único eixo que o painel move: os outros são os
mesmos em todos os prefixos, então o quociente é exato e acompanha qualquer mudança da grade."""

# O custo por corretor, `(fixo por prefixo, marginal por célula)`, em segundos, da grade padrão
# sobre o `pos_hilic` (6.035 features x 68 amostras), como em
# `tests/integration/test_multiverse_benchmark.py`. É medido com o SERRF em série, porque
# `multiverse/executor.py` força `SerrfCorrector(n_jobs=1)`; um número medido com `n_jobs=-1`
# subestimaria a espera.
CUSTO_POR_CORRETOR: Final[dict[str, tuple[float, float]]] = {
    "none": (0.7, 0.01),
    "loess": (35.3, 0.03),
    "serrf": (1176.6, 0.47),
}
"""Duas execuções do mesmo arnês sobre o mesmo arquivo diferiram em 9%, e a faixa entre execuções
não foi medida: o par vale como ordem de grandeza, não como precisão."""


# O custo medido em cada um dos quatro modos, em série (`n_jobs=1`). É tabela e não fórmula porque
# escalar pelo número de features estaria errado: o `neg_c18` tem mais features que o `neg_hilic`
# (2.184 contra 2.039) e custo fixo menor (423,2 s contra 442,3 s). Sem o modo carregado, a
# estimativa é a faixa sobre os quatro arquivos.
#
# A rota do multiverso corre a correção em paralelo (`corrector_jobs=-1`), então a espera não é a
# coluna em série: `GANHO_DO_CORRETOR` converte o custo fixo. `GANHO_DAS_FOLHAS` e
# `GRADE_EM_PARALELO_S` (os dois extremos, `neg_hilic` e `pos_hilic`, cronometrados com os dois
# níveis de paralelismo) descrevem um regime que a rota não usa.
GANHO_DO_CORRETOR: Final = 5.60
"""O ganho do paralelismo dentro da correção, medido no `pos_c18` a 12 threads (491,3 s para
87,8 s); o joelho da curva está em 4 threads, com 3,50x."""

GANHO_DAS_FOLHAS: Final = 4.05
"""O ganho do paralelismo entre as células, medido no `pos_hilic` a 12 threads (500,3 s para
123,5 s), sobre uma grade sem `serrf`, cujo custo de raiz esconderia o efeito."""

GRADE_EM_PARALELO_S: Final[dict[str, float]] = {
    "menor": 134.12,
    "maior": 383.58,
}


CUSTO_MEDIDO_POR_MODO: Final[dict[str, dict[str, float]]] = {
    "neg_hilic": {
        "none": 0.7,
        "loess": 12.541,
        "serrf": 442.271,
        "marginal": 0.0633,
    },  # 2039 features
    "neg_c18": {
        "none": 0.7,
        "loess": 13.335,
        "serrf": 423.173,
        "marginal": 0.07853,
    },  # 2184 features
    "pos_c18": {
        "none": 0.7,
        "loess": 15.76,
        "serrf": 552.898,
        "marginal": 0.07939,
    },  # 2572 features
    "pos_hilic": {
        "none": 0.7,
        "loess": 35.327,
        "serrf": 1176.618,
        "marginal": 0.15958,
    },  # 6035 features
}

SEM_CORRETOR: Final = (
    "Escolha ao menos um corretor acima. Uma grade sem corretor nenhum não tem o que executar"
)
"""Uma frase, e não silêncio: botão que recusa sem dizer é indistinguível de botão morto."""


def estimativa_de_custo(
    corretores: Sequence[str], modo: str | None = None
) -> tuple[int, float, str]:
    """`(células, segundos, texto)` para a escolha atual; o texto declara a população.

    Com `modo`, a estimativa é o ponto medido daquele modo. Sem ele, ou com um modo que a tabela
    não mede, é a faixa sobre os quatro arquivos de referência: ausência de informação não vira um
    número inventado. A estimativa existe para o clique que dispara o multiverso ser informado; a
    execução também pode ser cancelada.

    O número é uma projeção sobre outras matrizes, e o texto diz sobre quais. Corretor sem medição
    conta com custo zero e vai nomeado no texto: não impede o disparo, e também não some da conta
    em silêncio.
    """
    celulas = CELULAS_POR_CORRETOR * len(corretores)
    sem_medicao = [nome for nome in corretores if nome not in CUSTO_POR_CORRETOR]

    if not corretores:
        return 0, 0.0, SEM_CORRETOR

    ressalva = ""
    if sem_medicao:
        ressalva = (
            f" NÃO há custo medido para {', '.join(sorted(sem_medicao))}, então a estimativa é "
            "menor que o real."
        )
    # A conta segue o regime da rota: `rotas.py` passa `n_jobs=1` e `corrector_jobs=-1`. Só o custo
    # fixo, que é a correção de raiz, é dividido pelo ganho do corretor; o marginal são as folhas,
    # e o executor as corre em sequência. Dividir também as folhas prometeria uma espera duas a
    # três vezes menor que a real. Ligar `n_jobs=-1` na rota mudaria o comportamento, e no binário
    # congelado o corretor paralelo abre processos com janela.
    #
    # `modo` desconhecido cai na faixa, pela mesma razão por que corretor desconhecido conta zero.
    do_modo = CUSTO_MEDIDO_POR_MODO.get(modo) if modo else None
    tabela = {modo: do_modo} if do_modo is not None else CUSTO_MEDIDO_POR_MODO

    totais = [
        sum(custo.get(n, 0.0) for n in corretores) / GANHO_DO_CORRETOR + custo["marginal"] * celulas
        for custo in tabela.values()
    ]
    menor, maior = min(totais), max(totais)

    # `segundos` é o extremo alto da faixa, no mesmo regime do texto: quem planeja uma espera
    # planeja pelo pior caso.
    segundos = maior

    # A frase nomeia o regime (correção em paralelo, grade em sequência), para que uma divergência
    # entre o texto e a fórmula seja legível sem abrir o código.
    if do_modo is not None:
        alcance = (
            f"**{_duracao(maior)}** para o `{modo}` — o modo que você carregou, "
            f"derivado da medição de 31/08/2026 nesta máquina, no regime que esta ferramenta "
            f"corre: a correção em paralelo e a grade em sequência"
        )
    else:
        alcance = (
            f"**{_duracao(menor)} a {_duracao(maior)}**, conforme o modo — derivado da medição "
            f"dos quatro arquivos do laboratório em 31/08/2026, nesta máquina, no regime que esta "
            f"ferramenta corre: a correção em paralelo e a grade em sequência. O menor é o "
            f"`neg_hilic` (2.039 features) e o maior o `pos_hilic` (6.035)"
        )
    return (
        celulas,
        segundos,
        f"{len(corretores)} corretor(es) x {CELULAS_POR_CORRETOR} = **{celulas} células**. "
        f"Estimativa: {alcance}.{ressalva} A execução pode ser cancelada a qualquer momento.",
    )


def _duracao(segundos: float) -> str:
    """Segundos em prosa; minutos acima de 90 s, porque '1.844 s' não se lê como meia hora."""
    if segundos < 90:
        return f"~{segundos:.0f} s"
    return f"~{segundos / 60:.0f} min"


# ===========================================================================
# A normalidade por variável e por grupo, e a escolha do teste do contraste
# ===========================================================================
#
# Fica aqui, e não em `desktop/janela/qualidade.py`, pelo motivo de `valor_da_metrica`: aquele
# módulo proíbe aritmética e formato percentual, e o resumo é feito de percentagens. A janela lê a
# resposta e passa as peças; o que se afirma fica aqui.

# Os nomes que o seletor do teste do contraste mostra, por valor de `UnivariateTest`; o valor que
# viaja é a chave (`ComboComNomes`). Não reusa `NOMES_DAS_OPCOES` porque lá `auto` é o
# autoescalonamento, e a mesma aba teria uma palavra com dois sentidos.
NOMES_DO_TESTE_UNIVARIADO: Final[dict[str, str]] = {
    "automatic": "Automático",
    "welch": "Welch",
    "mann_whitney": "Mann-Whitney",
}

DICA_DO_TESTE_UNIVARIADO: Final = (
    "O teste do contraste entre os dois grupos. «Automático» é a decisão de sempre: um "
    "Shapiro-Wilk sobre os resíduos de todas as features juntos escolhe um dos dois para a matriz "
    "inteira. «Welch» e «Mann-Whitney» forçam o teste. A tabela de normalidade por variável, em "
    "«Qualidade do resultado», apoia a escolha; ela não a faz."
)

TITULO_DA_NORMALIDADE: Final = "Normalidade por variável e por grupo"

# O limite vem antes dos números, na mesma moldura. O resumo diz quantas variáveis rejeitam a
# normal, e quem o lesse sem o limite concluiria que as outras são normais. Com grupos pequenos o
# teste tem pouco poder: com `normality_by_group`, em variáveis log-normais com desvio 0,5 no log,
# ele rejeita só 33% com 13 amostras e 46% com 18.
LIMITE_DA_NORMALIDADE: Final = (
    "Shapiro-Wilk por variável, dentro de cada grupo de estudo, sobre a matriz que as análises "
    "recebem: depois da correção, da imputação, da normalização e da transformação. O "
    "escalonamento não muda o resultado, porque o W não muda quando a variável é multiplicada e "
    "somada por constantes. O QC fica de fora: é a mistura de controle, e não um grupo.\n\n"
    "**Com poucas amostras por grupo o teste detecta só uma parte dos desvios moderados da normal, "
    "e «não rejeitou» não é «é normal».** Por isso o resumo diz quantas variáveis rejeitam, e não "
    "quantas são normais. As variáveis com valor imputado no grupo ficam à parte: os valores que a "
    "imputação põe derrubam o teste sozinhos.\n\n"
    "A tabela por variável — W, p, q, n, o motivo de não avaliável e o valor imputado — sai em "
    "CSV. O teste do contraste é escolhido em «Análise estatística», e o «Automático» não lê esta "
    "tabela."
)

# Os dois estados de ausência da seção pedem ações diferentes. Sem execução, a pessoa ainda não
# correu nada; com a execução e sem a família, ela desmarcou a caixa, e a ação é marcá-la. Uma frase
# só para os dois mandaria esperar por um resumo que nunca vem.
NORMALIDADE_EM_REPOUSO: Final = (
    "As análises ainda não correram: o resumo por grupo aparece aqui depois da execução."
)
NORMALIDADE_NAO_PEDIDA: Final = (
    "Esta execução não pediu a normalidade. Para a ver, marque «Normalidade por grupo» em "
    "«Análises a correr» e execute de novo."
)

ROTULO_DA_TABELA_DA_NORMALIDADE: Final = "Salvar a tabela por variável (CSV)…"
NOME_DA_TABELA_DA_NORMALIDADE: Final = "normalidade-por-variavel-e-grupo.csv"

# As colunas do arquivo, na ordem em que saem. `nao_avaliavel` traz o código do motivo, o mesmo da
# resposta da API; `valor_imputado` é 'sim' ou 'não', porque é a coluna que a pessoa vai filtrar.
COLUNAS_DA_TABELA_DA_NORMALIDADE: Final[tuple[str, ...]] = (
    "grupo",
    "feature",
    "n",
    "w",
    "p",
    "q",
    "nao_avaliavel",
    "valor_imputado",
)


def _inteiro(valor: Any) -> str:
    """Um inteiro com o ponto de milhar, ou a ausência por extenso, nunca `None`."""
    if isinstance(valor, bool) or not isinstance(valor, int):
        return "não declarado"
    return f"{valor:,}".replace(",", ".")


def _percentagem(valor: Any) -> str:
    """Uma fração como percentagem com vírgula decimal: `0.0862` → `"8,6%"`, `0.05` → `"5%"`.

    O `,0` sai, porque '5,0%' ao lado de 'os 5% esperados' lê-se como dois números.
    """
    if isinstance(valor, bool) or not isinstance(valor, (int, float)) or math.isnan(valor):
        return "não declarado"
    texto = f"{100.0 * float(valor):.1f}".replace(".", ",")
    return (texto[:-2] if texto.endswith(",0") else texto) + "%"


def _decimal_curto(valor: Any) -> str:
    """Um nível ou um λ como a tela os escreve: `0.05` → `"0,05"`."""
    if isinstance(valor, bool) or not isinstance(valor, (int, float)):
        return "não declarado"
    return f"{float(valor):g}".replace(".", ",")


def _contada(quantas: Any, *, nenhuma: str, uma: str, varias: str) -> str:
    """A concordância dos três ramos: `0` → `nenhuma`, `1` → `uma`, e o número antes do plural.

    '0 rejeitam' e '1 constantes' fazem desconfiar do número ao lado, como em `frase_contada`, de
    `plots/recorte.py`.
    """
    if quantas == 0:
        return nenhuma
    if quantas == 1:
        return uma
    return f"{_inteiro(quantas)} {varias}"


def _a_parte_do_grupo(grupo: dict[str, Any]) -> str:
    """O que ficou fora do resumo do grupo: as imputadas e as não avaliáveis, só as que existem."""
    partes: list[str] = []
    imputadas = grupo.get("n_com_imputado")
    if imputadas:
        rejeitam = _contada(
            grupo.get("n_com_imputado_rejeitadas"),
            nenhuma="nenhuma rejeita a normal",
            uma="1 rejeita a normal",
            varias="rejeitam a normal",
        )
        partes.append(
            _contada(
                imputadas,
                nenhuma="",
                uma="1 variável com valor imputado no grupo",
                varias="variáveis com valor imputado no grupo",
            )
            + f" — {rejeitam}, e os valores que a imputação pôs bastam para isso"
        )
    poucos = grupo.get("n_menos_de_tres")
    if poucos:
        partes.append(
            _contada(
                poucos,
                nenhuma="",
                uma="1 com menos de 3 valores",
                varias="com menos de 3 valores",
            )
        )
    constantes = grupo.get("n_constantes")
    if constantes:
        partes.append(
            _contada(
                constantes,
                nenhuma="",
                uma="1 constante no grupo",
                varias="constantes no grupo",
            )
        )
    if not partes:
        return "Nenhuma variável ficou à parte."
    return f"À parte: {'; '.join(partes)}."


def prosa_dos_grupos_da_normalidade(
    grupos: Any, *, alfa: Any, nivel_do_fdr: Any, lambda_de_storey: Any
) -> str:
    """O resumo de cada grupo, um parágrafo por grupo, em markdown.

    Os números chegam prontos na resposta (`NormalidadeDoGrupo`), contados no núcleo dos mesmos
    vetores que vão para o CSV; aqui só se escreve. O π̂₀ nunca aparece como fração normal: a frase
    usa `1 - π̂₀` como piso do que não segue a normal, e quando ele é zero diz que isso não torna as
    outras normais. Sem grupo, a frase diz que a matriz não tem amostra de estudo.
    """
    if not isinstance(grupos, list) or not grupos:
        return (
            "Nenhum grupo de estudo nesta matriz: o QC não conta como grupo, e sem amostra de "
            "estudo não há variável a testar."
        )
    storey = f"1 - π̂₀ de Storey, λ = {_decimal_curto(lambda_de_storey)}"
    paragrafos: list[str] = []
    for grupo in grupos:
        if not isinstance(grupo, dict):
            continue
        cabeca = f"**{grupo.get('grupo')}** ({_inteiro(grupo.get('n_amostras'))} amostras):"
        testadas = grupo.get("n_testadas")
        if testadas == 0:
            corpo = "nenhuma variável testada — nenhuma é avaliável sem valor imputado no grupo."
        else:
            piso = grupo.get("fracao_minima_nao_normal")
            estimativa = (
                f"Pelo menos {_percentagem(piso)} não seguem a normal ({storey})."
                if isinstance(piso, (int, float)) and piso > 0
                else (
                    f"A estimativa conservadora da fração que não segue a normal é zero "
                    f"({storey}) — o que não torna as outras normais: o teste pode não ter tido "
                    "poder para as ver."
                )
            )
            rejeitam = _contada(
                grupo.get("n_rejeitadas"),
                nenhuma="Nenhuma rejeita",
                uma="1 rejeita",
                varias="rejeitam",
            )
            no_fdr = _contada(
                grupo.get("n_rejeitadas_fdr"),
                nenhuma="nenhuma fica",
                uma="1 fica",
                varias="ficam",
            )
            corpo = (
                f"testadas {_inteiro(testadas)} das {_inteiro(grupo.get('n_variaveis'))} "
                "variáveis — as avaliáveis sem valor imputado no grupo. "
                f"{rejeitam} a normal a {_decimal_curto(alfa)} "
                f"({_percentagem(grupo.get('fracao_rejeitada'))}, contra os "
                f"{_percentagem(alfa)} que o acaso daria se todas a seguissem), e {no_fdr} com q "
                f"de Benjamini-Hochberg abaixo de {_decimal_curto(nivel_do_fdr)}. {estimativa}"
            )
        paragrafos.append(f"{cabeca} {corpo} {_a_parte_do_grupo(grupo)}")
    return "\n\n".join(paragrafos)


def _celula(valor: Any) -> str:
    """Um número da tabela como texto de CSV: a forma exata do `float`, e vazio onde não há."""
    if valor is None or isinstance(valor, bool):
        return ""
    if isinstance(valor, int):
        return str(valor)
    if isinstance(valor, float):
        return repr(valor) if math.isfinite(valor) else ""
    return str(valor)


def csv_da_normalidade(features: Any, grupos: Any) -> str:
    """A tabela por variável e por grupo, como texto de CSV.

    Uma linha por (grupo, variável), na ordem da resposta. Célula vazia onde o número não existe
    (o motivo está na coluna ao lado), e nunca `0`, que se leria como medida. O `float` sai na forma
    exata de `repr`, com ponto decimal, porque o arquivo é para planilha e para programa. O fim de
    linha é `\\r\\n`, como em `io/writers.write_removidas`, que o Excel lê sem juntar linhas.
    """
    nomes = features if isinstance(features, list) else []
    saida = io.StringIO()
    escritor = csv.writer(saida, lineterminator="\r\n")
    escritor.writerow(COLUNAS_DA_TABELA_DA_NORMALIDADE)
    for grupo in grupos if isinstance(grupos, list) else []:
        if not isinstance(grupo, dict):
            continue
        colunas = [
            grupo.get(chave) or [None] * len(nomes)
            for chave in ("n_valores", "w", "pvalue", "qvalue", "nao_avaliavel", "valor_imputado")
        ]
        for nome, n, w, p, q, motivo, imputado in zip(nomes, *colunas, strict=True):
            escritor.writerow(
                [
                    grupo.get("grupo"),
                    nome,
                    _celula(n),
                    _celula(w),
                    _celula(p),
                    _celula(q),
                    motivo or "",
                    "sim" if imputado else "não",
                ]
            )
    return saida.getvalue()
