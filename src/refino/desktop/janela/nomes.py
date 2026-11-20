"""Os nomes que a tela mostra para os valores técnicos.

Quem opera num laboratório não lê `serrf`, `one_fifth_min` ou `mean_center`; esta tabela dá a cada
valor um nome na língua de quem opera. O valor não muda: cada seletor continua a guardar o valor
técnico como texto do item, `currentText()` o devolve e o pedido à API o leva. `ComboComNomes`, em
`componentes.py`, só pinta o nome por cima. Valor sem nome aqui aparece como veio, para nunca
esconder uma opção.

Os nomes descrevem o que o código faz, e não o nome do valor: `knn` procura amostras vizinhas e
`knn_features`, features vizinhas (`preprocessing/missing.py`); `nrsd` é MAD sobre mediana
(`preprocessing/filtering.py`); `sum` divide pela soma da amostra e multiplica por mil
(`preprocessing/normalization.py`). As figuras de PCA e as de 'um contra resto' do PLS-DA
multiclasse levam a análise no nome.
"""

from __future__ import annotations

from typing import Final

from refino.desktop.janela.leitura import NOMES_DOS_CORRETORES

__all__ = [
    "NOMES_DAS_FIGURAS",
    "NOMES_DAS_OPCOES",
    "NOMES_DOS_CORRETORES",
]

# Os nomes das opções dos seletores de pré-processamento e da variante do RSD, por valor. Um
# dicionário só, porque os valores repetidos entre eixos (`none`, `median`, `mean`) dizem a mesma
# coisa em todos. `none` é 'Não aplicar', e não 'Nenhuma': serve aos seis eixos sem concordância
# de gênero e diz o que acontece, que é saltar a etapa.
NOMES_DAS_OPCOES: Final[dict[str, str]] = {
    # Redução por posto
    "iqr": "Amplitude interquartil (IQR)",
    "sd": "Desvio-padrão",
    "mad": "Desvio absoluto mediano (MAD)",
    "rsd": "Desvio-padrão relativo (RSD)",
    "nrsd": "RSD não paramétrico (MAD/mediana)",
    "mean": "Média",
    "median": "Mediana",
    # Imputação de ausentes
    "one_fifth_min": "1/5 do menor valor positivo",
    "knn": "KNN — amostras vizinhas",
    "knn_features": "KNN — features vizinhas",
    "col_min": "Menor valor positivo",
    "exclude": "Excluir features com ausentes",
    "missforest": "MissForest",
    # Normalização por amostra
    "sum": "Pela soma",
    "pqn_group": "PQN — referência do grupo",
    "sample_reference": "Pela amostra de referência",
    "feature_reference": "Pela feature de referência",
    "supplied_factors": "Pelos fatores fornecidos",
    "quantile": "Por quantis",
    # Transformação
    "log10": "Log₁₀",
    "log2": "Log₂",
    "sqrt": "Raiz quadrada",
    "cbrt": "Raiz cúbica",
    # Escalonamento
    "mean_center": "Centrar na média",
    "auto": "Autoescalonamento",
    "pareto": "Pareto",
    "range": "Por amplitude",
    "vast": "VAST",
    # Variante do RSD dos QCs
    "plain": "Com todos os QCs",
    "outlier_removed": "Sem o QC aberrante",
    "none": "Não aplicar",
}

# Os nomes das figuras do seletor da aba 'Gráficos', pelo rótulo derivado que o registo lhes dá
# (`plots/catalogo.rotulo_do_construtor`). O derivado continua a ser o valor, e dele sai, sem
# acento de propósito, o nome de arquivo sugerido.
NOMES_DAS_FIGURAS: Final[dict[str, str]] = {
    "deriva do qc": "Deriva dos QCs",
    "caixas por feature": "Caixas por feature",
    "caixas por amostra": "Caixas por amostra",
    "densidade por feature": "Densidade por feature",
    "densidade por amostra": "Densidade por amostra",
    "volcano": "Volcano",
    "fold change": "Fold change",
    "testes t": "Testes t",
    "anova": "ANOVA",
    "scree": "PCA — variância por componente",
    "scores": "PCA — scores",
    "cargas": "PCA — cargas",
    "biplot": "PCA — biplot",
    "scores do plsda": "PLS-DA — scores",
    "cargas do plsda": "PLS-DA — cargas",
    "vip": "PLS-DA — VIP",
    "escores um contra resto": "PLS-DA multiclasse — scores",
    "cargas um contra resto": "PLS-DA multiclasse — cargas",
    "importancia um contra resto": "PLS-DA multiclasse — importância",
    "dendrograma": "Dendrograma",
    "mapa de calor": "Mapa de calor",
    "kmeans": "K-means",
    "mapa auto organizavel": "Mapa auto-organizável",
    "mapa de correlacao": "Mapa de correlação",
    "busca por padrao": "Busca por padrão",
    "random forest": "Random forest",
    "svm": "SVM",
    "rede de correlacao parcial": "Rede de correlação parcial",
    "painel da correcao": "Painel da correção",
    "violino do rsd por feature": "Violino do RSD por feature",
    "composicao por classe": "Composição por classe",
    # A curva do LOESS ao longo das injeções, com o envelope sem um QC.
    "curva do loess": "Curva do LOESS",
}
