"""`LearnedPolicy`: duas árvores rasas, uma para o corretor e outra para a confiança.

A medição fora da amostra sempre elege um vencedor, então o vencedor sozinho nunca diz quando não
confiar na escolha. O sinal de abstenção é a margem entre o melhor e o segundo. A margem observada
só existe depois de medir os três corretores, que é o trabalho que a política evita, e por isso
uma segunda árvore a prediz das características observáveis:

- classificação: qual corretor, em `Recommendation.correction`;
- regressão: a margem predita, que vira `Recommendation.confidence`.

As árvores têm profundidade 3 para que o caminho de decisão sirva de justificativa: são as
características e os limiares que de fato decidiram. Um ganho pequeno de acurácia não compensa
perder essa explicação.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from refino.decision.features import FEATURE_NAMES, feature_matrix, feature_vector
from refino.decision.policy import (
    ABSTENTION_MARGIN,
    ORIGEM_DO_LIMIAR,
    Recommendation,
    confidence_from_margin,
)
from refino.domain.diagnosis import DefectProfile
from refino.domain.labels import Correction

__all__ = [
    "CAMINHO_DE_FOLHA_SO",
    "MAX_DEPTH",
    "LearnedPolicy",
    "descreve_caminho",
    "monta_recomendacao",
]

# Profundidade das duas árvores. Três níveis dão no máximo 8 folhas e um caminho de decisão de três
# condições, que cabe numa justificativa legível. Não é hiperparâmetro a otimizar.
MAX_DEPTH = 3

CAMINHO_DE_FOLHA_SO = (
    "a árvore é uma folha só — o corpus de treino não separou nenhuma característica"
)


def descreve_caminho(passos: Sequence[tuple[int, float, float]]) -> str:
    """O caminho de decisão em texto, a partir de `(índice, valor, limiar)` por nó visitado.

    É função de módulo porque duas políticas a chamam: a que tem as árvores do `scikit-learn` em
    memória e a que as lê do artefato. O texto vai para a justificativa mostrada ao usuário e tem
    de ser o mesmo nas duas.
    """
    if not passos:
        return CAMINHO_DE_FOLHA_SO
    return "; ".join(
        f"`{FEATURE_NAMES[indice]}` = {valor:.4f} {'≤' if valor <= limiar else '>'} {limiar:.4f}"
        for indice, valor, limiar in passos
    )


def monta_recomendacao(
    *,
    correcao: Correction,
    margem: float,
    threshold: float,
    caminho: str,
    profile: DefectProfile,
) -> Recommendation:
    """Monta a `Recommendation` da política aprendida, com a justificativa em texto.

    `assessable` é falso quando a margem é `nan`. Margem não medida e margem zero levam à mesma
    abstenção, mas são contadas em separado.
    """
    confianca = confidence_from_margin(margem, threshold=threshold)
    veredito = "abstenção" if confianca < 0.5 else "confiança"
    razao = (
        f"Recomendado `{correcao}` com {veredito} de {confianca:.2f}. **Por quê:** "
        f"{caminho}. A margem relativa PREDITA entre o melhor e o segundo corretor é "
        f"{margem:.4f} do melhor, contra o limiar de {threshold:.4f}, {ORIGEM_DO_LIMIAR}. O RSD "
        f"mediano de QC é {profile.qc_rsd_median:.4f} e a correlação de "
        f"PC1 com a ordem de injeção é {profile.pc1_injection_correlation:.4f}."
    )
    return Recommendation(
        correction=correcao,
        confidence=confianca,
        rationale=razao,
        evidence={
            "predicted_margin": margem,
            "abstention_threshold": threshold,
            "qc_rsd_median": profile.qc_rsd_median,
            "pc1_injection_correlation": profile.pc1_injection_correlation,
        },
        assessable=margem == margem,
    )


@dataclass
class LearnedPolicy:
    """Política aprendida sobre o corpus rotulado, com a justificativa tirada do caminho de decisão.

    É determinística dada a semente: as duas árvores recebem `random_state=seed`, e a árvore do
    `scikit-learn` só sorteia no desempate entre divisões de igual ganho.
    """

    seed: int = 42
    threshold: float = ABSTENTION_MARGIN
    max_depth: int = MAX_DEPTH
    name: str = "learned"
    _corretor: Any = field(default=None, repr=False)
    _margem: Any = field(default=None, repr=False)

    @property
    def fitted(self) -> bool:
        return self._corretor is not None

    def fit(
        self,
        profiles: Sequence[DefectProfile],
        labels: Sequence[Correction],
        margins: Sequence[float],
    ) -> "LearnedPolicy":
        """Ajusta as duas árvores. Levanta com corpus vazio ou tamanhos discordantes.

        Não devolve política sem ajuste em silêncio, porque ela seria indistinguível de uma
        treinada.
        """
        from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

        if not profiles:
            raise ValueError("corpus vazio: não há o que treinar")
        if not (len(profiles) == len(labels) == len(margins)):
            raise ValueError(
                f"tamanhos discordantes: {len(profiles)} perfis, {len(labels)} rótulos, "
                f"{len(margins)} margens"
            )

        x = feature_matrix(profiles)
        y = np.asarray(labels, dtype=object)
        m = np.asarray(margins, dtype=np.float64)
        finitas = np.isfinite(m)

        self._corretor = DecisionTreeClassifier(
            max_depth=self.max_depth, random_state=self.seed
        ).fit(x, y)

        # Margem não finita fica fora do treino da regressão em vez de virar zero: não medida não
        # é o mesmo que nula. Se nenhuma for finita, não há árvore de margem e a política se
        # abstém sempre.
        self._margem = (
            DecisionTreeRegressor(max_depth=self.max_depth, random_state=self.seed).fit(
                x[finitas], m[finitas]
            )
            if finitas.any()
            else None
        )
        return self

    def recommend(self, profile: DefectProfile) -> Recommendation:
        if not self.fitted:
            raise RuntimeError(
                "`LearnedPolicy` não foi ajustada. Recomendar sem treino devolveria um padrão "
                "indistinguível de decisão, e a comparação com a política constante seria contra "
                "ele"
            )
        x = feature_vector(profile).reshape(1, -1)
        correcao: Correction = self._corretor.predict(x)[0]
        margem = float(self._margem.predict(x)[0]) if self._margem is not None else float("nan")
        return monta_recomendacao(
            correcao=correcao,
            margem=margem,
            threshold=self.threshold,
            caminho=self._caminho(x),
            profile=profile,
        )

    def predict_margin(self, profile: DefectProfile) -> float:
        """A margem predita para o perfil.

        É exposta para que as políticas constantes usem o mesmo estimador de confiança. Com
        estimadores diferentes, a comparação misturaria escolha de corretor com calibração.
        """
        if self._margem is None:
            return float("nan")
        return float(self._margem.predict(feature_vector(profile).reshape(1, -1))[0])

    def _caminho(self, x: np.ndarray) -> str:
        """O caminho de decisão da árvore de classificação, em texto.

        São as condições que a árvore avaliou para este perfil, na ordem em que as avaliou.
        """
        arvore = self._corretor.tree_
        no = 0
        passos: list[tuple[int, float, float]] = []
        while arvore.children_left[no] != -1:
            indice = int(arvore.feature[no])
            limiar = float(arvore.threshold[no])
            valor = float(x[0, indice])
            passos.append((indice, valor, limiar))
            no = arvore.children_left[no] if valor <= limiar else arvore.children_right[no]
        return descreve_caminho(passos)
