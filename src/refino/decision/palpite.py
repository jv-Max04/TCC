"""O Palpite medido: o corretor com o menor RSD dos QCs fora da amostra em 200 features sorteadas.

É a conta do `LabelingOracle` com `n_sampled_features = 200`: os três corretores, o RSD dos QCs
fora da amostra, a mediana por corretor nas features sorteadas, e o menor vence. O RSD de cada
feature sorteada é o mesmo que a medição da matriz inteira daria; o que muda é a mediana, tirada
de uma amostra. A mesma matriz e a mesma semente dão as mesmas features e o mesmo Palpite. Matriz
com 200 features ou menos é medida inteira, e `n_features_medidas` diz quantas foram.

O limiar de empate é próprio. O `ABSTENTION_MARGIN` da política vem do ruído da semente dos
corretores sobre todas as features; aqui o ruído é o do sorteio da amostra, que é maior. Abaixo
do limiar o Palpite se abstém, e a proposta continua a ser o vencedor da amostra. Trocá-lo pelo
SERRF no empate foi testado e aumentou o arrependimento médio.

Um corretor que o oráculo recusa por falta de QCs num lote fica fora da eleição e da evidência, e
a justificativa diz que ele não foi medido e por quê.

A árvore de `PoliticaDoArtefato` continua no código, como a política de `--auto aprendida`. O
Palpite que as telas mostram é esta medição. Núcleo puro: recebe a matriz em memória e devolve o
Palpite.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Final, cast

import numpy as np

from refino.decision.labeling import LabelingOracle, PrefixKey
from refino.decision.policy import Recommendation, confidence_from_margin
from refino.domain.cancellation import CancelCheck
from refino.domain.labels import Correction
from refino.domain.matrix import RefinoMatrix

__all__ = [
    "LIMIAR_DE_EMPATE_DO_PALPITE",
    "N_FEATURES_DO_PALPITE",
    "ORIGEM_DO_LIMIAR_DO_PALPITE",
    "PalpiteMedido",
    "decide_o_palpite",
    "mede_o_palpite",
]

# O custo da medição é linear neste número, e a concordância com o vencedor da matriz inteira sobe
# devagar acima dele.
N_FEATURES_DO_PALPITE: Final = 200

# Duas vezes o maior desvio-padrão, entre os quatro arquivos de referência, da margem entre o
# vencedor e o segundo quando só muda o sorteio das 200 features: 0,167027, arredondado.
LIMIAR_DE_EMPATE_DO_PALPITE: Final = 0.167

ORIGEM_DO_LIMIAR_DO_PALPITE: Final = (
    "duas vezes o maior desvio-padrão, entre os quatro arquivos reais, da margem relativa entre o "
    "vencedor e o segundo da população inteira quando só muda o sorteio de 200 features (2.000 "
    "sorteios por modo): 0,167027, com o máximo no pos_hilic"
)

_ORDEM: Final[tuple[Correction, ...]] = ("none", "loess", "serrf")


@dataclass(frozen=True)
class PalpiteMedido:
    """O que a medição na amostra achou e o que ela propõe.

    `margem` é `None` quando a margem relativa não se define: sem segundo corretor com escore
    finito (`segundo` também é `None`) ou com o menor escore igual a zero (`segundo` diz quem era).
    `avaliavel` é falso quando nenhum corretor teve escore finito. Os três casos levam à abstenção
    e ficam distinguíveis para contagem.

    `recusas` traz, pelo nome do corretor, o motivo de ele não ter sido medido. O escore dele é
    `nan`, como o do RSD indefinido, e é `recusas` que separa os dois.
    """

    correction: Correction
    escores: Mapping[str, float]
    margem: float | None
    avaliavel: bool
    n_features_medidas: int
    limiar: float = LIMIAR_DE_EMPATE_DO_PALPITE
    segundo: Correction | None = None
    evidencia_extra: Mapping[str, float] = field(default_factory=dict)
    recusas: Mapping[str, str] = field(default_factory=dict)

    @property
    def confianca(self) -> float:
        """A confiança de `confidence_from_margin` com o limiar do Palpite: 0,5 no limiar."""
        if not self.avaliavel or self.margem is None:
            return 0.0
        return confidence_from_margin(self.margem, threshold=self.limiar)

    @property
    def abstem(self) -> bool:
        """Abstém-se quando a margem da amostra não chega ao limiar ou quando não houve medição."""
        return self.confianca < 0.5

    @property
    def rationale(self) -> str:
        """A justificativa em texto, com os números que a sustentam."""
        if not self.avaliavel:
            if not self.recusas:
                return (
                    "o RSD dos QCs ficou indefinido nos três corretores, nas "
                    f"{self.n_features_medidas} features medidas: não houve o que comparar"
                )
            medidos = ", ".join(c for c in _ORDEM if c in self.escores and c not in self.recusas)
            return (
                f"o RSD dos QCs ficou indefinido em {medidos}, nas {self.n_features_medidas} "
                f"features medidas, e não houve o que comparar{self._frase_das_recusas()}"
            )
        # O corretor recusado fica fora da lista de escores, como fica fora da eleição: a frase
        # seguinte diz que ele não foi medido.
        escores = ", ".join(
            f"{c} {self.escores[c]:.4f}".replace(".", ",")
            for c in _ORDEM
            if c not in self.recusas and math.isfinite(self.escores.get(c, math.nan))
        )
        base = (
            f"medido em {self.n_features_medidas} features: o menor RSD de QC fora da amostra é o "
            f"de {self.correction} ({escores})"
        )
        base += self._frase_das_recusas()
        if self.segundo is None:
            return base + "; não houve segundo corretor com escore finito"
        if self.margem is None:
            return (
                base + f"; o menor escore é zero, e a margem relativa para o segundo, "
                f"{self.segundo}, não se define"
            )
        cauda = (
            f"; a margem relativa para o segundo, {self.segundo}, é "
            f"{self.margem * 100:.1f}% do melhor"
        ).replace(".", ",")
        if self.abstem:
            limiar = f"{self.limiar * 100:.1f}%".replace(".", ",")
            return (
                base
                + cauda
                + f", abaixo do limiar de empate de {limiar}: a amostra não separa os dois"
            )
        return base + cauda

    def _frase_das_recusas(self) -> str:
        """`; o loess não foi medido: <motivo>` para cada corretor recusado, na ordem do oráculo."""
        return "".join(
            f"; o {c} não foi medido: {self.recusas[c]}" for c in _ORDEM if c in self.recusas
        )

    @property
    def evidencia(self) -> dict[str, float]:
        """Os números que a justificativa cita, por nome, para o registro da decisão.

        O corretor recusado fica de fora, porque nunca foi medido. O de RSD indefinido entra como
        `nan`.
        """
        evidencia = {f"rsd_{c}": float(v) for c, v in self.escores.items() if c not in self.recusas}
        evidencia["n_features_medidas"] = float(self.n_features_medidas)
        evidencia["limiar_de_empate"] = float(self.limiar)
        if self.margem is not None:
            evidencia["margem"] = float(self.margem)
        evidencia.update(self.evidencia_extra)
        return evidencia

    def como_recomendacao(self) -> Recommendation:
        """O Palpite como `Recommendation`, a forma que a delegação e o registro da decisão leem."""
        return Recommendation(
            correction=self.correction,
            confidence=self.confianca,
            rationale=self.rationale,
            evidence=self.evidencia,
            assessable=self.avaliavel,
        )


def decide_o_palpite(
    escores: Mapping[str, float],
    *,
    n_features_medidas: int,
    limiar: float = LIMIAR_DE_EMPATE_DO_PALPITE,
    recusas: Mapping[str, str] | None = None,
) -> PalpiteMedido:
    """O vencedor da amostra e a margem para o segundo. Função pura: não mede nada.

    É separada de `mede_o_palpite` para que a regra seja testável sem rodar corretor. Empate exato
    entre dois escores resolve pela ordem `none`, `loess`, `serrf`, a mesma de
    `LabelingOracle.prefixes`.

    `recusas` é o motivo, por corretor, de ele não ter sido medido. O recusado fica fora da
    eleição e da margem mesmo que venha com escore finito.
    """
    recusados = dict(recusas or {})
    finitos = [
        (c, float(escores[c]))
        for c in _ORDEM
        if c in escores and c not in recusados and math.isfinite(escores[c])
    ]
    if not finitos:
        # O primeiro na ordem declarada, como o oráculo devolve quando nada é avaliável. Quem diz
        # que não houve escolha é `abstem`.
        return PalpiteMedido(
            correction=_ORDEM[0],
            escores=dict(escores),
            margem=None,
            avaliavel=False,
            n_features_medidas=n_features_medidas,
            limiar=limiar,
            recusas=recusados,
        )
    ordenados = sorted(finitos, key=lambda par: par[1])
    melhor, valor = ordenados[0]
    segundo = ordenados[1][0] if len(ordenados) > 1 else None
    if segundo is None or valor == 0.0:
        return PalpiteMedido(
            correction=melhor,
            escores=dict(escores),
            margem=None,
            avaliavel=True,
            n_features_medidas=n_features_medidas,
            limiar=limiar,
            segundo=segundo,
            recusas=recusados,
        )
    valor2 = ordenados[1][1]
    return PalpiteMedido(
        correction=melhor,
        escores=dict(escores),
        margem=float((valor2 - valor) / abs(valor)),
        avaliavel=True,
        n_features_medidas=n_features_medidas,
        limiar=limiar,
        segundo=segundo,
        recusas=recusados,
    )


def mede_o_palpite(
    matriz: RefinoMatrix,
    *,
    batches: np.ndarray | None = None,
    n_jobs: int = -1,
    depois_de_cada: Callable[[PrefixKey], None] | None = None,
    cancelled: CancelCheck | None = None,
    seed: int = 42,
) -> PalpiteMedido:
    """Mede o Palpite: o oráculo de rotulagem em 200 features sorteadas e a regra de decisão.

    `batches` é o vetor de lote alinhado às amostras, ou `None` para lote único, como em
    `LabelingOracle.label`. O Palpite mede na partição em que a correção vai rodar, e os prefixos
    levam o número de lotes que o vetor declara.
    """
    n_lotes = 1 if batches is None else len(set(np.asarray(batches).tolist()))
    oraculo = LabelingOracle(
        prefixes=(("none", n_lotes), ("loess", n_lotes), ("serrf", n_lotes)),
        n_sampled_features=N_FEATURES_DO_PALPITE,
        n_jobs=n_jobs,
        seed=seed,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        rotulo = oraculo.label(
            matriz, batches=batches, depois_de_cada=depois_de_cada, cancelled=cancelled
        )
    escores = {cast(str, correcao): float(v) for (correcao, _lotes), v in rotulo.scores.items()}
    # O motivo pelo nome do corretor; o prefixo `(c, n)` é detalhe do oráculo.
    recusas = {
        cast(str, correcao): motivo for (correcao, _lotes), motivo in rotulo.refusals.items()
    }
    return decide_o_palpite(
        escores,
        n_features_medidas=min(N_FEATURES_DO_PALPITE, int(matriz.n_features)),
        recusas=recusas,
    )
