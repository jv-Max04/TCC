"""As políticas de decisão: o que uma recomendação contém e as políticas constantes de comparação.

Uma recomendação tem duas partes que não saem uma da outra. O corretor vem do vencedor da medição
fora da amostra, que existe sempre e por isso não diz quando abster-se. A confiança vem da margem
relativa entre o melhor e o segundo: quando ela é menor do que o ruído da própria medição, a
escolha é um empate e a política se abstém.

`AlwaysSERRFPolicy` e `BestFixedCorrectorPolicy` são as linhas de base contra as quais a política
aprendida (`model.py`) é comparada. Não há política heurística a partir do diagnóstico: ele
classifica o tipo de defeito e propõe lotes, e o tipo de defeito não separa os corretores
vencedores.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from refino.domain.diagnosis import DefectProfile
from refino.domain.labels import Correction

__all__ = [
    "ABSTENTION_MARGIN",
    "ORIGEM_DO_LIMIAR",
    "AlwaysSERRFPolicy",
    "BestFixedCorrectorPolicy",
    "DecisionPolicy",
    "Recommendation",
    "confidence_from_margin",
    "elect_best_fixed",
]

# Margem relativa abaixo da qual a política se abstém. Vem do ruído da semente: nos quatro
# arquivos de referência, duas vezes o maior desvio da margem quando só a semente dos corretores
# muda dá cerca de 0,0126. Margem menor do que isso é empate, porque trocar a semente a apaga.
ABSTENTION_MARGIN = 0.012

# A frase da justificativa que diz de onde o limiar vem. Fica aqui porque `_monta` e a política
# aprendida (`model.py`) a escrevem, e duas cópias acabariam diferentes.
ORIGEM_DO_LIMIAR = (
    "a ordem do quanto a margem muda quando só a semente do corretor muda, medida nos quatro "
    "arquivos reais (duas vezes o desvio: 0,0126)"
)


def confidence_from_margin(margin: float, *, threshold: float = ABSTENTION_MARGIN) -> float:
    """Confiança em `[0, 1]` a partir da margem relativa entre o melhor e o segundo corretor.

    É monotônica e satura: margem maior nunca dá confiança menor, e margem de 200% não vale o
    dobro da de 100%. Em `margin == threshold` a confiança é exatamente 0,5, o corte de
    `Recommendation.abstains`, então o limiar de margem e o de confiança são o mesmo ponto.

    Margem não finita ou não positiva devolve 0,0, e não `nan`: nos dois casos a ação é abster-se.
    A diferença entre 'não medida' e 'nula' fica em `Recommendation.assessable`.
    """
    if margin != margin or margin in (float("inf"), float("-inf")) or margin <= 0.0:
        return 0.0
    if margin <= threshold:
        return float(0.5 * margin / threshold)
    return float(1.0 - 0.5 * threshold / margin)


@dataclass(frozen=True)
class Recommendation:
    """O que uma política devolve: o corretor, a confiança e o que sustenta cada um.

    `rationale` cita os números de `evidence`, para que a confiança não venha sem origem.
    `assessable` é separado de `confidence == 0`: margem medida e minúscula e margem que não pôde
    ser medida levam à mesma abstenção, mas são contadas em separado.
    """

    correction: Correction
    confidence: float
    rationale: str
    evidence: dict[str, float]
    assessable: bool = True

    @property
    def abstains(self) -> bool:
        """Diz se a política se abstém: `confidence < 0,5`."""
        return self.confidence < 0.5


@runtime_checkable
class DecisionPolicy(Protocol):
    """O contrato de uma política. `LabelingOracle` não o satisfaz de propósito: vê o desfecho.

    `name` é propriedade só de leitura porque há políticas que são `dataclass` congeladas, e um
    atributo `name: str` no protocolo exigiria campo mutável.
    """

    @property
    def name(self) -> str: ...

    def recommend(self, profile: DefectProfile) -> Recommendation: ...


@dataclass(frozen=True)
class AlwaysSERRFPolicy:
    """Sempre SERRF: a linha de base constante, que é o que o laboratório faz hoje.

    Não é um piso fácil de superar: nos arquivos de referência o SERRF vence a medição fora da
    amostra na maioria dos modos. A confiança vem da margem estimada para o perfil, e não é
    constante.
    """

    name: str = "always_serrf"
    threshold: float = ABSTENTION_MARGIN
    margin_estimator: Callable[[DefectProfile], float] | None = None

    def recommend(self, profile: DefectProfile) -> Recommendation:
        margem = _estima(self.margin_estimator, profile)
        return _monta("serrf", margem, self.threshold, "constante", profile)


@dataclass(frozen=True)
class BestFixedCorrectorPolicy:
    """O melhor corretor constante da dobra de treino.

    É a linha de base que a política aprendida tem de superar: se não supera, escolher por matriz
    não compra nada. `corrector` é eleito fora desta classe, por `elect_best_fixed`, só com os
    rótulos do treino. Elegê-lo aqui exigiria ver o conjunto de teste.
    """

    corrector: Correction
    name: str = "best_fixed"
    threshold: float = ABSTENTION_MARGIN
    margin_estimator: Callable[[DefectProfile], float] | None = None

    def recommend(self, profile: DefectProfile) -> Recommendation:
        margem = _estima(self.margin_estimator, profile)
        return _monta(self.corrector, margem, self.threshold, "melhor constante do treino", profile)


def _estima(estimador: "Callable[[DefectProfile], float] | None", profile: DefectProfile) -> float:
    """A margem estimada, ou `nan` sem estimador, o que faz a política se abster.

    O estimador é injetado, e não lido do perfil: uma margem observada no perfil significaria que
    a medição já rodou. Na prática é `LearnedPolicy.predict_margin`, o mesmo para todas as
    políticas comparadas, para que a diferença entre elas seja só a escolha do corretor.
    """
    return float("nan") if estimador is None else float(estimador(profile))


def elect_best_fixed(labels: Sequence[Correction]) -> Correction:
    """O corretor mais frequente entre os rótulos da dobra de treino.

    Empate resolve pela ordem fixa `none`, `loess`, `serrf`, e não pela de chegada: o mesmo treino
    embaralhado tem de eleger o mesmo corretor.
    """
    if not labels:
        raise ValueError("não há rótulos na dobra de treino para eleger o melhor constante")
    canonica: tuple[Correction, ...] = ("none", "loess", "serrf")
    return max(canonica, key=lambda c: (labels.count(c), -canonica.index(c)))


def _margem_do_perfil(profile: DefectProfile) -> float:  # pragma: no cover - função desativada
    """Desativada: a margem é predita por um estimador, e não lida do perfil."""
    raise NotImplementedError("a margem é PREDITA, não lida do perfil")


def _monta(
    correcao: Correction, margem: float, limiar: float, origem: str, profile: DefectProfile
) -> Recommendation:
    """Monta a recomendação de uma política constante, com a justificativa em texto."""
    avaliavel = margem == margem
    confianca = confidence_from_margin(margem, threshold=limiar)
    evidencia = {
        "oracle_margin": margem,
        "abstention_threshold": limiar,
        "qc_rsd_median": profile.qc_rsd_median,
        "pc1_injection_correlation": profile.pc1_injection_correlation,
    }
    if not avaliavel:
        razao = (
            f"Recomendado `{correcao}` por política {origem}. **Confiança 0,00, abstenção**: a "
            f"margem entre o melhor e o segundo corretor NÃO foi medida para esta matriz. O RSD "
            f"mediano de QC é {profile.qc_rsd_median:.4f} e a correlação de PC1 com a ordem de "
            f"injeção é {profile.pc1_injection_correlation:.4f}, mas sem a margem não há base para "
            "confiar na escolha — execute o teste de robustez."
        )
    else:
        veredito = "abstenção" if confianca < 0.5 else "confiança"
        razao = (
            f"Recomendado `{correcao}` por política {origem}, com {veredito} de {confianca:.2f}. A "
            f"margem relativa do oráculo entre o melhor e o segundo corretor é {margem:.4f} do "
            f"melhor, contra o limiar de {limiar:.4f}, {ORIGEM_DO_LIMIAR}. O RSD "
            f"mediano de QC é {profile.qc_rsd_median:.4f} e a correlação de PC1 com a ordem de "
            f"injeção é {profile.pc1_injection_correlation:.4f}."
        )
    return Recommendation(
        correction=correcao,
        confidence=confianca,
        rationale=razao,
        evidence=evidencia,
        assessable=avaliavel,
    )
