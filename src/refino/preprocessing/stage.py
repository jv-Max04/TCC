"""`StageResult`: o que cada etapa do pipeline reporta além da matriz.

Cada etapa registra quantas features entraram e quantas saíram, para que o
caminho até o número final seja conferível, e não só o número.

Quando há mais de um motivo de descarte, as contagens vêm separadas: RSD acima
do corte é instabilidade medida, e RSD indefinido é ausência de medida. A
diferença entre entrada e saída dá só o total e apaga essa partição, por isso a
etapa a reporta.
"""

from dataclasses import dataclass, field

__all__ = ["StageResult"]


@dataclass(frozen=True)
class StageResult:
    """Contagens e rótulos de uma etapa do pipeline pós-correção.

    `discarded_by_reason` é vazio quando a etapa tem um motivo só de descarte, e
    o total sai de `n_features_in - n_features_out`. Com mais de um motivo, a
    soma dos valores tem de fechar com essa diferença.

    `labels` carrega os rótulos que a etapa consultou: variante de RSD,
    estimador, método de imputação. Quantidade sem rótulo não é comparável: o
    mesmo arquivo golden dá RSD de 3,78% ou de 104,73% conforme o estimador.
    """

    stage: str
    n_features_in: int
    n_features_out: int
    discarded_by_reason: dict[str, int] = field(default_factory=dict)
    labels: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Confere que a partição por motivo fecha com a diferença.

        Um motivo esquecido apareceria como descarte sem causa.
        """
        if self.n_features_out > self.n_features_in:
            raise ValueError(
                f"etapa {self.stage!r} reporta {self.n_features_out} features na saída e "
                f"{self.n_features_in} na entrada. Etapa de filtro não cria feature; se a "
                "contagem cresceu, a etapa está medindo matriz diferente da que recebeu."
            )
        if not self.discarded_by_reason:
            return

        soma = sum(self.discarded_by_reason.values())
        esperado = self.n_features_in - self.n_features_out
        if soma != esperado:
            raise ValueError(
                f"etapa {self.stage!r}: os motivos somam {soma} descartes e a diferença "
                f"entre entrada e saída é {esperado}. A partição por motivo tem de ser "
                "EXAUSTIVA — motivo esquecido aparece como descarte sem causa, que é "
                "indistinguível de etapa que não classifica o que descarta."
            )

    @property
    def n_discarded(self) -> int:
        """Total de descartes. Derivado, para que não exista uma segunda fonte."""
        return self.n_features_in - self.n_features_out
