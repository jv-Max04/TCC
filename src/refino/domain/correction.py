"""`CorrectionResult`: o que um corretor devolve.

Fica em `domain` porque `correction/` o produz e `metrics/` o consome sem importar
quem corrige. `domain` é a única camada que as duas podem importar; se o tipo morasse
em `correction/`, o ciclo de imports voltaria pelo tipo de retorno.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from hashlib import sha256

import numpy as np

from refino.domain.labels import UncorrectedCause, validate_label
from refino.domain.matrix import RefinoMatrix


@dataclass(frozen=True)
class CorrectionResult:
    """Matriz corrigida mais o registro das features que não foram corrigidas.

    `uncorrected` mapeia causa para features, e não o contrário. Assim a regra 'cada
    feature não corrigida tem exatamente uma causa' é a disjunção entre os conjuntos,
    e um teste pode plantar a violação. Mapa vazio é o caso do corretor `none`, que
    não tenta corrigir nada.

    `requested` é o conjunto de features pedido, e a saída tem de ter exatamente
    essas. Existe para a execução parcial: corrigir só um subconjunto de features,
    com o pool completo de preditores. `None` significa execução completa, em que o
    pedido é o conjunto de features da matriz.

    Feature não pedida não entra em `uncorrected`: não ter sido pedida não é causa de
    não correção. As causas cobrem só as não corrigidas entre as pedidas.
    """

    matrix: RefinoMatrix
    uncorrected: Mapping[UncorrectedCause, frozenset[str]] = field(default_factory=dict)
    requested: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        """Valida na construção: rótulo válido, feature existente, causas disjuntas.

        Fica aqui, e não só nos testes, porque o mapa também chega desserializado
        do `metrics_json`, onde o mypy não alcança.
        """
        if self.requested is None:
            object.__setattr__(self, "requested", tuple(self.matrix.feature_names))
        elif set(self.requested) != set(self.matrix.feature_names):
            faltando = sorted(set(self.requested) - set(self.matrix.feature_names))
            sobrando = sorted(set(self.matrix.feature_names) - set(self.requested))
            raise ValueError(
                "o conjunto de features da saída tem de ser idêntico ao PEDIDO; "
                f"pedidas e ausentes da saída: {faltando[:5]}; "
                f"na saída e não pedidas: {sobrando[:5]}. "
                "Execução parcial declara o pedido; execução completa deixa `requested` "
                "em `None` e o pedido é a entrada."
            )

        conhecidas = set(self.matrix.feature_names)
        vistas: dict[str, UncorrectedCause] = {}
        for causa, features in self.uncorrected.items():
            validate_label("UncorrectedCause", causa)
            desconhecidas = sorted(set(features) - conhecidas)
            if desconhecidas:
                raise ValueError(
                    f"causa {causa!r} nomeia feature que não existe na matriz: "
                    f"{desconhecidas[:5]}. Marcação que aponta para fora do espaço "
                    "de features não é auditável — e o digest a esconderia."
                )
            for nome in features:
                if nome in vistas:
                    raise ValueError(
                        f"feature {nome!r} marcada em duas causas: {vistas[nome]!r} e "
                        f"{causa!r}. Causas são disjuntas por contrato — cada uma "
                        "leva a conclusão diferente sobre o dado, e "
                        "sobreposição faz a contagem por causa somar mais que o total."
                    )
                vistas[nome] = causa

    @property
    def uncorrected_mask(self) -> np.ndarray:
        """Booleana alinhada a `matrix.feature_names`, derivada do mapa.

        Não é guardada: a máscara se reconstrói do mapa sem perda, e o contrário
        não vale, porque ela não carrega a causa.
        """
        marcadas = {nome for features in self.uncorrected.values() for nome in features}
        return np.fromiter(
            (nome in marcadas for nome in self.matrix.feature_names),
            dtype=bool,
            count=len(self.matrix.feature_names),
        )

    def counts_by_cause(self) -> tuple[tuple[UncorrectedCause, int], ...]:
        """Contagem por causa, ordenada pelo nome da causa.

        Não devolve total: as causas dizem coisas diferentes, e um número agregado
        apagaria a distinção.

        A ordem é por nome, e não de inserção, porque o valor vai para o
        `metrics_json` e para o `--log`, e dois relatórios idênticos têm de
        serializar igual.
        """
        return tuple(sorted((causa, len(f)) for causa, f in self.uncorrected.items()))

    def uncorrected_digest(self) -> str:
        """Digest do conjunto de não corrigidas, para viajar no `QualityReport`.

        Contagem igual não é conjunto igual: duas especificações podem deixar 500
        features sem corrigir e deixar features diferentes.
        """
        marcadas = sorted(nome for features in self.uncorrected.values() for nome in features)
        return sha256("\n".join(marcadas).encode("utf-8")).hexdigest()
