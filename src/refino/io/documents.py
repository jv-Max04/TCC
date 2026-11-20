"""Os documentos que os leitores devolvem.

Cada um representa o arquivo como ele é, antes de virar `RefinoMatrix`, e carrega
em `issues` o que a leitura revelou. Assim o chamador não precisa reler o arquivo
nem consultar estado global para saber o que o leitor viu.
"""

from dataclasses import dataclass, field

import numpy as np

from refino.io.issues import ValidationIssue
from refino.io.sniffer import Dialect


@dataclass(frozen=True)
class MeasurementsDocument:
    """O arquivo `measurements` lido, na orientação do arquivo.

    `values` tem shape `(n_features, n_samples)`: features em linhas, como o
    Progenesis QI exporta. Quem transpõe para a orientação interna é `load_matrix`.

    `raw_cells` guarda o texto original das células de intensidade, e só delas.
    `NON_NUMERIC_CELL` e `THOUSANDS_SEPARATOR_CORRUPTION` são propriedades do
    texto, que a conversão para `float64` destrói. As linhas `NAME`/`LABEL` ficam
    de fora porque rótulo é texto por contrato.
    """

    values: np.ndarray
    feature_names: tuple[str, ...]
    sample_names: tuple[str, ...]
    group_labels: tuple[str, ...]
    raw_cells: tuple[tuple[str, ...], ...]
    dialect: Dialect
    issues: tuple[ValidationIssue, ...] = field(default=())

    @property
    def n_features(self) -> int:
        """Linhas da matriz no arquivo."""
        return int(self.values.shape[0])

    @property
    def n_samples(self) -> int:
        """Colunas da matriz no arquivo."""
        return int(self.values.shape[1])


@dataclass(frozen=True)
class ToSerrfDocument:
    """O arquivo `toserrf` lido.

    Não tem campo `group`: o formato não carrega rótulo clínico, e a linha `label`
    contém nomes de amostra. Sem um grupo inventado, dois documentos se comparam
    diretamente. Para virar `RefinoMatrix` passa por `to_matrix`, que exige
    `groups`.
    """

    batch: tuple[str, ...]
    sample_type: tuple[str, ...]
    time: tuple[int, ...]
    sample_names: tuple[str, ...]
    feature_names: tuple[str, ...]
    values: np.ndarray
    issues: tuple[ValidationIssue, ...] = field(default=())

    @property
    def n_features(self) -> int:
        return int(self.values.shape[0])

    @property
    def n_samples(self) -> int:
        return int(self.values.shape[1])


class MissingGroupsError(ValueError):
    """`to_matrix` recebeu `groups` incompleto.

    É exceção e não `ValidationIssue` porque o defeito é do argumento do
    chamador, não do arquivo: o formato `toserrf` não carrega esse campo.
    """

    def __init__(self, missing: tuple[str, ...]) -> None:
        self.missing = missing
        super().__init__(
            f"{len(missing)} amostra(s) não-QC sem grupo em `groups`: "
            f"{', '.join(sorted(missing))}. O formato toserrf não carrega rótulo "
            "clínico — forneça o mapa completo ou use as amostras QC apenas."
        )
