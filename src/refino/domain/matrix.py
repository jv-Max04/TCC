"""Matriz de intensidades: `RefinoMatrix`.

Orientação interna: amostras em linhas, features em colunas. Os arquivos do
laboratório usam a orientação oposta, e a conversão acontece uma vez, na ingestão.

A imutabilidade alcança o array, não só o atributo: `frozen=True` impede religar o
nome, mas o `ndarray` continuaria escrevível por dentro. Por isso ele é marcado
como não escrevível na construção.
"""

from dataclasses import dataclass, replace

import numpy as np

from refino.domain.metadata import SampleMetadata


@dataclass(frozen=True)
class RefinoMatrix:
    """Matriz de intensidades. Imutável.

    `values` tem shape `(n_samples, n_features)` e dtype `float64`. `samples` está
    ordenado por `injection_order` crescente, e a construção verifica isso.

    `provenance` é o histórico de operações aplicadas. Cada transformação devolve
    uma matriz nova com o registro estendido, o que permite reconstituir como um
    valor foi obtido.
    """

    values: np.ndarray
    feature_names: tuple[str, ...]
    samples: tuple[SampleMetadata, ...]
    mode: str
    provenance: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.values.ndim != 2:
            raise ValueError(f"values deve ser 2-D, recebi shape {self.values.shape}")
        n_samples, n_features = self.values.shape

        if len(self.samples) != n_samples:
            raise ValueError(
                f"{len(self.samples)} metadados de amostra para {n_samples} linhas de values"
            )
        if len(self.feature_names) != n_features:
            raise ValueError(
                f"{len(self.feature_names)} nomes de feature para {n_features} colunas de values"
            )
        if self.values.dtype != np.float64:
            raise ValueError(f"values deve ser float64, recebi {self.values.dtype}")

        orders = [s.injection_order for s in self.samples]
        if orders != sorted(orders):
            raise ValueError(
                "samples deve estar ordenado por injection_order crescente; "
                f"recebi {orders[:8]}{'...' if len(orders) > 8 else ''}"
            )

        # Toda amostra pertence a exatamente uma máscara. A guarda de
        # `SampleMetadata` barra valor inválido vindo de arquivo; esta barra um
        # `SampleType` que ganhe um quarto valor sem a máscara correspondente, o
        # que deixaria amostras válidas fora das três.
        coverage = self.qc_mask.astype(int) + self.sample_mask.astype(int)
        coverage += self.validate_mask.astype(int)
        if not (coverage == 1).all():
            uncovered = [
                self.samples[i].name for i, count in enumerate(coverage.tolist()) if count != 1
            ]
            raise ValueError(
                f"as máscaras não cobrem exatamente uma vez: {uncovered}. "
                "Se SampleType ganhou um valor novo, RefinoMatrix precisa da máscara "
                "correspondente — senão amostras válidas ficam invisíveis às três."
            )

        # A imutabilidade tem de alcançar o conteúdo do array.
        if self.values.flags.writeable:
            frozen = self.values.view()
            frozen.flags.writeable = False
            object.__setattr__(self, "values", frozen)

    # -- dimensões ---------------------------------------------------------

    @property
    def n_samples(self) -> int:
        """Número de injeções."""
        # int() explícito: nos stubs do numpy, indexar `.shape` devolve Any, e
        # `warn_return_any` recusa devolver Any de uma função anotada como int.
        return int(self.values.shape[0])

    @property
    def n_features(self) -> int:
        """Número de variáveis medidas."""
        return int(self.values.shape[1])

    # -- máscaras ----------------------------------------------------------

    @property
    def qc_mask(self) -> np.ndarray:
        """Booleano por amostra: `True` onde `sample_type == "qc"`."""
        return np.array([s.is_qc for s in self.samples], dtype=bool)

    @property
    def sample_mask(self) -> np.ndarray:
        """Booleano por amostra: `True` onde `sample_type == "sample"`.

        Não é o complemento de `qc_mask`: amostras `validate` são `False` nas duas
        máscaras. Um complemento as trataria como amostras de estudo e contaminaria
        a estatística a jusante.
        """
        return np.array([s.is_study_sample for s in self.samples], dtype=bool)

    @property
    def validate_mask(self) -> np.ndarray:
        """Booleano por amostra: `True` onde `sample_type == "validate"`."""
        return np.array([s.sample_type == "validate" for s in self.samples], dtype=bool)

    # -- metadados derivados ----------------------------------------------

    @property
    def injection_orders(self) -> np.ndarray:
        """Ordens de injeção, na ordem das linhas de `values`."""
        return np.array([s.injection_order for s in self.samples], dtype=np.int64)

    @property
    def sample_names(self) -> tuple[str, ...]:
        """Nomes das amostras, na ordem das linhas."""
        return tuple(s.name for s in self.samples)

    @property
    def groups(self) -> tuple[str, ...]:
        """Rótulos clínicos, na ordem das linhas."""
        return tuple(s.group for s in self.samples)

    @property
    def batches(self) -> tuple[str, ...]:
        """Lotes, na ordem das linhas."""
        return tuple(s.batch for s in self.samples)

    # -- transformação -----------------------------------------------------

    def with_values(self, values: np.ndarray, operation: str) -> "RefinoMatrix":
        """Nova matriz com outros valores e `provenance` estendido.

        Não muta nada. É por aqui que cada etapa do pipeline devolve uma matriz
        nova e registra o que fez.
        """
        if not operation:
            raise ValueError("with_values exige nomear a operação, para o provenance")
        if values.shape != self.values.shape:
            raise ValueError(
                f"shape {values.shape} incompatível com a matriz {self.values.shape}; "
                "with_values preserva dimensões — mudar shape exige construir nova matriz"
            )
        return replace(
            self,
            values=np.asarray(values, dtype=np.float64),
            provenance=(*self.provenance, operation),
        )

    def with_provenance(self, note: str) -> "RefinoMatrix":
        """Registra uma decisão que não alterou valores.

        Escolhas como a política de `on_label_mismatch` precisam ficar no histórico
        mesmo sem mudar número nenhum: 'alguém decidiu confiar no LABEL' é
        informação diferente de 'ninguém decidiu nada'.
        """
        if not note:
            raise ValueError("with_provenance exige uma nota não vazia")
        return replace(self, provenance=(*self.provenance, note))
