"""Cache em disco das células processadas da grade.

A chave junta o `fingerprint` da especificação com o digest da matriz de entrada. O
`fingerprint` sozinho não basta: a mesma especificação aplicada a matrizes diferentes
produz células diferentes, e grades de modos distintos compartilham especificações.
"""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from refino.domain.matrix import RefinoMatrix
from refino.domain.spec import PipelineSpec

__all__ = ["CellCache", "cache_key", "matrix_digest"]


def matrix_digest(matrix: RefinoMatrix) -> str:
    """`sha256` dos valores e dos nomes de feature e de amostra, com grupo e tipo.

    Os nomes entram porque mudam o resultado: o `q` de Benjamini-Hochberg depende de
    quantas features foram testadas, e o teste univariado depende de quais amostras
    estão em cada grupo. Os valores entram por `tobytes()`, que não perde bits.
    """
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(np.asarray(matrix.values, dtype=np.float64)).tobytes())
    h.update("\n".join(matrix.feature_names).encode("utf-8"))
    h.update("\n".join(f"{a.name}|{a.group}|{a.sample_type}" for a in matrix.samples).encode())
    return h.hexdigest()


def cache_key(spec: PipelineSpec, matrix: RefinoMatrix) -> str:
    """A chave: `fingerprint` da especificação mais o início do digest da entrada."""
    return f"{spec.fingerprint()}.{matrix_digest(matrix)[:16]}"


@dataclass(frozen=True)
class CellCache:
    """Cache em disco de células processadas, por `cache_key`.

    Falha de leitura é miss, e não erro: arquivo truncado por interrupção ou de formato
    antigo leva a recalcular, nunca a devolver resultado parcial. A escrita do array é
    atômica, por renomeação.
    """

    root: Path

    def __post_init__(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)

    def _paths(self, key: str) -> tuple[Path, Path]:
        return self.root / f"{key}.npy", self.root / f"{key}.json"

    def get(self, spec: PipelineSpec, matrix: RefinoMatrix) -> np.ndarray | None:
        """Os valores da célula, ou `None` no miss. Qualquer falha de leitura é miss."""
        chave = cache_key(spec, matrix)
        npy, meta = self._paths(chave)
        if not npy.exists() or not meta.exists():
            return None
        try:
            declarado = json.loads(meta.read_text(encoding="utf-8"))
            valores = np.load(npy)
        except (OSError, ValueError):
            return None
        # O metadado guarda a chave por extenso: se o sistema de arquivos truncar o nome,
        # a chave truncada não é aceita como a pedida.
        if declarado.get("key") != chave:
            return None
        return np.asarray(valores, dtype=np.float64)

    def put(self, spec: PipelineSpec, matrix: RefinoMatrix, values: np.ndarray) -> None:
        """Grava a célula, atomicamente."""
        chave = cache_key(spec, matrix)
        npy, meta = self._paths(chave)
        # O temporário termina em `.npy` porque `np.save` acrescenta a extensão quando o
        # nome não a tem, e o `replace` não acharia o arquivo.
        temporario = npy.with_suffix(".tmp.npy")
        np.save(temporario, np.asarray(values, dtype=np.float64))
        temporario.replace(npy)
        meta.write_text(
            json.dumps({"key": chave, "fingerprint": spec.fingerprint()}), encoding="utf-8"
        )
