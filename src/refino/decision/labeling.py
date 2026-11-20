"""O oráculo de rotulagem: mede os corretores numa matriz e diz qual venceu.

O rótulo é o corretor com o menor RSD mediano dos QCs fora da amostra (`leave_one_qc_out_rsd`):
cada QC é retirado, o corretor é ajustado nos outros e o RSD é medido com o que ficou de fora. O
estimador importa. Dentro da amostra, um método que passa uma curva pelos QCs zera o RSD que
depois se mede neles, e o LOESS ganharia sempre; fora da amostra, nos quatro arquivos de
referência, ele é pior que não corrigir em três. Um alvo dentro da amostra ensinaria a política a
recomendar LOESS em toda parte.

O oráculo vê o desfecho da matriz que rotula. Ele produz o alvo do treino e a medição mostrada ao
usuário, e não é uma política a comparar com as outras. Os rótulos possíveis são os pares
`(correção, número de lotes)`, porque a métrica é medida logo depois da correção e só depende
deles. Com lote único são três: `none`, `loess` e `serrf`.
"""

import warnings
from collections.abc import Callable
from dataclasses import dataclass, field, replace

import numpy as np

from refino.correction.loess import LoessCorrector
from refino.correction.none import NoneCorrector
from refino.correction.serrf import SerrfCorrector
from refino.correction.serrf_na_vez import SerrfNaVez
from refino.domain.cancellation import CancelCheck
from refino.domain.labels import Correction, RsdVariant
from refino.domain.matrix import RefinoMatrix
from refino.metrics.crossval import (
    JACKKNIFE_LEVEL,
    BatchQcShortfallError,
    JackknifeInterval,
    leave_one_qc_out_rsd,
    leave_one_qc_out_rsd_with_jackknife,
)

__all__ = ["Label", "LabelingOracle", "PrefixKey", "build_corrector_for_labeling", "com_lotes"]


def build_corrector_for_labeling(
    name: Correction,
    *,
    n_jobs: int = -1,
) -> NoneCorrector | LoessCorrector | SerrfCorrector:
    """O corretor usado na medição fora da amostra, com `n_jobs` repassado ao LOESS e ao SERRF.

    Não é o `build_corrector` do executor do multiverso. Lá o SERRF recebe `n_jobs=1` porque o
    executor já paraleliza por fora. Aqui a medição são `n_qc` reajustes em série de um corretor,
    e o paralelismo fica dentro dele; os dois níveis ao mesmo tempo abririam um pool por worker.
    Ficou dentro do corretor, e não entre os reajustes, porque paralelizar os reajustes atrasava
    muito o atendimento do cancelamento.

    O padrão `-1` usa todos os núcleos. Quem passa outro valor é a rota que mede os corretores,
    com o número efetivo da opção de vários núcleos da janela.
    """
    if name == "none":
        return NoneCorrector()
    if name == "loess":
        return LoessCorrector(n_jobs=n_jobs)
    if name == "serrf":
        return SerrfNaVez(n_jobs=n_jobs)
    raise ValueError(f"corretor desconhecido: {name!r}")


PrefixKey = tuple[Correction, int]


@dataclass(frozen=True)
class Label:
    """O rótulo de uma matriz, com os números que o produziram.

    `scores` traz o RSD mediano de cada prefixo medido. Sem ele o rótulo não distingue uma vitória
    larga de uma no quinto decimal, e na segunda o certo é abster-se. `margin` é a diferença
    relativa entre o melhor e o segundo.

    `refusals` traz o motivo de cada prefixo que não foi medido porque o menor lote não tem QCs
    bastantes para o ajuste fora da amostra do corretor. O score dele é `nan`, como o do prefixo
    medido cujo RSD ficou indefinido em toda feature, e é o motivo que separa os dois. Ambos
    ficam fora da eleição. Fica vazio quando nenhum corretor é recusado.
    """

    prefix: PrefixKey
    scores: dict[PrefixKey, float]
    estimator: str
    variant: RsdVariant
    refusals: dict[PrefixKey, str] = field(default_factory=dict)

    @property
    def correction(self) -> Correction:
        return self.prefix[0]

    @property
    def margin(self) -> float:
        """`(segundo - melhor) / |melhor|`. É `nan` sem segundo finito ou com o melhor em zero."""
        finitos = sorted(v for v in self.scores.values() if np.isfinite(v))
        if len(finitos) < 2 or finitos[0] == 0.0:
            return float("nan")
        return float((finitos[1] - finitos[0]) / abs(finitos[0]))

    @property
    def assessable(self) -> bool:
        """Diz se ao menos um prefixo teve valor finito. Sem isso não houve eleição."""
        return any(np.isfinite(v) for v in self.scores.values())


@dataclass(frozen=True)
class LabelingOracle:
    """Rotula uma matriz pelo prefixo que minimiza o RSD mediano dos QCs fora da amostra.

    Não implementa `DecisionPolicy` de propósito: ele vê o desfecho da matriz que rotula, e a
    mesma interface convidaria a compará-lo com as políticas.
    """

    prefixes: tuple[PrefixKey, ...] = (("none", 1), ("loess", 1), ("serrf", 1))
    variant: RsdVariant = "outlier_removed"
    n_sampled_features: int | None = None
    # O `n_jobs` do LOESS e do SERRF desta medição. `-1` usa todos os núcleos; a rota que mede os
    # corretores e o Palpite passam o valor da opção de vários núcleos.
    n_jobs: int = -1
    # Sorteia as features amostradas e semeia cada ajuste fora da amostra. Quem a troca é o
    # `--seed` da execução, para a escolha do corretor obedecer à mesma semente que a correção.
    seed: int = 42

    def __post_init__(self) -> None:
        if not self.prefixes:
            raise ValueError("o oráculo precisa de ao menos um prefixo para eleger")
        if len(set(self.prefixes)) != len(self.prefixes):
            raise ValueError(f"prefixos repetidos: {self.prefixes}")

    def label(
        self,
        matrix: RefinoMatrix,
        *,
        batches: np.ndarray | None = None,
        depois_de_cada: Callable[[PrefixKey], None] | None = None,
        cancelled: CancelCheck | None = None,
    ) -> Label:
        """Mede os prefixos e devolve o rótulo da matriz.

        `n_sampled_features` é `None` por padrão, e então todas as features são medidas.

        `batches` é o vetor de lote, alinhado a `matrix.samples`. Ele é gravado no `batch` de cada
        amostra antes de medir, que é o campo que os corretores e a guarda de
        `leave_one_qc_out_rsd` leem. Prefixo com `n` lotes exige vetor com `n` rótulos distintos:
        medir com outra partição que a declarada responderia a outra pergunta.

        Um lote com menos de três QCs não deixa treino bastante quando um deles é retido, e a
        medição levanta `DegenerateCrossValidationError`; este método não converte isso em empate.
        Acima desse piso, o corretor que exige mais QCs de treino por lote, como o LOESS, levanta
        `BatchQcShortfallError` antes de ajustar. Essa recusa não derruba a medição: o prefixo
        fica com `nan` e com o motivo em `Label.refusals`, e os outros são medidos.

        `depois_de_cada` é chamado com o prefixo depois de cada um, medido ou recusado. Serve ao
        progresso por corretor, e é nele que o chamador pode levantar para cancelar. Não recebe o
        score, para que a interface não mostre um ranking parcial, que pode inverter quando o
        último corretor chega. `cancelled` é repassado a `leave_one_qc_out_rsd` e daí ao corretor,
        para que a parada não espere o fim de um corretor inteiro.
        """
        rotulo, _intervalos = self._mede(
            matrix,
            batches=batches,
            depois_de_cada=depois_de_cada,
            cancelled=cancelled,
            nivel=None,
        )
        return rotulo

    def label_com_intervalos(
        self,
        matrix: RefinoMatrix,
        *,
        batches: np.ndarray | None = None,
        depois_de_cada: Callable[[PrefixKey], None] | None = None,
        cancelled: CancelCheck | None = None,
        nivel: float = JACKKNIFE_LEVEL,
    ) -> tuple[Label, dict[PrefixKey, JackknifeInterval]]:
        """O rótulo de `label` e o intervalo jackknife-t do RSD mediano de cada prefixo.

        O rótulo é idêntico ao de `label`. O intervalo custa `n(n-1)/2` ajustes a mais por
        corretor, e por isso é um método à parte: quem não o pede chama `label` e não paga por
        ele. `Label` não muda; os intervalos vêm ao lado, num dicionário pelo mesmo prefixo. O
        gancho continua a ser chamado uma vez por prefixo, depois do intervalo dele.
        """
        rotulo, intervalos = self._mede(
            matrix,
            batches=batches,
            depois_de_cada=depois_de_cada,
            cancelled=cancelled,
            nivel=nivel,
        )
        return rotulo, intervalos

    def _mede(
        self,
        matrix: RefinoMatrix,
        *,
        batches: np.ndarray | None,
        depois_de_cada: Callable[[PrefixKey], None] | None,
        cancelled: CancelCheck | None,
        nivel: float | None,
    ) -> tuple[Label, dict[PrefixKey, JackknifeInterval]]:
        """O laço dos prefixos, comum a `label` e `label_com_intervalos`.

        Com `nivel=None` chama `leave_one_qc_out_rsd` e não calcula intervalo nenhum.
        """
        lotes_da_matriz = com_lotes(matrix, batches)
        distintos = len(set(lotes_da_matriz.batches))
        scores: dict[PrefixKey, float] = {}
        intervalos: dict[PrefixKey, JackknifeInterval] = {}
        recusas: dict[PrefixKey, str] = {}
        for prefixo in self.prefixes:
            correcao, n_lotes = prefixo
            if n_lotes != 1 and batches is None:
                raise NotImplementedError(
                    f"prefixo {prefixo}: lotes múltiplos exigem o vetor de lote em `batches`"
                )
            if batches is not None and n_lotes != distintos:
                raise ValueError(
                    f"prefixo {prefixo} declara {n_lotes} lote(s) e o vetor de lote tem "
                    f"{distintos} rótulo(s) distinto(s)"
                )
            corretor = build_corrector_for_labeling(correcao, n_jobs=self.n_jobs)
            try:
                valores = self._mede_um(
                    lotes_da_matriz, corretor, prefixo, intervalos, nivel, cancelled
                )
            except BatchQcShortfallError as recusa:
                # O corretor recusado fica com `nan` e com o motivo, fora da eleição. Com o
                # intervalo pedido, o dele leva o mesmo motivo.
                scores[prefixo] = float("nan")
                recusas[prefixo] = recusa.reason
                if nivel is not None:
                    intervalos[prefixo] = JackknifeInterval(
                        estimate=float("nan"),
                        level=nivel,
                        n_qc=int(lotes_da_matriz.qc_mask.sum()),
                        reason=recusa.reason,
                    )
                if depois_de_cada is not None:
                    depois_de_cada(prefixo)
                continue
            with warnings.catch_warnings():
                # `All-NaN slice` é esperado quando o RSD é indefinido em toda feature, como numa
                # matriz de QCs todos zero. O desfecho vai em `assessable`.
                warnings.simplefilter("ignore", RuntimeWarning)
                scores[prefixo] = float(np.nanmedian(valores))
            if depois_de_cada is not None:
                depois_de_cada(prefixo)

        finitos = {k: v for k, v in scores.items() if np.isfinite(v)}
        if not finitos:
            # Não levanta: RSD indefinido em todo prefixo é um desfecho da matriz. `assessable`
            # o separa de um empate, e o prefixo devolvido é o primeiro na ordem declarada.
            return (
                Label(self.prefixes[0], scores, "leave_one_qc_out_rsd", self.variant, recusas),
                intervalos,
            )

        melhor = min(finitos, key=lambda k: finitos[k])
        return Label(melhor, scores, "leave_one_qc_out_rsd", self.variant, recusas), intervalos

    def _mede_um(
        self,
        lotes_da_matriz: RefinoMatrix,
        corretor: NoneCorrector | LoessCorrector | SerrfCorrector,
        prefixo: PrefixKey,
        intervalos: dict[PrefixKey, JackknifeInterval],
        nivel: float | None,
        cancelled: CancelCheck | None,
    ) -> np.ndarray:
        """Mede um prefixo: o RSD por feature e, quando pedido, o intervalo dele.

        Fica separada de `_mede` para que `BatchQcShortfallError` seja apanhada num lugar só.
        """
        if nivel is None:
            valores = leave_one_qc_out_rsd(
                lotes_da_matriz,
                corretor,
                variant=self.variant,
                seed=self.seed,
                n_sampled_features=self.n_sampled_features,
                cancelled=cancelled,
                # Os reajustes em série: os núcleos vão para dentro do corretor, e os dois níveis
                # juntos abririam um pool por worker.
                n_jobs=1,
            )
        else:
            # Os pares do jackknife também em série, pelo mesmo motivo.
            valores, intervalos[prefixo] = leave_one_qc_out_rsd_with_jackknife(
                lotes_da_matriz,
                corretor,
                variant=self.variant,
                seed=self.seed,
                n_sampled_features=self.n_sampled_features,
                level=nivel,
                cancelled=cancelled,
                n_jobs=1,
            )
        return valores


def com_lotes(matrix: RefinoMatrix, batches: np.ndarray | None) -> RefinoMatrix:
    """A matriz com o vetor de lote gravado no `batch` de cada amostra, ou ela mesma sem vetor.

    Devolve uma matriz nova; a de entrada não muda. O rótulo vira texto porque é o tipo do campo.
    É pública para que a rota da figura da curva do LOESS grave os lotes pelo mesmo caminho e meça
    na mesma partição que a medição dos corretores.
    """
    if batches is None:
        return matrix
    rotulos = np.asarray(batches)
    if rotulos.shape != (matrix.n_samples,):
        raise ValueError(
            f"`batches` tem de estar alinhado a `samples`: recebi forma {rotulos.shape} para "
            f"{matrix.n_samples} amostras"
        )
    amostras = tuple(
        replace(amostra, batch=str(rotulo))
        for amostra, rotulo in zip(matrix.samples, rotulos.tolist(), strict=True)
    )
    return replace(matrix, samples=amostras).with_provenance(
        f"labeling:batches({len(set(rotulos.tolist()))} lotes)"
    )
