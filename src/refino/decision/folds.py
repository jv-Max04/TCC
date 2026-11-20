"""Partição do corpus por modo de origem, para validar a política sem vazamento.

O gerador de matrizes sintéticas reamostra resíduos de um dos quatro arquivos reais. Duas sementes
diferentes dão duas reamostragens do mesmo modo, com a mesma estrutura de correlação entre
features, o mesmo padrão de zeros e o mesmo perfil de QC. Separar treino e teste por semente,
portanto, não separa nada: o que vaza é a covariância. A partição que vale é por modo, em quatro
dobras, e cada modo é avaliado por um modelo treinado só em matrizes derivadas dos outros três.

Os quatro modos vêm do mesmo estudo, laboratório e equipamento, então isso mede generalização
entre modos cromatográficos, e não entre estudos. `leakage_pair` monta também a partição por
semente, para que o tamanho do vazamento possa ser medido pela diferença entre as duas.
"""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from refino.domain.labels import ImitatedMode

__all__ = ["Fold", "LeakagePair", "leakage_pair", "mode_folds"]


@dataclass(frozen=True)
class Fold:
    """Uma dobra: o modo validado e os índices de treino e de teste.

    `held_out_mode` fica separado de `test` porque a pergunta a conferir é se alguma matriz de
    treino deriva do modo validado. Índices disjuntos não respondem a isso: uma partição que
    mistura modos também os tem.
    """

    held_out_mode: ImitatedMode
    train: tuple[int, ...]
    test: tuple[int, ...]

    def defects_against(self, modes: Sequence[str]) -> list[str]:
        """Lista os defeitos desta dobra, dados os modos de origem das matrizes.

        Devolve a lista em vez de levantar, para que um teste possa plantar um defeito e ver a
        acusação pelo mesmo caminho.
        """
        found: list[str] = []
        vazados = [i for i in self.train if modes[i] == self.held_out_mode]
        if vazados:
            found.append(
                f"dobra {self.held_out_mode}: {len(vazados)} matriz(es) de treino derivam do modo "
                f"validado, nos índices {vazados[:5]}. O canal de vazamento é a covariância — elas "
                "carregam a estrutura de correlação, o padrão de zeros e o perfil de QC do modo em "
                "que o modelo vai ser avaliado"
            )
        fora = [i for i in self.test if modes[i] != self.held_out_mode]
        if fora:
            found.append(
                f"dobra {self.held_out_mode}: {len(fora)} matriz(es) de teste NÃO derivam do modo "
                f"validado, nos índices {fora[:5]}. A dobra deixaria de medir generalização para "
                "aquele modo, e o número sairia otimista sem nada acusar"
            )
        if set(self.train) & set(self.test):
            found.append(f"dobra {self.held_out_mode}: treino e teste compartilham índices")
        if not self.train:
            found.append(f"dobra {self.held_out_mode}: treino vazio")
        if not self.test:
            found.append(f"dobra {self.held_out_mode}: teste vazio")
        return found


def mode_folds(modes: Sequence[str]) -> tuple[Fold, ...]:
    """As dobras, uma por modo presente, na ordem fixa dos quatro modos do estudo.

    A ordem não depende da de chegada: um corpus embaralhado dá a mesma sequência de dobras. Modo
    que ficaria sem teste ou sem treino não gera dobra, para não ser contado como dobra avaliada.
    """
    canonica = ("pos_hilic", "pos_c18", "neg_hilic", "neg_c18")
    presentes = [m for m in canonica if m in set(modes)]
    desconhecidos = sorted(set(modes) - set(canonica))
    if desconhecidos:
        raise ValueError(
            f"modos fora do vocabulário: {desconhecidos}. `ImitatedMode` é fronteira de "
            "confiança, e modo inválido daria dobra sem arquivo real correspondente"
        )

    dobras: list[Fold] = []
    for modo in presentes:
        teste = tuple(i for i, m in enumerate(modes) if m == modo)
        treino = tuple(i for i, m in enumerate(modes) if m != modo)
        if not teste or not treino:
            continue
        dobras.append(Fold(held_out_mode=modo, train=treino, test=teste))  # type: ignore[arg-type]
    return tuple(dobras)


@dataclass(frozen=True)
class LeakagePair:
    """O par que mede o tamanho do vazamento: a partição por modo e a que vaza.

    `leaky` é instrumento de medição, e não opção para o usuário. Existe para que a diferença de
    desempenho entre as duas possa ser reportada.
    """

    honest: tuple[Fold, ...]
    leaky: tuple[Fold, ...]


def leakage_pair(modes: Sequence[str], *, seed_groups: Iterable[int] | None = None) -> LeakagePair:
    """As duas partições sobre o mesmo corpus: por modo e por semente.

    A `leaky` tira o teste de um grupo de semente (por padrão índices pares ou ímpares, ou os
    grupos de `seed_groups`) e deixa no treino matrizes do modo que a dobra valida. A diferença de
    desempenho entre as duas é o quanto a partição por semente superestima.
    """
    honesta = mode_folds(modes)
    grupos = list(seed_groups) if seed_groups is not None else [i % 2 for i in range(len(modes))]
    if len(grupos) != len(modes):
        raise ValueError(f"{len(grupos)} grupos para {len(modes)} matrizes")

    vazada: list[Fold] = []
    for dobra in honesta:
        # Mesmo número de matrizes de teste que a dobra por modo, para que a comparação não
        # misture vazamento com tamanho de amostra.
        alvo = len(dobra.test)
        marca = grupos[dobra.test[0]] if dobra.test else 0
        candidatos = [i for i, g in enumerate(grupos) if g == marca]
        teste = tuple(candidatos[:alvo])
        treino = tuple(i for i in range(len(modes)) if i not in set(teste))
        vazada.append(Fold(held_out_mode=dobra.held_out_mode, train=treino, test=teste))
    return LeakagePair(honest=honesta, leaky=tuple(vazada))
