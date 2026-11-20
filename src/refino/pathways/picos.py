"""Os nomes de feature lidos como picos de massa, com as duas quantidades que eles carregam.

Os nomes exportados pelo Progenesis QI têm duas formas, que não medem a mesma coisa:

- sufixo `m/z`, como `2.25_810.7739m/z`: razão massa-carga do íon medido;
- sufixo `n`, como `23.59_464.2346n`: massa neutra já deconvoluída pelo fabricante.

O `n` é de neutral: o Progenesis conseguiu juntar os adutos daquele composto e resolver
a massa da molécula; quando não consegue, publica o `m/z` cru.

As duas partes numéricas casam a mesma expressão regular. Quem ler o número sem olhar o
sufixo recebe um valor válido nas duas formas, e tratar massa neutra como `m/z` erra por
um próton, cerca de 1,007 Da, em toda feature, sem nada levantar.

Nos arquivos do laboratório só aparecem essas duas formas, mas outro arquivo pode ter
outras. Por isso `interpreta` devolve `None` em vez de levantar, e
`ResumoDeInterpretacao` conta os não interpretados em campo próprio.
"""

import re
from dataclasses import dataclass
from typing import Literal

__all__ = [
    "MASSA_DO_PROTON",
    "Pico",
    "ResumoDeInterpretacao",
    "TipoDePico",
    "interpreta",
    "interpreta_todos",
]

# Massa do próton em Da. É o mesmo valor que o mummichog usa em `config.PROTON`: os
# nossos números são comparados com os dele, e constantes com precisão diferente dariam
# divergência de milésimos de Da, que a janela de ppm não tolera.
MASSA_DO_PROTON = 1.00727646677

# Este `Literal` não vai para `domain/labels.py`: a conferência em tempo de execução de
# lá é para rótulo que vem de fora. Este é produzido aqui, a partir do sufixo do nome, e
# o mypy basta.
TipoDePico = Literal["ion", "neutral_mass"]

# Duas expressões, e não uma com alternância, porque o sufixo decide o tipo. Uma regex
# só, com o sufixo opcional, convidaria a ler o número sem olhar qual dos dois casou.
_ION = re.compile(r"^(?P<rt>\d+(?:\.\d+)?)_(?P<massa>\d+(?:\.\d+)?)m/z$")
_NEUTRA = re.compile(r"^(?P<rt>\d+(?:\.\d+)?)_(?P<massa>\d+(?:\.\d+)?)n$")


@dataclass(frozen=True)
class Pico:
    """Um nome de feature lido como tempo de retenção mais uma massa.

    `tipo` diz o que `massa` significa. Não há campos `mz` e `massa_neutra` separados,
    porque dois campos em que um é sempre `None` permitiriam ler o errado sem erro.

    `rt` fica na unidade em que veio, minutos nos arquivos deste laboratório. A
    tolerância de coeluição do mummichog é fração do tempo máximo
    (`RETENTION_TIME_TOLERANCE_FRAC`, 2%) e não um absoluto, então não depende da
    escala.
    """

    nome: str
    rt: float
    massa: float
    tipo: TipoDePico

    def como_mz(self, modo: Literal["positive", "negative"]) -> float:
        """A massa deste pico expressa como `m/z`, que é o que o algoritmo consome.

        Para `ion` é identidade. Para `neutral_mass` assume o íon primário do modo,
        `[M+H]+` ou `[M-H]-`. É hipótese: o fabricante devolveu a massa neutra e não
        disse por qual aduto ela foi observada.

        A hipótese custa pouco porque a busca por adutos do mummichog inclui o íon
        primário, então a massa neutra verdadeira está entre os candidatos testados.
        Os outros adutos geram candidatos espúrios, a mesma ambiguidade que toda
        feature `m/z` já traz e que o teste de permutação desconta.

        Quem não quiser a hipótese pede a política `exclude` em `analise.py`, e essas
        features não entram.
        """
        if self.tipo == "ion":
            return self.massa
        if modo == "positive":
            return self.massa + MASSA_DO_PROTON
        return self.massa - MASSA_DO_PROTON


@dataclass(frozen=True)
class ResumoDeInterpretacao:
    """O que entrou, separado por forma, e o que não foi interpretado.

    Os não interpretados têm campo próprio em vez de serem descartados: o número de
    vias no fim não denuncia features que nunca chegaram lá.
    """

    ions: tuple[Pico, ...]
    massas_neutras: tuple[Pico, ...]
    nao_interpretados: tuple[str, ...]

    @property
    def total(self) -> int:
        return len(self.ions) + len(self.massas_neutras) + len(self.nao_interpretados)


def interpreta(nome: str) -> Pico | None:
    """Lê um nome de feature como `Pico`, ou devolve `None` quando nenhuma das formas casa.

    Forma desconhecida é dado, não defeito de programação. Levantar faria um arquivo
    com uma terceira forma abortar a análise inteira em vez de reportar o que não
    soube ler.
    """
    limpo = nome.strip()
    if casado := _ION.match(limpo):
        return Pico(nome=nome, rt=float(casado["rt"]), massa=float(casado["massa"]), tipo="ion")
    if casado := _NEUTRA.match(limpo):
        return Pico(
            nome=nome,
            rt=float(casado["rt"]),
            massa=float(casado["massa"]),
            tipo="neutral_mass",
        )
    return None


def interpreta_todos(nomes: tuple[str, ...]) -> ResumoDeInterpretacao:
    """Interpreta uma lista inteira, preservando a ordem e separando por forma."""
    ions: list[Pico] = []
    neutras: list[Pico] = []
    falhos: list[str] = []
    for nome in nomes:
        pico = interpreta(nome)
        if pico is None:
            falhos.append(nome)
        elif pico.tipo == "ion":
            ions.append(pico)
        else:
            neutras.append(pico)
    return ResumoDeInterpretacao(
        ions=tuple(ions),
        massas_neutras=tuple(neutras),
        nao_interpretados=tuple(falhos),
    )
