"""A série categórica por grupo clínico: ordem canônica, cor e rótulo com o `n`.

Os grupos vêm do dado e não do código, em qualquer número. A ordem canônica é a ordenação
dos nomes por ponto de código, e não a ordem de aparecimento: assim a mesma coorte recebe as
mesmas cores em qualquer módulo, seja qual for a ordem das linhas. O custo é que um grupo
novo desloca as cores dos que vêm depois dele. Grupo sem amostra no vetor não gera série,
para a legenda não prometer pontos que não foram desenhados.

Sem estilo, as cores saem do ciclo dos `rcParams` e o marcador é o círculo. Com `estilos`, a
cor e o símbolo são os que a pessoa escolheu, dentro de um vocabulário fechado de oito cores
de Okabe e Ito (distinguíveis nas formas comuns de daltonismo) e oito marcadores cheios
(distinguíveis em impressão em tons de cinza). O estilo é resolvido sobre a coorte inteira
que o chamador informa, e não sobre os grupos presentes numa figura, para o mesmo grupo ter
a mesma cor em figuras que desenham subconjuntos diferentes.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Final, Literal

import numpy as np

from refino.plots.base import RecusaDoConstrutorError, cores_do_ciclo
from refino.plots.recorte import milhar

if TYPE_CHECKING:  # pragma: no cover - só para o mypy
    from collections.abc import Iterable, Mapping, Sequence

__all__ = [
    "CORES_DOS_GRUPOS",
    "MARCADOR_SEM_ESTILO",
    "PARES_DE_ESTILO",
    "SIMBOLOS_DOS_GRUPOS",
    "CorDoGrupo",
    "EstiloDoGrupo",
    "EstiloInvalidoError",
    "EstiloRepetidoError",
    "EstilosDosGrupos",
    "GrupoInvalidoError",
    "SerieDeGrupo",
    "SimboloDoGrupo",
    "cores_dos_grupos",
    "estilos_dos_grupos",
    "grupos_do_fio",
    "ordem_canonica",
    "rotulo_do_grupo",
    "series_por_grupo",
]

# ---------------------------------------------------------------------------
# O vocabulário fechado da escolha de estilo
# ---------------------------------------------------------------------------

CorDoGrupo = Literal[
    "azul", "vermelho", "verde-azulado", "rosa", "azul-celeste", "laranja", "amarelo", "preto"
]

# As oito cores de Okabe e Ito, pelo nome que a tela mostra. O vocabulário é o nome, e não o
# código hexadecimal, para a recusa do par repetido não comparar duas grafias da mesma cor.
#
# A ordem é a do contraste contra o fundo branco das figuras (razão da WCAG 2), do maior
# para o menor: azul 5,19, vermelho 3,87, verde-azulado 3,42, rosa 3,06, azul-celeste 2,31,
# laranja 2,25 e amarelo 1,32. O preto vai por último porque é a tinta dos eixos e do texto.
# O grupo `i` da coorte leva a cor `i`, então as três cores abaixo de 3:1 só entram no padrão
# a partir do quinto grupo.
_CORES: dict[CorDoGrupo, str] = {
    "azul": "#0072B2",
    "vermelho": "#D55E00",
    "verde-azulado": "#009E73",
    "rosa": "#CC79A7",
    "azul-celeste": "#56B4E9",
    "laranja": "#E69F00",
    "amarelo": "#F0E442",
    "preto": "#000000",
}
CORES_DOS_GRUPOS: Final[Mapping[CorDoGrupo, str]] = MappingProxyType(_CORES)

SimboloDoGrupo = Literal[
    "círculo",
    "quadrado",
    "triângulo",
    "losango",
    "triângulo invertido",
    "cruz",
    "xis",
    "estrela",
]

# Oito marcadores cheios do `matplotlib`, com o círculo primeiro. Cheios porque um marcador
# vazado de cor clara some na impressão em tons de cinza. O padrão só passa do círculo no
# nono grupo, quando as oito cores se esgotam.
_SIMBOLOS: dict[SimboloDoGrupo, str] = {
    "círculo": "o",
    "quadrado": "s",
    "triângulo": "^",
    "losango": "D",
    "triângulo invertido": "v",
    "cruz": "P",
    "xis": "X",
    "estrela": "*",
}
SIMBOLOS_DOS_GRUPOS: Final[Mapping[SimboloDoGrupo, str]] = MappingProxyType(_SIMBOLOS)

MARCADOR_SEM_ESTILO: Final = _SIMBOLOS["círculo"]
"""O marcador de toda série por grupo quando nenhum estilo foi passado."""

PARES_DE_ESTILO: Final[tuple[tuple[CorDoGrupo, SimboloDoGrupo], ...]] = tuple(
    (cor, simbolo) for simbolo in _SIMBOLOS for cor in _CORES
)
"""Os 64 pares distintos, com a cor variando primeiro: o par `i` é o padrão do grupo `i`."""


class GrupoInvalidoError(RecusaDoConstrutorError):
    """Um nome de grupo chegou em branco, ou não chegou como texto.

    É recusa e não um nome substituto: o nome do grupo liga a figura ao fator clínico do
    estudo, e um rótulo inventado poria a legenda a contar uma população que ninguém declarou.
    """


class EstiloInvalidoError(RecusaDoConstrutorError):
    """A cor não é da paleta, ou o símbolo não é do conjunto.

    A guarda corre em tempo de execução, em `EstiloDoGrupo`, porque o valor chega de um seletor
    da janela e não passa pelo `mypy`. Um nome desconhecido que caísse num padrão desenharia
    outra cor que a pedida.
    """


class EstiloRepetidoError(RecusaDoConstrutorError):
    """Dois grupos com a mesma cor e o mesmo símbolo.

    Fica separada de `EstiloInvalidoError` porque o conserto é outro: aqui o valor existe e já
    é de outro grupo, e a janela precisa do tipo para dizer isso.
    """


@dataclass(frozen=True)
class EstiloDoGrupo:
    """A cor e o símbolo de um grupo, pelos nomes do vocabulário fechado, validados juntos."""

    cor: CorDoGrupo
    simbolo: SimboloDoGrupo

    def __post_init__(self) -> None:
        # `isinstance` antes do `in`: um valor não hasheável faria o `in` levantar `TypeError`,
        # que não é a recusa que o painel apanha.
        if not isinstance(self.cor, str) or self.cor not in CORES_DOS_GRUPOS:
            raise EstiloInvalidoError(
                f"a cor {self.cor!r} está fora da paleta dos grupos — as oito são "
                f"{', '.join(CORES_DOS_GRUPOS)}. Um nome desconhecido não pode cair numa cor "
                "por omissão: a figura sairia com outra cor que a escolhida"
            )
        if not isinstance(self.simbolo, str) or self.simbolo not in SIMBOLOS_DOS_GRUPOS:
            raise EstiloInvalidoError(
                f"o símbolo {self.simbolo!r} está fora do conjunto dos grupos — os oito são "
                f"{', '.join(SIMBOLOS_DOS_GRUPOS)}"
            )

    @property
    def hex(self) -> str:
        """A cor como o `matplotlib` a desenha."""
        return CORES_DOS_GRUPOS[self.cor]

    @property
    def marcador(self) -> str:
        """O símbolo como código de marcador do `matplotlib`."""
        return SIMBOLOS_DOS_GRUPOS[self.simbolo]

    @property
    def par(self) -> tuple[CorDoGrupo, SimboloDoGrupo]:
        """O par que dois grupos não podem ter em comum."""
        return (self.cor, self.simbolo)

    @property
    def descricao(self) -> str:
        """A forma em que a tela e as recusas escrevem o estilo: `"vermelho com quadrado"`."""
        return f"{self.cor} com {self.simbolo}"


@dataclass(frozen=True)
class EstilosDosGrupos:
    """Um estilo por grupo, sem par repetido e congelado. É o parâmetro dos construtores.

    O par repetido é recusado aqui, e não só na tabela da janela, porque dois grupos com a
    mesma cor e o mesmo símbolo ficam indistinguíveis na figura por qualquer ponto de entrada.
    """

    por_grupo: Mapping[str, EstiloDoGrupo]

    def __post_init__(self) -> None:
        congelado: dict[str, EstiloDoGrupo] = {}
        donos: dict[tuple[CorDoGrupo, SimboloDoGrupo], str] = {}
        for grupo, estilo in self.por_grupo.items():
            nome = _nome_ou_recusa(grupo)
            if not isinstance(estilo, EstiloDoGrupo):
                raise EstiloInvalidoError(
                    f"o estilo do grupo {nome!r} chegou como {estilo!r}, fora do tipo que a "
                    "paleta e o conjunto de símbolos conferem"
                )
            if estilo.par in donos:
                raise EstiloRepetidoError(
                    f"«{donos[estilo.par]}» e «{nome}» chegaram os dois com {estilo.descricao}: "
                    "dois grupos com a mesma cor e o mesmo símbolo não se distinguem na figura"
                )
            donos[estilo.par] = nome
            congelado[nome] = estilo
        # `frozen=True` impede reatribuir o campo, mas não alterar o dicionário.
        object.__setattr__(self, "por_grupo", MappingProxyType(congelado))

    def para(self, grupos: Iterable[str]) -> dict[str, EstiloDoGrupo]:
        """O estilo de cada grupo de uma figura, na ordem canônica.

        O grupo que a coorte não tem recebe o primeiro par que ela não usa. É o caso do QC no
        K-means e no SOM, que correm sobre a matriz inteira enquanto a composição por classe
        deixa o QC de fora. Com os 64 pares tomados, os que faltam repetem a partir do
        primeiro: a legenda continua nomeando cada série.
        """
        tomados = {estilo.par for estilo in self.por_grupo.values()}
        livres = [par for par in PARES_DE_ESTILO if par not in tomados] or list(PARES_DE_ESTILO)
        resultado: dict[str, EstiloDoGrupo] = {}
        faltantes = 0
        for grupo in ordem_canonica(grupos):
            estilo = self.por_grupo.get(grupo)
            if estilo is None:
                cor, simbolo = livres[faltantes % len(livres)]
                estilo = EstiloDoGrupo(cor=cor, simbolo=simbolo)
                faltantes += 1
            resultado[grupo] = estilo
        return resultado


@dataclass(frozen=True)
class SerieDeGrupo:
    """Um grupo clínico desenhável: quem é, quantos são, com que cor e onde está no eixo."""

    grupo: str
    """O nome tal como veio do dado, sem normalização."""

    mascara: np.ndarray
    """Booleano por amostra, na ordem do vetor recebido. `True` onde a amostra é deste grupo."""

    cor: Any
    """Do ciclo dos `rcParams`, ou da paleta dos grupos quando há estilo. `Any` porque o ciclo
    pode trazer uma cadeia hexadecimal ou uma tupla RGB."""

    posicao: int
    """O índice na ordem canônica. Sem estilo é ele que decide a cor; com estilo, quem decide
    é a coorte informada pelo chamador."""

    marcador: str = MARCADOR_SEM_ESTILO
    """O código de marcador do `matplotlib`: o círculo, ou o símbolo escolhido. Barras e
    rótulos não o leem, porque ali o grupo é só cor."""

    @property
    def n(self) -> int:
        return int(self.mascara.sum())

    @property
    def rotulo(self) -> str:
        """A entrada de legenda, com a contagem."""
        return rotulo_do_grupo(self.grupo, self.n)


def grupos_do_fio(valores: Any) -> tuple[str, ...]:
    """O vetor de grupos por amostra vindo do JSON, com a guarda de fronteira.

    Um nome que chegue `null`, numérico ou vazio é recusado: aceito, a amostra sairia de todas
    as séries e a legenda contaria menos do que a matriz tem.

    O que não é lista devolve tupla vazia, e o construtor levanta `MatrizVaziaError`.
    """
    if not isinstance(valores, (list, tuple)):
        return ()
    return tuple(_nome_ou_recusa(v) for v in valores)


def _nome_ou_recusa(valor: Any) -> str:
    """Um nome de grupo não vazio, ou a recusa."""
    if isinstance(valor, str) and valor.strip():
        return valor
    raise GrupoInvalidoError(
        f"o grupo de uma amostra chegou como {valor!r}. O nome do grupo é o que liga a figura ao "
        "fator clínico do estudo, e um recuo aqui poria a legenda a nomear uma população que "
        "ninguém declarou — com a figura perfeitamente bem formada"
    )


def ordem_canonica(grupos: Iterable[str]) -> tuple[str, ...]:
    """Os grupos distintos, ordenados por ponto de código.

    Não é ordem alfabética de língua nenhuma (`ATLL` vem antes de `Asymptomatic`), mas é a
    mesma em qualquer máquina e não depende da ordem em que os grupos chegam.
    """
    return tuple(sorted(set(grupos)))


def cores_dos_grupos(grupos: Iterable[str]) -> dict[str, Any]:
    """Uma cor por grupo, na ordem canônica, tiradas do ciclo dos `rcParams`.

    O ciclo é lido no momento da construção, e assim o tema configurado pela janela chega
    aqui. Com mais grupos do que cores no ciclo, a atribuição volta ao início: a cor deixa de
    distinguir as séries, e por isso toda legenda leva o nome do grupo.
    """
    canonica = ordem_canonica(grupos)
    return dict(zip(canonica, cores_do_ciclo(len(canonica)), strict=True))


def estilos_dos_grupos(
    grupos: Iterable[str], escolhidos: Mapping[str, EstiloDoGrupo] | None = None
) -> EstilosDosGrupos | None:
    """Um estilo por grupo da coorte: o escolhido, ou o par da posição canônica.

    O grupo `i` leva o par `i` de `PARES_DE_ESTILO`, seja qual for o subconjunto que uma figura
    desenhe. É o que mantém a cor de um grupo entre dois contrastes do PLS-DA.

    O grupo cujo par padrão foi tomado por uma escolha recebe o primeiro par livre, e só ele
    muda. Os pares padrão dos demais são reservados antes, para a troca não se propagar. A
    escolha feita para um grupo fora da coorte é ignorada, e duas escolhas iguais dentro dela
    são recusadas por `EstilosDosGrupos`.

    Devolve `None` sem grupo nenhum e com mais grupos do que os 64 pares distintos. Nos dois
    casos os construtores desenham com o ciclo e o círculo.
    """
    canonica = ordem_canonica(grupos)
    if not canonica or len(canonica) > len(PARES_DE_ESTILO):
        return None
    validos = {g: e for g, e in (escolhidos or {}).items() if g in canonica}
    tomados = {estilo.par for estilo in validos.values()}
    resolvidos: dict[str, EstiloDoGrupo] = dict(validos)
    deslocados: list[str] = []
    for posicao, grupo in enumerate(canonica):
        if grupo in validos:
            continue
        cor, simbolo = PARES_DE_ESTILO[posicao]
        if (cor, simbolo) in tomados:
            deslocados.append(grupo)
            continue
        resolvidos[grupo] = EstiloDoGrupo(cor=cor, simbolo=simbolo)
        tomados.add((cor, simbolo))
    # Não esgota: há 64 pares, no máximo 64 grupos, e cada grupo ocupa um par só.
    livres = iter([par for par in PARES_DE_ESTILO if par not in tomados])
    for grupo in deslocados:
        cor, simbolo = next(livres)
        resolvidos[grupo] = EstiloDoGrupo(cor=cor, simbolo=simbolo)
    return EstilosDosGrupos({grupo: resolvidos[grupo] for grupo in canonica})


def rotulo_do_grupo(grupo: str, quantas: int) -> str:
    """A entrada de legenda de um grupo, com a contagem: `"HAMTSP — 14 amostras"`.

    A contagem é por série porque é ela que decide o que se pode ler: o total esconde a
    diferença de tamanho entre os grupos.
    """
    plural = "amostra" if quantas == 1 else "amostras"
    return f"{grupo} — {milhar(quantas)} {plural}"


def series_por_grupo(
    grupos: Sequence[str], *, estilos: EstilosDosGrupos | None = None
) -> tuple[SerieDeGrupo, ...]:
    """Uma série por grupo presente no vetor, na ordem canônica, com cor, máscara e rótulo.

    `grupos` é o vetor por amostra, na ordem das linhas da matriz. A máscara devolvida indexa
    os vetores de coordenadas, que chegam na mesma ordem.

    Sem `estilos`, a cor sai da ordem canônica dos grupos presentes: um subconjunto da coorte
    leva as primeiras cores do ciclo, e não as que teria na figura da coorte inteira. Com
    `estilos`, a cor e o marcador são os resolvidos sobre a coorte do chamador.
    """
    presentes = ordem_canonica(grupos)
    if estilos is None:
        paleta = cores_dos_grupos(presentes)
        marcadores = dict.fromkeys(presentes, MARCADOR_SEM_ESTILO)
    else:
        escolhidos = estilos.para(presentes)
        paleta = {nome: estilo.hex for nome, estilo in escolhidos.items()}
        marcadores = {nome: estilo.marcador for nome, estilo in escolhidos.items()}
    vetor = np.array(list(grupos), dtype=object)
    return tuple(
        SerieDeGrupo(
            grupo=nome,
            mascara=np.array([g == nome for g in vetor], dtype=bool),
            cor=paleta[nome],
            posicao=posicao,
            marcador=marcadores[nome],
        )
        for posicao, nome in enumerate(presentes)
    )
