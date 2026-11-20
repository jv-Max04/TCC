"""O recorte da entrada do enriquecimento pelo MetaboAnalystR.

Da tabela de identificação do Progenesis e do teste univariado da execução sai a lista que vai à
ORA e o fundo contra o qual ela é testada. Uma análise faz duas chamadas ao R: `recorta` monta a
lista e o fundo com os nomes crus, o R os mapeia, e `prepara_ora` aplica as recusas sobre o
resultado do mapeamento. Tudo aqui opera em memória; quem lê a tabela é `io/identificacao.py`, e
quem chama o R é `io/enriquecimento_r.py`.

O corte é por composto, não por feature: um composto com k features passa por acaso com
probabilidade 1-(1-a)^k, com `a` o limiar, se se olhar a feature mais significativa. Por isso o p
do composto é o mínimo das suas features corrigido pelo número delas (Šidák). Feature com vários
candidatos usa o marcado como aceito, ou o primeiro. Lista de 1 ou 2 compostos é calculada, com
aviso: a ORA que o roteiro chama, `CalculateHyperScore` (`R/enrich_stats.R:17-136` no commit
`6bf5462` do MetaboAnalystR), só recusa lista vazia, biblioteca de um conjunto só e zero acertos.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Final, Literal, cast, get_args

import numpy as np
from scipy.stats import hypergeom

from refino.statistics.univariate import benjamini_hochberg

__all__ = [
    "POUCOS_COMPOSTOS",
    "CompostoTestado",
    "Corte",
    "Degraus",
    "EntradaInvalidaError",
    "EnvioRecusadoError",
    "Mapeado",
    "PreparoDaOra",
    "RecorteDaIdentificacao",
    "Recusa",
    "RespostaInvalidaError",
    "ResultadoDaOra",
    "ViaEnriquecida",
    "aviso_de_poucos",
    "canonicos_do_envio",
    "confere_g1",
    "mapeamento_da_resposta",
    "ora_da_resposta",
    "p_da_via_confere",
    "p_hipergeometrico",
    "prepara_ora",
    "recorta",
    "recusa_previa",
]

TipoDeCorte = Literal["p", "q", "n_melhores"]
MotivoDeRecusa = Literal["lista_vazia", "igual_ao_fundo", "nada_mapeou"]
# Abaixo disto o resultado sai com aviso, e não é recusado. Não é limiar de decisão: é onde uma
# via com um único acerto passa a ser a maior parte da lista.
POUCOS_COMPOSTOS: Final = 3
# A ORA do MSEA arredonda a tabela a três algarismos significativos (`signif(., 3)`), então o p
# devolvido difere do exato em até meia unidade do terceiro algarismo: 5e-3 relativo.
_TOLERANCIA_RELATIVA_DO_SIGNIF: Final = 5e-3


class EntradaInvalidaError(ValueError):
    """A entrada não pode ser recortada; a mensagem diz por quê."""


class EnvioRecusadoError(ValueError):
    """A carga do modo externo contém texto que não pode sair da máquina."""


@dataclass(frozen=True)
class Corte:
    tipo: TipoDeCorte
    valor: float

    def __post_init__(self) -> None:
        if self.tipo not in get_args(TipoDeCorte):
            raise EntradaInvalidaError(f"corte desconhecido: {self.tipo!r}")
        if self.tipo == "n_melhores":
            if self.valor < 1 or self.valor != int(self.valor):
                raise EntradaInvalidaError(
                    f"N melhores precisa de um inteiro positivo: {self.valor!r}"
                )
        elif not 0.0 < self.valor < 1.0:
            raise EntradaInvalidaError(
                f"o limiar do corte por {self.tipo} fica entre 0 e 1: {self.valor!r}"
            )


@dataclass(frozen=True)
class Degraus:
    """As contagens de cada degrau do recorte, na ordem em que a tela as mostra."""

    linhas_lidas: int
    features_identificadas: int
    compostos_distintos: int
    features_em_comum: int
    features_sem_p: int
    # Em feature com vários candidatos, os não escolhidos.
    candidatos_descartados: int
    # Compostos identificados cuja feature não está na execução (saiu no processamento): ficam fora
    # do fundo, e a tela os conta para que o encolhimento não pareça perda silenciosa.
    compostos_fora_da_execucao: int


@dataclass(frozen=True)
class CompostoTestado:
    nome: str
    p: float
    features: tuple[str, ...]


@dataclass(frozen=True)
class RecorteDaIdentificacao:
    degraus: Degraus
    corte: Corte
    testados: tuple[CompostoTestado, ...]
    lista: tuple[str, ...]
    fundo: tuple[str, ...]
    esperado_por_acaso: float
    descricao_do_esperado: str


@dataclass(frozen=True)
class Recusa:
    motivo: MotivoDeRecusa
    texto: str
    oferece_fundo_da_biblioteca: bool = False


@dataclass(frozen=True)
class Mapeado:
    """Uma linha da resposta de `mapear`: o nome consultado, o canônico e o identificador."""

    consulta: str
    casado: str | None
    hmdb: str | None


@dataclass(frozen=True)
class PreparoDaOra:
    lista: tuple[str, ...]
    # `None` é o fundo da biblioteca inteira, que só existe quando o usuário o aceitou.
    fundo: tuple[str, ...] | None
    nao_mapeados_da_lista: tuple[str, ...]
    nao_mapeados_do_fundo: tuple[str, ...]


def _sidak(p_minimo: float, k: int) -> float:
    # -expm1(k·log1p(-p)) é 1-(1-p)^k sem a perda de precisão de p pequeno.
    if p_minimo >= 1.0:
        return 1.0
    return float(-math.expm1(k * math.log1p(-p_minimo)))


def _escolhe_candidatos(
    linhas: Sequence[tuple[str, str, bool | None]],
) -> tuple[dict[str, str], int]:
    """Um composto por feature: o aceito, ou o primeiro. Devolve o mapa e os descartados."""
    por_feature: dict[str, list[tuple[str, bool | None]]] = {}
    for feature, composto, aceito in linhas:
        por_feature.setdefault(feature, []).append((composto, aceito))
    escolha: dict[str, str] = {}
    descartados = 0
    for feature, candidatos in por_feature.items():
        aceitos = [c for c, a in candidatos if a]
        escolha[feature] = aceitos[0] if aceitos else candidatos[0][0]
        descartados += len({c for c, _ in candidatos}) - 1
    return escolha, descartados


def _compostos(n: int) -> str:
    return "1 composto" if n == 1 else f"{n} compostos"


def _decimal(numero: str) -> str:
    """Vírgula decimal: o texto é lido na tela, em português, ao lado de números que já a usam."""
    return numero.replace(".", ",")


def recorta(
    linhas: Sequence[tuple[str, str, bool | None]],
    features: Sequence[str],
    pvalues: Sequence[float],
    corte: Corte,
) -> RecorteDaIdentificacao:
    """Monta a lista e o fundo, com nomes crus, e as contagens de cada degrau.

    `features` e `pvalues` são as duas colunas do teste univariado da execução
    (`UnivariateResult.feature` e `.pvalue`); `NaN` é feature sem p calculado.
    """
    if not linhas:
        raise EntradaInvalidaError("a tabela de identificação não tem linhas")
    escolha, descartados = _escolhe_candidatos(linhas)
    p_por_feature = {f: float(p) for f, p in zip(features, pvalues, strict=True)}
    em_comum = [f for f in escolha if f in p_por_feature]
    com_p = [f for f in em_comum if not math.isnan(p_por_feature[f])]
    features_por_composto: dict[str, list[str]] = {}
    for f in com_p:
        features_por_composto.setdefault(escolha[f], []).append(f)
    compostos_identificados = set(escolha.values())
    na_execucao = {escolha[f] for f in em_comum}
    testados = sorted(
        (
            CompostoTestado(
                nome,
                _sidak(min(p_por_feature[f] for f in fs), len(fs)),
                tuple(sorted(fs)),
            )
            for nome, fs in features_por_composto.items()
        ),
        key=lambda c: (c.p, c.nome),
    )
    n = len(testados)
    if corte.tipo == "p":
        lista = [c.nome for c in testados if c.p < corte.valor]
        esperado = corte.valor * n
        descricao = (
            f"com p < {_decimal(f'{corte.valor:g}')}, cerca de {_decimal(f'{esperado:.1f}')} "
            f"dos {n} compostos "
            "passariam por acaso"
        )
    elif corte.tipo == "q":
        q = (
            benjamini_hochberg(np.array([c.p for c in testados], dtype=float))
            if n
            else np.array([])
        )
        lista = [c.nome for c, qv in zip(testados, q, strict=True) if qv < corte.valor]
        esperado = corte.valor * len(lista)
        descricao = (
            f"com q < {_decimal(f'{corte.valor:g}')}, espera-se que cerca de "
            f"{_decimal(f'{esperado:.1f}')} dos {len(lista)} "
            "compostos da lista sejam falsas descobertas"
        )
    else:
        n_melhores = int(corte.valor)
        lista = [c.nome for c in testados[:n_melhores]]
        alfa = testados[min(n_melhores, n) - 1].p if n else 0.0
        esperado = alfa * n
        descricao = (
            f"os {len(lista)} melhores vão até p = {_decimal(f'{alfa:.3g}')}; nesse limiar, "
            f"cerca de {_decimal(f'{esperado:.1f}')} "
            f"dos {n} compostos passariam por acaso"
        )
    degraus = Degraus(
        linhas_lidas=len(linhas),
        features_identificadas=len(escolha),
        compostos_distintos=len(compostos_identificados),
        features_em_comum=len(em_comum),
        features_sem_p=len(em_comum) - len(com_p),
        candidatos_descartados=descartados,
        compostos_fora_da_execucao=len(compostos_identificados - na_execucao),
    )
    return RecorteDaIdentificacao(
        degraus,
        corte,
        tuple(testados),
        tuple(lista),
        tuple(c.nome for c in testados),
        esperado,
        descricao,
    )


def recusa_previa(recorte: RecorteDaIdentificacao) -> Recusa | None:
    """Aviso antecipado, antes de chamar o R. A recusa que vale é a de `prepara_ora`."""
    if not recorte.lista:
        return Recusa("lista_vazia", "nenhum composto passa o corte escolhido")
    if set(recorte.lista) == set(recorte.fundo):
        return _igual_ao_fundo()
    return None


def aviso_de_poucos(na_lista: int) -> str | None:
    """O aviso de lista curta, ou `None` quando a lista não é curta.

    `na_lista` é a contagem que o R devolve, a lista depois do mapeamento e do universo, e não a
    do recorte: é sobre ela que a ORA foi calculada. Zero não chega aqui, porque a lista sem
    composto no universo é recusada antes.
    """
    if not 0 < na_lista < POUCOS_COMPOSTOS:
        return None
    return (
        f"A lista tem só {_compostos(na_lista)} no universo: com tão poucos, uma via com um único "
        "acerto já aparece enriquecida. Leia o resultado com cautela."
    )


def _igual_ao_fundo() -> Recusa:
    return Recusa(
        "igual_ao_fundo",
        "a lista é igual ao fundo: todo composto do ensaio está na lista, e então todo p sai 1",
        oferece_fundo_da_biblioteca=True,
    )


def _canonicos(mapa: Sequence[Mapeado]) -> tuple[dict[str, str], tuple[str, ...]]:
    """Identificador → nome canônico, deduplicado pelo identificador; e os que não mapearam."""
    por_id: dict[str, str] = {}
    nao_mapeados: list[str] = []
    for m in mapa:
        if m.hmdb and m.casado:
            por_id.setdefault(m.hmdb, m.casado)
        else:
            nao_mapeados.append(m.consulta)
    return por_id, tuple(nao_mapeados)


def canonicos_do_envio(mapa_lista: Sequence[Mapeado]) -> tuple[str, ...] | Recusa:
    """Os nomes que o modo externo envia: os canônicos que o mapeamento local confirmou.

    Usa a mesma regra de `_canonicos`: um nome só sai se o mapeamento devolveu o canônico e o
    identificador HMDB, e a lista se deduplica pelo identificador, de modo que dois sinônimos da
    mesma molécula contam uma vez, como na ORA local.

    Não há fundo: o servidor devolve o mesmo universo e os mesmos valores com e sem o filtro do
    fundo na ORA KEGG, então o modo externo não o envia nem o mapeia. Por isso também não há aqui
    a recusa de lista igual ao fundo.

    A saída é ordenada e sem repetição de nome: dois identificadores com o mesmo canônico seriam o
    mesmo texto na carga, e `monta_carga` recusa nome repetido.
    """
    por_id, _ = _canonicos(mapa_lista)
    if not por_id:
        return Recusa(
            "nada_mapeou", "nenhum composto da lista foi reconhecido pela base do MetaboAnalyst"
        )
    return tuple(sorted(set(por_id.values())))


def prepara_ora(
    mapa_lista: Sequence[Mapeado],
    mapa_fundo: Sequence[Mapeado],
    *,
    fundo_da_biblioteca: bool = False,
) -> PreparoDaOra | Recusa:
    """Aplica as recusas sobre o resultado do mapeamento e devolve a carga da ORA."""
    lista, nao_lista = _canonicos(mapa_lista)
    fundo, nao_fundo = _canonicos(mapa_fundo)
    efetiva = set(lista) if fundo_da_biblioteca else set(lista) & set(fundo)
    if not efetiva:
        return Recusa(
            "nada_mapeou", "nenhum composto da lista foi reconhecido pela base do MetaboAnalyst"
        )
    if not fundo_da_biblioteca and efetiva == set(fundo):
        return _igual_ao_fundo()
    return PreparoDaOra(
        tuple(sorted(lista[i] for i in efetiva)),
        None if fundo_da_biblioteca else tuple(sorted(fundo.values())),
        nao_lista,
        nao_fundo,
    )


def confere_g1(
    textos: Iterable[str], *, permitidos: Iterable[str], proibidos: Iterable[str]
) -> None:
    """A primeira camada da guarda do envio externo. Levanta `EnvioRecusadoError` dizendo o quê.

    Só passa o que o mapeamento local confirmou, e nada que repita um nome de amostra, um rótulo
    de grupo ou um cabeçalho da tabela. Vocabulário proibido vazio é erro, porque aprovaria tudo.
    """
    vetados = {p.casefold() for p in proibidos if p.strip()}
    if not vetados:
        raise ValueError(
            "a guarda do envio externo precisa do vocabulário proibido (amostras, grupos, "
            "cabeçalho): "
            "vazio aprovaria tudo"
        )
    autorizados = set(permitidos)
    for texto in textos:
        if texto.casefold() in vetados:
            raise EnvioRecusadoError(
                f"«{texto}» é um nome da execução ou da tabela, e não sai da máquina"
            )
        if texto not in autorizados:
            raise EnvioRecusadoError(f"«{texto}» não é um nome que o mapeamento local confirmou")


def p_hipergeometrico(acertos: int, total: int, na_lista: int, universo: int) -> float:
    """O p da ORA como o MSEA o calcula.

    É o `phyper(acertos - 1, total, universo - total, na_lista)` de `enrich_stats.R`.
    """
    return float(hypergeom.sf(acertos - 1, universo, total, na_lista))


def p_da_via_confere(
    p_informado: float, acertos: int, total: int, na_lista: int, universo: int
) -> bool:
    """Diz se o p que voltou do R bate com o recalculado, dentro do arredondamento do pacote.

    Divergência é tratada como resposta malformada. Um erro comum aos dois hospedeiros, como
    coluna trocada ou universo declarado diferente do usado, passaria na paridade entre eles, e
    não passa aqui.
    """
    esperado = p_hipergeometrico(acertos, total, na_lista, universo)
    return math.isclose(
        p_informado, esperado, rel_tol=_TOLERANCIA_RELATIVA_DO_SIGNIF, abs_tol=1e-300
    )


# ---------------------------------------------------------------------------
# A resposta do roteiro R, lida para estruturas tipadas
# ---------------------------------------------------------------------------
#
# A janela não lê o JSON do R por chave. O dicionário vira dataclass aqui, no núcleo, que recebe o
# objeto já decodificado e não abre processo nem arquivo. Um campo fora do contrato levanta
# `RespostaInvalidaError`, que o painel mostra como resposta malformada.


class RespostaInvalidaError(ValueError):
    """A resposta do roteiro não tem a forma do contrato."""


@dataclass(frozen=True)
class ViaEnriquecida:
    via: str
    total: int
    esperado: float
    acertos: int
    p: float
    fdr: float
    compostos: tuple[str, ...]


# O que o roteiro R declara como universo da ORA. É conferido na leitura da resposta: um valor
# fora dele cairia no ramo da biblioteca inteira na tela, que é outra afirmação.
TipoDoUniverso = Literal["fundo_do_ensaio", "biblioteca"]
TIPOS_DO_UNIVERSO: Final = frozenset(get_args(TipoDoUniverso))


@dataclass(frozen=True)
class ResultadoDaOra:
    biblioteca: str
    tipo_do_universo: TipoDoUniverso
    tamanho_do_universo: int
    na_lista: int
    vias: tuple[ViaEnriquecida, ...]


def _campo(dados: object, nome: str) -> object:
    if not isinstance(dados, dict) or nome not in dados:
        raise RespostaInvalidaError(f"a resposta não tem o campo «{nome}»")
    return dados[nome]


def _mapeados(bloco: object) -> tuple[Mapeado, ...]:
    linhas = _campo(bloco, "mapeamento")
    if not isinstance(linhas, list):
        raise RespostaInvalidaError("«mapeamento» não é lista")
    saida: list[Mapeado] = []
    for linha in linhas:
        consulta, casado, hmdb = (
            _campo(linha, "consulta"),
            _campo(linha, "casado"),
            _campo(linha, "hmdb"),
        )
        if (
            not isinstance(consulta, str)
            or not isinstance(casado, (str, type(None)))
            or not isinstance(hmdb, (str, type(None)))
        ):
            raise RespostaInvalidaError(f"linha de mapeamento fora do contrato: {linha!r}")
        saida.append(Mapeado(consulta, casado, hmdb))
    return tuple(saida)


def mapeamento_da_resposta(dados: object) -> tuple[tuple[Mapeado, ...], tuple[Mapeado, ...]]:
    """A lista e o fundo mapeados. `"fundo": null`, do pedido sem fundo, vira tupla vazia.

    A chave ausente levanta: o roteiro declara `fundo` em toda resposta de `mapear`, `null` quando
    não houve fundo (`r_service/vias_entrypoint.R`). Uma resposta sem ela é de outro contrato, e
    tratá-la como pedido sem fundo mudaria o universo da ORA em silêncio.
    """
    fundo = _campo(dados, "fundo")
    return _mapeados(_campo(dados, "lista")), (() if fundo is None else _mapeados(fundo))


def _numero(valor: object, nome: str) -> float:
    if isinstance(valor, bool) or not isinstance(valor, (int, float)) or math.isnan(float(valor)):
        raise RespostaInvalidaError(f"«{nome}» não é número: {valor!r}")
    return float(valor)


def _inteiro(valor: object, nome: str) -> int:
    x = _numero(valor, nome)
    if x != int(x) or x < 0:
        raise RespostaInvalidaError(f"«{nome}» não é contagem: {valor!r}")
    return int(x)


def ora_da_resposta(dados: object) -> ResultadoDaOra:
    """Lê a tabela de vias, conferindo cada via contra o p hipergeométrico recalculado.

    O p que voltou do R tem de bater com o recalculado a partir das contagens e do universo
    declarado. Divergência levanta, e a tela a mostra como resposta malformada.
    """
    universo = _campo(dados, "universo")
    tipo = _campo(universo, "tipo")
    if tipo not in TIPOS_DO_UNIVERSO:
        raise RespostaInvalidaError(f"«universo.tipo» fora do contrato: {tipo!r}")
    tamanho = _inteiro(_campo(universo, "tamanho"), "universo.tamanho")
    na_lista = _inteiro(_campo(dados, "na_lista"), "na_lista")
    vias_brutas = _campo(dados, "vias")
    if not isinstance(vias_brutas, list):
        raise RespostaInvalidaError("«vias» não é lista")
    vias: list[ViaEnriquecida] = []
    for v in vias_brutas:
        compostos = _campo(v, "compostos")
        if not isinstance(compostos, list) or not all(isinstance(c, str) for c in compostos):
            raise RespostaInvalidaError(f"«compostos» fora do contrato em {v!r}")
        via = ViaEnriquecida(
            via=str(_campo(v, "via")),
            total=_inteiro(_campo(v, "total"), "total"),
            esperado=_numero(_campo(v, "esperado"), "esperado"),
            acertos=_inteiro(_campo(v, "acertos"), "acertos"),
            p=_numero(_campo(v, "p"), "p"),
            fdr=_numero(_campo(v, "fdr"), "fdr"),
            compostos=tuple(compostos),
        )
        if len(via.compostos) != via.acertos:
            raise RespostaInvalidaError(
                f"{via.via}: {via.acertos} acertos e {len(via.compostos)} compostos"
            )
        if not p_da_via_confere(via.p, via.acertos, via.total, na_lista, tamanho):
            raise RespostaInvalidaError(
                f"{via.via}: o p devolvido ({via.p:.3g}) não é o hipergeométrico das contagens"
            )
        vias.append(via)
    return ResultadoDaOra(
        biblioteca=str(_campo(dados, "biblioteca")),
        tipo_do_universo=cast(TipoDoUniverso, tipo),
        tamanho_do_universo=tamanho,
        na_lista=na_lista,
        vias=tuple(sorted(vias, key=lambda x: (x.p, x.via))),
    )
