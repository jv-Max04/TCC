"""A estrutura do relatório, e a conferência do conteúdo obrigatório sobre ela.

O relatório é um objeto antes de ser HTML. `defeitos_do_relatorio` confere o objeto, e não a
string renderizada, para não medir o gerador de template.

É um documento com duas leituras. O corpo é para o pesquisador e traz o que ele pode usar para
agir; o apêndice é para o revisor e traz o que justifica confiar no que está escrito. Cada afirmação
do corpo carrega o identificador do item de apêndice que a sustenta, e a conferência reprova
ponteiro que não resolve. Os avisos aparecem nas duas leituras, em campos separados e com textos
diferentes. A tabela de métricas é conferida contra o conjunto que o `QualityReport` declara ter
calculado, e não contra uma lista escrita aqui.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Final

from refino.reporting.avisos import AVISOS

__all__ = [
    "ITENS_OBRIGATORIOS",
    "Afirmacao",
    "ItemApendice",
    "ItemCorpo",
    "Relatorio",
    "defeitos_do_relatorio",
]


@dataclass(frozen=True)
class ItemCanonico:
    """Um item do conteúdo obrigatório do relatório."""

    numero: int
    titulo: str


# Os itens obrigatórios, em ordem. É a lista de referência, como `AVISOS`: a conferência compara o
# relatório contra ela, nunca contra o que o relatório produziu.
ITENS_OBRIGATORIOS: Final[tuple[ItemCanonico, ...]] = (
    ItemCanonico(1, "Identificação do estudo, do modo e do arquivo de origem, com checksum"),
    ItemCanonico(2, "Problemas de validação encontrados na ingestão"),
    ItemCanonico(3, "Perfil de diagnóstico, com trajetória dos QCs e pontos de mudança"),
    ItemCanonico(4, "Recomendação, com justificativa em texto e alternativas consideradas"),
    ItemCanonico(5, "Tabela de qualidade antes e depois, em todas as métricas"),
    ItemCanonico(6, "Gráfico de PCA antes e depois, com QCs destacados"),
    ItemCanonico(7, "Curva de especificação"),
    ItemCanonico(8, "Lista de features robust, com a advertência"),
    ItemCanonico(9, "Registro completo e reexecutável"),
    ItemCanonico(
        10, "Dispersão do RSD por feature, das amostras contra a dos QCs, com o n de cada lado"
    ),
    ItemCanonico(
        11, "Composição por classe, com o percentual de células faltantes e a abundância típica"
    ),
    # O item 12 só tem conteúdo quando o corretor é o LOESS; nos outros ele é `vazio_declarado`.
    # O título diz que o envelope é sensibilidade à remoção de um QC, para não ser lido como
    # intervalo de confiança.
    ItemCanonico(
        12,
        "Curva do LOESS ao longo das injeções, com a predição de cada QC pela curva que não o viu "
        "e a sensibilidade à remoção de um QC, que não é intervalo",
    ),
)


@dataclass(frozen=True)
class Afirmacao:
    """Uma afirmação do corpo, com o ponteiro para o item de apêndice que a limita."""

    texto: str
    ponteiro: str


@dataclass(frozen=True)
class ItemCorpo:
    """Um item do corpo, na leitura do pesquisador: a consequência, o que ele pode usar.

    `vazio_declarado` é a legenda de um item sem conteúdo. Nos dados de referência nenhuma feature
    sobrevive ao FDR, então a curva de especificação e a lista `robust` saem vazias, e um gráfico
    vazio sem legenda é indistinguível de um gráfico quebrado.
    """

    numero: int
    titulo: str
    afirmacoes: tuple[Afirmacao, ...] = ()
    vazio_declarado: str | None = None


@dataclass(frozen=True)
class ItemApendice:
    """Um item de procedência, na leitura do revisor: o que justifica confiar."""

    id: str
    titulo: str
    conteudo: str


@dataclass(frozen=True)
class Relatorio:
    """O relatório inteiro: um documento com dois caminhos de leitura, o corpo e o apêndice."""

    corpo: tuple[ItemCorpo, ...]
    apendice: tuple[ItemApendice, ...]
    metricas: Mapping[str, float]
    metricas_computadas: frozenset[str]
    avisos_corpo: Mapping[str, str] = field(default_factory=dict)
    avisos_apendice: Mapping[str, str] = field(default_factory=dict)


def defeitos_do_relatorio(rel: Relatorio) -> list[str]:
    """Todos os defeitos do relatório, um por linha. Lista vazia quer dizer relatório conforme.

    Devolve a lista em vez de levantar, para que um teste possa plantar um defeito e ver a
    acusação pelo mesmo caminho.
    """
    achados: list[str] = []
    achados += _defeitos_dos_onze_itens(rel)
    achados += _defeitos_das_metricas(rel)
    achados += _defeitos_dos_avisos(rel)
    achados += _defeitos_dos_ponteiros(rel)
    return achados


def _defeitos_dos_onze_itens(rel: Relatorio) -> list[str]:
    """Confere os itens obrigatórios, localizados um por um pelo número, e não contados.

    Contar deixaria passar o número certo de seções vazias. O nome fala em onze porque a lista já
    teve onze itens; o que a função confere é `ITENS_OBRIGATORIOS`, que hoje tem doze.
    """
    presentes = {i.numero for i in rel.corpo}
    faltando = [c for c in ITENS_OBRIGATORIOS if c.numero not in presentes]
    achados = [
        f"item {c.numero} do conteúdo obrigatório ausente do corpo: {c.titulo!r}" for c in faltando
    ]
    for item in rel.corpo:
        if not item.afirmacoes and item.vazio_declarado is None:
            achados.append(
                f"item {item.numero} presente e SEM conteúdo: nenhuma afirmação e nenhum vazio "
                "declarado. Seção vazia sem legenda é indistinguível de seção quebrada"
            )
    return achados


def _defeitos_das_metricas(rel: Relatorio) -> list[str]:
    """Confere a tabela de métricas nas duas direções.

    Toda métrica calculada tem de estar na tabela, e a tabela tem de ter mais de uma: qualidade
    nunca é um número só.
    """
    achados: list[str] = []
    ausentes = sorted(rel.metricas_computadas - set(rel.metricas))
    if ausentes:
        achados.append(
            f"métricas calculadas e ausentes da tabela: {ausentes}. O `computed` do "
            "`QualityReport` é a referência, e o relatório não pode mostrar menos do que mediu"
        )
    if len(rel.metricas) <= 1:
        achados.append(
            f"a tabela de qualidade tem {len(rel.metricas)} métrica(s). Métrica única é proibida, "
            "e o porquê está medido: RSD e agrupamento de QC no PCA discordam "
            "e elegem métodos diferentes"
        )
    return achados


def _defeitos_dos_avisos(rel: Relatorio) -> list[str]:
    """Confere os três avisos nas duas leituras, com textos diferentes em cada uma.

    `avisos_corpo` e `avisos_apendice` são campos separados, então as duas checagens são
    independentes. Texto idêntico nos dois também reprova.
    """
    achados: list[str] = []
    for aviso in AVISOS:
        no_corpo = rel.avisos_corpo.get(aviso.id, "").strip()
        no_apendice = rel.avisos_apendice.get(aviso.id, "").strip()
        if not no_corpo:
            achados.append(
                f"aviso `{aviso.id}` ausente do CORPO. Ele limita o que o leitor pode concluir — "
                f"{aviso.por_que_limita} —, e limite em apêndice é limite que não foi lido"
            )
        if not no_apendice:
            achados.append(
                f"aviso `{aviso.id}` ausente do APÊNDICE. Sem a procedência o revisor perde a "
                "justificação, e a consequência no corpo fica sem origem"
            )
        if no_corpo and no_corpo == no_apendice:
            achados.append(
                f"aviso `{aviso.id}`: corpo e apêndice têm o MESMO texto. Pede-se consequência "
                "no corpo e procedência no apêndice; texto idêntico é um enquadramento só, e as "
                "duas verificações passam a medir a mesma coisa"
            )
    return achados


def _defeitos_dos_ponteiros(rel: Relatorio) -> list[str]:
    """Confere que toda afirmação do corpo tem ponteiro e que ele resolve no apêndice."""
    ids = {a.id for a in rel.apendice}
    achados: list[str] = []
    for item in rel.corpo:
        for af in item.afirmacoes:
            if not af.ponteiro:
                achados.append(
                    f"item {item.numero}: afirmação sem ponteiro — {af.texto[:60]!r}. Afirmação do "
                    "corpo sem ponteiro repete o defeito de `best_by_metric` devolvendo "
                    "`fingerprint` sozinho"
                )
            elif af.ponteiro not in ids:
                achados.append(
                    f"item {item.numero}: ponteiro `{af.ponteiro}` não resolve no apêndice. "
                    f"Existem: {sorted(ids)}"
                )
    return achados
