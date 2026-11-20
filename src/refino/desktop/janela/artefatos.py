"""O painel dos artefatos: um download por chave, com o controle de salvar existindo se e somente
se houve bytes.

Um controle mostrado antes de a operação correr é promessa vazia, e um ausente depois de ela
correr perde o arquivo já pago; por isso `Resposta.conteudo` é `bytes | None`, e não `b""`. As
rotas de artefato respondem `202`, e os bytes vêm de `GET /trabalhos/{id}`: a terminalidade é o
código HTTP, nunca um campo do corpo. O painel escreve os bytes como a rota os devolveu, sem
reprocessar, e cada artefato tem um botão só, que pede a pasta no clique e escreve quando os bytes
chegam.

Clicar de novo regenera, em vez de servir os bytes guardados: a identidade da entrada
(`interface/fluxo.chave_da_entrada`, `modo|metodo|digest`) não cobre `max_features`, `rsd_variant`
nem o
corte de RSD, e o atalho entregaria o arquivo da configuração anterior a quem mudou um deles.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QGridLayout, QLabel, QPushButton, QSizePolicy, QWidget

from refino.desktop.janela.componentes import RotuloQueSome
from refino.interface.fluxo import CHAVES_DE_ARTEFATO, DENTRO_DO_FLUXO, PassoDoFluxo
from refino.io.writers import FeatureRemovida, write_removidas

if TYPE_CHECKING:  # pragma: no cover - só para o mypy
    from pathlib import Path
    from typing import Protocol

    class EscolhedorDePasta(Protocol):
        """Onde o lote escreve os três arquivos: um diretório, ou `None` se o usuário cancelou."""

        def __call__(self) -> Path | None: ...


__all__ = [
    "ALTERNATIVA_SEM_NEGATIVOS",
    "ARTEFATOS",
    "ARTEFATOS_DA_JANELA",
    "ARTEFATO_DO_RELATORIO",
    "CAMPOS_POR_ARTEFATO",
    "CHAVES_DO_RELATORIO",
    "CHAVES_DO_RESULTADO",
    "CHAVES_NA_ORDEM",
    "CHAVE_DA_ALTERNATIVA",
    "EM_CURSO",
    "FORMATOS",
    "FORMATO_LABORATORIO",
    "FORMATO_METABOANALYST",
    "MARCA_DE_GERADO",
    "NOMES_DOS_FORMATOS",
    "PASSOS_DOS_ARTEFATOS",
    "SEM_ARTEFATO",
    "SEM_BYTES",
    "ArtefatoDaJanela",
    "PainelDeArtefatos",
    "campos_do_download",
    "escreve_o_corte_ao_lado",
    "nome_do_arquivo",
    "nome_do_arquivo_irmao",
    "pasta_por_dialogo",
    "relatorio_da_resposta",
    "removidas_do_relatorio",
    "texto_do_corte",
    "texto_do_decorrido",
    "texto_do_tamanho",
]

# ---------------------------------------------------------------------------
# O que cada rota de artefato aceita, do que a janela sabe mandar
# ---------------------------------------------------------------------------
#
# Os campos que cada rota declara na OpenAPI, entre os que a janela controla. Campo não
# declarado o FastAPI descarta em silêncio, e o pedido passaria a afirmar uma influência que a
# rota não tem; por isso os três conjuntos diferem:
#
# - `/matrizes/corrigir`: o corretor, os lotes e `auto`, que só esta rota aceita (`processar`
#   fixa `correction="none"` e `exportar` serializa);
# - `/matrizes/processar`: o corretor, os lotes, o corte de RSD e os eixos do pré-processamento;
# - `/matrizes/exportar`: `max_features`, `rsd_variant`, o corretor e os lotes, porque a rota
#   corrige antes de serializar. `max_features` é o limite de 5.000 features do MetaboAnalyst,
#   e fixá-lo reproduziria por omissão o corte cego que a ferramenta substitui.
#
# `paralelo`, `batch_cuts` e `seed` vão nas três, porque as três correm um corretor por lote:
# sem `batch_cuts` o arquivo seria corrigido em blocos iguais enquanto a execução usa as
# fronteiras do diagnóstico, e sem `seed` o resumo metodológico não poderia dizer a semente do
# pedido. `janela/pipeline.py` reexporta `CAMPOS_POR_ARTEFATO`, que sai daqui.
_CAMPOS_DO_CORRIGIR: Final[tuple[str, ...]] = (
    "method",
    "n_batches",
    "batch_cuts",
    "auto",
    "paralelo",
    "seed",
)
_CAMPOS_DO_PROCESSAR: Final[tuple[str, ...]] = (
    "method",
    "n_batches",
    "rsd_cutoff",
    # A variante acompanha o corte: o pipeline corta com
    # `filter_by_qc_rsd(atual, spec.rsd_cutoff, variant=spec.rsd_variant)`, e sem esta chave o
    # download cortaria sempre por `outlier_removed` enquanto o combo mostra outra coisa. No
    # `pos_hilic`, `plain` e `outlier_removed` deixam 102 e 234 features sem RSD mensurável.
    "rsd_variant",
    # O filtro por posto, em dois passes, e a referência da normalização. Se faltassem aqui, a
    # tela não os mandaria, a rota usaria o padrão, e a comparação do par de concordância, que é
    # por texto, diria 'discordando' sobre specs iguais.
    "ranking_filter",
    "ranking_keep_fraction",
    "ranking_filter_2",
    "ranking_keep_fraction_2",
    "normalization_reference",
    # Os fatores viajam como arquivo e não como campo, e quem os envia é o painel de arquivos;
    # o eixo entra aqui mesmo assim, porque esta tupla é o contrato da janela contra a rota.
    "normalization_factors",
    "missing_threshold",
    "imputation",
    "sample_normalization",
    "transformation",
    "scaling",
    "paralelo",
    "batch_cuts",
    "seed",
)
_CAMPOS_DO_EXPORTAR: Final[tuple[str, ...]] = (
    "max_features",
    "rsd_variant",
    "method",
    "n_batches",
    "batch_cuts",
    "paralelo",
    "seed",
)
# Os eixos do pré-processamento que `/matrizes/exportar` aceita, com padrão neutro na rota. Só
# o download da Normalização os manda: o da Correção sai sem eles, byte a byte igual ao da CLI.
# Os fatores entram pela razão dada no `processar`.
_EIXOS_DO_PRE_PROCESSAMENTO: Final[tuple[str, ...]] = (
    "rsd_cutoff",
    "missing_threshold",
    "ranking_filter",
    "ranking_keep_fraction",
    "ranking_filter_2",
    "ranking_keep_fraction_2",
    "imputation",
    "sample_normalization",
    "normalization_reference",
    "normalization_factors",
    "transformation",
    "scaling",
)

# ---------------------------------------------------------------------------
# O mapa das chaves de artefato da janela
# ---------------------------------------------------------------------------
#
# Cada chave é um download: a rota que o serve, os campos que ele leva e a aba dona dele. Todas
# servem ao Resultado da Correção ou ao da Normalização, e cada uma reaproveita o passo de
# `interface/fluxo.py` de que deriva, com a rota e o método dele.
#
# Chaves próprias por aba, e não os três passos reaproveitados, porque `/matrizes/exportar`
# serve às duas abas: com uma chave só, a espera, a guarda da resposta superada e o arquivo
# escrito de uma aba seriam os da outra. A chave é também o nome do arquivo (`nome_do_arquivo`),
# e por isso diz o que ele é, 'corrigida', 'normalizada', e não a rota. Deste mapa saem
# `PASSOS_DOS_ARTEFATOS`, `CAMPOS_POR_ARTEFATO` e, em `principal.py`, as chaves que corrigem e
# as que têm guarda de resposta superada: uma chave nova entra nas duas travas sozinha.

FORMATOS: Final[tuple[str, ...]] = ("laboratorio", "metaboanalyst")
# O formato das matrizes do laboratório: features nas linhas, como a exportação do Progenesis.
FORMATO_LABORATORIO: Final = FORMATOS[0]
# O formato que só aceita positivos, e por isso o único em que há recusa e alternativa.
FORMATO_METABOANALYST: Final = FORMATOS[1]
# O nome de cada formato na tela. O do laboratório diz a orientação, porque é ela que o separa do
# outro aos olhos de quem abre o arquivo: features nas linhas, como a exportação do Progenesis.
NOMES_DOS_FORMATOS: Final[dict[str, str]] = {
    FORMATO_LABORATORIO: "Formato do laboratório (features nas linhas)",
    FORMATO_METABOANALYST: "Formato MetaboAnalyst",
}
# A alternativa à recusa do formato MetaboAnalyst: o mesmo pedido, com os dois eixos que geram
# negativo em `none`. O MetaboAnalyst aplica os dois por conta própria
# (`docs/metodos-metaboanalyst.md`), e a alternativa não precisa de rota nova.
ALTERNATIVA_SEM_NEGATIVOS: Final[dict[str, str]] = {"transformation": "none", "scaling": "none"}


@dataclass(frozen=True)
class ArtefatoDaJanela:
    """Uma chave de artefato da janela: o passo que a serve, os campos dela e a aba dona.

    `passo` é o de `interface/fluxo.py` com a chave trocada pela desta entrada; a rota e o método
    são
    os dele. `base` é a chave do passo de origem: é por ela que o resumo sabe que rota produziu
    o arquivo, e que a faixa dos trabalhos o nomeia. `etapa` é a etapa cujo Resultado mostra o
    download, e `descricao` é o que o arquivo é, para o resumo.
    """

    passo: PassoDoFluxo
    base: str
    campos: tuple[str, ...]
    etapa: int
    formato: str
    descricao: str
    alternativa: bool = False

    @property
    def chave(self) -> str:
        """A chave de espera, de roteamento e do nome do arquivo."""
        return self.passo.chave


_PASSOS_DE_ARTEFATO: Final[dict[str, PassoDoFluxo]] = {
    passo.chave: passo for passo in DENTRO_DO_FLUXO if passo.chave in CHAVES_DE_ARTEFATO
}


def _do_resultado(
    chave: str,
    base: str,
    campos: tuple[str, ...],
    *,
    etapa: int,
    formato: str,
    descricao: str,
    alternativa: bool = False,
) -> ArtefatoDaJanela:
    # O passo de origem com a chave desta entrada, campo a campo, e não por `dataclasses.replace`:
    # o portão de escrita em disco deste módulo lê `replace` como `os.replace` (um nome só, sem o
    # módulo), e o que este passo faz é copiar uma estrutura, e não mover um arquivo.
    origem = _PASSOS_DE_ARTEFATO[base]
    return ArtefatoDaJanela(
        PassoDoFluxo(chave, origem.titulo, origem.rota, origem.metodo, origem.rotulo_de_acao),
        base,
        campos,
        etapa=etapa,
        formato=formato,
        alternativa=alternativa,
        descricao=descricao,
    )


ARTEFATOS_DA_JANELA: Final[dict[str, ArtefatoDaJanela]] = {
    artefato.chave: artefato
    for artefato in (
        # O download da Correção não delega: ele leva o corretor efetivo da correção feita, e
        # mandar `auto` de novo mediria o Palpite outra vez, e o arquivo poderia sair de outro
        # corretor sem a tela saber. Por isso a tupla é a do `corrigir` sem `auto`.
        _do_resultado(
            "corrigida",
            "corrigir",
            tuple(campo for campo in _CAMPOS_DO_CORRIGIR if campo != "auto"),
            etapa=1,
            formato=FORMATO_LABORATORIO,
            descricao=(
                "matriz da etapa de correção, no formato do laboratório (features nas linhas)"
            ),
        ),
        _do_resultado(
            "corrigida_metaboanalyst",
            "exportar",
            _CAMPOS_DO_EXPORTAR,
            etapa=1,
            formato=FORMATO_METABOANALYST,
            descricao="matriz da etapa de correção, no formato do MetaboAnalyst",
        ),
        _do_resultado(
            "normalizada",
            "processar",
            _CAMPOS_DO_PROCESSAR,
            etapa=2,
            formato=FORMATO_LABORATORIO,
            descricao=(
                "matriz da etapa de normalização, no formato do laboratório (features nas linhas)"
            ),
        ),
        _do_resultado(
            "normalizada_metaboanalyst",
            "exportar",
            _CAMPOS_DO_EXPORTAR + _EIXOS_DO_PRE_PROCESSAMENTO,
            etapa=2,
            formato=FORMATO_METABOANALYST,
            descricao="matriz da etapa de normalização, no formato do MetaboAnalyst",
        ),
        # A alternativa à recusa diz o que é no nome do arquivo, e não só no resumo: o arquivo
        # viaja sozinho para o MetaboAnalyst, e quem o abrir depois tem de saber que a
        # transformação e o escalonamento não estão nele.
        _do_resultado(
            "normalizada_metaboanalyst_antes_da_transformacao_e_do_escalonamento",
            "exportar",
            _CAMPOS_DO_EXPORTAR + _EIXOS_DO_PRE_PROCESSAMENTO,
            etapa=2,
            formato=FORMATO_METABOANALYST,
            alternativa=True,
            descricao=(
                "matriz da etapa de normalização no formato do MetaboAnalyst, a ALTERNATIVA à "
                "recusa: anterior à transformação e ao escalonamento, que o MetaboAnalyst aplica "
                "por conta própria depois do upload"
            ),
        ),
    )
}

# Os três passos de `fluxo.py`, na ordem de lá: a grade deste painel montado sozinho. O
# Resultado de cada etapa dispõe as chaves dele por conta própria (`grade=False`).
ARTEFATOS: Final[tuple[PassoDoFluxo, ...]] = tuple(_PASSOS_DE_ARTEFATO.values())
# Os passos dos downloads da janela: é por eles que ela roteia a resposta e acha a rota de um
# disparo.
PASSOS_DOS_ARTEFATOS: Final[tuple[PassoDoFluxo, ...]] = tuple(
    artefato.passo for artefato in ARTEFATOS_DA_JANELA.values()
)
CAMPOS_POR_ARTEFATO: Final[dict[str, tuple[str, ...]]] = {
    chave: artefato.campos for chave, artefato in ARTEFATOS_DA_JANELA.items()
}
# A chave de cada formato no Resultado de cada etapa, e a da alternativa onde ela existe.
CHAVES_DO_RESULTADO: Final[dict[int, dict[str, str]]] = {
    etapa: {
        artefato.formato: artefato.chave
        for artefato in ARTEFATOS_DA_JANELA.values()
        if artefato.etapa == etapa and not artefato.alternativa
    }
    for etapa in sorted({a.etapa for a in ARTEFATOS_DA_JANELA.values()})
}
CHAVE_DA_ALTERNATIVA: Final[dict[int, str]] = {
    artefato.etapa: artefato.chave
    for artefato in ARTEFATOS_DA_JANELA.values()
    if artefato.alternativa
}

# A ordem das chaves da grade deste painel montado sozinho. Cada painel tem a sua
# (`PainelDeArtefatos._chaves`); esta é a ordem pela qual o teste da bicondicional lê os rótulos.
CHAVES_NA_ORDEM: Final[tuple[str, ...]] = tuple(passo.chave for passo in ARTEFATOS)

# O código que significa 'isto não é o resultado', com o mesmo nome que `pipeline.py` usa: um
# `== 202` solto não diz de onde vem o 202.
EM_CURSO: Final = 202

SEM_ARTEFATO: Final = "Nada gerado ainda"

# A marca de que há artefato no rótulo. É constante para poder ser lida: com um botão só, o
# texto do rótulo é o segundo observável da bicondicional, ao lado de `salvaveis()`, e um teste
# que redigitasse a frase divergiria dela no dia em que uma mudasse. São duas superfícies porque
# um `recebe` que guardasse bytes para os três na resposta de um passaria em todo teste que
# olhasse só `salvaveis()`, que lê o mesmo dicionário que o defeito corrompe.
MARCA_DE_GERADO: Final = "gerados"

# A chave do pedido do relatório do corte, que vem depois dos bytes: uma por artefato que corta.
# Ela não é a do artefato: se fosse, a resposta do relatório entraria no ramo dos artefatos de
# `_encaminha` e o painel a leria como bytes, um `200` com JSON onde ele espera CSV. Uma por
# artefato porque o corte é do escritor do MetaboAnalyst, e a Correção e a Normalização têm cada
# uma o seu download por ele: com uma chave só, o relatório do download de uma aba iria parar no
# rótulo e no arquivo irmão da outra.
CHAVES_DO_RELATORIO: Final[dict[str, str]] = {
    chave: f"{chave}_relatorio"
    for chave, artefato in ARTEFATOS_DA_JANELA.items()
    if artefato.base == "exportar"
}
ARTEFATO_DO_RELATORIO: Final[dict[str, str]] = {
    relatorio: chave for chave, relatorio in CHAVES_DO_RELATORIO.items()
}

# A frase do `200` sem bytes, que não é a mensagem de `cliente.py`: para qualquer código de
# sucesso `cliente.mensagem_de` devolve 'OK', e escrever 'OK' ao lado de um controle de salvar
# ausente seria uma palavra de sucesso descrevendo a ausência do que se pediu.
SEM_BYTES: Final = "A rota respondeu e não produziu artefato"


def nome_do_arquivo(modo: str, chave: str, metodo: str | None = None) -> str:
    """O nome sugerido do arquivo: `f"{modo}_{chave}.csv"`, ou com o corretor no fim.

    Uma função e não uma f-string no ponto de uso, para que todo lugar que sugere o nome sugira o
    mesmo: dois literais divergem sem nada acusar.

    `metodo` é opcional, e só o lote o passa. Sem ele, os três arquivos do lote, escritos na
    mesma pasta, deixariam um arquivo só, e o profissional levaria o do último corretor
    acreditando que tem os três.
    """
    if metodo is None:
        return f"{modo}_{chave}.csv"
    return f"{modo}_{chave}_{metodo}.csv"


SEM_DESTINO: Final = "Cancelado: nenhuma pasta escolhida"
PREFIXO_DE_ACAO: Final = "Gerar "


def rotulo_unificado(passo: PassoDoFluxo) -> str:
    """O rótulo do botão único da janela, derivado de `rotulo_de_acao` do passo.

    O botão calcula e entrega os bytes num gesto só, e o rótulo promete o gesto inteiro: 'Gerar
    e baixar ...'. É derivado, e não uma segunda tabela, porque `rotulo_de_acao` vive em
    `fluxo.py`, e uma tabela nova com os mesmos substantivos divergiria sem nada acusar no dia em
    que um dos dois lados mudasse.

    O ramo de saída existe porque `rotulo_de_acao` tem padrão `""` em `PassoDoFluxo`: um passo
    sem o prefixo produziria 'Gerar e baixar Baixar a matriz corrigida'. Os passos de hoje não o
    alcançam, e ele é testado em vez de levar `pragma`.
    """
    if not passo.rotulo_de_acao.startswith(PREFIXO_DE_ACAO):
        return passo.rotulo_de_acao or passo.titulo
    return f"Gerar e baixar {passo.rotulo_de_acao.removeprefix(PREFIXO_DE_ACAO)}"


def campos_do_download(
    artefato: ArtefatoDaJanela, campos: Mapping[str, Any], *, metodo: str, max_features: int
) -> dict[str, Any]:
    """Os campos de um download do Resultado, tirados da etapa feita, e não dos controles.

    `campos` são os que a etapa feita mandou (`EtapaFeita.campos`), no vocabulário das etapas, e
    `metodo` é o corretor que produziu a matriz dela: o efetivo da Correção, ou o da entrada da
    Normalização (`none` na original). O arquivo baixado é o que a etapa mostrou: os controles
    podem ter mudado desde então, e lê-los no clique entregaria a matriz de outros parâmetros com
    o nome desta.

    `max_features` é a exceção, pela natureza dele: é o teto de features do formato
    MetaboAnalyst, e não um parâmetro de etapa nenhuma. Ele vem do controle, no clique.

    Só o que a rota do artefato aceita, e nunca `None`: campo não declarado o FastAPI descarta em
    silêncio, e o vazio de um eixo opcional é a falta dele, que é como a rota lê 'sem filtro'. Na
    alternativa à recusa, a transformação e o escalonamento vão em `none`.
    """
    disponiveis: dict[str, Any] = {**campos, "method": metodo, "max_features": max_features}
    if artefato.alternativa:
        disponiveis.update(ALTERNATIVA_SEM_NEGATIVOS)
    return {
        nome: disponiveis[nome] for nome in artefato.campos if disponiveis.get(nome) is not None
    }


def nome_do_arquivo_irmao(escolhido: Path) -> Path:
    """O caminho do registro do corte, ao lado do CSV que o usuário acabou de salvar.

    Derivado do caminho escolhido, e não do nome sugerido: quem salvar `pos_hilic_exportar.csv`
    como `experimento_marco.csv` recebe `experimento_marco-removidas.csv`. Derivar do sugerido
    poria o registro ao lado de um arquivo com outro nome, ou noutra pasta.
    """
    return escolhido.with_name(f"{escolhido.stem}-removidas{escolhido.suffix}")


def removidas_do_relatorio(corte: dict[str, Any]) -> tuple[FeatureRemovida, ...]:
    """O corpo da rota de volta às estruturas do núcleo, para o escritor de lá.

    A janela não formata o arquivo: reconstrói `FeatureRemovida` e chama `write_removidas`, o
    mesmo escritor do núcleo. Uma formatação daqui divergiria da dele no dia em que a coluna
    mudasse, sem nada acusar.

    A ordem vem da lista e não é reordenada aqui: indefinidas primeiro, e as de RSD alto da pior
    para a melhor, como decidido em `MetaboAnalystExport`.
    """
    return tuple(
        FeatureRemovida(nome=item["nome"], causa=item["causa"], rsd=item.get("rsd"))
        for item in corte.get("removidas", ())
    )


def relatorio_da_resposta(codigo: int, corpo: object) -> dict[str, Any] | None:
    """O relatório do corte que a resposta traz, ou `None` quando ela não traz nenhum.

    Uma regra para 'isto é um relatório', e dois leitores: `PainelDeArtefatos.recebe_relatorio`
    e o ramo do lote em `JanelaPrincipal._encaminha`, que não passa pela guarda de bytes do
    painel. `409` é a operação que não corta, ou o trabalho podado; um corpo que não é dicionário
    não tem lista de removidas a escrever.
    """
    if codigo != 200 or not isinstance(corpo, dict):
        return None
    return corpo


def escreve_o_corte_ao_lado(relatorio: dict[str, Any], escolhido: Path) -> str | None:
    """Escreve o registro do corte ao lado de `escolhido`. Devolve a frase da falha, ou `None`.

    Uma escrita só, e dois caminhos a chamam. O clique avulso chega aqui por
    `PainelDeArtefatos.registra_corte_ao_lado`, com o relatório que o painel guardou; o lote
    'Gerar os três' chega direto, com o corpo da resposta, porque quando o relatório de uma perna
    volta o painel já descartou os bytes dela (a perna seguinte redefiniu a entrada) e a guarda
    dele o recusa.

    A falha devolve a frase e não levanta: o CSV já está em disco quando isto corre, e quem chama
    a põe no rótulo do Resultado. Reportar falha sobre um arquivo que existe faria a pessoa
    procurar um CSV que está lá.
    """
    try:
        write_removidas(removidas_do_relatorio(relatorio), nome_do_arquivo_irmao(escolhido))
    except OSError as erro:
        return f"O arquivo foi salvo; o registro do corte NÃO: {type(erro).__name__}"
    return None


def texto_do_corte(corte: dict[str, Any]) -> str:
    """O resumo do corte para o rótulo do painel.

    Contagem por motivo, e não só o total, como `refino exportar` imprime: ausência de medida e
    medida ruim levam a ações diferentes, olhar o dado ou corrigir melhor.

    Sem remoção nenhuma, escreve 'nenhuma removida' e não omite a linha. Silêncio ali seria
    indistinguível de 'o painel não sabe', e o profissional precisa ler que o corte correu e não
    tirou nada, que é o caso comum quando a matriz não chega a 5.000 features.
    """
    removidas = corte.get("removidas", ())
    mantidas = corte.get("mantidas", 0)
    if not removidas:
        return f"corte: {mantidas} mantidas, nenhuma removida"

    causas = [item.get("causa") for item in removidas]
    linhas = [f"corte: {mantidas} mantidas, {len(removidas)} removidas"]
    for causa, frase in (
        ("qcs_todos_zero", "QCs todos zero"),
        ("outliers_apagaram_qcs", "a remoção de outliers apagou os QCs"),
        ("rsd_acima_do_corte", "RSD acima do corte"),
    ):
        quantas = causas.count(causa)
        if quantas:
            linhas.append(f"    {frase}: {quantas}")
    return "\n".join(linhas)


def texto_do_tamanho(conteudo: bytes) -> str:
    """O tamanho do artefato, em KB ou em bytes, e nunca `0 KB` para um arquivo que existe.

    `len(conteudo) // 1024` dá zero para qualquer artefato abaixo de 1 KiB, e o rótulo diria '0
    KB gerados' sobre um arquivo que a janela acabou de escrever em disco.
    """
    return f"{len(conteudo) // 1024} KB" if len(conteudo) >= 1024 else f"{len(conteudo)} bytes"


def texto_do_decorrido(segundos: float) -> str:
    """O tempo de espera, na unidade que uma pessoa lê sem fazer conta.

    Abaixo de um minuto, `:.3g` segundos: `:.0f` escreveria '0 s' para uma espera de 50 ms, e
    `:g` daria `0.157123`, precisão inventada. Acima, minutos e segundos, porque com esperas de
    até uma hora `:.3g` escreveria `3.6e+03 s`.

    O `divmod` é sobre o valor arredondado, e não `%` sobre o `float`: `119.7 % 60` dá `59.7`,
    que `:.0f` escreve como 60, e o rótulo diria '1 min 60 s'. Arredondar primeiro dá
    `2 min 0 s`.
    """
    if segundos < 60:
        return f"{segundos:.3g} s"
    minutos, resto = divmod(round(segundos), 60)
    return f"{minutos} min {resto} s"


def pasta_por_dialogo(pai: QWidget | None = None) -> EscolhedorDePasta:
    """A pasta de destino do lote: `QFileDialog.getExistingDirectory`, uma vez só.

    Uma pergunta antes dos três, e não três perguntas: o lote corre `none`, `loess` e `serrf` em
    sequência, em minutos, e um diálogo por arquivo faria o profissional voltar ao computador
    três vezes durante a espera.

    String vazia é o cancelamento, traduzida para `None`: `Path("")` é o diretório corrente, e
    escrever três CSVs num caminho que ninguém escolheu é pior que não escrever nenhum.

    Fábrica e não classe, e testável pelo mesmo molde: o arnês remenda
    `QFileDialog.getExistingDirectory` e afirma as duas direções, a pasta escolhida e o
    cancelamento.
    """
    from pathlib import Path as _Path

    from PySide6.QtWidgets import QFileDialog

    def escolhe() -> _Path | None:
        escolhido = QFileDialog.getExistingDirectory(pai, "Pasta para os três arquivos")
        return _Path(escolhido) if escolhido else None

    return escolhe


class PainelDeArtefatos(QWidget):
    """Os downloads, com o controle de salvar existindo se e somente se houve bytes.

    `gerar` é o sinal que a janela liga à piscina (`JanelaPrincipal._gera_artefato`). Ele carrega
    a chave do artefato, e é a chave que permite ao painel filtrar a resposta que é dele.
    """

    gerar = Signal(str)

    def __init__(
        self,
        parent: QWidget | None = None,
        escolhedor: EscolhedorDePasta | None = None,
        *,
        artefatos: Sequence[PassoDoFluxo] = ARTEFATOS,
        grade: bool = True,
    ) -> None:
        """`artefatos` são as chaves que o painel serve: os três passos de `fluxo.py`, por padrão,
        ou os do Resultado de uma etapa, que é a forma que a janela compõe.

        Sem `grade`, o painel não dispõe nada: quem herda dispõe os botões e os rótulos e os põe
        em `botoes_de_geracao` e `rotulos`, um por chave de `artefatos`. É o Resultado das etapas,
        que serve duas ou três chaves com um botão de download só. O estado dos bytes, a
        identidade da entrada e o relatório do corte são estes, e não uma cópia deles.
        """
        super().__init__(parent)
        # As chaves deste painel, na ordem de `artefatos`, para `salvaveis` ser determinística:
        # iterar `self._bytes` daria a ordem de inserção, e o teste teria de ordenar, o que
        # esconderia uma implementação que devolvesse as chaves erradas na ordem certa.
        self._chaves: tuple[str, ...] = tuple(passo.chave for passo in artefatos)
        # O escolhedor de pasta do botão único, injetado pela razão declarada em `arquivos.py`: o
        # diálogo real abre uma janela, e um arnês que não o remende pendura.
        self._escolhedor: EscolhedorDePasta = escolhedor or pasta_por_dialogo(self)
        # O destino escolhido no clique, à espera de que `principal` o recolha. Vive aqui porque
        # quem o pede é o botão: o sinal `gerar` continua a significar só 'o profissional pediu
        # para gerar', e um arnês que emita o sinal direto não tem destino nem escrita automática.
        self.destinos: dict[str, Path] = {}
        self._bytes: dict[str, bytes] = {}
        # `None` e não `""`: `""` cobriria 'nunca definido' e 'definido como vazio' (o combo de
        # modo é editável, e o usuário pode limpá-lo), e a igualdade em `define_entrada` sairia
        # antes de descartar.
        self._identidade: str | None = None
        self._modo = ""
        # O relatório do corte, por chave. Só `exportar` o tem: as outras rotas devolvem a matriz
        # inteira. A chave fica ausente quando não há relatório, e não com um dicionário vazio:
        # 'esta operação não corta' e 'cortou e não removeu nada' pedem frases opostas no rótulo.
        self._relatorios: dict[str, dict[str, Any]] = {}

        self.botoes_de_geracao: dict[str, QPushButton] = {}
        self.rotulos: dict[str, QLabel] = {}
        if grade:
            self._dispoe_a_grade(artefatos)

    def _dispoe_a_grade(self, artefatos: Sequence[PassoDoFluxo]) -> None:
        """Um botão e um rótulo de estado por artefato, numa grade: o painel montado sozinho."""
        # Grade e não pilha: cada artefato ocupa uma linha, com o botão à esquerda, na largura que
        # o texto dele pede, e o estado à direita. Empilhados num `QVBoxLayout`, os botões se
        # esticariam pela largura toda, com um vazio grande abaixo deles.
        #
        # A folga é da coluna do estado e da linha vazia do fim, declaradas: numa grade não existe
        # `addStretch`, e quem recebe a folga é a linha ou a coluna nomeada.
        disposicao = QGridLayout(self)
        disposicao.setColumnStretch(0, 0)
        disposicao.setColumnStretch(1, 1)
        # Sem margem própria: o painel mora no corpo de uma `Secao`, que já tem a dela, e as duas
        # somadas empurrariam a grade para dentro duas vezes.
        disposicao.setContentsMargins(0, 0, 0, 0)
        disposicao.setHorizontalSpacing(16)
        disposicao.setVerticalSpacing(8)

        for linha, passo in enumerate(artefatos):
            gerar = QPushButton(rotulo_unificado(passo), self)
            gerar.clicked.connect(lambda _=False, chave=passo.chave: self.pede_e_gera(chave))
            # `Maximum` e não o padrão `Preferred`: o botão fica na largura do texto e não
            # cresce com a coluna. É o que separa três botões legíveis de três faixas.
            gerar.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
            self.botoes_de_geracao[passo.chave] = gerar
            disposicao.addWidget(gerar, linha, 0)

            # Em repouso some: 'Nada gerado ainda' três vezes, ao lado de três botões que já dizem
            # o que há para gerar, é ruído. O estado aparece quando o arquivo é escrito.
            rotulo = RotuloQueSome(SEM_ARTEFATO, self, vazios=frozenset({SEM_ARTEFATO}))
            rotulo.setWordWrap(True)
            self.rotulos[passo.chave] = rotulo
            disposicao.addWidget(rotulo, linha, 1)

        # A linha vazia do fim é a dona da folga vertical: sem ela a grade repartiria a altura
        # entre as linhas, e os rótulos flutuariam.
        disposicao.setRowStretch(len(artefatos), 1)

    # ------------------------------------------------------------------
    # A bicondicional
    # ------------------------------------------------------------------

    def pede_e_gera(self, chave: str) -> None:
        """Pergunta onde antes de pedir para gerar, e não emite se o profissional desistir.

        O diálogo acontece no instante do clique, que é o único em que o profissional está à
        frente da máquina: com `serrf` a correção leva minutos, e um diálogo abrindo sozinho no
        fim roubaria o foco de onde quer que ele estivesse. Cancelar não emite: quem desiste do
        diálogo pediu justamente para não correr a correção.
        """
        destino = self._escolhedor()
        if destino is None:
            self.rotulos[chave].setText(SEM_DESTINO)
            return
        self.destinos[chave] = destino
        self.gerar.emit(chave)

    def salvaveis(self) -> tuple[str, ...]:
        """As chaves que têm bytes agora, na ordem das chaves deste painel.

        É uma das duas superfícies da bicondicional; a outra é o texto do rótulo
        (`MARCA_DE_GERADO`). Um `recebe` que marcasse os três na resposta de um passaria em todo
        teste que olhasse só este conjunto.
        """
        return tuple(chave for chave in self._chaves if self._bytes.get(chave))

    def define_entrada(self, identidade: str, modo: str) -> None:
        """A identidade da entrada corrente. Descarta os artefatos se ela mudar.

        `identidade` é `interface/fluxo.chave_da_entrada`, `modo|metodo|digest`, e `modo` entra só
        no
        nome sugerido: o nome tem de ser legível, e a identidade tem de ser completa. Com o modo
        sozinho, trocar o par de arquivos ou o corretor no mesmo modo não descartaria nada, e o
        botão escreveria os bytes da entrada anterior sob o nome da seleção corrente.
        """
        if identidade == self._identidade:
            return
        self._identidade = identidade
        self._modo = modo
        self._bytes.clear()
        # O relatório do corte segue o ciclo de vida dos bytes: ele descreve aquele arquivo, e
        # deixá-lo aqui faria a próxima exportação escrever o irmão do corte anterior ao lado do
        # CSV novo.
        self._relatorios.clear()
        for chave in self._chaves:
            self.rotulos[chave].setText(SEM_ARTEFATO)

    def recebe(self, chave: str, codigo: int, conteudo: bytes | None, mensagem: str) -> str:
        """O desfecho de uma reclamação de artefato. Devolve o texto escrito no rótulo.

        A terminalidade é o código, e os bytes são `conteudo`, que `cliente.envia` já leu pelo
        `content-type`. Quatro desfechos:

        - chave que não é de artefato: nada, e devolve vazio;
        - `202`: ainda correndo; o rótulo diz isso e não há bytes;
        - `200` com `conteudo`: os bytes ficam, e o controle de salvar passa a existir;
        - erro, ou `200` sem bytes: a mensagem no rótulo, e o controle não existe.

        A guarda da chave vem primeiro: `JanelaPrincipal.resposta_recebida` é emitido para toda
        chamada (validação, diagnóstico, execução, progresso), e sem a guarda
        `self.rotulos["validacao"]` levantaria `KeyError` depois de `_bytes.pop`; num slot de Qt a
        exceção é engolida, e o painel pararia de atualizar.

        A limpeza alcança só a chave que é dele: um `clear()` revogaria o download de `corrigir`
        quando `processar` respondesse.
        """
        if chave not in self.rotulos:
            return ""

        self._bytes.pop(chave, None)
        # O relatório do corte descreve os bytes que acabaram de ser descartados. Mantê-lo faria a
        # próxima escrita do irmão registrar uma exportação anterior.
        self._relatorios.pop(chave, None)

        if codigo == EM_CURSO:
            texto = "a geração foi aceita e corre no servidor"
        elif conteudo:
            self._bytes[chave] = conteudo
            texto = f"{texto_do_tamanho(conteudo)} {MARCA_DE_GERADO}"
        elif codigo == 200:
            # Nem a mensagem de `cliente.py`, que para um `200` é 'OK': ver `SEM_BYTES`.
            texto = SEM_BYTES
        else:
            # A mensagem vem de `cliente.py`, e não nasce aqui.
            texto = mensagem
        self.rotulos[chave].setText(texto)
        return texto

    def recebe_relatorio(self, chave: str, codigo: int, corpo: object) -> str:
        """O relatório do corte de uma exportação. Devolve o texto escrito no rótulo.

        Chega depois dos bytes e por uma segunda chamada, porque `GET /trabalhos/{id}` serve
        `text/csv` e não há onde pôr a lista de nomes ali. A rota irmã responde `409` para as
        operações que não cortam, e para elas este método não faz nada: o rótulo continua a dizer
        o tamanho do artefato.

        - chave sem bytes, ou que não é de artefato: nada, e devolve vazio;
        - código que não é `200`: nada, porque a operação não corta ou o trabalho foi podado;
        - `200` com corpo de relatório: guarda, e acrescenta o resumo ao rótulo.

        A guarda dos bytes impede o rótulo de mentir: um relatório que chegasse depois de
        `recebe` ter descartado os bytes, resposta atrasada de um trabalho superado, escreveria as
        contagens de um corte sobre um rótulo que já diz que não há artefato.

        O rótulo acrescenta e não substitui: '2312 KB gerados' continua a ser a frase que diz que
        há artefato, e o corte é informação a mais sobre ele.
        """
        if chave not in self.rotulos or not self._bytes.get(chave):
            return ""
        relatorio = relatorio_da_resposta(codigo, corpo)
        if relatorio is None:
            return ""

        self._relatorios[chave] = relatorio
        tamanho = texto_do_tamanho(self._bytes[chave])
        texto = f"{tamanho} {MARCA_DE_GERADO}\n{texto_do_corte(relatorio)}"
        self.rotulos[chave].setText(texto)
        return texto

    def mostra_espera(self, chave: str, *, feito: int, total: int, leituras: int) -> str:
        """O rótulo enquanto a janela relê o progresso sozinha. Devolve o texto escrito.

        As frases da espera nascem aqui, e não na janela, porque `recebe`, `SEM_ARTEFATO` e
        `SEM_BYTES` já vivem neste módulo, e uma terceira em `principal.py` faria duas fontes de
        verdade sobre o que este rótulo diz. Não é desfecho de rota, e por isso não usa
        `causa_de` nem `mensagem_de`: o assunto é o laço da janela.

        Não toca os bytes nem o controle de salvar: quem invalida artefato velho é
        `define_entrada`, pela identidade da entrada, e desligar aqui seria uma segunda regra sobre
        quando o download desaparece.

        `total` zero é contagem ausente, e não zero passos: o `202` do `POST` traz `total` e não
        traz `feito`, e escrever '0 de 0' afirmaria uma medida que não se tem.
        """
        if chave not in self.rotulos:
            return ""
        parte = f"{feito} de {total}" if total else "sem contagem de passos"
        texto = f"a correr no servidor: {parte} — leitura {leituras}"
        self.rotulos[chave].setText(texto)
        return texto

    def mostra_desistencia(self, chave: str, *, leituras: int, segundos: float) -> str:
        """O rótulo quando o teto da espera passou. Devolve o texto escrito.

        A frase diz o que fazer: clicar outra vez no botão do artefato reclama o mesmo trabalho
        em vez de recomeçar, porque `POST /matrizes/<passo>` devolve o mesmo identificador para a
        mesma entrada (o campo `reclamado` da resposta). Diz também que o trabalho continua no
        servidor e que a demora é normal com `serrf`, que pode passar de vinte minutos numa matriz
        grande: sem isso, a frase se lê como falha.

        Os dois números vão no texto porque sem eles a frase não é acionável: 'não terminou' não
        distingue dois segundos de cinco minutos. `segundos` é medido pelo chamador, e não
        derivado do intervalo, porque o intervalo ignora a latência das leituras.
        """
        if chave not in self.rotulos:
            return ""
        # O nome do botão é lido dele, para a frase citar o texto que o botão de fato mostra, que
        # varia entre os painéis.
        botao = self.botoes_de_geracao[chave].text()
        texto = (
            f"a janela parou de perguntar depois de {leituras} leituras e "
            f"{texto_do_decorrido(segundos)} — mas o trabalho CONTINUA a correr no servidor. "
            f"Esperas longas são normais com o corretor `serrf`. Clique em «{botao}» outra vez "
            "para voltar a acompanhar: isso não recomeça o cálculo, reclama o mesmo trabalho"
        )
        self.rotulos[chave].setText(texto)
        return texto

    def registra_corte_ao_lado(self, chave: str, escolhido: Path) -> Path | None:
        """Escreve o arquivo irmão ao lado do CSV que acabou de ser escrito.

        O registro nasce do mesmo gesto que produz o arquivo cortado, sem segundo botão nem
        diálogo: um registro que a pessoa tem de pedir é um registro que não existe, e o caminho
        já foi escolhido no clique. Quem chama é a janela, no ramo do relatório, com o caminho que
        a escrita guardou (`JanelaPrincipal._escreve_no_destino`). A chamada não pode ser no
        instante da escrita do CSV: o relatório chega numa resposta posterior à dos bytes, e o
        irmão sairia sempre só com o cabeçalho.

        Escreve mesmo quando nada foi removido: o arquivo só com o cabeçalho diz 'o corte correu e
        não tirou nada', e a ausência dele seria ambígua. Uma falha de escrita vai ao rótulo e
        devolve `None`, e a janela continua a declarar o caminho do artefato: o CSV cortado já
        está em disco, e reportar falha sobre ele faria a pessoa procurar um arquivo que está lá.
        """
        relatorio = self._relatorios.get(chave)
        if relatorio is None:
            return None
        # A escrita e o tratamento da falha são os de `escreve_o_corte_ao_lado`, que o lote também
        # usa; aqui fica só o que é do painel: o relatório guardado e o rótulo.
        falha = escreve_o_corte_ao_lado(relatorio, escolhido)
        if falha is not None:
            self.rotulos[chave].setText(falha)
            return None
        return nome_do_arquivo_irmao(escolhido)
