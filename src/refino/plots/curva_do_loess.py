"""Curva do LOESS ao longo das injeções, com o envelope das curvas refeitas sem um QC.

Para uma feature escolhida pela pessoa, a figura mostra os QCs medidos, a curva com que o LOESS
corrige o sinal, a predição de cada QC pela curva refeita sem ele e o envelope dessas curvas.
A curva só é rotulada como a que corrigiu o sinal quando o corretor do pedido é o LOESS; com
outro, ou sem corretor decidido, o rótulo e o rodapé dizem que ela é a que o LOESS usaria.

Não há banda de erro. Com `span` até 0,70 o LOESS de grau 2 quase passa pelos QCs e não sobra
resíduo de onde tirar uma banda; com `span` maior sobra resíduo, mas com poucos graus de
liberdade. Por isso o `span` eleito vai no rodapé. O que se mede em qualquer caso é quanto a
curva muda quando cada QC sai do ajuste: o envelope é o mínimo e o máximo, injeção a injeção,
das curvas sem um QC. O rótulo diz sensibilidade e não intervalo, porque o envelope não tem
probabilidade de cobertura.

O módulo não recalcula nada: recebe os vetores do fio e desenha. A curva sai de
`LoessCorrector.fitted_curve`, o caminho da correção, e as predições sem cada QC saem dos
passos de `leave_one_qc_out_rsd`, com a mesma semente.

Três coisas vão marcadas de outra forma:

- Os trechos que alguma curva sem um QC não ajustou vão hachurados. Sem o primeiro QC a curva
  não alcança o começo da corrida, sem o último não alcança o fim, e o passo 3 do LOESS
  preenche essas posições com a predição do tempo mais próximo; onde a curva sai negativa, o
  passo 4 sorteia. O mínimo e o máximo incluem esses valores, que são os da curva que a régua
  usa, então a largura do envelope ali pode vir do preenchimento ou do sorteio.
- As posições da curva que corrigiu e que não foram ajustadas (preenchidas pelo passo 3 ou
  sorteadas pelo passo 4) levam marcadores sobre a curva.
- O QC aberrante, que o passo 1 deixa fora do ajuste, vai vazado: a curva não passa por ele.
  O QC sem valor finito também fica fora do ajuste, mas não tem ponto a desenhar; ele conta
  no título.

Quando a correção devolve a feature sem correção (um lote com menos de três QCs utilizáveis,
ou mais de um quinto da curva negativa), não há curva e a figura diz por quê. As predições e o
envelope continuam a ser os da régua, quando ela os mede.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final

import numpy as np
from matplotlib.patches import Polygon

from refino.plots.base import (
    MatrizVaziaError,
    RecusaDoConstrutorError,
    cores_do_ciclo,
    inteiro_do_fio,
    nova_figura,
    vetor_do_fio,
)
from refino.plots.recorte import Recorte, aplica, legenda_fora

if TYPE_CHECKING:  # pragma: no cover - só para o mypy
    from collections.abc import Sequence

    from matplotlib.artist import Artist
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

__all__ = [
    "MARCA_SEM_CURVA",
    "MATRIZ",
    "ROTULO_DA_CURVA",
    "ROTULO_DA_CURVA_DE_OUTRO_CORRETOR",
    "ROTULO_DA_POSICAO_PREENCHIDA",
    "ROTULO_DA_POSICAO_SORTEADA",
    "ROTULO_DA_PREDICAO",
    "ROTULO_DE_X",
    "ROTULO_DE_Y",
    "ROTULO_DO_ENVELOPE",
    "ROTULO_DO_ENVELOPE_NAO_AJUSTADO",
    "ROTULO_DO_LIMITE_INFERIOR",
    "ROTULO_DO_LIMITE_SUPERIOR",
    "ROTULO_DO_QC",
    "ROTULO_DO_QC_FORA_DO_AJUSTE",
    "TITULO",
    "CurvaDesalinhadaError",
    "CurvaDoLoess",
    "curva_do_fio",
    "figura_da_curva_do_loess",
    "frase_da_populacao",
    "frase_do_corretor",
    "frases_das_ausencias",
    "metodo_da_curva",
    "recorte_da_curva_do_loess",
    "rotulo_da_curva",
]

TITULO: Final = "Curva do LOESS"

ROTULO_DE_X: Final = "ordem de injeção"

# A curva é o divisor da correção e está na escala da feature medida, antes de corrigir.
ROTULO_DE_Y: Final = "intensidade, antes da correção"

# Os rótulos longos vão em duas linhas. A legenda fica fora do eixo, à direita, e a largura
# dela sai do rótulo mais comprido: numa linha só, o eixo ficaria com menos de metade da
# largura e o título seria cortado à esquerda.
ROTULO_DA_CURVA: Final = "curva que corrigiu o sinal"
# Rótulo da mesma curva quando o corretor do pedido não é o LOESS (SERRF, sem correção ou sem
# corretor decidido). O rodapé nomeia o corretor do pedido.
ROTULO_DA_CURVA_DE_OUTRO_CORRETOR: Final = "curva com que o LOESS\ncorrigiria o sinal"
CORRETOR_DA_CURVA: Final = "loess"
ROTULO_DA_POSICAO_PREENCHIDA: Final = "curva preenchida pelo tempo\nmais próximo (passo 3)"
ROTULO_DA_POSICAO_SORTEADA: Final = "curva sorteada, negativa ali\n(passo 4)"
ROTULO_DO_QC: Final = "QC medido"
# Só 'aberrante': o QC sem valor finito também fica fora do ajuste, mas não é desenhado.
ROTULO_DO_QC_FORA_DO_AJUSTE: Final = "QC fora do ajuste\n(aberrante, passo 1)"
ROTULO_DA_PREDICAO: Final = "predição de cada QC\npela curva que não o viu"

# O rótulo diz o que o envelope é e o que não é, para ele não ser lido como banda de confiança.
ROTULO_DO_ENVELOPE: Final = "envelope das curvas sem um QC\n(sensibilidade, não intervalo)"
ROTULO_DO_ENVELOPE_NAO_AJUSTADO: Final = "trecho em que alguma curva\nsem um QC não foi ajustada"
ROTULO_DO_LIMITE_INFERIOR: Final = "mínimo das curvas sem um QC"
MARCA_SEM_CURVA: Final = "sem curva — o motivo está no rodapé"
ROTULO_DO_LIMITE_SUPERIOR: Final = "máximo das curvas sem um QC"

MATRIZ: Final = "a matriz ingerida, antes da correção"


class CurvaDesalinhadaError(RecusaDoConstrutorError):
    """A carga da curva não descreve as mesmas injeções em todos os vetores.

    Cobre vetores por amostra com comprimentos diferentes, ordens de injeção não finitas ou
    fora de ordem, e envelope com o mínimo acima do máximo. O conserto é o mesmo em todos os
    casos: o corpo foi montado errado.
    """


@dataclass(frozen=True)
class CurvaDoLoess:
    """A carga da figura, convertida do fio. Todos os vetores têm uma posição por amostra."""

    feature: str
    ordens: np.ndarray
    lotes: tuple[str, ...]
    qc: np.ndarray
    valores: np.ndarray
    curva: np.ndarray
    curva_preenchida: np.ndarray
    curva_sorteada: np.ndarray
    qc_no_ajuste: np.ndarray
    motivo_sem_curva: str | None
    predicao_sem_o_qc: np.ndarray
    motivo_sem_predicao: tuple[str | None, ...]
    envelope_inferior: np.ndarray
    envelope_superior: np.ndarray
    envelope_nao_ajustado: np.ndarray
    n_curvas_do_envelope: int
    motivo_sem_envelope: str | None
    semente: int
    spans_por_lote: tuple[float, ...] = ()
    # O corretor do pedido. A rota devolve a curva do LOESS qualquer que seja o corretor, e só
    # quem pede sabe se ela é a que corrige. Com `None`, a figura não diz que a curva corrigiu.
    corretor: str | None = None

    @property
    def tem_curva(self) -> bool:
        """Se a correção dividiu por uma curva. Decide o que a figura desenha."""
        return self.motivo_sem_curva is None and bool(np.isfinite(self.curva).any())


def _booleanos(valores: Any) -> np.ndarray:
    """Um vetor de `bool` do fio; o que não é `True` literal conta como falso."""
    if not isinstance(valores, list):
        return np.empty(0, dtype=bool)
    return np.array([x is True for x in valores], dtype=bool)


def _texto_ou_nada(valor: Any) -> str | None:
    return valor if isinstance(valor, str) and valor.strip() else None


def curva_do_fio(
    *,
    feature: Any,
    ordens_de_injecao: Any,
    lotes: Any,
    qc: Any,
    valores: Any,
    curva: Any,
    curva_preenchida: Any,
    curva_sorteada: Any,
    qc_no_ajuste: Any,
    motivo_sem_curva: Any,
    predicao_sem_o_qc: Any,
    motivo_sem_predicao: Any,
    envelope_inferior: Any,
    envelope_superior: Any,
    envelope_nao_ajustado: Any,
    n_curvas_do_envelope: Any,
    motivo_sem_envelope: Any,
    semente: Any,
    spans_por_lote: Any = None,
    corretor: Any = None,
) -> CurvaDoLoess:
    """Confere a carga do fio na fronteira e a converte para arrays.

    Sem injeção nenhuma, ou sem QC nenhum, levanta `MatrizVaziaError`, que a janela troca por
    texto. Vetores que não descrevem as mesmas injeções levantam `CurvaDesalinhadaError`:
    desenhá-los ligaria o valor de uma injeção à posição de outra.
    """
    ordens = vetor_do_fio(ordens_de_injecao)
    if ordens.size == 0:
        raise MatrizVaziaError(
            "a curva do LOESS não traz ordem de injeção nenhuma: não há corrida a desenhar. Sem a "
            "rota da curva corrida para uma feature, não há população, e uma figura em branco "
            "seria indistinguível de figura quebrada"
        )
    n = ordens.size
    rotulos = tuple(str(r) for r in lotes) if isinstance(lotes, list) else ()
    por_amostra: dict[str, np.ndarray] = {
        "qc": _booleanos(qc),
        "valores": vetor_do_fio(valores),
        "curva": vetor_do_fio(curva),
        "curva_preenchida": _booleanos(curva_preenchida),
        "curva_sorteada": _booleanos(curva_sorteada),
        "qc_no_ajuste": _booleanos(qc_no_ajuste),
        "predicao_sem_o_qc": vetor_do_fio(predicao_sem_o_qc),
        "envelope_inferior": vetor_do_fio(envelope_inferior),
        "envelope_superior": vetor_do_fio(envelope_superior),
        "envelope_nao_ajustado": _booleanos(envelope_nao_ajustado),
    }
    motivos = (
        tuple(_texto_ou_nada(m) for m in motivo_sem_predicao)
        if isinstance(motivo_sem_predicao, list)
        else ()
    )
    desalinhados = sorted(nome for nome, vetor in por_amostra.items() if vetor.size != n)
    if len(rotulos) != n:
        desalinhados.append("lotes")
    if len(motivos) != n:
        desalinhados.append("motivo_sem_predicao")
    if desalinhados:
        raise CurvaDesalinhadaError(
            f"{n} ordem(ns) de injeção e os vetores {desalinhados} com outro comprimento. Todos "
            "descrevem as mesmas injeções, e um vetor desalinhado poria o valor de uma na posição "
            "de outra"
        )
    if not np.isfinite(ordens).all() or bool(np.any(np.diff(ordens) < 0.0)):
        raise CurvaDesalinhadaError(
            "as ordens de injeção não são finitas e não decrescentes: `RefinoMatrix` mantém as "
            "linhas em ordem de injeção, e um vetor que desce é outra coisa com o mesmo nome"
        )
    inferior, superior = por_amostra["envelope_inferior"], por_amostra["envelope_superior"]
    ambos = np.isfinite(inferior) & np.isfinite(superior)
    if bool(np.any(inferior[ambos] > superior[ambos])):
        raise CurvaDesalinhadaError(
            "o envelope tem posição com o mínimo acima do máximo: as duas pontas foram trocadas "
            "ou vêm de populações diferentes"
        )
    if not por_amostra["qc"].any():
        raise MatrizVaziaError(
            "nenhuma injeção da carga é QC: a curva do LOESS ajusta-se aos QCs, e sem eles não há "
            "curva, predição nem envelope a desenhar"
        )
    return CurvaDoLoess(
        feature=_texto_ou_nada(feature) or "feature não declarada",
        ordens=ordens,
        lotes=rotulos,
        qc=por_amostra["qc"],
        valores=por_amostra["valores"],
        curva=por_amostra["curva"],
        curva_preenchida=por_amostra["curva_preenchida"],
        curva_sorteada=por_amostra["curva_sorteada"],
        qc_no_ajuste=por_amostra["qc_no_ajuste"],
        motivo_sem_curva=_texto_ou_nada(motivo_sem_curva),
        predicao_sem_o_qc=por_amostra["predicao_sem_o_qc"],
        motivo_sem_predicao=motivos,
        envelope_inferior=inferior,
        envelope_superior=superior,
        envelope_nao_ajustado=por_amostra["envelope_nao_ajustado"],
        n_curvas_do_envelope=inteiro_do_fio(n_curvas_do_envelope),
        motivo_sem_envelope=_texto_ou_nada(motivo_sem_envelope),
        semente=inteiro_do_fio(semente),
        spans_por_lote=tuple(float(s) for s in vetor_do_fio(spans_por_lote) if np.isfinite(s)),
        corretor=_texto_ou_nada(corretor),
    )


def metodo_da_curva(semente: int, spans: Sequence[float] = ()) -> str:
    """O método do rodapé, com a semente e o `span` eleito, que juntos decidem a curva.

    O `span` vai por lote, na ordem dos lotes, e diz se a curva passa pelos QCs: até 0,70 quase
    passa, acima disso sobra resíduo. Sem curva não há `span` eleito, e a frase diz isso.
    """
    eleitos = (
        "nenhum span eleito, porque não houve curva"
        if not spans
        else "span eleito " + " / ".join(f"{s:.2f}".replace(".", ",") for s in spans)
    )
    return (
        f"LOESS de grau 2 por lote, com o span escolhido por validação cruzada com a semente "
        f"{semente} ({eleitos}); cada curva sem um QC é refeita pelos passos de "
        "leave_one_qc_out_rsd, com a "
        "mesma semente. O envelope é o mínimo e o máximo dessas curvas em cada injeção: "
        "sensibilidade à remoção de um QC, e não intervalo de confiança. Hachurado: injeções em "
        "que alguma curva sem um QC não foi ajustada — o passo 3 pôs ali a predição do tempo mais "
        "próximo, ou o passo 4 sorteou porque ela era negativa —, e onde o mínimo e o máximo "
        "incluem esses valores."
    )


def rotulo_da_curva(corretor: str | None) -> str:
    """O rótulo da curva: 'a que corrigiu o sinal' só quando o corretor do pedido é o LOESS."""
    if corretor == CORRETOR_DA_CURVA:
        return ROTULO_DA_CURVA
    return ROTULO_DA_CURVA_DE_OUTRO_CORRETOR


def frase_do_corretor(corretor: str | None) -> str | None:
    """A frase do rodapé quando a curva não é a que corrige o sinal. `None` com o LOESS.

    São três casos: o LOESS, outro corretor nomeado e nenhum declarado. Os dois últimos ficam
    separados porque 'a correção escolhida é outra' e 'ainda não se sabe qual corre' pedem ações
    diferentes de quem lê.
    """
    if corretor == CORRETOR_DA_CURVA:
        return None
    if corretor is None:
        return (
            "O pedido não declara o corretor da correção: a curva desenhada é a que o LOESS "
            "usaria, e só corrige o sinal se o corretor for o LOESS."
        )
    return (
        f"A correção escolhida no pedido é «{corretor}», e não o LOESS: a curva desenhada é a que "
        "o LOESS usaria, e não a que corrige o sinal."
    )


def recorte_da_curva_do_loess(*, curva: CurvaDoLoess, n_features_totais: Any) -> Recorte:
    """O rodapé: uma feature de quantas a matriz tem, escolhida e não sorteada."""
    totais = inteiro_do_fio(n_features_totais)
    if totais < 1:
        raise MatrizVaziaError(
            "a curva do LOESS não declara de quantas features a feature saiu: "
            f"`n_features` é {n_features_totais!r}. Sem a rota da curva corrida, não há população"
        )
    ausencias = frases_das_ausencias(curva)
    do_corretor = frase_do_corretor(curva.corretor)
    return Recorte(
        desenhadas=1,
        disponiveis=totais,
        amostragem=f"a feature escolhida — {curva.feature} —, sem sorteio",
        matriz=MATRIZ,
        metodo=" ".join(
            (
                metodo_da_curva(curva.semente, curva.spans_por_lote),
                *(() if do_corretor is None else (do_corretor,)),
                *ausencias,
            )
        ),
    )


def frases_das_ausencias(curva: CurvaDoLoess) -> tuple[str, ...]:
    """O porquê de cada ausência: sem curva, sem curvas sem um QC, QC sem predição.

    Vai no rodapé e não dentro do eixo, onde o texto cairia por cima dos pontos e por baixo da
    legenda. O rodapé quebra a frase e reserva o espaço dela. No eixo fica só uma marca curta
    quando falta a curva.
    """
    frases: list[str] = []
    if not curva.tem_curva:
        motivo = curva.motivo_sem_curva or "a correção não devolveu curva"
        frases.append(f"Sem curva: {_sem_ponto(motivo)}.")
    if curva.motivo_sem_envelope is not None:
        frases.append(f"Sem curvas sem um QC: {_sem_ponto(curva.motivo_sem_envelope)}.")
    sem_predicao = [m for q, m in zip(curva.qc, curva.motivo_sem_predicao, strict=True) if q and m]
    if sem_predicao:
        frases.append(
            f"{len(sem_predicao)} QC(s) sem predição — a curva sem ele não existe: "
            f"{_sem_ponto(sem_predicao[0])}."
        )
    return tuple(frases)


def _sem_ponto(motivo: str) -> str:
    """O motivo sem o ponto final. A frase do rodapé põe o dela, e o motivo da régua já traz um."""
    return motivo.rstrip().rstrip(".")


def frase_da_populacao(curva: CurvaDoLoess) -> str:
    """A segunda linha do título: quantos QCs, quantos com predição e de quantas curvas o envelope.

    O número de QCs decide o que se pode ler (sete pontos é pouco), por isso vai no título.
    """
    n_qc = int(curva.qc.sum())
    com_predicao = int(np.isfinite(curva.predicao_sem_o_qc[curva.qc]).sum())
    qcs = "1 QC" if n_qc == 1 else f"{n_qc} QCs"
    predicoes = (
        "nenhum com predição sem ele"
        if com_predicao == 0
        else f"{com_predicao} com predição sem ele"
    )
    envelope = (
        "sem envelope"
        if curva.n_curvas_do_envelope == 0
        else f"envelope de {curva.n_curvas_do_envelope} curva(s)"
    )
    return f"{qcs}, {predicoes}; {envelope}"


def figura_da_curva_do_loess(*, curva: CurvaDoLoess, recorte: Recorte) -> Figure:
    """Os QCs, a curva que corrigiu, a predição de cada QC sem ele e o envelope, por lote.

    Nenhuma linha atravessa a fronteira de lote: cada lote tem a sua curva, ajustada aos QCs
    dele, e um traço a ligá-los desenharia uma continuidade que a correção não tem.
    """
    figura = nova_figura()
    eixo = figura.add_subplot(1, 1, 1)
    cor_da_curva, cor_dos_qcs, cor_da_predicao = cores_do_ciclo(3)

    lotes = _lotes_na_ordem(curva.lotes)
    for lote in lotes:
        do_lote = np.array([r == lote for r in curva.lotes], dtype=bool)
        sufixo = f" — lote {lote}" if len(lotes) > 1 else ""
        _desenha_o_envelope(eixo, curva, do_lote, cor_da_curva, sufixo)
        if curva.tem_curva:
            eixo.plot(
                curva.ordens[do_lote],
                curva.curva[do_lote],
                color=cor_da_curva,
                linewidth=1.6,
                label=f"{rotulo_da_curva(curva.corretor)}{sufixo}",
            )

    if curva.tem_curva:
        _marca_as_posicoes_nao_ajustadas(eixo, curva, cor_da_curva)
    _desenha_os_qcs(eixo, curva, cor_dos_qcs)
    _desenha_as_predicoes(eixo, curva, cor_da_predicao)
    _declara_o_que_falta(eixo, curva)

    eixo.set_xlabel(ROTULO_DE_X)
    eixo.set_ylabel(ROTULO_DE_Y)
    eixo.set_title(f"{TITULO} — {curva.feature}\n{frase_da_populacao(curva)}")
    legenda_fora(eixo, _maos_da_legenda(eixo))
    aplica(figura, recorte)
    return figura


# ---------------------------------------------------------------------------
# As camadas do desenho
# ---------------------------------------------------------------------------


def _desenha_o_envelope(
    eixo: Axes, curva: CurvaDoLoess, do_lote: np.ndarray, cor: Any, sufixo: str
) -> None:
    """O envelope do lote: os dois limites como linhas e a faixa entre eles.

    Os limites são `Line2D` e não `fill_between`, cuja coleção os testes e o cursor da janela
    leem como um ponto só em `(0, 0)`. As duas linhas são o dado, e é sobre elas que os testes
    afirmam.

    A faixa é um `Polygon` e só repete as linhas: cheia no lote todo, e hachurada por cima nos
    trechos em que alguma curva sem um QC não foi ajustada. As máscaras desses trechos são
    dilatadas de uma posição, para a hachura ir até a injeção vizinha.
    """
    x = curva.ordens[do_lote]
    inferior = curva.envelope_inferior[do_lote]
    superior = curva.envelope_superior[do_lote]
    finito = np.isfinite(inferior) & np.isfinite(superior)
    if not finito.any():
        return
    for limite, nome in (
        (inferior, ROTULO_DO_LIMITE_INFERIOR),
        (superior, ROTULO_DO_LIMITE_SUPERIOR),
    ):
        eixo.plot(
            x,
            np.where(finito, limite, np.nan),
            color=cor,
            linewidth=0.7,
            alpha=0.8,
            label=f"{nome}{sufixo}",
        )
    for trecho in _trechos(finito):
        eixo.add_patch(
            Polygon(
                _contorno(x[trecho], inferior[trecho], superior[trecho]),
                closed=True,
                facecolor=cor,
                alpha=0.22,
                linewidth=0.0,
                label=ROTULO_DO_ENVELOPE,
            )
        )
    for trecho in _trechos(_dilata(curva.envelope_nao_ajustado[do_lote] & finito) & finito):
        eixo.add_patch(
            Polygon(
                _contorno(x[trecho], inferior[trecho], superior[trecho]),
                closed=True,
                fill=False,
                edgecolor=cor,
                hatch="///",
                linewidth=0.0,
                label=ROTULO_DO_ENVELOPE_NAO_AJUSTADO,
            )
        )


def _marca_as_posicoes_nao_ajustadas(eixo: Axes, curva: CurvaDoLoess, cor: Any) -> None:
    """Os marcadores sobre a curva onde ela não é o ajuste: preenchida (passo 3) ou sorteada (4)."""
    for mascara, rotulo, marcador in (
        (curva.curva_preenchida, ROTULO_DA_POSICAO_PREENCHIDA, "|"),
        (curva.curva_sorteada, ROTULO_DA_POSICAO_SORTEADA, "v"),
    ):
        if mascara.any():
            eixo.plot(
                curva.ordens[mascara],
                curva.curva[mascara],
                linestyle="none",
                marker=marcador,
                markersize=6,
                color=cor,
                label=rotulo,
            )


def _desenha_os_qcs(eixo: Axes, curva: CurvaDoLoess, cor: Any) -> None:
    """Os QCs medidos. Os que ficaram fora do ajuste vão vazados, só quando há curva."""
    finito = curva.qc & np.isfinite(curva.valores)
    no_ajuste = finito & curva.qc_no_ajuste if curva.tem_curva else finito
    fora = finito & ~no_ajuste
    if no_ajuste.any():
        eixo.plot(
            curva.ordens[no_ajuste],
            curva.valores[no_ajuste],
            linestyle="none",
            marker="o",
            color=cor,
            label=ROTULO_DO_QC,
        )
    if fora.any():
        eixo.plot(
            curva.ordens[fora],
            curva.valores[fora],
            linestyle="none",
            marker="o",
            markerfacecolor="none",
            color=cor,
            label=ROTULO_DO_QC_FORA_DO_AJUSTE,
        )


def _desenha_as_predicoes(eixo: Axes, curva: CurvaDoLoess, cor: Any) -> None:
    """A predição de cada QC pela curva que não o viu, e o segmento até o valor medido."""
    tem = curva.qc & np.isfinite(curva.predicao_sem_o_qc)
    if not tem.any():
        return
    ligavel = tem & np.isfinite(curva.valores)
    if ligavel.any():
        eixo.vlines(
            curva.ordens[ligavel],
            curva.valores[ligavel],
            curva.predicao_sem_o_qc[ligavel],
            colors=[cor],
            linestyles="dotted",
            linewidth=0.9,
            label="_nolegend_",
        )
    eixo.plot(
        curva.ordens[tem],
        curva.predicao_sem_o_qc[tem],
        linestyle="none",
        marker="D",
        markerfacecolor="none",
        markersize=6,
        color=cor,
        label=ROTULO_DA_PREDICAO,
    )


def _declara_o_que_falta(eixo: Axes, curva: CurvaDoLoess) -> None:
    """A marca curta no eixo quando falta a curva; o porquê por extenso está no rodapé."""
    if not curva.tem_curva:
        eixo.text(
            0.01,
            0.99,
            MARCA_SEM_CURVA,
            transform=eixo.transAxes,
            ha="left",
            va="top",
            fontsize="small",
        )


# ---------------------------------------------------------------------------
# Auxiliares
# ---------------------------------------------------------------------------


def _maos_da_legenda(eixo: Axes) -> list[Artist]:
    """Uma entrada por rótulo, sem os limites do envelope, que a faixa representa na legenda.

    Os limites têm rótulo próprio porque é por ele que os testes e o cursor os nomeiam; na
    legenda repetiriam a faixa duas vezes por lote. As faixas de vários trechos partilham o
    rótulo, e a legenda mostra a primeira.
    """
    maos: list[Artist] = []
    vistos: set[str] = set()
    artistas, rotulos = eixo.get_legend_handles_labels()
    for artista, rotulo in zip(artistas, rotulos, strict=True):
        if rotulo.startswith((ROTULO_DO_LIMITE_INFERIOR, ROTULO_DO_LIMITE_SUPERIOR)):
            continue
        if rotulo not in vistos:
            vistos.add(rotulo)
            maos.append(artista)
    return maos


def _trechos(mascara: np.ndarray) -> list[slice]:
    """Os trechos contíguos em que a máscara é verdadeira, como fatias."""
    trechos: list[slice] = []
    inicio: int | None = None
    for indice, valor in enumerate(mascara.tolist()):
        if valor and inicio is None:
            inicio = indice
        elif not valor and inicio is not None:
            trechos.append(slice(inicio, indice))
            inicio = None
    if inicio is not None:
        trechos.append(slice(inicio, mascara.size))
    return trechos


def _contorno(x: np.ndarray, inferior: np.ndarray, superior: np.ndarray) -> np.ndarray:
    """O polígono da faixa: o limite de baixo para a direita e o de cima de volta."""
    return np.column_stack(
        (np.concatenate((x, x[::-1])), np.concatenate((inferior, superior[::-1])))
    )


def _dilata(mascara: np.ndarray) -> np.ndarray:
    """A máscara com um vizinho a mais de cada lado."""
    saida = mascara.copy()
    saida[1:] |= mascara[:-1]
    saida[:-1] |= mascara[1:]
    return saida


def _lotes_na_ordem(rotulos: Sequence[str]) -> tuple[str, ...]:
    """Os lotes distintos, na ordem da primeira aparição, que é a de injeção."""
    vistos: dict[str, None] = {}
    for rotulo in rotulos:
        vistos.setdefault(rotulo, None)
    return tuple(vistos)
