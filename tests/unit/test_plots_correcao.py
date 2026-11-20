"""O painel da correção.

> **Este arquivo testava TRES modulos e foi dividido em 17/09/2026**, no mesmo commit em
> que `plots/correcao.py` se dividiu. Um arquivo de teste chamado `correcao` a cobrir tambem o
> violino e a composicao reproduziria, do lado do teste, exactamente o descasamento de assunto
> que a divisao veio desfazer.
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import TYPE_CHECKING, Any, get_args

import numpy as np
import pytest
from matplotlib.colors import to_hex
from scipy.stats import chi2

from refino.decision.palpite import LIMIAR_DE_EMPATE_DO_PALPITE
from refino.decision.policy import ABSTENTION_MARGIN
from refino.domain.labels import Correction
from refino.plots.base import (
    CORRETOR_NAO_DECLARADO,
    MatrizVaziaError,
)
from refino.plots.catalogo import CONSTRUTORES, Vinculo
from refino.plots.correcao import (
    ALTURAS_DAS_LINHAS,
    ALTURAS_SEM_FILA,
    AMBAR_DO_MELHOR,
    CHEIOS,
    CINZA_POR_OMISSAO,
    COR_DA_REGIAO,
    CORES_DA_PCA,
    DICA_DE_MEDIR,
    INTERVALO_MALFORMADO,
    NOME_DO_NONE,
    PRETO_DO_NONE,
    ROTULO_ANTES,
    ROTULO_DA_REGIAO,
    ROTULO_DEPOIS,
    ROTULO_DO_EIXO_DO_ANTES,
    ROTULO_DO_INTERVALO,
    ROTULO_SEM_QC,
    SEM_RANKING,
    TITULO_DAS_BARRAS,
    TITULO_DO_PAINEL,
    VERMELHO_DO_SEGUNDO,
    FilaAusente,
    FilaDeBarras,
    IntervaloAusente,
    IntervaloDeBarra,
    LadoDaCorrecao,
    ParDaCorrecao,
    cores_da_fila,
    figura_do_painel_da_correcao,
    fila_do_fio,
    intervalo_do_fio,
    lado_da_correcao,
    par_da_correcao,
    percentagem_do_r,
    recorte_do_painel_da_correcao,
    rotulo_da_regiao,
    rotulo_da_serie,
)
from refino.plots.pca import (
    DIMENSAO_DA_ELIPSE,
    MINIMO_PARA_ELIPSE,
    Elipse,
    ElipseAusente,
    PcaDesalinhadaError,
    TipoDeAmostra,
    elipse_das_amostras,
)
from refino.plots.recorte import (
    RecorteInvalidoError,
    decimal,
    linhas_do_rodape,
)
from refino.reporting.model import ITENS_OBRIGATORIOS
from tests.arnes_de_figura import (
    confere_a_matriz_do_titulo,
    confere_series,
    confere_vista,
    numeros_declarados,
    planta_corte_de_eixo,
    planta_eixos_trocados,
    sem_a_matriz,
    series,
    texto_corrido,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

# **Os três números do PNG que o laboratório lê**, do `pos_hilic`:
# `serrf/data/output/pos_hilic_toserrf - normalization result/Bar Plot and PCA plot.png`. São
# **observação sobre aquele arquivo** e não invariante — aqui servem de forma plausível e
# de âncora da convenção de escrita do valor, que é o que `percentagem_do_r` replica.
SCORES_DO_R: dict[str, float | None] = {"none": 1.447, "batch_loess": 1.050, "SERRF": 0.693}

N_AMOSTRAS = 24

# **O vocabulário fechado dos corretores, DERIVADO e não redigitado.** A lista existe aqui para uma
# asserção só — que o recuo do eixo direito não seja um nome de corretor —, e uma cópia à mão
# deixaria de acusar no dia em que o vocabulário ganhasse um membro chamado como o recuo.
VOCABULARIO_DOS_CORRETORES: tuple[str, ...] = get_args(Correction)


# ---------------------------------------------------------------------------
# As fixturas: um par de lados de estrutura CONHECIDA
# ---------------------------------------------------------------------------


def tipos_de_prova(*, com_qc: bool = True, com_validacao: bool = False) -> list[str]:
    """Um vetor de tipos com as três séries endereçáveis, e o QC minoritário como no corpus real."""
    tipos: list[str] = []
    for i in range(N_AMOSTRAS):
        if com_qc and i % 6 == 0:
            tipos.append("qc")
        elif com_validacao and i % 7 == 3:
            tipos.append("validate")
        else:
            tipos.append("sample")
    return tipos


def lado_de_prova(
    rotulo: str,
    matriz: str,
    *,
    deslocamento: float = 0.0,
    dispersao: float | None = 0.98,
    usadas: int = 6035,
    totais: int = 6035,
    digest: str = "a" * 16,
    tipos: list[str] | None = None,
) -> LadoDaCorrecao:
    """Um lado com `x` e `y` DIFERENTES entre si, e é isso que torna a planta dos eixos honesta.

    `planta_eixos_trocados` recusa uma série com `x == y` — trocar os eixos seria a identidade —, e
    a função a que se chama no teste tem de cair sobre dado em que as duas coisas diferem.
    """
    quantos = len(tipos) if tipos is not None else N_AMOSTRAS
    x = np.linspace(-4.0, 4.0, quantos) + deslocamento
    y = np.cos(np.linspace(0.0, 3.0, quantos)) * 2.0
    return lado_da_correcao(
        rotulo=rotulo,
        matriz=matriz,
        pc1=[float(v) for v in x],
        pc2=[float(v) for v in y],
        tipos=tipos if tipos is not None else tipos_de_prova(),
        variancia_explicada=[0.34, 0.098],
        qc_dispersion_ratio=dispersao,
        n_features_usadas=usadas,
        n_features_totais=totais,
        digest_das_colunas_retidas=digest,
    )


def par_de_prova(**troca: Any) -> ParDaCorrecao:
    return par_da_correcao(
        antes=troca.pop("antes", lado_de_prova(ROTULO_ANTES, "ingerida")),
        depois=troca.pop(
            "depois", lado_de_prova(ROTULO_DEPOIS, "corrigida", deslocamento=1.5, dispersao=0.23)
        ),
        pretransformacao=troca.pop("pretransformacao", "none"),
        n_componentes=troca.pop("n_componentes", 2),
        # **`loess` e não `none` por omissão, e a escolha é do par causal e não do gosto.** Com o
        # corretor da fixtura a ser `none`, um construtor que escrevesse o **estágio** no eixo
        # direito continuaria a reprovar — mas um que escrevesse `raw data` dos dois lados passaria,
        # porque «não corrigir» e «dado cru» leem-se igual. A fixtura tem de cair sobre dado em que
        # as duas coisas diferem.
        corretor=troca.pop("corretor", "loess"),
    )


def fila_de_prova(**troca: Any) -> FilaDeBarras | FilaAusente:
    campos: dict[str, Any] = {
        "scores": dict(SCORES_DO_R),
        "vencedor": "SERRF",
        "abstem": False,
        "estimador": "leave_one_qc_out",
        "variante": "outlier_removed",
        "n_features_medidas": 200,
    }
    campos.update(troca)
    return fila_do_fio(**campos)


def figura_de_prova(
    par: ParDaCorrecao | None = None, fila: FilaDeBarras | FilaAusente | None = None
) -> Figure:
    par = par if par is not None else par_de_prova()
    fila = fila if fila is not None else fila_de_prova()
    return figura_do_painel_da_correcao(
        par=par, fila=fila, recorte=recorte_do_painel_da_correcao(par=par, fila=fila)
    )


def eixo_das_barras(figura: Figure) -> Axes:
    """O primeiro eixo é o da fila — `add_subplot(grelha[0, :])` corre primeiro."""
    return figura.axes[0]


def rodape_de(figura: Figure) -> str:
    """O texto do rodapé, lido do ARTISTA e não reconstruído do `Recorte` que o produziu."""
    return texto_corrido(figura.texts[-1].get_text())


# ---------------------------------------------------------------------------
# 1. A fila de barras — a ordem, as cores, e os dois valores escritos
# ---------------------------------------------------------------------------


def test_a_fila_tem_uma_barra_por_corretor_e_sai_DECRESCENTE() -> None:  # noqa: N802
    """`tools.R:1528`: `sort(qc_RSD_performance, decreasing = TRUE)` — o pior primeiro.

    **A âncora de não-vacuidade vem antes**: um `scores` que chegasse vazio daria fila
    ausente, e *"as barras estão ordenadas"* passaria sobre lista vazia.
    """
    fila = fila_de_prova()
    assert isinstance(fila, FilaDeBarras), fila
    assert len(fila.barras) == len(SCORES_DO_R), fila.barras

    valores = [b.rsd for b in fila.barras]
    assert valores == sorted(valores, reverse=True), valores
    assert [b.corretor for b in fila.barras] == ["none", "batch_loess", "SERRF"]


def test_a_ordem_das_barras_NAO_depende_da_ordem_do_dicionario() -> None:  # noqa: N802
    """Regra 3 do README: a mesma medição desenha a mesma figura, venha o JSON na ordem que vier.

    **O par causal cai sobre dado em que as duas coisas diferem:** os dois dicionários têm as chaves
    em ordens **opostas**, e a asserção é que a fila sai idêntica. Sem a primeira asserção, dois
    dicionários iguais passariam sem separar nada.
    """
    direta = dict(SCORES_DO_R)
    inversa = dict(reversed(list(SCORES_DO_R.items())))
    assert list(direta) != list(inversa), "âncora: as duas ordens de entrada são a mesma"

    uma = fila_de_prova(scores=direta)
    outra = fila_de_prova(scores=inversa)
    assert isinstance(uma, FilaDeBarras) and isinstance(outra, FilaDeBarras)
    assert [b.corretor for b in uma.barras] == [b.corretor for b in outra.barras]


def test_o_empate_resolve_se_pelo_NOME_e_nao_pela_entrada() -> None:  # noqa: N802
    """Dois corretores com o MESMO RSD saem sempre na mesma ordem — e é a ordem alfabética."""
    empatados = {"zeta": 0.5, "alfa": 0.5, "none": 1.0}
    uma = fila_de_prova(scores=empatados, vencedor="alfa")
    outra = fila_de_prova(scores=dict(reversed(list(empatados.items()))), vencedor="alfa")
    assert isinstance(uma, FilaDeBarras) and isinstance(outra, FilaDeBarras)
    assert [b.corretor for b in uma.barras] == ["none", "alfa", "zeta"]
    assert [b.corretor for b in outra.barras] == [b.corretor for b in uma.barras]


def test_as_QUATRO_cores_sao_as_do_tools_R_e_o_preto_do_none_ganha_do_ambar() -> None:  # noqa: N802
    """`tools.R:1529-1532`, com a ORDEM de aplicação dele — e a ordem é o conteúdo.

    O R pinta cinza, depois vermelho no penúltimo, depois âmbar no último, e **só então** preto no
    `none`. Quando o `none` é o vencedor as duas regras colidem, e é a do preto que ganha: um `none`
    de âmbar afirmaria que a correção eleita foi «não corrigir».
    """
    quatro = {"none": 1.4, "a": 1.2, "b": 1.0, "c": 0.8}
    fila = fila_de_prova(scores=quatro, vencedor="c")
    assert isinstance(fila, FilaDeBarras)
    assert [b.corretor for b in fila.barras] == ["none", "a", "b", "c"]
    assert cores_da_fila(fila) == (
        PRETO_DO_NONE,
        CINZA_POR_OMISSAO,
        VERMELHO_DO_SEGUNDO,
        AMBAR_DO_MELHOR,
    )

    # A colisão: com o `none` a ganhar, ele fica PRETO e não âmbar.
    ganhador = fila_de_prova(scores={"none": 0.3, "a": 1.2, "b": 1.0}, vencedor="none")
    assert isinstance(ganhador, FilaDeBarras)
    assert [b.corretor for b in ganhador.barras] == ["a", "b", "none"]
    cores = cores_da_fila(ganhador)
    assert cores[-1] == PRETO_DO_NONE, cores
    assert AMBAR_DO_MELHOR not in cores, cores


def test_com_abstem_NENHUMA_barra_leva_o_ambar_e_o_par_causal_separa() -> None:  # noqa: N802
    """R8: a política recusou a eleição, e a imagem não pode afirmá-la.

    **O par causal cai sobre dado em que as duas coisas diferem:** os dois corpos são idênticos
    menos o `abstem`, e a asserção é que as cores **divergem**. Sem a metade de baixo, *"não há
    âmbar"* passaria com uma figura que nunca o põe.
    """
    com = fila_de_prova(abstem=True)
    sem = fila_de_prova(abstem=False)
    assert isinstance(com, FilaDeBarras) and isinstance(sem, FilaDeBarras)

    cores_com = cores_da_fila(com)
    cores_sem = cores_da_fila(sem)
    assert AMBAR_DO_MELHOR in cores_sem, (
        f"sem abstenção também não há âmbar: {cores_sem}. A planta não separa nada"
    )
    assert AMBAR_DO_MELHOR not in cores_com, (
        f"com `abstem=True` uma barra continua em âmbar: {cores_com}. A figura afirma uma eleição "
        "que a política recusou"
    )
    assert cores_com != cores_sem

    # E o título diz que houve abstenção — a cor sozinha é uma ausência, e ausência não se lê.
    figura = figura_de_prova(fila=com)
    assert "ABSTEVE-SE" in texto_corrido(eixo_das_barras(figura).get_title())


def test_um_score_None_NAO_vira_barra_e_a_ausencia_e_declarada() -> None:  # noqa: N802
    """`nan` serializa como `null`: o corretor sai da fila e entra no título, nunca a altura zero.

    **O par causal separa `None` de `0.0`**, que é o desfecho que um `float(v or 0)` colapsaria — e
    uma barra colada ao eixo é o retrato de uma correção perfeita.
    """
    com_nulo = fila_de_prova(scores={"none": 1.4, "loess": 1.2, "serrf": None}, vencedor="loess")
    com_zero = fila_de_prova(scores={"none": 1.4, "loess": 1.2, "serrf": 0.0}, vencedor="serrf")
    assert isinstance(com_nulo, FilaDeBarras) and isinstance(com_zero, FilaDeBarras)

    assert [b.corretor for b in com_nulo.barras] == ["none", "loess"]
    assert com_nulo.nao_medidos == ("serrf",)
    assert [b.corretor for b in com_zero.barras] == ["none", "loess", "serrf"]
    assert com_zero.nao_medidos == ()
    assert len(com_nulo.barras) != len(com_zero.barras), "a planta não separa nada"

    titulo = texto_corrido(eixo_das_barras(figura_de_prova(fila=com_nulo)).get_title())
    assert "RSD indefinido" in titulo and "serrf" in titulo, titulo


def test_SO_o_none_e_o_vencedor_levam_valor_escrito() -> None:  # noqa: N802
    """`tools.R:1554-1555`: duas chamadas de `text`, e só duas.

    **O oráculo é derivado da entrada e não do construtor** (classe (d) do arnês): a percentagem
    esperada é recalculada aqui a partir do `scores` que se passou.
    """
    figura = figura_de_prova()
    anotados = {t.get_text() for t in eixo_das_barras(figura).texts}
    esperados = {percentagem_do_r(SCORES_DO_R["none"]), percentagem_do_r(SCORES_DO_R["SERRF"])}
    assert anotados == esperados, anotados
    assert percentagem_do_r(SCORES_DO_R["batch_loess"]) not in anotados, (
        "a barra do meio levou valor escrito: o R escreve só sobre o `none` e sobre o melhor"
    )


def test_com_o_none_a_VENCER_sai_um_rotulo_so() -> None:  # noqa: N802
    """As duas regras apontam para a mesma barra, e o R também escreve uma vez só."""
    fila = fila_de_prova(scores={"none": 0.3, "a": 1.2}, vencedor="none")
    anotados = [t.get_text() for t in eixo_das_barras(figura_de_prova(fila=fila)).texts]
    assert anotados == [percentagem_do_r(0.3)], anotados


def test_a_percentagem_e_signif_DA_FRACAO_e_so_depois_vezes_cem() -> None:  # noqa: N802
    """`signif(v, 4) * 100` e não `signif(v * 100, 4)` — a ordem muda o dígito impresso.

    **O oráculo é o PNG do laboratório**: a barra do `SERRF` do `pos_hilic` leva `69.3%`, e não
    `69.30%`. É a única cadeia desta figura que outra pessoa pode conferir contra o artefato que ela
    já tem aberto ao lado.
    """
    assert percentagem_do_r(0.69299) == "69,3%"
    assert percentagem_do_r(1.447) == "144,7%"
    # E a ordem inversa daria outro texto — sem esta linha, a asserção acima não separa as duas
    # convenções, porque para alguns valores elas coincidem.
    assert f"{float(f'{0.69299 * 100:.4g}'):g}".replace(".", ",") == "69,3", "âncora da convenção"
    assert percentagem_do_r(0.693001) == "69,3%"
    assert percentagem_do_r(1.9e-15) == "1,9e-13%"


def test_as_barras_sao_LineCollection_legivel_e_nunca_Rectangle() -> None:  # noqa: N802
    """`ax.bar` põe `Rectangle` em `ax.patches`, que nem o arnês nem o painel leem.

    **A asserção de ausência tem a âncora de presença ANTES**: `ax.patches` vazio é o que uma
    figura sem artista nenhum também dá.
    """
    eixo = eixo_das_barras(figura_de_prova())
    desenhadas = series(eixo)
    assert len(desenhadas) == len(SCORES_DO_R), [s.rotulo for s in desenhadas]
    assert {s.rotulo for s in desenhadas} == set(SCORES_DO_R), [s.rotulo for s in desenhadas]
    assert not eixo.patches, (
        f"a fila desenhou {len(eixo.patches)} `Rectangle`: `ax.bar` produz artista que nem o "
        "arnês nem `PainelDeFigura._artistas` leem, e o teste passaria a medir nada (KTD5)"
    )


def test_cada_barra_e_uma_SERIE_PROPRIA_e_a_altura_dela_e_o_RSD_em_percentagem() -> None:  # noqa: N802
    """Uma série por barra, e a classe (a) sobre cada uma.

    Uma chamada só de `ax.vlines` daria **uma** coleção com um rótulo, e o arnês indexa por rótulo:
    as três barras colapsariam numa série e nenhuma asserção conseguiria falar de uma delas.
    """
    eixo = eixo_das_barras(figura_de_prova())
    esperadas = {
        nome: ([posicao, posicao], [0.0, valor * 100.0])
        for posicao, (nome, valor) in enumerate(
            sorted(SCORES_DO_R.items(), key=lambda p: -p[1])  # type: ignore[arg-type, operator]
        )
    }
    assert confere_series(eixo, esperadas) == []


def test_nenhum_dos_tres_eixos_e_logaritmico() -> None:
    """A tentação é real — as barras vão de 1,9e-15 a 1,4 conforme o estimador —, e a resposta é o
    valor anotado sobre a barra, não a escala."""
    figura = figura_de_prova()
    assert len(figura.axes) == 3, [a.get_title() for a in figura.axes]
    for eixo in figura.axes:
        assert (eixo.get_xscale(), eixo.get_yscale()) == ("linear", "linear"), eixo.get_title()


# ---------------------------------------------------------------------------
# 2. A fila AUSENTE — um desfecho legítimo
# ---------------------------------------------------------------------------


def test_sem_medicao_nao_ha_barra_e_o_lugar_leva_TEXTO_declarado() -> None:  # noqa: N802
    """R2: a figura nunca desenha um RSD in-sample sobre um método que ajusta modelo.

    **O par causal compara os dois desfechos e exige que divirjam.** Sem a metade com fila, *"não há
    barra"* passaria sobre um construtor que nunca desenha barra nenhuma.
    """
    ausente = fila_de_prova(scores=None)
    assert isinstance(ausente, FilaAusente) and ausente.motivo == DICA_DE_MEDIR

    com = eixo_das_barras(figura_de_prova(fila=fila_de_prova()))
    sem = eixo_das_barras(figura_de_prova(fila=ausente))
    assert series(com), "âncora: com medição também não há barra, e a planta não separa nada"
    assert not series(sem), f"sem medição desenhou {len(series(sem))} série(s)"

    # **`texto_corrido` dos dois lados.** `aplica` requebra o
    # texto marcado pela largura REAL do eixo, então o que está desenhado leva quebras de linha
    # que ninguém escreveu. O contrato é o texto, não a quebra.
    textos = [texto_corrido(t.get_text()) for t in sem.texts]
    assert textos == [texto_corrido(DICA_DE_MEDIR)], textos
    # **O nome do botão e o custo, e não mais «medir corretor» — item 84, 25/09/2026.** O texto
    # antigo mandava correr «medir corretor» na «aba da correção»: nem o botão nem a aba têm esse
    # nome na janela. Que o nome citado é o do botão que EXISTE é conferido em
    # `tests/test_desktop_figuras.py::test_a_dica_nomeia_o_botao_e_a_aba_que_a_janela_tem`, porque
    # `plots/` não importa Qt e este arquivo também não.
    assert "«Medir qual corretor vence»" in DICA_DE_MEDIR, "o texto não diz onde clicar (R2)"
    assert "minutos" in DICA_DE_MEDIR, "o texto não diz o custo de clicar"


def test_medicao_sem_NENHUM_score_finito_e_outro_fato_que_medicao_ausente() -> None:  # noqa: N802
    """*"Não correu"* e *"correu e nada deu número"* pedem a mesma ação, e contam à parte.

    Um único motivo para os dois tornaria-os indistinguíveis em contagem — e é a distinção que faz
    `unassessable` existir do lado da decisão.
    """
    nada = fila_de_prova(scores={"none": None, "loess": None})
    ausente = fila_de_prova(scores={})
    assert isinstance(nada, FilaAusente) and isinstance(ausente, FilaAusente)
    assert nada.motivo == SEM_RANKING
    assert ausente.motivo == DICA_DE_MEDIR
    assert nada.motivo != ausente.motivo


def test_o_rodape_declara_o_motivo_da_fila_ausente_e_MUDA_com_ele() -> None:  # noqa: N802
    """O rodapé da variante sem fila não pode ser o mesmo da variante com fila."""
    par = par_de_prova()
    com = recorte_do_painel_da_correcao(par=par, fila=fila_de_prova()).texto()
    sem = recorte_do_painel_da_correcao(par=par, fila=fila_de_prova(scores=None)).texto()
    assert com != sem
    assert "leave_one_qc_out" in com and "leave_one_qc_out" not in sem
    assert "Medir qual corretor vence" in sem


# ---------------------------------------------------------------------------
# 3. A POPULAÇÃO por barra não é servida, e nada a deriva do teto do amostrador
# ---------------------------------------------------------------------------


def test_o_rodape_declara_que_a_populacao_por_barra_NAO_e_servida() -> None:  # noqa: N802
    """R6: a população não existe no corpo, e a figura diz isso.

    **Declarar é o oposto de omitir** — um rodapé silencioso sobre a população lê-se como um rodapé
    cuja população é a do denominador ao lado, que é da PCA e não das barras.
    """
    rodape = rodape_de(figura_de_prova())
    assert "população de cada barra: NÃO SERVIDA" in rodape, rodape


def test_nenhum_numero_da_figura_vem_do_TETO_do_amostrador() -> None:  # noqa: N802
    """**O par causal que impede a derivação proibida: rótulo certo sobre a população errada.**

    `n_features_medidas` é o `n_sampled_features` do **pedido**: o teto pedido ao amostrador, e não
    a população em que a mediana correu. Medido no `pos_hilic` real, com as mesmas 200
    features amostradas, o `none` teve **193** finitos e o `loess` **172** — as populações diferem
    entre barras, e o teto não é nenhuma das duas.

    A planta são dois corpos idênticos **menos** esse campo, e a asserção é que os números
    declarados pela figura são os **mesmos**. A âncora de não-vacuidade vem primeiro: uma figura que
    não declarasse número nenhum satisfaria a igualdade sobre duas listas vazias.
    """
    duzentas = fila_de_prova(n_features_medidas=200)
    setecentas = fila_de_prova(n_features_medidas=700)
    assert isinstance(duzentas, FilaDeBarras) and isinstance(setecentas, FilaDeBarras)
    assert duzentas.teto_do_amostrador != setecentas.teto_do_amostrador, (
        "âncora: o campo plantado não chegou ao objeto, e a planta não separa nada"
    )

    numeros_duzentas = numeros_declarados(figura_de_prova(fila=duzentas))
    numeros_setecentas = numeros_declarados(figura_de_prova(fila=setecentas))
    assert numeros_duzentas, "âncora: a figura não declara número nenhum"
    assert numeros_duzentas == numeros_setecentas, (
        f"trocar o teto do amostrador mudou o que a figura declara: {numeros_duzentas} contra "
        f"{numeros_setecentas}. Algum número da imagem foi derivado dele, e ele NÃO é a população "
        "em que a mediana de cada barra correu"
    )
    assert "200" not in numeros_duzentas and "700" not in numeros_setecentas


# ---------------------------------------------------------------------------
# 4. Os dois painéis de PCA
# ---------------------------------------------------------------------------


def test_os_dois_paineis_desenham_o_MESMO_numero_de_pontos_e_a_soma_bate() -> None:  # noqa: N802
    """A contagem por série sai do vetor de tipos, derivada aqui e não do construtor.

    **A linha da região é uma série e NÃO é ponto.** Ela entra no eixo com os vértices do
    contorno, e somá-los aos pontos daria uma contagem de injeções que a matriz não tem. Por isso é
    afirmada à parte, e exactamente uma por painel; a soma continua exacta sobre as séries de ponto.
    """
    tipos = tipos_de_prova(com_validacao=True)
    par = par_de_prova(
        antes=lado_de_prova(ROTULO_ANTES, "ingerida", tipos=tipos),
        depois=lado_de_prova(ROTULO_DEPOIS, "corrigida", deslocamento=1.5, tipos=tipos),
    )
    figura = figura_de_prova(par=par)
    _barras, antes, depois = figura.axes

    por_tipo = {t: sum(1 for x in tipos if x == t) for t in ("qc", "sample", "validate")}
    assert all(por_tipo.values()), f"âncora: alguma série ficou vazia — {por_tipo}"

    for eixo, lado in ((antes, ROTULO_ANTES), (depois, ROTULO_DEPOIS)):
        desenhadas = {s.rotulo: s for s in series(eixo)}
        assert len(desenhadas) == 4, sorted(desenhadas)
        assert rotulo_da_regiao(lado) in desenhadas, sorted(desenhadas)
        pontos = {r: s for r, s in desenhadas.items() if r != rotulo_da_regiao(lado)}
        assert len(pontos) == 3, sorted(pontos)
        for bruto, tipo in (
            ("qc", TipoDeAmostra.QC),
            ("sample", TipoDeAmostra.AMOSTRA),
            ("validate", TipoDeAmostra.VALIDACAO),
        ):
            rotulo = rotulo_da_serie(tipo, lado)
            assert rotulo in pontos, sorted(pontos)
            assert pontos[rotulo].x.size == por_tipo[bruto], rotulo
        assert sum(s.x.size for s in pontos.values()) == len(tipos)


def test_os_rotulos_de_serie_sao_UNICOS_entre_os_dois_paineis() -> None:  # noqa: N802
    """«QC — Before» e «QC — After» são rótulos diferentes.

    `confere_series` indexa por rótulo, e o dia em que alguém quiser afirmar sobre as duas séries de
    QC ao mesmo tempo não pode ser o dia em que se descobre que elas têm o mesmo nome.
    """
    _barras, antes, depois = figura_de_prova().axes
    do_antes = {s.rotulo for s in series(antes)}
    do_depois = {s.rotulo for s in series(depois)}
    assert do_antes and do_depois, "âncora: um dos painéis não desenhou série nenhuma"
    assert not (do_antes & do_depois), f"rótulos partilhados: {sorted(do_antes & do_depois)}"


def test_trocar_os_dois_lados_MUDA_a_figura_e_lados_iguais_nao_mudam() -> None:  # noqa: N802
    """**O par causal com as duas direções, e a segunda é a que prova que a primeira não é vácua.**

    Trocar «antes» por «depois» tem de mudar o que se desenha; e com os dois lados **iguais** a
    troca tem de ser a identidade. Sem a segunda metade, *"a troca muda"* passaria com um construtor
    que produzisse figura diferente a cada chamada.
    """
    a = lado_de_prova(ROTULO_ANTES, "ingerida")
    b = lado_de_prova(ROTULO_DEPOIS, "corrigida", deslocamento=1.5, dispersao=0.23)
    assert not np.allclose(a.escores.x, b.escores.x), "âncora: os dois lados são iguais"

    direto = figura_de_prova(
        par=par_da_correcao(
            antes=a, depois=b, pretransformacao="none", n_componentes=2, corretor="loess"
        )
    )
    trocado = figura_de_prova(
        par=par_da_correcao(
            antes=lado_de_prova(ROTULO_ANTES, "corrigida", deslocamento=1.5, dispersao=0.23),
            depois=lado_de_prova(ROTULO_DEPOIS, "ingerida"),
            pretransformacao="none",
            n_componentes=2,
            corretor="loess",
        )
    )
    esquerda_direta = {s.rotulo: s.x.tolist() for s in series(direto.axes[1])}
    esquerda_trocada = {s.rotulo: s.x.tolist() for s in series(trocado.axes[1])}
    assert esquerda_direta != esquerda_trocada, "trocar os dois lados não mudou o painel esquerdo"

    iguais = figura_de_prova(
        par=par_da_correcao(
            antes=lado_de_prova(ROTULO_ANTES, "ingerida"),
            depois=lado_de_prova(ROTULO_DEPOIS, "ingerida"),
            pretransformacao="none",
            n_componentes=2,
            # **`none` aqui e não `loess`, porque os dois lados são a MESMA matriz.** O corretor é a
            # aresta entre os painéis, e declarar um que corrigiu sobre dois lados idênticos seria
            # a fixtura a afirmar o que a planta existe para negar.
            corretor="none",
        )
    )
    assert {s.rotulo.split(" — ")[0]: s.x.tolist() for s in series(iguais.axes[1])} == {
        s.rotulo.split(" — ")[0]: s.x.tolist() for s in series(iguais.axes[2])
    }, "com os dois lados iguais os painéis divergiram — a planta acima não é atribuível à troca"


def test_o_qc_sai_CHEIO_e_a_amostra_ABERTA_conferido_no_artista() -> None:  # noqa: N802
    """`pch=16` contra `pch=1` — `tools.R:1573-1574`, e R1 pede-o por extenso.

    **Conferido no artista e não no rodapé**, que é a diferença entre a figura fazer e a figura
    dizer que faz: um rodapé que prometesse o marcador cheio sobre uma série aberta seria a única
    coisa da imagem que outra pessoa pode desmentir.
    """
    _barras, antes, _depois = figura_de_prova().axes
    por_rotulo = {linha.get_label(): linha for linha in antes.lines}

    do_qc = por_rotulo[rotulo_da_serie(TipoDeAmostra.QC, ROTULO_ANTES)]
    da_amostra = por_rotulo[rotulo_da_serie(TipoDeAmostra.AMOSTRA, ROTULO_ANTES)]

    assert do_qc.get_markerfacecolor() == CORES_DA_PCA[TipoDeAmostra.QC], (
        do_qc.get_markerfacecolor()
    )
    assert da_amostra.get_markerfacecolor() == "none", da_amostra.get_markerfacecolor()
    assert do_qc.get_markerfacecolor() != da_amostra.get_markerfacecolor()
    assert TipoDeAmostra.QC in CHEIOS and TipoDeAmostra.AMOSTRA not in CHEIOS


def test_o_xlab_DIREITO_e_o_nome_do_CORRETOR_e_o_ESQUERDO_continua_raw_data() -> None:  # noqa: N802
    """`tools.R:1585` e `:1597`.

    O R escreve `xlab='raw data'` à esquerda e
    `names(qc_RSD_performance)[length(qc_RSD_performance)]` à direita, que é o nome do corretor cuja
    matriz aquele painel desenha. Antes escrevia-se ali «corrigida» — o **estágio** do pipeline —,
    e não
    por escolha: o nome do corretor não atravessava `ParDePcas`, e tirá-lo de outro campo do corpo
    seria procedência inventada num rótulo de eixo.

    **O par causal tem as duas metades, e a segunda é a que o torna atribuível:** trocar o corretor
    muda o rótulo da **direita**, e deixa o da **esquerda** intacto. Sem ela, *"o rótulo mudou"*
    passaria com um construtor que escrevesse o corretor nos dois eixos — que é o defeito oposto, e
    faria a imagem afirmar que a matriz da esquerda também foi corrigida.
    """
    com_loess = figura_de_prova(par=par_de_prova(corretor="loess"))
    com_serrf = figura_de_prova(par=par_de_prova(corretor="serrf"))

    direito_loess = texto_corrido(com_loess.axes[2].get_xlabel())
    direito_serrf = texto_corrido(com_serrf.axes[2].get_xlabel())

    assert direito_loess.startswith("loess"), direito_loess
    assert direito_serrf.startswith("serrf"), direito_serrf
    assert direito_loess != direito_serrf, (
        f"o rótulo do eixo direito não muda com o corretor: {direito_loess!r}. O painel continua a "
        "escrever o estágio do pipeline, e o do R escreve o nome do corretor"
    )
    assert "corrigida" not in direito_loess, (
        f"o eixo direito continua a levar o ESTÁGIO: {direito_loess!r}. O estágio já está no "
        "título daquele painel, e o que o R põe no eixo é o corretor"
    )

    esquerdo_loess = texto_corrido(com_loess.axes[1].get_xlabel())
    esquerdo_serrf = texto_corrido(com_serrf.axes[1].get_xlabel())
    assert esquerdo_loess.startswith(ROTULO_DO_EIXO_DO_ANTES), esquerdo_loess
    assert esquerdo_loess == esquerdo_serrf, (
        f"trocar o corretor mexeu no eixo ESQUERDO: {esquerdo_loess!r} contra {esquerdo_serrf!r}. "
        "Aquele painel desenha a matriz ingerida, e nenhum corretor a tocou"
    )
    assert "loess" not in esquerdo_loess and "serrf" not in esquerdo_serrf, (
        "o nome do corretor apareceu no eixo do «Before»: a imagem passa a afirmar que a matriz da "
        "esquerda também foi corrigida"
    )


def test_o_corretor_AUSENTE_no_fio_nao_vira_nome_de_corretor_nenhum() -> None:  # noqa: N802
    """Na ponta do desenho, campo que não veio não pode sair como `none` no eixo.

    **`none` é corretor legítimo deste vocabulário**, e é essa a razão de o recuo não poder ser um
    nome de corretor: um servidor que omitisse o campo faria a figura afirmar que a execução correu
    **sem correção** — procedência inventada, com a aparência de um rótulo correto.

    **O par causal separa**: com o campo a figura escreve o corretor; sem ele escreve o recuo, e os
    dois textos divergem.
    """
    com = texto_corrido(figura_de_prova(par=par_de_prova(corretor="serrf")).axes[2].get_xlabel())
    sem = texto_corrido(figura_de_prova(par=par_de_prova(corretor=None)).axes[2].get_xlabel())

    assert com.startswith("serrf"), com
    assert sem.startswith(CORRETOR_NAO_DECLARADO), sem
    assert com != sem, "o campo ausente produziu o mesmo rótulo do campo presente"
    assert CORRETOR_NAO_DECLARADO not in VOCABULARIO_DOS_CORRETORES, (
        f"o recuo {CORRETOR_NAO_DECLARADO!r} é um nome de corretor do vocabulário "
        f"{VOCABULARIO_DOS_CORRETORES}: a ausência do campo passa a ser indistinguível de uma "
        "execução real com aquele corretor"
    )


def test_o_agrupamento_de_QC_de_cada_lado_vai_no_TITULO_daquele_painel() -> None:  # noqa: N802
    """O número que responde à pergunta do laboratório, ao lado da nuvem a que pertence."""
    _barras, antes, depois = figura_de_prova().axes
    assert "0,98" in texto_corrido(antes.get_title()), antes.get_title()
    assert "0,23" in texto_corrido(depois.get_title()), depois.get_title()
    assert texto_corrido(antes.get_title()) != texto_corrido(depois.get_title())


def test_um_agrupamento_INDEFINIDO_e_declarado_e_nao_vira_zero() -> None:  # noqa: N802
    """`None` no corpo é *"não é definível"*, e zero seria agrupamento perfeito."""
    par = par_de_prova(antes=lado_de_prova(ROTULO_ANTES, "ingerida", dispersao=None))
    titulo = texto_corrido(figura_de_prova(par=par).axes[1].get_title())
    assert "agrupamento de QC indefinido" in titulo, titulo
    assert not math.isfinite(par.antes.dispersao)


def test_com_ZERO_QCs_a_figura_declara_a_ausencia_em_texto() -> None:  # noqa: N802
    """R7: insumo ausente é achado válido, nunca espaço em branco.

    **O QC é a régua técnica desta figura**, e é por isso que ele é o único tipo cuja ausência é
    escrita: um painel sem QC e sem aviso lê-se como um em que o QC agrupou tão bem que sumiu.

    **O par causal separa**: com QC não há a frase; sem QC há — e sem a metade de cima, *"há a
    frase"* passaria com um construtor que a escrevesse sempre.
    """
    sem_qc = tipos_de_prova(com_qc=False)
    assert "qc" not in sem_qc, "âncora: o vetor sem QC ainda tem QC"

    par = par_de_prova(
        antes=lado_de_prova(ROTULO_ANTES, "ingerida", tipos=sem_qc),
        depois=lado_de_prova(ROTULO_DEPOIS, "corrigida", deslocamento=1.5, tipos=sem_qc),
    )
    esperada = texto_corrido(ROTULO_SEM_QC)
    com_qc = figura_de_prova()
    assert esperada not in [texto_corrido(t.get_text()) for t in com_qc.axes[1].texts], (
        "a frase da ausência aparece COM QC desenhado — a planta não separa nada"
    )

    # `texto_corrido` dos dois lados: `aplica` requebra o texto marcado pela largura real do eixo.
    esquerda = figura_de_prova(par=par).axes[1]
    assert esperada in [texto_corrido(t.get_text()) for t in esquerda.texts], [
        t.get_text() for t in esquerda.texts
    ]
    # A âncora de não-vacuidade vem ANTES da asserção de ausência, e na mesma função:
    # um painel que não desenhasse série nenhuma satisfaria *"não há série de QC"*.
    rotulos = {s.rotulo for s in series(esquerda)}
    assert rotulos, "o painel sem QC ficou sem série nenhuma, e a asserção abaixo seria vácua"
    assert rotulo_da_serie(TipoDeAmostra.QC, ROTULO_ANTES) not in rotulos, rotulos


def test_um_painel_sem_ponto_finito_RECUSA_em_vez_de_desenhar_eixo_vazio() -> None:  # noqa: N802
    """`MatrizVaziaError` é a família que `PainelDeFigura._troca_a_figura` apanha.

    Um eixo montado e vazio ao lado de um cheio lê-se como figura quebrada, e é o **outro** lado que
    ficaria a parecer errado.
    """
    vazio = lado_da_correcao(
        rotulo=ROTULO_ANTES,
        matriz="ingerida",
        pc1=[float("nan")] * 4,
        pc2=[float("nan")] * 4,
        tipos=["qc", "sample", "sample", "sample"],
        variancia_explicada=[0.3, 0.1],
        qc_dispersion_ratio=0.5,
        n_features_usadas=10,
        n_features_totais=10,
        digest_das_colunas_retidas="b" * 16,
    )
    with pytest.raises(MatrizVaziaError, match="não há ponto a desenhar"):
        figura_de_prova(par=par_de_prova(antes=vazio))


def test_um_sampleType_FORA_do_dominio_levanta_na_fronteira() -> None:  # noqa: N802
    """Um valor desconhecido não pode cair numa série por omissão — a amostra sumiria."""
    with pytest.raises(PcaDesalinhadaError, match="não é um dos três"):
        lado_de_prova(ROTULO_ANTES, "ingerida", tipos=["blank"] * N_AMOSTRAS)


# ---------------------------------------------------------------------------
# 5. O rodapé — o par contagem/digest, e o que ele cabe
# ---------------------------------------------------------------------------


def test_o_rodape_diz_quando_os_dois_ESPACOS_divergem_e_o_digest_e_a_prova() -> None:  # noqa: N802
    """R1: contagem igual não é conjunto igual, e é só o digest que o denuncia.

    **O par causal cai sobre dado em que as duas coisas diferem:** os dois pares têm a **mesma**
    contagem de features retidas dos dois lados e digests diferentes. Um portão que lesse a contagem
    passaria nos dois, e é por isso que a planta é esta e não «retiveram números diferentes».
    """
    iguais = par_de_prova(
        antes=lado_de_prova(ROTULO_ANTES, "ingerida", usadas=6035, digest="c" * 16),
        depois=lado_de_prova(ROTULO_DEPOIS, "corrigida", usadas=6035, digest="c" * 16),
    )
    divergentes = par_de_prova(
        antes=lado_de_prova(ROTULO_ANTES, "ingerida", usadas=6035, digest="c" * 16),
        depois=lado_de_prova(ROTULO_DEPOIS, "corrigida", usadas=6035, digest="d" * 16),
    )
    assert iguais.antes.n_features_usadas == divergentes.depois.n_features_usadas, (
        "âncora: as contagens diferem entre os dois braços, e a planta deixaria de ser sobre o "
        "digest"
    )
    assert not iguais.espacos_divergem and divergentes.espacos_divergem

    texto_iguais = recorte_do_painel_da_correcao(par=iguais, fila=fila_de_prova()).texto()
    texto_divergentes = recorte_do_painel_da_correcao(par=divergentes, fila=fila_de_prova()).texto()
    assert "COINCIDENTES" in texto_iguais and "DIVERGENTES" not in texto_iguais
    assert "DIVERGENTES" in texto_divergentes
    assert texto_iguais != texto_divergentes


def test_o_rodape_declara_a_NAO_comparabilidade_dos_eixos_e_a_pretransformacao() -> None:  # noqa: N802
    """São dois SVD de matrizes diferentes, e sem a frase *"andou para a direita"* lê-se como
    fato do dado."""
    rodape = rodape_de(figura_de_prova())
    assert "NÃO são comparáveis" in rodape, rodape
    assert "«none»" in rodape, rodape
    assert "«ingerida» à esquerda e «corrigida» à direita" in rodape, rodape


def test_o_rodape_do_topo_declara_de_QUAL_painel_e_o_par_de_features() -> None:  # noqa: N802
    """`Recorte.texto()` escreve um único *"N de M features"*, e esta figura tem DUAS populações.

    Sem a frase, o leitor atribuiria o número do «antes» aos dois painéis, num
    campo cujo formato não é desta função.
    """
    par = par_de_prova(
        antes=lado_de_prova(ROTULO_ANTES, "ingerida", usadas=6035, totais=6035),
        depois=lado_de_prova(ROTULO_DEPOIS, "corrigida", usadas=5321, totais=6035),
    )
    recorte = recorte_do_painel_da_correcao(par=par, fila=fila_de_prova())
    assert recorte.desenhadas == 6035 and recorte.disponiveis == 6035
    texto = recorte.texto()
    assert "o par acima é do «Before»" in texto, texto
    assert "5.321" in texto, texto


@pytest.mark.parametrize(
    ("largura", "altura", "ppp"),
    [(10.0, 6.0, 100.0), (8.0, 4.5, 96.0), (6.5, 3.4, 96.0), (12.0, 7.0, 150.0)],
)
def test_o_rodape_cabe_nos_tamanhos_EXTREMOS_sem_levantar(  # noqa: N802
    largura: float, altura: float, ppp: float
) -> None:
    """Os quatro tamanhos, com o rodapé mais longo que esta figura produz.

    **O mais longo é o dos espaços DIVERGENTES**, e escolhê-lo é o que impede este teste de medir o
    caso fácil: a variante coincidente é 74 caracteres mais curta.

    O teto de `aplica` é `RESERVA_MAXIMA` = 45% da altura; a asserção deriva a reserva das linhas
    que `linhas_do_rodape` de facto produz naquele tamanho, e não de um literal.
    """
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure as FiguraNua

    from refino.plots.recorte import ENTRELINHA, MARGEM, RESERVA_MAXIMA, TAMANHO_DA_FONTE

    par = par_de_prova(
        antes=lado_de_prova(ROTULO_ANTES, "ingerida", digest="c" * 64),
        depois=lado_de_prova(ROTULO_DEPOIS, "corrigida", usadas=5321, digest="d" * 64),
    )
    assert par.espacos_divergem, "âncora: o rodapé medido não é o mais longo dos dois"
    texto = recorte_do_painel_da_correcao(par=par, fila=fila_de_prova()).texto()

    figura = FiguraNua(figsize=(largura, altura), dpi=ppp)
    FigureCanvasAgg(figura)
    linhas = linhas_do_rodape(figura, texto)
    reservado = len(linhas) * TAMANHO_DA_FONTE * ENTRELINHA / (altura * 72.0) + 2.0 * MARGEM
    assert reservado <= RESERVA_MAXIMA, (
        f"a {largura}x{altura}@{ppp} o rodapé quebra em {len(linhas)} linhas e precisaria de "
        f"{reservado:.1%} da altura, acima do teto de {RESERVA_MAXIMA:.0%} — `aplica` levantaria "
        "`RecorteInvalidoError` e a figura não desenharia"
    )


def test_a_figura_padrao_MONTA_e_o_rodape_nao_recusa() -> None:  # noqa: N802
    """A metade que o teste acima não alcança: `aplica` corre de facto, no tamanho de nascença."""
    figura = figura_de_prova()
    assert len(figura.axes) == 3
    assert figura.get_figheight() > 4.5, (
        "a altura voltou à padrão do pacote: com 4,5 polegadas e três eixos, o título das barras "
        "cai sobre o rótulo do eixo e as marcas caem sobre os títulos de baixo"
    )
    assert ALTURAS_DAS_LINHAS[1] > ALTURAS_DAS_LINHAS[0], (
        "a linha das barras passou a ser a mais alta: três barras lêem-se em menos altura do que "
        "duas nuvens de dispersão precisam"
    )
    assert texto_corrido(sem_a_matriz(figura.texts[0].get_text())) == TITULO_DO_PAINEL
    assert confere_a_matriz_do_titulo(figura) == []
    assert TITULO_DAS_BARRAS in texto_corrido(eixo_das_barras(figura).get_title())


def test_um_denominador_ausente_RECUSA_por_populacao_vazia() -> None:  # noqa: N802
    """Sem execução não há população, e `MatrizVaziaError` é o que o painel apanha."""
    par = par_de_prova(antes=lado_de_prova(ROTULO_ANTES, "ingerida", usadas=0, totais=0))
    with pytest.raises(MatrizVaziaError, match="não declara de quantas features"):
        recorte_do_painel_da_correcao(par=par, fila=fila_de_prova())


def test_declarar_mais_desenhadas_que_disponiveis_RECUSA() -> None:  # noqa: N802
    """`RecorteInvalidoError`: numerador e denominador de populações diferentes."""
    par = par_de_prova(antes=lado_de_prova(ROTULO_ANTES, "ingerida", usadas=99, totais=10))
    with pytest.raises(RecorteInvalidoError, match="Desenhar mais do que existe"):
        recorte_do_painel_da_correcao(par=par, fila=fila_de_prova())


# ---------------------------------------------------------------------------
# 6. As três plantas do arnês sobre esta figura
# ---------------------------------------------------------------------------


def test_a_figura_limpa_passa_na_classe_c_e_as_plantas_acusam() -> None:
    """A classe (c) sobre os três eixos, e as duas plantas que ela e a (a) apanham.

    **A planta entra como argumento**, então o negativo percorre exactamente o caminho que o portão
    percorre — negativo em processo não consegue ficar inerte.
    """
    figura = figura_de_prova()
    # **`confere_vista` devolve lista, e lista vazia é o que um eixo SEM ARTISTA também dá** — a
    # asserção de ausência tem a de presença ao lado, no mesmo laço.
    for eixo in figura.axes:
        assert series(eixo), f"o eixo {eixo.get_title()[:30]!r} não tem artista de dado nenhum"
        assert confere_vista(eixo) == [], eixo.get_title()

    cortado = figura_de_prova()
    planta_corte_de_eixo(eixo_das_barras(cortado))
    assert confere_vista(eixo_das_barras(cortado)) != []

    trocado = figura_de_prova()
    esquerda = trocado.axes[1]
    esperadas = {s.rotulo: (s.x.tolist(), s.y.tolist()) for s in series(esquerda)}
    planta_eixos_trocados(esquerda)
    assert confere_series(esquerda, esperadas) != []


# ---------------------------------------------------------------------------
# 7. O registo — exactamente um `item-6`, com vínculo de relatório
# ---------------------------------------------------------------------------


def test_o_painel_da_correcao_e_o_UNICO_dono_da_chave_item_6() -> None:  # noqa: N802
    """Um construtor, uma chave, e o `item-4` fica sem construtor — declarado.

    **Contar e não perguntar se existe**: `"item-6" in {...}` devolve o desfecho de sucesso com
    duas entradas iguais, e duas fontes para o mesmo desenho é o que se recusa.
    """
    donos = [c for c in CONSTRUTORES if c.chave == "item-6"]
    assert len(donos) == 1, [c.funcao for c in donos]
    assert donos[0].modulo == "correcao"
    assert donos[0].funcao == "figura_do_painel_da_correcao"
    assert donos[0].vinculo is Vinculo.RELATORIO

    # E o `item-4` continua sem construtor, que é a condição normal — a âncora afirma que ele
    # EXISTE no conteúdo obrigatório, senão *"ninguém o construiu"* passaria sobre item inventado.
    assert any(i.numero == 4 for i in ITENS_OBRIGATORIOS), [i.numero for i in ITENS_OBRIGATORIOS]
    assert not [c for c in CONSTRUTORES if c.chave == "item-4"], (
        "alguém registou a metade de cima desta mesma imagem como uma segunda entrada: um "
        "construtor tem uma chave só, e duas seriam duas fontes para um desenho (KTD2)"
    )


def test_o_item_6_pede_DUAS_matrizes_e_esta_figura_desenha_duas() -> None:  # noqa: N802
    """A bijeção é contra o **número** do item, e o título é o que diz o que ele pede.

    Sem esta leitura, *"o item 6 tem construtor"* seria verdade sobre uma figura que desenhasse uma
    matriz só — que é exactamente o que as quatro de `pca` fazem, e o motivo de nenhuma delas o ser.
    """
    item6 = next(i for i in ITENS_OBRIGATORIOS if i.numero == 6)
    assert "antes e depois" in item6.titulo and "QCs" in item6.titulo, item6.titulo

    figura = figura_de_prova()
    _barras, antes, depois = figura.axes
    assert ROTULO_ANTES in antes.get_title() and ROTULO_DEPOIS in depois.get_title()
    assert "ingerida" in antes.get_title() and "corrigida" in depois.get_title()
    assert any(b.corretor == NOME_DO_NONE for b in fila_de_prova().barras)  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# Item 84 — sem medição a linha de cima encolhe, e o texto não invade nem avisa
# ---------------------------------------------------------------------------


def _folgas_do_texto_sem_fila(figura: Figure) -> tuple[float, float]:
    """`(acima, abaixo)` em px: do texto ao título da figura, e dos títulos das PCAs ao texto.

    **Medidas a 7 x 4,8 pol., depois de `reajusta`, e não no tamanho de nascença** — que é o que a
    janela faz: põe a figura no tamanho do widget e refaz só a faixa do rodapé, sem novo
    `tight_layout`. No tamanho de nascença (7 x 7) a folga é larga com qualquer repartição, e uma
    linha encolhida a um quinto passava (medido: 1,8 e 0,8 px); a 7 x 4,8 a linha da repartição de
    antes deixava 24,5 / 19,1 px e a de `0,7` deixa 9,4 / 4,0 px.

    **Lidas da figura DESENHADA** — `canvas.draw()` antes —, porque antes dele as caixas são as de
    nascença.
    """
    from refino.plots.base import GID_DO_RODAPE
    from refino.plots.recorte import reajusta

    rodape = next(texto for texto in figura.texts if texto.get_gid() == GID_DO_RODAPE)
    figura.set_size_inches(7.0, 4.8)
    reajusta(figura, rodape)
    figura.canvas.draw()
    desenhador = figura.canvas.get_renderer()  # type: ignore[attr-defined]
    texto = eixo_das_barras(figura).texts[0].get_window_extent(desenhador)
    titulo = figura._suptitle
    assert titulo is not None, "âncora: a figura perdeu o título e a folga de cima não se mede"
    de_cima = titulo.get_window_extent(desenhador).y0 - texto.y1
    das_pcas = max(eixo.title.get_window_extent(desenhador).y1 for eixo in figura.axes[1:])
    return float(de_cima), float(texto.y0 - das_pcas)


def test_sem_medicao_a_linha_de_cima_ENCOLHE_e_com_medicao_nao() -> None:  # noqa: N802
    """A repartição muda com o desfecho da fila, e o par causal exige os dois.

    O item 84 mediu a metade de cima em branco: com a repartição das barras, um texto de quatro
    linhas ocupava uma linha feita para três barras. A linha de cima encolhe só quando não há fila —
    com fila, as barras continuam a ter a altura que sempre tiveram.
    """
    assert ALTURAS_SEM_FILA[0] < ALTURAS_DAS_LINHAS[0], "âncora: a repartição sem fila não encolhe"
    assert ALTURAS_SEM_FILA[1] == ALTURAS_DAS_LINHAS[1], "a linha das PCAs mudou junto — não devia"
    sem = figura_de_prova(fila=fila_de_prova(scores=None))
    com = figura_de_prova()
    assert tuple(eixo_das_barras(sem).get_gridspec().get_height_ratios()) == ALTURAS_SEM_FILA
    assert tuple(eixo_das_barras(com).get_gridspec().get_height_ratios()) == ALTURAS_DAS_LINHAS


def test_sem_medicao_o_texto_cabe_na_linha_de_cima_e_nao_invade() -> None:
    """O texto requebrado fica entre o título da figura e os títulos das PCAs, sem os invadir.

    **É a asserção que dá origem ao `0,7`** de `ALTURAS_SEM_FILA`: um número menor põe o texto
    sobre os títulos, e o teste abaixo planta exactamente isso.
    """
    acima, abaixo = _folgas_do_texto_sem_fila(figura_de_prova(fila=fila_de_prova(scores=None)))
    assert acima > 0.0 and abaixo > 0.0, (
        f"o texto sem fila invade: folga de {acima:.1f} px até o título da figura e de "
        f"{abaixo:.1f} px até os títulos das PCAs"
    )


def test_com_a_linha_encolhida_demais_a_colisao_ACUSA(  # noqa: N802
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """O par causal da colisão: a mesma figura, com a linha de cima reduzida a um quinto."""
    from refino.plots import correcao as modulo

    limpa = _folgas_do_texto_sem_fila(figura_de_prova(fila=fila_de_prova(scores=None)))
    assert min(limpa) > 0.0, "âncora: sem a planta já colidia"
    monkeypatch.setattr(modulo, "ALTURAS_SEM_FILA", (0.2, 1.6))
    plantada = _folgas_do_texto_sem_fila(figura_de_prova(fila=fila_de_prova(scores=None)))
    assert min(plantada) < 0.0, f"a linha encolhida demais não fez o texto invadir: {plantada}"


def test_sem_medicao_a_figura_NAO_emite_o_aviso_do_tight_layout(  # noqa: N802
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """O aviso *«Tight layout not applied»* do item 84, e o par causal que mostra de onde vinha.

    O texto nasce numa linha só, mais larga que a figura, e o primeiro `tight_layout` de `aplica`
    — o que corre antes do requebrar — desistia. Pô-lo fora do layout fecha o aviso; repô-lo dentro
    tem de o fazer voltar, senão este teste não mede nada.
    """
    import warnings

    from refino.plots import correcao as modulo

    def avisos() -> list[str]:
        with warnings.catch_warnings(record=True) as achados:
            warnings.simplefilter("always")
            figura_de_prova(fila=fila_de_prova(scores=None)).canvas.draw()
        return [str(a.message) for a in achados if "Tight layout" in str(a.message)]

    assert avisos() == [], "sem medição a figura ainda avisa que o tight_layout desistiu"

    # **A planta é ESTREITA**: repõe no layout só o texto que `_desenha_a_fila` pôs no eixo das
    # barras, e mais nenhum artista da figura — senão o aviso podia vir de outro sítio.
    original = modulo._desenha_a_fila
    repostos: list[object] = []

    def com_o_texto_no_layout(eixo: Axes, fila: FilaDeBarras | FilaAusente) -> None:
        original(eixo, fila)
        for texto in eixo.texts:
            texto.set_in_layout(True)
            repostos.append(texto)

    monkeypatch.setattr(modulo, "_desenha_a_fila", com_o_texto_no_layout)
    assert avisos(), "com o texto de volta ao layout o aviso não voltou"
    assert repostos, "âncora: a planta não aterrou — nenhum texto foi reposto no layout"


# ---------------------------------------------------------------------------
# 8. A região de predição de cada painel
# ---------------------------------------------------------------------------


def _esperadas_do_lado(lado: LadoDaCorrecao) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """As séries de ponto de um lado, derivadas das máscaras e não lidas da figura (classe (a))."""
    esperadas: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for tipo in (TipoDeAmostra.VALIDACAO, TipoDeAmostra.AMOSTRA, TipoDeAmostra.QC):
        mascara = lado.escores.mascara(tipo)
        if mascara.any():
            esperadas[rotulo_da_serie(tipo, lado.rotulo)] = (
                lado.escores.x[mascara],
                lado.escores.y[mascara],
            )
    return esperadas


def _com_raio_qui_quadrado(regiao: Elipse) -> Elipse:
    """A MESMA elipse com o quantil de qui-quadrado — a aproximação recusada.

    Os semieixos são `sqrt(c² · autovalor)`, então trocar o quantil é multiplicá-los pela raiz da
    razão dos dois quantis; centro e ângulo não mudam. É a planta: a região que o
    MetaboAnalyst desenharia sobre os mesmos pontos.
    """
    corte = float(chi2.ppf(1.0 - regiao.alfa, DIMENSAO_DA_ELIPSE))
    fator = math.sqrt(corte / regiao.c2)
    return replace(
        regiao, c2=corte, semieixos=(regiao.semieixos[0] * fator, regiao.semieixos[1] * fator)
    )


def _tipos_com_estudo(quantas: int) -> list[str]:
    """`N_AMOSTRAS` tipos com exactamente `quantas` amostras de estudo; QC e validação no resto.

    **O resto é validação e não amostra**, porque a região é calculada sobre as amostras de ESTUDO:
    com o resto como `sample`, a fixtura não teria poucas amostras de estudo — teria muitas.
    """
    tipos = ["qc" if i % 6 == 0 else "validate" for i in range(N_AMOSTRAS)]
    livres = [i for i, tipo in enumerate(tipos) if tipo == "validate"]
    for i in livres[:quantas]:
        tipos[i] = "sample"
    return tipos


def test_cada_lado_desenha_UMA_regiao_e_ela_e_a_de_elipse_das_amostras() -> None:  # noqa: N802
    """Classe (a) com a série nova: cada painel tem exactamente uma linha de região, e ela é o
    contorno de `pca.elipse_das_amostras` sobre os scores DAQUELE painel.

    **O sinal do componente não é canónico entre as duas decomposições**, então cada lado tem
    a própria região; a âncora afirma que as duas diferem, senão *"cada lado tem a sua"* passaria
    com um construtor que desenhasse a do «Before» nos dois.

    **E a planta cai sobre dado em que as duas regiões diferem:** a de qui-quadrado sobre os
    mesmos pontos é outra curva, e o mesmo `confere_series` acusa-a.
    """
    par = par_de_prova()
    figura = figura_de_prova(par=par)
    _barras, antes, depois = figura.axes

    regioes = [elipse_das_amostras(lado.escores) for lado in (par.antes, par.depois)]
    assert all(isinstance(r, Elipse) for r in regioes), "âncora: um lado não tem região a desenhar"
    centros = [r.centro for r in regioes if isinstance(r, Elipse)]
    assert not np.allclose(centros[0], centros[1]), (
        "âncora: as duas regiões coincidem, e «cada lado tem a sua» não seria medido"
    )

    for eixo, lado, regiao in zip((antes, depois), (par.antes, par.depois), regioes, strict=True):
        assert isinstance(regiao, Elipse)
        rotulo = rotulo_da_regiao(lado.rotulo)
        com_o_rotulo = [s for s in series(eixo) if s.rotulo == rotulo]
        assert len(com_o_rotulo) == 1, [s.rotulo for s in series(eixo)]

        esperadas = {**_esperadas_do_lado(lado), rotulo: regiao.contorno()}
        assert confere_series(eixo, esperadas) == [], confere_series(eixo, esperadas)

        qui = _com_raio_qui_quadrado(regiao)
        assert not np.allclose(qui.contorno()[0], regiao.contorno()[0]), (
            "âncora: a região de qui-quadrado coincide com a de Hotelling neste dado, e a planta "
            "não separa nada"
        )
        plantada = {**esperadas, rotulo: qui.contorno()}
        assert confere_series(eixo, plantada) != [], (
            f"o painel «{lado.rotulo}» passa com a região de qui-quadrado no lugar da de Hotelling"
        )


def test_o_painel_chama_a_MESMA_funcao_que_a_simulacao_de_cobertura_valida(  # noqa: N802
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A região do painel sai de `pca.elipse_de_hotelling` — a que
    `test_ktd11_a_cobertura_da_elipse_bate_o_nominal_e_a_qui_quadrado_NAO` valida por Monte Carlo
    com n = 7, 13 e 30 — e sobre as amostras de ESTUDO de cada lado, com o QC de fora.

    **A espia deixa a função correr**, então o que se afirma é a chamada e a população, e não um
    desenho diferente. A âncora diz que a população de estudo é menor que a nuvem: sem QC na
    fixtura, «só as de estudo» e «todas as injeções» seriam o mesmo vetor.
    """
    from refino.plots import pca as modulo_pca

    original = modulo_pca.elipse_de_hotelling
    recebidos: list[tuple[np.ndarray, np.ndarray]] = []

    def espia(x: np.ndarray, y: np.ndarray, **opcoes: Any) -> Elipse | ElipseAusente:
        recebidos.append((np.array(x, dtype=float), np.array(y, dtype=float)))
        return original(x, y, **opcoes)

    monkeypatch.setattr(modulo_pca, "elipse_de_hotelling", espia)
    par = par_de_prova()
    figura_de_prova(par=par)

    assert recebidos, (
        "o painel não chamou `pca.elipse_de_hotelling`: a região desenhada não é a que a "
        "simulação de cobertura valida"
    )
    for lado in (par.antes, par.depois):
        estudo = lado.escores.mascara(TipoDeAmostra.AMOSTRA)
        assert int(estudo.sum()) < lado.escores.n, "âncora: a fixtura não tem QC"
        assert any(
            np.array_equal(x, lado.escores.x[estudo]) and np.array_equal(y, lado.escores.y[estudo])
            for x, y in recebidos
        ), f"nenhuma chamada recebeu as {int(estudo.sum())} amostras de estudo do «{lado.rotulo}»"
    estudos = {
        int(lado.escores.mascara(TipoDeAmostra.AMOSTRA).sum()) for lado in (par.antes, par.depois)
    }
    assert {x.size for x, _ in recebidos} <= estudos, (
        "uma região foi calculada sobre outra população: com os QCs dentro, ela cresce para os "
        "conter, e a pergunta «o QC está na nuvem?» passa a responder-se sozinha"
    )


def test_com_menos_de_tres_amostras_de_estudo_a_regiao_falta_e_o_EIXO_diz_por_que() -> None:  # noqa: N802
    """`ElipseAusente` vira texto CURTO dentro do eixo daquele lado, e o rodapé NÃO muda.

    **O rodapé não leva a frase, e é medição e não gosto:** no tamanho da janela a 1280x860
    o rodapé mais longo deste painel já ocupa 8 das 9 linhas que cabem no teto — ver o cabeçalho de
    `plots/correcao.py`. A metade final afirma que a ausência não tocou nele.

    **O par causal está no mesmo eixo de leitura:** o «Before» tem amostras de estudo que cheguem,
    a região e nenhum texto de ausência; o «After» tem duas, nenhuma região e o texto.
    """
    poucas = _tipos_com_estudo(MINIMO_PARA_ELIPSE - 1)
    assert poucas.count("sample") == MINIMO_PARA_ELIPSE - 1, "âncora: a fixtura não tem n = p"
    par = par_de_prova(
        depois=lado_de_prova(
            ROTULO_DEPOIS, "corrigida", deslocamento=1.5, dispersao=0.23, tipos=poucas
        )
    )
    assert isinstance(elipse_das_amostras(par.depois.escores), ElipseAusente)
    figura = figura_de_prova(par=par)
    _barras, antes, depois = figura.axes

    esperada = f"menos de {MINIMO_PARA_ELIPSE} amostras de estudo"
    textos_depois = [texto_corrido(t.get_text()) for t in depois.texts]
    assert any(t.startswith("sem região") and esperada in t for t in textos_depois), textos_depois
    textos_antes = [texto_corrido(t.get_text()) for t in antes.texts]
    assert not any(t.startswith("sem região") for t in textos_antes), textos_antes

    # Presença antes de ausência, na mesma função.
    rotulos_antes = {s.rotulo for s in series(antes)}
    rotulos_depois = {s.rotulo for s in series(depois)}
    assert rotulo_da_regiao(ROTULO_ANTES) in rotulos_antes, rotulos_antes
    assert rotulos_depois, "o painel sem região ficou sem série nenhuma"
    assert rotulo_da_regiao(ROTULO_DEPOIS) not in rotulos_depois, rotulos_depois

    assert rodape_de(figura) == rodape_de(figura_de_prova()), (
        "a ausência da região mexeu no rodapé, que no tamanho da janela a 1280x860 não comporta "
        "mais uma frase"
    )


def test_amostras_de_estudo_num_PONTO_SO_dao_o_outro_motivo_encurtado() -> None:  # noqa: N802
    """O outro ramo de `ElipseAusente` — covariância singular com n >= 3 —, e é o que dispara por
    ausência da primeira condição — por isso se começa pelo `else`.

    O texto é a cabeça do motivo que `pca.elipse_de_hotelling` deu, e não «poucas amostras»: com
    vinte amostras de estudo no mesmo ponto, dizer que faltam amostras seria a causa errada.
    """
    tipos = tipos_de_prova()
    estudo = [tipo == "sample" for tipo in tipos]
    x = [1.0 if e else -3.0 + 0.5 * i for i, e in enumerate(estudo)]
    y = [1.0 if e else math.cos(float(i)) for i, e in enumerate(estudo)]
    lado = lado_da_correcao(
        rotulo=ROTULO_DEPOIS,
        matriz="corrigida",
        pc1=x,
        pc2=y,
        tipos=tipos,
        variancia_explicada=[0.34, 0.098],
        qc_dispersion_ratio=0.23,
        n_features_usadas=6035,
        n_features_totais=6035,
        digest_das_colunas_retidas="a" * 16,
    )
    regiao = elipse_das_amostras(lado.escores)
    assert isinstance(regiao, ElipseAusente), "âncora: as amostras num ponto só deram região"
    assert int(lado.escores.mascara(TipoDeAmostra.AMOSTRA).sum()) >= MINIMO_PARA_ELIPSE, (
        "âncora: a fixtura caiu no outro motivo, e este teste mediria o mesmo ramo do anterior"
    )

    textos = [
        texto_corrido(t.get_text())
        for t in figura_de_prova(par=par_de_prova(depois=lado)).axes[2].texts
    ]
    cabeca = regiao.motivo.split(":", 1)[0]
    assert any(t.startswith("sem região") and cabeca in t for t in textos), textos
    assert not any("menos de" in t for t in textos), (
        f"o motivo singular saiu como «poucas amostras», que é a causa errada: {textos}"
    )


def test_a_legenda_DECLARA_a_regiao_com_a_marca_da_linha_e_so_quando_ha_uma() -> None:  # noqa: N802
    """O tipo da região dito na figura, que é a legenda: a entrada existe quando algum lado tem
    região, e a amostra dela é a linha desenhada — cor e traço conferidos no ARTISTA.

    **O par causal:** sem região em lado nenhum, a entrada sai — uma entrada que prometesse uma
    linha ausente ensinaria a procurar o que não está lá, que é a regra de `_provas_da_legenda`.
    """
    figura = figura_de_prova()
    legenda = figura.axes[2].get_legend()
    assert legenda is not None, "a figura perdeu a legenda"
    textos = [texto_corrido(t.get_text()) for t in legenda.get_texts()]
    assert texto_corrido(ROTULO_DA_REGIAO) in textos, textos

    mao = legenda.legend_handles[textos.index(texto_corrido(ROTULO_DA_REGIAO))]
    linhas = [
        linha
        for linha in figura.axes[1].lines
        if linha.get_label() == rotulo_da_regiao(ROTULO_ANTES)
    ]
    assert len(linhas) == 1, f"o «Before» tem {len(linhas)} linha(s) de região"
    assert to_hex(mao.get_color()) == to_hex(linhas[0].get_color()), mao.get_color()
    assert mao.get_linestyle() == linhas[0].get_linestyle(), mao.get_linestyle()

    poucas = _tipos_com_estudo(MINIMO_PARA_ELIPSE - 1)
    sem = figura_de_prova(
        par=par_de_prova(
            antes=lado_de_prova(ROTULO_ANTES, "ingerida", tipos=poucas),
            depois=lado_de_prova(ROTULO_DEPOIS, "corrigida", deslocamento=1.5, tipos=poucas),
        )
    )
    legenda_sem = sem.axes[2].get_legend()
    assert legenda_sem is not None
    textos_sem = [texto_corrido(t.get_text()) for t in legenda_sem.get_texts()]
    assert textos_sem, "âncora: a legenda sem região ficou vazia, e a ausência seria vácua"
    assert texto_corrido(ROTULO_DA_REGIAO) not in textos_sem, textos_sem


def test_a_cor_da_regiao_NAO_e_a_de_nenhuma_serie_e_e_a_que_o_artista_leva() -> None:  # noqa: N802
    """A região não é uma quarta série, e a cor dela não pode ser lida como a de uma das três.

    **A âncora é a planta**: sem `color=` explícito, o ciclo do `matplotlib` daria à linha uma cor
    da paleta das séries — medido, `#1f77b4`, o azul da validação.
    """
    paleta = {to_hex(cor) for cor in CORES_DA_PCA.values()}
    assert to_hex(COR_DA_REGIAO) not in paleta, (COR_DA_REGIAO, sorted(paleta))

    figura = figura_de_prova()
    for eixo, lado in ((figura.axes[1], ROTULO_ANTES), (figura.axes[2], ROTULO_DEPOIS)):
        linhas = [linha for linha in eixo.lines if linha.get_label() == rotulo_da_regiao(lado)]
        assert len(linhas) == 1, f"o painel «{lado}» tem {len(linhas)} linha(s) de região"
        assert to_hex(linhas[0].get_color()) == to_hex(COR_DA_REGIAO), (lado, linhas[0].get_color())

    from refino.plots.base import nova_figura

    eixo = nova_figura().add_subplot(1, 1, 1)
    for cor in CORES_DA_PCA.values():
        eixo.plot([0.0, 1.0], [0.0, 1.0], color=cor)
    (sem_cor,) = eixo.plot([0.0, 1.0], [1.0, 0.0], linestyle="--")
    assert to_hex(sem_cor.get_color()) in paleta, (
        f"âncora: sem cor explícita a linha sai {to_hex(sem_cor.get_color())}, fora da paleta das "
        "séries — a planta não mostra a colisão que a cor explícita evita"
    )


# ---------------------------------------------------------------------------
# 9. O intervalo por barra, por escolha
# ---------------------------------------------------------------------------
#
# O intervalo só existe quando a medição o trouxe — a caixa desmarcada por padrão. Os
# números abaixo são de forma plausível sobre os escores do PNG do R, e não medição de estudo
# nenhum: o que se afirma é o desenho e a declaração.


def _intervalos_de_prova(**troca: Any) -> dict[str, Any]:
    """Um intervalo por corretor, em fração, em volta do escore de `SCORES_DO_R`."""
    campos: dict[str, dict[str, Any]] = {
        nome: {
            "inferior": valor - 0.05,
            "superior": valor + 0.08,
            "nivel": 0.95,
            "n_qc": 7,
            "metodo": "jackknife-t com reajuste",
            "motivo": None,
        }
        for nome, valor in SCORES_DO_R.items()
        if valor is not None
    }
    for nome, mudanca in troca.items():
        campos[nome] = {**campos[nome], **mudanca}
    return {nome: intervalo_do_fio(**campo) for nome, campo in campos.items()}


def test_sem_o_intervalo_a_figura_e_a_de_hoje_e_o_rodape_tambem() -> None:
    """**Quem não pede não vê nada a mais**: a fila sem o campo, e com ele `None`, desenham as
    mesmas séries, os mesmos limites e o mesmo rodapé. A comparação dos bytes do PNG com a árvore
    anterior a esta decisão está no registro dela — ela não cabe num teste, porque a árvore anterior
    não está no processo."""
    sem_campo = figura_de_prova(fila=fila_de_prova())
    com_none = figura_de_prova(fila=fila_de_prova(intervalos=None, margem=0.0123))
    for figura in (sem_campo, com_none):
        assert not any(
            s.rotulo.startswith(ROTULO_DO_INTERVALO) for s in series(eixo_das_barras(figura))
        )
    assert [s.rotulo for s in series(eixo_das_barras(sem_campo))] == [
        s.rotulo for s in series(eixo_das_barras(com_none))
    ]
    assert eixo_das_barras(sem_campo).get_ylim() == eixo_das_barras(com_none).get_ylim()
    assert rodape_de(sem_campo) == rodape_de(com_none)
    assert "jackknife" not in rodape_de(sem_campo) and "margem" not in rodape_de(sem_campo)


def test_com_o_intervalo_cada_barra_leva_UMA_serie_do_inferior_ao_superior() -> None:  # noqa: N802
    """Uma série por intervalo, com o rótulo do corretor, e os vértices de `y` são os dois limites
    em percentagem — a mesma unidade da barra. A âncora é que há três: sem ela, uma figura sem
    intervalo nenhum passaria por todas as asserções do laço."""
    intervalos = _intervalos_de_prova()
    fila = fila_de_prova(intervalos=intervalos, margem=0.1)
    assert isinstance(fila, FilaDeBarras)
    eixo = eixo_das_barras(figura_de_prova(fila=fila))
    desenhadas = {s.rotulo: s for s in series(eixo) if s.rotulo.startswith(ROTULO_DO_INTERVALO)}
    assert len(desenhadas) == 3, sorted(desenhadas)
    for posicao, barra in enumerate(fila.barras):
        intervalo = intervalos[barra.corretor]
        assert isinstance(intervalo, IntervaloDeBarra)
        serie = desenhadas[f"{ROTULO_DO_INTERVALO} {barra.corretor}"]
        assert float(np.min(serie.y)) == pytest.approx(intervalo.inferior * 100.0)
        assert float(np.max(serie.y)) == pytest.approx(intervalo.superior * 100.0)
        assert float(np.mean(serie.x)) == pytest.approx(float(posicao))
    # **O topo do eixo dá folga ao topo do INTERVALO, e não só ao da barra:** o valor escrito sobe
    # para cima do traço, e tem de caber dentro do eixo como cabia sem ele. A primeira escrita
    # afirmava só `>=` o topo do intervalo, e a folga de 12% sobre a barra mais alta a satisfazia
    # sozinha — a planta que tirava os intervalos da conta passou verde.
    from refino.plots.correcao import FOLGA_DO_ROTULO

    topo = max(i.superior for i in intervalos.values() if isinstance(i, IntervaloDeBarra))
    assert eixo.get_ylim()[1] == pytest.approx(topo * 100.0 * (1.0 + FOLGA_DO_ROTULO))


def test_o_valor_escrito_fica_ACIMA_do_intervalo_e_nao_por_cima_dele() -> None:  # noqa: N802
    """O número sobre a barra do `none` e a do vencedor é ancorado no topo do intervalo quando ele
    passa da barra: no topo da barra, o texto cairia em cima da linha."""
    intervalos = _intervalos_de_prova()
    eixo = eixo_das_barras(figura_de_prova(fila=fila_de_prova(intervalos=intervalos, margem=0.1)))
    anotacoes = {a.get_text(): a for a in eixo.texts if hasattr(a, "xy")}
    assert anotacoes, "âncora: a fila não escreveu valor nenhum"
    for texto, anotacao in anotacoes.items():
        corretor = next(c for c, v in SCORES_DO_R.items() if v and percentagem_do_r(v) == texto)
        intervalo = intervalos[corretor]
        assert isinstance(intervalo, IntervaloDeBarra)
        assert anotacao.xy[1] == pytest.approx(intervalo.superior * 100.0), (texto, anotacao.xy)


def test_o_rodape_diz_o_que_o_intervalo_mede_e_o_nivel_e_NOMINAL() -> None:  # noqa: N802
    """Sensibilidade a QUAIS QCs foram injetados — jackknife-t sobre eles, o nível, o número de QCs
    —, e não às features; e a cobertura não foi medida, então o nível é dito nominal."""
    texto = rodape_de(figura_de_prova(fila=fila_de_prova(intervalos=_intervalos_de_prova())))
    for trecho in ("jackknife-t com reajuste", "sobre 7 QCs", "95% nominal", "QCs injetados"):
        assert trecho in texto, (trecho, texto)
    assert "não às features" in texto, texto


def test_a_margem_vai_com_a_regua_da_MEDICAO_e_nao_com_a_do_palpite() -> None:  # noqa: N802
    """**Rótulo igual, quantidade diferente, e o teste que a planta fez reprovar.** A margem entre
    os dois melhores da medição completa lê-se contra o limiar de abstenção da medição —
    `ABSTENTION_MARGIN`, ruído da semente sobre a população inteira —, e não contra o 16,7% do
    Palpite, que é ruído do sorteio de 200 features. Os dois números estão à mão, e só um é desta
    população.

    **E a margem é RELATIVA, e a frase di-lo.** `Label.margin` é `(segundo - melhor) / |melhor|`, e
    ao lado de barras em % de RSD absoluto um «2,56%» sem a palavra lê-se como pontos percentuais
    entre as barras — rótulo igual e quantidade diferente outra vez."""
    texto = rodape_de(
        figura_de_prova(fila=fila_de_prova(intervalos=_intervalos_de_prova(), margem=0.0256))
    )
    assert "Margem relativa dos dois melhores: 2,56% do melhor" in texto, texto
    assert f"abaixo de {decimal(ABSTENTION_MARGIN * 100.0)}%" in texto, texto
    proibidos = (
        decimal(LIMIAR_DE_EMPATE_DO_PALPITE * 100.0),
        decimal(LIMIAR_DE_EMPATE_DO_PALPITE),
        "16,7",
        "0,167",
    )
    assert not any(p in texto for p in proibidos), texto
    assert ABSTENTION_MARGIN != LIMIAR_DE_EMPATE_DO_PALPITE, "âncora: as duas réguas coincidem"


def test_sem_segundo_colocado_a_margem_e_INDEFINIDA_e_nao_zero() -> None:  # noqa: N802
    texto = rodape_de(figura_de_prova(fila=fila_de_prova(intervalos=_intervalos_de_prova())))
    assert "Margem relativa dos dois melhores: indefinida" in texto, texto


def test_a_barra_SEM_intervalo_nao_desenha_nada_e_o_rodape_diz_o_motivo() -> None:  # noqa: N802
    """O corretor sem intervalo não ganha um traço de largura zero, que se leria como
    certeza; o motivo vai no rodapé, uma vez por motivo, com os corretores que o têm."""
    motivo = "menos de 4 QCs num lote (A=3 QCs): cada replicado tira um QC"
    intervalos = _intervalos_de_prova(SERRF={"motivo": motivo, "inferior": None, "superior": None})
    assert isinstance(intervalos["SERRF"], IntervaloAusente)
    figura = figura_de_prova(fila=fila_de_prova(intervalos=intervalos, margem=0.1))
    rotulos = {s.rotulo for s in series(eixo_das_barras(figura))}
    assert f"{ROTULO_DO_INTERVALO} SERRF" not in rotulos
    assert f"{ROTULO_DO_INTERVALO} none" in rotulos, "âncora: os outros dois continuam lá"
    texto = rodape_de(figura)
    assert f"Sem traço em SERRF: {motivo}" in texto, texto


@pytest.mark.parametrize(
    ("campo", "valor"),
    [
        ("inferior", None),
        ("superior", float("nan")),
        ("inferior", True),
        ("nivel", 1.5),
        ("superior", 0.1),
    ],
)
def test_um_intervalo_MALFORMADO_do_fio_vira_ausencia_declarada(campo: str, valor: Any) -> None:  # noqa: N802
    """Na fronteira: limite ausente, não finito, booleano, nível fora de (0, 1) ou o
    inferior acima do superior não viram traço — viram ausência, com a frase dela."""
    base = {
        "inferior": 0.5,
        "superior": 0.7,
        "nivel": 0.95,
        "n_qc": 7,
        "metodo": "jackknife-t com reajuste",
        "motivo": None,
    }
    assert isinstance(intervalo_do_fio(**base), IntervaloDeBarra), "âncora: a base é válida"
    malformado = intervalo_do_fio(**{**base, campo: valor})
    assert isinstance(malformado, IntervaloAusente), malformado
    assert malformado.motivo == INTERVALO_MALFORMADO


@pytest.mark.parametrize(
    ("largura", "altura", "ppp"),
    [(10.0, 6.0, 100.0), (8.0, 4.5, 96.0), (6.5, 3.4, 96.0), (12.0, 7.0, 150.0)],
)
@pytest.mark.parametrize("sem_traco", [False, True])
def test_o_rodape_COM_o_intervalo_cabe_nos_tamanhos_EXTREMOS(  # noqa: N802
    largura: float, altura: float, ppp: float, sem_traco: bool
) -> None:
    """Os quatro tamanhos, com o rodapé mais longo que esta figura produz COM o intervalo
    pedido: os espaços divergentes, e o traço nas três barras ou em nenhuma — com o motivo do núcleo
    para a guarda por lote, gerado e não redigitado. Barras com e sem traço na mesma figura NÃO
    cabem a 6,5 x 3,4, e isso está medido e declarado em `_frase_do_intervalo`."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure as FiguraNua

    from refino.correction.none import NoneCorrector
    from refino.metrics.crossval import (
        MIN_QC_PER_BATCH_FOR_JACKKNIFE,
        leave_one_qc_out_rsd_with_jackknife,
    )
    from refino.plots.recorte import ENTRELINHA, MARGEM, RESERVA_MAXIMA, TAMANHO_DA_FONTE
    from tests.unit.test_intervalo_das_barras import matriz_com_deriva

    par = par_de_prova(
        antes=lado_de_prova(ROTULO_ANTES, "ingerida", digest="c" * 64),
        depois=lado_de_prova(ROTULO_DEPOIS, "corrigida", usadas=5321, digest="d" * 64),
    )
    assert par.espacos_divergem
    _vetor, guarda = leave_one_qc_out_rsd_with_jackknife(
        matriz_com_deriva(n_qc=7, lotes_dos_qcs=("0",) * 4 + ("1",) * 3), NoneCorrector()
    )
    assert guarda.reason is not None and str(MIN_QC_PER_BATCH_FOR_JACKKNIFE) in guarda.reason
    troca = {nome: {"motivo": guarda.reason} for nome in SCORES_DO_R} if sem_traco else {}
    fila = fila_de_prova(intervalos=_intervalos_de_prova(**troca), margem=0.0256)
    texto = recorte_do_painel_da_correcao(par=par, fila=fila).texto()
    assert ("Sem traço:" in texto) is sem_traco and ("Traço:" in texto) is not sem_traco

    figura = FiguraNua(figsize=(largura, altura), dpi=ppp)
    FigureCanvasAgg(figura)
    linhas = linhas_do_rodape(figura, texto)
    reservado = len(linhas) * TAMANHO_DA_FONTE * ENTRELINHA / (altura * 72.0) + 2.0 * MARGEM
    assert reservado <= RESERVA_MAXIMA, (
        f"a {largura}x{altura}@{ppp} o rodapé com o intervalo quebra em {len(linhas)} linhas e "
        f"precisaria de {reservado:.1%} da altura, acima do teto de {RESERVA_MAXIMA:.0%}"
    )


@pytest.mark.parametrize(
    ("largura", "altura", "ppp"),
    [(10.0, 6.0, 100.0), (8.0, 4.5, 96.0), (6.5, 3.4, 96.0), (12.0, 7.0, 150.0)],
)
def test_o_caso_MISTO_do_lote_de_quatro_so_NAO_cabe_a_6_5_x_3_4(  # noqa: N802
    largura: float, altura: float, ppp: float
) -> None:
    """**A limitação declarada, afirmada, e não só escrita**.

    Num lote de exatamente quatro QCs o LOESS fica sem traço — o par de dentro do lote treinaria
    com dois, e ele não ajusta com menos de três — e o `none` e o SERRF ficam com ele: barras com e
    sem traço na mesma figura. Esse rodapé, com o motivo que o núcleo gera, NÃO cabe no tamanho da
    janela a 1280 x 860 (6,5 x 3,4 pol. a 96 ppp) e cabe nos outros três; medido com a DejaVu.
    Se um dia passar a caber, ou deixar de caber noutro
    tamanho, a declaração de `_frase_do_intervalo` e do registro está errada, e este teste di-lo.
    """
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure as FiguraNua

    from refino.correction.loess import LoessCorrector
    from refino.metrics.crossval import leave_one_qc_out_rsd_with_jackknife
    from refino.plots.recorte import ENTRELINHA, MARGEM, RESERVA_MAXIMA, TAMANHO_DA_FONTE
    from tests.unit.test_intervalo_das_barras import matriz_com_deriva

    par = par_de_prova(
        antes=lado_de_prova(ROTULO_ANTES, "ingerida", digest="c" * 64),
        depois=lado_de_prova(ROTULO_DEPOIS, "corrigida", usadas=5321, digest="d" * 64),
    )
    _vetor, do_loess = leave_one_qc_out_rsd_with_jackknife(
        matriz_com_deriva(n_qc=8, n_amostras=40, lotes_dos_qcs=("0",) * 4 + ("1",) * 4),
        LoessCorrector(n_jobs=1),
    )
    assert do_loess.reason is not None, "âncora: o LOESS com lotes de quatro tem intervalo"
    fila = fila_de_prova(
        intervalos=_intervalos_de_prova(batch_loess={"motivo": do_loess.reason}), margem=0.0256
    )
    texto = recorte_do_painel_da_correcao(par=par, fila=fila).texto()
    assert "Traço:" in texto and "Sem traço em batch_loess:" in texto, texto

    figura = FiguraNua(figsize=(largura, altura), dpi=ppp)
    FigureCanvasAgg(figura)
    linhas = linhas_do_rodape(figura, texto)
    reservado = len(linhas) * TAMANHO_DA_FONTE * ENTRELINHA / (altura * 72.0) + 2.0 * MARGEM
    na_janela = (largura, altura) == (6.5, 3.4)
    assert (reservado <= RESERVA_MAXIMA) is not na_janela, (
        f"a {largura}x{altura}@{ppp} o rodapé misto quebra em {len(linhas)} linhas, "
        f"{reservado:.1%} da altura contra o teto de {RESERVA_MAXIMA:.0%}: a limitação declarada "
        "mudou"
    )


# ---------------------------------------------------------------------------
# 10. O corretor RECUSADO por QCs no lote
# ---------------------------------------------------------------------------
#
# Quando um lote tem menos QCs do que o corretor precisa para o ajuste fora da amostra, a barra
# desse corretor diz o motivo em vez de um número. O corpo traz o corretor com `null` em `scores`
# e o motivo em `recusas`; o motivo é o que a régua gera (`metrics.crossval.BatchQcShortfallError`),
# e aqui ele vem do núcleo, e não redigitado.


def _motivo_do_nucleo() -> str:
    """O motivo que a régua dá ao LOESS com um lote de três QCs — gerado, e não redigitado."""
    from refino.correction.loess import LoessCorrector
    from refino.metrics.crossval import BatchQcShortfallError, leave_one_qc_out_rsd
    from tests.unit.test_intervalo_das_barras import matriz_com_deriva

    matriz = matriz_com_deriva(n_qc=7, n_amostras=35, lotes_dos_qcs=("0",) * 3 + ("1",) * 4)
    with pytest.raises(BatchQcShortfallError) as recusa:
        leave_one_qc_out_rsd(matriz, LoessCorrector(n_jobs=1))
    return recusa.value.reason


def test_o_corretor_RECUSADO_nao_vira_barra_e_o_titulo_e_o_rodape_dizem_o_motivo() -> None:  # noqa: N802
    """Sem barra — nem de altura zero —, fora de `nao_medidos`, no título com a causa curta, e no
    rodapé com o motivo inteiro, como o do intervalo («Sem traço em …»)."""
    motivo = _motivo_do_nucleo()
    fila = fila_de_prova(
        scores={"none": 1.447, "batch_loess": None, "SERRF": 0.693},
        recusas={"batch_loess": motivo},
    )
    assert isinstance(fila, FilaDeBarras)
    assert [b.corretor for b in fila.barras] == ["none", "SERRF"]
    assert fila.recusados == {"batch_loess": motivo} and fila.nao_medidos == ()
    figura = figura_de_prova(fila=fila)
    titulo = texto_corrido(eixo_das_barras(figura).get_title())
    assert "QCs insuficientes no lote: batch_loess" in titulo, titulo
    assert "RSD indefinido" not in titulo, titulo
    assert f"Sem barra em batch_loess: {motivo}" in rodape_de(figura), rodape_de(figura)


def test_RECUSADO_e_INDEFINIDO_sao_ausencias_DIFERENTES_na_mesma_figura() -> None:  # noqa: N802
    """**Duas ausências distintas.** Os dois ficam sem barra; o indefinido é o que a medição deu, o
    recusado é o que ela não fez — e o título e o rodapé dizem cada um com a sua frase."""
    motivo = _motivo_do_nucleo()
    fila = fila_de_prova(
        scores={"none": 1.447, "batch_loess": None, "SERRF": None},
        vencedor="none",
        recusas={"batch_loess": motivo},
    )
    assert isinstance(fila, FilaDeBarras)
    assert fila.nao_medidos == ("SERRF",) and set(fila.recusados) == {"batch_loess"}
    figura = figura_de_prova(fila=fila)
    titulo = texto_corrido(eixo_das_barras(figura).get_title())
    assert "RSD indefinido: SERRF" in titulo and "no lote: batch_loess" in titulo, titulo
    rodape = rodape_de(figura)
    assert "Sem barra em batch_loess:" in rodape and "Sem barra em SERRF" not in rodape, rodape


def test_o_motivo_VENCE_o_numero_e_recusas_malformadas_nao_inventam_recusa() -> None:  # noqa: N802
    """Na fronteira: o servidor nunca manda os dois, e se mandar, o motivo vence — é ele a
    dizer que não houve medição, como no intervalo. E o que não é texto não vira recusa."""
    com_os_dois = fila_de_prova(recusas={"batch_loess": "menos de 4 QCs num lote (0=3 QCs)"})
    assert isinstance(com_os_dois, FilaDeBarras)
    assert "batch_loess" not in [b.corretor for b in com_os_dois.barras]
    assert set(com_os_dois.recusados) == {"batch_loess"}
    for malformada in (None, [], "batch_loess", {"batch_loess": None}, {"batch_loess": "  "}):
        fila = fila_de_prova(recusas=malformada)
        assert isinstance(fila, FilaDeBarras)
        assert fila.recusados == {} and len(fila.barras) == 3, malformada


def test_sem_recusa_a_figura_e_o_rodape_sao_os_de_antes() -> None:
    """Quem não tem lote curto não vê nada a mais: `recusas` vazio, ausente ou `None` dá o mesmo
    rodapé e o mesmo título."""
    sem = figura_de_prova(fila=fila_de_prova())
    vazio = figura_de_prova(fila=fila_de_prova(recusas={}))
    assert rodape_de(sem) == rodape_de(vazio) and "Sem barra" not in rodape_de(sem)
    assert eixo_das_barras(sem).get_title() == eixo_das_barras(vazio).get_title()


def test_so_recusados_e_indefinidos_sem_barra_nenhuma_dizem_os_dois() -> None:
    """Nenhum corretor com número: a fila não se desenha, e o texto no lugar dela já não diz
    «indefinido em todos os corretores» quando um deles não foi medido."""
    motivo = _motivo_do_nucleo()
    fila = fila_de_prova(
        scores={"none": None, "batch_loess": None, "SERRF": None},
        recusas={"batch_loess": motivo},
    )
    assert isinstance(fila, FilaAusente)
    assert fila.motivo != SEM_RANKING and motivo in fila.motivo, fila.motivo
    assert "indefinido em SERRF, none" in fila.motivo, fila.motivo
    sem_recusa = fila_de_prova(scores={"none": None, "batch_loess": None, "SERRF": None})
    assert isinstance(sem_recusa, FilaAusente) and sem_recusa.motivo == SEM_RANKING


@pytest.mark.parametrize(
    ("largura", "altura", "ppp"),
    [(10.0, 6.0, 100.0), (8.0, 4.5, 96.0), (6.5, 3.4, 96.0), (12.0, 7.0, 150.0)],
)
@pytest.mark.parametrize("com_intervalo", [False, True])
def test_o_rodape_com_o_RECUSADO_cabe_sem_o_intervalo_e_com_ele_NAO_a_6_5_x_3_4(  # noqa: N802
    largura: float, altura: float, ppp: float, com_intervalo: bool
) -> None:
    """**A geometria medida, afirmada e não só escrita.** Os quatro tamanhos, nos espaços
    divergentes — o rodapé mais longo —, com o motivo da recusa gerado pelo núcleo.

    Sem o intervalo, a população passa à forma curta e o motivo cabe nos quatro: 808 caracteres,
    9 linhas a 6,5 x 3,4 com a DejaVu, contra 811 e 9 antes da recusa. **Com o intervalo pedido e um
    lote de três, NÃO cabe na janela a 1280 x 860**: o `none` e o SERRF ficam sem traço pelo piso
    do intervalo e o LOESS sem barra, e são 1.000 caracteres e 11 linhas — a mesma classe do caso
    misto
    do intervalo, com o mesmo destino (`recorte.reajusta` na janela, o `.txt` na exportação). Se
    um dia passar a caber, ou deixar de caber noutro tamanho, a declaração está errada e este teste
    di-lo.
    """
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure as FiguraNua

    from refino.correction.none import NoneCorrector
    from refino.metrics.crossval import leave_one_qc_out_rsd_with_jackknife
    from refino.plots.recorte import ENTRELINHA, MARGEM, RESERVA_MAXIMA, TAMANHO_DA_FONTE
    from tests.unit.test_intervalo_das_barras import matriz_com_deriva

    par = par_de_prova(
        antes=lado_de_prova(ROTULO_ANTES, "ingerida", digest="c" * 64),
        depois=lado_de_prova(ROTULO_DEPOIS, "corrigida", usadas=5321, digest="d" * 64),
    )
    assert par.espacos_divergem
    motivo = _motivo_do_nucleo()
    scores = {"none": 1.447, "batch_loess": None, "SERRF": 0.693}
    if com_intervalo:
        _vetor, piso = leave_one_qc_out_rsd_with_jackknife(
            matriz_com_deriva(n_qc=7, n_amostras=35, lotes_dos_qcs=("0",) * 3 + ("1",) * 4),
            NoneCorrector(),
        )
        assert piso.reason is not None, "âncora: o lote de três tira o traço do `none`"
        fila = fila_de_prova(
            scores=scores,
            recusas={"batch_loess": motivo},
            margem=0.1,
            intervalos=_intervalos_de_prova(
                none={"motivo": piso.reason},
                SERRF={"motivo": piso.reason},
                batch_loess={"motivo": motivo},
            ),
        )
    else:
        fila = fila_de_prova(scores=scores, recusas={"batch_loess": motivo})
    texto = recorte_do_painel_da_correcao(par=par, fila=fila).texto()
    assert f"Sem barra em batch_loess: {motivo}" in texto, texto

    figura = FiguraNua(figsize=(largura, altura), dpi=ppp)
    FigureCanvasAgg(figura)
    linhas = linhas_do_rodape(figura, texto)
    reservado = len(linhas) * TAMANHO_DA_FONTE * ENTRELINHA / (altura * 72.0) + 2.0 * MARGEM
    nao_cabe = com_intervalo and (largura, altura) == (6.5, 3.4)
    assert (reservado <= RESERVA_MAXIMA) is not nao_cabe, (
        f"a {largura}x{altura}@{ppp} o rodapé com o recusado quebra em {len(linhas)} linhas, "
        f"{reservado:.1%} da altura contra o teto de {RESERVA_MAXIMA:.0%}: a geometria declarada "
        "mudou"
    )
