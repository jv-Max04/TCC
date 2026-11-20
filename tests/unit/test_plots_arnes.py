"""O arnês das figuras, o auxiliar do recorte, e os quatro regimes em que `plots/` nasce.

## O que este módulo prova, e por que a ordem importa

`matplotlib` tinha **zero sítios de import** em `src/` antes desta unidade. Não havia código de
construção de figura neste repositório, nem teste que afirmasse sobre dado de artista. Então o que
aqui se escreve não é «o teste da primeira figura» — é o **padrão** que as seis unidades de figura
seguintes copiam, e um padrão errado custa seis vezes.

Quatro blocos, e os dois últimos são portões e não testes:

| Bloco | O que prova |
|---|---|
| o arnês | as quatro classes acusam, cada uma na sua direção, e poupam a figura limpa |
| o auxiliar do recorte | R3 e R5b saem na imagem, quebram, e recusam em vez de sair pela borda |
| direção de import | `plots/` ignora Qt, HTTP, API e `pyplot` — sintática e **transitivamente** |
| a bijeção de R1 | os construtores e a partição `construída` do `[G-1]` são o mesmo conjunto |

## As quatro direções do par causal, e por que são quatro

Série removida, eixos trocados e sinal invertido são as três plantas que ocorrem a quem escreve um
teste de figura, e as três são mutações de **dado**. Um `set_ylim` que corta a série inteira não é:
`get_ydata` continua a devolver o array certo, e a tela fica em branco. As três primeiras passariam.
Daí a quarta, e daí a classe (c) do arnês.

Cada uma corre no seu próprio teste, sobre a **mesma** figura de prova, e o teste do caminho feliz é
a outra ponta do par: ele é a execução com a planta removida.

## O que este módulo NÃO alcança, e vai declarado

**A varredura de direção de import percorre `src/refino/plots/` e não percorre `tests/`.** É
deliberado, e a razão é que a proibição não vale do outro lado: os testes do widget vão
importar `PySide6` **e** `refino.plots` no mesmo módulo, legitimamente, e um portão que os acusasse
seria afrouxado no primeiro dia — que é a morte de um portão. O precedente da
casa concorda: `tests/test_desktop_janela.py` varre só `src`.

**E ele não afirma nada sobre a APARÊNCIA das figuras.** As quatro classes de asserção leem o objeto
`Figure`; nenhuma delas sabe se a figura é legível. A camada 3 — `pytest-mpl` — compara imagem, e o
que ela prova é que a comparação de imagem **consegue** acusar um eixo trocado, não que a figura
esteja bonita. A leitura humana é outra etapa.
"""

from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Final

import matplotlib
import numpy as np
import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure

from refino.api.esquemas import COEFICIENTE_DE_TUKEY
from refino.plots import (
    agrupamento,
    anova,
    base,
    catalogo,
    classificacao,
    composicao,
    correcao,
    correlacao,
    curva_do_loess,
    deriva,
    dispersao,
    distribuicao,
    pca,
    plsda,
    plsda_multiclasse,
    recorte,
    univariada,
    validacao,
    volcano,
)
from refino.plots import series as plots_series
from refino.plots.base import GID_DO_RODAPE, MatrizVaziaError, exige_populacao, nova_figura
from refino.plots.catalogo import (
    ARTIGOS_DO_CONSTRUTOR,
    CONSTRUTORES,
    PREFIXO_DO_CONSTRUTOR,
    ConstrutorDeFigura,
    EstadoDoCatalogo,
    Vinculo,
    familias_do_seletor,
    rotulo_do_construtor,
)
from refino.plots.recorte import (
    ENTRELINHA,
    MARGEM,
    PONTOS_POR_POLEGADA,
    TAMANHO_DA_FONTE,
    Recorte,
    RecorteInvalidoError,
    aplica,
    colunas_da_figura,
    frase_contada,
    largura_util_em_pontos,
    marcas_que_cabem,
    quebra,
    reajusta,
    retira,
)
from refino.reporting.model import ITENS_OBRIGATORIOS
from tests.arnes_de_figura import (
    PlantaQueNaoSeparaError,
    confere_conteudo,
    confere_numeros,
    confere_rotulos,
    confere_series,
    confere_vista,
    numeros_declarados,
    planta_conteudo_trocado,
    planta_corte_de_eixo,
    planta_eixos_trocados,
    planta_serie_removida,
    planta_sinal_invertido,
    sem_a_matriz,
    series,
    vista,
)
from tests.conftest import REPO

PACOTE = REPO / "src" / "refino" / "plots"
CATALOGO_MD = REPO / "docs" / "metodos-metaboanalyst.md"

# **A população é a do `pos_hilic`, e é literal aqui de propósito.** Ela é a fonte independente da
# classe (d): o construtor recebe-a por argumento, e o teste deriva dela o texto esperado por um
# caminho que não passa por `recorte.milhar`. Se as duas contas divergirem, é isso que a classe (d)
# existe para acusar.
POPULACAO: Final = 6035


# ---------------------------------------------------------------------------
# A figura de prova
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Prova:
    """A figura mínima e o dado de que ela saiu, para o teste conferir sem reconstruir."""

    figura: Figure
    x: np.ndarray
    linha: np.ndarray
    pontos: np.ndarray


def figura_de_prova(*, disponiveis: int = POPULACAO) -> Prova:
    """Duas séries sobre sete pontos, e cada escolha aqui serve uma âncora.

    **Duas séries**, porque a planta da série removida precisa de mais de uma para a reprovação ser
    atribuível. **`x` diferente de `y`**, porque sobre a diagonal a troca de eixos é a identidade.
    **`y` com valores não nulos e com extensão**, porque sobre zeros a inversão de sinal é a
    identidade e sobre série constante o corte de eixo pode não excluir nada.

    Sete pontos porque é o número de QCs por corrida nos quatro arquivos deste estudo — observação,
    não invariante. Aqui ele é só um tamanho conveniente.
    """
    x = np.arange(1.0, 8.0)
    linha = np.array([2.0, 3.5, 3.0, 5.0, 4.25, 6.0, 5.5])
    pontos = np.array([1.5, 2.75, 4.0, 3.25, 5.5, 4.75, 7.0])

    figura = nova_figura()
    ax = figura.add_subplot(1, 1, 1)
    ax.plot(x, linha, marker="o", label="observado")
    ax.scatter(x, pontos, label="controlo")
    ax.set_xlabel("injeção")
    ax.set_ylabel("intensidade relativa")
    ax.set_title("figura de prova do arnês")
    ax.legend()

    aplica(
        figura,
        Recorte(
            desenhadas=int(x.size),
            disponiveis=disponiveis,
            amostragem="os QCs da corrida, sem amostragem",
            matriz="corrigida por SERRF",
            metodo="sem teste estatístico",
        ),
    )
    return Prova(figura=figura, x=x, linha=linha, pontos=pontos)


def esperadas(prova: Prova) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    return {"observado": (prova.x, prova.linha), "controlo": (prova.x, prova.pontos)}


# ---------------------------------------------------------------------------
# 1. Caminho feliz — a outra ponta dos quatro pares causais
# ---------------------------------------------------------------------------


def test_a_figura_limpa_passa_nas_quatro_classes() -> None:
    """A execução com a planta REMOVIDA, e é ela que torna as quatro seguintes atribuíveis.

    Ver um portão reprovar não é o mesmo que vê-lo deixar de acusar quando não há o que acusar.
    """
    prova = figura_de_prova()
    ax = prova.figura.axes[0]

    assert len(series(ax)) == 2, "âncora: a figura de prova tem de ter duas séries"
    assert confere_series(ax, esperadas(prova)) == []
    assert (
        confere_rotulos(
            ax,
            x="injeção",
            y="intensidade relativa",
            titulo="figura de prova do arnês",
            legenda=("observado", "controlo"),
        )
        == []
    )
    assert confere_vista(ax) == []
    assert confere_numeros(prova.figura, numeros_esperados(prova)) == []


def numeros_esperados(prova: Prova) -> list[str]:
    """Os números que a figura tem de declarar, derivados de fonte independente do construtor.

    `desenhadas` sai do **artista** — o que está de facto na figura —, e não do `Recorte` que o
    construtor recebeu. `disponiveis` sai da constante do módulo, formatada por uma expressão que
    não chama `recorte.milhar`. Se o teste chamasse o mesmo auxiliar, os dois errariam juntos.
    """
    ax = prova.figura.axes[0]
    desenhadas = series(ax)[0].x.size
    return [str(desenhadas), f"{POPULACAO:,}".replace(",", ".")]


# ---------------------------------------------------------------------------
# 2. As quatro direções do par causal, cada uma na sua execução
# ---------------------------------------------------------------------------


def test_par_causal_serie_removida() -> None:
    prova = figura_de_prova()
    ax = prova.figura.axes[0]
    apagada = planta_serie_removida(ax)

    achados = confere_series(ax, esperadas(prova))
    assert achados, "o arnês não viu uma série desaparecer"
    assert apagada in achados[0]


def test_par_causal_eixos_trocados() -> None:
    prova = figura_de_prova()
    ax = prova.figura.axes[0]
    mutilada = planta_eixos_trocados(ax)

    achados = confere_series(ax, esperadas(prova))
    assert achados, "o arnês não viu os eixos trocados"
    assert all(mutilada in a for a in achados), achados


def test_par_causal_sinal_invertido() -> None:
    prova = figura_de_prova()
    ax = prova.figura.axes[0]
    mutilada = planta_sinal_invertido(ax)

    achados = confere_series(ax, esperadas(prova))
    assert achados, "o arnês não viu o sinal invertido"
    assert all("eixo y" in a for a in achados), achados
    assert all(mutilada in a for a in achados), achados


def test_par_causal_limite_que_corta_a_serie() -> None:
    """A quarta direção, e a razão de a classe (c) existir.

    **As três primeiras não a apanham, e este teste prova-o em vez de o afirmar:** com o limite
    plantado, `confere_series`, `confere_rotulos` e `confere_numeros` continuam todos verdes — o
    dado está lá, os rótulos estão lá, o rodapé está lá. Só `confere_vista` acusa, e a tela está em
    branco.
    """
    prova = figura_de_prova()
    ax = prova.figura.axes[0]
    limite = planta_corte_de_eixo(ax)

    assert confere_series(ax, esperadas(prova)) == [], (
        "a planta do corte de eixo mudou dado — deixaria de ser a direção que as outras três não "
        "alcançam"
    )
    assert confere_numeros(prova.figura, numeros_esperados(prova)) == []

    achados = confere_vista(ax)
    assert achados, f"o arnês não viu o limite {limite} excluir a série inteira"
    assert all("em branco" in a for a in achados), achados


def test_par_causal_do_numero_declarado() -> None:
    """Classe (d): um denominador errado no rodapé reprova, porque o teste o deriva de outra fonte.

    A planta é o próprio argumento do construtor — negativo **em processo**, que pela taxonomia de
    Assim a construção não consegue ficar inerte.
    """
    prova = figura_de_prova(disponiveis=POPULACAO // 2)

    achados = confere_numeros(prova.figura, numeros_esperados(prova))
    assert achados, "o arnês não viu o denominador errado"
    assert "6.035" in achados[0], achados


def test_par_causal_do_artista_invisivel() -> None:
    """A classe (c) alcança mais do que o limite: série presente no objeto e ausente da tela."""
    prova = figura_de_prova()
    ax = prova.figura.axes[0]
    ax.lines[0].set_alpha(0.0)

    achados = confere_vista(ax)
    assert achados, "o arnês não viu uma série transparente"
    assert "alpha" in achados[0]


# ---------------------------------------------------------------------------
# 2b. As plantas trazem a âncora, e ela dispara
# ---------------------------------------------------------------------------


def test_a_planta_dos_eixos_recusa_dado_em_que_nao_separa() -> None:
    """`x == y`: trocar os eixos é a identidade, e o negativo passaria COM a planta aplicada."""
    figura = nova_figura()
    ax = figura.add_subplot(1, 1, 1)
    diagonal = np.arange(1.0, 5.0)
    ax.plot(diagonal, diagonal, label="diagonal")

    with pytest.raises(PlantaQueNaoSeparaError, match="x == y"):
        planta_eixos_trocados(ax)


def test_a_planta_do_sinal_recusa_serie_toda_a_zeros() -> None:
    figura = nova_figura()
    ax = figura.add_subplot(1, 1, 1)
    ax.plot(np.arange(1.0, 5.0), np.zeros(4), label="resíduo")

    with pytest.raises(PlantaQueNaoSeparaError, match="todo zero"):
        planta_sinal_invertido(ax)


def test_a_planta_do_corte_recusa_serie_constante() -> None:
    figura = nova_figura()
    ax = figura.add_subplot(1, 1, 1)
    ax.plot(np.arange(1.0, 5.0), np.full(4, 3.0), label="constante")

    with pytest.raises(PlantaQueNaoSeparaError, match="constante"):
        planta_corte_de_eixo(ax)


def test_a_planta_da_serie_removida_recusa_figura_de_uma_serie_so() -> None:
    figura = nova_figura()
    ax = figura.add_subplot(1, 1, 1)
    ax.plot(np.arange(1.0, 5.0), np.arange(2.0, 6.0), label="única")

    with pytest.raises(PlantaQueNaoSeparaError, match="duas séries"):
        planta_serie_removida(ax)


# ---------------------------------------------------------------------------
# 3. O auxiliar do recorte — R3 e R5b
# ---------------------------------------------------------------------------


def test_o_rodape_declara_os_cinco_campos() -> None:
    """Os cinco campos na frase, e a frase na figura.

    **Em duas asserções e não numa, porque a primeira redação deste teste reprovou por confundir as
    duas coisas.** `Recorte.texto()` é a frase; o que está desenhado é a frase **quebrada**, e a
    quebra cai onde calhar — no rodapé real ela caiu dentro de *"sem teste estatístico"*. Procurar
    um campo inteiro no artista seria afirmar, sem querer, que a frase nunca quebra.
    """
    declarado = Recorte(
        desenhadas=7,
        disponiveis=POPULACAO,
        amostragem="os QCs da corrida, sem amostragem",
        matriz="corrigida por SERRF",
        metodo="sem teste estatístico",
    )
    frase = declarado.texto()
    assert "7 de 6.035 features" in frase
    assert "amostragem: os QCs da corrida, sem amostragem" in frase
    assert "matriz: corrigida por SERRF" in frase
    assert "método: sem teste estatístico" in frase

    prova = figura_de_prova()
    desenhado = prova.figura.texts[0].get_text()
    assert desenhado == "\n".join(quebra(frase, colunas_da_figura(prova.figura)))


@pytest.mark.parametrize(
    ("campos", "trecho"),
    [
        ({"disponiveis": 0}, "`disponiveis` é 0"),
        ({"desenhadas": -1}, "não pode ser negativo"),
        ({"desenhadas": 9000}, "desenhar mais do que existe".capitalize()[:8]),
        ({"amostragem": "   "}, "`amostragem` está em branco"),
        ({"matriz": ""}, "`matriz` está em branco"),
        ({"metodo": ""}, "`metodo` está em branco"),
    ],
)
def test_o_recorte_recusa_declaracao_impossivel(campos: dict[str, object], trecho: str) -> None:
    base_do_recorte: dict[str, object] = {
        "desenhadas": 7,
        "disponiveis": 6035,
        "amostragem": "sem amostragem",
        "matriz": "bruta",
        "metodo": "sem teste estatístico",
    }
    with pytest.raises(RecorteInvalidoError, match=trecho):
        Recorte(**{**base_do_recorte, **campos})  # type: ignore[arg-type]


def test_o_recorte_valido_nao_levanta() -> None:
    """A outra ponta: a guarda deixa passar o que é possível.

    Sem isto, uma guarda que levantasse para TUDO passaria nos seis casos acima.
    """
    valido = Recorte(0, 1, "nenhuma", "bruta", "sem teste estatístico")
    assert valido.texto().startswith("0 de 1 features")


def test_frase_contada_tem_os_TRES_ramos_e_o_do_UM_concorda() -> None:  # noqa: N802
    """Os três desfechos de `frase_contada`, e a planta cai onde as três redações DIFEREM.

    **Defeito já consertado.** Antes do conserto esta função tinha
    dois ramos, e `quantas == 1` caía no do plural: `as 1 partes da partição não receberam amostra
    nenhuma`. **Medido antes do conserto**, com estas mesmas orações.

    **A planta cai sobre dado em que os três diferem** — as três orações são textos distintos, e
    afirmar sobre orações iguais passaria com qualquer ramificação, inclusive com a de dois ramos
    que esta unidade veio substituir.

    **E `quantas == 2` está aqui por ser o vizinho do um**: um conserto que mandasse `<= 1` para o
    singular tornaria `as 2 partes` em `1 parte`, e nenhuma asserção sobre 0 e 1 o veria.
    """
    tres = {
        "nenhuma": "nenhum agrupamento ficou vazio",
        "uma": "1 parte da partição não recebeu amostra nenhuma",
        "algumas": "partes da partição não receberam amostra nenhuma",
    }
    assert len({tres["nenhuma"], tres["uma"], tres["algumas"]}) == 3, (
        "âncora: a planta só separa se as três orações forem textos diferentes"
    )

    assert frase_contada(0, **tres) == "nenhum agrupamento ficou vazio"
    assert frase_contada(1, **tres) == "1 parte da partição não recebeu amostra nenhuma"
    assert frase_contada(2, **tres) == "as 2 partes da partição não receberam amostra nenhuma"

    # O defeito, nomeado: nenhum dos três desfechos pode ser a frase que a sonda encontrou.
    for quantas in (0, 1, 2):
        assert "as 1 " not in frase_contada(quantas, **tres)

    # **O milhar continua a sair do ramo do plural**, e o negativo continua a cair nele — que é o
    # comportamento de antes do ramo do um, e não um desvio para `uma`.
    assert frase_contada(1200, **tres).startswith("as 1.200 ")
    assert frase_contada(-1, **tres).startswith("as -1 ")


def test_frase_contada_NAO_TEM_padrao_para_o_ramo_do_um() -> None:  # noqa: N802
    """A metade que faz o conserto durar: `uma` é obrigatório, e a décima família não o herda.

    **É a guarda das cópias irmãs na forma que ela toma quando o irmão ainda não existe.** Um
    padrão que recaísse no plural devolveria `as 1 partes` à próxima figura sem nada a acusar — o
    valor natural que executa menos do caminho. Sem esta asserção, alguém pode acrescentar
    `uma: str = ""` e os três ramos acima continuam verdes.
    """
    with pytest.raises(TypeError, match="uma"):
        frase_contada(1, nenhuma="nenhuma", algumas="algumas")  # type: ignore[call-arg]


def test_TODA_chamada_de_frase_contada_do_pacote_passa_as_TRES_oracoes() -> None:  # noqa: N802
    """A varredura da CLASSE, e não da instância que a sonda reportou.

    A sonda reportou **uma** frase — a do K-means. A classe são todas as chamadas do pacote, e a
    asserção é de PRESENÇA e não de ausência: a não-vacuidade vai na mesma função e antes,
    senão um `glob` que não casasse ficheiro nenhum passaria por vazio.

    **O oráculo é a assinatura e não um `grep` de texto**: `inspect` resolve os argumentos
    nomeados que cada chamada de facto passa, então uma chamada que omitisse `uma` já teria
    levantado `TypeError` na importação do módulo — e o que este teste acrescenta é a **contagem**,
    que é o que denuncia uma chamada nova a nascer sem o ramo.
    """
    raiz = Path(recorte.__file__).parent
    chamadas: list[tuple[str, int]] = []
    for arquivo in sorted(raiz.glob("*.py")):
        arvore = ast.parse(arquivo.read_text(encoding="utf-8"))
        for no in ast.walk(arvore):
            if not isinstance(no, ast.Call) or not isinstance(no.func, ast.Name):
                continue
            if no.func.id != "frase_contada":
                continue
            nomeados = {palavra.arg for palavra in no.keywords}
            chamadas.append((f"{arquivo.name}:{no.lineno}", len(nomeados)))
            assert {"nenhuma", "uma", "algumas"} <= nomeados, (
                f"{arquivo.name}:{no.lineno} chama `frase_contada` sem as três orações: {nomeados}"
            )

    assert len(chamadas) >= 9, (
        "não-vacuidade: a varredura tem de achar as chamadas antes de afirmar sobre elas. "
        f"Achei {len(chamadas)}, e a U22 mediu NOVE em CINCO módulos — `pca` (2), "
        "`agrupamento` (3), "
        f"`correlacao` (2), `plsda` (1) e `plsda_multiclasse` (1)."
    )


def test_marcas_que_cabem_tem_os_DOIS_desfechos_e_o_comprimento_e_que_DECIDE() -> None:  # noqa: N802
    """O predicado unificado: os dois desfechos, e a planta é o **comprimento**.

    **A planta cai sobre o eixo que MUDA a resposta.** Com a contagem fixa e só o comprimento a
    variar, uma implementação que ignorasse o comprimento — e devolvesse sempre `True`, ou sempre
    `False` — reprova. Variar a contagem também separa, e é por isso que as duas direções estão
    aqui: o predicado tem dois argumentos e um só deles exercitado deixa o outro inerte.
    """
    linha = TAMANHO_DA_FONTE * ENTRELINHA

    # A borda exacta, nos dois lados: o comprimento decide.
    assert marcas_que_cabem(linha * 10.0, 10) is True
    assert marcas_que_cabem(linha * 10.0 - 0.01, 10) is False

    # E a contagem decide, com o comprimento fixo.
    assert marcas_que_cabem(linha * 10.0, 20) is False

    # Sem marca nenhuma a resposta é «não cabem», e não uma divisão por zero.
    assert marcas_que_cabem(1000.0, 0) is False

    # **`is True` e não `assert ...`, de propósito:** um `np.bool_` devolvido por engano passaria
    # num `assert` cru e reprova aqui.
    assert isinstance(marcas_que_cabem(linha * 10.0, 10), bool)


def test_rotulos_cabem_e_marcas_que_cabem_sao_UMA_conta_com_dois_nomes() -> None:  # noqa: N802
    """A âncora do que foi unificado, e ela é o gesto: contar antes de escrever.

    `distribuicao.rotulos_cabem` passou a ser `marcas_que_cabem(largura útil da figura, n)`. Esta
    função afirma a EQUIVALÊNCIA sobre a população inteira em que a equivalência foi medida — 82
    valores de `n` —, e não sobre o caso em que a vi.

    **E a asserção de não-vacuidade vai antes:** uma comparação `a == b` sobre uma lista
    vazia de `n` passaria, e é exactamente o que um `range` mal escrito produz. A segunda âncora é
    mais forte: os dois desfechos têm de APARECER na varredura, senão a equivalência é entre duas
    constantes.
    """
    figura = nova_figura()
    util = largura_util_em_pontos(figura)

    ns = [*range(1, 80), 100, 200, 500]
    assert len(ns) == 82, "não-vacuidade: a população da equivalência é a de 82 valores"

    respostas = [(n, distribuicao.rotulos_cabem(figura, n), marcas_que_cabem(util, n)) for n in ns]
    assert {resposta for _n, resposta, _ in respostas} == {True, False}, (
        "a varredura tem de produzir os DOIS desfechos: uma equivalência entre dois `True` "
        "constantes não separa implementação nenhuma"
    )
    divergem = [(n, a, b) for n, a, b in respostas if a != b]
    assert not divergem, f"as duas contas divergem em {divergem}"


def test_largura_util_em_pontos_desce_da_MARGEM_e_nao_e_a_largura_crua() -> None:  # noqa: N802
    """A terceira cópia unificada, e a planta separa-a da largura crua.

    Sem a margem, este auxiliar devolveria `figwidth * 72` — e as duas contas que o lêem passariam
    a achar que há mais largura do que existe. A planta é a diferença: a margem é `MARGEM` dos
    **dois** lados, e o valor tem de ser estritamente menor que a largura crua.
    """
    figura = nova_figura(largura=7.0)
    crua = 7.0 * PONTOS_POR_POLEGADA
    util = largura_util_em_pontos(figura)

    assert util < crua, "a margem não foi descontada"
    assert util == pytest.approx(crua * (1.0 - 2.0 * MARGEM))
    # E ela acompanha a figura: uma figura mais larga tem mais largura útil.
    assert largura_util_em_pontos(nova_figura(largura=14.0)) > util


def test_o_rodape_quebra_em_vez_de_sair_pela_borda() -> None:
    """A borda: texto mais largo que a figura. Ele quebra, e a caixa desenhada cabe no canvas.

    **A medida é do renderizador e não da conta**: `get_window_extent` devolve a caixa que o `Agg`
    de facto desenhou. Conferir a conta contra si mesma seria o instrumento a certificar-se.
    """
    figura = nova_figura()
    figura.add_subplot(1, 1, 1)
    longo = "corrigida por SERRF sobre a matriz do pos_hilic " * 8
    artista = aplica(figura, Recorte(7, POPULACAO, "sem amostragem", longo, "Welch"))

    linhas = artista.get_text().split("\n")
    assert len(linhas) > 1, "o texto longo não quebrou"
    assert max(len(x) for x in linhas) <= colunas_da_figura(figura)

    figura.canvas.draw()
    caixa = artista.get_window_extent(figura.canvas.get_renderer())
    assert caixa.x0 >= 0 and caixa.y0 >= 0, f"o rodapé sai pela borda de baixo/esquerda: {caixa}"
    assert caixa.x1 <= figura.bbox.x1 and caixa.y1 <= figura.bbox.y1, (
        f"o rodapé sai pela borda de cima/direita: {caixa} contra {figura.bbox}"
    )


def test_o_rodape_recusa_quando_nem_quebrado_cabe() -> None:
    """O teto de `RESERVA_MAXIMA`: recusa tipada em vez de desenhar por cima dos eixos."""
    figura = nova_figura()
    figura.add_subplot(1, 1, 1)
    enorme = "x" * 4000

    with pytest.raises(RecorteInvalidoError, match="não é truncado"):
        aplica(figura, Recorte(7, POPULACAO, "sem amostragem", enorme, "Welch"))


def test_o_rodape_reserva_espaco_e_nunca_o_devolve() -> None:
    """`aplica` só sobe o fundo dos eixos, e sobe quando o rodapé de facto precisa.

    **A segunda metade foi MEDIDA e não suposta, e a primeira redação deste teste estava errada:**
    ela afirmava que um rodapé de uma linha sobe o fundo, e ele não sobe — o padrão do `matplotlib`
    é `bottom = 0.11`, e uma linha de 8 pt numa figura de 4,5 polegadas ocupa 0,033. Não subir ali é
    o comportamento certo; o que se tem de provar é que ele sobe quando o rodapé passa do padrão.

    ## A primeira asserção era IGUALDADE e passou a ser `>=` — e a mudança é declarada

    Antes, `aplica` terminava no `subplots_adjust`, e o fundo pedido saía intacto. Ela passou a
    afastar também as **marcas e o rótulo** do eixo x da faixa do rodapé (ver a seção «reservar
    espaço não é afastar» de `recorte.py`), e `tight_layout` acrescenta ao fundo pedido o espaço
    que essas marcas ocupam: medido nesta figura, `0.4` sai `0.486`.

    > **O que aquela igualdade fixava era *«nunca desce»*, e isso continua verdade.** A forma —
    > sair **exactamente** o valor pedido — era consequência de a função fazer uma coisa só, e
    > deixou de a fazer. Trocar a asserção pela propriedade é a leitura certa:
    > perguntar por que a forma era contrato antes de a mudar. A direção oposta continua guardada:
    > um `aplica` que **descesse** o fundo reprova aqui.
    """
    ja_reservado = nova_figura()
    ja_reservado.add_subplot(1, 1, 1)
    ja_reservado.subplots_adjust(bottom=0.4)
    aplica(ja_reservado, Recorte(7, POPULACAO, "sem amostragem", "bruta", "Welch"))
    assert ja_reservado.subplotpars.bottom >= 0.4

    de_varias_linhas = nova_figura()
    de_varias_linhas.add_subplot(1, 1, 1)
    padrao = de_varias_linhas.subplotpars.bottom
    aplica(
        de_varias_linhas,
        Recorte(7, POPULACAO, "sem amostragem", "corrigida por SERRF " * 20, "Welch"),
    )
    artista = de_varias_linhas.texts[0]
    assert artista.get_text().count("\n") >= 3, "âncora: o rodapé desta prova tem de quebrar"
    assert de_varias_linhas.subplotpars.bottom > padrao


def test_D668_retira_devolve_o_texto_PRISTINO_e_reajusta_repoe_o_rodape() -> None:  # noqa: N802
    """`retira` apaga o rodapé e devolve o texto prístino; `reajusta` o repõe como era.

    **O prístino e não o desenhado**: o desenhado leva as quebras de linha da largura da figura, e
    o texto vai para a legenda de um artigo, que a revista reflui. A âncora é um rodapé que QUEBRA
    — num de uma linha só os dois coincidem, e a asserção passaria a devolver o desenhado.

    **E o contrato de `reajusta` vale para ela**: um `Text` com a marca do rodapé, posto à mão sem
    passar por `aplica`, não tem texto prístino guardado — ninguém o reporia depois —, e `retira`
    devolve `None` sem lhe tocar.
    """
    figura = nova_figura()
    figura.add_subplot(1, 1, 1)
    recorte = Recorte(7, POPULACAO, "sem amostragem", "corrigida por SERRF " * 20, "Welch")
    rodape = aplica(figura, recorte)
    desenhado = rodape.get_text()
    assert "\n" in desenhado, "âncora: o rodapé desta prova não quebrou"

    assert retira(figura, rodape) == recorte.texto()
    assert rodape.get_text() == "", "o rodapé continua na figura"
    assert reajusta(figura, rodape) is True, "o rodapé apagado não foi reposto"
    assert rodape.get_text() == desenhado

    a_mao = nova_figura()
    a_mao.add_subplot(1, 1, 1)
    marcado = a_mao.text(0.0, 0.0, "rodapé posto à mão", gid=GID_DO_RODAPE)
    assert retira(a_mao, marcado) is None
    assert marcado.get_text() == "rodapé posto à mão"


def test_a_quebra_de_texto_vazio_devolve_uma_linha_vazia() -> None:
    """`textwrap.wrap("")` devolve lista VAZIA, e uma tupla vazia faria `"\\n".join` sumir o rodapé.

    Na escala de uma função: *"nada a escrever"* e *"uma linha em branco"* têm de ser
    o mesmo valor de retorno bem definido, senão o chamador ramifica sobre o vazio.
    """
    assert quebra("", 40) == ("",)
    assert quebra("uma frase curta", 40) == ("uma frase curta",)


def test_a_largura_util_sai_da_figura_e_nao_de_um_literal() -> None:
    estreita = nova_figura(largura=3.0)
    larga = nova_figura(largura=12.0)
    assert colunas_da_figura(larga) > colunas_da_figura(estreita)
    assert colunas_da_figura(nova_figura(largura=0.1)) == recorte.COLUNAS_MINIMAS


# ---------------------------------------------------------------------------
# 4. A recusa tipada para população vazia
# ---------------------------------------------------------------------------


def test_populacao_vazia_produz_recusa_tipada_e_nao_figura_em_branco() -> None:
    with pytest.raises(MatrizVaziaError, match="não há população para desenhar"):
        exige_populacao("intensidades", np.empty(0))
    with pytest.raises(MatrizVaziaError):
        exige_populacao("intensidades", [])


def test_populacao_nao_vazia_devolve_o_array_convertido() -> None:
    convertido = exige_populacao("intensidades", [1, 2, 3])
    assert convertido.dtype == np.float64
    assert convertido.tolist() == [1.0, 2.0, 3.0]


# ---------------------------------------------------------------------------
# 5. Tema — o construtor não tem opinião sobre cor
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("fundo", "cor_do_texto"),
    [("#101014", "#EAEAEA"), ("#FFFFFF", "#000000")],
)
def test_a_figura_segue_os_rcparams_em_vez_de_pintar(fundo: str, cor_do_texto: str) -> None:
    """As duas direções: a MESMA chamada produz figuras de cores opostas conforme o contexto.

    O que este cenário afirma é que o construtor **não tem opinião sobre cor** — ele herda os
    `rcParams` vigentes. Quem pinta é `desktop/janela/figuras.aplica_o_tema`, sobre a figura já
    construída; se o construtor pintasse, a cor passaria a ser uma correção posterior sobre cada
    artista.

    **A motivação escrita aqui até 14/09/2026 caducou por dois lados, e os dois vão declarados.**
    Ela dizia que *«o widget lê o tema do Windows e escreve nos `rcParams`»*: (a) o widget
    **nunca** escreveu nos `rcParams` — `grep -rn "rcParams" src/refino/` não devolve escrita
    nenhuma, e `aplica_o_tema` põe a cor artista a artista; e (b) hoje ele já nem lê o
    tema — pinta com uma trinca fixa, branca. **A propriedade sob teste não mudou**: é sobre o
    construtor, e é por isso que o cenário fica como está.
    """
    with matplotlib.rc_context({"figure.facecolor": fundo, "text.color": cor_do_texto}):
        prova = figura_de_prova()

    esperado = matplotlib.colors.to_rgba(fundo)
    assert prova.figura.get_facecolor() == pytest.approx(esperado)
    assert prova.figura.texts[0].get_color() == cor_do_texto


def test_nenhum_modulo_do_pacote_fixa_cor_de_fundo_ou_de_texto() -> None:
    """O portão da metade que o teste acima não alcança: um construtor FUTURO que pinte.

    Proxy de forma, e a estreiteza vai declarada: ele acusa `facecolor=`, `edgecolor=` e `color=`
    como argumento nomeado com literal de texto. Não acusa uma cor passada por variável. O que ele
    compra é a direção que importa — a cor fixada onde é natural fixá-la, na chamada.
    """
    proibidos = {"facecolor", "edgecolor", "color", "colors"}
    achados: list[str] = []
    assert FONTES_DO_PACOTE, "âncora: nenhum arquivo de `plots/` foi encontrado"
    for caminho in FONTES_DO_PACOTE:
        arvore = ast.parse(caminho.read_text(encoding="utf-8"))
        for no in ast.walk(arvore):
            if not isinstance(no, ast.Call):
                continue
            for palavra in no.keywords:
                if palavra.arg in proibidos and isinstance(palavra.value, ast.Constant):
                    achados.append(f"{caminho.name}:{no.lineno} fixa {palavra.arg}=")
    assert achados == [], achados


def test_o_portao_de_cor_acusa_uma_cor_fixada() -> None:
    """Par causal em processo: a planta é o texto que o extractor recebe."""
    plantado = "figura.set_facecolor(x)\nax.plot(a, b, color='#FFFFFF')\n"
    achados = [
        palavra.arg
        for no in ast.walk(ast.parse(plantado))
        if isinstance(no, ast.Call)
        for palavra in no.keywords
        if palavra.arg == "color" and isinstance(palavra.value, ast.Constant)
    ]
    assert achados == ["color"], "o extractor de cor fixada não acusa a planta"


# ---------------------------------------------------------------------------
# 6. Direção de import — o regime que não existia
# ---------------------------------------------------------------------------

FONTES_DO_PACOTE = sorted(PACOTE.glob("*.py"))

# Raiz de import proibida em `plots/`. `matplotlib.pyplot` está aqui pelo motivo escrito no
# `__init__.py` do pacote: o gestor global de figuras vaza entre painéis Qt.
PROIBIDOS: Final = frozenset(
    {
        "PySide6",
        "PyQt5",
        "PyQt6",
        "shiboken6",
        "httpx",
        "httpx2",
        "requests",
        "urllib3",
        "fastapi",
        "starlette",
        "streamlit",
        "sqlalchemy",
        "psycopg2",
        "alembic",
        "refino.api",
        "refino.desktop",
        "refino.interface",
        "refino.persistence",
        "matplotlib.pyplot",
    }
)


def importados(fonte: str) -> set[str]:
    """Todo módulo importado, na forma qualificada E na raiz, para casar as duas sintaxes.

       `import matplotlib.pyplot as plt` e `from matplotlib import pyplot` são a mesma coisa
       escrita de dois modos, e um portão que só casasse a primeira ficaria cego à segunda — é o
       proxy de forma
    nesta superfície.
    """
    nomes: set[str] = set()
    for no in ast.walk(ast.parse(fonte)):
        if isinstance(no, ast.Import):
            for alias in no.names:
                nomes.add(alias.name)
                nomes.add(alias.name.split(".")[0])
        elif isinstance(no, ast.ImportFrom) and no.module:
            nomes.add(no.module)
            nomes.add(no.module.split(".")[0])
            for alias in no.names:
                nomes.add(f"{no.module}.{alias.name}")
    return nomes


def proibidos_em(fonte: str) -> list[str]:
    return sorted(importados(fonte) & PROIBIDOS)


def test_o_pacote_de_figuras_nao_conhece_qt_http_api_nem_pyplot() -> None:
    """A varredura sintática, com a âncora de presença antes.

    `not (importados & proibidos)` passa sobre conjunto **vazio**, e conjunto vazio é o que uma
    extração que deixasse de casar devolveria. A âncora afirma que houve import extraído em cada
    arquivo antes de a asserção de ausência correr.
    """
    assert FONTES_DO_PACOTE, "âncora: nenhum arquivo de `plots/` foi encontrado"
    for caminho in FONTES_DO_PACOTE:
        fonte = caminho.read_text(encoding="utf-8")
        assert importados(fonte), f"{caminho.name}: nenhum import extraído — a asserção seria vácua"
        assert proibidos_em(fonte) == [], f"{caminho.name} importa: {proibidos_em(fonte)}"


@pytest.mark.parametrize(
    "plantado",
    [
        "import PySide6\n",
        "from PySide6.QtWidgets import QWidget\n",
        "import matplotlib.pyplot as plt\n",
        "from matplotlib import pyplot\n",
        "from refino.api import rotas\n",
        "import httpx2\n",
    ],
)
def test_o_portao_de_import_acusa_a_planta(plantado: str) -> None:
    """As seis formas, uma por execução. Negativo **em processo**: a planta é o argumento."""
    assert proibidos_em(plantado), f"a planta {plantado!r} passou pelo portão"


def test_o_portao_de_import_nao_acusa_o_que_e_legitimo() -> None:
    """A outra ponta: sem planta, não acusa — senão um portão que acusasse tudo passaria acima."""
    assert proibidos_em("import numpy as np\nfrom matplotlib.figure import Figure\n") == []


def test_importar_o_pacote_nao_puxa_qt_transitivamente() -> None:
    """A metade que o AST não alcança: um import indireto, por uma dependência de uma dependência.

    Corre em subprocesso com `PYTHONPATH` no `src` **desta árvore** — num worktree, o `refino`
    instalado aponta para a árvore principal, e o portão mediria o pacote errado. O
    arquivo lido é reportado de volta e conferido.

    **Âncora de não-vacuidade:** `matplotlib` TEM de estar em `sys.modules` depois do import. Um
    subprocesso que falhasse a meio, ou um `refino.plots` vazio, dariam interseção vazia com os
    proibidos — que é o desfecho de sucesso.
    """
    sonda = textwrap.dedent(
        """
        import json, sys
        import refino.plots as p
        raizes = {m.split(".")[0] for m in sys.modules}
        print(json.dumps({"arquivo": p.__file__, "raizes": sorted(raizes)}))
        """
    )
    resultado = subprocess.run(
        [sys.executable, "-c", sonda],
        capture_output=True,
        text=True,
        env={**dict(__import__("os").environ), "PYTHONPATH": str(REPO / "src")},
        check=False,
    )
    assert resultado.returncode == 0, resultado.stderr
    lido = json.loads(resultado.stdout.strip().splitlines()[-1])

    assert Path(lido["arquivo"]).resolve().is_relative_to(REPO), (
        f"o subprocesso importou `refino.plots` de {lido['arquivo']}, fora desta árvore"
    )
    assert "matplotlib" in lido["raizes"], (
        "âncora: `matplotlib` não está em sys.modules — o import não aconteceu, e a asserção "
        "de ausência abaixo passaria por vacuidade"
    )
    raizes_proibidas = {p.split(".")[0] for p in PROIBIDOS} - {"refino", "matplotlib"}
    assert not (set(lido["raizes"]) & raizes_proibidas), (
        f"importar `refino.plots` puxa {sorted(set(lido['raizes']) & raizes_proibidas)}"
    )


# ---------------------------------------------------------------------------
# 7. A bijeção de R1, sobre a PARTIÇÃO do catálogo
# ---------------------------------------------------------------------------

_POR_EXTENSO: Final = {
    "cinco": 5,
    "sete": 7,
    "dezanove": 19,
    "dezenove": 19,
    "vinte": 20,
}

_LINHA_DE_TABELA = "|"


@dataclass(frozen=True)
class LinhaDoCatalogo:
    entrada: str
    grupo: str
    estado: str


def _celulas(linha: str) -> list[str]:
    return [c.strip().replace("**", "") for c in linha.strip().strip("|").split("|")]


def _bloco(texto: str, titulo: str) -> str:
    """O trecho entre um cabeçalho `####` e o seguinte."""
    corpo = texto.split(titulo, 1)
    assert len(corpo) == 2, f"o cabeçalho {titulo!r} não existe em {CATALOGO_MD.name}"
    return corpo[1].split("\n#### ", 1)[0].split("\n---", 1)[0]


def contagens_declaradas(texto: str) -> tuple[int, int]:
    """As duas palavras por extenso do cabeçalho do `#### 2.` — o agregado que ninguém confere.

    É a forma: o número que se lê primeiro é o do título, e nenhum portão o conferia.
    """
    import re

    achado = re.search(r"A seção STATISTICS — (\w+) entradas, em (\w+) grupos", texto)
    assert achado, "o cabeçalho do `#### 2.` mudou de forma — o portão perdeu a fonte independente"
    return _POR_EXTENSO[achado.group(1).lower()], _POR_EXTENSO[achado.group(2).lower()]


def grupos_do_levantamento(texto: str) -> dict[str, tuple[str, ...]]:
    """A tabela `| Grupo | Entradas |` do `#### 2.`, que existe desde antes desta unidade."""
    grupos: dict[str, tuple[str, ...]] = {}
    for linha in _bloco(texto, "#### 2. A seção STATISTICS").splitlines():
        if not linha.startswith(_LINHA_DE_TABELA):
            continue
        celulas = _celulas(linha)
        if len(celulas) != 2 or celulas[0] in {"Grupo", ""} or set(celulas[0]) <= {"-"}:
            continue
        grupos[celulas[0]] = tuple(e.strip() for e in celulas[1].split(","))
    return grupos


def estados_do_catalogo(texto: str) -> tuple[LinhaDoCatalogo, ...]:
    """A tabela do `#### 7.`: uma linha por entrada, com o estado."""
    linhas: list[LinhaDoCatalogo] = []
    for linha in _bloco(texto, "#### 7. O ESTADO de cada entrada").splitlines():
        if not linha.startswith(_LINHA_DE_TABELA):
            continue
        celulas = _celulas(linha)
        if len(celulas) != 4 or celulas[0] in {"Entrada", ""} or set(celulas[0]) <= {"-"}:
            continue
        linhas.append(LinhaDoCatalogo(entrada=celulas[0], grupo=celulas[1], estado=celulas[2]))
    return tuple(linhas)


# KD3, emendada: o conjunto tinha DOIS membros e passou a ter UM. A
# exclusão continua derivada do GRUPO, nunca escolhida linha a linha — é o que impede a lista de
# isenção de crescer uma entrada de cada vez, e é a razão de a emenda ser aqui e não nas duas
# células do documento.
#
# **`Classification & Feature Selection` saiu inteiro**, com Random Forest e SVM construídos.
# `Advanced Significance` fica, e SAM e EBAM continuam `excluída-KD3` por **pertencerem a ele** — e
# não por uma linha própria, que seria a lista de isenção com outro nome.
#
# **Esta constante é lida em DUAS DIREÇÕES OPOSTAS por `defeitos_do_catalogo`**, e é por isso que a
# emenda não se faz satisfazendo uma delas (na metade de direção). As duas foram vistas
# reprovar com a emenda pela metade:
#
# | Metade aplicada | O que o portão acusou |
# |---|---|
# | as duas células viradas, este conjunto intacto | `'Random Forest' é do grupo ... e KD3 |
# | | exclui-o inteiro` — 2 achados |
# | este conjunto encolhido, as células intactas | `'Random Forest' está 'excluída-KD3' ... |
# | | lista de isenção` — 2 achados |
GRUPOS_EXCLUIDOS_POR_KD3: Final = frozenset({"Advanced Significance"})


def defeitos_do_catalogo(
    linhas: tuple[LinhaDoCatalogo, ...], grupos: dict[str, tuple[str, ...]]
) -> list[str]:
    """Vocabulário fechado, bijeção com o levantamento, e KD3 derivado do grupo.

    Devolve lista para que as plantas entrem como **argumento** — negativo em processo.
    """
    achados: list[str] = []
    vocabulario = {e.value for e in EstadoDoCatalogo}

    for linha in linhas:
        if linha.estado not in vocabulario:
            achados.append(
                f"{linha.entrada!r} tem estado {linha.estado!r}, fora do vocabulário "
                f"{sorted(vocabulario)}. Estado desconhecido tratado como «adiada» tornaria todo "
                "erro de digitação numa isenção silenciosa"
            )
        excluida_pelo_grupo = linha.grupo in GRUPOS_EXCLUIDOS_POR_KD3
        marcada = linha.estado == EstadoDoCatalogo.EXCLUIDA_KD3
        if excluida_pelo_grupo and not marcada:
            achados.append(f"{linha.entrada!r} é do grupo {linha.grupo!r} e KD3 exclui-o inteiro")
        if marcada and not excluida_pelo_grupo:
            achados.append(
                f"{linha.entrada!r} está `excluída-KD3` e o grupo {linha.grupo!r} não é nenhum "
                f"dos {len(GRUPOS_EXCLUIDOS_POR_KD3)} que KD3 nomeia — exclusão linha a linha é "
                "lista de isenção"
            )

    do_levantamento = {e for entradas in grupos.values() for e in entradas}
    da_tabela = {x.entrada for x in linhas if x.grupo in grupos}
    if faltam := sorted(do_levantamento - da_tabela):
        achados.append(f"entradas do levantamento sem estado declarado: {faltam}")
    if sobram := sorted(da_tabela - do_levantamento):
        achados.append(f"entradas com estado e sem linha no levantamento: {sobram}")
    return achados


def construtores_descobertos() -> tuple[tuple[str, str], ...]:
    """Toda função `figura_*` de módulo em `plots/`, lida por AST — nunca por import.

    Ler por AST e não importar é o que faz o portão ver o **arquivo desta árvore** em vez do módulo
    que o `sys.path` calhou resolver.
    """
    achados: list[tuple[str, str]] = []
    for caminho in FONTES_DO_PACOTE:
        arvore = ast.parse(caminho.read_text(encoding="utf-8"))
        for no in arvore.body:
            if isinstance(no, ast.FunctionDef) and no.name.startswith(PREFIXO_DO_CONSTRUTOR):
                achados.append((caminho.stem, no.name))
    return tuple(sorted(achados))


def defeitos_da_bijecao(
    descobertos: tuple[tuple[str, str], ...],
    registados: tuple[ConstrutorDeFigura, ...],
    construidas: frozenset[str],
) -> list[str]:
    """As três direções: registo contra código, registo contra catálogo, catálogo contra registo."""
    achados: list[str] = []

    no_codigo = set(descobertos)
    no_registo = {(c.modulo, c.funcao) for c in registados}
    if faltam := sorted(no_codigo - no_registo):
        achados.append(f"construtores em `plots/` e fora de CONSTRUTORES: {faltam}")
    if sobram := sorted(no_registo - no_codigo):
        achados.append(f"entradas de CONSTRUTORES sem função em `plots/`: {sobram}")

    chaves_do_relatorio = {f"item-{i.numero}" for i in ITENS_OBRIGATORIOS}
    do_catalogo: set[str] = set()
    for construtor in registados:
        if construtor.vinculo is Vinculo.CATALOGO:
            do_catalogo.add(construtor.chave)
        elif construtor.chave not in chaves_do_relatorio:
            achados.append(
                f"{construtor.funcao!r} bijeta contra o relatório e a chave {construtor.chave!r} "
                f"não é um dos {len(chaves_do_relatorio)} itens de ITENS_OBRIGATORIOS"
            )

    if faltam_construtores := sorted(construidas - do_catalogo):
        achados.append(f"entradas `construída` do [G-1] sem construtor: {faltam_construtores}")
    if faltam_entradas := sorted(do_catalogo - construidas):
        achados.append(
            f"construtores que dizem desenhar uma entrada do [G-1] que não está `construída`: "
            f"{faltam_entradas}"
        )
    return achados


def test_a_bijecao_de_r1_corre_sobre_a_particao_construida() -> None:
    """O portão de R1, **com as âncoras de não-vacuidade na mesma função**.

    **A partição `construída` já esteve VAZIA, e a bijeção passava trivialmente.**
    O motor e a primeira figura bijetam contra o **relatório** e não contra o
    `[G-1]`; há lá três entradas — as caixas por feature e as duas densidades —, e é a partir
    dela que as três direções deste portão correm sobre população de facto.

    As **cinco** âncoras abaixo são o que impede este teste de dizer «passou» sobre nada, e vivem
    **aqui** e não num teste vizinho, que poderia ser removido sem que nada ligasse os dois.
    A quinta é a que fecha justamente a metade que acendeu a população.
    """
    texto = CATALOGO_MD.read_text(encoding="utf-8")
    grupos = grupos_do_levantamento(texto)
    linhas = estados_do_catalogo(texto)
    entradas_declaradas, grupos_declarados = contagens_declaradas(texto)

    # Âncora 1 — o agregado do título contra o que a tabela do `#### 2.` tem. Fonte independente
    # desta unidade: o levantamento foi escrito antes de a tabela de estado existir.
    assert len(grupos) == grupos_declarados, f"{len(grupos)} grupos lidos, {grupos_declarados}"
    do_levantamento = [e for entradas in grupos.values() for e in entradas]
    assert len(do_levantamento) == entradas_declaradas, (
        f"o cabeçalho declara {entradas_declaradas} entradas e a tabela tem {len(do_levantamento)}"
    )

    # Âncora 2 — a tabela de estado foi lida de facto, e cobre o levantamento inteiro mais as
    # quatro linhas de Normalização, que vêm do `#### 1.`.
    assert len(linhas) == entradas_declaradas + 4, (
        f"{len(linhas)} linhas de estado lidas; esperadas {entradas_declaradas} do STATISTICS mais "
        "as quatro do painel de normalização"
    )

    # Âncora 3 — o descobridor de construtores lê esta árvore, e ela não está vazia.
    assert FONTES_DO_PACOTE, "âncora: nenhum arquivo de `plots/` foi encontrado"

    # **Âncora 4 — a metade do CÓDIGO não é vazia.** Antes, as três direções
    # da bijeção corriam sobre conjuntos vazios dos dois lados; a partição `construída` do catálogo
    # continua vazia — a trajetória do QC bijeta contra o relatório —, mas a direção
    # código↔registo passou a ter população, e é esta a linha que o afirma.
    descobertos = construtores_descobertos()
    assert descobertos, "âncora: a varredura por AST não achou construtor nenhum em `plots/`"

    assert defeitos_do_catalogo(linhas, grupos) == []

    construidas = frozenset(x.entrada for x in linhas if x.estado == EstadoDoCatalogo.CONSTRUIDA)

    # **Âncora 5 — a partição `construída` NÃO É VAZIA.** Antes a bijeção
    # deste teste corria entre dois conjuntos vazios do lado do catálogo: `construidas` era
    # `frozenset()`, `do_catalogo` era `set()`, e as duas direções passavam trivialmente. As
    # âncoras 1 a 4 provam que houve tabela, houve levantamento e houve varredura de código; **esta
    # é a que prova que houve BIJEÇÃO**, e sem ela um dia em que alguém pusesse as quatro linhas de
    # volta em `adiada` deixaria este teste verde sobre nada.
    assert construidas, (
        "a partição `construída` do [G-1] está vazia: a bijeção de R1 passa por vacuidade. Ela "
        "deixou de estar vazia na U4, e voltar a estar é verificação inerte"
    )
    # E a metade correspondente do lado do código, pela mesma razão e na mesma função.
    do_catalogo = [c for c in CONSTRUTORES if c.vinculo is Vinculo.CATALOGO]
    assert do_catalogo, (
        "nenhum construtor bijeta contra o [G-1]: com a partição cheia e este lado vazio a "
        "asserção abaixo acusaria, mas com os dois vazios ela passaria"
    )

    assert defeitos_da_bijecao(descobertos, CONSTRUTORES, construidas) == []


@pytest.mark.parametrize(
    ("linha", "trecho"),
    [
        (LinhaDoCatalogo("PCA", "Chemometrics", ""), "fora do vocabulário"),
        (LinhaDoCatalogo("PCA", "Chemometrics", "pendente"), "fora do vocabulário"),
        (LinhaDoCatalogo("SAM", "Advanced Significance", "adiada"), "KD3 exclui-o inteiro"),
        (LinhaDoCatalogo("PCA", "Chemometrics", "excluída-KD3"), "lista de isenção"),
        # **A planta que a emenda torna possível, e ela é a metade B dela.** Antes desta
        # linha era uma tabela SÃ; com o grupo fora do conjunto, ela é a violação — «encolhi a
        # constante e esqueci a célula do documento». É o desfecho que a verificação desta unidade
        # nomeia, e ele foi visto na árvore real antes de entrar aqui.
        (
            LinhaDoCatalogo("SVM", "Classification & Feature Selection", "excluída-KD3"),
            "lista de isenção",
        ),
    ],
)
def test_o_portao_do_catalogo_acusa_a_planta(linha: LinhaDoCatalogo, trecho: str) -> None:
    """Entrada sem estado, estado fora do vocabulário, e as duas direções de KD3."""
    achados = defeitos_do_catalogo((linha,), {linha.grupo: (linha.entrada,)})
    assert any(trecho in a for a in achados), achados


def test_o_portao_do_catalogo_nao_acusa_uma_tabela_sa() -> None:
    """A outra ponta: sem isto, um portão que acusasse qualquer linha passaria nas quatro plantas.

    **A segunda linha era `SVM` e passou a ser `SAM`, e a troca é o teste a seguir a
    constante em vez de a contradizer.** Com `Classification & Feature Selection` fora do conjunto
    de grupos excluídos, uma linha daquele grupo marcada `excluída-KD3` é exactamente a **planta**
    da segunda direção, não uma tabela sã: este teste passaria a afirmar o oposto do que o nome
    dele diz, e o par da linha 1126 deixaria de ter contraparte.
    """
    sa = (
        LinhaDoCatalogo("PCA", "Chemometrics", "adiada"),
        LinhaDoCatalogo("SAM", "Advanced Significance", "excluída-KD3"),
        LinhaDoCatalogo("SVM", "Classification & Feature Selection", "construída"),
    )
    grupos = {
        "Chemometrics": ("PCA",),
        "Advanced Significance": ("SAM",),
        "Classification & Feature Selection": ("SVM",),
    }
    assert defeitos_do_catalogo(sa, grupos) == []


@pytest.mark.parametrize(
    ("descobertos", "registados", "construidas", "trecho"),
    [
        ((("volcano", "figura_volcano"),), (), frozenset(), "fora de CONSTRUTORES"),
        (
            (),
            (ConstrutorDeFigura("volcano", "figura_volcano", Vinculo.CATALOGO, "Volcano plot"),),
            frozenset({"Volcano plot"}),
            "sem função em `plots/`",
        ),
        (
            (("volcano", "figura_volcano"),),
            (ConstrutorDeFigura("volcano", "figura_volcano", Vinculo.CATALOGO, "Volcano plot"),),
            frozenset(),
            "não está `construída`",
        ),
        ((), (), frozenset({"PCA"}), "sem construtor"),
        (
            (("deriva", "figura_deriva"),),
            (ConstrutorDeFigura("deriva", "figura_deriva", Vinculo.RELATORIO, "item-42"),),
            frozenset(),
            f"não é um dos {len(ITENS_OBRIGATORIOS)} itens",
        ),
    ],
)
def test_o_portao_da_bijecao_acusa_a_planta(
    descobertos: tuple[tuple[str, str], ...],
    registados: tuple[ConstrutorDeFigura, ...],
    construidas: frozenset[str],
    trecho: str,
) -> None:
    """As cinco direções, uma por execução. A planta é o argumento — negativo em processo."""
    achados = defeitos_da_bijecao(descobertos, registados, construidas)
    assert any(trecho in a for a in achados), achados


def test_o_portao_da_bijecao_nao_acusa_um_par_coerente() -> None:
    """A outra ponta, e ela cobre os DOIS vínculos.

    Sem isto, um portão que acusasse qualquer entrada passaria nas cinco plantas acima.
    """
    par = (
        ConstrutorDeFigura("volcano", "figura_volcano", Vinculo.CATALOGO, "Volcano plot"),
        ConstrutorDeFigura("deriva", "figura_deriva", Vinculo.RELATORIO, "item-3"),
    )
    descobertos = (("deriva", "figura_deriva"), ("volcano", "figura_volcano"))
    assert defeitos_da_bijecao(descobertos, par, frozenset({"Volcano plot"})) == []


def test_o_vocabulario_de_estado_e_o_que_o_documento_usa() -> None:
    """As três palavras do `StrEnum` são as três que a tabela escreve, e não uma tradução delas."""
    texto = CATALOGO_MD.read_text(encoding="utf-8")
    usados = {x.estado for x in estados_do_catalogo(texto)}
    assert usados, "âncora: nenhuma linha de estado lida"
    assert usados <= {e.value for e in EstadoDoCatalogo}


def test_o_registo_tem_a_primeira_figura_e_ela_bijeta_contra_o_RELATORIO() -> None:  # noqa: N802
    """**O andaime morreu aqui, e este é o que fica no lugar dele.**

    O teste anterior afirmava `CONSTRUTORES == ()` e mandava-se apagar quando a primeira figura
    chegasse. Apagá-lo e não pôr nada no lugar deixaria a bijeção a passar sobre um registo que
    ninguém confere ser não vazio — que é a mesma vacuidade com outro nome.

    **A trajetória NÃO vira entrada nenhuma do `[G-1]` para `construída`**, e isso é o conteúdo: a
    trajetória do QC não é figura do MetaboAnalyst. Ela é a metade de trajetória do item 3 de
    `ITENS_OBRIGATORIOS`, e é contra ele que bijeta.

    **Quem encheu a partição `construída` foi o conjunto das figuras**, com três figuras que
    **são** do MetaboAnalyst; o que este teste continua a afirmar é o outro vínculo — que o registo
    tem construtor de relatório e que a chave dele é um item que existe. Os dois vínculos coexistem
    por desenho, e é a âncora 5 de `test_a_bijecao_de_r1_corre_sobre_a_particao_construida` que
    responde pelo lado do catálogo.
    """
    assert CONSTRUTORES, "o registo está vazio: a bijeção de R1 passaria por vacuidade"
    do_relatorio = [c for c in CONSTRUTORES if c.vinculo is Vinculo.RELATORIO]
    assert do_relatorio, f"nenhum construtor bijeta contra o relatório: {CONSTRUTORES}"

    # O item que a chave nomeia existe, e é o da trajetória. Derivado de `ITENS_OBRIGATORIOS` e não
    # do registo — se os dois viessem da mesma fonte, errariam juntos.
    por_chave = {f"item-{i.numero}": i.titulo for i in ITENS_OBRIGATORIOS}
    for construtor in do_relatorio:
        assert construtor.chave in por_chave, construtor
    assert "trajetória dos QCs" in por_chave["item-3"], por_chave["item-3"]

    # **E os PONTOS DE MUDANÇA, que completam o item 3, ficam FORA desta rodada** (KD2). A âncora
    # está aqui para que a leitura *"o item 3 está construído"* não passe por ninguém a ter escrito
    # o contrário: o título pede as duas metades e `plots/` só constrói uma.
    assert "pontos de mudança" in por_chave["item-3"], por_chave["item-3"]
    assert not any("mudanca" in c.modulo or "changepoint" in c.modulo for c in CONSTRUTORES)


# ---------------------------------------------------------------------------
# 8. Camada 3 — `pytest-mpl`, e ele corre de facto
# ---------------------------------------------------------------------------

_MODULO_DE_IMAGEM = """
import numpy as np
import pytest
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

X = np.arange(1.0, 8.0)
Y = np.array([2.0, 3.5, 3.0, 5.0, 4.25, 6.0, 5.5])


def _figura(x, y):
    figura = Figure(figsize=(4.0, 3.0))
    FigureCanvasAgg(figura)
    ax = figura.add_subplot(1, 1, 1)
    ax.plot(x, y, marker="o")
    return figura


@pytest.mark.mpl_image_compare(deterministic=True, filename="prova.png")
def test_limpa():
    return _figura(X, Y)


@pytest.mark.mpl_image_compare(deterministic=True, filename="prova.png")
def test_eixos_trocados():
    return _figura(Y, X)
"""


def _pytest_nu(tmp: Path, *args: str) -> subprocess.CompletedProcess[str]:
    """Um `pytest` sem o `addopts` deste repositório, na pasta temporária."""
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "-q", "-o", "addopts=", *args],
        cwd=tmp,
        capture_output=True,
        text=True,
        check=False,
    )


def test_a_camada_3_de_imagem_acusa_um_eixo_trocado(tmp_path: Path) -> None:
    """`pytest-mpl` existe, corre, e a tolerância PADRÃO separa a figura limpa da trocada.

    **Por que em subprocesso e não por marcador aqui.** `--mpl` não está no `addopts` do projeto, e
    sem ele o marcador `mpl_image_compare` não compara nada — o teste passa e a camada 3 é um
    mecanismo construído e nunca exercitado. Em subprocesso a comparação corre
    dentro do portão de cada commit, e as duas direções são lidas na mesma execução.

    **A tolerância é a padrão de propósito.** Tolerância larga o bastante para sobreviver à
    reescrita de renderização de texto do `matplotlib` 3.11 é larga o bastante para sobreviver a um
    eixo trocado — e este teste é o que o mede em vez de o supor.

    **R14: a camada 3 nunca é a única asserção.** O par causal de eixos trocados da camada 1 fica,
    logo acima, e é ele que diz *qual* série divergiu e em que índice.
    """
    (tmp_path / "test_imagem.py").write_text(_MODULO_DE_IMAGEM, encoding="utf-8")
    base_de_imagens = tmp_path / "baseline"

    gerada = _pytest_nu(
        tmp_path, "test_imagem.py", "-k", "limpa", f"--mpl-generate-path={base_de_imagens}"
    )
    assert gerada.returncode == 0, gerada.stdout + gerada.stderr
    assert (base_de_imagens / "prova.png").is_file(), (
        "âncora: `--mpl-generate-path` não escreveu a imagem de referência — a comparação abaixo "
        "seria sobre nada"
    )

    comparada = _pytest_nu(
        tmp_path, "test_imagem.py", "--mpl", f"--mpl-baseline-path={base_de_imagens}"
    )
    assert "1 failed" in comparada.stdout and "1 passed" in comparada.stdout, (
        f"esperava a limpa a passar e a trocada a reprovar:\n{comparada.stdout}"
    )
    assert "test_eixos_trocados" in comparada.stdout, comparada.stdout


# ---------------------------------------------------------------------------
# 9. Os regimes declarados, conferidos
# ---------------------------------------------------------------------------


def test_o_pacote_esta_na_lista_do_mypy_estrito() -> None:
    """`tests/test_mypy_strict_parity.py` confere os FLAGS e nunca a lista de módulos.

    Então a lista precisa de quem a confira, e este é o portão dessa metade para `plots/`.
    """
    import tomllib

    configuracao = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    modulos: set[str] = set()
    for override in configuracao["tool"]["mypy"]["overrides"]:
        if override.get("disallow_untyped_defs") is True:
            modulos.update(override["module"])
    assert {"refino.plots", "refino.plots.*"} <= modulos, (
        f"`plots/` fora do override estrito; a lista é {sorted(modulos)}"
    )


def test_o_pino_do_pytest_mpl_esta_declarado() -> None:
    """`tests/test_dependency_pins.py` exige que todo pino DECLARADO bata com o instalado, e é cego
    ao inverso — instalar sem declarar não reprova. Declarar é o que torna a camada 3 verificável.
    """
    import tomllib

    configuracao = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    dev = configuracao["project"]["optional-dependencies"]["dev"]
    assert any(p.startswith("pytest-mpl==") for p in dev), dev


def test_os_modulos_do_pacote_sao_os_que_o_init_reexporta() -> None:
    """Âncora contra módulo novo que nasça fora do `__all__` — e sem `no_implicit_reexport` a
    ausência não se manifestaria como erro."""
    assert {c.stem for c in FONTES_DO_PACOTE} == {
        "__init__",
        "agrupamento",
        "anova",
        "base",
        "catalogo",
        "classificacao",
        "composicao",
        "correcao",
        "correlacao",
        "curva_do_loess",
        "deriva",
        "dispersao",
        "distribuicao",
        "pca",
        "plsda",
        "plsda_multiclasse",
        "recorte",
        "rede",
        "series",
        "univariada",
        "validacao",
        "volcano",
    }
    for modulo in (
        agrupamento,
        anova,
        base,
        catalogo,
        classificacao,
        composicao,
        correcao,
        correlacao,
        curva_do_loess,
        deriva,
        dispersao,
        distribuicao,
        pca,
        plsda,
        plsda_multiclasse,
        recorte,
        plots_series,
        univariada,
        validacao,
        volcano,
    ):
        for nome in modulo.__all__:
            assert hasattr(modulo, nome), f"{modulo.__name__}.{nome} está em __all__ e não existe"


def test_a_vista_lida_e_derivada_e_nao_literal() -> None:
    """A classe (c) declara o que leu, e o teste do arnês pode inspecioná-la."""
    prova = figura_de_prova()
    lida = vista(prova.figura.axes[0])
    assert lida.xscale == "linear" and lida.yscale == "linear"
    assert lida.visiveis == (True, True)
    assert lida.ylim[0] < prova.linha.min() and lida.ylim[1] > prova.linha.max()


def test_os_numeros_declarados_saem_do_artista_e_nao_do_recorte() -> None:
    prova = figura_de_prova()
    assert "6.035" in numeros_declarados(prova.figura)
    prova.figura.texts[0].set_text("nada")
    assert "6.035" not in numeros_declarados(prova.figura)


# ---------------------------------------------------------------------------
# 8. A vírgula decimal do eixo, e o requebrar do que não cabe
#
# **As três coisas desta seção são invisíveis às quatro classes**, e é por isso que
# sobreviveram a onze unidades: em todas o dado está certo, os rótulos estão lá, os limites contêm
# os pontos e o número declarado bate. O que está errado é a CONVENÇÃO com que o número é escrito, e
# o sítio onde o texto cai. Foram vistas a olho, na imagem montada, e o que fica aqui é a
# medida que impede a volta.
# ---------------------------------------------------------------------------


def _figura_com_marcas_fracionarias() -> Figure:
    """Uma figura cujas marcas de eixo TÊM casas decimais. A âncora desta seção.

    **Uma planta de vírgula sobre marcas inteiras não separa nada:** `-2`, `0` e `2` são idênticos
    com ponto e com vírgula, e um teste sobre eles passaria com o formatador desligado. Daí os
    valores serem milésimos, e daí a asserção de que há ponto ANTES de trocar.
    """
    figura = nova_figura()
    eixo = figura.add_subplot(1, 1, 1)
    eixo.plot([0.001, 0.002, 0.003], [-0.015, 0.0, 0.015], label="marcas fracionárias")
    return figura


def _marcas(figura: Figure) -> list[str]:
    figura.canvas.draw()
    return [
        t.get_text()
        for eixo in figura.axes
        for escala in (eixo.xaxis, eixo.yaxis)
        for t in escala.get_ticklabels()
        if t.get_text().strip()
    ]


def test_a_virgula_decimal_troca_o_ponto_das_MARCAS_nas_duas_pontas() -> None:  # noqa: N802
    """O par causal da vírgula: sem ela há ponto decimal, com ela não há, e o menos sobrevive.

    **A ponta A é uma ÂNCORA e não cerimônia**: se a figura de prova tivesse marcas inteiras,
    a ponta B passaria com a função desligada, e este teste seria a verificação vazia mais
    convincente que existe.
    """
    crua = _figura_com_marcas_fracionarias()
    antes = _marcas(crua)
    assert [m for m in antes if re.search(r"\d\.\d", m)], (
        f"âncora: esta prova não tem marca fracionária nenhuma — {antes}"
    )

    trocadas = base.virgula_decimal(crua)
    assert trocadas > 0, "a função não tocou em escala nenhuma"
    depois = _marcas(crua)
    assert not [m for m in depois if re.search(r"\d\.\d", m)], (
        f"sobrou ponto decimal depois da troca: {depois}"
    )
    assert [m for m in depois if re.search(r"\d,\d", m)], f"nenhuma vírgula apareceu: {depois}"
    assert chr(0x2212) in "".join(depois), (
        "o sinal de menos U+2212 desapareceu — a troca mexeu em mais do que o ponto"
    )


def test_a_virgula_decimal_NAO_toca_em_rotulo_CATEGORICO() -> None:  # noqa: N802
    """A metade que protege o dado: nome de feature leva ponto, e ele NÃO é decimal.

    **Este é o defeito que a função existe para não cometer.** O eixo y do VIP imprime
    `21.37_508.2876m/z` por um `FuncFormatter`; trocar ali fabricaria `21,37_508,2876m/z`, que não é
    o nome de nada — um identificador inventado, impresso com a autoridade de uma marca de eixo.
    Sem este teste, a direção destrutiva de `virgula_decimal` nunca foi exercitada.
    """
    figura = nova_figura()
    eixo = figura.add_subplot(1, 1, 1)
    nomes = ["21.37_508.2876m/z", "2.25_810.7739m/z"]
    eixo.plot([0.0, 1.0], [0.0, 1.0], label="prova")
    eixo.set_yticks([0.0, 1.0])
    eixo.set_yticklabels(nomes)

    base.virgula_decimal(figura)
    desenhadas = _marcas(figura)
    for nome in nomes:
        assert nome in desenhadas, (
            f"o nome de feature {nome!r} foi reescrito pela vírgula decimal: {desenhadas}. "
            "Nome não é número, e o ponto dentro dele não é separador decimal"
        )


def test_o_titulo_de_eixo_ESTREITO_requebra_e_o_de_eixo_largo_NAO() -> None:  # noqa: N802
    """O par causal do requebrar, e as duas pontas correm sobre a MESMA frase.

    **É a âncora na forma mais direta que esta unidade tem:** a única coisa que difere entre
    as duas pontas é quantas colunas de eixos a figura tem. Se a frase quebrasse nas duas, ou em
    nenhuma, o teste não separaria a largura de coisa nenhuma.
    """
    frase = "50 caixas; 16.566 ponto(s) fora dos bigodes, contados e não desenhados"

    larga = nova_figura()
    um_eixo = larga.add_subplot(1, 1, 1)
    um_eixo.plot([0.0, 1.0], [0.0, 1.0])
    um_eixo.set_title(frase, fontsize=recorte.TAMANHO_DA_FONTE)
    aplica(larga, Recorte(50, POPULACAO, "sem amostragem", "bruta", "Tukey"))

    estreita = nova_figura()
    esquerdo, direito = estreita.subplots(1, 2)
    for eixo in (esquerdo, direito):
        eixo.plot([0.0, 1.0], [0.0, 1.0])
        eixo.set_title(frase, fontsize=recorte.TAMANHO_DA_FONTE)
    aplica(estreita, Recorte(50, POPULACAO, "sem amostragem", "bruta", "Tukey"))

    assert "\n" not in sem_a_matriz(um_eixo.get_title()), (
        f"a frase quebrou num eixo que ocupa a figura inteira: {um_eixo.get_title()!r}. "
        "Quebrar o que cabe é o defeito simétrico — o título fica mais alto sem precisar"
    )
    for lado, eixo in (("esquerdo", esquerdo), ("direito", direito)):
        assert "\n" in eixo.get_title(), (
            f"o título do eixo {lado} não quebrou: {eixo.get_title()!r}. Numa figura de dois lados "
            "esta frase é mais larga que o eixo, e sem quebra ela colide com a do outro lado"
        )
    assert " ".join(sem_a_matriz(esquerdo.get_title()).split()) == frase, (
        "o requebrar mudou o TEXTO e não só as quebras — ele quebra, nunca trunca"
    )


def test_o_titulo_requebrado_CABE_na_coluna_de_eixos_que_lhe_toca() -> None:  # noqa: N802
    """A propriedade que interessa não é «quebrou», é «cabe» — e mede-se no renderizador.

    Ler `"\\n" in titulo` diz que houve quebra e não diz que ela bastou. Aqui a caixa desenhada de
    cada título é comparada com a metade da figura que aquele eixo ocupa, que é o espaço que ele tem
    antes de tocar no vizinho.
    """
    figura = nova_figura()
    esquerdo, direito = figura.subplots(1, 2)
    for eixo in (esquerdo, direito):
        eixo.plot([0.0, 1.0], [0.0, 1.0])
        eixo.set_title(
            "50 caixas; 16.566 ponto(s) fora dos bigodes, contados e não desenhados",
            fontsize=recorte.TAMANHO_DA_FONTE,
        )
    aplica(figura, Recorte(50, POPULACAO, "sem amostragem", "bruta", "Tukey"))
    figura.canvas.draw()

    desenhador = figura.canvas.get_renderer()
    metade = figura.bbox.width / 2.0
    for lado, eixo in (("esquerdo", esquerdo), ("direito", direito)):
        largura = eixo.title.get_window_extent(desenhador).width
        assert largura <= metade, (
            f"o título do eixo {lado} mede {largura:.0f} px e a coluna dele tem {metade:.0f} px: "
            "ele entra no espaço do vizinho, que é a colisão que o requebrar existe para evitar"
        )


def test_o_requebrar_conta_o_que_TOCOU_e_devolve_zero_quando_nada_precisa() -> None:  # noqa: N802
    """A terceira direção do requebrador: ele sabe dizer que não fez nada.

    Contar **execuções do caminho** e não testes que passam é o que separa um requebrador que
    funciona de um que nunca toca em nada — os dois deixam a figura bonita quando ela já cabia.
    """
    figura = nova_figura()
    eixo = figura.add_subplot(1, 1, 1)
    eixo.plot([0.0, 1.0], [0.0, 1.0])
    eixo.set_title("curto", fontsize=recorte.TAMANHO_DA_FONTE)
    aplica(figura, Recorte(1, POPULACAO, "sem amostragem", "bruta", "nenhum"))
    assert recorte.requebra_o_que_couber(figura) == 0, (
        "o requebrador mexeu num título que já cabia — ele tem de ser idempotente, senão o laço de "
        "`aplica` nunca assenta"
    )

    eixo.set_title(
        "uma frase deliberadamente comprida que não cabe em coluna nenhuma desta figura porque "
        "tem muitas palavras seguidas e nenhuma delas é curta o bastante para ajudar",
        fontsize=recorte.TAMANHO_DA_FONTE,
    )
    assert recorte.requebra_o_que_couber(figura) == 1, (
        "o requebrador devolveu zero sobre um título que não cabe — a contagem é o único sinal de "
        "que ele correu, e uma contagem sempre nula é indistinguível de um mecanismo morto"
    )


def test_o_texto_MARCADO_e_requebrado_e_o_nao_marcado_fica_intacto() -> None:  # noqa: N802
    """O par causal da marca: a mesma frase, no mesmo eixo, com e sem `GID_REQUEBRAVEL`.

    **A marca existe porque requebrar prosa alheia é o defeito simétrico.** Uma anotação que o
    clique deixou, ou um rótulo posto sobre um ponto, não devem ser reflowados por um auxiliar de
    rodapé — quem quer ser requebrado diz que quer.
    """
    frase = (
        "a covariância destes pontos é singular ou não finita: eles caem sobre uma reta (ou num "
        "ponto só), e a região degenera num segmento"
    )
    figura = nova_figura()
    eixo = figura.add_subplot(1, 1, 1)
    eixo.plot([0.0, 1.0], [0.0, 1.0])
    marcado = eixo.text(0.02, 0.98, frase, transform=eixo.transAxes, va="top")
    marcado.set_gid(base.GID_REQUEBRAVEL)
    solto = eixo.text(0.02, 0.50, frase, transform=eixo.transAxes, va="top")

    aplica(figura, Recorte(1, POPULACAO, "sem amostragem", "bruta", "nenhum"))

    assert "\n" in marcado.get_text(), "o texto marcado não foi requebrado"
    assert solto.get_text() == frase, (
        f"o texto SEM marca foi requebrado: {solto.get_text()!r}. O auxiliar do recorte não pode "
        "reflowar prosa que não se declarou requebrável"
    )
    assert " ".join(marcado.get_text().split()) == frase, "o requebrar truncou em vez de quebrar"


def test_o_texto_marcado_CABE_no_eixo_que_a_legenda_encolheu() -> None:  # noqa: N802
    """A geometria que o motivou: legenda fora da área de dado, e o texto cabe no que sobra.

    **É o caso em que uma quebra feita no construtor sairia errada** — ali o eixo ainda tem a
    largura de origem, e a legenda só o encolhe no `tight_layout` que vem depois. Medido em
    11/09/2026: entre as duas larguras há quase o dobro.
    """
    frase = (
        "a elipse não se calcula com 1 ponto(s) finito(s): são precisos ao menos 3, porque com "
        "n = p a covariância amostral é singular por construção"
    )
    figura = nova_figura()
    eixo = figura.add_subplot(1, 1, 1)
    eixo.plot([0.0, 1.0], [0.0, 1.0], label="uma série com um rótulo comprido para a legenda")
    eixo.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0)
    marcado = eixo.text(0.02, 0.98, frase, transform=eixo.transAxes, va="top")
    marcado.set_gid(base.GID_REQUEBRAVEL)

    aplica(figura, Recorte(1, POPULACAO, "sem amostragem", "bruta", "nenhum"))
    figura.canvas.draw()

    desenhador = figura.canvas.get_renderer()
    caixa_do_texto = marcado.get_window_extent(desenhador)
    caixa_do_eixo = eixo.get_window_extent(desenhador)
    assert caixa_do_texto.x1 <= caixa_do_eixo.x1, (
        f"o texto acaba em {caixa_do_texto.x1:.0f} px e o eixo em {caixa_do_eixo.x1:.0f}: ele sai "
        "da área de dado, que é onde a legenda está"
    )
    legenda = eixo.get_legend().get_window_extent(desenhador)
    assert not caixa_do_texto.overlaps(legenda), (
        "o texto do motivo sobrepõe-se à legenda — foi assim que ele saiu na primeira renderização"
    )


# ---------------------------------------------------------------------------
# 9. A EXTENSÃO DA CLASSE (a) — imagem e segmentos
#
# **Os dois pares causais desta seção foram vistos NÃO separar antes da extensão**, e é isso que os
# torna atribuíveis. Medido em 13/09/2026, contra o arnês de então, com o mesmo par de conteúdos:
#
# | Artista | O que a classe (a) lia | As duas leituras diferiam? |
# |---|---|---|
# | `AxesImage` de `imshow` | `()` — `ax.images` não era percorrido | **não** |
# | `LineCollection` de 250 arestas | uma série com **1** ponto em `(0, 0)` | **não** |
#
# Depois: a imagem lê `3 x 4` células e o feixe lê **500** vértices, e as duas leituras divergem.
# ---------------------------------------------------------------------------

# O conteúdo do mapa de prova. **`3 x 4` e não quadrado, de propósito:** uma matriz quadrada faz a
# transposta ter a mesma forma, e a mutação mais provável desta família — trocar features por
# amostras — passaria na conferência de forma.
CONTEUDO_DO_MAPA: Final = np.arange(12.0).reshape(3, 4)

# Quantas arestas o feixe de prova desenha. Bastante para que *"uma série de um ponto"* não se possa
# ler como arredondamento: são 500 vértices contra 1.
ARESTAS_DO_FEIXE: Final = 250


def figura_de_mapa(conteudo: np.ndarray = CONTEUDO_DO_MAPA) -> Figure:
    """Um `imshow` rotulado. O artista é um `AxesImage`, que não é linha nem coleção."""
    figura = nova_figura()
    ax = figura.add_subplot(1, 1, 1)
    ax.imshow(conteudo, aspect="auto", label="mapa")
    return figura


def segmentos_de_prova(fator: float = 1.0) -> np.ndarray:
    """`ARESTAS_DO_FEIXE` arestas, cada uma com dois vértices. `fator` escala o conteúdo inteiro."""
    passo = np.arange(ARESTAS_DO_FEIXE, dtype=float)
    inicio = np.column_stack((passo, passo * 2.0))
    fim = np.column_stack((passo + 1.0, passo * 3.0))
    return np.stack([inicio, fim], axis=1) * fator


def figura_de_feixe(segmentos: np.ndarray) -> Figure:
    """Um feixe de segmentos, que é como um dendrograma e uma rede se desenham."""
    figura = nova_figura()
    ax = figura.add_subplot(1, 1, 1)
    ax.add_collection(LineCollection(list(segmentos), label="arestas"))
    ax.autoscale_view()
    return figura


def test_a_classe_a_LE_o_conteudo_de_uma_imagem_e_nao_so_o_retangulo() -> None:  # noqa: N802
    """Par causal da imagem: **o retângulo é o mesmo e a leitura difere**, porque o conteúdo mudou.

    **A ponta do retângulo é uma âncora e não cerimônia:** se ele também mudasse, a
    divergência seria atribuível à posição da imagem e não ao conteúdo — que é justamente a metade
    que o arnês não via. Os dois conteúdos são o mesmo `imshow` no mesmo eixo.
    """
    limpa = figura_de_mapa()
    trocada = figura_de_mapa(CONTEUDO_DO_MAPA[::-1, ::-1] * 10.0)
    lida, outra = series(limpa.axes[0])[0], series(trocada.axes[0])[0]

    assert lida.valores is not None, "a classe (a) não leu o conteúdo do `AxesImage`"
    assert lida.valores.shape == CONTEUDO_DO_MAPA.shape, lida.valores.shape
    assert np.array_equal(lida.x, outra.x) and np.array_equal(lida.y, outra.y), (
        "âncora: o retângulo tem de ser o mesmo nas duas, senão a divergência não é do conteúdo"
    )
    assert not np.array_equal(lida.valores, outra.valores), (
        "as duas leituras do artista são IDÊNTICAS com o conteúdo trocado — é a cegueira que a "
        "U18 mediu contra o arnês de 12/09/2026, e que esta extensão existe para fechar"
    )


def test_par_causal_do_conteudo_trocado() -> None:
    """A quinta planta, e a classe (a) acusa-a. **As outras quatro não lhe tocam.**

    A planta muda a matriz e deixa o retângulo; `confere_series` — que compara coordenadas — não vê
    diferença nenhuma, e a asserção de conteúdo vê.
    """
    figura = figura_de_mapa()
    ax = figura.axes[0]
    lida = series(ax)[0]
    retangulo = {"mapa": (lida.x.copy(), lida.y.copy())}

    trocado = planta_conteudo_trocado(ax)

    assert trocado == "mapa"
    assert confere_series(ax, retangulo) == [], (
        "a planta do conteúdo mexeu em coordenadas — deixaria de ser a direção que as outras "
        "quatro não alcançam"
    )
    achados = confere_conteudo(ax, {"mapa": CONTEUDO_DO_MAPA})
    assert achados, "o arnês não viu o conteúdo do mapa mudar"
    assert "célula(s) divergem" in achados[0], achados


def test_a_conferencia_de_conteudo_nao_acusa_a_imagem_limpa() -> None:
    """A outra ponta: com a planta removida, ela deixa de acusar.

    Sem esta metade, uma conferência que acusasse **tudo** passaria no teste acima.
    """
    figura = figura_de_mapa()
    assert confere_conteudo(figura.axes[0], {"mapa": CONTEUDO_DO_MAPA}) == []


def test_a_conferencia_de_conteudo_acusa_a_TRANSPOSTA() -> None:  # noqa: N802
    """A mutação mais provável de um mapa de calor: features e amostras trocadas.

    **Ela é apanhada pela FORMA e não pelos valores**, e é por isso que o mapa de prova é `3 x 4` e
    não quadrado. Num mapa quadrado a transposta tem a mesma forma, e a asserção passaria a depender
    de os valores serem assimétricos — que é uma propriedade do dado e não do teste.
    """
    figura = figura_de_mapa(CONTEUDO_DO_MAPA.T.copy())
    achados = confere_conteudo(figura.axes[0], {"mapa": CONTEUDO_DO_MAPA})
    assert achados, "o arnês não viu o mapa transposto"
    assert "forma" in achados[0], achados


def test_a_assercao_de_ausencia_deixou_de_passar_sobre_uma_figura_de_imagem() -> None:
    """`confere_series(ax, {})` sobre um mapa de calor com conteúdo. **Antes devolvia `[]`.**

    É a asserção vácua na superfície mais convidativa que existe: a asserção de ausência é
    satisfeita por **vacuidade** quando o instrumento não lê o artista, e o desfecho é o de sucesso.
    """
    figura = figura_de_mapa()
    achados = confere_series(figura.axes[0], {})
    assert achados, "a asserção de ausência voltou a passar por vacuidade sobre uma imagem"
    assert "mapa" in achados[0], achados


def test_a_classe_a_LE_os_segmentos_e_nao_o_par_nulo_da_origem() -> None:  # noqa: N802
    """Par causal dos segmentos: 250 arestas não leem como **uma série de um ponto**.

    **O defeito era o veredito do instrumento em estado puro:** `Collection.get_offsets()` de uma
    `LineCollection` sem deslocamento devolve `[[0, 0]]`, o arnês registava *"um ponto na origem"*,
    e a frase era sobre o **alvo** quando a causa estava no instrumento. `0` e `1` leem-se como
    medida.
    """
    figura = figura_de_feixe(segmentos_de_prova())
    lida = series(figura.axes[0])[0]

    assert lida.x.size == 2 * ARESTAS_DO_FEIXE, (
        f"o arnês leu {lida.x.size} ponto(s) de um feixe com {ARESTAS_DO_FEIXE} arestas "
        f"({2 * ARESTAS_DO_FEIXE} vértices)"
    )
    assert not np.allclose(lida.x, 0.0), "o arnês está a ler o par nulo dos deslocamentos"

    outra = series(figura_de_feixe(segmentos_de_prova(fator=7.0)).axes[0])[0]
    assert not np.allclose(lida.x, outra.x), (
        "as duas leituras do feixe são IDÊNTICAS com o conteúdo trocado"
    )


def test_as_plantas_de_dado_alcancam_um_feixe_de_segmentos() -> None:
    """As duas plantas de mutação de dado sobre uma `LineCollection`, e não sobre linhas.

    **Sem esta metade a extensão criaria uma classe nova de planta inerte:** as plantas mexiam em
    `set_offsets`, que numa coleção de segmentos não toca o que está desenhado — a leitura passaria
    a ser dos segmentos e a mutação a ser dos deslocamentos, e o par causal deixaria de separar sem
    que nada acusasse.
    """
    figura = figura_de_feixe(segmentos_de_prova())
    ax = figura.axes[0]
    antes = series(ax)[0]
    esperado = {"arestas": (antes.x.copy(), antes.y.copy())}

    planta_sinal_invertido(ax)
    achados = confere_series(ax, esperado)
    assert achados, "o arnês não viu o sinal do feixe invertido"
    assert all("eixo y" in a for a in achados), achados

    outra = figura_de_feixe(segmentos_de_prova())
    planta_eixos_trocados(outra.axes[0])
    assert confere_series(outra.axes[0], esperado), "o arnês não viu os eixos do feixe trocados"


def test_a_planta_do_conteudo_recusa_uma_figura_sem_imagem() -> None:
    """A âncora desta planta: sem imagem, não há conteúdo que trocar."""
    prova = figura_de_prova()
    with pytest.raises(PlantaQueNaoSeparaError, match="não há imagem"):
        planta_conteudo_trocado(prova.figura.axes[0])


def test_as_plantas_de_coordenada_recusam_uma_figura_so_de_imagem() -> None:
    """A direção simétrica: trocar os eixos de uma imagem é transpô-la, que é outra mutação.

    **Antes desta guarda o desfecho era `IndexError`** — `ax.collections[0]` sobre eixos que só têm
    imagem —, e uma reprovação por `IndexError` não é atribuível à planta.
    """
    for planta in (planta_eixos_trocados, planta_sinal_invertido):
        figura = figura_de_mapa()
        with pytest.raises(PlantaQueNaoSeparaError, match="imagem"):
            planta(figura.axes[0])


def test_a_classe_c_conta_os_MESMOS_artistas_que_a_classe_a() -> None:  # noqa: N802
    """`confere_vista` faz `zip(..., strict=True)` das duas leituras, e isto é o contrato.

    Um grupo de artistas lido por `series` e não por `vista` levanta `ValueError` em toda figura que
    o tenha; a asserção aqui é sobre as três superfícies de uma vez — linha, feixe e imagem.
    """
    figura = nova_figura()
    ax = figura.add_subplot(1, 1, 1)
    ax.plot([0.0, 1.0], [0.0, 1.0], label="linha")
    ax.add_collection(LineCollection(list(segmentos_de_prova()), label="arestas"))
    ax.imshow(CONTEUDO_DO_MAPA, aspect="auto", label="mapa")

    lida = vista(ax)
    assert len(series(ax)) == 3, [s.rotulo for s in series(ax)]
    assert len(lida.visiveis) == 3 and len(lida.alphas) == 3, lida
    # O `zip(strict=True)` de `confere_vista` é a asserção: se as duas leituras contassem artistas
    # diferentes, esta chamada levantaria `ValueError` em vez de devolver uma lista.
    assert isinstance(confere_vista(ax), list)


# ---------------------------------------------------------------------------
# 10. O RÓTULO DO SELETOR é propriedade do REGISTO INTEIRO
#
# **Ele vivia em `tests/test_desktop_figuras.py`, onde precisa de Qt e corre sobre um painel.** A
# unicidade é propriedade do registo e de mais nada: `rotulo_do_construtor` é uma função pura de
# `plots/catalogo.py`, e conferi-la aqui custa zero por figura — contra um cenário por unidade, que
# correria só sobre as figuras que a unidade toca.
# ---------------------------------------------------------------------------


def test_os_rotulos_do_seletor_sao_UNICOS_sobre_o_registo_inteiro() -> None:  # noqa: N802
    """Dezasseis entradas, dezasseis rótulos — e com o par `(modulo, chave)` eram OITO.

    > **Um rótulo derivado pode ser derivado do campo errado.** A derivação protege contra a lista
    > que caduca; não diz nada sobre a **unicidade** do que se derivou.

    **A âncora vem primeiro:** com registo vazio a asserção de distinção passaria por
    vacuidade, e registo vazio é exactamente o estado inicial.
    """
    assert CONSTRUTORES, "âncora: o registo está vazio e a asserção abaixo seria vácua"

    rotulos = [rotulo_do_construtor(c.funcao) for c in CONSTRUTORES]
    repetidos = sorted({r for r in rotulos if rotulos.count(r) > 1})
    assert not repetidos, (
        f"{len(rotulos) - len(set(rotulos))} entrada(s) do seletor não se distinguem de outra: "
        f"{repetidos}. Quem abre o seletor vê linhas iguais e não consegue pedir uma figura "
        "específica"
    )
    assert len(set(rotulos)) == len(CONSTRUTORES), (
        f"{len(CONSTRUTORES)} construtores e {len(set(rotulos))} rótulos distintos"
    )
    assert all(r.strip() for r in rotulos), f"algum rótulo saiu em branco: {rotulos}"


def test_o_portao_do_rotulo_unico_acusa_dois_nomes_que_COLAPSAM() -> None:  # noqa: N802
    """Par causal **em processo**: a planta é o argumento, e ela colapsa pela derivação.

    `figura_do_mapa` e `figura_da_mapa` são funções diferentes, em módulos diferentes, e as duas
    saem daqui como *«mapa»* — porque o artigo é o que a derivação tira. É exactamente a forma do
    defeito, onde quatro construtores de PCA colapsavam no mesmo rótulo.
    """
    plantados = (
        ConstrutorDeFigura("agrupamento", "figura_do_mapa", Vinculo.CATALOGO, "Heatmap"),
        ConstrutorDeFigura("correlacao", "figura_da_mapa", Vinculo.CATALOGO, "Correlations"),
    )
    rotulos = [rotulo_do_construtor(c.funcao) for c in plantados]
    assert len(set(rotulos)) == 1, (
        f"a planta não colapsa: {rotulos}. Sem colapso este negativo passaria com o portão "
        "desligado"
    )


def test_a_derivacao_do_rotulo_tira_o_prefixo_e_o_ARTIGO_da_familia() -> None:  # noqa: N802
    """Os cinco artigos, e a lista é da FAMÍLIA e não das instâncias.

    **`de_` é o que prova a diferença**: nenhuma das treze funções do registo o usa, e derivar a
    lista das instâncias à frente deixaria `figura_de_arnes` a sair como *«de arnes»*.
    """
    assert set(ARTIGOS_DO_CONSTRUTOR) == {"do_", "da_", "de_", "dos_", "das_"}
    assert rotulo_do_construtor("figura_de_arnes") == "arnes"
    assert rotulo_do_construtor("figura_dos_scores_do_plsda") == "scores do plsda"
    assert rotulo_do_construtor("figura_das_caixas_por_feature") == "caixas por feature"
    # A direção que se «consertaria» por engano: `dos_` NÃO é casado por `do_`, porque o sublinhado
    # faz parte do padrão. Sem isto, `dos_scores` sairia como `s scores`.
    assert rotulo_do_construtor("figura_dos_scores") == "scores"


def test_as_familias_do_seletor_derivam_da_CHAVE_e_nao_de_uma_segunda_lista() -> None:  # noqa: N802
    """R23: o agrupamento sai do campo que o registo já tem, e a partição é TOTAL.

    **As duas âncoras vivem aqui:** o registo não está vazio, e há ao menos uma família com
    mais de um membro — sem a segunda, um agrupamento que devolvesse uma família por construtor
    passaria em tudo o que se afirma abaixo.
    """
    assert CONSTRUTORES, "âncora: o registo está vazio"
    familias = familias_do_seletor(CONSTRUTORES)

    assert sum(len(m) for _, m in familias) == len(CONSTRUTORES), (
        f"a partição não é total: {sum(len(m) for _, m in familias)} membros para "
        f"{len(CONSTRUTORES)} construtores"
    )
    assert [c for _, membros in familias for c in membros] == list(CONSTRUTORES), (
        "o agrupamento reordenou o registo; a ordem dentro da família é a ordem em que se olha a "
        "figura, e reordenar troca uma ordem com significado por uma sem"
    )
    assert {chave for chave, _ in familias} == {c.chave for c in CONSTRUTORES}
    assert len(familias) == len({c.chave for c in CONSTRUTORES})
    assert any(len(m) > 1 for _, m in familias), (
        "âncora: nenhuma família tem mais de um membro, e o agrupamento não está a agrupar nada"
    )


def test_o_agrupamento_NAO_e_uma_lista_de_familias_escrita_a_mao() -> None:  # noqa: N802
    """Par causal em processo: uma chave nova aparece como família nova, sem ninguém a registar.

    É a metade que uma lista escrita à mão não teria — lá a figura nova cairia no grupo errado, ou
    em grupo nenhum, e nada acusaria.
    """
    plantados = (
        ConstrutorDeFigura("m", "figura_a", Vinculo.CATALOGO, "Heatmap"),
        ConstrutorDeFigura("m", "figura_b", Vinculo.CATALOGO, "Dendrogram"),
        ConstrutorDeFigura("m", "figura_c", Vinculo.CATALOGO, "Heatmap"),
    )
    familias = familias_do_seletor(plantados)
    assert [chave for chave, _ in familias] == ["Heatmap", "Dendrogram"], familias
    assert [c.funcao for c in familias[0][1]] == ["figura_a", "figura_c"], familias
    assert familias_do_seletor(()) == ()


# ---------------------------------------------------------------------------
# 11. As extrações não moveram número — as âncoras contra as cópias substituídas
#
# **A propriedade que se quer é *«saída idêntica»*, e ela é afirmada contra a implementação
# ANTERIOR reescrita aqui**, não contra a nova. Chamar a função nova dos dois lados seria o
# instrumento a certificar-se.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "entrada",
    [None, [], [1.0, 2.0], [1, None, 3], (4.0, 5.0), np.array([[1.0, 2.0], [3.0, 4.0]]), "texto"],
)
def test_o_vetor_partilhado_devolve_o_MESMO_que_a_copia_que_substituiu(entrada: object) -> None:  # noqa: N802
    """`pca._vetor` e `plsda._vetor` eram verbatim; esta é a implementação dos dois, reescrita."""

    def anterior(valores: object) -> np.ndarray:
        if isinstance(valores, np.ndarray):
            return base.vetor_do_fio(valores.ravel().tolist())
        if isinstance(valores, (list, tuple)):
            return base.vetor_do_fio(list(valores))
        return np.empty(0, dtype=float)

    obtido = base.vetor_do_fio_ou_nucleo(entrada)
    esperado = anterior(entrada)
    assert obtido.shape == esperado.shape
    assert np.array_equal(obtido, esperado, equal_nan=True)


@pytest.mark.parametrize("entrada", [None, "", "   ", "bruta", 12, True, ["x"]])
def test_o_texto_partilhado_devolve_o_MESMO_que_a_copia_que_substituiu(entrada: object) -> None:  # noqa: N802
    """`volcano._texto` e `distribuicao._texto` eram verbatim; esta é a implementação dos dois."""

    def anterior(valor: object, recuo: str) -> str:
        return valor if isinstance(valor, str) and valor.strip() else recuo  # type: ignore[return-value]

    assert base.texto_do_fio(entrada, base.NAO_DECLARADA) == anterior(entrada, base.NAO_DECLARADA)


def test_o_recuo_da_procedencia_e_UM_e_o_mesmo_nos_dois_modulos() -> None:  # noqa: N802
    """A constante estava escrita duas vezes, com o comentário de uma a nomear a outra.

    **Duas cópias do mesmo texto divergem no dia em que uma for reescrita**, e o sintoma seriam duas
    figuras a declarar ausência com palavras diferentes — que se lê como duas ausências diferentes.
    """
    assert volcano.NAO_DECLARADA is base.NAO_DECLARADA
    assert distribuicao.NAO_DECLARADA is base.NAO_DECLARADA


@pytest.mark.parametrize(
    ("desenhadas", "totais"), [(6035, 6035), (5942, 6035), (0, 1), (1, 6035), (6034, 6035)]
)
def test_a_frase_do_indefinido_contado_devolve_o_MESMO_que_as_copias(  # noqa: N802
    desenhadas: int, totais: int
) -> None:
    """As três frases da família, contra as três implementações anteriores reescritas aqui.

    > **O RAMO DO UM foi MUDADO de propósito, e a mudança fica declarada aqui em vez de
    > o teste ser afrouxado.** As reescritas abaixo eram as implementações de antes, e
    > elas tinham **dois** ramos: com `quantas == 1` escreviam `as 1 descartadas`, que é defeito.
    > Este teste passou a **defender** esse defeito no instante em que ele foi consertado.
    >
    > **A resolução é ADITIVA e não substitutiva:** as reescritas continuam a ser as de
    > antes, e o que se acrescenta é o ramo do um com a redação NOVA ao lado da antiga — de modo que
    > a equivalência continua afirmada em **zero** e em **muitos**, que são os ramos
    > extraídos e não tocados, e a única diferença que sobra é a declarada aqui.
    >
    > **Os dois casos novos existem porque os três antigos não alcançavam o ramo do um** — `(1,
    > 6035)` põe `desenhadas == 1` na frase do VIP, e `(6034, 6035)` põe `fora == 1` nas duas
    > frases da PCA. Sem eles, esta função passaria com o ramo do um a devolver seja o que for.
    """
    fora = totais - desenhadas

    def como_era(quantas: int, *, nenhuma: str, algumas: str) -> str:
        """A ramificação de DOIS ramos, exactamente como as três cópias a escreviam antes."""
        return nenhuma if quantas == 0 else f"as {recorte.milhar(quantas)} {algumas}"

    anterior_amostragem = como_era(
        fora,
        nenhuma="nenhuma foi descartada pelo passo 3",
        algumas="descartadas pelo passo 3 são constantes ou não finitas",
    )
    agora_amostragem = (
        "1 descartada pelo passo 3 é constante ou não finita" if fora == 1 else anterior_amostragem
    )
    assert pca._amostragem_completa(desenhadas, totais) == (
        f"nenhuma — as {recorte.milhar(desenhadas)} features que a PCA reteve entram todas; "
        f"{agora_amostragem}"
    )

    anterior_indefinidas = como_era(
        fora,
        nenhuma="nenhuma tem carga não finita",
        algumas="de carga não finita são contadas, não descartadas",
    )
    agora_indefinidas = (
        "1 de carga não finita é contada, não descartada" if fora == 1 else anterior_indefinidas
    )
    assert pca._frase_das_indefinidas(fora) == agora_indefinidas

    anterior_selecao = como_era(
        desenhadas,
        nenhuma="nenhuma feature foi desenhada, porque nenhuma tem VIP calculável",
        algumas=f"features de MAIOR VIP, de {recorte.milhar(totais)} com VIP calculável",
    )
    agora_selecao = (
        f"1 feature de MAIOR VIP, de {recorte.milhar(totais)} com VIP calculável"
        if desenhadas == 1
        else anterior_selecao
    )
    assert plsda._frase_da_selecao(desenhadas, totais) == agora_selecao

    # **A planta que prova que a mudança é REAL e não uma reescrita do mesmo texto.** Sem esta
    # asserção, `agora_*` poderia ser igual a `anterior_*` no ramo do um e os três `==` acima
    # continuariam verdes — a equivalência entre duas cópias da mesma frase (na forma de
    # asserção vácua sobre uma diferença).
    if fora == 1:
        assert agora_amostragem != anterior_amostragem
        assert agora_indefinidas != anterior_indefinidas
        assert anterior_amostragem.startswith("as 1 "), (
            "âncora: a redação ANTIGA tem de ser a defeituosa, senão não há o que a U22 consertou"
        )
    if desenhadas == 1:
        assert agora_selecao != anterior_selecao
        assert anterior_selecao.startswith("as 1 ")


def _quem_reexporta_de_plsda(nomes: frozenset[str]) -> tuple[str, ...]:
    """Os módulos de PRODUÇÃO que importam `nomes` **de `refino.plots.plsda`**, por AST.

    **Derivado e não escrito, e o motivo foi medido em 15/09/2026.** A mensagem
    deste portão nomeava `desktop.janela.qualidade`, e o consumidor mudou de módulo num refator do
    plano dos gráficos: quem importa a trinca é `desktop/janela/graficos.py`. A mensagem continuava
    a passar — ela só é lida quando o portão reprova —, e no dia em que reprovasse mandaria olhar um
    arquivo que já não tem o import.

    **Por AST e não por `grep`**: o nome dentro de uma docstring ou de um comentário não é import,
    e é justamente prosa deste género que esta função existe para deixar de ser fonte.
    """
    raiz = Path(__file__).resolve().parents[2] / "src" / "refino"
    achados: set[str] = set()
    for caminho in sorted(raiz.rglob("*.py")):
        arvore = ast.parse(caminho.read_text(encoding="utf-8"))
        for no in ast.walk(arvore):
            if not isinstance(no, ast.ImportFrom) or no.module != "refino.plots.plsda":
                continue
            if nomes & {alias.name for alias in no.names}:
                relativo = caminho.relative_to(raiz.parent).with_suffix("")
                achados.add(".".join(relativo.parts))
    return tuple(sorted(achados))


def test_a_trinca_da_validacao_e_A_MESMA_pelos_dois_caminhos_de_import() -> None:  # noqa: N802
    """Ela mudou-se para `plots/validacao.py` e é REEXPORTADA por `plsda`, que é quem a janela lê.

    **Identidade e não igualdade:** dois objetos iguais seriam duas definições, que é exactamente a
    segunda cópia que a mudança existe para não criar.

    **E o nome do consumidor é DERIVADO** — ver `_quem_reexporta_de_plsda`. Escrito à mão, ele
    caducou na primeira mudança de aba e ninguém o viu, porque a mensagem de um portão verde não é
    lida por ninguém.
    """
    assert plsda.Validacao is validacao.Validacao
    assert plsda.validacao_do_fio is validacao.validacao_do_fio
    assert plsda.frase_da_validacao is validacao.frase_da_validacao
    assert plsda.frase_curta_da_validacao is validacao.frase_curta_da_validacao
    assert plsda.veredito_da_validacao is validacao.veredito_da_validacao

    exigidos = frozenset(
        ("Validacao", "validacao_do_fio", "frase_da_validacao", "veredito_da_validacao")
    )
    consumidores = _quem_reexporta_de_plsda(exigidos)
    # **Âncora de não-vacuidade, e ela vai ANTES**: sem consumidor nenhum a frase da
    # mensagem abaixo ficaria vazia e o portão passaria a afirmar sobre o nada.
    assert consumidores, (
        "âncora: nenhum módulo de produção importa a trinca de `refino.plots.plsda`. Ou a "
        "reexportação deixou de ter consumidor — e então ela sai —, ou o extrator por AST parou "
        "de ver os imports"
    )
    for nome in sorted(exigidos):
        assert nome in plsda.__all__, (
            f"{nome} saiu do `__all__` de `plsda`: com `no_implicit_reexport`, quem o importa "
            f"de lá deixa de o poder fazer — e {', '.join(consumidores)} importa(m) de lá"
        )


# ---------------------------------------------------------------------------
# 12. A IRMÃ DECLARADA: `desktop.janela.figuras._nome_legivel` — SAIU COM A DUPLICAÇÃO
#
# **O portão que aqui vivia foi REMOVIDO em 15/09/2026, e a saída estava escrita dentro dele.** Ele
# lia `_nome_legivel` da janela por AST, executava-a isolada e comparava-a com
# `rotulo_do_construtor` em dezassete nomes — mais um negativo que plantava um artigo a menos do
# lado da janela. A mensagem do próprio portão dizia: *"Se ela foi removida, o mais provável é que a
# delegação já tenha sido feita — e então este portão inteiro sai, com a duplicação"*. Foi o que
# aconteceu: uma mudança nos gráficos fez `_nome_legivel` delegar a `rotulo_do_construtor`, e a
# propriedade que este portão media — *as duas cópias concordam* — deixou de ser falsificável.
#
# **Não foi substituído por um teste de igualdade das duas pontas**, e a razão é a
# planta: depois da delegação qualquer mutação move as duas por igual, a igualdade continua a valer,
# e o teste não acusa nada. O que passou a guardar a propriedade é um cenário por AST em
# `tests/test_desktop_figuras.py` — *a janela não contém mais uma segunda regra do rótulo* —, com a
# planta de reintroduzir uma.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 12. O rodapé cabe na figura em TODO tamanho
# ---------------------------------------------------------------------------


# Os quatro tamanhos da tabela, em (polegadas, polegadas, ppp, rótulo).
_TAMANHOS_DE_D451: Final = (
    (10.0, 6.0, 100.0, "10x6 @100 — a nascença do construtor"),
    (7.0, 4.5, 96.0, "7x4,5 @96 — o padrão do pacote"),
    (1220.0 / 96.0, 302.0 / 96.0, 96.0, "1220 px @96 — a janela a 1280x860"),
    (1860.0 / 96.0, 302.0 / 96.0, 96.0, "1860 px @96 — a janela a 1920x1080"),
)


def _rodape_real_de_744() -> str:
    """O rodapé prístino medido: `figura_das_caixas_por_amostra` na forma do `neg_c18`.

    **Construído pelo construtor de recorte do pacote e não escrito à mão.** Uma cadeia literal
    aqui envelheceria no dia em que a frase do método mudasse, e o teste passaria a medir um texto
    que nenhuma figura desenha.

    **Os dígitos da DejaVu Sans são tabulares**, e isso foi medido: `"00000"`, `"88888"` e
    `"21510"` ocupam os mesmos 35,0 px a 8 pt. Então o valor exacto de `n_comparadas` não muda
    largura nenhuma — só a **contagem** de dígitos muda —, e é por isso que esta reconstrução bate
    os 744 caracteres medidos sobre o corpus real.
    """
    return distribuicao.recorte_das_caixas_por_amostra(
        n_caixas=50,
        n_amostras_comparadas=68,
        n_comparadas=2151,
        n_antes=2184,
        amostrada=True,
        seed=42,
        teto_de_caixas=50,
        coeficiente_de_tukey=COEFICIENTE_DE_TUKEY,
        matriz_antes="ingerida",
        matriz_depois="processada",
    ).texto()


def _figura_medida(largura: float, altura: float, ppp: float) -> Figure:
    figura = Figure(figsize=(largura, altura), dpi=ppp)
    FigureCanvasAgg(figura)
    figura.canvas.draw()
    return figura


def _mais_larga(figura: Figure, linhas: tuple[str, ...]) -> float:
    """A largura em píxeis da linha mais LARGA — medida no renderizador, nunca contada.

    Contar caracteres aqui seria conferir a estimativa contra ela própria, que é o instrumento a
    certificar-se. O que se quer saber é o que o `Agg` de facto desenhou.
    """
    desenhador = figura.canvas.get_renderer()
    sonda = figura.text(
        0.0, 0.0, "", fontsize=recorte.TAMANHO_DA_FONTE, linespacing=recorte.ENTRELINHA
    )
    try:
        maior = 0.0
        for linha in linhas:
            sonda.set_text(linha)
            maior = max(maior, float(sonda.get_window_extent(desenhador).width))
        return maior
    finally:
        sonda.remove()


def _largura_util(figura: Figure) -> float:
    return float(figura.get_figwidth()) * float(figura.dpi) * (1.0 - 2.0 * recorte.MARGEM)


def test_o_rodape_real_cabe_nos_quatro_tamanhos_e_a_ESTIMATIVA_nao_cabia() -> None:  # noqa: N802
    """A âncora de não-vacuidade vem PRIMEIRO, e sem ela o cenário passa sobre um rodapé curto.

    **O par causal é a estimativa contra a medição, sobre a MESMA figura e o MESMO texto** — não
    duas figuras nem dois textos. Se a âncora não afirmasse que a estimativa transborda em três dos
    quatro, este teste passaria com o mecanismo inteiro removido, porque um rodapé que já cabe cabe
    pelos dois caminhos.
    """
    texto = _rodape_real_de_744()
    assert len(texto) == 744, (
        f"o rodapé prístino tem {len(texto)} caracteres e a medição foi sobre 744. A frase do "
        "método ou a da amostragem mudou, e a tabela da medição passou a descrever outro referente"
    )

    transbordavam = 0
    for largura, altura, ppp, rotulo in _TAMANHOS_DE_D451:
        figura = _figura_medida(largura, altura, ppp)
        if _mais_larga(figura, quebra(texto, colunas_da_figura(figura))) > _largura_util(figura):
            transbordavam += 1

        medidas = recorte.linhas_do_rodape(figura, texto)
        excesso = _mais_larga(figura, medidas) - _largura_util(figura)
        assert excesso <= 0.0, (
            f"em {rotulo} a linha mais larga do rodapé sai {excesso:.1f} px pela direita. O rodapé "
            "é a única coisa da figura que outra pessoa pode conferir contra fonte independente, e "
            "cortado ele lê-se como se a frase acabasse ali"
        )

    assert transbordavam == 3, (
        f"a ÂNCORA falhou: com a estimativa o rodapé transbordava em {transbordavam} dos quatro "
        "tamanhos, e a medição deu três. Em zero, este cenário está a passar por vacuidade sobre "
        "um texto que já cabia, e não mede o mecanismo que existe para o fazer caber"
    )


def test_o_rodape_que_passou_a_caber_NAO_colide_com_o_eixo_nos_quatro_tamanhos() -> None:  # noqa: N802
    """A irmã: fazer caber na horizontal não pode empurrar o rodapé para cima do eixo.

    **É o risco principal desta unidade, e vai medido em vez de argumentado.** Menos colunas dão
    linhas mais curtas e **mais** linhas, e mais linhas é exactamente a colisão vertical que
    acabou de fechar. A âncora afirma que o número de linhas de facto **aumentou** em algum dos
    quatro — senão o cenário não exercita o risco que nomeia.
    """
    texto = _rodape_real_de_744()
    cresceu = False
    for largura, altura, ppp, rotulo in _TAMANHOS_DE_D451:
        figura = _figura_medida(largura, altura, ppp)
        figura.add_subplot(1, 1, 1)
        estimadas = quebra(texto, colunas_da_figura(figura))
        medidas = recorte.linhas_do_rodape(figura, texto)
        cresceu = cresceu or len(medidas) > len(estimadas)

        rodape = aplica(figura, Recorte(2151, 2184, "prova", "ingerida", texto))
        figura.canvas.draw()
        desenhador = figura.canvas.get_renderer()
        topo_do_rodape = float(rodape.get_window_extent(desenhador).y1)
        fundo_do_eixo = min(float(e.get_window_extent(desenhador).y0) for e in figura.axes)
        assert fundo_do_eixo - topo_do_rodape > 0.0, (
            f"em {rotulo} o rodapé de {len(medidas)} linhas encosta no eixo — folga de "
            f"{fundo_do_eixo - topo_do_rodape:.1f} px. Trocar o transbordo horizontal por colisão "
            "vertical é trocar um defeito visível por outro"
        )

    assert cresceu, (
        "a ÂNCORA falhou: a medição não aumentou o número de linhas em tamanho nenhum, então este "
        "cenário não exercitou o risco de KTD6 que ele existe para medir"
    )


def test_a_medicao_ENCOLHE_de_facto_as_colunas_e_o_recuo_devolve_a_estimativa() -> None:  # noqa: N802
    """Os dois desfechos do mecanismo, e o segundo é o ramo que nenhum outro cenário toca.

    **Contar execuções do código sob teste e não testes que passam:** um requebrador que
    devolvesse sempre a estimativa passaria em todos os cenários acima cujo rodapé já cabe, e só
    este o separa. A planta é o texto real na largura em que se mediu +33,8 pt de transbordo.
    """
    texto = _rodape_real_de_744()
    figura = _figura_medida(1860.0 / 96.0, 302.0 / 96.0, 96.0)
    estimadas = quebra(texto, colunas_da_figura(figura))
    medidas = recorte.linhas_do_rodape(figura, texto)
    assert max(len(x) for x in medidas) < max(len(x) for x in estimadas), (
        "a medição devolveu as mesmas colunas da estimativa na largura em que se mediu +33,8 px "
        "de transbordo. O ramo que encolhe não correu, e o mecanismo é decorativo"
    )

    # **O recuo declarado, e ele é o ramo `else` — o que dispara por AUSÊNCIA de condição.** Sem
    # renderizador a resposta tem de ser exactamente a de antes desta unidade.
    nua = Figure(figsize=(1860.0 / 96.0, 302.0 / 96.0), dpi=96.0)
    nua.canvas = SimpleNamespace()  # type: ignore[assignment]
    assert recorte.linhas_do_rodape(nua, texto) == quebra(texto, colunas_da_figura(nua))


def test_os_dois_extremos_que_a_medicao_NAO_pode_quebrar() -> None:  # noqa: N802
    """Um rodapé de uma palavra continua a caber; um de mil caracteres continua a ser RECUSADO.

    **O contrato deste módulo é «quebra, nunca trunca»**, e o teto é `RESERVA_MAXIMA` com recusa
    tipada — **não há marca de truncagem a conferir**, e o cenário do plano que a pedia descrevia
    outro módulo. O que a unidade não pode quebrar é qualquer uma das duas pontas.

    **Este cenário é GUARDA e não par causal, e dizê-lo é o que o impede de se ler como prova do
    mecanismo:** as duas pontas valem com a estimativa e com a medição, e o que ele acusa é a
    medição a engolir uma delas. A planta que o separa é `linhas_do_rodape` devolver o texto
    inteiro sem quebrar — aí a recusa deixa de disparar.
    """
    figura = _figura_medida(7.0, 4.5, 96.0)
    assert recorte.linhas_do_rodape(figura, "features") == ("features",)

    # O MESMO texto do cenário da recusa que já existe neste arquivo, de propósito: o que se afirma
    # é **preservação**, e um texto novo mediria outra coisa.
    alta = nova_figura()
    alta.add_subplot(1, 1, 1)
    with pytest.raises(RecorteInvalidoError, match="não é truncado"):
        aplica(alta, Recorte(7, POPULACAO, "sem amostragem", "x" * 4000, "Welch"))
