"""O painel do recorte e a estrutura que ele desenha.

**O oráculo é o WIDGET e não o valor de retorno.** Um teste que lê o que a função devolve deixa
passar um painel que devolve a lista certa sem a desenhar. Aqui `PainelDeRecorte.textos()` lê os
rótulos que estão no `layout`.

O estado de cada linha é derivado da composição da janela, e os portões conferem o `onde` de
cada seção contra os painéis que a janela declara, nas duas direções.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Literal

import pytest

from tests.qt_guard import garante_plataforma_offscreen, membros_ausentes, sonda_o_ambiente

_FALTAM = membros_ausentes(sonda_o_ambiente())
if _FALTAM:  # pragma: no cover — a decisão é exercitada em `tests/test_qt_guard.py`, com planta
    pytest.skip(
        f"grupo opcional `desktop` ausente: falta {', '.join(_FALTAM)}. "
        'Instale com `uv pip install -e ".[desktop]"` para exercitar os testes de janela.',
        allow_module_level=True,
    )

garante_plataforma_offscreen()

from PySide6.QtCore import Qt  # noqa: E402

from refino.desktop.janela.leitura import (  # noqa: E402
    CHAVE_DO_PALPITE,
    EXPLICACOES_DA_LEITURA,
    ROTULOS_DOS_EIXOS,
    EixoSemRotuloError,
    PainelDeLeitura,
    rotulo_do_eixo,
)
from refino.desktop.janela.principal import (  # noqa: E402
    ABAS,
    PAINEIS,
    JanelaPrincipal,
)
from refino.desktop.janela.recorte import (  # noqa: E402
    SEM_ITENS,
    PainelDeRecorte,
    texto_do_item,
)
from refino.desktop.janela.trabalhos import CHAVE_DOS_TRABALHOS  # noqa: E402
from refino.desktop.recorte import (  # noqa: E402
    FORA_DO_CORTE,
    MARCA_DE_FORA,
    PARA_QUEM_OPERA,
    PREFIXO_COBERTO,
    PREFIXO_FORA,
    PREFIXO_PENDENTE,
    PREFIXO_SO_LEITURA,
    SECOES_DO_RECORTE,
    PainelDesconhecidoError,
    SecaoDoRecorte,
    SemTraducaoError,
    itens_para_a_janela,
    traduz,
)
from refino.interface.fluxo import COMO_O_PALPITE_DECIDE, CONTROLES_DE_SOBRESCRITA  # noqa: E402

if TYPE_CHECKING:  # pragma: no cover - só para o mypy
    from collections.abc import Mapping

    from pytestqt.qtbot import QtBot

REPO: Final = Path(__file__).resolve().parents[1]
PAINEL: Final = REPO / "src" / "refino" / "desktop" / "janela" / "recorte.py"

DECLARADOS: Final = {painel.chave: painel.unidade for painel in PAINEIS}
TODOS: Final = frozenset(DECLARADOS)

# **O limiar da varredura de prosa, e ele tem ORIGEM medida**. Medido em 23/08/2026: o
# motivo mais curto da política tem **128** caracteres e o único literal legítimo deste painel,
# `SEM_ITENS`, tem **65**. O teste `test_o_limiar_de_prosa_separa_de_facto` afirma a separação sobre
# os dados de hoje, então o número não é escolhido no ar nem congelado sem quem o confira.
LIMIAR_DE_PROSA: Final = 100


# ---------------------------------------------------------------------------
# A ESTRUTURA
# ---------------------------------------------------------------------------


def test_o_enriquecimento_metaboanalystr_esta_na_lista() -> None:
    """O enriquecimento da lista identificada é declarado fora da entrega, uma vez só."""
    achados = [item for item in FORA_DO_CORTE if "MetaboAnalystR" in item]
    assert len(achados) == 1, f"esperava um item do MetaboAnalystR, achei {len(achados)}"


def _incoerentes(secoes: tuple[SecaoDoRecorte, ...], textos: Mapping[str, str]) -> list[str]:
    """As seções cujo texto abre com `MARCA_DE_FORA` e têm painel, ou o contrário."""
    return [s.titulo for s in secoes if bool(s.onde) == textos[s.titulo].startswith(MARCA_DE_FORA)]


def test_o_texto_e_o_estado_de_cada_secao_concordam() -> None:
    """O único portão possível sobre `onde`, e ele mede coerência e não verdade.

    Nenhum mecanismo verifica que o painel `qualidade` mostra o que a seção do PLS-DA descreve:
    `onde` é afirmação humana. Trocar `onde=""` por `onde="qualidade"` na seção do fluxo faria a aba
    dizer que o fluxo está no programa, ao lado de um texto que diz que não está.

    O que é conferível é essa contradição: texto que abre com `MARCA_DE_FORA` ao lado de um `onde`
    preenchido, ou texto que não abre com ela ao lado de um `onde` vazio.
    """
    assert SECOES_DO_RECORTE, "a estrutura está vazia — o laço abaixo não olharia nada"
    textos = {s.titulo: traduz(s.titulo)[1] for s in SECOES_DO_RECORTE}
    incoerentes = _incoerentes(SECOES_DO_RECORTE, textos)
    assert not incoerentes, (
        f"seções cujo texto contradiz o `onde`: {incoerentes}. Texto que abre com "
        f"{MARCA_DE_FORA!r} exige `onde` vazio, e `onde` preenchido exige texto que não abra "
        "com ela"
    )
    # As duas classes existem de facto, senão o laço mede uma só.
    dentro = [s for s in SECOES_DO_RECORTE if s.onde]
    fora = [s for s in SECOES_DO_RECORTE if not s.onde]
    assert dentro and fora, (
        f"{len(dentro)} seções com painel e {len(fora)} sem: com uma das classes vazia esta "
        "verificação não separa nada"
    )


def test_a_coerencia_texto_estado_acusa_quando_ha_incoerencia() -> None:
    """O par causal, em processo: seções e textos são argumento, então não pode ficar inerte."""
    textos = {
        "dentro": "O conjunto completo das onze métricas.",
        "fora": f"{MARCA_DE_FORA} desenha como as etapas se encadeiam.",
    }
    limpa = (SecaoDoRecorte("dentro", "qualidade"), SecaoDoRecorte("fora", ""))
    assert _incoerentes(limpa, textos) == [], (
        "o par limpo já acusava — a verificação sempre reprova"
    )
    # A planta é o `onde` trocado, com o texto intacto.
    trocada = (limpa[0], SecaoDoRecorte("fora", "qualidade"))
    assert _incoerentes(trocada, textos) == ["fora"], "a verificação não acusou o `onde` trocado"
    # E a direção oposta: painel apagado ao lado de um texto que não declara ausência.
    invertida = (SecaoDoRecorte("dentro", ""), limpa[1])
    assert _incoerentes(invertida, textos) == ["dentro"], (
        "a verificação não acusou o `onde` apagado"
    )


# ---------------------------------------------------------------------------
# OS TRÊS ESTADOS, derivados da composição — e o quarto, que é erro
# ---------------------------------------------------------------------------


def test_uma_secao_de_painel_composto_e_operavel_sai_como_coberta() -> None:
    """Estado 1 de 4. **Coberta exige as DUAS coisas**, e é o achado central da revisão.

    A primeira versão decidia cobertura por `compostos` sozinho, e `compostos` responde *"foi
    construído um `QWidget` e posto numa aba"*. Três seções saíam `NO PAINEL «x»` sobre painéis de
    botão morto.
    """
    itens = itens_para_a_janela(compostos=TODOS, operaveis=TODOS, declarados=DECLARADOS)
    cabecas = [cabeca for cabeca, _ in itens]
    titulos_de_qualidade = {traduz(s.titulo)[0] for s in SECOES_DO_RECORTE if s.onde == "qualidade"}
    assert any(
        c == f"{PREFIXO_COBERTO} — {titulo}" for c in cabecas for titulo in titulos_de_qualidade
    ), f"nenhuma seção saiu como coberta pelo painel de qualidade: {cabecas}"


def test_uma_secao_composta_e_nao_operavel_sai_como_sem_disparo() -> None:
    """Estado 2 de 4, e ele não existia até a revisão de código o obrigar.

    **A planta cai sobre dado em que os dois estados DIFEREM**: `compostos` tem `qualidade`
    e `operaveis` não, então uma implementação que ignorasse `operaveis` — a versão anterior —
    produz `NO PAINEL «qualidade»` e reprova aqui. Com os dois conjuntos iguais este teste seria
    vácuo.
    """
    itens = itens_para_a_janela(compostos=TODOS, operaveis=frozenset(), declarados=DECLARADOS)
    cabecas = [cabeca for cabeca, _ in itens]
    assert not any(c.startswith(PREFIXO_COBERTO) for c in cabecas), (
        f"há seção declarada coberta sem painel operável nenhum: {cabecas}"
    )
    sem_disparo = [c for c in cabecas if c.startswith(PREFIXO_SO_LEITURA)]
    assert sem_disparo, f"nenhuma seção saiu como composta-sem-disparo: {cabecas}"
    esperado = traduz("Qualidade — o conjunto completo, nunca um número")[0]
    assert any(esperado in c for c in sem_disparo), (
        f"a seção da qualidade não saiu como composta-sem-disparo: {sem_disparo}"
    )


def test_operavel_sem_estar_composto_levanta() -> None:
    """A segunda forma do erro, e ela é incoerência e não recorte.

    Painel ligado e ausente das abas é defeito de composição. Cair num estado razoável esconderia a
    incoerência.
    """
    with pytest.raises(PainelDesconhecidoError, match="ausentes da composição"):
        itens_para_a_janela(
            compostos=frozenset({"qualidade"}),
            operaveis=frozenset({"qualidade", "pipeline"}),
            declarados=DECLARADOS,
        )


def test_a_ordem_dos_grupos_e_a_que_a_funcao_declara() -> None:
    """A função afirma *"as ausências vêm primeiro"*, e aqui isso passa a ser medido.

    A revisão adversarial inverteu a ordem dos grupos e **25 testes passaram**. A ordem dos GRUPOS
    é afirmada positivamente pela função, e afirmação positiva sem asserção é o que esta suíte
    existe para não ter.

    **Os quatro grupos têm de EXISTIR para a ordem ser observável**: `compostos` é um
    subconjunto estrito, senão não há pendentes; `operaveis` é um subconjunto estrito de
    `compostos`, senão não há sem-disparo. Com os três conjuntos iguais, este teste veria dois
    grupos e afirmaria a ordem de quatro.
    """
    compostos = frozenset({"pipeline", "artefatos", "qualidade", "recorte"})
    itens = itens_para_a_janela(
        compostos=compostos, operaveis=frozenset({"qualidade"}), declarados=DECLARADOS
    )
    grupos: list[str] = []
    for cabeca, _ in itens:
        atual = cabeca.split(" — ")[0].split(" «")[0]
        if not grupos or grupos[-1] != atual:
            grupos.append(atual)
    assert grupos == [PREFIXO_FORA, PREFIXO_PENDENTE, PREFIXO_SO_LEITURA, PREFIXO_COBERTO], (
        f"a ordem dos grupos não é a declarada: {grupos}"
    )


def test_nenhuma_secao_cai_em_dois_grupos_nem_em_nenhum() -> None:
    """A partição é total E disjunta, e a revisão achou que ela não era disjunta.

    `""` em `compostos` fazia `cobertas` casar as seções fora do corte — `"" in compostos` — e cada
    uma saía **duas** vezes: `FORA — X` e `NO PAINEL «» — X`, 27 rótulos em vez de 23, sem levantar.
    Inalcançável de `JanelaPrincipal`, cujas chaves são literais, mas `itens_para_a_janela` está em
    `__all__`.
    """
    for compostos, operaveis in (
        (TODOS, TODOS),
        (TODOS | {""}, TODOS | {""}),
        (frozenset({""}), frozenset({""})),
        (frozenset(), frozenset()),
    ):
        itens = itens_para_a_janela(
            compostos=frozenset(compostos), operaveis=frozenset(operaveis), declarados=DECLARADOS
        )
        titulos = [cabeca.split(" — ", 1)[1] for cabeca, _ in itens]
        assert len(titulos) == len(set(titulos)), (
            f"com compostos={sorted(compostos)} um título saiu mais de uma vez: "
            f"{sorted(t for t in titulos if titulos.count(t) > 1)}"
        )
        esperado = len(SECOES_DO_RECORTE) + len(FORA_DO_CORTE)
        assert len(titulos) == esperado, (
            f"com compostos={sorted(compostos)} saíram {len(titulos)} itens e a estrutura tem "
            f"{esperado}"
        )


def test_uma_secao_de_painel_declarado_e_nao_composto_sai_como_pendente() -> None:
    """Estado 2 de 3, e é o que o rascunho desta unidade não tinha.

    O rascunho declarava *"1. Os seus arquivos"* como coberta pelo painel `arquivos` — e aquele
    painel **existe como classe e não está na composição**. O painel teria afirmado cobertura mais
    larga que o mecanismo, em duas linhas.
    """
    itens = itens_para_a_janela(
        compostos=frozenset({"qualidade"}), operaveis=frozenset(), declarados=DECLARADOS
    )
    cabecas = [cabeca for cabeca, _ in itens]
    pendentes = [c for c in cabecas if c.startswith(PREFIXO_PENDENTE)]
    assert pendentes, f"nenhuma seção saiu como pendente: {cabecas}"
    esperado = traduz("1. Os seus arquivos")[0]
    assert any(esperado in c for c in pendentes), (
        f"a seção dos arquivos não saiu como pendente: {pendentes}"
    )


def test_uma_secao_sem_painel_sai_como_fora() -> None:
    """Estado 3 de 3, e a seção do exemplo é **derivada** e não escolhida à mão.

    > **A primeira versão pinava `"FORA — 5. Multiverso"`, e reprovou** quando aquela seção
    > passou a ter painel, porque a rota de conclusão passou a existir. O teste estava a medir a
    > coisa certa pelo exemplo errado: o que ele afirma é que **toda** seção sem painel sai como
    > `FORA`, e citar uma pelo nome amarra o portão ao estado de um dia.

    A não-vacuidade vem primeiro: sem nenhuma seção sem painel, a comparação passaria por vazio.
    """
    sem_painel = tuple(s.titulo for s in SECOES_DO_RECORTE if not s.onde)
    assert sem_painel, (
        "nenhuma seção está sem painel: este teste passaria por vacuidade. Se todas "
        "passaram a ter painel, o estado `FORA` ficou inalcançável e é isso que tem de ser afirmado"
    )

    itens = itens_para_a_janela(compostos=TODOS, operaveis=TODOS, declarados=DECLARADOS)
    cabecas = [cabeca for cabeca, _ in itens]
    for titulo in sem_painel:
        assert f"{PREFIXO_FORA} — {traduz(titulo)[0]}" in cabecas, (
            f"a seção «{titulo}» não tem painel e não saiu como fora do corte: {cabecas}"
        )


def test_os_estados_alcancaveis_na_janela_de_hoje(qtbot: QtBot) -> None:
    """O estado real da janela, medido — e ele MUDOU DUAS VEZES, com o teste a reprovar nas duas.

    ## A história deste teste é a do painel, e ela vale mais que a asserção

    - **A 1.ª versão** afirmava *"os três estados separam"*, com `NO PAINEL` entre eles.
      Caiu porque a revisão mediu que nenhum botão dispara: `NO PAINEL` era **falso**.
    - **A 2.ª versão** afirmava *"são TRÊS e não quatro"*, com `NO PAINEL` inalcançável.
      Caiu quando quatro painéis foram ligados, e `NO PAINEL` passou a ser verdade.
    - **A de hoje** afirma os quatro estados, **derivados da janela**.

    > **A lição é a forma da asserção e não o conteúdo dela.** As duas primeiras versões
    > pinavam um retrato do estado de então; esta **deriva** de `_preenchidos` e do que o
    > recorte declara.

    **A separação continua a ser exigida onde ela existe**: `AINDA NÃO NA JANELA` só
    é alcançável se `compostos` for subconjunto estrito de `TODOS`, e hoje **não é**: a janela
    compõe todos. Então este teste afirma o que a janela de hoje produz, e a asserção
    sobre o estado pendente fica na função de estado, que o constrói com conjuntos
    escolhidos.

    **O `qtbot` não é decoração.** A primeira versão construiu a `JanelaPrincipal` sem ele e o
    processo **abortou** — `Fatal Python error: Aborted` —, porque não existe `QApplication`.
    """
    janela = JanelaPrincipal(_TransporteMudo())
    qtbot.addWidget(janela)
    reais = frozenset(janela._preenchidos)
    assert reais == TODOS, (
        f"a janela compõe {sorted(reais)} e declara {sorted(TODOS)}. A U11 compôs os seis; se um "
        "painel saiu da composição, o recorte tem de voltar a mostrar `AINDA NÃO NA JANELA`"
    )

    # O que o painel de facto mostra, lido do widget — o oráculo é o widget.
    grupos = {t.split("\n")[0].split(" — ")[0].split(" «")[0] for t in janela.recorte.textos()}
    assert PREFIXO_FORA in grupos, f"nenhuma seção fora do corte: {sorted(grupos)}"
    assert PREFIXO_COBERTO in grupos, (
        f"nenhuma seção coberta, e a U11 ligou quatro painéis: {sorted(grupos)}"
    )
    # **`AINDA NÃO NA JANELA` deixou de ser alcançável, e é MEDIÇÃO e não regressão:** os seis
    # painéis estão compostos. Afirmar a ausência dele é o que faz este teste reprovar no dia em que
    # um painel sair da composição.
    assert PREFIXO_PENDENTE not in grupos, (
        f"há seção pendente com os seis painéis compostos: {sorted(grupos)}"
    )
    # E `NA JANELA, SEM DISPARO` continua alcançável, porque `qualidade` e `recorte` são compostos e
    # não têm controle que dispare rota. Sem esta asserção, um `operaveis` que incluísse os seis
    # passaria — e diria ao usuário que há botão onde não há.
    assert PREFIXO_SO_LEITURA in grupos, (
        f"nenhuma seção composta-sem-disparo, e `qualidade` não tem botão: {sorted(grupos)}"
    )


def _classes_de_painel() -> frozenset[str]:
    """As classes `PainelDe*` que existem em `desktop/janela/*.py`, por AST.

    Derivada e não escrita: a lista de permissão nasce a cobrir o painel da unidade seguinte, e é o
    que separa este portão de uma enumeração que envelhece.
    """
    nomes: set[str] = set()
    for caminho in sorted((REPO / "src" / "refino" / "desktop" / "janela").glob("*.py")):
        arvore = ast.parse(caminho.read_text(encoding="utf-8"))
        nomes |= {
            no.name
            for no in ast.walk(arvore)
            if isinstance(no, ast.ClassDef) and no.name.startswith("PainelDe")
        }
    return frozenset(nomes)


def test_todo_painel_composto_e_de_uma_classe_de_painel_conhecida(qtbot: QtBot) -> None:
    """O painel OCO, e ele escapou de trinta testes até a revisão adversarial o plantar.

    **`_preenchidos` responde *"há um `QWidget` nesta aba"*, e um `QWidget` NU satisfaz isso.** A
    planta era uma linha — `"leitura": QWidget(raiz)` — e o painel do recorte passava a rotular duas
    seções como cobertas por um painel sem um único filho. Tudo o mais continuava verde, inclusive a
    igualdade contra o conjunto derivado: `_preenchidos` é o oráculo dos **dois** lados, então o
    painel oco entra nos dois e a comparação não o vê.

    O que o vê é perguntar de que CLASSE é cada widget composto, com a lista de permissão derivada
    das classes que existem — nunca escrita à mão.
    """
    janela = JanelaPrincipal(_TransporteMudo())
    qtbot.addWidget(janela)

    conhecidas = _classes_de_painel()
    assert len(conhecidas) >= 4, (
        f"a varredura achou {len(conhecidas)} classes de painel: {sorted(conhecidas)}. Com menos "
        "de quatro ela mediria quase nada, e a asserção abaixo passaria por lista curta"
    )
    assert janela._preenchidos, "a janela não compõe painel nenhum — o laço abaixo seria vácuo"
    ocos = {
        chave: type(painel).__name__
        for chave, painel in janela._preenchidos.items()
        if type(painel).__name__ not in conhecidas
    }
    assert not ocos, (
        f"painéis compostos que não são classe de painel: {ocos}. Um `QWidget` nu numa aba faz o "
        f"recorte declarar cobertura sobre nada. As classes conhecidas são {sorted(conhecidas)}"
    )


def test_uma_secao_que_aponta_para_painel_inexistente_levanta() -> None:
    """O quarto desfecho, e ele é erro de propósito.

    Um `onde` inválido que caísse num estado razoável se comportaria como outra política. O portão
    de bijeção sobre a constante torna o caminho
    inalcançável em produção, e ele é exercitado de todo modo (N desfechos, N direções).
    """
    with pytest.raises(PainelDesconhecidoError, match="não declara"):
        itens_para_a_janela(compostos=TODOS, operaveis=TODOS, declarados={"qualidade": "U7"})


def test_todo_onde_declarado_e_chave_de_painel_da_janela() -> None:
    """A bijeção do outro eixo: `onde` sai de `PAINEIS`, não de string escrita à mão.

    É o portão que torna o quarto desfecho inalcançável em produção.

    A âncora vai nesta função: com `SECOES_DO_RECORTE` vazia o conjunto de `onde` é vazio,
    e a diferença passa sem olhar chave nenhuma.
    """
    declarados = {s.onde for s in SECOES_DO_RECORTE if s.onde}
    assert declarados, "nenhuma seção declara painel — a asserção abaixo seria vácua"
    assert TODOS, "a janela não declara painel nenhum — a asserção abaixo seria vácua"
    fantasmas = declarados - TODOS
    assert not fantasmas, f"`onde` que não é painel da janela: {sorted(fantasmas)}"


def test_todo_painel_da_janela_e_nomeado_por_alguma_secao() -> None:
    """A direção oposta, e sem ela um painel pode existir sem nada que o explique.

    Painel novo na janela sem seção que o nomeie **reprova**: sem esta direção, a aba poderia
    não dizer nada sobre um painel que a janela mostra.

    A âncora vai nesta função: `TODOS` vazio faria a diferença ser vazia.
    """
    assert TODOS, "a janela não declara painel nenhum — a asserção abaixo seria vácua"
    assert SECOES_DO_RECORTE, "a estrutura está vazia — a asserção abaixo seria vácua"
    orfaos = TODOS - {s.onde for s in SECOES_DO_RECORTE if s.onde}
    assert not orfaos, (
        f"painéis sem seção na aba: {sorted(orfaos)}. Todo painel da janela tem ao menos uma "
        "seção em `SECOES_DO_RECORTE`, com texto para quem opera"
    )


# ---------------------------------------------------------------------------
# O PAINEL — e o oráculo é o widget
# ---------------------------------------------------------------------------


def test_o_painel_antes_de_define_diz_que_nao_calculou(qtbot: QtBot) -> None:
    """A âncora: *"não calculei"* e *"nada ficou fora"* são fatos opostos.

    **O oráculo é o WIDGET, e a primeira versão deste teste comparava a CONSTANTE.**
    `SEM_ITENS.startswith(…)` afirma sobre o literal do módulo e passa com o `QLabel` escondido, ou
    ausente do `layout`, ou com outro texto — que é exactamente o defeito de oráculo que a docstring
    deste arquivo diz ter evitado. Achado pela `refino-review`, não por teste.

    **E o par é visível-antes / escondido-depois**, porque `setVisible(not itens)` só é `True`
    enquanto `define` não correu: `itens` nunca é vazio — são 15 seções mais 8 itens, com âncoras de
    não-vacuidade em duas funções acima. Sem as duas direções, o ramo do aviso fica inerte.
    """
    painel = PainelDeRecorte()
    qtbot.addWidget(painel)
    assert painel.textos() == (), "há rótulos de item antes de `define`"
    assert painel._aviso.text() == SEM_ITENS, (
        f"o aviso mostra {painel._aviso.text()!r} e não o texto declarado"
    )
    assert not painel._aviso.isHidden(), "o aviso está escondido antes de `define`"

    painel.define(compostos=TODOS, operaveis=TODOS, declarados=DECLARADOS)
    assert painel._aviso.isHidden(), (
        "o aviso continua visível depois de `define` — ele diria «não calculei» ao lado de 23 "
        "rótulos calculados"
    )


def test_todo_item_da_estrutura_aparece_no_painel(qtbot: QtBot) -> None:
    """Cenário do plano: asserção de **presença**, por item.

    Presença e não ausência: uma asserção de que nada falta passaria com o painel vazio.
    """
    painel = PainelDeRecorte()
    qtbot.addWidget(painel)
    painel.define(
        compostos=frozenset({"pipeline", "qualidade"}),
        operaveis=frozenset(),
        declarados=DECLARADOS,
    )

    visivel = "\n".join(painel.textos())
    assert visivel, "o painel não desenhou rótulo nenhum"
    # O painel desenha o texto de `PARA_QUEM_OPERA`, e toda entrada aparece.
    for secao in SECOES_DO_RECORTE:
        titulo, _texto = traduz(secao.titulo)
        assert titulo in visivel, f"a seção {secao.titulo!r} não aparece no painel"
    for item in FORA_DO_CORTE:
        titulo, _texto = traduz(item)
        assert titulo in visivel, f"o item {item!r} não aparece no painel"


def test_o_painel_mostra_todos_os_itens_e_a_contagem_e_derivada(qtbot: QtBot) -> None:
    """A contagem sai da estrutura, não de um número escrito aqui.

    Um número literal aqui envelheceria a cada seção nova — e o teste que o guardasse reprovaria por
    acréscimo legítimo.
    """
    painel = PainelDeRecorte()
    qtbot.addWidget(painel)
    painel.define(compostos=TODOS, operaveis=TODOS, declarados=DECLARADOS)
    esperado = len(SECOES_DO_RECORTE) + len(FORA_DO_CORTE)
    assert len(painel.textos()) == esperado, (
        f"o painel desenhou {len(painel.textos())} rótulos e a estrutura tem {esperado} itens"
    )


def test_define_duas_vezes_nao_acumula(qtbot: QtBot) -> None:
    """O par causal do redesenho, e ele separa *"reconstrói"* de *"acrescenta"*.

    Sem ele, um painel que só acrescentasse mostraria o estado antigo junto do novo — e como o texto
    do estado antigo continua correto para a seção, ninguém notaria pela leitura.
    """
    painel = PainelDeRecorte()
    qtbot.addWidget(painel)
    painel.define(compostos=frozenset({"qualidade"}), operaveis=frozenset(), declarados=DECLARADOS)
    primeira = painel.textos()
    painel.define(compostos=TODOS, operaveis=TODOS, declarados=DECLARADOS)
    segunda = painel.textos()

    assert len(primeira) == len(segunda), (
        f"a segunda chamada deixou {len(segunda)} rótulos contra {len(primeira)} — acumulou"
    )
    assert primeira != segunda, (
        "as duas composições produziram texto idêntico, então este teste não separa acumular de "
        "reconstruir: escolha conjuntos de `compostos` que difiram"
    )
    assert any(
        c.startswith(f"{PREFIXO_PENDENTE} — {traduz('4. Pipeline com parâmetros explícitos')[0]}")
        for c in primeira
    )
    assert not any(
        c.startswith(f"{PREFIXO_PENDENTE} — {traduz('4. Pipeline com parâmetros explícitos')[0]}")
        for c in segunda
    )


def test_a_janela_compoe_o_painel_de_recorte_e_o_alimenta(qtbot: QtBot) -> None:
    """Painel construído e nunca alimentado.

    Um painel pode ter 100% de cobertura em linha **e** em ramo e nenhum chamador de produção. O
    que o pega não é cobertura: é ler o widget depois de construir a **janela**, sem chamar `define`
    do teste.
    """
    janela = JanelaPrincipal(_TransporteMudo())
    qtbot.addWidget(janela)

    textos = janela.recorte.textos()
    assert textos, "a janela construiu o painel do recorte e não o alimentou"

    # **IGUALDADE e não presença de prefixo, e a troca matou três mutantes que a revisão
    # achou.** A primeira versão afirmava `any("AINDA NÃO NA JANELA" in t)`, e por isso
    # passava com: o `_preenchidos` mutado DEPOIS da chamada de `define`; a ordem dos
    # grupos invertida; e `textos()` a devolver os rótulos ao contrário.
    #
    # **E os TRÊS argumentos são derivados da janela, incluindo `operaveis`.** Uma versão
    # anterior pinava `frozenset()` ali, e reprovou por acréscimo legítimo quando os painéis
    # foram ligados. Hoje `janela.operaveis` é atributo, e os dois lados mudam juntos.
    # **`texto_do_item` e nao um literal aqui — 14/09/2026.** A versao anterior reconstruia a frase
    # a mao, e era a SEGUNDA copia da formatacao: quando o rotulo passou a Markdown — com separador
    # de paragrafo —, este teste reprovou sem que houvesse defeito nenhum. Ler o simbolo que a
    # producao usa e o que faz o arnes medir o que corre em vez da propria copia.
    esperado = tuple(
        texto_do_item(cabeca, motivo)
        for cabeca, motivo in itens_para_a_janela(
            compostos=frozenset(janela._preenchidos),
            operaveis=janela.operaveis,
            declarados={painel.chave: painel.unidade for painel in PAINEIS},
            # **O QUARTO argumento, 25/09/2026, e o teste reprovou por acréscimo legítimo** — a
            # linha das vias passou a «NESTE PROGRAMA». Derivado da janela, como os outros três.
            calculam_aqui=janela.calculam_aqui,
        )
    )
    assert textos == esperado, (
        "o que o painel mostra não é o que a composição da janela deriva. Diferenças: "
        f"{[t for t in textos if t not in esperado][:2]}"
    )
    # **A CONTAGEM DE ABAS DEIXOU DE SER A DE PAINÉIS**, e a asserção antiga afirmava a
    # igualdade das duas. Ela era `abas.count() == len(PAINEIS)`, e valia enquanto cada painel era
    # uma aba; com o agrupamento por função são **cinco** abas para **oito** painéis, mais a faixa
    # dos trabalhos no rodapé. O que ela protegia — que a janela não perdesse painel pelo caminho —
    # é afirmado por `test_todo_painel_declarado_esta_composto_e_diz_de_quem_e`, que faz a bijeção
    # painel↔aba com a faixa como única exceção declarada.
    assert janela.abas.count() == len(ABAS), (
        f"a janela tem {janela.abas.count()} abas e `ABAS` declara {len(ABAS)}"
    )

    # **A contagem de abas NÃO basta, e o mutante 11 mediu isso.** Tirar `"recorte"` da composição
    # deixa a contagem intacta — a aba passa a ser a marca de painel vazio —, o painel continua a
    # ser construído, `define` continua a ser chamado, e `textos()` continua não vazio. As três
    # asserções acima passam sobre um painel que o usuário **não vê**. O que separa é perguntar
    # de quem é o widget da aba.
    # **`isAncestorOf` e já não `indexOf`.** `indexOf` responde *"este widget
    # É o widget de uma aba"*, e isso valia enquanto cada painel era a sua própria aba. Com o
    # agrupamento por função, dois painéis partilham uma aba cujo widget é um composer — então
    # `indexOf` passou a devolver `-1` para painéis que **estão na tela**, e a asserção acusaria
    # cinco deles.
    #
    # A propriedade que se quer não mudou: *"o painel é desenhado"*. Ela é que se diz de outra
    # forma agora — o painel está na ÁRVORE de alguma aba, aninhado ou não.
    def _e_desenhado(painel: object) -> bool:
        if painel is janela.trabalhos:
            # **A faixa do rodapé é a única exceção, e ela é declarada.** O painel dos trabalhos
            # saiu das abas e é desenhado fora delas, visível em qualquer aba.
            # Deixá-lo cair no laço faria a asserção acusar um painel que o usuário vê sempre.
            moldura = janela.faixa_de_trabalhos.layout()
            return moldura is not None and moldura.indexOf(janela.trabalhos) >= 0
        if painel is janela.pipeline:
            # **O painel do pipeline é desenhado pelos GRUPOS, nas três abas de etapa** — desde que
            # «Executar e exportar», a única aba que o tinha inteiro, saiu (09/11/2026). O widget
            # dele é o controlador, escondido; desenhado é cada grupo, na página da etapa dele.
            return all(
                janela._paginas_de_etapa[etapa].isAncestorOf(grupo)
                for etapa, grupos in janela.pipeline.grupos_das_etapas.items()
                for grupo in grupos
            )
        return any(
            janela.abas.widget(i) is painel or janela.abas.widget(i).isAncestorOf(painel)
            for i in range(janela.abas.count())
        )

    assert _e_desenhado(janela.recorte), (
        "o painel do recorte foi construído, alimentado, e não é desenhado em aba nenhuma: a "
        "janela mostra a marca de painel vazio no lugar dele"
    )
    # A não-vacuidade vai ANTES do laço: `_preenchidos` vazio faria a asserção abaixo passar
    # sem olhar painel nenhum, e afirmar que todos estão nas abas.
    assert len(janela._preenchidos) >= 2, (
        f"a janela compõe {len(janela._preenchidos)} painéis; com menos de dois o laço abaixo "
        "passaria por vacuidade"
    )
    fora_das_abas = [
        chave for chave, painel in janela._preenchidos.items() if not _e_desenhado(painel)
    ]
    assert not fora_das_abas, (
        f"painéis compostos que não são aba: {fora_das_abas}. Todo painel de `_preenchidos` está "
        "no `QTabWidget`, ou foi construído para ninguém"
    )


class _TransporteMudo:
    """Um `Transporte` cujas duas chamadas levantam: esta unidade não dispara rota nenhuma.

    Levantar em vez de devolver é deliberado — se algum caminho desta unidade falasse com a API, o
    teste reprovaria em vez de passar sobre uma resposta inventada. É o molde do
    `_TransporteQueLevanta` de `tests/test_desktop_janela.py`, e não herda nada porque `Transporte`
    é `Protocol`.
    """

    def post(self, url: str, **kwargs: object) -> object:
        raise AssertionError(f"a U8 não dispara rota, e alguém chamou POST {url}")

    def get(self, url: str, **kwargs: object) -> object:
        raise AssertionError(f"a U8 não dispara rota, e alguém chamou GET {url}")


# ---------------------------------------------------------------------------
# A REGRA 8 DO README NA FRONTEIRA: o painel desenha, a política declara
# ---------------------------------------------------------------------------


# **Só estes três nós têm docstring.** A primeira versão usava `getattr(no, "body", None)`, o que
# trata a primeira string de `If`, `Try`, `For`, `While` e `With` como docstring — e uma revisão
# adversarial mediu **seis** desvios por essa porta, um deles com uma CÓPIA EXACTA de um motivo
# dentro de um `if True:`, passando ruff, ruff format e mypy. Sobre-excluir é o defeito.
PORTADORES_DE_DOCSTRING: Final = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)


def _prosa_declarada_em(fonte: str) -> list[str]:
    """Instruções do módulo que carregam prosa de política, e não texto de apresentação.

    **A unidade é a INSTRUÇÃO e não o literal**, e a troca é o que fecha três desvios de uma vez:
    `"a" + "b"`, `" ".join([...])` e f-string com as metades separadas produzem literais curtos e
    uma instrução longa. Somar por instrução apanha as três sem enumerar formas de sintaxe.

    Duas armas: igualdade exacta com um motivo da política — que apanha a cópia — e comprimento
    somado acima do limiar, que apanha a paráfrase.
    """
    motivos = {corpo for _titulo, corpo in PARA_QUEM_OPERA.values()}
    arvore = ast.parse(fonte)

    docstrings: set[int] = set()
    for no in ast.walk(arvore):
        if isinstance(no, PORTADORES_DE_DOCSTRING) and no.body:
            primeiro = no.body[0]
            if (
                isinstance(primeiro, ast.Expr)
                and isinstance(primeiro.value, ast.Constant)
                and isinstance(primeiro.value.value, str)
            ):
                docstrings.add(id(primeiro.value))

    achados: list[str] = []
    for no in ast.walk(arvore):
        if not isinstance(no, ast.stmt):
            continue
        textos = [
            filho.value
            for filho in ast.walk(no)
            if isinstance(filho, ast.Constant)
            and isinstance(filho.value, str)
            and id(filho) not in docstrings
        ]
        if not textos:
            continue
        # Instruções aninhadas seriam contadas duas vezes; o que interessa é haver ALGUMA que
        # exceda, então reportar por instrução e deduplicar pelo texto junto basta.
        junto = "".join(textos)
        if any(t in motivos for t in textos) or len(junto) > LIMIAR_DE_PROSA:
            achados.append(junto)
    return achados


def test_o_limiar_de_prosa_separa_de_facto() -> None:
    """A origem do limiar, medida sobre os dados de hoje.

    Sem esta afirmação, `LIMIAR_DE_PROSA` seria um número escolhido no ar: alto demais e o portão
    não apanha paráfrase nenhuma, baixo demais e ele acusa o aviso legítimo do painel.
    """
    motivos = [corpo for _titulo, corpo in PARA_QUEM_OPERA.values()]
    assert len(SEM_ITENS) < LIMIAR_DE_PROSA < min(len(m) for m in motivos), (
        f"o limiar {LIMIAR_DE_PROSA} não separa: o aviso tem {len(SEM_ITENS)} caracteres, "
        f"e o texto mais curto tem {min(len(m) for m in motivos)}"
    )


def test_o_painel_nao_declara_prosa_de_politica() -> None:
    """Regra 8 do README na fronteira da janela — e o par causal está no teste abaixo."""
    achados = _prosa_declarada_em(PAINEL.read_text(encoding="utf-8"))
    assert not achados, (
        f"o painel declara prosa que pertence a `desktop/recorte.py`: {achados}. "
        "Nenhuma frase de motivo nasce na janela"
    )


def test_a_varredura_de_prosa_acusa_quando_ha_prosa() -> None:
    """O par causal, e ele é **em processo**: a fonte é argumento, então não pode ficar inerte.

    Duas plantas, uma por arma. Sem a segunda, um portão que só comparasse por igualdade passaria
    numa paráfrase — e paráfrase é a forma que a política tomaria se alguém a reescrevesse aqui.
    """
    limpa = PAINEL.read_text(encoding="utf-8")
    assert _prosa_declarada_em(limpa) == [], "a fonte limpa já acusava — a varredura sempre reprova"

    copiada = f"{limpa}\n\nCOPIA = {traduz(SECOES_DO_RECORTE[0].titulo)[1]!r}\n"
    assert _prosa_declarada_em(copiada), "a varredura não acusou uma cópia exacta de motivo"

    parafraseada = f'{limpa}\n\nPARAFRASE = "{"x" * (LIMIAR_DE_PROSA + 1)}"\n'
    assert _prosa_declarada_em(parafraseada), "a varredura não acusou prosa longa"


def test_a_estrutura_do_recorte_nao_importa_qt() -> None:
    """A política é núcleo puro no sentido da regra 8 do README: ela não conhece a camada de janela.

    Um `import` de Qt aqui tornaria a estrutura inutilizável para qualquer outra superfície — e é a
    estrutura que a próxima interface vai consumir.
    """
    fonte = (REPO / "src" / "refino" / "desktop" / "recorte.py").read_text(encoding="utf-8")
    arvore = ast.parse(fonte)
    importados = [
        (no.module or "") for no in ast.walk(arvore) if isinstance(no, ast.ImportFrom)
    ] + [alias.name for no in ast.walk(arvore) if isinstance(no, ast.Import) for alias in no.names]
    assert importados, "a varredura de imports não achou nada — mediria o vazio"
    qt = [nome for nome in importados if nome.startswith(("PySide6", "refino.desktop.janela"))]
    assert not qt, f"a estrutura do recorte importa a camada de janela: {qt}"


def test_o_item_do_recorte_e_imutavel() -> None:
    """`frozen=True` na seção, exercitado e não lido do decorador."""
    with pytest.raises((AttributeError, TypeError)):
        SecaoDoRecorte("t", "").onde = "qualidade"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# A prosa nao AFIRMA estado de ligacao; o estado e derivado
# ---------------------------------------------------------------------------
#
# **Medido em 23/08/2026: quatro afirmacoes do painel estavam falsas.** Os ESTADOS
# actualizaram-se sozinhos — sao derivados de `operaveis` —, e as frases escritas a mao ao lado
# deles nao. Nada as conferia.
#
# | Onde | O que dizia | Medido |
# |---|---|---|
# | item dos controles | *"NENHUM botao desta janela dispara rota"* | quatro disparam |
# | secao dos artefatos | *"quando os botoes estiverem ligados"* | estao |
# | secao dos artefatos | *"cada download refara a operacao"* | a rota RECLAMA |
# | secao COM RESSALVA | *"a janela nao tem seletor de corretor"* | `pipeline.py:383` |
#
# **A regra que este portao mecaniza:** prosa escrita a mao diz POR QUE; o estado e derivado e a
# prosa nao o repete. Repetir e a segunda fonte de verdade sobre a mesma coisa, e a copia envelhece.

# **As aspas angulares SALVAM a citacao, e isso e requisito e nao conveniencia.** Contradicao se
# resolve de forma ADITIVA: a afirmacao falsa fica, com a
# medicao que a desfez ao lado. Um portao que proibisse a frase em qualquer lugar proibiria a
# propria correcao — e as quatro correcoes desta unidade citam o texto que retratam.
CITACAO = re.compile(r"«.*?»", re.S)

# **Vocabulario FECHADO de afirmacao de estado**, e ele e escrito a mao de proposito: nao ha como
# derivar "frases que afirmam ligacao" de coisa nenhuma. E um fio de tropeco, nao uma prova — e
# dize-lo e o que impede a proxima sessao de o ler como cobertura.
AFIRMACOES_DE_LIGACAO: tuple[str, ...] = (
    "nenhum botão desta janela dispara rota",
    "nenhum botão dispara rota",
    "estiverem ligados",
    "quando os botões",
    "não tem seletor de corretor",
    "botão morto",
)


def _prosa_do_painel() -> dict[str, str]:
    """Toda string que o painel MOSTRA, por rótulo. **As montadas, e não a fonte.**

    A distinção não é cerimônia: este arquivo quebra os textos em literais adjacentes, então
    `"NENHUM botão "` e `"desta janela dispara rota"` não formam a frase no **código-fonte**. Um
    portão que lesse o arquivo passaria sobre a afirmação que existe na tela.
    """
    prosa: dict[str, str] = {}
    for secao in SECOES_DO_RECORTE:
        prosa[f"seção «{secao.titulo}»"] = "\n".join(traduz(secao.titulo))
    for item in FORA_DO_CORTE:
        prosa[f"item «{item}»"] = "\n".join(traduz(item))
    return prosa


def test_u17_a_prosa_do_painel_nao_afirma_estado_de_ligacao() -> None:
    """O portão que faltava — e a ausência dele custou quatro afirmações falsas por uma unidade.

    **Fora das aspas angulares** a prosa não afirma estado de ligação; **dentro** é citação de texto
    retratado, que a resolução aditiva exige preservar.
    """
    prosa = _prosa_do_painel()
    assert len(prosa) >= 20, (
        f"âncora: só {len(prosa)} texto(s) a conferir — as constantes não carregaram, e a asserção "
        "abaixo é vácua"
    )
    assert sum(len(t) for t in prosa.values()) > 5_000, (
        "âncora: a prosa somada é curta demais para ser a do painel"
    )

    culpadas: list[str] = []
    for rotulo, texto in prosa.items():
        sem_citacao = CITACAO.sub("«…»", texto)
        for frase in AFIRMACOES_DE_LIGACAO:
            if frase in sem_citacao.lower():
                culpadas.append(f"{rotulo}: {frase!r}")
    assert culpadas == [], (
        "estas prosas AFIRMAM estado de ligação fora de citação, e o estado é derivado de "
        f"`operaveis`: {culpadas}. Se a afirmação for histórica, ponha-a entre «aspas angulares» "
        "com a medição que a desfez ao lado — é a resolução aditiva"
    )


def test_u17_o_portao_da_prosa_acusa_quando_a_afirmacao_esta_viva() -> None:
    """A planta, e sem ela o portão acima é decorativo.

    **Duas direções na mesma função:** a mesma frase fora das aspas é acusada, e dentro delas não.
    Sem a segunda metade, o portão proibiria a própria correção — que é o que a resolução aditiva
    escreve.
    """
    viva = "Medido: nenhum botão dispara rota, e nada os liga."
    citada = "Até 23/08/2026 este item dizia «nenhum botão dispara rota», e a U11 ligou quatro."

    def acusa(texto: str) -> bool:
        sem_citacao = CITACAO.sub("«…»", texto)
        return any(f in sem_citacao.lower() for f in AFIRMACOES_DE_LIGACAO)

    assert acusa(viva), "o portão NÃO acusa a afirmação viva: ele é decorativo"
    assert not acusa(citada), "o portão acusa a CITAÇÃO, e assim proibiria a resolução aditiva"


def test_u17_o_item_dos_controles_nao_enumera_paineis(qtbot: Any) -> None:
    """Quais abas têm controle **não** está escrito no item: é derivado e aparece nos estados.

    Uma lista escrita no texto envelheceria a cada painel ligado. O oráculo é que nenhuma chave
    de painel nem título de aba apareça no texto daquele item, e que ele mande ler a lista.
    """
    alvo = next(
        (item for item in FORA_DO_CORTE if "Controles próprios" in item),
        None,
    )
    assert alvo is not None, "âncora: o item dos controles não existe mais com esse título"

    texto = " ".join(traduz(alvo)[1].split())
    nomes = [p.chave for p in PAINEIS] + [a.titulo for a in ABAS]
    assert len(nomes) >= 10, f"âncora: só {len(nomes)} nomes de painel e de aba"
    citados = [nome for nome in nomes if nome in texto]
    assert citados == [], (
        f"o item enumera painéis ou abas: {citados}. Quais têm controle é derivado de "
        "`operaveis`, e uma lista escrita aqui envelhece a cada painel ligado"
    )
    assert "lista acima" in texto, (
        "o item não diz onde está a informação, e sem isso quem o lê procura a lista"
    )


# ---------------------------------------------------------------------------
# A SEGUNDA AUDIÊNCIA — que o painel passou a servir em 26/08/2026
# ---------------------------------------------------------------------------


def test_toda_entrada_do_recorte_tem_texto_para_quem_opera() -> None:
    """Cobertura TOTAL do mapa, nas duas direções, e é isto que o impede de apodrecer.

    Um mapa por chave morre de duas formas: entrada nova sem tradução — e a janela levantaria em
    runtime, na frente de quem abriu o programa — e tradução órfã de uma entrada que já saiu, que
    envelhece em silêncio. As duas são afirmadas aqui, e a segunda é a que ninguém procura.
    """
    chaves_reais = {s.titulo for s in SECOES_DO_RECORTE} | set(FORA_DO_CORTE)
    assert chaves_reais, "a varredura não achou entrada nenhuma — mediria o vazio"

    sem_traducao = sorted(chaves_reais - set(PARA_QUEM_OPERA))
    assert not sem_traducao, (
        f"entradas que a janela desenharia sem texto para quem opera: {sem_traducao}"
    )

    orfas = sorted(set(PARA_QUEM_OPERA) - chaves_reais)
    assert not orfas, f"traduções para entradas que já não existem: {orfas}"


def test_o_texto_para_quem_opera_nao_fala_a_lingua_do_repositorio() -> None:
    r"""Nenhum identificador do projeto no que a janela desenha.

    **O par causal é o próprio painel antes do conserto**, e por isso os padrões não são
    inventados: cada um foi lido na captura de 26/08/2026 do painel desenhado, que mostrava
    rótulos de critério, números de decisão, caminhos de documento, alvos de make, módulos
    internos, rótulos de pergunta aberta e identificadores de ACL do Windows.

    A asserção é sobre o TEXTO RENDERIZADO e não sobre o mapa: é o rendido que a pessoa lê, e um
    teste sobre a constante deixaria passar um prefixo de categoria que vazasse identificador —
    que é exactamente o que `NO PAINEL «{s.onde}»` e `é da {declarados[s.onde]}` faziam.
    """
    desenhado = "\n".join(
        f"{cabeca}\n{corpo}"
        for cabeca, corpo in itens_para_a_janela(
            # **Derivado de `PAINEIS` e não pinado.** A primeira versão listava seis chaves à mão, e
            # reprovou no primeiro painel novo, quando `multiverso` e `trabalhos` entraram.
            compostos=frozenset(painel.chave for painel in PAINEIS),
            operaveis=frozenset(painel.chave for painel in PAINEIS),
            declarados={painel.chave: painel.unidade for painel in PAINEIS},
        )
    )
    assert len(desenhado) > 1000, "o painel rendeu quase nada — a asserção mediria o vazio"

    proibidos = {
        "[CA-": "rótulo de critério de aceitação",
        "[TO-": "rótulo de teste obrigatório",
        "docs/": "caminho de documento do repositório",
        "make test": "alvo de make",
        "ui/fluxo": "módulo interno",
        "ui/app": "módulo interno",
        "BUILTIN": "identificador de ACL do Windows",
        "OQ2": "rótulo de pergunta aberta do plano",
        "`0600`": "modo POSIX",
    }
    vazando = {p: razao for p, razao in proibidos.items() if p in desenhado}
    assert not vazando, f"o painel desenha linguagem de repositório: {vazando}"

    # `D<n>` precisa de padrão e não de substring: um número de decisão vaza, «Diagnosticar» não.
    decisoes = re.findall(r"\bD\d{2,3}\b", desenhado)
    assert not decisoes, f"o painel desenha números de decisão: {sorted(set(decisoes))}"


def test_traduz_levanta_em_vez_de_recuar() -> None:
    """A direção negativa do mapa: chave ausente LEVANTA, e não cai num texto razoável.

    Sem esta asserção, `traduz` poderia devolver o `motivo` interno como recuo e o teste de
    cobertura acima continuaria verde — a entrada nova apareceria na tela em linguagem de
    repositório, que é o defeito que o mapa existe para fechar.
    """
    with pytest.raises(SemTraducaoError, match="não tem texto"):
        traduz("uma entrada que ninguém escreveu")

    # E o caminho de sucesso, para o teste não passar por a função levantar sempre.
    titulo, texto = traduz(FORA_DO_CORTE[0])
    assert titulo and texto, "a tradução de uma entrada real veio vazia"


def test_o_leia_me_nomeia_as_abas_que_a_janela_de_facto_tem(qtbot: QtBot) -> None:
    """As instruções que viajam com o executável citam aba por aba, e elas têm de existir.

    **É a mesma classe que esta sessão consertou dentro do painel, um degrau acima:** prosa mantida
    à mão que descreve o estado da janela, num arquivo que a mudança da janela não obriga a tocar.
    Uma aba renomeada deixaria o LEIA-ME a mandar o profissional procurar algo que não está lá — e
    ele é a primeira coisa que quem recebe a pasta lê.

    Medido em 26/08/2026 ao escrever este teste: **duas** das seis estavam citadas por um nome
    encurtado — «2. Validação» e «4. Pipeline» — que não casa o título real.

    ## A SEGUNDA METADE, e a primeira passava por SORTE

    Até 03/09/2026 este portão lia só os títulos de `PAINEIS`, porque cada painel era uma aba. Com
    o agrupamento por função as duas coisas separaram-se, e o portão passou a medir a errada:
    **ele confere o nome dos painéis e não o nome das abas que o profissional vê**.

    E ele continuava **verde**, o que é pior que reprovar. Medido no instante da separação: os
    oito títulos de painel perderam a numeração — «1. Os seus arquivos» virou «Os seus arquivos» —
    e o LEIA-ME antigo, que ainda dizia «1. Os seus arquivos», **contém** a versão nova como
    substring. Os oito passavam; **três dos cinco títulos de aba estavam ausentes** — «Preparar o
    estudo», «Corrigir e levar o resultado» e «Qualidade do resultado».

    > **É a forma exacta do defeito que este portão existe para não ter, um degrau acima:** prosa
    > mantida à mão que descreve a janela, num arquivo que a mudança da janela não obriga a tocar —
    > só que agora com um portão a atestar que estava em dia.

    As duas listas ficam, e cada uma protege uma coisa: os títulos de **aba** são o que o
    profissional procura na barra; os de **painel** são o que ele lê nas molduras de dentro.
    """
    leia_me = REPO / "docs" / "LEIA-ME-aplicativo.txt"
    assert leia_me.is_file(), "o LEIA-ME que o cenário 1b copia para a pasta não existe"
    texto = leia_me.read_text(encoding="utf-8")
    assert len(texto) > 500, "o LEIA-ME está quase vazio — a asserção mediria o nada"

    assert ABAS and PAINEIS, "âncora: as estruturas estão vazias e as asserções seriam vácuas"

    sem_aba = [aba.titulo for aba in ABAS if aba.titulo not in texto]
    assert not sem_aba, (
        f"o LEIA-ME não nomeia estas ABAS pelo título que a barra mostra: {sem_aba}. É o primeiro "
        f"nome que quem recebe a pasta procura na tela"
    )
    # As etapas divididas usam o mesmo controlador, e seu título técnico já não é uma moldura.
    # Conferir as seções realmente compostas evita pedir no guia um nome que saiu da interface.
    from refino.desktop.janela.componentes import Secao

    janela = JanelaPrincipal(_TransporteMudo())
    qtbot.addWidget(janela)
    titulos_das_molduras = [
        secao.titulo.text()
        for indice, aba in enumerate(ABAS)
        if len(aba.paineis) > 1
        for secao in janela.abas.widget(indice).findChildren(Secao)
    ]
    assert titulos_das_molduras, "âncora: nenhuma moldura composta foi encontrada"
    titulos_das_molduras.append(next(p.titulo for p in PAINEIS if p.chave == CHAVE_DOS_TRABALHOS))
    ausentes = [titulo for titulo in titulos_das_molduras if titulo not in texto]
    assert not ausentes, (
        f"o LEIA-ME não nomeia estes PAINÉIS pelo título que a moldura mostra: {ausentes}"
    )


def test_u25_a_folga_do_painel_fica_no_fim_e_nao_entre_o_aviso_e_a_lista(qtbot: QtBot) -> None:
    """O esticador é o ÚLTIMO item da pilha depois de `define`, e continua a ser depois do segundo.

    ## O defeito, medido antes do conserto

    `__init__` punha `addStretch(1)` logo a seguir ao aviso, e `define` usava `addWidget` — que
    **apende**. Medido em 03/09/2026 imprimindo a pilha depois de um `define` com os oito painéis
    declarados: `aviso` na posição 0, **esticador na posição 1**, e as 23 seções da 2 à 24. A folga
    ficava entre o aviso e a lista.

    **Ele não tinha sintoma visível porque as 23 seções transbordam a `QScrollArea`** e o esticador
    colapsa a zero. Aparece assim que a lista é curta — e a lista curta é a partição `AINDA NÃO`,
    que é justamente a que alguém abre para saber o que falta.

    ## Por que a asserção é a POSIÇÃO e não «existe um esticador»

    *"Há um `addStretch`"* era satisfeito por um painel com o defeito oposto. Aqui o esticador
    **existe** nas duas versões
    — o que muda é onde ele está —, então um oráculo de presença não distingue conserto de defeito.

    ## As duas direções, e a segunda é o `define` repetido

    `define` é chamado a cada recomposição, e uma implementação que reinserisse o esticador a cada
    passagem acumularia um por chamada — folga a crescer sem ninguém ver, porque cada um sozinho
    colapsa. A contagem de esticadores é afirmada, e não só a posição do último.
    """
    painel = PainelDeRecorte()
    qtbot.addWidget(painel)
    todos = frozenset(p.chave for p in PAINEIS)
    declarados = {p.chave: p.unidade for p in PAINEIS}

    for passagem in (1, 2):
        painel.define(compostos=todos, operaveis=todos, declarados=declarados)
        pilha = painel._pilha
        assert pilha.count() > 2, (
            f"âncora: a pilha tem {pilha.count()} itens na passagem {passagem} — com dois ou menos "
            "a asserção da posição passaria por não haver lista que pudesse ficar do lado errado"
        )
        esticadores = [i for i in range(pilha.count()) if pilha.itemAt(i).widget() is None]
        assert esticadores == [pilha.count() - 1], (
            f"passagem {passagem}: esticador(es) em {esticadores} de {pilha.count()} itens. "
            "Ele tem de ser o ÚLTIMO e ser UM: no meio, a folga separa o aviso da lista; "
            "repetido, ela cresce a cada `define`"
        )


def test_u25_todo_eixo_de_sobrescrita_tem_nome_na_lingua_de_quem_opera() -> None:
    """Bijeção `CONTROLES_DE_SOBRESCRITA` ↔ `ROTULOS_DOS_EIXOS`, nas DUAS direções.

    ## O defeito que ele fecha, e ele estava na tela

    `ROTULOS_DOS_EIXOS` nasceu para consertar caixas que mostravam o nome do campo da API em
    inglês. Medido na captura de 03/09/2026, **seis das nove voltaram a mostrá-lo**: `eu decido
    imputation`, `sample_normalization`, `transformation`, `scaling`, `missing_threshold` e
    `rsd_variant`. O mapa tinha três entradas e a tabela de controles tinha nove, e o
    `.get(eixo, eixo)` recuava em silêncio para o nome cru.

    **Nenhum portão ligava as duas tabelas** — é a forma mais convidativa de caducidade que este
    projeto mede: as duas constantes continuam plausíveis, e só quem desenha a janela vê.

    ## As duas direções, e a segunda é a que apodrece sem sintoma

    Eixo sem rótulo aparece na tela em inglês; rótulo órfão não aparece em lado nenhum e fica
    para sempre, a fazer alguém supor que o eixo existe. A segunda é a que nada mais apanha.
    """
    eixos = {controle.eixo for controle in CONTROLES_DE_SOBRESCRITA}
    assert eixos, "âncora: não há controle de sobrescrita, e as duas asserções seriam vácuas"
    assert ROTULOS_DOS_EIXOS, "âncora: o mapa de rótulos está vazio"

    sem_rotulo = sorted(eixos - set(ROTULOS_DOS_EIXOS))
    assert not sem_rotulo, (
        f"estes eixos não têm nome na língua de quem opera: {sem_rotulo}. A caixa mostraria o "
        f"nome do campo da API em inglês, que é o defeito que `ROTULOS_DOS_EIXOS` existe para "
        f"não ter — e que já aconteceu em seis eixos ao mesmo tempo"
    )
    orfaos = sorted(set(ROTULOS_DOS_EIXOS) - eixos)
    assert not orfaos, (
        f"estes rótulos não correspondem a eixo nenhum de `CONTROLES_DE_SOBRESCRITA`: {orfaos}. "
        f"Rótulo órfão não aparece na tela e fica para sempre, a sugerir um eixo que não existe"
    )


def test_u25_o_rotulo_de_um_eixo_desconhecido_levanta_em_vez_de_recuar() -> None:
    """A direção NEGATIVA de `rotulo_do_eixo`, e ela é o que impede o defeito de voltar em silêncio.

    O portão acima confere as tabelas de hoje. Este confere o **mecanismo**: se alguém trocar o
    `ROTULOS_DOS_EIXOS[eixo]` por um `.get(eixo, eixo)`, a bijeção continua verde e a tela volta a
    poder mostrar o nome cru no dia em que as duas tabelas divergirem entre uma edição e a outra.

    É a mesma decisão que `desktop/recorte.SemTraducaoError` regista para o mapa irmão: *"um mapa
    por chave só sobrevive se a chave em falta LEVANTAR"*.
    """
    with pytest.raises(EixoSemRotuloError) as erro:
        rotulo_do_eixo("um_eixo_que_nao_existe")
    assert "um_eixo_que_nao_existe" in str(erro.value), (
        "a mensagem não nomeia o eixo em falta, e quem a ler não sabe o que acrescentar"
    )

    # A âncora positiva: um eixo REAL continua a devolver o nome, e não levanta.
    real = next(iter(CONTROLES_DE_SOBRESCRITA)).eixo
    assert rotulo_do_eixo(real), f"âncora: o eixo real `{real}` não devolveu rótulo nenhum"


def test_u25_nenhuma_prosa_do_recorte_nomeia_aba_que_nao_existe() -> None:
    """Toda «aba X» citada em `PARA_QUEM_OPERA` é uma aba que a janela tem.

    ## O buraco que ele fecha, e ele foi nomeado antes de ser medido

    Este mapa é o que o painel «Métodos e limites» mostra ao usuário, e ele existe
    para não mentir sobre a janela. **Treze das 23 entradas citam «aba»**, medido em 03/09/2026, e
    o agrupamento por função tornou falsas as que nomeavam uma aba pelo nome antigo — «Primeira
    aba», «Quarta aba», «a aba «Qualidade»», «a aba do pipeline».

    **Nenhum mecanismo via isso.** `recorte.py:129` regista que trocar um `onde` *"passou por 25
    testes"*; a prosa ao lado do `onde` não tinha sequer essa cobertura. O modo de falha é o pior
    que este painel pode ter: ele manda o profissional procurar uma aba que não está lá, com toda
    a autoridade de um painel cuja função é declarar limites.

    ## O oráculo, e por que ele é ESTREITO de propósito

    Ele acusa só a forma `aba «Nome»` — a citação entre aspas angulares precedida da palavra. Não
    tenta entender «esta aba», «na mesma aba» nem «noutra aba», que são referências relativas e
    continuam verdadeiras sob qualquer agrupamento.

    **Contado antes de escrever:** sobre a árvore de 03/09/2026 o padrão
    casa as citações de nome de aba e **zero** referências relativas. Um oráculo mais largo — «a
    palavra aba aparece» — acusaria as treze, incluindo as que estão certas, e a tabela de isenção
    viraria a lista de tudo.
    """
    padrao = re.compile(r"\baba\s+«([^»]+)»", re.IGNORECASE)
    titulos = {aba.titulo for aba in ABAS}
    assert titulos, "âncora: a janela não declara aba nenhuma, e a asserção seria vácua"

    citadas = [
        (chave, nome)
        for chave, (_cabeca, motivo) in PARA_QUEM_OPERA.items()
        for nome in padrao.findall(motivo)
    ]
    assert citadas, (
        "âncora: nenhuma entrada de `PARA_QUEM_OPERA` cita uma aba pelo nome. Ou o padrão deixou "
        "de casar a forma que a prosa usa, e o portão ficou inerte, ou a prosa deixou de nomear "
        "abas — e nos dois casos alguém tem de olhar"
    )

    mortas = sorted({(chave, nome) for chave, nome in citadas if nome not in titulos})
    assert not mortas, (
        f"a prosa do recorte nomeia abas que a janela não tem: {mortas}. As abas de hoje são "
        f"{sorted(titulos)}. Este painel existe para não mentir sobre a janela, e mandar o "
        f"profissional procurar uma aba que não está lá é a mentira mais cara que ele pode contar"
    )


# ---------------------------------------------------------------------------
# A prosa do recorte não diz onde a figura vive se ela já não vive lá
# ---------------------------------------------------------------------------

# **O vocabulário de figura, e ele é o único literal desta varredura.** As formas de dizer
# «desenho» que a prosa deste projeto usa, mais o nome que o domínio dá à figura do PLS-DA.
_FIGURA: Final = r"(?:gr[áa]fico|figura|dispers[ãa]o|desenho|painel de pontos)"

# **«continua fora» e as suas conjugações.** O verbo vai enumerado em vez de `\w+` porque a
# largura é o que separa este padrão do falso positivo: sem ela, *"não há nada a clicar nela — as
# figuras mudaram-se"* casaria, e o portão acusaria a própria correção. Medido em 15/09/2026.
_ESTA_FORA: Final = r"(?:continuam?|ficam?|est[áãa]o?|permanecem?|segue[m]?)\s+(?:de\s+)?fora\b"

FIGURA_DECLARADA_FORA: Final = re.compile(
    rf"{_FIGURA}s?[^.]{{0,60}}{_ESTA_FORA}|{_ESTA_FORA}[^.]{{0,60}}{_FIGURA}", re.IGNORECASE
)
SEM_ABA_PROPRIA: Final = re.compile(r"n[ãa]o\s+numa?\s+aba\s+pr[óo]pria", re.IGNORECASE)


def _prosa_do_recorte() -> list[tuple[str, str, str]]:
    """`(rótulo, onde, texto)` para toda a prosa de `recorte.py`: o texto que a aba mostra de
    cada seção e de cada item.

    O `onde` viaja junto porque é ele que diz de que painel o texto fala.
    """
    itens: list[tuple[str, str, str]] = []
    for secao in SECOES_DO_RECORTE:
        cabeca, corpo = traduz(secao.titulo)
        itens.append((f"opera«{secao.titulo}»", secao.onde, f"{cabeca}\n{corpo}"))
    for item in FORA_DO_CORTE:
        cabeca, corpo = traduz(item)
        itens.append((f"opera«{item}»", "", f"{cabeca}\n{corpo}"))
    return itens


def _painel_das_figuras_alimenta_o_registo() -> bool:
    """Se o painel das figuras sabe construir TODA figura do registo.

    É a âncora de P2, e ela é a metade que torna o predicado honesto: proibir *"a figura está
    fora"* só é legítimo enquanto a figura estiver **dentro**. Sem ela, o portão continuaria a
    reprovar no dia em que as figuras saíssem do produto — a proibir a frase verdadeira.
    """
    from refino.desktop.janela.graficos import PainelDeGraficos
    from refino.plots.catalogo import CONSTRUTORES

    painel = PainelDeGraficos()
    try:
        return bool(CONSTRUTORES) and set(painel._montadores()) == {c.funcao for c in CONSTRUTORES}
    finally:
        painel.deleteLater()


def test_u2_a_prosa_do_recorte_nao_diz_onde_a_figura_vive_se_ela_ja_nao_vive_la(
    qtbot: QtBot,
) -> None:
    """**Os três predicados, e a população que cada um NÃO alcança vai declarada.**

    O painel de figura saiu da aba «Qualidade do resultado» para uma aba própria. Quatro textos
    deste módulo descreviam a posição antiga, e **um deles escapou à primeira leitura**: um
    oráculo escrito depois do conserto codifica as instâncias que já não existem, e fica cego às
    que sobraram.

    ## Os três predicados, contados ANTES do conserto e depois

    O gesto foi correr cada candidato contra a árvore de `HEAD` — a de antes desta unidade — e
    contra a de depois. Medido em 15/09/2026, sobre os **46** textos das duas estruturas:

    | Predicado | Antes | Depois | De onde vem o referente |
    |---|---|---|---|
    | **P1** controle do painel das figuras fora da seção dele | 1 | 0 | do WIDGET construído |
    | **P2** figura declarada fora do produto | 2 | 0 | de `plots.CONSTRUTORES` |
    | **P3** «não numa aba própria» sobre painel que TEM aba | 1 | 0 | de `ABAS` |

    **Juntos acusam 3 dos 4**, e não 4: a quarta instância — a que dizia que o par do PLS-DA era
    desenhado no mesmo painel das métricas — só se distingue da verdade por saber qual painel é
    qual, e nenhum predicado barato faz isso. Ela é afirmada por **enumeração nominal** abaixo, e
    não por contagem: *"quatro frases"* passa se uma sair e outra entrar.

    > **Duas versões mais largas foram MEDIDAS e recusadas no mesmo dia.** *"Texto de seção
    > `onde='qualidade'` que nomeie figura"* acusava 3 de 4 antes — e **1 falso positivo depois**,
    > a própria frase que diz que as figuras se mudaram. E `não há` dentro do padrão de ausência
    > casava *"não há nada a clicar nela — as figuras…"*. Um padrão que acusa o legítimo morre no
    > primeiro dia.

    **O que NENHUM dos três alcança**, e dizê-lo é o que impede a leitura larga: prosa que nomeie
    a posição de um widget sem usar nenhuma das três formas acima — *"no fim da aba"*, *"logo
    abaixo das métricas"*. A varredura que apanha essa classe é a leitura humana do painel montado,
    e foi ela que achou a quarta instância.
    """
    from PySide6.QtWidgets import QPushButton

    from refino.desktop.janela.graficos import PainelDeGraficos

    itens = _prosa_do_recorte()
    assert len(itens) == len(SECOES_DO_RECORTE) + len(FORA_DO_CORTE) >= 20, (
        f"âncora: a varredura achou {len(itens)} textos e mediria quase nada"
    )

    # --- P1: o controle é DERIVADO do widget, e não escrito aqui --------------------------------
    painel = PainelDeGraficos()
    qtbot.addWidget(painel)
    controles = [b.text() for b in painel.findChildren(QPushButton) if b.text().strip()]
    assert controles, (
        "âncora: o painel das figuras não tem botão com texto, e P1 não compararia nada"
    )
    fora_do_lugar = [
        (rotulo, controle)
        for rotulo, onde, texto in itens
        for controle in controles
        if controle in texto and onde != "graficos"
    ]
    assert not fora_do_lugar, (
        f"a prosa do recorte nomeia um controle do painel das figuras numa seção que não é a dele: "
        f"{fora_do_lugar}. Os controles são {controles}, e eles vivem na aba «Gráficos»"
    )

    # --- P2: as treze figuras estão DENTRO do produto, então nenhuma está «fora» -----------------
    assert _painel_das_figuras_alimenta_o_registo(), (
        "âncora: o painel das figuras não alimenta o registo inteiro — se as figuras de facto não "
        "existissem, P2 estaria a proibir uma frase verdadeira"
    )
    declaradas_fora = [
        (rotulo, achado.group(0)[:80])
        for rotulo, _onde, texto in itens
        if (achado := FIGURA_DECLARADA_FORA.search(texto))
    ]
    assert not declaradas_fora, (
        f"a prosa do recorte diz que uma figura está fora do produto: {declaradas_fora}. As treze "
        "estão no registo e todas têm montador"
    )

    # --- P3: todo painel com `onde` tem aba própria, então «não numa aba própria» é falso --------
    com_aba = {chave for aba in ABAS for chave in aba.paineis}
    assert com_aba, "âncora: nenhuma aba declara painel, e P3 não teria referente"
    negam_a_aba = [
        rotulo for rotulo, onde, texto in itens if onde in com_aba and SEM_ABA_PROPRIA.search(texto)
    ]
    assert not negam_a_aba, (
        f"a prosa diz «não numa aba própria» sobre um painel que TEM aba: {negam_a_aba}. Todo "
        f"painel de `ABAS` é mostrado em exactamente uma aba: {sorted(com_aba)}"
    )

    # --- A quarta instância, por ENUMERAÇÃO, porque nenhum predicado a alcança -------------------
    proibidas = {
        "No MESMO painel da qualidade": "onde o par do PLS-DA é desenhado",
        "só mudam o que está à vista": "os controles do painel de figura, na aba de qualidade",
    }
    vivas = [
        (rotulo, frase) for rotulo, _onde, texto in itens for frase in proibidas if frase in texto
    ]
    assert not vivas, (
        f"frases que a U2 tornou falsas e continuam na prosa do recorte: {vivas}. O assunto de "
        f"cada uma está em {proibidas}"
    )


def test_u2_os_tres_predicados_da_prosa_ACUSAM_com_a_frase_reposta() -> None:  # noqa: N802
    """**A planta, e ela repõe cada uma das frases que a mudança da figura tirou.**

    Sem esta função os predicados acima são asserções de ausência sobre textos que talvez nenhum
    padrão alcance — verdes por não medirem nada. Aqui cada padrão cai sobre a frase que de facto
    existia na árvore de `HEAD`, verbatim, e tem de acusar.

    **As frases são as da árvore anterior e não inventadas**, e é isso que separa esta planta de
    uma que testa o `regex` contra si mesmo.
    """
    plantas = {
        "P2, do motivo do PLS-DA": (
            FIGURA_DECLARADA_FORA,
            "O que continua FORA é o GRÁFICO de dispersão: um painel de pontos sem escala de "
            "confiança convida à leitura que o Q² desmente.",
        ),
        "P2, do texto que o profissional lê": (
            FIGURA_DECLARADA_FORA,
            "O gráfico de dispersão continua fora: um painel de pontos convida justamente à "
            "leitura que os dois números desmentem.",
        ),
        "P3, do motivo do PLS-DA": (
            SEM_ABA_PROPRIA,
            "No MESMO painel da qualidade e não numa aba própria, porque o par `q2` mais "
            "`permutation_p` é escrito junto do resto.",
        ),
    }
    for rotulo, (padrao, frase) in plantas.items():
        assert padrao.search(frase), f"o padrão de `{rotulo}` não acusa a frase que o motivou"

    # E o contraste: a prosa de HOJE, que fala do mesmo assunto, NÃO é acusada. Sem esta metade o
    # padrão poderia estar a casar qualquer frase que contenha a palavra «figura».
    corretas = (
        "A aba «Qualidade do resultado» mostra o conjunto completo depois de uma execução. Ela é "
        "preenchida pelo resultado, e não há nada a clicar nela — as figuras mudaram-se para a "
        "aba «Gráficos», que fica logo ao lado.",
        "O gráfico de dispersão está na aba «Gráficos», e o Q² vai escrito no TÍTULO dele.",
    )
    for frase in corretas:
        assert not FIGURA_DECLARADA_FORA.search(frase), (
            f"o padrão acusa uma frase correta: {frase!r}. Um padrão que acusa o legítimo morre no "
            "primeiro dia"
        )
        assert not SEM_ABA_PROPRIA.search(frase)


# ---------------------------------------------------------------------------
# Toda seção com painel tem linha, e nenhum texto manda à versão web
# ---------------------------------------------------------------------------


def _secoes_sem_linha(textos: tuple[str, ...], secoes: tuple[SecaoDoRecorte, ...]) -> list[str]:
    """As seções com painel cujo título para quem opera não abre nenhum rótulo desenhado.

    O título é procurado como cabeça (a primeira linha, depois do prefixo de estado) e não como
    substring solta: o título de uma seção citado dentro do corpo de outra não é a linha dela.
    """
    cabecas = {t.split("\n", 1)[0].split(" — ", 1)[-1] for t in textos}
    return [s.titulo for s in secoes if s.onde and traduz(s.titulo)[0] not in cabecas]


def test_toda_secao_com_painel_tem_linha_na_aba_e_no_estado_derivado(qtbot: QtBot) -> None:
    """Cada seção com painel tem linha na aba «Métodos e limites», lida do WIDGET.

    O estado é derivado e não pinado: o prefixo esperado sai de `janela.operaveis` e de
    `janela.calculam_aqui`, e o dia em que um painel mudar de balde muda os dois lados juntos. Que
    o membro de `calculam_aqui` calcule de facto é conferido por emissão em
    `test_quem_calcula_aqui_e_exactamente_quem_calculam_aqui_declara`.
    """
    janela = JanelaPrincipal(_TransporteMudo())
    qtbot.addWidget(janela)
    textos = janela.recorte.textos()
    com_painel = [s for s in SECOES_DO_RECORTE if s.onde]
    assert com_painel, "âncora: nenhuma seção tem painel e o laço abaixo não olharia nada"
    assert textos, "âncora: a janela não alimentou o painel do recorte"

    assert _secoes_sem_linha(textos, SECOES_DO_RECORTE) == [], (
        f"seções com painel sem linha na aba: {_secoes_sem_linha(textos, SECOES_DO_RECORTE)}"
    )
    cabecas = {t.split("\n", 1)[0] for t in textos}
    for secao in com_painel:
        opera = secao.onde in janela.operaveis or secao.onde in janela.calculam_aqui
        prefixo = PREFIXO_COBERTO if opera else PREFIXO_SO_LEITURA
        esperado = f"{prefixo} — {traduz(secao.titulo)[0]}"
        assert esperado in cabecas, (
            f"o painel {secao.onde!r} está composto e a linha da seção {secao.titulo!r} não traz o "
            f"estado derivado. Esperava {esperado!r} entre {sorted(cabecas)}"
        )


def test_sem_a_secao_a_verificacao_da_linha_ACUSA(  # noqa: N802
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """O par causal, em processo: a planta é tirar a seção das vias da estrutura que a função lê.

    A mesma chamada, com e sem a seção, e a verificação tem de acusar a seção que ficou sem linha.
    """
    import refino.desktop.recorte as modulo

    real = SECOES_DO_RECORTE
    vias = [s for s in real if s.onde == "vias"]
    assert len(vias) == 1, "âncora: sem a seção das vias a planta não teria o que retirar"

    def desenhado() -> tuple[str, ...]:
        return tuple(
            texto_do_item(cabeca, corpo)
            for cabeca, corpo in modulo.itens_para_a_janela(
                compostos=TODOS, operaveis=frozenset(), declarados=DECLARADOS
            )
        )

    assert _secoes_sem_linha(desenhado(), real) == [], "sem a planta a verificação já acusava"
    monkeypatch.setattr(modulo, "SECOES_DO_RECORTE", tuple(s for s in real if s.onde != "vias"))
    assert _secoes_sem_linha(desenhado(), real) == [vias[0].titulo], (
        "com a seção retirada a verificação não acusou a linha que faltou"
    )


def test_quem_calcula_aqui_e_exactamente_quem_calculam_aqui_declara(
    qtbot: QtBot, tmp_path: Path
) -> None:
    """`janela.calculam_aqui` é MEDIÇÃO: cada membro calcula ao clicar, e sem tocar em rota.

    **É o irmão de `test_quem_dispara_rota_e_exactamente_quem_operaveis_declara`** (em
    `tests/test_contrato_janela_api.py`), e mede do mesmo jeito — por **emissão** e não por
    introspecção. O transporte é `_TransporteMudo`, que levanta em qualquer chamada: um membro que
    precisasse de rota para calcular não chegaria à calculadora.

    **A bijeção é pelo dicionário `calcula`**: um painel novo em `calculam_aqui` sem emissão aqui
    reprova na igualdade das chaves, e não passa por omissão. O par causal é o contador da
    calculadora antes e depois do clique — zero antes, um depois.
    """
    janela = JanelaPrincipal(_TransporteMudo())
    qtbot.addWidget(janela)
    chamadas: list[tuple[object, ...]] = []

    def espia(*args: object, **kwargs: object) -> object:
        chamadas.append(args)
        raise ValueError("espia: a calculadora foi alcançada")

    def clica_nas_vias() -> None:
        painel = janela.vias
        painel._calcula = espia
        painel.recebe_volcano(
            {"volcano": {"features": ["1.0_100.0m/z"], "pvalue": [0.01], "statistic": [3.2]}}
        )
        chegou: list[str] = []
        painel._sinais.concluido.connect(lambda *_a: chegou.append("concluido"))
        painel._sinais.falhou.connect(lambda *_a: chegou.append("falhou"))
        assert chamadas == [], "âncora: a calculadora correu antes do clique"
        painel.botao.click()
        qtbot.waitUntil(lambda: bool(chegou), timeout=20000)

    def clica_em_enriquecer() -> None:
        """O segundo modo da aba de vias: o botão «Enriquecer» chega ao R sem rota.

        O executor do R é a espia; a tabela, as bibliotecas e o motor são injetados porque o que se
        mede é o caminho do clique até à calculadora, não o disco desta máquina.
        """
        from refino.desktop.janela.enriquecimento import OrigemDasBibliotecas
        from refino.desktop.vias_r import MotorDoR

        painel = janela.vias.enriquecimento
        tabela = tmp_path / "identificacao.csv"
        tabela.write_text(
            "Compound,Description\n"
            + "".join(f"f{i},{n}\n" for i, n in enumerate(("Glycine", "Urea", "Citric acid")))
            + "f3,Taurine\n",
            encoding="utf-8",
        )
        painel._executor = espia
        painel._motor = lambda: MotorDoR(tmp_path, tmp_path / "r.R", tmp_path, tmp_path / "b")
        painel._abre_arquivo = lambda _pai: tabela
        painel._origem_das_bibliotecas = lambda _m: OrigemDasBibliotecas(tmp_path, "teste", True)
        janela.vias.recebe_volcano(
            {
                "volcano": {
                    "features": ["f0", "f1", "f2", "f3"],
                    "pvalue": [0.001, 0.002, 0.003, 0.5],
                    "statistic": [0.0] * 4,
                }
            }
        )
        painel._carrega_tabela()
        painel._confirma_colunas()
        chegou: list[object] = []
        painel._sinais.concluido.connect(lambda r, _t: chegou.append(r))
        assert painel.botao_enriquece.isEnabled(), "âncora: o recorte não ligou o botão"
        painel.botao_enriquece.click()
        qtbot.waitUntil(lambda: bool(chegou), timeout=20000)

    # Cada painel pode ter mais de um botão que calcula; cada um tem de alcançar a calculadora.
    calcula = {"vias": (clica_nas_vias, clica_em_enriquecer)}
    assert set(calcula) == set(janela.calculam_aqui), (
        f"`calculam_aqui` declara {sorted(janela.calculam_aqui)} e este portão emite "
        f"{sorted(calcula)} — um membro sem emissão seria declaração sem medição"
    )
    for chave, emissores in calcula.items():
        for emite in emissores:
            antes = len(chamadas)
            emite()
            assert len(chamadas) == antes + 1, (
                f"o painel {chave!r} não calculou ao clicar ({emite.__name__})"
            )


def test_sem_calculam_aqui_a_linha_das_vias_ABRE_com_so_leitura_e_contradiz_o_corpo() -> None:  # noqa: N802
    """O par causal do balde, **em processo**: a mesma chamada, com e sem `calculam_aqui`.

    Sem ele reaparece o defeito da revisão de 25/09/2026, verbatim: a cabeça «SÓ PARA LEITURA» e,
    no corpo da mesma linha, o botão «Predizer vias» que *«calcula aqui dentro»*. Com ele a linha
    vai para os cobertos. **A planta cai sobre dado em que os dois desfechos diferem**.
    """
    titulo = traduz("Vias metabólicas")[0]

    def linha(calculam_aqui: frozenset[str]) -> tuple[str, str]:
        itens = itens_para_a_janela(
            compostos=TODOS,
            operaveis=frozenset(),
            declarados=DECLARADOS,
            calculam_aqui=calculam_aqui,
        )
        achadas = [(c, m) for c, m in itens if c.endswith(f" — {titulo}")]
        assert len(achadas) == 1, f"âncora: a linha das vias aparece {len(achadas)} vezes"
        return achadas[0]

    cabeca_sem, corpo_sem = linha(frozenset())
    assert cabeca_sem.startswith(PREFIXO_SO_LEITURA), cabeca_sem
    assert "calcula aqui dentro" in corpo_sem, "âncora: o corpo já não descreve o botão a calcular"

    cabeca_com, _ = linha(frozenset({"vias"}))
    assert cabeca_com.startswith(PREFIXO_COBERTO), cabeca_com


def test_calculam_aqui_fora_da_composicao_LEVANTA() -> None:  # noqa: N802
    """A guarda de `operaveis`, estendida: painel que calcula e não está em aba é defeito."""
    with pytest.raises(PainelDesconhecidoError, match="vias"):
        itens_para_a_janela(
            compostos=frozenset({"qualidade"}),
            operaveis=frozenset(),
            declarados=DECLARADOS,
            calculam_aqui=frozenset({"vias"}),
        )


# **O oráculo foi CONTADO antes de ser escrito**: sobre `PARA_QUEM_OPERA` de `HEAD`
# (`118690c`) este padrão casa **2** — as duas entradas que remetiam à versão web, «As cinco
# etapas» e «O fluxo» — e **0** na árvore de 25/09/2026. Nenhum falso positivo nas outras entradas.
REMETE_A_WEB: Final = re.compile(r"\bweb\b|Streamlit|navegador", re.IGNORECASE)


def test_nenhum_texto_para_quem_opera_manda_para_a_versao_web() -> None:
    """Nenhum texto da aba manda quem opera para uma versão web, que o executável não tem.

    Um texto de tela que diga *«continua só na versão web»* manda o usuário procurar uma superfície
    que não existe para ele — com a autoridade da aba cuja função é declarar limites. O texto tem de
    dizer o que falta NESTE aplicativo, e onde está o que existe.
    """
    assert PARA_QUEM_OPERA, "âncora: o mapa está vazio e a asserção seria vácua"
    remetem = sorted(
        chave
        for chave, (cabeca, corpo) in PARA_QUEM_OPERA.items()
        if REMETE_A_WEB.search(f"{cabeca}\n{corpo}")
    )
    assert not remetem, f"textos de tela que remetem à versão web: {remetem}"


def test_o_padrao_da_versao_web_ACUSA_as_frases_que_existiam() -> None:  # noqa: N802
    """A planta são as duas frases de `HEAD`, verbatim — não uma frase inventada contra o padrão."""
    antigas = (
        "A versão web mostra o que cada uma das cinco etapas decidiu e por quê. Aqui isso ainda "
        "não aparece: você vê o resultado final e as métricas, não o passo a passo.",
        "A visão geral de como as etapas se encadeiam continua só na versão web.",
    )
    for frase in antigas:
        assert REMETE_A_WEB.search(frase), f"o padrão não acusa a frase que o motivou: {frase!r}"
    # E o contraste: as frases de hoje, sobre o mesmo assunto, não são acusadas.
    for chave in ("As cinco etapas, e o que cada uma decidiu", "O fluxo"):
        cabeca, corpo = traduz(chave)
        assert not REMETE_A_WEB.search(f"{cabeca}\n{corpo}"), chave


# ---------------------------------------------------------------------------
# 25/09/2026 — a aba não pode negar TODO cálculo das famílias que a rota calcula
# ---------------------------------------------------------------------------

ROTAS: Final = REPO / "src" / "refino" / "api" / "rotas.py"

# As duas formas que a aba usou até 25/09/2026, e que ficaram falsas quando a primeira família
# passou a calcular, a ANOVA. O padrão é sobre a NEGAÇÃO TOTAL, e não sobre a lista — quais famílias
# calculam não vai escrito na aba, por decisão de desenho do próprio `recorte.py`.
NEGA_TODO_CALCULO: Final = re.compile(
    r"c[áa]lculo de nenhuma|nenhuma delas est[áa] pront|"
    r"ainda n[ãa]o existe [ée] o c[áa]lculo de cada uma",
    re.IGNORECASE,
)


def familias_que_a_rota_calcula() -> set[str]:
    """Os `campo` de família que têm ramo no despacho de `rotas.py`, lidos da fonte.

    Lido do código e não escrito aqui, pela mesma razão que a aba não os enumera: a lista muda a
    cada família implementada.
    """
    return set(re.findall(r'familia\.campo == "([a-z_]+)"', ROTAS.read_text(encoding="utf-8")))


def test_a_aba_nao_nega_o_calculo_das_familias_que_a_rota_calcula() -> None:
    """A aba dizia que nenhuma família calculava, e hoje a maioria calcula."""
    from refino.interface.fluxo import FAMILIAS_DE_ANALISE

    calculadas = familias_que_a_rota_calcula()
    assert calculadas, "âncora: nenhum ramo `familia.campo ==` em rotas.py; a leitura falhou"
    assert calculadas <= {f.campo for f in FAMILIAS_DE_ANALISE}, (
        f"o despacho nomeia campos que não são família: {calculadas}"
    )
    textos = [f"{cabeca}\n{corpo}" for cabeca, corpo in PARA_QUEM_OPERA.values()]
    assert textos, "âncora: sem textos, a asserção abaixo seria vácua"
    negam = [texto[:90] for texto in textos if NEGA_TODO_CALCULO.search(texto)]
    assert not negam, (
        f"a rota calcula {sorted(calculadas)}, e a aba diz que nenhuma calcula: {negam}"
    )


def test_o_padrao_do_calculo_ACUSA_as_duas_frases_de_HEAD() -> None:  # noqa: N802
    """A planta são as frases de `118690c`, verbatim — a do corpo e a do motivo."""
    antigas = (
        "ali viaja no pedido e aparece na barra de progresso como uma etapa por família. **O "
        "cálculo de nenhuma delas está pronto ainda**: a etapa passa num instante e não há figura "
        "nem número no fim.",
        "uma etapa por família. **O que ainda não existe é o cálculo de cada uma** — é das "
        "unidades U5 a U22 —, então a etapa fecha instantaneamente e não produz figura nem "
        "número.",
    )
    for frase in antigas:
        assert NEGA_TODO_CALCULO.search(frase), f"o padrão não acusa a frase de HEAD: {frase!r}"


def test_o_cartao_do_palpite_mostra_como_ele_decide_ao_abrir_os_detalhes(qtbot: QtBot) -> None:
    """O bloco «Como o Palpite decide» está nos detalhes do cartão, e aparece quando eles abrem.

    **O oráculo é o widget visível, e não a constante**: um texto certo num rótulo que o
    cartão nunca mostra deixaria o teste verde e a tela muda. Por isso as duas
    direções: fechado o bloco não se vê, aberto vê-se.
    """
    painel = PainelDeLeitura()
    qtbot.addWidget(painel)
    bloco = painel.como_o_palpite_decide
    assert bloco.text() == COMO_O_PALPITE_DECIDE
    assert bloco.textFormat() == Qt.TextFormat.MarkdownText
    assert painel.paineis_de_detalhes[CHAVE_DO_PALPITE].isAncestorOf(bloco)
    assert not bloco.isVisibleTo(painel), "âncora: com os detalhes fechados o bloco não se vê"
    painel.cartoes[CHAVE_DO_PALPITE].alterna.setChecked(True)
    qtbot.waitUntil(lambda: bloco.isVisibleTo(painel))


def test_a_explicacao_curta_do_palpite_diz_o_que_o_empate_faz() -> None:
    """No empate o Palpite continua a propor o vencedor da amostra — `decide_o_palpite` e a
    delegação de `auto=medida` —, e a frase dizia que ele «não escolhe»."""
    explicacao = EXPLICACOES_DA_LEITURA[CHAVE_DO_PALPITE]
    assert "continua a propor" in explicacao
    assert "não escolhe" not in explicacao


# ---------------------------------------------------------------------------
# A prosa do recorte diz onde as coisas ficam na janela de dez abas
# ---------------------------------------------------------------------------
#
# **O oráculo é a janela MONTADA, e não uma lista de palavras proibidas.** Cada predicado abaixo
# lê uma quantidade da janela — onde a faixa está, se as duas seções do preparo estão lado a lado,
# em que aba moram os eixos que o programa propõe, que botões existem — e só depois pergunta à
# prosa se ela diz o mesmo. Uma prosa certa sobre a janela de sete abas é errada sobre esta, e o
# que muda entre as duas é a medição, não o texto.
#
# **Contado antes de escrito, sobre `c566764`:** a prosa citava 7 botões, e 2 não existem
# («ver o recorte medido», «Palpite»); 2 entradas que citam «Preparar o estudo» usam um lado
# («à esquerda» e «à direita»), e a da faixa diz rodapé; e a frase de onde o valor proposto vem
# preenchido aparece em 5 textos — os dois dos pontos de leitura e três do recorte —, e os 5 erram
# a aba.

CHAVE_DA_FAIXA_NO_RECORTE: Final = "3. Trabalhos em curso"
_LADO: Final = re.compile(r"à esquerda|à direita|do outro lado", re.IGNORECASE)


def _janela_vestida(qtbot: QtBot) -> JanelaPrincipal:
    """A janela como o aplicativo a abre: a barra lateral e a faixa no topo (`veste_a_janela`)."""
    from PySide6.QtWidgets import QApplication

    from refino.desktop.janela import tema

    janela = JanelaPrincipal(_TransporteMudo())
    qtbot.addWidget(janela)
    tema.veste_a_janela(janela)
    janela.resize(1440, 900)
    janela.show()
    QApplication.processEvents()
    return janela


LugarDaFaixa = Literal["topo", "rodapé", "barra lateral"]


def _lugares_medidos(janela: JanelaPrincipal) -> tuple[LugarDaFaixa, bool]:
    """`(lugar_da_faixa, preparo_empilhado)`, medidos na geometria da janela vestida.

    A faixa está na barra lateral quando a `BarraLateral` a contém; fora dela, está no topo
    quando o fundo dela fica acima do topo da pilha das páginas, e no rodapé quando não. As duas
    seções de «Preparar o estudo» estão empilhadas quando começam na mesma coluna e a segunda
    começa abaixo do fim da primeira.
    """
    from PySide6.QtCore import QPoint

    from refino.desktop.janela.preparo import AbaDePreparo
    from refino.desktop.janela.tema import BarraLateral

    faixa = janela.faixa_de_trabalhos
    assert faixa.isAncestorOf(janela.trabalhos), "âncora: a faixa não leva os trabalhos"
    barra = janela.findChild(BarraLateral)
    assert barra is not None, "âncora: a janela vestida não tem a barra lateral"
    lugar: LugarDaFaixa
    if barra.isAncestorOf(faixa):
        lugar = "barra lateral"
    else:
        fundo = faixa.mapTo(janela, QPoint(0, faixa.height())).y()
        lugar = "topo" if fundo <= janela.abas.mapTo(janela, QPoint(0, 0)).y() else "rodapé"

    preparo = janela.findChild(AbaDePreparo)
    assert preparo is not None, "âncora: a aba de preparo não foi montada"
    arquivos, leitura = preparo.secao_dos_arquivos, preparo.secao_da_leitura
    canto_a = arquivos.mapTo(preparo, QPoint(0, 0))
    canto_b = leitura.mapTo(preparo, QPoint(0, 0))
    assert arquivos.height() > 0 and leitura.height() > 0, "âncora: as seções não têm altura"
    preparo_empilhado = (
        canto_a.x() == canto_b.x() and canto_b.y() >= canto_a.y() + arquivos.height()
    )
    return lugar, preparo_empilhado


_LUGARES_DA_FAIXA: Final[tuple[LugarDaFaixa, ...]] = ("topo", "rodapé", "barra lateral")


def _o_que_a_prosa_erra_sobre_o_lugar(
    textos: Mapping[str, tuple[str, str]], *, lugar_da_faixa: LugarDaFaixa, preparo_empilhado: bool
) -> list[str]:
    """As afirmações de lugar da prosa que a geometria medida desmente — o oráculo é o argumento."""
    erros: list[str] = []
    faixa = " ".join(textos[CHAVE_DA_FAIXA_NO_RECORTE][1].split())
    errados = [lugar for lugar in _LUGARES_DA_FAIXA if lugar != lugar_da_faixa]
    if any(lugar in faixa for lugar in errados) or lugar_da_faixa not in faixa:
        erros.append(f"{CHAVE_DA_FAIXA_NO_RECORTE}: a faixa está em «{lugar_da_faixa}»")
    if preparo_empilhado:
        for chave, (_cabeca, corpo) in textos.items():
            normal = " ".join(corpo.split())
            lado = _LADO.search(normal)
            if "«Preparar o estudo»" in normal and lado is not None:
                erros.append(f"{chave}: «{lado.group(0)}» numa aba de seções empilhadas")
    return erros


def test_a_prosa_do_recorte_situa_a_faixa_e_o_preparo_onde_a_janela_vestida_os_pos(
    qtbot: QtBot,
) -> None:
    """A faixa «Trabalhos em curso» e as duas seções do preparo, onde a janela de dez abas as põe.

    A janela de sete abas tinha a faixa no rodapé e os dois painéis do preparo lado a lado; a de dez
    tem as duas seções uma abaixo da outra (`AbaDePreparo`), e a faixa, que já esteve no topo,
    está no fim da barra lateral (`tema._monta_faixa_de_trabalhos`). A prosa que dizia
    *«FAIXA FIXA do rodapé»*, *«à esquerda»* e *«à direita»* descrevia a primeira, e o painel
    desta aba existe para não mentir sobre a janela.

    **O par causal é em processo, nas duas pontas:** a prosa antiga é acusada com a geometria de
    hoje, e a prosa de hoje é acusada com a geometria oposta — o predicado depende da medição.
    """
    janela = _janela_vestida(qtbot)
    lugar_da_faixa, preparo_empilhado = _lugares_medidos(janela)
    assert lugar_da_faixa == "barra lateral" and preparo_empilhado, (
        "âncora: a geometria medida não é a da janela de dez abas — faixa em "
        f"{lugar_da_faixa!r}, preparo empilhado {preparo_empilhado}"
    )
    assert (
        _o_que_a_prosa_erra_sobre_o_lugar(
            PARA_QUEM_OPERA, lugar_da_faixa=lugar_da_faixa, preparo_empilhado=preparo_empilhado
        )
        == []
    )

    antiga = {
        **PARA_QUEM_OPERA,
        CHAVE_DA_FAIXA_NO_RECORTE: ("x", "Esta é a FAIXA FIXA do rodapé, visível em qualquer aba."),
        "1. Os seus arquivos": ("x", "Na aba «Preparar o estudo», à esquerda: os CSV."),
    }
    assert (
        len(
            _o_que_a_prosa_erra_sobre_o_lugar(
                antiga, lugar_da_faixa="barra lateral", preparo_empilhado=True
            )
        )
        == 2
    ), "a prosa da janela de sete abas não foi acusada nas duas afirmações de lugar"
    outros: tuple[LugarDaFaixa, ...] = ("topo", "rodapé")
    for outro in outros:
        assert _o_que_a_prosa_erra_sobre_o_lugar(
            PARA_QUEM_OPERA, lugar_da_faixa=outro, preparo_empilhado=True
        ) == [f"{CHAVE_DA_FAIXA_NO_RECORTE}: a faixa está em «{outro}»"], (
            f"com a faixa medida em «{outro}», a prosa de hoje devia ser acusada — o predicado não "
            "lê a geometria"
        )


def _abas_dos_eixos_propostos(janela: JanelaPrincipal) -> set[str]:
    """As abas em que moram os controles dos eixos que o programa PROPÕE (`EIXOS_PROPOSTOS`)."""
    from refino.interface.fluxo import EIXOS_PROPOSTOS

    assert EIXOS_PROPOSTOS, "âncora: nenhum eixo proposto, e o conjunto seria vazio"
    controles = janela.pipeline._controles_de_valor()
    assert janela.abas.count() == len(ABAS), "âncora: a pilha não tem uma página por aba"
    abas: set[str] = set()
    for eixo in EIXOS_PROPOSTOS:
        donas = [
            ABAS[i].titulo
            for i in range(janela.abas.count())
            if janela.abas.widget(i).isAncestorOf(controles[eixo])
        ]
        assert len(donas) == 1, f"o controle de {eixo} está em {donas}, e não numa aba só"
        abas |= set(donas)
    return abas


_FRASE_DA_PROPOSTA: Final = re.compile(
    r"(?=.*preenchid)(?=.*(?:prop[oõ]|recomend))|valor propost", re.IGNORECASE
)


def _abas_ditas_da_proposta(texto: str) -> list[set[str]]:
    """Para cada frase que diz onde o valor proposto vem, as abas que ela cita (ou nenhuma)."""
    titulos = {aba.titulo for aba in ABAS}
    frases = re.split(r"(?<=\.)\s+", " ".join(texto.replace("**", "").split()))
    return [
        {nome for nome in re.findall(r"«([^»]+)»", frase) if nome in titulos}
        for frase in frases
        if _FRASE_DA_PROPOSTA.search(frase)
    ]


def _o_que_erra_a_aba_da_proposta(textos: Mapping[str, str], esperado: set[str]) -> list[str]:
    return sorted(
        f"{chave}: diz {sorted(abas) or 'nenhuma aba'}"
        for chave, texto in textos.items()
        for abas in _abas_ditas_da_proposta(texto)
        if abas != esperado
    )


def test_o_valor_proposto_e_dito_na_aba_onde_os_eixos_propostos_moram(qtbot: QtBot) -> None:
    """A frase de onde o valor proposto vem preenchido nomeia a aba em que o controle mora.

    Os dois eixos que o programa propõe — `PROPOSTAS_POR_ROTA` — ficam os dois no quadro «Correção
    do sinal», na aba «Correção». A frase dos pontos de leitura dizia *«nas abas «Correção» e
    «Normalização»»*, a do palpite no recorte dizia *«do outro lado»*, e duas outras do recorte
    davam «Normalização» ou nenhuma aba. **A aba esperada sai da janela montada**, e não deste
    teste: o dia em que um eixo proposto mudar de aba, a prosa reprova até dizer a nova.
    """
    janela = JanelaPrincipal(_TransporteMudo())
    qtbot.addWidget(janela)
    esperado = _abas_dos_eixos_propostos(janela)
    textos = {f"recorte {chave}": corpo for chave, (_cabeca, corpo) in PARA_QUEM_OPERA.items()}
    textos |= {f"leitura {chave}": r.text() for chave, r in janela.leitura.eixos_do_ponto.items()}

    # A âncora de presença vai ANTES: sem frase nenhuma, a asserção seguinte seria vácua.
    com_frase = [chave for chave, texto in textos.items() if _abas_ditas_da_proposta(texto)]
    assert len(com_frase) >= 3, f"âncora: só {com_frase} dizem onde o valor proposto vem"
    assert any(chave.startswith("leitura") for chave in com_frase), (
        "âncora: a frase dos pontos de leitura deixou de casar — o predicado ficou cego a ela"
    )
    assert _o_que_erra_a_aba_da_proposta(textos, esperado) == []

    plantado = {"x": "O que ele propõe já vem preenchido nas abas «Correção» e «Normalização»."}
    assert _o_que_erra_a_aba_da_proposta(plantado, esperado) == [
        "x: diz ['Correção', 'Normalização']"
    ], "a frase com a aba a mais não foi acusada"


# **`clicando` entrou depois da revisão:** a reescrita da faixa escreveu *«clicando em
# «Parar o trabalho selecionado»»*, e a forma de antes deixava essa citação fora do oráculo — 7
# citações na base, 6 na árvore, e a que saiu era desta unidade.
_BOTAO_CITADO: Final = re.compile(
    r"\bbot[ãa]o\s+«([^»]+)»|\bcli(?:que|car|cando)\s+(?:em|no|na)\s+«([^»]+)»", re.IGNORECASE
)
_MARCA_DINAMICA: Final = "{}"


def _literais_de_tela() -> dict[str, str]:
    """Os literais de texto de `desktop/` e de `ui/fluxo.py`, lidos pela árvore sintática.

    **A população é o código que pode chegar à tela, e não uma lista de constantes escolhidas.** A
    primeira versão deste portão lia `PARA_QUEM_OPERA` e `SEM_CORRETOR_DECIDIDO`, e a revisão achou
    a irmã fora delas: *«Clique em «Gerar» outra vez»*, em `artefatos.mostra_desistencia`. Ficam de
    fora os literais que são uma instrução sozinha — docstrings de módulo, classe, função e
    atributo —, porque nunca são valor. Uma f-string entra com `{}` no lugar de cada campo, e um
    nome citado que contém `{}` é composto em tempo de execução e não se confere aqui.
    """
    raiz = REPO / "src" / "refino"
    arquivos = [*sorted((raiz / "desktop").rglob("*.py")), raiz / "interface" / "fluxo.py"]
    literais: dict[str, str] = {}
    for caminho in arquivos:
        arvore = ast.parse(caminho.read_text(encoding="utf-8"))
        soltos = {
            id(no.value)
            for no in ast.walk(arvore)
            if isinstance(no, ast.Expr) and isinstance(no.value, ast.Constant)
        }
        dentro_de_f = {
            id(parte)
            for no in ast.walk(arvore)
            if isinstance(no, ast.JoinedStr)
            for parte in no.values
        }
        nome = caminho.relative_to(REPO).as_posix()
        for no in ast.walk(arvore):
            if isinstance(no, ast.JoinedStr):
                texto = "".join(
                    p.value if isinstance(p, ast.Constant) else _MARCA_DINAMICA for p in no.values
                )
            elif (
                isinstance(no, ast.Constant)
                and isinstance(no.value, str)
                and id(no) not in soltos
                and id(no) not in dentro_de_f
            ):
                texto = no.value
            else:
                continue
            literais[f"{nome}:{no.lineno}:{no.col_offset}"] = texto
    return literais


def _botoes_citados_que_nao_existem(textos: Mapping[str, str], botoes: set[str]) -> list[str]:
    return sorted(
        f"{chave}: «{nome}»"
        for chave, texto in textos.items()
        for par in _BOTAO_CITADO.findall(" ".join(texto.split()))
        for nome in par
        if nome and _MARCA_DINAMICA not in nome and nome not in botoes
    )


def test_todo_botao_que_a_prosa_manda_clicar_existe_na_janela(qtbot: QtBot) -> None:
    """«botão «X»», «clique em «X»» e «clicando em «X»» no texto de tela: X é um botão da janela.

    A população é a prosa do recorte e **todo literal de `desktop/` e de `ui/fluxo.py`**
    (`_literais_de_tela`).
    Na janela de sete abas o terceiro botão do preparo chamava-se «Palpite»; na de dez, o cartão
    chama-se «Palpite» e o botão dele «Executar». O botão do recorte medido escreve-se «Ver o
    recorte medido» na tela, com maiúscula. E não há botão «Gerar»: os três do painel dos artefatos
    dizem «Gerar e baixar …», e a desistência deles manda clicar no nome inteiro, lido do botão.

    **Os botões são os da janela vestida**, que é a que o aplicativo abre: a barra lateral também é
    de botões, e uma prosa que mandasse clicar numa aba seria conferida contra ela.
    """
    from PySide6.QtWidgets import QAbstractButton

    janela = _janela_vestida(qtbot)
    botoes = {b.text() for b in janela.findChildren(QAbstractButton) if b.text()}
    assert len(botoes) > 50, f"âncora: a janela montada tem só {len(botoes)} botões com texto"
    textos = {chave: corpo for chave, (_cabeca, corpo) in PARA_QUEM_OPERA.items()}
    literais = _literais_de_tela()
    assert len(literais) > 1000, f"âncora: só {len(literais)} literais lidos de desktop/"
    textos |= literais
    citados = [
        n
        for t in textos.values()
        for p in _BOTAO_CITADO.findall(" ".join(t.split()))
        for n in p
        if n
    ]
    assert len(citados) >= 10, f"âncora: a prosa cita só {citados}, e o predicado ficou cego"
    assert "Parar o trabalho selecionado" in citados, (
        "âncora: a citação da faixa com «clicando em» saiu do alcance do predicado"
    )
    assert _botoes_citados_que_nao_existem(textos, botoes) == []

    plantado = {
        "x": "abre o botão «ver o recorte medido», ou clique em «Palpite»",
        "artefatos": "Clique em «Gerar» outra vez para voltar a acompanhar",
        "faixa": "clicando em «Parar tudo»",
    }
    assert _botoes_citados_que_nao_existem(plantado, botoes) == [
        "artefatos: «Gerar»",
        "faixa: «Parar tudo»",
        "x: «Palpite»",
        "x: «ver o recorte medido»",
    ], "os nomes que a janela de dez abas não tem não foram acusados"


_EM_BAIXO: Final = re.compile(r"rodap[ée]|de baixo|em baixo|embaixo|no fundo|inferior", re.I)
_EM_CIMA: Final = re.compile(r"\btopo\b|de cima|em cima|superior", re.I)


def _frases_que_poem_a_faixa(textos: Mapping[str, str]) -> list[tuple[str, str]]:
    """`(chave, frase)` de cada frase que fala da faixa e dá um lugar a ela."""
    return [
        (chave, frase)
        for chave, texto in textos.items()
        for frase in re.split(r"(?<=[.;:!?])\s+", " ".join(texto.replace("**", "").split()))
        if re.search(r"\bfaixa", frase, re.I)
        and (_EM_BAIXO.search(frase) or _EM_CIMA.search(frase))
    ]


def _o_que_poe_a_faixa_fora_do_lugar(
    textos: Mapping[str, str], *, faixa_em_cima: bool
) -> list[str]:
    """As frases que põem a faixa no lado oposto ao que a geometria mediu; a medição é argumento."""
    errado = _EM_BAIXO if faixa_em_cima else _EM_CIMA
    return sorted(
        f"{chave}: {frase}"
        for chave, frase in _frases_que_poem_a_faixa(textos)
        if errado.search(frase)
    )


def test_nenhum_texto_de_tela_poe_a_faixa_onde_a_janela_vestida_nao_a_pos(qtbot: QtBot) -> None:
    """Todo literal de `desktop/` e de `ui/fluxo.py` que dá lugar à faixa diz o que a janela mede.

    **A irmã que a revisão achou fora do recorte**: a estimativa da medição mandava parar o
    trabalho *«na faixa de baixo»*, e a faixa está no topo da janela vestida. O teste da
    prosa do recorte lia só a entrada da faixa em `PARA_QUEM_OPERA`; este lê todo literal que pode
    chegar à tela (`_literais_de_tela`) e as duas frases da estimativa compostas em tempo de
    execução, e pergunta à geometria de que lado a faixa está.

    **O par causal é em processo:** a frase antiga é acusada com a geometria de hoje, e as frases
    de hoje são acusadas com a geometria oposta.
    """
    from refino.interface.fluxo import estimativa_da_medicao

    janela = _janela_vestida(qtbot)
    faixa_em_cima, _preparo = _lugares_medidos(janela)
    assert faixa_em_cima, "âncora: a faixa medida não está acima das páginas"

    textos = _literais_de_tela()
    textos |= {
        f"estimativa_da_medicao(serial={serial})": estimativa_da_medicao(2184, serial=serial)
        for serial in (False, True)
    }
    # A âncora de presença vai ANTES: sem frase de lugar, a asserção seguinte seria vácua.
    com_lugar = _frases_que_poem_a_faixa(textos)
    assert len(com_lugar) >= 3, f"âncora: só {com_lugar} dão lugar à faixa"
    assert any(chave.startswith("estimativa_da_medicao") for chave, _ in com_lugar), (
        "âncora: a frase composta da estimativa deixou de dar lugar à faixa"
    )
    assert _o_que_poe_a_faixa_fora_do_lugar(textos, faixa_em_cima=faixa_em_cima) == []

    antiga = {
        "x": "A janela não desiste de esperar, e você pode parar o trabalho na faixa de baixo."
    }
    assert _o_que_poe_a_faixa_fora_do_lugar(antiga, faixa_em_cima=True) == [f"x: {antiga['x']}"], (
        "a frase da janela de sete abas não foi acusada"
    )
    assert len(_o_que_poe_a_faixa_fora_do_lugar(textos, faixa_em_cima=False)) == len(com_lugar), (
        "com a faixa medida em baixo, cada frase de hoje devia ser acusada — o predicado não lê a "
        "geometria"
    )
