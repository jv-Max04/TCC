"""O que a janela diz ENQUANTO a execução corre, e o que ela diz de QUEM escolheu o corretor.

Achados da verificação da janela e frases de tela, num arquivo só porque são todos sobre a mesma
pergunta: *o que está a acontecer com o trabalho que eu disparei?*

- a tela dizia `executando: 0/7 etapas` de 29 s a 362 s e saltava para «concluído»; passa a dizer
  qual etapa corre, com que corretor, e há quanto tempo — sem fração que o servidor não mede.
- a faixa dizia «sem leitura» depois de o painel dizer «concluído»; passa a reler-se sozinha
  enquanto houver trabalho em curso, e para quando todos terminam.
- com «Automático» ticado, nenhuma superfície dizia QUAL corretor a política escolheu; passa a
  dizê-lo na linha do corretor da execução, na Análise, com a ressalva da aba de limites.
- o aviso de fecho de `trabalhos.py` fala com quem usa o executável.
- a faixa listava chave interna, identificador e estado cru numa linha de texto; passa a ter, por
  trabalho, o nome legível, o estado, a barra de etapas, a etapa em curso e o botão de parar
  daquela linha.

**Cada conserto tem o seu teste que reprova sem ele**, e foi visto reprovar: cada metade do
conserto foi desligada à mão e o teste correspondente corrido contra ela.

## Escopo negativo

**Nada aqui corre um SERRF.** O que se afirma é o que a janela escreve a partir das leituras que a
rota devolve; o número de 6 min do `pos_hilic` vem da verificação da janela e não é remedido. **E
nada aqui mede fração dentro da correção**: ela continua a não existir no servidor, e é por isso que
a frase nova não a inventa.
"""

from __future__ import annotations

import ast
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

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

from refino.api.progresso import ESTADOS, TERMINAIS  # noqa: E402
from refino.desktop.janela import pipeline as modulo_do_pipeline  # noqa: E402
from refino.desktop.janela import principal as modulo_da_janela  # noqa: E402
from refino.desktop.janela.pipeline import (  # noqa: E402
    EM_CURSO,
    PainelDePipeline,
    texto_da_etapa_em_curso,
)
from refino.desktop.janela.principal import ABAS, JanelaPrincipal  # noqa: E402
from refino.desktop.janela.trabalhos import (  # noqa: E402
    AVISO_DO_FECHO,
    NAO_LIDO,
    NOMES_DOS_TRABALHOS,
    PARANDO,
    ROTULOS_DOS_ESTADOS,
    SEM_TRABALHO_A_PARAR,
    TONS_DOS_ESTADOS,
    PainelDeTrabalhos,
    nome_do_trabalho,
)
from refino.desktop.recorte import PARA_QUEM_OPERA  # noqa: E402
from refino.interface.cliente import Resposta  # noqa: E402

RAIZ = Path(__file__).resolve().parents[1]
ROTAS = RAIZ / "src" / "refino" / "api" / "rotas.py"

# O corpo do `202` do `POST /execucoes` com delegação, na forma de `TrabalhoAceito` — os campos
# são os de `DelegacaoResposta`, e o `caveat` vai como a rota o monta.
_DELEGACAO = {
    "rotulo": "aprendida",
    "politica": "aprendida",
    "correction": "serrf",
    "confidence": 0.91,
    "abstains": False,
    "assessable": True,
    "rationale": "margem 0,40 entre os dois melhores",
    "evidence": {},
    "caveat": {
        "item": "--auto aprendida",
        "resumo": "política APRENDIDA: …",
        "detalhe": "…",
        "evidencia": "…",
        "portao_inerte": "…",
    },
}


def _painel(qtbot: Any, instantes: list[float]) -> PainelDePipeline:
    """Um painel cujo relógio percorre `instantes`, um por leitura — o arnês da janela.

    Antes recebia `automatico`, que ticava ou desticava a caixa «Automático»; a caixa saiu, e
    quem diz se o pedido delegou é `anota_o_pedido`, como já era nos testes do aceite.
    """
    restantes = list(instantes)
    painel = PainelDePipeline(relogio=lambda: restantes.pop(0))
    qtbot.addWidget(painel)
    return painel


def _progresso(estado: str, feito: int, total: int) -> dict[str, Any]:
    """Um corpo de `ProgressoResposta`, com os campos obrigatórios que a rota serve."""
    return {
        "identificador": "t",
        "estado": estado,
        "feito": feito,
        "total": total,
        "fracao": feito / total,
        "terminou": False,
    }


# ---------------------------------------------------------------------------
# «0/7 etapas» parado durante a correção inteira
# ---------------------------------------------------------------------------


def test_item85_a_primeira_etapa_e_nomeada_com_o_corretor_e_o_contador_fica(qtbot: Any) -> None:
    """Com `feito == 0` e `executando`, a etapa que corre é a correção — e a tela diz qual.

    **O contador `0/7 etapas` FICA**, porque é verdadeiro e é o que outros testes pinam; o que muda
    é que ele deixa de ser a única coisa na linha. A frase declara que a barra só anda entre
    etapas, que é o que faz o «parado» deixar de parecer travado sem inventar fração.
    """
    painel = _painel(qtbot, [130.0])
    # O pedido da Análise, como `_dispara_etapa(3)` o anota: o corretor da entrada, sem delegar.
    # Até 09/11/2026 quem o anotava aqui era o botão «Executar pipeline» (`_pede_disparo`),
    # que saiu.
    painel.anota_o_pedido("serrf", delegou=False)
    painel.aceita_disparo({"identificador": "t", "total": 7, "reclamado": False}, 100.0)

    texto = painel.recebe_leitura(EM_CURSO, _progresso("executando", 0, 7))

    # «Executando» com maiúscula: a linha abre frase.
    assert "Executando: 0/7 etapas" in texto, f"o contador saiu da linha: {texto}"
    assert "etapa 1 de 7" in texto, f"a etapa em curso não foi nomeada: {texto}"
    assert "correção" in texto and "«serrf»" in texto, (
        f"a primeira etapa não diz que é a correção, nem com que corretor: {texto}"
    )
    assert "decorrido: 30 s desde o disparo" in texto, f"o tempo da etapa sumiu: {texto}"
    assert "%" not in texto, f"a frase inventou fração que o servidor não mede: {texto}"


def test_item85_as_etapas_seguintes_dizem_ha_quanto_tempo_correm(qtbot: Any) -> None:
    """Depois da primeira, a etapa `k+1` corre desde a PRIMEIRA leitura que viu `feito == k`.

    **«Há pelo menos», e não «há»**, porque o fim da etapa anterior aconteceu entre duas leituras:
    a janela só sabe que ela já tinha acabado quando olhou. O número é um piso honesto.

    **E a contagem da etapa não recomeça a cada leitura**: um número que volta a zero a cada
    consulta é plausível e errado.
    """
    painel = _painel(qtbot, [130.0, 200.0, 245.0])
    painel.anota_o_pedido("loess", delegou=False)
    painel.aceita_disparo({"identificador": "t", "total": 7, "reclamado": False}, 100.0)

    painel.recebe_leitura(EM_CURSO, _progresso("executando", 0, 7))
    primeira_vez = painel.recebe_leitura(EM_CURSO, _progresso("executando", 1, 7))
    depois = painel.recebe_leitura(EM_CURSO, _progresso("executando", 1, 7))

    assert "etapa 2 de 7" in primeira_vez, primeira_vez
    assert "há pelo menos" not in primeira_vez, (
        f"na leitura que VIU a etapa começar, ela corre há zero segundos: {primeira_vez}"
    )
    assert "etapa 2 de 7" in depois and "há pelo menos 45 s" in depois, (
        f"a etapa 2 foi vista aos 100 s de decorrido e esta leitura é aos 145 s: {depois}"
    )
    assert "correção" not in depois, f"a etapa 2 foi chamada de correção: {depois}"


def test_item85_os_quatro_desfechos_da_frase_da_etapa() -> None:
    """**Quatro desfechos, e o `else` vem primeiro**: sem frase quando não se sabe.

    | Corpo | Frase |
    |---|---|
    | `pendente` | nenhuma — a thread ainda não pegou, e dizer «a correção corre» seria falso |
    | `executando`, `feito == 0` | a correção, com o corretor |
    | `executando`, `0 < feito < total` | a etapa `feito + 1`, sem nome |
    | `executando`, `feito == total` | nenhuma — todas marcadas, falta só o resultado |
    """
    assert (
        texto_da_etapa_em_curso(_progresso("pendente", 0, 7), corretor="serrf", desde_s=None) == ""
    )
    assert texto_da_etapa_em_curso(_progresso("executando", 7, 7), corretor=None, desde_s=3.0) == ""
    assert texto_da_etapa_em_curso(None, corretor="serrf", desde_s=None) == ""

    primeira = texto_da_etapa_em_curso(_progresso("executando", 0, 7), corretor=None, desde_s=None)
    assert "etapa 1 de 7" in primeira and "correção" in primeira
    assert "«" not in primeira, f"sem corretor conhecido, a frase inventou um: {primeira}"

    meio = texto_da_etapa_em_curso(_progresso("executando", 3, 7), corretor="serrf", desde_s=12.4)
    assert "etapa 4 de 7" in meio and "há pelo menos 12 s" in meio, meio


def _primeira_etapa_da_execucao(fonte: str) -> str | None:
    """O primeiro literal da tupla de etapas que `/execucoes` passa a `_despacha`. Por AST.

    Acha a função decorada com `router.post("/execucoes", …)` e, nela, o `return _despacha(…)`:
    o quarto argumento posicional é a tupla. `None` quando a forma não é a esperada — e o teste
    reprova com isso, que é a direção segura.
    """
    for no in ast.walk(ast.parse(fonte)):
        if not isinstance(no, ast.FunctionDef):
            continue
        rota = [
            d
            for d in no.decorator_list
            if isinstance(d, ast.Call)
            and d.args
            and isinstance(d.args[0], ast.Constant)
            and d.args[0].value == "/execucoes"
        ]
        if not rota:
            continue
        for filho in ast.walk(no):
            if (
                isinstance(filho, ast.Call)
                and isinstance(filho.func, ast.Name)
                and filho.func.id == "_despacha"
                and len(filho.args) >= 4
                and isinstance(filho.args[3], ast.Tuple)
                and filho.args[3].elts
                and isinstance(filho.args[3].elts[0], ast.Constant)
            ):
                return str(filho.args[3].elts[0].value)
    return None


def test_item85_a_etapa_que_a_janela_nomeia_e_a_primeira_que_a_rota_declara() -> None:
    """**A janela nomeia UMA etapa, e o nome dela é conferido contra a rota** — não repetido.

    A tupla de etapas vive literal em `api/rotas.py` e a resposta de progresso não traz nomes. Pôr
    a tupla inteira na janela seria uma segunda fonte de verdade; nomear só a primeira, e conferi-la
    aqui, é o mínimo que torna a frase verdadeira.

    **A planta cai sobre dado em que as duas coisas diferem**: com a ordem trocada na fonte,
    o extrator devolve outro nome e a igualdade reprova.
    """
    fonte = ROTAS.read_text(encoding="utf-8")
    assert _primeira_etapa_da_execucao(fonte) == modulo_do_pipeline.ETAPA_INICIAL_DA_EXECUCAO, (
        "a primeira etapa de `/execucoes` já não é a que a janela chama de «a correção» — a frase "
        "do progresso nomearia a etapa errada"
    )

    ancora = '(\n            "correcao",\n            "pipeline",'
    assert fonte.count(ancora) == 1, f"a âncora da planta casa {fonte.count(ancora)} vezes"
    plantado = fonte.replace(ancora, '(\n            "pipeline",\n            "correcao",', 1)
    assert _primeira_etapa_da_execucao(plantado) == "pipeline", (
        "a planta não mudou o que o extrator lê — a igualdade acima seria vácua"
    )


def test_item85_a_frase_da_tela_LE_o_nome_da_etapa_e_nao_um_literal(  # noqa: N802
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """**O portão acima confere a constante; este confere que a TELA a lê** — revisão de 25/09/2026.

    A frase escrevia «a correção» à mão, e `ETAPA_INICIAL_DA_EXECUCAO` só tinha o teste por leitor:
    actualizar a constante para o portão voltar a passar deixaria a tela a nomear a etapa antiga
    (o portão lia uma instância, e a tela mostrava outra). O par causal troca o nome do módulo
    e a frase tem de mudar com ele.
    """
    corpo = _progresso("executando", 0, 7)
    antes = texto_da_etapa_em_curso(corpo, corretor="serrf", desde_s=None)
    assert modulo_do_pipeline.NOME_DA_ETAPA_INICIAL in antes, antes

    monkeypatch.setattr(modulo_do_pipeline, "NOME_DA_ETAPA_INICIAL", "a etapa plantada")
    depois = texto_da_etapa_em_curso(corpo, corretor="serrf", desde_s=None)
    assert "a etapa plantada" in depois, f"a frase não leu o nome do módulo: {depois}"
    assert "a correção" not in depois, f"a frase ainda tem o literal: {depois}"


# ---------------------------------------------------------------------------
# Qual corretor a política escolheu
# ---------------------------------------------------------------------------


def test_d511_o_corretor_escolhido_pela_politica_aparece_no_resultado(qtbot: Any) -> None:
    """Com um pedido delegado, o `202` traz `delegacao`, e a janela passa a lê-la.

    **E ela SOBREVIVE ao resultado**: o rótulo do corretor é escrito no aceite — que é quando a
    decisão já está tomada, e a docstring de `DelegacaoResposta` diz por que ali — e o `200` não o
    apaga. Um rótulo que só vivesse até a primeira leitura seria a mesma ausência, um segundo
    depois.

    **O pedido delegado é anotado à mão desde 09/11/2026.** Quem delegava era o botão «Executar
    pipeline» com «Automático» ticado, e ele saiu com a aba «Executar e exportar»; a Análise, que é
    quem dispara `/execucoes` agora, nunca delega. O que fica testado é a leitura do aceite pelo
    painel, que continua a valer para qualquer `202` que traga a delegação.
    """
    painel = _painel(qtbot, [130.0, 200.0])
    painel.anota_o_pedido(painel.corretor.currentText(), delegou=True)
    antes = painel.rotulo_do_corretor.text()
    assert "serrf" not in antes, f"âncora: o rótulo já falava de `serrf` antes do aceite: {antes}"

    painel.aceita_disparo(
        {"identificador": "t", "total": 7, "reclamado": False, "delegacao": _DELEGACAO}, 100.0
    )
    no_aceite = painel.rotulo_do_corretor.text()
    painel.recebe_leitura(EM_CURSO, _progresso("executando", 0, 7))
    em_curso = painel.rotulo_de_progresso.text()
    painel.recebe_leitura(200, {"run_id": 3, "estado_concordancia": "nao_sobrescreveu"})
    no_fim = painel.rotulo_do_corretor.text()

    for texto in (no_aceite, no_fim):
        assert "«serrf»" in texto, f"o corretor escolhido não aparece: {texto}"
        assert "política automática" in texto, f"não diz QUEM escolheu: {texto}"
        assert modulo_do_pipeline.FRASE_DA_ABA_DE_LIMITES in texto, (
            f"a escolha aparece sem a ressalva que a aba de limites dá: {texto}"
        )
    # **E a frase da etapa em curso nomeia o MESMO corretor**, lido do aceite e não do seletor, que
    # repousa em `none`. Até a revisão de 25/09/2026 esta linha era `... or "«serrf»" in no_fim`,
    # sempre verdadeira porque o laço acima já afirmara a segunda metade — decorativa.
    assert "«serrf»" in em_curso, f"a etapa em curso não nomeia o corretor escolhido: {em_curso}"


def test_d511_os_quatro_desfechos_do_rotulo_do_corretor(qtbot: Any) -> None:
    """Manual, delegado, delegado com abstenção, e delegado SEM a escolha no corpo.

    **O quarto é o que não pode cair num `else` razoável**: com a caixa ticada e o corpo sem
    `delegacao`, dizer «escolhido por você» seria afirmar o oposto do que aconteceu, e usar o
    seletor seria um defeito: o seletor repousa em `none` enquanto a política corre outro.
    """
    manual = _painel(qtbot, [])
    manual.anota_o_pedido("loess", delegou=False)
    manual.aceita_disparo({"identificador": "t", "total": 7, "reclamado": False}, 1.0)
    texto_manual = manual.rotulo_do_corretor.text()

    abst = _painel(qtbot, [])
    abst.anota_o_pedido(abst.corretor.currentText(), delegou=True)
    abst.aceita_disparo(
        {
            "identificador": "t",
            "total": 7,
            "reclamado": False,
            "delegacao": {**_DELEGACAO, "correction": "none", "abstains": True},
        },
        1.0,
    )
    texto_abst = abst.rotulo_do_corretor.text()

    muda = _painel(qtbot, [])
    muda.anota_o_pedido(muda.corretor.currentText(), delegou=True)
    muda.aceita_disparo({"identificador": "t", "total": 7, "reclamado": False}, 1.0)
    texto_muda = muda.rotulo_do_corretor.text()

    assert "«loess»" in texto_manual and "você" in texto_manual, texto_manual
    assert modulo_do_pipeline.FRASE_DA_ABA_DE_LIMITES not in texto_manual, (
        f"a ressalva da política apareceu numa escolha manual: {texto_manual}"
    )
    assert "«none»" in texto_abst and "ABSTEVE-SE" in texto_abst, texto_abst
    assert "não disse qual" in texto_muda, texto_muda
    assert "você" not in texto_muda, f"a escolha delegada foi atribuída ao usuário: {texto_muda}"
    assert "«none»" not in texto_muda, f"o seletor em repouso foi lido como a escolha: {texto_muda}"
    assert len({texto_manual, texto_abst, texto_muda}) == 3


def test_d511_o_rotulo_mora_na_acao_da_analise_e_nao_no_ponto_de_escolha(qtbot: Any) -> None:
    """**A ressalva saiu do ponto de escolha, a pedido de quem usa** — e ela não volta lá.

    O rótulo é da execução, e não do grupo da correção: se fosse filho dele, a ressalva
    reapareceria ao lado do seletor, que é exactamente o que se removeu. Ele mora no bloco 2 da
    Análise, «Execução da etapa», que dispara a execução, entre as notas por baixo da linha da
    ação; o seletor do corretor está no bloco 1 da Correção, outra página.
    """
    painel = _painel(qtbot, [])
    assert not painel.grupo_da_correcao.isAncestorOf(painel.rotulo_do_corretor)
    assert painel.rotulo_do_corretor.parent() is not painel.corretor.parent()

    janela = JanelaPrincipal(_SemRede())
    qtbot.addWidget(janela)
    rotulo = janela.pipeline.rotulo_do_corretor
    assert janela.pagina_da_analise.secao_da_etapa.isAncestorOf(rotulo), (
        "o corretor da execução não está no bloco «Execução da etapa» da Análise, que a dispara"
    )
    assert rotulo in janela.pagina_da_analise.notas, "o corretor saiu das notas da linha da ação"
    assert not janela.pagina_da_correcao.isAncestorOf(rotulo), (
        "o corretor da execução foi parar na página da Correção, ao lado do seletor"
    )


def test_d511_o_ponteiro_da_ressalva_aponta_para_o_que_existe() -> None:
    """A frase manda ler uma aba e um item — os dois têm de existir, e a frase é deles.

    Um ponteiro para o que já não existe afirma mais do que o mecanismo cobre. O item foi escolhido
    porque o par 50,0 / 60,0 foi levado para lá; a frase de ressalva sai **verbatim** do corpo dele
    — não é a terceira redação.
    """
    titulos_de_aba = {aba.titulo for aba in ABAS}
    itens = {titulo: corpo for titulo, corpo in PARA_QUEM_OPERA.values()}

    assert modulo_do_pipeline.ABA_DE_LIMITES in titulos_de_aba, sorted(titulos_de_aba)
    assert modulo_do_pipeline.ITEM_DA_ESCOLHA_AUTOMATICA in itens, sorted(itens)
    assert (
        _o_que_o_ponteiro_nao_acha(
            aba=modulo_do_pipeline.ABA_DE_LIMITES,
            item=modulo_do_pipeline.ITEM_DA_ESCOLHA_AUTOMATICA,
            frase=modulo_do_pipeline.FRASE_DA_ABA_DE_LIMITES,
        )
        == []
    ), "o ponteiro da ressalva aponta para o que não existe"

    # **O par causal, em processo: a MESMA verificação acusa cada uma das três pontas plantadas.**
    # Até a revisão de 25/09/2026 a planta era `ITEM + " (antigo)" not in itens` — verdadeira por
    # construção, e sem passar pela verificação de cima.
    real = {
        "aba": modulo_do_pipeline.ABA_DE_LIMITES,
        "item": modulo_do_pipeline.ITEM_DA_ESCOLHA_AUTOMATICA,
        "frase": modulo_do_pipeline.FRASE_DA_ABA_DE_LIMITES,
    }
    for ponta in ("aba", "item", "frase"):
        plantado = {**real, ponta: real[ponta] + " (antigo)"}
        assert _o_que_o_ponteiro_nao_acha(**plantado) == [ponta], (
            f"a {ponta} plantada não foi acusada, ou outra ponta foi acusada no lugar dela"
        )


def _o_que_o_ponteiro_nao_acha(*, aba: str, item: str, frase: str) -> list[str]:
    """As pontas do ponteiro da ressalva que não existem na janela: aba, item ou frase no corpo."""
    titulos_de_aba = {a.titulo for a in ABAS}
    itens = {titulo: corpo for titulo, corpo in PARA_QUEM_OPERA.values()}
    faltam = []
    if aba not in titulos_de_aba:
        faltam.append("aba")
    if item not in itens:
        faltam.append("item")
    elif frase not in " ".join(itens[item].split()):
        faltam.append("frase")
    return faltam


# ---------------------------------------------------------------------------
# A faixa «Trabalhos em curso» e o desfecho
# ---------------------------------------------------------------------------


@pytest.fixture
def faixa(qtbot: Any) -> PainelDeTrabalhos:
    alvo = PainelDeTrabalhos()
    qtbot.addWidget(alvo)
    alvo.intervalo_da_releitura_ms = 20
    # **Mostrada**, porque a faixa só se relê à vista — ver `_rele_sozinha`. Sob `offscreen` o
    # `show()` não abre nada no ecrã e põe `isVisible()` a `True`.
    alvo.show()
    return alvo


def test_item86_a_faixa_rele_sozinha_enquanto_ha_trabalho_em_curso(
    qtbot: Any, faixa: PainelDeTrabalhos
) -> None:
    """Com um trabalho sem desfecho lido, a faixa pede a leitura sem ninguém carregar em nada."""
    pedidos: list[int] = []
    faixa.atualizar.connect(lambda: pedidos.append(1))

    faixa.mostra({"execucao": "id-a"})

    qtbot.waitUntil(lambda: len(pedidos) >= 1, timeout=2_000)


def test_item86_a_faixa_escondida_nao_pede_leitura_nem_se_rearma(
    qtbot: Any, faixa: PainelDeTrabalhos
) -> None:
    """**Faixa escondida não tem leitor** — e o caso que importa é a janela fechada e ainda não
    destruída, cujo relógio pediria leituras à piscina durante o encerramento.

    A âncora é a mesma faixa, à vista, a pedir: sem ela, *"escondida não pede"* seria
    indistinguível de *"o relógio nunca dispara"*.
    """
    pedidos: list[int] = []
    faixa.atualizar.connect(lambda: pedidos.append(1))
    faixa.mostra({"execucao": "id-a"})
    qtbot.waitUntil(lambda: len(pedidos) >= 1, timeout=2_000)

    faixa.hide()
    faixa._releitura.start(1)
    antes = len(pedidos)
    qtbot.wait(150)
    assert len(pedidos) == antes, "a faixa escondida pediu leitura"
    assert not faixa._releitura.isActive(), "a faixa escondida re-armou o relógio sozinha"


def test_d683_a_faixa_recolhida_com_a_janela_aberta_continua_a_reler(qtbot: Any) -> None:
    """**A faixa recolhida esconde o painel, e não fecha a janela.**

    Com a guarda sobre o próprio painel, a releitura parava assim que a faixa recolhia, e o selo do
    cabeçalho, que fica à vista, congelava no último estado. A direção oposta — janela escondida
    não pede — é `test_item86_a_faixa_escondida_nao_pede_leitura_nem_se_rearma`.
    """
    from PySide6.QtWidgets import QWidget

    janela = QWidget()
    qtbot.addWidget(janela)
    corpo = QWidget(janela)
    alvo = PainelDeTrabalhos(corpo)
    alvo.intervalo_da_releitura_ms = 20
    corpo.hide()
    janela.show()
    # A planta: o painel escondido com a janela aberta, que é o estado da faixa recolhida.
    assert janela.isVisible()
    assert not alvo.isVisible()
    pedidos: list[int] = []
    alvo.atualizar.connect(lambda: pedidos.append(1))

    alvo.mostra({"execucao": "id-a"})

    qtbot.waitUntil(lambda: len(pedidos) >= 1, timeout=2_000)


def test_d683_compactar_as_linhas_depois_de_desenhadas_levanta(qtbot: Any) -> None:
    """Sem linha, a forma muda; com linha já desenhada, levanta em vez de misturar as duas."""
    vazio = PainelDeTrabalhos()
    qtbot.addWidget(vazio)
    vazio.compacta_as_linhas()
    vazio.mostra({"execucao": "id-a"})
    assert vazio.linhas["execucao"].compacta

    desenhado = PainelDeTrabalhos()
    qtbot.addWidget(desenhado)
    desenhado.mostra({"execucao": "id-a"})
    with pytest.raises(RuntimeError, match="já desenhada"):
        desenhado.compacta_as_linhas()
    assert not desenhado.linhas_compactas
    assert not desenhado.linhas["execucao"].compacta


def test_item86_a_faixa_para_quando_todos_terminam(qtbot: Any, faixa: PainelDeTrabalhos) -> None:
    """**A metade que custa é PARAR**, e ela é o que separa isto de polling agressivo.

    Um desfecho terminal — `200`, erro, `404`, `410` — encerra a releitura daquela chave; sem chave
    em curso, o relógio não volta a armar. A âncora é o relógio ARMADO antes do desfecho: sem ela,
    *"parou"* seria indistinguível de *"nunca armou"*.
    """
    faixa.mostra({"execucao": "id-a", "corrigir": "id-b"})
    assert faixa._releitura.isActive(), "âncora: a releitura não armou com dois trabalhos"

    faixa.mostra_estado("execucao", 200, {})
    faixa.mostra({"execucao": "id-a", "corrigir": "id-b"})
    assert faixa._releitura.isActive(), "um dos dois ainda corre, e a releitura parou"

    faixa.mostra_estado("corrigir", 410, {"detail": "cancelado"})
    faixa.mostra({"execucao": "id-a", "corrigir": "id-b"})
    assert not faixa._releitura.isActive(), "os dois terminaram, e a faixa continua a reler"

    pedidos: list[int] = []
    faixa.atualizar.connect(lambda: pedidos.append(1))
    qtbot.wait(150)
    assert pedidos == [], f"a faixa pediu {len(pedidos)} leituras depois de tudo terminar"


def test_item86_o_202_com_estado_terminal_tambem_para(faixa: PainelDeTrabalhos) -> None:
    """A rota não serve terminal com `202` hoje; se servir, `TERMINAIS` decide e não um literal."""
    assert TERMINAIS, "âncora: `TERMINAIS` está vazio"
    for estado in sorted(TERMINAIS):
        faixa.mostra_estado("execucao", EM_CURSO, {"estado": estado, "feito": 1, "total": 1})
        faixa.mostra({"execucao": "id-a"})
        assert not faixa._releitura.isActive(), f"`{estado}` com 202 manteve a releitura"
    faixa.mostra_estado("execucao", EM_CURSO, {"estado": "executando", "feito": 0, "total": 1})
    faixa.mostra({"execucao": "id-a"})
    assert faixa._releitura.isActive(), "um 202 em curso não rearmou — a direção oposta é vácua"


def test_item86_um_trabalho_novo_na_mesma_chave_nao_herda_o_desfecho_do_antigo(
    faixa: PainelDeTrabalhos,
) -> None:
    """**A irmã da releitura**: a faixa guardava o estado por CHAVE e não por trabalho.

    Redisparar a execução troca o identificador e mantinha o «concluído» da corrida anterior ao
    lado do identificador novo — o mesmo desacordo entre regiões, na direção oposta. E sem o
    conserto a releitura nem armaria, porque a chave já constava como terminada.
    """
    concluido = ROTULOS_DOS_ESTADOS["concluido"]
    faixa.mostra({"execucao": "id-a"})
    faixa.mostra_estado("execucao", 200, {})
    assert concluido in faixa.mostra({"execucao": "id-a"}), "âncora: o desfecho não foi anotado"

    texto = faixa.mostra({"execucao": "id-b"})

    assert concluido not in texto, f"o trabalho novo herdou o desfecho do antigo: {texto}"
    assert NAO_LIDO in texto, texto
    assert faixa._releitura.isActive(), "o trabalho novo não é relido"


def test_item86_a_faixa_diz_que_se_rele_sozinha(faixa: PainelDeTrabalhos) -> None:
    """O «Atualizar» continua a existir; a frase diz que já não é preciso esperar por ele."""
    texto = faixa.mostra({"execucao": "id-a"})
    assert "relida sozinha" in texto, texto
    faixa.mostra_estado("execucao", 200, {})
    assert "relida sozinha" not in faixa.mostra({"execucao": "id-a"}), (
        "a faixa promete releitura depois de ter parado de reler"
    )


# ---------------------------------------------------------------------------
# A faixa legível: nome, etapa, barra, estado e o botão de parar de cada trabalho
# ---------------------------------------------------------------------------
#
# O progresso fica evidente na faixa, com o estado e a interrupção. Antes,
# a faixa listava cada trabalho como chave interna, identificador e estado cru, numa linha de texto,
# e a barra não existia nela.


class _SemRede:
    """O transporte da sonda: nenhuma chamada sai daqui, e a que sair falha como rede recusada."""

    def post(self, url: str, **kwargs: Any) -> Any:
        raise ConnectionRefusedError("sonda")

    def get(self, url: str, **kwargs: Any) -> Any:
        raise ConnectionRefusedError("sonda")


def _sitios_que_registram() -> list[tuple[int, int]]:
    """`(primeira, última linha)` de cada chamada a `registra_trabalho` na janela — por AST.

    **Lê o arquivo do módulo que de facto foi importado**, e não um caminho montado à mão: a
    sonda corre o código desse módulo, e as linhas têm de ser as dele.
    """
    arvore = ast.parse(Path(modulo_da_janela.__file__).read_text(encoding="utf-8"))
    return sorted(
        (no.lineno, no.end_lineno or no.lineno)
        for no in ast.walk(arvore)
        if isinstance(no, ast.Call)
        and isinstance(no.func, ast.Attribute)
        and no.func.attr == "registra_trabalho"
    )


def _chaves_em(valor: object) -> set[str]:
    """As chaves que um valor de módulo carrega: o texto, as chaves de um dicionário, os itens de
    uma coleção, o primeiro elemento de um par e o `.chave` de um passo."""
    if isinstance(valor, str):
        return {valor}
    if isinstance(valor, Mapping):
        return {k for k in valor if isinstance(k, str)}
    if not isinstance(valor, tuple | list | set | frozenset):
        return set()
    chaves: set[str] = set()
    for item in valor:
        if isinstance(item, str):
            chaves.add(item)
        elif isinstance(item, tuple) and item and isinstance(item[0], str):
            chaves.add(item[0])
        elif isinstance(getattr(item, "chave", None), str):
            chaves.add(item.chave)
    return chaves


def _universo_de_chaves() -> frozenset[str]:
    """As chaves que `_encaminha` sabe distinguir — derivadas dele e não escritas aqui.

    Os valores dos nomes de módulo que o corpo dele cita (`CHAVE_DA_MEDICAO`, `ETAPA_DA_CHAVE`,
    `ARTEFATOS`…) e as strings que ele compara. **Superconjunto de propósito:** a sonda alimenta
    cada uma com um `202` de aceite, e quem decide o que registra é a janela. Se uma chamada nova a
    `registra_trabalho` dependesse de uma chave que isto não alcança, o teste o diz — a cobertura
    dos sítios é conferida à parte.
    """
    arvore = ast.parse(Path(modulo_da_janela.__file__).read_text(encoding="utf-8"))
    funcao = next(
        no for no in ast.walk(arvore) if isinstance(no, ast.FunctionDef) and no.name == "_encaminha"
    )
    chaves: set[str] = set()
    for no in ast.walk(funcao):
        if isinstance(no, ast.Name) and hasattr(modulo_da_janela, no.id):
            chaves |= _chaves_em(getattr(modulo_da_janela, no.id))
        elif isinstance(no, ast.Compare):
            for lado in (no.left, *no.comparators):
                if isinstance(lado, ast.Constant) and isinstance(lado.value, str):
                    chaves.add(lado.value)
    return frozenset(chaves)


def _sonda_o_registro(
    janela: JanelaPrincipal, universo: frozenset[str], monkeypatch: Any
) -> tuple[set[str], set[int]]:
    """Alimenta `_encaminha` com um `202` de aceite por chave e devolve `(registradas, linhas)`.

    `linhas` são as linhas de `principal.py` de onde `registra_trabalho` foi chamado: é o que diz
    quais sítios a sonda de facto alcançou.
    """
    registradas: set[str] = set()
    linhas: set[int] = set()
    original = JanelaPrincipal.registra_trabalho

    def espiao(self: JanelaPrincipal, chave: str, identificador: str) -> None:
        registradas.add(chave)
        linhas.add(sys._getframe(1).f_lineno)
        original(self, chave, identificador)

    monkeypatch.setattr(JanelaPrincipal, "registra_trabalho", espiao)
    for chave in sorted(universo):
        janela._encaminha(
            chave,
            Resposta(
                codigo=EM_CURSO,
                ok=True,
                causa=None,
                mensagem="",
                corpo={"identificador": f"sonda-{chave}", "total": 2, "reclamado": False},
                conteudo=None,
            ),
        )
    monkeypatch.undo()
    return registradas, linhas


def sitios_nao_alcancados(sitios: list[tuple[int, int]], linhas: set[int]) -> list[int]:
    """Os sítios de registro que nenhuma linha chamadora cobre. Função PURA."""
    return [inicio for inicio, fim in sitios if not any(inicio <= x <= fim for x in linhas)]


def faltam_nomes(registradas: set[str], nomes: Mapping[str, str]) -> tuple[list[str], list[str]]:
    """`(chaves sem nome, nomes sem chave)` — as duas direções da bijeção. Função PURA."""
    return sorted(registradas - set(nomes)), sorted(set(nomes) - registradas)


def test_toda_chave_que_a_janela_registra_tem_nome_legivel(qtbot: Any, monkeypatch: Any) -> None:
    """**A bijeção é contra o que a janela REGISTRA**, medido por sonda, e não contra uma lista.

    A sonda alimenta `_encaminha` com um `202` de aceite para cada chave que ele conhece, e espia
    `registra_trabalho`. Três coisas reprovam: um sítio de registro que a sonda não alcançou (o
    universo ficou curto, e ela mediria uma fração); uma chave registrada sem nome legível; e um
    nome cuja chave a janela já não registra.

    **Os três pares causais estão no mesmo teste**, e cada um cai sobre dado em que as duas coisas
    diferem: a tabela sem `medicao` tem de acusar `medicao`; a tabela com uma chave inventada tem de
    acusá-la; e a sonda sem `execucao` no universo tem de deixar exactamente um sítio por alcançar.
    """
    janela = JanelaPrincipal(_SemRede())
    qtbot.addWidget(janela)
    janela.trabalhos.intervalo_da_releitura_ms = 3_600_000
    sitios = _sitios_que_registram()
    universo = _universo_de_chaves()
    assert len(sitios) >= 5, (
        f"âncora: só {len(sitios)} sítios de registro achados — o AST mediu nada"
    )

    registradas, linhas = _sonda_o_registro(janela, universo, monkeypatch)
    janela.espera_a_piscina(10_000)

    assert sitios_nao_alcancados(sitios, linhas) == [], (
        f"a sonda não alcançou os sítios de registro {sitios_nao_alcancados(sitios, linhas)} de "
        "`principal.py`: a chave que os alimenta não está no universo derivado de `_encaminha`"
    )
    assert registradas, "âncora: a sonda não registrou trabalho nenhum"
    sem_nome, sem_chave = faltam_nomes(registradas, NOMES_DOS_TRABALHOS)
    assert sem_nome == [], f"chaves que a janela registra sem nome legível: {sem_nome}"
    assert sem_chave == [], f"nomes legíveis de chaves que a janela já não registra: {sem_chave}"

    # Os pares causais, em processo.
    sem_medicao = {k: v for k, v in NOMES_DOS_TRABALHOS.items() if k != "medicao"}
    assert faltam_nomes(registradas, sem_medicao) == (["medicao"], [])
    com_fantasma = {**NOMES_DOS_TRABALHOS, "fantasma": "Fantasma"}
    assert faltam_nomes(registradas, com_fantasma) == ([], ["fantasma"])
    _, linhas_sem_execucao = _sonda_o_registro(janela, universo - {"execucao"}, monkeypatch)
    janela.espera_a_piscina(10_000)
    assert len(sitios_nao_alcancados(sitios, linhas_sem_execucao)) == 1, (
        "a cobertura dos sítios não acusou o universo sem `execucao`"
    )
    # E a chave que escapasse aos dois aparece MARCADA na faixa, e não em silêncio.
    assert "nova" in nome_do_trabalho("nova") and nome_do_trabalho("nova") not in set(
        NOMES_DOS_TRABALHOS.values()
    )


def test_todo_estado_que_o_registro_produz_tem_rotulo_legivel(faixa: PainelDeTrabalhos) -> None:
    """Bijeção com `ESTADOS`, e o estado que chega fora do vocabulário vem MARCADO.

    Um `estado` desconhecido num `202` não pode cair num `else` razoável que o leia como outro: ele
    aparece com o valor cru e a marca, e continua a ser relido, porque nada diz que terminou.
    """
    assert set(ROTULOS_DOS_ESTADOS) == set(ESTADOS) == set(TONS_DOS_ESTADOS)
    assert len(set(ROTULOS_DOS_ESTADOS.values())) == len(ESTADOS), "dois estados com o mesmo rótulo"

    faixa.mostra({"corrigir": "id-a"})
    rotulo = faixa.mostra_estado("corrigir", EM_CURSO, _progresso("zumbi", 0, 2))
    assert "zumbi" in rotulo and "desconhecido" in rotulo.lower(), rotulo
    assert not rotulo.startswith(tuple(ROTULOS_DOS_ESTADOS.values())), (
        f"o estado desconhecido foi lido como um do vocabulário: {rotulo}"
    )
    faixa.mostra({"corrigir": "id-a"})
    assert faixa._releitura.isActive(), "o estado desconhecido foi lido como fim"


def test_um_202_com_etapa_mostra_nome_etapa_contagem_e_a_barra_a_meio(
    faixa: PainelDeTrabalhos,
) -> None:
    """`feito=2`, `total=4` e a etapa nomeada: o nome do trabalho, o da etapa, «2 de 4» e 50%.

    **A segunda leitura move tudo**, e é o par causal: uma linha que mostrasse a etapa da primeira
    leitura, ou uma barra presa, reprovaria nela.
    """
    faixa.mostra({"execucao": "id-a"})
    corpo = {**_progresso("executando", 2, 4), "etapa_em_curso": "relatorio de qualidade"}
    rotulo = faixa.mostra_estado("execucao", EM_CURSO, corpo)
    texto = faixa.mostra({"execucao": "id-a"})
    linha = faixa.linhas["execucao"]

    assert linha.nome.text() == NOMES_DOS_TRABALHOS["execucao"]
    assert "relatorio de qualidade" in linha.nota.text()
    assert "2 de 4" in linha.barra.text()
    assert (linha.barra.minimum(), linha.barra.maximum(), linha.barra.value()) == (0, 4, 2)
    assert linha.estado.text() == ROTULOS_DOS_ESTADOS["executando"]
    assert "2 de 4" in rotulo and "relatorio de qualidade" in rotulo
    assert NOMES_DOS_TRABALHOS["execucao"] in texto and "relatorio de qualidade" in texto
    assert "id-a" not in texto and "id-a" in linha.nome.toolTip(), (
        "o identificador saiu da linha e vive na dica do nome"
    )

    corpo = {**_progresso("executando", 3, 4), "etapa_em_curso": "gravacao no banco"}
    faixa.mostra_estado("execucao", EM_CURSO, corpo)
    faixa.mostra({"execucao": "id-a"})
    assert "gravacao no banco" in linha.nota.text() and "relatorio" not in linha.nota.text()
    assert (linha.barra.maximum(), linha.barra.value()) == (4, 3)


def test_um_202_sem_total_mostra_a_barra_indeterminada(faixa: PainelDeTrabalhos) -> None:
    """Sem `total` há trabalho e não se sabe quanto falta: o indeterminado é o certo.

    **E antes da primeira leitura a barra fica VAZIA, e não indeterminada** — a lição do painel do
    pipeline: sinal de vida falso é pior que percentagem falsa. A âncora é a mesma linha a sair do
    vazio para o indeterminado com o `202`.
    """
    faixa.mostra({"processar": "id-a"})
    barra = faixa.linhas["processar"].barra
    assert (barra.minimum(), barra.maximum(), barra.value()) == (0, 1, 0), "repouso não é vazio"

    faixa.mostra_estado("processar", EM_CURSO, {"identificador": "id-a", "estado": "executando"})
    faixa.mostra({"processar": "id-a"})
    assert (barra.minimum(), barra.maximum()) == (0, 0), (
        "sem `total` a barra não ficou indeterminada"
    )


def test_um_trabalho_terminado_mostra_o_estado_final_legivel(faixa: PainelDeTrabalhos) -> None:
    """Os três fins que a leitura pode trazer, cada um com o seu rótulo, e nenhum a animar.

    O botão de parar de um trabalho que terminou fica **desligado**: carregar nele só produziria um
    `409`. A âncora é o mesmo botão LIGADO enquanto o trabalho corre.
    """
    trabalhos = {"corrigir": "id-c", "exportar": "id-e", "processar": "id-p"}
    faixa.mostra(trabalhos)
    for chave in trabalhos:
        faixa.mostra_estado(
            chave, EM_CURSO, {**_progresso("executando", 1, 2), "identificador": chave}
        )
    faixa.mostra(trabalhos)
    assert all(faixa.linhas[c].botao_de_parar.isEnabled() for c in trabalhos), "âncora"

    faixa.mostra_estado("corrigir", 200, {})
    faixa.mostra_estado("exportar", 410, {"detail": "o trabalho foi CANCELADO a seu pedido"})
    faixa.mostra_estado("processar", 500, {"detail": "o trabalho falhou na etapa «correcao»"})
    faixa.mostra(trabalhos)

    estados = {c: faixa.linhas[c].estado.text() for c in trabalhos}
    assert estados == {
        "corrigir": ROTULOS_DOS_ESTADOS["concluido"],
        "exportar": ROTULOS_DOS_ESTADOS["cancelado"],
        "processar": ROTULOS_DOS_ESTADOS["falhou"],
    }, estados
    assert "«correcao»" in faixa.linhas["processar"].nota.text()
    barra = faixa.linhas["corrigir"].barra
    assert (barra.maximum(), barra.value()) == (2, 2), "o concluído não encheu a barra"
    for chave in trabalhos:
        assert faixa.linhas[chave].barra.maximum() > 0, f"{chave}: a barra terminada anima"
        assert not faixa.linhas[chave].botao_de_parar.isEnabled(), f"{chave}: parar ficou ligado"
    assert not faixa._releitura.isActive()


def test_o_botao_de_parar_de_uma_linha_manda_a_chave_dela(faixa: PainelDeTrabalhos) -> None:
    """Cada botão emite a chave da SUA linha — o par é o clique na outra linha.

    É a forma do defeito de fecho tardio: um `lambda` sobre a variável do laço faria os botões todos
    emitirem a última chave, e só dois cliques em linhas diferentes o separam. Depois, o `409` e o
    `404` do pedido de parada continuam tratados, e cada um é um fim.
    """
    faixa.mostra({"corrigir": "id-c", "exportar": "id-e"})
    pedidos: list[str] = []
    faixa.cancelar.connect(pedidos.append)

    faixa.linhas["exportar"].botao_de_parar.click()
    faixa.linhas["corrigir"].botao_de_parar.click()
    assert pedidos == ["exportar", "corrigir"], pedidos

    faixa.mostra_cancelamento("exportar", 409, {})
    faixa.mostra_cancelamento("corrigir", 404, {"detail": "trabalho desconhecido"})
    faixa.mostra({"corrigir": "id-c", "exportar": "id-e"})
    assert "terminado" in faixa.linhas["exportar"].estado.text()
    assert "desconhecido" in faixa.linhas["corrigir"].estado.text().lower()
    assert "podado" in faixa.linhas["corrigir"].nota.text()
    assert not faixa.linhas["exportar"].botao_de_parar.isEnabled()
    assert not faixa.linhas["corrigir"].botao_de_parar.isEnabled()


def test_o_pedido_de_parada_aceite_continua_a_dizer_a_parar_na_releitura(
    faixa: PainelDeTrabalhos,
) -> None:
    """Entre o clique e a parada a leitura traz `cancelamento_pedido` e `executando`.

    Sem ler o campo, a releitura seguinte escrevia «Executando» por cima de «A parar…», e o botão
    parecia não ter feito nada — o relato que o campo existe para evitar. O par é a mesma leitura
    sem o pedido.
    """
    faixa.mostra({"corrigir": "id-c"})
    faixa.mostra_cancelamento("corrigir", EM_CURSO, {})
    corpo = {**_progresso("executando", 0, 2), "cancelamento_pedido": True}
    faixa.mostra_estado("corrigir", EM_CURSO, corpo)
    faixa.mostra({"corrigir": "id-c"})
    linha = faixa.linhas["corrigir"]
    assert linha.estado.text() == PARANDO
    assert linha.botao_de_parar.isEnabled(), "pedir outra vez é idempotente e fica possível"

    faixa.mostra_estado("corrigir", EM_CURSO, _progresso("executando", 0, 2))
    faixa.mostra({"corrigir": "id-c"})
    assert linha.estado.text() == ROTULOS_DOS_ESTADOS["executando"]


def test_a_recusa_de_parar_sem_trabalho_aparece_na_faixa(faixa: PainelDeTrabalhos) -> None:
    """**A frase existia e caía num rótulo ESCONDIDO** — o botão recusava em silêncio.

    Desde que a lista passou a esconder-se quando está vazia, `pede_cancelamento` escrevia
    `SEM_TRABALHO_A_PARAR` num rótulo invisível: o texto estava lá, e o teste que só lia o texto
    passava. O que se afirma agora é que a pessoa o vê.
    """
    faixa.mostra({})
    faixa.pede_cancelamento()

    assert faixa.rotulo_da_lista.text() == SEM_TRABALHO_A_PARAR
    assert faixa.rotulo_da_lista.isVisibleTo(faixa), "a recusa foi escrita num rótulo escondido"


def test_as_colunas_alinham_e_o_seletor_mostra_o_nome_inteiro(
    qtbot: Any, faixa: PainelDeTrabalhos
) -> None:
    """**Os dois defeitos que a captura da faixa montada mostrou**, e nenhum teste lia.

    Com a largura de cada rótulo a decidir as colunas, a barra mudava de sítio em toda linha sem
    nota, ou com um nome mais curto; e o seletor, preso nos 150 px do mínimo, cortava «Execução do
    pipelin». As linhas daqui diferem no comprimento do nome e na presença da nota, que é onde as
    duas coisas que se separam diferem.
    """
    from PySide6.QtCore import QPoint

    faixa.resize(900, 400)
    trabalhos = {"execucao": "id-e", "exportar": "id-x", "etapa_preprocessamento": "id-p"}
    faixa.mostra(trabalhos)
    corpo = {**_progresso("executando", 1, 6), "etapa_em_curso": "pipeline"}
    faixa.mostra_estado("execucao", EM_CURSO, corpo)
    faixa.mostra_estado("exportar", 200, {})
    faixa.mostra(trabalhos)
    qtbot.wait(50)

    colunas = {c: faixa.linhas[c].barra.mapTo(faixa, QPoint(0, 0)).x() for c in trabalhos}
    assert len(set(colunas.values())) == 1, f"a barra muda de coluna entre linhas: {colunas}"
    seletor = faixa.seletor_de_trabalho
    mais_longo = max(
        seletor.fontMetrics().horizontalAdvance(seletor.itemText(i)) for i in range(seletor.count())
    )
    assert seletor.width() > mais_longo, (
        f"o seletor tem {seletor.width()} px e o nome mais longo pede {mais_longo}"
    )


# ---------------------------------------------------------------------------
# O aviso de fecho de `trabalhos.py` — para quem usa o executável
# ---------------------------------------------------------------------------


def test_o_aviso_do_fecho_fala_do_executavel_e_nao_da_tela_web() -> None:
    """Quem lê esta frase é quem abriu o executável.

    As duas metades que ficam são as de `test_u21_o_painel_declara_que_fechar_encerra_os_trabalhos`
    — o fato e a causa. O que sai é a comparação com uma interface que essa pessoa nunca viu.
    """
    assert "FECHAR O APLICATIVO ENCERRA OS TRABALHOS" in AVISO_DO_FECHO
    assert "dentro deste processo" in AVISO_DO_FECHO
    for termo in ("web", "navegador", "refresh", "API"):
        assert termo not in AVISO_DO_FECHO, f"«{termo}» continua no aviso: {AVISO_DO_FECHO}"
