"""O ciclo completo do multiverso na janela.

## O que esta suíte prova, e por que ela é separada de `test_desktop_janela.py`

Aquele arquivo tem 5.200 linhas e cobre a janela inteira; este cobre **uma unidade**, e a separação
é a mesma que `test_cancelamento.py` fez para o núcleo. O que se afirma aqui:

| Camada | Desfechos exercitados |
|---|---|
| estimativa de custo | grade vazia; um corretor; a grade inteira; corretor sem medição |
| escolhas do disparo | par completo; par pela metade; nenhum |
| estado dos botões | repouso; a correr; terminada; cancelada |
| respostas | `202` e erro do disparo; progresso nos cinco estados; `202` e `409` do cancelar |
| ligação à janela | as três rotas novas chegam ao transporte, por EMISSÃO e não por introspecção |

## A planta tem de cair sobre dado em que as duas coisas diferem

O par que importa é `test_a_estimativa_muda_quando_o_serrf_sai`: uma estimativa que ignorasse os
corretores devolveria o mesmo texto nos dois braços, e todos os outros testes de estimativa
continuariam verdes. A asserção é que os dois números **divergem**, e por um fator que a medição
prevê.

## Escopo negativo

Nada aqui mede latência de cancelamento — isso é `tests/test_cancelamento.py`. E nada aqui desenha
a janela: os testes de layout são os de captura, e esta suíte é de comportamento.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Final

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QLabel

from refino.desktop.janela.multiverso import (
    CELULAS_POR_CORRETOR,
    COLUNAS,
    COMO_LER_A_CONCLUSAO,
    CORRETORES_DA_GRADE,
    EXEMPLO_DO_MULTIVERSO,
    EXPLICACAO_DAS_COLUNAS,
    EXPLICACAO_DOS_VEREDITOS,
    LIMIARES_DO_VEREDITO,
    PADRAO_DO_LIMIAR,
    SEM_CORRETOR,
    SEM_EXECUCAO,
    SEM_IDENTIFICADOR_DO_MULTIVERSO,
    VEREDITOS,
    PainelDeMultiverso,
    campos_dos_limiares,
    estimativa_de_custo,
    rota_do_cancelamento,
    rota_do_progresso_do_multiverso,
    texto_do_progresso,
)
from refino.desktop.recorte import FUNCIONALIDADE_EM_BETA

pytestmark = pytest.mark.usefixtures("qtbot")


@pytest.fixture
def painel(qtbot: Any) -> PainelDeMultiverso:
    alvo = PainelDeMultiverso()
    qtbot.addWidget(alvo)
    return alvo


# ---------------------------------------------------------------------------
# A ESTIMATIVA DE CUSTO — é ela que torna o clique informado em vez de acidental
# ---------------------------------------------------------------------------


def test_a_grade_de_referencia_e_derivada_e_nao_escrita() -> None:
    """A âncora: os corretores e as células por corretor vêm de `GridDefinition`.

    Sem esta, um literal `1440` aqui passaria despercebido e divergiria da grade no dia em que um
    eixo mudasse — como quando o repositório afirmava `1.296` contra `4.320`.
    """
    from refino.multiverse.grid import GridDefinition

    grade = GridDefinition()
    assert tuple(grade.corrections) == CORRETORES_DA_GRADE
    assert CELULAS_POR_CORRETOR * len(CORRETORES_DA_GRADE) == grade.size()


def test_sem_corretor_a_estimativa_recusa_em_vez_de_dizer_zero() -> None:
    """Zero células e «não escolheu» são fatos diferentes.

    Um texto dizendo *"0 células, ~0 s"* leria-se como *"esta grade é barata"*, e o botão
    dispararia contra uma `GridDefinition` que levanta com eixo vazio.
    """
    celulas, segundos, texto = estimativa_de_custo(())

    assert (celulas, segundos) == (0, 0.0)
    assert texto == SEM_CORRETOR


def test_a_estimativa_declara_a_populacao_em_que_foi_medida() -> None:
    """O escopo da medida, declarado na tela.

    **O contrato MUDOU em 31/08/2026, e ficou mais forte.** Era uma projeção dos tempos do
    `pos_hilic` aplicada a qualquer matriz, com o texto a declarar a limitação em vez de a
    corrigir. Os quatro modos passaram a estar cronometrados, e a estimativa virou **faixa
    medida** — a asserção passa a exigir os dois extremos nomeados, e não um único modo.

    **Por que faixa e não o valor do modo carregado:** o painel não sabe qual matriz está aberta,
    e passá-la exigiria encanamento novo pela janela. Uma faixa medida sobre os quatro arquivos
    reais informa mais que um ponto medido sobre um só e aplicado aos outros três — que era o
    estado anterior, e errava por 2,6x no menor deles.
    """
    _celulas, _segundos, texto = estimativa_de_custo(("none",))

    assert "neg_hilic" in texto and "2.039 features" in texto, "o extremo BAIXO tem de ir nomeado"
    assert "pos_hilic" in texto and "6.035" in texto, "o extremo ALTO tem de ir nomeado"
    assert " a " in texto, "é faixa, e uma faixa sem os dois lados é um ponto disfarçado"
    # O oráculo é o RADICAL e não a palavra. Ele era
    # `"medido" in texto`, e o texto passou a dizer *"derivado da medição dos quatro arquivos do
    # laboratório em 31/08/2026"* — a procedência que este portão protege está lá, e mais
    # explícita: o número **é** derivado de uma medição, e não medido ele próprio. Afrouxar seria
    # tirar a asserção; o que se faz aqui é perguntar o que ela afirmava e afirmá-lo com um
    # oráculo que a forma honesta satisfaz.
    assert "medi" in texto, (
        "o texto perdeu a procedência — nem «medido» nem «medição». Sem ela o número é projeção "
        f"sem origem, que é o que este portão existe para impedir: {texto}"
    )
    assert "cancelada" in texto, "não diz que dá para parar, que é o que torna o clique aceitável"


def test_a_faixa_da_estimativa_cresce_com_o_modo() -> None:
    """O extremo baixo é MENOR que o alto — sem isto, uma faixa degenerada passaria por faixa.

    É a asserção de não-vacuidade aplicada a um intervalo: `min` e `max` sobre uma tabela
    de um elemento só devolvem o mesmo número, e o texto continuaria a dizer «de X a X» sem que
    nada acusasse.
    """
    from refino.interface.fluxo import CUSTO_MEDIDO_POR_MODO

    assert len(CUSTO_MEDIDO_POR_MODO) == 4, "âncora: os quatro modos têm de estar na tabela"
    serrfs = [c["serrf"] for c in CUSTO_MEDIDO_POR_MODO.values()]
    assert min(serrfs) < max(serrfs), "a tabela colapsou num valor só"
    assert CUSTO_MEDIDO_POR_MODO["pos_hilic"]["serrf"] == max(serrfs), (
        "o `pos_hilic` é o maior dos quatro, e é ele que o texto nomeia como extremo alto"
    )


def test_a_estimativa_muda_quando_o_serrf_sai() -> None:
    """**O par causal, e sem ele os outros testes de estimativa são vácuos.**

    Uma implementação que ignorasse `corretores` e devolvesse um texto fixo passaria em todos eles.
    A asserção é que os dois braços **divergem**, e por uma razão que a medição fixa.

    **O limiar caiu de 10x para 3x em 31/08/2026, e a razão é MEDIDA e não conveniência.** Duas
    coisas mudaram por baixo dele:

    1. O `~97% do custo` vale sobre a CORREÇÃO e não sobre a GRADE. Cronometrada, a raiz é
       57% a 64% do total e as folhas são o resto — o `99,96%` do executor compara uma raiz contra
       UMA folha, não contra as 4.320.
    2. A estimativa passou a descrever o regime PARALELO, que é o que a rota executa. E o
       paralelismo comprime a diferença, porque ajuda mais a raiz (5,60x) do que as folhas (4,05x)
       — exactamente a parte que o `serrf` domina.

    Medido: 386,8 s contra 119,9 s, razão **3,2x**. O limiar fica em 3, e não em 3,1: um limiar
    colado no valor medido reprova por ruído de medição e ensina a gente a afrouxá-lo outra vez.

    ## O LIMIAR INVERTEU, e ele estava a medir o regime errado

    O parágrafo acima descreve o regime dos **dois** níveis de paralelismo. A rota corre com um, e
    a estimativa passou a descrever o que a rota corre. **Derivado agora, a razão sã é 1,94x — e
    toda degeneração dá MAIOR, não menor:**

    | Cenário | razão |
    |---|---|
    | **são**, o regime que a rota corre | **1,94x** |
    | se voltasse o regime dos dois níveis | 3,23x |
    | se o ganho do corretor deixasse de ser aplicado | 3,84x |
    | se as folhas não entrassem na conta | 33,66x |

    **O `> 3` teria passado em três das quatro linhas e reprovado na única correta.** Ele não é
    afrouxado: vira uma **banda**, com o limiar de cada lado derivado da tabela — 1,5 abaixo (23%
    de folga sobre 1,94) e 2,5 acima (29% de folga até 3,23, que é a degeneração mais próxima).

    **O que a banda protege continua a ser o mesmo:** que a conta use os custos medidos, e que o
    paralelismo alcance o braço certo. O que mudou é que a violação hoje se manifesta por cima.
    """
    _c1, com_serrf, _t1 = estimativa_de_custo(("none", "loess", "serrf"))
    _c2, sem_serrf, _t2 = estimativa_de_custo(("none", "loess"))

    assert com_serrf > sem_serrf, "a estimativa não depende dos corretores escolhidos"
    razao = com_serrf / sem_serrf
    assert 1.5 < razao < 2.5, (
        f"tirar o `serrf` mudou a estimativa de {com_serrf:.0f} s para {sem_serrf:.0f} s, "
        f"razão {razao:.2f}x. Derivado em 04/09/2026 no regime que a rota corre — a correção em "
        "paralelo e a grade em sequência —, a razão sã é 1,94x. Acima de 2,5: ou voltou o regime "
        "dos dois níveis (3,23x), ou o ganho do corretor deixou de ser aplicado (3,84x), ou as "
        "folhas saíram da conta (33,66x). Abaixo de 1,5: os custos medidos deixaram de ser usados"
    )


def test_corretor_sem_custo_medido_e_nomeado_e_nao_somado_como_zero() -> None:
    """Um valor novo em `Correction` que ainda não tenha medição.

    Não pode impedir o disparo e não pode sumir da conta em silêncio: somar zero produziria uma
    estimativa otimista sem nada a acusar, indistinguível de sucesso.
    """
    _celulas, _segundos, texto = estimativa_de_custo(("none", "corretor_inventado"))

    assert "corretor_inventado" in texto
    assert "menor que o real" in texto


# ---------------------------------------------------------------------------
# AS ESCOLHAS que viajam no disparo
# ---------------------------------------------------------------------------


def test_todos_os_corretores_nascem_marcados_e_o_padrao_e_a_grade_inteira(
    painel: PainelDeMultiverso,
) -> None:
    """O padrão é a grade que foi avaliada.

    Um padrão menor faria a janela executar por omissão algo diferente do que se mediu, com
    as duas coisas a chamarem-se *"o multiverso"*.
    """
    assert painel.corretores_escolhidos() == CORRETORES_DA_GRADE


def test_o_par_de_grupos_so_viaja_inteiro(painel: PainelDeMultiverso) -> None:
    """**Meio par é recusa na rota, não omissão** — e mandá-lo daqui produziria um `422` no fim.

    A rota diz: *"ignorar um `group_a` sozinho executaria uma grade de horas e devolveria progresso
    a quem pediu conclusão"*. O painel completa ou limpa; nunca manda metade.
    """
    painel.grupo_a.setCurrentText("ATLL")
    painel.grupo_b.setCurrentText("")
    so_um = painel.escolhas_do_disparo()

    painel.grupo_b.setCurrentText("NoHTLV")
    os_dois = painel.escolhas_do_disparo()

    assert "group_a" not in so_um and "group_b" not in so_um
    assert os_dois["group_a"] == "ATLL"
    assert os_dois["group_b"] == "NoHTLV"


def test_corrections_viaja_como_json_porque_a_rota_o_le_com_json_loads(
    painel: PainelDeMultiverso,
) -> None:
    """O formato é o de `_vetor_ou_422`. Mandar uma lista crua daria um `422` de sintaxe."""
    painel.corretores["serrf"].setChecked(False)

    campos = painel.escolhas_do_disparo()

    assert json.loads(campos["corrections"]) == [
        nome for nome in CORRETORES_DA_GRADE if nome != "serrf"
    ]


def test_os_grupos_do_diagnostico_enchem_os_combos_sem_apagar_o_que_foi_escrito(
    painel: PainelDeMultiverso,
) -> None:
    """**Preservar a escolha é requisito e não cortesia.**

    Correr o diagnóstico outra vez depois de escolher o par apagaria a escolha, e o profissional
    teria de a refazer sem entender por quê.
    """
    painel.define_grupos(["ATLL", "Asymptomatic", "HAMTSP", "NoHTLV"])
    escolhido_a = painel.grupo_a.currentText()
    painel.grupo_b.setCurrentText("HAMTSP")

    painel.define_grupos(["ATLL", "Asymptomatic", "HAMTSP", "NoHTLV"])

    assert painel.grupo_a.currentText() == escolhido_a
    assert painel.grupo_b.currentText() == "HAMTSP", "a segunda leitura apagou a escolha"


def test_sem_diagnostico_os_combos_aceitam_texto_livre(painel: PainelDeMultiverso) -> None:
    """Quem não correu o diagnóstico escreve. Um combo fechado e vazio recusaria sem dizer."""
    painel.grupo_a.setCurrentText("Grupo_A_escrito_a_mao")
    painel.grupo_b.setCurrentText("Grupo_B_escrito_a_mao")

    campos = painel.escolhas_do_disparo()

    assert campos["group_a"] == "Grupo_A_escrito_a_mao"


# ---------------------------------------------------------------------------
# OS BOTÕES — quatro estados, e cada um habilita um conjunto diferente
# ---------------------------------------------------------------------------


def test_em_repouso_so_o_disparo_esta_ligado(painel: PainelDeMultiverso) -> None:
    """**O ramo que dispara por AUSÊNCIA de condição.** Não há o que plantar nele."""
    assert painel.botao_de_disparo.isEnabled()
    assert not painel.botao_de_progresso.isEnabled()
    assert not painel.botao_de_cancelamento.isEnabled()
    assert painel.rotulo_de_progresso.text() == SEM_EXECUCAO
    assert painel.identificador() is None


def test_a_correr_o_disparo_desliga_e_o_cancelar_liga(painel: PainelDeMultiverso) -> None:
    painel.aceita_disparo(202, {"identificador": "abc123", "total": 4320})

    assert not painel.botao_de_disparo.isEnabled()
    assert painel.botao_de_progresso.isEnabled()
    assert painel.botao_de_cancelamento.isEnabled()
    assert painel.identificador() == "abc123"
    assert painel.campo_do_identificador.text() == "abc123", (
        "o disparo não encheu o campo de leitura, e o profissional teria de copiar o identificador"
    )


def test_terminada_o_cancelar_desliga_e_o_disparo_volta(painel: PainelDeMultiverso) -> None:
    painel.aceita_disparo(202, {"identificador": "abc123", "total": 9})

    painel.recebe_progresso(
        200,
        {"estado": "concluido", "feito": 9, "total": 9, "terminou": True},
    )

    assert painel.botao_de_disparo.isEnabled()
    assert not painel.botao_de_cancelamento.isEnabled()
    assert painel.botao_de_progresso.isEnabled()


def test_d674_a_conclusao_de_outra_execucao_nao_tira_o_painel_de_a_correr(
    painel: PainelDeMultiverso,
) -> None:
    """**A conclusão lida pelo campo pode ser de OUTRA execução**, e ela não diz que esta acabou.

    `_pede` lê o campo, que aceita o identificador de uma execução antiga colado por cima. A
    conclusão dela punha o painel fora de «a correr» com o disparo desta ainda em curso: o botão de
    disparo voltava, e a trava de uma correção por vez, que lê `a_correr`, deixava passar.
    **Par causal:** a conclusão da execução disparada aqui é a que desliga.
    """
    painel.aceita_disparo(202, {"identificador": "m-atual", "total": 8, "persistido": True})
    assert painel.a_correr(), "âncora: o disparo aceito não pôs o painel a correr"

    antiga = _corpo_com_veredito()
    antiga["identificador"] = "m-antiga"
    painel.campo_do_identificador.setText("m-antiga")
    painel.mostra(200, antiga)
    assert painel.linhas_mostradas(), "âncora: a conclusão antiga não foi mostrada"
    assert painel.a_correr(), "a conclusão de outra execução tirou o painel de «a correr»"
    assert not painel.botao_de_disparo.isEnabled()

    atual = _corpo_com_veredito()
    atual["identificador"] = "m-atual"
    painel.mostra(200, atual)
    assert not painel.a_correr(), "a conclusão desta execução não a deu por terminada"
    assert painel.botao_de_disparo.isEnabled()


def test_um_disparo_recusado_nao_deixa_o_painel_a_correr(painel: PainelDeMultiverso) -> None:
    """**A direção que um teste de caminho feliz não pega.**

    Um `422` que deixasse `_a_correr = True` travaria o botão de disparo para sempre, e o
    profissional não teria como tentar outra vez — botão morto por estado, que é pior que botão
    ausente.
    """
    texto = painel.aceita_disparo(422, {"detail": "grupo 'X' não existe nesta matriz"})

    assert painel.botao_de_disparo.isEnabled()
    assert not painel.botao_de_cancelamento.isEnabled()
    assert "não existe" in texto


# ---------------------------------------------------------------------------
# O PROGRESSO — os cinco estados dizem coisas diferentes
# ---------------------------------------------------------------------------


def test_a_parar_e_diferente_de_a_correr_e_de_parado() -> None:
    """**Os três não-terminais, e é o do meio que o botão de cancelar cria.**

    `cancelamento_pedido` com `estado` ainda `executando` é *«a parar»*. Escrever *«parado»* ali
    afirmaria um terminal que não aconteceu; escrever *«a correr»* faria o botão parecer morto.
    """
    correndo = texto_do_progresso(
        {"estado": "executando", "feito": 10, "total": 100, "cancelamento_pedido": False}
    )
    a_parar = texto_do_progresso(
        {"estado": "executando", "feito": 10, "total": 100, "cancelamento_pedido": True}
    )
    parado = texto_do_progresso({"estado": "cancelado", "feito": 10, "total": 100})

    assert len({correndo, a_parar, parado}) == 3, "dois dos três estados dizem a mesma coisa"
    assert "a parar" in a_parar
    assert "a correr" in correndo
    assert "parada a seu pedido" in parado


def test_o_cancelado_diz_quanto_correu_e_por_que_nao_ha_conclusao() -> None:
    """A fração parcial é **o dado**: distingue parar no primeiro segundo de parar no fim."""
    texto = texto_do_progresso({"estado": "cancelado", "feito": 137, "total": 4320})

    assert "137 de 4320" in texto
    assert "grade inteira" in texto, "não explica por que não há conclusão parcial"


def test_o_falhou_nomeia_a_celula() -> None:
    """*"O multiverso falhou"* não diz onde reexecutar, e a grade tem milhares de células."""
    texto = texto_do_progresso(
        {"estado": "falhou", "feito": 3, "total": 9, "celula_que_falhou": "c-3", "erro": "boom"}
    )

    assert "c-3" in texto and "boom" in texto


# ---------------------------------------------------------------------------
# O CANCELAMENTO
# ---------------------------------------------------------------------------


def test_cancelar_sem_execucao_diz_em_vez_de_voltar_em_silencio(
    painel: PainelDeMultiverso, qtbot: Any
) -> None:
    """O defeito na direção certa: *"clicar nao muda nada"* era um slot mudo."""
    vistos: list[str] = []
    painel.cancelar.connect(vistos.append)

    painel._pede_cancelamento()

    assert vistos == [], "emitiu sem execução"
    assert painel.rotulo_de_progresso.text() == SEM_IDENTIFICADOR_DO_MULTIVERSO


def test_cancelar_escreve_a_parar_antes_de_a_resposta_chegar(
    painel: PainelDeMultiverso,
) -> None:
    """**O intervalo entre o clique e o `202` não pode ser mudo.**

    Sem esta linha o usuário carregaria outra vez — que é exactamente o comportamento para o qual a
    rota foi feita idempotente.
    """
    painel.aceita_disparo(202, {"identificador": "abc123", "total": 9})
    vistos: list[str] = []
    painel.cancelar.connect(vistos.append)

    painel._pede_cancelamento()

    assert vistos == ["abc123"]
    assert "a parar" in painel.rotulo_de_progresso.text()
    assert not painel.botao_de_cancelamento.isEnabled(), "dá para pedir a parada duas vezes"


def test_o_409_do_cancelamento_manda_ler_a_conclusao_e_nao_repetir(
    painel: PainelDeMultiverso,
) -> None:
    """`409` significa *"já terminou"*, e a ação é ler — não repetir o pedido."""
    painel.aceita_disparo(202, {"identificador": "abc123", "total": 9})

    texto = painel.aceita_cancelamento(
        409, {"detail": "a execução já terminou — o estado é «concluido»"}
    )

    assert "Ler conclusão" in texto
    assert painel.botao_de_disparo.isEnabled()


# ---------------------------------------------------------------------------
# AS ROTAS — uma fonte para cada caminho
# ---------------------------------------------------------------------------


def test_as_duas_rotas_novas_batem_com_o_esquema_da_api() -> None:
    """As rotas que o painel compõe existem na aplicação. **Bijeção contra a OpenAPI.**

    Sem isto, um erro de digitação num caminho produziria `404` só em produção — e o painel
    mostraria a mensagem da rota, que nesse caso não existe.
    """
    from refino.api.app import build_app

    # **A OpenAPI e não `app.routes`, e a primeira versão usou o segundo e reprovou.** Nesta
    # versão do FastAPI `routes` traz objetos `_IncludedRouter` sem `.path`, então a travessia
    # media a estrutura interna do framework em vez do contrato. A OpenAPI é o que os clientes
    # leem, e é contra ela que `tests/test_api_contrato.py` já compara.
    caminhos = set(build_app().openapi()["paths"])

    assert rota_do_progresso_do_multiverso("{identificador}") in caminhos
    assert rota_do_cancelamento("{identificador}") in caminhos


# ---------------------------------------------------------------------------
# O veredito passa a ser servido, com o corte ao lado
# ---------------------------------------------------------------------------


def _corpo_com_veredito(verdict: str = "robust", com_limiares: bool = True) -> dict[str, Any]:
    """Uma conclusão como a rota a devolve, com o veredito e o corte."""
    corpo: dict[str, Any] = {
        "identificador": "m1",
        "grupos": ["A", "B"],
        "n_celulas": 96,
        "n_features": 1,
        "estabilidade": [
            {
                "feature": "2.25_810.7739m/z",
                "verdict": verdict,
                "frac_significant": 0.9,
                "n_cells": 96,
                "median_qvalue": 0.01,
                "median_rank": 3.0,
                "rank_iqr": 1.0,
                "sign_consistency": 0.99,
            }
        ],
    }
    if com_limiares:
        corpo["limiares"] = {"robust_frac": 0.8, "robust_sign": 0.95, "absent_frac": 0.05}
    return corpo


def test_u7_a_tabela_mostra_o_veredito_traduzido(qtbot: Any) -> None:
    """O rótulo do domínio não chega cru à tela.

    `robust`/`fragile`/`absent` são vocabulário do modelo, e quem lê é pesquisador de metabolómica.
    **A âncora vem antes:** o texto traduzido tem de ser DIFERENTE do cru, senão a tradução é um
    laço de identidade e este teste não mede nada.
    """
    painel = PainelDeMultiverso()
    qtbot.addWidget(painel)
    assert VEREDITOS["robust"] != "robust", "âncora: a tradução é identidade e não traduz nada"

    painel.mostra(200, _corpo_com_veredito("robust"))
    vistos = painel.vereditos_mostrados()
    assert vistos == (VEREDITOS["robust"],), f"a coluna do veredito trouxe {vistos}"


def test_u7_os_tres_vereditos_dao_textos_distintos(qtbot: Any) -> None:
    """Três desfechos, e o mecanismo tem de produzir os três.

    Um dicionário que mapeasse dois dos três para a mesma palavra faria a tela colapsar «frágil» e
    «não sobrevive», que pedem ações opostas: um manda olhar com cuidado, o outro manda descartar.
    """
    painel = PainelDeMultiverso()
    qtbot.addWidget(painel)

    vistos = []
    for rotulo in ("robust", "fragile", "absent"):
        painel.mostra(200, _corpo_com_veredito(rotulo))
        vistos.append(painel.vereditos_mostrados()[0])

    assert len(set(vistos)) == 3, f"dois dos três vereditos deram o mesmo texto: {vistos}"


def test_u7_veredito_desconhecido_sai_como_veio_em_vez_de_virar_nao_medido(qtbot: Any) -> None:
    """O ramo `else` da tradução.

    Se o domínio ganhar um quarto veredito, a tela mostra o nome novo — feio e honesto — em vez de
    o esconder atrás de «não medido», que significa outra coisa.
    """
    painel = PainelDeMultiverso()
    qtbot.addWidget(painel)
    painel.mostra(200, _corpo_com_veredito("quarto_veredito"))
    visto = painel.vereditos_mostrados()[0]
    assert visto == "quarto_veredito", f"o rótulo desconhecido não sobreviveu: {visto!r}"
    assert visto != "não medido", (
        "um veredito desconhecido virou «não medido», que significa *não foi testada em célula "
        "nenhuma* — dois fatos diferentes na mesma célula"
    )


def test_u7_o_cabecalho_declara_o_corte_quando_ele_vem(qtbot: Any) -> None:
    """Veredito sem o corte que o produziu é número sem escopo na tela."""
    painel = PainelDeMultiverso()
    qtbot.addWidget(painel)
    texto = painel.mostra(200, _corpo_com_veredito())
    assert "0.8" in texto and "0.95" in texto, f"o cabeçalho não declara o corte: {texto}"
    assert "não foi calibrado" in texto, (
        "o cabeçalho declara o corte e não diz que ele não foi calibrado para estes dados — quem "
        "lê toma o veredito por sentença"
    )


def test_u7_sem_limiares_o_cabecalho_declara_a_ausencia_em_vez_de_inventar(qtbot: Any) -> None:
    """**O par causal do cabeçalho.** Uma resposta de versão anterior não traz `limiares`.

    Afirmar um corte que não veio seria pior que declarar que não há: o leitor confiaria num
    número que a resposta não sustenta.
    """
    painel = PainelDeMultiverso()
    qtbot.addWidget(painel)
    com = painel.mostra(200, _corpo_com_veredito(com_limiares=True))
    sem = painel.mostra(200, _corpo_com_veredito(com_limiares=False))
    assert com != sem, "âncora: o cabeçalho não muda com e sem limiares, e o par causal é vácuo"
    assert "não veio na resposta" in sem, f"a ausência do corte não é declarada: {sem}"


# ---------------------------------------------------------------------------
# A estimativa passa a usar o modo carregado
# ---------------------------------------------------------------------------


def test_u9_com_modo_a_estimativa_e_ponto_e_sem_modo_e_faixa() -> None:
    """O par causal do modo, e sem ele a assinatura nova seria decorativa.

    Uma implementação que aceitasse `modo` e o ignorasse devolveria o mesmo texto nos dois braços,
    e todos os outros testes de estimativa continuariam verdes.
    """
    _c1, _s1, com = estimativa_de_custo(("none", "loess", "serrf"), "neg_hilic")
    _c2, _s2, sem = estimativa_de_custo(("none", "loess", "serrf"))

    assert com != sem, "passar o modo não mudou o texto: o argumento está a ser ignorado"
    assert "neg_hilic" in com and "carregou" in com, f"o texto não nomeia o modo aberto: {com}"
    assert " a " in sem, "sem modo o texto tem de continuar a ser faixa"


def test_u9_o_segundos_do_modo_bate_com_a_tabela_daquele_modo() -> None:
    """O número, e não só o texto. **A âncora é a ORDEM entre dois modos que a tabela separa.**

    `pos_hilic` é o mais caro dos quatro e `neg_hilic` o mais barato — está afirmado em
    `test_a_faixa_da_estimativa_cresce_com_o_modo`. Se a estimativa por modo devolvesse o mesmo
    número para os dois, ela não estaria a usar a tabela.
    """
    _c1, caro, _t1 = estimativa_de_custo(("serrf",), "pos_hilic")
    _c2, barato, _t2 = estimativa_de_custo(("serrf",), "neg_hilic")

    assert caro > barato, (
        f"o `pos_hilic` estimou {caro:.0f} s e o `neg_hilic` {barato:.0f} s: a estimativa por modo "
        "não está a ler a tabela por modo"
    )


def test_u9_modo_desconhecido_cai_na_faixa_em_vez_de_inventar_um_ponto() -> None:
    """**Ausência de informação não pode virar número.**

    Mesma política que a função já usa para corretor sem custo medido: a ferramenta não finge que
    sabe. Um `KeyError` seria pior — impediria o disparo por causa de um nome de modo novo.
    """
    _c1, _s1, desconhecido = estimativa_de_custo(("none",), "modo_que_nao_existe")
    _c2, _s2, sem = estimativa_de_custo(("none",))

    assert desconhecido == sem, (
        "um modo sem medição produziu um texto próprio: ou inventou um ponto, ou levantou onde "
        "devia degradar para a faixa"
    )


def test_u9_o_texto_nao_afirma_ausencia_de_paralelismo_que_a_conta_aplica() -> None:
    """A frase dizia «sem paralelismo» e a conta é COM.

    A fórmula divide por `GANHO_DO_CORRETOR` e `GANHO_DAS_FOLHAS`, os dois medidos, e o comentário
    ao lado dela declara que a faixa descreve o regime paralelo — que é o que a rota executa. O
    texto na tela afirmava o oposto três linhas abaixo da conta que o contradizia.

    **A direção do erro é a que importa:** quem lesse «sem paralelismo» esperaria a execução
    demorar MAIS que o previsto, e ela demora menos. Um usuário que planeasse a tarde por aquele
    número planearia errado.
    """
    _celulas, _segundos, texto = estimativa_de_custo(("none", "loess", "serrf"))
    assert "sem paralelismo" not in texto, (
        f"o texto afirma ausência de paralelismo e a fórmula o aplica: {texto}"
    )
    # E o oráculo mudou de PALAVRA para PROPRIEDADE. Ele exigia a
    # palavra `paralelismo`, e o texto passou a nomear o regime inteiro — *"no regime que esta
    # ferramenta corre: a correção em paralelo e a grade em sequência"* —, que é mais informativo
    # que a palavra sozinha e não a contém.
    #
    # **A propriedade é: o texto diz de que lado o paralelismo está.** Ela sobrevive às duas
    # caducidades que esta linha já apanhou — *«sem paralelismo»* sobre uma conta que era com, e
    # *«com o paralelismo que a execução usa»* sobre um regime que a rota não corre.
    assert "paralelo" in texto, f"o texto deixou de declarar o regime em que o número vale: {texto}"
    assert "sequência" in texto or "sequencial" in texto, (
        "o texto nomeia o lado paralelo e não o sequencial. Os dois níveis existem, a rota liga "
        f"um, e dizer só metade é o que fazia a estimativa parecer descrever o outro: {texto}"
    )


# ---------------------------------------------------------------------------
# Os TRÊS LIMIARES DO VEREDITO, controláveis
# ---------------------------------------------------------------------------


def test_o_painel_oferece_os_tres_limiares_e_repousa_no_padrao(qtbot: Any) -> None:
    """**O repouso é «não escolhi», e é a metade que uma implementação desatenta perderia.**

    Se o painel nascesse com os valores padrão já postos, toda leitura mandaria os três na query —
    e a rota deixaria de usar os próprios padrões, passando a depender de a tela os repetir
    corretamente. É a segunda fonte de verdade sobre um limiar.

    **As chaves saem de `LIMIARES_DO_VEREDITO` e não são redigitadas aqui**, e é isso que impede
    o arnês de continuar verde no dia em que o painel oferecer um quarto controle sem que a rota
    o aceite.
    """
    painel = PainelDeMultiverso()
    qtbot.addWidget(painel)

    assert set(painel.limiares) == {chave for chave, _rotulo in LIMIARES_DO_VEREDITO}

    # O MECANISMO mudou e a PROPRIEDADE não, depois de uma passagem humana pela tela.
    #
    # Os campos mostram os valores padrão da especificação, e não um texto que diz «padrão da
    # especificação» sem dizer qual é.
    #
    # > **A asserção de antes era `campo.value() == SEM_LIMIAR` e `campo.text() == "padrão da
    # > especificação"`, e o argumento dela continua verdadeiro:** *"se o painel nascesse com os
    # > valores padrão já postos, toda leitura mandaria os três na query — e a rota deixaria de usar
    # > os próprios padrões"*. **O que o desarma é a CAIXA:** quem decide o que viaja passou a ser
    # > ela, e não o valor. O campo mostra `0,80`; o payload continua vazio.
    for chave, campo in painel.limiares.items():
        assert campo.value() == PADRAO_DO_LIMIAR[chave], (
            f"`{chave}` repousa em {campo.value()} e o padrão da especificação é "
            f"{PADRAO_DO_LIMIAR[chave]} — o campo tem de MOSTRAR o número que o serviço aplica"
        )
        assert not campo.isEnabled(), (
            f"`{chave}` nasce editável debaixo de uma caixa que diz «Automático»"
        )
    assert painel.automatico_dos_limiares.isChecked(), "a caixa nasce desticada"
    assert painel.campos_da_conclusao() == {}, (
        "o painel manda limiar com a caixa ticada — a rota perde a distinção entre «aceitei o "
        "padrão» e «escolhi este número», e as duas divergem no dia em que a §7 mudar"
    )

    # O par causal, na MESMA função: desticar faz os três viajarem.
    painel.automatico_dos_limiares.setChecked(False)
    enviados = painel.campos_da_conclusao()
    assert set(enviados) == set(PADRAO_DO_LIMIAR), (
        f"desticar devia mandar os três e mandou {sorted(enviados)} — sem isto, a asserção acima "
        f"passaria num painel que nunca manda nada"
    )


def test_o_campo_so_aceita_o_dominio_e_nao_ha_valor_que_suma(qtbot: Any) -> None:
    """A distinção entre zero e «não escolhi» continua, e mudou de PORTADOR.

    Este teste exigia `minimum() < 0.0`, com o argumento de que zero é leitura legítima
    (`absent_frac=0` significa *«nada é ausente»*) e portanto não podia ser o sentinela de *«não
    escolhi»*. **O argumento continua inteiro; o que mudou é quem responde «não escolhi»** — passou
    a ser a caixa «Automático», e o campo deixou de precisar de um valor fora do domínio.

    > **A revisão de código apanhou o buraco que sobrou:** com o sentinela ainda alcançável pela
    > roda, o profissional podia girar até `-1,0` e ver aquela chave **sumir do payload em
    > silêncio**, contra o «tudo ou nada» que o painel promete.

    O que se afirma agora é mais forte: **nenhum valor que o campo aceita desaparece do payload.**
    """
    painel = PainelDeMultiverso()
    qtbot.addWidget(painel)
    painel.automatico_dos_limiares.setChecked(False)

    for chave, campo in painel.limiares.items():
        assert campo.minimum() == 0.0, (
            f"o mínimo de `{chave}` é {campo.minimum()}: um valor fora de `[0, 1]` que o campo "
            f"aceite é um valor que a rota recusa ou que o filtro descarta em silêncio"
        )
        assert campo.maximum() == 1.0, f"`{chave}` aceita acima de 1, e os três são frações"

        # O par causal do buraco: os extremos do domínio VIAJAM, os dois.
        for extremo in (campo.minimum(), campo.maximum()):
            campo.setValue(extremo)
            enviados = painel.campos_da_conclusao()
            assert enviados.get(chave) == extremo, (
                f"`{chave}` posto em {extremo} não chegou ao payload ({enviados.get(chave)}) — um "
                f"valor que o campo aceita e que some é «escolhi» a virar «não escolhi»"
            )


def test_um_limiar_posto_em_zero_viaja_na_query(qtbot: Any) -> None:
    """**O par causal do sentinela**, e é a asserção que o `0.0` teria estragado.

    Uma implementação que filtrasse por `if valor` descartaria `absent_frac=0` em silêncio: o
    usuário escolheria *"nada é ausente"* e receberia o padrão.

    > **O filtro do sentinela saiu em 07/09/2026**, e este teste ficou — ele nunca foi sobre o
    > sentinela, foi sobre o ZERO. Com o filtro removido a asserção é a mesma e o modo de falha que
    > ela apanha é outro: hoje seria um `if valor` reintroduzido, ou o mínimo a subir acima de zero.
    """
    painel = PainelDeMultiverso()
    qtbot.addWidget(painel)

    painel.automatico_dos_limiares.setChecked(False)
    painel.limiares["absent_frac"].setValue(0.0)

    assert painel.campos_da_conclusao()["absent_frac"] == 0.0, (
        "o zero foi descartado: «escolhi zero» virou «não escolhi»"
    )


def test_desticar_manda_os_tres_e_o_que_foi_mexido_chega_com_o_valor_novo(qtbot: Any) -> None:
    """Desticar é TUDO ou NADA, e o eixo mexido chega com o valor que o humano pôs.

    ## O nome deste teste MUDOU em 07/09/2026, e a mudança é o conteúdo

    Ele chamava-se `test_escolher_um_limiar_nao_manda_os_outros_dois`, e a propriedade era essa:
    o eixo não tocado ficava no sentinela e era filtrado, então mexer num mandava **um**. Com a
    caixa «Automático», quem decide o que viaja é ela — desticada, **os três** viajam, porque os
    não tocados repousam no padrão da especificação e não num sentinela.

    > **Manter o nome antigo sobre a asserção nova seria pior que não ter teste:** ele afirmaria
    > exclusividade num desenho que a aboliu, e a próxima sessão leria o nome como contrato. A
    > revisão de código apanhou-o em dois revisores independentes.

    **A propriedade que o nome antigo protegia não se perdeu** — *«a rota pode usar os próprios
    padrões nos que ninguém tocou»* passou a ser a da caixa TICADA, e é afirmada em
    `test_o_painel_oferece_os_tres_limiares_e_repousa_no_padrao`, que exige `{}` no repouso.
    """
    painel = PainelDeMultiverso()
    qtbot.addWidget(painel)

    painel.automatico_dos_limiares.setChecked(False)
    painel.limiares["robust_frac"].setValue(0.50)

    enviados = painel.campos_da_conclusao()
    assert set(enviados) == set(PADRAO_DO_LIMIAR), (
        f"desticar é tudo ou nada e viajaram {sorted(enviados)} — um subconjunto faria a rota "
        f"misturar o que o humano escolheu com o padrão dela, e as duas metades ficariam "
        f"indistinguíveis no registro"
    )
    assert enviados["robust_frac"] == 0.50, "o eixo mexido não chegou com o valor novo"
    for chave, padrao in PADRAO_DO_LIMIAR.items():
        if chave != "robust_frac":
            assert enviados[chave] == padrao, (
                f"`{chave}` não foi tocado e viajou como {enviados[chave]} em vez do padrão "
                f"{padrao} — o campo mostra um número e manda outro"
            )


def test_campos_dos_limiares_recusa_chave_que_a_rota_nao_declara() -> None:
    """**Função de módulo, exercitada sem construir a janela** — o molde de `rota_do_resultado`.

    A guarda de chave existe porque este dicionário vira **query**: uma chave a mais seria
    descartada em silêncio pelo FastAPI, e o payload passaria a afirmar uma influência que a rota
    não tem — o defeito que `CAMPOS_POR_ARTEFATO` já existe para não ter, aqui noutra superfície.
    """
    saida = campos_dos_limiares({"robust_frac": 0.5, "inventado": 0.9})

    assert saida == {"robust_frac": 0.5}, "uma chave que a rota não declara atravessou"


def test_o_painel_abre_dizendo_que_a_funcionalidade_e_beta(painel: PainelDeMultiverso) -> None:
    """A PRIMEIRA linha do painel diz a quem opera que o multiverso é beta, fora da entrega.

    **A posição é o conteúdo**: a frase enterrada no fim do painel não avisaria ninguém. Por isso o
    par causal vive no mesmo teste — o mesmo predicado aplicado ao SEGUNDO item do layout tem de
    dar falso; sem isso, um predicado que fosse verdadeiro para qualquer item passaria aqui.
    """
    disposicao = painel.layout()
    assert disposicao is not None, "âncora: o painel não tem layout"
    assert disposicao.count() >= 2, f"âncora: o layout tem {disposicao.count()} item(ns)"

    def e_o_aviso(item: object) -> bool:
        return isinstance(item, QLabel) and item.text() == FUNCIONALIDADE_EM_BETA

    assert e_o_aviso(disposicao.itemAt(0).widget()), "o painel não abre com a frase de beta"
    assert not e_o_aviso(disposicao.itemAt(1).widget()), "par causal: o predicado não separa"


# ---------------------------------------------------------------------------
# COMO LER A CONCLUSÃO — cada coluna e cada veredito explicados, com um exemplo medido
# ---------------------------------------------------------------------------

# O exemplo vem da grade completa medida nos quatro arquivos reais, copiada byte a byte para o
# repositório. Ela é de setembro de 2026, anterior às mudanças no SERRF e no LOESS, e o texto do
# exemplo diz isso; o que este teste confere é que o texto diz o que o
# artefato diz, e não que o artefato é de hoje.
GRADE_MEDIDA: Final = Path(__file__).resolve().parents[1] / "data" / "fixtures" / "grade_completa"
ARQUIVOS_DA_GRADE: Final = (
    "pos_hilic-sem-ca88.json",
    "pos_c18-sem-ca88.json",
    "neg_hilic.json",
    "neg_c18-sem-ca88.json",
)


def _fracao(valor: float) -> str:
    return f"{valor:.2f}".replace(".", ",")


def _milhar(numero: int) -> str:
    return f"{numero:,}".replace(",", ".")


def test_cada_coluna_da_conclusao_tem_explicacao_e_nenhuma_sobra() -> None:
    """Bijeção com `COLUNAS`, na ordem da tabela: coluna nova sem explicação reprova, e explicação
    de coluna que saiu também."""
    assert list(EXPLICACAO_DAS_COLUNAS) == [chave for chave, _titulo in COLUNAS]
    assert all(texto.strip() for texto in EXPLICACAO_DAS_COLUNAS.values())


def test_cada_veredito_diz_o_corte_do_nucleo_e_o_que_fazer() -> None:
    """Os números do corte são os de `StabilityThresholds`, lidos e não redigitados."""
    assert list(EXPLICACAO_DOS_VEREDITOS) == list(VEREDITOS)
    robusta = _fracao(PADRAO_DO_LIMIAR["robust_frac"])
    sinal = _fracao(PADRAO_DO_LIMIAR["robust_sign"])
    ausente = _fracao(PADRAO_DO_LIMIAR["absent_frac"])
    assert robusta in EXPLICACAO_DOS_VEREDITOS["robust"]
    assert sinal in EXPLICACAO_DOS_VEREDITOS["robust"]
    assert ausente in EXPLICACAO_DOS_VEREDITOS["absent"]
    for limiar in (robusta, sinal, ausente):
        assert limiar in EXPLICACAO_DOS_VEREDITOS["fragile"]
    for chave, rotulo in VEREDITOS.items():
        assert EXPLICACAO_DOS_VEREDITOS[chave].startswith(f"**«{rotulo}»**")
        assert "O que fazer:" in EXPLICACAO_DOS_VEREDITOS[chave]


def _divergencias_do_exemplo_do_multiverso(texto: str) -> list[str]:
    """O que o exemplo afirma e a grade medida não sustenta; vazio quando os dois batem."""
    pares = []
    for nome in ARQUIVOS_DA_GRADE:
        grade = json.loads((GRADE_MEDIDA / nome).read_text(encoding="utf-8"))
        pares += [
            (grade["modo"], par, dados, grade["n_celulas"]) for par, dados in grade["pares"].items()
        ]
    divergencias: list[str] = []
    for celulas in sorted({celulas for *_resto, celulas in pares}):
        if f"{_milhar(celulas)} células" not in texto:
            divergencias.append(f"o exemplo não diz {_milhar(celulas)} células")
    if f"{len(pares)} pares" not in texto:
        divergencias.append(f"o exemplo não diz {len(pares)} pares")
    if any("robust" in dados["vereditos"] for _modo, _par, dados, _c in pares):
        divergencias.append("o artefato tem «sobrevive», e o exemplo diz que nenhuma sobreviveu")
    com_outro = [(m, p, d) for m, p, d, _c in pares if set(d["vereditos"]) != {"absent"}]
    if len(com_outro) != 1:
        divergencias.append(
            f"{len(com_outro)} pares com algo além de «não sobrevive», e o texto diz 1"
        )
        return divergencias
    modo, par, dados = com_outro[0]
    if dados["vereditos"] != {"absent": dados["n_features"] - 1, "fragile": 1}:
        divergencias.append(
            f"o par {par} não tem uma frágil e o resto ausente: {dados['vereditos']}"
        )
    for trecho in (
        f"`{modo}`",
        f"par {par}",
        _milhar(dados["n_features"]),
        _milhar(dados["vereditos"].get("absent", 0)),
        _fracao(dados["maior_frac_significant"]),
        f"outros {len(pares) - 1} pares",
        # O corte pela FRASE dele: «0,05» sozinho também casaria o «q < 0,05», que é outra
        # quantidade com o mesmo número.
        f"acima de {_fracao(PADRAO_DO_LIMIAR['absent_frac'])} e abaixo de "
        f"{_fracao(PADRAO_DO_LIMIAR['robust_frac'])}",
    ):
        if trecho not in texto:
            divergencias.append(f"o exemplo não diz {trecho!r}")
    return divergencias


def test_o_exemplo_do_multiverso_e_o_da_grade_medida() -> None:
    assert _divergencias_do_exemplo_do_multiverso(EXEMPLO_DO_MULTIVERSO) == []


@pytest.mark.parametrize(
    ("de", "para"),
    [
        ("0,26", "0,62"),
        ("2.135", "2.134"),
        ("HAMTSP x NoHTLV", "ATLL x NoHTLV"),
        ("24 pares", "20 pares"),
    ],
)
def test_a_conferencia_do_exemplo_do_multiverso_acusa_a_planta(de: str, para: str) -> None:
    assert EXEMPLO_DO_MULTIVERSO.count(de) == 1, f"a planta {de!r} não aterra numa instância só"
    assert _divergencias_do_exemplo_do_multiverso(EXEMPLO_DO_MULTIVERSO.replace(de, para))


def test_o_painel_explica_a_tabela_ao_abrir_e_no_cabecalho(
    painel: PainelDeMultiverso, qtbot: Any
) -> None:
    """O texto está no painel, e não só na constante: o bloco abre e mostra-o, e cada
    cabeçalho da tabela traz a explicação da sua coluna na dica."""
    assert painel.texto_de_como_ler.text() == COMO_LER_A_CONCLUSAO
    for parte in (
        *EXPLICACAO_DAS_COLUNAS.values(),
        *EXPLICACAO_DOS_VEREDITOS.values(),
        EXEMPLO_DO_MULTIVERSO,
    ):
        assert parte in COMO_LER_A_CONCLUSAO
    for indice, (chave, _titulo) in enumerate(COLUNAS):
        item = painel.tabela.horizontalHeaderItem(indice)
        assert item is not None
        assert item.toolTip() == EXPLICACAO_DAS_COLUNAS[chave]
    assert not painel.texto_de_como_ler.isVisibleTo(painel), "âncora: fechado, o texto não se vê"
    painel.como_ler.alterna.setChecked(True)
    qtbot.waitUntil(lambda: painel.texto_de_como_ler.isVisibleTo(painel))
