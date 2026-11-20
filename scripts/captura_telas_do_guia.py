#!/usr/bin/env python3
"""Recaptura as telas do `docs/GUIA-do-aplicativo.html`, aba a aba.

## Por que existe

As seis imagens do guia eram PNGs feitos **à mão**, e por isso envelheciam sem que nada
acusasse: `grep` por captura, screenshot ou `grab` no repositório devolvia **zero**. Uma vez
o guia descreveu três pares de botões que já tinham virado um, e a imagem mostrava a versão
antiga.

> **Andaime que informa é instrumento.** Estas imagens vão para o guia, e
> uma tela que mostra um botão que não existe mais é afirmação errada com aparência de
> evidência.

## O que ele NÃO faz, declarado

Não inventa dado. A janela é construída com o **transporte real** contra o servidor
embutido, e os painéis são preenchidos por chamadas de verdade sobre os arquivos de
referência. Um duplo com respostas de mentira produziria telas plausíveis e falsas, que
é pior que a tela velha — ela ao menos foi verdade um dia.

Não conduz o rato. Os painéis são preenchidos pela API pública deles, que é o mesmo que
`tests/test_desktop_janela.py` faz; o que se captura é o widget desenhado.

## Uso

    .venv/Scripts/python.exe scripts/captura_telas_do_guia.py [--saida out/telas]
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any

# **A plataforma NATIVA, e `offscreen` foi MEDIDO e recusado.** A primeira versão deste
# arnês forçava `offscreen`, que é o que `tests/qt_guard.py` usa. Ele produziu as oito
# imagens, todas acima do piso de tamanho, com `EXIT=0` — e **todo o texto era caixinha**:
# sob `offscreen` não há fonte, e cada caractere sai como `□`.
#
# > **O piso de tamanho não viu, porque caixinha tem pixels.** A guarda media o arquivo
# > não estar vazio, e a pergunta era outra — se o texto tinha desenhado. O veredito era
# > sobre o alvo e o defeito estava no instrumento, com o desfecho de SUCESSO. Quem pegou
# > foi abrir o PNG e olhar.
#
# Aqui a janela abre de verdade, por um instante, e é a mesma que o usuário vê.
if os.environ.get("REFINO_CAPTURA_OFFSCREEN") == "1":  # pragma: no cover - só para diagnóstico
    os.environ["QT_QPA_PLATFORM"] = "offscreen"

sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]

REPO = Path(__file__).resolve().parent.parent

# O menor dos quatro conjuntos, e o que o guia usa como exemplo:
# 2.184 features por 68 amostras.
MODO = "neg_c18"

# **O tamanho do `GUIA`, e ele é o PADRÃO para o passo 0 do roteiro de entrega não mudar.** Quem
# pede outro passa `--tamanho`; quem não pede recebe exactamente o que `embute_telas_no_guia.py`
# espera encontrar em `out/telas/`, com os nomes achatados na pasta.
PADRAO_DE_TAMANHO = "1280x860"

# Teto de espera da execucao. `none` mede 1,3 s neste conjunto; 90 s e folga larga.
TETO_DA_EXECUCAO_MS = 90000

# Voltas de espera pela sondagem automatica. `none` fecha em segundos; 12 x 1,5 s e folga larga.
# O que o numero mede e o tempo que a execucao tem para acabar.
TENTATIVAS_DE_PROGRESSO = 12


def _dimensao(texto: str) -> tuple[int, int]:
    """`1920x1080` vira `(1920, 1080)`. **Levanta em vez de recuar num padrão.**

    Um recuo silencioso aqui produziria capturas de 1280 com nome de 1920 — o arnês imprimiria
    sucesso sobre um tamanho que ninguém pediu.
    """
    largura, _, altura = texto.lower().partition("x")
    if not largura.isdigit() or not altura.isdigit():
        raise argparse.ArgumentTypeError(f"tamanho inválido: {texto!r} — esperado LARGURAxALTURA")
    return int(largura), int(altura)


def _bombeia(app: object, ms: int) -> None:
    """Gira o laço de eventos por `ms`, para a resposta da rota chegar e a tela redesenhar.

    **A chamada é assíncrona:** o painel emite sinal, a janela põe a rota numa piscina, e
    o rótulo só muda quando a resposta volta. Capturar sem esperar produziria a tela do
    estado anterior — e ela é indistinguível de uma tela legítima.
    """
    from PySide6.QtCore import QDeadlineTimer, QEventLoop

    prazo = QDeadlineTimer(ms)
    while not prazo.hasExpired():
        app.processEvents(QEventLoop.ProcessEventsFlag.AllEvents, 50)  # type: ignore[attr-defined]


def _clica_quando_ligado(app: object, botao: object, teto_ms: int = 5000) -> str | None:
    """Clica o botão da etapa assim que ele ligar. Devolve `None`, ou por que ele seguiu desligado.

    **Clicar um botão desligado não faz nada, e não diz nada.** A
    trava do Palpite desliga as etapas enquanto ele mede, e é o relógio das etapas, de meio em meio
    segundo, que as religa depois de ele chegar. Enquanto existiu, o gesto que precedia o clique —
    desticar a caixa «Automático» — dava esse tempo; sem ele, este arnês clicou 200 ms depois do
    Palpite, o clique caiu num botão ainda desligado, e a etapa ficou «não feita» até o teto.
    """
    prazo = time.monotonic() + teto_ms / 1000.0
    while not botao.isEnabled():  # type: ignore[attr-defined]
        if time.monotonic() > prazo:
            return str(botao.toolTip() or "o botão seguiu desligado, sem motivo à vista")  # type: ignore[attr-defined]
        _bombeia(app, 100)
    botao.click()  # type: ignore[attr-defined]
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--saida", default="out/telas", help="pasta onde gravar os PNG")
    parser.add_argument(
        "--tamanho",
        action="append",
        type=_dimensao,
        metavar="LARGURAxALTURA",
        help=f"repetível; cada um numa subpasta. Padrão: {PADRAO_DE_TAMANHO}",
    )
    args = parser.parse_args()
    dimensoes: list[tuple[int, int]] = args.tamanho or [_dimensao(PADRAO_DE_TAMANHO)]

    destino = REPO / args.saida
    destino.mkdir(parents=True, exist_ok=True)

    from PySide6.QtWidgets import QApplication

    from refino.desktop.arranque import arranca
    from refino.desktop.janela.arquivos import ROTULO_DA_LISTA, ROTULO_DAS_MEDIDAS
    from refino.desktop.janela.pipeline import ROTULO_DE_CONCLUSAO, TEXTO_DO_ESTADO_DA_ETAPA
    from refino.desktop.janela.principal import ABAS, JanelaPrincipal
    from refino.interface.cliente import constroi_transporte

    servidor, _banco = arranca(migrar=True)
    try:
        # **O par de arquivos REAIS, injetado pelo escolhedor.** A janela aceita um
        # `Escolhedor` por construtor exactamente para isto; o diálogo de ficheiro não
        # abre, e o que os painéis recebem são os caminhos do laboratório.
        raiz_dos_dados = REPO / "serrf" / "data" / "input"
        # **Por TÍTULO, e não por posição.** O painel tem dois botões e pede um arquivo de cada
        # vez, dizendo qual no título do diálogo. Um escolhedor que ignorasse o título e servisse
        # os caminhos por ordem de chamada continuaria a produzir a captura certa **e** deixaria
        # de exercitar o que os dois botões existem para consertar.
        por_titulo = {
            ROTULO_DAS_MEDIDAS: raiz_dos_dados / f"measurements_{MODO}.csv",
            ROTULO_DA_LISTA: raiz_dos_dados / f"Sample_List_{MODO}.csv",
        }
        for caminho in por_titulo.values():
            if not caminho.is_file():
                print(f"  dado de referência ausente: {caminho}", file=sys.stderr)
                print(
                    "  Sem ele as telas sairiam vazias, e inventar número é pior.", file=sys.stderr
                )
                return 1

        # **A JANELA VESTIDA COMO O APLICATIVO A ABRE.** `arranque.abre_janela` chama
        # `veste_a_aplicacao` antes de construir e `veste_a_janela` depois, e é isso que monta a
        # barra lateral. Sem as duas, a captura mostrava as abas em
        # cima, uma janela que o usuário nunca vê. A captura é do artefato montado, e o artefato é
        # o vestido.
        from refino.desktop.janela.tema import veste_a_aplicacao, veste_a_janela

        app = QApplication.instance() or QApplication([])
        veste_a_aplicacao(app)  # type: ignore[arg-type]
        janela = JanelaPrincipal(
            constroi_transporte(), escolhedor=lambda titulo: por_titulo[titulo]
        )
        veste_a_janela(janela)
        janela.resize(*dimensoes[0])
        janela.show()
        app.processEvents()

        # Preenche o que o usuário veria: escolher os arquivos, e correr as três
        # leituras. Nada aqui fabrica resposta — quem responde é o servidor embutido.
        janela.arquivos.escolhe_medidas()
        janela.arquivos.escolhe_lista()
        _bombeia(app, 300)
        # **O par tem de ter ficado COMPLETO, e isso vai conferido.** Um `KeyError` do escolhedor
        # por título já levantaria; o que esta guarda apanha é a metade que ficou por escolher —
        # e sem ela a captura sairia com a tela por preencher e `EXIT=0`.
        if janela.arquivos.par is None:
            print(f"  o par não ficou completo: {janela.arquivos.rotulo.text()!r}", file=sys.stderr)
            return 1

        # **O modo tem de casar com os arquivos, senão a tela sai cheia de aviso.** Medido:
        # com o combo no padrão `pos_hilic` e os arquivos do `neg_c18`, a validação devolve
        # `MODE_CONTENT_MISMATCH` em 68 de 68 amostras. O aviso está **certo** — é o produto
        # a funcionar —, mas uma tela de guia que mostra o programa a reclamar de um erro
        # que o próprio arnês cometeu ensina a coisa errada.
        janela.arquivos.modo.setCurrentText(MODO)
        _bombeia(app, 200)
        if janela.arquivos.modo.currentText() != MODO:
            print(
                f"  o combo não aceitou `{MODO}`: está em `{janela.arquivos.modo.currentText()}`",
                file=sys.stderr,
            )
            return 1
        # **As chaves são as do painel e vão CONFERIDAS, não supostas.** A primeira versão
        # usou os verbos das rotas — `validar`, `diagnosticar`, `recomendar` — e o painel
        # usa os substantivos. O `.get()` devolvia `None` e o arnês seguia em silêncio,
        # capturando a tela vazia com `EXIT=0`.
        esperadas = ("validacao", "diagnostico", "recomendacao")
        faltam = [c for c in esperadas if c not in janela.leitura.botoes]
        if faltam:
            print(f"  chaves ausentes em `botoes`: {faltam}", file=sys.stderr)
            print(f"  o painel oferece: {sorted(janela.leitura.botoes)}", file=sys.stderr)
            return 1
        for chave in esperadas:
            janela.leitura.botoes[chave].click()
            _bombeia(app, 5000)

        # **O PALPITE MEDIDO LEVA PERTO DE UM MINUTO, e até ele chegar a janela RECUSA a correção
        # que este arnês dispara a seguir.** Os 5 s acima bastavam à árvore, que
        # respondia em 0,4 s; o Palpite passou a medir os três corretores em 200 features, e a trava
        # `PALPITE_EM_CURSO` recusa o clique em «Executar» enquanto ele corre. Sem esta espera a
        # captura sairia com o Palpite a «medir» e a aba de qualidade inteira «não calculada», com
        # `EXIT=0` — o desfecho que o parágrafo de baixo existe para evitar.
        sys.path.insert(0, str(REPO / "scripts"))
        from espera_o_palpite import CHEGOU, espera_o_palpite, frase_do_palpite

        from refino.desktop.janela.leitura import CHAVE_DO_PALPITE

        desfecho = espera_o_palpite(janela, lambda ms: _bombeia(app, ms), chave=CHAVE_DO_PALPITE)
        if desfecho != CHEGOU:
            print(
                f"  o Palpite não chegou ({desfecho}): "
                f"{frase_do_palpite(janela, CHAVE_DO_PALPITE)!r}",
                file=sys.stderr,
            )
            return 1
        # **As três frases vão impressas porque o GUIA as transcreve**, e ler o PNG não é
        # transcrevê-lo: o bloco «O que apareceu no exemplo» passa a ter de onde vir.
        for chave in esperadas:
            print(f"  {chave}: {janela.leitura.rotulos[chave].text()}")

        # **E uma execução de verdade, senão a aba de qualidade sai toda «não calculada».**
        # Medido: sem isto, as onze métricas aparecem sem valor, e a tela do guia deixa de
        # ensinar o que ela existe para ensinar. O corretor é `none` de propósito — é o mais
        # barato (1,3 s neste conjunto) e a aba de qualidade preenche igual.
        #
        # O corretor do seletor é sempre o que corre, e a escolha à mão de `none` vai conferida.
        # **O Palpite é esperado acima**: ele não decide o corretor, mas a trava `PALPITE_EM_CURSO`
        # recusa a correção enquanto ele mede, porque os dois disputariam os mesmos núcleos.
        janela.pipeline.corretor.setCurrentText("none")
        _bombeia(app, 200)
        if janela.pipeline.corretor.currentText() != "none":
            print(
                f"  o seletor não aceitou `none`: está em "
                f"`{janela.pipeline.corretor.currentText()}`",
                file=sys.stderr,
            )
            return 1

        # **As etapas, pelos botões delas**: cada etapa roda na aba dela. A Correção
        # primeiro — a espera do Palpite acima veio antes, porque ele trava as correções enquanto
        # mede —, e depois a Análise, que enche a aba de qualidade a partir da matriz corrigida. O
        # oráculo de cada uma é o estado ao lado do botão dela, que é o que a pessoa lê.
        recusa = _clica_quando_ligado(app, janela.pipeline.botoes_de_etapa[1])
        if recusa is not None:
            print(f"  «1. Corrigir» seguiu desligado: {recusa}", file=sys.stderr)
            return 1
        limite = time.monotonic() + TETO_DA_EXECUCAO_MS / 1000.0
        em_dia = TEXTO_DO_ESTADO_DA_ETAPA["em_dia"]
        while janela.pipeline.rotulos_de_etapa[1].text() != em_dia:
            if time.monotonic() > limite:
                print(
                    "  a correção não concluiu: "
                    f"{janela.pipeline.rotulos_de_etapa[1].text()!r} — "
                    f"{janela.rotulo_de_desfecho.text()!r}",
                    file=sys.stderr,
                )
                return 1
            _bombeia(app, 1000)
        recusa = _clica_quando_ligado(app, janela.pipeline.botoes_de_etapa[3])
        if recusa is not None:
            print(f"  «3. Analisar e registrar» seguiu desligado: {recusa}", file=sys.stderr)
            return 1
        _bombeia(app, 3000)

        # **NINGUÉM CLICA: a aba sonda-se sozinha.** O relógio da
        # janela relê de segundo em segundo, e o que este arnês precisa é de dar tempo ao laço de
        # eventos para ele tocar.
        for _ in range(TENTATIVAS_DE_PROGRESSO):
            _bombeia(app, 1500)

        # **E a execução tem de ter CONCLUÍDO, e isso passou a ser conferido.** Este arnês
        # nunca o conferiu: uma correção recusada pela trava, ou que falhasse, deixava a aba de
        # qualidade «não calculada» e saía com `EXIT=0`. O teto é `TETO_DA_EXECUCAO_MS`, que estava
        # declarado e sem leitor.
        limite = time.monotonic() + TETO_DA_EXECUCAO_MS / 1000.0
        # **A palavra vem da constante da tela**: a forma literal «concluído» deixou de
        # existir com a maiúscula, e o laço só saía pelo teto.
        while ROTULO_DE_CONCLUSAO not in janela.pipeline.rotulo_de_progresso.text():
            if time.monotonic() > limite:
                print(
                    "  a execução não concluiu: "
                    f"{janela.pipeline.rotulo_de_progresso.text()!r} — "
                    f"{janela.rotulo_de_desfecho.text()!r}",
                    file=sys.stderr,
                )
                return 1
            _bombeia(app, 1000)

        # **OS TAMANHOS CORREM NA MESMA SESSAO, e a economia NAO e o argumento.**
        #
        # Encher a janela custa ~40 s — tres leituras e uma execucao —, e repetir isso por tamanho
        # daria quatro sessoes. O que decide e outra coisa: quatro sessoes produziriam **quatro
        # estados**, e a leitura humana passaria a comparar layout com dado diferente por baixo.
        # Redimensionar a MESMA janela mantem constante tudo menos a variavel que se quer ler.
        escritos: list[tuple[Path, int]] = []
        for largura, altura in dimensoes:
            janela.resize(largura, altura)
            _bombeia(app, 800)
            # **O tamanho EFETIVO vai impresso, porque `resize` e um PEDIDO e nao uma ordem.** Numa
            # tela de 1920x1032 uteis, uma janela de 1080 de altura pode ser encolhida pelo gestor,
            # e a leitura passaria a descrever um tamanho que ninguem pediu.
            #
            # **E a BARRA DE ABAS mede-se AQUI, em vez de se herdar de `_dimensiona`.** O `1583` que
            # a docstring dele cita e a barra de OITO legendas, e a propria docstring ja o declara
            # caduco. O numero abaixo e o das legendas de hoje, no tamanho de hoje.
            #
            # **SAO DOIS NUMEROS E MEDEM COISAS DIFERENTES, e a primeira versao imprimiu so o
            # errado.** Com `ElideRight` na barra, `minimumSizeHint` e pequeno POR CONSTRUCAO — ele
            # diz ate onde a barra encolhe cortando as legendas com reticencias, e por isso NUNCA
            # responde «as seis legendas cabem?». Quem responde e o `sizeHint`, que e a largura que
            # a barra quer para as desenhar inteiras. Imprimir so o minimo daria um numero
            # confortavel sobre o instrumento, lido como veredito sobre o alvo.
            # **E o denominador e a largura do `QTabWidget`, nao a da barra.** `barra.width()`
            # devolve o que a barra ocupa — que e o que ela pediu —, entao comparar um com o outro
            # da folga zero em qualquer tela e nao responde nada.
            barra = janela.abas.tabBar()
            print(
                f"\n{largura}x{altura} — janela efetiva {janela.width()}x{janela.height()}, "
                f"barra de {barra.count()} legendas: quer {barra.sizeHint().width()} px inteiras "
                f"contra {janela.abas.width()} px de aba, e encolhe ate "
                f"{barra.minimumSizeHint().width()} px com reticencias"
            )
            pasta = destino if len(dimensoes) == 1 else destino / f"{largura}x{altura}"
            pasta.mkdir(parents=True, exist_ok=True)
            grupo = _captura_as_abas(app, janela, pasta)
            if grupo is None:
                return 1
            escritos.extend(grupo)

        # **A ANCORA DE NAO-VACUIDADE E DERIVADA, e nunca escrita.** O numero de imagens e o de
        # abas vezes o de tamanhos. Escrever `24` faria o portao concordar com o engano no dia em
        # que uma aba entrasse sem passo de captura — que e o defeito de 03/09/2026, de novo.
        #
        # **E ela conta CAMINHOS DISTINTOS EM DISCO, e nao voltas do laco.** A primeira versao
        # comparava `len(escritos)`, e isso e tautologico: o laco apende exactamente uma vez por
        # aba, entao a igualdade nao podia falhar — verificacao decorativa, indistinguivel de
        # verificacao ativa. Contar caminhos apanha o caso real que resta: dois tamanhos IGUAIS na
        # linha de comando escrevem na mesma pasta, a segunda volta sobrescreve a primeira, e o
        # disco fica com metade das imagens que o relatorio anuncia.
        imagens_esperadas = len(ABAS) * len(dimensoes)
        distintos = {caminho for caminho, _tamanho in escritos}
        if len(distintos) != imagens_esperadas:
            print(
                f"FALHA: {len(distintos)} imagens distintas em disco para {len(ABAS)} abas x "
                f"{len(dimensoes)} tamanhos = {imagens_esperadas}. Dois tamanhos iguais na "
                "linha de comando escrevem na mesma pasta e uma volta sobrescreve a outra",
                file=sys.stderr,
            )
            return 1

        print("\nO piso acima só diz que os arquivos não estão vazios.")
        print(f"ABRA AS {len(escritos)} E OLHE antes de embutir: se o texto sair como caixinha, a")
        print("plataforma do Qt não achou fonte, e o desfecho de sucesso é falso.")
        print(f"\n{len(escritos)} telas em {destino.relative_to(REPO)}")
        return 0
    finally:
        servidor.desliga()


def _captura_as_abas(app: Any, janela: Any, pasta: Path) -> list[tuple[Path, int]] | None:
    """Uma imagem por aba, na pasta dada. Devolve o que escreveu, ou `None` se reprovou.

    **Separada de `main` porque o laço dos tamanhos a chama uma vez por tamanho**, e porque as
    duas guardas de piso são sobre o GRUPO de um tamanho: duas telas do mesmo tamanho em bytes são
    a assinatura do defeito de índice, e essa comparação só significa alguma coisa dentro do grupo.
    """
    from PySide6.QtWidgets import QScrollArea

    from refino.desktop.janela.principal import ABAS

    escritos: list[tuple[Path, int]] = []
    # **ITERA `ABAS` E NAO `PAINEIS`, e a diferenca produziu um desfecho de SUCESSO falso.**
    #
    # Com o agrupamento por funcao as abas deixaram de corresponder aos paineis — ha abas com
    # varios paineis, e as tres abas de etapa nao tem painel —, e um indice fora
    # da faixa num `QTabWidget` NAO LEVANTA: ele e ignorado e a aba corrente nao muda. Medido
    # em 03/09/2026, quando eram cinco abas para oito paineis: o arnes gravou oito arquivos,
    # todos acima do piso de 5 KB, com `EXIT=0` — e os quatro ultimos eram byte a byte o
    # MESMO PNG, 91.271 bytes cada.
    #
    # > O veredito era sobre o alvo e o defeito estava no instrumento, com o
    # > desfecho de SUCESSO. O piso de tamanho nao podia ver — quatro copias da mesma tela
    # > passam qualquer assercao sobre bytes.
    for indice, aba in enumerate(ABAS):
        janela.abas.setCurrentIndex(indice)
        app.processEvents()
        # **A conferencia do indice, e ela e o par causal do defeito acima.** `setCurrentIndex`
        # e mudo quando o indice nao existe, entao perguntar de volta e o que separa "mudei de
        # aba" de "pedi para mudar".
        if janela.abas.currentIndex() != indice:
            print(
                f"  a aba {indice} ({aba.titulo}) nao foi selecionada: o widget esta em "
                f"{janela.abas.currentIndex()} de {janela.abas.count()} abas",
                file=sys.stderr,
            )
            return None
        # **NENHUMA ABA PODE PRECISAR DE BARRA HORIZONTAL, e este portao nasceu de um
        # defeito VISTO nesta captura, em 07/09/2026.**
        #
        # Uma mudanca acrescentou o segundo botao do lote ao grupo «Correcao do sinal», e os
        # dois rotulos longos levaram o minimo do formulario a **730 px** contra um viewport
        # de **701**: barra horizontal, e o grupo cortado ao meio na imagem que vai para o
        # guia. Encurtar os rotulos — a lista dos tres foi para a dica — levou o minimo
        # a **645**, e a barra sumiu.
        #
        # **O portao vive AQUI e nao no `pytest`, e a razao e medida.** Sob `offscreen` nao ha
        # fonte, e o mesmo formulario mede **1194 px** de minimo em vez de 645: um portao
        # escrito la reprovaria hoje, com a janela nativa correcta, e estaria a medir a
        # largura da caixinha. O veredito falaria do instrumento.
        for rolagem in janela.abas.currentWidget().findChildren(QScrollArea):
            dentro = rolagem.widget()
            if dentro is None or not rolagem.horizontalScrollBar().isVisible():
                continue
            print(
                f"  a aba {indice} ({aba.titulo}) precisa de BARRA HORIZONTAL: o conteudo "
                f"pede {dentro.minimumSizeHint().width()} px e o viewport tem "
                f"{rolagem.viewport().width()}",
                file=sys.stderr,
            )
            print(
                "  Um rotulo longo de mais empurra o minimo do formulario. Encurte-o e ponha "
                "o resto na dica.",
                file=sys.stderr,
            )
            return None

        alvo = pasta / f"{indice}-{aba.nome_da_captura}.png"
        imagem = janela.grab()
        if not imagem.save(str(alvo)):
            print(f"  FALHA ao gravar {alvo.name}", file=sys.stderr)
            return None
        escritos.append((alvo, alvo.stat().st_size))

    # **Piso contra arquivo vazio, e ele NÃO diz que o texto desenhou.** O escopo vai
    # declarado porque a primeira versão deste arnês passou neste piso com as oito
    # telas em caixinha. Verificar que a fonte desenhou não se mecaniza bem; o que
    # substitui é alguém abrir os oito PNG e olhar, antes de embutir.
    # **DUAS telas com o mesmo tamanho sao a assinatura do defeito de indice**, e o piso
    # sozinho nao a ve. Nao e prova de igualdade — dois PNG diferentes podem coincidir —, mas
    # e o sinal barato que teria apanhado as quatro copias de 91.271 bytes.
    bytes_por_tela = [tamanho for _caminho, tamanho in escritos]
    if len(set(bytes_por_tela)) != len(bytes_por_tela):
        repetidos = sorted({t for t in bytes_por_tela if bytes_por_tela.count(t) > 1})
        print(
            f"FALHA: telas com tamanho repetido: {repetidos} bytes.",
            file=sys.stderr,
        )
        print("  E a assinatura da mesma aba capturada duas vezes.", file=sys.stderr)
        for caminho, tamanho in escritos:
            print(f"  {caminho.name:<34} {tamanho:>8}", file=sys.stderr)
        return None

    magras = [c.name for c, tamanho in escritos if tamanho < 5000]
    for caminho, tamanho in escritos:
        print(f"  {caminho.name:<34} {tamanho:>8} bytes")
    if magras:
        print(f"\nFALHA: {len(magras)} imagem(ns) abaixo de 5 KB: {magras}", file=sys.stderr)
        return None
    return escritos


if __name__ == "__main__":
    raise SystemExit(main())
