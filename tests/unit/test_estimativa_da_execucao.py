"""O palpite de «Executar pipeline» contra o que a execução DE FACTO custou — item 5.

**As palavras dele:** *"o executar pipeline faz o que exatamente? correcao + pre processamento dos
que foram selecionados os parametros? tem como deixar uma estimativa de tempo tambem"*.

## O oráculo é a MEDIÇÃO e não a própria constante

Um teste que recalculasse `n * custo * fator` estaria a afirmar que a aritmética é a aritmética —
verdade vácua, com o desfecho de sucesso. O que estes testes comparam é o palpite contra os
**segundos que `scripts/mede_custo_da_execucao.py` cronometrou ponta a ponta**, nos modos reais.

**A tolerância é declarada e é larga de propósito: ±35%.** O palpite é uma constante única para
quatro matrizes de tamanhos diferentes, medida numa máquina de 12 núcleos lógicos, e a frase que o
profissional lê diz isso. Uma tolerância apertada faria este portão reprovar por ruído de máquina —
e é exatamente o que aconteceu na medição que o alimentou: o `neg_c18` correu com carga e deu
**+54%** no `loess` contra a remedição de máquina livre de 02/09/2026.

## O risco que o plano do item 5 nomeia, e que estes testes fecham

*"`CUSTO_POR_FEATURE_MS` mede o estimador leave-one-QC-out — `n_qc` reajustes por corretor —, que é
o que o botão «Medir qual corretor vence» faz. Esta rota faz **um** ajuste. Reaproveitar o número
diria «3,4 h» para algo que custa minutos: número certo, população errada."*
"""

from __future__ import annotations

import re

import pytest

from refino.interface.fluxo import (
    CUSTO_DE_UM_AJUSTE_MS,
    estimativa_da_execucao,
    estimativa_da_medicao,
)

# **O que a execução DE FACTO custou**, por `scripts/mede_custo_da_execucao.py` em 07/09/2026, no
# regime que o congelado corre. `(features, {corretor: segundos})`.
#
# **O `neg_c18` fica de fora, e a exclusão é declarada e não silenciosa:** ele correu com carga na
# máquina — o `loess` dele deu 20,19 s contra 13,12 s da remedição de máquina livre, **+54%** — e
# usá-lo como oráculo mediria a contenção de CPU daquele instante em vez do custo do corretor.
#
# **Os outros três são os que a máquina correu mais descarregada**, e os números mostram-no: o
# `loess` deles dá 6,58 · 6,75 · 6,49 ms por feature — **4%** de amplitude —, contra os 9,24 ms do
# `neg_c18`.
# **A coluna do `serrf` foi REMEDIDA em 08/09/2026, dentro do binario e com a caixa desticada.**
# Os valores anteriores — 452,62 / 594,71 / 1387,61 — sao de 07/09 e de OUTRA construcao, medidos
# por `scripts/mede_custo_da_execucao.py` em processo. Eles nao estavam errados; o que mudou e que
# o regime serial deixou de ser «o que o executavel faz» e passou a ser «o que a caixa desticada
# faz», e a construcao mudou entre as duas datas.
#
# **O `loess` NAO foi remedido, e a ausencia vai declarada:** ele nao usa `joblib`, entao os dois
# regimes sao o mesmo regime para ele — nao ha par a medir. Os valores continuam os de 07/09.
# **A razao caiu:** o LOESS le a caixa e corre no `loky`
# desde entao. A ausencia continua, e passa a ser outra: o par do LOESS dentro do executavel nao
# foi medido, e os valores de 07/09 sao os da serie.
MEDIDO: dict[str, tuple[int, dict[str, float]]] = {
    "neg_hilic": (2040, {"loess": 13.42, "serrf": 439.6}),
    "pos_c18": (2573, {"loess": 17.37, "serrf": 555.4}),
    "pos_hilic": (6036, {"loess": 39.16, "serrf": 1287.6}),
}

# **±20%, e o número desceu de 35% quando a medição fechou.** A primeira versão pôs 35% por
# prudência, com dois modos medidos. Com os quatro, o palpite do `serrf` acerta entre **1% e 6%**
# — 1,06 · 1,02 · 1,01 · 1,04 —, e o custo por feature varia **9%** sobre matrizes que diferem
# **3x** em tamanho: 221,9 · 241,9 · 231,1 · 229,9 ms.
#
# **As razões de hoje, remedidas em 08/09/2026 dentro do binário, e todas a errar para CIMA:**
# série 1,06 · 1,05 · 1,07 (os três modos da tabela abaixo) e paralelo 1,05 · 1,05 · 1,00 · 1,07
# (os quatro de `MEDIDO_PARALELO`). A folga gasta é de 7 pontos nos 20 — a tolerância continua a
# ser portão e não decoração.
#
# **Uma tolerância que o alvo nunca chega perto de gastar não é portão, é decoração.** Vinte por
# cento deixa folga para o ruído de máquina — o `neg_c18` mostrou +54% sob carga, e é por isso que
# ele fica de fora — sem deixar passar uma constante que caducou por um fator.
TOLERANCIA = 0.20

# ---------------------------------------------------------------------------
# O BRACO PARALELO, DENTRO do executavel congelado
# ---------------------------------------------------------------------------
#
# **A tabela de cima e o regime SERIAL, e ela continua a valer** — e o que se obtem desticando a
# caixa «usar varios nucleos». Esta e o regime em que o executavel corre por omissao, e
# ate essa data ela nao podia existir: o congelado nao paralelizava.
#
# **Populacao:** uma correcao `serrf` por modo, com a matriz inteira, um lote, semente 42, dentro
# do binario, na maquina de 12 nucleos que constroi a entrega. Cada corrida e um lancamento novo do
# executavel, e as tres leituras independentes do `neg_c18` desta sessao — 82,3, 82,5 e 82,6 s —
# concordam em **0,4%**.
#
# **O `pos_hilic` do LOESS NAO entra aqui**, e a ausencia e declarada: o LOESS nao usa `joblib`, e
# um numero paralelo para ele descreveria um mecanismo que nao existe naquele corretor.
# **O mecanismo passou a existir**, e a ausencia fica por outra razao: o braco
# paralelo do LOESS dentro do executavel nao foi medido.
MEDIDO_PARALELO: dict[str, tuple[int, float]] = {
    "neg_hilic": (2039, 77.0),
    "neg_c18": (2184, 82.5),
    "pos_c18": (2572, 102.0),
    "pos_hilic": (6035, 224.5),
}


def _segundos_do_palpite(texto: str) -> float:
    """Os segundos que a frase promete.

    **Levanta quando ela não traz número**, e isso é contrato e não defesa: a frase de ausência é um
    desfecho legítimo da função, e um teste que a esperasse com número tem de reprovar em vez de
    ler zero.
    """
    # **As DUAS irmãs escrevem a duração de formas diferentes**, e o leitor tem de aceitar as duas:
    # a da execução diz `~24 min`, a da medição diz `cerca de 3,4 h`. Um leitor que só conhecesse
    # uma das formas leria a outra como *«não traz duração»* — e o teste que as compara passaria a
    # reprovar por prosa em vez de por número.
    achado = re.search(r"(\d+(?:[.,]\d+)?)\s*(s|min|h)\b", texto)
    if achado is None:
        raise AssertionError(f"a frase não traz duração: {texto!r}")
    valor = float(achado.group(1).replace(",", "."))
    return valor * {"s": 1, "min": 60, "h": 3600}[achado.group(2)]


@pytest.mark.parametrize("modo", sorted(MEDIDO_PARALELO))
def test_o_palpite_paralelo_bate_com_o_cronometro_dentro_do_binario(modo: str) -> None:
    """O palpite do regime PARALELO contra o relogio, nos QUATRO modos.

    **E o irmao do teste serial abaixo, e a existencia dos dois e o conteudo.** Um par de
    constantes que so fosse conferido num regime deixaria o outro livre para caducar — e foi
    exactamente o que aconteceu com o braco paralelo entre 26/08 e 08/09/2026: ele era DERIVADO do
    serial pelo fator, e ninguem o media porque o executavel nao paralelizava.

    **A ancora contra vacuo e o par com o teste serial:** se os dois palpites coincidissem, um dos
    dois regimes nao estaria a chegar a conta.
    `test_u3_o_par_de_palpites_difere_por_mais_de_tres_vezes` afirma a separacao do lado da
    janela; aqui afirma-se o VALOR de cada lado.
    """
    n_features, medido = MEDIDO_PARALELO[modo]
    prometido = _segundos_do_palpite(estimativa_da_execucao(n_features, "serrf", serial=False))
    razao = prometido / medido
    assert abs(razao - 1) <= TOLERANCIA, (
        f"o palpite paralelo do `serrf` no `{modo}` promete {prometido:.0f} s e a execucao dentro "
        f"do binario custou {medido:.0f} s — razao {razao:.2f}, fora de +-{TOLERANCIA:.0%}. Ou a "
        f"constante caducou, ou a maquina que a mediu nao e comparavel com a de hoje"
    )


@pytest.mark.parametrize("modo", sorted(MEDIDO))
@pytest.mark.parametrize("corretor", ("loess", "serrf"))
def test_o_palpite_bate_com_o_que_a_execucao_custou(modo: str, corretor: str) -> None:
    """O palpite contra o cronómetro, nos dois modos medidos em máquina descarregada."""
    n_features, medido = MEDIDO[modo]
    prometido = _segundos_do_palpite(estimativa_da_execucao(n_features, corretor, serial=True))
    real = medido[corretor]
    razao = prometido / real
    assert abs(razao - 1) <= TOLERANCIA, (
        f"o palpite de `{corretor}` no `{modo}` promete {prometido:.0f} s e a execução custou "
        f"{real:.0f} s — razão {razao:.2f}, fora de ±{TOLERANCIA:.0%}. Ou a constante caducou, ou "
        f"a máquina que a mediu não é comparável com a de hoje"
    )


def test_o_palpite_erra_para_cima_no_serrf_e_a_direcao_e_escolhida() -> None:
    """A constante é a do modo mais caro dos medidos, e a preferência vai declarada.

    `ui/fluxo.py` diz por quê, sobre a irmã: *"palpite curto faz o profissional achar que travou"*.
    Um palpite alto assusta; um baixo faz desistir de um trabalho que estava a correr.
    """
    for modo, (n_features, medido) in MEDIDO.items():
        prometido = _segundos_do_palpite(estimativa_da_execucao(n_features, "serrf", serial=True))
        assert prometido >= medido["serrf"] * 0.95, (
            f"o palpite do `serrf` no `{modo}` promete {prometido:.0f} s contra "
            f"{medido['serrf']:.0f} s medidos — ele erra para BAIXO, que é a direção que faz o "
            f"profissional achar que travou"
        )


def test_sem_validacao_e_com_corretor_desconhecido_diz_que_nao_sabe() -> None:
    """Três caminhos de ausência, e nenhum inventa número.

    **O terceiro é declarar o que não foi medido:** um método novo no
    domínio chega à tela antes de alguém o medir, e um palpite sobre ele seria invenção na célula
    que o profissional lê antes de clicar.
    """
    for texto in (
        estimativa_da_execucao(None, "serrf", serial=True),
        estimativa_da_execucao(0, "serrf", serial=True),
        estimativa_da_execucao(2040, "metodo_que_ninguem_mediu", serial=True),
    ):
        assert "Validação" in texto, f"a frase de ausência mudou de forma: {texto!r}"
        with pytest.raises(AssertionError):
            _segundos_do_palpite(texto)

    # **A âncora, na MESMA função:** com features E corretor conhecido ela TRAZ número. Sem
    # isto, uma implementação que devolvesse a frase de ausência sempre passaria nas três acima.
    assert _segundos_do_palpite(estimativa_da_execucao(2040, "serrf", serial=True)) > 0


def test_as_duas_frases_de_repouso_se_distinguem_na_tela() -> None:
    """Em REPOUSO os dois palpites também têm de se distinguir.

    ## A irmã que faltava

    O teste abaixo compara os dois palpites **preenchidos** e exige 3x de diferença. Ninguém
    comparava as duas frases de **repouso**, e elas eram a mesma: *"quanto vai demorar: a
    estimativa aparece depois de «Validação» — é ela que diz quantas features a matriz tem"*,
    diferindo em «sua» e num ponto final.

    E isso importa porque os dois rótulos são **vizinhos**: medido em 12/09/2026, ocupam as linhas
    7 e 8 do `QFormLayout` do quadro «Correção do sinal». Quem abria a janela lia a mesma frase
    duas vezes, e antes da «Validação» é esse o estado — o repouso é o que se vê primeiro.

    > **Clonar a frase de um irmão e mudar uma palavra produz dois rótulos indistinguíveis para
    > quem lê, e idênticos para nenhum teste.** Quem achou foi capturar a janela e olhar.
    """
    execucao = estimativa_da_execucao(None, "serrf", serial=False)
    medicao = estimativa_da_medicao(None, serial=False)

    # Âncora, na MESMA função: as duas SÃO frases de repouso. Sem isto, uma implementação
    # que devolvesse número nas duas passaria a desigualdade abaixo por outro motivo.
    for rotulo, texto in (("execução", execucao), ("medição", medicao)):
        assert "Validação" in texto, f"âncora: a frase de repouso da {rotulo} mudou: {texto!r}"

    assert execucao != medicao, (
        "os dois palpites mostram a MESMA frase em repouso, e são rótulos vizinhos no mesmo "
        f"quadro: {execucao!r}"
    )
    # E distinguir-se não é diferir por pontuação: cada frase nomeia a sua operação, com o nome
    # do botão que está acima dela.
    assert "pipeline" in execucao, f"a frase da execução não nomeia a operação: {execucao!r}"
    assert "corretor vence" in medicao, f"a frase da medição não nomeia a operação: {medicao!r}"


def test_as_quatro_frases_dizem_estimativa_e_nao_palpite() -> None:
    """A palavra é **estimativa** nas quatro células de tempo, e **palpite** só no botão.

    **Eram cinco**: a quinta, `SEM_CORRETOR_DECIDIDO`, dizia que a escolha do corretor estava
    delegada ao Palpite ainda não medido. A janela deixou de delegar, o estado que ela descrevia não
    existe mais na tela, e a constante saiu de `ui/fluxo.py` com a caixa «Automático».

    ## Por que este teste existe, e é a razão que o torna necessário e não decorativo

    **Duas coisas diferentes tinham o mesmo nome na mesma janela:** o botão «Palpite» adivinha
    QUAL corretor, e estas células estimam QUANTO TEMPO.

    > **As duas únicas asserções que existiam sobre este literal eram de AUSÊNCIA** — `"palpite:"
    > not in repouso`, em `tests/test_desktop_janela.py` —, **e elas continuam a passar depois da
    > troca**. Asserção de ausência é satisfeita por vacuidade: nada na suíte afirmava que
    > a palavra nova aparece, e sem isto a renomeação ficaria sem verificação nenhuma.

    **A âncora de não-vacuidade está na mesma função e antes**: as quatro frases não são vazias.

    **O botão «Palpite» NÃO muda, e isso é conferido noutro sítio** — `test_desktop_janela.py`
    afirma `ROTULOS_DA_LEITURA["recomendacao"] == "Palpite"`. A resolução da queixa foi a palavra
    passar a significar UMA coisa, não desaparecer. *Na janela de dez abas o rótulo é o do
    CARTÃO «Palpite», e o botão dele diz «Executar»; a palavra continua a nomear uma coisa só.*
    """
    frases = {
        "repouso da execução": estimativa_da_execucao(None, "serrf", serial=False),
        "execução com valor": estimativa_da_execucao(6035, "serrf", serial=False),
        "repouso da medição": estimativa_da_medicao(None, serial=False),
        "medição com valor": estimativa_da_medicao(6035, serial=False),
    }
    for nome, texto in frases.items():
        assert texto.strip(), f"âncora: a frase «{nome}» veio vazia, e tudo abaixo seria vácuo"

    # **PRESENÇA, que é o que faltava.** Cada frase abre pelo nome da operação, e o nome é o do
    # botão que está acima do rótulo dela.
    esperado = {
        "repouso da execução": "estimativa de executar o pipeline",
        "execução com valor": "estimativa de executar o pipeline com",
        "repouso da medição": "estimativa de medir qual corretor vence",
        "medição com valor": "estimativa de medir qual corretor vence",
    }
    for nome, agulha in esperado.items():
        assert agulha in frases[nome], f"a frase «{nome}» não diz {agulha!r}: {frases[nome]!r}"

    # **E a outra direção, e o ORÁCULO é o RÓTULO e não o texto inteiro.**
    #
    # A primeira versão deste laço afirmava `"palpite" not in texto.lower()` sobre a frase toda, e
    # **reprovou na primeira execução**: `SEM_CORRETOR_DECIDIDO` dizia então *«clique em «Palpite»,
    # na aba «Preparar o estudo»»*, que era o nome do BOTÃO e o uso correto da palavra. O oráculo
    # tinha codificado a forma das instâncias que eu acabara de corrigir, e acusava a que estava
    # certa. *Na janela de dez abas a frase diz «clique em «Executar» no cartão «Palpite»»:
    # «Palpite» é o nome do CARTÃO, e o botão dele diz «Executar».*
    #
    # O rótulo é o que vem antes do primeiro `:` — é ele que nomeia a célula, e é ali que a
    # conflação vive. Uma regressão a `**palpite:**` cai aqui; uma referência ao botão dentro da
    # prosa, não.
    for nome, texto in frases.items():
        rotulo = texto.split(":", 1)[0].lower()
        assert rotulo != texto.lower(), f"âncora: a frase «{nome}» não tem rótulo: {texto!r}"
        assert "palpite" not in rotulo, (
            f"o RÓTULO da frase «{nome}» voltou a dizer «palpite», que é o nome do BOTÃO que "
            f"adivinha qual corretor — as duas coisas voltam a ter o mesmo nome na mesma janela: "
            f"{rotulo!r}"
        )


def test_o_palpite_da_execucao_nao_reusa_o_da_medicao() -> None:
    """O risco que o plano do item 5 nomeia, e ele tem ordem de grandeza.

    A medição corre `n_qc` reajustes por corretor, os **três** corretores; a execução corre **um**
    ajuste de **um**. Se os dois números se aproximassem, um deles estaria a medir a população do
    outro: rótulo certo sobre a quantidade errada.
    """
    n_features = 6036
    execucao = _segundos_do_palpite(estimativa_da_execucao(n_features, "serrf", serial=True))
    medicao = _segundos_do_palpite(estimativa_da_medicao(n_features, serial=True))

    assert medicao > execucao * 3, (
        f"a medição promete {medicao:.0f} s e a execução {execucao:.0f} s — menos de 3x de "
        f"diferença. A medição faz `n_qc` reajustes dos TRÊS corretores e a execução faz UM ajuste "
        f"de um: se os dois números se aproximam, um deles está a medir a população do outro"
    )


def test_o_fator_serial_alcanca_o_serrf_e_mais_nenhum() -> None:
    """O par causal do regime, e a assimetria é medida e não estética.

    O `none` não corre nada, e aplicar-lhe o fator inflaria o palpite por um mecanismo que não
    existe nele. O LOESS lê a caixa, mas o fator dele dentro do executável não foi
    medido, e o custo dele na tabela é o da série.
    """
    n_features = 2040
    for corretor in CUSTO_DE_UM_AJUSTE_MS:
        serial = _segundos_do_palpite(estimativa_da_execucao(n_features, corretor, serial=True))
        paralelo = _segundos_do_palpite(estimativa_da_execucao(n_features, corretor, serial=False))
        if corretor == "serrf":
            assert serial > paralelo * 2, (
                f"o regime não muda o palpite do `serrf`: {serial:.0f} s contra {paralelo:.0f} s. "
                f"O congelado corre em série e custa 5,5x — esconder isso faz o palpite errar por "
                f"minutos"
            )
        else:
            assert serial == paralelo, (
                f"o fator serial alcançou `{corretor}`, que não tem fator medido dentro do "
                f"executável: {serial:.0f} s contra {paralelo:.0f} s"
            )


# **O EIXO DOS LOTES, medido em 07/09/2026 no `neg_c18` — 2.184 features.**
#
# O par que motivou tudo: a tela prometeu `~9 min` e a execução custou **15 min 14 s** (913 s no
# registro do programa, 914 s no relógio da tela), com `n_batches=2`. O palpite não conhecia o eixo.
#
# `(corretor, lotes) -> segundos`, no mesmo regime e na mesma máquina.
MEDIDO_POR_LOTE: dict[tuple[str, int], float] = {
    ("serrf", 1): 514.9,
    ("serrf", 2): 974.2,
    ("loess", 1): 14.7,
    ("loess", 2): 12.1,
}


def test_o_palpite_do_serrf_conta_os_lotes_e_o_do_loess_nao() -> None:
    """A assimetria é MEDIDA e tem mecanismo, e este teste afirma as duas metades.

    O SERRF treina uma floresta por feature **e por lote**: dois lotes quase dobram o custo, e o
    medido é **1,89x**. O LOESS ajusta uma curva sobre os QCs de cada lote — os mesmos pontos,
    repartidos —, e o medido é **0,82x**, mais barato.

    **Aplicar o fator aos dois** inflaria o palpite do LOESS por um mecanismo que ele não tem;
    **não aplicar a nenhum** é o defeito que foi relatado.
    """
    n_features = 2184
    for corretor in ("serrf", "loess"):
        um = _segundos_do_palpite(
            estimativa_da_execucao(n_features, corretor, serial=True, n_lotes=1)
        )
        dois = _segundos_do_palpite(
            estimativa_da_execucao(n_features, corretor, serial=True, n_lotes=2)
        )
        medido = MEDIDO_POR_LOTE[(corretor, 2)] / MEDIDO_POR_LOTE[(corretor, 1)]
        if corretor == "serrf":
            assert dois > um * 1.5, (
                f"o palpite do `serrf` não conta os lotes: {um:.0f} s com um e {dois:.0f} s com "
                f"dois, contra {medido:.2f}x medido. É o defeito relatado — «~9 min» para 15 min"
            )
        else:
            assert dois == um, (
                f"o fator dos lotes alcançou `{corretor}`, que fica MAIS BARATO com dois "
                f"({medido:.2f}x medido): {um:.0f} s contra {dois:.0f} s"
            )


def test_o_palpite_com_dois_lotes_bate_com_o_cronometro() -> None:
    """O palpite contra os 974,2 s que dois lotes de facto custaram — a mesma régua da irmã."""
    prometido = _segundos_do_palpite(estimativa_da_execucao(2184, "serrf", serial=True, n_lotes=2))
    real = MEDIDO_POR_LOTE[("serrf", 2)]
    razao = prometido / real
    assert abs(razao - 1) <= TOLERANCIA, (
        f"o palpite de dois lotes promete {prometido:.0f} s e a correção custou {real:.0f} s — "
        f"razão {razao:.2f}, fora de ±{TOLERANCIA:.0%}"
    )
    # **A âncora de DIREÇÃO, na mesma função:** o fator linear sobrestima 6% sobre o par medido, e
    # sobrestimar é o lado escolhido. Um palpite curto faz o profissional achar que travou — que é
    # exactamente o que aconteceu antes deste eixo existir.
    assert prometido >= real, (
        f"o palpite de dois lotes erra para BAIXO: {prometido:.0f} s contra {real:.0f} s medidos"
    )


def test_o_vazio_e_um_lote_e_nao_multiplica() -> None:
    """`n_lotes` ausente é UM, e não zero nem `None`.

    O `QSpinBox` da janela mostra «1 — um lote só» no mínimo e o payload manda `None`; um palpite
    que multiplicasse por zero devolveria a cauda sozinha, que são 2,5 s para um trabalho de
    minutos.
    """
    padrao = _segundos_do_palpite(estimativa_da_execucao(2184, "serrf", serial=True))
    um = _segundos_do_palpite(estimativa_da_execucao(2184, "serrf", serial=True, n_lotes=1))
    zero = _segundos_do_palpite(estimativa_da_execucao(2184, "serrf", serial=True, n_lotes=0))
    assert padrao == um == zero > 60, f"o vazio não vale um lote: {padrao}, {um}, {zero}"


# ---------------------------------------------------------------------------
# O intervalo por barra, por escolha
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("n_qc", "fator"), [(7, 4.0), (11, 6.0), (21, 11.0)])
@pytest.mark.parametrize("serial", [False, True])
def test_com_o_intervalo_a_estimativa_da_medicao_multiplica_por_n_mais_1_sobre_2(
    n_qc: int, fator: float, serial: bool
) -> None:
    """A estimativa sob o botão cresce pelo mesmo fator que a contagem de ajustes — `(n+1)/2`, a
    razão de `jackknife_cost_ratio` — e a frase diz o fator e o número de QCs de que ele vem."""
    from refino.interface.fluxo import prosa_de_duracao, segundos_da_medicao
    from refino.metrics.crossval import jackknife_cost_ratio

    assert jackknife_cost_ratio(n_qc) == fator
    sem = segundos_da_medicao(6035, serial=serial)
    com = segundos_da_medicao(6035, serial=serial, intervalo=True, n_qc=n_qc)
    assert com == pytest.approx(sem * fator)
    frase = estimativa_da_medicao(6035, serial=serial, intervalo=True, n_qc=n_qc)
    assert prosa_de_duracao(com) in frase, frase
    assert f"{n_qc} QCs" in frase and f"{fator:g} vezes".replace(".", ",") in frase, frase


def test_sem_o_intervalo_a_frase_da_estimativa_e_a_de_antes() -> None:
    """Quem não marca a caixa não vê nada mudar: a frase sem o intervalo é a mesma com ou sem o
    número de QCs, e não fala de intervalo."""
    antes = estimativa_da_medicao(6035, serial=False)
    assert estimativa_da_medicao(6035, serial=False, intervalo=False, n_qc=7) == antes
    assert "intervalo" not in antes


def test_com_o_intervalo_e_sem_saber_os_qcs_a_frase_diz_a_conta_e_nao_inventa_numero() -> None:
    """Sem a validação, o número de QCs não se sabe: a frase diz `(n+1)/2` em vez de um fator."""
    frase = estimativa_da_medicao(6035, serial=False, intervalo=True, n_qc=None)
    assert "(n+1)/2" in frase, frase
    assert _segundos_do_palpite(frase) == pytest.approx(
        _segundos_do_palpite(estimativa_da_medicao(6035, serial=False))
    )
