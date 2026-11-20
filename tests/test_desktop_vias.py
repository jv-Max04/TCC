"""O painel das vias metabólicas na janela nativa.

**O oráculo é o WIDGET e não o valor de retorno.** `PainelDeVias.linhas()` lê as
células da `QTableWidget`; um relatório guardado num atributo e nunca desenhado é um painel
em branco, e o dicionário de retorno não sabe disso.

**A `calculadora` é injetada nos testes e o PADRÃO é a função de produção.** Um painel cujo
padrão fosse o duplo passaria em tudo aqui e não calcularia nada
na máquina do usuário — sucesso aparente. `test_o_padrao_do_painel_e_a_funcao_de_producao`
afirma o padrão, e é ele que impede a injeção de virar a porta de saída.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
from PySide6.QtWidgets import QLabel

from refino.desktop.janela.vias import (
    COLUNAS,
    SEM_EXECUCAO,
    SEM_PAR_DE_GRUPOS,
    PainelDeVias,
    numero,
    rotulos_da_busca,
    texto_do_recorte,
)
from refino.desktop.recorte import FUNCIONALIDADE_EM_BETA
from refino.pathways.analise import (
    RecorteDaEntrada,
    RelatorioDeVias,
    ViaPredita,
    prediz_vias,
)

if TYPE_CHECKING:  # pragma: no cover - só para o mypy
    from pathlib import Path

    from pytestqt.qtbot import QtBot


def _via(nome: str, p: float, sobreposicao: int = 2) -> ViaPredita:
    return ViaPredita(
        nome=nome,
        sobreposicao=sobreposicao,
        detectados_na_via=9,
        pvalue=p,
        pvalue_fisher=p / 2,
        compostos_candidatos=("C00001",),
        nomes_candidatos=("Água",),
    )


def _relatorio(*, vias: tuple[ViaPredita, ...] = (), **recorte: Any) -> RelatorioDeVias:
    campos = {
        "recebidas": 100,
        "ions": 80,
        "massas_neutras_incluidas": 20,
        "massas_neutras_excluidas": 0,
        "nao_interpretadas": 0,
        "sem_estatistica": 0,
        "fora_da_faixa_de_massa": 0,
        "analisadas": 100,
        "features_significativas": 12,
        "corte_de_significancia": 0.05,
    }
    campos.update(recorte)
    return RelatorioDeVias(
        vias=vias,
        recorte=RecorteDaEntrada(**campos),
        modo_de_ionizacao="positive",
        politica_de_massa_neutra="primary_ion",
        busca_de_adutos="default",
        adutos_procurados=8,
        ppm=10.0,
        permutacoes=100,
        seed=0,
        versao_do_modelo="MFN_1.10.4",
    )


def _corpo(**campos: Any) -> dict[str, Any]:
    base = {
        "features": ["1.0_100.0m/z", "2.0_200.0n"],
        "pvalue": [0.01, 0.9],
        "statistic": [3.2, -0.4],
    }
    base.update(campos)
    return {"volcano": base}


def _clica(qtbot: QtBot, painel: PainelDeVias) -> None:
    """Clica e espera o worker terminar, por SINAL e nao por relogio.

    **Nasce de o calculo ter saido da thread da GUI.** Antes disso
    `botao.click()` era sincrono e a linha seguinte ja podia ler a tabela; agora o clique
    so agenda. Esperar por `qtbot.wait(ms)` seria medir a maquina — o sinal mede o
    desfecho.

    Espera QUALQUER um dos dois desfechos, e por isso NAO usa `waitSignals`: aquele espera
    por TODOS os sinais da lista, e um cálculo que termina bem nunca emite `falhou` — a
    espera estouraria sempre. O sentinela é alimentado pelos dois e `waitUntil` para no
    primeiro que chegar.
    """
    chegou: list[str] = []
    painel._sinais.concluido.connect(lambda *_a: chegou.append("concluido"))
    painel._sinais.falhou.connect(lambda *_a: chegou.append("falhou"))
    painel.botao.click()
    qtbot.waitUntil(lambda: bool(chegou), timeout=20000)


# ---------------------------------------------------------------------------
# O estado antes de haver execução
# ---------------------------------------------------------------------------


def test_antes_de_qualquer_execucao_o_painel_diz_que_nao_ha_execucao(qtbot: QtBot) -> None:
    """*"Não há execução"* e *"nenhuma via"* são fatos opostos.

    O oráculo é o rótulo visível, e o botão desabilitado: um painel que dissesse "nenhuma
    via enriquecida" antes de calcular afirmaria um resultado que ninguém mediu.
    """
    painel = PainelDeVias()
    qtbot.addWidget(painel)
    assert painel.recorte.text() == SEM_EXECUCAO
    assert not painel.botao.isEnabled()
    assert painel.linhas() == ()


def test_o_botao_so_liga_quando_ha_corpo(qtbot: QtBot) -> None:
    """Par causal do botão: sem corpo desligado, com corpo ligado.

    Sem a segunda metade, um botão permanentemente desabilitado passaria no teste acima e o
    painel nunca calcularia nada.
    """
    painel = PainelDeVias()
    qtbot.addWidget(painel)
    assert not painel.botao.isEnabled()

    painel.recebe_volcano(_corpo())
    assert painel.botao.isEnabled()

    # E volta a desligar quando a resposta seguinte não traz volcano — aqui um corpo sem a
    # chave, que é o do progresso ou o do erro. Sem isto, o painel calcularia sobre o corpo
    # ANTERIOR sob o rótulo da execução corrente, que é a troca que a chave de invalidação
    # existe para impedir. O corpo CONCLUÍDO sem par de grupos tem teste próprio abaixo.
    painel.recebe_volcano({})
    assert not painel.botao.isEnabled()
    assert painel.recorte.text() == SEM_EXECUCAO


def test_execucao_CONCLUIDA_sem_par_de_grupos_nao_diz_que_nao_ha_execucao(  # noqa: N802
    qtbot: QtBot,
) -> None:
    """«Não há execução» e «a execução não tem par» são fatos distintos.

    A rota devolve `volcano = None` quando `group_a`/`group_b` não foram declarados, e o campo
    EXISTE no corpo — é a presença da chave que diz que a execução concluiu. **Visto reprovar
    antes do conserto**: o recorte mostrava `SEM_EXECUCAO` para este corpo, a dizer que não
    havia execução a quem acabou de a ver concluir.

    O par causal está no mesmo teste: o corpo do progresso — sem a chave — continua a dizer
    `SEM_EXECUCAO`, e as duas frases são afirmadas diferentes (senão a separação seria vácua).
    """
    assert SEM_PAR_DE_GRUPOS != SEM_EXECUCAO, "âncora: as duas frases colapsaram numa só"

    painel = PainelDeVias()
    qtbot.addWidget(painel)
    painel.recebe_volcano(_corpo())
    assert painel.botao.isEnabled(), "âncora: o corpo com volcano tinha de ligar o botão"

    painel.recebe_volcano({"volcano": None, "chave_de_invalidacao": "execucao-sem-par"})
    assert not painel.botao.isEnabled()
    assert painel.recorte.text() == SEM_PAR_DE_GRUPOS
    assert "Grupo A do contraste" in painel.recorte.text(), (
        "a frase não nomeia o campo a preencher, e a ação fica por adivinhar"
    )

    painel.recebe_volcano({"estado": "em_curso"})
    assert not painel.botao.isEnabled()
    assert painel.recorte.text() == SEM_EXECUCAO


# ---------------------------------------------------------------------------
# O padrão da calculadora
# ---------------------------------------------------------------------------


def test_o_padrao_do_painel_e_a_funcao_de_producao(qtbot: QtBot) -> None:
    """O painel construído sem argumento calcula de VERDADE.

    Sem esta asserção, a injeção usada pelos outros testes poderia ser o padrão, e o painel
    passaria em tudo aqui mostrando uma tabela vazia na máquina do usuário.
    """
    painel = PainelDeVias()
    qtbot.addWidget(painel)
    assert painel._calcula is prediz_vias


# ---------------------------------------------------------------------------
# O que o painel DESENHA
# ---------------------------------------------------------------------------


def test_as_vias_aparecem_na_tabela_na_ordem_do_relatorio(qtbot: QtBot) -> None:
    relatorio = _relatorio(vias=(_via("Ciclo do ácido cítrico", 0.001), _via("Glicólise", 0.04)))
    painel = PainelDeVias(calculadora=lambda *a, **k: relatorio)
    qtbot.addWidget(painel)
    painel.recebe_volcano(_corpo())
    _clica(qtbot, painel)

    linhas = painel.linhas()
    assert len(linhas) == 2
    assert [linha[0] for linha in linhas] == ["Ciclo do ácido cítrico", "Glicólise"]
    assert linhas[0][3] == "0,001"


def test_a_tela_inteira_usa_a_MESMA_virgula_decimal(qtbot: QtBot) -> None:  # noqa: N802
    """O defeito que só a leitura da aba montada achou, agora com portão.

    O recorte imprimia `p < 0.2` a quatro linhas do rodapé, que diz `p = 0,05` — dois números
    da mesma natureza, com separadores diferentes, na mesma tela. Nenhum dos vinte testes de
    peça o pegava, porque cada um lia o seu rótulo.

    **A asserção é sobre a UNIÃO dos três textos**, que é o que o olho vê de uma vez. Ancorada
    em não-vacuidade: sem números na tela, a varredura passaria por não ter o que reprovar.
    """
    relatorio = _relatorio(vias=(_via("A", 0.0123),), corte_de_significancia=0.2)
    painel = PainelDeVias(calculadora=lambda *a, **k: relatorio)
    qtbot.addWidget(painel)
    painel.recebe_volcano(_corpo())
    _clica(qtbot, painel)

    tudo = " ".join(
        [
            painel.recorte.text(),
            painel.rodape.text(),
            *(c for linha in painel.linhas() for c in linha),
        ]
    )
    assert any(char.isdigit() for char in tudo), "âncora: não há número na tela para conferir"

    # Um ponto entre dois dígitos é um separador decimal à inglesa. `MFN_1.10.4` é versão e
    # não número — vai isento por nome, e não por o padrão não o alcançar.
    import re

    pontos = [m.group(0) for m in re.finditer(r"\d+\.\d+", tudo.replace("MFN_1.10.4", ""))]
    assert not pontos, (
        f"números com ponto decimal numa tela cuja prosa usa vírgula: {pontos}. "
        "A mesma grandeza com dois separadores na mesma tela lê-se como duas grandezas"
    )


def test_a_tabela_e_reconstruida_e_nao_acrescentada(qtbot: QtBot) -> None:
    """Dois cálculos seguidos não empilham resultados.

    Um painel que apendasse mostraria o resultado de uma polaridade junto do de outra, e as
    duas não são comparáveis linha a linha — a tabela pareceria ter o dobro da evidência.
    """
    relatorios = iter(
        [
            _relatorio(vias=(_via("A", 0.01), _via("B", 0.02))),
            _relatorio(vias=(_via("C", 0.03),)),
        ]
    )
    painel = PainelDeVias(calculadora=lambda *a, **k: next(relatorios))
    qtbot.addWidget(painel)
    painel.recebe_volcano(_corpo())

    _clica(qtbot, painel)
    assert len(painel.linhas()) == 2
    _clica(qtbot, painel)
    assert [linha[0] for linha in painel.linhas()] == ["C"]


def test_a_tabela_tem_as_colunas_declaradas(qtbot: QtBot) -> None:
    painel = PainelDeVias()
    qtbot.addWidget(painel)
    assert painel.tabela.columnCount() == len(COLUNAS)
    cabecalhos = [
        painel.tabela.horizontalHeaderItem(i).text() for i in range(painel.tabela.columnCount())
    ]
    assert cabecalhos == [titulo for _chave, titulo in COLUNAS]


# ---------------------------------------------------------------------------
# O recorte — a segunda coisa que o painel tem de dizer
# ---------------------------------------------------------------------------


def test_o_recorte_aparece_com_a_contagem_e_o_corte(qtbot: QtBot) -> None:
    """Sem o recorte na tela, a tabela convida à conclusão de que tudo foi analisado.

    O corte importa tanto quanto a contagem: a mesma tabela sob `p < 0,2` não afirma o que
    ela afirmaria sob `p < 0,001`, e o corte é escolhido pela biblioteca quando o sinal é
    fraco.
    """
    relatorio = _relatorio(
        vias=(_via("A", 0.01),),
        analisadas=91,
        recebidas=100,
        # **O fixture FECHA a conta**, como o de produção fecha: 91 + 9 = 100. A primeira
        # versão punha 91 de 100 com as quatro gavetas a zero, e o painel dizia "nenhuma
        # feature ficou de fora" — um arnês incoerente que ninguém leria como defeito.
        sem_estatistica=9,
        corte_de_significancia=0.2,
    )
    painel = PainelDeVias(calculadora=lambda *a, **k: relatorio)
    qtbot.addWidget(painel)
    painel.recebe_volcano(_corpo())
    _clica(qtbot, painel)

    texto = painel.recorte.text()
    assert "91 de 100" in texto
    assert "0,2" in texto


def test_o_recorte_nomeia_a_causa_de_cada_saida(qtbot: QtBot) -> None:
    """Cada gaveta de exclusão aparece com a contagem, e nenhuma some.

    A planta cai sobre um recorte em que as quatro causas são NÃO NULAS — sobre um recorte
    limpo, um `texto_do_recorte` que ignorasse as causas passaria na mesma.
    """
    relatorio = _relatorio(
        analisadas=60,
        recebidas=100,
        massas_neutras_excluidas=15,
        nao_interpretadas=10,
        sem_estatistica=8,
        fora_da_faixa_de_massa=7,
    )
    texto = texto_do_recorte(relatorio)
    # **A asserção é sobre a FRASE e não sobre o número — corrigido em 18/09/2026.**
    #
    # Ela procurava só `"15"`, `"10"`, `"8"` e `"7"` no texto todo, e a revisão de código
    # reproduziu o desfecho: **apagando `nao_interpretadas` e `sem_estatistica` de
    # `texto_do_recorte`, o teste continuava verde**. Os números colidiam com outros do
    # mesmo texto — `10` vinha de «100 permutações» e `8` de «8 formas iônicas».
    #
    # Duas das quatro causas podiam sumir da tela sem nada reprovar, no teste que existe
    # para garantir que nenhuma some. Era uma planta que não separava, na verificação.
    for quantas, porque in (
        (15, "massas neutras deixadas de fora por escolha sua"),
        (10, "nomes que não têm a forma `tempo_massa`"),
        (8, "sem p-valor utilizável"),
        (7, "fora da faixa de 50 a 2.000 Da"),
    ):
        assert f"{quantas} {porque}" in texto, (
            f"a causa «{porque}» não aparece no recorte com a contagem {quantas}: {texto}"
        )


def test_recorte_sem_exclusao_diz_que_nada_ficou_de_fora(qtbot: QtBot) -> None:
    """A direção oposta: sem exclusões, o texto AFIRMA isso em vez de calar.

    Um texto que simplesmente omitisse a linha faria "nada ficou de fora" e "esqueci de
    contar" ficarem indistinguíveis.
    """
    texto = texto_do_recorte(_relatorio())
    assert "Nenhuma feature ficou de fora" in texto


def test_o_rodape_declara_que_os_compostos_nao_sao_identificacoes(qtbot: QtBot) -> None:
    """A linha que impede a leitura que o escopo do projeto proíbe.

    Identificar metabólito a partir de `RT_mz` está fora de escopo por decisão registrada, e
    uma tabela com nomes de composto ao lado de vias convida exactamente a essa leitura.
    """
    painel = PainelDeVias(calculadora=lambda *a, **k: _relatorio(vias=(_via("A", 0.01),)))
    qtbot.addWidget(painel)
    painel.recebe_volcano(_corpo())
    _clica(qtbot, painel)
    assert "candidatos" in painel.rodape.text()
    assert "não identifica" in painel.rodape.text()


# ---------------------------------------------------------------------------
# Os parâmetros chegam ao núcleo
# ---------------------------------------------------------------------------


def test_os_combos_chegam_ao_calculo(qtbot: QtBot) -> None:
    """Par causal dos dois combos: trocar a escolha TEM de mudar o argumento.

    Sem ele, dois combos decorativos passariam em todo o resto — o painel calcularia sempre
    com o padrão, e o usuário veria a escolha na tela sem efeito nenhum.
    """
    vistos: list[dict[str, Any]] = []

    def espia(*_a: Any, **k: Any) -> RelatorioDeVias:
        vistos.append(k)
        return _relatorio()

    painel = PainelDeVias(calculadora=espia)
    qtbot.addWidget(painel)
    painel.recebe_volcano(_corpo())

    painel.polaridade.setCurrentIndex(painel.polaridade.findData("negative"))
    painel.politica.setCurrentIndex(painel.politica.findData("exclude"))
    painel.busca.setCurrentIndex(painel.busca.findData("broad"))
    _clica(qtbot, painel)

    assert vistos[-1]["modo_de_ionizacao"] == "negative"
    assert vistos[-1]["politica_de_massa_neutra"] == "exclude"
    assert vistos[-1]["busca_de_adutos"] == "broad"

    painel.polaridade.setCurrentIndex(painel.polaridade.findData("positive"))
    painel.politica.setCurrentIndex(painel.politica.findData("primary_ion"))
    painel.busca.setCurrentIndex(painel.busca.findData("default"))
    _clica(qtbot, painel)

    assert vistos[-1]["modo_de_ionizacao"] == "positive"
    assert vistos[-1]["politica_de_massa_neutra"] == "primary_ion"
    assert vistos[-1]["busca_de_adutos"] == "default"


@pytest.mark.parametrize(
    ("modo", "esperado"),
    [
        ("pos_hilic", "positive"),
        ("pos_c18", "positive"),
        ("neg_hilic", "negative"),
        ("neg_c18", "negative"),
    ],
)
def test_a_polaridade_e_sugerida_pelo_modo(qtbot: QtBot, modo: str, esperado: str) -> None:
    painel = PainelDeVias()
    qtbot.addWidget(painel)
    painel.sugere_polaridade(modo)
    assert painel.polaridade.currentData() == esperado


@pytest.mark.parametrize("modo", ["", None, "lipidomica_x", "HILIC"])
def test_modo_irreconhecivel_nao_muda_a_polaridade(qtbot: QtBot, modo: str | None) -> None:
    """Sugestão que não sabe não escolhe — cair num padrão seria escolher errado em silêncio.

    O sintoma de uma polaridade errada não é erro: é *"menos vias"*, que se lê como resultado
    biológico.
    """
    painel = PainelDeVias()
    qtbot.addWidget(painel)
    painel.polaridade.setCurrentIndex(painel.polaridade.findData("negative"))
    painel.sugere_polaridade(modo)
    assert painel.polaridade.currentData() == "negative"


# ---------------------------------------------------------------------------
# O rótulo da busca ampla por polaridade
# ---------------------------------------------------------------------------


def _rotulos_do_combo(painel: PainelDeVias) -> dict[str, str]:
    """`{valor: texto visível}` do combo da busca — o que o usuário LÊ, e não a constante."""
    return {painel.busca.itemData(i): painel.busca.itemText(i) for i in range(painel.busca.count())}


def test_no_NEGATIVO_a_busca_ampla_nao_afirma_concordancia(qtbot: QtBot) -> None:  # noqa: N802
    """A concordância com o MetaboAnalyst foi medida só no `pos_hilic`.

    **Visto reprovar antes do conserto**: o rótulo era o mesmo nas duas polaridades, e no
    negativo dizia *«concorda com o MetaboAnalyst»* — uma medição que nunca aconteceu, sobre
    uma opção que ali procura a mesma lista da estreita.

    O oráculo é o texto do WIDGET, e a troca passa pelo caminho real — `sugere_polaridade`,
    que é o que a janela chama ao mudar o modo do estudo. O par causal é a volta ao positivo,
    onde a afirmação é a medida e fica.
    """
    painel = PainelDeVias()
    qtbot.addWidget(painel)

    painel.sugere_polaridade("pos_hilic")
    assert "concorda com o MetaboAnalyst" in _rotulos_do_combo(painel)["broad"], (
        "âncora: no positivo o rótulo afirma a concordância medida em DV42"
    )

    painel.sugere_polaridade("neg_hilic")
    negativo = _rotulos_do_combo(painel)
    assert "concorda" not in negativo["broad"], (
        f"no modo negativo o rótulo afirma uma concordância nunca medida: {negativo['broad']!r}"
    )
    assert "não medida" in negativo["broad"]
    assert "mesma lista da estreita" in negativo["broad"]

    painel.sugere_polaridade("pos_c18")
    assert "concorda com o MetaboAnalyst" in _rotulos_do_combo(painel)["broad"]


def test_trocar_a_polaridade_NAO_troca_a_busca_escolhida(qtbot: QtBot) -> None:  # noqa: N802
    """Só o TEXTO muda; o valor marcado é o do usuário e atravessa a troca.

    Reescrever o combo apagando e repovoando o poria de volta no primeiro item — a ampla —, e
    quem tinha escolhido a estreita passaria a calcular com a outra sem ter mexido nela.
    """
    painel = PainelDeVias()
    qtbot.addWidget(painel)
    painel.busca.setCurrentIndex(painel.busca.findData("default"))

    painel.sugere_polaridade("neg_hilic")
    assert painel.busca.currentData() == "default"
    painel.sugere_polaridade("pos_hilic")
    assert painel.busca.currentData() == "default"


def test_as_duas_polaridades_oferecem_os_MESMOS_valores() -> None:  # noqa: N802
    """O rótulo depende da polaridade; o cálculo, não — os mesmos valores, na mesma ordem.

    É o que deixa `_rotula_busca` achar cada item por `findData` sem guarda, e o que garante
    que a ampla continua a ser o padrão (o primeiro item) nas duas.
    """
    positivo = [valor for valor, _rotulo in rotulos_da_busca("positive")]
    negativo = [valor for valor, _rotulo in rotulos_da_busca("negative")]
    assert positivo == negativo == ["broad", "default"]


def test_polaridade_desconhecida_cai_no_rotulo_que_NAO_afirma() -> None:  # noqa: N802
    """A direção do padrão é a que não inventa medição.

    Cair no positivo afirmaria concordância com o MetaboAnalyst sobre um caso que ninguém mediu.
    """
    assert rotulos_da_busca(None) == rotulos_da_busca("negative")
    assert rotulos_da_busca("POSITIVE") == rotulos_da_busca("negative")
    assert rotulos_da_busca("positive") != rotulos_da_busca("negative")


# ---------------------------------------------------------------------------
# A falha
# ---------------------------------------------------------------------------


def test_entrada_sem_feature_analisavel_mostra_a_causa_e_nao_uma_tabela_vazia(
    qtbot: QtBot,
) -> None:
    """A mensagem do núcleo chega INTEIRA à tela.

    Ela nomeia as quatro causas com a contagem de cada uma; trocá-la por "falhou" trocaria um
    diagnóstico por um rótulo. E a tabela fica **vazia com explicação**, em vez de vazia e
    muda — que se leria como "nenhuma via enriquecida".
    """

    def recusa(*_a: Any, **_k: Any) -> RelatorioDeVias:
        raise ValueError("nenhuma feature sobrou para análise de vias. Causas separadas: 7 nomes")

    painel = PainelDeVias(calculadora=recusa)
    qtbot.addWidget(painel)
    painel.recebe_volcano(_corpo())
    _clica(qtbot, painel)

    assert painel.linhas() == ()
    assert "7 nomes" in painel.recorte.text()
    # O botão volta a ficar utilizável: a falha é da ENTRADA, e o usuário pode trocar os
    # combos e tentar de novo. Um botão que ficasse morto obrigaria a refazer a execução.
    assert painel.botao.isEnabled()


def test_pvalor_nulo_do_json_nao_rebenta_o_painel(qtbot: QtBot) -> None:
    """`None` do JSON vira `nan`, e a POSIÇÃO sobrevive.

    O esquema manda `float | None` porque as listas indexam as mesmas features; converter
    encurtando a lista desalinharia o p de uma feature com a massa de outra — defeito que não
    levanta e que produz vias plausíveis e erradas.
    """
    vistos: list[tuple[Any, ...]] = []

    def espia(*a: Any, **_k: Any) -> RelatorioDeVias:
        vistos.append(a)
        return _relatorio()

    painel = PainelDeVias(calculadora=espia)
    qtbot.addWidget(painel)
    painel.recebe_volcano(_corpo(pvalue=[0.01, None], statistic=[3.2, None]))
    _clica(qtbot, painel)

    nomes, pvalores, estatisticas = vistos[-1]
    assert len(nomes) == len(pvalores) == len(estatisticas) == 2
    assert pvalores[0] == pytest.approx(0.01)
    assert pvalores[1] != pvalores[1]  # nan


# ---------------------------------------------------------------------------
# A LIGAÇÃO, e não o método — a lição de um defeito que um portão apanhou
# ---------------------------------------------------------------------------


class _TransporteMudo:
    """Um `Transporte` cujas chamadas levantam: este teste não dispara rota nenhuma."""

    def post(self, url: str, **kwargs: object) -> object:
        raise AssertionError(f"este teste não dispara rota, e alguém chamou POST {url}")

    def get(self, url: str, **kwargs: object) -> object:
        raise AssertionError(f"este teste não dispara rota, e alguém chamou GET {url}")


def test_a_janela_LIGA_o_modo_do_estudo_a_polaridade(qtbot: QtBot) -> None:  # noqa: N802
    """Par causal da LIGAÇÃO: trocar o modo na janela TEM de mudar a polaridade do painel.

    **Este teste existe porque o método passava e a ligação não existia.**
    `test_a_polaridade_e_sugerida_pelo_modo` chama `sugere_polaridade` diretamente e sempre passou;
    a `JanelaPrincipal` nunca a chamava, e o usuário recebia polaridade positiva em modo negativo —
    o que desloca toda massa neutra em dois prótons. Quem o apanhou foi
    `test_u21_todo_metodo_publico_de_painel_tem_uso_em_producao_ou_e_declarado`, cuja mensagem
    nomeia os dois precedentes da mesma classe.

    > **Testar o método não é testar a ligação**, e a diferença não aparece como erro: aparece como
    > *"menos vias"*, que se lê como resultado biológico.
    """
    from refino.desktop.janela.principal import JanelaPrincipal

    janela = JanelaPrincipal(_TransporteMudo())
    qtbot.addWidget(janela)

    modos = [janela.arquivos.modo.itemText(i) for i in range(janela.arquivos.modo.count())]
    positivos = [m for m in modos if m.lower().startswith("pos")]
    negativos = [m for m in modos if m.lower().startswith("neg")]
    assert positivos and negativos, (
        f"âncora: o combo de modos não tem as duas polaridades ({modos}), e o par causal abaixo "
        "compararia duas vezes a mesma coisa"
    )

    janela.arquivos.modo.setCurrentText(negativos[0])
    assert janela.vias.polaridade.currentData() == "negative"

    janela.arquivos.modo.setCurrentText(positivos[0])
    assert janela.vias.polaridade.currentData() == "positive"


def test_a_polaridade_ja_esta_certa_SEM_o_usuario_tocar_no_combo(  # noqa: N802
    qtbot: QtBot, tmp_path: Path
) -> None:
    """A chamada inicial, e ela não é redundância da ligação acima.

    Só com o `connect`, a polaridade acertaria apenas depois de o usuário **trocar** o modo. Quem
    abrisse o programa e corresse o modo que já está selecionado receberia a polaridade padrão do
    combo — certa por acaso quando o padrão é positivo, e errada em silêncio se o padrão mudar.

    **O combo do modo nasce vazio, e o modo chega pela dedução** quando os arquivos são
    escolhidos. A pergunta deste teste passou a ser a mesma com outro gatilho: o modo deduzido de
    `measurements_neg_c18.csv`, sem ninguém tocar no combo, leva a polaridade a negativa — e a
    âncora é que ela nasce positiva, senão a asserção seria satisfeita pelo padrão.
    """
    from refino.desktop.janela.arquivos import ROTULO_DA_LISTA, ROTULO_DAS_MEDIDAS
    from refino.desktop.janela.principal import JanelaPrincipal

    medidas = tmp_path / "measurements_neg_c18.csv"
    lista = tmp_path / "lista.csv"
    medidas.write_bytes(b"NAME,a\nf1,1\n")
    lista.write_bytes(b"ID,inj_order_name\n1,a\n")
    janela = JanelaPrincipal(
        _TransporteMudo(),
        escolhedor={ROTULO_DAS_MEDIDAS: medidas, ROTULO_DA_LISTA: lista}.get,
    )
    qtbot.addWidget(janela)

    assert janela.arquivos.modo.currentText() == "", "âncora: o modo não nasce vazio"
    assert janela.vias.polaridade.currentData() == "positive", (
        "âncora: a polaridade não nasce positiva"
    )
    janela.arquivos.escolhe_medidas()
    assert janela.arquivos.modo.currentText() == "neg_c18"
    assert janela.vias.polaridade.currentData() == "negative"


def test_o_combo_da_busca_NASCE_na_ampla(qtbot: QtBot) -> None:  # noqa: N802
    """O padrão do combo é a busca ampla, e é o que corre sem ninguém mexer.

    **O oráculo é `currentData()` e não a ordem da constante**, porque é o valor corrente
    que vai ao cálculo. Um `setCurrentIndex` posterior noutro ponto do construtor mudaria
    o que o usuário recebe sem mexer na ordem da lista, e uma asserção sobre `_BUSCAS`
    passaria na mesma.
    """
    painel = PainelDeVias()
    qtbot.addWidget(painel)
    assert painel.busca.currentData() == "broad"


def test_o_padrao_do_combo_CHEGA_ao_calculo(qtbot: QtBot) -> None:  # noqa: N802
    """E o par causal: sem tocar em nada, o argumento que sai é o da busca ampla.

    Testar só `currentData()` deixaria passar um painel que lesse o combo errado — ou que
    não lesse nenhum e passasse um literal.
    """
    vistos: list[dict[str, Any]] = []

    def espia(*_a: Any, **k: Any) -> RelatorioDeVias:
        vistos.append(k)
        return _relatorio()

    painel = PainelDeVias(calculadora=espia)
    qtbot.addWidget(painel)
    painel.recebe_volcano(_corpo())
    _clica(qtbot, painel)

    assert vistos[-1]["busca_de_adutos"] == "broad"


# ---------------------------------------------------------------------------
# Os caminhos que a revisão de código abriu — falha larga, invalidação, worker
# ---------------------------------------------------------------------------


def test_falha_NAO_ValueError_limpa_a_tela_em_vez_de_a_deixar_velha(qtbot: QtBot) -> None:  # noqa: N802
    """O achado de quatro revisores independentes: a tabela anterior ficava na tela.

    Antes de 18/09/2026 só `ValueError` era tratado. Qualquer outra exceção saía do slot,
    o PySide6 imprimia o traceback em stderr — invisível num `.exe` sem console —, o botão
    voltava a ficar ativo, e **a tabela da corrida anterior continuava à vista**. Trocar de
    polaridade, clicar, não receber erro e ler o resultado da polaridade antiga é o pior
    desfecho que esta aba pode ter.

    Reproduzível com uma única feature analisável, que faz a biblioteca levantar
    `IndexError`, e com o `RuntimeError` das próprias guardas do núcleo.
    """
    relatorios = iter([_relatorio(vias=(_via("A", 0.01),))])

    def as_vezes(*_a: Any, **_k: Any) -> RelatorioDeVias:
        try:
            return next(relatorios)
        except StopIteration:
            raise RuntimeError("a faixa de massa do mummichog mudou") from None

    painel = PainelDeVias(calculadora=as_vezes)
    qtbot.addWidget(painel)
    painel.recebe_volcano(_corpo())

    with qtbot.waitSignal(painel._sinais.concluido, timeout=20000):
        painel.botao.click()
    assert painel.linhas(), "âncora: a primeira corrida tinha de desenhar alguma coisa"

    with qtbot.waitSignal(painel._sinais.falhou, timeout=20000):
        painel.botao.click()
    qtbot.waitUntil(lambda: painel.botao.isEnabled(), timeout=5000)

    assert painel.linhas() == (), "a tabela da corrida ANTERIOR ficou na tela após a falha"
    assert painel.rodape.text() == ""
    assert "inesperada" in painel.recorte.text()
    assert "RuntimeError" in painel.recorte.text(), (
        "a mensagem não diz de que tipo foi a falha, e quem reporta o defeito fica sem o nome"
    )


def test_a_mensagem_do_nucleo_so_chega_inteira_para_a_excecao_PROPRIA(qtbot: QtBot) -> None:  # noqa: N802
    """`EntradaSemFeatureAnalisavelError` fala; as outras são declaradas inesperadas.

    A tela promete que a mensagem *nomeia as quatro causas*. Essa promessa só vale para a
    exceção própria — um `ValueError` do `scipy` ou um `InvalidLabelError` (que herda de
    `ValueError`) apareceriam sob o mesmo cabeçalho como se fossem o diagnóstico.
    """
    from refino.pathways.analise import EntradaSemFeatureAnalisavelError

    def recusa(*_a: Any, **_k: Any) -> RelatorioDeVias:
        raise EntradaSemFeatureAnalisavelError("nenhuma feature sobrou. Causas: 7 nomes")

    painel = PainelDeVias(calculadora=recusa)
    qtbot.addWidget(painel)
    painel.recebe_volcano(_corpo())
    with qtbot.waitSignal(painel._sinais.falhou, timeout=20000):
        painel.botao.click()

    assert "7 nomes" in painel.recorte.text()
    assert "inesperada" not in painel.recorte.text()


def test_execucao_nova_INVALIDA_a_tabela_anterior(qtbot: QtBot) -> None:  # noqa: N802
    """Par causal da invalidação: o corpo novo limpa a tela ANTES de qualquer cálculo.

    Sem isto, uma execução nova deixava na tela a tabela da execução ANTERIOR sob o recorte
    dela — rótulo certo sobre conteúdo errado, que é o que a chave de invalidação do resto
    da janela existe para impedir.
    """
    painel = PainelDeVias(calculadora=lambda *a, **k: _relatorio(vias=(_via("A", 0.01),)))
    qtbot.addWidget(painel)
    painel.recebe_volcano(_corpo())
    with qtbot.waitSignal(painel._sinais.concluido, timeout=20000):
        painel.botao.click()
    assert painel.linhas(), "âncora: a primeira execução tinha de desenhar"

    painel.recebe_volcano(_corpo(features=["9.0_900.0m/z"], pvalue=[0.02], statistic=[1.0]))

    assert painel.linhas() == (), "a tabela da execução anterior sobreviveu ao corpo novo"
    assert painel.rodape.text() == ""
    assert "Execução nova" in painel.recorte.text()


def test_o_calculo_NAO_corre_na_thread_da_GUI(qtbot: QtBot) -> None:  # noqa: N802
    """O cálculo sai do laço de eventos.

    **Medido antes da correção:** 1,0 a 1,2 s com o processo aquecido e 4,6 a 7,1 s no
    primeiro clique, tudo com a janela congelada. O oráculo aqui é a identidade da thread
    em que a calculadora corre, e não o tempo: tempo mediria a máquina, a thread mede o
    desenho.
    """
    import threading

    vistas: list[int] = []

    def anota(*_a: Any, **_k: Any) -> RelatorioDeVias:
        vistas.append(threading.get_ident())
        return _relatorio()

    painel = PainelDeVias(calculadora=anota)
    qtbot.addWidget(painel)
    painel.recebe_volcano(_corpo())

    principal = threading.get_ident()
    with qtbot.waitSignal(painel._sinais.concluido, timeout=20000):
        painel.botao.click()

    assert vistas, "a calculadora nunca correu"
    assert vistas[0] != principal, (
        "a predição correu na thread da GUI: a janela congela durante o cálculo, que é "
        "exactamente o que KTD9 proíbe"
    )


def test_o_painel_avisa_que_esta_a_calcular(qtbot: QtBot) -> None:
    """Com o cálculo fora da thread, o recorte não pode ficar com o texto antigo.

    O botão desabilitado não é sinal suficiente: sem uma frase, a aba mostraria o recorte
    da corrida ANTERIOR enquanto a nova corre — a mesma confusão que o ramo de falha existe
    para evitar.
    """
    from refino.desktop.janela.vias import A_CALCULAR

    painel = PainelDeVias(calculadora=lambda *a, **k: _relatorio())
    qtbot.addWidget(painel)
    painel.recebe_volcano(_corpo())
    painel._calcula_e_mostra()

    assert painel.recorte.text() == A_CALCULAR
    assert not painel.botao.isEnabled()


# ---------------------------------------------------------------------------
# O caminho REAL — a função de produção sobre o arquivo real
# ---------------------------------------------------------------------------

# O menor dos quatro arquivos, pelo mesmo motivo de `tests/unit/test_pathways_analise.py`: a
# escolha é de CUSTO e não de conteúdo. E o mesmo par de grupos da medição de concordância.
_MODO_REAL = "neg_hilic"
_GRUPOS_REAIS = ("ATLL", "NoHTLV")


@pytest.fixture(scope="module")
def volcano_real() -> dict[str, Any]:
    """O corpo `{"volcano": ...}` com os três vetores que o painel lê, do dado REAL.

    Dado real e não factory: a predição depende de a massa de cada feature casar
    com um composto do modelo, e massa inventada casa por acaso ou nunca.

    **Escopo, declarado:** os vetores saem de `univariate_test` sobre a matriz CARREGADA, e não
    da processada que a rota usa — o painel não sabe a diferença, e o que este teste liga é o
    painel à função de produção. `nan` vira `None`, como o JSON da rota o entrega.
    """
    import numpy as np

    from refino.io.readers import load_matrix
    from refino.statistics.univariate import univariate_test
    from tests.conftest import measurements_path, sample_list_path

    matriz = load_matrix(measurements_path(_MODO_REAL), sample_list_path(_MODO_REAL), _MODO_REAL)
    resultado = univariate_test(matriz, _GRUPOS_REAIS)

    def _json(valores: Any) -> list[float | None]:
        return [float(v) if np.isfinite(v) else None for v in valores]

    return {
        "volcano": {
            "features": list(resultado.feature),
            "pvalue": _json(resultado.pvalue),
            "statistic": _json(resultado.statistic),
        }
    }


def _espera_desfecho(qtbot: QtBot, painel: PainelDeVias) -> str:
    """Clica e devolve QUAL dos dois sinais chegou — o real pode falhar, e o teste tem de o ver."""
    chegou: list[str] = []
    painel._sinais.concluido.connect(lambda *_a: chegou.append("concluido"))
    painel._sinais.falhou.connect(lambda *_a: chegou.append("falhou"))
    painel.botao.click()
    qtbot.waitUntil(lambda: bool(chegou), timeout=60000)
    return chegou[0]


def _nan(valores: list[float | None]) -> Any:
    """O inverso do `_json` do fixture: `None` volta a `nan`, como `vias._vetor` faz."""
    import numpy as np

    return np.array([np.nan if x is None else x for x in valores], dtype=np.float64)


def test_o_painel_PADRAO_prediz_vias_sobre_o_arquivo_real(  # noqa: N802
    qtbot: QtBot, volcano_real: dict[str, Any]
) -> None:
    """Volcano real → `prediz_vias` de produção → tabela e rodapé.

    Os outros testes deste arquivo injetam a calculadora; este usa o PADRÃO, e é o único que
    percorre o caminho em que o rodapé errado *«119 via(s) abaixo de p = 0,05»* era visível.

    **O oráculo é uma segunda chamada a `prediz_vias` com os mesmos argumentos e a mesma
    semente**, formatada pelos mesmos `numero` e `texto_do_recorte` que o painel
    usa: a tela tem de mostrar exactamente o relatório da função. E há duas conferências que não
    passam pelo relatório — o rodapé contra as linhas visíveis, e o p das vias sem
    sobreposição —, porque são elas que o rodapé errado violava.
    """
    painel = PainelDeVias()
    qtbot.addWidget(painel)
    assert painel._calcula is prediz_vias, "âncora: o painel não está com a função de produção"
    painel.sugere_polaridade(_MODO_REAL)
    painel.recebe_volcano(volcano_real)

    desfecho = _espera_desfecho(qtbot, painel)
    assert desfecho == "concluido", f"a predição real falhou: {painel.recorte.text()}"

    v = volcano_real["volcano"]
    esperado = prediz_vias(
        tuple(v["features"]),
        _nan(v["pvalue"]),
        _nan(v["statistic"]),
        modo_de_ionizacao="negative",
        politica_de_massa_neutra=painel.politica.currentData(),
        busca_de_adutos=painel.busca.currentData(),
    )

    linhas = painel.linhas()
    # Não-vacuidade ANTES das comparações: uma tabela vazia igualaria um relatório vazio.
    assert len(linhas) == len(esperado.vias) > 0
    assert linhas == tuple(
        (
            via.nome,
            str(via.sobreposicao),
            str(via.detectados_na_via),
            numero(via.pvalue),
            numero(via.pvalue_fisher),
        )
        for via in esperado.vias
    )
    assert painel.recorte.text() == texto_do_recorte(esperado)

    # O rodapé contra a TELA, e não contra o relatório: quantas linhas visíveis têm p < 0,05.
    visiveis = sum(1 for linha in linhas if float(linha[3].replace(",", ".")) < 0.05)
    assert painel.rodape.text().startswith(f"**{visiveis} via(s) abaixo de p = 0,05.**"), (
        f"o rodapé não conta as linhas que a tabela mostra abaixo de 0,05 ({visiveis}): "
        f"{painel.rodape.text()[:60]!r}"
    )
    # E o fato do rodapé na tela: via sem sobreposição tem p = 1, e nunca entra naquela conta.
    sem_sobreposicao = [linha for linha in linhas if linha[1] == "0"]
    assert sem_sobreposicao, "âncora: nenhuma via sem sobreposição, e a asserção seria vácua"
    assert all(linha[3] == "1" for linha in sem_sobreposicao), sorted(
        {linha[3] for linha in sem_sobreposicao}
    )


def test_uma_feature_real_e_RECUSA_com_motivo_e_nao_defeito_a_reportar(  # noqa: N802
    qtbot: QtBot, volcano_real: dict[str, Any]
) -> None:
    """Na tela: uma feature analisável, o painel padrão, e a recusa do núcleo.

    **Visto reprovar antes do conserto**: a biblioteca levantava `IndexError` e o recorte dizia
    *«A predição de vias falhou de forma inesperada … é um defeito a reportar»*.
    """
    v = volcano_real["volcano"]
    i = next(
        k
        for k, nome in enumerate(v["features"])
        if nome.endswith("m/z") and v["pvalue"][k] is not None and v["statistic"][k] is not None
    )
    corpo = {
        "volcano": {
            "features": [v["features"][i]],
            "pvalue": [v["pvalue"][i]],
            "statistic": [v["statistic"][i]],
        }
    }
    painel = PainelDeVias()
    qtbot.addWidget(painel)
    painel.sugere_polaridade(_MODO_REAL)
    painel.recebe_volcano(corpo)

    assert _espera_desfecho(qtbot, painel) == "falhou"
    texto = painel.recorte.text()
    assert texto.startswith("**Não foi possível predizer vias.**"), texto
    assert "inesperada" not in texto
    assert "pelo menos 2" in texto
    assert painel.linhas() == ()
    assert painel.botao.isEnabled()


def test_o_painel_abre_dizendo_que_a_funcionalidade_e_beta(qtbot: QtBot) -> None:
    """A PRIMEIRA linha do painel diz a quem opera que a predição de vias é beta.

    **A posição é o conteúdo**: a frase enterrada no fim do painel não avisaria ninguém. Por isso o
    par causal vive no mesmo teste — o mesmo predicado aplicado ao SEGUNDO item do layout tem de
    dar falso; sem isso, um predicado que fosse verdadeiro para qualquer item passaria aqui.
    """
    painel = PainelDeVias()
    qtbot.addWidget(painel)
    disposicao = painel.layout()
    assert disposicao is not None, "âncora: o painel não tem layout"
    assert disposicao.count() >= 2, f"âncora: o layout tem {disposicao.count()} item(ns)"

    def e_o_aviso(item: object) -> bool:
        return isinstance(item, QLabel) and item.text() == FUNCIONALIDADE_EM_BETA

    assert e_o_aviso(disposicao.itemAt(0).widget()), "o painel não abre com a frase de beta"
    assert not e_o_aviso(disposicao.itemAt(1).widget()), "par causal: o predicado não separa"


# ---------------------------------------------------------------------------
# O segundo modo da aba: o seletor, a pilha e o token de geração também aqui
# ---------------------------------------------------------------------------


def test_a_linha_1_e_o_seletor_do_modo_e_o_modo_atual_e_o_padrao(qtbot: QtBot) -> None:
    painel = PainelDeVias()
    qtbot.addWidget(painel)
    disposicao = painel.layout()
    assert disposicao is not None
    assert disposicao.itemAt(1).widget() is painel.modo
    assert painel.modo.currentData() == "picos"
    assert painel.pilha.currentIndex() == 0
    assert painel.modo.count() == 2


def test_trocar_de_modo_preserva_o_estado_de_cada_sub_painel(qtbot: QtBot) -> None:
    painel = PainelDeVias()
    qtbot.addWidget(painel)
    painel.polaridade.setCurrentIndex(painel.polaridade.findData("negative"))
    painel.enriquecimento.valor_do_corte.setValue(0.01)
    painel.modo.setCurrentIndex(1)
    assert painel.pilha.currentWidget() is painel.enriquecimento
    painel.modo.setCurrentIndex(0)
    assert painel.polaridade.currentData() == "negative"
    assert painel.enriquecimento.valor_do_corte.value() == pytest.approx(0.01)


def test_o_volcano_chega_aos_dois_modos(qtbot: QtBot) -> None:
    painel = PainelDeVias()
    qtbot.addWidget(painel)
    painel.recebe_volcano({"volcano": {"features": ["f1"], "pvalue": [0.01], "statistic": [1.0]}})
    assert painel.enriquecimento._volcano is not None


def test_execucao_nova_durante_a_predicao_nao_pinta_a_velha(qtbot: QtBot) -> None:
    """A guarda de obsolescência neste modo, com o par causal: sem a execução nova no meio, o
    mesmo resultado pinta."""
    import threading

    trava = threading.Event()

    def lenta(*_a: object, **_k: object) -> RelatorioDeVias:
        trava.wait(5)
        return _relatorio(vias=(_via("Via antiga", 0.01),))

    painel = PainelDeVias(calculadora=lenta)
    qtbot.addWidget(painel)
    corpo = {"volcano": {"features": ["f1"], "pvalue": [0.01], "statistic": [1.0]}}
    painel.recebe_volcano(corpo)
    painel.botao.click()
    painel.recebe_volcano(corpo)  # execução nova com a predição em voo
    trava.set()
    painel._piscina.waitForDone(5000)
    qtbot.wait(100)
    assert painel.linhas() == (), "a tabela da execução anterior foi pintada sob a nova"
    assert painel.botao.isEnabled()
    trava.clear()
    painel.botao.click()
    trava.set()
    qtbot.waitUntil(lambda: painel.linhas() != (), timeout=5000)


def test_encerra_drena_as_duas_piscinas(qtbot: QtBot) -> None:
    painel = PainelDeVias()
    qtbot.addWidget(painel)
    assert painel.encerra(teto_ms=2000)


def test_segundo_clique_com_a_velha_em_voo_nao_pinta_a_velha(qtbot: QtBot) -> None:
    """O caso que a guarda por atributo deixava passar (revisão de 01/10/2026).

    Predição A em voo; chega a execução nova; o usuário clica outra vez (B). A piscina de uma thread
    termina A primeiro: com a geração num atributo sobrescrito pelo clique de B, A passava na guarda
    e pintava sob o recorte novo. A geração tem de viajar com o trabalho.
    """
    import threading

    travas = [threading.Event(), threading.Event()]
    chamadas: list[int] = []

    def lenta(*_a: object, **_k: object) -> RelatorioDeVias:
        indice = len(chamadas)
        chamadas.append(indice)
        travas[indice].wait(5)
        return _relatorio(vias=(_via("Via antiga" if indice == 0 else "Via nova", 0.01),))

    painel = PainelDeVias(calculadora=lenta)
    qtbot.addWidget(painel)
    corpo = {"volcano": {"features": ["f1"], "pvalue": [0.01], "statistic": [1.0]}}
    painel.recebe_volcano(corpo)
    painel.botao.click()  # A
    qtbot.waitUntil(lambda: chamadas == [0], timeout=5000)
    painel.recebe_volcano(corpo)  # execução nova com A em voo
    assert painel.botao.isEnabled(), "âncora: o segundo clique tem de ser possível"
    painel.botao.click()  # B, enfileirado atrás de A
    travas[0].set()
    qtbot.waitUntil(lambda: chamadas == [0, 1], timeout=5000)
    qtbot.wait(100)
    assert painel.linhas() == (), "a tabela da predição velha foi pintada sob o recorte novo"
    travas[1].set()
    qtbot.waitUntil(lambda: painel.linhas() != (), timeout=5000)
    assert painel.linhas()[0][0] == "Via nova"


def test_encerra_espera_a_propria_piscina_mesmo_quando_a_do_enriquecimento_falha(
    qtbot: QtBot,
) -> None:
    """O `and` curto-circuitado deixava de esperar esta piscina quando a outra não drenava."""

    class PiscinaEspia:
        def __init__(self) -> None:
            self.esperas: list[int] = []

        def clear(self) -> None:
            pass

        def waitForDone(self, teto: int) -> bool:  # noqa: N802 - o nome é o do Qt
            self.esperas.append(teto)
            return True

    painel = PainelDeVias()
    qtbot.addWidget(painel)
    real = painel._piscina
    espia = PiscinaEspia()
    painel._piscina = espia  # type: ignore[assignment]
    painel.enriquecimento.encerra = lambda teto_ms=0: False  # type: ignore[method-assign]
    try:
        assert painel.encerra(teto_ms=1234) is False
        assert espia.esperas == [1234], "a piscina das vias não foi drenada"
    finally:
        painel._piscina = real


def test_despacha_vias_leva_a_geracao_nos_dois_desfechos(qtbot: QtBot) -> None:
    """Na thread principal, onde o `coverage` vê: a geração viaja com o desfecho."""
    from refino.desktop.janela.vias import _SinaisDeVias, despacha_vias

    sinais = _SinaisDeVias()
    vistos: list[tuple[str, int]] = []
    sinais.concluido.connect(lambda _r, g: vistos.append(("concluido", g)))
    sinais.falhou.connect(lambda _e, g: vistos.append(("falhou", g)))

    def levanta() -> RelatorioDeVias:
        raise RuntimeError("caiu")

    despacha_vias(lambda: _relatorio(vias=()), sinais, 3)
    despacha_vias(levanta, sinais, 4)
    assert vistos == [("concluido", 3), ("falhou", 4)]
