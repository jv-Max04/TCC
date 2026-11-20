"""As convenções de aparência da janela, com portão por CLASSE — itens 9 e 11 da passagem humana.

Os dois achados vieram da mesma passagem humana, e nenhum apareceu nos 3.889 testes da
suíte: são propriedades do artefato montado sendo usado.

| Achado | O conserto |
|---|---|
| botões que não parecem botões | um ícone de informação, e a forma visual de botão |
| a ressalva do SERRF gritando | um ícone com dica |

## Por que o portão é por CLASSE e não pelas duas instâncias

Consertar os dois botões de hoje não impede o terceiro de nascer chapado. O portão do `setFlat`
percorre o pacote inteiro e reprova quando acha um — é a forma de `test_identidade_do_corretor.py`
aplicada a aparência.

Medido antes de escrever: o oráculo acusava 2 ocorrências na
árvore de antes e **0** depois. Duas de duas é o número certo: era a população inteira, medida com
`grep -rn "setFlat" src/refino/desktop/`.
"""

from __future__ import annotations

import ast
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

from refino.cli import AUTO_CAVEATS, METHOD_CAVEATS  # noqa: E402
from refino.desktop.janela.aparencia import GRITO_ANTIGO, valor_do_campo  # noqa: E402
from refino.desktop.janela.pipeline import PainelDePipeline  # noqa: E402

PACOTE = Path("src/refino/desktop/janela")

# O corretor com ressalva. `SEM_RESSALVA_ALGUM` saiu com o par causal do ícone: o lado
# «apaga» daquele par media um widget que já não existe, e o par causal que
# resta, o do botão «ver o recorte medido», vive em `tests/test_desktop_janela.py`.
COM_RESSALVA = "serrf"


def _chamadas_de_set_flat(caminho: Path) -> list[tuple[int, str]]:
    """`(linha, código)` de cada `…​.setFlat(True)` do arquivo. Lista vazia é o desfecho bom."""
    fonte = caminho.read_text(encoding="utf-8")
    arvore = ast.parse(fonte)
    linhas = fonte.splitlines()
    achados: list[tuple[int, str]] = []
    for no in ast.walk(arvore):
        if not isinstance(no, ast.Call) or not isinstance(no.func, ast.Attribute):
            continue
        if no.func.attr != "setFlat" or not no.args:
            continue
        primeiro = no.args[0]
        if isinstance(primeiro, ast.Constant) and primeiro.value is True:
            achados.append((no.lineno, linhas[no.lineno - 1].strip()))
    return achados


def test_nenhum_botao_da_janela_nasce_chapado() -> None:
    """`setFlat(True)` tira a moldura, e sem moldura um botão lê-se como texto.

    A âncora vem antes: um pacote sem arquivo nenhum faria a asserção de ausência
    abaixo passar por vacuidade — nada estaria «chapado» porque nada seria lido.
    """
    arquivos = sorted(PACOTE.glob("*.py"))
    assert len(arquivos) >= 10, (
        f"âncora: o pacote da janela tem {len(arquivos)} módulos — com poucos, o portão mede o nada"
    )

    chapados = {
        str(caminho): achados for caminho in arquivos if (achados := _chamadas_de_set_flat(caminho))
    }
    assert not chapados, (
        f"estes botões nascem chapados: {chapados}. Um botão sem moldura lê-se como texto — foi o "
        f"que a passagem humana de 07/09/2026 relatou. Use `veste_de_botao` de "
        f"`refino.desktop.janela.aparencia`, que dá moldura e ícone de informação"
    )


def test_o_portao_do_chapado_acusa_quando_a_planta_existe(tmp_path: Path) -> None:
    """A segunda direção, e sem ela a de cima é um portão que nunca viu o que procura.

    **Par causal sobre o MESMO arquivo:** duas leituras que diferem só pela planta, e os desfechos
    têm de divergir. Sem a metade limpa, um oráculo que acusasse sempre passaria neste teste.
    """
    limpo = tmp_path / "limpo.py"
    limpo.write_text("botao.setFlat(False)\noutro.setCheckable(True)\n", encoding="utf-8")
    assert _chamadas_de_set_flat(limpo) == [], (
        "o oráculo acusa código que NÃO planta — `setFlat(False)` é exatamente o que se quer"
    )

    plantado = tmp_path / "plantado.py"
    plantado.write_text("botao.setFlat(False)\nbotao.setFlat(True)\n", encoding="utf-8")
    achados = _chamadas_de_set_flat(plantado)
    assert len(achados) == 1, f"o oráculo devia achar UMA planta e achou {achados}"
    assert achados[0][0] == 2, "a linha reportada tem de ser a da planta"


def test_os_dois_reveladores_tem_moldura_e_icone(qtbot: Any) -> None:
    """Os dois botões que revelam prosa parecem botões, e trazem o ícone de informação."""
    painel = PainelDePipeline()
    qtbot.addWidget(painel)

    reveladores = {
        "de onde vêm os nomes dos grupos": painel.botao_da_dica,
        "ver o recorte medido": painel.botao_do_detalhe,
    }
    for nome, botao in reveladores.items():
        assert not botao.isFlat(), f"«{nome}» continua chapado e lê-se como texto"
        assert not botao.icon().isNull(), f"«{nome}» não tem ícone de informação"
        assert botao.isCheckable(), (
            f"«{nome}» deixou de alternar — o `setCheckable` é o que faz o revelador revelar, e "
            f"tirá-lo trocaria um defeito de aparência por um de comportamento"
        )


# O par causal do ícone da ressalva saiu daqui.
#
# Ele afirmava que `painel.icone_da_ressalva` acendia com dica e apagava sem ela. **Os dois ícones
# do ponto de escolha foram retirados**, e um teste sobre um widget que já não existe não
# é teste afrouxado: é `AttributeError`.
#
# **O que ele media e para onde foi:** a ressalva inteira continua atrás do botão «ver o recorte
# medido», e o par causal desse botão está em `tests/test_desktop_janela.py`. `icone_de_aviso` e
# `linha_com_icone` ficam em `janela/aparencia.py` sem consumidor de produção, declarado lá.


def test_a_ressalva_nao_grita_na_fonte(qtbot: Any) -> None:
    """O `ATENÇÃO — ` não volta ao texto dos `Caveat`. **Âncora sobre a FONTE e não sobre a tela.**

    ## Por que só resta a metade da FONTE

    Este teste tinha duas metades. A primeira lia `painel.rotulo_da_ressalva` e afirmava, em texto,
    que tirar a ressalva do ponto de escolha desfaria uma decisão de escopo. A ressalva saiu do
    ponto de escolha, e aquele rótulo já não existe.

    **A metade que fica é a que lê a FONTE**, e ela nunca dependeu do rótulo: se o grito viesse de
    dentro do próprio `Caveat`, tirá-lo do `f`-string da janela não bastaria — o texto chegaria
    gritado à CLI também. É a razão de ela sobreviver sozinha.
    """
    assert COM_RESSALVA in METHOD_CAVEATS, (
        f"âncora: `{COM_RESSALVA}` devia ter ressalva em `METHOD_CAVEATS` e não tem — sem ela a "
        f"metade da tela, abaixo, não mede nada"
    )
    painel = PainelDePipeline()
    qtbot.addWidget(painel)
    painel.corretor.setCurrentText(COM_RESSALVA)

    fontes = [c.resumo for c in METHOD_CAVEATS.values()] + [c.resumo for c in AUTO_CAVEATS.values()]
    assert fontes, "âncora: nenhuma ressalva declarada — não há o que conferir"
    for resumo in fontes:
        assert GRITO_ANTIGO not in resumo, (
            f"o grito vem de dentro do próprio `Caveat` ({resumo[:60]!r}); tirá-lo da janela não "
            f"bastaria, e o texto chegaria gritado à CLI e à web também"
        )

    # E a tela que resta também não grita, porque o `detalhe` é o que ela desenha.
    texto = painel.rotulo_do_detalhe.text()
    assert texto, "âncora: o rótulo ficou vazio, e vazio lê-se como «a ressalva não carregou»"
    assert GRITO_ANTIGO not in texto, (
        f"o texto ainda grita {GRITO_ANTIGO!r} — o dobro do peso visual que o pedido era para tirar"
    )


# ---------------------------------------------------------------------------
# Itens 1, 2 e 8 — as caixas «Automático»
# ---------------------------------------------------------------------------
#
# **As três caixas têm a MESMA forma e repõem coisas DIFERENTES, e a distinção é o conteúdo.**
#
# | Caixa | O que repõe | Por quê |
# |---|---|---|
# | correção do sinal | a proposta da máquina | `correction` é um dos dois eixos de
#   `PROPOSTAS_POR_ROTA` |
# | corte · pré-processamento | o valor de repouso | *"os dois cortes (…) não são
#   propostos por rota nenhuma"* |
# | limiares do multiverso | o padrão | lido de `StabilityThresholds`, não redigitado |
#
# As duas primeiras linhas saíram da janela, e no lugar delas há um botão «Seguir sugestões do
# app» na parte de cima da seção de entrada. O botão repõe o mesmo que elas repunham, sem travar
# nada; a caixa do multiverso fica, e é ela que `CaixaDeAutomatico` ainda serve.


def test_a_caixa_de_automatico_trava_repoe_e_liberta(qtbot: Any) -> None:
    """O ciclo de TRÊS estados, e é a terceira volta que carrega o trabalho.

    | Gesto | Efeito |
    |---|---|
    | ticado (padrão) | os campos ficam **bloqueados** |
    | desticar | **libertam** |
    | voltar a ticar | **voltam ao repouso** e bloqueiam outra vez |

    Re-ticar é um **desfazer**: sem a terceira volta, a caixa seria um `setEnabled` com nome
    grande — e ticar a caixa depois de modificar os parâmetros volta às recomendações iniciais.
    """
    from PySide6.QtWidgets import QDoubleSpinBox, QWidget

    from refino.desktop.janela.aparencia import CaixaDeAutomatico

    pai = QWidget()
    qtbot.addWidget(pai)
    campo = QDoubleSpinBox(pai)
    campo.setRange(0.0, 1.0)
    campo.setValue(0.30)

    caixa = CaixaDeAutomatico("Automático", pai, campos=(campo,))
    assert caixa.isChecked(), "a caixa nasce desticada"
    assert not campo.isEnabled(), "ticada e o campo continua editável"
    assert caixa.valores_de_repouso() == (0.30,), (
        f"o repouso lido foi {caixa.valores_de_repouso()} e o campo valia 0,30 — a caixa reporia "
        f"um valor que a tela nunca mostrou"
    )

    caixa.setChecked(False)
    assert campo.isEnabled(), "desticar não libertou o campo"
    campo.setValue(0.75)
    assert campo.value() == 0.75, (
        "âncora: o campo não aceitou o valor novo, e o desfazer abaixo não teria o que desfazer"
    )

    caixa.setChecked(True)
    assert campo.value() == 0.30, (
        f"voltar a ticar devia REPOR 0,30 e o campo ficou em {campo.value()} — a caixa marca e não "
        f"desfaz, que é o `setEnabled` com nome grande"
    )
    assert not campo.isEnabled()


def test_a_caixa_sem_campos_e_recusada_na_construcao(qtbot: Any) -> None:
    """Um controle que não governa nada é indistinguível de um que está partido."""
    from PySide6.QtWidgets import QWidget

    from refino.desktop.janela.aparencia import CaixaDeAutomatico

    pai = QWidget()
    qtbot.addWidget(pai)
    with pytest.raises(ValueError, match="sem campos"):
        CaixaDeAutomatico("Automático", pai, campos=())


def test_o_botao_das_sugestoes_da_normalizacao_repoe_os_campos_das_duas_caixas_que_sairam(
    qtbot: Any,
) -> None:
    """`repoe_os_valores_iniciais` repõe os campos que as duas caixas governavam, e só eles.

    Era `test_as_duas_caixas_do_pipeline_governam_campos_disjuntos`. As duas caixas da
    Normalização — *«por grupo — uma para o pré-processamento, outra para o corte»* — saíram com a
    da Correção, e o botão «Seguir sugestões do app» repõe a união dos campos delas, sem travar
    nenhum. **Par causal:** a referência, a variante do RSD e o corretor também foram mudados, e
    ficam — a referência é escolha da pessoa entre os nomes que a «Validação» ofereceu, e os outros
    dois são da Correção.
    """
    painel = PainelDePipeline()
    qtbot.addWidget(painel)
    campos = (painel.corte, painel.ausentes, painel.max_features, *painel.eixos.values())
    iniciais = [valor_do_campo(c) for c in (*campos, painel.fracao_posto, painel.fracao_posto_2)]
    painel.corte.setValue(0.3)
    painel.ausentes.setValue(0.5)
    painel.max_features.setValue(1234)
    for eixo, combo in painel.eixos.items():
        outro = next(
            (
                combo.itemText(i)
                for i in range(combo.count())
                if combo.itemText(i) != combo.currentText()
            ),
            None,
        )
        assert outro is not None, f"âncora: o eixo {eixo} só tem uma opção"
        combo.setCurrentText(outro)
    painel.fracao_posto.setValue(0.5)
    painel.fracao_posto_2.setValue(0.5)
    painel.referencia.setCurrentText("QC1")
    painel.variante_do_rsd.setCurrentText("plain")
    painel.corretor.setCurrentText("serrf")
    mexidos = [valor_do_campo(c) for c in (*campos, painel.fracao_posto, painel.fracao_posto_2)]
    assert all(a != b for a, b in zip(iniciais, mexidos, strict=True)), (
        f"âncora: a planta não mudou todos os campos: {iniciais} -> {mexidos}"
    )

    mudados = painel.repoe_os_valores_iniciais()

    repostos = [valor_do_campo(c) for c in (*campos, painel.fracao_posto, painel.fracao_posto_2)]
    assert repostos == iniciais
    assert mudados == len(iniciais), f"o botão contou {mudados} campos mudados de {len(iniciais)}"
    assert (
        painel.referencia.currentText(),
        painel.variante_do_rsd.currentText(),
        painel.corretor.currentText(),
    ) == ("QC1", "plain", "serrf"), "o botão da Normalização repôs um campo que não é dela"
    assert painel.repoe_os_valores_iniciais() == 0, "o segundo clique contou mudanças que não houve"


def test_d554_a_variante_do_rsd_vive_na_correcao_e_fica_editavel(qtbot: Any) -> None:
    """Uma decisão do plano do passo a passo.

    A variante é parâmetro da etapa 1 (`ETAPA_DO_CAMPO`), e mora no quadro dela, editável.
    O teste afirmava que nenhuma das três caixas «Automático» a travava, com o par causal
    de um eixo do pré-processamento travado pela caixa dele; as caixas saíram, e todos os campos
    ficam editáveis — o que resta afirmar é o quadro e a edição.
    """
    from refino.desktop.etapas import ETAPA_DO_CAMPO

    painel = PainelDePipeline()
    qtbot.addWidget(painel)

    assert ETAPA_DO_CAMPO["rsd_variant"] == 1
    grupo = painel.variante_do_rsd.parentWidget()
    assert grupo is not None and grupo.title() == "Correção do sinal", (
        "a variante do RSD saiu do quadro da etapa a que ela pertence"
    )
    assert grupo is painel.corretor.parentWidget(), "âncora: o corretor mora no mesmo quadro"
    assert painel.variante_do_rsd.isEnabled() and painel.eixos["imputation"].isEnabled()
