"""A roda do rato não mexe no campo sob o cursor — item 3 da passagem humana.

Campos de escolha e de número não mudam com a roda do rato por cima: só mudam clicando.

É comportamento padrão do Qt e é a armadilha clássica do formulário longo: o painel tem duas
colunas e rola, e passar a roda para descer a página muda em silêncio o campo sob o cursor. **O
profissional não vê que mudou** — e o campo alterado vai para o payload.

## Por que o portão é de RUNTIME e não por AST

Um portão que procurasse `sem_roda_sem_foco(` no código conferiria a **forma** da chamada, e
passaria numa árvore em que a função existisse e não fizesse nada. Este monta a
`JanelaPrincipal` inteira, percorre **todos** os widgets roláveis que ela tem, e manda-lhes uma
roda de verdade: montar o artefato e usá-lo como quem o recebe.

## A ÂNCORA é a metade que faz este portão valer alguma coisa

Um evento de roda sintético que o Qt ignorasse produziria *"nenhum valor mudou"* em toda a janela —
o desfecho exato que se quer ver, a partir de um instrumento que não mede nada.

Então o par causal está na **mesma função**: com o widget FOCADO, a mesma roda **tem** de mudar o
valor. Se não mudar, o instrumento é que está partido, e é isso que a mensagem diz.
"""

from __future__ import annotations

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

from fastapi.testclient import TestClient  # noqa: E402
from PySide6.QtCore import QPoint, QPointF, Qt  # noqa: E402
from PySide6.QtGui import QWheelEvent  # noqa: E402
from PySide6.QtWidgets import QAbstractSpinBox, QApplication, QComboBox  # noqa: E402

from refino.api.app import build_app  # noqa: E402
from refino.desktop.janela.principal import JanelaPrincipal  # noqa: E402

# Medido sobre a janela montada. É piso e não igualdade: um controle novo não deve
# reprovar um portão que existe para exigir política, e o que interessa é que a varredura não meça
# o nada.
PISO_DE_ROLAVEIS = 13

# Uma coorte de dois grupos, no corpo em que ela chega à janela — só o nome de cada
# classe importa à tabela dos estilos, que é o que a fixture abaixo precisa de fazer nascer.
COMPOSICAO_COM_DOIS_GRUPOS: dict[str, Any] = {
    "composicao_por_classe": {"classes": [{"grupo": "Controle"}, {"grupo": "Caso"}]}
}


def _rolaveis(janela: JanelaPrincipal) -> list[Any]:
    """Todo `QComboBox` e todo `QAbstractSpinBox` da janela montada, em qualquer profundidade."""
    return [*janela.findChildren(QComboBox), *janela.findChildren(QAbstractSpinBox)]


def _valor(widget: Any) -> Any:
    """O que a roda mudaria: o índice do combo, ou o valor do spin."""
    return widget.currentIndex() if isinstance(widget, QComboBox) else widget.value()


def _gira(widget: Any, *, passos: int = -3) -> None:
    """Manda uma roda de verdade ao widget. Negativo desce, que é o gesto de rolar a página."""
    ponto = QPointF(widget.rect().center())
    evento = QWheelEvent(
        ponto,
        widget.mapToGlobal(QPointF(ponto).toPoint()).toPointF()
        if hasattr(widget.mapToGlobal(QPoint(0, 0)), "toPointF")
        else ponto,
        QPoint(0, passos * 40),
        QPoint(0, passos * 120),
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
        Qt.ScrollPhase.NoScrollPhase,
        False,
    )
    QApplication.sendEvent(widget, evento)


@pytest.fixture
def janela(qtbot: Any) -> JanelaPrincipal:
    """A janela MOSTRADA, e o `show()` não é enfeite — foi a âncora que o exigiu.

    **Medido em 07/09/2026, e é um defeito que eu próprio escrevi no arnês.** A primeira versão
    desta fixture não chamava `show()`, e sob `QT_QPA_PLATFORM=offscreen` um widget de janela não
    ativa **não aceita foco**: `setFocus()` corria, `hasFocus()` devolvia `False`, e a âncora
    reprovou.

    O que ela impediu é o que importa: sem foco possível em lado nenhum, o teste irmão — *«a roda
    não muda campo nenhum SEM foco»* — passaria em **qualquer** implementação, inclusive numa sem
    filtro. Seria uma varredura de treze widgets a medir o nada, com o desfecho de sucesso.
    """
    j = JanelaPrincipal(TestClient(build_app(), raise_server_exceptions=False))
    qtbot.addWidget(j)
    j.show()
    qtbot.waitExposed(j)
    j.activateWindow()
    # Os seletores de cor e de símbolo dos grupos só nascem com a coorte, que chega na
    # composição por classe de uma execução. Na janela em repouso eles não existem, e a varredura
    # passaria ao lado deles — medindo o nada sobre os seletores mais novos da janela.
    j.graficos.recebe_composicao(COMPOSICAO_COM_DOIS_GRUPOS, corresponde=True)
    return j


def test_a_roda_nao_muda_campo_nenhum_da_janela_sem_foco(janela: JanelaPrincipal) -> None:
    """A varredura inteira: nenhum campo rolável muda de valor com a roda e sem foco."""
    alvos = _rolaveis(janela)
    assert len(alvos) >= PISO_DE_ROLAVEIS, (
        f"âncora: a janela montada expõe {len(alvos)} campos roláveis e o piso é "
        f"{PISO_DE_ROLAVEIS} — com menos, a varredura mede o nada"
    )
    # Âncora dos seletores dos grupos: quatro — cor e símbolo de cada um dos dois —, e
    # todos dentro da varredura. Sem ela, uma tabela que não nascesse passaria por este portão.
    dos_grupos = [s for par in janela.graficos.tabela_dos_grupos.seletores.values() for s in par]
    assert len(dos_grupos) == 4 and all(s in alvos for s in dos_grupos), dos_grupos

    # **OS DOIS SENTIDOS, e isto foi um defeito MEDIDO no proprio arnes (07/09/2026).** A
    # primeira versao girava so para baixo, e com a planta posta — `corte` a nascer sem politica —
    # este teste **passou**: `corte` repousa no MINIMO, e para baixo nao ha para onde ir. Um combo
    # no indice 0 tem o mesmo problema ao contrario. Girar nos dois sentidos e o que faz a
    # varredura separar em todo widget, seja qual for o valor em que ele repousa.
    # **AS CAIXAS «AUTOMÁTICO» SÃO DESTICADAS ANTES DE VARRER, e isto foi MEDIDO — 07/09/2026.**
    #
    # A revisão adversarial não afirmou: **plantou**. Medido na janela montada: **18 campos
    # roláveis, 12 DESABILITADOS** no repouso, porque as duas caixas de `CaixaDeAutomatico` nascem
    # ticadas. Um widget desabilitado engole a roda pelo `setEnabled`, **não** pelo `FiltroDaRoda`.
    #
    # A âncora anterior pedia `>= 3` habilitados — **agregada**, não por widget —, e com ela o
    # revisor plantou o erro realista (filtro esquecido, `StrongFocus` posto à mão) num dos eixos
    # da grade e **os três portões passaram**. Em produção aquele combo salta de `0` para `1` no
    # instante em que o profissional destica a caixa, que é o gesto normal de entrada manual.
    #
    # **Desticar antes de varrer é o que torna a varredura completa**, e é também o estado em que
    # a roda importa: com a caixa ticada o campo nem se edita.
    # As três caixas do pipeline saíram, e os campos que elas travavam nascem editáveis;
    # a do multiverso fica, e é a única que ainda precisa de ser desticada aqui.
    janela.multiverso.automatico_dos_limiares.setChecked(False)

    habilitados = [w for w in alvos if w.isEnabled()]
    assert habilitados == alvos, (
        f"{len(alvos) - len(habilitados)} dos {len(alvos)} campos roláveis continuam "
        f"DESABILITADOS depois de desticar as caixas: "
        f"{[type(w).__name__ for w in alvos if not w.isEnabled()]}. Um campo desabilitado engole a "
        f"roda pelo `setEnabled` e não pelo filtro — a varredura passaria sobre ele sem medir nada"
    )

    mudaram: list[str] = []
    for widget in alvos:
        widget.clearFocus()
        for sentido in (-3, 3):
            antes = _valor(widget)
            _gira(widget, passos=sentido)
            if _valor(widget) != antes:
                mudaram.append(
                    f"{type(widget).__name__} {widget.objectName() or '<sem nome>'} "
                    f"(giro {sentido:+d}): {antes} -> {_valor(widget)}"
                )

    assert not mudaram, (
        f"a roda mudou estes campos SEM foco: {mudaram}. Quem rola a página para ler o resto do "
        f"formulário altera em silêncio o campo sob o cursor, e o valor alterado vai para o "
        f"payload. Envolva a construção com `sem_roda_sem_foco` de "
        f"`refino.desktop.janela.aparencia`"
    )


def test_a_roda_com_foco_continua_a_mudar_o_valor(janela: JanelaPrincipal) -> None:
    """A âncora do teste acima, e sem ela ele passa com um evento que o Qt ignora.

    **Este é o lado que prova que o instrumento mede.** Um `QWheelEvent` mal construído produziria
    *"nenhum campo mudou"* na janela inteira — indistinguível de sucesso.

    **A escolha «só sem foco» é declarada e é dele por consequência:** ele disse *"apenas mude
    clicando"*, o que leria como tirar a roda de vez; esta é a forma mais estreita, e preserva a
    roda para quem a usa de propósito depois de clicar no campo.
    """
    # **`isEnabled()` entrou em 07/09/2026, e foi a âncora que o exigiu — pela TERCEIRA vez.** As
    # caixas «Automático» dos itens 1, 2 e 8 nascem ticadas e **desabilitam** os campos que
    # governam, e um widget desabilitado não aceita foco: o primeiro combo da janela passou a ser
    # um deles, e a âncora reprovou. O que ela impediu é o de sempre — sem foco possível, o teste
    # irmão passaria em qualquer implementação.
    alvos = [
        w for w in _rolaveis(janela) if isinstance(w, QComboBox) and w.count() > 1 and w.isEnabled()
    ]
    assert alvos, (
        "âncora: nenhum combo HABILITADO com mais de um item. As caixas «Automático» desabilitam "
        "os campos que governam, e um campo desabilitado não aceita foco nem roda — se todos "
        "estiverem travados, o teste irmão não separa nada"
    )

    combo = alvos[0]
    # **O PRIMEIRO item e nao o ultimo, e a diferenca foi medida.** A roda para BAIXO avanca para o
    # item seguinte; a partir do ultimo nao ha seguinte, e o valor nao muda — a planta nao separa,
    # e o desfecho seria «a roda com foco tambem nao mexe», que e a conclusao oposta a
    # verdadeira. Do primeiro item ha sempre para onde ir.
    combo.setCurrentIndex(0)
    combo.setFocus(Qt.FocusReason.MouseFocusReason)
    assert combo.hasFocus(), (
        "âncora: o combo não aceitou foco. Sob `offscreen` isso acontece se a política de foco "
        "for `NoFocus` — e então o teste de cima passaria por não haver foco em lado nenhum"
    )

    antes = combo.currentIndex()
    _gira(combo)
    assert combo.currentIndex() != antes, (
        "com FOCO a roda devia continuar a mudar o valor, e não mudou. Ou o filtro passou a "
        "engolir "
        "sempre — e a decisão registada é «só sem foco» —, ou o evento sintético deste arnês não é "
        "uma roda de verdade, e nesse caso o teste irmão não mede nada"
    )


def test_todo_rolavel_pede_foco_forte(janela: JanelaPrincipal) -> None:
    """`StrongFocus` é metade do conserto, e sem ele a política diria uma coisa e faria outra.

    O padrão de um `QComboBox` é `WheelFocus`: ele **ganha foco pela roda**. Com essa política, o
    primeiro giro focaria o widget e o segundo mudaria o valor — *«só com foco»* no papel, e o
    comportamento de antes com um giro de atraso.
    """
    alvos = _rolaveis(janela)
    assert len(alvos) >= PISO_DE_ROLAVEIS, f"âncora: só {len(alvos)} roláveis"

    fracos = [
        f"{type(w).__name__} {w.objectName() or '<sem nome>'}: {w.focusPolicy().name}"
        for w in alvos
        if w.focusPolicy() == Qt.FocusPolicy.WheelFocus
    ]
    assert not fracos, (
        f"estes campos ganham foco PELA RODA: {fracos}. O filtro deixa passar o evento a partir do "
        f"segundo giro, e a política diz uma coisa e faz outra"
    )
