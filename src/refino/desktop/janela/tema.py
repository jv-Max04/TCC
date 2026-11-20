"""Tema e navegação da janela nativa.

O visual é preto e grafite, com cartões separados, ícones de linha, controles legíveis e conteúdo
centralizado. Paleta e QSS compartilham os mesmos tokens, nos dois
esquemas, escuro (o padrão) e claro; `troca_o_esquema` troca folha, paleta, ícones e moldura com a
janela aberta. A casca comanda o `QTabWidget` original sem trocar a posse dos painéis, preservando
controles e sinais. As figuras científicas conservam sua apresentação própria para exportação.
"""

from __future__ import annotations

import sys
from dataclasses import asdict, dataclass
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Literal, cast

from PySide6.QtCore import QEvent, QObject, QSize, Qt
from PySide6.QtGui import QAction, QColor, QFont, QFontDatabase, QIcon, QPalette, QPixmap
from PySide6.QtWidgets import (
    QAbstractButton,
    QApplication,
    QButtonGroup,
    QComboBox,
    QDockWidget,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QScrollBar,
    QSizePolicy,
    QStyledItemDelegate,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from refino.desktop.janela.expansao import ExpansaoSuave

if TYPE_CHECKING:
    from collections.abc import Mapping

    from refino.desktop.etapas import EstadoDaEtapa
    from refino.desktop.janela.principal import JanelaPrincipal

__all__ = [
    "ATRIBUTOS_DA_MOLDURA_ESCURA",
    "CLARO",
    "DESCRICOES",
    "ESCURO",
    "ESQUEMA_PADRAO",
    "ICONES",
    "ICONES_DAS_FICHAS",
    "ITENS_A_VISTA_NO_SELETOR",
    "LARGURA_DA_BARRA",
    "NOMES_DOS_ESQUEMAS",
    "PAPEIS",
    "PONTOS_DAS_ETAPAS",
    "PROPRIEDADE_DO_ESQUEMA",
    "PROPRIEDADE_DO_ICONE",
    "SETAS",
    "SUPERFICIES_NEUTRAS",
    "SUPERFICIE_DOS_ICONES",
    "BarraLateral",
    "Cores",
    "SegueAAltura",
    "escurece_a_moldura",
    "esquema_em_vigor",
    "folha_de_estilo",
    "icone",
    "icone_nitido",
    "marca_papel",
    "marca_preenchido",
    "pagina_rolavel",
    "paleta",
    "pasta_de_recursos",
    "poe_icone",
    "troca_o_esquema",
    "veste_a_aplicacao",
    "veste_a_janela",
    "veste_os_seletores",
]

Esquema = Literal["claro", "escuro"]

# Escuro por padrão. O claro está no controle 'Escuro | Claro' da barra lateral
# (`troca_o_esquema`), e a escolha não fica gravada entre sessões. O esquema não segue o do Windows:
# num Windows claro, isso deixaria a janela toda branca.
ESQUEMA_PADRAO: Final[Esquema] = "escuro"

# O nome de cada esquema no controle da barra lateral, na ordem em que aparecem, o padrão primeiro.
NOMES_DOS_ESQUEMAS: Final[dict[Esquema, str]] = {"escuro": "Escuro", "claro": "Claro"}

# Onde mora o esquema em vigor: numa propriedade da `QApplication`. É ela que a paleta e a folha
# vestem, então o esquema é dela; um ícone desenhado depois da troca lê daqui a cor certa.
PROPRIEDADE_DO_ESQUEMA: Final = "refino_esquema"

# A propriedade onde cada ícone desenhado com a cor de um token grava a receita. O ícone é desenho
# fixo, feito na cor do esquema em que nasceu; a troca de esquema relê a receita e o redesenha.
PROPRIEDADE_DO_ICONE: Final = "refino_icone"

# O corpo do texto, em pontos. Ver `veste_a_aplicacao`.
TAMANHO_DA_FONTE_DO_CORPO: Final = 10.0

# Quantos itens um seletor aberto mostra antes de rolar. Ver `veste_os_seletores`.
ITENS_A_VISTA_NO_SELETOR: Final = 12

# O papel visual de um botão, lido pela folha como `QPushButton[papel="primario"]`. Domínio fechado
# e conferido em runtime em `marca_papel`: um papel digitado errado cairia no estilo padrão sem
# nada acusar.
PAPEIS: Final = frozenset({"primario", "perigo"})


@dataclass(frozen=True)
class Cores:
    """Os tokens de cor, num lugar só: a folha e a paleta leem daqui, e nunca divergem."""

    fundo: str  # o fundo da página, atrás dos cartões
    cartao: str  # o fundo dos grupos
    cartao_interno: str  # grupo dentro de grupo: um degrau de profundidade
    campo: str  # o fundo dos campos de entrada
    borda: str
    borda_forte: str
    texto: str
    texto_suave: str
    destaque: str  # a cor de ação: foco, sublinhado da aba, contorno
    destaque_texto: str  # o destaque legível como texto sobre o cartão
    destaque_fundo: str  # o destaque bem diluído: etiqueta de seção, botão marcado, seleção
    gradiente_a: str  # o degradê dos botões primários, da esquerda...
    gradiente_b: str  # ...para a direita
    sobre_destaque: str  # texto por cima do degradê
    perigo: str
    perigo_fundo: str
    desabilitado: str
    desabilitado_fundo: str
    rolagem: str
    sucesso: str  # o estado 'feito': arquivo escolhido, leitura respondida
    sucesso_fundo: str
    aviso: str  # o estado 'aceito, com ressalva'
    aviso_fundo: str
    topo: str  # a banda do cabeçalho e das abas, da esquerda...
    topo_2: str  # ...para a direita
    topo_texto: str
    topo_texto_suave: str
    # O preenchimento da barra de progresso tem token próprio: o degradê dos botões primários é
    # quase branco no escuro, e sob o texto claro o percentual sumiria a 100%. O degradê continua a
    # servir aos botões, onde o texto é escuro.
    progresso: str
    # Os tokens abaixo dão nome à cor de cada regra da folha, para ela valer nos dois esquemas: uma
    # cor escrita por extenso na folha vale igual no claro, onde fica ilegível. No escuro cada token
    # tem o valor original, um por cor, e por isso há cinzentos quase iguais (`#888888`, `#8b8b8b`,
    # `#8c8c8c`): fundi-los mudaria a janela do escuro.
    nav_hover: str  # o 'passar o rato' da navegação
    texto_maximo: str  # o texto do item marcado da navegação e do 'Selecionar arquivo' sob o rato
    nota_info: str  # a faixa à esquerda do aviso informativo
    nota_info_icone: str
    nota_aviso: str  # a faixa e o ícone do aviso 'aviso'
    nota_aviso_fundo: str
    nota_aviso_borda: str
    nota_erro: str  # a faixa e o ícone do aviso de erro
    nota_erro_fundo: str
    nota_erro_borda: str
    nota_sucesso: str
    nota_fechar: str  # o botão de fechar do aviso
    ref_borda: str  # a borda dos cartões da referência
    ref_divisoria: str  # as linhas finas: a contagem de arquivos, as classes da legenda
    ref_superficie: str  # o trilho dos segmentos do modo e o fundo do ícone do cartão
    ref_texto_suave: str  # subtítulos, notas e explicações da referência
    texto_apagado: str  # descrições dos arquivos, a seta de 'Detalhes' e de 'Opções avançadas'
    segmento_texto: str
    segmento_hover_texto: str
    segmento_marcado_texto: str
    invertido: str  # o botão de tom invertido: 'Ler…' nos cartões e o segmento marcado
    invertido_hover: str
    sobre_invertido: str  # o texto e o ícone do botão invertido
    foco: str  # o contorno de foco dos controles da referência
    texto_forte: str  # títulos dentro dos detalhes e da legenda
    obrigatorio_texto: str  # o 'Obrigatório' dos cartões de arquivo
    acao_suave: str  # o 'Remover' e os ícones de envio e de remoção
    superficie_funda: str  # o fundo dos detalhes de leitura e da legenda: abaixo do cartão
    detalhe_borda: str
    legenda_introducao: str
    campo_hover_borda: str
    icone_do_arquivo: str
    beta: str  # a etiqueta 'beta' das funcionalidades fora da entrega
    beta_fundo: str
    beta_borda: str


# Paleta grafite: superfícies discretas, texto claro e estados semânticos.
ESCURO: Final = Cores(
    fundo="#070707",
    cartao="#0f0f0f",
    cartao_interno="#1b1b1b",
    campo="#141414",
    borda="#252525",
    borda_forte="#303030",
    texto="#ebebeb",
    texto_suave="#868686",
    destaque="#bdbdbd",
    destaque_texto="#f2f2f2",
    destaque_fundo="#1f1f1f",
    gradiente_a="#f2f2f2",
    gradiente_b="#f2f2f2",
    sobre_destaque="#070707",
    perigo="#f6a0a9",
    perigo_fundo="#382026",
    desabilitado="#777777",
    desabilitado_fundo="#141414",
    rolagem="#383838",
    sucesso="#8bd4b0",
    sucesso_fundo="#18332a",
    aviso="#e8c47e",
    aviso_fundo="#352d20",
    topo="#030303",
    topo_2="#030303",
    topo_texto="#ebebeb",
    topo_texto_suave="#868686",
    # Contraste 4,8:1 com o texto `#ebebeb` por cima, e 3,2:1 com o trilho `#141414` em volta.
    progresso="#666666",
    # Os tokens de papel abaixo guardam os valores originais do escuro, sem mudar um dígito.
    nav_hover="rgba(255, 255, 255, 0.06)",
    texto_maximo="#ffffff",
    nota_info="#8b9ab0",
    nota_info_icone="#aeb9c9",
    nota_aviso="#d6b773",
    nota_aviso_fundo="#1b1811",
    nota_aviso_borda="#50422b",
    nota_erro="#ee9999",
    nota_erro_fundo="#201415",
    nota_erro_borda="#573234",
    nota_sucesso="#a5c6b1",
    nota_fechar="#a5a5a5",
    ref_borda="#262626",
    ref_divisoria="#292929",
    ref_superficie="#1a1a1a",
    ref_texto_suave="#8b8b8b",
    texto_apagado="#888888",
    segmento_texto="#8c8c8c",
    segmento_hover_texto="#eeeeee",
    segmento_marcado_texto="#151515",
    invertido="#f4f4f4",
    invertido_hover="#dedede",
    sobre_invertido="#111111",
    foco="#777777",
    texto_forte="#f4f4f4",
    obrigatorio_texto="#f1f1f1",
    acao_suave="#aaaaaa",
    superficie_funda="#090909",
    detalhe_borda="#222222",
    legenda_introducao="#c8c8c8",
    campo_hover_borda="#484848",
    icone_do_arquivo="#858585",
    beta="#d4a72c",
    beta_fundo="#2a2210",
    beta_borda="#7a5f1c",
)

CLARO: Final = Cores(
    fundo="#edf1f8",
    cartao="#ffffff",
    cartao_interno="#f4f7fc",
    campo="#f7f9fd",
    borda="#dce3ef",
    borda_forte="#b7c4d9",
    texto="#19263d",
    texto_suave="#526580",
    destaque="#2868e3",
    destaque_texto="#2156b9",
    destaque_fundo="#e2ecff",
    gradiente_a="#2868e3",
    gradiente_b="#2056c5",
    sobre_destaque="#ffffff",
    perigo="#b42318",
    perigo_fundo="#fdecea",
    desabilitado="#9aa7b4",
    desabilitado_fundo="#edf1f5",
    rolagem="#c1cbd5",
    # Verde escuro o bastante para texto: 4,98:1 sobre o fundo da página, 4,99:1 sobre o selo
    # 'Aceita' e 5,64:1 sobre o cartão.
    sucesso="#0a7650",
    sucesso_fundo="#e3f5ed",
    aviso="#a15c07",
    aviso_fundo="#fdf2dc",
    # A barra lateral clara é uma superfície da página, como no escuro: as regras da navegação leem
    # os tokens da página (`texto_suave`, `borda`, `destaque_fundo`), e sobre uma barra azul-noite
    # o título da seção e o item marcado ficariam ilegíveis.
    topo="#f7f9fd",
    topo_2="#f7f9fd",
    topo_texto="#19263d",
    topo_texto_suave="#526580",
    # O preenchimento dá 4,61:1 com o texto `#19263d` por cima (texto pede 4,5:1) e 3,12:1 com o
    # trilho `#f7f9fd` em volta (elemento gráfico pede 3:1). A faixa que serve às duas é estreita:
    # luminância entre 0,262 e 0,282.
    progresso="#628ed9",
    # As cores de cada papel da folha. Os contrastes estão conferidos em
    # `tests/test_desktop_tema.py`.
    nav_hover="rgba(25, 38, 61, 0.06)",
    texto_maximo="#19263d",
    nota_info="#526580",
    nota_info_icone="#526580",
    nota_aviso="#a15c07",
    nota_aviso_fundo="#fdf2dc",
    nota_aviso_borda="#efd8a6",
    nota_erro="#b42318",
    nota_erro_fundo="#fdecea",
    nota_erro_borda="#f4c7c3",
    nota_sucesso="#0a7650",
    nota_fechar="#526580",
    ref_borda="#dce3ef",
    ref_divisoria="#dce3ef",
    ref_superficie="#edf1f8",
    ref_texto_suave="#526580",
    texto_apagado="#526580",
    segmento_texto="#526580",
    segmento_hover_texto="#19263d",
    segmento_marcado_texto="#ffffff",
    invertido="#19263d",
    invertido_hover="#2c3d5c",
    sobre_invertido="#ffffff",
    foco="#2868e3",
    texto_forte="#19263d",
    obrigatorio_texto="#19263d",
    acao_suave="#526580",
    superficie_funda="#f7f9fd",
    detalhe_borda="#dce3ef",
    legenda_introducao="#33445f",
    campo_hover_borda="#8a9bb5",
    icone_do_arquivo="#526580",
    beta="#8a6400",
    beta_fundo="#fbf1d6",
    beta_borda="#d8b85e",
)

_POR_ESQUEMA: Final[dict[str, Cores]] = {"claro": CLARO, "escuro": ESCURO}

# As superfícies sobre as quais um texto sem fundo próprio pode estar: a página, os cartões, os
# campos, os detalhes e a barra lateral. Em cada esquema elas ficam do mesmo lado da escala, e é
# isso que permite conferir um texto contra todas sem saber em qual delas ele cai.
SUPERFICIES_NEUTRAS: Final = (
    "fundo",
    "cartao",
    "cartao_interno",
    "campo",
    "superficie_funda",
    "topo",
)

# Cada cor de ícone, e as superfícies sobre as quais ela é desenhada. O ícone é desenho fixo, feito
# na cor do esquema em que nasceu; `poe_icone` só aceita uma cor daqui, e é este mapa que o teste de
# contraste lê (3:1 contra cada superfície, WCAG 1.4.11). Uma cor de ícone nova entra com a
# superfície dela, ou não desenha. O mapa é declarado: o teste que o confere contra a folha só vê o
# fundo que uma regra dá ao widget pelo nome dele (o quadrado do ícone do cartão, 'Remover' e o
# fechar do aviso sob o rato).
_AVISOS: Final = ("cartao_interno", "nota_aviso_fundo", "nota_erro_fundo")
SUPERFICIE_DOS_ICONES: Final[dict[str, tuple[str, ...]]] = {
    "texto": SUPERFICIES_NEUTRAS,
    "texto_suave": SUPERFICIES_NEUTRAS,
    "texto_apagado": SUPERFICIES_NEUTRAS,
    "destaque_texto": (*SUPERFICIES_NEUTRAS, "destaque_fundo", "ref_superficie"),
    "perigo": SUPERFICIES_NEUTRAS,
    "acao_suave": (*SUPERFICIES_NEUTRAS, "perigo_fundo"),
    "icone_do_arquivo": SUPERFICIES_NEUTRAS,
    "topo_texto": ("topo", "destaque_fundo"),
    "topo_texto_suave": ("topo",),
    "sobre_invertido": ("invertido", "invertido_hover"),
    "nota_info_icone": ("cartao_interno",),
    "nota_sucesso": ("cartao_interno",),
    "nota_aviso": ("nota_aviso_fundo",),
    "nota_erro": ("nota_erro_fundo",),
    "nota_fechar": (*_AVISOS, "cartao"),
}

# A folha, com os tokens entre chaves. `{{` e `}}` são as chaves literais do QSS, porque a montagem
# é `str.format`. Ordem: base, cabeçalho, abas, cartões, botões, campos, listas, barras.
_MODELO: Final = """
QMainWindow, QDialog {{ background: {fundo}; }}
QWidget {{ color: {texto}; font-family: "IBM Plex Sans"; font-size: 13px; }}
QLabel {{ background: transparent; }}
QLabel:disabled {{ color: {desabilitado}; }}
QScrollArea, QScrollArea > QWidget > QWidget {{ background: transparent; border: none; }}

QFrame#barra_lateral {{
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 {topo}, stop:1 {topo_2});
    border: none;
    border-right: 1px solid {borda};
}}
QLabel#marca {{ color: {topo_texto}; font-size: 13pt; font-weight: 700; }}
QLabel#logo {{
    color: {sobre_destaque};
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                                stop:0 {gradiente_a}, stop:1 {gradiente_b});
    border-radius: 7px;
    font-size: 15pt;
    font-weight: 800;
}}
QLabel#selo {{
    color: {sobre_destaque};
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                stop:0 {gradiente_a}, stop:1 {gradiente_b});
    border-radius: 9px;
    padding: 2px 9px;
    font-size: 8pt;
    font-weight: 700;
}}
QLabel#subtitulo {{ color: {topo_texto_suave}; font-size: 9pt; padding: 4px 2px; }}
QLabel#secao_nav {{
    color: {topo_texto_suave};
    font-size: 8pt;
    font-weight: 700;
    padding: 8px 10px 4px 10px;
}}
QLabel#rodape_nav {{ color: {topo_texto_suave}; font-size: 8pt; padding: 8px 10px; }}
QPushButton#nav {{
    text-align: left;
    background: transparent;
    color: {topo_texto_suave};
    border: none;
    border: 1px solid transparent;
    border-radius: 7px;
    padding: 6px 10px;
    min-height: 26px;
    font-size: 10pt;
    font-weight: 500;
}}
QPushButton#nav:hover {{ background: {nav_hover}; color: {topo_texto}; }}
QPushButton#nav:checked {{
    background: {destaque_fundo};
    color: {texto_maximo};
    border: 1px solid {borda};
    font-weight: 600;
}}
QPushButton#nav:focus {{ border: 1px solid {destaque}; }}

QWidget#area {{ background: {fundo}; }}
QMainWindow::separator {{ width: 0px; height: 0px; }}
QDockWidget#doca_da_barra {{ border: none; }}
QFrame#cabecalho_da_pagina {{ background: transparent; border: none; }}
QLabel#contexto_da_pagina {{ color: {texto_suave}; font-size: 8pt; font-weight: 600; }}
QLabel#titulo_da_pagina {{ color: {texto}; font-size: 19pt; font-weight: 650; }}
QLabel#descricao_da_pagina {{ color: {texto_suave}; font-size: 10pt; }}
QLabel#icone_da_pagina {{ background: {destaque_fundo}; border-radius: 16px; }}
QLabel#indice_da_pagina {{
    color: {destaque_texto}; background: {destaque_fundo};
    border-radius: 10px; padding: 6px 12px; font-size: 9pt; font-weight: 600;
}}

QTabWidget#abas::pane {{ border: none; background: transparent; }}
QTabWidget#abas > QStackedWidget {{ background: transparent; }}

QGroupBox {{
    background: {cartao};
    border: 1px solid {borda};
    border-radius: 10px;
    margin-top: 0px;
    padding: 30px 8px 8px 8px;
    font-size: 10pt;
    font-weight: 700;
}}
QGroupBox QGroupBox {{
    background: {cartao_interno};
    border-radius: 8px;
    padding: 23px 1px 2px 1px;
}}
QGroupBox::title {{
    subcontrol-origin: border;
    subcontrol-position: top left;
    top: 10px;
    left: 16px;
    padding: 0 2px;
    color: {texto};
    background: transparent;
}}
QGroupBox QGroupBox::title {{ color: {destaque_texto}; }}
QFrame#secao QGroupBox {{
    background: {cartao_interno};
    border-radius: 11px;
    padding: 23px 1px 2px 1px;
}}
QFrame#secao QGroupBox::title {{ color: {destaque_texto}; }}
QGroupBox#sem_moldura {{ background: transparent; border: none; padding: 0px; }}
QGroupBox#barra_de_execucao {{ border-color: {borda_forte}; }}
QGroupBox#faixa_de_trabalhos {{ padding: 0px; border-radius: 12px; }}
QLabel#titulo_da_faixa {{ font-weight: 700; color: {texto}; }}
QFrame#estado_vazio {{
    background: {cartao};
    border: 1px solid {borda};
    border-radius: 18px;
}}
QLabel#texto_do_vazio {{ color: {texto_suave}; font-size: 10.5pt; }}
QLabel#faixa_beta {{
    background: {aviso_fundo};
    color: {aviso};
    border-radius: 10px;
    padding: 8px 14px;
}}
QFrame#aviso_do_estudo {{
    background: {cartao_interno}; border: 1px solid {borda_forte};
    border-left: 3px solid {nota_info}; border-radius: 4px;
}}
QFrame#aviso_do_estudo[tipo="aviso"] {{
    background: {nota_aviso_fundo}; border-color: {nota_aviso_borda};
    border-left-color: {nota_aviso};
}}
QFrame#aviso_do_estudo[tipo="erro"] {{
    background: {nota_erro_fundo}; border-color: {nota_erro_borda}; border-left-color: {nota_erro};
}}
QFrame#aviso_do_estudo[tipo="sucesso"] {{ border-left-color: {nota_sucesso}; }}
QLabel#titulo_do_aviso {{ color: {texto}; font-size: 13px; font-weight: 600; }}
QLabel#rotulo_de_desfecho {{ color: {texto_suave}; font-size: 12px; padding: 0; }}
QToolButton#fechar_aviso {{
    background: transparent; border: none; border-radius: 3px;
    min-width: 0; min-height: 0; padding: 0;
}}
QToolButton#fechar_aviso:hover {{ background: {cartao}; }}
QToolButton#fechar_aviso:focus {{ border: 1px solid {destaque}; }}
QFrame#faixa_de_aviso {{
    border: 1px solid {borda_forte}; border-left: 3px solid {borda_forte}; border-radius: 4px;
}}
QFrame#faixa_de_aviso[nivel="aviso"] {{
    background: {nota_aviso_fundo}; border-color: {nota_aviso_borda};
    border-left-color: {nota_aviso};
}}
QFrame#faixa_de_aviso[nivel="critico"] {{
    background: {nota_erro_fundo}; border-color: {nota_erro_borda}; border-left-color: {nota_erro};
}}
QLabel#titulo_da_faixa_de_aviso {{ font-weight: 600; }}
QLabel#motivo_do_botao {{ color: {texto_suave}; font-size: 8.5pt; }}

QLabel#rotulo_do_campo {{ color: {texto_suave}; font-weight: 600; }}
QLabel#aviso_do_fecho {{ color: {texto_suave}; font-size: 8pt; }}

QFrame#secao {{
    background: {cartao};
    border: 1px solid {borda};
    border-radius: 10px;
}}
QLabel#numero_da_secao {{
    background: {destaque_fundo};
    color: {destaque_texto};
    border-radius: 13px;
    font-weight: 800;
}}
QLabel#titulo_da_secao {{ font-size: 12pt; font-weight: 600; color: {texto}; }}
QLabel#subtitulo_da_secao {{ color: {texto_suave}; }}
QFrame#divisor {{ background: {borda}; border: none; }}
QLabel#nota_do_campo {{ color: {texto_suave}; font-size: 8.5pt; }}
QLabel#campo_de_arquivo {{
    background: {campo};
    border: 1px solid {borda_forte};
    border-radius: 10px;
    min-height: 22px;
    padding: 5px 12px;
    color: {texto_suave};
}}
QLabel#campo_de_arquivo[preenchido="true"] {{ color: {texto}; border-color: {sucesso}; }}
QLabel#selo_de_estado {{
    border-radius: 10px;
    padding: 2px 10px;
    font-size: 8.5pt;
    font-weight: 700;
}}
QLabel#selo_de_estado[tipo="neutro"] {{
    background: {campo};
    color: {texto_suave};
    border: 1px solid {borda};
}}
QLabel#selo_de_estado[tipo="sucesso"] {{ background: {sucesso_fundo}; color: {sucesso}; }}
QLabel#selo_de_estado[tipo="aviso"] {{ background: {aviso_fundo}; color: {aviso}; }}
QLabel#selo_de_estado[tipo="erro"] {{ background: {perigo_fundo}; color: {perigo}; }}
QLabel#selo_de_estado[tipo="info"] {{ background: {destaque_fundo}; color: {destaque_texto}; }}
QFrame#cartao {{
    background: {cartao_interno};
    border: 1px solid {borda};
    border-radius: 14px;
}}
QLabel#titulo_do_cartao {{ font-size: 11pt; font-weight: 700; color: {texto}; }}
QLabel#icone_do_cartao {{ background: {destaque_fundo}; border-radius: 7px; }}
QFrame#cartao_de_arquivo {{
    background: {cartao}; border: 1px solid {borda}; border-radius: 8px;
}}
QFrame#cartao_de_arquivo:hover {{ border-color: {borda_forte}; }}
QLabel#icone_do_arquivo {{ color: {texto_suave}; background: transparent; }}
QLabel#titulo_do_arquivo {{ color: {texto}; font-size: 11pt; font-weight: 600; }}
QLabel#descricao_do_arquivo {{ color: {texto_suave}; font-size: 9pt; }}
QLabel#obrigatoriedade_do_arquivo {{
    color: {texto_suave}; font-size: 8pt; font-weight: 500;
}}
QLabel#obrigatoriedade_do_arquivo[obrigatorio="true"] {{ color: {texto}; }}
QFrame#cartao_de_arquivo QLabel#campo_de_arquivo {{
    border: none; background: transparent; padding: 2px 0; font-size: 8.5pt;
}}
QLabel#manchete {{ font-size: 13pt; font-weight: 700; color: {texto}; }}
QLabel#manchete[vazia="true"] {{ color: {desabilitado}; }}
QFrame#detalhes {{ background: {campo}; border: 1px solid {borda}; border-radius: 8px; }}
QFrame#recolhivel {{
    background: {cartao};
    border: 1px solid {borda};
    border-radius: 12px;
}}
QToolButton#alterna {{
    background: transparent;
    border: none;
    color: {destaque_texto};
    font-weight: 600;
    padding: 2px 0;
}}
QToolButton#alterna:hover {{ color: {texto}; }}
QPushButton#item_do_recorte {{
    text-align: left;
    background: transparent;
    border: none;
    padding: 6px 4px;
    font-weight: 600;
    color: {texto};
}}
QPushButton#item_do_recorte:hover {{ color: {destaque_texto}; }}
QPushButton#item_do_recorte:checked {{
    background: transparent;
    border: none;
    color: {destaque_texto};
}}
QLabel#corpo_do_recorte {{ color: {texto_suave}; padding: 0 4px 8px 26px; }}
QFrame#barra_de_trabalhos {{
    background: {cartao};
    border: 1px solid {borda};
    border-radius: 12px;
}}
QLabel#lista_de_trabalhos {{ color: {texto_suave}; }}

QPushButton {{
    background: {cartao_interno};
    color: {texto};
    border: 1px solid {borda_forte};
    border-radius: 10px;
    min-height: 26px;
    padding: 4px 14px;
    font-weight: 500;
}}
QPushButton:hover {{ border-color: {destaque}; color: {destaque_texto}; }}
QPushButton:pressed {{ background: {destaque_fundo}; }}
QPushButton:checked {{
    background: {destaque_fundo};
    border-color: {destaque};
    color: {destaque_texto};
}}
QPushButton:focus {{ border-color: {destaque}; }}
QPushButton:disabled {{
    background: {desabilitado_fundo};
    color: {desabilitado};
    border-color: {borda};
}}
QPushButton[papel="primario"] {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                stop:0 {gradiente_a}, stop:1 {gradiente_b});
    color: {sobre_destaque};
    border: none;
    padding: 5px 18px;
    font-weight: 600;
}}
QPushButton[papel="primario"]:hover {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                stop:0 {destaque}, stop:1 {gradiente_b});
    color: {sobre_destaque};
}}
QPushButton[papel="primario"]:pressed {{ background: {gradiente_b}; }}
QPushButton[papel="primario"]:disabled {{
    background: {desabilitado_fundo};
    color: {desabilitado};
    border: 1px solid {borda};
}}
QPushButton[papel="perigo"] {{
    color: {perigo};
    border-color: {perigo};
    background: transparent;
}}
QPushButton[papel="perigo"]:hover {{ background: {perigo_fundo}; color: {perigo}; }}
QPushButton[papel="perigo"]:disabled {{
    background: {desabilitado_fundo};
    color: {desabilitado};
    border-color: {borda};
}}

QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QPlainTextEdit, QTextEdit {{
    background: {campo};
    color: {texto};
    border: 1px solid {borda_forte};
    border-radius: 10px;
    padding: 5px 9px;
    min-height: 22px;
    selection-background-color: {gradiente_b};
    selection-color: {sobre_destaque};
}}
QLineEdit:hover, QComboBox:hover, QSpinBox:hover, QDoubleSpinBox:hover {{
    border-color: {texto_suave};
}}
QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus,
QPlainTextEdit:focus, QTextEdit:focus {{ border: 1px solid {destaque}; }}
QPlainTextEdit#caixa_do_resumo {{ min-height: 148px; }}
QLineEdit:disabled, QComboBox:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled {{
    background: {desabilitado_fundo};
    color: {desabilitado};
    border-color: {borda};
}}
QSpinBox::up-button, QDoubleSpinBox::up-button,
QSpinBox::down-button, QDoubleSpinBox::down-button {{
    subcontrol-origin: border;
    width: 16px;
    border: none;
    border-left: 1px solid {borda};
    background: transparent;
}}
QSpinBox::up-button, QDoubleSpinBox::up-button {{ subcontrol-position: top right; }}
QSpinBox::down-button, QDoubleSpinBox::down-button {{ subcontrol-position: bottom right; }}
QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {{ background: {destaque_fundo}; }}
QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {{ image: url({seta_cima}); width: 9px; height: 9px; }}
QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {{
    image: url({seta_baixo}); width: 9px; height: 9px;
}}
QComboBox {{
    combobox-popup: 0;
    background: {cartao_interno};
    border-radius: 10px;
    padding: 5px 10px;
    min-height: 22px;
}}
QComboBox:editable {{ background: {campo}; }}
QComboBox:on {{ border-color: {destaque}; }}
QComboBox::drop-down {{
    subcontrol-origin: padding;
    subcontrol-position: center right;
    width: 26px;
    border: none;
}}
QComboBox::down-arrow {{ image: url({seta_baixo}); width: 11px; height: 11px; }}
QComboBox::down-arrow:on {{ image: url({seta_cima}); }}
QComboBoxPrivateContainer {{ background: transparent; border: none; }}
QComboBox QAbstractItemView {{
    background: {cartao_interno};
    border: 1px solid {borda_forte};
    border-radius: 10px;
    padding: 6px;
    outline: none;
    selection-background-color: {destaque_fundo};
    selection-color: {destaque_texto};
}}
QComboBox QAbstractItemView::item {{
    min-height: 30px;
    padding: 0px 10px;
    border-radius: 6px;
    color: {texto};
}}
QComboBox QAbstractItemView::item:hover {{ background: {destaque_fundo}; color: {texto}; }}
QComboBox QAbstractItemView::item:selected {{
    background: {destaque_fundo};
    color: {destaque_texto};
}}

QCheckBox, QRadioButton {{ spacing: 8px; background: transparent; }}
QCheckBox::indicator {{
    width: 16px;
    height: 16px;
    border: 1px solid {borda_forte};
    border-radius: 5px;
    background: {campo};
}}
QCheckBox::indicator:hover {{ border-color: {destaque}; }}
QCheckBox::indicator:checked {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                                stop:0 {gradiente_a}, stop:1 {gradiente_b});
    border-color: {destaque};
    image: url({visto});
}}
QCheckBox::indicator:disabled {{ background: {desabilitado_fundo}; border-color: {borda}; }}
QCheckBox::indicator:checked:disabled {{
    background: {desabilitado};
    border-color: {desabilitado};
}}

QListWidget, QTableWidget, QTableView, QTreeView, QListView {{
    background: {campo};
    alternate-background-color: {cartao};
    border: 1px solid {borda};
    border-radius: 8px;
    gridline-color: {borda};
    selection-background-color: {destaque_fundo};
    selection-color: {texto};
}}
QHeaderView::section {{
    background: {cartao_interno};
    color: {texto_suave};
    border: none;
    border-bottom: 1px solid {borda};
    padding: 6px 8px;
    font-weight: 600;
}}
QTableCornerButton::section {{ background: {cartao_interno}; border: none; }}

QProgressBar {{
    background: {campo};
    border: 1px solid {borda};
    border-radius: 8px;
    text-align: center;
    color: {texto};
    min-height: 18px;
}}
QProgressBar::chunk {{
    background: {progresso};
    border-radius: 6px;
}}

QScrollBar:vertical {{ background: transparent; width: 10px; margin: 1px; }}
QScrollBar:horizontal {{ background: transparent; height: 12px; margin: 2px; }}
QScrollBar::handle:vertical {{ background: {rolagem}; border-radius: 4px; min-height: 32px; }}
QScrollBar::handle:horizontal {{ background: {rolagem}; border-radius: 4px; min-width: 32px; }}
QScrollBar::handle:hover {{ background: {borda_forte}; }}
QScrollBar::handle:vertical:disabled {{ background: transparent; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; border: none; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}

QToolTip {{
    background: {cartao_interno};
    color: {texto};
    border: 1px solid {borda_forte};
    border-radius: 6px;
    padding: 6px 8px;
}}

QMenu {{ background: {cartao}; border: 1px solid {borda}; padding: 4px; }}
QMenu::item {{ padding: 6px 18px; border-radius: 4px; }}
QMenu::item:selected {{ background: {destaque_fundo}; color: {texto}; }}

/* Medidas e hierarquia da referência fornecida para o preparo do estudo. */
QFrame#secao[referencia="true"] {{
    background: {cartao}; border-color: {ref_borda}; border-radius: 6px;
}}
QFrame#secao[referencia="true"] QLabel#titulo_da_secao {{ font-size: 16px; font-weight: 700; }}
QFrame#secao[referencia="true"] QLabel#subtitulo_da_secao {{
    color: {ref_texto_suave}; font-size: 13px;
}}
QFrame#secao[referencia="true"] QLabel#numero_da_secao {{
    background: {cartao_interno}; border-radius: 16px; font-size: 14px; font-weight: 700;
}}
QLabel#selo_de_estado[contagem_arquivos="true"] {{
    background: transparent; border: 1px solid {ref_divisoria}; border-radius: 3px;
    color: {ref_texto_suave}; font-family: "IBM Plex Mono"; font-size: 11px;
    font-weight: 400; padding: 2px 8px; min-height: 14px; min-width: 130px;
}}
QLabel#rotulo_do_modo {{ font-size: 13px; font-weight: 600; }}
QFrame#faixaModos {{ background: {ref_superficie}; border: none; border-radius: 6px; }}
QPushButton#segmentoModo {{
    background: transparent; color: {segmento_texto}; border: none; border-radius: 4px;
    min-height: 24px; padding: 2px 8px; font-family: "IBM Plex Mono";
    font-size: 11px; font-weight: 400;
}}
QPushButton#segmentoModo:hover {{ background: {borda}; color: {segmento_hover_texto}; }}
QPushButton#segmentoModo:checked {{
    background: {invertido}; color: {segmento_marcado_texto}; border: none;
}}
QPushButton#segmentoModo:focus {{ border: 1px solid {foco}; }}
/* O seletor da entrada das etapas: o trilho e os segmentos do modo, com o texto da prosa. */
QFrame#faixaDeEntradas {{ background: {ref_superficie}; border: none; border-radius: 6px; }}
QPushButton#segmentoDaEntrada {{
    background: transparent; color: {segmento_texto}; border: none; border-radius: 4px;
    min-height: 26px; padding: 2px 12px;
}}
QPushButton#segmentoDaEntrada:hover {{ background: {borda}; color: {segmento_hover_texto}; }}
QPushButton#segmentoDaEntrada:checked {{
    background: {invertido}; color: {segmento_marcado_texto}; border: none;
}}
QPushButton#segmentoDaEntrada:focus {{ border: 1px solid {foco}; }}
QFrame#cartao_de_arquivo {{
    background: {cartao}; border: 1px solid {ref_borda}; border-radius: 4px;
}}
QLabel#titulo_do_arquivo {{ font-size: 14px; font-weight: 700; }}
QLabel#descricao_do_arquivo {{ color: {texto_apagado}; font-size: 12px; }}
QLabel#obrigatoriedade_do_arquivo {{ color: {texto_apagado}; font-size: 9px; font-weight: 400; }}
QLabel#obrigatoriedade_do_arquivo[obrigatorio="true"] {{
    color: {obrigatorio_texto}; font-weight: 700;
}}
QPushButton#seleciona_arquivo {{
    background: transparent; border: none; border-radius: 0;
    padding: 0; min-height: 18px; font-size: 12px; font-weight: 600;
    text-align: left;
}}
QPushButton#seleciona_arquivo:hover {{ color: {texto_maximo}; }}
QPushButton#seleciona_arquivo:focus {{ border: 1px solid {foco}; }}
QPushButton#remove_arquivo {{
    background: transparent; border: none; border-radius: 4px;
    color: {acao_suave}; font-size: 12px; font-weight: 400;
    padding: 0 2px; min-height: 18px;
}}
QPushButton#remove_arquivo:hover {{ color: {perigo}; background: {perigo_fundo}; }}
QPushButton#remove_arquivo:focus {{ border: 1px solid {perigo}; }}
QFrame#cartao[referencia="true"] {{
    background: {cartao}; border: 1px solid {ref_borda}; border-radius: 4px;
}}
QFrame#cartao[referencia="true"] QLabel#icone_do_cartao {{
    background: {ref_superficie}; border-radius: 6px;
}}
QFrame#cartao[referencia="true"] QLabel#titulo_do_cartao {{ font-size: 14px; font-weight: 700; }}
QFrame#cartao[referencia="true"] QLabel#nota_do_campo {{
    color: {ref_texto_suave}; font-size: 12px;
}}
QFrame#cartao[referencia="true"] QLabel#selo_de_estado {{
    border: none; background: transparent; padding: 0; font-size: 11px; font-weight: 400;
}}
QFrame#cartao[referencia="true"] QLabel#selo_de_estado[tipo="neutro"] {{
    color: {ref_texto_suave};
}}
QLabel#ponto_do_estado {{ color: {texto_apagado}; font-size: 10px; }}
QLabel#ponto_do_estado[tipo="sucesso"] {{ color: {sucesso}; }}
QLabel#ponto_do_estado[tipo="erro"] {{ color: {perigo}; }}
QLabel#ponto_do_estado[tipo="aviso"] {{ color: {aviso}; }}
QLabel#ponto_do_estado[tipo="info"] {{ color: {destaque_texto}; }}
QPushButton#executa_leitura {{
    background: {invertido}; color: {sobre_invertido}; border: none; border-radius: 5px;
    font-size: 12px; font-weight: 500; min-height: 20px; padding: 4px 14px;
}}
QPushButton#executa_leitura:hover {{ background: {invertido_hover}; color: {sobre_invertido}; }}
QPushButton#executa_leitura:disabled {{ background: {borda_forte}; color: {texto_apagado}; }}
QFrame#cartao[referencia="true"] QToolButton#alterna {{
    color: {ref_texto_suave}; font-size: 12px; font-weight: 400; padding: 0; min-height: 28px;
}}
QFrame#recolhivel[referencia="true"] {{ background: transparent; border: none; border-radius: 0; }}
QFrame#recolhivel[referencia="true"] QToolButton#alterna {{
    color: {texto_apagado}; font-size: 13px; font-weight: 400; padding: 0;
}}
QFrame#detalhe_da_leitura {{
    background: {superficie_funda}; border: 1px solid {detalhe_borda}; border-radius: 2px;
}}
QLabel#titulo_do_detalhe {{ font-size: 14px; font-weight: 600; color: {texto_forte}; }}
QLabel#explicacao_da_leitura {{ font-size: 12px; color: {ref_texto_suave}; }}
QLabel#resultado_da_ficha {{ font-size: 12px; color: {texto}; }}
QWidget#grade_da_legenda {{ background: {superficie_funda}; border: 1px solid {rolagem}; }}
QLabel#introducao_da_legenda {{ color: {legenda_introducao}; font-size: 13px; }}
QFrame#classe_da_legenda {{
    background: transparent; border: none; border-top: 1px solid {ref_divisoria};
    border-radius: 0;
}}
QFrame#classe_da_legenda QToolButton#alterna {{
    color: {texto_forte}; font-size: 14px; font-weight: 600; padding: 0;
}}
QLabel#pergunta_da_legenda {{ color: {texto_forte}; font-size: 12px; font-weight: 600; }}
QLabel#descricao_da_legenda {{ color: {ref_texto_suave}; font-size: 12px; }}

/* Tokens e medidas dos arquivos originais do Lovable (styles.css / ui-bits). */
QLabel#marca {{ font-size: 14px; font-weight: 600; }}
QLabel#logo {{ background: transparent; border: none; border-radius: 0; }}
QLabel#secao_nav {{
    font-size: 10px; color: {texto_suave}; padding: 8px 12px;
    border-bottom: 1px solid {borda};
}}
QLabel#rodape_nav {{
    font-size: 10px; padding: 12px 10px 0 10px; border-top: 1px solid {borda};
}}
QPushButton#nav {{
    font-size: 13px; font-weight: 400; border-radius: 2px;
    min-height: 18px; padding: 8px 10px; border: 1px solid transparent;
}}
QPushButton#nav:checked {{ background: {destaque_fundo}; border-color: {borda}; }}
QLabel#beta_nav {{
    color: {beta}; background: {beta_fundo}; border: 1px solid {beta_borda};
    font-size: 9px; font-weight: 700; border-radius: 4px; padding: 1px 5px;
}}
QPushButton#gaveta_nav {{
    font-size: 13px; font-weight: 600; border-radius: 2px; text-align: left;
    min-height: 18px; padding: 8px 10px; border: 1px solid transparent;
}}
QWidget#gaveta {{ background: transparent; }}
QFrame#cabecalho_da_pagina {{ border-bottom: 1px solid {borda}; border-radius: 0; }}
QLabel#titulo_da_pagina {{ font-size: 20px; font-weight: 600; }}
QLabel#descricao_da_pagina {{ font-size: 12px; color: {texto_suave}; }}
QWidget[folhaDeEtapa="true"] {{
    background: {cartao}; border: 1px solid {borda}; border-radius: 2px;
}}
QFrame#secao[painelPlano="true"], QFrame#secao[painel_plano="true"] {{
    background: transparent; border: none; border-bottom: 1px solid {borda}; border-radius: 0;
}}
QGroupBox[painelPlano="true"] {{
    background: transparent; border: none; border-bottom: 1px solid {borda}; border-radius: 0;
    margin: 0; padding: 0; font-size: 16px; font-weight: 600;
}}
QGroupBox[painelPlano="true"]::title {{
    subcontrol-origin: border; subcontrol-position: top left; top: 0; left: 0;
    padding: 0; color: {texto};
}}
QGroupBox[painelPlano="true"][ultimoPainel="true"] {{ border-bottom: none; }}
QGroupBox#barra_de_execucao[painelPlano="true"] {{ border: none; padding: 0; margin: 0; }}
QLabel[rotuloFormulario="true"] {{ color: {texto_suave}; font-size: 13px; font-weight: 400; }}
QPushButton[divulgacao="true"] {{
    background: transparent; border: none; color: {texto_suave};
    font-size: 13px; padding: 0; min-height: 24px;
}}
QPushButton {{ border-radius: 2px; min-height: 20px; padding: 6px 12px; font-size: 13px; }}
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {{
    background: {campo}; border: 1px solid {borda_forte}; border-radius: 2px;
    min-height: 20px; padding: 5px 10px; font-size: 13px;
}}
QLineEdit:hover, QComboBox:hover, QSpinBox:hover, QDoubleSpinBox:hover {{
    border-color: {campo_hover_borda};
}}
QComboBox QAbstractItemView {{ border-radius: 4px; background: {cartao}; padding: 4px; }}
QComboBox QAbstractItemView::item {{ min-height: 28px; border-radius: 2px; font-size: 13px; }}
QCheckBox {{ spacing: 8px; font-size: 12px; }}
QCheckBox::indicator {{ width: 16px; height: 16px; border-radius: 2px; }}
QGroupBox#tema_da_qualidade {{
    background: transparent; border: none; border-bottom: 1px solid {borda};
    border-radius: 0; padding: 0; margin: 0; font-size: 16px; font-weight: 600;
}}
QGroupBox#tema_da_qualidade::title {{
    subcontrol-origin: border; subcontrol-position: top left; left: 0; top: 0; padding: 0;
}}
QGroupBox#metrica_da_qualidade {{
    background: {cartao}; border: 1px solid {borda}; border-radius: 2px; padding: 0; margin: 0;
}}
QGroupBox#metrica_da_qualidade::title {{ color: transparent; width: 0; height: 0; padding: 0; }}
QLabel#titulo_da_metrica {{ font-size: 14px; font-weight: 500; }}
QLabel#valor_da_metrica {{ color: {texto}; font-family: "IBM Plex Mono"; }}
QLabel#legenda_da_metrica {{ color: {texto_suave}; font-size: 12px; }}
QFrame#secao[referencia="true"] {{ border-radius: 2px; }}
QFrame#secao[referencia="true"] QLabel#numero_da_secao {{ border-radius: 2px; font-size: 12px; }}
QFrame#secao[referencia="true"] QLabel#titulo_da_secao {{ font-size: 16px; font-weight: 600; }}
QFrame#secao[referencia="true"] QLabel#subtitulo_da_secao {{ font-size: 12px; }}
QLabel#titulo_do_arquivo {{ font-size: 16px; font-weight: 600; }}
QFrame#cartao[referencia="true"] QLabel#titulo_do_cartao {{ font-size: 16px; font-weight: 600; }}
QFrame#cartao_de_arquivo, QFrame#cartao[referencia="true"] {{ border-radius: 2px; }}
QFrame#cartao[referencia="true"] QLabel#icone_do_cartao {{ border-radius: 2px; }}
QLabel#ponto_do_estado {{ background: {texto_suave}; border-radius: 3px; }}
QLabel#ponto_do_estado[tipo="sucesso"] {{ background: {sucesso}; }}
QLabel#ponto_do_estado[tipo="erro"] {{ background: {perigo}; }}
QLabel#ponto_do_estado[tipo="aviso"] {{ background: {aviso}; }}
QLabel#ponto_do_estado[tipo="info"] {{ background: {destaque_texto}; }}
QMenuBar {{ background: {cartao}; border-bottom: 1px solid {borda}; padding: 3px 8px; }}
QMenuBar::item {{ color: {texto_suave}; padding: 5px 10px; font-size: 12px; border-radius: 2px; }}
QMenuBar::item:selected {{ color: {texto}; background: {cartao_interno}; }}
QFrame#faixa_trabalhos_lateral {{
    background: transparent; border: none; border-top: 1px solid {borda}; border-radius: 0;
}}
QToolButton#alterna_dos_trabalhos {{
    color: {topo_texto}; background: transparent; border: none; border-radius: 2px;
    font-size: 12px; font-weight: 600; padding: 4px 2px; text-align: left;
}}
QToolButton#alterna_dos_trabalhos:hover {{ background: {destaque_fundo}; }}
QGroupBox#faixa_de_trabalhos {{
    background: transparent; border: none; border-radius: 0; padding: 0;
}}
QGroupBox#faixa_de_trabalhos QLabel {{ font-size: 11px; }}
QGroupBox#faixa_de_trabalhos QPushButton {{
    font-size: 11px; min-height: 18px; padding: 3px 8px;
    background: transparent; border-radius: 2px;
}}
QGroupBox#faixa_de_trabalhos QComboBox {{
    font-size: 11px; min-height: 18px; padding: 2px 8px; border-radius: 2px;
}}
QFrame#secao QGroupBox#barra_de_execucao[painelPlano="true"] {{
    background: transparent; border: none; border-radius: 0; padding: 0; margin: 0;
}}
QFrame#secao[ultimoPainel="true"] {{ border-bottom: none; }}
/* A sugestão ao lado de «Seguir sugestões do app», o «?» e o quadro que ele abre. O «?» é
   pequeno: um círculo de contorno que só ganha o texto pleno sob o cursor ou o foco. O quadro
   tem o fundo e a borda da dica de mouse, e dentro dele as molduras dos pontos de decisão ficam
   planas, separadas por uma linha, para não haver um cartão dentro do outro. */
QLabel#indicacaoDasSugestoes {{ color: {texto}; font-size: 13px; font-weight: 600; }}
QToolButton#botaoDoPorque {{
    background: transparent; color: {ref_texto_suave}; border: 1px solid {borda_forte};
    border-radius: 9px; padding: 0; font-size: 11px; font-weight: 600;
}}
QToolButton#botaoDoPorque:hover {{ color: {texto}; border-color: {texto_suave}; }}
QToolButton#botaoDoPorque:focus {{ color: {texto}; border-color: {foco}; }}
QFrame#quadroDaJustificativa {{
    background: {cartao_interno}; border: 1px solid {borda_forte}; border-radius: 2px;
}}
QFrame#quadroDaJustificativa QLabel {{ font-size: 12px; font-weight: 400; }}
QFrame#quadroDaJustificativa QLabel#tituloDaJustificativa {{
    color: {texto}; font-size: 13px; font-weight: 600;
}}
QFrame#quadroDaJustificativa QGroupBox {{
    background: transparent; border: none; border-top: 1px solid {borda}; border-radius: 0;
    margin: 0; padding: 26px 0 0 0; font-size: 12px; font-weight: 600;
}}
QFrame#quadroDaJustificativa QGroupBox::title {{
    subcontrol-origin: border; subcontrol-position: top left; top: 8px; left: 0; padding: 0;
    color: {texto};
}}
/* Os grupos de parâmetros dentro da seção da etapa: o mesmo painel plano, um nível abaixo do
   título da seção. Sem esta regra valeria `QFrame#secao QGroupBox`, que tem o id e por isso é mais
   específica que a do painel plano: o fundo e o raio dos cartões internos. */
QFrame#secao QGroupBox[painelPlano="true"] {{
    background: transparent; border: none; border-bottom: 1px solid {borda}; border-radius: 0;
    margin: 0; padding: 0; font-size: 14px; font-weight: 600;
}}
QFrame#secao QGroupBox[painelPlano="true"]::title {{ color: {texto}; }}
QLabel#subtitulo_da_etapa {{ font-size: 14px; font-weight: 600; color: {texto}; }}

/* O controle do esquema, na barra lateral: dois segmentos, o marcado como a navegação. */
QFrame#seletor_de_esquema {{
    background: {campo}; border: 1px solid {borda}; border-radius: 4px;
}}
QPushButton#opcao_de_esquema {{
    background: transparent; color: {topo_texto_suave}; border: 1px solid transparent;
    border-radius: 2px; min-height: 18px; padding: 4px 8px; font-size: 12px; font-weight: 400;
}}
QPushButton#opcao_de_esquema:hover {{ color: {topo_texto}; }}
QPushButton#opcao_de_esquema:checked {{
    background: {destaque_fundo}; color: {texto_maximo}; border-color: {borda}; font-weight: 600;
}}
QPushButton#opcao_de_esquema:focus {{ border-color: {foco}; }}
"""


SETAS: Final = ("seta-baixo.svg", "seta-cima.svg", "visto.svg", "visto-preto.svg")


def pasta_de_recursos() -> Path:
    """Onde estão as setas: o mesmo desvio de `principal.caminho_do_icone`, pela mesma razão.

    Congelado, o PyInstaller expande `recursos/` para `sys._MEIPASS` com o destino declarado em
    `refino-desktop.spec`, que já leva a pasta inteira: as setas viajam sem lista nova.
    """
    if getattr(sys, "frozen", False):
        return Path(getattr(sys, "_MEIPASS", "")) / "refino" / "desktop" / "recursos"
    return Path(__file__).resolve().parents[1] / "recursos"


@cache
def folha_de_estilo(esquema: Esquema = ESQUEMA_PADRAO) -> str:
    """A folha de estilo montada para um esquema, em cache: montada uma vez por processo.

    Levanta `KeyError` em esquema desconhecido em vez de recuar num padrão, que pintaria a janela
    errada sem nada acusar.

    As setas são arquivos porque, quando a folha estiliza um `QComboBox` ou um `QSpinBox`, o Qt
    deixa de desenhar a seta nativa e exige uma `image:` para ela, e o QSS não aceita SVG embutido
    em `data:`. São `.svg` pequenos em `recursos/`: as duas setas e o visto da caixa de seleção (em
    branco e em preto), que pela mesma regra perde a moldura e fica invisível quando desmarcada. O
    caminho vai com barras `/`, que é o que o QSS lê em qualquer sistema.
    """
    pasta = pasta_de_recursos()
    baixo, cima, visto_branco, visto_preto = (
        str(pasta / nome).replace("\\", "/") for nome in SETAS
    )
    visto = visto_preto if esquema == "escuro" else visto_branco
    return _MODELO.format(
        **asdict(_POR_ESQUEMA[esquema]), seta_baixo=baixo, seta_cima=cima, visto=visto
    )


def paleta(cores: Cores) -> QPalette:
    """A `QPalette` derivada dos mesmos tokens da folha.

    A folha cobre os widgets da janela; a paleta cobre o que ela não alcança: a cor de seleção de
    texto, os diálogos do sistema (`QFileDialog`, `QMessageBox`) e qualquer widget que desenhe
    direto por `palette()`. Sem ela, um diálogo aberto no tema escuro nasceria com o fundo claro do
    Fusion.
    """
    p = QPalette()
    papeis = {
        QPalette.ColorRole.Window: cores.fundo,
        QPalette.ColorRole.WindowText: cores.texto,
        QPalette.ColorRole.Base: cores.campo,
        QPalette.ColorRole.AlternateBase: cores.cartao,
        QPalette.ColorRole.Text: cores.texto,
        QPalette.ColorRole.Button: cores.cartao_interno,
        QPalette.ColorRole.ButtonText: cores.texto,
        QPalette.ColorRole.Highlight: cores.gradiente_b,
        QPalette.ColorRole.HighlightedText: cores.sobre_destaque,
        QPalette.ColorRole.ToolTipBase: cores.cartao_interno,
        QPalette.ColorRole.ToolTipText: cores.texto,
        QPalette.ColorRole.PlaceholderText: cores.texto_suave,
        QPalette.ColorRole.Link: cores.destaque_texto,
        QPalette.ColorRole.Mid: cores.borda,
        QPalette.ColorRole.Dark: cores.borda_forte,
    }
    for papel, cor in papeis.items():
        p.setColor(papel, QColor(cor))
    for papel in (
        QPalette.ColorRole.WindowText,
        QPalette.ColorRole.Text,
        QPalette.ColorRole.ButtonText,
    ):
        p.setColor(QPalette.ColorGroup.Disabled, papel, QColor(cores.desabilitado))
    return p


def _carrega_fontes(app: QApplication) -> None:
    """Registra as fontes da referência localmente, uma vez por aplicação.

    Os TTF e a licença acompanham os recursos do executável: a interface não
    depende de download nem de uma fonte instalada no computador do laboratório.
    """
    if app.property("refino_fontes_carregadas"):
        return
    pasta = pasta_de_recursos() / "fontes"
    for nome in (
        "IBMPlexSans-Regular.ttf",
        "IBMPlexSans-Medium.ttf",
        "IBMPlexSans-SemiBold.ttf",
        "IBMPlexSans-Bold.ttf",
        "IBMPlexMono-Regular.ttf",
        "IBMPlexMono-Medium.ttf",
    ):
        QFontDatabase.addApplicationFont(str(pasta / nome))
    app.setProperty("refino_fontes_carregadas", True)


def veste_a_aplicacao(app: QApplication, esquema: Esquema = ESQUEMA_PADRAO) -> Esquema:
    """Aplica estilo, fonte, paleta e folha. Chamar antes de construir a janela.

    Devolve o esquema aplicado, para quem chama poder registrá-lo. O corpo vai a
    `TAMANHO_DA_FONTE_DO_CORPO` pontos: 9 pt é pequeno para formulários lidos o dia inteiro, e com
    10 pt nenhuma aba ganha rolagem horizontal a 1280x760. A hierarquia tipográfica (títulos,
    marca, navegação) vive na folha, onde não custa largura de formulário.
    """
    app.setStyle("Fusion")
    _carrega_fontes(app)
    fonte = app.font()
    fonte.setFamilies(["IBM Plex Sans", "Segoe UI", "DejaVu Sans", "sans-serif"])
    fonte.setPointSizeF(TAMANHO_DA_FONTE_DO_CORPO)
    fonte.setStyleStrategy(QFont.StyleStrategy.PreferAntialias)
    app.setFont(fonte)
    _aplica_cores(app, esquema)
    return esquema


def _aplica_cores(app: QApplication, esquema: Esquema) -> None:
    """Aplica o esquema, a paleta e a folha: o que a troca de esquema refaz com a janela aberta.

    O esquema vai para a propriedade antes da folha, para um widget repolido pela folha nova que
    desenhe um ícone já ler a cor do esquema novo. `_POR_ESQUEMA[esquema]` levanta em esquema
    desconhecido antes de qualquer coisa mudar.
    """
    cores = _POR_ESQUEMA[esquema]
    app.setProperty(PROPRIEDADE_DO_ESQUEMA, esquema)
    app.setPalette(paleta(cores))
    app.setStyleSheet(folha_de_estilo(esquema))


def esquema_em_vigor() -> Esquema:
    """O esquema que a aplicação veste.

    Sem tema aplicado, é o padrão. Um valor que não é esquema levanta: recuar no padrão pintaria a
    janela errada sem nada acusar.
    """
    app = QApplication.instance()
    valor = app.property(PROPRIEDADE_DO_ESQUEMA) if app is not None else None
    if valor is None:
        return ESQUEMA_PADRAO
    if valor not in _POR_ESQUEMA:
        raise ValueError(f"esquema desconhecido na aplicação: {valor!r}")
    return cast("Esquema", valor)


def poe_icone(
    alvo: QLabel | QAbstractButton,
    tracos: str,
    cor: str,
    tamanho: int,
    *,
    tracos_ligado: str | None = None,
) -> None:
    """Desenha um ícone na cor de um token do esquema em vigor, e grava a receita no widget.

    A cor é um token, e não uma cor por extenso, porque o desenho é fixo: feito com uma cor do
    escuro, ele fica ilegível no claro (o `#aaaaaa` dá 2,32:1 sobre o cartão branco), e na troca de
    esquema não muda sozinho. Com o token, o ícone nasce no esquema em vigor, e a receita gravada
    deixa `troca_o_esquema` desenhá-lo de novo.

    O token tem de estar em `SUPERFICIE_DOS_ICONES`, conferido aqui: isso põe o ícone sob o teste
    de contraste, e um nome errado ou uma cor por extenso levanta em vez de desenhar. Ícone
    desenhado por outro caminho escapa desta conferência; quem o procura são os testes da janela
    vestida e do código em `tests/test_desktop_tema.py`.

    `tracos_ligado` dá ao botão marcável um segundo desenho para o estado ligado, na mesma cor.
    """
    if cor not in SUPERFICIE_DOS_ICONES:
        raise ValueError(
            f"cor de ícone sem superfície declarada: {cor!r}; as válidas são "
            f"{sorted(SUPERFICIE_DOS_ICONES)}"
        )
    tinta = getattr(_POR_ESQUEMA[esquema_em_vigor()], cor)
    if isinstance(alvo, QLabel):
        alvo.setPixmap(icone_nitido(tracos, tinta, tamanho))
    elif tracos_ligado is None:
        alvo.setIcon(QIcon(icone_nitido(tracos, tinta, tamanho)))
    else:
        figura = QIcon()
        figura.addPixmap(icone_nitido(tracos, tinta, tamanho), QIcon.Mode.Normal, QIcon.State.Off)
        figura.addPixmap(
            icone_nitido(tracos_ligado, tinta, tamanho), QIcon.Mode.Normal, QIcon.State.On
        )
        alvo.setIcon(figura)
    receita = {"tracos": tracos, "cor": cor, "tamanho": tamanho, "ligado": tracos_ligado or ""}
    alvo.setProperty(PROPRIEDADE_DO_ICONE, receita)


def _repinta_os_icones(raiz: QWidget) -> int:
    """Desenha de novo, no esquema em vigor, cada ícone gravado por `poe_icone`. Devolve quantos."""
    repintados = 0
    for widget in raiz.findChildren(QWidget):
        receita = widget.property(PROPRIEDADE_DO_ICONE)
        if receita and isinstance(widget, QLabel | QAbstractButton):
            poe_icone(
                widget,
                receita["tracos"],
                receita["cor"],
                int(receita["tamanho"]),
                tracos_ligado=receita["ligado"] or None,
            )
            repintados += 1
    return repintados


def troca_o_esquema(janela: QWidget, esquema: Esquema) -> int:
    """Troca o esquema com a janela aberta. Devolve quantos ícones gravados repintou.

    A folha e a paleta valem para a aplicação inteira, e o Qt repole cada widget sozinho. O que ele
    não refaz é o que foi desenhado uma vez: os ícones e a barra de título do Windows. São quatro
    passos, nesta ordem, com o esquema primeiro para o ícone desenhado no meio do repolimento já
    nascer na cor nova:
    - esquema, paleta e folha (`_aplica_cores`);
    - os ícones da barra lateral, e o botão do esquema marcado (`BarraLateral.pinta`);
    - todo ícone gravado por `poe_icone`;
    - a moldura do Windows, escura ou clara (`escurece_a_moldura`).

    As figuras do matplotlib não mudam: o fundo delas é branco fixo, nos dois esquemas.
    """
    app = QApplication.instance()
    if not isinstance(app, QApplication):  # pragma: no cover - a janela existe, então a aplicação
        raise RuntimeError("não há QApplication para trocar o esquema")
    _aplica_cores(app, esquema)
    barra = janela.findChild(BarraLateral)
    if barra is not None:
        barra.pinta(esquema)
    repintados = _repinta_os_icones(janela)
    escurece_a_moldura(janela, escura=esquema == "escuro")
    return repintados


def marca_papel(botao: QPushButton, papel: str) -> QPushButton:
    """Dá a um botão o papel `primario` ou `perigo`. Devolve o próprio botão.

    Levanta em papel fora de `PAPEIS`: a folha casa por igualdade de texto, e um papel digitado
    errado deixaria o botão com o estilo comum sem ninguém notar.

    O `polish` depois do `setProperty` é obrigatório: o Qt só reavalia seletores por propriedade
    quando o widget é repolido, e sem ele o papel marcado depois da construção não aparece.
    """
    if papel not in PAPEIS:
        raise ValueError(f"papel de botão desconhecido: {papel!r}; os válidos são {sorted(PAPEIS)}")
    botao.setProperty("papel", papel)
    estilo = botao.style()
    if estilo is not None:  # pragma: no branch - um QWidget construído sempre tem estilo
        estilo.unpolish(botao)
        estilo.polish(botao)
    return botao


# ---------------------------------------------------------------------------
# A navegação lateral
# ---------------------------------------------------------------------------
#
# Uma barra lateral, e não abas: abas no topo e caixas com o título na borda são o vocabulário de
# programa de desktop, e a janela usa o de aplicação web (navegação lateral com ícones, título
# grande de página e cartões com cabeçalho dentro).
#
# A barra lateral comanda o `QTabWidget`, não o substitui. A barra de abas é escondida e cada botão
# chama `setCurrentIndex`; o `currentChanged` devolve a marcação ao botão. Assim os testes que
# medem `janela.abas` continuam a medir a mesma coisa, e o atalho de `principal.py` que leva à aba
# dos gráficos por código (`setCurrentIndex(INDICE_DA_ABA_DOS_GRAFICOS)`) move a barra junto.

# O traço de cada ícone, no estilo de linha de 24 por 24 que as bibliotecas web usam. Desenhados à
# mão aqui, e não baixados: nenhuma licença a conferir, nenhum arquivo a viajar no executável.
ICONES: Final[dict[str, str]] = {
    "Preparar o estudo": (
        '<path d="M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/>'
    ),
    "Correção": (
        '<path d="M4 6h10M18 6h2M4 12h4M12 12h8M4 18h12M20 18h0"/>'
        '<circle cx="16" cy="6" r="2"/><circle cx="10" cy="12" r="2"/>'
        '<circle cx="18" cy="18" r="2"/>'
    ),
    "Teste de robustez (beta)": (
        '<rect x="3" y="3" width="7" height="7" rx="1.5"/><rect x="14" y="3" width="7" height="7"'
        ' rx="1.5"/><rect x="3" y="14" width="7" height="7" rx="1.5"/><rect x="14" y="14" width="7"'
        ' height="7" rx="1.5"/>'
    ),
    "Qualidade do resultado": '<circle cx="12" cy="12" r="9"/><path d="M8 12.5l2.5 2.5L16 9.5"/>',
    "Gráficos": '<path d="M4 20V10M10 20V4M16 20v-7M22 20H2"/>',
    "Vias metabólicas (beta)": (
        '<circle cx="5" cy="12" r="2.5"/><circle cx="19" cy="5" r="2.5"/><circle cx="19" cy="19"'
        ' r="2.5"/><path d="M7.3 10.8l9.4-4.6M7.3 13.2l9.4 4.6"/>'
    ),
    # O título é o de `pipeline.ABA_DE_LIMITES`, escrito aqui porque este módulo vem antes do do
    # pipeline na cadeia de importações; o portão das chaves contra `ABAS` confere os dois.
    "Métodos e limites": '<circle cx="12" cy="12" r="9"/><path d="M12 11v6M12 7.5v.01"/>',
}

# As etapas mantêm o mesmo controlador; os ícones só distinguem suas páginas.
ICONES.update(
    {
        "Normalização": '<path d="M4 5h16M7 12h10M10 19h4"/>'
        '<circle cx="8" cy="5" r="2"/><circle cx="15" cy="12" r="2"/>',
        "Análise estatística": '<path d="M4 20V10M10 20V4M16 20v-7M22 20H2"/>'
        '<path d="M3 6l5-3 6 5 6-5"/>',
    }
)

# Uma linha por página, sob o título grande. Cada frase descreve o que a aba já mostra, lida da
# composição de `ABAS` e dos painéis, sem prometer nada que a aba não faça.
DESCRICOES: Final[dict[str, str]] = {
    "Preparar o estudo": "Escolha os arquivos do estudo e leia a validação, o diagnóstico e o "
    "palpite da máquina.",
    "Teste de robustez (beta)": (
        "Percorra a grade de pipelines e veja quais features se mantêm em todos."
    ),
    "Qualidade do resultado": "As métricas da execução, cada uma com como ler e a referência.",
    "Gráficos": "As figuras do diagnóstico e da correção, para ver e gravar em PNG, TIFF ou PDF.",
    "Vias metabólicas (beta)": "Predição de vias metabólicas a partir dos picos da execução.",
    "Métodos e limites": (
        "O que fica fora desta ferramenta, e por quê; e as referências de cada método."
    ),
}

DESCRICOES.update(
    {
        "Correção": "Configure e avalie a correção do sinal das suas medições.",
        "Normalização": "Prepare os dados: imputação, normalização, transformação e escala.",
        "Análise estatística": "Defina o estudo, os grupos de comparação e as análises desejadas.",
    }
)


def icone(tracos: str, cor: str, *, lado: int = 40, fundo: str | None = None) -> QPixmap:
    """O ícone desenhado a partir do traço, numa cor. Em memória: nada é escrito em disco.

    Desenhado a 40 px e mostrado a 20, para ficar nítido em tela de alta densidade. Usa o mesmo
    plugin `qsvg` das setas da folha, e o executável precisa levá-lo para os dois.
    """
    conteudo = tracos
    if fundo is not None:
        conteudo = (
            f'<rect width="24" height="24" rx="6" fill="{fundo}" stroke="none"/>'
            f'<g transform="translate(3 3) scale(.75)">{tracos}</g>'
        )
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{lado}" height="{lado}" '
        f'viewBox="0 0 24 24" fill="none" stroke="{cor}" stroke-width="1.8" '
        f'stroke-linecap="round" stroke-linejoin="round">{conteudo}</svg>'
    )
    mapa = QPixmap()
    mapa.loadFromData(svg.encode("utf-8"))
    return mapa


# Os ícones dos cartões de dentro das páginas, no mesmo traço dos da navegação. Chave pelo nome do
# cartão, e não por título de aba: são outra população.
ICONES_DAS_FICHAS: Final[dict[str, str]] = {
    "medidas": (
        '<rect x="3" y="4" width="18" height="16" rx="2"/><path d="M3 9h18M3 14h18M9 4v16"/>'
    ),
    "lista": '<path d="M9 6h11M9 12h11M9 18h11"/><circle cx="4.5" cy="6" r="1"/>'
    '<circle cx="4.5" cy="12" r="1"/><circle cx="4.5" cy="18" r="1"/>',
    "fatores": '<path d="M12 4v16M5 20h14M6 8h12"/><path d="M6 8l-3 6a3 3 0 0 0 6 0z"/>'
    '<path d="M18 8l-3 6a3 3 0 0 0 6 0z"/>',
    "validacao": '<path d="M9 12l2 2 4-4"/><path d="M12 3l7 3v6c0 4.5-3 8-7 9-4-1-7-4.5-7-9V6z"/>',
    "diagnostico": '<path d="M3 12h4l3-7 4 14 3-7h4"/>',
    "recomendacao": '<path d="M9 18h6M10 21h4"/><path d="M12 3a6 6 0 0 0-4 10.5c.8.8 1 1.5 1 '
    '2.5h6c0-1 .2-1.7 1-2.5A6 6 0 0 0 12 3z"/>',
}


def icone_nitido(tracos: str, cor: str, tamanho: int) -> QPixmap:
    """O ícone desenhado ao dobro do tamanho e marcado com razão 2, para ficar nítido em tela de
    alta densidade sem ficar maior em tela comum."""
    mapa = icone(tracos, cor, lado=tamanho * 2)
    mapa.setDevicePixelRatio(2.0)
    return mapa


class SegueAAltura(QObject):
    """Mantém a altura mínima de um widget igual à que o texto dele pede na largura atual.

    Texto que quebra linha tem altura que depende da largura (`heightForWidth`), e as molduras de
    fora (`QGroupBox`, a página da aba) dimensionam pelo `sizeHint`, que não sabe a largura: sem
    isto o fim de uma frase longa sai cortado.

    Filho do widget que vigia, pelo tempo de vida (a razão de `FiltroDaRoda`). Só reage a `Resize`
    e `LayoutRequest`, e só mexe quando a altura pedida muda; sem isso, fixar o mínimo dispararia
    outro `Resize` e o laço não pararia.
    """

    def __init__(self, alvo: QWidget) -> None:
        # Sem referência Python ao alvo: ele é o pai deste objeto, e guardar `self._alvo` criaria
        # um ciclo pai-filho entre os embrulhos do Shiboken que só a coleta de lixo desfaz, e a
        # coleta de lixo pode derrubar o processo. O alvo chega a cada evento como `watched`.
        super().__init__(alvo)
        alvo.installEventFilter(self)

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:  # noqa: N802 - API do Qt
        if event.type() in (QEvent.Type.Resize, QEvent.Type.LayoutRequest) and isinstance(
            watched, QWidget
        ):
            pedida = watched.heightForWidth(watched.width())
            if pedida > 0 and pedida != watched.minimumHeight():
                watched.setMinimumHeight(pedida)
        return False


class _LarguraDeLeitura(QObject):
    """Centraliza o conteúdo sem mover a pilha Qt nem seus painéis de pai.

    Em monitores largos, formulários de ponta a ponta afastam rótulos e campos. As margens
    calculadas limitam o conteúdo a cerca de 960 px e devolvem todo o espaço em janelas menores.
    """

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:  # noqa: N802
        if event.type() == QEvent.Type.Resize and isinstance(watched, QWidget):
            disposicao = watched.layout()
            if disposicao is not None:
                lateral = max(32, (watched.width() - 1024) // 2 + 32)
                margens = disposicao.contentsMargins()
                if margens.left() != lateral or margens.right() != lateral:
                    disposicao.setContentsMargins(lateral, 24, lateral, 24)
        return False


class _EstadoDaBarraReservada(QObject):
    """Esconde o controle vazio, conservando o espaço que evita reformatar os textos."""

    def __init__(self, barra: QScrollBar) -> None:
        super().__init__(barra)
        barra.rangeChanged.connect(self._atualiza)
        self._atualiza(barra.minimum(), barra.maximum())

    def _atualiza(self, minimo: int, maximo: int) -> None:
        barra = self.parent()
        if isinstance(barra, QScrollBar):
            barra.setEnabled(maximo > minimo)


def marca_preenchido(rotulo: QLabel, preenchido: bool) -> None:
    """Liga ou desliga o estilo de 'feito' de um rótulo de estado, e repole para a folha o ver."""
    rotulo.setProperty("preenchido", preenchido)
    estilo = rotulo.style()
    if estilo is not None:  # pragma: no branch - um QWidget construído sempre tem estilo
        estilo.unpolish(rotulo)
        estilo.polish(rotulo)


# O ponto de cada estado de etapa na barra lateral, e a palavra que a dica diz. O tipo é o da folha
# (`QLabel#ponto_do_estado[tipo="..."]`, as mesmas regras do ponto dos cartões de leitura), e a
# palavra existe porque a cor sozinha não basta para ler. 'Não feita' não tem ponto: é o estado de
# repouso, e um ponto em toda etapa antes da primeira corrida seria ruído. O vocabulário é o de
# `etapas.EstadoDaEtapa`, fechado: um estado fora dele levanta em `BarraLateral.mostra_os_estados`.
PONTOS_DAS_ETAPAS: Final[dict[EstadoDaEtapa, tuple[str | None, str]]] = {
    "nao_feita": (None, "não feita"),
    "em_curso": ("info", "em curso"),
    "em_dia": ("sucesso", "em dia"),
    "desatualizada": ("aviso", "desatualizada"),
}


_SETA_FECHADA: Final = '<path d="M9 6l6 6-6 6"/>'
_SETA_ABERTA: Final = '<path d="M6 9l6 6 6-6"/>'


def _poe_a_etiqueta_beta(botao: QPushButton) -> None:
    """A etiqueta 'beta' na ponta direita do botão. Os cliques a atravessam até o botão."""
    etiqueta = QLabel("BETA", botao)
    etiqueta.setObjectName("beta_nav")
    etiqueta.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
    detalhe = QHBoxLayout(botao)
    detalhe.setContentsMargins(0, 0, 10, 0)
    detalhe.addStretch(1)
    # Centrada na altura do botão: sem o alinhamento, o layout estica a etiqueta até as bordas.
    detalhe.addWidget(etiqueta, 0, Qt.AlignmentFlag.AlignVCenter)


class BarraLateral(QFrame):
    """A navegação: marca no topo, um botão por aba com ícone e nome, e o rodapé.

    Sempre larga. A largura de que o formulário de parâmetros precisava veio
    de tirar do lado dele o bloco que explica a proposta da máquina, e não de encolher a barra.

    O controle 'Escuro | Claro' mora aqui, entre a navegação e o rodapé, porque a barra fica à
    vista em toda página e reúne o que vale para a janela inteira. Os ícones da barra são
    desenhados por `pinta`, que a troca de esquema chama de novo.
    """

    def __init__(
        self,
        titulos: list[str],
        esquema: Esquema,
        marca: str,
        descricao: str,
        pai: QWidget | None = None,
    ) -> None:
        super().__init__(pai)
        self.setObjectName("barra_lateral")
        self.titulos = titulos
        self.setFixedWidth(LARGURA_DA_BARRA)
        coluna = QVBoxLayout(self)
        coluna.setContentsMargins(6, 0, 6, 12)
        coluna.setSpacing(1)

        topo = QHBoxLayout()
        topo.setContentsMargins(10, 20, 10, 20)
        topo.setSpacing(8)
        self.logo = QLabel(self)
        self.logo.setObjectName("logo")
        self.logo.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.logo.setFixedSize(20, 20)
        self.nome = QLabel(marca, self)
        self.nome.setObjectName("marca")
        topo.addWidget(self.logo)
        topo.addWidget(self.nome)
        topo.addStretch(1)
        coluna.addLayout(topo)
        # O selo 'LC-MS' fica no fim do subtítulo, que é derivado de `TITULO`: mostrá-lo inteiro
        # evita uma segunda cópia da frase. Hífen que não quebra (U+2011), só na exibição, para
        # 'LC-' e 'MS' não caírem em linhas diferentes; `TITULO` não muda.
        self.subtitulo = QLabel(descricao.replace("-", "\u2011"), self)
        self.subtitulo.setObjectName("subtitulo")
        self.subtitulo.setWordWrap(True)
        self.subtitulo.hide()
        coluna.addWidget(self.subtitulo)
        self.secao = QLabel("NAVEGAÇÃO", self)
        self.secao.setObjectName("secao_nav")
        coluna.addWidget(self.secao)
        coluna.addSpacing(2)

        self.grupo = QButtonGroup(self)
        self.grupo.setExclusive(True)
        self.botoes: list[QPushButton] = []
        # A gaveta 'Trabalhos futuros' recebe os botões das abas cujo título termina em ' (beta)',
        # que `ABAS` põe juntas, logo antes da última. Os botões continuam em `botoes` e em `grupo`
        # no índice da aba; a gaveta só os agrupa e os esconde.
        self.cabecalho_da_gaveta = QPushButton("Trabalhos futuros", self)
        self.cabecalho_da_gaveta.setObjectName("gaveta_nav")
        self.cabecalho_da_gaveta.setCheckable(True)
        self.cabecalho_da_gaveta.setCursor(Qt.CursorShape.PointingHandCursor)
        self.cabecalho_da_gaveta.setIconSize(QSize(14, 14))
        self.cabecalho_da_gaveta.setToolTip(
            "Funcionalidades em beta, fora da entrega ao laboratório."
        )
        _poe_a_etiqueta_beta(self.cabecalho_da_gaveta)
        self.gaveta = QWidget(self)
        self.gaveta.setObjectName("gaveta")
        self._coluna_da_gaveta = QVBoxLayout(self.gaveta)
        self._coluna_da_gaveta.setContentsMargins(12, 0, 0, 0)
        self._coluna_da_gaveta.setSpacing(1)
        self.gaveta.hide()
        self.cabecalho_da_gaveta.toggled.connect(self._mostra_a_gaveta)
        # O ponto de estado de cada aba de etapa, pelo índice da aba. Ver `mostra_os_estados`.
        self.pontos: dict[int, QLabel] = {}
        for indice, titulo in enumerate(titulos):
            # O nome da barra é o título da aba sem o ' (beta)', que vira a etiqueta à direita.
            nome = titulo.removesuffix(" (beta)")
            botao = QPushButton(nome, self)
            botao.setObjectName("nav")
            botao.setCheckable(True)
            botao.setCursor(Qt.CursorShape.PointingHandCursor)
            botao.setToolTip(titulo)
            botao.setIconSize(QSize(14, 14))
            self.grupo.addButton(botao, indice)
            self.botoes.append(botao)
            if titulo.endswith(" (beta)"):
                _poe_a_etiqueta_beta(botao)
                if self._coluna_da_gaveta.count() == 0:
                    coluna.addWidget(self.cabecalho_da_gaveta)
                    coluna.addWidget(self.gaveta)
                self._coluna_da_gaveta.addWidget(botao)
                # A aba ativa nunca fica com o botão escondido, por clique ou por código.
                botao.toggled.connect(self._abre_se_marcado)
            else:
                coluna.addWidget(botao)
        if self._coluna_da_gaveta.count() == 0:
            # Sem aba em beta, o título da gaveta não entra na coluna e não pode ficar à vista.
            self.cabecalho_da_gaveta.hide()

        coluna.addStretch(1)
        self.seletor_de_esquema = QFrame(self)
        self.seletor_de_esquema.setObjectName("seletor_de_esquema")
        self.seletor_de_esquema.setAccessibleName("Esquema de cores da janela")
        self.seletor_de_esquema.setToolTip("O esquema de cores da janela: escuro ou claro.")
        linha = QHBoxLayout(self.seletor_de_esquema)
        linha.setContentsMargins(3, 3, 3, 3)
        linha.setSpacing(3)
        self.grupo_do_esquema = QButtonGroup(self)
        self.grupo_do_esquema.setExclusive(True)
        self.botoes_do_esquema: dict[Esquema, QPushButton] = {}
        for indice, (chave, nome_do_esquema) in enumerate(NOMES_DOS_ESQUEMAS.items()):
            opcao = QPushButton(nome_do_esquema, self.seletor_de_esquema)
            opcao.setObjectName("opcao_de_esquema")
            opcao.setCheckable(True)
            opcao.setCursor(Qt.CursorShape.PointingHandCursor)
            self.grupo_do_esquema.addButton(opcao, indice)
            self.botoes_do_esquema[chave] = opcao
            linha.addWidget(opcao, 1)
        self.grupo_do_esquema.idClicked.connect(self._escolhe)
        coluna.addWidget(self.seletor_de_esquema)
        coluna.addSpacing(10)
        self.rodape = QLabel("TCC · Ciência da Computação · USCS 2026", self)
        self.rodape.setObjectName("rodape_nav")
        self.rodape.setWordWrap(True)
        coluna.addWidget(self.rodape)
        self.pinta(esquema)

    def pinta(self, esquema: Esquema) -> None:
        """Desenha os ícones da barra no esquema dado, e marca o botão dele.

        Os ícones da navegação têm dois tons, o da página marcada e o das outras, e são desenhados
        a 48 px sem razão de pixel; por isso a barra os desenha aqui, e não por `poe_icone`. Marcar
        o botão não emite `idClicked`, então a troca não volta a si mesma.
        """
        cores = _POR_ESQUEMA[esquema]
        self.logo.setPixmap(icone_nitido(_TRACO_DA_MARCA, cores.topo_texto, 20))
        for titulo, botao in zip(self.titulos, self.botoes, strict=True):
            tracos = ICONES[titulo]
            figura = QIcon()
            for estado in (QIcon.State.Off, QIcon.State.On):
                cor = cores.topo_texto if estado == QIcon.State.On else cores.topo_texto_suave
                figura.addPixmap(icone(tracos, cor, lado=48), QIcon.Mode.Normal, estado)
            botao.setIcon(figura)
        # A seta do título da gaveta: para a direita fechada, para baixo aberta.
        seta = _SETA_ABERTA if self.cabecalho_da_gaveta.isChecked() else _SETA_FECHADA
        self.cabecalho_da_gaveta.setIcon(QIcon(icone(seta, cores.topo_texto_suave, lado=48)))
        self.botoes_do_esquema[esquema].setChecked(True)

    def _mostra_a_gaveta(self, aberta: bool) -> None:
        self.gaveta.setVisible(aberta)
        self.pinta(esquema_em_vigor())

    def _abre_se_marcado(self, marcado: bool) -> None:
        if marcado and not self.cabecalho_da_gaveta.isChecked():
            self.cabecalho_da_gaveta.setChecked(True)

    def _escolhe(self, indice: int) -> None:
        """O clique num botão do esquema troca o da janela inteira (a janela é a da barra).

        O clique no esquema que já está em vigor não faz nada: o grupo exclusivo emite `idClicked`
        também para o botão marcado, e a troca refaria folha, repolimento e ícones para nada mudar.
        """
        esquema = list(NOMES_DOS_ESQUEMAS)[indice]
        if esquema != esquema_em_vigor():
            troca_o_esquema(self.window(), esquema)

    def mostra_os_estados(self, estados: Mapping[int, EstadoDaEtapa | None]) -> None:
        """O ponto e a dica de cada aba de etapa. `estados` vai do índice da aba ao estado da etapa
        dela, e `None` é a etapa sem estado a dizer: não há par carregado.

        Quem diz quais abas são etapa, e em que estado, é a janela, com o mesmo recálculo que liga
        os botões das etapas: o ponto e o botão não podem discordar. O ponto nasce na primeira vez
        que a aba tem o que mostrar, e as abas que não são etapa continuam sem ele.

        Tudo é conferido antes de mudar alguma coisa: um estado fora de `PONTOS_DAS_ETAPAS` levanta
        com os válidos na mensagem, e a barra fica como estava. A janela chama isto a cada meio
        segundo, então a folha só é reaplicada ao ponto quando o tipo dele muda.
        """
        invalidos = sorted(
            str(estado)
            for estado in estados.values()
            if estado is not None and estado not in PONTOS_DAS_ETAPAS
        )
        if invalidos:
            raise ValueError(
                f"estado de etapa desconhecido: {', '.join(invalidos)}. "
                f"Os válidos são {', '.join(PONTOS_DAS_ETAPAS)}."
            )
        for indice, estado in estados.items():
            titulo = self.titulos[indice]
            tipo, dica = (
                (None, titulo)
                if estado is None
                else (PONTOS_DAS_ETAPAS[estado][0], f"{titulo} — {PONTOS_DAS_ETAPAS[estado][1]}")
            )
            ponto = self.pontos.get(indice)
            if ponto is None and tipo is not None:
                ponto = self._poe_o_ponto(indice)
            if ponto is not None:
                ponto.setVisible(tipo is not None)
                if tipo is not None and ponto.property("tipo") != tipo:
                    ponto.setProperty("tipo", tipo)
                    estilo = ponto.style()
                    if estilo is not None:  # pragma: no branch - um QWidget construído tem estilo
                        estilo.unpolish(ponto)
                        estilo.polish(ponto)
            botao = self.botoes[indice]
            if botao.toolTip() != dica:
                botao.setToolTip(dica)

    def _poe_o_ponto(self, indice: int) -> QLabel:
        """O ponto de 6 px na ponta direita do botão da aba, centrado na altura do nome.

        É o ponto dos cartões de leitura, com o mesmo nome de objeto e o mesmo tamanho, e é isso
        que o pinta com as regras que a folha já tem, nos dois esquemas. Os cliques atravessam-no
        até o botão.
        """
        botao = self.botoes[indice]
        ponto = QLabel(botao)
        ponto.setObjectName("ponto_do_estado")
        ponto.setFixedSize(6, 6)
        ponto.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        linha = QHBoxLayout(botao)
        linha.setContentsMargins(0, 0, 12, 0)
        linha.addStretch(1)
        linha.addWidget(ponto, 0, Qt.AlignmentFlag.AlignVCenter)
        self.pontos[indice] = ponto
        return ponto

    @staticmethod
    def da_janela(janela: QWidget) -> BarraLateral | None:
        """A barra que `veste_a_janela` acoplou à `janela`, ou `None` com a janela ainda sem ela.

        Procura só entre os painéis acoplados filhos diretos da janela, que é onde `veste_a_janela`
        a põe. O relógio das etapas chama isto a cada meio segundo, e a busca pela árvore inteira
        custa 1,1 ms numa janela vestida, contra 0,004 ms desta.
        """
        docas = janela.findChildren(QDockWidget, options=Qt.FindChildOption.FindDirectChildrenOnly)
        return next(
            (barra for doca in docas if isinstance(barra := doca.widget(), BarraLateral)), None
        )


# O frasco da marca, no topo da barra lateral.
_TRACO_DA_MARCA: Final = (
    '<path d="M9 3h6M10 3v7l-5 9a1.4 1.4 0 0 0 1.3 2h11.4a1.4 1.4 0 0 0 1.3-2l-5-9V3M8 15h8"/>'
)


# A largura fixa da barra lateral, em pixels: a barra é sempre larga, sem modo só de ícones.
LARGURA_DA_BARRA: Final = 208


# Os dois números do atributo de moldura escura do Windows (`DWMWA_USE_IMMERSIVE_DARK_MODE`): 20 a
# partir do Windows 10 20H1, 19 nas versões de 2019. Tenta-se o novo e, recusado, o antigo.
ATRIBUTOS_DA_MOLDURA_ESCURA: Final = (20, 19)


def escurece_a_moldura(janela: QWidget, *, escura: bool = True, dwm: Any = None) -> bool:
    """Pede ao Windows a barra de título escura, ou a clara com `escura=False`. Devolve se aceitou.

    O branco no topo é a moldura do próprio Windows, e não da janela. `QStyleHints.setColorScheme`,
    a forma do Qt, deixa a barra branca no Windows 10; o atributo nativo
    `DWMWA_USE_IMMERSIVE_DARK_MODE` a escurece. A clara usa o mesmo atributo com 0 no lugar de 1,
    para a troca de esquema com a janela aberta não deixar a moldura preta.

    Não levanta nunca, como `icone_da_janela`: moldura é aparência, e recusar abrir porque a cor da
    barra falhou trocaria um defeito invisível por um bloqueio total. Fora do Windows, sem a DLL, ou
    com o Windows a recusar os dois atributos, devolve `False` e a barra fica como o sistema a
    desenha. `dwm` entra por injeção para o teste: é o objeto com `DwmSetWindowAttribute`, e o
    padrão é o `dwmapi` do Windows.
    """
    if dwm is None:
        if sys.platform != "win32":
            return False
        try:
            import ctypes

            dwm = ctypes.windll.dwmapi
        except (ImportError, OSError, AttributeError):  # pragma: no cover - Windows sem dwmapi
            return False
    import ctypes

    ligado = ctypes.c_int(1 if escura else 0)
    try:
        janela_nativa = int(janela.winId())
        for atributo in ATRIBUTOS_DA_MOLDURA_ESCURA:
            resposta = dwm.DwmSetWindowAttribute(
                janela_nativa, atributo, ctypes.byref(ligado), ctypes.sizeof(ligado)
            )
            if resposta == 0:
                return True
    except (OSError, ValueError, TypeError):
        return False
    return False


def pagina_rolavel(conteudo: QWidget, pai: QWidget) -> QScrollArea:
    """Uma página que rola na vertical, para compor antes de a página entrar na aba.

    Embrulhar depois de composta (`removeTab`, `QScrollArea`, `insertTab`) quebra o processo: mover
    uma página que já esteve na pilha da aba deixa a contabilidade de posse do Shiboken
    inconsistente, e a coleta de lixo destrói o mesmo objeto duas vezes (`access violation`). No
    aplicativo, a quebra aconteceria ao fechar a janela, antes de o servidor desligar. Embrulhar na
    composição evita o movimento: a página só entra na aba já dentro da área.

    Só vertical: a rolagem horizontal fica desligada, porque largura a mais num formulário é
    regressão.
    """
    rolagem = QScrollArea(pai)
    rolagem.setWidgetResizable(True)
    rolagem.setFrameShape(QFrame.Shape.NoFrame)
    rolagem.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    rolagem.setWidget(conteudo)
    return rolagem


def _mantem_atalhos(janela: JanelaPrincipal) -> None:
    """Os atalhos de arquivo continuam disponíveis sem ocupar uma barra de menus."""
    for texto, botao, atalho in (
        ("Abrir medições…", janela.arquivos.botao_medidas, "Ctrl+O"),
        ("Abrir lista de amostras…", janela.arquivos.botao_lista, "Ctrl+Shift+O"),
    ):
        acao = QAction(texto, janela)
        acao.triggered.connect(botao.click)
        acao.setShortcut(atalho)
        janela.addAction(acao)
    fechar = QAction("Fechar", janela)
    fechar.setShortcut("Ctrl+Q")
    fechar.triggered.connect(janela.close)
    janela.addAction(fechar)


# ---------------------------------------------------------------------------
# Os trabalhos em curso, no fim da barra lateral e recolhidos
# ---------------------------------------------------------------------------
#
# A faixa de trabalhos fica recolhida no fim da barra lateral. Recolhida, continua à vista o selo
# ('Nenhum trabalho', '2 em execução', 'Concluídos'), que responde 'já ficou pronto?'. Os controles,
# as linhas e o aviso do fecho ficam no corpo, que abre num clique.


class _SegueOCorpo(QObject):
    """Avisa o layout da barra quando o corpo da faixa muda de altura.

    A janela não pode crescer com a faixa: com o corpo aberto e as linhas cheias, a barra pediria
    perto de 980 px de altura mínima, mais que a tela útil de um monitor de 768. Por isso o corpo
    vive numa `QScrollArea` que encolhe e rola quando falta espaço.

    `QScrollArea` não acompanha o conteúdo por si: o pedido de layout das linhas chega à viewport, e
    o layout da barra nunca o vê. Este filtro repassa o pedido com `updateGeometry`, e a barra relê
    o `sizeHint` da área. É filho do conteúdo e não guarda referência Python à área (a razão de
    `SegueAAltura`): a área é o avô do conteúdo, lido a cada evento.
    """

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:  # noqa: N802 - API do Qt
        if event.type() == QEvent.Type.LayoutRequest and isinstance(watched, QWidget):
            viewport = watched.parentWidget()
            area = viewport.parentWidget() if viewport is not None else None
            if isinstance(area, QScrollArea):
                area.updateGeometry()
        return False


def _monta_faixa_de_trabalhos(
    janela: JanelaPrincipal, disposicao: QVBoxLayout, barra: BarraLateral
) -> None:
    """Põe os trabalhos no fim da barra lateral, recolhidos, com o selo à vista.

    O painel e a faixa são os de `principal.py`, com os mesmos objetos e sinais; muda a disposição.
    A barra tem 196 px úteis (208 de largura, 6 de margem de cada lado), então o que era uma
    fileira vira uma coluna: o seletor, 'Atualizar' e 'Parar o trabalho selecionado' um debaixo do
    outro, as linhas empilhadas (`PainelDeTrabalhos.compacta_as_linhas`), e o aviso do fecho no
    fim, quebrando linha. O corpo inteiro rola quando a janela é baixa (`_SegueOCorpo`).

    Os textos não mudam: 'Parar o trabalho selecionado' é citado pela documentação e pelo recorte
    (`desktop/recorte.py`), e com a fonte de 11 px da faixa ele cabe.
    """
    faixa = janela.faixa_de_trabalhos
    pilha = faixa.layout()
    painel = janela.trabalhos
    linha = painel.layout()
    primeiro = linha.itemAt(0) if linha is not None else None
    comando = primeiro.layout() if primeiro is not None else None
    titulo = faixa.findChild(QLabel, "titulo_da_faixa")
    if not (
        isinstance(pilha, QVBoxLayout)
        and isinstance(linha, QVBoxLayout)
        and isinstance(comando, QHBoxLayout)
        and titulo is not None
    ):  # pragma: no cover - desenho: `principal.py` monta sempre estas peças
        raise RuntimeError("a faixa dos trabalhos não tem as peças que a barra lateral espera")
    painel.compacta_as_linhas()

    # A cabeça da faixa esvazia: o título vira o botão que recolhe, o selo e o aviso descem. O
    # layout vazio fica no lugar: tirá-lo devolveria a posse dele ao Python, e vazio não ocupa.
    cabeca_item = pilha.itemAt(0)
    cabeca = cabeca_item.layout() if cabeca_item is not None else None
    if cabeca is not None:
        while cabeca.count():
            cabeca.takeAt(0)
    texto_do_titulo = titulo.text()
    titulo.hide()

    # Os controles, um por fileira, na ordem da leitura.
    for widget in (
        painel.selo,
        painel.rotulo_da_lista,
        painel.seletor_de_trabalho,
        painel.botao_de_atualizacao,
        painel.botao_de_cancelamento,
    ):
        comando.removeWidget(widget)
    while comando.count():
        comando.takeAt(0)
    comando.addWidget(painel.seletor_de_trabalho, 1)
    linha.insertWidget(0, painel.rotulo_da_lista)
    for posicao, botao in enumerate(
        (painel.botao_de_atualizacao, painel.botao_de_cancelamento), start=2
    ):
        botao.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        linha.insertWidget(posicao, botao)

    # As linhas ficam onde estavam, debaixo dos botões; o aviso do fecho vai para o fim, quebrando.
    painel.aviso_curto.setWordWrap(True)
    linha.insertWidget(5, painel.aviso_do_fecho)
    linha.setSpacing(6)
    pilha.setContentsMargins(0, 6, 0, 4)
    pilha.setSpacing(6)
    poe_icone(
        painel.botao_de_atualizacao,
        '<path d="M20 7v5h-5M4 17v-5h5M6 7a7 7 0 0 1 12-2l2 2M4 17l2 2a7 7 0 0 0 12-2"/>',
        "texto_suave",
        12,
    )
    poe_icone(
        painel.botao_de_cancelamento,
        '<rect x="4" y="4" width="16" height="16" rx="1"/>',
        "perigo",
        12,
    )
    # O quadro do fim da barra: o botão que recolhe, o selo, e a faixa como corpo. Entra antes do
    # rodapé e depois da folga que empurra a navegação para cima, e é isso que o põe no fim.
    disposicao.removeWidget(faixa)
    lateral = QFrame(barra)
    lateral.setObjectName("faixa_trabalhos_lateral")
    lateral.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
    coluna = QVBoxLayout(lateral)
    coluna.setContentsMargins(0, 8, 0, 8)
    coluna.setSpacing(4)
    alterna = QToolButton(lateral)
    alterna.setObjectName("alterna_dos_trabalhos")
    alterna.setText(texto_do_titulo)
    alterna.setCheckable(True)
    alterna.setChecked(False)
    alterna.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
    alterna.setIconSize(QSize(14, 14))
    alterna.setCursor(Qt.CursorShape.PointingHandCursor)
    alterna.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
    alterna.setToolTip("Mostra ou recolhe os trabalhos em curso, os controles e o aviso do fecho")
    poe_icone(
        alterna,
        '<path d="m9 5 7 7-7 7"/>',
        "topo_texto",
        14,
        tracos_ligado='<path d="m5 9 7 7 7-7"/>',
    )
    coluna.addWidget(alterna)
    linha_do_selo = QHBoxLayout()
    linha_do_selo.setContentsMargins(4, 0, 0, 0)
    linha_do_selo.addWidget(painel.selo)
    linha_do_selo.addStretch(1)
    coluna.addLayout(linha_do_selo)
    # O corpo é a faixa, numa área que encolhe e rola quando a janela é baixa (`_SegueOCorpo`).
    corpo = QScrollArea(lateral)
    corpo.setObjectName("corpo_dos_trabalhos")
    corpo.setWidgetResizable(True)
    corpo.setFrameShape(QFrame.Shape.NoFrame)
    corpo.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    corpo.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
    corpo.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
    corpo.setWidget(faixa)
    faixa.installEventFilter(_SegueOCorpo(faixa))
    # Recolhida ao nascer: o corpo começa escondido, e só então a expansão é criada, porque ela lê
    # o estado inicial do alvo na construção.
    corpo.setVisible(False)
    coluna.addWidget(corpo)
    expansao = ExpansaoSuave(corpo)
    # A seta segue o estado pelo desenho ligado de `poe_icone`, e repinta na troca de esquema.
    alterna.toggled.connect(expansao.alterna)
    corpo_da_barra = barra.layout()
    assert isinstance(corpo_da_barra, QVBoxLayout), "a barra lateral nasce com a coluna"
    corpo_da_barra.insertWidget(corpo_da_barra.indexOf(barra.rodape), lateral)


def veste_a_janela(janela: JanelaPrincipal) -> BarraLateral:
    """Monta a navegação lateral, com os trabalhos recolhidos no fim dela, e as páginas originais.

    A faixa de trabalhos fica no fim da barra lateral, fora da rolagem, e nasce recolhida com o
    selo à vista. As notificações sobrepõem a área central, sem retirar espaço das páginas. A pilha
    de abas permanece no mesmo pai durante toda a composição.

    A barra, os ícones e o botão marcado do controle nascem no esquema que a aplicação veste
    (`esquema_em_vigor`), o mesmo da folha. Devolve a barra, para o teste e para quem quiser ler os
    botões.
    """
    esquema = esquema_em_vigor()
    titulos = [janela.abas.tabText(i) for i in range(janela.abas.count())]
    _mantem_atalhos(janela)
    marca, _, descricao = janela.windowTitle().partition(" — ")
    # A pilha de abas não sai do lugar: mover o `QTabWidget` para outro pai depois de composto
    # quebra o processo na coleta de lixo, pelo mecanismo descrito em `pagina_rolavel`. Por isso a
    # barra entra como painel acoplado à esquerda do `QMainWindow`, ao lado do widget central sem
    # tocar nele, e o título da página é inserido no topo do layout existente.
    raiz = janela.centralWidget()
    disposicao = raiz.layout() if raiz is not None else None
    if raiz is None or not isinstance(disposicao, QVBoxLayout):  # pragma: no cover - desenho
        raise RuntimeError("a janela não tem o widget central em pilha que a casca espera")
    janela.abas.setObjectName("abas")
    janela.abas.setDocumentMode(True)
    janela.abas.tabBar().hide()
    janela.rotulo_de_desfecho.setObjectName("rotulo_de_desfecho")
    janela.trabalhos.rotulo_do_aviso.setObjectName("aviso_do_fecho")
    janela.trabalhos.rotulo_da_lista.setObjectName("lista_de_trabalhos")
    raiz.setObjectName("area")
    raiz.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
    disposicao.setContentsMargins(32, 24, 32, 24)
    disposicao.setSpacing(24)
    largura_de_leitura = _LarguraDeLeitura(raiz)
    raiz.installEventFilter(largura_de_leitura)

    doca = QDockWidget(janela)
    doca.setObjectName("doca_da_barra")
    doca.setFeatures(QDockWidget.DockWidgetFeature.NoDockWidgetFeatures)
    # Sem barra de título: um `QWidget` vazio no lugar dela é a forma que o Qt documenta.
    doca.setTitleBarWidget(QWidget(doca))
    barra = BarraLateral(titulos, esquema, marca, descricao, doca)
    doca.setWidget(barra)
    janela.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, doca)

    cabecalho = QFrame(raiz)
    cabecalho.setObjectName("cabecalho_da_pagina")
    cabeca = QHBoxLayout(cabecalho)
    cabeca.setContentsMargins(0, 0, 0, 16)
    cabeca.setSpacing(16)
    textos = QVBoxLayout()
    textos.setSpacing(4)
    titulo = QLabel(cabecalho)
    titulo.setObjectName("titulo_da_pagina")
    titulo.setWordWrap(True)
    subtitulo = QLabel(cabecalho)
    subtitulo.setObjectName("descricao_da_pagina")
    # Quebra em vez de alargar: sem quebra, a frase mais longa ditaria a largura mínima da janela
    # inteira (719 px só para esta linha, contra 579 da faixa de baixo).
    subtitulo.setWordWrap(True)
    subtitulo.setMaximumWidth(576)
    textos.addWidget(titulo)
    linha_do_subtitulo = QHBoxLayout()
    linha_do_subtitulo.setContentsMargins(0, 0, 0, 0)
    linha_do_subtitulo.addWidget(subtitulo, 100)
    linha_do_subtitulo.addStretch(1)
    textos.addLayout(linha_do_subtitulo)
    cabeca.addLayout(textos, 1)
    disposicao.insertWidget(0, cabecalho)
    _monta_faixa_de_trabalhos(janela, disposicao, barra)

    def _mostra(indice: int) -> None:
        if not 0 <= indice < len(barra.botoes):
            return
        barra.botoes[indice].setChecked(True)
        nome = titulos[indice]
        titulo.setText(nome)
        subtitulo.setText(DESCRICOES[nome])

    barra.grupo.idClicked.connect(janela.abas.setCurrentIndex)
    janela.abas.currentChanged.connect(_mostra)
    _mostra(janela.abas.currentIndex())
    if esquema == "escuro":
        escurece_a_moldura(janela)

    primarios = [
        janela.multiverso.botao_de_disparo,
        janela.vias.botao,
        *janela.leitura.botoes.values(),
        *janela.pipeline.botoes_de_etapa.values(),
    ]
    for botao in primarios:
        marca_papel(botao, "primario")
    for botao in (
        janela.trabalhos.botao_de_cancelamento,
        janela.multiverso.botao_de_cancelamento,
    ):
        marca_papel(botao, "perigo")
    # Uma barra de rolagem que aparece tira 10 px da página e quebra de novo todos os textos
    # durante uma expansão. O espaço fica reservado, e o controle vazio some.
    for area_rolavel in janela.findChildren(QScrollArea):
        # O corpo dos trabalhos fica de fora: a barra lateral tem 196 px, e uma barra de rolagem
        # sempre ligada tiraria deles o que 'Parar o trabalho selecionado' precisa.
        if area_rolavel.objectName() in ("corpo_do_aviso", "corpo_dos_trabalhos"):
            continue
        area_rolavel.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        _EstadoDaBarraReservada(area_rolavel.verticalScrollBar())
    veste_os_seletores(janela)
    janela.aviso_de_desfecho.reposiciona()
    return barra


def veste_os_seletores(raiz: QWidget) -> int:
    """Prepara cada seletor de `raiz` para a lista arredondada da folha. Devolve quantos.

    A folha desenha o campo e os itens; três coisas ela não alcança:
    - a lista aberta é uma janela própria, e os cantos arredondados só aparecem se ela não tiver
      moldura e o fundo for transparente;
    - as regras `::item` só valem com um `QStyledItemDelegate`; quem já tem um (o
      `ComboComNomes`, que desenha os nomes legíveis) fica com o seu;
    - doze itens à vista antes de rolar, para a lista de figuras não ocupar a tela.

    Nada aqui muda valor, sinal ou texto: é desenho, e por isso mora na casca.
    """
    seletores = raiz.findChildren(QComboBox)
    for seletor in seletores:
        if not isinstance(seletor.itemDelegate(), QStyledItemDelegate):
            seletor.setItemDelegate(QStyledItemDelegate(seletor))
        seletor.setMaxVisibleItems(ITENS_A_VISTA_NO_SELETOR)
        lista = seletor.view().window()
        lista.setWindowFlags(
            lista.windowFlags()
            | Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.NoDropShadowWindowHint
        )
        lista.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
    return len(seletores)
