"""Painel do multiverso: escolher, estimar o custo, disparar, acompanhar, cancelar e ler.

A estimativa de custo antes do disparo faz parte do controle: o eixo `corrections` é quase todo o
custo da grade e é o único que este painel oferece; os outros sete herdam `GridDefinition`, para a
grade disparada aqui ser a mesma que foi avaliada. O par de grupos vem de
`DiagnosticoResposta.grupos_disponiveis`, em combos editáveis, para quem não correu o diagnóstico
poder escrever. O identificador de uma execução só vale no processo que a disparou, porque o
registro de progresso é por processo.

Como os outros painéis, este não chama rota: emite, e a janela faz a chamada. O veredito vem da
rota, recalculado dos limiares a cada leitura; os limiares padrão não foram calibrados para os
dados de quem lê, e `COMO_LER_A_CONCLUSAO` diz isso. A aba está em beta, fora da entrega ao
laboratório, e a primeira linha do painel é `FUNCIONALIDADE_EM_BETA`.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any, Final

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from refino.desktop.recorte import FUNCIONALIDADE_EM_BETA

__all__ = [
    "CELULAS_POR_CORRETOR",
    "CHAVE_DO_CANCELAMENTO",
    "CHAVE_DO_DISPARO",
    "CHAVE_DO_MULTIVERSO",
    "CHAVE_DO_PROGRESSO",
    "COLUNAS",
    "COMO_LER_A_CONCLUSAO",
    "CORRETORES_DA_GRADE",
    "CUSTO_POR_CORRETOR",
    "EXEMPLO_DO_MULTIVERSO",
    "EXPLICACAO_DAS_COLUNAS",
    "EXPLICACAO_DOS_VEREDITOS",
    "MOTIVO_JA_CORRE",
    "MOTIVO_JA_TERMINOU",
    "MOTIVO_NADA_A_CANCELAR",
    "MOTIVO_PARADA_PEDIDA",
    "MOTIVO_SEM_EXECUCAO_A_LER",
    "ROTA_DO_DISPARO",
    "SEM_CONCLUSAO",
    "SEM_CORRETOR",
    "SEM_EXECUCAO",
    "SEM_IDENTIFICADOR_DO_MULTIVERSO",
    "TITULO_DE_COMO_LER",
    "PainelDeMultiverso",
    "estimativa_de_custo",
    "rota_da_gravada",
    "rota_do_cancelamento",
    "rota_do_historico",
    "rota_do_progresso_do_multiverso",
    "rota_do_resultado",
    "texto_do_cabecalho",
]

CHAVE_DO_MULTIVERSO: Final = "multiverso"
"""A chave da leitura da conclusão, a mesma que a janela usa para rotear a resposta.

Constante porque aparece no painel, em `PAINEIS` e no roteamento de `_encaminha`."""

CHAVE_DO_DISPARO: Final = "multiverso_disparo"
CHAVE_DO_PROGRESSO: Final = "multiverso_progresso"
CHAVE_DO_CANCELAMENTO: Final = "multiverso_cancelamento"
"""As chaves do disparo, do progresso e do cancelamento.

Uma por rota porque o roteamento é por chave: o `202` do disparo e o do cancelamento têm corpos
diferentes (`MultiversoAceito` e `ProgressoResposta`), indistinguíveis por um
`isinstance(corpo, dict)`."""

CHAVE_DO_HISTORICO: Final = "multiverso_historico"
CHAVE_DA_GRAVADA: Final = "multiverso_gravada"
"""As duas chaves do histórico: a lista e o detalhe de uma execução gravada.

Separadas pelo mesmo motivo: `HistoricoDeMultiversos` e `ConclusaoDoHistorico` são dicionários
com `200` que vão para tabelas diferentes, e com uma chave só uma lista de execuções cairia na
tabela de features."""

ROTA_DO_DISPARO: Final = "/multiversos"
"""O caminho do disparo, num lugar só; a janela o lê para montar a chamada.

Nenhum teste o compara com o esquema OpenAPI, e o mesmo vale para `rota_do_resultado`: um erro de
digitação aqui daria `404` em produção. Fechar a lacuna custa duas asserções em
`test_as_duas_rotas_novas_batem_com_o_esquema_da_api`, que já compara com
`build_app().openapi()["paths"]`."""

# As constantes da grade de referência e `estimativa_de_custo` vivem em `interface/fluxo.py`, num
# bloco
# puro de dados medidos e funções sem estado. O reexporte abaixo mantém este módulo como o lugar
# de onde o painel e os testes as leem.
from refino.desktop.janela.aparencia import (  # noqa: E402
    CaixaDeAutomatico,
    FiltroDaRoda,
    sem_roda_sem_foco,
)
from refino.desktop.janela.componentes import (  # noqa: E402
    Recolhivel,
    Secao,
    desliga_com_motivo,
    religa,
)
from refino.desktop.janela.nomes import NOMES_DOS_CORRETORES  # noqa: E402
from refino.interface.fluxo import (  # noqa: E402
    CELULAS_POR_CORRETOR,
    CORRETORES_DA_GRADE,
    CUSTO_POR_CORRETOR,
    SEM_CORRETOR,
    estimativa_de_custo,
)
from refino.multiverse.stability import StabilityThresholds  # noqa: E402

SEM_EXECUCAO: Final = "Nenhuma execução disparada nesta janela"

SEM_IDENTIFICADOR_DO_MULTIVERSO: Final = (
    "Não há execução para ler. Clique em «Executar teste de robustez» primeiro, ou cole um "
    "identificador desta mesma sessão no campo acima"
)
"""O que o painel diz quando se pede leitura sem identificador.

Só vale identificador desta sessão: o registro de progresso é por processo, e um identificador de
outra instância do executável devolve `404`. O nome leva `DO_MULTIVERSO` porque
`janela/pipeline.py` já tem um `SEM_IDENTIFICADOR` com outro texto."""

# As frases ao lado dos três botões do ciclo quando desligados: cada uma diz por que e o que fazer.
# 'Cancelar' tem três, porque 'nada corre', 'já terminou' e 'a parada foi pedida' pedem ações
# diferentes; a de quem já terminou é ler a conclusão, a mesma que o `409` do cancelamento manda.
MOTIVO_JA_CORRE: Final = "Um teste de robustez já corre: espere o fim, ou clique em «Cancelar»."
MOTIVO_SEM_EXECUCAO_A_LER: Final = (
    "Ainda não há execução a ler: clique em «Executar teste de robustez»."
)
MOTIVO_NADA_A_CANCELAR: Final = "Nenhum teste de robustez corre: o botão liga quando um começar."
MOTIVO_JA_TERMINOU: Final = (
    "A execução terminou e não há o que cancelar: clique em «Ler conclusão»."
)
MOTIVO_PARADA_PEDIDA: Final = "A parada foi pedida: espere a resposta."

SEM_CONCLUSAO: Final = (
    "Nenhuma conclusão lida. Escolha os corretores, o par de grupos, e clique em «Executar "
    "teste de robustez» — o custo estimado aparece antes do disparo, e a execução pode ser "
    "cancelada a qualquer momento."
)
"""O estado de repouso, que diz o que fazer."""

# Quantas execuções gravadas a janela pede de uma vez. A resposta devolve o `limite`, para uma
# lista cheia não ser lida como completa.
LIMITE_DO_HISTORICO: Final = 50

SEM_HISTORICO: Final = (
    "Nenhuma execução gravada ainda. O que aparece aqui sobrevive ao fechamento do programa — "
    "clique em «Ver histórico» depois de executar um teste de robustez."
)
"""O estado de repouso do histórico, que diz o que fazer."""

# As colunas do histórico respondem 'que execuções existiram?', e as da conclusão, 'quais features
# sobrevivem?'. `n_feitas` e `total` são colunas separadas: a distância entre as duas é até onde a
# execução chegou, o que distingue um `cancelado` que parou no começo de um que parou no fim.
COLUNAS_DO_HISTORICO: Final[tuple[tuple[str, str], ...]] = (
    ("identificador", "Identificador"),
    ("estado", "Estado"),
    ("estudo", "Estudo"),
    ("mode", "Modo"),
    ("n_feitas", "Células feitas"),
    ("total", "Células pedidas"),
    ("n_features", "Features"),
    ("iniciado_em", "Início"),
)

# O veredito é a primeira coluna depois do nome: é a resposta à pergunta que o profissional traz
# ('esta feature sobrevive?'), e as outras medidas a sustentam. A tradução dos rótulos fica aqui e
# não no domínio, porque `robust`/`fragile`/`absent` são vocabulário do modelo. O corte que os
# produz aparece abaixo da tabela, uma vez: ele vale para a execução inteira, não para a feature.
VEREDITOS: Final[dict[str, str]] = {
    "robust": "sobrevive",
    "fragile": "frágil",
    "absent": "não sobrevive",
}

# `n_cells` é coluna, e não nota: `frac_significant = 0,5` sobre 4 células e sobre 9.600 é a mesma
# fração e evidência incomparável, e sem o denominador ao lado a fração esconde a população.
COLUNAS: Final[tuple[tuple[str, str], ...]] = (
    ("feature", "Feature"),
    ("verdict", "Veredito"),
    ("frac_significant", "Fração significativa"),
    ("n_cells", "Células"),
    ("median_qvalue", "q mediano"),
    ("median_rank", "Posto mediano"),
    ("rank_iqr", "IQR do posto"),
    ("sign_consistency", "Consistência de sinal"),
)


# Sentinela antigo de 'não escolhi' nos limiares, abaixo do domínio [0, 1] porque zero é leitura
# legítima (`absent_frac=0` significa 'nada é ausente'). Hoje quem diz 'não escolhi' é a caixa
# 'Automático', e nenhum código lê esta constante.
SEM_LIMIAR = -1.0
"""Sem uso; o mínimo dos campos é `0.0`."""

# Os padrões da especificação, lidos de `StabilityThresholds`, o objeto que o serviço usa quando a
# tela não manda nada. Escrever `0.80` aqui criaria uma segunda fonte, e a tela prometeria um
# padrão que o serviço já não aplica. A tela pode ler uma constante do domínio; é o núcleo que não
# pode conhecer a tela.
_PADROES_DA_SPEC: Final = StabilityThresholds()
PADRAO_DO_LIMIAR: Final[dict[str, float]] = {
    "robust_frac": _PADROES_DA_SPEC.robust_frac,
    "robust_sign": _PADROES_DA_SPEC.robust_sign,
    "absent_frac": _PADROES_DA_SPEC.absent_frac,
}

# Os três limiares do veredito, como `(chave de query, rótulo)`. A chave é a que a rota declara;
# `campos_dos_limiares` deriva daqui, e um teste confere a bijeção contra `LimiaresDoVeredito`.
LIMIARES_DO_VEREDITO: Final[tuple[tuple[str, str], ...]] = (
    ("robust_frac", "Fração mínima de células significativas para `robust`"),
    ("robust_sign", "Consistência de sinal mínima para `robust`"),
    ("absent_frac", "Fração máxima abaixo da qual a feature é `absent`"),
)


# Como ler a conclusão: cada coluna e cada veredito têm a sua frase, o veredito diz o que fazer, e
# há um exemplo medido. As chaves são as de `COLUNAS` e de `VEREDITOS`, na mesma ordem, e um teste
# confere a bijeção. As frases descrevem o que o código calcula (`feature_stability` e
# `classifica_veredito`, em `multiverse/stability.py`): a fração é sobre as células em que a
# feature foi testada, o posto é dentro da célula, e a consistência de sinal é a fração do lado
# mais comum entre as células com sentido definido.
EXPLICACAO_DAS_COLUNAS: Final[dict[str, str]] = {
    "feature": (
        "o sinal medido, pelo nome que ele tem no arquivo: tempo de retenção e m/z. Não é uma "
        "molécula; uma molécula costuma gerar vários sinais."
    ),
    "verdict": (
        "a leitura das colunas seguintes pelo corte dos três limiares acima: «sobrevive», "
        "«frágil» ou «não sobrevive». É a coluna a ler primeiro; as outras dizem por quê."
    ),
    "frac_significant": (
        "em que fração das células em que a feature foi testada a diferença entre os dois grupos "
        "deu q < 0,05. 1 quer dizer em todas as formas de processar; 0, em nenhuma."
    ),
    "n_cells": (
        "em quantas células a feature foi testada: é o denominador da fração. Um filtro pode "
        "tirá-la de uma célula, e aí ela não conta nem como significativa nem como não "
        "significativa. Com 0, nenhuma célula a testou, e as colunas do q, do posto e do sinal "
        "dizem «não medido»."
    ),
    "median_qvalue": (
        "a mediana, entre as células, do q da feature: o valor-p corrigido para os muitos testes "
        "da célula, por Benjamini-Hochberg. Quanto menor, mais forte a diferença na célula típica."
    ),
    "median_rank": (
        "a posição mediana da feature quando as features de cada célula são ordenadas pelo q; 1 é "
        "a mais significativa da célula."
    ),
    "rank_iqr": (
        "quanto essa posição varia entre as células, pela distância entre os quartis. Pequeno: a "
        "feature fica no mesmo lugar qualquer que seja o processamento. Grande: o lugar depende "
        "dele."
    ),
    "sign_consistency": (
        "entre as células em que a diferença tem sentido definido, a fração que aponta para o "
        "lado mais comum, isto é, qual dos dois grupos tem mais. 1 é sempre o mesmo lado; perto "
        "de 0,5, o lado troca com o processamento."
    ),
}


def _fracao(valor: float) -> str:
    """`0.8` vira `"0,80"`: a escala das colunas, que são frações e não percentagens."""
    return f"{valor:.2f}".replace(".", ",")


# Os números do corte são lidos de `PADRAO_DO_LIMIAR`, que os lê do núcleo. São os padrões; quem
# muda os três campos muda o corte, e a introdução de `COMO_LER_A_CONCLUSAO` o diz.
EXPLICACAO_DOS_VEREDITOS: Final[dict[str, str]] = {
    "robust": (
        f"**«{VEREDITOS['robust']}»**: fração significativa de pelo menos "
        f"{_fracao(PADRAO_DO_LIMIAR['robust_frac'])} e consistência de sinal de pelo menos "
        f"{_fracao(PADRAO_DO_LIMIAR['robust_sign'])}. A diferença entre os grupos resiste às "
        "escolhas de processamento e não troca de lado. O que fazer: é a candidata a achado; "
        "reporte-a com a fração e o número de células ao lado."
    ),
    "fragile": (
        f"**«{VEREDITOS['fragile']}»**: o que fica entre os dois, com fração de "
        f"{_fracao(PADRAO_DO_LIMIAR['absent_frac'])} ou mais e abaixo de "
        f"{_fracao(PADRAO_DO_LIMIAR['robust_frac'])}, ou de "
        f"{_fracao(PADRAO_DO_LIMIAR['robust_frac'])} ou mais com consistência de sinal abaixo de "
        f"{_fracao(PADRAO_DO_LIMIAR['robust_sign'])}. A diferença aparece em parte das formas de "
        "processar e some, ou troca de lado, nas outras. O que fazer: não a reporte como achado "
        "sem dizer qual processamento a produziu; trate-a como hipótese, e veja no posto mediano "
        "e no IQR do posto o quanto ela oscila."
    ),
    "absent": (
        f"**«{VEREDITOS['absent']}»**: fração significativa abaixo de "
        f"{_fracao(PADRAO_DO_LIMIAR['absent_frac'])}, ou feature testada em nenhuma célula. O que "
        "fazer: não a reporte como diferença entre os grupos; se uma execução sua a deu como "
        "significativa, foi aquele processamento que a produziu."
    ),
}

# O exemplo é medido: a grade completa nos quatro arquivos reais, com os limiares padrão, cujos
# artefatos estão em `data/fixtures/grade_completa/`. Ela correu
# antes das mudanças de outubro no SERRF e no LOESS e não foi remedida, e o texto o diz. No
# `neg_c18`, par HAMTSP x NoHTLV, uma das 2.136 linhas é 'frágil', com fração 0,26; nos outros 23
# pares todas são 'não sobrevive'. `tests/test_multiverso_na_janela.py` confere cada número.
EXEMPLO_DO_MULTIVERSO: Final = (
    "**Um exemplo medido.** A grade padrão, de 9.600 células, foi corrida nos quatro arquivos de "
    "referência deste estudo, com os seis pares de grupos de cada um: 24 pares. Nenhuma feature "
    "saiu «sobrevive». Uma só saiu «frágil»: no `neg_c18`, no par HAMTSP x NoHTLV, a diferença "
    "deu q < 0,05 numa fração de 0,26 das células em que a feature foi testada, acima de 0,05 e "
    "abaixo de 0,80. Das 2.136 features da tabela desse par, as outras 2.135 saíram «não "
    "sobrevive», como todas as dos outros 23 pares. Leitura: nesses arquivos, nenhuma diferença "
    "entre os grupos resiste às escolhas de processamento, e a única que aparece depende delas. "
    "Essa grade foi medida em setembro de 2026, antes das mudanças de outubro no SERRF e no LOESS, "
    "e não foi medida de novo."
)

TITULO_DE_COMO_LER: Final = (
    "Como ler esta tabela: o que cada coluna mede e o que fazer com cada veredito"
)

COMO_LER_A_CONCLUSAO: Final = (
    "Cada linha é uma feature, e cada célula da grade é uma forma de processar os mesmos "
    "arquivos: um corretor, um corte de RSD, um limiar de valores ausentes, uma imputação, uma "
    "normalização, uma transformação e uma escala. Em cada célula, o programa testa a diferença "
    "entre os dois grupos escolhidos, feature a feature, e as colunas resumem o que aconteceu ao "
    "longo das células. A tabela vem ordenada pela fração significativa, da maior para a menor. "
    "O corte do veredito vem da especificação e não foi calibrado para os seus dados: leia-o como "
    "leitura, e não como sentença. Se mudar os limiares acima, os vereditos mudam com eles.\n\n"
    "**O que cada coluna mede**\n\n"
    + "\n".join(f"- **{titulo}**: {EXPLICACAO_DAS_COLUNAS[chave]}" for chave, titulo in COLUNAS)
    + "\n\n**Os três vereditos, e o que fazer com cada um**\n\n"
    + "\n".join(f"- {EXPLICACAO_DOS_VEREDITOS[chave]}" for chave in VEREDITOS)
    + "\n\n"
    + EXEMPLO_DO_MULTIVERSO
)


def rota_do_resultado(identificador: str) -> str:
    """A rota da conclusão, num lugar só.

    A janela a chama para montar a chamada; uma f-string repetida em dois lugares seria uma segunda
    fonte do caminho.
    """
    return f"/multiversos/{identificador}/resultado"


def campos_dos_limiares(escolhidos: dict[str, float]) -> dict[str, float]:
    """Os limiares que o usuário escolheu, como query da conclusão.

    Só entram as chaves da rota (`LIMIARES_DO_VEREDITO`): este dicionário vira query, e uma chave a
    mais seria descartada em silêncio pelo FastAPI, afirmando uma influência que a rota não tem.
    Zero é valor legítimo (`absent_frac=0` é 'nada é ausente') e viaja. É função de módulo para o
    teste a exercitar sem construir a janela.
    """
    return {
        chave: valor for chave, valor in escolhidos.items() if chave in dict(LIMIARES_DO_VEREDITO)
    }


def rota_do_progresso_do_multiverso(identificador: str) -> str:
    """A rota do progresso do multiverso.

    O nome difere de `rota_do_progresso` de `janela/pipeline.py`, que serve `/trabalhos/{id}`: as
    duas devolvem `ProgressoResposta` e apontam para rotas diferentes.
    """
    return f"/multiversos/{identificador}"


def rota_do_cancelamento(identificador: str) -> str:
    """A rota da parada cooperativa."""
    return f"/multiversos/{identificador}/cancelar"


def rota_do_historico(limite: int = LIMITE_DO_HISTORICO) -> str:
    """A rota da lista das execuções gravadas em disco."""
    return f"/multiversos/historico?limite={limite}"


def rota_da_gravada(identificador: str) -> str:
    """A rota da conclusão gravada em disco, que não é a da conclusão em memória.

    `rota_do_resultado` lê o registro do processo, que some quando o programa fecha ou quando a poda
    aperta; esta lê o banco. Um nome só para as duas faria o painel prometer persistência ao servir
    memória.
    """
    return f"/multiversos/historico/{identificador}"


def texto_do_cabecalho(corpo: Any) -> str:
    """A linha que diz sobre o quê a tabela é, e por isso vem antes dela.

    Os grupos e a contagem de células saem do corpo: uma tabela de frações sem o par de grupos não
    diz o que se lê, e quatro classes dão seis pares possíveis. O corte do veredito é lido da
    resposta; sem `limiares` no corpo (resposta de versão anterior), a frase diz que nenhum corte
    veio, em vez de afirmar um.
    """
    if not isinstance(corpo, dict):
        return SEM_CONCLUSAO
    grupos = corpo.get("grupos")
    if not isinstance(grupos, (list, tuple)) or len(grupos) != 2:
        return SEM_CONCLUSAO
    limiares = corpo.get("limiares")
    if isinstance(limiares, dict):
        corte = (
            f"«Sobrevive» é fração significativa de ao menos "
            f"{limiares.get('robust_frac')} com consistência de sinal de ao menos "
            f"{limiares.get('robust_sign')}; abaixo de {limiares.get('absent_frac')} a feature "
            "não sobrevive. O corte vem da especificação, e não foi calibrado para estes dados."
        )
    else:
        corte = "nenhum corte de «confiável» é aplicado, porque o limiar não veio na resposta."
    return (
        f"{corpo.get('n_features')} features sobre {corpo.get('n_celulas')} células da grade, "
        f"no contraste **{grupos[0]}** contra **{grupos[1]}**. A fração é a proporção de células "
        f"em que a feature saiu significativa. {corte}"
    )


def texto_do_progresso(corpo: Any) -> str:
    """A leitura do progresso na língua de quem opera.

    Os estados não terminais dizem coisas diferentes: `cancelamento_pedido` com `estado` ainda
    `executando` é 'a parar', porque a parada leva até ~0,75 s. 'Parado' afirmaria um estado
    terminal que não aconteceu, e 'a correr' faria o botão de cancelar parecer morto.
    """
    if not isinstance(corpo, dict):
        return SEM_EXECUCAO
    feito = corpo.get("feito")
    total = corpo.get("total")
    estado = str(corpo.get("estado", ""))
    if estado == "cancelado":
        return (
            f"**parada a seu pedido**, depois de {feito} de {total} células. Não há conclusão: ela "
            "é medida sobre a grade inteira. Dispare outra vez, com menos corretores se quiser "
            "esperar menos."
        )
    if estado == "falhou":
        return (
            f"a execução **falhou** na célula «{corpo.get('celula_que_falhou')}»: "
            f"{corpo.get('erro')}"
        )
    if estado == "concluido":
        return f"**concluída** — {feito} de {total} células. Clique em «Ler conclusão»."
    if corpo.get("cancelamento_pedido"):
        return (
            f"**a parar…** — {feito} de {total} células. O pedido foi registrado; a execução para "
            "no próximo ponto de checagem, o que leva até cerca de um segundo."
        )
    return f"a correr — {feito} de {total} células."


def _celula(chave: str, linha: dict[str, Any]) -> str:
    """O texto de uma célula da tabela; `None` vira palavra, e não string vazia.

    `median_qvalue = None` significa que a feature não foi testada em célula nenhuma, e uma célula
    vazia se leria como zero ou como tabela quebrada.
    """
    valor = linha.get(chave)
    if valor is None:
        return "não medido"
    # O veredito é traduzido. Um rótulo desconhecido sai como veio, em vez de virar 'não medido':
    # um quarto veredito no domínio aparece com o nome novo, e não atrás de outra palavra.
    if chave == "verdict":
        return VEREDITOS.get(str(valor), str(valor))
    if isinstance(valor, float):
        return f"{valor:.4g}"
    return str(valor)


# Uma frase por seção, do lado de quem usa.
SUBTITULO_DA_GRADE: Final = (
    "Os corretores e o par de grupos. A estimativa de tempo aparece antes do disparo."
)
SUBTITULO_DA_CONCLUSAO: Final = "Os limiares que classificam cada feature, e a execução a ler."
SUBTITULO_DO_HISTORICO: Final = "As execuções gravadas, que sobrevivem ao fechamento do programa."

# A altura mínima das duas tabelas, em pixels: umas seis linhas. Dentro da página que rola, uma
# tabela sem mínimo pode ser espremida ao cabeçalho quando a janela é baixa.
ALTURA_MINIMA_DAS_TABELAS: Final = 220


def _rotulo_do_campo(texto: str, pai: QWidget) -> QLabel:
    """O nome de um campo, no estilo dos rótulos de formulário da janela."""
    rotulo = QLabel(texto, pai)
    rotulo.setObjectName("rotulo_do_campo")
    return rotulo


class PainelDeMultiverso(QWidget):
    """O multiverso inteiro: escolher, estimar, disparar, acompanhar, parar e ler.

    Um sinal por rota, porque os corpos diferem: um sinal único com um campo `acao` obrigaria a
    janela a desempacotar antes de rotear por chave.
    """

    pedir = Signal(str)
    disparar = Signal(dict)
    ler_progresso = Signal(str)
    cancelar = Signal(str)
    ler_historico = Signal(int)
    abrir_gravada = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        disposicao = QVBoxLayout(self)
        self.setProperty("folhaDeEtapa", True)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        # Três seções numeradas, na ordem do fluxo: 1. a grade a percorrer, 2. a conclusão, 3. o
        # histórico.
        disposicao.setContentsMargins(20, 20, 20, 20)
        disposicao.setSpacing(0)

        # A primeira linha do painel, antes de qualquer controle: quem abre a aba lê que ela está em
        # beta antes de pedir uma grade. O texto vem de `desktop/recorte.py`, porque o painel de
        # vias e a aba 'Métodos e limites' dizem a mesma frase.
        self.rotulo_de_beta = QLabel(FUNCIONALIDADE_EM_BETA, self)
        self.rotulo_de_beta.setWordWrap(True)
        # Uma faixa de aviso, e não uma linha de corpo: é a primeira coisa que se lê.
        self.rotulo_de_beta.setObjectName("faixa_beta")
        disposicao.addWidget(self.rotulo_de_beta)
        # As seções contíguas centralizam o cabeçalho entre suas divisórias.
        # A faixa beta conserva seu respiro próprio, fora dessa sequência.
        disposicao.addSpacing(16)

        self.secao_da_grade = Secao(
            "A grade a percorrer", self, numero=1, subtitulo=SUBTITULO_DA_GRADE
        )
        self.secao_da_grade.setProperty("painel_plano", True)
        disposicao.addWidget(self.secao_da_grade)
        corpo_da_grade = self.secao_da_grade.corpo

        # ------------------------------------------------------------------
        # A escolha: corretores e par de grupos
        # ------------------------------------------------------------------
        escolhas = QGridLayout()
        escolhas.setHorizontalSpacing(16)
        escolhas.setVerticalSpacing(10)
        escolhas.addWidget(_rotulo_do_campo("Corretores a percorrer", self), 0, 0)
        caixa_dos_corretores = QHBoxLayout()
        caixa_dos_corretores.setSpacing(16)
        # Derivado de `GridDefinition`, como `CORRETORES` em `janela/pipeline.py`: valor novo na
        # grade aparece aqui sozinho.
        self.corretores: dict[str, QCheckBox] = {}
        for nome in CORRETORES_DA_GRADE:
            # O nome legível na caixa; a chave do dicionário continua a ser o valor, e é ela que
            # vai ao pedido.
            caixa = QCheckBox(NOMES_DOS_CORRETORES.get(nome, nome), self)
            # Todos marcados por padrão: a grade inteira é a que foi avaliada, e um padrão menor
            # faria a janela executar por omissão outra coisa com o mesmo nome. Quem quiser esperar
            # menos desmarca e vê a estimativa mudar.
            caixa.setChecked(True)
            caixa.stateChanged.connect(self._atualiza_estimativa)
            self.corretores[nome] = caixa
            caixa_dos_corretores.addWidget(caixa)
        caixa_dos_corretores.addStretch(1)
        escolhas.addLayout(caixa_dos_corretores, 0, 1)

        escolhas.addWidget(_rotulo_do_campo("Contraste (grupo A x grupo B)", self), 1, 0)
        par = QHBoxLayout()
        par.setSpacing(10)
        # Editáveis: quem correu o diagnóstico escolhe da lista, quem não correu escreve. Um combo
        # fechado e vazio seria um controle que recusa sem dizer por quê.
        # O filtro da roda, um por painel e com pai: sem pai ele seria coletado, o Qt continuaria a
        # apontar para o objeto morto, e o filtro deixaria de agir sem nada acusar.
        self._filtro_da_roda = FiltroDaRoda(self)

        self.grupo_a = sem_roda_sem_foco(QComboBox(self), self._filtro_da_roda)
        self.grupo_a.setEditable(True)
        self.grupo_b = sem_roda_sem_foco(QComboBox(self), self._filtro_da_roda)
        self.grupo_b.setEditable(True)
        par.addWidget(self.grupo_a, 1)
        par.addWidget(QLabel("contra", self))
        par.addWidget(self.grupo_b, 1)
        escolhas.addLayout(par, 1, 1)
        escolhas.setColumnStretch(1, 1)
        corpo_da_grade.addLayout(escolhas)

        self.rotulo_da_estimativa = QLabel(self)
        self.rotulo_da_estimativa.setWordWrap(True)
        self.rotulo_da_estimativa.setTextFormat(Qt.TextFormat.MarkdownText)
        self.rotulo_da_estimativa.setObjectName("nota_do_campo")
        corpo_da_grade.addWidget(self.rotulo_da_estimativa)

        # ------------------------------------------------------------------
        # Os botões do ciclo
        # ------------------------------------------------------------------
        # Os três botões na largura do que dizem, com o esticador à direita a levar a folga. Ficam
        # explícitos, e não num laço com `setattr`, que esconderia os atributos do `mypy`.
        botoes = QHBoxLayout()
        self.botao_de_disparo = QPushButton("Executar teste de robustez", self)
        self.botao_de_disparo.clicked.connect(self._pede_disparo)
        self.botao_de_disparo.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
        botoes.addWidget(self.botao_de_disparo)
        self.botao_de_progresso = QPushButton("Ler progresso", self)
        self.botao_de_progresso.clicked.connect(self._pede_progresso)
        self.botao_de_progresso.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
        botoes.addWidget(self.botao_de_progresso)
        self.botao_de_cancelamento = QPushButton("Cancelar", self)
        self.botao_de_cancelamento.clicked.connect(self._pede_cancelamento)
        self.botao_de_cancelamento.setSizePolicy(
            QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed
        )
        botoes.addWidget(self.botao_de_cancelamento)
        botoes.addStretch(1)
        corpo_da_grade.addLayout(botoes)

        self.rotulo_de_progresso = QLabel(SEM_EXECUCAO, self)
        self.rotulo_de_progresso.setWordWrap(True)
        self.rotulo_de_progresso.setTextFormat(Qt.TextFormat.MarkdownText)
        corpo_da_grade.addWidget(self.rotulo_de_progresso)

        self.secao_da_conclusao = Secao(
            "A conclusão", self, numero=2, subtitulo=SUBTITULO_DA_CONCLUSAO
        )
        self.secao_da_conclusao.setProperty("painel_plano", True)
        disposicao.addWidget(self.secao_da_conclusao, 1)
        corpo_da_conclusao = self.secao_da_conclusao.corpo

        # ------------------------------------------------------------------
        # A leitura da conclusão
        # ------------------------------------------------------------------
        # Os três limiares do veredito, controláveis. Vão na query e reclassificam no serviço, então
        # mudar de corte custa um clique, e não uma grade nova, que leva dezenas de minutos.
        limiares = QGridLayout()
        limiares.setHorizontalSpacing(16)
        limiares.setVerticalSpacing(8)
        self.limiares: dict[str, QDoubleSpinBox] = {}
        for i, (chave, rotulo) in enumerate(LIMIARES_DO_VEREDITO):
            # Markdown: os rótulos trazem `robust` e `absent` entre crases.
            nome_do_limiar = _rotulo_do_campo(rotulo, self)
            nome_do_limiar.setTextFormat(Qt.TextFormat.MarkdownText)
            limiares.addWidget(nome_do_limiar, i, 0)
            campo = sem_roda_sem_foco(QDoubleSpinBox(self), self._filtro_da_roda)
            # O campo mostra o padrão da especificação. O mínimo é `0.0`, e zero é escolha: quem
            # diz 'não escolhi' é a caixa 'Automático', e não o valor.
            campo.setRange(0.0, 1.0)
            campo.setSingleStep(0.05)
            campo.setDecimals(2)
            campo.setValue(PADRAO_DO_LIMIAR[chave])
            # Teto de largura: são três números entre 0 e 1, e sem ele o `QGridLayout` lhes daria
            # a coluna inteira.
            campo.setMaximumWidth(240)
            self.limiares[chave] = campo
            limiares.addWidget(campo, i, 1)
        # Uma terceira coluna vazia leva a folga; senão a coluna do controle estica e o spinbox,
        # com teto de 240 px, fica encostado à direita, longe do rótulo que o nomeia.
        limiares.setColumnStretch(2, 1)

        # A caixa 'Automático' sobre os três: ticada, eles mostram o padrão da especificação e
        # ficam bloqueados; desticar os liberta; voltar a ticar repõe os padrões.
        self.automatico_dos_limiares = CaixaDeAutomatico(
            "Automático — usar os limiares da especificação",
            self,
            campos=tuple(self.limiares.values()),
        )
        # A caixa antes dos três, porque é ela que os liberta: lida depois, o profissional tentaria
        # mexer num campo bloqueado sem saber por quê.
        corpo_da_conclusao.addWidget(self.automatico_dos_limiares)
        corpo_da_conclusao.addLayout(limiares)

        linha = QHBoxLayout()
        linha.setSpacing(8)
        self.campo_do_identificador = QLineEdit(self)
        self.campo_do_identificador.setPlaceholderText(
            "Identificador da execução do teste de robustez"
        )
        linha.addWidget(self.campo_do_identificador)
        self.botao_de_leitura = QPushButton("Ler conclusão", self)
        self.botao_de_leitura.clicked.connect(self._pede)
        linha.addWidget(self.botao_de_leitura)
        corpo_da_conclusao.addLayout(linha)

        self.rotulo_de_estado = QLabel(SEM_CONCLUSAO, self)
        self.rotulo_de_estado.setWordWrap(True)
        # Markdown, e não `AutoText`: `AutoText` escolhe entre texto simples e HTML, e mostraria os
        # asteriscos do markdown.
        self.rotulo_de_estado.setTextFormat(Qt.TextFormat.MarkdownText)
        corpo_da_conclusao.addWidget(self.rotulo_de_estado)

        # Como ler a tabela, num bloco fechado: é referência, e quem já sabe ler não paga a altura
        # dele em toda visita. Fica antes da tabela, para quem abre ler a chave antes das linhas.
        self.como_ler = Recolhivel(TITULO_DE_COMO_LER, self)
        corpo_de_como_ler = QVBoxLayout(self.como_ler.corpo)
        corpo_de_como_ler.setContentsMargins(16, 8, 16, 8)
        self.texto_de_como_ler = QLabel(COMO_LER_A_CONCLUSAO, self.como_ler.corpo)
        self.texto_de_como_ler.setObjectName("como_ler_a_conclusao")
        self.texto_de_como_ler.setWordWrap(True)
        self.texto_de_como_ler.setTextFormat(Qt.TextFormat.MarkdownText)
        corpo_de_como_ler.addWidget(self.texto_de_como_ler)
        corpo_da_conclusao.addWidget(self.como_ler)

        self.tabela = QTableWidget(0, len(COLUNAS), self)
        # O título de cada coluna, e a explicação dela na dica do cabeçalho, para quem quer saber
        # de uma coluna só sem abrir o bloco. O item é construído aqui com os dois, em vez de lido
        # de volta de `setHorizontalHeaderLabels`, que o devolve como opcional e pediria um ramo
        # que nunca corre.
        for indice, (chave, titulo) in enumerate(COLUNAS):
            item_do_cabecalho = QTableWidgetItem(titulo)
            item_do_cabecalho.setToolTip(EXPLICACAO_DAS_COLUNAS[chave])
            self.tabela.setHorizontalHeaderItem(indice, item_do_cabecalho)
        cabecalho = self.tabela.horizontalHeader()
        if cabecalho is not None:  # pragma: no cover - o Qt devolve-o sempre num widget construído
            cabecalho.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        # A folga é da tabela: a seção dela tem fator 1 na página, e a tabela fator 1 na seção. A
        # altura mínima é a de umas seis linhas, para a página que rola não a espremer.
        self.tabela.setMinimumHeight(ALTURA_MINIMA_DAS_TABELAS)
        corpo_da_conclusao.addWidget(self.tabela, 1)

        # ------------------------------------------------------------------
        # O histórico: o que sobreviveu ao fechamento do programa
        # ------------------------------------------------------------------
        self.secao_do_historico = Secao(
            "Histórico", self, numero=3, subtitulo=SUBTITULO_DO_HISTORICO
        )
        self.secao_do_historico.setProperty("painel_plano", True)
        disposicao.addWidget(self.secao_do_historico, 1)
        corpo_do_historico = self.secao_do_historico.corpo
        # Os dois botões no cabeçalho da seção, à direita, que é onde vivem as ações dela.
        self.botao_do_historico = QPushButton("Ver histórico", self)
        self.botao_do_historico.clicked.connect(self._pede_historico)
        self.secao_do_historico.acoes.addWidget(self.botao_do_historico)
        self.botao_de_abertura = QPushButton("Abrir a selecionada", self)
        self.botao_de_abertura.clicked.connect(self._pede_gravada)
        self.secao_do_historico.acoes.addWidget(self.botao_de_abertura)

        self.rotulo_do_historico = QLabel(SEM_HISTORICO, self)
        self.rotulo_do_historico.setWordWrap(True)
        self.rotulo_do_historico.setTextFormat(Qt.TextFormat.MarkdownText)
        corpo_do_historico.addWidget(self.rotulo_do_historico)

        self.tabela_do_historico = QTableWidget(0, len(COLUNAS_DO_HISTORICO), self)
        self.tabela_do_historico.setHorizontalHeaderLabels(
            [titulo for _chave, titulo in COLUNAS_DO_HISTORICO]
        )
        # Seleção por linha inteira, porque a unidade é a execução: selecionar uma célula e abrir a
        # execução dela faria o clique num `estado` parecer não fazer nada.
        self.tabela_do_historico.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        # Duplo clique abre, porque é o gesto que qualquer lista de histórico ensina.
        self.tabela_do_historico.doubleClicked.connect(self._pede_gravada)
        cabecalho_do_historico = self.tabela_do_historico.horizontalHeader()
        if cabecalho_do_historico is not None:  # pragma: no cover - o Qt devolve-o sempre
            cabecalho_do_historico.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.tabela_do_historico.setMinimumHeight(ALTURA_MINIMA_DAS_TABELAS)
        corpo_do_historico.addWidget(self.tabela_do_historico, 1)

        self._identificador: str | None = None
        self._a_correr = False
        # O modo da matriz carregada. Nasce `None` porque, ao abrir o programa, não há matriz, e a
        # estimativa é a faixa dos quatro modos.
        self._modo: str | None = None
        self._atualiza_estimativa()
        self._atualiza_botoes()

    # ------------------------------------------------------------------
    # O estado dos controles
    # ------------------------------------------------------------------

    def _atualiza_estimativa(self) -> None:
        """Reescreve a estimativa. Ligado ao `stateChanged` de cada caixa.

        Recalcular a cada clique, e não no disparo, faz da estimativa um instrumento de decisão:
        quem desmarca um corretor vê o tempo cair antes de escolher.
        """
        _celulas, _segundos, texto = estimativa_de_custo(self.corretores_escolhidos(), self._modo)
        self.rotulo_da_estimativa.setText(texto)

    def define_modo(self, modo: str | None) -> None:
        """Recebe o modo da matriz carregada, vindo do painel de arquivos.

        O modo é empurrado pela janela, e não puxado, para este painel não saber quem o escolhe.
        `None` é legítimo: antes de haver modo, a estimativa é a faixa dos quatro.
        """
        self._modo = modo
        self._atualiza_estimativa()

    def _atualiza_botoes(self) -> None:
        """Habilita o que faz sentido no estado atual e desabilita o resto.

        - em repouso, sem execução: disparar, sim; progresso e cancelar, não;
        - a correr: disparar, não; progresso e cancelar, sim;
        - terminada: disparar e progresso, sim; cancelar, não.

        'Cancelar' sem execução fica desligado, e não escondido, como nos outros painéis. Botão
        desligado vai sempre com a frase ao lado (`desliga_com_motivo`), que diz por que e o que
        fazer.
        """
        if self._a_correr:
            desliga_com_motivo(self.botao_de_disparo, MOTIVO_JA_CORRE)
            religa(self.botao_de_cancelamento)
        else:
            religa(self.botao_de_disparo)
            desliga_com_motivo(
                self.botao_de_cancelamento,
                MOTIVO_NADA_A_CANCELAR if self._identificador is None else MOTIVO_JA_TERMINOU,
            )
        if self._identificador is None:
            desliga_com_motivo(self.botao_de_progresso, MOTIVO_SEM_EXECUCAO_A_LER)
        else:
            religa(self.botao_de_progresso)

    def corretores_escolhidos(self) -> tuple[str, ...]:
        """Os corretores marcados, na ordem da grade.

        É a ordem em que o executor os percorre, então a estimativa e a execução falam da mesma
        sequência.
        """
        return tuple(nome for nome in CORRETORES_DA_GRADE if self.corretores[nome].isChecked())

    def define_grupos(self, disponiveis: Sequence[str]) -> tuple[str, ...]:
        """Enche os dois combos com os grupos do diagnóstico. Devolve o que ficou lá.

        Preserva o que o usuário escreveu: correr o diagnóstico outra vez apagaria a escolha do par.
        Sem nada escrito, põe os dois primeiros grupos; qualquer par é uma pergunta diferente, e o
        painel preenche só para haver algo executável.
        """
        nomes = tuple(str(n) for n in disponiveis)
        for indice, combo in enumerate((self.grupo_a, self.grupo_b)):
            escrito = combo.currentText().strip()
            combo.clear()
            combo.addItems(list(nomes))
            if escrito:
                combo.setCurrentText(escrito)
            elif len(nomes) > indice:
                combo.setCurrentText(nomes[indice])
            else:
                combo.setCurrentText("")
        return nomes

    def escolhas_do_disparo(self) -> dict[str, Any]:
        """O que os controles deste painel escolheram, no formato que a rota recebe.

        `corrections` viaja como JSON porque a rota o recebe por `Form()` e o lê com `json.loads`.
        O par de grupos só viaja se os dois estiverem preenchidos: a rota recusa meio par com
        `422`, porque ignorá-lo executaria uma grade de horas e devolveria progresso a quem pediu
        conclusão.
        """
        campos: dict[str, Any] = {
            "corrections": json.dumps(list(self.corretores_escolhidos())),
        }
        a = self.grupo_a.currentText().strip()
        b = self.grupo_b.currentText().strip()
        if a and b:
            campos["group_a"] = a
            campos["group_b"] = b
        return campos

    # ------------------------------------------------------------------
    # Os slots
    # ------------------------------------------------------------------

    def _pede_disparo(self) -> None:
        """O clique de executar. Não chama rota: emite, e a janela junta os arquivos e o modo.

        Recusa grade vazia aqui, e não na rota: `GridDefinition` levanta com eixo vazio, e o
        profissional atravessaria o envio de dois CSV para receber uma mensagem que o painel já
        podia dar.
        """
        if not self.corretores_escolhidos():
            self.rotulo_de_progresso.setText(SEM_CORRETOR)
            return
        self.disparar.emit(self.escolhas_do_disparo())

    def _pede_progresso(self) -> None:
        """A releitura manual. Sem identificador, diz que não há o que ler."""
        if self._identificador is None:
            self.rotulo_de_progresso.setText(SEM_IDENTIFICADOR_DO_MULTIVERSO)
            return
        self.ler_progresso.emit(self._identificador)

    def _pede_cancelamento(self) -> None:
        """O clique de cancelar. Escreve 'a parar…' antes de a resposta chegar.

        A rota devolve `202` e a parada leva até ~0,75 s; sem esta linha o intervalo seria mudo, e
        o usuário clicaria outra vez, que é o caso para o qual a rota é idempotente.
        """
        if self._identificador is None:
            self.rotulo_de_progresso.setText(SEM_IDENTIFICADOR_DO_MULTIVERSO)
            return
        desliga_com_motivo(self.botao_de_cancelamento, MOTIVO_PARADA_PEDIDA)
        self.rotulo_de_progresso.setText("**a parar…** — o pedido foi enviado.")
        self.cancelar.emit(self._identificador)

    def campos_da_conclusao(self) -> dict[str, float]:
        """Os limiares escolhidos, prontos para a query da conclusão.

        É método do painel, e não carga do sinal `pedir`, porque quem tem o estado é o painel e a
        janela pergunta, como em `campos_da_execucao` do painel do pipeline.

        Quem decide o que viaja é a caixa 'Automático', e não o valor: ticada, devolve vazio, e a
        rota usa os padrões da especificação; desticada, devolve os três. Comparar o valor com o
        padrão mandaria `0,80` sempre e apagaria a diferença entre 'aceitei o padrão' e 'escolhi
        0,80', que importa no dia em que a especificação mudar de número.
        """
        if self.automatico_dos_limiares.isChecked():
            return {}
        return campos_dos_limiares({chave: campo.value() for chave, campo in self.limiares.items()})

    def _pede(self) -> None:
        """O clique de ler a conclusão. Identificador vazio escreve a frase.

        Lê o campo, e não `self._identificador`: o campo é o que o usuário vê, um disparo o enche, e
        um identificador colado por cima é o que vale.
        """
        identificador = self.campo_do_identificador.text().strip()
        if not identificador:
            self.rotulo_de_estado.setText(SEM_IDENTIFICADOR_DO_MULTIVERSO)
            return
        self.pedir.emit(identificador)

    def _pede_historico(self) -> None:
        """O clique de listar o que está em disco, sem pré-condição.

        O histórico existe por causa das sessões anteriores, e exigir um disparo antes o negaria.
        """
        self.ler_historico.emit(LIMITE_DO_HISTORICO)

    def _pede_gravada(self) -> None:
        """O clique de abrir a execução selecionada. Sem seleção, escreve a frase.

        Lê o identificador da linha selecionada, e não o campo de texto: o campo serve para colar um
        identificador desta sessão, e a tabela para escolher um de qualquer sessão.
        """
        linha = self.tabela_do_historico.currentRow()
        item = None if linha < 0 else self.tabela_do_historico.item(linha, 0)
        if item is None or not item.text().strip():
            self.rotulo_do_historico.setText(
                "Nenhuma execução selecionada. Clique numa linha da tabela do histórico, ou dê "
                "duplo clique nela."
            )
            return
        self.abrir_gravada.emit(item.text().strip())

    # ------------------------------------------------------------------
    # As respostas
    # ------------------------------------------------------------------

    def aceita_disparo(self, codigo: int, corpo: Any) -> str:
        """Guarda o identificador do `202` e escreve o que aconteceu. Devolve o texto.

        Enche o campo de leitura, para o profissional não ter de copiar o identificador de um
        rótulo.
        """
        if codigo == 202 and isinstance(corpo, dict) and corpo.get("identificador"):
            self._identificador = str(corpo["identificador"])
            self._a_correr = True
            self.campo_do_identificador.setText(self._identificador)
            # A gravação é declarada no disparo. A persistência é melhor esforço: sem banco, a
            # grade corre e nada fica em disco, e calar isso faria o profissional contar com um
            # histórico que não existe e descobri-lo só ao reabrir.
            aviso = (
                ""
                if corpo.get("persistido")
                else (
                    "\n\n**Esta execução NÃO será gravada** e some ao fechar o programa: "
                    f"{corpo.get('motivo_nao_gravado')}"
                )
            )
            texto = (
                f"disparada: **{corpo.get('total')} células**. Clique em «Ler progresso» para "
                f"acompanhar, ou em «Cancelar» para parar.{aviso}"
            )
        else:
            self._a_correr = False
            texto = _detalhe_ou_codigo(codigo, corpo)
        self.rotulo_de_progresso.setText(texto)
        self._atualiza_botoes()
        return texto

    def recebe_progresso(self, codigo: int, corpo: Any) -> str:
        """Uma leitura do progresso. Devolve o texto escrito.

        Lê `terminou`, e não `estado == "concluido"`, porque três dos cinco estados são terminais;
        enumerar rótulos faria o painel esperar para sempre por uma execução cancelada.
        """
        if codigo != 200 or not isinstance(corpo, dict):
            texto = _detalhe_ou_codigo(codigo, corpo)
        else:
            texto = texto_do_progresso(corpo)
            self._a_correr = not bool(corpo.get("terminou"))
        self.rotulo_de_progresso.setText(texto)
        self._atualiza_botoes()
        return texto

    def aceita_cancelamento(self, codigo: int, corpo: Any) -> str:
        """A resposta do pedido de parada. Devolve o texto escrito.

        `409` não é falha: significa que a execução já terminou, e a ação é ler a conclusão.
        """
        if codigo == 202 and isinstance(corpo, dict):
            texto = texto_do_progresso(corpo)
        elif codigo == 409:
            texto = (
                f"não há o que cancelar: {_detalhe_ou_codigo(codigo, corpo)} Clique em «Ler "
                "conclusão»."
            )
            self._a_correr = False
        else:
            texto = _detalhe_ou_codigo(codigo, corpo)
        self.rotulo_de_progresso.setText(texto)
        self._atualiza_botoes()
        return texto

    def mostra(self, codigo: int, corpo: Any) -> str:
        """Escreve o desfecho da conclusão e preenche a tabela. Devolve o texto do rótulo.

        A tabela é limpa em todo desfecho que não seja `200`: deixá-la ao lado de uma mensagem de
        erro afirmaria que aquela conclusão é desta leitura.

        - `200`: o cabeçalho com o par de grupos, e a tabela;
        - `202`: ainda a correr, com a contagem de células que já terminaram;
        - `409`: a ação que a rota nomeia (redisparar declarando o par, ou por que foi cancelada);
        - outro código: a mensagem que a rota deu, sem interpretação.
        """
        if codigo == 200 and isinstance(corpo, dict):
            self._preenche(corpo.get("estabilidade"))
            texto = texto_do_cabecalho(corpo)
            self.rotulo_de_estado.setText(texto)
            # Só a conclusão desta execução diz que ela acabou. `_pede` lê o campo, que aceita o
            # identificador de outra execução já terminada; a conclusão dela tiraria o painel de 'a
            # correr' com este disparo ainda em curso, e a trava de uma correção por vez, que lê
            # `a_correr`, deixaria passar. Sem identificador no corpo, o painel fica como estava.
            if self._identificador is None or corpo.get("identificador") == self._identificador:
                self._a_correr = False
            self._atualiza_botoes()
            return texto

        self.tabela.setRowCount(0)
        if codigo == 202 and isinstance(corpo, dict):
            texto = (
                f"a execução ainda corre: {corpo.get('feito')} de {corpo.get('total')} células. "
                "A conclusão aparece aqui quando ela terminar."
            )
        elif isinstance(corpo, dict) and isinstance(corpo.get("detail"), str):
            texto = str(corpo["detail"])
        else:
            texto = f"{SEM_CONCLUSAO} (código {codigo})"
        self.rotulo_de_estado.setText(texto)
        return texto

    def mostra_historico(self, codigo: int, corpo: Any) -> str:
        """Preenche a tabela do histórico. Devolve o texto do rótulo.

        A tabela é limpa em todo desfecho que não seja `200`, como em `mostra`. Lista vazia com
        `200` não é erro: banco em pé e nada gravado é 'ainda não corri nada', e o `503` (banco que
        não responde) é 'não consigo ler o que corri'.
        """
        if codigo != 200 or not isinstance(corpo, dict):
            self.tabela_do_historico.setRowCount(0)
            texto = _detalhe_ou_codigo(codigo, corpo)
            self.rotulo_do_historico.setText(texto)
            return texto

        execucoes = corpo.get("execucoes")
        linhas = execucoes if isinstance(execucoes, list) else []
        self.tabela_do_historico.setRowCount(len(linhas))
        for indice, execucao in enumerate(linhas):
            if not isinstance(execucao, dict):  # pragma: no cover - a rota devolve objetos
                continue
            for coluna, (chave, _titulo) in enumerate(COLUNAS_DO_HISTORICO):
                self.tabela_do_historico.setItem(
                    indice, coluna, QTableWidgetItem(_celula_do_historico(chave, execucao))
                )

        if not linhas:
            texto = SEM_HISTORICO
        else:
            limite = corpo.get("limite")
            # O teto vai no texto quando a lista o alcança: aí ela deixa de ser o histórico e passa
            # a ser as N mais recentes.
            corte = (
                f" Este é o teto de **{limite}**: pode haver mais."
                if isinstance(limite, int) and len(linhas) >= limite
                else ""
            )
            texto = (
                f"**{len(linhas)} execução(ões)** gravadas, da mais recente para a mais antiga. "
                f"Selecione uma e clique em «Abrir a selecionada».{corte}"
            )
        self.rotulo_do_historico.setText(texto)
        return texto

    def mostra_gravada(self, codigo: int, corpo: Any) -> str:
        """Mostra a conclusão vinda do disco, na mesma tabela da conclusão em memória.

        As duas têm as mesmas colunas e respondem a mesma pergunta; duas tabelas lado a lado fariam
        comparar dois números que são o mesmo. O cabeçalho diz de onde veio. Execução sem conclusão
        (grade cancelada antes do fim, ou disparada sem par de grupos) existe e não tem tabela, e o
        texto o diz; `404` é identificador que o banco não conhece, e a rota já o distingue.
        """
        if codigo != 200 or not isinstance(corpo, dict):
            self.tabela.setRowCount(0)
            texto = _detalhe_ou_codigo(codigo, corpo)
            self.rotulo_de_estado.setText(texto)
            return texto

        execucao = corpo.get("execucao")
        execucao = execucao if isinstance(execucao, dict) else {}
        estabilidade = corpo.get("estabilidade")
        self._preenche(estabilidade)
        grupos = execucao.get("grupos")
        par = " x ".join(str(g) for g in grupos) if isinstance(grupos, list) and grupos else None
        if not isinstance(estabilidade, list) or not estabilidade:
            texto = (
                f"**do histórico** — execução `{execucao.get('identificador')}`, estado "
                f"**{execucao.get('estado')}**, {execucao.get('n_feitas')} de "
                f"{execucao.get('total')} células. Ela não produziu conclusão: "
                + (
                    "nenhum par de grupos foi declarado no disparo."
                    if par is None
                    else "ela parou antes de a medir."
                )
            )
        else:
            # O corte vai no texto também aqui: a rota serve `limiares` para o veredito não viajar
            # sem o corte que o produziu, e é o caminho do disco que atravessa sessões, onde quem
            # lê tem menos contexto. Os rótulos saem de `LIMIARES_DO_VEREDITO`.
            servidos = corpo.get("limiares")
            if isinstance(servidos, dict):
                lista = "; ".join(
                    f"{rotulo}: {servidos.get(chave)}" for chave, rotulo in LIMIARES_DO_VEREDITO
                )
                corte = (
                    f"O corte que produziu os vereditos — {lista}. Ele vem da especificação, e "
                    "não foi calibrado para estes dados."
                )
            else:
                corte = (
                    "nenhum corte de «confiável» é aplicado, porque o limiar não veio na resposta."
                )
            texto = (
                f"**do histórico** — execução `{execucao.get('identificador')}`, estado "
                f"**{execucao.get('estado')}**, contraste **{par}**, "
                f"{execucao.get('n_feitas')} de {execucao.get('total')} células. "
                f"**{len(estabilidade)} features** medidas. {corte}"
            )
        self.rotulo_de_estado.setText(texto)
        return texto

    def linhas_do_historico(self) -> tuple[str, ...]:
        """Os identificadores que a tabela do histórico mostra, na ordem em que estão nela.

        Acessor para os testes afirmarem o que o painel mostra sem percorrer `QTableWidgetItem`.
        """
        nomes = []
        for indice in range(self.tabela_do_historico.rowCount()):
            item = self.tabela_do_historico.item(indice, 0)
            nomes.append("" if item is None else item.text())
        return tuple(nomes)

    def _preenche(self, estabilidade: Any) -> None:
        """Preenche a tabela, na ordem em que a resposta a entregou.

        Não reordena: a rota já entrega por estabilidade decrescente, com o nome da feature como
        desempate, e reordenar aqui criaria uma segunda fonte da ordem.
        """
        linhas = estabilidade if isinstance(estabilidade, list) else []
        self.tabela.setRowCount(len(linhas))
        for indice, linha in enumerate(linhas):
            if not isinstance(linha, dict):  # pragma: no cover - a rota devolve objetos
                continue
            for coluna, (chave, _titulo) in enumerate(COLUNAS):
                self.tabela.setItem(indice, coluna, QTableWidgetItem(_celula(chave, linha)))

    def identificador(self) -> str | None:
        """O identificador do último disparo desta sessão, ou `None`.

        Acessor, e não atributo público, para não convidar a escrevê-lo de fora.
        """
        return self._identificador

    def a_correr(self) -> bool:
        """Diz se a última resposta que este painel leu dá a execução como em curso.

        É o que o painel sabe, e não o que o servidor faz: a leitura do progresso é manual, então
        uma execução que terminou sem ninguém a ler continua 'a correr' aqui até a leitura seguinte.
        A janela o lê para pôr o multiverso na trava de uma correção por vez, e a frase da recusa
        diz como desfazer o atraso. Só a conclusão da execução disparada aqui o desliga (ver
        `mostra`).
        """
        return self._a_correr

    def linhas_mostradas(self) -> tuple[str, ...]:
        """Os nomes de feature que a tabela mostra, na ordem em que estão nela.

        Acessor para os testes afirmarem o que o painel mostra sem percorrer `QTableWidgetItem`.
        """
        return self._coluna("feature")

    def vereditos_mostrados(self) -> tuple[str, ...]:
        """Os vereditos que a tabela mostra, na ordem das linhas.

        Acessor, como `linhas_mostradas`.
        """
        return self._coluna("verdict")

    def _coluna(self, chave: str) -> tuple[str, ...]:
        """O texto de uma coluna inteira, pela chave e não pelo índice.

        Um índice literal leria a coluna vizinha em silêncio quando `COLUNAS` mudasse de ordem.
        """
        coluna = next(i for i, (nome, _) in enumerate(COLUNAS) if nome == chave)
        textos = []
        for indice in range(self.tabela.rowCount()):
            item = self.tabela.item(indice, coluna)
            textos.append("" if item is None else item.text())
        return tuple(textos)


def _celula_do_historico(chave: str, execucao: dict[str, Any]) -> str:
    """O texto de uma célula da tabela do histórico.

    `n_feitas` de um `interrompido` é '—', e não `0`: a gravação acontece na abertura e no fecho, e
    não por célula, então uma execução interrompida pelo fechamento do programa guarda o `0` com que
    nasceu, e '0' afirmaria que ela não fez nada quando pode ter feito quase tudo.

    Os instantes saem em hora local, e não no UTC gravado: quem lê compara com o próprio relógio.
    """
    valor = execucao.get(chave)
    if valor is None:
        return ""
    if chave == "n_feitas" and execucao.get("estado") == "interrompido":
        return "—"
    if chave in {"iniciado_em", "terminado_em"}:
        return _instante_local(str(valor))
    return str(valor)


def _instante_local(bruto: str) -> str:
    """Um carimbo ISO em hora local, ou o texto cru quando ele não se deixa ler.

    Devolve o cru em vez de levantar: uma tabela que morre por um formato de data inesperado é pior
    que uma que mostra a data como veio. O que não se pode é inventar um instante.
    """
    try:
        instante = datetime.fromisoformat(bruto)
    except ValueError:
        return bruto
    if instante.tzinfo is None:
        # Sem fuso, o único valor honesto é o que a API grava: UTC.
        instante = instante.replace(tzinfo=UTC)
    return instante.astimezone().strftime("%d/%m/%Y %H:%M:%S")


def _detalhe_ou_codigo(codigo: int, corpo: Any) -> str:
    """A mensagem que a rota deu, ou o código nu quando não há nenhuma.

    A mensagem vem crua, como em `janela/pipeline.py`: as mensagens desta API nomeiam a ação, e
    reescrevê-las aqui criaria uma segunda redação.
    """
    if isinstance(corpo, dict) and isinstance(corpo.get("detail"), str):
        return str(corpo["detail"])
    return f"a chamada devolveu {codigo} sem mensagem."
