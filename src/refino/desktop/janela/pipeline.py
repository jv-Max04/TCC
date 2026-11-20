"""Painel do pipeline: o disparo, o progresso e o par de sobrescrita.

Este módulo compõe o payload e formata as leituras. Não chama rota nem lê relógio: quem chama a
rota é `JanelaPrincipal`, e quem conta os segundos é `interface/decorrido.py`. A única subtração
daqui é
entre dois valores de `decorrido_da_leitura`, para dizer há quanto tempo a etapa em curso começou.
O `DepositoEmMemoria` vem do módulo compartilhado porque um decorrido reimplementado aqui poderia
mostrar um contador parado, plausível e de outra execução. Dois portões conferem essas ausências:
o de AST sobre `desktop/janela/` e o do funil do carimbo.

A proposta da máquina vem de `_proposta_da_maquina`, em `desktop/etapas.py`, e não é remontada
aqui. Ela serializa com `sort_keys=True` e `ensure_ascii=False` porque o par de sobrescrita compara
dois textos: uma segunda montagem, com outra ordem de chaves, classificaria como discordância uma
spec idêntica, sem erro nenhum.

A terminalidade vem do código HTTP, e não de campo do corpo. `GET /trabalhos/{id}` responde:

- `404` com `detail`: identificador desconhecido;
- `202` com `ProgressoResposta`, sempre com `terminou=False`: sinal de vida;
- `200` com `ExecucaoResposta` ou `text/csv`: o resultado;
- `409`, `422` ou `500` com `detail` que nomeia a etapa: falhou.

`terminou=True` nunca chega ao cliente. Por isso `recebe_leitura` roteia pelo código (`terminou` é
`codigo != 202`), e a etapa da falha vem do `detail`.
"""

from __future__ import annotations

import json
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Final, NamedTuple, get_args

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from refino.cli import (
    AUTO_APRENDIDA,
    AUTO_CAVEATS,
    AUTO_MEDIDA,
    AVAILABLE_METHODS,
    IMPUTATION_CAVEATS,
    METHOD_CAVEATS,
    Caveat,
)
from refino.correction.loess import LoessCorrector
from refino.correction.none import NoneCorrector
from refino.desktop.etapas import (
    CAMPO_SEM_PROPOSTA,
    CHAVE_DA_ETAPA,
    ETAPAS,
    _proposta_da_maquina,
)
from refino.desktop.janela.aparencia import (
    FiltroDaRoda,
    sem_roda_sem_foco,
    valor_do_campo,
    veste_de_botao,
)
from refino.desktop.janela.artefatos import (
    ARTEFATOS_DA_JANELA,
    CAMPOS_POR_ARTEFATO,
    CHAVES_DO_RESULTADO,
    FORMATO_METABOANALYST,
)
from refino.desktop.janela.componentes import (
    ComboComNomes,
    RotuloQueSome,
    com_maiuscula,
    desliga_com_motivo,
    em_grade,
    religa,
)
from refino.desktop.janela.expansao import ExpansaoSuave
from refino.desktop.janela.nomes import NOMES_DAS_OPCOES, NOMES_DOS_CORRETORES

# A capacidade do processo é lida ao vivo: `EM_REGIME_SERIAL`, abaixo, é o retrato da importação,
# e `n_jobs_efetivo` consulta o regime no instante da pergunta.
from refino.desktop.paralelismo import SERIE, n_jobs_efetivo, regime_do_processo
from refino.domain.analysis import PADRAO_DA_ANALISE, AnalysisSpec, analise_para_json
from refino.domain.labels import (
    ClassifierKind,
    Correction,
    RsdVariant,
    UnivariateTest,
)
from refino.domain.lotes import qcs_por_lote_declarado
from refino.interface.decorrido import (
    DepositoEmMemoria,
    agora,
    carimba_disparo,
    decorrido_da_leitura,
)
from refino.interface.fluxo import (
    CONTROLES_DE_SOBRESCRITA,
    DICA_DA_FAMILIA,
    DICA_DO_TESTE_UNIVARIADO,
    EIXOS_DA_GRADE,
    FAMILIAS_DE_ANALISE,
    NOMES_DO_TESTE_UNIVARIADO,
    estimativa_da_execucao,
    estimativa_da_medicao,
    familia_e_barata,
    prosa_da_concordancia,
    prosa_da_equivalencia,
    prosa_da_medicao,
    prosa_das_etapas_pedidas,
    prosa_do_numero,
    prosa_do_par,
)
from refino.metrics.crossval import jackknife_cost_ratio, jackknife_qc_minimum

if TYPE_CHECKING:  # pragma: no cover - só para o mypy
    from collections.abc import Callable, Sequence

    Relogio = Callable[[], float]

__all__ = [
    "ABA_DE_LIMITES",
    "CAMPOS_POR_ARTEFATO",
    "CAMPO_DO_INVENTARIO",
    "CHAVE_DO_INICIO",
    "CORRETOR_EM_REPOUSO",
    "DICA_DO_LOTE",
    "EIXOS_DA_GRADE",
    "EIXO_DOS_LOTES",
    "EIXO_DO_CORRETOR",
    "EM_CURSO",
    "ETAPA_INICIAL_DA_EXECUCAO",
    "FIXOS_DECLARADOS",
    "FRASE_DA_ABA_DE_LIMITES",
    "INVENTARIO_EM_REPOUSO",
    "INVENTARIO_SEM_GRUPO",
    "ITEM_DA_ESCOLHA_AUTOMATICA",
    "NOME_DA_ETAPA_INICIAL",
    "PROGRESSO_EM_REPOUSO",
    "RESULTADO_EM_REPOUSO",
    "ROTA_DA_EXECUCAO",
    "ROTULO_DE_CONCLUSAO",
    "SEMENTE_DA_JANELA",
    "SEM_EXECUCAO",
    "SEM_MEDICAO",
    "SEM_RESSALVA",
    "CombinacaoProibidaError",
    "GrupoNoInventario",
    "PainelDePipeline",
    "PropostaAusenteError",
    "e_corpo_de_aceite",
    "e_corpo_de_leitura",
    "inventario_do_corpo",
    "rota_do_progresso",
    "rota_do_relatorio",
    "rotulo_da_caixa_do_intervalo",
    "texto_da_etapa_em_curso",
    "texto_da_leitura",
    "texto_do_corretor",
    "texto_do_resultado",
]

ROTA_DA_EXECUCAO: Final = "/execucoes"

# O código que significa 'isto ainda não é o resultado'. A terminalidade do trabalho é a negação
# dele, e um `!= 202` solto não diz de onde vem o 202.
EM_CURSO: Final = 202

# A chave do carimbo do disparo. Só `carimba_disparo`, em `interface/decorrido.py`, a escreve: o
# portão
# do funil acusa qualquer escrita no depósito fora das funções legítimas, sem olhar o nome da chave.
CHAVE_DO_INICIO: Final = "inicio_execucao"

SEM_EXECUCAO: Final = "nenhuma execução disparada"

# Os rótulos de repouso dizem de que são, porque dois rótulos com o mesmo texto inicial não mostram
# estado. O prefixo fica na constante, e não num `setText` da construção, porque `recebe_leitura` e
# `mostra_resultado` substituem o texto inteiro e o prefixo sumiria na primeira leitura.
PROGRESSO_EM_REPOUSO: Final = f"Progresso — {SEM_EXECUCAO}"
# A palavra da conclusão, que os roteiros de captura também procuram. Uma constante só, para a tela
# e quem a espera não divergirem.
ROTULO_DE_CONCLUSAO: Final = "Concluído"
RESULTADO_EM_REPOUSO: Final = f"Resultado — {SEM_EXECUCAO}"
# O terceiro rótulo do grupo da execução, prefixado pela mesma razão. Quem o escreve é
# `aceita_disparo`, quando o corretor passa a ser sabido.
CORRETOR_EM_REPOUSO: Final = f"Corretor — {SEM_EXECUCAO}"


class FotoDoGrupoDaExecucao(NamedTuple):
    """O que o grupo da execução mostra: os três rótulos.

    A janela guarda isto com o desfecho de uma execução, pela identidade da matriz, e repõe quando a
    pessoa volta a ela. Guarda o texto que a tela escreveu em vez de reler o desfecho: reler
    `recebe_leitura` na volta recalcularia o decorrido a partir do disparo corrente, que pode ser o
    de outra execução.

    A barra de progresso fica fora. Ela é a da Análise e mede a etapa como a tela a pede agora
    (`_poe_a_barra_no_estado`); repor a de outro instante a deixaria cheia ao lado de
    'desatualizada'.
    """

    progresso: str
    resultado: str
    corretor: str


# O grupo da execução como a construção o deixa: os três rótulos em repouso.
GRUPO_DA_EXECUCAO_EM_REPOUSO: Final = FotoDoGrupoDaExecucao(
    PROGRESSO_EM_REPOUSO, RESULTADO_EM_REPOUSO, CORRETOR_EM_REPOUSO
)

# O nome da primeira etapa de `/execucoes`, a única que este painel nomeia. A correção é a primeira
# etapa e a que domina o tempo, então a contagem 'k/n etapas' fica parada durante ela; o nome diz o
# que está a correr. Copiar a tupla inteira de `api/rotas.py` seria outra fonte de verdade, que
# nomearia a etapa errada quando a rota ganhasse uma etapa nova. Este nome é conferido contra a rota
# por AST em `tests/test_desktop_execucao_em_curso.py`. `ProgressoResposta` já serve
# `etapa_em_curso`, mas este painel ainda não o lê.
ETAPA_INICIAL_DA_EXECUCAO: Final = "correcao"

# O nome que a tela mostra sai da constante acima, e não de um literal na frase. A indexação é
# deliberada: uma primeira etapa sem nome aqui levanta `KeyError` na importação, em vez de a tela
# nomear a etapa antiga.
_NOME_PARA_QUEM_OPERA: Final[dict[str, str]] = {"correcao": "a correção"}
NOME_DA_ETAPA_INICIAL: Final = _NOME_PARA_QUEM_OPERA[ETAPA_INICIAL_DA_EXECUCAO]

# O ponteiro e a frase da ressalva da escolha automática. A frase é copiada literalmente do item da
# aba de limites, e um teste confere que a aba existe, que o item existe e que a frase está no corpo
# dele. Uma ressalva escrita aqui seria uma terceira redação da mesma limitação.
#
# `ABAS` lê daqui o título da última aba: uma fonte só para o nome que a barra mostra e para o que
# as frases de ressalva citam.
ABA_DE_LIMITES: Final = "Métodos e limites"
ITEM_DA_ESCOLHA_AUTOMATICA: Final = "Decidir tudo automaticamente, sem você"
FRASE_DA_ABA_DE_LIMITES: Final = "ela não supera a regra mais simples possível"

# Os campos numéricos de corte mostram o vazio com `specialValueText` (`SEM_CORTE` e as sentinelas
# abaixo): `QSpinBox` sempre tem um valor, e o Qt mostra esse texto no lugar do mínimo. O mínimo
# nunca é enviado, porque `escolhas_do_disparo` o traduz para `None`; um `0` que viajasse
# confundiria 'corte de zero', que a API recusa, com 'sem corte'.

# O que o painel diz quando o método escolhido não tem ressalva medida. Um rótulo em branco ao lado
# do seletor se leria como 'a ressalva não carregou'.
SEM_RESSALVA: Final = "Sem ressalva medida para este método"

# A frase de antes da medição. Um rótulo em branco ao lado do botão se leria como 'a medição
# falhou'; esta diz que ela ainda não correu e quanto custa, porque o botão dispara minutos de
# trabalho.
SEM_MEDICAO: Final = (
    "A medição ainda não correu. Ela ajusta os três corretores sobre os seus QCs, um de cada "
    "vez, e leva **minutos** — o resultado diz qual deles de facto vence no seu dado."
)

# `CAMPOS_POR_ARTEFATO`, os campos deste painel que cada rota de artefato aceita, mora em
# `janela/artefatos.py`. Este módulo o reexporta porque `campos_do_artefato` o lê.

# Quantos lotes o vazio envia. `escolhas_do_disparo` traduz o mínimo do campo para `None`, e quem
# dispara precisa de um número: `POST /execucoes` exige `n_batches`, e as rotas de artefato o têm
# com padrão `1`. Uma constante só, para o rótulo do campo e o payload não divergirem.
LOTES_QUANDO_VAZIO: Final = 1

# Os três botões do passo a passo e o estado de cada etapa, que vai em palavras ao rodapé da página
# de cada etapa, ao lado do botão.
ROTULOS_DAS_ETAPAS: Final[dict[int, str]] = {
    1: "1. Corrigir",
    # 'Normalizar', e não o nome da rota: o botão mora na aba 'Normalização', cujo aviso diz
    # 'Normalizar de novo'.
    2: "2. Normalizar",
    3: "3. Analisar e registrar",
}
ETAPA_DA_CHAVE: Final[dict[str, int]] = {chave: etapa for etapa, chave in CHAVE_DA_ETAPA.items()}
TEXTO_DO_ESTADO_DA_ETAPA: Final[dict[str, str]] = {
    "nao_feita": "não feita",
    "em_curso": "em curso",
    "em_dia": "em dia",
    "desatualizada": "desatualizada — os parâmetros mudaram depois; refaça",
}


def rotulo_da_caixa_do_intervalo(n_qc: int | None) -> str:
    """O rótulo da caixa do intervalo por barra, com o custo pelos QCs do estudo.

    O custo vai no rótulo, e não só na dica, porque é ele que decide se a pessoa marca: a caixa vem
    desmarcada para que quem não a usa não pague nada, e quem a marca precisa saber quanto paga. O
    fator é `(n+1)/2`, a razão da contagem de ajustes (`metrics.crossval.jackknife_cost_ratio`).
    Sem a validação o número de QCs é desconhecido, e o rótulo mostra a conta em vez de um fator
    inventado.
    """
    if n_qc is None or n_qc < 1:
        return "Intervalo em cada barra — a medição fica (n+1)/2 vezes mais lenta, com n QCs"
    fator = f"{jackknife_cost_ratio(n_qc):g}".replace(".", ",")
    return f"Intervalo em cada barra — a medição fica {fator} vezes mais lenta com {n_qc} QCs"


def frase_dos_qcs_por_lote(
    contagens: tuple[int, ...], *, nos_cortes_propostos: bool = False
) -> str:
    """A linha 'QCs por lote: 3, 2 e 2.', debaixo do número de lotes declarados.

    Ela diz qual partição a correção vai usar. Mantido o número de lotes que o diagnóstico propôs, a
    correção corta nas fronteiras dele; com outro número, são blocos iguais pela ordem de injeção.
    """
    numeros = [str(c) for c in contagens]
    lista = numeros[0] if len(numeros) == 1 else ", ".join(numeros[:-1]) + " e " + numeros[-1]
    if nos_cortes_propostos:
        return f"QCs por lote: {lista} — nas fronteiras que o diagnóstico propôs."
    return f"QCs por lote: {lista}."


# A dica da caixa do intervalo por barra. O rótulo diz o custo; a dica diz o que o intervalo mede e
# o que não mede, como o rodapé da figura. Os mínimos de QCs vêm do núcleo (`jackknife_qc_minimum`),
# e não de uma cópia aqui, porque o LOESS exige mais que o piso.
_DICA_DO_INTERVALO: Final = (
    "Desenha em cada barra do painel da correção um intervalo jackknife-t sobre os QCs, com "
    "reajuste: cada QC é tirado por vez e a régua inteira é refeita sem ele. Mede quanto a barra "
    "depende de QUAIS QCs foram injetados, e não das features. O nível, 95%, é nominal: a "
    "cobertura não foi medida.\n\n"
    f"Com menos de {jackknife_qc_minimum(NoneCorrector())} QCs num lote "
    f"({jackknife_qc_minimum(LoessCorrector())} com o LOESS) aquela barra fica sem intervalo, e o "
    "corretor dela custa o mesmo que sem a caixa.\n\n"
    "O fator do rótulo é a contagem de ajustes, (n+1)/2. No relógio, a única medição é a do LOESS "
    "no neg_c18, com 7 QCs: 4,05 vezes. O SERRF não foi medido.\n\n"
    "Não muda o vencedor, nem o Palpite, nem a regra de empate dele."
)


# A dica da caixa dos núcleos traz cada número com a população em que foi medido, porque nenhum é
# invariante. O SERRF: `neg_c18` dentro do executável, numa máquina de 12 núcleos, com as duas
# corridas do mesmo binário (em lançamentos separados, porque a chave do cache exclui `n_jobs`). O
# LOESS: `neg_c18` fora do executável, numa máquina de 4 núcleos partilhados, mediana de três
# corridas; o ganho variou de 1,38x a 1,97x com o modo e a carga. Os tetos das duas máquinas são
# diferentes (4x e 12x), então a dica diz que os dois pares não se comparam. A memória: `pos_hilic`
# com o SERRF no executável, 12 núcleos lógicos, uma corrida por célula, 3.246,2 MiB com a caixa
# marcada e 947,7 MiB desmarcada, medida antes de o LOESS ler a caixa.
#
# `n_jobs_efetivo` não tem teto: o pico de memória cresce com os núcleos de quem recebe, e a dica
# existe para a caixa ser uma escolha informada. A caixa governa as rotas longas; o multiverso tem
# eixo próprio (`corrector_jobs`), que ela não alcança.
_DICA_DOS_NUCLEOS: Final = (
    "Divide o SERRF e o LOESS pelos processadores da sua máquina. O resultado é IDÊNTICO nos "
    "dois regimes; o que muda é o tempo.\n\n"
    "No SERRF, medido no neg_c18 dentro do executável, em 12 núcleos: 490,1 s em série contra "
    "82,3 s — 6x. No LOESS, medido fora do executável, numa máquina de 4 núcleos partilhados: "
    "no neg_c18, 16,2 s em série contra 8,9 s; o ganho variou com o modo e com a carga. Os "
    "dois pares vêm de máquinas com números de núcleos diferentes, e não se comparam.\n\n"
    "CUSTA MEMÓRIA: no executável, no pos_hilic com o SERRF, até 3,2 GB de pico com 12 "
    "núcleos, contra 948 MB em um só — uma corrida de cada, antes de o LOESS a usar. O "
    "número cresce com os processadores da SUA máquina — cada um carrega a sua cópia do "
    "numpy e do scikit-learn.\n\n"
    "O none não a usa: ele não calcula.\n\n"
    "Desticar põe a CORREÇÃO e a MEDIÇÃO em um processador só — use se a máquina "
    "ficar pesada demais enquanto o cálculo corre. A aba «Teste de robustez (beta)» tem regime "
    "próprio e não segue esta caixa."
)


# A ressalva da política que o Palpite da janela mede, mostrada atrás de 'Ver o recorte medido'. É a
# ressalva da proposta que 'Seguir sugestões do app' aplica. Uma política renomeada em `cli.py`
# levanta `KeyError` na importação.
CAVEAT_DO_PALPITE: Final = AUTO_CAVEATS[AUTO_MEDIDA]
# A variante do RSD com que a aba 'Correção' abre: 'Com todos os QCs'.
VARIANTE_DO_RSD_EM_REPOUSO: Final[RsdVariant] = "plain"

# O eixo do corretor, que 'Seguir sugestões do app' preenche com o Palpite. A constante cobre o eixo
# como `_controles_de_valor`, `_proposta` e `_tocados` o conhecem. O `"correction"` do payload de
# `campos_da_execucao` é outra coisa, o nome do campo na API: a mesma string hoje, com referente
# diferente, e amarrá-lo aqui faria uma renomeação de controle mudar o que viaja no `POST`.
EIXO_DO_CORRETOR: Final = "correction"
# O eixo do número de lotes, que a leitura do diagnóstico propõe junto com os cortes.
EIXO_DOS_LOTES: Final = "n_batches"
# O campo dos cortes de lote no pedido, que viaja com o número de lotes quando ele é o proposto.
CAMPO_DOS_CORTES: Final = "batch_cuts"
# O botão que produz a proposta de cada eixo, e o nome do eixo na língua da tela, para a recusa de
# `PropostaAusenteError` dizer o gesto que resolve e não um nome de campo da API.
BOTAO_DA_PROPOSTA: Final[dict[str, str]] = {
    EIXO_DO_CORRETOR: "«Palpite»",
    EIXO_DOS_LOTES: "«Diagnóstico»",
}
NOME_DO_EIXO: Final[dict[str, str]] = {
    EIXO_DO_CORRETOR: "o corretor",
    EIXO_DOS_LOTES: "o número de lotes",
}

# As duas chaves de artefato que o lote dos três percorre: as do Resultado da Correção, uma por
# formato. Quem dispara o lote é o 'Gerar os três' de lá, no formato escolhido. Elas saem do mapa de
# `artefatos.py`, e não são escritas aqui: um erro de digitação faria o lote abortar no clique, na
# primeira perna, com 'artefato desconhecido'; uma chave que o mapa não tenha levanta na
# importação.
CHAVE_DO_LOTE_METABO: Final = CHAVES_DO_RESULTADO[1][FORMATO_METABOANALYST]
CHAVE_DO_LOTE_CORRIGIDA: Final = next(
    chave for chave in CHAVES_DO_RESULTADO[1].values() if chave != CHAVE_DO_LOTE_METABO
)

# A guarda confere que as duas chaves são da aba da Correção e de formatos diferentes. Se não
# fossem, os dois lotes escreveriam os mesmos nomes na mesma pasta.
assert {
    ARTEFATOS_DA_JANELA[CHAVE_DO_LOTE_METABO].etapa,
    ARTEFATOS_DA_JANELA[CHAVE_DO_LOTE_CORRIGIDA].etapa,
} == {1} and CHAVE_DO_LOTE_METABO != CHAVE_DO_LOTE_CORRIGIDA, (
    f"as chaves do lote {CHAVE_DO_LOTE_METABO!r} e {CHAVE_DO_LOTE_CORRIGIDA!r} têm de ser as dos "
    "dois formatos do Resultado da Correção"
)

# O texto do repouso do campo de lotes; o valor que viaja é `LOTES_QUANDO_VAZIO`.
SEM_LOTES: Final = f"{LOTES_QUANDO_VAZIO} — Não dividir em lotes"
SEM_CORTE: Final = "Vazio — não cortar por RSD"
# A sentinela do corte de ausentes, com a forma e a razão de `SEM_CORTE`: `missing_threshold` é
# `float | None`, e um `0` que viajasse confundiria 'corte de zero por ausentes', que não descarta
# feature nenhuma, com 'sem corte'.
SEM_CORTE_DE_AUSENTES: Final = "Vazio — não cortar por ausentes"
# A sentinela das duas frações de posto, com a mesma forma e uma diferença: vazio aqui não salta a
# etapa. Quem decide se ela corre é o método do combo; vazio devolve a fração à regra por tamanho da
# matriz, que é o que a etapa faz quando ninguém escolhe.
FRACAO_POR_TAMANHO: Final = "Vazio — regra por tamanho"
# O repouso da referência diz o que o vazio significa: a lista só existe depois de validar, e quem
# não validou precisa saber que pode escrever o nome.
SEM_REFERENCIA: Final = "Valide para escolher, ou escreva o nome"


def _em_regime_serial() -> bool:
    """Diz se este processo corre o `joblib` em série.

    O palpite de duração depende disso: no `serrf` a diferença entre os dois regimes chega a 6x. A
    resposta vem de `regime_do_processo()`, que o arranque resolveu: ele já viu o interpretador,
    aplicou os remendos e caiu para `SERIE` se algum falhou. `joblib.effective_n_jobs(-1)` não
    serve, porque responde o número de núcleos mesmo quando o `loky` não pode lançar worker nenhum;
    o palpite ficaria baixo justamente na queda para série, e um palpite baixo faz o profissional
    achar que a execução travou.

    É lido uma vez, na importação (`EM_REGIME_SERIAL`). Quem consulta o regime ao vivo é
    `n_jobs_efetivo`.
    """
    return regime_do_processo() == SERIE


EM_REGIME_SERIAL: Final = _em_regime_serial()

# A ordem do corretor é a do domínio, e não a alfabética, e isso é comportamento:
# `get_args(Correction)` põe `none` primeiro, o mesmo padrão de `--method` na CLI.
# `sorted(AVAILABLE_METHODS)` poria `loess` primeiro e corrigiria, por omissão, a matriz de quem não
# pediu correção. A lista é derivada do vocabulário e da disponibilidade (`AVAILABLE_METHODS`), as
# mesmas fontes da CLI; uma terceira lista seria outra fonte de verdade sobre o que a ferramenta
# sabe corrigir.
CORRETORES: Final[tuple[str, ...]] = tuple(
    m for m in get_args(Correction) if m in AVAILABLE_METHODS
)

# A lista dos três corretores fica na dica, e não no rótulo dos botões: no rótulo ela alargava o
# formulário além da largura da janela e fazia aparecer barra de rolagem horizontal. A ordem vem de
# `CORRETORES`, a mesma que `_inicia_lote` percorre.
DICA_DO_LOTE: Final = (
    f"Roda {', '.join(CORRETORES)} — nessa ordem — e escreve os três numa pasta que você "
    f"escolhe uma vez, com o método no nome de cada arquivo.\n\n"
    f"Espere minutos: o SERRF domina o custo."
)

# ---------------------------------------------------------------------------
# Os eixos da grade que o usuário escolhe
# ---------------------------------------------------------------------------
#
# `EIXOS_DA_GRADE` vem de `interface/fluxo.py` e é reexportado daqui, de onde o painel e os testes o
# leem.
# Estes eixos ficam com o usuário porque a ferramenta existe para variá-los: fixá-los faria cada
# execução ser uma célula da grade escolhida por nós, e a conclusão do multiverso mede justamente o
# quanto o resultado depende dessa escolha. O vocabulário é derivado de `domain/labels.py`, como em
# `CORRETORES`. Mudar o padrão de um eixo muda, em silêncio, o resultado de quem não mexe no
# controle.

# Quais eixos têm ressalva medida, por eixo. Dicionário de dicionários porque a chave de dentro é um
# rótulo de domínio e os rótulos se repetem entre eixos (`none` vale em todos): achatar faria a
# ressalva do `none` da imputação aparecer sob o escalonamento. Eixo vazio é declaração: nenhuma
# medição achou recorte que quem escolhe aqueles valores precise ver antes de escolher.
RESSALVAS_POR_EIXO: Final[dict[str, dict[str, Caveat]]] = {
    "ranking_filter": {},
    "ranking_filter_2": {},
    "imputation": IMPUTATION_CAVEATS,
    "sample_normalization": {},
    "transformation": {},
    "scaling": {},
}

# Os campos fixos do pedido, nenhum deles eixo da grade: a semente e `apresentacao`, que é fato da
# interação. Os eixos da grade e a variante do RSD são escolhas do usuário nesta janela, porque
# mudam o número que a tela mostra, e um controle escondido seria uma escolha nossa apresentada
# como propriedade do dado.
#
# A semente tem nome porque dois pedidos a leem: o da execução, por `FIXOS_DECLARADOS`, e o dos três
# artefatos, por `campos_do_artefato`. Duas cópias do `42` divergiriam na primeira mudança, e o
# resumo metodológico diria a semente errada.
SEMENTE_DA_JANELA: Final = 42

# O que o botão da curva do LOESS diz quando não há feature escrita.
FRASE_DA_CURVA_SEM_FEATURE: Final = (
    "Escolha ou escreva uma feature: a lista chega com a «Validação» dos arquivos."
)

FIXOS_DECLARADOS: Final[dict[str, Any]] = {
    "seed": SEMENTE_DA_JANELA,
    # Esta janela exibe o resultado, e é isso que se registra.
    "apresentacao": "exibida",
}


class PropostaAusenteError(ValueError):
    """Eixo de sobrescrita marcado sem proposta da máquina com que comparar.

    A janela recusa antes de a rota responder `422`, porque a mensagem da rota ('Faltou um campo, ou
    um valor não pertence ao vocabulário aceito') culparia o profissional por um defeito da tela.
    `janela/arquivos.py::arquivos_para_envio` levanta pela mesma razão, em vez de enviar multipart
    vazio.
    """


class CombinacaoProibidaError(ValueError):
    """`imputation='none'` escolhida junto de normalização ativa, combinação que o domínio proíbe.

    A rota também recusa, na construção de `PipelineSpec`, mas a mensagem que sobe é a do domínio,
    em linguagem de repositório, e o painel mostra o `detail` como veio. Esta guarda explica a
    recusa a quem a tentou pela janela; a da rota continua a guardar a CLI, a API e outros clientes.
    """


# ---------------------------------------------------------------------------
# As caixas das análises
# ---------------------------------------------------------------------------

# O campo de `AnalysisSpec` que NÃO é bandeira: `classifiers` é uma tupla de rótulos, e por isso
# recebe uma caixa por rótulo em vez de uma só. Nomeado porque o laço que monta as caixas precisa
# de o distinguir, e um literal solto ali seria a comparação que ninguém liga à tabela.
CAMPO_DOS_CLASSIFICADORES: Final = "classifiers"

# O campo do corpo de `POST /execucoes` por onde o contrato de análise viaja: um campo JSON, e não
# vinte de formulário (o motivo está em `api/rotas.py::_analise_ou_422`). O portão do contrato entre
# a janela e a API lê esta chave.
CAMPO_DA_ANALISE: Final = "analise_json"


# ---------------------------------------------------------------------------
# O inventário de grupos
# ---------------------------------------------------------------------------

# O campo de `DiagnosticoResposta` onde o inventário viaja. Nomeado porque duas grafias para a
# mesma chave é a segunda fonte de verdade sobre o contrato, e o portão do contrato lê o literal.
CAMPO_DO_INVENTARIO: Final = "grupos_disponiveis"

# Os dois repousos dizem de que são, como `PROGRESSO_EM_REPOUSO`. Os textos diferem porque pedem
# ações opostas: 'a leitura ainda não correu' pede clicar em 'Leitura', e 'a leitura correu e não há
# grupo' pede conferir a linha `LABEL` do arquivo.
INVENTARIO_EM_REPOUSO: Final = (
    "Grupos desta matriz — a «Leitura» ainda não correu. É ela que lê a linha `LABEL` do seu "
    "`measurements` e diz quais grupos existem e com quantas amostras cada um."
)
INVENTARIO_SEM_GRUPO: Final = (
    "Grupos desta matriz — **nenhum**. A leitura correu e não achou grupo a contrastar fora do "
    "QC: confira a linha `LABEL` do seu `measurements`, que é a **segunda linha** do arquivo."
)


class GrupoNoInventario(NamedTuple):
    """Um grupo e o seu `n`, já atravessado o fio.

    O espelho, do lado da janela, de `api.esquemas.GrupoComContagem`. Tipo próprio em vez de `dict`
    para cada consumidor não repetir a leitura das chaves; são três: a resposta do diagnóstico, a
    mensagem de recusa da rota e o combo da aba de multiverso.

    `n` conta amostras, e não features: as amostras da matriz sem o QC, que é régua e não grupo,
    como `inventario_de_grupos` em `api/rotas.py` as produz.
    """

    nome: str
    n: int


def inventario_do_corpo(corpo: Any) -> tuple[GrupoNoInventario, ...] | None:
    """O inventário de grupos de um corpo de `/matrizes/diagnosticar`, ou `None`.

    `None` quer dizer que a resposta não traz inventário: outro corpo, uma rota que não o calcula,
    um erro. `()` quer dizer que traz e não há grupo fora do QC, estado legítimo de uma matriz sem
    coluna de grupo. Colapsar os dois faria uma resposta de erro apagar os combos do multiverso como
    uma matriz sem grupos os apaga.

    Entrada malformada devolve `None` em vez de levantar: isto corre dentro de um slot do Qt, onde
    uma exceção iria para o `stderr` e o painel pararia de atualizar em silêncio.
    """
    if not isinstance(corpo, dict):
        return None
    bruto = corpo.get(CAMPO_DO_INVENTARIO)
    if not isinstance(bruto, list | tuple):
        return None
    inventario: list[GrupoNoInventario] = []
    for grupo in bruto:
        if not isinstance(grupo, dict):
            return None
        nome, n = grupo.get("nome"), grupo.get("n")
        if not isinstance(nome, str) or not isinstance(n, int) or isinstance(n, bool):
            return None
        inventario.append(GrupoNoInventario(nome, n))
    return tuple(inventario)


def rota_do_progresso(identificador: str) -> str:
    """A rota da reclamação de um trabalho.

    Função, e não f-string no ponto de uso, porque o portão de contrato atribui cada leitura a uma
    rota, e uma rota com nome é mais fácil de atribuir que uma montada dentro de um `lambda` de
    despacho.
    """
    return f"/trabalhos/{identificador}"


def rota_de_cancelamento(identificador: str) -> str:
    """A rota da parada de um trabalho longo. Mesma razão da irmã acima."""
    return f"/trabalhos/{identificador}/cancelar"


def rota_do_relatorio(identificador: str) -> str:
    """A rota do registro do corte de uma exportação. Mesma razão das irmãs.

    Separada da rota do artefato porque o artefato é `text/csv`: os bytes ocupam o corpo, e a lista
    das features removidas não cabe num cabeçalho. A janela pede as duas, e a segunda alimenta o
    arquivo irmão.
    """
    return f"/trabalhos/{identificador}/relatorio"


def _detalhe(corpo: Any) -> str:
    """O `detail` de uma resposta de erro, na forma que o FastAPI emite. Vazio quando não há."""
    if not isinstance(corpo, dict):
        return ""
    detalhe = corpo.get("detail")
    if isinstance(detalhe, str):
        return detalhe
    if isinstance(detalhe, list):
        return "; ".join(str(item.get("msg", item)) for item in detalhe if isinstance(item, dict))
    return ""


def _texto_do_decorrido(decorrido: tuple[float, bool] | None) -> str:
    """A linha do decorrido, ou vazio quando não há disparo conhecido.

    O sufixo distingue o congelado do vivo: um número parado sem rótulo convida a lê-lo como a
    duração, e ele é um teto, o intervalo entre o disparo e a leitura que viu o fim. Sem disparo
    conhecido a linha não aparece, porque contar a partir de um disparo alheio daria um número
    plausível sobre a coisa errada.
    """
    if decorrido is None:
        return ""
    segundos, congelado = decorrido
    rotulo = "até a leitura do fim" if congelado else "desde o disparo"
    return f"decorrido: {segundos:.0f} s {rotulo}"


def e_corpo_de_leitura(corpo: Any) -> bool:
    """Diz se o corpo é uma leitura de `GET /trabalhos/{id}`, sem olhar o código.

    `aceita_disparo` recebe só `(corpo, instante)` e precisa recusar uma leitura antes de limpar o
    que quer que seja; a função existe para o discriminador ter uma cópia só. `ProgressoResposta`
    declara `estado` obrigatório, e `TrabalhoAceito` não o tem (traz `identificador`, `total` e
    `reclamado`). É essa assimetria de esquema que faz do campo um discriminador, e não uma
    heurística.
    """
    return isinstance(corpo, dict) and "estado" in corpo


def e_corpo_de_aceite(codigo: int, corpo: Any) -> bool:
    """Diz se o `202` é o aceite do `POST`, e não uma leitura de progresso.

    Tem dois leitores: `texto_da_leitura`, para escolher a formatação, e `principal.py`, para saber
    quando chamar `aceita_disparo`. O discriminador é `estado`, via `e_corpo_de_leitura`, e as duas
    funções concordam que um corpo que não é `dict` não é leitura nem aceite.
    """
    return codigo == EM_CURSO and isinstance(corpo, dict) and not e_corpo_de_leitura(corpo)


def texto_da_etapa_em_curso(corpo: Any, *, corretor: str | None, desde_s: float | None) -> str:
    """Diz qual etapa corre agora, e há quanto tempo. Vazio quando não se sabe.

    A frase só afirma o que a rota mede. `marca()` dispara depois de cada etapa terminar, e
    `_executa_trabalho` põe `executando` antes da primeira; com `estado == "executando"`,
    `feito == k` quer dizer que a etapa `k + 1` está a correr. Não há fração dentro da etapa, e a
    frase diz isso: a barra só anda quando uma etapa termina.

    - estado diferente de `executando`: nada (`pendente` quer dizer que a thread ainda não pegou);
    - `feito == 0`: a correção, com o corretor quando ele é sabido;
    - `0 < feito < total`: a etapa `feito + 1`, sem nome (ver `ETAPA_INICIAL_DA_EXECUCAO`);
    - `feito >= total`: nada, porque todas foram marcadas e falta o resultado.

    `desde_s` é um piso, e a frase diz 'há pelo menos': o fim da etapa anterior aconteceu entre duas
    leituras. Na primeira etapa ele não é pedido, porque o decorrido desde o disparo já é o tempo
    dela.
    """
    if not isinstance(corpo, dict) or corpo.get("estado") != "executando":
        return ""
    feito, total = corpo.get("feito"), corpo.get("total")
    if not isinstance(feito, int) or not isinstance(total, int) or not 0 <= feito < total:
        return ""
    if feito == 0:
        com = f" com «{corretor}»" if corretor else ""
        return (
            f"agora: etapa 1 de {total}, {NOME_DA_ETAPA_INICIAL}{com} — a barra só anda quando "
            f"uma etapa termina, e {NOME_DA_ETAPA_INICIAL} não reporta fração por dentro"
        )
    tempo = f", há pelo menos {desde_s:.0f} s" if desde_s is not None and desde_s >= 1 else ""
    return f"agora: etapa {feito + 1} de {total}{tempo}"


def texto_do_corretor(corpo: Any, *, delegou: bool | None, pedido: str | None) -> str:
    """Diz qual corretor esta execução corre e quem o escolheu, lendo o aceite do `POST`.

    `TrabalhoAceito.delegacao` traz a escolha da política quando o pedido delegou o corretor. Sem
    esta linha, nada na janela diria que corretor corrigiu o dado. Quatro desfechos:

    - manual: o corretor pedido, 'escolhido por você';
    - delegado: o corretor da política, com a ressalva dela;
    - delegado com abstenção: o mesmo, e 'ABSTEVE-SE', que quer dizer 'não sei' e não 'tanto faz';
    - delegado sem `delegacao` no corpo: a frase diz que a resposta não disse qual.

    O quarto não cai num `else`: atribuí-lo ao usuário afirmaria o oposto do que aconteceu, e ler o
    seletor também erraria, porque ele pode não mostrar o que correu. `delegou` é `None` quando a
    janela não registrou o clique (um aceite que chega sem `_pede_disparo`), e aí a frase também não
    inventa.

    A ressalva da política aprendida aponta para o item da aba de limites que carrega o número, em
    vez de o repetir; a frase é a de `FRASE_DA_ABA_DE_LIMITES`.
    """
    delegacao = corpo.get("delegacao") if isinstance(corpo, dict) else None
    if isinstance(delegacao, dict) and delegacao.get("correction"):
        escolhido = f"«{delegacao['correction']}», escolhido pela política automática"
        politica = delegacao.get("politica")
        if politica == AUTO_APRENDIDA:
            ressalva = (
                f"Ressalva: {FRASE_DA_ABA_DE_LIMITES} — escolher sempre o mesmo método. O "
                f"número está na aba «{ABA_DE_LIMITES}», item «{ITEM_DA_ESCOLHA_AUTOMATICA}»."
            )
        else:
            caveat = AUTO_CAVEATS.get(str(politica))
            ressalva = f"Ressalva: {caveat.resumo}." if caveat is not None else ""
        partes = [f"Corretor — {escolhido}."]
        if delegacao.get("abstains"):
            nao_medida = "" if delegacao.get("assessable", True) else ", ou não pôde ser medida"
            partes.append(
                "Ela ABSTEVE-SE: a margem entre os dois melhores corretores ficou abaixo do "
                f"limiar{nao_medida}. Leia como «não sei», e decida você."
            )
        if ressalva:
            partes.append(ressalva)
        return " ".join(partes)
    if delegou:
        return (
            "Corretor — escolhido pela política automática, e a resposta do disparo não disse "
            "qual. A execução corre com ele mesmo assim."
        )
    if delegou is False and pedido:
        return f"Corretor — «{pedido}», escolhido por você."
    return "Corretor — esta janela não registou o que foi pedido neste disparo."


def texto_da_leitura(
    codigo: int, corpo: Any, decorrido: tuple[float, bool] | None, *, etapa: str = ""
) -> str:
    """Formata uma leitura de `GET /trabalhos/{id}`, roteada pelo código HTTP.

    `etapa` é a frase de `texto_da_etapa_em_curso`, composta por quem tem o estado
    (`recebe_leitura`); vazia, a linha não a traz. O roteamento é pelo código porque `terminou=True`
    nunca chega ao cliente (ver a docstring do módulo).

    - `202`: `estado`, `feito/total` e o decorrido vivo;
    - `200`: a conclusão e o decorrido congelado (o resultado detalhado é de `texto_do_resultado`);
    - `404`: identificador desconhecido, com o `detail`, que não é falha do trabalho;
    - outro: falhou, com a etapa nomeada no `detail`.

    O `404` fica separado do `409` e do `500` porque pedem ações opostas: redisparar, ou corrigir o
    dado. A própria rota declara essa separação.
    """
    linha_do_tempo = _texto_do_decorrido(decorrido)

    if codigo == EM_CURSO:
        if not isinstance(corpo, dict):
            return SEM_EXECUCAO
        # Dois corpos chegam com `202`: o aceite de `POST /execucoes` (`identificador`, `total`,
        # `reclamado`) e o progresso de `GET /trabalhos/{id}` (`ProgressoResposta`, com `estado` e
        # `feito`). O discriminador é `estado`, que `ProgressoResposta` declara obrigatório.
        if e_corpo_de_aceite(codigo, corpo):
            aceite = [f"Aceita: {corpo.get('total')} etapas"]
            if corpo.get("reclamado"):
                aceite.append("trabalho já existente, reclamado")
            if linha_do_tempo:
                aceite.append(linha_do_tempo)
            return " · ".join(aceite)
        # O estado vem da API em minúscula; a frase abre a linha, então a letra sobe.
        partes = [
            com_maiuscula(
                f"{corpo.get('estado')}: {corpo.get('feito')}/{corpo.get('total')} etapas"
            )
        ]
        if etapa:
            partes.append(etapa)
        # `celula_que_falhou` e `erro` não são lidos aqui: o produtor só os preenche no estado
        # `falhou`, que implica `terminou`, e esta rota serve progresso só enquanto o trabalho
        # corre. O texto da falha vem do braço de erro, abaixo, que lê `detail`.
        if linha_do_tempo:
            partes.append(linha_do_tempo)
        return " · ".join(partes)

    if codigo == 200:
        partes = [ROTULO_DE_CONCLUSAO]
        if linha_do_tempo:
            partes.append(linha_do_tempo)
        return " · ".join(partes)

    detalhe = _detalhe(corpo)
    rotulo = "Identificador desconhecido" if codigo == 404 else "O trabalho falhou"
    partes = [f"{rotulo} (HTTP {codigo})"]
    if detalhe:
        partes.append(detalhe)
    if linha_do_tempo:
        partes.append(linha_do_tempo)
    return " · ".join(partes)


def texto_do_resultado(corpo: Any) -> str:
    """O que a reclamação devolveu: o id da execução, o estado de concordância e o antes/depois.

    `estado_concordancia` sai em prosa (`PROSA_DA_CONCORDANCIA`, de `interface/fluxo.py`, em bijeção
    com
    `EstadoDeConcordancia` por teste), com a chave crua entre parênteses, que é o que o registro
    grava.

    Os nomes de campo lidos aqui não são declarados em lugar nenhum: o teste extrai por AST os
    literais que cada formatador usa como chave e confere o conjunto contra `ProgressoResposta`,
    `ExecucaoResposta` e `RsdAntesEDepois`. Um nome plausível que não existe no esquema devolveria
    `None` sem traceback, indistinguível de funcionar; um literal novo escrito aqui entra na
    conferência sozinho.
    """
    if not isinstance(corpo, dict):
        return SEM_EXECUCAO
    partes = [f"execução {corpo.get('run_id')}"]
    if corpo.get("estado_concordancia"):
        partes.append(f"concordância: {prosa_da_concordancia(str(corpo['estado_concordancia']))}")
    # Os três números de RSD passam por `prosa_do_numero`, a política de quatro casas que a aba de
    # qualidade usa para as mesmas medianas. Sem ela a linha mostraria dezassete casas, que afirmam
    # uma precisão que a medida não tem, e outra política poria dois formatos do mesmo número em
    # duas abas. `n_features_saida` e `run_id` ficam crus porque são contagem e identificador.
    if corpo.get("qc_rsd_median") is not None:
        partes.append(f"RSD mediana dos QCs: {prosa_do_numero(corpo['qc_rsd_median'])}")
    if corpo.get("n_features_saida") is not None:
        partes.append(f"{corpo['n_features_saida']} features na saída")
    comparacao = corpo.get("rsd_antes_e_depois")
    if isinstance(comparacao, dict):
        partes.append(
            f"antes e depois: {prosa_do_numero(comparacao.get('mediana_antes'))} → "
            f"{prosa_do_numero(comparacao.get('mediana_depois'))}"
        )
    return " · ".join(partes)


def _compacta_formulario(forma: QFormLayout) -> None:
    """Alinha os controles às medidas do formulário da interface de referência."""
    # O título nativo ocupa 24 px; o primeiro campo começa 16 px abaixo dele.
    forma.setContentsMargins(0, 40, 0, 20)
    forma.setHorizontalSpacing(16)
    forma.setVerticalSpacing(12)
    forma.setLabelAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
    forma.setFormAlignment(Qt.AlignmentFlag.AlignTop)
    forma.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
    for linha in range(forma.rowCount()):
        item = forma.itemAt(linha, QFormLayout.ItemRole.LabelRole)
        rotulo = item.widget() if item is not None else None
        if isinstance(rotulo, QLabel):
            rotulo.setFixedWidth(192)
            rotulo.setWordWrap(True)
            rotulo.setProperty("rotuloFormulario", True)
        item = forma.itemAt(linha, QFormLayout.ItemRole.FieldRole)
        campo = item.widget() if item is not None else None
        if isinstance(campo, (QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox)):
            campo.setFixedHeight(32)


class PainelDePipeline(QWidget):
    """Os controles das três etapas, o progresso delas e o par de sobrescrita.

    É o controlador que as abas 'Correção', 'Normalização' e 'Análise estatística' compõem. Não tem
    disposição própria: os grupos nascem aqui e a janela os põe nas páginas das etapas
    (`AbaDeEtapa`), com o botão, o estado, a barra e as notas de cada etapa na linha da ação da
    página dela.

    `disparar_etapa` é o sinal que a janela liga à piscina; a Análise monta o próprio `/execucoes` a
    partir da entrada escolhida. O progresso é lido pela sondagem automática da janela, e o lote dos
    três corretores é o 'Gerar os três' do Resultado da Correção
    (`execucao.ResultadoDaEtapa.gerar_os_tres`).

    O painel emite sinais em vez de receber um `chama` injetado: um chamador invocado daqui deixaria
    `envia` alcançável por uma cadeia que o portão de AST não segue. Emitir mantém a única aresta de
    rota na janela.
    """

    # A etapa do passo a passo pedida: 1, 2 ou 3. O painel emite; a janela despacha.
    disparar_etapa = Signal(int)
    # O sinal da medição. Sem argumento: quem monta o formulário é a janela, que tem os arquivos. O
    # painel emite e não fala com o transporte, o que mantém verde o portão de AST
    # `test_a_janela_nao_fala_com_o_transporte_direto`.
    medir_corretor = Signal()
    # A curva do LOESS de uma feature. Leva o nome da feature, porque é o painel que tem o campo;
    # quem monta o formulário (arquivos, lotes, semente) é a janela, como na medição.
    ver_curva_do_loess = Signal(str)
    # A etapa cujo campo sugerido a pessoa mudou à mão depois de 'Seguir sugestões do app'. A janela
    # apaga a frase do desfecho daquela página, que deixou de descrever a tela.
    sugestao_desfeita = Signal(int)

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        relogio: Relogio = agora,
    ) -> None:
        super().__init__(parent)
        # O depósito é o do módulo compartilhado, e é ele que carrega o congelamento entre leituras.
        # Um dicionário local aqui seria um segundo decorrido.
        self._deposito = DepositoEmMemoria()
        # O relógio entra por injeção: sem ele, afirmar que o decorrido da segunda corrida não é o
        # da primeira dependeria do relógio real.
        self._relogio = relogio
        self._identificador: str | None = None
        # O que o clique pediu, guardado no clique: `(delegou, corretor do seletor)`, ou `None`
        # antes de qualquer clique. No clique e não no aceite, porque entre os dois o profissional
        # pode mexer no seletor, e o aceite descreveria um pedido que não foi feito.
        self._pedido_do_disparo: tuple[bool, str] | None = None
        # O corretor que esta execução corre, sabido no aceite: o da política quando delegou, o
        # pedido quando não. `None` quer dizer que não se sabe, e a frase da etapa omite o nome em
        # vez de o inventar.
        self._corretor_em_curso: str | None = None
        # O par (feito, decorrido quando esse feito foi visto pela primeira vez) dá à etapa em curso
        # o seu próprio 'há quanto tempo'. Os dois números vêm de `decorrido_da_leitura`, e a
        # diferença entre duas leituras dele não é uma segunda fonte de tempo.
        self._etapa_vista: tuple[int, float | None] | None = None

        # ------------------------------------------------------------------
        # Os grupos nascem aqui, e quem os põe na tela é a página de cada etapa
        # ------------------------------------------------------------------
        #
        # Cada grupo nasce filho deste painel, e a composição da janela o põe na página da etapa
        # dele: 'Correção do sinal' na Correção, 'Pré-processamento e corte' na Normalização,
        # 'Estudo e contraste' e 'Análises a correr' na Análise estatística.

        # ------------------------------------------------------------------
        # O estudo e o contraste
        # ------------------------------------------------------------------
        #
        # Os campos usam `QFormLayout`, com o rótulo à esquerda do controle, em vez de `QLabel`
        # solto e widget na linha seguinte: poupa uma linha por campo, e o rótulo passa a ser do Qt.
        #
        # O filtro da roda é um por painel, com pai, e os campos roláveis passam por
        # `sem_roda_sem_foco`. Sem pai ele seria coletado e o Qt continuaria a apontar para o objeto
        # morto, sem nada acusar. Nasce antes do grupo do estudo porque o seletor do teste do
        # contraste, que vive lá, é o primeiro campo rolável do painel.
        self._filtro_da_roda = FiltroDaRoda(self)

        grupo_do_estudo = QGroupBox("Estudo e contraste", self)
        grupo_do_estudo.setProperty("painelPlano", True)
        self.grupo_do_estudo = grupo_do_estudo
        forma_do_estudo = QFormLayout(grupo_do_estudo)
        # `WrapLongRows` porque os rótulos deste painel são frases; sem isto o Qt reserva a largura
        # do rótulo mais longo para a coluna inteira dos rótulos.
        forma_do_estudo.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)

        self.estudo = QLineEdit(grupo_do_estudo)
        self.estudo.setPlaceholderText("Como este estudo se chama")
        forma_do_estudo.addRow("Identificação do estudo", self.estudo)

        # O par de grupos do PLS-DA, opcional. Sem par a execução corre igual e o PLS-DA fica 'não
        # calculado'; com par, o `Q²` e o `p` de permutação passam a existir. É opcional porque a
        # escolha do par é uma pergunta biológica: quatro grupos dão seis pares, cada um responde
        # outra coisa, e um padrão produziria uma tabela correta sobre a pergunta errada.
        #
        # Texto livre, e não combo, porque exigir o diagnóstico antes de executar acrescentaria um
        # passo obrigatório a quem só quer correr o pipeline. A rota recusa nome inexistente com
        # `422`, nomeando os disponíveis.
        #
        # O grupo de cada amostra vem da linha `LABEL` do `measurements`, a segunda linha do arquivo
        # (`read_measurements` lê `group_labels` de `rows[1]`), e não do `Sample_List`, que só traz
        # o ID e a ordem de injeção. A dica diz isso porque o campo aceita texto livre, e um nome
        # procurado no arquivo errado só apareceria como `422` depois de disparar. O `QC` não é
        # grupo: `grupos_contrastaveis` o exclui pela máscara, e a rota recusa quem o peça.
        self.rotulo_do_par = QLabel(
            "PLS-DA (opcional) — os dois grupos a comparar. Eles são lidos da linha `LABEL` do "
            "seu arquivo de medições (`measurements_*.csv`), que é a **segunda linha** dele; o "
            "`Sample_List` só traz a ordem de injeção. O programa **normaliza a grafia** — "
            "`HAM/TSP` e `HAM-TSP` viram `HAMTSP` —, então o nome a digitar pode não ser o que "
            "está escrito no arquivo. Se errar, a resposta lista os disponíveis. O `QC` não conta "
            "como grupo. Deixe os dois vazios e a execução corre igual, sem o Q² e o p de "
            "permutação.",
            self,
        )
        # Markdown porque a dica tem crases e negrito: o `AutoText` do Qt escolhe entre texto
        # simples e HTML, e sem esta linha os acentos graves apareceriam desenhados.
        self.rotulo_do_par.setTextFormat(Qt.TextFormat.MarkdownText)
        self.rotulo_do_par.setWordWrap(True)

        # Cada grupo na sua linha, com rótulo próprio: o texto de sugestão de dentro do campo some
        # assim que se escreve, e não diria qual é o A e qual é o B.
        self.grupo_a = QLineEdit(grupo_do_estudo)
        self.grupo_a.setPlaceholderText("Deixe vazio para não calcular o PLS-DA")
        self.grupo_b = QLineEdit(grupo_do_estudo)
        self.grupo_b.setPlaceholderText("Deixe vazio para não calcular o PLS-DA")
        forma_do_estudo.addRow("Grupo A do contraste", self.grupo_a)
        forma_do_estudo.addRow("Grupo B do contraste", self.grupo_b)

        # O teste do contraste, à mão ou automático, logo abaixo dos dois grupos, porque o volcano e
        # a contagem de FDR saem dele. 'Automático' é o padrão do domínio; 'Welch' e 'Mann-Whitney'
        # forçam. O texto desenhado é o nome legível e o valor que viaja é o do `Literal`
        # (`ComboComNomes`), com tabela de nomes própria porque `auto` já é um valor do
        # escalonamento. Nasce no padrão por `setCurrentText`, e não por índice, para o padrão não
        # depender da ordem do vocabulário.
        self.teste_univariado = sem_roda_sem_foco(
            ComboComNomes(NOMES_DO_TESTE_UNIVARIADO, grupo_do_estudo), self._filtro_da_roda
        )
        self.teste_univariado.addItems(list(get_args(UnivariateTest)))
        self.teste_univariado.setCurrentText(PADRAO_DA_ANALISE.univariate_test)
        self.teste_univariado.setToolTip(DICA_DO_TESTE_UNIVARIADO)
        forma_do_estudo.addRow("Teste do contraste", self.teste_univariado)

        # A linha do inventário diz quais grupos existem e com quantas amostras cada um. A contagem
        # diz se o contraste escolhido tem poder; sem ela, grupos de tamanhos diferentes parecem a
        # mesma escolha.
        #
        # O `n` fica aqui, e não no combo do multiverso: lá o texto do item é o nome que viaja no
        # disparo (`escolhas_do_disparo` envia `currentText()`), e um `"ATLL (13)"` seria recusado
        # pela rota com `422`.
        #
        # O rótulo nasce do método, e não de um `setText` com a constante, para o repouso acompanhar
        # a frase do método se ela mudar. Em repouso ele some, e o lugar dele é o primeiro
        # inventário que chegar.
        self.rotulo_do_inventario = RotuloQueSome(
            pai=grupo_do_estudo, vazios=frozenset({INVENTARIO_EM_REPOUSO})
        )
        self.rotulo_do_inventario.setWordWrap(True)
        # Markdown pela mesma razão de `rotulo_do_par`.
        self.rotulo_do_inventario.setTextFormat(Qt.TextFormat.MarkdownText)
        forma_do_estudo.addRow(self.rotulo_do_inventario)
        self.mostra_inventario_de_grupos(None)

        # A dica dos nomes dos grupos nasce recolhida: é longa, e aberta ocupava quatro linhas entre
        # o campo do estudo e os campos de grupo. `setVisible`, e não `QGroupBox.setCheckable`,
        # porque um `QGroupBox` marcável e desmarcado desabilita os filhos em vez de os esconder:
        # seria a mesma prosa, cinzenta, no mesmo espaço.
        self.botao_da_dica = QPushButton("De onde vêm os nomes dos grupos?", grupo_do_estudo)
        self.botao_da_dica.setProperty("divulgacao", True)
        self.botao_da_dica.setCheckable(True)
        veste_de_botao(self.botao_da_dica)
        self.botao_da_dica.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
        self._expansao_da_dica = ExpansaoSuave(self.rotulo_do_par)
        self.botao_da_dica.toggled.connect(self._expansao_da_dica.alterna)
        self.rotulo_do_par.setVisible(False)
        forma_do_estudo.addRow(self.botao_da_dica)
        forma_do_estudo.addRow(self.rotulo_do_par)

        # O grupo do pré-processamento vive na página da Normalização e é declarado aqui, onde nasce
        # o primeiro controle dele. Ele junta o que a matriz sofre depois de a deriva ser corrigida:
        # correção de sinal e normalização são etapas distintas, nessa ordem.
        grupo_do_preproc = QGroupBox("Pré-processamento e corte", self)
        grupo_do_preproc.setProperty("painelPlano", True)
        self.grupo_do_preproc = grupo_do_preproc
        forma_do_preproc = QFormLayout(grupo_do_preproc)
        forma_do_preproc.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)

        # O teto de features da exportação. O MetaboAnalyst recusa, ou filtra por IQR sozinho, acima
        # de 5.000 features, e o corte manual em planilha que motiva a ferramenta nasce daí; por
        # isso o teto é um controle visível, e não um valor fixo. O máximo é 100.000, e não os
        # 10.000 que o site do MetaboAnalyst declara, porque quem usa o pacote R não tem esse
        # limite. Quem exporta mais do que o destino aceita descobre-o no destino.
        self.max_features = sem_roda_sem_foco(QSpinBox(grupo_do_preproc), self._filtro_da_roda)
        self.max_features.setRange(1, 100_000)
        self.max_features.setValue(5_000)
        forma_do_preproc.addRow("Máximo de features na exportação", self.max_features)

        # O grupo da correção do sinal. O seletor fica sempre editável e o pedido leva sempre o
        # corretor dele; 'Seguir sugestões do app', no topo da Entrada, põe nele o corretor do
        # Palpite (`segue_a_proposta`). A janela não delega o corretor à política; quem delega é a
        # linha de comando, com `--auto`.
        grupo_da_correcao = QGroupBox("Correção do sinal", self)
        grupo_da_correcao.setProperty("painelPlano", True)
        self.grupo_da_correcao = grupo_da_correcao
        forma_da_correcao = QFormLayout(grupo_da_correcao)
        forma_da_correcao.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)

        # Nome legível por cima do valor: `currentText()` continua a ser `serrf`.
        self.corretor = sem_roda_sem_foco(
            ComboComNomes(NOMES_DOS_CORRETORES, grupo_da_correcao), self._filtro_da_roda
        )
        self.corretor.addItems(CORRETORES)
        forma_da_correcao.addRow("Corretor", self.corretor)

        # A caixa 'Usar vários núcleos', marcada por padrão e sem número de núcleos, por escolha do
        # usuário. O `serrf` e o `loess` a leem, e a dica diz isso; o `none` não calcula.
        # `tests/unit/test_caixa_dos_nucleos.py` deriva das rotas quem a caixa alcança.
        #
        # Desmarcá-la é sempre atendido; marcá-la é permissão e não ordem. Num processo em que o
        # paralelismo caiu para série no arranque, `n_jobs_efetivo` devolve série com a caixa
        # marcada, e por isso o palpite lê `_serial_agora()` e não o widget: a tela prometeria o
        # tempo paralelo sobre uma máquina que vai correr em série.
        self.paralelo = QCheckBox("Usar vários núcleos", grupo_da_correcao)
        self.paralelo.setChecked(True)
        self.paralelo.setToolTip(_DICA_DOS_NUCLEOS)
        # As duas estimativas dependem dela, e `mostra_palpite_da_medicao` reescreve as duas (guarda
        # `n_features` e chama a irmã). Um `connect` por palpite reescreveria a da execução duas
        # vezes.
        self.paralelo.toggled.connect(lambda _: self.mostra_palpite_da_medicao(self._n_features))
        forma_da_correcao.addRow("Processadores", self.paralelo)

        # A linha da medição. O lote dos três corretores é o 'Gerar os três' do Resultado da
        # Correção (`execucao.ResultadoDaEtapa`), no formato escolhido lá; deste grupo ele usa o
        # combo que percorre e a dica, `DICA_DO_LOTE`.
        linha_da_medicao = QHBoxLayout()
        linha_da_medicao.setSpacing(8)
        linha_da_medicao.addStretch(1)
        forma_da_correcao.addRow(linha_da_medicao)

        # A recomendação medida fica no ponto de escolha do corretor, e não ao lado do Palpite. O
        # Palpite é uma predição; esta é uma medição, que corre `none`, `loess` e `serrf` e os
        # compara pela régua fora da amostra. Juntas, seriam duas respostas com a mesma forma e
        # naturezas opostas. O rótulo diz 'medir', e não 'recomendar', porque é a palavra que separa
        # a medição da predição.
        self.medir_botao = QPushButton("Medir qual corretor vence (minutos)", grupo_da_correcao)
        self.medir_botao.clicked.connect(self.medir_corretor)
        self.medir_botao.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
        linha_da_medicao.insertWidget(0, self.medir_botao)

        # O intervalo por barra é opcional e vem desmarcado, por escolha do usuário: quem não o
        # marca não paga nada. Fica debaixo do botão que ele muda; o rótulo diz o custo pelos QCs do
        # estudo carregado, e a estimativa de tempo abaixo multiplica-se pelo mesmo fator quando
        # marcada.
        self.intervalo_por_barra = QCheckBox(rotulo_da_caixa_do_intervalo(None), grupo_da_correcao)
        self.intervalo_por_barra.setChecked(False)
        self.intervalo_por_barra.setToolTip(_DICA_DO_INTERVALO)
        self.intervalo_por_barra.toggled.connect(
            lambda _: self.mostra_palpite_da_medicao(self._n_features)
        )
        forma_da_correcao.addRow(self.intervalo_por_barra)

        # `MarkdownText` porque `prosa_da_medicao`, de `interface/fluxo.py`, escreve com crases e
        # negrito.
        # `RotuloQueSome`: vazio até a primeira medição, ele não ocupa linha.
        self.rotulo_da_medicao = RotuloQueSome(pai=grupo_da_correcao)
        self.rotulo_da_medicao.setWordWrap(True)
        self.rotulo_da_medicao.setTextFormat(Qt.TextFormat.MarkdownText)
        forma_da_correcao.addRow(self.rotulo_da_medicao)

        # O palpite de duração, num rótulo próprio: o do progresso é reescrito a cada leitura, e o
        # palpite precisa continuar legível enquanto a medição corre. Antes de validar não se sabe
        # quantas features a matriz tem; em repouso o rótulo some, e o número aparece quando a
        # 'Validação' volta.
        self.rotulo_do_palpite = RotuloQueSome(
            pai=grupo_da_correcao,
            vazios=frozenset({com_maiuscula(estimativa_da_medicao(None, serial=False))}),
        )
        self.rotulo_do_palpite.setWordWrap(True)
        self.rotulo_do_palpite.setTextFormat(Qt.TextFormat.MarkdownText)
        # O texto inicial usa `_serial_agora()`, a leitura ao vivo, e não `EM_REGIME_SERIAL`, o
        # retrato da importação, que divergiria se a ordem de arranque mudasse. `self.paralelo` já
        # existe acima, então não há dependência de ordem. A frase de `interface/fluxo.py` começa em
        # minúscula; aqui ela abre a linha.
        self.rotulo_do_palpite.setText(
            com_maiuscula(estimativa_da_medicao(None, serial=self._serial_agora()))
        )
        forma_da_correcao.addRow(self.rotulo_do_palpite)

        # A curva do LOESS de uma feature, no quadro da correção: quem quer ver a curva com que o
        # LOESS corrigiu uma feature, e quanto ela muda sem cada QC, está a decidir sobre a
        # correção. A figura aparece na aba 'Gráficos', para onde a janela vai quando a curva volta.
        #
        # Combo editável, e não lista fechada: são milhares de features, e a lista que a 'Validação'
        # devolve serve de completador. Um nome que a matriz não tenha volta com a recusa da rota,
        # no rótulo de baixo.
        self.feature_da_curva = sem_roda_sem_foco(
            QComboBox(grupo_da_correcao), self._filtro_da_roda
        )
        self.feature_da_curva.setEditable(True)
        self.feature_da_curva.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self.feature_da_curva.setMinimumContentsLength(18)
        self.ver_curva_botao = QPushButton("Ver a curva do LOESS", grupo_da_correcao)
        self.ver_curva_botao.clicked.connect(self._pede_a_curva)
        self.ver_curva_botao.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
        linha_da_curva = QHBoxLayout()
        linha_da_curva.setSpacing(8)
        linha_da_curva.addWidget(self.feature_da_curva, 1)
        linha_da_curva.addWidget(self.ver_curva_botao)
        forma_da_correcao.addRow("Curva do LOESS da feature", linha_da_curva)
        self.rotulo_da_curva = RotuloQueSome(pai=grupo_da_correcao)
        self.rotulo_da_curva.setWordWrap(True)
        forma_da_correcao.addRow(self.rotulo_da_curva)

        # O palpite da execução, num rótulo separado do da medição: são duas quantidades sobre
        # populações diferentes (um ajuste de um corretor contra `n_qc` reajustes dos três), e na
        # mesma célula uma seria lida pela outra.
        self.rotulo_do_palpite_da_execucao = RotuloQueSome(
            pai=grupo_da_correcao,
            vazios=frozenset({com_maiuscula(estimativa_da_execucao(None, "none", serial=False))}),
        )
        self.rotulo_do_palpite_da_execucao.setWordWrap(True)
        self.rotulo_do_palpite_da_execucao.setTextFormat(Qt.TextFormat.MarkdownText)
        forma_da_correcao.addRow(self.rotulo_do_palpite_da_execucao)

        # A ressalva do método, no ponto de escolha. `Caveat` tem `resumo` e `detalhe`; a janela
        # desenha só o `detalhe`, atrás do botão 'Ver o recorte medido', e o `resumo` chega ao
        # `--help`, ao `stderr`, ao `--log` e à API a partir de `cli.py`. O texto vem de lá e nunca
        # é reescrito aqui, porque uma segunda redação da ressalva divergiria da primeira. As
        # limitações gerais estão na aba 'Métodos e limites', em `desktop/recorte.py`; quem não abre
        # o botão nem a aba não lê a ressalva, que fica recolhida a pedido de quem usa.
        #
        # Markdown porque o `detalhe` traz crases, lista e linhas em branco, e o `AutoText` do Qt
        # mostraria as crases literais.
        self.rotulo_do_detalhe = QLabel(grupo_da_correcao)
        self.rotulo_do_detalhe.setWordWrap(True)
        self.rotulo_do_detalhe.setTextFormat(Qt.TextFormat.MarkdownText)

        # O detalhe da política do Palpite tem rótulo próprio, ao lado do do método: escrever uma
        # ressalva sobre a outra apagaria a do corretor escolhido.
        self.rotulo_do_detalhe_da_politica = QLabel(grupo_da_correcao)
        self.rotulo_do_detalhe_da_politica.setWordWrap(True)
        self.rotulo_do_detalhe_da_politica.setTextFormat(Qt.TextFormat.MarkdownText)

        # Os dois detalhes nascem recolhidos. `Caveat` já separa as duas metades ('`resumo` é o que
        # aparece no ponto de escolha; `detalhe` é o recorte inteiro'), e os dois recortes inteiros,
        # abertos, ocupam centenas de pixels de altura, mais quanto mais estreita for a janela.
        self.botao_do_detalhe = QPushButton("Ver o recorte medido", grupo_da_correcao)
        self.botao_do_detalhe.setProperty("divulgacao", True)
        self.botao_do_detalhe.setCheckable(True)
        veste_de_botao(self.botao_do_detalhe)
        self.botao_do_detalhe.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
        self._expansao_do_detalhe = ExpansaoSuave(self.rotulo_do_detalhe)
        self._expansao_da_politica = ExpansaoSuave(self.rotulo_do_detalhe_da_politica)
        self.botao_do_detalhe.toggled.connect(self._expansao_do_detalhe.alterna)
        self.botao_do_detalhe.toggled.connect(self._expansao_da_politica.alterna)
        self.rotulo_do_detalhe.setVisible(False)
        self.rotulo_do_detalhe_da_politica.setVisible(False)
        forma_da_correcao.addRow(self.botao_do_detalhe)
        forma_da_correcao.addRow(self.rotulo_do_detalhe)
        forma_da_correcao.addRow(self.rotulo_do_detalhe_da_politica)

        # Ligado ao combo para a ressalva seguir a escolha. Sem isso ela descreveria o método com
        # que o painel nasceu: um recorte medido sobre o corretor errado.
        self.corretor.currentTextChanged.connect(self.mostra_ressalva)
        self.mostra_ressalva(self.corretor.currentText())

        # `minimum=0` com `specialValueText` é como o Qt representa o vazio, e o 0 não viaja:
        # `escolhas_do_disparo` o traduz para `None`. O mínimo enviável é 1, porque `PipelineSpec`
        # recusa `n_batches < 1` com 422.
        self.n_lotes = sem_roda_sem_foco(QSpinBox(self), self._filtro_da_roda)
        self.n_lotes.setRange(0, 99)
        self.n_lotes.setSpecialValueText(SEM_LOTES)
        # O número de lotes é parâmetro de correção, e não filtro: a rota que só corrige
        # (`CAMPOS_POR_ARTEFATO["corrigida"]`) aceita o número de lotes e nenhum corte de filtro. O
        # lote é a estrutura dentro da qual o `batch_loess` e o SERRF corrigem a deriva; agrupá-lo
        # com os filtros ensinaria a coisa errada.
        forma_da_correcao.addRow("Quantos lotes declarar", self.n_lotes)
        # Quantos QCs cada lote declarado vai ter. A correção parte a corrida em blocos pela ordem
        # de injeção e corrige cada um com os QCs dele; a tela mostra a conta pela mesma partição
        # (`domain/lotes.py`), sem recusar nem avisar abaixo de limiar nenhum.
        self._amostras_dos_lotes: tuple[np.ndarray, np.ndarray] | None = None
        # `RotuloQueSome`, e o rótulo vazio da coluna da esquerda some com ele: os dois segurariam
        # uma linha em branco até a lógica escrever a conta.
        self.qcs_por_lote = RotuloQueSome(pai=self)
        self.qcs_por_lote.setWordWrap(True)
        forma_da_correcao.addRow("", self.qcs_por_lote)
        self.qcs_por_lote.acompanha(forma_da_correcao.labelForField(self.qcs_por_lote))

        # A variante do RSD, com itens derivados de `get_args(RsdVariant)`, pelo mesmo argumento de
        # `CORRETORES`. Vive no grupo da correção porque alimenta os resumos da etapa 1, o RSD dos
        # QCs antes e depois da correção (`ETAPA_DO_CAMPO` a põe na etapa 1); a máquina não a
        # propõe. O repouso é `plain`, 'Com todos os QCs', por escolha do usuário, e só nesta aba: a
        # linha de comando e a API não mudam.
        self.variante_do_rsd = sem_roda_sem_foco(
            ComboComNomes(NOMES_DAS_OPCOES, grupo_da_correcao), self._filtro_da_roda
        )
        self.variante_do_rsd.addItems(list(get_args(RsdVariant)))
        self.variante_do_rsd.setCurrentText(VARIANTE_DO_RSD_EM_REPOUSO)
        forma_da_correcao.addRow("Variante do RSD dos QCs", self.variante_do_rsd)

        # Repousa em vazio, e não em zero: o corte é fração em `(0, 1]`, e a API recusa `0.0`.
        self.corte = sem_roda_sem_foco(QDoubleSpinBox(self), self._filtro_da_roda)
        self.corte.setRange(0.0, 1.0)
        self.corte.setDecimals(3)
        self.corte.setSingleStep(0.01)
        self.corte.setSpecialValueText(SEM_CORTE)
        forma_do_preproc.addRow("Corte de RSD por fração", self.corte)

        # O corte por fração de ausentes, com o mesmo widget do corte de RSD. Vazio não aplica corte
        # nenhum: o valor entra em `_spec_ou_422` sem substituição, e `process_matrix` salta
        # `drop_by_group_missingness` como salta o corte de RSD. Não há um 'padrão do servidor'.
        self.ausentes = sem_roda_sem_foco(QDoubleSpinBox(self), self._filtro_da_roda)
        self.ausentes.setRange(0.0, 1.0)
        self.ausentes.setDecimals(3)
        self.ausentes.setSingleStep(0.01)
        self.ausentes.setSpecialValueText(SEM_CORTE_DE_AUSENTES)
        forma_do_preproc.addRow("Corte por fração de ausentes", self.ausentes)

        # As duas frações do filtro por posto, com o mesmo widget e a mesma tradução de vazio dos
        # campos acima, para 'não escolhi' significar uma coisa só.
        #
        # O rótulo diz 'retida' porque a convenção oposta é a intuitiva: quem lê 'fração' ao lado de
        # um filtro tende a pensar no que sai. A guarda `(0, 1]` não apanha a inversão, porque os
        # dois sentidos cabem nela.
        #
        # Construídas aqui e anexadas abaixo, logo depois do combo de cada passe: a ordem dos
        # controles é a ordem em que o pipeline os aplica, e o método decide se a etapa corre antes
        # de a fração dizer quanto ela retém.
        self.fracao_posto = sem_roda_sem_foco(QDoubleSpinBox(self), self._filtro_da_roda)
        self.fracao_posto_2 = sem_roda_sem_foco(QDoubleSpinBox(self), self._filtro_da_roda)
        fracoes_por_eixo = {
            "ranking_filter": ("Fração retida — 1º passe", self.fracao_posto),
            "ranking_filter_2": ("Fração retida — 2º passe", self.fracao_posto_2),
        }
        for _rotulo_da_fracao, _campo in fracoes_por_eixo.values():
            _campo.setRange(0.0, 1.0)
            _campo.setDecimals(3)
            _campo.setSingleStep(0.05)
            _campo.setSpecialValueText(FRACAO_POR_TAMANHO)

        # A referência da normalização, num combo editável. Editável porque a lista vem de
        # `POST /matrizes/validar`, e exigir a validação antes de executar acrescentaria um passo
        # obrigatório: quem validou escolhe da lista, quem não validou escreve. Combo, e não
        # `QLineEdit`, porque a leitura de valor deste painel levanta com outros tipos de widget, e
        # porque a proposta da máquina chega como `currentText`.
        #
        # A largura visível é limitada de propósito: os nomes de feature (`RT_mz`, como
        # `2.25_810.7739m/z`) são milhares, e o mais comprido fixaria a largura mínima do formulário
        # inteiro. O completamento por conteúdo torna a lista longa utilizável.
        self.referencia = sem_roda_sem_foco(QComboBox(self), self._filtro_da_roda)
        self.referencia.setEditable(True)
        self.referencia.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self.referencia.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self.referencia.setMinimumContentsLength(24)
        # O `lineEdit` só existe porque `setEditable(True)` correu acima; a guarda é para o mypy,
        # que não sabe disso, e não para um caso que aconteça.
        campo_de_texto = self.referencia.lineEdit()
        if campo_de_texto is not None:  # pragma: no branch - editável desde a linha acima
            campo_de_texto.setPlaceholderText(SEM_REFERENCIA)
        completador = self.referencia.completer()
        if completador is not None:
            completador.setFilterMode(Qt.MatchFlag.MatchContains)
            completador.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)

        # Os eixos da grade, um combo por eixo, com os itens do domínio e o padrão de
        # `EIXOS_DA_GRADE`. Um dicionário por chave de eixo, e não atributos nomeados, para
        # `campos_da_execucao` iterar em vez de enumerar: um eixo novo na constante ganha widget e
        # entra no payload sozinho.
        self.eixos: dict[str, QComboBox] = {}
        self.ressalvas_dos_eixos: dict[str, QLabel] = {}
        self.equivalencias_dos_eixos: dict[str, QLabel] = {}
        for chave, rotulo, padrao, opcoes in EIXOS_DA_GRADE:
            # Nome legível por cima do valor: o valor do item continua a ser o do domínio.
            combo = sem_roda_sem_foco(ComboComNomes(NOMES_DAS_OPCOES, self), self._filtro_da_roda)
            combo.addItems(opcoes)
            # `setCurrentText` e não um índice: o índice amarra o padrão à ordem do `Literal`, e um
            # rótulo novo antes do padrão mudaria em silêncio o que executa quem não mexe no
            # controle.
            combo.setCurrentText(padrao)
            self.eixos[chave] = combo
            forma_do_preproc.addRow(rotulo, combo)

            # O nome do método no MetaboAnalyst, colado ao combo, para quem já usa o MetaboAnalyst
            # entender de imediato para que serve. Todos os eixos o recebem, ao contrário da
            # ressalva: aqui todo valor tem frase (o rótulo deles ou o motivo de não haver), então
            # não há caso vazio. O texto vem de `prosa_da_equivalencia` e nunca é reescrito aqui,
            # porque uma segunda redação divergiria da primeira.
            rotulo_do_equivalente = QLabel(self)
            rotulo_do_equivalente.setWordWrap(True)
            # `MarkdownText` porque a frase traz crases à volta do rótulo deles e negrito na
            # divergência; o `AutoText` do Qt mostraria os asteriscos literais.
            rotulo_do_equivalente.setTextFormat(Qt.TextFormat.MarkdownText)
            forma_do_preproc.addRow(rotulo_do_equivalente)
            self.equivalencias_dos_eixos[chave] = rotulo_do_equivalente
            # A ligação faz a linha seguir a escolha. Sem ela o rótulo mostraria para sempre a
            # contraparte do valor inicial: nome de controle errado, que é pior que nome nenhum.
            combo.currentTextChanged.connect(
                lambda valor, c=chave: self.mostra_equivalente_do_eixo(c, valor)
            )
            self.mostra_equivalente_do_eixo(chave, combo.currentText())

            # A fração do passe logo abaixo do método dele, quando existe: o método decide se a
            # etapa corre, e a fração decide quanto ela retém.
            fracao = fracoes_por_eixo.get(chave)
            if fracao is not None:
                forma_do_preproc.addRow(fracao[0], fracao[1])

            # A referência logo abaixo da normalização por amostra, pela mesma regra de ordem.
            if chave == "sample_normalization":
                forma_do_preproc.addRow("Referência da normalização", self.referencia)
                # O lugar da importação dos fatores, logo abaixo da referência, a pedido do
                # laboratório. Quem a constrói é `PainelDeArquivos`, dono do estado do arquivo, e
                # quem a põe aqui é a janela; sem janela, o lugar fica vazio.
                self.lugar_dos_fatores = QWidget(self)
                lugar = QVBoxLayout(self.lugar_dos_fatores)
                lugar.setContentsMargins(0, 0, 0, 0)
                forma_do_preproc.addRow(self.lugar_dos_fatores)

            # A ressalva do eixo, no ponto de escolha, com a mesma forma da do corretor. Só os eixos
            # que têm ressalva ganham rótulo, porque um 'sem ressalva medida' sob os outros seria
            # ruído que esconde o que importa. O `if` é sobre a constante, e não sobre o nome do
            # eixo: um eixo novo com ressalva ganha o rótulo sozinho.
            if RESSALVAS_POR_EIXO.get(chave):
                rotulo_da_ressalva = QLabel(self)
                rotulo_da_ressalva.setWordWrap(True)
                rotulo_da_ressalva.setTextFormat(Qt.TextFormat.MarkdownText)
                forma_do_preproc.addRow(rotulo_da_ressalva)
                self.ressalvas_dos_eixos[chave] = rotulo_da_ressalva
                # A ligação faz a ressalva seguir a escolha; sem ela o rótulo mostraria para sempre
                # a do valor inicial.
                combo.currentTextChanged.connect(
                    lambda valor, c=chave: self.mostra_ressalva_do_eixo(c, valor)
                )
                self.mostra_ressalva_do_eixo(chave, combo.currentText())

        # Os valores iniciais desta página, que 'Seguir sugestões do app' repõe. São pontos de
        # partida, e não uma recomendação: só `correction` e `n_batches` têm proposta de rota
        # (`PROPOSTAS_POR_ROTA`), e a máquina não tem opinião sobre os eixos da grade, os cortes e
        # as frações. Os campos são os cortes de RSD e de ausentes, o teto de features (que também é
        # um corte), os eixos e as duas frações de posto. A referência da normalização fica de fora:
        # o repouso dela é vazio, e repor apagaria o nome que a pessoa escolheu da lista da
        # 'Validação'.
        #
        # Os valores são lidos agora, na construção, e não redigitados: uma tabela de padrões ao
        # lado seria outra fonte de verdade e divergiria do `setValue` de cada campo.
        self._campos_dos_valores_iniciais: tuple[QWidget, ...] = (
            self.corte,
            self.ausentes,
            self.max_features,
            *self.eixos.values(),
            self.fracao_posto,
            self.fracao_posto_2,
        )
        self._valores_iniciais = tuple(
            valor_do_campo(campo) for campo in self._campos_dos_valores_iniciais
        )

        # ------------------------------------------------------------------
        # Quais análises correm
        # ------------------------------------------------------------------
        #
        # As caixas derivam de `FAMILIAS_DE_ANALISE`: uma família nova entra na tabela e aparece
        # aqui, no pedido e na barra de progresso ao mesmo tempo. Escrevê-las à mão criaria uma
        # lista que diverge quando nascer uma família nova, e a tela deixaria de oferecer o que a
        # rota já calcula.
        grupo_das_analises = QGroupBox("Análises a correr", self)
        grupo_das_analises.setProperty("painelPlano", True)
        self.grupo_das_analises = grupo_das_analises
        pilha_das_analises = QVBoxLayout(grupo_das_analises)
        pilha_das_analises.setContentsMargins(0, 40, 0, 20)
        pilha_das_analises.setSpacing(16)

        # As caixas das famílias que são bandeira booleana, por `campo` de `AnalysisSpec`.
        self.analises: dict[str, QCheckBox] = {}
        # As caixas dos classificadores, por rótulo de `ClassifierKind`, separadas porque o campo é
        # uma tupla e não uma bandeira: `classifiers=("random_forest",)` e
        # `classifiers=("random_forest", "svm")` são pedidos diferentes, e uma caixa só colapsaria
        # os dois.
        self.classificadores: dict[str, QCheckBox] = {}

        caixas: list[QCheckBox] = []
        for familia in FAMILIAS_DE_ANALISE:
            # A dica diz o que a família faz e quanto custa: o custo explica por que umas vêm
            # marcadas e outras não, e 'segundos contra minutos' decide se vale esperar. Vive na
            # dica, e não no rótulo, porque rótulo é largura (ver `FamiliaDeAnalise.rotulo`).
            dica = f"{DICA_DA_FAMILIA[familia.campo]}. " + (
                "Custa SEGUNDOS — vem ligada."
                if familia_e_barata(familia)
                else "Custa MINUTOS — vem desligada; tique se quiser correr."
            )
            if familia.campo == CAMPO_DOS_CLASSIFICADORES:
                for tipo in get_args(ClassifierKind):
                    caixa = QCheckBox(f"{familia.rotulo}: {tipo}", grupo_das_analises)
                    # O padrão vem de `PADRAO_DA_ANALISE`, onde o domínio decide o que nasce
                    # ligado.
                    caixa.setChecked(tipo in PADRAO_DA_ANALISE.classifiers)
                    caixa.setToolTip(dica)
                    self.classificadores[tipo] = caixa
                    caixas.append(caixa)
                continue
            caixa = QCheckBox(familia.rotulo, grupo_das_analises)
            caixa.setChecked(familia_e_barata(familia))
            caixa.setToolTip(dica)
            self.analises[familia.campo] = caixa
            caixas.append(caixa)

        # Duas colunas: são rótulos de uma a quatro palavras, e numa coluna só o bloco teria o dobro
        # das linhas.
        grade_das_analises = QGridLayout()
        grade_das_analises.setHorizontalSpacing(32)
        grade_das_analises.setVerticalSpacing(8)
        em_grade(grade_das_analises, list(caixas), 2)
        pilha_das_analises.addLayout(grade_das_analises)

        # O rótulo diz o que a barra ganha; o palpite diz quanto demora. São dois fatos em dois
        # rótulos, e o mesmo texto nos dois seria ruído.
        self.rotulo_das_analises = QLabel(grupo_das_analises)
        self.rotulo_das_analises.setWordWrap(True)
        pilha_das_analises.addWidget(self.rotulo_das_analises)

        # O grupo das análises vive na página da Análise estatística, logo abaixo do grupo do estudo
        # e do contraste.

        # Os grupos de cada etapa, na ordem em que a página os mostra. A composição da janela os
        # tira daqui, e `etapas_dos_eixos` sabe por aqui em que página mora o controle de cada eixo:
        # uma fonte só para 'qual grupo é de qual etapa'.
        self.grupos_das_etapas: dict[int, tuple[QGroupBox, ...]] = {
            1: (grupo_da_correcao,),
            2: (grupo_do_preproc,),
            3: (grupo_do_estudo, grupo_das_analises),
        }

        # ------------------------------------------------------------------
        # A ação de cada etapa: o botão, o estado, a barra e a nota
        # ------------------------------------------------------------------
        #
        # Tudo nasce aqui e escondido, e quem o põe à vista é a página da etapa, na linha da ação,
        # pelas mãos da composição da janela (`AbaDeEtapa`). Escondido, um widget montado sem a
        # página não aparece solto no canto. A barra da execução é a da Análise, que é quem dispara
        # `/execucoes`, e as linhas de estado da execução vão ao rodapé dela.
        self.botoes_de_etapa: dict[int, QPushButton] = {}
        self.rotulos_de_etapa: dict[int, QLabel] = {}
        self.barras_de_etapa: dict[int, QProgressBar] = {}
        # A nota de cada etapa: o progresso em texto, a espera pelo Palpite, a desistência, o que a
        # barra e o estado não dizem. Ela sobrevive ao relógio da janela, que reescreve o estado a
        # cada meio segundo, e só muda quando outra frase a substitui. A falha e a recusa vão ao
        # aviso vermelho da aba, e não aqui, para a mesma frase não aparecer duas vezes.
        self.notas_de_etapa: dict[int, QLabel] = {}
        for etapa, rotulo in ROTULOS_DAS_ETAPAS.items():
            botao = QPushButton(rotulo, self)
            botao.clicked.connect(lambda _=False, e=etapa: self.disparar_etapa.emit(e))
            # Nasce ligado: uma etapa só espera que nenhuma outra esteja em curso, e ao nascer
            # nenhuma está. O texto vem do estado inicial das etapas, escrito mais abaixo.
            estado = QLabel(self)
            estado.setWordWrap(True)
            estado.setProperty("textoAuxiliar", True)
            barra = QProgressBar(self)
            barra.setFixedHeight(20)
            # Em repouso a barra fica vazia, e não indeterminada: `setRange(0, 0)` desenha a listra
            # a correr sem fim e faz a tela parecer a trabalhar quando não está. `0..1` com valor
            # `0` não afirma percentagem nem trabalho; o indeterminado fica para 'está a correr e
            # não sei quanto falta'.
            barra.setRange(0, 1)
            barra.setValue(0)
            for parte in (botao, estado, barra):
                parte.hide()
            self.botoes_de_etapa[etapa] = botao
            self.rotulos_de_etapa[etapa] = estado
            self.barras_de_etapa[etapa] = barra
            if etapa != ETAPAS[-1]:
                # Some enquanto vazia (`RotuloQueSome`): sem nota, o rodapé não ganha altura.
                nota = RotuloQueSome(pai=self)
                # Markdown pela razão de `rotulo_da_medicao`: o progresso tem negrito.
                nota.setTextFormat(Qt.TextFormat.MarkdownText)
                self.notas_de_etapa[etapa] = nota

        # As três linhas da execução, que vão ao rodapé da Análise. A Análise é o `/execucoes`, e é
        # dela o que a leitura do trabalho diz: o progresso com o decorrido e a etapa em curso, o
        # corretor e quem o escolheu, e o resultado reclamado com o registro gravado. Somem em
        # repouso. A linha de progresso é a nota da Análise, porque uma segunda linha de progresso
        # no mesmo rodapé repetiria a primeira.
        self.rotulo_de_progresso = RotuloQueSome(
            PROGRESSO_EM_REPOUSO, self, vazios=frozenset({PROGRESSO_EM_REPOUSO})
        )
        self.notas_de_etapa[ETAPAS[-1]] = self.rotulo_de_progresso
        # O corretor desta execução fica na linha da execução, e não no grupo da correção: a
        # ressalva da política aparece no resultado, onde a escolha deixou de ser hipótese, e não ao
        # lado do seletor, de onde saiu a pedido de quem usa.
        self.rotulo_do_corretor = RotuloQueSome(
            CORRETOR_EM_REPOUSO, self, vazios=frozenset({CORRETOR_EM_REPOUSO})
        )
        self.rotulo_de_resultado = RotuloQueSome(
            RESULTADO_EM_REPOUSO, self, vazios=frozenset({RESULTADO_EM_REPOUSO})
        )
        for linha_do_rodape in (
            *self.notas_de_etapa.values(),
            self.rotulo_do_corretor,
            self.rotulo_de_resultado,
        ):
            linha_do_rodape.setWordWrap(True)
            linha_do_rodape.setProperty("textoAuxiliar", True)

        for forma in (forma_do_estudo, forma_da_correcao, forma_do_preproc):
            _compacta_formulario(forma)
        for botao in self.botoes_de_etapa.values():
            botao.setFixedHeight(32)
            botao.setProperty("acaoDaEtapa", True)
        self.medir_botao.setFixedHeight(32)
        for texto_auxiliar in (
            self.rotulo_do_par,
            self.rotulo_do_inventario,
            self.rotulo_da_medicao,
            self.rotulo_do_palpite,
            self.rotulo_do_palpite_da_execucao,
            self.rotulo_do_detalhe,
            self.rotulo_do_detalhe_da_politica,
            self.rotulo_das_analises,
            *self.equivalencias_dos_eixos.values(),
            *self.ressalvas_dos_eixos.values(),
        ):
            texto_auxiliar.setProperty("textoAuxiliar", True)
        # O estado que `mostra_etapas` compara, para mexer na barra só quando ele muda.
        self._estados_das_etapas: tuple[str, ...] = ("nao_feita",) * 3
        # O rótulo de cada etapa nasce com o estado inicial, da mesma tupla que `mostra_etapas`
        # compara: uma fonte só para 'não feita'.
        for etapa, estado_inicial in zip(ETAPAS, self._estados_das_etapas, strict=True):
            self.rotulos_de_etapa[etapa].setText(TEXTO_DO_ESTADO_DA_ETAPA[estado_inicial])

        # A proposta da máquina e o registro de quem mexeu. `_proposta` guarda o que cada rota
        # propôs e `_tocados` os eixos em que o humano mexeu: divergência entre o valor corrente e a
        # proposta é a sobrescrita, e `_tocados` impede que uma leitura que chegue depois da escolha
        # a apague.
        #
        # A ligação é feita depois de os widgets serem preenchidos: antes, os `setCurrentText` do
        # próprio construtor marcariam todos os eixos como tocados, e o pré-preenchimento nunca
        # aconteceria, sem nada levantar.
        # Quantas features a matriz tem, quando a validação já voltou. `None` é 'ainda não sei', que
        # o palpite precisa distinguir de zero.
        self._n_features: int | None = None
        self._proposta: dict[str, Any] = {}
        # Os cortes que o diagnóstico propôs. Só valem com a proposta numérica de `_proposta`, e
        # ficam ao lado e não dentro dela: um eixo de sobrescrita é um controle da tela, e os cortes
        # não têm controle; vão ou não vão com o número.
        self._cortes_propostos: tuple[int, ...] | None = None
        # O número de lotes que o 'Diagnóstico' propôs, guardado e não escrito: o campo fica em 1
        # (`guarda_a_sugestao_de_lotes`). Ele só passa a `_proposta`, o lado da máquina do par e a
        # condição dos cortes, quando a pessoa o adota por 'Seguir sugestões do app'
        # (`segue_a_proposta`).
        self._lotes_sugeridos: int | None = None
        # O corretor que o Palpite sugeriu, pelo mesmo caminho: a aba fica nos valores padrão até a
        # pessoa os mudar ou seguir as sugestões.
        self._corretor_sugerido: str | None = None
        self._tocados: set[str] = set()
        # A guarda de reentrância. `_escreve_valor` chama `setCurrentText`, que emite
        # `currentTextChanged`, que marcaria o eixo como tocado pelo humano. Uma segunda leitura da
        # máquina, gesto legítimo porque os botões da aba de preparo podem ser clicados outra vez,
        # gravaria então `campos_sobrescritos=["correction"]` para quem não mexeu em nada, e toda
        # execução ficaria registrada como manual. A guarda cobre toda escrita da máquina, e não só
        # a do construtor.
        self._escrevendo = False
        for eixo, widget in self._controles_de_valor().items():
            sinal = (
                widget.currentTextChanged if isinstance(widget, QComboBox) else widget.valueChanged
            )
            sinal.connect(lambda _valor=None, e=eixo: self._marca_tocado(e))

        # O desfazer das sugestões. Mudar à mão um campo que 'Seguir sugestões do app' preencheu faz
        # a frase 'Sugestões aplicadas' deixar de descrever a tela, e a janela a apaga pelo sinal.
        # As escritas da máquina passam por `_escrevendo` e não contam, senão a própria aplicação
        # apagaria a frase que vem escrever.
        self._campos_das_sugestoes: dict[int, tuple[QWidget, ...]] = {
            ETAPAS[0]: (self.corretor, self.n_lotes),
            2: self._campos_dos_valores_iniciais,
        }
        for etapa_sugerida, campos in self._campos_das_sugestoes.items():
            for campo in campos:
                mudou = (
                    campo.currentTextChanged if isinstance(campo, QComboBox) else campo.valueChanged  # type: ignore[attr-defined]
                )
                mudou.connect(lambda _valor=None, e=etapa_sugerida: self._desfaz_a_sugestao(e))

        # A ressalva do Palpite atrás de 'Ver o recorte medido', ao lado da do corretor: é a da
        # proposta que o botão das sugestões aplica.
        self.rotulo_do_detalhe_da_politica.setText(CAVEAT_DO_PALPITE.detalhe)

        # O palpite da execução depende de quantas features a matriz tem, que vem da validação, e de
        # qual corretor vai correr, que é este combo. Ligado ao sinal e chamado uma vez na
        # construção, porque o combo nasce com um valor e ninguém emite sinal por isso.
        self.corretor.currentTextChanged.connect(lambda _=None: self.mostra_palpite_da_execucao())
        # E segue o número de lotes: no `serrf` o custo quase dobra com dois lotes, e sem o eixo o
        # palpite prometeria bem menos do que a execução custa.
        self.n_lotes.valueChanged.connect(lambda _=None: self.mostra_palpite_da_execucao())
        self.n_lotes.valueChanged.connect(lambda _=None: self._mostra_qcs_por_lote())
        # E segue as caixas das análises, o terceiro eixo do palpite: marcar uma família cara muda o
        # que a tela promete. Ligadas e chamadas uma vez na construção, porque as caixas nascem com
        # um valor e ninguém emite sinal por isso.
        for caixa in (*self.analises.values(), *self.classificadores.values()):
            caixa.toggled.connect(lambda _=None: self._mostra_o_que_as_analises_custam())
        self._mostra_o_que_as_analises_custam()

    # ------------------------------------------------------------------
    # O payload, e o par de concordância entre a proposta da máquina e a escolha da pessoa
    # ------------------------------------------------------------------

    def mostra_espera(self, chave: str, *, feito: int, total: int, leituras: int) -> str:
        """O sinal de vida da medição, com a mesma assinatura da irmã de `artefatos.py`.

        A assinatura é igual para `JanelaPrincipal` reusar o laço de espera dos artefatos e
        escolher o mostrador por chave. É nesse laço que vivem o teto de leituras, o descarte de
        cadeia substituída e o relógio, e assinaturas diferentes obrigariam a duplicá-lo.

        Fora das prévias das etapas, `chave` não é usada: o painel mostra uma medição de cada vez.
        O progresso conta corretores, porque a rota declara uma etapa por corretor, pelo gancho
        `depois_de_cada` do oráculo.
        """
        etapa = ETAPA_DA_CHAVE.get(chave)
        if etapa is not None:
            # As prévias do passo a passo escrevem na nota da etapa, abaixo da ação da página
            # dela, e não na linha da medição.
            texto = (
                f"**{feito} de {total}** passos. {leituras}.ª consulta de progresso — a etapa pode "
                "levar minutos, e a janela continua utilizável."
            )
            self.anota_etapa(etapa, texto)
            # A barra da etapa fica no rodapé da página dela; sem `total`, segue indeterminada.
            if total > 0:
                barra = self.barras_de_etapa[etapa]
                barra.setRange(0, total)
                barra.setValue(min(feito, total))
            return texto
        del chave
        # A contagem é de consultas desta janela, e a frase diz isso. Clicar outra vez depois de a
        # janela desistir recomeça o contador local (`_gera_artefato` encerra a espera anterior),
        # mas o servidor deduplica por `(operação, matriz, campos)` e devolve o mesmo trabalho.
        # O progresso que não recomeça é o de corretores ajustados.
        texto = (
            f"Medindo... **{feito} de {total}** corretores ajustados. "
            f"{leituras}.ª consulta de progresso desta janela — o trabalho corre no servidor e "
            f"não recomeça se você consultar outra vez. Cada corretor custa minutos; a janela "
            f"continua utilizável."
        )
        self.rotulo_da_medicao.setText(texto)
        return texto

    def mostra_desistencia(self, chave: str, *, leituras: int, segundos: float) -> str:
        """O teto de leituras passou. Irmã de `artefatos.mostra_desistencia`, e diz o que fazer.

        Desistir de esperar não cancela: o trabalho continua no servidor, e clicar outra vez
        reclama o resultado em vez de recalcular, porque `_despacha` deduplica por
        `(operação, matriz, campos)`.
        """
        etapa = ETAPA_DA_CHAVE.get(chave)
        if etapa is not None:
            texto = (
                f"Parei de esperar a etapa depois de **{leituras}** leituras e "
                f"**{segundos:.0f} s**. Ela continua a correr — clicar outra vez reclama o "
                "resultado em vez de recomeçar."
            )
            self.anota_etapa(etapa, texto)
            return texto
        del chave
        texto = (
            f"parei de esperar depois de **{leituras}** leituras e **{segundos:.0f} s**. "
            "O trabalho continua a correr — clicar outra vez reclama o resultado em vez de "
            "recomeçar."
        )
        self.rotulo_da_medicao.setText(texto)
        return texto

    def mostra_inventario_de_grupos(self, inventario: Sequence[GrupoNoInventario] | None) -> str:
        """Escreve quais grupos esta matriz tem e com quantas amostras. Devolve o texto.

        É chamado quando a leitura volta, porque é `POST /matrizes/diagnosticar` que traz o campo;
        o painel não conhece rota nenhuma. Há três desfechos:
        - `None`: a leitura não correu, ou a resposta não traz o campo; a linha pede a leitura;
        - `()`: correu e não há grupo fora do QC; a linha manda conferir a linha `LABEL`;
        - com entradas: os grupos, com o `n` de cada um.
        Juntar os dois primeiros faria a tela dizer 'nenhum grupo' a quem ainda não leu o arquivo.

        O total vai ao lado porque o QC fica fora do inventário (é régua, não grupo), e quem
        somasse os grupos contra as linhas do arquivo suspeitaria de perda. A frase é curta porque
        a coluna esquerda já rola e o rótulo tem pouca altura.
        """
        if inventario is None:
            self.rotulo_do_inventario.setText(INVENTARIO_EM_REPOUSO)
            return INVENTARIO_EM_REPOUSO
        if not inventario:
            self.rotulo_do_inventario.setText(INVENTARIO_SEM_GRUPO)
            return INVENTARIO_SEM_GRUPO
        # A ordem é a da rota: `inventario_de_grupos` ordena por nome, e `_grupos_ou_422` nomeia os
        # disponíveis na mesma ordem. Uma terceira ordem faria quem lesse as duas supor conjuntos
        # diferentes.
        corpo = ", ".join(f"**{grupo.nome}** ({grupo.n})" for grupo in inventario)
        total = sum(grupo.n for grupo in inventario)
        texto = (
            f"Grupos desta matriz: {corpo} — {total} amostras, com o QC de fora. "
            "O `n` é o que diz se o contraste tem amostras que cheguem."
        )
        self.rotulo_do_inventario.setText(texto)
        return texto

    def mostra_palpite_da_medicao(self, n_features: int | None) -> str:
        """O palpite de duração, a partir do tamanho da entrada. Devolve o que escreveu.

        É chamado quando a validação volta, porque é ela que traz `n_features`; o painel não
        conhece rota nenhuma. O custo por feature vive em `interface/fluxo.CUSTO_POR_FEATURE_MS`,
        com o
        motivo de ele extrapolar, e repeti-lo aqui seria uma segunda fonte da mesma quantidade.
        """
        texto = com_maiuscula(
            estimativa_da_medicao(
                n_features,
                serial=self._serial_agora(),
                # Marcada, a caixa do intervalo por barra multiplica a estimativa pelo fator que o
                # rótulo dela diz, calculado com os QCs da validação.
                intervalo=self.intervalo_por_barra.isChecked(),
                n_qc=self._n_qc_da_validacao(),
            )
        )
        self.rotulo_do_palpite.setText(texto)
        # A validação é a única leitura que traz `n_features`; guardá-lo deixa o palpite da
        # execução seguir o combo do corretor sem nova ida à rota.
        self._n_features = n_features
        self.mostra_palpite_da_execucao()
        return texto

    def mostra_etapas(
        self, estados: Sequence[str], motivos: Sequence[str | None]
    ) -> tuple[str, ...]:
        """Escreve o estado das três etapas e liga ou desliga os botões que a janela decidiu.
        Devolve o estado em palavras de cada uma, como o rodapé dela o mostra.

        `motivos` traz, por etapa, a frase de por que o botão fica desligado, ou `None` para
        ligá-lo. Não há como desligar sem explicar: a frase vai ao lado do botão, por
        `desliga_com_motivo`.

        O estado de cada etapa vai ao rótulo e à barra dela, no rodapé da página. A barra só muda
        quando o estado muda: a janela chama isto a cada meio segundo, e o relógio não pode apagar
        a leitura de progresso que a moveu a meio de uma etapa.
        """
        anteriores = self._estados_das_etapas
        textos: list[str] = []
        for etapa, estado, motivo, anterior in zip(
            ETAPAS, estados, motivos, anteriores, strict=True
        ):
            botao = self.botoes_de_etapa[etapa]
            if motivo is None:
                religa(botao)
            else:
                desliga_com_motivo(botao, motivo)
            rotulo = self.rotulos_de_etapa[etapa]
            texto = TEXTO_DO_ESTADO_DA_ETAPA[estado]
            # Só reescreve o que mudou: o relógio da janela chama isto a cada meio segundo, e um
            # `setText` igual faria o Qt refazer o desenho do rótulo à toa.
            if rotulo.text() != texto:
                rotulo.setText(texto)
            if estado != anterior:
                self._poe_a_barra_no_estado(etapa, estado)
            textos.append(texto)
        self._estados_das_etapas = tuple(estados)
        return tuple(textos)

    def _poe_a_barra_no_estado(self, etapa: int, estado: str) -> None:
        """Põe a barra da etapa no estado que ela acabou de tomar.

        Em curso, a barra fica indeterminada até a primeira leitura de progresso. Em dia, cheia.
        Não feita ou desatualizada, vazia: a barra mede a etapa como a tela a pede agora, e um
        resultado de outros parâmetros não conta como progresso dela.
        """
        barra = self.barras_de_etapa[etapa]
        if estado == "em_curso":
            barra.setRange(0, 0)
            return
        barra.setRange(0, 1)
        barra.setValue(1 if estado == "em_dia" else 0)

    def anota_etapa(self, etapa: int, nota: str) -> str:
        """Escreve a nota da `etapa` no rodapé da página dela. Vazia, some.

        A nota leva o que o estado em palavras e a barra não dizem: o progresso em texto, a espera
        pelo Palpite, a desistência. Há uma por etapa, perto do botão que a pessoa clicou. Na
        Análise, ela é a linha de progresso da execução, para não haver duas linhas no mesmo rodapé
        a dizer a mesma coisa.
        """
        rotulo = self.notas_de_etapa[etapa]
        if rotulo.text() != nota:
            rotulo.setText(nota)
        return nota

    def confere_a_combinacao(self) -> None:
        """Recusa na janela a combinação proibida de imputação e normalização; para a etapa 2."""
        self._recusa_a_combinacao_proibida()

    def define_amostras(self, ordens: object, qc: object) -> None:
        """Guarda a ordem de injeção e a marca de QC de cada amostra, vindas da validação.

        Os dados atravessam a fronteira e são conferidos aqui: duas listas do mesmo tamanho, uma
        de inteiros e outra de booleanos. O que não confere vira 'sem dado', e a linha fica vazia
        em vez de mostrar uma conta sobre a lista errada.
        """
        if (
            isinstance(ordens, list)
            and isinstance(qc, list)
            and ordens
            and len(ordens) == len(qc)
            and all(isinstance(o, int) and not isinstance(o, bool) for o in ordens)
            and all(isinstance(q, bool) for q in qc)
        ):
            self._amostras_dos_lotes = (np.array(ordens, dtype=np.int64), np.array(qc, dtype=bool))
        else:
            self._amostras_dos_lotes = None
        self._mostra_qcs_por_lote()
        # O custo do intervalo por barra depende dos QCs desta validação.
        self.intervalo_por_barra.setText(rotulo_da_caixa_do_intervalo(self._n_qc_da_validacao()))
        self.mostra_palpite_da_medicao(self._n_features)

    def _n_qc_da_validacao(self) -> int | None:
        """Quantos QCs a validação contou, ou `None` sem validação que confira."""
        if self._amostras_dos_lotes is None:
            return None
        return int(self._amostras_dos_lotes[1].sum())

    def _mostra_qcs_por_lote(self) -> None:
        """Escreve a linha debaixo do número de lotes. Vazia sem validação, ou com um lote só.

        A conta é sobre a partição que a correção vai usar, que pode ser a das fronteiras
        propostas: decide `cortes_em_uso`, a mesma pergunta que `campos_base` faz antes de enviar.
        Um corte que não caiba nas ordens desta validação (cortes de outro par) volta à conta dos
        blocos iguais em vez de derrubar a tela.
        """
        n = self.n_lotes.value()
        if self._amostras_dos_lotes is None or n < 2:
            self.qcs_por_lote.setText("")
            return
        ordens, qc = self._amostras_dos_lotes
        cortes = self.cortes_em_uso(n)
        try:
            contagens = qcs_por_lote_declarado(ordens, qc, n, cortes)
        except ValueError:
            contagens, cortes = qcs_por_lote_declarado(ordens, qc, n), None
        self.qcs_por_lote.setText(
            frase_dos_qcs_por_lote(contagens, nos_cortes_propostos=cortes is not None)
        )

    def define_cortes_propostos(self, cortes: object) -> None:
        """Guarda os cortes que o diagnóstico propôs junto com o número de lotes.

        Conferidos na fronteira: uma lista de inteiros, ou nada. Qualquer outra forma vira 'sem
        cortes', e a correção cai nos blocos iguais, em vez de num terceiro comportamento.
        """
        if isinstance(cortes, list) and all(
            isinstance(c, int) and not isinstance(c, bool) for c in cortes
        ):
            self._cortes_propostos = tuple(cortes)
        else:
            self._cortes_propostos = None
        self._mostra_qcs_por_lote()

    def esquece_os_cortes_propostos(self) -> None:
        """Um par novo invalida os cortes do diagnóstico anterior: são ordens de outra corrida."""
        self._cortes_propostos = None
        self._mostra_qcs_por_lote()

    def particao_sugerida(self) -> tuple[int, tuple[int, ...] | None]:
        """A partição em que o Palpite é medido: a que o 'Diagnóstico' sugeriu, ou um lote.

        O Palpite não depende da aba de correção: mexer nela enquanto ele roda não o para nem muda
        o resultado. Por isso ele lê o que 'Preparar o estudo' sugeriu, que é também o que 'Seguir
        sugestões do app' põe nos campos; assim o corretor sugerido é medido na partição em que a
        correção sugerida vai correr.

        Sem sugestão, um lote, que é o repouso do campo. Os cortes só vão com o número que os
        propôs, e só quando a contagem casa, como em `cortes_em_uso`.
        """
        n_lotes = self._lotes_sugeridos or LOTES_QUANDO_VAZIO
        cortes = self._cortes_propostos
        if self._lotes_sugeridos is None or cortes is None or len(cortes) != n_lotes - 1:
            cortes = None
        return n_lotes, cortes

    def esquece_o_toque(self, eixo: str) -> None:
        """Esquece que a pessoa mexeu no `eixo`.

        O clique em 'Seguir sugestões do app' é a pessoa pedindo as sugestões: um toque de antes
        do clique não conta contra elas. Um toque de depois, durante a espera do Palpite, conta
        (`foi_tocado`), e o que ela escolheu no intervalo fica.
        """
        self._tocados.discard(eixo)

    def foi_tocado(self, eixo: str) -> bool:
        """Se a pessoa mexeu no `eixo` desde a última vez que o toque foi esquecido."""
        return eixo in self._tocados

    @property
    def corretor_sugerido(self) -> str | None:
        """O corretor que o Palpite sugeriu e que pode ainda não estar no seletor."""
        return self._corretor_sugerido

    @property
    def lotes_sugeridos(self) -> int | None:
        """O número de lotes que o 'Diagnóstico' deixou guardado, ainda fora do campo.

        É o que 'Seguir sugestões do app' põe no campo (`segue_a_proposta`) e o que a sugestão ao
        lado do botão mostra. Só leitura: quem o guarda é `guarda_a_sugestao_de_lotes`, que confere
        a forma na fronteira, e quem o esquece é `esquece_as_propostas_do_par`.
        """
        return self._lotes_sugeridos

    def guarda_a_sugestao_do_corretor(self, corretor: object) -> bool:
        """Guarda o corretor que o Palpite sugeriu sem o escrever no seletor. Devolve se guardou.

        A página de correção fica nos valores padrão até a pessoa preenchê-la ou pedir as
        sugestões do app. Por isso o corretor espera por 'Seguir sugestões do app', como o número
        de lotes, e fica fora de `_proposta` até ser adotado: assim o par de decisão não grava como
        trocada uma proposta que ninguém aceitou. Uma sugestão diferente desfaz a adoção da
        anterior.

        Na fronteira, só um corretor que o seletor oferece é guardado, em vez de cair num valor
        qualquer.
        """
        oferecidos = {self.corretor.itemText(i) for i in range(self.corretor.count())}
        if not isinstance(corretor, str) or corretor not in oferecidos:
            return False
        if self._proposta.get(EIXO_DO_CORRETOR) not in (None, corretor):
            self._proposta.pop(EIXO_DO_CORRETOR)
        self._corretor_sugerido = corretor
        return True

    def guarda_a_sugestao_de_lotes(self, n_lotes: object) -> bool:
        """Guarda o número de lotes do 'Diagnóstico' sem o escrever no campo. Devolve se guardou.

        O campo abre sempre em 1, e a sugestão espera por 'Seguir sugestões do app', que a adota.
        Até lá ela fica fora de `_proposta`, porque `eixos_divergentes` compara o campo com
        `_proposta` sem perguntar quem mexeu: com a proposta registrada e o campo em 1, toda
        execução sairia gravada como 'a proposta da máquina foi trocada'. Os cortes continuam
        presos à proposta adotada, por `cortes_em_uso`.

        Uma sugestão nova e diferente desfaz a adoção da anterior, porque o número no campo veio de
        outro 'Diagnóstico'; a mesma sugestão de novo não desfaz nada. Na fronteira, só um inteiro
        de 1 para cima é guardado.
        """
        if not isinstance(n_lotes, int) or isinstance(n_lotes, bool) or n_lotes < 1:
            return False
        if self._proposta.get(EIXO_DOS_LOTES) not in (None, n_lotes):
            self._proposta.pop(EIXO_DOS_LOTES)
        self._lotes_sugeridos = n_lotes
        self._mostra_qcs_por_lote()
        return True

    def registra_a_sugestao(self, eixo: str) -> bool:
        """Registra a sugestão guardada do `eixo` como o lado da máquina do par, sem a escrever no
        campo. Devolve se registrou.

        Trocar um campo durante a espera, depois de clicar 'Seguir sugestões do app', conta como
        sobrescrever a proposta da máquina. O campo fica com o que a pessoa escolheu, e a sugestão
        que chegou passa a `_proposta`: é isso que faz `eixos_divergentes` ver o eixo e a rota
        gravar `sobrescreveu_discordando`, em vez de `nao_sobrescreveu` com a escolha da pessoa na
        linha da máquina.

        Só vale para quem pediu as sugestões, e quem chama sabe se houve pedido: guardada e não
        pedida, a sugestão não é uma proposta que a pessoa trocou, e registrá-la faria toda
        Correção sem o clique sair como trocada. Sem sugestão guardada, nada é registrado.
        """
        sugerida: Any = None
        if eixo == EIXO_DO_CORRETOR:
            sugerida = self._corretor_sugerido
        elif eixo == EIXO_DOS_LOTES:
            sugerida = self._lotes_sugeridos
        if sugerida is None:
            return False
        self._proposta[eixo] = sugerida
        if eixo == EIXO_DOS_LOTES:
            # Os cortes valem com o número proposto (`cortes_em_uso`), e a linha dos QCs por lote
            # é a conta sobre a partição que a correção vai usar.
            self._mostra_qcs_por_lote()
        return True

    def esquece_as_propostas_do_par(self) -> None:
        """Um par novo invalida o que o 'Diagnóstico' anterior propôs: os lotes e os cortes.

        Lotes de outro par são de outra ordem de injeção, como os cortes. Sem isso, 'Seguir
        sugestões do app' sobre o par B poria os lotes propostos para A, e o Palpite de B seria
        medido nessa partição.

        Não estende `esquece_os_cortes_propostos` de propósito: o teste dele confere a fronteira
        dos cortes com a proposta de lotes ainda de pé, e apagá-la ali tornaria aquela asserção
        vácua, com `cortes_em_uso` devolvendo `None` pelo motivo errado.
        """
        self._proposta.pop(EIXO_DOS_LOTES, None)
        self._lotes_sugeridos = None
        # O corretor sugerido e o adotado também saem: o Palpite do par anterior não vale para
        # este.
        self._proposta.pop(EIXO_DO_CORRETOR, None)
        self._corretor_sugerido = None
        self.esquece_os_cortes_propostos()

    def cortes_em_uso(self, n_lotes: int) -> tuple[int, ...] | None:
        """Os cortes do diagnóstico, se o número de lotes é o que ele propôs; senão `None`.

        A tela e o pedido partilham esta regra. Mantido o número proposto, a correção corta nas
        fronteiras propostas; mudado o número, os cortes perdem o sentido (três cortes não servem a
        dois lotes) e a correção volta aos blocos iguais.
        """
        cortes = self._cortes_propostos
        if cortes is None or n_lotes < 2 or len(cortes) != n_lotes - 1:
            return None
        if self._proposta.get(EIXO_DOS_LOTES) != n_lotes:
            return None
        return cortes

    def _serial_agora(self) -> bool:
        """Diz se este pedido vai correr em série: pela caixa ou pela capacidade do processo.

        São duas quantidades. `EM_REGIME_SERIAL` é a capacidade, resolvida no arranque e `False`
        fora do congelado; a caixa é a escolha, que muda a cada clique. Basta uma dizer série.

        Deriva de `n_jobs_efetivo` em vez de repetir a conta da rota, para a tela não prometer um
        tempo que a rota não cumpre se uma das duas mudar. E lê o regime no instante da pergunta,
        não o retrato de `EM_REGIME_SERIAL` tirado na importação, o que continua certo se a ordem
        de arranque mudar.
        """
        return n_jobs_efetivo(self.paralelo.isChecked()) == SERIE

    def analise_escolhida(self) -> AnalysisSpec:
        """O contrato de análise que as caixas deste painel descrevem.

        Parte do padrão, e não do zero: `AnalysisSpec` tem parâmetros que a janela não expõe (o
        post-hoc, a métrica e a ligação do agrupamento, o número de permutações), e montá-lo campo
        a campo congelaria os valores de hoje. `replace` sobre o padrão faz os não expostos seguirem
        o domínio.

        Os classificadores vão como tupla na ordem do vocabulário, e não na ordem em que foram
        marcados: `AnalysisSpec` recusa duplicado, a ordem é a que a figura desenha, e dois pedidos
        iguais não podem parecer diferentes.
        """
        # `dict[str, Any]` porque o mypy casa cada chave desempacotada contra cada parâmetro de
        # `replace` e reprovaria os não booleanos. A correspondência é garantida por outro lado:
        # as chaves saem de `FAMILIAS_DE_ANALISE`, que tem bijeção conferida com os campos que
        # `AnalysisSpec.pediu_alguma_coisa` lê, e chave inventada levanta `TypeError`.
        bandeiras: dict[str, Any] = {
            campo: caixa.isChecked() for campo, caixa in self.analises.items()
        }
        escolhidos = tuple(
            tipo for tipo in get_args(ClassifierKind) if self.classificadores[tipo].isChecked()
        )
        # O teste do contraste vai pelo valor do item, o rótulo do `Literal` em `ComboComNomes`, e
        # não pelo nome desenhado; `AnalysisSpec` recusa qualquer outro na construção.
        escolhas: dict[str, Any] = {"univariate_test": self.teste_univariado.currentText()}
        return replace(PADRAO_DA_ANALISE, **bandeiras, **escolhas, classifiers=escolhidos)

    def _mostra_o_que_as_analises_custam(self) -> str:
        """Reescreve as duas frases que dependem das caixas. Devolve a do bloco das análises.

        O rótulo do bloco diz quantas etapas a barra ganha, e o palpite da execução, junto do
        corretor, diz quanto tempo o número não cobre. Um chamador só para as duas impede uma de
        ficar parada quando a outra atualiza.
        """
        texto = prosa_das_etapas_pedidas(self.analise_escolhida())
        self.rotulo_das_analises.setText(texto)
        self.mostra_palpite_da_execucao()
        return texto

    def mostra_palpite_da_execucao(self) -> str:
        """Quanto 'Executar pipeline' deve custar com o corretor corrente. Devolve o que escreveu.

        Não recebe argumento: `n_features`, guardado quando a validação voltou, e o combo do
        corretor já vivem no painel, e passá-los obrigaria a janela a saber quando o combo muda.

        Não reusa o número de `mostra_palpite_da_medicao`: aquela estima o leave-one-QC-out, com
        `n_qc` reajustes por corretor nos três corretores; esta estima um ajuste de um corretor.
        """
        # O corretor que vai correr é o do seletor: a janela não delega a escolha.
        texto = estimativa_da_execucao(
            self._n_features,
            self.corretor.currentText(),
            serial=self._serial_agora(),
            # O vazio vale um lote, a mesma constante do payload: o mínimo do `QSpinBox` é o 'não
            # dividir em lotes', e `escolhas_do_disparo` o traduz para `None`, que a rota lê como
            # um lote. Ler o widget cru multiplicaria pelo mínimo em vez do que vai correr.
            n_lotes=self.n_lotes.value()
            if self.n_lotes.value() > self.n_lotes.minimum()
            else LOTES_QUANDO_VAZIO,
            # As análises pedidas entram na cauda do palpite, que vale 2,5 s sem nenhuma. É o mesmo
            # objeto que `campos_base` serializa, para a tela prometer o que a rota corre.
            analise=self.analise_escolhida(),
        )
        texto = com_maiuscula(texto)
        self.rotulo_do_palpite_da_execucao.setText(texto)
        return texto

    def _pede_a_curva(self) -> None:
        """Slot do botão da curva: emite o nome escrito, ou diz por que não emite.

        Nome vazio não emite, e a frase vai para o rótulo ao lado do botão: botão que recusa em
        silêncio parece botão morto.
        """
        feature = self.feature_da_curva.currentText().strip()
        if not feature:
            self.rotulo_da_curva.setText(FRASE_DA_CURVA_SEM_FEATURE)
            return
        self.rotulo_da_curva.setText(f"Pedindo a curva do LOESS de {feature}...")
        self.ver_curva_do_loess.emit(feature)

    def povoa_features_da_curva(self, nomes: list[str]) -> None:
        """Põe os nomes das features da matriz validada no combo da curva.

        Preserva o que já estava escrito, como `povoa_referencias`: um nome digitado antes de
        validar não pode sumir quando a lista chega.
        """
        escrito = self.feature_da_curva.currentText()
        self.feature_da_curva.clear()
        self.feature_da_curva.addItems(nomes)
        self.feature_da_curva.setCurrentText(escrito)

    def mostra_curva_do_loess(self, texto: str) -> None:
        """Escreve o desfecho do pedido da curva, ao lado do botão que o fez."""
        self.rotulo_da_curva.setText(texto)

    def mostra_medicao(self, corpo: dict[str, Any] | None) -> str:
        """Escreve a recomendação medida, em prosa. Devolve o que escreveu.

        Devolver o texto deixa o teste afirmar o que foi escrito sem reconstruir a frase. A prosa
        vem de `interface/fluxo`, e não é escrita aqui, para haver uma redação só da mesma resposta.

        `None` escreve a frase de espera, e não string vazia: um rótulo em branco ao lado do botão
        se lê como 'a medição falhou', e não como 'ainda não medi'.
        """
        if not corpo:
            texto = SEM_MEDICAO
        else:
            texto = prosa_da_medicao(
                vencedor=corpo.get("vencedor"),
                scores=corpo.get("scores"),
                margem=corpo.get("margem"),
                avaliavel=corpo.get("avaliavel"),
                abstem=corpo.get("abstem"),
                n_features_medidas=corpo.get("n_features_medidas"),
                # O corretor que não foi medido, e o motivo.
                recusas=corpo.get("recusas"),
            )
            par = prosa_do_par(
                corpo.get("palpite"), corpo.get("palpite_acertou"), corpo.get("vencedor")
            )
            # Linha em branco entre a medição e o par: no `MarkdownText` do Qt é o que separa dois
            # parágrafos, e sem ela as duas frases se colam.
            texto = f"{texto}\n\n{par}"
        self.rotulo_da_medicao.setText(texto)
        return texto

    def mostra_ressalva(self, metodo: str) -> tuple[str, str]:
        """Escreve a ressalva do método escolhido, com o texto de `METHOD_CAVEATS`.

        Devolve `(resumo, detalhe)` para o teste afirmar o que foi escrito sem reconstruir a
        ressalva. Só o `detalhe` é desenhado, atrás do botão 'ver o recorte medido'; o `resumo`
        continua a chegar ao `--help`, ao `stderr`, ao `--log` e à API por `cli.py`.

        Método sem ressalva escreve uma frase, e não string vazia: um rótulo em branco se lê como
        'a ressalva não carregou'. O texto nunca é reescrito aqui, para não haver duas redações.
        """
        ressalva = METHOD_CAVEATS.get(metodo)
        if ressalva is None:
            resumo, detalhe = SEM_RESSALVA, ""
        else:
            resumo, detalhe = ressalva.resumo, ressalva.detalhe
        # Sem ressalva não há `detalhe`, e o `resumo` diz que não há; um branco atrás do botão
        # aberto se leria como 'não carregou'.
        self.rotulo_do_detalhe.setText(detalhe or resumo)
        return resumo, detalhe

    # ------------------------------------------------------------------
    # Seguir sugestões do app
    # ------------------------------------------------------------------

    def segue_a_proposta(self, eixo: str) -> Any:
        """Põe no controle do `eixo` o que a máquina propôs, por cima do que a pessoa escolheu.
        Devolve a proposta, ou `None` sem proposta para pôr.

        Esquecer o toque antes de reaplicar é o que torna o clique um desfazer: sem o `discard`,
        `pre_preenche`, que só age em eixo não tocado, recusaria escrever. O toque só sai quando há
        proposta, e volta se o controle a recusar; senão o 'Diagnóstico' seguinte escreveria por
        cima da escolha feita à mão.

        Sem proposta, o controle fica como estava, e não em branco: o valor que vai viajar continua
        a existir. Quem chama diz por que não havia. Devolve `None` também quando o controle recusa
        a proposta (um corretor que o seletor não oferece), porque devolvê-la afirmaria que ela
        entrou na tela.

        Para os lotes e o corretor, a proposta é a sugestão guardada do 'Diagnóstico' e do
        Palpite: este clique a adota, e `pre_preenche` a registra como o lado da máquina do par,
        o que também torna os cortes utilizáveis.
        """
        proposta = self._proposta.get(eixo)
        if eixo == EIXO_DOS_LOTES and self._lotes_sugeridos is not None:
            proposta = self._lotes_sugeridos
        if eixo == EIXO_DO_CORRETOR and self._corretor_sugerido is not None:
            proposta = self._corretor_sugerido
        if proposta is None:
            return None
        tocado = eixo in self._tocados
        self._tocados.discard(eixo)
        self.pre_preenche(eixo, proposta)
        widget = self._controles_de_valor().get(eixo)
        if widget is None or self._le_valor(widget) != proposta:
            if tocado:
                self._tocados.add(eixo)
            return None
        return proposta

    def repoe_os_valores_iniciais(self) -> int:
        """Volta os campos da Normalização aos valores com que a tela abriu. Devolve quantos mudou.

        A escrita passa por `_escreve_valor`, que é escrita da máquina: não marca o eixo como
        tocado pela pessoa, nem apaga a frase das sugestões que a janela escreve a seguir. Os
        sinais de cada campo continuam a disparar, e é isso que faz as ressalvas, as equivalências e
        o palpite seguirem o valor reposto.
        """
        mudados = 0
        for campo, valor in zip(
            self._campos_dos_valores_iniciais, self._valores_iniciais, strict=True
        ):
            _aceitou, mudou = self._escreve_valor(campo, valor)
            mudados += mudou
        return mudados

    def _desfaz_a_sugestao(self, etapa: int) -> None:
        """Um campo sugerido da `etapa` mudou: avisa a janela, se quem mudou foi a pessoa."""
        if not self._escrevendo:
            self.sugestao_desfeita.emit(etapa)

    def mostra_equivalente_do_eixo(self, eixo: str, valor: str) -> str:
        """Escreve o nome que o MetaboAnalyst dá ao valor escolhido.

        Irmã de `mostra_ressalva_do_eixo`, e devolve a frase pela mesma razão: o teste afirma o que
        foi escrito sem reconstruir a expectativa. A guarda de `alvo is None` cobre um eixo que não
        é da grade, que é erro de programação e não estado da tela.
        """
        frase = prosa_da_equivalencia(eixo, valor)
        alvo = self.equivalencias_dos_eixos.get(eixo)
        if alvo is not None:
            alvo.setText(frase)
            # Frase vazia (com `none`, a maioria dos eixos não tem equivalente) esconde a linha,
            # para não deixar um vão no formulário; não há texto a perder.
            alvo.setHidden(not frase)
        return frase

    def mostra_ressalva_do_eixo(self, eixo: str, valor: str) -> tuple[str, str]:
        """Escreve a ressalva do valor escolhido num eixo da grade; irmã de `mostra_ressalva`.

        Devolve `(resumo, detalhe)` pela mesma razão que a irmã. Valor sem ressalva escreve uma
        frase, e não string vazia: o rótulo só existe nos eixos que têm alguma ressalva, e um
        branco ali não diria se é 'não há' ou 'não carregou'. O texto sai de
        `RESSALVAS_POR_EIXO` e nunca é reescrito aqui.
        """
        ressalva = RESSALVAS_POR_EIXO.get(eixo, {}).get(valor)
        if ressalva is None:
            resumo, detalhe = SEM_RESSALVA, ""
        else:
            resumo, detalhe = f"**ATENÇÃO — {ressalva.resumo}**", ressalva.detalhe
        alvo = self.ressalvas_dos_eixos.get(eixo)
        if alvo is not None:
            alvo.setText(f"{resumo}\n\n{detalhe}" if detalhe else resumo)
        return resumo, detalhe

    def _recusa_a_combinacao_proibida(self) -> None:
        """Recusa `imputation='none'` com normalização ativa, na língua de quem clicou.

        Sem imputar, as células vazias continuam vazias, e normalizar sobre elas devolve uma tabela
        que parece pronta e não está. A rota também recusa, mas não diz o que fazer a seguir. É
        método, e não bloco inline, porque todo caminho que envia os eixos precisa da mesma recusa.
        """
        normalizacoes = {
            "Normalização por amostra": self.eixos["sample_normalization"].currentText(),
            "Transformação": self.eixos["transformation"].currentText(),
            "Escalonamento": self.eixos["scaling"].currentText(),
        }
        ativas = [rotulo for rotulo, valor in normalizacoes.items() if valor != "none"]
        if self.eixos["imputation"].currentText() == "none" and ativas:
            # As opções vão pelo nome que os seletores mostram, e não pelo valor técnico (`none`,
            # `one_fifth_min`). O padrão sai de `EIXOS_DA_GRADE`, de onde o seletor nasce.
            nenhum = f"«{NOMES_DAS_OPCOES['none']}»"
            padrao = next(p for chave, _r, p, _o in EIXOS_DA_GRADE if chave == "imputation")
            raise CombinacaoProibidaError(
                f"«Imputação de ausentes» está em {nenhum}, então "
                + ("os seletores " if len(ativas) > 1 else "o seletor ")
                + " e ".join(f"«{rotulo}»" for rotulo in ativas)
                + (" têm" if len(ativas) > 1 else " tem")
                + f" de ficar em {nenhum} também. Sem imputar, as células vazias continuam vazias, "
                "e normalizar sobre elas devolve uma tabela que parece pronta e não está. Escolha "
                f"um método de imputação — «{NOMES_DAS_OPCOES[padrao]}» é o padrão —, ou ponha "
                + ("os três" if len(ativas) > 1 else "esse seletor")
                + f" em {nenhum} para levar a matriz corrigida sem normalizar."
            )

    def campos_do_artefato(self, chave: str) -> dict[str, Any]:
        """Os campos deste painel que a rota do artefato `chave` aceita.

        O mapa vem de `CAMPOS_POR_ARTEFATO`, declarado por artefato porque as rotas aceitam
        conjuntos diferentes. Mandar a união a todas seria pior que não mandar nada: o FastAPI
        descarta em silêncio o campo não declarado, e o pedido afirmaria uma influência que a rota
        não tem.

        O corretor tem de viajar: sem ele a rota corrige com `none`, e como trocar o corretor
        descarta os bytes e pede de novo, a tela mostraria atividade e entregaria sempre a matriz
        por corrigir. `null_denominator` fica de fora porque o painel não o controla. A semente vai
        como `SEMENTE_DA_JANELA`, a mesma da execução, porque o resumo metodológico do arquivo diz
        a semente a partir do pedido.
        """
        nomes = CAMPOS_POR_ARTEFATO.get(chave, ())
        # A combinação proibida é sobre normalizar sem imputar, e só existe onde o pedido leva os
        # eixos; recusar nos outros artefatos bloquearia um gesto legítimo. A condição deriva de
        # `CAMPOS_POR_ARTEFATO` para seguir sozinha se outra rota passar a levar os eixos.
        if set(nomes) & {eixo for eixo, *_ in EIXOS_DA_GRADE}:
            self._recusa_a_combinacao_proibida()
        disponiveis: dict[str, Any] = {
            "method": self.corretor.currentText(),
            # Os dois da exportação entram sempre: os widgets têm valor por construção, e não há
            # vazio a traduzir.
            "max_features": self.max_features.value(),
            "rsd_variant": self.variante_do_rsd.currentText(),
            # Entra sempre, pelo mesmo motivo dos dois de cima: a caixa tem valor por construção.
            "paralelo": self.paralelo.isChecked(),
            **self.valores_dos_eixos(),
        }
        escolhas = self.escolhas_do_disparo()
        # O número de lotes viaja sempre, com o vazio como `LOTES_QUANDO_VAZIO` (o padrão das
        # rotas), porque o resumo metodológico do arquivo diz os lotes a partir do pedido; o mínimo
        # do campo, `0`, nunca sai daqui. O corte de RSD só entra quando foi escolhido: o mínimo do
        # `QSpinBox` é o vazio, e `0` está fora de `(0, 1]`.
        disponiveis["n_batches"] = escolhas.get("n_lotes") or LOTES_QUANDO_VAZIO
        disponiveis["seed"] = SEMENTE_DA_JANELA
        # Os cortes só vão quando o número de lotes é o que o diagnóstico propôs (`cortes_em_uso`);
        # sem número escolhido, a rota corrige com `LOTES_QUANDO_VAZIO` e não há corte a mandar.
        cortes = self.cortes_em_uso(escolhas.get("n_lotes") or LOTES_QUANDO_VAZIO)
        if cortes is not None:
            disponiveis["batch_cuts"] = json.dumps(list(cortes))
        if escolhas.get("rsd_cutoff") is not None:
            disponiveis["rsd_cutoff"] = escolhas["rsd_cutoff"]
        # O corte de ausentes também só entra quando escolhido: mandar o mínimo afirmaria 'não
        # descarte nada' e anularia o padrão do servidor sem ninguém o ter pedido.
        if escolhas.get("missing_threshold") is not None:
            disponiveis["missing_threshold"] = escolhas["missing_threshold"]
        # As frações de posto idem: o mínimo enviado seria `0.0`, fora de `(0, 1]`, e daria `422`
        # sobre um pedido que ninguém fez; omitir devolve a etapa à regra por tamanho. Vão uma a
        # uma, com chave literal, para a atribuição ser auditável estaticamente.
        if escolhas.get("ranking_keep_fraction") is not None:
            disponiveis["ranking_keep_fraction"] = escolhas["ranking_keep_fraction"]
        if escolhas.get("ranking_keep_fraction_2") is not None:
            disponiveis["ranking_keep_fraction_2"] = escolhas["ranking_keep_fraction_2"]
        if escolhas.get("normalization_reference"):
            disponiveis["normalization_reference"] = escolhas["normalization_reference"]
        # Nenhum download da janela declara `auto`: o do Resultado leva o corretor efetivo da
        # etapa feita.
        return {nome: disponiveis[nome] for nome in nomes if nome in disponiveis}

    def valores_dos_eixos(self) -> dict[str, str]:
        """O que os combos de eixo escolheram, por chave de eixo.

        Um acessor só, em vez de ler os widgets em cada lugar, para não haver duas fontes do mesmo
        estado. Devolve `str`, e não `Literal`, de propósito: o texto de um `QComboBox` não tem
        garantia estática, e quem o valida é `PipelineSpec`, na rota, com `validate_label`.
        """
        return {chave: combo.currentText() for chave, combo in self.eixos.items()}

    def proposta_do_pedido(
        self,
        campos: dict[str, Any],
        eixos: list[str] | tuple[str, ...],
        recomendacao: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """A linha da máquina do par para o pedido `campos`: `{"proposta_json": ...}`, vazio
        quando a proposta foi seguida sem eixo sobrescrito, ou `{"sem_proposta": True}` quando não
        houve proposta da máquina a seguir.

        A aba 'Correção' abre nos valores padrão, e as sugestões só entram por 'Seguir sugestões
        do app'. Sem esse clique não há proposta adotada, e o vazio gravaria `nao_sobrescreveu`
        ('a proposta da máquina foi seguida') sobre uma proposta que ninguém seguiu; daí o
        terceiro desfecho, que a rota grava como `sem_proposta`.

        A Correção monta o par no clique dela, sem a recusa da combinação proibida, que é do
        pré-processamento e não pode travar uma correção; as entradas que derivam da Correção o
        levam à execução.

        `campos` tem de ser o pedido completo, com o corte de RSD dentro. A proposta da máquina é
        a spec da pessoa com os eixos da máquina no lugar; montada sobre um pedido sem o corte, ela
        sairia sempre sem corte, e a comparação diria 'discordando' sobre o que ninguém mudou.

        `_sobrescrita_ou_422` recusa as duas direções incoerentes, e cada uma é fechada aqui:
        - proposta sem eixos, por construção: `proposta_json` só entra quando há `eixos`;
        - eixos sem proposta, por recusa local com `PropostaAusenteError`, antes de chamar a rota:
          o corretor pede o 'Palpite', e o número de lotes pede o 'Diagnóstico'. Mandar o pedido
          inválido faria a rota responder 'Faltou um campo', culpando a pessoa por um defeito da
          tela; a exceção diz o que fazer.
        """
        palpite = (recomendacao or {}).get("correction_proposta")
        da_maquina = self._eixos_propostos_pela_maquina(palpite)
        sem_proposta = [eixo for eixo in eixos if eixo not in da_maquina]
        if sem_proposta:
            # A mensagem nomeia o botão que traz a proposta de volta.
            botoes = sorted({BOTAO_DA_PROPOSTA.get(eixo, eixo) for eixo in sem_proposta})
            raise PropostaAusenteError(
                f"você mudou {', '.join(NOME_DO_EIXO.get(e, e) for e in sem_proposta)} em relação "
                "ao que a máquina propôs, e a proposta dela não está disponível para o registro "
                f"de concordância. Peça {' e '.join(botoes)} de novo antes de executar, ou volte o "
                "valor ao proposto"
            )
        if not eixos:
            # Sem eixo, o vazio quer dizer 'proposta seguida', e só vale se houve uma. Com eixo,
            # `da_maquina` nunca está vazio aqui: a recusa acima já levantou.
            return {} if da_maquina else {CAMPO_SEM_PROPOSTA: True}
        # A linha da máquina leva todos os eixos que ela propôs, para quem muda só os lotes não
        # ficar registrado como concordando. A origem é a do 'Palpite' quando ele propôs o
        # corretor; sem ele, a proposta é só do diagnóstico, a classificação por regras
        # (`heuristica`).
        return {
            "proposta_json": _proposta_da_maquina(
                campos,
                str(da_maquina.get(EIXO_DO_CORRETOR, campos["correction"])),
                str((recomendacao or {}).get("rationale") or "")
                if palpite
                else (self._justificativa_dos_lotes()),
                str((recomendacao or {}).get("origem") or "modelo") if palpite else "heuristica",
                outros={k: v for k, v in da_maquina.items() if k != EIXO_DO_CORRETOR},
            )
        }

    def _eixos_propostos_pela_maquina(self, palpite: object) -> dict[str, Any]:
        """Os eixos que a máquina propôs, com o valor dela: o corretor do 'Palpite', e o número de
        lotes e os cortes do 'Diagnóstico'."""
        da_maquina: dict[str, Any] = {}
        if palpite:
            da_maquina[EIXO_DO_CORRETOR] = str(palpite)
        lotes = self._proposta.get(EIXO_DOS_LOTES)
        if isinstance(lotes, int) and not isinstance(lotes, bool):
            da_maquina[EIXO_DOS_LOTES] = lotes
            cortes = self._cortes_propostos
            da_maquina[CAMPO_DOS_CORTES] = (
                list(cortes) if cortes is not None and len(cortes) == lotes - 1 else None
            )
        return da_maquina

    def _justificativa_dos_lotes(self) -> str:
        """A justificativa da linha da máquina quando só o diagnóstico propôs."""
        lotes = self._proposta.get(EIXO_DOS_LOTES)
        cortes = self._cortes_propostos
        if cortes:
            antes = ", ".join(str(c) for c in cortes)
            return f"diagnóstico: {lotes} lotes, com cortes antes das injeções {antes}"
        return f"diagnóstico: {lotes} lote(s)"

    def campos_base(
        self,
        modo: str,
        correcao: str,
        n_lotes: int,
        *,
        eixos: list[str] | tuple[str, ...] = (),
        rsd_cutoff: float | None = None,
    ) -> dict[str, Any]:
        """Os campos do pedido, a base de que as três etapas partem.

        Cada etapa filtra dele o que a rota dela aceita (`desktop.etapas.campos_da_rota`), e a
        entrada escolhida substitui o que ela herda (`EstadoDasEtapas.campos_da_entrada`). Um
        dicionário só para as três é o que faz cada uma mandar o que a tela mostra.
        """

        campos: dict[str, Any] = {
            "mode": modo,
            "estudo": self.estudo.text() or "sem_identificacao",
            "correction": correcao,
            "n_batches": n_lotes,
            # Os seis eixos da grade, lidos dos widgets, e não por parâmetro: são controles deste
            # painel, e um parâmetro a mais seria um segundo canal para o mesmo estado.
            **self.valores_dos_eixos(),
            "rsd_variant": self.variante_do_rsd.currentText(),
            # A caixa dos núcleos, lida do widget pelo mesmo motivo. Sem ela, `/execucoes`
            # construiria `AVAILABLE_METHODS[spec.correction]()`, cujo padrão no SERRF é
            # `n_jobs=1`, e o SERRF correria em série mesmo fora do congelado.
            "paralelo": self.paralelo.isChecked(),
            # O contrato de análise viaja sempre e inteiro. Sempre, porque uma família desmarcada
            # tem de chegar como `false`: omiti-la faria a rota aplicar o padrão, que liga três, e
            # desmarcar 'ANOVA' seria ignorado sem aviso. Inteiro, derivado por
            # `analise_para_json`, porque o contrato tem mais campos que as caixas da janela, e é a
            # mesma serialização da chave de invalidação. Nos classificadores, marcar acrescenta à
            # lista e desmarcar tira; nas bandeiras, troca o valor.
            CAMPO_DA_ANALISE: json.dumps(
                analise_para_json(self.analise_escolhida()), ensure_ascii=False, sort_keys=True
            ),
            **FIXOS_DECLARADOS,
            "campos_sobrescritos": json.dumps(list(eixos)),
        }
        if rsd_cutoff is not None:
            campos["rsd_cutoff"] = rsd_cutoff

        # As frações de posto são controles deste painel, lidas aqui pelo mesmo motivo dos eixos.
        # São condicionais: o mínimo do widget é o vazio, e `0.0` daria `422` na guarda `(0, 1]`
        # sobre um pedido que ninguém fez.
        escolhas = self.escolhas_do_disparo()
        if escolhas.get("ranking_keep_fraction") is not None:
            campos["ranking_keep_fraction"] = escolhas["ranking_keep_fraction"]
        if escolhas.get("ranking_keep_fraction_2") is not None:
            campos["ranking_keep_fraction_2"] = escolhas["ranking_keep_fraction_2"]
        # O corte por fração de ausentes segue a mesma regra condicional. Valor lido e não enviado
        # não levanta erro: a matriz voltaria sem o filtro pedido, sem `422` e sem aviso.
        if escolhas.get("missing_threshold") is not None:
            campos["missing_threshold"] = escolhas["missing_threshold"]
        # A referência vai pela mesma porta, e o vazio não viaja: a guarda do contrato recusa
        # string vazia, e mandar `""` daria `422` sobre uma escolha que ninguém fez.
        if escolhas.get("normalization_reference"):
            campos["normalization_reference"] = escolhas["normalization_reference"]
        # Os cortes do diagnóstico vão quando o número de lotes é o que ele propôs. Sem eles, a
        # correção parte a corrida em blocos iguais, e o degrau achado pode cair no meio de um
        # lote. A rota lê a falta do campo como blocos iguais, por isso ele só viaja com cortes.
        cortes = self.cortes_em_uso(n_lotes)
        if cortes is not None:
            campos["batch_cuts"] = json.dumps(list(cortes))

        # A janela manda sempre o corretor do seletor, e nunca `auto`; quem delega a escolha é a
        # linha de comando, com `--auto`. A proposta da máquina viaja no par de decisão
        # (`proposta_do_pedido`), comparada com o que a tela mostra.

        # Os grupos viajam em par ou não viajam: a rota recusa um grupo sozinho com `422`, em vez
        # de ignorá-lo e executar uma grade de horas, e mandar metade daqui produziria esse erro no
        # fim de uma escolha que o painel podia ter completado ou limpado.
        grupo_a = self.grupo_a.text().strip()
        grupo_b = self.grupo_b.text().strip()
        if grupo_a and grupo_b:
            campos["group_a"] = grupo_a
            campos["group_b"] = grupo_b
        return campos

    # ------------------------------------------------------------------
    # A proposta da máquina: pré-preenchimento e divergência
    # ------------------------------------------------------------------
    #
    # Não há caixa de marcar 'eu decido': mudar um campo que a máquina preencheu é a decisão manual
    # da pessoa. Por isso a proposta é escrita no campo antes de comparar. Derivar a sobrescrita da
    # divergência sem pré-preencher mediria a distância entre o repouso do combo (`none`) e a
    # proposta, e toda execução sairia como manual; pré-preencher é o que faz 'igual à proposta'
    # significar 'aceitei'.
    #
    # Perde-se uma distinção, de propósito: 'abri, avaliei e mantive' e 'nunca abri o controle'
    # gravam o mesmo `origem="modelo"`.

    def proposta_da_maquina(self) -> dict[str, Any]:
        """O que a máquina propôs para cada eixo, ou vazio enquanto nada chegou.

        Devolve uma cópia: o dicionário é o lado da máquina do par de concordância, e um
        consumidor que lhe escrevesse mudaria o que a execução seguinte considera divergência.
        """
        return dict(self._proposta)

    def _marca_tocado(self, eixo: str) -> bool:
        """Marca o eixo como mexido pela pessoa. Devolve se marcou.

        Ignora a escrita da máquina: `setCurrentText` e `setValue` emitem os mesmos sinais que um
        clique, e o Qt não diz quem os originou. Sem a guarda, pré-preencher marcaria o eixo como
        tocado, e a leitura seguinte gravaria sobrescrita para quem não mexeu em nada. O `bool`
        deixa o teste afirmar a distinção sem olhar um conjunto privado.
        """
        if self._escrevendo:
            return False
        self._tocados.add(eixo)
        return True

    def pre_preenche(self, eixo: str, valor: Any) -> bool:
        """Põe a proposta da máquina no controle daquele eixo. Devolve se ela mudou o widget.

        Só age enquanto a pessoa não tocou no eixo: as leituras correm em paralelo e podem chegar
        depois da escolha dela, e um pré-preenchimento tardio a apagaria sem aviso. `_tocados`
        guarda quem já mexeu, alimentado pelos sinais dos próprios widgets. O `bool` deixa o teste
        afirmar que o pré-preenchimento aconteceu em vez de inferi-lo do widget.

        Eixo que este painel não controla é ignorado: a rota da recomendação pode ganhar um campo
        antes de a janela ganhar o controle dele, e levantar derrubaria a exibição da leitura. Eixo
        sem proposta nunca entra em `campos_sobrescritos`, porque não há com que comparar.
        """
        widget = self._controles_de_valor().get(eixo)
        if widget is None:
            return False
        if eixo in self._tocados:
            # A proposta é registrada mesmo com o eixo tocado: é o lado da máquina do par, e sem
            # ela a escolha que a pessoa fez antes de a leitura voltar não contaria como
            # sobrescrita.
            self._proposta[eixo] = valor
            return False
        # A proposta só é registrada se o widget aceitar o valor; senão `_proposta` guardaria um
        # valor que a tela nunca mostrou, e `eixos_divergentes` acusaria sobrescrita contra ele.
        # Aceitar e mudar são perguntas diferentes (ver `_escreve_valor`): propor o valor em que o
        # combo já repousa aceita sem mudar, e isso não é recusa.
        aceitou, mudou = self._escreve_valor(widget, valor)
        if not aceitou:
            return False
        self._proposta[eixo] = valor
        if eixo == EIXO_DO_CORRETOR:
            # O palpite da execução passa a descrever o corretor proposto, e não o anterior.
            self.mostra_palpite_da_execucao()
        return mudou

    def eixos_divergentes(self) -> tuple[str, ...]:
        """Os eixos cujo valor corrente difere do que a máquina propôs.

        Só entram eixos com proposta. Os seis da grade, o corte de RSD, o corte de ausentes e a
        variante do RSD não são propostos por rota nenhuma, e sem o lado da máquina não há par a
        gravar: `/execucoes` recusaria com `422` um eixo sobrescrito sem proposta.

        A ordem é a de `CONTROLES_DE_SOBRESCRITA`, e não a de edição: o par de concordância compara
        textos de spec, e a ordem de edição daria specs diferentes para a mesma escolha.
        """
        controles = self._controles_de_valor()
        return tuple(
            controle.eixo
            for controle in CONTROLES_DE_SOBRESCRITA
            if controle.eixo in self._proposta
            and controle.eixo in controles
            and self._le_valor(controles[controle.eixo]) != self._proposta[controle.eixo]
        )

    def povoa_referencias(self, nomes: list[str]) -> None:
        """Enche o combo de referência com os nomes que a validação devolveu.

        Preserva o que a pessoa escreveu: o controle é editável para ela poder digitar antes de
        validar, e um `clear()` cego apagaria a escolha quando a lista chega. Não insere o texto
        digitado na lista (`NoInsert`): a lista é o que o arquivo tem, e um nome inventado
        pareceria existir na matriz.
        """
        # A escrita é da máquina e passa pela bandeira: `self.referencia` está em
        # `_controles_de_valor`, e o `currentTextChanged` dele marca toque. Sem a bandeira, cada
        # validação marcaria o eixo como mexido pela pessoa.
        escrito = self.referencia.currentText()
        self._escrevendo = True
        try:
            self.referencia.clear()
            self.referencia.addItems(nomes)
            self.referencia.setCurrentText(escrito)
        finally:
            self._escrevendo = False

    def _controles_de_valor(self) -> dict[str, Any]:
        """O widget que carrega o valor de cada eixo, por chave de eixo.

        Derivado dos widgets que já existem, sem tabela nova: os seis da grade saem de
        `self.eixos`, derivado de `EIXOS_DA_GRADE`; os outros são nomeados aqui porque cada um é um
        widget próprio com um tipo próprio.
        """
        return {
            EIXO_DO_CORRETOR: self.corretor,
            "n_batches": self.n_lotes,
            "rsd_cutoff": self.corte,
            "missing_threshold": self.ausentes,
            "ranking_keep_fraction": self.fracao_posto,
            "ranking_keep_fraction_2": self.fracao_posto_2,
            "normalization_reference": self.referencia,
            "rsd_variant": self.variante_do_rsd,
            **self.eixos,
        }

    def etapas_dos_eixos(self) -> dict[str, int]:
        """A etapa cuja página mostra o controle de valor de cada eixo, lida dos grupos.

        Derivada da árvore de widgets, e não de uma tabela: o controle está na página da etapa cujo
        grupo o contém (`grupos_das_etapas`), e a justificativa repartida usa isso para dizer cada
        eixo na aba onde a pessoa o muda. Uma tabela à mão divergiria quando um controle mudasse de
        grupo. Eixo cujo controle não está em grupo nenhum fica de fora, e quem precisa dele
        levanta.
        """
        return {
            eixo: etapa
            for eixo, widget in self._controles_de_valor().items()
            for etapa, grupos in self.grupos_das_etapas.items()
            if any(grupo.isAncestorOf(widget) for grupo in grupos)
        }

    @staticmethod
    def _le_valor(widget: Any) -> Any:
        """O valor corrente, na moeda em que a proposta chega: texto ou número."""
        if isinstance(widget, QComboBox):
            return widget.currentText()
        return widget.value()

    def _escreve_valor(self, widget: Any, valor: Any) -> tuple[bool, bool]:
        """Escreve, e devolve `(aceitou, mudou)`, que são perguntas diferentes.

        `setCurrentText` num `QComboBox` não editável com texto fora da lista não levanta: deixa o
        combo como estava, e só comparar o valor depois diz se a escrita aconteceu. Mas 'o widget
        não mudou' cobre dois estados opostos: ele recusou o valor, ou já estava nele. Com um
        `bool` só, propor `none` sobre um combo em `none` seria lido como recusa, a proposta não
        ficaria registrada, e a mudança seguinte da pessoa não contaria como sobrescrita.
        - `aceitou`: o widget ficou com o valor; decide se a proposta é registrada.
        - `mudou`: o widget mudou; é o que `pre_preenche` devolve.
        """
        antes = PainelDePipeline._le_valor(widget)
        self._escrevendo = True
        try:
            self._aplica_valor(widget, valor)
        finally:
            self._escrevendo = False
        depois = PainelDePipeline._le_valor(widget)
        return depois == PainelDePipeline._le_valor_pedido(widget, valor), depois != antes

    @staticmethod
    def _le_valor_pedido(widget: Any, valor: Any) -> Any:
        """O valor pedido, na moeda em que `_le_valor` devolve: `str` para combo, número para spin.

        Sem esta tradução a comparação seria entre `"3"` e `3`, e todo pré-preenchimento de número
        seria lido como recusa.
        """
        return str(valor) if isinstance(widget, QComboBox) else valor

    @staticmethod
    def _aplica_valor(widget: Any, valor: Any) -> None:
        """A escrita nua, separada de `_escreve_valor` para a guarda ter um bloco a envolver.

        Não devolve nada: quem decide se a escrita pegou é a leitura de depois, e não a intenção
        de quem escreveu.
        """
        if isinstance(widget, QComboBox):
            widget.setCurrentText(str(valor))
        elif isinstance(valor, (int, float)) and not isinstance(valor, bool):
            widget.setValue(valor)

    # ------------------------------------------------------------------
    # O disparo e a leitura
    # ------------------------------------------------------------------

    def escolhas_do_disparo(self) -> dict[str, Any]:
        """O que os controles deste painel escolheram, com o vazio já traduzido para `None`.

        O `modo` não está aqui, como em `ParDeArquivos`: ele vive no painel de arquivos e é lido
        no instante da chamada. Guardá-lo aqui faria quem escolhe o modo, nota o erro e corrige o
        combo enviar o modo antigo com a tela mostrando o novo.

        O mínimo dos campos numéricos não viaja. `QSpinBox` sempre tem valor, e o vazio é o
        `specialValueText` no lugar do mínimo; aqui ele vira `None`, que a rota lê como 'não
        escolhi'. Um `0` enviado significaria ao mesmo tempo 'corte de zero', que a API recusa por
        ser fração em `(0, 1]`, e 'sem corte'.
        """
        lotes = self.n_lotes.value()
        corte = self.corte.value()
        ausentes = self.ausentes.value()
        fracao = self.fracao_posto.value()
        fracao_2 = self.fracao_posto_2.value()
        return {
            "correcao": self.corretor.currentText(),
            "n_lotes": None if lotes == self.n_lotes.minimum() else lotes,
            "rsd_cutoff": None if corte == self.corte.minimum() else corte,
            # Todos os campos com vazio seguem a mesma tradução, para haver uma regra só do que
            # 'não escolhi' significa.
            "ranking_keep_fraction": (None if fracao == self.fracao_posto.minimum() else fracao),
            "ranking_keep_fraction_2": (
                None if fracao_2 == self.fracao_posto_2.minimum() else fracao_2
            ),
            # O texto do combo editável, aparado; vazio é 'não escolhi'.
            "normalization_reference": self.referencia.currentText().strip() or None,
            "missing_threshold": None if ausentes == self.ausentes.minimum() else ausentes,
            # A caixa dos núcleos viaja sempre: é binária, não tem vazio, e omiti-la deixaria o
            # caso comum no padrão do servidor.
            "paralelo": self.paralelo.isChecked(),
        }

    def anota_o_pedido(self, correcao: str, *, delegou: bool) -> None:
        """Guarda quem escolheu o corretor e qual, para `aceita_disparo`.

        O pedido fica guardado no clique, e não no aceite, porque entre os dois a pessoa pode mexer
        no seletor. Quem clica é a Análise, com o corretor da entrada escolhida (o da Correção
        feita, ou nenhum), e a janela nunca delega: passa `delegou=False`. `delegou` é obrigatório
        para nenhuma omissão afirmar que a política escolheu.
        """
        self._pedido_do_disparo = (delegou, correcao)

    def aceita_disparo(self, corpo: Any, instante: float) -> str | None:
        """Guarda o identificador do `202` e carimba o disparo pelo funil. Devolve o id.

        O carimbo passa por `carimba_disparo`, e não por `deposito.guarda`, porque é o funil que
        descarta o decorrido congelado da corrida anterior; escrever a chave direto deixaria na
        tela um contador parado de outra execução, e o portão do funil acusa quem o faça.

        Um corpo de leitura sai antes de tudo: ele também traz `identificador`, passaria pelas
        outras guardas e recarimbaria o disparo. A guarda vem antes da limpeza porque esta apaga o
        identificador, e o que uma leitura precisa é que a função não corra. O predicado é
        `e_corpo_de_leitura`, partilhado com `e_corpo_de_aceite` para os dois não divergirem.

        Depois, a limpeza é incondicional e vem antes das outras guardas: quem disparou não quer
        ver a corrida passada, aceito ou não o disparo. Sem ela, o progresso e a barra da corrida
        anterior ficariam na tela, e um disparo recusado preservaria o identificador antigo, que a
        sondagem automática leria como o corrente.
        """
        if e_corpo_de_leitura(corpo):
            return None
        self._identificador = None
        # Os textos de repouso são prefixados ('Resultado', 'Progresso'), para os dois rótulos lado
        # a lado não parecerem contraditórios; esta função também os escreve.
        self.rotulo_de_resultado.setText(RESULTADO_EM_REPOUSO)
        self.rotulo_de_progresso.setText(PROGRESSO_EM_REPOUSO)
        # O corretor e a etapa vista também são da corrida anterior: o corretor de uma execução ao
        # lado do progresso de outra seria o mesmo defeito.
        self.rotulo_do_corretor.setText(CORRETOR_EM_REPOUSO)
        self._corretor_em_curso = None
        self._etapa_vista = None
        self.barras_de_etapa[ETAPAS[-1]].setRange(0, 0)
        # O carimbo do depósito sai também: sem isso, depois de um disparo recusado, a próxima
        # leitura mediria o decorrido desde a corrida anterior.
        carimba_disparo(self._deposito, CHAVE_DO_INICIO, None)

        if not isinstance(corpo, dict):
            return None
        identificador = corpo.get("identificador")
        if not identificador:
            return None
        self._identificador = str(identificador)
        carimba_disparo(self._deposito, CHAVE_DO_INICIO, instante)
        # Qual corretor, e quem o escolheu. A delegação vem no aceite, e não no resultado, porque
        # é antes de esperar que a pessoa quer saber.
        delegou, pedido = self._pedido_do_disparo or (None, None)
        self.rotulo_do_corretor.setText(texto_do_corretor(corpo, delegou=delegou, pedido=pedido))
        delegacao = corpo.get("delegacao")
        if isinstance(delegacao, dict) and delegacao.get("correction"):
            self._corretor_em_curso = str(delegacao["correction"])
        elif delegou is False:
            self._corretor_em_curso = pedido
        return self._identificador

    def recebe_leitura(self, codigo: int, corpo: Any) -> str:
        """Trata uma leitura de `GET /trabalhos/{id}`. Devolve o texto de progresso escrito.

        `codigo` decide a terminalidade, e não `corpo["terminou"]` (o docstring do módulo diz por
        quê). Num `200`, o corpo é o resultado e vai também para `rotulo_de_resultado`. A barra é
        a da Análise, no rodapé dela, porque a execução é a etapa 3.
        """
        barra = self.barras_de_etapa[ETAPAS[-1]]
        terminou = codigo != EM_CURSO
        decorrido = decorrido_da_leitura(
            self._deposito,
            chave_do_inicio=CHAVE_DO_INICIO,
            terminou=terminou,
            relogio=self._relogio,
        )
        texto = texto_da_leitura(
            codigo, corpo, decorrido, etapa=self._etapa_em_curso(corpo, codigo, decorrido)
        )

        if codigo == EM_CURSO and isinstance(corpo, dict):
            total = corpo.get("total")
            # O corpo de aceite também move a barra: ele traz `total` e não `feito`, e por isso
            # `feito` cai em zero.
            feito = corpo.get("feito", 0)
            if isinstance(total, int) and total > 0 and isinstance(feito, int):
                barra.setRange(0, total)
                barra.setValue(feito)
            else:
                # Sem `total`, indeterminada: há trabalho e não se sabe quanto falta.
                barra.setRange(0, 0)
        elif codigo == 200:
            # Concluído, a barra vai a cheio: a última leitura de progresso pode tê-la deixado em
            # 3/4, e ela discordaria do 'concluído'.
            barra.setRange(0, 1)
            barra.setValue(1)
            self.mostra_resultado(corpo)

        self.rotulo_de_progresso.setText(texto)
        return texto

    def _etapa_em_curso(self, corpo: Any, codigo: int, decorrido: tuple[float, bool] | None) -> str:
        """A frase da etapa em curso, e o registro de quando cada `feito` foi visto.

        O registro muda só quando `feito` muda, o que dá à etapa um relógio que não recomeça a cada
        consulta: a primeira leitura que vê `feito == k` guarda o decorrido dela, e as seguintes
        medem a partir dali. Na primeira etapa o início é o disparo, porque `_executa_trabalho` põe
        `executando` antes de começar a corrigir.
        """
        if codigo != EM_CURSO or not e_corpo_de_leitura(corpo):
            return ""
        feito = corpo.get("feito")
        if not isinstance(feito, int):
            return ""
        segundos = None if decorrido is None else decorrido[0]
        if self._etapa_vista is None or self._etapa_vista[0] != feito:
            self._etapa_vista = (feito, 0.0 if feito == 0 else segundos)
        inicio = self._etapa_vista[1]
        desde = None if segundos is None or inicio is None else segundos - inicio
        return texto_da_etapa_em_curso(corpo, corretor=self._corretor_em_curso, desde_s=desde)

    def mostra_resultado(self, corpo: Any) -> str:
        """Escreve o resultado reclamado e devolve o texto escrito."""
        texto = texto_do_resultado(corpo)
        self.rotulo_de_resultado.setText(texto)
        return texto

    def foto_do_grupo_da_execucao(self) -> FotoDoGrupoDaExecucao:
        """O que o grupo da execução mostra agora, para a janela guardar com o desfecho."""
        return FotoDoGrupoDaExecucao(
            self.rotulo_de_progresso.text(),
            self.rotulo_de_resultado.text(),
            self.rotulo_do_corretor.text(),
        )

    def repoe_o_grupo_da_execucao(self, foto: FotoDoGrupoDaExecucao | None) -> None:
        """Escreve no grupo da execução o que uma foto guardou, ou o repouso com `None`.

        Só os widgets: o identificador do trabalho e o carimbo do disparo são da execução em voo,
        e quem os troca é `aceita_disparo`. A barra da Análise volta ao estado da etapa, porque a
        última leitura de progresso pode tê-la deixado noutro ponto (ver `FotoDoGrupoDaExecucao`).
        """
        foto = GRUPO_DA_EXECUCAO_EM_REPOUSO if foto is None else foto
        self.rotulo_de_progresso.setText(foto.progresso)
        self.rotulo_de_resultado.setText(foto.resultado)
        self.rotulo_do_corretor.setText(foto.corretor)
        self._poe_a_barra_no_estado(ETAPAS[-1], self._estados_das_etapas[-1])

    @property
    def identificador(self) -> str | None:
        """O trabalho corrente, ou `None`. Só leitura: quem o troca é `aceita_disparo`."""
        return self._identificador
