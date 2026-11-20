"""O painel das leituras: validação, diagnóstico e Palpite, e a justificativa das propostas.

Cada leitura produz texto, e o texto difere entre o desfecho bom e o mau; nada aqui devolve
`bool`. `/matrizes/validar` responde `200` nos dois casos, com `aceita: false` e a lista de
problemas no corpo, e um `bool` derivado do código HTTP diria 'deu certo' sobre uma matriz
recusada. A abstenção também não vira número: `unassessable` traz `confidence=0.0`, e
`fluxo.prosa_da_confianca_do_diagnostico` e `fluxo.prosa_da_recomendacao` dizem que não houve
medição em vez de imprimir '0%'.

Os pontos de decisão e os controles saem de `interface/fluxo.py` (`PONTOS_DE_DECISAO_AUTOMATICA`,
`CONTROLES_DE_SOBRESCRITA`), e não de lista nova: cada ponto precisa de justificativa e controle,
e uma lista escrita à mão divergiria no dia em que um ponto nascesse. Este painel compõe texto e
junta campos; quem chama a rota é a janela, e o portão de AST sobre `desktop/janela/` reprova um
`envia` fora de `lambda`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Final

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFrame,
    QGroupBox,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from refino.desktop.janela.componentes import CartaoDeLeitura, GradeResponsiva, com_maiuscula
from refino.interface.fluxo import (
    COMO_O_PALPITE_DECIDE,
    CONTROLES_DE_SOBRESCRITA,
    EIXOS_DA_GRADE,
    PONTOS_DE_DECISAO_AUTOMATICA,
    PROPOSTAS_POR_ROTA,
    ControleDaTela,
    PontoDeDecisao,
    eixos_propostos_do_ponto,
    eixos_so_do_humano_do_ponto,
    frase_das_recusas,
    justificativa_em_prosa,
    partes_do_defeito,
    prosa_da_confianca_do_diagnostico,
    prosa_da_recomendacao,
    prosa_do_defeito,
    prosa_do_palpite_medido,
)

if TYPE_CHECKING:  # pragma: no cover - só para o mypy
    from collections.abc import Mapping

__all__ = [
    "CHAVE_DO_PALPITE",
    "FRASES_DA_LEITURA",
    "NOMES_DOS_CORRETORES",
    "ROTAS_DA_LEITURA",
    "ROTA_DO_PALPITE",
    "ROTULOS_DA_LEITURA",
    "ROTULOS_DOS_EIXOS",
    "ROTULO_DO_BOTAO_DE_LEITURA",
    "SEM_LEITURA",
    "TITULO_DA_JUSTIFICATIVA",
    "EixoSemRotuloError",
    "PainelDeLeitura",
    "resumo_da_leitura",
    "rotulo_do_eixo",
    "texto_da_recomendacao",
    "texto_do_diagnostico",
    "texto_dos_problemas",
]


class EixoSemRotuloError(KeyError):
    """Um eixo de sobrescrita não tem nome na língua de quem opera.

    Erro e não recuo, pela razão que `rotulo_do_eixo` declara: o recuo silencioso mostraria o
    nome do campo da API na tela.
    """


# As três rotas da leitura, na ordem em que os cartões se mostram. É também a ordem da cadeia:
# pedir o Palpite roda antes a Validação e o Diagnóstico que faltarem, e quem impõe isso é a
# janela (`principal.ORDEM_DAS_LEITURAS`), que dispara a leitura seguinte quando a anterior
# chega. Um portão confere que as duas ordens são a mesma.
#
# O Palpite mede: corrige uma amostra de 200 features com cada corretor, leva perto de um minuto
# e responde `202`, como as outras rotas longas. Quem encadeia a espera é a janela, e este painel
# mostra o progresso por `mostra_espera`. A chave continua `recomendacao`, que é o ponto de
# decisão de `interface/fluxo.py` e a linha de `PROPOSTAS_POR_ROTA`.
CHAVE_DO_PALPITE: Final = "recomendacao"
ROTA_DO_PALPITE: Final = "/matrizes/palpite"
ROTAS_DA_LEITURA: Final[tuple[tuple[str, str], ...]] = (
    ("validacao", "/matrizes/validar"),
    ("diagnostico", "/matrizes/diagnosticar"),
    (CHAVE_DO_PALPITE, ROTA_DO_PALPITE),
)

SEM_LEITURA: Final = "Nenhuma leitura feita"

# O título do bloco que explica as propostas. Ele explica e não oferece controle: quem assume a
# escolha assume-a mudando o valor no controle da etapa.
TITULO_DA_JUSTIFICATIVA: Final = "Por que o programa propôs isto"

# O nome de cada eixo de sobrescrita na língua de quem opera. Os eixos são nomes de campo da
# API, em inglês; um mapa, e não só tirar as crases, porque o que a pessoa precisa ler é o nome
# da coisa.
#
# Quatro são derivados de `EIXOS_DA_GRADE`, que já traz o rótulo em português de `imputation`,
# `sample_normalization`, `transformation` e `scaling`, o mesmo que o painel do pipeline mostra
# ao lado de cada combo; redigitá-los daria duas frases para o mesmo eixo. Os outros são campos
# de spec e não eixos da grade, escritos aqui com o vocabulário que o painel do pipeline usa nos
# rótulos deles. A bijeção contra `CONTROLES_DE_SOBRESCRITA` tem portão, nas duas direções.
#
# Sem artigo: o gênero varia com o substantivo, e derivá-lo produziria 'a escalonamento'. Os
# rótulos entram numa lista de nomes, sob um título que já abre a frase.
ROTULOS_DOS_EIXOS: Final[dict[str, str]] = {
    "correction": "corretor recomendado",
    "rsd_cutoff": "corte de RSD",
    "n_batches": "número de lotes",
    # Derivados de `EIXOS_DA_GRADE`, com a inicial em minúscula para caberem numa lista.
    **{
        chave: f"{rotulo[0].lower()}{rotulo[1:]}"
        for chave, rotulo, _padrao, _opcoes in EIXOS_DA_GRADE
    },
    "missing_threshold": "corte por fração de ausentes",
    # As frações do filtro por posto, escritas à mão como o corte de RSD: `EIXOS_DA_GRADE` só
    # traz eixos de vocabulário, e estes são numéricos. O rótulo diz 'retida' porque é essa a
    # metade que a etiqueta do widget promete.
    "ranking_keep_fraction": "fração retida no 1º passe por posto",
    "ranking_keep_fraction_2": "fração retida no 2º passe por posto",
    "normalization_reference": "referência da normalização",
    "rsd_variant": "variante do RSD dos QCs",
}


def rotulo_do_eixo(eixo: str) -> str:
    """O nome do eixo na língua de quem opera. Levanta em vez de recuar.

    O recuo natural, `ROTULOS_DOS_EIXOS.get(eixo, eixo)`, mostraria o nome do campo da API em
    inglês na tela, sem nada acusar. Um mapa por chave só sobrevive se a chave em falta
    levantar, a mesma decisão de `desktop/recorte.SemTraducaoError` para o mapa irmão.
    """
    try:
        return ROTULOS_DOS_EIXOS[eixo]
    except KeyError:  # pragma: no cover - o portão da bijeção torna isto inalcançável
        raise EixoSemRotuloError(
            f"o eixo `{eixo}` não tem rótulo em ROTULOS_DOS_EIXOS. Sem ele a caixa mostraria o "
            f"nome do campo da API em inglês, que é o defeito que aquele mapa existe para não ter"
        ) from None


# Os rótulos dos botões, acentuados: as chaves são identificadores e não têm acento por desenho,
# e o rótulo é para ler. Dicionário separado, e não um terceiro elemento na tupla, porque
# `principal.py` faz `dict(ROTAS_DA_LEITURA)` para resolver a rota. A bijeção entre os dois tem
# portão.
ROTULOS_DA_LEITURA: Final[dict[str, str]] = {
    "validacao": "Validação",
    "diagnostico": "Diagnóstico",
    # 'Palpite' e não 'Recomendação': é uma medição numa amostra de 200 features, e a medição
    # de todas as features vem depois, no botão 'Medir qual corretor vence'. Chamá-la
    # recomendação ancoraria a escolha na primeira tela.
    "recomendacao": "Palpite",
}

# A frase de cada cartão: uma linha do que a leitura faz. As do Diagnóstico e do Palpite dizem o
# que roda antes deles, porque a cadeia é Validação, Diagnóstico e Palpite, e pedir um deles roda
# antes os que faltarem. Bijeta com `ROTULOS_DA_LEITURA`, por portão.
FRASES_DA_LEITURA: Final[dict[str, str]] = {
    "validacao": "Confere os arquivos e lista todos os problemas encontrados.",
    "diagnostico": (
        "Classifica o defeito do sinal e sugere fronteiras de lote. Roda antes a Validação, se "
        "ela faltar."
    ),
    "recomendacao": (
        "Mede qual corretor deixa os QCs mais estáveis, numa amostra de 200 features. Roda antes "
        "a Validação e o Diagnóstico que faltarem."
    ),
}

# A explicação de cada leitura, nos detalhes do cartão. No empate o Palpite continua a propor o
# vencedor da amostra e avisa que é empate (`decide_o_palpite` devolve `abstem`); o bloco 'Como o
# Palpite decide', logo abaixo nos detalhes, o detalha.
EXPLICACOES_DA_LEITURA: Final[dict[str, str]] = {
    "validacao": (
        "Confere a estrutura das tabelas, os nomes de amostras e features, a correspondência "
        "com a lista e a ordem de injeção. Também aponta valores negativos e problemas com "
        "os QCs. O resultado separa erros, avisos e informações."
    ),
    "diagnostico": (
        "Examina a variação dos QCs ao longo da ordem de injeção, classifica o padrão do sinal "
        "e sugere possíveis fronteiras de lote. A confiança indica a margem da regra de "
        "classificação; não é uma probabilidade de acerto. "
        "Precisa da Validação destes arquivos: pedido sem ela, o app a roda antes; com os "
        "arquivos recusados, ele não roda."
    ),
    "recomendacao": (
        "Corrige uma amostra de 200 features com cada um dos três corretores e mede o RSD dos "
        "QCs fora da amostra: cada QC é previsto por um ajuste que não o viu. Propõe o corretor "
        "de menor RSD. Quando a margem até o segundo colocado fica abaixo do limiar de empate, "
        "a amostra não separa os dois: ele continua a propor o primeiro e avisa que é empate, e "
        "para decidir é preciso medir todas as features. "
        "Com 7 QCs leva perto de um minuto com vários núcleos, e mais com mais QCs. "
        "Precisa da Validação e do Diagnóstico destes arquivos, nesta ordem, e é medido nos lotes "
        "que o Diagnóstico sugere: pedido sem eles, o app roda antes os que faltarem."
    ),
}


def texto_dos_problemas(corpo: Any) -> str:
    """O veredito da validação e os problemas dela: os dois desfechos numa função.

    `/matrizes/validar` responde `200` nos dois casos e põe o veredito no corpo: `aceita: bool`,
    `problemas: [{codigo, severidade, mensagem}]`, e `n_samples`/`n_features` preenchidos só
    quando aceita. A rota relata em vez de recusar com `422`, e é o desenho melhor: um `422`
    obrigaria o profissional a adivinhar o que estava errado.

    O `detail` continua tratado porque outras rotas o usam (um `422` por campo faltando vem com
    ele), e este painel exibe qualquer corpo que a leitura receba.
    """
    if not isinstance(corpo, dict):
        return SEM_LEITURA

    # O caminho da validação, e é o único que a rota de facto produz.
    if "aceita" in corpo:
        problemas = corpo.get("problemas") or []
        lista = "; ".join(
            f"{p.get('codigo')} ({p.get('severidade')}): {p.get('mensagem')}"
            for p in problemas
            if isinstance(p, dict)
        )
        # O veredito sai de `aceita`, e não da vacuidade de `problemas`: a rota devolve
        # `aceita=True` com avisos, como um `MODE_CONTENT_MISMATCH` de `severity="warning"`, e o
        # aviso existe para o profissional corrigir o seletor, e não para ler recusa.
        if corpo.get("aceita"):
            contagens = f"{corpo.get('n_features')} features, {corpo.get('n_samples')} amostras"
            return (
                f"aceita: {contagens}. Avisos: {lista}"
                if lista
                else (f"aceita: {contagens}, nenhum problema")
            )
        # Recusa sem problema declarado não é estado esperado, e cair no mesmo texto da recusa com
        # lista apagaria a diferença entre 'recusou e disse por quê' e 'recusou e não disse'.
        return f"recusada. {lista}" if lista else "recusada, e a resposta não diz por quê"

    detalhe = corpo.get("detail")
    if isinstance(detalhe, str):
        return detalhe
    if isinstance(detalhe, list):
        return "; ".join(str(item.get("msg", item)) for item in detalhe if isinstance(item, dict))
    return SEM_LEITURA


def texto_do_diagnostico(corpo: Any) -> str:
    """O padrão do defeito e a confiança dele, sem número quando não houve medição.

    As duas frases se juntam com `. `: nenhum dos rótulos de `ROTULOS_DE_DEFEITO` termina em
    pontuação, e com um espaço só a confiança colaria no fim do rótulo como se fosse uma oração
    dele. O portão afirma a junção, e não só as partes, porque contenção de substring passa
    igual com e sem o separador.
    """
    if not isinstance(corpo, dict):
        return SEM_LEITURA
    tipo = corpo.get("defect_type")
    # `prosa_do_defeito`, e não o identificador cru: `gradual_drift` e `erratic` são nomes de
    # código. As duas frases começam em maiúscula, porque a prosa de `interface/fluxo.py` começa em
    # minúscula para entrar no meio de outra frase, e aqui cada uma abre uma frase.
    return f"{com_maiuscula(prosa_do_defeito(tipo))}. " + com_maiuscula(
        prosa_da_confianca_do_diagnostico(tipo, corpo.get("confidence"))
    )


def texto_da_recomendacao(corpo: Any) -> str:
    """O corretor do Palpite e a confiança em três estados, e a segunda opinião, pela verdade.

    Os campos entram explícitos em `prosa_da_recomendacao`, e não o objeto: o portão de contrato
    atribui cada leitura à rota que a produziu e não segue objeto que chega por parâmetro.

    O cartão mostra só o Palpite e a confiança da proposta, sem a ressalva da política: ela
    continua no corpo da resposta (`ressalva_da_politica`), ao lado do seletor quando se delega
    a escolha do corretor, e em `AUTO_CAVEATS`, de onde o painel do pipeline a lê
    (`CAVEAT_DO_PALPITE`). O número que desautoriza a política pertence ao ponto em que se
    delega a decisão, e repeti-lo aqui faria a mesma frase aparecer duas vezes na janela.

    São duas opiniões, cada uma numa linha. A primeira é a do critério que 'Seguir sugestões do
    app' aplica, com o rótulo dele; a segunda é o Palpite pela verdade, sem confiança, porque a
    margem dela é de erro contra a verdade, e a porcentagem ao lado da primeira mediria outra
    coisa com o mesmo nome (ver `SegundaOpiniao`). Um corpo sem os campos novos, de um servidor
    anterior, produz a frase de sempre, numa linha.
    """
    if not isinstance(corpo, dict):
        return SEM_LEITURA
    # A recusa da rota é frase, e não um Palpite chamado `None`: o Palpite medido recusa com
    # `422` a matriz com menos de três QCs num lote, desfecho comum com os lotes que o
    # diagnóstico propõe.
    if "correction_proposta" not in corpo:
        detalhe = corpo.get("detail")
        if isinstance(detalhe, str):
            return f"O Palpite não foi medido: {detalhe}"
        return "O Palpite não foi medido, e a resposta não diz por quê — veja o aviso de desfecho"
    # O Palpite medido tem prosa própria: a abstenção dele é empate na amostra, e não uma
    # política que se abstém. `n_features_medidas` é o campo que só a medição traz.
    if "n_features_medidas" in corpo:
        confianca_do_palpite = prosa_do_palpite_medido(
            avaliavel=corpo.get("avaliavel"),
            abstem=corpo.get("abstem"),
            confianca=corpo.get("confianca"),
            margem=corpo.get("margem"),
            limiar=corpo.get("limiar_de_empate"),
            # Sem escore finito, a frase dos 'três corretores' seria falsa com um recusado.
            recusas=corpo.get("recusas"),
        )
    else:
        confianca_do_palpite = prosa_da_recomendacao(
            avaliavel=corpo.get("avaliavel"),
            abstem=corpo.get("abstem"),
            confianca=corpo.get("confianca"),
        )
    primeira = f"{corpo.get('correction_proposta')}. " + com_maiuscula(confianca_do_palpite)
    # O corretor que o Palpite não mediu, e o motivo: sem esta frase a proposta pareceria
    # escolhida entre os três. A frase é a mesma da prosa da medição, de `interface/fluxo`.
    primeira += frase_das_recusas(corpo.get("recusas"))
    rotulo = corpo.get("rotulo_da_politica")
    if rotulo:
        primeira = f"**{rotulo}** — a que «Seguir sugestões do app» aplica: {primeira}"
    pela_verdade = corpo.get("segunda_opiniao")
    if not isinstance(pela_verdade, dict):
        return primeira
    return (
        f"{primeira}\n\n**{pela_verdade.get('rotulo')}** — não decide o que corre: "
        f"{pela_verdade.get('correction_proposta')}."
    )


# O que o botão de cada cartão diz. O título do cartão já diz qual leitura
# (`ROTULOS_DA_LEITURA`), então o botão diz só a ação.
ROTULO_DO_BOTAO_DE_LEITURA: Final = "Executar"


# Como o nome de cada corretor aparece na manchete. SERRF e LOESS são siglas; `none` é a escolha
# de não corrigir, e a tela diz isso em português. Só para mostrar: o identificador que viaja
# para a API continua o de `correction_proposta`. Corretor que não esteja aqui aparece como veio.
NOMES_DOS_CORRETORES: Final[dict[str, str]] = {
    "serrf": "SERRF",
    "loess": "LOESS",
    "none": "Sem correção",
}


def _por_cento(valor: Any) -> str | None:
    """`0.87` vira `"87%"`. A escala é a de `prosa_da_confianca`: fração, e não percentagem."""
    if isinstance(valor, bool) or not isinstance(valor, int | float):
        return None
    return f"{float(valor) * 100:.0f}%"


def resumo_da_leitura(chave: str, corpo: Any) -> tuple[str, str, str]:
    """`(tipo do selo, texto do selo, manchete)` de uma leitura.

    É o que fica à vista no cartão: uma palavra de estado com cor, e uma linha. O texto inteiro
    da leitura fica em 'Detalhes'. Lê campos do corpo, e não a prosa já formatada: recortar a
    frase de `texto_dos_problemas` seria uma segunda leitura do mesmo texto, que divergiria no
    dia em que a frase mudasse, e os campos são os que o portão de contrato confere contra o
    esquema da API.
    """
    if not isinstance(corpo, dict):
        return ("erro", "Ilegível", "A resposta não pôde ser lida")
    if chave == "validacao" and "aceita" in corpo:
        problemas = [p for p in corpo.get("problemas") or [] if isinstance(p, dict)]
        if corpo.get("aceita"):
            manchete = f"{corpo.get('n_features')} features · {corpo.get('n_samples')} amostras"
            if problemas:
                return ("aviso", "Aceita com avisos", f"{manchete} · {len(problemas)} aviso(s)")
            return ("sucesso", "Aceita", manchete)
        if problemas:
            return ("erro", "Recusada", f"{len(problemas)} problema(s) — veja os detalhes")
        return ("erro", "Recusada", "Sem motivo declarado na resposta")
    if chave == "diagnostico" and "defect_type" in corpo:
        tipo = corpo.get("defect_type")
        try:
            nome = partes_do_defeito(str(tipo))[0].strip("*")
        except (KeyError, ValueError):
            nome = str(tipo)
        if tipo == "unassessable":
            return ("neutro", "Sem medição", nome.capitalize())
        confianca = _por_cento(corpo.get("confidence"))
        estado = f"{confianca} de confiança" if confianca else "Lido"
        return ("info", estado, nome.capitalize())
    if chave == "recomendacao" and "correction_proposta" in corpo:
        proposta = str(corpo.get("correction_proposta"))
        manchete = f"Corretor: {NOMES_DOS_CORRETORES.get(proposta, proposta)}"
        if corpo.get("avaliavel") is False:
            return ("neutro", "Confiança não medida", manchete)
        if corpo.get("abstem"):
            # O Palpite medido não se abstém por política: a amostra de 200 features não separou
            # os dois primeiros, e o selo diz isso. 'Abstém-se' fica para o corpo sem
            # `n_features_medidas`, que é o da política de modelo.
            selo = "Empate na amostra" if "n_features_medidas" in corpo else "Abstém-se"
            return ("aviso", selo, manchete)
        confianca = _por_cento(corpo.get("confianca"))
        return ("info", f"{confianca} de confiança" if confianca else "Lido", manchete)
    if chave == "recomendacao" and isinstance(corpo.get("detail"), str):
        # A recusa do Palpite medido (`422`, `503`), o mesmo caso de `texto_da_recomendacao`.
        return ("erro", "Não medido", "O Palpite não foi medido — veja os detalhes")
    return ("erro", "Erro", "A rota recusou o pedido — veja os detalhes")


def _onde_o_valor_proposto_entra(propostos: tuple[str, ...] | list[str]) -> str:
    """O fim da frase 'O programa propõe: …': onde cada valor proposto entra na aba 'Correção'.

    Os dois valores, o corretor do Palpite e o número de lotes do 'Diagnóstico', entram por
    'Seguir sugestões do app'; até lá a aba fica nos valores padrão, com o número de lotes em 1.
    Só fala do que foi proposto: um ponto que propõe só o corretor não ganha a metade dos lotes.
    """
    corretor, _campo = PROPOSTAS_POR_ROTA["recomendacao"]
    lotes, _campo = PROPOSTAS_POR_ROTA["diagnostico"]
    partes = []
    if corretor in propostos:
        partes.append("o corretor")
    if lotes in propostos:
        partes.append("o número de lotes")
    if not partes:
        return ""
    return (
        f" — na aba «Correção», o valor proposto ({' e '.join(partes)}) entra por «Seguir "
        "sugestões do app»; até lá, os campos ficam nos valores padrão. Mudar o valor lá é assumir "
        "a escolha."
    )


class PainelDeLeitura(QWidget):
    """Os cartões da validação, do diagnóstico e do Palpite, e a justificativa de cada ponto de
    decisão.

    Os cartões mostram a leitura e pedem a rota pelo sinal `pedir`. A justificativa nasce aqui e
    é repartida pelas páginas das etapas (`reparte_a_justificativa`), porque a proposta que ela
    explica vem do corpo das rotas deste painel (`recomendacao_guardada`).
    """

    # Um sinal com a chave, e não três sinais: as três rotas de `ROTAS_DA_LEITURA` diferem no
    # caminho e em nada mais, e três sinais seriam três nomes para o mesmo evento. A chave é a
    # mesma de `ROTAS_DA_LEITURA`, então quem recebe resolve a rota sem segunda tabela.
    pedir = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        # Três cartões de resultado. Cada um tem ícone e título, um selo de estado, uma manchete,
        # o botão, e o texto inteiro da leitura recolhido em 'Detalhes': o botão e o resultado
        # ficam no mesmo cartão, e o texto corrido não ocupa a tela.
        disposicao = QVBoxLayout(self)
        disposicao.setContentsMargins(0, 0, 0, 0)
        disposicao.setSpacing(16)
        cartoes_em_linha: list[QWidget] = []

        # `None` e não `{}`: ver `recomendacao_guardada`.
        self._recomendacao: dict[str, Any] | None = None
        # A etapa da cadeia que cada cartão pedido está mostrando, 'Etapa 2 de 3: diagnosticando
        # o sinal', guardada para o progresso do Palpite repetir ao lado da contagem de corretores
        # (`mostra_espera`). Sai quando a leitura do cartão termina.
        self._etapas: dict[str, str] = {}

        self.rotulos: dict[str, QLabel] = {}
        self.botoes: dict[str, QPushButton] = {}
        self.cartoes: dict[str, CartaoDeLeitura] = {}
        self.paineis_de_detalhes: dict[str, QFrame] = {}
        self.como_o_palpite_decide: QLabel
        for chave, _rota in ROTAS_DA_LEITURA:
            cartao = CartaoDeLeitura(
                chave,
                ROTULOS_DA_LEITURA[chave],
                FRASES_DA_LEITURA[chave],
                self,
                detalhes_no_cartao=False,
            )
            self.cartoes[chave] = cartao
            self.paineis_de_detalhes[chave] = cartao.detalhes
            cartao.detalhes.setObjectName("detalhe_da_leitura")
            corpo_do_detalhe = cartao.detalhes.layout()
            if isinstance(corpo_do_detalhe, QVBoxLayout):
                corpo_do_detalhe.setContentsMargins(16, 16, 16, 16)
                corpo_do_detalhe.setSpacing(8)
                titulo_do_detalhe = QLabel(
                    f"{ROTULOS_DA_LEITURA[chave]} — em detalhe", cartao.detalhes
                )
                titulo_do_detalhe.setObjectName("titulo_do_detalhe")
                corpo_do_detalhe.addWidget(titulo_do_detalhe)
                explicacao = QLabel(EXPLICACOES_DA_LEITURA[chave], cartao.detalhes)
                explicacao.setObjectName("explicacao_da_leitura")
                explicacao.setWordWrap(True)
                corpo_do_detalhe.addWidget(explicacao)
                if chave == CHAVE_DO_PALPITE:
                    # 'Como o Palpite decide': o critério, o empate e um exemplo medido, nos
                    # detalhes do cartão, que é onde a pessoa abre para entender.
                    self.como_o_palpite_decide = QLabel(COMO_O_PALPITE_DECIDE, cartao.detalhes)
                    self.como_o_palpite_decide.setObjectName("como_o_palpite_decide")
                    self.como_o_palpite_decide.setWordWrap(True)
                    # Markdown, pela razão de `rotulo` abaixo: `AutoText` não o reconhece.
                    self.como_o_palpite_decide.setTextFormat(Qt.TextFormat.MarkdownText)
                    corpo_do_detalhe.addWidget(self.como_o_palpite_decide)
            cartao.alterna.toggled.connect(
                lambda aberto, qual=chave: self._alterna_detalhe(qual, aberto)
            )
            botao = QPushButton(ROTULO_DO_BOTAO_DE_LEITURA, cartao)
            botao.setToolTip(f"Executar: {ROTULOS_DA_LEITURA[chave]}")
            botao.clicked.connect(lambda _=False, c=chave: self.pedir.emit(c))
            botao.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
            self.botoes[chave] = botao
            cartao.poe_botao(botao)

            rotulo = QLabel(SEM_LEITURA, cartao)
            rotulo.setWordWrap(True)
            rotulo.setObjectName("resultado_da_ficha")
            rotulo.setAlignment(Qt.AlignmentFlag.AlignTop)
            # A prosa da API vem em markdown, e `AutoText` decide entre texto simples e HTML, sem
            # markdown: sem isto a tela mostraria os asteriscos.
            rotulo.setTextFormat(Qt.TextFormat.MarkdownText)
            rotulo.hide()
            self.rotulos[chave] = rotulo
            cartao.poe_detalhe(rotulo)
            cartoes_em_linha.append(cartao)

        # Os detalhes já nascem no painel, fora dos cartões, antes de a página
        # entrar nas abas. Nenhum widget da pilha Qt muda de pai ao abrir detalhes.
        self.grade_dos_cartoes = GradeResponsiva(cartoes_em_linha, 3, self, espaco=16)
        disposicao.addWidget(self.grade_dos_cartoes)
        for detalhe in self.paineis_de_detalhes.values():
            disposicao.addWidget(detalhe)

        # ------------------------------------------------------------------
        # A justificativa, construída aqui e repartida pelas páginas das etapas
        # ------------------------------------------------------------------
        #
        # Cada parte vai à página cujo controle ela explica (`reparte_a_justificativa`): a do
        # corretor e a dos lotes à Correção, onde os dois são propostos e preenchidos; e a frase
        # dos eixos sobre os quais o programa não opina, a cada página onde o controle de um deles
        # mora. Lá ela abre pelo '?' ao lado da sugestão (`execucao.QuadroDaJustificativa`), e
        # `mostra_justificativa` continua a escrever os rótulos aqui.
        #
        # Não há caixa de marcar para assumir a decisão: o controle repousa no que a máquina
        # propôs, e mudar o valor é a sobrescrita. `PainelDePipeline.eixos_divergentes()` devolve
        # os eixos cujo valor difere da proposta, e a janela os entrega a `campos_base` e a
        # `proposta_do_pedido`. O que se perde é distinguir 'avaliei e mantive a proposta' de
        # 'nunca olhei': as duas gravam `origem="modelo"`.
        #
        # O widget nasce aqui porque `recomendacao_guardada` tem de ficar neste painel: é o corpo
        # da rota que ele pede, e `/execucoes` recusa com `422` eixo sobrescrito sem proposta da
        # máquina. Separar a justificativa da proposta que a produz poria as duas metades do mesmo
        # `422` em módulos diferentes; quem põe as partes no layout é a página de cada etapa.
        #
        # Um sub-grupo por ponto de decisão, com o título `ponto.o_que_decide`: um parágrafo solto
        # por ponto preservaria o texto e perderia o emparelhamento com o que ele justifica.
        self.justificativas: dict[str, QLabel] = {}
        self.eixos_do_ponto: dict[str, QLabel] = {}
        # A moldura de cada ponto, que `reparte_a_justificativa` põe na aba do eixo proposto; e a
        # frase dos eixos sobre os quais o programa não opina, uma por aba, que nasce lá.
        self.molduras_dos_pontos: dict[str, QGroupBox] = {}
        self.eixos_sem_proposta: dict[int, QLabel] = {}
        for ponto in PONTOS_DE_DECISAO_AUTOMATICA:
            # O título abre a moldura, então sobe a primeira letra. Nasce escondida: até a
            # composição a pôr numa página, ela é filha deste painel sem lugar no layout dele, e à
            # vista seria desenhada solta no canto.
            grupo = QGroupBox(com_maiuscula(ponto.o_que_decide), self)
            grupo.hide()
            dentro = QVBoxLayout(grupo)

            # O rótulo da justificativa não leva o nome do ponto: quem o nomeia é a moldura, e o
            # prefixo repetiria o título dela dentro da caixa. Por isso o teste lê o par montado,
            # título da moldura mais corpo, e não o rótulo sozinho.
            rotulo = QLabel(SEM_LEITURA, grupo)
            rotulo.setWordWrap(True)
            rotulo.setTextFormat(Qt.TextFormat.MarkdownText)
            self.justificativas[ponto.chave] = rotulo
            dentro.addWidget(rotulo)

            # A frase do que o programa propõe, que diz o que mais aquele ponto move. Os eixos se
            # partem em duas frases, derivadas de `PROPOSTAS_POR_ROTA`: o que a máquina propõe, na
            # moldura do ponto, e o que fica por conta da pessoa, que vai, eixo a eixo, à página
            # onde o controle de cada um mora (`reparte_a_justificativa`). Chamar de decisão
            # automática um eixo para o qual nenhuma rota propõe valor seria falso. A soma das
            # frases de todas as páginas continua a ser `CONTROLES_DE_SOBRESCRITA` do ponto.
            self.eixos_do_ponto[ponto.chave] = self._frase_do_que_o_programa_propoe(ponto, grupo)
            dentro.addWidget(self.eixos_do_ponto[ponto.chave])
            self.molduras_dos_pontos[ponto.chave] = grupo

    @staticmethod
    def _frase_do_que_o_programa_propoe(ponto: PontoDeDecisao, pai: QWidget) -> QLabel:
        """A frase do ponto que diz o que a máquina propõe e onde o valor proposto entra.

        Um ponto sem eixo proposto não tem frase, porque 'o programa propõe: ' com a lista vazia
        seria uma frase truncada na tela; `reparte_a_justificativa` recusa-o, porque sem eixo
        proposto a moldura dele não tem aba onde morar.
        """
        propostos = eixos_propostos_do_ponto(ponto.chave)
        texto = (
            "**O programa propõe:** "
            + ", ".join(rotulo_do_eixo(eixo) for eixo in propostos)
            # Uma aba só: o corretor e o número de lotes moram os dois no quadro 'Correção do
            # sinal', e `tests/test_desktop_recorte.py` reprova se esta frase disser outra.
            + _onde_o_valor_proposto_entra(propostos)
            if propostos
            else ""
        )
        rotulo = QLabel(texto, pai)
        rotulo.setWordWrap(True)
        rotulo.setTextFormat(Qt.TextFormat.MarkdownText)
        return rotulo

    def reparte_a_justificativa(self, etapa_do_eixo: Mapping[str, int]) -> dict[int, QWidget]:
        """As partes da justificativa, uma por etapa cujo controle ela explica. Chamada uma vez,
        pela composição da janela, antes de as páginas entrarem nas abas.

        `etapa_do_eixo` diz em que página mora o controle de valor de cada eixo
        (`PainelDePipeline.etapas_dos_eixos`). A moldura de cada ponto vai à página do eixo que
        ele propõe (o corretor e os lotes, os dois na Correção), e os eixos sobre os quais o
        programa não opina são ditos, numa frase por página, onde o controle de cada um mora. Na
        página, a parte abre pelo '?' ao lado da sugestão (`execucao.QuadroDaJustificativa`), e
        uma parte para uma página sem o bloco das sugestões levanta lá, em vez de sumir.

        Um eixo sem página levanta `KeyError`, e um ponto cujos eixos propostos morem em páginas
        diferentes, ou que não proponha eixo nenhum, levanta `ValueError`: um recuo razoável
        poria a justificativa numa aba que não tem o controle dela, ou não a poria em lugar
        nenhum, e ela sumiria sem nada acusar.
        """
        conteudos: dict[int, list[QWidget]] = {}

        def etapa_de(eixo: str) -> int:
            if eixo not in etapa_do_eixo:
                raise KeyError(
                    f"o eixo `{eixo}` não tem página: a justificativa dele não teria onde morar"
                )
            return etapa_do_eixo[eixo]

        for ponto in PONTOS_DE_DECISAO_AUTOMATICA:
            etapas = {etapa_de(eixo) for eixo in eixos_propostos_do_ponto(ponto.chave)}
            if len(etapas) != 1:
                raise ValueError(
                    f"o ponto `{ponto.chave}` propõe eixos em {len(etapas)} páginas, e a moldura "
                    "dele precisa de uma só"
                )
            conteudos.setdefault(etapas.pop(), []).append(self.molduras_dos_pontos[ponto.chave])
        sem_proposta: dict[int, list[str]] = {}
        for ponto in PONTOS_DE_DECISAO_AUTOMATICA:
            for eixo in eixos_so_do_humano_do_ponto(ponto.chave):
                eixos = sem_proposta.setdefault(etapa_de(eixo), [])
                if eixo not in eixos:
                    eixos.append(eixo)
        for etapa, eixos in sem_proposta.items():
            frase = QLabel(
                "**Sobre estes o programa não opina, e a escolha é sua:** "
                + ", ".join(rotulo_do_eixo(eixo) for eixo in eixos)
                + ".",
                self,
            )
            frase.setWordWrap(True)
            frase.setTextFormat(Qt.TextFormat.MarkdownText)
            self.eixos_sem_proposta[etapa] = frase
            conteudos.setdefault(etapa, []).append(frase)
        partes: dict[int, QWidget] = {}
        for etapa in sorted(conteudos):
            parte = QWidget(self)
            coluna = QVBoxLayout(parte)
            coluna.setContentsMargins(0, 0, 0, 0)
            for widget in conteudos[etapa]:
                coluna.addWidget(widget)
                widget.show()
            # A folga é do fim: sem dono, o layout a repartiria entre as molduras, e as frases
            # flutuariam no quadro que o '?' abre quando ele é mais alto que o texto.
            coluna.addStretch(1)
            partes[etapa] = parte
        return partes

    # ------------------------------------------------------------------
    # As três leituras
    # ------------------------------------------------------------------

    def _alterna_detalhe(self, chave: str, aberto: bool) -> None:
        """Só a explicação escolhida ocupa a faixa de largura inteira."""
        if aberto:
            for outra, cartao in self.cartoes.items():
                if outra != chave:
                    cartao.alterna.setChecked(False)

    def mostra(self, chave: str, corpo: Any) -> str:
        """Escreve a leitura de uma das três rotas e devolve o texto escrito.

        Também guarda o corpo do Palpite: `/execucoes` recusa eixo sobrescrito sem proposta da
        máquina, e a proposta vem daqui (ver `recomendacao_guardada`). Guardar no instante da
        exibição garante que o que a execução compara é o que o profissional viu, e não uma
        segunda chamada com outro resultado.
        """
        formatador = {
            "validacao": texto_dos_problemas,
            "diagnostico": texto_do_diagnostico,
            "recomendacao": texto_da_recomendacao,
        }[chave]
        # Corpo ilegível não é ausência de leitura. Os três formatadores devolvem `SEM_LEITURA`
        # para corpo que não é dicionário, o caso de um `500` servido como `text/plain`, que
        # `interface/cliente.py` traduz para `corpo=None`, e `_encaminha` chama este método sem
        # olhar o
        # código. A frase de repouso manda clicar, e esta manda olhar o desfecho.
        texto = (
            formatador(corpo)
            if isinstance(corpo, dict)
            else "A rota respondeu e o corpo não é legível — veja o aviso de desfecho"
        )
        self.rotulos[chave].setText(texto)
        self.rotulos[chave].show()
        self.cartoes[chave].mostra_resumo(*resumo_da_leitura(chave, corpo))
        self._etapas.pop(chave, None)
        # Só um Palpite de fato é guardado: uma recusa (`422`, `503`) chega como dicionário com
        # `detail` e apaga a anterior, porque a anterior era de outra leitura.
        if chave == CHAVE_DO_PALPITE and isinstance(corpo, dict):
            self._recomendacao = corpo if "correction_proposta" in corpo else None
        return texto

    def mostra_espera(self, chave: str, *, feito: int, total: int, leituras: int) -> str:
        """O sinal de vida do Palpite medido, com a mesma assinatura das irmãs.

        A janela reusa o laço de espera das rotas longas e escolhe o mostrador por chave; este é o
        do Palpite, que é a única leitura assíncrona. A barra anda por corretor, como na medição.
        """
        texto = (
            f"Medindo o Palpite em 200 features: **{feito} de {total}** corretores. "
            f"{leituras}.ª consulta de progresso — perto de um minuto com vários núcleos numa "
            "corrida de 7 QCs, mais em corridas maiores, e a janela continua utilizável."
        )
        self.rotulos[chave].setText(texto)
        # O progresso vai ao que está à vista: o texto inteiro fica em 'Detalhes', que o cartão
        # recolhe, e o selo e a manchete são o que a pessoa vê sem abrir nada.
        contagem = f"{feito} de {total} corretores"
        # A etapa da cadeia fica ao lado da contagem, senão 'Etapa 3 de 3: medindo o Palpite'
        # sairia da tela no primeiro `202`, que chega em menos de um segundo.
        etapa = self._etapas.get(chave)
        manchete = contagem if etapa is None else f"{etapa} — {contagem}"
        self.cartoes[chave].mostra_resumo("info", "Medindo", manchete)
        return texto

    def mostra_desistencia(self, chave: str, *, leituras: int, segundos: float) -> str:
        """O teto de leituras passou. Irmã das outras: desistir de esperar não é cancelar."""
        texto = (
            f"Parei de esperar o Palpite depois de **{leituras}** leituras e **{segundos:.0f} s**. "
            "A medição continua a correr — clicar outra vez reclama o resultado em vez de "
            "recomeçar."
        )
        self.rotulos[chave].setText(texto)
        self.cartoes[chave].mostra_resumo(
            "aviso", "Espera parada", "A medição continua — clique outra vez para o resultado"
        )
        self._etapas.pop(chave, None)
        return texto

    def mostra_etapa(self, chave: str, etapa: str) -> str:
        """A etapa da cadeia no cartão pedido, 'Etapa 1 de 3: validando os arquivos'.

        Quem pede o Palpite vê, no cartão dele, a Validação e o Diagnóstico correndo antes: sem
        isto o cartão ficava parado enquanto outros dois mudavam, e a ordem só existiria no
        código. A etapa vai ao selo e à manchete, e não ao texto de 'Detalhes', que é o da última
        leitura do cartão e fica lá até a nova chegar. Ela fica guardada para `mostra_espera`.
        """
        self._etapas[chave] = etapa
        self.cartoes[chave].mostra_resumo("info", "Em curso", etapa)
        return etapa

    def etapa_em_curso(self, chave: str) -> str | None:
        """A etapa da cadeia que o cartão `chave` mostra agora, ou `None`.

        Quem para ou avança a cadeia pergunta isto aos cartões dos alvos anteriores: o que ainda
        mostra uma etapa não recebeu desfecho nenhum desde que pediu a cadeia.
        """
        return self._etapas.get(chave)

    def mostra_a_parada(self, chave: str, frase: str, manchete: str) -> str:
        """A cadeia parou antes da leitura `chave`, e a frase diz por quê. Devolve a frase.

        A frase inteira vai a 'Detalhes' e a manchete curta fica à vista, como nas outras
        leituras: o cartão é estreito, e a frase diz também o que fazer. A leitura anterior que o
        cartão mostrava sai de vista, porque o pedido de agora não a refez.
        """
        self.rotulos[chave].setText(frase)
        self.rotulos[chave].show()
        self.cartoes[chave].mostra_resumo("aviso", "Não rodou", manchete)
        self._etapas.pop(chave, None)
        return frase

    # ------------------------------------------------------------------
    # Os pontos de decisão
    # ------------------------------------------------------------------

    def mostra_justificativa(self, ponto: PontoDeDecisao, corpo: Any) -> str:
        """A justificativa do ponto, do campo que a estrutura declara.

        O campo vem de `ponto.campo_da_justificativa`, e não de um nome escrito aqui: um ponto
        novo em `fluxo.py` traz o campo dele, e este painel o exibe sem mudar.
        """
        bruto = corpo.get(ponto.campo_da_justificativa) if isinstance(corpo, dict) else None
        # `justificativa_em_prosa`, e não `str(bruto)`: o campo do ponto `diagnostico` é `lotes`,
        # um objeto (`SugestaoDeLotes`), e `str()` despejaria o dicionário na tela, com uma
        # entrada por amostra. O texto não leva o nome do ponto, que é o título da moldura.
        texto = justificativa_em_prosa(ponto.chave, bruto)
        self.justificativas[ponto.chave].setText(texto)
        return texto

    def controles_do_ponto(self, chave: str) -> tuple[ControleDaTela, ...]:
        """Os controles que sobrescrevem um ponto, derivados e não redigitados."""
        return tuple(c for c in CONTROLES_DE_SOBRESCRITA if c.ponto == chave)

    # ------------------------------------------------------------------
    # A troca de entrada, e a proposta guardada para `/execucoes`
    # ------------------------------------------------------------------

    def descarta(self, chave: str, frase: str) -> str:
        """A resposta que chegou era de outra matriz: a frase no lugar dela. Devolve-a.

        Vai ao rótulo e ao cartão, porque o rótulo fica recolhido em 'Detalhes' e o selo e a
        manchete são o que está à vista. A proposta guardada sai, como quando uma recusa a apaga:
        a proposta que `/execucoes` compara tem de ser a que a pessoa viu, e uma anterior, de outro
        pedido, não é.
        """
        self.rotulos[chave].setText(frase)
        self.rotulos[chave].show()
        self.cartoes[chave].mostra_resumo("aviso", "Descartado", "Chegou depois da troca")
        self._etapas.pop(chave, None)
        if chave in self.justificativas:
            self.justificativas[chave].setText(SEM_LEITURA)
        if chave == CHAVE_DO_PALPITE:
            self._recomendacao = None
        return frase

    def limpa(self) -> None:
        """Descarta as leituras da entrada removida sem alterar as escolhas manuais."""
        self._recomendacao = None
        self._etapas.clear()
        for rotulo in (*self.rotulos.values(), *self.justificativas.values()):
            rotulo.setText(SEM_LEITURA)
        for rotulo in self.rotulos.values():
            rotulo.hide()
        for cartao in self.cartoes.values():
            cartao.limpa()

    def recomendacao_guardada(self) -> dict[str, Any] | None:
        """O corpo do último Palpite recebido, ou `None` se nenhum chegou.

        Guardar é requisito: `/execucoes` recusa com `422` eixo sobrescrito sem proposta da
        máquina com que comparar (é o que `PropostaAusenteError` do painel do pipeline diz), e a
        proposta vem do corpo desta rota.

        `None` e não `{}`, porque os dois pedem ações opostas: `None` é 'não pedi' e mandaria
        pedir; `{}` seria 'pedi e veio vazia', que é resposta da rota.
        """
        return self._recomendacao
