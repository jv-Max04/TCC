"""A aba 'Métodos e limites': o que o programa faz, com que ressalva, e o que não faz.

Cada seção de `SECOES_DO_RECORTE` nomeia em `onde` o painel que a cobre, ou nenhum, e
`FORA_DO_CORTE` lista o que o programa não faz. O texto que a aba mostra vem de `PARA_QUEM_OPERA`.
O estado de cada linha não é escrito aqui: `itens_para_a_janela` o deriva do que a janela informa
(os painéis que ela compõe, os que têm controle ligado a uma rota e os que calculam no próprio
processo) e põe cada linha num dos quatro grupos de `CATEGORIAS_DO_RECORTE`. Uma lista escrita à
mão envelheceria a cada painel ligado; derivada, ela muda junto com a janela.

O módulo não importa Qt: a janela só desenha o que ele devolve.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:  # pragma: no cover - só para o mypy
    from collections.abc import Mapping

__all__ = [
    "CATEGORIAS_DO_RECORTE",
    "FORA_DO_CORTE",
    "FUNCIONALIDADE_EM_BETA",
    "MARCA_DE_FORA",
    "PARA_QUEM_OPERA",
    "PREFIXO_COBERTO",
    "PREFIXO_FORA",
    "PREFIXO_PENDENTE",
    "PREFIXO_SO_LEITURA",
    "SECOES_DO_RECORTE",
    "CategoriaDoRecorte",
    "PainelDesconhecidoError",
    "SecaoDoRecorte",
    "SemTraducaoError",
    "itens_para_a_janela",
    "traduz",
]


class SemTraducaoError(KeyError):
    """Uma entrada da aba não tem texto para quem opera o programa.

    Erro e não recuo: recuar para a chave interna faria uma entrada nova aparecer na tela em
    linguagem de repositório, sem que nada acusasse.
    """


class PainelDesconhecidoError(KeyError):
    """Seção com painel que a janela não declara, ou painel operável fora da composição.

    Levanta em vez de cair num grupo, porque um valor inválido que vira estado razoável esconde o
    defeito. Os testes conferem `SECOES_DO_RECORTE` contra os painéis da janela, então isto não
    acontece em produção.
    """


# O começo do texto de uma seção sem painel. Ele torna `onde` conferível: um `onde` preenchido
# ao lado de um texto que abre com esta marca é contradição, e um portão a acusa.
MARCA_DE_FORA: Final = "Este programa não"

# Os prefixos dos quatro grupos. O painel e os testes leem estas constantes em vez de repetir
# o texto, e nenhum deles mostra a chave interna do painel a quem opera.
PREFIXO_FORA: Final = "FORA DESTE PROGRAMA"
PREFIXO_PENDENTE: Final = "AINDA NÃO"
PREFIXO_SO_LEITURA: Final = "SÓ PARA LEITURA"
PREFIXO_COBERTO: Final = "NESTE PROGRAMA"


@dataclass(frozen=True)
class CategoriaDoRecorte:
    """Um grupo da aba: o prefixo que o identifica, o nome, uma linha e o tipo do selo.

    Fica aqui e não no painel porque a frase que explica o grupo é conteúdo, e a janela só desenha.
    """

    prefixo: str
    titulo: str
    subtitulo: str
    tipo_do_selo: str


# Os quatro grupos, na ordem de `itens_para_a_janela`. Cada subtítulo diz o que põe uma linha no
# grupo, e o tipo do selo é um dos de `janela.componentes.TIPOS_DE_SELO`.
CATEGORIAS_DO_RECORTE: Final[tuple[CategoriaDoRecorte, ...]] = (
    CategoriaDoRecorte(
        PREFIXO_FORA, "Fora deste programa", "O que esta ferramenta não faz, e por quê.", "neutro"
    ),
    CategoriaDoRecorte(
        PREFIXO_PENDENTE, "Ainda não", "Painéis declarados que esta janela não compõe.", "aviso"
    ),
    CategoriaDoRecorte(
        PREFIXO_SO_LEITURA,
        "Só para leitura",
        "Abas sem controle que lance trabalho no servidor.",
        "info",
    ),
    CategoriaDoRecorte(
        PREFIXO_COBERTO,
        "Neste programa",
        "Abas com controle que lança trabalho ou calcula aqui dentro.",
        "sucesso",
    ),
)

# A frase que declara uma funcionalidade em beta. Os painéis do multiverso e das vias abrem com
# ela, e as entradas deles em `PARA_QUEM_OPERA` também: com uma fonte só, a janela não diz duas
# coisas sobre o mesmo estado. Fica neste módulo porque ele não importa Qt.
FUNCIONALIDADE_EM_BETA: Final = (
    "Funcionalidade em beta: está implementada e pode ser usada, mas não faz parte da entrega ao "
    "laboratório nem do trabalho de conclusão de curso."
)


@dataclass(frozen=True)
class SecaoDoRecorte:
    """Uma seção da aba: a chave do texto em `PARA_QUEM_OPERA` e o painel que a cobre.

    `onde` é a chave do painel, ou string vazia quando o programa não faz o que a seção
    descreve. Ele diz qual painel, não se o painel está na janela: isso a janela informa.
    """

    titulo: str
    onde: str


# A ordem das seções com painel e a das sem painel é a ordem em que aparecem em cada grupo.
SECOES_DO_RECORTE: Final[tuple[SecaoDoRecorte, ...]] = (
    SecaoDoRecorte("O que esta ferramenta NÃO faz", "recorte"),
    SecaoDoRecorte("O que esta ferramenta faz COM RESSALVA", "pipeline"),
    SecaoDoRecorte("Exportação e vias metabólicas", "recorte"),
    SecaoDoRecorte("As cinco etapas, e o que cada uma decidiu", ""),
    SecaoDoRecorte("O fluxo", ""),
    SecaoDoRecorte("Onde o sistema decide, e onde você decide", "leitura"),
    SecaoDoRecorte("Qualidade — o conjunto completo, nunca um número", "qualidade"),
    SecaoDoRecorte("PLS-DA — nunca sem validação", "graficos"),
    SecaoDoRecorte("Antes e depois da correção — o RSD dos QCs", "qualidade"),
    SecaoDoRecorte("1. Os seus arquivos", "arquivos"),
    SecaoDoRecorte("2. Validação, diagnóstico e palpite", "leitura"),
    SecaoDoRecorte("3. Trabalhos em curso", "trabalhos"),
    SecaoDoRecorte("5. Teste de robustez", "multiverso"),
    SecaoDoRecorte("4. Pipeline com parâmetros explícitos", "pipeline"),
    SecaoDoRecorte("6. Levar o resultado", "pipeline"),
    SecaoDoRecorte("Vias metabólicas", "vias"),
)


PARA_QUEM_OPERA: Final[dict[str, tuple[str, str]]] = {
    # --- seções ---
    "O que esta ferramenta NÃO faz": (
        "Os limites desta ferramenta",
        "É esta aba. Ela lista o que o programa não faz e por quê, para você não descobrir tarde, "
        "e, no fim, as referências de cada método.",
    ),
    "O que esta ferramenta faz COM RESSALVA": (
        "O que a ferramenta faz com ressalva",
        "Alguns métodos têm limites conhecidos que valem antes de você confiar no resultado. Em "
        "«Correção», o botão «Ver o recorte medido» abre o que foi medido "
        "sobre o método escolhido e sobre a escolha automática.\n\n"
        "O do SERRF é este. Ele corrige, e nos quatro arquivos deste estudo foi o melhor dos três "
        "corretores em três dos quatro modos — no quarto, HILIC em modo negativo, quem vence é o "
        "LOESS. Ele usa uma floresta aleatória que conta como a do SERRF "
        "original em R, escolhida para o laboratório reconhecer os números a que está "
        "acostumado: do excesso de variação que o original remove, este programa remove de 91% "
        "a 159% — um pouco menos no HILIC negativo, tanto ou mais nos outros três modos.\n\n"
        "O que continua sem fechar é a concordância número a número: ela passa em três dos quatro "
        "modos, e o próprio original não a tem consigo mesmo — ele não fixa a semente, então duas "
        "execuções dele sobre os mesmos dados não dão a mesma tabela.\n\n"
        "Se o seu objetivo é reproduzir número a número uma tabela saída do SERRF original, este "
        "programa chega perto e não faz isso.",
    ),
    "Exportação e vias metabólicas": (
        "Exportar para o MetaboAnalyst",
        "O programa gera o arquivo no formato que o MetaboAnalyst aceita — no «Resultado» das abas "
        "«Correção» e «Normalização», escolhendo o «Formato MetaboAnalyst».\n\n"
        "A predição de vias metabólicas acontece **aqui dentro**, na aba «Vias "
        "metabólicas (beta)», sem serviço externo e sem internet — uma funcionalidade em beta, "
        "que não faz parte da entrega ao laboratório nem do trabalho de conclusão de curso. "
        "Na mesma aba, um segundo modo faz o enriquecimento do MetaboAnalystR, que é outra "
        "coisa: ele parte de uma lista de compostos **já identificados** no Progenesis, e roda "
        "num R que vem junto com o programa — salvo no modo externo, sobre a KEGG, em que a "
        "análise é feita no servidor do MetaboAnalyst —, também em beta.",
    ),
    "As cinco etapas, e o que cada uma decidiu": (
        "Um resumo das etapas, com a alternativa de cada uma",
        "Este programa não junta num só lugar o que cada etapa — leitura, diagnóstico, lotes, "
        "correção e corte — decidiu, lado a lado com a alternativa que ela tinha. O que cada "
        "uma produz aparece na aba em que ela corre, e as escolhas que você pode mudar estão nos "
        "controles das abas «Correção», «Normalização» e «Análise estatística». Os valores que o "
        "programa propõe só são preenchidos na aba «Correção» quando você clica «Seguir "
        "sugestões do app».",
    ),
    "O fluxo": (
        "Um diagrama do fluxo completo",
        "Este programa não desenha como as etapas se encadeiam. As abas estão agrupadas por "
        "função e não impõem uma ordem: o único requisito para começar é escolher os dois "
        "arquivos na aba «Preparar o estudo» — o modo é deduzido do nome do arquivo ou dos nomes "
        "das amostras, e escolhe-se em «Opções avançadas», na mesma aba, quando não é deduzido. "
        "Cada etapa — «Correção», «Normalização» e «Análise estatística» — roda sozinha, na aba "
        "dela, a partir da matriz que você escolhe ali. A qualidade é preenchida pela análise, "
        "e as vias pedem uma análise com dois grupos declarados.",
    ),
    "Onde o sistema decide, e onde você decide": (
        "Onde o programa decide, e onde você decide",
        "O programa recomenda; você confirma ou muda. Toda recomendação vem com a razão dela. "
        "Na aba «Correção», o valor recomendado só é preenchido quando você clica «Seguir "
        "sugestões do app»; até lá, os campos ficam nos valores padrão. Mudar o valor ali é o que "
        "registra que a decisão foi sua.",
    ),
    "Qualidade — o conjunto completo, nunca um número": (
        "As métricas de qualidade, nunca resumidas a um número",
        "A aba «Qualidade do resultado» mostra o conjunto completo depois de uma execução. Ela é "
        "preenchida "
        "pelo resultado, e não há nada a clicar nela — as figuras mudaram-se para a aba "
        "«Gráficos», que fica logo ao lado. Duas das onze métricas ainda não são "
        "calculadas por nenhum caminho do programa, e aparecem marcadas como tal em vez de "
        "aparecerem com um número "
        "inventado. As duas que restam medem o RSD dos QCs fora da amostra, e ficam de fora "
        "por custo: obtê-las exige refazer a correção uma vez para cada QC.",
    ),
    "PLS-DA — nunca sem validação": (
        "PLS-DA",
        "O programa calcula o PLS-DA quando você declara em «Análise estatística» os dois "
        "grupos a "
        "comparar. Vêm sempre dois números juntos: o Q², que diz quanto o modelo acerta em "
        "amostras que ele não viu, e o p de permutação, que diz se esse acerto poderia ter saído "
        "por acaso. Nunca um sem o outro — um PLS-DA sem eles mostra separação convincente entre "
        "grupos que não existe, e foi o que houve noutro estudo do laboratório, com Q² de ~24%. "
        "O gráfico de dispersão está na aba «Gráficos», e o Q² vai escrito no TÍTULO dele: o "
        "desenho e o número que o desmente chegam juntos aos seus olhos. Os dois números também "
        "ficam na aba «Qualidade do resultado», cada um na sua moldura.",
    ),
    "Antes e depois da correção — o RSD dos QCs": (
        "O RSD dos QCs, antes e depois",
        "A comparação aparece na aba «Qualidade do resultado» depois de uma execução, com o valor "
        "antes e "
        "depois da correção.",
    ),
    "1. Os seus arquivos": (
        "Escolher os arquivos",
        "Na aba «Preparar o estudo», na primeira seção, «Os seus arquivos»: você escolhe os dois "
        "CSV obrigatórios, cada um no seu cartão e com o seu «Selecionar arquivo» — o das "
        "medições e o da lista de amostras —, e o selo no título da seção conta quantos já "
        "entraram. O modo cromatográfico é deduzido do nome do arquivo ou dos nomes das "
        "amostras, e fica em «Opções avançadas», na mesma seção, para quando não é deduzido ou "
        "para o trocar. A planilha de fatores de normalização, OPCIONAL, importa-se na aba "
        "«Normalização».",
    ),
    "2. Validação, diagnóstico e palpite": (
        "Validar, diagnosticar e palpitar",
        "Na mesma aba «Preparar o estudo», na seção de baixo, «Validação, diagnóstico e "
        "palpite»: o programa confere o que veio, mede a deriva e recomenda os parâmetros, um "
        "cartão por leitura, cada um com o seu «Executar». O que ele propõe só é preenchido na "
        "aba «Correção» quando você clica «Seguir sugestões do app»; mudar o valor é assumir a "
        "escolha.",
    ),
    "3. Trabalhos em curso": (
        "A lista de trabalhos em curso",
        "Esta é a faixa no fim da barra lateral, à esquerda, visível em qualquer aba. Ela fica "
        "recolhida e mostra só o estado — «Nenhum trabalho», «2 em execução», «Concluídos» —, e "
        "um clique no título dela a abre: ela lista o que você "
        "disparou nesta sessão, um trabalho por linha, com o nome, o estado e a etapa em curso, "
        "e permite PARAR o que ainda está a correr — no «Parar» da linha dele, ou escolhendo o "
        "trabalho no seletor e clicando em «Parar o trabalho selecionado». A "
        "parada leva menos de um segundo, contra os minutos que uma correção pode custar. Ela não "
        "sobrevive a fechar o programa: aqui o serviço interno vive dentro da janela, então fechar "
        "a janela encerra os trabalhos junto. Se uma correção estiver a meio, ela se perde.",
    ),
    "5. Teste de robustez": (
        "O teste de robustez (beta) — testar muitas combinações de uma vez",
        FUNCIONALIDADE_EM_BETA + "\n\n"
        "A aba «Teste de robustez (beta)» executa o teste de robustez e mostra quais features "
        "sobrevivem à grade, ordenadas por "
        "estabilidade. Você escolhe os corretores e o par de grupos a comparar, vê quanto a "
        "execução deve demorar ANTES de a começar, e pode pará-la a qualquer momento — o botão "
        "«Cancelar» interrompe em menos de um segundo. Duas ressalvas: ela dá um veredito por "
        "feature — sobrevive, frágil ou não sobrevive —, e o corte que o produz vem da "
        "especificação e NÃO foi calibrado para os seus dados, então leia-o como leitura e não "
        "como sentença; e a execução vive dentro deste programa, então fechá-lo a encerra e o "
        "identificador deixa de valer.",
    ),
    "4. Pipeline com parâmetros explícitos": (
        "Rodar as etapas",
        "Os parâmetros ficam nas abas «Correção», «Normalização» e «Análise estatística», e cada "
        "uma tem o seu botão no rodapé, que roda aquela etapa sobre a matriz que você escolhe na "
        "seção «Entrada» dela. Os SETE eixos são seus, e a variante do RSD também — ela deixou "
        "de ser fixa. Ficam fixas a semente e a forma de apresentação, e o programa usa as "
        "mesmas em toda execução.",
    ),
    "6. Levar o resultado": (
        "Baixar o resultado",
        "O resultado da correção e o da normalização se baixam no «Resultado» da aba de cada "
        "uma: a matriz da etapa feita, no formato do laboratório ou no do MetaboAnalyst, com o "
        "resumo dos métodos ao lado do arquivo. O botão faz o gesto inteiro: pergunta em que "
        "pasta gravar, manda calcular, e escreve o arquivo lá quando ele fica pronto. Na "
        "«Correção», «Gerar os três» gera os três corretores de uma vez. Clicar de novo sobre a "
        "MESMA entrada NÃO recalcula: o programa reencontra o trabalho que já correu e entrega o "
        "mesmo arquivo, então com o `serrf` você não paga os minutos duas vezes. A «Análise "
        "estatística» não tem download: o resultado dela fica em «Qualidade do resultado» e em "
        "«Gráficos».",
    ),
    "Vias metabólicas": (
        "Vias metabólicas (beta): predição pelos picos e enriquecimento da lista identificada",
        FUNCIONALIDADE_EM_BETA + "\n\n"
        "O botão «Predizer vias», na aba «Vias metabólicas (beta)», calcula aqui dentro e sem "
        "internet "
        "quais vias metabólicas parecem ativas, a partir dos picos de uma execução com dois "
        "grupos declarados — o mesmo algoritmo da análise «MS Peaks to Pathways» do "
        "MetaboAnalyst. O que ele NÃO faz é identificar metabólitos: os compostos por trás de "
        "cada via são candidatos compatíveis em massa, não identificações, e a aba diz isso ao "
        "lado do resultado.\n\n"
        "A mesma aba tem um segundo modo, escolhido no topo dela: o **enriquecimento da lista "
        "que você identificou no Progenesis**. Você carrega a tabela de identificação exportada "
        "de lá, confirma as colunas e escolhe o corte, e o botão «Enriquecer» faz a análise de "
        "sobre-representação do MetaboAnalystR. No modo local, que é o padrão, ela roda num R "
        "que vem junto com o programa, também aqui dentro, e nada sai da máquina. "
        "No modo externo, sobre a KEGG, a análise é feita no servidor do MetaboAnalyst: o R "
        "daqui só reconhece os nomes, e só os nomes de compostos confirmados vão para lá, depois "
        "de você confirmar o envio. Aqui os compostos são os que você identificou.",
    ),
    # --- itens fora do corte ---
    "O enriquecimento MetaboAnalystR": (
        "Enriquecimento de uma lista JÁ IDENTIFICADA (MetaboAnalystR)",
        "O enriquecimento de uma lista que você já identificou não faz parte da entrega ao "
        "laboratório nem do trabalho de conclusão de curso. Ele existe em beta, no segundo modo "
        "da aba «Vias metabólicas (beta)», que roda o MetaboAnalystR num R que vem junto com o "
        "programa — no modo local sem enviar nada; no modo externo a análise é feita no servidor "
        "do MetaboAnalyst, o R daqui só reconhece os nomes, e só os nomes de compostos "
        "confirmados saem, depois da sua confirmação. Exportar no formato do MetaboAnalyst faz "
        "parte da entrega.\n\n"
        "O primeiro modo da mesma aba prediz vias a partir "
        "dos picos e existe para dispensar a identificação; o segundo parte da lista que você "
        "identificou.",
    ),
    # O diagnóstico confunde degrau com errático: o degrau é plantado contra a variação de todas
    # as amostras e julgado na trajetória dos sete QCs.
    "Transferência de limiares entre modos": (
        "Reaproveitar um limiar de um modo em outro",
        "Não é oferecido, e o motivo é um resultado e não uma pendência: sobre 200 matrizes "
        "geradas, o diagnóstico acertou o tipo de defeito em 69,5% delas — 139 de 200 — "
        "contra os mais de 85% que consideraríamos suficiente para transferir um limiar de "
        "um modo para outro.",
    ),
    "A política de decisão automática": (
        "Decidir tudo automaticamente, sem você",
        "Existe, e vem com o número que o desautoriza como atalho: a política automática que "
        "testamos acerta o corretor em 3 dos 4 modos, e escolher sempre o mesmo método também "
        "acerta 3, com o mesmo erro — as duas escolhem o SERRF nos quatro, e no HILIC negativo "
        "quem vence é o LOESS. Medida, ela não supera a regra mais simples possível. "
        "O número honesto é este: medida sem nunca ter visto o modo em que é testada — a única "
        "forma que não se engana a si mesma —, ela acerta 67,5% de 40 matrizes simuladas, contra "
        "75,0% de escolher sempre o mesmo método — diferença que o teste de McNemar não "
        "distingue de acaso (p = 0,5078). Avaliada nas mesmas matrizes em que treinou faz 87,5%, "
        "que é o número bonito e o que mede menos; deixando uma matriz de fora de cada vez, mas "
        "ainda treinando com nove do mesmo modo, faz 55,0% contra os mesmos 75,0%, e ali a regra "
        "fixa é melhor com significância (p = 0,0078). "
        "Esta política opina pela ESTABILIDADE DOS QCs, que é o critério da literatura: prediz "
        "qual corretor deixa os QCs com menor RSD. QCs mais estáveis não são, por si, amostras "
        "mais perto do sinal verdadeiro, e isso foi medido onde a verdade existe — em matrizes "
        "simuladas: em 170 delas com um melhor corretor claro, esta política acertou o corretor "
        "que mais recupera o sinal em 69, e uma árvore treinada por esse critério acertou 132 "
        "(teste de McNemar, p = 9,6e-10). Nas 78 em que não corrigir era o melhor, esta política "
        "nunca o escolheu: ela não responde «não corrija». Essa segunda opinião foi construída e "
        "está DESLIGADA, por decisão dos autores; o Palpite mostra só esta. Nos "
        "seus arquivos não há verdade para medir qual das duas acertaria. "
        "Por isso, nesta janela, a escolha automática nunca corre sozinha: o botão «Seguir "
        "sugestões do app», no topo da aba «Correção», põe o corretor proposto no seletor "
        "«Corretor», à vista e editável, e quem corrige é o seu clique em «1. Corrigir», com o "
        "corretor que o seletor mostrar. "
        "A PROPOSTA DO BOTÃO É O PALPITE MEDIDO, E NÃO A POLÍTICA DESCRITA ACIMA. O Palpite "
        "deixou de prever e passou a medir: ele corre os três corretores — não "
        "corrigir, LOESS e SERRF — sobre 200 features sorteadas dos seus arquivos, e propõe o que "
        "deixa os QCs com o menor RSD fora da amostra. Em 88 matrizes simuladas, concordou com a "
        "medição de todas as features em 92,5% dos sorteios, contra 78% de escolher sempre o "
        "SERRF; nos quatro arquivos deste estudo propõe o mesmo corretor que a medição completa, e "
        "nos dois modos negativos diz que é empate — ali a medição completa também dá margens "
        "relativas "
        # As margens comparam a mediana do RSD dos 7 QCs fora da amostra, nas features de RSD
        # finito.
        "pequenas, de 2,56% e 3,82% do melhor. Ele custa perto de um minuto com vários núcleos "
        "numa corrida "
        "como as deste estudo, com 7 QCs; num computador de 12 núcleos, com 11 QCs levou cerca de "
        "95 s, e com 21, cerca de 215 s. A janela o "
        "mede quando você pede as sugestões, se ainda não o tiver medido para os mesmos "
        "arquivos e lotes. A política descrita acima continua no programa, pela linha de comando "
        "(`--auto aprendida`), e é a ela que os números acima se referem.",
    ),
    # O aviso do SmartScreen não foi testado; o texto segue o que se sabe do formato do
    # PyInstaller sem assinatura.
    "Assinatura de código": (
        "Assinatura digital do programa",
        "Este executável não é assinado, porque uma assinatura exige um certificado pago que o "
        "projeto não tem. Na primeira abertura o Windows pode mostrar um aviso azul — ele é "
        "esperado e não indica defeito. O arquivo LEIA-ME que acompanha a pasta explica como "
        "prosseguir.",
    ),
    "Instalador": (
        "Instalador",
        "Não há um: o programa é uma pasta que se copia e se abre. Ele não pede privilégio de "
        "administrador, e o que ele grava fica DENTRO da própria pasta — quem mais consegue "
        "ler isso depende de onde você a pôs, e o item «Quem mais nesta máquina pode ler os "
        "seus dados» diz como o programa mede isso a cada abertura.",
    ),
    # O programa avisa e não recusa, porque recusar deixaria quem recebe a pasta sem ferramenta.
    # O risco é o par apelido e grupo da amostra, que o banco guarda junto.
    "Proteção do arquivo de banco contra outras contas da máquina": (
        "Quem mais nesta máquina pode ler os seus dados",
        "O registro das execuções fica DENTRO da pasta deste programa, ao lado do refino.exe. "
        "Quem consegue lê-lo depende de ONDE essa pasta está: numa pasta sua — a Área de "
        "Trabalho, os Downloads — só você, os administradores e o próprio Windows; numa pasta "
        "comum, como a raiz do disco C: ou a pasta Público, também outras contas da máquina. O "
        "programa mede isso a cada abertura e escreve o resultado no arquivo refino.log, que fica "
        "na mesma pasta. O registro guarda o apelido da amostra ao lado do grupo dela, então "
        "trate a pasta como trataria a planilha original.",
    ),
    # A porta não autentica: qualquer processo da máquina, ou uma página aberta no navegador,
    # pode fazer POST para ela. A porta efêmera aleatória mitiga só em parte.
    "Autenticação na API embutida": (
        "Segurança da comunicação interna",
        "O programa conversa consigo mesmo por uma porta local enquanto a janela está aberta. "
        "Nada disso é alcançável de outro computador. Na própria máquina, porém, essa porta não "
        "pede senha — evite deixar a janela aberta numa máquina compartilhada com alguém em quem "
        "você não confia.",
    ),
    "O CÁLCULO das famílias de análise que as caixas pedem": (
        "As análises que você pode pedir, e o que elas já produzem",
        "A aba «Análise estatística» tem um bloco «Análises a correr», e o que você tica "
        "ali viaja no pedido e aparece na barra de progresso como uma etapa por família. **Parte "
        "delas já é calculada** junto com a execução, e o resultado aparece na aba «Gráficos»; "
        "**outra parte ainda não**: para essas, a etapa passa num instante e a figura "
        "correspondente na aba «Gráficos» diz que o dado não veio, em vez de desenhar vazio. A "
        "PERMANOVA não depende desta caixa: ela é calculada no «Diagnóstico» da aba «Preparar o "
        "estudo». Qual família está em qual caso não vai escrito aqui, porque muda a cada família "
        "implementada — a aba «Gráficos» é quem o diz, figura a figura.",
    ),
    "Controles próprios nos painéis que apenas mostram": (
        "Quais abas têm botão, e quais só mostram",
        "Nem toda aba tem o que clicar. As que só mostram são preenchidas pelo resultado de uma "
        "execução que você dispara noutra aba — e a lista acima diz, uma a uma, qual é qual.",
    ),
}


def traduz(chave: str) -> tuple[str, str]:
    """O par `(título, texto)` que a janela desenha para uma entrada da aba.

    Levanta `SemTraducaoError` quando a chave não tem tradução, em vez de recuar.
    """
    achado = PARA_QUEM_OPERA.get(chave)
    if achado is None:
        raise SemTraducaoError(
            f"a entrada {chave!r} não tem texto em `PARA_QUEM_OPERA`, então a janela a desenharia "
            f"em linguagem de repositório ou não a desenharia. Acrescente a tradução."
        )
    return achado


# O que o programa não faz, além das seções sem painel. Cada chave tem o texto em
# `PARA_QUEM_OPERA`.
FORA_DO_CORTE: Final[tuple[str, ...]] = (
    "O enriquecimento MetaboAnalystR",
    "Transferência de limiares entre modos",
    "A política de decisão automática",
    "Assinatura de código",
    "Instalador",
    "Proteção do arquivo de banco contra outras contas da máquina",
    "Autenticação na API embutida",
    "O CÁLCULO das famílias de análise que as caixas pedem",
    "Controles próprios nos painéis que apenas mostram",
)


def itens_para_a_janela(
    *,
    compostos: frozenset[str],
    operaveis: frozenset[str],
    declarados: Mapping[str, str],
    calculam_aqui: frozenset[str] = frozenset(),
) -> tuple[tuple[str, str], ...]:
    """Os pares `(rótulo, texto)` que o painel desenha, na ordem em que aparecem.

    Os argumentos vêm da janela, e é isso que torna o estado derivado em vez de afirmado:

    - `compostos`: as chaves dos painéis que a janela mostra;
    - `operaveis`: as dos painéis cujos controles a janela ligou a uma rota;
    - `declarados`: os painéis que a janela declara (só as chaves contam);
    - `calculam_aqui`: as dos painéis cujo botão calcula no próprio processo, sem rota.

    Composto não quer dizer operável: um painel pode estar numa aba sem ter controle que lance
    trabalho. Painel que calcula aqui entra no grupo 'neste programa', porque para quem opera ele
    é um botão que produz resultado. As ausências vêm primeiro, porque quem abre a aba quer saber
    o que falta.
    """
    desconhecidas = sorted(
        {s.onde for s in SECOES_DO_RECORTE if s.onde and s.onde not in declarados}
    )
    if desconhecidas:
        raise PainelDesconhecidoError(
            f"seções apontam para painel que a janela não declara: {desconhecidas}. "
            f"A janela declara {sorted(declarados)}"
        )
    # Painel operável fora da composição é defeito da janela, não um recorte.
    fantasmas = sorted((operaveis | calculam_aqui) - compostos)
    if fantasmas:
        raise PainelDesconhecidoError(
            f"painéis declarados operáveis e ausentes da composição: {fantasmas}. "
            f"A janela compõe {sorted(compostos)}"
        )

    fora_da_tela = tuple(
        (f"{PREFIXO_FORA} — {traduz(s.titulo)[0]}", traduz(s.titulo)[1])
        for s in SECOES_DO_RECORTE
        if not s.onde
    )
    fora_do_corte = tuple(
        (f"{PREFIXO_FORA} — {traduz(item)[0]}", traduz(item)[1]) for item in FORA_DO_CORTE
    )
    por_painel = tuple((s.onde, s.titulo) for s in SECOES_DO_RECORTE if s.onde)
    pendentes = tuple(
        (
            f"{PREFIXO_PENDENTE} — {traduz(entrada)[0]}",
            traduz(entrada)[1],
        )
        for onde, entrada in por_painel
        if onde not in compostos
    )
    sem_disparo = tuple(
        (
            f"{PREFIXO_SO_LEITURA} — {traduz(entrada)[0]}",
            # A frase vale para todo o grupo. Um botão que só grava arquivo, como o de exportar
            # figura, não torna o painel operável: operável quer dizer ligado a uma rota.
            f"Nenhum controle desta aba lança trabalho no servidor. {traduz(entrada)[1]}",
        )
        for onde, entrada in por_painel
        if onde in compostos and onde not in operaveis and onde not in calculam_aqui
    )
    cobertas = tuple(
        (f"{PREFIXO_COBERTO} — {traduz(entrada)[0]}", traduz(entrada)[1])
        for onde, entrada in por_painel
        if onde in operaveis or onde in calculam_aqui
    )
    return fora_da_tela + fora_do_corte + pendentes + sem_disparo + cobertas
