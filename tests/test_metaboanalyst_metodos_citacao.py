"""A citação dos MÉTODOS do MetaboAnalyst é o referente da afirmação de paridade.

**Por que este módulo existe, e não é cerimônia sobre um documento.** A ferramenta passou a
afirmar que o pré-processamento dela é o do MetaboAnalyst. Sem citação, essa afirmação é
satisfeita por qualquer semelhança, defeito que retirou um critério e que
motivou `docs/formato-metaboanalyst.md`.

Feito o documento, o modo de falha muda de lugar em vez de desaparecer: passa a ser a
**degradação silenciosa da citação**. Um item que perde a frase citada mas mantém o rótulo
`verbatim` deixa a afirmação de paridade a apoiar-se em menos coisas sem nada acusar — saída
indistinguível de sucesso.

**Este módulo é IRMÃO de `test_metaboanalyst_spec_citation.py` e cobre outro documento.**
Aquele guarda o formato de arquivo; este guarda os métodos. A forma é deliberadamente a
mesma — quem souber ler um sabe ler o outro —, e a duplicação de forma é preferível a um
módulo genérico que servisse os dois: o genérico teria de parametrizar as contagens
explícitas, e é justamente elas que não podem ser derivadas.

**Onde este módulo NÃO alcança, e a lacuna vai nomeada:** ele não confere que a frase citada
seja de fato a frase da fonte. Isso exigiria buscar a página em teste, o que a tornaria
dependência de rede e faria o portão reprovar quando o GitHub caísse — e a fonte **caduca**
de todo modo, que é a razão de a data de acesso existir. O que este módulo prova é coerência
interna da citação; a fidelidade dela ao original é procedência declarada, não verificada.
"""

import re

import pytest

from tests.conftest import REPO

CITACAO = REPO / "docs" / "metodos-metaboanalyst.md"

# Os sete elementos transcritos. A contagem é explícita e NÃO derivada do arquivo:
# um item que desaparecesse do documento faria uma contagem derivada concordar consigo mesma,
# e o portão passaria vacuamente sobre seis. É o mesmo argumento do módulo irmão.
ELEMENTOS_ESPERADOS = ("N-1", "N-2", "N-3", "N-4", "N-5", "N-6", "N-7")

# `[N-1]` é `derivado` porque as duas passadas discordaram sobre a FRAÇÃO do valor de
# substituição — uma disse metade, a outra um quinto. `[N-4]` é `derivado` porque a fonte
# descreve a regra em prosa e o que se conferiu foram os quatro pares de número, não a
# redação. Fixar os dois aqui é o que impede uma edição futura de os promover em silêncio:
# promover `[N-1]` autorizaria a afirmar qual é o padrão do MetaboAnalyst com base numa
# medição que não existe.
#
# **`[N-1]` SAIU desta lista em 09/09/2026, e a promoção tem medição.** A discordância
# entre metade e um quinto foi desempatada por uma TERCEIRA passada — a tela `ProcMissing.xhtml`
# do site, que diz `LoD (1/5 of the min. positive value)` no rótulo e repete *"1/5 of the minimum
# positive values"* no texto. A promoção que este comentário proibia era a promoção **sem**
# medição; esta tem a fonte, a data de acesso e o caminho pelo qual a tela foi alcançada.
DERIVADOS = frozenset({"N-4"})

_STATUS_VALIDOS = frozenset({"verbatim", "derivado"})

# Quantas passadas cobriram cada elemento. É dimensão SEPARADA do status, e agregá-las foi
# defeito real do módulo irmão — a primeira redação daquela citação marcou os seis itens como
# concordantes em duas passadas e um deles teve uma só.
#
# **Aqui a assimetria tem causa nomeada: a passada pela INTERFACE não aconteceu.** A
# ferramenta de navegador não ficou disponível, e os itens que dependiam de ver a tela ficaram
# com uma passada. Declarar `duas` em qualquer um deles seria afirmar concordância entre uma
# leitura que houve e outra que não houve.
# **REMEDIDO em 09/09/2026: a passada pela interface ACONTECEU e todos subiram.** O
# parágrafo acima dizia que a ausência dela era a causa da assimetria, e a assimetria acabou —
# nenhum item declara `uma`. O caminho foi outro que o previsto: a extensão de navegador não
# conectou, e as cinco telas de processamento foram alcançadas por **sessão HTTP própria**,
# carregando o dataset de exemplo do próprio site. Era indisponibilidade de UMA ferramenta.
_PASSADAS = {
    "N-1": "três",
    "N-2": "duas",
    "N-3": "duas",
    "N-4": "duas",
    "N-5": "três",
    "N-6": "três",
    "N-7": "três",
}


def _itens() -> dict[str, str]:
    """Mapeia `N-k` para o corpo da sua seção.

    Corta na próxima marca de cabeçalho de qualquer nível para que um item novo acrescentado
    no fim não herde o corpo do anterior.

    ## Esta docstring era FALSA até 10/09/2026, e a inserção de `[G-1]` expô-la

    Ela prometia *"cabeçalho de qualquer nível"* e o corte olhava só para dois: o próximo
    `### [N-k]` e o próximo `## `. Um `###` de outra série passava despercebido, e o corpo do
    **último** `[N-k]` engolia tudo até à seção seguinte.

    Foi o que aconteceu ao acrescentar o levantamento dos gráficos: `[N-7]` passou a declarar
    **cinco** blocos de citação onde a tabela mede três, e o portão reprovou — pelo motivo certo,
    e sobre o alvo errado. O defeito era do fatiador, não do documento.

    **A promessa vem antes do mecanismo, e o mecanismo passa a cumpri-la:** o corte agora é no
    próximo cabeçalho markdown de qualquer nível, que é o que a frase acima sempre disse.
    """
    texto = CITACAO.read_text(encoding="utf-8")
    marcas = list(re.finditer(r"^### `\[(N-\d+)\]`", texto, re.MULTILINE))
    corpos: dict[str, str] = {}
    for indice, marca in enumerate(marcas):
        inicio = marca.end()
        seguinte = re.search(r"^#{1,6} ", texto[inicio:], re.MULTILINE)
        fim = marcas[indice + 1].start() if indice + 1 < len(marcas) else len(texto)
        if seguinte is not None:
            fim = min(fim, inicio + seguinte.start())
        corpos[marca.group(1)] = texto[inicio:fim]
    return corpos


def _status(corpo: str) -> str | None:
    achado = re.search(r"\*\*Status\.\*\*\s*`([^`]+)`", corpo)
    return achado.group(1) if achado else None


def _passadas(corpo: str) -> str | None:
    achado = re.search(r"\*\*Passadas\.\*\*\s*`([^`]+)`", corpo)
    return achado.group(1) if achado else None


def test_a_citacao_existe() -> None:
    assert CITACAO.is_file(), (
        f"{CITACAO} não existe. É o referente da afirmação de paridade com o MetaboAnalyst, "
        "e sem ele ela é satisfeita por qualquer semelhança (AR66)."
    )


def test_carrega_fonte_e_data_de_acesso() -> None:
    """Sem as duas a caducidade fica indetectável.

    A data não é metadado decorativo: ela é o que permite a alguém, no futuro, perguntar se a
    fonte ainda diz isto. Citação sem data não é remedível, porque não se sabe em relação a
    quando ela era verdadeira. E o MetaboAnalystR é código VIVO — muda mais depressa que uma
    página de documentação.
    """
    texto = CITACAO.read_text(encoding="utf-8")
    assert "MetaboAnalystR" in texto, "a citação não nomeia a fonte"
    assert "general_norm_utils.R" in texto, (
        "a citação não nomeia o arquivo de onde as fórmulas de normalização saíram"
    )
    assert re.search(r"\b2026-\d{2}-\d{2}\b", texto), (
        "a citação não traz data de acesso em ISO. Sem ela, ninguém sabe em relação a que "
        "instante a transcrição era verdadeira."
    )


def test_os_sete_elementos_estao_presentes_e_sem_lacuna() -> None:
    achados = tuple(_itens())
    assert achados == ELEMENTOS_ESPERADOS, (
        f"os elementos da citação são {achados}, esperava {ELEMENTOS_ESPERADOS}. "
        "Item que desaparece reduz o que a afirmação de paridade cobre, sem nada acusar."
    )


@pytest.mark.parametrize("item", ELEMENTOS_ESPERADOS)
def test_todo_elemento_declara_status_legivel_por_maquina(item: str) -> None:
    status = _status(_itens()[item])
    assert status in _STATUS_VALIDOS, (
        f"`[{item}]` tem status {status!r}, e a leitura precisa de distinguir "
        f"{sorted(_STATUS_VALIDOS)} para não tratar derivado como citação."
    )


@pytest.mark.parametrize("item", ELEMENTOS_ESPERADOS)
def test_todo_elemento_declara_quantas_passadas_o_cobriram(item: str) -> None:
    """Concordância é dimensão separada do `verbatim`, e agregá-las foi defeito real.

    Aumentar a contagem exige extração nova, não edição de prosa. Em particular: os quatro
    itens com `uma` passada só passam a `duas` quando alguém percorrer a interface do
    MetaboAnalyst, que é a leitura que faltou.
    """
    assert _passadas(_itens()[item]) == _PASSADAS[item], (
        f"`[{item}]` declara {_passadas(_itens()[item])!r} passadas, e o medido é "
        f"{_PASSADAS[item]!r}. Aumentar a contagem exige extração nova, não edição de prosa."
    )


def test_o_status_derivado_e_exatamente_o_medido() -> None:
    """Promover `[N-1]` a `verbatim` afirmaria uma fração que as duas passadas disputaram."""
    derivados = {item for item, corpo in _itens().items() if _status(corpo) == "derivado"}
    assert derivados == DERIVADOS, (
        f"os elementos `derivado` são {sorted(derivados)}, esperava {sorted(DERIVADOS)}. "
        "Mudança aqui altera o que o documento está autorizado a afirmar, e precisa de "
        "decisão registrada — não de edição de teste."
    )


# Quantos BLOCOS de citação cada item `verbatim` carrega, e quantos caracteres citados ao
# todo. **Os dois números foram MEDIDOS antes de este critério ser escrito**, e a medição é o
# que o torna um oráculo em vez de um limiar sem origem:
#
#     N-2: 1 bloco,  375 chars     N-5: 2 blocos, 533 chars
#     N-3: 1 bloco,  218 chars     N-6: 2 blocos, 310 chars
#                                  N-7: 2 blocos, 298 chars
#
# **A primeira versão deste teste dizia apenas «existe ao menos uma linha citada», e ela NÃO
# apanhou a planta.** Removendo um dos dois blocos de `[N-5]` sobravam seis linhas do outro, e
# o portão passava — codificara a forma da instância em vez da propriedade. Contar
# blocos é o que separa «a citação está lá» de «a citação está inteira».
#
# **REMEDIDO em 09/09/2026, e o volume CRESCEU em todos.** A passada pela tela acrescentou
# um bloco a cada item — os rótulos que o usuário vê, que o código-fonte não declara — e `[N-1]`
# entrou na lista ao subir a `verbatim`. Os números abaixo foram medidos com as funções deste
# próprio módulo sobre o documento reescrito, não estimados:
#
#     N-1: 2 blocos, 215 chars     N-5: 3 blocos, 854 chars
#     N-2: 3 blocos, 823 chars     N-6: 3 blocos, 537 chars
#     N-3: 2 blocos, 626 chars     N-7: 3 blocos, 605 chars
BLOCOS_ESPERADOS = {"N-1": 2, "N-2": 3, "N-3": 2, "N-5": 3, "N-6": 3, "N-7": 3}

# Quantos CARACTERES citados cada item carrega, medidos. **É contagem exata e não piso, e a
# rigidez é a propriedade e não um efeito colateral.** Um piso de 150 deixava passar a planta
# que tirava o `>` de UMA linha de `[N-5]`: os blocos continuavam dois — as linhas restantes
# seguem contíguas — e o volume caía de 533 para 436, ainda muito acima do piso.
#
# Para uma citação `verbatim`, qualquer mudança no texto citado exige extração nova da fonte.
# É isso que o rótulo significa, e um número exato é a única forma de o portão o exigir. Quando
# a fonte mudar de verdade, a contagem muda aqui **com a data de acesso nova ao lado** — que é
# trabalho deliberado, não edição de prosa.
CARACTERES_CITADOS = {"N-1": 215, "N-2": 823, "N-3": 626, "N-5": 854, "N-6": 537, "N-7": 605}


def _linhas_citadas(corpo: str) -> list[str]:
    return [linha for linha in corpo.splitlines() if linha.lstrip().startswith(">")]


def _blocos_de_citacao(corpo: str) -> int:
    """Grupos contíguos de linhas citadas. Duas linhas seguidas são um bloco, não dois."""
    blocos, dentro = 0, False
    for linha in corpo.splitlines():
        citada = linha.lstrip().startswith(">")
        if citada and not dentro:
            blocos += 1
        dentro = citada
    return blocos


@pytest.mark.parametrize("item", sorted(BLOCOS_ESPERADOS))
def test_verbatim_carrega_os_blocos_de_citacao_medidos(item: str) -> None:
    """A asserção que carrega o peso do módulo.

    Rótulo `verbatim` sem a citação inteira é o rótulo mentindo: o item continua a parecer
    transcrição e transcreve menos do que declara. O modo de falha: o
    documento fica menor e o portão continua verde.
    """
    corpo = _itens()[item]
    assert _blocos_de_citacao(corpo) == BLOCOS_ESPERADOS[item], (
        f"`[{item}]` tem {_blocos_de_citacao(corpo)} blocos de citação e o medido é "
        f"{BLOCOS_ESPERADOS[item]}. Bloco que some leva com ele parte do que a paridade "
        "afirma, e o item continua a dizer `verbatim`."
    )
    citados = sum(len(linha.strip().lstrip(">").strip()) for linha in _linhas_citadas(corpo))
    assert citados == CARACTERES_CITADOS[item], (
        f"a citação de `[{item}]` tem {citados} caracteres citados e o medido é "
        f"{CARACTERES_CITADOS[item]}. Texto `verbatim` que muda exige extração NOVA da fonte, "
        "com data de acesso nova — não edição de prosa. Se a fonte mudou, remeça e atualize "
        "este número junto com a data no cabeçalho do documento."
    )


@pytest.mark.parametrize("item", sorted(DERIVADOS))
def test_derivado_nao_carrega_citacao(item: str) -> None:
    """A direção oposta, e ela faltava.

    Um item `derivado` com bloco de citação afirma mais do que o status autoriza: `derivado`
    existe porque as passadas discordaram ou porque o que se conferiu foram números e não uma
    frase. Uma citação ali faria o próximo leitor tratar como transcrição o que não é — e o
    teste que só olha os `verbatim` nunca o veria.
    """
    corpo = _itens()[item]
    assert _blocos_de_citacao(corpo) == 0, (
        f"`[{item}]` está marcado `derivado` e traz {_blocos_de_citacao(corpo)} bloco(s) de "
        "citação. Ou o status sobe a `verbatim` com a passada que o justifique, ou a citação "
        "sai — o status é o que diz ao leitor o que ele pode citar daqui."
    )


def test_declara_o_que_nao_estabelece() -> None:
    """O escopo negativo vai escrito, e aqui é verificável.

    A distância entre «as fórmulas são as mesmas» e «os números são os mesmos» é exatamente
    onde o risco está, e ela fica escrita no documento em vez de subentendida por quem o ler.
    """
    texto = CITACAO.read_text(encoding="utf-8")
    assert "NÃO estabelece" in texto, "a citação não declara o próprio escopo negativo (AR48)"
    assert "Statistics" in texto, (
        "a citação não declara que a seção Statistics do MetaboAnalyst está fora. Sem isso, "
        "«os métodos estão de acordo» lê-se como cobertura da análise inteira."
    )
    assert "formato-metaboanalyst.md" in texto, (
        "a citação não se distingue do documento do FORMATO; sem isso, equivalência de "
        "método e conformidade de formato lêem-se como a mesma afirmação."
    )


def test_a_passada_pela_interface_declara_o_caminho_e_o_que_ficou_fora() -> None:
    """A varredura declara as superfícies que NÃO abre.

    **Este teste INVERTEU-SE em 09/09/2026, e a propriedade que ele protege não mudou.** Ele
    exigia a declaração de que a passada pela interface **não** aconteceu; ela aconteceu, e
    exigir a declaração antiga faria o portão pedir uma afirmação falsa — o teste perfeito sobre
    o alvo errado.

    O que continua a ser exigido é o mesmo de antes: **que a varredura diga o que ficou de
    fora**. Mudou apenas o que está fora. Antes era a tela inteira; agora são as duas telas de
    análise que não abriram e as telas de resultado dos dois módulos de enriquecimento.

    E o CAMINHO é exigido junto, porque é ele que torna a afirmação conferível: a extensão de
    navegador não conectou, e as telas foram alcançadas por sessão HTTP própria. Sem o caminho
    escrito, «a interface foi verificada» não diz a ninguém como repetir a verificação.
    """
    texto = CITACAO.read_text(encoding="utf-8")
    assert "interface" in texto.lower(), "a citação não menciona a passada pela interface"
    assert "ACONTECEU EM 09/09/2026" in texto.upper(), (
        "a citação não declara que a passada pela interface aconteceu, com a data. A data é o "
        "que permite remedir a caducidade sobre uma fonte que muda."
    )
    assert "sessão HTTP própria" in texto, (
        "a citação não declara o CAMINHO pelo qual as telas foram alcançadas. A extensão de "
        "navegador não conectou, e sem o caminho escrito ninguém sabe como repetir a leitura."
    )
    for fora in ("EBAM", "SVM"):
        assert fora in texto, (
            f"a citação não declara que a tela de {fora} ficou FORA. Varredura que não diz o "
            "que não abriu lê-se como cobertura completa."
        )


# ---------------------------------------------------------------------------
# A tabela-resumo, e ela era a região que NENHUMA asserção lia
# ---------------------------------------------------------------------------

_LINHA_DA_TABELA = re.compile(
    r"^\|\s*`\[(?P<item>N-\d+)\]`\s*\|\s*`(?P<status>[^`]+)`\s*\|\s*(?P<passadas>[^|]+)\|",
    re.MULTILINE,
)


def _tabela_resumo() -> dict[str, tuple[str, str]]:
    """`N-k` -> `(status, passadas)` como a tabela do topo os declara."""
    texto = CITACAO.read_text(encoding="utf-8")
    return {
        m.group("item"): (m.group("status"), m.group("passadas").strip())
        for m in _LINHA_DA_TABELA.finditer(texto)
    }


def test_a_tabela_resumo_cobre_todos_os_itens() -> None:
    """Âncora de não-vacuidade da comparação abaixo.

    `_tabela_resumo()` devolve um dicionário construído por `finditer`, e um `regex` que
    deixasse de casar devolveria **vazio** — sobre o qual a comparação item a item passaria
    por vacuidade, que é exatamente essa classe.
    """
    tabela = _tabela_resumo()
    assert tuple(sorted(tabela)) == tuple(sorted(ELEMENTOS_ESPERADOS)), (
        f"a tabela-resumo declara {sorted(tabela)} e os itens são {sorted(ELEMENTOS_ESPERADOS)}. "
        "Linha que some da tabela leva com ela a única declaração que o leitor vê primeiro."
    )


@pytest.mark.parametrize("item", ELEMENTOS_ESPERADOS)
def test_a_tabela_resumo_concorda_com_o_corpo_do_item(item: str) -> None:
    """**A região que o portão não lia, e é a que o leitor lê primeiro.**

    O portão pinava `Status` e `Passadas` a partir do CORPO de cada item — `_itens()` começa a
    casar no primeiro `### ` e a tabela vive acima dele. A tabela duplica os dois campos, e
    virar `[N-1]` para `verbatim` ali deixava toda a suíte verde enquanto concedia exactamente
    a autorização que o `derivado` existe para negar.

    A duplicação fica: a tabela é o resumo que se lê primeiro, e removê-la custaria mais ao
    leitor do que custa esta asserção. O que muda é ela deixar de ser afirmação sem verificação.
    """
    status_na_tabela, passadas_na_tabela = _tabela_resumo()[item]
    corpo = _itens()[item]

    assert status_na_tabela == _status(corpo), (
        f"`[{item}]` diz `{status_na_tabela}` na tabela e `{_status(corpo)}` no corpo. "
        "As duas declarações são do mesmo campo, e a tabela é a que se lê primeiro."
    )
    assert passadas_na_tabela.startswith(_PASSADAS[item]), (
        f"`[{item}]` diz {passadas_na_tabela!r} na tabela e o medido é {_PASSADAS[item]!r}. "
        "Inflar a contagem aqui é a mesma afirmação sem medição que o corpo já não permite."
    )


def test_a_contagem_de_itens_sem_duas_passadas_e_derivada_e_nao_escrita() -> None:
    """O número que o documento cita sobre si mesmo tem de vir da fonte.

    O texto dizia *"cinco dos sete itens declaram `uma` passada ou status `derivado`"*, e a
    união real era **quatro**. Um documento cujo propósito é procedência a errar uma contagem
    sobre si mesmo é o pior lugar possível para esse defeito, e derivá-la aqui é o que impede a
    próxima edição de a repetir.

    **O PREDICADO mudou em 09/09/2026 e a propriedade não.** Ele era `p != "duas"`, porque duas
    era o máximo que qualquer item tinha. Com a terceira passada, `!= "duas"` passou a contar
    `três` como deficiência — media o oposto do que o nome diz. A medida é **incompletude**, e
    três passadas não são incompletas: o predicado passa a ser `== "uma"`.

    Isto aconteceu neste próprio módulo: o portão reprovou, e o veredito era sobre o
    instrumento e não sobre o documento. A resposta não foi afrouxar — foi perguntar o que a
    forma protegia, e a resposta é *o número que o documento cita sobre si mesmo vem da fonte*.
    """
    sem_duas_passadas = {item for item, p in _PASSADAS.items() if p == "uma"} | DERIVADOS
    quantos = len(sem_duas_passadas)
    assert quantos == 1, f"a união mudou para {quantos}: {sorted(sem_duas_passadas)}"

    numero_por_extenso = {1: "um", 2: "dois", 3: "três", 4: "quatro", 5: "cinco"}[quantos]
    # Comparação sem caixa: o número pode abrir uma frase, e exigir minúscula faria o portão
    # reprovar por ortografia correta. A propriedade é o número aparecer, não como está grafado.
    texto = CITACAO.read_text(encoding="utf-8").lower()
    assert f"{numero_por_extenso} dos sete itens" in texto, (
        f"o documento não diz «{numero_por_extenso} dos sete itens», e {quantos} é o que "
        "`_PASSADAS` e `DERIVADOS` medem. O número que ele cita sobre si mesmo tem de bater."
    )
