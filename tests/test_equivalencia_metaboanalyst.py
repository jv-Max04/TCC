"""O rótulo do MetaboAnalyst que vai para a TELA é conferido contra a transcrição.

## Por que este módulo existe, e a superfície é o que o torna caro

`EQUIVALENCIA_NO_METABOANALYST` põe, sob cada controle da janela, o nome que aquele mesmo
controle tem na tela de `metaboanalyst.ca`. O produto inteiro desse rótulo é **reconhecimento**:
o profissional lê `Sample-specific normalization (i.e. weight, volume)` e sabe na hora o que o
controle faz, porque já clicou nele.

Um rótulo inventado destrói exactamente isso, e de forma **pior que a ausência dele**: quem o
lê e não reconhece conclui que usa outra versão, ou que não se lembra — nunca que a ferramenta
mentiu. É procedência inventada na superfície que o usuário lê, e a assimetria
aplica-se inteira: nome sem procedência **pede** conferência, nome com procedência
inventada **dispensa**-a.

**Então a conferência é mecânica.** `docs/metodos-metaboanalyst.md` é o documento que carrega a
fonte, a data de acesso e o método — e é ele o oráculo. Rótulo que não exista lá reprova.

## A NORMALIZAÇÃO DE ESPAÇO NÃO É DETALHE — é a armadilha

A transcrição quebra linha por volta dos 87 caracteres, e as citações vivem em bloco `>`. O
rótulo `Square root transformation (square root of data values)` está no arquivo assim:

    > `Log transformation (base 10)` · `(base 2)` · `Square root transformation (square root of data
    > values)` · `Cube root transformation (cube root of data values)` · ...

Uma comparação de substring crua sobre o arquivo devolve **zero** para esse rótulo, e o veredito
seria *"a fonte não contém isto"* — que é a direção perigosa: agir sobre ele apagaria uma citação
correta, que é o modo. Por isso `_normaliza` tira o prefixo de citação e colapsa espaço
**dos dois lados** antes de comparar, que é o que `scripts/verifica_citacao.py::_normaliza` faz
pelo mesmo motivo, escrito na docstring dele.

## O que este módulo NÃO alcança, e a lacuna vai nomeada

Ele prova que o rótulo está **na transcrição**. Não prova que a transcrição é fiel à tela do
MetaboAnalyst — isso é procedência declarada, com data de acesso, e não verificada, exactamente
como `test_metaboanalyst_metodos_citacao.py` declara para o documento inteiro. Buscar a página em
teste faria o portão depender de rede e reprovar quando o site caísse, e a tela **caduca** de
todo modo, que é a razão de a data de acesso existir.
"""

from __future__ import annotations

import re
from typing import get_args

import pytest

from refino.domain.labels import (
    Imputation,
    RankingFilter,
    SampleNormalization,
    Scaling,
    Transformation,
)
from refino.interface.fluxo import (
    CORTE_SUGERIDO_NO_METABOANALYST,
    EIXOS_DA_GRADE,
    EQUIVALENCIA_NO_METABOANALYST,
    SEM_CONTRAPARTE_TRANSCRITA,
    VALOR_DE_REPOUSO,
    NoMetaboAnalyst,
    equivalencia_do_eixo,
    prosa_da_equivalencia,
)
from tests.conftest import REPO

TRANSCRICAO = REPO / "docs" / "metodos-metaboanalyst.md"

# O vocabulário de cada eixo, do DOMÍNIO e não escrito aqui. Escrever a lista tornaria o portão
# cego a rótulo novo: acrescentar um valor ao `Literal` passaria sem contraparte e sem acusar,
# que é o defeito que este módulo existe para não ter.
VOCABULARIO_POR_EIXO: dict[str, tuple[str, ...]] = {
    "ranking_filter": get_args(RankingFilter),
    "ranking_filter_2": get_args(RankingFilter),
    "imputation": get_args(Imputation),
    "sample_normalization": get_args(SampleNormalization),
    "transformation": get_args(Transformation),
    "scaling": get_args(Scaling),
}


def _normaliza(texto: str) -> str:
    """Tira o prefixo de citação e colapsa espaço, para a comparação atravessar quebra de linha.

    As duas operações são necessárias e a ordem importa: retirar o `> ` **por linha**, antes de
    colapsar, senão o marcador fica no meio da frase colada e a comparação falha por ele.
    """
    sem_citacao = "\n".join(re.sub(r"^\s*>\s?", "", linha) for linha in texto.splitlines())
    return " ".join(sem_citacao.split())


def _transcricao_normalizada() -> str:
    bruto = TRANSCRICAO.read_text(encoding="utf-8")
    # **Âncora de não-vacuidade, e ela vai AQUI e não num teste vizinho.** Toda asserção
    # abaixo é «este rótulo está contido no documento»; com o documento vazio — arquivo movido,
    # leitura falhada, caminho errado — a contenção falha e o portão fica vermelho, que é o lado
    # bom. Mas a asserção INVERSA, a de que um rótulo inventado NÃO está, passaria por vacuidade.
    assert len(bruto) > 10_000, (
        f"a transcrição tem {len(bruto)} bytes, o que não é um documento de sete elementos. "
        "Sem ela, a asserção de que um rótulo inventado está ausente passa por vacuidade."
    )
    return _normaliza(bruto)


def _todas_as_entradas() -> list[tuple[str, str, NoMetaboAnalyst]]:
    return [
        (eixo, valor, equivalente)
        for eixo, por_valor in EQUIVALENCIA_NO_METABOANALYST.items()
        for valor, equivalente in por_valor.items()
    ]


ENTRADAS = _todas_as_entradas()


def test_ha_entradas_a_conferir() -> None:
    """A âncora de não-vacuidade da parametrização — sem ela, os testes abaixo somem em silêncio.

    `pytest.mark.parametrize` sobre lista vazia **não reprova**: colhe zero casos e a suíte fica
    verde com o portão inteiro inerte. É a forma mais barata do portão inerte, e o número
    explícito é o que a fecha — derivar a expectativa da mesma lista faria a contagem concordar
    consigo mesma.
    """
    assert len(ENTRADAS) >= 30, (
        f"só {len(ENTRADAS)} entradas de equivalência. Eram 39 em 16/09/2026 — seis eixos, e a "
        "normalização por amostra sozinha traz oito. Uma queda desta ordem é eixo que sumiu do "
        "dicionário, e a parametrização abaixo encolheria com ele sem nada acusar."
    )


@pytest.mark.parametrize(
    ("eixo", "valor", "equivalente"),
    ENTRADAS,
    ids=[f"{eixo}:{valor}" for eixo, valor, _ in ENTRADAS],
)
def test_todo_rotulo_existe_na_transcricao(
    eixo: str, valor: str, equivalente: NoMetaboAnalyst
) -> None:
    """O portão da procedência: o rótulo que vai para a tela está no documento que o cita.

    **A conferência exige o rótulo ENTRE CRASES, e a substring crua foi MEDIDA e recusada** —
    16/09/2026, na revisão desta unidade. A transcrição escreve todo rótulo de opção entre crases,
    porque é assim que ela os distingue da prosa à volta; conferir a substring nua faz o portão
    ficar quase vácuo nos rótulos curtos, que são justamente os que alguém escreveria de memória:

    | Rótulo | substring crua | entre crases |
    |---|---|---|
    | `min` | **29** | 3 |
    | `mean` | 20 | 2 |
    | `median` | 12 | 2 |
    | `None` | 1 | 1 |

    **29 ocorrências de `min` é o portão a dizer «sim» à palavra «mínimo», «minimum» e «min.»** —
    e um rótulo inventado de três letras passaria pelo mesmo caminho. Medido antes de mudar: os
    **30** rótulos aparecem entre crases, então a forma estrita não perde nenhum verdadeiro.
    """
    documento = _transcricao_normalizada()
    assert f"`{_normaliza(equivalente.rotulo)}`" in documento, (
        f"`{eixo}.{valor}` diz que o MetaboAnalyst chama a isto "
        f"`{equivalente.rotulo}`, e esse rótulo NÃO está em {TRANSCRICAO.name}.\n\n"
        "Um nome do MetaboAnalyst que não esteja na transcrição é procedência inventada na "
        "superfície que o usuário lê — e o conserto NÃO é acrescentá-lo ao documento de "
        "memória: é medir a tela deles outra vez, com data de acesso, como o documento manda."
    )


@pytest.mark.parametrize(
    ("eixo", "valor", "equivalente"),
    [(e, v, q) for e, v, q in ENTRADAS if q.grupo],
    ids=[f"{eixo}:{valor}" for eixo, valor, q in ENTRADAS if q.grupo],
)
def test_todo_grupo_existe_na_transcricao(
    eixo: str, valor: str, equivalente: NoMetaboAnalyst
) -> None:
    """O grupo é metade do sentido do rótulo, então tem a mesma exigência de procedência.

    O `log2` é o caso que o mostra: o rótulo dele é `(base 2)`, e sem o grupo
    *Data transformation* ao lado ele não diz nada. Um grupo inventado seria o mesmo defeito do
    rótulo inventado, num campo que parece decoração.
    """
    documento = _transcricao_normalizada()
    assert _normaliza(equivalente.grupo) in documento, (
        f"`{eixo}.{valor}` põe a opção no grupo `{equivalente.grupo}`, que não está em "
        f"{TRANSCRICAO.name}."
    )


def test_o_portao_acusa_o_rotulo_plausivel_que_eu_nao_escrevi() -> None:
    """O par causal, e a planta é o erro que eu QUASE cometi — a segunda pergunta.

    Exercitar nas duas direções prova que o portão consegue acusar. **Não prova que ele acusa a
    coisa certa**, e o modo de falha é sistemático: o oráculo nasce depois de eu consertar as
    instâncias que vi, então é construído a partir das que já não existem e fica cego às que
    sobraram.

    Então a planta não é uma string absurda — `xxx` reprovaria e não provaria nada. É
    `Log transformation (base 2)`: a completação **natural** do rótulo do `log2`, que na tela
    deles é só `(base 2)` porque herda o começo da frase da opção de cima. Escrevê-la assim é o
    que qualquer autor cuidadoso faria para o rótulo ficar apresentável, e é inventar
    transcrição.

    **E a direção oposta vai junto**, porque ver o portão reprovar não é o mesmo que vê-lo
    deixar de acusar: o rótulo verdadeiro, no mesmo teste, tem de passar. Sem esse braço, um
    oráculo que reprovasse tudo pareceria igualmente saudável.
    """
    documento = _transcricao_normalizada()

    plantado = "Log transformation (base 2)"
    assert _normaliza(plantado) not in documento, (
        f"a planta `{plantado}` FOI achada na transcrição, então ela não separa nada e este "
        "teste passaria por coincidência. A planta tem de cair sobre dado em que as duas coisas "
        "diferem."
    )

    verdadeiro = equivalencia_do_eixo("transformation", "log2")
    assert verdadeiro is not None
    assert verdadeiro.rotulo == "(base 2)", (
        "o rótulo do `log2` deixou de ser o que a tela deles traz. Se a transcrição mudou, o "
        "conserto é aqui; se foi 'arrumado' para ficar apresentável, é a planta deste teste."
    )
    assert _normaliza(verdadeiro.rotulo) in documento


def test_a_particao_e_exaustiva__todo_valor_tem_um_dos_dois_desfechos() -> None:
    """Todo valor de todo vocabulário está num dos dois baldes — a direção que acha a LACUNA.

    Um terceiro balde implícito — «ainda não olhei» — seria residual, e residual lê-se como
    pronto. Esta direção é a que dispara quando alguém acrescenta um rótulo ao `Literal`: ele
    nasce sem contraparte e sem motivo, e o portão diz qual.
    """
    faltam: list[str] = []
    for eixo, vocabulario in VOCABULARIO_POR_EIXO.items():
        com_rotulo = set(EQUIVALENCIA_NO_METABOANALYST.get(eixo, {}))
        declarados = set(SEM_CONTRAPARTE_TRANSCRITA.get(eixo, {}))
        faltam += [f"{eixo}.{v}" for v in vocabulario if v not in com_rotulo | declarados]

    assert not faltam, (
        f"{len(faltam)} valor(es) de vocabulário sem contraparte E sem motivo declarado: "
        f"{faltam}.\n\nOu o MetaboAnalyst tem um rótulo para ele — e vai em "
        "`EQUIVALENCIA_NO_METABOANALYST`, conferido contra a transcrição —, ou não tem, e o "
        "motivo vai em `SEM_CONTRAPARTE_TRANSCRITA`. Silêncio não é nenhum dos dois."
    )


def test_a_particao_e_disjunta__nenhum_valor_nos_dois_baldes() -> None:
    """A direção OPOSTA, e é ela que não tem sintoma próprio.

    Um valor nos dois baldes lê `EQUIVALENCIA_NO_METABOANALYST` primeiro e o motivo fica morto —
    o texto continua no arquivo, continua a ser lido por quem edita, e descreve um estado que não
    é o do produto. Nada acusa, e é um modo de falha silencioso.
    """
    nos_dois = [
        f"{eixo}.{valor}"
        for eixo in VOCABULARIO_POR_EIXO
        for valor in set(EQUIVALENCIA_NO_METABOANALYST.get(eixo, {}))
        & set(SEM_CONTRAPARTE_TRANSCRITA.get(eixo, {}))
    ]
    assert not nos_dois, (
        f"valor(es) nos DOIS baldes: {nos_dois}. O rótulo vence e o motivo fica morto no arquivo, "
        "a descrever um estado que não é o do produto."
    )


def test_nenhum_balde_nomeia_valor_que_o_vocabulario_nao_tem() -> None:
    """A terceira direção: entrada órfã, que é o rótulo que sobrou de uma remoção.

    Sem ela, retirar um valor do `Literal` deixa a contraparte para trás — e ela continua a
    passar no portão da procedência, porque a transcrição não mudou. Portão verde sobre um
    rótulo que nada oferece.
    """
    orfas: list[str] = []
    for eixo, vocabulario in VOCABULARIO_POR_EIXO.items():
        conhecidos = set(vocabulario)
        orfas += [
            f"{eixo}.{v}"
            for v in set(EQUIVALENCIA_NO_METABOANALYST.get(eixo, {}))
            | set(SEM_CONTRAPARTE_TRANSCRITA.get(eixo, {}))
            if v not in conhecidos
        ]
    assert not orfas, (
        f"contraparte declarada para valor que o vocabulário não tem: {orfas}. "
        "Ou o valor saiu do `Literal` e a entrada ficou, ou a chave está mal escrita."
    )


def test_os_eixos_sao_exatamente_os_da_constante_da_grade() -> None:
    """As chaves de fora são as da constante que desenha os controles, nas duas direções.

    É o que impede a contraparte de existir para um eixo que a tela não mostra, e a tela de
    mostrar um eixo sem contraparte nenhuma.
    """
    da_grade = {chave for chave, _r, _p, _o in EIXOS_DA_GRADE}
    assert set(EQUIVALENCIA_NO_METABOANALYST) == da_grade
    assert set(SEM_CONTRAPARTE_TRANSCRITA) == da_grade
    assert set(VOCABULARIO_POR_EIXO) == da_grade


@pytest.mark.parametrize(
    ("eixo", "valor"),
    [
        (eixo, v)
        for eixo, vocab in VOCABULARIO_POR_EIXO.items()
        for v in vocab
        if v != VALOR_DE_REPOUSO
    ],
)
def test_todo_valor_falante_produz_frase(eixo: str, valor: str) -> None:
    """Fora os calados, todo valor tem frase — e o branco seria defeito.

    Um rótulo em branco sob um eixo cujos vizinhos têm texto não diz «não há»; diz «não
    carregou». As duas pedem ações opostas de quem lê, e é por isso que `vast` e `exclude`
    falam: são métodos de verdade, e quem já usa o MetaboAnalyst procura o equivalente deles.
    """
    frase = prosa_da_equivalencia(eixo, valor)
    assert frase.strip(), f"`{eixo}.{valor}` não produziu frase nenhuma"
    assert frase.startswith("No MetaboAnalyst:")


def test_o_repouso_cala_e_a_aba_abre_quieta() -> None:
    """A direção oposta da anterior, e ela é sobre o ARTEFATO MONTADO.

    A pergunta *«quantas linhas a aba mostra ao ABRIR?»* só a captura respondia, e foi ela que
    achou o defeito de 16/09/2026: cinco frases de «não precisa de tradução» a empurrar o
    conteúdo para baixo. Aqui ela vira número, e passa a reprovar sem ninguém olhar.

    **A primeira asserção é o par causal do silêncio.** Sem ela, uma regra que calasse DEMAIS —
    o método inteiro, e não só o repouso — deixaria a suíte verde com a aba vazia, que é
    exactamente o desfecho indistinguível de sucesso. Ela afirma que os valores que
    NÃO são repouso continuam a falar, sobre os dois baldes.
    """
    falam = [
        f"{eixo}.{valor}"
        for eixo, vocabulario in VOCABULARIO_POR_EIXO.items()
        for valor in vocabulario
        if valor != VALOR_DE_REPOUSO and prosa_da_equivalencia(eixo, valor)
    ]
    esperados = sum(
        1 for vocabulario in VOCABULARIO_POR_EIXO.values() for v in vocabulario if v != "none"
    )
    assert len(falam) == esperados, (
        f"{len(falam)} de {esperados} valores não-repouso produzem linha. Os que faltam calaram "
        "sem declaração — e um método calado some da tela sem nada acusar."
    )

    # **E o repouso cala vindo de QUALQUER balde**, que é o que a regra por valor compra sobre o
    # campo por entrada: o `none` da normalização por amostra TEM contraparte transcrita — `None`
    # é um dos oito rótulos deles — e mesmo assim não fala.
    assert equivalencia_do_eixo("sample_normalization", VALOR_DE_REPOUSO) is not None
    calam = [
        chave
        for chave, _rotulo, _padrao, opcoes in EIXOS_DA_GRADE
        if VALOR_DE_REPOUSO in opcoes and prosa_da_equivalencia(chave, VALOR_DE_REPOUSO)
    ]
    assert not calam, f"o repouso destes eixos ainda fala: {calam}"

    com_linha = [
        chave
        for chave, _rotulo, padrao, _opcoes in EIXOS_DA_GRADE
        if prosa_da_equivalencia(chave, padrao)
    ]
    assert com_linha == ["imputation"], (
        f"ao abrir, a aba mostraria linha de equivalência em {com_linha}. Cinco dos seis eixos "
        "repousam em `none`, e o desenho é que a aba abra quieta: a linha aparece quando a "
        "pessoa escolhe um método."
    )


def test_difere_sem_nota_e_recusado_na_construcao() -> None:
    """A guarda da classe, exercitada nas duas direções — mecanismo novo não nasce inerte.

    `conta='difere'` sem `nota` é a promessa de equivalência que a classe existe para não fazer.
    """
    with pytest.raises(ValueError, match="não diz o que difere"):
        NoMetaboAnalyst(tela="t", grupo="g", rotulo="r", conta="difere", nota="   ")

    # A direção oposta: com nota, constrói. Sem este braço, uma guarda que recusasse TUDO
    # pareceria igualmente saudável.
    assert NoMetaboAnalyst(tela="t", grupo="g", rotulo="r", conta="difere", nota="x").nota == "x"
    assert NoMetaboAnalyst(tela="t", grupo="g", rotulo="r", conta="identica").nota == ""


def test_toda_divergencia_declarada_diz_o_que_difere() -> None:
    """Sobre as entradas REAIS, e não sobre um objeto construído no teste.

    O teste acima prova que a guarda funciona; este prova que ela foi **aplicada** às entradas
    que vão para a tela. São perguntas diferentes: a primeira é sobre o mecanismo, a segunda
    sobre o dado — e a segunda é a que pega uma entrada que passasse a `difere` por outro
    caminho.
    """
    divergentes = [(e, v, q) for e, v, q in ENTRADAS if q.conta == "difere"]
    assert divergentes, (
        "nenhuma entrada declara `difere`, e três declaravam em 16/09/2026 — os dois KNN e o "
        "missForest, mais `auto` e `pareto` pelo denominador do desvio. Zero aqui faz este teste "
        "passar por vacuidade."
    )
    for eixo, valor, equivalente in divergentes:
        assert equivalente.nota.strip(), f"`{eixo}.{valor}` diz `difere` e não diz o quê"


def test_o_corte_sugerido_na_tela_deles_existe_na_transcricao() -> None:
    """A citação que a aba de QUALIDADE mostra tem a mesma exigência que os rótulos da de preparo.

    **Por que ela entra neste módulo e não no dos limiares.** O gênero é o mesmo das entradas
    acima — uma frase atribuída ao MetaboAnalyst, numa tela que o usuário lê — e o defeito que
    a ameaça é o mesmo: uma citação plausível escrita de memória. O que muda é só o eixo em que
    ela aparece, e separá-la daqui daria dois portões com o mesmo oráculo, a divergir.

    A âncora primária daquela legenda continua a ser Broadhurst et al. (2018), e
    `tests/test_ui_fluxo.py` é quem a guarda. Esta é a segunda confirmação — a que o profissional
    reconhece por a ter visto na tela.
    """
    documento = _transcricao_normalizada()
    assert _normaliza(CORTE_SUGERIDO_NO_METABOANALYST) in documento, (
        f"a legenda do corte de RSD atribui ao MetaboAnalyst a frase "
        f"«{CORTE_SUGERIDO_NO_METABOANALYST}», que NÃO está em {TRANSCRICAO.name}."
    )

    # **A direção que prova que a asserção acima consegue falhar**, sobre uma frase do mesmo
    # gênero e igualmente plausível: o MetaboAnalyst separa LC-MS de GC-MS, e inverter os dois é
    # o erro que alguém cometeria a citar de cabeça.
    invertido = "suggested threshold: 30% for LC-MS and 20% for GC-MS"
    assert _normaliza(invertido) not in documento, (
        "a frase invertida foi achada na transcrição, então esta asserção não separa nada."
    )


def test_so_o_missforest_dispensa_o_grupo() -> None:
    """A porta que o filtro `if q.grupo` deixava aberta — achado da revisão de 16/09/2026.

    `test_todo_grupo_existe_na_transcricao` salta as entradas de grupo vazio, porque não há o que
    conferir nelas. **Isso faz do grupo vazio um caminho para dribar a conferência:** uma entrada
    nova com `grupo=""` passaria os dois portões sem que o grupo dela fosse comparado com coisa
    nenhuma, e o desfecho seria verde.

    A entrada única que legitimamente o dispensa é o `missForest`: a transcrição achou-o no TEXTO
    da tela `ProcMissing.xhtml` e não entre os dez botões de rádio, então pôr-lhe um dos três
    nomes de grupo seria inventar onde ele aparece. Vazio é o que é verdade — e fixá-lo aqui é o
    que impede a exceção de virar porta.
    """
    sem_grupo = {f"{eixo}.{valor}" for eixo, valor, q in ENTRADAS if not q.grupo}
    assert sem_grupo == {"imputation.missforest"}, (
        f"entradas sem grupo: {sorted(sem_grupo)}. Grupo vazio salta a conferência de "
        "procedência, então só se admite onde a transcrição genuinamente não o dá — e isso é "
        "uma entrada, não um padrão."
    )


def test_a_crase_e_o_que_torna_o_portao_estreito() -> None:
    """O par causal da forma exigida — sem ele, apertar o oráculo seria mudança sem evidência.

    **A planta é um rótulo de três letras que NÃO está na transcrição como opção**, mas cuja
    substring crua está lá dezenas de vezes dentro de outras palavras. É o que o portão antigo
    deixava passar, e é a classe inteira: rótulo curto escrito de memória.
    """
    documento = _transcricao_normalizada()

    plantado = "min"
    assert _normaliza(plantado) in documento, (
        "a planta tem de ser uma substring que EXISTE crua, senão o teste não separa a forma "
        "estrita da frouxa — separaria só «existe» de «não existe»."
    )
    assert f"`{plantado}`" in documento, (
        "`min` É rótulo verdadeiro do grupo *Univariate statistical methods*, então a forma "
        "estrita tem de o aceitar. Sem este braço, o teste provaria apenas que a crase recusa."
    )

    # E a direção que importa: um rótulo curto INVENTADO passa cru e cai entre crases.
    inventado = "value"
    assert _normaliza(inventado) in documento, (
        f"«{inventado}» tem de existir cru para a comparação abaixo significar alguma coisa"
    )
    assert f"`{inventado}`" not in documento, (
        f"«{inventado}» é rótulo de verdade na transcrição, então não serve de planta. Troque-a "
        "por outra palavra que apareça DENTRO de rótulos e nunca como rótulo."
    )
