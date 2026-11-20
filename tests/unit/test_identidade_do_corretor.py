"""A IDENTIDADE de um corretor, e o portão que a mantém exaustiva.

## O que esta unidade protege

O reuso de correção entre `corrigir`, `processar` e `exportar` deduplica por uma chave, e a metade
dela que vem do **construtor** é `Corrector.identidade()`. Um eixo de construtor que mude valores e
não apareça ali produz o defeito mais silencioso desta família: a matriz do outro parâmetro,
plausível, com o rótulo certo.

**Este projeto já mediu esse defeito duas vezes**, e as duas estão citadas nos testes abaixo. O que
faltava era o mecanismo que impede a terceira.

## Por que um portão por ASSINATURA e não uma lista

Uma lista de eixos escrita à mão é a segunda fonte de verdade sobre o construtor, e diverge no dia
em que alguém acrescenta um parâmetro — em silêncio, porque nada liga as duas. `inspect.signature`
lê o construtor **real**, então a lista não pode ficar para trás: ela é derivada dele.

É a paridade entre o registro de corretores e `get_args(Correction)` — aplicada a
construtores em vez de a rótulos.
"""

from __future__ import annotations

import inspect
from typing import Any

import pytest

from refino.correction.loess import LoessCorrector
from refino.correction.none import NoneCorrector
from refino.correction.serrf import SerrfCorrector

CORRETORES: tuple[Any, ...] = (NoneCorrector, LoessCorrector, SerrfCorrector)

# **Os eixos que NÃO entram na identidade, cada um com o teste que autoriza a exclusão.**
#
# Tabela e não conjunto: o que a torna honesta é a coluna da direita. Excluir um eixo é afirmar que
# ele **não muda valor nenhum**, e essa afirmação tem de ter dono — senão a tabela vira lista de
# isenção com outro nome.
#
# **Medido antes de escrever:** com a tabela vazia o portão acusa TRÊS
# eixos, todos do SERRF; com estas três linhas acusa zero. Três de três é o número certo — uma
# tabela que isentasse metade dos eixos de um corretor seria o sintoma de o oráculo estar errado.
FORA_DA_IDENTIDADE: dict[str, str] = {
    "SerrfCorrector.n_jobs": (
        "bit-idêntico entre 1 e -1, por fluxo de semente derivado de (seed, feature, lote, passo). "
        "O par causal é `test_a_saida_e_bit_identica_serial_e_paralela`, em "
        "`tests/unit/test_forest_speedup.py`, e ele compara valores, `uncorrected` E `provenance` "
        "— MAS a evidência é condicional a `effective_n_jobs(-1) > 1`: sob `LOKY_MAX_CPU_COUNT=1` "
        "aquele par causal PULA, e não é produzida. Até 08/09/2026 isso não mordia porque as três "
        "rotas construíam o SERRF num sítio só, sempre com `n_jobs=-1`; com a caixa «usar vários "
        "núcleos» elas passaram a construir 1 OU -1, e a exclusão deixou de ser inerte. A "
        "evidência NÃO é produzida no regime de um trabalhador, e isso vai declarado: o que "
        "`test_o_skip_do_par_causal_cai_onde_os_dois_regimes_sao_o_mesmo` afirma é a coincidência "
        "de predicados — onde a prova falta, os dois valores da caixa dão o mesmo trabalhador "
        "— o último é o único sinal que denunciaria floresta diferente sem mudar um número"
    ),
    "SerrfCorrector.hoist_correlations": (
        "içar a correlação do passo 2 é a mesma função com os mesmos argumentos, calculada uma vez "
        "em vez de N. Par causal em `tests/unit/test_forest_speedup.py`"
    ),
    "SerrfCorrector.fast_param_names": (
        "cache do `_get_param_names` do scikit-learn: overhead e nada mais. Par causal em "
        "`tests/unit/test_forest_speedup.py`"
    ),
    # **A quarta linha entrou depois, e a contagem foi refeita antes de a escrever:** com a tabela
    # vazia o portão acusa QUATRO eixos — os três do SERRF e este —, e com as quatro linhas, zero.
    "LoessCorrector.n_jobs": (
        "bit-idêntico entre 1 e 2 núcleos, por fluxo de semente derivado de (seed, índice da "
        "feature) e nada mais partilhado entre features. O par causal é "
        "`test_dois_nucleos_dao_a_matriz_da_serie_bit_a_bit_num_modo_real`, em "
        "`tests/unit/test_loess_corrector.py`, sobre 300 features do `pos_hilic` real: valores por "
        "bytes, `uncorrected` e `provenance`. Ele NÃO pula com um núcleo — `n_jobs=2` são dois "
        "trabalhadores em qualquer máquina —, e o espião do mesmo arquivo afirma que o braço de "
        "dois núcleos corre fora do processo do pai"
    ),
}


def _eixos_do_construtor(classe: Any) -> tuple[str, ...]:
    """Os parâmetros nomeados do `__init__`, sem `self`. Vazio quando não há `__init__` próprio."""
    if "__init__" not in vars(classe):
        return ()
    return tuple(
        nome
        for nome, parametro in inspect.signature(classe.__init__).parameters.items()
        if nome != "self" and parametro.kind is not inspect.Parameter.VAR_KEYWORD
    )


def _esquecidos(classes: list[Any], tabela: dict[str, str]) -> list[str]:
    """Os eixos de construtor que a identidade não menciona e a tabela não isenta.

    **Extraída para que a PLANTA caia sobre ela.** Enquanto esta lógica vivia dentro do
    teste do portão, o negativo afirmava propriedades **da planta** — que ela tem o eixo, que a
    identidade não o menciona — e nunca corria o portão sobre ela. Apagar o portão inteiro teria
    deixado aquele negativo verde, que é a definição de verificação inerte.

    **A `tabela` é PARÂMETRO e não a constante do módulo**, e é isso que torna o par causal
    possível: a mesma função, a mesma planta, e só a isenção muda entre as duas chamadas.
    """
    fora: list[str] = []
    for classe in classes:
        identidade = classe().identidade()
        for eixo in _eixos_do_construtor(classe):
            if f"{eixo}=" in identidade:
                continue
            if f"{classe.__name__}.{eixo}" in tabela:
                continue
            fora.append(f"{classe.__name__}.{eixo}")
    return fora


def test_todo_eixo_de_construtor_esta_na_identidade_ou_declarado_fora() -> None:
    """Nas DUAS direções: eixo esquecido reprova, e isenção obsoleta também.

    **A segunda direção é a que ninguém escreve, e é a que apanha o lixo.** Um eixo removido do
    construtor deixa a linha na tabela a isentar coisa nenhuma, e uma tabela com entradas mortas
    perde a autoridade de dizer que as vivas foram pensadas.
    """
    assert CORRETORES, "âncora: não há corretor a medir e o laço não correria"

    for classe in CORRETORES:
        identidade = classe().identidade()
        assert identidade.startswith(f"{classe.name}("), (
            f"a identidade de `{classe.__name__}` não começa pelo nome do corretor: "
            f"{identidade!r}. Sem o nome, dois corretores sem eixos colidem"
        )

    esquecidos = _esquecidos(CORRETORES, FORA_DA_IDENTIDADE)
    assert not esquecidos, (
        f"estes eixos de construtor não entram na identidade e não estão declarados fora: "
        f"{esquecidos}. Ou eles mudam valor — e a chave de reuso serve a matriz errada —, ou não "
        f"mudam, e a exclusão precisa do teste que a autorize"
    )

    reais = {
        f"{classe.__name__}.{eixo}"
        for classe in CORRETORES
        for eixo in _eixos_do_construtor(classe)
    }
    obsoletos = sorted(set(FORA_DA_IDENTIDADE) - reais)
    assert not obsoletos, (
        f"a tabela isenta eixos que já não existem: {obsoletos}. Isenção morta faz a tabela deixar "
        f"de provar que as vivas foram pensadas"
    )


def test_o_portao_acusa_um_eixo_plantado_fora_da_identidade() -> None:
    """A PLANTA, e sem ela o portão acima é indistinguível de um que nunca acusa.

    A planta é um corretor com um eixo de construtor que a identidade não menciona — exactamente o
    estado que o portão existe para recusar. Ele tem de aparecer na lista de esquecidos.
    """

    class _ComEixoEsquecido:
        name = "plantado"

        def __init__(self, *, politica: str = "a") -> None:
            self.politica = politica

        def identidade(self) -> str:
            return f"{self.name}()"

    eixos = _eixos_do_construtor(_ComEixoEsquecido)
    assert eixos == ("politica",), f"a planta não tem o eixo que devia ter: {eixos}"
    assert "politica=" not in _ComEixoEsquecido().identidade(), (
        "âncora: a planta já declara o eixo, e não é a planta que o portão recusa"
    )

    # **O PAR CAUSAL, e ele é o conteúdo deste teste.** Até 06/09/2026 as três linhas acima eram o
    # teste inteiro: afirmavam propriedades da planta e **não corriam o portão sobre ela**. Apagar
    # a detecção inteira teria deixado este negativo verde, na superfície
    # do próprio negativo.
    acusados = _esquecidos([_ComEixoEsquecido], {})
    assert acusados == ["_ComEixoEsquecido.politica"], (
        f"o portão NÃO acusa o eixo plantado: {acusados}. Ele está a aceitar exactamente o estado "
        f"que existe para recusar"
    )

    # A outra direção: com a isenção declarada, o MESMO portão deixa de acusar. Sem esta metade, um
    # portão que acusasse tudo passaria na de cima.
    isento = _esquecidos([_ComEixoEsquecido], {"_ComEixoEsquecido.politica": "declarado no teste"})
    assert isento == [], (
        f"o portão continua a acusar um eixo DECLARADO na tabela: {isento}. A tabela de isenção "
        f"deixou de isentar, e então ela não é a tabela — é ornamento"
    )


def test_a_identidade_separa_o_que_muda_valores_e_ignora_o_que_nao_muda() -> None:
    """O par causal da identidade, nas duas direções, sobre o LOESS e sobre o SERRF.

    **O LOESS é o caso vivo e não um exemplo construído:** as rotas de artefato têm padrões
    diferentes para `null_denominator` — `r_faithful` em `corrigir` e `processar`,
    `uncorrected_feature` em `exportar` —, e a janela não manda o campo a nenhuma delas. Sob
    `loess`, o par que o profissional clica **não partilha correção nenhuma**, e uma identidade que
    ignorasse o eixo serviria uma pela outra.

    **O SERRF é a direção oposta:** `n_jobs` muda o tempo e não o valor, e a identidade tem de o
    ignorar — senão o reuso nunca acerta, e o cache passa a ser custo sem ganho.
    """
    assert (
        LoessCorrector(null_denominator="r_faithful").identidade()
        != LoessCorrector(null_denominator="uncorrected_feature").identidade()
    ), (
        "as duas políticas de denominador dão a mesma identidade: sob `loess`, `corrigir` e "
        "`exportar` receberiam a mesma matriz, e elas correm denominadores diferentes por padrão"
    )

    assert SerrfCorrector(n_jobs=1).identidade() == SerrfCorrector(n_jobs=-1).identidade(), (
        "`n_jobs` entrou na identidade: ele muda o tempo e não o valor, e com ele na chave o reuso "
        "nunca acerta entre uma rota que usa `-1` e outra que usa o padrão"
    )
    # O mesmo para o LOESS: a caixa ticada e a desticada pedem a mesma correção.
    assert LoessCorrector(n_jobs=1).identidade() == LoessCorrector(n_jobs=-1).identidade(), (
        "`n_jobs` do LOESS entrou na identidade: a mesma correção com a caixa ticada e desticada "
        "deixaria de se reaproveitar"
    )

    assert (
        SerrfCorrector(n_correlated=10).identidade() != SerrfCorrector(n_correlated=20).identidade()
    ), "`n_correlated` muda o pool de preditores e tem de separar"

    assert NoneCorrector().identidade() != LoessCorrector().identidade(), (
        "dois corretores diferentes têm a mesma identidade — o nome não está a separar"
    )


@pytest.mark.parametrize("classe", CORRETORES)
def test_a_identidade_e_estavel_entre_chamadas(classe: Any) -> None:
    """Duas chamadas na mesma instância dão a mesma string.

    Parece óbvio e não é: uma identidade que misturasse `id(self)`, um relógio ou a iteração de um
    conjunto passaria por todos os testes acima e faria o reuso **nunca** acertar — cache que
    sempre erra é custo puro, e o sintoma é ausência de ganho, não erro.
    """
    corretor = classe()
    assert corretor.identidade() == corretor.identidade()
    assert classe().identidade() == classe().identidade(), (
        "duas instâncias com os mesmos argumentos dão identidades diferentes"
    )


def test_o_skip_do_par_causal_cai_onde_os_dois_regimes_sao_o_mesmo() -> None:
    """A evidência da exclusão de `n_jobs` falta exactamente onde não é precisa.

    ## O buraco aparente, e por que ele não é buraco

    A exclusão de `SerrfCorrector.n_jobs` de `identidade()` apoia-se num par causal que **pula**
    quando `joblib.effective_n_jobs(-1)` devolve `1`. Até 08/09/2026 isso era declarado inerte
    porque as rotas construíam sempre `-1`; a caixa «usar vários núcleos» acabou com essa premissa,
    e a exclusão passou a ser **carregada**.

    **O que a mantém honesta é uma coincidência de predicados, e é ela que este teste afirma:**

    | `effective_n_jobs(-1)` | o par causal | os dois regimes da caixa |
    |---|---|---|
    | `> 1` | corre | **distintos** — e a prova é produzida |
    | `== 1` | pula | **o mesmo** — `-1` resolve para um trabalhador |

    Quando a prova pula, não há duas florestas a comparar. A afirmação é sobre o `joblib` desta
    instalação e não sobre o nosso código, e é por isso que ela mora aqui e não numa docstring: uma
    versão futura que fizesse `effective_n_jobs(-1)` devolver `1` **sem** que `-1` e `1` corressem
    igual abriria o buraco em silêncio, e este teste reprova nesse dia.

    ## O QUE ESTE TESTE AFIRMA, e o que ele NÃO afirma — corrigido em 08/09/2026

    **A primeira versão era tautológica, e cinco revisores independentes apanharam-na.** A âncora
    fixava `trabalhadores_serie == 1` e a asserção do ramo comparava-o com `trabalhadores_paralelo`,
    que o `if` já tinha fixado em `1`. Era `1 == 1`: um teste escrito para dar evidência a uma
    exclusão que deixou de ser inerte, **e que não podia reprovar**.

    **O que fica é a coincidência de predicados, e ela é verificável.** A âncora não pré-fixa
    nenhum dos dois lados; se uma versão futura do `joblib` fizer `effective_n_jobs(1)` devolver
    outra coisa, esta asserção acusa.

    **E o que este teste NÃO faz vai declarado.** Ele não prova a identidade bit a bit no regime de
    um trabalhador — **ela é inverificável ali**, porque não há duas florestas para comparar. Quem a
    prova é `test_a_saida_e_bit_identica_serial_e_paralela`, e ele **pula** exactamente nesse
    regime. A lacuna existe, é declarada, e o argumento que a torna aceitável é este: onde a prova
    falta, os dois valores da caixa produzem o mesmo número de trabalhadores.
    """
    import joblib

    from refino.desktop.paralelismo import PARALELO, SERIE

    trabalhadores_paralelo = int(joblib.effective_n_jobs(PARALELO))
    trabalhadores_serie = int(joblib.effective_n_jobs(SERIE))

    # **A âncora não pré-fixa nenhum dos dois lados.** Fixar `trabalhadores_serie == 1` aqui é o
    # que tornava a asserção do ramo `1 == 1`.
    assert trabalhadores_paralelo >= 1 and trabalhadores_serie >= 1, (
        f"`effective_n_jobs` devolveu {trabalhadores_paralelo} e {trabalhadores_serie}: o "
        "vocabulário do joblib mudou, e a coincidência de predicados deixa de ser verificável"
    )

    if trabalhadores_paralelo == 1:
        assert trabalhadores_paralelo == trabalhadores_serie, (
            f"`effective_n_jobs(-1)` devolve 1 — o par causal de `test_forest_speedup.py` PULA —, "
            f"e `effective_n_jobs(1)` devolve {trabalhadores_serie}. Os dois valores da caixa "
            "«usar vários núcleos» deixaram de produzir o mesmo número de trabalhadores, e a "
            "exclusão de "
            "`n_jobs` de `identidade()` fica sem evidência num regime em que ela importa: duas "
            "entradas com florestas diferentes podem coexistir na mesma chave"
        )
