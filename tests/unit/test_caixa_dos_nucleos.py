"""A caixa «usar vários núcleos».

## O que esta unidade protege

A caixa é uma **escolha** que atravessa cinco rotas até `SerrfCorrector(n_jobs=…)` e
`LoessCorrector(n_jobs=…)`. O que se pode perder no caminho são três coisas diferentes, e cada
uma tem um teste próprio aqui:

| O que se perde | Sintoma para quem usa |
|---|---|
| a caixa não alcança a rota | ticar e desticar dá o mesmo tempo |
| a **capacidade** não vence a caixa | doze janelas na tela |
| o padrão da rota muda | um cliente que nunca ouviu falar da caixa muda de regime |

**A segunda é a que importa.** `aponta_o_loky` cai para série nos três gatilhos de queda mas **não
desliga o `loky`**: um chamador que pedisse `-1` nesse estado faria o `loky` relançar o
`refino.exe`. A caixa é permissão e não ordem, e é `n_jobs_efetivo` que o garante.
"""

from __future__ import annotations

import ast
import re
from collections.abc import Iterator
from typing import Final, get_args

import pytest

from refino.api.rotas import _corretor_da_spec, _corretor_ou_422
from refino.cli import AVAILABLE_METHODS
from refino.correction.loess import LoessCorrector
from refino.correction.serrf import SerrfCorrector
from refino.decision.labeling import LabelingOracle, build_corrector_for_labeling
from refino.desktop import paralelismo
from refino.desktop.paralelismo import PARALELO, SERIE, n_jobs_efetivo
from refino.domain.labels import Correction
from tests.conftest import REPO


@pytest.fixture
def regime_restaurado() -> Iterator[None]:
    """Devolve o `_REGIME` do módulo ao que era. Ele é global do processo.

    **O irmão deste fixture é `autouse` em `tests/unit/test_paralelismo_do_congelado.py`**, e as
    razões são as mesmas: `desktop/janela/pipeline.py` lê o regime na IMPORTAÇÃO do módulo, então
    um vazamento daqui reprova a dois arquivos de distância.
    """
    anterior = paralelismo._REGIME
    yield
    paralelismo._REGIME = anterior


@pytest.mark.parametrize(
    ("capacidade", "caixa", "esperado"),
    (
        (PARALELO, True, PARALELO),
        (PARALELO, False, SERIE),
        # **As duas linhas que importam: a capacidade VENCE a caixa.** Sem elas, ticar a caixa num
        # processo em que a rede de `aponta_o_loky` caiu manda `-1` ao `loky`, que relança o
        # `refino.exe`.
        (SERIE, True, SERIE),
        (SERIE, False, SERIE),
    ),
)
def test_a_conjuncao_entre_a_caixa_e_a_capacidade(
    capacidade: int, caixa: bool, esperado: int, regime_restaurado: object
) -> None:
    """A tabela verdade inteira, e a terceira linha é a que impede as doze janelas."""
    paralelismo._REGIME = capacidade
    assert n_jobs_efetivo(caixa) == esperado, (
        f"com capacidade {capacidade} e a caixa {'ticada' if caixa else 'desticada'}, o efetivo "
        f"devia ser {esperado}"
    )


def test_a_caixa_alcanca_o_serrf_e_o_loess_das_rotas_e_nao_o_none(
    regime_restaurado: object,
) -> None:
    """`_corretor_ou_422` muda `n_jobs` do SERRF e do LOESS com a caixa, e não toca no `none`.

    O laço por feature do LOESS corre em vários núcleos, com a matriz da série bit a bit. **O
    `none` fica como âncora de não-vacuidade do lado oposto**: sem ele, um `_corretor_ou_422` que
    devolvesse sempre um corretor com `n_jobs` passaria as asserções de cima.
    """
    paralelismo._REGIME = PARALELO

    for nome, classe in (("serrf", SerrfCorrector), ("loess", LoessCorrector)):
        ticada = _corretor_ou_422(nome, "r_faithful", paralelo=True)
        desticada = _corretor_ou_422(nome, "r_faithful", paralelo=False)
        assert isinstance(ticada, classe) and isinstance(desticada, classe), nome
        assert (ticada.n_jobs, desticada.n_jobs) == (PARALELO, SERIE), (
            f"a caixa não alcançou o `{nome}`: {ticada.n_jobs} e {desticada.n_jobs}"
        )

    # O `null_denominator` do pedido continua a chegar ao LOESS: é o eixo que muda valores, e a
    # chamada passou a levar dois argumentos.
    loess = _corretor_ou_422("loess", "uncorrected_feature", paralelo=True)
    assert isinstance(loess, LoessCorrector) and loess.null_denominator == "uncorrected_feature"

    um = _corretor_ou_422("none", "r_faithful", paralelo=True)
    outro = _corretor_ou_422("none", "r_faithful", paralelo=False)
    assert type(um) is type(outro) and not hasattr(um, "n_jobs"), (
        "`none` mudou de classe com a caixa, ou ganhou `n_jobs` — ele não tem laço a paralelizar"
    )


def test_o_padrao_da_rota_e_a_caixa_ticada(regime_restaurado: object) -> None:
    """Quem chama sem o campo recebe o paralelo — o padrão preserva o comportamento antigo."""
    paralelismo._REGIME = PARALELO
    sem_campo = _corretor_ou_422("serrf", "r_faithful")
    assert isinstance(sem_campo, SerrfCorrector) and sem_campo.n_jobs == PARALELO, (
        "omitir `paralelo` deixou de significar `-1`: todo cliente escrito antes da caixa mudou de "
        "regime sem pedir"
    )
    loess = _corretor_ou_422("loess", "r_faithful")
    assert isinstance(loess, LoessCorrector) and loess.n_jobs == PARALELO, (
        "omitir `paralelo` não leva o LOESS aos vários núcleos, e o SERRF da mesma rota vai"
    )


def test_execucoes_construia_o_serrf_em_serie_e_o_padrao_e_a_prova(
    regime_restaurado: object,
) -> None:
    """O defeito fechado em `/execucoes`, com a âncora que o torna medível.

    **`POST /execucoes` construía `AVAILABLE_METHODS[spec.correction]()`**, e o padrão de
    `SerrfCorrector` é `n_jobs=1` — então o botão «Executar pipeline» corria o SERRF em série
    **mesmo fora do congelado**, onde o `loky` sempre funcionou. O valor natural é o que executa
    menos do caminho.

    **A primeira asserção é a âncora e não decoração.** Ela afirma que o padrão do construtor
    continua a ser série; no dia em que ele mudar, a segunda passa por coincidência e este teste
    deixa de medir o que diz medir.
    """
    padrao = AVAILABLE_METHODS["serrf"]()
    assert isinstance(padrao, SerrfCorrector) and padrao.n_jobs == SERIE, (
        "o padrão de `SerrfCorrector` deixou de ser série: a asserção abaixo passa por coincidência"
    )

    paralelismo._REGIME = PARALELO
    corretor = _corretor_da_spec("serrf", paralelo=True)
    assert isinstance(corretor, SerrfCorrector) and corretor.n_jobs == PARALELO, (
        "`/execucoes` voltou a construir o SERRF pelo padrão do construtor, que é série"
    )
    assert type(_corretor_da_spec("loess", paralelo=True)) is type(AVAILABLE_METHODS["loess"]()), (
        "`loess` deixou de nascer da classe de `AVAILABLE_METHODS`: os padrões dele mudaram "
        "sem pedido"
    )

    # O LOESS tem a mesma âncora. O padrão da classe é série; sem a primeira
    # asserção, a de `n_jobs` passaria por coincidência no dia em que ele mudasse.
    padrao_loess = AVAILABLE_METHODS["loess"]()
    assert isinstance(padrao_loess, LoessCorrector) and padrao_loess.n_jobs == SERIE
    ticada = _corretor_da_spec("loess", paralelo=True)
    desticada = _corretor_da_spec("loess", paralelo=False)
    assert isinstance(ticada, LoessCorrector) and isinstance(desticada, LoessCorrector)
    assert (ticada.n_jobs, desticada.n_jobs) == (PARALELO, SERIE), (
        f"a caixa não alcançou o LOESS de `/execucoes`: {ticada.n_jobs} e {desticada.n_jobs}"
    )
    assert ticada.identidade() == padrao_loess.identidade(), (
        "o LOESS de `/execucoes` nasce com outro `null_denominator` que o padrão: a rota passou a "
        "mudar resultado, e o que a caixa muda é tempo"
    )


def test_a_caixa_alcanca_a_rotulagem_que_constroi_por_dentro() -> None:
    """`medir-corretor` é a rota mais cara, e o corretor dela nasce dentro do `LabelingOracle`."""
    assert build_corrector_for_labeling("serrf").n_jobs == PARALELO, (
        "o padrão da rotulagem deixou de ser `-1`: as ~100 matrizes do corpus passam a custar 5,5x"
    )
    assert build_corrector_for_labeling("serrf", n_jobs=SERIE).n_jobs == SERIE
    assert LabelingOracle().n_jobs == PARALELO, "o campo do oráculo nasceu com outro padrão"
    assert LabelingOracle(n_jobs=SERIE).n_jobs == SERIE

    # O LOESS da rotulagem recebe a mesma caixa — é o Palpite das telas.
    assert build_corrector_for_labeling("loess").n_jobs == PARALELO
    assert build_corrector_for_labeling("loess", n_jobs=SERIE).n_jobs == SERIE


# ---------------------------------------------------------------------------
# O PORTAO DE EXAUSTIVIDADE, e a ausencia dele foi medida em 08/09/2026
# ---------------------------------------------------------------------------
#
# **`POST /multiversos` ficou fora da rede, e nada o acusou.** Antes, o
# `JOBLIB_MULTIPROCESSING=0` desligava o `joblib` inteiro e protegia aquela rota **por acidente**.
# A variavel saiu, e o que a substituiu e uma conjuncao aplicada sitio a sitio — que nao a
# alcancava, e a suite nao tinha como acusar.
#
# **O oraculo foi CONTADO antes de escrito.** A primeira forma — «a funcao tem de nomear a
# capacidade» — acusava **quatro** rotas legitimas, que delegam em `_corretor_ou_422(paralelo=...)`.
# Aceitar tambem o repasse por `paralelo=` baixou a contagem para **zero na arvore consertada** e
# **uma na arvore anterior**, que e exactamente `dispara_multiverso`.

CAPACIDADE: Final[frozenset[str]] = frozenset({"n_jobs_efetivo", "regime_do_processo"})


def _eixos_de_paralelismo(funcao: ast.FunctionDef) -> list[str]:
    """Os parametros pelos quais um pedido pode pedir mais de um trabalhador ao `loky`."""
    nomes = [a.arg for a in funcao.args.args + funcao.args.kwonlyargs]
    return [n for n in nomes if n == "paralelo" or n == "n_jobs" or n.endswith("_jobs")]


def _consulta_a_capacidade(funcao: ast.FunctionDef) -> bool:
    """A funcao pergunta pela capacidade, ou repassa a escolha a quem pergunta?

    **As duas formas contam, e a segunda e o que evita quatro falsos positivos.** `corrigir`,
    `processar`, `exportar` e `executar` nao nomeiam `n_jobs_efetivo`: entregam `paralelo=` a
    `_corretor_ou_422` ou a `_corretor_da_spec`, que sao quem faz a conjuncao. Exigir a mencao
    direta transformaria o portao numa lista de isencoes.
    """
    nomes = {n.id for n in ast.walk(funcao) if isinstance(n, ast.Name)}
    if nomes & CAPACIDADE:
        return True
    return any(
        isinstance(no, ast.Call) and any(k.arg == "paralelo" for k in no.keywords)
        for no in ast.walk(funcao)
    )


def _rotas_com_eixo(fonte: str) -> list[tuple[str, list[str], bool]]:
    """`(nome, eixos, consulta)` para toda funcao de `rotas.py` com eixo de paralelismo."""
    achadas = []
    for no in ast.walk(ast.parse(fonte)):
        if isinstance(no, ast.FunctionDef) and (eixos := _eixos_de_paralelismo(no)):
            achadas.append((no.name, eixos, _consulta_a_capacidade(no)))
    return achadas


def test_toda_rota_com_eixo_de_paralelismo_consulta_a_capacidade() -> None:
    """Nenhum sitio pede trabalhador ao `loky` sem passar pela capacidade do processo.

    O que este portao impede: com a rede em baixo — sem interpretador,
    com o `loky` de outra forma, ou com o caminho fora da pasta — um `-1` que escape a conjuncao
    faz o `loky` relancar o `refino.exe`. Sao as catorze janelas de 26/08/2026.
    """
    fonte = (REPO / "src" / "refino" / "api" / "rotas.py").read_text(encoding="utf-8")
    com_eixo = _rotas_com_eixo(fonte)

    # A ancora de nao-vacuidade, NA MESMA funcao: um `ast.parse` que falhasse, ou um
    # detector que deixasse de reconhecer os eixos, daria lista vazia — e lista vazia satisfaz
    # a asserção de ausencia abaixo.
    assert len(com_eixo) >= 6, (
        f"o detector achou {len(com_eixo)} funcoes com eixo de paralelismo em `rotas.py`, e em "
        f"08/09/2026 havia oito: {[n for n, _, _ in com_eixo]}. Ou as rotas mudaram de forma, ou "
        "o detector deixou de as reconhecer — e a asserção abaixo passaria por vacuidade"
    )

    fora = [(nome, eixos) for nome, eixos, consulta in com_eixo if not consulta]
    assert not fora, (
        f"estas rotas pedem trabalhador ao `loky` sem consultar a capacidade do processo: {fora}. "
        "Com a rede de `aponta_o_loky` em baixo, o `-1` chega ao `loky` nao remendado e ele "
        "relanca o `refino.exe` — catorze processos COM JANELA, que e AR192. Ou a rota chama "
        "`n_jobs_efetivo`/`regime_do_processo`, ou repassa `paralelo=` a quem chama"
    )


def test_o_portao_acusa_quando_o_grampo_do_multiverso_sai() -> None:
    """O par causal: sem o grampo de `/multiversos`, o portao acima REPROVA.

    **Sem esta funcao o portao seria decorativo.** Ele nasceu contra uma rota que ja esta
    consertada, entao a arvore limpa nao o exercita — e um portao que nunca viu a violacao e
    indistinguivel de um que nao a consegue ver.

    A planta e a remocao das duas linhas do grampo, sobre o texto lido do arquivo: nao se toca no
    disco, e o que se afirma e que o DETECTOR muda de veredito.
    """
    fonte = (REPO / "src" / "refino" / "api" / "rotas.py").read_text(encoding="utf-8")
    grampo = (
        "    if regime_do_processo() == SERIE:\n"
        "        n_jobs = SERIE\n"
        "        corrector_jobs = SERIE\n"
    )
    assert fonte.count(grampo) == 1, (
        "o grampo de `/multiversos` mudou de forma; esta planta deixou de aterrar, e o par causal "
        "passa a medir outra coisa. Ver AR193"
    )

    plantado = fonte.replace(grampo, "")
    acusadas = [nome for nome, _, consulta in _rotas_com_eixo(plantado) if not consulta]
    assert acusadas == ["dispara_multiverso"], (
        f"com o grampo removido o portao acusa {acusadas}, e devia acusar exactamente "
        "`dispara_multiverso`. Se acusa demais, ele tem falso positivo; se acusa de menos, ele "
        "nao consegue ver a violacao que existiu de facto"
    )


# ---------------------------------------------------------------------------
# A dica da caixa diz a quem ela chega, e não o que era antes do LOESS paralelo
# ---------------------------------------------------------------------------


def _dica_da_janela() -> str:
    """`_DICA_DOS_NUCLEOS`, lida da fonte: importar o módulo da janela exigiria o Qt aqui."""
    fonte = (REPO / "src" / "refino" / "desktop" / "janela" / "pipeline.py").read_text("utf-8")
    for no in ast.walk(ast.parse(fonte)):
        if isinstance(no, ast.AnnAssign) and getattr(no.target, "id", "") == "_DICA_DOS_NUCLEOS":
            assert no.value is not None
            valor = ast.literal_eval(no.value)
            assert isinstance(valor, str)
            return valor
    raise AssertionError("`_DICA_DOS_NUCLEOS` saiu de `desktop/janela/pipeline.py`")


def _alcancados_pela_caixa() -> tuple[set[str], set[str]]:
    """`(alcançados, não alcançados)`: os corretores cujo `n_jobs` a caixa muda na rota."""
    alcancados, fora = set(), set()
    anterior = paralelismo._REGIME
    paralelismo._REGIME = PARALELO
    try:
        for nome in get_args(Correction):
            ticada = getattr(_corretor_ou_422(nome, "r_faithful", paralelo=True), "n_jobs", None)
            desticada = getattr(
                _corretor_ou_422(nome, "r_faithful", paralelo=False), "n_jobs", None
            )
            (alcancados if ticada != desticada else fora).add(nome)
    finally:
        paralelismo._REGIME = anterior
    return alcancados, fora


def _o_que_a_dica_erra(dica: str, alcancados: set[str], fora: set[str]) -> list[str]:
    """Os corretores alcançados que a dica põe numa frase de «não a usa», ou que ela não nomeia."""
    erros = []
    frases = re.split(r"(?<=\.)\s+|\n+", dica)
    for frase in frases:
        if re.search(r"não a us", frase, re.IGNORECASE):
            erros += [
                f"{nome}: «{frase.strip()}»"
                for nome in sorted(alcancados)
                if re.search(rf"\b{nome}\b", frase, re.IGNORECASE)
            ]
    erros += [
        f"{nome}: não nomeado"
        for nome in sorted(alcancados)
        if not re.search(rf"\b{nome}\b", dica, re.IGNORECASE)
    ]
    erros += [
        f"{nome}: não dito fora"
        for nome in sorted(fora)
        if not any(
            re.search(rf"\b{nome}\b", f, re.IGNORECASE) and re.search(r"não a us", f, re.I)
            for f in frases
        )
    ]
    return erros


def test_as_duas_dicas_da_caixa_dizem_a_quem_ela_chega() -> None:
    """A dica da janela nomeia o SERRF e o LOESS como alcançados, e o `none` fora.

    Antes, a dica dizia *«O loess e o none não a usam: um não paraleliza, o outro não calcula»*,
    que era verdade até o LOESS paralelizar. **O conjunto dos alcançados sai das rotas** — o
    corretor cujo `n_jobs` muda quando a caixa muda —, e a asserção sobre a dica lê esse conjunto e
    não um literal: o dia em que outro corretor paralelizar, a dica reprova até o nomear, e a planta
    do fim mostra isso com o `none` posto entre os alcançados.

    **A âncora é de não-vacuidade e não um pino**: os dois conjuntos não vazios, e o SERRF, o
    primeiro a ler a caixa, entre os alcançados. *A primeira versão pinava os dois conjuntos, e
    então quem reprovava primeiro era a âncora e não a dica.*
    """
    alcancados, fora = _alcancados_pela_caixa()
    assert alcancados and fora and "serrf" in alcancados, (
        f"âncora: a caixa alcança {sorted(alcancados)} e deixa {sorted(fora)} — com um dos dois "
        "vazio a asserção seguinte seria vácua"
    )
    for nome, dica in (("janela", _dica_da_janela()),):
        assert _o_que_a_dica_erra(dica, alcancados, fora) == [], f"a dica da {nome}: {dica}"

    # **O corretor novo que paraleliza**, plantado: o `none` entre os alcançados, e a dica de
    # hoje acusada por o pôr numa frase de «não a usa».
    for nome, dica in (("janela", _dica_da_janela()),):
        erros = _o_que_a_dica_erra(dica, alcancados | fora, set())
        assert erros and all(e.startswith("none: ") for e in erros), (
            f"a dica da {nome} não foi acusada com o `none` alcançado: {erros}"
        )

    antiga = "Divide o SERRF. O loess e o none não a usam: um não paraleliza, o outro não calcula."
    assert _o_que_a_dica_erra(antiga, alcancados, fora) == [
        "loess: «O loess e o none não a usam: um não paraleliza, o outro não calcula.»"
    ], "a frase antiga não foi acusada"
