"""Qualidade, PLS-DA e antes-e-depois na janela nativa.

O conjunto completo das métricas e o PLS-DA sempre com validação são o assunto, e o portão é sobre
estrutura e não sobre cuidado.

## O ORÁCULO É O WIDGET, e não o valor de retorno — a v1 deste arquivo caiu por isso

A revisão adversarial construiu dezasseis mutantes; **cinco** passavam porque toda asserção lia o
dicionário que os `mostra_*` devolvem:

| Mutante | O que o usuário via | Antes |
|---|---|---|
| rótulos construídos e **fora do layout** | painel **em branco** | passava |
| `setText(f"{nome}:")`, retorno intacto | onze nomes, **zero valores** | passava |
| o valor do `q2` copiado para o `p` | a validação a ser **cópia do score** | passava |
| a comparação nunca escrita | *"nenhuma comparação disponível"* para sempre | passava |

> **Um rótulo que existe e não está no layout é um painel em branco, e o dicionário de retorno não
> sabe disso.** `PainelDeQualidade.texto_visivel()` é o oráculo: ele lê `.text()` **e** confere
> `layout().indexOf(rotulo) >= 0`, e é por ele que toda asserção passa.

## E OS DOIS PORTÕES DE AST PASSARAM A SER LISTA DE PERMISSÃO

O do par proibia `plsda["literal"]`, e **sete** formas escapavam — variável, `next(iter(...))`,
fatia
de tupla. O da aritmética exigia que o texto do `BinOp` contivesse `float(`, `int(`, `.get(` ou
`corpo[`, e **quatro** escapavam — variável local, `AugAssign`, `operator.mul`, e o especificador de
formato `:.1%`, que não é `BinOp` nenhum.

**Proibir uma sintaxe é enumerar o que se conhece; permitir uma é enumerar o que se quer.** Os dois
portões passaram à segunda forma: o do par exige que toda escrita seja dirigida pelo laço, e o da
aritmética proíbe **toda** operação aritmética no módulo — que legitimamente não tem nenhuma, logo a
lista de permissão é vazia e não precisa de discriminador.
"""

from __future__ import annotations

import ast
import re
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from tests.pedido_da_tela import campos_da_tela
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

from refino.api.app import build_app  # noqa: E402
from refino.api.esquemas import RESSALVA_DO_ESTIMADOR, RsdAntesEDepois  # noqa: E402
from refino.desktop.janela.pipeline import (  # noqa: E402
    PainelDePipeline,
    rota_do_progresso,
)
from refino.desktop.janela.principal import ABAS  # noqa: E402
from refino.desktop.janela.qualidade import (  # noqa: E402
    ESTILO_DA_AUSENCIA,
    ESTILO_DA_LEGENDA,
    ESTILO_DO_VALOR,
    PREFIXO_DO_VALOR,
    SEM_COMPARACAO,
    TITULO_DA_SECAO_DO_PLSDA,
    PainelDeQualidade,
    aviso_do_plsda,
    nomes_do_plsda,
    texto_da_comparacao,
    texto_da_metrica,
)
from refino.interface.fluxo import (  # noqa: E402
    CAMINHO_DA_METRICA,
    METRICAS_DA_TELA_DE_QUALIDADE,
    METRICAS_DE_VALIDACAO_DO_PLSDA,
    METRICAS_DO_PLSDA,
    METRICAS_SOB_PAR_DE_GRUPOS,
    NAO_CALCULADA,
    NENHUMA_ROTA_CALCULA,
    PARES_DE_VALIDACAO_OBRIGATORIA,
    SEM_CAMINHO,
    SEM_CAMINHO_PARA_PLSDA,
    SEM_PLSDA,
    SEM_VALOR_DEFINIDO,
    SOB_PAR_DE_GRUPOS_NA_JANELA,
    TRADUCAO_DA_METRICA,
    agrupa_metricas_da_qualidade,
    endereco_da_metrica,
    legenda_da_metrica,
    procedencia_do_plsda,
    prosa_da_variante_de_rsd,
    prosa_do_estimador_de_rsd,
    rotulo_da_metrica,
)
from tests.conftest import measurements_path, sample_list_path  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
QUALIDADE = REPO / "src" / "refino" / "desktop" / "janela" / "qualidade.py"
GRAFICOS = REPO / "src" / "refino" / "desktop" / "janela" / "graficos.py"

# A população do portão de aritmética: os módulos que MONTAM a tela do resultado de uma
# execução — hoje `qualidade.py` e `graficos.py`.
#
# **Uma lista de dois com o critério ao lado, e NÃO uma derivação por glob**, e isso está medido:
# a propriedade *"este módulo não faz aritmética"* é **falsa** para a maioria do pacote. Um glob
# sobre `desktop/janela/*.py` reprovaria em cinco dos treze módulos já na primeira execução —
# `principal` com 14 operações, `leitura` com 6, `pipeline` com 6, `artefatos` com 1 e
# `multiverso` com 1. A frase acima é o critério de pertença, e é ela que diz ao próximo se um
# módulo novo entra.
#
# **E a extensão não é zelo: sem ela, metade do código que o portão media saía da população por
# mudar de arquivo, sem nada acusar.** O portão continuava verde, e o eixo em que ele pode falhar
# tinha-se mudado de casa.
MODULOS_QUE_MONTAM_O_RESULTADO = (QUALIDADE, GRAFICOS)


@pytest.fixture
def painel(qtbot: Any) -> PainelDeQualidade:
    p = PainelDeQualidade()
    qtbot.addWidget(p)
    return p


def _valores(painel: PainelDeQualidade) -> dict[str, str]:
    """O que cada rótulo MOSTRA, sem o prefixo. **O oráculo.**

    Rótulo fora do layout devolve `""` de `texto_visivel`, e o `split` sobre string vazia dá `""` —
    então um painel em branco produz valores vazios, que é o que as asserções apanham.

    > O prefixo era o nome técnico e passou a `PREFIXO_DO_VALOR`. O `split(": ", 1)` fica
    > exactamente como estava, e é isso que a constante compra: medido, com prefixo reprovam 14
    > testes e sem ele 19 — medido sobre os três arquivos como estavam em `HEAD`, 268 testes. As
    > cinco a mais são arnêses como este, que passariam a devolver `""` e a reportar um painel em
    > branco que não existe.
    """
    visivel = painel.texto_visivel()
    return {
        nome: (texto.split(": ", 1)[1] if ": " in texto else "")
        for nome, texto in visivel.items()
        if not nome.startswith("__")
    }


# ---------------------------------------------------------------------------
# O conjunto completo, nunca um número
# ---------------------------------------------------------------------------


def test_u7_o_conjunto_completo_aparece_no_layout_e_a_enumeracao_e_derivada(
    painel: PainelDeQualidade,
) -> None:
    """O conjunto completo na apresentação: todas as métricas VISÍVEIS, e a lista derivada.

    RSD e agrupamento de QC no PCA **discordam e elegem métodos diferentes**; eleger um valor na
    única camada em que o usuário olha é reduzir qualidade a um número.

    **A asserção é sobre o LAYOUT e não sobre o dicionário de retorno.** A revisão adversarial
    construiu um painel que criava os onze rótulos e nunca os punha no layout — painel **em
    branco**,
    e a v1 deste teste passava.

    O número vem com a população: são onze métricas, e a partição de `CAMINHO_DA_METRICA` é nove
    `api` e duas `nenhum` — eram oito e três antes de a contagem de FDR ganhar caminho.

    > Era seis e cinco antes, e a mudança é o par do PLS-DA. `POST /execucoes`
    > passou a calcular `plsda_q2` e `plsda_permutation_p` quando o par de grupos e declarado, e os
    > dois sairam de «nenhum caminho» para «api». Sobram tres sem caminho: `n_significant_fdr`,
    > `qc_rsd_cv_median` e `qc_rsd_r_cv_median`.
    """
    assert METRICAS_DA_TELA_DE_QUALIDADE, "âncora: o conjunto de métricas está vazio"
    assert len(METRICAS_DA_TELA_DE_QUALIDADE) == 11, (
        f"a população mudou: {len(METRICAS_DA_TELA_DE_QUALIDADE)} métricas"
    )
    caminhos = Counter(CAMINHO_DA_METRICA[n] for n in METRICAS_DA_TELA_DE_QUALIDADE)
    assert caminhos == {"api": 9, "nenhum": 2}, f"a partição de caminho mudou: {caminhos}"

    painel.mostra_qualidade({})
    visivel = painel.texto_visivel()

    for nome in METRICAS_DA_TELA_DE_QUALIDADE:
        assert nome in visivel, f"`{nome}` não tem rótulo"
        assert visivel[nome], (
            f"o rótulo de `{nome}` não está VISÍVEL — construído e fora do layout é um painel em "
            f"branco"
        )
        assert visivel[nome].startswith(PREFIXO_DO_VALOR), (
            f"a linha do valor de `{nome}` perdeu o prefixo, e os arnêses que fazem "
            f'`split(": ", 1)` passam a ler o painel como vazio: {visivel[nome]!r}'
        )

    # E os rótulos são widgets DISTINTOS: um só, reusado, mostraria apenas a última métrica.
    assert len({id(w) for w in painel.metricas.values()}) == len(painel.metricas), (
        "há rótulos partilhados entre métricas — o painel mostraria só a última"
    )


def test_u7_os_valores_aparecem_e_nao_apenas_os_nomes(painel: PainelDeQualidade) -> None:
    """O rótulo mostra `nome: VALOR`, e o valor não é vazio.

    A revisão adversarial construiu `setText(f"{nome}:")` — nomes sem valores, com o dicionário de
    retorno intacto — e a v1 passava, porque a única asserção de widget era `nome in text()`, que o
    nome sozinho satisfaz.

    **A âncora é a distinção:** com um valor real, o texto do rótulo **muda**.
    """
    painel.mostra_qualidade({})
    antes = _valores(painel)
    assert all(antes.values()), f"há métrica sem valor visível: {antes}"

    painel.mostra_qualidade({"computed": ["qc_rsd_median"], "qc_rsd_median": 12.5})
    depois = _valores(painel)
    # O oráculo mudou, e a mudança é de contrato de apresentação. Ele era `"12.5"`, o `repr` do
    # `float`; o valor passa pelo formatador partilhado, que escreve o RSD na unidade em que o
    # profissional o lê e acrescenta a população e os dois rótulos. Não é ajustar o número
    # esperado: o número medido é o mesmo, e o que muda é como se escreve.
    assert depois["qc_rsd_median"] == (
        "1250,0% — população não declarada nesta resposta; variante e estimador não declarados"
    ), f"o valor não chegou ao rótulo: {depois['qc_rsd_median']!r}"
    assert depois != antes, "âncora: o valor real não mudou nada no que está visível"


def test_u7_os_tres_estados_de_cada_metrica_sao_distintos(painel: PainelDeQualidade) -> None:
    """`computed` → valor; `nenhum` → `SEM_CAMINHO`; o resto → `NAO_CALCULADA`.

    O terceiro estado existe porque *"não calculada"* cobria *"a rota calcula e esta execução não
    tem valor"* e *"nenhuma rota calcula isto"* — fatos opostos, e o segundo é o de DUAS das onze;
    a contagem viva está na asserção abaixo.

    **A âncora é a distinção mútua**, e ela é sobre o que está VISÍVEL.
    """
    painel.mostra_qualidade({"computed": ["qc_rsd_median"], "qc_rsd_median": 12.5})
    visto = _valores(painel)

    # O oráculo passou do `repr` do `float` ao texto formatado. A propriedade que este
    # teste afirma é a DISTINÇÃO mútua dos três estados, e ela não depende da forma do número.
    assert visto["qc_rsd_median"].startswith("1250,0%")
    # O exemplo de «sem caminho» mudou duas vezes: era `plsda_q2` e depois `n_significant_fdr`, e
    # os dois ganharam caminho. Sobram as duas de RSD
    # fora da amostra, que ficam por CUSTO e nao por maquinaria ausente. A propriedade medida — os
    # tres estados sao mutuamente distintos — nao mudou nenhuma das duas vezes, e e ela que este
    # teste afirma; o nome e so o portador.
    assert visto["qc_rsd_cv_median"] == SEM_CAMINHO, (
        "uma métrica que nenhuma rota calcula saiu como «não calculada» — quem lê espera que outra "
        "execução preencha, e nenhuma preenche"
    )
    # O par do PLS-DA passou ao estado do meio. Ele tem caminho — logo não é *"sem caminho"* —, e
    # esta execução não declarou grupos: a diferença face a *"não calculada"* é que a ação existe e
    # é de quem opera, e a tela passou a dizer qual. Na janela, a frase é a versão própria dela — o
    # mesmo estado, a mesma ação.
    assert visto["plsda_q2"] == SOB_PAR_DE_GRUPOS_NA_JANELA, (
        "o par do PLS-DA depende do par de grupos, e a tela tem de dizer de que depende e onde se "
        "escolhe — nunca «sem caminho», que mandaria procurar um número que existe"
    )
    assert visto["frac_features_under_20"] == NAO_CALCULADA

    quatro = {
        visto["qc_rsd_median"],
        visto["qc_rsd_cv_median"],
        visto["plsda_q2"],
        visto["frac_features_under_20"],
    }
    assert len(quatro) == 4, f"dois dos quatro estados colapsaram: {quatro}"

    contagem = Counter(visto.values())
    assert contagem[SEM_CAMINHO] == 2, f"os «sem caminho» deviam ser DOIS: {contagem}"


def test_u7_os_rotulos_nascem_no_estado_certo(qtbot: Any) -> None:
    """Antes de qualquer execução: DOIS em `SEM_CAMINHO`, TRÊS sob argumento e seis em branco.

    > Era cinco e seis até o par do PLS-DA ganhar caminho, e três e oito até a contagem de FDR
    > ganhar o dela. E passou a três estados: as três que dependem do par de grupos diziam *"não
    > calculada"* como as outras, e quem abre a aba lia três linhas em branco como falha.

    > E passou de nove para onze `NAO_CALCULADA`, e os dois que entraram não são métricas novas
    > da grelha: são os da família multiclasse do PLS-DA, que vive só na seção dedicada ao PLS-DA.
    > `_valores` funde `self.metricas` com `self.plsda`, e a segunda passou de dois nomes a
    > quatro. As **onze** da grelha continuam onze — `tests/test_ui_fluxo.py` guarda-o.

    A primeira versão semeava todos com `NAO_CALCULADA`, e é a conflação que `SEM_CAMINHO` existe
    para desfazer — no módulo que argumenta por ela. Combinado com o painel não estar ligado, era o
    que o usuário de facto via.

    **A partição é DERIVADA das constantes e não escrita**, e é o que impede este teste de pinar o
    retrato de hoje: no dia em que uma métrica ganhar ou perder caminho, ele acompanha.
    """
    novo = PainelDeQualidade()
    qtbot.addWidget(novo)
    contagem = Counter(_valores(novo).values())
    # **A população é a UNIÃO dos dois conjuntos e não as onze — o merge do bloco das análises.**
    # O painel desenha o que a tela escreve, e a tela escreve de duas origens: a grelha
    # das onze escalares e a seção do PLS-DA, que inclui o par do multiclasse. Contar só as onze
    # faria esta partição reprovar por a família nova ter nascido, que é o alvo certo pelo motivo
    # errado — a mesma conflação que a âncora de `TRADUCAO_DA_METRICA` tinha.
    desenhadas = METRICAS_DA_TELA_DE_QUALIDADE | METRICAS_DE_VALIDACAO_DO_PLSDA
    sem_caminho = {n for n, c in CAMINHO_DA_METRICA.items() if c == "nenhum"}
    sob_par = METRICAS_SOB_PAR_DE_GRUPOS - sem_caminho
    restantes = desenhadas - sem_caminho - sob_par
    assert (len(sem_caminho), len(sob_par), len(restantes)) == (2, 3, 8), (
        f"a partição de estado mudou: {len(sem_caminho)} sem caminho, {len(sob_par)} sob "
        f"argumento, {len(restantes)} restantes, sobre {len(desenhadas)} desenhadas"
    )
    assert contagem == {
        NAO_CALCULADA: len(restantes),
        SOB_PAR_DE_GRUPOS_NA_JANELA: len(sob_par),
        SEM_CAMINHO: len(sem_caminho),
    }, f"os rótulos nascem no estado errado: {contagem}"
    # E o aviso do PLS-DA nasce escrito. A frase mudou: era «nenhum caminho calcula»,
    # e passou a ser «esta execução não declarou o par» — que é o estado de um painel recém-aberto,
    # e tem ação do lado de quem opera.
    assert novo.texto_visivel()["__aviso__"] == SEM_PLSDA, (
        "o aviso só apareceria depois da primeira execução, e quem abre a aba procuraria o score"
    )


def test_u7_computed_e_consultado_antes_do_valor() -> None:
    """A ordem das três perguntas é o conteúdo.

    **O par causal:** o **mesmo** valor `None`, uma vez dentro de `computed` e outra fora. Se a
    função lesse o valor primeiro, os dois sairiam iguais.
    """
    dentro = texto_da_metrica(
        "qc_rsd_median", {"computed": ["qc_rsd_median"], "qc_rsd_median": None}
    )
    fora = texto_da_metrica("qc_rsd_median", {"computed": [], "qc_rsd_median": None})

    assert dentro != fora, (
        f"«calculada e indefinida» e «não calculada» saíram iguais ({dentro!r}) — o valor foi lido "
        f"antes de `computed`"
    )
    # `"None"` era o que a tela mostrava, literalmente. `nan` chega ao cliente como
    # `null`, e `str(None)` é a palavra do nulo: nem português nem inglês, o nome de um objeto de
    # uma linguagem que quem lê a tela não programa. O estado é o mesmo — *calculada e
    # indefinida* —, e o que mudou é ele passar a dizê-lo.
    assert dentro.startswith(SEM_VALOR_DEFINIDO), dentro
    assert "None" not in dentro, f"a palavra do nulo voltou à tela: {dentro!r}"
    assert fora == NAO_CALCULADA
    assert (
        texto_da_metrica("qc_rsd_median", {"computed": [], "qc_rsd_median": 9.9}) == NAO_CALCULADA
    )


def test_u7_um_nome_fora_do_conjunto_nao_levanta() -> None:
    """`.get` e não subscrito: um exigido fora do conjunto de métricas cai em `NAO_CALCULADA`.

    Um `KeyError` aqui correria dentro de um slot de Qt, onde a excepção é engolida para o `stderr`
    —
    e num executável `--windowed` não há `stderr`. O painel simplesmente pararia de actualizar.
    """
    assert texto_da_metrica("uma_metrica_que_nao_existe", {}) == NAO_CALCULADA
    assert texto_da_metrica("uma_metrica_que_nao_existe", {"computed": []}) == NAO_CALCULADA


def test_u7_um_relatorio_que_nao_e_dicionario_nao_derruba_o_painel(
    painel: PainelDeQualidade,
) -> None:
    """Corpo que não é dicionário escreve os estados de ausência, e não levanta.

    É o achado de 15/08 que o plano nomeia — *"`float(valor) * 100` sobre valor ausente não produz
    traceback de painel inteiro"* —, e a forma que ele toma aqui: um `relatorio.get` sobre `None`
    seria `AttributeError` dentro de um slot.
    """
    for corpo in (None, "texto", 42, [], ()):
        painel.mostra_qualidade(corpo)
        visto = _valores(painel)
        assert set(visto) >= set(METRICAS_DA_TELA_DE_QUALIDADE)
        assert visto["qc_rsd_median"] == NAO_CALCULADA, f"corpo {corpo!r}"

    painel.mostra_qualidade({"computed": ["qc_rsd_median"], "qc_rsd_median": 1.5})
    assert _valores(painel)["qc_rsd_median"].startswith("150,0%"), (
        "âncora: nem com dicionário válido o valor aparece, e as asserções acima são vácuas"
    )


def test_u7_o_painel_nao_faz_aritmetica_nenhuma() -> None:
    """**Portão por AST com lista de permissão VAZIA: nenhuma operação aritmética nos módulos.**

    > A população passou de um módulo a dois quando cerca de metade de `qualidade.py` foi para
    > `graficos.py`, e um portão que continuasse a ler
    > só o primeiro ficaria verde sobre metade do código que ele existia para medir. O critério de
    > pertença está escrito ao lado de `MODULOS_QUE_MONTAM_O_RESULTADO`, e a planta abaixo corre
    > **sobre cada um dos dois** — senão a extensão seria intenção.

    O achado de 15/08 é `float(valor) * 100` sobre valor ausente a derrubar o painel inteiro. O
    conserto na raiz não é um `try`: é **não fazer a conta**.

    ## A v1 exigia um discriminador, e quatro formas escapavam

    Ela pedia que o texto do `BinOp` contivesse `float(`, `int(`, `.get(` ou `corpo[`. Escapavam,
    medidos: `bruto = float(...)` e depois `bruto * 100`; `bruto *= 100`, que é `AugAssign` e não
    `BinOp`; `operator.mul(...)`; e `f"{x:.1%}"`, que não tem operador nenhum.

    > **Este módulo legitimamente não tem aritmética**, então a lista de permissão é **vazia** e o
    > portão não precisa de discriminador: qualquer `BinOp`, `AugAssign` ou especificador de formato
    > percentual é acusado. Proibir uma sintaxe enumera o que se conhece; permitir uma enumera o que
    > se quer.

    As cinco plantas são as quatro que escapavam mais o controle, **em processo**.
    """

    def contas(texto: str) -> list[str]:
        achados = []
        arvore = ast.parse(texto)
        for no in ast.walk(arvore):
            if isinstance(no, ast.BinOp) and isinstance(
                no.op, ast.Mult | ast.Div | ast.Sub | ast.Add | ast.Pow | ast.FloorDiv | ast.Mod
            ):
                achados.append(f"linha {no.lineno} — BinOp {ast.unparse(no)[:40]}")
            elif isinstance(no, ast.AugAssign):
                achados.append(f"linha {no.lineno} — AugAssign {ast.unparse(no)[:40]}")
            elif isinstance(no, ast.Call):
                alvo = no.func
                nome = (
                    alvo.attr
                    if isinstance(alvo, ast.Attribute)
                    else alvo.id
                    if isinstance(alvo, ast.Name)
                    else ""
                )
                if nome in {"mul", "truediv", "add", "sub", "pow", "round"}:
                    achados.append(f"linha {no.lineno} — {nome}()")
            elif isinstance(no, ast.FormattedValue) and no.format_spec is not None:
                especificador = ast.unparse(no.format_spec)
                if "%" in especificador:
                    achados.append(f"linha {no.lineno} — formato {especificador}")
        return sorted(achados)

    assert len(MODULOS_QUE_MONTAM_O_RESULTADO) == 2, (
        "âncora: a população do portão encolheu, e o laço abaixo mediria menos do que declara"
    )
    for modulo in MODULOS_QUE_MONTAM_O_RESULTADO:
        assert modulo.is_file(), f"âncora: `{modulo.name}` não existe, e o laço leria o vazio"
        fonte_do_modulo = modulo.read_text(encoding="utf-8")
        assert contas(fonte_do_modulo) == [], (
            f"há aritmética em `{modulo.name}`: {contas(fonte_do_modulo)}"
        )

    plantas = {
        "controle: float * 100": (
            "\n\ndef _p(c: dict) -> str:\n    return f\"{float(c.get('x')) * 100:.1f}%\"\n"
        ),
        "variável local": (
            '\n\ndef _p(c: dict) -> str:\n    b = float(c.get("x"))\n    return f"{b * 100:.1f}%"\n'
        ),
        "AugAssign": (
            '\n\ndef _p(c: dict) -> str:\n    b = float(c.get("x"))\n'
            "    b *= 100\n    return str(b)\n"
        ),
        "operator.mul": (
            "\n\ndef _p(c: dict) -> str:\n    import operator\n\n"
            '    return str(operator.mul(float(c.get("x")), 100))\n'
        ),
        "especificador de formato": (
            "\n\ndef _p(c: dict) -> str:\n    return f\"{float(c.get('x')):.1%}\"\n"
        ),
    }
    # **As plantas correm sobre CADA módulo da população, e não só sobre o primeiro.** Plantar num
    # arquivo e afirmar cobertura sobre dois é a afirmação de cobertura mais larga que o mecanismo.
    for modulo in MODULOS_QUE_MONTAM_O_RESULTADO:
        limpo = modulo.read_text(encoding="utf-8")
        for rotulo, planta in plantas.items():
            plantado = f"{limpo}{planta}"
            assert plantado != limpo, f"a planta `{rotulo}` NÃO aterrou em `{modulo.name}`"
            assert contas(plantado), (
                f"o portão não acusou a planta `{rotulo}` em `{modulo.name}`: a asserção da "
                "árvore limpa é vácua para ela"
            )


# ---------------------------------------------------------------------------
# As onze deixam de ser nomes de campo em inglês
# ---------------------------------------------------------------------------


def test_u8_cada_metrica_tem_a_legenda_legivel_visivel_no_layout(
    painel: PainelDeQualidade,
) -> None:
    """O rótulo em português, a unidade e o sentido — no layout, e antes de qualquer clique.

    **A asserção é sobre o LAYOUT e não sobre o dicionário**, pela mesma razão que o teste do
    conjunto completo: uma legenda construída e nunca acrescentada é um painel sem glossário, e o
    dicionário de widgets não sabe disso.

    **E é EM REPOUSO**, sem alimentar o painel: era assim que o profissional via os onze nomes de
    campo em inglês — a aba aberta, nada clicado.
    """
    visivel = painel.texto_visivel()

    assert len(METRICAS_DA_TELA_DE_QUALIDADE) == 11, (
        f"a população mudou: {len(METRICAS_DA_TELA_DE_QUALIDADE)} métricas"
    )
    for nome in sorted(METRICAS_DA_TELA_DE_QUALIDADE):
        chave = f"__legenda__{nome}"
        assert chave in visivel, f"`{nome}` não tem legenda"
        legenda = visivel[chave]
        assert legenda, (
            f"a legenda de `{nome}` não está VISÍVEL — construída e fora do layout é um painel sem "
            "glossário"
        )
        # O nome em português mudou de sítio, e o portão passa a olhar os dois. Ele
        # era a primeira linha da legenda, em negrito, e é hoje o título da moldura que contém o
        # valor e a legenda — a mudança foi de apresentação, e a propriedade é a mesma: *"o nome
        # em português está na tela, junto da métrica"*. Aceitar qualquer um dos dois é o que
        # impede o portão de exigir a forma em vez da propriedade.
        moldura = visivel.get(f"__moldura__{nome}", "")
        assert rotulo_da_metrica(nome) in legenda or rotulo_da_metrica(nome) in moldura, (
            f"o nome em português de `{nome}` não está nem na legenda nem no título da moldura: "
            f"legenda={legenda!r} moldura={moldura!r}"
        )
        assert TRADUCAO_DA_METRICA[nome].unidade in legenda, (
            f"a legenda de `{nome}` não diz a unidade — e ler fração como percentagem trocava "
            "`clean` por `step_change` num modo inteiro (AR54)"
        )

    # E as legendas são widgets DISTINTOS: uma só, reusada, mostraria apenas a última.
    assert len({id(w) for w in painel.legendas.values()}) == len(painel.legendas), (
        "há legendas partilhadas entre métricas — o painel explicaria só a última"
    )


def test_u8_o_nome_tecnico_nao_sai_da_tela(painel: PainelDeQualidade) -> None:
    """A tradução acrescenta; ela não substitui. E o nome passou de rótulo a endereço.

    O nome técnico é o que o profissional usa para falar connosco e para achar a métrica no
    documento. Uma tela que só mostrasse *"RSD mediano dos QCs"* trocaria um defeito de
    legibilidade por um de rastreabilidade.

    > A posição mudou: os nomes com underscore saíram da linha, para a qualidade ser legível a
    > quem é só da metabolómica. Antes
    > ele ficava na linha do valor; ele sai de lá e reaparece **uma vez**, como última frase da
    > legenda — que é onde a consulta que de facto acontece, a **inversa**, o
    > procura: o profissional tem o nome cru num CSV e quer saber o que ele é.

    **As DUAS direções, e por isso este teste não é de ausência apenas:** o nome está na legenda
    (presença, e ela ancora) e não está na linha do valor (ausência).

    > E as duas entradas, porque a primeira versão deste teste ficou inerte contra metade do
    > caminho — achado pelo par causal. Ele lia só o painel **em
    > repouso**, que é semeado pelo construtor; uma planta que repusesse o identificador em
    > `mostra_qualidade` deixava-o **verde**. O painel em repouso é o valor natural, e é o que
    > executa menos do caminho.
    """
    assert len(METRICAS_DA_TELA_DE_QUALIDADE) == 11, (
        f"âncora: a população mudou para {len(METRICAS_DA_TELA_DE_QUALIDADE)} e o laço mediria "
        "outra coisa"
    )

    def confere(visivel: dict[str, str], onde: str) -> None:
        for nome in sorted(METRICAS_DA_TELA_DE_QUALIDADE):
            legenda = visivel[f"__legenda__{nome}"]
            assert legenda.endswith(endereco_da_metrica(nome)), (
                f"[{onde}] a legenda de `{nome}` não termina no endereço — quem tem o nome cru "
                f"num CSV exportado não acha o que ele é: {legenda!r}"
            )
            # **Uma vez e não duas**, que é a diferença entre endereço e rótulo: o nome aparecia na
            # abertura da legenda E na linha do valor, e o pedido é que ele deixe de ser o rótulo.
            assert legenda.count(nome) == 1, (
                f"[{onde}] `{nome}` aparece {legenda.count(nome)} vezes na própria legenda: "
                f"{legenda!r}"
            )
            assert nome not in visivel[nome], (
                f"[{onde}] a linha do valor de `{nome}` voltou a ser encabeçada pelo "
                f"identificador: {visivel[nome]!r}"
            )

    # **Em repouso**, que é o estado em que o profissional abre a aba antes de clicar em nada.
    confere(painel.texto_visivel(), "em repouso")

    # **E ALIMENTADO**, que é o caminho que o construtor não percorre. Com o par do PLS-DA, para
    # que a seção dedicada seja reescrita também — ela é a irmã, na mesma aba.
    relatorio = {
        "computed": sorted(METRICAS_DA_TELA_DE_QUALIDADE),
        "rsd_variant": "plain",
        "rsd_estimator": "in_sample",
        "n_features_rsd_defined": 2151,
        **{nome: 0.25 for nome in METRICAS_DA_TELA_DE_QUALIDADE},
    }
    escritos = painel.mostra_qualidade(relatorio)
    assert set(escritos) == METRICAS_DA_TELA_DE_QUALIDADE, (
        "âncora: a alimentação não escreveu as onze, e a conferência abaixo mediria o repouso"
    )
    assert painel.mostra_plsda(relatorio), "âncora: a seção dedicada do PLS-DA não foi escrita"
    alimentado = painel.texto_visivel()
    assert "25,0%" in alimentado["qc_rsd_median"], (
        f"âncora: o valor não chegou, e o painel continua no estado de repouso: "
        f"{alimentado['qc_rsd_median']!r}"
    )
    confere(alimentado, "alimentado")
    for nome in nomes_do_plsda():
        assert nome not in alimentado[f"__moldura_plsda__{nome}"], (
            f"a moldura da seção dedicada de `{nome}` mostra o identificador"
        )


def test_u13_a_legenda_chega_RENDERIZADA_em_quatro_blocos_e_a_quebra_sobrevive(  # noqa: N802
    painel: PainelDeQualidade,
) -> None:
    """**A legenda deixou de ser um muro, e o separador foi MEDIDO antes de escrito.**

    Ela era um parágrafo único de 200 a 500 caracteres — unidade, sentido, ressalva e referência
    bibliográfica colados —, onze vezes seguidas numa coluna só.

    **Em markdown uma `\\n` só é quebra *soft* e COLAPSA num espaço**, então pôr os gêneros em
    linhas separadas sem duplicar o separador não mudaria nada na tela. É a mesma medição que pôs
    `\\n\\n` na linha de antes-e-depois, aqui na legenda.

    **A planta é o separador de uma `\\n` só sobre exactamente o mesmo texto**, e é ela que prova
    que a contagem de blocos não passaria com qualquer separador.
    """
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QTextDocument

    nome = "frac_features_under_20"
    rotulo = painel.legendas[nome]
    assert rotulo.textFormat() == Qt.TextFormat.MarkdownText, (
        "a legenda voltou a `AutoText`, e os `**` de `ui/fluxo.py` voltam a aparecer literais"
    )

    texto = rotulo.text()
    # Âncora de não-vacuidade, e vai ANTES: sem o separador na fonte, a contagem de blocos
    # mediria um texto que nunca o teve.
    assert texto.count("\n\n") == 3, f"a legenda não tem os três separadores: {texto!r}"

    documento = QTextDocument()
    documento.setMarkdown(texto)
    assert documento.blockCount() == 4, (
        f"os gêneros da legenda colapsaram em {documento.blockCount()} bloco(s) — o separador "
        "voltou a ser uma `\\n` só, que em markdown é quebra soft e vira um espaço"
    )
    renderizado = documento.toPlainText()
    assert "**" not in renderizado, f"a marcação chegou LITERAL à tela: {renderizado!r}"

    # A PLANTA, sobre exactamente o mesmo texto: uma `\n` só onde hoje há duas.
    com_uma_quebra = QTextDocument()
    com_uma_quebra.setMarkdown(texto.replace("\n\n", "\n"))
    assert com_uma_quebra.blockCount() == 1, (
        f"a planta não separou: com uma `\\n` só o Qt deu {com_uma_quebra.blockCount()} blocos, e "
        "não 1 — o mecanismo do colapso mudou e este cenário deixou de medi-lo"
    )


def test_u13_a_linha_do_valor_tem_prefixo_e_hierarquia_de_widget(
    painel: PainelDeQualidade,
) -> None:
    """**`PREFIXO_DO_VALOR` é MECANISMO, e a hierarquia é do widget e não do texto.**

    O nome técnico saiu da linha do valor. O que entra no lugar dele não é enfeite: cinco
    arnêses deste repositório extraem o valor com `texto.split(": ", 1)[1]`, e sem prefixo eles
    passariam a devolver `""` — que se lê como *"o painel está em branco"*, e é falso. O veredito
    falaria do instrumento e acusaria o alvo.

    **Medido em 15/09/2026**, com as duas pontas na mesma sessão e sobre os três arquivos de
    teste como estavam em `HEAD` — 268 testes: com a constante reprovam **14**; sem ela, **19**.

    **A planta é o rótulo fora do layout**, que é o único estado em que *"em branco"* é verdade — e
    é ela que separa esta asserção de uma tautologia sobre o ajudante.
    """
    assert PREFIXO_DO_VALOR.endswith(": "), (
        f"o prefixo perdeu a forma `algo: `, e é ela que os cinco arnêses cortam: "
        f"{PREFIXO_DO_VALOR!r}"
    )

    painel.mostra_qualidade({"computed": ["qc_rsd_median"], "qc_rsd_median": 0.25})
    visivel = painel.texto_visivel()
    assert visivel["qc_rsd_median"].startswith(PREFIXO_DO_VALOR), visivel["qc_rsd_median"]
    # O ajudante corta o prefixo e devolve o valor — não `""`, e não a linha inteira.
    cortado = _valores(painel)["qc_rsd_median"]
    assert cortado.startswith("25,0%"), cortado
    assert cortado != visivel["qc_rsd_median"], "o corte virou a identidade: o prefixo sumiu"

    # **A PLANTA: fora do layout, e aí o vazio é a resposta certa.**
    painel.metricas["qc_rsd_median"].setParent(None)
    assert _valores(painel)["qc_rsd_median"] == "", (
        "o ajudante continua a mostrar um rótulo que não está desenhado"
    )

    # **A hierarquia é do widget** — 13pt com peso contra 9pt —, e encurta o caminho de leitura
    # sem apagar uma palavra. Apagar está fora de questão: a referência da área foi escolha do
    # usuário.
    #
    # O corpo do número é só do NÚMERO: a métrica calculada leva-o, e a ausência — a
    # frase que diz onde se escolhe o par — leva o de nota. O par está no mesmo relatório:
    # `qc_rsd_median` foi calculada acima e `qc_rsd_mean` não.
    assert len({ESTILO_DO_VALOR, ESTILO_DA_LEGENDA, ESTILO_DA_AUSENCIA}) == 3, (
        "âncora: dois dos três estilos são o mesmo, e o par abaixo não separaria nada"
    )
    assert painel.metricas["qc_rsd_median"].styleSheet() == ESTILO_DO_VALOR
    assert painel.metricas["qc_rsd_mean"].styleSheet() == ESTILO_DA_AUSENCIA
    assert painel.legendas["qc_rsd_mean"].styleSheet() == ESTILO_DA_LEGENDA
    assert painel.plsda["plsda_q2"].styleSheet() == ESTILO_DA_AUSENCIA, (
        "a seção dedicada do PLS-DA ficou com a hierarquia antiga — a irmã está na mesma aba"
    )
    # A irmã também sobe ao corpo do número quando o par chega calculado.
    painel.mostra_plsda(
        {
            "computed": ["plsda_q2", "plsda_permutation_p"],
            "plsda_q2": 0.24,
            "plsda_permutation_p": 0.04,
        }
    )
    assert painel.plsda["plsda_q2"].styleSheet() == ESTILO_DO_VALOR, (
        "o Q² calculado ficou com o corpo da nota"
    )


def test_u8_a_legenda_fora_do_layout_e_acusada(painel: PainelDeQualidade) -> None:
    """**O par causal do oráculo**: a mesma legenda, dentro e fora do layout.

    Sem esta direção, *"a legenda está visível"* seria satisfeito por um `texto_visivel` que
    devolvesse o texto de qualquer rótulo construído — e foi exactamente esse o mutante que a
    revisão adversarial construiu para os valores: rótulos criados e nunca acrescentados,
    painel em branco, suíte verde.
    """
    nome = sorted(METRICAS_DA_TELA_DE_QUALIDADE)[0]
    antes = painel.texto_visivel()[f"__legenda__{nome}"]
    assert antes, "âncora: a legenda já estava invisível, e a planta não distinguiria nada"

    painel.legendas[nome].setParent(None)

    depois = painel.texto_visivel()[f"__legenda__{nome}"]
    assert depois == "", (
        f"a legenda saiu do layout e o oráculo continua a mostrá-la: {depois!r} — ele leria um "
        "painel sem glossário como um painel com glossário"
    )


def test_u8_as_tres_sem_caminho_dizem_por_que_e_as_outras_nao(
    painel: PainelDeQualidade,
) -> None:
    """A razão chega a quem lê, em português — e só onde ela se aplica.

    *"não disponível nesta versão"* sozinho manda o profissional procurar o botão que a preencheria.
    A frase de ausência diz que não há botão nenhum, e o motivo diz por quê.

    A âncora do tamanho vem antes: com conjunto vazio, o laço não afirmaria nada, e a
    segunda metade — *"as outras oito não emitem a frase"* — é asserção de ausência, que passa por
    vacuidade sobre uma tela em branco.
    """
    visivel = painel.texto_visivel()
    sem_caminho = {n for n, c in CAMINHO_DA_METRICA.items() if c == "nenhum"}
    assert sem_caminho == {"qc_rsd_cv_median", "qc_rsd_r_cv_median"}, (
        f"a partição mudou: {sorted(sem_caminho)}. **O conjunto e não a contagem**: o número "
        "sozinho passaria se uma saísse e outra entrasse"
    )

    for nome in sorted(sem_caminho):
        legenda = visivel[f"__legenda__{nome}"]
        assert NENHUMA_ROTA_CALCULA in legenda, (
            f"`{nome}` nunca terá valor e a legenda não o diz: {legenda!r}"
        )
        assert TRADUCAO_DA_METRICA[nome].por_que_sem_caminho in legenda, (
            f"a legenda de `{nome}` não diz o motivo, e sem motivo não há o que decidir"
        )
        assert visivel[nome].endswith(SEM_CAMINHO), (
            f"o valor de `{nome}` devia ser a sentinela de ausência: {visivel[nome]!r}"
        )

    for nome in sorted(METRICAS_DA_TELA_DE_QUALIDADE - sem_caminho):
        assert NENHUMA_ROTA_CALCULA not in visivel[f"__legenda__{nome}"], (
            f"`{nome}` TEM caminho e a legenda diz que nada a calcula — quem lê deixaria de "
            "preencher os campos que a produzem"
        )


def test_u8_a_secao_do_plsda_tambem_traduz_os_dois_nomes(painel: PainelDeQualidade) -> None:
    """Os dois nomes do par aparecem DUAS vezes no painel, e as duas com legenda.

    A seção dedicada ao PLS-DA repete `plsda_q2` e `plsda_permutation_p` de propósito — é ali que o
    score e a validação ficam lado a lado. Traduzir só a lista de cima deixaria dois dos onze nomes
    de campo em inglês na única seção onde eles são o assunto.
    """
    visivel = painel.texto_visivel()
    nomes = nomes_do_plsda()
    assert nomes, "âncora: a lista de nomes do par está vazia"

    for nome in nomes:
        chave = f"__legenda_plsda__{nome}"
        assert chave in visivel, f"a seção do PLS-DA não tem legenda para `{nome}`"
        assert visivel[chave], f"a legenda de `{nome}` na seção do PLS-DA não está no layout"
        assert legenda_da_metrica(nome) in visivel[chave], (
            "a seção do PLS-DA redigitou a legenda em vez de a derivar da tabela partilhada: "
            f"{visivel[chave]!r}"
        )
        moldura_do_par = visivel.get(f"__moldura_plsda__{nome}", "")
        assert (
            rotulo_da_metrica(nome) in visivel[chave] or rotulo_da_metrica(nome) in moldura_do_par
        ), (
            f"o nome em português de `{nome}` não está na seção do PLS-DA — nem na legenda nem "
            f"no título da moldura"
        )
        # A comparação com a grelha só vale para quem ESTÁ nas duas seções. A família
        # multiclasse do PLS-DA vive só aqui — é coleção e não escalar, e não é uma das onze —,
        # então `__legenda__plsda_multiclass_q2` não existe. Comparar com uma chave ausente daria
        # `KeyError` dentro de um teste, sobre um par que não tem gêmeo por desenho.
        if f"__legenda__{nome}" in visivel:
            assert visivel[chave] == visivel[f"__legenda__{nome}"], (
                "as duas seções mostram legendas DIFERENTES para a mesma métrica, e é a "
                "divergência que a fonte única existe para não ter"
            )

    # A âncora de não-vacuidade do `if` acima: sem ela, uma mudança que tirasse o par
    # binário da grelha deixaria o ramo de comparação sem passar nunca, e o teste continuaria verde.
    assert any(f"__legenda__{nome}" in visivel for nome in nomes), (
        "nenhum nome do PLS-DA está também na grelha das onze: a comparação entre as duas seções "
        "nunca correu"
    )


# ---------------------------------------------------------------------------
# A lista lê-se por tema, e nenhuma métrica se perde no caminho
# ---------------------------------------------------------------------------


def test_u6_os_cabecalhos_de_tema_estao_no_layout_e_as_onze_continuam_visiveis(
    painel: PainelDeQualidade,
) -> None:
    """A partição chegou à TELA, e não só à tabela.

    `tests/test_ui_fluxo.py` mede que `agrupa_metricas_da_qualidade` reparte as onze; isso não diz
    nada sobre o painel as **desenhar** repartidas. Um construtor que chamasse a função e ignorasse
    os grupos deixaria aquele teste verde e esta aba na coluna única de sempre.

    E o conjunto completo é reafirmado AQUI, e não só no teste dele: a construção mudou, e a
    evidência que aquele teste tinha era sobre um laço que já não existe. Agrupar não pode esconder.

    **As asserções são sobre o LAYOUT**, e é a distinção que `_no_layout` mede: um `QGroupBox`
    construído e fora do layout não é desenhado, e o dicionário de retorno não sabe disso.
    """
    grupos = agrupa_metricas_da_qualidade()
    visivel = painel.texto_visivel()

    assert grupos, "âncora: a partição está vazia e o laço abaixo seria vácuo"
    for grupo in grupos:
        chave = f"__tema__{grupo.chave}"
        assert chave in visivel, f"o tema `{grupo.chave}` não tem moldura no painel"
        assert visivel[chave] == grupo.titulo, (
            f"o cabeçalho de `{grupo.chave}` não está VISÍVEL ou foi redigitado: {visivel[chave]!r}"
        )
        assert visivel[f"__nota_tema__{grupo.chave}"] == grupo.nota, (
            f"a nota do tema `{grupo.chave}` não chegou ao layout"
        )

    # O conjunto completo, depois do agrupamento: o CONJUNTO e não a contagem.
    desenhadas = {n for n in visivel if not n.startswith("__")}
    assert desenhadas >= METRICAS_DA_TELA_DE_QUALIDADE, (
        f"o agrupamento escondeu {sorted(METRICAS_DA_TELA_DE_QUALIDADE - desenhadas)}"
    )
    for nome in METRICAS_DA_TELA_DE_QUALIDADE:
        assert visivel[nome], f"o rótulo de `{nome}` não está visível depois do agrupamento"


def test_u6_uma_metrica_indefinida_nao_apaga_o_tema_nem_os_vizinhos(
    painel: PainelDeQualidade,
) -> None:
    """**BORDA:** um `nan` numa métrica não leva o cabeçalho nem os outros membros com ele.

    O `nan` é alcançável e não hipotético: `_fraction_under` devolve `nan` quando nenhuma feature
    tem RSD definido, e a serialização entrega-o como `null`.

    **A âncora é a distinção:** o membro indefinido diz *"calculada e indefinida"* e o vizinho do
    MESMO tema continua com o número. Sem os dois lados, um painel que apagasse a moldura inteira
    passaria — os dois estariam vazios, e vazio não contradiz vazio.
    """
    tema = next(g for g in agrupa_metricas_da_qualidade() if g.chave == "dispersao_na_amostra")
    assert len(tema.metricas) >= 2, "âncora: o tema tem um membro só e o par abaixo é vácuo"
    indefinida, vizinha = tema.metricas[0], tema.metricas[1]

    painel.mostra_qualidade(
        {
            "computed": [indefinida, vizinha],
            indefinida: float("nan"),
            vizinha: 0.25,
            "n_features_rsd_defined": 4,
            "rsd_variant": "plain",
            "rsd_estimator": "in_sample",
        }
    )
    visivel = painel.texto_visivel()

    assert visivel[f"__tema__{tema.chave}"] == tema.titulo, (
        "o cabeçalho do tema sumiu por causa de uma métrica indefinida"
    )
    assert SEM_VALOR_DEFINIDO in visivel[indefinida], visivel[indefinida]
    assert "25,0%" in visivel[vizinha], (
        f"o vizinho do mesmo tema perdeu o valor: {visivel[vizinha]!r}"
    )
    assert visivel[indefinida] != visivel[vizinha], "âncora: os dois lados do par saíram iguais"


def test_u6_as_tres_sob_argumento_nomeiam_a_aba_no_painel(painel: PainelDeQualidade) -> None:
    """No widget: o texto que nomeia a aba chega ao rótulo, e não fica na constante.

    **Par causal sobre o MESMO painel:** as três sob argumento nomeiam a aba, e uma métrica que
    tem caminho e não depende de argumento continua a dizer só *"não calculada"*. Sem o segundo
    lado, um painel que escrevesse a frase longa em todas passaria.

    **A aba é nomeada com o literal que `principal.ABAS` declara**, e a asserção compara com esse
    literal em vez de com uma redação: trocar o título da aba sem trocar esta frase deixaria o
    painel a mandar quem lê a uma aba que não existe.
    """
    painel.mostra_qualidade({"computed": []})
    visivel = painel.texto_visivel()

    assert METRICAS_SOB_PAR_DE_GRUPOS, "âncora: a classe «sob argumento» está vazia"
    for nome in sorted(METRICAS_SOB_PAR_DE_GRUPOS):
        assert "Análise estatística" in visivel[nome], (
            f"`{nome}` não diz em que aba o par de grupos se escolhe: {visivel[nome]!r}"
        )
        assert "Grupo A do contraste" in visivel[nome], visivel[nome]

    # A âncora de não-vacuidade vai AQUI e ANTES: `not in ""` passa, e string vazia é o
    # que `texto_visivel` devolve para um rótulo fora do layout — a asserção seguinte seria
    # satisfeita por um painel em branco.
    sem_argumento = visivel["frac_features_under_20"]
    assert sem_argumento.startswith(PREFIXO_DO_VALOR), sem_argumento
    assert "Análise estatística" not in sem_argumento, (
        "uma métrica que não depende do par de grupos passou a mandar preencher dois campos"
    )

    # **A aba existe com este nome**, e é isto que impede a frase de envelhecer sozinha.
    assert any(aba.titulo == "Análise estatística" for aba in ABAS), (
        f"a frase nomeia uma aba que a janela não tem: {[aba.titulo for aba in ABAS]}"
    )


def test_u6_a_secao_dedicada_do_plsda_tem_moldura_propria_e_o_titulo_da_tela_web(
    painel: PainelDeQualidade,
) -> None:
    """Quem exigiu isto foi abrir a aba, e nenhum teste o teria achado.

    Enquanto a lista era plana, as duas molduras repetidas do par liam-se como *"a lista
    continua"*. Com as onze dentro de cinco molduras de tema, elas passaram a flutuar fora de
    qualquer grupo — e uma captura da tela mostrava dois quadros órfãos logo abaixo do residual,
    que se liam como defeito de desenho e não como a seção dedicada ao PLS-DA.

    **A duplicação FICA declarada**, e é isto que esta função afirma: o par está na lista de cima,
    sob o tema da validação, **e** nesta seção. O agrupamento não a absorve.
    """
    visivel = painel.texto_visivel()

    assert visivel["__secao_plsda__"] == TITULO_DA_SECAO_DO_PLSDA, (
        f"a seção dedicada do PLS-DA não tem moldura VISÍVEL: {visivel['__secao_plsda__']!r}"
    )
    # **A duplicação, afirmada:** o par está no tema da validação E na seção dedicada.
    tema_do_par = next(
        grupo for grupo in agrupa_metricas_da_qualidade() if grupo.chave == "validacao_do_plsda"
    )
    # **A igualdade virou INTERSEÇÃO no merge, e a razão é de população.** `agrupa_metricas_da_
    # qualidade` percorre as ONZE escalares; `nomes_do_plsda()` devolve os QUATRO nomes de
    # validação, que incluem o par do multiclasse — e esse par não é escalar, logo não entra na
    # grelha temática. Exigir igualdade obrigaria a pôr uma coleção na grelha das onze, o que foi
    # recusado.
    assert set(tema_do_par.metricas) == set(nomes_do_plsda()) & METRICAS_DA_TELA_DE_QUALIDADE, (
        "o tema da validação deixou de mostrar as métricas de PLS-DA que SÃO da grelha"
    )
    assert tema_do_par.metricas, "âncora: o tema está vazio e a comparação acima seria vácua"
    for nome in nomes_do_plsda():
        assert visivel[f"__moldura_plsda__{nome}"], (
            f"`{nome}` perdeu a moldura da seção dedicada: a regra 6 do README põe o score e a "
            "validação lado a lado, e é ali que isso acontece"
        )


# ---------------------------------------------------------------------------
# O PLS-DA nunca sem validação
# ---------------------------------------------------------------------------


def test_u7_o_par_do_plsda_sai_inteiro_ou_nao_sai(painel: PainelDeQualidade) -> None:
    """O PLS-DA na apresentação: o score e a validação, ou nada.

    A guarda de ausência recusa desenhar, e o plano diz isso nas mesmas palavras: *"a resposta
    sem `q2` faz o painel **recusar** desenhar os scores, com mensagem"*. A v1 deste painel escrevia
    sempre os dois, e o teste dela afirmava o oposto.

    **E `nomes_do_plsda()` inclui o SCORE**, não só os exigidos: a v1 iterava `exigidos` e acertava
    por coincidência, porque o par declarado põe o score dentro deles. Trocá-lo por um nome próprio
    fazia o painel deixar de o mostrar em silêncio.
    """
    assert PARES_DE_VALIDACAO_OBRIGATORIA, "âncora: não há par declarado"
    nomes = nomes_do_plsda()
    assert nomes, "âncora: a lista de nomes do par está vazia"
    for score, exigidos in PARES_DE_VALIDACAO_OBRIGATORIA:
        assert score in nomes, f"o SCORE `{score}` não está na lista que o painel escreve"
        for nome in exigidos:
            assert nome in nomes, f"o exigido `{nome}` não está na lista"

    # (a) sem score, nada é escrito — o painel recusa desenhar
    for vazio in ({}, None, {"computed": []}, {"plsda_q2": None}):
        assert painel.mostra_plsda(vazio) == {}, f"o painel desenhou o par sem score: {vazio!r}"

    # (b) com score, o par sai INTEIRO
    escritos = painel.mostra_plsda(
        {
            "computed": ["plsda_q2", "plsda_permutation_p"],
            "plsda_q2": 0.24,
            "plsda_permutation_p": 0.31,
        }
    )
    assert set(escritos) == set(nomes), f"o par saiu incompleto: {sorted(escritos)}"

    visivel = painel.texto_visivel()
    for nome in nomes:
        assert visivel[nome], f"`{nome}` foi escrito e não está VISÍVEL"

    # (c) e os dois valores são DISTINTOS — a validação não é cópia do score
    #
    # Por PAR e não sobre todos os nomes. Com duas famílias, os nomes da que
    # esta execução não pediu saem os dois como `NAO_CALCULADA` — duas sentinelas iguais, que não
    # são «o `p` a ser cópia do `q2`». A propriedade medida é sobre a família que TEM valor, e
    # afirmá-la sobre o conjunto todo a trocaria por outra.
    valores = {nome: visivel[nome].split(": ", 1)[1] for nome in nomes}
    com_valor = [
        (score, exigidos)
        for score, exigidos in PARES_DE_VALIDACAO_OBRIGATORIA
        if valores.get(score) not in (None, NAO_CALCULADA, SEM_CAMINHO)
    ]
    assert com_valor, (
        f"nenhuma família do PLS-DA tem valor nesta execução ({valores}): a asserção abaixo "
        "passaria por vacuidade"
    )
    for _score, exigidos in com_valor:
        do_par = {nome: valores[nome] for nome in exigidos}
        assert len(set(do_par.values())) == len(do_par), (
            f"dois membros do par mostram o MESMO valor: {do_par}. A revisão adversarial "
            f"construiu exactamente isto — o `p` a ser cópia do `q2` —, que é o falso positivo "
            f"que a regra 6 do README existe para não esconder"
        )
    assert valores["plsda_q2"] == "0,240"
    assert valores["plsda_permutation_p"] == "0,310"
    # E a família que a execução NÃO pediu sai declarada, nunca em branco.
    assert valores["plsda_multiclass_q2"] == NAO_CALCULADA


def test_u8_a_lista_de_pares_ALCANCA_o_multiclasse_e_o_p_sai_com_a_medida(  # noqa: N802
    painel: PainelDeQualidade,
) -> None:
    """A segunda metade do cenário obrigatório do multiclasse.

    A primeira metade está em `tests/test_plsda_ligado.py::
    test_MEIO_PAR_MULTICLASSE_e_recusado_nas_duas_direcoes`: o **mesmo** relatório — medida
    presente, `p` ausente — faz `QualityReport` levantar. Aqui a pergunta é a da tela, e a resposta
    dela não é levantar: é **nunca escrever a medida sozinha**.

    Antes a lista de pares conhecia só o par binário, e um relatório multiclasse passava sem tocar
    nenhuma das duas guardas: a medida chegaria à tela e o `p` não teria por onde aparecer — não
    como «não calculada», mas como nada, que é o PLS-DA sem validação.

    **As duas direções, e a segunda é a que o par causal exige:** com a medida presente, os dois
    nomes da família saem; sem ela, nenhum dos dois — a tela **recusa** desenhar, como já faz para
    o binário.
    """
    nomes = nomes_do_plsda()
    assert {"plsda_multiclass_q2", "plsda_multiclass_permutation_p"} <= set(nomes), (
        f"a lista de pares não alcança o multiclasse: {nomes}. Um caminho multiclasse escreveria "
        "a medida sem o p ao lado, e nada o acusaria"
    )

    # (a) a medida veio e o `p` NÃO — a planta do cenário obrigatório
    escritos = painel.mostra_plsda(
        {
            "computed": ["plsda_multiclass_q2"],
            "plsda_multiclass_q2": [0.31, 0.12, -0.04, 0.22],
        }
    )
    assert "plsda_multiclass_q2" in escritos, "âncora: a medida não foi escrita"
    assert "plsda_multiclass_permutation_p" in escritos, (
        f"a medida multiclasse saiu SEM o p ao lado: {escritos}. É o falso positivo servido sem o "
        "número que o desmente — a regra 6 do README reaberta na classe nova"
    )
    assert escritos["plsda_multiclass_permutation_p"] == NAO_CALCULADA, (
        "o p apareceu com valor num relatório que não o traz: a tela inventou o número que a "
        f"guarda existe para exigir — {escritos['plsda_multiclass_permutation_p']!r}"
    )

    # (b) sem medida nenhuma, a tela recusa desenhar — visto passar com a planta REMOVIDA
    assert painel.mostra_plsda({"computed": [], "plsda_multiclass_q2": None}) == {}, (
        "o painel desenhou a família multiclasse sem score: a recusa não acontece"
    )


def test_u7_toda_escrita_no_par_e_dirigida_pelo_laco() -> None:
    """**Portão por AST com lista de permissão: nenhuma escrita no par fora do laço.**

    A regra do PLS-DA diz *"não pode existir caminho de código"*, e um teste de comportamento
    cobre os caminhos que alguém pensou em exercitar.

    ## A v1 proibia `plsda["literal"]`, e SETE formas escapavam

    Medidas: `so_um = PARES[0][0]` e depois `plsda[so_um]`; `next(iter(plsda.values()))`;
    `list(plsda.values())[0]`; uma fatia; um `if` dentro do método com `chave = exigidos[0]`.

    > **Proibir uma sintaxe enumera o que se conhece.** Este portão permite **uma** forma: o
    > subscrito cuja chave é a variável do `for … in nomes_do_plsda()`. Toda outra escrita — por
    > literal, por variável de fora do laço, ou por `.values()` — é acusada.

    Cinco plantas, uma por forma de escape, **em processo**.
    """

    def escritas_fora_do_laco(texto: str) -> list[str]:
        arvore = ast.parse(texto)
        achados: list[str] = []

        # As variáveis que um `for … in nomes_do_plsda()` liga — a única chave permitida.
        permitidas: set[str] = set()
        for no in ast.walk(arvore):
            if (
                isinstance(no, ast.For)
                and isinstance(no.iter, ast.Call)
                and isinstance(no.iter.func, ast.Name)
                and no.iter.func.id == "nomes_do_plsda"
                and isinstance(no.target, ast.Name)
            ):
                permitidas.add(no.target.id)

        for no in ast.walk(arvore):
            # (a) subscrito de `plsda`
            if isinstance(no, ast.Subscript):
                alvo = no.value
                nome = (
                    alvo.attr
                    if isinstance(alvo, ast.Attribute)
                    else alvo.id
                    if isinstance(alvo, ast.Name)
                    else ""
                )
                if nome == "plsda" and not (
                    isinstance(no.slice, ast.Name) and no.slice.id in permitidas
                ):
                    achados.append(f"linha {no.lineno} — plsda[{ast.unparse(no.slice)[:24]}]")
            # (b) qualquer acesso a `plsda` que NÃO seja subscrito nem o `for` do laço:
            #     `.values()`, `.items()`, `.get(...)`, `list(...)`.
            if isinstance(no, ast.Attribute) and no.attr in {"values", "items", "get", "keys"}:
                dono = no.value
                nome_do_dono = (
                    dono.attr
                    if isinstance(dono, ast.Attribute)
                    else dono.id
                    if isinstance(dono, ast.Name)
                    else ""
                )
                if nome_do_dono == "plsda":
                    achados.append(f"linha {no.lineno} — plsda.{no.attr}()")
        return sorted(achados)

    fonte = QUALIDADE.read_text(encoding="utf-8")
    assert escritas_fora_do_laco(fonte) == [], (
        f"há escrita no par do PLS-DA fora do laço: {escritas_fora_do_laco(fonte)}"
    )

    plantas = {
        "literal": (
            '\n\ndef _p(self: object) -> None:\n    self.plsda["plsda_q2"].setText("0.24")\n'
        ),
        "variável de fora do laço": (
            "\n\ndef _p(self: object) -> None:\n"
            "    so_um = PARES_DE_VALIDACAO_OBRIGATORIA[0][0]\n"
            '    self.plsda[so_um].setText("0.24")\n'
        ),
        "next(iter(values))": (
            '\n\ndef _p(self: object) -> None:\n    next(iter(self.plsda.values())).setText("x")\n'
        ),
        "list(values)[0]": (
            '\n\ndef _p(self: object) -> None:\n    list(self.plsda.values())[0].setText("x")\n'
        ),
        "ramo com exigidos[0]": (
            "\n\ndef _p(self: object, corpo: dict) -> None:\n"
            "    for _s, exigidos in PARES_DE_VALIDACAO_OBRIGATORIA:\n"
            "        chave = exigidos[0]\n"
            '        self.plsda[chave].setText("x")\n'
        ),
    }
    for rotulo, planta in plantas.items():
        plantado = f"{fonte}{planta}"
        assert plantado != fonte, f"a planta `{rotulo}` NÃO aterrou"
        assert escritas_fora_do_laco(plantado), (
            f"o portão não acusou a planta `{rotulo}`: a asserção da árvore limpa é vácua"
        )


def test_d336_o_aviso_do_plsda_tem_os_tres_desfechos(painel: PainelDeQualidade) -> None:
    """Eram dois e passaram a ser três. A pergunta é *"quantos desfechos este mecanismo pode
    reportar, e eu vi todos?"* — e "dois" era o piso, não o número.

    Ele chamava-se `test_u7_o_aviso_do_plsda_tem_os_dois_desfechos` e a primeira asserção era
    `set(caminhos.values()) == {"nenhum"}`, com a mensagem *"algum membro do par ganhou caminho:
    o aviso deixou de ser verdade e o painel tem de ser revisitado"*. **O par ganhou caminho, o
    aviso deixou de ser verdade, e o painel foi revisitado** — o portão fez exactamente o que a
    própria mensagem dele previa.

    Os três, e cada um pede uma ação diferente de quem opera:

    | Desfecho | Ação |
    |---|---|
    | `SEM_CAMINHO_PARA_PLSDA` | nenhuma — é limitação da ferramenta |
    | `SEM_PLSDA` | preencher os dois campos de grupo e executar outra vez |
    | procedência | ler o Q² ao lado do R²Y, do p e da semente |

    **A injeção de `caminhos` fica, e passou a ser MAIS necessária:** o primeiro desfecho já não
    acontece em produção, então sem ela ele ficaria inalcançável — que é a condição exacta que a
    versão anterior deste teste registrava sobre o *outro* ramo.
    """
    # Sobre `METRICAS_DO_PLSDA` e não sobre `nomes_do_plsda()`. A segunda
    # passou a incluir a família multiclasse, que **não** tem entrada em `CAMINHO_DA_METRICA` — não
    # é das onze e nenhuma rota a produz. A pergunta deste desfecho é sobre a tabela de caminhos, e
    # `aviso_do_plsda` foi corrigido para a fazer sobre o mesmo conjunto.
    caminhos = {n: CAMINHO_DA_METRICA.get(n) for n in sorted(METRICAS_DO_PLSDA)}
    assert caminhos, "âncora: o conjunto com caminho está vazio"
    assert set(caminhos.values()) == {"api"}, (
        f"o par do PLS-DA perdeu o caminho: {caminhos}. `POST /execucoes` calcula os dois quando o "
        "par de grupos é declarado"
    )

    # Desfecho 1 — só alcançável por injeção, e é por isso que a injeção existe.
    assert aviso_do_plsda({n: "nenhum" for n in METRICAS_DO_PLSDA}) == SEM_CAMINHO_PARA_PLSDA

    # Desfecho 2 — o do painel recém-aberto, e o de toda execução sem par declarado.
    assert painel.texto_visivel()["__aviso__"] == SEM_PLSDA
    assert "não declarou o par de grupos" in SEM_PLSDA
    assert aviso_do_plsda(relatorio={"plsda_q2": None}) == SEM_PLSDA

    # Desfecho 3 — o par veio, e a linha traz a PROCEDÊNCIA.
    com_par = aviso_do_plsda(
        relatorio={
            "plsda_q2": 0.24,
            "plsda_r2y": 0.31,
            "plsda_groups": ["ATLL", "NoHTLV"],
            "plsda_cv_folds": 7,
            "plsda_n_permutations": 200,
            "plsda_seed": 42,
        }
    )
    assert "ATLL contra NoHTLV" in com_par, "a linha não diz sobre que contraste o Q² foi medido"
    assert "200 permutações" in com_par and "semente 42" in com_par
    assert len({SEM_CAMINHO_PARA_PLSDA, SEM_PLSDA, com_par}) == 3, "dois desfechos colapsaram"


def test_d336_o_alerta_de_r2y_alto_com_q2_baixo_e_um_par_causal() -> None:
    """A planta cai sobre dado em que as duas coisas diferem.

    O alerta é o achado do domínio a chegar à tela: separação convincente com Q² baixo é o falso
    positivo que a validação do PLS-DA existe para expor. Um `procedencia_do_plsda` que ignorasse a
    distância devolveria o mesmo texto nos dois braços, e o teste de cima continuaria verde.

    Os dois braços diferem **só** no `plsda_r2y`.
    """
    base = {
        "plsda_q2": 0.24,
        "plsda_groups": ["A", "B"],
        "plsda_cv_folds": 7,
        "plsda_n_permutations": 200,
        "plsda_seed": 42,
    }
    proximo = procedencia_do_plsda({**base, "plsda_r2y": 0.31})
    distante = procedencia_do_plsda({**base, "plsda_r2y": 0.95})

    assert "ATENÇÃO" not in proximo, "alertou sobre um modelo que generaliza"
    assert "ATENÇÃO" in distante, "não alertou sobre R²Y 0,95 contra Q² 0,24"
    assert proximo != distante


# ---------------------------------------------------------------------------
# O antes-e-depois
# ---------------------------------------------------------------------------


def test_u7_a_comparacao_le_todos_os_campos_do_esquema_e_a_bijecao_e_derivada() -> None:
    """**Bijeção nas DUAS direções, com o conjunto lido DERIVADO por AST.**

    A v1 tinha uma lista de seis nomes escrita à mão e checava **uma** direção. A revisão
    adversarial provou o resto: um campo inventado — `mediana_depois_corrigida` — passava, e
    `rsd_variant` e `estimador_alternativo` estavam **fora** do texto sem nada acusar.

    A omissão não é frouxa: o RSD carrega dois rótulos, `rsd_variant` e `rsd_estimator`, ou não é
    comparável. Duas execuções do mesmo modo sob variantes diferentes dão
    medianas diferentes que sem o rótulo ninguém distingue.
    """
    fonte = QUALIDADE.read_text(encoding="utf-8")
    alvo = next(
        no
        for no in ast.walk(ast.parse(fonte))
        if isinstance(no, ast.FunctionDef) and no.name == "texto_da_comparacao"
    )
    lidos = {
        no.args[0].value
        for no in ast.walk(alvo)
        if isinstance(no, ast.Call)
        and isinstance(no.func, ast.Attribute)
        and no.func.attr == "get"
        and no.args
        and isinstance(no.args[0], ast.Constant)
        and isinstance(no.args[0].value, str)
    } | {
        no.slice.value
        for no in ast.walk(alvo)
        if isinstance(no, ast.Subscript)
        and isinstance(no.slice, ast.Constant)
        and isinstance(no.slice.value, str)
    }

    do_esquema = set(RsdAntesEDepois.model_fields)
    assert lidos and do_esquema, "âncora: um dos dois conjuntos está vazio"

    inventados = lidos - do_esquema
    assert not inventados, (
        f"`texto_da_comparacao` lê campos que `RsdAntesEDepois` NÃO declara: {sorted(inventados)}"
    )
    omitidos = do_esquema - lidos
    assert not omitidos, (
        f"`RsdAntesEDepois` declara campos que a comparação NÃO mostra: {sorted(omitidos)}. Se a "
        f"omissão for deliberada, declare-a aqui com o motivo — `rsd_variant` foi omitido "
        f"na v1, e sem ele o RSD não é comparável"
    )


def test_a_ressalva_do_estimador_chega_RENDERIZADA_e_a_quebra_sobrevive(  # noqa: N802
    painel: PainelDeQualidade,
) -> None:
    """Os oito marcadores de `RESSALVA_DO_ESTIMADOR` chegavam à tela LITERAIS.

    A constante é da API e traz quatro `**` e quatro crases. Este rótulo era `AutoText`, que
    escolhe entre texto simples e HTML: markdown **não** está na escolha. Um conserto anterior tirou
    as crases das quatro interpolações desta mesma expressão de retorno e deixou a constante
    concatenada ao lado delas — a irmã dentro da própria linha.

    ## As duas metades não se separam, e a segunda foi medida antes de escrita

    Pôr `MarkdownText` sem trocar o separador removeria os oito marcadores **e perderia a quebra**:
    em markdown uma única `\\n` é quebra *soft* e colapsa num espaço. Por isso o oráculo é duplo —
    zero marcadores E dois blocos —, e a planta abaixo é o separador de uma `\\n` só, que dá **um**
    bloco sobre exactamente o mesmo texto.
    """
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QTextDocument

    assert painel.comparacao.textFormat() == Qt.TextFormat.MarkdownText, (
        "o rótulo do antes-e-depois voltou a `AutoText`, e os `**` e as crases de "
        "`RESSALVA_DO_ESTIMADOR` voltam a aparecer literais na tela"
    )

    texto = painel.mostra_comparacao(
        {
            "mediana_antes": 0.149,
            "mediana_depois": 0.034,
            "correcao": "loess",
            "rsd_variant": "outlier_removed",
            "rsd_estimator": "in_sample",
            "estimador_alternativo": "leave_one_qc_out",
            "n_features_comparadas": 400,
            "n_features": 500,
        }
    )
    # Âncora de não-vacuidade, e vai ANTES: sem os marcadores na fonte, a asserção de
    # que eles somem da renderização passaria sobre um texto que nunca os teve.
    assert RESSALVA_DO_ESTIMADOR.count("**") == 4 and RESSALVA_DO_ESTIMADOR.count("`") == 4, (
        "âncora: a constante da API deixou de trazer marcação, e este cenário deixa de medir a "
        f"renderização — `**`: {RESSALVA_DO_ESTIMADOR.count('**')}, crases: "
        f"{RESSALVA_DO_ESTIMADOR.count('`')}"
    )
    assert "**" in texto and "`" in texto, "âncora: a marcação não chegou ao texto do rótulo"

    documento = QTextDocument()
    documento.setMarkdown(texto)
    renderizado = documento.toPlainText()
    assert "**" not in renderizado and "`" not in renderizado, (
        f"a marcação chegou LITERAL à tela: {renderizado!r}"
    )
    assert documento.blockCount() == 2, (
        f"a linha dos números e a ressalva colapsaram em {documento.blockCount()} bloco(s). O "
        "separador de `texto_da_comparacao` voltou a ser uma `\\n` só, que em markdown é quebra "
        "soft e vira um espaço"
    )

    # A PLANTA, sobre exactamente o mesmo texto: uma `\n` só onde hoje há duas.
    com_uma_quebra = QTextDocument()
    com_uma_quebra.setMarkdown(texto.replace("\n\n", "\n"))
    assert com_uma_quebra.toPlainText() != renderizado, (
        "a planta não separou: com uma quebra só o texto renderizado sai igual, então a asserção "
        "de dois blocos acima passaria com qualquer separador"
    )
    assert com_uma_quebra.blockCount() == 1, (
        f"a planta não separou: com uma `\\n` só o Qt deu {com_uma_quebra.blockCount()} blocos, "
        "e não 1 — o mecanismo do colapso mudou e este cenário deixou de medi-lo"
    )


def test_u7_a_comparacao_traz_a_ressalva_e_a_populacao(painel: PainelDeQualidade) -> None:
    """O par de medianas nunca sai sem o estimador, a variante e a população.

    `RESSALVA_DO_ESTIMADOR` é a constante da API, e não uma frase escrita no painel.
    `rsd_estimator` in-sample favorece o LOESS por construção, e um par de medianas sem isso convida
    a uma conclusão inválida.

    **A população sai como `«N de M»` e não como dois números soltos:** a revisão adversarial trocou
    `n_features_comparadas` por `n_features` e a v1 passava, porque ela afirmava os dois valores
    separadamente. A ordem é o conteúdo.
    """
    assert painel.mostra_comparacao(None) == SEM_COMPARACAO
    assert painel.mostra_comparacao({}) == SEM_COMPARACAO
    assert painel.mostra_comparacao({"mediana_antes": 1.0}) == SEM_COMPARACAO, (
        "meia comparação saiu como comparação"
    )

    texto = painel.mostra_comparacao(
        {
            "mediana_antes": 0.149,
            "mediana_depois": 0.034,
            "correcao": "loess",
            "rsd_variant": "outlier_removed",
            "rsd_estimator": "in_sample",
            "estimador_alternativo": "leave_one_qc_out",
            "n_features_comparadas": 400,
            "n_features": 500,
        }
    )
    assert texto != SEM_COMPARACAO, "âncora: a comparação completa saiu como ausência"
    assert texto == painel.texto_visivel()["__comparacao__"], (
        "o texto devolvido não é o que está VISÍVEL — a revisão adversarial construiu um "
        "painel que devolvia e não escrevia, e a v1 passava"
    )

    assert "400 de 500 features" in texto, (
        f"a população não saiu como «N de M» — dois números soltos não têm ordem: {texto}"
    )
    # As agulhas são a prosa e não o valor cru, e a irmã é a linha do valor. Esta linha
    # dizia *"variante outlier_removed, estimador in_sample (alternativa: leave_one_qc_out)"*, a
    # dois centímetros das quatro linhas de valor que diziam o mesmo par — mesma classe, mesma
    # aba. Os TRÊS rótulos continuam; o que muda é a língua.
    #
    # **A âncora da distinção vem ANTES**: com as três prosas iguais, o laço seria satisfeito por
    # uma composição que escrevesse a mesma frase três vezes.
    variante = prosa_da_variante_de_rsd("outlier_removed")
    usado = prosa_do_estimador_de_rsd("in_sample")
    alternativo = prosa_do_estimador_de_rsd("leave_one_qc_out")
    assert len({variante, usado, alternativo}) == 3, (
        f"duas das três prosas colapsaram: {(variante, usado, alternativo)}"
    )
    for agulha in ("loess", variante, usado, alternativo):
        assert agulha in texto, f"`{agulha}` não saiu na comparação"
    # E os valores crus saíram — as duas que restam vivem dentro de `RESSALVA_DO_ESTIMADOR`, que
    # é a constante da API e NÃO é reescrita daqui.
    sem_a_ressalva = texto.replace(RESSALVA_DO_ESTIMADOR, "")
    assert "outlier_removed" not in sem_a_ressalva and "in_sample" not in sem_a_ressalva, (
        f"um valor `snake_case` voltou à linha do antes-e-depois: {sem_a_ressalva!r}"
    )
    assert RESSALVA_DO_ESTIMADOR in texto, "a ressalva do estimador não é a constante da API"


def test_u7_uma_mediana_indefinida_nao_e_ausencia_de_comparacao() -> None:
    """`None` numa mediana é *"pedi e não deu"*, e `SEM_COMPARACAO` é *"não pedi"*.

    `api/rotas.py` compõe `float(np.median(...)) if comum.any() else float("nan")`, e pydantic
    serializa não finito como `null` — então o estado é **alcançável**. `prosa_do_numero` existe
    para isso, e colapsá-lo aqui apagaria a diferença entre dois fatos com ações opostas.

    **A âncora é a distinção tripla**: ausência, indefinido de um lado, e dois números.
    """
    base = {
        "correcao": "none",
        "rsd_variant": "outlier_removed",
        "rsd_estimator": "in_sample",
        "estimador_alternativo": "leave_one_qc_out",
        "n_features_comparadas": 0,
        "n_features": 500,
    }
    ausente = texto_da_comparacao(None)
    indefinido = texto_da_comparacao({**base, "mediana_antes": None, "mediana_depois": None})
    definido = texto_da_comparacao({**base, "mediana_antes": 0.1, "mediana_depois": 0.2})

    assert len({ausente, indefinido, definido}) == 3, (
        f"dois dos três estados colapsaram: {(ausente, indefinido, definido)}"
    )
    assert ausente == SEM_COMPARACAO
    assert "indefinido" in indefinido, f"o `None` não saiu como indefinido: {indefinido}"
    assert indefinido != SEM_COMPARACAO, (
        "«pedi e não deu» saiu como «não pedi» — apaga a diferença entre dois fatos opostos"
    )


def test_u7_a_comparacao_le_antes_no_campo_de_antes(painel: PainelDeQualidade) -> None:
    """A planta troca o antes pelo depois, sobre dado em que as duas medianas DIFEREM.

    O defeito: com `correction=none` a matriz corrigida **é** a de
    entrada, então trocar `antes` pela corrigida deixa a árvore **verde**.

    **Medido em 23/08/2026 contra a rota real, `neg_c18`:** `none` dá
    0,15503551210769762 nos dois lados; `loess` dá 0,14923231113959795 e 0,03378924918245431 — 4,4
    vezes.

    A âncora vem primeiro: os dois valores diferem, e nenhum é sufixo do outro.
    """
    antes, depois = 0.14923231113959795, 0.03378924918245431
    assert antes != depois, "âncora: os dois valores são iguais e a planta seria vácua"

    base = {
        "correcao": "loess",
        "rsd_variant": "outlier_removed",
        "rsd_estimator": "in_sample",
        "estimador_alternativo": "leave_one_qc_out",
        "n_features_comparadas": 400,
        "n_features": 500,
    }
    texto = painel.mostra_comparacao({**base, "mediana_antes": antes, "mediana_depois": depois})

    # `prosa_do_numero` formata com quatro casas e vírgula decimal, então as agulhas são as
    # formatadas.
    ag_antes = f"{antes:.4f}".replace(".", ",")
    ag_depois = f"{depois:.4f}".replace(".", ",")
    assert ag_antes != ag_depois, "âncora: as duas agulhas formatadas coincidem"
    assert texto.index(ag_antes) < texto.index(ag_depois), (
        f"o «depois» saiu na posição do «antes»: {texto}"
    )

    trocado = texto_da_comparacao({**base, "mediana_antes": depois, "mediana_depois": depois})
    assert trocado != texto, (
        "ler `mediana_depois` nos dois lados produziu o MESMO texto — a asserção acima é "
        "vácua, e é exactamente o que acontece com `correction=none`"
    )


@pytest.mark.integration
def test_u7_o_antes_e_depois_da_rota_real_com_um_duplo_que_separa(
    painel: PainelDeQualidade, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A mesma planta sobre dado que a rota produz, e por um duplo O(1) em vez do `loess`.

    ## A escolha do duplo é MEDIDA, e a árvore já a tinha feito

    `tests/test_persistencia_sqlite.py::_CorretorQueApagaADispersaoDosQCs` existe com este argumento
    verbatim: *"`loess` no `neg_c18` custa **11,18 s** (2.184 features), e pagar isso no portão de
    commit por uma propriedade de ligação de um argumento é o alvo que a equipe aprende a ignorar.
    Zerar as linhas de QC custa uma atribuição."*

    A primeira versão deste teste corria `loess` de verdade — **28,7 s medidos**, que é a rota
    inteira e não o corretor —, e apresentava a escolha como *"pagar no portão ou marcar `slow`"*,
    omitindo a terceira opção que a árvore já implementava. A revisão de padrões apanhou-o.

    > **O duplo garante divergência por CONSTRUÇÃO e não por sorteio:** com os QCs todos iguais o
    > desvio-padrão é zero e o RSD é **0,0** para toda feature, então `mediana_depois == 0.0` e
    > `mediana_antes > 0.0`. `correction=none` real daria os dois iguais, e a planta seria vácua.

    **QC constante e não QC escalado**, e a diferença é aritmética: RSD é `sd/média`, logo
    **invariante a fator constante** — multiplicar as linhas de QC por dois não mudaria número
    nenhum.
    """
    from dataclasses import replace as _replace

    from sqlalchemy import create_engine

    from refino.api import rotas
    from refino.correction.none import NoneCorrector
    from refino.persistence.config import VARIAVEL_DA_URL, prepara_sqlite
    from refino.persistence.schema import METADATA

    class _ApagaADispersaoDosQCs:
        """O `none` real, mais uma alteração O(1) que zera o RSD dos QCs.

        `__getattr__` delega ao real. `POST /execucoes` passou a GUARDAR a correção para os três
        botões de download, e guardar exige a chave — que sai de `Corrector.identidade()`. Sem a
        delegação, o duplo morria com `AttributeError` **dentro da thread de trabalho**, e a rota
        respondia `500`: um duplo que não implementa a barreira inteira reprova pelo motivo
        errado, e a mensagem fala do duplo e não do que se está a medir.
        """

        def __init__(self) -> None:
            self._real = NoneCorrector()

        def __getattr__(self, nome: str) -> Any:
            """Tudo o que não é a barreira vai ao real."""
            return getattr(self._real, nome)

        def identidade(self) -> str:
            """A identidade é PRÓPRIA, e não a do real — ela é a CHAVE DO CACHE.

            **Par causal medido em 07/09/2026**, com este duplo e três testes de
            `test_ui_consumo.py` no mesmo processo:

            | Quem corre antes | `test_o_par_causal_com_valor_o_corte_e_aplicado` |
            |---|---|
            | ninguém | **passa** |
            | este teste | **reprova** — *"corte não aplicado: 2188 de 2188"* |

            A cadeia: `POST /execucoes` passou a **guardar** a correção
            (`_corrige_guardando`), a chave sai de `Corrector.identidade()`, e delegar o método
            ao real fazia a matriz **plantada** — com o RSD dos QCs zerado — ser guardada debaixo
            da chave do `none` **honesto**. Os `/matrizes/*` de qualquer teste seguinte
            reusavam-na, e o corte por RSD deixava de cortar.

            **Delegar tudo é errado exactamente aqui:** um duplo que mente sobre os VALORES não
            pode dizer-se o mesmo corretor, porque a identidade é o que separa uma correção da
            outra na memória partilhada. O `__getattr__` continua a valer para o resto da
            barreira.
            """
            return "duplo_que_apaga_a_dispersao_dos_qcs()"

        def fit_transform(self, matriz: Any, **kwargs: Any) -> Any:
            real = NoneCorrector().fit_transform(matriz, **kwargs)
            valores = real.matrix.values.copy()
            valores[matriz.qc_mask] = 1.0
            return _replace(real, matrix=real.matrix.with_values(valores, "planta do teste da U7"))

    url = f"sqlite:///{tmp_path / 'refino.db'}"
    monkeypatch.delenv("REFINO_DB_HOST", raising=False)
    monkeypatch.setenv(VARIAVEL_DA_URL, url)
    monkeypatch.setitem(rotas.AVAILABLE_METHODS, "none", _ApagaADispersaoDosQCs)
    prepara_sqlite(url)
    METADATA.create_all(create_engine(url))

    cliente = TestClient(build_app(), raise_server_exceptions=False)
    modo = "neg_c18"
    medidas, lista = measurements_path(modo), sample_list_path(modo)

    # O payload sai do construtor de produção, e foi uma reprovação deste teste que o obrigou.
    #
    # Ele montava o `data` à mão com `**FIXOS_DECLARADOS`, tratando aquela constante como *"tudo o
    # que o painel fixa"*. Os quatro eixos da grade saíram dela — passaram a ser combos —, e a rota
    # devolveu **422**: quatro campos obrigatórios em falta.
    #
    # É o mesmo conserto que `test_desktop_janela.py` levou, e o argumento é este: um dicionário
    # montado à mão pode conter — ou faltar — o que a produção não produz.
    painel_de_pipeline = PainelDePipeline()
    # O nome do estudo vem do WIDGET e não de um parâmetro, como `campos_da_execucao` declara.
    painel_de_pipeline.estudo.setText("u7_antes_e_depois")
    # A caixa «Automático» era desticada aqui: ticada, o payload
    # levava `auto`, a política escolhia o corretor, e o duplo `_ApagaADispersaoDosQCs` — que este
    # teste registou no lugar do `none` — deixava de correr. A caixa saiu, e o payload leva sempre o
    # corretor do seletor.
    disparo = cliente.post(
        "/execucoes",
        files={
            "measurements": (medidas.name, medidas.read_bytes(), "text/csv"),
            "sample_list": (lista.name, lista.read_bytes(), "text/csv"),
        },
        data=campos_da_tela(painel_de_pipeline, modo=modo, correcao="none", n_lotes=1),
    )
    assert disparo.status_code == 202, f"o disparo deu {disparo.status_code}"
    identificador = disparo.json()["identificador"]

    corpo = None
    for _ in range(2000):
        reclamada = cliente.get(rota_do_progresso(identificador))
        if reclamada.status_code != 202:
            corpo = reclamada.json()
            break
    assert corpo is not None, "a execução não terminou"
    assert reclamada.status_code == 200, f"a reclamação deu {reclamada.status_code}: {corpo}"

    comparacao = corpo["rsd_antes_e_depois"]
    antes, depois = comparacao["mediana_antes"], comparacao["mediana_depois"]
    assert antes != depois, (
        f"o duplo não separou as duas medianas ({antes} == {depois}), então a planta é vácua. Ele "
        f"zera o RSD dos QCs por construção — `mediana_depois` devia ser 0,0"
    )
    assert depois == 0.0, f"o duplo devia zerar o RSD dos QCs, e deu {depois}"

    texto = painel.mostra_comparacao(comparacao)
    # Quatro casas e vírgula decimal, que é a forma de `prosa_do_numero`.
    ag_antes = f"{antes:.4f}".replace(".", ",")
    ag_depois = f"{depois:.4f}".replace(".", ",")
    assert texto.index(ag_antes) < texto.index(ag_depois), (
        f"o «depois» saiu na posição do «antes»: {texto}"
    )
    # Os dois rótulos da rota REAL chegam ao painel, e chegam em PORTUGUÊS. A agulha é
    # derivada do que a rota devolveu, e não de um literal: é ela que liga o campo da resposta à
    # frase da tela, e é por isso que ela continua a ser a resposta e não uma cópia da constante.
    assert prosa_do_estimador_de_rsd(comparacao["rsd_estimator"]) in texto, texto
    assert prosa_da_variante_de_rsd(comparacao["rsd_variant"]) in texto, texto
    # **E a marca do desconhecido NÃO saiu**, que é a âncora que separa isto de uma tradução que
    # devolvesse o valor cru embrulhado: a rota devolve rótulos do vocabulário fechado.
    assert "sem tradução nesta tela" not in texto, (
        f"a rota devolveu um rótulo de RSD que a tela não traduz: {comparacao}"
    )

    # E o corpo inteiro atravessa a janela até os três widgets.
    assert corpo.get("qualidade"), "âncora: a resposta não trouxe o relatório de qualidade"


# ---------------------------------------------------------------------------
# A LIGAÇÃO — o painel é alimentado, e não só composto
# ---------------------------------------------------------------------------


def test_u7_a_janela_alimenta_o_painel_de_qualidade(qtbot: Any) -> None:
    """**O P0 que os dois revisores acharam:** o painel era construído e nunca alimentado.

    `grep` sobre `src/` achava `mostra_qualidade`, `mostra_plsda` e `mostra_comparacao` **só em
    testes**. A aba mostrava onze rótulos congelados para sempre, com 100% de cobertura a dizer o
    contrário.

    **A âncora é o estado ANTES:** os rótulos estão no estado de nascimento, e é a mudança que prova
    a ligação. Um teste que só olhasse o depois passaria num painel semeado com os valores.
    """
    from refino.desktop.janela.principal import JanelaPrincipal
    from refino.interface.cliente import Resposta

    janela = JanelaPrincipal(TestClient(build_app(), raise_server_exceptions=False))
    qtbot.addWidget(janela)

    antes = _valores(janela.qualidade)
    assert antes["qc_rsd_median"] == NAO_CALCULADA, "âncora: o painel já tinha valores"
    assert janela.qualidade.texto_visivel()["__comparacao__"] == SEM_COMPARACAO

    corpo = {
        "run_id": 1,
        # Os TRÊS campos de contexto não são enfeite: eles são o que `asdict(relatorio)` de facto
        # põe na resposta, e sem eles este arnês exercitava o ramo em que a população **não** é
        # declarada — que é o caminho mais curto do formatador e o que menos executa: o valor
        # natural é o que corre menos do caminho.
        "qualidade": {
            "computed": ["qc_rsd_median"],
            "qc_rsd_median": 7.5,
            "rsd_variant": "plain",
            "rsd_estimator": "in_sample",
            "n_features_rsd_defined": 5933,
        },
        "rsd_antes_e_depois": {
            "mediana_antes": 0.2,
            "mediana_depois": 0.1,
            "correcao": "loess",
            "rsd_variant": "outlier_removed",
            "rsd_estimator": "in_sample",
            "estimador_alternativo": "leave_one_qc_out",
            "n_features_comparadas": 400,
            "n_features": 500,
        },
    }
    janela._recebe(
        "execucao", Resposta(codigo=200, ok=True, causa=None, mensagem="OK", corpo=corpo)
    )

    depois = _valores(janela.qualidade)
    assert depois["qc_rsd_median"] == (
        "750,0% — sobre as 5933 features com RSD definido; sem remoção de outliers, "
        "medido nos próprios QCs do ajuste"
    ), f"o relatório de qualidade não chegou ao painel: {depois['qc_rsd_median']!r}"
    assert "400 de 500" in janela.qualidade.texto_visivel()["__comparacao__"], (
        "o par antes/depois não chegou ao painel"
    )

    # E uma chave que não é da execução NÃO toca o painel.
    janela._recebe(
        "validacao",
        Resposta(codigo=200, ok=True, causa=None, mensagem="OK", corpo={"aceita": True}),
    )
    assert _valores(janela.qualidade)["qc_rsd_median"] == depois["qc_rsd_median"], (
        "uma resposta de outra rota reescreveu o painel de qualidade"
    )

    # E um `202` de progresso também não — ele não tem `qualidade` nenhuma.
    janela._recebe(
        "execucao",
        Resposta(codigo=202, ok=True, causa=None, mensagem="OK", corpo={"estado": "executando"}),
    )
    assert _valores(janela.qualidade)["qc_rsd_median"] == depois["qc_rsd_median"], (
        "o corpo de PROGRESSO reescreveu a qualidade com «não calculada» sobre valores que existem"
    )


def test_u7_o_painel_de_qualidade_esta_composto_na_janela(qtbot: Any) -> None:
    """O painel deixou de ser marca, e a partição continua a fechar."""
    from refino.desktop.janela.principal import ABAS, PAINEIS, JanelaPrincipal

    janela = JanelaPrincipal(TestClient(build_app(), raise_server_exceptions=False))
    qtbot.addWidget(janela)

    # A posição em `PAINEIS` deixou de ser a posição na aba. Aquela tupla é o registro
    # dos painéis; quem diz qual aba mostra o quê é `ABAS`, e desde o agrupamento por função os
    # índices não coincidem. Derivado da estrutura, e não do numeral: uma aba nova antes desta
    # continua a passar, que é o que separa portão de literal pinado.
    indice = next(i for i, aba_declarada in enumerate(ABAS) if "qualidade" in aba_declarada.paineis)
    aba = janela.abas.widget(indice)
    assert isinstance(aba, PainelDeQualidade), f"o painel ainda é marca: {type(aba).__name__}"
    assert janela.abas.tabText(indice) == "Qualidade do resultado", (
        f"o painel de qualidade está na aba «{janela.abas.tabText(indice)}»"
    )
    assert "qualidade" not in janela._vazios
    assert set(janela._vazios) | set(janela._preenchidos) == {p.chave for p in PAINEIS}
    assert not set(janela._vazios) & set(janela._preenchidos)


def test_u7_o_painel_esta_no_glob_que_os_portoes_da_janela_percorrem() -> None:
    """A asserção de PRESENÇA no conjunto que os dois portões de AST da janela usam.

    **E só ela.** A v1 reimplementava o teste de HTTP com um `f"import {x}" not in fonte`, que é
    estritamente mais fraco que o portão irmão — ele não vê `from httpx import Client` — e a
    docstring da função que ela importava diz por que isso é defeito: *"negativo que reimplementa a
    regra fica verde enquanto a regra apodrece"*.

    O portão real é `test_desktop_janela.py`, por glob. O que falta aqui é garantir que este arquivo
    **está** no glob.

    > E o módulo NOVO entra com ele. `graficos.py` recebeu cerca de
    > metade de `qualidade.py`, e o glob alcança-o por construção; afirmá-lo aqui é o que torna
    > essa cobertura verificada em vez de suposta. **Sete populações derivadas por glob sobre
    > `desktop/janela/*.py` passam a alcançar o módulo novo**, e esta asserção é sobre a que este
    > arquivo nomeia.
    """
    from tests.test_desktop_janela import FONTES_DA_JANELA

    assert FONTES_DA_JANELA, "âncora: o glob está vazio"
    ausentes = [m.name for m in MODULOS_QUE_MONTAM_O_RESULTADO if m not in FONTES_DA_JANELA]
    assert not ausentes, (
        f"estes módulos não estão no glob que os portões da janela percorrem: {ausentes}. O glob "
        f"tem {[p.name for p in FONTES_DA_JANELA]}"
    )


# ---------------------------------------------------------------------------
# «O LOESS zera por construção» não chega à janela
# ---------------------------------------------------------------------------

# O oráculo foi CONTADO antes de ser escrito: sobre os 45 textos de tela de `HEAD`
# (`118690c`) — notas de `TEMAS_DA_QUALIDADE`, ressalvas de `TRADUCAO_DA_METRICA`,
# `RESSALVA_DO_ESTIMADOR`, `DICA_DE_MEDIR`, `SEM_RANKING` e os corpos de `PARA_QUEM_OPERA` — ele
# acusa **3**: a nota do tema «dentro da amostra», a ressalva de `qc_rsd_cv_median` e a dica do
# painel da correção. Na árvore de 25/09/2026, **0** em 46. Contado em 25/09/2026 importando uma
# cópia de `src/` do `HEAD` com o mesmo padrão e a mesma colheita.
LOESS_ZERA = re.compile(r"LOESS[^.]*\bzer(?:a|ar|am)\b", re.IGNORECASE)


def _textos_de_tela_sobre_o_rsd() -> dict[str, str]:
    """Os textos que a janela mostra e que falam do RSD in-sample, pelo nome da fonte."""
    from refino.desktop.recorte import PARA_QUEM_OPERA
    from refino.interface.fluxo import TEMAS_DA_QUALIDADE
    from refino.plots.correcao import DICA_DE_MEDIR, SEM_RANKING

    textos = {f"tema {k}": v.nota for k, v in TEMAS_DA_QUALIDADE.items()}
    textos |= {f"métrica {k}": v.ressalva for k, v in TRADUCAO_DA_METRICA.items()}
    textos |= {
        "RESSALVA_DO_ESTIMADOR": RESSALVA_DO_ESTIMADOR,
        "DICA_DE_MEDIR": DICA_DE_MEDIR,
        "SEM_RANKING": SEM_RANKING,
    }
    textos |= {f"recorte {k}": corpo for k, (_titulo, corpo) in PARA_QUEM_OPERA.items()}
    return textos


def test_item90_nenhum_texto_de_tela_diz_que_o_LOESS_zera_o_rsd() -> None:  # noqa: N802
    """O «zera» é do `batch_loess` GOLDEN do R, e a janela corre o `loess` desta ferramenta.

    Medido em 25/09/2026 (`plots/correcao.py`, cabeçalho): a mediana in-sample do nosso `loess` vai
    de 3,38% (`neg_c18`) a 84,0% (`pos_hilic`) — nenhuma é zero. A conclusão que o texto guarda
    sobrevive: in-sample favorece quem ajusta a curva, e não ordena corretores.

    **O par causal são as três frases de `HEAD`, verbatim**: o padrão tem de acusar cada uma, senão
    a asserção de ausência de cima passaria por o padrão não casar nada.
    """
    textos = _textos_de_tela_sobre_o_rsd()
    assert len(textos) >= 40, f"âncora: só {len(textos)} textos — a colheita falhou"
    acusados = sorted(k for k, v in textos.items() if LOESS_ZERA.search(" ".join(v.split())))
    assert acusados == [], f"textos de tela que ainda dizem que o LOESS zera o RSD: {acusados}"

    de_head = (
        "Medido dentro da amostra, o RSD favorece quem ajustou a curva através dos próprios QCs "
        "que ele mede — o LOESS zera-o por construção.",
        "o RSD medido dentro da amostra premia quem ajusta a curva através dos próprios QCs que a "
        "métrica mede, e o LOESS o zera por construção",
        "método que ajusta modelo, porque in-sample o LOESS o zera por construção e o ranking "
        "inverte",
    )
    for frase in de_head:
        assert LOESS_ZERA.search(frase), f"o padrão não acusa a frase de `HEAD`: {frase!r}"


# ---------------------------------------------------------------------------
# A normalidade por variável e por grupo, na aba «Qualidade do resultado»
# ---------------------------------------------------------------------------


def _raiz_da_normalidade() -> dict[str, Any]:
    """Uma raiz `normalidade` REAL: a função do núcleo, passada pelo esquema da resposta.

    Construída e não lida dos arquivos do laboratório: o que estes testes
    afirmam é o desenho da seção — os três estados, a ordem do limite, a gravação —, e não uma
    contagem observada. O fio é o do esquema, para uma chave lida que ele não declarasse aparecer.
    """
    import numpy as np

    from refino.api.rotas import _resumo_da_normalidade
    from refino.domain.matrix import RefinoMatrix
    from refino.domain.metadata import SampleMetadata
    from refino.statistics.normalidade import normality_by_group

    rng = np.random.default_rng(20261106)
    valores = np.exp(rng.normal(size=(14, 30)))
    grupos = ["A"] * 6 + ["B"] * 5 + ["QC"] * 3
    tipos = ["sample"] * 11 + ["qc"] * 3
    matriz = RefinoMatrix(
        values=valores,
        feature_names=tuple(f"f{j}" for j in range(30)),
        samples=tuple(
            SampleMetadata(name=f"S{i}", injection_order=i + 1, sample_type=t, group=g)  # type: ignore[arg-type]
            for i, (g, t) in enumerate(zip(grupos, tipos, strict=True))
        ),
        mode="sintetico",
    )
    imputadas = np.zeros(valores.shape, dtype=bool)
    imputadas[0, :4] = True
    raiz: dict[str, Any] = _resumo_da_normalidade(
        normality_by_group(matriz, imputadas)
    ).model_dump()
    return raiz


def test_d663_a_secao_nasce_em_repouso_com_o_limite_antes_dos_numeros(painel: Any) -> None:
    """O título, o limite e a frase de repouso, no layout e nessa ordem — e o botão desligado.

    **A ordem é o requisito**: o resumo diz quantas variáveis rejeitam, e lido sem o limite ele
    leva à conclusão inversa — que as outras são normais. O oráculo é a posição no layout da
    moldura, e não a ordem em que os atributos foram criados.
    """
    from refino.interface.fluxo import (
        LIMITE_DA_NORMALIDADE,
        NORMALIDADE_EM_REPOUSO,
        TITULO_DA_NORMALIDADE,
    )

    visivel = painel.texto_visivel()
    assert visivel["__secao_normalidade__"] == TITULO_DA_NORMALIDADE
    assert visivel["__limite_normalidade__"] == LIMITE_DA_NORMALIDADE
    assert visivel["__normalidade__"] == NORMALIDADE_EM_REPOUSO
    disposicao = painel.secao_da_normalidade.layout()
    limite = disposicao.indexOf(painel.limite_da_normalidade)
    resumo = disposicao.indexOf(painel.resumo_da_normalidade)
    assert 0 <= limite < resumo, f"o limite está na posição {limite} e o resumo na {resumo}"
    assert "não rejeitou» não é «é normal" in LIMITE_DA_NORMALIDADE
    assert not painel.botao_da_tabela_da_normalidade.isEnabled(), (
        "sem execução não há tabela, e o botão gravaria nada"
    )


def test_d663_os_tres_estados_da_secao_e_a_volta_ao_repouso(painel: Any) -> None:
    """Repouso, «não pedida» e o resumo — e as duas ausências dizem coisas DIFERENTES.

    **O par causal tem as duas pontas:** a raiz chega e o texto muda para o resumo dos grupos; um
    corpo vazio, que é o que a janela passa depois de um erro, devolve o repouso e desliga o botão.
    """
    from refino.interface.fluxo import NORMALIDADE_EM_REPOUSO, NORMALIDADE_NAO_PEDIDA

    assert NORMALIDADE_EM_REPOUSO != NORMALIDADE_NAO_PEDIDA
    assert painel.mostra_normalidade({"run_id": 1, "normalidade": None}) == NORMALIDADE_NAO_PEDIDA
    assert painel.texto_visivel()["__normalidade__"] == NORMALIDADE_NAO_PEDIDA
    assert not painel.botao_da_tabela_da_normalidade.isEnabled()

    texto = painel.mostra_normalidade({"run_id": 1, "normalidade": _raiz_da_normalidade()})
    assert texto.startswith("**A** (6 amostras)") and "**B** (5 amostras)" in texto, texto
    assert "**QC**" not in texto, "o QC apareceu como grupo no resumo"
    assert painel.texto_visivel()["__normalidade__"] == texto
    assert painel.botao_da_tabela_da_normalidade.isEnabled()

    assert painel.mostra_normalidade({}) == NORMALIDADE_EM_REPOUSO
    assert not painel.botao_da_tabela_da_normalidade.isEnabled()


def test_d663_a_tabela_sai_em_csv_pelo_dialogo_injetado(qtbot: Any, tmp_path: Path) -> None:
    """Gravar escreve o CSV de `csv_da_normalidade`; cancelar não escreve; a falha diz que falhou.

    Os três desfechos, e cada um com o seu sinal: o destino devolvido, `None` sem
    arquivo, e `None` com a frase da falha.
    """
    from refino.interface.fluxo import csv_da_normalidade

    destinos: list[Path | None] = []
    painel = PainelDeQualidade(salvador=lambda _nome: destinos.pop(0))
    qtbot.addWidget(painel)
    raiz = _raiz_da_normalidade()
    painel.mostra_normalidade({"run_id": 1, "normalidade": raiz})

    destino = tmp_path / "tabela.csv"
    destinos.append(destino)
    assert painel.salva_tabela_da_normalidade() == destino
    # Os BYTES e não `read_text`, que traduziria o `\r\n` do arquivo em `\n` e esconderia o fim de
    # linha que a planilha lê.
    assert destino.read_bytes().decode("utf-8") == csv_da_normalidade(
        raiz["features"], raiz["grupos"]
    )
    assert "Tabela salva em" in painel.aviso_da_tabela_da_normalidade.text()

    destinos.append(None)
    assert painel.salva_tabela_da_normalidade() is None

    destinos.append(tmp_path / "pasta-que-nao-existe" / "tabela.csv")
    assert painel.salva_tabela_da_normalidade() is None
    assert "Não foi possível salvar" in painel.aviso_da_tabela_da_normalidade.text()


def test_d663_o_dialogo_de_producao_traduz_o_cancelamento(
    qtbot: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """String vazia do `QFileDialog` é o cancelamento, e vira `None`, nunca o diretório corrente.

    **Pelo painel sem salvador injetado**, que é o caminho de produção: o diálogo é o de
    `arquivos.salvador_por_dialogo`, e o que este teste afirma é que o painel lhe passa
    o título do botão e o filtro de CSV — a única diferença que a cópia apagada tinha.
    """
    from PySide6.QtWidgets import QFileDialog

    from refino.interface.fluxo import (
        NOME_DA_TABELA_DA_NORMALIDADE,
        ROTULO_DA_TABELA_DA_NORMALIDADE,
    )

    pedidos: list[tuple[str, str, str]] = []
    respostas = [("", ""), (str(tmp_path / "escolhido.csv"), "CSV (*.csv)")]

    def dialogo(_pai: Any, titulo: str, sugerido: str, filtro: str) -> tuple[str, str]:
        pedidos.append((titulo, sugerido, filtro))
        return respostas.pop(0)

    monkeypatch.setattr(QFileDialog, "getSaveFileName", dialogo)
    painel = PainelDeQualidade()
    qtbot.addWidget(painel)
    painel.mostra_normalidade({"run_id": 1, "normalidade": _raiz_da_normalidade()})

    assert painel.salva_tabela_da_normalidade() is None
    assert not (tmp_path / "escolhido.csv").exists()
    assert painel.salva_tabela_da_normalidade() == tmp_path / "escolhido.csv"
    assert (tmp_path / "escolhido.csv").is_file()
    esperado = (
        ROTULO_DA_TABELA_DA_NORMALIDADE.rstrip("…"),
        NOME_DA_TABELA_DA_NORMALIDADE,
        "CSV (*.csv)",
    )
    assert pedidos == [esperado, esperado], pedidos
    # O título sai da mesma expressão no teste e no código; o texto que a refatoração preserva só o
    # literal afirma.
    assert esperado[0] == "Salvar a tabela por variável (CSV)", esperado


def test_d663_a_janela_leva_o_corpo_da_execucao_a_secao_da_normalidade(qtbot: Any) -> None:
    """A ligação: o `200` da execução chega à seção, e o erro seguinte a devolve ao repouso.

    **A âncora é o estado ANTES** — o repouso —, e é a mudança que prova a ligação, pelo argumento
    de `test_u7_a_janela_alimenta_o_painel_de_qualidade`.
    """
    from refino.desktop.janela.principal import JanelaPrincipal
    from refino.interface.cliente import Resposta
    from refino.interface.fluxo import NORMALIDADE_EM_REPOUSO

    janela = JanelaPrincipal(TestClient(build_app(), raise_server_exceptions=False))
    qtbot.addWidget(janela)
    assert janela.qualidade.texto_visivel()["__normalidade__"] == NORMALIDADE_EM_REPOUSO

    corpo = {"run_id": 1, "normalidade": _raiz_da_normalidade()}
    janela._recebe(
        "execucao", Resposta(codigo=200, ok=True, causa=None, mensagem="OK", corpo=corpo)
    )
    assert janela.qualidade.texto_visivel()["__normalidade__"].startswith("**A** (6 amostras)")

    janela._recebe(
        "execucao", Resposta(codigo=500, ok=False, causa="ambiente", mensagem="falhou", corpo={})
    )
    assert janela.qualidade.texto_visivel()["__normalidade__"] == NORMALIDADE_EM_REPOUSO
