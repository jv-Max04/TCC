"""As três figuras de PLS-DA, e a regra 6 do README como CONTRATO em vez de cuidado.

**Não é `test_plsda.py`, que já existe em `tests/unit/` e é do cálculo.** Aquele prova que
`PLSDAResult` não admite construção parcial; este prova que o **fio** e o **desenho** herdam a mesma
propriedade — e que entre os dois não sobrou nenhum caminho que devolva score sem validação.

## O que este arquivo prova, e por que a primeira seção é a mais importante

A regra 6 do README diz que não pode existir caminho de código que retorne scores de PLS-DA sem
`q2` e `permutation_pvalue`. Até esta unidade a regra vivia nas duas pontas, e o meio — a rota, o
corpo da resposta e o construtor da figura — nunca precisou dela, porque os três vetores eram
descartados dentro da rota. A primeira seção é o par causal que fecha o vão em cada camada:

1. o modelo do fio **recusa** a construção sem o par, e aceita com ele;
2. **nenhum outro** modelo de `esquemas.py` carrega score de PLS-DA sem o par — o portão é sobre a
   propriedade, não sobre esta instância;
3. as três cargas do desenho exigem `Validacao` sem padrão, e as três figuras **imprimem** o par.

As três vivem no mesmo arquivo de propósito: um teste vizinho que ligue as pontas pode ser
removido sem que nada ligue os dois.
"""

from __future__ import annotations

import dataclasses
import inspect
import math
import re
import time

import numpy as np
import pytest
from matplotlib.colors import to_hex
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from pydantic import BaseModel, ValidationError
from sklearn.cross_decomposition import PLSRegression

from refino.api import esquemas
from refino.api.esquemas import (
    ResumoDoModeloUmContraResto,
    ResumoDoPlsda,
    ResumoDoPlsdaMulticlasse,
)
from refino.domain.matrix import RefinoMatrix
from refino.domain.metadata import SampleMetadata
from refino.plots import plsda as modulo
from refino.plots.base import MatrizVaziaError
from refino.plots.catalogo import CONSTRUTORES, Vinculo
from refino.plots.pca import Elipse, elipse_de_hotelling
from refino.plots.plsda import (
    ESCORES_DE_VALIDACAO_CRUZADA,
    FEATURES_DO_VIP,
    LIMIAR_DE_REFERENCIA_DO_VIP,
    PRETRATAMENTO,
    ROTULO_DA_REFERENCIA,
    TITULO_DAS_CARGAS,
    TITULO_DO_VIP,
    TITULO_DOS_SCORES,
    CargasDoPlsda,
    EscoresDoPlsda,
    ImportanciaVip,
    PlsdaDesalinhadoError,
    Validacao,
    cargas_do_plsda,
    cores_dos_lados,
    escores_do_plsda,
    expoente_fora_do_rotulo,
    figura_das_cargas_do_plsda,
    figura_do_vip,
    figura_dos_scores_do_plsda,
    frase_curta_da_validacao,
    frase_da_validacao,
    recorte_das_cargas_do_plsda,
    recorte_do_vip,
    recorte_dos_scores_do_plsda,
    rotulo_do_componente_do_plsda,
    selecao_do_vip,
    validacao_do_fio,
    veredito_da_validacao,
    vip_do_plsda,
)
from refino.plots.recorte import RecorteInvalidoError, decimal, milhar
from refino.plots.series import EstiloDoGrupo, EstilosDosGrupos, estilos_dos_grupos
from refino.statistics.plsda import PLSDAResult, _vip, plsda
from tests.arnes_de_figura import (
    confere_numeros,
    confere_rotulos,
    confere_series,
    confere_vista,
    planta_corte_de_eixo,
    planta_eixos_trocados,
    planta_serie_removida,
    planta_sinal_invertido,
    series,
)
from tests.conftest import matrix_from_toserrf

MODO = "pos_hilic"

# O par de grupos do corpus real. **`Asymptomatic` contra `HAMTSP` e não outro**: é o par de maior
# `Q²` dos quatro medidos, e por isso o caso mais favorável que este estudo tem para oferecer — o
# `Q²` continua negativo, e é sobre o melhor caso que a leitura da regra 6 do README tem de valer.
PAR_REAL = ("Asymptomatic", "HAMTSP")

# **O orçamento de redesenho desta rodada, em milissegundos — e ele NÃO é asserção.**
#
# A primeira versão deste arquivo chamava-o `TETO_DE_REDESENHO_MS` e escrevia
# `assert ms < 5 * TETO`. **O portão de `tests/test_orcamento_contrato.py` reprovou, e está
# certo:** comparar um teto de orçamento fora de `tests/orcamento.py::afirma_orcamento` é
# Exactamente o estado em que o critério viveu — uma comparação de tempo de parede escrita à mão,
# Que sob contenção reprova sem distinguir regressão de máquina ocupada.
#
# **Passar pelo funil também seria errado aqui.** `afirma_orcamento` existe para o orçamento da
# Com `skip` declarado acima de `load_per_core` 0,5; um redesenho de figura não é esse
# contrato, e pendurá-lo ali daria ao número uma autoridade que ele não tem.
#
# **Então a forma é a: MEDIR e DECLARAR.** O número vive na docstring do teste que o mede, o
# teste afirma uma **propriedade da construção** — que o texto domina o custo — e nenhuma asserção
# de tempo de parede entra na suíte.
ORCAMENTO_DECLARADO_MS = 100.0


# ---------------------------------------------------------------------------
# As fixturas
# ---------------------------------------------------------------------------


def matriz_sintetica(
    *, por_grupo: int = 10, n_features: int = 8, separacao: float = 4.0, seed: int = 20260911
) -> RefinoMatrix:
    """Duas classes com separação CONHECIDA nas três primeiras features.

    As três primeiras colunas deslocam-se de `separacao` no segundo grupo e as outras são ruído: o
    oráculo do caminho feliz é que o VIP eleja as três plantadas, e que os scores separem.
    """
    gerador = np.random.default_rng(seed)
    valores = gerador.normal(10.0, 1.0, size=(2 * por_grupo, n_features))
    valores[por_grupo:, :3] += separacao
    amostras = tuple(
        SampleMetadata(
            name=f"a{i:02d}",
            injection_order=i + 1,
            sample_type="sample",
            group="controle" if i < por_grupo else "caso",
        )
        for i in range(2 * por_grupo)
    )
    return RefinoMatrix(
        values=valores,
        feature_names=tuple(f"f{j:02d}" for j in range(n_features)),
        samples=amostras,
        mode=MODO,
    )


def resultado_de(matriz: RefinoMatrix, par: tuple[str, str], *, cv_folds: int = 4) -> PLSDAResult:
    return plsda(matriz, par, n_components=2, cv_folds=cv_folds, n_permutations=20, seed=11)


def validacao_de(resultado: PLSDAResult, *, cv_folds: int = 4) -> Validacao:
    return Validacao(
        q2=resultado.q2,
        permutation_pvalue=resultado.permutation_pvalue,
        r2y=resultado.r2y,
        n_permutacoes=resultado.n_permutations,
        cv_folds=cv_folds,
        n_componentes=resultado.n_components,
    )


def cargas_de(matriz: RefinoMatrix, resultado: PLSDAResult, val: Validacao) -> CargasDoPlsda:
    return cargas_do_plsda(
        comp1=resultado.loadings[:, 0],
        comp2=resultado.loadings[:, 1],
        features=list(matriz.feature_names),
        validacao=val,
    )


def escores_de(
    matriz: RefinoMatrix, resultado: PLSDAResult, par: tuple[str, str], val: Validacao
) -> EscoresDoPlsda:
    """Os escores que a figura desenha: os de VALIDAÇÃO CRUZADA, como os lê a janela."""
    return escores_do_plsda(
        comp1=resultado.cv_scores[:, 0],
        comp2=resultado.cv_scores[:, 1],
        grupos=[a.group for a in matriz.samples if a.group in par],
        grupo_positivo=par[1],
        grupo_negativo=par[0],
        validacao=val,
    )


# O rótulo da linha de cada região, escrito AQUI por extenso e não importado: é o que a legenda tem
# de dizer, e um rótulo emprestado do módulo sob teste mudaria junto com ele.
ROTULO_DA_REGIAO_NO_TESTE = "Região de predição 95% (T²) — {grupo}"


def esperadas_dos_scores(escores: EscoresDoPlsda) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """As séries que o gráfico de escores tem de desenhar: os pontos de cada grupo e a região dele.

    **A região é recalculada aqui sobre os pontos DAQUELE grupo**, e a população é o que se afirma:
    a função é a mesma do construtor, e o que o teste fixa é sobre que pontos ela corre. Um grupo
    sem região — menos de três pontos, ou pontos alinhados — não tem linha a esperar.
    """
    esperadas: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for grupo, papel in (
        (escores.negativo, "classe de referência, y = 0"),
        (escores.positivo, "classe positiva, y = 1"),
    ):
        mascara = escores.mascara(grupo)
        quantas = int(mascara.sum())
        plural = "amostra" if quantas == 1 else "amostras"
        esperadas[f"{grupo} — {quantas} {plural} ({papel})"] = (
            escores.x[mascara],
            escores.y[mascara],
        )
        regiao = elipse_de_hotelling(escores.x[mascara], escores.y[mascara])
        if isinstance(regiao, Elipse):
            esperadas[ROTULO_DA_REGIAO_NO_TESTE.format(grupo=grupo)] = regiao.contorno()
    return esperadas


def vip_de(matriz: RefinoMatrix, resultado: PLSDAResult, val: Validacao) -> ImportanciaVip:
    return vip_do_plsda(vip=resultado.vip, features=list(matriz.feature_names), validacao=val)


@pytest.fixture(scope="module")
def sintetico() -> dict[str, object]:
    """Um PLS-DA sintético de separação conhecida, com as três cargas prontas."""
    matriz = matriz_sintetica()
    par = ("controle", "caso")
    resultado = resultado_de(matriz, par)
    val = validacao_de(resultado)
    return {
        "matriz": matriz,
        "par": par,
        "resultado": resultado,
        "validacao": val,
        "escores": escores_de(matriz, resultado, par, val),
        "cargas": cargas_de(matriz, resultado, val),
        "vip": vip_de(matriz, resultado, val),
    }


@pytest.fixture(scope="module")
def matriz_real() -> RefinoMatrix:
    """O `pos_hilic` do laboratório. **Escopo de módulo: um PLS-DA real custa ~3,5 s.**"""
    return matrix_from_toserrf(MODO)


def rodape_de(figura: Figure) -> str:
    """O rodapé com o espaço em branco COLAPSADO — e isto não é conveniência.

    `recorte.aplica` quebra a frase em linhas pela largura útil, então uma citação de meia dúzia de
    palavras atravessa a quebra e `in` devolve `False` sobre texto verdadeiro. É o falso negativo de
    Asserção dentro de um teste, e a direção é a perigosa.
    """
    rodapes = [t for t in figura.texts if t.get_va() == "bottom"]
    assert len(rodapes) == 1, [t.get_text() for t in figura.texts]
    return " ".join(rodapes[0].get_text().split())


# ---------------------------------------------------------------------------
# 1. A REGRA 6 DO README — o par causal em cada camada
# ---------------------------------------------------------------------------


CORPO_DO_PLSDA: dict[str, object] = {
    "q2": -0.0125,
    "permutation_pvalue": 0.1095,
    "r2y": 0.3498,
    "n_permutations": 200,
    "cv_folds": 7,
    "n_components": 2,
    "seed": 42,
    "scores_componente_1": [1.0, -1.0, 2.0, -2.0],
    "scores_componente_2": [0.5, -0.5, 1.5, -1.5],
    "scores_cv_componente_1": [0.8, -0.6, 1.1, -1.4],
    "scores_cv_componente_2": [0.2, -0.9, 1.0, -0.4],
    "grupos_por_amostra": ["controle", "controle", "caso", "caso"],
    "grupo_positivo": "caso",
    "grupo_negativo": "controle",
    "n_positivo": 2,
    "n_negativo": 2,
    "features": ["f00", "f01", "f02"],
    "loadings_componente_1": [0.7, -0.2, 0.1],
    "loadings_componente_2": [0.1, 0.6, -0.3],
    "vip": [1.6, 0.8, None],
    "n_features": 3,
}


def test_o_modelo_do_fio_RECUSA_score_sem_q2_e_ACEITA_com_ele() -> None:  # noqa: N802
    """**O par causal da regra 6 do README no fio, e a planta é argumento do construtor.**

    Ponta A: o corpo completo constrói. Ponta B: o MESMO corpo sem `q2` — e depois sem
    `permutation_pvalue`, e sem cada um dos dois vetores de escores de validação cruzada — reprova,
    com o pydantic a nomear o campo em falta. Sem a ponta A, a reprovação da B poderia ser de
    qualquer outro campo.
    """
    completo = ResumoDoPlsda(**CORPO_DO_PLSDA)  # type: ignore[arg-type]
    assert completo.q2 == pytest.approx(-0.0125)
    assert completo.scores_componente_1 == [1.0, -1.0, 2.0, -2.0]

    # **E os escores de validação cruzada, que são os que a figura desenha**: um padrão
    # vazio num deles deixaria o corpo sem o que desenhar, e a janela sem nada a distinguir de um
    # PLS-DA que não correu.
    for ausente in ("q2", "permutation_pvalue", "scores_cv_componente_1", "scores_cv_componente_2"):
        parcial = {k: v for k, v in CORPO_DO_PLSDA.items() if k != ausente}
        with pytest.raises(ValidationError) as erro:
            ResumoDoPlsda(**parcial)  # type: ignore[arg-type]
        assert ausente in str(erro.value), (
            f"o modelo aceitou scores sem {ausente!r}, ou reprovou por outro motivo: {erro.value}"
        )


def _modelo_um_contra_resto(classe: str, q2: float, p: float) -> dict[str, object]:
    return {
        "q2": q2,
        "permutation_pvalue": p,
        "classe": classe,
        "r2y": 0.51,
        "n_positivo": 2,
        "n_negativo": 2,
        "scores_componente_1": [1.0, -1.0, 2.0, -2.0],
        "scores_componente_2": [0.5, -0.5, 1.5, -1.5],
        "loadings_componente_1": [0.7, -0.2, 0.1],
        "loadings_componente_2": [0.1, 0.6, -0.3],
        "vip": [1.6, 0.8, None],
        "n_vip_indefinido": 1,
    }


def _corpo_multiclasse() -> dict[str, object]:
    modelos = [
        _modelo_um_contra_resto("ATLL", 0.31, 0.0476),
        _modelo_um_contra_resto("NoHTLV", 0.12, 0.2381),
    ]
    return {
        "q2": [m["q2"] for m in modelos],
        "permutation_pvalue": [m["permutation_pvalue"] for m in modelos],
        "classes": [m["classe"] for m in modelos],
        "modelos": modelos,
        "n_modelos": 2,
        "n_permutations": 20,
        "cv_folds": 7,
        "n_components": 2,
        "seed": 42,
        "grupos_por_amostra": ["ATLL", "ATLL", "NoHTLV", "NoHTLV"],
        "features": ["f00", "f01", "f02"],
        "n_features": 3,
    }


def test_o_resumo_MULTICLASSE_recusa_desempenho_sem_o_p_e_aceita_com_ele() -> None:  # noqa: N802
    """**A regra 6 do README na classe aberta, e o par causal é o mesmo de `ResumoDoPlsda`.**

    Ponta A: o corpo completo constrói. Ponta B: o MESMO corpo sem `q2` — e depois sem
    `permutation_pvalue` — reprova, nos **dois** modelos: o resumo do topo e o de cada modelo
    um-contra-resto. Sem a ponta A, a reprovação da B poderia ser de qualquer outro campo.

    **Os dois níveis são testados porque são dois caminhos**: quem lê o resumo nunca obtém a lista
    de `q2` sem a de `p`, e quem desce a um modelo nunca obtém `scores` sem os dois escalares.
    """
    completo = ResumoDoPlsdaMulticlasse(**_corpo_multiclasse())  # type: ignore[arg-type]
    assert completo.q2 == [0.31, 0.12]
    assert completo.modelos[0].scores_componente_1 == [1.0, -1.0, 2.0, -2.0]
    assert completo.modelos[0].vip == [1.6, 0.8, None], "o `None` do VIP indefinido não atravessou"

    for ausente in ("q2", "permutation_pvalue"):
        parcial = {k: v for k, v in _corpo_multiclasse().items() if k != ausente}
        with pytest.raises(ValidationError) as erro:
            ResumoDoPlsdaMulticlasse(**parcial)  # type: ignore[arg-type]
        assert ausente in str(erro.value), (
            f"o resumo aceitou desempenho sem {ausente!r}, ou reprovou por outro motivo: "
            f"{erro.value}"
        )

        um = {
            k: v for k, v in _modelo_um_contra_resto("ATLL", 0.31, 0.0476).items() if k != ausente
        }
        with pytest.raises(ValidationError) as erro_do_modelo:
            ResumoDoModeloUmContraResto(**um)  # type: ignore[arg-type]
        assert ausente in str(erro_do_modelo.value)


def test_o_resumo_multiclasse_recusa_listas_DESALINHADAS_dos_modelos() -> None:  # noqa: N802
    """**A redundância do topo é VERIFICADA, e é isso que a impede de ser segunda fonte.**

    `q2` e `permutation_pvalue` repetem o que `modelos` já traz — de propósito, porque é o modelo do
    topo que o oráculo de classe inspeciona. Repetição não verificada diverge no dia em que alguém
    mudar um lado, e a divergência apareceria como `q2[2]` a responder pela classe errada: número
    certo, população errada, e nada levanta.

    **Três plantas, e cada uma quebra uma cláusula diferente:** comprimento, valor, e rótulo de
    classe. Sem a ponta A — o corpo alinhado a construir — as três reprovações poderiam ser de
    qualquer outro campo.
    """
    assert ResumoDoPlsdaMulticlasse(**_corpo_multiclasse()).n_modelos == 2  # type: ignore[arg-type]

    curto = _corpo_multiclasse()
    curto["q2"] = [0.31]
    with pytest.raises(ValidationError, match="desalinhou"):
        ResumoDoPlsdaMulticlasse(**curto)  # type: ignore[arg-type]

    trocado = _corpo_multiclasse()
    trocado["q2"] = [0.99, 0.12]
    with pytest.raises(ValidationError, match="discordam"):
        ResumoDoPlsdaMulticlasse(**trocado)  # type: ignore[arg-type]

    rotulo = _corpo_multiclasse()
    rotulo["classes"] = ["NoHTLV", "ATLL"]
    with pytest.raises(ValidationError, match="discordam"):
        ResumoDoPlsdaMulticlasse(**rotulo)  # type: ignore[arg-type]


def _carrega_score_de_plsda(nome: str, modelo: type[BaseModel]) -> bool:
    """O oráculo do portão abaixo, e ele é sobre a PROPRIEDADE e não sobre esta instância.

    **Três sinais, e nenhum é o nome de um campo de score** — `scores` sozinho é ambíguo: hoje
    `TrajetoriaDosQcs`, `ResumoDaPca` e `MedicaoDoCorretorResposta` têm campos com essa palavra e
    **nenhum** é PLS-DA. Um oráculo por `"score" in campo` acusaria os três, e a resposta seria uma
    lista de isenção — o que mata um portão.

    Os três sinais que de facto identificam PLS-DA: o nome da classe, o campo `vip` (a importância
    na projeção só existe em PLS), e a presença de um dos dois membros do par — um modelo que
    declare `q2` e não `permutation_pvalue` é exactamente o defeito que se quer apanhar.

    ## O detector fica LARGO, e é `_e_do_plsda` que separa a exigência (18/09/2026)

    **`permutation_pvalue` deixou de ser exclusivo do PLS-DA**, que passou a ter a mesma
    bicondicional aos classificadores por R10 — *"o `p` é campo obrigatório ao lado da medida, pela
    mesma bicondicional que já guarda o par do PLS-DA"*. Desde então `ResumoDoClassificador` casa
    este oráculo, **e não é falso positivo**: ele carrega a disciplina de validação de propósito.

    **Estreitar o detector para o deixar de acusar seria a resposta errada.** Um
    `ResumoDosScoresDoPlsda` de amanhã que declarasse só `permutation_pvalue` escaparia — e é
    exactamente a instância que a segunda pergunta manda o oráculo achar. O que muda não é
    **quem** ele acusa; é **o que se exige** de cada acusado.
    """
    campos = set(modelo.model_fields)
    return "plsda" in nome.lower() or "vip" in campos or bool(campos & {"q2", "permutation_pvalue"})


def _e_do_plsda(nome: str, modelo: type[BaseModel]) -> bool:
    """Dos acusados pelo detector, quais carregam de facto um score de **PLS-DA**.

    **`q2` é a quantidade que separa, e a razão é de domínio.** Ele é o R² validado por dobras da
    projeção em componentes latentes: existe para PLS e não tem definição para uma floresta ou um
    SVM, que não projetam nada. Exigi-lo de `ResumoDoClassificador` seria pedir um campo sem
    referente — e um campo sem referente ou fica `None` (e a bicondicional cai) ou recebe um número
    inventado.

    **`permutation_pvalue` NÃO entra aqui**, e é essa a assimetria: ele é a validação, comum às
    duas famílias, e continua exigido de **todos** os acusados. É `q2` que é do PLS-DA.
    """
    return (
        "plsda" in nome.lower() or "vip" in set(modelo.model_fields) or "q2" in modelo.model_fields
    )


def test_nenhum_modelo_de_esquemas_carrega_score_de_plsda_SEM_a_validacao() -> None:  # noqa: N802
    """**A regra 6 do README sobre a CLASSE e não sobre a instância que eu acabei de escrever.**

    A pergunta é *"quais instâncias deste defeito eu já consertei, e o meu oráculo acha as
    que eu não consertei?"*. A que eu consertei é `ResumoDoPlsda`; as que ele tem de achar são as
    que ainda não existem — um `ResumoDosScoresDoPlsda` de amanhã, ou um modelo que ganhe `vip`.
    Por isso o oráculo é o de `_carrega_score_de_plsda`, e a planta abaixo prova que ele acusa.

    > **E ele ACHOU as duas de amanhã.** `ResumoDoPlsdaMulticlasse` e
    > `ResumoDoModeloUmContraResto` nasceram na mesma sessão, e o oráculo acusou os dois na
    > primeira execução — o segundo **pelo `vip`**, porque o nome dele não tem «plsda». É a segunda
    > pergunta respondida por medição em vez de por argumento: o oráculo achou instâncias
    > que não existiam quando ele foi escrito, e os dois eram portadores legítimos.
    """
    modelos = {
        nome: cls
        for nome, cls in inspect.getmembers(esquemas, inspect.isclass)
        if issubclass(cls, BaseModel) and cls is not BaseModel
    }
    assert len(modelos) >= 20, f"a extração achou {len(modelos)} modelos — o portão seria vácuo"

    portadores = {n: c for n, c in modelos.items() if _carrega_score_de_plsda(n, c)}
    assert portadores, "nenhum modelo casa o oráculo — a asserção abaixo correria sobre o vazio"
    assert set(portadores) == {
        "ResumoDoClassificador",
        "ResumoDoModeloUmContraResto",
        "ResumoDoPlsda",
        "ResumoDoPlsdaMulticlasse",
    }, (
        f"o oráculo acusa {sorted(portadores)} e hoje são estes quatro que carregam a disciplina "
        "de validação — o PLS-DA binário, os dois do multiclasse, e o "
        "classificador da U15, que traz o `p` de permutação por R10. Um nome novo aqui é ou um "
        "portador legítimo, que tem de levar a validação, ou um falso positivo que mata o portão"
    )

    # **`permutation_pvalue` é exigido de TODOS, e `q2` só de quem é do PLS-DA.** A assimetria é de
    # domínio e não isenção: a validação por permutação vale para qualquer medida de desempenho — é
    # o que a regra 6 do README pede e o que R10 estendeu ao classificador —, enquanto `q2` é o R²
    # validado da projeção em componentes latentes e não tem definição para uma floresta.
    #
    # **Não é afrouxamento: nada que era exigido deixou de o ser.** Os três portadores de PLS-DA
    # continuam a levar o par inteiro, e o membro novo ganha a metade que tem referente. O que se
    # recusou foi pedir-lhe um campo sem referente, que ou ficaria `None` — derrubando a própria
    # bicondicional — ou receberia número inventado.
    for nome, modelo in portadores.items():
        exigidos = (
            ("q2", "permutation_pvalue") if _e_do_plsda(nome, modelo) else ("permutation_pvalue",)
        )
        for exigido in exigidos:
            campo = modelo.model_fields.get(exigido)
            assert campo is not None, f"{nome} carrega score e não declara {exigido!r}"
            assert campo.is_required(), (
                f"{nome}.{exigido} ganhou valor padrão: com padrão passa a existir construção "
                "válida sem validação, que é a regra 6 do README"
            )


def test_o_oraculo_da_inviolavel_7_ACUSA_um_portador_plantado() -> None:  # noqa: N802
    """A planta do portão acima: um modelo com score de PLS-DA e sem o par.

    **Em processo, e por isso não consegue ficar inerte**: a planta é argumento do oráculo, e
    o teste percorre exactamente o caminho que o portão percorre.
    """

    class ResumoDosScoresDoPlsda(BaseModel):
        scores_componente_1: list[float]

    assert _carrega_score_de_plsda("ResumoDosScoresDoPlsda", ResumoDosScoresDoPlsda)
    assert "q2" not in ResumoDosScoresDoPlsda.model_fields

    class ModeloComVip(BaseModel):
        vip: list[float]

    assert _carrega_score_de_plsda("SemNomeQueDenuncie", ModeloComVip), (
        "o oráculo perdeu o portador que se denuncia pelo campo `vip` e não pelo nome"
    )

    class ModeloSemNadaDisto(BaseModel):
        scores: list[float]

    assert not _carrega_score_de_plsda("TrajetoriaDosQcs", ModeloSemNadaDisto), (
        "o oráculo acusa um campo chamado `scores` que não é de PLS-DA — com este falso positivo "
        "ele vira lista de isenção"
    )


def test_as_tres_cargas_do_desenho_EXIGEM_validacao_sem_padrao() -> None:  # noqa: N802
    """**O mesmo mecanismo do lado que desenha, e o portão é TOTAL sobre o módulo.**

    Não é *"as três que eu escrevi têm o campo"* — é *"todo dataclass de carga deste módulo tem"*.
    Uma quarta carga escrita amanhã sem `validacao` reprova aqui, que é a diferença entre um portão
    sobre a propriedade e um sobre a instância.
    """
    cargas = [
        (nome, cls)
        for nome, cls in inspect.getmembers(modulo, inspect.isclass)
        if dataclasses.is_dataclass(cls)
        and cls.__module__ == modulo.__name__
        and cls is not Validacao
    ]
    assert {n for n, _ in cargas} == {"CargasDoPlsda", "EscoresDoPlsda", "ImportanciaVip"}, cargas
    for nome, cls in cargas:
        campos = {c.name: c for c in dataclasses.fields(cls)}
        assert "validacao" in campos, f"{nome} carrega dado de PLS-DA e não exige `Validacao`"
        assert campos["validacao"].default is dataclasses.MISSING, (
            f"{nome}.validacao ganhou valor padrão — passa a existir construção sem o par"
        )
        assert cls.__dataclass_params__.frozen  # type: ignore[attr-defined]

    # E a `Validacao` ela própria não admite construção parcial: são seis campos obrigatórios.
    for campo in dataclasses.fields(Validacao):
        assert campo.default is dataclasses.MISSING, campo.name
    with pytest.raises(TypeError):
        Validacao(q2=0.1)  # type: ignore[call-arg]


def test_as_tres_figuras_IMPRIMEM_o_par_na_propria_imagem(sintetico: dict) -> None:  # noqa: N802
    """**R7 na imagem, e o oráculo é o texto DESENHADO — não a frase que o construtor compôs.**

    A regra 6 do README alcança a legenda da figura, não só a função: o número que desmente a
    separação tem de sobreviver a uma captura de tela. Os dois valores são derivados aqui, do
    `PLSDAResult`, por caminho independente do construtor — que é a classe (d).
    """
    resultado: PLSDAResult = sintetico["resultado"]  # type: ignore[assignment]
    q2 = decimal(resultado.q2)
    p = decimal(resultado.permutation_pvalue)

    figuras = {
        "scores": figura_dos_scores_do_plsda(
            escores=sintetico["escores"],
            recorte=recorte_dos_scores_do_plsda(escores=sintetico["escores"], n_features_totais=8),
        ),
        "cargas": figura_das_cargas_do_plsda(
            cargas=sintetico["cargas"],
            recorte=recorte_das_cargas_do_plsda(cargas=sintetico["cargas"], n_features_totais=8),
        ),
        "vip": figura_do_vip(
            importancia=sintetico["vip"],
            n_features=4,
            recorte=recorte_do_vip(importancia=sintetico["vip"], n_desenhadas=4),
        ),
    }
    for nome, figura in figuras.items():
        titulo = figura.axes[0].get_title()
        assert q2 in titulo, f"{nome}: o título não traz o Q² ({titulo!r}); esperava {q2!r}"
        assert p in titulo, f"{nome}: o título não traz o p de permutação ({titulo!r})"
        rodape = rodape_de(figura)
        assert q2 in rodape and p in rodape, f"{nome}: o rodapé não repete o par — {rodape!r}"


# ---------------------------------------------------------------------------
# 2. O VIP: zero NÃO é indefinido
# ---------------------------------------------------------------------------


def test_o_vip_do_nucleo_devolve_NAN_e_nao_zeros_quando_a_soma_de_quadrados_e_zero() -> None:  # noqa: N802
    """**O par causal na origem, e a planta é sobre o modelo ajustado.**

    Ponta A: um modelo normal dá VIP finito e não negativo. Ponta B: o MESMO modelo com
    `y_loadings_` zerado tem `total == 0`, e o VIP sai `nan` — não zeros.

    **A planta é `y_loadings_` e não uma matriz degenerada, e a razão foi medida:** `PLSRegression`
    sobre `X` inteiramente constante nem **ajusta** — levanta `ValueError: slice 0 has a NaN entry`
    dentro de `scipy.linalg.pinv`. O ramo é defensivo, nenhuma entrada por `plsda()` o alcança neste
    corpus, e é justamente por isso que ele precisa de teste próprio: ramo que o dado real nunca
    exercita é o que fica inerte sem ninguém notar.
    """
    x = np.random.default_rng(3).normal(size=(10, 4))
    y = np.array([0.0] * 5 + [1.0] * 5)
    modelo = PLSRegression(n_components=2, scale=False).fit(x, y)

    saudavel = _vip(modelo)
    assert np.all(np.isfinite(saudavel)) and np.all(saudavel >= 0.0), saudavel

    modelo.y_loadings_ = np.zeros_like(np.asarray(modelo.y_loadings_))
    degenerado = _vip(modelo)
    assert degenerado.shape == saudavel.shape
    assert np.all(np.isnan(degenerado)), (
        f"o VIF do modelo degenerado saiu {degenerado!r}. Zeros aqui fariam «todas as features sem "
        "importância» e «não foi possível calcular» chegarem idênticos"
    )


def test_par_causal_do_VIP_zero_contra_VIP_indefinido(sintetico: dict) -> None:  # noqa: N802
    """**Os dois fatos opostos, sobre a MESMA população, e a figura os separa.**

    Ponta A: VIP todo **zero** — medida legítima, *"nenhuma feature tem importância"*. A figura
    desenha os pontos e o rodapé diz que nenhuma ficou sem VIP calculável.
    Ponta B: VIP todo **`nan`** — ausência de medida. A figura escreve o porquê no lugar dos pontos
    e o rodapé conta as indefinidas.

    **A âncora é que os dois rodapés DIFEREM**, e não que cada um contenha a frase certa:
    sem ela, um construtor que escrevesse a mesma frase nos dois casos passaria nas duas metades.
    """
    val: Validacao = sintetico["validacao"]  # type: ignore[assignment]
    nomes = [f"f{i:02d}" for i in range(8)]

    zeros = vip_do_plsda(vip=[0.0] * 8, features=nomes, validacao=val)
    nada = vip_do_plsda(vip=[float("nan")] * 8, features=nomes, validacao=val)

    assert zeros.n_calculaveis == 8 and zeros.n_indefinidas == 0
    assert nada.n_calculaveis == 0 and nada.n_indefinidas == 8

    fig_zeros = figura_do_vip(
        importancia=zeros,
        n_features=8,
        recorte=recorte_do_vip(importancia=zeros, n_desenhadas=8),
    )
    fig_nada = figura_do_vip(
        importancia=nada, n_features=8, recorte=recorte_do_vip(importancia=nada, n_desenhadas=0)
    )

    # (a) o dado do artista separa: oito pontos de um lado, nenhum do outro.
    pontos_zeros = [s for s in series(fig_zeros.axes[0]) if "maior VIP" in s.rotulo]
    assert len(pontos_zeros) == 1 and pontos_zeros[0].x.size == 8
    assert not [s for s in series(fig_nada.axes[0]) if "maior VIP" in s.rotulo]

    # (b) e a figura sem pontos DIZ o porquê, em vez de ficar em branco (R4b).
    textos = [t.get_text() for t in fig_nada.axes[0].texts]
    assert any("não foi possível calcular" in t for t in textos), textos
    assert not fig_nada.axes[0].get_xticks().tolist(), (
        "sem ponto nenhum o matplotlib desenha os limites de origem, e a figura passaria a afirmar "
        "que o VIP deste modelo vai de zero a um"
    )

    # (c) os dois rodapés DIFEREM, e cada um diz o seu fato.
    rod_zeros, rod_nada = rodape_de(fig_zeros), rodape_de(fig_nada)
    assert rod_zeros != rod_nada, "os dois rodapés são iguais: a planta não separa nada"
    assert "nenhuma feature ficou sem VIP calculável" in rod_zeros
    assert "8 de 8 features NÃO têm VIP calculável" in rod_nada
    assert "não foi possível calcular" in rod_nada and "importância zero" in rod_nada


def test_a_selecao_do_vip_deixa_as_indefinidas_de_FORA_e_na_contagem() -> None:  # noqa: N802
    """Uma indefinida no meio das calculáveis: fora da seleção, dentro da contagem."""
    val = Validacao(
        q2=0.5, permutation_pvalue=0.01, r2y=0.8, n_permutacoes=200, cv_folds=7, n_componentes=2
    )
    importancia = vip_do_plsda(
        vip=[2.0, float("nan"), 3.0, 1.0],
        features=["a", "b", "c", "d"],
        validacao=val,
    )
    indices = selecao_do_vip(importancia, 10)
    assert [importancia.features[int(i)] for i in indices] == ["c", "a", "d"], indices
    assert importancia.n_indefinidas == 1

    figura = figura_do_vip(
        importancia=importancia,
        n_features=10,
        recorte=recorte_do_vip(importancia=importancia, n_desenhadas=int(indices.size)),
    )
    rodape = rodape_de(figura)
    assert "1 de 4 features NÃO têm VIP calculável" in rodape, rodape
    assert [t.get_text() for t in figura.axes[0].get_yticklabels()] == ["c", "a", "d"]


def test_o_empate_do_vip_e_resolvido_pelo_NOME_e_nao_pela_ordem_de_chegada() -> None:  # noqa: N802
    """Determinismo da seleção — a regra 3 do README aplicada a um recorte, não a um sorteio."""
    val = Validacao(
        q2=0.5, permutation_pvalue=0.01, r2y=0.8, n_permutacoes=200, cv_folds=7, n_componentes=2
    )
    empatadas = vip_do_plsda(vip=[1.0, 1.0, 1.0], features=["zeta", "alfa", "meio"], validacao=val)
    escolhidos = [empatadas.features[int(i)] for i in selecao_do_vip(empatadas, 2)]
    assert escolhidos == ["alfa", "meio"], escolhidos

    invertidas = vip_do_plsda(vip=[1.0, 1.0, 1.0], features=["meio", "alfa", "zeta"], validacao=val)
    assert [invertidas.features[int(i)] for i in selecao_do_vip(invertidas, 2)] == escolhidos


# ---------------------------------------------------------------------------
# 3. A ORDEM DO PAR — e o que a medição diz dela
# ---------------------------------------------------------------------------


def test_trocar_a_ordem_do_par_NAO_muda_numero_nenhum(sintetico: dict) -> None:  # noqa: N802
    """**A afirmação do plano é FALSA, e este teste é o que a fixa depois de medida.**

    O plano previu que `(A, B)` e `(B, A)` produzissem *"imagens espelhadas"*. Medido em 11/09/2026,
    no sintético e no `pos_hilic`: os dois devolvem `scores`, `loadings`, `vip`, `q2`,
    `permutation_pvalue` e `r2y` **byte a byte iguais**. A causa é a orientação canônica — trocar a
    ordem troca o sinal de `y`, o que troca o sinal de tudo, e `orient_canonically` desfaz a troca.

    **E isso torna o rótulo mais necessário, não menos:** se a figura espelhasse, um
    `grupo_positivo` trocado seria visível. Como ela é idêntica, ele é indistinguível da imagem.
    """
    matriz: RefinoMatrix = sintetico["matriz"]  # type: ignore[assignment]
    direto: PLSDAResult = sintetico["resultado"]  # type: ignore[assignment]
    invertido = resultado_de(matriz, ("caso", "controle"))

    assert np.array_equal(direto.scores, invertido.scores), (
        f"os scores diferem em {float(np.max(np.abs(direto.scores - invertido.scores)))}; se o "
        "helper canônico deixou de os igualar, a figura passou a espelhar e a docstring do módulo "
        "descreve outra coisa"
    )
    assert np.array_equal(direto.loadings, invertido.loadings)
    assert np.array_equal(direto.vip, invertido.vip)
    assert direto.q2 == invertido.q2 and direto.permutation_pvalue == invertido.permutation_pvalue

    # **Os escores de validação cruzada — iguais à precisão da máquina, e não byte a byte.**
    # A partição é a mesma nas duas ordens e cada modelo de dobra vira inteiro, mas o giro sai de um
    # SVD, e nada no desenho obriga o SVD de uma matriz negada a devolver o negado bit a bit. Neste
    # sintético eles saem iguais byte a byte; afirmá-lo seria fixar a sorte do fixture, e a
    # tolerância de 1e-12 da escala continua a acusar um referencial trocado, que muda o sinal.
    escala = float(np.max(np.abs(direto.cv_scores)))
    assert escala > 0.0, "âncora: escores todos nulos seriam iguais por vacuidade"
    assert np.allclose(direto.cv_scores, invertido.cv_scores, rtol=0.0, atol=1e-12 * escala)


def test_a_figura_DIZ_qual_grupo_e_o_positivo_e_os_dois_lados_sao_enderecaveis(  # noqa: N802
    sintetico: dict,
) -> None:
    """**O par causal do rótulo: o que muda com a ordem é a legenda, e ela tem de dizê-lo.**

    Ponta A: com `positivo="caso"` a legenda chama `caso` de `y = 1`. Ponta B: a MESMA nuvem, com o
    par declarado ao contrário, chama `controle` de `y = 1` — e as duas séries continuam a ter
    exactamente os mesmos pontos, porque só o rótulo mudou.
    """
    resultado: PLSDAResult = sintetico["resultado"]  # type: ignore[assignment]
    matriz: RefinoMatrix = sintetico["matriz"]  # type: ignore[assignment]
    val: Validacao = sintetico["validacao"]  # type: ignore[assignment]
    grupos = [a.group for a in matriz.samples]

    def legenda(positivo: str, negativo: str) -> tuple[str, ...]:
        escores = escores_do_plsda(
            comp1=resultado.scores[:, 0],
            comp2=resultado.scores[:, 1],
            grupos=grupos,
            grupo_positivo=positivo,
            grupo_negativo=negativo,
            validacao=val,
        )
        figura = figura_dos_scores_do_plsda(
            escores=escores,
            recorte=recorte_dos_scores_do_plsda(escores=escores, n_features_totais=8),
        )
        alvo = figura.axes[0].get_legend()
        assert alvo is not None
        return tuple(t.get_text() for t in alvo.get_texts())

    a = legenda("caso", "controle")
    b = legenda("controle", "caso")
    assert a != b, "trocar o papel não mudou a legenda: o rótulo não está a ser lido"
    assert "caso — 10 amostras (classe positiva, y = 1)" in a
    assert "controle — 10 amostras (classe de referência, y = 0)" in a
    assert "controle — 10 amostras (classe positiva, y = 1)" in b


def test_o_terceiro_grupo_no_vetor_de_rotulos_e_RECUSA_e_nao_recuo(sintetico: dict) -> None:  # noqa: N802
    """O domínio dos rótulos é FECHADO em dois, e a fronteira valida em runtime.

    Um terceiro nome não pode cair numa das séries por omissão: a legenda contaria uma população que
    o modelo não viu, e nada levantaria — que é o defeito exacto que a asserção ignora.
    """
    val: Validacao = sintetico["validacao"]  # type: ignore[assignment]
    with pytest.raises(PlsdaDesalinhadoError, match="ATLL"):
        escores_do_plsda(
            comp1=[1.0, 2.0, 3.0],
            comp2=[1.0, 2.0, 3.0],
            grupos=["controle", "caso", "ATLL"],
            grupo_positivo="caso",
            grupo_negativo="controle",
            validacao=val,
        )


# ---------------------------------------------------------------------------
# 4. O caminho feliz e as quatro classes de asserção do arnês
# ---------------------------------------------------------------------------


def test_o_plsda_sintetico_separa_e_o_VIP_elege_as_features_plantadas(sintetico: dict) -> None:  # noqa: N802
    """O oráculo do caminho feliz: as três features deslocadas são as de maior VIP."""
    importancia: ImportanciaVip = sintetico["vip"]  # type: ignore[assignment]
    resultado: PLSDAResult = sintetico["resultado"]  # type: ignore[assignment]
    assert resultado.q2 > 0.5, resultado.q2
    eleitas = {importancia.features[int(i)] for i in selecao_do_vip(importancia, 3)}
    assert eleitas == {"f00", "f01", "f02"}, eleitas


def test_a_figura_dos_scores_passa_pelas_quatro_classes_do_arnes(sintetico: dict) -> None:
    """Classes (a), (b), (c) e (d) sobre a figura limpa, com o esperado derivado do dado.

    **Quatro séries **: os pontos de cada grupo e a região de predição de cada um, que
    são linhas e por isso entram no arnês como qualquer série.
    """
    escores: EscoresDoPlsda = sintetico["escores"]  # type: ignore[assignment]
    figura = figura_dos_scores_do_plsda(
        escores=escores, recorte=recorte_dos_scores_do_plsda(escores=escores, n_features_totais=8)
    )
    eixo = figura.axes[0]

    esperadas = esperadas_dos_scores(escores)
    assert len(esperadas) == 4, f"âncora: os dois grupos têm região neste sintético — {esperadas}"
    assert confere_series(eixo, esperadas) == []
    assert (
        confere_rotulos(
            eixo,
            x=rotulo_do_componente_do_plsda(0),
            y=rotulo_do_componente_do_plsda(1),
            titulo=eixo.get_title(),
            legenda=(
                "controle — 10 amostras (classe de referência, y = 0)",
                "caso — 10 amostras (classe positiva, y = 1)",
                "Região de predição 95% (T²) — controle",
                "Região de predição 95% (T²) — caso",
            ),
        )
        == []
    )
    assert confere_vista(eixo) == []
    assert confere_numeros(figura, ["8"]), (
        "a figura declara mais números que este, e o arnês tem de o dizer"
    )


@pytest.mark.parametrize(
    "planta",
    [planta_serie_removida, planta_eixos_trocados, planta_sinal_invertido],
)
def test_as_tres_plantas_de_dado_ACUSAM_na_figura_dos_scores(sintetico: dict, planta) -> None:  # noqa: N802
    """As três mutações de dado, cada uma com a âncora de separação dentro dela."""
    escores: EscoresDoPlsda = sintetico["escores"]  # type: ignore[assignment]
    figura = figura_dos_scores_do_plsda(
        escores=escores, recorte=recorte_dos_scores_do_plsda(escores=escores, n_features_totais=8)
    )
    eixo = figura.axes[0]
    esperadas = esperadas_dos_scores(escores)
    assert confere_series(eixo, esperadas) == [], "a planta caiu sobre uma figura já suja"
    planta(eixo)
    assert confere_series(eixo, esperadas) != [], f"{planta.__name__} não foi acusada"


def test_a_planta_do_corte_de_eixo_ACUSA_o_que_as_tres_de_dado_nao_alcancam(  # noqa: N802
    sintetico: dict,
) -> None:
    """A classe (c): dado nenhum muda, e a tela fica em branco."""
    escores: EscoresDoPlsda = sintetico["escores"]  # type: ignore[assignment]
    figura = figura_dos_scores_do_plsda(
        escores=escores, recorte=recorte_dos_scores_do_plsda(escores=escores, n_features_totais=8)
    )
    eixo = figura.axes[0]
    assert confere_vista(eixo) == []
    planta_corte_de_eixo(eixo)
    assert confere_vista(eixo) != []


def test_as_cargas_desenham_UM_artista_de_dado_e_as_duas_cruzes(sintetico: dict) -> None:  # noqa: N802
    """Um artista de dado só, e as cruzes com rótulos DISTINTOS — requisito do arnês.

    `arnes_de_figura.series` indexa por rótulo num dicionário: dois artistas com o mesmo texto
    colapsam num só, e o primeiro fica invisível à classe (a).
    """
    cargas: CargasDoPlsda = sintetico["cargas"]  # type: ignore[assignment]
    figura = figura_das_cargas_do_plsda(
        cargas=cargas, recorte=recorte_das_cargas_do_plsda(cargas=cargas, n_features_totais=8)
    )
    eixo = figura.axes[0]
    assert len(eixo.collections) == 1, "mais de uma dispersão desalinha a identificação de R2"
    rotulos = {s.rotulo for s in series(eixo)}
    assert rotulos == {"origem", "origem do componente 1", "8 features"}, rotulos
    assert confere_vista(eixo) == []
    assert len(eixo.get_legend().get_texts()) == 2


# A validação que reproduz a geometria da PRIMEIRA renderização do `pos_hilic`. **Os valores são
# os medidos, e não uma escolha:** a altura do rodapé depende do comprimento destas frases, o
# `tight_layout` reposiciona os eixos a partir dela, e a colisão que se quer reproduzir é
# sensível a isso — ver a docstring do teste.
VALIDACAO_DA_RENDERIZACAO = Validacao(
    q2=-0.0125,
    permutation_pvalue=0.1095,
    r2y=0.3498,
    n_permutacoes=200,
    cv_folds=7,
    n_componentes=2,
)


# O tamanho do painel da janela a 1280x860, que é onde a colisão aparece — o da tabela
# de `recorte.reajusta`, aplicado como a janela o aplica: a figura posta no tamanho do widget e a
# faixa do rodapé refeita, sem novo `tight_layout`.
TAMANHO_DA_JANELA: tuple[float, float, float] = (6.5, 3.4, 96.0)

# Os pontos de ordem `1e8` — que é o que faz o `matplotlib` fatorizar o expoente — em duas formas:
# uma NUVEM, em que cada grupo tem a sua região, que é a figura de todo contraste real; e uma RETA,
# em que nenhum grupo a tem e o eixo leva o motivo, .
PONTOS_DO_EXPOENTE: dict[str, tuple[np.ndarray, np.ndarray]] = {
    "nuvem": (np.arange(20, dtype=float) * 1e7, np.random.default_rng(5).normal(size=20) * 1e7),
    "reta": (np.arange(20, dtype=float) * 1e7, np.arange(20, dtype=float) * 1e7),
}


def _figura_do_expoente(
    val: Validacao,
    *,
    pontos: str,
    tamanho: tuple[float, float, float] | None,
    rotulo_de_duas_linhas: bool = False,
) -> Figure:
    """A figura dos escores com os nomes do contraste do `pos_hilic`, no tamanho pedido."""
    from refino.plots.base import GID_DO_RODAPE
    from refino.plots.recorte import reajusta

    comp1, comp2 = PONTOS_DO_EXPOENTE[pontos]
    escores = escores_do_plsda(
        comp1=comp1,
        comp2=comp2,
        grupos=["Asymptomatic"] * 10 + ["HAMTSP"] * 10,
        grupo_positivo="HAMTSP",
        grupo_negativo="Asymptomatic",
        validacao=val,
    )
    figura = figura_dos_scores_do_plsda(
        escores=escores,
        recorte=recorte_dos_scores_do_plsda(escores=escores, n_features_totais=6035),
    )
    if rotulo_de_duas_linhas:
        figura.axes[0].set_xlabel("componente 1 de PLS-DA\n(sinal canônico, [CA-6.7])")
    if tamanho is not None:
        rodape = next(texto for texto in figura.texts if texto.get_gid() == GID_DO_RODAPE)
        largura, altura, ppp = tamanho
        figura.set_size_inches(largura, altura)
        figura.set_dpi(ppp)
        reajusta(figura, rodape)
    return figura


def _colidem_expoente_e_rotulo(figura: Figure, *, restaura_o_padrao: bool) -> bool:
    """As caixas desenhadas do expoente do eixo x e do `xlabel` sobrepõem-se?

    Lê a caixa **renderizada** e não a posição pedida: colisão é propriedade da imagem montada, e a
    posição sozinha não a decide — quem a decide é onde o `tight_layout` deixou os eixos.
    """
    eixo = figura.axes[0]
    expoente = eixo.xaxis.get_offset_text()
    if restaura_o_padrao:
        expoente.set_position((1.0, 0.0))
        expoente.set_verticalalignment("top")
    figura.canvas.draw()
    a = expoente.get_window_extent().bounds
    b = eixo.xaxis.label.get_window_extent().bounds
    return not (
        a[0] + a[2] < b[0] or b[0] + b[2] < a[0] or a[1] + a[3] < b[1] or b[1] + b[3] < a[1]
    )


def test_o_expoente_do_eixo_x_NAO_cai_em_cima_do_rotulo(sintetico: dict) -> None:  # noqa: N802
    """**O par causal de um defeito que só a imagem montada mostra.**

    Ponta A: como a figura é construída, as caixas do expoente e do `xlabel` não se tocam — em
    **dezesseis** geometrias: nuvem e reta, duas validações, rótulo de uma e de duas linhas, no
    tamanho da construção e no do painel da janela, porque a propriedade é o que interessa. Ponta B:
    devolvido o expoente à posição padrão do `matplotlib`, as duas caixas **sobrepõem-se** numa
    dessas dezesseis, a nuvem no painel da janela — com `1e8` impresso por cima de `de PLS-DA`, que
    é o defeito que a primeira renderização do `pos_hilic` mostrou.

    **A geometria da ponta B MUDOU, e a margem depende da FONTE — as duas coisas foram
    medidas.** Na DejaVu Sans, que é a do produto — o padrão do `matplotlib`, porque nada neste
    repositório escolhe fonte —, a sobreposição sem o auxiliar no tamanho da construção era de
    +3,14 px na reta e de +4,64 px na nuvem antes, e passou a -2,87 e -1,86 px: as regiões,
    o motivo da ausência dentro do eixo e a frase nova do rodapé mudaram onde o `tight_layout` deixa
    os eixos, e a ponta B antiga deixou de ter colisão a desfazer. No painel da janela ela é de
    **+20,74 px em x e +12,00 px em y** na nuvem, nas duas validações e nos dois rótulos — e de
    +20,68 px antes. O auxiliar sobe o expoente ~13 px, e a caixa dele fica 1,33 px acima
    da do `xlabel`.

    > **Na Inter que o `MATPLOTLIBRC` do contêiner de desenvolvimento impõe, a ponta B NÃO se
    > cumpre** — nem nesta árvore nem na anterior: sem o auxiliar, as caixas não se tocam em
    > nenhuma das 64 geometrias medidas — as duas formas de pontos, duas validações, dois rótulos,
    > e a construção mais sete tamanhos de `reajusta` — (a nuvem no painel da janela fica a
    > -4,89 px). Esta
    > reprovação lá é da fonte do ambiente e não do código, e só se lê como linha de base depois de
    > a mesma seleção correr com o `MATPLOTLIBRC` removido, nas duas árvores.
    """
    do_sintetico: Validacao = sintetico["validacao"]  # type: ignore[assignment]
    vistas = 0
    for pontos in PONTOS_DO_EXPOENTE:
        for val in (VALIDACAO_DA_RENDERIZACAO, do_sintetico):
            for tamanho in (None, TAMANHO_DA_JANELA):
                for duas_linhas in (False, True):
                    figura = _figura_do_expoente(
                        val, pontos=pontos, tamanho=tamanho, rotulo_de_duas_linhas=duas_linhas
                    )
                    assert not _colidem_expoente_e_rotulo(figura, restaura_o_padrao=False), (
                        f"colisão com {pontos}, q2={val.q2}, tamanho {tamanho} e rótulo de "
                        f"{'duas' if duas_linhas else 'uma'} linha(s)"
                    )
                    vistas += 1
    assert vistas == 16, "âncora: a ponta A percorreu menos geometrias do que declara"

    figura = _figura_do_expoente(
        VALIDACAO_DA_RENDERIZACAO, pontos="nuvem", tamanho=TAMANHO_DA_JANELA
    )
    assert len(figura.axes[0].lines) == 4, (
        "âncora: a nuvem da ponta B tem de levar as duas regiões, que é a figura de todo contraste "
        f"real — {[str(linha.get_label()) for linha in figura.axes[0].lines]}"
    )
    assert _colidem_expoente_e_rotulo(figura, restaura_o_padrao=True), (
        "sem o auxiliar as duas caixas deviam colidir nesta geometria — se já não colidem, a "
        "planta não separa nada e a ponta A passou a ser decorativa"
    )


def test_o_auxiliar_do_expoente_move_o_texto_e_nao_o_esconde() -> None:
    """A pergunta sobre o próprio auxiliar: ele resolve por mover, ou por apagar?"""
    from matplotlib.figure import Figure as FiguraNua

    figura = FiguraNua()
    eixo = figura.add_subplot(1, 1, 1)
    antes = eixo.xaxis.get_offset_text().get_position()
    expoente_fora_do_rotulo(eixo)
    depois = eixo.xaxis.get_offset_text()
    assert depois.get_visible(), "o auxiliar escondeu o expoente em vez de o mover"
    assert depois.get_position() != antes
    assert depois.get_verticalalignment() == "bottom"


# ---------------------------------------------------------------------------
# 5. O veredito tem TRÊS desfechos, e a verificação pede todos
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("q2", "trecho"),
    [
        (float("nan"), "Q² NÃO FOI MEDIDO"),
        (-0.18, "Q² <= 0"),
        (0.0, "Q² <= 0"),
        (0.91, "Q² > 0"),
    ],
)
def test_os_tres_desfechos_do_veredito(q2: float, trecho: str) -> None:
    """Três desfechos, e o do meio é o do corpus real.

    **Zero cai no ramo de baixo de propósito, e não é limiar escolhido:** `Q² = 1 - PRESS/TSS` com
    `PRESS = TSS` é exactamente *"errar tanto quanto errar a média"*. Qualquer outro corte seria
    limiar sem a distribuição sobre a qual corta.
    """
    val = Validacao(
        q2=q2, permutation_pvalue=0.05, r2y=0.5, n_permutacoes=200, cv_folds=7, n_componentes=2
    )
    assert trecho in veredito_da_validacao(val)


def test_a_frase_da_validacao_diz_NAO_DECLARADO_em_vez_de_inventar_zero() -> None:  # noqa: N802
    """`nan` no fio é ausência de medida, e um `0,0` ali seria a medida mais enganosa possível."""
    ausente = validacao_do_fio(
        q2=None,
        permutation_pvalue="talvez",
        r2y=float("inf"),
        n_permutacoes=None,
        cv_folds=True,
        n_componentes=2,
    )
    assert math.isnan(ausente.q2) and math.isnan(ausente.permutation_pvalue)
    assert ausente.cv_folds == 0, "`True` não é uma contagem de dobras"
    frase = frase_da_validacao(ausente)
    assert frase.count("não declarado") == 3, frase
    assert "não declarado" in frase_curta_da_validacao(ausente)


# ---------------------------------------------------------------------------
# 6. As recusas tipadas
# ---------------------------------------------------------------------------


def test_populacao_vazia_e_MatrizVaziaError_nos_tres_construtores_do_fio() -> None:  # noqa: N802
    """Sem par de grupos declarado a rota não corre o PLS-DA, e o fio chega vazio.

    `MatrizVaziaError` é a única excepção que `PainelDeFigura._troca_a_figura` apanha; qualquer
    outra dentro de um *slot* do Qt é engolida para o `stderr`, com o painel a parar de actualizar.
    """
    val = Validacao(
        q2=0.1, permutation_pvalue=0.5, r2y=0.2, n_permutacoes=200, cv_folds=7, n_componentes=2
    )
    with pytest.raises(MatrizVaziaError):
        escores_do_plsda(
            comp1=None,
            comp2=None,
            grupos=None,
            grupo_positivo="a",
            grupo_negativo="b",
            validacao=val,
        )
    with pytest.raises(MatrizVaziaError):
        cargas_do_plsda(comp1=None, comp2=None, features=None, validacao=val)
    with pytest.raises(MatrizVaziaError):
        vip_do_plsda(vip=None, features=None, validacao=val)


@pytest.mark.parametrize(
    ("kwargs", "trecho"),
    [
        ({"comp2": [1.0]}, "score"),
        ({"grupos": ["controle"]}, "rótulo"),
        ({"grupo_positivo": ""}, "positivo"),
        ({"grupo_negativo": None}, "negativo"),
        ({"grupo_positivo": "controle"}, "mesmo nome"),
    ],
)
def test_o_desalinhamento_e_recusa_TIPADA_e_nao_figura_bem_formada(  # noqa: N802
    kwargs: dict, trecho: str
) -> None:
    """Cinco formas de corpo mal montado, cada uma com a sua mensagem.

    **`PlsdaDesalinhadoError` e não `MatrizVaziaError`, e a distinção importa:** *"não veio
    população"* e *"veio população desalinhada"* têm consertos diferentes, e o painel mostra a
    frase errada para o segundo se os dois colapsarem num tipo só.
    """
    val = Validacao(
        q2=0.1, permutation_pvalue=0.5, r2y=0.2, n_permutacoes=200, cv_folds=7, n_componentes=2
    )
    base: dict = {
        "comp1": [1.0, 2.0],
        "comp2": [1.0, 2.0],
        "grupos": ["controle", "caso"],
        "grupo_positivo": "caso",
        "grupo_negativo": "controle",
        "validacao": val,
    }
    with pytest.raises(PlsdaDesalinhadoError, match=trecho):
        escores_do_plsda(**{**base, **kwargs})


@pytest.mark.parametrize(
    ("construtor", "kwargs", "trecho"),
    [
        (
            "cargas",
            {"comp1": [0.1, 0.2], "comp2": [0.3], "features": ["f0", "f1"]},
            "carga(s) do componente 1 para",
        ),
        (
            "cargas",
            {"comp1": [0.1, 0.2], "comp2": [0.3, 0.4], "features": ["f0"]},
            "nome(s) de feature para",
        ),
        ("vip", {"vip": [1.0, 2.0], "features": ["f0"]}, "nome(s) de feature para"),
    ],
    ids=["cargas-componentes", "cargas-nomes", "vip-nomes"],
)
def test_as_cargas_e_o_VIP_recusam_desalinhamento_como_os_escores(  # noqa: N802
    construtor: str, kwargs: dict, trecho: str
) -> None:
    """O desalinhamento nos DOIS irmãos que ficaram sem cobertura.

    **A guarda de vazio dos três já estava coberta** por
    `test_populacao_vazia_e_MatrizVaziaError_nos_tres_construtores_do_fio`, e o desalinhamento
    estava coberto **só para `escores_do_plsda`**, pelo teste acima. `cargas_do_plsda` e
    `vip_do_plsda` levantam com o mesmo código e nunca eram chamados com entrada torta — todas as
    chamadas da suíte passavam vetores já alinhados.

    O desfecho que isto impede é o de R2: um nome de feature ao lado da carga de outra, numa figura
    bem formada em que nada levanta.
    """
    val = Validacao(
        q2=0.1, permutation_pvalue=0.5, r2y=0.2, n_permutacoes=200, cv_folds=7, n_componentes=2
    )
    with pytest.raises(PlsdaDesalinhadoError, match=re.escape(trecho)):
        if construtor == "cargas":
            cargas_do_plsda(validacao=val, **kwargs)
        else:
            vip_do_plsda(validacao=val, **kwargs)


def test_o_recorte_recusa_denominador_que_nao_existe() -> None:
    """Sem execução não há população, e o rodapé não inventa denominador."""
    val = Validacao(
        q2=0.1, permutation_pvalue=0.5, r2y=0.2, n_permutacoes=200, cv_folds=7, n_componentes=2
    )
    cargas = cargas_do_plsda(comp1=[0.1], comp2=[0.2], features=["f"], validacao=val)
    with pytest.raises(MatrizVaziaError, match="n_features_totais"):
        recorte_das_cargas_do_plsda(cargas=cargas, n_features_totais=None)


def test_o_recorte_recusa_desenhar_mais_do_que_existe() -> None:
    """`desenhadas > disponiveis` é numerador e denominador de populações diferentes."""
    val = Validacao(
        q2=0.1, permutation_pvalue=0.5, r2y=0.2, n_permutacoes=200, cv_folds=7, n_componentes=2
    )
    importancia = vip_do_plsda(vip=[1.0, 2.0], features=["a", "b"], validacao=val)
    with pytest.raises(RecorteInvalidoError):
        recorte_do_vip(importancia=importancia, n_desenhadas=5)


def test_menos_amostras_que_dobras_e_recusa_ANTES_de_haver_figura() -> None:  # noqa: N802
    """A recusa é do núcleo e nomeia AS DUAS contagens — não só a que falhou."""
    with pytest.raises(ValueError, match="cv_folds"):
        plsda(matriz_sintetica(por_grupo=3), ("controle", "caso"), cv_folds=7, n_permutations=2)


# ---------------------------------------------------------------------------
# 7. O corpus real — o único cenário desta unidade sobre os arquivos do laboratório
# ---------------------------------------------------------------------------


def test_real_o_pos_hilic_separa_e_o_Q2_desmente_a_separacao(matriz_real: RefinoMatrix) -> None:  # noqa: N802
    """**O caso que a regra 6 do README existe para tornar visível, no dado do laboratório.**

    Medido em 11/09/2026 sobre o `pos_hilic` ingerido, com `n_components=2`, `cv_folds=7` e 200
    permutações: `Asymptomatic` contra `HAMTSP` dá `R²Y = 0,3498` — ajuste dentro da amostra — e
    `Q² = -0,0125` com `p = 0,1095`. É o **melhor** dos quatro pares deste estudo, e ainda assim o
    modelo prediz pior que a média dos rótulos.

    **A asserção é sobre o SINAL e não sobre o valor**, de propósito: o número exacto depende da
    versão do `scikit-learn` e congelá-lo faria o teste reprovar por atualização de dependência, que
    é reprovar pelo motivo errado. O que não pode mudar sem alguém decidir
    é a leitura — `R²Y` positivo e `Q²` negativo é sobreajuste, e a figura tem de o dizer.
    """
    resultado = plsda(
        matriz_real, PAR_REAL, n_components=2, cv_folds=7, n_permutations=200, seed=42
    )
    assert resultado.r2y > 0.0, resultado.r2y
    assert resultado.q2 < 0.0, (
        f"o Q² deste par passou a ser {resultado.q2}, e era negativo em 11/09/2026. Isso é achado, "
        "não defeito: relatar antes de mexer no teste"
    )

    val = validacao_de(resultado, cv_folds=7)
    assert "Q² <= 0" in veredito_da_validacao(val)

    escores = escores_de(matriz_real, resultado, PAR_REAL, val)
    assert escores.n == 32, escores.n
    assert int(escores.mascara("Asymptomatic").sum()) == 18
    assert int(escores.mascara("HAMTSP").sum()) == 14

    figura = figura_dos_scores_do_plsda(
        escores=escores,
        recorte=recorte_dos_scores_do_plsda(
            escores=escores, n_features_totais=matriz_real.n_features
        ),
    )
    rodape = rodape_de(figura)
    assert (
        f"{milhar(matriz_real.n_features)} de {milhar(matriz_real.n_features)} features" in rodape
    )
    assert "prediz PIOR que a média dos rótulos" in rodape
    assert decimal(resultado.q2) in figura.axes[0].get_title()

    # **Os pontos são os de validação cruzada, e cada grupo tem a sua região**, sobre os 18
    # e os 14 que o contraste tem — contados acima da máscara —, as duas na legenda e no desenho.
    assert "escores de VALIDAÇÃO CRUZADA" in rodape
    assert confere_series(figura.axes[0], esperadas_dos_scores(escores)) == []
    legenda = figura.axes[0].get_legend()
    assert legenda is not None
    textos = [t.get_text() for t in legenda.get_texts()]
    for grupo in PAR_REAL:
        assert ROTULO_DA_REGIAO_NO_TESTE.format(grupo=grupo) in textos, textos


def test_real_o_vip_do_corpus_e_TODO_finito_e_a_figura_o_declara(matriz_real: RefinoMatrix) -> None:  # noqa: N802
    """A âncora do ramo degenerado: neste corpus ele **não** dispara, e o rodapé diz isso.

    Sem esta asserção, o teste do VIP indefinido poderia estar a descrever o caso normal — e a
    distinção passaria por não haver nada de que a distinguir.
    """
    resultado = plsda(matriz_real, PAR_REAL, n_components=2, cv_folds=7, n_permutations=20, seed=42)
    importancia = vip_de(matriz_real, resultado, validacao_de(resultado, cv_folds=7))
    assert importancia.n == matriz_real.n_features
    assert importancia.n_indefinidas == 0, importancia.n_indefinidas

    figura = figura_do_vip(
        importancia=importancia,
        n_features=FEATURES_DO_VIP,
        recorte=recorte_do_vip(importancia=importancia, n_desenhadas=FEATURES_DO_VIP),
    )
    rodape = rodape_de(figura)
    assert "nenhuma feature ficou sem VIP calculável" in rodape
    marcas = [t.get_text() for t in figura.axes[0].get_yticklabels()]
    assert len(marcas) == FEATURES_DO_VIP
    esperadas = [importancia.features[int(i)] for i in selecao_do_vip(importancia, FEATURES_DO_VIP)]
    assert marcas == esperadas


# ---------------------------------------------------------------------------
# 8. A latência do redesenho
# ---------------------------------------------------------------------------


def mediana_do_redesenho(figura: Figure, repeticoes: int = 5) -> float:
    """A mediana de `repeticoes` redesenhos, em milissegundos. **A primeira não conta.**

    O primeiro `draw` paga o cache de fontes do `matplotlib`, que é uma vez por processo e não por
    figura — medi-lo junto reportaria o custo de arranque como custo de redesenho.
    """
    figura.canvas.draw()
    tempos = []
    for _ in range(repeticoes):
        comecou = time.perf_counter()
        figura.canvas.draw()
        tempos.append((time.perf_counter() - comecou) * 1000.0)
    return sorted(tempos)[len(tempos) // 2]


def test_o_custo_do_redesenho_e_do_TEXTO_e_nao_do_dado(matriz_real: RefinoMatrix) -> None:  # noqa: N802
    """**A causa do custo, medida — e NENHUMA asserção de tempo de parede.**

    ## O que foi medido, e onde o número vive

    Mediana de cinco redesenhos da mesma figura, 11/09/2026, `pos_hilic` com 6.035 features:

    | Figura | Redesenho | População desenhada |
    |---|---|---|
    | scores | **74,3 ms** | 32 pontos |
    | cargas | **71,2 ms** | 6.035 pontos |
    | VIP, 12 marcas | **89,9 ms** | 12 pontos + 12 rótulos de marca |

    **A linha de base é de TEXTO, e é o que explica a tabela:** a mesma figura de scores sem título,
    rodapé e legenda desenha em **10,1 ms**. Cargas com 6.035 pontos custa *menos* que scores com
    32, porque o que difere entre elas é o comprimento das frases e não a nuvem. O orçamento de
    interação de `ORCAMENTO_DECLARADO_MS` fica **declarado** e não asserido — ver o comentário
    daquela constante, e o portão que reprovou a primeira versão deste teste.

    ## O que ele AFIRMA, e por que é uma propriedade e não um relógio

    Que apagar os rótulos de marca do VIP derruba o redesenho, **sobre a mesma máquina e na mesma
    execução**. É a causa, e é o que mudaria se alguém tornasse o desenho
    caro por outra via — 6.035 artistas individuais, por exemplo. A margem medida é de
    134,1 contra 75,8 ms com 40 marcas: uma comparação relativa que sobrevive a uma máquina lenta,
    ao contrário de qualquer limite absoluto.
    """
    resultado = plsda(matriz_real, PAR_REAL, n_components=2, cv_folds=7, n_permutations=20, seed=42)
    val = validacao_de(resultado, cv_folds=7)
    escores = escores_de(matriz_real, resultado, PAR_REAL, val)
    cargas = cargas_de(matriz_real, resultado, val)
    importancia = vip_de(matriz_real, resultado, val)
    totais = matriz_real.n_features

    # As três desenham, e a medição acontece — sem isto as comparações abaixo correriam sobre
    # figuras que nunca foram construídas.
    medidas = {
        "scores": mediana_do_redesenho(
            figura_dos_scores_do_plsda(
                escores=escores,
                recorte=recorte_dos_scores_do_plsda(escores=escores, n_features_totais=totais),
            )
        ),
        "cargas": mediana_do_redesenho(
            figura_das_cargas_do_plsda(
                cargas=cargas,
                recorte=recorte_das_cargas_do_plsda(cargas=cargas, n_features_totais=totais),
            )
        ),
    }
    assert all(ms > 0.0 for ms in medidas.values()), medidas
    assert ORCAMENTO_DECLARADO_MS == 100.0, "o orçamento declarado mudou sem ninguém o remedir"

    # A propriedade: com os rótulos de marca apagados, o MESMO desenho custa menos.
    muitas = figura_do_vip(
        importancia=importancia,
        n_features=40,
        recorte=recorte_do_vip(importancia=importancia, n_desenhadas=40),
    )
    com_rotulos = mediana_do_redesenho(muitas)
    muitas.axes[0].set_yticklabels([""] * 40)
    sem_rotulos = mediana_do_redesenho(muitas)
    assert sem_rotulos < com_rotulos, (
        f"apagar 40 rótulos de marca não baratear o redesenho ({sem_rotulos:.1f} ms contra "
        f"{com_rotulos:.1f} ms) significa que o custo deixou de ser do texto — e então o número "
        f"de marcas do VIP deixou de ser a variável a escolher. Medido em 11/09/2026: 75,8 contra "
        f"134,1 ms. As três figuras: {medidas}"
    )


def test_o_numero_de_marcas_do_vip_e_o_que_a_LATENCIA_permite() -> None:  # noqa: N802
    """O padrão medido, e a âncora de que ele não encostou na travessia.

    Medido neste clone: a travessia do teto de 100 ms está entre 16 e 20 marcas, e cada marca custa
    ~1,5 ms sobre uma base de ~74 ms que não depende de `n`. O padrão leva folga de propósito — a
    base é do texto fixo, e numa máquina mais lenta ela sobe toda.
    """
    assert FEATURES_DO_VIP == 12
    assert FEATURES_DO_VIP < 16, (
        "o padrão passou a estar na zona de travessia medida; se a medição mudou, remedir e "
        "registar em vez de subir o número"
    )


# ---------------------------------------------------------------------------
# 9. O registo: as três entram no catálogo, e a linha do `[G-1]` virou junto
# ---------------------------------------------------------------------------


def test_as_tres_estao_no_registo_com_a_chave_do_catalogo() -> None:
    """As três bijetam contra a entrada `PLS-DA` do `[G-1]`, e não contra `ITENS_OBRIGATORIOS`.

    A bijeção completa — contra a partição `construída` do catálogo — vive em
    `tests/unit/test_plots_arnes.py`, que é onde o portão está. Aqui fica a âncora de que as três
    funções deste módulo estão registadas, que é o que apanharia um construtor esquecido.
    """
    do_modulo = {c for c in CONSTRUTORES if c.modulo == "plsda"}
    assert {c.funcao for c in do_modulo} == {
        "figura_dos_scores_do_plsda",
        "figura_das_cargas_do_plsda",
        "figura_do_vip",
    }
    assert {c.chave for c in do_modulo} == {"PLS-DA"}
    assert {c.vinculo for c in do_modulo} == {Vinculo.CATALOGO}
    for construtor in do_modulo:
        assert hasattr(modulo, construtor.funcao)


def test_os_tres_titulos_sao_distintos_e_nenhum_e_vazio() -> None:
    """Âncora barata: três títulos iguais fariam o seletor do painel oferecer três vezes o mesmo."""
    titulos = {TITULO_DOS_SCORES, TITULO_DAS_CARGAS, TITULO_DO_VIP}
    assert len(titulos) == 3 and all(t.strip() for t in titulos)
    assert LIMIAR_DE_REFERENCIA_DO_VIP == 1.0
    assert "média de VIP²" in ROTULO_DA_REFERENCIA


def test_a_particao_por_familia_separa_e_os_DOIS_ramos_acusam() -> None:  # noqa: N802
    """O negativo do ramo NOVO — a partição `_e_do_plsda`, acrescentada em 18/09/2026.

    **A condição de nascimento exige ver os dois desfechos, e o portão acima passou a ter
    dois ramos.** O de cima — `q2` mais `permutation_pvalue` — já tinha planta
    (`test_o_oraculo_da_inviolavel_7_ACUSA_um_portador_plantado`); o de baixo, que exige só o `p`,
    nasceu sem nenhuma. Sem esta função, um `_e_do_plsda` que devolvesse **sempre `False`**
    deixaria de exigir `q2` dos três portadores de PLS-DA e **nada reprovaria** — a verificação
    que nunca acusa, num ramo com um dia de vida.

    **Três asserções, e a primeira é a que impede o colapso:** a partição tem de SEPARAR — se ela
    puser os quatro do mesmo lado, as duas outras passam por vacuidade sobre um conjunto só.
    """
    from pydantic import BaseModel as ModeloBase
    from pydantic import Field as Campo

    class SoComP(ModeloBase):
        """Um classificador honesto: medida de desempenho com o `p`, e sem `q2`."""

        balanced_accuracy: float
        permutation_pvalue: float

    class PlsdaSemQ2(ModeloBase):
        """Um portador de PLS-DA a que falta metade do par — o defeito que o ramo de cima apanha."""

        vip: list[float]
        permutation_pvalue: float

    class ClassificadorSemP(ModeloBase):
        """Um classificador sem validação nenhuma — o defeito que o ramo de BAIXO apanha."""

        balanced_accuracy: float
        permutation_pvalue: float = Campo(default=1.0)

    # 1. A partição SEPARA. Sem isto, o resto corre sobre um lado só e passa por vacuidade.
    assert not _e_do_plsda("SoComP", SoComP), (
        "um modelo com `permutation_pvalue` e sem `q2`/`vip`/«plsda» no nome NÃO é do PLS-DA — se "
        "a partição o puser do lado de lá, `q2` volta a ser exigido de quem não o pode ter"
    )
    assert _e_do_plsda("PlsdaSemQ2", PlsdaSemQ2), (
        "um modelo com `vip` É do PLS-DA — se a partição o deixar de fora, o ramo que exige `q2` "
        "fica sem ninguém e o portão deixa de guardar o par"
    )

    # 2. Os dois são acusados pelo DETECTOR — senão nem chegavam à exigência.
    assert _carrega_score_de_plsda("SoComP", SoComP)
    assert _carrega_score_de_plsda("PlsdaSemQ2", PlsdaSemQ2)

    # 3. O ramo de CIMA acusa: portador de PLS-DA sem `q2`.
    assert "q2" not in PlsdaSemQ2.model_fields, "a planta do ramo de cima NÃO aterrou"

    # 4. O ramo de BAIXO acusa: classificador cujo `p` ganhou padrão. É o mesmo defeito que a
    #    regra 6 do README nomeia — com padrão passa a existir construção válida sem validação.
    campo = ClassificadorSemP.model_fields["permutation_pvalue"]
    assert not campo.is_required(), "a planta do ramo de baixo NÃO aterrou"
    assert not _e_do_plsda("ClassificadorSemP", ClassificadorSemP), (
        "a planta do ramo de baixo caiu do lado errado da partição e não exercita o ramo novo"
    )


# ---------------------------------------------------------------------------
# A cor e o símbolo de cada GRUPO
# ---------------------------------------------------------------------------
#
# Antes o gráfico de scores pintava por LADO do contraste: a referência com a primeira cor do
# ciclo e a classe positiva com a segunda. O mesmo grupo mudava de cor conforme o contraste em que
# entrava. Com estilo, a cor é do grupo; sem ele, continua a de hoje, e os testes acima não mudaram.


def _escores_do_contraste(negativo: str, positivo: str) -> EscoresDoPlsda:
    """Três amostras de cada lado, com coordenadas distintas. **O dado não é o assunto aqui.**"""
    return escores_do_plsda(
        comp1=[-2.0, -1.5, -1.0, 1.0, 1.5, 2.0],
        comp2=[0.5, -0.5, 0.25, -0.25, 0.75, -0.75],
        grupos=[negativo, negativo, negativo, positivo, positivo, positivo],
        grupo_positivo=positivo,
        grupo_negativo=negativo,
        validacao=VALIDACAO_DA_RENDERIZACAO,
    )


def _scores(escores: EscoresDoPlsda, estilos: EstilosDosGrupos | None = None) -> Figure:
    return figura_dos_scores_do_plsda(
        escores=escores,
        recorte=recorte_dos_scores_do_plsda(escores=escores, n_features_totais=8),
        estilos=estilos,
    )


def _linha_do_grupo(figura: Figure, grupo: str) -> Line2D:
    """A série de um grupo, pelo nome no início do rótulo — o papel vem depois e muda de lado."""
    achadas = [
        linha for linha in figura.axes[0].lines if str(linha.get_label()).startswith(f"{grupo} — ")
    ]
    assert len(achadas) == 1, [str(linha.get_label()) for linha in figura.axes[0].lines]
    return achadas[0]


def _linha_da_regiao(figura: Figure, grupo: str) -> Line2D:
    """A linha da região de predição de um grupo, pelo rótulo inteiro."""
    rotulo = ROTULO_DA_REGIAO_NO_TESTE.format(grupo=grupo)
    achadas = [linha for linha in figura.axes[0].lines if str(linha.get_label()) == rotulo]
    assert len(achadas) == 1, [str(linha.get_label()) for linha in figura.axes[0].lines]
    return achadas[0]


def test_com_CONTROLE_vermelho_e_quadrado_a_serie_do_controle_sai_assim() -> None:  # noqa: N802
    """O cenário do pedido, pelo arnês: a série do Controle tem os pontos dela, a cor e o `s`.

    **O arnês afirma o DADO antes da aparência**, e é o que impede o teste de passar sobre a série
    errada: a cor certa no artista do Caso seria o defeito, com a figura bem formada.

    **O par causal é a mesma chamada sem o estilo**: o Controle volta à primeira cor do ciclo e ao
    círculo. Se as duas saíssem iguais, a planta não separaria nada.
    """
    escores = _escores_do_contraste("Controle", "Caso")
    estilos = estilos_dos_grupos(
        ("Controle", "Caso"), {"Controle": EstiloDoGrupo(cor="vermelho", simbolo="quadrado")}
    )
    figura = _scores(escores, estilos)

    esperadas = esperadas_dos_scores(escores)
    controle = "Controle — 3 amostras (classe de referência, y = 0)"
    assert np.array_equal(esperadas[controle][0], escores.x[:3]), "âncora: a série é a do Controle"
    assert confere_series(figura.axes[0], esperadas) == []
    linha = _linha_do_grupo(figura, "Controle")
    assert (to_hex(linha.get_color()), linha.get_marker()) == (to_hex("#D55E00"), "s")
    # **A região do grupo leva a cor DELE**: é a cor que diz de quem é cada região.
    regiao = _linha_da_regiao(figura, "Controle")
    assert to_hex(regiao.get_color()) == to_hex("#D55E00")

    sem = _linha_do_grupo(_scores(escores), "Controle")
    assert sem.get_marker() == "o"
    assert to_hex(sem.get_color()) == to_hex(cores_dos_lados()[0])
    assert to_hex(sem.get_color()) != to_hex("#D55E00"), "âncora: a planta não separa"


def test_o_mesmo_grupo_tem_a_MESMA_cor_em_dois_contrastes_com_o_estilo_da_coorte() -> None:  # noqa: N802
    """`B` é o `y = 1` de um contraste e o `y = 0` do outro, e a cor dele não muda.

    **O estilo vem da COORTE e não dos dois grupos presentes** — é a razão de o parâmetro existir:
    a ordem canônica de dois grupos muda de contraste para contraste, a da coorte não.

    **O par causal é o desenho de hoje**, por lado: sem estilo, `B` sai com a cor da classe
    positiva num e com a da referência no outro. É o defeito que o pedido descreve.
    """
    coorte = estilos_dos_grupos(["A", "B", "C"])
    assert coorte is not None

    def cores_de_b(estilos: EstilosDosGrupos | None) -> tuple[str, str]:
        positivo = _linha_do_grupo(_scores(_escores_do_contraste("A", "B"), estilos), "B")
        negativo = _linha_do_grupo(_scores(_escores_do_contraste("B", "C"), estilos), "B")
        return to_hex(positivo.get_color()), to_hex(negativo.get_color())

    no_primeiro, no_segundo = cores_de_b(coorte)
    assert no_primeiro == no_segundo == to_hex(coorte.por_grupo["B"].hex)

    hoje_positivo, hoje_negativo = cores_de_b(None)
    assert hoje_positivo != hoje_negativo, (
        "âncora: sem estilo B já tinha a mesma cor nos dois contrastes — o teste não separaria nada"
    )
    assert (hoje_negativo, hoje_positivo) == tuple(to_hex(c) for c in cores_dos_lados())


# ---------------------------------------------------------------------------
# Os escores de VALIDAÇÃO CRUZADA e a região de cada grupo
# ---------------------------------------------------------------------------
#
# O gráfico de escores passou a mostrar os escores que o modelo NÃO viu, e uma região de predição de
# Hotelling T² por grupo sobre eles. Os de calibração tendem a separar, e com muito mais features
# que amostras separam mesmo sobre rótulos sem informação (medido em `statistics/plsda.py`); a
# região sobre eles reforçaria uma separação que o Q² desmente, que é o erro da regra 6 do README.


def test_a_regiao_de_cada_grupo_e_a_DAQUELE_grupo_e_na_cor_da_serie_dele(sintetico: dict) -> None:  # noqa: N802
    """**A população da região é o que ela afirma**, e a planta é a região sobre a nuvem toda.

    Uma região calculada sobre os dois grupos juntos descreveria onde cai uma amostra qualquer do
    contraste — e cresceria para conter os dois grupos, dizendo nada sobre se eles se separam. A
    âncora é que, neste sintético, as duas populações dão regiões DIFERENTES: sem ela o oráculo
    aceitaria a planta por coincidência.

    **E a cor da região é a da série do grupo**: é a única coisa no desenho que diz de quem é cada
    região, já que as duas são tracejadas.
    """
    escores: EscoresDoPlsda = sintetico["escores"]  # type: ignore[assignment]
    figura = _scores(escores)
    eixo = figura.axes[0]
    esperadas = esperadas_dos_scores(escores)
    assert confere_series(eixo, esperadas) == []

    finitos = np.isfinite(escores.x) & np.isfinite(escores.y)
    da_nuvem = elipse_de_hotelling(escores.x[finitos], escores.y[finitos])
    assert isinstance(da_nuvem, Elipse)
    plantadas = dict(esperadas)
    plantadas[ROTULO_DA_REGIAO_NO_TESTE.format(grupo="caso")] = da_nuvem.contorno()
    assert confere_series(eixo, plantadas) != [], (
        "âncora: a região da nuvem inteira coincide com a do grupo neste sintético, e o oráculo "
        "não distinguiria a população certa da errada"
    )

    for grupo in ("controle", "caso"):
        serie = _linha_do_grupo(figura, grupo)
        regiao = _linha_da_regiao(figura, grupo)
        assert to_hex(regiao.get_color()) == to_hex(serie.get_color()), grupo
        assert regiao.get_linestyle() == "--", "região contínua leria-se como dado"
    assert to_hex(_linha_da_regiao(figura, "controle").get_color()) != to_hex(
        _linha_da_regiao(figura, "caso").get_color()
    ), "as duas regiões com a mesma cor não dizem de quem é cada uma"


def _regioes_por_baixo_dos_pontos(figura: Figure) -> list[str]:
    """O oráculo: toda linha de região tem `zorder` MENOR que toda série de pontos.

    Devolve a lista em vez de afirmar, para a planta entrar pelo mesmo caminho do positivo.
    """
    linhas = figura.axes[0].lines
    regioes = [linha for linha in linhas if str(linha.get_label()).startswith("Região de predição")]
    pontos = [linha for linha in linhas if linha not in regioes]
    assert regioes and pontos, (
        "âncora: sem regiões ou sem pontos a ordem seria satisfeita por vacuidade"
    )
    teto = min(linha.get_zorder() for linha in pontos)
    return [
        f"{linha.get_label()} com zorder {linha.get_zorder()}, e os pontos começam em {teto}"
        for linha in regioes
        if linha.get_zorder() >= teto
    ]


def test_a_regiao_fica_POR_BAIXO_dos_pontos(  # noqa: N802
    sintetico: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    """**A região é sobre a nuvem e não pode esconder uma amostra** — e o teste consegue reprovar.

    Ponta A: na figura construída, as duas regiões ficam por baixo das duas séries. Ponta B: a MESMA
    figura com a ordem da região posta acima da dos pontos — a planta entra pelo módulo, no nome que
    `_desenha_as_regioes` lê na chamada —, e o oráculo acusa as duas. **A planta é conferida antes
    do desfecho:** se a ordem plantada não chegasse às linhas, a ponta B passaria sem ter plantado.
    """
    escores: EscoresDoPlsda = sintetico["escores"]  # type: ignore[assignment]
    assert _regioes_por_baixo_dos_pontos(_scores(escores)) == []

    monkeypatch.setattr(modulo, "_ORDEM_DA_REGIAO", 5.0)
    plantada = _scores(escores)
    assert _linha_da_regiao(plantada, "caso").get_zorder() == 5.0, "a planta não aterrou"
    assert len(_regioes_por_baixo_dos_pontos(plantada)) == 2


def _escores_pequenos(*, alinhados: bool) -> EscoresDoPlsda:
    """Dois grupos sem região: dois pontos em cada um, ou quatro pontos alinhados em cada um."""
    if alinhados:
        comp1 = [-4.0, -3.0, -2.0, -1.0, 1.0, 2.0, 3.0, 4.0]
        comp2 = [-8.0, -6.0, -4.0, -2.0, 2.0, 4.0, 6.0, 8.0]
        grupos = ["controle"] * 4 + ["caso"] * 4
    else:
        comp1 = [-2.0, -1.0, 1.0, 2.0]
        comp2 = [0.5, -0.5, 0.25, -0.25]
        grupos = ["controle"] * 2 + ["caso"] * 2
    return escores_do_plsda(
        comp1=comp1,
        comp2=comp2,
        grupos=grupos,
        grupo_positivo="caso",
        grupo_negativo="controle",
        validacao=VALIDACAO_DA_RENDERIZACAO,
    )


@pytest.mark.parametrize(
    ("alinhados", "motivo"),
    [
        (False, "menos de 3 amostras (2)"),
        (True, "a covariância destes pontos é singular ou não finita"),
    ],
)
def test_sem_regiao_o_eixo_DIZ_por_que(alinhados: bool, motivo: str) -> None:  # noqa: N802
    """Os dois motivos de `ElipseAusente`, cada um com a frase dele — região ausente sem explicação
    lê-se como região que o construtor esqueceu, e a figura diz o porquê em vez de calar.

    **Nenhuma linha de região e nenhuma entrada na legenda**: a legenda que prometesse uma região
    que não está desenhada ensinaria a procurá-la. **O motivo vai no eixo**, como na PCA da
    correção: é o único sítio da figura com lugar para ele.
    """
    escores = _escores_pequenos(alinhados=alinhados)
    figura = _scores(escores)
    eixo = figura.axes[0]

    assert confere_series(eixo, esperadas_dos_scores(escores)) == []
    assert len(eixo.lines) == 2, [str(linha.get_label()) for linha in eixo.lines]
    legenda = eixo.get_legend()
    assert legenda is not None
    assert len(legenda.get_texts()) == 2, "âncora: a legenda tem as duas séries"
    assert not any("Região" in t.get_text() for t in legenda.get_texts())

    no_eixo = " ".join(" ".join(t.get_text().split()) for t in eixo.texts)
    for grupo in ("controle", "caso"):
        assert f"sem região de {grupo}: {motivo}" in no_eixo, no_eixo


def _falta_o_par_no_titulo(figura: Figure, validacao: Validacao) -> list[str]:
    """O oráculo: o título DESENHADO traz o `Q²` e o `p`, escritos como a figura os escreve.

    Devolve a lista em vez de afirmar, e é o que deixa a planta entrar como argumento: o negativo
    percorre exactamente o caminho do positivo.
    """
    titulo = " ".join(figura.axes[0].get_title().split())
    achados: list[str] = []
    for nome, valor in (("Q²", validacao.q2), ("p de permutação", validacao.permutation_pvalue)):
        if f"{nome} = {decimal(valor)}" not in titulo:
            achados.append(f"o título não traz {nome} = {decimal(valor)}: {titulo!r}")
    return achados


def test_tirar_o_Q2_do_titulo_dos_escores_REPROVA(  # noqa: N802
    sintetico: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    """**A regra 6 do README na figura que mudou de escores, e o teste consegue reprovar (regra
    3).**

    Ponta A: o título da figura dos escores de validação cruzada traz o `Q²` e o `p`. Ponta B: a
    MESMA figura, com a frase do título trocada por uma sem o `Q²` — a planta entra pelo construtor,
    no nome que ele resolve na chamada —, e o oráculo acusa. **A planta é conferida antes do
    desfecho:** se o título plantado ainda trouxesse o `Q²`, a ponta B passaria sem ter plantado.
    """
    escores: EscoresDoPlsda = sintetico["escores"]  # type: ignore[assignment]
    val: Validacao = sintetico["validacao"]  # type: ignore[assignment]
    assert _falta_o_par_no_titulo(_scores(escores), val) == []

    monkeypatch.setattr(
        modulo,
        "frase_curta_da_validacao",
        lambda v: f"p de permutação = {decimal(v.permutation_pvalue)}",
    )
    plantada = _scores(escores)
    assert "Q²" not in plantada.axes[0].get_title(), "a planta não aterrou no título"
    achados = _falta_o_par_no_titulo(plantada, val)
    assert len(achados) == 1 and "Q²" in achados[0], achados


def test_o_rodape_DIZ_que_pontos_sao_estes_e_a_legenda_que_regioes(sintetico: dict) -> None:  # noqa: N802
    """O rodapé declara os escores de validação cruzada, e a legenda o tipo de cada região: a figura
    declara o método na própria imagem.

    **E o rodapé deixa de dizer que a separação desenhada é ajuste dentro da amostra**, que era
    verdade dos escores de calibração e é falso dos de validação cruzada. A âncora da ausência é a
    presença do veredito do mesmo ramo — `Q² <= 0` —, senão a frase faltaria por faltar o ramo.

    **A frase dos escores entra no lugar de `PRETRATAMENTO` e não é mais longa que ela**, e as
    asserções de comprimento e de ausência são as que guardam isso. Medido com os nomes e as
    contagens do contraste do `pos_hilic` e 758 features, e a folga depende da FONTE: na DejaVu
    Sans, a do produto, a frase AO LADO de `PRETRATAMENTO` põe o rodapé por cima do rótulo do eixo
    no painel da janela a 1280x860 (6,5 x 3,4 pol.) — -5,42 pt com Q² <= 0 e -16,22 pt com Q² > 0,
    contra +12,63 e +5,38 pt no lugar dela, os da base —; na Inter do `MATPLOTLIBRC` do contêiner
    de desenvolvimento, já no tamanho da construção — -4,10 pt contra os +0,44 pt da base. O número
    de cada medição está ao lado de `ESCORES_DE_VALIDACAO_CRUZADA`. O que a frase deixou de repetir
    — a matriz e o pré-tratamento — está no título e no campo `matriz` do mesmo rodapé, e o sinal
    canônico ela leva. **O comprimento é o proxy, e é declarado:** a folga em pontos depende das
    métricas da fonte de cada sistema, e o número de caracteres não.
    """
    origem: EscoresDoPlsda = sintetico["escores"]  # type: ignore[assignment]
    escores = escores_do_plsda(
        comp1=origem.x,
        comp2=origem.y,
        grupos=list(origem.grupos),
        grupo_positivo=origem.positivo,
        grupo_negativo=origem.negativo,
        validacao=VALIDACAO_DA_RENDERIZACAO,
    )
    figura = _scores(escores)
    rodape = rodape_de(figura)

    assert "escores de VALIDAÇÃO CRUZADA" in rodape, rodape
    assert "dobra que não a viu" in rodape and "Procrustes" in rodape, rodape
    assert "a matriz processada" in rodape, "o pré-tratamento saiu do rodapé inteiro"
    legenda = figura.axes[0].get_legend()
    assert legenda is not None
    textos = [t.get_text() for t in legenda.get_texts()]
    for grupo in ("controle", "caso"):
        assert ROTULO_DA_REGIAO_NO_TESTE.format(grupo=grupo) in textos, textos

    assert "Q² <= 0" in rodape, "âncora: o veredito do ramo negativo não está no rodapé"
    assert "separação desenhada é ajuste dentro da amostra" not in rodape, rodape

    assert len(ESCORES_DE_VALIDACAO_CRUZADA) <= len(PRETRATAMENTO), (
        f"a frase dos escores tem {len(ESCORES_DE_VALIDACAO_CRUZADA)} caracteres e a que ela "
        f"substitui tem {len(PRETRATAMENTO)}: o rodapé pode passar a ter mais linhas que o da "
        "base, e no painel da janela ele passa por cima do rótulo do eixo x"
    )
    # **No lugar, e não ao lado.** A asserção de comprimento guarda a CONSTANTE; esta guarda o USO —
    # `PRETRATAMENTO` posto ao lado da frase nova passaria pela de comprimento e daria as 11 e 12
    # linhas que, na DejaVu Sans e no painel da janela, põem o rodapé sobre o rótulo do eixo.
    # A âncora é a frase nova achada pela MESMA normalização: sem ela, a ausência passaria por o
    # rodapé estar partido de um jeito que a normalização não junta.
    assert " ".join(ESCORES_DE_VALIDACAO_CRUZADA.split()) in rodape, rodape
    assert " ".join(PRETRATAMENTO.split()) not in rodape, rodape
    # **O critério com o referente dele.** `PRETRATAMENTO` levava a orientação canônica do
    # sinal, e a frase que a substitui tem de a levar: os escores de validação cruzada são girados
    # para a calibração JÁ virada por ela, e a etiqueta do critério sem o que ele afirma seria
    # citação sem sujeito. A primeira redação da frase nova perdeu-a e manteve a etiqueta.
    assert "girada por Procrustes à calibração de sinal canônico" in rodape, rodape
