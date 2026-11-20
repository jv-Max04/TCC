"""O PLS-DA passou a ser CALCULADO por `POST /execucoes`.

## O que mudou, e o que NÃO mudou

**O algoritmo não é novo.** `src/refino/statistics/plsda.py` existia inteiro antes, com
`Q²` por validação cruzada estratificada, o p de permutação na convenção `(k+1)/(n+1)`, e onze
testes em `tests/unit/test_plsda.py`. O que faltava era **a chamada**: medido em 27/08/2026, `grep`
por `plsda(` em `src/` fora do próprio módulo achava apenas o painel a **exibir**.

> É o alvo errado na forma mais cara: o mecanismo estava correto, verde, e **o alvo dele não
> existia**. A tela dizia *"sem caminho"* nas duas métricas.

Esta suíte cobre a **ligação**, não o algoritmo. O algoritmo continua coberto por
`tests/unit/test_plsda.py`, e duplicar aquelas asserções aqui criaria a segunda fonte de verdade
sobre o que o PLS-DA calcula.

## A REGRA 6 DO README tem agora DUAS guardas, e elas são de camadas diferentes

| Onde | O que impede |
|---|---|
| `PLSDAResult` — dez campos obrigatórios, congelado | produzir scores sem `q2` e sem `p` |
| `QualityReport._check_plsda_completo` | **relatar** um sem o outro, ou sem procedência |

A segunda é nova e não é redundante: os dois campos chegam ao relatório como **opcionais**, e um
chamador distraído poderia preencher um só. Sem ela, a regra 6 do README estaria garantida em quem
calcula e não em quem publica.

## Escopo negativo

**Nenhum teste aqui corre um PLS-DA real sobre a matriz de referência.** Aquilo são 200 permutações
sobre 6.035 features, e o custo não é de teste de unidade; o que se afirma é que a **chamada
acontece**, que a guarda recusa no disparo, e que o par viaja inteiro. O caminho ponta a ponta com
dado real fica para `-m slow`, e a ausência vai declarada em vez de residual.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient

from refino.api.app import build_app
from refino.api.rotas import PLSDA_CV_FOLDS
from refino.domain.spec import QualityReport
from refino.io.readers import load_matrix
from refino.metrics.report import build_quality_report
from refino.statistics.plsda import PLSDAResult

FIXTURES = Path(__file__).resolve().parent.parent / "data" / "fixtures"


def _plsda_construido(*, q2: float = 0.24, r2y: float = 0.31) -> PLSDAResult:
    """Um `PLSDAResult` com os dez campos, sem correr o algoritmo.

    **Construir é seguro aqui porque a classe é congelada e sem padrões**: não há estado que a
    construção manual possa produzir e que o algoritmo não produza. O que se testa nesta suíte é a
    ligação, e correr 200 permutações para a exercitar seria medir outra coisa.
    """
    return PLSDAResult(
        scores=np.zeros((4, 2)),
        cv_scores=np.zeros((4, 2)),
        loadings=np.zeros((3, 2)),
        vip=np.zeros(3),
        r2y=r2y,
        q2=q2,
        permutation_pvalue=0.3,
        n_components=2,
        n_permutations=200,
        seed=42,
    )


def _relatorio_minimo(*, computed_extra: set[str] | None = None, **extra: Any) -> QualityReport:
    """Um `QualityReport` válido com o mínimo obrigatório, para as guardas serem o único assunto.

    `computed_extra` existe porque `plsda_q2` e `plsda_permutation_p` estão em
    `OPTIONAL_METRIC_FIELDS`, e a bicondicional dispara ANTES da guarda do par quando o
    campo tem valor e o nome está fora de `computed`. Sem este parâmetro, os testes do par mediriam
    a guarda e acreditariam ter medido a.
    """
    base: dict[str, Any] = {
        "measurement_point": "post_pipeline",
        "rsd_variant": "plain",
        "rsd_estimator": "in_sample",
        "structural_pretransform": "none",
        "computed": frozenset(
            {
                "qc_rsd_median",
                "qc_rsd_mean",
                "frac_features_under_20",
                "frac_features_under_30",
                "qc_dispersion_ratio",
                "pc1_injection_correlation_abs",
            }
        ),
        "qc_rsd_median": 1.0,
        "qc_rsd_mean": 1.0,
        "frac_features_under_20": 0.5,
        "frac_features_under_30": 0.5,
        "n_features_rsd_undefined": 0,
        # **3 e não zero.** As duas frações acima são `0.5`, e a bicondicional de
        # `_check_denominador_do_rsd` diz que fração definida exige denominador positivo. Três é
        # o tamanho da matriz que esta base descreve, e é o que a mantém coerente com as zero
        # indefinidas da linha acima.
        "n_features_rsd_defined": 3,
        "qc_dispersion_ratio": 1.0,
        "pc1_injection_correlation_abs": 0.1,
        "structural_features_used": 3,
        "structural_features_dropped_zero_variance": 0,
        "structural_features_dropped_non_finite": 0,
        "structural_column_digest": "abc",
        "qc_dispersion_ratio_spread": 0.0,
        "pc1_injection_correlation_spread": 0.0,
        "stability_replicates": 1,
        "stability_seed": 42,
    }
    if computed_extra:
        base["computed"] = frozenset(base["computed"] | computed_extra)
    return QualityReport(**{**base, **extra})


# ---------------------------------------------------------------------------
# A regra 6 do README no relatório — quatro desfechos
# ---------------------------------------------------------------------------


def test_nenhum_dos_dois_e_o_caminho_de_antes_e_continua_valido() -> None:
    """**O ramo que dispara por AUSÊNCIA**, e é o de toda execução que não declara par.

    Sem esta âncora, uma guarda escrita para exigir o par tornaria inválido todo relatório
    existente — e o defeito apareceria como a suíte inteira a reprovar, não como um teste.
    """
    relatorio = _relatorio_minimo()

    assert relatorio.plsda_q2 is None
    assert relatorio.plsda_permutation_p is None
    assert "plsda_q2" not in relatorio.computed


@pytest.mark.parametrize(
    ("campos", "porque"),
    [
        ({"plsda_q2": 0.24}, "q2 sem o p"),
        ({"plsda_permutation_p": 0.3}, "p sem o q2"),
    ],
)
def test_meio_par_e_recusado_nas_duas_direcoes(campos: dict[str, Any], porque: str) -> None:
    """**As DUAS direções**, e não só a intuitiva.

    Um Q² sem o p é o falso positivo que a regra 6 do README existe para expor; um p sem o Q² é um
    teste sem a quantidade que ele testa. Exigir só a primeira deixaria a segunda passar em
    silêncio.

    **O nome vai em `computed` de propósito**, e a razão é que sem ele a guarda que dispara é a
    bicondicional — *"está fora de `computed` e não vale `None`"* —, que é OUTRA guarda e
    outra mensagem. Este teste tem de exercitar a do meio par; deixá-lo cair na mediria um
    mecanismo diferente e passaria a acreditar que este existe.
    """
    with pytest.raises(ValueError, match="meio par"):
        _relatorio_minimo(computed_extra=set(campos), **campos)


def test_o_par_sem_procedencia_e_recusado() -> None:
    """**No dataclass:** número sem a população em que foi medido é recusado.

    Um `Q²` sem `plsda_groups` não diz que pergunta respondeu — os dados reais têm quatro grupos e
    portanto seis pares —, e sem a semente e as permutações o `p` não é reproduzível (regra 3 do
    README).
    """
    with pytest.raises(ValueError, match="sem procedência"):
        _relatorio_minimo(
            computed_extra={"plsda_q2", "plsda_permutation_p"},
            plsda_q2=0.24,
            plsda_permutation_p=0.3,
        )


@pytest.mark.parametrize("grupos", [("A", "A"), ("A",), ("A", "")])
def test_par_de_grupos_degenerado_e_recusado(grupos: tuple[str, ...]) -> None:
    """Par igual, par de um, e par com nome vazio. Nenhum dos três é contraste."""
    with pytest.raises(ValueError, match="par de nomes distintos"):
        _relatorio_minimo(
            computed_extra={"plsda_q2", "plsda_permutation_p"},
            plsda_q2=0.24,
            plsda_permutation_p=0.3,
            plsda_groups=grupos,  # type: ignore[arg-type]
            plsda_cv_folds=7,
            plsda_n_permutations=200,
            plsda_seed=42,
        )


# ---------------------------------------------------------------------------
# A regra 6 do README na família MULTICLASSE
# ---------------------------------------------------------------------------
#
# **A guarda conhecia SÓ o par binário antes da extensão.** Um relatório multiclasse com a medida e
# sem o `p` construía-se sem nada a acusar — o buraco que `_check_plsda_completo` existe para
# fechar, reaberto na classe nova. A tabela `FAMILIAS_DO_PLSDA` é o que o fecha para as duas de uma
# vez.

_MULTICLASSE_COMPLETO: dict[str, Any] = {
    "plsda_multiclass_q2": (0.31, 0.12, -0.04, 0.22),
    "plsda_multiclass_permutation_p": (0.0476, 0.2381, 0.7143, 0.0952),
    "plsda_multiclass_groups": ("ATLL", "Asymptomatic", "HAMTSP", "NoHTLV"),
    "plsda_multiclass_n_components": 2,
    "plsda_multiclass_cv_folds": 7,
    "plsda_multiclass_n_permutations": 20,
    "plsda_multiclass_seed": 42,
}
_COMPUTED_MULTICLASSE = {"plsda_multiclass_q2", "plsda_multiclass_permutation_p"}


def test_o_par_multiclasse_completo_e_aceito() -> None:
    """**A âncora da família nova.** Sem ela, as recusas abaixo seriam satisfeitas por uma guarda
    que recusasse todo relatório multiclasse — e o mecanismo pareceria funcionar sem nunca ter
    deixado passar o caso legítimo.
    """
    relatorio = _relatorio_minimo(computed_extra=_COMPUTED_MULTICLASSE, **_MULTICLASSE_COMPLETO)

    assert relatorio.plsda_multiclass_groups == ("ATLL", "Asymptomatic", "HAMTSP", "NoHTLV")
    assert relatorio.plsda_multiclass_q2 == (0.31, 0.12, -0.04, 0.22)
    assert "plsda_multiclass_q2" in relatorio.computed


@pytest.mark.parametrize(
    ("ausente", "porque"),
    [
        ("plsda_multiclass_permutation_p", "a medida sem o p"),
        ("plsda_multiclass_q2", "o p sem a medida"),
    ],
)
def test_MEIO_PAR_MULTICLASSE_e_recusado_nas_duas_direcoes(  # noqa: N802
    ausente: str, porque: str
) -> None:
    """**O cenário obrigatório, e é ele que prova que a extensão do passo 6 é mecanismo.**

    Planta: um relatório multiclasse com a medida presente e o `p` **ausente**. Sem a extensão,
    `_check_plsda_completo` olhava para `plsda_q2`/`plsda_permutation_p` e este relatório
    construía-se — desempenho publicado sem o teste que diz se ele é ruído, na classe nova.

    **As duas direções**, e não só a intuitiva: um `p` sem a medida é um teste sem a quantidade que
    ele testa, e exigir só a primeira deixaria a segunda passar em silêncio.
    """
    campos = {k: v for k, v in _MULTICLASSE_COMPLETO.items() if k != ausente}
    computed = _COMPUTED_MULTICLASSE - {ausente}

    with pytest.raises(ValueError, match="meio par"):
        _relatorio_minimo(computed_extra=computed, **campos)


def test_o_par_multiclasse_sem_procedencia_e_recusado() -> None:
    """Quatro `Q²` sem saber de que classes, com que dobras nem com que
    semente não dizem que perguntas responderam."""
    campos = {
        k: v
        for k, v in _MULTICLASSE_COMPLETO.items()
        if k not in {"plsda_multiclass_groups", "plsda_multiclass_seed"}
    }
    with pytest.raises(ValueError, match="sem procedência"):
        _relatorio_minimo(computed_extra=_COMPUTED_MULTICLASSE, **campos)


@pytest.mark.parametrize(
    ("troca", "pedaco"),
    [
        ({"plsda_multiclass_q2": (0.31, 0.12)}, "desalinhou"),
        ({"plsda_multiclass_groups": ("ATLL",)}, "desalinhou"),
        (
            {
                "plsda_multiclass_q2": (0.31,),
                "plsda_multiclass_permutation_p": (0.0476,),
                "plsda_multiclass_groups": ("ATLL",),
            },
            "abaixo de 2",
        ),
        (
            {"plsda_multiclass_groups": ("ATLL", "ATLL", "HAMTSP", "NoHTLV")},
            "repetido ou vazio",
        ),
        (
            {"plsda_multiclass_groups": ("ATLL", "", "HAMTSP", "NoHTLV")},
            "repetido ou vazio",
        ),
    ],
)
def test_colecoes_multiclasse_degeneradas_sao_recusadas(troca: dict[str, Any], pedaco: str) -> None:
    """**As três coleções descrevem os MESMOS K modelos, e cada planta quebra uma cláusula.**

    Comprimento diferente, uma classe só, nome repetido e nome vazio. Sem isto,
    `plsda_multiclass_q2[2]` poderia responder por outra classe que não
    `plsda_multiclass_groups[2]` — número certo sobre a população errada, e nada levanta.
    """
    with pytest.raises(ValueError, match=pedaco):
        _relatorio_minimo(
            computed_extra=_COMPUTED_MULTICLASSE, **{**_MULTICLASSE_COMPLETO, **troca}
        )


def test_o_par_multiclasse_com_valor_e_FORA_de_computed_e_recusado() -> None:  # noqa: N802
    """**A bicondicional para os dois campos que ficam fora de `OPTIONAL_METRIC_FIELDS`.**

    Eles ficam fora daquele conjunto porque não são das onze métricas escalares da tela — são
    coleções de K —, e `_check_computed_membership` só percorre a bicondicional sobre aquele
    conjunto. Sem esta cláusula, um par presente e fora de `computed` seria valor que ninguém pode
    saber se vale, que é exactamente o defeito que a guarda fecha.
    """
    with pytest.raises(ValueError, match="não está em `computed`"):
        _relatorio_minimo(**_MULTICLASSE_COMPLETO)


def test_as_duas_familias_convivem_no_mesmo_relatorio() -> None:
    """Binário e multiclasse ao mesmo tempo, e nenhuma das guardas confunde uma com a outra.

    É o que a tabela `FAMILIAS_DO_PLSDA` compra: a guarda percorre famílias em vez de conhecer uma.
    """
    relatorio = _relatorio_minimo(
        computed_extra={"plsda_q2", "plsda_permutation_p"} | _COMPUTED_MULTICLASSE,
        plsda_q2=0.24,
        plsda_permutation_p=0.3,
        plsda_groups=("ATLL", "NoHTLV"),
        plsda_cv_folds=7,
        plsda_n_permutations=200,
        plsda_seed=42,
        **_MULTICLASSE_COMPLETO,
    )

    assert relatorio.plsda_q2 == 0.24
    assert relatorio.plsda_multiclass_q2 is not None
    assert len(relatorio.plsda_multiclass_q2) == 4


def test_par_causal_o_multiclasse_muda_o_computed_e_nada_mais_muda(matriz: Any) -> None:
    """**Duas chamadas idênticas menos por `plsda_multiclass=`, e a asserção é que DIVERGEM.**

    Um `build_quality_report` que ignorasse o argumento devolveria relatórios iguais, e todas as
    asserções de estrutura acima continuariam verdes.
    """
    from refino.statistics.plsda_multiclasse import MulticlassPLSDAResult, OneVsRestModel

    def _modelo(nome: str, q2: float) -> OneVsRestModel:
        return OneVsRestModel(
            group=nome,
            scores=np.zeros((4, 2)),
            loadings=np.zeros((3, 2)),
            vip=np.zeros(3),
            r2y=0.5,
            q2=q2,
            permutation_pvalue=0.1,
        )

    multi = MulticlassPLSDAResult(
        models=(_modelo("A", 0.31), _modelo("B", 0.12), _modelo("C", -0.04)),
        groups=("A", "B", "C"),
        n_components=2,
        cv_folds=7,
        n_permutations=20,
        seed=42,
    )
    comum = {"measurement_point": "post_pipeline", "structural_pretransform": "none"}
    sem = build_quality_report(matriz, **comum)  # type: ignore[arg-type]
    com = build_quality_report(matriz, **comum, plsda_multiclass=multi)  # type: ignore[arg-type]

    assert set(com.computed) - set(sem.computed) == _COMPUTED_MULTICLASSE
    assert sem.plsda_multiclass_q2 is None and com.plsda_multiclass_q2 == (0.31, 0.12, -0.04)
    assert com.plsda_multiclass_groups == ("A", "B", "C")
    assert com.plsda_multiclass_seed == 42 and com.plsda_multiclass_n_permutations == 20
    assert com.qc_rsd_median == sem.qc_rsd_median, (
        "declarar o multiclasse mudou uma métrica que não é do PLS-DA"
    )
    assert com.plsda_q2 is None, "o multiclasse preencheu o par BINÁRIO: as famílias colidiram"


def test_o_par_completo_com_procedencia_e_aceito() -> None:
    """O caminho feliz. Sem ele, as três recusas acima seriam satisfeitas por um dataclass que
    recusasse tudo."""
    relatorio = _relatorio_minimo(
        computed_extra={"plsda_q2", "plsda_permutation_p"},
        plsda_q2=0.24,
        plsda_permutation_p=0.3,
        plsda_groups=("ATLL", "NoHTLV"),
        plsda_r2y=0.31,
        plsda_n_components=2,
        plsda_cv_folds=7,
        plsda_n_permutations=200,
        plsda_seed=42,
    )

    assert relatorio.plsda_groups == ("ATLL", "NoHTLV")


# ---------------------------------------------------------------------------
# `build_quality_report` — o par causal
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def matriz() -> Any:
    return load_matrix(FIXTURES / "tiny_clean.csv", FIXTURES / "tiny_clean_sample_list.csv", "tiny")


def test_par_causal_o_plsda_muda_o_computed_e_nada_mais_muda(matriz: Any) -> None:
    """**Duas chamadas idênticas menos por `plsda=`, e a asserção é que DIVERGEM.**

    Um `build_quality_report` que ignorasse o argumento devolveria relatórios iguais, e todos os
    testes de estrutura acima continuariam verdes — a planta tem de cair sobre dado em que as duas
    coisas diferem.

    **E a segunda metade é que o RESTO não muda:** o PLS-DA não pode alterar as seis obrigatórias,
    senão declarar o par mudaria números que nada têm a ver com ele.
    """
    comum = {
        "measurement_point": "post_pipeline",
        "structural_pretransform": "none",
    }
    sem = build_quality_report(matriz, **comum)  # type: ignore[arg-type]
    com = build_quality_report(
        matriz,  # type: ignore[arg-type]
        **comum,
        plsda=_plsda_construido(),
        plsda_groups=("A", "B"),
        plsda_cv_folds=7,
    )

    assert set(com.computed) - set(sem.computed) == {"plsda_q2", "plsda_permutation_p"}
    assert sem.plsda_q2 is None and com.plsda_q2 == 0.24
    assert com.plsda_seed == 42 and com.plsda_n_permutations == 200
    assert com.qc_rsd_median == sem.qc_rsd_median, (
        "declarar o par de grupos mudou uma métrica que não é do PLS-DA"
    )


# ---------------------------------------------------------------------------
# A ROTA — as guardas acontecem no DISPARO
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def cliente() -> TestClient:
    return TestClient(build_app())


def _campos(**extra: str) -> dict[str, str]:
    return {
        "mode": "tiny",
        "estudo": "e",
        "correction": "none",
        "n_batches": "1",
        "rsd_variant": "plain",
        "imputation": "median",
        "sample_normalization": "none",
        "transformation": "none",
        "scaling": "none",
        "apresentacao": "exibida",
        **extra,
    }


def _arquivos() -> dict[str, tuple[str, bytes, str]]:
    return {
        "measurements": (
            "m.csv",
            (FIXTURES / "tiny_clean.csv").read_bytes(),
            "text/csv",
        ),
        "sample_list": (
            "s.csv",
            (FIXTURES / "tiny_clean_sample_list.csv").read_bytes(),
            "text/csv",
        ),
    }


@pytest.mark.parametrize(
    ("extra", "pedaco"),
    [
        ({"group_a": "NoHTLV"}, "só um foi declarado"),
        ({"group_a": "NoHTLV", "group_b": "NoHTLV"}, "não há contraste"),
        ({"group_a": "NoHTLV", "group_b": "Inexistente"}, "não existe nesta matriz"),
    ],
)
def test_par_invalido_reprova_no_disparo_e_nao_na_thread(
    cliente: TestClient, extra: dict[str, str], pedaco: str
) -> None:
    """**Tudo o que produz `4xx` acontece no disparo** — a cláusula.

    Um nome de grupo errado descoberto pela thread viraria estado `falhou` minutos depois, com a
    mensagem longe da causa e a correção já paga. Os três casos são os de `_grupos_ou_422`, e o
    reuso é o que impede duas redações para a mesma falta.
    """
    resposta = cliente.post("/execucoes", data=_campos(**extra), files=_arquivos())

    assert resposta.status_code == 422, resposta.text
    assert pedaco in resposta.json()["detail"]


def test_grupo_pequeno_demais_para_a_validacao_cruzada_reprova_no_disparo(
    cliente: TestClient,
) -> None:
    """**O ramo que o dado real NUNCA exercita, e é por isso que ele precisa de teste próprio.**

    O menor grupo dos quatro arquivos do laboratório tem 13 amostras, contra `cv_folds = 7`. A
    `tiny_clean` tem **um** por grupo, então é ela que alcança a guarda — e sem este teste o ramo
    ficaria inerte sem ninguém notar.

    A mensagem nomeia **as duas contagens**: quem vê *"o grupo X tem 1"* sem saber que o outro tem
    1 também não sabe se o problema é o par ou o desenho do estudo.
    """
    resposta = cliente.post(
        "/execucoes",
        data=_campos(group_a="NoHTLV", group_b="ATLL"),
        files=_arquivos(),
    )

    assert resposta.status_code == 422, resposta.text
    detalhe = resposta.json()["detail"]
    assert f"{PLSDA_CV_FOLDS} dobras" in detalhe
    assert "NoHTLV" in detalhe and "ATLL" in detalhe, "a mensagem não traz as duas contagens"
    assert "sem declarar grupos" in detalhe, "não diz a saída — executar sem o par continua válido"


# ---------------------------------------------------------------------------
# `grupos_disponiveis` — e o QC NÃO é um grupo a contrastar
# ---------------------------------------------------------------------------


def test_o_qc_nao_entra_na_lista_de_grupos_a_contrastar(cliente: TestClient) -> None:
    """**O QC é a régua, não um grupo** — e a primeira versão o oferecia.

    Medido em 27/08/2026 contra o executável congelado: a rota devolvia
    `["ATLL","Asymptomatic","HAMTSP","NoHTLV","QC"]`. Um QC é uma **mistura de todas as amostras**,
    injetada periodicamente para servir de régua técnica; qualquer variação nele é técnica e não
    biológica. Oferecê-lo faria a interface convidar a um PLS-DA entre *«todas as amostras
    misturadas»* e um diagnóstico — um contraste que produz número e não significa nada.

    **A exclusão é pela MÁSCARA e não pelo nome.** O rótulo vem do `Sample_List` do laboratório e o
    domínio é aberto: comparar contra a string `"QC"` casaria por acidente de escrita e falharia num
    arquivo que o escrevesse de outra forma. `qc_mask` é o que a ingestão já decidiu.

    **As duas metades**: os grupos biológicos estão lá, e o QC não. Só a segunda passaria sobre uma
    lista vazia.

    **O campo passou a ser NOME + CONTAGEM, e este teste é um dos três
    consumidores que a cláusula da irmã manda procurar.** O que ele afirmava — o QC fora, a
    ordem igual à da mensagem de recusa — continua a ser afirmado, agora sobre a projeção dos
    nomes; e a contagem ganha asserção própria em
    `test_o_inventario_traz_a_contagem_de_cada_grupo`.
    """
    resposta = cliente.post(
        "/matrizes/diagnosticar",
        data={"mode": "tiny"},
        files=_arquivos(),
    )

    assert resposta.status_code == 200, resposta.text
    inventario = resposta.json()["grupos_disponiveis"]

    assert inventario, "a lista veio vazia: a asserção de ausência abaixo passaria por vacuidade"
    grupos = [entrada["nome"] for entrada in inventario]
    assert "QC" not in grupos, f"o QC foi oferecido como grupo a contrastar: {grupos}"
    assert grupos == sorted(grupos), (
        "a ordem não é a mesma da mensagem de recusa de `_grupos_ou_422`"
    )
