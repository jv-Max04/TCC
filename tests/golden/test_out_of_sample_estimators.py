"""Os dois estimadores fora da amostra, contra o in-sample.

**O portão é UM e o resto é resultado.** O critério espera
`in_sample < leave_one_qc_out` sobre a **mediana entre features**, e é o único
portão aqui: modelo avaliado no ponto retido não vai, **em tendência**, melhor que
avaliado no ponto de treino. A relação entre `leave_one_qc_out` e `r_cv` é
**resultado** — esquemas diferentes, partição completa de 6/1 contra cinco sorteios
de 5/2, e nenhuma desigualdade entre eles decorre do desenho.

**A inversão por feature é contada e reportada, não diagnosticada.** Com sete
pontos, o vetor fora da amostra de uma feature isolada pode sair mais homogêneo que
o in-sample **sem vazamento nenhum**. A v1.2 dizia "igual ou menor é vazamento", o
que transforma ruído em diagnóstico — falso positivo à espera.

**As TRÊS medianas são sobre o MESMO conjunto de features, e isso é o conteúdo do
teste e não detalhe dele.** Com a amostragem os dois estimadores fora da
amostra medem 500 features; se o in-sample fosse calculado sobre as 6.035, as
medianas seriam de **populações diferentes** e a comparação responderia outra
pergunta. É o rótulo certo sobre a quantidade errada, na forma mais fácil de cometer aqui: os três
números existem, os três se chamam "mediana de RSD", e dois medem outra coisa.

Por isso o conjunto amostrado é resolvido **uma vez** e passado aos três — incluindo
o in-sample, que roda como execução parcial sobre exatamente as mesmas 500.
"""

import json

import numpy as np
import pytest

from refino.correction.serrf import SerrfCorrector
from refino.io.readers import load_matrix
from refino.metrics.crossval import (
    SAMPLED_FEATURE_COUNT,
    leave_one_qc_out_rsd,
    qc_rsd_r_cv,
    sampled_feature_names,
)
from refino.metrics.rsd import rsd
from tests.conftest import REPO, measurements_path, sample_list_path

pytestmark = [pytest.mark.golden, pytest.mark.slow]

MODES = ("pos_hilic", "pos_c18", "neg_hilic", "neg_c18")
SEED = 42
VARIANT = "outlier_removed"


def _medianas(mode: str) -> dict[str, object]:
    """Os três estimadores sobre o mesmo conjunto amostrado, num modo real."""
    matrix = load_matrix(measurements_path(mode), sample_list_path(mode), mode)
    corrector = SerrfCorrector(n_jobs=-1)

    # Resolvido UMA vez e compartilhado pelos três. `sampled_feature_names` é
    # determinística em `(matrix, n, seed)`, então chamá-la de novo daria o mesmo
    # conjunto — passar explicitamente é o que impede que uma mudança de assinatura
    # separe silenciosamente as populações.
    amostradas = sampled_feature_names(matrix, SAMPLED_FEATURE_COUNT, SEED)
    alvo = frozenset(amostradas) if amostradas is not None else None

    corrigida = corrector.fit_transform(matrix, seed=SEED, target_features=alvo)
    in_sample = rsd(corrigida.matrix.values[corrigida.matrix.qc_mask], axis=0, variant=VARIANT)

    loo = leave_one_qc_out_rsd(
        matrix, corrector, variant=VARIANT, seed=SEED, n_sampled_features=SAMPLED_FEATURE_COUNT
    )
    r_cv = qc_rsd_r_cv(
        matrix, corrector, variant=VARIANT, seed=SEED, n_sampled_features=SAMPLED_FEATURE_COUNT
    )

    # Os dois estimadores devolvem vetor de comprimento `n_features` com `nan` nas
    # não amostradas; o in-sample devolve vetor curto, na ordem de `amostradas`.
    posicao = {name: i for i, name in enumerate(matrix.feature_names)}
    indices = np.array([posicao[n] for n in (amostradas or matrix.feature_names)])
    loo_amostrado = loo[indices]
    r_cv_amostrado = r_cv[indices]

    # A população comum: features com RSD definido nos TRÊS. Comparar medianas sobre
    # conjuntos de definidos diferentes reintroduziria o defeito que o compartilhamento
    # do conjunto amostrado existe para evitar.
    definidas = np.isfinite(in_sample) & np.isfinite(loo_amostrado) & np.isfinite(r_cv_amostrado)
    n_qc = int(matrix.qc_mask.sum())
    return {
        "n_pedidas": len(amostradas or matrix.feature_names),
        "n_definidas_nos_tres": int(definidas.sum()),
        # **`n` ao lado de cada número, e é ele que declara a incomparabilidade.**
        # O `r_cv` mede sobre DOIS pontos e os outros dois sobre SETE;
        # publicar as medianas sem `n` convida à conclusão de que o `r_cv` corrige
        # melhor, que contradiz a razão de ele existir.
        "in_sample": {"mediana": float(np.median(in_sample[definidas])), "n_pontos": n_qc},
        "leave_one_qc_out": {
            "mediana": float(np.median(loo_amostrado[definidas])),
            "n_pontos": n_qc,
        },
        "r_cv": {
            "mediana": float(np.median(r_cv_amostrado[definidas])),
            "n_pontos": n_qc - round(n_qc * 0.7),
            "conclusivo_contra_leave_one_qc_out": False,
            "motivo": (
                "estruturas de amostragem diferentes sobre a mesma população: dois "
                "pontos retidos por sorteio contra sete, um de cada vez. Nenhuma "
                "correção de viés as torna comparáveis, e corrigir daria aparência de "
                "rigor a uma comparação sem significado."
            ),
        },
        "inversoes_por_feature": int((loo_amostrado[definidas] <= in_sample[definidas]).sum()),
    }


@pytest.fixture(scope="module")
def medidas() -> dict[str, dict[str, object]]:
    """Uma medição por modo, reusada pelas asserções — são 13 ajustes por modo."""
    return {mode: _medianas(mode) for mode in MODES}


# ---------------------------------------------------------------------------
# O portão
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", MODES)
def test_ca_3_11_in_sample_e_menor_que_leave_one_qc_out(
    mode: str, medidas: dict[str, dict[str, object]]
) -> None:
    """**O portão, sobre a MEDIANA e não por feature.**

    Desigualdade estrita: igualdade exata entre as duas medianas sobre milhares de
    features seria coincidência suspeita, não sucesso.
    """
    m = medidas[mode]
    ins, loo = m["in_sample"], m["leave_one_qc_out"]
    assert ins["n_pontos"] == loo["n_pontos"], (
        "o portão compara medianas com `n` diferente, e aí ele não decorre do desenho"
    )
    assert ins["mediana"] < loo["mediana"], (
        f"{mode}: mediana in-sample {ins['mediana']:.6g} não é menor que a de "
        f"leave_one_qc_out {loo['mediana']:.6g}. O modelo avaliado no ponto "
        "retido foi melhor que no ponto de treino, o que não decorre do desenho."
    )


@pytest.mark.parametrize("mode", MODES)
def test_a_populacao_dos_tres_e_a_mesma_e_nao_e_vazia(
    mode: str, medidas: dict[str, dict[str, object]]
) -> None:
    """A guarda de vacuidade do portão acima, e ela é a asserção com dente.

    Sem ela, um modo em que a interseção dos definidos fosse vazia passaria o portão
    por `median([])` — que é `nan`, e `nan < nan` é `False`, então reprovaria. Mas
    um modo com **duas** features definidas passaria ou reprovaria por ruído, e o
    número seria indistinguível de medição.
    """
    m = medidas[mode]
    assert m["n_definidas_nos_tres"] >= 100, (
        f"{mode}: só {m['n_definidas_nos_tres']} features definidas nos três "
        f"estimadores, de {m['n_pedidas']} pedidas. Mediana sobre essa população não "
        "sustenta o portão."
    )


# ---------------------------------------------------------------------------
# O que é RESULTADO e não portão
# ---------------------------------------------------------------------------


def test_os_tres_numeros_sao_persistidos_e_nao_so_calculados(
    medidas: dict[str, dict[str, object]],
) -> None:
    """**Reportar os três é o resultado publicável, e "reportar" tem de ser durável.**

    A primeira versão deste teste computava os três números e os **descartava**: as
    asserções passavam, e `pytest` engole `stdout` e mensagem de asserção de teste
    que passa. O relatório só existiria quando fosse inútil.
    Dezenove minutos de medição para produzir nada
    legível.

    Por isso o teste escreve em `out/`, e a asserção é sobre o **arquivo existir com o
    conteúdo**, não sobre os números terem sido calculados.

    A relação entre `leave_one_qc_out` e `r_cv` **não** é portão: esquemas
    diferentes, nenhuma desigualdade derivável. A predição é que o `r_cv`
    seja pior e mais ruidoso, por ter menos treino e medir sobre dois pontos — e
    predição verificável não é critério.
    """
    destino = REPO / "out" / "ca_3_11_estimadores.json"
    destino.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "criterio": "[CA-3.11]",
        "variante": VARIANT,
        "semente": SEED,
        "features_amostradas": SAMPLED_FEATURE_COUNT,
        "por_modo": {mode: medidas[mode] for mode in MODES},
    }
    destino.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    lido = json.loads(destino.read_text(encoding="utf-8"))
    assert set(lido["por_modo"]) == set(MODES), "o relatório persistido perdeu um modo"
    for mode in MODES:
        entrada = lido["por_modo"][mode]
        assert np.isfinite(entrada["r_cv"]["mediana"]), f"{mode}: r_cv não finito"
        assert np.isfinite(entrada["leave_one_qc_out"]["mediana"])
        assert np.isfinite(entrada["in_sample"]["mediana"])
        # `n` ao lado de cada número, e a não conclusividade declarada NO ARTEFATO —
        # não só na prosa do critério. Quem ler o JSON sem abrir a especificação tem de
        # encontrar o motivo ali.
        assert entrada["r_cv"]["n_pontos"] == 2, "o r_cv deixou de medir sobre dois pontos"
        assert entrada["in_sample"]["n_pontos"] == entrada["leave_one_qc_out"]["n_pontos"] == 7
        assert entrada["r_cv"]["conclusivo_contra_leave_one_qc_out"] is False
        assert "sem significado" in entrada["r_cv"]["motivo"]


@pytest.mark.parametrize("mode", MODES)
def test_a_inversao_por_feature_e_contada_e_nao_diagnosticada(
    mode: str, medidas: dict[str, dict[str, object]]
) -> None:
    """**Contada e reportada, nunca tratada como vazamento.**

    Com sete pontos, uma feature isolada pode sair mais homogênea fora da amostra
    por sorte de sorteio. A asserção é que a contagem **existe e é sã** — entre zero
    e a população —, não que seja zero: exigir zero é a v1.2, que transforma ruído
    em diagnóstico.
    """
    m = medidas[mode]
    assert 0 <= int(m["inversoes_por_feature"]) <= int(m["n_definidas_nos_tres"])
