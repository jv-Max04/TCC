"""As políticas e a abstenção.

**O que estes testes protegem é a metade (b) do critério de abstenção**, e ela existe porque a
metade (a)
sozinha é satisfeita por confiança constante 0,49. Então todo teste de abstenção aqui é
**par**: abstém-se onde deve **e não se abstém** onde não deve.

**E o eixo é a MARGEM e não o vencedor, por medição.** `leave_one_qc_out_rsd` elege corretor em 16
de 16 matrizes — a régua nunca responde "não corrija". A política não pode aprender a
abster-se pelo eixo do vencedor porque o vencedor existe sempre; a margem é o único sinal.
"""

import json
import math
from pathlib import Path

import pytest

from refino.decision.policy import (
    ABSTENTION_MARGIN,
    AlwaysSERRFPolicy,
    BestFixedCorrectorPolicy,
    DecisionPolicy,
    Recommendation,
    confidence_from_margin,
    elect_best_fixed,
)
from refino.domain.diagnosis import DefectProfile

pytestmark = pytest.mark.property

# Margens MEDIDAS nos quatro arquivos reais com a floresta ANTIGA. Sao OBSERVACAO sobre esses
# quatro e nao invariante do metodo — um quinto modo tem a sua propria margem medida.
# **Ficam como historia da derivacao do limiar**: a floresta padrao e a do R.
MARGENS_REAIS_FLORESTA_ANTIGA = {
    "neg_c18": 0.0022,
    "pos_hilic": 0.0615,
    "neg_hilic": 0.1150,
    "pos_c18": 0.1196,
}

# As margens dos quatro arquivos reais com a floresta do R, semente 42 — a R1 do estado novo da SF,
# que a rodada PP reproduziu bit a bit. Nenhuma fica abaixo do limiar: os quatro sao claros.
# `test_d581_as_margens_pinadas_sao_as_das_fontes` confere-as contra o arquivo.
MARGENS_REAIS = {
    "neg_c18": 0.023275659189884074,
    "neg_hilic": 0.04389872058502665,
    "pos_hilic": 0.3056773208529423,
    "pos_c18": 0.6832712000571912,
}

# As margens das duas matrizes do corpus de treino que ficam abaixo do limiar — onde a metade (a)
# passou a ser medida. Conferidas contra o corpus pelo mesmo teste.
MARGENS_AMBIGUAS_DO_CORPUS = {"neg_hilic-2": 0.008683, "neg_hilic-5": 0.005333}

# O intervalo de 95% do limiar pelo ruído da semente, conferido contra o resultado da PP.
INTERVALO_DO_RUIDO = (0.010114883213051196, 0.014703110877670627)

REPO = Path(__file__).resolve().parents[2]
R1_NOVA = REPO / "data" / "fixtures" / "medicoes" / "sf" / "r1_oraculo_nos_reais.json"
CORPUS = REPO / "data" / "fixtures" / "corpus_fase9_7qcs_ranger.jsonl"
PP = REPO / "data" / "fixtures" / "medicoes" / "pp" / "pp_resultado.json"


def _politica(margem: float | None, **kw: object) -> AlwaysSERRFPolicy:
    """`AlwaysSERRFPolicy` com a margem INJETADA por estimador, e não lida do perfil.

    A diferença é um defeito: ler `oracle_margin` de `raw_features` significaria
    que o oráculo já rodou. O estimador vem de fora e é o mesmo para todas as políticas comparadas.
    """
    est = None if margem is None else (lambda _p, m=margem: m)
    return AlwaysSERRFPolicy(margin_estimator=est, **kw)  # type: ignore[arg-type]


def _perfil(margem: float | None = None, **troca: float) -> DefectProfile:
    campos: dict[str, object] = {
        "defect_type": "gradual_drift",
        "confidence": 0.8,
        "suggested_batches": (1,),
        "changepoints": (),
        "qc_rsd_median": 0.2415,
        "qc_rsd_p75": 0.44,
        "zero_fraction": 0.034,
        "missing_fraction": 0.0,
        "pc1_injection_correlation": 0.52,
        "qc_dispersion_ratio": 0.7957,
        "n_features": 2572,
        "n_samples": 68,
        "n_qcs": 7,
        "raw_features": {} if margem is None else {"oracle_margin": margem},
    }
    campos.update(troca)
    return DefectProfile(**campos)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# A abstenção — as duas direções, sobre as margens medidas
# ---------------------------------------------------------------------------


def test_ca_9_5_o_par_abstem_no_modo_ambiguo_e_nao_abstem_nos_claros() -> None:
    """**O par causal da abstenção, sobre as margens medidas.**

    A metade (a) sozinha é satisfeita por confiança constante 0,49; a metade (b) é o que a torna
    informativa. As duas juntas, sobre as margens medidas, é o critério inteiro.

    **O ambíguo saiu dos arquivos reais e foi para o corpus.** Com a floresta
    do R nenhum modo real é ambíguo pelo ruído da semente, e a metade (a) passou a ser medida
    nas duas matrizes do corpus com margem abaixo do limiar; a (b), nos quatro reais, agora os
    quatro claros. Como antes, a margem é injetada: o teste verifica o mapa margem → abstenção, e
    não que a política a consiga prever — isso é
    `test_d581_o_ca_9_5_a_reprova_nas_ambiguas_do_corpus`.
    """
    for nome, margem in MARGENS_AMBIGUAS_DO_CORPUS.items():
        ambiguo = _politica(margem).recommend(_perfil())
        assert ambiguo.abstains, (
            f"na `{nome}` do corpus a margem é {margem:.4f} e a política tinha de abster-se. "
            f"Deu confiança {ambiguo.confidence:.4f}"
        )

    for modo in ("neg_c18", "pos_hilic", "neg_hilic", "pos_c18"):
        clara = _politica(MARGENS_REAIS[modo]).recommend(_perfil())
        assert not clara.abstains, (
            f"em `{modo}` a margem é {MARGENS_REAIS[modo]:.4f}, vencedor claro, e a política se "
            f"absteve com confiança {clara.confidence:.4f}. Abstenção onde há resposta é o defeito "
            "que a metade (b) do critério existe para reprovar"
        )


def test_o_limiar_derivado_cai_no_vao_e_nao_numa_encosta() -> None:
    """**O corte é derivado do vão, não movido até passar.**

    Entre a margem do modo ambíguo e a menor entre os claros há uma ordem de magnitude. Qualquer
    limiar dentro dela produz a **mesma** partição sobre os quatro modos — e é isso que torna a
    escolha exata do valor irrelevante, que é o oposto de calibrar até o resultado sair.

    **É a derivação com a floresta ANTIGA, e fica como história.** Com a
    floresta do R o vão sumiu, e a derivação passou a ser o ruído da semente — ver
    `test_d581_o_limiar_cai_no_intervalo_do_ruido_e_a_particao_nao_depende_dele`.
    """
    margens = MARGENS_REAIS_FLORESTA_ANTIGA
    piso, teto = margens["neg_c18"], margens["pos_hilic"]
    assert teto / piso > 10.0, f"o vão é de {teto / piso:.1f}x, e o argumento pressupõe uma ordem"
    assert piso < ABSTENTION_MARGIN < teto, f"{ABSTENTION_MARGIN} não está no vão"

    for limiar in (0.005, 0.012, 0.03, 0.05):
        abstem = {
            m
            for m in margens
            if _politica(margens[m], threshold=limiar).recommend(_perfil()).abstains
        }
        assert abstem == {"neg_c18"}, (
            f"com limiar {limiar} a partição foi {sorted(abstem)}, e devia ser só `neg_c18` para "
            "qualquer valor dentro do vão"
        )


def test_confianca_constante_reprovaria_o_par() -> None:
    """**A planta que demonstra que o par não é decorativo.**

    Uma política que devolve 0,49 sempre satisfaz a metade (a) e reprova a (b). Se o teste do par
    passasse com ela, ele não estaria lendo a segunda direção.
    """

    class ConfiancaConstante:
        name = "covarde"

        def recommend(self, profile: DefectProfile) -> Recommendation:
            return Recommendation("serrf", 0.49, "sempre 0,49", {})

    covarde = ConfiancaConstante()
    ambigua = MARGENS_AMBIGUAS_DO_CORPUS["neg_hilic-5"]
    assert covarde.recommend(_perfil(ambigua)).abstains, "satisfaz a metade (a)"
    assert covarde.recommend(_perfil(MARGENS_REAIS["pos_c18"])).abstains, "e reprova a metade (b)"


def test_d581_o_limiar_cai_no_intervalo_do_ruido_e_a_particao_nao_depende_dele() -> None:
    """**A derivação que substitui a do vão: o ruído da semente.**

    A rodada PP mediu quanto a margem se move só com a troca de semente, nos quatro reais com a
    floresta do R, e o limiar — duas vezes o maior desvio entre os modos — teve intervalo de 95% de
    0,0101 a 0,0147. O 0,012 cai dentro dele; e **qualquer limiar dentro dele dá a mesma
    partição** sobre as seis margens que o critério lê — as duas ambíguas do corpus abaixo, e os
    quatro reais acima —, que é o argumento
    do vão com outra régua: o valor exacto não decide nada.
    """
    baixo, alto = INTERVALO_DO_RUIDO
    assert baixo < ABSTENTION_MARGIN < alto, f"{ABSTENTION_MARGIN} saiu do intervalo do ruído"
    assert max(MARGENS_AMBIGUAS_DO_CORPUS.values()) < baixo, "uma ambígua entrou no intervalo"
    assert min(MARGENS_REAIS.values()) > alto, "uma margem real entrou no intervalo"

    for limiar in (baixo * 1.0001, ABSTENTION_MARGIN, alto * 0.9999):
        abstem = {
            nome
            for nome, margem in (MARGENS_AMBIGUAS_DO_CORPUS | MARGENS_REAIS).items()
            if _politica(margem, threshold=limiar).recommend(_perfil()).abstains
        }
        assert abstem == set(MARGENS_AMBIGUAS_DO_CORPUS), (
            f"com limiar {limiar} a partição foi {sorted(abstem)}, e devia ser só as ambíguas do "
            "corpus para qualquer valor dentro do intervalo do ruído"
        )


def test_d581_as_margens_pinadas_sao_as_das_fontes() -> None:
    """As três constantes acima, contra os arquivos de onde vieram — o outro caminho como oráculo.

    **A ambígua do corpus é DERIVADA aqui do corpus inteiro**, e não só conferida pelo nome: se um
    corpus novo trouxer uma terceira matriz abaixo do limiar, o conjunto difere e o teste reprova.
    """
    r1 = json.loads(R1_NOVA.read_text(encoding="utf-8"))["modos"]
    assert {m: r1[m]["margem"] for m in MARGENS_REAIS} == MARGENS_REAIS

    linhas = [json.loads(x) for x in CORPUS.read_text(encoding="utf-8").splitlines() if x.strip()]
    assert len(linhas) == 40, f"âncora: o corpus tem {len(linhas)} linhas"
    abaixo = {
        f"{r['modo']}-{r['indice']}": r["margem"] for r in linhas if r["margem"] < ABSTENTION_MARGIN
    }
    assert abaixo == MARGENS_AMBIGUAS_DO_CORPUS, abaixo

    pp = json.loads(PP.read_text(encoding="utf-8"))
    assert tuple(pp["pp2"]["intervalo_95_de_tau"]) == INTERVALO_DO_RUIDO


# ---------------------------------------------------------------------------
# A função de confiança
# ---------------------------------------------------------------------------


def test_a_confianca_e_monotonica_e_satura() -> None:
    anteriores = [confidence_from_margin(m) for m in (0.0, 0.001, 0.006, 0.012, 0.05, 0.5, 5.0)]
    assert anteriores == sorted(anteriores), f"não é monotônica: {anteriores}"
    assert all(0.0 <= c <= 1.0 for c in anteriores)
    assert confidence_from_margin(1e9) < 1.0, "satura sem alcançar 1"


def test_o_limiar_de_margem_e_o_de_confianca_nomeiam_o_mesmo_ponto() -> None:
    """Em `margin == threshold` a confiança é exactamente 0,5, o corte da abstenção.

    Sem isso seriam dois limiares que alguém teria de manter em acordo, e o desacordo entre eles
    seria invisível — a política se absteria com margem acima do corte, ou o contrário.
    """
    assert confidence_from_margin(ABSTENTION_MARGIN) == pytest.approx(0.5)
    assert confidence_from_margin(ABSTENTION_MARGIN * 1.001) > 0.5
    assert confidence_from_margin(ABSTENTION_MARGIN * 0.999) < 0.5


def test_margem_ausente_da_confianca_zero_e_assessable_falso() -> None:
    """A ação é a mesma e a contagem não, com a distinção preservada em `assessable`."""
    sem = _politica(None).recommend(_perfil())
    assert sem.abstains and sem.confidence == 0.0
    assert not sem.assessable, "sem margem é `não avaliável`, não `avaliado e deu zero`"
    assert math.isnan(sem.evidence["oracle_margin"])

    zero = _politica(0.0).recommend(_perfil())
    assert zero.abstains and zero.confidence == 0.0
    assert zero.assessable, "margem medida e nula É avaliável — foi medida"


def test_margem_nao_finita_nao_propaga_nan_para_a_confianca() -> None:
    for ruim in (float("nan"), float("inf"), float("-inf"), -0.5):
        c = confidence_from_margin(ruim)
        assert c == 0.0, f"margem {ruim} deu confiança {c}"


# ---------------------------------------------------------------------------
# O protocolo das políticas e a separação do oráculo
# ---------------------------------------------------------------------------


def test_ca_9_6_o_rationale_cita_duas_caracteristicas_numericas_concretas() -> None:
    """E as duas componentes: o corretor escolhido e a margem que sustenta a confiança."""
    for margem in (MARGENS_REAIS["neg_c18"], MARGENS_REAIS["pos_c18"], None):
        est = None if margem is None else (lambda _p, m=margem: m)
        for politica in (
            AlwaysSERRFPolicy(margin_estimator=est),
            BestFixedCorrectorPolicy(corrector="loess", margin_estimator=est),
        ):
            r = politica.recommend(_perfil())
            assert r.rationale, "rationale vazio"
            assert f"{r.correction}" in r.rationale, "não nomeia o corretor recomendado"
            assert "RSD mediano de QC é 0.2415" in r.rationale
            assert "correlação de PC1 com a ordem de injeção é 0.5200" in r.rationale
            assert len(r.evidence) >= 4


def test_ca_9_7_a_politica_e_deterministica() -> None:
    politica = _politica(0.03)
    a = politica.recommend(_perfil())
    b = politica.recommend(_perfil())
    assert (a.correction, a.confidence, a.rationale) == (b.correction, b.confidence, b.rationale)


def test_as_politicas_satisfazem_o_protocolo_e_o_oraculo_nao() -> None:
    """A separação de tipo é o mecanismo que impede o oráculo de passar por política."""
    from refino.decision.labeling import LabelingOracle

    assert isinstance(AlwaysSERRFPolicy(), DecisionPolicy)
    assert isinstance(BestFixedCorrectorPolicy(corrector="none"), DecisionPolicy)
    assert not isinstance(LabelingOracle(), DecisionPolicy)


# ---------------------------------------------------------------------------
# `elect_best_fixed`
# ---------------------------------------------------------------------------


def test_elege_o_mais_frequente_e_desempata_pela_ordem_canonica() -> None:
    assert elect_best_fixed(["serrf", "serrf", "loess"]) == "serrf"
    assert elect_best_fixed(["loess", "loess", "serrf"]) == "loess"
    # Empate: a ordem canonica decide, e nao a ordem de chegada.
    assert elect_best_fixed(["serrf", "loess"]) == elect_best_fixed(["loess", "serrf"])


def test_a_eleicao_nao_depende_da_ordem_do_treino() -> None:
    """Duas execuções com o treino embaralhado elegem o mesmo — senão a linha de base varia."""
    base = ["serrf"] * 7 + ["loess"] * 5
    assert elect_best_fixed(base) == elect_best_fixed(list(reversed(base))) == "serrf"


def test_treino_vazio_levanta_em_vez_de_eleger_por_padrao() -> None:
    with pytest.raises(ValueError, match="não há rótulos"):
        elect_best_fixed([])
