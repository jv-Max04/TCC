"""`best_by_metric` e o critério — e o EMPATE, que é o que separa eleição de desempate.

**O critério que este arquivo protege não é "as métricas discordam".** É que a discordância medida
seja discordância e não artefato de ordem. A primeira versão de `best_by_metric` devolvia
`dict[str, str]`, e sobre o `pos_hilic` real ela deu *"quatro métricas, quatro especificações
distintas"* — com `n_significant_fdr` **empatada em zero nas 96 células** e o "vencedor" sendo a
primeira da tupla. O número estava certo, o nome estava certo, e a eleição não aconteceu.

Então os testes daqui plantam as duas formas de empate e afirmam que a saída as distingue.
"""

import numpy as np
import pytest

from refino.domain.matrix import RefinoMatrix
from refino.domain.metadata import SampleMetadata
from refino.multiverse.executor import run_multiverse
from refino.multiverse.grid import GridDefinition
from refino.multiverse.stability import MetricWinner, best_by_metric, metrics_disagree

pytestmark = pytest.mark.property

PAR = ("A", "B")


def _matriz(n_amostras: int = 24, n_features: int = 16) -> RefinoMatrix:
    gerador = np.random.default_rng(5)
    valores = gerador.lognormal(10.0, 0.5, size=(n_amostras, n_features))
    rotulos = ["QC" if i % 6 == 0 else ("A" if i % 2 else "B") for i in range(n_amostras)]
    alvo = np.array([r == "A" for r in rotulos])
    valores[np.ix_(alvo, np.arange(5))] *= 4.0
    eh_qc = np.array([r == "QC" for r in rotulos])
    pool = valores[~eh_qc].mean(axis=0)
    valores[eh_qc] = pool * gerador.lognormal(0.0, 0.05, size=(int(eh_qc.sum()), n_features))
    amostras = tuple(
        SampleMetadata(
            name=f"S{i:03d}",
            injection_order=i + 1,
            sample_type="qc" if rotulos[i] == "QC" else "sample",
            group=rotulos[i],
            batch="A",
        )
        for i in range(n_amostras)
    )
    return RefinoMatrix(
        values=valores,
        feature_names=tuple(f"f{j:03d}" for j in range(n_features)),
        samples=amostras,
        mode="sintetico",
    )


def _grade() -> GridDefinition:
    return GridDefinition(
        corrections=("none", "loess"),
        rsd_cutoffs=(None, 0.30),
        missing_thresholds=(0.8,),
        imputations=("one_fifth_min", "median"),
        sample_norms=("none", "sum"),
        transformations=("none", "log10"),
        scalings=("none", "auto"),
    )


@pytest.fixture(scope="module")
def vencedores() -> dict[str, MetricWinner]:
    return best_by_metric(run_multiverse(_matriz(), _grade()).cells, PAR)


# ---------------------------------------------------------------------------
# O empate é visível — dois casos que não colapsam
# ---------------------------------------------------------------------------


def test_metrica_constante_e_marcada_como_nao_discriminante_e_nao_como_vencedora() -> None:
    """**O par causal do defeito que motivou `MetricWinner`.**

    Duas execuções sobre a mesma grade: uma em que a métrica separa células e outra em que ela é
    constante. A saída tem de **distinguir** as duas — e a versão que devolvia só o `fingerprint`
    não distinguia, porque as duas devolviam um fingerprint com a mesma aparência.

    A planta é uma matriz sem QC suficiente para `qc_rsd_median_post_correction` existir e sem
    efeito nenhum, o que zera `n_significant_fdr_post_pipeline` em toda célula.
    """
    gerador = np.random.default_rng(1)
    n = 18
    valores = gerador.lognormal(10.0, 0.4, size=(n, 12))
    rotulos = ["QC" if i % 6 == 0 else ("A" if i % 2 else "B") for i in range(n)]
    eh_qc = np.array([r == "QC" for r in rotulos])
    valores[eh_qc] = valores[~eh_qc].mean(axis=0) * gerador.lognormal(
        0.0, 0.03, size=(int(eh_qc.sum()), 12)
    )
    plana = RefinoMatrix(
        values=valores,
        feature_names=tuple(f"f{j:03d}" for j in range(12)),
        samples=tuple(
            SampleMetadata(
                name=f"S{i:03d}",
                injection_order=i + 1,
                sample_type="qc" if rotulos[i] == "QC" else "sample",
                group=rotulos[i],
                batch="A",
            )
            for i in range(n)
        ),
        mode="sintetico",
    )
    celulas = run_multiverse(plana, _grade()).cells
    sem_efeito = best_by_metric(celulas, PAR)

    sig = sem_efeito["n_significant_fdr_post_pipeline"]
    assert sig.value == 0.0, "premissa da planta: sem efeito, nenhuma feature significativa"
    assert sig.n_distinct_values == 1, (
        "a métrica deu o mesmo valor em toda célula e `n_distinct_values` não é 1 — então a "
        "contagem de distintos não mede o que separa eleição de empate"
    )
    assert not sig.discriminates, "métrica constante não pode se declarar discriminante"
    assert sig.n_tied_at_best == len(celulas), (
        "com valor único, TODAS as células estão no ótimo. Um número menor significa que "
        "`tied_at_best` não é o conjunto ótimo, e a interseção de `metrics_disagree` seria falsa"
    )


def test_o_conjunto_otimo_contem_o_fingerprint_reportado(vencedores: dict) -> None:
    """`fingerprint` é um **representante** de `tied_at_best`, e tem de pertencer a ele.

    Sem isto, `metrics_disagree` — que cruza os conjuntos — e o `fingerprint` reportado poderiam
    contar histórias diferentes sobre a mesma métrica.
    """
    for nome, v in vencedores.items():
        assert v.fingerprint in v.tied_at_best, f"{nome}: o vencedor não está no próprio ótimo"
        assert v.n_tied_at_best == len(set(v.tied_at_best)), f"{nome}: ótimo com repetição"
        assert 1 <= v.n_distinct_values <= v.n_cells_scored


def test_as_cinco_metricas_declaram_o_ponto_de_medicao_no_nome(vencedores: dict) -> None:
    """Um nome sem ponto volta a ser a métrica única que a regra 5 do README proíbe.

    Portão contra a **próxima** métrica: acrescentar uma sem sufixo passaria despercebido, e a
    comparação entre pontos de medição diferentes é o defeito que o sufixo existe para impedir.
    """
    assert set(vencedores) == {
        "n_significant_fdr_post_pipeline",
        "qc_dispersion_ratio_post_pipeline",
        "pc1_injection_correlation_post_pipeline",
        "qc_tighter_than_sample_frac_post_pipeline",
        "qc_rsd_median_post_correction",
    }
    for nome in vencedores:
        assert nome.endswith(("_post_pipeline", "_post_correction")), nome


def test_a_metrica_pos_correcao_ranqueia_prefixo_e_nao_especificacao(vencedores: dict) -> None:
    """**Limite estrutural, medido e não argumentado.**

    A matriz pós-correção depende só de `(correction, n_batches)`: todas as folhas do prefixo
    compartilham a mesma. Então uma métrica medida ali tem no máximo `len(prefix_keys())` valores
    distintos, e o conjunto ótimo dela é uma **classe de equivalência** de especificações.

    O teste afirma exactamente isso, e é o que impede alguém de ler o vencedor dela como "a melhor
    especificação segundo a repetibilidade".
    """
    grade = _grade()
    v = vencedores["qc_rsd_median_post_correction"]
    assert v.n_distinct_values <= len(grade.prefix_keys()), (
        f"a métrica pós-correção deu {v.n_distinct_values} valores distintos e a grade tem "
        f"{len(grade.prefix_keys())} prefixos. Ou ela não está medindo pós-correção, ou a árvore "
        "não está compartilhando a matriz corrigida entre as folhas do prefixo"
    )
    assert v.n_tied_at_best == grade.size() // len(grade.prefix_keys())


def test_qc_tighter_than_sample_frac_e_invariante_ao_escalonamento() -> None:
    """A propriedade que a métrica existe para ter, exercitada — não declarada.

    É uma **contagem** de comparações dentro de cada feature, então multiplicar a coluna inteira por
    qualquer fator positivo não pode mudá-la. É essa invariância que a torna repetibilidade em
    qualquer ponto da grade, onde `sd/média` deixa de ser a mesma coisa (inversão de sinal do
    `vast`).
    """
    from refino.multiverse.stability import _qc_tighter_frac

    matriz = _matriz()
    fatores = np.random.default_rng(2).lognormal(0.0, 1.0, size=matriz.n_features)
    escalada = RefinoMatrix(
        values=np.asarray(matriz.values, dtype=np.float64) * fatores,
        feature_names=matriz.feature_names,
        samples=matriz.samples,
        mode=matriz.mode,
    )
    assert _qc_tighter_frac(matriz) == _qc_tighter_frac(escalada)


# ---------------------------------------------------------------------------
# `metrics_disagree`
# ---------------------------------------------------------------------------


def _vencedor(nome: str, otimo: tuple[str, ...], distintos: int = 5) -> MetricWinner:
    return MetricWinner(
        metric=nome,
        fingerprint=otimo[0],
        value=1.0,
        n_cells_scored=10,
        n_distinct_values=distintos,
        tied_at_best=otimo,
    )


def test_disagree_e_falso_quando_os_conjuntos_otimos_se_cruzam() -> None:
    """**A distinção que o critério precisa, e a que um teste de fingerprint não faria.**

    Dois representantes diferentes com conjuntos ótimos que se cruzam: existe uma especificação que
    satisfaz as duas métricas, logo elas **não** discordam. Ler os fingerprints diferentes como
    discordância seria ler o desempate como medição.
    """
    assert not metrics_disagree(
        {
            "a": _vencedor("a", ("x", "y")),
            "b": _vencedor("b", ("y", "z")),
        }
    ), "os ótimos compartilham `y`, então existe especificação que agrada as duas"

    assert metrics_disagree(
        {
            "a": _vencedor("a", ("x", "w")),
            "b": _vencedor("b", ("y", "z")),
        }
    ), "os ótimos são disjuntos: nenhuma especificação satisfaz as duas"


def test_metrica_que_nao_discrimina_nao_vota_e_uma_votante_so_nao_e_discordancia() -> None:
    """As duas formas de "não há discordância a medir", e elas são diferentes.

    Métrica constante tem conjunto ótimo igual à grade inteira: incluí-la faria **toda** interseção
    ser não-vazia e a resposta seria sempre "concordam", pelo motivo errado. E uma votante só não
    tem com quem discordar — o desfecho é `False` por falta de par, não por acordo.
    """
    assert not metrics_disagree(
        {
            "constante": _vencedor("constante", ("x", "y", "z"), distintos=1),
            "real": _vencedor("real", ("x",)),
        }
    )
    assert not metrics_disagree({"unica": _vencedor("unica", ("x",))})
    assert not metrics_disagree({})


def test_celulas_vazias_devolvem_dicionario_vazio() -> None:
    assert best_by_metric((), PAR) == {}
