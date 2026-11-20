"""A VERDADE FUNDAMENTAL do gerador — o que nunca foi verificado.

**Um `grep` por `true_differential` e `ground_truth` em `tests/` não retornava nada.** O gerador
existe para plantar verdade conhecida, a etapa fechou com os critérios verdes, e nenhum teste
tocava a verdade plantada: toda a verificação era sobre o **perfil estatístico** da matriz — RSD de
QC, fração de zeros, correlação mediana. É o mecanismo são sobre o alvo errado, e a assimetria está
inteira aqui: os mecanismos estavam corretos, exercitados nas duas direções, e
afirmavam a coisa errada.

Dois defeitos apareceram no primeiro teste escrito, e nenhum dos dois levantava:

| Defeito | Como se manifestava |
|---|---|
| `true_differential` dava `f00116`, a matriz usa `2.32_232.1092m/z` | interseção vazia |
| a verdade plantada não é recuperável ao FDR | zero das 50 sobrevivem, menor `q` = 0,0750 |

O primeiro está corrigido. O segundo **não**, e o teste que o mede é `xfail(strict=True)` — pelo
precedente do critério e do critério: o número fica medido e visível, e o dia em que a
calibração mudar o `strict` reprova e obriga a reler isto.
"""

import numpy as np
import pytest

from refino.domain.synthetic import SyntheticSpec
from refino.statistics.univariate import univariate_test
from refino.synthetic.generator import generate

pytestmark = [pytest.mark.synthetic, pytest.mark.slow]

ESPECIFICACAO = SyntheticSpec(
    imitated_mode="pos_hilic",
    drift_to_noise_ratio=1.0,
    n_features=2000,
    differential_fraction=0.025,
    effect_size=3.0,
    seed=42,
)


@pytest.fixture(scope="module")
def sintetico():  # type: ignore[no-untyped-def]
    """A fonte real, pelos resolvedores do `conftest` — nunca por caminho montado à mão.

    Eles fazem `skip` com mensagem quando o arquivo falta, que é a regra 1 do README: dado de
    referência ausente vira `skip`, nunca matriz inventada.
    """
    from refino.io.readers import load_matrix
    from tests.conftest import measurements_path, sample_list_path

    fonte = load_matrix(
        measurements_path("pos_hilic"), sample_list_path("pos_hilic"), mode="pos_hilic"
    )
    return generate(ESPECIFICACAO, fonte)


def test_true_differential_nomeia_colunas_que_existem_na_matriz(sintetico) -> None:  # type: ignore[no-untyped-def]
    """**O defeito que foi fechado sem ver, e o teste é uma linha.**

    `true_differential` são nomes de feature. Se não forem os nomes **da matriz que os acompanha**,
    a verdade fundamental é inutilizável por nome — e o modo de falha é silencioso: um consumidor
    faz `set(true_differential) & set(feature_names)`, recebe vazio, e conclui que nada foi
    recuperado em vez de concluir que nada foi comparado.
    """
    nomes = set(sintetico.matrix.feature_names)
    plantadas = set(sintetico.true_differential)
    esperadas = max(1, round(ESPECIFICACAO.differential_fraction * ESPECIFICACAO.n_features))

    assert len(plantadas) == esperadas, (
        f"{len(plantadas)} features plantadas e a especificação pede {esperadas} — "
        f"`differential_fraction={ESPECIFICACAO.differential_fraction}` sobre "
        f"{ESPECIFICACAO.n_features} features"
    )
    ausentes = plantadas - nomes
    assert not ausentes, (
        f"{len(ausentes)} de {len(plantadas)} features da verdade fundamental não existem na "
        f"matriz. Exemplo: {sorted(ausentes)[:2]}. Nomes reais da matriz: "
        f"{list(sintetico.matrix.feature_names)[:2]}"
    )


def test_o_efeito_plantado_esta_nas_colunas_plantadas_e_nao_nas_outras(sintetico) -> None:  # type: ignore[no-untyped-def]
    """O **par causal** da verdade fundamental: plantadas deslocadas, não plantadas não.

    Sem a segunda metade o teste passaria com o efeito aplicado à matriz inteira, que é o defeito
    de alinhamento que este projeto encontra com regularidade. A razão é sobre `G1` contra os
    demais grupos biológicos, porque é `G1` que `_efeito_biologico` desloca.
    """
    matriz = sintetico.matrix
    nomes = list(matriz.feature_names)
    valores = np.asarray(matriz.values, dtype=np.float64)
    grupos = np.array([a.group for a in matriz.samples])
    em_g1 = grupos == "G1"
    outros = (grupos != "G1") & (grupos != "QC")

    plantadas = {nomes.index(n) for n in sintetico.true_differential}
    nao_plantadas = [j for j in range(len(nomes)) if j not in plantadas][:200]

    def _razao(indices) -> float:  # type: ignore[no-untyped-def]
        with np.errstate(invalid="ignore", divide="ignore"):
            razoes = [
                np.nanmedian(valores[em_g1, j]) / max(np.nanmedian(valores[outros, j]), 1e-30)
                for j in indices
            ]
        return float(np.nanmedian(razoes))

    razao_plantada = _razao(sorted(plantadas))
    razao_neutra = _razao(nao_plantadas)

    assert razao_plantada > 10.0 * razao_neutra, (
        f"razão G1/outros nas plantadas = {razao_plantada:.3g} e nas não plantadas = "
        f"{razao_neutra:.3g}. O efeito não está concentrado onde a verdade diz que está"
    )
    assert 0.5 < razao_neutra < 2.0, (
        f"as features NÃO plantadas têm razão {razao_neutra:.3g} entre G1 e os outros grupos. O "
        "efeito vazou para fora das colunas plantadas — alinhamento entre `indices` e `colunas`"
    )


@pytest.mark.xfail(
    strict=True,
    reason=(
        "A verdade plantada NÃO é recuperável ao FDR de 0,05. Medido no pos_hilic com "
        "effect_size=3,0: das 50 plantadas, 29 têm p<0,05, o menor p é 1,138e-04 com 206 de 225 "
        "pares separados, e BH sobre 2.000 testes exige p<2,5e-05 na primeira. Duas das 50 ficam "
        "todas zero nos dois grupos, apagadas por `_introduz_ausencia`. "
        "O PODER FOI CALCULADO e EXCLUÍDO como causa: o p mínimo do Mann-Whitney bilateral com "
        "n=15 por grupo é 1,289e-08 contra o limiar BH de 2,5e-05 — bastariam 10 por grupo. "
        "O que BH sobre 2.000 testes exige são 220 de 225 pares separados (97,8%), e a melhor "
        "plantada alcançou 206 (91,6%). O vão é de MAGNITUDE DO EFEITO e não de tamanho de "
        "amostra. DESTRAVA: effect_size maior, ou n_features menor — com 500 o limiar sobe para "
        "1e-4 e o p observado fica a um fio, e com 100 passa. NÃO destrava mexer em n_samples. "
        "Reabre os [CA-7.*] e é decisão de fase, não deste teste."
    ),
)
def test_a_verdade_plantada_e_recuperavel_ao_fdr(sintetico) -> None:  # type: ignore[no-untyped-def]
    """O que o gerador precisa entregar para servir de oráculo às etapas seguintes.

    **Este é o teste que faltava, e ele é o que dá sentido ao módulo inteiro.** Um gerador de
    verdade conhecida cuja verdade não é detectável não pode medir recuperação — e o multiverso
    queria usá-lo exactamente para responder *"quais escolhas do multiverso preservam sinal e quais
    o destroem"*, que é uma pergunta sem resposta quando célula nenhuma acha sinal.
    """
    resultado = univariate_test(sintetico.matrix, ("G1", "G2"))
    q = np.asarray(resultado.qvalue_fdr, dtype=np.float64)
    posicao = {nome: i for i, nome in enumerate(resultado.feature)}
    plantadas = [posicao[n] for n in sintetico.true_differential if n in posicao]

    recuperadas = int(np.nansum(q[plantadas] < 0.05))
    assert recuperadas > 0, (
        f"nenhuma das {len(plantadas)} features plantadas sobreviveu ao FDR. Menor q entre elas: "
        f"{np.nanmin(q[plantadas]):.4f}"
    )
