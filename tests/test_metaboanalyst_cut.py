"""O corte de 5.000 e as três causas de remoção.

**Onde este módulo mora, e não é em `tests/golden/`.** O marcador `golden` é
comparação contra as **saídas do R**; este par causal compara contra
contagens medidas sobre os arquivos de **entrada**. Pôr aqui o que é `golden` faria
a seleção `-m golden` e o noturno medirem uma população que não é a deles.

**A população é a da INGESTÃO, e a declaração é o conteúdo.** Medido em
04/08/2026: 6.035 / 2.572 / 2.039 / 2.184 features na ingestão para `pos_hilic`,
`pos_c18`, `neg_hilic` e `neg_c18`; a jusante de `filter_by_qc_rsd` com corte de
0,30 sobrevivem 758 / 1.922 / 224 / 1.569. **Nenhum modo alcança 5.000 depois do
filtro**, então um export sobre a matriz final deixaria o corte inerte em 4 de 4 e
o critério trivialmente satisfeito.

Os números desta suíte são **observação sobre estes quatro arquivos**, não
invariante. Cada asserção nomeia a população.
"""

from pathlib import Path

import numpy as np
import pytest

from refino.domain.labels import InvalidLabelError
from refino.io.readers import load_matrix
from refino.io.writers import write_metaboanalyst
from refino.metrics.rsd import qc_rsd
from tests.conftest import measurements_path, sample_list_path

# Populações medidas na ingestão, por modo. `None` onde o corte não é exercitável.
INGESTION = {"pos_hilic": 6035, "pos_c18": 2572, "neg_hilic": 2039, "neg_c18": 2184}

# Indefinidas por variante, medidas. `plain` é a causa A isolada;
# `outlier_removed` é a soma das duas causas.
UNDEFINED = {
    "pos_hilic": {"plain": 102, "outlier_removed": 234},
    "neg_hilic": {"plain": 136, "outlier_removed": 212},
    "pos_c18": {"plain": 35, "outlier_removed": 60},
    "neg_c18": {"plain": 25, "outlier_removed": 33},
}

MODES = tuple(INGESTION)


@pytest.fixture(scope="module")
def matrices() -> dict[str, object]:
    return {
        mode: load_matrix(measurements_path(mode), sample_list_path(mode), mode) for mode in MODES
    }


# ---------------------------------------------------------------------------
# O par causal do corte
# ---------------------------------------------------------------------------


def test_to_11_1_o_pos_hilic_mantem_5000_e_remove_1035(matrices: dict, tmp_path: Path) -> None:
    """Direção que CORTA. O `pos_hilic` é o único modo que excede 5.000.

    A conta fecha por construção do critério: 6.035 na ingestão, 5.000 mantidas,
    1.035 removidas — das quais **234 são indefinidas** e as outras 801 saem por
    RSD alto.
    """
    matrix = matrices["pos_hilic"]
    assert matrix.n_features == 6035, "a população do `pos_hilic` mudou; remedir"

    report = write_metaboanalyst(matrix, tmp_path / "pos_hilic.csv")

    assert len(report.kept) == 5000, "o corte tem de manter exatamente 5.000"
    assert len(report.removed) == 1035, "6.035 menos 5.000 são 1.035 removidas"
    assert len(report.removed_undefined) == 234, (
        "as indefinidas em `outlier_removed` do `pos_hilic` são 234"
    )
    assert len(report.removed_high_rsd) == 801, "1.035 removidas menos 234 indefinidas"
    assert len(report.kept) + len(report.removed) == matrix.n_features, "a partição não fecha"


def test_to_11_1_as_mantidas_sao_as_de_menor_rsd(matrices: dict, tmp_path: Path) -> None:
    """O núcleo do critério: "as de **menor** RSD", e nada mais afirmava isto.

    **Esta asserção nasceu de uma planta que passou.** Inverter a ordenação do corte
    — manter as 5.000 **piores** em vez das melhores — passou os catorze testes deste
    módulo. As contagens fechavam, a partição fechava, as indefinidas saíam primeiro,
    e a única coisa que o critério de fato pede estava sem verificação. É a
    forma mais crua de negativo inerte: o mecanismo estava exercitado e a direção que importa,
    inerte.

    A asserção é de fronteira e não de conjunto, porque é ela que distingue as duas
    ordenações: o **pior** RSD entre as mantidas tem de ser melhor que o **melhor**
    RSD entre as removidas por RSD alto.
    """
    matrix = matrices["pos_hilic"]

    report = write_metaboanalyst(matrix, tmp_path / "pos_hilic.csv")

    rsd = qc_rsd(matrix, variant="outlier_removed")
    by_name = dict(zip(matrix.feature_names, rsd.tolist(), strict=True))

    kept_rsd = [by_name[name] for name in report.kept]
    cut_rsd = [by_name[name] for name in report.removed_high_rsd]

    assert max(kept_rsd) <= min(cut_rsd), (
        "há mantida com RSD pior que uma removida por RSD alto — o corte manteve as "
        "erradas. `[CA-11.1]` pede as de MENOR RSD."
    )

    # E o conjunto, para o caso de a fronteira coincidir por empate.
    defined = [(name, value) for name, value in by_name.items() if not np.isnan(value)]
    best = {name for name, _ in sorted(defined, key=lambda pair: pair[1])[:5000]}
    assert set(report.kept) == best, "o conjunto mantido não é o das 5.000 de menor RSD"


def test_to_11_1_as_indefinidas_saem_primeiro(matrices: dict, tmp_path: Path) -> None:
    """Nenhuma indefinida sobrevive ao corte, e é isso que "primeiro" significa.

    RSD indefinido significa que não se sabe nada sobre a estabilidade daquela
    feature; manter o desconhecido no lugar do medido inverteria o critério.
    """
    matrix = matrices["pos_hilic"]

    report = write_metaboanalyst(matrix, tmp_path / "pos_hilic.csv")

    rsd = qc_rsd(matrix, variant="outlier_removed")
    undefined_names = {matrix.feature_names[i] for i in np.flatnonzero(np.isnan(rsd)).tolist()}
    assert not (set(report.kept) & undefined_names), (
        "sobrou indefinida entre as mantidas; as indefinidas têm de sair primeiro"
    )
    assert set(report.removed_undefined) == undefined_names


@pytest.mark.parametrize("mode", ("pos_c18", "neg_hilic", "neg_c18"))
def test_to_11_1_os_outros_tres_exportam_integros(
    mode: str, matrices: dict, tmp_path: Path
) -> None:
    """Direção que NÃO corta — e sem ela o teste acima não prova nada.

    É o par causal exigido: um modo em que o corte dispara e um em que não. Sem a
    segunda direção, uma implementação que cortasse **sempre** passaria.

    A lista de removidas é **vazia** — inclusive as indefinidas, que ficam. Descartar
    indefinida sem haver excesso foi o meu primeiro desenho, e foi este item que o
    reprovou: "primeiro" é prioridade dentro do corte, não filtro incondicional.
    """
    matrix = matrices[mode]
    assert matrix.n_features == INGESTION[mode] < 5000

    report = write_metaboanalyst(matrix, tmp_path / f"{mode}.csv")

    assert report.removed == (), f"{mode} não excede 5.000; a lista de removidas é vazia"
    assert len(report.kept) == matrix.n_features, "a exportação é íntegra"


# ---------------------------------------------------------------------------
# A separação das duas causas. Um só balde reprova.
# ---------------------------------------------------------------------------


def test_to_11_2_as_duas_causas_de_indefinicao_saem_separadas(
    matrices: dict, tmp_path: Path
) -> None:
    """Um só balde reprova.

    A causa A é indefinida **também** em `plain` — QCs todos zero, feature nunca
    detectada. A causa B é indefinida só em `outlier_removed` — a remoção de
    outliers apagou os QCs não nulos, feature detectada em um único QC.

    No `pos_hilic` são **102** e **132**, e a maioria é a segunda. As duas levam a
    ações diferentes: a primeira sugere olhar o dado, a segunda sugere olhar o
    corretor.

    **Só o `pos_hilic`**, e o recorte é consequência do achado acima: sem excesso não
    há corte, então nos outros três não há removida a classificar. O critério é
    exercitável em 1 de 4 modos reais, pela mesma razão que o corte é.
    """
    mode = "pos_hilic"
    matrix = matrices[mode]
    expected_a = UNDEFINED[mode]["plain"]
    expected_total = UNDEFINED[mode]["outlier_removed"]

    report = write_metaboanalyst(matrix, tmp_path / f"{mode}.csv")

    assert len(report.removed_all_zero_qcs) == expected_a, (
        f"{mode}: a causa A (QCs todos zero) é a contagem de `plain`, {expected_a}"
    )
    assert len(report.removed_outliers_erased) == expected_total - expected_a, (
        f"{mode}: a causa B é a diferença entre as variantes, {expected_total} menos {expected_a}"
    )
    assert not (set(report.removed_all_zero_qcs) & set(report.removed_outliers_erased)), (
        "as duas causas têm de ser disjuntas"
    )


def test_to_11_2_um_so_balde_nao_reproduz_a_particao(matrices: dict, tmp_path: Path) -> None:
    """A direção que prova que a separação não é decorativa.

    Se as duas causas fossem um só campo, esta asserção seria impossível de
    escrever: **102 e 132 não são recuperáveis de 234**. O valor de
    retorno colapsado torna indistinguíveis em contagem duas coisas que levam a
    ações opostas.
    """
    report = write_metaboanalyst(matrices["pos_hilic"], tmp_path / "pos_hilic.csv")

    assert len(report.removed_undefined) == 234
    assert (len(report.removed_all_zero_qcs), len(report.removed_outliers_erased)) == (102, 132)
    assert report.removed_all_zero_qcs != report.removed_undefined, (
        "a causa A não pode ser igual ao balde agregado; se for, a partição colapsou"
    )


# ---------------------------------------------------------------------------
# A variante muda o conjunto
# ---------------------------------------------------------------------------


def test_to_11_3_trocar_a_variante_muda_o_conjunto(matrices: dict, tmp_path: Path) -> None:
    """O rótulo não é decorativo, e os pares estão medidos.

    `plain` contra `outlier_removed`: 102/234 no `pos_hilic`, 136/212 no
    `neg_hilic`, 35/60 no `pos_c18`, 25/33 no `neg_c18` — contenção estrita nos
    quatro. Sob `plain` a causa B é **vazia por construção**: sem remoção de
    outliers não há QC apagado por ela.

    **Só o `pos_hilic`**, pelo mesmo recorte do critério: nos outros três não há
    corte, logo não há conjunto de removidas que a variante possa mudar.
    """
    mode = "pos_hilic"
    matrix = matrices[mode]

    default = write_metaboanalyst(matrix, tmp_path / f"{mode}_or.csv")
    plain = write_metaboanalyst(matrix, tmp_path / f"{mode}_pl.csv", rsd_variant="plain")

    assert len(plain.removed_undefined) == UNDEFINED[mode]["plain"]
    assert len(default.removed_undefined) == UNDEFINED[mode]["outlier_removed"]
    assert set(plain.removed_undefined) != set(default.removed_undefined), (
        "trocar a variante tem de mudar o conjunto de removidas, senão o rótulo é decorativo"
    )
    assert set(plain.removed_undefined) < set(default.removed_undefined), (
        "a contenção medida é estrita: toda indefinida em `plain` também o é "
        "em `outlier_removed`, e não o contrário"
    )
    assert plain.removed_outliers_erased == (), (
        "sob `plain` a causa B é vazia por construção — não há remoção de outliers "
        "que possa ter apagado QC"
    )


# ---------------------------------------------------------------------------
# Os quatro desfechos vistos, e as guardas
# ---------------------------------------------------------------------------


def test_os_quatro_desfechos_aparecem_numa_execucao(matrices: dict, tmp_path: Path) -> None:
    """Quantos desfechos este mecanismo pode reportar, e eu vi todos?

    O `pos_hilic` é o único modo em que os quatro ramos disparam na mesma execução
    — mantida, causa A, causa B e causa C. O ramo `mantida` é o que dispara por
    **ausência** de condição, e por isso o mais fácil de ficar inerte.
    """
    report = write_metaboanalyst(matrices["pos_hilic"], tmp_path / "pos_hilic.csv")

    assert report.kept, "ramo `mantida` inerte"
    assert report.removed_all_zero_qcs, "ramo causa A inerte"
    assert report.removed_outliers_erased, "ramo causa B inerte"
    assert report.removed_high_rsd, "ramo causa C inerte"


def test_rsd_variant_invalido_levanta_na_fronteira(matrices: dict, tmp_path: Path) -> None:
    """Regra 2: `Literal` é análise estática, e este valor atravessa a CLI.

    O modo de falha que a guarda impede é o insidioso: `"outlier-removed"` com
    hífen cairia num `else` razoável, se comportaria como a outra variante, e
    produziria conjunto de removidas diferente do pedido **sem erro**.
    """
    with pytest.raises(InvalidLabelError):
        write_metaboanalyst(
            matrices["pos_c18"],
            tmp_path / "x.csv",
            rsd_variant="outlier-removed",  # type: ignore[arg-type]
        )


def test_matriz_com_infinito_levanta(matrices: dict, tmp_path: Path) -> None:
    """Com `±Inf` a separação das duas causas deixa de ser válida.

    `sd` de vetor com `Inf` é `nan` por `inf - inf`, então a feature fica
    indefinida em `plain` **sem** ter os QCs todos zero — e a causa A absorveria um
    mecanismo que não é o dela. Recusar é melhor que classificar errado em silêncio.
    """
    matrix = matrices["pos_c18"]
    values = matrix.values.copy()
    values[0, 0] = np.inf

    with pytest.raises(ValueError, match="infinitos"):
        write_metaboanalyst(matrix.with_values(values, "teste"), tmp_path / "x.csv")


@pytest.mark.parametrize("limit", (0, -1))
def test_max_features_sem_sentido_levanta(limit: int, matrices: dict, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="max_features"):
        write_metaboanalyst(matrices["pos_c18"], tmp_path / "x.csv", limit)


def test_o_corte_e_deterministico(matrices: dict, tmp_path: Path) -> None:
    """Mesma entrada, mesma saída.

    Empate de RSD na fronteira das 5.000 é resolvido por índice original com
    ordenação estável; sem isso o conjunto dependeria da ordem interna do
    `argsort`, e o determinismo reprovaria de forma intermitente — que é pior que
    reprovar sempre.
    """
    matrix = matrices["pos_hilic"]

    first = write_metaboanalyst(matrix, tmp_path / "a.csv")
    second = write_metaboanalyst(matrix, tmp_path / "b.csv")

    assert first == second
    assert (tmp_path / "a.csv").read_bytes() == (tmp_path / "b.csv").read_bytes()
