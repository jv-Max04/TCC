"""`stability.py`, e a VERIFICAÇÃO DE ORDEM que decide se o cache pode existir.

**A pergunta é do critério e é respondida aqui**: se alguma agregação depender da ordem em que as
células foram executadas, duas execuções com o mesmo `fingerprint` divergem, e o cache serve
valor que
não corresponde à chave.

**E ela se responde comparando DUAS ORDENS, não raciocinando sobre a estatística.** Redução de ponto
flutuante depende da ordem dos termos: `(a + b) + c` e `a + (b + c)` diferem no último bit. Fração
sobre `[0, 1]` e contagem são invariantes por serem inteiro sobre inteiro; média acumula e não é.
Cada
agregação declara a classe em `ORDER_CLASS`, e estes testes **exercitam a declaração** — classe
declarada e não exercitada é declaração, não verificação.
"""

import numpy as np
import pytest

from refino.domain.matrix import RefinoMatrix
from refino.domain.metadata import SampleMetadata
from refino.multiverse.executor import run_multiverse
from refino.multiverse.grid import GridDefinition
from refino.multiverse.stability import (
    ORDER_CLASS,
    StabilityThresholds,
    axis_decomposition,
    feature_stability,
)
from refino.statistics.univariate import univariate_test

pytestmark = pytest.mark.property

PAR = ("A", "B")


def _matriz(n_amostras: int = 20, n_features: int = 14) -> RefinoMatrix:
    gerador = np.random.default_rng(11)
    valores = gerador.lognormal(10.0, 0.5, size=(n_amostras, n_features))
    rotulos = ["QC" if i % 5 == 0 else ("A" if i % 2 else "B") for i in range(n_amostras)]

    # Efeito plantado nas quatro primeiras features, para que haja significancia a medir: sem ele
    # `frac_significant` seria zero em toda parte e os testes de ordem compararariam zeros.
    alvo = np.array([r == "A" for r in rotulos])
    valores[np.ix_(alvo, np.arange(4))] *= 3.0

    # OS QCs SAO POOL COM RUIDO PEQUENO, e a primeira versao deste fixture os deixou como
    # amostras quaisquer — RSD de ~53%, acima do corte de 0,30, e `filter_by_qc_rsd` descartava
    # TODAS as 14 features com `EmptyMatrixError`. E a licao na direcao oposta: la o QC
    # sintetico era repetivel demais, aqui era variavel demais. O corte de RSD e o primeiro passo
    # da cascata, entao um fixture cujos QCs nao passam por ele nao exercita nada depois.
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
        transformations=("log10",),
        scalings=("auto",),
    )


@pytest.fixture(scope="module")
def celulas() -> tuple:
    return run_multiverse(_matriz(), _grade()).cells


# ---------------------------------------------------------------------------
# A verificação de ordem
# ---------------------------------------------------------------------------


def test_a_estabilidade_e_identica_sob_duas_ordens_de_celula(celulas: tuple) -> None:
    """**A verificação que o critério precisa.** Duas ordens, resultado bit-idêntico.

    Se este teste reprovar, o cache **não pode existir** na forma desenhada: o `fingerprint` é da
    especificação, e duas execuções da mesma grade em ordens diferentes produziriam valores
    diferentes
    sob a mesma chave. A saída seria ordenar as células antes de agregar — mas ordenar por garantia
    esconde qual agregação é o problema, e é por isso que a classe é declarada por agregação.
    """
    invertidas = tuple(reversed(celulas))
    direta = feature_stability(celulas, PAR)
    inversa = feature_stability(invertidas, PAR)

    assert [f.feature for f in direta] == [f.feature for f in inversa], (
        "a ORDEM DA SAÍDA mudou com a ordem das células — então duas execuções com paralelismo "
        "diferente produzem tuplas em ordens diferentes, e igualdade por `==` não as veria iguais"
    )
    for a, b in zip(direta, inversa, strict=True):
        assert a.frac_significant == b.frac_significant, f"{a.feature}: frac dependeu da ordem"
        assert a.sign_consistency == b.sign_consistency
        assert a.n_cells == b.n_cells
        assert a.verdict == b.verdict
        for campo in ("median_qvalue", "median_rank", "rank_iqr"):
            x, y = getattr(a, campo), getattr(b, campo)
            assert (np.isnan(x) and np.isnan(y)) or x == y, (
                f"{a.feature}.{campo} dependeu da ordem: {x} contra {y}. `ORDER_CLASS` o declara "
                f"como {ORDER_CLASS.get(campo)}, e a declaração acabou de ser falsificada"
            )


def test_a_media_da_decomposicao_e_o_caso_order_dependent_e_o_teste_o_mede(
    celulas: tuple,
) -> None:
    """A média é declarada como dependente de ordem, e este teste **mede** se ela diverge.

    **A declaração não é a verificação.** `ORDER_CLASS` diz que a média acumula, e isso é verdade em
    geral — mas se ela divergir **aqui**, o cache não pode guardar a decomposição, e se não
    divergir,
    a divergência é abaixo do último bit representável nesta população. Os dois desfechos são
    informação; supor um deles não é.

    Registrado como o resultado que ele der: o teste afirma a **igualdade dentro do epsilon de
    máquina** e reporta a divergência máxima, para que o número exista quando alguém precisar dele.
    """
    direta = axis_decomposition(celulas, PAR)
    inversa = axis_decomposition(tuple(reversed(celulas)), PAR)

    assert set(direta) == set(inversa)
    maior = 0.0
    for eixo, opcoes in direta.items():
        assert set(opcoes) == set(inversa[eixo])
        for chave, valor in opcoes.items():
            maior = max(maior, abs(valor - inversa[eixo][chave]))

    print(f"\ndivergência máxima da média sob inversão de ordem: {maior:.3e}")
    assert maior < 1e-12, (
        f"a média da decomposição divergiu em {maior:.3e} sob inversão de ordem. Isso é maior que "
        "ruído de último bit, e o cache do [CA-8.7] não pode guardar a decomposição sem uma ordem "
        "canônica declarada — que é a decisão que este teste existe para forçar"
    )


def test_toda_agregacao_publica_tem_classe_de_ordem_declarada() -> None:
    """Nenhum campo agregado sem classe. **É a guarda contra a próxima agregação.**

    Um campo novo em `FeatureStability` sem entrada em `ORDER_CLASS` é agregação cuja invariância
    ninguém declarou — e o cache passaria a guardá-la sem que a pergunta fosse feita.
    """
    import dataclasses

    from refino.multiverse.stability import FeatureStability

    agregados = {
        c.name
        for c in dataclasses.fields(FeatureStability)
        if c.name not in {"feature", "verdict", "n_cells"}
    }
    faltando = agregados - set(ORDER_CLASS)
    assert not faltando, (
        f"agregações sem classe de ordem declarada: {sorted(faltando)}. Cada uma tem de dizer se é "
        "invariante à ordem, senão o cache do [CA-8.7] a guarda sem ninguém ter perguntado"
    )


# ---------------------------------------------------------------------------
# As propriedades de `feature_stability`
# ---------------------------------------------------------------------------


def test_frac_significant_esta_em_zero_um_e_n_cells_e_o_denominador(celulas: tuple) -> None:
    """A fração e o denominador viajam juntos — o escopo aplicado ao denominador."""
    perfil = feature_stability(celulas, PAR)
    assert len(perfil) == _matriz().n_features
    for f in perfil:
        assert 0.0 <= f.frac_significant <= 1.0
        assert 0 <= f.n_cells <= len(celulas)


def test_feature_filtrada_em_todas_as_celulas_e_absent_com_nan_e_nao_zero() -> None:
    """`absent` com agregados `nan`: "não medido" não é "medido e deu zero"."""
    matriz = _matriz()
    celulas = run_multiverse(matriz, _grade()).cells
    # Uma feature que nenhuma celula reteve nao existe neste fixture; entao a asserção e sobre o
    # CAMINHO: chamado com celulas cujas features nao incluem um nome, ele o marca `absent`.
    perfil = feature_stability(celulas, PAR)
    ausentes = [f for f in perfil if f.n_cells == 0]
    for f in ausentes:
        assert f.verdict == "absent"
        assert np.isnan(f.median_qvalue) and np.isnan(f.sign_consistency)
        assert f.frac_significant == 0.0


def test_o_veredito_e_exaustivo_e_a_lacuna_da_secao_7_cai_em_fragile() -> None:
    """As três condições dadas não cobrem tudo, e a escolha da lacuna é declarada.

    `frac >= 0,80` com `sign < 0,95` não é `robust` pela primeira condição, não é `fragile` pela
    segunda — que exige `frac < 0,80` — e não é `absent`. Cai em `fragile`, porque significância
    estável com sinal **inconsistente** é o caso mais frágil que existe: o efeito aparece e troca de
    direção.
    """
    from refino.multiverse.stability import classifica_veredito

    limiares = StabilityThresholds()
    assert classifica_veredito(0.90, 0.99, limiares) == "robust"
    assert classifica_veredito(0.50, 0.99, limiares) == "fragile"
    assert classifica_veredito(0.01, 0.99, limiares) == "absent"
    # A LACUNA das condições: fracao alta, sinal inconsistente.
    assert classifica_veredito(0.90, 0.50, limiares) == "fragile"


def test_os_limiares_sao_configuraveis_e_mudam_o_veredito(celulas: tuple) -> None:
    """O par causal dos limiares: mudá-los muda a classificação, senão são decorativos."""
    frouxo = feature_stability(
        celulas, PAR, thresholds=StabilityThresholds(robust_frac=0.0, robust_sign=0.0)
    )
    apertado = feature_stability(
        celulas, PAR, thresholds=StabilityThresholds(robust_frac=1.0, robust_sign=1.0)
    )
    assert sum(f.verdict == "robust" for f in frouxo) > sum(f.verdict == "robust" for f in apertado)


def test_a_decomposicao_cobre_os_sete_eixos_e_todas_as_opcoes_da_grade(celulas: tuple) -> None:
    """A decomposição é o que o critério reporta junto do perfil, e tem de ser completa.

    Eixo faltando é escolha cujo efeito ninguém vê — e o critério existe para dizer **quais**
    escolhas
    destroem sinal.
    """
    decomposicao = axis_decomposition(celulas, PAR)
    grade = _grade()
    assert set(decomposicao) == {
        "correction",
        "rsd_cutoff",
        "missing_threshold",
        "imputation",
        "sample_normalization",
        "transformation",
        "scaling",
    }
    assert set(decomposicao["correction"]) == set(grade.corrections)
    assert set(decomposicao["imputation"]) == set(grade.imputations)
    for opcoes in decomposicao.values():
        for valor in opcoes.values():
            assert 0.0 <= valor <= 1.0


def test_celulas_vazias_devolvem_tupla_vazia() -> None:
    """Sem célula não há estabilidade a medir, e a resposta não é levantar."""
    assert feature_stability((), PAR) == ()


# ---------------------------------------------------------------------------
# A NÃO-REGRESSÃO DE `sign_consistency` — o número já persistido que a mudança aditiva protege
# ---------------------------------------------------------------------------

# **Medido na árvore de 11/09/2026, ANTES de `univariate_test` ganhar o parâmetro de matriz.**
#
# Não é tolerância nem meta: é o valor que a implementação de hoje produz sobre este fixture,
# Congelado para que a mudança aditiva seja **medida** e não suposta. `sign_consistency`
# está persistida (`stability.py:198`) e tem invariância declarada em `ORDER_CLASS`; migrar os
# chamadores em vez de acrescentar o parâmetro deslocaria estes catorze números em toda a grade.
#
# Os valores são razões de inteiros pequenos — `max((>0).mean(), (<0).mean())` sobre no máximo 16
# células —, exactamente representáveis em binário. Congelá-los com `==` é legítimo; uma tolerância
# aqui esconderia justamente o deslocamento que o teste existe para apanhar.
#
# **[MOVIDO 17/10/2026] O veredito da `f000` passou de `"absent"` a `"fragile"`, e a causa é
# a decisão de normalidade, não a mudança aditiva.** A decisão antiga centrava os resíduos pela
# mediana e rejeitava a normalidade de qualquer dado: escolhia Mann-Whitney nas 16 células deste
# fixture. A corrigida escolhe Welch em 4 das 16, e a `f000` — uma das quatro com efeito plantado —
# passa a ser significativa em alguma delas. Conferido antes de mexer: com a decisão antiga
# reposta, esta tabela inteira é reproduzida exactamente, e nenhuma outra feature se move com a
# nova. O par `sign_consistency` das catorze não mudou.
SIGN_CONSISTENCY_ANTES_DE_KTD8: dict[str, tuple[float, str]] = {
    "f000": (0.0, "fragile"),
    "f001": (0.0, "fragile"),
    "f002": (0.0, "fragile"),
    "f003": (0.0, "fragile"),
    "f004": (0.0, "absent"),
    "f005": (0.0, "absent"),
    "f006": (0.0, "fragile"),
    "f007": (0.0, "fragile"),
    "f008": (1.0, "absent"),
    "f009": (0.0, "absent"),
    "f010": (1.0, "absent"),
    "f011": (1.0, "absent"),
    "f012": (0.5, "absent"),
    "f013": (0.0, "fragile"),
}


def test_o_sinal_plantado_no_fixture_e_POSITIVO_e_conhecido() -> None:  # noqa: N802
    """A âncora de sinal conhecido, sem a qual a tabela congelada não mede nada.

    `_matriz` multiplica as quatro primeiras features do grupo `A` por 3,0. Com o par `("A", "B")`
    o numerador é `A`, então `log2fc` na matriz **ingerida** tem de ser positivo nas quatro.

    **Sem esta asserção a tabela de baixo seria só um retrato**: `sign_consistency = 0` nas quatro
    features com efeito plantado leria-se como propriedade do dado, quando é consequência de o
    fold change sair da matriz processada.
    """
    resultado = univariate_test(_matriz(), PAR)

    assert np.isfinite(resultado.log2fc).all(), (
        "na matriz ingerida todos os 14 `log2fc` têm de ser finitos; se não são, o fixture mudou "
        "e a âncora de sinal deixou de existir"
    )
    plantadas = resultado.log2fc[:4]
    assert (plantadas > 0.0).all(), (
        f"o sinal plantado é POSITIVO (grupo A = 3x grupo B nas quatro primeiras features) e "
        f"medi {plantadas}. A tabela congelada não mede nada sem esta âncora"
    )


def test_sign_consistency_nao_se_move_e_o_numero_persistido_de_KTD8(  # noqa: N802
    celulas: tuple,
) -> None:
    """**A asserção que protege o número já gravado.** Catorze features, valor exacto.

    A mudança é aditiva por causa deste teste: as três chamadas de `univariate_test` em
    `multiverse/stability.py` ficam com a semântica de hoje — fold change da matriz **processada**
    —, e só a rota das figuras pede a pré-transformação. Se algum destes catorze valores mudar, o
    parâmetro vazou para o multiverso, e isso é regra 2 do README: parar e reportar, nunca ajustar a
    tabela.

    **A não-vacuidade é afirmada na mesma função**: a tabela cobre as catorze features e
    tem ao menos um valor não nulo. Uma tabela toda a zero passaria com um pipeline quebrado que
    devolvesse sinal indefinido em toda parte.
    """
    assert len(SIGN_CONSISTENCY_ANTES_DE_KTD8) == _matriz().n_features == 14
    assert any(v != 0.0 for v, _ in SIGN_CONSISTENCY_ANTES_DE_KTD8.values()), (
        "a tabela congelada é toda zero, então ela passaria com qualquer implementação que "
        "devolvesse sinal indefinido em toda parte"
    )

    perfil = feature_stability(celulas, PAR)
    medido = {f.feature: (f.sign_consistency, f.verdict) for f in perfil}
    assert medido == SIGN_CONSISTENCY_ANTES_DE_KTD8, (
        "`sign_consistency` MOVEU-SE. Ela está persistida e tem invariância declarada; a mudança "
        "de KTD8 é aditiva justamente para que estes números não se desloquem. Isto é a regra 2 "
        "do README — relatar o caso, nunca acertar a tabela"
    )


def test_o_fold_change_da_matriz_processada_e_indefinido_no_caso_NORMAL(  # noqa: N802
    celulas: tuple,
) -> None:
    """**A medição que explica a tabela acima, e é o argumento inteiro.**

    Depois de `log10` e `auto`, a mediana de um grupo é negativa de rotina, e `_fold_change`
    devolve `nan` para toda razão não positiva. Medido neste fixture: das 14 features, só **2 ou
    3** por célula têm `log2fc` definido na matriz processada, contra **14 de 14** na ingerida.

    Não é caso de borda — é o caso normal, e é por isso que o volcano precisa do parâmetro.
    """
    definidos = [int(np.isfinite(univariate_test(c.matrix, PAR).log2fc).sum()) for c in celulas]
    totais = {c.matrix.n_features for c in celulas}

    assert set(definidos) == {2, 3}, (
        f"medi {sorted(set(definidos))} fold changes definidos por célula e esperava 2 ou 3 de "
        f"{totais}. Se subiu, a matriz processada deixou de ter mediana negativa e o argumento "
        "de KTD8 precisa de ser remedido"
    )
    assert int(np.isfinite(univariate_test(_matriz(), PAR).log2fc).sum()) == 14
