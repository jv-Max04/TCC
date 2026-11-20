"""A composição, a equivalência com a execução ingênua, o contador e o executor.

**A ordem é deliberada.** O critério prova a árvore por **equivalência bit-idêntica** e o
o contador por **contador**; cache, paralelismo e progresso não têm nenhum dos dois, e é sobre esse
código que qualquer perfil vai rodar. Medir perfil sobre executor sem teste é medir dois artefatos.

**Sobre fixture pequeno e não sobre matriz real, de propósito.** A equivalência entre árvore e
execução ingênua é propriedade do **executor** e não do dado: a ingênua corrige uma vez por
especificação, então sobre a grade padrão ela custaria as 4.320 correções que a árvore existe para
evitar. Grade reduzida com corretor `none` e `loess` exercita o mesmo caminho por um custo que cabe
no portão de commit.
"""

import numpy as np
import pytest

from refino.domain.matrix import RefinoMatrix
from refino.domain.metadata import SampleMetadata
from refino.multiverse.executor import (
    CorrectorCallCounter,
    build_corrector,
    run_multiverse,
    run_naive,
)
from refino.multiverse.grid import GridDefinition

pytestmark = pytest.mark.property


def _matriz(n_amostras: int = 16, n_features: int = 12) -> RefinoMatrix:
    """Fixture com QCs intercalados e dois grupos, na forma que os corretores exigem."""
    gerador = np.random.default_rng(7)
    valores = gerador.lognormal(10.0, 0.6, size=(n_amostras, n_features))
    amostras = []
    for indice in range(n_amostras):
        e_qc = indice % 4 == 0
        amostras.append(
            SampleMetadata(
                name=f"{'QC' if e_qc else 'S'}{indice:03d}",
                injection_order=indice + 1,
                sample_type="qc" if e_qc else "sample",
                group="QC" if e_qc else ("A" if indice % 2 else "B"),
                batch="A",
            )
        )
    return RefinoMatrix(
        values=valores,
        feature_names=tuple(f"f{j:03d}" for j in range(n_features)),
        samples=tuple(amostras),
        mode="sintetico",
    )


def _grade_reduzida() -> GridDefinition:
    """Grade pequena que ainda tem MAIS DE UM prefixo — é o que torna a árvore exercitável.

    Com um prefixo só, árvore e execução ingênua coincidem trivialmente e o critério não
    distingue nada: `size()` seria igual a `len(prefix_keys())` por construção.
    """
    return GridDefinition(
        corrections=("none", "loess"),
        batch_options=(1,),
        rsd_cutoffs=(None, 0.30),
        missing_thresholds=(0.8,),
        imputations=("one_fifth_min", "median"),
        sample_norms=("none", "sum"),
        transformations=("none", "log10"),
        scalings=("auto",),
    )


# ---------------------------------------------------------------------------
# A composição, não um número escrito
# ---------------------------------------------------------------------------


def test_ca_8_1_size_bate_com_a_composicao_declarada() -> None:
    """`size()` é o produto da composição, e a composição é afirmada.

    **Afirmar o produto sozinho não bastaria**: dois eixos com contagens trocadas dão o mesmo
    `size()` e grades diferentes. É por isso que o critério cobra a composição, e por isso
    este teste multiplica em vez de comparar contra um inteiro.
    """
    grade = GridDefinition()
    esperado = 1
    for _, n in grade.composition():
        esperado *= n

    assert grade.size() == esperado
    assert dict(grade.composition()) == {
        "corrections": 3,
        "batch_options": 1,
        "rsd_cutoffs": 4,
        "missing_thresholds": 2,
        "imputations": 5,
        "sample_norms": 5,
        "transformations": 4,
        "scalings": 4,
    }


def test_enumerate_rende_exatamente_size_especificacoes() -> None:
    """O iterador e o contador têm de concordar — senão um dos dois mente sobre a grade."""
    grade = GridDefinition()
    assert sum(1 for _ in grade.enumerate()) == grade.size()


def test_enumerate_agrupa_por_prefixo_em_blocos_contiguos() -> None:
    """As especificações do mesmo prefixo saem **juntas**, e é isso que a árvore consome.

    Se a ordem intercalasse prefixos, o `groupby` do executor abriria o mesmo grupo várias vezes e
    corrigiria mais de uma vez por prefixo — que é exactamente o defeito que o critério pega.
    Este teste o pega antes, na fonte.
    """
    vistos: list[tuple[str, int]] = []
    for spec in GridDefinition().enumerate():
        chave = (spec.correction, spec.n_batches)
        if not vistos or vistos[-1] != chave:
            vistos.append(chave)
    assert len(vistos) == len(set(vistos)), (
        f"prefixos reabertos: {vistos}. A ordem intercala grupos, e a árvore corrigiria mais de "
        "uma vez por prefixo"
    )


def test_eixo_vazio_e_recusado_na_construcao() -> None:
    """Grade com eixo vazio zera o produto, e executor sobre grade vazia parece sucesso rápido."""
    with pytest.raises(ValueError, match="vazio"):
        GridDefinition(scalings=())


def test_imputation_none_com_normalizacao_e_recusada_na_grade() -> None:
    """A combinação é recusada; a grade a recusa **antes** de enumerar.

    Sem esta guarda, `enumerate()` levantaria no meio, depois de já ter rendido especificações — e
    um consumidor que já processou metade não tem estado consistente para reportar.
    """
    with pytest.raises(ValueError, match="o PipelineSpec recusa a combinação"):
        GridDefinition(imputations=("none",))
    # E a combinacao VALIDA passa: `none` com normalizacao tambem nula.
    assert GridDefinition(imputations=("none",), sample_norms=("none",)).size() > 0


# ---------------------------------------------------------------------------
# O contador, e o par causal dele
# ---------------------------------------------------------------------------


def test_ca_8_4_a_arvore_corrige_uma_vez_por_prefixo() -> None:
    """Invocações do corretor iguais a `len(prefix_keys())`, não a `size()`."""
    grade = _grade_reduzida()
    execucao = run_multiverse(_matriz(), grade)

    assert execucao.counter.defects_against(grade) == []
    assert execucao.counter.total == len(grade.prefix_keys())
    assert execucao.counter.total < grade.size()
    assert len(execucao.cells) == grade.size()


def test_ca_8_4_a_execucao_ingenua_corrige_uma_vez_por_especificacao() -> None:
    """**O par causal do contador.** A ingênua dá `size()` invocações, e o contador as vê.

    Sem esta metade, um contador que sempre devolvesse `len(prefix_keys())` — por estar contando a
    coisa errada — passaria o teste acima. Aqui ele tem de reportar o número **maior**, e o
    predicado tem de acusar.
    """
    grade = _grade_reduzida()
    execucao = run_naive(_matriz(), grade)

    # SEM numero escrito a mao, e a primeira versao deste teste afirmava `== 16` quando
    # `size()` da 32 — a multiplicacao errada, no teste do criterio que existe para nao haver
    # Numero a mao. E a confirmacao mais direta possivel do argumento.
    assert execucao.counter.total == grade.size()
    defeitos = execucao.counter.defects_against(grade)
    assert defeitos != [], "o predicado não acusou a execução que corrige uma vez por especificação"
    assert "amortiza" in defeitos[0]


def test_o_contador_acusa_prefixo_nao_executado_e_prefixo_repetido() -> None:
    """As duas formas que o total agregado esconderia, plantadas como argumento.

    Total igual ao esperado com uma chave chamada duas vezes e outra nenhuma é a árvore quebrada de
    um jeito que um contador de total não vê. A planta é argumento do predicado, então o negativo
    percorre o mesmo caminho que o portão.
    """
    grade = _grade_reduzida()

    faltando = CorrectorCallCounter(per_prefix={("none", 1): 2})
    assert any("esperado exactamente 1" in d for d in faltando.defects_against(grade))

    estranho = CorrectorCallCounter(per_prefix={("none", 1): 1, ("serrf", 1): 1})
    assert any("não pertence à grade" in d for d in estranho.defects_against(grade))

    correto = CorrectorCallCounter(per_prefix={("none", 1): 1, ("loess", 1): 1})
    assert correto.defects_against(grade) == []


# ---------------------------------------------------------------------------
# Equivalência bit-idêntica
# ---------------------------------------------------------------------------


def test_ca_8_3_arvore_e_ingenua_produzem_saida_bit_identica() -> None:
    """A árvore é equivalente à execução ingênua, célula a célula.

    **Bit-idêntica e não "dentro de tolerância"**: a árvore reusa o mesmo `CorrectionResult` em vez
    de recalculá-lo, e reuso de resultado idêntico não pode mudar dígito nenhum. Tolerância aqui
    esconderia exactamente o defeito que interessa — um prefixo corrigido com argumento diferente.
    """
    matriz, grade = _matriz(), _grade_reduzida()
    arvore = run_multiverse(matriz, grade)
    ingenua = run_naive(matriz, grade)

    assert len(arvore.cells) == len(ingenua.cells)
    for a, b in zip(arvore.cells, ingenua.cells, strict=True):
        assert a.spec == b.spec
        assert np.array_equal(np.asarray(a.matrix.values), np.asarray(b.matrix.values)), (
            f"divergência em {a.spec.fingerprint()}: a árvore reusa o CorrectionResult e a ingênua "
            "o recalcula, e reuso de resultado idêntico não pode mudar dígito nenhum"
        )
        assert a.matrix.feature_names == b.matrix.feature_names
        assert [s.n_features_out for s in a.stages] == [s.n_features_out for s in b.stages]


def test_o_correction_result_inteiro_e_guardado_por_prefixo() -> None:
    """O nó guarda o `CorrectionResult`, não só a matriz.

    As contagens por causa são do **nó** e valem para a subárvore toda; recalculá-las nas folhas
    seria refazer a correção. Sem elas, uma especificação em que features não foram corrigidas
    apareceria ao lado de outra em que todas foram, com a mesma aparência.
    """
    grade = _grade_reduzida()
    execucao = run_multiverse(_matriz(), grade)

    assert set(execucao.corrections) == set(grade.prefix_keys())
    for prefixo, resultado in execucao.corrections.items():
        assert hasattr(resultado, "uncorrected"), f"{prefixo} guardou só a matriz"
        assert resultado.matrix.n_features == _matriz().n_features


# ---------------------------------------------------------------------------
# Os testes do executor: paralelismo, progresso, contrato do corretor
# ---------------------------------------------------------------------------


def test_o_paralelismo_das_folhas_nao_muda_o_numero() -> None:
    """Serial e paralelo produzem saída idêntica — e o serial é o padrão, não o fallback.

    A única forma de o `joblib` mudar o número seria alguma etapa consumindo estado global, e é
    justamente o que este teste pega.
    """
    matriz, grade = _matriz(), _grade_reduzida()
    serial = run_multiverse(matriz, grade, n_jobs=1)
    paralelo = run_multiverse(matriz, grade, n_jobs=2)

    assert serial.counter.per_prefix == paralelo.counter.per_prefix
    for a, b in zip(serial.cells, paralelo.cells, strict=True):
        assert a.spec == b.spec
        assert np.array_equal(np.asarray(a.matrix.values), np.asarray(b.matrix.values))


def test_o_padrao_de_n_jobs_e_1_e_o_serrf_recebe_1_explicito() -> None:
    """O contrato: paralelismo nas folhas, corretor em 1.

    **Explícito e não herdado**: se alguém mudar o padrão do `SerrfCorrector` para `-1` por outra
    razão, este teste reprova em vez de a sobrescrição aparecer como lentidão inexplicada.
    """
    assert run_multiverse(_matriz(), _grade_reduzida()).n_jobs == 1
    assert build_corrector("serrf").n_jobs == 1
    assert build_corrector("none").name == "none"
    assert build_corrector("loess").name == "loess"
    with pytest.raises(ValueError, match="desconhecido"):
        build_corrector("inexistente")  # type: ignore[arg-type]


def test_o_progresso_e_chamado_uma_vez_por_celula_e_em_ordem() -> None:
    """`progress(feito, total)` — monotônico, terminando em `total == size()`.

    Relatório de progresso que salta ou repete número é o defeito que só aparece na leitura, e
    ninguém lê progresso de teste que passa. Aqui ele é afirmado.
    """
    grade = _grade_reduzida()
    chamadas: list[tuple[int, int]] = []
    run_multiverse(_matriz(), grade, progress=lambda f, t: chamadas.append((f, t)))

    assert len(chamadas) == grade.size()
    assert [f for f, _ in chamadas] == list(range(1, grade.size() + 1))
    assert {t for _, t in chamadas} == {grade.size()}


def test_o_contador_pode_ser_injetado_e_acumula_entre_execucoes() -> None:
    """Contador injetado é o que permite ao critério ser verificado de fora do executor.

    E acumular é a propriedade que torna duas execuções distinguíveis de uma: sem ela, um executor
    que corrigisse duas vezes por chamada pareceria correto quando o contador fosse novo a cada vez.
    """
    matriz, grade = _matriz(), _grade_reduzida()
    contador = CorrectorCallCounter()
    run_multiverse(matriz, grade, counter=contador)
    run_multiverse(matriz, grade, counter=contador)

    assert contador.total == 2 * len(grade.prefix_keys())
    assert any("esperado exactamente 1" in d for d in contador.defects_against(grade))


def test_a_grade_de_um_prefixo_so_nao_exercita_a_arvore() -> None:
    """A guarda contra o fixture que torna o critério vácuo.

    Com um prefixo só, `size()` e `len(prefix_keys())` podem coincidir e o critério não distingue
    árvore de execução ingênua. Este teste afirma que a grade reduzida usada aqui **não** é desse
    caso — é o par causal do fixture, na forma.
    """
    degenerada = GridDefinition(
        corrections=("none",),
        rsd_cutoffs=(0.30,),
        missing_thresholds=(0.8,),
        imputations=("median",),
        sample_norms=("sum",),
        transformations=("log10",),
        scalings=("auto",),
    )
    assert degenerada.size() == len(degenerada.prefix_keys()) == 1

    reduzida = _grade_reduzida()
    assert len(reduzida.prefix_keys()) > 1
    assert reduzida.size() > len(reduzida.prefix_keys())


# ---------------------------------------------------------------------------
# Cobertura de eixos: um eixo por vez, e a UNIÃO cobre a grade completa
# ---------------------------------------------------------------------------


def _grades_de_cobertura() -> dict[str, GridDefinition]:
    """Uma grade por eixo, com o eixo CHEIO e os outros num valor só.

    **A tensão que isto resolve.** `enumerate()` é produto cartesiano, então cobrir todos os
    valores de todos os eixos ao mesmo tempo **é** a grade completa — 4.320 células. Mas a
    equivalência entre árvore e execução ingênua é propriedade do executor, e o que a grade
    reduzida pode não cobrir é outra coisa: **valor de eixo que só existe na grade completa**, e
    que poderia falhar por si — um `sample_normalization` que levante, um `scaling` degenerado.

    Um eixo por vez separa as duas perguntas: a álgebra da árvore é exercitada pela grade de 32
    com prefixos compartilhados, e cada valor de eixo é exercitado aqui, por
    `soma(len(eixo))` células em vez do produto delas.
    """
    padrao = GridDefinition()
    base: dict[str, object] = {
        "corrections": ("none",),
        "batch_options": (1,),
        "rsd_cutoffs": (0.30,),
        "missing_thresholds": (0.8,),
        "imputations": ("median",),
        "sample_norms": ("sum",),
        "transformations": ("log10",),
        "scalings": ("auto",),
    }
    grades = {}
    for eixo, _ in padrao.composition():
        grades[eixo] = GridDefinition(**{**base, eixo: getattr(padrao, eixo)})  # type: ignore[arg-type]
    return grades


def test_a_uniao_das_grades_de_teste_cobre_todo_valor_da_grade_completa() -> None:
    """Nenhum valor de eixo da grade padrão fica sem execução em teste.

    **É o que a grade de 32 não cobre**, e o levantamento é explícito: ela tem 2 de 3 correções,
    2 de 4 cortes, 1 de 2 limiares de ausência, 2 de 3 imputações, 2 de 5 normalizações, 2 de 4
    transformações e 1 de 3 escalonamentos. Sete eixos incompletos.

    A união das grades de cobertura fecha os sete, e o custo é a **soma** dos comprimentos dos
    eixos — 25 células — e não o produto.
    """
    padrao = GridDefinition()
    grades = _grades_de_cobertura()

    for eixo, _ in padrao.composition():
        coberto: set[object] = set()
        for grade in grades.values():
            coberto |= set(getattr(grade, eixo))
        faltando = set(getattr(padrao, eixo)) - coberto
        assert not faltando, (
            f"eixo {eixo!r} com valor sem cobertura em teste nenhum: {sorted(map(str, faltando))}. "
            "Valor que só existe na grade completa é caminho que nada exercita"
        )


@pytest.mark.parametrize("eixo", [n for n, _ in GridDefinition().composition()])
def test_ca_8_3_equivalencia_por_eixo_com_todos_os_valores(eixo: str) -> None:
    """Por eixo: árvore e ingênua bit-idênticas com **todos** os valores daquele eixo.

    Estende a equivalência de "32 células com prefixos compartilhados" para "todo valor de eixo
    que a grade padrão contém", sem pagar as 4.320 correções da execução ingênua sobre a grade
    real — que é exactamente o custo que a árvore existe para tornar desnecessário.
    """
    grade = _grades_de_cobertura()[eixo]
    matriz = _matriz()

    arvore = run_multiverse(matriz, grade)
    ingenua = run_naive(matriz, grade)

    assert arvore.counter.defects_against(grade) == []
    for a, b in zip(arvore.cells, ingenua.cells, strict=True):
        assert a.spec == b.spec
        assert np.array_equal(np.asarray(a.matrix.values), np.asarray(b.matrix.values)), (
            f"eixo {eixo!r}, especificação {a.spec.fingerprint()}: árvore e ingênua divergiram"
        )


# ---------------------------------------------------------------------------
# `corrector_jobs` — o segundo eixo de paralelismo, e a condição de o ter
# ---------------------------------------------------------------------------


def test_corrector_jobs_nao_muda_o_resultado() -> None:
    """Paralelizar o corretor é escolha de TEMPO e não de resultado, e isso é afirmado.

    **A condição declarada ao parametrizar `corrector_jobs` em 31/08/2026.** O `SerrfCorrector`
    deriva o fluxo aleatório de cada unidade de `(seed, identificador da unidade)` justamente para
    que a ordem em que o `joblib` entrega as features não alcance a saída.
    Mas *"foi desenhado para"* não é *"foi verificado que"*, e a regra 3 do README
    exige a segunda.

    **Com `serrf` na grade, de propósito.** É o único corretor que recebe `n_jobs`; uma grade de
    `none` e `loess` passaria por vacuidade — o parâmetro não chegaria a lugar nenhum, e o teste
    afirmaria uma igualdade que nenhuma execução paralela produziu.
    """
    matriz = _matriz(n_amostras=16, n_features=12)
    grade = GridDefinition(
        corrections=("serrf",),
        batch_options=(1,),
        rsd_cutoffs=(None,),
        missing_thresholds=(0.8,),
        imputations=("median",),
        sample_norms=("none",),
        transformations=("none",),
        scalings=("auto",),
    )

    um = run_multiverse(matriz, grade, corrector_jobs=1)
    dois = run_multiverse(matriz, grade, corrector_jobs=2)

    chave = ("serrf", 1)
    assert chave in um.corrections, (
        "âncora: a raiz do serrf tem de existir, senão nada foi paralelizado"
    )
    np.testing.assert_array_equal(
        um.corrections[chave].matrix.values,
        dois.corrections[chave].matrix.values,
        err_msg=(
            "a matriz corrigida mudou com `corrector_jobs`. Isso é a regra 3 do README quebrada: "
            "paralelismo passou a ser escolha de RESULTADO e não de tempo."
        ),
    )
    assert len(um.cells) == len(dois.cells) == grade.size()


def test_corrector_jobs_chega_ao_corretor() -> None:
    """O par causal do teste acima: sem isto, ele passaria mesmo com o parâmetro descartado.

    `build_corrector` é o único ponto em que o valor vira comportamento. Um teste que só compare
    duas execuções não distingue *"o paralelismo não altera o resultado"* de *"o parâmetro foi
    ignorado"* — e o segundo é o defeito que a asserção de igualdade esconderia.
    """
    assert build_corrector("serrf", 1).n_jobs == 1
    assert build_corrector("serrf", 4).n_jobs == 4
    assert build_corrector("serrf").n_jobs == 1, "o padrão tem de continuar 1"
