"""`CorrectionResult`: a marcação por feature e a exaustividade dela.

Três grupos, e o segundo é o que dá dente ao primeiro.

A construção guarda três coisas (rótulo válido, feature existente, causas
disjuntas) porque o mapa pode chegar desserializado do `metrics_json`, onde nada
passou pelo mypy — é a guarda em runtime aplicada a uma entidade em vez de a um rótulo.

A exaustividade tem **violação plantada em cada uma das três pontas**. Sem isso as
três asserções passam vacuamente sobre um resultado que nunca marca nada:
verificação decorativa indistinguível de verificação ativa.
"""

import numpy as np
import pytest

from refino.domain.correction import CorrectionResult
from refino.domain.labels import InvalidLabelError
from refino.domain.matrix import RefinoMatrix
from refino.domain.metadata import SampleMetadata


def _samples(n: int = 4) -> tuple[SampleMetadata, ...]:
    return tuple(
        SampleMetadata(
            name=f"S{i + 1}",
            injection_order=i + 1,
            sample_type="qc" if i in (0, 3) else "sample",
            group="QC" if i in (0, 3) else "NoHTLV",
        )
        for i in range(n)
    )


def _matrix(n_features: int = 4) -> RefinoMatrix:
    values = np.arange(4 * n_features, dtype=np.float64).reshape(4, n_features)
    return RefinoMatrix(
        values=values,
        feature_names=tuple(f"F{i + 1}" for i in range(n_features)),
        samples=_samples(),
        mode="tiny",
    )


# ---------------------------------------------------------------------------
# A forma do resultado
# ---------------------------------------------------------------------------


def test_mapa_vazio_e_o_caso_do_corretor_none() -> None:
    """Vazio, não a matriz inteira e não indefinido.

    É a asserção com dente: `none` não corrige por desenho, então o
    conjunto de "tentou e desistiu" é vazio. Uma implementação que marcasse todas as
    features passaria em tudo o mais e dominaria a contagem por causa do multiverso.
    """
    resultado = CorrectionResult(matrix=_matrix())
    assert resultado.uncorrected == {}
    assert not resultado.uncorrected_mask.any()
    assert resultado.counts_by_cause() == ()


def test_mascara_marca_exatamente_a_uniao_das_causas() -> None:
    resultado = CorrectionResult(
        matrix=_matrix(),
        uncorrected={
            "negative_fitted_curve": frozenset({"F1", "F3"}),
            "fit_failed": frozenset({"F4"}),
        },
    )
    assert resultado.uncorrected_mask.tolist() == [True, False, True, True]


def test_mascara_esta_alinhada_a_ordem_das_features_e_nao_a_ordem_do_mapa() -> None:
    """Ordem do `dict` não pode decidir posição na máscara.

    O modo de falha é silencioso: com features nomeadas em ordem diferente da do
    mapa, uma implementação que iterasse o mapa produziria máscara plausível e
    deslocada, e nenhum teste de contagem pegaria.
    """
    resultado = CorrectionResult(
        matrix=_matrix(),
        uncorrected={"fit_failed": frozenset({"F4"}), "null_denominator": frozenset({"F2"})},
    )
    assert resultado.uncorrected_mask.tolist() == [False, True, False, True]


def test_contagem_por_causa_e_ordenada_por_nome_e_nao_por_insercao() -> None:
    """Duas construções com a mesma informação serializam igual.

    Sem isso, dois relatórios idênticos gerariam `metrics_json` diferentes e a
    deduplicação do multiverso os trataria como distintos.
    """
    a = CorrectionResult(
        matrix=_matrix(),
        uncorrected={"null_denominator": frozenset({"F1"}), "fit_failed": frozenset({"F2"})},
    )
    b = CorrectionResult(
        matrix=_matrix(),
        uncorrected={"fit_failed": frozenset({"F2"}), "null_denominator": frozenset({"F1"})},
    )
    esperado = (("fit_failed", 1), ("null_denominator", 1))
    assert a.counts_by_cause() == b.counts_by_cause() == esperado


def test_contagem_por_causa_nao_devolve_total() -> None:
    """Causas dizem coisas diferentes, e um total convida a ignorá-las."""
    resultado = CorrectionResult(
        matrix=_matrix(),
        uncorrected={
            "no_usable_predictors": frozenset({"F1"}),
            "negative_fitted_curve": frozenset({"F2", "F3"}),
        },
    )
    chaves = {nome for nome, _ in resultado.counts_by_cause()}
    assert chaves == {"no_usable_predictors", "negative_fitted_curve"}
    assert "total" not in chaves
    assert len(resultado.counts_by_cause()) == 2


def test_digest_distingue_conjuntos_de_mesmo_tamanho() -> None:
    """Contagem igual não é conjunto igual."""
    base = _matrix()
    a = CorrectionResult(matrix=base, uncorrected={"fit_failed": frozenset({"F1", "F2"})})
    b = CorrectionResult(matrix=base, uncorrected={"fit_failed": frozenset({"F3", "F4"})})
    assert a.counts_by_cause() == b.counts_by_cause()
    assert a.uncorrected_digest() != b.uncorrected_digest()


def test_digest_nao_depende_da_causa_que_marcou() -> None:
    """O digest responde QUAIS features, e a causa viaja na contagem."""
    base = _matrix()
    a = CorrectionResult(matrix=base, uncorrected={"fit_failed": frozenset({"F1"})})
    b = CorrectionResult(matrix=base, uncorrected={"null_denominator": frozenset({"F1"})})
    assert a.uncorrected_digest() == b.uncorrected_digest()
    assert a.counts_by_cause() != b.counts_by_cause()


# ---------------------------------------------------------------------------
# Exaustividade — as três pontas, cada uma com violação plantada
# ---------------------------------------------------------------------------


def test_causa_fora_do_vocabulario_reprova() -> None:
    """Primeira ponta: todo nome de causa é membro de `UncorrectedCause`."""
    with pytest.raises(InvalidLabelError, match="no_usable_predictors"):
        CorrectionResult(
            matrix=_matrix(),
            uncorrected={"loess_falhou": frozenset({"F1"})},  # type: ignore[dict-item]
        )


def test_causa_de_grafia_plausivel_tambem_reprova() -> None:
    """A guarda pega o caso insidioso, não só o absurdo.

    `negative_curve` é o que alguém escreveria de memória, e cair num `else`
    razoável faria a feature ser marcada com causa que não existe.
    """
    with pytest.raises(InvalidLabelError):
        CorrectionResult(
            matrix=_matrix(),
            uncorrected={"negative_curve": frozenset({"F1"})},  # type: ignore[dict-item]
        )


def test_feature_marcada_em_duas_causas_reprova() -> None:
    """Segunda ponta: as causas são disjuntas — exatamente uma causa por feature.

    É a asserção que a forma causa -> features torna verificável. Na forma inversa
    a unicidade seria garantia do `dict` e nenhum teste poderia plantar isto.
    """
    with pytest.raises(ValueError, match=r"marcada em duas causas"):
        CorrectionResult(
            matrix=_matrix(),
            uncorrected={
                "fit_failed": frozenset({"F1", "F2"}),
                "negative_fitted_curve": frozenset({"F2"}),
            },
        )


def test_uniao_das_causas_e_a_mascara_por_construcao() -> None:
    """Terceira ponta: união igual à máscara.

    Aqui a igualdade é estrutural — a máscara é derivada do mapa —, e o teste
    afirma justamente isso: não há segunda representação para divergir. A ponta que
    poderia falhar é a de cima, e é onde a violação foi plantada.
    """
    resultado = CorrectionResult(
        matrix=_matrix(),
        uncorrected={"fit_failed": frozenset({"F1"}), "null_denominator": frozenset({"F3"})},
    )
    uniao = {nome for features in resultado.uncorrected.values() for nome in features}
    marcadas = {
        nome
        for nome, marcada in zip(
            resultado.matrix.feature_names, resultado.uncorrected_mask, strict=True
        )
        if marcada
    }
    assert uniao == marcadas


def test_feature_inexistente_reprova() -> None:
    """Marcação que aponta para fora do espaço de features não é auditável.

    E o digest a esconderia: ele resume nomes, então um nome fantasma entraria no
    resumo sem que nada acusasse.
    """
    with pytest.raises(ValueError, match=r"não existe na matriz"):
        CorrectionResult(matrix=_matrix(), uncorrected={"fit_failed": frozenset({"F99"})})


def test_conjunto_vazio_numa_causa_e_aceito_e_conta_zero() -> None:
    """Causa declarada sem feature não é erro: é a causa que não disparou.

    Reprovar aqui obrigaria o corretor a omitir a chave, e omissão e zero são
    informações diferentes.
    """
    resultado = CorrectionResult(matrix=_matrix(), uncorrected={"fit_failed": frozenset()})
    assert resultado.counts_by_cause() == (("fit_failed", 0),)
    assert not resultado.uncorrected_mask.any()


# ---------------------------------------------------------------------------
# Imutabilidade e camada
# ---------------------------------------------------------------------------


def test_resultado_e_congelado() -> None:
    resultado = CorrectionResult(matrix=_matrix())
    with pytest.raises(AttributeError):
        resultado.matrix = _matrix()  # type: ignore[misc]


def test_dominio_da_correcao_nao_importa_as_camadas_de_fora() -> None:
    """`domain/correction.py` existe para que `metrics/` e `correction/` conversem.

    Se ele importasse `correction`, o ciclo de importação voltaria pela porta do
    tipo de retorno.
    """
    from pathlib import Path

    texto = (
        Path(__file__).resolve().parents[2] / "src" / "refino" / "domain" / "correction.py"
    ).read_text(encoding="utf-8")
    for proibido in ("refino.metrics", "refino.correction", "refino.preprocessing", "refino.io"):
        assert proibido not in texto, f"inversão de camada da §5.1: importa {proibido}"
