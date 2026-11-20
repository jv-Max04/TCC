"""O vetor de características da política — e o portão que foi declarado e nunca tinha sido escrito.

A docstring de `features.py` afirma que `OBSERVAVEIS` e `PROIBIDAS`
*"são conferidas por teste, nas duas direções"*. **Não eram**: o módulo tinha zero testes. É a
verificação construída e não exercitada — mecanismo declarado em decisão que nasce com a intenção e
não com o teste —, e é justamente a que sustenta *"o que não se lê não vaza"*.

A paridade é conferida nos **dois** sentidos de propósito. Uma lista de permitidos sozinha fica muda
quando alguém acrescenta campo ao `DefectProfile`; as duas juntas fazem o campo novo cair fora de
ambas e o portão reprovar.
"""

import numpy as np
import pytest

from refino.decision.features import (
    FEATURE_NAMES,
    OBSERVAVEIS,
    PROIBIDAS,
    TIPOS,
    feature_matrix,
    feature_vector,
)
from refino.domain.diagnosis import DefectProfile

CAMPOS_ESTRUTURAIS = frozenset(
    {"defect_type", "confidence", "suggested_batches", "changepoints", "raw_features"}
)


def _perfil(**kw: object) -> DefectProfile:
    base: dict[str, object] = {
        "defect_type": "erratic",
        "confidence": 0.5,
        "suggested_batches": (),
        "changepoints": (),
        "qc_rsd_median": 0.3,
        "qc_rsd_p75": 0.4,
        "zero_fraction": 0.1,
        "missing_fraction": 0.0,
        "pc1_injection_correlation": 0.2,
        "qc_dispersion_ratio": 1.0,
        "n_features": 2000,
        "n_samples": 68,
        "n_qcs": 7,
    }
    base.update(kw)
    return DefectProfile(**base)  # type: ignore[arg-type]


def test_todo_campo_do_perfil_esta_declarado_num_dos_dois_lados() -> None:
    """**A direção que pega campo NOVO.** É esta que faltava.

    Um campo acrescentado ao `DefectProfile` cai fora de `OBSERVAVEIS` e de `PROIBIDAS`, e este
    teste reprova — em vez de o campo entrar em silêncio por um dos lados.
    """
    campos = set(DefectProfile.__dataclass_fields__) - CAMPOS_ESTRUTURAIS
    declarados = set(OBSERVAVEIS) | set(PROIBIDAS)
    assert not campos - declarados, (
        f"campo do DefectProfile em nenhuma das duas listas: {sorted(campos - declarados)}. "
        "Decida se a política pode vê-lo e declare-o explicitamente."
    )


def test_nenhuma_proibida_esta_entre_as_observaveis() -> None:
    """A outra direção: as duas listas não se sobrepõem, senão a proibição seria decorativa."""
    assert not set(OBSERVAVEIS) & set(PROIBIDAS)


def test_nenhuma_proibida_aparece_no_vetor() -> None:
    """O nome de uma proibida não pode estar em `FEATURE_NAMES` — nem como prefixo de derivada."""
    for proibida in PROIBIDAS:
        assert not any(
            nome == proibida or nome.startswith(f"{proibida}_") for nome in FEATURE_NAMES
        )


def test_o_vetor_nao_muda_quando_raw_features_carrega_o_oraculo() -> None:
    """**O par causal.** A garantia é estrutural: o que não se lê não vaza.

    Plantar a resposta do oráculo em `raw_features` não pode mover uma única posição do vetor. É o
    negativo em processo da taxonomia de negativo plantado — a planta é argumento da função sob
    teste, então não há intermediário que possa não recebê-la.
    """
    limpo = _perfil()
    delator = _perfil(
        raw_features={"oracle_margin": 0.99, "oracle_label": 1.0, "best_corrector": 2.0}
    )
    np.testing.assert_array_equal(feature_vector(limpo), feature_vector(delator))


def test_n_suggested_batches_conta_lotes_e_nao_amostras() -> None:
    """**O defeito medido no corpus.** `suggested_batches` é rótulo POR AMOSTRA.

    Com `len()` a característica valia o número de amostras — constante em 68 nas 40 matrizes, e
    portanto morta. O nome promete número de lotes e agora é o que ele mede.
    """
    j = FEATURE_NAMES.index("n_suggested_batches")
    # 68 amostras em DOIS lotes: a caracteristica vale 2, nunca 68.
    dois = _perfil(suggested_batches=tuple([0] * 34 + [1] * 34))
    assert feature_vector(dois)[j] == 2.0
    tres = _perfil(suggested_batches=tuple([0] * 20 + [1] * 24 + [2] * 24))
    assert feature_vector(tres)[j] == 3.0
    assert feature_vector(_perfil(suggested_batches=tuple([0] * 68)))[j] == 1.0


def test_n_suggested_batches_separa_onde_o_comprimento_nao_separava() -> None:
    """O par causal do conserto: dois perfis que a versão antiga tornava indistinguíveis.

    Mesmo comprimento, número de lotes diferente. Sob `len()` os dois vetores eram idênticos nessa
    posição: um valor que colapsava dois casos.
    """
    j = FEATURE_NAMES.index("n_suggested_batches")
    um = feature_vector(_perfil(suggested_batches=tuple([0] * 68)))
    dois = feature_vector(_perfil(suggested_batches=tuple([0] * 34 + [1] * 34)))
    assert len(um) == len(dois)
    assert um[j] != dois[j]


def test_as_contagens_grandes_entram_em_log10_e_as_fracoes_nao() -> None:
    """`n_features` e `n_samples` em log10; fração fica crua. Sem isso a escala pesa por modo."""
    v = feature_vector(_perfil(n_features=2000, n_samples=68, n_qcs=7, zero_fraction=0.158))
    assert v[OBSERVAVEIS.index("n_features")] == pytest.approx(np.log10(2000))
    assert v[OBSERVAVEIS.index("n_samples")] == pytest.approx(np.log10(68))
    assert v[OBSERVAVEIS.index("zero_fraction")] == pytest.approx(0.158)


def test_as_indicadoras_de_tipo_somam_um_e_ficam_na_ordem_canonica() -> None:
    """Ordem canônica, não de aparição: corpus embaralhado não move coluna sob o mesmo nome."""
    for tipo in TIPOS:
        v = feature_vector(_perfil(defect_type=tipo))
        indicadoras = [v[FEATURE_NAMES.index(f"is_{t}")] for t in TIPOS]
        assert sum(indicadoras) == 1.0
        assert indicadoras[TIPOS.index(tipo)] == 1.0


def test_a_largura_do_vetor_e_a_de_feature_names() -> None:
    assert feature_vector(_perfil()).shape == (len(FEATURE_NAMES),)


def test_matriz_vazia_tem_forma_correta_e_nao_array_sem_colunas() -> None:
    """Quem consome faz `X.shape[1]`; `array([])` levantaria `IndexError` no lugar errado."""
    assert feature_matrix([]).shape == (0, len(FEATURE_NAMES))


def test_matriz_empilha_na_ordem_recebida() -> None:
    perfis = [_perfil(qc_rsd_median=0.1), _perfil(qc_rsd_median=0.9)]
    x = feature_matrix(perfis)
    assert x.shape == (2, len(FEATURE_NAMES))
    j = OBSERVAVEIS.index("qc_rsd_median")
    assert (x[0, j], x[1, j]) == (0.1, 0.9)
