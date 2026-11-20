"""Escopo estudo.

O teste central é o de blindagem: um mapa canônico não injetivo faz
`PATIENT_GROUP_CONFLICT` dar zero **por cegueira**, e o zero passaria a
significar o oposto do que parece.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from refino.domain.matrix import RefinoMatrix
from refino.domain.metadata import SampleMetadata
from refino.io.readers import load_matrix
from refino.io.validators import (
    CANONICAL_GROUPS,
    NonInjectiveCanonicalMapError,
    canonical_group,
    validate_study,
    verify_canonical_map,
)
from tests.conftest import MODES, measurements_path, sample_list_path


def _tiny(mode: str, groups: dict[str, str]) -> RefinoMatrix:
    """Matriz mínima com os pacientes e grupos dados."""
    samples = tuple(
        SampleMetadata(
            name=f"RUN_{mode}_{i + 1}_ID{pid[2:]}_{group}",
            injection_order=i + 1,
            sample_type="sample",
            group=group,
        )
        for i, (pid, group) in enumerate(sorted(groups.items()))
    )
    return RefinoMatrix(
        values=np.ones((len(samples), 2), dtype=np.float64),
        feature_names=("F1", "F2"),
        samples=samples,
        mode=mode,
    )


# ---------------------------------------------------------------------------
# Os quatro modos reais — OBSERVAÇÃO, não invariante
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_no_patient_group_conflict_in_the_four_real_modes() -> None:
    """Zero nos quatro arquivos do laboratório — e é propriedade DELES.

    Os quatro contêm os mesmos 61 pacientes e todo paciente tem o mesmo grupo nos
    quatro quando a fonte é o sufixo do nome. Não decorre do pipeline: se
    disparasse, a escolha de `prefer_name` como padrão precisaria ser reaberta.
    """
    matrices = {
        mode: load_matrix(measurements_path(mode), sample_list_path(mode), mode) for mode in MODES
    }
    conflicts = [i for i in validate_study(matrices) if i.code == "PATIENT_GROUP_CONFLICT"]
    assert not conflicts, [str(i) for i in conflicts]


@pytest.mark.integration
def test_no_spelling_inconsistency_after_load_matrix() -> None:
    """Depois da canonicalização os quatro modos grafam igual.

    Sobre os arquivos CRUS o código dispararia — `hamtsp` tem três grafias. Aqui
    não dispara porque `load_matrix` já normalizou, e é a convergência que é
    pré-requisito da análise integrada.
    """
    matrices = {
        mode: load_matrix(measurements_path(mode), sample_list_path(mode), mode) for mode in MODES
    }
    assert not [
        i for i in validate_study(matrices) if i.code == "GROUP_LABEL_INCONSISTENT_ACROSS_MODES"
    ]


# ---------------------------------------------------------------------------
# O que o pipeline por arquivo NÃO alcança
# ---------------------------------------------------------------------------


def test_patient_group_conflict_fires_on_a_synthetic_case() -> None:
    """O erro que atravessaria a ingestão inteira sem um único issue.

    `ID73` é `Asymptomatic` num modo e `ATLL` em outro, **ambos vindos do nome**.
    Nenhum validador de escopo arquivo alcança isso: a canonicalização normaliza
    grafias variantes do mesmo grupo, não funde grupos distintos, e `prefer_name`
    resolve `LABEL` contra nome DENTRO de um arquivo, nunca nome contra nome ENTRE
    arquivos.
    """
    matrices = {
        "pos_hilic": _tiny("pos_hilic", {"ID73": "Asymptomatic", "ID56": "NoHTLV"}),
        "neg_c18": _tiny("neg_c18", {"ID73": "ATLL", "ID56": "NoHTLV"}),
    }
    conflicts = [i for i in validate_study(matrices) if i.code == "PATIENT_GROUP_CONFLICT"]

    assert len(conflicts) == 1
    assert conflicts[0].location == "ID73"
    assert conflicts[0].is_error
    assert "asymptomatic" in conflicts[0].message
    assert "atll" in conflicts[0].message


def test_spelling_inconsistency_fires_across_modes() -> None:
    matrices = {
        "a": _tiny("a", {"ID71": "HAM-TSP"}),
        "b": _tiny("b", {"ID71": "HAMTSP"}),
    }
    issues = [
        i for i in validate_study(matrices) if i.code == "GROUP_LABEL_INCONSISTENT_ACROSS_MODES"
    ]
    assert len(issues) == 1
    assert issues[0].severity == "warning"


def test_a_single_matrix_cannot_produce_cross_mode_inconsistency() -> None:
    """Inconsistência ENTRE modos exige dois — um só nunca dispara."""
    single = {"a": _tiny("a", {"ID71": "HAMTSP", "ID72": "ATLL"})}
    assert not [
        i for i in validate_study(single) if i.code == "GROUP_LABEL_INCONSISTENT_ACROSS_MODES"
    ]


def test_qc_samples_do_not_participate_in_patient_conflict() -> None:
    """Um QC não é paciente: é mistura de alíquotas.

    O mesmo nome-base aparece nos quatro modos sem que isso signifique identidade
    de sujeito — tratá-lo como paciente produziria conflito onde não há.
    """
    qc = SampleMetadata("RUN_QC1", 1, "qc", "QC")
    matrices = {
        mode: RefinoMatrix(
            values=np.ones((1, 2), dtype=np.float64),
            feature_names=("F1", "F2"),
            samples=(qc,),
            mode=mode,
        )
        for mode in ("a", "b")
    }
    assert validate_study(matrices) == []


def test_validate_study_never_raises_and_accepts_an_empty_map() -> None:
    assert validate_study({}) == []


# ---------------------------------------------------------------------------
# A blindagem: o único jeito de cegar PATIENT_GROUP_CONFLICT
# ---------------------------------------------------------------------------


def test_the_default_canonical_map_is_injective() -> None:
    """Os quatro grupos da especificação canonicalizam em quatro destinos distintos."""
    verify_canonical_map(CANONICAL_GROUPS)
    assert len({canonical_group(g) for g in CANONICAL_GROUPS}) == len(CANONICAL_GROUPS)


def test_a_non_injective_map_is_rejected() -> None:
    """Dois grupos distintos no mesmo destino é configuração inválida."""
    with pytest.raises(NonInjectiveCanonicalMapError, match="não é injetivo"):
        verify_canonical_map(frozenset({"HAM-TSP", "HAMTSP", "ATLL"}))


def test_validate_study_refuses_to_run_with_a_blinding_map() -> None:
    """A verificação roda ANTES de qualquer detecção.

    Sem ela, o mapa cego produziria zero conflitos e o zero seria indistinguível
    de ausência de conflito — que é o modo de falha exato.
    """
    matrices = {
        "a": _tiny("a", {"ID71": "HAMTSP"}),
        "b": _tiny("b", {"ID71": "HAM-TSP"}),
    }
    with pytest.raises(NonInjectiveCanonicalMapError):
        validate_study(matrices, canonical_groups=frozenset({"HAMTSP", "HAM/TSP"}))


def test_a_blinding_map_would_hide_a_real_conflict() -> None:
    """Demonstra a cegueira que a blindagem impede.

    Com o mapa cego desativado, `HAMTSP` e `HAM-TSP` — que seriam grupos
    distintos numa configuração assim — colapsam, e o conflito real desaparece.
    Este teste prova que o risco é concreto e não hipotético.
    """
    blinding = frozenset({"HAMTSP", "HAM-TSP"})
    assert len({canonical_group(g) for g in blinding}) == 1, (
        "os dois colapsam — é exatamente o que torna o mapa cego"
    )
    with pytest.raises(NonInjectiveCanonicalMapError):
        verify_canonical_map(blinding)


def test_validate_structure_also_guards_the_map() -> None:
    """A blindagem vale nos dois escopos, não só no estudo."""
    from refino.io.documents import MeasurementsDocument
    from refino.io.sniffer import Dialect
    from refino.io.validators import validate_structure

    doc = MeasurementsDocument(
        values=np.ones((1, 1), dtype=np.float64),
        feature_names=("F1",),
        sample_names=("S1",),
        group_labels=("QC",),
        raw_cells=(("1",),),
        dialect=Dialect(",", ".", None, "utf-8", "\r\n"),
    )
    with pytest.raises(NonInjectiveCanonicalMapError):
        validate_structure(doc, canonical_groups=frozenset({"HAMTSP", "HAM-TSP"}))


def test_conflict_survives_dataclass_replace_of_groups() -> None:
    """Sanidade: o detector olha `sample.group`, não o nome.

    Se olhasse o nome, `prefer_label` esconderia conflitos — porque o nome
    continuaria dizendo uma coisa e o grupo resolvido outra.
    """
    base = _tiny("a", {"ID71": "ATLL"})
    changed = dataclasses.replace(
        base, samples=(dataclasses.replace(base.samples[0], group="NoHTLV"),)
    )
    conflicts = [
        i for i in validate_study({"a": base, "b": changed}) if i.code == "PATIENT_GROUP_CONFLICT"
    ]
    assert len(conflicts) == 1
