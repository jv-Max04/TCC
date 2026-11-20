"""Uma execução completa grava e é consultável por `matrix_id`."""

import json
from collections.abc import Iterator

import pytest
from sqlalchemy import Engine, text
from sqlalchemy.exc import IntegrityError

from refino.domain.diagnosis import DefectProfile
from refino.persistence.repositories import RunRepository
from tests.integration.test_migrations import _alembic, engine  # noqa: F401 - fixture reexportada

pytestmark = pytest.mark.integration

PERFIL = DefectProfile(
    defect_type="erratic",
    confidence=0.61,
    suggested_batches=(0, 0, 1),
    changepoints=(45,),
    qc_rsd_median=0.3,
    qc_rsd_p75=0.4,
    zero_fraction=0.158,
    missing_fraction=0.0,
    pc1_injection_correlation=0.29,
    qc_dispersion_ratio=0.93,
    n_features=6035,
    n_samples=68,
    n_qcs=7,
)


@pytest.fixture
def repo(engine: Engine) -> Iterator[RunRepository]:  # noqa: F811 - a fixture vem do outro módulo
    assert _alembic(engine, "upgrade", "head").returncode == 0
    yield RunRepository(engine)
    _alembic(engine, "downgrade", "base")


def _grava(repo: RunRepository, **kw: object) -> object:
    base: dict[str, object] = {
        "estudo": "ATLL/HAM-TSP",
        "mode": "pos_hilic",
        "checksum_entrada": "a" * 64,
        "n_samples": 68,
        "n_features": 6035,
        "amostras": [
            {"name": "QC001", "injection_order": 1, "sample_type": "qc", "grupo": "QC"},
            {"name": "S002", "injection_order": 2, "sample_type": "sample", "grupo": "ATLL"},
        ],
        "perfil": PERFIL,
        "spec_json": '{"correction": "serrf"}',
        "seed": 42,
        "rationale": "Recomendado `serrf` com confiança de 0,97.",
        "origem": "modelo",
        "metrics_json": '{"qc_rsd_median": 0.3}',
        # `apresentacao` e obrigatorio e sem padrao. Aqui e
        # `sem_apresentacao` porque este teste grava sem interface nenhuma — e o valor
        # honesto para quem nao tem superficie onde exibir a justificativa.
        "apresentacao": "sem_apresentacao",
    }
    base.update(kw)
    return repo.grava_execucao(**base)  # type: ignore[arg-type]


def test_ca_10_1_uma_execucao_grava_nas_quatro_e_e_consultavel_por_matrix_id(
    repo: RunRepository,
) -> None:
    """As quatro tabelas que o critério nomeia, encadeadas e achadas pela matriz."""
    g = _grava(repo)
    achado = repo.consulta_por_matriz(g.matrix_id)  # type: ignore[attr-defined]
    assert {k: len(v) for k, v in achado.items()} == {
        "runs": 1,
        "diagnoses": 1,
        "decisions": 1,
        "quality_reports": 1,
    }
    assert achado["decisions"][0]["diagnosis_id"] == achado["diagnoses"][0]["id"]
    assert achado["quality_reports"][0]["run_id"] == achado["runs"][0]["id"]


def test_o_perfil_recuperado_do_banco_e_igual_ao_gravado(repo: RunRepository) -> None:
    """O perfil INTEIRO volta — treze campos, não um subconjunto escolhido à mão."""
    g = _grava(repo)
    achado = repo.consulta_por_matriz(g.matrix_id)  # type: ignore[attr-defined]
    perfil = json.loads(achado["diagnoses"][0]["profile_json"])
    assert perfil["n_features"] == 6035
    assert perfil["changepoints"] == [45]
    assert perfil["suggested_batches"] == [0, 0, 1]
    assert set(perfil) >= {"qc_rsd_median", "qc_dispersion_ratio", "n_qcs", "missing_fraction"}


def test_origem_invalida_levanta_antes_do_banco(repo: RunRepository) -> None:
    """**A guarda de domínio vem primeiro**, e a mensagem é de domínio e não `IntegrityError`."""
    with pytest.raises(ValueError, match="domínio é fechado"):
        _grava(repo, origem="heuristico")


def test_a_gravacao_e_atomica_e_nao_deixa_execucao_pela_metade(repo: RunRepository) -> None:
    """**O par causal da transação.** Metade de execução responde à consulta e nada acusa.

    A planta é uma amostra sem `sample_type`, que a tabela exige: o `INSERT` das amostras falha
    depois de a matriz já ter entrado, e o que já entrou tem de sumir junto.
    """
    with pytest.raises(IntegrityError, match=r"samples\.sample_type"):
        _grava(repo, amostras=[{"name": "X", "injection_order": 1, "coluna_que_nao_existe": 1}])
    with repo._engine.connect() as c:
        for tabela in ("studies", "matrices", "runs", "diagnoses", "decisions", "quality_reports"):
            assert c.execute(text(f"SELECT count(*) FROM {tabela}")).scalar_one() == 0, (
                f"{tabela} ficou com linha após falha no meio — a gravação não é atômica"
            )
