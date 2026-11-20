"""`cache.py` e o critério — e o PAR CAUSAL da chave insegura.

**O teste que importa aqui não é "o cache devolve o que guardou".** É que a chave **não** pode ser
só o `fingerprint` da especificação: a mesma especificação sobre matrizes diferentes tem o mesmo
`fingerprint` e resultados sem relação, e o critério existe porque "reexecuções de grades
sobrepostas reaproveitam" — grades sobrepostas **entre modos** é exactamente o caso que colide.

Um teste que só guardasse e lesse com uma matriz passaria com a chave errada. O par causal é o que
distingue: duas matrizes, uma especificação, e a afirmação de que as chaves **divergem**.
"""

import json

import numpy as np
import pytest

from refino.domain.matrix import RefinoMatrix
from refino.domain.metadata import SampleMetadata
from refino.domain.spec import PipelineSpec
from refino.multiverse.cache import CellCache, cache_key, matrix_digest

pytestmark = pytest.mark.property


def _matriz(semente: int = 3, n_features: int = 6, deslocamento: float = 0.0) -> RefinoMatrix:
    gerador = np.random.default_rng(semente)
    valores = gerador.lognormal(10.0, 0.4, size=(12, n_features)) + deslocamento
    amostras = tuple(
        SampleMetadata(
            name=f"S{i:03d}",
            injection_order=i + 1,
            sample_type="qc" if i % 4 == 0 else "sample",
            group="QC" if i % 4 == 0 else ("A" if i % 2 else "B"),
            batch="A",
        )
        for i in range(12)
    )
    return RefinoMatrix(
        values=valores,
        feature_names=tuple(f"f{j:03d}" for j in range(n_features)),
        samples=amostras,
        mode="sintetico",
    )


def _spec(**troca: object) -> PipelineSpec:
    campos: dict[str, object] = {
        "correction": "none",
        "n_batches": 1,
        "ranking_filter": "none",
        "ranking_keep_fraction": None,
        "ranking_filter_2": "none",
        "ranking_keep_fraction_2": None,
        "rsd_variant": "outlier_removed",
        "rsd_cutoff": None,
        "missing_threshold": 0.8,
        "imputation": "one_fifth_min",
        "sample_normalization": "none",
        "normalization_reference": None,
        "normalization_factors": None,
        "transformation": "log10",
        "scaling": "auto",
    }
    campos.update(troca)
    return PipelineSpec(**campos)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# O par causal: a chave só com `fingerprint` colidiria
# ---------------------------------------------------------------------------


def test_a_mesma_especificacao_sobre_matrizes_diferentes_tem_chaves_diferentes() -> None:
    """**O par causal do critério, e é o teste que a chave errada reprovaria.**

    As duas metades importam. `fingerprint` **igual** é a premissa — sem ela o teste passaria por
    não haver colisão a evitar, que é a forma inerte do teste. `cache_key` **diferente** é
    a conclusão.
    """
    spec = _spec()
    a, b = _matriz(semente=3), _matriz(semente=99)

    assert spec.fingerprint() == spec.fingerprint(), "premissa: o fingerprint é da especificação"
    assert not np.array_equal(np.asarray(a.values), np.asarray(b.values)), (
        "premissa do par: as duas matrizes têm de diferir, senão a chave igual estaria certa"
    )
    assert cache_key(spec, a) != cache_key(spec, b), (
        "a mesma especificação sobre matrizes diferentes produziu a MESMA chave. O cache serviria "
        "o resultado de um modo como se fosse de outro, e nada acusaria"
    )


def test_a_mesma_matriz_e_a_mesma_especificacao_dao_a_mesma_chave() -> None:
    """A outra direção: sem isto a chave nunca acertaria e o cache seria miss permanente.

    É a segunda direção que a verificação exige — um mecanismo que **sempre** reprova some do
    agregado igual a um que nunca reprova.
    """
    spec = _spec()
    assert cache_key(spec, _matriz()) == cache_key(spec, _matriz())


def test_o_digest_muda_com_valores_com_nomes_de_feature_e_com_metadado_de_amostra() -> None:
    """Os três entram no digest, e cada um por um motivo diferente do outro.

    Valores é o óbvio. **Nomes de feature** mudam o denominador do BH; **grupo da amostra** muda
    quem é comparado com quem. Uma chave que ignorasse qualquer um dos dois últimos serviria
    resultado calculado sobre outra pergunta.
    """
    base = _matriz()
    valores = np.asarray(base.values).copy()
    valores[0, 0] *= 1.0000001
    outro_valor = RefinoMatrix(
        values=valores, feature_names=base.feature_names, samples=base.samples, mode=base.mode
    )
    outro_nome = RefinoMatrix(
        values=base.values,
        feature_names=("ZZZ", *base.feature_names[1:]),
        samples=base.samples,
        mode=base.mode,
    )
    import dataclasses

    trocadas = (dataclasses.replace(base.samples[1], group="Z"), *base.samples[2:])
    outro_grupo = RefinoMatrix(
        values=base.values,
        feature_names=base.feature_names,
        samples=(base.samples[0], *trocadas),
        mode=base.mode,
    )
    digests = {
        matrix_digest(base),
        matrix_digest(outro_valor),
        matrix_digest(outro_nome),
        matrix_digest(outro_grupo),
    }
    assert len(digests) == 4, (
        "duas das quatro matrizes distinguíveis colidiram no digest. Quais colidem diz qual das "
        "três componentes não entrou no hash"
    )


# ---------------------------------------------------------------------------
# `get` e `put`
# ---------------------------------------------------------------------------


def test_hit_devolve_bit_identico_e_miss_devolve_none(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Guardar e ler não altera nem um bit, e o que não foi guardado é `None`."""
    cache = CellCache(root=tmp_path / "celulas")
    spec, matriz = _spec(), _matriz()
    valores = np.asarray(matriz.values, dtype=np.float64) * 1.7

    assert cache.get(spec, matriz) is None, "miss antes de gravar"
    cache.put(spec, matriz, valores)
    lido = cache.get(spec, matriz)
    assert lido is not None
    assert np.array_equal(lido, valores), "o cache alterou os valores entre gravar e ler"

    # A chave carrega a matriz: outra matriz sob a mesma especificacao continua miss.
    assert cache.get(spec, _matriz(semente=99)) is None
    # E outra especificacao sobre a mesma matriz tambem.
    assert cache.get(_spec(scaling="pareto"), matriz) is None


def test_arquivo_corrompido_e_miss_e_nao_erro(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """**A política declarada, exercitada.** Leitura que falha recomputa; nunca levanta.

    O negativo é em processo: a corrupção é escrita no arquivo que `get` vai abrir, então não há
    intermediário que possa não recebê-la.
    """
    cache = CellCache(root=tmp_path / "celulas")
    spec, matriz = _spec(), _matriz()
    cache.put(spec, matriz, np.asarray(matriz.values, dtype=np.float64))
    assert cache.get(spec, matriz) is not None, "premissa: era hit antes de corromper"

    npy = tmp_path / "celulas" / f"{cache_key(spec, matriz)}.npy"
    npy.write_bytes(b"nao sou um npy")
    assert cache.get(spec, matriz) is None, "arquivo truncado tinha de virar miss, não exceção"


def test_metadado_com_chave_divergente_e_miss(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """A chave por extenso no metadado é conferida — nome de arquivo truncado não decide sozinho."""
    cache = CellCache(root=tmp_path / "celulas")
    spec, matriz = _spec(), _matriz()
    cache.put(spec, matriz, np.asarray(matriz.values, dtype=np.float64))
    meta = tmp_path / "celulas" / f"{cache_key(spec, matriz)}.json"
    meta.write_text(json.dumps({"key": "outra-coisa"}), encoding="utf-8")
    assert cache.get(spec, matriz) is None


def test_a_escrita_nao_deixa_temporario_para_tras(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Atomicidade por renomeação: o `.tmp` não sobrevive à gravação bem-sucedida."""
    cache = CellCache(root=tmp_path / "celulas")
    cache.put(_spec(), _matriz(), np.asarray(_matriz().values, dtype=np.float64))
    assert list((tmp_path / "celulas").glob("*.tmp.npy")) == []
