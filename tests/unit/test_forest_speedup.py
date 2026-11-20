"""O par causal que autoriza a otimização de overhead da floresta.

**Escrito ANTES da otimização, e é ele que a autoriza.** O limite é que a floresta
ajustada continue sendo a mesma: o vão de fator 31 contra o `ranger` é overhead de
instanciação do sklearn, e fechá-lo não pode mudar árvore nenhuma — mudar o ajuste
deixa de ser implementação.

**As três asserções, e a terceira é a que impede o par de ser vácuo.**

1. Saída **bit-idêntica** com e sem o cache, mesma semente, mesma entrada.
2. O cache **está ativo** quando ligado. Sem isto, um par causal passaria com a
   flag não fazendo nada — verificação sã sobre o alvo errado.
3. O estado global do sklearn é **restaurado** ao sair, nas duas direções. Um
   `contextmanager` que instala monkeypatch e não desinstala contamina toda a
   suíte que rodar depois, e o sintoma apareceria em outro arquivo.
"""

import os

import joblib
import numpy as np
import pytest
from sklearn.base import BaseEstimator

from refino.correction._forest_speedup import cached_param_names, is_installed
from refino.correction.serrf import SerrfCorrector
from refino.domain.matrix import RefinoMatrix
from refino.domain.metadata import SampleMetadata

N_QC = 7


def _matrix(values: np.ndarray, qc_at: tuple[int, ...]) -> RefinoMatrix:
    samples = tuple(
        SampleMetadata(
            name=f"s{i:03d}",
            injection_order=i + 1,
            sample_type="qc" if i in qc_at else "sample",
            group="qc" if i in qc_at else "case",
            batch="A",
        )
        for i in range(values.shape[0])
    )
    return RefinoMatrix(
        values=np.asarray(values, dtype=np.float64),
        feature_names=tuple(f"f{j}" for j in range(values.shape[1])),
        samples=samples,
        mode="synthetic",
    )


def _realistic(n_samples: int = 30, n_features: int = 14, seed: int = 3) -> RefinoMatrix:
    """Mesma forma do helper de `test_serrf_corrector.py`: sete QCs e deriva leve."""
    rng = np.random.default_rng(seed)
    qc_at = tuple(round(i) for i in np.linspace(0, n_samples - 1, N_QC))
    drift = 1.0 + 0.004 * np.arange(n_samples)
    base = rng.uniform(1e3, 1e5, size=n_features)
    values = (
        base[None, :] * drift[:, None] * (1.0 + 0.06 * rng.standard_normal((n_samples, n_features)))
    )
    return _matrix(values, qc_at=qc_at)


def _dois_lotes(n_samples: int = 32, n_features: int = 14, seed: int = 4) -> RefinoMatrix:
    """Dois lotes com QCs em cada um, para o içamento não perder `in_batch`.

    Com **um** lote, içar para fora do laço de lotes e içar para fora do laço de
    features coincidem numericamente — é o mesmo mecanismo que fez o defeito de
    escopo do SERRF sobreviver a duas revisões da especificação.
    """
    rng = np.random.default_rng(seed)
    metade = n_samples // 2
    qc_at = tuple(round(i) for i in np.linspace(0, metade - 2, 4)) + tuple(
        round(i) for i in np.linspace(metade, n_samples - 1, 4)
    )
    drift = 1.0 + 0.004 * np.arange(n_samples)
    base = rng.uniform(1e3, 1e5, size=n_features)
    values = (
        base[None, :] * drift[:, None] * (1.0 + 0.06 * rng.standard_normal((n_samples, n_features)))
    )
    samples = tuple(
        SampleMetadata(
            name=f"s{i:03d}",
            injection_order=i + 1,
            sample_type="qc" if i in qc_at else "sample",
            group="qc" if i in qc_at else "case",
            batch="A" if i < metade else "B",
        )
        for i in range(n_samples)
    )
    return RefinoMatrix(
        values=np.asarray(values, dtype=np.float64),
        feature_names=tuple(f"f{j}" for j in range(n_features)),
        samples=samples,
        mode="synthetic",
    )


# ---------------------------------------------------------------------------
# O par causal
# ---------------------------------------------------------------------------


class _EspiaoDeCache(SerrfCorrector):
    """Afirma o estado do cache **dentro do processo que ajusta a floresta**.

    Módulo-nível de propósito: o `loky` serializa o corretor para o worker, e uma
    classe local não sobreviveria à viagem.
    """

    espera: bool = True

    def _correct_feature_uncached(self, *args: object, **kwargs: object) -> object:
        if is_installed() is not self.espera:
            raise AssertionError(
                f"cache instalado={is_installed()} no pid {os.getpid()}, "
                f"esperado {self.espera} — este é o processo onde a floresta ajusta"
            )
        return super()._correct_feature_uncached(*args, **kwargs)  # type: ignore[misc]


@pytest.mark.parametrize("n_jobs", [1, -1])
def test_o_cache_esta_ativo_no_processo_que_ajusta_a_floresta(n_jobs: int) -> None:
    """**O teste que faltava, e a ausência dele custou um número publicado.**

    A versão anterior instalava o cache em `fit_transform`, no processo pai, e
    afirmava atividade chamando `is_installed()` **no pai**. Sob `n_jobs != 1` a
    floresta ajusta dentro do `Parallel` com backend `loky`, isto é, em outro
    processo, onde o monkeypatch nunca chegou. Medido depois de suspeitar:
    `True` no pai, `False` nos quatro workers — e os três chamadores reais usam
    `n_jobs=-1`.

    O teste antigo não podia pegar isso, e não por descuido: ele exercitava o
    mecanismo e obtinha os dois desfechos. Estava **são sobre o alvo errado**, e a
    detecção é externa por construção.

    `n_jobs = -1` é o caso que importa; `n_jobs = 1` entra para que o teste
    também cubra o caminho em processo, onde o defeito nunca existiu.
    """
    matrix = _realistic()
    corretor = _EspiaoDeCache(n_jobs=n_jobs, fast_param_names=True)
    corretor.espera = True
    corretor.fit_transform(matrix, seed=42)


@pytest.mark.parametrize("n_jobs", [1, -1])
def test_o_espiao_de_cache_reprova_quando_o_cache_nao_chega(n_jobs: int) -> None:
    """**A planta, e ela é o que dá dente ao teste acima.**

    Com `fast_param_names=False` o cache não é instalado em lugar nenhum, então um
    espião que exija `is_installed()` verdadeiro **tem de** reprovar — inclusive
    dentro do worker, o que prova que a asserção atravessa a fronteira de processo
    em vez de sumir nela.

    Sem este par, o teste acima passaria com o espião inerte, que é exatamente a
    família de negativo verde que não exercita nada.
    """
    matrix = _realistic()
    corretor = _EspiaoDeCache(n_jobs=n_jobs, fast_param_names=False)
    corretor.espera = True
    with pytest.raises(AssertionError, match="cache instalado=False"):
        corretor.fit_transform(matrix, seed=42)


@pytest.mark.parametrize("n_jobs", [1, -1])
def test_a_flag_desligada_de_fato_desliga_no_processo_que_ajusta(n_jobs: int) -> None:
    """O braço desligado do par causal, afirmado onde ele importa."""
    matrix = _realistic()
    corretor = _EspiaoDeCache(n_jobs=n_jobs, fast_param_names=False)
    corretor.espera = False
    corretor.fit_transform(matrix, seed=42)


def test_a_saida_e_bit_identica_com_e_sem_o_cache() -> None:
    """**O par causal.** Duas execuções que diferem só pela otimização.

    `array_equal` e não `allclose`: o limite é bit a bit. Tolerância aqui
    admitiria uma otimização que mudasse o ajuste "pouco", e "pouco" é
    exatamente o que o espelho não permite negociar.
    """
    matrix = _realistic()

    sem = SerrfCorrector(fast_param_names=False).fit_transform(matrix, seed=42)
    com = SerrfCorrector(fast_param_names=True).fit_transform(matrix, seed=42)

    assert np.array_equal(sem.matrix.values, com.matrix.values), (
        "o cache mudou a predição: deixa de ser otimização de overhead e passa a mexer no ajuste"
    )
    assert sem.uncorrected == com.uncorrected, "as causas de não correção divergiram"
    assert sem.matrix.provenance == com.matrix.provenance, (
        "o provenance divergiu — inclui a profundidade média das árvores, então "
        "divergência aqui é divergência de floresta"
    )


def test_a_saida_e_bit_identica_serial_e_paralela() -> None:
    """**O segundo par causal, e ele é o que autoriza `n_jobs = -1` por padrão.**

    Paralelizar sobre features só é fiel porque o fluxo de semente é derivado de
    `(seed, feature, lote, passo)` e não consumido de uma semente global.
    Com semente global a saída passaria a depender da ordem em que o
    escalonador entregasse as features, e o teste reprovaria de forma
    **intermitente**, que é pior que reprovar sempre.

    A profundidade média entra no `provenance`, então compará-lo cobre também a
    preservação de ORDEM: `Parallel` devolve na ordem da entrada, e uma troca por
    entrega fora de ordem mudaria a média sem mudar valor nenhum da matriz.
    """
    matrix = _realistic()

    # **A ÂNCORA DE NÃO-VACUIDADE, e ela é de 06/09/2026.** Sob `LOKY_MAX_CPU_COUNT=1`,
    # `effective_n_jobs(-1)` devolve **1** — medido neste clone: 12 sem a variável, 1 com ela. As
    # duas linhas abaixo passariam a comparar `n_jobs=1` com `n_jobs=1`, e o teste passaria sem
    # medir coisa nenhuma.
    #
    # **E a variável é o contorno que torna a suíte completa executável aqui**, onde o arranque de
    # workers do `loky` trava a execução. O regime dispara, portanto, exactamente na configuração
    # em que o portão de facto corre.
    #
    # `skip` e não asserção: sob um núcleo a propriedade não é falsa, é **inverificável** — e a
    # ausência de referente se declara em vez de se tratar como falha.
    if joblib.effective_n_jobs(-1) <= 1:
        pytest.skip(
            "paralelismo indisponível: `effective_n_jobs(-1)` devolve 1 — com "
            "`LOKY_MAX_CPU_COUNT=1` este par causal compararia 1 contra 1 e passaria por vacuidade"
        )

    serial = SerrfCorrector(n_jobs=1).fit_transform(matrix, seed=42)
    paralela = SerrfCorrector(n_jobs=-1).fit_transform(matrix, seed=42)

    assert np.array_equal(serial.matrix.values, paralela.matrix.values), (
        "serial e paralelo divergiram: o fluxo de semente depende da ordem de "
        "execução, e isso é defeito de determinismo e não de desempenho"
    )
    assert serial.uncorrected == paralela.uncorrected
    assert serial.matrix.provenance == paralela.matrix.provenance, (
        "o provenance divergiu — ordem de acumulação das profundidades"
    )


def test_a_saida_e_bit_identica_com_e_sem_o_icamento_da_correlacao() -> None:
    """**O terceiro par causal, e ele autoriza o içamento da correlação.**

    `correlation_matrices` depende só da matriz, de `is_qc`, de `in_batch` e do alvo — nada
    do índice da feature —, e era chamada **uma vez por feature-alvo**. *A matriz é
    a de antes do passo 1.* Para 6.035
    features isso é uma Spearman de 6.035 por 6.035 recomputada 6.035 vezes, e o perfil
    com o pool completo mostra `np.cov` em 33% e `corrcoef` em 41%.

    Bit-identicidade aqui é quase tautológica — mesma função, mesmos argumentos,
    menos vezes —, e o par existe apesar disso: "quase tautológico" é o que se diz
    antes de descobrir que o laço de lotes reusava a matriz do lote anterior.

    A matriz tem **dois** lotes de propósito. Com um só, içar para fora do laço de
    lotes e içar para fora do laço de features coincidem, e o teste não distinguiria
    um içamento que perdesse a dependência de `in_batch`.
    """
    matrix = _dois_lotes()

    sem = SerrfCorrector(hoist_correlations=False).fit_transform(matrix, seed=42)
    com = SerrfCorrector(hoist_correlations=True).fit_transform(matrix, seed=42)

    assert np.array_equal(sem.matrix.values, com.matrix.values), (
        "o içamento mudou a predição: a correlação içada não é a mesma que a "
        "recomputada, e o candidato é ter perdido a dependência de `in_batch`"
    )
    assert sem.uncorrected == com.uncorrected, "as causas de não correção divergiram"
    assert sem.matrix.provenance == com.matrix.provenance


def test_o_icamento_e_bit_identico_tambem_em_execucao_parcial() -> None:
    """As duas otimizações compostas — içamento e pedido parcial — não interagem.

    Cada uma tem par causal próprio; esta afirma que compor as duas não introduz
    divergência. Sem ela, um içamento que indexasse a matriz de correlação pela
    posição no subconjunto passaria nos dois pares isolados e falharia junto.
    """
    matrix = _dois_lotes()
    pedido = frozenset(matrix.feature_names[::2])

    sem = SerrfCorrector(hoist_correlations=False).fit_transform(
        matrix, seed=42, target_features=pedido
    )
    com = SerrfCorrector(hoist_correlations=True).fit_transform(
        matrix, seed=42, target_features=pedido
    )

    assert np.array_equal(sem.matrix.values, com.matrix.values)
    assert sem.matrix.feature_names == com.matrix.feature_names


def test_o_cache_esta_ativo_quando_ligado() -> None:
    """**A asserção que impede o par causal de ser vácuo.**

    Se a flag não instalasse nada, o teste acima passaria por comparar duas
    execuções idênticas — sã sobre o alvo errado. Aqui se afirma que os dois
    braços do par de fato diferem no mecanismo.
    """
    assert not is_installed(), "o cache não pode estar instalado fora do bloco"

    with cached_param_names(enabled=True) as installed:
        assert installed is True
        assert is_installed(), "o bloco disse ter instalado e não instalou"

    with cached_param_names(enabled=False) as installed:
        assert installed is False
        assert not is_installed(), "o braço desligado instalou o cache"


def test_o_estado_global_do_sklearn_e_restaurado() -> None:
    """Monkeypatch que não desinstala contamina a suíte inteira depois dele.

    O sintoma apareceria em outro arquivo, o que o torna caro de atribuir — e a
    restauração tem de devolver o `classmethod`, não a versão ligada, senão a
    herança para as subclasses quebra em silêncio.
    """
    original = BaseEstimator.__dict__["_get_param_names"]

    with cached_param_names(enabled=True):
        pass
    assert BaseEstimator.__dict__["_get_param_names"] is original

    with pytest.raises(RuntimeError, match="planta"), cached_param_names(enabled=True):
        raise RuntimeError("planta: exceção no meio do bloco")
    assert BaseEstimator.__dict__["_get_param_names"] is original, (
        "exceção dentro do bloco deixou o monkeypatch instalado"
    )
    assert isinstance(BaseEstimator.__dict__["_get_param_names"], classmethod)


def test_o_cache_devolve_os_mesmos_nomes_que_a_introspeccao() -> None:
    """O cache tem de ser transparente, e a lista devolvida não pode ser a interna.

    Devolver o objeto cacheado compartilharia estado mutável entre chamadas: um
    consumidor que ordenasse ou truncasse a lista corromperia todas as chamadas
    seguintes, e o defeito apareceria longe daqui.
    """
    from sklearn.ensemble import RandomForestRegressor

    esperado = BaseEstimator.__dict__["_get_param_names"].__func__(RandomForestRegressor)

    with cached_param_names(enabled=True):
        primeira = RandomForestRegressor._get_param_names()
        segunda = RandomForestRegressor._get_param_names()

    assert primeira == esperado
    assert segunda == esperado
    assert primeira is not segunda, "duas chamadas devolveram o MESMO objeto lista"
