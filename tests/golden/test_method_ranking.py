"""Comparação de métodos sob estimador UNIFORME, por fração pareada.

**A versão anterior deste arquivo cobrava o irreproduzível.** Ela
comparava o ranking do `qc_rsd_r_cv` contra a linha da tabela 4.4, cujas três células
vêm de **dois** estimadores: `none` é in-sample sobre sete QCs (`tools.R:505`), e
`SERRF` e `batch_loess` são subamostragem 5/2 sobre **dois** retidos (linhas 886-977 e
1246-1277). A legenda da tabela diz isso. Nenhuma implementação de estimador
uniforme reproduz aquela ordem, então sete dos treze testes reprovavam por construção -
e quando nenhuma implementação correta reproduz o número, o defeito é do critério.

**O que mudou.** O critério passa a cobrar **fração pareada** de features em
que o SERRF supera o `none`, sob `leave_one_qc_out_rsd`: sete pontos fora da amostra, um
por QC, com a matriz inteira presente no ajuste. A comparação contra o R desce a
**observação** e nunca é portão.

**Por que fração pareada e não mediana, e é poder e não gosto.** As diferenças de
mediana entre SERRF e `none` são -2,33, -2,09, -0,82 e +0,07 pontos percentuais, contra
erros padrão da mediana amostral em `n = 500` de 2,73, 1,62, 2,69 e 6,45: só um
modo separa. A mediana comprime centenas de features num número e descarta o
pareamento; a fração usa as duas coisas.

**Custo, declarado porque decide o marcador.** São sete ajustes por corretor e por modo
para o estimador de sete pontos, mais um por corretor para o in-sample. Medido em doze
núcleos contendidos, só a parte do SERRF: 256 s no `pos_hilic`, 226 s no `pos_c18`,
123 s no `neg_hilic`, 116 s no `neg_c18`. `slow`, e vive na metade lenta do noturno.
"""

import json
from typing import Any

import numpy as np
import pytest

from refino.correction.loess import LoessCorrector
from refino.correction.none import NoneCorrector
from refino.correction.serrf import SerrfCorrector
from refino.io.readers import load_matrix
from refino.metrics.crossval import leave_one_qc_out_rsd, sampled_feature_names
from refino.metrics.rsd import rsd
from tests.conftest import REPO, measurements_path, sample_list_path

pytestmark = [pytest.mark.golden, pytest.mark.slow]

MODES = ("pos_hilic", "pos_c18", "neg_hilic", "neg_c18")
VARIANT = "outlier_removed"
SEED = 42
N_AMOSTRADAS = 500

CORRETORES = {
    "none": NoneCorrector,
    "batch_loess": LoessCorrector,
    "SERRF": lambda: SerrfCorrector(n_jobs=-1),
}

# **Os três modos em que corrigir ajuda, e o quarto que é cara ou coroa.** A separação
# é do achado 4.5 e não nossa: no `neg_c18` o dado já nasce com 15,5% de RSD e corrigir
# tem tanta chance de piorar quanto de melhorar. Exigir maioria nos quatro contradiria
# o achado - que é exactamente o defeito que a v1.1 deste critério tinha.
MODOS_EM_QUE_CORRIGIR_AJUDA = ("pos_hilic", "pos_c18", "neg_hilic")
MODO_CARA_OU_COROA = "neg_c18"
FAIXA_CARA_OU_COROA = (45.0, 60.0)

# **Observação, NUNCA portão**. A especificação mede a mesma quantidade pelo estimador do
# R sobre as matrizes golden dele. Serve de concordância independente e é persistida no
# relatório; nenhuma asserção a compara, porque instrumento diferente sobre população
# diferente não arbitra o nosso número.
FRACAO_SECAO_4_5 = {"pos_hilic": 82.0, "pos_c18": 85.7, "neg_hilic": 62.0, "neg_c18": 49.5}

# **Contagem observada nestes quatro arquivos**. O `batch_loess` fica pior que não
# corrigir em três dos quatro sob o estimador honesto, e melhor só no `neg_hilic`. Num
# arquivo novo pode ser outra contagem, e isso é achado sobre o dado.
LOESS_PIOR_QUE_NONE = ("pos_hilic", "pos_c18", "neg_c18")

_CACHE: dict[str, dict[str, Any]] = {}


def _medidas(mode: str) -> dict[str, Any]:
    """Os dois estimadores sobre o MESMO subconjunto amostrado, resolvidos uma vez.

    O in-sample entra porque a segunda metade do critério é a **discordância** entre
    os dois, e ela é a tese central do projeto: comparar métodos por RSD in-sample é
    inválido, não frouxo, porque o LOESS o zera por construção (achado 4.3).
    """
    if mode in _CACHE:
        return _CACHE[mode]

    matrix = load_matrix(measurements_path(mode), sample_list_path(mode), mode)
    amostradas = sampled_feature_names(matrix, N_AMOSTRADAS, SEED)
    alvo = frozenset(amostradas) if amostradas is not None else None

    fora_da_amostra: dict[str, np.ndarray] = {}
    in_sample: dict[str, float] = {}
    for nome, construir in CORRETORES.items():
        fora_da_amostra[nome] = leave_one_qc_out_rsd(
            matrix, construir(), variant=VARIANT, seed=SEED, n_sampled_features=N_AMOSTRADAS
        )
        corrigida = construir().fit_transform(matrix, seed=SEED, target_features=alvo)
        vetor = rsd(
            np.asarray(corrigida.matrix.values)[corrigida.matrix.qc_mask], axis=0, variant=VARIANT
        )
        in_sample[nome] = float(np.median(vetor[np.isfinite(vetor)]) * 100.0)

    def _fracao_pareada(metodo: str) -> tuple[float, int]:
        """Fração de features em que `metodo` bate o `none`, sobre as definidas nos DOIS.

        O pareamento é o ponto: comparar sobre conjuntos diferentes de features
        responderia outra pergunta, e a política de indefinido obriga a
        declarar o que se faz com `nan` - aqui, exigir definido nos dois lados e
        reportar o denominador junto.
        """
        a, b = fora_da_amostra[metodo], fora_da_amostra["none"]
        definidas = np.isfinite(a) & np.isfinite(b)
        return float(np.mean(a[definidas] < b[definidas]) * 100.0), int(definidas.sum())

    fracao = {metodo: _fracao_pareada(metodo) for metodo in ("SERRF", "batch_loess")}
    medianas = {nome: float(np.nanmedian(vetor) * 100.0) for nome, vetor in fora_da_amostra.items()}

    _CACHE[mode] = {
        "mode": mode,
        "n_amostradas": N_AMOSTRADAS,
        "semente": SEED,
        "variante": VARIANT,
        "fracao_bate_none_pct": {metodo: valor for metodo, (valor, _) in fracao.items()},
        "n_features_pareadas": {metodo: n for metodo, (_, n) in fracao.items()},
        "mediana_fora_da_amostra_pct": medianas,
        "mediana_in_sample_pct": in_sample,
        "ranking_fora_da_amostra": tuple(sorted(medianas, key=lambda k: medianas[k])),
        "ranking_in_sample": tuple(sorted(in_sample, key=lambda k: in_sample[k])),
        "fracao_secao_4_5_pct_OBSERVACAO": FRACAO_SECAO_4_5[mode],
    }
    return _CACHE[mode]


@pytest.mark.parametrize("mode", MODOS_EM_QUE_CORRIGIR_AJUDA)
def test_ca_3_3_serrf_supera_none_na_maioria_das_features(mode: str) -> None:
    """Nos três modos em que corrigir ajuda, o SERRF ganha na maioria das features."""
    dados = _medidas(mode)
    fracao = dados["fracao_bate_none_pct"]["SERRF"]
    n = dados["n_features_pareadas"]["SERRF"]
    assert fracao > 50.0, (
        f"{mode}: SERRF supera `none` em {fracao:.1f}% de {n} features pareadas, "
        f"e o critério pede maioria. Medianas fora da amostra: "
        f"{dados['mediana_fora_da_amostra_pct']}"
    )


def test_ca_3_3_no_neg_c18_corrigir_e_cara_ou_coroa() -> None:
    """**A segunda metade obrigatória**, e é ela que dá dente ao critério.

    Sem esta asserção, "o SERRF ganha na maioria" passaria também se ele ganhasse
    onde o achado 4.5 diz que não deveria - e o critério premiaria correção
    indiscriminada. A faixa é declarada com folga: a especificação mede 49,5% pelo estimador
    do R e nós medimos 54,5%, um de cada lado de 50%.
    """
    dados = _medidas(MODO_CARA_OU_COROA)
    fracao = dados["fracao_bate_none_pct"]["SERRF"]
    baixo, alto = FAIXA_CARA_OU_COROA
    assert baixo <= fracao <= alto, (
        f"{MODO_CARA_OU_COROA}: SERRF supera `none` em {fracao:.1f}% das features, fora da "
        f"faixa de cara ou coroa [{baixo}, {alto}]. O achado 4.5 diz que aqui corrigir tem "
        "tanta chance de piorar quanto de melhorar; sair da faixa é achado sobre o dado, "
        "não defeito de código - investigar antes de mover a faixa."
    )


@pytest.mark.parametrize("mode", MODES)
def test_ca_3_3_o_insample_poe_batch_loess_primeiro_nos_quatro(mode: str) -> None:
    """`batch_loess < SERRF < none` in-sample, nos quatro - e é OBSERVAÇÃO.

    A especificação obriga o rótulo: é contagem sobre os quatro
    arquivos do laboratório, não invariante do método.
    """
    dados = _medidas(mode)
    assert dados["ranking_in_sample"] == ("batch_loess", "SERRF", "none"), (
        f"{mode}: ranking in-sample saiu {dados['ranking_in_sample']}, esperado "
        f"('batch_loess', 'SERRF', 'none'). Medianas: {dados['mediana_in_sample_pct']}"
    )


@pytest.mark.parametrize("mode", MODES)
def test_ca_3_3_sob_sete_pontos_o_loess_inverte(mode: str) -> None:
    """**A discordância entre estimadores é afirmada, não tolerada.**

    O in-sample põe o `batch_loess` em primeiro nos quatro modos. Sob sete pontos
    fora da amostra ele fica **pior que não corrigir** em três, e melhor só no
    `neg_hilic`. É o achado 4.3 medido em vez de afirmados: o LOESS zera o
    RSD in-sample por construção, então aquele estimador mede graus de liberdade
    gastos e não qualidade de correção.

    Se a inversão desaparecesse, a tese central do projeto perderia sua evidência
    mais direta - e seria achado novo, a investigar antes de aceitar.
    """
    dados = _medidas(mode)
    medianas = dados["mediana_fora_da_amostra_pct"]
    loess_pior = medianas["batch_loess"] > medianas["none"]
    esperado = mode in LOESS_PIOR_QUE_NONE
    assert loess_pior == esperado, (
        f"{mode}: `batch_loess` {'pior' if loess_pior else 'melhor'} que `none` fora da "
        f"amostra ({medianas['batch_loess']:.2f} contra {medianas['none']:.2f}), e a "
        f"contagem observada nestes quatro arquivos esperava "
        f"{'pior' if esperado else 'melhor'}. In-sample: {dados['mediana_in_sample_pct']}"
    )


def test_os_numeros_por_modo_sao_persistidos() -> None:
    """Reportar tem de ser durável: `pytest` engole saída de teste que passa."""
    destino = REPO / "out" / "ca_3_3_ranking.json"
    destino.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "criterio": "[CA-3.3]",
        "estimador_do_portao": "leave_one_qc_out_rsd (7 pontos fora da amostra)",
        "nota": (
            "fracao PAREADA de features em que o metodo bate o none, sobre as definidas nos "
            "dois lados; a comparacao com a secao 4.5 e OBSERVACAO e nunca portao (AR53); "
            "as quatro fracoes sao observacao sobre os quatro arquivos do laboratorio"
        ),
        "por_modo": {mode: _medidas(mode) for mode in MODES},
    }
    destino.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=list), "utf-8")
    lido = json.loads(destino.read_text(encoding="utf-8"))
    assert set(lido["por_modo"]) == set(MODES)
    for mode in MODES:
        assert len(lido["por_modo"][mode]["mediana_fora_da_amostra_pct"]) == 3
        assert len(lido["por_modo"][mode]["fracao_bate_none_pct"]) == 2
