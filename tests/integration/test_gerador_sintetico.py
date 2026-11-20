"""Os critérios do gerador que são mensuráveis hoje, e o portão de detectabilidade.

**O critério opera em razão 0,8 e o sujeito dele é o GERADOR**. A curva foi remedida no
gerador por reamostragem — a anterior era do paramétrico, e o joelho moveu de 0,25 para ~0,75 —, e
0,8 é a menor razão em que os quatro modos excedem 0,60, com folga de 0,05 no pior. Folga pequena
é deliberada: critério com folga grande é trivialmente satisfeito.

**O critério está inerte por `xfail(strict=True)`**, e a razão não é defeito do gerador: é
contradição entre ele e o critério. A condição de invalidação foi aplicada e
**não** se aplica — nenhuma das seis propriedades fora de tolerância é plausivelmente causal.

**A ORDEM É O CONTEÚDO.** O teste de detectabilidade vem primeiro e é
portão dos que consomem o gerador: *"sem isso, um gerador quebrado produz suíte verde e
envenena o diagnóstico"*. O caso limite a evitar é gerador que injeta defeito indetectável, detector
que não detecta, critério verde porque ambos concordam.
"""

from collections import Counter
from itertools import pairwise

import numpy as np
import pytest
from scipy.stats import ks_2samp

from refino.diagnosis.changepoint import detect_changepoints
from refino.diagnosis.policy import DiagnosisPolicy
from refino.diagnosis.profile import diagnose
from refino.domain.matrix import RefinoMatrix
from refino.domain.synthetic import SyntheticSpec
from refino.io.readers import load_matrix
from refino.metrics.rsd import rsd
from refino.metrics.structure import structural_metrics
from refino.synthetic.generator import PERFIL_REAL, generate
from tests.conftest import measurements_path, sample_list_path

pytestmark = [pytest.mark.integration, pytest.mark.slow]

MODOS = ("pos_hilic", "pos_c18", "neg_hilic", "neg_c18")
N_FEATURES = 1500
SEMENTE = 7

# Tolerancias do critério, DECLARADAS ANTES de qualquer medicao do gerador. A base
# nao vem dele: a dispersao ENTRE MODOS e o piso da exigencia — `qc_dispersion_ratio` varia 4,5x
# e `|r|` 2,7x —, entao +/-30% e de 3 a 25 vezes mais estreito que essa variacao, e passar exige
# imitar o modo CERTO e nao "um modo".
TOLERANCIA_EMERGENTE = 0.30
TOLERANCIA_DESCRITIVA = 0.10

# `PC1+PC2` medido nos quatro arquivos reais. Vive aqui e nao em `PERFIL_REAL` porque o gerador nao
# o usa como alvo — ele emerge —, e por o valor de um perfil que o codigo nao consulta ser convite
# a divergencia silenciosa.
PC12_REAL = {"pos_hilic": 0.560, "pos_c18": 0.514, "neg_hilic": 0.359, "neg_c18": 0.288}


@pytest.fixture(scope="module")
def fontes() -> dict[str, RefinoMatrix]:
    """As quatro matrizes reais.

    O gerador as recebe: `synthetic/` é núcleo puro (regra 8 do README).
    """
    return {m: load_matrix(measurements_path(m), sample_list_path(m), m) for m in MODOS}


def _spec(modo: str, **trocas: object) -> SyntheticSpec:
    base: dict[str, object] = {
        "imitated_mode": modo,
        "drift_to_noise_ratio": 0.0,
        "defect_type": "clean",
        "n_features": N_FEATURES,
        "seed": SEMENTE,
    }
    return SyntheticSpec(**{**base, **trocas})  # type: ignore[arg-type]


def _rsd_qc(matriz: RefinoMatrix) -> float:
    valores = np.asarray(matriz.values)
    return float(np.nanmedian(rsd(valores[matriz.qc_mask], axis=0, variant="outlier_removed")))


def _correlacao_mediana(matriz: RefinoMatrix, semente: int = 0) -> float:
    """`|r|` mediano par-a-par sobre 400 features sorteadas, com semente fixa.

    A semente é fixa porque a própria **medida** tem ruído de amostragem — medido, ~0,02 a 0,03
    entre subamostras diferentes —, e comparar contra um alvo medido com outra subamostra
    introduziria variação que não é do gerador.
    """
    valores = np.asarray(matriz.values)
    gerador = np.random.default_rng(semente)
    indices = gerador.choice(valores.shape[1], size=min(400, valores.shape[1]), replace=False)
    log = np.log10(valores[:, indices] + 1.0)
    utilizavel = log.std(axis=0) > 0
    correlacoes = np.corrcoef(log[:, utilizavel].T)
    return float(np.nanmedian(np.abs(correlacoes[np.triu_indices_from(correlacoes, k=1)])))


# ---------------------------------------------------------------------------
# O portão: detectabilidade antes de tudo
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("modo", MODOS)
def test_clean_nao_produz_assinatura_de_deriva(fontes: dict[str, RefinoMatrix], modo: str) -> None:
    """**O par causal do gerador, e é o portão.**

    Sem esta metade, um gerador que injetasse deriva em **toda** matriz passaria os quatro tipos, e
    a suíte ficaria verde sobre um gerador que não distingue nada. É o caso limite:
    gerador que injeta defeito indetectável, detector que não detecta, critério verde porque ambos
    concordam.

    Reamostrar em ordem aleatória é o que destrói a deriva da fonte — verificado independentemente:
    `pc1_injection_correlation` da matriz real cai de 0,4849 para 0,0276 sob permutação de linhas,
    enquanto `|r|` e `PC1+PC2` ficam idênticos, porque são da covariância entre colunas.
    """
    limpa = generate(_spec(modo), fontes[modo])
    metricas = structural_metrics(limpa.matrix, pretransform="log10p1", n_components=2)

    real = structural_metrics(fontes[modo], pretransform="log10p1", n_components=2)
    assert metricas.pc1_injection_correlation_abs < real.pc1_injection_correlation_abs, (
        f"{modo}: a sintética `clean` tem correlação com a ordem de injeção "
        f"({metricas.pc1_injection_correlation_abs:.4f}) maior ou igual à da fonte real "
        f"({real.pc1_injection_correlation_abs:.4f}). A reamostragem deveria ter destruído a "
        "deriva; se não destruiu, `clean` não é limpa e o [CA-7.4] mediria sobre rótulo errado"
    )


@pytest.mark.parametrize("modo", MODOS)
def test_a_deriva_injetada_cresce_com_a_razao(fontes: dict[str, RefinoMatrix], modo: str) -> None:
    """A assinatura cresce monotonicamente com a razão sinal-ruído.

    É a propriedade que faz a razão ser **o eixo**: se ela não crescer, o parâmetro não controla a
    quantidade que o nome diz, e todo critério que dependa da magnitude mede outra coisa.
    """
    correlacoes = []
    for razao in (0.1, 0.5, 1.0, 2.0):
        dado = generate(
            _spec(modo, drift_to_noise_ratio=razao, defect_type="gradual_drift"), fontes[modo]
        )
        metricas = structural_metrics(dado.matrix, pretransform="log10p1", n_components=2)
        correlacoes.append(metricas.pc1_injection_correlation_abs)

    assert all(a <= b + 1e-9 for a, b in pairwise(correlacoes)), (
        f"{modo}: a correlação não é monotônica na razão — {correlacoes}. Então há mais de um "
        "mecanismo e `drift_to_noise_ratio` não é o eixo que o nome promete"
    )


@pytest.mark.parametrize("modo", MODOS)
def test_a_razao_efetiva_bate_a_pedida(fontes: dict[str, RefinoMatrix], modo: str) -> None:
    """A razão **medida** no resultado contra a **pedida** — deixa de ser promessa.

    `SyntheticDataset` reporta a razão efetiva porque razão pedida é parâmetro e razão efetiva é
    resultado. Sem medi-la, "magnitude 0,5" continuaria sendo defeito.
    """
    for razao in (0.25, 0.5, 1.0):
        dado = generate(
            _spec(modo, drift_to_noise_ratio=razao, defect_type="gradual_drift"), fontes[modo]
        )
        assert dado.effective_drift_to_noise == pytest.approx(razao, rel=0.05), (
            f"{modo}: razão pedida {razao} e efetiva {dado.effective_drift_to_noise:.4f}. A "
            "magnitude declarada não é a aplicada, e todo critério de magnitude fica sem base"
        )


# ---------------------------------------------------------------------------
# Os critérios mensuráveis
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "modo",
    [
        pytest.param(
            m,
            marks=pytest.mark.xfail(
                strict=True,
                reason=(
                    "o `pos_hilic` fica a 42,9% da faixa, contra tolerancia de 30%, com os sete "
                    "QCs na semente 7 (a tabela de DV32). Nao e margem fina. O RSD real "
                    "dele e 143,81% — quase 10x o do `neg_c18` —, e a reamostragem de residuo de "
                    "QC nao alcanca essa dispersao. DESTRAVADO por entender o limite da "
                    "reamostragem naquele regime, e por nada mais: nem tolerancia maior, nem "
                    "ruido sintetico por cima. Ver DV22. `strict=True` e deliberado, e o "
                    "marcador e DECLARATIVO, e nao `pytest.xfail()` imperativo chamado antes de "
                    "`generate`: assim o teste executa a medicao que esta razao descreve, e da "
                    "`XPASS` se o gerador melhorar."
                ),
            ),
        )
        if m == "pos_hilic"
        else pytest.param(
            m,
            marks=pytest.mark.xfail(
                strict=True,
                reason=(
                    "o conserto de `_posicoes_de_qc` para SETE QCs leva `pos_c18` a "
                    "-42,4% e `neg_hilic` a -42,2%, contra tolerancia de 30%. E o conserto NAO e "
                    "a causa: ele EXPOE. Medido em 5 sementes (7, 11, 23, 42, 101) e nos quatro "
                    "modos, o criterio ja reprovava em 13 de 20 celulas com os OITO QCs de antes, "
                    "e passa a 16 de 20 com os sete. Ele era verde por SORTE DE SEMENTE: a 7 e "
                    "favoravel no `pos_c18` (+8,5%) e no `neg_c18` (+2,4%), e nas outras quatro "
                    "reprova nos dois. DESTRAVADO por o gerador alcancar a dispersao de RSD de QC "
                    "do modo imitado — e por nada mais: nem tolerancia maior, nem semente "
                    "escolhida, nem ruido sintetico por cima. Ver DV32 e DV22."
                ),
            ),
        )
        if m in ("pos_c18", "neg_hilic")
        else m
        for m in MODOS
    ],
)
def test_ca_7_1_rsd_de_qc_na_faixa_do_modo_imitado(
    fontes: dict[str, RefinoMatrix], modo: str
) -> None:
    """O RSD mediano de QC na faixa do modo imitado, ±30%.

    **O `0,15` absoluto do critério original era número declarado sem referente medido**, e nenhum
    dos quatro modos o satisfaz — o menor é 15,50% no `neg_c18`, meio ponto acima. Mesma forma do
    `0,70` da regra 3 do diagnóstico e dos `5 min` do critério.

    O `pos_hilic` fica a **42,9%** — **dizia 33,3% até 06/09/2026, e aquele era o valor com os oito
    QCs anteriores ao conserto de `_posicoes_de_qc`**. Com 42,9% contra tolerância de
    30% já não é margem fina. Não é ruído: o RSD real dele é 143,81%, quase dez vezes o do
    `neg_c18`, e reproduzir variação dessa ordem por reamostragem de resíduo tem limite. Está no
    perfil do critério.
    """
    dado = generate(_spec(modo), fontes[modo])
    medido, alvo = _rsd_qc(dado.matrix), PERFIL_REAL[modo]["rsd_qc_mediano"]
    desvio = abs(medido - alvo) / alvo

    assert desvio <= TOLERANCIA_EMERGENTE, (
        f"{modo}: RSD de QC {medido:.4f} contra {alvo:.4f} do arquivo real, desvio "
        f"{desvio:.1%} acima de {TOLERANCIA_EMERGENTE:.0%}"
    )


@pytest.mark.parametrize(
    "modo",
    [
        pytest.param(
            m,
            marks=pytest.mark.xfail(
                strict=True,
                reason=(
                    "o `neg_hilic` reprova NA SEMENTE 7, e a causa e realizacao e nao mecanismo. "
                    "Medido em 5 sementes (7, 11, 23, 42, 101) e nos quatro modos: com os SETE "
                    "QCs do conserto o criterio acerta 18 de 20 celulas, e com os OITO de antes "
                    "acertava 16 de 20 — o conserto MELHORA a deteccao. O `neg_hilic` acerta 4/5 "
                    "com sete e 3/5 com oito; o que muda e QUAIS sementes falham, e a 7 e a unica "
                    "que falha depois do conserto. A hipotese de que menos QCs dificultariam o "
                    "salto — e que isso ligaria ao residuo `step_change` do `[CA-7.4]` — foi "
                    "MEDIDA E REFUTADA antes de ser escrita (AR40). DESTRAVADO por o "
                    "criterio passar a medir sobre varias sementes em vez de uma, que e o defeito "
                    "que DV32 nomeia. Ver DV32."
                ),
            ),
        )
        if m == "neg_hilic"
        else m
        for m in MODOS
    ],
)
def test_ca_7_3_o_salto_cai_no_primeiro_qc_depois_da_posicao(
    fontes: dict[str, RefinoMatrix], modo: str
) -> None:
    """O ponto de mudança na posição pedida, **na resolução da grade de QC**.

    **A resolução é física e a primeira versão deste teste a ignorou.** A trajetória de PC1 dos QCs
    tem um ponto a cada `qc_every` injeções — 11 nestes arquivos —, então um salto na injeção 34 só
    é localizável no intervalo `(34, 45]`: o QC da injeção 34 é o **último antes** do salto, e o da
    45 é o **primeiro depois**. Exigir 34 ± 2 é exigir resolução que o instrumento não tem.

    A asserção é sobre o **primeiro QC estritamente depois** da posição pedida, que é o ponto mais
    cedo em que a informação existe.
    """
    dado = generate(
        _spec(modo, drift_to_noise_ratio=1.0, defect_type="step_change", changepoint_position=0.5),
        fontes[modo],
    )
    metricas = structural_metrics(dado.matrix, pretransform="log10p1", n_components=2)
    ordens_qc = np.array([a.injection_order for a in dado.matrix.samples if a.sample_type == "qc"])
    pontos = detect_changepoints(np.array(metricas.qc_pc1_trajectory), ordens_qc)

    corte = round(0.5 * dado.matrix.n_samples)
    depois = ordens_qc[ordens_qc > corte]
    esperado = int(depois[0]) if depois.size else int(ordens_qc[-1])

    assert list(pontos) == [esperado], (
        f"{modo}: pontos detectados {list(pontos)}, esperado exatamente [{esperado}] — o primeiro "
        f"QC depois da injeção {corte}, que é a resolução da grade de QC. Ponto a mais significa "
        "que a deriva injetada tem estrutura além do salto; ponto a menos, que o salto não é "
        "detectável na magnitude pedida"
    )


@pytest.mark.parametrize("modo", MODOS)
def test_ca_7_5_a_distribuicao_bate_com_a_fonte(fontes: dict[str, RefinoMatrix], modo: str) -> None:
    """KS abaixo de 0,10 contra a distribuição real.

    **O critério dizia "contra a distribuição real do `pos_c18`" e aqui é contra a fonte de cada
    modo**, que é mais forte: com reamostragem, casar o `pos_c18` seria trivial para uma sintética
    derivada dele e não diria nada sobre as outras três.

    Sobre células não nulas: zero é ausência e não intensidade, e incluí-lo faria a estatística
    medir a fração de zeros, que já é propriedade própria no critério.
    """
    dado = generate(_spec(modo), fontes[modo])
    gerado = np.asarray(dado.matrix.values).ravel()
    real = np.asarray(fontes[modo].values).ravel()
    gerado, real = np.log10(gerado[gerado > 0]), np.log10(real[real > 0])

    gerador = np.random.default_rng(0)
    n = min(20_000, gerado.size, real.size)
    estatistica = ks_2samp(
        gerador.choice(gerado, size=n, replace=False),
        gerador.choice(real, size=n, replace=False),
    ).statistic

    assert estatistica < 0.10, (
        f"{modo}: KS {estatistica:.4f} contra a fonte, acima de 0,10. Com reamostragem isso não "
        "deveria acontecer — a base É o resíduo real —, então indica que os passos 3 a 6 "
        "deslocaram a distribuição"
    )


@pytest.mark.parametrize("modo", MODOS)
def test_ca_7_6_determinismo_e_o_par_causal_da_semente(
    fontes: dict[str, RefinoMatrix], modo: str
) -> None:
    """Mesma semente dá matriz idêntica; semente diferente dá matriz diferente.

    A segunda metade é obrigatória: sem ela, um gerador que ignorasse `seed` passaria a primeira.
    """
    argumentos = {"drift_to_noise_ratio": 0.5, "defect_type": "gradual_drift"}
    primeira = generate(_spec(modo, **argumentos), fontes[modo])
    segunda = generate(_spec(modo, **argumentos), fontes[modo])
    outra = generate(_spec(modo, seed=SEMENTE + 1, **argumentos), fontes[modo])

    assert np.array_equal(primeira.matrix.values, segunda.matrix.values)
    assert primeira.effective_drift_to_noise == segunda.effective_drift_to_noise
    assert not np.array_equal(primeira.matrix.values, outra.matrix.values), (
        f"{modo}: sementes diferentes deram matriz idêntica; `seed` não está sendo consultado"
    )


@pytest.mark.parametrize("modo", MODOS)
def test_ca_7_8_o_perfil_e_reportado_com_desvio_por_propriedade(
    fontes: dict[str, RefinoMatrix], modo: str
) -> None:
    """Critério de **reporte obrigatório**, não portão.

    **O que este teste afirma é que o perfil EXISTE e é completo**, não que ele passe. As quatro
    propriedades são medidas, o desvio de cada uma é calculado contra o arquivo real, e o perfil
    inteiro vai para a saída do teste — porque é ele que o consumidor consulta ao interpretar um
    resultado, e "reprovou" não distingue seis desvios marginais de seis catastróficos.

    **A condição que preserva o rigor está no critério:** se o critério ou o consumidor reprovarem,
    o perfil é a primeira coisa a consultar, e propriedade fora de tolerância plausivelmente causal
    invalida a conclusão até ser descartada **por medição**.
    """
    dado = generate(_spec(modo), fontes[modo])
    matriz = dado.matrix
    metricas = structural_metrics(matriz, pretransform="log10p1", n_components=2)
    valores = np.asarray(matriz.values)

    medido = {
        "rsd_qc_mediano": _rsd_qc(matriz),
        "dispersao_qc_amostra": metricas.qc_dispersion_ratio,
        "correlacao_mediana": _correlacao_mediana(matriz),
        "pc1_pc2": float(sum(metricas.explained_variance_ratio)),
        "fracao_de_zeros": float((valores == 0).mean()),
    }
    alvo = dict(PERFIL_REAL[modo])
    alvo["pc1_pc2"] = PC12_REAL[modo]

    perfil = []
    for chave, valor in medido.items():
        tolerancia = TOLERANCIA_DESCRITIVA if chave == "fracao_de_zeros" else TOLERANCIA_EMERGENTE
        desvio = abs(valor - alvo[chave]) / abs(alvo[chave])
        perfil.append(
            f"{chave}={valor:.4f}/{alvo[chave]:.4f} ({desvio:+.1%}, tol {tolerancia:.0%})"
        )

    # O perfil e o PRODUTO deste teste. `-rA` no noturno o torna legivel; o print existe porque
    # pytest engole `stdout` de teste que passa, e relatorio que so aparece na falha e o defeito.
    print(f"\n[CA-7.8] {modo}: " + " | ".join(perfil))

    assert len(perfil) == 5, "o perfil tem de cobrir as cinco propriedades declaradas"
    assert all(np.isfinite(v) for v in medido.values()), (
        f"{modo}: propriedade não finita no perfil — {medido}. Perfil com `nan` não é consultável, "
        "e a condição de invalidação do critério depende de ele ser legível"
    )


# ---------------------------------------------------------------------------
# Detectabilidade em razão 0,8 e acurácia do diagnose
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("modo", MODOS)
def test_ca_7_2_a_injecao_produz_o_efeito_pedido(
    fontes: dict[str, RefinoMatrix], modo: str
) -> None:
    """`gradual_drift` em razão **0,8** dá `pc1_injection_correlation` acima de 0,60.

    **O sujeito é o GERADOR e não um detector**. Este é o único critério que verifica
    que a injeção produz o efeito pedido, e isso precisa existir independentemente de detector —
    sem ele, "gerador irrealista" e "política que não generaliza" voltam a ser indistinguíveis.

    **0,8 é a menor razão em que os quatro modos passam**, com folga de 0,05 no pior. Folga pequena
    é deliberada: em razão 1,0 o mínimo é 0,7938, e folga grande torna o critério trivialmente
    satisfeito — foi o defeito da versão em razão 0,5.
    """
    dado = generate(
        _spec(modo, drift_to_noise_ratio=0.8, defect_type="gradual_drift"), fontes[modo]
    )
    metricas = structural_metrics(dado.matrix, pretransform="log10p1", n_components=2)

    assert metricas.pc1_injection_correlation_abs > 0.60, (
        f"{modo}: correlação {metricas.pc1_injection_correlation_abs:.4f} em razão 0,8, abaixo de "
        "0,60. A curva medida dá mínimo 0,6522 entre os quatro modos; desvio aqui é do gerador, "
        "porque o sujeito deste critério é ele e não um detector"
    )


def acuracia_do_diagnose_em_200(
    fontes: dict[str, RefinoMatrix],
) -> tuple[int, Counter[tuple[str, str]]]:
    """A medição do critério: acertos do `diagnose` em 200 sintéticas, 50 por tipo, e a confusão.

    **Função e não corpo de teste, porque são DOIS critérios a pedir a mesma quantidade**
    (C11): o critério do
    diagnóstico pede acurácia acima de 85% em 200 matrizes sintéticas com defeito conhecido, e o
    teste dele importava um gerador que nunca existiu —
    o `xfail` absorvia o `ImportError`, e o critério nunca mediu nada. Uma segunda cópia deste
    laço lá divergiria da primeira no dia em que alguém mexesse numa delas.
    """
    from refino.api.rotas import RSD_MEDIANA_MAX_POR_MODO

    politica = DiagnosisPolicy(
        max_spread=0.10,
        rsd_mediana_max=0.20,
        pc1_injection_correlation_max=0.30,
        step_sustain_min=2.0,
        qc_monotonicity_min=0.487,
        trajectory_movement_min=1e-9,
        rsd_mediana_max_por_modo=RSD_MEDIANA_MAX_POR_MODO,
    )
    tipos = ("gradual_drift", "step_change", "erratic", "clean")
    confusao: Counter[tuple[str, str]] = Counter()

    for indice in range(200):
        tipo = tipos[indice % 4]
        modo = MODOS[(indice // 4) % 4]
        extra: dict[str, object] = {
            "defect_type": tipo,
            "drift_to_noise_ratio": 0.0 if tipo == "clean" else 0.8,
            "n_features": 1200,
            "seed": 1000 + indice,
        }
        if tipo == "step_change":
            extra["changepoint_position"] = 0.3 + 0.4 * ((indice // 16) % 2)
        dado = generate(_spec(modo, **extra), fontes[modo])
        confusao[(tipo, diagnose(dado.matrix, policy=politica).defect_type)] += 1

    return sum(confusao[(t, t)] for t in tipos), confusao


@pytest.mark.xfail(
    strict=True,
    reason=(
        "[CA-7.4] da 56,0% no corpus deste repositorio com a politica ESCALAR — 112 de 200 — "
        "e 66,5% sob o limiar POR MODO em outro corpus; os dois sao comparaveis em FORMA e nao "
        "numero a numero. A causa e CONTRADICAO com o [CA-7.1] e nao defeito "
        "do gerador. A regra 1 da cascata exige rsd_mediana_max = 0,20 para classificar "
        "`clean`, e o [CA-7.1] exige que a sintetica `clean` tenha o RSD do modo imitado — acima "
        "de 0,20 em TRES dos quatro modos reais. Nenhum gerador satisfaz os dois. A condicao de "
        "invalidacao foi aplicada e NAO se aplica: nenhuma das seis propriedades fora de "
        "tolerancia e plausivelmente causal. "
        "Resolver a contradicao pelo limiar por modo e PARCIAL: "
        "deu +12,0 dos 30,5 pontos que faltavam, e o criterio continua reprovando. "
        "Os 18,5 restantes sao `step_change` a 38%, com 29 de 50 caindo em `erratic` pelas "
        "REGRAS 2 E 3. "
        "Continua valendo o que NAO destrava: nem politica reajustada, nem magnitude maior, "
        "nem `clean` menos realista. `strict=True` e deliberado: se passar a passar, alguma "
        "coisa fechou os 18,5 e o marcador tem de sair. "
        "Este teste usa o mapa por modo que a producao passa, e nao a politica ESCALAR — que a "
        "`DiagnosisPolicy` declara ser a RESERVA de modo desconhecido. Com o mapa, o corpus "
        "REPRODUZIVEL deste repositorio da 139/200 = 69,5%, contra 112/200 = 56,0% com a "
        "reserva. Os 13,5 pp de diferenca sao TODOS do `clean`, que vai de 18/50 a 46/50. "
        "Os 66,5% continuam citados acima porque foram "
        "medidos, mas o corpus deles NAO e reconstruivel daqui: o numero "
        "reproduzivel sob a cascata real e 69,5%."
    ),
)
def test_ca_7_4_diagnose_recupera_o_tipo(fontes: dict[str, RefinoMatrix]) -> None:
    """Acurácia de `diagnose` acima de 85% sobre 200 matrizes, 50 por tipo.

    **A política é a real do diagnóstico, sem reajuste.** Se a acurácia ficar abaixo, o
    desfecho é achado sobre o gerador ou sobre a política — nunca ajustar até 85%.

    > **E ela DEIXOU de ser a real sem que nada acusasse.** O
    > limiar por modo entrou em `policy.py`, `profile.py` e `api/rotas.py` e não tocou aqui, então
    > este
    > critério passou a medir a política **anterior** — o escalar, que a `DiagnosisPolicy` declara
    > ser a *reserva de modo desconhecido*. Alinhado; a caducidade era o instrumento a ler a
    > instância errada, e o conserto **não toca o estímulo**, que é o que a razão do `xfail` proíbe.

    **A matriz de confusão é reportada**, porque acurácia agregada esconde qual tipo o detector
    confunde: medido, 107 das 200 classificações saem `erratic`, que é o desfecho de **queda** da
    cascata, e `clean` acerta só 36%.
    """
    # **O MAPA POR MODO ENTRA em 31/08/2026, e isto é caducidade e NÃO emenda.** A docstring
    # de `DiagnosisPolicy` declara qual dos dois a cascata usa, verbatim: `rsd_mediana_max_por_modo`
    # é *"quatro, calibrados | regra 1 — é ESTE que a cascata usa"*, e o escalar `0,20` é
    # *"RESERVA de modo desconhecido"*. Sem o mapa, este critério exercitava a reserva sobre os
    # quatro modos CONHECIDOS: o portão media uma cascata que a produção já não roda: o
    # instrumento lia a instância errada do referente.
    #
    # **QUATRO rotas de produção passam o mapa** — `/matrizes/recomendar`, `POST /execucoes`,
    # `/matrizes/medir-corretor` e `/matrizes/diagnosticar`.
    #
    # > Duas delas passam a dá-lo só aos quatro arquivos de referência,
    # > pelo `sha256` das medições (`mapa_calibrado_para`): `/matrizes/diagnosticar` e `POST
    # > /execucoes`. As outras duas continuam com `perfil_para_a_politica`, que é o perfil da
    # > política aprendida e tem de ser o do corpus em que ela treinou. Este critério
    # > não muda: as matrizes dele são sintéticas, e a política é construída aqui, com o mapa.
    #
    # Sem número de linha, de propósito: o alvo é `grep -n
    # 'rsd_mediana_max_por_modo=RSD_MEDIANA_MAX_POR_MODO' src/refino/api/rotas.py`, e linha
    # pinada caduca em silêncio — foi o que aconteceu com o `:593` e o `:1419` que esta linha
    # citava, que hoje são `scores=estrutura.qc_pc1_trajectory,` e um comentário sobre o
    # `corpo.get("vencedor")`.
    # **Importado e não redigitado**, pelo
    # motivo: uma segunda cópia dos quatro números divergiria em silêncio no dia em que
    # a calibração fosse remedida, e o portão ficaria a afirmar sobre os limiares de ontem.
    #
    # **`max_spread` e `trajectory_movement_min` FICAM como estão**, e não é omissão: a mesma
    # docstring diz que os dois são *"sem valor declarado"* — *"quem tem os dados calibra"* —,
    # então aqui não há valor de produção de que divergir. Os de `LIMIARES_INICIAIS` (3,0 e 1,0)
    # são os padrões que o cliente pode sobrescrever **na requisição**, que é outra quantidade.
    acertos, confusao = acuracia_do_diagnose_em_200(fontes)
    print(f"\n[CA-7.4] acurácia {acertos}/200; confusão: {dict(confusao)}")
    assert acertos / 200 > 0.85, (
        f"acurácia {acertos}/200 = {acertos / 200:.1%}, abaixo de 85%. Consulte o perfil de "
        "[CA-7.8] ANTES de concluir sobre a política: é a condição de invalidação do critério"
    )


# ---------------------------------------------------------------------------
# `n_qcs`: sete, como nos quatro arquivos reais
# ---------------------------------------------------------------------------

# As injeções dos QCs nos quatro arquivos reais, em base 0. As diferenças são 11, 11, 11, 11, 11
# e **12** — periódico com o último puxado para o fim da corrida, e não periódico mais um extra.
POSICOES_REAIS_DE_QC = (0, 11, 22, 33, 44, 55, 67)


@pytest.mark.parametrize("modo", MODOS)
def test_o_gerador_produz_sete_qcs_nas_posicoes_dos_arquivos_reais(
    fontes: dict[str, RefinoMatrix], modo: str
) -> None:
    """Sete QCs, nas posições reais — e o par causal do defeito está no teste vizinho.

    **O defeito está corrigido.** O
    `_posicoes_de_qc` **acrescentava** o índice final em vez de substituir a última posição
    periódica, e com `n_samples=68`/`qc_every=11` produzia **oito** QCs com dois adjacentes no fim.

    **Por que isto não era detalhe de contagem.** É uma das seis discrepâncias que o critério
    mede, e morde a atribuição central do projeto: as conclusões do R valem sobre a floresta
    **sob sete pontos de treino** — `ranger` com `min.node.size = 5` e `n = 7` faz uma a duas
    divisões — e a validação sintética que as sustenta corria com oito. Número
    certo, população errada.

    A asserção é sobre as POSIÇÕES e não só sobre a contagem, porque contar sete não distingue
    `(0, 11, 22, 33, 44, 55, 67)` de qualquer outro conjunto de sete.
    """
    dado = generate(_spec(modo), fontes[modo])
    posicoes = tuple(
        i for i, amostra in enumerate(dado.matrix.samples) if amostra.sample_type == "qc"
    )
    assert posicoes == POSICOES_REAIS_DE_QC, (
        f"{modo}: QCs em {posicoes}, esperava {POSICOES_REAIS_DE_QC} — as posições dos quatro "
        "arquivos reais. Oito QCs com dois adjacentes no fim é o defeito do item 16"
    )


def test_o_negativo_de_sete_qcs_acusa_a_versao_que_acrescentava(
    fontes: dict[str, RefinoMatrix],
) -> None:
    """Planta a versão antiga de `_posicoes_de_qc` e afirma que o teste acima reprovaria.

    **O teste que prova que a verificação CONSEGUE falhar.** Sem ele, o teste vizinho é
    indistinguível de verificação decorativa — mecanismo construído e nunca exercitado no sentido
    da falha.

    A planta é a linha exata que estava no código até hoje: `append` no lugar de substituir. O par
    causal é o que separa *"o teste passa"* de *"o teste passa porque a propriedade vale"* — as
    duas execuções diferem só pela planta, e os desfechos têm de divergir.
    """
    spec = _spec("pos_c18")

    def posicoes_da_versao_antiga(n_samples: int, qc_every: int) -> tuple[int, ...]:
        posicoes = list(range(0, n_samples, qc_every))
        if posicoes[-1] != n_samples - 1:
            posicoes.append(n_samples - 1)
        return tuple(posicoes)

    plantadas = posicoes_da_versao_antiga(spec.n_samples, spec.qc_every)

    # A planta CAI sobre dado em que as duas coisas diferem: oito contra sete, e o
    # penúltimo intervalo de 1 em vez de 12. Sem esta asserção o negativo poderia estar a
    # comparar duas listas iguais e a passar por coincidência.
    assert len(plantadas) == 8, f"a planta não reproduz o defeito: {plantadas}"
    assert plantadas != POSICOES_REAIS_DE_QC
    assert plantadas[-2:] == (66, 67), "o defeito é o par adjacente no fim"

    # E o desfecho oposto: o código de HOJE, sobre a mesma especificação, dá sete.
    dado = generate(spec, fontes["pos_c18"])
    de_hoje = tuple(
        i for i, amostra in enumerate(dado.matrix.samples) if amostra.sample_type == "qc"
    )
    assert de_hoje == POSICOES_REAIS_DE_QC
    assert len(de_hoje) != len(plantadas), "os dois desfechos têm de divergir, ou o par é vácuo"
