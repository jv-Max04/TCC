#!/usr/bin/env python3
"""Gera `data/fixtures/r_cv_containment_prereg.json` — o pré-registro da validação cruzada do R.

**Este script não mede nada nosso, e é isso que o torna pré-registro.** Ele lê o
alvo — as colunas `SERRF` e `batch_loess` dos quatro `QC - RSD.csv` do
laboratório —, congela as 20 sementes **por valor**, e escreve os critérios de
desfecho. Nenhuma linha aqui importa `refino.metrics.crossval` nem executa um
corretor: se executasse, o artefato deixaria de sustentar a afirmação "foi fixado
antes de medir", que é a única coisa que ele existe para sustentar.

A ordem é vinculante: este arquivo entra no tronco **antes** da
primeira execução da medição de contenção. O `sha256` no histórico é o que torna
"fixado antes" verificável por terceiro; um valor que só existe numa branch numa
máquina faz a mesma afirmação depender de confiança.

**As 20 sementes são declaradas por VALOR e não por contagem.** O
`r_serrf_divergence.json` registrou `n_python_seeds: 20` como contagem, mas a
mediana congelada depende do conjunto e não só do tamanho.

**O espaçamento das sementes é decisão de desenho e não detalhe de escrita.** O
`LoessCorrector` deriva o fluxo de cada feature por `seed + index`, com `index`
sendo o índice ORIGINAL da coluna (`loess.py:140` e `loess.py:120-124`), que
percorre `[0, n_features)`. Logo duas execuções com sementes `s` e `s'`
compartilham fluxo por feature exatamente quando `|s - s'| < n_features`.
Com sementes consecutivas as 20 execuções seriam a mesma
sequência deslocada, e o intervalo do `batch_loess` sairia estreito **por construção**,
medindo o defeito em vez da variabilidade. Com espaçamento maior que o maior
`n_features` dos quatro modos, as faixas são disjuntas e as 20 execuções são
realizações independentes — que é a premissa que a contenção assume.

Isso **não** afrouxa régua: nenhuma linha de `loess.py`
muda, nenhuma tolerância muda. É satisfazer a premissa do critério com o
parâmetro que o critério já tinha livre. A consequência para a atribuição está
declarada no próprio fixture, e é o que impede a leitura de que o
estreitamento bilateral continua previsto.

O SERRF é indiferente ao espaçamento: os cinco sítios estocásticos do
`SerrfCorrector` derivam de `SeedSequence` sobre tupla, que não colide
aditivamente.

Uso:

    .venv/bin/python scripts/gen_r_cv_prereg.py            # escreve
    .venv/bin/python scripts/gen_r_cv_prereg.py --check    # não escreve
"""

import argparse
import csv
import datetime
import hashlib
import itertools
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parent.parent
FIXTURE = REPO / "data" / "fixtures" / "r_cv_containment_prereg.json"

MODES = ("pos_hilic", "pos_c18", "neg_hilic", "neg_c18")

# As duas metades do critério. `none` fica de fora de propósito: o critério
# nomeia `SERRF` e `batch_loess`, e o caminho `none` é oráculo exato ancorado
# por outro teste — incluí-lo aqui sugeriria que a contenção o cobre.
METHODS = ("SERRF", "batch_loess")

# ---------------------------------------------------------------------------
# As 20 sementes
# ---------------------------------------------------------------------------
#
# Regra: `SEED_GAP * k` para k de 1 a 20. Duas propriedades, e as duas são
# verificáveis por quem ler o arquivo em vez de acreditar nele.
#
# 1. O conjunto é gerado por regra fechada, não escolhido a dedo. Um conjunto
#    escolhido valor a valor seria indistinguível de um conjunto escolhido
#    depois de ver o resultado, e o pré-registro deixaria de comprar o que
#    existe para comprar.
# 2. O espaçamento excede o maior `n_features` dos quatro modos, então as faixas
#    `[s, s + n_features)` de duas execuções nunca se tocam. A margem é
#    conferida em `verificar_disjuncao` contra os arquivos, não suposta.
# **50.000 e não 10.000.** O `qc_rsd_r_cv` deriva semente POR SORTEIO, para que a
# média seja sobre replicatas independentes. Com a derivação por sorteio, o
# espaçamento entre sementes tem de comportar `n_draws` faixas de
# `n_features`, e não uma: 5 x 10.000 = 50.000.
SEED_GAP = 50_000
N_SEEDS = 20
SEEDS = tuple(SEED_GAP * k for k in range(1, N_SEEDS + 1))

# Espelha `refino.metrics.crossval.DRAW_SEED_GAP`. Declarado aqui e não importado
# para manter a promessa do cabeçalho — este gerador não importa o módulo que
# mede —, e conferido mecanicamente contra ele por
# `test_o_preregistro_ainda_descreve_o_que_este_modulo_roda`. Constante duplicada
# sem teste de paridade é deriva à espera.
DRAW_SEED_GAP = 10_000

# ---------------------------------------------------------------------------
# A configuração do estimador, congelada
# ---------------------------------------------------------------------------

N_DRAWS = 5
TRAIN_RATIO = 0.7
VARIANT = "outlier_removed"
N_SAMPLED_FEATURES = 500

# Erro padrão da mediana amostral em `n = 500`, medido por bootstrap sobre a
# própria coluna que o critério compara. Entra no relatório como
# DECOMPOSIÇÃO da largura, nunca como margem somada ao intervalo.
SAMPLING_SE_PCT = {
    "pos_hilic": 2.73,
    "neg_hilic": 2.69,
    "pos_c18": 1.62,
    "neg_c18": 6.45,
}

# Teto de parede da medição, em minutos. Declarado com o
# marcador `slow` explícito: critério de dezenas de minutos sem marcador é o
# tipo de coisa que alguém "conserta" reduzindo a cobertura.
WALL_BUDGET_MINUTES = 335

_REFERENCE_CANDIDATES = (REPO / "data" / "reference", REPO / "serrf" / "data")


def reference_root() -> Path:
    """As duas raízes candidatas dos dados de referência, na mesma ordem do `tests/conftest.py`.

    A duplicação é deliberada: um gerador não importa de `tests/`, e replicar
    duas linhas é melhor que acoplar o gerador à suíte.
    """
    for candidate in _REFERENCE_CANDIDATES:
        if (candidate / "input").is_dir():
            return candidate
    raise SystemExit(
        "dados de referência ausentes: nenhum de "
        f"{[str(c) for c in _REFERENCE_CANDIDATES]} tem `input/`. "
        "Rode scripts/fetch_reference_data.sh."
    )


def qc_rsd_relative(mode: str) -> str:
    """Caminho relativo do `QC - RSD.csv`. Nome com espaços, não underscores."""
    return f"output/{mode}_toserrf - normalization result/QC - RSD.csv"


def digest_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_published_column(path: Path, method: str) -> np.ndarray:
    """Uma coluna do `QC - RSD.csv`, como FRAÇÃO — é a escala do arquivo.

    Medido no `pos_hilic`: a primeira célula da coluna `none` é `0.3109...`, e a
    tabela 4.4 publica 143,8 para a mediana daquela coluna. Logo o arquivo está
    em fração e a tabela em pontos percentuais, e todo número deste fixture
    carrega o sufixo `_pct` justamente porque essa conversão já produziu defeito
    real antes.
    """
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    header = rows[0]
    if method not in header:
        raise SystemExit(f"coluna {method!r} ausente de {path.name}: {header}")
    column = header.index(method)
    return np.array(
        [
            np.nan if row[column].strip() in ("", "NA", "NaN") else float(row[column])
            for row in rows[1:]
        ],
        dtype=np.float64,
    )


def published_target(path: Path, method: str) -> dict[str, Any]:
    """A mediana publicada e a população sobre a qual ela é tomada.

    **A política de indefinido é declarada e não implícita.** A mediana é
    sobre as features com valor finito na coluna publicada; `n_defined` viaja
    junto porque mediana sobre subconjunto diferente não é a mesma mediana.
    """
    values = read_published_column(path, method)
    defined = np.isfinite(values)
    return {
        "median_rsd_pct": float(np.median(values[defined]) * 100.0),
        "n_features": int(values.size),
        "n_defined": int(defined.sum()),
        "n_undefined": int((~defined).sum()),
    }


def verificar_disjuncao(feature_counts: dict[str, int]) -> dict[str, Any]:
    """Prova aritmética de que NENHUMA faixa de `seed + index` toca outra.

    **Cobre as `n_seeds x n_draws` faixas e não as `n_seeds`, e a diferença é que
    há semente por sorteio.** Com semente por sorteio há `20 x 5 = 100` faixas, e a
    condição é sobre o menor gap entre **elas**, ordenadas.

    Vai ao fixture como número conferido contra os arquivos, e não como afirmação
    de que o espaçamento "é grande o bastante" — o maior `n_features` é observação
    sobre estes quatro arquivos, e um quinto modo refaz a conta.
    """
    maior = max(feature_counts.values())
    derivadas = sorted(s + d * DRAW_SEED_GAP for s in SEEDS for d in range(N_DRAWS))
    gaps = [b - a for a, b in itertools.pairwise(derivadas)]
    menor_gap = min(gaps)
    return {
        "n_faixas": len(derivadas),
        "menor_gap_entre_faixas": menor_gap,
        "menor_gap_entre_sementes": min(b - a for a, b in itertools.pairwise(SEEDS)),
        "draw_seed_gap": DRAW_SEED_GAP,
        "maior_n_features_nos_quatro_modos": maior,
        "modo_do_maior": max(feature_counts, key=lambda m: feature_counts[m]),
        "faixas_disjuntas": bool(menor_gap >= maior),
        "margem": menor_gap - maior,
        "regra": (
            "duas execucoes do LoessCorrector compartilham fluxo por feature "
            "exatamente quando |s - s'| < n_features, porque loess.py passa "
            "seed = seed + index com index sendo o indice ORIGINAL da coluna. As "
            "faixas sao as n_seeds * n_draws sementes DERIVADAS "
            "(seed + draw * DRAW_SEED_GAP), nao as n_seeds declaradas: desde a "
            "correcao de AR46 cada sorteio tem fluxo proprio, e a disjuncao tem de "
            "valer entre sorteios tambem. Conferida contra os arquivos, nao suposta."
        ),
    }


def build() -> dict[str, Any]:
    root = reference_root()

    sources: dict[str, str] = {}
    targets: list[dict[str, Any]] = []
    feature_counts: dict[str, int] = {}

    for mode in MODES:
        relative = qc_rsd_relative(mode)
        path = root / relative
        if not path.is_file():
            raise SystemExit(f"ausente: {relative}. Rode scripts/fetch_reference_data.sh.")
        sources[relative] = digest_of(path)
        for method in METHODS:
            alvo = published_target(path, method)
            feature_counts[mode] = alvo["n_features"]
            targets.append({"mode": mode, "method": method, **alvo})

    return {
        "criterio": "[CA-3.12]",
        "natureza": (
            "PRE-REGISTRO. Escrito e commitado ANTES da primeira execucao do criterio. "
            "Nenhum numero medido do lado do Python aparece aqui: os alvos sao do "
            "oraculo e as sementes sao escolha declarada."
        ),
        "unidades": (
            "median_rsd_pct e sampling_se_pct estao em PONTOS PERCENTUAIS. A coluna do "
            "QC - RSD.csv esta em FRACAO (0,31 = 31%) e a conversao acontece na leitura. "
            "Sufixo de unidade em toda quantidade e reparo de defeito real: a secao 4.4 "
            "trouxe 1,9e-15 ao lado de uma tabela em pontos percentuais."
        ),
        "preregistration": {
            "seeds": list(SEEDS),
            "n_seeds": N_SEEDS,
            "seed_rule": (
                f"SEED_GAP * k para k de 1 a {N_SEEDS}, com SEED_GAP = {SEED_GAP}. Regra "
                "fechada e nao escolha valor a valor: um conjunto escolhido a dedo seria "
                "indistinguivel de um conjunto escolhido depois de ver o resultado."
            ),
            "seed_spacing_rationale": (
                "O espacamento e decisao de desenho, nao formatacao. O LoessCorrector "
                "deriva o fluxo de cada feature por seed + index (loess.py:140), com index "
                "sendo o indice ORIGINAL da coluna, logo em [0, n_features). Sementes "
                "consecutivas fariam as 20 execucoes serem a mesma sequencia deslocada de "
                "uma posicao, e o intervalo do batch_loess sairia estreito POR CONSTRUCAO "
                "— mediria DV15 e nao a variabilidade semente-a-semente. Com o espacamento "
                "acima do maior n_features as faixas sao disjuntas e as execucoes sao "
                "realizacoes independentes, que e a premissa que a contencao assume. "
                "NAO e correcao de DV15 nem afrouxamento de regua: nenhuma linha de "
                "loess.py muda e nenhuma tolerancia muda."
            ),
            "seed_disjointness": verificar_disjuncao(feature_counts),
            "estimator": "refino.metrics.crossval.qc_rsd_r_cv",
            "n_draws": N_DRAWS,
            "draw_seed_gap": DRAW_SEED_GAP,
            "draw_seed_rule": (
                "o corretor recebe seed + draw * DRAW_SEED_GAP em cada sorteio, e nao a "
                "mesma semente nos n_draws. A versao anterior passava seed=seed nos cinco: so "
                "a particao variava, os fluxos do LOESS e do SERRF eram identicos entre eles, e "
                "a media era sobre replicatas correlacionadas — n_draws=5 afirmava cinco "
                "realizacoes independentes e entregava uma repetida cinco vezes (AR46). Achado "
                "por REVISAO DE CODIGO e nao por olhar o numero, que e o que torna esta uma "
                "correcao de defeito e nao um ajuste de resultado."
            ),
            "train_ratio": TRAIN_RATIO,
            "rsd_variant": VARIANT,
            "rsd_variant_note": (
                "outlier_removed espelha o oraculo: RSD() do tools.R so implementa essa "
                "variante. Ela e VACUA nas duas metades deste criterio porque o retido tem "
                "n = 2 (round(7 * 0,7) = 5 de treino, 2 no alvo), entao plain daria os "
                "mesmos numeros. Declarado porque escolher a variante que casa com o "
                "oraculo e transcricao fiel, e a vacuidade e propriedade do recorte de "
                "hoje: com mais QCs no retido as duas deixam de coincidir."
            ),
            "n_sampled_features": N_SAMPLED_FEATURES,
            "feature_sampling": (
                "O subconjunto de 500 features VARIA com a semente: sampled_feature_names "
                "deriva de default_rng([seed, n_features]) (crossval.py:157). Consequencia "
                "declarada: o intervalo dos 20 absorve DUAS fontes — estocasticidade da "
                "correcao e subamostragem —, que juntas sao a incerteza total do nosso "
                "estimador contra um alvo fixo. Por isso sampling_se_pct entra como "
                "DECOMPOSICAO da largura e NAO e somado ao intervalo: somar seria alargar "
                "para conter, que e a regra 2 do README."
            ),
            "sampling_se_pct": SAMPLING_SE_PCT,
            "enforce_batch_representation": (
                "derivado do corretor e nao fixo: a restricao da linha 1263 do "
                "tools.R existe no caminho SERRF e NAO no do batch_loess, que sorteia cinco "
                "vezes sem condicao (linhas 893-903). Com lote unico a diferenca e inerte "
                "nos quatro modos — e o ramo 7 do AR9 —, mas fixa-la aqui reproduziria o "
                "oraculo em metade do criterio e o contradiria na outra."
            ),
            "containment_rule": (
                "intervalo = [min, max] das 20 medianas entre features, uma por semente. A "
                "mediana publicada do R tem de cair dentro. A LARGURA do intervalo e "
                "resultado e vai ao relatorio junto — largura grande nao e sucesso "
                "silencioso. Mediana, media e sd_ddof1 dos 20 acompanham, porque colapsar "
                "vinte valores em um apaga a informacao que separa acoes opostas."
            ),
            "comparison_note": (
                "Os dois lados nao medem a mesma populacao de features e isso e declarado: "
                "a mediana publicada e sobre todas as features definidas na coluna, a nossa "
                "sobre 500 amostradas. E o que sampling_se_pct quantifica. O que os "
                "dois lados TEM de compartilhar e o estimador — subamostragem repetida 5/2 "
                "— e e por isso que a comparacao usa qc_rsd_r_cv e nao o in-sample: contra "
                "a tabela 4.4 o in-sample daria 21 a 29 pontos de distancia com uma "
                "implementacao correta (secao 7, [CA-3.2])."
            ),
        },
        "outcome_criteria": {
            "nota": (
                "Escritos ANTES de rodar. A coluna de "
                "proibicoes e metade do valor: sem ela qualquer falha seria lida como "
                "confirmacao da suspeita a mao, porque a explicacao estaria disponivel e "
                "encaixaria num padrao conhecido (precedente AR40)."
            ),
            "casos": [
                {
                    "se": "o intervalo contem a mediana publicada",
                    "entao": "[CA-3.12] verde; a largura e o resultado",
                    "proibido": "tratar largura grande como sucesso sem reporta-la",
                },
                {
                    "se": "falha POR CIMA na metade do SERRF",
                    "entao": (
                        "DV12 e a suspeita — deficit de capacidade da floresta desloca o "
                        "intervalo EM BLOCO, e menos capacidade significa correcao menos "
                        "responsiva, logo RSD residual mais alto do nosso lado. E a "
                        "predicao direcional escrita antes desta medicao. A "
                        "saida C — reportar a divergencia como achado, com largura e "
                        "distancia — e a resposta esperada, porque intervalo deslocado "
                        "por correcao deixa de ser contencao e vira contencao depois de "
                        "corrigir na direcao que eu esperava."
                    ),
                    "proibido": "alargar o intervalo acrescentando sementes ate conter",
                },
                {
                    "se": "falha POR BAIXO na metade do SERRF",
                    "entao": (
                        "DV12 NAO e a causa e a investigacao comeca do zero. Esta linha e "
                        "metade do valor do pre-registro."
                    ),
                    "proibido": "atribuir a DV12 porque a explicacao esta a mao",
                },
                {
                    "se": "falha na metade do batch_loess, em qualquer direcao",
                    "entao": (
                        "DV15 NAO e a causa, e a mudanca de atribuicao esta registrada "
                        "abaixo em attribution_amendment. Com as faixas de seed + index "
                        "disjuntas o mecanismo de DV15 e inerte nesta medicao, entao o "
                        "estreitamento bilateral que DV15 previa deixa de estar previsto. "
                        "Falha aqui e investigacao nova, sem suspeito pre-registrado."
                    ),
                    "proibido": (
                        "atribuir a DV15 por inercia da atribuicao anterior. DV15 continua "
                        "verdadeira como defeito do codigo e continua por corrigir; o que "
                        "mudou e que ela nao alcanca ESTA medicao."
                    ),
                },
                {
                    "se": "contem, mas a largura excede a propria mediana",
                    "entao": (
                        "achado sobre o estimador e nao sobre nos. Asercao sem poder "
                        "recebe ROTULO, e o rotulo e o que diz ao leitor onde esta o dente "
                        "— precedente do [CA-3.10] e do [CA-3.2]."
                    ),
                    "proibido": "declarar verde sem nomear que o portao nao tem dente",
                },
            ],
        },
        "attribution_amendment": {
            "o_que_mudou": (
                "A atribuicao anterior era: SERRF aponta DV12 com deslocamento em bloco, "
                "batch_loess aponta DV15 com estreitamento bilateral. A metade do SERRF "
                "continua exatamente assim. A metade do batch_loess muda: com o "
                "espacamento declarado acima, DV15 fica inerte e nao ha suspeito "
                "pre-registrado para aquela metade."
            ),
            "por_que": (
                "DV15 mordia por SOBREPOSICAO de faixas de seed + index, e a sobreposicao "
                "e funcao do espacamento das sementes — que era parametro livre ate este "
                "arquivo fixa-lo. Manter sementes consecutivas testaria a predicao de DV15 "
                "e produziria, na outra ponta, um intervalo sabidamente errado: estreito "
                "por construcao, e portanto sem valor como evidencia sobre a nossa "
                "implementacao do LOESS. Gastar metade do teto de parede para reconfirmar "
                "por medicao um defeito ja estabelecido por leitura de codigo e o pior dos "
                "dois usos do orcamento."
            ),
            "o_que_isso_NAO_e": (
                "Nao e correcao de DV15: loess.py continua com seed + index, o defeito "
                "continua registrado e continua por corrigir. Nao e ajuste de regua: "
                "nenhuma tolerancia e nenhum numero esperado mudou. A condicao que destrava "
                "DV15 continua sendo a regeneracao do golden do LOESS, e a predicao "
                "direcional dela continua testavel LA, onde a correcao e a regeneracao sao "
                "um unico ato."
            ),
            "onde_isso_deixa_DV15": (
                "A frase de DV15 'falha bilateral na metade do batch_loess aponta para ca' "
                "passa a ser condicional ao espacamento das sementes."
            ),
        },
        "budget": {
            "wall_minutes_ceiling": WALL_BUDGET_MINUTES,
            "ajustes_por_modo_e_metodo": N_SEEDS * N_DRAWS,
            "ajustes_totais": N_SEEDS * N_DRAWS * len(MODES) * len(METHODS),
            "forma": (
                "Orcamento em parede e nucleo-minutos, com marcador slow "
                "explicito e jobs separados. Exige-se UM ajuste de r_cv medido antes de "
                "estimar o total, e nao estimativa: o reajuste do r_cv roda sobre a "
                "submatriz de QC — 7 colunas, nao 68 —, entao o orcamento de 5 min "
                "por ajuste do [CA-3.8] nao se transfere para ca."
            ),
            "se_estourar": (
                "Condicao de parada. Nao se reduz n_draws, nem o numero de sementes, nem se "
                "eleva o teto: as tres mudam o que o criterio afirma ou movem a regua. "
                "A saida ja tomada foi amostrar features, que preserva a quantidade "
                "e declara a precisao, e ela ja esta dentro deste orcamento."
            ),
        },
        "targets_published": targets,
        "provenance": {
            "command": "scripts/gen_r_cv_prereg.py",
            "generated_at": datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%S+0000"),
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}."
            f"{sys.version_info.micro}",
            "numpy_version": np.__version__,
            "target_source": (
                "Colunas SERRF e batch_loess dos quatro QC - RSD.csv do laboratorio. Sao "
                "artefato publicado do oraculo e nao medicao nossa — e por isso conge-las "
                "aqui nao contamina o pre-registro, congela o ALVO."
            ),
            "input_sources": sources,
        },
        "regeneration_contract": {
            "exact": [
                "preregistration",
                "outcome_criteria",
                "attribution_amendment",
                "budget",
                "targets_published",
                "provenance.input_sources",
            ],
            "note": (
                "TUDO neste arquivo e exato na regeneracao, e isso o distingue do "
                "r_serrf_divergence.json, cujo bloco medido nao pode bater por construcao. "
                "Aqui nao ha bloco medido: alvo publicado e escolha declarada sao ambos "
                "deterministicos. Regeneracao que diverge em qualquer campo e mudanca de "
                "pre-registro, e mudar pre-registro depois de medir o anula."
            ),
        },
    }


def serialize(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=False) + "\n"


# **A granularidade em que `python_version` é comparado.** O argumento inteiro está em
# `scripts/gen_insample_medians.py`; aqui fica o resumo, porque este é o SEGUNDO membro da
# classe e foi achado por varredura, **não por reprovação**: o teste que o cobre é `slow`, então
# ele reprovaria só no `make test-all`.
#
# O arquivo grava o interpretador inteiro; a comparação olha `major.minor`, que é a
# granularidade dos pinos que o projeto declara. Entre 3.12.13 e 3.12.14 a **única**
# linha divergente deste fixture era esta, e todo o resto é idêntico.
#
# `tests/test_procedencia_comparavel.py` exercita esta função e as das outras três, e é ele que
# impede as quatro cópias de divergirem.
_PATCH_DO_PYTHON = re.compile(r'("python_version":\s*"\d+\.\d+)\.\d+"')


def comparavel(raw: str) -> list[str]:
    """As linhas que o `--check` compara: sem `generated_at`, com o patch do CPython truncado."""
    return [
        _PATCH_DO_PYTHON.sub(r'\1"', linha)
        for linha in raw.splitlines()
        if '"generated_at"' not in linha
    ]


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help=(
            "compara com o fixture versionado sem escrever; `generated_at` sai da comparação "
            "e `python_version` é comparado em major.minor"
        ),
    )
    args = parser.parse_args()
    payload = build()
    text = serialize(payload)

    disjuncao = payload["preregistration"]["seed_disjointness"]
    if not disjuncao["faixas_disjuntas"]:
        print(
            "ABORTADO: as faixas de `seed + index` se tocam — menor gap entre as "
            f"{disjuncao['n_faixas']} faixas derivadas e "
            f"{disjuncao['menor_gap_entre_faixas']}, abaixo do maior n_features "
            f"{disjuncao['maior_n_features_nos_quatro_modos']}. Ajuste SEED_GAP "
            f"(hoje {disjuncao['menor_gap_entre_sementes']}) ou DRAW_SEED_GAP "
            f"(hoje {disjuncao['draw_seed_gap']}): o produto tem de comportar "
            "n_draws faixas por semente. O intervalo do batch_loess sairia "
            "estreito por construcao (DV15/AR46).",
            file=sys.stderr,
        )
        return 1

    if not args.check:
        FIXTURE.write_text(text, encoding="utf-8")
        print(
            f"escrito: {FIXTURE.relative_to(REPO)}  "
            f"({len(payload['targets_published'])} alvos, {N_SEEDS} sementes, "
            f"margem de disjuncao {disjuncao['margem']})"
        )
        return 0

    if not FIXTURE.exists():
        print(f"ausente: {FIXTURE.relative_to(REPO)}", file=sys.stderr)
        return 1

    if comparavel(FIXTURE.read_text(encoding="utf-8")) == comparavel(text):
        print("fixture idêntico à regeneração (ignorando generated_at e o patch do CPython)")
        return 0
    print("fixture DIVERGE da regeneração; rode sem --check para atualizar", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
