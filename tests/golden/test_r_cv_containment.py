"""A mediana publicada do R cai no intervalo das 20 sementes.

**Tudo que este módulo assume vem do fixture pré-registrado, e nada é literal
aqui.** As sementes, `n_draws`, `train_ratio`, a variante, `n_sampled_features` e
os oito alvos são lidos de `data/fixtures/r_cv_containment_prereg.json`, que
entrou no tronco **sozinho e antes** de qualquer execução deste teste. Repetir
qualquer um deles como constante neste arquivo criaria uma segunda fonte que
poderia divergir do pré-registro sem nada acusar — e o valor inteiro do
pré-registro é ele ser a única fonte.

**O portão é contenção, e a largura é RESULTADO e não sobra.** O alvo é a média
de cinco sorteios de dois pontos num bloco que não chama `set.seed`: exigir
proximidade a uma tolerância fixa cobraria de nós precisão que o oráculo não tem.
Se o intervalo não contiver o número publicado, a divergência é achado; se contiver
e for largo, a largura é a medida de quanto o
estimador do laboratório oscila — e a quarta linha da tabela de desfechos existe
para que largura acima da própria mediana seja **rotulada** em vez de passar como
sucesso.

**Persistir é obrigatório, e não cortesia.** `pytest` só mostra saída de teste que
**falha**, então diagnóstico que dependa de teste verde é invisível por construção
— é a quarta superfície da família. Cada célula
escreve o próprio arquivo **assim que mede**, o que também é o que torna os jobs
separáveis: um job por modo ou por método não clobbera o do vizinho.
"""

import hashlib
import itertools
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from scipy.stats import spearmanr

from refino.correction.loess import LoessCorrector
from refino.correction.serrf import SerrfCorrector
from refino.io.readers import load_matrix
from refino.metrics.crossval import DRAW_SEED_GAP, qc_rsd_r_cv, sampled_feature_names
from tests.conftest import REPO, measurements_path, read_qc_rsd_column, sample_list_path

pytestmark = [pytest.mark.golden, pytest.mark.slow]

PREREG_PATH = REPO / "data" / "fixtures" / "r_cv_containment_prereg.json"
CELLS_DIR = REPO / "out" / "ca_3_12_cells"

PREREG: dict[str, Any] = json.loads(PREREG_PATH.read_text(encoding="utf-8"))
CONFIG = PREREG["preregistration"]
ALVOS = {(t["mode"], t["method"]): t for t in PREREG["targets_published"]}
CELULAS = sorted(ALVOS)

# O corretor de cada metade. `LoessCorrector.name` é `"loess"` e não está em
# `BATCH_CONSTRAINED_CORRECTORS`, então `enforce_batch_representation` deriva
# para `False` ali e para `True` no SERRF, e é por isso que este
# módulo NÃO passa o parâmetro: passá-lo fixaria a política e reproduziria o
# oráculo em metade do critério contradizendo-o na outra.
CORRETORES = {"SERRF": lambda: SerrfCorrector(n_jobs=-1), "batch_loess": LoessCorrector}


# **Identidade da execucao, e ela e o que separa celula medida de celula velha.**
# O agregado decidia cobertura por PRESENCA DE ARQUIVO num diretorio nao
# versionado. Medido: o agregado publicado cobria sete celulas anteriores a uma
# correcao de codigo e declarava `cobertura_completa: true`. Naquele caso os
# numeros nao foram afetados — mas o mecanismo nao tinha como saber, e essa e a
# forma de passagem silenciosa.
def _head_ou_desconhecido() -> str:
    """O `HEAD` do repositório, ou um rótulo explícito quando o `git` não está no `PATH`.

    **`check=False` NÃO cobria isto, e a distinção custou o `make check` inteiro.** Ele tolera
    saída não-zero; binário **ausente** levanta `FileNotFoundError` antes de haver código de saída
    a tolerar. E como esta expressão corre em tempo de **coleta**, a ausência do `git` não degradava
    um teste: abortava a coleta e matava a execução — medido em contêiner limpo, `217 deselected,
    1 error`, com `make check` saindo em `Error 2`.

    A guarda parecia defensiva e cobria o modo de falha errado. É a pergunta na forma de
    tolerância: *a tolerância cobre o que falha, ou o que eu imaginei que falharia?*

    **Devolve rótulo nomeado e não string vazia.** `""` seria indistinguível de um `git` que
    respondeu vazio, e a identidade da execução existe justamente para separar célula medida de
    célula velha — um valor ambíguo aqui apaga a distinção que o campo carrega.
    """
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        ).stdout.strip()
    except FileNotFoundError:
        return "git-ausente-do-PATH"


IDENTIDADE = {
    "prereg_sha256": hashlib.sha256(PREREG_PATH.read_bytes()).hexdigest(),
    "head": _head_ou_desconhecido(),
}


def _escreve_atomico(destino: Path, payload: dict[str, Any]) -> None:
    """Escreve por temporario + `os.replace`, que e atomico no mesmo sistema.

    A celula leva de minutos a meia hora; interrupcao no meio de um `write_text`
    deixa arquivo PRESENTE e truncado, e o agregado o leria como celula medida ou
    morreria com `JSONDecodeError` sem dizer qual. Arquivo presente e corrompido e
    pior que ausente: ausente o agregado ja nomeia.
    """
    destino.parent.mkdir(parents=True, exist_ok=True)
    temporario = destino.with_suffix(".json.parcial")
    temporario.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporario, destino)


def desfecho_e_direcao(alvo: float, baixo: float, alto: float) -> tuple[str, str | None]:
    """Desfecho e direção, **na convenção do pré-registro e não na do leitor**.

    A convenção é do NOSSO INTERVALO e não do alvo, e é literal:
    *"nosso intervalo de 20 sementes assenta **acima** da mediana publicada, e a
    contenção falha **por cima**"*, com a linha simétrica *"se o intervalo
    assentar **abaixo** da mediana do R, a divergência do R não é a causa"*.

    | Situação | Rótulo | O que a tabela de desfechos atribui |
    |---|---|---|
    | `alvo < baixo` — nosso intervalo **acima** | `por_cima` | divergência do R é a suspeita |
    | `alvo > alto` — nosso intervalo **abaixo** | `por_baixo` | divergência do R **não** é a causa
    |

    **Esta função existe porque a primeira versão invertia os dois**, escrevendo a
    condição da perspectiva do alvo — "o alvo caiu abaixo" — em vez da do
    intervalo. Mesma situação física, rótulo oposto, e a consequência não é
    cosmética: quem lesse o artefato iria à linha **oposta** da tabela e concluiria
    "a divergência do R não é a causa" onde o pré-registro atribui exatamente o contrário.

    Nenhum teste negativo pegaria: o portão **reprovou**, corretamente, com os
    números certos. Só comparar o referente contra o documento pegou — uma detecção
    **externa** por construção, e que ninguém faz
    espontaneamente sobre um mecanismo que acabou de funcionar.
    """
    if alvo < baixo:
        return "falha", "por_cima"
    if alvo > alto:
        return "falha", "por_baixo"
    return "contem", None


def _medida(mode: str, method: str) -> dict[str, Any]:
    """As 20 medianas de uma célula, medidas e persistidas.

    Uma execução de `qc_rsd_r_cv` por semente pré-registrada. A mediana de cada
    uma é sobre as features **com RSD definido** naquela execução, e `n_defined`
    viaja junto de cada valor: mediana sobre subconjunto diferente não é a mesma
    mediana, e a política de indefinido manda declarar.
    """
    matrix = load_matrix(measurements_path(mode), sample_list_path(mode), mode)
    corrector = CORRETORES[method]()
    publicado = ALVOS[(mode, method)]

    nomes_golden, coluna_golden = read_qc_rsd_column(mode, method)
    # **A coluna golden e indexada por POSICAO abaixo, entao a igualdade de nomes
    # e pre-condicao e nao cortesia.** O irmao `test_qc_rsd_none_column.py:41` ja
    # afirma isto depois da mesma chamada; aqui faltava. Como a Spearman e
    # reportada SEM limiar por desenho, um reordenamento publicaria rho proximo de
    # zero sem nenhuma asercao reagir.
    assert tuple(matrix.feature_names) == nomes_golden, (
        f"{mode}: a ordem de features da matriz nao e a do QC - RSD.csv; "
        "indexar por posicao publicaria uma Spearman entre features diferentes"
    )
    golden_pct = coluna_golden * 100.0
    posicao = {name: i for i, name in enumerate(matrix.feature_names)}

    por_semente: list[dict[str, Any]] = []
    inicio = time.perf_counter()
    for seed in CONFIG["seeds"]:
        vetor = qc_rsd_r_cv(
            matrix,
            corrector,
            n_draws=CONFIG["n_draws"],
            train_ratio=CONFIG["train_ratio"],
            variant=CONFIG["rsd_variant"],
            seed=seed,
            n_sampled_features=CONFIG["n_sampled_features"],
        )
        nosso_pct = vetor * 100.0
        definidas = np.isfinite(nosso_pct)

        # Spearman por feature contra a coluna golden, **sem limiar** — com dois
        # pontos por sorteio o RSD por feature é dominado por ruído, e fixar um
        # limiar aqui seria inventar número indefensável. Sobre as features
        # definidas dos DOIS lados, porque `rho` sobre subconjunto diferente não
        # é o mesmo `rho`.
        amostradas = sampled_feature_names(matrix, CONFIG["n_sampled_features"], seed)
        indices = np.array([posicao[n] for n in (amostradas or matrix.feature_names)])
        par = definidas[indices] & np.isfinite(golden_pct[indices])
        rho = (
            float(spearmanr(nosso_pct[indices][par], golden_pct[indices][par]).statistic)
            if par.sum() >= 2
            else float("nan")
        )

        por_semente.append(
            {
                "seed": seed,
                "median_rsd_pct": float(np.median(nosso_pct[definidas])),
                "n_defined": int(definidas.sum()),
                "spearman_vs_golden": rho,
                "n_features_used_spearman": int(par.sum()),
            }
        )

    medianas = np.array([linha["median_rsd_pct"] for linha in por_semente])
    # **Uma mediana `nan` faria a contencao passar incondicionalmente.** `min` e
    # `max` propagam `nan`, e `alvo < nan` e `alvo > nan` sao ambos `False`, entao
    # `desfecho_e_direcao` devolveria "contem" com distancia 0. A guarda de
    # degenerescencia tambem nao pegaria: `n_valores_distintos` continua 20.
    assert np.isfinite(medianas).all(), (
        f"{mode}/{method}: semente(s) sem mediana finita — "
        f"{[x['seed'] for x in por_semente if not np.isfinite(x['median_rsd_pct'])]}. "
        "Formar intervalo com `nan` faria a contencao passar sem medir nada."
    )
    baixo, alto = float(medianas.min()), float(medianas.max())
    alvo = publicado["median_rsd_pct"]
    largura = alto - baixo
    mediana_das_20 = float(np.median(medianas))

    desfecho, direcao = desfecho_e_direcao(alvo, baixo, alto)

    rhos = np.array([linha["spearman_vs_golden"] for linha in por_semente])
    celula = {
        "criterio": "[CA-3.12]",
        "identidade_da_execucao": IDENTIDADE,
        "mode": mode,
        "method": method,
        "prereg_sha256_curto": PREREG["provenance"]["input_sources"][
            f"output/{mode}_toserrf - normalization result/QC - RSD.csv"
        ][:16],
        "alvo_publicado": publicado,
        "intervalo": {"min_pct": baixo, "max_pct": alto, "largura_pct": largura},
        "resumo_das_20": {
            "n": int(medianas.size),
            "mediana_pct": float(np.median(medianas)),
            "media_pct": float(medianas.mean()),
            "sd_ddof1_pct": float(medianas.std(ddof=1)),
            "n_valores_distintos": int(np.unique(medianas).size),
        },
        # **AS TRÊS QUANTIDADES QUE O CRITÉRIO REPORTA.** A contenção deixou
        # de ser portão: sete das oito células têm largura de 60% a 130% da mediana,
        # e conter quantidade positiva num intervalo desses é quase garantido.
        #
        # **A primeira é a que responde "concordamos?" e a segunda diz quanto ela
        # vale** — e a distinção entre as duas está AQUI e não na prosa porque é
        # exatamente ela que o `pos_hilic` tornou indispensável: aquela célula passou
        # a conter enquanto a mediana das 20 se afastava 13 pontos do alvo. Quem
        # lesse só `desfecho` leria "melhorou"; quem lê estas três lê o contrário.
        "concordancia": {
            "distancia_da_mediana_ao_publicado_pct": abs(mediana_das_20 - alvo),
            "mediana_das_20_pct": mediana_das_20,
            "publicado_pct": alvo,
            "nota": (
                "É ESTA a quantidade que responde se concordamos com o oráculo. "
                "Contenção não responde: intervalo largo contém por largura. "
                "Concordância melhor move a MEDIANA das 20 na direção do alvo; "
                "alargamento move só os EXTREMOS."
            ),
        },
        "poder": {
            "largura_relativa_pct": 100.0 * largura / alvo,
            "informativo": bool(100.0 * largura / alvo <= 20.0),
            "nota": (
                "Contenção só é informativa com intervalo abaixo de ~20% da mediana "
                "publicada. Acima disso o desfecho binário mede largura, não "
                "concordância, e `contem` não é aprovação."
            ),
        },
        "desfecho": desfecho,
        "direcao_da_falha": direcao,
        "desfecho_e_observacao_nao_aprovacao": True,
        "distancia_ao_intervalo_pct": (
            0.0 if desfecho == "contem" else min(abs(alvo - baixo), abs(alvo - alto))
        ),
        # A quarta linha da tabela de desfechos: contenção com largura acima da
        # própria mediana é achado sobre o ESTIMADOR e não sobre nós, e recebe
        # ROTULO. Sem o rótulo, verde sem dente é indistinguível de verde.
        "largura_excede_a_propria_mediana": bool(largura > alvo),
        "sampling_se_pct": CONFIG["sampling_se_pct"][mode],
        "sampling_se_nota": (
            "DECOMPOSIÇÃO da largura, não margem: o subconjunto de 500 varia com a "
            "semente, então este SE já está DENTRO do intervalo. Somá-lo ao intervalo "
            "seria alargar para conter."
        ),
        "spearman_por_feature": {
            "nota": "REPORTADA SEM LIMIAR — dois pontos por sorteio, dominada por ruído",
            "mediana": float(np.nanmedian(rhos)),
            "min": float(np.nanmin(rhos)),
            "max": float(np.nanmax(rhos)),
        },
        "por_semente": por_semente,
        "segundos_de_parede": time.perf_counter() - inicio,
    }

    CELLS_DIR.mkdir(parents=True, exist_ok=True)
    _escreve_atomico(CELLS_DIR / f"{mode}__{method}.json", celula)
    return celula


_CACHE: dict[tuple[str, str], dict[str, Any]] = {}


def medida(mode: str, method: str) -> dict[str, Any]:
    """Memoiza por célula, para que as três asserções não remeçam 20 sementes."""
    chave = (mode, method)
    if chave not in _CACHE:
        _CACHE[chave] = _medida(mode, method)
    return _CACHE[chave]


# ---------------------------------------------------------------------------
# O que o pré-registro afirma sobre si mesmo
# ---------------------------------------------------------------------------


def test_o_preregistro_ainda_descreve_o_que_este_modulo_roda() -> None:
    """Guarda contra deriva silenciosa entre o fixture e o que é executado.

    Sem ela, mudar `SAMPLED_FEATURE_COUNT` no código ou o conjunto de sementes no
    fixture faria as duas fontes discordarem e o teste continuaria verde medindo
    outra coisa: mecanismo são sobre o alvo errado.
    """
    disjuncao = CONFIG["seed_disjointness"]

    # **RECALCULADO aqui, e não lido do fixture.** A versão anterior afirmava
    # `disjuncao["faixas_disjuntas"]` — booleano que o próprio gerador escreveu —,
    # então um erro na aritmética DELE passaria por este teste sem tocar nada.
    # Asserção sobre o valor que o objeto sob teste reporta de si mesmo não é
    # verificação, é eco.
    derivadas = sorted(
        s + d * CONFIG["draw_seed_gap"] for s in CONFIG["seeds"] for d in range(CONFIG["n_draws"])
    )
    menor_gap = min(b - a for a, b in itertools.pairwise(derivadas))
    maior_n_features = max(t["n_features"] for t in PREREG["targets_published"])
    assert menor_gap >= maior_n_features, (
        f"as faixas de `seed + index` se tocam: menor gap entre as {len(derivadas)} "
        f"faixas derivadas é {menor_gap}, abaixo do maior n_features "
        f"{maior_n_features}. O intervalo do batch_loess sairia estreito por "
        "construção e mediria DV15/AR46, não a variabilidade"
    )
    assert disjuncao["faixas_disjuntas"] is True
    assert disjuncao["menor_gap_entre_faixas"] == menor_gap, (
        "o fixture reporta um gap que não é o dos próprios `seeds` que ele declara"
    )
    assert disjuncao["maior_n_features_nos_quatro_modos"] == maior_n_features

    # Paridade da constante duplicada: o gerador declara `draw_seed_gap` e o módulo
    # que mede tem o seu. Duas fontes sem teste de paridade é deriva à espera
    # — e aqui a deriva reintroduziria a deriva em silêncio.
    assert CONFIG["draw_seed_gap"] == DRAW_SEED_GAP, (
        f"o pré-registro declara draw_seed_gap={CONFIG['draw_seed_gap']} e "
        f"`refino.metrics.crossval` usa {DRAW_SEED_GAP}: os sorteios não têm as "
        "faixas que o fixture afirma"
    )
    assert len(CONFIG["seeds"]) == CONFIG["n_seeds"] == 20
    assert len(set(CONFIG["seeds"])) == 20, "semente repetida reduz o n efetivo em silêncio"
    assert set(ALVOS) == {
        (mode, method)
        for mode in ("pos_hilic", "pos_c18", "neg_hilic", "neg_c18")
        for method in ("SERRF", "batch_loess")
    }


def test_a_direcao_da_falha_usa_a_convencao_do_preregistro() -> None:
    """**O rótulo é amarrado ao DOCUMENTO, não à minha leitura dele.**

    A asserção não é "`por_cima` significa o que eu acho": ela lê o caso do
    fixture pré-registrado, confirma que é o caso em que **a divergência do R é a suspeita**, e
    afirma que a função rotula assim a situação em que o **nosso intervalo assenta
    acima** do alvo — que é a situação que a predição direcional descreve.

    Este teste existe porque a primeira versão da função invertia os dois rótulos,
    e nada no mecanismo acusava: o portão reprovou corretamente, com os números
    certos, e só o nome da direção estava trocado. A detecção é externa.
    """
    casos = {c["se"]: c for c in PREREG["outcome_criteria"]["casos"]}
    cima = next(c for se, c in casos.items() if "POR CIMA" in se and "SERRF" in se)
    baixo = next(c for se, c in casos.items() if "POR BAIXO" in se and "SERRF" in se)
    assert "DV12 e a suspeita" in cima["entao"], "o fixture mudou de convenção sob o teste"
    assert "DV12 NAO e a causa" in baixo["entao"]

    # Nosso intervalo ACIMA do alvo — a predição — é `por_cima`.
    assert desfecho_e_direcao(69.3012, 69.3894, 111.2616) == ("falha", "por_cima")
    # Nosso intervalo ABAIXO do alvo — o caso que falseia a atribuição.
    assert desfecho_e_direcao(120.0, 69.3894, 111.2616) == ("falha", "por_baixo")
    # E o caso que contém, incluindo as duas bordas fechadas.
    assert desfecho_e_direcao(80.0, 69.3894, 111.2616) == ("contem", None)
    assert desfecho_e_direcao(69.3894, 69.3894, 111.2616) == ("contem", None)
    assert desfecho_e_direcao(111.2616, 69.3894, 111.2616) == ("contem", None)


# ---------------------------------------------------------------------------
# O portão
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("mode", "method"), CELULAS)
def test_ca_3_12_reporta_concordancia_poder_e_contencao(mode: str, method: str) -> None:
    """**O critério REPORTA e não aprova.** Três quantidades por célula.

    **A contenção deixou de ser portão, e removê-la não afrouxa nada.** Sete das
    oito células têm largura de 60% a 130% da mediana publicada, e conter uma
    quantidade positiva num intervalo desses é quase garantido — o critério estava
    medindo **largura**, não concordância. A prova é a célula que passou com maior
    margem: o `pos_hilic/SERRF` foi de falha a contenção enquanto a mediana das 20
    se afastava **13 pontos** do alvo e a Spearman caía de 0,5509 para 0,5095. Um
    portão que aprova enquanto a quantidade que ele vigia se deteriora aprova
    **pelo motivo errado**.

    O que este teste afirma é que as três quantidades **existem, são finitas e
    estão persistidas** — a decisão sobre elas é de quem lê, e a contenção vai
    anotada com a largura ao lado, nunca como aprovação. O portão da unidade é o
    critério.
    """
    c = medida(mode, method)
    concordancia, poder, i = c["concordancia"], c["poder"], c["intervalo"]

    assert np.isfinite(concordancia["distancia_da_mediana_ao_publicado_pct"])
    assert np.isfinite(poder["largura_relativa_pct"])
    assert np.isfinite(c["spearman_por_feature"]["mediana"])
    assert c["desfecho"] in ("contem", "falha")
    assert c["desfecho_e_observacao_nao_aprovacao"] is True, (
        "o artefato deixou de declarar que contenção aqui é observação e não aprovação"
    )
    # A coerência entre as duas: `informativo` é derivado da largura relativa e não
    # redigido. Rótulo que discordasse do número que descreve seria pior que rótulo
    # nenhum.
    assert poder["informativo"] is (poder["largura_relativa_pct"] <= 20.0)
    assert (c["desfecho"] == "contem") is (
        i["min_pct"] <= c["alvo_publicado"]["median_rsd_pct"] <= i["max_pct"]
    )


@pytest.mark.parametrize(("mode", "method"), CELULAS)
def test_o_intervalo_nao_e_degenerado(mode: str, method: str) -> None:
    """A guarda de poder do portão, e é ela que impede verde vazio.

    Se as 20 sementes dessem a **mesma** mediana, o intervalo seria um ponto e a
    contenção só passaria por coincidência exata — mas o defeito real que isto
    pega é outro e é pior: semente que não alcança os sítios estocásticos produz
    20 execuções idênticas, e o critério passaria a afirmar algo sobre uma única
    realização enquanto o relatório diria "20 sementes": mecanismo são sobre o alvo errado.
    """
    c = medida(mode, method)
    assert c["resumo_das_20"]["n_valores_distintos"] > 1, (
        f"{mode}/{method}: as 20 sementes produziram {c['resumo_das_20']['n_valores_distintos']} "
        "mediana(s) distinta(s). A semente não está alcançando os sítios estocásticos, "
        "e o intervalo não mede variabilidade nenhuma."
    )


@pytest.mark.parametrize(("mode", "method"), CELULAS)
def test_a_largura_e_reportada_e_rotulada_quando_nao_tem_dente(mode: str, method: str) -> None:
    """**A largura é resultado, e largura sem dente recebe rótulo.**

    A asserção não é que a largura seja pequena — exigir isso seria inventar
    tolerância sobre quantidade cujo alvo é irreprodutível. É que ela esteja
    **medida e persistida**, e que o caso sem dente esteja **nomeado** no
    artefato. Precedente de critérios anteriores: asserção sem poder recebe
    rótulo, e o rótulo é o que diz ao leitor onde está o dente.
    """
    c = medida(mode, method)
    destino = CELLS_DIR / f"{mode}__{method}.json"
    lido = json.loads(destino.read_text(encoding="utf-8"))
    # O que foi medido é o que foi escrito. Sem esta igualdade, o artefato podia
    # ser de uma execução anterior e o teste afirmaria sobre ele achando que
    # afirma sobre esta — arquivo velho é a forma silenciosa de relato mudo.
    assert lido["intervalo"] == c["intervalo"], "o artefato persistido não é o desta medição"
    assert lido["intervalo"]["largura_pct"] >= 0.0
    assert np.isfinite(lido["intervalo"]["largura_pct"])
    assert lido["largura_excede_a_propria_mediana"] is (
        lido["intervalo"]["largura_pct"] > lido["alvo_publicado"]["median_rsd_pct"]
    ), "o rótulo de 'sem dente' discorda da largura que ele descreve"
    assert "nota" in lido["spearman_por_feature"], "a Spearman perdeu o aviso de que não tem limiar"
    assert len(lido["por_semente"]) == 20, "o artefato persistido perdeu semente"


# ---------------------------------------------------------------------------
# O relatório agregado
# ---------------------------------------------------------------------------


def test_o_relatorio_agregado_existe_e_nomeia_o_que_falta() -> None:
    """Agrega as células que existirem e **declara as ausentes pelo nome**.

    Jobs separados podem rodar subconjuntos, então o agregado tem de dizer o que
    viu e o que não viu. Agregado que não nomeia o ausente lê-se como cobertura
    completa, que é o modo de falha por omissão.
    """
    presentes: dict[str, Any] = {}
    obsoletas: list[str] = []
    ilegiveis: list[str] = []
    for mode, method in CELULAS:
        caminho = CELLS_DIR / f"{mode}__{method}.json"
        if not caminho.is_file():
            continue
        rotulo = f"{mode}/{method}"
        try:
            celula = json.loads(caminho.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            # Arquivo presente e truncado: nomear e seguir. Morrer aqui com
            # `JSONDecodeError` esconderia QUAL célula está corrompida.
            ilegiveis.append(rotulo)
            continue
        # **Presença de arquivo NÃO é cobertura.** Célula de outra execução tem
        # outro `prereg_sha256` ou outro `HEAD`, e contá-la faz o agregado
        # declarar completa uma cobertura que mistura código diferente.
        if celula.get("identidade_da_execucao") != IDENTIDADE:
            obsoletas.append(rotulo)
            continue
        presentes[rotulo] = celula

    ausentes = [
        f"{m}/{me}"
        for m, me in CELULAS
        if f"{m}/{me}" not in presentes and f"{m}/{me}" not in obsoletas + ilegiveis
    ]
    destino: Path = REPO / "out" / "ca_3_12_contencao.json"
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(
        json.dumps(
            {
                "criterio": "[CA-3.12]",
                "preregistro": str(PREREG_PATH.relative_to(REPO)),
                "sementes": CONFIG["seeds"],
                "celulas_medidas": sorted(presentes),
                "celulas_ausentes": ausentes,
                "celulas_obsoletas": sorted(obsoletas),
                "celulas_ilegiveis": sorted(ilegiveis),
                "identidade_da_execucao": IDENTIDADE,
                "cobertura_completa": not (ausentes or obsoletas or ilegiveis),
                "por_celula": presentes,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    lido = json.loads(destino.read_text(encoding="utf-8"))
    # **A asserção é sobre a CONTABILIDADE do agregador, não sobre quantas células
    # rodaram.** Quantas rodaram é escolha de quem selecionou os testes, e o
    # portão por célula é quem cobra contenção. O que este teste garante é que
    # medidas e ausentes particionam as oito exatamente, e que a bandeira de
    # cobertura concorda com a lista — bandeira que discordasse da própria lista
    # faria uma execução parcial se ler como completa, que é o modo de falha por
    # omissão.
    quatro = (
        set(lido["celulas_medidas"]),
        set(lido["celulas_ausentes"]),
        set(lido["celulas_obsoletas"]),
        set(lido["celulas_ilegiveis"]),
    )
    assert set().union(*quatro) == {f"{mode}/{method}" for mode, method in CELULAS}, (
        "o agregado perdeu célula: as quatro listas não particionam as oito"
    )
    assert sum(len(s) for s in quatro) == len(set().union(*quatro)), (
        "uma célula aparece em duas listas do agregado"
    )
    assert lido["cobertura_completa"] == (
        not (lido["celulas_ausentes"] or lido["celulas_obsoletas"] or lido["celulas_ilegiveis"])
    ), "a bandeira de cobertura discorda das próprias listas"


# ---------------------------------------------------------------------------
# O gerador do pré-registro é mecanismo, e mecanismo precisa rodar
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("planta", [None, "SEED_GAP = 1_000", "DRAW_SEED_GAP = 3_000"])
def test_o_gerador_do_preregistro_confere_e_aborta(planta: str | None) -> None:
    """**`--check` e a guarda de disjunção, exercitados por processo.**

    O `regeneration_contract` do fixture declara exatidão TOTAL na regeneração, e
    nada exercitava isso: `grep` sobre `tests/`, `Makefile` e CI não achava o
    gerador. Os dois geradores irmãos têm teste de subprocesso; este não tinha, e
    As duas direções estavam exercitadas só **à mão** — legítimo, mas sem proteção contra
    regressão.

    É negativo **externo** pela taxonomia: constrói cópia mutada e invoca
    subprocesso. A exigência daquela taxonomia é ter sido visto passar com a planta
    **removida**, e é isso que o caso `None` é — mesmo comando, mesma invocação,
    sem mutação, exit 0. Sem ele os dois negativos poderiam estar acusando por
    qualquer motivo.
    """
    fonte = REPO / "scripts" / "gen_r_cv_prereg.py"
    original = fonte.read_text(encoding="utf-8")

    # **A cópia mutada mora em `scripts/` e NÃO em `tmp_path`, e isto é portão inerte.** O gerador
    # resolve a raiz do repositório por `Path(__file__).resolve().parent.parent`; uma cópia em
    # `tmp_path` calcularia a raiz a partir de `tmp_path`, não acharia os dados de referência, e
    # sairia antes de alcançar a guarda — negativo verde que não exercita nada. A primeira
    # versão deste teste fez exatamente isso e reprovou por essa razão.
    mutado = fonte.with_name(f"_planta_{abs(hash(planta)) % 10**8}.py")
    try:
        if planta is None:
            script, esperado = fonte, 0
        else:
            constante = planta.split(" = ")[0]
            atual = next(
                linha for linha in original.splitlines() if linha.startswith(f"{constante} = ")
            )
            mutado.write_text(original.replace(atual, planta), encoding="utf-8")
            script, esperado = mutado, 1

        fim = subprocess.run(
            [sys.executable, str(script), "--check"],
            cwd=REPO,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
    finally:
        mutado.unlink(missing_ok=True)

    assert fim.returncode == esperado, (
        f"planta={planta!r} devia sair com {esperado} e saiu com {fim.returncode}. "
        f"stdout={fim.stdout!r} stderr={fim.stderr!r}"
    )
    if planta is not None:
        assert "ABORTADO" in fim.stderr, (
            f"saiu 1 mas não pela guarda de disjunção: stderr={fim.stderr!r}"
        )
        assert "faixas derivadas" in fim.stderr
