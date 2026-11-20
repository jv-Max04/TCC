"""A política aprendida recebe o perfil do diagnóstico com que o seu corpus foi diagnosticado.

**O corpus.** A árvore distribuída era a do corpus de oito QCs
(`data/fixtures/corpus_fase9.jsonl`), diagnosticado com a política de
`scripts/gera_corpus_fase9.py`, e se servia a ela esse perfil — e não o da tela, que o aplicativo
calcula com `LIMIARES_INICIAIS` mais o limiar de RSD por modo. A árvore foi retreinada no corpus de
7 QCs (`data/fixtures/corpus_fase9_7qcs.jsonl`), que um script de medição
gerou e diagnosticou com a política do APLICATIVO; a constante `POLITICA_DE_DIAGNOSTICO_DO_CORPUS`
passou a ser essa, e o treino, o serviço e a tela fazem a mesma conta.

**O que estes testes protegem não mudou:** a árvore recebe o perfil calculado pela política com que
o corpus foi diagnosticado, nos três sítios que a servem — a rota da recomendação, o palpite da
medição e `cli.recomenda`, que a delegação da API também usa.

> O palpite da medição deixou de consultar a árvore: o par de
> `medir-corretor` é o Palpite medido em 200 features, lido da memória no disparo. Os sítios que a
> servem passaram a dois, e o teste do terceiro afirma agora a ausência da consulta, com âncora.

**O que mudou é como se separa** (a planta tem de cair sobre dado em que as duas coisas
diferem). Com a constante igual à política do aplicativo, um sítio que chamasse o diagnóstico do
aplicativo directamente daria o mesmo perfil, e a igualdade passaria sem provar nada. Então cada
teste dos três sítios **substitui a constante** por uma política que dá outro perfil na matriz da
factory — a do corpus de oito QCs, lida de `scripts/gera_corpus_fase9.py` — e afirma que o sítio
segue a constante. A matriz é escolhida para isso: RSD mediano de 0,355 nos QCs e correlação do PC1
de 0,021, que o aplicativo declara `clean` (o limiar do `pos_hilic` é 1,0466) e a política antiga,
com limiar de 0,20, `erratic`. A primeira asserção de cada teste é essa diferença.

**E a constante é DERIVADA, não pinada, por dois caminhos:** contra a política que o
gerador de registro do corpus constrói (`comum.politica_do_aplicativo()`, a função que a P2 chama
para o `profile`), e contra o perfil que o corpus GRAVOU — regerada a matriz, o diagnóstico com a
constante tem de reproduzi-lo a seis casas. O segundo caminho é o que acusa a divergência que o
primeiro não vê: se `LIMIARES_INICIAIS` mudar, a constante e a função mudam juntas, e só o corpus
fica de outra versão.

**Por isso se regeram as 40, e a escolha foi MEDIDA.** A primeira versão deste teste regerava a
matriz 0 de cada modo e **passou com a constante antiga** nos quatro: ali as duas políticas dão o
mesmo perfil, e a planta não separava nada. A segunda regerava, por modo, a primeira matriz em que
os dois perfis gravados — `profile` e `profile_politica_antiga` — diferem; só 8 das 40 diferem
(`pos_hilic` 6 e 7, `pos_c18` 3 e 5, `neg_c18` 2, 4, 5 e 7; nenhuma no `neg_hilic`), as oito
`clean`. A revisão de código mostrou que as quatro escolhidas só liam os limiares da regra
1: as 14 matrizes cuja confiança sai de `step_sustain_min` ou de `qc_monotonicity_min` sem saturar
— 1 degrau, 4 de deriva e 9 erráticas — ficavam de fora. Regerar as 40 custa cerca de 12 s, e a
marca `slow` é para orçamento de dezenas de minutos (`pyproject.toml`), então o teste é da suíte
de cada commit.

**As âncoras são três, e nenhuma regera:** o corpus tem matrizes que separam as duas políticas, e
cada regra com limiar na confiança tem ao menos uma matriz com confiança entre 0 e 1 — sem isso a
regeração passaria sem ler aquele limiar. **E as plantas são duas, em processo:** a política do
corpus de oito QCs, e `step_sustain_min` alterado. **O que nenhuma regeração alcança** são as
guardas `max_spread` e `trajectory_movement_min`, que não entram na confiança: uma mudança delas só
aparece se mudar o desfecho de alguma das 40.
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import asdict, replace
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest
from fastapi import status
from fastapi.testclient import TestClient

from refino.api.app import build_app
from refino.decision import features
from refino.decision.artefato import PoliticaDoArtefato
from refino.decision.features import (
    POLITICA_DE_DIAGNOSTICO_DO_CORPUS,
    feature_vector,
    perfil_para_a_politica,
)
from refino.decision.labeling import Label
from refino.diagnosis.policy import LIMIARES_INICIAIS, RSD_MEDIANA_MAX_POR_MODO, DiagnosisPolicy
from refino.diagnosis.profile import diagnose
from refino.domain.diagnosis import DefectProfile
from refino.domain.matrix import RefinoMatrix
from refino.domain.synthetic import SyntheticSpec
from refino.io.readers import load_matrix
from refino.synthetic.generator import generate
from tests.conftest import measurements_path, sample_list_path
from tests.factories import _feature_name, base_spec, render_pair
from tests.reclamacao import reclama

REPO = Path(__file__).resolve().parents[1]
CORPUS = REPO / "data" / "fixtures" / "corpus_fase9_7qcs.jsonl"
COMUM = REPO / "tests" / "apoio" / "comum.py"
MODOS = ("pos_hilic", "pos_c18", "neg_hilic", "neg_c18")
N_ESPERADO = 40
CASAS = 6
MODO = "pos_hilic"
POLITICA_DO_APLICATIVO = DiagnosisPolicy(
    **LIMIARES_INICIAIS, rsd_mediana_max_por_modo=RSD_MEDIANA_MAX_POR_MODO
)


def _carrega_comum() -> ModuleType:
    """O `comum.py` dos scripts do `qz.md`, **compilado da fonte em disco** e não do `__pycache__`.

    É a forma de `tests/test_procedencia_comparavel.py` (`_carrega`), pelo motivo medido lá: o
    `SourceFileLoader` valida o `.pyc` por data e tamanho da fonte, e uma planta do mesmo tamanho
    restaurada no mesmo segundo deixaria o teste a ler bytecode que já não está no disco.

    O `comum.py` acrescenta a raiz e o `src` ao `sys.path` ao carregar, para os scripts correrem
    soltos; aqui o caminho é reposto depois, e o teste não muda o que os outros importam.
    """
    modulo = ModuleType("_comum_qz")
    modulo.__file__ = str(COMUM)
    antes = list(sys.path)
    try:
        exec(compile(COMUM.read_text(encoding="utf-8"), str(COMUM), "exec"), modulo.__dict__)
    finally:
        sys.path[:] = antes
    return modulo


@pytest.fixture(scope="module")
def linhas_do_corpus() -> list[dict[str, Any]]:
    """**Sem `skip`:** o corpus é fixture versionado, e a ausência é quebra do repositório."""
    assert CORPUS.exists(), f"corpus ausente: {CORPUS}. Ele é versionado e não falta"
    linhas = [
        json.loads(linha)
        for linha in CORPUS.read_text(encoding="utf-8").splitlines()
        if linha.strip()
    ]
    assert len(linhas) == N_ESPERADO, (
        f"o corpus tem {len(linhas)} linhas e esperava {N_ESPERADO}: a árvore distribuída foi "
        "ajustada sobre 40, e um corpus de outro tamanho não a reproduz"
    )
    return linhas


def _par() -> tuple[str, str]:
    """30 features sem deriva, dispersão log-normal de 40%, semente fixa (regra 3 do README)."""
    base = base_spec(n_features=30, n_samples=40, n_qcs=8)
    gerador = np.random.default_rng(0)
    nivel = gerador.uniform(1e3, 1e5, size=(30, 1))
    valores = nivel * np.exp(gerador.normal(0.0, 0.4, size=(30, 40)))
    espec = replace(
        base,
        values=tuple(tuple(float(round(v, 3)) for v in linha) for linha in valores),
        feature_names=tuple(_feature_name(f) for f in range(30)),
    )
    return render_pair(espec)


@pytest.fixture(scope="module")
def matriz(tmp_path_factory: pytest.TempPathFactory) -> RefinoMatrix:
    medidas, lista = _par()
    pasta = tmp_path_factory.mktemp("d558")
    # `newline=""` porque `render_pair` já emite CRLF: sem ele o Windows grava CR CR LF, o leitor vê
    # uma linha vazia entre cada linha e a fixture morre em `IndexError`. O portão de
    # `test_escrita_csv_crlf.py` não viu este sítio porque o conteúdo chega por `_par()`.
    (pasta / "measurements.csv").write_text(medidas, encoding="utf-8", newline="")
    (pasta / "Sample_List.csv").write_text(lista, encoding="utf-8", newline="")
    return load_matrix(pasta / "measurements.csv", pasta / "Sample_List.csv", MODO)


def _arquivos() -> dict[str, tuple[str, bytes, str]]:
    medidas, lista = _par()
    return {
        "measurements": ("measurements.csv", medidas.encode("utf-8"), "text/csv"),
        "sample_list": ("Sample_List.csv", lista.encode("utf-8"), "text/csv"),
    }


def _mesmo_vetor(a: DefectProfile, b: DefectProfile) -> bool:
    """A igualdade que a árvore enxerga: o vetor de características, com `nan` igual a `nan`."""
    return bool(np.array_equal(feature_vector(a), feature_vector(b), equal_nan=True))


@pytest.fixture
def constante_substituida(monkeypatch: pytest.MonkeyPatch) -> DiagnosisPolicy:
    """A constante trocada pela política do corpus de oito QCs, só neste teste.

    `perfil_para_a_politica` lê a constante do módulo no instante da chamada, então trocá-la aqui
    alcança os três sítios — desde que eles passem pela função. Um sítio que chamasse o diagnóstico
    do aplicativo directamente, ou que tivesse importado a constante por valor, ficaria de fora da
    troca, e é exactamente isso que o teste dele acusa.
    """
    antiga: DiagnosisPolicy = _carrega_comum().politica_do_corpus_gravado()
    monkeypatch.setattr(features, "POLITICA_DE_DIAGNOSTICO_DO_CORPUS", antiga)
    return antiga


def _ancora(matriz: RefinoMatrix, constante: DiagnosisPolicy) -> DefectProfile:
    """O perfil que a árvore deve receber, depois de afirmar que ele DIFERE do da tela."""
    da_constante = diagnose(matriz, policy=constante)
    da_tela = diagnose(matriz, policy=POLITICA_DO_APLICATIVO)
    assert not _mesmo_vetor(da_constante, da_tela), (
        "âncora: nesta matriz a constante substituída e a política do aplicativo dão o mesmo "
        "vetor, e a igualdade abaixo não separaria um sítio que segue a constante de um que não"
    )
    assert (da_tela.defect_type, da_constante.defect_type) == ("clean", "erratic")
    assert _mesmo_vetor(perfil_para_a_politica(matriz), da_constante), (
        "âncora: a troca da constante não alcançou `perfil_para_a_politica`"
    )
    return da_constante


@pytest.fixture
def perfis_servidos(monkeypatch: pytest.MonkeyPatch) -> list[DefectProfile]:
    """Os perfis que a árvore recebe, capturados no único sítio por onde ela os recebe."""
    recebidos: list[DefectProfile] = []
    original = PoliticaDoArtefato.recommend

    def espia(self: PoliticaDoArtefato, profile: DefectProfile) -> Any:
        recebidos.append(profile)
        return original(self, profile)

    monkeypatch.setattr(PoliticaDoArtefato, "recommend", espia)
    return recebidos


# ---------------------------------------------------------------------------
# A constante: derivada da política do gerador e do perfil que o corpus gravou
# ---------------------------------------------------------------------------


def test_a_constante_e_a_politica_com_que_o_gerador_diagnosticou_o_corpus(
    linhas_do_corpus: list[dict[str, Any]],
) -> None:
    """Cada linha do corpus diz com que política o `profile` foi calculado, e a P2 a constrói.

    `"politica_do_profile": "aplicativo"` é o que a P2 grava, e `comum.politica_do_aplicativo()` é
    a função que ela chama para esse perfil. A constante tem de ser essa — e não a do corpus de oito
    QCs, que é a âncora de que a troca mudou alguma coisa.
    """
    rotulos = {r.get("politica_do_profile") for r in linhas_do_corpus}
    assert rotulos == {"aplicativo"}, (
        f"o corpus declara o `profile` calculado com {sorted(map(str, rotulos))}, e a constante "
        "só sabe ser a política do aplicativo"
    )
    comum = _carrega_comum()
    assert asdict(POLITICA_DE_DIAGNOSTICO_DO_CORPUS) == asdict(comum.politica_do_aplicativo()), (
        "a constante divergiu da política com que o gerador de registro diagnosticou o corpus"
    )
    antiga = asdict(comum.politica_do_corpus_gravado())
    assert antiga != asdict(POLITICA_DE_DIAGNOSTICO_DO_CORPUS), (
        "âncora: a política do corpus de oito QCs é a constante, e não haveria diferença a medir"
    )


def _perfil_gravado_bate(gravado: dict[str, Any], regerado: DefectProfile) -> list[str]:
    """As chaves do `profile` gravado que o perfil regerado NÃO reproduz a seis casas."""
    divergentes: list[str] = []
    for chave, valor in gravado.items():
        atual = getattr(regerado, chave)
        if isinstance(valor, float):
            # `nan` contra `nan` é igual, e `nan` de um lado só é divergência — sem isto a
            # subtração dá `nan`, a comparação dá `False`, e o perfil indefinido passaria calado.
            arredondado = round(float(atual), CASAS)
            if math.isnan(valor) or math.isnan(arredondado):
                if not (math.isnan(valor) and math.isnan(arredondado)):
                    divergentes.append(f"{chave}: {arredondado} != {valor}")
            elif abs(arredondado - valor) > 0.5 * 10**-CASAS + 1e-12:
                divergentes.append(f"{chave}: {arredondado} != {valor}")
        elif isinstance(valor, list):
            if list(atual) != valor:
                divergentes.append(f"{chave}: {list(atual)} != {valor}")
        elif atual != valor:
            divergentes.append(f"{chave}: {atual!r} != {valor!r}")
    return divergentes


def _matrizes_que_separam(linhas: list[dict[str, Any]]) -> list[str]:
    """As matrizes do corpus cujos dois perfis gravados — aplicativo e corpus antigo — diferem."""
    return [
        f"{r['modo']} #{r['indice']}"
        for r in linhas
        if r["profile"] != r["profile_politica_antiga"]
    ]


def _linha_nao_saturada(linhas: list[dict[str, Any]], tipo: str) -> dict[str, Any] | None:
    """A primeira matriz do tipo com confiança gravada estritamente entre 0 e 1, ou `None`.

    A confiança de uma regra é a distância relativa do valor medido ao limiar daquela regra,
    saturada em 1 (`profile._margem`). Saturada, ela não muda quando o limiar muda, e a regeração
    não acusaria nada; entre 0 e 1, ela depende dos limiares da regra que decidiu.
    """
    candidatas = sorted(
        (r for r in linhas if r["profile"]["defect_type"] == tipo),
        key=lambda r: (MODOS.index(r["modo"]), int(r["indice"])),
    )
    return next((r for r in candidatas if 0.0 < float(r["profile"]["confidence"]) < 1.0), None)


def _divergencias_da_regeracao(
    linha: dict[str, Any], politica: DiagnosisPolicy, *, gravado: str = "profile"
) -> list[str]:
    """Regera a matriz da linha com o gerador de hoje, diagnostica com `politica` e compara.

    **A política entra por argumento**, e é isso que deixa as plantas serem em processo: a planta é
    argumento da função sob teste e não pode ficar inerte. `gravado` diz contra qual dos dois
    perfis da linha se compara. Precisa dos arquivos do laboratório; sem eles o resolvedor pula com
    a mensagem de sempre.
    """
    assert len(linha[gravado]) == 13, "âncora: o perfil gravado não tem as treze quantidades"
    modo = linha["modo"]
    fonte = load_matrix(measurements_path(modo), sample_list_path(modo), mode=modo)
    ds = generate(
        SyntheticSpec(
            imitated_mode=modo,  # type: ignore[arg-type]
            drift_to_noise_ratio=float(linha["deriva_pedida"]),
            n_features=int(linha["profile"]["n_features"]),
            seed=int(linha["seed"]),
        ),
        fonte,
    )
    return _perfil_gravado_bate(linha[gravado], diagnose(ds.matrix, policy=politica))


def test_o_corpus_tem_matrizes_que_separam_as_duas_politicas(
    linhas_do_corpus: list[dict[str, Any]],
) -> None:
    """Âncora da regeração: sem matriz que separe, ela passaria com a constante de oito QCs."""
    assert _matrizes_que_separam(linhas_do_corpus), (
        "nenhuma matriz do corpus tem perfis diferentes sob as duas políticas, e a regeração "
        "passaria com a constante do corpus de oito QCs"
    )


def test_a_ancora_da_separacao_reprova_num_corpus_em_que_as_politicas_coincidem(
    linhas_do_corpus: list[dict[str, Any]],
) -> None:
    """Par causal EM PROCESSO da âncora acima: com os dois perfis iguais, ela não acha nenhuma."""
    coincidentes = [{**r, "profile_politica_antiga": r["profile"]} for r in linhas_do_corpus]
    assert _matrizes_que_separam(linhas_do_corpus), "âncora: o corpus real tem de separar"
    assert _matrizes_que_separam(coincidentes) == [], (
        "com os dois perfis gravados iguais a âncora ainda achou matriz que separa, e não "
        "consegue reprovar"
    )


TIPOS_COM_LIMIAR = ("clean", "step_change", "gradual_drift", "erratic")


@pytest.mark.parametrize("tipo", TIPOS_COM_LIMIAR)
def test_o_corpus_le_os_limiares_de_cada_regra(
    tipo: str, linhas_do_corpus: list[dict[str, Any]]
) -> None:
    """Âncora: cada regra com limiar na confiança tem uma matriz com confiança entre 0 e 1.

    Sem ela, a regeração das 40 passaria sem ler os limiares daquela regra — a confiança saturada
    não muda quando o limiar muda.
    """
    assert _linha_nao_saturada(linhas_do_corpus, tipo) is not None, (
        f"o corpus não tem matriz `{tipo}` com confiança entre 0 e 1, e os limiares dessa regra "
        "ficariam sem conferência"
    )


def test_o_corpus_inteiro_se_reproduz_com_a_constante(
    linhas_do_corpus: list[dict[str, Any]],
) -> None:
    """As 40 matrizes regeradas, diagnosticadas com a constante, dão o perfil que o corpus gravou.

    É a conferência que o primeiro teste não faz: lá a constante e a função do gerador derivam das
    mesmas constantes de `refino.diagnosis.policy`, e mudariam juntas. Aqui o referente é o que o
    corpus gravou, e uma constante de outra versão — ou um gerador de outra versão — não o reproduz.
    """
    divergentes = {
        f"{r['modo']} #{r['indice']}": d[:4]
        for r in linhas_do_corpus
        if (d := _divergencias_da_regeracao(r, features.POLITICA_DE_DIAGNOSTICO_DO_CORPUS))
    }
    assert not divergentes, (
        f"{len(divergentes)} das {N_ESPERADO} matrizes regeradas não reproduzem o perfil gravado: "
        f"{divergentes}. A constante, o gerador ou o corpus são de versões diferentes"
    )


def test_a_regeracao_acusa_a_politica_do_corpus_de_oito_qcs(
    linhas_do_corpus: list[dict[str, Any]],
) -> None:
    """Par causal EM PROCESSO: com a política antiga, as matrizes que separam deixam de bater.

    E, regeradas com ela, reproduzem o OUTRO perfil gravado — o que prova que a divergência é da
    política e não de uma regeração que deixou de reproduzir o gerador.
    """
    antiga: DiagnosisPolicy = _carrega_comum().politica_do_corpus_gravado()
    separam = [r for r in linhas_do_corpus if r["profile"] != r["profile_politica_antiga"]]
    assert separam, "âncora: o corpus não tem matriz que separe as duas políticas"
    for r in separam:
        rotulo = f"{r['modo']} #{r['indice']}"
        assert _divergencias_da_regeracao(r, antiga), (
            f"{rotulo}: a política antiga reproduziu o perfil do aplicativo, e a regeração não "
            "separa as duas"
        )
        assert _divergencias_da_regeracao(r, antiga, gravado="profile_politica_antiga") == [], (
            f"{rotulo}: a política antiga não reproduz o perfil que o corpus gravou para ela"
        )


def test_a_regeracao_acusa_um_limiar_fora_da_regra_1(
    linhas_do_corpus: list[dict[str, Any]],
) -> None:
    """Par causal EM PROCESSO: `step_sustain_min` alterado, e a matriz em degrau deixa de bater.

    A planta cai sobre dado em que as duas coisas diferem: a matriz em degrau tem confiança
    0,643, que é a distância relativa ao limiar dessa regra, e as oito matrizes que separam as duas
    políticas são `clean` e não a leem.
    """
    linha = _linha_nao_saturada(linhas_do_corpus, "step_change")
    assert linha is not None, "âncora: o corpus não tem matriz em degrau com confiança não saturada"
    constante = features.POLITICA_DE_DIAGNOSTICO_DO_CORPUS
    plantada = replace(constante, step_sustain_min=constante.step_sustain_min + 0.5)
    assert _divergencias_da_regeracao(linha, constante) == [], "âncora: sem a planta, bate"
    assert _divergencias_da_regeracao(linha, plantada), (
        "o limiar da regra do degrau mudou e a regeração não acusou"
    )


# ---------------------------------------------------------------------------
# Os três sítios: cada um segue a constante, e não o diagnóstico da tela
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("ativa", "consultas"), [(False, 1), (True, 2)])
def test_a_rota_da_recomendacao_serve_o_perfil_da_constante(
    matriz: RefinoMatrix,
    perfis_servidos: list[DefectProfile],
    constante_substituida: DiagnosisPolicy,
    monkeypatch: pytest.MonkeyPatch,
    ativa: bool,
    consultas: int,
) -> None:
    from refino.io import politica

    monkeypatch.setattr(politica, "SEGUNDA_OPINIAO_ATIVA", ativa)
    esperado = _ancora(matriz, constante_substituida)
    resposta = TestClient(build_app()).post(
        "/matrizes/recomendar", files=_arquivos(), data={"mode": MODO}
    )
    assert resposta.status_code == status.HTTP_200_OK, resposta.text
    # **Uma consulta por opinião** — a pelo RSD e, ligada, a pela verdade —, e todas têm de
    # receber o perfil da constante: as duas árvores foram ajustadas sobre ele. **A
    # segunda está em stand-by**, e desligada a rota consulta uma árvore só.
    assert len(perfis_servidos) == consultas, (
        f"as árvores foram consultadas {len(perfis_servidos)} vezes, e esperava {consultas}"
    )
    assert all(_mesmo_vetor(p, esperado) for p in perfis_servidos), (
        "a recomendação serviu a uma das árvores um perfil que não é o da constante"
    )


def test_a_medicao_dos_corretores_nao_consulta_a_arvore(
    matriz: RefinoMatrix,
    perfis_servidos: list[DefectProfile],
    constante_substituida: DiagnosisPolicy,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """O par que a medição grava ao lado do vencedor medido — a metade «antes».

    Este teste afirmava que o palpite da medição consultava a árvore com o perfil da
    constante. Agora o par é o Palpite MEDIDO, lido da memória no disparo, e sem ele o par é
    nulo: a medição não consulta a árvore. **A ausência tem âncora:** o espião conta a consulta do
    `--auto` logo a seguir, então ele estava ligado durante a medição.
    """
    from refino.api import rotas
    from refino.cli import AUTO_APRENDIDA, recomenda

    def oraculo(self: Any, matrix: Any, **_kw: Any) -> Label:
        return Label(
            prefix=("none", 1),
            scores={("none", 1): 0.1, ("loess", 1): 0.2, ("serrf", 1): 0.3},
            estimator="leave_one_qc_out",
            variant="outlier_removed",
        )

    monkeypatch.setattr(rotas.LabelingOracle, "label", oraculo)
    monkeypatch.setattr(rotas, "_PALPITES", type(rotas._PALPITES)())
    esperado = _ancora(matriz, constante_substituida)
    resposta = reclama(
        TestClient(build_app()),
        "/matrizes/medir-corretor",
        files=_arquivos(),
        data={"mode": MODO},
        teto_s=120,
    )
    assert resposta.status_code == status.HTTP_200_OK, resposta.text
    assert resposta.json()["palpite"] is None, "sem Palpite guardado, o par da medição é nulo"
    assert perfis_servidos == [], "a medição consultou a árvore"
    recomenda(matriz, AUTO_APRENDIDA)
    assert len(perfis_servidos) == 1 and _mesmo_vetor(perfis_servidos[0], esperado), (
        "âncora: o espião não contou a consulta do `--auto`, e a ausência acima não prova nada"
    )


def test_a_cli_e_a_delegacao_usam_o_perfil_da_constante(
    matriz: RefinoMatrix,
    perfis_servidos: list[DefectProfile],
    constante_substituida: DiagnosisPolicy,
) -> None:
    """`cli.recomenda` serve o `--auto` e a delegação da API (`_delegacao_ou_422`)."""
    from refino.cli import AUTO_APRENDIDA, recomenda

    esperado = _ancora(matriz, constante_substituida)
    recomenda(matriz, AUTO_APRENDIDA)
    assert len(perfis_servidos) == 1 and _mesmo_vetor(perfis_servidos[0], esperado), (
        "o --auto consultou a árvore com outro perfil que o da constante"
    )


def test_a_segunda_opiniao_da_cli_usa_o_perfil_da_constante(
    matriz: RefinoMatrix,
    perfis_servidos: list[DefectProfile],
    constante_substituida: DiagnosisPolicy,
) -> None:
    """`cli.segunda_opiniao` é o quarto sítio, e a irmã de `recomenda`.

    O Palpite pela verdade foi ajustado sobre o mesmo perfil que o pelo RSD — o corpus pela verdade
    carrega os perfis das matrizes do plano, calculados com a política do aplicativo —, então ele
    também tem de receber o perfil da constante, e não o do diagnóstico da tela.
    """
    from refino.cli import segunda_opiniao

    esperado = _ancora(matriz, constante_substituida)
    segunda_opiniao(matriz)
    assert len(perfis_servidos) == 1 and _mesmo_vetor(perfis_servidos[0], esperado), (
        "a segunda opinião consultou a árvore com outro perfil que o da constante"
    )
