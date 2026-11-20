"""O artefato da política aprendida, e os dois portões que o impedem de envelhecer.

## Por que o artefato existe

`LearnedPolicy` só era ajustada dentro de três scripts de medição, sobre o corpus de 40 matrizes.
**Não havia modelo persistido em lugar nenhum**, e `recommend()` levanta se a política não estiver
ajustada — então o `--auto` na política aprendida não tinha o que delegar.

## Os DOIS portões, e a divisão é deliberada

| Portão | O que afirma | Onde corre |
|---|---|---|
| **barato** | o JSON é legível, tem procedência, e a ordem de | portão de commit |
| | características é a que o código espera | |
| **concordância** | o artefato reproduz, nos 40 perfis, a recomendação | portão de commit, |
| | da política ajustada na hora | **medido barato** |

**O segundo era previsto como `slow` e não é**, e a medição desfez a suposição: o corpus é um
fixture de 40 linhas — `data/fixtures/corpus_fase9_7qcs.jsonl` agora, como era
`data/fixtures/corpus_fase9.jsonl` até ali —, não uma regeneração de matrizes sintéticas. Um
artefato cuja única verificação corre em alvo que ninguém roda por rotina é um portão inerte à
espera de acontecer, e a medição evitou construir exactamente isso.

## O par causal, e ele é o que separa este portão de decoração

A concordância sozinha passa com um artefato **certo**. O que prova que ela **consegue** reprovar é
adulterar um limiar e ver a recomendação divergir — e é por isso que
`test_um_limiar_adulterado_faz_a_concordancia_reprovar` existe, e por que ele afirma que a planta
cai sobre dado em que as duas coisas diferem: o limiar escolhido é o da **raiz**, que toda
entrada avalia.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from refino.decision.artefato import (
    ArtefatoInvalidoError,
    PoliticaDoArtefato,
    exporta,
)
from refino.decision.features import FEATURE_NAMES
from refino.decision.model import LearnedPolicy

REPO = Path(__file__).resolve().parents[1]
# **O corpus do modelo distribuído, e ele mudou.** O de oito QCs
# (`corpus_fase9.jsonl`) fica no repositório porque os números antigos se reproduzem dele; o de 7
# QCs com a floresta `scikit_learn` (`corpus_fase9_7qcs.jsonl`) fica porque o Palpite antigo,
# guardado em `ANTIGO`, se reproduz dele. O artefato é treinado nos mesmos 40 perfis, com os
# rótulos da floresta que conta como o `ranger`, que passou a ser o padrão do SERRF.
CORPUS = REPO / "data" / "fixtures" / "corpus_fase9_7qcs_ranger.jsonl"
ARTEFATO = REPO / "src" / "refino" / "decision" / "politica_aprendida.json"
CORPUS_ANTIGO = REPO / "data" / "fixtures" / "corpus_fase9_7qcs.jsonl"
ANTIGO = REPO / "data" / "fixtures" / "politica_aprendida_scikit_learn.json"

# A semente do artefato. **Não é escolha livre:** é a mesma que os scripts de medição
# usam, e mudá-la produziria um modelo diferente do que aquelas medições reportaram.
SEMENTE = 42

# O corpus tem 40 matrizes sintéticas, 10 derivadas de cada modo. O número vai aqui e é AFIRMADO,
# porque uma comparação sobre corpus vazio passaria por vacuidade. **Esta linha dizia «4
# reais e 36 sintéticas»**, e as 40 linhas de qualquer dos dois corpora têm semente e
# deriva pedida: são todas sintéticas.
N_ESPERADO = 40


def _carrega_corpus(caminho: Path) -> tuple[list[Any], list[str], list[float]]:
    """Perfis, rótulos e margens do corpus. Espelha `carrega` de `scripts/mede_b_fase9.py`.

    **Duplicado de propósito e não importado:** `scripts/` não é pacote e o teste não pode
    depender de `sys.path` manipulado por um script. O que impede as duas cópias de divergirem é
    `test_o_corpus_traz_as_treze_quantidades`, que afirma o conjunto de chaves lido.
    """
    from refino.domain.diagnosis import DefectProfile

    perfis: list[Any] = []
    rotulos: list[str] = []
    margens: list[float] = []
    for linha in caminho.read_text(encoding="utf-8").splitlines():
        if not linha.strip():
            continue
        registro = json.loads(linha)
        p = registro["profile"]
        perfis.append(
            DefectProfile(
                defect_type=p["defect_type"],
                confidence=p["confidence"],
                suggested_batches=tuple(p["suggested_batches"]),
                changepoints=tuple(p["changepoints"]),
                qc_rsd_median=p["qc_rsd_median"],
                qc_rsd_p75=p["qc_rsd_p75"],
                zero_fraction=p["zero_fraction"],
                missing_fraction=p["missing_fraction"],
                pc1_injection_correlation=p["pc1_injection_correlation"],
                qc_dispersion_ratio=p["qc_dispersion_ratio"],
                n_features=p["n_features"],
                n_samples=p["n_samples"],
                n_qcs=p["n_qcs"],
            )
        )
        rotulos.append(registro["rotulo"])
        margens.append(float(registro["margem"]))
    return perfis, rotulos, margens


@pytest.fixture(scope="module")
def corpus() -> tuple[list[Any], list[str], list[float]]:
    """**Sem `skip` de propósito, e a ausência é a decisão.**

    A primeira versão pulava quando o corpus faltasse. Mas `corpus_fase9.jsonl` é fixture
    **versionado** — ele não falta —, e um `skip` que nunca dispara não protege nada: ele só sabe
    absorver outra falha, e é a forma em que exercitar o mecanismo
    NÃO revela o defeito. Corpus ausente aqui é quebra do repositório, e tem de falhar alto.

    O portão de política de `skip` acusou isto antes de mim: a mensagem não casava regime nenhum,
    e a regra do projeto diz o que isso é — *"falha à espera da condição externa"*, que passa em
    toda máquina onde a condição não ocorre.
    """
    assert CORPUS.exists(), (
        f"corpus ausente: {CORPUS}. Ele é versionado — a ausência é quebra do repositório, e não "
        "uma condição de ambiente a pular"
    )
    perfis, rotulos, margens = _carrega_corpus(CORPUS)
    assert len(perfis) == N_ESPERADO, (
        f"o corpus tem {len(perfis)} perfis e esperava {N_ESPERADO}. Isto é ACHADO: as medições da "
        "Fase 9 foram feitas sobre 40, e um corpus de outro tamanho não as reproduz"
    )
    return perfis, rotulos, margens


@pytest.fixture(scope="module")
def ajustada(corpus: tuple[list[Any], list[str], list[float]]) -> LearnedPolicy:
    perfis, rotulos, margens = corpus
    return LearnedPolicy(seed=SEMENTE).fit(perfis, rotulos, margens)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# O portão BARATO — forma, procedência e ordem das características
# ---------------------------------------------------------------------------


def test_o_artefato_existe_e_e_json_legivel() -> None:
    """Sem isto, todos os outros deste arquivo falham por um motivo que não é o deles."""
    assert ARTEFATO.exists(), (
        f"artefato ausente: {ARTEFATO}. Gere-o com `python scripts/gera_politica_aprendida.py` — "
        "ele é versionado, não construído no arranque"
    )
    dados = json.loads(ARTEFATO.read_text(encoding="utf-8"))
    assert isinstance(dados, dict) and dados, "o artefato está vazio"


def test_o_artefato_declara_a_ordem_das_caracteristicas_que_o_codigo_espera() -> None:
    """A ordem é o contrato, e é ela que torna o vetor interpretável fora do `scikit-learn`.

    **Um artefato com a ordem de ontem produziria recomendação plausível e errada**, porque o
    índice de característica de cada nó passaria a apontar para outra coluna. É a forma que a
    caducidade toma neste artefato: nada estaria "errado", e o referente teria mudado por baixo.
    """
    dados = json.loads(ARTEFATO.read_text(encoding="utf-8"))
    assert tuple(dados["caracteristicas"]) == FEATURE_NAMES, (
        "a ordem das características do artefato diverge de `FEATURE_NAMES`. REGERE o artefato; "
        "não ajuste este teste"
    )


def test_o_artefato_carrega_a_procedencia() -> None:
    """Artefato sem procedência é número sem a população que o produziu."""
    dados = json.loads(ARTEFATO.read_text(encoding="utf-8"))
    procedencia = dados["procedencia"]
    assert procedencia["semente"] == SEMENTE
    assert procedencia["n_corpus"] == N_ESPERADO
    assert procedencia["corpus"] == CORPUS.name, (
        f"o artefato declara o corpus {procedencia['corpus']!r} e este portão confere contra "
        f"{CORPUS.name!r}: REGERE o artefato a partir do corpus de 7 QCs da floresta do R"
    )
    assert procedencia["gerado_em"], "a data de geração está vazia"


def test_o_corpus_traz_as_treze_quantidades() -> None:
    """Âncora do leitor duplicado: se o corpus mudar de forma, este teste acusa antes do resto."""
    primeira = json.loads(CORPUS.read_text(encoding="utf-8").splitlines()[0])
    assert set(primeira) >= {"profile", "rotulo", "margem", "modo", "assessable"}
    assert len(primeira["profile"]) == 13, (
        f"o perfil do corpus tem {len(primeira['profile'])} quantidades e o leitor deste arquivo "
        "reconstrói 13"
    )


# ---------------------------------------------------------------------------
# O portão de CONCORDÂNCIA — e o par causal que prova que ele reprova
# ---------------------------------------------------------------------------


def test_o_artefato_reproduz_a_politica_ajustada_nos_quarenta_perfis(
    corpus: tuple[list[Any], list[str], list[float]], ajustada: LearnedPolicy
) -> None:
    """O portão que impede o artefato de envelhecer em silêncio.

    Compara **recomendação a recomendação**, e não só o corretor: confiança e justificativa são o
    que os critérios de justificativa entregam, e um artefato que acertasse o corretor com a
    confiança errada passaria por um portão que só olhasse a classe.
    """
    perfis, _, _ = corpus
    do_artefato = PoliticaDoArtefato.de_dict(json.loads(ARTEFATO.read_text(encoding="utf-8")))

    assert perfis, "âncora: o corpus está vazio e a comparação abaixo seria vácua"
    divergentes = []
    for i, perfil in enumerate(perfis):
        a = do_artefato.recommend(perfil)
        b = ajustada.recommend(perfil)
        if (a.correction, round(a.confidence, 9), a.rationale) != (
            b.correction,
            round(b.confidence, 9),
            b.rationale,
        ):
            divergentes.append((i, a.correction, b.correction))

    assert not divergentes, (
        f"{len(divergentes)} de {len(perfis)} perfis divergem entre o artefato e o treino em "
        f"memória: {divergentes[:5]}. O artefato é de outra versão do caminho que o produz — "
        "REGERE com `python scripts/gera_politica_aprendida.py`, não ajuste o limiar"
    )


def test_um_limiar_adulterado_faz_a_concordancia_reprovar(
    corpus: tuple[list[Any], list[str], list[float]], ajustada: LearnedPolicy
) -> None:
    """O par causal. **Sem ele o portão acima é indistinguível de um que nunca reprova.**

    A planta cai na **raiz** da árvore do corretor, que toda entrada avalia — e o teste AFIRMA que
    a raiz é um nó de decisão antes de plantar, senão a mutação cairia numa folha e não separaria
    nada.
    """
    dados = json.loads(ARTEFATO.read_text(encoding="utf-8"))
    raiz = dados["corretor"]["nos"][0]
    assert raiz.get("folha") is not True, (
        "âncora: a raiz da árvore do corretor é uma folha, então não há limiar a adulterar e esta "
        "planta não separaria nada"
    )

    dados["corretor"]["nos"][0]["limiar"] = float(raiz["limiar"]) + 1_000_000.0
    adulterada = PoliticaDoArtefato.de_dict(dados)

    perfis, _, _ = corpus
    divergiu = any(
        adulterada.recommend(p).correction != ajustada.recommend(p).correction for p in perfis
    )
    assert divergiu, (
        "adulterar o limiar da RAIZ não mudou recomendação nenhuma nos 40 perfis — o portão de "
        "concordância não consegue reprovar, e passar nele não prova nada"
    )


def test_duas_leituras_do_mesmo_arquivo_dao_a_mesma_recomendacao(
    corpus: tuple[list[Any], list[str], list[float]],
) -> None:
    """Reprodutibilidade na fronteira do artefato: o caminho de leitura não pode ter sorteio."""
    perfis, _, _ = corpus
    assert perfis, "âncora: o corpus está vazio e o laço abaixo não compararia nada"
    bruto = ARTEFATO.read_text(encoding="utf-8")
    uma = PoliticaDoArtefato.de_dict(json.loads(bruto))
    outra = PoliticaDoArtefato.de_dict(json.loads(bruto))
    for perfil in perfis:
        a, b = uma.recommend(perfil), outra.recommend(perfil)
        assert (a.correction, a.confidence, a.rationale) == (
            b.correction,
            b.confidence,
            b.rationale,
        )


# ---------------------------------------------------------------------------
# Os desfechos de ERRO, e nenhum deles pode cair num padrão razoável
# ---------------------------------------------------------------------------


def test_artefato_vazio_levanta_com_nome(corpus: tuple[list[Any], list[str], list[float]]) -> None:
    with pytest.raises(ArtefatoInvalidoError, match="vazio"):
        PoliticaDoArtefato.de_dict({})


def test_ordem_de_caracteristicas_divergente_levanta_em_vez_de_recomendar(
    corpus: tuple[list[Any], list[str], list[float]],
) -> None:
    """**O modo de falha mais insidioso desta unidade**, e por isso ele levanta.

    Uma ordem trocada não produz erro em lugar nenhum: cada nó continua a comparar um número com
    um limiar, e a recomendação sai plausível. Valor inválido
    que cai num `else` razoável comporta-se como outra política.
    """
    dados = json.loads(ARTEFATO.read_text(encoding="utf-8"))
    dados["caracteristicas"] = list(reversed(dados["caracteristicas"]))
    with pytest.raises(ArtefatoInvalidoError, match="ordem"):
        PoliticaDoArtefato.de_dict(dados)


def test_artefato_sem_a_arvore_do_corretor_levanta(
    corpus: tuple[list[Any], list[str], list[float]],
) -> None:
    dados = json.loads(ARTEFATO.read_text(encoding="utf-8"))
    del dados["corretor"]
    with pytest.raises(ArtefatoInvalidoError, match="corretor"):
        PoliticaDoArtefato.de_dict(dados)


def test_folha_com_rotulo_fora_do_dominio_levanta_no_carregamento(
    corpus: tuple[list[Any], list[str], list[float]],
) -> None:
    """A guarda do valor inválido, e o teste afirma **onde** ela dispara.

    `Correction` é `Literal`, que o mypy prova nas chamadas internas e não alcança um rótulo vindo
    de disco. A guarda tem de estar no carregamento e não no ponto de uso: um rótulo inválido só
    apareceria três camadas adiante, como uma correção que nenhum corretor conhece.

    **A âncora vem antes da planta:** o artefato tem de ter ao menos uma folha, senão adulterar
    "uma folha" não adulteraria nada.
    """
    dados = json.loads(ARTEFATO.read_text(encoding="utf-8"))
    folhas = [no for no in dados["corretor"]["nos"] if no.get("folha")]
    assert folhas, "âncora: a árvore do corretor não tem folha nenhuma e a planta seria vácua"

    folhas[0]["valor"] = "xpto"
    with pytest.raises(ArtefatoInvalidoError, match="fora do domínio"):
        PoliticaDoArtefato.de_dict(dados)


def test_o_dominio_valido_e_o_do_repositorio_e_nao_uma_lista_a_mao() -> None:
    """O que a guarda aceita sai de `get_args(Correction)`, e este teste o afirma.

    Uma lista escrita à mão no módulo do artefato divergiria de `Correction` no dia em que um
    corretor novo entrasse — e o artefato recusaria um rótulo legítimo, que é o pior desfecho de
    uma guarda: reprovar o certo.
    """
    from typing import get_args

    from refino.domain.labels import Correction

    dados = json.loads(ARTEFATO.read_text(encoding="utf-8"))
    rotulos = {no["valor"] for no in dados["corretor"]["nos"] if no.get("folha")}
    validos = set(get_args(Correction))
    assert rotulos, "âncora: nenhuma folha lida"
    assert rotulos <= validos, (
        f"o artefato traz rótulos que o domínio não declara: {sorted(rotulos - validos)}"
    )


# ---------------------------------------------------------------------------
# A exportação, do lado do produtor
# ---------------------------------------------------------------------------


def test_a_justificativa_tem_a_forma_que_o_ca_9_6_pede(
    corpus: tuple[list[Any], list[str], list[float]], ajustada: LearnedPolicy
) -> None:
    """Âncora INDEPENDENTE do refactor, e ela existe por um vão que o portão vizinho tem.

    `test_o_artefato_reproduz_a_politica_ajustada_nos_quarenta_perfis` compara duas políticas que
    **partilham** `monta_recomendacao` — a função extraída para o texto não viver em dois lugares.
    Uma quebra dentro dela apareceria nos dois lados igualmente, e aquele teste passaria.

    É a pergunta aplicada ao meu próprio portão: *o que este mecanismo afirma é o que eu
    preciso que seja verdade?* Ali a resposta era **não** — ele afirma que os dois concordam, e o
    que eu preciso é que a justificativa continue a ser a que o critério entrega. Este teste
    afirma a forma contra literais, que nenhum dos dois caminhos pode reescrever sem reprovar.
    """
    perfis, _, _ = corpus
    do_artefato = PoliticaDoArtefato.de_dict(json.loads(ARTEFATO.read_text(encoding="utf-8")))
    assert perfis, "âncora: corpus vazio"

    for politica in (ajustada, do_artefato):
        razao = politica.recommend(perfis[0]).rationale
        assert razao.startswith("Recomendado `"), razao[:80]
        assert "**Por quê:**" in razao
        assert "A margem relativa PREDITA entre o melhor e o segundo corretor é" in razao
        assert "O RSD mediano de QC é" in razao
        assert "a correlação de PC1 com a ordem de injeção é" in razao
        # As duas componentes que o critério pede NOMEADAS, e não só o veredito.
        assert "≤" in razao or ">" in razao, (
            "a justificativa não traz condição nenhuma do caminho de decisão"
        )


def test_exportar_politica_nao_ajustada_levanta() -> None:
    """O desfecho oposto de `exporta`, e ele não é cerimônia.

    Uma política não ajustada exportada em silêncio gravaria um artefato **vazio de treino** que o
    carregador aceitaria: o produto passaria a recomendar a partir de um modelo que ninguém
    ajustou, e nada acusaria. É a mesma razão de `LearnedPolicy.recommend` levantar.
    """
    with pytest.raises(ArtefatoInvalidoError, match="não está ajustada"):
        exporta(LearnedPolicy(seed=SEMENTE), corpus=CORPUS, n_corpus=0)


def test_arvore_com_ciclo_levanta_em_vez_de_travar(
    corpus: tuple[list[Any], list[str], list[float]],
) -> None:
    """A guarda do laço, e ela existe porque o desfecho alternativo é SILÊNCIO.

    Um artefato editado à mão pode apontar um filho para um ancestral. Sem limite de passos o
    processo giraria para sempre — e um laço a girar parece trabalho em andamento, que é o modo de
    falha mais difícil de ver.
    """
    perfis, _, _ = corpus
    dados = json.loads(ARTEFATO.read_text(encoding="utf-8"))
    raiz = dados["corretor"]["nos"][0]
    assert raiz.get("folha") is not True, "âncora: a raiz é folha e não há ciclo a plantar"

    # Os dois filhos da raiz apontam para a própria raiz: qualquer entrada cicla.
    raiz["esquerda"] = 0
    raiz["direita"] = 0
    politica = PoliticaDoArtefato.de_dict(dados)
    with pytest.raises(ArtefatoInvalidoError, match="ciclo"):
        politica.recommend(perfis[0])


def test_predict_margin_do_artefato_bate_com_o_da_politica_ajustada(
    corpus: tuple[list[Any], list[str], list[float]], ajustada: LearnedPolicy
) -> None:
    """As linhas de base partilham o estimador de margem, então ele tem de sobreviver.

    Se o artefato devolvesse outra margem, a comparação do critério misturaria escolha de
    corretor com calibração de confiança — e nenhuma das duas seria atribuível.
    """
    perfis, _, _ = corpus
    do_artefato = PoliticaDoArtefato.de_dict(json.loads(ARTEFATO.read_text(encoding="utf-8")))
    assert perfis, "âncora: corpus vazio"
    for perfil in perfis:
        assert do_artefato.predict_margin(perfil) == pytest.approx(
            ajustada.predict_margin(perfil), rel=1e-12
        )


def test_predict_margin_sem_cabeca_de_margem_e_nao_medida() -> None:
    """Sem cabeça, a margem é `nan` — *"não medida"* —, e não zero."""
    dados = json.loads(ARTEFATO.read_text(encoding="utf-8"))
    dados["margem"] = None
    politica = PoliticaDoArtefato.de_dict(dados)

    from refino.domain.diagnosis import DefectProfile

    perfil = DefectProfile(
        defect_type="clean",
        confidence=0.9,
        suggested_batches=(),
        changepoints=(),
        qc_rsd_median=0.1,
        qc_rsd_p75=0.2,
        zero_fraction=0.0,
        missing_fraction=0.0,
        pc1_injection_correlation=0.1,
        qc_dispersion_ratio=1.0,
        n_features=10,
        n_samples=10,
        n_qcs=3,
    )
    margem = politica.predict_margin(perfil)
    assert margem != margem, "a margem ausente devia ser `nan`, e veio um número"


def test_exporta_produz_um_dicionario_que_o_carregador_aceita(ajustada: LearnedPolicy) -> None:
    """Ida e volta em memória, sem tocar disco. `exporta` e `de_dict` são o mesmo contrato."""
    dados = exporta(ajustada, corpus=CORPUS, n_corpus=N_ESPERADO)
    assert PoliticaDoArtefato.de_dict(dados) is not None


def test_a_margem_ausente_sobrevive_a_ida_e_volta() -> None:
    """`_margem` é `None` quando nenhuma margem do corpus é finita, e isso NÃO é zero.

    Um artefato que gravasse a cabeça ausente como zero tornaria *"não medida"* e *"medida e
    nula"* indistinguíveis — e a política passaria a ter confiança em vez de se abster.
    """
    from refino.domain.diagnosis import DefectProfile

    perfil = DefectProfile(
        defect_type="clean",
        confidence=0.9,
        suggested_batches=(),
        changepoints=(),
        qc_rsd_median=0.1,
        qc_rsd_p75=0.2,
        zero_fraction=0.0,
        missing_fraction=0.0,
        pc1_injection_correlation=0.1,
        qc_dispersion_ratio=1.0,
        n_features=10,
        n_samples=10,
        n_qcs=3,
    )
    politica = LearnedPolicy(seed=SEMENTE).fit(
        [perfil, perfil],
        ["serrf", "none"],  # type: ignore[list-item]
        [float("nan"), float("nan")],
    )
    dados = exporta(politica, corpus=CORPUS, n_corpus=2)
    assert dados["margem"] is None, (
        "a cabeça de margem ausente foi gravada como algo que não é nulo"
    )

    do_artefato = PoliticaDoArtefato.de_dict(dados)
    recomendacao = do_artefato.recommend(perfil)
    assert recomendacao.assessable is False, "margem não medida foi tratada como avaliável"
    assert recomendacao.abstains, "sem cabeça de margem a política tem de se abster"


# ---------------------------------------------------------------------------
# O estado do critério na árvore da margem
# ---------------------------------------------------------------------------


def _alcanca_a_abstencao(dados: dict[str, Any]) -> bool:
    """Se alguma folha da árvore da margem fica abaixo do limiar de abstenção do próprio artefato.

    A folha de uma árvore de regressão vale a média das margens de treino que caem nela, então a
    margem predita nunca sai de `[menor folha; maior folha]`: com todas as folhas acima do limiar,
    a política não pode abster-se em entrada nenhuma.
    """
    assert dados["margem"] is not None, "âncora: o artefato não tem árvore da margem"
    folhas = [float(no["valor"]) for no in dados["margem"]["nos"] if no.get("folha")]
    assert folhas, "âncora: a árvore da margem não tem folha nenhuma"
    return min(folhas) < float(dados["limiar_de_abstencao"])


def test_a_abstencao_nao_e_alcancavel_com_o_palpite_da_floresta_do_r() -> None:
    """O estado do critério, afirmado no artefato para não mudar em silêncio.

    **Antes** a folha mínima da margem era 0,015269, acima do limiar de 0,012, e a política não
    podia abster-se em entrada nenhuma — o «insatisfazível». Com o corpus de 7 QCs
    uma folha valia 0,008299, de uma matriz só, e a abstenção passou a ser alcançável. **Com os
    rótulos da floresta que conta como o `ranger` a folha mínima é 0,1139**, e o
    critério volta a ser insatisfazível: com a floresta do R o SERRF vence com margem maior, e a
    árvore da margem não prediz nenhuma abaixo do limiar. O Palpite antigo, guardado em `ANTIGO`,
    continua a alcançar — e é a âncora de que o predicado não responde «não» para tudo.

    Se uma regeração do artefato voltar a pôr uma folha abaixo do limiar, este teste reprova, e a
    mudança de estado do critério precisa de registro.
    """
    dados = json.loads(ARTEFATO.read_text(encoding="utf-8"))
    assert _alcanca_a_abstencao(dados) is False, (
        "uma folha da árvore da margem ficou abaixo do limiar de abstenção: o `[CA-9.5]`(a) deixou "
        "de ser insatisfazível com este artefato, e isso muda o estado declarado na §7"
    )
    antigo = json.loads(ANTIGO.read_text(encoding="utf-8"))
    assert _alcanca_a_abstencao(antigo) is True, (
        "âncora: o Palpite antigo alcançava a abstenção; se ele não alcança, o predicado "
        "deixou de separar"
    )


def test_o_predicado_da_abstencao_separa_os_dois_estados() -> None:
    """Par causal EM PROCESSO: a mesma árvore com uma folha baixada abaixo do limiar alcança.

    A planta é argumento da função sob teste, e por isso este negativo não pode ficar inerte; e
    ela cai sobre dado em que as duas coisas diferem, porque o artefato real não alcança.
    """
    dados = json.loads(ARTEFATO.read_text(encoding="utf-8"))
    limiar = float(dados["limiar_de_abstencao"])
    baixada = copy.deepcopy(dados)
    folha = next(no for no in baixada["margem"]["nos"] if no.get("folha"))
    folha["valor"] = limiar / 2
    assert _alcanca_a_abstencao(dados) is False, "âncora: o artefato real não alcança"
    assert _alcanca_a_abstencao(baixada) is True, (
        "baixar uma folha abaixo do limiar não mudou o desfecho, e o predicado não separa nada"
    )


def test_d581_o_ca_9_5_a_reprova_nas_ambiguas_do_corpus(
    corpus: tuple[list[Any], list[str], list[float]],
) -> None:
    """O critério na forma medida, no artefato distribuído: REPROVA.

    **Com a floresta do R nenhum modo real é ambíguo**, e a metade (a) passou para as matrizes
    do corpus cuja margem medida fica abaixo do limiar: nelas a
    política tem de reportar confiança abaixo de 0,5. São duas, e o Palpite responde às duas com a
    margem predita da menor folha, 0,1139, e confiança 0,947. **A avaliação é dentro da amostra**,
    que é a mais favorável à política: reprovando aqui, reprova em qualquer forma mais exigente.

    **Par causal em processo:** a mesma árvore com as folhas da margem baixadas abaixo do limiar
    abstém-se nas mesmas duas — a medição sabe dizer «passa», e quem a faz dizer «reprova» é a
    árvore. A âncora de não vacuidade é o conjunto das ambíguas, que tem de ter as duas.
    """
    from refino.decision.policy import ABSTENTION_MARGIN

    perfis, _, margens = corpus
    ambiguas = [perfis[i] for i, m in enumerate(margens) if m < ABSTENTION_MARGIN]
    assert len(ambiguas) == 2, f"âncora: o corpus tem {len(ambiguas)} matrizes abaixo do limiar"

    dados = json.loads(ARTEFATO.read_text(encoding="utf-8"))
    politica = PoliticaDoArtefato.de_dict(dados)
    assert not any(politica.recommend(p).abstains for p in ambiguas), (
        "o Palpite passou a abster-se numa matriz ambígua do corpus: o `[CA-9.5]`(a) mudou de "
        "estado, e isso muda o declarado na §7"
    )

    baixada = copy.deepcopy(dados)
    for no in baixada["margem"]["nos"]:
        if no.get("folha"):
            no["valor"] = ABSTENTION_MARGIN / 2
    assert all(PoliticaDoArtefato.de_dict(baixada).recommend(p).abstains for p in ambiguas), (
        "com as folhas abaixo do limiar a política não se absteve: a medição não separa nada"
    )


# ---------------------------------------------------------------------------
# O Palpite antigo continua no projeto
# ---------------------------------------------------------------------------


def test_o_palpite_antigo_guardado_se_reproduz_do_corpus_antigo() -> None:
    """O Palpite da floresta antiga fica no projeto, para se poder voltar a ele.

    Guardado não basta: ele tem de ser **o** Palpite daquele corpus. Este teste o reconstrói do
    corpus antigo e compara a recomendação nos 40 perfis — o mesmo portão de concordância do
    artefato ativo, aplicado ao guardado. **E o par causal vai junto:** os dois Palpites diferem em
    ao menos um perfil, senão «guardar o antigo» não guardaria nada que o ativo já não tivesse.
    """
    perfis, rotulos, margens = _carrega_corpus(CORPUS_ANTIGO)
    assert len(perfis) == N_ESPERADO, "âncora: o corpus antigo não tem os 40 perfis"
    reconstruido = LearnedPolicy(seed=SEMENTE).fit(perfis, rotulos, margens)  # type: ignore[arg-type]
    antigo = PoliticaDoArtefato.de_dict(json.loads(ANTIGO.read_text(encoding="utf-8")))
    ativo = PoliticaDoArtefato.de_dict(json.loads(ARTEFATO.read_text(encoding="utf-8")))
    divergentes = [
        i
        for i, p in enumerate(perfis)
        if (antigo.recommend(p).correction, antigo.recommend(p).rationale)
        != (reconstruido.recommend(p).correction, reconstruido.recommend(p).rationale)
    ]
    assert not divergentes, f"o Palpite antigo guardado diverge do seu corpus em {divergentes}"
    assert any(antigo.recommend(p).correction != ativo.recommend(p).correction for p in perfis), (
        "o Palpite antigo e o ativo recomendam igual nos 40: guardar o antigo não guarda nada"
    )
