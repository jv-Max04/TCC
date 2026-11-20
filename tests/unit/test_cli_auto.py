"""O `--auto` do `corrigir`.

## O que este arquivo verifica, e o que ele deliberadamente NÃO verifica

Ele verifica a **delegação**: que o rótulo é recusado na fronteira, que cada rótulo escolhe a
política que diz escolher, que o método que corre é o que a política devolveu e não o padrão do
argumento, e que a justificativa chega ao log e a ressalva ao `stderr`.

Ele **não** verifica que a escolha é boa. Não pode: a medição que a qualifica está na própria
ressalva — 70,0% da aprendida contra 55,0% da constante deixando um modo de fora, sem
significância. Um
teste que afirmasse acerto sobre a fixture minúscula mediria a fixture, não a política.

## A planta cai sobre dado em que as duas coisas DIFEREM

O risco desta unidade é o par vácuo: as duas políticas escolhem `serrf` na maioria das matrizes —
`serrf` é a classe majoritária do corpus, e é por isso que `AlwaysSERRFPolicy` **é** a melhor
constante e não um piso ingênuo. Um teste que só comparasse `correction` passaria com uma
implementação que ignorasse `--auto` por completo.

Então o que se afirma é o que **diverge** por construção: `confidence`, `abstains` e `assessable`.
Medido nas quatro matrizes reais e na fixture: a aprendida mede a margem a partir da segunda árvore
e a constante **não tem estimador de margem**, então ela abstém-se sempre — e é honesto que o faça,
porque escolher sempre o mesmo método não produz distância até o segundo.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pytest

from refino.cli import (
    AUTO_APRENDIDA,
    AUTO_CAVEATS,
    AUTO_CONSTANTE,
    AUTO_MEDIDA,
    AUTO_VALIDOS,
    EXIT_INPUT_REJECTED,
    METHOD_CAVEATS,
    Options,
    build_parser,
    emite_ressalva_do_auto,
    main,
    parse_options,
    politica_do_rotulo,
    recomenda,
)
from refino.decision.artefato import ArtefatoInvalidoError, PoliticaDoArtefato
from refino.decision.palpite import PalpiteMedido, decide_o_palpite
from refino.decision.policy import AlwaysSERRFPolicy
from refino.domain.labels import InvalidLabelError
from refino.io.politica import ARTEFATO_EMBALADO, carrega_politica_aprendida

REPO = Path(__file__).resolve().parent.parent.parent
FIXTURES = REPO / "data" / "fixtures"


def _argv(output: Path, *extra: str, log: Path | None = None) -> list[str]:
    argv = [
        "corrigir",
        "--measurements",
        str(FIXTURES / "tiny_drift.csv"),
        "--sample-list",
        str(FIXTURES / "tiny_drift_sample_list.csv"),
        "--mode",
        "tiny",
        "--output",
        str(output),
    ]
    if log is not None:
        argv += ["--log", str(log)]
    return argv + list(extra)


def _corre(tmp_path: Path, nome: str, *extra: str) -> tuple[int, dict[str, Any]]:
    log = tmp_path / f"{nome}.json"
    codigo = main(_argv(tmp_path / f"{nome}.csv", *extra, log=log))
    dados: dict[str, Any] = json.loads(log.read_text("utf-8"))
    return codigo, dados


# ---------------------------------------------------------------------------
# A fronteira: TRÊS desfechos de rótulo, e o quarto é a recusa
# ---------------------------------------------------------------------------


def test_os_tres_desfechos_do_rotulo_chegam_a_options(tmp_path: Path) -> None:
    """Ausente, nu e nomeado — e os três produzem valores DIFERENTES em `Options.auto`.

    Sem esta asserção, `nargs="?"` com `const` é fácil de escrever errado de um jeito que não
    levanta: `const` esquecido faz `--auto` nu virar `None`, indistinguível de não o passar, e a
    delegação some sem que nada acuse.
    """
    parser = build_parser()
    base = _argv(tmp_path / "x.csv")

    assert parse_options(parser.parse_args(base)).auto is None
    assert parse_options(parser.parse_args([*base, "--auto"])).auto == AUTO_APRENDIDA
    assert parse_options(parser.parse_args([*base, "--auto", "constante"])).auto == AUTO_CONSTANTE


def test_rotulo_fora_do_vocabulario_e_recusado_na_fronteira(tmp_path: Path) -> None:
    """A guarda de fronteira, e a mensagem nomeia os DOIS rótulos aceitos.

    O modo de falha que ela evita é o insidioso, o que **não** levanta: `--auto constant` — sem o
    `e` — a cair num `else` razoável e a correr a política aprendida, com o log a declarar uma
    delegação que o usuário não pediu.
    """
    parser = build_parser()
    namespace = parser.parse_args([*_argv(tmp_path / "x.csv"), "--auto", "constant"])

    with pytest.raises(InvalidLabelError) as erro:
        parse_options(namespace)

    for rotulo in AUTO_VALIDOS:
        assert rotulo in str(erro.value), f"a recusa nao diz que `{rotulo}` e aceito"


def test_o_auto_nao_usa_choices_do_argparse() -> None:
    """Irmão do teste que protege `validate_label`, pelo mesmo argumento.

    Com `choices=`, o `argparse` recusaria antes de a guarda de `parse_options` correr: ela
    continuaria escrita e nunca executada — dois verificadores para a mesma coisa, um deles
    inalcançável. E a mensagem do `argparse` diria apenas `invalid choice`, sem explicar que a
    ausência do valor tem significado próprio.
    """
    parser = build_parser()
    sub = next(a for a in parser._actions if isinstance(a, argparse._SubParsersAction))
    acao = next(a for a in sub.choices["corrigir"]._actions if "--auto" in a.option_strings)

    assert acao.choices is None, "--auto usa choices= e torna a guarda de fronteira inerte (AR37)"
    assert acao.const == AUTO_APRENDIDA, "`--auto` nu deixaria de significar a politica aprendida"
    assert acao.default is None, "a ausencia deixaria de ser distinguivel do valor nu"


# ---------------------------------------------------------------------------
# As DUAS políticas, e a asserção é sobre o que as separa
# ---------------------------------------------------------------------------


def test_cada_rotulo_devolve_a_politica_que_ele_nomeia() -> None:
    """Os dois rótulos, e são dois **tipos** diferentes — não dois nomes do mesmo objeto.

    A direção `constante` é a que ficaria inerte sem esta asserção: ela é o ramo que não é o
    padrão, e a contagem de `passed` não distingue teste que exercita os dois de teste que
    exercita um.
    """
    assert isinstance(politica_do_rotulo(AUTO_CONSTANTE), AlwaysSERRFPolicy)
    assert isinstance(politica_do_rotulo(AUTO_APRENDIDA), PoliticaDoArtefato)


def test_as_duas_politicas_diferem_no_que_relatam(tmp_path: Path) -> None:
    """**A planta cai onde as duas DIVERGEM, e não onde elas coincidem.**

    As duas escolhem `serrf` — `serrf` é a classe majoritária, e é por isso que a constante é a
    melhor política constante deste corpus. Afirmar só `correction` seria vácuo: passaria com uma
    implementação que ignorasse o rótulo.

    O que diverge é a **confiança**: a aprendida prediz a margem pela segunda árvore do artefato;
    a constante não tem estimador de margem e por isso `assessable` é falso e ela se abstém. É a
    distinção lida no lugar certo — *"a margem é minúscula"* e *"a margem não foi medida"* levam à
    mesma ação e a contagens diferentes.
    """
    _, aprendida = _corre(tmp_path, "aprendida", "--auto")
    _, constante = _corre(tmp_path, "constante", "--auto", "constante")

    assert aprendida["auto"]["politica"] == AUTO_APRENDIDA
    assert constante["auto"]["politica"] == AUTO_CONSTANTE

    assert constante["auto"]["assessable"] is False, (
        "a constante passou a declarar a margem como medida — ou ganhou estimador, e entao esta "
        "assercao precisa de ser reargumentada, ou o campo deixou de ser preenchido"
    )
    assert constante["auto"]["abstains"] is True
    assert aprendida["auto"]["confidence"] != constante["auto"]["confidence"], (
        "as duas politicas relataram a MESMA confianca: o rotulo pode nao estar a ser lido"
    )


# ---------------------------------------------------------------------------
# O que corre é o que a política escolheu — não o padrão do argumento
# ---------------------------------------------------------------------------


def test_sem_auto_nada_muda_e_o_log_nao_ganha_a_chave(tmp_path: Path) -> None:
    """O caminho antigo, intacto. **Chave ausente e não `null`** — presença é o sinal."""
    codigo, log = _corre(tmp_path, "sem")

    assert codigo == 0
    assert "auto" not in log, "o log ganhou a chave da delegacao sem ninguem ter delegado"
    assert log["spec"]["method"] == "none"


def test_o_metodo_do_log_e_o_que_correu_e_nao_o_padrao_do_argumento(tmp_path: Path) -> None:
    """**O critério desta função é "realimentar reproduz a saída"**, e é ele que decide isto.

    `--method` fica no padrão `none` quando se delega, porque não se passou nenhum. Gravar `none`
    em `spec.method` faria o log realimentado produzir OUTRA matriz — a identidade em vez da
    corrigida —, e a reexecução pelo log deixaria de valer sem que nada reprovasse.

    A planta cai onde os dois diferem: a fixture leva a política a escolher algo que **não** é
    `none`, e é a asserção de desigualdade que torna o teste não-vácuo.
    """
    _, log = _corre(tmp_path, "delegado", "--auto")

    escolhido = log["auto"]["correction"]
    assert escolhido != "none", (
        "a politica escolheu `none` nesta fixture, e entao este teste nao separa o metodo efetivo "
        "do padrao do argumento — troque a fixture, nao a assercao"
    )
    assert log["spec"]["method"] == escolhido
    assert log["auto"]["rotulo"] == AUTO_APRENDIDA


def test_a_justificativa_e_a_ressalva_chegam_ao_log(tmp_path: Path) -> None:
    """A justificativa tem de ficar registada, e o log é o que sobrevive à sessão."""
    _, log = _corre(tmp_path, "just", "--auto")

    assert log["auto"]["rationale"].strip(), "delegou e nao disse por que"
    assert log["auto"]["evidence"], "a justificativa veio sem as caracteristicas que a sustentam"
    for campo in ("resumo", "detalhe", "evidencia", "portao_inerte"):
        assert log["auto"]["caveat"][campo].strip(), f"ressalva do `--auto` com `{campo}` vazio"


def test_a_ressalva_do_metodo_efetivo_tambem_entra_no_log(tmp_path: Path) -> None:
    """Quem delega e recebe `serrf` tem de ver a ressalva do `serrf`, não só a da política.

    São recortes **diferentes**: um diz o que a política não afirma, o outro o que o método não
    afirma. Colapsá-los perderia o segundo justamente para quem não o escolheu.
    """
    _, log = _corre(tmp_path, "duplo", "--auto")

    if log["auto"]["correction"] in METHOD_CAVEATS:
        assert log["caveat"]["method"] == log["auto"]["correction"]
    else:  # pragma: no cover - depende do metodo que a fixture leva a politica a escolher
        assert "caveat" not in log


def test_a_ressalva_da_politica_sai_no_stderr_e_sem_auto_nao_sai(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """**Par causal: duas execuções que diferem só pela delegação.**

    Afirmar apenas que a ressalva aparece seria satisfeito por uma implementação que a imprimisse
    sempre; afirmar apenas que não aparece seria satisfeito por uma que nunca a imprimisse. As
    duas asserções juntas é que separam.
    """
    main(_argv(tmp_path / "com.csv", "--auto", "constante"))
    com = capsys.readouterr().err

    main(_argv(tmp_path / "sem.csv"))
    sem = capsys.readouterr().err

    assert AUTO_CAVEATS[AUTO_CONSTANTE].resumo in com
    assert AUTO_CAVEATS[AUTO_CONSTANTE].resumo not in sem
    assert AUTO_CAVEATS[AUTO_APRENDIDA].resumo not in sem


def test_a_segunda_opiniao_vai_ao_log_e_ao_stderr_so_quando_se_delega(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """O Palpite pela verdade ao lado da escolha — e **não decide**.

    **Par causal sobre a delegação**, como o da ressalva acima: duas execuções que diferem só pelo
    `--auto`. E a asserção de que ela não decide é a que importa: o método que correu é o da
    primeira opinião, e o log grava as duas sob o mesmo `auto` para quem o reler.

    **Com o interruptor LIGADO**, porque a segunda fica em stand-by: é este teste que
    garante que religá-la devolve o que ela fazia. O desligado é o teste seguinte.
    """
    from refino.cli import RESSALVA_PELA_VERDADE, ROTULO_PELA_VERDADE
    from refino.io import politica

    monkeypatch.setattr(politica, "SEGUNDA_OPINIAO_ATIVA", True)
    _, log = _corre(tmp_path, "segunda", "--auto")
    err_com = capsys.readouterr().err
    _, sem = _corre(tmp_path, "segunda_sem")
    err_sem = capsys.readouterr().err

    segunda = log["auto"]["segunda_opiniao"]
    assert segunda["rotulo"] == ROTULO_PELA_VERDADE
    assert segunda["correction"] in {"none", "loess", "serrf"}
    assert segunda["caminho"].strip(), "a segunda opinião veio sem o caminho da árvore"
    assert segunda["caveat"]["resumo"] == RESSALVA_PELA_VERDADE.resumo
    assert not {"confidence", "abstains", "assessable"} & set(segunda), sorted(segunda)
    assert log["spec"]["method"] == log["auto"]["correction"], (
        "o método que correu deixou de ser o da primeira opinião"
    )
    assert "auto" not in sem
    # O par sobre o stderr: a linha sai com a delegação e não sai sem ela. Sem `capsys` a ler o
    # `stderr` de `_corre`, a asserção de ausência abaixo passaria por vacuidade — e a de presença
    # é a âncora.
    assert ROTULO_PELA_VERDADE in err_com, err_com
    assert ROTULO_PELA_VERDADE not in err_sem


def test_d578_em_stand_by_o_auto_nao_consulta_nem_grava_a_segunda_opiniao(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A segunda opinião em stand-by: o `--auto` decide pela primeira e cala a segunda.

    **O carregador da segunda é trocado por um que levanta**, e a execução tem de terminar mesmo
    assim: é o que prova que, desligada, ela não é consultada — e não só que deixou de ser
    impressa. A âncora de presença é a ressalva da primeira opinião, no mesmo `stderr`.
    """
    from refino import cli
    from refino.cli import ROTULO_PELA_VERDADE
    from refino.io import politica

    def quebrado() -> PoliticaDoArtefato:
        raise ArtefatoInvalidoError("planta: o artefato da segunda opinião não abre")

    monkeypatch.setattr(cli, "carrega_politica_pela_verdade", quebrado)
    assert politica.SEGUNDA_OPINIAO_ATIVA is False, "o estado vigente é a segunda desligada"

    codigo, log = _corre(tmp_path, "standby", "--auto")
    err = capsys.readouterr().err

    assert codigo == 0
    assert AUTO_CAVEATS[AUTO_APRENDIDA].resumo in err, "ÂNCORA: a primeira opinião não falou"
    assert "segunda_opiniao" not in log["auto"], sorted(log["auto"])
    assert ROTULO_PELA_VERDADE not in err, err


def test_emite_ressalva_do_auto_devolve_se_escreveu(capsys: pytest.CaptureFixture[str]) -> None:
    """`bool` e não `None`: sem retorno, *"nada foi escrito"* é vácuo."""
    assert emite_ressalva_do_auto(AUTO_APRENDIDA, sys.stderr) is True
    assert capsys.readouterr().err.strip()


# ---------------------------------------------------------------------------
# O carregador: os três desfechos de falha levam à MESMA ação e a mensagens diferentes
# ---------------------------------------------------------------------------


def test_o_artefato_embalado_vive_dentro_do_pacote() -> None:
    """**O valor decide quanto do caminho corre, e o custo disso seria só no congelado.**

    O `datas=` de `refino-desktop.spec` lista as migrações e os recursos, e mais nada. Um artefato
    em `data/fixtures/` resolve-se, num `--onedir`, para dentro de `_internal/` — onde aquele
    diretório não existe. O `--auto` funcionaria em todo teste desta suíte e **só** no executável
    a política não carregaria, com quatro superfícies declaradas como cobertas.
    """
    assert ARTEFATO_EMBALADO.exists()
    assert ARTEFATO_EMBALADO.parent.name == "decision"
    assert "fixtures" not in ARTEFATO_EMBALADO.parts, (
        "o artefato voltou para `data/fixtures/` e deixa de viajar no congelado"
    )


@pytest.mark.parametrize(
    ("nome", "conteudo"),
    [
        ("ausente", None),
        ("nao_json", "isto nao e json {"),
        ("ordem_trocada", "trocar"),
    ],
)
def test_artefato_ruim_levanta_em_vez_de_degradar(
    nome: str, conteudo: str | None, tmp_path: Path
) -> None:
    """**Levanta em vez de cair numa política padrão**, e é a guarda de fronteira aplicada a
    artefato.

    Uma recomendação que ninguém treinou, servida como se tivesse saído do corpus, é o modo de
    falha que não levanta: o usuário recebe um método plausível e nada diz que a política não
    carregou.

    Os três desfechos levam à **mesma ação** de quem chama — recusar o `--auto` — e por isso
    partilham a exceção; o que os separa é a mensagem, que é onde a distinção importa.
    """
    alvo = tmp_path / f"p_{nome}.json"
    if conteudo == "trocar":
        dados = json.loads(ARTEFATO_EMBALADO.read_text("utf-8"))
        # A chave é `caracteristicas` e foi LIDA do artefato, não citada de memória — a primeira
        # escrita deste teste usou `feature_names`, que não existe, e o `KeyError` acusou.
        assert len(dados["caracteristicas"]) > 1, (
            "com uma caracteristica so, inverter a ordem nao muda nada e a planta e vacua"
        )
        dados["caracteristicas"] = list(reversed(dados["caracteristicas"]))
        alvo.write_text(json.dumps(dados), "utf-8")
    elif conteudo is not None:
        alvo.write_text(conteudo, "utf-8")

    with pytest.raises(ArtefatoInvalidoError):
        carrega_politica_aprendida(alvo)


def test_o_carregador_le_o_artefato_embalado_por_omissao() -> None:
    """A planta que prova que o parâmetro opcional NÃO é o caminho normal.

    Sem esta asserção, `carrega_politica_aprendida(alvo)` nos testes acima deixaria o caminho
    padrão — o único que a CLI usa — sem nenhuma execução, e ele é o que quebra no congelado.
    """
    politica = carrega_politica_aprendida()

    assert politica.corretor, "a politica embalada carregou sem arvore de corretor"
    assert politica.name == "learned"


def test_options_carrega_o_campo_auto() -> None:
    """`Options` espelha um-para-um o que afeta a saída — o campo `auto` afeta, então entra.

    Campo que afete a saída e não esteja em `Options` não entra no log, e o log deixa de
    reproduzir a execução sem que nada acuse. É o contrato que a docstring da classe declara.
    """
    assert "auto" in Options.__dataclass_fields__


# ---------------------------------------------------------------------------
# `medida` — o Palpite MEDIDO, e ele não é política de perfil
# ---------------------------------------------------------------------------


class _Medidor:
    """O duplo de `mede_o_palpite`: devolve um Palpite decidido e guarda o que recebeu.

    **A medição verdadeira tem teste próprio**, em `tests/unit/test_palpite_medido.py`, contra os
    vetores gravados dos arquivos reais. Aqui o que se afirma é a delegação: que `medida` chega ao
    medidor, e que o que ele decide é o que corre.
    """

    def __init__(self, escores: dict[str, float]) -> None:
        self.escores = escores
        self.chamadas: list[dict[str, Any]] = []

    def __call__(self, matrix: Any, **kwargs: Any) -> PalpiteMedido:
        self.chamadas.append({"n_features": matrix.n_features, **kwargs})
        return decide_o_palpite(self.escores, n_features_medidas=min(200, matrix.n_features))


def test_medida_nao_e_politica_de_perfil_e_recusa_em_vez_de_cair_na_arvore() -> None:
    """**A recusa é o mecanismo.** Um `else` razoável devolveria a árvore, e `medida` correria
    outra coisa sem levantar — o modo de falha da fronteira na forma mais silenciosa."""
    with pytest.raises(ValueError, match="medida"):
        politica_do_rotulo(AUTO_MEDIDA)


def test_medida_chega_a_options_e_e_aceita_na_fronteira(tmp_path: Path) -> None:
    parser = build_parser()
    namespace = parser.parse_args([*_argv(tmp_path / "x.csv"), "--auto", AUTO_MEDIDA])
    assert parse_options(namespace).auto == AUTO_MEDIDA


def test_recomenda_medida_mede_a_matriz_e_nao_consulta_a_arvore(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """O par causal sobre o rótulo: o carregador da árvore é trocado por um que levanta, e
    `medida` tem de decidir mesmo assim — porque não passa por ele. `aprendida`, com o mesmo
    carregador, levanta: é a âncora de que a planta aterrou."""
    from refino import cli
    from refino.io.readers import load_matrix

    def quebrado() -> PoliticaDoArtefato:
        raise ArtefatoInvalidoError("planta: a árvore não abre")

    medidor = _Medidor({"none": 0.30, "loess": 0.10, "serrf": 0.20})
    monkeypatch.setattr(cli, "mede_o_palpite", medidor)
    monkeypatch.setattr(cli, "carrega_politica_aprendida", quebrado)
    matriz = load_matrix(
        FIXTURES / "tiny_drift.csv", FIXTURES / "tiny_drift_sample_list.csv", "tiny"
    )

    recomendacao = recomenda(matriz, AUTO_MEDIDA, n_jobs=2)

    assert recomendacao.correction == "loess"
    assert recomendacao.evidence["n_features_medidas"] == float(matriz.n_features)
    assert medidor.chamadas == [
        {"n_features": matriz.n_features, "batches": None, "n_jobs": 2, "seed": 42}
    ]
    with pytest.raises(ArtefatoInvalidoError):
        recomenda(matriz, AUTO_APRENDIDA)


def test_o_log_grava_medida_como_politica_e_o_metodo_que_correu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`politica` é o rótulo, e não `aprendida`: derivar por `if` gravaria a medida como a árvore.

    **O Palpite do duplo propõe `loess`**, que não é o padrão `none` do argumento — é a planta que
    separa o método efetivo do padrão, pelo argumento do teste irmão acima.
    """
    from refino import cli

    monkeypatch.setattr(
        cli, "mede_o_palpite", _Medidor({"none": 0.30, "loess": 0.10, "serrf": 0.20})
    )

    codigo, log = _corre(tmp_path, "medida", "--auto", AUTO_MEDIDA)
    err = capsys.readouterr().err

    assert codigo == 0
    assert log["auto"]["politica"] == AUTO_MEDIDA
    assert log["auto"]["correction"] == "loess"
    assert log["spec"]["method"] == "loess"
    assert log["auto"]["caveat"]["resumo"] == AUTO_CAVEATS[AUTO_MEDIDA].resumo
    assert AUTO_CAVEATS[AUTO_MEDIDA].resumo in err


def test_medida_numa_matriz_de_dois_qcs_recusa_com_mensagem_e_aprendida_corre(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """**Sem duplo: a recusa é a da régua de verdade.** A fixture tem dois QCs, e esconder um
    deixa um só para ajustar — a régua levanta. A matriz é corrigível, então o par é a
    mesma entrada com `aprendida`, que corre: o que se recusa é a decisão, não o arquivo."""
    codigo = main(_argv(tmp_path / "m.csv", "--auto", AUTO_MEDIDA))
    err = capsys.readouterr().err

    assert codigo == EXIT_INPUT_REJECTED
    assert "Palpite medido não mede esta matriz" in err, err
    assert "--auto aprendida" in err, "a recusa não diz a saída que resta"
    assert not (tmp_path / "m.csv").exists(), "recusou e escreveu a saída mesmo assim"

    assert main(_argv(tmp_path / "a.csv", "--auto", AUTO_APRENDIDA)) == 0


def test_a_semente_do_corrigir_chega_a_escolha_do_corretor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Com `--seed 7` a escolha do corretor mede com 7, e não com o 42 fixo.

    Par causal sobre a mesma entrada: sem `--seed`, o medidor recebe o padrão 42.
    """
    from refino import cli

    medidor = _Medidor({"none": 0.30, "loess": 0.10, "serrf": 0.20})
    monkeypatch.setattr(cli, "mede_o_palpite", medidor)

    assert _corre(tmp_path, "s7", "--auto", AUTO_MEDIDA, "--seed", "7")[0] == 0
    assert _corre(tmp_path, "s42", "--auto", AUTO_MEDIDA)[0] == 0

    assert [c["seed"] for c in medidor.chamadas] == [7, 42]
