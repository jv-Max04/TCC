"""A fronteira da CLI.

**O que esta suíte cobre é a FRONTEIRA, não a correção.** Se o `batch_loess` está
certo é medido contra o golden por feature; se o SERRF concorda com o
R é outro teste. Aqui a pergunta é outra e é a que a CLI introduz: **um valor que
nunca passou pelo mypy pode mudar o comportamento em silêncio?**

Por isso o teste mais importante do arquivo não afirma número nenhum —
`test_nenhum_argumento_de_vocabulario_usa_choices` afirma que o `argparse` **não**
valida, para que a guarda em runtime seja alcançável.
"""

import argparse
import importlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import get_args

import numpy as np
import pytest

from refino.cli import (
    AVAILABLE_METHODS,
    EXIT_INPUT_REJECTED,
    EXIT_OK,
    EXIT_UNAVAILABLE,
    EXIT_USAGE,
    METHOD_CAVEATS,
    UNAVAILABLE_METHODS,
    Unavailable,
    build_parser,
    main,
)
from refino.domain.labels import VOCABULARY, Correction
from refino.io.readers import load_matrix, read_toserrf

REPO = Path(__file__).resolve().parent.parent.parent
FIXTURES = REPO / "data" / "fixtures"

VOCABULARY_FLAGS = ("--method", "--rsd-variant", "--null-denominator", "--orientation")

# Rótulo de vocabulário por opção, para que os testes por opção não repitam o
# mapeamento — e para que acrescentar opção de vocabulário sem cobri-la apareça.
FLAG_KINDS = {
    "--method": "Correction",
    "--rsd-variant": "RsdVariant",
    "--null-denominator": "NullDenominatorPolicy",
    "--orientation": "Orientation",
}


def _subparser(name: str) -> argparse.ArgumentParser:
    """O subparser de `name`, achado pelo tipo da ação e não por índice."""
    parser = build_parser()
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            return action.choices[name]
    raise AssertionError("build_parser não registrou subcomando nenhum")


def _argv(output: Path, **extra: str) -> list[str]:
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
    for flag, value in extra.items():
        argv += [f"--{flag.replace('_', '-')}", value]
    return argv


# ---------------------------------------------------------------------------
# Como a CLI nasce: TRÊS métodos disponíveis, um deles COM RESSALVA
# ---------------------------------------------------------------------------


def test_o_vocabulario_de_metodos_esta_particionado() -> None:
    """Disponível mais indisponível é **exatamente** o vocabulário, sem sobreposição.

    É o mecanismo que impede as duas derivas opostas: método novo no `Literal` sem
    ninguém dizer se está disponível, e corretor implementado que não sai da lista de
    indisponíveis — o segundo produziria uma CLI que recusa um corretor que existe, e
    foi exactamente o estado do `serrf` enquanto ele não existia.

    **A guarda de não-vacuidade é a primeira asserção, e ela é necessária.**
    Com `UNAVAILABLE_METHODS` vazio, a igualdade de conjuntos ainda tem
    dente porque o lado esquerdo tem membros — mas se AMBOS esvaziassem, `set() == set()`
    seria verdade e o teste passaria sobre nada.
    """
    declarado = set(get_args(Correction))
    assert declarado, "o vocabulário de `Correction` está vazio; nada abaixo tem dente"
    assert AVAILABLE_METHODS, "nenhum método disponível: a partição passaria por vacuidade"
    assert set(AVAILABLE_METHODS) | set(UNAVAILABLE_METHODS) == declarado
    assert not set(AVAILABLE_METHODS) & set(UNAVAILABLE_METHODS)


def test_todo_metodo_indisponivel_nomeia_o_portao_e_a_unidade_dele() -> None:
    """Motivo sem gatilho de revisão é a lista que ninguém revisa.

    E o gatilho é o **portão**, não a unidade que escreve o código: é o portão que
    autoriza expor, e nomear só a unidade faria a entrada parecer resolvida no dia em
    que o corretor existisse.

    **`UNAVAILABLE_METHODS` está VAZIO, e o laço passaria por vacuidade.**
    A resposta não é apagar o teste — a tabela volta a ter membro no dia em que algum
    método for bloqueado, e é aí que ele precisa de existir. A resposta é o mesmo
    padrão de `_planta`: exercitar a regra sobre uma entrada CONSTRUÍDA, de forma que
    a asserção seja verificada mesmo com a tabela vazia, e o laço real fique como
    cobertura de quem entrar depois.
    """
    planta = Unavailable(
        reason="motivo qualquer que cita [CA-9.9] como portão",
        implemented=False,
        gate="[CA-9.9]",
        gate_unit="U99",
        corrector="refino.correction.none:NoneCorrector",
    )
    for method, entry in [*UNAVAILABLE_METHODS.items(), ("planta", planta)]:
        assert re.fullmatch(r"\[CA-\d+\.\d+\]", entry.gate), (
            f"{method}: `gate` {entry.gate!r} não é um critério de aceitação"
        )
        assert re.fullmatch(r"U\d+", entry.gate_unit), (
            f"{method}: `gate_unit` {entry.gate_unit!r} não nomeia unidade"
        )
        assert entry.gate in entry.reason, f"{method}: o motivo não cita o próprio portão"


def test_a_flag_implemented_bate_com_a_existencia_do_corretor() -> None:
    """**O gatilho, e ele DISPAROU — nas duas direções.**

    Ele existia para reprovar no instante em que `SerrfCorrector` passasse a existir
    com a entrada ainda dizendo `implemented=False`. Isso aconteceu, a entrada foi
    para `implemented=True`, e agora o método saiu da tabela por inteiro.

    **Deixou de ser `parametrize` e o motivo é a vacuidade.** `parametrize`
    sobre coleção vazia coleta **zero** casos, e o relatório diz `passed` — o teste
    desaparece sem nada acusar: verificação construída e nunca exercitada.
    Como função única com laço interno, ele continua a rodar e a asserção de
    não-vacuidade abaixo é o que declara sobre o que ele rodou.
    """
    assert not UNAVAILABLE_METHODS, (
        "`UNAVAILABLE_METHODS` voltou a ter membro: este teste passou a ter alvo real, "
        "e a asserção abaixo tem de ser conferida para cada entrada"
    )
    for method, entry in UNAVAILABLE_METHODS.items():  # pragma: no cover - tabela vazia
        modulo, _, classe = entry.corrector.partition(":")
        importado = importlib.import_module(modulo)
        existe = hasattr(importado, classe)
        assert existe is entry.implemented, (
            f"{method}: `implemented={entry.implemented}` e {entry.corrector} "
            f"{'existe' if existe else 'não existe'}"
        )


def test_serrf_e_aceito_e_produz_saida(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """**O inverso do teste que aqui estava, e a inversão é o ponto.**

    Ele afirmava `EXIT_UNAVAILABLE` e *"não escreveu saída"*. Hoje afirma o oposto: o
    `serrf` corre, sai `EXIT_OK`, e o CSV existe.

    **A asserção com dente continua sendo sobre o ARQUIVO**, e ela só trocou de
    direção: antes *"recusou e escreveu de todo modo"* era o defeito; agora é
    *"aceitou e não escreveu"* — aceitar sem produzir seria o pior desfecho, porque o
    código de saída diria sucesso.
    """
    output = tmp_path / "out.csv"
    code = main(_argv(output, method="serrf"))
    assert code == EXIT_OK
    assert output.exists(), "aceitou o método e não escreveu saída nenhuma"
    assert output.read_text(encoding="utf-8").strip(), "escreveu arquivo vazio"


def test_serrf_avisa_no_stderr_e_os_outros_metodos_nao(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """**A ressalva chega ao ponto de execução — par causal.**

    As duas direções na mesma função, e é isso que impede a asserção de ausência de
    passar por vacuidade: se `emite_ressalva` nunca escrevesse nada, o ramo
    `none` passaria e o ramo `serrf` reprovaria. Um teste que só afirmasse o silêncio
    de `none` seria satisfeito por uma função que não faz nada.
    """
    com_ressalva = tmp_path / "serrf.csv"
    main(_argv(com_ressalva, method="serrf"))
    erro_serrf = capsys.readouterr().err

    sem_ressalva = tmp_path / "none.csv"
    main(_argv(sem_ressalva, method="none"))
    erro_none = capsys.readouterr().err

    assert "ATENÇÃO" in erro_serrf and "serrf" in erro_serrf.lower()
    assert METHOD_CAVEATS["serrf"].portao_inerte in erro_serrf, (
        "o aviso não nomeia o portão que continua inerte"
    )
    assert "ATENÇÃO" not in erro_none, "método sem ressalva emitiu aviso"
    assert erro_serrf != erro_none, "os dois desfechos são indistinguíveis"


def test_o_codigo_de_indisponivel_difere_do_de_uso_invalido(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Dois desfechos, dois códigos, **hoje sobre uma PLANTA**.

    Colapsá-los diria "você errou de digitação" a quem pediu algo que a ferramenta
    ainda não faz — e as ações são opostas: corrigir o comando, ou esperar a versão.

    **A única instância viva saiu, e o mecanismo continua a ser exercitado.**
    Com `UNAVAILABLE_METHODS` vazio não há método que produza `EXIT_UNAVAILABLE`, e
    o teste anterior passaria a afirmar sobre o `serrf` liberado. Apagá-lo perderia a
    guarda — que é a distinção de código, não o `serrf`.

    **É par causal e não só planta:** os três braços na mesma função — com a planta,
    sem ela, e com rótulo inválido — dando três códigos distintos. Sem o braço do
    meio, uma guarda que devolvesse `EXIT_UNAVAILABLE` para tudo passaria.
    """
    planta = Unavailable(
        reason="planta de teste: bloqueado por [CA-9.9]",
        implemented=True,
        gate="[CA-9.9]",
        gate_unit="U99",
        corrector="refino.correction.none:NoneCorrector",
    )
    monkeypatch.setitem(UNAVAILABLE_METHODS, "loess", planta)
    com_planta = main(_argv(tmp_path / "a.csv", method="loess"))
    monkeypatch.delitem(UNAVAILABLE_METHODS, "loess")
    sem_planta = main(_argv(tmp_path / "b.csv", method="loess"))
    invalido = main(_argv(tmp_path / "c.csv", method="SERRF"))
    capsys.readouterr()

    assert com_planta == EXIT_UNAVAILABLE
    assert sem_planta == EXIT_OK, "a planta não foi removida; o braço de controle é vácuo"
    assert invalido == EXIT_USAGE
    assert len({com_planta, sem_planta, invalido}) == 3, "os três desfechos não se distinguem"


# ---------------------------------------------------------------------------
# Os quatro `Literal` na fronteira
# ---------------------------------------------------------------------------


def test_nenhum_argumento_de_vocabulario_usa_choices() -> None:
    """**A guarda em runtime tem de ser alcançável, e é isto que garante.**

    Com `choices=`, o `argparse` recusaria antes de `validate_label` rodar: a
    guarda continuaria escrita, nunca executada, e nenhum teste poderia
    exercitá-la. Dois verificadores para a mesma coisa, um deles inalcançável — o
    defeito na forma mais fácil de cometer, porque `choices` parece a
    escolha idiomática.

    Este teste é o que reprova quando alguém "melhora" o parser.
    """
    corrigir = _subparser("corrigir")
    for flag in VOCABULARY_FLAGS:
        action = next(a for a in corrigir._actions if flag in a.option_strings)
        assert action.choices is None, (
            f"{flag} usa choices= do argparse, o que torna validate_label inalcançável "
            "e a guarda inerte (AR37)"
        )


@pytest.mark.parametrize("flag", VOCABULARY_FLAGS)
def test_valor_invalido_em_qualquer_rotulo_recusa_e_lista_os_aceitos(
    flag: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Os quatro, e não só `--method`: a guarda é por rótulo, não por opção famosa.

    E a mensagem tem de **listar os aceitos**. Recusar sem dizer o que vale obriga
    quem usa a ler o código, e o valor de uma fronteira explícita é justamente
    poupar isso.
    """
    output = tmp_path / "out.csv"
    argv = _argv(output)
    argv += [flag, "valor-que-nao-existe"]
    code = main(argv)
    assert code == EXIT_USAGE
    assert not output.exists()
    erro = capsys.readouterr().err
    assert "esperava um de" in erro


@pytest.mark.parametrize("flag", VOCABULARY_FLAGS)
def test_o_valor_padrao_de_cada_rotulo_pertence_ao_vocabulario(flag: str) -> None:
    """O outro lado: guarda que recusasse o padrão reprovaria toda execução.

    Sem esta direção, um padrão escrito errado — `outlier-removed` com hífen —
    passaria pelo teste negativo e reprovaria só em uso real. E a
    comparação é contra `VOCABULARY`, não contra uma cópia da lista aqui.
    """
    action = next(a for a in _subparser("corrigir")._actions if flag in a.option_strings)
    assert action.default in VOCABULARY[FLAG_KINDS[flag]], (
        f"o padrão de {flag} é {action.default!r}, fora de {VOCABULARY[FLAG_KINDS[flag]]}"
    )


# ---------------------------------------------------------------------------
# O caminho que funciona
# ---------------------------------------------------------------------------


def test_none_escreve_csv_relegivel_com_os_mesmos_valores(tmp_path: Path) -> None:
    """`none` de ponta a ponta: o que sai, relido, é o que a biblioteca tinha.

    A tolerância é de round-trip textual (1e-9 relativa) e não igualdade de
    bits: `write_toserrf` escreve 15 dígitos significativos, e exigir igualdade
    exata cobraria do formato uma propriedade que ele não declara.

    A comparação é contra `load_matrix`, e não contra o CSV de entrada: entre os
    dois há a transposição da ingestão e a ordenação por injeção, e comparar com o
    arquivo confundiria defeito de escrita com defeito de ordem.
    """
    output = tmp_path / "out.csv"
    assert main(_argv(output, method="none")) == EXIT_OK

    esperado = load_matrix(
        FIXTURES / "tiny_drift.csv", FIXTURES / "tiny_drift_sample_list.csv", "tiny"
    )
    doc = read_toserrf(output)
    # `read_toserrf` devolve features em linhas — a orientação do arquivo.
    assert doc.values.shape == (esperado.n_features, esperado.n_samples)
    assert np.allclose(doc.values, esperado.values.T, rtol=1e-9, atol=0)
    assert list(doc.feature_names) == list(esperado.feature_names)


def test_loess_roda_e_marca_as_features_que_nao_corrigiu(tmp_path: Path) -> None:
    """Com dois QCs o ajuste não tem pontos, e o desfecho correto é marcar.

    Não é caso artificial: é a matriz mínima legível à mão, e o que ela
    exercita é a via de marcação — que o `--log` tem de reportar por causa.
    """
    output = tmp_path / "out.csv"
    log = tmp_path / "log.json"
    code = main([*_argv(output, method="loess"), "--log", str(log)])
    assert code == EXIT_OK
    payload = json.loads(log.read_text(encoding="utf-8"))
    assert payload["uncorrected"]["by_cause"], "nenhuma causa reportada com 2 QCs"
    assert sum(payload["uncorrected"]["by_cause"].values()) == payload["uncorrected"]["n_features"]


def test_o_log_carrega_a_especificacao_com_os_nomes_das_opcoes(tmp_path: Path) -> None:
    """`spec` usa os nomes das OPÇÕES, e é o que torna o log realimentável."""
    log = tmp_path / "log.json"
    main([*_argv(tmp_path / "out.csv", method="none"), "--log", str(log), "--seed", "7"])
    spec = json.loads(log.read_text(encoding="utf-8"))["spec"]
    assert spec["method"] == "none"
    assert spec["seed"] == 7
    for flag in VOCABULARY_FLAGS:
        assert flag.removeprefix("--") in spec, f"{flag} não entrou no log"


def test_realimentar_o_log_reproduz_a_saida_bit_a_bit(tmp_path: Path) -> None:
    """O log é suficiente para repetir a execução, e o teste é a prova disso.

    Reconstrói `argv` **a partir do log** e compara bytes. Opção que afete a saída
    e não esteja em `spec` reprova aqui — que é o mecanismo pelo qual "o log
    documenta a execução" deixa de ser afirmação sobre boa intenção.
    """
    primeiro = tmp_path / "um.csv"
    log = tmp_path / "log.json"
    main([*_argv(primeiro, method="loess"), "--log", str(log), "--seed", "5"])
    spec = json.loads(log.read_text(encoding="utf-8"))["spec"]

    segundo = tmp_path / "dois.csv"
    argv = ["corrigir", "--output", str(segundo)]
    for key, value in spec.items():
        argv += [f"--{key}", str(value)]
    assert main(argv) == EXIT_OK
    assert segundo.read_bytes() == primeiro.read_bytes()


def test_a_mesma_semente_reproduz_a_saida(tmp_path: Path) -> None:
    """Regra 3 do README na fronteira: a CLI passa a semente adiante, e não a reinventa."""
    um, dois = tmp_path / "um.csv", tmp_path / "dois.csv"
    main([*_argv(um, method="loess"), "--seed", "13"])
    main([*_argv(dois, method="loess"), "--seed", "13"])
    assert um.read_bytes() == dois.read_bytes()


# ---------------------------------------------------------------------------
# `--orientation`, e a transposição que NÃO é a da ingestão
# ---------------------------------------------------------------------------


def _transposed_text(path: Path) -> str:
    rows = [line.split(",") for line in path.read_text(encoding="utf-8").splitlines() if line]
    return "\n".join(",".join(column) for column in zip(*rows, strict=True)) + "\n"


def test_orientation_transposta_produz_a_mesma_saida(tmp_path: Path) -> None:
    """A mesma matriz em dois arquivos com eixos trocados dá a **mesma** saída.

    É mais forte que "não levanta": eixo trocado
    sobrevive a toda checagem de soma total, então a única asserção que o
    pega é comparar contra a execução na orientação nativa.
    """
    nativo = tmp_path / "nativo.csv"
    assert main(_argv(nativo, method="none")) == EXIT_OK

    transposto = tmp_path / "transposto.csv"
    (tmp_path / "t.csv").write_text(_transposed_text(FIXTURES / "tiny_drift.csv"), encoding="utf-8")
    code = main(
        [
            "corrigir",
            "--measurements",
            str(tmp_path / "t.csv"),
            "--sample-list",
            str(FIXTURES / "tiny_drift_sample_list.csv"),
            "--mode",
            "tiny",
            "--output",
            str(transposto),
            "--orientation",
            "samples-in-rows",
        ]
    )
    assert code == EXIT_OK
    assert transposto.read_bytes() == nativo.read_bytes()


def test_grade_irregular_transposta_e_recusada_com_ragged_rows(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Transpor grade irregular não é definido, e completar com vazio apagaria o defeito.

    O par causal: o **mesmo** arquivo sem a linha curta passa. Sem a segunda
    metade, a guarda poderia estar recusando por qualquer outro motivo.
    """
    regular = _transposed_text(FIXTURES / "tiny_drift.csv")
    irregular = regular.rstrip("\n") + "\ncampo,so\n"

    bom, ruim = tmp_path / "bom.csv", tmp_path / "ruim.csv"
    bom.write_text(regular, encoding="utf-8")
    ruim.write_text(irregular, encoding="utf-8")

    def executa(entrada: Path, saida: Path) -> int:
        return main(
            [
                "corrigir",
                "--measurements",
                str(entrada),
                "--sample-list",
                str(FIXTURES / "tiny_drift_sample_list.csv"),
                "--mode",
                "tiny",
                "--output",
                str(saida),
                "--orientation",
                "samples-in-rows",
            ]
        )

    assert executa(bom, tmp_path / "ok.csv") == EXIT_OK
    assert executa(ruim, tmp_path / "nao.csv") == EXIT_INPUT_REJECTED
    assert "RAGGED_ROWS" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# Entrada recusada
# ---------------------------------------------------------------------------


def test_entrada_com_erro_aborta_com_a_lista_completa(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A lista completa, nunca o primeiro: quatro defeitos pedem quatro linhas.

    Um arquivo com vários problemas relatado um por vez transforma diagnóstico em
    laço de tentativa e erro, que é a experiência a evitar.
    """
    output = tmp_path / "out.csv"
    code = main(
        [
            "corrigir",
            "--measurements",
            str(FIXTURES / "corrupt_ragged.csv"),
            "--sample-list",
            str(FIXTURES / "corrupt_ragged_sample_list.csv"),
            "--mode",
            "tiny",
            "--output",
            str(output),
        ]
    )
    assert code == EXIT_INPUT_REJECTED
    assert not output.exists()
    linhas = [line for line in capsys.readouterr().err.splitlines() if line.startswith("  [")]
    assert linhas, "recusou sem listar nenhum problema"


# ---------------------------------------------------------------------------
# O invólucro executável
# ---------------------------------------------------------------------------


def test_o_ponto_de_entrada_declarado_aponta_para_um_callable() -> None:
    """`refino` no `pyproject` e `refino.cli:main` têm de casar.

    Sem esta paridade, renomear `main` deixaria o `[project.scripts]` apontando
    para nada, e o defeito apareceria só depois de `pip install` — isto é, para
    quem instala e não para quem desenvolve.
    """
    import tomllib

    config = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    alvo = config["project"]["scripts"]["refino"]
    modulo, _, atributo = alvo.partition(":")
    importado = __import__(modulo, fromlist=[atributo])
    assert callable(getattr(importado, atributo)), f"{alvo} não é chamável"


def test_o_modulo_e_executavel_por_python_m(tmp_path: Path) -> None:
    """`python -m refino.cli` funciona, e a evidência é uma execução de verdade.

    Sem isto, `main` estaria testado e o caminho que um usuário de fato digita —
    processo novo, sem `sys.path` de teste — não.
    """
    output = tmp_path / "out.csv"
    resultado = subprocess.run(
        [sys.executable, "-m", "refino.cli", *_argv(output, method="none")],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=REPO,
    )
    assert resultado.returncode == EXIT_OK, resultado.stderr
    assert output.is_file()
