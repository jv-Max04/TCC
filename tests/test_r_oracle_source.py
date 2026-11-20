"""Propriedades do TEXTO do `tools.R` que sustentam as duas trocas.

Por que este arquivo não é `docker` e roda no CI de commit: ele não executa R
nenhum. Lê o arquivo do oráculo e conta. A justificativa de trocar
`library(tidyverse)` por `library(magrittr)` é uma **contagem** — zero `%>%`
dentro de `signalCorrection` —, e contagem que só é verificada no contêiner
noturno é contagem que ninguém vê deixar de valer no momento em que o arquivo
muda.

O que cada teste protege, e é sempre a mesma coisa por ângulos diferentes: o
`entrypoint.R` reescreve duas linhas de uma cópia do oráculo. Reescrita cega é o
primeiro elo da cadeia. Aqui os alvos são fixados pelo texto exato, e o
runner reprova com `patch_target_absent` quando o texto muda — mas reprovar no
runner é tarde, porque o runner roda à noite. Estes testes reprovam no commit.
"""

from __future__ import annotations

import re

import pytest

from tests.conftest import tools_r_path

# ---------------------------------------------------------------------------
# Localização de `signalCorrection` no arquivo
# ---------------------------------------------------------------------------

# Definição de topo: nome no início da linha seguido de `<-` ou `=` e `function`.
_TOP_LEVEL_DEF = re.compile(r"^[A-Za-z_.][A-Za-z0-9_.]*\s*(?:<-|=)\s*function")


@pytest.fixture(scope="module")
def tools_r_lines() -> list[str]:
    return tools_r_path().read_text(encoding="utf-8", errors="surrogateescape").splitlines()


def _span_of(lines: list[str], name: str) -> tuple[int, int]:
    """Intervalo de linhas (1-indexado, inclusivo) do corpo de uma função de topo.

    Delimitar pela PRÓXIMA definição de topo, e não por contagem de chaves, é
    deliberado: contar chaves exigiria tratar string e comentário, e o arquivo
    tem `"\\\\"` em vários lugares. A próxima definição é um delimitador que não
    depende de parsing.
    """
    start = next(
        (
            i
            for i, line in enumerate(lines, 1)
            if re.match(rf"^{re.escape(name)}\s*(?:<-|=)\s*function", line)
        ),
        None,
    )
    assert start is not None, f"{name} não encontrada em tools.R"
    end = next(
        (i for i, line in enumerate(lines[start:], start + 1) if _TOP_LEVEL_DEF.match(line)),
        len(lines) + 1,
    )
    return start, end - 1


def test_signal_correction_span_is_where_the_audit_says() -> None:
    """Fixa o intervalo que todas as outras contagens usam.

    Sem isto, uma contagem de zero poderia significar "não há `%>%`" ou "o
    intervalo ficou vazio porque o nome mudou" — dois significados para o mesmo
    número.
    """
    lines = tools_r_path().read_text(encoding="utf-8", errors="surrogateescape").splitlines()
    start, end = _span_of(lines, "signalCorrection")
    assert (start, end) == (301, 1692), (
        f"signalCorrection agora ocupa {start}..{end}, não 301..1692. As contagens "
        f"foram medidas sobre o intervalo antigo e precisam ser refeitas."
    )


# ---------------------------------------------------------------------------
# A troca da linha 1: `library(tidyverse)` -> `library(magrittr)`
# ---------------------------------------------------------------------------


def test_line_one_is_the_only_library_tidyverse(tools_r_lines: list[str]) -> None:
    """O alvo da primeira troca existe, é único, e está na linha 1.

    `source()` do `tools.R` executa exatamente um statement, e é este. É por ele
    que o metapacote entraria na imagem.
    """
    hits = [i for i, line in enumerate(tools_r_lines, 1) if line.strip() == "library(tidyverse)"]
    assert hits == [1], f"esperava `library(tidyverse)` só na linha 1; achei em {hits}"


def test_signal_correction_never_uses_the_pipe(tools_r_lines: list[str]) -> None:
    """**Este é o teste que sustenta a troca.**

    Se `signalCorrection` usasse `%>%` em qualquer ponto, trocar o metapacote por
    `magrittr` continuaria correto — `magrittr` é quem fornece o operador. O que
    este zero prova é mais forte: nem o operador é usado ali, então a superfície
    da troca é ainda menor do que a justificativa precisa.
    """
    start, end = _span_of(tools_r_lines, "signalCorrection")
    body = tools_r_lines[start - 1 : end]
    offenders = [(start + i, line.strip()) for i, line in enumerate(body) if "%>%" in line]
    assert offenders == [], (
        f"`%>%` dentro de signalCorrection em {[n for n, _ in offenders]}. "
        f"A justificativa media zero; remeça a contagem antes de manter a troca."
    )


def test_the_pipe_is_needed_exactly_once_and_in_serrf_read_data(tools_r_lines: list[str]) -> None:
    """O outro lado da troca: `magrittr` não é opcional.

    Contagem de zero em `signalCorrection` só justifica trocar o metapacote se
    algo no caminho ainda precisar do operador. Precisa: `serrf.read.data`, que
    `signalCorrection` chama na linha 373. Sem `magrittr` a leitura falha, e a
    troca teria de ser remoção — que é outra decisão.
    """
    start, end = _span_of(tools_r_lines, "serrf.read.data")
    body = tools_r_lines[start - 1 : end]
    users = [
        start + i
        for i, line in enumerate(body)
        if "%>%" in line and not line.strip().startswith("#")
    ]
    assert users, "nenhum `%>%` em serrf.read.data — `magrittr` deixou de ser necessário"
    assert 248 in users, f"esperava o `%>%` da linha 248; achei em {users}"


# ---------------------------------------------------------------------------
# A troca da linha 320: `pacman::p_load` reduzido
# ---------------------------------------------------------------------------

_PLOAD_TARGET = (
    "pacman::p_load(affy, parallel,ranger,caret,pcaMethods,ggplot2,tidyr,"
    "graphics,grDevices,Hmisc,gtools,cowplot,RColorBrewer,readr,plotly,"
    "stringr,GGally,dplyr,e1071,officer,bootstrap,pdist)"
)

# Os quinze sem uma única referência em `signalCorrection`. `parallel`,
# `graphics` e `grDevices` estão fora da lista porque são usados; `affy` e
# `caret` também, em ramos que esta imagem não habilita.
_NEVER_CALLED = (
    "pcaMethods",
    "ggplot2",
    "tidyr",
    "Hmisc",
    "gtools",
    "cowplot",
    "RColorBrewer",
    "readr",
    "plotly",
    "stringr",
    "dplyr",
    "GGally",
    "officer",
    "e1071",
    "pdist",
)


def test_pload_target_is_exact_and_unique(tools_r_lines: list[str]) -> None:
    """O alvo da segunda troca, fixado pelo texto exato.

    A linha 319 é a variante comentada e inclui `metabolomicsR`; casar por
    substring pegaria as duas. O `entrypoint.R` casa a linha inteira sem espaço
    em volta, e é isso que este teste fixa.
    """
    hits = [i for i, line in enumerate(tools_r_lines, 1) if line.strip() == _PLOAD_TARGET]
    assert hits == [320], f"esperava o `p_load` ativo só na linha 320; achei em {hits}"
    assert tools_r_lines[318].strip().startswith("#"), (
        "a linha 319 deveria ser a variante comentada"
    )


@pytest.mark.parametrize("package", _NEVER_CALLED)
def test_fifteen_packages_are_never_called(tools_r_lines: list[str], package: str) -> None:
    """Nenhum dos quinze aparece fora das duas linhas de `p_load`.

    É a evidência que autoriza não instalá-los. Feita pacote por pacote de
    propósito: um teste agregado diria "algum dos quinze é usado" e deixaria a
    identificação para quem investigasse.
    """
    start, end = _span_of(tools_r_lines, "signalCorrection")
    offenders = [
        start + i
        for i, line in enumerate(tools_r_lines[start - 1 : end])
        if package in line and "p_load" not in line
    ]
    assert offenders == [], (
        f"`{package}` referenciado em {offenders} dentro de signalCorrection. "
        f"Ele foi classificado como não chamado e a imagem não o instala."
    )


@pytest.mark.parametrize(
    ("package", "expected_lines", "call_line"),
    [
        # `ranger` aparece três vezes, e só UMA é chamada: a 1135. As outras duas
        # são a lista de argumentos do `parSapply` (1071 abre, 1203 fecha), que
        # passa a própria função aos workers em vez de dar `library()` neles.
        # Contar as três como "chamadas" seria errar a contagem: o nome está lá, a
        # quantidade que a pergunta pede é outra.
        ("ranger", [1071, 1135, 1203], 1135),
        ("bootstrap", [344], 344),
    ],
)
def test_the_two_installed_packages_are_really_called(
    tools_r_lines: list[str], package: str, expected_lines: list[int], call_line: int
) -> None:
    """O inverso: os que a imagem instala têm chamada real, na linha auditada.

    Portão que só reprova numa direção mede metade. Se `ranger` deixasse de ser
    chamado, a imagem estaria compilando um pacote por nada — e ninguém saberia.
    """
    start, end = _span_of(tools_r_lines, "signalCorrection")
    mentions = [
        start + i
        for i, line in enumerate(tools_r_lines[start - 1 : end])
        if re.search(rf"\b{package}\b", line) and "p_load" not in line
    ]
    assert mentions == expected_lines, (
        f"`{package}` citado em {mentions}, esperado em {expected_lines}"
    )
    invoked = tools_r_lines[call_line - 1]
    assert re.search(rf"\b{package}(?:::[A-Za-z._]+)?\s*\(", invoked), (
        f"a linha {call_line} deveria conter a CHAMADA de `{package}`; ela diz: {invoked.strip()!r}"
    )


# ---------------------------------------------------------------------------
# O quinto defeito da especificação: separador de caminho misturado na mesma função
# ---------------------------------------------------------------------------


def test_path_join_uses_backslash_while_a_sibling_uses_forward_slash(
    tools_r_lines: list[str],
) -> None:
    """Fixa o defeito que obriga a tradução do runner.

    A 1518 divide em `/` — correto no POSIX. A 1522 **rejunta com `\\`**. A 1625
    usa `//`. É por isso que o `QC - RSD.csv` cai no lugar certo e os datasets
    normalizados não: no Windows os dois separadores funcionam, e o laboratório
    nunca viu diferença.

    Se alguém consertar o `tools.R`, este teste reprova — e é o desfecho certo,
    porque a tradução do `entrypoint.R` passaria a ser desnecessária e teria de
    sair junto.
    """
    join_line = tools_r_lines[1521]
    assert 'collapse = "\\\\"' in join_line, (
        f"a linha 1522 deveria juntar caminho com barra invertida; ela diz: {join_line.strip()!r}"
    )
    split_line = tools_r_lines[1517]
    assert 'strsplit(file_location,"/")' in split_line, (
        f"a linha 1518 deveria dividir em barra normal; ela diz: {split_line.strip()!r}"
    )
    qc_line = tools_r_lines[1624]
    assert '"//"' in qc_line, (
        f"a linha 1625 deveria usar `//` — o separador misturado é o defeito; "
        f"ela diz: {qc_line.strip()!r}"
    )


# O catálogo do `"all"`, na linha 312. Quinze nomes.
_CATALOGUE = (
    "mTIC",
    "sum",
    "median",
    "PQN",
    "contrast",
    "quantile",
    "linear",
    "liwong",
    "cubic",
    "batch_ratio",
    "batch_loess",
    "SERRF",
    "svm",
    "nomis",
    "bmis",
)


def test_the_three_method_counts_are_different_numbers(tools_r_lines: list[str]) -> None:
    """Quinze no catálogo, quatorze com ramo, treze que produzem matriz.

    Os três são fáceis de confundir porque descrevem "os métodos do `tools.R`", e
    é a forma mais barata de errar: o número está à mão — o catálogo
    da linha 312 é uma lista literal — e usá-lo não parece uma decisão.

    O que cada um mede:

    - **quinze** nomes aceitos por `methods`, e é o domínio que a fronteira do
      `entrypoint.R` valida: nome fora daqui é erro de digitação do usuário.
    - **quatorze** ramos `if('X' %in% methods)`. `bmis` não tem ramo.
    - **treze** que produzem matriz corrigida: os quatorze menos `nomis`, cujo
      ramo chama função inexistente. Dez medem RSD in-sample e três avaliam fora
      da amostra, que é a partição que a especificação já usava.
    """
    catalogue_line = tools_r_lines[311]
    for name in _CATALOGUE:
        assert name in catalogue_line, f"`{name}` saiu do catálogo da linha 312"

    branches = {
        match.group(1)
        for line in tools_r_lines
        if (match := re.search(r"if\(['\"]([A-Za-z_]+)['\"] *%in% *methods\)", line))
    }
    assert len(branches) == 14, f"esperava 14 ramos; achei {len(branches)}: {sorted(branches)}"
    assert set(_CATALOGUE) - branches == {"bmis"}, (
        f"o catálogo sem ramo deveria ser só {{bmis}}; é {set(_CATALOGUE) - branches}"
    )

    # `none` está entre os que atribuem `qc_RSDs` e NÃO é método do catálogo: ele
    # é sempre calculado. Contá-lo como método daria quatorze produtivos, e é
    # exatamente a confusão que este teste existe para travar.
    assert "none" not in _CATALOGUE
    produce = branches - {"nomis"}
    assert len(produce) == 13, f"esperava 13 métodos produtivos; achei {len(produce)}"


def test_bmis_has_no_branch_and_is_ignored_in_silence(tools_r_lines: list[str]) -> None:
    """`bmis` é o pior dos dois modos de morte, porque não levanta.

    `nomis` falha ruidosamente — chama função inexistente e cai no `tryCatch`.
    `bmis` não tem ramo nenhum: aparece no comentário da 300 e no catálogo da
    312, e em mais nenhum lugar. Pedi-lo não levanta, não avisa e não escreve
    nada, então a execução termina com status de sucesso sem o dataset pedido.

    É o modo de falha mais insidioso, e é por isso que o
    `entrypoint.R` o recusa na fronteira com causa própria em vez de deixar a
    ausência aparecer como `method_produced_no_dataset` — ou, pior, não aparecer.
    """
    mentions = [i for i, line in enumerate(tools_r_lines, 1) if "bmis" in line]
    assert mentions == [300, 312], (
        f"`bmis` citado em {mentions}; esperado só no comentário (300) e no catálogo (312). "
        f"Se ganhou ramo, a recusa do `entrypoint.R` ficou obsoleta."
    )


def test_normalise_is_called_but_never_defined(tools_r_lines: list[str]) -> None:
    """`nomis` chama função que não existe — a mesma classe do `add.label()`.

    O único pacote que a forneceria, `metabolomicsR`, está apenas na linha 319,
    comentada. O `entrypoint.R` recusa o método com causa própria em vez de
    deixá-lo cair no `tryCatch` e devolver "faltou dataset" sem explicação.
    """
    calls = [i for i, line in enumerate(tools_r_lines, 1) if re.search(r"\bNormalise\s*\(", line)]
    definitions = [
        i
        for i, line in enumerate(tools_r_lines, 1)
        if re.search(r"\bNormalise\s*(?:<-|=)\s*function", line)
    ]
    assert calls == [1434], f"esperava a chamada de `Normalise` na 1434; achei em {calls}"
    assert definitions == [], (
        f"`Normalise` definida em {definitions} — a recusa do runner ficou obsoleta"
    )
    metabolomics = [i for i, line in enumerate(tools_r_lines, 1) if "metabolomicsR" in line]
    assert metabolomics == [319], (
        f"`metabolomicsR` citado em {metabolomics}, esperado só na 319 (comentada)"
    )
