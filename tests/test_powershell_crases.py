"""Crase de MARKDOWN dentro de string do PowerShell — o portão que faltava.

## O defeito, medido em 30/08/2026

A prosa deste repositório cita identificadores entre crases, e o hábito atravessou para dentro
de uma string do PowerShell::

    Passo "as $n rotas que entregaram deixaram tempo em `runs`" ($novas -eq $n) `
        "antes=$antes depois=$depois"

No PowerShell a crase é o **caractere de escape**. ```r`` é retorno de carro, e a crase de fecho
escapa a aspa que deveria terminar a string — que passa a engolir as linhas seguintes até achar
outra aspa. Renderizado isoladamente, o `powershell.exe` diz::

    The string is missing the terminator: ".

## Por que este portão existe, e por que `ParseFile` NÃO basta

**`[Parser]::ParseFile` devolveu «sem erros» sobre o arquivo defeituoso.** Não é falha dele: com
mais conteúdo abaixo, a string não terminada encontra uma aspa alguns tokens à frente e o arquivo
**passa a analisar** — significando outra coisa. É o pior desfecho: verificação que responde
sucesso sobre algo quebrado.

> **O oráculo aqui não pode ser «o arquivo analisa».** Tem de ser a propriedade que distingue
> escape de formatação.

## O oráculo, e por que é ```palavra``` e não «crase seguida de letra»

A regra natural — *"crase seguida de um caractere que não é escape válido"* — **não apanha este
defeito**, e é isso que a torna a escolha errada: ```r`` **é** escape válido, então ```runs```
passaria. O que separa formatação de escape é o **par**: markdown fecha a crase na mesma linha, à
volta de uma palavra. Escape não fecha nada.

Contado antes de escolher, sobre os cinco roteiros de PowerShell da árvore:

===============================  ========  =====================================================
oráculo                          acusa     veredito
===============================  ========  =====================================================
crase seguida de não-escape      **3**     e **perde a linha que quebra**. Ver abaixo
qualquer crase fora de           **9**     acusa ``"`n"`` três vezes, que é a separação de linhas
continuação                                legítima do `congela_windows.ps1`. Vira lista de isenção
```palavra``` na mesma linha       **3**     as três reais, e nenhuma legítima
===============================  ========  =====================================================

A terceira é a que fica, e o motivo de a primeira sair é o dado e não o gosto:

**a regra «crase seguida de não-escape» apanha a linha 82 e PERDE a 155** — e a 155 é a única que
quebra a string. Ali a crase de fecho vem imediatamente antes de uma aspa, e ```"`` *é* escape
válido, então a regra passa por cima. Na 82 a crase de fecho cai antes de um espaço e a regra
acusa. **Uma regra que acerta a ocorrência inofensiva e perde a que derruba o roteiro é pior que
nenhuma**, porque fica verde depois de acusar.

## Escopo declarado

Alcança apenas `scripts/*.ps1` e apenas linhas que **não** são comentário — a prosa do cabeçalho
de cada roteiro usa crase de markdown de propósito, e ali ela não é executada.

**Não** verifica se a string diz a coisa certa, nem se o roteiro faz o que promete. Verifica uma
propriedade sintática, que é a que este defeito tem.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ROTEIROS = sorted((REPO / "scripts").glob("*.ps1"))

# ``palavra`` — crase, um ou mais caracteres de palavra, crase. É a forma da citação de markdown,
# e nenhuma sequência de escape do PowerShell tem essa forma: o escape é UM caractere e não fecha.
CRASE_DE_MARKDOWN = re.compile(r"`\w+`")


def _linhas_de_codigo(texto: str) -> list[tuple[int, str]]:
    """As linhas que o PowerShell executa — sem comentário e sem a continuação de linha final.

    A crase final de uma linha é continuação e é legítima; tirá-la antes de procurar impede o
    portão de acusar toda linha que se estende.
    """
    saida: list[tuple[int, str]] = []
    for numero, linha in enumerate(texto.split("\n"), 1):
        if linha.lstrip().startswith("#"):
            continue
        corpo = linha[:-1] if linha.rstrip().endswith("`") else linha
        saida.append((numero, corpo))
    return saida


def test_nenhum_roteiro_de_powershell_tem_crase_de_markdown_em_string() -> None:
    """``runs`` numa string quebra-a, e o analisador sintático diz que está tudo bem.

    ## A não-vacuidade vem primeiro, e ela é dupla

    *"nenhuma ocorrência"* é satisfeito por uma lista vazia de roteiros **e** por um leitor que
    devolva texto vazio. As duas âncoras estão abaixo, antes da asserção que interessa.
    """
    assert ROTEIROS, "nenhum roteiro `.ps1` foi encontrado: o portão estaria vácuo"

    total_de_linhas = 0
    achados: list[str] = []
    for roteiro in ROTEIROS:
        texto = roteiro.read_text(encoding="utf-8")
        for numero, corpo in _linhas_de_codigo(texto):
            total_de_linhas += 1
            for casado in CRASE_DE_MARKDOWN.findall(corpo):
                achados.append(f"{roteiro.name}:{numero}: {casado} em {corpo.strip()[:70]}")

    assert total_de_linhas > 100, (
        f"só {total_de_linhas} linhas de código lidas em {len(ROTEIROS)} roteiros — "
        "o leitor devolveu quase nada e a asserção abaixo passaria por vacuidade"
    )

    assert not achados, (
        "crase de markdown dentro de código PowerShell — a crase é ESCAPE ali, e a de fecho "
        "escapa a aspa que terminaria a string:\n  " + "\n  ".join(achados)
    )


def test_a_forma_do_defeito_e_reconhecida_e_a_do_escape_legitimo_NAO_e() -> None:  # noqa: N802
    """O par causal do oráculo: ele separa as duas coisas sobre dado em que elas diferem.

    Sem este teste, `CRASE_DE_MARKDOWN` poderia ser qualquer expressão que não casa nada na árvore
    limpa — inclusive uma que nunca casa. A árvore verde não distingue as duas.
    """
    quebra = 'Passo "deixaram tempo em `runs`" ($a -eq $b)'
    legitimo = '$partes = $texto -split "`n"'

    assert CRASE_DE_MARKDOWN.findall(quebra) == ["`runs`"], "não reconhece a forma que quebra"
    assert CRASE_DE_MARKDOWN.findall(legitimo) == [], (
        "acusa a separação de linhas, que é escape legítimo e aparece três vezes no "
        "`congela_windows.ps1` — um oráculo que a acusa vira lista de isenção"
    )

    # E a continuação de linha, que é a outra forma legítima: o corpo sai sem a crase final.
    continuacao = 'Relata "o passo" $ok `'
    ((_, corpo),) = _linhas_de_codigo(continuacao)
    assert not corpo.endswith("`"), "a continuação de linha não foi retirada antes da busca"


# ---------------------------------------------------------------------------
# `$Var:` — o QUALIFICADOR DE DRIVE que o PowerShell vê onde se quis um separador
# ---------------------------------------------------------------------------
#
# ## O defeito, medido em 04/09/2026
#
# Uma linha de detalhe de um portão do congelamento foi escrita assim::
#
#     $detalhe += " (ignorados $($alheios.Count) refino de FORA de $Dist: $ids)"
#
# `$Dist:` não é «o valor de `$Dist`, depois dois-pontos». O `:` depois do nome faz o PowerShell
# ler aquilo como **qualificador de drive** — a mesma forma de `$env:PATH` e `$script:Local` —, e
# o arquivo morre em `ParserError` **antes da primeira linha**::
#
#     Variable reference is not valid. ':' was not followed by a valid variable name character.
#
# **O custo foi um ciclo inteiro de congelamento**, porque o roteiro só é exercitado ao ser corrido
# e ele demora minutos até chegar ao passo que a linha serve. A forma correta é `${Dist}:`.
#
# ## O oráculo, e por que a lista de conhecidos é NOMEADA
#
# `$env:`, `$script:`, `$global:` e as PSDrives de registo são a forma **legítima** e o repositório
# usa-as. O oráculo é `$Nome:` com `Nome` **fora** dessa lista — e a lista vai escrita, e não
# derivada de «o que já está no repositório», porque derivá-la faria o primeiro defeito entrar
# nela.
#
# Contado antes de escolher, sobre os cinco roteiros:
# **45 ocorrências de `$Nome:`, 45 legítimas, 0 acusadas**. Com o defeito plantado, 1.
#
# **E a primeira versão deste oráculo acusou a si mesma:** ela não descartava comentários, e o
# comentário que explica o defeito contém `$Dist:` literalmente. Contar antes de escrever mostrou.

QUALIFICADOR = re.compile(r"\$([A-Za-z_][A-Za-z0-9_]*):")

# As PSDrives e escopos que o PowerShell reconhece, e que este repositório usa.
QUALIFICADORES_LEGITIMOS = frozenset(
    {
        "env",
        "script",
        "global",
        "local",
        "private",
        "using",
        "workflow",
        "function",
        "variable",
        "HKLM",
        "HKCU",
        "HKCR",
    }
)


def _sem_comentario_inline(linha: str) -> str:
    """Corta no primeiro `#` que não esteja dentro de aspas.

    **Aproximação declarada**, e ela é suficiente aqui: o alvo é uma forma que só importa em
    código, e um falso negativo por aspas aninhadas custa menos que a alternativa — analisar
    PowerShell em Python.
    """
    aspas: str | None = None
    for i, c in enumerate(linha):
        if aspas is not None:
            if c == aspas:
                aspas = None
        elif c in "\"'":
            aspas = c
        elif c == "#":
            return linha[:i]
    return linha


def _qualificadores_suspeitos(texto: str) -> list[tuple[int, str]]:
    """`(linha, nome)` de cada `$Nome:` que não é qualificador conhecido. Pura, para o negativo."""
    achados: list[tuple[int, str]] = []
    for numero, linha in _linhas_de_codigo(texto):
        for casa in QUALIFICADOR.finditer(_sem_comentario_inline(linha)):
            if casa.group(1) not in QUALIFICADORES_LEGITIMOS:
                achados.append((numero, casa.group(1)))
    return achados


def test_nenhum_roteiro_interpola_variavel_seguida_de_dois_pontos() -> None:
    """`"... de $Dist: $ids"` não analisa, e o roteiro morre antes da primeira linha."""
    assert ROTEIROS, "âncora: não há roteiro de PowerShell em `scripts/` para varrer"

    # A âncora de não-vacuidade: se o padrão deixasse de casar, a lista abaixo ficaria
    # vazia e o teste passaria sem medir nada. O repositório TEM `$env:` e `$script:` aos montes.
    legitimos = sum(
        1
        for roteiro in ROTEIROS
        for _n, linha in _linhas_de_codigo(roteiro.read_text(encoding="utf-8"))
        for casa in QUALIFICADOR.finditer(_sem_comentario_inline(linha))
        if casa.group(1) in QUALIFICADORES_LEGITIMOS
    )
    assert legitimos >= 10, (
        f"âncora: o padrão achou {legitimos} qualificadores legítimos nos roteiros. Com poucos, "
        "a asserção abaixo é satisfeita por vacuidade — o padrão provavelmente deixou de casar"
    )

    suspeitos = [
        f"{roteiro.name}:{numero}: ${nome}:"
        for roteiro in ROTEIROS
        for numero, nome in _qualificadores_suspeitos(roteiro.read_text(encoding="utf-8"))
    ]
    assert not suspeitos, (
        f"o PowerShell lê estes como QUALIFICADOR DE DRIVE e o roteiro nem analisa: {suspeitos}. "
        "A forma correta é `${Nome}:`. Se for um qualificador novo e legítimo, acrescente-o a "
        "`QUALIFICADORES_LEGITIMOS` — a lista é escrita à mão de propósito"
    )


def test_o_portao_do_qualificador_acusa_a_planta_e_nao_acusa_o_legitimo() -> None:
    """Par causal em processo, com as duas direções e a planta conferida antes do desfecho."""
    legitimo = '$caminho = "$env:APPDATA\refino"\nWrite-Host "$script:Local"\n'
    assert _qualificadores_suspeitos(legitimo) == [], (
        "o portão acusa `$env:` e `$script:`, que são a forma correta — ele reprovaria a árvore "
        "inteira e seria desligado no primeiro dia"
    )

    plantado = legitimo + '$detalhe += " (de FORA de $Dist: $ids)"\n'
    assert plantado != legitimo, "a planta não aterrou"
    achados = _qualificadores_suspeitos(plantado)
    assert [nome for _n, nome in achados] == ["Dist"], (
        f"o portão não viu a planta, ou viu de mais: {achados}"
    )

    # E a forma CORRIGIDA não é acusada — sem isto o portão diria «errado» aos dois lados.
    assert _qualificadores_suspeitos(legitimo + '$d += " de FORA de ${Dist}: $ids"\n') == [], (
        "`${Dist}:` é a forma correta e o portão acusa-a — ele não distingue o conserto do defeito"
    )
