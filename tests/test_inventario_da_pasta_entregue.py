"""Todo arquivo que VIAJA na pasta do usuário está descrito no `LEIA-ME.txt`.

## O defeito que este portão paga, e ele passou por quatro listas

`scripts/congela_windows.ps1` copia documentos para `dist\refino\\`, e
`scripts/prepara_entrega.ps1` confere que eles estão lá e não estão truncados. Entre os dois há
**quatro** listas, e o comentário do roteiro de entrega celebra, na entrada do `FAQ.html` em
07/09/2026, que *"as QUATRO listas foram escritas na mesma edicao"*.

**Havia uma quinta, e o próprio comentário é o que a tornou invisível.** O `LEIA-ME.txt` — a
primeira coisa que quem recebe a pasta abre — tem um inventário que diz o que é cada arquivo, e o
`FAQ.html` nunca entrou nele. Entre 07/09 e 14/09/2026 a pasta levava um documento de 35 KB que o
seu próprio índice não mencionava.

> Afirmação de varredura completa declara as superfícies que ela NÃO abre. *"As quatro listas"*
> é uma afirmação de completude, e a quinta superfície existia quando ela foi escrita.

## Por que a direção é esta, e não a inversa

Este portão afirma **cobertura**: tudo o que viaja está descrito. Ele **não** exige o contrário —
que tudo o que o `LEIA-ME` menciona viaje —, e a assimetria é deliberada: o texto cita
`LEIA-ME-das-saidas.txt`, que vive DENTRO de `saidas/`, e nomes de exemplo como `..._c18.csv`. Uma
bijeção acusaria os dois, e a resposta certa seria afrouxar o portão — que é como um portão aprende
a ser ignorado.

## O oráculo é DERIVADO do roteiro que produz a pasta

Os nomes saem dos `$destino... = Join-Path $Dist "X"` do `congela_windows.ps1`, e não de uma lista
escrita aqui. Escrevê-los à mão criaria a **sexta** superfície, com o mesmo defeito que este arquivo
existe para fechar: o oráculo codificaria a forma que a pasta tinha hoje.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Final

import pytest

REPO: Final = Path(__file__).resolve().parent.parent
CONGELA: Final = REPO / "scripts" / "congela_windows.ps1"
LEIAME: Final = REPO / "docs" / "LEIA-ME-aplicativo.txt"

# `$destinoX = Join-Path $Dist "NOME"` — a forma com que o roteiro nomeia cada arquivo que copia
# para a raiz da pasta entregue.
DESTINO: Final = re.compile(r'^\$destino\w+\s*=\s*Join-Path\s+\$Dist\s+"([^"]+)"', re.M)

# **O que NÃO é documento e por isso não entra**: `VERSAO.txt` é carimbo de procedência e o
# `LEIA-ME.txt` é o próprio índice — um arquivo que se nomeasse a si mesmo não acrescenta nada a
# quem já o está a ler.
FORA: Final = frozenset({"LEIA-ME.txt", "VERSAO.txt"})


def _copiados() -> list[str]:
    texto = CONGELA.read_text(encoding="utf-8", errors="replace")
    return sorted({n for n in DESTINO.findall(texto) if n not in FORA})


def test_a_ancora_o_roteiro_nomeia_arquivos_a_copiar() -> None:
    """Sem esta âncora, um `regex` que deixasse de casar tornaria o portão abaixo VÁCUO.

    Asserção de ausência — *"nenhum copiado ficou de fora"* — é satisfeita por vacuidade quando a
    lista está vazia, e lista vazia é o que se obtém de uma extração que falhou. A guarda vai na
    mesma função que a usa, e antes dela.
    """
    copiados = _copiados()
    assert len(copiados) >= 6, (
        f"o roteiro de congelamento nomeia {len(copiados)} arquivos a copiar: {copiados}. "
        "Com poucos, este portão mede o nada — o padrão de extração provavelmente mudou"
    )
    pastas = [n for n in copiados if "." not in n]
    assert pastas, (
        f"nenhuma pasta entre os copiados: {copiados}. O negativo do cabeçalho de pasta seria "
        "parametrizado sobre o nada"
    )


def _cabecalho_de_pasta(nome: str) -> re.Pattern[str]:
    """`A PASTA nome/` em início de linha — a forma com que o inventário do `LEIA-ME` abre a seção
    de cada pasta que viaja (`saidas/`, `R/` e `vias-dados/` em 01/10/2026)."""
    return re.compile(rf"^A PASTA {re.escape(nome)}/", re.M)


def _descrito(nome: str, texto: str) -> bool:
    """Arquivo: o nome em qualquer lugar do texto. Pasta: o CABEÇALHO da sua seção no inventário."""
    if "." in nome:
        return nome in texto
    return _cabecalho_de_pasta(nome).search(texto) is not None


def _pastas() -> list[str]:
    return [n for n in _copiados() if "." not in n]


@pytest.mark.parametrize("nome", _copiados())
def test_todo_arquivo_que_viaja_esta_descrito_no_leiame(nome: str) -> None:
    """Direção única: o que viaja é descrito. Ver a docstring do módulo para a assimetria.

    **Pasta é procurada pelo cabeçalho `A PASTA nome/`, em início de linha**, que é a forma com que
    o inventário do `LEIA-ME` abre a seção de cada uma. Com o oráculo `nome in texto` para tudo, a
    pasta `R` passaria por vacuidade: qualquer «R» maiúsculo da prosa satisfaz a substring. A
    primeira correção procurava `nome/` em qualquer lugar, e uma revisão mediu que ela aceitava
    menção fora do inventário: numa
    cópia sem as seções «A PASTA R/» e «A PASTA vias-dados/», `R/` sobrevivia numa nota do fim do
    texto e `vias-dados/` em duas, e o portão ficava verde. Arquivo continua com `nome in texto`.
    """
    texto = LEIAME.read_text(encoding="utf-8")
    procurado = nome if "." in nome else f"A PASTA {nome}/"
    # `relative_to` só quando cabe: com o caminho trocado por uma cópia fora do clone (o par
    # causal), ele levantava ValueError DENTRO da mensagem, e a reprovação saía pelo motivo errado.
    onde = LEIAME.relative_to(REPO) if LEIAME.is_relative_to(REPO) else LEIAME
    assert _descrito(nome, texto), (
        f"`{nome}` viaja na pasta entregue e o `LEIA-ME.txt` não traz `{procurado}`. "
        "Quem recebe a pasta abre um arquivo que o índice dela não explica. "
        f"Acrescente um parágrafo em {onde}, no bloco do inventário"
        + ("" if "." in nome else f", aberto pela linha `{procurado}`")
    )


def _sem_a_secao(texto: str, nome: str) -> str:
    """O texto sem a seção da pasta: do cabeçalho até a próxima linha em branco dupla."""
    cabecalho = _cabecalho_de_pasta(nome).search(texto)
    assert cabecalho is not None, f"o LEIA-ME não tem a seção `A PASTA {nome}/` para tirar"
    fim = texto.find("\n\n\n", cabecalho.end())
    return texto[: cabecalho.start()] + (texto[fim:] if fim >= 0 else "")


@pytest.mark.parametrize("nome", _pastas())
def test_pasta_sem_a_secao_do_inventario_reprova(nome: str) -> None:
    """O negativo, em processo: o `LEIA-ME` real sem SÓ a seção do inventário de uma pasta reprova.

    A planta é o argumento da função sob teste, então não há intermediário que possa não a receber.
    As duas âncoras vão antes: a seção existe no texto real, e a planta tirou alguma coisa.
    """
    texto = LEIAME.read_text(encoding="utf-8")
    assert _descrito(nome, texto)
    plantado = _sem_a_secao(texto, nome)
    assert len(plantado) < len(texto)
    assert not _descrito(nome, plantado), (
        f"sem a seção `A PASTA {nome}/`, o oráculo ainda dá `{nome}` por descrita: ele aceita "
        "menção fora do inventário, que é o defeito de P2#6"
    )
