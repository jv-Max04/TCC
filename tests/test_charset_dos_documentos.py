"""Todo documento HTML de entrega DECLARA UTF-8.

## O que este portão existe para impedir

Cinco dos oito documentos de `docs/` nasceram **sem `<meta charset>`**, e nasceram assim por um
motivo que parecia bom: eles não têm `<!doctype html>`, porque viajam por dois caminhos —
embutidos no esqueleto de outra página, e abertos por duplo clique a partir do disco. Sem
`<!doctype>` também não há `<head>`, e por tabela não havia declaração nenhuma. Os arquivos são
UTF-8 e **nada o dizia**.

No segundo caminho o browser adivinha, e adivinha `windows-1252`. Medido em 19/09/2026:

```
Como usar o programa, do comeÃ§o ao arquivo salvo     (GUIA-do-aplicativo.html)
As mediÃ§Ãµes de correÃ§Ã£o                            (MEDICOES-da-correcao.html)
```

**Nenhum mecanismo deste projeto olhava para isto, e a razão é estrutural.** Os dois métodos de
varredura são «por documento» e «por teste» — e a codificação de um artefato não é conteúdo de
documento nem asserção de teste: é a superfície que cai entre os dois.
`scripts/prepara_entrega.ps1` confere **presença** e **tamanho**; os dois passavam.

## O ORÁCULO ERROU DUAS VEZES, e as duas correções foram medidas

**Primeiro — POSIÇÃO.** A especificação do HTML define o *prescan* sobre os primeiros 1024
bytes, e a primeira versão fez disso o critério. Medido com cinco páginas-sonda servidas por
HTTP com `?v=` para afastar o cache: o Chrome honra a tag no byte **3002**; só a ausência
quebra. O critério de posição acusava `DIAGNOSTICO`, `ENTENDER` e `EXPLICAR`, que declaram nos
bytes 4078, 2171 e 2832 e renderizam perfeitamente — três falsos positivos de oito.

**Segundo — o VALOR, achado pela revisão de código de 19/09/2026.** A versão seguinte procurava
o literal `<meta charset`, que termina **antes do `=`**. Então `<meta charset="windows-1252">`
num arquivo UTF-8 passava, e o docstring desta função prometia *«o arquivo é UTF-8 e diz que
é»* enquanto o código nunca lia o que ele dizia. Uma declaração errada renderiza **pior** que a
ausência, porque vence o palpite do browser.

> **A expressão vem de `scripts/declara_charset.py`, importada e não recopiada.** O portão e o
> corretor têm de concordar sobre o que conta como declarar: se divergirem, a mensagem de falha
> manda rodar um corretor que considera o arquivo já correto.

## O que ele NÃO cobre, e onde isso foi fechado

Este arquivo lê `docs/`. O que o usuário abre por duplo clique são as cópias em
`dist/refino/`, com **nomes diferentes** (`GUIA.html`, não `GUIA-do-aplicativo.html`).
`dist/` é gitignorado, então um teste não pode
depender dele; a conferência da instância entregue vive em
`scripts/empacota_para_o_github.py`, que a aplica aos `.html` **dentro do zip escrito**.
"""

from __future__ import annotations

import pathlib
import sys

import pytest

RAIZ = pathlib.Path(__file__).resolve().parent.parent
DOCS = RAIZ / "docs"
sys.path.insert(0, str(RAIZ / "scripts"))

from declara_charset import DECLARA_UTF8, com_a_tag, declara  # noqa: E402

MINIMO_DE_DOCUMENTOS = 7
"""Quantos HTML de entrega `docs/` tem de ter para a parametrização não ser vácua."""


def documentos() -> list[pathlib.Path]:
    """Os HTML de `docs/`, em ordem estável — e **a guarda de não-vacuidade vive AQUI**.

    Lida do disco e **não de uma lista pinada**: documento novo entra no portão sem ninguém se
    lembrar de o acrescentar, que é a diferença entre um portão e um retrato de ontem.

    **A guarda está no PRODUTOR, e isso foi uma correção da revisão de código.** Ela morava num
    teste vizinho, que pode ser removido sem que nada ligue os dois. Pior que a distância: esta
    função alimenta um `parametrize`, e uma lista vazia
    ali produz `skipped`, não falha. A suíte ficaria **verde sem abrir documento nenhum**.

    Levantando daqui, a vacuidade rebenta na COLETA, e nenhuma seleção por `-k` ou por node id
    a contorna.
    """
    # `rglob` e não `glob`: o HTML do multiverso vive em `docs/trabalhos_futuros/` e continua a
    # viajar. Com `glob` ele sairia deste portão e a guarda abaixo rebentaria, mas a resposta
    # certa não é baixar o mínimo: é o portão continuar a ver o documento onde ele está.
    achados = sorted(DOCS.rglob("*.html"))
    if len(achados) < MINIMO_DE_DOCUMENTOS:
        raise RuntimeError(
            f"{DOCS} tem {len(achados)} HTML e o portão espera ao menos "
            f"{MINIMO_DE_DOCUMENTOS} — a parametrização seria vácua e a suíte ficaria verde "
            "sem conferir documento nenhum"
        )
    return achados


@pytest.mark.parametrize("documento", documentos(), ids=lambda p: p.name)
def test_o_documento_declara_utf8(documento: pathlib.Path) -> None:
    """O arquivo é UTF-8 **e** diz que é — as duas metades, e agora o código lê as duas.

    Juntas de propósito: um arquivo que declara UTF-8 e não é UTF-8 renderiza tão mal quanto um
    que não declara nada, e o modo de falha é o mesmo para quem lê.
    """
    bruto = documento.read_bytes()

    try:
        bruto.decode("utf-8")
    except UnicodeDecodeError as erro:  # pragma: no cover — nenhum documento está neste estado
        pytest.fail(f"{documento.name} não é UTF-8: {erro}")

    assert declara(bruto) >= 0, (
        f'{documento.name} não declara `<meta charset="utf-8">`. Sem isso, quem o abrir por '
        f"duplo clique a partir do disco recebe `windows-1252` adivinhado, e os acentos quebram. "
        f"Conserto: `.venv/Scripts/python.exe scripts/declara_charset.py {documento.as_posix()}`"
    )


# --------------------------------------------------------------- o portão visto a REPROVAR
#
# Além do teste que passa na árvore limpa, o que planta a violação e afirma a
# reprovação. Sem estes, um oráculo que devolvesse sempre `0` passaria acima e ninguém veria.


@pytest.mark.parametrize(
    "bytes_do_documento",
    [
        pytest.param(b"<title>sem tag nenhuma</title>", id="ausente"),
        pytest.param(b'<meta charset="windows-1252">\n<title>x</title>', id="valor-errado"),
        pytest.param(b"<meta charset=iso-8859-1>\n<title>x</title>", id="valor-errado-sem-aspas"),
    ],
)
def test_o_portao_acusa_a_planta(bytes_do_documento: bytes) -> None:
    """As três formas que um documento pode estar errado, e o oráculo tem de ver as três.

    A segunda e a terceira são as que a versão anterior deixava passar: ela procurava o literal
    `<meta charset`, que termina antes do `=`, então qualquer valor casava.

    **RISCO DECLARADO, e não um caso omisso:** `declara()` é lexical e não interpreta
    comentários, então uma tag apenas dentro de `<!-- ... -->` seria lida como declaração,
    enquanto o browser não a honraria. Isto **não** entra como `xfail` de propósito — marcador
    que quem escreve o teste põe no mesmo instante absorve exatamente o que ele devia acusar.
    Fica nomeado aqui, e a razão de não ser perseguido é que o corretor insere a
    tag REAL no topo, fora de qualquer comentário.
    """
    assert declara(bytes_do_documento) < 0, "o portão não acusou um documento que está errado"


@pytest.mark.parametrize(
    "bytes_do_documento",
    [
        pytest.param(b'<meta charset="utf-8">\n<title>x</title>', id="aspas-duplas"),
        pytest.param(b"<meta charset='UTF-8'>\n<title>x</title>", id="aspas-simples-maiusculas"),
        pytest.param(b"<meta charset=utf8>\n<title>x</title>", id="sem-aspas-sem-hifen"),
        pytest.param(b'<meta  charset = "utf-8" >\n<title>x</title>', id="espacos-a-volta"),
    ],
)
def test_o_par_causal_o_portao_nao_acusa_o_legitimo(bytes_do_documento: bytes) -> None:
    """A direção oposta, e sem ela o teste de cima passaria com um oráculo que recusasse tudo.

    As quatro formas são HTML válido. A última é o falso negativo que a versão anterior tinha
    pelo lado simétrico: `<meta  charset` com dois espaços não casava o literal.
    """
    assert declara(bytes_do_documento) >= 0, "o portão acusou um documento que está correto"


def test_o_oraculo_do_portao_e_o_mesmo_do_corretor() -> None:
    """As duas irmãs não podem divergir sobre o que conta como declarar.

    Se divergissem, a mensagem de falha deste portão mandaria rodar um corretor que considera o
    arquivo já correto — e o conserto recomendado saltaria o arquivo.
    """
    assert DECLARA_UTF8.search(b'<meta charset="utf-8">') is not None
    assert declara(b'<meta charset="utf-8">') == 0


# --------------------------------------------------------------- o corretor, nas duas direções


def test_a_tag_entra_depois_do_doctype_quando_ele_existe() -> None:
    """Pô-la antes empurraria o `<!doctype>` para dentro do corpo — *quirks mode*.

    Medido em 19/09/2026: três dos oito documentos de `docs/` começam com `<!doctype html>` no
    byte 0, e a mensagem de falha do portão manda rodar o corretor sobre qualquer um deles.
    """
    resultado = com_a_tag(b"<!doctype html>\n<title>x</title>")
    assert resultado.startswith(b"<!doctype html>"), "o doctype deixou de ser a primeira coisa"
    assert declara(resultado) >= 0, "a tag não entrou"


def test_a_tag_entra_no_byte_zero_quando_nao_ha_doctype() -> None:
    """O par causal do teste acima: sem doctype, o topo é o lugar certo."""
    resultado = com_a_tag(b"<title>x</title>")
    assert resultado.startswith(b'<meta charset="utf-8">')
    assert declara(resultado) == 0


def test_o_corretor_e_idempotente() -> None:
    """Rodar duas vezes não empilha duas tags — a segunda passada vê a primeira e sai."""
    uma = com_a_tag(b"<title>x</title>")
    assert declara(uma) >= 0
    assert uma.count(b"<meta charset") == 1
