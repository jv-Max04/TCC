"""Os sete `.html` de documentação são ARTEFATOS MONTADOS, e este portão lê-os como tal.

## Por que ele existe

Os outros portões sobre estes documentos leem o texto, e **nenhum olha para o `<style>`**. Uma
classe usada e não definida renderiza como texto de corpo, sem erro nenhum: um parágrafo marcado
fica indistinguível dos outros. **Um portão sobre o TEXTO não alcança a APRESENTAÇÃO.**

O `GUIA` usava `class="cilada"` e `class="errado"` — as duas que os outros documentos definem —
sem definir nenhuma: o bloco saía como prosa corrente e a palavra marcada como errada saía igual
às certas.

## As três propriedades, e por que estas três

Cada uma tem um modo de falha que **não** produz erro nenhum ao abrir a página — o navegador
tolera as três, e o desfecho é uma página que parece certa:

| Propriedade | O que o defeito produz |
|---|---|
| classe usada sem regra no `<style>` | o parágrafo perde a marca, ou a tabela não rola |
| tag que não fecha, ou fecha o que não abriu | o resto da seção herda o estilo errado |
| `href="#x"` sem `id="x"` | o índice leva a lugar nenhum |

**A classe órfã apanha duas famílias de uma vez:** a classe que ninguém definiu, e o nome digitado
ao contrário — foi assim que apareceu `class="rolar"` num documento que define `.rolo`, uma letra
de diferença, numa tabela que por isso nunca rolou.

## O que ele NÃO faz, declarado

Não julga se o desenho é bom, nem se a regra CSS faz o que o nome promete. Ele fecha o buraco
mecânico: a classe existir e não ter regra, a tag não fechar, a âncora não ter destino. Que o
`.nota` pareça uma nota é julgamento humano, e nenhum mecanismo desta forma o alcança.

**E há um caso que ele NÃO apanha, medido ao plantar a violação.** O oráculo das classes é
*«este nome aparece como seletor em algum lugar do `<style>`»*, e isso é grosseiro de propósito:
uma classe mencionada **apenas** num seletor descendente — `.nota b { … }` sem `.nota { … }` —
conta como definida, e o elemento que a leva continua sem estilo próprio. Distinguir os dois
exigiria parsear CSS, e o que se ganharia é um caso que não ocorreu nos sete. **A imprecisão vai
declarada em vez de suposta:** ao remover só a primeira das duas regras de `.nota`, este portão
continua verde — foi assim que o limite se mediu.
"""

from __future__ import annotations

import re
from html.parser import HTMLParser
from pathlib import Path
from typing import Final

import pytest

from tests.conftest import REPO

# **Os SETE, e não os seis que viajam.** O `DIAGNOSTICO` não entra no executável, e entra aqui:
# quem o abre é quem lê a documentação do repositório, e uma página malformada é igualmente
# malformada fora da pasta do usuário. Derivar de `docs/*.html` em vez de listar é deliberado —
# um documento novo nasce coberto, e uma lista escrita seria o retrato de ontem.
#
# `rglob`, e não `glob`: o HTML do multiverso vive em `docs/trabalhos_futuros/` e continua a
# viajar; com `glob` ele sairia do portão sem reprovação nenhuma.
DOCUMENTOS: Final[tuple[Path, ...]] = tuple(sorted((REPO / "docs").rglob("*.html")))

# Tags que não fecham, por definição de HTML. Sem esta lista o verificador de equilíbrio acusaria
# todo `<br>` como aberto e nunca fechado — o falso positivo que mata um portão no primeiro dia.
VAZIOS: Final[frozenset[str]] = frozenset(
    {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "source", "wbr"}
)

# **`head` e `html` são isentos, e a isenção é MEDIDA e não conveniente.** Três dos sete fecham
# `</head>` e `</html>` sem os abrir — o documento começa direto no conteúdo. É malformação
# tolerada por todo navegador e não muda um pixel; incluí-la aqui faria o portão nascer vermelho
# em três arquivos por uma propriedade que ninguém vai consertar. Se algum dia se consertar, esta
# isenção sai. Ela é uma afirmação sobre o produto, e quem a confere é a linha de baixo, que
# exige que ela continue a ser verdade.
FECHAM_SEM_ABRIR: Final[frozenset[str]] = frozenset({"head", "html"})


class _Equilibrio(HTMLParser):
    """A pilha de tags abertas, e o que fechou fora de ordem."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.pilha: list[tuple[str, int]] = []
        self.erros: list[str] = []

    def handle_starttag(self, tag: str, attrs: object) -> None:
        if tag not in VAZIOS:
            self.pilha.append((tag, self.getpos()[0]))

    def handle_endtag(self, tag: str) -> None:
        if tag in VAZIOS:
            return
        if not self.pilha or self.pilha[-1][0] != tag:
            if tag not in FECHAM_SEM_ABRIR:
                self.erros.append(f"</{tag}> na linha {self.getpos()[0]} fecha fora de ordem")
            for i in range(len(self.pilha) - 1, -1, -1):
                if self.pilha[i][0] == tag:
                    del self.pilha[i:]
                    return
            return
        self.pilha.pop()


def _classes_usadas(texto: str) -> set[str]:
    """Toda classe que algum elemento declara. `class="a b"` conta duas."""
    return {c for attr in re.findall(r'class="([^"]+)"', texto) for c in attr.split()}


def _classes_definidas(texto: str) -> set[str]:
    """Os seletores de classe de todos os blocos `<style>` do documento.

    **Lê o `<style>` e não o arquivo inteiro**, e a distinção decide: `.nota` aparece em prosa
    dentro do corpo — «a classe `.nota`» —, e aquilo não é uma regra, é um exemplo.
    """
    estilo = "\n".join(re.findall(r"<style[^>]*>(.*?)</style>", texto, re.DOTALL))
    return set(re.findall(r"\.([A-Za-z][\w-]*)", estilo))


def test_ha_documentos_a_conferir() -> None:
    """A âncora de não-vacuidade, e ela vem PRIMEIRO.

    Os três testes abaixo são parametrizados por `DOCUMENTOS`. Com o glob a devolver vazio, o
    `pytest` não os corre e a suíte fica verde sobre zero arquivos — o desfecho de sucesso que
    não mede nada.
    """
    assert len(DOCUMENTOS) >= 7, f"o glob de `docs/*.html` devolveu {len(DOCUMENTOS)}: {DOCUMENTOS}"


@pytest.mark.parametrize("documento", DOCUMENTOS, ids=lambda p: p.stem)
def test_toda_classe_usada_tem_regra_no_proprio_documento(documento: Path) -> None:
    """Classe sem regra é marca que não aparece.

    Cada `.html` carrega o seu próprio `<style>` inteiro, de propósito: eles viajam soltos na pasta
    do usuário, sem CSS externo e sem internet. Então a regra tem de estar **no mesmo arquivo**,
    e herdá-la de um irmão não é possível.
    """
    texto = documento.read_text(encoding="utf-8")
    usadas = _classes_usadas(texto)

    assert usadas, f"âncora: {documento.name} não usa classe nenhuma, e a asserção seria vácua"

    orfas = sorted(usadas - _classes_definidas(texto))

    assert not orfas, (
        f"{documento.name} usa {len(orfas)} classe(s) que o próprio `<style>` não define: {orfas}. "
        f"O elemento perde a marca em silêncio — o navegador não acusa nada"
    )


@pytest.mark.parametrize("documento", DOCUMENTOS, ids=lambda p: p.stem)
def test_as_tags_fecham(documento: Path) -> None:
    """Tag que não fecha faz o resto da seção herdar o estilo errado."""
    equilibrio = _Equilibrio()
    equilibrio.feed(documento.read_text(encoding="utf-8"))

    abertas = [f"<{t}> na linha {linha}" for t, linha in equilibrio.pilha]

    assert not equilibrio.erros and not abertas, f"{documento.name}: {equilibrio.erros + abertas}"


def _ancoras_internas(texto: str) -> set[str]:
    """Os destinos de `href="#…"`. Só os internos — `http` e caminho relativo saem daqui.

    Link externo não é verificável sem rede, e este portão não abre rede.
    """
    return {a for a in re.findall(r'href="#([^"]+)"', texto) if a}


def test_algum_documento_tem_link_interno() -> None:
    """A não-vacuidade do teste de âncoras, e ela é da POPULAÇÃO e não de cada documento.

    **A primeira versão pôs `assert ancoras` dentro do teste parametrizado, e reprovou em três dos
    sete** — `DIAGNOSTICO`, `GUIA` e `MULTIVERSO` não têm link interno nenhum, porque são
    documentos lineares sem índice. *Não ter índice não é defeito*, e um portão que o chame defeito
    nasce vermelho por uma propriedade que ninguém vai consertar.

    > O que a âncora de não-vacuidade protege é o MECANISMO, não cada caso. A pergunta certa não
    > é *"este documento tem links?"* — é *"o verificador de links alguma vez olha para algum?"*.
    > A primeira confunde ausência legítima com defeito; a segunda é a que impede o teste de passar
    > por vacuidade nos sete de uma vez.
    """
    com_links = {d.name: len(_ancoras_internas(d.read_text(encoding="utf-8"))) for d in DOCUMENTOS}
    total = sum(com_links.values())

    assert total > 0, (
        f"nenhum dos {len(DOCUMENTOS)} documentos tem link interno: o teste de âncoras abaixo "
        f"passaria por vacuidade nos sete. Contagem por arquivo: {com_links}"
    )


@pytest.mark.parametrize("documento", DOCUMENTOS, ids=lambda p: p.stem)
def test_toda_ancora_interna_tem_destino(documento: Path) -> None:
    """`href="#x"` sem `id="x"` leva o leitor a lugar nenhum.

    Documento sem link interno passa por ausência, e a legitimidade disso está na docstring de
    `test_algum_documento_tem_link_interno`, que é quem garante que o mecanismo não fica inerte.
    """
    texto = documento.read_text(encoding="utf-8")

    quebradas = sorted(_ancoras_internas(texto) - set(re.findall(r'id="([^"]+)"', texto)))

    assert not quebradas, (
        f"{documento.name} tem {len(quebradas)} link(s) interno(s) para `id` que não existe: "
        f"{quebradas}"
    )
