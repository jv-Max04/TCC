"""O leitor e o escritor RESTRITOS do formato de serialização do R.

O modo externo do enriquecimento lê a resposta do servidor do MetaboAnalyst sem o `unserialize` do
R, porque desserializar com o R é rodar o que veio de fora. Este arquivo prova as duas metades da
fronteira, e a ordem importa:

1. **o que o leitor ACEITA sai com os valores certos** — contra fixtures que o próprio R gravou
   (`data/fixtures/rds/`, o `gera.R` ao lado), e não contra bytes que o escritor deste módulo
   produz. Ler só o que o próprio escritor grava provaria que os dois lados concordam entre si, não
   que concordam com o R;
2. **o que ele RECUSA é recusado pelo motivo certo** — toda recusa afirma a mensagem, e não só o
   tipo da exceção. Uma recusa pelo motivo errado (uma função lida como lixo e reprovada adiante por
   «bytes depois do objeto») passaria num `pytest.raises` sem `match` e seria indistinguível de
   guarda ativa.

Os casos montados à mão vêm em PAR sempre que a guarda depende de um detalhe: os mesmos bytes, com
o detalhe trocado, são aceitos. Sem o par, um teste de recusa passaria também sobre bytes que
estivessem malformados por outro motivo — a planta tem de cair sobre dado em que as duas coisas
diferem.

**Nenhuma resposta real do servidor está aqui.** Ela traz dados derivados do KEGG, cuja licença não
permite redistribuir; o que precisa da forma da resposta é montado com `escreve_rds`.
"""

from __future__ import annotations

import gzip
import re
import struct

import pytest

from refino.io import rds_restrito as rds
from refino.io.rds_restrito import (
    Lista,
    Objeto,
    RdsRecusadoError,
    Simbolo,
    Vetor,
    escreve_rds,
    inventario,
    le_rds,
)
from tests.conftest import REPO

FIXTURES = REPO / "data" / "fixtures" / "rds"
FORMATOS = ("ser", "rds")

# ---------------------------------------------------------------------------
# Montagem de bytes à mão. Os códigos são os de `Rinternals.h`, repetidos aqui de propósito: se o
# teste importasse as constantes do módulo, um código errado no módulo seria um código errado no
# teste, e os dois concordariam.
# ---------------------------------------------------------------------------

_SYM, _LIST, _CHAR, _LGL, _INT, _REAL, _STR, _VEC = 1, 2, 9, 10, 13, 14, 16, 19
_ALTREP, _NILVALUE, _REF = 238, 254, 255
_TEM_ATRIBUTO, _TEM_ROTULO = 1 << 9, 1 << 10
_ASCII, _UTF8, _LATIN1, _BYTES = 1 << 6, 1 << 3, 1 << 2, 1 << 1


def _i(*valores: int) -> bytes:
    return struct.pack(f">{len(valores)}i", *valores)


def _cabecalho(versao: int = 2) -> bytes:
    return b"X\n" + _i(versao, (4 << 16) | (4 << 8) | 3, (2 << 16) | (3 << 8))


def _charsxp(texto: str, nivel: int = _ASCII, bruto: bytes | None = None) -> bytes:
    dados = texto.encode("utf-8") if bruto is None else bruto
    return _i(_CHAR | (nivel << 12), len(dados)) + dados


def _simbolo(nome: str) -> bytes:
    return _i(_SYM) + _charsxp(nome)


def _real(*valores: float) -> bytes:
    return _i(_REAL, len(valores)) + struct.pack(f">{len(valores)}d", *valores)


def _inteiro(*valores: int) -> bytes:
    return _i(_INT, len(valores), *valores)


def _texto(*valores: str) -> bytes:
    return _i(_STR, len(valores)) + b"".join(_charsxp(v) for v in valores)


def _altrep(classe: str, pacote: str, estado: bytes, atributos: bytes | None = None) -> bytes:
    """Um ALTREP na forma em que o R 4.4.3 o grava: a descrição é a pairlist (classe, pacote, tipo).

    A forma foi lida do `aceito.ser` gerado pelo R — o `1:10` sai como `compact_intseq` —, e não
    da documentação: é ela que garante que o par aceito/recusado abaixo difere SÓ pelo nome.
    """
    descricao = (
        _i(_LIST) + _simbolo(classe) + _i(_LIST) + _simbolo(pacote) + _i(_LIST) + _inteiro(_INT)
    )
    fim = _i(_NILVALUE) if atributos is None else atributos
    return _i(_ALTREP) + descricao + _i(_NILVALUE) + estado + fim


def _fixture(nome: str, formato: str) -> bytes:
    return (FIXTURES / f"{nome}.{formato}").read_bytes()


def _cru(nome: str) -> bytes:
    """Os bytes XDR sem a compressão, para as âncoras sobre a forma que o R gravou."""
    return gzip.decompress(_fixture(nome, "rds"))


def _vetor(objeto: Objeto) -> Vetor:
    assert isinstance(objeto, Vetor), objeto
    return objeto


def _lista(objeto: Objeto) -> Lista:
    assert isinstance(objeto, Lista), objeto
    return objeto


# ---------------------------------------------------------------------------
# O que o R gravou, lido com os valores certos
# ---------------------------------------------------------------------------


def test_as_fixtures_cobrem_os_caminhos_que_os_testes_de_leitura_afirmam() -> None:
    """Âncora de que os testes de leitura exercitam o que dizem exercitar.

    Ler `1:10` certo não prova nada sobre o ramo ALTREP se o R o tivesse gravado expandido; ler um
    texto acentuado não prova nada sobre a marca UTF-8 se ele viesse sem marca. O valor natural é o
    que exercita menos do caminho, então a forma dos bytes é conferida, e não só os valores.
    """
    cru = _cru("aceito")
    assert cru == _fixture("aceito", "ser"), "o .rds e o .ser deviam ser o mesmo objeto"
    assert cru[:6] == b"X\n" + _i(3)[:4], "o R 4.4.3 grava o formato 3; o escritor daqui, o 2"
    assert cru.count(_i(_ALTREP) + _i(_LIST) + _simbolo("compact_intseq")) == 1
    assert cru.count(_charsxp("β-Alanine", _UTF8)) == 1
    # «names» é gravado UMA vez como símbolo e reaparece por referência (REFSXP) nas outras duas
    # listas nomeadas: é o que faz a leitura abaixo passar pelo ramo de referência.
    assert cru.count(_simbolo("names")) == 1


@pytest.mark.parametrize("formato", FORMATOS)
def test_aceito_le_a_matriz_com_dim_e_dimnames(formato: str) -> None:
    raiz = _lista(le_rds(_fixture("aceito", formato)))
    assert raiz.nomes() == (
        "analSet",
        "textos",
        "logicos",
        "inteiros",
        "sequencia",
        "vazio",
        "fator",
    )
    anal = _lista(raiz.campo("analSet"))
    matriz = _vetor(anal.campo("ora.mat"))
    assert matriz.tipo == "real"
    # A ordem é a do R, por coluna: a primeira coluna é `total` das duas vias.
    assert matriz.valores == (10.0, 5.0, 1.5, 0.5, 3.0, 1.0, 0.0123, 0.4, 0.1, 1.0, 0.05, 0.8)
    assert _vetor(matriz.atributos["dim"]).valores == (2, 6)
    linhas, colunas = _lista(matriz.atributos["dimnames"]).itens
    assert _vetor(linhas).valores == ("Urea Cycle", "Glycolysis")
    assert _vetor(colunas).valores == ("total", "expected", "hits", "Raw p", "Holm p", "FDR")

    acertos = _lista(anal.campo("ora.hits"))
    assert acertos.nomes() == ("Urea Cycle", "Glycolysis")
    assert _vetor(acertos.campo("Urea Cycle")).valores == ("Urea", "L-Arginine", "Ornithine")
    assert _vetor(acertos.campo("Glycolysis")).valores == ("Pyruvate",)


@pytest.mark.parametrize("formato", FORMATOS)
def test_aceito_le_textos_com_na_e_utf8(formato: str) -> None:
    textos = _vetor(_lista(le_rds(_fixture("aceito", formato))).campo("textos"))
    assert textos.tipo == "texto"
    assert textos.valores == ("Ácido cítrico", None, "β-Alanine", "")


@pytest.mark.parametrize("formato", FORMATOS)
def test_aceito_le_logico_e_inteiro_com_na(formato: str) -> None:
    raiz = _lista(le_rds(_fixture("aceito", formato)))
    logicos = _vetor(raiz.campo("logicos"))
    assert logicos.tipo == "logico"
    assert logicos.valores == (True, False, None)
    # `1 == True` em Python: sem conferir o tipo, um leitor que devolvesse 0/1 passaria.
    assert [type(v) for v in logicos.valores] == [bool, bool, type(None)]

    inteiros = _vetor(raiz.campo("inteiros"))
    assert inteiros.tipo == "inteiro"
    assert inteiros.valores == (1, None, -7)


@pytest.mark.parametrize("formato", FORMATOS)
def test_aceito_expande_a_sequencia_compacta(formato: str) -> None:
    sequencia = _vetor(_lista(le_rds(_fixture("aceito", formato))).campo("sequencia"))
    assert sequencia.tipo == "inteiro"
    assert sequencia.valores == tuple(range(1, 11))
    assert sequencia.atributos == {}


@pytest.mark.parametrize("formato", FORMATOS)
def test_aceito_le_o_nulo_e_o_fator_com_classe(formato: str) -> None:
    raiz = _lista(le_rds(_fixture("aceito", formato)))
    assert raiz.campo("vazio") is None
    fator = _vetor(raiz.campo("fator"))
    assert fator.tipo == "inteiro"
    assert fator.valores == (1, 2, 1)
    assert _vetor(fator.atributos["levels"]).valores == ("a", "b")
    assert _vetor(fator.atributos["class"]).valores == ("factor",)


def test_o_inventario_do_aceito_nomeia_cada_folha_com_tipo_e_tamanho() -> None:
    assert inventario(le_rds(_fixture("aceito", "rds"))) == [
        ("raiz$analSet$ora.mat", "real", 12),
        ("raiz$analSet$ora.hits$Urea Cycle", "texto", 3),
        ("raiz$analSet$ora.hits$Glycolysis", "texto", 1),
        ("raiz$textos", "texto", 4),
        ("raiz$logicos", "logico", 3),
        ("raiz$inteiros", "inteiro", 3),
        ("raiz$sequencia", "inteiro", 10),
        ("raiz$vazio", "nulo", 0),
        ("raiz$fator", "inteiro", 3),
    ]


def test_o_inventario_nomeia_lista_vazia_simbolo_e_item_sem_nome() -> None:
    """Item sem nome entra pela posição. A grafia exacta do caminho dele não é afirmada aqui.

    O inventário escreve `raiz$[[1]]` e o leitor, nas mensagens, `raiz[[1]]`; a carga do modo
    externo só tem folhas nomeadas, e congelar uma das duas grafias seria fixar uma escolha que
    ninguém fez.
    """
    linhas = inventario(Lista((Lista(()), Simbolo("s"))))
    assert [(t, n) for _, t, n in linhas] == [("lista", 0), ("símbolo", 1)]
    assert [c.endswith(f"[[{i}]]") for i, (c, _, _) in enumerate(linhas, start=1)] == [True, True]


# ---------------------------------------------------------------------------
# Código recusado, nos dois formatos
# ---------------------------------------------------------------------------

_RECUSAS_DAS_FIXTURES = [
    ("com_funcao", "função (CLOSXP)"),
    ("com_ambiente", "ambiente (ENVSXP)"),
    ("com_formula", "linguagem (LANGSXP)"),
]


@pytest.mark.parametrize("formato", FORMATOS)
@pytest.mark.parametrize(("nome", "tipo"), _RECUSAS_DAS_FIXTURES)
def test_recusa_codigo_nomeando_o_tipo_e_o_caminho(nome: str, tipo: str, formato: str) -> None:
    """`list(ok = 1, <código>)`: o segundo item é recusado, e a mensagem diz qual e onde.

    O `match` é a mensagem inteira até o tipo, de propósito: um leitor que pulasse o código e
    reprovasse adiante por «bytes depois do objeto» também levantaria `RdsRecusadoError`.
    """
    with pytest.raises(RdsRecusadoError, match=rf"^raiz\[\[2\]\]: {re.escape(tipo)} recusado"):
        le_rds(_fixture(nome, formato))


def test_a_lista_das_fixtures_recusadas_sem_o_codigo_e_aceita() -> None:
    """O par das recusas acima: a mesma forma, `list(ok = 1, <dado>)`, passa.

    Sem ele a recusa poderia vir de a lista nomeada em si não ser lida.
    """
    objeto = Lista(
        (Vetor("real", (1.0,)), Vetor("real", (2.0,))),
        {"names": Vetor("texto", ("ok", "f"))},
    )
    assert le_rds(escreve_rds(objeto)) == objeto


@pytest.mark.parametrize(
    ("codigo", "tipo"),
    [(3, "função (CLOSXP)"), (4, "ambiente (ENVSXP)"), (24, "bytes (RAWSXP)"), (99, "tipo 99")],
)
def test_tipo_fora_do_dado_puro_e_recusado_sem_ler_o_corpo(codigo: int, tipo: str) -> None:
    """O código é recusado no cabeçalho do item: o corpo que vem depois nem é lido.

    O corpo plantado aqui é lixo de propósito — se o leitor tentasse percorrê-lo, a mensagem seria
    outra.
    """
    with pytest.raises(RdsRecusadoError, match=rf"^raiz: {re.escape(tipo)} recusado"):
        le_rds(_cabecalho() + _i(codigo) + b"\xff" * 16)


# ---------------------------------------------------------------------------
# Cabeçalho, versão, fim e truncamento
# ---------------------------------------------------------------------------


def test_cabecalho_que_nao_e_xdr_e_recusado() -> None:
    # «A\n» é o cabeçalho do formato ASCII do R, que este leitor não lê.
    # O resto dos bytes é o do `aceito.ser`, que é aceito: os dois lados do par diferem só nos dois
    # primeiros bytes.
    completo = _fixture("aceito", "ser")
    assert isinstance(le_rds(completo), Lista)
    with pytest.raises(RdsRecusadoError, match=r"^não é serialização XDR do R"):
        le_rds(b"A\n" + completo[2:])


def test_versao_1_e_recusada_e_a_2_com_o_mesmo_corpo_e_aceita() -> None:
    corpo = _i(_NILVALUE)
    assert le_rds(_cabecalho(2) + corpo) is None
    with pytest.raises(RdsRecusadoError, match="versão de serialização 1 não aceita"):
        le_rds(_cabecalho(1) + corpo)


def test_formato_3_com_nome_de_codificacao_absurdo_e_recusado() -> None:
    with pytest.raises(RdsRecusadoError, match="nome de codificação com 0 bytes"):
        le_rds(_cabecalho(3) + _i(0) + _i(_NILVALUE))


def test_bytes_sobrando_no_fim_sao_recusados() -> None:
    inteiro = escreve_rds(Vetor("inteiro", (1, 2)))
    assert le_rds(inteiro) == Vetor("inteiro", (1, 2))
    with pytest.raises(RdsRecusadoError, match=r"^4 bytes depois do objeto"):
        le_rds(inteiro + _i(_NILVALUE))


def test_todo_prefixo_proprio_do_aceito_e_recusado_como_truncado() -> None:
    """Cortar em QUALQUER posição dá «o conteúdo acabou», e não outra mensagem.

    Vale por construção: os bytes antes do corte são os do arquivo válido, então toda guarda que
    olhe para eles passa, e a primeira leitura que falha é a que atravessa o corte. Por isso a
    afirmação é sobre todos os 1.009 prefixos do `aceito.ser`, e não sobre um corte escolhido.
    """
    completo = _fixture("aceito", "ser")
    assert len(completo) == 1009
    for corte in range(len(completo)):
        with pytest.raises(RdsRecusadoError, match=r"^o conteúdo acabou"):
            le_rds(completo[:corte])


def test_gzip_truncado_e_ilegivel() -> None:
    comprimido = _fixture("aceito", "rds")
    with pytest.raises(RdsRecusadoError, match=r"^gzip ilegível"):
        le_rds(comprimido[: len(comprimido) // 2])


def test_gzip_com_lixo_e_ilegivel() -> None:
    with pytest.raises(RdsRecusadoError, match=r"^gzip ilegível"):
        le_rds(b"\x1f\x8b" + b"isto nao e deflate" * 4)


# ---------------------------------------------------------------------------
# Tetos. Os reais são grandes demais para um teste montar; o `monkeypatch` baixa cada um e o par
# acima/no teto mostra que é ELE que recusa.
# ---------------------------------------------------------------------------


def test_bytes_acima_do_teto_sao_recusados_antes_de_ler(monkeypatch: pytest.MonkeyPatch) -> None:
    dados = escreve_rds(Vetor("inteiro", (1, 2, 3)))
    monkeypatch.setattr(rds, "MAX_BYTES", len(dados))
    assert le_rds(dados) == Vetor("inteiro", (1, 2, 3))
    monkeypatch.setattr(rds, "MAX_BYTES", len(dados) - 1)
    with pytest.raises(RdsRecusadoError, match=rf"^{len(dados)} bytes acima do teto"):
        le_rds(dados)


def test_gzip_que_descomprime_acima_do_teto_e_recusado(monkeypatch: pytest.MonkeyPatch) -> None:
    """A bomba de descompressão: o comprimido cabe no teto, o descomprimido não."""
    cru = escreve_rds(Vetor("inteiro", (0,) * 200))
    comprimido = gzip.compress(cru)
    teto = 100
    assert len(comprimido) <= teto < len(cru), (len(comprimido), len(cru))
    monkeypatch.setattr(rds, "MAX_BYTES", teto)
    with pytest.raises(RdsRecusadoError, match=f"^descomprimido passa do teto de {teto} bytes"):
        le_rds(comprimido)
    monkeypatch.setattr(rds, "MAX_BYTES", len(cru))
    assert le_rds(comprimido) == Vetor("inteiro", (0,) * 200)


def test_texto_acima_do_teto_e_recusado(monkeypatch: pytest.MonkeyPatch) -> None:
    """O teto é em BYTES de UTF-8: «ββ» cabe em 4, «ββa» não."""
    monkeypatch.setattr(rds, "MAX_TEXTO", 4)
    assert _vetor(le_rds(escreve_rds(Vetor("texto", ("abcd", "ββ"))))).valores == ("abcd", "ββ")
    with pytest.raises(RdsRecusadoError, match=r"^raiz\[2\]: texto de 5 bytes fora do teto de 4"):
        le_rds(escreve_rds(Vetor("texto", ("abcd", "ββa"))))


def test_comprimento_acima_do_teto_e_recusado(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rds, "MAX_COMPRIMENTO", 3)
    assert le_rds(escreve_rds(Vetor("real", (1.0, 2.0, 3.0)))) == Vetor("real", (1.0, 2.0, 3.0))
    with pytest.raises(RdsRecusadoError, match=r"^raiz: comprimento 4 fora do teto de 3"):
        le_rds(escreve_rds(Vetor("real", (1.0, 2.0, 3.0, 4.0))))
    with pytest.raises(RdsRecusadoError, match=r"^raiz: comprimento 4 fora do teto de 3"):
        le_rds(escreve_rds(Lista((None, None, None, None))))


def test_comprimento_longo_passa_pelo_mesmo_teto(monkeypatch: pytest.MonkeyPatch) -> None:
    """O R grava vetor longo com `-1` e o comprimento em dois inteiros: a guarda vale para ele."""
    longo = _cabecalho() + _i(_INT, -1, 0, 4, 7, 8, 9, 10)
    assert le_rds(longo) == Vetor("inteiro", (7, 8, 9, 10))
    monkeypatch.setattr(rds, "MAX_COMPRIMENTO", 3)
    with pytest.raises(RdsRecusadoError, match=r"^raiz: comprimento 4 fora do teto de 3"):
        le_rds(longo)
    with pytest.raises(RdsRecusadoError, match=r"^raiz: comprimento 4294967296 fora do teto"):
        le_rds(_cabecalho() + _i(_INT, -1, 1, 0))


def test_comprimento_negativo_e_recusado() -> None:
    with pytest.raises(RdsRecusadoError, match=r"^raiz: comprimento -5 fora do teto"):
        le_rds(_cabecalho() + _i(_INT, -5))


def _aninhada(niveis: int) -> Lista:
    objeto: Objeto = None
    for _ in range(niveis):
        objeto = Lista((objeto,))
    assert isinstance(objeto, Lista)
    return objeto


def test_profundidade_no_teto_e_aceita_e_acima_e_recusada() -> None:
    """Sem `monkeypatch`: o teto real é pequeno o bastante para montar os dois lados dele."""
    teto = rds.MAX_PROFUNDIDADE
    assert le_rds(escreve_rds(_aninhada(teto))) == _aninhada(teto)
    with pytest.raises(RdsRecusadoError, match=rf"profundidade acima de {teto}$"):
        le_rds(escreve_rds(_aninhada(teto + 1)))


def test_itens_acima_do_teto_sao_recusados(monkeypatch: pytest.MonkeyPatch) -> None:
    # A raiz e os três nulos são quatro itens.
    monkeypatch.setattr(rds, "MAX_ITENS", 4)
    assert le_rds(escreve_rds(Lista((None, None, None)))) == Lista((None, None, None))
    with pytest.raises(RdsRecusadoError, match=r"^mais de 4 itens"):
        le_rds(escreve_rds(Lista((None, None, None, None))))


# ---------------------------------------------------------------------------
# Referência (REFSXP)
# ---------------------------------------------------------------------------


def _lista_que_reusa_names(indice: int) -> bytes:
    """`list(x = c(a = 7L))` como o R a grava: o segundo «names» é REFERÊNCIA ao primeiro.

    O vetor de dentro grava o símbolo `names` (referência 1); a lista de fora o reusa por índice.
    """
    dentro = _i(_INT | _TEM_ATRIBUTO, 1, 7) + _i(_LIST | _TEM_ROTULO) + _simbolo("names")
    dentro += _texto("a") + _i(_NILVALUE)
    fora = _i(_LIST | _TEM_ROTULO) + _i((indice << 8) | _REF) + _texto("x") + _i(_NILVALUE)
    return _cabecalho() + _i(_VEC | _TEM_ATRIBUTO, 1) + dentro + fora


def test_referencia_a_simbolo_ja_lido_e_resolvida() -> None:
    lido = _lista(le_rds(_lista_que_reusa_names(1)))
    assert lido.nomes() == ("x",)
    assert _vetor(lido.campo("x")).atributos == {"names": Vetor("texto", ("a",))}


@pytest.mark.parametrize("indice", [2, 99])
def test_referencia_que_nao_aponta_para_simbolo_e_recusada(indice: int) -> None:
    """O mesmo objeto do teste acima, só com o índice trocado."""
    with pytest.raises(RdsRecusadoError, match=f"referência {indice} a algo que não é símbolo"):
        le_rds(_lista_que_reusa_names(indice))


def test_referencia_com_indice_no_inteiro_seguinte_e_recusada_sem_simbolo() -> None:
    """Índice 0 nos bits altos manda ler o índice no inteiro seguinte — e ele também é conferido."""
    with pytest.raises(RdsRecusadoError, match=r"^raiz: referência 7 a algo que não é símbolo"):
        le_rds(_cabecalho() + _i(_REF, 7))


# ---------------------------------------------------------------------------
# ALTREP
# ---------------------------------------------------------------------------


def test_altrep_compacto_da_base_montado_a_mao_e_expandido() -> None:
    """O par aceito das recusas abaixo: os bytes só diferem no nome da classe ou do pacote."""
    assert le_rds(_cabecalho() + _altrep("compact_intseq", "base", _real(5, 3, 1))) == Vetor(
        "inteiro", (3, 4, 5, 6, 7)
    )
    assert le_rds(_cabecalho() + _altrep("compact_intseq", "base", _real(3, 0, -1))) == Vetor(
        "inteiro", (0, -1, -2)
    )
    assert le_rds(_cabecalho() + _altrep("compact_realseq", "base", _real(3, 1.5, 1))) == Vetor(
        "real", (1.5, 2.5, 3.5)
    )


def _par(cabeca: bytes, cauda: bytes) -> bytes:
    """O estado de um embrulho como o R o grava: o par pontilhado `CONS(x, meta)` (LISTSXP sem
    atributo nem rótulo, a cabeça, e a cauda no lugar do `NULL` que fecharia uma pairlist)."""
    return _i(_LIST) + cabeca + cauda


def test_altrep_embrulho_da_base_devolve_o_vetor_de_dentro() -> None:
    estado = _par(_inteiro(4, 5), _inteiro(0, 0))
    assert le_rds(_cabecalho() + _altrep("wrap_integer", "base", estado)) == Vetor(
        "inteiro", (4, 5)
    )
    lista = _par(_i(_VEC, 1) + _inteiro(4), _inteiro(0, 0))
    assert le_rds(_cabecalho() + _altrep("wrap_list", "base", lista)) == Lista(
        (Vetor("inteiro", (4,)),)
    )


@pytest.mark.parametrize("formato", FORMATOS)
def test_os_embrulhos_que_o_r_grava_sao_lidos(formato: str) -> None:
    """Os seis `wrap_*` de `embrulho.ser`, gravados pelo R 4.4.3 com `wrap_meta` (revisão de
    02/10/2026). Até então o leitor só tinha visto o estado de um embrulho montado à mão, como
    lista, e recusava TODO embrulho real — inclusive o que `sort()` produz."""
    cru = _fixture("embrulho", "ser")
    assert _cru("embrulho") == cru, "o .rds e o .ser deviam ser o mesmo objeto"
    # âncora: o estado do `wrap_real` está nos bytes como par pontilhado, não como pairlist
    assert cru.count(_par(_real(2.5, 1.5), _inteiro(0, 0))) == 1
    assert cru.count(_simbolo("wrap_integer")) == 1
    lido = _lista(le_rds(_fixture("embrulho", formato)))
    assert lido.nomes() == ("real", "inteiro", "texto", "logico", "lista", "nomeado")
    assert lido.itens[:4] == (
        Vetor("real", (2.5, 1.5)),
        Vetor("inteiro", (3, None, 1)),
        Vetor("texto", ("b", "a")),
        Vetor("logico", (True, None)),
    )
    assert lido.itens[4] == Lista((Vetor("inteiro", (1,)), Vetor("texto", ("z",))))
    assert lido.itens[5] == Vetor("real", (1.5, 2.5), {"names": Vetor("texto", ("a", "b"))})


def test_altrep_leva_os_atributos_gravados_depois_do_estado() -> None:
    """`c(a = 1L, b = 2L, c = 3L)` compacto: os nomes vêm depois do estado e ficam no vetor."""
    nomes = _i(_LIST | _TEM_ROTULO) + _simbolo("names") + _texto("a", "b", "c") + _i(_NILVALUE)
    lido = le_rds(_cabecalho() + _altrep("compact_intseq", "base", _real(3, 1, 1), nomes))
    assert lido == Vetor("inteiro", (1, 2, 3), {"names": Vetor("texto", ("a", "b", "c"))})


def _vetor_com_atributo_de_celula(atributo_da_celula: bytes) -> bytes:
    """`c(x = 1L)` cuja célula da pairlist de atributos traz ela própria um atributo.

    O R grava o atributo da célula ANTES do rótulo; o leitor o percorre e o descarta — e percorrer
    é o que importa: um código escondido ali é recusado como em qualquer outro lugar.
    """
    celula = _i(_LIST | _TEM_ATRIBUTO | _TEM_ROTULO) + atributo_da_celula + _simbolo("names")
    return _cabecalho() + _i(_INT | _TEM_ATRIBUTO, 1, 1) + celula + _texto("x") + _i(_NILVALUE)


def test_o_atributo_da_celula_de_pairlist_e_lido_e_descartado() -> None:
    dado = _i(_LIST | _TEM_ROTULO) + _simbolo("y") + _inteiro(9) + _i(_NILVALUE)
    lido = le_rds(_vetor_com_atributo_de_celula(dado))
    assert lido == Vetor("inteiro", (1,), {"names": Vetor("texto", ("x",))})
    with pytest.raises(
        RdsRecusadoError, match=r"^raiz@atributos@atributos: função \(CLOSXP\) recusado"
    ):
        le_rds(_vetor_com_atributo_de_celula(_i(3)))


@pytest.mark.parametrize(
    ("classe", "pacote"),
    [("deferred_string", "base"), ("mmap_integer", "base"), ("compact_intseq", "pacote_alheio")],
)
def test_altrep_de_classe_desconhecida_e_recusado(classe: str, pacote: str) -> None:
    """`deferred_string` é uma classe REAL da base — e o estado dela é um objeto R arbitrário."""
    with pytest.raises(RdsRecusadoError, match=f"^raiz: ALTREP {pacote}::{classe} recusado"):
        le_rds(_cabecalho() + _altrep(classe, pacote, _real(5, 3, 1)))


@pytest.mark.parametrize(
    "estado",
    [_real(5, 3, 2), _real(2.5, 1, 1), _real(5, 1), _inteiro(5, 3, 1)],
    ids=["passo-2", "n-fracionario", "dois-campos", "estado-inteiro"],
)
def test_sequencia_compacta_fora_da_forma_e_recusada(estado: bytes) -> None:
    with pytest.raises(RdsRecusadoError, match=r"^raiz: sequência compacta"):
        le_rds(_cabecalho() + _altrep("compact_intseq", "base", estado))


def test_sequencia_inteira_com_inicio_fracionario_e_recusada() -> None:
    with pytest.raises(RdsRecusadoError, match=r"^raiz: sequência inteira com início 1.5"):
        le_rds(_cabecalho() + _altrep("compact_intseq", "base", _real(3, 1.5, 1)))


@pytest.mark.parametrize(
    ("estado", "aceita"),
    [
        (_real(3, 2**31 - 3, 1), True),
        (_real(3, 2**31 - 2, 1), False),
        (_real(2, -(2**31) + 2, -1), True),
        (_real(2, -(2**31) + 1, -1), False),
    ],
    ids=["ate-o-maior", "passa-do-maior", "ate-o-menor", "passa-do-menor-que-e-o-NA"],
)
def test_sequencia_inteira_fica_no_inteiro_do_r(estado: bytes, aceita: bool) -> None:
    """O R não grava sequência inteira fora de 32 bits, e -2^31 é o NA dele: uma que passe dali é
    malformada, e lê-la como inteiro Python daria valores que nenhum R produz (revisão de
    02/10/2026). Os pares ficam um de cada lado da fronteira."""
    corpo = _cabecalho() + _altrep("compact_intseq", "base", estado)
    if aceita:
        assert len(_vetor(le_rds(corpo)).valores) in (2, 3)
    else:
        with pytest.raises(RdsRecusadoError, match=r"^raiz: sequência inteira além do inteiro"):
            le_rds(corpo)


def test_sequencia_compacta_passa_pelo_teto_de_comprimento(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """O par de `test_comprimento_acima_do_teto_e_recusado` para a forma compacta, cujo `n` vem
    num real do estado e não no campo de comprimento."""
    monkeypatch.setattr(rds, "MAX_COMPRIMENTO", 3)
    assert le_rds(_cabecalho() + _altrep("compact_intseq", "base", _real(3, 1, 1))) == Vetor(
        "inteiro", (1, 2, 3)
    )
    with pytest.raises(RdsRecusadoError, match=r"^raiz: sequência compacta fora da forma"):
        le_rds(_cabecalho() + _altrep("compact_intseq", "base", _real(4, 1, 1)))


def _lista_de_compactos(*tamanhos: int) -> bytes:
    itens = b"".join(_altrep("compact_intseq", "base", _real(n, 1, 1)) for n in tamanhos)
    return _cabecalho() + _i(_VEC, len(tamanhos)) + itens


def test_o_orcamento_de_valores_cobra_a_forma_compacta(monkeypatch: pytest.MonkeyPatch) -> None:
    """Uma sequência compacta é UM item de 76 bytes que vira até `MAX_COMPRIMENTO` valores: medido
    pela revisão de 02/10/2026, 436 bytes viravam 5 milhões de valores e 200 MB. O orçamento soma
    os valores de todos os vetores, compactos ou não, e cobra ANTES de construir."""
    monkeypatch.setattr(rds, "MAX_VALORES", 18)
    # A lista conta os seus 2 itens, e cada ALTREP conta 1 pelo inteiro do tipo na descrição e 3
    # pelo estado real (n, início, passo): 2 + (4 + 4) + (4 + 4) = 18 cabe; com 5, 19 não.
    assert len(_lista(le_rds(_lista_de_compactos(4, 4))).itens) == 2
    with pytest.raises(RdsRecusadoError, match=r"^raiz\[\[2\]\]: mais de 18 valores"):
        le_rds(_lista_de_compactos(4, 5))


def test_o_orcamento_de_valores_cobra_o_vetor_por_extenso(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rds, "MAX_VALORES", 3)
    assert le_rds(escreve_rds(Vetor("real", (1.0, 2.0, 3.0)))) == Vetor("real", (1.0, 2.0, 3.0))
    with pytest.raises(RdsRecusadoError, match=r"^raiz: mais de 3 valores"):
        le_rds(escreve_rds(Vetor("real", (1.0, 2.0, 3.0, 4.0))))


def test_o_ataque_medido_pela_revisao_e_recusado_sem_construir() -> None:
    """Com os tetos de produção: duas sequências de 3 milhões cabem cada uma em `MAX_COMPRIMENTO`
    e passam juntas de `MAX_VALORES`, em menos de 400 bytes."""
    corpo = _lista_de_compactos(3_000_000, 3_000_000)
    assert len(corpo) < 400
    assert rds.MAX_COMPRIMENTO >= 3_000_000 and rds.MAX_VALORES < 6_000_000
    with pytest.raises(RdsRecusadoError, match="valores"):
        le_rds(corpo)


def test_o_teto_de_valores_e_o_que_os_bytes_escreveriam_por_extenso() -> None:
    assert rds.MAX_VALORES == rds.MAX_BYTES // 4


def test_gzip_com_bloco_deflate_corrompido_e_recusado() -> None:
    """`zlib.error` não é `OSError`: até a revisão de 02/10/2026 ele escapava de `le_rds`."""
    bom = bytearray(gzip.compress(escreve_rds(Vetor("inteiro", (1, 2, 3))), mtime=0))
    assert le_rds(bytes(bom)) == Vetor("inteiro", (1, 2, 3))
    bom[10] = 0x07  # BFINAL=1 e BTYPE=11, o tipo de bloco reservado
    with pytest.raises(RdsRecusadoError, match=r"^gzip ilegível"):
        le_rds(bytes(bom))


# ---------------------------------------------------------------------------
# Malformações que não são tipo recusado
# ---------------------------------------------------------------------------

_MALFORMADOS = [
    ("logico-2", _i(_LGL, 1, 2), "^raiz: lógico com valor 2"),
    ("texto-sem-charsxp", _i(_STR, 1) + _inteiro(1), r"^raiz\[1\]: esperava texto \(CHARSXP\)"),
    ("texto-bytes", _i(_STR, 1) + _charsxp("x", _BYTES), r"^raiz\[1\]: texto marcado como bytes"),
    (
        "texto-nao-utf8",
        _i(_STR, 1) + _charsxp("", _UTF8, b"\xff\xfe"),
        r"^raiz\[1\]: texto que não é UTF-8",
    ),
    ("simbolo-sem-nome", _i(_SYM, _CHAR, -1), "^raiz: símbolo sem nome"),
    (
        "simbolo-como-valor-na-raiz",
        _i(_SYM) + _charsxp("system"),
        "^raiz: símbolo como valor recusado",
    ),
    (
        "simbolo-como-valor-numa-lista",
        _i(_VEC, 1) + _i(_SYM) + _charsxp("system"),
        r"^raiz\$\[\[1\]\]: símbolo como valor recusado",
    ),
    (
        "rotulo-nao-simbolo",
        _i(_LIST | _TEM_ROTULO) + _texto("x") + _i(_NILVALUE),
        "^raiz: rótulo de pairlist que não é símbolo",
    ),
    (
        "pairlist-sem-fim",
        _i(_LIST) + _i(_NILVALUE) + _i(_INT, 0),
        "^raiz: pairlist que não termina em NULL",
    ),
    (
        "atributo-nao-pairlist",
        _i(_INT | _TEM_ATRIBUTO, 0) + _inteiro(1),
        "^raiz: atributos que não são pairlist",
    ),
    (
        "atributo-sem-nome",
        _i(_INT | _TEM_ATRIBUTO, 0) + _i(_LIST) + _inteiro(1) + _i(_NILVALUE),
        "^raiz: atributo sem nome",
    ),
    ("altrep-sem-descricao", _i(_ALTREP) + _i(_NILVALUE), "^raiz: ALTREP sem a descrição"),
    (
        "altrep-classe-texto",
        _i(_ALTREP)
        + _i(_LIST)
        + _texto("compact_intseq")
        + _i(_LIST)
        + _simbolo("base")
        + _i(_NILVALUE),
        "^raiz: ALTREP com classe que não é símbolo",
    ),
    (
        "embrulho-sem-par",
        _altrep("wrap_integer", "base", _i(_VEC, 0)),
        r"^raiz@estado: embrulho ALTREP sem o par \(vetor, metadados\)",
    ),
    (
        "embrulho-com-rotulo",
        _altrep("wrap_integer", "base", _i(_LIST | _TEM_ROTULO) + _simbolo("x") + _inteiro(1)),
        r"^raiz@estado: embrulho ALTREP sem o par",
    ),
    (
        "embrulho-como-pairlist",
        _altrep("wrap_integer", "base", _i(_LIST) + _inteiro(4) + _i(_NILVALUE)),
        "^raiz@estado: metadados do embrulho fora da forma",
    ),
    (
        "embrulho-de-nulo",
        _altrep("wrap_integer", "base", _i(_LIST) + _i(_NILVALUE) + _inteiro(0, 0)),
        "^raiz@estado: embrulho ALTREP de algo que não é vetor",
    ),
]


@pytest.mark.parametrize(
    ("corpo", "mensagem"), [(c, m) for _, c, m in _MALFORMADOS], ids=[n for n, _, _ in _MALFORMADOS]
)
def test_malformado_e_recusado_pelo_motivo_certo(corpo: bytes, mensagem: str) -> None:
    with pytest.raises(RdsRecusadoError, match=mensagem):
        le_rds(_cabecalho() + corpo)


def test_texto_latin1_e_decodificado_como_latin1() -> None:
    """O par de «texto-nao-utf8»: o mesmo byte `0xE9`, marcado latin-1, é lido como «é»."""
    assert le_rds(_cabecalho() + _i(_STR, 1) + _charsxp("", _LATIN1, b"caf\xe9")) == Vetor(
        "texto", ("café",)
    )


# ---------------------------------------------------------------------------
# O escritor
# ---------------------------------------------------------------------------

_OBJETOS_DE_IDA_E_VOLTA: list[Objeto] = [
    None,
    Lista(()),
    Vetor("real", (0.0, -1.5, 1e300, 5e-324, 0.0123)),
    Vetor("inteiro", (2**31 - 1, -(2**31) + 1, None)),
    Vetor("logico", (True, None, False)),
    Vetor("texto", ("", None, "Ácido cítrico", "β-Alanine", "日本")),
    Lista(
        (Vetor("real", (1.0,)), None, Lista((Vetor("texto", ("x",)),))),
        {"names": Vetor("texto", ("um", None, "três"))},
    ),
    Vetor(
        "inteiro",
        (1, 2, 1),
        {"levels": Vetor("texto", ("a", "b")), "class": Vetor("texto", ("factor",))},
    ),
]


def test_ida_e_volta_do_aceito_lido_do_r_e_igual() -> None:
    """O objeto que o R gravou, reescrito por `escreve_rds` e relido, é o mesmo objeto.

    O caminho inverso — o R ler o que `escreve_rds` grava — não roda aqui: exige o R, e está medido
    em `data/fixtures/rds/LEIA-ME.md`.
    """
    original = le_rds(_fixture("aceito", "ser"))
    escrito = escreve_rds(original)
    assert escrito[:6] == b"X\n" + _i(2), "o escritor grava o formato 2"
    assert le_rds(escrito) == original
    assert le_rds(gzip.compress(escrito)) == original
    assert escreve_rds(le_rds(escrito)) == escrito, "a segunda escrita devia ser byte a byte igual"


@pytest.mark.parametrize("objeto", _OBJETOS_DE_IDA_E_VOLTA)
def test_ida_e_volta_de_objetos_montados_e_igual(objeto: Objeto) -> None:
    lido = le_rds(escreve_rds(objeto))
    assert lido == objeto
    if isinstance(objeto, Vetor) and objeto.tipo == "logico":
        assert isinstance(lido, Vetor)
        assert [type(v) for v in lido.valores] == [type(v) for v in objeto.valores]


def test_o_escritor_marca_objeto_so_quando_ha_classe() -> None:
    """O bit de objeto (`1 << 8`) acompanha `class`: sem ele o R lê o fator e `identical` dá FALSE.

    A medição do R está no LEIA-ME das fixtures; aqui fica a forma dos bytes que ela exige.
    """
    com_classe = escreve_rds(_OBJETOS_DE_IDA_E_VOLTA[-1])
    sem_classe = escreve_rds(Vetor("inteiro", (1, 2, 1), {"levels": Vetor("texto", ("a",))}))
    assert com_classe[14:18] == _i(_INT | (1 << 8) | _TEM_ATRIBUTO)
    assert sem_classe[14:18] == _i(_INT | _TEM_ATRIBUTO)


@pytest.mark.parametrize(
    "objeto",
    [
        Simbolo("f"),
        Lista((Vetor("real", (1.0,)), Simbolo("f"))),
        Vetor("real", (1.0,), {"names": Simbolo("f")}),
    ],
    ids=["raiz", "item", "atributo"],
)
def test_o_escritor_recusa_simbolo_solto(objeto: Objeto) -> None:
    with pytest.raises(RdsRecusadoError, match=r"^símbolo solto não é dado"):
        escreve_rds(objeto)


@pytest.mark.parametrize(("valor", "nome_do_tipo"), [(1, "int"), (b"x", "bytes"), (1.5, "float")])
def test_o_escritor_recusa_texto_que_nao_e_str(valor: object, nome_do_tipo: str) -> None:
    vetor = Vetor("texto", ("ok", valor))  # type: ignore[arg-type]
    with pytest.raises(RdsRecusadoError, match=f"^vetor de texto com {nome_do_tipo}$"):
        escreve_rds(vetor)


# ---------------------------------------------------------------------------
# Lista.campo
# ---------------------------------------------------------------------------


def _nomeada(*nomes: str | None) -> Lista:
    return Lista(
        tuple(Vetor("inteiro", (i,)) for i in range(len(nomes))),
        {"names": Vetor("texto", nomes)},
    )


def test_campo_devolve_o_item_de_nome_unico() -> None:
    assert _nomeada("a", "b", None).campo("b") == Vetor("inteiro", (1,))


@pytest.mark.parametrize(
    ("lista", "nome", "ocorrencias"),
    [
        (_nomeada("a", "b"), "z", 0),
        (_nomeada("a", "a"), "a", 2),
        (_nomeada("a", None, "a", "a"), "a", 3),
        (Lista((None,)), "a", 0),
        # `names` com comprimento diferente do da lista não nomeia nada.
        (Lista((None, None), {"names": Vetor("texto", ("a",))}), "a", 0),
    ],
    ids=["ausente", "repetido", "triplicado", "sem-names", "names-de-outro-tamanho"],
)
def test_campo_ausente_ou_repetido_levanta(lista: Lista, nome: str, ocorrencias: int) -> None:
    with pytest.raises(RdsRecusadoError, match=f"^campo «{nome}»: {ocorrencias} ocorrências"):
        lista.campo(nome)


def test_as_fixtures_sao_as_que_o_leia_me_descreve() -> None:
    """Os dez arquivos e o gerador existem lado a lado com o LEIA-ME que os explica."""
    esperados = {
        f"{n}.{f}"
        for n in ("aceito", "com_funcao", "com_ambiente", "com_formula", "embrulho")
        for f in FORMATOS
    }
    presentes = {p.name for p in FIXTURES.iterdir() if p.suffix in (".ser", ".rds")}
    assert presentes == esperados
    assert (FIXTURES / "gera.R").is_file()
    leia_me = (FIXTURES / "LEIA-ME.md").read_text(encoding="utf-8")
    for nome in esperados | {"gera.R"}:
        assert nome in leia_me, nome
