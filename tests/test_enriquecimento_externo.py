"""O modo externo do enriquecimento — a carga, a G2 e o transporte, SEM REDE.

`refino.io.enriquecimento_externo` é o primeiro caminho do produto que manda algo para fora da
máquina, e o que este arquivo confere é a fronteira inteira dele: a carga sai por esquema fechado, a
G2 recusa o que não é nome confirmado nem vocabulário fixo e o que não é a forma canônica desse
esquema, e o transporte nunca levanta por rede.

**Nenhum teste aqui fala com host nenhum.** O transporte recebe uma conexão FALSA pelo argumento
`conexao` de `envia_ao_servidor`, que é o mesmo ponto de injeção da produção — o resto da receita
(corpo multipart, teto de bytes, tipo de conteúdo, leitura restrita) é o mesmo código.

**As respostas são SINTÉTICAS e as vias têm nomes INVENTADOS.** A resposta real do servidor traz
dados derivados do KEGG, cuja licença não permite redistribuir: nenhuma resposta real é versionada.
A sintética é escrita com `escreve_rds`, e os p dela são os da hipergeométrica arredondados a três
algarismos significativos, que a conferência de `ora_da_resposta` aceita dentro da tolerância dela.
É a mesma relação medida numa resposta real em 01/10/2026: cada p conferiu com a hipergeométrica,
40 de 40, com a lista igual ao número de nomes enviados e o universo igual a `dataSet$uniq.count`.
"""

from __future__ import annotations

import copy
import gzip
import http.client
import inspect
import io
import itertools
import json
import re
import socket
import ssl
import threading
import time
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from refino.io import enriquecimento_externo as ext
from refino.io import rds_restrito
from refino.io.rds_restrito import (
    Lista,
    Objeto,
    RdsRecusadoError,
    Vetor,
    escreve_rds,
    inventario,
    le_rds,
)
from refino.pathways.enriquecimento import (
    RespostaInvalidaError,
    ora_da_resposta,
    p_hipergeometrico,
)

RAIZ = Path(__file__).resolve().parent.parent
FIXTURES_RDS = RAIZ / "data" / "fixtures" / "rds"
INVENTARIO_TOSEND = RAIZ / "data" / "fixtures" / "vias-ora" / "inventario_tosend.json"
ENTRADA_DA_PARIDADE = RAIZ / "data" / "fixtures" / "vias-ora" / "entrada.json"
MAPEAMENTO_DA_PARIDADE = RAIZ / "data" / "fixtures" / "vias-ora" / "arvore.json"

# Nomes públicos de compostos; nenhum vem de resposta do servidor.
NOMES: tuple[str, ...] = (
    "Urea",
    "Glycine",
    "L-Alanine",
    "Citric acid",
    "Pyruvic acid",
    "Taurine",
)
COLUNAS_DA_ORA = ("total", "expected", "hits", "Raw p", "Holm p", "FDR")
# Universo e vias SINTÉTICOS: o tamanho não é o da biblioteca KEGG e os nomes de via não existem.
UNIVERSO_SINTETICO = 1_000
VIAS_INVENTADAS: tuple[tuple[str, int, tuple[str, ...]], ...] = (
    ("Via inventada Alfa", 12, ("Urea", "Glycine", "L-Alanine")),
    ("Via inventada Beta", 40, ("Citric acid", "Pyruvic acid")),
    ("Via inventada Gama", 7, ("Taurine",)),
)


# ---------------------------------------------------------------------------
# Construção de objetos R à mão
# ---------------------------------------------------------------------------


def _texto(*valores: str | None) -> Vetor:
    return Vetor("texto", tuple(valores))


def _lista(pares: Sequence[tuple[str, Objeto]]) -> Lista:
    return Lista(tuple(v for _, v in pares), {"names": _texto(*(n for n, _ in pares))})


_TIRA = object()


def _arvore_da_carga(nomes: Sequence[str]) -> dict[str, Any]:
    """A carga como dicionário aninhado, na ordem de `monta_carga`, para plantar campo a campo.

    É uma segunda escrita do esquema, independente da do módulo: o teste do molde confere que as
    duas dão os mesmos bytes, e só por isso cada planta difere da carga legítima pela planta só.
    """
    nomes_ = _texto(*nomes)
    falso = Vetor("logico", (False,))
    exclusao = Vetor("real", (2.0,))
    return {
        "mSet": {
            "dataSet": {"type": _texto("conc")},
            "analSet": {"type": _texto("msetora"), "msetlibname": _texto("kegg_pathway")},
            "api": {
                "filter": falso,
                "libname": _texto("kegg_pathway"),
                "excludeNum": exclusao,
                "oraVec": nomes_,
            },
        },
        "libNm": _texto("kegg_pathway"),
        "filter": falso,
        "oraVec": nomes_,
        "excludeNum": exclusao,
    }


def _objeto(no: Any) -> Objeto:
    if isinstance(no, dict):
        return _lista([(nome, _objeto(filho)) for nome, filho in no.items()])
    folha: Objeto = no
    return folha


def _carga_a_mao(nomes: Sequence[str], caminho: str | None = None, valor: object = None) -> bytes:
    """A carga escrita com `escreve_rds`, com a folha `caminho` trocada por `valor` (ou tirada)."""
    arvore = copy.deepcopy(_arvore_da_carga(nomes))
    if caminho is not None:
        *pais, folha = caminho.split("$")
        no = arvore
        for pai in pais:
            no = no[pai]
        if valor is _TIRA:
            del no[folha]
        else:
            no[folha] = valor
    return escreve_rds(_objeto(arvore))


# ---------------------------------------------------------------------------
# A carga e a G2
# ---------------------------------------------------------------------------


def test_a_carga_tem_exactamente_o_esquema_fechado_e_passa_na_g2() -> None:
    carga = ext.monta_carga(NOMES)
    folhas = inventario(le_rds(carga))
    assert len(ext.ESQUEMA_DA_CARGA) == 11, "âncora: o esquema declarado tem onze folhas"
    assert tuple((c, t) for c, t, _ in folhas) == ext.ESQUEMA_DA_CARGA
    tamanhos = {c: n for c, _, n in folhas}
    for caminho, _ in ext.ESQUEMA_DA_CARGA:
        esperado = len(NOMES) if caminho.endswith("$oraVec") else 1
        assert tamanhos[caminho] == esperado, caminho
    ext.confere_carga(carga, NOMES)


def test_os_textos_que_saem_sao_os_nomes_e_o_vocabulario_fixo() -> None:
    """O que sai, folha por folha: os nomes na ordem dada, e mais só o vocabulário fixo."""
    objeto = le_rds(ext.monta_carga(NOMES))
    for caminho in ("raiz$oraVec", "raiz$mSet$api$oraVec"):
        assert _folha(objeto, caminho).valores == NOMES
    textos = {
        v for c, t, _ in inventario(objeto) if t == "texto" for v in _folha(objeto, c).valores
    }
    assert textos == set(NOMES) | {"conc", "msetora", ext.BIBLIOTECA_EXTERNA}
    for caminho in ("raiz$filter", "raiz$mSet$api$filter"):
        assert _folha(objeto, caminho) == Vetor("logico", (False,))
    for caminho in ("raiz$excludeNum", "raiz$mSet$api$excludeNum"):
        assert _folha(objeto, caminho) == Vetor("real", (2.0,))


def _folha(objeto: Objeto, caminho: str) -> Vetor:
    no = objeto
    for nome in caminho.split("$")[1:]:
        assert isinstance(no, Lista)
        no = no.campo(nome)
    assert isinstance(no, Vetor)
    return no


def _valores_das_folhas(bruto: bytes) -> list[tuple[str, str, tuple[object, ...]]]:
    """Caminho, tipo e valores de cada folha: o que a G2 via antes da forma canônica."""
    objeto = le_rds(bruto)
    return [(c, t, _folha(objeto, c).valores) for c, t, _ in inventario(objeto)]


@pytest.mark.parametrize(
    ("nomes", "motivo"),
    [([], "ao menos um nome"), (["Urea", "Glycine", "Urea"], "repetido")],
    ids=["vazia", "nome-repetido"],
)
def test_monta_carga_recusa_lista_vazia_e_nome_repetido(nomes: list[str], motivo: str) -> None:
    with pytest.raises(ext.CargaRecusadaError, match=motivo):
        ext.monta_carga(nomes)


def test_o_molde_a_mao_reproduz_monta_carga_byte_a_byte() -> None:
    """Âncora das plantas abaixo: sem ela, uma planta que mudasse outra coisa além do campo
    plantado reprovaria por esse outro motivo, e o teste dela passaria por coincidência."""
    assert _carga_a_mao(NOMES) == ext.monta_carga(NOMES)
    ext.confere_carga(_carga_a_mao(NOMES), NOMES)


def test_g2_recusa_nome_fora_dos_permitidos() -> None:
    carga = ext.monta_carga(NOMES)
    ext.confere_carga(carga, NOMES)  # a mesma carga, com o nome permitido, passa
    with pytest.raises(ext.CargaRecusadaError, match=r"não é nome confirmado.*Taurine"):
        ext.confere_carga(carga, NOMES[:-1])


_PLANTAS_DA_G2: list[tuple[str, str, object, str]] = [
    ("texto-fora-do-vocabulario", "libNm", _texto("segredo"), "vocabulário fixo"),
    ("nome-permitido-fora-do-vetor-de-nomes", "mSet$dataSet$type", _texto("Urea"), "vocabulário"),
    ("filtro-ligado-na-raiz", "filter", Vetor("logico", (True,)), "filtro"),
    ("filtro-ligado-no-api", "mSet$api$filter", Vetor("logico", (True,)), "filtro"),
    ("filtro-NA", "filter", Vetor("logico", (None,)), "filtro"),
    ("excludeNum-diferente-na-raiz", "excludeNum", Vetor("real", (3.0,)), "fora do fixo"),
    ("excludeNum-diferente-no-api", "mSet$api$excludeNum", Vetor("real", (1.0,)), "fora do fixo"),
    ("vetor-de-nomes-vazio", "oraVec", _texto(), "nome confirmado"),
    ("vetor-de-nomes-com-NA", "oraVec", _texto("Urea", None), "nome confirmado"),
    ("folha-com-dois-valores", "libNm", _texto("kegg_pathway", "kegg_pathway"), "esperava 1"),
    ("campo-a-mais", "mSet$api$fundo", _texto("conc"), "esquema fechado"),
    ("campo-a-menos", "excludeNum", _TIRA, "esquema fechado"),
    ("tipo-trocado", "excludeNum", Vetor("inteiro", (2,)), "esquema fechado"),
    ("folha-NULL", "libNm", None, "esquema fechado"),
    ("folha-lista", "libNm", Lista((), {}), "esquema fechado"),
    # As três de baixo passam em cada folha e só a forma canônica as recusa: cada valor é
    # permitido ou fixo, e o que está errado é ONDE ele está ou quantas vezes. Medido em 02/10/2026:
    # a G2 que conferia só as folhas aceitava as três.
    ("vocabulario-na-folha-errada", "mSet$dataSet$type", _texto("kegg_pathway"), "forma canônica"),
    ("vetores-de-nomes-diferentes", "mSet$api$oraVec", _texto(*reversed(NOMES)), "forma canônica"),
    ("nome-repetido-no-vetor", "oraVec", _texto(*NOMES, NOMES[0]), "repetido"),
]


@pytest.mark.parametrize(
    ("caminho", "valor", "motivo"),
    [p[1:] for p in _PLANTAS_DA_G2],
    ids=[p[0] for p in _PLANTAS_DA_G2],
)
def test_g2_recusa_a_carga_plantada(caminho: str, valor: object, motivo: str) -> None:
    ext.confere_carga(_carga_a_mao(NOMES), NOMES)  # sem a planta, a mesma escrita passa
    with pytest.raises(ext.CargaRecusadaError, match=motivo):
        ext.confere_carga(_carga_a_mao(NOMES, caminho, valor), NOMES)


@pytest.mark.parametrize(
    "bruto",
    [
        b"",
        b"isto nao e rds",
        ext.monta_carga(NOMES)[:-1],
        ext.monta_carga(NOMES) + b"segredo",
    ],
    ids=["vazio", "texto", "truncada", "bytes-depois-do-objeto"],
)
def test_g2_recusa_bytes_que_nao_sao_um_rds_so(bruto: bytes) -> None:
    with pytest.raises(ext.CargaRecusadaError, match="não é dado puro"):
        ext.confere_carga(bruto, NOMES)


@pytest.mark.parametrize("arquivo", ["com_funcao.ser", "com_funcao.rds"])
def test_g2_recusa_carga_com_funcao(arquivo: str) -> None:
    """A planta é a fixture que o R da árvore gravou (`data/fixtures/rds/gera.R`): uma função de
    verdade, e não bytes montados aqui à semelhança de uma."""
    with pytest.raises(ext.CargaRecusadaError, match="função"):
        ext.confere_carga((FIXTURES_RDS / arquivo).read_bytes(), ["ok", "f"])


def _carga_com_atributo_na_folha() -> bytes:
    objeto = le_rds(ext.monta_carga(NOMES))
    assert isinstance(objeto, Lista)
    i = objeto.nomes().index("libNm")
    folha = objeto.itens[i]
    assert isinstance(folha, Vetor)
    itens = list(objeto.itens)
    itens[i] = Vetor(folha.tipo, folha.valores, {"segredo": _texto("dado de paciente")})
    return escreve_rds(Lista(tuple(itens), objeto.atributos))


def _carga_com_atributo_na_raiz() -> bytes:
    objeto = le_rds(ext.monta_carga(NOMES))
    assert isinstance(objeto, Lista)
    extra = {**objeto.atributos, "segredo": _texto("dado de paciente")}
    return escreve_rds(Lista(objeto.itens, extra))


def _carga_no_formato_3_com_codificacao_inventada() -> bytes:
    """O cabeçalho do formato 3 leva um nome de codificação, que o leitor pula sem ler."""
    carga = ext.monta_carga(NOMES)
    cabecalho_2 = carga[:14]  # «X\n» e três inteiros
    assert cabecalho_2[2:6] == (2).to_bytes(4, "big"), "âncora: monta_carga escreve o formato 2"
    segredo = b"dado de paciente"
    cabecalho_3 = b"X\n" + (3).to_bytes(4, "big") + cabecalho_2[6:14]
    return cabecalho_3 + len(segredo).to_bytes(4, "big") + segredo + carga[14:]


def _carga_com_versao_do_r_inventada() -> bytes:
    """Os oito bytes das duas versões do R no cabeçalho, que o leitor lê e descarta: não é texto
    de folha nenhuma, e é um canal de oito bytes arbitrários em cada envio."""
    carga = ext.monta_carga(NOMES)
    return carga[:6] + b"segredo!" + carga[14:]


def _carga_em_gzip_com_nome_de_arquivo() -> bytes:
    """O envelope `gzip`, que `le_rds` abre antes de ler: o campo de nome do arquivo é texto que
    sai nos bytes e que o objeto lido não contém."""
    saida = io.BytesIO()
    with gzip.GzipFile(filename="dado de paciente", mode="wb", fileobj=saida, mtime=0) as g:
        g.write(ext.monta_carga(NOMES))
    return saida.getvalue()


_PLANTAS_FORA_DAS_FOLHAS: list[tuple[str, Callable[[], bytes], bytes, str]] = [
    ("atributo-na-folha", _carga_com_atributo_na_folha, b"dado de paciente", "folha com atributo"),
    ("atributo-na-raiz", _carga_com_atributo_na_raiz, b"dado de paciente", r"raiz: atributos"),
    (
        "codificacao-do-formato-3",
        _carga_no_formato_3_com_codificacao_inventada,
        b"dado de paciente",
        "cabeçalho do escritor",
    ),
    ("versao-do-R", _carga_com_versao_do_r_inventada, b"segredo!", "cabeçalho do escritor"),
    (
        "nome-no-envelope-gzip",
        _carga_em_gzip_com_nome_de_arquivo,
        b"dado de paciente",
        "cabeçalho do escritor",
    ),
]


def _ancora_da_planta(montagem: Callable[[], bytes], planta: bytes) -> bytes:
    ext.confere_carga(ext.monta_carga(NOMES), NOMES)  # sem a planta, a mesma base passa
    bruto = montagem()
    assert planta in bruto, "âncora: a planta está nos bytes que sairiam"
    assert _valores_das_folhas(bruto) == _valores_das_folhas(ext.monta_carga(NOMES)), (
        "âncora: as folhas são as da base, e a planta está fora delas"
    )
    return bruto


@pytest.mark.parametrize(
    ("montagem", "planta", "guarda"),
    [p[1:] for p in _PLANTAS_FORA_DAS_FOLHAS],
    ids=[p[0] for p in _PLANTAS_FORA_DAS_FOLHAS],
)
def test_g2_recusa_texto_que_sai_fora_das_folhas(
    montagem: Callable[[], bytes], planta: bytes, guarda: str
) -> None:
    """A G2 promete que todo texto que sai é nome confirmado ou vocabulário fixo. Atributo, nome de
    codificação, versão do R e envelope `gzip` saem nos mesmos bytes, e as folhas não os contêm.

    Medido em 02/10/2026: a G2 que conferia só as folhas do inventário aceitava as cinco. Desde a
    revisão do mesmo dia, cada uma é recusada por uma conferência que NÃO depende de `monta_carga`
    — o cabeçalho literal e os atributos —, e o `match` é o da conferência que a apanha.
    """
    bruto = _ancora_da_planta(montagem, planta)
    with pytest.raises(ext.CargaRecusadaError, match=guarda):
        ext.confere_carga(bruto, NOMES)


@pytest.mark.parametrize(
    ("montagem", "planta"),
    [p[1:3] for p in _PLANTAS_FORA_DAS_FOLHAS],
    ids=[p[0] for p in _PLANTAS_FORA_DAS_FOLHAS],
)
def test_a_forma_canonica_sozinha_tambem_recusa_as_cinco(
    montagem: Callable[[], bytes], planta: bytes, monkeypatch: pytest.MonkeyPatch
) -> None:
    """As duas camadas são independentes: sem o cabeçalho literal e sem a conferência de atributos,
    a forma canônica ainda recusa cada planta. Um defeito numa camada não abre a carga."""
    bruto = _ancora_da_planta(montagem, planta)
    monkeypatch.setattr(ext, "_CABECALHO_DA_CARGA", b"")
    monkeypatch.setattr(ext, "_confere_atributos", lambda *_: None)
    with pytest.raises(ext.CargaRecusadaError, match="forma canônica"):
        ext.confere_carga(bruto, NOMES)


def test_g2_apanha_o_escritor_que_muda_o_cabecalho(monkeypatch: pytest.MonkeyPatch) -> None:
    """A comparação canônica é cega a um defeito DENTRO de `monta_carga`: os dois lados mudam
    juntos. O cabeçalho literal não muda junto (revisão de 02/10/2026)."""
    ext.confere_carga(ext.monta_carga(NOMES), NOMES)
    monkeypatch.setattr(rds_restrito, "_VERSAO_DO_R", int.from_bytes(b"S01\x00", "big"))
    carga = ext.monta_carga(NOMES)
    assert b"S01" in carga, "âncora: o escritor plantado grava o canal no cabeçalho"
    with pytest.raises(ext.CargaRecusadaError, match="cabeçalho do escritor"):
        ext.confere_carga(carga, NOMES)


def test_g2_apanha_o_escritor_que_acrescenta_atributo(monkeypatch: pytest.MonkeyPatch) -> None:
    """O mesmo para um atributo que `monta_carga` passasse a gravar na raiz: a forma canônica o
    aceitaria, porque é ela própria que o escreve."""
    original = ext._nomeada

    def com_segredo(**campos: Objeto) -> Lista:
        lista = original(**campos)
        return Lista(lista.itens, {**lista.atributos, "comment": _texto("PACIENTE_S01")})

    monkeypatch.setattr(ext, "_nomeada", com_segredo)
    carga = ext.monta_carga(NOMES)
    assert b"PACIENTE_S01" in carga, "âncora: o escritor plantado grava o atributo"
    with pytest.raises(ext.CargaRecusadaError, match="atributos"):
        ext.confere_carga(carga, NOMES)


# ---------------------------------------------------------------------------
# O inventário registrado do que sairia
# ---------------------------------------------------------------------------

_IDENTIFICADOR_KEGG = re.compile(r"C\d{5}")


def _nomes_da_paridade() -> tuple[list[str], set[str]]:
    """Os nomes que a captura do inventário podia ter deixado no arquivo, e de onde vêm.

    A captura de 01/10/2026 correu sobre a lista pública da paridade
    (`data/fixtures/vias-ora/entrada.json`). O objeto do pacote leva a lista consultada e o nome
    casado de cada composto, e o fundo vai como identificador KEGG. O primeiro elemento devolvido
    são os casados da lista, na ordem do mapeamento: os nomes canônicos que `monta_carga` recebe.
    O segundo é a população em que a ausência é conferida — a lista e o fundo da entrada, mais os
    nomes casados dos dois em `data/fixtures/vias-ora/arvore.json` —, e só ela: um nome de composto
    fora destas fontes não é procurado.
    """
    entrada = json.loads(ENTRADA_DA_PARIDADE.read_text(encoding="utf-8"))
    mapear = json.loads(MAPEAMENTO_DA_PARIDADE.read_text(encoding="utf-8"))["mapear"]
    casados_da_lista = [m["casado"] for m in mapear["lista"]["mapeamento"] if m.get("casado")]
    casados = {
        m["casado"] for parte in mapear.values() for m in parte["mapeamento"] if m.get("casado")
    }
    return casados_da_lista, set(entrada["lista"]) | set(entrada["fundo"]) | casados


def _valores_proibidos(dados: object, nomes: Iterable[str]) -> tuple[list[str], list[str]]:
    """Os identificadores KEGG e os nomes de `nomes` em alguma chave ou texto de `dados`.

    Sobre o JSON já lido, e não sobre o texto do arquivo: um nome escrito com escape `\\uXXXX` não
    casaria com o texto cru. Nomes sem distinguir maiúsculas, e como trecho, não como valor inteiro.
    """
    cadeias: list[str] = []

    def anda(no: object) -> None:
        if isinstance(no, str):
            cadeias.append(no)
        elif isinstance(no, dict):
            for chave, filho in no.items():
                cadeias.append(str(chave))
                anda(filho)
        elif isinstance(no, list):
            for filho in no:
                anda(filho)

    anda(dados)
    texto = "\n".join(cadeias)
    dobrado = texto.casefold()
    kegg = sorted(set(_IDENTIFICADOR_KEGG.findall(texto)))
    return kegg, sorted(n for n in nomes if n.casefold() in dobrado)


def _confere_inventario_sem_valores(caminho: Path, nomes: Iterable[str]) -> dict[str, Any]:
    """O registro em `caminho`, depois de conferir que nele não há identificador KEGG nem nome."""
    registro: dict[str, Any] = json.loads(caminho.read_text(encoding="utf-8"))
    kegg, vazados = _valores_proibidos(registro, nomes)
    assert not kegg, f"identificador KEGG no inventário: {kegg}"
    assert not vazados, f"nome de composto no inventário: {vazados}"
    return registro


def test_o_inventario_registrado_e_o_esquema_fechado_e_nao_traz_nome_nem_identificador(
    tmp_path: Path,
) -> None:
    """`data/fixtures/vias-ora/inventario_tosend.json` é versionado, e o que o torna versionável é
    não trazer os valores que a carga do pacote levaria: nem nome, nem identificador KEGG.

    O esquema fechado registrado nele tem de ser o que o código monta hoje, e o tamanho registrado
    (1.542 bytes) é o de `monta_carga` sobre os 25 nomes CASADOS — sobre os 25 consultados dá outro
    número, porque em 11 dos 25 o casado difere do consultado — `L-Lactic acid` casa como
    `Lactic acid`, por exemplo — e a carga leva o canônico.
    """
    casados_da_lista, nomes = _nomes_da_paridade()
    # Âncoras de não-vacuidade, nas fontes: há nomes a procurar, e o instrumento, sobre o mapeamento
    # que os contém, acha identificador KEGG e acha cada nome casado — o oráculo à mão.
    assert len(casados_da_lista) == 25 and set(casados_da_lista) <= nomes
    kegg, achados = _valores_proibidos(
        json.loads(MAPEAMENTO_DA_PARIDADE.read_text(encoding="utf-8")), nomes
    )
    assert kegg and set(casados_da_lista) <= set(achados)

    registro = _confere_inventario_sem_valores(INVENTARIO_TOSEND, nomes)

    # Âncoras de não-vacuidade, no registro: ele tem folhas, e o esquema tem as onze.
    for captura in registro["objeto_do_pacote"].values():
        assert captura["folhas"] == len(captura["inventario"]) > 0
    esquema = registro["esquema_fechado"]
    assert esquema["folhas"] == len(esquema["inventario"]) == len(ext.ESQUEMA_DA_CARGA) == 11
    assert tuple((f["caminho"], f["tipo"]) for f in esquema["inventario"]) == ext.ESQUEMA_DA_CARGA

    carga = ext.monta_carga(casados_da_lista)
    assert esquema["bytes_com_25_nomes"] == len(carga)
    objeto = le_rds(carga)
    for folha in esquema["inventario"]:
        vetor = _folha(objeto, folha["caminho"])
        assert folha["tamanho"] == len(vetor.valores), folha["caminho"]
        if folha["caminho"] in ("raiz$oraVec", "raiz$mSet$api$oraVec"):
            assert "valores" not in folha, f"{folha['caminho']}: os nomes não são registrados"
        else:
            assert tuple(folha["valores"]) == vetor.valores, folha["caminho"]

    # Par causal, na mesma função: a mesma conferência sobre uma cópia com a planta reprova, e sobre
    # a cópia sem ela passa — a cópia difere do registro só pela planta.
    def copia(planta: str | None, onde: str) -> Path:
        dados = copy.deepcopy(registro)
        if planta is not None:
            secao = (
                dados["esquema_fechado"]
                if onde.startswith("raiz")
                else dados["objeto_do_pacote"]["com_fundo"]
            )
            (folha,) = (f for f in secao["inventario"] if f["caminho"] == onde)
            folha["valores"] = [planta]
        destino = tmp_path / f"{onde.replace('$', '_')}_{planta}.json"
        destino.write_text(json.dumps(dados, ensure_ascii=False, indent=2), encoding="utf-8")
        return destino

    _confere_inventario_sem_valores(copia(None, "controle"), nomes)
    for planta, onde in (
        ("C00037", "toSend$filterData"),
        (casados_da_lista[0], "raiz$oraVec"),
    ):
        plantado = copia(planta, onde)
        assert planta in plantado.read_text(encoding="utf-8"), "âncora: a planta aterrou"
        with pytest.raises(AssertionError, match=re.escape(planta)):
            _confere_inventario_sem_valores(plantado, nomes)


# ---------------------------------------------------------------------------
# A resposta sintética
# ---------------------------------------------------------------------------


def _tres_significativos(x: float) -> float:
    return float(f"{x:.3g}")


def _resposta_sintetica(
    *,
    universo_declarado: int = UNIVERSO_SINTETICO,
    universo_dos_p: int = UNIVERSO_SINTETICO,
    sem: str | None = None,
) -> bytes:
    """Uma resposta na forma da do servidor, com vias inventadas e p da hipergeométrica.

    `universo_dos_p` diferente de `universo_declarado` planta o erro que a conferência da ORA
    existe para apanhar; `sem` tira um campo de `analSet` ou de `dataSet`.
    """
    na_lista = len(NOMES)
    k = len(VIAS_INVENTADAS)
    p = [
        _tres_significativos(p_hipergeometrico(len(c), total, na_lista, universo_dos_p))
        for _, total, c in VIAS_INVENTADAS
    ]
    ordem = sorted(range(k), key=lambda i: p[i])
    posto = {i: r + 1 for r, i in enumerate(ordem)}
    colunas = [
        [float(total) for _, total, _ in VIAS_INVENTADAS],
        [
            _tres_significativos(na_lista * total / universo_dos_p)
            for _, total, _ in VIAS_INVENTADAS
        ],
        [float(len(c)) for _, _, c in VIAS_INVENTADAS],
        p,
        [_tres_significativos(min(1.0, p[i] * (k - posto[i] + 1))) for i in range(k)],
        [_tres_significativos(min(1.0, p[i] * k / posto[i])) for i in range(k)],
    ]
    matriz = Vetor(
        "real",
        tuple(v for coluna in colunas for v in coluna),  # o R guarda por coluna
        {
            "dim": Vetor("inteiro", (k, len(COLUNAS_DA_ORA))),
            "dimnames": Lista(
                (_texto(*(v for v, _, _ in VIAS_INVENTADAS)), _texto(*COLUNAS_DA_ORA)), {}
            ),
        },
    )
    acertos = _lista([(via, _texto(*c)) for via, _, c in VIAS_INVENTADAS])
    anal = [("type", _texto("msetora")), ("ora.mat", matriz), ("ora.hits", acertos)]
    conjunto = [("type", _texto("conc")), ("uniq.count", Vetor("inteiro", (universo_declarado,)))]
    return escreve_rds(
        _lista(
            [
                ("dataSet", _lista([par for par in conjunto if par[0] != sem])),
                ("analSet", _lista([par for par in anal if par[0] != sem])),
            ]
        )
    )


# ---------------------------------------------------------------------------
# O transporte, com conexão falsa
# ---------------------------------------------------------------------------


@dataclass
class _RespostaFalsa:
    status: int = 200
    reason: str = "OK"
    cabecalhos: dict[str, str] = field(
        default_factory=lambda: {"Content-Type": "application/octet-stream"}
    )
    corpo: bytes = b""
    erro_na_leitura: BaseException | None = None
    leituras: list[int] = field(default_factory=list)

    def getheader(self, nome: str, /) -> str | None:
        return self.cabecalhos.get(nome)

    def read(self, n: int = -1, /) -> bytes:
        self.leituras.append(n)
        if self.erro_na_leitura is not None:
            raise self.erro_na_leitura
        return self.corpo if n < 0 else self.corpo[:n]


@dataclass
class _ConexaoFalsa:
    resposta: _RespostaFalsa = field(default_factory=_RespostaFalsa)
    erro_no_pedido: BaseException | None = None
    erro_na_resposta: BaseException | None = None
    pedidos: list[tuple[str, str, bytes, dict[str, str]]] = field(default_factory=list)
    fechamentos: int = 0

    def request(self, metodo: str, url: str, body: bytes, headers: dict[str, str], /) -> None:
        self.pedidos.append((metodo, url, body, dict(headers)))
        if self.erro_no_pedido is not None:
            raise self.erro_no_pedido

    def getresponse(self) -> _RespostaFalsa:
        if self.erro_na_resposta is not None:
            raise self.erro_na_resposta
        return self.resposta

    def close(self) -> None:
        self.fechamentos += 1


@dataclass
class _Fabrica:
    """O lugar de `http.client.HTTPSConnection`: registra para onde se tentou abrir."""

    conexao: _ConexaoFalsa = field(default_factory=_ConexaoFalsa)
    erro: BaseException | None = None
    chamadas: list[tuple[str, float]] = field(default_factory=list)

    def __call__(self, host: str, tempo_limite: float) -> _ConexaoFalsa:
        self.chamadas.append((host, tempo_limite))
        if self.erro is not None:
            raise self.erro
        return self.conexao


def _relogio() -> Callable[[], float]:
    instantes = itertools.count(100.0, 2.5)
    return lambda: next(instantes)


def _envia(
    fabrica: _Fabrica,
    *,
    carga: bytes | None = None,
    cancelado: Callable[[], bool] = lambda: False,
) -> ext.RespostaDoServidor:
    return ext.envia_ao_servidor(
        ext.monta_carga(NOMES) if carga is None else carga,
        NOMES,
        conexao=fabrica,
        cancelado=cancelado,
        relogio=_relogio(),
    )


def _fabrica_ok(corpo: bytes | None = None, **resposta: Any) -> _Fabrica:
    corpo_ = _resposta_sintetica() if corpo is None else corpo
    return _Fabrica(_ConexaoFalsa(_RespostaFalsa(corpo=corpo_, **resposta)))


def test_200_com_resposta_valida_e_ok_e_a_conferencia_da_ora_aceita() -> None:
    fabrica = _fabrica_ok()
    r = _envia(fabrica)
    assert (r.desfecho, r.motivo, r.segundos) == ("ok", "", 2.5)
    assert r.dados is not None
    assert r.dados["universo"] == {"tipo": "biblioteca", "tamanho": UNIVERSO_SINTETICO}
    assert r.dados["na_lista"] == len(NOMES)
    assert r.dados["biblioteca"] == ext.BIBLIOTECA_EXTERNA
    resultado = ora_da_resposta(r.dados)
    assert resultado.tamanho_do_universo == UNIVERSO_SINTETICO
    assert resultado.tipo_do_universo == "biblioteca"
    assert resultado.na_lista == len(NOMES)
    por_via = {v.via: v for v in resultado.vias}
    assert set(por_via) == {via for via, _, _ in VIAS_INVENTADAS}
    for via, total, compostos in VIAS_INVENTADAS:
        assert por_via[via].total == total
        assert por_via[via].compostos == tuple(sorted(compostos))
    assert fabrica.conexao.resposta.leituras == [ext.MAX_BYTES + 1], "a leitura tem teto"


def test_p_que_nao_e_do_universo_declarado_passa_no_transporte_e_cai_na_conferencia() -> None:
    """Par causal da resposta sintética: o transporte não confere p — quem confere é
    `ora_da_resposta`, e é ela que torna o «ok» acima uma afirmação sobre os p."""
    r = _envia(_fabrica_ok(_resposta_sintetica(universo_dos_p=2 * UNIVERSO_SINTETICO)))
    assert r.desfecho == "ok" and r.dados is not None
    with pytest.raises(RespostaInvalidaError, match="hipergeométrico"):
        ora_da_resposta(r.dados)


def test_o_pedido_leva_exactamente_os_bytes_da_carga_ao_destino_constante() -> None:
    carga = ext.monta_carga((*NOMES, "β-Alanine"))
    fabrica = _fabrica_ok()
    ext.envia_ao_servidor(carga, NOMES, conexao=fabrica, relogio=_relogio())
    assert ext.DESTINO == "https://www.xialab.ca/api/msetora"
    assert fabrica.chamadas == [("www.xialab.ca", ext.TEMPO_LIMITE_S)]
    assert len(fabrica.conexao.pedidos) == 1
    metodo, url, corpo, cabecalhos = fabrica.conexao.pedidos[0]
    assert (metodo, url) == ("POST", "/api/msetora")
    tipo, _, fronteira = cabecalhos["Content-Type"].partition("; boundary=")
    assert tipo == "multipart/form-data" and fronteira
    assert cabecalhos["Content-Length"] == str(len(corpo))
    abertura = f"--{fronteira}\r\n".encode("ascii")
    fecho = f"\r\n--{fronteira}--\r\n".encode("ascii")
    assert corpo.startswith(abertura) and corpo.endswith(fecho)
    assert corpo.count(f"--{fronteira}".encode("ascii")) == 2, "a fronteira só delimita"
    cabecalho_da_parte, _, conteudo = corpo[len(abertura) : -len(fecho)].partition(b"\r\n\r\n")
    assert conteudo == carga
    assert cabecalho_da_parte.split(b"\r\n") == [
        b'Content-Disposition: form-data; name="rds"; filename="tosend.rds"',
        b"Content-Type: application/octet-stream",
    ]


@pytest.mark.parametrize("status", [302, 500])
def test_status_diferente_de_200_e_servidor_recusou_sem_seguir(status: int) -> None:
    fabrica = _fabrica_ok(
        status=status, reason="Found", cabecalhos={"Location": "https://outro.exemplo/"}
    )
    r = _envia(fabrica)
    assert (r.desfecho, r.dados) == ("servidor_recusou", None)
    assert str(status) in r.motivo
    assert len(fabrica.chamadas) == 1 and len(fabrica.conexao.pedidos) == 1, "nada foi seguido"


@pytest.mark.parametrize(
    "tipo", ["text/html; charset=utf-8", "application/json", None], ids=["html", "json", "ausente"]
)
def test_tipo_de_conteudo_errado_e_resposta_malformada(tipo: str | None) -> None:
    cabecalhos = {} if tipo is None else {"Content-Type": tipo}
    r = _envia(_fabrica_ok(cabecalhos=cabecalhos))
    assert (r.desfecho, r.dados) == ("resposta_malformada", None)
    assert "tipo de conteúdo" in r.motivo


def test_parametro_depois_do_tipo_de_conteudo_nao_muda_o_tipo() -> None:
    r = _envia(_fabrica_ok(cabecalhos={"Content-Type": "application/octet-stream; x=1"}))
    assert r.desfecho == "ok"


def test_resposta_acima_do_teto_e_resposta_malformada(monkeypatch: pytest.MonkeyPatch) -> None:
    """O teto é o do módulo, lido na hora do envio; com ele baixado, a MESMA resposta que passa
    sem a planta é recusada — par causal na mesma função."""
    corpo = _resposta_sintetica()
    assert _envia(_fabrica_ok(corpo)).desfecho == "ok"
    teto = 64
    assert len(corpo) > teto, "âncora: a resposta passa do teto plantado"
    monkeypatch.setattr(ext, "MAX_BYTES", teto)
    fabrica = _fabrica_ok(corpo)
    r = _envia(fabrica)
    assert (r.desfecho, r.dados) == ("resposta_malformada", None)
    assert f"acima de {teto} bytes" in r.motivo
    assert fabrica.conexao.resposta.leituras == [teto + 1], "lê no máximo um byte além do teto"
    assert fabrica.conexao.fechamentos == 1


@pytest.mark.parametrize(
    "erro",
    [ssl.SSLCertVerificationError("certificate verify failed"), ssl.SSLError("handshake")],
    ids=["certificado", "tls"],
)
def test_tls_recusado_e_transporte_recusado(erro: ssl.SSLError) -> None:
    """`SSLCertVerificationError` é também `OSError`: sem a ordem certa dos `except`, um
    certificado recusado viraria «sem rede», que é outra coisa a dizer ao usuário."""
    assert isinstance(erro, OSError), "âncora: o erro de TLS cairia no ramo de rede"
    fabrica = _Fabrica(_ConexaoFalsa(erro_no_pedido=erro))
    r = _envia(fabrica)
    assert (r.desfecho, r.dados) == ("transporte_recusado", None)
    assert fabrica.conexao.fechamentos == 1


@pytest.mark.parametrize(
    "conexao",
    [
        _ConexaoFalsa(erro_no_pedido=OSError("rede inalcançável")),
        _ConexaoFalsa(erro_na_resposta=TimeoutError("timed out")),
        _ConexaoFalsa(_RespostaFalsa(erro_na_leitura=http.client.IncompleteRead(b"X\n"))),
    ],
    ids=["OSError-no-pedido", "TimeoutError-na-resposta", "IncompleteRead-na-leitura"],
)
def test_falha_de_rede_e_sem_rede(conexao: _ConexaoFalsa) -> None:
    fabrica = _Fabrica(conexao)
    r = _envia(fabrica)
    assert (r.desfecho, r.dados) == ("sem_rede", None)
    assert fabrica.conexao.fechamentos == 1


def test_falha_ao_abrir_a_conexao_e_sem_rede() -> None:
    fabrica = _Fabrica(erro=OSError("sem rota"))
    r = _envia(fabrica)
    assert (r.desfecho, r.dados) == ("sem_rede", None)
    assert fabrica.chamadas == [("www.xialab.ca", ext.TEMPO_LIMITE_S)]


@pytest.mark.parametrize(
    ("fabrica", "desfecho"),
    [
        (lambda: _Fabrica(erro=OSError("sem rota")), "sem_rede"),
        (lambda: _Fabrica(_ConexaoFalsa(erro_na_resposta=TimeoutError("timed out"))), "sem_rede"),
        (
            lambda: _Fabrica(_ConexaoFalsa(erro_no_pedido=ssl.SSLCertVerificationError("x"))),
            "transporte_recusado",
        ),
    ],
    ids=["abrir-a-conexao", "tempo-esgotado", "certificado"],
)
def test_a_espera_e_medida_tambem_quando_a_rede_falha(
    fabrica: Callable[[], _Fabrica], desfecho: str
) -> None:
    """Quem esperou até o tempo esgotar esperou de verdade, e `segundos` o registra.

    O relógio falso avança 2,5 s por leitura: o início e a leitura na falha dão 2,5. Até
    02/10/2026 estes dois ramos não liam o relógio e devolviam 0,0.
    """
    r = _envia(fabrica())
    assert r.desfecho == desfecho, "âncora: o cenário chegou ao ramo que declara"
    assert r.segundos == 2.5


def test_o_teto_de_tempo_do_envio_e_60_s() -> None:
    """Os quatro envios de teste de 01/10/2026 levaram de 5,4 a 6,1 s, e 60 s é cerca de dez
    vezes o maior deles. O valor é o que a fábrica de conexão recebe, e não só o da constante."""
    fabrica = _fabrica_ok()
    assert _envia(fabrica).desfecho == "ok"
    assert fabrica.chamadas == [("www.xialab.ca", 60.0)]


def test_cancelado_antes_nao_chama_a_conexao() -> None:
    controle = _fabrica_ok()
    assert _envia(controle).desfecho == "ok"
    assert len(controle.chamadas) == 1, "âncora: o registro da fábrica registra"
    fabrica = _fabrica_ok()
    r = _envia(fabrica, cancelado=lambda: True)
    assert (r.desfecho, r.dados) == ("cancelado", None)
    assert fabrica.chamadas == [] and fabrica.conexao.pedidos == []


def _cancela_depois_do_pedido(fabrica: _Fabrica) -> Callable[[], bool]:
    """O cancelamento pelo ESTADO da conexão — falso até o pedido sair, verdadeiro depois —, e não
    por uma sequência de respostas: a espera consulta o cancelamento a cada 0,1 s, e quantas vezes
    ela o faz depende da máquina (revisão de 02/10/2026)."""
    return lambda: bool(fabrica.conexao.pedidos)


def test_cancelado_durante_e_cancelado_mesmo_com_resposta_valida() -> None:
    fabrica = _fabrica_ok()
    r = _envia(fabrica, cancelado=_cancela_depois_do_pedido(fabrica))
    assert (r.desfecho, r.dados) == ("cancelado", None)
    assert "o pedido chegou ao servidor" in r.motivo
    assert len(fabrica.conexao.pedidos) == 1 and fabrica.conexao.fechamentos == 1


def test_cancelado_antes_diz_que_nada_saiu() -> None:
    r = _envia(_fabrica_ok(), cancelado=lambda: True)
    assert r.motivo == "cancelado antes do envio: nada saiu da máquina"


@pytest.mark.parametrize(
    ("corpo", "motivo"),
    [
        ((FIXTURES_RDS / "com_funcao.ser").read_bytes(), "função"),
        (_resposta_sintetica(sem="ora.mat"), "ora.mat"),
        (_resposta_sintetica(sem="uniq.count"), "uniq.count"),
        (b"<html>servidor em manutencao</html>", "cabeçalho"),
    ],
    ids=["com-funcao", "sem-ora.mat", "sem-uniq.count", "html-como-octet-stream"],
)
def test_corpo_que_o_leitor_restrito_recusa_e_resposta_malformada(
    corpo: bytes, motivo: str
) -> None:
    r = _envia(_fabrica_ok(corpo))
    assert (r.desfecho, r.dados) == ("resposta_malformada", None)
    assert motivo in r.motivo


_NAO, _SIM = False, True  # se o cancelamento chega depois de o pedido sair

_CENARIOS_DE_FECHAMENTO: list[tuple[str, str, Callable[[], _ConexaoFalsa], bool]] = [
    ("ok", "ok", lambda: _fabrica_ok().conexao, _NAO),
    ("302", "servidor_recusou", lambda: _fabrica_ok(status=302).conexao, _NAO),
    ("500", "servidor_recusou", lambda: _fabrica_ok(status=500).conexao, _NAO),
    (
        "tipo-errado",
        "resposta_malformada",
        lambda: _fabrica_ok(cabecalhos={"Content-Type": "text/html"}).conexao,
        _NAO,
    ),
    (
        "tls",
        "transporte_recusado",
        lambda: _ConexaoFalsa(erro_no_pedido=ssl.SSLCertVerificationError("x")),
        _NAO,
    ),
    ("oserror", "sem_rede", lambda: _ConexaoFalsa(erro_no_pedido=OSError("x")), _NAO),
    ("timeout", "sem_rede", lambda: _ConexaoFalsa(erro_na_resposta=TimeoutError()), _NAO),
    (
        "leitura-incompleta",
        "sem_rede",
        lambda: _ConexaoFalsa(_RespostaFalsa(erro_na_leitura=http.client.IncompleteRead(b""))),
        _NAO,
    ),
    ("corpo-recusado", "resposta_malformada", lambda: _fabrica_ok(b"nao e rds").conexao, _NAO),
    ("cancelado-durante", "cancelado", lambda: _fabrica_ok().conexao, _SIM),
]


@pytest.mark.parametrize(
    ("desfecho", "conexao", "cancelamentos"),
    [c[1:] for c in _CENARIOS_DE_FECHAMENTO],
    ids=[c[0] for c in _CENARIOS_DE_FECHAMENTO],
)
def test_a_conexao_aberta_e_fechada_uma_vez_em_todo_desfecho(
    desfecho: str, conexao: Callable[[], _ConexaoFalsa], cancelamentos: bool
) -> None:
    """Uma conexão esquecida aberta não reprova nada por si: o fechamento só se vê contando."""
    fabrica = _Fabrica(conexao())
    cancela = _cancela_depois_do_pedido(fabrica)
    r = _envia(fabrica, cancelado=lambda: cancelamentos and cancela())
    assert r.desfecho == desfecho, "âncora: o cenário chegou ao ramo que declara"
    assert fabrica.conexao.fechamentos == 1


# ---------------------------------------------------------------------------
# A espera que se interrompe (revisão de 02/10/2026)
# ---------------------------------------------------------------------------


@dataclass
class _SocketFalso:
    """O socket da conexão: `shutdown` solta quem está preso em `getresponse`."""

    solta: threading.Event = field(default_factory=threading.Event)
    derrubadas: int = 0

    def shutdown(self, como: int, /) -> None:
        self.derrubadas += 1
        self.solta.set()


@dataclass
class _ConexaoQuePrende(_ConexaoFalsa):
    """Prende em `getresponse` até o socket ser derrubado — ou até `solta_sozinha` ser posto —,
    como um servidor que aceitou o pedido e não responde."""

    sock: _SocketFalso | None = field(default_factory=_SocketFalso)
    solta_sozinha: threading.Event = field(default_factory=threading.Event)
    preso: threading.Event = field(default_factory=threading.Event)

    def getresponse(self) -> _RespostaFalsa:
        self.preso.set()
        while not self.solta_sozinha.is_set():
            if self.sock is not None and self.sock.solta.wait(0.01):
                raise OSError("o socket foi derrubado")
        return self.resposta


def test_cancelar_durante_a_espera_derruba_o_socket_e_volta_logo() -> None:
    """O «Cancelar» deixa de esperar o servidor: a espera consulta o cancelamento, derruba o
    socket, e a thread da troca sai e fecha a conexão. Antes, o cancelamento só era visto depois
    da resposta, até 60 s por operação de socket depois."""
    conexao = _ConexaoQuePrende(_RespostaFalsa(corpo=_resposta_sintetica()))
    fabrica = _Fabrica(conexao)
    inicio = time.monotonic()
    r = ext.envia_ao_servidor(
        ext.monta_carga(NOMES), NOMES, conexao=fabrica, cancelado=conexao.preso.is_set
    )
    assert conexao.preso.is_set(), "âncora: a troca chegou a prender em `getresponse`"
    assert time.monotonic() - inicio < 5.0
    assert r.desfecho == "cancelado" and "pode ter chegado" in r.motivo
    assert conexao.sock is not None and conexao.sock.derrubadas == 1
    assert conexao.fechamentos == 1, "a thread soltou-se e fechou a conexão"


def test_o_teto_do_envio_inteiro_corta_o_servidor_que_nao_termina(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """O teto por operação de socket não corta um servidor que manda um byte de vez em quando; o
    teto do envio inteiro corta. O par: com o teto de produção, a mesma conexão responde."""
    monkeypatch.setattr(ext, "TEMPO_TOTAL_S", 0.3)
    conexao = _ConexaoQuePrende(_RespostaFalsa(corpo=_resposta_sintetica()))
    r = _envia(_Fabrica(conexao))
    assert r.desfecho == "sem_rede" and "não terminou de responder em 0.3 s" in r.motivo
    assert conexao.sock is not None and conexao.sock.derrubadas == 1
    controle = _ConexaoQuePrende(_RespostaFalsa(corpo=_resposta_sintetica()))
    controle.solta_sozinha.set()
    monkeypatch.setattr(ext, "TEMPO_TOTAL_S", 120.0)
    assert _envia(_Fabrica(controle)).desfecho == "ok"


def test_conexao_sem_socket_a_derrubar_ainda_devolve_o_cancelamento() -> None:
    """Antes de o socket existir — a resolução do nome — não há o que derrubar: a espera volta
    mesmo assim, e a thread abandonada termina sozinha (é `daemon`)."""
    conexao = _ConexaoQuePrende(_RespostaFalsa(corpo=_resposta_sintetica()), sock=None)
    try:
        r = _envia(_Fabrica(conexao), cancelado=conexao.preso.is_set)
        assert r.desfecho == "cancelado"
        assert conexao.fechamentos == 0, "âncora: a thread continua presa, sem socket a derrubar"
    finally:
        conexao.solta_sozinha.set()


@pytest.fixture
def servidor_mudo() -> Iterator[tuple[int, list[socket.socket]]]:
    """Um servidor em 127.0.0.1 que aceita e não responde — nem HTTP, nem o aperto de mão do TLS.
    Sem rede: nada sai da máquina."""
    servidor = socket.create_server(("127.0.0.1", 0))
    servidor.settimeout(0.05)
    aceitos: list[socket.socket] = []
    parar = threading.Event()

    def aceita() -> None:
        while not parar.is_set():
            try:
                aceitos.append(servidor.accept()[0])
            except OSError:
                continue

    fio = threading.Thread(target=aceita, daemon=True)
    fio.start()
    try:
        yield servidor.getsockname()[1], aceitos
    finally:
        parar.set()
        fio.join(2)
        servidor.close()
        for c in aceitos:
            c.close()


@pytest.mark.parametrize("protocolo", ["http", "tls-no-aperto-de-mao"])
def test_cancelar_solta_a_thread_num_socket_de_verdade(
    servidor_mudo: tuple[int, list[socket.socket]], protocolo: str
) -> None:
    """O duplo de `_ConexaoQuePrende` solta quando o seu `shutdown` é chamado — e um socket de
    verdade NÃO solta assim neste Windows: medido em 02/10/2026, a espera voltava e a thread da
    troca continuava presa, por HTTP e no aperto de mão do TLS. Este é o teste que o duplo não
    podia ser: o `recv` é do sistema, e a conexão TLS é a classe de produção."""
    porta, aceitos = servidor_mudo
    fabricas: dict[str, Callable[[str, float], Any]] = {
        "http": lambda _h, t: http.client.HTTPConnection("127.0.0.1", porta, timeout=t),
        "tls-no-aperto-de-mao": lambda _h, t: ext._ConexaoHttps(f"127.0.0.1:{porta}", t),
    }
    cancelar = threading.Event()
    presas: list[threading.Thread] = []

    def cancela_quando_aceitar() -> None:
        prazo = time.monotonic() + 5
        while not aceitos and time.monotonic() < prazo:
            time.sleep(0.01)
        time.sleep(0.2)  # a troca já está presa no `recv`
        # Capturadas ANTES do cancelamento: depois dele uma thread que terminou some de
        # `enumerate`, e uma lista vazia passaria a asserção de baixo por vacuidade.
        presas.extend(t for t in threading.enumerate() if t.name == "envio-ao-metaboanalyst")
        cancelar.set()

    threading.Thread(target=cancela_quando_aceitar, daemon=True).start()
    r = ext.envia_ao_servidor(b"x", ["x"], conexao=fabricas[protocolo], cancelado=cancelar.is_set)
    assert aceitos, "âncora: a troca chegou a conectar ao servidor mudo"
    assert r.desfecho == "cancelado"
    assert presas, "âncora: a thread da troca estava viva e presa quando o cancelamento veio"
    for th in presas:
        th.join(2)
    assert not any(th.is_alive() for th in presas), "a thread da troca continua presa no socket"


def test_defeito_que_nao_e_de_rede_nao_vira_desfecho() -> None:
    """`NUNCA levanta por rede ou resposta` — e só por elas: um defeito do programa na thread da
    troca chega a quem chamou como a exceção que é, em vez de virar «sem rede»."""
    fabrica = _Fabrica(_ConexaoFalsa(erro_no_pedido=TypeError("defeito do programa")))
    with pytest.raises(TypeError, match="defeito do programa"):
        _envia(fabrica)


def test_a_conexao_de_producao_verifica_o_certificado_e_o_nome() -> None:
    """A fábrica de produção, sem rede: `HTTPSConnection` só conecta no primeiro pedido."""
    c = ext._conexao_padrao(ext.HOST, 7.0)
    assert isinstance(c, http.client.HTTPSConnection)
    assert (c.host, c.timeout) == (ext.HOST, 7.0)
    contexto = c._context  # type: ignore[attr-defined]
    assert contexto.verify_mode == ssl.CERT_REQUIRED and contexto.check_hostname
    padrao = inspect.signature(ext.envia_ao_servidor).parameters["conexao"].default
    assert padrao is ext._conexao_padrao


# ---------------------------------------------------------------------------
# A resposta malformada nunca escapa como outra exceção (revisão de 02/10/2026)
# ---------------------------------------------------------------------------


def _com_matriz(muda: Callable[[Vetor], Vetor]) -> bytes:
    raiz = le_rds(_resposta_sintetica())
    assert isinstance(raiz, Lista)
    anal = raiz.campo("analSet")
    assert isinstance(anal, Lista)
    i = anal.nomes().index("ora.mat")
    matriz = anal.itens[i]
    assert isinstance(matriz, Vetor)
    itens = list(anal.itens)
    itens[i] = muda(matriz)
    j = raiz.nomes().index("analSet")
    de_fora = list(raiz.itens)
    de_fora[j] = Lista(tuple(itens), anal.atributos)
    return escreve_rds(Lista(tuple(de_fora), raiz.atributos))


def _com_dim(*dim: int | None) -> Callable[[Vetor], Vetor]:
    return lambda m: Vetor(m.tipo, m.valores, {**m.atributos, "dim": Vetor("inteiro", dim)})


_K = len(VIAS_INVENTADAS)
_MATRIZES_TORTAS: list[tuple[str, Callable[[Vetor], Vetor], str]] = [
    ("dim-com-NA", _com_dim(_K, None), "forma esperada"),
    ("dim-negativo", _com_dim(-1, 6), "forma esperada"),
    ("dim-com-tres", _com_dim(_K, 6, 1), "forma esperada"),
    (
        "matriz-curta",
        lambda m: Vetor(m.tipo, m.valores[:-1], m.atributos),
        f"tem {_K * 6 - 1} valores, e `dim` diz {_K} por 6",
    ),
    (
        "matriz-longa",
        lambda m: Vetor(m.tipo, (*m.valores, 0.5), m.atributos),
        f"tem {_K * 6 + 1} valores",
    ),
    ("valor-NaN", lambda m: Vetor(m.tipo, (float("nan"), *m.valores[1:]), m.atributos), "linha 1"),
]


@pytest.mark.parametrize(
    ("muda", "motivo"), [c[1:] for c in _MATRIZES_TORTAS], ids=[c[0] for c in _MATRIZES_TORTAS]
)
def test_matriz_torta_e_resposta_malformada_com_o_motivo_da_guarda(
    muda: Callable[[Vetor], Vetor], motivo: str
) -> None:
    """Até a revisão de 02/10/2026, `dim` com NA levantava `ValueError` e a matriz curta
    `IndexError`, e as duas escapavam de `envia_ao_servidor` — e a tela dizia «O motor R falhou»
    sobre uma resposta do servidor. Cada caso tem a SUA guarda, e o `match` é o dela."""
    assert _envia(_fabrica_ok(_com_matriz(lambda m: m))).desfecho == "ok", "âncora: sem a planta"
    r = _envia(_fabrica_ok(_com_matriz(muda)))
    assert (r.desfecho, r.dados) == ("resposta_malformada", None)
    assert motivo in r.motivo


def test_byte_trocado_na_resposta_so_da_ok_ou_resposta_malformada() -> None:
    """Toda posição da resposta sintética, trocada por três valores: `le_resposta` devolve ou
    levanta `RdsRecusadoError`, e nenhuma outra exceção — a promessa de `envia_ao_servidor`.

    É uma amostra e não uma prova; o que ela fixa é que as guardas explícitas cobrem o que um
    byte trocado alcança, sem um `except` largo que as dispense.
    """
    base = _resposta_sintetica()
    gz = gzip.compress(base, mtime=0)
    corpos = [
        base[:i] + bytes([v]) + base[i + 1 :] for i in range(len(base)) for v in (0, 0x7F, 0xFF)
    ]
    corpos += [gz[:i] + bytes([v]) + gz[i + 1 :] for i in range(len(gz)) for v in (0, 0xFF)]
    desfechos = {"ok": 0, "recusada": 0}
    for corpo in corpos:
        try:
            ext.le_resposta(corpo, NOMES)
            desfechos["ok"] += 1
        except RdsRecusadoError:
            desfechos["recusada"] += 1
    assert desfechos["ok"] > 0 and desfechos["recusada"] > 0, desfechos


# ---------------------------------------------------------------------------
# A lista que o servidor usa (medido em 02/10/2026)
# ---------------------------------------------------------------------------


def test_na_lista_e_a_dos_nomes_em_alguma_via_e_nao_a_dos_enviados() -> None:
    """Um nome enviado que não está em via nenhuma não entra na lista do servidor. Medido em
    02/10/2026 com os 63 nomes públicos do fundo: `q = 61` em todas as 54 vias, e 61 nomes nos
    acertos. A resposta sintética tem os seis `NOMES` nas vias; enviar um sétimo, fora delas,
    é o mesmo caso."""
    corpo = _resposta_sintetica()
    enviados = (*NOMES, "Nome fora de toda via")
    dados = ext.le_resposta(corpo, enviados)
    assert dados["na_lista"] == len(NOMES)
    assert ora_da_resposta(dados).na_lista == len(NOMES), "a conferência aceita a conta"
    # O par: com a conta antiga (os enviados), a mesma resposta seria recusada inteira.
    with pytest.raises(RespostaInvalidaError):
        ora_da_resposta({**dados, "na_lista": len(enviados)})


def test_acerto_que_nao_foi_enviado_e_resposta_malformada() -> None:
    """A conta dos acertos só vale se os acertos forem nomes da lista."""
    assert ext.le_resposta(_resposta_sintetica(), NOMES)["na_lista"] == len(NOMES), (
        "âncora: com a lista inteira a mesma resposta passa"
    )
    sem_urea = [n for n in NOMES if n != "Urea"]
    with pytest.raises(RdsRecusadoError, match="um acerto que não foi enviado"):
        ext.le_resposta(_resposta_sintetica(), sem_urea)


def test_o_status_do_servidor_viaja_na_recusa() -> None:
    r = _envia(_fabrica_ok(status=500, reason="Internal Server Error"))
    assert (r.desfecho, r.status) == ("servidor_recusou", 500)
    assert _envia(_fabrica_ok()).status is None
