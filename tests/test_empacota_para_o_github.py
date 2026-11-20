"""O portão que decide o que sai da máquina, exercitado nas DUAS direções.

Este arquivo existe porque uma revisão de código achou três defeitos P0 num mecanismo que eu
tinha declarado exercitado. `scripts/empacota_para_o_github.py` decide o que vai para uma
*release* pública do GitHub, e a pasta que ele lê traz medição por amostra de um estudo clínico
real.

A primeira versão foi corrida à mão com uma planta de 44 bytes e vista abortar. Isso prova que
ela **conseguia** acusar; não prova que acusava a coisa certa. O que os revisores acharam:

| O defeito | Por que a planta manual não o viu |
|---|---|
| o regex cobria **três** de quatro rótulos | a planta levava `ATLL`, um dos três que ele já via |
| `refino.db-wal` e `refino.log` entravam | a planta caiu em `_internal/`, não na raiz |
| arquivo ilegível contava como limpo | nada na corrida manual era ilegível |

> A pergunta que morde aqui: *quais instâncias deste defeito eu JÁ CONSERTEI, e o meu oráculo
> acha as que eu não consertei?* A planta foi construída a partir do caso que eu já sabia; os
> três defeitos viviam exatamente onde ela não caía.

**Por isso cada teste abaixo planta na direção em que a primeira versão falhava**, e não na
direção em que ela passava.
"""

from __future__ import annotations

import pathlib
import sys
import zipfile

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from empacota_para_o_github import (  # noqa: E402
    ISENTOS_DE_CONTEUDO,
    PODEM_SUBIR,
    ROTULOS,
    e_isento_de_conteudo,
    e_privado,
    marcas,
)

from refino.io.validators import CANONICAL_GROUPS  # noqa: E402


# --------------------------------------------------------------------------- o oráculo dos rótulos
def test_o_oraculo_cobre_todos_os_grupos_canonicos_e_a_lista_nao_esta_vazia() -> None:
    """A bijeção com a fonte única, **nas duas direções e com a guarda de não-vacuidade aqui**.

    A guarda vai nesta função e não numa vizinha: se `CANONICAL_GROUPS` ficasse vazio, o
    laço abaixo não teria o que iterar e o teste passaria **sem conferir rótulo nenhum** — que é
    o desfecho de sucesso mais convincente que existe.

    **A direção de ida é a que pegou o defeito P0:** a primeira versão digitou `ATLL`,
    `HAM[_/ -]?TSP` e `NoHTLV` à mão e perdeu `Asymptomatic`, que é 18 dos 68 pacientes.
    """
    assert len(CANONICAL_GROUPS) >= 4, (
        f"o vocabulário canônico tem {len(CANONICAL_GROUPS)} membros — a asserção abaixo seria "
        "vácua, e foi exactamente a forma do defeito que este arquivo existe para impedir"
    )

    for grupo in sorted(CANONICAL_GROUPS):
        assert ROTULOS.search(grupo.encode("ascii")), (
            f"{grupo!r} é grupo canônico e o oráculo do empacotador não o reconhece — "
            "um arquivo que só traga este diagnóstico seria publicado"
        )


@pytest.mark.parametrize("forma", [b"HAMTSP", b"HAM/TSP", b"HAM-TSP", b"HAM_TSP", b"hamtsp"])
def test_as_formas_medidas_de_ham_tsp_sao_todas_reconhecidas(forma: bytes) -> None:
    """As três formas que o corpus real traz, medidas em 19/09/2026: 85, 26 e 18 ocorrências."""
    assert marcas(forma), f"{forma!r} existe no corpus e o oráculo não a vê"


def test_o_oraculo_nao_acusa_texto_inocente() -> None:
    """A direção oposta. Sem ela, um oráculo que casasse tudo passaria no teste de cima.

    `ATLANTICO` contém `ATL`, e `NoHTLV` não pode ser casado por `HTLV` solto — se o padrão
    fosse frouxo, `refino.exe` e as 63 DLLs acusariam e a publicação abortaria sem motivo.
    """
    assert marcas(b"ATLANTICO ocean, HAMSTER, nothing here") == []
    assert marcas(b"") == []


# --------------------------------------------------------------------------- os artefatos privados
@pytest.mark.parametrize(
    "caminho",
    [
        "refino.db",
        "refino.db-wal",
        "refino.db-journal",
        "refino.db-shm",
        "refino.log",
        "saidas/1-matrizes-corrigidas/pos_hilic_corrigir_none.csv",
        "Saidas/x.csv",
        "REFINO.DB-WAL",
    ],
)
def test_todo_artefato_de_privacidade_e_recusado(caminho: str) -> None:
    """A planta cai onde a primeira versão falhava: os **sidecars** e o **log**.

    `refino.db-wal` não é igual a `refino.db`, e a primeira versão comparava por igualdade —
    então os três sidecars do SQLite entravam. `persistence/config.py:184` regista que um
    revisor achou `PAC-0043` e `ATLL` crus dentro de um `-wal` real.

    `Saidas/` e `REFINO.DB-WAL` cobrem a outra metade: o sistema de arquivos do Windows não
    distingue maiúsculas e a comparação distinguia.
    """
    assert e_privado(pathlib.PurePosixPath(caminho)), f"{caminho} viajaria para a nuvem"


@pytest.mark.parametrize(
    "caminho", ["refino.exe", "VERSAO.txt", "GUIA.html", "_internal/base_library.zip"]
)
def test_o_par_causal_do_que_deve_viajar_nao_e_recusado(caminho: str) -> None:
    """Sem esta direção, um predicado que recusasse tudo passaria no teste de cima."""
    assert not e_privado(pathlib.PurePosixPath(caminho))


def test_a_lista_de_privados_cobre_os_cinco_nomes_do_roteiro_de_entrega() -> None:
    """As duas listas irmãs não podem divergir — e divergiam: duas contra cinco.

    `scripts/prepara_entrega.ps1` apaga cinco artefatos da pasta que o laboratório recebe em
    mão. O empacotador excluía dois, e estava no destino **mais** exposto.
    """
    roteiro = (REPO / "scripts" / "prepara_entrega.ps1").read_text(
        encoding="utf-8", errors="replace"
    )
    assert "$paraTirar" in roteiro, "o roteiro mudou de forma — reconfira a lista irmã à mão"
    for nome in ("refino.db", "refino.db-journal", "refino.db-wal", "refino.db-shm", "refino.log"):
        assert nome in roteiro, f"{nome} saiu do roteiro de entrega — as duas listas divergiram"
        assert e_privado(pathlib.PurePosixPath(nome)), f"{nome} é privado lá e não aqui"


# --------------------------------------------------------------------------- a isenção de conteúdo
def test_a_isencao_e_por_caminho_e_nao_por_extensao() -> None:
    """Um `.html` ou `.json` escrito pelo aplicativo na raiz **é lido**.

    A primeira versão isentava por extensão e dispensava 3.155 dos 4.770 arquivos da leitura.
    O `cwd` de um `refino.exe` aberto por duplo clique é a própria pasta, então é ali que uma
    exportação da janela cai.
    """
    assert e_isento_de_conteudo(pathlib.PurePosixPath("GUIA.html"))
    assert e_isento_de_conteudo(pathlib.PurePosixPath("_internal/refino/io/validators.pyc"))

    assert not e_isento_de_conteudo(pathlib.PurePosixPath("resultado_do_estudo.html"))
    assert not e_isento_de_conteudo(pathlib.PurePosixPath("export.json"))
    assert not e_isento_de_conteudo(pathlib.PurePosixPath("_internal/scipy/_lib/x.pyc"))


def test_todo_isento_de_conteudo_esta_na_allowlist_ou_sob_um_nome_que_esta() -> None:
    """Isentar de leitura algo que nem pode subir seria isenção sem referente."""
    for isento in ISENTOS_DE_CONTEUDO:
        topo = isento.split("/")[0]
        assert topo in PODEM_SUBIR, f"{isento} é isento de leitura e não está em PODEM_SUBIR"


def test_a_allowlist_e_os_privados_nao_se_sobrepoem() -> None:
    """Um nome nas duas listas tornaria o desfecho dependente da ordem dos testes."""
    for nome in PODEM_SUBIR:
        assert not e_privado(pathlib.PurePosixPath(nome)), f"{nome} está nas duas listas"
    assert "saidas" not in PODEM_SUBIR, "as saídas do estudo não podem subir para a nuvem"


# --------------------------------------------------------------------------- o fluxo, ponta a ponta
def _monta(raiz: pathlib.Path, arquivos: dict[str, bytes]) -> pathlib.Path:
    """Uma pasta entregue de mentira, com o mínimo que o roteiro exige."""
    dist = raiz / "dist" / "refino"
    base = {
        "VERSAO.txt": b"refino\n\ncommit    : abcdef1234567890\n",
        "refino.exe": b"MZ binario qualquer",
        "_internal/base_library.zip": b"PK conteudo",
    }
    for nome, dados in {**base, **arquivos}.items():
        alvo = dist / nome
        alvo.parent.mkdir(parents=True, exist_ok=True)
        alvo.write_bytes(dados)
    return dist


@pytest.fixture
def empacotador(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch):  # type: ignore[no-untyped-def]
    """O módulo apontado para uma árvore de teste — nunca para a pasta entregue real."""
    import empacota_para_o_github as modulo

    monkeypatch.setattr(modulo, "RAIZ", tmp_path)
    monkeypatch.setattr(modulo, "DIST", tmp_path / "dist" / "refino")
    return modulo


def test_arvore_limpa_escreve_o_zip_e_sai_zero(
    empacotador, tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:  # type: ignore[no-untyped-def]
    """A direção de sucesso, e ela é o par causal de todas as plantas abaixo."""
    _monta(tmp_path, {})
    assert empacotador.main() == 0
    zips = list((tmp_path / "dist").glob("*.zip"))
    assert len(zips) == 1, f"esperava um zip, achei {zips}"
    with zipfile.ZipFile(zips[0]) as z:
        assert "refino/VERSAO.txt" in z.namelist()
    assert "nenhum arquivo lido traz rotulo" in capsys.readouterr().out


def test_a_planta_com_rotulo_aborta_e_nao_escreve_zip(
    empacotador, tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:  # type: ignore[no-untyped-def]
    """A planta cai num `.html` da RAIZ, que é onde o aplicativo exporta — não em `_internal/`.

    E o rótulo é `Asymptomatic`, que é o que a primeira versão **não** conhecia.
    """
    _monta(tmp_path, {"figura_do_estudo.html": b"<p>paciente ID74 Asymptomatic</p>"})
    assert empacotador.main() == 1
    assert list((tmp_path / "dist").glob("*.zip")) == [], "escreveu o zip mesmo tendo abortado"
    saida = capsys.readouterr().out
    assert "ABORTADO" in saida


def test_o_nome_desconhecido_no_topo_aborta_em_vez_de_ser_descartado(
    empacotador, tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:  # type: ignore[no-untyped-def]
    """A metade que faltava: nenhuma lista de proibidos antecipa o que o aplicativo escreve.

    Um CSV exportado pela janela cai na raiz da pasta. Sob a *denylist* da primeira versão ele
    **subia**; sob a allowlist ele para a publicação e é nomeado.
    """
    _monta(tmp_path, {"matriz_corrigida.csv": b"sem rotulo nenhum aqui,1.0\n"})
    assert empacotador.main() == 1
    assert list((tmp_path / "dist").glob("*.zip")) == []
    assert "matriz_corrigida.csv" in capsys.readouterr().out


def test_o_arquivo_ilegivel_aborta_em_vez_de_contar_como_limpo(
    empacotador,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:  # type: ignore[no-untyped-def]
    """O P0 mais silencioso: `except OSError: continue` dizia «limpo» e empacotava na mesma.

    O gatilho real está documentado — o passo 2 de `prepara_entrega.ps1` («nenhum processo do
    programa pode estar vivo») descreve o programa aberto sobre esta pasta, e nessa condição há
    `-wal`/`-shm` com lock.
    """
    _monta(tmp_path, {})
    original = pathlib.Path.read_bytes

    def recusa(self: pathlib.Path) -> bytes:
        if self.name == "refino.exe":
            raise PermissionError("travado por outro processo")
        return original(self)

    monkeypatch.setattr(pathlib.Path, "read_bytes", recusa)
    assert empacotador.main() == 1
    assert list((tmp_path / "dist").glob("*.zip")) == []
    saida = capsys.readouterr().out
    assert "ILEGIVEL" in saida and "refino.exe" in saida


def test_os_artefatos_de_privacidade_ficam_fora_do_zip_escrito(
    empacotador, tmp_path: pathlib.Path
) -> None:  # type: ignore[no-untyped-def]
    """Conferido no ZIP, e não na lista que o produziu."""
    _monta(
        tmp_path,
        {
            "refino.db": b"SQLite format 3",
            "refino.db-wal": b"PAC-0043 ATLL bytes crus",
            "refino.log": b"refino: aviso",
            "saidas/m.csv": b"ID99_Asymptomatic,1.0",
        },
    )
    assert empacotador.main() == 0
    zips = list((tmp_path / "dist").glob("*.zip"))
    with zipfile.ZipFile(zips[0]) as z:
        nomes = z.namelist()
    for proibido in ("refino.db", "refino.db-wal", "refino.log", "saidas"):
        assert not [n for n in nomes if proibido in n], f"{proibido} entrou no zip publicado"


def test_o_html_entregue_sem_charset_aborta_a_publicacao(
    empacotador, tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:  # type: ignore[no-untyped-def]
    """A conferência da INSTÂNCIA QUE VIAJA.

    `tests/test_charset_dos_documentos.py` lê `docs/*.html`; o que quem recebe a pasta abre por
    duplo clique são estas cópias, com nomes diferentes. Este é o único ponto do fluxo que vê o
    artefato publicado, e por isso a conferência do charset vive aqui também.

    **E este é o único desfecho de acusação da SEGUNDA passada que um teste vê** (revisão de
    processo de 01/10/2026). A passada que reconfere o zip tem cinco fontes de acusação; as outras
    quatro ficam sem planta, e o motivo é declarado em vez de forçado com `monkeypatch`:

    - `artefato de privacidade` e `fora por licenca` são inalcançáveis por construção. O zip é
      escrito a partir de `entram`, de onde a primeira passada já tirou os dois, pelos mesmos
      `e_privado` e `e_fora_por_licenca` sobre o mesmo primeiro componente do caminho. A ausência
      deles no zip escrito é o que afirmam
      `test_os_artefatos_de_privacidade_ficam_fora_do_zip_escrito` e
      `test_a_pasta_de_dados_fica_fora_por_licenca_e_sem_ela_na_lista_abortaria`.
    - o motivo de aborto da árvore R e os rótulos de diagnóstico só acusam aqui com um arquivo cujos
      bytes mudem entre a leitura da primeira passada e a escrita do zip. Plantá-lo exigiria um
      duplo que devolvesse bytes diferentes em leituras sucessivas, e o teste passaria a medir o
      duplo. Os mesmos dois julgamentos têm par causal na primeira passada (`julga_na_arvore_r` e
      `marcas`, chamados pelas duas), nos testes da árvore R e da planta com rótulo.
    """
    _monta(tmp_path, {"GUIA.html": b"<title>Guia sem declarar codificacao</title>"})
    assert empacotador.main() == 1
    assert list((tmp_path / "dist").glob("*.zip")) == [], "publicou um documento com mojibake"
    assert "nao declara UTF-8" in capsys.readouterr().out


def test_o_par_causal_o_html_que_declara_passa(empacotador, tmp_path: pathlib.Path) -> None:  # type: ignore[no-untyped-def]
    """Sem esta direção, uma conferência que recusasse todo `.html` passaria no teste acima."""
    _monta(tmp_path, {"GUIA.html": b'<meta charset="utf-8">\n<title>Guia</title>'})
    assert empacotador.main() == 0


def test_a_pasta_vazia_aborta_em_vez_de_escrever_zip_vazio(
    empacotador, tmp_path: pathlib.Path
) -> None:  # type: ignore[no-untyped-def]
    """Não-vacuidade do outro lado: sem esta guarda, uma pasta vazia imprimia sucesso."""
    (tmp_path / "dist" / "refino").mkdir(parents=True)
    assert empacotador.main() == 1


# --------------------------------------------------------------------------- o R e as vias
def test_a_allowlist_e_o_que_fica_na_entrega_menos_as_duas_divergencias_declaradas() -> None:
    """`PODEM_SUBIR` afirma ser o `$temQueFicar` menos `saidas` e menos `vias-dados`.

    A frase vivia só na docstring, sem leitor. A lista de lá é DERIVADA do roteiro, não
    retipada: um nome novo na entrega que não seja decidido aqui reprova nas duas direções.
    """
    import re

    from empacota_para_o_github import FORA_POR_LICENCA

    texto = (REPO / "scripts" / "prepara_entrega.ps1").read_text(encoding="utf-8")
    bloco = re.search(r"\$temQueFicar = @\((.*?)\)", texto, re.S)
    assert bloco is not None, "o roteiro de entrega deixou de declarar `$temQueFicar`"
    na_entrega = set(re.findall(r'"([^"]+)"', bloco.group(1)))
    assert len(na_entrega) >= 10, f"âncora: a extração leu {sorted(na_entrega)}"
    decididos = set(PODEM_SUBIR) | {"saidas"} | set(FORA_POR_LICENCA)
    assert na_entrega == decididos, (
        f"só na entrega: {sorted(na_entrega - decididos)}; "
        f"só nas listas do empacotador: {sorted(decididos - na_entrega)}"
    )


def _sha(dados: bytes) -> str:
    import hashlib

    return hashlib.sha256(dados).hexdigest()


ARVORE: dict[str, bytes] = {
    "bin/x64/Rscript.exe": b"MZ rscript de mentira",
    # base64 de certificado com o rotulo por acaso — o falso positivo medido na arvore real
    "etc/curl-ca-bundle.crt": b"MIIB...AtlLqz...==",
    "library/MetaboAnalystR/DESCRIPTION": b"Package: MetaboAnalystR\nPackaged: ontem\n",
    # documento de pacote sem charset, como `timeDate/COPYRIGHT.html` na arvore real: a conferencia
    # de charset e dos documentos de entrega, e a arvore R e julgada pelo manifesto
    "library/timeDate/COPYRIGHT.html": b"<html><p>sem charset</p></html>",
}
ROTEIRO = b"# roteiro das vias\n"


@pytest.fixture
def com_r(empacotador, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch):  # type: ignore[no-untyped-def]
    """Uma árvore R de mentira, o manifesto dela, o roteiro e o ativo de fonte — tudo em `tmp`."""
    fora = tmp_path / "repo"
    fora.mkdir()
    manifesto = fora / "r_portatil_arvore.sha256"
    manifesto.write_text(
        "".join(f"{_sha(b)} {len(b)} {rel}\n" for rel, b in ARVORE.items()), encoding="utf-8"
    )
    pacotes = fora / "r_portatil_pacotes.txt"
    pacotes.write_text("MetaboAnalystR 4.0.0\n", encoding="utf-8")
    roteiro = fora / "vias_entrypoint.R"
    roteiro.write_bytes(ROTEIRO)
    monkeypatch.setattr(empacotador, "MANIFESTO_DA_ARVORE_R", manifesto)
    monkeypatch.setattr(empacotador, "LISTA_DE_PACOTES_DO_R", pacotes)
    monkeypatch.setattr(empacotador, "ROTEIRO_DAS_VIAS", roteiro)

    def ativo(*, pacotes_do_ativo: bytes | None = None) -> None:
        destino = tmp_path / "dist" / empacotador.NOME_DO_ATIVO
        destino.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(destino, "w") as z:
            z.writestr("FONTES.sha256", "x  fontes/R-4.4.3.tar.gz\n")
            z.writestr("roteiros/r_portatil_pacotes.txt", pacotes_do_ativo or pacotes.read_bytes())
            z.writestr("roteiros/r_portatil_arvore.sha256", manifesto.read_bytes())

    arquivos = {f"R/{rel}": b for rel, b in ARVORE.items()} | {"R/vias_entrypoint.R": ROTEIRO}
    return arquivos, ativo


def test_a_arvore_r_que_bate_com_o_manifesto_sobe_sem_julgar_rotulos(
    empacotador, com_r, tmp_path: pathlib.Path
) -> None:  # type: ignore[no-untyped-def]
    """O certificado com `AtlL` sobe, porque os bytes são os da montagem — e só por isso."""
    assert empacotador.marcas(ARVORE["etc/curl-ca-bundle.crt"]), (
        "âncora: sem rótulo no certificado, a isenção não teria o que dispensar"
    )
    arquivos, ativo = com_r
    ativo()
    _monta(tmp_path, arquivos)
    assert empacotador.main() == 0
    (zip_,) = list((tmp_path / "dist").glob("refino-windows-*.zip"))
    with zipfile.ZipFile(zip_) as z:
        assert "refino/R/etc/curl-ca-bundle.crt" in z.namelist()
        assert "refino/R/vias_entrypoint.R" in z.namelist()


@pytest.mark.parametrize(
    ("planta", "motivo"),
    [
        ({"R/etc/plantado.txt": b"inocente"}, "fora do manifesto"),
        ({"R/etc/curl-ca-bundle.crt": b"MIIB...outro...=="}, "difere do manifesto"),
        ({"R/vias_entrypoint.R": b"# outro roteiro\n"}, "roteiro das vias"),
        ({"R/library/MetaboAnalystR/DESCRIPTION": b"Packaged: PAC-0043 ATLL\n"}, "ATLL"),
    ],
    ids=["fora_do_manifesto", "difere", "roteiro_trocado", "carimbo_com_rotulo"],
)
def test_a_arvore_r_que_nao_bate_com_o_manifesto_aborta(
    empacotador, com_r, tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str], planta, motivo
) -> None:  # type: ignore[no-untyped-def]
    """O par causal do teste acima: a mesma pasta com UM arquivo plantado ou trocado."""
    arquivos, ativo = com_r
    ativo()
    _monta(tmp_path, arquivos | planta)
    assert empacotador.main() == 1
    assert list((tmp_path / "dist").glob("refino-windows-*.zip")) == []
    assert motivo in capsys.readouterr().out


def test_o_carimbo_de_build_que_difere_sem_rotulo_e_lido_e_sobe(
    empacotador, com_r, tmp_path: pathlib.Path
) -> None:  # type: ignore[no-untyped-def]
    """O carimbo muda a cada montagem (decisão 4 das resoluções): ele é LIDO, e não abortado."""
    arquivos, ativo = com_r
    ativo()
    _monta(tmp_path, arquivos | {"R/library/MetaboAnalystR/DESCRIPTION": b"Packaged: hoje\n"})
    assert empacotador.main() == 0


def test_a_arvore_r_incompleta_aborta(
    empacotador, com_r, tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:  # type: ignore[no-untyped-def]
    arquivos, ativo = com_r
    ativo()
    _monta(tmp_path, {k: v for k, v in arquivos.items() if not k.endswith("Rscript.exe")})
    assert empacotador.main() == 1
    assert "falta na arvore R" in capsys.readouterr().out


def test_o_r_sem_o_ativo_de_fonte_aborta_e_o_ativo_de_outra_arvore_tambem(
    empacotador, com_r, tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:  # type: ignore[no-untyped-def]
    """A GPL pede a fonte junto dos binários: sem o ativo, ou com o de outra montagem, não sobe."""
    arquivos, ativo = com_r
    _monta(tmp_path, arquivos)
    assert empacotador.main() == 1
    assert "ativo de fonte" in capsys.readouterr().out
    ativo(pacotes_do_ativo=b"MetaboAnalystR 3.0.0\n")
    assert empacotador.main() == 1
    assert "nao e o desta arvore" in capsys.readouterr().out
    ativo()
    assert empacotador.main() == 0, "com o ativo certo, a mesma pasta sobe"


@pytest.mark.parametrize(
    ("falta", "motivo"),
    [
        ("FONTES.sha256", "nao traz FONTES.sha256"),
        ("roteiros/r_portatil_pacotes.txt", "nao traz roteiros/r_portatil_pacotes.txt"),
        # a segunda volta do laço: o primeiro roteiro confere, e o que falta é o segundo
        ("roteiros/r_portatil_arvore.sha256", "nao traz roteiros/r_portatil_arvore.sha256"),
        (None, "ilegivel: BadZipFile"),
    ],
    ids=["sem_fontes", "sem_lista_de_pacotes", "sem_manifesto", "ilegivel"],
)
def test_o_ativo_de_fonte_sem_uma_peca_ou_ilegivel_aborta(
    empacotador,
    com_r,
    tmp_path: pathlib.Path,
    capsys: pytest.CaptureFixture[str],
    falta: str | None,
    motivo: str,
) -> None:  # type: ignore[no-untyped-def]
    """Os três desfechos de `confere_ativo_de_fonte` que o teste de cima não planta.

    Cada zip de mentira é o ativo certo menos UMA peça (ou bytes que não são zip), e o par causal
    está na mesma função: o mesmo zip com a peça de volta deixa a mesma pasta subir.
    """
    arquivos, _ = com_r
    _monta(tmp_path, arquivos)
    destino = tmp_path / "dist" / empacotador.NOME_DO_ATIVO
    inteiro = {
        "FONTES.sha256": b"x  fontes/R-4.4.3.tar.gz\n",
        "roteiros/r_portatil_pacotes.txt": empacotador.LISTA_DE_PACOTES_DO_R.read_bytes(),
        "roteiros/r_portatil_arvore.sha256": empacotador.MANIFESTO_DA_ARVORE_R.read_bytes(),
    }

    def escreve(entradas: dict[str, bytes]) -> None:
        with zipfile.ZipFile(destino, "w") as z:
            for nome, dados in entradas.items():
                z.writestr(nome, dados)

    if falta is None:
        destino.write_bytes(b"isto nao e um zip")
    else:
        assert falta in inteiro, "âncora: a peça plantada como ausente é uma das que o ativo leva"
        escreve({n: d for n, d in inteiro.items() if n != falta})
    assert empacotador.main() == 1
    assert list((tmp_path / "dist").glob("refino-windows-*.zip")) == []
    assert motivo in capsys.readouterr().out

    escreve(inteiro)
    assert empacotador.main() == 0, "com a peça de volta, a mesma pasta sobe"


def test_a_pasta_de_dados_fica_fora_por_licenca_e_sem_ela_na_lista_abortaria(
    empacotador, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:  # type: ignore[no-untyped-def]
    """Par causal da terceira lista: com ela, a pasta fica fora e o zip sai; sem ela, aborta."""
    _monta(tmp_path, {"vias-dados/compound_db.qs": b"qs2 bytes"})
    assert empacotador.main() == 0
    (zip_,) = list((tmp_path / "dist").glob("refino-windows-*.zip"))
    with zipfile.ZipFile(zip_) as z:
        assert not [n for n in z.namelist() if "vias-dados" in n]
    zip_.unlink()
    monkeypatch.setattr(empacotador, "FORA_POR_LICENCA", ())
    assert empacotador.main() == 1, "sem a terceira lista, a pasta seria um nome desconhecido"


def test_o_ativo_de_fonte_com_outro_fim_de_linha_ainda_e_o_desta_arvore(
    empacotador, com_r, tmp_path: pathlib.Path
) -> None:  # type: ignore[no-untyped-def]
    """O checkout troca LF por CRLF no versionado; o conteúdo é o mesmo, e o ativo vale."""
    arquivos, ativo = com_r
    empacotador.LISTA_DE_PACOTES_DO_R.write_bytes(b"MetaboAnalystR 4.0.0\r\n")
    ativo(pacotes_do_ativo=b"MetaboAnalystR 4.0.0\n")
    _monta(tmp_path, arquivos)
    assert empacotador.main() == 0
