"""O portão da árvore R portátil, sem instalar R.

## O que ele confere, e por que sem R

A montagem (`scripts/monta_r_portatil.ps1`) leva meia hora e instala dois programas; a sonda
(`scripts/prova_r_embarcado.py`) precisa da árvore montada. Nenhuma das duas cabe em `make check`.
O que cabe é conferir os **artefatos versionados** que a montagem de referência deixou em
`scripts/` — e é neles que um descuido de edição aparece primeiro:

- **a lista de pacotes cobre a união dos despejos por fluxo.** Um nome que um fluxo carrega e a
  lista perdeu é um pacote que a poda tira, e o fluxo morre no laboratório.
- **nenhum proibido na lista nem no manifesto.** `Rserve` abre um servidor R sem autenticação; os
  outros são caminhos de processo ou de rede que nenhum fluxo usa.
- **o manifesto descreve a árvore podada pela lista, com as licenças.** O congelamento confere a
  árvore contra ele; a GPL exige o texto da licença.
- **os pinos ficam fixados uma vez.** Um pino trocado num lado, ou repetido com outro valor no
  outro, deixa o valor ambíguo e reprova.
- **a reprodução aceita diferença só nos carimbos de build.** Qualquer outra diferença entre duas
  montagens é deriva, e tem de reprovar.

Cada afirmação tem o seu par causal **em processo**: a planta é argumento da função sob teste, então
não há intermediário que possa não a receber.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path
from types import ModuleType

REPO = Path(__file__).resolve().parent.parent
SCRIPTS = REPO / "scripts"
LISTA = SCRIPTS / "r_portatil_pacotes.txt"
DESPEJOS = SCRIPTS / "r_portatil_namespaces"
MANIFESTO = SCRIPTS / "r_portatil_arvore.sha256"
INSTALA = SCRIPTS / "r_portatil_instala.R"
MONTA = SCRIPTS / "monta_r_portatil.ps1"


def _modulo() -> ModuleType:
    spec = importlib.util.spec_from_file_location("r_portatil", SCRIPTS / "r_portatil.py")
    assert spec is not None and spec.loader is not None
    modulo = importlib.util.module_from_spec(spec)
    sys.modules["r_portatil"] = modulo
    spec.loader.exec_module(modulo)
    return modulo


rp = _modulo()


def _despejos() -> dict[str, list[str]]:
    arquivos = sorted(DESPEJOS.glob("*.txt"))
    return {a.stem: a.read_text(encoding="utf-8").split() for a in arquivos}


def test_ancora_os_artefatos_versionados_existem_e_nao_sao_vazios() -> None:
    lista = rp.le_lista(LISTA)
    despejos = _despejos()
    assert len(lista) >= 50, f"a lista tem {len(lista)} pacotes; a medição carregou 112"
    assert set(despejos) == set(rp.fluxos()), (
        f"um despejo por fluxo: {sorted(despejos)} contra {sorted(rp.fluxos())}"
    )
    assert all("MetaboAnalystR" in nomes for nomes in despejos.values()), (
        "um despejo sem o MetaboAnalystR mediu o nada"
    )


def test_a_lista_cobre_tudo_o_que_algum_fluxo_carrega() -> None:
    lista = rp.le_lista(LISTA)
    despejos = _despejos()
    assert despejos, "âncora: sem despejos, a cobertura passaria vazia"
    assert rp.faltam_na_lista(lista, despejos) == []
    # par causal: tirar da lista um nome que um despejo registra reprova
    tirado = despejos["ora_local"][0]
    assert rp.faltam_na_lista([n for n in lista if n != tirado], despejos) == [
        f"{f}:{tirado}" for f in sorted(despejos) if tirado in despejos[f]
    ]


def test_nenhum_pacote_proibido_na_lista() -> None:
    lista = rp.le_lista(LISTA)
    assert lista, "âncora: lista vazia passaria"
    assert not set(lista) & set(rp.PROIBIDOS)


def test_o_manifesto_descreve_a_arvore_podada_pela_lista() -> None:
    m = rp.le_manifesto(MANIFESTO)
    lista = rp.le_lista(LISTA)
    assert len(m) > 1000, (
        f"o manifesto tem {len(m)} arquivos; a árvore podada da medição tinha 3.504"
    )
    assert rp.confere_manifesto(m, lista) == []


def test_o_manifesto_reprova_cada_planta() -> None:
    m = rp.le_manifesto(MANIFESTO)
    lista = rp.le_lista(LISTA)
    assert rp.confere_manifesto(m, lista) == [], (
        "a planta só significa algo sobre um manifesto aprovado"
    )
    com_rserve = {**m, "library/Rserve/DESCRIPTION": ("0" * 64, 1)}
    assert any("Rserve" in e for e in rp.confere_manifesto(com_rserve, lista))
    sem_copyrights = {k: v for k, v in m.items() if k != "doc/COPYRIGHTS"}
    assert any("doc/COPYRIGHTS" in e for e in rp.confere_manifesto(sem_copyrights, lista))
    assert any(
        "fora da lista" in e
        for e in rp.confere_manifesto(m, lista[1:])
        + rp.confere_manifesto({**m, "library/xyz/DESCRIPTION": ("0" * 64, 1)}, lista)
    )
    absoluto = {**m, "C:/R/bin/R.exe": ("0" * 64, 1)}
    assert any("não relativo" in e for e in rp.confere_manifesto(absoluto, lista))


def _fixadas(texto: str) -> dict[str, str]:
    achados = {
        "p3m": re.findall(r"packagemanager\.posit\.co/cran/(\d{4}-\d{2}-\d{2})", texto)
        + re.findall(r"REFINO_CRAN_SNAPSHOT_DATE=(\d{4}-\d{2}-\d{2})", texto),
        "bioc": re.findall(r"bioconductor\.org/packages/(3\.\d+)/bioc", texto)
        + [f"3.{m}" for m in re.findall(r"RELEASE_3_(\d+)", texto)],
        "commit": re.findall(r"MetaboAnalystR[@/](?:tar\.gz/)?([0-9a-f]{40})", texto),
    }
    return {
        k: sorted(set(v))[0] if len(set(v)) == 1 else f"AMBÍGUO:{sorted(set(v))}"
        for k, v in achados.items()
        if v
    }


def test_os_tres_pinos_estao_fixados_uma_vez_nos_dois_arquivos_da_montagem() -> None:
    """Os pinos do R portátil: data do P3M e versão do Bioconductor em `r_portatil_instala.R`,
    commit do MetaboAnalystR em `monta_r_portatil.ps1`.

    Juntos, os dois arquivos têm de declarar os três, cada um com um valor só. Um pino trocado num
    lado, ou repetido com outro valor no outro, deixa o valor ambíguo e reprova.
    """
    instala_texto = INSTALA.read_text(encoding="utf-8")
    monta_texto = MONTA.read_text(encoding="utf-8")
    instala = _fixadas(instala_texto)
    monta = _fixadas(monta_texto)
    juntos = _fixadas(instala_texto + "\n" + monta_texto)

    assert set(instala) == {"p3m", "bioc"}, f"a instalação não declara P3M e Bioc: {instala}"
    assert set(monta) == {"commit"}, f"a montagem não declara o commit: {monta}"
    assert set(juntos) == {"p3m", "bioc", "commit"}, f"faltam pinos: {juntos}"
    assert not any(v.startswith("AMBÍGUO") for v in juntos.values()), juntos

    # par causal: o mesmo pino com outro valor no outro arquivo reprova
    plantado = monta_texto + "\n# https://packagemanager.posit.co/cran/2025-01-01\n"
    assert plantado != monta_texto
    assert _fixadas(instala_texto + "\n" + plantado)["p3m"].startswith("AMBÍGUO")
    trocado = instala_texto.replace(instala["p3m"], "2025-01-01")
    assert trocado != instala_texto
    assert _fixadas(trocado)["p3m"] == "2025-01-01"


def test_cada_artefato_baixado_tem_sha256() -> None:
    texto = MONTA.read_text(encoding="utf-8")
    urls = re.findall(r'Url = "([^"]+)"', texto)
    shas = re.findall(r'Sha = "([0-9a-f]{64})"', texto)
    assert len(urls) == 3, urls
    assert len(shas) == len(urls)


def test_reproduzir_aceita_diferenca_so_nos_carimbos_de_build() -> None:
    ref = {
        "bin/R.dll": ("a" * 64, 10),
        "library/MetaboAnalystR/DESCRIPTION": ("b" * 64, 20),
        "library/MetaboAnalystR/libs/x64/MetaboAnalystR.dll": ("c" * 64, 30),
        "library/qs2/DESCRIPTION": ("d" * 64, 40),
    }
    carimbo = {
        **ref,
        "library/MetaboAnalystR/DESCRIPTION": ("e" * 64, 21),
        "library/MetaboAnalystR/libs/x64/MetaboAnalystR.dll": ("f" * 64, 30),
    }
    assert rp.diferencas(carimbo, ref, fora_dos_carimbos=True) == []
    # par causal: o mesmo carimbo reprova sem a licença de carimbo, e uma diferença fora deles
    # reprova sempre
    assert len(rp.diferencas(carimbo, ref, fora_dos_carimbos=False)) == 2
    binario = {**ref, "library/qs2/DESCRIPTION": ("9" * 64, 40)}
    assert rp.diferencas(binario, ref, fora_dos_carimbos=True) == [
        "difere: library/qs2/DESCRIPTION"
    ]
    assert rp.diferencas(
        {k: v for k, v in ref.items() if k != "bin/R.dll"}, ref, fora_dos_carimbos=True
    ) == ["falta: bin/R.dll"]
    # o terceiro desfecho, e a licença de carimbo não o absorve: um arquivo a mais com caminho de
    # carimbo é sobra como qualquer outro — o congelamento conta com ela para a ordem dos passos
    sobra_carimbada = "library/MetaboAnalystR/Meta/package.rds"
    assert rp.e_carimbado(sobra_carimbada), "âncora: a sobra plantada tem caminho de carimbo"
    assert rp.diferencas(
        {**ref, "x": ("0" * 64, 1), sobra_carimbada: ("1" * 64, 2)}, ref, fora_dos_carimbos=True
    ) == [f"sobra: {sobra_carimbada}", "sobra: x"]


def test_o_ativo_de_fonte_e_a_montagem_pinam_o_mesmo_metaboanalystr_e_o_mesmo_r() -> None:
    """A GPL pede a fonte DO BINÁRIO: um pino que derive num lado publica a fonte de outra coisa.

    O commit, a URL e o sha256 do MetaboAnalystR e a versão do R vivem em `monta_r_portatil.ps1`
    (que monta) e em `monta_fonte_gpl.py` (que junta a fonte); este portão os amarra.
    """
    ps1 = (REPO / "scripts" / "monta_r_portatil.ps1").read_text(encoding="utf-8")
    sys.path.insert(0, str(REPO / "scripts"))
    try:
        import monta_fonte_gpl as fonte
    finally:
        sys.path.remove(str(REPO / "scripts"))
    url, nome, sha = fonte.METABO
    bloco = re.search(
        r'Nome = "MetaboAnalystR-[^"]+"\s+Url = "([^"]+)"\s+Sha = "([0-9a-f]{64})"', ps1
    )
    assert bloco is not None, "âncora: o artefato do MetaboAnalystR saiu da lista da montagem"
    assert (bloco.group(1), bloco.group(2)) == (url, sha)
    assert f'Nome = "{nome}"' in ps1
    versoes = set(re.findall(r"R-(\d+\.\d+\.\d+)-win\.exe", ps1))
    assert versoes == {re.search(r"R-(\d+\.\d+\.\d+)\.tar\.gz", fonte.R_FONTE).group(1)}  # type: ignore[union-attr]


def _fonte_gpl() -> ModuleType:
    sys.path.insert(0, str(REPO / "scripts"))
    try:
        import monta_fonte_gpl as fonte
    finally:
        sys.path.remove(str(REPO / "scripts"))
    return fonte


def test_o_leia_me_do_rtools_descreve_os_binarios_desta_arvore() -> None:
    """As versões do leia-me foram medidas sobre quatro binários: se a árvore R mudar, o
    leia-me passa a descrever outro binário, e a mensagem manda remedir."""
    fonte = _fonte_gpl()
    manifesto = (REPO / "scripts" / "r_portatil_arvore.sha256").read_text(encoding="utf-8")
    assert fonte.MEDIDO_SOBRE, "âncora: há binários medidos"
    caducos = fonte.leia_me_caducou(manifesto)
    assert not caducos, (
        f"a árvore R mudou nestes binários: {caducos}. Remeça as versões das bibliotecas no "
        "LEIA_ME_DO_RTOOLS de scripts/monta_fonte_gpl.py e atualize MEDIDO_SOBRE"
    )
    for versao in ("cairo 1.18.2", "cairo 1.18.0", "libidn2 2.3.7", "libgcrypt 1.11.0", "GLPK 5.0"):
        assert versao in fonte.LEIA_ME_DO_RTOOLS, versao
    for binario in fonte.MEDIDO_SOBRE:
        assert binario in fonte.LEIA_ME_DO_RTOOLS, (
            f"o leia-me não nomeia o binário medido {binario}"
        )


def test_o_leia_me_do_rtools_acusa_o_binario_trocado_e_o_sumido() -> None:
    """O par da direção oposta: o mesmo manifesto com um hash trocado, e sem uma linha."""
    fonte = _fonte_gpl()
    manifesto = (REPO / "scripts" / "r_portatil_arvore.sha256").read_text(encoding="utf-8")
    cairo = "library/Cairo/libs/x64/Cairo.dll"
    internet = "modules/x64/internet.dll"
    trocado = manifesto.replace(fonte.MEDIDO_SOBRE[cairo], "0" * 64)
    assert trocado != manifesto, "âncora: a planta aterrou"
    assert fonte.leia_me_caducou(trocado) == [cairo]
    sem_linha = "\n".join(ln for ln in manifesto.splitlines() if not ln.endswith(internet))
    assert fonte.leia_me_caducou(sem_linha) == [internet]
