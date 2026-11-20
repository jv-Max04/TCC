#!/usr/bin/env python3
"""Monta o ativo de fonte que acompanha toda release que leva o R embarcado.

O executável passa a redistribuir binários GPL compilados por nós — o `MetaboAnalystR.dll` contém
código só GPL-2 —, e a seção 3 da GPLv2 pede a fonte junto. Este roteiro monta um zip com:

- a fonte do R 4.4.3 (`cran.r-project.org/src/base/R-4/`);
- o tarball de fonte de **cada pacote da árvore**, na versão fixada em
  `scripts/r_portatil_pacotes.txt`, do mesmo instantâneo que a montagem usa (CRAN no P3M de
  2025-06-02, Bioconductor 3.20); os pacotes que fazem parte do R vêm na fonte do R;
- o tarball do MetaboAnalystR no commit fixado, conferido pelo sha256 da montagem;
- os roteiros de build (`scripts/monta_r_portatil.ps1` e os que ele chama);
- o `LEIA-ME-BIBLIOTECAS-DO-RTOOLS.txt`, que nomeia as bibliotecas de terceiros ligadas DENTRO dos
  binários do R e diz onde está a fonte delas;
- um `FONTES.sha256` com o hash de cada arquivo.

    .venv/Scripts/python.exe scripts/monta_fonte_gpl.py --destino <pasta> [--cache <pasta>]

O empacotador (`scripts/empacota_para_o_github.py`) recusa publicar uma pasta com o R embarcado sem
este ativo ao lado. Os downloads ficam num cache e não se repetem.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path
from typing import Final

AQUI = Path(__file__).resolve().parent
PACOTES = AQUI / "r_portatil_pacotes.txt"
ROTEIROS: Final = (
    "monta_r_portatil.ps1",
    "r_portatil.py",
    "r_portatil_instala.R",
    "r_portatil_despeja.R",
    "r_portatil_pacotes.txt",
    "r_portatil_arvore.sha256",
    "r_portatil_licencas.tsv",
)
R_FONTE: Final = "https://cran.r-project.org/src/base/R-4/R-4.4.3.tar.gz"
METABO: Final = (
    "https://codeload.github.com/xia-lab/MetaboAnalystR/tar.gz/6bf5462064cd1fbc704bf4fb4abdf430723198ac",
    "MetaboAnalystR-6bf5462.tar.gz",
    "5ff9a18203ef7acbc45882106144f9474f8aea3b2b64433f2fe63d5ba8e70d50",
)
REPOSITORIOS: Final = (
    "https://packagemanager.posit.co/cran/2025-06-02/src/contrib/",
    "https://packagemanager.posit.co/cran/2025-06-02/src/contrib/Archive/{pacote}/",
    "https://bioconductor.org/packages/3.20/bioc/src/contrib/",
    "https://bioconductor.org/packages/3.20/bioc/src/contrib/Archive/{pacote}/",
    "https://bioconductor.org/packages/3.20/data/annotation/src/contrib/",
)
NOME_DO_ATIVO: Final = "refino-fonte-gpl-do-R-embarcado.zip"

# **As bibliotecas que chegam LIGADAS dentro dos binários do R e dos pacotes, e cuja fonte não está
# nos tarballs acima**. Não são código nosso nem dos pacotes: vêm do Rtools44, a ferramenta
# com que o R e os pacotes são compilados para Windows. A fonte delas é APONTADA, e não
# incluída. A primeira versão deste texto nomeava duas
# bibliotecas, e a reverificação da release achou sete a mais em dois binários: o texto agora
# lista o que a medição alcançou, binário a binário, e aponta as receitas do Rtools44 como a fonte
# de TODAS — inclusive das que nenhuma medição por cadeia alcança. Os dois endereços foram abertos
# em 02/10/2026.
LEIA_ME_DO_RTOOLS: Final = """\
Bibliotecas de terceiros dentro dos binários do R embarcado
===========================================================

Os binários do R para Windows, e os de alguns pacotes, trazem ligadas por dentro
bibliotecas de terceiros compiladas com o Rtools44, a ferramenta com que o R e os
pacotes são compilados para Windows — o R pela CRAN, e os pacotes Cairo e igraph
desta árvore pelo P3M. Várias estão sob licenças copyleft, como a LGPL e a GPL. A
fonte delas NÃO está neste zip.

ONDE ESTÁ A FONTE, de todas elas. A página do Rtools44,
  https://cran.r-project.org/bin/windows/Rtools/rtools44/rtools.html
aponta as fontes dos "toolchain tarballs" em
  https://svn.r-project.org/R-dev-web/trunk/WindowsBuilds/winutf8/ucrt3/toolchain_libs/
com as receitas de compilação em mxe/src. A receita do cairo, por exemplo
(mxe/src/cairo.mk), nomeia a versão, o arquivo e o endereço de origem; a versão que
ela nomeia hoje é mais nova que as destes binários, e as anteriores estão no
histórico do repositório — os binários abaixo não vêm todos da mesma revisão do
Rtools44, e o cairo aparece neles em duas versões. O arquivo R/doc/COPYRIGHTS da
pasta R trata da distribuição binária do R para Windows na seção "Binary
components".

O QUE FOI MEDIDO em 02/10/2026, em quatro binários desta pasta R — os que trazem
bibliotecas copyleft que a verificação procurou —, por cadeias que o binário guarda: o
caminho de compilação, a cadeia de versão ou, onde está escrito "sem versão gravada",
nomes próprios da biblioteca — os apelidos de codificação do libiconv, como
CN-GB-ISOIR165, e as variáveis GETTEXT_ do gettext. A LISTA É PARCIAL: estes quatro
binários, e outros da pasta R, trazem ainda outras bibliotecas do Rtools — como libpng,
zlib, bzip2, libxml2, PCRE2 e ICU —, e a fonte de todas está no endereço acima.

  R/library/grDevices/libs/x64/winCairo.dll   (do próprio R)
      cairo 1.18.2, pixman 0.44.2, harfbuzz 10.2.0, expat 2.6.4;
      gettext (libintl) e libiconv, sem versão gravada
  R/library/Cairo/libs/x64/Cairo.dll          (pacote Cairo)
      cairo 1.18.0, pixman 0.43.2, harfbuzz 8.3.0, expat 2.6.0, libtiff (4.5.1 ou
      posterior: o binário só cita essa versão numa mensagem);
      gettext (libintl) e libiconv, sem versão gravada
  R/modules/x64/internet.dll                  (do próprio R)
      curl 8.11.1, libidn2 2.3.7, libunistring 1.3, libiconv 1.18,
      libgcrypt 1.11.0, libgpg-error 1.51, libpsl 0.21.5, libssh2 1.11.1,
      OpenSSL 3.4.0
  R/library/igraph/libs/x64/igraph.dll        (pacote igraph)
      GLPK 5.0, da árvore do Rtools e não da cópia que vem no tarball do igraph;
      libiconv, sem versão gravada

Os binários ligam ainda as bibliotecas de execução do compilador GCC.
"""
NOME_DO_LEIA_ME_DO_RTOOLS: Final = "LEIA-ME-BIBLIOTECAS-DO-RTOOLS.txt"
# Os binários sobre os quais o inventário acima foi medido, com o sha256 que eles têm no manifesto
# versionado da árvore (`r_portatil_arvore.sha256`). Se a árvore mudar, o leia-me descreve outro
# binário, e `leia_me_caducou` o diz — «remeça», e não «está errado».
MEDIDO_SOBRE: Final = {
    "library/grDevices/libs/x64/winCairo.dll": (
        "7ffb55d53bad8703d1b2553fcdd063c27675311c1591a9b661276d1738500b3c"
    ),
    "library/Cairo/libs/x64/Cairo.dll": (
        "4ccef6f363ada2c16022c180f07fcae5147e3567748f24158bff58c445be64db"
    ),
    "modules/x64/internet.dll": "972a1851f4ace8609f6d0817f8dc09d0bc912ef51741c0970b1bfbd289df7b40",
    "library/igraph/libs/x64/igraph.dll": (
        "7d0e0a5dfc3a1f808cc16aa6a1e4d175d8d89da45566115456a8970be184fe58"
    ),
}


def leia_me_caducou(manifesto: str) -> list[str]:
    """Os binários de `MEDIDO_SOBRE` cujo sha256 no `manifesto` não é o medido, ou que sumiram."""
    no_manifesto = {}
    for linha in manifesto.splitlines():
        partes = linha.split(maxsplit=2)
        if len(partes) == 3:
            no_manifesto[partes[2]] = partes[0]
    return [b for b, sha in MEDIDO_SOBRE.items() if no_manifesto.get(b) != sha]


def _sha256(caminho: Path) -> str:
    with caminho.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def _baixa(url: str, destino: Path) -> bool:
    if destino.is_file() and destino.stat().st_size > 0:
        return True
    temporario = destino.with_suffix(destino.suffix + ".parcial")
    try:
        # O Bioconductor responde 403 ao `User-Agent` padrão do `urllib` (medido em 01/10/2026); o
        # roteiro se identifica.
        pedido = urllib.request.Request(url, headers={"User-Agent": "refino-monta-fonte-gpl/1"})
        with urllib.request.urlopen(pedido, timeout=300) as resposta, temporario.open("wb") as f:
            shutil.copyfileobj(resposta, f)
    except OSError:
        temporario.unlink(missing_ok=True)
        return False
    temporario.replace(destino)
    return True


def pacotes_de_fora_do_r(lista: Path) -> list[tuple[str, str]]:
    """Os pacotes da árvore que não fazem parte do R: os que têm fonte própria a buscar."""
    da_base = {
        "base",
        "compiler",
        "datasets",
        "graphics",
        "grDevices",
        "grid",
        "methods",
        "parallel",
        "splines",
        "stats",
        "stats4",
        "tcltk",
        "tools",
        "utils",
    }
    saida = []
    for linha in lista.read_text(encoding="utf-8").splitlines():
        if linha.strip():
            nome, versao = linha.split()[:2]
            if nome not in da_base and nome != "MetaboAnalystR":
                saida.append((nome, versao))
    return saida


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--destino", required=True)
    ap.add_argument("--cache", default=str(Path.home() / "_refino_env" / "fonte-gpl-cache"))
    a = ap.parse_args()
    cache = Path(a.cache)
    cache.mkdir(parents=True, exist_ok=True)
    faltas: list[str] = []
    arquivos: list[Path] = []

    r = cache / Path(R_FONTE).name
    if not _baixa(R_FONTE, r):
        faltas.append(R_FONTE)
    arquivos.append(r)

    url, nome, sha = METABO
    metabo = cache / nome
    if not _baixa(url, metabo) or _sha256(metabo) != sha:
        faltas.append(f"{nome} (sha256 não confere)")
    arquivos.append(metabo)

    for pacote, versao in pacotes_de_fora_do_r(PACOTES):
        tarball = cache / f"{pacote}_{versao}.tar.gz"
        if not any(
            _baixa(base.format(pacote=pacote) + tarball.name, tarball) for base in REPOSITORIOS
        ):
            faltas.append(tarball.name)
        arquivos.append(tarball)

    if faltas:
        for f in faltas:
            print(f"FALTA: {f}")
        print(f"o ativo de fonte não foi montado: {len(faltas)} fonte(s) não obtida(s)")
        return 1

    destino = Path(a.destino)
    destino.mkdir(parents=True, exist_ok=True)
    ativo = destino / NOME_DO_ATIVO
    linhas_sha = []
    with zipfile.ZipFile(ativo, "w", compression=zipfile.ZIP_STORED) as z:
        for arq in arquivos:
            z.write(arq, f"fontes/{arq.name}")
            linhas_sha.append(f"{_sha256(arq)}  fontes/{arq.name}")
        for nome_roteiro in ROTEIROS:
            z.write(AQUI / nome_roteiro, f"roteiros/{nome_roteiro}")
            linhas_sha.append(f"{_sha256(AQUI / nome_roteiro)}  roteiros/{nome_roteiro}")
        leia_me = LEIA_ME_DO_RTOOLS.encode("utf-8")
        z.writestr(NOME_DO_LEIA_ME_DO_RTOOLS, leia_me)
        linhas_sha.append(f"{hashlib.sha256(leia_me).hexdigest()}  {NOME_DO_LEIA_ME_DO_RTOOLS}")
        z.writestr("FONTES.sha256", "\n".join(linhas_sha) + "\n")
    # «fontes» e não «pacotes»: uma delas é a do próprio R, que cobre os pacotes da base.
    print(
        f"ativo de fonte: {ativo} ({ativo.stat().st_size} bytes, {len(arquivos)} fontes: "
        f"a do R e {len(arquivos) - 1} pacotes)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
