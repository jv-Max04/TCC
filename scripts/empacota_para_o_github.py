"""Empacota a pasta entregue para o GitHub — SEM o dado clinico do laboratorio.

## O impedimento nao e de tamanho, e foi medido

Medido em 19/09/2026: nenhum
arquivo acima dos 100 MB do bloqueio duro do GitHub — o maior e o `refino.exe`, 34,1 MB.

Com `R/` na pasta, medido sobre `dist/refino` congelado de `afa0377`, o da
`windows-2026.10.02.2`, o maior arquivo passou a ser `R/bin/x64/R.dll`, com 39.333.888
bytes. Depois vem `refino.exe`, com 34.277.029, e `R/library/stringi/libs/icudt74l.dat`, com
30.783.664. Continua nenhum acima dos 100 MB.

**O impedimento e de DADO.** A pasta entregue traz as matrizes corrigidas do
estudo real e o banco com o historico sobre ele. Enviar isso a um servico web de terceiro e
divulgacao de dado clinico. **Que o repositorio seja privado nao muda a natureza do ato.**

## OS DEFEITOS DA PRIMEIRA VERSAO DESTE ARQUIVO

Ela tinha **tres defeitos de classe P0**, cada um capaz de
publicar dado de paciente. A primeira versao foi exercitada nas duas direcoes e mesmo assim
tinha os tres — *exercitado nao e revisado*.

**Os tres P0:**

1. **`ROTULOS` com tres rotulos digitados a mao.** `refino.io.validators.CANONICAL_GROUPS`
   tem **quatro** — faltava `Asymptomatic`. Medido: 18 dos 68 pacientes do estudo sao
   Asymptomatic, e um recorte so com eles passava limpo.
2. **Exclusao por NOME EXATO.** `refino.db-wal`, `-journal` e `-shm` nao sao `refino.db` e
   entravam. Um revisor achou `PAC-0043` e `ATLL`
   **crus** dentro de um `-wal` real. Pior: a lista tinha **dois** itens e a irma,
   o `$paraTirar` de `prepara_entrega.ps1`, tem **cinco** — faltava o `refino.log`, e ele
   ESTAVA dentro da release publicada. A lista mais curta estava no destino mais exposto.
3. **`except OSError: continue`.** Arquivo ilegivel contava como **limpo** e entrava no zip
   na mesma, sob a mensagem de sucesso.

**E tres defeitos de desenho por baixo deles:** era uma **denylist**, e o Qt abre os dialogos
de exportacao sem diretorio inicial — o `cwd` do executavel e `dist/refino`, entao uma figura
ou CSV do estudo exportado pela janela cai na raiz da pasta e viaja; a isencao era por
**extensao**, dispensando 3.155 dos 4.770 arquivos sob uma justificativa que descrevia outra
populacao; e a reconferencia do zip lia **nomes**, quando afirmava conferir o
conteudo escrito.

## O DESENHO DE AGORA: allowlist primeiro, conteudo depois, e o zip reconferido por BYTES

**1. Allowlist.** `PODEM_SUBIR` e a lista do `prepara_entrega.ps1` menos `saidas`. Nome de
primeiro nivel fora dela e fora de `PRIVADOS` **aborta** — nao e silenciosamente descartado.
Foi essa a metade que faltava: nenhuma lista de proibidos antecipa o que o aplicativo escreve.

**2. Privados saem por PREFIXO**, nao por igualdade, e sem distinguir maiusculas — o que cobre
os tres sidecars do SQLite sem os enumerar, e um diretorio `Saidas` criado a mao.

**3. O conteudo e sempre lido.** A isencao passou a ser por CAMINHO e nao por extensao, e tem
duas entradas, cada uma com motivo conferivel: os documentos de entrega, que trazem os rotulos
e vivem versionados em `docs/`; e `_internal/refino/`, que e o `src/` deste repositorio
compilado — e que essa segunda isencao nao basta sozinha esta na docstring de
`ISENTOS_DE_CONTEUDO`. Tudo o resto e aberto e lido. **Erro de leitura
ABORTA** — ilegivel nunca conta como limpo.

**4. O zip e reconferido por BYTES**, com o mesmo oraculo, sobre o arquivo temporario, antes de
ele virar o definitivo. A escrita e atomica: `.tmp` e `replace()` so no fim.

## O R EMBARCADO E AS BIBLIOTECAS DAS VIAS

**`R/` sobe, e o conteudo dele e julgado pelo MANIFESTO e nao pelos rotulos.** A varredura de
rotulos sobre a arvore real achou `AtlL` em base64 de certificado e em bytes comprimidos — falso
positivo que pararia toda publicacao. A isencao e condicionada: cada arquivo de `R/` tem de bater,
por sha256 e tamanho, com `scripts/r_portatil_arvore.sha256`; arquivo fora do manifesto, ou
diferente dele, ABORTA. Os poucos arquivos com carimbo de build (`DESCRIPTION`, `Meta/package.rds`
e as DLLs dos dois pacotes compilados da fonte) que diferirem sao LIDOS como qualquer outro.

**`vias-dados/` NAO sobe, por licenca, e nao aborta por existir.** E a terceira lista, ao lado da
allowlist e dos privados, casada pelo mesmo predicado nas duas passadas.

**Pasta com `R/` sem o ativo de fonte ao lado ABORTA.** A GPL pede a fonte junto dos binarios que
nos compilamos; o ativo e montado por `scripts/monta_fonte_gpl.py` e tem de ser o DESTA arvore — a
lista de pacotes e o manifesto dentro dele sao comparados com os versionados.
"""

from __future__ import annotations

import hashlib
import pathlib
import re
import sys
import zipfile
from typing import Literal

sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]

RAIZ = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "src"))

sys.path.insert(0, str(RAIZ / "scripts"))

from declara_charset import declara as declara_utf8  # noqa: E402
from monta_fonte_gpl import NOME_DO_ATIVO  # noqa: E402
from r_portatil import e_carimbado, le_manifesto  # noqa: E402

from refino.desktop.vias_r import PASTA_DAS_BIBLIOTECAS, PASTA_DO_R, ROTEIRO  # noqa: E402
from refino.io.validators import CANONICAL_GROUPS  # noqa: E402

DIST = RAIZ / "dist" / "refino"

# `PASTA_DO_R`, `ROTEIRO` e `PASTA_DAS_BIBLIOTECAS` vem de `refino.desktop.vias_r`, que e quem o
# executavel le: renomear la e aqui ao mesmo tempo e o que impede a constante lida por dois
# mecanismos de divergir.
MANIFESTO_DA_ARVORE_R = RAIZ / "scripts" / "r_portatil_arvore.sha256"
LISTA_DE_PACOTES_DO_R = RAIZ / "scripts" / "r_portatil_pacotes.txt"
ROTEIRO_DAS_VIAS = RAIZ / "r_service" / "vias_entrypoint.R"

PODEM_SUBIR = frozenset(
    {
        "refino.exe",
        "_internal",
        "LEIA-ME.txt",
        "GUIA.html",
        "ENTENDER.html",
        "MEDICOES.html",
        "MULTIVERSO.html",
        "EXPLICAR.html",
        "FAQ.html",
        "DIAGNOSTICO.html",
        "VERSAO.txt",
        "R",
    }
)
"""Os nomes de primeiro nivel que podem entrar no zip — e **nada mais sobe sem decisao**.

E o `$temQueFicar` de `scripts/prepara_entrega.ps1` **menos `saidas`**, que fica de fora
por levar medicao por amostra, **e menos `vias-dados`**, que fica de fora por licenca
(`FORA_POR_LICENCA`). As duas divergencias sao
deliberadas e sao as unicas:
`prepara_entrega.ps1` monta a pasta que o laboratorio recebe em mao, e la as duas viajam.
"""

FORA_POR_LICENCA = (PASTA_DAS_BIBLIOTECAS,)
"""Viaja na pasta do laboratorio e NAO sobe, casado por PREFIXO como os privados.

As bibliotecas de dados das vias vem do servidor do MetaboAnalyst e derivam do HMDB e da
SMPDB/PathBank; a licenca de redistribuicao nao foi conferida na fonte. A release publica as
baixa com confirmacao do usuario. **Fica fora sem abortar**, ao contrario de um nome
desconhecido: a pasta e esperada, e o que se decide e que ela nao sobe.
"""

PRIVADOS = ("saidas", "refino.db", "refino.log")
"""O que nunca sobe, casado por PREFIXO do primeiro componente.

E o `$paraTirar` de `scripts/prepara_entrega.ps1` colapsado: os cinco nomes de la
— `refino.db`, `-journal`, `-wal`, `-shm` e `refino.log` — sao estes tres prefixos. Prefixo e
nao igualdade porque foi exactamente assim que os tres sidecars do SQLite escaparam.
"""

ROTULOS = re.compile(
    "|".join(sorted((re.escape(g) for g in CANONICAL_GROUPS), key=len, reverse=True))
    .replace("HAMTSP", "HAM[_/\\- ]?TSP")
    .encode("ascii"),
    re.I,
)
"""Os rotulos de diagnostico, DERIVADOS da fonte unica em vez de digitados.

`refino.io.validators.CANONICAL_GROUPS` e quem define o vocabulario, e a primeira versao deste
arquivo retipou **tres** dos quatro membros. Agora deriva, e
`tests/test_empacota_para_o_github.py` confere a bijecao nas duas direcoes para o
caso de alguem mudar a forma de um membro sem mexer aqui.

A tolerancia de separador em `HAM/TSP` fica porque o corpus real traz as tres formas medidas em
19/09/2026: `HAMTSP` 85, `HAM/TSP` 26, `HAM-TSP` 18.
"""

ISENTOS_DE_CONTEUDO = (
    "GUIA.html",
    "ENTENDER.html",
    "MEDICOES.html",
    "MULTIVERSO.html",
    "EXPLICAR.html",
    "FAQ.html",
    "DIAGNOSTICO.html",
    "LEIA-ME.txt",
    "_internal/refino/",
)
"""Os caminhos cujo conteudo nao precisa de ser julgado, e o motivo de cada um.

**Isencao por CAMINHO e nao por extensao**, que foi o defeito da primeira versao: `.pyc`
dispensava 3.108 arquivos de `scipy`/`pandas`/`sklearn` sob uma justificativa que falava do
`src/` deste repositorio.

- **Os oito documentos de entrega** trazem `ATLL`, `HAM/TSP` e `NoHTLV` na prosa, e vivem
  versionados em `docs/`. Medido em 19/09/2026: os tres rotulos aparecem em **139 arquivos ja
  rastreados no git**, entre eles a especificacao e o `README.md`.
- **`_internal/refino/`** e o `src/` deste repositorio compilado. Um `.pyc` nao divulga nada que
  o `.py` ao lado ja nao divulgue.
  **Esta isencao nao basta sozinha.** Por ela um nome de amostra real, na
  *docstring* de `plots/agrupamento.py`, viajou nas *releases* sem que este arquivo o visse; e,
  depois de uma troca parcial, o numero e o grupo da mesma amostra, que as
  *releases* anteriores ainda levam no codigo empacotado. O oraculo daqui sao os rotulos, que o
  `src/` traz legitimamente (`CANONICAL_GROUPS`), entao ler `_internal/refino/` nao resolveria.
  Quem guarda o `src/` e `tests/test_sem_amostra_real_versionada.py`, que salta
  num clone sem os `Sample_List`.

Tudo o resto e lido — inclusive `refino.exe`, `VERSAO.txt` e as bibliotecas de terceiros.
"""


def _primeiro(rel: pathlib.PurePath) -> str:
    return rel.parts[0].lower() if rel.parts else ""


def _casa_por_prefixo(rel: pathlib.PurePath, nomes: tuple[str, ...]) -> bool:
    """A regra de prefixo das duas listas que tiram nome do zip — uma so, para nao divergirem."""
    primeiro = _primeiro(rel)
    return any(primeiro == p or primeiro.startswith(p + "-") for p in nomes)


def e_privado(rel: pathlib.PurePath) -> bool:
    """O caminho cai num dos artefatos de privacidade?

    Prefixo e minusculas de proposito: `refino.db-wal` e `Saidas/` passavam pela igualdade
    sensivel a maiusculas da primeira versao. **As duas passadas — a que filtra e a que
    reconfere o zip — chamam ESTA funcao**, para que endurecer uma nao possa afrouxar a outra.
    """
    return _casa_por_prefixo(rel, PRIVADOS)


def e_fora_por_licenca(rel: pathlib.PurePath) -> bool:
    """O caminho cai numa pasta que viaja ao laboratorio e nao sobe? Ver `FORA_POR_LICENCA`.

    Chamada pelas DUAS passadas, como `e_privado`, pelo mesmo motivo.
    """
    return _casa_por_prefixo(rel, FORA_POR_LICENCA)


def na_arvore_r(rel: pathlib.PurePath) -> bool:
    return bool(rel.parts) and rel.parts[0] == PASTA_DO_R


def dentro_da_arvore_r(rel: pathlib.PurePath) -> str:
    """O caminho relativo a `R/`, na forma do manifesto."""
    return pathlib.PurePosixPath(*rel.parts[1:]).as_posix()


DecisaoNaArvoreR = Literal["isento", "ler", "aborta"]


def julga_na_arvore_r(
    rel: pathlib.PurePath, bruto: bytes, manifesto: dict[str, tuple[str, int]]
) -> tuple[DecisaoNaArvoreR, str]:
    """`("isento", "")`, `("ler", "")` ou `("aborta", motivo)` para um caminho sob `R/`.

    Isento so quando os bytes sao os da montagem versionada; carimbo de build que difere e lido
    como qualquer arquivo; o resto aborta. O roteiro das vias, que nao e da montagem, e isento
    quando e o do repositorio byte a byte. **As duas passadas chamam ESTA funcao.**
    """
    dentro = dentro_da_arvore_r(rel)
    if dentro == ROTEIRO:
        if bruto == ROTEIRO_DAS_VIAS.read_bytes():
            return "isento", ""
        return "aborta", "o roteiro das vias nao e o do repositorio"
    esperado = manifesto.get(dentro)
    if esperado is None:
        return "aborta", "fora do manifesto da arvore R"
    if (hashlib.sha256(bruto).hexdigest(), len(bruto)) == esperado:
        return "isento", ""
    if e_carimbado(dentro):
        return "ler", ""
    return "aborta", "difere do manifesto da arvore R"


def confere_ativo_de_fonte(ativo: pathlib.Path) -> str | None:
    """`None` quando o ativo de fonte existe e e o DESTA arvore; o motivo quando nao."""
    if not ativo.is_file():
        return f"{ativo.name} nao existe em {ativo.parent}"
    try:
        with zipfile.ZipFile(ativo) as z:
            nomes = set(z.namelist())
            if "FONTES.sha256" not in nomes:
                return f"{ativo.name} nao traz FONTES.sha256"
            for nome, local in (
                ("roteiros/r_portatil_pacotes.txt", LISTA_DE_PACOTES_DO_R),
                ("roteiros/r_portatil_arvore.sha256", MANIFESTO_DA_ARVORE_R),
            ):
                if nome not in nomes:
                    return f"{ativo.name} nao traz {nome}"
                # Fim de linha normalizado: o ativo e um zip congelado, e o checkout pode trocar
                # o LF do arquivo versionado por CRLF sem que o conteudo mude.
                if z.read(nome).replace(b"\r\n", b"\n") != local.read_bytes().replace(
                    b"\r\n", b"\n"
                ):
                    return f"{nome} dentro de {ativo.name} nao e o desta arvore"
    except (OSError, zipfile.BadZipFile) as erro:
        return f"{ativo.name} ilegivel: {erro.__class__.__name__}"
    return None


def e_isento_de_conteudo(rel: pathlib.PurePath) -> bool:
    """O conteudo deste caminho dispensa julgamento? Ver `ISENTOS_DE_CONTEUDO`."""
    como_texto = rel.as_posix()
    return any(
        como_texto == isento or como_texto.startswith(isento) for isento in ISENTOS_DE_CONTEUDO
    )


def marcas(bruto: bytes) -> list[str]:
    """Os rotulos de diagnostico presentes nestes bytes, ordenados e sem repeticao."""
    return sorted({m.decode("latin-1") for m in ROTULOS.findall(bruto)})


def main() -> int:
    if not DIST.is_dir():
        print(f"{DIST} nao existe — congele antes")
        return 1

    # **A nao-vacuidade vem ANTES de tudo, e a ordem foi um defeito que um teste apanhou.**
    # A primeira escrita desta funcao lia o `VERSAO.txt` primeiro, entao uma pasta vazia
    # rebentava com `FileNotFoundError` em vez de dizer o que estava errado.
    todos = [p for p in DIST.rglob("*") if p.is_file()]
    if not todos:
        print(f"ABORTADO: {DIST} esta vazia — nao ha o que empacotar")
        return 1

    marca = DIST / "VERSAO.txt"
    if not marca.is_file():
        print(f"ABORTADO: {marca} nao existe — a pasta nao e um congelamento completo")
        return 1
    versao = marca.read_text(encoding="utf-8", errors="replace")
    commit = next(
        (ln.split(":", 1)[1].strip() for ln in versao.splitlines() if ln.startswith("commit")), ""
    )
    print(f"pasta:  {DIST}")
    print(f"commit: {commit or 'DESCONHECIDO'}")

    # ---------------------------------------------------------------- 1. a allowlist
    entram, privados, por_licenca, desconhecidos = [], 0, 0, set()
    for p in todos:
        rel = p.relative_to(DIST)
        if e_privado(rel):
            privados += 1
            continue
        if e_fora_por_licenca(rel):
            por_licenca += 1
            continue
        if rel.parts[0] not in PODEM_SUBIR:
            desconhecidos.add(rel.parts[0])
            continue
        entram.append(p)

    print(f"\narquivos na pasta: {len(todos):,}")
    print(f"  artefatos de privacidade, fora ({', '.join(PRIVADOS)}): {privados}")
    print(f"  fora por licenca ({', '.join(FORA_POR_LICENCA)}): {por_licenca}")
    print(
        f"  entram no zip: {len(entram):,}  ({sum(p.stat().st_size for p in entram) / 1e6:,.1f} MB)"
    )

    if desconhecidos:
        print(f"\nABORTADO: {len(desconhecidos)} nome(s) de primeiro nivel fora da allowlist:")
        for nome in sorted(desconhecidos):
            print(f"   {nome}")
        print("\n  O aplicativo escreve na pasta de onde corre. Se isto e saida de estudo,")
        print("  mova-a; se e artefato novo da entrega, acrescente-a a PODEM_SUBIR.")
        return 1

    # ---------------------------------------------------------------- 1b. o R e a fonte dele
    com_r = any(na_arvore_r(p.relative_to(DIST)) for p in entram)
    manifesto_r: dict[str, tuple[str, int]] = {}
    if com_r:
        motivo = confere_ativo_de_fonte(RAIZ / "dist" / NOME_DO_ATIVO)
        if motivo is not None:
            print(
                f"\nABORTADO: a pasta leva o R embarcado e o ativo de fonte da GPL falta: {motivo}"
            )
            print("  monte-o com: python scripts/monta_fonte_gpl.py --destino dist")
            return 1
        manifesto_r = le_manifesto(MANIFESTO_DA_ARVORE_R)

    # ---------------------------------------------------------------- 2. o conteudo
    lidos = 0
    isentos_por_manifesto = 0
    vistos_na_arvore_r: set[str] = set()
    achados: list[tuple[str, list[str]]] = []
    for p in entram:
        rel = p.relative_to(DIST)
        if e_isento_de_conteudo(rel):
            continue
        try:
            bruto = p.read_bytes()
        except OSError as erro:
            # **Ilegivel ABORTA.** A primeira versao fazia `continue`, e um arquivo que a
            # varredura nao conseguia abrir contava como limpo — e era escrito no zip na
            # mesma, porque a escrita o le por outro caminho.
            achados.append((rel.as_posix(), [f"ILEGIVEL: {erro.__class__.__name__}"]))
            continue
        if na_arvore_r(rel):
            vistos_na_arvore_r.add(dentro_da_arvore_r(rel))
            decisao, motivo = julga_na_arvore_r(rel, bruto, manifesto_r)
            if decisao == "aborta":
                achados.append((rel.as_posix(), [motivo]))
                continue
            if decisao == "isento":
                isentos_por_manifesto += 1
                continue
        lidos += 1
        if encontrados := marcas(bruto):
            achados.append((rel.as_posix(), encontrados))

    if com_r:
        # Arvore incompleta subiria um R que nao arranca, e nada no julgamento por arquivo o ve.
        for falta in sorted(set(manifesto_r) - vistos_na_arvore_r):
            achados.append((f"{PASTA_DO_R}/{falta}", ["falta na arvore R"]))
    print(
        f"\nlidos {lidos:,} de {len(entram):,} ({lidos / len(entram):.0%}); "
        f"{isentos_por_manifesto:,} da arvore R conferidos pelo manifesto; "
        f"o resto e isento por caminho"
    )
    if achados:
        print(f"\nABORTADO: {len(achados)} arquivo(s) levariam medicao por amostra:")
        for nome, marcados in achados[:20]:
            print(f"   {nome}  ->  {marcados}")
        return 1
    print("  nenhum arquivo lido traz rotulo de diagnostico")

    # ---------------------------------------------------------------- 3. escrita ATOMICA
    alvo = RAIZ / "dist" / f"refino-windows-{(commit or 'sem-commit')[:7]}.zip"
    temporario = alvo.with_suffix(".zip.parcial")
    print(f"\nescrevendo {alvo.name} ...")
    try:
        with zipfile.ZipFile(temporario, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
            for p in entram:
                z.write(p, pathlib.Path("refino") / p.relative_to(DIST))

        # **A reconferencia le os BYTES ARMAZENADOS, e nao a lista que os produziu.** A
        # primeira versao conferia so nomes, e afirmava que conferia a varredura.
        # Sao duas instancias do mesmo referente, e a que viaja e esta.
        intrusos: list[str] = []
        relidos = 0
        with zipfile.ZipFile(temporario) as z:
            nomes = z.namelist()
            for nome in nomes:
                if nome.endswith("/"):
                    continue
                dentro = pathlib.PurePosixPath(nome).relative_to("refino")
                if e_privado(dentro):
                    intrusos.append(f"{nome} -> artefato de privacidade")
                    continue
                if e_fora_por_licenca(dentro):
                    intrusos.append(f"{nome} -> fora por licenca")
                    continue
                if e_isento_de_conteudo(dentro):
                    continue
                guardado = z.read(nome)
                if na_arvore_r(dentro):
                    decisao, motivo = julga_na_arvore_r(dentro, guardado, manifesto_r)
                    if decisao == "aborta":
                        intrusos.append(f"{nome} -> {motivo}")
                        continue
                    if decisao == "isento":
                        continue
                relidos += 1
                if encontrados := marcas(guardado):
                    intrusos.append(f"{nome} -> {encontrados}")

            # **A INSTANCIA QUE VIAJA, e ela nao e a que o portao de testes le.**
            # `tests/test_charset_dos_documentos.py` confere `docs/*.html`; o que o usuario
            # abre por duplo clique sao estas copias, com nomes diferentes. Este e o unico
            # ponto do fluxo que ve o artefato publicado.
            # **A arvore R fica fora desta conferencia**, e o motivo e o mesmo da isencao de
            # rotulos: os documentos dela sao do R e dos pacotes, julgados pelo manifesto, e nao sao
            # o que o usuario abre por duplo clique. Medido no primeiro empacotamento com o R:
            # `timeDate/COPYRIGHT.html` e `grDevices/afm/MustRead.html` nao declaram charset.
            sem_charset = [
                nome
                for nome in nomes
                if nome.lower().endswith(".html")
                and not na_arvore_r(pathlib.PurePosixPath(nome).relative_to("refino"))
                and declara_utf8(z.read(nome)) < 0
            ]
            if sem_charset:
                intrusos.extend(f"{nome} -> nao declara UTF-8" for nome in sem_charset)

        print(f"\n{len(nomes):,} entradas no zip, {temporario.stat().st_size / 1e6:,.1f} MB")
        print(f"reconferidos {relidos:,} por CONTEUDO dentro do proprio zip")
        if intrusos:
            print(f"ABORTADO: {len(intrusos)} entrada(s) indevida(s): {intrusos[:5]}")
            temporario.unlink(missing_ok=True)
            return 1
    except BaseException:
        # **O parcial nunca sobrevive a uma falha.** A primeira versao escrevia direto no nome
        # final, entao um erro a meio deixava em disco um zip truncado com o nome do bom.
        temporario.unlink(missing_ok=True)
        raise

    temporario.replace(alvo)
    print(f"\n{alvo}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
