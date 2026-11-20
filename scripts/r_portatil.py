#!/usr/bin/env python3
"""O que a montagem do R portátil faz em Python: derivar a lista de pacotes, podar, e o manifesto.

Chamado por `scripts/monta_r_portatil.ps1`, um subcomando por passo. Fica em Python e não no
PowerShell porque as três coisas são cópia e comparação de árvore com ~3.500 arquivos, e o portão
das crases (`tests/test_powershell_crases.py`) já mostrou o custo de prosa longa em `.ps1`.

## `deriva` — a lista de pacotes sai do que os fluxos CARREGAM, e não do grafo declarado

A união de `loadedNamespaces()` depois de correr os fluxos do modo local sobre a árvore COMPLETA. O
grafo de dependências declarado tem ~170 pacotes e os fluxos carregam ~110: a diferença é `BH` (123
MiB de cabeçalhos), `rmarkdown`, `tcltk` e o resto que o pacote declara e o caminho que usamos não
toca. Um despejo por fluxo fica versionado em `scripts/r_portatil_namespaces/`, e
`tests/test_r_portatil_roteiro.py` reprova se a lista perder um nome que algum despejo registra.

**Os fluxos são os do modo LOCAL, e isso é decisão**: o pedido do modo externo é feito em Python
(`src/refino/io/enriquecimento_externo.py`), o roteiro R não fala com a rede, e a árvore não leva
`curl`, `openssl` nem `mime`. O `httr` está na lista-base porque os fluxos locais o carregam — os
despejos de `scripts/r_portatil_namespaces/` o trazem. Um caminho de erro está
entre os fluxos — mapear com zero casamentos — porque pacote carregado só num ramo de erro sumiria
na poda e o ramo morreria no laboratório.

## `poda` — o que sai, e o que fica de propósito

Sai: `doc/` do topo **menos `doc/COPYRIGHTS`**, `include/`, `src/`, `Tcl/`, `tests/`, o
desinstalador e o `MD5`; todo pacote fora da lista; e, dentro de cada pacote, `help/`, `html/`,
`doc/`, `tests/`, `demo/`, `include/`, `examples/` e `po/`.

Fica: `COPYING`, `doc/COPYRIGHTS`, `share/licenses/` e o arquivo de licença de cada pacote — a GPL
exige o texto da licença na redistribuição. A poda **reprova** se algum deles faltar no
destino, e reprova se um dos pacotes proibidos estiver lá: `Rserve` abre um servidor R sem
autenticação na porta 6311, e `RSclient`, `callr`, `processx` e `arrow` são os outros caminhos de
processo ou de rede que nenhum fluxo nosso usa.

## `manifesto` e `compara` — reproduzir é igualdade FORA dos carimbos de build

Os pacotes compilados da fonte gravam a hora da instalação: o `Built:` do `DESCRIPTION`, o
`Meta/package.rds` que o repete e o `TimeDateStamp` do cabeçalho PE da DLL. Uma segunda montagem dá
sha256 diferente neles por construção. `compara` aceita a
diferença **só** nesses arquivos, e qualquer outra reprova — o par causal está em
`tests/test_r_portatil_roteiro.py`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# A receita de isolamento e o caminho do `Rscript` são os do lançador do produto, e não uma cópia:
# a montagem deriva a lista de pacotes com esses dois iguais aos da janela. O resto difere: o
# roteiro corre por dentro de `r_portatil_despeja.R`, sobre a árvore COMPLETA e não a podada, com o
# `stdin` por tubo e não por arquivo, sem o Job Object e sem `CREATE_NO_WINDOW`, no Python do
# `.venv` e fora do congelado.
from refino.io.enriquecimento_r import ambiente_isolado
from refino.io.enriquecimento_r import rscript_da_arvore as rscript

TOPO_FORA = frozenset(
    {"doc", "include", "src", "Tcl", "tests", "unins000.dat", "unins000.exe", "MD5"}
)
PACOTE_FORA = frozenset({"help", "html", "doc", "tests", "demo", "include", "examples", "po"})
# Os que ficam mesmo dentro do que sai: a licença não é documentação opcional.
TOPO_FICA = ("COPYING", "doc/COPYRIGHTS")
PROIBIDOS = ("Rserve", "RSclient", "callr", "processx", "arrow")
# Os pacotes que a montagem compila da fonte — só eles têm carimbo de build.
DA_FONTE = ("crmn", "MetaboAnalystR")
NOMES_DE_LICENCA = ("LICENSE", "LICENCE", "COPYING", "COPYRIGHTS")


# Nomes públicos de compostos. O último par não deve casar: um é inventado, o outro testa a ida e
# volta de acento pela fronteira do processo.
COMPOSTOS = (
    "Glycine",
    "L-Alanine",
    "Citric acid",
    "L-Lactic acid",
    "D-Glucose",
    "Pyruvic acid",
    "Succinic acid",
    "Fumaric acid",
    "L-Malic acid",
    "Oxoglutaric acid",
    "L-Glutamic acid",
    "L-Glutamine",
    "L-Aspartic acid",
    "L-Serine",
    "L-Threonine",
    "L-Valine",
    "L-Leucine",
    "L-Isoleucine",
    "L-Proline",
    "L-Phenylalanine",
    "L-Tyrosine",
    "L-Tryptophan",
    "Urea",
    "L-Ornithine",
    "L-Citrulline",
    "naoexisteXYZ123",
    "Ácido cítrico çã",
)
FUNDO_EXTRA = (
    "L-Arginine",
    "Creatine",
    "Taurine",
    "Hypoxanthine",
    "Xanthine",
    "Uric acid",
    "Adenosine",
    "Inosine",
    "Guanosine",
    "Uridine",
    "Cytidine",
    "Thymine",
    "Uracil",
    "Choline",
    "Betaine",
    "L-Carnitine",
    "Glycerol",
    "Palmitic acid",
    "Oleic acid",
    "Stearic acid",
    "Cholesterol",
    "Cortisol",
    "Hippuric acid",
    "L-Kynurenine",
    "Serotonin",
    "Dopamine",
    "Histamine",
    "L-Histidine",
    "L-Lysine",
    "L-Methionine",
    "L-Cysteine",
    "Glutathione",
    "Spermidine",
    "Putrescine",
    "Niacinamide",
    "Pantothenic acid",
    "Riboflavin",
    "Ascorbic acid",
)


def fluxos() -> dict[str, tuple[dict[str, object], str]]:
    """Os pedidos que a derivação corre, e o desfecho que cada um TEM de dar.

    O nome é o do despejo em `scripts/r_portatil_namespaces/`. O contrato é o de
    `r_service/vias_entrypoint.R`. Um caminho de erro está entre eles de propósito — mapear sem
    casamento —, porque pacote carregado só num ramo de erro sumiria na poda e o ramo morreria no
    laboratório. O desfecho esperado é conferido: um fluxo que «deu certo» pelo motivo errado
    mediria outro caminho do roteiro.

    `ora_local_lista_curta` era o segundo caminho de erro, a recusa de «menos de 3», até o
    usuário retirar o mínimo; a lista de dois compostos passou a CALCULAR — o fluxo mede agora
    esse caminho, que é o comportamento novo.
    """
    validos = [c for c in COMPOSTOS if not c.startswith(("naoexiste", "Ácido"))]
    return {
        "mapear": (
            {"action": "mapear", "lista": list(COMPOSTOS), "fundo": validos + list(FUNDO_EXTRA)},
            "ok",
        ),
        "mapear_sem_casamento": (
            {"action": "mapear", "lista": ["naoexisteXYZ123", "outroInexistente987"]},
            "ok",
        ),
        "ora_local": ({"action": "ora_local", "lista": validos}, "ok"),
        "ora_local_com_fundo": (
            {"action": "ora_local", "lista": validos, "fundo": validos + list(FUNDO_EXTRA)},
            "ok",
        ),
        "ora_local_lista_curta": ({"action": "ora_local", "lista": validos[:2]}, "ok"),
    }


DESPEJA = Path(__file__).resolve().parent / "r_portatil_despeja.R"


def corre_roteiro(
    r_home: Path, roteiro: Path, pedido: dict[str, object], *, proxy_morto: bool = True
) -> dict[str, object]:
    """Corre o roteiro uma vez, com JSON em stdin, e devolve o JSON da última linha de stdout.

    O roteiro corre por dentro de `r_portatil_despeja.R`, e a resposta ganha `_carregados`: o que o
    R tinha carregado ao sair, por qualquer caminho.
    """
    with tempfile.TemporaryDirectory(prefix="refino-r-") as tmp:
        raiz = Path(tmp)
        env = ambiente_isolado(raiz, proxy_morto=proxy_morto)
        despejo = raiz / "carregados.txt"
        env["REFINO_R_ROTEIRO"] = str(roteiro)
        env["REFINO_R_DESPEJO"] = str(despejo)
        p = subprocess.run(
            [str(rscript(r_home)), "--vanilla", str(DESPEJA)],
            input=json.dumps(pedido, ensure_ascii=False).encode("utf-8"),
            capture_output=True,
            env=env,
            cwd=raiz,
            timeout=900,
            check=False,
        )
        carregados = despejo.read_text(encoding="utf-8").split() if despejo.is_file() else []
    linhas = [ln for ln in p.stdout.decode("utf-8", errors="replace").splitlines() if ln.strip()]
    try:
        resposta = json.loads(linhas[-1]) if linhas else {}
    except json.JSONDecodeError:
        resposta = {}
    if not isinstance(resposta, dict):
        resposta = {}
    resposta["_exit"] = p.returncode
    resposta["_carregados"] = carregados
    resposta["_stderr_cauda"] = p.stderr.decode("utf-8", errors="replace")[-2000:]
    return resposta


def versao_do_pacote(lib: Path, nome: str) -> str:
    desc = lib / nome / "DESCRIPTION"
    for linha in desc.read_text(encoding="utf-8", errors="replace").splitlines():
        if linha.startswith("Version:"):
            return linha.split(":", 1)[1].strip()
    raise SystemExit(f"sem Version: em {desc}")


def cmd_deriva(a: argparse.Namespace) -> int:
    r_home, roteiro, dados = Path(a.r_home), Path(a.roteiro), Path(a.dados)
    saida = Path(a.saida)
    saida.mkdir(parents=True, exist_ok=True)
    uniao: set[str] = set()
    for nome, (pedido, esperado) in fluxos().items():
        resp = corre_roteiro(r_home, roteiro, {**pedido, "dados_dir": str(dados)})
        carregados = resp.get("_carregados")
        # O fluxo sem casamento termina por outro ramo — é o que se quer cobrir. O que não pode é
        # faltar o despejo, ou o MetaboAnalystR não estar nele: aí a derivação mediu um R que nem
        # chegou ao pacote, e a lista sairia curta sem nada reprovar.
        if not isinstance(carregados, list) or "MetaboAnalystR" not in carregados:
            print(
                f"{nome}: despejo sem MetaboAnalystR (exit {resp.get('_exit')}, "
                f"status {resp.get('status')}): {resp.get('motivo')}"
            )
            print(resp.get("_stderr_cauda"))
            return 1
        obtido = "ok" if resp.get("status") == "ok" else str(resp.get("desfecho"))
        if obtido != esperado:
            print(f"{nome}: esperava {esperado} e deu {obtido}: {resp.get('motivo')}")
            return 1
        nomes = sorted({str(c) for c in carregados})
        (saida / f"{nome}.txt").write_text("\n".join(nomes) + "\n", encoding="utf-8")
        uniao |= set(nomes)
        print(f"{nome}: {obtido} carregados={len(nomes)}")
    lib = r_home / "library"
    proibidos = sorted(uniao & set(PROIBIDOS))
    if proibidos:
        print(f"os fluxos carregaram pacote proibido: {proibidos}")
        return 1
    linhas = [f"{n} {versao_do_pacote(lib, n)}" for n in sorted(uniao, key=str.lower)]
    Path(a.lista).write_text("\n".join(linhas) + "\n", encoding="utf-8")
    print(f"lista: {len(linhas)} pacotes em {a.lista}")
    return 0


def le_lista(lista: Path) -> list[str]:
    return [ln.split()[0] for ln in lista.read_text(encoding="utf-8").splitlines() if ln.strip()]


def faltam_na_lista(lista: list[str], despejos: dict[str, list[str]]) -> list[str]:
    """Os nomes que algum fluxo carregou e a lista não tem.

    Cada um é um pacote que a poda tiraria, e o fluxo que o carrega morreria na árvore podada.
    """
    na_lista = set(lista)
    return sorted(
        {f"{fluxo}:{n}" for fluxo, nomes in despejos.items() for n in nomes if n not in na_lista}
    )


def confere_manifesto(m: dict[str, tuple[str, int]], lista: list[str]) -> list[str]:
    """O manifesto versionado descreve uma árvore podada pela lista versionada, e nada além dela."""
    erros: list[str] = []
    for rel in m:
        if rel.startswith("/") or ":" in rel or "\\" in rel or ".." in rel.split("/"):
            erros.append(f"caminho não relativo: {rel}")
    pacotes = {rel.split("/")[1] for rel in m if rel.startswith("library/") and rel.count("/") >= 2}
    for nome in lista:
        if f"library/{nome}/DESCRIPTION" not in m:
            erros.append(f"pacote da lista sem DESCRIPTION no manifesto: {nome}")
    for nome in sorted(pacotes - set(lista)):
        erros.append(f"pacote no manifesto fora da lista: {nome}")
    for nome in PROIBIDOS:
        if nome in pacotes:
            erros.append(f"pacote proibido no manifesto: {nome}")
    for rel in TOPO_FICA:
        if rel not in m:
            erros.append(f"falta no manifesto: {rel}")
    if not any(rel.startswith("share/licenses/") for rel in m):
        erros.append("falta share/licenses/ no manifesto")
    return erros


def licencas_do_pacote(pacote: Path) -> list[Path]:
    return [pacote / n for n in NOMES_DE_LICENCA if (pacote / n).is_file()]


def confere_poda(destino: Path, manter: list[str]) -> list[str]:
    """As condições que a poda garante. Devolve a lista de violações; vazia é aprovação."""
    erros: list[str] = []
    for rel in TOPO_FICA:
        if not (destino / rel).is_file():
            erros.append(f"falta {rel}")
    if not (destino / "share" / "licenses").is_dir():
        erros.append("falta share/licenses/")
    lib = destino / "library"
    presentes = {p.name for p in lib.iterdir() if p.is_dir()} if lib.is_dir() else set()
    if not presentes:
        erros.append("library/ vazia ou ausente")
    for nome in PROIBIDOS:
        if nome in presentes:
            erros.append(f"pacote proibido presente: {nome}")
    for nome in manter:
        if nome not in presentes:
            erros.append(f"pacote da lista ausente: {nome}")
    for nome in sorted(presentes - set(manter)):
        erros.append(f"pacote fora da lista presente: {nome}")
    return erros


def cmd_poda(a: argparse.Namespace) -> int:
    origem, destino = Path(a.origem), Path(a.destino)
    manter = le_lista(Path(a.lista))
    if destino.exists():
        shutil.rmtree(destino)
    destino.mkdir(parents=True)
    for item in origem.iterdir():
        if item.name in TOPO_FORA:
            continue
        if item.is_file():
            shutil.copy2(item, destino / item.name)
        elif item.name != "library":
            shutil.copytree(item, destino / item.name)
    for rel in TOPO_FICA:
        if (origem / rel).is_file() and not (destino / rel).exists():
            (destino / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(origem / rel, destino / rel)
    lib_o, lib_d = origem / "library", destino / "library"
    lib_d.mkdir()
    for nome in manter:
        pk = lib_o / nome
        if not pk.is_dir():
            print(f"pacote da lista ausente na origem: {nome}")
            return 1
        shutil.copytree(
            pk,
            lib_d / nome,
            ignore=lambda d, nomes, raiz=pk: [
                n for n in nomes if Path(d) == raiz and n in PACOTE_FORA
            ],
        )
    erros = confere_poda(destino, manter)
    for e in erros:
        print(f"REPROVA: {e}")
    print(
        f"pacotes mantidos: {len(manter)}; "
        f"removidos: {len([p for p in lib_o.iterdir() if p.is_dir()]) - len(manter)}"
    )
    return 1 if erros else 0


def sha256(p: Path) -> str:
    with p.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def manifesto(arvore: Path) -> dict[str, tuple[str, int]]:
    saida: dict[str, tuple[str, int]] = {}
    for p in sorted(arvore.rglob("*")):
        if p.is_file():
            saida[p.relative_to(arvore).as_posix()] = (sha256(p), p.stat().st_size)
    return saida


def e_carimbado(rel: str) -> bool:
    """O arquivo cujo conteúdo muda a cada montagem por conter a hora da instalação."""
    partes = rel.split("/")
    if len(partes) < 3 or partes[0] != "library" or partes[1] not in DA_FONTE:
        return False
    resto = "/".join(partes[2:])
    return resto in {"DESCRIPTION", "Meta/package.rds"} or (
        resto.startswith("libs/x64/") and resto.endswith(".dll")
    )


def escreve_manifesto(m: dict[str, tuple[str, int]], saida: Path) -> None:
    linhas = [f"{h} {n} {rel}" for rel, (h, n) in m.items()]
    saida.write_text("\n".join(linhas) + "\n", encoding="utf-8")


def le_manifesto(arquivo: Path) -> dict[str, tuple[str, int]]:
    m: dict[str, tuple[str, int]] = {}
    for ln in arquivo.read_text(encoding="utf-8").splitlines():
        if ln.strip():
            h, n, rel = ln.split(" ", 2)
            m[rel] = (h, int(n))
    return m


def diferencas(
    atual: dict[str, tuple[str, int]], ref: dict[str, tuple[str, int]], *, fora_dos_carimbos: bool
) -> list[str]:
    erros: list[str] = []
    for rel in sorted(set(ref) - set(atual)):
        erros.append(f"falta: {rel}")
    for rel in sorted(set(atual) - set(ref)):
        erros.append(f"sobra: {rel}")
    for rel in sorted(set(atual) & set(ref)):
        if atual[rel] != ref[rel] and not (fora_dos_carimbos and e_carimbado(rel)):
            erros.append(f"difere: {rel}")
    return erros


def campo_do_description(desc: Path, campo: str) -> str:
    valor: list[str] = []
    dentro = False
    for linha in desc.read_text(encoding="utf-8", errors="replace").splitlines():
        if linha.startswith(campo + ":"):
            valor.append(linha.split(":", 1)[1].strip())
            dentro = True
        elif dentro and linha[:1].isspace():
            valor.append(linha.strip())
        elif dentro:
            break
    return " ".join(valor)


def cmd_licencas(a: argparse.Namespace) -> int:
    """O inventário de licenças da árvore podada, lido dos `DESCRIPTION`.

    É o insumo do ativo de fonte que a GPL exige junto dos binários que nós compilamos.
    """
    lib = Path(a.arvore) / "library"
    linhas = ["pacote\tversao\tlicenca\tarquivo_de_licenca"]
    sem_campo: list[str] = []
    for pk in sorted((p for p in lib.iterdir() if p.is_dir()), key=lambda p: p.name.lower()):
        licenca = campo_do_description(pk / "DESCRIPTION", "License")
        if not licenca:
            sem_campo.append(pk.name)
        arquivos = ",".join(p.name for p in licencas_do_pacote(pk)) or "-"
        linhas.append(f"{pk.name}\t{versao_do_pacote(lib, pk.name)}\t{licenca}\t{arquivos}")
    Path(a.saida).write_text("\n".join(linhas) + "\n", encoding="utf-8")
    print(
        f"licenças: {len(linhas) - 1} pacotes em {a.saida}; "
        f"sem campo License: {sem_campo or 'nenhum'}"
    )
    return 1 if sem_campo else 0


def cmd_manifesto(a: argparse.Namespace) -> int:
    m = manifesto(Path(a.arvore))
    escreve_manifesto(m, Path(a.saida))
    print(f"manifesto: {len(m)} arquivos, {sum(n for _, n in m.values())} bytes em {a.saida}")
    return 0


def cmd_compara(a: argparse.Namespace) -> int:
    atual = manifesto(Path(a.arvore))
    ref = le_manifesto(Path(a.manifesto))
    erros = diferencas(atual, ref, fora_dos_carimbos=a.fora_dos_carimbos)
    carimbados = [r for r in sorted(set(atual) & set(ref)) if atual[r] != ref[r] and e_carimbado(r)]
    for e in erros[:50]:
        print(f"REPROVA: {e}")
    print(
        f"comparados: {len(atual)} contra {len(ref)}; diferenças fora dos carimbos: {len(erros)}; "
        f"carimbados que diferem: {len(carimbados)}"
    )
    return 1 if erros else 0


def main() -> int:
    # Aqui e não no topo: o módulo é importado pelo portão, e lá a saída é do pytest.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("deriva")
    d.add_argument("--r-home", required=True)
    d.add_argument("--roteiro", required=True)
    d.add_argument("--dados", required=True)
    d.add_argument("--saida", required=True)
    d.add_argument("--lista", required=True)
    d.set_defaults(f=cmd_deriva)
    p = sub.add_parser("poda")
    p.add_argument("--origem", required=True)
    p.add_argument("--destino", required=True)
    p.add_argument("--lista", required=True)
    p.set_defaults(f=cmd_poda)
    m = sub.add_parser("manifesto")
    m.add_argument("--arvore", required=True)
    m.add_argument("--saida", required=True)
    m.set_defaults(f=cmd_manifesto)
    lic = sub.add_parser("licencas")
    lic.add_argument("--arvore", required=True)
    lic.add_argument("--saida", required=True)
    lic.set_defaults(f=cmd_licencas)
    c = sub.add_parser("compara")
    c.add_argument("--arvore", required=True)
    c.add_argument("--manifesto", required=True)
    c.add_argument("--fora-dos-carimbos", action="store_true")
    c.set_defaults(f=cmd_compara)
    a = ap.parse_args()
    return int(a.f(a))


if __name__ == "__main__":
    raise SystemExit(main())
