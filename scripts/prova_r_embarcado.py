#!/usr/bin/env python3
"""O R embarcado ARRANCA isolado, e a sonda consegue reprovar.

## O que esta sonda decide

Se a árvore podada por `scripts/monta_r_portatil.ps1` é um R que roda **sozinho**: arranca, resolve
`R.home()` e `.libPaths()` dentro dela, carrega o MetaboAnalystR, e calcula a ORA local sem rede. É
a prova de ambiente, nas duas direções; cobertura unitária não diria nada disto.

## As verificações, e a âncora de cada uma

- **arranca, e `R.home()` e `.libPaths()` resolvem dentro da árvore** — a base das outras.
- **`library(MetaboAnalystR)` carrega.** Âncora: numa cópia sem `qs2`, não carrega. Sem isto, um R
  da máquina satisfaria tudo.
- **a biblioteca do usuário não entra.** Planta: a pasta padrão do usuário existe e `R_LIBS_USER`
  está **ausente**; sem `.libPaths(.Library)` ela entra (a planta aterrou), com ele não entra.
- **a ORA local calcula sem rede.** Sem as bibliotecas de dados, ela recusa por biblioteca
  ausente — e não tenta baixar, porque o roteiro troca a função de biblioteca do pacote.
- **a árvore copiada para caminho com espaço e acento passa o mesmo** — é a cópia em que as
  plantas correm.
- **proibidos ausentes; `COPYING`, `doc/COPYRIGHTS` e licenças presentes.** Plantar `Rserve`, ou
  tirar `doc/COPYRIGHTS`, reprova.
- **o `[[:cntrl:]]` do roteiro, neste R do Windows, recusa U+0092, U+0085, U+2028 e U+2029 como
  `pedido_invalido`** — o conjunto que `src/refino/io/identificacao.py` declara ser o do R, e que
  sem esta verificação só tinha uma medição à mão. Âncora: o mesmo nome com U+00A0 passa
  a guarda.
- **`pedido_invalido` sem `action` e com `action` desconhecida, e `lista_igual_ao_fundo`** — três
  desfechos do vocabulário fechado. O último tem par: o
  mesmo pedido com o fundo maior que a lista calcula.

**A biblioteca do usuário é plantada num `LOCALAPPDATA` temporário, e não no do usuário.** O R
deriva a pasta padrão de `%LOCALAPPDATA%\\R\\win-library\\4.4`, então apontar a variável para um
temporário reproduz o mecanismo sem escrever na máquina de quem corre a sonda. O par causal é o que
prova que a planta chega ao R: sem ele, «o marcador não entrou» passaria também com uma planta que o
R nunca viu.

## Uso

    .venv/Scripts/python.exe scripts/prova_r_embarcado.py --arvore <pasta> --dados <pasta>
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]

AQUI = Path(__file__).resolve().parent
REPO = AQUI.parent
sys.path.insert(0, str(AQUI))

import r_portatil  # noqa: E402

# O que corre DENTRO do R. Não levanta: devolve JSON, para a sonda separar «não arrancou» de
# «arrancou e não carregou».
PROGRAMA = r"""
%(reset)s
r <- list(home = normalizePath(R.home(), "/", mustWork = FALSE),
          lib = normalizePath(.libPaths(), "/", mustWork = FALSE))
r$carrega <- tryCatch({ suppressPackageStartupMessages(library(MetaboAnalystR)); "ok" },
                      error = function(e) conditionMessage(e))
cat(jsonlite::toJSON(r, auto_unbox = TRUE), "\n")
"""

PACOTE_ANCORA = "qs2"


def roteiro_padrao() -> Path:
    # O roteiro que viaja, e só ele: o da medição é outro programa (sem recuo para lá).
    return REPO / "r_service" / "vias_entrypoint.R"


def corre_programa(
    arvore: Path, *, reset: bool, local_appdata: Path | None = None
) -> dict[str, object]:
    with tempfile.TemporaryDirectory(prefix="refino-sonda-") as tmp:
        raiz = Path(tmp)
        env = r_portatil.ambiente_isolado(raiz, proxy_morto=True)
        if local_appdata is not None:
            # A planta: a pasta padrão do usuário existe, e R_LIBS_USER não está no ambiente.
            env["LOCALAPPDATA"] = str(local_appdata)
            env.pop("R_LIBS_USER", None)
        prog = raiz / "sonda.R"
        prog.write_text(
            PROGRAMA % {"reset": ".libPaths(.Library)" if reset else ""}, encoding="utf-8"
        )
        p = subprocess.run(
            [str(r_portatil.rscript(arvore)), "--vanilla", str(prog)],
            capture_output=True,
            env=env,
            cwd=raiz,
            timeout=600,
            check=False,
        )
    linhas = [ln for ln in p.stdout.decode("utf-8", errors="replace").splitlines() if ln.strip()]
    try:
        r = json.loads(linhas[-1]) if linhas else {}
    except json.JSONDecodeError:
        r = {}
    r["_exit"] = p.returncode
    r["_stderr"] = p.stderr.decode("utf-8", errors="replace")[-1500:]
    return r


def dentro(caminho: str, arvore: Path) -> bool:
    try:
        Path(caminho).resolve().relative_to(arvore.resolve())
    except ValueError:
        return False
    return True


class Sonda:
    def __init__(self) -> None:
        self.reprovas: list[str] = []

    def afirma(self, nome: str, ok: bool, detalhe: str = "") -> None:
        print(f"{'OK      ' if ok else 'REPROVA '} {nome}{(' — ' + detalhe) if detalhe else ''}")
        if not ok:
            self.reprovas.append(nome)


def verifica_arvore(s: Sonda, arvore: Path, dados: Path, roteiro: Path, rotulo: str) -> None:
    r = corre_programa(arvore, reset=True)
    libs = r.get("lib") if isinstance(r.get("lib"), list) else [r.get("lib")]
    s.afirma(
        f"[{rotulo}] arranca e R.home() fica na árvore",
        r.get("_exit") == 0 and dentro(str(r.get("home")), arvore),
        str(r.get("home")),
    )
    s.afirma(
        f"[{rotulo}] .libPaths() só dentro da árvore",
        bool(libs) and all(dentro(str(x), arvore) for x in libs),
        str(libs),
    )
    s.afirma(
        f"[{rotulo}] library(MetaboAnalystR) carrega",
        r.get("carrega") == "ok",
        str(r.get("carrega"))[:200],
    )

    lista = [c for c in r_portatil.COMPOSTOS if not c.startswith(("naoexiste", "Ácido"))]
    pedido = {"action": "ora_local", "lista": lista}
    com = r_portatil.corre_roteiro(arvore, roteiro, {**pedido, "dados_dir": str(dados)})
    vias = com.get("vias")
    s.afirma(
        f"[{rotulo}] ORA local calcula sem rede",
        com.get("status") == "ok" and isinstance(vias, list) and len(vias) > 0,
        f"status={com.get('status')} vias={len(vias) if isinstance(vias, list) else vias}",
    )
    with tempfile.TemporaryDirectory(prefix="refino-sonda-vazia-") as vazia:
        sem = r_portatil.corre_roteiro(arvore, roteiro, {**pedido, "dados_dir": vazia})
    s.afirma(
        f"[{rotulo}] sem bibliotecas, a ORA recusa por biblioteca ausente",
        sem.get("status") == "erro" and sem.get("desfecho") == "biblioteca_ausente",
        f"status={sem.get('status')} desfecho={sem.get('desfecho')}",
    )


# Pontos do conjunto que `src/refino/io/identificacao.py` declara ser o do `[[:cntrl:]]` do roteiro:
# dois da faixa C1 (a aspa tipográfica do Excel lida como latin-1, e o NEL) e os dois separadores
# Unicode — a parte do conjunto que um R com outras tabelas de caractere poderia não casar.
CONTROLES = (("U+0092", "\u0092"), ("U+0085", "\u0085"), ("U+2028", "\u2028"), ("U+2029", "\u2029"))


def verifica_recusas(s: Sonda, arvore: Path, dados: Path, roteiro: Path) -> None:
    """Desfechos de recusa do vocabulário do roteiro.

    Os pedidos de `pedido_invalido` vão com uma pasta de bibliotecas VAZIA: a guarda corre antes da
    conferência dos arquivos, então um nome que a passa chega a `biblioteca_ausente` — e é esse o
    desfecho da âncora, o mesmo pedido com U+00A0 (que não é controle) no lugar do caractere. Sem
    ela, *"o R recusou"* passaria também com uma guarda que recusasse qualquer não-ASCII.
    """

    def pede(pedido: dict[str, object], pasta: str) -> dict[str, object]:
        r = r_portatil.corre_roteiro(arvore, roteiro, {**pedido, "dados_dir": pasta})
        return {k: r.get(k) for k in ("status", "desfecho", "motivo", "_exit")}

    with tempfile.TemporaryDirectory(prefix="refino-sonda-vazia-") as vazia:
        r = pede({"action": "mapear", "lista": ["Glycine", "Citric\u00a0acid"]}, vazia)
        s.afirma(
            "âncora: nome com U+00A0 passa a guarda de controle e chega a biblioteca_ausente",
            r["desfecho"] == "biblioteca_ausente",
            str(r),
        )
        for rotulo, ch in CONTROLES:
            r = pede({"action": "mapear", "lista": ["Glycine", f"Citric{ch}acid"]}, vazia)
            s.afirma(
                f"nome com {rotulo} na lista: o R recusa como pedido_invalido (guarda de controle)",
                r["status"] == "erro"
                and r["desfecho"] == "pedido_invalido"
                and "controle" in str(r["motivo"]),
                str(r),
            )
        r = pede({"lista": ["Glycine"]}, vazia)
        s.afirma(
            "pedido sem action: pedido_invalido",
            r["desfecho"] == "pedido_invalido" and "<ausente>" in str(r["motivo"]),
            str(r),
        )
        r = pede({"action": "ora_externa_inexistente", "lista": ["Glycine"]}, vazia)
        s.afirma(
            "action desconhecida: pedido_invalido",
            r["desfecho"] == "pedido_invalido" and "ora_externa_inexistente" in str(r["motivo"]),
            str(r),
        )

    # `lista_igual_ao_fundo`, com o par: o mesmo pedido com o fundo maior que a lista calcula.
    validos = [c for c in r_portatil.COMPOSTOS if not c.startswith(("naoexiste", "Ácido"))]
    igual = pede({"action": "ora_local", "lista": validos, "fundo": validos}, str(dados))
    s.afirma(
        "lista igual ao fundo: lista_igual_ao_fundo",
        igual["status"] == "erro" and igual["desfecho"] == "lista_igual_ao_fundo",
        str(igual),
    )
    maior = pede(
        {"action": "ora_local", "lista": validos, "fundo": validos + list(r_portatil.FUNDO_EXTRA)},
        str(dados),
    )
    s.afirma(
        "par: com o fundo MAIOR que a lista, a mesma ORA calcula",
        maior["status"] == "ok",
        str(maior),
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--arvore", required=True)
    ap.add_argument("--dados", required=True)
    ap.add_argument("--lista", default=str(AQUI / "r_portatil_pacotes.txt"))
    ap.add_argument("--roteiro", default=str(roteiro_padrao()))
    a = ap.parse_args()
    arvore, dados, roteiro = Path(a.arvore), Path(a.dados), Path(a.roteiro)
    lista = r_portatil.le_lista(Path(a.lista))
    s = Sonda()

    verifica_arvore(s, arvore, dados, roteiro, "original")
    verifica_recusas(s, arvore, dados, roteiro)
    s.afirma(
        "proibidos ausentes, licenças presentes, só a lista",
        not r_portatil.confere_poda(arvore, lista),
        "; ".join(r_portatil.confere_poda(arvore, lista))[:300],
    )

    carregados_dirs = [arvore / "library" / n for n in lista]
    n_arquivos = sum(1 for d in carregados_dirs for p in d.rglob("*") if p.is_file())
    print(
        f"INFO     arquivos nas pastas dos {len(lista)} pacotes que os fluxos carregam: "
        f"{n_arquivos} (teto do que library(MetaboAnalystR) pode abrir, não contagem de aberturas)"
    )

    with tempfile.TemporaryDirectory(prefix="refino-sonda-copia-") as tmp:
        copia = Path(tmp) / "pasta de prova çã" / "R"
        shutil.copytree(arvore, copia)
        verifica_arvore(s, copia, dados, roteiro, "espaço e acento")

        appdata = Path(tmp) / "appdata"
        planta = appdata / "R" / "win-library" / "4.4"
        planta.mkdir(parents=True)
        sem_reset = corre_programa(copia, reset=False, local_appdata=appdata)
        com_reset = corre_programa(copia, reset=True, local_appdata=appdata)
        libs_sem = (
            sem_reset.get("lib")
            if isinstance(sem_reset.get("lib"), list)
            else [sem_reset.get("lib")]
        )
        libs_com = (
            com_reset.get("lib")
            if isinstance(com_reset.get("lib"), list)
            else [com_reset.get("lib")]
        )
        aterrou = any("win-library" in str(x) for x in libs_sem)
        s.afirma(
            "par causal: sem .libPaths(.Library), a biblioteca do usuário plantada ENTRA",
            aterrou,
            str(libs_sem),
        )
        s.afirma(
            "com .libPaths(.Library), ela não entra",
            not any("win-library" in str(x) for x in libs_com),
            str(libs_com),
        )

        shutil.rmtree(copia / "library" / PACOTE_ANCORA)
        ancora = corre_programa(copia, reset=True)
        s.afirma(
            f"âncora: sem {PACOTE_ANCORA}, library(MetaboAnalystR) NÃO carrega",
            ancora.get("carrega") not in (None, "ok"),
            str(ancora.get("carrega"))[:200],
        )

        (copia / "library" / "Rserve").mkdir()
        (copia / "doc" / "COPYRIGHTS").unlink()
        erros = r_portatil.confere_poda(copia, [n for n in lista if n != PACOTE_ANCORA])
        s.afirma(
            "planta: Rserve presente reprova",
            any("Rserve" in e for e in erros),
            "; ".join(erros)[:200],
        )
        s.afirma("planta: doc/COPYRIGHTS ausente reprova", any("COPYRIGHTS" in e for e in erros))

    print(f"\n{'APROVADA' if not s.reprovas else 'REPROVADA'}: {len(s.reprovas)} reprovação(ões)")
    return 1 if s.reprovas else 0


if __name__ == "__main__":
    raise SystemExit(main())
