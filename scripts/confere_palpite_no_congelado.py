#!/usr/bin/env python3
"""Confere que o executável congelado leva o modelo do «Palpite» desta árvore.

## Por que existe

O congelamento confere que `POST /matrizes/recomendar` responde `200` dentro do binário — o cenário
5b de `scripts/congela_windows.ps1` —, e não O QUE ela responde. Um `.exe` com o modelo anterior
passaria no mesmo passo. O modelo que viaja é a pergunta que só o congelado responde: o artefato é
um JSON que o `.spec` leva como dado, e o analisador do PyInstaller não o vê.

Este conferidor aprovou na primeira corrida a `windows-2026.10.18.1`, de `629a4af`, a primeira
versão com a política aprendida retreinada no corpus de 7 QCs; o roteiro que a publicou está em
`git show 48fff39:qt.md`.

**E a correção proposta sozinha NÃO separa os dois modelos.** Medido em 18/10/2026 pela mesma rota,
sobre os quatro arquivos reais: o modelo de `a488add` — o da `windows-2026.10.15.1` — e o retreinado
propõem o mesmo corretor nos quatro, `serrf` no `neg_c18` e no `pos_hilic` e `loess` no `pos_c18` e
no `neg_hilic`, e diferem na confiança: 0,968 contra 0,942 no `neg_c18`. Um conferidor que olhasse
só `correction_proposta` aprovaria o binário velho. Por isso a resposta é comparada INTEIRA, campo a
campo, e o teste afirma que as duas respostas diferem antes de plantar a velha.

## O que confere, e em que ordem

1. **a pasta é deste commit**: o `VERSAO.txt` que o congelamento carimba diz o `HEAD` deste clone,
   e `src/` não tem alteração por commitar. Sem isto a comparação seria entre duas versões do
   código, e uma divergência não teria a quem ser atribuída;
2. **os artefatos que viajam são os de `src/`**: o `politica_aprendida.json` dentro de `_internal`
   tem o sha256 do versionado, e o `politica_pela_verdade.json` também — o segundo desde que o
   «Palpite» ganhou a segunda opinião;
3. **o binário responde o que `src/` responde**: o corpo que o congelamento gravou da rota —
   `fluxo-recomendar.json`, na pasta de saídas dele, `%TEMP%\\refino-u10` — é igual, campo a campo,
   à resposta da mesma rota corrida aqui sobre `src/`, com os mesmos dois arquivos do `neg_c18`.

Sai **0** se as quatro conferências conferem, **1** se alguma
diverge, e **2** se não foi possível medir — sem a pasta, sem o `VERSAO.txt`, sem um corpo gravado,
com ele ilegível ou mais velho que a pasta, sem o dado de referência, com a medição daqui a falhar,
ou com o clone noutro commit. **Medição que não aconteceu não é aprovação**, e
por isso tem código próprio. Uma divergência vence uma ausência: com as duas, sai 1.

> **A árvore deixou de ser o Palpite da janela, e há uma quarta conferência.**
> O botão «Palpite» usa `POST /matrizes/palpite`, que MEDE o
> RSD dos QCs fora da amostra em 200 *features*; a árvore das três conferências acima ficou no
> `--auto aprendida` e na rota `/matrizes/recomendar`, que o cenário 5b continua a gravar — e elas
> continuam a valer para ela. **O que a janela de facto usa não era conferido por nada no
> congelado.** A quarta conferência lê o corpo que `scripts/verifica_o_app.ps1` grava da rota nova,
> no `neg_c18` — `out\\verifica-app\\palpite-medido.json` —, e compara-o, campo a campo, com o da
> mesma rota corrida aqui sobre `src/`. A igualdade é EXACTA, e não aproximada: a execução parcial
> é exata, e a amostra é a da semente 42 — no mesmo
> sistema, com as mesmas bibliotecas, o binário e o código têm de dar os mesmos bits. **Corre ANTES
> desta, então, o `verifica_o_app.ps1`**; sem o corpo, ou com ele mais velho que a pasta, a
> conferência é `NAO_MEDIU`. O relatório guarda também os segundos da medição daqui, que são o
> tempo do Palpite no sistema em que o conferidor corre.

## O que NÃO confere

- **os outros três modos**: o cenário 5b só corre o `neg_c18`. Eles usam o mesmo artefato (item 2) e
  o mesmo código, compilado do mesmo commit (item 1);
- **a janela**: que o «Palpite» na tela traga o texto e a confiança do modelo novo é a etapa 5, a
  passagem humana;
- **as outras rotas que servem a árvore** — o palpite da medição dos corretores e o `--auto` —, que
  usam o mesmo artefato pelo mesmo carregador.

Uso, da raiz do clone, depois de `scripts\\congela_windows.ps1` **e de
`scripts\\verifica_o_app.ps1`** (é ele que grava o corpo da quarta conferência), e antes de
qualquer commit:

    .venv\\Scripts\\python.exe scripts\\confere_palpite_no_congelado.py

O relatório vai para `dist\\palpite-congelado-verdicto.json`, ao lado da pasta e fora dela — o que
está dentro de `dist\\refino` viaja, e isto não.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "src"))

MODO = "neg_c18"

# O destino que o `.spec` declara para o artefato (`refino-desktop.spec`, `datas`).
ARTEFATO_NA_PASTA = Path("_internal") / "refino" / "decision" / "politica_aprendida.json"
ARTEFATO_NO_REPO = RAIZ / "src" / "refino" / "decision" / "politica_aprendida.json"
# E o da segunda opinião, ao lado dele. A rota carrega os dois, e a resposta inteira que o
# item 3 compara já o alcança; o sha256 dele acusa sozinho o artefato trocado com a mesma resposta.
ARTEFATO_PELA_VERDADE_NA_PASTA = ARTEFATO_NA_PASTA.with_name("politica_pela_verdade.json")
ARTEFATO_PELA_VERDADE_NO_REPO = ARTEFATO_NO_REPO.with_name("politica_pela_verdade.json")

# `$Saidas = Join-Path $env:TEMP "refino-u10"` no congelamento, e o cenário 5b grava ali o corpo de
# cada rota como `fluxo-<rota>.json`. `gettempdir()` lê o mesmo `TEMP` no Windows.
RESPOSTA_GRAVADA = Path(tempfile.gettempdir()) / "refino-u10" / "fluxo-recomendar.json"

# O corpo do Palpite medido que `scripts/verifica_o_app.ps1` grava do binário.
PALPITE_MEDIDO_GRAVADO = RAIZ / "out" / "verifica-app" / "palpite-medido.json"
# O teto da medição daqui. 52,7 a 59,7 s por modo num Ryzen de 12 núcleos; 1.800 s cobre uma
# máquina de poucos núcleos sem pendurar o roteiro.
TETO_DO_PALPITE_LOCAL_S = 1800.0

APROVA, DIVERGE, NAO_MEDIU = 0, 1, 2
VEREDITOS = {APROVA: "APROVA", DIVERGE: "DIVERGE", NAO_MEDIU: "NAO_MEDIU"}


def sha256(caminho: Path) -> str:
    return hashlib.sha256(caminho.read_bytes()).hexdigest()


def commit_da_pasta(pasta: Path) -> str | None:
    """O commit que o cenário 1b de `congela_windows.ps1` carimbou no `VERSAO.txt` da pasta."""
    versao = pasta / "VERSAO.txt"
    if not versao.is_file():
        return None
    for linha in versao.read_text(encoding="utf-8", errors="replace").splitlines():
        chave, _, valor = linha.partition(":")
        if chave.strip() == "commit":
            return valor.strip() or None
    return None


def _git(raiz: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=raiz, capture_output=True, text=True, check=True
    ).stdout.strip()


def commit_do_clone(raiz: Path = RAIZ) -> str:
    return _git(raiz, "rev-parse", "HEAD")


def alteracoes_em_src(raiz: Path = RAIZ) -> str:
    """O `git status` de `src/`: vazio quando o código é o do commit."""
    return _git(raiz, "status", "--porcelain", "--", "src")


def resposta_local(entrada: Path, modo: str = MODO) -> dict[str, Any]:
    """A rota da recomendação corrida AQUI, sobre `src/`, com os mesmos dois arquivos.

    **A rota, e não a política chamada à mão**, porque foi pela rota que o binário respondeu: tudo o
    que fica entre o arquivo e a árvore — a ingestão, o perfil do diagnóstico, o texto da
    justificativa — tem de estar dos dois lados da comparação.
    """
    from fastapi.testclient import TestClient

    from refino.api.app import build_app

    medidas = entrada / f"measurements_{modo}.csv"
    lista = entrada / f"Sample_List_{modo}.csv"
    cliente = TestClient(build_app(), raise_server_exceptions=False)
    resposta = cliente.post(
        "/matrizes/recomendar",
        files={
            "measurements": (medidas.name, medidas.read_bytes(), "text/csv"),
            "sample_list": (lista.name, lista.read_bytes(), "text/csv"),
        },
        data={"mode": modo},
    )
    if resposta.status_code != 200:
        raise RuntimeError(f"a rota local devolveu {resposta.status_code}: {resposta.text[:300]}")
    corpo: dict[str, Any] = resposta.json()
    return corpo


def palpite_medido_local(
    entrada: Path, modo: str = MODO, *, teto_s: float = TETO_DO_PALPITE_LOCAL_S
) -> dict[str, Any]:
    """`POST /matrizes/palpite` corrida AQUI, sobre `src/`, até ao `200`.

    **A rota e não `mede_o_palpite`**, pelo argumento de `resposta_local`: o binário respondeu pela
    rota, e a ingestão, os lotes e o texto da justificativa têm de estar dos dois lados. Os campos
    são os que `verifica_o_app.ps1` manda — só os arquivos e o modo —, então os padrões da rota
    (`paralelo`, um lote) são os mesmos dos dois lados.
    """
    import time

    from fastapi.testclient import TestClient

    from refino.api.app import build_app

    medidas = entrada / f"measurements_{modo}.csv"
    lista = entrada / f"Sample_List_{modo}.csv"
    cliente = TestClient(build_app(), raise_server_exceptions=False)
    disparo = cliente.post(
        "/matrizes/palpite",
        files={
            "measurements": (medidas.name, medidas.read_bytes(), "text/csv"),
            "sample_list": (lista.name, lista.read_bytes(), "text/csv"),
        },
        data={"mode": modo},
    )
    if disparo.status_code != 202:
        raise RuntimeError(f"o Palpite local devolveu {disparo.status_code}: {disparo.text[:300]}")
    identificador = disparo.json()["identificador"]
    limite = time.monotonic() + teto_s
    while True:
        leitura = cliente.get(f"/trabalhos/{identificador}")
        if leitura.status_code == 200:
            corpo: dict[str, Any] = leitura.json()
            return corpo
        if leitura.status_code != 202:
            raise RuntimeError(
                f"o Palpite local terminou com {leitura.status_code}: {leitura.text[:300]}"
            )
        if time.monotonic() > limite:
            raise RuntimeError(f"o Palpite local passou do teto de {teto_s:.0f} s")
        time.sleep(1.0)


# **As falhas que fazem uma conferência NÃO ACONTECER.** Um corpo gravado
# truncado, a rota daqui a responder `422`, o teto da medição: até uma revisão, as três levantavam,
# o processo saía **1** — o código de DIVERGE — e o relatório nem era escrito. Medição que não
# aconteceu tem código próprio, e é o 2: quem lê o 1 conclui que o binário mede outra coisa.
FALHAS_DE_MEDICAO = (OSError, ValueError, KeyError, RuntimeError)


def le_corpo(caminho: Path) -> tuple[dict[str, Any] | None, str | None]:
    """O corpo JSON gravado, ou a razão de não o ter — nunca uma exceção."""
    try:
        corpo = json.loads(caminho.read_bytes())
    except (OSError, ValueError) as erro:
        return None, f"{type(erro).__name__}: {erro}"
    if not isinstance(corpo, dict):
        corpo = {"<corpo>": corpo}
    return corpo, None


def campos_divergentes(congelada: dict[str, Any], local: dict[str, Any]) -> list[str]:
    """Os campos em que as duas respostas diferem — INCLUINDO o que só uma das duas traz.

    Comparar só os campos comuns deixaria passar um binário que respondesse `{"detail": ...}`: a
    interseção com a resposta boa é vazia, e uma comparação sobre o vazio aprova.
    """
    ausente = object()
    return sorted(
        campo
        for campo in set(congelada) | set(local)
        if congelada.get(campo, ausente) != local.get(campo, ausente)
    )


def confere(
    pasta: Path,
    resposta_gravada: Path,
    entrada: Path,
    *,
    commit_esperado: str,
    alteracoes: Callable[[], str] | None = None,
    modo: str = MODO,
    palpite_gravado: Path | None = None,
    palpite_local: Callable[[Path, str], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """As quatro conferências e o que cada uma leu; o veredito sai das duas listas do relato.

    `alteracoes` é o `git status` de `src/`; o padrão é resolvido na CHAMADA, e não na definição,
    para que quem corre `main` sobre outra árvore possa trocá-lo. `palpite_local` é a medição daqui
    (`palpite_medido_local`), injetável porque custa minutos; `palpite_gravado` é o corpo que o
    binário deu, e `None` quer dizer que a quarta conferência não foi pedida — o relato diz isso,
    e `main` pede-a sempre.
    """
    divergencias: list[str] = []
    nao_medido: list[str] = []
    relato: dict[str, Any] = {
        "quando": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "pasta": str(pasta),
        "modo": modo,
        "commit_esperado": commit_esperado,
        "divergencias": divergencias,
        "nao_medido": nao_medido,
    }

    # 1. a pasta é deste commit, e o código daqui é o do commit
    commit = commit_da_pasta(pasta)
    relato["commit_da_pasta"] = commit
    sujo = (alteracoes or alteracoes_em_src)()
    if commit is None:
        nao_medido.append(f"sem a linha do commit em {pasta / 'VERSAO.txt'}")
    elif commit != commit_esperado:
        nao_medido.append(
            f"a pasta é de {commit[:7]} e este clone está em {commit_esperado[:7]}: "
            f"corra com o clone no commit da pasta (git checkout {commit[:7]}), ou congele de novo"
        )
    if sujo:
        nao_medido.append(f"src/ tem alterações por commitar, e o binário não as tem: {sujo[:200]}")
    if nao_medido:
        relato["veredito"] = VEREDITOS[NAO_MEDIU]
        return relato

    # 2. os artefatos que viajam são os de src/ — o do Palpite pelo RSD e o pela verdade
    for prefixo, na_pasta, no_repo in (
        ("", ARTEFATO_NA_PASTA, ARTEFATO_NO_REPO),
        ("pela_verdade_", ARTEFATO_PELA_VERDADE_NA_PASTA, ARTEFATO_PELA_VERDADE_NO_REPO),
    ):
        relato[f"sha256_{prefixo}no_repo"] = sha256(no_repo)
        artefato = pasta / na_pasta
        if not artefato.is_file():
            # A pasta existe e é deste commit: faltar o artefato é defeito do congelamento, e não
            # medição que não aconteceu — é exactamente o que a linha do artefato no `datas` do
            # `.spec` impede.
            divergencias.append(f"a pasta não leva o artefato: {na_pasta}")
            continue
        relato[f"sha256_{prefixo}na_pasta"] = sha256(artefato)
        if relato[f"sha256_{prefixo}na_pasta"] != relato[f"sha256_{prefixo}no_repo"]:
            divergencias.append(f"o {no_repo.name} da pasta não é o de src/")

    # 3. o binário responde o que src/ responde
    medidas = entrada / f"measurements_{modo}.csv"
    lista = entrada / f"Sample_List_{modo}.csv"
    if not resposta_gravada.is_file():
        nao_medido.append(f"sem a resposta gravada pelo congelamento: {resposta_gravada}")
    elif resposta_gravada.stat().st_mtime < (pasta / "VERSAO.txt").stat().st_mtime:
        # **A pasta de saídas do congelamento NUNCA é limpa** — o próprio roteiro o diz, no
        # cenário da prosa acentuada —, e o cenário 5b não apaga o `fluxo-*.json` antes de
        # escrever. Um congelamento cujo 5b não correu deixa ali a resposta do anterior, talvez
        # de outro modelo. O carimbo do `VERSAO.txt` é escrito no cenário 1b, antes do 5b.
        nao_medido.append(
            f"a resposta gravada é anterior a este congelamento (mais velha que o VERSAO.txt): "
            f"{resposta_gravada}"
        )
    elif not (medidas.is_file() and lista.is_file()):
        nao_medido.append(f"sem o dado de referência do {modo} em {entrada}")
    else:
        congelada, ilegivel = le_corpo(resposta_gravada)
        local: dict[str, Any] | None = None
        if congelada is None:
            nao_medido.append(
                f"a resposta gravada da árvore não é JSON legível: {resposta_gravada} ({ilegivel})"
            )
        else:
            try:
                local = resposta_local(entrada, modo)
            except FALHAS_DE_MEDICAO as erro:
                nao_medido.append(
                    f"a rota da árvore não respondeu aqui: {type(erro).__name__}: {erro}"
                )
        if congelada is not None and local is not None:
            diferentes = campos_divergentes(congelada, local)
            relato["resposta_do_binario"] = congelada
            relato["resposta_de_src"] = local
            relato["campos_divergentes"] = diferentes
            if diferentes:
                divergencias.append(
                    "a resposta do binário difere da de src/ em: " + ", ".join(diferentes)
                )

    # 4. o binário MEDE o Palpite que src/ mede
    if palpite_gravado is None:
        relato["palpite_medido"] = "não pedido"
    elif not palpite_gravado.is_file():
        nao_medido.append(
            f"sem o Palpite medido gravado do binário: {palpite_gravado} — ele é escrito por "
            "scripts\\verifica_o_app.ps1, que corre antes deste"
        )
    elif palpite_gravado.stat().st_mtime < (pasta / "VERSAO.txt").stat().st_mtime:
        # A pasta `out\verifica-app` também não é limpa entre corridas: a guarda é a do item 3.
        nao_medido.append(
            f"o Palpite medido gravado é anterior a este congelamento (mais velho que o "
            f"VERSAO.txt): {palpite_gravado}"
        )
    elif not (medidas.is_file() and lista.is_file()):
        nao_medido.append(f"sem o dado de referência do {modo} em {entrada}")
    else:
        congelado, ilegivel = le_corpo(palpite_gravado)
        local_medido: dict[str, Any] | None = None
        if congelado is None:
            nao_medido.append(
                f"o Palpite medido gravado não é JSON legível: {palpite_gravado} ({ilegivel})"
            )
        else:
            inicio = time.monotonic()
            try:
                local_medido = (palpite_local or palpite_medido_local)(entrada, modo)
            except FALHAS_DE_MEDICAO as erro:
                nao_medido.append(
                    f"o Palpite medido não foi medido aqui: {type(erro).__name__}: {erro}"
                )
            relato["segundos_do_palpite_de_src"] = round(time.monotonic() - inicio, 1)
        if congelado is not None and local_medido is not None:
            diferentes_medido = campos_divergentes(congelado, local_medido)
            relato["palpite_medido_do_binario"] = congelado
            relato["palpite_medido_de_src"] = local_medido
            relato["campos_divergentes_do_palpite_medido"] = diferentes_medido
            if diferentes_medido:
                divergencias.append(
                    "o Palpite medido do binário difere do de src/ em: "
                    + ", ".join(diferentes_medido)
                )

    if divergencias:
        relato["veredito"] = VEREDITOS[DIVERGE]
    elif nao_medido:
        relato["veredito"] = VEREDITOS[NAO_MEDIU]
    else:
        relato["veredito"] = VEREDITOS[APROVA]
    return relato


def codigo_de_saida(relato: dict[str, Any]) -> int:
    return {nome: codigo for codigo, nome in VEREDITOS.items()}[relato["veredito"]]


def main(argv: Sequence[str] | None = None) -> int:
    # A console do Windows é `cp1252` ou `cp932`, e o relato tem acento.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    analisador = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    analisador.add_argument("--pasta", type=Path, default=RAIZ / "dist" / "refino")
    analisador.add_argument("--resposta", type=Path, default=RESPOSTA_GRAVADA)
    analisador.add_argument(
        "--entrada", type=Path, default=RAIZ / "serrf" / "data" / "input", help="dado de referência"
    )
    analisador.add_argument("--commit", default=None, help="o commit esperado; padrão: o HEAD")
    analisador.add_argument(
        "--relatorio", type=Path, default=RAIZ / "dist" / "palpite-congelado-verdicto.json"
    )
    analisador.add_argument(
        "--palpite-medido",
        type=Path,
        default=PALPITE_MEDIDO_GRAVADO,
        help="o corpo do Palpite medido que verifica_o_app.ps1 gravou do binário",
    )
    argumentos = analisador.parse_args(argv)

    relato = confere(
        argumentos.pasta,
        argumentos.resposta,
        argumentos.entrada,
        commit_esperado=argumentos.commit or commit_do_clone(),
        palpite_gravado=argumentos.palpite_medido,
    )
    argumentos.relatorio.parent.mkdir(parents=True, exist_ok=True)
    argumentos.relatorio.write_text(
        json.dumps(relato, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )

    print(f"pasta:     {relato['pasta']}")
    da_pasta, esperado = relato.get("commit_da_pasta"), relato["commit_esperado"]
    print(f"commit:    da pasta {da_pasta}, esperado {esperado}")
    if "sha256_na_pasta" in relato:
        na_pasta, em_src = relato["sha256_na_pasta"][:16], relato["sha256_no_repo"][:16]
        print(f"artefato:  {na_pasta} na pasta, {em_src} em src/")
    if "resposta_do_binario" in relato:
        binario, de_src = relato["resposta_do_binario"], relato["resposta_de_src"]
        print(
            f"{relato['modo']}:  binário {binario.get('correction_proposta')} "
            f"{binario.get('confianca')}; src/ {de_src.get('correction_proposta')} "
            f"{de_src.get('confianca')}"
        )
    if "palpite_medido_do_binario" in relato:
        binario, de_src = relato["palpite_medido_do_binario"], relato["palpite_medido_de_src"]
        print(
            f"medido:    binário {binario.get('correction_proposta')} "
            f"margem {binario.get('margem')}; src/ {de_src.get('correction_proposta')} "
            f"margem {de_src.get('margem')}, em {relato['segundos_do_palpite_de_src']} s aqui"
        )
    for linha in relato["divergencias"]:
        print(f"DIVERGE:   {linha}")
    for linha in relato["nao_medido"]:
        print(f"NAO MEDIU: {linha}")
    print(f"veredito:  {relato['veredito']}  (relatório em {argumentos.relatorio})")
    return codigo_de_saida(relato)


if __name__ == "__main__":
    raise SystemExit(main())
