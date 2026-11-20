#!/usr/bin/env python3
"""Gera as saídas prontas que viajam na pasta entregue — os quatro modos, já corrigidos.

## Por que existe

Quem recebe a pasta não devia esperar por cálculo nenhum. As 32 saídas custam dezenas de minutos,
e a pasta chega com elas prontas.

## O que ele NÃO faz, declarado

**Não reimplementa nada.** As saídas vêm das MESMAS rotas que a janela chama —
`POST /matrizes/{corrigir,exportar,processar}` sobre um `TestClient` do `build_app()` —, com os
mesmos formulários e os mesmos padrões. Uma segunda implementação aqui produziria arquivos
plausíveis que o aplicativo não produz, que é pior que não os enviar.

**Os nomes saem de `nome_do_arquivo`**, a função que a janela usa para sugerir o nome ao salvar.
Escrevê-los com uma `f`-string daqui seria a segunda fonte de verdade sobre o nome, e ela diverge
no dia em que a janela mudar.

## Uso

    .venv/Scripts/python.exe scripts/gera_saidas_prontas.py [--saida out/saidas-prontas]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]

REPO = Path(__file__).resolve().parent.parent

MODOS: tuple[str, ...] = ("pos_hilic", "pos_c18", "neg_hilic", "neg_c18")

# **A ordem é a do DOMÍNIO e não a alfabética**, e é a mesma que os dois botões de lote da janela
# percorrem: `none` primeiro porque é a referência, e o `serrf` antes do `loess` porque é o caro.
CORRETORES: tuple[str, ...] = ("none", "serrf", "loess")

# **UM LOTE, e a escolha é do DADO e não da recomendação.** Os quatro arquivos deste estudo
# declaram um lote único `A`, e a aba de recomendação
# sugere dois para alguns modos por detecção de ponto de mudança. Sugestão não é fato: declarar
# dois aqui seria pôr uma hipótese nossa dentro de um arquivo que quem recebe vai ler como dado.
N_LOTES = 1

SEED = 42

# **AS DUAS VARIANTES DO «PROCESSADO», e a segunda existe porque a primeira sozinha engana.**
#
# Com os padrões do programa, os quatro eixos de pré-processamento são `none` e o único que age é
# a imputação — então a matriz processada sai quase igual à corrigida, e quem a abrisse concluiria
# que a ferramenta não normaliza. Ela normaliza; o programa é que **não opina** sobre esses eixos,
# e é isso que as duas pastas juntas mostram.
#
# A segunda variante é a cascata clássica do MetaboAnalyst — imputação, normalização por mediana,
# `log10` e Pareto —, que é o que o laboratório já faz na web.
VARIANTES_DO_PROCESSADO: dict[str, dict[str, Any]] = {
    "padrao": {},
    "metaboanalyst": {
        "sample_normalization": "median",
        "transformation": "log10",
        "scaling": "pareto",
    },
}


def _formulario(modo: str, **extra: Any) -> dict[str, Any]:
    return {"mode": modo, "seed": SEED, "n_batches": N_LOTES, **extra}


def _arquivos(modo: str) -> dict[str, tuple[str, bytes, str]]:
    raiz = REPO / "serrf" / "data" / "input"
    medidas = raiz / f"measurements_{modo}.csv"
    lista = raiz / f"Sample_List_{modo}.csv"
    for caminho in (medidas, lista):
        if not caminho.is_file():
            raise SystemExit(f"dado de referência ausente: {caminho}")
    return {
        "measurements": (medidas.name, medidas.read_bytes(), "text/csv"),
        "sample_list": (lista.name, lista.read_bytes(), "text/csv"),
    }


def _reclama(cliente: Any, rota: str, formulario: dict[str, Any], modo: str) -> bytes:
    """Dispara e espera. **O `202` devolve identificador, e os bytes vêm de `GET /trabalhos/{id}`.**

    As quatro rotas longas seguem este contrato; um cliente que esperasse bytes do
    `POST` receberia o JSON de aceite e escreveria um CSV que não é matriz nenhuma.
    """
    resposta = cliente.post(rota, files=_arquivos(modo), data=formulario)
    if resposta.status_code != 202:
        raise SystemExit(
            f"{rota} recusou {formulario}: {resposta.status_code} {resposta.text[:300]}"
        )
    identificador = resposta.json()["identificador"]
    while True:
        pronta = cliente.get(f"/trabalhos/{identificador}")
        if pronta.status_code == 202:
            time.sleep(0.5)
            continue
        if pronta.status_code != 200:
            raise SystemExit(f"{rota} falhou: {pronta.status_code} {pronta.text[:300]}")
        return bytes(pronta.content)


def escreve_leia_me(destino: Path, registro: list[dict[str, Any]]) -> Path:
    """A folha que quem recebe lê ao abrir a pasta. **Derivada do registro, não escrita à mão.**

    Os corretores recomendados, os parâmetros e as contagens saem do que de facto correu; um texto
    escrito à mão diria o que quisemos gerar, e não o que geramos — e envelheceria no primeiro modo
    novo.
    """
    recomendados = {r["modo"]: r["method"] for r in registro if r.get("variante") == "padrao"}
    por_pasta: dict[str, int] = {}
    for r in registro:
        chave = (
            "3-matrizes-processadas"
            if "variante" in r
            else (
                "1-matrizes-corrigidas"
                if r["rota"].endswith("corrigir")
                else "2-formato-metaboanalyst"
            )
        )
        por_pasta[chave] = por_pasta.get(chave, 0) + 1

    linhas = [
        "SAIDAS PRONTAS DOS QUATRO ESTUDOS",
        "=================================",
        "",
        "Estas tres pastas trazem os quatro modos deste estudo ja processados pelo programa,",
        "para o senhor poder abrir os arquivos sem ter de rodar nada. Tudo aqui foi produzido",
        "pelas MESMAS rotas que a janela usa, com os mesmos padroes.",
        "",
        "Os arquivos tem features nas LINHAS e amostras nas COLUNAS, como os do laboratorio.",
        "",
        f"SEMENTE: {SEED}   ·   LOTES DECLARADOS: {N_LOTES}",
        "",
        "Um lote so, e a escolha e do dado: os quatro arquivos declaram lote unico `A`. A aba de",
        "recomendacao chega a sugerir dois lotes para alguns modos, por deteccao de ponto de",
        "mudanca — sugestao nao e fato, e nao a aplicamos aqui.",
        "",
        "",
        f"1-matrizes-corrigidas/   ({por_pasta.get('1-matrizes-corrigidas', 0)} arquivos)",
        "-" * 60,
        "A matriz com a deriva de sinal corrigida, e nada mais. Quatro modos vezes os tres",
        "corretores: `none` (nenhuma correcao, a referencia), `serrf` e `loess`.",
        "",
        "O nome do arquivo traz o corretor: <modo>_corrigir_<corretor>.csv",
        "",
        "",
        f"2-formato-metaboanalyst/   ({por_pasta.get('2-formato-metaboanalyst', 0)} arquivos)",
        "-" * 60,
        "As mesmas correcoes, no formato que o MetaboAnalyst aceita — e ja com o corte de 5.000",
        "features que o site impoe, feito pelas features menos estaveis nos QCs em vez de pelo",
        "IQR que o site aplicaria sozinho.",
        "",
        "",
        f"3-matrizes-processadas/   ({por_pasta.get('3-matrizes-processadas', 0)} arquivos)",
        "-" * 60,
        "O pipeline completo: correcao, imputacao de ausentes e o pre-processamento. Sao DUAS",
        "variantes por modo, e a diferenca entre elas e o ponto:",
        "",
        "  <modo>_processada_padrao.csv",
        "      Os padroes do programa. So a imputacao age (1/5 do minimo); normalizacao,",
        "      transformacao e escalonamento ficam em `none`. E o que o senhor obtem clicando",
        "      em «Executar pipeline» sem tocar em nada.",
        "",
        "  <modo>_processada_metaboanalyst.csv",
        "      A cascata classica: imputacao, normalizacao pela mediana da amostra, log10 e",
        "      escalonamento de Pareto. E o que o laboratorio ja faz na web.",
        "",
        "AS DUAS EXISTEM PORQUE O PROGRAMA NAO OPINA SOBRE ESSES EIXOS. Ele propoe apenas duas",
        "coisas — o corretor e o numero de lotes —; sobre normalizacao, transformacao,",
        "escalonamento e os cortes, a escolha e sua, e o padrao dele e nao mexer.",
        "",
        "O CORRETOR DESTA PASTA E O QUE O PROGRAMA RECOMENDA para cada modo:",
        "",
    ]
    for modo in sorted(recomendados):
        linhas.append(f"    {modo:<12} {recomendados[modo]}")
    linhas += [
        "",
        "A recomendacao vem com a ressalva medida, que a janela mostra ao lado dela: a politica",
        "aprendida acerta o corretor em 3 dos 4 modos, e escolher sempre o mesmo metodo tambem",
        "acerta 3 — ela NAO e melhor que a regra mais simples. Por isso ela e segunda opiniao, e",
        "nao veredito.",
        "",
        "",
        "O QUE NAO ESTA AQUI",
        "-" * 60,
        "O enriquecimento por vias metabolicas: ele depende de um servico em conteiner que nao",
        "viaja com o programa. Exportar no formato do MetaboAnalyst continua funcionando.",
        "",
        "A lista das features removidas por corte: nestas saidas nenhum corte por RSD ou por",
        "ausentes foi aplicado, entao nao ha o que listar. Na janela, quando o senhor declara um",
        "corte, o programa grava ao lado do CSV um `<nome>-removidas.csv` com uma linha por",
        "feature que ficou de fora e a causa de cada uma.",
        "",
        "O `PROCEDENCIA.json` ao lado desta folha traz, arquivo a arquivo, a rota que o produziu,",
        "os parametros e quanto tempo custou.",
        "",
    ]
    alvo = destino / "LEIA-ME-das-saidas.txt"
    alvo.write_text("\n".join(linhas), encoding="utf-8")
    return alvo


def main() -> int:
    analisador = argparse.ArgumentParser(description=__doc__)
    analisador.add_argument("--saida", default="out/saidas-prontas")
    analisador.add_argument(
        "--modos", nargs="*", default=list(MODOS), help="para sondar com um modo só"
    )
    argumentos = analisador.parse_args()

    from fastapi.testclient import TestClient

    from refino.api.app import build_app
    from refino.desktop.janela.artefatos import nome_do_arquivo

    destino = REPO / argumentos.saida
    pastas = {
        "corrigir": destino / "1-matrizes-corrigidas",
        "exportar": destino / "2-formato-metaboanalyst",
        "processar": destino / "3-matrizes-processadas",
    }
    for pasta in pastas.values():
        pasta.mkdir(parents=True, exist_ok=True)

    cliente = TestClient(build_app(), raise_server_exceptions=False)
    registro: list[dict[str, Any]] = []
    comeco = time.perf_counter()

    for modo in argumentos.modos:
        for metodo in CORRETORES:
            for chave, rota in (
                ("corrigir", "/matrizes/corrigir"),
                ("exportar", "/matrizes/exportar"),
            ):
                marca = time.perf_counter()
                corpo = _reclama(cliente, rota, _formulario(modo, method=metodo), modo)
                alvo = pastas[chave] / nome_do_arquivo(modo, chave, metodo)
                alvo.write_bytes(corpo)
                gasto = time.perf_counter() - marca
                registro.append(
                    {
                        "arquivo": alvo.name,
                        "rota": rota,
                        "modo": modo,
                        "method": metodo,
                        "n_batches": N_LOTES,
                        "seed": SEED,
                        "bytes": len(corpo),
                        "segundos": round(gasto, 1),
                    }
                )
                print(f"  {alvo.name:<44} {len(corpo):>10} bytes  {gasto:6.1f} s", flush=True)

        # **O PROCESSADO usa o corretor que o PROGRAMA RECOMENDA para este modo.** É a única das
        # três pastas em que há uma escolha a fazer, e fazê-la com a recomendação medida é o que
        # mostra a ferramenta inteira em vez de um caminho que escolhemos à mão.
        palpite = cliente.post("/matrizes/recomendar", files=_arquivos(modo), data={"mode": modo})
        if palpite.status_code != 200:
            raise SystemExit(f"recomendar recusou {modo}: {palpite.status_code}")
        recomendado = str(palpite.json()["correction_proposta"])
        print(f"  [{modo}] o programa recomenda `{recomendado}`", flush=True)

        for rotulo, eixos in VARIANTES_DO_PROCESSADO.items():
            marca = time.perf_counter()
            corpo = _reclama(
                cliente,
                "/matrizes/processar",
                _formulario(modo, method=recomendado, **eixos),
                modo,
            )
            alvo = pastas["processar"] / f"{modo}_processada_{rotulo}.csv"
            alvo.write_bytes(corpo)
            gasto = time.perf_counter() - marca
            registro.append(
                {
                    "arquivo": alvo.name,
                    "rota": "/matrizes/processar",
                    "modo": modo,
                    "method": recomendado,
                    "variante": rotulo,
                    "eixos": eixos,
                    "n_batches": N_LOTES,
                    "seed": SEED,
                    "bytes": len(corpo),
                    "segundos": round(gasto, 1),
                }
            )
            print(f"  {alvo.name:<44} {len(corpo):>10} bytes  {gasto:6.1f} s", flush=True)

    (destino / "PROCEDENCIA.json").write_text(
        json.dumps(
            {
                "gerado_em": time.strftime("%Y-%m-%d %H:%M:%S"),
                "seed": SEED,
                "n_batches": N_LOTES,
                "corretores": list(CORRETORES),
                "variantes_do_processado": VARIANTES_DO_PROCESSADO,
                "arquivos": registro,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    folha = escreve_leia_me(destino, registro)
    print(f"  {folha.name}", flush=True)

    total = time.perf_counter() - comeco
    print(f"\n{len(registro)} arquivos em {destino.relative_to(REPO)}, {total / 60:.1f} min")
    # **A conferência de não-vacuidade.** Um arquivo de zero bytes passa em qualquer contagem, e o
    # que viaja na pasta entregue não pode ser uma pasta de CSV vazios.
    magros = [r["arquivo"] for r in registro if r["bytes"] < 10_000]
    if magros:
        print(f"FALHA: {len(magros)} arquivo(s) abaixo de 10 KB: {magros}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
