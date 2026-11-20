#!/usr/bin/env python3
"""Uma ABERTURA das figuras dentro da pasta congelada. O roteiro que a conduz é o `.ps1` irmão.

Este arquivo é a carga da etapa 3c; ele não é a etapa. Quem alterna as duas cópias, quem lança o
`refino.exe` e quem compara as aberturas é `scripts/prova_figuras_congelado.ps1`. Aqui corre **uma**
abertura, sobre **uma** pasta, e sai um JSON.

## Por que a carga é um arquivo e não uma cadeia dentro do PowerShell

Fonte de Python dentro de uma cadeia do PowerShell atravessa duas gramáticas: a crase é escape ali
e `tests/test_powershell_crases.py` existe por causa de uma linha que uma delas engoliu. Um arquivo
`.py` é lido pelo interpretador e por mais ninguém.

## O interpretador é o EMBARCADO, e é isso que torna esta sonda uma medição sobre o congelado

`refino.exe` não aceita argumento nenhum — `arranque.main` recebe `argv` e ignora-o, de propósito e
com causa: fazê-lo executar `-c` transformaria o executável entregue num interpretador de
código arbitrário. O que a pasta entregue tem é
`_internal\\pythonw.exe` com o seu `._pth`, e é dele que o `loky` lança os workers. É o mesmo
interpretador, o mesmo `_internal`, os mesmos `.pyc` e a mesma cópia do `matplotlib` que viajam
na pasta entregue.

**O que isto NÃO alcança, e vai declarado:** o `refino.exe` importa os módulos da sua própria PYZ,
não os `.pyc` soltos que esta sonda lê. Os dois saem da mesma construção, e a diferença que
sobraria — a janela a montar o `FigureCanvasQTAgg` num `QWidget` real, com o rato — é a **etapa 5**
do roteiro de entrega, que é humana por decisão. Esta sonda mede o motor de
desenho na pasta entregue; ela não clica em nada.

## As quatro coisas que uma abertura mede

1. **Procedência.** Todo módulo lido vem de dentro da pasta que se está a medir, e `pytest` não
   importa. Sem isto, um `PYTHONPATH` herdado faria a sonda medir o clone de quem a corre, e o
   desfecho seria verde.
2. **O cache de fontes.** O estado do `fontlist-v*.json` é lido **antes e depois** de desenhar, e o
   diretório é derivado aqui — com a verificação abaixo, que é o que impede isso de ser a minha
   aritmética em vez da do produto.
3. **O backend.** `matplotlib.get_backend()` e o canvas que a janela de facto monta — o
   `FigureCanvasQTAgg` que `PainelDeFigura` constrói em `desktop/janela/figuras.py`. Com `Agg` a
   figura desenha e nunca aparece: é desfecho de sucesso com tela vazia.
4. **Cada figura REGISTADA desenha.** A lista vem de `refino.plots.catalogo.CONSTRUTORES`, e a
   bijeção contra a tabela de entradas deste arquivo corre nas **duas** direções — construtor
   registado sem entrada aqui **reprova**, em vez de ser saltado em silêncio.

## A derivação do cache é COPIADA, e o que a valida é a pasta que o executável cria

`aponta_o_cache_de_figuras` vive em `src/refino/desktop/arranque.py`, que é o **script de entrada**
do PyInstaller: ele vira o `__main__` do executável e **não viaja** como
`_internal\\refino\\desktop\\arranque.pyc` — medido em 11/09/2026, `ModuleNotFoundError`. Importar
a função do produto era o desenho preferido e não é possível daqui.

Então a fórmula de `aponta_o_cache_de_figuras` é repetida abaixo, e a cópia é validada por
**observação do produto** em vez de por leitura: o `refino.exe` cria este diretório ao arrancar,
antes de qualquer figura, e o roteiro irmão lança-o antes de chamar esta carga. **A sonda exige
que o diretório previsto EXISTA**, e uma fórmula errada aponta para um diretório que ninguém
criou.

Medido em 11/09/2026, com a base `%LOCALAPPDATA%\\refino\\cache-de-figuras` inexistente antes:
lançar `dist\\refino\\refino.exe` criou `6e25a0d1cefa3f0a`, e era exactamente o previsto.

## O oráculo de «desenhou» é o CONTEÚDO, e a distinção entre as figuras faz parte dele

Uma figura que não levanta não desenhou nada: `Figure()` vazia devolve-se sem erro. Por isso cada
figura é **renderizada** e o que se afirma é o conteúdo — eixos, artistas com dado, e o resumo do
buffer RGBA. **E os resumos têm de ser distintos entre si**: um
arnês de captura escreveu quatro PNG idênticos com `EXIT=0`, e figuras todas iguais é exactamente
o desfecho que uma sonda ingénua chamaria de sucesso.

## Uso

    _internal\\pythonw.exe scripts\\prova_figuras_congelado.py --pasta dist\\refino --saida o.json

Sai 0 quando a abertura não achou defeito, 1 quando achou, e 2 quando não conseguiu medir.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

# **Antes da primeira linha impressa.** A consola deste clone é `cp932`, e prosa
# acentuada num relato mata a ferramenta com `UnicodeEncodeError` — inventando uma reprovação que
# não existe. `errors="replace"` e não `strict`: um caractere que a consola não desenhe vira `?`.
for _fluxo in (sys.stdout, sys.stderr):
    if _fluxo is not None and hasattr(_fluxo, "reconfigure"):
        _fluxo.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]


# O nome do arquivo que o `matplotlib` escreve no `MPLCONFIGDIR`. É o oráculo da reconstrução do
# cache, e é ele e não
# o relógio: uma reconstrução rápida — porque o antivírus já viu as fontes — é indistinguível de
# não ter havido reconstrução, se o que se medir for o tempo de abertura.
PADRAO_DO_CACHE = "fontlist-v*.json"

# A âncora de não-vacuidade da procedência, emprestada de `prova_interpretador_embarcado.py`:
# `pytest` é dependência de desenvolvimento e está excluída do congelado. Se ele importar, o
# `sys.path` alcançou algo fora da pasta medida e tudo o que está abaixo passa a ser sobre a
# máquina errada.
NAO_VIAJA = "pytest"


class NaoDaParaMedirError(RuntimeError):
    """A sonda não conseguiu chegar ao ponto de medir. É desfecho 2, e não 1."""


def cache_previsto(exe: Path) -> Path:
    """O `MPLCONFIGDIR` que o produto aponta para esta instalação.

    **Cópia do trecho de `aponta_o_cache_de_figuras`, em `src/refino/desktop/arranque.py`**, e a
    razão de ser cópia está na §«A derivação do cache» acima: o script de entrada não viaja como
    módulo. A base segue a mesma ordem de preferência do original — `LOCALAPPDATA`, depois
    `APPDATA` —, e as outras duas pernas daquele encadeado (`XDG_CACHE_HOME` e `HOME`) ficam de
    fora porque esta sonda só corre em Windows, atrás de um `refino.exe`.
    """
    base = (os.environ.get("LOCALAPPDATA") or "").strip() or (
        os.environ.get("APPDATA") or ""
    ).strip()
    if not base:
        raise NaoDaParaMedirError("nem LOCALAPPDATA nem APPDATA estao definidos")
    instalacao = str(exe.parent)
    marca = hashlib.sha256(instalacao.casefold().encode("utf-8")).hexdigest()[:16]
    return Path(base) / "refino" / "cache-de-figuras" / marca


def _resumo_do_cache(pasta: Path | None) -> dict[str, Any]:
    """Existência, marca de tempo e tamanho do `fontlist-v*.json`. Nunca o relógio."""
    if pasta is None:
        return {"pasta": None, "existe": False, "arquivos": []}
    achados = sorted(pasta.glob(PADRAO_DO_CACHE)) if pasta.is_dir() else []
    return {
        "pasta": str(pasta),
        "existe": bool(achados),
        "arquivos": [
            {
                "nome": c.name,
                "mtime_ns": c.stat().st_mtime_ns,
                "bytes": c.stat().st_size,
            }
            for c in achados
        ],
    }


def _procedencia(modulo: Any, pasta: Path) -> str:
    """O caminho de onde um módulo foi de facto lido, ou o motivo de ele não servir.

    **Devolve o defeito e não um booleano**: *"veio de fora"* e *"não tem `__file__`"*
    pedem investigações diferentes, e um `False` que cobrisse as duas apagaria a distinção.
    """
    caminho = getattr(modulo, "__file__", None)
    if not caminho:
        return f"{modulo.__name__} nao tem __file__"
    try:
        resolvido = Path(caminho).resolve()
    except OSError as erro:  # pragma: no cover - caminho ilegivel e falha de ambiente
        return f"{modulo.__name__}: caminho ilegivel ({type(erro).__name__}: {erro})"
    if not resolvido.is_relative_to(pasta):
        return f"{modulo.__name__} foi lido de {resolvido}, que esta FORA de {pasta}"
    return ""


def _entradas(pasta_de_figuras: Any) -> dict[tuple[str, str], dict[str, Any]]:
    """Os argumentos de cada construtor, um por entrada de `CONSTRUTORES`.

    **A tabela é escrita à mão e a lista de figuras NÃO é** — e é essa a divisão que importa. Uma
    figura nova aparece sozinha, vinda do registo; o que ela não pode é aparecer sem argumentos, e
    a bijeção abaixo faz disso uma reprovação em vez de um salto silencioso.

    Os valores são sintéticos de propósito: o que se mede aqui é que o motor de desenho da pasta
    entregue desenha, e não que um número esteja certo — esse é o trabalho da suíte, que corre
    contra o dado real. Dado real também não caberia: a sonda teria de correr uma execução inteira
    antes de poder desenhar.

    ## O módulo que a pasta congelada NÃO tem é desfecho PREVISTO, e tem de se ler

    `modulos` é derivado do registo **da pasta medida**, não deste clone. Quando a pasta é mais
    velha que este arquivo **e o intervalo acrescentou um MÓDULO** — não apenas uma figura num
    módulo que já existia —, um módulo que este script conhece **não está lá**, e a indexação crua
    levantava `KeyError: 'correcao'`. A condição é estreita e vai dita assim: pasta velha com
    módulo novo. Medido em
    17/09/2026 contra `dist/refino` de 14/09: `registados: 13`, `nao_deu_para_medir:
    "KeyError: 'correcao'"`, e **a bijeção do passo (4) nunca correu** — que é justamente o passo
    desenhado para dizer *«esta pasta está atrasada»*.

    > **É a mesma cegueira por outra porta**, e a nota dela está na chamada de
    > `par_da_correcao` abaixo: `_entradas` monta a tabela inteira **numa expressão**, então
    > qualquer levantamento leva as 32 consigo. Ali foi um `TypeError` por parâmetro novo;
    > aqui é um `KeyError` por pasta velha. O conserto é o mesmo em espírito — falhar com o nome
    > da causa em vez de com o nome da chave.
    >
    > As 32 são a cardinalidade de `CONSTRUTORES`.

    O desfecho continua a ser **2** e não 1: *«a pasta está
    desactualizada»* não é *«uma figura não desenhou»*, e as duas pedem ações opostas — recongelar
    contra investigar o construtor.
    """
    p = pasta_de_figuras
    # **`dispersao` e `composicao` entraram em 17/09/2026**, quando `correcao.py` se dividiu
    # em tres por assunto. A guarda so acusa modulo FALTANTE, e foi por isso que ela nao apanhou a
    # divisao: a pasta continuava a declarar `correcao`, e o que mudou foi ONDE cada construtor
    # vive. Quem apanhou foi esta sonda, a correr contra o congelado — 1 de 16 figuras entao
    # registadas desenhadas, com `AttributeError` na primeira. Nenhum teste a alcanca, porque isto
    # e script.
    #
    # **O «16» fica, e o REFERENTE dele vai dito:** era a cardinalidade de `CONSTRUTORES` na
    # arvore em que esta medicao foi feita, antes do merge do bloco das analises. Reescreve-lo para
    # o numero de hoje seria publicar uma medicao que ninguem fez: ele e um fato datado, nao uma
    # declaracao sobre a arvore de hoje.
    faltam = sorted(
        {
            "deriva",
            "distribuicao",
            "volcano",
            "pca",
            "plsda",
            "correcao",
            "dispersao",
            "composicao",
            # **`rede` e modulo novo**, e e exactamente a condicao que esta guarda nomeia: pasta
            # velha com MODULO novo. Sem ela, `p["rede"]` levantaria `KeyError: 'rede'` e a bijecao
            # do passo (4) — a que existe para dizer *«esta pasta esta atrasada»* — nunca correria.
            "rede",
            # **`curva_do_loess` tambem**, pela mesma razao de `rede`: pasta velha com
            # MODULO novo levantaria `KeyError` antes da bijecao do passo (4).
            "curva_do_loess",
        }
        - set(p)
    )
    if faltam:
        raise NaoDaParaMedirError(
            "a pasta congelada nao declara o(s) modulo(s) de figura "
            + ", ".join(faltam)
            + " — ela e MAIS VELHA que este script, e o registo dela tem "
            f"{len(p)} modulo(s). Recongele antes de medir: uma bijecao contra um registo "
            "caduco mede a arvore de ontem"
        )
    deriva = p["deriva"]
    dist = p["distribuicao"]
    volcano = p["volcano"]
    univariada = p["univariada"]
    anova = p["anova"]
    pca = p["pca"]
    plsda = p["plsda"]
    multiclasse = p["plsda_multiclasse"]
    agrupamento = p["agrupamento"]
    correlacao = p["correlacao"]
    correcao = p["correcao"]
    dispersao = p["dispersao"]
    # **`mod_composicao` e nao `composicao`:** ha uma variavel local com esse nome mais abaixo
    # — a instancia de `ComposicaoDasClasses` — e a primeira redacao desta divisao deixou o
    # modulo ser SOMBREADO por ela. O desfecho foi `AttributeError: 'ComposicaoDasClasses'
    # object has no attribute 'recorte_da_composicao_por_classe'`, e quem o apanhou foi esta
    # sonda a correr contra o congelado, na segunda volta.
    mod_composicao = p["composicao"]
    classificacao = p["classificacao"]
    # **`mod_rede` e nao `rede`:** ha uma variavel local com esse nome mais abaixo — a instancia de
    # `Rede` —, e deixar o modulo ser SOMBREADO por ela e o defeito que a divisao de 17/09/2026 ja
    # pagou uma vez, com `AttributeError` na segunda volta da sonda.
    mod_rede = p["rede"]
    mod_curva = p["curva_do_loess"]

    # --- deriva ------------------------------------------------------------
    # As sete injeções de QC dos quatro arquivos deste estudo, em lote único. São OBSERVAÇÃO sobre
    # aqueles arquivos e não invariante — aqui servem só de forma plausível.
    ordens = (1, 12, 23, 34, 45, 56, 68)
    scores = (-12.0, -8.5, -4.0, 1.5, 6.0, 11.5, 18.0)

    # --- distribuição ------------------------------------------------------
    def caixa(nome: str, centro: float, largura: float) -> dict[str, Any]:
        return {
            "feature": nome,
            "amostra": nome,
            "bigode_inferior": centro - 2.0 * largura,
            "q1": centro - largura,
            "mediana": centro,
            "q3": centro + largura,
            "bigode_superior": centro + 2.0 * largura,
            "n_discrepantes": 1,
        }

    nomes = tuple(f"1.{i}_10{i}.5m/z" for i in range(1, 7))
    brutas_antes = [caixa(n, 10.0 + i, 2.0 + 0.3 * i) for i, n in enumerate(nomes)]
    brutas_depois = [caixa(n, 10.0 + i, 1.0 + 0.2 * i) for i, n in enumerate(nomes)]
    caixas_antes = dist.caixas_do_fio(brutas_antes)
    caixas_depois = dist.caixas_do_fio(brutas_depois)
    por_amostra_antes = dist.caixas_por_amostra_do_fio(brutas_antes)
    por_amostra_depois = dist.caixas_por_amostra_do_fio(brutas_depois)
    medias_antes = [9.5, 10.2, 11.0, 11.8, 12.6, 13.1]
    medias_depois = [10.0, 10.4, 10.9, 11.3, 11.7, 12.0]
    matriz_antes = "none"
    matriz_depois = "serrf"

    recorte_caixas = dist.recorte_das_caixas(
        n_caixas=len(nomes),
        n_comparadas=len(nomes),
        n_antes=120,
        amostrada=True,
        seed=20260911,
        teto_de_caixas=len(nomes),
        coeficiente_de_tukey=1.5,
        matriz_antes=matriz_antes,
        matriz_depois=matriz_depois,
    )
    recorte_caixas_amostra = dist.recorte_das_caixas_por_amostra(
        n_caixas=len(nomes),
        n_amostras_comparadas=len(nomes),
        n_comparadas=len(nomes),
        n_antes=120,
        amostrada=False,
        seed=20260911,
        teto_de_caixas=len(nomes),
        coeficiente_de_tukey=1.5,
        matriz_antes=matriz_antes,
        matriz_depois=matriz_depois,
    )

    # --- volcano -----------------------------------------------------------
    nomes_v = tuple(f"2.{i}_2{i}0.1m/z" for i in range(1, 9))
    pontos = volcano.pontos_do_volcano(
        features=list(nomes_v),
        log2fc=[-2.4, -1.1, -0.2, 0.1, 0.7, 1.6, 2.9, 0.4],
        pvalue=[1e-6, 2e-4, 0.30, 0.80, 0.05, 3e-5, 1e-7, 0.44],
        qvalue_fdr=[1e-5, 9e-4, 0.42, 0.88, 0.11, 2e-4, 1e-6, 0.55],
        n_features_testadas=len(nomes_v),
        alfa=0.05,
    )
    contraste = volcano.contraste_do_fio(
        grupo_a="controle",
        grupo_b="tratado",
        n_grupo_a=6,
        n_grupo_b=6,
        parametrico=True,
        alfa_fdr=0.05,
        correcao="benjamini_hochberg",
        rota_da_normalidade="forced",
        shapiro_pvalue=None,
        n_residuos=None,
        matriz_do_pvalue="serrf",
        matriz_do_fold_change="none",
        matrizes_diferem=True,
    )

    # --- as duas univariadas -----------------------------------------------
    # **Os MESMOS vetores do volcano, e a partilha é o ponto.** As tres figuras leem a mesma raiz e
    # desenham o mesmo `UnivariateResult`; alimentar a sonda com valores diferentes faria os tres
    # desenhos deixarem de ser comparaveis entre si, que e exactamente o que a partilha da raiz
    # existe para garantir do lado do produto.
    fold_change = univariada.pontos_do_fold_change(
        features=list(nomes_v),
        log2fc=[-2.4, -1.1, -0.2, 0.1, 0.7, 1.6, 2.9, 0.4],
        qvalue_fdr=[1e-5, 9e-4, 0.42, 0.88, 0.11, 2e-4, 1e-6, 0.55],
        n_features_testadas=len(nomes_v),
        alfa=0.05,
    )
    testes_t = univariada.pontos_dos_testes_t(
        features=list(nomes_v),
        pvalue=[1e-6, 2e-4, 0.30, 0.80, 0.05, 3e-5, 1e-7, 0.44],
        qvalue_fdr=[1e-5, 9e-4, 0.42, 0.88, 0.11, 2e-4, 1e-6, 0.55],
        n_features_testadas=len(nomes_v),
        alfa=0.05,
    )

    # --- a ANOVA de N grupos -----------------------------------------------
    # **QUATRO grupos, que e a forma deste estudo, e nao dois.** O rodape da ANOVA carrega um par
    # `(nome, n)` por grupo, e e o rodape mais cheio do pacote: com dois grupos a sonda desenharia
    # uma frase mais curta do que a que o produto produz, e `recorte.aplica` nunca chegaria perto do
    # teto de `RESERVA_MAXIMA` que ela existe para exercitar.
    contexto_anova = anova.contexto_do_fio(
        grupos=["ATLL", "Asymptomatic", "HAMTSP", "NoHTLV"],
        n_por_grupo=[13, 18, 14, 16],
        metodo="anova",
        post_hoc_metodo="tukey",
        post_hoc_rota="ran",
        alfa_fdr=0.05,
        correcao="benjamini_hochberg",
        matriz="processada",
        min_por_grupo=2,
    )
    pontos_anova = anova.pontos_da_anova(
        features=list(nomes_v),
        pvalue=[1e-6, 2e-4, 0.30, 0.80, 0.05, 3e-5, 1e-7, 0.44],
        qvalue_fdr=[1e-5, 9e-4, 0.42, 0.88, 0.11, 2e-4, 1e-6, 0.55],
        n_features_testadas=len(nomes_v),
        alfa=0.05,
    )

    # --- PCA ---------------------------------------------------------------
    # Os tipos saem do próprio `StrEnum` em vez de cadeias escritas aqui: um valor renomeado no
    # domínio faria a sonda desenhar uma série vazia, e a figura sairia verde e errada.
    tipos_possiveis = [str(t) for t in pca.TipoDeAmostra]
    tipos = [tipos_possiveis[i % len(tipos_possiveis)] for i in range(12)]
    pc1 = [-4.1, -3.2, -2.0, -1.1, -0.3, 0.4, 1.0, 1.9, 2.6, 3.3, 4.0, 4.8]
    pc2 = [1.2, -0.8, 0.5, 2.1, -1.4, 0.2, -2.0, 1.7, -0.6, 0.9, -1.9, 0.3]
    escores = pca.escores_da_pca(pc1=pc1, pc2=pc2, tipos=tipos)
    nomes_c = tuple(f"3.{i}_3{i}0.9m/z" for i in range(1, 10))
    cargas = pca.cargas_da_pca(
        pc1=[-0.41, -0.22, -0.05, 0.03, 0.14, 0.26, 0.33, 0.45, 0.52],
        pc2=[0.18, -0.31, 0.09, 0.27, -0.12, 0.36, -0.24, 0.05, -0.40],
        features=list(nomes_c),
    )
    variancia = [0.41, 0.23, 0.12, 0.08, 0.05]
    regiao = pca.elipse_de_hotelling(escores.x, escores.y)
    n_setas = 3

    # **O par PERMANOVA / dispersão vai com valor NÃO PADRÃO.** O parâmetro é
    # anulável com padrão `None`, e `None` é precisamente o valor que executa **menos** do caminho:
    # ele escreve uma frase de ausência e nunca formata número nenhum. Uma sonda que o deixasse no
    # padrão renderizaria a figura de scores com o rodapé antigo, e a leitura humana — a
    # razão de esta sonda existir — nunca veria o acréscimo.
    #
    # **Os quatro grupos com as contagens do estudo são deliberados:** é o comprimento desta frase
    # que empurra o rodapé contra `RESERVA_MAXIMA`, e é ele que se quer ver desenhado.
    par_da_permanova = pca.Permanova(
        estatistica=0.96631,
        pvalue=0.432,
        estatistica_da_dispersao=1.2443,
        pvalue_da_dispersao=0.285,
        permutacoes=999,
        semente=42,
        grupos=("ATLL", "Asymptomatic", "HAMTSP", "NoHTLV"),
        n_por_grupo=(13, 18, 14, 16),
    )

    # --- PLS-DA ------------------------------------------------------------
    # **Q² baixo de propósito.** É o número que o PLS-DA nunca pode deixar de pôr na figura, e
    # noutro estudo do laboratório o PLS-DA mostrou separação convincente com Q² de ~24%.
    validacao = plsda.validacao_do_fio(
        q2=0.24,
        permutation_pvalue=0.31,
        r2y=0.88,
        n_permutacoes=200,
        cv_folds=5,
        n_componentes=2,
    )
    grupos = ["controle" if i % 2 == 0 else "tratado" for i in range(12)]
    escores_pls = plsda.escores_do_plsda(
        comp1=pc1,
        comp2=pc2,
        grupos=grupos,
        grupo_positivo="tratado",
        grupo_negativo="controle",
        validacao=validacao,
    )
    cargas_pls = plsda.cargas_do_plsda(
        comp1=[-0.38, -0.19, -0.02, 0.07, 0.16, 0.29, 0.31, 0.44, 0.50],
        comp2=[0.21, -0.28, 0.11, 0.25, -0.15, 0.34, -0.22, 0.08, -0.37],
        features=list(nomes_c),
        validacao=validacao,
    )
    importancia = plsda.vip_do_plsda(
        vip=[2.4, 1.9, 1.6, 1.4, 1.2, 1.0, 0.8, 0.6, 0.3],
        features=list(nomes_c),
        validacao=validacao,
    )

    # --- PLS-DA multiclasse -------------------------------------------------
    # **Tres classes e nao quatro, e a escolha e do EXERCICIO.** A coorte real tem quatro, mas
    # `COLUNAS_DA_GRELHA` e 2: com K impar a ultima linha fica com um painel so, que e o ramo da
    # grelha que um K par nunca percorre. A sonda mede se o motor de desenho da pasta entregue
    # desenha, e o ramo menos percorrido e o que mais vale exercitar.
    classes = ("controle", "tratado", "referencia")
    grupos_mc = [classes[i % len(classes)] for i in range(12)]
    # **Q2 diferente por modelo, de proposito.** O rodape lista os K pares e a contagem de
    # vereditos; com os tres iguais a frase sairia a mesma e a sonda nao veria a diferenca.
    validacoes = [
        plsda.validacao_do_fio(
            q2=q2,
            permutation_pvalue=pv,
            r2y=0.88,
            n_permutacoes=200,
            cv_folds=5,
            n_componentes=2,
        )
        for q2, pv in ((0.24, 0.31), (-0.11, 0.62), (0.05, 0.44))
    ]
    escores_mc = [
        multiclasse.escores_do_modelo(
            grupo=classe,
            comp1=[v + indice for v in pc1],
            comp2=[v - indice for v in pc2],
            grupos=list(grupos_mc),
            validacao=val,
        )
        for indice, (classe, val) in enumerate(zip(classes, validacoes, strict=True))
    ]
    cargas_mc = [
        multiclasse.cargas_do_modelo(
            grupo=classe,
            comp1=[-0.38, -0.19, -0.02, 0.07, 0.16, 0.29, 0.31, 0.44, 0.50 + indice],
            comp2=[0.21, -0.28, 0.11, 0.25, -0.15, 0.34, -0.22, 0.08, -0.37 - indice],
            features=list(nomes_c),
            validacao=val,
        )
        for indice, (classe, val) in enumerate(zip(classes, validacoes, strict=True))
    ]
    importancias_mc = [
        multiclasse.importancia_do_modelo(
            grupo=classe,
            vip=[2.4 + indice, 1.9, 1.6, 1.4, 1.2, 1.0, 0.8, 0.6, 0.3],
            features=list(nomes_c),
            validacao=val,
        )
        for indice, (classe, val) in enumerate(zip(classes, validacoes, strict=True))
    ]
    por_painel = 4

    # --- agrupamento hierarquico -------------------------------------------
    # **A ligacao e ESCRITA A MAO e nao calculada por `scipy`.** A sonda corre dentro da pasta
    # congelada e mede o motor de DESENHO; construir a ligacao aqui traria `scipy.cluster` para
    # dentro da medicao e faria um defeito de calculo parecer defeito de desenho. Os quatro campos
    # sao os do formato do `scipy` — os dois filhos, a altura e o numero de folhas do no —, e a
    # ligacao abaixo junta (0,1), (2,3) e depois os dois nos, que e a arvore de quatro folhas mais
    # simples que tem dois ramos.
    ligacao = [[0.0, 1.0, 1.0, 2.0], [2.0, 3.0, 1.5, 2.0], [4.0, 5.0, 3.0, 4.0]]
    amostras_ag = ["ESTUDO_POS_A1", "ESTUDO_POS_A2", "ESTUDO_POS_B1", "ESTUDO_POS_B2"]
    grupos_ag = ["controle", "controle", "tratado", "tratado"]
    features_ag = list(nomes_c[:5])
    # Valores com sinal dos dois lados e assimetricos: uma matriz simetrica faria a transposta
    # desenhar a mesma imagem, e o digest distinto entre figuras deixaria de significar nada.
    valores_ag = [
        [-1.8, -1.1, 0.2, 0.9, 1.6],
        [-1.5, -0.9, 0.1, 1.1, 1.4],
        [1.7, 1.0, -0.3, -1.0, -1.5],
        [1.4, 0.8, -0.1, -1.2, -1.7],
    ]
    arvore = agrupamento.arvore_do_fio(
        ligacao=ligacao,
        rotulos=amostras_ag,
        metrica="euclidean",
        ligacao_metodo="ward",
    )
    mapa = agrupamento.mapa_do_fio(
        valores=valores_ag,
        amostras=amostras_ag,
        features=features_ag,
        grupos=grupos_ag,
        ordem_das_amostras=[0, 1, 2, 3],
        ordem_das_features=[4, 3, 2, 1, 0],
    )

    # --- o K-means e o mapa auto-organizavel ------------------------------
    # **Os dois rotulos e as duas unidades sao NAO-PADRAO no eixo que muda maquinaria, e a
    # cegueira que isso evita ja foi medida:** a sonda chamava o construtor sem o argumento
    # novo e teria renderizado a figura de ANTES. Aqui os eixos que mudam maquinaria sao dois, e
    # nenhum deles fica no valor de nascenca:
    #
    # 1. **o `k` e 3 e nao os 2 que a particao MINIMA teria** — com dois agrupamentos, um
    #    construtor que ignorasse `k` e o derivasse do maior rotulo desenharia o mesmo; e com um
    #    agrupamento VAZIO (o 3, abaixo), a frase contada do rodape corre o ramo de «uma parte sem
    #    amostra» em vez do de «nenhum agrupamento ficou vazio».
    # 2. **a grade e 2x3 e nao quadrada** — numa quadrada, trocar linha por coluna e uma mutacao
    #    que a figura absorve, e a sonda mediria um construtor que transpusesse a grade sem
    #    acusar.
    #
    # **O `k` e 3, e ele ja foi 4 — o que se segue e o historico da ida e da volta.**
    #
    # Uma versao anterior pos aqui `k=4` com esta razao escrita: com tres agrupamentos sobra UM
    # vazio, e `recorte.frase_contada` escrevia `as 1 partes da particao ...` — o plural
    # aritmetico que aquele modulo ja pagara no ramo do zero e que **nao tratava no ramo do um**.
    # Com dois vazios a frase saia correcta, e a lacuna ficou REPORTADA em vez de consertada ali.
    #
    # **Depois `frase_contada` ganhou o ramo do um, e com isso a escolha de evitar deixou de ser
    # necessaria — mas voltar a 3 nao e desfazer, e cobrir.** Com `k=4` a sonda corre o ramo
    # do PLURAL, que meia duzia de outras figuras ja correm; com `k=3` ela corre o ramo do
    # SINGULAR, que nasceu naquela mudanca e que nenhuma outra entrada desta tabela exercita. O
    # valor que estava aqui era o que executava menos do caminho novo.
    #
    # E o argumento original do `k` sobrevive intacto: 3 continua a ser **maior que 2**, entao um
    # construtor que ignorasse `k` e o derivasse do maior rotulo desenhado continua a ser apanhado.
    rotulos_km = [0, 0, 1, 1]
    particao = agrupamento.kmeans_do_fio(
        rotulos=rotulos_km,
        grupos=grupos_ag,
        k=3,
        inercia=12.5,
    )
    # As quatro amostras em TRES unidades de seis, para o rodape declarar unidades vazias — o
    # ramo que uma grade cheia nunca exercitaria.
    grade = agrupamento.som_do_fio(
        unidades=[0, 0, 3, 5],
        grupos=grupos_ag,
        linhas=2,
        colunas=3,
        epocas=100,
        erro_de_quantizacao=44.69,
    )

    # --- os dois classificadores -------------------------------------------
    # **OS DOIS EIXOS QUE MUDAM MAQUINARIA ESTAO EM VALOR NAO-PADRAO, e nenhum deles e o
    # numero que so muda um numero:**
    #
    # 1. **o NOME**, que e o eixo do despacho. `figura_do_random_forest` e `figura_do_svm`
    #    desenham o MESMO traco, e o que os separa e a guarda `_exige_o_nome`. A sonda monta as
    #    duas cargas com nomes DIFERENTES e manda cada uma ao seu construtor: com as duas iguais,
    #    um construtor que ignorasse a guarda desenharia as duas e nada acusaria.
    # 2. **o RECALL com um ramo de ausencia**, que e o eixo do desenho. A carga do SVM tem uma
    #    classe com recall ZERO, e o construtor NAO desenha barra para ela — e a de dois zeros faz
    #    `frase_contada` correr o ramo do plural em vez do de «nenhuma». Com quatro barras cheias,
    #    o ramo que a figura de um resultado negativo percorre nunca seria exercitado, e este
    #    corpus e justamente o que produz resultados negativos.
    #
    # **OS NUMEROS SAO OS MEDIDOS NO `pos_hilic` REAL, em 18/09/2026**, e nao inventados: 61
    # amostras de estudo, 6.031 features utilizáveis de 6.035, quatro classes de 13/18/14/16, com
    # `classifier_folds=5` e 200 permutacoes. A floresta custou 288,1 s e o SVM 6,3 s.
    grupos_cl = ["ATLL", "Asymptomatic", "HAMTSP", "NoHTLV"]
    contagens_cl = [13, 18, 14, 16]
    floresta = classificacao.classificacao_do_fio(
        nome="Random Forest",
        grupos=grupos_cl,
        contagens=contagens_cl,
        recall=[0.23076923076923078, 0.5, 0.21428571428571427, 0.375],
        balanceada=0.3300137362637363,
        simples=0.3442622950819672,
        macro_f1=0.32684865444153366,
        p=0.11940298507462686,
        acaso=0.25,
        cv_folds=5,
        n_permutacoes=200,
        # **Uma semente de dez digitos de proposito**, como a do K-means: e o comprimento das
        # sementes DERIVADAS de `(seed, unidade)`, e e sobre um numero desses que a separacao de
        # milhar do rodape e visivel. A sonda mede o que a figura ESCREVE.
        semente=1921827941,
    )
    maquina = classificacao.classificacao_do_fio(
        nome="SVM",
        grupos=grupos_cl,
        contagens=contagens_cl,
        # A terceira classe tem recall ZERO — o SVM nunca acertou uma `HAMTSP` no corpus real —, e
        # a quarta esta abaixo do acaso. E a carga que exercita o ramo de «classe sem barra».
        recall=[0.07692307692307693, 0.3333333333333333, 0.0, 0.1875],
        balanceada=0.14943910256410256,
        simples=0.16393442622950818,
        macro_f1=0.14978070175438596,
        p=0.9701492537313433,
        acaso=0.25,
        cv_folds=5,
        n_permutacoes=200,
        semente=979431272,
    )

    # --- a correlacao e a busca por padrao ---------------------------------
    # **A matriz e ASSIMETRICA fora da diagonal? NAO — uma correlacao e simetrica por definicao**,
    # e e por isso que a planta desta figura e `planta_conteudo_trocado` e nao a transposta: a
    # transposta de uma matriz de correlacao e ela propria. O que a sonda mede aqui e que o
    # construtor desenha, e os valores sao escolhidos com os dois sinais para a paleta divergente
    # exercitar as duas metades da escala.
    valores_corr = [
        [1.0, 0.82, -0.35, 0.10],
        [0.82, 1.0, -0.60, 0.05],
        [-0.35, -0.60, 1.0, -0.20],
        [0.10, 0.05, -0.20, 1.0],
    ]
    features_corr = list(nomes_c[:4])
    mapa_corr = correlacao.mapa_de_correlacao_do_fio(
        valores=valores_corr,
        features=features_corr,
        # **`spearman` e NAO o padrao `pearson`, de proposito.** O eixo do metodo
        # muda MAQUINARIA do lado do desenho — `frase_do_coeficiente` faz uma busca num registo
        # fechado e recusa o que nao esta la —, e passar sempre o valor de nascenca deixaria a
        # sonda a exercitar o unico ramo que ja estava exercitado.
        metodo="spearman",
    )
    busca = correlacao.busca_do_fio(
        features=features_corr,
        # Os dois sinais, para as DUAS series por sinal serem desenhadas: com um lado so, a sonda
        # mediria uma figura com metade dos artistas que o construtor sabe pôr.
        coeficiente=[1.0, 0.61, -0.44, -0.88],
        referencia=features_corr[0],
        tipo_da_referencia="feature",
        perfil=[0.5, 1.5],
        ordem_dos_grupos=["controle", "tratado"],
        metodo="spearman",
    )

    # --- a rede de correlacao parcial --------------------------------------
    # **TRES EIXOS EM VALOR NAO-PADRAO, e nenhum deles e o numero que so muda um numero:**
    #
    # 1. **os DOIS SINAIS de correlacao parcial**, que sao o eixo dos artistas. O construtor monta
    #    uma `LineCollection` por sinal e SALTA a que ficar vazia; com um lado so, a sonda mediria
    #    uma figura com metade dos feixes que o construtor sabe pôr.
    # 2. **VERTICES ISOLADOS**, que sao o eixo do rodape. Com todos os vertices ligados,
    #    `frase_contada` corre o ramo de «nenhuma» e a frase que declara o isolamento nunca e
    #    escrita.
    # 3. **uma aresta ABAIXO do alfa no `q` e outras acima**, que e o eixo do veredito. Os tres
    #    ramos de `veredito_da_rede` sao alcancaveis, e este e o unico que escreve NUMERO.
    #
    # **O que esta sonda NAO exercita, e vai declarado:** o ramo da rede VAZIA — `_declara` dentro
    # do eixo, e os ramos de «nenhuma» de `frase_contada`. Ele e o desfecho normal deste corpus e
    # esta coberto em `tests/unit/test_plots_rede.py`; aqui a carga rica e a que desenha mais
    # artistas, que e o que uma sonda de RENDERIZACAO mede.
    #
    # Os oito nomes sao os mesmos `nomes_c` das outras familias, para a sonda nao inventar um
    # vocabulario proprio de feature.
    features_rede = list(nomes_c[:8])
    rede = mod_rede.rede_do_fio(
        features=features_rede,
        arestas=[
            {
                "feature_a": features_rede[0],
                "feature_b": features_rede[1],
                "correlacao_parcial": 0.62,
                "pvalue": 0.0008,
                "qvalue_fdr": 0.021,
            },
            {
                "feature_a": features_rede[1],
                "feature_b": features_rede[2],
                "correlacao_parcial": -0.48,
                "pvalue": 0.0041,
                "qvalue_fdr": 0.062,
            },
            {
                "feature_a": features_rede[2],
                "feature_b": features_rede[4],
                "correlacao_parcial": 0.31,
                "pvalue": 0.0192,
                "qvalue_fdr": 0.204,
            },
        ],
        # **Tres, e vem do lado do NUCLEO e nao de `len` da lista acima.** Os dois numeros saem de
        # caminhos diferentes, e a guarda do construtor compara-os — passar `len(arestas)` aqui
        # apagaria o segundo caminho dentro da propria sonda.
        n_arestas=3,
        n_pares_testados=28,
        alfa=0.05,
        penalidade=0.2431,
        n_amostras_usadas=61,
        # **Acentuada, e a cópia sem acento era DEFEITO — achado ao renderizar e olhar.** Esta
        # frase é a única prosa desta tabela que aterra **na imagem**: `Recorte.metodo` vai para o
        # rodapé, e o rodapé é o que um humano lê ao conferir o congelado. A redação anterior dizia
        # *"validacao cruzada"* e *"correcao de vies"* no meio de uma frase cujo resto tinha acentos
        # — *"sobrevivência"*, *"critério"*, *"correção de múltiplos testes"* —, e o contraste só
        # aparece na figura montada, nunca no diff.
        #
        # **A fonte é `api/rotas.py`, e esta é uma CÓPIA.** Lá a frase está acentuada; aqui
        # ela foi redigitada e perdeu os acentos no caminho. Mantê-las em sincronia é obrigação de
        # quem editar uma das duas — e uma varredura mediu que esta é a **única** prosa da
        # tabela nessa condição: as outras 39 palavras sem acento do arquivo são identificadores
        # (`metodo="spearman"`, `populacao="sample"`, nomes de módulo), que são ASCII por convenção.
        metodo=(
            "graphical lasso com penalidade por validação cruzada, mais a correção de viés de "
            "Jankova e van de Geer (2015) — o passo DSPC de Basu et al. (2017)"
        ),
    )

    # --- o painel da correcao ---------------------------------------------
    # **Os tres numeros sao os do PNG que o laboratorio ja le** — o `pos_hilic`, com `none` a
    # 144,7%, `batch_loess` a 105,0% e `SERRF` a 69,3%. Sao OBSERVACAO sobre aquele arquivo e nao
    # invariante; aqui servem de forma plausivel, e a fila com a medicao **presente** e o
    # caminho que exercita mais codigo — sem ela o construtor desenha o texto declarado e as barras
    # nao correm (o valor padrao e o que executa menos do caminho).
    par_da_correcao = correcao.par_da_correcao(
        antes=correcao.lado_da_correcao(
            rotulo=correcao.ROTULO_ANTES,
            matriz="ingerida",
            pc1=pc1,
            pc2=pc2,
            tipos=tipos,
            variancia_explicada=variancia,
            qc_dispersion_ratio=0.977,
            n_features_usadas=9,
            n_features_totais=120,
            digest_das_colunas_retidas="3151a4a6" * 8,
        ),
        depois=correcao.lado_da_correcao(
            rotulo=correcao.ROTULO_DEPOIS,
            matriz="corrigida",
            pc1=[v + 1.5 for v in pc1],
            pc2=[v * 0.6 for v in pc2],
            tipos=tipos,
            variancia_explicada=[0.245, 0.081],
            qc_dispersion_ratio=0.2329,
            n_features_usadas=9,
            n_features_totais=120,
            digest_das_colunas_retidas="3151a4a6" * 8,
        ),
        pretransformacao="none",
        n_componentes=2,
        # **O `corretor` FALTAVA aqui, e a sonda quebrava com `TypeError` antes de desenhar
        # figura nenhuma**. O parametro entrou sem valor por omissao em `par_da_correcao` e
        # esta chamada nao foi tocada; nada acusou, porque esta sonda **nao corre no `pytest`**.
        # E a cegueira nao era so desta figura: `_entradas` monta a tabela inteira
        # numa expressao, entao o `TypeError` levava as catorze consigo.
        #
        # **`loess` e nao `none`:** com `none` o rotulo do eixo
        # direito nao se distingue do `raw data` do esquerdo, e a leitura humana da prova ficaria
        # sobre dado em que as duas coisas nao diferem.
        corretor="loess",
    )
    fila_da_correcao = correcao.fila_do_fio(
        scores={"none": 1.447, "batch_loess": 1.050, "SERRF": 0.693},
        vencedor="SERRF",
        abstem=False,
        estimador="leave_one_qc_out",
        variante="outlier_removed",
        n_features_medidas=200,
    )

    # --- o violino de RSD por feature -------------------------------------
    # **Os dois lados levam contagens DIFERENTES e nenhuma coincide com contagem de injecao.** A
    # figura declara o `n` de cada lado, em features, e isso impede a prova de passar com o mesmo
    # numero escrito duas vezes. Os QCs dispersam menos que as amostras, que e a leitura que a
    # figura existe para dar.
    par_do_rsd = dispersao.par_do_rsd(
        qcs=dispersao.lado_do_rsd(
            rotulo=dispersao.ROTULO_DOS_QCS,
            valores=[0.08, 0.10, 0.11, 0.12, 0.13, 0.15, 0.17, 0.19, 0.24],
            n_features_finitas=9,
            n_injecoes=7,
        ),
        amostras=dispersao.lado_do_rsd(
            rotulo=dispersao.ROTULO_DAS_AMOSTRAS,
            valores=[0.21, 0.27, 0.30, 0.34, 0.38, 0.43, 0.51, 0.66],
            n_features_finitas=8,
            n_injecoes=61,
        ),
        matriz="corrigida",
        corretor="loess",
        variante="outlier_removed",
        estimador="in_sample",
        n_features=12,
    )

    # --- a composicao por classe -------------------------------------------
    # **Quatro classes com contagens DIFERENTES em tudo**, e uma delas SEM celula detectada: e o
    # desfecho em que a barra de abundancia nao existe e a ausencia vai declarada, e e o que
    # torna a prova humana capaz de o ver no executavel congelado.
    def classe(grupo: str, n: int, celulas: int, faltantes: int, log10: float | None) -> Any:
        return {
            "grupo": grupo,
            "n_injecoes": n,
            "n_celulas": celulas,
            "n_zeros": faltantes,
            "n_nao_finitas": 0,
            "n_faltantes": faltantes,
            "n_detectadas": 0 if log10 is None else celulas - faltantes,
            "log10_mediana_das_detectadas": log10,
        }

    composicao = mod_composicao.composicao_do_fio(
        classes=[
            classe("ATLL", 13, 78455, 12930, 4.0507),
            classe("Asymptomatic", 18, 108630, 17193, 4.0280),
            classe("HAMTSP", 14, 84490, 12058, 4.2031),
            classe("NoHTLV", 16, 96560, 96560, None),
        ],
        matriz="ingerida",
        populacao="sample",
        n_features=6035,
        n_injecoes_fora=7,
        definicao_de_faltante="zero_ou_nao_finita",
        definicao_de_abundancia="mediana_do_log10_das_detectadas",
    )

    # --- curva do LOESS -------------------------------------------------
    # Uma corrida de 68 injecoes com os sete QCs dos quatro arquivos deste estudo, em lote unico:
    # forma plausivel, como as ordens da deriva, e nao invariante. A curva e as curvas sem
    # um QC sao sinteticas — o que se mede aqui e que o motor desenha, e nao o numero.
    n_inj = 68
    posicoes_qc = {0, 11, 22, 33, 44, 55, 67}
    curva_y = [1000.0 + 4.0 * i + 60.0 * math.sin(i / 9.0) for i in range(n_inj)]
    medido = [y * (1.0 + 0.03 * math.cos(1.7 * i)) for i, y in enumerate(curva_y)]
    sem_o_qc = [
        curva_y[i] * (1.0 + 0.02 * math.sin(2.3 * i)) if i in posicoes_qc else None
        for i in range(n_inj)
    ]
    largura = [12.0 + (25.0 if i < 11 or i > 55 else 0.0) for i in range(n_inj)]
    curva_loess = mod_curva.curva_do_fio(
        feature="2.25_810.7739m/z",
        ordens_de_injecao=list(range(1, n_inj + 1)),
        lotes=["A"] * n_inj,
        qc=[i in posicoes_qc for i in range(n_inj)],
        valores=medido,
        curva=curva_y,
        curva_preenchida=[False] * n_inj,
        curva_sorteada=[False] * n_inj,
        qc_no_ajuste=[i in posicoes_qc for i in range(n_inj)],
        motivo_sem_curva=None,
        predicao_sem_o_qc=sem_o_qc,
        motivo_sem_predicao=[None] * n_inj,
        envelope_inferior=[y - w for y, w in zip(curva_y, largura, strict=True)],
        envelope_superior=[y + w for y, w in zip(curva_y, largura, strict=True)],
        envelope_nao_ajustado=[i < 11 or i > 55 for i in range(n_inj)],
        n_curvas_do_envelope=7,
        motivo_sem_envelope=None,
        semente=42,
        spans_por_lote=[0.85],
    )

    return {
        ("deriva", "figura_da_deriva_do_qc"): {
            "scores": scores,
            "ordens_de_injecao": ordens,
            "lotes": ("A",) * len(ordens),
            "recorte": deriva.recorte_da_deriva(n_features_usadas=40, n_features_totais=100),
        },
        ("distribuicao", "figura_das_caixas_por_feature"): {
            "antes": caixas_antes,
            "depois": caixas_depois,
            "recorte": recorte_caixas,
            "matriz_antes": matriz_antes,
            "matriz_depois": matriz_depois,
        },
        ("distribuicao", "figura_das_caixas_por_amostra"): {
            "antes": por_amostra_antes,
            "depois": por_amostra_depois,
            "recorte": recorte_caixas_amostra,
            "matriz_antes": matriz_antes,
            "matriz_depois": matriz_depois,
        },
        ("distribuicao", "figura_da_densidade_por_feature"): {
            "antes": medias_antes,
            "depois": medias_depois,
            "recorte": dist.recorte_da_densidade_por_feature(
                n_comparadas=len(medias_antes),
                n_antes=120,
                matriz_antes=matriz_antes,
                matriz_depois=matriz_depois,
            ),
            "matriz_antes": matriz_antes,
            "matriz_depois": matriz_depois,
        },
        ("distribuicao", "figura_da_densidade_por_amostra"): {
            "antes": medias_antes,
            "depois": medias_depois,
            "recorte": dist.recorte_da_densidade_por_amostra(
                n_comparadas=len(medias_antes),
                n_antes=120,
                n_amostras=len(medias_antes),
                matriz_antes=matriz_antes,
                matriz_depois=matriz_depois,
            ),
            "matriz_antes": matriz_antes,
            "matriz_depois": matriz_depois,
        },
        ("volcano", "figura_do_volcano"): {
            "pontos": pontos,
            "contraste": contraste,
            "recorte": volcano.recorte_do_volcano(pontos, contraste),
        },
        ("univariada", "figura_do_fold_change"): {
            "pontos": fold_change,
            "contraste": contraste,
            "recorte": univariada.recorte_do_fold_change(fold_change, contraste),
        },
        ("univariada", "figura_dos_testes_t"): {
            "pontos": testes_t,
            "contraste": contraste,
            "recorte": univariada.recorte_dos_testes_t(testes_t, contraste),
        },
        ("anova", "figura_da_anova"): {
            "pontos": pontos_anova,
            "contexto": contexto_anova,
            "recorte": anova.recorte_da_anova(pontos_anova, contexto_anova),
        },
        ("pca", "figura_do_scree"): {
            "variancia_explicada": variancia,
            "recorte": pca.recorte_do_scree(n_features_usadas=9, n_features_totais=120),
        },
        ("pca", "figura_dos_scores"): {
            "escores": escores,
            "variancia_explicada": variancia,
            "regiao": regiao,
            "recorte": pca.recorte_dos_scores(
                n_features_usadas=9,
                n_features_totais=120,
                regiao=regiao,
                permanova=par_da_permanova,
            ),
        },
        ("pca", "figura_das_cargas"): {
            "cargas": cargas,
            "variancia_explicada": variancia,
            "recorte": pca.recorte_das_cargas(cargas=cargas, n_features_totais=120),
        },
        ("pca", "figura_do_biplot"): {
            "escores": escores,
            "cargas": cargas,
            "n_setas": n_setas,
            "variancia_explicada": variancia,
            "recorte": pca.recorte_do_biplot(
                escores=escores, cargas=cargas, n_setas=n_setas, n_features_totais=120
            ),
        },
        ("plsda", "figura_dos_scores_do_plsda"): {
            "escores": escores_pls,
            "recorte": plsda.recorte_dos_scores_do_plsda(
                escores=escores_pls, n_features_totais=120
            ),
        },
        ("plsda", "figura_das_cargas_do_plsda"): {
            "cargas": cargas_pls,
            "recorte": plsda.recorte_das_cargas_do_plsda(cargas=cargas_pls, n_features_totais=120),
        },
        ("plsda", "figura_do_vip"): {
            "importancia": importancia,
            "recorte": plsda.recorte_do_vip(importancia=importancia, n_desenhadas=9),
        },
        ("plsda_multiclasse", "figura_dos_escores_um_contra_resto"): {
            "modelos": escores_mc,
            "recorte": multiclasse.recorte_dos_escores_um_contra_resto(
                modelos=escores_mc, n_features_totais=120
            ),
        },
        ("plsda_multiclasse", "figura_das_cargas_um_contra_resto"): {
            "modelos": cargas_mc,
            "recorte": multiclasse.recorte_das_cargas_um_contra_resto(
                modelos=cargas_mc, n_features_totais=120
            ),
        },
        ("plsda_multiclasse", "figura_da_importancia_um_contra_resto"): {
            "modelos": importancias_mc,
            "n_features": por_painel,
            "recorte": multiclasse.recorte_da_importancia_um_contra_resto(
                modelos=importancias_mc, n_desenhadas_por_modelo=por_painel
            ),
        },
        ("agrupamento", "figura_do_dendrograma"): {
            "arvore": arvore,
            "grupos": grupos_ag,
            "recorte": agrupamento.recorte_do_dendrograma(
                arvore=arvore,
                n_features_usadas=len(features_ag),
                n_features_totais=120,
                matriz="a matriz corrigida",
                amostragem="as 5 de maior variância",
            ),
        },
        ("agrupamento", "figura_do_mapa_de_calor"): {
            "mapa": mapa,
            "recorte": agrupamento.recorte_do_mapa_de_calor(
                mapa=mapa,
                metrica="euclidean",
                ligacao_metodo="ward",
                n_features_totais=120,
                matriz="a matriz corrigida",
                amostragem="as 5 de maior variância",
            ),
        },
        ("agrupamento", "figura_do_kmeans"): {
            "particao": particao,
            "recorte": agrupamento.recorte_do_kmeans(
                particao=particao,
                n_features_usadas=len(features_ag),
                n_features_totais=120,
                matriz="a matriz corrigida",
                amostragem="as 5 de maior variância",
                # **Uma semente de dez digitos de proposito**: e o comprimento das sementes
                # derivadas por `SeedSequence`, e e sobre um numero desses que a separacao de
                # milhar seria visivel no rodape. A sonda mede o que a figura ESCREVE.
                semente=1001697648,
            ),
        },
        ("agrupamento", "figura_do_mapa_auto_organizavel"): {
            "grade": grade,
            "recorte": agrupamento.recorte_do_mapa_auto_organizavel(
                grade=grade,
                n_features_usadas=len(features_ag),
                n_features_totais=120,
                matriz="a matriz corrigida",
                amostragem="as 5 de maior variância",
                semente=1172574093,
            ),
        },
        ("correlacao", "figura_do_mapa_de_correlacao"): {
            "mapa": mapa_corr,
            "recorte": correlacao.recorte_do_mapa_de_correlacao(
                mapa=mapa_corr,
                n_features_totais=120,
                n_amostras=61,
                n_indefinidas=2,
                matriz="a matriz corrigida",
                amostragem="as 4 sorteadas com semente 42",
            ),
        },
        ("correlacao", "figura_da_busca_por_padrao"): {
            "busca": busca,
            # **NAO-PADRAO de proposito:** o construtor nao tem valor de nascenca para este
            # eixo — ele e obrigatorio —, e o que a sonda escolhe e um numero MENOR que a populacao,
            # para o caminho do recorte correr. Com `n_features` acima da populacao, `_escolhidas`
            # devolveria a lista inteira e a selecao por modulo nunca seria exercitada.
            "n_features": 3,
            "recorte": correlacao.recorte_da_busca_por_padrao(
                busca=busca,
                n_desenhadas=3,
                n_features_totais=120,
                n_amostras=61,
                n_indefinidas=1,
                matriz="a matriz corrigida",
            ),
        },
        ("classificacao", "figura_do_random_forest"): {
            "classificacao": floresta,
            "recorte": classificacao.recorte_do_classificador(
                classificacao=floresta,
                n_features_usadas=6031,
                n_features_totais=6035,
                matriz="a matriz processada",
                amostragem="todas as 6.031 features utilizáveis, sem sorteio",
            ),
        },
        ("classificacao", "figura_do_svm"): {
            "classificacao": maquina,
            "recorte": classificacao.recorte_do_classificador(
                classificacao=maquina,
                n_features_usadas=6031,
                n_features_totais=6035,
                matriz="a matriz processada",
                amostragem="todas as 6.031 features utilizáveis, sem sorteio",
            ),
        },
        ("rede", "figura_da_rede_de_correlacao_parcial"): {
            "rede": rede,
            "recorte": mod_rede.recorte_da_rede(
                rede=rede,
                n_features_totais=120,
                matriz="a matriz corrigida",
                amostragem="as 8 sorteadas com semente 42",
            ),
        },
        ("correcao", "figura_do_painel_da_correcao"): {
            "par": par_da_correcao,
            "fila": fila_da_correcao,
            "recorte": correcao.recorte_do_painel_da_correcao(
                par=par_da_correcao, fila=fila_da_correcao
            ),
        },
        ("dispersao", "figura_do_violino_do_rsd_por_feature"): {
            "par": par_do_rsd,
            "recorte": dispersao.recorte_do_violino_do_rsd_por_feature(par=par_do_rsd),
        },
        ("composicao", "figura_da_composicao_por_classe"): {
            "composicao": composicao,
            "recorte": mod_composicao.recorte_da_composicao_por_classe(composicao=composicao),
        },
        ("curva_do_loess", "figura_da_curva_do_loess"): {
            "curva": curva_loess,
            "recorte": mod_curva.recorte_da_curva_do_loess(
                curva=curva_loess, n_features_totais=6035
            ),
        },
    }


def _desenha(figura: Any, canvas_qt: Any) -> dict[str, Any]:
    """Renderiza a figura e devolve o que ela tem DENTRO, não o facto de não ter levantado.

    O canvas é o `FigureCanvasQTAgg` — o mesmo que `PainelDeFigura` monta em
    `desktop/janela/figuras.py` —, e não o `Agg` que `plots.base.nova_figura` deixa attachado.
    **O símbolo e não o número de linha**: a versão anterior pinava `figuras.py:474`, que já não era
    a linha do canvas quando alguém foi lá ver. Com `Agg` a figura desenha e nunca
    aparece na janela: é o desfecho de sucesso com tela vazia, e trocá-lo aqui é a única forma de
    a sonda tocar no caminho que a janela usa.
    """
    import numpy as np

    canvas_qt(figura)
    figura.canvas.draw()
    buffer = np.asarray(figura.canvas.buffer_rgba())
    artistas = sum(
        len(eixo.lines) + len(eixo.collections) + len(eixo.patches) + len(eixo.images)
        for eixo in figura.axes
    )
    textos = [t.get_text() for t in figura.texts if t.get_text().strip()]
    for eixo in figura.axes:
        textos.extend(t.get_text() for t in [eixo.title, *eixo.texts] if t.get_text().strip())
    return {
        "eixos": len(figura.axes),
        "artistas_com_dado": artistas,
        "textos": len(textos),
        "tons": int(np.unique(buffer).size),
        "digest": hashlib.sha256(buffer.tobytes()).hexdigest()[:16],
        "canvas": type(figura.canvas).__name__,
    }


def _mede(relato: dict[str, Any], pasta: Path) -> dict[str, Any]:
    """Uma abertura inteira. Quem decide o código de saída é `main`.

    **O relato entra por argumento e não é criado aqui**, e isso é desenho: uma interrupção a
    meio — o `refino.plots` que não viajou, por exemplo — descartaria tudo o que já se mediu
    se o dicionário morresse com a exceção. O que se mediu antes de parar é justamente o que
    diz **onde** parou.
    """
    exe = pasta / "refino.exe"
    relato["exe"] = str(exe)
    if not exe.is_file():
        raise NaoDaParaMedirError(f"nao ha refino.exe em {pasta}")

    # A lista é ligada ao relato AGORA, e não no fim: as duas passam a ser o mesmo objeto, então
    # uma interrupção a meio deixa em `relato["falhas"]` tudo o que já se tinha achado.
    falhas: list[str] = []
    relato["falhas"] = falhas

    # (1) O CACHE DE FONTES, apontado ANTES do primeiro import do `matplotlib` — depois dele a
    #     variável já não é relida. A exigência de o diretório existir é o que valida a cópia da
    #     fórmula contra o comportamento do produto: quem o cria é o `refino.exe` ao arrancar.
    destino = cache_previsto(exe)
    relato["cache_apontado"] = str(destino)
    relato["cache_existia"] = destino.is_dir()
    if not destino.is_dir():
        falhas.append(
            f"o diretorio de cache previsto NAO existe: {destino}. Ou esta pasta nunca foi "
            "aberta pelo refino.exe, ou a derivacao de arranque.py mudou e esta copia caducou"
        )
    os.environ["MPLCONFIGDIR"] = str(destino)
    relato["cache_antes"] = _resumo_do_cache(destino)

    # (2) O BACKEND. `get_backend()` resolve o padrão; `Agg` aqui significa que o `PySide6` da
    #     pasta nao carregou, e a janela abriria vazia sem ninguem levantar.
    matplotlib = importlib.import_module("matplotlib")
    importlib.import_module("matplotlib.pyplot")  # é ele que força a resolução do padrão
    backend = str(matplotlib.get_backend())
    relato["backend"] = backend
    if "qt" not in backend.casefold():
        falhas.append(
            f"matplotlib.get_backend() devolveu {backend!r} e nao um backend Qt — com Agg a "
            "figura desenha e NAO aparece na janela"
        )

    qtagg = importlib.import_module("matplotlib.backends.backend_qtagg")
    qtwidgets = importlib.import_module("PySide6.QtWidgets")
    if qtwidgets.QApplication.instance() is None:
        relato["qapplication"] = "criada por esta sonda"
        qtwidgets.QApplication([])
    else:  # pragma: no cover - nao acontece num processo novo
        relato["qapplication"] = "ja existia"

    # (3) PROCEDÊNCIA. Sem isto, um `PYTHONPATH` herdado faria tudo o que está abaixo ser sobre o
    #     clone de quem corre a sonda, com desfecho verde.
    for modulo in (importlib.import_module("refino"), matplotlib, qtwidgets):
        defeito = _procedencia(modulo, pasta)
        if defeito:
            falhas.append(defeito)
    try:
        importlib.import_module(NAO_VIAJA)
    except BaseException:
        relato["nao_viaja"] = f"{NAO_VIAJA}: recusado, como tem de ser"
    else:
        falhas.append(
            f"{NAO_VIAJA} IMPORTOU — o sys.path alcancou algo fora de {pasta}, e esta sonda "
            "esta a medir a arvore errada"
        )

    # (4) A LISTA DE FIGURAS vem do REGISTO, e a bijeção contra a tabela de entradas corre nas
    #     duas direções.
    catalogo = importlib.import_module("refino.plots.catalogo")
    registados = tuple((c.modulo, c.funcao, c.vinculo, c.chave) for c in catalogo.CONSTRUTORES)
    relato["registados"] = len(registados)
    if not registados:
        raise NaoDaParaMedirError("refino.plots.catalogo.CONSTRUTORES esta VAZIO")

    modulos: dict[str, Any] = {}
    for nome in sorted({m for m, _, _, _ in registados}):
        modulos[nome] = importlib.import_module(f"refino.plots.{nome}")
        defeito = _procedencia(modulos[nome], pasta)
        if defeito:
            falhas.append(defeito)

    entradas = _entradas(modulos)
    chaves_registadas = {(m, f) for m, f, _, _ in registados}
    sem_entrada = sorted(chaves_registadas - set(entradas))
    sobrando = sorted(set(entradas) - chaves_registadas)
    if sem_entrada:
        falhas.append(
            "figura REGISTADA sem entrada nesta sonda, e por isso nao exercitada: "
            + ", ".join(f"{m}.{f}" for m, f in sem_entrada)
        )
    if sobrando:
        falhas.append(
            "esta sonda tem entrada para o que o registo nao declara: "
            + ", ".join(f"{m}.{f}" for m, f in sobrando)
        )

    # (5) DESENHA cada uma, e afirma pelo CONTEÚDO.
    desenhos: list[dict[str, Any]] = []
    for modulo, funcao, vinculo, chave in registados:
        item: dict[str, Any] = {
            "modulo": modulo,
            "funcao": funcao,
            "vinculo": str(vinculo),
            "chave": chave,
        }
        argumentos = entradas.get((modulo, funcao))
        if argumentos is None:
            item["erro"] = "sem entrada nesta sonda"
            desenhos.append(item)
            continue
        construtor = getattr(modulos[modulo], funcao, None)
        if construtor is None:
            item["erro"] = f"{funcao} nao existe em refino.plots.{modulo}"
            falhas.append(f"{modulo}.{funcao}: registado e AUSENTE do modulo congelado")
            desenhos.append(item)
            continue
        try:
            item.update(_desenha(construtor(**argumentos), qtagg.FigureCanvasQTAgg))
        except BaseException as erro:
            item["erro"] = f"{type(erro).__name__}: {erro}"
            falhas.append(f"{modulo}.{funcao} nao desenhou: {type(erro).__name__}: {erro}")
        else:
            if item["eixos"] < 1:
                falhas.append(f"{modulo}.{funcao}: figura sem eixo nenhum")
            if item["artistas_com_dado"] < 1:
                falhas.append(f"{modulo}.{funcao}: nenhum artista com dado")
            if item["tons"] < 8:
                falhas.append(
                    f"{modulo}.{funcao}: o buffer renderizado tem {item['tons']} tons — "
                    "uma imagem em branco tem 1"
                )
            if item["canvas"] != "FigureCanvasQTAgg":
                falhas.append(
                    f"{modulo}.{funcao}: o canvas montado foi {item['canvas']} e nao "
                    "FigureCanvasQTAgg"
                )
        desenhos.append(item)
    relato["desenhos"] = desenhos

    # (6) AS FIGURAS SAO DISTINTAS ENTRE SI. E a ancora contra quatro PNG identicos com
    #     `EXIT=0`. **A contagem e DERIVADA do registo e nao escrita aqui** — a versao anterior
    #     dizia «OS TREZE» e caducou quando entrou a decima quarta figura.
    digests = [d["digest"] for d in desenhos if "digest" in d]
    relato["digests_distintos"] = len(set(digests))
    relato["digests"] = {
        f"{d['modulo']}.{d['funcao']}": d["digest"] for d in desenhos if "digest" in d
    }
    if digests and len(set(digests)) != len(digests):
        falhas.append(
            f"{len(digests)} figuras renderizadas e so {len(set(digests))} imagens distintas — "
            "figuras iguais sao o desfecho que AR165 descreve"
        )

    # (7) OS TRÊS FORMATOS DA EXPORTAÇÃO. O PDF depende de `backend_pdf`, que o registro de
    #     formatos resolve por string, e o TIFF do plugin do Pillow: os dois têm de viajar, e só o
    #     congelado diz se viajaram. Grava e relê cada um, pelo mesmo `_grava` da janela.
    relato["exportacao"] = _prova_a_exportacao(falhas)

    relato["cache_depois"] = _resumo_do_cache(destino)
    relato["falhas"] = falhas
    relato["ok"] = not falhas
    return relato


def _prova_a_exportacao(falhas: list[str]) -> dict[str, Any]:
    """PNG, TIFF e PDF gravados a 600 ppp e relidos: assinatura, RGB do TIFF e fontes do PDF."""
    import tempfile

    figuras = importlib.import_module("refino.desktop.janela.figuras")
    base = importlib.import_module("refino.plots.base")
    imagem = importlib.import_module("PIL.Image")
    figura = base.nova_figura()
    eixo = figura.add_subplot(1, 1, 1)
    eixo.plot([1, 2, 3], [3, 1, 2], marker="o")
    eixo.set_xlabel("ordem de injeção")
    resultado: dict[str, Any] = {}
    with tempfile.TemporaryDirectory() as temporario:
        for sufixo, formato in ((".png", "png"), (".tif", "tiff"), (".pdf", "pdf")):
            arquivo = Path(temporario) / f"exportacao{sufixo}"
            try:
                figuras._grava(figura, arquivo, formato, 600.0)
            except Exception as erro:
                falhas.append(
                    f"exportar em {formato} falhou no congelado: {type(erro).__name__}: {erro}"
                )
                continue
            bruto = arquivo.read_bytes()
            item: dict[str, Any] = {"bytes": len(bruto), "cabeca": bruto[:4].hex()}
            if formato == "tiff":
                with imagem.open(arquivo) as relido:
                    item["modo"] = relido.mode
                    item["dpi"] = [float(v) for v in relido.info.get("dpi", (0, 0))]
                if item["modo"] != "RGB":
                    falhas.append(f"o TIFF saiu em {item['modo']} e nao em RGB")
            if formato == "pdf":
                item["tipo3"] = bruto.count(b"/Type3")
                item["fontfile2"] = bruto.count(b"/FontFile2")
                if item["tipo3"] or not item["fontfile2"]:
                    falhas.append("o PDF saiu com fonte Tipo 3 ou sem TrueType embutida")
            resultado[sufixo] = item
    return resultado


def main() -> int:
    analisador = argparse.ArgumentParser(description="Uma abertura das figuras no congelado.")
    analisador.add_argument("--pasta", required=True)
    analisador.add_argument("--saida", required=True)
    analisador.add_argument("--rotulo", default="unica")
    argumentos = analisador.parse_args()

    pasta = Path(argumentos.pasta).resolve()
    relato: dict[str, Any] = {"rotulo": argumentos.rotulo, "pasta": str(pasta)}
    try:
        _mede(relato, pasta)
        codigo = 0 if relato["ok"] else 1
    except BaseException as erro:
        motivo = (
            str(erro) if isinstance(erro, NaoDaParaMedirError) else f"{type(erro).__name__}: {erro}"
        )
        relato["ok"] = False
        relato["nao_deu_para_medir"] = motivo
        relato["falhas"] = [*relato.get("falhas", []), motivo]
        codigo = 2

    texto = json.dumps(relato, indent=2, ensure_ascii=False)
    Path(argumentos.saida).write_text(texto, encoding="utf-8")
    print(texto)
    return codigo


if __name__ == "__main__":
    raise SystemExit(main())
