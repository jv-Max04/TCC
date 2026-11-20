# -*- mode: python ; coding: utf-8 -*-
"""O spec do PyInstaller para o aplicativo de mesa.

## O que este arquivo decide, e cada decisão tem medição por trás

**Rodado no WINDOWS, e não em `make check`.** PyInstaller no WSL produz binário Linux, então este
spec não pode ser exercitado pelo portão de commit. O que o exercita é
`scripts/congela_windows.ps1`, corrido à mão, e o resultado vai na mensagem do commit.

## 1. `--onedir` e não `--onefile`

Uma pasta que se copia. `--onefile` descomprime num diretório temporário a cada execução, o que
custa segundos de arranque e põe o dado descomprimido num lugar que o antivírus varre.

## 2. `python_options` com `X utf8=1`

**O binário congelado IGNORA `PYTHONUTF8` do ambiente.** A code page ANSI desta máquina é `cp932`, e
prosa acentuada — que é toda a prosa deste projeto — levanta `UnicodeEncodeError` sem ele. A forma
`[("X", "utf8=1", "OPTION")]` na lista de opções do `EXE` é como um spec codifica
`--python-option X utf8=1`.

`arranque.garante_utf8` reconfigura os fluxos em runtime **e** isto entra: o primeiro cobre o que já
está aberto, e a opção do interpretador cobre o que abre depois. Nenhum dos dois é suficiente
sozinho, e a evidência é um defeito de arnês — uma sonda morreu no passo que
decide, com `UnicodeEncodeError: 'cp932' can't encode '\\xf3'`, medindo a recusa e não conseguindo
relatá-la.

## 3. `datas`: as MIGRAÇÕES viajam, o `alembic.ini` NÃO

Medido: o `Config` do alembic é construído **em memória** e o `script_location` sai do pacote —
`Path(persistence.__file__).parent / "migrations"`. O `alembic.ini` declara um caminho relativo à
raiz do repositório, que num congelado não existe, e por isso ele não é necessário.

O que **precisa** de viajar são os arquivos de `versions/`, porque o alembic os carrega por
**caminho** e não por `import`. Um `--add-data` que trouxesse só o `alembic.ini` embalaria o arquivo
inútil e deixaria de fora o que importa.

## 4. `excludes`: três pacotes, e a lista é MEDIDA e não adivinhada

A medição de 22/08/2026 mostrou o `--onedir` da CLI em **363,9 MB**, com `pyarrow` (78 MB),
`matplotlib` (13,7), `psycopg2` (6,2) e PySide6 (91,7) a entrar por **estarem na venv**. Daí duas
coisas, e a segunda é a que vale:

1. a venv do congelamento é construída para o ALVO — `congela_windows.ps1` instala o conjunto de
   runtime e o pacote com `--no-deps`;
2. `excludes` é o cinto, e cada nome foi conferido por varredura de import em `src`, `scripts` e
   `tests` — em 23/08/2026, com `import X` e `from X` no início da linha:

| Pacote | Sítios de import no repositório | Por que sai |
|---|---|---|
| `ruptures` | **zero** | nada no repositório o importa |
| `psycopg2` | **zero** | o SQLAlchemy carrega o driver pelo esquema da URL, e o aplicativo de mesa usa SQLite |
| `streamlit` | **zero** | a tela web saiu do repositório, e com ela o único sítio de import |

### 4b. `matplotlib` NÃO está nesta lista

`plots/` e `desktop/janela/figuras.py` importam `matplotlib` de propósito: a janela desenha com ele,
e três dos nove itens obrigatórios do relatório são gráficos.

**O custo vai declarado, e é o que `congela_windows.ps1` mede:** o pino entra em `$RuntimeDoAlvo`
**sem** `--no-deps`, então arrasta `contourpy`, `cycler`, `fonttools`, `kiwisolver`, `pillow`,
`pyparsing`, `python-dateutil` e `packaging`. Medidos em disco na venv de desenvolvimento, os nove
somam **51,4 MB em 1.267 arquivos**: o crescimento real é maior que os 13,7 MB do pacote sozinho, e
uma linha de log não o mostra a ninguém.

**E os TRÊS que NÃO saem, apesar de o fecho de importação não os ver:**

| Pacote | Onde | Por que fica |
|---|---|---|
| `alembic` | `desktop/banco.py`, em `migra`, **dentro de função**, mais o topo dos cinco módulos de migração — o `env.py` e os quatro de `versions\\` | a migração corre no arranque. Excluí-lo mataria o primeiro arranque |
| `httpx2` | `interface/cliente.py`, em `constroi_transporte`, **dentro de função** | é o cliente HTTP que a janela usa para falar com a API embutida |
| `matplotlib.backends.backend_qtagg` | resolvido **por string** em runtime, pelo registro de backends do próprio `matplotlib` | o analisador estático não vê nome nenhum. Sem ele o congelado cai para `Agg`, e `Agg` **desenha** — a figura sai correcta e não aparece na janela. Desfecho de sucesso com tela vazia, que é o pior modo de falha desta família |

> **Fecho de importação NÃO é fecho de runtime, e é por isso que a lista acima é de varredura de
> texto e não de `sys.modules`.** Medido: importar `refino.desktop.arranque`,
> `janela.principal`, `servidor` e `build_app()` deixa `alembic` e `httpx2` **fora** de
> `sys.modules`. Excluir por fecho de importação teria produzido um binário que arranca e morre na
> migração.

## 5. `pyarrow` não viaja, e os 78 MB dele não são piso

**Medido em 23/08/2026.** Na venv de **desenvolvimento** o `pyarrow` está instalado, e o `pandas`
importa-o **se ele existir**; por isso ele aparece no fecho de importação medido ali. Na venv do
alvo, construída com o conjunto de runtime explícito, o `pip` **não** o instala: ele não é
dependência obrigatória do `pandas 3.0.5`. Medido no congelado: `pyarrow` aparece **2** vezes no
`PYZ-00.toc`, e as duas são referência de dentro do `pandas`; o pacote não está lá, e o binário
funciona.

> **O fecho de importação depende da POPULAÇÃO.** *"Está no fecho de importação"* é verdade sobre a
> venv de desenvolvimento e falso sobre a venv do alvo — e a segunda é a que o congelamento usa.

O piso real, medido sobre a pasta produzida: **261 MB**, contra os 363,9 do `--onedir` da CLI de
22/08. Os maiores são PySide6 (92), scipy (47 mais 20 de `scipy.libs`), `numpy.libs` (21), sklearn
(14) e pandas (13).

## 6. Como se confere que um pacote VIAJOU, e por que a pergunta é mais difícil do que parece

**Presença de diretório em `_internal` NÃO é o oráculo**: procurar `_internal\\alembic`,
`_internal\\httpx2` e `_internal\\pyarrow` não acha nenhum, e declararia três ausências — das quais
**duas falsas**.

O PyInstaller põe módulo Python puro no **arquivo PYZ**, e só ganha diretório em `_internal` o pacote
que traz extensão binária ou dado. Então:

| Pergunta | Oráculo |
|---|---|
| um pacote de Python puro viajou? | `build/refino-desktop/PYZ-00.toc` — `alembic` tem **96** entradas, `httpx2` tem **60** |
| um pacote com extensão binária viajou? | o diretório em `_internal` — PySide6, numpy, scipy, pandas, sklearn |
| um dado viajou? | o caminho dentro de `_internal` — as **quatro** migrações ficam em `_internal\\refino\\persistence\\migrations\\versions\\`, e o `env.py` e o `script.py.mako` um nível acima. Os seis entram arquivo a arquivo por `DADOS_DAS_MIGRACOES`, sem `__pycache__`, e o passo «as migracoes viajaram» do `congela_windows.ps1` confere a contagem em `versions/` — «4 de 4» no congelamento de `afa0377` |

**E o oráculo final é comportamental**, porque é o único que responde *"funciona"*: a migração corre
no arranque, e o arranque é exercitado pelo script.

## 7. `hooksconfig` deixa de ser `{}`, e o gancho de RUNTIME do `matplotlib` NÃO se corrige aqui

**São dois ganchos do PyInstaller com o mesmo nome de família e efeitos que não se parecem**, e
confundi-los manda a correção para o arquivo errado.

**O gancho de ANÁLISE — `hooks/hook-matplotlib.backends.py` — é o que `hooksconfig` configura.** Sem
opção, ele corre em modo `auto`: varre o bytecode à procura de `matplotlib.use()`, e se não achar
nenhum tenta importar `QtAgg`, `Qt5Agg`, `Gtk4Agg`, `Gtk3Agg`, `TkAgg` e `WxAgg` **em subprocessos
isolados**, ficando com o primeiro que importar. Nós não chamamos `matplotlib.use()`, então o modo
`auto` decidiria por nós, num passo que depende do que está instalado na máquina de quem congela.
`hooksconfig={"matplotlib": {"backends": [...]}}` fecha a escolha: `QtAgg` porque é o que a janela
desenha, `Agg` porque é o que o relatório usa para gravar PNG sem tela.

**O gancho de RUNTIME — `hooks/rthooks/pyi_rth_mplconfig.py` — não tem opção, e o spec não o
alcança.** Ele faz `os.environ['MPLCONFIGDIR'] = secure_mkdtemp()` **incondicionalmente** e regista
um `atexit` que apaga o diretório, então o cache de fontes é reconstruído a **cada** abertura. O
comentário dele nomeia o que previne, e o motivo é do `--onefile`: ali o diretório de extração muda
a cada execução, e um cache com caminhos absolutos para dentro dele quebra com
`RuntimeError: Could not open facefile`.

> **Um `--runtime-hook` nosso NÃO corrige, e o motivo importa porque o motivo errado sugere a
> correção errada.** Conferido em 10/09/2026 no PyInstaller 6.22.2 da venv do alvo,
> `depend/analysis.py:730-751`: os ganchos de `runtime_hooks=[...]` são acrescentados a
> `rthooks_nodes` **antes** dos ganchos implicados pelos pacotes, e o comentário do próprio arquivo
> diz *"First hooks in the list are executed first."* O nosso correria primeiro e seria
> sobrescrito. **Reordenar a lista aqui não pode funcionar.**

**Quem corre depois de todos os ganchos é o SCRIPT DE ENTRADA**, e é por isso que a correção vive em
`src/refino/desktop/arranque.py` e não neste arquivo. Ela deriva o `MPLCONFIGDIR` do **caminho de
instalação**: construímos `--onedir` e não `--onefile`, então a pasta é estável enquanto ficar onde
está, o cache continua válido entre aberturas, e uma cópia noutro caminho recebe cache próprio em
vez de herdar caminhos absolutos mortos. O `atexit` do gancho apaga o diretório temporário que ele
mesmo criou — que fica vazio —, e não o nosso: ele guarda o caminho no fecho, não relê a variável.

**A reversão desta decisão tem uma segunda metade**, e sem ela fica órfão no disco de quem usa:
tirar `matplotlib` daqui é a lista de sítios de composição **mais** apagar o diretório de cache que
`arranque.py` criou — um por caminho de instalação, e nada os limpa sozinho.
"""

from pathlib import Path

# O `sys.argv[0]` de um spec é o próprio spec, então a raiz sai daqui e não do `cwd`.
RAIZ = Path(SPECPATH)  # noqa: F821 - `SPECPATH` é injetado pelo PyInstaller
FONTE = RAIZ / "src"
MIGRACOES = FONTE / "refino" / "persistence" / "migrations"
# **Arquivo a arquivo, sem `__pycache__`.** A entrada era o diretório inteiro, e com ele
# viajavam os `.pyc` que a suíte deixa na árvore de quem congela — medido na `windows-2026.10.02.1`:
# cinco, com o caminho desta máquina no `co_filename`. O `alembic` lê os `.py`, e o cache não faz
# falta.
DADOS_DAS_MIGRACOES = [
    (
        str(arquivo),
        str(Path("refino/persistence/migrations") / arquivo.parent.relative_to(MIGRACOES)),
    )
    for arquivo in sorted(MIGRACOES.rglob("*"))
    if arquivo.is_file() and "__pycache__" not in arquivo.relative_to(MIGRACOES).parts
]
# O diretório inteiro parava o PyInstaller quando sumia; a lista vazia não pararia nada.
assert DADOS_DAS_MIGRACOES, f"nenhuma migração em {MIGRACOES}"

# **`excludes` conferido por varredura; ver a docstring.** Nome novo aqui exige a mesma varredura.
#
# `matplotlib` não está nesta lista: a janela desenha com ele (seção 4b da docstring).
EXCLUIDOS = [
    "ruptures",
    "psycopg2",
    "psycopg2cffi",
    "streamlit",
    # **Os dois do mummichog.**
    # `mummichog.reporting` é o único módulo do pacote que escreve arquivo e o único que
    # importa `xlsxwriter`; o núcleo não o alcança, e medido em `sys.modules`
    # depois de uma predição real ele NÃO é carregado. Excluí-lo torna isso verificável no
    # artefato em vez de depender do estado da venv de quem congela.
    "mummichog.reporting",
    "xlsxwriter",
    # Ferramentas de teste e de desenvolvimento: nenhuma é importada por `src/refino`.
    "pytest",
    "pytestqt",
    "_pytest",
    "hypothesis",
    "mypy",
    "ruff",
    "IPython",
    "tkinter",
]

analise = Analysis(  # noqa: F821 - injetado pelo PyInstaller
    [str(FONTE / "refino" / "desktop" / "arranque.py")],
    pathex=[str(FONTE)],
    binaries=[],
    # O destino é o CAMINHO DO PACOTE dentro do congelado, porque é assim que
    # `Path(persistence.__file__).parent / "migrations"` o resolve em runtime.
    datas=[
        *DADOS_DAS_MIGRACOES,
        # O ícone entra DUAS vezes e por duas razões distintas: como `icon=` do `EXE`, que é o que
        # o Explorer desenha no arquivo, e como dado empacotado, que é o que `setWindowIcon` lê em
        # runtime para a barra de tarefas. Sem esta linha o `.exe` teria ícone e a janela aberta
        # não — que é a metade que o usuário vê durante toda a sessão de trabalho.
        (str(FONTE / "refino" / "desktop" / "recursos"), "refino/desktop/recursos"),
        # **A POLÍTICA APRENDIDA, e sem esta linha o `--auto` quebra SÓ no congelado.**
        #
        # `refino/io/politica.py` resolve-a por `Path(decision.__file__).parent`, que é o mesmo
        # idioma das migrações acima — e é por isso que o artefato foi movido para dentro do
        # pacote antes de alguém o ler. Mas JSON não é módulo: o analisador do PyInstaller não o
        # alcança, e sem esta entrada o `--auto` ficaria verde em toda a suíte e a política não
        # carregaria no executável.
        (
            str(FONTE / "refino" / "decision" / "politica_aprendida.json"),
            "refino/decision",
        ),
        # **O PALPITE PELA VERDADE, a segunda opinião, e a razão é a da linha acima.**
        # `carrega_politica_pela_verdade` resolve-o pelo mesmo `Path(decision.__file__).parent`, e
        # sem esta entrada a rota da recomendação devolveria 503 só no executável — as duas árvores
        # carregam juntas. `tests/test_politica_pela_verdade.py` confere que todo JSON de
        # `refino/decision` está neste `datas=`.
        (
            str(FONTE / "refino" / "decision" / "politica_pela_verdade.json"),
            "refino/decision",
        ),
        # **O MANIFESTO DAS BIBLIOTECAS DE DADOS DAS VIAS.** Mesma
        # razão da política acima: `refino/io/dados_vias.py` o resolve por
        # `Path(pathways.__file__).parent`, e JSON não é módulo. Sem esta linha o enriquecimento
        # pelo MetaboAnalystR passaria em toda a suíte e, no executável, não acharia o que conferir.
        (
            str(FONTE / "refino" / "pathways" / "dados_vias.json"),
            "refino/pathways",
        ),
    ],
    # `alembic` e `httpx2` são importados DENTRO de função, e o analisador estático do PyInstaller
    # não os alcança. Sem estes dois nomes o binário arranca e morre na migração.
    #
    # **E `backend_qtagg` não é importado por nome NENHUM**: o registro de backends do `matplotlib`
    # resolve-o por string em runtime, a partir de `"QtAgg"`. O analisador não tem o que ver. Sem
    # ele o congelado cai para `Agg`, que **desenha** — e a figura não aparece na janela.
    hiddenimports=[
        "alembic",
        "alembic.command",
        "alembic.config",
        "alembic.runtime.migration",
        "httpx2",
        "matplotlib.backends.backend_qtagg",
        # **Os dois que a exportação em PDF e em TIFF carrega TARDE.** O `savefig` em PDF
        # resolve `backend_pdf` pelo registro de formatos, por string, e o Pillow carrega o plugin
        # do TIFF na primeira gravação nesse formato: o analisador estático não vê nenhum dos dois.
        # Sem eles, a gravação levanta no executável e funciona no código-fonte.
        "matplotlib.backends.backend_pdf",
        "PIL.TiffImagePlugin",
        "sqlalchemy.dialects.sqlite",
        # **Os OITO submódulos do mummichog, e eles são declarados por serem importados TARDE.**
        #
        # A lista é o registro de que o congelado precisa deles, e um registro incompleto não
        # protege: um `--exclude` futuro que apanhasse
        # `compound_dicts` levaria 5,1 MB do modelo metabólico e a aba levantaria
        # `ImportError` na máquina do usuário, sem o `.spec` reprovar.
        #
        # **O conjunto é DERIVADO e não adivinhado:** é `sys.modules` filtrado por
        # `mummichog` depois de uma chamada a `prediz_vias`. Medido: nove entradas, das
        # quais o pacote e os oito abaixo.
        # `pathways/analise.py` importa dentro da função, de propósito: o pacote roda
        # `logging.basicConfig` no import, e no topo isso reconfiguraria o logger raiz da
        # janela inteira. O modulegraph do PyInstaller encontra import dentro de função, e
        # mesmo assim eles vão nomeados — a lista é o registro de que o congelado PRECISA
        # deles, e um `--exclude` futuro que os apanhasse reprovaria aqui em vez de produzir
        # uma aba que levanta `ModuleNotFoundError` na máquina do usuário.
        #
        # `JSON_metabolicModels` e `compound_dicts` são DADOS em forma de módulo `.py` — 6,8 MB
        # e 5,1 MB —, e é por isso que não há entrada em `datas`: o modelo metabólico viaja
        # como código, não como arquivo ao lado.
        #
        # **`mummichog.reporting` NÃO está aqui, e a ausência tem MECANISMO.**
        #
        # Um import acidental não reprovaria cedo por falta de `xlsxwriter`: o mummichog
        # declara `Requires-Dist: xlsxwriter` sem marcador, então ele entra por resolução
        # normal, e o próximo `make setup` o traz.
        #
        # O que mantém o módulo fora do congelado é `EXCLUIDOS`, que é verificável no
        # artefato construído em vez de depender do estado de uma venv.
        "mummichog.compound_dicts",
        "mummichog.config",
        "mummichog.functional_analysis",
        "mummichog.get_user_data",
        "mummichog.ions",
        "mummichog.JSON_metabolicModels",
        "mummichog.models",
        "mummichog.nx_modularity",
        # **Os dois módulos do modo externo do enriquecimento, declarados pela PROVA e não
        # pela janela.** `scripts/prova_enriquecimento_congelado.py` os importa por nome, pelo
        # `pythonw.exe` embarcado, que só alcança a cópia solta derivada de `analise.pure` — e a
        # análise parte de `arranque.py`, não daquele roteiro. Em 02/10/2026 o painel os importa
        # no topo (`desktop/janela/enriquecimento.py`, alcançado por `janela/vias.py`), então hoje
        # eles entrariam de qualquer modo; o nome aqui é o registro de que a prova do congelado
        # depende deles mesmo que o painel deixe de os importar.
        "refino.io.enriquecimento_externo",
        "refino.io.rds_restrito",
    ],
    hookspath=[],
    # **Os backends do `matplotlib` são ESCOLHIDOS aqui, e não descobertos.** Sem esta opção o
    # gancho de análise corre em modo `auto`: como não chamamos `matplotlib.use()` em lado nenhum,
    # ele tenta importar `QtAgg`, `Qt5Agg`, `Gtk4Agg`, `Gtk3Agg`, `TkAgg` e `WxAgg` em subprocessos
    # isolados e fica com o primeiro que passar — decisão que depende do que está instalado na
    # máquina de quem congela. `QtAgg` é o que a janela desenha; `Agg` é o que grava PNG sem tela,
    # que é o caminho do relatório. Ver a seção 7 da docstring.
    hooksconfig={"matplotlib": {"backends": ["QtAgg", "Agg"]}},
    runtime_hooks=[],
    excludes=EXCLUIDOS,
    # ========================================================================
    # `noarchive=False`, e o worker tem pasta propria
    # ========================================================================
    #
    # **O que `noarchive=True` resolvia, e o que ele custava.** Com `False`, os modulos puros vao
    # para o PYZ embutido no `refino.exe`, e **nenhum interpretador de fora os alcanca** — o worker
    # do `loky` nao conseguiria importar `numpy`, `sklearn` nem `refino`. Com `True` eles ficam
    # soltos como `.pyc` em `_internal`, e um Python comum importa cada um por
    # `SourcelessFileLoader`. Foi assim que o congelado passou a paralelizar.
    #
    # **O custo foi MEDIDO em 08/09/2026, e ele nao e de disco.** A pasta passou de 262 MB para
    # 399,6 MB e de ~1.400 para **4.126 arquivos**, e o arranque a frio saltou de ~21 s para
    # **84,1 s** na primeira abertura de conteudo que a maquina nunca viu:
    #
    #   copiar os 400 MB                                  2,5 a 5,1 s
    #   1a abertura, conteudo NUNCA visto                 84,1 s
    #   1a abertura, copia nova de conteudo ja visto      28,4 e 25,6 s
    #   2a abertura da mesma pasta                        2,8 s
    #
    # > **Ler os bytes custa 3 s e abrir custa 84: o custo e POR ARQUIVO.** As duas ultimas voltas
    # > copiaram o MESMO conteudo para pastas novas e cairam para ~26 s — o Defender guarda o
    # > veredito por conteudo, e so a primeira pagou a varredura. **Quem recebe a pasta esta sempre
    # > na primeira volta.**
    #
    # **A saida e ter os dois, e ela cabe.** Dos 4.126 arquivos, **2.708 sao `.pyc`**; os 301
    # binarios tem de ficar soltos de qualquer modo. Com `noarchive=False` os `.pyc` do APLICATIVO
    # voltam para o PYZ — um arquivo a abrir em vez de ~2.700 — e uma copia solta deles continua na
    # raiz de `_internal`, que e a unica que o `pythonw.exe` embarcado le, pelo `._pth`. O bloco
    # «Por que a RAIZ» abaixo diz por que a copia nao pode ficar numa subpasta.
    #
    # **O que se mede NAO e o tamanho da arvore, e essa foi a confusao.** A troca tira 96 arquivos
    # dos 4.126 — 2,3%, nada. O que ela muda e quantos arquivos o APLICATIVO ABRE ao arrancar, e
    # e isso que o antivirus cobra. Medido em 08/09/2026, tres copias, conteudo que o Defender ja
    # tinha visto, cada uma para um caminho novo:
    #
    #   1a abertura de pasta nova    28,4 e 25,6 s  ->  **9,5 / 11,5 / 8,6 s**, mediana 9,5
    #   2a abertura da mesma pasta            2,8 s  ->  2,7 s  (a ancora: nao devia mudar)
    #
    # **O preco e a duplicacao dos modulos puros**: a pasta sobe para perto de 425 MB.
    #
    # Com o R portatil embarcado a pasta mede 724,8 MB em 8319 arquivos no congelamento de afa0377
    # (o «MB» de scripts/congela_windows.ps1 e MiB). O que vigia a pasta sao os detectores de
    # crescimento daquele roteiro — MB e contagem de arquivos contra a base, nos dois sentidos — e o
    # teto de 8.000 arquivos em `_internal`.
    noarchive=False,
    optimize=0,
)

# ---------------------------------------------------------------------------
# A COPIA SOLTA PARA O WORKER, e ela e o que o PYZ nao consegue ser
# ---------------------------------------------------------------------------
#
# `analise.pure` e a lista de modulos de Python puro, como `(nome, caminho_do_fonte, tipo)`. Com
# `noarchive=True` o proprio PyInstaller os compilava para dentro da arvore; aqui compilamos para
# uma estufa e entregamos como dados, na RAIZ de `_internal`.
#
# **A hierarquia importa e nao e cosmetica:** `SourcelessFileLoader` acha `a/b/c.pyc` para o modulo
# `a.b.c`, e `a/b/__init__.pyc` para o pacote `a.b`. Um achatamento poria tudo na raiz e nada
# importaria.
#
# ## Por que a RAIZ, e nao uma subpasta `_worker` — medido em 08/09/2026
#
# Os `.pyc` debaixo de `_internal/_worker/`, com o `._pth` a listar as duas pastas, **nao
# arrancam**, e a causa e o PACOTE PARTIDO: `numpy` fica com os `.pyd`
# em `_internal/numpy/` e o `__init__.pyc` em `_internal/_worker/numpy/`. O Python resolve o nome
# numa pasta SO — a que tem `__init__` ganha —, entao `numpy.__path__` passa a ser a do worker e
# `numpy._core._multiarray_umath.pyd` fica inalcancavel. Medido: `numpy` e `sklearn` com
# `ImportError`, `joblib` (que e puro) a importar bem.
#
# > **`noarchive=False` e a subpasta sao knobs INDEPENDENTES.** O que
# > acelera o arranque e o PYZ — o aplicativo abre UM arquivo em vez de ~2.700. Onde a copia solta
# > mora nao muda isso, porque o `FrozenImporter` do PyInstaller vem primeiro no `sys.meta_path` e
# > o aplicativo nunca abre estes `.pyc`. Na raiz, cada pacote volta a ser um so, com as duas
# > metades juntas — que e exactamente o arranjo que ja provou 13 workers e SHA-256 identico.
#
# **O ganho medido, com o oraculo de janela corrigido** (tres copias, conteudo que o Defender ja
# tinha visto, cada uma para um caminho novo):
#
#   1a abertura de pasta nova    26 s  ->  **9,5 s**   (28,4 e 25,6 antes; 9,5 / 11,5 / 8,6 depois)
#   2a abertura da mesma pasta  2,8 s  ->   2,7 s      (a ancora: nao devia mudar, e nao mudou)
#
# **As falhas de compilacao sao CONTADAS e nao silenciadas.** Um modulo que nao compile nesta
# versao de Python — sintaxe de outra, arquivo de teste de terceiro — nao pode derrubar o
# congelamento, mas tambem nao pode sumir sem numero: o total vai para o log, e o portao do
# congelamento afirma a contagem final.
import py_compile  # noqa: E402
import sys as _sys  # noqa: E402

_ESTUFA = Path(SPECPATH) / "build" / "pyc-do-worker"  # noqa: F821 - injetado pelo PyInstaller
if _ESTUFA.exists():
    import shutil as _shutil

    _shutil.rmtree(_ESTUFA)

_soltos = []
_falhados = []
_espacos_de_nome = []
for _nome, _origem, _tipo in analise.pure:  # noqa: F821 - injetado pelo PyInstaller
    # **`'-'` e PACOTE DE ESPACO DE NOMES e nao falha, e a distincao foi medida.** O PyInstaller
    # marca assim o pacote sem `__init__.py` — `scipy._external`, `setuptools._vendor` e mais seis.
    # Nao ha codigo para compilar, e nao e preciso: o Python acha o pacote pelo DIRETORIO, que
    # existe porque os filhos dele vao para la. Conta-los como falha daria um numero que descreve
    # outra coisa.
    if _origem == "-" or not Path(_origem).is_file():
        _espacos_de_nome.append(_nome)
        continue
    _partes = _nome.split(".")
    if Path(_origem).name == "__init__.py":
        _rel = Path(*_partes) / "__init__.pyc"
    else:
        _rel = Path(*_partes[:-1]) / f"{_partes[-1]}.pyc"
    _alvo = _ESTUFA / _rel
    _alvo.parent.mkdir(parents=True, exist_ok=True)
    try:
        py_compile.compile(_origem, cfile=str(_alvo), dfile=_nome, doraise=True)
    except Exception as _erro:  # noqa: BLE001 - qualquer falha e a mesma decisao: contar e seguir
        _falhados.append((_nome, str(_erro)[:80]))
        continue
    # **TOC de TRES elementos, e nao o par de `datas=`.** `Analysis(datas=[(origem, destino)])`
    # aceita pares e normaliza-os; `analise.datas` JA e um TOC, e acrescentar pares nele rebenta em
    # `COLLECT` com `not enough values to unpack (expected 3, got 2)` — medido em 08/09/2026. A
    # forma e `(caminho_dentro_do_pacote, caminho_na_maquina, tipo)`.
    _soltos.append((str(_rel).replace("\\", "/"), str(_alvo), "DATA"))

print(
    f"[spec] `.pyc` do worker: {len(_soltos)} compilados, "
    f"{len(_espacos_de_nome)} espacos de nomes (sem codigo), {len(_falhados)} falharam"
)
for _n, _e in _falhados[:10]:
    print(f"[spec]   FALHOU {_n}: {_e}", file=_sys.stderr)
if _falhados:
    raise SystemExit(
        f"[spec] {len(_falhados)} modulo(s) nao compilaram para o worker. Cada um deles e um "
        "`ImportError` dentro de um worker do `loky`, que morre sem traceback e em silencio. Ver "
        "as linhas FALHOU acima."
    )
analise.datas += _soltos  # noqa: F821 - injetado pelo PyInstaller

pyz = PYZ(analise.pure)  # noqa: F821 - injetado pelo PyInstaller

# ---------------------------------------------------------------------------
# A TELA DE ARRANQUE, e ela NAO reduz um segundo
# ---------------------------------------------------------------------------
#
# **O que ela conserta e o silencio, e nao o tempo.** Medido em 08/09/2026, em tres copias
# independentes da pasta: a primeira abertura de conteudo que a maquina nunca viu custa **84,1 s**,
# e durante esse tempo **nada aparece**. O `LEIA-ME` tem um paragrafo inteiro a pedir que ninguem
# feche nem clique duas vezes — um pedido em prosa, num arquivo que a pessoa ainda nao abriu.
#
# **E do CARREGADOR e nao do Qt, e isso e requisito.** Esta tela e desenhada pelo bootloader, em C,
# **antes** de o Python arrancar. Uma `QSplashScreen` precisaria de o PySide6 ja estar importado —
# 92 MB e 145 arquivos, dentro exactamente da janela que se quer cobrir.
#
# **Quem a FECHA e `arranque._corre_a_janela`**, no instante em que a janela real aparece. Sem essa
# chamada ela fica por cima de tudo para sempre, e o desfecho e pior que nao a ter.
telaDeArranque = Splash(  # noqa: F821 - injetado pelo PyInstaller
    str(FONTE / "refino" / "desktop" / "recursos" / "splash.png"),
    binaries=analise.binaries,  # noqa: F821
    datas=analise.datas,  # noqa: F821
    # ---------------------------------------------------------------------
    # `text_pos` NAO E OPCIONAL SE SE QUER TEXTO QUE MUDA
    # ---------------------------------------------------------------------
    #
    # O `PyInstaller/building/splash.py` decide com `if self.text_pos is not None:`. **Sem esta
    # linha o rotulo de texto nem sequer entra no script Tcl**, e `pyi_splash.update_text()` corre
    # sem erro e sem efeito — a chamada devolve, ninguem levanta, e a tela fica parada. E o desfecho
    # de sucesso de um mecanismo construido que nao faz nada, indistinguivel de mecanismo
    # a funcionar, porque o unico sintoma seria alguem OLHAR.
    #
    # **O par (32, 196) e o mesmo `ETAPA_X, ETAPA_Y` de `scripts/gera_splash.py`**, que desenha a
    # regua em y=178 e deixa a faixa por baixo VAZIA de proposito. Divergir aqui poe a etapa por
    # cima da prosa fixa ou fora da imagem, e o PNG continua valido.
    text_pos=(32, 196),
    # `text_size` e em PONTOS do Tk e nao em pixels como o resto da imagem: 10 pt a 96 DPI da
    # aproximadamente os 13 px da prosa fixa desenhada pelo Pillow.
    text_size=10,
    text_color="#6ca8ff",
    text_default="a arrancar...",
    always_on_top=True,
)

executavel = EXE(  # noqa: F821 - injetado pelo PyInstaller
    pyz,
    analise.scripts,
    telaDeArranque,
    # **A opção do interpretador.** Ver a seção 2 da docstring.
    [("X", "utf8=1", "OPTION")],
    exclude_binaries=True,
    # **`refino`, e COM ícone.** O ícone é a primeira coisa que se vê, e um `.exe` com o ícone
    # genérico do Windows lê como protótipo antes de a janela abrir.
    #
    # `refino.ico` traz sete tamanhos (256 a 16) desenhados a 1024 px e reduzidos por reamostragem,
    # e o gerador afirma que o 16 px não reduziu a bloco chapado — que é a forma de falha que
    # «o arquivo existe» não distingue.
    name="refino",
    icon=str(FONTE / "refino" / "desktop" / "recursos" / "refino.ico"),
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    # **`console=False`.** É um aplicativo de janela, e `arranque.garante_utf8` já tolera
    # `sys.stdout is None` por desenho — o que torna esta escolha segura em vez de arriscada.
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

colecao = COLLECT(  # noqa: F821 - injetado pelo PyInstaller
    executavel,
    telaDeArranque.binaries,
    analise.binaries,
    analise.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="refino",
)
