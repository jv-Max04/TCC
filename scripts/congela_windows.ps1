# congela_windows.ps1 — o congelamento do aplicativo de mesa e o EXERCICIO do binario.
#
# Corrido a mao no Windows, e NAO em `make check`: PyInstaller no WSL produz binario Linux. O
# resultado de cada passo e lido e vai na mensagem do commit.
#
# ============================================================================
# TODO RELATO DESTE SCRIPT E SO-ASCII, e isso e requisito e nao estilo
# ============================================================================
#
# A code page ANSI desta maquina e cp932. Uma sonda anterior morreu no passo que decide, com
# `UnicodeEncodeError: 'cp932' can't encode '\xf3'` — o `o` acentuado de "dentro do repositorio" na
# mensagem de recusa. Ela MEDIU a recusa e nao conseguiu RELATA-LA, e inferir o veredito do crash foi
# o que a segunda versao substituiu por medicao. Prosa acentuada e o que o BINARIO tem de mostrar
# certo; o relato do arnes fica em ASCII.
#
# ============================================================================
# DUAS ARMADILHAS DE POWERSHELL, medidas, que este script NAO repete
# ============================================================================
#
# 1. `Select-Object -First N` sobre a saida de um processo nativo FECHA O CANO e mata o processo a
#    escrever, produzindo `NativeCommandError` que parece falha do programa. Aqui a saida vai
#    inteira para arquivo, e o corte e feito DEPOIS, sobre o arquivo.
# 2. O PowerShell mastiga a exibicao de travessao. Isso NAO e defeito do binario, e por isso as
#    asserções de prosa acentuada comparam BYTES lidos de arquivo, nunca o que aparece na consola.
#
# ============================================================================
# USO
# ============================================================================
#
#   pwsh -ExecutionPolicy Bypass -File scripts\congela_windows.ps1 [-SoConstroi] [-SoExercita]
#
# Os quatro numeros do CENARIO 3 tambem sao parametros: `-BaseMB`, `-BaseArquivos`,
# `-CrescimentoMaximoMB` e `-CrescimentoMaximoArquivos`. Nao ha teto de tamanho.
#
# Espera ser corrido da RAIZ do clone Windows, com esse clone actualizado. O script confere as duas
# coisas e para se alguma nao valer.

[CmdletBinding()]
param(
    [switch]$SoConstroi,
    [switch]$SoExercita,
    [string]$Venv = "$env:USERPROFILE\_refino_env\venv-alvo",
    [string]$PythonBase = "$env:APPDATA\uv\python\cpython-3.12.14-windows-x86_64-none\python.exe",
    # Os quatro numeros do CENARIO 3. Sao parametros e nao constantes por duas razoes: a medicao
    # pos-congelamento corrige a base sem editar o roteiro, e o par causal de cada detector corre-se
    # passando um valor que a pasta medida viole; plantar editando o arquivo e o que faz a planta
    # ficar esquecida la dentro. A procedencia de cada um esta no comentario do CENARIO 3.
    [double]$BaseMB = 724.7,
    [int]$BaseArquivos = 8322,
    [double]$CrescimentoMaximoMB = 45.0,
    [int]$CrescimentoMaximoArquivos = 650,
    # O R embarcado e as bibliotecas de dados das vias. Vem da montagem de
    # `scripts\monta_r_portatil.ps1`, e e por isso que o manifesto da MONTAGEM entra junto: a pasta
    # copiada e conferida contra ele por igualdade total, e ele contra o versionado fora dos
    # carimbos de build.
    [string]$ArvoreR = "$env:USERPROFILE\_refino_env\r-portatil\arvore",
    [string]$ManifestoDaMontagem = "$env:USERPROFILE\_refino_env\r-portatil\saida\r_portatil_arvore.sha256",
    [string]$DadosVias = "$env:USERPROFILE\_refino_env\vias-dados"
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

# ----------------------------------------------------------------------------
# A TERCEIRA ARMADILHA DE POWERSHELL, medida neste proprio script
# ----------------------------------------------------------------------------
#
# **`$ErrorActionPreference = "Stop"` transforma a saida de ERRO de qualquer ferramenta nativa num
# erro que ABORTA o script**, e o PyInstaller escreve o log de INFO no stderr por desenho. A
# primeira versao deste script morreu na linha do PyInstaller com:
#
#   python.exe : 182 INFO: PyInstaller: 6.22.2, contrib hooks: 2026.6
#   + CategoryInfo : NotSpecified: (...) [], RemoteException
#   + FullyQualifiedErrorId : NativeCommandError
#
# Nao ha falha nenhuma nessa mensagem: e a primeira linha do log. E `2>&1` NAO resolve — os
# registos continuam a ser `ErrorRecord`, e com `Stop` escrever um deles levanta.
#
# > **E a familia e a mesma das outras duas: `NativeCommandError` que PARECE falha do programa.** A
# > primeira mata o processo a escrever com `Select-Object -First N`; a segunda mastiga travessao;
# > esta aborta o script por causa de uma linha de log. As tres produzem sintoma que aponta para o
# > binario, e nas tres o binario esta bem.
#
# `Roda` e a resposta: baixa a preferencia SO em volta da chamada nativa, e o que decide e o
# `$LASTEXITCODE` — que e a unica coisa que a ferramenta de facto afirma sobre si.
function Roda {
    param(
        [Parameter(Mandatory)][string]$Executavel,
        [string[]]$Argumentos = @(),
        [string]$Log = $null
    )
    $anterior = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        if ($Log) {
            & $Executavel @Argumentos *>&1 | Out-File -FilePath $Log -Encoding utf8
        } else {
            & $Executavel @Argumentos *>&1 | Out-Null
        }
        return $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $anterior
    }
}

# ----------------------------------------------------------------------------
# O relatorio: uma linha por passo, com o codigo de saida LIDO
# ----------------------------------------------------------------------------
$script:Passos = @()
$script:Falhas = 0

function Relata {
    param([string]$Passo, [int]$Codigo, [string]$Detalhe = "")
    $marca = if ($Codigo -eq 0) { "OK  " } else { "FALHA" }
    if ($Codigo -ne 0) { $script:Falhas++ }
    $script:Passos += [pscustomobject]@{ Passo = $Passo; Codigo = $Codigo; Detalhe = $Detalhe }
    Write-Host ("[{0}] {1}  (exit={2}) {3}" -f $marca, $Passo, $Codigo, $Detalhe)
}

function ParaSe {
    param([string]$Passo, [int]$Codigo, [string]$Detalhe = "")
    Relata $Passo $Codigo $Detalhe
    if ($Codigo -ne 0) {
        Write-Host ""
        Write-Host "PARADO em: $Passo"
        Resumo
        exit 1
    }
}

function Resumo {
    Write-Host ""
    Write-Host "=== RESUMO ==="
    foreach ($p in $script:Passos) {
        Write-Host ("  {0,-58} exit={1} {2}" -f $p.Passo, $p.Codigo, $p.Detalhe)
    }
    Write-Host ("passos={0}  falhas={1}" -f $script:Passos.Count, $script:Falhas)

    # O resumo GRAVA, e nao so imprime. `Relata` nao e fatal, entao um cenario pode falhar e o
    # script seguir ate ao fim; quem lesse depois so teria o console, que some. Sem o arquivo, o
    # desfecho do 4b (o duplo clique) ja ficou nao-observavel.
    #
    # Grava ao lado da pasta congelada, porque e a pasta que viaja e e sobre ela que a pergunta
    # "isto passou?" se faz.
    if ($script:Passos.Count -gt 0) {
        $destino = Join-Path $script:Raiz "dist\congelamento-verdicto.json"
        $corpo = [pscustomobject]@{
            quando  = (Get-Date).ToString("o")
            commit  = $script:Local
            passos  = $script:Passos.Count
            falhas  = $script:Falhas
            veredito = $(if ($script:Falhas -gt 0) { "FALHA" } else { "OK" })
            detalhe = $script:Passos
        }
        # `WriteAllText` com `UTF8Encoding($false)` e nao `Out-File -Encoding utf8`: no PS 5.1 este
        # ultimo escreve COM BOM, e `json.load` do Python recusa com `Unexpected UTF-8 BOM`. Medido
        # sobre o proprio arquivo que este bloco acabara de gravar.
        [System.IO.File]::WriteAllText(
            $destino,
            ($corpo | ConvertTo-Json -Depth 4),
            (New-Object System.Text.UTF8Encoding($false))
        )
        Write-Host ("resumo gravado em {0}" -f $destino)
    }
}

# ----------------------------------------------------------------------------
# PRE-CONDICOES — o clone certo, actualizado, e as ferramentas
# ----------------------------------------------------------------------------
$Raiz = (Get-Location).Path

# **A porta do congelado e a que RESPONDE `/health`, nunca a primeira da lista.** A tela de
# arranque do PyInstaller abre uma porta de escuta no loopback ANTES da nossa; ela aceita a
# ligacao e nunca responde HTTP. Os TRES sitios abaixo escolhiam pela ordem e passavam por sorte
# de temporizacao — a tela ja tinha fechado quando eles liam a lista. Ver
# `scripts/porta_que_serve.ps1`.
. (Join-Path $PSScriptRoot "porta_que_serve.ps1")
if (-not (Test-Path (Join-Path $Raiz "refino-desktop.spec"))) {
    Write-Host "FALHA: rode este script da raiz do clone (nao achei refino-desktop.spec)"
    exit 2
}

# O clone Windows fica ATRAS, e congelar dele e congelar uma arvore antiga (ja se mediu varios
# commits atras do tronco). O script CONFERE em vez de confiar, e a conferencia e contra o remoto.
$anteriorPref = $ErrorActionPreference
$ErrorActionPreference = "Continue"
git fetch --quiet origin 2>$null
$local = (git rev-parse HEAD).Trim()
$script:Local = $local
$remoto = (git rev-parse origin/main).Trim()
$atras = (git rev-list --count "HEAD..origin/main").Trim()
$ErrorActionPreference = $anteriorPref
Relata "clone local em $($local.Substring(0,7))" 0 "origin/main=$($remoto.Substring(0,7))"
if ($atras -ne "0") {
    Write-Host ""
    Write-Host "FALHA: o clone esta $atras commit(s) ATRAS de origin/main."
    Write-Host "       Congelar daqui embala uma arvore antiga. Rode:  git pull --ff-only"
    exit 2
}

if (-not (Test-Path $PythonBase)) {
    Write-Host "FALHA: nao achei o Python 3.12 nativo em $PythonBase"
    exit 2
}

# ----------------------------------------------------------------------------
# CONSTRUCAO — a venv do ALVO, e nao a de desenvolvimento
# ----------------------------------------------------------------------------
#
# **O conjunto de runtime e explicito, e o pacote entra com `--no-deps`.** Uma medicao mostrou
# `pyarrow`, `matplotlib`, `psycopg2` e PySide6 a entrar no `--onedir` por ESTAREM na venv. Instalar
# `-e .` traria `streamlit` e `ruptures` de volta pela porta das dependencias declaradas, e o
# `excludes` do spec seria o unico a segurar. Duas defesas, e esta e a primeira.
#
# `matplotlib` e PINO e nao exclusao: a janela desenha figuras com ele. Ele entra exactamente como
# `matplotlib==3.11.1`, o mesmo pino do `pyproject.toml`, porque duas versoes declaradas em dois
# arquivos e a forma de uma delas envelhecer sozinha.
#
# **E ele entra SEM `--no-deps`, que e o regime desta lista** (linhas abaixo), entao arrasta
# `contourpy`, `cycler`, `fonttools`, `kiwisolver`, `pillow`, `pyparsing`, `python-dateutil` e
# `packaging` sem pino nenhum. Medidos em disco na venv de desenvolvimento, os nove somam
# **51,4 MB em 1.267 arquivos**: o crescimento da pasta e maior que os 13,7 MB do pacote sozinho, e
# e por isso que as asserções de tamanho do CENARIO 3 passaram de `Relata` a `ParaSe`.
#
# `pyarrow` NAO esta nesta lista e NAO vem. Na venv de DESENVOLVIMENTO ele esta instalado e o
# `pandas` importa-o SE existir, e um fecho de importacao medido la sugeria que viria. Medido na
# venv do alvo: o `pip` nao o instalou, e o `PYZ-00.toc` do congelado tem 2 referencias vindas de
# dentro do `pandas` e nenhum modulo dele.
$RuntimeDoAlvo = @(
    "numpy==2.5.1",
    "pandas==3.0.5",
    "scipy==1.18.0",
    "scikit-learn==1.9.0",
    "joblib==1.5.3",
    # Dependencia direta: `statistics/clustering.py` limita com ele o OpenMP do `KMeans`.
    "threadpoolctl==3.6.0",
    "fastapi==0.140.0",
    "uvicorn==0.51.0",
    "python-multipart==0.0.32",
    "SQLAlchemy==2.0.51",
    "alembic==1.18.5",
    "httpx2==2.9.1",
    "matplotlib==3.11.1",
    # A predicao de vias. Ver a justificativa no `pyproject.toml`, que e onde ela vive; aqui
    # fica so o pino, porque duplicar o motivo criaria a segunda fonte de verdade.
    "mummichog==2.7.0",
    "networkx==3.6.1",
    "PySide6==6.11.2",
    "pyinstaller==6.22.2"
)

$PyAlvo = Join-Path $Venv "Scripts\python.exe"
$Dist = Join-Path $Raiz "dist\refino"
$Interno = Join-Path $Dist "_internal"
$Exe = Join-Path $Dist "refino.exe"
$Saidas = Join-Path $env:TEMP "refino-u10"
New-Item -ItemType Directory -Force -Path $Saidas | Out-Null

if (-not $SoExercita) {
    if (-not (Test-Path $PyAlvo)) {
        $c = Roda -Executavel $PythonBase -Argumentos @("-m", "venv", $Venv) `
            -Log (Join-Path $Saidas "venv.log")
        ParaSe "venv do alvo criada" $c $Venv
    } else {
        Relata "venv do alvo ja existe" 0 $Venv
    }

    $c = Roda -Executavel $PyAlvo `
        -Argumentos @("-m", "pip", "install", "--quiet", "--upgrade", "pip") `
        -Log (Join-Path $Saidas "pip-upgrade.log")
    Relata "pip actualizado" $c

    $c = Roda -Executavel $PyAlvo `
        -Argumentos (@("-m", "pip", "install", "--quiet") + $RuntimeDoAlvo) `
        -Log (Join-Path $Saidas "pip-runtime.log")
    ParaSe "runtime do alvo instalado" $c "$($RuntimeDoAlvo.Count) pinos"

    $c = Roda -Executavel $PyAlvo `
        -Argumentos @("-m", "pip", "install", "--quiet", "--no-deps", "-e", ".") `
        -Log (Join-Path $Saidas "pip-pacote.log")
    ParaSe "pacote instalado com --no-deps" $c

    # **A venv do alvo NAO tem os tres, e isto e asserção de PRESENCA DE AUSENCIA por nome.**
    # Um total de MB menor passaria com o pacote ainda la; o nome nao passa.
    #
    # `matplotlib` nao esta nesta lista: esta em `$RuntimeDoAlvo`, porque a janela desenha com
    # ele, e a presenca dele e afirmada logo abaixo e no CENARIO 2.
    $ausentesEsperados = @("streamlit", "psycopg2", "ruptures")
    $anteriorPref = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    $listados = (& $PyAlvo -m pip list --format=freeze 2>$null) -join "`n"
    $ErrorActionPreference = $anteriorPref
    $presentes = @()
    foreach ($nome in $ausentesEsperados) {
        if ($listados -match "(?im)^$nome==") { $presentes += $nome }
    }
    ParaSe "os 3 pacotes fora da venv do alvo" ($presentes.Count) ($presentes -join ",")

    # E a direcao oposta, que a lista acima nao cobre: `matplotlib` tem de ESTAR la, e na versao
    # pinada. Sem esta metade, um pino que o `pip` nao instalasse — nome trocado, resolucao que
    # recuou para outra versao — passaria calado, e o congelado sairia sem o pacote que a janela
    # usa para desenhar. A asserção e por NOME E VERSAO, pelo argumento de sempre: o total de MB
    # nao distingue as duas coisas.
    $pinoDoMatplotlib = @($RuntimeDoAlvo | Where-Object { $_ -like "matplotlib==*" })
    $temMatplotlib = 0
    if ($pinoDoMatplotlib.Count -ne 1) {
        $temMatplotlib = 2
    } elseif ($listados -notmatch ("(?im)^" + [regex]::Escape($pinoDoMatplotlib[0]) + "\s*$")) {
        $temMatplotlib = 1
    }
    $rotuloDoPino = ($pinoDoMatplotlib -join ', ')
    ParaSe "o matplotlib pinado esta na venv do alvo" $temMatplotlib `
        "esperado [$rotuloDoPino] (codigo 2 = o pino sumiu de RuntimeDoAlvo)"

    # O BINARIO ANTERIOR ABERTO e o modo de falha mais comum daqui, e ele nao tinha desfecho.
    #
    # Medido: um `refino.exe` aberto desde a vespera segurava
    # `dist\refino\_internal\_greenlet.cp312-win_amd64.pyd`, o `Remove-Item` abaixo levantou
    # `UnauthorizedAccessException`, e o roteiro morreu despejando o rastro do .NET — sem dizer
    # QUAL janela fechar. Reconstruir com a versao anterior aberta e o caso natural de quem acabou
    # de a experimentar, entao a guarda nomeia o processo em vez de deixar a excecao vazar.
    #
    # `ParaSe` com a CONTAGEM e nao com um booleano: o detalhe leva os PIDs, que e o que a pessoa
    # precisa para fechar. Um `1` generico faria a mensagem descrever o problema sem o resolver.
    $abertos = @(Get-Process -ErrorAction SilentlyContinue | Where-Object {
        $_.Path -and $_.Path.StartsWith((Join-Path $Raiz "dist"), [StringComparison]::OrdinalIgnoreCase)
    })
    # `$(...)` e nao `(if ...)`: o parser ACEITA `(if ...)` como argumento e o runtime recusa com
    # `The term 'if' is not recognized` (medido). Analise estatica verde nao e execucao verde: a
    # guarda so vale depois de correr.
    $detalheDosAbertos = ""
    if ($abertos.Count) {
        $detalheDosAbertos = "feche: " + (($abertos | ForEach-Object {
            "$($_.ProcessName) (PID $($_.Id))"
        }) -join ", ")
    }
    ParaSe "nenhum binario do dist em execucao" $abertos.Count $detalheDosAbertos

    if (Test-Path (Join-Path $Raiz "build")) { Remove-Item -Recurse -Force (Join-Path $Raiz "build") }
    if (Test-Path (Join-Path $Raiz "dist")) { Remove-Item -Recurse -Force (Join-Path $Raiz "dist") }

    $c = Roda -Executavel $PyAlvo `
        -Argumentos @("-m", "PyInstaller", "--noconfirm", "--clean", "refino-desktop.spec") `
        -Log (Join-Path $Saidas "pyinstaller.log")
    ParaSe "PyInstaller correu" $c "log em $Saidas\pyinstaller.log"
}

# ----------------------------------------------------------------------------
# CENARIO 1 — a pasta e produzida e o executavel existe
# ----------------------------------------------------------------------------
$existe = if ((Test-Path $Exe) -and (Test-Path $Interno)) { 0 } else { 1 }
ParaSe "a pasta e o executavel existem" $existe $Exe

# ----------------------------------------------------------------------------
# CENARIO 1b — a pasta CARREGA a sua procedencia e as instrucoes de uso
# ----------------------------------------------------------------------------
#
# O defeito que isto fecha: nenhum artefato entregue dizia de que commit saiu. `dist/` esta no
# `.gitignore`, entao ausencia de commit nunca foi evidencia de ausencia de congelamento: ja houve
# um congelamento sem registro, e um binario dado como ainda defeituoso que ja nao o era.
#
# Uma copia anterior ao conserto morre em 2 s com codigo 1 e NAO e distinguivel da boa por
# inspecao. Com VERSAO.txt, e.
#
# O LEIA-ME viaja pela mesma razao pratica: quem recebe a pasta nao tem de onde saber que o aviso
# do SmartScreen e esperado, e quem pare nele nao chega a abrir a ferramenta.
$carimbo = Join-Path $Dist "VERSAO.txt"
# `Out-String` no meio: com a arvore LIMPA o `git status --porcelain` nao devolve NADA, e
# `$null.Trim()` levanta, matando o congelamento no ramo NORMAL (medido). Uma expressao que devolve
# vazio comporta-se diferente de uma que devolve string vazia.
$sujo = (git status --porcelain | Out-String).Trim()
# A linha diz QUAIS arquivos, e nao so que ha alteracoes. Medido: o carimbo
# dizia "COM ALTERACOES NAO COMMITADAS" com a sujeira a ser o `settings.json` de um plugin e um
# `uv.lock` — nada que entre no binario. Sem os nomes, a linha sugere que o executavel pode nao
# corresponder a commit nenhum, e quem a le nao tem como saber que corresponde.
$sujoCodigo = @($sujo -split "`n" | Where-Object { $_ -match "^\s*[AMDR?]{1,2}\s+src/" })
if (-not $sujo) {
    $estado = "arvore limpa"
} elseif ($sujoCodigo.Count -gt 0) {
    $estado = "COM ALTERACOES EM src/ NAO COMMITADAS - o binario pode divergir do commit acima"
} else {
    # O arquivo NAO RASTREADO e contado, e nao nomeado. O VERSAO.txt viaja na release publica, e
    # ja levou o nome de um arquivo que o usuario mantem fora de todo commit. O motivo de nomear,
    # acima, e dizer que o binario corresponde ao commit; um arquivo nao rastreado nao esta no
    # commit, e o nome dele nada acrescenta a essa prova.
    $linhas = @($sujo -split "`n" | Where-Object { $_.Trim() })
    $rastreados = @($linhas | Where-Object { $_ -notmatch "^\?\?" } |
                    ForEach-Object { ($_ -replace "^\s*[AMDR]{1,2}\s+", "").Trim() } |
                    Select-Object -First 4)
    $naoRastreados = @($linhas | Where-Object { $_ -match "^\?\?" }).Count
    $partes = @()
    if ($rastreados.Count -gt 0) { $partes += "alteracoes nao commitadas (" + ($rastreados -join ", ") + ")" }
    if ($naoRastreados -gt 0) { $partes += "$naoRastreados arquivo(s) nao rastreado(s), sem nome aqui" }
    $estado = "limpa em src/; fora dele ha " + ($partes -join " e ")
}
@(
    # ASCII puro, e nao e estilo: o Windows PowerShell 5.1 le um `.ps1` SEM BOM como ANSI, e nesta
    # maquina a code page e `cp932`. Um travessao aqui saiu como `窶・ dentro do VERSAO.txt que o
    # laboratorio le — mojibake na primeira linha do arquivo que identifica o build. Todo o resto
    # deste script ja era sem acento pela mesma razao.
    "refino - aplicativo nativo de Windows"
    ""
    "commit    : $script:Local"
    "arvore    : $estado"
    "congelado : $((Get-Date).ToString('yyyy-MM-dd HH:mm:ss'))"
    "maquina   : $env:COMPUTERNAME"
    ""
    "Cite estas linhas ao relatar qualquer problema: sem elas nao e possivel"
    "saber a qual versao do programa o relato se refere."
) | ForEach-Object { $_ } | Set-Content -LiteralPath $carimbo -Encoding ascii

$origemLeiaMe = Join-Path $Raiz "docs\LEIA-ME-aplicativo.txt"
$destinoLeiaMe = Join-Path $Dist "LEIA-ME.txt"
if (Test-Path $origemLeiaMe) { Copy-Item $origemLeiaMe $destinoLeiaMe -Force }

# O guia passo a passo, autocontido: as imagens vao embutidas como `data:` URI, entao ele abre no
# navegador sem rede e sem pasta de apoio.
$origemGuia = Join-Path $Raiz "docs\GUIA-do-aplicativo.html"
$destinoGuia = Join-Path $Dist "GUIA.html"
if (Test-Path $origemGuia) { Copy-Item $origemGuia $destinoGuia -Force }

# O documento DIDATICO viaja pela mesma razao que o GUIA, e a distincao entre os dois e o motivo
# de serem dois arquivos: o GUIA ensina a OPERAR (clique a clique, com telas), o ENTENDER ensina a
# ENTENDER (o que cada peca faz e por que). Quem recebe a pasta nao tem de onde saber que a
# segunda pergunta tem resposta escrita.
$origemEnt = Join-Path $Raiz "docs\ENTENDER-o-sistema.html"
$destinoEnt = Join-Path $Dist "ENTENDER.html"
if (Test-Path $origemEnt) { Copy-Item $origemEnt $destinoEnt -Force }

# O CADERNO DE MEDICOES viaja porque e a resposta a pergunta que o ENTENDER levanta e nao
# esgota: "de onde saem estes numeros?". O ENTENDER cita os resultados; este traz o desenho de
# cada medicao, quantas execucoes, com que parametros, e o que cada numero NAO quer dizer. Quem
# recebe a pasta e vai contestar um numero precisa dele, e nao
# tem de onde saber que existe se ele ficar so no repositorio.
$origemMed = Join-Path $Raiz "docs\MEDICOES-da-correcao.html"
$destinoMed = Join-Path $Dist "MEDICOES.html"
if (Test-Path $origemMed) { Copy-Item $origemMed $destinoMed -Force }

# O MULTIVERSO tem documento proprio porque a pergunta dele e outra: o ENTENDER explica o sistema
# e cita o multiverso de passagem; este diz o que ele faz, para que serve, e QUANTO CUSTA nesta
# maquina — as quatro grades cronometradas e a curva de threads. Quem recebe a pasta vai perguntar
# "quanto tempo isso demora?", e essa resposta nao estava escrita em lugar nenhum.
# O multiverso e funcionalidade beta, fora da entrega ao laboratorio, e a documentacao
# dele vive em `docs\trabalhos_futuros\`. A aba continua no executavel, marcada beta, e por isso o
# documento continua a viajar com o mesmo nome.
$origemMv = Join-Path $Raiz "docs\trabalhos_futuros\MULTIVERSO-o-que-e-e-o-que-custa.html"
$destinoMv = Join-Path $Dist "MULTIVERSO.html"
if (Test-Path $origemMv) { Copy-Item $origemMv $destinoMv -Force }

# O EXPLICAR viaja porque a pergunta dele e a terceira, e nenhum dos outros a
# responde: o GUIA ensina a OPERAR (clique a clique), o ENTENDER ensina o SISTEMA
# (o que cada peca faz e por que), e este ensina a EXPLICAR A JANELA — controle a
# controle, o que cada botao e campo faz ao dado, quando mexer, e como dizer isso
# numa frase.
$origemExp = Join-Path $Raiz "docs\EXPLICAR-cada-tela-e-cada-campo.html"
$destinoExp = Join-Path $Dist "EXPLICAR.html"
if (Test-Path $origemExp) { Copy-Item $origemExp $destinoExp -Force }

# O FAQ viaja porque a pergunta dele nao e de nenhum dos outros: os cinco documentos explicam o
# que o sistema FAZ, e este responde as duvidas que aparecem enquanto se USA.
# **E ele carrega os "nao sei".** Onde a medicao nao separou, esta escrito que nao separou; um FAQ
# que so tem respostas confiantes ensina a confiar nas erradas junto com as certas.
$origemFaq = Join-Path $Raiz "docs\FAQ-perguntas-frequentes.html"
$destinoFaq = Join-Path $Dist "FAQ.html"
if (Test-Path $origemFaq) { Copy-Item $origemFaq $destinoFaq -Force }

# O DIAGNOSTICO viaja porque responde de onde vem CADA numero do diagnostico: o que e consensual
# da area, o que apenas coincide com o consensual, e o que e construcao deste trabalho.
$origemDiag = Join-Path $Raiz "docs\DIAGNOSTICO-o-que-ele-mede.html"
$destinoDiag = Join-Path $Dist "DIAGNOSTICO.html"
if (Test-Path $origemDiag) { Copy-Item $origemDiag $destinoDiag -Force }

# A asserção e de PRESENCA e de CONTEUDO, nao so de existencia: um arquivo vazio existe.
$carimbadoOk = 0
if (-not (Test-Path $carimbo)) { $carimbadoOk = 1 }
elseif (-not (Select-String -Path $carimbo -Pattern ([regex]::Escape($script:Local)) -Quiet)) { $carimbadoOk = 1 }
if (-not (Test-Path $destinoLeiaMe)) { $carimbadoOk = 1 }
elseif ((Get-Item $destinoLeiaMe).Length -lt 500) { $carimbadoOk = 1 }
if (-not (Test-Path $destinoGuia)) { $carimbadoOk = 1 }
elseif ((Get-Item $destinoGuia).Length -lt 50000) { $carimbadoOk = 1 }
if (-not (Test-Path $destinoEnt)) { $carimbadoOk = 1 }
elseif ((Get-Item $destinoEnt).Length -lt 60000) { $carimbadoOk = 1 }
if (-not (Test-Path $destinoMed)) { $carimbadoOk = 1 }
elseif ((Get-Item $destinoMed).Length -lt 30000) { $carimbadoOk = 1 }
if (-not (Test-Path $destinoMv)) { $carimbadoOk = 1 }
elseif ((Get-Item $destinoMv).Length -lt 20000) { $carimbadoOk = 1 }
if (-not (Test-Path $destinoExp)) { $carimbadoOk = 1 }
elseif ((Get-Item $destinoExp).Length -lt 40000) { $carimbadoOk = 1 }
# O piso do FAQ e 20.000 sobre 35.346 medidos -- folga de 43%, pela mesma razao dos
# outros: o piso e contra TRUNCAMENTO e nao um limite justo, e um piso apertado reprovaria por
# edicao normal do documento, que e como um portao aprende a ser ignorado.
if (-not (Test-Path $destinoFaq)) { $carimbadoOk = 1 }
elseif ((Get-Item $destinoFaq).Length -lt 20000) { $carimbadoOk = 1 }
# O piso do DIAGNOSTICO e 30.000 sobre 47.067 medidos -- folga de 36%, pela mesma
# razao dos outros: o piso e contra TRUNCAMENTO e nao um limite justo.
if (-not (Test-Path $destinoDiag)) { $carimbadoOk = 1 }
elseif ((Get-Item $destinoDiag).Length -lt 30000) { $carimbadoOk = 1 }
ParaSe "a pasta carrega VERSAO, LEIA-ME e os SETE documentos HTML" $carimbadoOk "commit=$script:Local"

# ----------------------------------------------------------------------------
# O INTERPRETADOR EMBARCADO - o que faz o congelado paralelizar
# ----------------------------------------------------------------------------
#
# **Por que ele existe.** O `loky` gera os workers relancando `sys.executable`, que no congelado e
# o proprio `refino.exe`, e cada worker abria uma janela (14 medidas). A saida foi
# desligar o paralelismo inteiro, e o `serrf` passou a custar 6,7x mais. Com um interpretador de
# verdade na pasta, o `loky` lanca ELE, e nao o produto.
#
# **`pythonw.exe` e NAO `python.exe`, e a diferenca e uma janela por worker.** Os dois sitios do
# `loky` que geram processo chamam `CreateProcess` com `dwCreationFlags = 0`, e o `refino.exe` e
# `console=False` — nao ha console a herdar. Um filho de subsistema CONSOLE lancado por um pai sem
# console **ganha um console novo**: doze workers, doze janelas pretas. O `pythonw.exe` e de
# subsistema GUI, aceita `-c` igualmente, e custa 90 KB.
#
# **O `._pth` e obrigatorio e nao decorativo.** Um interpretador copiado sozinho procura a stdlib
# pelo marco `Lib\os.py` ao lado do executavel; em `_internal` esse marco nao existe, e ele cai no
# registro ou no prefixo de construcao — que na maquina que constroi resolve para a instalacao
# original do uv e **mascara a falha inteira**. O `._pth` fixa o `sys.path` e desliga o `site`.
$interpretadorFonte = Join-Path (Split-Path -Parent $PythonBase) "pythonw.exe"
$interpretadorAlvo = Join-Path $Interno "pythonw.exe"
$caminhoPth = Join-Path $Interno "python312._pth"
$interpretadorOk = 0
if (-not (Test-Path $interpretadorFonte)) {
    $interpretadorOk = 1
} else {
    Copy-Item $interpretadorFonte $interpretadorAlvo -Force
    # DUAS pastas, e a segunda entrou depois de uma versao so com a primeira nao arrancar.
    #
    # `.` e a propria `_internal`, e ela tem de conter CADA pacote inteiro — os `.pyd` e os `.pyc`
    # do mesmo pacote na mesma pasta. Uma tentativa pos os `.pyc` numa subpasta
    # `_worker` e listou as duas aqui: **nao arranca**, porque o Python resolve um nome de pacote
    # numa pasta so. `numpy` ficava com o `__init__.pyc` em `_worker` e com o
    # `_core/_multiarray_umath.pyd` em `_internal`, e o segundo ficava inalcancavel.
    #
    # `base_library.zip` e a STDLIB DE ARRANQUE. Com `noarchive=False` o PyInstaller volta a emitir
    # esse zip com os 155 modulos que o interpretador precisa ANTES de qualquer `sys.path` do
    # utilizador — `encodings` sao 122 deles, e `os.pyc` esta la tambem. Eles NAO chegam pelo
    # `.spec`, porque nao estao em `analise.pure`. Sem esta linha o `pythonw.exe` embarcado morre
    # em `init_fs_encoding` com codigo 1 e stdout vazio (medido).
    #
    # **A ordem importa:** `.` primeiro, para que o pacote completo seja achado onde ele de facto
    # esta, e o zip depois, so para o arranque.
    #
    # `import site` fica COMENTADO de proposito: sem ele o interpretador nao le `site-packages` de
    # lugar nenhum, que e exactamente o isolamento que se quer.
    Set-Content -Path $caminhoPth -Encoding ascii -Value @(".", "base_library.zip", "# import site")
    if (-not (Test-Path $interpretadorAlvo)) { $interpretadorOk = 1 }
    elseif ((Get-Item $interpretadorAlvo).Length -lt 50000) { $interpretadorOk = 1 }
    elseif (-not (Test-Path $caminhoPth)) { $interpretadorOk = 1 }
}
$tamInterp = 0
if (Test-Path $interpretadorAlvo) { $tamInterp = (Get-Item $interpretadorAlvo).Length }
ParaSe "o interpretador embarcado e o `._pth` estao na pasta" $interpretadorOk `
    ("pythonw.exe $tamInterp bytes + python312._pth")

# ----------------------------------------------------------------------------
# AS SAIDAS PRONTAS — as tres pastas que se abrem sem rodar nada
# ----------------------------------------------------------------------------
#
# Quem recebe a pasta nao devia esperar por calculo NENHUM, e as
# 32 saidas custam dezenas de minutos em qualquer regime.
#
# Elas NAO sao geradas aqui, e a separacao e deliberada. Quem as gera e
# `scripts/gera_saidas_prontas.py`, que corre FORA do congelado: uma medicao de dezenas de
# minutos nao pertence a um portao de embalagem, que responde outra pergunta: *"o binario esta bem
# formado?"*. O custo e o mesmo dentro e fora do binario (82,3 s contra 78,8 s no `neg_c18` com
# `serrf`); correr fora so e mais simples, porque nao ha binario a construir.
#
# **Ausencia PARA o roteiro em vez de embalar sem elas.** Um zip sem as saidas e indistinguivel de
# um zip com elas para quem nao as procura, e quem o recebe so descobre ao abrir a pasta.
$origemSaidas = Join-Path $Raiz "out\saidas-prontas"
$destinoSaidas = Join-Path $Dist "saidas"
$saidasOk = 0
if (-not (Test-Path $origemSaidas)) {
    $saidasOk = 1
} else {
    if (Test-Path $destinoSaidas) { Remove-Item $destinoSaidas -Recurse -Force }
    Copy-Item $origemSaidas $destinoSaidas -Recurse -Force
    # Presenca da pasta nao basta: o que viaja sao os CSV. Contagem e piso de tamanho, pelo mesmo
    # argumento dos documentos — `Copy-Item` que falhe a meio deixa arquivo de zero bytes.
    $csv = @(Get-ChildItem $destinoSaidas -Recurse -Filter *.csv)
    $magros = @($csv | Where-Object { $_.Length -lt 10000 })
    if ($csv.Count -lt 20 -or $magros.Count -gt 0) { $saidasOk = 1 }
}
if ($saidasOk -ne 0) {
    Write-Host ""
    Write-Host "  Gere-as antes:  .venv\Scripts\python.exe scripts\gera_saidas_prontas.py"
}
$quantos = 0
if (Test-Path $destinoSaidas) { $quantos = @(Get-ChildItem $destinoSaidas -Recurse -Filter *.csv).Count }
ParaSe "as tres pastas de saida prontas viajam" $saidasOk "$quantos CSV em $destinoSaidas"

# ----------------------------------------------------------------------------
# O R EMBARCADO, O ROTEIRO DAS VIAS E AS BIBLIOTECAS DE DADOS
# ----------------------------------------------------------------------------
#
# **O R entra como o `pythonw.exe` entrou: copiado por este roteiro, fora do `.spec`.** O
# PyInstaller nao sabe o que e uma arvore R e iria analisa-la como dado; aqui ela e copiada inteira
# e conferida, e e o lancador (`refino.io.enriquecimento_r`) que a resolve a partir do executavel.
#
# **Duas conferencias da arvore, e a ordem e o conteudo:**
#
#   (a) a pasta copiada contra o manifesto da MONTAGEM que se copiou, por igualdade TOTAL - o que
#       apanha copia truncada e arquivo trocado depois da montagem;
#   (b) a mesma pasta contra o manifesto VERSIONADO, fora dos carimbos de build - o que apanha
#       uma montagem que nao e a do repositorio.
#
# As duas correm ANTES de o roteiro entrar na pasta: `R\vias_entrypoint.R` nao e da montagem, e
# depois de copiado ele apareceria como "sobra" nas duas.
#
# **As bibliotecas de dados viajam so na pasta do laboratorio.** O zip publico as deixa de
# fora por licenca (`scripts\empacota_para_o_github.py`), e a release publica as baixa com
# confirmacao. Aqui a conferencia e por hash contra o manifesto do pacote, e reprova tambem arquivo
# a mais.
# A ultima linha de um log que casa um padrao, como TEXTO: `Select-String` devolve um objeto, e
# interpola-lo numa frase depende de como o PowerShell o converte.
function UltimaLinha([string]$Arquivo, [string]$Padrao) {
    if (-not (Test-Path $Arquivo)) { return "(sem log)" }
    $achado = Get-Content $Arquivo | Select-String $Padrao | Select-Object -Last 1
    if ($achado) { return $achado.Line.Trim() }
    return "(nada casou $Padrao)"
}
$PyRepo = Join-Path $Raiz ".venv\Scripts\python.exe"
$destinoR = Join-Path $Dist "R"
$destinoDados = Join-Path $Dist "vias-dados"
New-Item -ItemType Directory -Force -Path $Saidas | Out-Null
$rOk = 0
$detalheR = ""
if (-not (Test-Path (Join-Path $ArvoreR "bin\x64\Rscript.exe"))) {
    $rOk = 1; $detalheR = "sem Rscript.exe na arvore da montagem: $ArvoreR"
} elseif (-not (Test-Path $ManifestoDaMontagem)) {
    $rOk = 1; $detalheR = "sem o manifesto da montagem: $ManifestoDaMontagem"
} elseif (-not (Test-Path $PyRepo)) {
    $rOk = 1; $detalheR = "sem o Python do clone: $PyRepo"
} else {
    if (Test-Path $destinoR) { Remove-Item $destinoR -Recurse -Force }
    Copy-Item $ArvoreR $destinoR -Recurse -Force
    $comparador = Join-Path $Raiz "scripts\r_portatil.py"
    $logTotal = Join-Path $Saidas "r-compara-montagem.log"
    $logVersionado = Join-Path $Saidas "r-compara-versionado.log"
    $cTotal = Roda -Executavel $PyRepo -Log $logTotal -Argumentos @(
        $comparador, "compara", "--arvore", $destinoR, "--manifesto", $ManifestoDaMontagem)
    $cVersionado = Roda -Executavel $PyRepo -Log $logVersionado -Argumentos @(
        $comparador, "compara", "--arvore", $destinoR,
        "--manifesto", (Join-Path $Raiz "scripts\r_portatil_arvore.sha256"), "--fora-dos-carimbos")
    $resumoTotal = UltimaLinha $logTotal "comparados"
    $resumoVers = UltimaLinha $logVersionado "comparados"
    if ($cTotal -ne 0) { $rOk = 1 }
    if ($cVersionado -ne 0) { $rOk = 1 }
    $detalheR = "montagem: [$resumoTotal] versionado: [$resumoVers]"
}
ParaSe "a arvore R copiada e a da montagem, e a montagem e a versionada" $rOk $detalheR

# O roteiro das vias, byte a byte o do repositorio.
#
# Esta linha e INFORMATIVA, e nao portao. Comparar o sha256 da copia com o da fonte logo depois do
# `Copy-Item` nao tem desfecho de reprovacao alcancavel: com `$ErrorActionPreference = "Stop"`
# (posto no topo; os trechos que o trocam por "Continue" o restauram antes desta linha), uma copia
# que falha encerra o roteiro ANTES da comparacao, e uma copia que termina sem erro deixa o arquivo
# igual ao da fonte. Nenhum passo deste roteiro escreve em `R\` depois desta copia, entao mover a
# conferencia para mais tarde tambem nao a tornaria alcancavel. A conferencia com os dois desfechos
# VISTOS e a de scripts\prova_enriquecimento_congelado.ps1 ("o roteiro da pasta e o do
# repositorio"), que corre sobre a pasta depois de pronta. Aqui fica o sha256 da copia, para o
# registo.
$fonteRoteiro = Join-Path $Raiz "r_service\vias_entrypoint.R"
$destinoRoteiro = Join-Path $destinoR "vias_entrypoint.R"
Copy-Item $fonteRoteiro $destinoRoteiro -Force
$hashCopia = (Get-FileHash -Algorithm SHA256 $destinoRoteiro).Hash
Relata "o roteiro das vias copiado para a pasta (informativo; a conferencia e da prova do congelado)" 0 `
    ("sha256 " + $hashCopia.Substring(0, 16))

$dadosOk = 0
$detalheDados = ""
if (-not (Test-Path $DadosVias)) {
    $dadosOk = 1; $detalheDados = "sem a pasta de bibliotecas: $DadosVias"
} else {
    if (Test-Path $destinoDados) { Remove-Item $destinoDados -Recurse -Force }
    Copy-Item $DadosVias $destinoDados -Recurse -Force
    $logDados = Join-Path $Saidas "vias-dados-confere.log"
    $dadosOk = Roda -Executavel $PyRepo -Log $logDados -Argumentos @(
        (Join-Path $Raiz "scripts\busca_dados_vias.py"), "--destino", $destinoDados, "--so-confere")
    $detalheDados = UltimaLinha $logDados "confer"
}
ParaSe "as bibliotecas de dados das vias viajam e conferem com o manifesto" $dadosOk "$detalheDados"
$tamR = 0
$arqR = 0
if (Test-Path $destinoR) {
    $itensR = @(Get-ChildItem $destinoR -Recurse -File)
    $arqR = $itensR.Count
    $tamR = ($itensR | Measure-Object -Property Length -Sum).Sum
}
Relata "tamanho do R embarcado" 0 ("{0:N1} MB em {1} arquivos" -f ($tamR / 1MB), $arqR)

# ----------------------------------------------------------------------------
# CENARIO 2 — os pacotes excluidos estao AUSENTES de _internal, POR NOME
# ----------------------------------------------------------------------------
#
# Asserção de presenca de ausencia, por pacote nomeado. Um pacote que continue la com o total a cair
# passaria por uma asserção de tamanho — e e por isso que o tamanho vem depois, como registro, e nao
# como criterio.
#
# `matplotlib` nao esta na lista porque a janela desenha com ele; a assercao dele e a OPOSTA,
# logo abaixo: presenca do diretorio e da pasta de dados dele.
$deveEstarAusente = @("streamlit", "psycopg2", "ruptures")
$achados = @()
foreach ($nome in $deveEstarAusente) {
    $alvo = Join-Path $Interno $nome
    if (Test-Path $alvo) { $achados += $nome }
}
$achados = @($achados)
Relata "os 3 excluidos ausentes de _internal" ($achados.Count) ($achados -join ",")

# **O `matplotlib` VIAJOU, e o oraculo nao e o diretorio sozinho.** Ele traz extensao binaria E
# dado: `mpl-data` leva as fontes que o desenho usa, e um congelado com o pacote e sem elas
# levanta na primeira figura. Sao as duas classes da secao 6 da docstring do `refino-desktop.spec`
# no mesmo pacote, entao as duas se afirmam, e a segunda tem PISO de contagem, porque uma pasta
# vazia existe.
$pastaDoMatplotlib = Join-Path $Interno "matplotlib"
$dadosDoMatplotlib = Join-Path $pastaDoMatplotlib "mpl-data"
$fontes = 0
if (Test-Path $dadosDoMatplotlib) {
    $fontes = @(Get-ChildItem $dadosDoMatplotlib -Recurse -File -Include *.ttf, *.afm `
            -ErrorAction SilentlyContinue).Count
}
$faltaMatplotlib = 0
if (-not (Test-Path $pastaDoMatplotlib)) {
    $faltaMatplotlib = 2
} elseif ($fontes -lt 20) {
    $faltaMatplotlib = 1
}
ParaSe "o matplotlib viajou, com as fontes de mpl-data" $faltaMatplotlib `
    "$fontes fontes em mpl-data (piso 20; na venv de desenvolvimento sao 101; codigo 2 = nem o diretorio esta la)"

# E o backend do Qt, que e o nome que o analisador NAO ve — o `matplotlib` resolve-o por string a
# partir de "QtAgg". Sem ele o congelado cai para `Agg`, DESENHA, e a figura nao aparece: desfecho
# de sucesso com tela vazia.
#
# DOIS oraculos em OU. Um modulo de Python puro pode viajar no PYZ ou
# como `.pyc` solto, conforme o `noarchive` e a copia para o worker — exigir so um deles e
# reprovar o caso de sucesso, que e o defeito da primeira versao deste roteiro. Aqui basta um.
#
# `$toc` e definido AQUI e nao mais abaixo, onde estava: dois passos o leem agora, e duas
# definicoes da mesma coisa sao a forma de uma delas envelhecer sozinha.
$toc = Join-Path $Raiz "build\refino-desktop\PYZ-00.toc"
$backendSolto = Join-Path $Interno "matplotlib\backends\backend_qtagg.pyc"
$backendNoPyz = 0
if (Test-Path $toc) {
    $backendNoPyz = @(Select-String -Path $toc -Pattern "backend_qtagg" -SimpleMatch).Count
}
$temBackend = (Test-Path $backendSolto) -or ($backendNoPyz -gt 0)
ParaSe "o backend QtAgg viajou" ([int](-not $temBackend)) `
    "solto=$(Test-Path $backendSolto) no PYZ=$backendNoPyz (sem ele o congelado desenha em Agg e a janela fica vazia)"

# E a direcao oposta: o que TEM de estar la. Sem esta metade, um `_internal` vazio passaria na
# asserção de ausencia acima.
#
# **PRESENCA DE DIRETORIO NAO E O ORACULO, e a primeira versao deste script reprovou por isso.** Ela
# procurou `_internal\alembic`, `_internal\httpx2` e `_internal\pyarrow`, nao achou nenhum, e
# declarou tres ausencias — das quais DUAS eram falsas. O PyInstaller poe modulo de Python PURO no
# arquivo PYZ, e so ganha diretorio em `_internal` o pacote que traz extensao binaria ou dado.
#
# Entao a conferencia e por CLASSE de pacote, com oraculo diferente para cada:
# O oraculo da classe (a) sao os `.pyc` soltos, e nao entradas de `alembic` e `httpx2` no
# `build\refino-desktop\PYZ-00.toc`: com `noarchive=True` o PYZ fica vazio por desenho, e um
# oraculo de PYZ declararia os pacotes AUSENTES com eles a viajar. Um oraculo que reprova o caso de
# sucesso e afrouxado no primeiro dia.
#
# (a) Python puro -> `.pyc` soltos em `_internal`, ao lado dos `.pyd` do mesmo pacote.
#
# **A pasta e a RAIZ, e isso e requisito e nao arrumacao.** Com o PYZ de volta, os puros do
# APLICATIVO vivem dentro do `refino.exe` e a copia solta e a unica que o `pythonw.exe` embarcado
# alcanca. Ela tem de ficar na mesma pasta que os binarios: posta em `_internal\_worker`, o
# `numpy` deixa de importar (medido), porque o pacote fica partido em duas pastas e
# o Python so usa uma. **Este portao confere a copia solta, nao o arranque** — quem confere o
# arranque e `scripts/prova_interpretador_embarcado.py`.
$puros = @{ "alembic" = 10; "httpx2" = 10; "sklearn" = 100; "numpy" = 50 }
$semModulos = @()
foreach ($nome in $puros.Keys) {
    $pasta = Join-Path $Interno $nome
    $n = 0
    if (Test-Path $pasta) { $n = @(Get-ChildItem $pasta -Recurse -Filter *.pyc).Count }
    Write-Host ("      soltos: {0,-12} {1,4} .pyc (minimo {2})" -f $nome, $n, $puros[$nome])
    if ($n -lt $puros[$nome]) { $semModulos += "$nome($n)" }
}
ParaSe "os pacotes de Python puro viajaram (.pyc soltos em _internal)" ($semModulos.Count) ($semModulos -join ",")

# **E o APLICATIVO tem de os ter no PYZ, que e a outra metade do par.** Sem esta asserção, um
# `noarchive` que voltasse a `True` por acidente passaria no passo acima — os `.pyc` estariam
# soltos — e o arranque a frio triplicaria em silencio, que e o defeito que esta mudanca fecha.
$noPyz = 0
if (Test-Path $toc) { $noPyz = @(Select-String -Path $toc -Pattern "'sklearn" -SimpleMatch).Count }
ParaSe "os modulos puros do aplicativo estao no PYZ" ([int]($noPyz -lt 1)) `
    "entradas de sklearn no PYZ-00.toc: $noPyz (se for 0, o noarchive voltou a True)"

# **E a contagem TOTAL de arquivos, que e a quantidade que o arranque a frio de facto paga.**
#
# Medido: com os 2.708 `.pyc` soltos em `_internal`, a primeira abertura de conteudo
# que a maquina nunca viu custou **84,1 s** — contra 2,8 s da segunda. Ler os 400 MB custa 3 s, e o
# resto e o antivirus a varrer arquivo a arquivo. O teto e generoso de proposito: o que ele guarda
# e a ORDEM DE GRANDEZA, e uma subida de mil arquivos e sinal de que a duplicacao escapou.
$totalArquivos = @(Get-ChildItem $Interno -Recurse -File).Count
ParaSe "a contagem de arquivos de _internal ficou sob o teto" ([int]($totalArquivos -gt 8000)) `
    "$totalArquivos arquivos (teto 8000; com os .pyc soltos na raiz eram 4.126 e o arranque custava 84 s)"

# (b) extensao binaria -> o diretorio em `_internal`.
$comBinario = @("PySide6", "numpy", "scipy", "pandas", "sklearn")
$faltando = @()
foreach ($nome in $comBinario) {
    if (-not (Test-Path (Join-Path $Interno $nome))) { $faltando += $nome }
}
$faltando = @($faltando)
ParaSe "os pacotes com extensao binaria viajaram" ($faltando.Count) ($faltando -join ",")

# As migracoes viajaram? O alembic carrega-as por CAMINHO.
#
# O numero esperado e DERIVADO da arvore e nao pinado. Com o numero escrito a mao, uma migracao
# nova fez o portao reprovar com o congelamento CERTO e parar o embalamento inteiro.
#
# > Portao que PINA uma lista le a lista de ontem. Quem mais le esta lista e o `alembic`, que as
# > carrega por caminho, entao a fonte da verdade e o diretorio do repositorio.
#
# **A comparacao continua a ser de IGUALDADE e nao de minimo.** Migracao a MENOS quebra o
# `upgrade head`; migracao a MAIS no pacote significa que algo entrou ali que a arvore nao
# tem, e as duas direcoes sao defeito.
$versoesNaArvore = Join-Path $Raiz "src\refino\persistence\migrations\versions"
$nEsperado = @(Get-ChildItem $versoesNaArvore -Filter *.py -ErrorAction SilentlyContinue).Count
$versoes = Join-Path $Interno "refino\persistence\migrations\versions"
$nVersoes = if (Test-Path $versoes) { (Get-ChildItem $versoes -Filter *.py).Count } else { 0 }
# A ANCORA: zero esperadas significa que a varredura da arvore falhou, e ai `0 - 0` daria OK.
if ($nEsperado -lt 1) { $nEsperado = -1 }
ParaSe "as migracoes viajaram" ($nEsperado - $nVersoes) "$nVersoes de $nEsperado em versions/"

# **Nenhum `__pycache__` viaja nos dados do pacote.** O `datas=` levava o diretorio das
# migracoes inteiro, e com ele os `.pyc` que a suite deixa na arvore de quem congela: cinco na
# `windows-2026.10.02.1`, com o caminho desta maquina dentro. O `.spec` passou a listar os
# arquivos um a um, e este passo confere o EFEITO no pacote, nao a forma do `.spec`.
#
# O codigo de `refino` viaja como `.pyc` solto em `_internal\refino`, sem `__pycache__`, entao
# qualquer `__pycache__` ali veio de um `datas=`. A ancora contra a vacuidade e o passo de
# cima: ele so passa com `versions/` presente dentro desta mesma arvore.
$pycacheNosDados = @(Get-ChildItem (Join-Path $Interno "refino") -Recurse -Directory -Filter "__pycache__" -ErrorAction SilentlyContinue).Count
ParaSe "nenhum __pycache__ nos dados do pacote" $pycacheNosDados "$pycacheNosDados diretorio(s) __pycache__ em _internal\refino"

# ----------------------------------------------------------------------------
# CENARIO 3 - o tamanho da pasta: DUAS ASSERCOES, e nao uma linha de log
# ----------------------------------------------------------------------------
#
# Uma linha `Relata "tamanho medido" 0 "..."`, com o ZERO literal, NUNCA poderia reprovar:
# `Relata` so incrementa `$script:Falhas` quando o codigo difere de zero, e o resultado seria
# linha de log com aparencia de verificacao.
#
# Sao duas assercoes porque sao duas perguntas:
#
#   1. O DELTA DE MB contra a base, nas duas direcoes. A de baixo e a ANCORA: com a pasta ausente
#      ou vazia, `$bytes` e nulo e `$mb` da 0, e um desvio grande para baixo tambem para.
#   2. O DELTA DE CONTAGEM DE ARQUIVOS, pelo mesmo par de razoes, e porque a contagem e a
#      quantidade que o arranque a frio de facto paga: ler os bytes custa 3 s e abrir custa 84.
#
# Nao ha teto absoluto. Quem reprova sao os dois detectores
# de regressao, e eles nao tem par causal registado, o que fica como divida declarada.
#
# A PROCEDENCIA DE CADA NUMERO:
#
#   $BaseMB = 724.7               medidos no primeiro congelamento com o R embarcado, de `7c51847`.
#   $BaseArquivos = 8322          Composicao: `_internal` 350,3 MB / 4766, `refino.exe` 32,6 MB,
#                                 `saidas` 102,2 MB / 34, os dez documentos 1,8 MB, `R` 220,2 MB /
#                                 3506 e `vias-dados` 17,4 MB / 5. O "MB" deste roteiro e MiB,
#                                 porque ele divide por 1MB.
#
#                                 `saidas` sao 14% da pasta, e isso e o que o delta de dois lados
#                                 aceita comparar: elas sao deterministicas dado o mesmo dado e o
#                                 mesmo codigo, mas quem mudar o corpus move este numero sem que
#                                 nada de dependencia tenha mudado.
#   $CrescimentoMaximoMB = 45     MEDIDO NO CONGELADO, e nao estimado na venv. Os nove pacotes que
#                                 o pino do `matplotlib` arrasta ocupam 32,4 MB em 457 arquivos
#                                 dentro de `_internal` (`matplotlib` 18,6 MB/312, `PIL` 11,8 MB/82,
#                                 e os outros sete somando 2,0 MB/63). 45 e o numero inclusivo com
#                                 ~40% de folga. O que ele guarda e a ORDEM DE GRANDEZA: `pyarrow`
#                                 (78 MB) ou o QtWebEngine (297 MB) a voltar pela porta das
#                                 dependencias reprova.
#   $CrescimentoMaximoArquivos    a mesma medicao: 457 arquivos inclusivos, 425 genuinamente
#     = 650                       novos. 650 e ~40% de folga sobre o inclusivo.
#
# Medir na venv erraria para MAIS: os mesmos nove pacotes ocupam 51,4 MB em 1.267 arquivos na
# venv de desenvolvimento, porque o PyInstaller nao leva `__pycache__`, testes nem metadados de
# distribuicao. Um limite que erra para mais e um limite que nao dispara.
#
# AS ASSERCOES DE TAMANHO SO CORREM SOBRE UMA CONSTRUCAO DE ENTREGA, e isso e o que torna os
# deltas de dois lados seguros. O roteiro para no passo "as tres pastas de saida prontas viajam",
# que vem ANTES do CENARIO 3: sem as saidas geradas nao se chega aqui, entao a comparacao contra
# `$BaseMB`, que inclui as saidas, e sempre entre artefatos do mesmo genero. Uma construcao
# so-de-desenvolvimento mede 358,1 MB e nunca alcanca esta linha.
#
# As quatro sao parametros para que a medicao as corrija sem editar o roteiro.
$bytes = (Get-ChildItem $Dist -Recurse -File | Measure-Object -Property Length -Sum).Sum
$mb = [math]::Round($bytes / 1MB, 1)
$nArquivos = (Get-ChildItem $Dist -Recurse -File).Count
$deltaMB = [math]::Round($mb - $BaseMB, 1)
$deltaArquivos = $nArquivos - $BaseArquivos

# O tamanho continua impresso, sem teto; quem reprova sao os dois detectores abaixo.
Write-Host ("tamanho da pasta: " + $mb + " MB em " + $nArquivos + " arquivos (sem teto)")

ParaSe "o tamanho ficou dentro do crescimento previsto" `
    ([int]([math]::Abs($deltaMB) -gt $CrescimentoMaximoMB)) `
    "$mb MB contra base de $BaseMB MB: delta $deltaMB MB (maximo $CrescimentoMaximoMB, nos dois sentidos)"

ParaSe "a contagem de arquivos ficou dentro do crescimento previsto" `
    ([int]([math]::Abs($deltaArquivos) -gt $CrescimentoMaximoArquivos)) `
    "$nArquivos arquivos contra base de ${BaseArquivos}: delta $deltaArquivos (maximo $CrescimentoMaximoArquivos, nos dois sentidos)"

# Os cinco maiores diretorios de `_internal`, para o numero ter populacao e nao ser um total solto.
$maiores = Get-ChildItem $Interno -Directory | ForEach-Object {
    $s = (Get-ChildItem $_.FullName -Recurse -File -ErrorAction SilentlyContinue |
        Measure-Object -Property Length -Sum).Sum
    [pscustomobject]@{ Nome = $_.Name; MB = [math]::Round($s / 1MB, 1) }
} | Sort-Object MB -Descending | Select-Object -First 8
$maiores | ForEach-Object { Write-Host ("      {0,-28} {1,8} MB" -f $_.Nome, $_.MB) }

if ($SoConstroi) { Resumo; exit ([int]($script:Falhas -gt 0)) }

# ----------------------------------------------------------------------------
# CENARIO 4b — O DUPLO CLIQUE: o binario arranca SEM REDIRECAO NENHUMA
# ----------------------------------------------------------------------------
#
# **Este cenario e o portao que faltava, e a ausencia dele custou um binario inutilizavel com 21
# passos verdes.** Todos os outros cenarios lancam com `-RedirectStandardOutput` e
# `-RedirectStandardError` — o comentario do cenario 4 ate explica por que: *"o binario e
# `--windowed`: nao ha console de onde ler"*. Criar handles para poder ler e EXACTAMENTE o que
# mascara o defeito, e o arnes media o binario na unica condicao em que ele nao pode aparecer.
#
# Medido sobre um binario anterior ao conserto:
#
#   com stdout OU stderr redirecionado, ate para NUL  -> janela em 3 s
#   sem handle nenhum (duplo clique, explorer.exe)    -> morre em 2 s, codigo 1, em silencio
#
# A causa era `uvicorn/logging.py:42` a chamar `sys.stdout.isatty()` num `sys.stdout` que e `None`.
# O conserto e `arranque.garante_fluxos`, e ESTE cenario e o que o mantem consertado.
#
# **`Start-Process` SEM nenhum `-Redirect*` usa `UseShellExecute=$true`**, que e o mesmo caminho do
# Explorer. Com qualquer `-Redirect*` ele passa a `CreateProcess` com handles — e o defeito some.
# Nao ha meio-termo a medir: a presenca de um unico `-Redirect*` ja o mascara.
#
# **A JANELA e o oraculo aqui, e nao a porta.** Um binario que subisse o servidor e nao desenhasse
# nada passaria numa asserção de porta; `MainWindowHandle` e o que separa *"o processo vive"* de
# *"o profissional ve alguma coisa"*.
#
# **CORRE ANTES DO CENARIO 4, e a colocacao e requisito.** O cenario 4 levanta uma instancia que
# fica DE PE ate o cenario 6 — o 5b reusa as `$portas` dela. Com este depois, havia duas instancias
# a partilhar o mesmo SQLite de `%LOCALAPPDATA%`, e `alembic` concorrente sobre SQLite e disputa a
# espera de acontecer. Aqui ele levanta a sua propria, mede, e encerra-a.
$procDC = Start-Process -FilePath $Exe -PassThru
$janelaDC = 0
$segundosDC = 0
# O teto e 150 s por CAUSA MEDIDA: o primeiro arranque com o cache frio sobre 4.169 arquivos
# custou 59,4 s, e um teto de 60 s reprovava com a janela a caminho. 150 s da 2,5x de margem sobre
# o pior caso medido, que e o que se quer de um teto contra lentidao: a maquina de quem recebe pode
# ter disco mais lento e antivirus a varrer cada `.pyc` pela primeira vez. Portao que reprova por
# lentidao a equipa aprende a ignorar, e um que passa com a janela a nunca aparecer nao mede nada.
#
# O ORACULO E O TITULO E NAO O PUNHO. A tela de arranque e desenhada pelo carregador do
# PyInstaller em Tk, aparece ao primeiro segundo, e tem punho de janela como qualquer outra:
# `MainWindowHandle -ne 0` dava SUCESSO com o aplicativo ainda a carregar, e o resto do cenario
# corria contra ele sem log, sem banco e com o `GET /health` a expirar. A propriedade e "a janela
# DO APLICATIVO apareceu", e nao "apareceu uma janela".
#
# `refino` no titulo separa as duas: a do aplicativo comeca por `Refino`, e a do carregador
# chama-se `tk`. O `-like` e sobre o inicio, para o titulo poder ganhar sufixo sem o portao
# reprovar por edicao de texto, e o `-like` do PowerShell NAO diferencia maiusculas.
for ($t = 1; $t -le 150; $t++) {
    Start-Sleep -Seconds 1
    $procDC.Refresh()
    if ($procDC.HasExited) { break }
    if ($procDC.MainWindowHandle -ne 0 -and $procDC.MainWindowTitle -like "refino*") {
        $janelaDC = 1
        $segundosDC = $t
        break
    }
}
$procDC.Refresh()
$vivoDC = -not $procDC.HasExited
$detalheDC = if ($janelaDC -eq 1) {
    "janela em ${segundosDC}s, titulo: $($procDC.MainWindowTitle)"
} elseif ($vivoDC) {
    "150s vivo e SEM A JANELA DO APLICATIVO (titulo visto: '$($procDC.MainWindowTitle)')"
} else {
    "MORREU com codigo $($procDC.ExitCode) - e o defeito que garante_fluxos conserta"
}
Relata "duplo clique: a janela aparece sem redirecao de saida" (1 - $janelaDC) $detalheDC

# **O log tem de existir e ter recebido algo.** `garante_fluxos` liga os fluxos ao `refino.log`
# da PASTA DO EXECUTAVEL.
#
# Um arquivo ausente significa que o recuo para `NUL` entrou,
# e ai o arranque funciona e o diagnostico nao existe. A distincao vai medida em vez de suposta.
$logApp = Join-Path $Dist "refino.log"
$temLog = if (Test-Path $logApp) { 0 } else { 1 }
$tamLog = if (Test-Path $logApp) { (Get-Item $logApp).Length } else { -1 }
Relata "duplo clique: o log fica NA PASTA do executavel" $temLog "$logApp ($tamLog bytes)"

# **E o banco tambem, e esta assercao so passou a ser possivel com a mudanca de lugar:** antes
# o banco vivia fora da pasta e o congelamento nao tinha como afirmar nada sobre ele. E o que
# quem recebe a pasta devolve.
$dbApp = Join-Path $Dist "refino.db"
$temDb = if (Test-Path $dbApp) { 0 } else { 1 }
$tamDb = if (Test-Path $dbApp) { (Get-Item $dbApp).Length } else { -1 }
Relata "duplo clique: o banco fica NA PASTA do executavel" $temDb "$dbApp ($tamDb bytes)"

if ($vivoDC) { Stop-Process -Id $procDC.Id -Force -ErrorAction SilentlyContinue }
Start-Sleep -Seconds 2

# ----------------------------------------------------------------------------
# CENARIO 4 — o binario arranca, sobe o servidor no loopback, e uma rota responde
# ----------------------------------------------------------------------------
#
# O binario e `--windowed`: nao ha console de onde ler. O que se observa e a PORTA, e o
# `REFINO_API_BASE`... nao: o servidor embutido escolhe porta efemera. Entao a sonda le a saida do
# proprio processo por arquivo — o `arranque` escreve o desfecho — e observa as conexoes de loopback
# do PID.
$logSaida = Join-Path $Saidas "arranque.out"
$logErro = Join-Path $Saidas "arranque.err"
Remove-Item -Force $logSaida, $logErro -ErrorAction SilentlyContinue

$proc = Start-Process -FilePath $Exe -PassThru -RedirectStandardOutput $logSaida `
    -RedirectStandardError $logErro
Start-Sleep -Seconds 12

$vivo = if ($proc.HasExited) { 1 } else { 0 }
$detalhe = if ($proc.HasExited) { "saiu com $($proc.ExitCode)" } else { "PID $($proc.Id)" }
Relata "o binario arrancou e continua de pe" $vivo $detalhe

# As portas em LISTEN do PID, so loopback. `-ErrorAction SilentlyContinue` porque `Get-NetTCPConnection`
# levanta quando nao ha nenhuma, e "nenhuma" e um desfecho a medir e nao um erro a propagar.
$portas = @()
if (-not $proc.HasExited) {
    $portas = Get-NetTCPConnection -State Listen -OwningProcess $proc.Id -ErrorAction SilentlyContinue |
        Where-Object { $_.LocalAddress -eq "127.0.0.1" -or $_.LocalAddress -eq "::1" }
}
# **`@()` NAO e cerimonia, e o defeito que ele conserta esta registrado.** `$portas.Count`
# num UNICO objecto de `Get-NetTCPConnection` nao devolve 1: devolve nada, e `-ge 1` da
# falso. O sintoma foram TRES falhas a apontar para o binario nao servir, com a propria
# linha de detalhe a imprimir `porta(s): 50373` — porque `ForEach-Object` itera um escalar
# sem problema e `.Count` nao. O arnes relatou um desfecho que ele nao mediu.
$portas = @($portas)
$temPorta = if ($portas.Count -ge 1) { 0 } else { 1 }

# **A porta que a API atende, escolhida por `/health` e nao por posicao.** `$portas` continua a
# ser o insumo das duas asserções seguintes — quantas ha, e se alguma esta fora do loopback —, e
# esta variavel e a que os tres exercicios usam para FALAR com o binario.
$baseQueServe = if ($proc.HasExited) { "" } else { PortaQueServe -Processo $proc -TetoSeg 60 }
$portaQueServe = if ($baseQueServe) { [int](($baseQueServe -split ":")[-1]) } else { 0 }
Relata "alguma porta do PID responde /health" ([int](-not $baseQueServe)) "base: $baseQueServe"
Relata "o servidor esta em LISTEN no loopback" $temPorta "$($portas.Count) porta(s): $(($portas | ForEach-Object { $_.LocalPort }) -join ',')"

# **Nada fora do loopback.** O `bind` e 127.0.0.1 e ha portao sobre isso na suite; aqui e o binario
# real, e a asserção e de AUSENCIA — com a de presenca acima a sustenta-la, senao um processo sem
# porta nenhuma passaria.
$foraDoLoopback = @()
if (-not $proc.HasExited) {
    $foraDoLoopback = Get-NetTCPConnection -State Listen -OwningProcess $proc.Id -ErrorAction SilentlyContinue |
        Where-Object { $_.LocalAddress -ne "127.0.0.1" -and $_.LocalAddress -ne "::1" }
}
$foraDoLoopback = @($foraDoLoopback)
Relata "nada em LISTEN fora do loopback" ($foraDoLoopback.Count) (($foraDoLoopback | ForEach-Object { "$($_.LocalAddress):$($_.LocalPort)" }) -join ",")

# Uma rota real responde. A porta e efemera, entao usa-se a primeira do LISTEN do PID.
$respondeu = 1
$corpo = ""
if ($portas.Count -ge 1) {
    $porta = $portaQueServe
    try {
        $r = Invoke-WebRequest -Uri "http://127.0.0.1:$porta/health" -UseBasicParsing -TimeoutSec 20
        $respondeu = if ($r.StatusCode -eq 200) { 0 } else { 1 }
        $corpo = $r.Content
    } catch {
        $respondeu = 1
        $corpo = $_.Exception.Message
    }
}
Relata "GET /health respondeu 200" $respondeu ($corpo.Substring(0, [Math]::Min(60, $corpo.Length)))

# ----------------------------------------------------------------------------
# CENARIO 5 — PROSA ACENTUADA sai correta do binario congelado
# ----------------------------------------------------------------------------
#
# **Comparacao de BYTES lidos de arquivo, nunca do que aparece na consola** — o PowerShell mastiga
# a exibicao, e isso nao e defeito do binario.
# **O oraculo e a CLASSE DE BYTE, e nao um literal acentuado no padrao.** A primeira versao deste
# passo comparava com `-match "execu[c-cedilha]"` e reprovou — e o binario estava certo. O defeito
# era do PADRAO, num script que declara no cabecalho que todo relato dele e so-ASCII: a regra era
# sobre a SAIDA, e o literal acentuado entrou pela ENTRADA.
#
# A fonte da prosa e `POST /matrizes/recomendar`, que devolve o `rationale` em prosa do projeto.
# Nao e `/openapi.json`: a documentacao interativa esta desligada no congelado, a rota responde
# 404, e o cenario reprovaria com o binario CERTO.
#
# O limiar de 100 vem de medicao, e a populacao vai declarada: medido sobre `neg_c18`, a resposta
# traz 2.635 bytes, 148 nao-ASCII, 12 distintos. A folga e estreita (1,48x), e um limiar que so
# passa por folga larga nao mede o que diz medir.
#
# SEM dado de referencia o passo NAO E SALTADO, e a razao e que ele mede o BINARIO e nao o dado.
# A fonte de recurso e `GET /multiversos/<id-inexistente>`, cujo 404 traz prosa do projeto sem
# precisar de arquivo nenhum: medido, 87 bytes e 8 nao-ASCII (o corpo repete o identificador, entao
# o comprimento depende dele). O limiar daquele caminho e 5, e e outro numero porque a populacao e
# outra; os dois vao declarados no relato, que nomeia a fonte que de facto respondeu. Saltar seria
# perder a unica verificacao de codificacao que o congelamento tem.
#
# `curl.exe` e nao `Invoke-WebRequest`, pelas duas razoes que o cenario 5b ja declara mais abaixo: o
# `-Form` do segundo so existe no PowerShell 6+ e este script corre no 5.1, e o multipart que ele
# monta muda o nome do arquivo — e o `mode` da rota e conferido contra o CONTEUDO.
#
# ## O CODIGO DE SAIDA E LIDO, e o arquivo e APAGADO ANTES
#
# Medido: com o servidor morto, `curl.exe` sai 7 e NAO trunca o arquivo de saida. E `$Saidas`
# (`$env:TEMP\refino-u10`) e criado com `-Force` e nunca limpo, entao a partir da segunda execucao
# `prosa-acentuada.json` e um artefato persistente: lendo so o conteudo, o passo leria os bytes
# multibyte de ONTEM e aprovaria com o binario morto.
#
# Tres consertos, e nenhum e opcional:
#   1. o arquivo e apagado ANTES do curl, para que "o arquivo existe" signifique "o curl escreveu";
#   2. o codigo de saida do `Roda` e LIDO, e nao-zero reprova antes de qualquer byte ser contado;
#   3. `--max-time`: um servidor que aceite e nao responda penduraria o cenario sem teto.
#
# Tres asserções sobre os bytes, e nenhuma precisa de um acento escrito aqui:
#   1. ha byte multibyte de verdade (>127), e em quantidade;
#   2. decodificar como UTF-8 nao produz U+FFFD — o caractere de substituicao, que e o que aparece
#      quando a codificacao quebrou;
#   3. nao ha escape do tipo `\uXXXX`, que seria o JSON a fugir do problema em vez de o resolver.
$acentoOk = 1
$amostra = ""
if ($portas.Count -ge 1) {
    $porta = $portaQueServe
    $arquivoJson = Join-Path $Saidas "prosa-acentuada.json"
    $medidasAcento = Join-Path $Raiz "serrf\data\input\measurements_neg_c18.csv"
    $listaAcento   = Join-Path $Raiz "serrf\data\input\Sample_List_neg_c18.csv"
    $temDado = (Test-Path $medidasAcento) -and (Test-Path $listaAcento)
    $fonte = if ($temDado) { "POST /matrizes/recomendar" } else { "GET /multiversos/<inexistente>" }
    $minimo = if ($temDado) { 100 } else { 5 }
    # (1) O arquivo de ontem sai da frente. Sem isto, "o arquivo existe" nao prova que o curl
    # escreveu — e `$Saidas` nunca e limpo.
    Remove-Item $arquivoJson -Force -ErrorAction SilentlyContinue
    try {
        if ($temDado) {
            $codCurl = Roda -Executavel "curl.exe" -Argumentos @(
                "-s", "--max-time", "60",
                "-F", "measurements=@$medidasAcento;type=text/csv",
                "-F", "sample_list=@$listaAcento;type=text/csv",
                "-F", "mode=neg_c18",
                "-o", $arquivoJson,
                "http://127.0.0.1:$porta/matrizes/recomendar"
            ) -Log (Join-Path $Saidas "prosa-acentuada.codigo")
        } else {
            $codCurl = Roda -Executavel "curl.exe" -Argumentos @(
                "-s", "--max-time", "30", "-o", $arquivoJson,
                "http://127.0.0.1:$porta/multiversos/sonda-que-nao-existe"
            ) -Log (Join-Path $Saidas "prosa-acentuada.codigo")
        }
        # (2) O codigo de saida decide ANTES de qualquer byte ser contado. Sem conexao o curl sai
        # 7, e sem esta linha o passo cairia no `ReadAllBytes` de um arquivo que nao existe — ou,
        # pior, de um que sobrou.
        if ($codCurl -ne 0) {
            $amostra = "fonte=$fonte curl saiu $codCurl (o servidor nao respondeu)"
            Relata "prosa acentuada correta no congelado" 1 $amostra
            $acentoOk = -1
        }
        $bytes = [System.IO.File]::ReadAllBytes($arquivoJson)
        $naoAscii = 0
        foreach ($b in $bytes) { if ($b -gt 127) { $naoAscii++ } }
        $texto = [System.Text.Encoding]::UTF8.GetString($bytes)
        $substitutos = ([regex]::Matches($texto, [string][char]0xFFFD)).Count
        $escapes = ([regex]::Matches($texto, [regex]::Escape([string][char]92 + "u00"))).Count
        $acentoOk = if (($naoAscii -ge $minimo) -and ($substitutos -eq 0) -and ($escapes -eq 0)) {
            0
        } else {
            1
        }
        $amostra = "fonte=$fonte minimo=$minimo bytes=$($bytes.Length) multibyte=$naoAscii substitutos=$substitutos escapes=$escapes"
    } catch {
        $amostra = $_.Exception.Message
    }
}
Relata "prosa acentuada correta no congelado" $acentoOk $amostra

# ----------------------------------------------------------------------------
# CENARIO 5b - O FLUXO COMPLETO, contra o servidor DENTRO do binario
# ----------------------------------------------------------------------------
#
# Os botoes da janela disparam exactamente as rotas abaixo.
#
# **O QUE ESTE PASSO MEDE, e o que ele NAO mede.** Ele exercita a CADEIA que o clique percorre,
# contra o servidor embutido no `.exe`: validar, diagnosticar, recomendar, corrigir e reclamar os
# bytes. **O CLIQUE em si nao e exercitado** — o binario e `--windowed`, e automatizar Qt de fora
# exige ferramenta de UI que este projeto nao tem. O que se afirma e *"o caminho que o botao dispara
# funciona dentro do congelado"*, e nao *"o botao funciona"*.
#
# A diferenca entre os dois vai declarada em vez de suposta: e a distancia entre "composto" e
# "operavel".
if ($portas.Count -ge 1) {
    $porta = $portaQueServe
    $modo = "neg_c18"
    $medidas = Join-Path $Raiz "data\reference\input\measurements_$modo.csv"
    $lista = Join-Path $Raiz "data\reference\input\Sample_List_$modo.csv"

    # O alvo e `serrf/data/input` e NAO `data/reference/input`.
    # Aqueles dois sao SYMLINKS versionados, e o Windows traz-os como arquivo de TEXTO de 22 bytes
    # sem `core.symlinks=true` mais Modo de Desenvolvedor — medido neste clone. O alvo real do link
    # e `serrf/data/input`. A tentativa pelo link fica primeiro, para o dia em que o clone o resolva.
    if (-not (Test-Path $medidas)) {
        $medidas = Join-Path $Raiz ("serrf" + [IO.Path]::DirectorySeparatorChar + "data" +
            [IO.Path]::DirectorySeparatorChar + "input" +
            [IO.Path]::DirectorySeparatorChar + "measurements_$modo.csv")
        $lista = Join-Path $Raiz ("serrf" + [IO.Path]::DirectorySeparatorChar + "data" +
            [IO.Path]::DirectorySeparatorChar + "input" +
            [IO.Path]::DirectorySeparatorChar + "Sample_List_$modo.csv")
    }

    if (-not (Test-Path $medidas) -or -not (Test-Path $lista)) {
        # Ausencia de dado de referencia e SALTO declarado e nao falha, e nunca dado inventado.
        Relata "fluxo completo: dado de referencia ausente, SALTADO" 0 $medidas
    } else {
        # `curl.exe` e nao `Invoke-WebRequest`: o segundo monta multipart de forma que muda o nome
        # do arquivo, e o `mode` da rota e conferido contra o CONTEUDO. `curl` esta no Windows 11.
        # O corpo vai para um arquivo e o codigo para OUTRO, e a razao e medida. Com `-o -` e
        # `-w "<crase>n%{http_code}"` o codigo vem VAZIO: a crase-n e consumida pelo PowerShell
        # antes de chegar ao `curl`, e o corpo e o codigo ficam no mesmo fluxo sem separador. As
        # quatro POSTs relatavam FALHA com o fluxo a funcionar: o corpo chegava, o `identificador`
        # era extraido dele, e so o codigo nao.
        $comuns = @(
            "-s",
            "-F", "measurements=@$medidas;type=text/csv",
            "-F", "sample_list=@$lista;type=text/csv",
            "-F", "mode=$modo"
        )

        foreach ($rota in @("validar", "diagnosticar", "recomendar")) {
            $corpo = Join-Path $Saidas "fluxo-$rota.json"
            $cod = Join-Path $Saidas "fluxo-$rota.codigo"
            $c = Roda -Executavel "curl.exe" -Argumentos (
                $comuns + @("-o", $corpo, "-w", "%{http_code}",
                            "http://127.0.0.1:$porta/matrizes/$rota")
            ) -Log $cod
            $codigo = if (Test-Path $cod) { (Get-Content $cod -Raw).Trim() } else { "" }
            $bytes = if (Test-Path $corpo) { (Get-Item $corpo).Length } else { 0 }
            $ok = if ($codigo -eq "200" -and $bytes -gt 0) { 0 } else { 1 }
            Relata "fluxo: POST /matrizes/$rota" $ok "curl=$c http=$codigo bytes=$bytes"
        }

        # **A perna do artefato: `202` com identificador, e os bytes vem do `GET`.** E o mesmo
        # encadeamento que `_encaminha` faz na janela, medido aqui contra o congelado.
        $arqCorrigir = Join-Path $Saidas "fluxo-corrigir.json"
        $codCorrigir = Join-Path $Saidas "fluxo-corrigir.codigo"
        $c = Roda -Executavel "curl.exe" -Argumentos (
            $comuns + @("-o", $arqCorrigir, "-w", "%{http_code}",
                        "http://127.0.0.1:$porta/matrizes/corrigir")
        ) -Log $codCorrigir
        $codigo = if (Test-Path $codCorrigir) { (Get-Content $codCorrigir -Raw).Trim() } else { "" }
        $texto = if (Test-Path $arqCorrigir) { Get-Content $arqCorrigir -Raw } else { "" }
        $identificador = ""
        if ($texto -match '"identificador"\s*:\s*"([0-9a-f]+)"') { $identificador = $Matches[1] }
        $ok = if ($codigo -eq "202" -and $identificador) { 0 } else { 1 }
        Relata "fluxo: POST /matrizes/corrigir" $ok "http=$codigo id=$identificador"

        # **Reclama ate ter bytes, com teto.** E o desenho declarado em `_encaminha`: um `202` faz
        # clicar outra vez, e a rota devolve o MESMO identificador. Aqui o laco e do arnes.
        $bytesDoArtefato = 0
        $ultimo = ""
        if ($identificador) {
            for ($tentativa = 1; $tentativa -le 40; $tentativa++) {
                $arqTrab = Join-Path $Saidas "fluxo-trabalho.txt"
                Roda -Executavel "curl.exe" -Argumentos @(
                    "-s", "-o", $arqTrab, "-w", "%{http_code}",
                    "http://127.0.0.1:$porta/trabalhos/$identificador"
                ) -Log (Join-Path $Saidas "fluxo-trabalho-codigo.txt") | Out-Null
                $ultimo = (Get-Content (Join-Path $Saidas "fluxo-trabalho-codigo.txt") -Raw).Trim()
                if ($ultimo -eq "200") {
                    $bytesDoArtefato = (Get-Item $arqTrab).Length
                    break
                }
                Start-Sleep -Seconds 3
            }
        }
        $ok = if ($bytesDoArtefato -gt 0) { 0 } else { 1 }
        Relata "fluxo: GET /trabalhos/{id} devolveu o artefato" $ok `
            "http=$ultimo bytes=$bytesDoArtefato"

        # **O artefato e CSV de verdade**, e nao um corpo de erro com 200. A asserção e sobre o
        # conteudo: cabecalho com virgula e mais de uma linha.
        if ($bytesDoArtefato -gt 0) {
            $primeiras = Get-Content (Join-Path $Saidas "fluxo-trabalho.txt") -TotalCount 2
            $eCsv = if (($primeiras.Count -ge 2) -and ($primeiras[0] -like "*,*")) { 0 } else { 1 }
            Relata "fluxo: o artefato e CSV com cabecalho e linha" $eCsv `
                "linhas lidas=$($primeiras.Count)"
        }
    }
}

# ----------------------------------------------------------------------------
# CENARIO 5c — O CAMINHO PARALELO: `method=serrf` nao pode gerar processos
# ----------------------------------------------------------------------------
#
# **O defeito que isto fecha, e ele era IMPEDITIVO.** Medido sobre o binario
# congelado, no menor dos quatro modos: um `POST /matrizes/corrigir` com `method=serrf` produziu
# **14 processos `refino`, os catorze COM JANELA**. O caminho do artefato corre
# `SerrfCorrector(n_jobs=-1)`, o `joblib` lanca workers `loky`, e cada worker de um congelado
# `--windowed` reexecuta o executavel.
#
# A CAUSA de nada o apanhar era um PADRAO DE ARGUMENTO: o cenario 5b exercita o fluxo SEM o campo
# `method`, e o padrao da rota e `none`. A metade paralela do produto nunca correu dentro do
# binario, com 30 passos verdes sobre a metade que nao usa `joblib`. Todo eixo novo com valor
# padrao repoe a mesma cegueira.
#
# ============================================================================
# O ORACULO TEM TRES NUMEROS
# ============================================================================
#
# Processos sao o que se QUER: com o interpretador embarcado, eles sao o paralelismo. O que nao
# pode aparecer e JANELA.
#
# Um oraculo pelo nome do processo e pelo punho de janela mediria errado nos DOIS eixos:
#
#   1. `Get-Process -Name "refino"` conta o nome ERRADO. Os workers chamam-se `pythonw`, entao a
#      contagem de `refino` fica em 1 mesmo com o paralelismo perfeito: o portao reprovaria o caso
#      de sucesso, e um portao assim e afrouxado no primeiro dia.
#   2. `MainWindowHandle` NAO ENXERGA janela de console. Ela pertence ao `conhost.exe`, nao ao
#      processo que a abriu. Um interpretador de console daria "1 janela" com doze pretas na tela.
#
# Sao TRES numeros, e os TRES sao incondicionais. Nao ha ramo "sem o `pythonw.exe`": o passo que
# COPIA o interpretador para `_internal` esta fora do bloco `if (-not $SoExercita)`, entao toda
# corrida o repoe antes de chegar aqui, e se a FONTE faltar a copia para o script muito antes deste
# cenario. Um `else` para esse caso seria inalcancavel, e uma planta que apagasse o `pythonw.exe`
# da pasta leria `novos=13` (medido).
#
# A rede "cai para SERIE" e provada em `tests/unit/test_paralelismo_do_congelado.py`, que exercita
# os tres gatilhos de queda em processo: a propriedade e do codigo e nao da pasta.
#
#   os tres numeros  ->  processos > 1, janelas == 1, consoles == 0
#
# ============================================================================
# OS EIXOS DOS PROCESSOS E DOS CONSOLES FORAM VISTOS A REPROVAR, e o gesto foi mutar a FONTE
# ============================================================================
#
# A planta e no `-PythonBase`: mutar `_internal\pythonw.exe` nao serve, porque a copia acima o
# repoe.
#
# | Planta | O que se poe na fonte | O que o portao disse |
# |---|---|---|
# | processos | `pythonw.exe` de 100 KB de zeros: existe, passa o portao da copia (que so mede tamanho) e o `CreateProcess` falha | **`PARALELIZA exit=1`**, `novos=0` — com `janelas=1` e `consoles=0` |
# | consoles | o `python.exe` de CONSOLE com o nome `pythonw.exe` | **`nao abre CONSOLE nenhum exit=1`**, `consoles=13` — e `janelas` continuou a dizer **1** |
#
# > **A segunda planta e a prova do argumento inteiro deste cenario:** com treze consolas pretas na
# > tela, o eixo das JANELAS reportou `1`. Sem o terceiro numero, o portao teria passado.
#
# O eixo das janelas tem a planta HISTORICA e nao uma sintetica: os 14 processos COM JANELA
# medidos, que sao o defeito que este cenario nasceu para fechar.
#
# A asserção continua a NAO ser sobre o desfecho da correcao: o `serrf` no `neg_c18` pode nao
# terminar dentro do teto, e isso nao e o que se mede aqui.
if ($identificador -ne "" -and $porta -ne 0) {
    # OS TRES ORACULOS CONTAM SO O QUE ESTA PASTA CRIOU.
    #
    # `pythonw.exe` e o interpretador sem consola de qualquer instalacao de Python do Windows, e
    # `refino.exe` pode estar a correr a partir da copia ENTREGUE, que ja reprovou um congelamento
    # por um PID que ele nao criou (a nota do cenario 6).
    #
    # Contando a MAQUINA INTEIRA, a direcao do erro seria falso VERDE no eixo dos processos (um
    # `pythonw` alheio faria `novos >= 1` com o paralelismo em baixo) e falso VERMELHO no das
    # janelas. O predicado e o mesmo do cenario 6.
    $eDesta = { $_.Path -and $_.Path.StartsWith($Dist, [StringComparison]::OrdinalIgnoreCase) }
    $antesSerrf = @(Get-Process -Name "pythonw" -EA SilentlyContinue | Where-Object $eDesta).Count
    $arqSerrf = Join-Path $Saidas "serrf-corrigir.json"
    $codSerrf = Join-Path $Saidas "serrf-corrigir.codigo"
    & curl.exe -s -o $arqSerrf -w "%{http_code}" `
        -F "measurements=@$medidas;type=text/csv" `
        -F "sample_list=@$lista;type=text/csv" `
        -F "mode=$modo" -F "method=serrf" `
        "http://127.0.0.1:$porta/matrizes/corrigir" > $codSerrf
    $codigoSerrf = if (Test-Path $codSerrf) { (Get-Content $codSerrf -Raw).Trim() } else { "" }

    # Observa por 60 s: os workers do `loky` nascem nos primeiros segundos do ajuste, e o
    # arranque de cada um paga o import de `numpy` e `sklearn` a partir de `.pyc` soltos.
    $maxWorkers = 0
    $maxJanelas = 0
    $maxConsoles = 0
    for ($k = 1; $k -le 30; $k++) {
        Start-Sleep -Seconds 2
        $workers = @(Get-Process -Name "pythonw" -EA SilentlyContinue | Where-Object $eDesta)
        $nossos = @(Get-Process -Name "refino", "pythonw" -EA SilentlyContinue | Where-Object $eDesta)
        $comJanela = @($nossos | Where-Object { $_.MainWindowHandle -ne 0 })
        # O `conhost` E CONTADO PELO PAI, e nao pelo nome. `Get-Process -Name conhost` conta a
        # maquina inteira: mediu 20 neste computador com o aplicativo ainda em serie, e um numero
        # desses ao lado de "janelas=1" descreve o Windows de quem mede e nao o
        # produto. O console que este passo procura e o que um worker de SUBSISTEMA CONSOLE
        # abriria, e esse tem por pai um processo NOSSO.
        $idsNossos = @($nossos | ForEach-Object { $_.Id })
        $consoles = @(Get-CimInstance Win32_Process -Filter "Name='conhost.exe'" -ErrorAction SilentlyContinue |
                Where-Object { $idsNossos -contains $_.ParentProcessId })
        if ($workers.Count -gt $maxWorkers) { $maxWorkers = $workers.Count }
        if ($comJanela.Count -gt $maxJanelas) { $maxJanelas = $comJanela.Count }
        if ($consoles.Count -gt $maxConsoles) { $maxConsoles = $consoles.Count }
    }
    $novos = $maxWorkers - $antesSerrf

    # `Relata` E NAO `ParaSe` NOS TRES EIXOS.
    #
    # `ParaSe` faz `exit 1` imediatamente, e com isso **salta o cenario 6** — que e o passo que
    # encerra a instancia e confere que nao ficou processo orfao. Uma reprovacao aqui deixava a
    # arvore VIVA: um `refino.exe` e ate treze `pythonw.exe` a consumir 2,3 GB, na maquina de quem
    # corre o congelamento (medido, com a planta do eixo dos processos).
    #
    # `Relata` incrementa `$script:Falhas`, e a ultima linha do roteiro ja faz
    # `exit ([int]($script:Falhas -gt 0))` — o veredito continua vermelho, e a limpeza acontece.
    Relata "o serrf no congelado nao abre JANELA nenhuma" ([int]($maxJanelas -gt 1)) `
        "http=$codigoSerrf janelas=$maxJanelas (a do aplicativo conta 1) consoles=$maxConsoles"

    Relata "o serrf no congelado nao abre CONSOLE nenhum" ([int]($maxConsoles -gt 0)) `
        "conhost com pai nosso: maximo=$maxConsoles (esperado 0)"

    # (c) o eixo dos PROCESSOS — incondicional, pelo motivo escrito na cabeca deste cenario
    #
    # O LIMIAR NAO E `>= 1`. Com `novos -lt 1` o passo ficava VERDE sobre um defeito real: o
    # worker NASCIA (a contagem de `pythonw` ia de 0 a 1) e morria em 4,6 s com
    # `TerminatedWorkerError`. Um worker so nao e paralelismo; e um worker. A planta de um arquivo
    # que nao arranca DE TODO (`novos = 0`) nao separa os dois limiares.
    #
    # **O piso vem da MAQUINA e nao de um numero a mao:** metade dos nucleos logicos, com minimo
    # de 2. A corrida medida deu 13 em 12 nucleos; metade da margem para uma maquina mais pequena
    # ou com carga, e `-ge 2` sozinho ja mata o caso do worker transitorio.
    $esperados = [math]::Max(2, [int]([Environment]::ProcessorCount / 2))
    Relata "o serrf no congelado PARALELIZA" ([int]($novos -lt $esperados)) `
        "workers pythonw DESTA pasta: antes=$antesSerrf maximo=$maxWorkers novos=$novos esperado>=$esperados de $([Environment]::ProcessorCount) nucleos"
}

# ----------------------------------------------------------------------------
# CENARIO 6 — o binario desliga limpo: a porta e liberada e nenhum processo fica
# ----------------------------------------------------------------------------
$portasAntes = ($portas | ForEach-Object { $_.LocalPort })
if (-not $proc.HasExited) {
    # `CloseMainWindow` primeiro: e o caminho que o usuario percorre. `Kill` seria medir outra coisa.
    $fechou = $proc.CloseMainWindow()
    if (-not $proc.WaitForExit(20000)) { $proc.Kill(); $proc.WaitForExit(10000) }
    # `Refresh()` antes de ler `ExitCode`: sem ele o campo vem vazio, e vazio nao e zero.
    $proc.Refresh()
    $codigoDoProcesso = if ($proc.HasExited) { $proc.ExitCode } else { "ainda vivo" }
    Relata "o processo terminou" ([int]!$proc.HasExited) "CloseMainWindow=$fechou exit=$codigoDoProcesso"
}
Start-Sleep -Seconds 3
$aindaEmUso = @()
foreach ($p in $portasAntes) {
    $c = Get-NetTCPConnection -State Listen -LocalPort $p -ErrorAction SilentlyContinue
    if ($c) { $aindaEmUso += $p }
}
$aindaEmUso = @($aindaEmUso)
Relata "as portas foram liberadas" ($aindaEmUso.Count) ($aindaEmUso -join ",")

# O ORACULO E O CAMINHO, e nao o NOME.
#
# Pelo nome, o passo reprovou um congelamento por um processo que ele NAO criou: um `refino.exe`
# arrancado de uma copia ENTREGUE, aberta noutra pasta para testar.
#
# Isto NAO e afrouxar: a propriedade e *"este congelamento nao deixou processo orfao"*, e filtrar
# pelo caminho de `$Dist` afirma-a com precisao. O nome afirmava algo mais largo que ninguem
# queria, e a linha de detalhe DIZ quantos ignorou: silenciar o alheio sem o contar seria trocar um
# falso positivo por um ponto cego.
#
# **Escopo negativo declarado:** `Path` vem vazio para processo de outro usuario ou
# elevado, e um `refino.exe` nessas condicoes deixaria de ser visto. O caso nao ocorre
# neste arnes, que corre como o dono da pasta.
#
# OS DOIS NOMES: os workers sao ate treze `pythonw.exe` por correcao, que vivem em
# `$Dist\_internal` e caem sob o mesmo predicado de caminho. Contando so `refino`, um congelamento
# que deixasse os workers vivos passaria.
$todosRefino = @(Get-Process -Name "refino", "pythonw" -ErrorAction SilentlyContinue)
$deste = { $_.Path -and $_.Path.StartsWith($Dist, [StringComparison]::OrdinalIgnoreCase) }
$sobrando = @($todosRefino | Where-Object $deste)
$alheios = @($todosRefino | Where-Object { -not (& $deste) })
$detalhe = ($sobrando | ForEach-Object { $_.Id }) -join ","
if ($alheios.Count -gt 0) {
    $ids = ($alheios | ForEach-Object { $_.Id }) -join ","
    # `${Dist}` e nao `$Dist`: um `:` depois do nome faz o PowerShell ler `$Dist:` como
    # QUALIFICADOR DE DRIVE, e o script morre em ParserError antes da primeira linha. Foi
    # o desfecho medido ao correr este conserto pela primeira vez.
    $detalhe += " (ignorados $($alheios.Count) refino de FORA de ${Dist}: $ids)"
}
Relata "nenhum processo NOSSO sobrando (refino e pythonw)" ($sobrando.Count) $detalhe

Resumo
exit ([int]($script:Falhas -gt 0))
