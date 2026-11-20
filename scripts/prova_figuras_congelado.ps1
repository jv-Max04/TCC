# prova_figuras_congelado.ps1 - a etapa 3c do roteiro de entrega.
#
# Prova, DENTRO da pasta entregue, que cada figura REGISTADA desenha, que o backend que resolve e o
# do Qt e nao o de recuo, e que duas copias em caminhos diferentes desenham as duas sem que uma
# reconstrua o cache de fontes da outra.
#
# ============================================================================
# A LISTA DE FIGURAS NAO ESTA AQUI, e isso e o desenho
# ============================================================================
#
# Quem enumera as figuras e `refino.plots.catalogo.CONSTRUTORES`, lido DENTRO do congelado pela
# carga irma `prova_figuras_congelado.py`. Uma lista escrita neste arquivo seria a irma que
# envelhece: a decima quarta figura entraria no registo e este roteiro continuaria verde sobre
# treze. Com o registo como fonte, a figura nova aparece sozinha — e, se ninguem lhe der
# argumentos, a sonda REPROVA em vez de a saltar. Com uma decima quarta entrada plantada no
# `catalogo` do congelado a sonda saiu 1, e com ela removida saiu 0.
#
# ============================================================================
# AS DUAS COPIAS SAO O UNICO SITIO ONDE O CACHE POR CAMINHO E EXERCITADO
# ============================================================================
#
# O gancho `pyi_rth_mplconfig.py` do PyInstaller aponta o `MPLCONFIGDIR` para um diretorio
# temporario NOVO a cada abertura, e o cache de fontes e reconstruido sempre. O script de entrada
# desfaz isso apontando para um caminho persistente DERIVADO DO CAMINHO DE INSTALACAO. O
# «derivado do caminho» e a metade que so este cenario mede: o `fontlist-v*.json` guarda caminhos
# absolutos para dentro do congelado, entao uma copia noutro lugar tem de receber cache PROPRIO em
# vez de herdar os caminhos da pasta de origem.
#
# ============================================================================
# O ORACULO DO CACHE E O ARQUIVO, NUNCA O RELOGIO
# ============================================================================
#
# «A segunda abertura foi rapida» e satisfeito por um cache que se reconstroi depressa porque o
# antivirus ja viu as fontes. O que se afirma e a existencia e a MARCA DE TEMPO do
# `fontlist-v*.json`, lidas antes e depois de cada abertura.
#
# E o diretorio nao e adivinhado: a carga deriva-o com a formula de `arranque.py`, e este roteiro
# confere essa derivacao contra o comportamento do produto — lista os subdiretorios do cache antes
# e depois de lancar o `refino.exe` da copia, e exige que o UNICO novo seja exactamente o que a
# carga aponta. Uma formula errada apontaria para um diretorio que ninguem criou.
#
# Sem acentos no relato, como todo o ferramental deste repositorio: a code page desta maquina e
# cp932, e prosa acentuada na consola ja matou uma sonda no passo que decide.
#
# ## Uso
#
#     powershell -ExecutionPolicy Bypass -File scripts\prova_figuras_congelado.ps1
#     powershell -ExecutionPolicy Bypass -File scripts\prova_figuras_congelado.ps1 -Manter
#
# Sai 0 quando as quatro aberturas passaram, 1 quando alguma reprovou, e 2 quando nao foi possivel
# medir — sem binario, sem interpretador embarcado, ou sem espaco para a segunda copia.

param(
    [string]$Dist = "dist\refino",
    [string]$SegundaCopiaEm = "",
    [string]$Saida = "dist\figuras-congelado-verdicto.json",
    [int]$EsperaSeg = 240,
    [switch]$Manter
)

$ErrorActionPreference = "Stop"
$Raiz = Split-Path -Parent $PSScriptRoot
Set-Location $Raiz

. (Join-Path $PSScriptRoot "porta_que_serve.ps1")

$Carga = Join-Path $PSScriptRoot "prova_figuras_congelado.py"
$PastaA = (Join-Path $Raiz $Dist)
$BaseDoCache = Join-Path $env:LOCALAPPDATA "refino\cache-de-figuras"

if (-not $SegundaCopiaEm) {
    # **Carimbo de tempo no nome, e nao um nome fixo.** O diretorio de cache e o hash do caminho de
    # instalacao: um nome fixo faria a segunda corrida encontrar o cache que a primeira deixou, e a
    # afirmacao «este diretorio nasceu agora» passaria a ser falsa sem ninguem notar.
    $SegundaCopiaEm = Join-Path $env:TEMP ("refino-figuras-" + (Get-Date).ToString("yyyyMMdd-HHmmss"))
}
$PastaB = $SegundaCopiaEm

# NORMALIZAR ANTES DE COMPARAR, e isto e conserto de perda de dado.
#
# `$PastaA` e absoluto — `Join-Path $Raiz $Dist` — e `$PastaB` vinha do parametro CRU. A guarda
# logo abaixo era um `-eq` entre os dois, entao `-SegundaCopiaEm dist` passava: a string `dist`
# nao e igual a `C:\...\dist`. Tres linhas depois corre
# `Remove-Item -LiteralPath $PastaB -Recurse -Force`, e ela resolvia para o congelado. O valor
# `dist` apagava a arvore inteira; era o unico dos tres parametros de caminho nao juntado a `$Raiz`.
#
# **`Join-Path $Raiz` e nao a localizacao corrente, e a distincao morde:** `Set-Location` muda a
# localizacao do PowerShell e **nao** o diretorio corrente do PROCESSO, entao um
# `[System.IO.Path]::GetFullPath` de um caminho relativo resolveria contra o CWD do processo, que
# aqui nao e `$Raiz`. Enraizar primeiro torna a chamada seguinte independente disso — ela fica so
# a colapsar `..` e a uniformizar separadores.
if (-not [System.IO.Path]::IsPathRooted($PastaB)) { $PastaB = Join-Path $Raiz $PastaB }
$PastaB = [System.IO.Path]::GetFullPath($PastaB).TrimEnd('\')
$PastaA = [System.IO.Path]::GetFullPath($PastaA).TrimEnd('\')

$script:falhas = @()
function Afirma {
    param([string]$Passo, [bool]$Ok, [string]$Detalhe)
    $marca = if ($Ok) { "OK  " } else { "FALHA" }
    if (-not $Ok) { $script:falhas += $Passo }
    Write-Host ("[{0}] {1}  {2}" -f $marca, $Passo, $Detalhe)
}

function Para {
    param([string]$Motivo, [string]$Conserto)
    Write-Host ""
    Write-Host ("PARADO na etapa 3c: {0}" -f $Motivo)
    if ($Conserto) { Write-Host ("  {0}" -f $Conserto) }
    exit 2
}

function SubdiretoriosDoCache {
    if (-not (Test-Path $BaseDoCache)) { return @() }
    return @(Get-ChildItem -LiteralPath $BaseDoCache -Directory | ForEach-Object { $_.FullName })
}

function Encerra {
    param([string]$Sob)
    Get-Process refino -ErrorAction SilentlyContinue |
        Where-Object { $_.Path -and $_.Path.StartsWith($Sob, [StringComparison]::OrdinalIgnoreCase) } |
        Stop-Process -Force -ErrorAction SilentlyContinue
}

function Abre {
    # Lanca o `refino.exe` de uma pasta e espera que ELE responda `/health`. Nao mede figura
    # nenhuma: serve para provar que a pasta abre de onde esta, e para que o proprio programa crie
    # o diretorio de cache que a carga depois exige.
    param([string]$Pasta, [string]$Rotulo)
    $exe = Join-Path $Pasta "refino.exe"
    $relogio = [Diagnostics.Stopwatch]::StartNew()
    $proc = Start-Process -FilePath $exe -PassThru
    $url = PortaQueServe -Processo $proc -TetoSeg $EsperaSeg
    $segundos = [math]::Round($relogio.Elapsed.TotalSeconds, 1)
    Afirma "$Rotulo abre e responde /health" ([bool]$url) "url=$url em $segundos s (PID $($proc.Id))"
    $proc | Stop-Process -Force -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 2
    Encerra -Sob $Pasta
    return $segundos
}

function Abertura {
    # Uma abertura das FIGURAS: corre a carga com o interpretador embarcado DA PASTA medida.
    #
    # **`Start-Process -Wait` e nao o operador de chamada.** `pythonw.exe` e do subsistema GUI, e o
    # `&` do PowerShell devolve o controlo sem esperar por processo nenhum desse subsistema — o
    # codigo de saida viria vazio e a sonda leria «passou».
    param([string]$Pasta, [string]$Rotulo)
    $interno = Join-Path $Pasta "_internal"
    $interpretador = Join-Path $interno "pythonw.exe"
    $json = Join-Path $Raiz ("dist\figuras-{0}.json" -f ($Rotulo -replace "[^A-Za-z0-9]", "-"))
    $saidaTxt = "$json.out.txt"
    $erroTxt = "$json.err.txt"

    # O ambiente vai LIMPO, pelo mesmo motivo de `prova_interpretador_embarcado.py`: um
    # `PYTHONPATH` herdado faria a carga importar do clone de quem corre a sonda, e o desfecho
    # seria verde sobre a arvore errada.
    $guardaPath = $env:PYTHONPATH
    $guardaHome = $env:PYTHONHOME
    $env:PYTHONPATH = $null
    $env:PYTHONHOME = $null
    try {
        $proc = Start-Process -FilePath $interpretador -PassThru -Wait -WorkingDirectory $interno `
            -RedirectStandardOutput $saidaTxt -RedirectStandardError $erroTxt `
            -ArgumentList @($Carga, "--pasta", $Pasta, "--saida", $json, "--rotulo", $Rotulo)
        $codigo = $proc.ExitCode
    }
    finally {
        $env:PYTHONPATH = $guardaPath
        $env:PYTHONHOME = $guardaHome
    }

    if (-not (Test-Path $json)) {
        Para -Motivo "a abertura $Rotulo nao deixou relato em $json (codigo $codigo)" `
            -Conserto "stderr em $erroTxt"
    }
    $lido = Get-Content -LiteralPath $json -Raw | ConvertFrom-Json
    $motivo = if ($lido.nao_deu_para_medir) { $lido.nao_deu_para_medir } else { "" }
    Afirma "abertura $Rotulo sem defeito" ($codigo -eq 0) `
        ("codigo=$codigo falhas=" + (@($lido.falhas) -join " | ") + $motivo)
    return $lido
}

function MarcasDeUmLado {
    # As marcas de tempo do `fontlist-v*.json` numa leitura — `cache_antes` ou `cache_depois`.
    #
    # Uma leitura por chamada, e nao as duas juntas: as tres leituras que a verificacao usa nao sao
    # as duas de uma abertura, e uma funcao que devolvesse sempre o par obrigaria a descartar
    # metade no sitio errado.
    param($Lado)
    $marcas = @()
    foreach ($arq in @($Lado.arquivos)) { $marcas += $arq.mtime_ns }
    return $marcas
}

Write-Host ""
Write-Host "=== ETAPA 3c - as figuras registadas desenham dentro do congelado"
Write-Host ("  pasta A : {0}" -f $PastaA)
Write-Host ("  pasta B : {0}" -f $PastaB)

# --- 0. O QUE PRECISA DE EXISTIR ANTES DE MEDIR -----------------------------
if (-not (Test-Path (Join-Path $PastaA "refino.exe"))) {
    Para -Motivo "nao ha refino.exe em $PastaA" `
        -Conserto "Congele primeiro: powershell -File scripts\congela_windows.ps1"
}
if (-not (Test-Path (Join-Path $PastaA "_internal\pythonw.exe"))) {
    Para -Motivo "nao ha _internal\pythonw.exe em $PastaA" `
        -Conserto "O interpretador embarcado entra no congelamento. Refaca a etapa 1."
}
if (-not (Test-Path $Carga)) {
    Para -Motivo "a carga desta sonda nao esta em $Carga"
}
if ($PastaB -eq $PastaA) {
    Para -Motivo "a segunda copia teria o MESMO caminho da primeira, e R12b nao seria exercitado"
}
# **Igualdade nao basta, e as duas direcoes tem consequencias diferentes.** Conter e ser contido
# levam ambos o `Remove-Item -Recurse` de baixo a apagar o congelado — `-SegundaCopiaEm dist\copia`
# apaga so a copia mas mede o cache do caminho errado; `-SegundaCopiaEm .` apaga a raiz inteira.
# A guarda de igualdade sozinha nao ve nenhum dos dois.
if ($PastaB.StartsWith($PastaA + '\', [StringComparison]::OrdinalIgnoreCase)) {
    Para -Motivo "a segunda copia ficaria DENTRO de $PastaA, e o Remove-Item apagaria parte do congelado"
}
if ($PastaA.StartsWith($PastaB + '\', [StringComparison]::OrdinalIgnoreCase)) {
    Para -Motivo "a segunda copia CONTEM $PastaA, e o Remove-Item apagaria o congelado inteiro"
}

Encerra -Sob $PastaA

# --- 1. A PASTA A ABRE, e o programa cria o proprio diretorio de cache ------
$tempoA = Abre -Pasta $PastaA -Rotulo "A"

# --- 2. PRIMEIRA ABERTURA DAS FIGURAS EM A ----------------------------------
#
# Antes da copia de propositio: se as figuras nao desenham, copiar 375 MB nao acrescenta
# informacao nenhuma ao relato.
$a1 = Abertura -Pasta $PastaA -Rotulo "A1"

# --- 3. A SEGUNDA COPIA, NOUTRO CAMINHO -------------------------------------
$antesDaCopia = SubdiretoriosDoCache
if (Test-Path $PastaB) { Remove-Item -LiteralPath $PastaB -Recurse -Force }
$eco = & robocopy.exe $PastaA $PastaB /E /NFL /NDL /NJH /NJS /NP /R:1 /W:1
$codigoRobocopy = $LASTEXITCODE
if ($codigoRobocopy -ge 8) {
    Para -Motivo "robocopy falhou com $codigoRobocopy ao copiar para $PastaB" -Conserto "$eco"
}
if (-not (Test-Path (Join-Path $PastaB "refino.exe"))) {
    Para -Motivo "a copia em $PastaB ficou sem refino.exe"
}
# **A afirmacao e sobre o INTERPRETADOR e nao sobre a copia ter acontecido.** A primeira versao
# desta linha passava `$true` literal, o que a tornava decorativa: as duas guardas acima ja param a
# sonda se a copia falhar. O que ainda podia estar em falta e o que a carga precisa — `robocopy`
# sabe saltar arquivo em uso, e o `pythonw.exe` teria ficado para tras em silencio.
$interpretadorDaCopia = Join-Path $PastaB "_internal\pythonw.exe"
Afirma "a segunda copia levou o interpretador embarcado" (Test-Path $interpretadorDaCopia) `
    "$PastaB (robocopy $codigoRobocopy)"

# --- 4. A COPIA ABRE DE ONDE ESTA, e RECEBE CACHE PROPRIO -------------------
$tempoB = Abre -Pasta $PastaB -Rotulo "B"
$depoisDaCopia = SubdiretoriosDoCache
$novos = @($depoisDaCopia | Where-Object { $antesDaCopia -notcontains $_ })
Afirma "abrir a copia criou UM diretorio de cache novo" ($novos.Count -eq 1) `
    ("novos=" + $novos.Count + " -> " + (@($novos) -join ", "))

# --- 5. AS OUTRAS TRES ABERTURAS, ALTERNADAS --------------------------------
$b1 = Abertura -Pasta $PastaB -Rotulo "B1"
$a2 = Abertura -Pasta $PastaA -Rotulo "A2"
$b2 = Abertura -Pasta $PastaB -Rotulo "B2"

Write-Host ""
Write-Host "=== VEREDITO"

# (1) A NAO-VACUIDADE VEM PRIMEIRO. «Nenhuma figura falhou» e satisfeito por zero figuras.
$registados = @($a1.registados, $b1.registados, $a2.registados, $b2.registados)
$mesmoRegisto = (@($registados | Sort-Object -Unique).Count -eq 1)
Afirma "as quatro aberturas viram o MESMO registo, e ele nao esta vazio" `
    ($mesmoRegisto -and ($a1.registados -ge 1)) "registados: $($registados -join ', ')"

$desenhadas = @($a1.digests.PSObject.Properties).Count
Afirma "cada figura registada foi de facto desenhada" ($desenhadas -eq $a1.registados) `
    "desenhadas=$desenhadas de $($a1.registados) registadas"

# (2) AS IMAGENS SAO DISTINTAS ENTRE SI: sem esta ancora, um arnes que escrevesse a mesma
#     imagem para todas as figuras passaria com EXIT=0.
Afirma "as figuras renderizadas sao distintas umas das outras" `
    ($a1.digests_distintos -eq $desenhadas) `
    "distintas=$($a1.digests_distintos) de $desenhadas"

# (3) O BACKEND. Com Agg a figura desenha e nunca aparece: sucesso com tela vazia.
$backends = @($a1.backend, $b1.backend, $a2.backend, $b2.backend)
$todosQt = @($backends | Where-Object { $_ -notmatch "(?i)qt" }).Count -eq 0
Afirma "o backend resolvido e o do Qt nas quatro aberturas" $todosQt "$($backends -join ', ')"

# (4) CADA COPIA TEM O SEU CACHE, e o da copia foi criado pelo proprio programa.
Afirma "as duas copias apontam para diretorios de cache DIFERENTES" `
    ($a1.cache_apontado -ne $b1.cache_apontado) `
    "A=$($a1.cache_apontado)  B=$($b1.cache_apontado)"
Afirma "o diretorio que a copia criou e o que a carga deriva" `
    (($novos.Count -eq 1) -and ($novos[0] -eq $b1.cache_apontado)) `
    "criado=$(@($novos) -join ',')  derivado=$($b1.cache_apontado)"

# (5) AS DUAS DESENHAM O MESMO. Sem esta, "as duas desenharam" e satisfeito por duas
#     figuras diferentes com o mesmo nome.
$iguais = 0
$diferentes = @()
foreach ($p in @($a1.digests.PSObject.Properties)) {
    $doB = $b1.digests.$($p.Name)
    if ($doB -eq $p.Value) { $iguais++ } else { $diferentes += $p.Name }
}
Afirma "as duas copias desenham a MESMA imagem em cada figura" `
    (($iguais -eq $desenhadas) -and ($desenhadas -gt 0)) `
    "iguais=$iguais de $desenhadas; divergentes: $(@($diferentes) -join ', ')"

# (6) O CACHE DE FONTES NAO E RECONSTRUIDO. O oraculo e o arquivo, nao o relogio.
#
# AS TRES LEITURAS SAO "depois da 1a", "antes da 2a" e "depois da 2a"; a de ANTES da primeira
# fica de fora. Num caminho novo em folha, lancar o `refino.exe` cria o DIRETORIO mas o
# `fontlist-v*.json` so aparece quando alguem de facto desenha. Exigir as quatro reprovaria uma
# copia perfeita, e afirmaria que o programa escreve o cache ao arrancar, que e uma propriedade
# que ele nao tem e que ninguem lhe pediu.
#
# O que fica afirmado: a PRIMEIRA abertura produz o cache, e a SEGUNDA nao lhe
# toca. A leitura de antes da primeira vira numero REPORTADO.
$primeiraJaTinha = @()
foreach ($par in @(@{ nome = "A"; um = $a1; dois = $a2 }, @{ nome = "B"; um = $b1; dois = $b2 })) {
    $nome = $par.nome
    # Acumulado com `+=` e nao com `(f x) + (f y)`: a soma direta de duas chamadas rebenta com
    # `[System.Object[]] does not contain a method named 'op_Addition'` na SEGUNDA volta do laco,
    # depois de a primeira ter passado (medido).
    $marcas = @()
    $marcas += @(MarcasDeUmLado $par.um.cache_depois)
    $marcas += @(MarcasDeUmLado $par.dois.cache_antes)
    $marcas += @(MarcasDeUmLado $par.dois.cache_depois)
    $distintas = @($marcas | Sort-Object -Unique)
    $antesDaPrimeira = @(MarcasDeUmLado $par.um.cache_antes).Count
    $primeiraJaTinha += "$nome=$antesDaPrimeira"

    Afirma "a primeira abertura da copia $nome deixou fontlist" ($marcas.Count -eq 3) `
        "leituras com arquivo: $($marcas.Count) de 3 (depois da 1a, antes da 2a, depois da 2a)"
    Afirma "o fontlist da copia $nome NAO foi reconstruido" `
        (($marcas.Count -eq 3) -and ($distintas.Count -eq 1)) `
        "marcas distintas: $($distintas.Count) -> $($distintas -join ', ')"
}
Write-Host ""
Write-Host "--- reportado (nao e assercao: depende de a pasta ja ter sido aberta antes)"
Write-Host ("  fontlist ja existia ANTES da primeira abertura: {0}" -f ($primeiraJaTinha -join "  "))

# --- 6. O VEREDITO EM DISCO -------------------------------------------------
$verdicto = @{
    pasta_a = $PastaA
    pasta_b = $PastaB
    abertura_do_exe_s = @{ a = $tempoA; b = $tempoB }
    registados = $a1.registados
    desenhadas = $desenhadas
    digests_distintos = $a1.digests_distintos
    backends = $backends
    cache = @{
        a = $a1.cache_apontado
        b = $b1.cache_apontado
        criado_ao_abrir_a_copia = @($novos)
    }
    aberturas = @{
        a1 = $a1.ok; b1 = $b1.ok; a2 = $a2.ok; b2 = $b2.ok
    }
    falhas = @($script:falhas)
    quando = (Get-Date).ToString("s")
}
$destino = Join-Path $Raiz $Saida
[System.IO.File]::WriteAllText($destino, ($verdicto | ConvertTo-Json -Depth 6),
    (New-Object System.Text.UTF8Encoding($false)))

# --- 7. A LIMPEZA, e ela alcanca o cache que a copia deixou -----------------
#
# `cache-de-figuras` acumula um subdiretorio por caminho de instalacao e nada os limpa sozinho — a
# propria docstring de `aponta_o_cache_de_figuras` o diz. Uma sonda que corre por entrega e deixa
# um orfao por corrida e a fonte do lixo que ela avisa existir.
if ($Manter) {
    Write-Host ""
    Write-Host ("copia mantida em {0} (e o cache dela em {1})" -f $PastaB, $b1.cache_apontado)
}
else {
    Encerra -Sob $PastaB
    Remove-Item -LiteralPath $PastaB -Recurse -Force -ErrorAction SilentlyContinue
    # A IRMA da guarda de `$PastaB`. Este caminho nao vem de um parametro: vem do JSON que o
    # aplicativo CONGELADO escreveu, e apagar recursivamente o que
    # o programa sob teste nos disse e uma fronteira de confianca com um `-Recurse` do outro lado.
    # O `-and (Test-Path ...)` que ja estava aqui cobre nulo e vazio, e nao cobre um caminho
    # valido e ERRADO — um `MPLCONFIGDIR` que apontasse para a pasta de instalacao apagaria o
    # congelado, que e exactamente o desfecho da guarda de cima.
    #
    # `$BaseDoCache` ja e calculado no topo, entao confinar custa uma linha.
    $cache = $b1.cache_apontado
    if ($cache) { $cache = [System.IO.Path]::GetFullPath($cache).TrimEnd('\') }
    $base = [System.IO.Path]::GetFullPath($BaseDoCache).TrimEnd('\')
    $sobABase = $cache -and $cache.StartsWith($base + '\', [StringComparison]::OrdinalIgnoreCase)
    if ($sobABase -and (Test-Path $cache)) {
        Remove-Item -LiteralPath $cache -Recurse -Force -ErrorAction SilentlyContinue
    }
    elseif ($cache) {
        Afirma "o cache relatado fica sob a base de cache" $false `
            "o congelado relatou $cache, que nao esta sob $base - NAO foi apagado"
    }
}

Write-Host ""
Write-Host ("veredito gravado em {0}" -f $destino)
if ($script:falhas.Count -gt 0) {
    Write-Host ("FALHAS: {0}" -f ($script:falhas -join " | "))
    exit 1
}
Write-Host "TUDO VERDE: as figuras registadas desenham nas duas copias, e o cache nao foi refeito."
exit 0
