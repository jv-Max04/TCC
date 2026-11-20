# Monta a arvore R portatil com o MetaboAnalystR, poda-a e emite o manifesto.
#
# ============================================================================
# O QUE ELE FAZ, NA ORDEM
# ============================================================================
#
#  1. baixa o instalador do R 4.4.3, o do Rtools44 e o tarball do MetaboAnalystR no commit fixado, e
#     confere o sha256 de cada um contra o fixado abaixo;
#  2. mapeia a pasta de build numa unidade SUBST curta: o instalador do Rtools falha em caminho longo
#     com "MoveFile failed; code 3" e desfaz a instalacao inteira (medido);
#  3. instala o R e o Rtools em modo silencioso, por usuario, dentro da unidade;
#  4. instala o grafo de dependencias e compila o MetaboAnalystR (scripts/r_portatil_instala.R);
#  5. corre os fluxos do modo local sobre a arvore COMPLETA e deriva a lista de pacotes do que eles
#     carregam (scripts/r_portatil.py deriva);
#  6. poda para uma arvore nova, que e a que viaja (scripts/r_portatil.py poda);
#  7. emite o manifesto da arvore podada (scripts/r_portatil.py manifesto);
#  8. desinstala o Rtools e o R pelos desinstaladores deles e confere que as chaves de registro que
#     eles criam sumiram: a arvore podada nao depende de registro (medido).
#
# A arvore fica em <Build>\arvore e as saidas em <Build>\saida. Com -Versiona, a lista de pacotes,
# os despejos por fluxo e o manifesto sao copiados para scripts\ - e o que se faz na montagem de
# referencia. A segunda montagem, que prova a reproducao, corre SEM -Versiona e compara:
#
#   .venv\Scripts\python.exe scripts\r_portatil.py compara --arvore <Build2>\arvore `
#       --manifesto scripts\r_portatil_arvore.sha256 --fora-dos-carimbos
#
# ============================================================================
# USO
# ============================================================================
#
#   powershell -ExecutionPolicy Bypass -File scripts\monta_r_portatil.ps1 -Dados <pasta> [-Versiona]
#
# -Dados e a pasta com as bibliotecas de dados que os fluxos leem (compound_db.qs, syn_nms.qs,
# master_compound_db.qs, msets_smpdb_pathway.qs, smpdb_hsa.qs). Elas so servem para os fluxos que
# derivam a lista; a arvore nao as contem.
#
# Corre da RAIZ do clone. Recusa-se a comecar se o R 4.4.3 ou o Rtools44 ja estiverem instalados
# para este usuario: o instalador por usuario sobrescreveria a chave de desinstalacao da instalacao
# que ja existe, e o passo 8 apagaria a dela.

[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$Dados,
    [string]$Build = "$env:USERPROFILE\_refino_env\r-portatil",
    [string]$Downloads = "$env:USERPROFILE\_refino_env\r-portatil-downloads",
    [string]$Roteiro = "",
    # A letra da unidade SUBST e FIXA, e nao a primeira livre: o R CMD INSTALL grava o caminho da
    # biblioteca no namespace do pacote compilado (.__NAMESPACE__.$path, dentro do .rdb), entao a
    # letra faz parte do que a segunda montagem tem de reproduzir.
    [string]$Letra = "R",
    [switch]$Versiona,
    [switch]$ManterInstalado
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
# O Python reconfigura a propria saida para UTF-8; sem isto o log a decodificaria pela pagina OEM.
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$Raiz = Split-Path -Parent $PSScriptRoot
$Py = Join-Path $Raiz ".venv\Scripts\python.exe"
$Marca = ".refino-r-portatil"

# Os tres artefatos, com o sha256 fixado de cada um.
$Artefatos = @(
    @{ Nome = "R-4.4.3-win.exe"
       Url = "https://cran.r-project.org/bin/windows/base/old/4.4.3/R-4.4.3-win.exe"
       Sha = "7eddfbab527aa3dc784ef165d9393ea85d3b93810e37fcf24b9c20087e2bbd29" },
    @{ Nome = "rtools44-6459-6401.exe"
       Url = "https://cran.r-project.org/bin/windows/Rtools/rtools44/files/rtools44-6459-6401.exe"
       Sha = "de9602992848faaeddca41f118dd6dc3de9bbfb00b8d87ebe5522af1dad1904b" },
    @{ Nome = "MetaboAnalystR-6bf5462.tar.gz"
       Url = "https://codeload.github.com/xia-lab/MetaboAnalystR/tar.gz/6bf5462064cd1fbc704bf4fb4abdf430723198ac"
       Sha = "5ff9a18203ef7acbc45882106144f9474f8aea3b2b64433f2fe63d5ba8e70d50" }
)

# As chaves que os dois instaladores criam no registro do usuario (medido).
$ChavesUninstall = @(
    "HKCU:\Software\Microsoft\Windows\CurrentVersion\Uninstall\R for Windows 4.4.3_is1",
    "HKCU:\Software\Microsoft\Windows\CurrentVersion\Uninstall\Rtools44_is1"
)

function Passo([string]$texto) {
    Write-Host ("[" + (Get-Date -Format "HH:mm:ss") + "] " + $texto)
}

function Falha([string]$texto) {
    Write-Host ("FALHA: " + $texto)
    exit 1
}

function Corre([string]$exe, [string[]]$argumentos, [string]$log) {
    # Saida para arquivo e codigo de saida lido aqui: nada atravessa tubo.
    # No PowerShell 5.1, redirecionar o stderr de um executavel nativo com "Stop" em vigor transforma
    # a primeira linha de stderr em erro terminante - e o R escreve progresso no stderr. Dentro desta
    # funcao vale "Continue", e quem decide e o codigo de saida.
    $ErrorActionPreference = "Continue"
    & $exe @argumentos *> $log
    $codigo = $LASTEXITCODE
    if ($codigo -ne 0) {
        Get-Content $log -Tail 40 | Write-Host
        Falha ("'" + $exe + "' saiu com " + $codigo + "; log em " + $log)
    }
}

# Os instaladores que ja rodaram nesta montagem, na ordem em que se desinstalam. A desinstalacao corre
# no caminho feliz (passo 8, com a conferencia do registro) e TAMBEM no finally quando a montagem
# falha no meio: a primeira montagem parou na extracao com o R e o Rtools instalados, e as chaves de
# registro ficaram para tras ate a limpeza manual.
$script:Instalados = @()
$script:Desinstalou = $false

function Desinstala([bool]$estrito) {
    foreach ($par in $script:Instalados) {
        $unins = $par[0] + "\unins000.exe"
        if (-not (Test-Path $unins)) { continue }
        Passo ("desinstalando " + $par[1])
        Start-Process -FilePath $unins -ArgumentList @("/VERYSILENT", "/SUPPRESSMSGBOXES", "/NORESTART") -Wait | Out-Null
        # O desinstalador do Inno se copia para o temporario e devolve antes de terminar.
        if (-not (Espera-Sumir $unins 600)) { Write-Host ("o desinstalador do " + $par[1] + " nao terminou em 10 minutos") }
    }
    $script:Desinstalou = $true
    $sobra = @($ChavesUninstall | Where-Object { Test-Path $_ })
    $amb = Get-ItemProperty -Path "HKCU:\Environment" -Name "RTOOLS44_HOME" -ErrorAction SilentlyContinue
    if ($null -ne $amb) { $sobra += "HKCU\Environment\RTOOLS44_HOME" }
    if ($sobra.Count -gt 0) {
        if ($estrito) { Falha ("ficou no registro depois de desinstalar: " + ($sobra -join "; ")) }
        Write-Host ("ATENCAO: ficou no registro depois de desinstalar: " + ($sobra -join "; "))
    } else {
        Passo "registro limpo: as duas chaves de desinstalacao e RTOOLS44_HOME sumiram"
    }
}

function Espera-Sumir([string]$caminho, [int]$segundos) {
    $limite = (Get-Date).AddSeconds($segundos)
    while ((Test-Path $caminho) -and ((Get-Date) -lt $limite)) { Start-Sleep -Seconds 2 }
    return -not (Test-Path $caminho)
}

if (-not (Test-Path $Py)) { Falha ("falta a venv: " + $Py) }
if (-not (Test-Path $Dados)) { Falha ("a pasta de dados nao existe: " + $Dados) }
$Dados = (Resolve-Path $Dados).Path
# O roteiro que viaja e o unico de que se deriva a lista de pacotes: um recuo para outro roteiro
# daria uma lista errada sem acusar nada.
if ($Roteiro -eq "") { $Roteiro = Join-Path $Raiz "r_service\vias_entrypoint.R" }
if (-not (Test-Path $Roteiro)) { Falha ("o roteiro dos fluxos nao existe: " + $Roteiro) }
Passo ("roteiro dos fluxos: " + $Roteiro)

foreach ($chave in $ChavesUninstall) {
    if (Test-Path $chave) { Falha ("ja existe uma instalacao registrada em " + $chave + "; a montagem nao a toca") }
}

# ---------------------------------------------------------------------------- 1. downloads
New-Item -ItemType Directory -Force $Downloads | Out-Null
foreach ($a in $Artefatos) {
    $destino = Join-Path $Downloads $a.Nome
    $ok = (Test-Path $destino) -and ((Get-FileHash -Algorithm SHA256 $destino).Hash.ToLower() -eq $a.Sha)
    if (-not $ok) {
        Passo ("baixando " + $a.Nome)
        Invoke-WebRequest -UseBasicParsing -Uri $a.Url -OutFile $destino
        $h = (Get-FileHash -Algorithm SHA256 $destino).Hash.ToLower()
        if ($h -ne $a.Sha) { Falha ($a.Nome + " com sha256 " + $h + ", esperado " + $a.Sha) }
    }
    Passo ($a.Nome + " confere o sha256")
}

# ---------------------------------------------------------------------------- 2. pasta e SUBST
if (Test-Path $Build) {
    if (-not (Test-Path (Join-Path $Build $Marca))) { Falha ("a pasta de build existe e nao tem a marca " + $Marca + ": " + $Build) }
    Passo ("apagando a montagem anterior em " + $Build)
    Remove-Item -Recurse -Force $Build
}
New-Item -ItemType Directory -Force $Build | Out-Null
Set-Content -Path (Join-Path $Build $Marca) -Value "pasta de build do R portatil; apagada a cada montagem"

if (Test-Path ($Letra + ":\")) { Falha ("a unidade " + $Letra + ": esta em uso; escolha outra com -Letra, sabendo que a letra entra no manifesto") }
$U = $Letra + ":"
& subst.exe $U $Build
if ($LASTEXITCODE -ne 0) { Falha ("subst " + $U + " falhou") }
Passo ("unidade " + $U + " -> " + $Build)

try {
    $RDir = $U + "\R"
    $RtDir = $U + "\rt44"
    $Logs = $U + "\logs"
    $Saida = $U + "\saida"
    New-Item -ItemType Directory -Force $Logs, $Saida | Out-Null

    # ------------------------------------------------------------------------ 3. instaladores
    $comuns = @("/VERYSILENT", "/SUPPRESSMSGBOXES", "/NORESTART", "/CURRENTUSER", "/NOICONS")
    Passo "instalando o R 4.4.3"
    $argR = $comuns + @(('/DIR="' + $RDir + '"'), '/MERGETASKS="!desktopicon,!quicklaunchicon,!recordversion,!associate"', ('/LOG="' + $Logs + '\r-instalador.log"'))
    $p = Start-Process -FilePath (Join-Path $Downloads "R-4.4.3-win.exe") -ArgumentList $argR -Wait -PassThru
    $script:Instalados = @(, @($RDir, "R"))
    if ($p.ExitCode -ne 0) { Falha ("instalador do R saiu com " + $p.ExitCode) }
    Passo "instalando o Rtools44"
    $argRt = $comuns + @(('/DIR="' + $RtDir + '"'), ('/LOG="' + $Logs + '\rtools-instalador.log"'))
    $p = Start-Process -FilePath (Join-Path $Downloads "rtools44-6459-6401.exe") -ArgumentList $argRt -Wait -PassThru
    $script:Instalados = @(@($RtDir, "Rtools44"), @($RDir, "R"))
    if ($p.ExitCode -ne 0) { Falha ("instalador do Rtools saiu com " + $p.ExitCode) }

    Passo "extraindo o MetaboAnalystR"
    New-Item -ItemType Directory -Force ($U + "\src") | Out-Null
    # O tar do Windows, por caminho: o que o PATH resolve primeiro pode ser o do Git, que le "C:" como
    # nome de host remoto e falha com "Cannot connect to C: resolve failed" (medido na primeira montagem).
    Corre (Join-Path $env:SystemRoot "System32\tar.exe") @("-xzf", (Join-Path $Downloads "MetaboAnalystR-6bf5462.tar.gz"), "-C", ($U + "\src")) ($Logs + "\tar.log")
    $Fonte = (Get-ChildItem ($U + "\src") -Directory | Select-Object -First 1).FullName

    # ------------------------------------------------------------------------ 4. pacotes
    # O ambiente deste processo e o que o R herda: nada de R_* da maquina, e as quatro variaveis que
    # o R le apontadas para um caminho que nao existe. Sem isso, numa maquina com R instalado, a
    # biblioteca do usuario entraria antes da que se monta.
    Get-ChildItem env: | Where-Object { $_.Name -like "R_*" } | ForEach-Object { Remove-Item ("env:" + $_.Name) }
    foreach ($v in @("R_LIBS_USER", "R_LIBS_SITE", "R_ENVIRON_USER", "R_PROFILE_USER")) {
        Set-Item ("env:" + $v) ($U + "\nao-existe")
    }
    $env:RTOOLS44_HOME = $RtDir
    $env:REFINO_R_FONTE = $Fonte
    $env:REFINO_R_LOGS = $Logs
    $Rscript = $RDir + "\bin\x64\Rscript.exe"
    Passo "instalando dependencias e compilando o MetaboAnalystR"
    Corre $Rscript @("--vanilla", (Join-Path $Raiz "scripts\r_portatil_instala.R")) ($Logs + "\instala.log")
    Remove-Item env:RTOOLS44_HOME

    # ------------------------------------------------------------------------ 5. lista de pacotes
    Passo "derivando a lista de pacotes dos fluxos do modo local"
    Corre $Py @((Join-Path $Raiz "scripts\r_portatil.py"), "deriva", "--r-home", $RDir, "--roteiro", $Roteiro, "--dados", $Dados, "--saida", ($Saida + "\namespaces"), "--lista", ($Saida + "\r_portatil_pacotes.txt")) ($Logs + "\deriva.log")
    Get-Content ($Logs + "\deriva.log") | Write-Host

    # ------------------------------------------------------------------------ 6 e 7. poda e manifesto
    Passo "podando"
    Corre $Py @((Join-Path $Raiz "scripts\r_portatil.py"), "poda", "--origem", $RDir, "--destino", ($U + "\arvore"), "--lista", ($Saida + "\r_portatil_pacotes.txt")) ($Logs + "\poda.log")
    Get-Content ($Logs + "\poda.log") | Write-Host
    Corre $Py @((Join-Path $Raiz "scripts\r_portatil.py"), "manifesto", "--arvore", ($U + "\arvore"), "--saida", ($Saida + "\r_portatil_arvore.sha256")) ($Logs + "\manifesto.log")
    Get-Content ($Logs + "\manifesto.log") | Write-Host
    Corre $Py @((Join-Path $Raiz "scripts\r_portatil.py"), "licencas", "--arvore", ($U + "\arvore"), "--saida", ($Saida + "\r_portatil_licencas.tsv")) ($Logs + "\licencas.log")
    Get-Content ($Logs + "\licencas.log") | Write-Host

    # ------------------------------------------------------------------------ 8. desinstalar
    if (-not $ManterInstalado) { Desinstala $true }

    if ($Versiona) {
        Passo "versionando a lista, os despejos e o manifesto em scripts\"
        Copy-Item ($Saida + "\r_portatil_pacotes.txt") (Join-Path $Raiz "scripts\r_portatil_pacotes.txt") -Force
        Copy-Item ($Saida + "\r_portatil_arvore.sha256") (Join-Path $Raiz "scripts\r_portatil_arvore.sha256") -Force
        Copy-Item ($Saida + "\r_portatil_licencas.tsv") (Join-Path $Raiz "scripts\r_portatil_licencas.tsv") -Force
        $dirNs = Join-Path $Raiz "scripts\r_portatil_namespaces"
        if (Test-Path $dirNs) { Remove-Item -Recurse -Force $dirNs }
        Copy-Item ($Saida + "\namespaces") $dirNs -Recurse
    }
}
finally {
    if ((-not $ManterInstalado) -and (-not $script:Desinstalou) -and ($script:Instalados.Count -gt 0)) {
        Desinstala $false
    }
    & subst.exe $U /D
}

Passo ("pronto: arvore em " + (Join-Path $Build "arvore"))
exit 0
