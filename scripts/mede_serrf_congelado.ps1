# Mede quanto custa `serrf` no MAIOR modo, DENTRO do executavel congelado.
#
# A lacuna que isto fecha e de POPULACAO e nao de valor: existiam tres medicoes de SERRF no
# `pos_hilic` - 209,9 s (processos, correcao sozinha), 1.076,6 s (serial, n_jobs=1) e
# 1.209,2 s (threads, no regime do congelado mas FORA dele) - e nenhuma dentro do binario.
# O que havia dentro do binario era uma
# desistencia: 872 leituras em 900 s sem estado terminal, sem causa atribuida.
#
# "Nao terminou em 900 s" NAO e "nao termina". Esta sonda espera ate o fim.
#
# Sem acentos de proposito: o Windows PowerShell 5.1 le um .ps1 sem BOM como ANSI, e a
# code page desta maquina e cp932 — um travessao sai como mojibake. Todo o resto do
# ferramental deste repositorio ja e assim.

# ============================================================================
# TRES PARAMETROS, e a razao e NAO haver segunda copia do laco de espera
# ============================================================================
#
# `prova_paralelismo_congelado.ps1` precisa de duas corridas do MESMO corretor em regimes
# diferentes, e de comparar os bytes. Escrever um segundo laco de espera e o que este bloco evita:
# duas politicas de espera divergem no dia em que uma delas mudar, e o numero medido passa a
# depender de qual sonda correu.
#
#   -Paralelo     manda `paralelo=true|false` a rota, que e a caixa "usar varios nucleos"
#   -ArtefatoEm   guarda o corpo do artefato em vez de so lhe medir o tamanho
#   -EsperaSeg    quanto se espera pela porta antes de desistir
#
# A sonda CONTA os workers e mede o pico de memoria da arvore. Sem isso, "paralelizou" seria
# inferido do relogio, e uma maquina carregada daria "nao paralelizou" sobre um binario perfeito.
param(
    [string]$Exe   = "dist\refino\refino.exe",
    [string]$Modo  = "pos_hilic",
    [string]$Metodo = "serrf",
    [int]$TetoMin  = 45,
    [string]$Saida = "dist\serrf-congelado-verdicto.json",
    # **`[string]` e nao `[bool]`, e o motivo e a fronteira do `-File`.** O PowerShell
    # converte todo argumento de `-File` para TEXTO antes de o entregar, e um parametro
    # `[bool]` recusa string: `-Paralelo $true` chegava como "True" e o script morria em
    # `Cannot convert value "System.String" to type "System.Boolean"`. Aqui o valor tem
    # o mesmo vocabulario que o `curl` manda a rota, e nao ha conversao no meio.
    [ValidateSet("true", "false")][string]$Paralelo = "true",
    [string]$ArtefatoEm = "",
    [int]$EsperaSeg = 180
)

$ErrorActionPreference = "Stop"
$Raiz = Split-Path -Parent $PSScriptRoot
Set-Location $Raiz

# **A porta e a que RESPONDE `/health`, nunca a primeira da lista.** A tela de arranque do
# PyInstaller abre uma porta de escuta no loopback ANTES da nossa, ela aceita a ligacao e nunca
# responde HTTP, e o `curl` fica pendurado. O motivo e a medicao estao em
# `scripts/porta_que_serve.ps1`.
. (Join-Path $PSScriptRoot "porta_que_serve.ps1")

$med = "serrf\data\input\measurements_$Modo.csv"
$lst = "serrf\data\input\Sample_List_$Modo.csv"
foreach ($f in @($Exe, $med, $lst)) {
    if (-not (Test-Path $f)) { Write-Host "FALTA: $f"; exit 1 }
}

$features = (Get-Content $med | Measure-Object -Line).Lines - 2
Write-Host ""
Write-Host "  alvo   : $Exe"
Write-Host "  modo   : $Modo  ($features features)"
Write-Host "  metodo : $Metodo"
Write-Host "  teto   : $TetoMin min"
Write-Host ""

# A espera pela porta e um LACO e nao um tempo fixo. Com `noarchive` o primeiro arranque
# sobre cache frio custa 59,4 s medidos, e 25 s fixos davam "o servidor nao subiu" sobre um
# binario a arrancar bem.
#
$proc = Start-Process -FilePath $Exe -PassThru
$base = PortaQueServe -Processo $proc -TetoSeg $EsperaSeg
if (-not $base) {
    Write-Host "  nenhuma porta deste PID respondeu /health em $EsperaSeg s"
    $proc | Stop-Process -Force -ErrorAction SilentlyContinue
    exit 1
}
Write-Host "  servidor em $base"

# O relogio comeca no DISPARO, que e o que o profissional sente.
#
# A forma do upload e `-F "campo=@arquivo"` e nao `-F "campo=<arquivo"`: o `@` envia como
# ARQUIVO, e o `<` envia o conteudo como campo comum — que estoura o `max_part_size` de
# 1024 KB do multipart, porque os quatro `measurements` passam de 1 MB. A sonda do
# multiverso ja usava `@`; esta nasceu com `<` e morreu no disparo com
# "Part exceeded maximum size of 1024KB". Defeito da sonda, nao do produto.
$t0 = Get-Date
$paraleloTexto = $Paralelo

# **OS WORKERS SAO OS FILHOS DESTE PROCESSO, e nao todo `pythonw.exe` da maquina.**
#
# `Get-Process -Name pythonw` conta o interpretador sem consola de QUALQUER instalacao de Python
# do Windows. A direccao do erro e falso VERDE: um `pythonw` alheio faria a sonda reportar
# paralelismo que nao houve. O gesto e o mesmo do eixo dos consoles do congelamento: filtrar
# pelo PAI.
function _WorkersNossos {
    @(Get-CimInstance Win32_Process -Filter "Name='pythonw.exe'" -ErrorAction SilentlyContinue |
        Where-Object { $_.ParentProcessId -eq $proc.Id } |
        ForEach-Object { Get-Process -Id $_.ProcessId -ErrorAction SilentlyContinue })
}
$disparo = curl.exe -s -X POST "$base/matrizes/corrigir" `
    -F "measurements=@$med" -F "sample_list=@$lst" `
    -F "mode=$Modo" -F "method=$Metodo" -F "n_batches=1" -F "paralelo=$paraleloTexto"
$idJob = ($disparo | ConvertFrom-Json).identificador
if (-not $idJob) {
    Write-Host "  o disparo nao devolveu identificador: $disparo"
    $proc | Stop-Process -Force
    exit 1
}
Write-Host "  disparado: $idJob"
Write-Host ""

$leituras = 0
$estado = "executando"
$codigo = 0
$limite = (Get-Date).AddMinutes($TetoMin)
$proximoAviso = (Get-Date).AddMinutes(2)

# **OS TRES NUMEROS QUE O RELOGIO NAO DA.** `workers` e a contagem de `pythonw` —
# «paralelizou» CONTADO e nao cronometrado; `nucleos` e o tempo de CPU da arvore a
# dividir pelo de parede, que separa «doze processos» de «doze processos a fazer alguma
# coisa»; `rssPico` e o custo em memoria que a ressalva dos doze workers tinha sem
# quantidade nenhuma.
$workersPico = 0
$rssPico = 0
$cpuTotal = 0.0
$marcoCpu = Get-Date

# **A LINHA DE BASE DO CPU, e sem ela o numero e a mistura de duas quantidades.**
#
# `TotalProcessorTime` conta desde o ARRANQUE do processo, e o denominador conta desde este marco.
# Sem subtrair o que o binario ja gastou a arrancar — importar `numpy`, `sklearn`, subir o
# servidor —, os «nucleos ocupados» somam o arranque a correcao e dividem pelo tempo da correcao.
# **O vies e para CIMA, e e maior nas corridas curtas**: o mesmo arranque diluido em 82 s pesa seis
# vezes mais que diluido em 490 s.
$arvoreBase = @(Get-Process -Id $proc.Id -ErrorAction SilentlyContinue) + $(_WorkersNossos)
$cpuBase = ($arvoreBase | ForEach-Object { $_.TotalProcessorTime.TotalSeconds } |
            Measure-Object -Sum).Sum
if (-not $cpuBase) { $cpuBase = 0.0 }
while ((Get-Date) -lt $limite) {
    Start-Sleep -Seconds 5
    $leituras++
    $meus = _WorkersNossos
    $arvore = @(Get-Process -Id $proc.Id -ErrorAction SilentlyContinue) + $meus
    $nPythonw = @($meus).Count
    if ($nPythonw -gt $workersPico) { $workersPico = $nPythonw }
    $rss = ($arvore | Measure-Object -Property WorkingSet64 -Sum).Sum
    if ($rss -gt $rssPico) { $rssPico = $rss }
    $cpuAgora = ($arvore | ForEach-Object { $_.TotalProcessorTime.TotalSeconds } |
                 Measure-Object -Sum).Sum
    if ($cpuAgora -gt $cpuTotal) { $cpuTotal = $cpuAgora }
    # O codigo de saida NAO atravessa tubo: `-w` escreve no fim do corpo e le-se de la.
    $r = curl.exe -s -o NUL -w "%{http_code}" "$base/trabalhos/$idJob"
    $codigo = [int]$r
    if ($codigo -ne 202) { $estado = "terminal"; break }
    if ((Get-Date) -ge $proximoAviso) {
        $decorrido = [math]::Round(((Get-Date) - $t0).TotalSeconds, 1)
        Write-Host ("    {0,7:N1}s  ainda 202  ({1} leituras)" -f $decorrido, $leituras)
        $proximoAviso = (Get-Date).AddMinutes(2)
    }
}

$parede = [math]::Round(((Get-Date) - $t0).TotalSeconds, 1)
$segundosDeAmostragem = [math]::Max(1.0, ((Get-Date) - $marcoCpu).TotalSeconds)
$nucleos = [math]::Round([math]::Max(0.0, $cpuTotal - $cpuBase) / $segundosDeAmostragem, 2)
$tamanho = 0
$linhas = 0
if ($codigo -eq 200) {
    # O artefato vai para onde o chamador pedir, porque a prova do paralelismo compara os
    # bytes das duas corridas: um temporario apagado no fim tornaria a comparacao impossivel.
    $destino = if ($ArtefatoEm) { $ArtefatoEm } else { [System.IO.Path]::GetTempFileName() }
    curl.exe -s -o $destino "$base/trabalhos/$idJob" | Out-Null
    $tamanho = (Get-Item $destino).Length
    $linhas = (Get-Content $destino | Measure-Object -Line).Lines
    if (-not $ArtefatoEm) { Remove-Item -LiteralPath $destino -Force }
}

Write-Host ""
Write-Host ("  parede   : {0} s  ({1} min {2} s)" -f $parede, [math]::Floor($parede/60), [math]::Round($parede%60))
Write-Host ("  leituras : {0}" -f $leituras)
Write-Host ("  codigo   : {0}" -f $codigo)
Write-Host ("  artefato : {0} bytes, {1} linhas" -f $tamanho, $linhas)
Write-Host ("  regime   : paralelo={0}  workers pythonw (pico)={1}  nucleos ocupados={2}" -f $Paralelo, $workersPico, $nucleos)
Write-Host ("  memoria  : pico da arvore {0:N0} MB" -f ($rssPico / 1MB))
Write-Host ""
if ($codigo -eq 200) {
    Write-Host "  VEREDITO: TERMINOU. O par $Modo + $Metodo tem numero dentro do binario."
} elseif ($estado -eq "executando") {
    Write-Host "  VEREDITO: NAO terminou em $TetoMin min. A lacuna continua aberta, com teto novo."
} else {
    Write-Host "  VEREDITO: desfecho $codigo - nao e sucesso nem espera. Ler o corpo."
}

@{
    modo = $Modo; metodo = $Metodo; features = $features
    exe = $Exe; base = $base
    parede_s = $parede; leituras = $leituras; codigo_final = $codigo
    artefato_bytes = $tamanho
    artefato_linhas = $linhas
    artefato_em = $ArtefatoEm
    paralelo = $Paralelo
    workers_pico = $workersPico
    nucleos_ocupados = $nucleos
    rss_pico_mb = [math]::Round($rssPico / 1MB, 1)
    teto_min = $TetoMin
    quando = (Get-Date).ToString("s")
    # `regime_superado` guarda a frase antiga ao lado da que a desfez. Com o interpretador
    # embarcado o congelado lanca processos `loky` de verdade; `JOBLIB_MULTIPROCESSING`
    # ja nao e fixado.
    regime = "dentro do executavel congelado; loky com interpretador embarcado (U2)"
    regime_superado = "ate 07/09/2026: joblib em threads sob sys.frozen"
} | ConvertTo-Json | Set-Content -LiteralPath $Saida -Encoding utf8

$proc | Stop-Process -Force
if ($codigo -eq 200) { exit 0 } else { exit 1 }
