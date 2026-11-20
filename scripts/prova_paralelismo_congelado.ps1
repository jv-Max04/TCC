# prova_paralelismo_congelado.ps1 - a etapa 3b do roteiro de entrega.
#
# Prova, DENTRO do executavel entregue, que a caixa "usar varios nucleos" muda o TEMPO e nao o
# RESULTADO. E o determinismo e a garantia de `correction/base.py` medidos onde eles passam a
# importar: a suite prova bit-identidade em processo, e o que viaja na entrega e o binario.
#
# ============================================================================
# AS DUAS CORRIDAS SAO EM LANCAMENTOS SEPARADOS, e sem isso a prova e VACUA
# ============================================================================
#
# `chave_da_correcao` **exclui `n_jobs`** — e a premissa da propria exclusao de
# `Corrector.identidade()` —, entao dois pedidos que diferem so na caixa **colidem na mesma
# chave**: o segundo nao correria o corretor, devolveria o guardado, e a comparacao seria de um
# artefato consigo mesmo. `CORRECOES` e memoria do PROCESSO, entao relancar o binario entre as
# duas restaura a separacao sem mudar entrada nem semente.
#
# `mede_serrf_congelado.ps1` lanca e encerra o binario por corrida, entao a separacao vem de
# graca — e e por isso que este script o INVOCA em vez de repetir o laco de espera dele.
#
# ============================================================================
# «PARALELIZOU» E CONTADO E NAO CRONOMETRADO
# ============================================================================
#
# Medir paralelismo por relogio engana: uma maquina carregada daria "nao paralelizou" sobre um
# binario perfeito. O que se afirma sao os WORKERS e os NUCLEOS OCUPADOS, que a sonda conta; o
# tempo fica como numero REPORTADO.
#
# Sem acentos no relato, como todo o ferramental deste repositorio: a code page desta maquina e
# cp932, e prosa acentuada na consola ja matou uma sonda no passo que decide.

param(
    [string]$Modo = "neg_c18",
    [int]$TetoMin = 45,
    [string]$Saida = "dist\paralelismo-congelado-verdicto.json"
)

$ErrorActionPreference = "Stop"
$Raiz = Split-Path -Parent $PSScriptRoot
Set-Location $Raiz

$Sonda = Join-Path $Raiz "scripts\mede_serrf_congelado.ps1"
$med = "serrf\data\input\measurements_$Modo.csv"
$features = (Get-Content $med | Measure-Object -Line).Lines - 2

$paraleloJson = Join-Path $Raiz "dist\u5-paralelo.json"
$serieJson = Join-Path $Raiz "dist\u5-serie.json"
$paraleloCsv = Join-Path $Raiz "dist\u5-paralelo.csv"
$serieCsv = Join-Path $Raiz "dist\u5-serie.csv"

# MATAR POR NOME ATINGE PROGRAMAS DE QUEM CORRE.
#
# `pythonw.exe` e o interpretador sem consola de QUALQUER instalacao de Python do Windows: um
# `Get-Process refino, pythonw | Stop-Process -Force` mataria o que nao e desta sonda, como o
# `pytest` de quem corre a suite ao lado. So se encerra o que corre de dentro de `dist`.
function Encerra {
    $nosso = Join-Path $Raiz "dist"
    Get-Process refino, pythonw -ErrorAction SilentlyContinue |
        Where-Object { $_.Path -and $_.Path.StartsWith($nosso, [StringComparison]::OrdinalIgnoreCase) } |
        Stop-Process -Force -ErrorAction SilentlyContinue
}

$falhas = @()
function Afirma {
    param([string]$Passo, [bool]$Ok, [string]$Detalhe)
    $marca = if ($Ok) { "OK  " } else { "FALHA" }
    if (-not $Ok) { $script:falhas += $Passo }
    Write-Host ("[{0}] {1}  {2}" -f $marca, $Passo, $Detalhe)
}

Write-Host ""
Write-Host "=== U5 - a identidade bit a bit DENTRO do binario"
Write-Host ("  modo: {0} ({1} features)  teto: {2} min" -f $Modo, $features, $TetoMin)

Encerra
Start-Sleep -Seconds 3

Write-Host ""
Write-Host "--- corrida 1: a caixa TICADA"
& powershell.exe -NoProfile -ExecutionPolicy Bypass -File $Sonda `
    -Modo $Modo -TetoMin $TetoMin -Paralelo "true" `
    -Saida $paraleloJson -ArtefatoEm $paraleloCsv
$codigo1 = $LASTEXITCODE

Encerra
Start-Sleep -Seconds 5

Write-Host ""
Write-Host "--- corrida 2: a caixa DESTICADA, em lancamento NOVO do binario"
& powershell.exe -NoProfile -ExecutionPolicy Bypass -File $Sonda `
    -Modo $Modo -TetoMin $TetoMin -Paralelo "false" `
    -Saida $serieJson -ArtefatoEm $serieCsv
$codigo2 = $LASTEXITCODE

Encerra

Write-Host ""
Write-Host "=== VEREDITO"
Afirma "as duas corridas terminaram com 200" (($codigo1 -eq 0) -and ($codigo2 -eq 0)) `
    "exit paralelo=$codigo1 serie=$codigo2"
if ($falhas.Count -gt 0) {
    Write-Host ""
    Write-Host "PARADO: sem as duas corridas nao ha o que comparar."
    exit 1
}

$p = Get-Content $paraleloJson -Raw | ConvertFrom-Json
$s = Get-Content $serieJson -Raw | ConvertFrom-Json

# ---------------------------------------------------------------------------
# (1) A IDENTIDADE - o que este script existe para provar
# ---------------------------------------------------------------------------
$hashP = (Get-FileHash -Algorithm SHA256 -LiteralPath $paraleloCsv).Hash
$hashS = (Get-FileHash -Algorithm SHA256 -LiteralPath $serieCsv).Hash
Afirma "os dois artefatos sao BYTE A BYTE identicos" ($hashP -eq $hashS) `
    "SHA-256 paralelo=$($hashP.Substring(0,16))... serie=$($hashS.Substring(0,16))..."

# ---------------------------------------------------------------------------
# (2) A ANCORA CONTRA VACUO — sem ela, dois arquivos vazios sao identicos
# ---------------------------------------------------------------------------
# O CABECALHO E CONTADO NO ARQUIVO, e nao escrito a mao. O formato `toserrf` tem QUATRO linhas
# de metadados antes da primeira feature (`batch`, `sampleType`, `time`, `label`), e uma ancora
# `linhas == features + 1` reprovaria dois artefatos perfeitos.
#
# As linhas de metadados comecam por `NA,` ou `No,`; as de feature comecam por um numero. Derivar o
# bloco do proprio arquivo e o que impede a ancora de caducar se o formato ganhar uma quinta.
$cabecalho = @(Get-Content $paraleloCsv | Select-Object -First 20 |
                Where-Object { $_ -match '^(NA|No),' }).Count
$linhasEsperadas = $features + $cabecalho
Afirma "o cabecalho do formato foi RECONHECIDO no arquivo" ($cabecalho -ge 1) `
    "linhas de metadados: $cabecalho (o formato toserrf tem batch, sampleType, time e label)"
Afirma "os dois artefatos tem CONTEUDO e o tamanho do modo" `
    (($p.artefato_bytes -gt 1MB) -and ($s.artefato_bytes -gt 1MB) -and `
     ($p.artefato_linhas -eq $linhasEsperadas) -and ($s.artefato_linhas -eq $linhasEsperadas)) `
    "bytes P=$($p.artefato_bytes) S=$($s.artefato_bytes)  linhas P=$($p.artefato_linhas) S=$($s.artefato_linhas) (esperado $features features + $cabecalho de cabecalho = $linhasEsperadas)"

# ---------------------------------------------------------------------------
# (3) CADA CORRIDA CORREU O CORRETOR — um acerto de cache e sub-segundo
# ---------------------------------------------------------------------------
#
# Sem esta, a identidade de cima passa por vacuidade: se a segunda corrida tivesse batido no
# cache da primeira, os bytes seriam iguais por SEREM OS MESMOS BYTES.
Afirma "as duas corridas correram o corretor de facto" `
    (($p.parede_s -gt 5) -and ($s.parede_s -gt 5)) `
    "parede paralelo=$($p.parede_s)s serie=$($s.parede_s)s (acerto de cache e sub-segundo)"

# ---------------------------------------------------------------------------
# (4) PARALELIZOU — contado, nao cronometrado
# ---------------------------------------------------------------------------
Afirma "a corrida TICADA abriu workers e ocupou mais de um nucleo" `
    (($p.workers_pico -gt 1) -and ($p.nucleos_ocupados -gt 1.5)) `
    "workers=$($p.workers_pico) nucleos=$($p.nucleos_ocupados)"
Afirma "a corrida DESTICADA nao abriu worker nenhum" ($s.workers_pico -eq 0) `
    "workers=$($s.workers_pico) nucleos=$($s.nucleos_ocupados)"

# ---------------------------------------------------------------------------
# (5) OS NUMEROS QUE SE REPORTAM, e NAO se afirmam
# ---------------------------------------------------------------------------
$ganho = if ($p.parede_s -gt 0) { [math]::Round($s.parede_s / $p.parede_s, 2) } else { 0 }
Write-Host ""
Write-Host "--- reportado (nao e assercao: o relogio depende da carga da maquina)"
Write-Host ("  parede    : paralelo {0} s  |  serie {1} s  |  ganho {2}x" -f $p.parede_s, $s.parede_s, $ganho)
Write-Host ("  memoria   : pico da arvore {0} MB paralelo  |  {1} MB serie" -f $p.rss_pico_mb, $s.rss_pico_mb)
Write-Host ("  workers   : {0} paralelo  |  {1} serie" -f $p.workers_pico, $s.workers_pico)

@{
    modo = $Modo; features = $features
    sha256 = $hashP
    identicos = ($hashP -eq $hashS)
    paralelo = @{
        parede_s = $p.parede_s; workers = $p.workers_pico
        nucleos = $p.nucleos_ocupados; rss_pico_mb = $p.rss_pico_mb
        bytes = $p.artefato_bytes; linhas = $p.artefato_linhas
    }
    serie = @{
        parede_s = $s.parede_s; workers = $s.workers_pico
        nucleos = $s.nucleos_ocupados; rss_pico_mb = $s.rss_pico_mb
        bytes = $s.artefato_bytes; linhas = $s.artefato_linhas
    }
    ganho = $ganho
    quando = (Get-Date).ToString("s")
    lancamentos = 'SEPARADOS - chave_da_correcao exclui n_jobs, e CORRECOES e memoria do processo'
} | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $Saida -Encoding utf8

Write-Host ""
Write-Host ("veredito gravado em {0}" -f $Saida)
if ($falhas.Count -gt 0) {
    Write-Host ("FALHAS: {0}" -f ($falhas -join " | "))
    exit 1
}
Write-Host "TUDO VERDE: o mesmo resultado, nos dois regimes, em dois lancamentos do binario."
exit 0
