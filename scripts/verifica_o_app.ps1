# verifica_o_app.ps1 — exercita o APLICATIVO CONGELADO pelas rotas que a janela usa.
#
# **Por que existe, e por que nao e mais um dos quatro `prova_*`.** Aqueles provam propriedades
# especificas do binario — a grade corre, os artefatos saem, o paralelismo muda o tempo e nao o
# resultado, as figuras desenham. Este responde a outra pergunta: *o aplicativo
# faz, de ponta a ponta, o que foi implementado para fazer?*
#
# **So-ASCII no relato, pela mesma razao que `congela_windows.ps1` declara:** a code page desta
# maquina e cp932, e prosa acentuada num relato de arnes mata o proprio arnes.
#
# Uso: powershell -NoProfile -ExecutionPolicy Bypass -File scripts\verifica_o_app.ps1

$ErrorActionPreference = "Continue"
$Repo = Split-Path $PSScriptRoot -Parent
$Dist = Join-Path $Repo "dist\refino"
$Exe  = Join-Path $Dist "refino.exe"
$Modo = "neg_c18"
$Med  = Join-Path $Repo "serrf\data\input\measurements_$Modo.csv"
$Lst  = Join-Path $Repo "serrf\data\input\Sample_List_$Modo.csv"
$Saida = Join-Path $Repo "out\verifica-app"
New-Item -ItemType Directory -Force -Path $Saida | Out-Null

$script:ok = 0
$script:falhas = 0
function Passo([string]$nome, [bool]$bem, [string]$detalhe) {
    if ($bem) { $script:ok++; $m = "[OK  ]" } else { $script:falhas++; $m = "[FALHA]" }
    "{0} {1,-52} {2}" -f $m, $nome, $detalhe
}

. (Join-Path $PSScriptRoot "porta_que_serve.ps1")

"=== subindo o binario congelado ==="
$proc = Start-Process -FilePath $Exe -PassThru
$base = PortaQueServe -Processo $proc -TetoSeg 180
Passo "o binario arrancou" (-not $proc.HasExited) "PID $($proc.Id)"
Passo "a API responde" ([bool]$base) "base: $base"
if (-not $base) { $proc | Stop-Process -Force -ErrorAction SilentlyContinue; exit 1 }

$comuns = @("-s", "-F", "measurements=@$Med", "-F", "sample_list=@$Lst", "-F", "mode=$Modo")

# --- 1. VALIDACAO: a matriz e aceita e os problemas sao listados --------------------------------
$r = & curl.exe @comuns "$base/matrizes/validar" 2>$null | ConvertFrom-Json
Passo "validar: a matriz e aceita" ($r.aceita -eq $true) "features=$($r.n_features) amostras=$($r.n_samples) problemas=$($r.problemas.Count)"

# --- 2. DIAGNOSTICO: classifica o padrao do defeito ---------------------------------------------
$r = & curl.exe @comuns "$base/matrizes/diagnosticar" 2>$null | ConvertFrom-Json
Passo "diagnosticar: devolve classe e confianca" ([bool]$r.defect_type) "tipo=$($r.defect_type) confianca=$([math]::Round($r.confidence,4))"
Passo "diagnosticar: traz os lotes e os dados de figura" ($null -ne $r.lotes -and $null -ne $r.trajetoria_dos_qcs) "lotes=$($r.lotes.Count) trajetoria=$($null -ne $r.trajetoria_dos_qcs) pca=$($null -ne $r.pca) grupos=$($r.grupos_disponiveis -join ',')"

# --- 3. A ARVORE do `--auto aprendida`: a rota de recomendar ------------------------------------
#
# O botao Palpite e a caixa Automatico usam outra rota, a do passo 6b. Esta fica porque a rota
# continua no produto, e o conferidor da arvore le a resposta dela que o congelamento grava.
$r = & curl.exe @comuns "$base/matrizes/recomendar" 2>$null | ConvertFrom-Json
Passo "recomendar: devolve corretor e confianca" ([bool]$r.correction_proposta) "corretor=$($r.correction_proposta) confianca=$([math]::Round($r.confianca,4)) abstem=$($r.abstem)"
Passo "recomendar: traz a justificativa" ([bool]$r.rationale) "$($r.rationale.Substring(0,[Math]::Min(58,$r.rationale.Length)))"

# --- 4 a 6. As rotas de ARTEFATO sao ASSINCRONAS: 202 com identificador, depois `/trabalhos/{id}`.
#
# Tratadas como sincronas, o arnes leria os 97 bytes do `202` como se fossem o CSV: cinco
# FALHAS que acusariam o aplicativo e seriam do instrumento.
function Artefato([string]$rota, [string[]]$extra, [string]$destino, [int]$tetoSeg = 600) {
    $disparo = (& curl.exe @comuns @extra "$base/matrizes/$rota" 2>$null) -join ""
    $id = ($disparo | ConvertFrom-Json -ErrorAction SilentlyContinue).identificador
    if (-not $id) { return @{ ok = $false; nota = "sem identificador: $($disparo.Substring(0,[Math]::Min(60,$disparo.Length)))" } }
    $limite = (Get-Date).AddSeconds($tetoSeg)
    while ((Get-Date) -lt $limite) {
        Start-Sleep -Seconds 3
        $codigo = & curl.exe -s -o NUL -w "%{http_code}" "$base/trabalhos/$id" 2>$null
        if ($codigo -eq "200") {
            & curl.exe -s -o $destino "$base/trabalhos/$id" 2>$null | Out-Null
            return @{ ok = $true; nota = "id=$($id.Substring(0,8)) codigo=200" }
        }
        if ($codigo -ne "202") { return @{ ok = $false; nota = "codigo inesperado: $codigo" } }
    }
    return @{ ok = $false; nota = "teto de $tetoSeg s sem 200" }
}

foreach ($metodo in @("none", "loess")) {
    $arq = Join-Path $Saida "corrigida_$metodo.csv"
    $r = Artefato "corrigir" @("-F", "method=$metodo", "-F", "n_batches=1") $arq
    $linhas = if (Test-Path $arq) { (Get-Content $arq | Measure-Object -Line).Lines } else { 0 }
    Passo "corrigir($metodo): CSV com cabecalho e linhas" ($r.ok -and $linhas -gt 100) "linhas=$linhas $($r.nota)"
}

$arq = Join-Path $Saida "processada.csv"
$r = Artefato "processar" @("-F", "method=loess", "-F", "imputation=one_fifth_min", "-F", "transformation=log10", "-F", "scaling=auto") $arq
$linhas = if (Test-Path $arq) { (Get-Content $arq | Measure-Object -Line).Lines } else { 0 }
Passo "processar: pipeline pos-correcao devolve matriz" ($r.ok -and $linhas -gt 100) "linhas=$linhas $($r.nota)"

$arq = Join-Path $Saida "metaboanalyst.csv"
$r = Artefato "exportar" @("-F", "max_features=5000") $arq
$cab = if (Test-Path $arq) { (Get-Content $arq -TotalCount 1) } else { "" }
Passo "exportar: cabecalho do MetaboAnalyst" ($cab -like "Sample,Label,*") "$($cab.Substring(0,[Math]::Min(46,$cab.Length))) $($r.nota)"

# --- 6b. O PALPITE MEDIDO: a rota que o botao Palpite e a caixa Automatico usam ----------------
#
# E a rota que a janela pede, e nenhuma outra prova do congelado a exercita. E assincrona
# como as de artefato, e o resultado e o JSON do Palpite medido em 200 features, que
# `scripts\confere_palpite_no_congelado.py` compara, campo a campo, com o da mesma rota corrida
# sobre `src/`. O arquivo antigo sai antes do disparo: sem isso, um disparo que falhasse deixaria
# o corpo da corrida anterior a ser lido como se fosse desta. O teto e 900 s (52,7 a 59,7 s por
# modo medidos num Ryzen de 12 nucleos), e os segundos vao no relato porque sao o tempo do
# Palpite no binario.
$arq = Join-Path $Saida "palpite-medido.json"
if (Test-Path $arq) { Remove-Item $arq -Force }
$relogio = [Diagnostics.Stopwatch]::StartNew()
$r = Artefato "palpite" @() $arq 900
$segundos = [Math]::Round($relogio.Elapsed.TotalSeconds, 1)
$p = $null
if (Test-Path $arq) { $p = Get-Content $arq -Raw -Encoding UTF8 | ConvertFrom-Json -ErrorAction SilentlyContinue }
Passo "palpite medido: terminou com 200" ($r.ok -and ($null -ne $p)) "segundos=$segundos $($r.nota)"
$completo = ($null -ne $p) -and [bool]$p.correction_proposta -and ($p.n_features_medidas -eq 200) -and [bool]$p.rationale
Passo "palpite medido: corretor, 200 features, razao" $completo "corretor=$($p.correction_proposta) features=$($p.n_features_medidas) abstem=$($p.abstem) margem=$($p.margem)"

# --- 7. EXECUCOES: o botao «Executar pipeline» da janela, com o par de grupos -----------------------
#
# **Nao ha rota `/figuras`, e isso e desenho e nao lacuna.** As figuras sao desenhadas do lado de
# quem mostra, a partir dos DADOS que as respostas carregam — `trajetoria_dos_qcs` e `pca` no
# diagnostico, e o relatorio de qualidade aqui. A primeira versao deste arnes procurou um catalogo
# que nao existe e reportou "0 figuras", que e o instrumento a acusar o alvo.
# Os DOZE campos obrigatorios vao todos, e nenhum tem padrao na rota: quem executa declara cada
# escolha, e a rota recusa a que faltar em vez de a preencher, nomeando o campo que faltou.
$disparo = (& curl.exe @comuns `
    "-F" "estudo=verificacao-do-app" "-F" "apresentacao=exibida" `
    "-F" "correction=loess" "-F" "n_batches=1" "-F" "rsd_variant=outlier_removed" `
    "-F" "imputation=one_fifth_min" "-F" "sample_normalization=none" `
    "-F" "transformation=log10" "-F" "scaling=auto" `
    "-F" "group_a=ATLL" "-F" "group_b=NoHTLV" `
    "$base/execucoes" 2>$null) -join ""
$id = ($disparo | ConvertFrom-Json -ErrorAction SilentlyContinue).identificador
Passo "execucoes: o pipeline completo aceita e devolve id" ([bool]$id) "$($disparo.Substring(0,[Math]::Min(64,$disparo.Length)))"
if ($id) {
    $limite = (Get-Date).AddSeconds(900); $corpo = $null
    while ((Get-Date) -lt $limite) {
        Start-Sleep -Seconds 5
        $codigo = & curl.exe -s -o NUL -w "%{http_code}" "$base/trabalhos/$id" 2>$null
        if ($codigo -eq "200") { $corpo = (& curl.exe -s "$base/trabalhos/$id" 2>$null | ConvertFrom-Json); break }
        if ($codigo -ne "202") { break }
    }
    Passo "execucoes: terminou e devolve o relatorio" ($null -ne $corpo) "campos: $(($corpo.PSObject.Properties.Name | Select-Object -First 6) -join ',')"
    $q = $corpo.qualidade
    Passo "execucoes: o relatorio de qualidade veio" ($null -ne $q) "metricas: $(($q.PSObject.Properties.Name | Select-Object -First 5) -join ',')"
    # PLS-DA nunca sem validacao: se ha score, tem de haver `q2` E `p` de permutacao.
    # O PLS-DA vem no TOPO da resposta, e nao dentro de `qualidade`:
    # lido de `$q`, ele era sempre `$null` -- sem `Set-StrictMode` o PowerShell nao acusa membro
    # que nao existe --, e este passo dava verde sem nunca olhar um score. O pedido traz `group_a` e
    # `group_b`, entao o PLS-DA TEM de vir: a presenca e o passo de antes, e e ela que impede o
    # seguinte de passar por vacuidade outra vez.
    $pls = $corpo.plsda
    Passo "regra 6 do README: o PLS-DA veio, porque o pedido traz os dois grupos" ($null -ne $pls) "plsda=$($null -ne $pls)"
    $temScore = $null -ne $pls
    $temPar = $temScore -and ($null -ne $pls.q2) -and ($null -ne $pls.permutation_pvalue)
    Passo "regra 6 do README: PLS-DA nunca sem q2 e p" ((-not $temScore) -or $temPar) "score=$temScore par completo=$temPar"
}

# --- 8. MULTIVERSO: a grade, com a estimativa de custo antes do disparo ---------------------------
# `corrections`, e nao `correcoes`: o FastAPI ignora campo de formulario que a rota nao
# declara, e com o nome errado este passo disparava a grade PADRAO e so conferia o id.
# As aspas do JSON vao com barra invertida: o Windows PowerShell 5.1 passa o argumento ao
# programa nativo sem escapar as aspas internas, e o curl recebia `[none]`, que a rota recusa com
# "`corrections` nao e JSON". Com `\"`, o curl recebe `["none"]`.
$disparo = (& curl.exe @comuns "-F" "corrections=[\`"none\`"]" "$base/multiversos" 2>$null) -join ""
$idmv = ($disparo | ConvertFrom-Json -ErrorAction SilentlyContinue).identificador
Passo "multiversos: a grade aceita e devolve id" ([bool]$idmv) "$($disparo.Substring(0,[Math]::Min(64,$disparo.Length)))"

"" ; "passos=$($script:ok + $script:falhas)  ok=$($script:ok)  falhas=$($script:falhas)"
$proc.CloseMainWindow() | Out-Null
Start-Sleep -Seconds 3
if (-not $proc.HasExited) { $proc | Stop-Process -Force -ErrorAction SilentlyContinue }
if ($script:falhas -gt 0) { exit 1 }
