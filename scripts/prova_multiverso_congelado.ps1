# Prova, NO BINARIO CONGELADO, o ciclo do multiverso e o PLS-DA.
#
# ## Por que este script existe SEPARADO de `congela_windows.ps1`
#
# O verdicto daquele script exercita a leitura, o diagnostico, a recomendacao, os tres artefatos
# e a guarda de processos do `serrf`, e nenhum cenario de multiverso: `grep` por `multivers` no
# `congelamento-verdicto.json` devolve nada.
#
# **Separado e nao dentro**, por uma razao de custo: este script dispara uma grade de 1.440
# celulas com `serrf` para ter algo que demore o bastante para a parada ser medivel, e o
# congelamento ja custa minutos. Junta-los faria o portao de embalagem carregar uma medicao que
# ele nao precisa para responder *"o binario esta bem formado?"*.
#
# ## O que ele responde
#
# *"multiverso nao esta funcionando no exe windows?"*: a execucao tem de parar logo depois do
# pedido de cancelamento (0,29 s medidos). Dois grupos de passos pedem explicacao:
#
#   - a ANCORA do oraculo 405-contra-404 (passo 1). Um `GET` numa rota so-POST devolve 405, um
#     caminho que ninguem registou devolve 404. A ancora afirma o 404, sem a qual um servidor que
#     respondesse 405 a tudo faria os dois passos passar por vacuidade.
#   - o HISTORICO e a documentacao desligada (passos 6 e 7). Os tres passos do historico sao os
#     UNICOS desta sonda e do congelamento que tocam o banco por usuario, e por isso os unicos que
#     provam que a migracao correu no arranque do congelado.
#
# **A execucao que o historico procura e a que acabou de ser CANCELADA**, e nao uma concluida: e o
# desfecho que desapareceria sem o registro em disco, e o passo compara o estado do DISCO contra o
# da MEMORIA — se divergirem, o gancho de fecho nao correu.
#
# ## Como rodar
#
#     powershell -NoProfile -ExecutionPolicy Bypass -File scripts\prova_multiverso_congelado.ps1
#
# Exige `dist\refino\refino.exe` ja construido. Sai com o numero de falhas.
$ErrorActionPreference = "Continue"
$Raiz = Split-Path -Parent $PSScriptRoot
$Exe  = Join-Path $Raiz "dist\refino\refino.exe"
$Ent  = Join-Path $Raiz "serrf\data\input"

function Passo($nome, $ok, $detalhe = "") {
    $marca = if ($ok) { "OK  " } else { "FALHA" }
    Write-Host ("{0,-6} {1,-56} {2}" -f $marca, $nome, $detalhe)
    if (-not $ok) { $script:Falhas++ }
}
$script:Falhas = 0
# O eixo em ARQUIVO e nao inline, e a razao e a fronteira do shell. Com o JSON entre aspas, o
# servidor responde `422: corrections nao e JSON`: o PowerShell consome as aspas antes de o
# curl as ver. O `-F "nome=<arquivo"` le do disco e nao atravessa fronteira nenhuma.
$Json = Join-Path $PSScriptRoot "prova_multiverso_eixo.json"

# Marca de tempo do arranque: e ela que torna a direccao negativa do passo do banco capaz de
# distinguir escrita DESTA execucao de arquivo velho deixado por outra.
$inicioDaSonda = Get-Date
# **A porta e a que RESPONDE `/health`, nunca a primeira da lista.** A tela de arranque do
# PyInstaller abre uma porta de escuta no loopback ANTES da nossa, ela aceita a ligacao e nunca
# responde HTTP, e o `curl` fica pendurado. O motivo e a medicao estao em
# `scripts/porta_que_serve.ps1`.
. (Join-Path $PSScriptRoot "porta_que_serve.ps1")

$proc = Start-Process -FilePath $Exe -PassThru
$base = PortaQueServe -Processo $proc -TetoSeg 180
Passo "o binario arrancou" (-not $proc.HasExited) "PID $($proc.Id)"
Passo "o servidor esta em LISTEN e responde /health" ([bool]$base) "base: $base"
if (-not $base) { $proc | Stop-Process -Force -ErrorAction SilentlyContinue; exit 1 }

# --- 1. A ROTA DE CANCELAMENTO EXISTE no binario -----------------------------
# O oraculo e 405-CONTRA-404, e nao casamento de texto no `/openapi.json`.
#
# Duas razoes, e a segunda e a que obriga:
#
#   1. `/openapi.json` NAO EXISTE MAIS no congelado. A documentacao interativa foi desligada no
#      binario, e as quatro rotas dela respondem 404. Um `-match` sobre o
#      corpo de um 404 devolve falso e o passo reprovaria com o binario CERTO.
#   2. Mesmo quando existia, casar o texto do manifesto media a DECLARACAO e nao a rota: o
#      oraculo codificava a FORMA (o caminho escrito com chavetas) e nao a propriedade, que e a
#      rota responder.
#
# **A propriedade, e ela nao depende de manifesto nenhum:** um `GET` numa rota registada so para
# `POST` devolve **405 Method Not Allowed**; um `GET` num caminho que ninguem registou devolve
# **404**. Medido nas duas direcoes contra a arvore antes de se trocar este bloco.
#
# `-o NUL` porque o corpo nao interessa e `-w "%{http_code}"` porque e o codigo que decide.
$codMv = curl.exe -s -o NUL -w "%{http_code}" "$base/multiversos/sonda/cancelar"
$codTr = curl.exe -s -o NUL -w "%{http_code}" "$base/trabalhos/sonda/cancelar"
# A ANCORA, e ela vem primeiro: um caminho que de facto nao existe tem de dar 404. Sem ela, um
# servidor que respondesse 405 a tudo faria os dois passos abaixo passar por vacuidade.
$codAusente = curl.exe -s -o NUL -w "%{http_code}" "$base/multiversos/sonda/rota-que-nao-existe"
Passo "ancora: rota inexistente da 404" ($codAusente -eq "404") "codigo=$codAusente"
Passo "POST /multiversos/{id}/cancelar existe" ($codMv -eq "405") "GET nela deu $codMv (405=existe, 404=nao existe)"
Passo "POST /trabalhos/{id}/cancelar existe" ($codTr -eq "405") "GET nela deu $codTr (405=existe, 404=nao existe)"

# --- 2. O DIAGNOSTICO SERVE OS GRUPOS ----------------------------------------
$diag = curl.exe -s -X POST "$base/matrizes/diagnosticar" `
    -F "measurements=@$Ent\measurements_neg_c18.csv" `
    -F "sample_list=@$Ent\Sample_List_neg_c18.csv" `
    -F "mode=neg_c18"
$temGrupos = $diag -match "grupos_disponiveis"
$nomes = if ($diag -match '"grupos_disponiveis":\[(.*?)\]') { $Matches[1] } else { "" }
Passo "o diagnostico serve grupos_disponiveis" $temGrupos $nomes

# --- 3. O PLS-DA E CALCULADO -------------------------------------------------
# Grade minima e `correction=none` para a execucao ser barata; o que se mede e se o par
# de grupos faz o Q2 e o p existirem.
$exec = curl.exe -s -X POST "$base/execucoes" `
    -F "measurements=@$Ent\measurements_neg_c18.csv" `
    -F "sample_list=@$Ent\Sample_List_neg_c18.csv" `
    -F "mode=neg_c18" -F "estudo=prova" -F "correction=none" -F "n_batches=1" `
    -F "rsd_variant=outlier_removed" -F "imputation=median" -F "sample_normalization=none" `
    -F "transformation=none" -F "scaling=none" -F "apresentacao=exibida" `
    -F "group_a=Asymptomatic" -F "group_b=ATLL"
$idExec = if ($exec -match '"identificador":"([0-9a-f]+)"') { $Matches[1] } else { "" }
Passo "POST /execucoes com par de grupos aceitou" ($idExec -ne "") "id=$idExec"

if ($idExec -ne "") {
    $corpo = ""
    for ($i = 0; $i -lt 90; $i++) {
        Start-Sleep -Seconds 2
        $corpo = curl.exe -s -w "%{http_code}" "$base/trabalhos/$idExec"
        if ($corpo -match "200$") { break }
    }
    $temQ2 = $corpo -match '"plsda_q2":\s*(-?[0-9.eE+-]+)'
    $q2 = if ($temQ2) { $Matches[1] } else { "ausente" }
    $temP = $corpo -match '"plsda_permutation_p":\s*(-?[0-9.eE+-]+)'
    $pv = if ($temP) { $Matches[1] } else { "ausente" }
    $temProc = $corpo -match '"plsda_groups"'
    Passo "o PLS-DA foi CALCULADO no congelado" ($temQ2 -and $temP) "q2=$q2 p=$pv"
    Passo "e traz a PROCEDENCIA (grupos, dobras, semente)" $temProc
}

# --- 4. O MULTIVERSO DISPARA E CANCELA ---------------------------------------
# Grade com `serrf` de proposito: e a unica que demora o bastante para a parada ser medivel.
$mv = curl.exe -s -X POST "$base/multiversos" `
    -F "measurements=@$Ent\measurements_neg_c18.csv" `
    -F "sample_list=@$Ent\Sample_List_neg_c18.csv" `
    -F "mode=neg_c18" -F "corrections=<$Json" `
    -F "group_a=Asymptomatic" -F "group_b=ATLL"
$idMv = if ($mv -match '"identificador":"([0-9a-f]+)"') { $Matches[1] } else { "" }
$total = if ($mv -match '"total":([0-9]+)') { $Matches[1] } else { "?" }
Passo "POST /multiversos disparou" ($idMv -ne "") "id=$idMv total=$total celulas"

if ($idMv -ne "") {
    Start-Sleep -Seconds 5
    $antes = curl.exe -s "$base/multiversos/$idMv"
    $estadoAntes = if ($antes -match '"estado":"(\w+)"') { $Matches[1] } else { "?" }

    $t0 = Get-Date
    $canc = curl.exe -s -w "|%{http_code}" -X POST "$base/multiversos/$idMv/cancelar"
    $codigo = ($canc -split "\|")[-1]
    $pediu = $canc -match '"cancelamento_pedido":true'
    Passo "POST .../cancelar devolveu 202 com a bandeira" (($codigo -eq "202") -and $pediu) "http=$codigo"

    $parou = $false
    $estadoFinal = "?"
    for ($i = 0; $i -lt 120; $i++) {
        Start-Sleep -Milliseconds 250
        $ler = curl.exe -s "$base/multiversos/$idMv"
        if ($ler -match '"estado":"(\w+)"') { $estadoFinal = $Matches[1] }
        if ($estadoFinal -eq "cancelado") { $parou = $true; break }
    }
    $decorrido = [math]::Round(((Get-Date) - $t0).TotalSeconds, 2)
    Passo "a execucao PAROU, e o estado e 'cancelado'" $parou "antes=$estadoAntes final=$estadoFinal em ${decorrido}s"

    $res = curl.exe -s -w "|%{http_code}" "$base/multiversos/$idMv/resultado"
    $codRes = ($res -split "\|")[-1]
    $diz = $res -match "CANCELADA a seu pedido"
    $naoMente = -not ($res -match "group_a")
    Passo "o resultado diz CANCELADA e nao mente sobre grupos" (($codRes -eq "409") -and $diz -and $naoMente) "http=$codRes"
}

# --- 6. O HISTORICO: o que sobrevive ao fechamento do programa ---------
#
# **E o unico passo desta sonda que toca o BANCO POR USUARIO**, e por isso e o unico que prova
# que a migracao correu no arranque do congelado. A migracao `a3f7c1b90e42` deu as tabelas doze
# colunas, e nada disso e exercitado pelos passos do congelamento.
#
# A execucao que se procura e a que acabou de ser CANCELADA acima — o desfecho que mais interessa
# ao historico, porque e o que desapareceria sem ele.
if ($idMv) {
    $hist = curl.exe -s -w "|%{http_code}" "$base/multiversos/historico"
    $codHist = ($hist -split "\|")[-1]
    $temNoHistorico = $hist -match [regex]::Escape($idMv)
    Passo "GET /multiversos/historico serve o que ficou" (($codHist -eq "200") -and $temNoHistorico) "http=$codHist contem_o_id=$temNoHistorico"

    $det = curl.exe -s -w "|%{http_code}" "$base/multiversos/historico/$idMv"
    $codDet = ($det -split "\|")[-1]
    # O estado tem de ser o MESMO que a memoria reportou — se divergirem, o gancho de fecho nao
    # correu, ou correu com o estado errado.
    $estadoNoDisco = ""
    if ($det -match '"estado":"(\w+)"') { $estadoNoDisco = $Matches[1] }
    Passo "a conclusao GRAVADA tem o mesmo estado da memoria" (($codDet -eq "200") -and ($estadoNoDisco -eq $estadoFinal)) "http=$codDet disco=$estadoNoDisco memoria=$estadoFinal"

    # **E o banco por usuario existe de facto em disco.** Sem esta linha, um historico servido de
    # uma tabela em memoria passaria nos dois passos acima.
    # O caminho e DERIVADO do executavel que esta sonda lancou, e nao pinado em
    # `$env:LOCALAPPDATA`: o congelado escreve o banco AO LADO DO EXECUTAVEL, e pinado no lugar
    # antigo o passo mediria um arquivo que o binario ja nao escreve, e um `refino.db` velho
    # deixado la o faria PASSAR.
    $arquivoDb = Join-Path (Split-Path -Parent $Exe) "refino.db"
    $tamDb = if (Test-Path $arquivoDb) { (Get-Item $arquivoDb).Length } else { 0 }
    Passo "o banco vive AO LADO DO EXECUTAVEL e nao esta vazio" ($tamDb -gt 0) "$arquivoDb ($tamDb bytes)"

    # A direccao NEGATIVA do mesmo passo, e ela e o que separa *mudou de sitio* de *escreve
    # nos dois*. Um `refino.db` antigo em `%LOCALAPPDATA%` nao reprova — o que reprova e ele
    # ser mais recente que o arranque desta sonda, porque isso e escrita DESTA execucao.
    $antigo = Join-Path $env:LOCALAPPDATA "refino\refino.db"
    $tocouNoAntigo = (Test-Path $antigo) -and ((Get-Item $antigo).LastWriteTime -gt $inicioDaSonda)
    Passo "o congelado NAO escreve mais no lugar antigo" (-not $tocouNoAntigo) "$antigo tocado_agora=$tocouNoAntigo"
}

# --- 7. A DOCUMENTACAO INTERATIVA NAO E SERVIDA pelo congelado ------------
#
# As QUATRO rotas, e nao as duas que se escreve de memoria. Este passo e o par causal do oraculo
# 405-contra-404 do passo 1: la mede-se que uma rota de negocio EXISTE, aqui que as de
# documentacao NAO existem — e as duas leituras usam o mesmo instrumento.
$docsVivas = @()
foreach ($rota in @("/docs", "/redoc", "/openapi.json", "/docs/oauth2-redirect")) {
    $cod = curl.exe -s -o NUL -w "%{http_code}" "$base$rota"
    if ($cod -ne "404") { $docsVivas += "$rota=$cod" }
}
Passo "as 4 rotas de documentacao respondem 404" ($docsVivas.Count -eq 0) ($(if ($docsVivas.Count) { $docsVivas -join "," } else { "as quatro em 404" }))

$proc | Stop-Process -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 2
Get-Process refino -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
Write-Host ""
Write-Host "falhas=$script:Falhas"
exit $script:Falhas
