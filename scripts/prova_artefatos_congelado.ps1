# prova_artefatos_congelado.ps1 — as tres rotas de artefato dentro do executavel congelado,
# em QUATRO disparos: o quarto forca o corte da exportacao e reclama o registro dele
#
# ## Por que esta sonda existe
#
# Sem ela, `processar` e `exportar` nunca correriam dentro do binario: dois dos tres botoes da
# aba de onde o usuario tira o arquivo. `mede_serrf_congelado.ps1` exercita `corrigir` e mais
# nada.
#
# E ha uma segunda razao: as tres rotas gravam em runs o tempo que gastaram, e nenhuma outra
# sonda exercita esse codigo no mundo congelado. Um mecanismo de melhor esforco e exactamente o
# que falha em silencio: `_grava_tempo_do_artefato`
# engole a falha de proposito, para nao derrubar a resposta. Entao um erro de caminho de banco,
# de permissao ou de esquema dentro do congelado produziria **artefato entregue e tempo nao
# gravado**, com codigo 200 e sem sintoma nenhum.
#
# > **O que esta sonda mede que nenhuma outra mede: a linha de runs EXISTE depois de a rota
# > responder 200.** Contar as linhas antes e depois, e exigir que a diferenca seja exactamente
# > o numero de rotas disparadas.
#
# ## O oraculo, e por que e a DIFERENCA e nao o total
#
# A pasta congelada ja pode ter registro de execucoes anteriores — o cenario do duplo clique
# abre a janela, e a sonda do multiverso corre uma grade. Um oraculo de total exigiria conhecer
# o passado; a diferenca nao exige nada: deriva em vez de pinar.
#
# ## Uso
#
#     powershell -ExecutionPolicy Bypass -File scripts\prova_artefatos_congelado.ps1
#
# Sai 0 se os quatro disparos entregarem artefato, os quatro deixarem tempo em disco, e o
# relatorio do corte responder com lista NAO vazia.

param(
    [string]$Modo = "neg_c18",
    [int]$TetoMin = 20,
    # O teto que FORCA o corte, e ele nao e o realista: essa e a questao.
    #
    # Com `max_features=5000` no `neg_c18`, que tem 2.184 features na ingestao, nada se corta: a
    # lista de removidas sai vazia, e a metade do caminho que classifica por causa e escreve o
    # relatorio nunca correria dentro do binario. O valor natural e o que executa menos do
    # caminho.
    #
    # 1000 e nao um numero por modo, porque os quatro excedem-no na ingestao: 6.035 no
    # `pos_hilic`, 2.572 no `pos_c18`, 2.039 no `neg_hilic` e 2.184 no `neg_c18`.
    # Um teto por modo seria uma segunda tabela a envelhecer ao lado da que ja existe.
    #
    # **O disparo realista FICA**, e os dois correm: sem ele a sonda deixaria de medir o caminho
    # que o usuario de facto percorre, e trocar uma cobertura pela outra e o defeito que esta
    # extensao existe para nao ter.
    [int]$CorteEm = 1000,
    # Corre so `ContaRuns` e sai. Existe para o desfecho NAO MEDIDO ser exercitavel sem o
    # executavel e sem os minutos das tres rotas — um desfecho que ninguem consegue ver nasce
    # inerte, e inerte e indistinguivel de ausente.
    [switch]$SoContaRuns
)

$ErrorActionPreference = "Stop"
$Raiz = Split-Path -Parent $PSScriptRoot
$Dist = Join-Path $Raiz "dist\refino"
$Exe = Join-Path $Dist "refino.exe"
$Banco = Join-Path $Dist "refino.db"

$script:Falhas = 0
function Passo([string]$nome, [bool]$ok, [string]$detalhe) {
    $marca = if ($ok) { "ok  " } else { "FALHA"; }
    if (-not $ok) { $script:Falhas++ }
    Write-Host ("  {0}  {1}" -f $marca, $nome)
    if ($detalhe) { Write-Host ("        {0}" -f $detalhe) }
}

# Conta as linhas de runs LENDO O BANCO DA PASTA DO EXECUTAVEL — derivado, nao pinado.
# A leitura e por `sqlite3` do Python do clone: e o mesmo arquivo, e nao ha instrumento mais
# barato. Se o banco ainda nao existir, o total e zero e a diferenca continua a valer.
function ContaRuns {
    if (-not (Test-Path $Banco)) { return 0 }
    # O interpretador vive na raiz do CLONE, e `$Raiz` pode ser uma WORKTREE. Corrido de
    # uma worktree, `$Raiz\.venv` nao existe: o passo devolveria `-1`, entraria no
    # ramo NAO MEDIDO, e o veredito *"1 passo(s) reprovaram"* acusaria o PRODUTO, quando a
    # causa estaria no instrumento.
    #
    # `git rev-parse --git-common-dir` devolve o `.git` do clone PRINCIPAL mesmo de dentro de uma
    # worktree — e a unica pergunta que responde *"onde vive o `.venv`?"* sem o pinar. A ordem e
    # deliberada: tenta-se primeiro `$Raiz`, que e o caso comum e nao paga um `git`; so depois o
    # clone comum. Se nenhum dos dois achar, `-1` continua a ser a resposta honesta.
    $py = Join-Path $Raiz ".venv\Scripts\python.exe"
    if (-not (Test-Path $py)) {
        $comum = (& git -C $Raiz rev-parse --path-format=absolute --git-common-dir 2>$null)
        if ($LASTEXITCODE -eq 0 -and $comum) {
            $py = Join-Path (Split-Path -Parent $comum.Trim()) ".venv\Scripts\python.exe"
        }
    }
    # Sem esta guarda, `$ErrorActionPreference = "Stop"` derruba a sonda inteira no meio e a
    # mensagem fala do executavel em vez do passo — quem a corre conclui que a SONDA quebrou,
    # quando o que falta e o ambiente. `-1` e a resposta honesta, e o passo dela sabe le-la.
    if (-not (Test-Path $py)) { return -1 }
    $saida = & $py -c @"
import sqlite3, sys
try:
    con = sqlite3.connect(r'$Banco')
    print(con.execute('select count(*) from runs').fetchone()[0])
    con.close()
except Exception:
    print(-1)
"@
    # Vazio nao e zero. Uma saida vazia — o Python morreu, a console engoliu — daria `0` num
    # `[int]` e entraria na subtracao como contagem. O valor tem de significar uma coisa so,
    # e `-1` significa *nao medido*.
    $limpo = ("" + $saida).Trim()
    if ($limpo -notmatch '^-?\d+$') { return -1 }
    return [int]$limpo
}

if ($SoContaRuns) {
    $n = ContaRuns
    Write-Host ""
    Write-Host "  ContaRuns sobre $Banco  ->  $n"
    Write-Host ("  leitura: " + $(if ($n -ge 0) { "MEDIDA" } else { "NAO MEDIDA" }))
    Write-Host ""
    exit ([int]($n -lt 0))
}

if (-not (Test-Path $Exe)) { Write-Host "  nao existe: $Exe"; exit 1 }

# `serrf\data\input` e nao `data\reference\input`. Os dois sao o mesmo no Linux, onde
# `data/reference/input` e um symlink; num clone Windows o Git materializa-o como um ARQUIVO de
# 22 bytes com o texto do destino la dentro (medido), e o caminho logico nao resolve.
$med = Join-Path $Raiz "serrf\data\input\measurements_$Modo.csv"
$lst = Join-Path $Raiz "serrf\data\input\Sample_List_$Modo.csv"
foreach ($f in @($med, $lst)) {
    if (-not (Test-Path $f)) { Write-Host "  nao existe: $f"; exit 1 }
}

Write-Host ""
Write-Host "PROVA DAS ROTAS DE ARTEFATO NO CONGELADO - modo $Modo, corte em $CorteEm"
Write-Host ""

$runsAntes = ContaRuns
Write-Host "  linhas de runs antes: $runsAntes"

# **A porta e a que RESPONDE `/health`, nunca a primeira da lista.** A tela de arranque do
# PyInstaller abre uma porta de escuta no loopback ANTES da nossa, ela aceita a ligacao e nunca
# responde HTTP, e o `curl` fica pendurado. O motivo e a medicao estao em
# `scripts/porta_que_serve.ps1`.
. (Join-Path $PSScriptRoot "porta_que_serve.ps1")

$proc = Start-Process -FilePath $Exe -PassThru
$base = PortaQueServe -Processo $proc -TetoSeg 180
Passo "o binario arrancou" (-not $proc.HasExited) "PID $($proc.Id)"
Passo "o servidor esta a escutar em 127.0.0.1 e responde /health" ([bool]$base) "base: $base"
if (-not $base) { $proc | Stop-Process -Force -ErrorAction SilentlyContinue; exit 1 }

# `-F "campo=@arquivo"` e nao `=<arquivo`: o `<` envia o conteudo como campo comum e estoura o
# `max_part_size` de 1024 KB. E a armadilha que `mede_serrf_congelado.ps1` registou ao nascer.
$comuns = @("-F", "measurements=@$med", "-F", "sample_list=@$lst", "-F", "mode=$Modo")

# `rotulo` e separado de `nome` porque `exportar` aparece DUAS vezes: uma com o
# teto realista, que no `neg_c18` nao corta, e outra com o teto que forca o corte. Dois passos com
# o mesmo texto seriam indistinguiveis na saida, e o veredito nomearia o errado.
#
# `corta` marca o unico disparo de que se reclama o relatorio. Ele e `$true` numa entrada so, e a
# assercao daquele passo exige lista NAO VAZIA — sem isso, um relatorio vazio passaria por
# vacuidade e a sonda diria que o registro funciona sobre um corte que nao houve.
$rotas = @(
    @{ nome = "corrigir";  rotulo = "corrigir";  corta = $false; extra = @("-F", "method=none", "-F", "n_batches=1") },
    @{ nome = "processar"; rotulo = "processar"; corta = $false; extra = @("-F", "method=none", "-F", "imputation=one_fifth_min", "-F", "scaling=none") },
    @{ nome = "exportar";  rotulo = "exportar (teto realista)"; corta = $false; extra = @("-F", "max_features=5000") },
    @{ nome = "exportar";  rotulo = "exportar (com corte)";     corta = $true;  extra = @("-F", "max_features=$CorteEm") }
)

$entregues = 0
foreach ($r in $rotas) {
    $nome = $r.nome
    $rotulo = $r.rotulo
    $t0 = Get-Date
    # UM array so, com splatting UMA vez. `@comuns @($r.extra)` NAO funciona: `@(...)` em posicao
    # de argumento e criacao de array, nao splatting — os tres pares de `-F` extra chegariam ao
    # `curl` como UM argumento, e o disparo falharia com uma mensagem que nao aponta para aqui.
    # `$args` NAO serve: e variavel AUTOMATICA do PowerShell.
    $argumentos = @("-s", "-X", "POST", "$base/matrizes/$nome") + $comuns + $r.extra
    # `-join` porque o `curl` pode devolver varias linhas, e ai `$disparo` e um array: `.Substring`
    # levantaria, e a mensagem de erro falaria do metodo em vez de falar da rota.
    $disparo = (& curl.exe @argumentos) -join ""
    $idJob = ($disparo | ConvertFrom-Json -ErrorAction SilentlyContinue).identificador
    if (-not $idJob) {
        $recorte = if ($disparo.Length -gt 160) { $disparo.Substring(0, 160) } else { $disparo }
        Passo "$rotulo aceitou o disparo" $false "resposta: $recorte"
        continue
    }

    $estado = "executando"
    $codigo = "0"
    $limite = (Get-Date).AddMinutes($TetoMin)
    while ((Get-Date) -lt $limite) {
        Start-Sleep -Seconds 3
        $codigo = & curl.exe -s -o NUL -w "%{http_code}" "$base/trabalhos/$idJob"
        if ($codigo -ne "202") { $estado = "terminou"; break }
    }
    $seg = [Math]::Round(((Get-Date) - $t0).TotalSeconds, 1)

    # O artefato tem de ter CORPO. `200` com zero bytes seria o desfecho de sucesso sobre nada.
    $tmp = Join-Path $env:TEMP "artefato_$($idJob).bin"
    if ($codigo -eq "200") { & curl.exe -s -o $tmp "$base/trabalhos/$idJob" | Out-Null }
    $bytes = if (Test-Path $tmp) { (Get-Item $tmp).Length } else { 0 }
    Remove-Item $tmp -Force -ErrorAction SilentlyContinue

    $ok = ($codigo -eq "200") -and ($bytes -gt 0)
    if ($ok) { $entregues++ }
    Passo "$rotulo entrega artefato com corpo" $ok "http=$codigo bytes=$bytes parede=${seg}s estado=$estado"

    # --- A PERNA DO REGISTRO DO CORTE, e ela so existe para o disparo que corta --------------
    #
    # `GET /trabalhos/{id}/relatorio` alimenta o arquivo irmao que a janela escreve ao lado do
    # CSV: o registro de o que saiu e por qual causa. Um binario em que ela nao responda entrega
    # o arquivo cortado e perde essa metade.
    #
    # **A assercao e sobre a LISTA e nao sobre o codigo.** Um `200` com `removidas` vazia e
    # exactamente o desfecho que o teto realista produz neste modo, e aceita-lo aqui faria o
    # passo passar por vacuidade sobre o disparo que devia cortar.
    if ($r.corta -and $ok) {
        # O corpo vai para um arquivo e o codigo para OUTRO. `congela_windows.ps1` explica por
        # extenso: `-w "<crase>n%{http_code}"` nao funciona porque a crase-n e consumida pelo
        # PowerShell antes de chegar ao `curl`, e o corpo e o codigo ficam no mesmo fluxo sem
        # separador.
        $relArq = Join-Path $env:TEMP "relatorio_$idJob.json"
        $relCod = Join-Path $env:TEMP "relatorio_$idJob.codigo"
        & curl.exe -s -o $relArq -w "%{http_code}" "$base/trabalhos/$idJob/relatorio" > $relCod
        $relCodigo = if (Test-Path $relCod) { (Get-Content $relCod -Raw).Trim() } else { "" }
        $relCorpo = if (Test-Path $relArq) {
            (Get-Content $relArq -Raw) | ConvertFrom-Json -ErrorAction SilentlyContinue
        } else { $null }
        # `-1` e nao `0` para o nao medido, pelo mesmo argumento de `ContaRuns` acima: `0`
        # e uma contagem plausivel e entraria na comparacao como medida.
        $mantidas = if ($null -ne $relCorpo) { [int]$relCorpo.mantidas } else { -1 }
        $removidas = if ($null -ne $relCorpo) { @($relCorpo.removidas).Count } else { -1 }
        $causas = if ($removidas -gt 0) {
            (@($relCorpo.removidas) | ForEach-Object { $_.causa } | Sort-Object -Unique) -join ","
        } else { "" }
        Remove-Item $relArq, $relCod -Force -ErrorAction SilentlyContinue
        Passo "o relatorio do corte responde e a lista NAO e vazia" `
            (($relCodigo -eq "200") -and ($mantidas -eq $CorteEm) -and ($removidas -gt 0)) `
            "http=$relCodigo mantidas=$mantidas (esperado $CorteEm) removidas=$removidas causas=[$causas]"
    }
}

# --- O QUE NENHUMA OUTRA SONDA MEDE: o tempo ficou GRAVADO ---------------------
#
# `_grava_tempo_do_artefato` e MELHOR ESFORCO — engole a falha para nao derrubar a resposta. Sem
# este passo, um erro de banco dentro do congelado daria `200` nas tres rotas e zero linhas em
# runs, e o desfecho seria indistinguivel de sucesso. E o motivo inteiro de gravar o tempo e
# comparar hardware, entao a gravacao E o produto aqui, e nao um efeito colateral.
#
# O oraculo e a DIFERENCA e nao o total: a pasta pode ter registro de execucoes anteriores.
$runsDepois = ContaRuns
$mediu = ($runsAntes -ge 0) -and ($runsDepois -ge 0)
$novas = if ($mediu) { $runsDepois - $runsAntes } else { -1 }
# Tres desfechos e nao dois: mediu-e-bate, mediu-e-nao-bate, e NAO MEDIU. Sem o
# terceiro, um `.venv` ausente daria `-1 - -1 = 0` e o passo passaria por igualdade com zero
# entregues — o desfecho de sucesso sobre uma medicao que nao aconteceu.
Passo "as $entregues rotas que entregaram deixaram tempo em runs" ($mediu -and ($novas -eq $entregues)) `
    $(if ($mediu) { "antes=$runsAntes depois=$runsDepois novas=$novas entregues=$entregues" }
      else { "NAO MEDIDO: falta .venv\Scripts\python.exe ou a consulta a runs falhou" })

# A direccao que impede o passo acima de passar por vacuidade: se NENHUMA rota entregou, a
# igualdade `0 -eq 0` seria verdadeira e o passo verde diria que a gravacao funciona.
Passo "e alguma rota entregou de facto" ($entregues -gt 0) "entregues=$entregues de $($rotas.Count)"

$proc | Stop-Process -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 2

Write-Host ""
if ($script:Falhas -gt 0) {
    Write-Host "  VEREDITO: $($script:Falhas) passo(s) reprovaram"
} else {
    Write-Host "  VEREDITO: os $($rotas.Count) disparos correm no congelado, todos gravam o tempo, e o corte fica registrado"
}
Write-Host ""
exit ([int]($script:Falhas -gt 0))
