# prova_enriquecimento_congelado.ps1 - a prova do enriquecimento pelo MetaboAnalystR DENTRO da pasta
# congelada.
#
# Corre a carga irma prova_enriquecimento_congelado.py com o interpretador embarcado da propria pasta
# (_internal\pythonw.exe), que importa o refino EMPACOTADO e chama o lancador de producao sobre a
# pasta R e a pasta vias-dados da propria entrega. O que a carga mede esta na docstring dela; este
# roteiro le o JSON que ela grava e reprova item a item.
#
# **DUAS COPIAS, e a segunda e em caminho com ESPACO e ACENTO.** O lancador passa caminhos ao R por
# argumento e por variavel de ambiente, e o R tem as suas proprias regras de caminho no Windows: uma
# pasta que funciona em dist\refino e nada diz sobre a de quem recebe, que pode estar em
# "Area de Trabalho" com acento. A segunda copia e feita por robocopy, medida, e apagada no fim.
#
# O item 6 e o modo externo do enriquecimento, SEM REDE. A carga monta e confere a carga
# do envio (G2), ve a G2 recusar duas plantas, passa uma resposta SINTETICA pelo envio de producao
# com uma conexao FALSA e confere cada p, e ve o leitor restrito recusar uma resposta com funcao.
# Isto afirma que os modulos do modo externo viajaram e funcionam no interpretador embarcado; o
# envio bem-sucedido ao servidor NAO e conferido aqui.
#
# A PRIMEIRA chamada de cada copia e a primeira do processo, e NAO e a frio: na copia A o
# congelamento acabou de ler a arvore R inteira duas vezes (o sha256 de cada um dos 3.505
# arquivos), entao e cache quente; a da copia B, recem-feita, e a mais perto de "pasta nunca
# aberta" (6,91 s medidos), e os 120 s do teto do lancador sao 17x esse numero (a primeira
# chamada de cada processo vai sem teto). Nenhuma das duas e cache frio do SO, que nao foi
# medido. O tempo e IMPRESSO aqui, por copia.
#
# Relato so-ASCII, como todo o ferramental deste repositorio. O acento do caminho e montado por
# codigo de caractere, e nao escrito neste arquivo.
#
# USO:
#   powershell -ExecutionPolicy Bypass -File scripts\prova_enriquecimento_congelado.ps1 [-Pasta dist\refino] [-SemSegundaCopia]

[CmdletBinding()]
param(
    [string]$Pasta = "dist\refino",
    [switch]$SemSegundaCopia
)

$ErrorActionPreference = "Stop"
$Raiz = Split-Path -Parent $PSScriptRoot
$Pasta = (Resolve-Path $Pasta).Path
$Carga = Join-Path $PSScriptRoot "prova_enriquecimento_congelado.py"
$script:Falhas = 0

function Afirma([string]$texto, [bool]$ok, [string]$detalhe) {
    if ($ok) { Write-Host ("OK       " + $texto + " - " + $detalhe) }
    else { Write-Host ("REPROVA  " + $texto + " - " + $detalhe); $script:Falhas += 1 }
}

function ProvaNaPasta([string]$alvo, [string]$rotulo) {
    Write-Host ""
    Write-Host ("=== " + $rotulo + ": " + $alvo)
    $interpretador = Join-Path $alvo "_internal\pythonw.exe"
    if (-not (Test-Path -LiteralPath $interpretador)) { Afirma "ha interpretador embarcado" $false $interpretador; return }

    # O roteiro da pasta tem de ser o do repositorio, byte a byte.
    $fonte = (Get-FileHash -Algorithm SHA256 (Join-Path $Raiz "r_service\vias_entrypoint.R")).Hash
    $copia = Join-Path $alvo "R\vias_entrypoint.R"
    $hashCopia = if (Test-Path -LiteralPath $copia) { (Get-FileHash -Algorithm SHA256 -LiteralPath $copia).Hash } else { "ausente" }
    Afirma "o roteiro da pasta e o do repositorio" ($fonte -eq $hashCopia) ("sha256 " + $hashCopia.Substring(0, [Math]::Min(16, $hashCopia.Length)))

    $saida = Join-Path $env:TEMP ("prova-enriquecimento-" + [guid]::NewGuid().ToString("N") + ".json")
    $p = Start-Process -FilePath $interpretador -ArgumentList @(('"' + $Carga + '"'), ('"' + $alvo + '"'), ('"' + $saida + '"')) -Wait -PassThru
    if (-not (Test-Path $saida)) { Afirma "a carga escreveu o resultado" $false ("saiu com " + $p.ExitCode); return }
    $r = Get-Content $saida -Raw -Encoding UTF8 | ConvertFrom-Json
    Remove-Item $saida -Force

    if ($null -ne $r.erro) { Afirma "a carga correu sem erro" $false $r.erro }
    Afirma "o motor resolve dentro da pasta" ([bool]$r.r_home_dentro) $r.r_home
    Afirma "as bibliotecas fixadas conferem com o manifesto" ($r.bibliotecas -eq "ok") $r.bibliotecas
    # Limite ABERTO desta afirmacao e da de "nenhuma DLL do R vem de fora", tres linhas abaixo:
    # "foi herdado" le so o retorno de SetDllDirectoryW no processo Python, e nao mede a heranca
    # no Rscript filho; "nenhuma DLL do R vem de fora" tem por oraculo getLoadedDLLs(), que nao
    # lista dependencia que o Windows resolve pelo nome. O desfecho "fora" nunca foi visto.
    Afirma "o diretorio de DLL do PyInstaller foi herdado (a planta aterrou)" ([bool]$r.set_dll_directory) "SetDllDirectoryW(_internal)"
    Afirma "mapear responde (primeira chamada do processo)" ($r.mapear.desfecho -eq "ok") ("" + $r.mapear.desfecho + " em " + $r.mapear.segundos + " s")
    Afirma "a ORA local calcula e cada p confere" ($r.ora.desfecho -eq "ok" -and $r.ora.vias -gt 0) ("" + $r.ora.vias + " vias em " + $r.ora.segundos + " s")
    Afirma "nenhuma DLL do R vem de fora da pasta R" ($r.dlls.desfecho -eq "ok" -and $r.dlls.n -gt 0 -and @($r.dlls.fora_do_R).Count -eq 0) ("" + $r.dlls.n + " DLLs; fora: " + (@($r.dlls.fora_do_R) -join ", "))
    Afirma "cancelar encerra a corrida" ($r.cancelar.desfecho -eq "cancelado") $r.cancelar.desfecho
    # refino-vias-* E Rtmp* no TEMP, e nao so o prefixo do lancador: o R cria o seu proprio Rtmp*.
    # O desfecho de reprovacao da diferenca e visto em processo por
    # tests\test_prova_enriquecimento_congelado.py.
    Afirma "nao sobra diretorio temporario da corrida (refino-vias-* nem Rtmp* no TEMP)" (@($r.cancelar.sobrou_temporario).Count -eq 0) (@($r.cancelar.sobrou_temporario) -join ", ")
    $restantes = @(Get-Process -Name "Rscript", "Rterm" -ErrorAction SilentlyContinue | Where-Object { $_.Path -and $_.Path.StartsWith($alvo) })
    Afirma "nenhum processo do R da pasta sobrou" ($restantes.Count -eq 0) ("" + $restantes.Count + " processo(s)")

    # 6. O modo externo SEM REDE, medido por prova_externa numa chave propria do JSON. Nada
    # sai: o transporte e uma conexao falsa, e a resolucao de nome e a conexao de socket ficam
    # bloqueadas e contadas durante o item. Sem a chave, cada afirmacao abaixo reprova - nenhuma
    # passa por ausencia. O registro que cada uma le e virado por uma planta em processo em
    # tests\test_prova_enriquecimento_congelado.py, que confere tambem que toda chave lida aqui
    # existe no registro da carga. Este bloco foi corrido no PowerShell 5.1 contra o JSON de
    # sucesso (0 reprovacoes), contra o JSON sem a chave externo (as seis reprovam) e contra nove
    # JSONs com uma planta cada (cada um reprovou so a sua afirmacao).
    $x = $r.externo
    if ($null -ne $x.erro) { Afirma "o modo externo correu sem erro" $false $x.erro }
    Afirma "externo: a G2 aprova a carga dos nomes publicos da prova" ([bool]$x.carga.aprovada -and $x.carga.nomes -gt 0 -and $x.carga.nomes -eq $x.carga.nomes_esperados) ("" + $x.carga.nomes + " nomes em " + $x.carga.bytes + " bytes " + $x.carga.motivo)
    $plantas = @($x.g2_plantas)
    $plantasOk = ($null -ne $x.g2_plantas) -and ($plantas.Count -gt 0)
    $motivos = @()
    foreach ($planta in $plantas) {
        if (-not ([bool]$planta.aterrou -and [bool]$planta.recusada -and [bool]$planta.pelo_motivo_certo)) { $plantasOk = $false }
        $motivos += ("" + $planta.planta + ": " + $planta.motivo)
    }
    Afirma "externo: a G2 recusa cada carga plantada, pelo motivo dela" $plantasOk ("" + $plantas.Count + " plantas; " + ($motivos -join " | "))
    Afirma "externo: o envio leva exactamente a carga conferida ao destino constante (conexao falsa)" ([bool]$x.resposta.leva_a_carga -and [bool]$x.resposta.destino_constante) $x.resposta.destino
    Afirma "externo: a resposta sintetica e lida pelo leitor restrito e cada p confere" ($x.resposta.desfecho -eq "ok" -and [bool]$x.resposta.ora_confere -and [bool]$x.resposta.vias_conferem) ("" + $x.resposta.desfecho + "; " + @($x.resposta.vias).Count + " vias; universo " + $x.resposta.universo + "; " + $x.resposta.motivo + $x.resposta.motivo_da_ora)
    Afirma "externo: o leitor restrito recusa a resposta com funcao" ($x.com_funcao.desfecho -eq "resposta_malformada" -and [bool]$x.com_funcao.pelo_motivo_certo) ("" + $x.com_funcao.desfecho + ": " + $x.com_funcao.motivo)
    Afirma "externo: nenhuma tentativa de rede durante o item (com a guarda posta)" ([bool]$x.rede.guarda_posta -and @($x.rede.tentativas).Count -eq 0) ("tentativas: " + (@($x.rede.tentativas) -join ", "))

    Write-Host ("tempos: mapear na primeira chamada do processo " + $r.mapear.segundos + " s; ORA " + $r.ora.segundos + " s")
}

ProvaNaPasta $Pasta "copia A (a pasta congelada)"

if (-not $SemSegundaCopia) {
    $acento = [string][char]0x00E1
    $base = Join-Path $env:TEMP ("refino prova vias " + $acento + " " + [guid]::NewGuid().ToString("N").Substring(0, 8))
    $PastaB = Join-Path $base "refino"
    if ($PastaB.StartsWith($Pasta, [StringComparison]::OrdinalIgnoreCase)) { Write-Host "a segunda copia cairia dentro da primeira"; exit 1 }
    $inicio = Get-Date
    & robocopy.exe $Pasta $PastaB /E /NFL /NDL /NJH /NJS /NP /R:1 /W:1 | Out-Null
    $codigoRobocopy = $LASTEXITCODE
    $segundos = [math]::Round(((Get-Date) - $inicio).TotalSeconds, 1)
    # robocopy: 0 a 7 sao sucesso; 8 ou mais e falha.
    Afirma "a segunda copia foi feita" (($codigoRobocopy -lt 8) -and (Test-Path -LiteralPath (Join-Path $PastaB "R\bin\x64\Rscript.exe"))) ("robocopy " + $codigoRobocopy + " em " + $segundos + " s")
    try {
        ProvaNaPasta $PastaB "copia B (espaco e acento no caminho)"
    } finally {
        Remove-Item -LiteralPath $base -Recurse -Force -ErrorAction SilentlyContinue
    }
}

Write-Host ("")
if ($script:Falhas -gt 0) { Write-Host ("REPROVADA: " + $script:Falhas + " reprovacao(oes)"); exit 1 }
Write-Host "APROVADA"
exit 0
