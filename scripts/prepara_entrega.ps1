# prepara_entrega.ps1 — a ultima coisa que corre antes de o zip sair
#
# ## A ORDEM E O CONTEUDO, e este roteiro existe por causa dela
#
# O fluxo de entrega tem CINCO etapas numeradas (a 3 em quatro partes), e QUATRO roteiros CORREM o
# executavel: as etapas 2, 3, 3b e 3c. A etapa 5 nao tem roteiro nenhum, e e a unica que conduz a
# INTERFACE:
#
#     1. scripts\congela_windows.ps1            constroi e mede o binario
#     2. scripts\prova_multiverso_congelado.ps1 corre uma grade inteira
#     3. scripts\prova_artefatos_congelado.ps1  corre as tres rotas de artefato
#    3b. scripts\prova_paralelismo_congelado.ps1  a caixa «usar varios nucleos» muda o TEMPO e
#                                                nao o RESULTADO - ~10 min
#    3c. scripts\prova_figuras_congelado.ps1    as figuras registadas desenham DENTRO da pasta
#                                                entregue, e as duas copias em caminhos diferentes
#                                                nao reconstroem o cache de fontes uma da outra
#    3d. scripts\prova_enriquecimento_congelado.ps1  o R embarcado calcula o enriquecimento DENTRO
#                                                da pasta, nas duas copias, e cancelar nao deixa
#                                                processo nem temporario
#     4. scripts\prepara_entrega.ps1            ESTE
#     5. a passagem HUMANA pela janela          sem roteiro: alguem abre o refino.exe e percorre o
#                                                artefato montado, e o que ela produz e uma
#                                                lista de achados
#
# A etapa 5 esta nesta lista porque um passo humano que so vive na prosa de outro arquivo e o
# passo que fica por fazer quando a sessao acaba. A 3d nao lanca o `refino.exe`: ela corre o
# interpretador embarcado da pasta (`_internal\pythonw.exe`) contra as pastas `R` e `vias-dados`
# da propria entrega, e por isso nao conta entre os roteiros que CORREM o executavel.
#
# Com o banco ao lado do executavel, tudo o que as tres primeiras fizeram fica gravado em
# `dist\refino\refino.db`. Zipar nesse estado entrega a quem recebe um registro que ja tem estudos,
# execucoes e uma grade de multiverso — nenhum deles dele.
#
# > **A primeira versao disto era um cenario 7 dentro do `congela_windows.ps1`, e a colocacao
# > estava ERRADA:** as duas sondas correm DEPOIS daquele roteiro, e repovoariam o banco que ele
# > acabara de limpar. O mecanismo estava na camada onde eu tinha alavanca, e nao na superficie
# > que importa — que e o instante imediatamente anterior ao zip.
#
# ## As tres consequencias de nao limpar, e a terceira e a que custa
#
#   1. a tela de historico abre com linhas que ele nao reconhece;
#   2. o `LEIA-ME` diz *"refino.db: tudo o que VOCE executou"*, e a frase fica falsa;
#   3. **os tempos por etapa ficam misturados** — e o motivo inteiro de os gravar e comparar
#      hardware. Um registro com medidas de duas maquinas, sem coluna que as separe, nao responde
#      a pergunta que o motivou.
#
# ## Uso
#
#     powershell -ExecutionPolicy Bypass -File scripts\prepara_entrega.ps1
#     powershell -ExecutionPolicy Bypass -File scripts\prepara_entrega.ps1 -Zipar
#
# Sai 0 se a pasta ficou pronta. Com `-Zipar`, escreve `dist\refino.zip`.

param(
    [switch]$Zipar
)

$ErrorActionPreference = "Stop"
$Raiz = Split-Path -Parent $PSScriptRoot
$Dist = Join-Path $Raiz "dist\refino"

$script:Falhas = 0
function Passo([string]$nome, [bool]$ok, [string]$detalhe) {
    $marca = if ($ok) { "ok   " } else { "FALHA" }
    if (-not $ok) { $script:Falhas++ }
    Write-Host ("  {0}  {1}" -f $marca, $nome)
    if ($detalhe) { Write-Host ("         {0}" -f $detalhe) }
}

Write-Host ""
Write-Host "PREPARA A PASTA PARA ENTREGA - $Dist"
Write-Host ""

if (-not (Test-Path $Dist)) {
    Write-Host "  nao existe: $Dist. Corra scripts\congela_windows.ps1 primeiro."
    exit 1
}

# --- 1. O ARTEFATO E DESTA ARVORE -------------------------------------------
#
# O que este passo apanha e mudo. Sem ele, este roteiro citaria `VERSAO.txt` em dois lugares (na
# lista do que tem de ficar na pasta, e num minimo de 20 bytes) e NUNCA leria o commit que ele
# carrega. Ja se mediu um `dist\refino\VERSAO.txt` 25 commits atras do clone, com CINCO arquivos
# de `src/` mudados (entre eles uma migracao de banco nova e o texto que o usuario le dentro do
# programa), e os outros passos todos verdes.
#
# **O `congela_windows.ps1` ja tem a guarda simetrica** -- ele confere `HEAD` contra `origin/main`
# e sai 2 se o clone estiver atras, porque congelar de arvore velha embala trabalho que nao esta
# la. Faltava a outra ponta: o congelado pode ser velho mesmo com o clone em dia, e e este roteiro
# -- "a ultima coisa que corre antes de o zip sair" -- o unico sitio onde isso ainda se ve.
#
# **A comparacao e contra `HEAD` e nao contra `origin/main`**, e a diferenca importa: o que se
# entrega tem de ser o que ESTA neste clone, que e o que quem prepara acabou de verificar. Se o
# clone estiver atras do remoto, quem acusa e o roteiro de congelamento, na etapa 1.
$carimbo = Join-Path $Dist "VERSAO.txt"
$commitDoArtefato = ""
if (Test-Path $carimbo) {
    $linha = Select-String -LiteralPath $carimbo -Pattern "^commit\s*:\s*([0-9a-f]{40})" |
        Select-Object -First 1
    if ($linha) { $commitDoArtefato = $linha.Matches[0].Groups[1].Value }
}

$anteriorPref = $ErrorActionPreference
$ErrorActionPreference = "Continue"
$commitDoClone = (git rev-parse HEAD 2>$null)
$ErrorActionPreference = $anteriorPref
if ($commitDoClone) { $commitDoClone = $commitDoClone.Trim() } else { $commitDoClone = "" }

if (-not $commitDoArtefato) {
    Passo "o artefato declara o commit de que foi construido" $false `
        "VERSAO.txt nao traz uma linha 'commit    : <sha de 40>'. Sem ela nao se sabe o que se envia"
    exit 1
} elseif (-not $commitDoClone) {
    # Sair 0 por nao haver git seria a guarda a absorver a propria pre-condicao, em silencio.
    # Sem `git` nao se pode comparar, e isso e uma FALHA e nao um ok.
    Passo "o artefato e desta arvore" $false `
        "nao consegui ler `git rev-parse HEAD`. Sem o commit do clone nao ha comparacao"
    exit 1
} elseif ($commitDoArtefato -ne $commitDoClone) {
    $atras = (git rev-list --count "$commitDoArtefato..HEAD" 2>$null)
    if ($atras) { $atras = $atras.Trim() } else { $atras = "?" }
    $tocados = @(git diff --name-only "$commitDoArtefato..HEAD" -- src 2>$null)
    $detalhe = "artefato em $($commitDoArtefato.Substring(0,7)), clone em " +
        "$($commitDoClone.Substring(0,7)) - $atras commit(s) a frente, " +
        "$($tocados.Count) arquivo(s) de src/ mudados"
    Passo "o artefato e desta arvore" $false $detalhe
    Write-Host ""
    Write-Host "  O zip levaria um binario que nao tem o trabalho deste clone."
    Write-Host "  Recongele:  powershell -ExecutionPolicy Bypass -File scripts\congela_windows.ps1"
    # **SAI AQUI, e a saida e o conteudo.** Os passos seguintes APAGAM `refino.db` da
    # pasta. Limpar o registro de um artefato que nao vai ser enviado destroi estado sem
    # ganho nenhum, e deixa a pasta num meio-termo que ninguem pediu.
    exit 1
} else {
    Passo "o artefato e desta arvore" $true "commit $($commitDoClone.Substring(0,7))"
}

# --- 2. NENHUM PROCESSO DO PROGRAMA PODE ESTAR VIVO ---------------------------
#
# Apagar `refino.db` com o programa aberto deixa o SQLite a escrever num arquivo que ja nao tem
# nome, e o `-journal` fica orfao. E o mesmo argumento que o `LEIA-ME` da a quem recebe para fechar
# antes de zipar, aplicado a quem prepara.
$vivos = @(Get-Process -Name "refino" -ErrorAction SilentlyContinue)
Passo "nenhum processo refino esta vivo" ($vivos.Count -eq 0) `
    $(if ($vivos.Count) { "PIDs: " + (($vivos | ForEach-Object { $_.Id }) -join ",") } else { "nenhum" })
if ($vivos.Count -gt 0) {
    Write-Host ""
    Write-Host "  Feche o programa antes de preparar a entrega."
    exit 1
}

# --- 3. O REGISTRO DOS NOSSOS TESTES SAI --------------------------------------
$paraTirar = @("refino.db", "refino.db-journal", "refino.db-wal", "refino.db-shm", "refino.log")
$apagados = @()
foreach ($nome in $paraTirar) {
    $alvo = Join-Path $Dist $nome
    if (Test-Path $alvo) {
        $apagados += ("{0} ({1} bytes)" -f $nome, (Get-Item $alvo).Length)
        Remove-Item $alvo -Force
    }
}
$apagados = @($apagados)

$sobrou = @()
foreach ($nome in $paraTirar) {
    if (Test-Path (Join-Path $Dist $nome)) { $sobrou += $nome }
}
$sobrou = @($sobrou)
Passo "a pasta nao leva o registro dos nossos testes" ($sobrou.Count -eq 0) `
    ("apagados: " + $(if ($apagados.Count) { $apagados -join ", " } else { "nenhum - a pasta ja estava limpa" }))

# --- 4. E A LIMPEZA NAO TOCOU NO QUE VIAJA ------------------------------------
#
# A direccao NEGATIVA, e ela e o que impede este roteiro de virar `rm -rf` da pasta: um passo que
# apagasse a mais reportaria exactamente o mesmo desfecho de sucesso no passo 3.
# A lista e a dos nomes que `congela_windows.ps1` copia para a pasta, e o passo 4b abaixo confere
# que ela continua a cobrir todos. Escrita OLHANDO A PASTA, ela ja deixou de fora documentos que o
# congelamento passou a copiar, e eles chegaram a viajar sem estar aqui ate o 4b os acusar.
# `saidas` sao as tres pastas de CSV prontos, para quem recebe nao ter de rodar os quatro estudos
# na maquina dele. Ela e PASTA e nao arquivo, e por isso o piso de tamanho do passo 5 nao a
# alcanca; quem confere o conteudo dela e o passo 5b, logo abaixo.
$temQueFicar = @("refino.exe", "_internal", "LEIA-ME.txt", "GUIA.html",
                 "ENTENDER.html", "MEDICOES.html", "MULTIVERSO.html", "EXPLICAR.html",
                 "FAQ.html", "DIAGNOSTICO.html", "VERSAO.txt", "saidas", "R", "vias-dados")
$faltando = @()
foreach ($nome in $temQueFicar) {
    if (-not (Test-Path (Join-Path $Dist $nome))) { $faltando += $nome }
}
$faltando = @($faltando)
Passo "a limpeza NAO tocou no que viaja" ($faltando.Count -eq 0) `
    ("conferidos $($temQueFicar.Count): " + $(if ($faltando.Count) { "FALTAM " + ($faltando -join ",") } else { "todos presentes" }))

# --- 4a. OS DOIS ARQUIVOS QUE FAZEM O PARALELISMO EXISTIR --------------------
#
# **`_internal` esta na lista acima, mas `Test-Path` sobre uma PASTA nao diz nada do que ela tem.**
# O paralelismo do congelado depende de dois arquivos la dentro, e nenhum outro passo os ve:
#
#   `pythonw.exe`      o interpretador que o `loky` lanca. Sem ele a rede cai para SERIE e o
#                      usuario espera 6x mais, em silencio -- nada na tela o diz
#   `python*._pth`     o que ISOLA o worker. Sem ele o interpretador cai no registro do Windows
#                      ou no prefixo de construcao: na maquina que constroi mascara a falha, na
#                      do usuario mata o worker
#
# O piso de 50 KB e o mesmo que `congela_windows.ps1` usa quando copia -- um arquivo truncado
# passa `Test-Path` e nao arranca. O `._pth` e por glob e nao pelo nome, para nao acoplar a versao.
$interno = Join-Path $Dist "_internal"
$interpretador = Join-Path $interno "pythonw.exe"
$temInterp = (Test-Path $interpretador) -and ((Get-Item $interpretador).Length -ge 50000)
$temPth = @(Get-ChildItem -Path $interno -Filter "python*._pth" -ErrorAction SilentlyContinue).Count -ge 1
$tamInterp = if (Test-Path $interpretador) { (Get-Item $interpretador).Length } else { 0 }
Passo "o interpretador embarcado e o ._pth viajam" ($temInterp -and $temPth) `
    "pythonw.exe $tamInterp bytes (piso 50000), ._pth presente=$temPth"

# --- 4c. O R EMBARCADO: PRESENCA E CAPACIDADE ---------------------------------
#
# `R` esta na lista do passo 4, e `Test-Path` sobre uma pasta nao diz se o R arranca: presenca
# nao e capacidade. Aqui o R e CHAMADO: ele carrega o MetaboAnalystR e diz de onde o
# carregou, e o caminho tem de estar dentro da pasta R desta entrega - nunca de um R da maquina.
# O roteiro das vias tem de ser o do repositorio, byte a byte, e as bibliotecas de dados tem de
# conferir com o manifesto. A prova completa, com calculo e cancelamento, e a etapa 3d.
$pastaR = Join-Path $Dist "R"
$rscript = Join-Path $pastaR "bin\x64\Rscript.exe"
$roteiroR = Join-Path $pastaR "vias_entrypoint.R"
$problemasR = @()
if (-not (Test-Path $rscript)) { $problemasR += "sem Rscript.exe" }
if (-not (Test-Path $roteiroR)) {
    $problemasR += "sem o roteiro"
} else {
    $hashRepo = (Get-FileHash -Algorithm SHA256 (Join-Path $Raiz "r_service\vias_entrypoint.R")).Hash
    if ((Get-FileHash -Algorithm SHA256 $roteiroR).Hash -ne $hashRepo) { $problemasR += "roteiro diverge do repositorio" }
}
$ondeCarregou = ""
if (Test-Path $rscript) {
    $sondaR = Join-Path $env:TEMP ("sonda-r-" + [guid]::NewGuid().ToString("N") + ".R")
    Set-Content -Path $sondaR -Encoding ascii -Value @(
        "suppressPackageStartupMessages(library(MetaboAnalystR))",
        "cat(normalizePath(find.package('MetaboAnalystR'), '/'))")
    # O isolamento e o do lancador do produto, numa forma curta: sem as variaveis de biblioteca do
    # usuario, um R da maquina nao entra pela porta de R_LIBS_USER.
    $guardadas = @{}
    foreach ($v in @("R_LIBS_USER", "R_LIBS_SITE", "R_ENVIRON_USER", "R_PROFILE_USER", "R_HOME")) {
        $guardadas[$v] = [Environment]::GetEnvironmentVariable($v, "Process")
        # R_HOME sai do ambiente: o Rscript deriva-o da propria pasta. As outras quatro apontam para
        # um caminho que nao existe, como no lancador do produto.
        $valor = if ($v -eq "R_HOME") { $null } else { Join-Path $env:TEMP "refino-r-inexistente" }
        [Environment]::SetEnvironmentVariable($v, $valor, "Process")
    }
    $anterior = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        $ondeCarregou = ((& $rscript --vanilla $sondaR 2>$null) -join "").Trim()
        $codigoR = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $anterior
        foreach ($v in $guardadas.Keys) { [Environment]::SetEnvironmentVariable($v, $guardadas[$v], "Process") }
        Remove-Item $sondaR -Force -ErrorAction SilentlyContinue
    }
    $esperadoR = (Join-Path $pastaR "library\MetaboAnalystR") -replace "\\", "/"
    if ($codigoR -ne 0) { $problemasR += "o R saiu com $codigoR" }
    elseif ($ondeCarregou.ToLower() -ne $esperadoR.ToLower()) { $problemasR += "MetaboAnalystR de fora da pasta: $ondeCarregou" }
}
$pyRepo = Join-Path $Raiz ".venv\Scripts\python.exe"
$anterior = $ErrorActionPreference
$ErrorActionPreference = "Continue"
& $pyRepo (Join-Path $PSScriptRoot "busca_dados_vias.py") --destino (Join-Path $Dist "vias-dados") --so-confere *> $null
$codigoDados = $LASTEXITCODE
$ErrorActionPreference = $anterior
if ($codigoDados -ne 0) { $problemasR += "as bibliotecas de dados nao conferem com o manifesto" }
$problemasR = @($problemasR)
Passo "o R embarcado arranca e carrega o MetaboAnalystR de dentro da pasta" ($problemasR.Count -eq 0) `
    $(if ($problemasR.Count) { $problemasR -join "; " } else { "carregado de $ondeCarregou" })

# --- 4b. E A LISTA ACIMA CONCORDA COM O QUE O CONGELAMENTO COPIA --------------
#
# Sem este passo, a lista do passo 4 e o retrato de ontem. Se `congela_windows.ps1`
# passar a copiar um arquivo NOVO, o passo 4 fica CEGO a ele: continua a conferir os que ja
# conhece, verde, e o arquivo novo pode sumir sem ninguem notar.
#
# O oraculo e derivado: todo `Join-Path $Dist "<nome>"` daquele roteiro e um destino de copia.
# E um proxy, e a imprecisao vai declarada: ele apanha tambem os alvos que o roteiro apenas LE,
# como `refino.db` e `refino.log`, que sao exactamente os que o passo 3 apaga. Por isso a
# comparacao desconta `$paraTirar`.
$roteiro = Get-Content (Join-Path $PSScriptRoot "congela_windows.ps1") -Raw
$nomeados = [regex]::Matches($roteiro, 'Join-Path \$Dist "([A-Za-z0-9_.-]+)"') |
    ForEach-Object { $_.Groups[1].Value } | Sort-Object -Unique
$naoConferidos = @($nomeados | Where-Object { ($temQueFicar -notcontains $_) -and ($paraTirar -notcontains $_) })
Passo "a lista do que viaja cobre tudo o que o congelamento copia" ($naoConferidos.Count -eq 0) `
    ("o roteiro nomeia $($nomeados.Count); fora das duas listas: " + $(if ($naoConferidos.Count) { $naoConferidos -join "," } else { "nenhum" }))

# --- 5. OS DOCUMENTOS TEM CONTEUDO --------------------------------------------
#
# O titulo nao leva numero: a linha que o passo IMPRIME deriva-o de `$quantos`, e titulo com
# numero pinado ao lado de saida derivada e a forma mais barata de caducar.
#
# Presenca nao basta: um `Copy-Item` que falhasse a meio deixaria arquivo de zero bytes, e o
# usuario abriria uma pagina em branco. Arquivo que VIAJA e nao tem minimo passa vazio:
# presenca satisfeita, conteudo nenhum, na superficie que o usuario abre.
# Os pisos sao contra TRUNCAMENTO e nao limites justos (valem de 7% a 59% do tamanho medido), e
# um piso apertado reprovaria por edicao normal do documento, que e como um portao aprende a ser
# ignorado. Medidos: `MEDICOES.html` 51.115 bytes (piso 30.000), `MULTIVERSO.html` 36.128
# (piso 20.000), `FAQ.html` 35.346 (piso 20.000).
$minimos = @{ "LEIA-ME.txt" = 4000; "GUIA.html" = 20000; "ENTENDER.html" = 60000;
              "MEDICOES.html" = 30000; "MULTIVERSO.html" = 20000;
              "EXPLICAR.html" = 40000; "FAQ.html" = 20000; "DIAGNOSTICO.html" = 30000;
              "VERSAO.txt" = 20 }
$magros = @()
foreach ($nome in $minimos.Keys) {
    $alvo = Join-Path $Dist $nome
    $tam = if (Test-Path $alvo) { (Get-Item $alvo).Length } else { 0 }
    if ($tam -lt $minimos[$nome]) { $magros += ("{0}={1}b<{2}" -f $nome, $tam, $minimos[$nome]) }
}
$magros = @($magros)
$quantos = $minimos.Count
Passo "os $quantos documentos tem conteudo" ($magros.Count -eq 0) `
    $(if ($magros.Count) { $magros -join ", " } else { "os $quantos acima do minimo" })

# --- 5b. AS TRES PASTAS DE SAIDA, e a conferencia e por PASTA -----------------
#
# O passo 5 mede arquivo por arquivo, e a pasta `saidas` nao tem tamanho proprio. O que se afirma
# aqui e o que quem recebe vai abrir: as tres pastas existem, cada uma tem CSV dentro, e nenhum
# deles esta truncado.
#
# **Os numeros sao DERIVADOS e nao pinados:** quatro modos vezes tres corretores dao 12 na pasta
# das corrigidas e 12 na do MetaboAnalyst; a das processadas leva duas variantes por modo, 8. Um
# modo novo ou um corretor novo mudam os tres numeros juntos, e a conta acompanha.
$saidas = Join-Path $Dist "saidas"
$modos = 4
$corretores = 3
$variantes = 2
$esperado = [ordered]@{
    "1-matrizes-corrigidas"   = $modos * $corretores
    "2-formato-metaboanalyst" = $modos * $corretores
    "3-matrizes-processadas"  = $modos * $variantes
}
$problemas = @()
foreach ($pasta in $esperado.Keys) {
    $alvo = Join-Path $saidas $pasta
    if (-not (Test-Path $alvo)) { $problemas += "$pasta AUSENTE"; continue }
    $csv = @(Get-ChildItem $alvo -Filter *.csv)
    if ($csv.Count -ne $esperado[$pasta]) {
        $problemas += ("{0}={1} csv (esperado {2})" -f $pasta, $csv.Count, $esperado[$pasta])
    }
    $magrosCsv = @($csv | Where-Object { $_.Length -lt 10000 })
    if ($magrosCsv.Count) { $problemas += ("{0}: {1} truncado(s)" -f $pasta, $magrosCsv.Count) }
}
$problemas = @($problemas)
$totalCsv = 0
if (Test-Path $saidas) { $totalCsv = @(Get-ChildItem $saidas -Recurse -Filter *.csv).Count }
Passo "as tres pastas de saida estao completas" ($problemas.Count -eq 0) `
    $(if ($problemas.Count) { $problemas -join ", " } else { "$totalCsv CSV nas tres pastas" })

# --- 6. O ZIP, quando pedido --------------------------------------------------
if ($Zipar) {
    $zip = Join-Path $Raiz "dist\refino.zip"
    if (Test-Path $zip) { Remove-Item $zip -Force }
    Compress-Archive -Path $Dist -DestinationPath $zip -CompressionLevel Optimal
    $tam = if (Test-Path $zip) { (Get-Item $zip).Length } else { 0 }
    Passo "o zip foi criado e nao esta vazio" ($tam -gt 1000000) ("{0} ({1:N0} bytes)" -f $zip, $tam)

    # O zip tem de conter o executavel. `Compress-Archive` sobre uma pasta cria um nivel a mais, e
    # um zip com a arvore errada e indistinguivel de um zip bom pelo tamanho.
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $arq = [System.IO.Compression.ZipFile]::OpenRead($zip)
    $temExe = @($arq.Entries | Where-Object { $_.Name -eq "refino.exe" }).Count
    $temLeiaMe = @($arq.Entries | Where-Object { $_.Name -eq "LEIA-ME.txt" }).Count
    $total = $arq.Entries.Count
    $arq.Dispose()
    Passo "o zip contem o executavel e o LEIA-ME" (($temExe -eq 1) -and ($temLeiaMe -eq 1)) `
        "entradas=$total refino.exe=$temExe LEIA-ME.txt=$temLeiaMe"
}

Write-Host ""
if ($script:Falhas -gt 0) {
    Write-Host "  VEREDITO: $($script:Falhas) passo(s) reprovaram - NAO envie"
} elseif ($Zipar) {
    Write-Host "  VEREDITO: pronto. Envie dist\refino.zip"
} else {
    Write-Host "  VEREDITO: pronto. Compacte a pasta $Dist, ou corra outra vez com -Zipar"
}
Write-Host ""
exit ([int]($script:Falhas -gt 0))
