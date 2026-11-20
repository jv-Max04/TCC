# A porta do congelado e a que RESPONDE `/health` — nunca a primeira da lista.
#
# ============================================================================
# POR QUE ESTE ARQUIVO EXISTE, e o defeito que ele fecha foi medido
# ============================================================================
#
# **A tela de arranque do PyInstaller abre uma porta de escuta no loopback, e ela chega ANTES da
# nossa.** O `pyi_splash` fala com o interpretador Tcl da tela por TCP em 127.0.0.1, o carregador
# em C abre-a antes de o Python arrancar, e ela fecha quando a tela fecha. Tracado segundo a
# segundo, num lancamento so:
#
#   t=1s   pid 28680 -> [::1:57245  127.0.0.1:57245]   health=[]
#   t=3s   pid 28680 -> [127.0.0.1:57247]              health=[{"status":"ok"}]
#
# **`$portas[0]` apanhava a 57245, e o desfecho nao era um erro.** Aquela porta ACEITA a ligacao e
# nunca responde HTTP, entao o `curl` fica pendurado ate o teto. As duas corridas da prova do
# paralelismo morreram com `o disparo nao devolveu identificador` e um `$disparo` VAZIO; o processo,
# olhado no mesmo instante, dizia `titulo=tk`. Isso leu-se como o produto a congelar no arranque,
# e o produto estava bem: respondia `{"status":"ok"}` na outra porta, no mesmo segundo.
#
# > O instrumento le o referente certo (uma porta em LISTEN deste PID) e a INSTANCIA errada, e o
# > veredito `[FALHA] as duas corridas terminaram com 200` acusava o alvo quando a causa estava
# > no instrumento.
#
# As outras sondas (`congela_windows.ps1` em tres sitios, `mede_serrf_congelado.ps1`,
# `prova_multiverso_congelado.ps1` e `prova_artefatos_congelado.ps1`) escolhiam a porta pela ORDEM e
# passavam porque, quando liam a lista, a tela ja tinha fechado. Passar por sorte de temporizacao
# nao e passar: o primeiro arranque a frio (77 s medidos, com a tela de pe o tempo todo) e
# exactamente onde a sorte acaba.
#
# O oraculo aqui nao caduca se a tela mudar de forma: ele pergunta pelo que a sonda de facto
# precisa, que e a porta onde a NOSSA API atende.
#
# Uso, com o ponto de dot-source:
#
#     . (Join-Path $PSScriptRoot "porta_que_serve.ps1")
#     $base = PortaQueServe -Processo $proc -TetoSeg 180
#     if (-not $base) { ...  }

function PortaQueServe {
    param(
        [Parameter(Mandatory = $true)]$Processo,
        [int]$TetoSeg = 180
    )
    # **`TetoSeg` e um RELOGIO e nao uma contagem de voltas, e a diferenca foi medida.** A
    # primeira versao contava iteracoes: com a planta do processo mudo, `-TetoSeg 8` gastou
    # **41,2 s** — cada volta custa 1 s de espera mais 4 s de `--max-time` POR porta candidata.
    # Com 180, o pior caso seriam ~15 min de sonda pendurada a dizer que espera 3.
    $relogio = [Diagnostics.Stopwatch]::StartNew()
    while ($relogio.Elapsed.TotalSeconds -lt $TetoSeg) {
        Start-Sleep -Seconds 1
        if ($Processo.HasExited) { return "" }
        $candidatas = @(Get-NetTCPConnection -State Listen -OwningProcess $Processo.Id -ErrorAction SilentlyContinue |
                        Where-Object { $_.LocalAddress -eq "127.0.0.1" } |
                        ForEach-Object { $_.LocalPort } | Sort-Object -Unique)
        foreach ($porta in $candidatas) {
            # **`--max-time` e obrigatorio e nao e prudencia.** Sem ele, a porta da tela de arranque
            # pendura o `curl` para sempre — foi esse o modo de falha, e nao um erro de ligacao.
            $saude = curl.exe -s --max-time 4 "http://127.0.0.1:$porta/health" 2>&1
            if ("$saude" -match '"status"\s*:\s*"ok"') { return "http://127.0.0.1:$porta" }
        }
    }
    return ""
}
