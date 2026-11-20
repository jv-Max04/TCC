# Corre um roteiro de vias e, ao SAIR do R, grava `loadedNamespaces()` num arquivo.
#
# Existe porque a lista de pacotes da árvore podada tem de vir também dos RAMOS DE ERRO: o roteiro
# termina por `quit()` em vários pontos, e só alguns devolvem a lista na resposta. Um finalizador com
# `onexit = TRUE` corre em toda saída normal da sessão — inclusive por `quit()` —, então o despejo
# não depende de por onde o roteiro saiu. Usado por `scripts/r_portatil.py deriva`.
#
# O caminho do roteiro e o do despejo chegam por `REFINO_R_ROTEIRO` e `REFINO_R_DESPEJO`, e não por
# `R_*`: o lancador limpa do ambiente toda variavel que comeca por `R_`.
local({
  destino <- Sys.getenv("REFINO_R_DESPEJO")
  if (!nzchar(destino)) stop("REFINO_R_DESPEJO vazio")
  reg.finalizer(globalenv(), function(e) writeLines(sort(loadedNamespaces()), destino), onexit = TRUE)
})
source(Sys.getenv("REFINO_R_ROTEIRO"), local = FALSE, encoding = "UTF-8")
