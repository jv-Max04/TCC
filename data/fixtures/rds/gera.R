# Gera as fixtures do leitor restrito de RDS (`src/refino/io/rds_restrito.py`).
#
# Cada objeto sai em dois formatos: `serialize(x, NULL)` cru — a forma da resposta do servidor do
# MetaboAnalyst — e `saveRDS(x, compress = TRUE)` — a forma do `tosend.rds` que o pacote envia.
# Corrido em 01/10/2026 com o Rscript da arvore R portatil (R 4.4.3), e de novo em 02/10/2026
# com o objeto `embrulho` acrescentado (os oito arquivos anteriores sairam com o mesmo sha256),
# assim:
#   Rscript --vanilla data/fixtures/rds/gera.R data/fixtures/rds
args <- commandArgs(trailingOnly = TRUE)
destino <- args[1]
grava <- function(nome, x) {
  writeBin(serialize(x, NULL), file.path(destino, paste0(nome, ".ser")))
  saveRDS(x, file.path(destino, paste0(nome, ".rds")), compress = TRUE)
}
m <- matrix(c(10, 5, 1.5, 0.5, 3, 1, 0.0123, 0.4, 0.1, 1, 0.05, 0.8), nrow = 2,
            dimnames = list(c("Urea Cycle", "Glycolysis"), c("total", "expected", "hits", "Raw p", "Holm p", "FDR")))
aceito <- list(
  analSet = list(ora.mat = m, ora.hits = list(`Urea Cycle` = c("Urea", "L-Arginine", "Ornithine"), Glycolysis = "Pyruvate")),
  textos = c("Ácido cítrico", NA, "β-Alanine", ""),
  logicos = c(TRUE, FALSE, NA),
  inteiros = c(1L, NA, -7L),
  sequencia = 1:10,
  vazio = NULL,
  fator = factor(c("a", "b", "a"))
)
grava("aceito", aceito)
grava("com_funcao", list(ok = 1, f = function(x) x + 1))
grava("com_ambiente", list(ok = 1, e = new.env()))
grava("com_formula", list(ok = 1, fo = y ~ x))
# Os embrulhos ALTREP (`wrap_*`) da base: o estado deles e um par pontilhado `CONS(x, meta)`, que
# um leitor que o tratasse como pairlist recusaria em todo embrulho real.
embrulho <- list(
  real = .Internal(wrap_meta(c(2.5, 1.5), 0L, 0L)),
  inteiro = .Internal(wrap_meta(c(3L, NA, 1L), 0L, 0L)),
  texto = .Internal(wrap_meta(c("b", "a"), 0L, 0L)),
  logico = .Internal(wrap_meta(c(TRUE, NA), 0L, 0L)),
  lista = .Internal(wrap_meta(list(1L, "z"), 0L, 0L)),
  nomeado = .Internal(wrap_meta(c(a = 1.5, b = 2.5), 0L, 0L))
)
grava("embrulho", embrulho)
cat("ok\n")
