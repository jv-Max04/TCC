# Instala, na .Library do R portátil em montagem, o grafo de dependências do MetaboAnalystR no commit
# fixado, e depois o próprio MetaboAnalystR, compilado com o Rtools44.
#
# Os caminhos chegam pelo ambiente (`REFINO_R_FONTE`, `REFINO_R_LOGS`) e nao fixos, porque quem
# escolhe a unidade `subst` e `scripts/monta_r_portatil.ps1`.
#
# As versões fixadas são CRAN no instantâneo P3M 2025-06-02 e Bioconductor 3.20; o commit do
# MetaboAnalystR fica em `scripts/monta_r_portatil.ps1`. `tests/test_r_portatil_roteiro.py` confere
# que os três pinos estão declarados uma vez só entre os dois arquivos.
#
# Dependências DURAS só (Depends, Imports, LinkingTo), mais `qs` e `qs2`: o `DESCRIPTION` exige `qs`
# e o namespace carrega `qs2`. O que nao tem binario do Windows no instantaneo (o `crmn`) sai
# da fonte, e o MetaboAnalystR, que tem `src/` em C e C++, também.
t0 <- Sys.time()
options(repos = c(
  CRAN     = "https://packagemanager.posit.co/cran/2025-06-02",
  BioCsoft = "https://bioconductor.org/packages/3.20/bioc",
  BioCann  = "https://bioconductor.org/packages/3.20/data/annotation",
  BioCexp  = "https://bioconductor.org/packages/3.20/data/experiment"),
  timeout = 600)

fonte <- Sys.getenv("REFINO_R_FONTE")
logs <- Sys.getenv("REFINO_R_LOGS")
if (!nzchar(fonte) || !dir.exists(fonte)) stop("REFINO_R_FONTE nao aponta para a fonte extraida do MetaboAnalystR")
if (!nzchar(logs) || !dir.exists(logs)) stop("REFINO_R_LOGS nao aponta para um diretorio existente")

# A biblioteca alvo é a do próprio R, e só ela. Se `.libPaths()` trouxer outra coisa, o ambiente do
# processo vazou (R_LIBS*, biblioteca do usuário) e o que se instalaria iria parar fora da árvore.
if (!identical(normalizePath(.libPaths()), normalizePath(.Library))) {
  stop(paste("o .libPaths() nao e so a .Library:", paste(.libPaths(), collapse = " | ")))
}
cat("R:", R.version.string, "\nR.home:", R.home(), "\n")

desc <- read.dcf(file.path(fonte, "DESCRIPTION"))
campo <- function(f) {
  x <- desc[1, f]
  if (is.na(x)) return(character(0))
  x <- gsub("[(][^)]*[)]", "", x)
  x <- trimws(strsplit(x, ",")[[1]])
  x[nzchar(x)]
}
diretas <- unique(c(campo("Imports"), campo("LinkingTo"), setdiff(campo("Depends"), "R"), "qs", "qs2"))
cat("diretas (", length(diretas), "):", paste(sort(diretas), collapse = " "), "\n")

ap <- available.packages(type = "win.binary")
base_pkgs <- rownames(installed.packages(lib.loc = .Library, priority = "base"))
rec <- tools::package_dependencies(diretas, db = ap, which = c("Depends", "Imports", "LinkingTo"), recursive = TRUE)
todas <- setdiff(unique(c(diretas, unlist(rec))), base_pkgs)
faltam <- setdiff(todas, rownames(installed.packages(lib.loc = .Library)))
sem_bin <- setdiff(faltam, rownames(ap))
cat("grafo (sem base):", length(todas), " a instalar:", length(faltam), " sem binario:", paste(sem_bin, collapse = " "), "\n")
writeLines(sort(todas), file.path(logs, "grafo-deps.txt"))

install.packages(setdiff(faltam, sem_bin), lib = .Library, type = "win.binary", dependencies = FALSE)

# O que sobrou sem binário: só se aceita o que se sabe compilar. Um pacote novo nesta lista é
# mudança do instantâneo, e a montagem para em vez de compilar às cegas.
esperado_da_fonte <- c("crmn", "MetaboAnalystR")
inesperado <- setdiff(sem_bin, esperado_da_fonte)
if (length(inesperado)) stop(paste("sem binario e fora do esperado:", paste(inesperado, collapse = " ")))
# Os dois que saem da fonte vão por `R CMD INSTALL` com a biblioteca SEM normalizar, e não por
# `install.packages(type = "source")`. Medido numa segunda montagem: o `install.packages`
# normaliza o caminho da biblioteca, o que no Windows troca a unidade SUBST pelo caminho longo da
# pasta de build — e esse caminho fica gravado em `.__NAMESPACE__.$path` dentro de `crmn.rdb`. Duas
# montagens em pastas diferentes davam bancos de carga diferentes por isso e só por isso. O
# MetaboAnalystR, que já ia por `R CMD INSTALL`, gravava a unidade curta e não variava.
instala_da_fonte <- function(caminho, nome) {
  cat("RTOOLS44_HOME =", Sys.getenv("RTOOLS44_HOME"), " make:", Sys.which("make"), "\n")
  st <- system2(file.path(R.home("bin"), "R.exe"),
                c("CMD", "INSTALL", "--no-docs", paste0("--library=", shQuote(.Library)), shQuote(caminho)))
  if (!identical(as.integer(st), 0L)) stop(paste("R CMD INSTALL de", nome, "saiu com", st))
}
if ("crmn" %in% sem_bin) {
  baixado <- download.packages("crmn", destdir = logs, type = "source")
  instala_da_fonte(baixado[1, 2], "crmn")
}
instala_da_fonte(fonte, "MetaboAnalystR")

falta <- setdiff(c(todas, "MetaboAnalystR"), rownames(installed.packages(lib.loc = .Library)))
if (length(falta)) stop(paste("faltando depois de instalar:", paste(falta, collapse = " ")))
cat("MetaboAnalystR", as.character(packageVersion("MetaboAnalystR", lib.loc = .Library)), "\n")
cat("segundos:", round(as.numeric(difftime(Sys.time(), t0, units = "secs"))), "\n")
cat("FIM-INSTALA\n")
