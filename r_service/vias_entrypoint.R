#!/usr/bin/env Rscript
# Roteiro das vias pelo MetaboAnalystR, executado pela árvore R portátil que viaja na pasta do
# executável.
#
# ---------------------------------------------------------------------------
# O CONTRATO: JSON numa linha em stdin, JSON numa linha em stdout, nunca `rpy2`
# ---------------------------------------------------------------------------
#
#   mapear     {"action": "mapear", "dados_dir": "<pasta>", "lista": [...], "fundo": [...]?}
#              -> {"status": "ok", "lista": {consultados, casados, mapeamento}, "fundo": {...} | null}
#
#   ora_local  {"action": "ora_local", "dados_dir": "<pasta>", "lista": [...], "fundo": [...]?}
#              -> {"status": "ok", "biblioteca": "smpdb_pathway", "universo": {tipo, tamanho},
#                  "na_lista": q, "vias": [{via, total, esperado, acertos, p, holm, fdr, compostos}]}
#
#   recusa     {"status": "erro", "desfecho": "<vocabulário fechado>", "motivo": "..."}
#
# Uma análise faz duas chamadas: `mapear` devolve os nomes canônicos da lista e do fundo, o núcleo
# Python aplica as recusas sobre esse resultado, e só então `ora_local` recebe os nomes canônicos
# confirmados. Lista de 1 ou 2 compostos calcula, e o aviso é do lado Python: `CalculateHyperScore`
# (`R/enrich_stats.R:17-136`, commit `6bf5462`) só recusa lista vazia, biblioteca de um conjunto só
# e zero acertos; o «Less than 3 metabolites left» é de `CalculateOraScore`
# (`R/enrich_path_stats.R:91-92`), a rota de topologia, que este roteiro não usa. Fica a recusa da
# lista que não tem nenhum composto no universo, contada depois da interseção.
#
# Os desfechos de recusa são vocabulário fechado, porque o lançador do lado Python os valida como
# fronteira de confiança: `pedido_invalido`, `biblioteca_ausente`, `fundo_invalido`,
# `fundo_recusado`, `lista_recusada`, `lista_igual_ao_fundo`, `sem_casamento_na_biblioteca` e
# `motor_falhou`. Nada mapeado não é recusa: é `status: ok` com `casados: 0`, porque nome que não
# está na base é desfecho legítimo da entrada.
#
# ---------------------------------------------------------------------------
# O QUE ESTE ROTEIRO FAZ DIFERENTE DO PACOTE, e cada item é contorno medido
# ---------------------------------------------------------------------------
#
#  1. As bibliotecas vêm só da pasta do pedido. A função de biblioteca do pacote (`.get.my.lib`)
#     baixa quando o arquivo falta ou tem mais de 30 dias, e sem rede o arquivo some. Ela é trocada
#     no namespace por uma leitura só de `dados_dir`. A porta da API (`.do.api.call`) é trocada por
#     uma que aborta: este roteiro não fala com a rede.
#  2. `SetCurrentMsetLib` não passa por `.get.my.lib`: lê `smpdb_pathway.qs` do diretório de
#     trabalho e, se faltar, chama `download.file(method = "curl")` (`R/enrich_mset.R:58-66`, no
#     commit fixado). O roteiro copia a biblioteca da pasta para o diretório de trabalho antes.
#  3. `CalculateHyperScore` morre depois de calcular, fora do servidor web: grava a tabela e então
#     chama exportadores e um gráfico com `.get.mSet(NA)`, que fora da web devolve `NA`. Os três
#     (`ExportResultMatArrow`, `ExportOraMembershipJson`, `PlotORAMembership`) viram nulos no
#     namespace; o cálculo, que vem antes deles, não muda.
#  4. `default.dpi` explícito: a assinatura de `InitDataObjects` tem o argumento por padrão ele
#     mesmo, e não há `default.dpi` global, então a chamada que o omite morre em `promise already
#     under evaluation`. E `ClearErrMsg()` antes de mapear: o pacote pressupõe os globais
#     `err.vec` e `current.msg` sem os inicializar, e `ClearErrMsg()` é o inicializador que ele
#     expõe.
#  5. O fundo lido tem de ter o tamanho do enviado. O pacote lê o fundo de um arquivo, uma linha
#     por nome; um nome com quebra de linha viraria dois e mudaria o universo em silêncio.

local({
  # Isolamento: só a biblioteca do próprio R. `.libPaths(new)` sempre acrescenta `.Library.site`,
  # e um MetaboAnalystR instalado em `site-library` continuaria visível.
  .libPaths(.Library)
})

suppressPackageStartupMessages(library(jsonlite))

DPI_PADRAO <- 72L
BIBLIOTECA_DE_VIAS <- "smpdb_pathway"
DESFECHOS <- c("pedido_invalido", "biblioteca_ausente", "fundo_invalido", "fundo_recusado",
               "lista_recusada", "lista_igual_ao_fundo", "sem_casamento_na_biblioteca", "motor_falhou")
ARQUIVOS <- list(
  mapear = c("compound_db.qs", "syn_nms.qs"),
  ora_local = c("compound_db.qs", "syn_nms.qs", "master_compound_db.qs", "msets_smpdb_pathway.qs")
)

responder <- function(payload) {
  payload$ambiente <- list(
    R_home = normalizePath(R.home(), "/", mustWork = FALSE),
    libPaths = normalizePath(.libPaths(), "/", mustWork = FALSE),
    versao_R = R.version.string,
    versao_metaboanalystr = tryCatch(as.character(utils::packageVersion("MetaboAnalystR")), error = function(e) NA_character_)
  )
  txt <- toJSON(payload, auto_unbox = TRUE, null = "null", na = "null", digits = NA)
  cat(enc2utf8(as.character(txt)), "\n", sep = "")
}

recusar <- function(desfecho, motivo) {
  stopifnot(desfecho %in% DESFECHOS)
  responder(list(status = "erro", desfecho = desfecho, motivo = motivo))
  quit(save = "no", status = 1L)
}

# O pacote IMPRIME em stdout; o contrato é uma linha de JSON. A conversa dele vai para o stderr,
# que é diagnóstico útil quando algo dá errado.
calado <- function(expr_fun) {
  sink(stderr(), type = "output")
  on.exit(sink(type = "output"), add = TRUE)
  tryCatch(expr_fun(), error = function(e) e)
}

le_pedido <- function() {
  bruto <- paste(readLines(file("stdin", encoding = "UTF-8"), warn = FALSE), collapse = "\n")
  if (!nzchar(trimws(bruto))) recusar("pedido_invalido", "stdin vazio")
  pedido <- tryCatch(fromJSON(bruto, simplifyVector = TRUE),
                     error = function(e) recusar("pedido_invalido", paste("JSON invalido:", conditionMessage(e))))
  acao <- pedido[["action"]]
  if (is.null(acao) || !(as.character(acao) %in% names(ARQUIVOS))) {
    recusar("pedido_invalido", paste("acao nao suportada:", if (is.null(acao)) "<ausente>" else as.character(acao)))
  }
  dados_dir <- pedido[["dados_dir"]]
  if (is.null(dados_dir) || !dir.exists(as.character(dados_dir))) {
    recusar("biblioteca_ausente", paste("a pasta das bibliotecas nao existe:", if (is.null(dados_dir)) "<ausente>" else dados_dir))
  }
  nomes <- function(campo, obrigatorio) {
    x <- pedido[[campo]]
    if (is.null(x)) {
      if (obrigatorio) recusar("pedido_invalido", paste0("campo `", campo, "` ausente"))
      return(NULL)
    }
    x <- trimws(enc2utf8(as.character(x)))
    x <- x[nzchar(x)]
    if (any(grepl("[[:cntrl:]]", x))) {
      recusar(if (campo == "fundo") "fundo_invalido" else "pedido_invalido",
              paste0("`", campo, "` tem nome com caractere de controle; o pacote le um nome por linha"))
    }
    x
  }
  lista <- nomes("lista", TRUE)
  if (!length(lista)) recusar("pedido_invalido", "`lista` vazia")
  list(acao = as.character(acao), dados_dir = normalizePath(as.character(dados_dir), "/"), lista = lista,
       fundo = nomes("fundo", FALSE))
}

instala_contornos <- function(dados_dir) {
  faltou <- character(0)
  le_biblioteca <- function(filenm, sub.dir = NULL) {
    alvo <- file.path(dados_dir, if (is.null(sub.dir)) filenm else paste0(sub.dir, "_", filenm))
    if (!file.exists(alvo)) {
      faltou <<- c(faltou, basename(alvo))
      stop(paste("biblioteca ausente na pasta do pedido:", basename(alvo)))
    }
    MetaboAnalystR:::ov_qs_read(alvo)
  }
  sem_rede <- function(...) stop("este roteiro nao fala com a rede (.do.api.call foi chamada)")
  nulo <- function(...) invisible(NULL)
  ns <- asNamespace("MetaboAnalystR")
  utils::assignInNamespace(".get.my.lib", le_biblioteca, ns = ns)
  utils::assignInNamespace(".do.api.call", sem_rede, ns = ns)
  for (f in c("ExportResultMatArrow", "ExportOraMembershipJson", "PlotORAMembership")) {
    utils::assignInNamespace(f, nulo, ns = ns)
  }
  function() faltou
}

mapeia <- function(nomes, anal) {
  MetaboAnalystR:::ClearErrMsg()
  m <- calado(function() {
    x <- InitDataObjects("conc", anal, FALSE, default.dpi = DPI_PADRAO)
    x <- Setup.MapData(x, nomes)
    CrossReferencing(x, "name")
  })
  if (inherits(m, "error")) return(m)
  if (!is.list(m)) {
    return(list(mSet = NULL, tabela = data.frame(Query = nomes, Match = NA_character_, HMDB = NA_character_,
                                                  KEGG = NA_character_, stringsAsFactors = FALSE)))
  }
  m <- calado(function() CreateMappingResultTable(m))
  if (inherits(m, "error")) return(m)
  list(mSet = m, tabela = read.csv("name_map.csv", stringsAsFactors = FALSE, encoding = "UTF-8"))
}

descreve_mapeamento <- function(tab) {
  col <- function(n) if (n %in% names(tab)) as.character(tab[[n]]) else rep(NA_character_, nrow(tab))
  hmdb <- col("HMDB"); kegg <- col("KEGG"); casado <- col("Match")
  tem <- function(v) !is.na(v) & nzchar(trimws(v)) & v != "NA"
  list(
    consultados = nrow(tab),
    # `casados` conta identificador curado, e não nome devolvido.
    casados = sum(tem(hmdb)),
    mapeamento = lapply(seq_len(nrow(tab)), function(i) list(
      consulta = as.character(tab$Query[i]),
      casado = if (tem(casado[i])) casado[i] else NULL,
      hmdb = if (tem(hmdb[i])) hmdb[i] else NULL,
      kegg = if (tem(kegg[i])) kegg[i] else NULL))
  )
}

acao_mapear <- function(p, faltou) {
  r <- mapeia(p$lista, "msetora")
  if (inherits(r, "error")) {
    if (length(faltou())) recusar("biblioteca_ausente", conditionMessage(r))
    recusar("motor_falhou", paste("o mapeamento da lista levantou:", conditionMessage(r)))
  }
  saida <- list(status = "ok", acao = "mapear", lista = descreve_mapeamento(r$tabela), fundo = NULL)
  if (length(p$fundo)) {
    f <- mapeia(p$fundo, "msetora")
    if (inherits(f, "error")) {
      if (length(faltou())) recusar("biblioteca_ausente", conditionMessage(f))
      recusar("motor_falhou", paste("o mapeamento do fundo levantou:", conditionMessage(f)))
    }
    saida$fundo <- descreve_mapeamento(f$tabela)
  }
  responder(saida)
}

acao_ora_local <- function(p, faltou) {
  # Contorno 2: a biblioteca de vias tem de estar no diretório de trabalho ANTES de ser selecionada.
  if (!file.copy(file.path(p$dados_dir, "msets_smpdb_pathway.qs"), paste0(BIBLIOTECA_DE_VIAS, ".qs"), overwrite = TRUE)) {
    recusar("biblioteca_ausente", "msets_smpdb_pathway.qs nao foi copiada para o diretorio de trabalho")
  }
  r <- mapeia(p$lista, "msetora")
  if (inherits(r, "error")) {
    if (length(faltou())) recusar("biblioteca_ausente", conditionMessage(r))
    recusar("motor_falhou", paste("o mapeamento da lista levantou:", conditionMessage(r)))
  }
  if (is.null(r$mSet)) recusar("lista_recusada", "nenhum nome da lista casou com a base")
  mSet <- r$mSet
  com_fundo <- length(p$fundo) > 0L
  if (com_fundo) {
    writeLines(enc2utf8(p$fundo), "fundo.txt", useBytes = TRUE)
    # Contorno 5: relê como o pacote lê, e confere o tamanho.
    lido <- scan("fundo.txt", "character", strip.white = TRUE, sep = "\n", quiet = TRUE, encoding = "UTF-8")
    if (length(lido) != length(p$fundo)) {
      recusar("fundo_invalido", sprintf("o pacote leria %d nomes no fundo, e foram enviados %d", length(lido), length(p$fundo)))
    }
    f <- calado(function() Setup.HMDBReferenceMetabolome(mSet, "fundo.txt"))
    if (inherits(f, "error")) {
      if (length(faltou())) recusar("biblioteca_ausente", conditionMessage(f))
      recusar("motor_falhou", paste("o fundo levantou:", conditionMessage(f)))
    }
    if (!is.list(f)) recusar("fundo_recusado", paste(MetaboAnalystR:::GetErrMsg(), collapse = " | "))
    mSet <- f
  }
  mSet <- calado(function() SetMetabolomeFilter(mSet, com_fundo))
  if (inherits(mSet, "error")) recusar("motor_falhou", paste("SetMetabolomeFilter levantou:", conditionMessage(mSet)))
  mSet <- calado(function() SetCurrentMsetLib(mSet, BIBLIOTECA_DE_VIAS, 2))
  if (inherits(mSet, "error")) {
    if (length(faltou())) recusar("biblioteca_ausente", conditionMessage(mSet))
    recusar("motor_falhou", paste("SetCurrentMsetLib levantou:", conditionMessage(mSet)))
  }

  # A recusa da lista sem composto no universo e a de «lista igual ao fundo». A CONTAGEM é a de
  # `CalculateHyperScore`: a lista é o HMDB distinto do mapeamento, e o universo é o fundo filtrado
  # ou a união dos conjuntos. Uma lista de 1 ou 2 compostos segue para o cálculo.
  nm <- MetaboAnalystR:::GetFinalNameMap(mSet)
  ora_vec <- unique(nm$hmdb[!is.na(nm$hmdb)])
  membros <- MetaboAnalystR:::ov_qs_read("current.msetlib.qs")$member
  universo <- if (com_fundo) unique(mSet$dataSet$metabo.filter.hmdb) else unique(unlist(membros, use.names = FALSE))
  na_lista <- intersect(ora_vec, universo)
  if (length(na_lista) == 0L) {
    recusar("sem_casamento_na_biblioteca",
            "depois do mapeamento e do universo, nenhum composto da lista esta nas vias da biblioteca")
  }
  if (com_fundo && setequal(na_lista, universo)) {
    recusar("lista_igual_ao_fundo", "a lista e igual ao fundo: todo p sai 1, e a analise nao tem sentido")
  }

  res <- calado(function() CalculateHyperScore(mSet))
  if (inherits(res, "error")) recusar("motor_falhou", paste("CalculateHyperScore levantou:", conditionMessage(res)))
  if (!is.list(res)) recusar("sem_casamento_na_biblioteca", paste(MetaboAnalystR:::GetErrMsg(), collapse = " | "))
  om <- res$analSet$ora.mat
  acertos <- res$analSet$ora.hits
  vias <- lapply(seq_len(nrow(om)), function(i) {
    via <- rownames(om)[i]
    list(via = via, total = om[i, "total"], esperado = om[i, "expected"], acertos = om[i, "hits"],
         p = om[i, "Raw p"], holm = om[i, "Holm p"], fdr = om[i, "FDR"],
         compostos = I(sort(as.character(acertos[[via]]))))
  })
  responder(list(
    status = "ok", acao = "ora_local", biblioteca = BIBLIOTECA_DE_VIAS,
    universo = list(tipo = if (com_fundo) "fundo_do_ensaio" else "biblioteca", tamanho = length(universo)),
    na_lista = length(na_lista), vias = vias
  ))
}

principal <- function() {
  p <- le_pedido()
  faltando <- ARQUIVOS[[p$acao]][!file.exists(file.path(p$dados_dir, ARQUIVOS[[p$acao]]))]
  if (length(faltando)) recusar("biblioteca_ausente", paste("faltam na pasta das bibliotecas:", paste(faltando, collapse = ", ")))
  carregou <- tryCatch({ suppressPackageStartupMessages(library(MetaboAnalystR)); TRUE },
                       error = function(e) conditionMessage(e))
  if (!isTRUE(carregou)) recusar("motor_falhou", paste("library(MetaboAnalystR) falhou:", carregou))
  faltou <- instala_contornos(p$dados_dir)
  # O pacote lê e ESCREVE no diretório de trabalho; quem o escolhe é o lançador (um temporário por
  # corrida). Aqui ele só é conferido gravável.
  if (file.access(".", 2) != 0) recusar("motor_falhou", "o diretorio de trabalho nao e gravavel")
  if (p$acao == "mapear") acao_mapear(p, faltou) else acao_ora_local(p, faltou)
}

principal()
