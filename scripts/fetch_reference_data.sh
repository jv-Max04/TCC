#!/usr/bin/env bash
#
# Procedência e integridade dos dados de referência.
#
# Este script não baixa nada: verifica os arquivos contra o manifesto.
#
#   - Teste com arquivo de referência ausente vira `skip`, nunca dado fabricado
#     (regra 1 do produto, no README). Este verificador é o que alimenta essa
#     decisão.
#   - Não há de onde baixar: é dado de laboratório parceiro, não publicado.
#   - A pasta `serrf/` fica fora do git, então o caso "ausente" é o que um clone
#     novo encontra. A queda para `serrf/data` abaixo é o caminho de quem tem a
#     pasta em disco, que é o arranjo previsto.
#
# Uso:
#   scripts/fetch_reference_data.sh            verifica contra o manifesto
#   scripts/fetch_reference_data.sh --write    (re)gera o manifesto
#
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Raiz dos dados: o symlink `data/reference`, com queda para a árvore real. O
# symlink não materializa em checkout Windows sem `core.symlinks=true`, e depender
# dele fazia a verificação de integridade falhar por plataforma.
if [[ -d "$REPO/data/reference/input" ]]; then
  REFERENCE="$REPO/data/reference"
elif [[ -d "$REPO/serrf/data/input" ]]; then
  REFERENCE="$REPO/serrf/data"
else
  echo "Nenhuma raiz de dados de referência resolve." >&2
  echo "  esperado: data/reference/input (symlink) ou serrf/data/input" >&2
  echo "  Sem dados, os testes golden dão skip — o que é o comportamento" >&2
  echo "  correto da §8.4, e não falha." >&2
  exit 0
fi
MANIFEST="$REPO/scripts/reference_data.sha256"

# ---------------------------------------------------------------------------
# Procedência
# ---------------------------------------------------------------------------
#
# Origem: laboratório de metabolômica parceiro.
# Estudo: HTLV em plasma humano, 68 injeções = 61 pacientes + 7 QCs, quatro
# corridas analíticas independentes combinando duas colunas cromatográficas
# (HILIC, C18) e dois modos de ionização (positiva, negativa).
#
# Três famílias de arquivo:
#
#   input/measurements_{modo}.csv    exportação convertida do Progenesis QI.
#                                    Matriz transposta: features em linhas.
#   input/Sample_List_{modo}.csv     ordem de injeção. ÚNICO lugar onde a
#                                    ordem temporal existe; sem esse join não
#                                    há correção de deriva possível.
#   output/{modo}_toserrf.csv        formato intermediário gerado pelo
#                                    pipeline R, entrada do motor SERRF.
#   output/{modo}_toserrf - normalization result/
#                                    saídas do pipeline R usadas como golden
#                                    files: QC - RSD.csv e as três matrizes
#                                    normalized by - {none,batch_loess,SERRF}.
#
# Layout físico. `data/reference/input` e `data/reference/output` são symlinks
# para `serrf/data/`, não cópias. `serrf/script/serrf.R` tem `serrf/data/output/`
# fixo nas linhas 78 e 160, então mover os arquivos quebraria o pipeline R de
# referência, que precisa poder ser reexecutado. Copiar duplicaria 69 MB e
# criaria duas cópias que divergem em silêncio.
#
# Pendências com o laboratório sobre estes arquivos: os rótulos de grupo
# divergentes, e o arquivo NEG_Lipidomica_comcorreção_SERRF.csv, presente apenas
# no neg_c18 e sem descrição.

log()  { printf '%s\n' "$*"; }
fail() { printf 'ERRO: %s\n' "$*" >&2; exit 1; }

[ -e "$REFERENCE/input" ]  || fail "data/reference/input ausente. Recrie o symlink: ln -sfn ../../serrf/data/input data/reference/input"
[ -e "$REFERENCE/output" ] || fail "data/reference/output ausente. Recrie o symlink: ln -sfn ../../serrf/data/output data/reference/output"

# Lista estável e ordenada, para que o manifesto seja reprodutível.
# -L segue os symlinks; sem isso, find não desce em input/ e output/.
collect() {
  find -L "$REFERENCE" -type f \( -name '*.csv' -o -name '*.png' \) \
    | sed "s|^$REFERENCE/||" \
    | LC_ALL=C sort
}

if [ "${1:-}" = "--write" ]; then
  log "Gerando manifesto em scripts/reference_data.sha256"
  {
    echo "# Manifesto de integridade dos dados de referência."
    echo "# Gerado por scripts/fetch_reference_data.sh --write"
    echo "# Caminhos relativos a data/reference/. Não editar à mão."
  } > "$MANIFEST"
  count=0
  while IFS= read -r relative; do
    digest="$(sha256sum "$REFERENCE/$relative" | cut -d' ' -f1)"
    printf '%s  %s\n' "$digest" "$relative" >> "$MANIFEST"
    count=$((count + 1))
  done < <(collect)
  log "  $count arquivos registrados."
  exit 0
fi

[ -f "$MANIFEST" ] || fail "manifesto ausente. Gere com: scripts/fetch_reference_data.sh --write"

log "Verificando integridade contra scripts/reference_data.sha256"
ok=0; bad=0; missing=0
while IFS= read -r line; do
  # O `\r` do CRLF desliga a verificação inteira em silêncio. Num clone Windows com
  # `core.autocrlf=true` o manifesto chega com CRLF, e sem esta linha o `\r` vai parar ao fim do
  # caminho: toda entrada vira `AUSENTE`, `bad` fica em 0 para sempre, e a única coisa que este
  # script existe para acusar, checksum divergente, nunca chega a ser comparada. Medido num clone
  # Windows: `0 íntegros, 0 divergentes, 33 ausentes.` com os 33 arquivos no lugar e íntegros.
  # Os três consumidores Python do mesmo manifesto usam `.strip()` e não sofrem disto.
  line="${line%$'\r'}"
  case "$line" in ''|\#*) continue ;; esac
  expected="${line%%  *}"
  relative="${line#*  }"
  path="$REFERENCE/$relative"
  if [ ! -f "$path" ]; then
    printf '  AUSENTE   %s\n' "$relative"; missing=$((missing + 1)); continue
  fi
  actual="$(sha256sum "$path" | cut -d' ' -f1)"
  if [ "$actual" = "$expected" ]; then
    ok=$((ok + 1))
  else
    printf '  DIVERGE   %s\n     esperado %s\n     obtido   %s\n' "$relative" "$expected" "$actual"
    bad=$((bad + 1))
  fi
done < "$MANIFEST"

log "  $ok íntegros, $bad divergentes, $missing ausentes."

# Arquivo ausente não é erro deste script: o teste vira skip, e quem decide isso
# é o resolvedor em tests/conftest.py. Checksum divergente,
# sim, é erro — significa que o dado mudou sob os pés dos golden tests.
[ "$bad" -eq 0 ] || fail "$bad arquivo(s) com checksum divergente."
log "OK."
