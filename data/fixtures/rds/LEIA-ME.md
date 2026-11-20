# Fixtures do leitor restrito de RDS

Insumo de `tests/unit/test_rds_restrito.py`, que testa `src/refino/io/rds_restrito.py`: o
leitor do formato de serialização do R que o modo externo do enriquecimento usa no lugar do
`unserialize`, e o escritor que monta a carga enviada.

## O que são

Cinco objetos R, cada um em dois formatos:

| Arquivo | O que contém | O que o leitor faz |
|---|---|---|
| `aceito.ser`, `aceito.rds` | uma lista com uma matriz real 2×6 com `dim` e `dimnames`, uma lista nomeada de textos, textos com `NA`, acento e `β`, lógicos e inteiros com `NA`, a sequência `1:10`, um `NULL` e um fator | lê, com os valores certos |
| `com_funcao.ser`, `com_funcao.rds` | `list(ok = 1, f = function(x) x + 1)` | recusa: *função (CLOSXP)* |
| `com_ambiente.ser`, `com_ambiente.rds` | `list(ok = 1, e = new.env())` | recusa: *ambiente (ENVSXP)* |
| `com_formula.ser`, `com_formula.rds` | `list(ok = 1, fo = y ~ x)` | recusa: *linguagem (LANGSXP)* |
| `embrulho.ser`, `embrulho.rds` | uma lista com seis embrulhos ALTREP `wrap_*` da base — real, inteiro com `NA`, texto, lógico com `NA`, lista e real com nomes —, gravados com `wrap_meta` | lê, com o vetor de dentro e os atributos |

- **`.ser`** é `serialize(x, NULL)` gravado cru: a forma em que o servidor do MetaboAnalyst devolve a
  resposta.
- **`.rds`** é `saveRDS(x, compress = TRUE)`: a forma do `tosend.rds` que o pacote envia, com `gzip`.

Nos dez arquivos o R 4.4.3 gravou o **formato 3** de serialização, e o `.rds` descomprimido é
byte a byte o `.ser` do mesmo objeto (conferido em 02/10/2026: quatro de quatro, e o `embrulho`
no mesmo dia, ao ser acrescentado). No `aceito`, o
`1:10` sai como ALTREP `compact_intseq` e não expandido — é por isso que ele exercita o ramo ALTREP
do leitor —, o `β` sai com a marca UTF-8, e o símbolo `names` é gravado uma vez e reaparece por
referência. O teste `test_as_fixtures_cobrem_os_caminhos_que_os_testes_de_leitura_afirmam` confere
essas três formas nos bytes, para que os testes de valor não passem sobre uma forma que não exercita
o ramo. O escritor deste projeto grava o **formato 2**, então os dois cabeçalhos aceitos ficam
exercitados.

**Nenhuma resposta real do servidor está aqui, e nenhuma deve entrar.** Ela traz dados derivados do
KEGG, cuja licença não permite redistribuir. O que precisa da forma da resposta é montado nos
testes com `escreve_rds` e nomes de via inventados.

**Por que o `embrulho` existe (revisão de 02/10/2026).** O estado de um `wrap_*` é o PAR
PONTILHADO `CONS(x, meta)` de `src/main/altclasses.c`, e não uma pairlist: a cauda é o vetor de
metadados, sem o `NULL` que fecha a lista. O teste de embrulho anterior montava o estado à mão,
como lista, numa forma que o R nunca grava — e o leitor recusava todo embrulho real, inclusive o
que `sort()` produz. O teste `test_os_embrulhos_que_o_r_grava_sao_lidos` lê esta fixture e confere
o par nos bytes.

## Como foram geradas

Pelo `gera.R` desta pasta, com o `Rscript` da árvore R portátil — a que
`scripts/monta_r_portatil.ps1` monta, R 4.4.3 (2025-02-28). O comentário do `gera.R` registra a
execução de 01/10/2026. A partir da raiz do repositório, no Git Bash:

```bash
"$USERPROFILE/_refino_env/r-portatil/arvore/bin/Rscript.exe" --vanilla \
    data/fixtures/rds/gera.R data/fixtures/rds
```

**A geração é reprodutível byte a byte, e isso foi medido em 02/10/2026:** rodar o mesmo comando
com destino numa pasta temporária produziu os oito arquivos com `sha256` igual ao dos versionados,
oito de oito.
No mesmo dia, ao acrescentar o `embrulho` ao `gera.R`, a nova corrida sobre esta pasta regravou
os oito anteriores com o mesmo `sha256`.

## O R lê o que `escreve_rds` grava

O teste de ida e volta do Python (`escreve_rds` → `le_rds`) prova que os dois lados deste módulo
concordam **entre si**. Não prova que o R lê a carga — e a carga existe para o R do servidor ler.
Essa metade exige o R e por isso não roda na suíte; ela foi medida à parte.

**Medido em 02/10/2026, com o `Rscript` acima:**

1. em Python, `le_rds` sobre o `aceito.ser` e `escreve_rds` sobre o objeto lido, gravando 964 bytes
   no formato 2;
2. no R, `unserialize` desses bytes e `identical()` contra `readRDS("aceito.rds")`: **`TRUE`**;
3. o par causal: os mesmos bytes com o **bit de objeto** (`1 << 8`) apagado no cabeçalho do fator —
   a única ocorrência, conferida por contagem antes da troca — dão **`FALSE`**.

O passo 3 é o que mostra que a comparação separa: sem o bit, o R lê o atributo `class` e não trata o
vetor como objeto. O comentário em `_escreve_item` registra a mesma observação, feita com um fator em
01/10/2026; o teste `test_o_escritor_marca_objeto_so_quando_ha_classe` guarda a forma dos bytes que
ela exige.

**O R desserializa aqui só bytes locais:** as fixtures desta pasta e o que o escritor deste projeto
gravou a partir delas. O que vem do servidor é lido pelo leitor restrito, e não pelo `unserialize`
nem pelo `readRDS` do R — é para isso que ele existe.
