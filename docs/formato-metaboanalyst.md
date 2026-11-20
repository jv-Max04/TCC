# O formato de entrada do MetaboAnalyst — a especificação citada

Este documento existe para ser o **referente** do `[CA-11.6]`. O critério diz que a exportação é
conforme à especificação **publicada** do formato, verificada por mecanismo; sem uma citação estável,
"conforme" não tem contra o quê ser conferido, e o critério fica satisfeito por qualquer coisa — que é
AR66, e é a razão pela qual o `[CA-11.2]` foi retirado.

**Procedência, e ela é o conteúdo desta seção.**

| Campo | Valor |
|---|---|
| Fonte | `https://www.metaboanalyst.ca/docs/Format.xhtml` |
| Data de acesso | **2026-08-04** |
| Método | busca automatizada da página, com extração mediada por modelo, **duas passadas independentes** |
| Versão da ferramenta | MetaboAnalyst 6.0 (Pang et al., *Nucleic Acids Research* 52(W1), 2024 — §11 das referências) |

**O método é declarado porque ele limita o que este documento prova.** As frases abaixo foram
extraídas por consulta automatizada, não por leitura direta do HTML por pessoa.

**Cada elemento declara quantas passadas o cobriram, e isso é dimensão separada do `verbatim`.**
Agregar as duas foi um defeito real desta redação: a primeira versão marcou os seis como *"concordante
nas duas passadas"*, e o `[F-6]` teve **uma** — a segunda consulta não perguntou por aquele tópico.
Afirmar concordância onde só houve uma medição é exatamente a classe de erro que este documento existe
para não cometer, e foi o portão de `tests/test_metaboanalyst_spec_citation.py` que a pegou, na
primeira execução.

| Elemento | Status | Passadas |
|---|---|---|
| `[F-1]` | `derivado` | duas, **discordantes** — daí o status |
| `[F-2]` a `[F-5]` | `verbatim` | duas, concordantes |
| `[F-6]` | `verbatim` | **uma** — sem concordância a alegar |

**Por que verbatim e não paráfrase.** Duas razões, e a segunda é a que decide. A paráfrase permite ao
mecanismo verificar uma regra que a especificação não tem — e um portão que reprova por regra
inventada é pior que portão ausente, porque parece rigor. E página de terceiro **caduca**: o texto
citado aqui é o referente estável, e a data de acesso é o que torna a caducidade detectável quando
alguém remedir.

---

## Os elementos obrigatórios

Cada item tem um identificador `[F-N]` para que o mecanismo do `[CA-11.6]` os cite um por um, em vez
de reescrevê-los em código. Divergência entre citação e mecanismo fica impossível por construção.

### `[F-1]` Tipo de arquivo e separador

**Status.** `derivado`
**Passadas.** `duas` — e elas **discordaram**, que é a razão do status
**Verificado pelo mecanismo.** `sim`

A página descreve os formatos aceitos como CSV e delimitado por tabulação, e menciona também mzTab
2.0-M e arquivo compactado. A primeira passada devolveu *"Files must be comma-separated (.csv) or
tab-delimited (.txt)"*; a segunda **não reproduziu essa frase**, devolvendo em lugar dela *"These two
formats are used for concentration data, peak intensity table, and MS/NMR spectral bins."*

**As duas passadas discordaram, então este item NÃO é citação.** O que fica é a derivação: um arquivo
com extensão `.csv` é separado por vírgula. Isso é convenção do formato, não frase da especificação.

**Consequência vinculante para o mecanismo:** o `[CA-11.6]` pode conferir que o arquivo que **nós**
escrevemos é vírgula-separado, porque é a nossa escolha de extensão. Não pode reprovar um arquivo
alheio por usar tabulação, porque a especificação aceita as duas. Confundir os dois faria o portão
reprovar dado conforme.

### `[F-2]` Valores numéricos e positivos, e a codificação de ausente

**Status.** `verbatim`
**Passadas.** `duas` — concordantes
**Verificado pelo mecanismo.** `sim`

> "Data values (concentrations, bins, peak intensities) should contain only numeric and positive
> values (using empty or NA for missing values)."

**O que isto decide no nosso lado (KTD-3 do plano):** 15,8% das células da matriz do `pos_hilic` são
zero, e zero não é positivo. A convenção do próprio formato para ausente é **célula vazia ou `NA`**, e
o fato de domínio deste projeto diz que zero é não-detecção e não medida — o equipamento deixa a
célula vazia quando não detecta. Então o export escreve **vazio**, e registra a contagem de células
convertidas no relatório.

**A consequência vai declarada e não é neutra:** a partir daí a imputação daquelas células é feita
pela política do MetaboAnalyst, não pela nossa. A contagem é informação de procedência.

### `[F-3]` Número sem espaço

**Status.** `verbatim`
**Passadas.** `duas` — concordantes
**Verificado pelo mecanismo.** `sim`

> "In addition, there should not be spaces between numbers. For instance, 1 600 should be formatted as
> 1600, if not the value will get read as 1."

**A cláusula final é o que torna este item interessante:** o modo de falha declarado pela própria
especificação é **silencioso** — `1 600` é lido como `1`, não como erro. É a forma de defeito que este
projeto persegue em todo lugar, e aqui ela está documentada pelo terceiro.

### `[F-4]` Nome de amostra e de feature

**Status.** `verbatim`
**Passadas.** `duas` — concordantes
**Verificado pelo mecanismo.** `sim`

> "Both sample or feature names must be unique and consist of a combination of common English letters,
> underscores and numbers for naming purpose. Latin/Greek letters are not supported."

**Medido sobre os arquivos reais em 2026-08-04, e é o achado que mais muda a implementação:**

| População | Viola `[F-4]` | Observação |
|---|---|---|
| features do `pos_hilic` | **6.035 de 6.035** | `2.25_810.7739m/z` tem ponto e barra |
| amostras do `pos_hilic` | **0 de 68** | `HTLV_HILIC_PLASMA_MSe_POS_QC1` já é conforme |
| grupos | **0 de 5** | `ATLL`, `Asymptomatic`, `HAMTSP`, `NoHTLV`, `QC` |

Sanitizar feature para `[A-Za-z0-9_]` dá **6.035 nomes distintos — zero colisões**. Isso é
**observação sobre estes quatro arquivos**, não invariante (regra 1): matriz nova pode colidir, e por
isso a verificação de unicidade fica no código em vez de na suposição.

### `[F-5]` Posição do rótulo de classe

**Status.** `verbatim`
**Passadas.** `duas` — concordantes
**Verificado pelo mecanismo.** `sim`

Amostras em **linhas**:

> "The class label is in the second column."

Amostras em **colunas**:

> "The class label must be in the second row."

**O projeto exporta amostras em linhas**, que é a orientação interna canônica — a transposição
acontece uma vez, na ingestão, e nunca depois. Então o rótulo de classe vai na **segunda coluna**.

### `[F-6]` Conteúdo do rótulo de classe

**Status.** `verbatim`
**Passadas.** `uma` — a segunda consulta **não perguntou** por este tópico,
então não há concordância a alegar aqui
**Verificado pelo mecanismo.** `nao` — só o caso não emparelhado nos alcança, e ele é conforme por construção

Para dado **não emparelhado**:

> "numeric (i.e. 0/1) or character (i.e. Healthy/Disease)"

Para comparação **emparelhada**:

> "class labels are required to be the numeric integers between -1 and -n/2 and between 1 and n/2"

**Só o caso não emparelhado nos alcança.** Os cinco grupos deste estudo são de caractere e conformes.
O caso emparelhado fica registrado por completude da citação e **não** é verificado pelo mecanismo:
não há dado emparelhado neste projeto, e um portão sobre ramo que nunca dispara é portão que ninguém
vê reprovar.

---

## O que este documento NÃO estabelece

- **Não** estabelece que o MetaboAnalyst aceita de fato um arquivo nosso. Aceitação exigiria enviar
  dado clínico humano identificável a serviço web de terceiro, e é por isso que o `[CA-11.2]` foi
  retirado. Conformidade de formato e aceitação de fato são afirmações diferentes.
- **Não** cobre requisito de cabeçalho além da posição do rótulo de classe. A primeira passada
  registrou que requisitos explícitos de cabeçalho não são declarados na página; o mecanismo não
  inventa nenhum.
- **Não** cobre os outros formatos de entrada que a página menciona — mzTab 2.0-M e arquivo
  compactado. O projeto exporta tabela de intensidade, e só.
- **Não** é a especificação do MetaboAnalystR, que é o pacote R. Este documento é sobre o **formato de
  arquivo**; a invocação do pacote é o `[CA-11.7]`. A predição de vias que o executável faz é
  outra rota — o algoritmo do mummichog, em processo local e a partir dos picos, sem lista
  identificada — e não passa por este formato. O executável enriquece também a lista
  identificada no Progenesis, pelo próprio MetaboAnalystR num R portátil embarcado, como segundo
  modo da aba de vias, em beta; esse caminho recebe a lista por JSON
  (`r_service/vias_entrypoint.R`) e também não passa por este formato.
