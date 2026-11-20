# Os métodos de processamento do MetaboAnalyst — a citação e o confronto

Este documento existe para dar **referente** à afirmação de que o pré-processamento desta
ferramenta é o do MetaboAnalyst. Sem ele, *"os métodos estão de acordo"* não tem contra o quê ser
conferido, e a afirmação fica satisfeita por qualquer semelhança — que é AR66, e é a mesma ausência
que motivou `docs/formato-metaboanalyst.md` a existir.

**Ele é irmão do `formato-metaboanalyst.md` e cobre outra coisa.** Aquele é sobre o **formato de
arquivo** que o MetaboAnalyst aceita; este é sobre os **métodos** que ele oferece nas telas
*Processing* e *Normalization*. Conformidade de formato e equivalência de método são afirmações
diferentes, e antes de 07/09/2026 só a primeira tinha documento.

> **O que este documento NÃO cobre:** a comparação da predição de vias com a aba *MS Peaks to
> Pathways* do MetaboAnalyst. Ela existe, e está no capítulo 10 de `docs/MEDICOES-da-correcao.html`.

**Procedência, e ela é o conteúdo desta seção.**

| Campo | Valor |
|---|---|
| Fonte primária | `https://raw.githubusercontent.com/xia-lab/MetaboAnalystR/master/R/general_norm_utils.R` |
| Fonte secundária | `https://raw.githubusercontent.com/xia-lab/MetaboAnalystR/master/R/general_proc_utils.R` |
| Fonte terciária | `https://www.metaboanalyst.ca/resources/vignettes/Introductions.html` |
| **Fonte quaternária** | **as TELAS VIVAS de `www.metaboanalyst.ca`, módulo *Statistical Analysis [one factor]*: `/Secure/process/SanityCheck.xhtml`, `FilterView.xhtml`, `ProcMissing.xhtml`, `DataEditView.xhtml` e `NormalizationView.xhtml`** |
| Data de acesso | **2026-09-07** (fontes 1 a 3) · **2026-09-09** (as telas) |
| Método | fontes 1 a 3: busca automatizada das páginas, com extração mediada por modelo. **Telas: sessão HTTP própria com o dataset de exemplo `mspkint` do próprio site, e o HTML de cada tela lido e transcrito** |
| Versão da ferramenta | MetaboAnalyst 6.0 / MetaboAnalystR (Pang et al., *Nucleic Acids Research* 52(W1), 2024); rodapé do site em 09/09/2026: *"Xia Lab @ McGill (last updated 2026-09-06)"* |

**O método é declarado porque ele limita o que este documento prova.** As frases abaixo foram
extraídas por consulta automatizada ao **código-fonte** do MetaboAnalystR, não por leitura da
interface por pessoa.

**A PASSADA PELA INTERFACE VIVA ACONTECEU EM 09/09/2026.** As telas foram alcançadas por **sessão
HTTP própria** — carregando o dataset de exemplo `mspkint` do próprio site e lendo o HTML de cada
tela de processamento.

**O que a terceira fonte muda, e ela não é redundante:** o código do MetaboAnalystR diz o que a
função **aceita**; a tela diz o que o usuário **vê e escolhe**, com os rótulos e os valores padrão.
Os dois divergem em dois lugares — `[N-1]` e `[N-2]` —, e nos dois a tela é que governa, porque é
ela que define o que o laboratório de facto usa.

| Elemento | Status | Passadas |
|---|---|---|
| `[N-1]` | `verbatim` | três — as duas primeiras discordaram e a TELA desempatou |
| `[N-2]` | `verbatim` | duas — e a tela ACRESCENTA dois métodos ao que o código declara |
| `[N-3]` | `verbatim` | duas, concordantes — com uma precisão de eixo que só a tela dá |
| `[N-4]` | `derivado` | duas — a regra derivada CONFIRMOU-SE em duas observações |
| `[N-5]` | `verbatim` | três, concordantes |
| `[N-6]` | `verbatim` | três, concordantes |
| `[N-7]` | `verbatim` | três, concordantes |

**A tabela não leva negrito, e isso é contrato e não estilo.** O portão
`tests/test_metaboanalyst_metodos_citacao.py` casa a coluna `Status` por backtick e compara a
coluna `Passadas` por prefixo; negrito à frente do valor quebra as duas leituras.

**Um dos sete itens** declara `uma` passada ou status `derivado` — é o `[N-4]`, e o número é
derivado pelo portão a partir das contagens, não escrito aqui à mão.

**Por que verbatim e não paráfrase.** A paráfrase permite ao mecanismo verificar uma regra que a
fonte não tem — e um portão que reprova por regra inventada é pior que portão ausente, porque parece
rigor. E página de terceiro **caduca**: o texto citado aqui é o referente estável, e a data de acesso
é o que torna a caducidade detectável quando alguém remedir.

---

## Os elementos

Cada item tem um identificador `[N-k]` para que o mecanismo os cite um por um, em vez de reescrevê-los
em código. Divergência entre citação e mecanismo fica impossível por construção.

### `[N-1]` O valor de substituição padrão dos ausentes

**Status.** `verbatim` — **a TELA desempatou**
**Passadas.** `três`

A primeira consulta devolveu que o padrão substitui os ausentes por *"half of the smallest positive
value in the original data"*. A segunda, sobre a vinheta de introdução, devolveu *"1/5 of min
positive values"*. **As duas discordaram sobre a fração**, e por isso este item não era citação.

**A terceira passada é a tela `ProcMissing.xhtml`, e ela diz UM QUINTO, duas vezes na mesma
página.** No rótulo do botão de rádio que vem marcado por padrão:

> LoD (1/5 of the min. positive value)

E no texto explicativo acima dele:

> The default method LoD (limit of detection) will replace all the missing values with 1/5 of the
> minimum positive values of individual features as proxies of their detection limits.

**A fração é 1/5, e a primeira fonte estava errada.** Não é surpresa de que lado: `master` do
GitHub é o código de desenvolvimento, e a tela é o que o servidor executa — duas instâncias do
mesmo referente, com a que o usuário vê governando.

**O que isto decide no nosso lado.** `one_fifth_min` **é** o padrão do MetaboAnalyst, e agora isso
é afirmável. `col_min` continua legítimo e deixa de ser "a outra leitura da fração": ele é o `min`
deles, que a tela oferece como opção **separada**, no grupo dos métodos univariados.

**E o nosso padrão já era `one_fifth_min` por escolha do laboratório, não por esta leitura** — a
coincidência é bem-vinda e não é evidência: o `LEIA-ME-das-saidas.txt` declara *"só a imputação age
(1/5 do mínimo)"* desde antes desta conferência.

### `[N-2]` Os métodos de imputação

**Status.** `verbatim`
**Passadas.** `duas` — **e a tela ACRESCENTA dois métodos ao que o código declara**

Do `@param method` de `ImputeMissingVar`:

> `"min"` – replacement based on the minimum value of each feature; `"mean"` – replacement based on
> the mean value of each feature column; `"median"` – replacement based on the median value of each
> feature column; `"KNN"` – k-nearest neighbour imputation; `"PPCA"` – probabilistic PCA method;
> `"BPCA"` – Bayesian PCA method; `"svdImpute"` – Singular Value Decomposition imputation

**A segunda passada é a tela `ProcMissing.xhtml`, e a lista dela é MAIOR.** Os dez rótulos dos
botões de rádio, verbatim e na ordem da tela, agrupados como ela os agrupa:

> **Left-censored data estimation** — `LoD (1/5 of the min. positive value)` · `quantile regression
> (QRILC)`
> **Univariate statistical methods** — `mean` · `median` · `min`
> **Multivariate statistical methods** — `KNN (feature-wise)` · `KNN (sample-wise)` · `PPCA` ·
> `BPCA` · `SVD Impute`

E do texto da mesma tela, mais um método e duas constantes:

> ...and **missForest** based on random forest
>
> Note k is set to 10 for KNN; the max missing values is set to 30000 for missForest due to its
> computational intensive nature.

**DUAS CORREÇÕES ao que este item dizia, e a segunda era afirmação falsa:**

1. **`QRILC` não aparece no `@param` e aparece na tela** — *Quantile Regression Imputation of
   Left-Censored data*. Era ausência do catálogo, não escolha de escopo.
2. **`missforest` NÃO é só nosso.** Este item afirmava *"`missforest` é **nosso** e não tem
   contraparte lá"*, e a tela cita `missForest` pelo nome, com teto declarado de 30.000 ausentes. A
   afirmação era falsa, e o mecanismo que a produziu é conhecido: o `@param` do código não o lista, e
   concluir ausência de uma fonte que não o menciona é a asserção de ausência satisfeita por
   vacuidade.

**E há uma divergência de PARÂMETRO que só a tela revela.** O `k` do KNN é **10 fixo** do lado deles;
o nosso é `round(sqrt(n_features))`, que com 2.184 features dá **47**. Não é o mesmo método com outro
nome — é outra vizinhança.

**A distinção feature-wise / sample-wise também é da tela, e ela nos classifica.** O nosso `knn`
roda o `KNNImputer` sobre `[amostras, features]`, então acha vizinhos entre **linhas**: é o
`KNN (sample-wise)` deles. **O feature-wise — o primeiro dos dois na tela — faltava, e entrou em
10/09/2026 como `knn_features`**, sobre a transposta e com o nome próprio que R2 exige: o
existente manteve o dele para não reinterpretar execução já gravada.

**A vizinhança dos dois NÃO é a mesma, e isso é consequência da regra e não escolha.** A §7 fixa
`round(sqrt(n_colunas))`, e transpor troca o que são as colunas: o sample-wise conta features e o
feature-wise conta amostras, que neste estudo são 61 e dão **8**. É por isso que a tabela de
recuperação corre um braço com o `k` igual nos dois — com transposição e vizinhança a diferirem ao
mesmo tempo, nenhuma diferença seria atribuível. O `k` de **10 fixo** deles continua a não
ser nem um nem outro.

**OS CINCO MÉTODOS DO BLOCO DE 10/09/2026 ENTRARAM, e com eles a lista de ausências encolhe.**

| O que faltava | Entrou como | Onde |
|---|---|---|
| `KNN (feature-wise)` | `knn_features` | `preprocessing/missing.py`, sobre a transposta |
| `SamplePQN` | `sample_reference` | `preprocessing/normalization.py`, com o nome em `normalization_reference` |
| `CompNorm` | `feature_reference` | idem, dividindo pela feature nomeada |
| `SpecNorm` | `supplied_factors` | idem, com o mapa em `normalization_factors` |
| o segundo eixo de filtro | `ranking_filter_2` e as duas frações retidas | `preprocessing/filtering.py`, com nome de etapa próprio |

**A diferença de `k` do KNN continua**, e agora ela é dupla: o deles é 10 fixo, o nosso
sample-wise é `round(sqrt(n_features))` e o nosso feature-wise é `round(sqrt(n_amostras))` — que
neste estudo dá **8**. Não é o mesmo método com três nomes; são três vizinhanças.

**O que continua fora:** `PPCA`, `BPCA`, `svdImpute`, `QRILC` — decisão de prioridade e não de
viabilidade, medida em 09/09/2026 — e a subtração de branco.

**O que isto decide no nosso lado.** Entraram `mean`, `col_min` (o `min` deles) e `exclude`.
**Ficam fora, e o motivo de 07/09 enfraqueceu:** `PPCA`, `BPCA` e `svdImpute` foram excluídos por
*"não têm equivalente em `scikit-learn`"*.
Medido em 09/09/2026: `svdImpute` e `PPCA` são implementáveis sobre o `PCA` do `scikit-learn` sem
dependência nova, e o `QRILC` sobre `numpy.polyfit` mais `scipy.stats.truncnorm`. **Só o `BPCA`
continua sendo porte de dias.** A exclusão passa a ser de prioridade e não de viabilidade.

### `[N-3]` Os métodos de filtragem por posto

**Status.** `verbatim`
**Passadas.** `duas`, concordantes — **com uma precisão de eixo que só a tela dá**

Do `@param var.filter` de `FilterVariable`:

> `"rsd"` – relative standard deviation; `"nrsd"` – non-parametric relative standard deviation;
> `"mean"` – mean-based ranking; `"sd"` – standard deviation; `"mad"` – median absolute deviation;
> `"iqr"` – interquartile range

**A segunda passada é a tela `FilterView.xhtml`, e ela mostra DOIS EIXOS SEPARADOS, não um.** Os
rótulos verbatim, com o percentil próprio de cada eixo:

> **Low-variance filter** — `Interquantile range (IQR)` [SIC, no original] · `Standard deviation
> (SD)` · `Median absolute deviation (MAD)` · `Relative standard deviation (RSD = SD/mean)` ·
> `Non-parametric relative standard deviation (MAD/median)` — com *Percentage to filter out*
> **40%** neste dado
> **Low-abundance filter** — `Mean intensity value` · `Median intensity value` — com *Percentage to
> filter out* **0%**

E a tela nomeia **quatro** grupos de variável não informativa, não dois: acrescenta o
*Low-quality filter* (contaminação por BLANK, ou ausentes demais) e o *Low-repeatability filter*
(RSD nos QCs, com *"suggested threshold: 20% for LC-MS and 30% for GC-MS"*).

**A precisão que isto traz, e ela corrige a leitura anterior deste item.** Dizia *"os sete — os seis
acima mais `median`"*, como se fossem um eixo de sete opções. São **dois eixos**: cinco medidas de
variância e duas de abundância, cada um com o seu percentil, aplicáveis **em sequência**. Do nosso
lado os sete valores existem em `RankingFilter`, mas `PipelineSpec` tem **um** campo de filtro por
posto — então hoje é possível escolher qualquer um dos sete e **impossível** encadear variância com
abundância, que é o que a tela faz por desenho.

**Não confundir com o nosso `rsd_cutoff`**, que é outra etapa: aquele mede o RSD **dos QCs** e
reprova a feature acima do corte; este ordena por uma estatística sobre **todas** as amostras e
mantém a fração superior. Rótulo parecido, população diferente. **A tela confirma a distinção**, ao
pôr o RSD de QC num grupo próprio — *Low-repeatability* — separado do *Low-variance*.

### `[N-4]` A fração retida por número de variáveis

**Status.** `derivado`
**Passadas.** `duas` — **e a segunda é OBSERVAÇÃO da regra em funcionamento**

A fonte descreve a regra em prosa e não em bloco citável: menos de 250 variáveis removem 5%; de 250
a 500 removem 10%; de 500 a 1.000 removem 25%; acima de 1.000 removem 40%.

**Marcado `derivado` porque é transcrição de números e não de frase.** Uma citação em bloco daria a
impressão de que a redação foi conferida palavra a palavra, e o que foi conferido foram os quatro
pares de limiar e percentagem.

**A segunda passada confirma a regra por dois pontos, e o valor não é citado — é LIDO do controle:**

| Features no dado | *Percentage to filter out* que a tela trouxe | A faixa da regra | Confere? |
|---|---|---|---|
| **409** | **10%** | 250 a 500 → 10% | sim |
| **2.174** | **40%** | acima de 1.000 → 40% | sim |

**Isto é mais forte que uma citação, e vale dizer por quê:** citação prova que a regra está escrita;
dois pontos em faixas diferentes provam que ela está **aplicada**. E resolve uma pergunta que a
prosa deixava aberta — o valor é **recalculado a cada dado**, não é constante da interface. Qualquer
relatório que diga *"usei o padrão do filtro de variância"* é irreprodutível sem dizer quantas
features havia.

**O que isto decide no nosso lado.** `fracao_retida_por_tamanho` existe e implementa a regra. As
duas observações são a primeira evidência externa de que ela não está só escrita, e a segunda ponta
— 2.174 features dando 40% — foi colhida sobre **a nossa própria matriz** `neg_c18` corrigida por
SERRF, o que faz dela o caso mais próximo do uso real que este documento tem.

### `[N-5]` A normalização por amostra

**Status.** `verbatim`
**Passadas.** `três` — concordantes

**A terceira passada é a tela `NormalizationView.xhtml`, e ela resolve a ambiguidade dos nomes.** Os
oito rótulos do grupo *Sample normalization*, verbatim:

> `None` · `Sample-specific normalization (i.e. weight, volume)` · `Normalization by sum` ·
> `Normalization by median` · `Normalization by a reference sample (PQN)` · `Normalization by a
> pooled sample from group (group PQN)` · `Normalization by reference feature` · `Quantile
> normalization (suggested only for > 1000 features)`

**É isto que nomeia as três variantes de PQN**, que era justamente a ambiguidade que fez o nosso
`pqn` passar a `pqn_group`: *por amostra de referência*, *por grupo agrupado*, e a variante sem
referência do código (`ProbNormT`). O nosso é o do meio, e agora o nome tem contra o quê ser
conferido.

Do `@param rowNorm` de `Normalization`:

> "QuantileNorm" for Quantile Normalization, "ProbNormT" for Probabilistic Quotient Normalization
> without using a reference sample, "ProbNormF" for Probabilistic Quotient Normalization based on a
> reference sample, "CompNorm" for Normalization by a reference feature, "SumNorm" for Normalization
> to constant sum, "MedianNorm" for Normalization to sample median, and "SpecNorm" for Normalization
> by a sample-specific factor.

E as contas, do corpo do módulo:

> `SumNorm`: `1000*x/sum(x, na.rm=T)` — `MedianNorm`: `x / median(x, na.rm=TRUE)` — `ProbNorm`:
> `x / median(x/ref.smpl)`

**O que isto decide no nosso lado, e são duas coisas.** O `sum` passou a carregar o fator **1000**,
que é a constante deles — o nosso dividia pela soma e mais nada, e a §Fase 5 foi emendada junto,
porque o teste obrigatório que afirmava *"a linha soma 1"* passou a afirmar a constante. E o `pqn`
passou a `pqn_group` **sem mudar a conta**: `pqn` sozinho não diz qual das três variantes é, e a
nossa sempre foi o PQN contra um **grupo** de referência, com o grupo fixo nos QCs.

**Ficaram fora:** `CompNorm` (feature de referência) e `SpecNorm` (fator por amostra), porque os
dois exigem **entrada nova do usuário** — qual feature, quais fatores — e portanto campo novo em
`PipelineSpec` e controle novo nas duas telas.

### `[N-6]` A transformação

**Status.** `verbatim`
**Passadas.** `três` — concordantes

**A terceira passada é a tela, e os rótulos do grupo *Data transformation* são cinco:**

> `Log transformation (base 10)` · `(base 2)` · `Square root transformation (square root of data
> values)` · `Cube root transformation (cube root of data values)` · `Variance stabilizing
> normalization (data-adaptive transformation)`

A tela acrescenta a ressalva de que *"a simple mathematical approach is used to deal with negative
values in log and square root"*, que é a forma estabilizada citada abaixo — e é a confirmação, pela
interface, de que os dois não usam o logaritmo comum.

Do `@param transNorm` e do corpo:

> "LogNorm" for Log Normalization, and "CrNorm" for Cubic Root Transformation.

> `LogNorm`: `log10((x + sqrt(x^2 + min.val^2))/2)` onde `min.val = min(abs(data[data!=0]),
> na.rm=T)/10` — `Log2Norm`: o mesmo em base 2 — `SrNorm`: `((x + sqrt(x^2 + min.val^2))/2)^(1/2)` —
> `CrNorm`: `abs(data)^(1/3)` com o sinal reposto

**O que isto decide no nosso lado, e o achado é que já estávamos lá.** O nosso `log10` e `log2`
**já eram** essa fórmula, e por um caminho que vale registrar: ela aparece igual em
`serrf/script/tools.R:130-134`, dentro de `preprocessing()`. O laboratório copiou-a do MetaboAnalyst
e nós transcrevemos do laboratório.

O que **mudou** foi o `sqrt`: era `sign(x)·sqrt(|x|)` e passou à forma estabilizada do `SrNorm`. A
justificativa antiga argumentava contra a raiz **comum**, que dá `nan` sobre negativo; nunca
argumentou contra a forma estabilizada, que também não dá. Era **incompleta**, não errada.

`cbrt` já coincidia com o `CrNorm`. **Ficou fora** o `VsnNorm`, que é estabilização de variância por
máxima verossimilhança do `limma`.

### `[N-7]` O escalonamento

**Status.** `verbatim`
**Passadas.** `três` — concordantes

**A terceira passada é a tela, e os rótulos do grupo *Data scaling* trazem a conta em prosa:**

> `Mean centering (mean-centered only)` · `Auto scaling (mean-centered and divided by the standard
> deviation of each variable)` · `Pareto scaling (mean-centered and divided by the square root of
> the standard deviation of each variable)` · `Range scaling (mean-centered and divided by the range
> of each variable)`

**E há o que a tela NÃO diz, e vale registrar porque custou uma medição:** *"divided by the standard
deviation"* não declara o **denominador** do desvio. A conta do `@param` diz `sd(x, na.rm=T)`, que em
R é `n - 1`; o nosso padrão é `n`. A diferença é `sqrt(n/(n-1))` e foi **medida** em 09/09/2026 —
0,829% com 61 amostras —, está em DV38, e é mantida por decisão. **Nenhuma das três fontes
deste documento resolvia isso**: as duas textuais dizem `sd()` sem o denominador explícito e a tela
diz menos ainda. Só a comparação célula a célula resolveu.

Do `@param scaleNorm` e do corpo:

> "MeanCenter" for Mean Centering, "AutoNorm" for Autoscaling, "ParetoNorm" for Pareto Scaling, amd
> "RangeNorm" for Range Scaling.

> `MeanCenter`: `x - mean(x)` — `AutoNorm`: `(x - mean(x))/sd(x, na.rm=T)` — `ParetoNorm`:
> `(x - mean(x))/sqrt(sd(x, na.rm=T))` — `RangeNorm`: `(x - mean(x))/(max(x)-min(x))`

**O que isto decide no nosso lado.** `auto`, `pareto` e `range` já eram idênticos. Entrou
`mean_center`: centrar **sem** dividir era operação que nenhum valor do nosso eixo fazia — o nosso
`none` não centra. `vast` é nosso e não tem contraparte lá.

---

## Os GRÁFICOS — um gênero próprio

### `[G-1]` As telas, e o que custaria cada figura — o que as telas mostram, e o que custaria cada um

**Data do levantamento.** 2026-09-10, por sessão de navegador sobre `metaboanalyst.ca`, versão
**6.0** (rodapé: *last updated 2026-09-08*).

**O dado usado foi o DELES, e a escolha é deliberada.** O conjunto de teste *Concentrations* —
77 amostras de urina de pacientes com câncer por 1H NMR (Eisner R. et al.), 63 compostos, dois
grupos —, carregado pelo botão *«Try our test data»*. **Nenhuma matriz deste laboratório saiu
desta máquina:** os nomes de amostra dos quatro arquivos embutem códigos de grupo clínico
(`_HAMTSP`, `_ATLL`, `_Asymptomatic`), e enviá-los a um serviço externo seria decisão de dado
sensível que ninguém tomou.

**O caminho percorrido:** `Statistical Analysis [one factor]` → *Upload* → *Data check* →
*Normalization* com `median / log10 / auto`, que é o caminho de paridade → *Statistics*.

> **Escopo do levantamento.** Enumera **o que existe e a forma de cada figura**. Não transcreve
> fórmula de gráfico nenhum, não mede número nenhum, e não estabelece equivalência com o nosso
> lado. Os sete elementos `[N-1]` a `[N-7]` são de **processamento** e transcrevem FÓRMULA; este é
> de **apresentação** e enumera TELA. **Por isso é `G` e não `N-8`:** o portão de citação
> exige de cada `[N-k]` um status de transcrição e a contagem de passadas que o conferiram,
> e nenhum dos dois se aplica a um levantamento de figura. Dar-lhe `N-8` seria rótulo igual
> para gêneros diferentes, e teria feito o portão pedir-lhe uma propriedade que ele não tem.

#### 1. A tela de NORMALIZAÇÃO já é gráfico, e este documento não o registava

Depois de *Normalize*, o botão *View Result* abre um painel com **duas abas** — `Feature View` e
`Sample View` — e cada aba traz **quatro** figuras:

| | Antes | Depois |
|---|---|---|
| **densidade** | curva sobre todos os valores | idem |
| **caixas** | uma caixa por feature (ou por amostra) | idem |

A própria tela declara o recorte, e ele é informativo: *"the boxplots show at most 50
features/samples due to space limitation; the density plots are based on all data"*.

**São oito painéis, de dois tipos, e respondem uma pergunta só:** *o que a minha escolha de
normalização fez com a distribuição?* É a figura que acompanha exactamente o que os `[N-5]` a
`[N-7]` descrevem, e o nosso lado não tem nenhuma.

#### 2. A seção STATISTICS — DEZENOVE entradas, em cinco grupos

Contadas na navegação lateral depois de a normalização correr:

| Grupo | Entradas |
|---|---|
| **Univariate Analysis** | Fold Change, T-tests, **Volcano plot**, One-way ANOVA, Correlation Heatmaps, Pattern Search, Correlation Networks (DSPC) |
| **Advanced Significance** | SAM, EBAM |
| **Chemometrics** | **PCA**, **PLS-DA**, sPLS-DA, orthoPLS-DA |
| **Cluster Analysis** | Dendrogram, Heatmaps, K-means, Self Organizing Map (SOM) |
| **Classification & Feature Selection** | Random Forest, SVM |

#### 3. UMA ANÁLISE NÃO É UM GRÁFICO, e é a medida que muda o custo

O `PCA` abre com **seis abas**: `Overview`, `Scree Plot`, `2D Scores Plot`, `Loadings Plot`,
`Synchronized 3D Plots` e `Biplot`. O *Overview* sozinho é uma matriz de dispersão par a par dos
cinco primeiros componentes, com **p de PERMANOVA (999 permutações)** em cada célula acima da
diagonal e a densidade por grupo na diagonal.

> **Dezenove entradas não são dezenove figuras — são dezenove FAMÍLIAS.** Qualquer estimativa
> feita sobre o número de entradas subconta por um fator que não é pequeno, e este parágrafo
> existe para que ninguém o faça.

#### 4. O que já temos o NÚMERO para desenhar, e o que exigiria estatística nova

**A distinção é a que decide custo**, e ela é verificável no nosso código:

| Figura | O que ela consome | Do nosso lado |
|---|---|---|
| **Volcano plot** | fold change no eixo x, `-log10(p)` no y, corte de FC e de p | **`univariate_test` já devolve `log2fc`, `pvalue` e `qvalue_fdr`.** É desenho puro |
| **Fold Change** | a razão por feature, ordenada | **`fold_change`, razão de MEDIANAS**, já existe |
| **T-tests** | p por feature, com FDR | **`pvalue` e `qvalue_fdr`**, já existem |
| **PCA** (6 abas) | scores, loadings, variância explicada | **`pca()` já devolve** scores e loadings com orientação canônica |
| **PLS-DA** | scores, loadings, VIP | **`plsda()` já devolve** Q², R²Y, VIP e p de permutação |
| Correlation Heatmaps / DSPC | matriz de correlação, rede | **`feature_correlation` e `pattern_search` já devolvem** a matriz entre features e as features ordenadas pelo coeficiente contra um perfil (U12, `statistics/correlacao.py`); **`dspc_network` já devolve** a rede de correlação parcial desenviesada, com `p` e `q` por aresta (U14, `statistics/dspc.py`) |
| Dendrogram / Heatmap | ligação hierárquica, ordem das folhas, z-score por feature | **`hierarchical_clustering` já devolve** as duas ligações, as duas ordens e a matriz escalonada (U10) |
| K-means / SOM | agrupamento particional | **`kmeans_clustering` e `self_organizing_map` já devolvem** a partição das amostras e o mapa, com semente declarada (U11, `statistics/clustering.py`) |
| ANOVA de N grupos | ANOVA de uma via | **`one_way_anova` já devolve** `F`, `p`, `q` e o post-hoc de Tukey ou Fisher; `kruskal_wallis` devolve `H` com Dunn (U5, U19) |
| SAM / EBAM / sPLS-DA / orthoPLS-DA | métodos próprios | não existe |
| Random Forest / SVM | classificadores | **`classify` já devolve** acurácia balanceada, matriz de confusão, importâncias e p de permutação, para os dois (U15, `statistics/classificadores.py`) |

**Dez famílias são renderização sobre número que já produzimos.** Na API, o despacho de
`rotas.py` tem ramo para `anova`, `correlation`, `kmeans`, `som`, `dspc` e `classifiers`; o
agrupamento hierárquico ainda não atravessa o fio.

E para o Volcano há uma correção de fonte: o modo `Classical` **não** dá t de Student no
servidor actual. Desde 07/2026 os dois modos põem um `p` moderado no eixo y — o seletor troca
apenas o **fold change** —, e o padrão histórico do MetaboAnalyst era t de variância igual, nunca
Welch. O nosso lado usa Welch, então **nenhum dos dois modos reproduz exactamente**.

#### 5. Dois detalhes de forma que valem para quem for implementar

**O volcano é interativo e declara o recorte na legenda:** *"Sig.Down [1] | Sig.Up [2] | Unsig.
[60]"* — a contagem em cada classe, ao lado da cor. E oferece o eixo y sobre `Raw` **ou** `FDR`,
escolhido pelo usuário.

**A PERMANOVA do Overview do PCA é um número que não temos.** Ela aparece como `p = 0.125` em
cada painel acima da diagonal, e é 999 permutações sobre os scores de cada par de componentes.

#### 6. O que este levantamento NÃO abriu

- As telas de **resultado** de `Enrichment` e `Pathway`, e os módulos `EBAM` e `SVM`, que a nota
  anterior deste documento já registrava como não abertos.
- Os outros **17 módulos** do MetaboAnalyst 6.0. Só o `Statistical Analysis [one factor]` foi
  percorrido, porque é o que consome o formato que a nossa ferramenta exporta.
- **Qualquer fórmula.** Este item diz o que a tela mostra, não como ela calcula — e a distinção
  é a mesma que separa os `[N-1]` a `[N-7]` de uma transcrição.

#### 7. O ESTADO de cada entrada — a partição sobre a qual o portão de R1 corre

**Esta tabela é insumo de portão, e não prosa.** `tests/unit/test_plots_arnes.py` lê-a, exige que
cada linha traga um estado do vocabulário fechado — `construída`, `excluída-KD3`, `adiada` — e corre
a bijeção de R1 **só** sobre a partição `construída`, contra `refino.plots.catalogo.CONSTRUTORES`.

**Por que uma partição e não o catálogo inteiro (KTD15).** Sobre o catálogo inteiro o portão seria
insatisfazível no dia um, nas duas direções: as entradas excluídas por KD3 e as adiadas ficam todas
sem construtor, e a trajetória de QC tem construtor e **não** é figura do MetaboAnalyst — é o item 3
de `ITENS_OBRIGATORIOS`, e tem bijeção própria. A saída errada seria uma lista de isenção, que
AR140 já mediu como o que mata um portão.

**Estado sem construtor é a condição normal; construtor sem estado `construída` é defeito.** Cada
unidade de figura vira a sua entrada para `construída` no mesmo commit em que acrescenta a função —
e o portão é o que impede que uma das duas metades aconteça sozinha.

**As duas colunas de grupo e de entrada não são livres:** têm de bijetar com o levantamento do
`#### 2.` acima, que foi escrito antes desta tabela existir. É a fonte independente da classe (d) de
KTD12 aplicada a um portão em vez de a uma figura — a contagem desta tabela é conferida contra
aquela, e as duas contra o `DEZENOVE` e o `cinco` que o cabeçalho do `#### 2.` escreve por extenso.

| Entrada | Grupo | Estado | Unidade prevista |
|---|---|---|---|
| Densidade (Feature View) | Normalização | construída | U4 |
| Caixas (Feature View) | Normalização | construída | U4 |
| Densidade (Sample View) | Normalização | construída | U4 |
| Caixas (Sample View) | Normalização | construída | U4 |
| Fold Change | Univariate Analysis | construída | U6 (análises) |
| T-tests | Univariate Analysis | construída | U6 (análises) |
| Volcano plot | Univariate Analysis | construída | U6 |
| One-way ANOVA | Univariate Analysis | construída | U6 (análises) |
| Correlation Heatmaps | Univariate Analysis | construída | U12 |
| Pattern Search | Univariate Analysis | construída | U12 |
| Correlation Networks (DSPC) | Univariate Analysis | construída | U21 |
| SAM | Advanced Significance | excluída-KD3 | — |
| EBAM | Advanced Significance | excluída-KD3 | — |
| PCA | Chemometrics | construída | U7 |
| PLS-DA | Chemometrics | construída | U8 |
| sPLS-DA | Chemometrics | adiada | — |
| orthoPLS-DA | Chemometrics | adiada | — |
| Dendrogram | Cluster Analysis | construída | U20 |
| Heatmaps | Cluster Analysis | construída | U20 |
| K-means | Cluster Analysis | construída | U11 |
| Self Organizing Map (SOM) | Cluster Analysis | construída | U11 |
| Random Forest | Classification & Feature Selection | construída | U15 |
| SVM | Classification & Feature Selection | construída | U15 |

**As quatro linhas de `Normalização` não vêm do `#### 2.`, e vêm do `#### 1.`** — duas abas,
`Feature View` e `Sample View`, e dois tipos de figura em cada, cada um trazendo o antes e o depois
lado a lado. São elas as *"quatro"* do painel de antes-e-depois.

> **A U4 construiu TRÊS das quatro; a quarta fechou depois, e por ACRÉSCIMO.**
> `Caixas (Sample View)` pede **uma caixa por amostra** — a distribuição de cada amostra sobre as
> features —, e essa distribuição **não atravessava o fio**: `CaixaDaDistribuicao`
> (`src/refino/api/esquemas.py`) tem o campo `feature`, e `rotas._caixas` reduz com `axis=0`, uma
> caixa por **coluna**. Do eixo das amostras o corpo trazia só `medias_por_amostra_*`, que é **um
> número por amostra** e serve a densidade, não uma caixa.
>
> A lacuna era do **plano** e não da U11: a tabela da §«A superfície nova do fio» descreve a carga
> como *"caixa de Tukey por feature ... até 50 caixas; vetor de médias por eixo para a
> densidade"*, então o fio foi desenhado com três das quatro figuras. Desenhar a quarta a partir
> das médias por amostra seria uma caixa de 68 números com o nome de uma caixa sobre 5.801, e
> foi por isso que a U4 a deixou `adiada` em vez de a inventar.
>
> **O que a fechou é exactamente o que estava previsto aqui**, e nada mais: um modelo novo ao lado
> de `CaixaDaDistribuicao` — `CaixaPorAmostra`, com `amostra` no lugar de `feature` —, a **mesma**
> redução de Tukey aplicada à transposta (`rotas._tukey`, uma implementação para os dois eixos), e
> o mesmo teto com a mesma semente. **O teto vale nos dois eixos porque o número é do
> MetaboAnalyst nos dois:** a tela dele diz *"at most 50 features/samples"*, e o `/samples` é a
> metade que faltava consumir — nos quatro arquivos deste laboratório o eixo das amostras tem
> **68**, então ele também amostra.

> **A LINHA `PCA` FICOU `construída` NA U7 COM QUATRO DAS SEIS ABAS, e o recorte vai aqui porque a
> coluna de estado não tem onde o pôr.** O `#### 2.` acima regista que o `PCA` do MetaboAnalyst
> abre com **seis abas** — `Overview`, `Scree Plot`, `2D Scores Plot`, `Loadings Plot`,
> `Synchronized 3D Plots` e `Biplot`. A U7 construiu **quatro**: *scree*, scores 2D com elipse e
> QC, cargas e *biplot*.
>
> **Ficam de fora, e o motivo é medido e não de gosto:** o `Overview` é uma matriz de dispersão par
> a par dos cinco primeiros componentes — quinze painéis, e a §«O que continua fora» já regista que
> a **PERMANOVA** que ele imprime é um número que este projeto não tem —, e o `Synchronized 3D` é
> tridimensional, o que `PainelDeFigura` não embute nesta rodada. Nenhum dos dois é *"adiado por
> esquecimento"*: os dois estão nomeados, com a razão.
>
> **O vocabulário desta coluna diz «há construtor», e é isso que `construída` afirma.** Um quinto
> estado — *"construída em parte"* — seria a lista de isenção com outro nome (AR140): a
> primeira entrada a usá-lo tornaria a partição de R1 negociável, e a bijeção deixaria de bijetar.
> O recorte é prosa porque é prosa que ele é; o que tem de ser mecânico — existe construtor, e ele
> está registado — continua a ser.
>
> **E as quatro NÃO constroem o item 6 de `ITENS_OBRIGATORIOS`.** Aquele pede *"PCA antes e
> depois"*, que são **duas** matrizes; estas quatro descrevem **uma**. Dizê-lo aqui e em
> `plots/catalogo.py` é o que impede o relatório de ganhar uma segunda implementação — nada
> acusaria a divergência (KD2).
>
> `reporting/model.py::defeitos_do_relatorio` compara o **número** de item — `presentes =
> {i.numero for i in rel.corpo}` —, e o título do relatório nunca é lido por lado nenhum: com o
> título fora da comparação, o que bijeta é a **chave**, e é por isso que registar estas quatro
> com vínculo de relatório declararia construído o que ninguém construiu.

> **A LINHA `PLS-DA` FICOU `construída` NA U8 COM TRÊS DAS SEIS ABAS, e a U9 ACRESCENTOU o
> multiclasse SEM mexer nesta tabela.** O recorte vai aqui pela mesma razão do da `PCA`: a coluna de
> estado não tem onde o pôr.
>
> O `#### 2.` acima regista que o `PLS-DA` do MetaboAnalyst abre em abas — `Overview`,
> `2D Scores Plot`, `Loadings`, `Cross Validation`, `Imp. Features` e `Permutation`. **A U8
> construiu três**: scores 2D, cargas e VIP. **Ficam de fora, e os motivos são medidos:** o
> `Overview` é par a par de vários componentes, e a aba de permutação desenha o histograma dos
> sorteios — que **não atravessa o fio**, porque `PLSDAResult` guarda o `p` e não os `Q²` sorteados.
>
> **A U9 acrescentou TRÊS construtores com a MESMA chave `PLS-DA`, e por isso nada nesta linha
> muda.** Eles desenham o PLS-DA **um-contra-resto**, escolhido por medição — uma grelha de
> K painéis, um modelo por classe —, e a bijeção de R1 corre sobre **conjuntos de chaves**: seis
> construtores com a mesma chave são um conjunto de um, que é o que este documento tem. **Inventar
> uma entrada `PLS-DA multiclasse` seria a saída errada duas vezes:** esta tabela descreve a tela
> **deles**, e as duas colunas da esquerda bijetam com o levantamento do `#### 2.`, que foi escrito
> antes desta tabela existir.
>
> **O MetaboAnalyst não tem aba de um-contra-resto**, e isso vai dito em vez de subentendido: o
> PLS-DA dele é multiclasse direto sobre a matriz indicadora. As três arquiteturas foram medidas e
> o um-contra-resto foi escolhido com a tabela de critérios escrita antes — é uma **divergência
> deliberada**, e nenhum critério ativo compara este número com o deles.
>
> **O vocabulário desta coluna diz «há construtor»**, e um quinto estado — *"construída em parte"* —
> seria a lista de isenção com outro nome (AR140).

**`excluída-KD3` é derivado do GRUPO e não escolhido linha a linha**, e o portão o confere assim:
toda entrada dos grupos que KD3 exclui é `excluída-KD3`, e nenhuma entrada de fora deles pode sê-lo.
É o que impede a exclusão de crescer uma linha de cada vez — que é como uma lista de isenção nasce.

> **KD3 exclui um grupo, `Advanced Significance`.** SAM e EBAM são `excluída-KD3` por
> **pertencerem a ele**, e não por uma linha própria — que é exactamente a propriedade que este
> parágrafo descreve; o conjunto de grupos excluídos vive em `tests/unit/test_plots_arnes.py`.
>
> **As duas metades sozinhas reprovam:** virar as células sem mudar o conjunto dispara *«KD3
> exclui-o inteiro»*; mudar o conjunto sem virar as células dispara *«lista de isenção»*.
>
> **Nenhuma palavra nova entrou no vocabulário de estado**, e a recusa é o conteúdo de KTD13: um
> quarto estado para *«declarado para a rodada seguinte»* seria a lista de isenção com outro nome.
> sPLS-DA e orthoPLS-DA continuam `adiada`, que é outro estado com outra causa (KD4).

**`adiada` não é promessa.** A coluna *Unidade prevista* diz de onde o trabalho **está planeado**
vir; onde ela traz `—`, ninguém o vai construir nesta rodada, e a entrada continua a exigir o
MetaboAnalyst. A postura da ferramenta perante isso é KD4, e é tentativa.

---

## O que este documento NÃO estabelece

- **Não** estabelece que os dois produzam o mesmo **número** sobre a mesma matriz — mas **isso deixou
  de ser lacuna em 09/09/2026 para um caminho**. O que este
  estabelece continua a ser a **fórmula**, transcrita dos dois lados. A igualdade numérica foi medida
  executando o MetaboAnalyst 6.0 sobre a nossa matriz `neg_c18` corrigida por SERRF:
  bijeção exata no filtro por RSD de QC, e 100% das 104.856 e das 94.062 células dentro de 1e-6 na
  imputação e na normalização, com `ddof=1`. **O recorte é um caminho de uma grade de 9.600** —
  `qc_rsd → one_fifth_min → median → log10 → auto` —, e nada aqui autoriza generalizar daí.
- **Não** transcreve fórmula da seção *Statistics* do MetaboAnalyst: o levantamento do que existe
  nela é o `[G-1]` acima, que enumera telas. PCA, PLS-DA com Q², R²Y, VIP e p-valor de permutação,
  t de Welch, Mann-Whitney, FDR de Benjamini-Hochberg e fold change foram construídos para
  **avaliar a correção**. Reusá-los para apresentar análise é mudança de propósito, e a regra 6 do
  README continua a valer sobre o PLS-DA aconteça o que acontecer: nunca um gráfico de scores sem o
  Q² e o p de permutação ao lado.
- **Não** cobre a correção de sinal. `SERRF` e `batch_loess` são portes do `tools.R` do laboratório;
  o MetaboAnalyst os oferece noutro lugar — a utilidade *Batch Effect Correction*, com nove métodos
  (ComBat, EigenMS, QC-RLSC, ANCOVA, RUV-random, RUV2, RUVseq, NOMIS, CCMN) — e a comparação entre
  os dois conjuntos não foi feita aqui.
- **A interface FOI verificada em 09/09/2026.** As cinco telas de
  processamento do módulo *Statistical Analysis* foram lidas por sessão HTTP própria, e nenhum dos
  sete itens declara mais `uma` passada. **O que continua não verificado na interface são as telas
  que não abriram:** `EBAM` e `SVM` (que é `RSVMView.xhtml`, um R-SVM recursivo) falharam com a
  página genérica de erro em cinco tentativas, e as telas de **resultado** do *Enrichment* e do
  *Pathway* não foram abertas. Nenhuma das três é de processamento, então não há item `[N-k]` a
  preencher — mas quem for mapear análise precisa saber que a lacuna existe.
- **Não** cobre o que a tela oferece e o nosso eixo não tem como ETAPA. Três casos, e os três são de
  desenho do `PipelineSpec` e não de fórmula: encadear *Low-variance* com *Low-abundance* (um campo
  de filtro por posto, não dois), o *Low-quality filter* por BLANK (não há amostra BLANK nos quatro
  arquivos, então não é verificável), e `Sample-specific normalization` mais `Normalization by
  reference feature`, que pedem **entrada nova do usuário** e portanto campo novo no contrato.
- **Não** é a especificação do formato de arquivo, que é `docs/formato-metaboanalyst.md`, nem a
  invocação do MetaboAnalystR, que é o `[CA-11.7]`.
