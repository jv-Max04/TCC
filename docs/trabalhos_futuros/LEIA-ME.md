# Trabalhos futuros — o multiverso e o enriquecimento de vias

O multiverso e o enriquecimento de vias estão implementados e são funcionalidades em beta: não fazem
parte da entrega ao laboratório. Esta pasta reúne a documentação e os arquivos relacionados às duas.

## O que isto quer dizer, e o que NÃO quer dizer

| Pergunta | Resposta |
|---|---|
| As duas funcionalidades existem? | **Sim.** Estão implementadas e testadas. O multiverso e a predição de vias estão no executável, e o enriquecimento de lista identificada pelo MetaboAnalystR também: um R portátil embarcado o roda como segundo modo da aba «Vias metabólicas (beta)» — também beta, fora da entrega. |
| Fazem parte da entrega ao laboratório? | **Não.** As abas são **«Teste de robustez (beta)»**, a do multiverso, e **«Vias metabólicas (beta)»**, com uma linha que diz isso a quem as abre (`ABAS` em `src/refino/desktop/janela/principal.py`). |
| O código foi apagado ou movido? | **Não**, e o motivo está medido na seção seguinte. Ele continua coberto pelos testes de cada commit, para não apodrecer enquanto espera. |
| As fases 8 e 11 deixaram de estar fechadas? | **Não.** O que mudou foi a **entrega**, não a implementação. Nenhum critério mudou de estado. |

## O que está nesta pasta

| Caminho | O que é |
|---|---|
| [`MULTIVERSO-o-que-e-e-o-que-custa.html`](MULTIVERSO-o-que-e-e-o-que-custa.html) | O mecanismo do multiverso e a ficha de custo medida. **Viaja na pasta do executável como `MULTIVERSO.html`**, porque a aba está lá, marcada beta: `scripts/congela_windows.ps1` copia daqui. |
| `telas/2026-09-24-m7-vias-metabolicas.png` | A captura da aba de vias. |

## O que ficou FORA desta pasta, e por quê

**Mover não é de graça neste repositório**: cada raiz — `src/`, `tests/`, `scripts/`,
`data/fixtures/` — é o escopo de um portão, e um arquivo que sai dela deixa de ser conferido **sem
que nada reprove**.

### O código fica em `src/`, porque o executável o carrega ao arrancar

A cadeia é `src/refino/desktop/arranque.py:314` → `src/refino/api/app.py:42` →
`src/refino/api/rotas.py:158-159`, que importa `refino.multiverse` no topo. Tirar o pacote faz a
**janela inteira** não abrir, e não só a aba.

| Código | Tema | Por que não se move |
|---|---|---|
| `src/refino/multiverse/` (`grid`, `executor`, `stability`, `cache`) | multiverso | importado no arranque, pela cadeia acima, e por `src/refino/interface/fluxo.py` |
| as seis rotas `/multiversos*` em `src/refino/api/rotas.py`, os esquemas em `api/esquemas.py`, as funções de progresso em `api/progresso.py` | multiverso | `docs/rotas-api.md` e o portão do contrato da API fazem bijeção com elas |
| as duas tabelas do histórico e a migração `a3f7c1b90e42` | multiverso | a migração está **no meio** da cadeia: `c8b1e64f27a9` aponta para ela, e tirá-la impede de migrar qualquer banco que já exista |
| `src/refino/desktop/janela/multiverso.py` e a entrada em `ABAS` de `principal.py` | multiverso | é a aba que fica, marcada beta |
| `src/refino/pathways/` (o mummichog) e `src/refino/desktop/janela/vias.py` | vias | é a aba que fica, marcada beta; o mummichog entra no executável pelo `pyproject.toml` e pelos `hiddenimports` de `refino-desktop.spec` |
| `calculam_aqui` em `principal.py` | vias | mecanismo que só existe por causa da aba de vias, com âncora de presença nos testes |

O enriquecimento no executável, também beta, fica fora desta pasta pelos mesmos motivos, e o código
dele é: `src/refino/desktop/janela/enriquecimento.py` (o segundo modo da aba de vias, importado por
`janela/vias.py`); `src/refino/desktop/vias_r.py` (onde estão o R embarcado, o roteiro e as
bibliotecas); `src/refino/io/enriquecimento_r.py` (o lançador do R embarcado);
`io/identificacao.py` (a tabela de identificação do Progenesis); `io/dados_vias.py` (as bibliotecas
de dados e o download); `io/enriquecimento_externo.py` e `io/rds_restrito.py` (o modo externo);
`src/refino/pathways/enriquecimento.py` e `pathways/dados_vias.json` (o manifesto das bibliotecas);
e `r_service/vias_entrypoint.R`, o roteiro que `scripts/congela_windows.ps1` copia para a pasta
`R\`.

### Os testes ficam em `tests/`

`pyproject.toml` declara `testpaths = ["tests"]`: teste que sai daí deixa de rodar sem reprovar.

- **multiverso:** `unit/test_multiverse_{best_by_metric,cache,executor,stability}.py`,
  `test_multiverso_em_disco.py`, `test_multiverso_na_janela.py`, `test_historico_de_multiversos.py`,
  `integration/test_multiverse_benchmark.py` (o `[TO-8.4]`), e partes de `test_api_progresso.py` e
  `test_cancelamento.py`;
- **vias:** `unit/test_pathways_{analise,picos}.py` e `test_desktop_vias.py`.

Os testes do enriquecimento no executável são 11 arquivos: `test_enriquecimento_r.py`,
`test_enriquecimento_externo.py`, `test_io_identificacao.py`, `test_dados_vias.py`,
`test_busca_dados_vias.py`, `test_desktop_enriquecimento.py`, `test_vias_entrypoint_contrato.py`,
`test_prova_enriquecimento_congelado.py`, `test_r_portatil_roteiro.py`,
`unit/test_pathways_enriquecimento.py` e `unit/test_rds_restrito.py`.

### Os scripts e os fixtures ficam em `scripts/` e `data/fixtures/`

Os scripts calculam a raiz do repositório pela profundidade da própria pasta, e saem do `ruff` e dos
portões de scripts se mudarem de lugar.

| Arquivo | Observação |
|---|---|
| `scripts/prova_multiverso_congelado.ps1` e `scripts/prova_multiverso_eixo.json` | a prova no executável congelado; é também a **única asserção positiva** de que o PLS-DA é calculado no binário |
| os `data/fixtures/*teto*.json` | o teto do `[TO-8.4]`, de que depende o sinal de merge enquanto o `[CA-8.5]` estiver em remediação |
| `data/fixtures/custo_do_multiverso.json` | **misto**: guarda também a curva de threads do SERRF, e é a única fonte do custo que a tela mostra |

Os scripts e os fixtures do R portátil e do enriquecimento no executável:
`scripts/monta_r_portatil.ps1`, `scripts/r_portatil.py`, `scripts/r_portatil_instala.R`,
`scripts/r_portatil_despeja.R`, `scripts/r_portatil_pacotes.txt`, `scripts/r_portatil_arvore.sha256`,
`scripts/r_portatil_licencas.tsv`, `scripts/r_portatil_namespaces/`, `scripts/prova_r_embarcado.py`,
`scripts/prova_enriquecimento_congelado.ps1` e `.py`, `scripts/monta_fonte_gpl.py`,
`scripts/busca_dados_vias.py`, `data/fixtures/rds/` e `data/fixtures/vias-ora/`.

## O que NÃO é enriquecimento de vias, e continua na entrega

- as análises estatísticas no estilo do MetaboAnalyst — PCA, PLS-DA, ANOVA, volcano e as figuras;
- a **exportação** no formato do MetaboAnalyst (`docs/formato-metaboanalyst.md`) — na Fase 11, o
  `[CA-11.1]` e o `[CA-11.6]` são exportação, e só o `[CA-11.5]` e o `[CA-11.7]` são do
  enriquecimento;
- `docs/metodos-metaboanalyst.md`, que é processamento e normalização.

Atenção ao `grep`: o token `MetaboAnalystR` aparece também em `src/refino/statistics/anova.py`, como
oráculo da ANOVA, e isso é estatística.

## Onde o núcleo ainda aponta para o multiverso

Nenhum destes é defeito novo; ficam listados para quem retomar.

- `src/refino/decision/policy.py:179` — quando a Fase 9 se abstém, a justificativa diz *«execute o
  teste de robustez»*. Continua verdadeiro, porque a aba existe.
- `src/refino/reporting/model.py:47-48` — os itens 7 e 8 do relatório da Fase 10 são o multiverso.
- `src/refino/interface/fluxo.py:2089` — a ressalva de uma métrica da aba «Qualidade do resultado»,
  que a janela lê deste módulo, cita *«96 combinações do teste de robustez»*. É uma medição feita
  pelo multiverso, e continua verdadeira.

## Verificações em aberto do enriquecimento

- **o oráculo de DLL da prova do congelado**: `getLoadedDLLs()` não vê a dependência que o Windows
  resolve pelo nome, e a verificação ainda não foi vista produzir o desfecho «fora»;
- **a metade não testada do mecanismo do 500**: dois compostos cuja única via é a mesma;
- **a janela congelada de ponta a ponta**: não correu; as medições do enriquecimento usaram o painel
  em Qt sem tela, com o código do repositório, ou o interpretador embarcado sem passar pelo
  `refino.exe`;
- **outra máquina**: a primeira chamada ao R vai sem teto porque a medição é de uma máquina só, a
  que constrói, e as corridas do modo externo são desta máquina.

## O que falta para cada uma voltar à entrega

**As duas já funcionam.** Para voltarem à entrega, falta tirar o «(beta)» das duas abas e a linha
que as declara fora da entrega.

O estado de cada item vive aqui.
