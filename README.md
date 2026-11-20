# Refino

Ferramenta de refino quimiométrico para dados de metabolômica LC-MS. Ela lê a tabela de
medições exportada do equipamento e a lista de injeção, diagnostica o erro sistemático da
corrida, corrige o sinal (sem correção, LOESS por lote ou SERRF), informa a qualidade do
resultado com várias métricas e exporta a matriz no formato de entrada do MetaboAnalyst.

A entrega é um executável Windows com janela e uma linha de comando, a `refino`.

Trabalho de Conclusão de Curso de Ciência da Computação, Universidade Municipal de São
Caetano do Sul (USCS), 2026. Licença: GPL-3.0-or-later.

## Instalação

Requisitos:

- Python 3.12, exatamente. O `pyproject.toml` recusa outras versões, e as dependências têm
  versão fixa.
- `make`, `bash` e `git`.

Na raiz do clone:

```bash
make setup
```

O alvo cria a venv em `.venv`, atualiza o `pip`, instala o projeto em modo editável com os
extras `dev` e `desktop`, e confere a integridade dos dados de referência. São estes
comandos, que também podem ser dados à mão:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e ".[dev,desktop]"
bash scripts/fetch_reference_data.sh
```

O extra `desktop` é opcional. Ele traz o PySide6, que é a janela, o `pytest-qt` e o
PyInstaller. Quem só usa a biblioteca ou a linha de comando pode instalar `.[dev]`; sem o
`desktop`, os testes da janela são pulados.

Os comandos deste arquivo supõem um shell POSIX, com a venv em `.venv/bin/`.

## Dados de referência

Os dados do laboratório e o pipeline R de referência ficam na pasta `serrf/`, que não está
no repositório: são dados clínicos de um laboratório parceiro, não publicados, e não há de
onde baixá-los. Quem os tem coloca a pasta na raiz do clone. `data/reference/input` e
`data/reference/output` são links simbólicos para dentro dela.

Sem a pasta, os testes que dependem dela são pulados (`skip`), entre eles os que comparam
com as saídas do R. A regra do projeto é não trocar dado real por dado inventado em teste.

`scripts/fetch_reference_data.sh` não baixa nada. Ele compara os arquivos presentes com o
manifesto `scripts/reference_data.sha256` e falha se algum estiver diferente. Arquivo
ausente não é erro.

## Uso

### Linha de comando

```bash
.venv/bin/refino --help
```

| Subcomando | O que faz |
|---|---|
| `corrigir` | corrige a deriva de sinal pelo método pedido e escreve a matriz corrigida |
| `processar` | aplica as etapas depois da correção: filtros, imputação, normalização, transformação e escalonamento |
| `exportar` | corrige, pré-processa e escreve a matriz no formato de entrada do MetaboAnalyst |

Cada subcomando tem o próprio `--help`, com todas as opções e os valores aceitos. Dois
exemplos, sobre um dos modos dos dados de referência:

```bash
.venv/bin/refino corrigir \
    --measurements data/reference/input/measurements_pos_hilic.csv \
    --sample-list data/reference/input/Sample_List_pos_hilic.csv \
    --mode pos_hilic --method loess \
    --output out/pos_hilic_loess.csv --log out/pos_hilic_loess.json

.venv/bin/refino exportar \
    --measurements data/reference/input/measurements_pos_hilic.csv \
    --sample-list data/reference/input/Sample_List_pos_hilic.csv \
    --mode pos_hilic --method none \
    --output out/pos_hilic_metaboanalyst.csv
```

`--method` aceita `none`, `loess` e `serrf`, e o padrão é `none`. `--seed` fixa a semente:
a mesma entrada com a mesma semente dá a mesma saída. Os arquivos de entrada têm features
em linhas e amostras em colunas, que é o padrão de `--orientation`. Com `loess` ou `serrf`
a ferramenta imprime uma ressalva sobre o que o método entrega; leia-a antes de usar a
saída. A pasta `out/` é ignorada pelo git.

A linha de comando cobre correção, pré-processamento e exportação. O diagnóstico, as
análises estatísticas e os gráficos estão na janela.

### A janela

Em desenvolvimento, com o extra `desktop` instalado:

```bash
.venv/bin/python -m refino.desktop.arranque
```

O arranque prepara o banco, sobe uma API interna no mesmo processo e abre a janela, que
fala com essa API por HTTP. A API escuta só em `127.0.0.1`, numa porta livre escolhida na
hora, porque os dados são clínicos e não devem sair da máquina. As rotas estão descritas em
`docs/rotas-api.md`.

As execuções ficam num arquivo SQLite, o `refino.db`. No executável ele fica ao lado do
`refino.exe`. Em desenvolvimento ele fica no diretório de dados do usuário:
`%LOCALAPPDATA%\refino` no Windows, e `$XDG_DATA_HOME/refino` ou `~/.local/share/refino`
nos outros sistemas. A variável `REFINO_DB_URL` muda o lugar. Ela tem de ser uma URL de
SQLite, por exemplo `sqlite:///out/refino.db`; dentro do clone, o banco só é aceito sob
`out/`.

### O executável

O executável é construído no Windows, à mão, da raiz do clone:

```powershell
pwsh -ExecutionPolicy Bypass -File scripts\congela_windows.ps1
powershell -ExecutionPolicy Bypass -File scripts\prepara_entrega.ps1 -Zipar
```

`scripts/congela_windows.ps1` congela a janela com o PyInstaller, a partir de
`refino-desktop.spec`, na pasta `dist\refino\`, e depois exercita o binário. Os caminhos da
venv de congelamento e do R embarcado são parâmetros do script; o cabeçalho dele os
explica. `scripts/prepara_entrega.ps1` confere que a pasta foi construída do commit atual,
tira dela o registro deixado pelos testes do binário e, com `-Zipar`, escreve o zip da
entrega.

Quem recebe a pasta abre o `refino.exe`; não precisa de Python instalado. A documentação
de uso que viaja com ela está em `docs/`: `docs/LEIA-ME-aplicativo.txt` e
`docs/GUIA-do-aplicativo.html` são o começo.

### Recursos em beta

A análise de multiverso e as vias metabólicas estão implementadas e vão no executável, nas
abas «Teste de robustez (beta)» e «Vias metabólicas (beta)». A primeira roda uma grade de
especificações do pipeline e mede a estabilidade do resultado. A segunda prediz atividade
de via a partir dos picos e enriquece listas de compostos identificados. As duas são
tratadas como trabalho futuro. O estado delas está em `docs/trabalhos_futuros/LEIA-ME.md`.

## Testes

A suíte não roda numa invocação só: o processo quebra. São quatro seleções, em série:

```bash
.venv/bin/python -m pytest tests/unit -m "not slow" --no-cov
.venv/bin/python -m pytest -m "not slow" --ignore=tests/unit \
    --ignore=tests/test_desktop_janela.py --ignore=tests/test_desktop_figuras.py --no-cov
.venv/bin/python -m pytest tests/test_desktop_janela.py --no-cov
.venv/bin/python -m pytest tests/test_desktop_figuras.py --no-cov
```

O `--no-cov` está ali porque a configuração do pytest liga a cobertura com piso de 85%
sobre a suíte inteira, e uma seleção parcial não o alcança.

Há reprovações que dependem da máquina. O critério não é «tudo verde»: é a lista de
reprovações não ganhar item em relação a uma linha de base medida na mesma máquina, antes
da mudança.

Os marcadores estão declarados no `pyproject.toml`. Entre eles:

| Marcador | Seleciona |
|---|---|
| `golden` | comparação contra as saídas do pipeline R; precisa dos dados de referência |
| `integration` | o pipeline completo num modo real |
| `property` | invariantes, com o `hypothesis` |
| `synthetic` | recuperação de uma verdade conhecida em dado gerado |
| `slow` | testes que levam dezenas de minutos; ficam fora das quatro seleções |

Lint e tipos, que é o que `make lint` roda:

```bash
.venv/bin/python -m ruff check src tests scripts
.venv/bin/python -m ruff format --check src tests scripts
.venv/bin/python -m mypy src/refino
```

O `Makefile` tem também `make test-golden`, que roda só o marcador `golden`, e os alvos
`make test`, `make check` e `make test-all`. Esses três chamam o pytest uma vez só, então
valem para eles a mesma ressalva do começo desta seção.

## Estrutura do repositório

```
src/refino/
  domain/         a matriz imutável (RefinoMatrix) e os tipos do domínio
  metrics/        métricas de qualidade: RSD dos QCs, estrutura no PCA, montagem do QualityReport
  correction/     os corretores de sinal: none, LOESS por lote e SERRF
  preprocessing/  filtros, imputação, normalização, transformação e escalonamento
  statistics/     PCA, PLS-DA com validação, testes univariados, ANOVA, agrupamento
  diagnosis/      diagnóstico da corrida: trajetória dos QCs, pontos de mudança, lotes
  plots/          construtores de figura
  pathways/       vias metabólicas (beta)
  io/             leitura e escrita de arquivos; chama o R embarcado por processo
  decision/       escolha automática do corretor
  multiverse/     análise de multiverso (beta)
  synthetic/      gerador de matrizes com verdade conhecida
  reporting/      estrutura do relatório e avisos
  persistence/    SQLite e migrações (alembic)
  api/            a API interna (FastAPI), embutida no processo da janela
  interface/      cliente HTTP e lógica de fluxo que a janela usa
  desktop/        a janela (PySide6) e o arranque do executável
  cli.py          a linha de comando `refino`
tests/            unit/, golden/ (contra as saídas do R), integration/ e os testes da raiz
scripts/          congelamento e entrega (.ps1), conferências e geradores de fixture
data/fixtures/    fixtures versionadas
data/reference/   links simbólicos para os dados em serrf/
serrf/            dados do laboratório e pipeline R de referência; fora do git
r_service/        roteiro em R das vias, copiado para o executável no congelamento
docs/             documentação do aplicativo
```

Os oito primeiros pacotes de `src/refino/` são o núcleo: recebem e devolvem estruturas em
memória, e não conhecem banco, HTTP, sistema de arquivos, Qt nem R.

## Regras do produto

1. Teste nunca usa dado fabricado no lugar de dado real. Arquivo de referência ausente é `skip`.
2. Tolerância e número esperado não se afrouxam para um teste passar. Se a implementação está
   certa e o número não bate, o caso é relatado, não ajustado.
3. Toda operação estocástica recebe `seed`. Mesma entrada e mesma semente dão a mesma saída.
4. No SERRF não se busca igualdade numérica com o R; o critério é concordância estatística.
5. Qualidade nunca é um número só: toda avaliação devolve o `QualityReport` completo.
6. PLS-DA nunca devolve scores sem Q² e p de permutação.
7. `RefinoMatrix` é imutável, inclusive o array. Cada etapa devolve uma matriz nova.
8. O núcleo não conhece banco, HTTP, sistema de arquivos, Qt nem R.
9. Sem `rpy2`. O R roda por processo, com JSON na entrada e na saída.
10. A API escuta só em loopback: os dados são clínicos.

Convenções: tipagem completa em `src/`, com `mypy` estrito em `domain`, `metrics` e
`correction`; `numpy` no numérico e `pandas` só nas bordas de I/O; caminhos com `pathlib.Path`;
nomes de domínio em inglês no código (`qc`, `rsd`, `drift`) e prosa em português; docstring diz o
que a coisa faz e, quando não é óbvio, por quê; commits em português, no formato conventional
commits. Correção de sinal e normalização são etapas distintas, nessa ordem, e `none` é opção
legítima de correção.
