# Linha de base da suíte no Windows

Como rodar a suíte num clone Windows e quais reprovações são conhecidas nesta máquina. O critério de
merge não é "tudo verde": é a lista de reprovações não ganhar item em relação à linha de base medida
antes da mudança, na mesma máquina.

## As quatro seleções

A suíte nunca roda numa invocação só: o processo quebra no meio, e o par `test_desktop_janela.py` e
`test_desktop_figuras.py` não pode correr na mesma sessão. São quatro seleções, em série, da raiz do
clone, com a venv do projeto:

```bash
.venv/Scripts/python.exe -m pytest tests/unit -m "not slow" --no-cov
.venv/Scripts/python.exe -m pytest -m "not slow" --ignore=tests/unit \
    --ignore=tests/test_desktop_janela.py --ignore=tests/test_desktop_figuras.py --no-cov
.venv/Scripts/python.exe -m pytest tests/test_desktop_janela.py --no-cov
.venv/Scripts/python.exe -m pytest tests/test_desktop_figuras.py --no-cov
```

Para comparar listas, acrescente `--junitxml=selN.xml` a cada uma e compare os `testcase` com
`failure` ou `error` contra os da linha de base. Código de saída nunca passa por pipe:
`cmd > saida.txt 2>&1; echo "EXIT=$?"`.

A pasta `serrf/` (dados do laboratório, fora do git) tem de estar na raiz do clone; sem ela os
testes golden pulam e as seleções provam menos.

## Cobertura

O piso de 85% vale sobre a união das seleções. Cada uma grava num arquivo próprio, e o relatório sai
da combinação:

```bash
COVERAGE_FILE=cob/.coverage.s1 .venv/Scripts/python.exe -m pytest tests/unit -m "not slow" \
    --cov=src/refino --cov-report= --cov-fail-under=0
# idem para as seleções 2, 3 e 4, com .coverage.s2, .s3 e .s4
.venv/Scripts/python.exe -m coverage combine cob/.coverage.s1 cob/.coverage.s2 \
    cob/.coverage.s3 cob/.coverage.s4
.venv/Scripts/python.exe -m coverage report --fail-under=85
```

## Reprovações conhecidas

Medidas em 16 e 17/11/2026, no fim da limpeza do repositório, num Windows 11 com 8 núcleos e
Python 3.12 da venv criada pelo `uv`. Quatro reprovações, todas de ambiente ou de plataforma:

| Teste | Causa |
|---|---|
| `tests/unit/test_dspc.py::test_o_PAR_CAUSAL_da_correcao_de_vies_diverge` | contagem numérica que difere no Windows: 12 valores abaixo de 1,0 onde o teste espera 10 |
| `tests/test_reference_data.py::test_verification_script_succeeds` | o teste chama `bash` com um caminho Windows, e o `bash` do sistema perde as barras invertidas |
| `tests/test_reference_data.py::test_verification_script_is_idempotent` | a mesma causa |
| `tests/test_readme_setup.py::test_todo_modulo_executado_com_menos_m_existe` | o README manda rodar `python -m pip`, e a venv criada pelo `uv` não tem `pip` |

Um teste é instável com a máquina carregada:
`tests/test_enriquecimento_r.py::test_cancelar_encerra_o_filho_e_o_neto_e_nao_sobra_temporario`
cancela assim que o arquivo do PID existe, e o arquivo nasce vazio antes de o PID ser escrito; se
o cancelamento cai entre os dois passos, a leitura dá `''` e o teste reprova com `ValueError`.
Sozinho, ou com a máquina livre, ele passa.

Totais da medição de 17/11/2026: 6.977 testes nas quatro seleções, 4 reprovados (os da tabela),
8 pulados e 1 falha esperada (`xfail`). A cobertura combinada, medida em 16/11/2026 sobre a suíte
de então, foi de 97%. Tempo, com a máquina livre e sem cobertura: cerca de 10, 18, 2 e 1,5
minutos.

## O executável

O congelamento e as provas do executável têm roteiro próprio: `scripts/congela_windows.ps1` e
`scripts/prepara_entrega.ps1`. O congelamento lê, de `%USERPROFILE%\_refino_env\`, a venv do alvo,
a árvore do R portátil (montada por `scripts/monta_r_portatil.ps1`) e a pasta das bibliotecas das
vias (montada por `scripts/busca_dados_vias.py`, que recusa arquivo cujo hash não seja o do
manifesto). As saídas prontas que viajam na pasta saem de `scripts/gera_saidas_prontas.py`.
