---
title: SAVEPOINT no SQLite sem BEGIN confirma a gravação ao ser liberado
date: 2026-11-18
category: database-issues
module: persistence
problem_type: database_issue
component: database
symptoms:
  - "grava_execucao falha no meio e deixa estudo e matriz gravados sem a execução"
  - "O teste de atomicidade só existia no caminho Postgres, que é pulado sem banco configurado"
root_cause: wrong_api
resolution_type: code_fix
severity: high
tags: [sqlite, savepoint, begin-nested, transacao, atomicidade, sqlalchemy]
---

# SAVEPOINT no SQLite sem BEGIN confirma a gravação ao ser liberado

## Problem
Uma gravação que falhava no meio (por exemplo, uma amostra sem `sample_type`) não era
desfeita por inteiro no SQLite: o estudo e a matriz criados antes do erro ficavam no banco.

## Symptoms
- Linhas órfãs em `studies`/`matrices` depois de um `grava_execucao` que levantou exceção.
- Nenhum teste acusava: o teste de atomicidade rodava só contra Postgres e era pulado.

## What Didn't Work
- Confiar em `Connection.begin_nested()` como em Postgres. No driver padrão do SQLite
  (pysqlite), a transação só começa no primeiro INSERT. Um SAVEPOINT emitido antes disso
  vira a própria transação, e liberá-lo confirma o que foi gravado.

## Solution
Todo SAVEPOINT passa por `_ponto_de_retorno` em `src/refino/persistence/repositories.py`,
que abre a transação explicitamente quando a conexão ainda não está em uma:

```python
def _ponto_de_retorno(c: Connection) -> NestedTransaction:
    if c.dialect.name == "sqlite":
        bruta = c.connection.dbapi_connection
        if bruta is not None and not bruta.in_transaction:
            c.exec_driver_sql("BEGIN")
    return c.begin_nested()
```

O teste `test_a_gravacao_e_atomica_e_nao_deixa_execucao_pela_metade`, em `tests/integration/test_persistence.py`,
passou a conferir também a tabela de estudos, no SQLite.

## Why This Works
Com o BEGIN explícito, o SAVEPOINT fica aninhado numa transação de verdade. Liberá-lo não
confirma nada; a exceção desfaz a transação externa inteira.

## Prevention
- Não chamar `begin_nested()` direto em código novo de persistência; usar `_ponto_de_retorno`.
- Testes de atomicidade precisam rodar no dialeto que o aplicativo entrega (SQLite), não só
  num caminho Postgres que fica pulado na máquina de quem roda a suíte.
