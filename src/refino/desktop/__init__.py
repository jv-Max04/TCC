"""Camada do aplicativo de mesa, o executável de Windows.

É fronteira, como `io/` e `api/`, e não núcleo: escolhe porta, abre socket, resolve caminho de
usuário, migra banco e abre janela. O núcleo (`domain/`, `metrics/`, `correction/`,
`preprocessing/`, `statistics/`, `diagnosis/`) não sabe que ela existe.

A janela alcança o núcleo pela API local, e não por chamada direta. Assim ela usa as mesmas rotas
de `api/rotas.py`, com as mesmas validações, e o artefato que a rota devolve é byte a byte o que a
CLI escreve, sem nada reimplementado. O custo é uma porta, e é por isso que o `servidor` existe.
"""
