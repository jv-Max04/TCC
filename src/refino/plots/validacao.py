"""Validação de um modelo supervisionado e a frase que toda figura dele imprime.

Fica num módulo próprio porque mais de uma família de figuras usa a mesma trinca (Q², p de
permutação e regime), e uma cópia por família divergiria. `Validacao` é um dataclass congelado
sem valor padrão: a carga de desenho que a exige como campo não admite construção parcial, e
por isso não há como desenhar scores de PLS-DA sem `q2` e `permutation_pvalue`.

O regime viaja com o número. Q² de sete dobras e de dez não são a mesma quantidade, e um p de
permutação tem resolução `1/(n+1)`: sem o `n`, `p = 0,00498` parece medida e é o mínimo
possível de 200 sorteios.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from refino.plots.base import inteiro_do_fio, numero_do_fio
from refino.plots.recorte import decimal, milhar

__all__ = [
    "Validacao",
    "frase_curta_da_validacao",
    "frase_da_validacao",
    "validacao_do_fio",
    "veredito_da_validacao",
]


@dataclass(frozen=True)
class Validacao:
    """`q2` e `permutation_pvalue`, mais o regime em que foram medidos.

    É campo obrigatório das cargas de desenho que a exigem. Como o dataclass é congelado e não
    tem valor padrão, não existe `EscoresDoPlsda` sem `Validacao`, logo não existe caminho que
    desenhe um score sem o Q² ao lado. `PLSDAResult` faz o mesmo do lado de quem calcula.

    O regime (dobras, sorteios, componentes) vai junto porque o número sozinho não é comparável.
    """

    q2: float
    permutation_pvalue: float
    r2y: float
    n_permutacoes: int
    cv_folds: int
    n_componentes: int


def validacao_do_fio(
    *,
    q2: Any,
    permutation_pvalue: Any,
    r2y: Any,
    n_permutacoes: Any,
    cv_folds: Any,
    n_componentes: Any,
) -> Validacao:
    """Monta a `Validacao` a partir dos campos da raiz `plsda`, com guarda em cada um.

    Recebe os campos um a um, e não o dicionário, porque o teste de contrato entre a janela e a
    API só confere as chaves lidas no código da janela. Se o dicionário fosse lido aqui, uma
    chave mal escrita devolveria `nan` para sempre sem nada acusar.

    O que falta vira `nan`, não `0.0`: `frase_da_validacao` escreve 'não declarado' para o não
    finito, e um zero afirmaria que o modelo prediz exatamente como a média.
    """
    return Validacao(
        q2=numero_do_fio(q2),
        permutation_pvalue=numero_do_fio(permutation_pvalue),
        r2y=numero_do_fio(r2y),
        n_permutacoes=inteiro_do_fio(n_permutacoes),
        cv_folds=inteiro_do_fio(cv_folds),
        n_componentes=inteiro_do_fio(n_componentes),
    )


def _numero_ou_ausente(valor: float) -> str:
    """`0,2371` ou `"não declarado"`. Nunca um número no lugar da ausência dele."""
    return decimal(valor) if math.isfinite(valor) else "não declarado"


def frase_da_validacao(validacao: Validacao) -> str:
    """A linha que vai na imagem: `Q²`, `p` de permutação, `R²Y` e o regime dos dois.

    É uma função, e não um literal por figura, para que todas as figuras escrevam a mesma frase.

    `R²Y` entra ao lado do `Q²` porque o par é o que se lê: `R²Y` mede o ajuste dentro da amostra
    e sobe com qualquer coisa, e a distância entre os dois é o sintoma de sobreajuste.
    """
    return (
        f"Q² = {_numero_ou_ausente(validacao.q2)} "
        f"(validação cruzada de {milhar(validacao.cv_folds)} dobras estratificadas) · "
        f"p de permutação = {_numero_ou_ausente(validacao.permutation_pvalue)} "
        f"em {milhar(validacao.n_permutacoes)} sorteios · "
        f"R²Y = {_numero_ou_ausente(validacao.r2y)} (dentro da amostra)"
    )


def frase_curta_da_validacao(validacao: Validacao) -> str:
    """`"Q² = -0,01253 · p de permutação = 0,1095"`: a versão que cabe no título.

    A frase completa não cabe na segunda linha do título: ele é centrado nos eixos, que são mais
    estreitos que a figura porque a legenda fica fora deles, e o texto sairia cortado na borda.
    No título ficam `q2` e `permutation_pvalue`, que são o que tem de estar ao lado do score. O
    número de dobras, o de sorteios e o `R²Y` continuam no rodapé, que quebra em linhas.
    """
    return (
        f"Q² = {_numero_ou_ausente(validacao.q2)} · "
        f"p de permutação = {_numero_ou_ausente(validacao.permutation_pvalue)}"
    )


def veredito_da_validacao(validacao: Validacao) -> str:
    """O que o `Q²` significa, em três desfechos.

    - `q2` não finito: não foi medido, a validação cruzada não produziu número.
    - `q2 <= 0`: o modelo prediz pior que a média dos rótulos.
    - `q2 > 0`: prediz melhor que a média; quanto é o número, e o `p` diz quantas vezes um
      sorteio o alcança.

    Zero é a definição, não um limiar escolhido: `Q² = 1 - PRESS/TSS` com `PRESS = TSS` é errar
    tanto quanto errar a média. Qualquer outro corte (0,4, 0,5) exigiria uma calibração que o
    projeto não tem.

    O ramo do meio não afirma que a figura desenha separação de calibração: nas três figuras do
    PLS-DA binário, que são as que leem esta frase, os escores são os de validação cruzada, e as
    cargas e o VIP não desenham amostra nenhuma.
    """
    if not math.isfinite(validacao.q2):
        return (
            "Q² NÃO FOI MEDIDO: a validação cruzada não produziu número, então esta figura não "
            "tem como dizer se a separação prediz alguma coisa fora da amostra"
        )
    if validacao.q2 <= 0.0:
        return (
            "Q² <= 0: o modelo prediz PIOR que a média dos rótulos. Uma separação no ajuste "
            "dentro da amostra não é predição — é o falso positivo que a validação existe "
            "para tornar visível"
        )
    return (
        "Q² > 0: o modelo prediz melhor que a média dos rótulos fora da amostra. QUANTO melhor é "
        "o número acima, e o p de permutação diz com que frequência um sorteio de rótulos o "
        "alcança — separação visual convincente com Q² baixo é o modo de falha desta análise"
    )
