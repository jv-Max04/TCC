"""As posições de QC do gerador, e o ramo que AFIRMA sem nunca ter sido exercitado.

**O ramo `else` é inalcançável hoje, e isto foi medido.** Os
**cinco** locais que constroem `SyntheticSpec` no repositório — o `_spec` dos testes de integração,
`test_synthetic_ground_truth`, `dimensiona_corpus`, `gera_corpus_fase9` e o arnês de recuperação —
**nenhum passa `qc_every` nem `n_samples`**, então `qc_every` é sempre 11 e `n_samples` sempre 68, e
`len(posicoes)` é sempre 7. O `else` nunca corre.

Eram cinco quando esta frase foi escrita; contados, são doze
arquivos fora deste — o corpus de 7 QCs da P2 e o
teste de regeração de `tests/test_perfil_da_politica.py` entre eles —, e **nenhum passa `qc_every`
nem `n_samples`**. A conclusão continua: `qc_every` é 11, `n_samples` é 68, e o `else` não corre.

Ele não é código morto: **é alcançável**, e a docstring da função afirma o que ele faz — *"o ramo de
`append` sobrevive para o caso de haver uma posição periódica só, em que substituir deixaria a
corrida sem QC no início"*. Afirmação sem asserção atrás é verificação que não existe, e o risco não
é hoje: é na primeira vez que alguém variar `qc_every`, quando o desfecho — **dois** QCs — é regime
que a correção inteira supõe não existir.

**Rápido de propósito.** `_posicoes_de_qc` é função pura sobre dois inteiros; não precisa de
`generate`, de matriz real nem do marcador `slow`. Por isso este arquivo entra no portão de commit,
enquanto os testes que exercitam o gerador inteiro ficam fora dele.
"""

from __future__ import annotations

import pytest

from refino.domain.synthetic import SyntheticSpec
from refino.synthetic.generator import _posicoes_de_qc

# As injeções dos QCs nos quatro arquivos reais, em base 0: diferenças de 11, 11, 11, 11, 11 e 12.
REAIS = (0, 11, 22, 33, 44, 55, 67)


def _spec(**trocas: object) -> SyntheticSpec:
    base: dict[str, object] = {
        "imitated_mode": "pos_c18",
        "drift_to_noise_ratio": 0.0,
        "defect_type": "clean",
    }
    return SyntheticSpec(**{**base, **trocas})  # type: ignore[arg-type]


def test_o_padrao_reproduz_as_posicoes_dos_quatro_arquivos_reais() -> None:
    """Com os padrões — 68 amostras, um QC a cada 11 — saem os sete das corridas reais."""
    assert _posicoes_de_qc(_spec()) == REAIS


@pytest.mark.parametrize(
    ("n_samples", "qc_every", "esperado"),
    [
        (68, 11, REAIS),
        # `range(0,68,34)` da `[0,34]`: substituir move o ultimo QC de 34 para 67. **O
        # deslocamento e `n-1 - ultima_periodica`, e ele cresce com `qc_every`** — 1 no caso real
        # (66 -> 67) e 33 aqui. Vai como caso de teste porque a docstring da funcao so descreve o
        # caso real, onde o deslocamento e desprezivel.
        (68, 34, (0, 67)),
        (68, 22, (0, 22, 44, 67)),
        (30, 7, (0, 7, 14, 21, 29)),
    ],
)
def test_o_ultimo_substitui_a_ultima_posicao_periodica(
    n_samples: int, qc_every: int, esperado: tuple[int, ...]
) -> None:
    """Sempre que há mais de uma posição periódica, a última é TROCADA pelo índice final.

    O que isto afirma, e é o conserto: o número de QCs é o das posições periódicas, e o
    último intervalo é o único que difere. Acrescentar em vez de substituir daria um QC a mais e
    dois adjacentes no fim.
    """
    assert _posicoes_de_qc(_spec(n_samples=n_samples, qc_every=qc_every)) == esperado


@pytest.mark.parametrize(
    ("n_samples", "qc_every"),
    [(68, 68), (68, 100), (10, 10), (2, 2), (2, 5)],
)
def test_o_ramo_else_acrescenta_e_preserva_o_qc_do_inicio(n_samples: int, qc_every: int) -> None:
    """O ramo que nunca correu: com **uma** posição periódica, acrescenta em vez de substituir.

    **A asserção é sobre o QC do início, que é o que o ramo existe para preservar.** Substituir
    aqui daria `(n-1,)` — uma corrida cujo único ponto de controlo está no fim, e portanto sem
    referência para a deriva do começo. É o desfecho que a docstring da função declara e que
    nenhum teste afirmava.
    """
    posicoes = _posicoes_de_qc(_spec(n_samples=n_samples, qc_every=qc_every))

    assert posicoes == (0, n_samples - 1)
    assert posicoes[0] == 0, "o ramo existe para NÃO perder o QC do início"
    assert len(posicoes) == 2


def test_o_par_causal_os_dois_ramos_dao_desfechos_diferentes() -> None:
    """Sem isto, os dois testes acima poderiam passar sobre o mesmo caminho e ninguém notaria.

    A planta cai sobre dado em que as duas coisas diferem: `qc_every = 11` percorre o ramo
    de substituição e `qc_every = 68` o de acréscimo, e as contagens **têm** de divergir — senão o
    par é vácuo e o `else` continua sem cobertura, com dois testes verdes por cima.
    """
    substitui = _posicoes_de_qc(_spec(n_samples=68, qc_every=11))
    acrescenta = _posicoes_de_qc(_spec(n_samples=68, qc_every=68))

    assert len(substitui) == 7
    assert len(acrescenta) == 2
    assert len(substitui) != len(acrescenta), "os dois ramos têm de produzir desfechos distintos"
    assert substitui[-1] == acrescenta[-1] == 67, "o índice final é comum aos dois"


def test_nenhuma_posicao_se_repete_nem_excede_a_corrida() -> None:
    """Invariante estrutural sobre a grade toda, e ela decorre do desenho e não de observação.

    Índice repetido daria duas amostras com o mesmo lugar na ordem de injeção, e índice fora da
    faixa quebraria a indexação da matriz. Vale para qualquer par, incluindo os que o `else` cobre.
    """
    for n_samples in (2, 10, 30, 68, 200):
        for qc_every in (1, 2, 7, 11, 34, 68, 100):
            if qc_every > n_samples:
                continue
            posicoes = _posicoes_de_qc(_spec(n_samples=n_samples, qc_every=qc_every))
            assert len(set(posicoes)) == len(posicoes), f"repetida em {n_samples}/{qc_every}"
            assert all(0 <= p < n_samples for p in posicoes), (
                f"fora da faixa em {n_samples}/{qc_every}"
            )
            assert list(posicoes) == sorted(posicoes), f"fora de ordem em {n_samples}/{qc_every}"
