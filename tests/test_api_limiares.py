"""Os limiares do veredito controláveis pelo cliente.

## O que muda, e o que deliberadamente NÃO muda

O veredito é leitura e não portão, e os três limiares são configuráveis. Antes, nenhuma superfície
os deixava mudar: as interfaces os exibiam e ninguém os escolhia.

**Só o `verdict` é recalculado.** Os seis números por feature são medições e não se tocam; o
veredito é um **corte** sobre dois deles. É isso que torna a operação barata — mudar de limiar não
obriga a recorrer uma grade de 4.320 células.

## Por que QUERY e não corpo do disparo

Pô-los em `POST /multiversos` faria mudar de opinião custar uma grade nova. A grade padrão leva
de ~12 a ~32 min conforme o modo e o corretor.

## A guarda, e o modo de falha que ela evita

Os três são frações, e sem guarda o FastAPI aceitaria `7.5` sem uma palavra. O modo de falha é o
insidioso, o que **não** levanta: `frac_significant >= 7.5` é `False` para tudo, então **nenhuma**
feature seria `robust` e a tabela sairia plausível — e vazia de robustas — sem nada a acusar.
"""

from __future__ import annotations

import pytest
from fastapi import HTTPException

from refino.api.esquemas import EstabilidadeDeFeature
from refino.api.rotas import _limiares_ou_422, _reclassifica
from refino.multiverse.stability import StabilityThresholds

PADRAO = StabilityThresholds()


def _medida(nome: str, frac: float, sinal: float | None) -> EstabilidadeDeFeature:
    """Uma linha da tabela de estabilidade, com os campos que o corte NÃO olha em valores fixos."""
    return EstabilidadeDeFeature(
        feature=nome,
        frac_significant=frac,
        median_qvalue=0.01,
        median_rank=1.0,
        rank_iqr=0.0,
        sign_consistency=sinal,
        n_cells=4320,
        verdict="fragile",
    )


# ---------------------------------------------------------------------------
# A fronteira
# ---------------------------------------------------------------------------


def test_sem_nenhum_limiar_os_tres_sao_os_da_especificacao() -> None:
    """O caminho de antes, intacto. Os valores padrão não são redigitados aqui.

    Comparar contra `StabilityThresholds()` e não contra `0.80` escrito à mão é o que impede este
    teste de defender o número velho no dia em que o padrão mudar: seria a segunda fonte de verdade
    sobre um limiar.
    """
    assert _limiares_ou_422(None, None, None) == PADRAO


@pytest.mark.parametrize("qual", ["robust_frac", "robust_sign", "absent_frac"])
def test_cada_limiar_pode_ser_mudado_sozinho_e_os_outros_dois_ficam(qual: str) -> None:
    """**Um de cada vez, e é a direção que uma implementação desatenta perderia.**

    Um `if` que substituísse os três quando qualquer um viesse passaria num teste que mudasse os
    três juntos, e apagaria em silêncio os dois que o cliente não pediu para mudar.
    """
    pedido: dict[str, float | None] = dict.fromkeys(("robust_frac", "robust_sign", "absent_frac"))
    pedido[qual] = 0.42
    limiares = _limiares_ou_422(**pedido)

    assert getattr(limiares, qual) == 0.42
    for outro in ("robust_frac", "robust_sign", "absent_frac"):
        if outro != qual:
            assert getattr(limiares, outro) == getattr(PADRAO, outro), (
                f"mudar `{qual}` apagou `{outro}`"
            )


@pytest.mark.parametrize("valor", [-0.1, 1.1, 7.5, 100.0])
def test_fracao_fora_de_zero_um_e_recusada_com_422(valor: float) -> None:
    """A guarda de fronteira num domínio que não é `Literal`.

    **`7.5` é o caso que dá nome ao defeito:** ele não levanta em lado nenhum, torna a condição
    impossível, e a tabela sai sem features robustas. A mensagem tem de dizer que são **frações**,
    porque o número sozinho não o diz.
    """
    with pytest.raises(HTTPException) as erro:
        _limiares_ou_422(valor, None, None)

    assert erro.value.status_code == 422
    assert "FRAÇÕES" in str(erro.value.detail) or "fração" in str(erro.value.detail).lower()
    assert str(valor) in str(erro.value.detail), "a recusa não diz qual valor foi recusado"


@pytest.mark.parametrize("extremo", [0.0, 1.0])
def test_os_extremos_do_intervalo_sao_aceitos(extremo: float) -> None:
    """**Fechado nas duas pontas, e a escolha é declarada.**

    `absent_frac=0` significa *"nada é ausente"* e `robust_frac=1` significa *"só o unânime é
    robusto"*. As duas são leituras legítimas de um perfil, e recusá-las seria inventar uma
    restrição que ninguém declarou.
    """
    assert _limiares_ou_422(extremo, extremo, extremo) == StabilityThresholds(
        robust_frac=extremo, robust_sign=extremo, absent_frac=extremo
    )


# ---------------------------------------------------------------------------
# A reclassificação: só o veredito muda, e ele MUDA
# ---------------------------------------------------------------------------


def test_baixar_o_piso_torna_robusta_uma_feature_que_era_fragil() -> None:
    """A planta cai onde as duas coisas diferem.

    Uma feature com `frac_significant=0.60` e sinal consistente é `fragile` sob o piso padrão de
    0,80 e `robust` sob um piso de 0,50. Sem esta divergência o teste passaria com uma implementação
    que ignorasse os limiares por completo.
    """
    medidas = (_medida("f1", 0.60, 0.99),)

    padrao = _reclassifica(medidas, PADRAO)
    frouxo = _reclassifica(medidas, _limiares_ou_422(0.50, None, None))

    assert padrao[0].verdict != frouxo[0].verdict, (
        "o veredito não mudou ao baixar o piso: os limiares podem não estar a ser lidos"
    )
    assert frouxo[0].verdict == "robust"


def test_a_reclassificacao_nao_toca_em_nenhuma_medicao() -> None:
    """**O corte muda; a medida não.** É o que torna a operação honesta e barata.

    Se a reclassificação mexesse num dos seis números, mudar de limiar deixaria de ser leitura e
    passaria a ser uma segunda medição — e duas leituras da mesma execução ficariam incomparáveis.
    """
    medidas = (_medida("f1", 0.60, 0.99), _medida("f2", 0.02, None))

    saida = _reclassifica(medidas, _limiares_ou_422(0.50, 0.50, 0.50))

    assert len(saida) == len(medidas), "a reclassificação perdeu ou inventou features"
    for antes, depois in zip(medidas, saida, strict=True):
        assert depois.model_dump(exclude={"verdict"}) == antes.model_dump(exclude={"verdict"}), (
            f"a reclassificação alterou uma MEDIÇÃO de `{antes.feature}`, e não só o corte"
        )


def test_sinal_nao_medido_nao_pode_ser_robusto_por_mais_frouxo_que_o_limiar_seja() -> None:
    """`sign_consistency` nulo é *"não medido"*, e `nan >= x` é `False` para todo `x`.

    **É o desfecho honesto e não um efeito colateral:** sem consistência de sinal medida não há
    como afirmar robustez. Tratá-lo como `0.0` diria *"medi, e deu zero"*, que é outra coisa — e
    com limiar `0.0` tornaria a feature robusta, que é exactamente o que não se pode afirmar.
    """
    medidas = (_medida("sem_sinal", 1.0, None),)

    saida = _reclassifica(medidas, _limiares_ou_422(0.0, 0.0, 0.0))

    assert saida[0].verdict != "robust", (
        "uma feature sem consistência de sinal medida foi declarada robusta"
    )
