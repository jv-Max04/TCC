"""A margem relativa diz-se «relativa» e «do melhor» onde se escreve com «%».

## O defeito: rótulo certo sobre a quantidade errada

A margem entre o melhor e o segundo é RELATIVA — `(segundo - melhor) / |melhor|`, em
`Label.margin` e em `palpite_da_amostra` — e era escrita só com «%» em textos ao lado de RSD também
em «%». O custo mede-se na figura: com os escores do `neg_c18`, barras de 15,5% e 15,47% estão a
0,03 ponto percentual, e o rodapé dizia *«Margem dos dois melhores: 0,217%»*. Ali passou a dizer
*«Margem relativa dos dois melhores: X% do melhor»*, e os textos de antes ficaram com a forma velha.

Os cinco textos, conferidos no código — que número é, e relativo a quê:

| Texto | O número, relativo | Ao lado |
|---|---|---|
| `ui/fluxo.prosa_da_medicao` | `margem` de `/matrizes/medir-corretor` | escores em fração |
| `ui/fluxo.prosa_do_palpite_medido` | `margem` e `limiar` do Palpite | nada |
| `PalpiteMedido.rationale` | `margem` de `palpite_da_amostra` | escores em fração |
| ressalva do SERRF (`METHOD_CAVEATS`) | as quatro da F2 da rodada JH | **RSD em «%»** |
| ressalva da constante (`AUTO_CAVEATS`) | duas das mesmas | taxas de acerto em «%» |

**As duas ressalvas são as irmãs**, e a do SERRF é a forma mais ambígua das cinco: *«SERRF,
14,4% — margem 67,88%»* põe um RSD absoluto e uma margem relativa na mesma linha, com o mesmo sinal.

## As irmãs que a revisão achou

A primeira procura foi por «%» ao lado de «margem», e por isso não viu duas classes de irmã: a
margem escrita como **fração**, e a mesma quantidade com o nome **«diferença»**.

| Texto | O número, relativo | Ao lado |
|---|---|---|
| `policy._monta`, as constantes | a margem do estimador, fração | RSD de QC em fração |
| `model.monta_recomendacao` | a margem predita, fração | RSD de QC em fração |
| ressalva do Palpite (`AUTO_CAVEATS`) | o empate de 16,7%, os 20%, duas margens | acertos em «%» |
| evidência da ressalva do SERRF | as outras três margens da F2 | RSD em fração |
| a camada de decisão no recorte (`PARA_QUEM_OPERA`) | as duas margens | acertos em «%» |

A metade das duas justificativas que não tem margem — *«NÃO foi medida»* — não leva número e não
mudou.
"""

from __future__ import annotations

import re

import pytest

from refino.cli import AUTO_CAVEATS, AUTO_CONSTANTE, AUTO_MEDIDA, METHOD_CAVEATS
from refino.decision.model import monta_recomendacao
from refino.decision.palpite import PalpiteMedido
from refino.decision.policy import AlwaysSERRFPolicy
from refino.desktop.recorte import PARA_QUEM_OPERA
from refino.domain.diagnosis import DefectProfile
from refino.interface.fluxo import prosa_da_medicao, prosa_do_palpite_medido


@pytest.mark.parametrize("abstem", [False, True], ids=["elege", "abstem-se"])
def test_d681_a_prosa_da_medicao_diz_que_a_margem_e_relativa_ao_melhor(abstem: bool) -> None:
    """A margem da medição, ao lado dos escores, diz-se relativa e diz a que é relativa."""
    texto = prosa_da_medicao(
        vencedor="serrf",
        scores={"none": 0.1550, "serrf": 0.1547, "loess": 0.1932},
        margem=0.00217,
        avaliavel=True,
        abstem=abstem,
        n_features_medidas=None,
    )

    assert "A **margem relativa** até o segundo é de **0,2%** do melhor" in texto, texto


def test_d681_o_empate_do_palpite_diz_que_a_margem_e_relativa_ao_melhor() -> None:
    """A margem do empate na amostra e o limiar ao lado dela são relativos, e a frase o diz."""
    texto = prosa_do_palpite_medido(
        avaliavel=True, abstem=True, confianca=0.4, margem=0.076, limiar=0.167
    )

    assert "a margem relativa até o segundo colocado, 7,6% do melhor, ficou abaixo de 16,7%" in (
        texto
    ), texto


@pytest.mark.parametrize("margem", [0.5, 0.05], ids=["elege", "empate"])
def test_d681_a_justificativa_do_palpite_diz_que_a_margem_e_relativa_ao_melhor(
    margem: float,
) -> None:
    """A justificativa do Palpite — na resposta da rota e no registro — diz a margem relativa."""
    palpite = PalpiteMedido(
        correction="serrf",
        escores={"none": 0.30, "loess": 0.30, "serrf": 0.20},
        margem=margem,
        avaliavel=True,
        n_features_medidas=200,
        segundo="loess",
    )

    esperado = f"a margem relativa para o segundo, loess, é {margem * 100:.1f}% do melhor"
    assert esperado.replace(".", ",") in palpite.rationale, palpite.rationale


def test_d681_as_ressalvas_dizem_que_a_margem_e_relativa_ao_melhor() -> None:
    """As duas irmãs em `cli.py`: a do SERRF, com RSD em «%» na mesma linha, e a da constante.

    A do SERRF diz uma vez, no cabeçalho da lista, o que a palavra «margem» quer dizer em cada uma
    das quatro linhas abaixo dele; repetir «relativa» em cada linha não acrescenta nada à leitura.
    """
    serrf = METHOD_CAVEATS["serrf"].detalhe
    assert "a margem relativa para o segundo colocado, em percentagem do melhor:" in (
        " ".join(serrf.split())
    ), serrf

    constante = " ".join(AUTO_CAVEATS[AUTO_CONSTANTE].detalhe.split())
    assert "da MARGEM relativa entre o melhor e o segundo — 2,56% do melhor naquele modo" in (
        constante
    ), constante


# ---------------------------------------------------------------------------
# As irmãs da revisão: a margem em fração, e a mesma quantidade chamada «diferença»
# ---------------------------------------------------------------------------


def _perfil() -> DefectProfile:
    """Um perfil cujo RSD mediano de QC, 0,1512, é da ordem da margem — a ambiguidade à vista."""
    return DefectProfile(
        defect_type="gradual_drift",
        confidence=0.8,
        suggested_batches=(1,),
        changepoints=(),
        qc_rsd_median=0.1512,
        qc_rsd_p75=0.30,
        zero_fraction=0.03,
        missing_fraction=0.0,
        pc1_injection_correlation=0.52,
        qc_dispersion_ratio=0.80,
        n_features=2184,
        n_samples=68,
        n_qcs=7,
        raw_features={},
    )


def test_d681_a_justificativa_das_politicas_diz_que_a_margem_e_relativa_ao_melhor() -> None:
    """As justificativas das políticas escrevem a margem em fração, ao lado do RSD em fração.

    *«A margem ... é 0.0256 ... O RSD mediano de QC é 0.1512»* lê-se como 0,0256 de RSD, quando é
    2,56% do RSD do melhor. As duas chegam ao usuário — o `--auto` da CLI, as rotas e a janela.
    """
    constante = AlwaysSERRFPolicy(margin_estimator=lambda _perfil: 0.0256).recommend(_perfil())
    assert (
        "A margem relativa do oráculo entre o melhor e o segundo corretor é 0.0256 do melhor, "
        "contra o limiar de" in constante.rationale
    ), constante.rationale

    aprendida = monta_recomendacao(
        correcao="serrf", margem=0.0256, threshold=0.012, caminho="folha", profile=_perfil()
    )
    assert (
        "A margem relativa PREDITA entre o melhor e o segundo corretor é 0.0256 do melhor, "
        "contra o limiar de 0.0120" in aprendida.rationale
    ), aprendida.rationale


# Uma margem ou uma diferença seguida de um número em «%», na mesma frase, sem «relativa» entre os
# dois. É a forma do defeito, e não uma lista de frases: um texto novo que a repita reprova.
_SEM_RELATIVA = re.compile(
    r"\b(?:margem|margens|diferen[çc]a)\b(?P<entre>[^.%]{0,80}?)\d+(?:,\d+)?%", re.IGNORECASE
)


def _margens_sem_relativa(texto: str) -> list[str]:
    """Os trechos em que a margem ou a diferença vai em «%» sem dizer que é relativa."""
    plano = " ".join(texto.split())
    return [m.group(0) for m in _SEM_RELATIVA.finditer(plano) if "relativ" not in m.group("entre")]


def test_d681_o_detector_acusa_a_margem_sem_relativa_e_poupa_a_relativa() -> None:
    """O par do detector: sem a metade que acusa, os testes abaixo passariam por vacuidade."""
    assert _margens_sem_relativa("a margem 2,56%; a diferença na amostra abaixo de 16,7%") == [
        "margem 2,56%",
        "diferença na amostra abaixo de 16,7%",
    ]
    assert _margens_sem_relativa("a margem relativa 2,56% do melhor; RSD de 15,1%") == []


@pytest.mark.parametrize(
    ("onde", "texto"),
    [
        ("ressalva do Palpite medido", AUTO_CAVEATS[AUTO_MEDIDA].detalhe),
        ("evidência da ressalva do SERRF", METHOD_CAVEATS["serrf"].evidencia),
        ("ressalva da constante", AUTO_CAVEATS[AUTO_CONSTANTE].detalhe),
        (
            "linha da política automática no recorte",
            PARA_QUEM_OPERA["A política de decisão automática"][1],
        ),
    ],
    ids=["palpite-medido", "evidencia-do-serrf", "constante", "fase-9-no-recorte"],
)
def test_d681_nenhuma_margem_destes_textos_vai_em_percentagem_sem_dizer_relativa(
    onde: str, texto: str
) -> None:
    """Toda margem ou diferença em «%» destes textos diz que é relativa.

    **A lista da ressalva do SERRF fica de fora de propósito:** o cabeçalho dela diz uma vez que a
    margem de cada linha é relativa e em percentagem do melhor, e o teste de cima o afirma.
    """
    assert "%" in texto, f"âncora: {onde} não tem «%», e o teste passaria por vacuidade"
    assert not _margens_sem_relativa(texto), (onde, _margens_sem_relativa(texto))
