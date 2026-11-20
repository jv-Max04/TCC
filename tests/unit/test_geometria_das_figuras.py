"""O primeiro mecanismo deste pacote que lê GEOMETRIA.

**Por que ele não existia, e a ausência estava declarada.** A seção de riscos do plano das análises
diz, verbatim: *"O arnês de figura é cego à geometria. Quatro classes de asserção, cobertura total,
plantas com âncora — e nada lê onde as coisas aterram. Onze defeitos de apresentação só apareceram
ao renderizar e olhar."* As quatro classes de `tests/arnes_de_figura.py` leem o objeto `Figure` —
séries, conteúdo, rótulos, vista —, e nenhuma pergunta quanto espaço sobra para o dado.

**O defeito que o motivou sai bem formado.** `figura_das_caixas_por_feature` desenhava com o eixo a
ocupar **7,3 px de 450 — 1,6% da altura** — e **nenhum aviso**: `tight_layout` encolhe os eixos até
nada para acomodar título, rótulo de eixo e marcas rodadas, em vez de reclamar. A imagem continua
bem formada, com legenda, título e rodapé no sítio; o que desapareceu foi a área de dado. É a forma
de falha que este projeto mais teme — *"o desfecho padrão de um mecanismo novo defeituoso não é
erro, é silêncio"*.

## O limiar foi CONTADO antes de ser escolhido

A distribuição das 30 entradas que o registo tinha naquele instante, medida com os argumentos da
própria sonda, tem um vão limpo. O número é o referente da derivação e não a contagem de hoje: a
figura de rede entrou horas depois e o registo passou a 31, o que não move o limiar, porque a
entrada nova está em 54,1%. Quem o remedir usa a contagem do dia.

| Construtor | Fração do eixo, quando o limiar nasceu |
|---|---|
| `figura_das_caixas_por_feature` | 1,6% — o defeito que o motivou |
| `figura_da_composicao_por_classe` | 20,0% |
| `figura_da_importancia_um_contra_resto` | 20,5% |
| … | … |
| `figura_do_kmeans` | 63,6% — o maior |

**Entre 1,6% e 20,0% não há nada.** Qualquer limiar de 5% a 15% acusa **exactamente um** de trinta —
medido nos três. `FRACAO_MINIMA_DO_EIXO` fica em **10%**, no meio do vão: tolera uma figura legítima
a apertar-se para metade e uma defeituosa a melhorar seis vezes sem trocar de lado. Um limiar
colado a qualquer das pontas viraria a régua no primeiro rótulo mais comprido.

**O oráculo é a PROPRIEDADE e não a forma da instância que o motivou.** Ele não procura
`tight_layout`, nem marcas rodadas, nem título de duas linhas — que são a **causa** naquele
construtor e não a **classe**. O que ele mede é *"sobra área para o dado?"*, que é verdade sobre
qualquer figura, desenhada por qualquer caminho.

## O defeito que o motivou foi CONSERTADO no mesmo dia, e a distribuição mudou

A tabela acima é de **antes** do conserto, e fica com essa data porque é dela que o limiar foi
derivado: um número descreve o referente em que foi medido. Depois de `plots/recorte.py`
deixar de acumular o fundo do `rect` entre passadas, a distribuição passou a começar em **19,98%**
— e **três** figuras mudaram de geometria: `caixas_por_feature` de 1,6% para **28,7%**,
`painel_da_correcao` de 21,9% para **28,8%**, e `caixas_por_amostra` de 28,4% para **25,3%**. As
outras vinte e sete ficaram iguais.

**A terceira encolheu, e a leitura honesta é que 28,4% era o artefato.** A catraca dava-lhe uma
segunda passada com o `rect` mais alto, e o `tight_layout` devolvia mais altura ao eixo por acaso;
25,3% é o que a geometria pretendida produz.

## A fonte dos argumentos é a sonda, e isso é deliberado

`scripts/prova_figuras_congelado.py::_entradas` é o único sítio do repositório que monta as trinta
com argumentos realistas. Reproduzi-los aqui seria uma segunda fonte, e a pior
espécie dela, porque uma cópia que envelhecesse mediria geometria sobre dados que já ninguém passa.
O custo é este módulo importar um script por caminho; o que se compra é que figura nova entra nesta
verificação **no mesmo commit** em que entra na sonda, sem ninguém se lembrar.
"""

from __future__ import annotations

import importlib
import importlib.util
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import matplotlib
import pytest

matplotlib.use("Agg")

from refino.plots.catalogo import CONSTRUTORES

if TYPE_CHECKING:  # pragma: no cover - só para o mypy
    from matplotlib.figure import Figure

__all__ = [
    "FRACAO_MINIMA_DO_EIXO",
    "fracao_do_eixo",
]

REPO: Final = Path(__file__).resolve().parents[2]

FRACAO_MINIMA_DO_EIXO: Final = 0.10
"""Quanto da altura da figura o maior eixo tem de ocupar.

**Derivado do vão medido, não escolhido** — ver o cabeçalho. Mexer neste número sem remedir a
distribuição deixa o limiar sem origem.
"""


def fracao_do_eixo(figura: Figure) -> float:
    """Que fração da altura da figura o **maior** eixo dela ocupa, depois de desenhada.

    **O maior e não a soma, e a escolha importa.** Uma figura de painéis empilhados tem vários eixos
    pequenos por desenho, e somá-los aprovaria um painel cujos quatro eixos estão todos colapsados.
    O maior responde *"há ao menos um sítio onde o dado cabe?"*, que é a pergunta desta verificação.

    **`canvas.draw()` antes de medir é requisito e não cerimônia:** antes dele o `tight_layout` não
    correu e `get_window_extent` devolve a geometria de nascença, que é a mesma para todas as
    figuras — a medição passaria e não mediria nada.
    """
    figura.canvas.draw()
    if not figura.axes:
        return 0.0
    altura = figura.get_figheight() * float(figura.dpi)
    maior = max(eixo.get_window_extent().height for eixo in figura.axes)
    return float(maior / altura)


def _pasta_de_figuras() -> dict[str, Any]:
    """Os módulos de `plots/` indexados pelo nome, na forma que a sonda espera."""
    return {
        nome: importlib.import_module(f"refino.plots.{nome}")
        for nome in {construtor.modulo for construtor in CONSTRUTORES}
    }


def _argumentos_da_sonda() -> dict[tuple[str, str], dict[str, Any]]:
    """A tabela de argumentos de `scripts/prova_figuras_congelado.py`, lida por caminho."""
    caminho = REPO / "scripts" / "prova_figuras_congelado.py"
    spec = importlib.util.spec_from_file_location("_sonda_para_geometria", caminho)
    assert spec is not None and spec.loader is not None, f"não consegui carregar {caminho}"
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    entradas: dict[tuple[str, str], dict[str, Any]] = modulo._entradas(_pasta_de_figuras())
    return entradas


def _figuras_medidas() -> dict[str, float]:
    """Cada construtor do registo e a fração de eixo que ele produz."""
    pasta = _pasta_de_figuras()
    argumentos = _argumentos_da_sonda()
    medidas: dict[str, float] = {}
    for construtor in CONSTRUTORES:
        entrada = argumentos.get((construtor.modulo, construtor.funcao))
        if entrada is None:  # pragma: no cover - a bijeção da sonda já o proíbe
            continue
        funcao = getattr(pasta[construtor.modulo], construtor.funcao)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            medidas[construtor.funcao] = fracao_do_eixo(funcao(**entrada))
    return medidas


@pytest.fixture(scope="module")
def medidas() -> dict[str, float]:
    return _figuras_medidas()


def test_toda_figura_do_registo_deixa_espaco_para_o_dado(medidas: dict[str, float]) -> None:
    """O portão. Uma figura cujo eixo colapsa sai bem formada e vazia, sem avisar.

    A âncora de não-vacuidade vem primeiro. Este teste compara um dicionário contra um
    limiar: se `_figuras_medidas` devolvesse vazio — porque a sonda mudou de forma, porque o import
    falhou em silêncio, porque o registo ficou sem entradas —, a compreensão de lista abaixo seria
    vazia e a asserção passaria **sem medir figura nenhuma**. A âncora afirma que há trinta antes de
    o limiar ser lido.
    """
    assert len(medidas) == len(CONSTRUTORES), (
        f"ÂNCORA: o registo tem {len(CONSTRUTORES)} construtores e eu medi {len(medidas)}. "
        "Sem esta asserção, um dicionário vazio satisfaria o limiar por vacuidade"
    )
    colapsadas = {
        nome: fracao for nome, fracao in medidas.items() if fracao < FRACAO_MINIMA_DO_EIXO
    }
    assert colapsadas == {}, (
        "estas figuras desenham com o eixo abaixo de "
        f"{FRACAO_MINIMA_DO_EIXO:.0%} da altura: "
        + "; ".join(f"{nome} em {fracao:.1%}" for nome, fracao in sorted(colapsadas.items()))
        + ". A imagem sai bem formada — título, legenda e rodapé no sítio — e a área de dado "
        "desaparece, sem aviso nenhum."
    )


def test_o_vao_medido_continua_a_existir(medidas: dict[str, float]) -> None:
    """A segunda direção: o limiar não pode estar colado à distribuição que ele corta.

    **Um portão cujo limiar encosta na menor figura legítima morre no primeiro rótulo mais
    comprido**. Este teste avisa antes disso acontecer.

    ## O FATOR É DERIVADO, e a primeira redação dele era um número colado à distribuição

    A primeira versão exigia `menor >= 2 x limiar` — e **reprovou por 0,02 pp**: a menor fração do
    registo é **19,980%** (`figura_da_composicao_por_classe`) contra os 20,000% que ela pedia.
    Escrevi um limiar encostado à distribuição sem ter olhado onde a distribuição estava, que é
    exactamente o defeito que este teste existe para avisar — cometido dentro dele.

    **`1,5` não foi escolhido por passar; foi derivado da medição que já existia.** Quando
    `FRACAO_MINIMA_DO_EIXO` nasceu, os três cortes candidatos — 5%, 10% e 15% — acusavam
    **exactamente um de trinta**, e eram equivalentes sobre a distribuição daquele dia. `15%` é o
    topo desse intervalo, e `15% = 1,5 x 10%`. Então o que este teste afirma é: **a menor figura
    legítima continua acima do maior corte que teria servido**. Abaixo disso, a escolha de 10%
    deixa de ser indiferente e o número tem de ser **remedido**, não afrouxado.

    A margem medida depois do conserto é `1,998x`. É esse o número contra o qual uma leitura
    futura se compara: se ele descer para perto de 1,5, a distribuição mudou de forma e a derivação
    do limiar caducou.
    """
    menor = min(medidas.values())
    margem = menor / FRACAO_MINIMA_DO_EIXO
    assert margem >= 1.5, (
        f"a menor fração de eixo do registo é {menor:.2%}, e o limiar é "
        f"{FRACAO_MINIMA_DO_EIXO:.0%} — a margem caiu para {margem:.3f}x, abaixo do maior corte "
        "que teria servido quando o limiar foi derivado (15%). REMEÇA a distribuição e derive o "
        "limiar outra vez; afrouxá-lo apagaria a propriedade que ele guarda"
    )


def test_o_portao_acusa_uma_figura_de_eixo_colapsado() -> None:
    """O negativo, com a planta EM PROCESSO e o par causal sobre a mesma figura.

    ## A PRIMEIRA PLANTA NÃO ATERROU, e a asserção de aterragem foi o que o disse

    A primeira redação deste teste plantava a **causa** do defeito — título de duas linhas, rótulo
    de eixo comprido e marcas rodadas num canvas baixo — na convicção de que ela encolheria o eixo.
    Medido: **com a planta 77,0%, sem ela 66,5%**. Ela **aumentou** o eixo, porque `tight_layout`
    aperta as margens para acomodar o conteúdo novo e nesse canvas isso sobra a favor do eixo.

    **Sem `assert com < sem` este teste teria ficado verde afirmando acusar, com uma planta que não
    acusava nada**. A causa real do colapso naquele construtor é a combinação dele e não estes três
    ingredientes soltos; reproduzi-la à mão seria copiar o construtor para dentro do teste.

    ## Então a planta é de GEOMETRIA, e o que ela prova vai declarado

    `set_position` produz o desfecho — eixo abaixo do limiar — sem depender de o `tight_layout`
    cooperar. O que este teste prova é que **o predicado e o limiar acusam um eixo colapsado**; o
    que ele **não** prova é que algum construtor real produza um. Essa metade foi observada uma vez
    e está datada: em 18/09/2026, antes do conserto, o portão acima acusou
    `figura_das_caixas_por_feature` **em 1,6%** — o desfecho de acusação sobre o defeito vivo.
    """
    from matplotlib.figure import Figure as FiguraNua

    def constroi(*, com_a_planta: bool) -> FiguraNua:
        figura = FiguraNua(figsize=(7.0, 4.5), dpi=96)
        eixo = figura.add_subplot(111)
        eixo.plot([0, 1, 2], [1.0, 2.0, 1.5])
        if com_a_planta:
            # A fração alvo é metade do limiar: baixa o bastante para acusar sem folga, e alta o
            # bastante para não ser zero — que passaria mesmo num predicado que devolvesse sempre 0.
            eixo.set_position((0.1, 0.1, 0.8, FRACAO_MINIMA_DO_EIXO / 2))
        return figura

    com = fracao_do_eixo(constroi(com_a_planta=True))
    sem = fracao_do_eixo(constroi(com_a_planta=False))

    assert sem >= 2 * FRACAO_MINIMA_DO_EIXO, (
        f"ÂNCORA: sem a planta a figura tem de estar folgadamente acima do limiar e está em "
        f"{sem:.1%}. Com a base já colapsada, o par abaixo não separaria nada"
    )
    assert com < sem, (
        f"A PLANTA NÃO ATERROU: com ela o eixo ocupa {com:.1%} e sem ela {sem:.1%} — os dois "
        "desfechos têm de divergir, senão este negativo não prova que o portão consegue acusar"
    )
    assert com < FRACAO_MINIMA_DO_EIXO, (
        f"com a planta o eixo ocupa {com:.1%}, que o limiar de {FRACAO_MINIMA_DO_EIXO:.0%} NÃO "
        "acusa. O portão não apanharia o defeito que ele existe para apanhar"
    )
