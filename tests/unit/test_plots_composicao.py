"""A composição por classe.

> **Saiu de `test_plots_correcao.py` em 17/09/2026**, com `plots/composicao.py`.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, get_args

import pytest
from matplotlib.colors import to_hex

from refino.api.esquemas import (
    DEFINICAO_DE_ABUNDANCIA,
    DEFINICAO_DE_FALTANTE,
    DefinicaoDeAbundancia,
    DefinicaoDeFaltante,
)
from refino.domain.labels import Correction
from refino.domain.metadata import SampleType
from refino.plots.base import (
    NAO_DECLARADA,
    MatrizVaziaError,
    nova_figura,
)
from refino.plots.catalogo import CONSTRUTORES, Vinculo
from refino.plots.composicao import (
    ALTURA_DA_COMPOSICAO,
    DEFINICAO_AUSENTE,
    DEFINICOES_DA_ABUNDANCIA,
    DEFINICOES_DO_FALTANTE,
    FOLGA_DA_COMPOSICAO,
    PAPEL_DA_ABUNDANCIA,
    PAPEL_DOS_FALTANTES,
    POPULACAO_NAO_DECLARADA,
    ROTULO_DO_EIXO_DA_ABUNDANCIA,
    SEM_CLASSE_DECLARADA,
    SEM_MARCAS_DE_CLASSE,
    TITULO_DA_COMPOSICAO,
    ComposicaoDasClasses,
    MarcasDasClasses,
    classes_do_fio,
    composicao_do_fio,
    definicao_lida,
    figura_da_composicao_por_classe,
    frase_sem_abundancia,
    grupos_da_composicao,
    marca_da_classe,
    marcas_das_classes,
    recorte_da_composicao_por_classe,
    rotulo_da_classe,
    rotulos_das_classes_cabem,
)
from refino.plots.recorte import (
    em_percentagem,
    linhas_do_rodape,
)
from refino.plots.series import EstiloDoGrupo, estilos_dos_grupos
from refino.reporting.model import ITENS_OBRIGATORIOS
from tests.arnes_de_figura import (
    confere_a_matriz_do_titulo,
    confere_series,
    confere_vista,
    milhar_do_teste,
    planta_corte_de_eixo,
    planta_eixos_trocados,
    sem_a_matriz,
    series,
    texto_corrido,
)

if TYPE_CHECKING:
    from matplotlib.figure import Figure

# **Os três números do PNG que o laboratório lê**, do `pos_hilic`:
# `serrf/data/output/pos_hilic_toserrf - normalization result/Bar Plot and PCA plot.png`. São
# **observação sobre aquele arquivo** e não invariante — aqui servem de forma plausível e
# de âncora da convenção de escrita do valor, que é o que `percentagem_do_r` replica.
SCORES_DO_R: dict[str, float | None] = {"none": 1.447, "batch_loess": 1.050, "SERRF": 0.693}

N_AMOSTRAS = 24

# **O vocabulário fechado dos corretores, DERIVADO e não redigitado.** A lista existe aqui para uma
# asserção só — que o recuo do eixo direito não seja um nome de corretor —, e uma cópia à mão
# deixaria de acusar no dia em que o vocabulário ganhasse um membro chamado como o recuo.
VOCABULARIO_DOS_CORRETORES: tuple[str, ...] = get_args(Correction)


# ---------------------------------------------------------------------------
# A composição por classe
# ---------------------------------------------------------------------------
#
# **As fixturas têm classes com contagens DIFERENTES em tudo** — injeções, células, faltantes e
# abundância —, porque é isso que torna as plantas honestas: duas classes iguais fariam a troca de
# uma pela outra ser a identidade.
#
# **E `n_celulas` NÃO é `n_injecoes * N_FEATURES_DA_COMPOSICAO`.** A distância é deliberada: uma
# figura que derivasse o denominador do par em vez de o ler da classe daria número certo enquanto as
# duas coisas coincidissem.

N_FEATURES_DA_COMPOSICAO = 6035
N_INJECOES_FORA = 7


def classe_de_prova(
    grupo: str = "ATLL",
    *,
    n_injecoes: int = 13,
    n_celulas: int = 78455,
    n_zeros: int = 12930,
    n_nao_finitas: int = 0,
    n_detectadas: int | None = None,
    log10_mediana: Any = 4.0507,
) -> dict[str, Any]:
    """Uma entrada da lista `classes` do corpo, na forma que `ClasseDaComposicao` declara."""
    faltantes = n_zeros + n_nao_finitas
    return {
        "grupo": grupo,
        "n_injecoes": n_injecoes,
        "n_celulas": n_celulas,
        "n_zeros": n_zeros,
        "n_nao_finitas": n_nao_finitas,
        "n_faltantes": faltantes,
        "n_detectadas": n_celulas - faltantes if n_detectadas is None else n_detectadas,
        "log10_mediana_das_detectadas": log10_mediana,
    }


def composicao_de_prova(**troca: Any) -> ComposicaoDasClasses:
    """Duas classes com tudo diferente, e os seis rótulos do conjunto."""
    campos: dict[str, Any] = {
        "classes": [
            classe_de_prova(),
            classe_de_prova(
                "NoHTLV", n_injecoes=16, n_celulas=96560, n_zeros=15568, log10_mediana=4.0791
            ),
        ],
        "matriz": "ingerida",
        "populacao": "sample",
        "n_features": N_FEATURES_DA_COMPOSICAO,
        "n_injecoes_fora": N_INJECOES_FORA,
        "definicao_de_faltante": DEFINICAO_DE_FALTANTE,
        "definicao_de_abundancia": DEFINICAO_DE_ABUNDANCIA,
    }
    campos.update(troca)
    return composicao_do_fio(**campos)


def composicao_desenhada(composicao: ComposicaoDasClasses | None = None) -> Figure:
    alvo = composicao if composicao is not None else composicao_de_prova()
    return figura_da_composicao_por_classe(
        composicao=alvo, recorte=recorte_da_composicao_por_classe(composicao=alvo)
    )


def series_da_composicao(figura: Figure) -> dict[str, Any]:
    """Todas as séries dos dois painéis, indexadas pelo rótulo — que é único."""
    return {s.rotulo: s for eixo in figura.axes for s in series(eixo)}


def test_os_TOKENS_do_corpo_sao_exactamente_as_chaves_que_a_figura_traduz() -> None:  # noqa: N802
    """**O par que impede a definição de virar prosa divergente** — e ele é DERIVADO.

    O corpo carrega um **token** e a figura guarda a **frase**; são dois sítios, e o que os liga é
    esta asserção. Ela deriva as chaves esperadas de `get_args` dos `Literal` de `api/esquemas.py`
    em vez de as redigitar: uma definição nova na rota passa a reprovar aqui, em vez de cair no ramo
    de recuo e pôr *"NÃO RECONHECIDA"* na imagem sem ninguém ver.

    **As asserções de não-vacuidade vêm primeiro e na mesma função**: `frozenset()` é
    subconjunto de tudo.
    """
    do_faltante = frozenset(get_args(DefinicaoDeFaltante))
    da_abundancia = frozenset(get_args(DefinicaoDeAbundancia))
    assert do_faltante and da_abundancia, (do_faltante, da_abundancia)
    assert DEFINICOES_DO_FALTANTE and DEFINICOES_DA_ABUNDANCIA

    assert do_faltante == frozenset(DEFINICOES_DO_FALTANTE), (
        f"o vocabulário do corpo é {sorted(do_faltante)} e a figura traduz "
        f"{sorted(DEFINICOES_DO_FALTANTE)}: um token servido e não traduzido sai da imagem como "
        "«definição NÃO RECONHECIDA»"
    )
    assert da_abundancia == frozenset(DEFINICOES_DA_ABUNDANCIA), (
        f"{sorted(da_abundancia)} contra {sorted(DEFINICOES_DA_ABUNDANCIA)}"
    )
    assert {DEFINICAO_DE_FALTANTE} <= do_faltante
    assert {DEFINICAO_DE_ABUNDANCIA} <= da_abundancia


def test_um_token_DESCONHECIDO_e_nomeado_e_nao_cai_numa_definicao_plausivel() -> None:  # noqa: N802
    """**As duas direções de `definicao_lida`**, e o ramo de recuo é o que importa.

    Um mapa de **um** membro torna a escolha *"devolver o único"* invisivelmente errada: a imagem
    declararia uma definição que a rota não usou, e nada levantaria — é o `else` razoável, o
    modo insidioso.
    """
    conhecida = definicao_lida(DEFINICAO_DE_FALTANTE, DEFINICOES_DO_FALTANTE)
    desconhecida = definicao_lida("zero", DEFINICOES_DO_FALTANTE)
    assert "is_missing" in conhecida, conhecida
    assert conhecida != desconhecida
    assert "NÃO RECONHECIDA" in desconhecida and "«zero»" in desconhecida, desconhecida
    assert "is_missing" not in desconhecida, (
        "o token desconhecido caiu na frase do conhecido: a imagem passa a declarar uma definição "
        "que a rota não usou"
    )

    rodape = recorte_da_composicao_por_classe(
        composicao=composicao_de_prova(definicao_de_faltante="zero")
    ).texto()
    assert "NÃO RECONHECIDA" in rodape, rodape


def test_a_percentagem_sai_de_N_CELULAS_DA_CLASSE_e_nao_do_par() -> None:  # noqa: N802
    """**O denominador é da classe** — e o par causal cai sobre dado em que as duas coisas diferem.

    Na fixtura, `n_injecoes * n_features` da primeira classe dá `13 x 6.035 = 78.455`, que **é**
    `n_celulas`; por isso a planta muda `n_features` do par e afirma que as alturas **não** se
    mexem. A âncora de que a planta aterrou vem primeiro.
    """
    normal = composicao_de_prova()
    plantada = composicao_de_prova(n_features=999)
    assert normal.n_features != plantada.n_features, "âncora: a planta não aterrou"

    das_normais = {
        r: max(s.y) for r, s in series_da_composicao(composicao_desenhada(normal)).items()
    }
    das_plantadas = {
        r: max(s.y) for r, s in series_da_composicao(composicao_desenhada(plantada)).items()
    }
    assert das_normais, "âncora: a figura não desenhou artista nenhum"
    assert das_normais == pytest.approx(das_plantadas), (
        f"mudar `n_features` do PAR mudou as barras: {das_normais} contra {das_plantadas}. O "
        "denominador da percentagem é `n_celulas` da classe"
    )

    esperada = 12930 / 78455 * 100.0
    assert das_normais["ATLL — faltantes"] == pytest.approx(esperada), das_normais


def test_cada_classe_tem_DOIS_artistas_com_rotulos_unicos() -> None:  # noqa: N802
    """`confere_series` indexa por rótulo, e duas barras da mesma classe colapsariam numa.

    **A planta mostra que o colapso aconteceria de facto**: com o nome do grupo sozinho como
    rótulo, os dois painéis teriam o mesmo texto e o arnês veria metade dos artistas.
    """
    figura = composicao_desenhada()
    desenhadas = series_da_composicao(figura)
    assert len(desenhadas) == 4, sorted(desenhadas)
    assert set(desenhadas) == {
        rotulo_da_classe(grupo, papel)
        for grupo in ("ATLL", "NoHTLV")
        for papel in (PAPEL_DOS_FALTANTES, PAPEL_DA_ABUNDANCIA)
    }
    assert len({rotulo.split(" — ")[0] for rotulo in desenhadas}) == 2, (
        "âncora: sem o papel os quatro rótulos seriam dois, e o arnês veria metade dos artistas"
    )


def test_a_composicao_limpa_passa_na_classe_c_e_as_plantas_acusam() -> None:
    """A classe (c) sobre os dois eixos, e as duas plantas que ela e a (a) apanham.

    **A planta entra como argumento**, então o negativo percorre exactamente o caminho que o portão
    percorre — negativo em processo não consegue ficar inerte.
    """
    for eixo in composicao_desenhada().axes:
        assert series(eixo), f"o eixo «{eixo.get_title()}» não tem artista de dado nenhum"
        assert confere_vista(eixo) == [], eixo.get_title()

    cortado = composicao_desenhada().axes[0]
    planta_corte_de_eixo(cortado)
    assert confere_vista(cortado) != []

    trocado = composicao_desenhada().axes[0]
    esperadas = {s.rotulo: (s.x.tolist(), s.y.tolist()) for s in series(trocado)}
    planta_eixos_trocados(trocado)
    assert confere_series(trocado, esperadas) != []


def test_nenhum_eixo_da_composicao_e_logaritmico() -> None:
    """**A tentação é real e a resposta é o VALOR e não a escala.** A abundância vive entre `10^3` e
    `10^5`, e um eixo logarítmico seria o gesto natural — `test_u12_nenhum_dos_eixos_...` proíbe-o,
    e com razão: `scroll_handler` amplia em unidades de eixo. O que se desenha é `log10(x)` num eixo
    **linear**, e a intensidade em unidades brutas vai anotada sobre a barra."""
    figura = composicao_desenhada()
    assert len(figura.axes) == 2, [a.get_title() for a in figura.axes]
    for eixo in figura.axes:
        assert (eixo.get_xscale(), eixo.get_yscale()) == ("linear", "linear"), eixo.get_title()
    assert texto_corrido(sem_a_matriz(figura.texts[0].get_text())) == TITULO_DA_COMPOSICAO
    assert confere_a_matriz_do_titulo(figura) == []


def test_a_anotacao_da_abundancia_e_a_intensidade_BRUTA_e_o_eixo_e_o_LOG() -> None:  # noqa: N802
    """As duas unidades convivem, e cada uma no sítio em que é legível.

    `10^4,0507 = 11.239`: a barra está em 4,0507 e a anotação diz 11.239. **As duas afirmações têm
    de ser derivadas aqui**, do valor do corpo, e não pedidas ao construtor — senão o teste
    confirmaria o que o construtor escolheu em vez do que ele devia escolher.
    """
    figura = composicao_desenhada()
    da_abundancia = figura.axes[1]
    assert max(series_da_composicao(figura)["ATLL — abundância"].y) == pytest.approx(4.0507)

    anotado = {texto_corrido(t.get_text()) for t in da_abundancia.texts}
    esperado = milhar_do_teste(round(10.0**4.0507))
    assert esperado in anotado, (esperado, sorted(anotado))
    assert "4,051" not in anotado, (
        "a anotação está em log e não em unidades brutas: `10^4,05` não é um número que alguém leia"
    )
    assert ROTULO_DO_EIXO_DA_ABUNDANCIA in da_abundancia.get_ylabel()


def test_uma_classe_SEM_detectada_nao_vira_barra_e_a_ausencia_e_DECLARADA() -> None:  # noqa: N802
    """**`None` no fio vira `nan` aqui e NUNCA `0.0`**, na ponta do desenho.

    Uma barra no piso do eixo diria um nível de sinal plausível e falso. **O par causal separa**:
    com abundância há duas barras no painel de baixo; sem ela há uma, e a frase nomeia a classe.
    """
    completa = composicao_de_prova()
    parcial = composicao_de_prova(
        classes=[
            classe_de_prova(),
            classe_de_prova(
                "NoHTLV",
                n_injecoes=16,
                n_celulas=96560,
                n_zeros=96560,
                n_detectadas=0,
                log10_mediana=None,
            ),
        ]
    )
    assert not completa.sem_abundancia, "âncora: a fixtura completa já tinha classe sem abundância"
    assert [c.grupo for c in parcial.sem_abundancia] == ["NoHTLV"], parcial.sem_abundancia
    assert not math.isfinite(parcial.classes[1].log10_mediana), parcial.classes[1]

    da_completa = series_da_composicao(composicao_desenhada(completa))
    da_parcial = series_da_composicao(composicao_desenhada(parcial))
    assert "NoHTLV — abundância" in da_completa
    # **A asserção de PRESENÇA vem antes da de ausência**: com `da_parcial` vazio, «a barra
    # não está lá» passaria sobre uma figura que não desenhou coisa nenhuma.
    assert "NoHTLV — faltantes" in da_parcial, (
        "a classe sem detectada perdeu também a barra de faltantes: 100% de ausência É o fato dela"
    )
    assert "NoHTLV — abundância" not in da_parcial, sorted(da_parcial)
    assert max(da_parcial["NoHTLV — faltantes"].y) == pytest.approx(100.0)
    assert "NoHTLV" in frase_sem_abundancia(parcial)


def test_com_NENHUMA_classe_detectada_as_MARCAS_sobrevivem_no_painel_de_cima() -> None:  # noqa: N802
    """**O QUARTO desfecho**, que a enumeração do módulo não tinha.

    Ele não é a soma dos vizinhos. *"Uma classe sem detectada"* deixa o painel de baixo a desenhar
    e a rotular; *"zero classes"* não tem partição a mostrar. Aqui há partição **e** o painel de
    baixo desliga — `_desenha_a_abundancia` cai em `_declara`, faz `set_axis_off()` e nunca chega
    a `_rotula_as_classes`. Com o painel de cima a esconder as marcas incondicionalmente, os nomes
    das classes **desapareciam dos dois**, e o que ficava era uma fila de barras anónimas.

    **O par causal separa, e as duas metades divergem:** com detectadas, as marcas vivem no painel
    de **baixo** e o de cima está limpo; sem nenhuma, elas vivem no de **cima**, que é o único
    visível. Em qualquer dos dois, a partição continua legível — que é a propriedade, e não o sítio.

    **`draw()` antes de ler**, senão o `matplotlib` devolve marcas ainda não formatadas.
    """
    com_detectadas = composicao_de_prova()
    sem_nenhuma = composicao_de_prova(
        classes=[
            classe_de_prova(n_zeros=78455, n_detectadas=0, log10_mediana=None),
            classe_de_prova(
                "NoHTLV",
                n_injecoes=16,
                n_celulas=96560,
                n_zeros=96560,
                n_detectadas=0,
                log10_mediana=None,
            ),
        ]
    )
    # A âncora vem antes, e é ela que torna o desfecho atribuível: a planta tem de aterrar no
    # predicado que decide o ramo, e não só nos números.
    assert not any(c.tem_abundancia for c in sem_nenhuma.classes), sem_nenhuma.classes
    assert all(c.tem_abundancia for c in com_detectadas.classes), com_detectadas.classes

    marcas: dict[str, list[list[str]]] = {}
    for nome, composicao in (("com", com_detectadas), ("sem", sem_nenhuma)):
        figura = composicao_desenhada(composicao)
        figura.canvas.draw()
        marcas[nome] = [
            [t.get_text() for t in eixo.get_xticklabels() if t.get_text()] for eixo in figura.axes
        ]

    esperadas = [marca_da_classe(c, rodado=False) for c in sem_nenhuma.classes]
    assert marcas["com"] != marcas["sem"], marcas
    # Com detectadas: as marcas estão em baixo e o painel de cima está limpo — a partilha do
    # `sharex` visível uma vez só.
    assert marcas["com"] == [[], [marca_da_classe(c, rodado=False) for c in com_detectadas.classes]]
    # Sem nenhuma: o painel de baixo está desligado, e as marcas TÊM de estar no de cima.
    assert marcas["sem"][0] == esperadas, (
        f"as marcas das classes saíram do painel visível: {marcas['sem']}. O de baixo está "
        "desligado e o de cima escondeu-as — os nomes das classes desapareceram dos dois"
    )

    figura = composicao_desenhada(sem_nenhuma)
    de_cima, de_baixo = figura.axes
    # `set_axis_off()` mexe em `axison` do eixo, e não na visibilidade do `XAxis` — ler o segundo
    # devolve `True` nos dois lados: o veredito falaria do instrumento e não do alvo.
    assert de_cima.axison and not de_baixo.axison, (de_cima.axison, de_baixo.axison)
    assert series(de_cima) and not series(de_baixo), "âncora: o painel de cima tem de ter barras"
    declarado = texto_corrido(" ".join(t.get_text() for t in de_baixo.texts))
    assert texto_corrido(frase_sem_abundancia(sem_nenhuma)) in declarado, declarado


def test_uma_classe_SEM_denominador_nao_colapsa_o_eixo_dos_faltantes() -> None:  # noqa: N802
    """`fracao_faltante` devolve `nan` por desenho, e `max` com `nan` é contagioso.

    **A ordem é o que torna a planta determinística**, e por isso a classe sem denominador vem
    **primeira**: `max([nan, x])` devolve `nan` e `max([x, nan])` devolve `x`. Com o `nan` no
    acumulador, `maior > 0.0` é `False`, o `ylim` cai no recuo de `(0, 1)` e a barra da outra classe
    — 16,5% — sai do eixo. A figura fica em branco sem que dado nenhum falte, que é exactamente o
    defeito da classe (c) do arnês.

    **O par causal separa as duas contas sobre o MESMO dado**, antes de ler a figura: a ingénua dá
    `nan` e a filtrada dá o valor da classe que tem denominador.
    """
    com_denominador = composicao_de_prova()
    sem_denominador = composicao_de_prova(
        classes=[
            classe_de_prova("SEM", n_injecoes=3, n_celulas=0, n_zeros=0),
            classe_de_prova(),
        ]
    )
    orfa, boa = sem_denominador.classes
    assert not math.isfinite(orfa.fracao_faltante), orfa
    assert math.isfinite(boa.fracao_faltante), boa

    alturas = [c.fracao_faltante * 100.0 for c in sem_denominador.classes]
    assert not math.isfinite(max(alturas)), (
        "âncora: a conta ingénua já devolve um número finito, e a planta não separa nada"
    )
    finitas = [a for a in alturas if math.isfinite(a)]
    assert max(finitas) == pytest.approx(12930 / 78455 * 100.0)

    figura = composicao_desenhada(sem_denominador)
    de_cima = figura.axes[0]
    assert confere_vista(de_cima) == [], confere_vista(de_cima)
    _, teto = de_cima.get_ylim()
    assert teto == pytest.approx(max(finitas) * (1.0 + FOLGA_DA_COMPOSICAO)), de_cima.get_ylim()

    # **A órfã não vira artista nenhum**, como a irmã da abundância já fazia — e a razão é medida:
    # uma `LineCollection` com `nan` devolve segmentos 1-D e `arnes_de_figura.series` rebenta com
    # `IndexError`, o que tornaria a figura inteira ilegível para o arnês por causa de uma classe.
    # **A asserção de PRESENÇA vem antes da de ausência**: num dicionário vazio, *"a barra
    # não está lá"* passaria sobre uma figura que não desenhou coisa nenhuma.
    desenhadas = series_da_composicao(figura)
    assert rotulo_da_classe("ATLL", PAPEL_DOS_FALTANTES) in desenhadas, sorted(desenhadas)
    assert rotulo_da_classe("SEM", PAPEL_DOS_FALTANTES) not in desenhadas, sorted(desenhadas)
    # E não é anotada: `em_percentagem(nan)` escrevia texto por cima de uma barra invisível.
    anotado = [texto_corrido(t.get_text()) for t in de_cima.texts]
    assert len(anotado) == 1, anotado
    assert anotado == [em_percentagem(boa.fracao_faltante)], anotado
    assert not any("nan" in t.lower() for t in anotado), anotado

    do_completo = composicao_desenhada(com_denominador).axes[0]
    assert len(do_completo.texts) == 2, [t.get_text() for t in do_completo.texts]


def test_o_predicado_de_largura_mede_a_MARCA_e_nao_so_o_nome_do_grupo() -> None:  # noqa: N802
    """A marca horizontal tem DUAS linhas, `nome\\n n=N`, e a primeira escrita media só a primeira.

    **A planta cai sobre dado em que as duas contas divergem**: oito classes de nome com
    **um** caractere, e a única coisa que muda entre os dois lados é o `n` de injeções. Medindo o
    nome, `maior` é `1` nos dois e o predicado responde `True` nos dois — o desfecho não depende da
    planta, e o teste não mediria nada. Medindo a marca, `n=1.000.000` são 11 caracteres e as oito
    marcas deixam de caber.

    **A geometria é a REAL** — `ALTURA_DA_COMPOSICAO` e o painel `211` —, senão o predicado seria
    medido sobre um eixo que esta figura não tem.
    """
    curto = composicao_de_prova(
        classes=[classe_de_prova(letra, n_injecoes=13) for letra in "ABCDEFGH"]
    )
    longo = composicao_de_prova(
        classes=[classe_de_prova(letra, n_injecoes=1_000_000) for letra in "ABCDEFGH"]
    )
    # A âncora de que a planta separa: os NOMES são idênticos, então a conta antiga é cega a ela.
    dos_nomes = {
        nome: max(len(c.grupo) for c in comp.classes)
        for nome, comp in (("curto", curto), ("longo", longo))
    }
    assert dos_nomes["curto"] == dos_nomes["longo"] == 1, dos_nomes
    das_marcas = {
        nome: max(
            len(linha)
            for c in comp.classes
            for linha in marca_da_classe(c, rodado=False).split("\n")
        )
        for nome, comp in (("curto", curto), ("longo", longo))
    }
    assert das_marcas["curto"] < das_marcas["longo"], das_marcas

    figura = nova_figura(altura=ALTURA_DA_COMPOSICAO)
    eixo = figura.add_subplot(211)
    assert rotulos_das_classes_cabem(eixo, curto) is True
    assert rotulos_das_classes_cabem(eixo, longo) is False, (
        f"o predicado mediu {dos_nomes['longo']} caractere e a marca desenhada mede "
        f"{das_marcas['longo']}: as oito marcas sobrepõem-se e nenhuma classe do arnês vê a colisão"
    )
    assert marcas_das_classes(figura, eixo, curto) is MarcasDasClasses.HORIZONTAIS
    assert marcas_das_classes(figura, eixo, longo) is not MarcasDasClasses.HORIZONTAIS


def test_com_ZERO_classes_a_figura_DECLARA_num_eixo_so_e_nao_levanta() -> None:  # noqa: N802
    """R7: *"insumo ausente por escolha do pesquisador é rotulado como achado válido"*.

    **É a diferença desta figura para o violino, e ela é deliberada:** ali, os dois lados vazios são
    `MatrizVaziaError` porque um eixo com duas frases e nenhum artista se lê como figura partida;
    aqui, *"esta matriz não tem injeção de estudo nenhuma"* é fato sobre o desenho do estudo.

    **UM eixo e não dois**, e isso veio da imagem montada: com dois, a mesma frase saía duas vezes.
    """
    vazia = composicao_de_prova(classes=[], n_injecoes_fora=68)
    assert not vazia.desenhavel and vazia.n_injecoes == 0, vazia

    figura = composicao_desenhada(vazia)
    assert len(figura.axes) == 1, [e.get_title() for e in figura.axes]
    assert not series(figura.axes[0]), series(figura.axes[0])
    declarado = texto_corrido(" ".join(t.get_text() for t in figura.axes[0].texts))
    assert "nenhuma injeção de estudo" in declarado, declarado
    assert texto_corrido(SEM_CLASSE_DECLARADA) in declarado, declarado

    assert len(composicao_desenhada().axes) == 2, (
        "âncora: a figura com classes também tem um eixo só, e «um e não dois» não mede nada"
    )


def test_com_ZERO_classes_a_figura_NAO_emite_o_aviso_do_tight_layout(  # noqa: N802
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A irmã do item 84 que a revisão de 25/09/2026 achou pela classe, e o par causal.

    O texto de `_declara` nasce numa linha só e o primeiro `tight_layout` de `recorte.aplica` — o
    que corre antes do requebrar — desistia com *«Tight layout not applied»*. Tirá-lo do layout
    fecha o aviso; a planta repõe-no dentro, só nos textos que `_declara` pôs, e o aviso tem de
    voltar — senão esta asserção de ausência não mede nada.
    """
    import warnings

    from refino.plots import composicao as modulo

    def avisos() -> list[str]:
        with warnings.catch_warnings(record=True) as achados:
            warnings.simplefilter("always")
            vazia = composicao_desenhada(composicao_de_prova(classes=[], n_injecoes_fora=68))
            vazia.canvas.draw()
        return [str(a.message) for a in achados if "Tight layout" in str(a.message)]

    assert avisos() == [], "com zero classes a figura ainda avisa que o tight_layout desistiu"

    original = modulo._declara
    repostos: list[object] = []

    def com_o_texto_no_layout(eixo: Any, frase: str) -> None:
        original(eixo, frase)
        for texto in eixo.texts:
            texto.set_in_layout(True)
            repostos.append(texto)

    monkeypatch.setattr(modulo, "_declara", com_o_texto_no_layout)
    assert avisos(), "com o texto de volta ao layout o aviso não voltou"
    assert repostos, "âncora: a planta não aterrou — nenhum texto foi reposto no layout"


def test_a_populacao_e_as_tres_contagens_de_ausencia_vao_no_RODAPE() -> None:  # noqa: N802
    """R7 — e as três contagens são o que torna o rótulo *«faltante»* **conferível** na imagem.

    Nos quatro arquivos deste laboratório as três candidatas coincidem por acidente; servir só a
    união faria a palavra significar outra coisa no primeiro dado com uma célula não finita, sem
    nada mudar de aparência.

    **O par do topo é de FEATURES, e a primeira escrita pôs ali INJEÇÕES** — `Recorte.texto()`
    escreve a palavra `features` literalmente, e a imagem dizia *"61 de 68 features"* sobre uma
    matriz de 6.035. Aqui o teste deriva o par do corpo e afirma a igualdade.
    """
    composicao = composicao_de_prova()
    rodape = texto_corrido(recorte_da_composicao_por_classe(composicao=composicao).texto())

    esperado = milhar_do_teste(N_FEATURES_DA_COMPOSICAO)
    assert rodape.startswith(f"{esperado} de {esperado} features"), rodape[:80]
    assert "é de INJEÇÕES" not in rodape, rodape[:200]

    assert f"{milhar_do_teste(13 + 16)} injeções em 2 classe(s)" in rodape, rodape
    assert f"{milhar_do_teste(N_INJECOES_FORA)} fora dela" in rodape, rodape
    assert "«sample»" in rodape and "«ingerida»" in rodape, rodape

    zeros = milhar_do_teste(12930 + 15568)
    assert f"{zeros} células, {zeros} por valor zero e 0 por não finito" in rodape, rodape


def test_as_TRES_contagens_de_ausencia_no_rodape_SEPARAM_quando_divergem() -> None:  # noqa: N802
    """**A planta cai sobre dado em que as três coisas diferem**, que a fixtura limpa não dá.

    Com `n_nao_finitas == 0` as três candidatas dão o mesmo número, e uma figura que somasse só os
    zeros passaria. Com uma célula não finita plantada, o rodapé tem de mostrar as três — e é isso
    que torna o rótulo conferível por quem lê.
    """
    limpa = composicao_de_prova()
    assert all(c.n_nao_finitas == 0 for c in limpa.classes), (
        "âncora: a fixtura já tem célula não finita, e a planta não seria atribuível"
    )
    do_limpo = texto_corrido(recorte_da_composicao_por_classe(composicao=limpa).texto())
    assert "e 0 por não finito" in do_limpo, do_limpo

    suja = composicao_de_prova(
        classes=[
            classe_de_prova(n_zeros=12000, n_nao_finitas=930),
            classe_de_prova(
                "NoHTLV", n_injecoes=16, n_celulas=96560, n_zeros=15568, log10_mediana=4.0791
            ),
        ]
    )
    do_sujo = texto_corrido(recorte_da_composicao_por_classe(composicao=suja).texto())
    assert do_limpo != do_sujo
    total = milhar_do_teste(12930 + 15568)
    assert f"{total} células, {milhar_do_teste(27568)} por valor zero e 930 por não finito" in (
        do_sujo
    ), do_sujo


def test_os_SEIS_rotulos_ausentes_do_corpo_nao_viram_nome_de_coisa_nenhuma() -> None:  # noqa: N802
    """Na ponta do desenho, campo que não veio não pode sair como um valor do vocabulário.

    `sample` é população legítima e `zero_ou_nao_finita` é definição legítima: um recuo escolhido
    entre os plausíveis faria a imagem afirmar procedência que ninguém declarou.

    **O par causal separa**: com os campos o rodapé nomeia-os; sem eles nomeia a ausência.
    """
    com = texto_corrido(recorte_da_composicao_por_classe(composicao=composicao_de_prova()).texto())
    sem = texto_corrido(
        recorte_da_composicao_por_classe(
            composicao=composicao_de_prova(
                matriz=None,
                populacao=None,
                definicao_de_faltante=None,
                definicao_de_abundancia=None,
            )
        ).texto()
    )
    assert com != sem
    assert NAO_DECLARADA in sem and NAO_DECLARADA not in com, sem
    assert POPULACAO_NAO_DECLARADA in sem, sem
    assert DEFINICAO_AUSENTE in sem, sem
    # **As âncoras de não-vacuidade vêm ANTES das três asserções de ausência, e na mesma função**:
    # um vocabulário vazio satisfaz *"o recuo não é um valor dele"* sobre coisa nenhuma.
    vocabularios = (
        get_args(SampleType),
        get_args(DefinicaoDeFaltante),
        get_args(DefinicaoDeAbundancia),
    )
    assert all(vocabularios), vocabularios
    assert POPULACAO_NAO_DECLARADA not in vocabularios[0], POPULACAO_NAO_DECLARADA
    assert DEFINICAO_AUSENTE not in vocabularios[1], DEFINICAO_AUSENTE
    assert DEFINICAO_AUSENTE not in vocabularios[2], DEFINICAO_AUSENTE


def test_uma_classe_que_nao_e_objeto_SAI_da_lista_e_nao_quebra_a_figura() -> None:  # noqa: N802
    """A guarda de fronteira de `classes_do_fio`, e ela tem as duas direções.

    Nada do que vem do fio passou pelo mypy: a lista pode não ser lista, e um elemento pode não ser
    um objeto. **O que não pode acontecer é a figura desenhar sobre um objeto que ela inventou.**
    """
    assert len(classes_do_fio([classe_de_prova(), "lixo", None, 7])) == 1
    assert classes_do_fio(None) == () and classes_do_fio("nao e lista") == ()

    orfa = classes_do_fio([{"grupo": "G1"}])[0]
    assert orfa.grupo == "G1"
    assert (orfa.n_injecoes, orfa.n_celulas, orfa.n_detectadas) == (0, 0, 0), orfa
    assert not math.isfinite(orfa.log10_mediana), orfa
    assert not orfa.tem_abundancia and not math.isfinite(orfa.fracao_faltante), orfa

    sem_nome = classes_do_fio([{"n_injecoes": 3}])[0]
    assert sem_nome.grupo == NAO_DECLARADA, (
        f"um grupo sem nome saiu como {sem_nome.grupo!r}: a cadeia «None» como rótulo de uma barra "
        "é procedência inventada num eixo"
    )


def test_os_TRES_desfechos_de_marcacao_das_classes_existem_e_DIFEREM() -> None:  # noqa: N802
    """**N desfechos, N direções**, e os dois últimos vieram da imagem montada.

    A largura do eixo e o comprimento dos nomes decidem, e os dois instrumentos já existiam. O que
    esta asserção afirma é que os três desfechos são **alcançáveis** e produzem imagens diferentes —
    um predicado que respondesse sempre a mesma coisa passaria em qualquer teste de aparência.
    """
    # **As três populações diferem em DUAS grandezas — quantas classes e quanto mede o nome —, e
    # as duas decidem.** Com duas classes só, as marcas horizontais cabem até cerca de 43
    # caracteres, e rodadas deixam de caber por volta dos 39: não existe comprimento que produza
    # `RODADAS`. A contagem de classes é o que abre a janela entre os dois limiares.
    curtos = composicao_de_prova()
    medios = composicao_de_prova(
        classes=[classe_de_prova("Classe_clinica_" + letra * 6) for letra in "ABCDEF"]
    )
    longos = composicao_de_prova(
        classes=[classe_de_prova("Classe_clinica_" + letra * 30) for letra in "ABCDEF"]
    )

    vistos = []
    for composicao in (curtos, medios, longos):
        # A geometria é a REAL — a altura da figura e a posição do painel de cima —, senão o
        # predicado seria medido sobre um eixo que esta figura não tem.
        figura = nova_figura(altura=ALTURA_DA_COMPOSICAO)
        eixo = figura.add_subplot(211)
        vistos.append(marcas_das_classes(figura, eixo, composicao))

    assert vistos == [
        MarcasDasClasses.HORIZONTAIS,
        MarcasDasClasses.RODADAS,
        MarcasDasClasses.AUSENTES,
    ], vistos

    # E o desfecho `AUSENTES` é o ÚNICO em que a legenda existe: com marcas, a legenda escreveria
    # cada nome uma segunda vez — e foi ela que pediu a largura que a disposição anterior não tinha.
    assert composicao_desenhada(curtos).axes[0].get_legend() is None
    com_legenda = composicao_desenhada(longos).axes[0]
    assert com_legenda.get_legend() is not None, (
        "sem marcas e sem legenda, as barras ficam anónimas — e nada na imagem diz qual é qual"
    )
    assert texto_corrido(SEM_MARCAS_DE_CLASSE) in texto_corrido(
        composicao_desenhada(longos).axes[1].get_xlabel()
    )


@pytest.mark.parametrize(
    ("largura", "altura", "ppp"),
    [(10.0, 6.0, 100.0), (8.0, 4.5, 96.0), (6.5, 3.4, 96.0), (12.0, 7.0, 150.0)],
)
def test_o_rodape_da_composicao_cabe_nos_tamanhos_EXTREMOS_sem_levantar(  # noqa: N802
    largura: float, altura: float, ppp: float
) -> None:
    """Os quatro tamanhos, com o rodapé que esta figura de facto produz."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure as FiguraNua

    from refino.plots.recorte import ENTRELINHA, MARGEM, RESERVA_MAXIMA, TAMANHO_DA_FONTE

    texto = recorte_da_composicao_por_classe(composicao=composicao_de_prova()).texto()
    figura = FiguraNua(figsize=(largura, altura), dpi=ppp)
    FigureCanvasAgg(figura)
    linhas = linhas_do_rodape(figura, texto)
    reservado = len(linhas) * TAMANHO_DA_FONTE * ENTRELINHA / (altura * 72.0) + 2.0 * MARGEM
    assert reservado <= RESERVA_MAXIMA, (
        f"a {largura}x{altura}@{ppp} o rodapé quebra em {len(linhas)} linhas e precisaria de "
        f"{reservado:.1%} da altura, acima do teto de {RESERVA_MAXIMA:.0%}"
    )


def test_um_denominador_ausente_na_composicao_RECUSA_por_populacao_vazia() -> None:  # noqa: N802
    """Sem execução não há população, e `MatrizVaziaError` é o que o painel apanha.

    **É outro fato que «a rota calculou e não há classe clínica»**: aquele tem `n_features` e
    é declarado em texto; este é *"nenhuma execução trouxe composição"* e não tem população nenhuma
    de onde recortar.
    """
    with pytest.raises(MatrizVaziaError, match="não declara de quantas features"):
        recorte_da_composicao_por_classe(composicao=composicao_de_prova(n_features=0))


def test_a_composicao_e_a_UNICA_dona_da_chave_item_11() -> None:  # noqa: N802
    """A figura D entra como `Vinculo.RELATORIO` com a chave do item que a emenda criou.

    **Contar e não perguntar se existe**: `"item-11" in {...}` devolve o desfecho de sucesso
    com duas entradas iguais, e duas fontes para o mesmo desenho é o que se recusa.
    """
    donos = [c for c in CONSTRUTORES if c.chave == "item-11"]
    assert len(donos) == 1, [c.funcao for c in donos]
    assert donos[0].modulo == "composicao"
    assert donos[0].funcao == "figura_da_composicao_por_classe"
    assert donos[0].vinculo is Vinculo.RELATORIO


# ---------------------------------------------------------------------------
# A coorte que a janela lê, e a cor de cada grupo nas barras
# ---------------------------------------------------------------------------


def test_os_grupos_da_composicao_sao_a_COORTE_ordenada_e_sem_nome_que_nao_chegou() -> None:  # noqa: N802
    """A lista de classes vira a coorte, na ordem canônica — e o que não é nome não vira grupo.

    **As três formas de ausência, cada uma com o desfecho dela:** o grupo em branco e o grupo que
    não veio saem — `classes_do_fio` troca-os por um recuo de rótulo de barra, que não é nome de
    grupo —, e o elemento que não é objeto sai antes. Uma tabela com uma linha *«matriz não
    declarada»* pediria cor para um grupo que ninguém declarou.
    """
    sem_nome = classe_de_prova()
    del sem_nome["grupo"]
    classes = [
        classe_de_prova("NoHTLV"),
        classe_de_prova("ATLL"),
        classe_de_prova("   "),
        sem_nome,
        "não é objeto",
    ]
    assert grupos_da_composicao(classes) == ("ATLL", "NoHTLV")
    assert grupos_da_composicao(None) == ()
    assert grupos_da_composicao([]) == ()
    assert NAO_DECLARADA not in grupos_da_composicao(classes)


def test_com_ESTILO_as_barras_dos_DOIS_paineis_tem_a_cor_do_grupo() -> None:  # noqa: N802
    """Cada classe tem uma barra em cada painel, e as duas levam a cor do grupo escolhida."""
    composicao = composicao_de_prova()
    estilos = estilos_dos_grupos(
        ["ATLL", "NoHTLV"], {"NoHTLV": EstiloDoGrupo(cor="vermelho", simbolo="losango")}
    )
    assert estilos is not None
    figura = figura_da_composicao_por_classe(
        composicao=composicao,
        recorte=recorte_da_composicao_por_classe(composicao=composicao),
        estilos=estilos,
    )
    vistas: dict[str, set[str]] = {}
    for eixo in figura.axes:
        for colecao in eixo.collections:
            grupo = str(colecao.get_label()).split(" — ")[0]
            vistas.setdefault(grupo, set()).add(to_hex(colecao.get_colors()[0]))
    assert vistas == {
        "ATLL": {to_hex(estilos.por_grupo["ATLL"].hex)},
        "NoHTLV": {to_hex("#D55E00")},
    }, vistas
    assert sum(len(eixo.collections) for eixo in figura.axes) == 4, (
        "âncora: duas barras por classe nos dois painéis — com menos, a cor de uma não foi lida"
    )

    item = next(i for i in ITENS_OBRIGATORIOS if i.numero == 11)
    assert "Composição por classe" in item.titulo, item.titulo
    assert "células faltantes" in item.titulo and "abundância típica" in item.titulo, item.titulo
