"""O dendrograma das amostras e o mapa de calor, a PRIMEIRA figura de matriz-imagem.

## A planta que só separa por causa, e ela é a razão de este arquivo existir

*"Mapa de calor com a matriz TRANSPOSTA produz figura diferente"* parece uma planta trivial, e até
que existisse o auxiliar de conteúdo ela **não separava nada**: `arnes_de_figura.series` não
percorria `ax.images`, então `confere_series` devolvia `[]` para qualquer conteúdo — asserção de
ausência satisfeita por vacuidade sobre um mapa de calor cheio. `confere_conteudo` é o que foi
acrescentado, e o par causal abaixo afirma **as duas metades**: que a classe que lê coordenadas
continua cega à transposição de uma matriz quadrada, e que a que lê conteúdo a acusa.

**A âncora vive dentro do teste, e sobre uma matriz QUADRADA de propósito.** Só com
`n_amostras == n_features` o retângulo do `imshow` fica idêntico nas duas orientações; com formas
diferentes a coordenada já separaria, e o teste passaria a provar outra coisa.

## O que este arquivo NÃO afirma

**Nada sobre legibilidade.** As classes leem o objeto `Figure`; nenhuma delas sabe se
trinta rótulos de feature cabem na altura do eixo. Isso foi medido a olho na imagem montada e
está no relatório da unidade — não aqui, porque a resposta depende do tamanho do widget,
que o construtor não conhece.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, get_args

import numpy as np
import pytest
from matplotlib.colors import to_hex
from matplotlib.markers import MarkerStyle
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import pdist

from refino.domain.labels import ClusterLinkage, ClusterMetric
from refino.plots.agrupamento import (
    NOME_DA_LIGACAO,
    NOME_DA_METRICA,
    ROTULO_DA_ARVORE,
    ROTULO_DO_MAPA,
    TETO_DE_FOLHAS_PARA_DESENHAR,
    TITULO_DO_KMEANS,
    TITULO_DO_SOM,
    Arvore,
    ArvoreInvalidaError,
    Grade,
    MapaDesalinhadoError,
    MetodoDesconhecidoError,
    Particao,
    ParticaoDesalinhadaError,
    arvore_do_fio,
    figura_do_dendrograma,
    figura_do_kmeans,
    figura_do_mapa_auto_organizavel,
    figura_do_mapa_de_calor,
    frase_da_grade,
    frase_da_semente,
    frase_do_metodo,
    frase_do_prefixo,
    kmeans_do_fio,
    mapa_do_fio,
    posicoes_na_grade,
    recorte_do_dendrograma,
    recorte_do_kmeans,
    recorte_do_mapa_auto_organizavel,
    recorte_do_mapa_de_calor,
    segmentos_do_dendrograma,
    sem_prefixo_comum,
    som_do_fio,
)
from refino.plots.base import (
    PALETA_DIVERGENTE,
    FeaturesDemaisError,
    MatrizVaziaError,
    RecusaDoConstrutorError,
)
from refino.plots.series import (
    SIMBOLOS_DOS_GRUPOS,
    EstiloDoGrupo,
    EstilosDosGrupos,
    cores_dos_grupos,
    estilos_dos_grupos,
)
from refino.statistics.clustering import (
    INICIALIZACOES_DO_KMEANS,
    TETO_DE_FEATURES_PARA_DESENHAR,
)
from tests.arnes_de_figura import (
    confere_conteudo,
    confere_rotulos,
    confere_series,
    confere_vista,
    numeros_declarados,
    planta_conteudo_trocado,
    planta_corte_de_eixo,
    planta_eixos_trocados,
    planta_serie_removida,
    series,
)

if TYPE_CHECKING:  # pragma: no cover - só para o mypy
    from matplotlib.figure import Figure

# A população do `pos_hilic`, literal aqui de propósito: é a fonte independente da classe (d).
POPULACAO: Final = 6035

MATRIZ: Final = "a matriz corrigida por SERRF"
AMOSTRAGEM: Final = "as 6 de maior variância, escolha determinística"


# ---------------------------------------------------------------------------
# Fixtures de dado — dois grupos separados POR CONSTRUÇÃO
# ---------------------------------------------------------------------------


def matriz_de_dois_grupos(n_amostras: int = 8, n_features: int = 6) -> np.ndarray:
    """z-score de uma matriz em que metade das amostras está deslocada. Separação por construção.

    **Semente fixa e `default_rng`**, e o ruído é pequeno em relação ao deslocamento: o que o teste
    afirma é a TOPOLOGIA, e uma separação que dependesse de sorte tornaria a asserção intermitente —
    que é pior que reprovar sempre.
    """
    rng = np.random.default_rng(20260914)
    valores = rng.normal(scale=0.25, size=(n_amostras, n_features))
    valores[: n_amostras // 2] += 6.0
    return (valores - valores.mean(axis=0)) / valores.std(axis=0)


def arvore_de(valores: np.ndarray, rotulos: list[str] | None = None) -> Arvore:
    ligacao = linkage(pdist(valores), method="ward")
    nomes = rotulos or [f"ESTUDO_POS_{i:02d}" for i in range(valores.shape[0])]
    return arvore_do_fio(
        ligacao=ligacao.tolist(), rotulos=nomes, metrica="euclidean", ligacao_metodo="ward"
    )


def mapa_de(valores: np.ndarray, **extra: Any) -> Any:
    """Um `MapaDeCalor` com nomes plausíveis e as duas ordens do agrupamento.

    **As ordens só são CALCULADAS quando `extra` não as traz**, e isso não é economia: a ligação de
    uma matriz com `nan` não existe — o `scipy` recusa a distância —, e um dicionário literal
    avaliaria a chamada antes de o `update` a substituir. O cenário da célula indefinida reprovaria
    com uma excepção do `scipy`, que é medição que não aconteceu e não reprovação.
    """
    n_a, n_f = valores.shape
    argumentos: dict[str, Any] = {
        "valores": valores.tolist(),
        "amostras": [f"ESTUDO_POS_{i:02d}" for i in range(n_a)],
        "features": [f"{2.0 + i / 10:.2f}_{810.0 + i:.4f}m/z" for i in range(n_f)],
    }
    if "ordem_das_amostras" not in extra:
        argumentos["ordem_das_amostras"] = [
            int(i) for i in leaves_list(linkage(pdist(valores), method="ward"))
        ]
    if "ordem_das_features" not in extra:
        argumentos["ordem_das_features"] = [
            int(i) for i in leaves_list(linkage(pdist(valores.T), method="ward"))
        ]
    argumentos.update(extra)
    return mapa_do_fio(**argumentos)


def recorte_de_mapa(mapa: Any) -> Any:
    return recorte_do_mapa_de_calor(
        mapa=mapa,
        metrica="euclidean",
        ligacao_metodo="ward",
        n_features_totais=POPULACAO,
        matriz=MATRIZ,
        amostragem=AMOSTRAGEM,
    )


def recorte_de_arvore(arvore: Arvore, n_usadas: int = 6) -> Any:
    return recorte_do_dendrograma(
        arvore=arvore,
        n_features_usadas=n_usadas,
        n_features_totais=POPULACAO,
        matriz=MATRIZ,
        amostragem=AMOSTRAGEM,
    )


def rodape_de(figura: Figure) -> str:
    """O texto do rodapé com o espaço em branco COLAPSADO. **Não é conveniência.**

    `recorte.aplica` quebra o rodapé em linhas pela largura real da figura, e a quebra cai onde
    calhar. Comparar com a quebra dentro fixa a QUEBRA como contrato quando o contrato é o TEXTO —
    e o veredito sai *"a figura não declara isto"* sobre uma frase que está lá inteira. **A direção
    é a perigosa:** agir sobre ela apaga declaração correcta.

    **Aconteceu neste arquivo, na primeira execução:** *"nenhuma célula é indefinida"* e
    *"as 6 de maior variância, escolha determinística"* atravessavam a quebra, e as duas asserções
    reprovaram sobre um rodapé certo. É o mesmo gesto de `arnes_de_figura.texto_corrido`.
    """
    return " ".join(" ".join(t.get_text() for t in figura.texts).split())


# ---------------------------------------------------------------------------
# 1. O dendrograma — topologia, e a figura limpa nas quatro classes
# ---------------------------------------------------------------------------


def test_dois_grupos_separados_por_construcao_caem_nos_dois_RAMOS(  # noqa: N802
) -> None:
    """A asserção é sobre a TOPOLOGIA e não sobre coordenadas — e a âncora vem primeiro.

    **A âncora é que os dois grupos existem no dado**: sem ela, uma matriz sem estrutura nenhuma
    produziria uma árvore qualquer e a asserção seguinte não teria o que afirmar. Ela é medida do
    dado — a distância entre grupos contra a distância dentro do grupo —, não do desenho.

    **A afirmação é sobre a última JUNÇÃO da ligação, que é a raiz da árvore**: os dois filhos dela
    têm de ser exactamente os dois grupos de quatro. Afirmar sobre a ordem das folhas seria mais
    fraco — a ordem também sairia certa numa árvore em cadeia, que não tem dois ramos.
    """
    valores = matriz_de_dois_grupos(8, 6)
    distancias = pdist(valores)
    quadrada = np.zeros((8, 8))
    indices = np.triu_indices(8, k=1)
    quadrada[indices] = distancias
    quadrada += quadrada.T
    dentro = float(quadrada[:4, :4][np.triu_indices(4, k=1)].mean())
    entre = float(quadrada[:4, 4:].mean())
    assert entre > 3.0 * dentro, (
        f"âncora: os dois grupos não estão separados no dado — distância média entre grupos "
        f"{entre:.3f} contra {dentro:.3f} dentro. A asserção de topologia abaixo não teria o que "
        "afirmar"
    )

    arvore = arvore_de(valores)
    raiz = arvore.ligacao[-1]
    assert int(raiz[3]) == 8, f"a última junção não é a raiz: cobre {int(raiz[3])} folhas"

    # Os dois filhos da raiz são nós internos (índice >= n_folhas); as folhas de cada um são um dos
    # dois grupos. Percorre-se a ligação em vez de chamar `fcluster`, para a asserção não depender
    # de uma segunda API do `scipy`.
    def folhas(no: int) -> set[int]:
        if no < 8:
            return {no}
        esquerda, direita = arvore.ligacao[no - 8][:2]
        return folhas(int(esquerda)) | folhas(int(direita))

    ramos = [folhas(int(raiz[0])), folhas(int(raiz[1]))]
    assert sorted(ramos, key=min) == [{0, 1, 2, 3}, {4, 5, 6, 7}], ramos


def test_a_figura_limpa_do_dendrograma_passa_nas_quatro_classes() -> None:
    """O caminho feliz, e a outra ponta de cada par causal deste arquivo."""
    valores = matriz_de_dois_grupos(8, 6)
    arvore = arvore_de(valores)
    grupos = ["controle"] * 4 + ["tratado"] * 4
    figura = figura_do_dendrograma(arvore=arvore, grupos=grupos, recorte=recorte_de_arvore(arvore))
    eixo = figura.axes[0]

    desenhadas = {s.rotulo for s in series(eixo)}
    assert desenhadas == {
        ROTULO_DA_ARVORE,
        "controle — 4 amostras",
        "tratado — 4 amostras",
    }, desenhadas
    assert confere_vista(eixo) == []
    assert (
        confere_rotulos(
            eixo,
            x="amostra",
            y=f"altura da junção — {NOME_DA_METRICA['euclidean']}",
            titulo="Agrupamento hierárquico das amostras",
            legenda=(ROTULO_DA_ARVORE, "controle — 4 amostras", "tratado — 4 amostras"),
        )
        == []
    )

    # Classe (d): os números declarados, derivados aqui e não pedidos ao construtor.
    declarados = numeros_declarados(figura)
    assert "6.035" in declarados and "6" in declarados and "8" in declarados, declarados


def test_a_cor_das_folhas_vem_do_AUXILIAR_da_U18_e_nunca_de_literal() -> None:  # noqa: N802
    """A mesma coorte, a mesma paleta — e a fonte é `series.cores_dos_grupos`.

    **É o par causal do requisito:** a cor desenhada é comparada com a que o auxiliar devolve, e
    não com um literal. Um construtor que escrevesse cores próprias passaria numa comparação com
    literal e falharia aqui, que é o desfecho certo.
    """
    valores = matriz_de_dois_grupos(8, 6)
    arvore = arvore_de(valores)
    grupos = ["controle"] * 4 + ["tratado"] * 4
    figura = figura_do_dendrograma(arvore=arvore, grupos=grupos, recorte=recorte_de_arvore(arvore))
    esperadas = cores_dos_grupos(grupos)
    assert len(esperadas) == 2, "âncora: o auxiliar devolveu menos de duas cores"

    import matplotlib.colors as mcolors

    for colecao in figura.axes[0].collections:
        rotulo = str(colecao.get_label())
        if rotulo == ROTULO_DA_ARVORE:
            continue
        grupo = rotulo.split(" — ")[0]
        desenhada = colecao.get_facecolor()[0]
        assert np.allclose(desenhada, mcolors.to_rgba(esperadas[grupo])), (
            f"a série {grupo!r} foi desenhada com {desenhada} e o auxiliar da U18 dá "
            f"{esperadas[grupo]!r}"
        )


def test_o_dendrograma_SEM_grupos_desenha_uma_serie_com_a_contagem() -> None:  # noqa: N802
    """O outro ramo: sem grupo clínico, uma dispersão só — e ainda DUAS séries nos eixos.

    **Duas e não uma é requisito do negativo, não estética:** `planta_serie_removida` recusa-se a
    correr sobre uma figura de uma série só, com `PlantaQueNaoSeparaError`. Um construtor que não
    desenhasse as folhas deixaria o par causal desta figura sem poder ser plantado.
    """
    valores = matriz_de_dois_grupos(8, 6)
    arvore = arvore_de(valores)
    figura = figura_do_dendrograma(arvore=arvore, recorte=recorte_de_arvore(arvore))
    desenhadas = {s.rotulo for s in series(figura.axes[0])}
    assert desenhadas == {ROTULO_DA_ARVORE, "8 amostras"}, desenhadas


def test_par_causal_serie_removida_do_dendrograma() -> None:
    """Par causal: com a planta acusa, sem a planta não acusa. **A planta é ARGUMENTO.**"""
    valores = matriz_de_dois_grupos(8, 6)
    arvore = arvore_de(valores)
    grupos = ["controle"] * 4 + ["tratado"] * 4
    esperadas = {
        s.rotulo: (s.x, s.y)
        for s in series(
            figura_do_dendrograma(
                arvore=arvore, grupos=grupos, recorte=recorte_de_arvore(arvore)
            ).axes[0]
        )
    }
    mutilada = figura_do_dendrograma(
        arvore=arvore, grupos=grupos, recorte=recorte_de_arvore(arvore)
    )
    eixo = mutilada.axes[0]
    assert confere_series(eixo, esperadas) == [], "sem a planta o portão já acusava"
    apagada = planta_serie_removida(eixo)
    achados = confere_series(eixo, esperadas)
    assert achados, f"a planta removeu {apagada!r} e o portão não acusou"


def test_o_dendrograma_de_UMA_folha_nao_existe_e_a_recusa_diz_isso() -> None:  # noqa: N802
    """Caso de borda: uma observação não tem par, não tem distância e não tem árvore."""
    with pytest.raises(ArvoreInvalidaError, match="formato do `scipy`"):
        arvore_do_fio(ligacao=[], rotulos=["a"], metrica="euclidean", ligacao_metodo="ward")


def test_a_posicao_das_folhas_e_a_CONVENCAO_do_scipy_e_nao_um_palpite() -> None:  # noqa: N802
    """O par `icoord` do `scipy` e as posições que o construtor deriva têm de coincidir.

    **É a âncora do alvo certo para a constante `_PASSO_DA_FOLHA`:** o construtor não lê `icoord`
    para pôr os pontos das folhas — deriva as posições de `10·i + 5`. Se a convenção do `scipy`
    mudar, isto acusa, em vez de a nuvem sair deslocada em silêncio numa figura bem formada.
    """
    valores = matriz_de_dois_grupos(8, 6)
    arvore = arvore_de(valores)
    segmentos, ordem = segmentos_do_dendrograma(arvore)
    assert segmentos.shape == (7, 4, 2), segmentos.shape
    assert sorted(ordem) == list(range(8)), ordem

    derivadas = np.arange(8, dtype=float) * 10.0 + 5.0
    # As pontas de cada segmento que tocam `y == 0` são folhas; o conjunto dos `x` delas tem de ser
    # exactamente o conjunto derivado.
    tocam_o_chao = segmentos[:, :, 1] == 0.0
    xs = np.unique(np.round(segmentos[:, :, 0][tocam_o_chao], 6))
    assert set(xs.tolist()) <= set(derivadas.tolist()), (
        f"o `scipy` pôs folhas em {xs.tolist()} e a derivação dá {derivadas.tolist()}"
    )


# ---------------------------------------------------------------------------
# 2. O mapa de calor — e a planta que só separa por causa
# ---------------------------------------------------------------------------


def test_a_figura_limpa_do_mapa_passa_nas_quatro_classes() -> None:
    valores = matriz_de_dois_grupos(8, 6)
    mapa = mapa_de(valores, grupos=["controle"] * 4 + ["tratado"] * 4)
    figura = figura_do_mapa_de_calor(mapa=mapa, recorte=recorte_de_mapa(mapa))
    eixo = figura.axes[0]

    desenhadas = {s.rotulo for s in series(eixo)}
    assert desenhadas == {ROTULO_DO_MAPA}, desenhadas
    assert confere_vista(eixo) == []
    assert (
        confere_rotulos(
            eixo, x="amostra", y="feature", titulo="Mapa de calor das features agrupadas"
        )
        == []
    )

    # O conteúdo desenhado é derivado AQUI, por caminho que não passa pelo construtor.
    esperado = valores[np.ix_(mapa.ordem_das_amostras, mapa.ordem_das_features)].T
    assert confere_conteudo(eixo, {ROTULO_DO_MAPA: esperado}) == []


def test_a_planta_da_TRANSPOSTA_separa_por_causa_da_U18_e_nao_antes() -> None:  # noqa: N802
    """**A planta obrigatória desta unidade, e as DUAS metades vão na mesma função.**

    **Âncora, antes de plantar:** a matriz tem de ser quadrada — senão as coordenadas já
    separariam — e tem de diferir da própria transposta — senão a planta é a identidade.

    A metade de cima afirma que `confere_series`, que lê coordenadas, **continua cega**: o retângulo
    do `imshow` é o mesmo nas duas orientações. A metade de baixo afirma que `confere_conteudo`, que
    acrescentada, acusa. Sem a primeira, ninguém sabe que a segunda era necessária.
    """
    valores = matriz_de_dois_grupos(6, 6)
    assert valores.shape[0] == valores.shape[1], "âncora: a matriz tem de ser quadrada"
    assert not np.allclose(valores, valores.T), (
        "âncora: a matriz é igual à própria transposta e a planta seria a identidade"
    )

    mapa = mapa_de(valores)
    figura = figura_do_mapa_de_calor(mapa=mapa, recorte=recorte_de_mapa(mapa))
    eixo = figura.axes[0]
    desenhado = valores[np.ix_(mapa.ordem_das_amostras, mapa.ordem_das_features)].T
    transposto = desenhado.T
    assert not np.allclose(desenhado, transposto), (
        "âncora: o conteúdo desenhado é simétrico e transpô-lo não muda nada"
    )

    # Metade 1 — a classe que lê COORDENADAS não separa. É o estado sem a segunda classe.
    coordenadas = {s.rotulo: (s.x, s.y) for s in series(eixo)}
    assert confere_series(eixo, coordenadas) == []
    assert confere_conteudo(eixo, {ROTULO_DO_MAPA: desenhado}) == []

    # Metade 2 — a classe que lê CONTEÚDO separa.
    achados = confere_conteudo(eixo, {ROTULO_DO_MAPA: transposto})
    assert achados, (
        "o mapa de calor com a matriz transposta não foi acusado: `confere_conteudo` é a classe "
        "que a U18 acrescentou, e sem ela esta planta não separa nada"
    )
    assert "célula" in achados[0], achados


def test_par_causal_do_conteudo_trocado_do_mapa() -> None:
    """A quinta direção do par causal, com a planta do arnês. **Em processo.**"""
    valores = matriz_de_dois_grupos(8, 6)
    mapa = mapa_de(valores)
    figura = figura_do_mapa_de_calor(mapa=mapa, recorte=recorte_de_mapa(mapa))
    eixo = figura.axes[0]
    esperado = valores[np.ix_(mapa.ordem_das_amostras, mapa.ordem_das_features)].T
    assert confere_conteudo(eixo, {ROTULO_DO_MAPA: esperado}) == [], (
        "sem a planta o portão já acusava"
    )
    rotulo = planta_conteudo_trocado(eixo)
    assert rotulo == ROTULO_DO_MAPA
    assert confere_conteudo(eixo, {ROTULO_DO_MAPA: esperado}), "a planta não foi acusada"


def test_par_causal_do_corte_de_eixo_do_mapa() -> None:
    """A planta que nenhuma mutação de dado alcança: a vista corta a imagem e a tela fica vazia."""
    valores = matriz_de_dois_grupos(8, 6)
    mapa = mapa_de(valores)
    figura = figura_do_mapa_de_calor(mapa=mapa, recorte=recorte_de_mapa(mapa))
    eixo = figura.axes[0]
    assert confere_vista(eixo) == [], "sem a planta a vista já era acusada"
    planta_corte_de_eixo(eixo)
    assert confere_vista(eixo), "o limite acima de todo o dado não foi acusado"


def test_os_limites_de_cor_sao_SIMETRICOS_em_torno_de_zero() -> None:  # noqa: N802
    """Sem isto o tom neutro da paleta divergente deixa de cair no zero, e a figura fica plausível.

    **A planta cai sobre dado ASSIMÉTRICO:** numa matriz cujo maior desvio positivo e
    negativo já fossem iguais, `vmin == -vmax` sairia por acaso. A âncora afirma a assimetria
    antes de a asserção correr.
    """
    valores = matriz_de_dois_grupos(8, 6).copy()
    valores[0, 0] = 9.0
    assert abs(valores.max()) != abs(valores.min()), (
        "âncora: o dado é simétrico e a asserção abaixo passaria por acaso"
    )
    mapa = mapa_de(valores)
    figura = figura_do_mapa_de_calor(mapa=mapa, recorte=recorte_de_mapa(mapa))
    imagem = figura.axes[0].images[0]
    vmin, vmax = imagem.get_clim()
    assert vmin == pytest.approx(-vmax), f"os limites são ({vmin}, {vmax}) e não são simétricos"
    assert vmax == pytest.approx(float(np.abs(valores).max())), (vmin, vmax)
    assert imagem.get_cmap().name == PALETA_DIVERGENTE


def test_uma_matriz_TODA_A_ZEROS_desenha_com_limite_de_recuo() -> None:  # noqa: N802
    """`vmin == vmax` quebra a norma do `matplotlib`, e uma matriz neutra é caso legítimo."""
    mapa = mapa_do_fio(
        valores=[[0.0, 0.0], [0.0, 0.0]],
        amostras=["A", "B"],
        features=["f1", "f2"],
    )
    figura = figura_do_mapa_de_calor(mapa=mapa, recorte=recorte_de_mapa(mapa))
    vmin, vmax = figura.axes[0].images[0].get_clim()
    assert (vmin, vmax) == (-1.0, 1.0), (vmin, vmax)


def test_a_celula_indefinida_e_CONTADA_no_rodape() -> None:  # noqa: N802
    """R6: «não foi medido» e «está na média» saem iguais da figura sem a contagem.

    **As duas direções na mesma função**, porque a frase de zero e a de N são ramos diferentes de
    `frase_contada` e só uma delas correria sozinha.
    """
    valores = matriz_de_dois_grupos(8, 6).copy()
    sem = figura_do_mapa_de_calor(mapa=(m := mapa_de(valores)), recorte=recorte_de_mapa(m))
    assert "nenhuma célula é indefinida" in rodape_de(sem)

    valores[1, 2] = np.nan
    valores[3, 4] = np.inf
    # **As ordens vão pela identidade e não pelo agrupamento, e isso é obrigatório:** a ligação de
    # uma matriz com `nan` não existe — o `scipy` recusa a distância —, e pedi-la aqui reprovaria
    # com uma excepção do `scipy` em vez de medir o que este teste mede. É a distinção entre
    # «reprovação» e «medição que não aconteceu».
    m2 = mapa_de(
        valores,
        ordem_das_amostras=list(range(valores.shape[0])),
        ordem_das_features=list(range(valores.shape[1])),
    )
    com = figura_do_mapa_de_calor(mapa=m2, recorte=recorte_de_mapa(m2))
    texto = rodape_de(com)
    assert "as 2 de 48 células são indefinidas" in texto, texto


# ---------------------------------------------------------------------------
# 3. O teto — HERDADO, e a recusa é TIPADA
# ---------------------------------------------------------------------------


def test_acima_do_teto_a_recusa_e_TIPADA_e_traz_a_contagem_E_o_teto() -> None:  # noqa: N802
    """**Par causal do teto, nas duas direções, e a mensagem carrega os dois números.**

    Sem a contagem e o teto na mensagem, quem chama sabe que falhou e não sabe por quanto — e a
    resposta certa (amostrar antes de chamar) exige o número.
    """
    teto = TETO_DE_FEATURES_PARA_DESENHAR
    no_teto = mapa_de(matriz_de_dois_grupos(4, teto))
    assert figura_do_mapa_de_calor(mapa=no_teto, recorte=recorte_de_mapa(no_teto)) is not None

    acima = mapa_de(matriz_de_dois_grupos(4, teto + 1))
    with pytest.raises(FeaturesDemaisError) as erro:
        figura_do_mapa_de_calor(mapa=acima, recorte=recorte_de_mapa(acima))
    mensagem = str(erro.value)
    assert str(teto + 1) in mensagem and str(teto) in mensagem, mensagem
    assert isinstance(erro.value, RecusaDoConstrutorError), (
        "a recusa não desce de `RecusaDoConstrutorError`: dentro de um slot do Qt ela seria "
        "engolida para o `stderr` e o painel pararia de actualizar sem dizer nada"
    )


def importa_de(texto: str, modulo: str) -> set[str]:
    """Os nomes que `texto` importa de `modulo`, lidos da ÁRVORE e não do texto cru.

    > **O ORÁCULO DESTE PORTÃO MUDOU, e a razão é o corolário.** Ele era a cadeia
    > literal `"from refino.statistics.clustering import TETO_DE_FEATURES_PARA_DESENHAR"`, e uma
    > mudança reprovou-o **sem tocar no teto**: ela passou a importar um segundo nome do mesmo
    > módulo, o `ruff` juntou os dois num bloco entre parênteses, e a cadeia deixou de existir. O
    > portão reprovava por causa da **forma** do código, não da propriedade.
    >
    > **Afrouxá-lo seria a resposta errada; a certa é perguntar por que aquela forma era
    > contrato** — e ela não era: a propriedade é *"o nome vem daquele módulo"*, e a forma
    > single-line era só a única que o módulo tinha no dia em que o portão nasceu. A árvore
    > sintática exprime a propriedade e é indiferente à forma; **e é mais forte**, porque a cadeia
    > literal casaria também dentro de um comentário ou de uma docstring.
    """
    return {
        alias.name
        for no in ast.walk(ast.parse(texto))
        if isinstance(no, ast.ImportFrom) and no.module == modulo
        for alias in no.names
    }


def test_o_teto_do_mapa_e_O_MESMO_objeto_da_U10_e_nao_uma_copia() -> None:  # noqa: N802
    """Um teto reescrito aqui seria a segunda fonte do mesmo número."""
    fonte = __import__("refino.plots.agrupamento", fromlist=["x"]).__dict__.get("__file__")
    assert fonte is not None
    texto = Path(fonte).read_text(encoding="utf-8")
    assert "TETO_DE_FEATURES_PARA_DESENHAR" in importa_de(texto, "refino.statistics.clustering"), (
        "o módulo deixou de importar o teto da U10 — um número reescrito aqui é a irmã que "
        "ninguém procura quando o outro for remedido"
    )
    assert not re.search(r"TETO_DE_FEATURES_PARA_DESENHAR\s*[:=]", texto), (
        "o módulo passou a DEFINIR o teto em vez de o importar"
    )


@pytest.mark.parametrize(
    ("plantado", "acusa"),
    [
        ("from refino.statistics.clustering import TETO_DE_FEATURES_PARA_DESENHAR\n", False),
        (
            "from refino.statistics.clustering import (\n    TETO_DE_FEATURES_PARA_DESENHAR,\n)\n",
            False,
        ),
        ("TETO_DE_FEATURES_PARA_DESENHAR = 60\n", True),
        ("from outro.modulo import TETO_DE_FEATURES_PARA_DESENHAR\n", True),
        ("# from refino.statistics.clustering import TETO_DE_FEATURES_PARA_DESENHAR\n", True),
    ],
)
def test_o_portao_do_teto_herdado_acusa_a_planta_e_nao_acusa_as_DUAS_formas(  # noqa: N802
    plantado: str, acusa: bool
) -> None:
    """O par causal do oráculo NOVO, nas cinco direções que ele tem de separar.

    **As duas primeiras são a razão de o oráculo ter mudado** — as duas formas de importar o mesmo
    nome, e as duas são legítimas. A terceira e a quarta são o defeito que o portão existe para
    apanhar. **A quinta é o que a cadeia literal NÃO conseguia**: um import comentado casava o
    texto e não importa nada.
    """
    nomes = importa_de(plantado, "refino.statistics.clustering")
    assert ("TETO_DE_FEATURES_PARA_DESENHAR" not in nomes) is acusa, (plantado, nomes)


def test_o_teto_das_FOLHAS_e_declarado_como_derivavel_e_nao_como_numero() -> None:  # noqa: N802
    """O teto de folhas não tem número: é `None` até haver medição que o derive."""
    assert TETO_DE_FOLHAS_PARA_DESENHAR is None


def test_o_dendrograma_de_68_folhas_NAO_e_recusado() -> None:  # noqa: N802
    """A consequência do teto de folhas ser `None`: o corpus real desenha.

    **É a âncora do alvo certo para a escolha de não herdar o teto de features:** herdá-lo recusaria
    as 68 amostras do `pos_hilic`, que é o caso primário desta figura — número certo sobre a
    grandeza errada.
    """
    valores = matriz_de_dois_grupos(68, 6)
    arvore = arvore_de(valores)
    assert arvore.n_folhas == 68 > TETO_DE_FEATURES_PARA_DESENHAR
    figura = figura_do_dendrograma(arvore=arvore, recorte=recorte_de_arvore(arvore))
    assert len(figura.axes[0].get_xticklabels()) == 68


# ---------------------------------------------------------------------------
# 4. A vírgula decimal — a barra RECEBE, os nomes NÃO
# ---------------------------------------------------------------------------


def test_a_barra_de_cor_RECEBE_a_virgula_e_os_nomes_de_feature_NAO() -> None:  # noqa: N802
    """**As duas metades na MESMA função, e é essa a decisão.**

    Afirmar só que os nomes não são reescritos passaria sobre uma figura **sem** barra de cor —
    asserção de ausência satisfeita por vacuidade. A âncora é a metade de presença: a barra
    tem de trazer ao menos uma marca com vírgula.

    `21,37_508,2876m/z` não é o nome de nada: seria um identificador inventado impresso com a
    autoridade de uma marca de eixo.
    """
    valores = matriz_de_dois_grupos(8, 6)
    mapa = mapa_de(valores)
    figura = figura_do_mapa_de_calor(mapa=mapa, recorte=recorte_de_mapa(mapa))
    figura.canvas.draw()

    barra = figura.axes[1]
    marcas = [t.get_text() for t in barra.get_yticklabels() if t.get_text().strip()]
    assert marcas, "âncora: a barra de cor não tem marca nenhuma e a asserção seria vácua"
    com_virgula = [m for m in marcas if "," in m]
    assert com_virgula, f"nenhuma marca da barra levou vírgula decimal: {marcas}"
    assert not any("." in m for m in marcas), f"a barra ainda escreve ponto decimal: {marcas}"

    nomes = [t.get_text() for t in figura.axes[0].get_yticklabels()]
    assert nomes, "âncora: o eixo das features não tem rótulo e a asserção seria vácua"
    assert nomes == [mapa.features[i] for i in mapa.ordem_das_features], nomes
    assert all("." in n and "," not in n for n in nomes), (
        f"um nome de feature foi reformatado pela vírgula decimal: {nomes}"
    )


# ---------------------------------------------------------------------------
# 5. O prefixo comum dos nomes de amostra
# ---------------------------------------------------------------------------


def test_o_prefixo_comum_e_FATORADO_e_declarado_verbatim_no_rodape() -> None:  # noqa: N802
    """O nome completo reconstrói-se da imagem: prefixo do rodapé mais o rótulo do eixo."""
    nomes = [
        "HTLV_HILIC_PLASMA_MSe_POS_QC1",
        "HTLV_HILIC_PLASMA_MSe_POS_0_ID99_Asymptomatic",
        "HTLV_HILIC_PLASMA_MSe_POS_2_ID97_ATLL",
        "HTLV_HILIC_PLASMA_MSe_POS_3_ID98_HAMTSP",
    ]
    curtos, prefixo = sem_prefixo_comum(nomes)
    assert prefixo == "HTLV_HILIC_PLASMA_MSe_POS_", prefixo
    assert curtos == ("QC1", "0_ID99_Asymptomatic", "2_ID97_ATLL", "3_ID98_HAMTSP"), curtos
    assert all(p + c == n for p, c, n in zip([prefixo] * 4, curtos, nomes, strict=True))
    assert prefixo in frase_do_prefixo(prefixo)


@pytest.mark.parametrize(
    ("nomes", "motivo"),
    [
        (["a"], "um nome só"),
        (["abc", "xyz"], "prefixo comum vazio"),
        (["A_B_", "A_B_C"], "o prefixo deixaria um nome vazio"),
        (["QC1", "QC2"], "o prefixo comum não tem separador"),
    ],
)
def test_o_prefixo_NAO_e_fatorado_quando_nao_ha_o_que_fatorar(  # noqa: N802
    nomes: list[str], motivo: str
) -> None:
    """Os quatro ramos de recuo, um por execução. **Negativo em processo.**"""
    curtos, prefixo = sem_prefixo_comum(nomes)
    assert prefixo == "", f"{motivo}: fatorou {prefixo!r}"
    assert curtos == tuple(nomes)
    assert "nomes completos" in frase_do_prefixo(prefixo)


def test_o_rodape_do_dendrograma_declara_quantas_de_quantas_e_QUAL_distancia() -> None:  # noqa: N802
    """R3 e R5b: o recorte na imagem declara a população, a amostragem e a distância."""
    valores = matriz_de_dois_grupos(8, 6)
    arvore = arvore_de(valores)
    figura = figura_do_dendrograma(arvore=arvore, recorte=recorte_de_arvore(arvore))
    texto = rodape_de(figura)
    assert "6 de 6.035 features" in texto, texto
    assert AMOSTRAGEM in texto and MATRIZ in texto, texto
    assert "distância euclidiana · ligação de Ward" in texto, texto
    assert "8 folhas" in texto, texto


# ---------------------------------------------------------------------------
# 6. O vocabulário fechado, e as guardas de fronteira
# ---------------------------------------------------------------------------


def test_os_NOMES_cobrem_o_Literal_do_dominio_nas_duas_direcoes() -> None:  # noqa: N802
    """Portão de paridade: nome sem método, ou método sem nome, reprova.

    É o mesmo oráculo que `tests/unit/test_labels.py` corre sobre
    `CLUSTER_METRIC_IMPLEMENTATIONS`, aqui sobre a tradução para o rodapé — que é outra superfície
    e tem de ser conferida por ela própria.
    """
    assert set(NOME_DA_METRICA) == set(get_args(ClusterMetric))
    assert set(NOME_DA_LIGACAO) == set(get_args(ClusterLinkage))
    assert all(v.strip() for v in NOME_DA_METRICA.values())
    assert all(v.strip() for v in NOME_DA_LIGACAO.values())


@pytest.mark.parametrize(
    ("metrica", "ligacao"),
    [("cosine", "ward"), ("euclidean", "centroid"), ("", "")],
)
def test_um_metodo_FORA_do_vocabulario_RECUSA_em_vez_de_cair_num_else(  # noqa: N802
    metrica: str, ligacao: str
) -> None:
    """Na fronteira, um nome desconhecido escrito cru no rodapé é procedência inventada."""
    with pytest.raises(MetodoDesconhecidoError, match="vocabulário fechado"):
        frase_do_metodo(metrica, ligacao)


@pytest.mark.parametrize(
    ("argumentos", "trecho"),
    [
        ({"ligacao": [[0.0, 1.0, 1.0]], "rotulos": ["a", "b"]}, "formato do `scipy`"),
        ({"ligacao": [[0.0, 9.0, 1.0, 2.0]], "rotulos": ["a", "b"]}, "nó ainda não formado"),
        ({"ligacao": [[0.0, 1.0, -1.0, 2.0]], "rotulos": ["a", "b"]}, "altura de junção negativa"),
        ({"ligacao": [[0.0, 1.0, 1.0, 7.0]], "rotulos": ["a", "b"]}, "contagem de folhas fora"),
        ({"ligacao": [[0.0, 1.0, float("nan"), 2.0]], "rotulos": ["a", "b"]}, "valor não finito"),
        ({"ligacao": [[0.0, 1.0, 1.0, 2.0]], "rotulos": ["a"]}, "chegaram 1 rótulo"),
        ({"ligacao": [[0.0, 1.0, 1.0, 2.0]], "rotulos": ["a", "  "]}, "rótulo de folha em branco"),
        ({"ligacao": "nada disto", "rotulos": ["a", "b"]}, "não converte"),
    ],
)
def test_a_arvore_do_fio_RECUSA_o_que_nao_e_arvore(  # noqa: N802
    argumentos: dict[str, Any], trecho: str
) -> None:
    """Cinco formas de ligação inválida, uma por execução. **Negativo em processo.**"""
    with pytest.raises(ArvoreInvalidaError, match=re.escape(trecho)):
        arvore_do_fio(metrica="euclidean", ligacao_metodo="ward", **argumentos)


def test_a_arvore_VALIDA_nao_levanta_e_a_ligacao_fica_IMUTAVEL() -> None:  # noqa: N802
    """A outra ponta do par: sem planta não acusa — e a imutabilidade é a regra 7 do README."""
    arvore = arvore_de(matriz_de_dois_grupos(8, 6))
    assert arvore.n_folhas == 8
    with pytest.raises(ValueError, match="read-only"):
        arvore.ligacao[0, 0] = 99.0


@pytest.mark.parametrize(
    ("argumentos", "excecao", "trecho"),
    [
        ({"valores": []}, MatrizVaziaError, "chegou vazia"),
        ({"valores": [1.0, 2.0]}, MapaDesalinhadoError, "dimensão"),
        ({"amostras": ["só uma"]}, MapaDesalinhadoError, "nome(s) de amostra"),
        ({"features": ["só uma"]}, MapaDesalinhadoError, "de feature"),
        ({"grupos": ["a"]}, MapaDesalinhadoError, "grupo(s) para"),
        ({"ordem_das_amostras": [0, 0, 2, 3]}, MapaDesalinhadoError, "não é uma permutação"),
        ({"ordem_das_features": [0, 1]}, MapaDesalinhadoError, "não é uma permutação"),
        ({"ordem_das_amostras": "x"}, MapaDesalinhadoError, "não é uma sequência"),
    ],
)
def test_o_mapa_do_fio_RECUSA_desalinhamento(  # noqa: N802
    argumentos: dict[str, Any], excecao: type[Exception], trecho: str
) -> None:
    """Oito formas de desalinhamento, uma por execução. **Negativo em processo.**

    Rótulo deslocado produz figura bem formada com o nome errado em cada coluna — a única classe de
    defeito de figura que ninguém vê olhando para ela.
    """
    base: dict[str, Any] = {
        "valores": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.5], [1.0, 0.0, 2.0]],
        "amostras": ["a", "b", "c", "d"],
        "features": ["f1", "f2", "f3"],
    }
    base.update(argumentos)
    with pytest.raises(excecao, match=re.escape(trecho)):
        mapa_do_fio(**base)


def test_o_mapa_VALIDO_nao_levanta_e_a_matriz_fica_IMUTAVEL() -> None:  # noqa: N802
    """A outra ponta do par causal acima — sem a qual as oito recusas poderiam acusar tudo."""
    mapa = mapa_do_fio(
        valores=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.5], [1.0, 0.0, 2.0]],
        amostras=["a", "b", "c", "d"],
        features=["f1", "f2", "f3"],
    )
    assert mapa.ordem_das_amostras == (0, 1, 2, 3)
    assert mapa.ordem_das_features == (0, 1, 2)
    assert mapa.grupos == ()
    with pytest.raises(ValueError, match="read-only"):
        mapa.valores[0, 0] = 99.0


def test_o_dendrograma_RECUSA_vetor_de_grupos_desalinhado() -> None:  # noqa: N802
    """A irmã da guarda do mapa, no outro construtor."""
    arvore = arvore_de(matriz_de_dois_grupos(8, 6))
    with pytest.raises(MapaDesalinhadoError, match="grupo"):
        figura_do_dendrograma(arvore=arvore, grupos=["controle"], recorte=recorte_de_arvore(arvore))


# ---------------------------------------------------------------------------
# 7. O K-means
# ---------------------------------------------------------------------------

GRUPOS_DE_OITO: Final = ["controle"] * 4 + ["tratado"] * 4
ROTULOS_DE_DUAS_PARTES: Final = [0, 0, 0, 1, 1, 1, 1, 1]
SEMENTE_DA_FIGURA: Final = 1001697648
"""Uma semente com dez dígitos **de propósito**: é o comprimento que as sementes derivadas por
`SeedSequence` de facto têm, e é sobre um número desses que a separação de milhar seria visível."""


def particao_de(rotulos: list[int], grupos: list[str] | tuple[()] = (), k: int = 2) -> Particao:
    return kmeans_do_fio(rotulos=rotulos, grupos=list(grupos), k=k, inercia=12.5)


def recorte_de_particao(particao: Particao) -> Any:
    return recorte_do_kmeans(
        particao=particao,
        n_features_usadas=6,
        n_features_totais=POPULACAO,
        matriz=MATRIZ,
        amostragem=AMOSTRAGEM,
        semente=SEMENTE_DA_FIGURA,
    )


def test_a_figura_limpa_do_kmeans_passa_nas_quatro_classes() -> None:
    """O caminho feliz, e a outra ponta de cada par causal desta família."""
    particao = particao_de(ROTULOS_DE_DUAS_PARTES, GRUPOS_DE_OITO)
    figura = figura_do_kmeans(particao=particao, recorte=recorte_de_particao(particao))
    eixo = figura.axes[0]

    # Classe (a): uma série por grupo clínico, e o dado de cada uma é a barra empilhada dele.
    # **Derivado aqui e não pedido ao construtor** — é o que faz a classe valer alguma coisa.
    assert (
        confere_series(
            eixo,
            {
                # `controle` são as amostras 0..3: TRÊS no agrupamento 0 e uma no 1. Dois segmentos.
                "controle — 4 amostras": ([0.0, 0.0, 1.0, 1.0], [0.0, 3.0, 0.0, 1.0]),
                # `tratado` são as 4..7: NENHUMA no agrupamento 0 e quatro no 1 — **um** segmento, e
                # não dois. É a consequência visível de a implementação omitir o segmento de
                # comprimento zero: dois vértices coincidentes entrariam na classe (a) como pontos
                # que ninguém vê.
                "tratado — 4 amostras": ([1.0, 1.0], [1.0, 5.0]),
            },
        )
        == []
    ), confere_series(eixo, {})
    assert confere_vista(eixo) == []
    assert (
        confere_rotulos(
            eixo,
            x="agrupamento",
            y="amostras",
            titulo=TITULO_DO_KMEANS,
            legenda=("controle — 4 amostras", "tratado — 4 amostras"),
        )
        == []
    )

    declarados = numeros_declarados(figura)
    for esperado in ("6.035", "4", "2", str(INICIALIZACOES_DO_KMEANS), "8"):
        assert esperado in declarados, (esperado, declarados)


def test_a_semente_do_rodape_sai_SEM_separador_de_milhar() -> None:  # noqa: N802
    """O achado da imagem montada, aqui como portão.

    **A planta é a própria semente de dez dígitos**: com `milhar` aplicado, o rodapé traria
    `1.001.697.648` — um identificador que ninguém consegue colar de volta. A asserção é nas duas
    direções: o número cru **está** na frase, e a forma pontuada **não** está.
    """
    particao = particao_de(ROTULOS_DE_DUAS_PARTES, GRUPOS_DE_OITO)
    rodape = recorte_de_particao(particao).texto()
    assert str(SEMENTE_DA_FIGURA) in rodape
    assert "1.001.697.648" not in rodape
    assert frase_da_semente(SEMENTE_DA_FIGURA) == str(SEMENTE_DA_FIGURA)


def test_o_kmeans_SEM_grupos_desenha_uma_serie_com_a_contagem() -> None:  # noqa: N802
    """O outro ramo. **E ele é o motivo de `planta_serie_removida` correr sobre a variante COM
    grupos:** uma série só não pode ser plantada, e a planta recusa-se em vez de passar."""
    particao = particao_de(ROTULOS_DE_DUAS_PARTES)
    figura = figura_do_kmeans(particao=particao, recorte=recorte_de_particao(particao))
    desenhadas = {s.rotulo for s in series(figura.axes[0])}
    assert desenhadas == {"8 amostras"}, desenhadas


def test_par_causal_serie_removida_do_kmeans() -> None:
    """Com a planta acusa, sem a planta não acusa. **A planta é ARGUMENTO.**"""
    particao = particao_de(ROTULOS_DE_DUAS_PARTES, GRUPOS_DE_OITO)
    esperadas = {
        s.rotulo: (s.x, s.y)
        for s in series(
            figura_do_kmeans(particao=particao, recorte=recorte_de_particao(particao)).axes[0]
        )
    }
    assert len(esperadas) == 2, "âncora: a planta precisa de ao menos duas séries"

    limpa = figura_do_kmeans(particao=particao, recorte=recorte_de_particao(particao)).axes[0]
    assert confere_series(limpa, esperadas) == []

    mutilada = figura_do_kmeans(particao=particao, recorte=recorte_de_particao(particao)).axes[0]
    removido = planta_serie_removida(mutilada)
    achados = confere_series(mutilada, esperadas)
    assert achados and removido in achados[0], achados


def test_par_causal_do_corte_de_eixo_do_kmeans() -> None:
    """A quarta planta: dado nenhum muda e a tela fica em branco."""
    particao = particao_de(ROTULOS_DE_DUAS_PARTES, GRUPOS_DE_OITO)
    figura = figura_do_kmeans(particao=particao, recorte=recorte_de_particao(particao))
    eixo = figura.axes[0]
    assert confere_vista(eixo) == []
    planta_corte_de_eixo(eixo)
    assert confere_vista(eixo) != []


def test_par_causal_do_k_uma_particao_com_OUTRO_k_desenha_outra_figura() -> None:  # noqa: N802
    """O `k` é lido, e não codificado. **A planta cai sobre dado em que as duas coisas diferem.**

    A âncora é explícita: as duas partições têm o mesmo vetor de rótulos até onde ele cabe, e o que
    muda é o número de agrupamentos — então um construtor que derivasse `k` do máximo dos rótulos,
    em vez de o ler do contrato, desenharia as duas iguais.
    """
    com_dois = particao_de(ROTULOS_DE_DUAS_PARTES, GRUPOS_DE_OITO, k=2)
    com_tres = particao_de([0, 0, 0, 1, 1, 1, 1, 2], GRUPOS_DE_OITO, k=3)
    assert com_dois.k != com_tres.k, "a planta não separa: os dois `k` coincidem"

    eixo_dois = figura_do_kmeans(particao=com_dois, recorte=recorte_de_particao(com_dois)).axes[0]
    eixo_tres = figura_do_kmeans(particao=com_tres, recorte=recorte_de_particao(com_tres)).axes[0]
    assert [s.x.tolist() for s in series(eixo_dois)] != [s.x.tolist() for s in series(eixo_tres)]
    assert eixo_dois.get_xlim() != eixo_tres.get_xlim()
    assert recorte_de_particao(com_dois).texto() != recorte_de_particao(com_tres).texto()


def test_k_IGUAL_ao_numero_de_amostras_desenha_uma_barra_por_amostra() -> None:  # noqa: N802
    """A borda do lado do desenho: oito agrupamentos de uma amostra cada."""
    particao = particao_de(list(range(8)), GRUPOS_DE_OITO, k=8)
    assert particao.tamanhos == (1,) * 8
    figura = figura_do_kmeans(particao=particao, recorte=recorte_de_particao(particao))
    assert confere_vista(figura.axes[0]) == []


def test_o_kmeans_do_fio_RECUSA_rotulo_fora_do_k_e_grupo_desalinhado() -> None:  # noqa: N802
    """As duas guardas de fronteira, e as duas com a mesma recusa tipada."""
    with pytest.raises(ParticaoDesalinhadaError, match=r"\[7\]"):
        kmeans_do_fio(rotulos=[0, 1, 7], k=2)
    with pytest.raises(ParticaoDesalinhadaError, match="grupo"):
        kmeans_do_fio(rotulos=[0, 1, 0], grupos=["a"], k=2)
    with pytest.raises(MatrizVaziaError, match="agrupamentos"):
        kmeans_do_fio(rotulos=[0], k=0)
    with pytest.raises(MatrizVaziaError, match="vazio"):
        kmeans_do_fio(rotulos=[], k=2)


def test_o_kmeans_do_fio_VALIDO_nao_levanta_e_deriva_os_tamanhos() -> None:  # noqa: N802
    """A outra ponta do par causal acima — sem ela as recusas poderiam acusar tudo."""
    particao = kmeans_do_fio(rotulos=[0, 1, 1, 2], grupos=["a", "a", "b", "b"], k=3, inercia=1.5)
    assert particao.tamanhos == (1, 2, 1)
    assert particao.n_amostras == 4
    assert particao.inercia == pytest.approx(1.5)


# ---------------------------------------------------------------------------
# 8. O mapa auto-organizável
# ---------------------------------------------------------------------------

UNIDADES_DE_OITO: Final = [0, 0, 1, 1, 4, 4, 5, 5]
"""Duas amostras por unidade em quatro das seis unidades de uma grade `2 x 3`.

**Duas por unidade é requisito e não conveniência:** com uma só, o desvio dentro da célula seria
sempre zero e o teste de `posicoes_na_grade` passaria sobre uma implementação que não o calculasse.
"""


def grade_de(unidades: list[int], grupos: list[str] | tuple[()] = ()) -> Grade:
    return som_do_fio(
        unidades=unidades,
        grupos=list(grupos),
        linhas=2,
        colunas=3,
        epocas=100,
        erro_de_quantizacao=44.69,
    )


def recorte_de_grade(grade: Grade) -> Any:
    return recorte_do_mapa_auto_organizavel(
        grade=grade,
        n_features_usadas=6,
        n_features_totais=POPULACAO,
        matriz=MATRIZ,
        amostragem=AMOSTRAGEM,
        semente=SEMENTE_DA_FIGURA,
    )


def test_a_figura_limpa_do_som_passa_nas_quatro_classes() -> None:
    """O caminho feliz, e a outra ponta de cada par causal desta família."""
    grade = grade_de(UNIDADES_DE_OITO, GRUPOS_DE_OITO)
    figura = figura_do_mapa_auto_organizavel(grade=grade, recorte=recorte_de_grade(grade))
    eixo = figura.axes[0]

    # Classe (a): as coordenadas saem de `posicoes_na_grade`, que é função pública e testada
    # abaixo contra a aritmética da grade — não é o construtor a conferir-se a si próprio.
    posicoes = posicoes_na_grade(grade)
    assert (
        confere_series(
            eixo,
            {
                "controle — 4 amostras": (posicoes[:4, 0], posicoes[:4, 1]),
                "tratado — 4 amostras": (posicoes[4:, 0], posicoes[4:, 1]),
            },
        )
        == []
    ), confere_series(eixo, {})
    assert confere_vista(eixo) == []
    assert (
        confere_rotulos(
            eixo,
            x="coluna da grade",
            y="linha da grade",
            titulo=TITULO_DO_SOM,
            legenda=("controle — 4 amostras", "tratado — 4 amostras"),
        )
        == []
    )

    declarados = numeros_declarados(figura)
    for esperado in ("6.035", "4", "100", "2", "3", "8", "44,69"):
        assert esperado in declarados, (esperado, declarados)


def test_o_eixo_y_do_som_e_INVERTIDO_para_a_unidade_zero_ficar_em_cima() -> None:  # noqa: N802
    """O índice plano cresce para baixo, e a figura tem de concordar com o número do rodapé."""
    grade = grade_de(UNIDADES_DE_OITO, GRUPOS_DE_OITO)
    figura = figura_do_mapa_auto_organizavel(grade=grade, recorte=recorte_de_grade(grade))
    inferior, superior = figura.axes[0].get_ylim()
    assert inferior > superior, "o eixo y não está invertido: a unidade 0 ficaria em baixo"


def test_par_causal_dos_eixos_trocados_do_som() -> None:
    """Trocar linha por coluna TRANSPÕE a grade, e a classe (a) tem de acusar.

    **A âncora é a grade NÃO quadrada**: numa `2 x 3` as coordenadas `x` e `y` de uma amostra
    diferem para quase todas, e a planta separa. Numa quadrada com a diagonal ocupada ela poderia
    ser a identidade, que é o que `planta_eixos_trocados` recusa.
    """
    grade = grade_de(UNIDADES_DE_OITO, GRUPOS_DE_OITO)
    esperadas = {
        s.rotulo: (s.x, s.y)
        for s in series(
            figura_do_mapa_auto_organizavel(grade=grade, recorte=recorte_de_grade(grade)).axes[0]
        )
    }
    limpa = figura_do_mapa_auto_organizavel(grade=grade, recorte=recorte_de_grade(grade)).axes[0]
    assert confere_series(limpa, esperadas) == []

    mutilada = figura_do_mapa_auto_organizavel(grade=grade, recorte=recorte_de_grade(grade)).axes[0]
    trocada = planta_eixos_trocados(mutilada)
    achados = confere_series(mutilada, esperadas)
    assert achados and trocada in achados[0], achados


def test_par_causal_serie_removida_do_som() -> None:
    grade = grade_de(UNIDADES_DE_OITO, GRUPOS_DE_OITO)
    esperadas = {
        s.rotulo: (s.x, s.y)
        for s in series(
            figura_do_mapa_auto_organizavel(grade=grade, recorte=recorte_de_grade(grade)).axes[0]
        )
    }
    assert len(esperadas) == 2, "âncora: a planta precisa de ao menos duas séries"
    mutilada = figura_do_mapa_auto_organizavel(grade=grade, recorte=recorte_de_grade(grade)).axes[0]
    removido = planta_serie_removida(mutilada)
    achados = confere_series(mutilada, esperadas)
    assert achados and removido in achados[0], achados


def test_as_posicoes_na_grade_sao_DETERMINISTICAS_e_nao_se_sobrepoem() -> None:  # noqa: N802
    """Regra 3 do README do lado do desenho, e a segunda metade é o que o desvio existe para fazer.

    **As duas asserções são diferentes e as duas importam:** repetível é o que a regra exige;
    *sem sobreposição dentro da unidade* é o que impede uma unidade com quatro amostras de parecer
    uma com uma. Um construtor que desenhasse tudo no centro da célula passaria na primeira.
    """
    grade = grade_de(UNIDADES_DE_OITO, GRUPOS_DE_OITO)
    primeira = posicoes_na_grade(grade)
    assert np.array_equal(primeira, posicoes_na_grade(grade))

    distintas = {(round(float(x), 9), round(float(y), 9)) for x, y in primeira}
    assert len(distintas) == grade.n_amostras, (
        "duas amostras foram desenhadas no MESMO ponto: a de baixo fica invisível, e a unidade "
        "parece ter menos população do que tem"
    )

    # E a coordenada inteira é a da unidade: a coluna no x, a linha no y — nunca o contrário.
    for indice, unidade in enumerate(grade.unidades):
        assert abs(primeira[indice, 0] - unidade % grade.colunas) < 0.5
        assert abs(primeira[indice, 1] - unidade // grade.colunas) < 0.5


def test_a_grade_de_UMA_unidade_desenha_e_a_MAIOR_que_a_coorte_declara_as_vazias() -> None:  # noqa: N802
    """As duas bordas, com desfechos **distintos** — e a segunda é R12b."""
    uma = som_do_fio(unidades=[0, 0, 0], linhas=1, colunas=1, epocas=100)
    assert uma.n_vazias == 0
    assert "nenhuma unidade ficou vazia" in frase_da_grade(uma)
    figura = figura_do_mapa_auto_organizavel(
        grade=uma,
        recorte=recorte_do_mapa_auto_organizavel(
            grade=uma,
            n_features_usadas=6,
            n_features_totais=POPULACAO,
            matriz=MATRIZ,
            amostragem=AMOSTRAGEM,
            semente=SEMENTE_DA_FIGURA,
        ),
    )
    assert confere_vista(figura.axes[0]) == []

    vazias = grade_de([0, 0, 1], GRUPOS_DE_OITO[:3])
    assert vazias.n_vazias == 4
    frase = frase_da_grade(vazias)
    assert "as 4 de 6 unidades não receberam amostra nenhuma" in frase
    assert "fronteira" in frase


def test_o_som_do_fio_RECUSA_unidade_fora_da_grade_e_grade_sem_lado() -> None:  # noqa: N802
    """As guardas de fronteira do SOM — as irmãs das do K-means, com a mesma recusa tipada."""
    with pytest.raises(ParticaoDesalinhadaError, match=r"\[9\]"):
        som_do_fio(unidades=[0, 1, 9], linhas=2, colunas=3)
    with pytest.raises(ParticaoDesalinhadaError, match="grupo"):
        som_do_fio(unidades=[0, 1, 2], grupos=["a"], linhas=2, colunas=3)
    with pytest.raises(MatrizVaziaError, match="unidades"):
        som_do_fio(unidades=[0], linhas=0, colunas=3)
    with pytest.raises(MatrizVaziaError, match="vazio"):
        som_do_fio(unidades=[], linhas=2, colunas=3)


def test_o_som_do_fio_VALIDO_nao_levanta_e_deriva_as_contagens() -> None:  # noqa: N802
    """A outra ponta do par causal acima."""
    grade = som_do_fio(unidades=[0, 0, 5], linhas=2, colunas=3, epocas=7, erro_de_quantizacao=1.25)
    assert grade.tamanhos == (2, 0, 0, 0, 0, 1)
    assert grade.n_vazias == 4
    assert grade.epocas == 7
    assert grade.erro_de_quantizacao == pytest.approx(1.25)


def test_as_DUAS_figuras_novas_declaram_o_ajuste_e_NAO_a_qualidade() -> None:  # noqa: N802
    """Regra 5 do README na prosa do rodapé: nenhuma das duas oferece uma métrica única de
    qualidade.

    **A asserção é sobre a palavra**, e ela é o mecanismo: as duas quantidades caem
    monotonicamente com o parâmetro que descrevem, então um rodapé que as chamasse de «qualidade»
    ensinaria a escolher sempre o maior `k` e a maior grade.
    """
    particao = particao_de(ROTULOS_DE_DUAS_PARTES, GRUPOS_DE_OITO)
    grade = grade_de(UNIDADES_DE_OITO, GRUPOS_DE_OITO)
    for rodape in (recorte_de_particao(particao).texto(), recorte_de_grade(grade).texto()):
        assert "AJUSTE e não a qualidade" in rodape
        assert "Sem teste estatístico" in rodape


# ---------------------------------------------------------------------------
# A cor e o símbolo de cada grupo nas quatro figuras
# ---------------------------------------------------------------------------
#
# A escolha é uma só e cai sobre `controle`; `tratado` fica com o padrão da coorte. Cada figura é
# lida pelo artista que ela de facto desenha — dispersão, segmento ou rótulo —, e o teste sem estilo
# de cada uma, acima, continua a afirmar o ciclo sem ter sido editado.

ESCOLHA_DO_CONTROLE: Final = EstiloDoGrupo(cor="vermelho", simbolo="quadrado")


def _estilos_dos_dois() -> EstilosDosGrupos:
    estilos = estilos_dos_grupos(GRUPOS_DE_OITO, {"controle": ESCOLHA_DO_CONTROLE})
    assert estilos is not None
    return estilos


def _simbolo_da_dispersao(colecao: Any) -> str:
    """O código do marcador de uma dispersão, lido da FORMA desenhada e não de um atributo.

    O `scatter` guarda o caminho do `MarkerStyle` já transformado; comparar os vértices com os dos
    oito símbolos devolve um só — medido nos oito antes de este leitor ser escrito.
    """
    caminho = colecao.get_paths()[0]
    for codigo in SIMBOLOS_DOS_GRUPOS.values():
        estilo = MarkerStyle(codigo)
        esperado = estilo.get_path().transformed(estilo.get_transform())
        if caminho.vertices.shape == esperado.vertices.shape and np.allclose(
            caminho.vertices, esperado.vertices
        ):
            return codigo
    return "?"


def _tracos_das_dispersoes(figura: Figure) -> dict[str, tuple[str, str]]:
    """`grupo → (cor, marcador)` de cada dispersão de grupo — a árvore e a grade ficam de fora."""
    tracos: dict[str, tuple[str, str]] = {}
    for colecao in figura.axes[0].collections:
        rotulo = str(colecao.get_label())
        if " — " not in rotulo or rotulo == ROTULO_DA_ARVORE:
            continue
        cor = to_hex(colecao.get_facecolor()[0])
        tracos[rotulo.split(" — ")[0]] = (cor, _simbolo_da_dispersao(colecao))
    return tracos


def test_com_ESTILO_as_folhas_do_dendrograma_tem_a_cor_e_o_simbolo_do_grupo() -> None:  # noqa: N802
    """As folhas são uma dispersão por grupo, e levam a cor E o marcador escolhidos."""
    valores = matriz_de_dois_grupos(8, 6)
    arvore = arvore_de(valores)
    estilos = _estilos_dos_dois()
    figura = figura_do_dendrograma(
        arvore=arvore, grupos=GRUPOS_DE_OITO, recorte=recorte_de_arvore(arvore), estilos=estilos
    )
    tracos = _tracos_das_dispersoes(figura)
    assert tracos["controle"] == (to_hex("#D55E00"), "s")
    tratado = estilos.por_grupo["tratado"]
    assert tracos["tratado"] == (to_hex(tratado.hex), tratado.marcador)

    sem = _tracos_das_dispersoes(
        figura_do_dendrograma(
            arvore=arvore, grupos=GRUPOS_DE_OITO, recorte=recorte_de_arvore(arvore)
        )
    )
    assert sem["controle"] != tracos["controle"], "âncora: sem estilo a planta não separa"
    assert {marcador for _cor, marcador in sem.values()} == {"o"}


def test_com_ESTILO_as_dispersoes_do_som_tem_a_cor_e_o_simbolo_do_grupo() -> None:  # noqa: N802
    """O SOM desenha uma dispersão por grupo sobre a grade, e o estilo chega a cada uma."""
    grade = grade_de(UNIDADES_DE_OITO, GRUPOS_DE_OITO)
    figura = figura_do_mapa_auto_organizavel(
        grade=grade, recorte=recorte_de_grade(grade), estilos=_estilos_dos_dois()
    )
    tracos = _tracos_das_dispersoes(figura)
    assert tracos["controle"] == (to_hex("#D55E00"), "s")
    assert len(set(tracos.values())) == 2, tracos


def test_com_ESTILO_os_segmentos_do_kmeans_tem_a_cor_do_grupo() -> None:  # noqa: N802
    """O K-means desenha segmentos e não pontos: o estilo chega como COR, e o símbolo não entra."""
    particao = particao_de(ROTULOS_DE_DUAS_PARTES, GRUPOS_DE_OITO)
    figura = figura_do_kmeans(
        particao=particao, recorte=recorte_de_particao(particao), estilos=_estilos_dos_dois()
    )
    cores = {
        str(colecao.get_label()).split(" — ")[0]: to_hex(colecao.get_colors()[0])
        for colecao in figura.axes[0].collections
    }
    assert cores["controle"] == to_hex("#D55E00")
    assert cores["tratado"] == to_hex(_estilos_dos_dois().por_grupo["tratado"].hex)


def test_com_ESTILO_os_rotulos_do_mapa_de_calor_tem_a_cor_do_grupo() -> None:  # noqa: N802
    """O mapa de calor pinta o RÓTULO de cada amostra, e é a cor do grupo dela que vai ali."""
    valores = matriz_de_dois_grupos(8, 6)
    mapa = mapa_de(valores, grupos=GRUPOS_DE_OITO)
    figura = figura_do_mapa_de_calor(
        mapa=mapa, recorte=recorte_de_mapa(mapa), estilos=_estilos_dos_dois()
    )
    rotulos = figura.axes[0].get_xticklabels()
    esperadas = {
        grupo: to_hex(estilo.hex) for grupo, estilo in _estilos_dos_dois().por_grupo.items()
    }
    vistas = [to_hex(r.get_color()) for r in rotulos]
    devidas = [esperadas[GRUPOS_DE_OITO[indice]] for indice in mapa.ordem_das_amostras]
    assert vistas == devidas
    assert to_hex("#D55E00") in vistas, "âncora: a cor escolhida não chegou a rótulo nenhum"
