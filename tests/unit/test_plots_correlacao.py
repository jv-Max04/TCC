"""O mapa de correlação e a busca por padrão, nas quatro classes do arnês.

## A planta do conteúdo é a que importa aqui, e ela só separa por causa de `confere_conteudo`

O mapa de correlação é uma **imagem**: `confere_series` compara o retângulo do `imshow`, que é o
mesmo para qualquer conteúdo. Antes de `confere_conteudo` existir, trocar a matriz inteira não
mudava
asserção nenhuma — asserção de ausência satisfeita por vacuidade sobre uma figura cheia. O
par
causal abaixo afirma **as duas metades**: que a classe das coordenadas continua cega
e
que a do conteúdo acusa.

## A planta da ESCALA FIXA é própria desta figura, e sem ela o ±1 seria intenção

`figura_do_mapa_de_calor` deriva o limite do dado; esta fixa-o em ±1 de propósito, para duas
execuções serem comparáveis. **Um par causal afirma isso contra a alternativa:** duas matrizes cujo
maior coeficiente fora da diagonal difere por um fator de dois desenham com os **mesmos** limites.
Sem essa asserção, trocar `vmin/vmax` por um limite derivado passaria despercebido — as duas figuras
continuariam bem formadas.

## O que este arquivo NÃO afirma

**Nada sobre legibilidade.** As quatro classes leem o objeto `Figure`; nenhuma delas sabe se
sessenta
rótulos de feature cabem na altura do eixo. A medição do redesenho está na docstring de
`plots/correlacao.py` e no relatório da unidade — não aqui, porque comparação de tempo de parede
escrita à mão reprova sob contenção sem distinguir regressão de máquina ocupada.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final, get_args

import numpy as np
import pytest

from refino.domain.labels import CorrelationMethod
from refino.plots.base import (
    PALETA_DIVERGENTE,
    FeaturesDemaisError,
    MatrizVaziaError,
    RecusaDoConstrutorError,
)
from refino.plots.correlacao import (
    NOME_DA_REFERENCIA,
    NOME_DO_COEFICIENTE,
    ROTULO_DA_ESCALA_DA_CORRELACAO,
    ROTULO_DAS_CONTRARIAS,
    ROTULO_DAS_SEGUIDORAS,
    ROTULO_DO_EIXO_DO_MAPA,
    SEM_NOMES_EM_EIXO_NENHUM,
    SEM_NOMES_NUM_EIXO,
    TITULO_DA_BUSCA,
    TITULO_DO_MAPA_DE_CORRELACAO,
    BuscaDesalinhadaError,
    CorrelacaoDesalinhadaError,
    MetodoDeCorrelacaoDesconhecidoError,
    busca_do_fio,
    figura_da_busca_por_padrao,
    figura_do_mapa_de_correlacao,
    frase_da_ordem,
    frase_da_referencia,
    frase_do_coeficiente,
    mapa_de_correlacao_do_fio,
    recorte_da_busca_por_padrao,
    recorte_do_mapa_de_correlacao,
)
from refino.plots.recorte import ENTRELINHA, TAMANHO_DA_FONTE, marcas_que_cabem
from refino.statistics.clustering import TETO_DE_FEATURES_PARA_DESENHAR
from tests.arnes_de_figura import (
    confere_conteudo,
    confere_numeros,
    confere_rotulos,
    confere_series,
    confere_vista,
    milhar_do_teste,
    planta_conteudo_trocado,
    planta_corte_de_eixo,
    planta_serie_removida,
    sem_a_matriz,
    series,
    texto_corrido,
)

if TYPE_CHECKING:  # pragma: no cover - só para o mypy
    from matplotlib.figure import Figure

# A população do `pos_hilic`, literal aqui de propósito: é a fonte independente da classe (d).
POPULACAO: Final = 6035
AMOSTRAS: Final = 61
MATRIZ: Final = "a matriz corrigida por SERRF"
AMOSTRAGEM: Final = "as 5 sorteadas com semente 20260918"


def rodape_de(figura: Figure) -> str:
    """O texto do rodapé com o espaço em branco COLAPSADO. **Não é conveniência.**

    `recorte.aplica` quebra o rodapé pela largura real da figura, e a quebra cai onde calhar.
    Comparar com a quebra dentro fixa a QUEBRA como contrato quando o contrato é o TEXTO, e o
    veredito sai *"a figura não declara isto"* sobre uma frase que está lá inteira.
    """
    return " ".join(" ".join(t.get_text() for t in figura.texts).split())


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def matriz_de_correlacao(n: int = 5, *, escala: float = 1.0, seed: int = 20260918) -> np.ndarray:
    """Uma matriz de correlação legítima: simétrica, diagonal 1, fora dela escalada por `escala`.

    `escala` existe para a planta da escala fixa: duas matrizes cujo maior coeficiente fora da
    diagonal difere por um fator de dois têm de desenhar com os MESMOS limites.
    """
    bruta = np.corrcoef(np.random.default_rng(seed).normal(size=(20, n)), rowvar=False)
    fora = ~np.eye(n, dtype=bool)
    matriz = np.eye(n)
    matriz[fora] = bruta[fora] * escala
    return matriz


def nomes_de(n: int) -> tuple[str, ...]:
    """Nomes na forma `RT_mz` do laboratório — é o que o eixo de facto desenha."""
    return tuple(f"{21.37 + i * 0.01:.2f}_{508.2876 + i:.4f}m/z" for i in range(n))


def mapa_limpo(n: int = 5, *, escala: float = 1.0, metodo: str = "pearson"):
    return mapa_de_correlacao_do_fio(
        valores=matriz_de_correlacao(n, escala=escala), features=list(nomes_de(n)), metodo=metodo
    )


def figura_limpa(n: int = 5, *, escala: float = 1.0, metodo: str = "pearson") -> Figure:
    mapa = mapa_limpo(n, escala=escala, metodo=metodo)
    return figura_do_mapa_de_correlacao(
        mapa=mapa,
        recorte=recorte_do_mapa_de_correlacao(
            mapa=mapa,
            n_features_totais=POPULACAO,
            n_amostras=AMOSTRAS,
            n_indefinidas=0,
            matriz=MATRIZ,
            amostragem=AMOSTRAGEM,
        ),
    )


def busca_limpa(
    *,
    coeficiente: tuple[float, ...] = (1.0, 0.5, -0.2, -0.9),
    metodo: str = "pearson",
    tipo: str = "feature",
):
    nomes = nomes_de(len(coeficiente))
    return busca_do_fio(
        features=list(nomes),
        coeficiente=list(coeficiente),
        referencia=nomes[0] if tipo == "feature" else "0,1,2,3",
        tipo_da_referencia=tipo,
        perfil=[0.0, 1.0, 2.0, 3.0],
        ordem_dos_grupos=["ATLL", "Asymptomatic", "HAMTSP", "NoHTLV"],
        metodo=metodo,
    )


def figura_da_busca_limpa(n_features: int = 4, **kwargs: object) -> Figure:
    busca = busca_limpa(**kwargs)  # type: ignore[arg-type]
    return figura_da_busca_por_padrao(
        busca=busca,
        n_features=n_features,
        recorte=recorte_da_busca_por_padrao(
            busca=busca,
            n_desenhadas=min(n_features, len(busca.features)),
            n_features_totais=POPULACAO,
            n_amostras=AMOSTRAS,
            n_indefinidas=0,
            matriz=MATRIZ,
        ),
    )


# ---------------------------------------------------------------------------
# 1. Paridade do vocabulário
# ---------------------------------------------------------------------------


def test_o_registo_de_nomes_cobre_exatamente_o_vocabulario() -> None:
    """Coeficiente novo sem nome legível, ou nome sem coeficiente."""
    assert set(NOME_DO_COEFICIENTE) == set(get_args(CorrelationMethod))


def test_as_duas_frases_recusam_valor_FORA_do_vocabulario() -> None:  # noqa: N802
    """O `else` razoável poria a figura a declarar um método que ninguém correu."""
    with pytest.raises(MetodoDeCorrelacaoDesconhecidoError, match="pearsons"):
        frase_do_coeficiente("pearsons")
    with pytest.raises(MetodoDeCorrelacaoDesconhecidoError, match="amostra"):
        frase_da_referencia("amostra", "x")
    with pytest.raises(MetodoDeCorrelacaoDesconhecidoError, match="branco"):
        frase_da_referencia("feature", "   ")
    assert set(NOME_DA_REFERENCIA) == {"feature", "profile"}


def test_as_duas_recusas_sao_da_familia_que_o_PAINEL_apanha() -> None:  # noqa: N802
    """Fora de `RecusaDoConstrutorError`, a excepção é engolida pelo slot do Qt e o painel
    emudece."""
    assert issubclass(CorrelacaoDesalinhadaError, RecusaDoConstrutorError)
    assert issubclass(BuscaDesalinhadaError, RecusaDoConstrutorError)
    assert issubclass(MetodoDeCorrelacaoDesconhecidoError, RecusaDoConstrutorError)


# ---------------------------------------------------------------------------
# 2. O mapa de correlação — as quatro classes na figura limpa
# ---------------------------------------------------------------------------


def test_o_mapa_limpo_passa_nas_quatro_classes() -> None:
    figura = figura_limpa()
    eixo = figura.axes[0]
    esperada = matriz_de_correlacao(5)

    assert confere_conteudo(eixo, {ROTULO_DA_ESCALA_DA_CORRELACAO: esperada}) == []
    assert (
        confere_rotulos(
            eixo, x="feature", y="feature", titulo=TITULO_DO_MAPA_DE_CORRELACAO, legenda=()
        )
        == []
    )
    assert confere_vista(eixo) == []
    # Classe (d): cada número derivado AQUI, por caminho independente do construtor.
    # `5` desenhadas de `6.035`; `5` e a semente vêm da frase de amostragem que o teste escreve;
    # `61` amostras; `1` e `1` são as duas pontas da escala fixa; `5` features dão
    # `5 x 4 / 2 = 10` pares.
    assert (
        confere_numeros(
            figura,
            [
                "5",
                milhar_do_teste(POPULACAO),
                "5",
                "20260918",
                str(AMOSTRAS),
                "1",
                "1",
                "5",
                str(5 * 4 // 2),
            ],
        )
        == []
    )


def test_a_escala_e_FIXA_em_mais_e_menos_um_e_o_par_causal_o_prova() -> None:  # noqa: N802
    """Duas matrizes com máximos diferentes desenham com os MESMOS limites.

    **A âncora está antes da asserção:** os dois máximos fora da diagonal têm de diferir, senão o
    par não separa — um limite derivado do dado daria o mesmo número nas duas e o teste passaria
    sobre a implementação errada.
    """
    fraca, forte = matriz_de_correlacao(5, escala=0.3), matriz_de_correlacao(5, escala=0.9)
    fora = ~np.eye(5, dtype=bool)
    assert abs(fraca[fora]).max() < abs(forte[fora]).max() * 0.6, (
        "os dois máximos fora da diagonal não diferem o bastante: a planta não separa"
    )

    limites = [figura_limpa(escala=escala).axes[0].images[0].get_clim() for escala in (0.3, 0.9)]
    assert limites[0] == limites[1] == (-1.0, 1.0), limites


def test_a_paleta_e_a_DIVERGENTE_partilhada_e_nao_um_literal() -> None:  # noqa: N802
    """Um literal aqui seria a segunda fonte da decisão."""
    assert figura_limpa().axes[0].images[0].get_cmap().name == PALETA_DIVERGENTE


def test_par_causal_do_conteudo_trocado_e_a_coordenada_CONTINUA_cega() -> None:  # noqa: N802
    """As duas metades: a classe do conteúdo acusa e a das coordenadas não.

    Sem `confere_conteudo`, trocar a matriz inteira não mudaria asserção
    nenhuma, e a planta seria vácua.
    """
    figura = figura_limpa()
    eixo = figura.axes[0]
    esperada = matriz_de_correlacao(5)
    antes_das_coordenadas = confere_series(
        eixo, {ROTULO_DA_ESCALA_DA_CORRELACAO: ([-0.5, 4.5], [4.5, -0.5])}
    )

    rotulo = planta_conteudo_trocado(eixo)
    assert rotulo == ROTULO_DA_ESCALA_DA_CORRELACAO
    assert confere_conteudo(eixo, {ROTULO_DA_ESCALA_DA_CORRELACAO: esperada}) != [], (
        "a planta do conteúdo não separou: a classe (a) continua cega à imagem"
    )
    assert (
        confere_series(eixo, {ROTULO_DA_ESCALA_DA_CORRELACAO: ([-0.5, 4.5], [4.5, -0.5])})
        == antes_das_coordenadas
    ), "a classe das coordenadas mudou de veredito: o retângulo do `imshow` não devia ter mexido"


def test_o_mapa_declara_a_contagem_de_INDEFINIDAS_na_imagem() -> None:  # noqa: N802
    """R12b: sem a contagem, *"são 5"* e *"eram 8 e três não dão para medir"* saem iguais."""
    mapa = mapa_limpo()
    com_tres = recorte_do_mapa_de_correlacao(
        mapa=mapa,
        n_features_totais=POPULACAO,
        n_amostras=AMOSTRAS,
        n_indefinidas=3,
        matriz=MATRIZ,
        amostragem=AMOSTRAGEM,
    )
    texto = rodape_de(figura_do_mapa_de_correlacao(mapa=mapa, recorte=com_tres))
    assert "as 3 features ficaram sem coeficiente definido" in texto
    assert "nenhuma feature ficou sem coeficiente definido" in rodape_de(figura_limpa())


def test_o_mapa_declara_o_METODO_e_a_populacao_de_amostras() -> None:  # noqa: N802
    """R12: o método, o recorte e a população, na própria imagem."""
    texto = rodape_de(figura_limpa(metodo="spearman"))
    assert "rho de Spearman" in texto
    assert f"sobre {AMOSTRAS} amostras de estudo" in texto
    assert "QC e validação ficam de fora" in texto
    assert f"{milhar_do_teste(POPULACAO)} features" in texto


def test_o_mapa_ACIMA_do_teto_recusa_com_a_contagem_e_o_teto() -> None:  # noqa: N802
    """O teto é herdado, e a mensagem carrega os dois números."""
    demais = TETO_DE_FEATURES_PARA_DESENHAR + 1
    mapa = mapa_limpo(demais)
    recorte = recorte_do_mapa_de_correlacao(
        mapa=mapa,
        n_features_totais=POPULACAO,
        n_amostras=AMOSTRAS,
        n_indefinidas=0,
        matriz=MATRIZ,
        amostragem=AMOSTRAGEM,
    )
    with pytest.raises(FeaturesDemaisError) as erro:
        figura_do_mapa_de_correlacao(mapa=mapa, recorte=recorte)
    assert str(demais) in str(erro.value)
    assert str(TETO_DE_FEATURES_PARA_DESENHAR) in str(erro.value)
    assert "mapa de correlação" in str(erro.value)


def test_no_teto_EXACTO_a_figura_desenha() -> None:  # noqa: N802
    """A outra direção do par: sem ela, um teto com o sinal trocado passaria."""
    figura = figura_limpa(TETO_DE_FEATURES_PARA_DESENHAR)
    assert len(figura.axes[0].images) == 1


# ---------------------------------------------------------------------------
# 3. As guardas de fronteira do mapa
# ---------------------------------------------------------------------------


def test_matriz_NAO_quadrada_recusa() -> None:  # noqa: N802
    """Uma correlação indexa a mesma população nos dois eixos."""
    with pytest.raises(CorrelacaoDesalinhadaError, match="quadrada"):
        mapa_de_correlacao_do_fio(
            valores=[[1.0, 0.2, 0.3], [0.2, 1.0, 0.1]],
            features=["a", "b"],
            metodo="pearson",
        )


def test_nomes_DESLOCADOS_recusam() -> None:  # noqa: N802
    """Rótulo deslocado é a única classe de defeito de figura que ninguém vê olhando para ela."""
    with pytest.raises(CorrelacaoDesalinhadaError, match="nome"):
        mapa_de_correlacao_do_fio(
            valores=matriz_de_correlacao(3), features=["a", "b"], metodo="pearson"
        )


def test_nome_EM_BRANCO_recusa() -> None:  # noqa: N802
    with pytest.raises(CorrelacaoDesalinhadaError, match="branco"):
        mapa_de_correlacao_do_fio(
            valores=matriz_de_correlacao(2), features=["a", "  "], metodo="pearson"
        )


def test_matriz_VAZIA_recusa_como_populacao_vazia_e_nao_como_desalinhamento() -> None:  # noqa: N802
    """*"Não veio população"* e *"veio com a forma errada"* pedem ações opostas."""
    with pytest.raises(MatrizVaziaError):
        mapa_de_correlacao_do_fio(valores=[], features=[], metodo="pearson")


def test_a_matriz_do_MAPA_e_imutavel() -> None:  # noqa: N802
    """Quem recebe não pode mutar o que a figura vai desenhar."""
    with pytest.raises(ValueError, match="read-only"):
        mapa_limpo().valores[0, 0] = 5.0


# ---------------------------------------------------------------------------
# 4. A busca por padrão — as quatro classes e as duas séries
# ---------------------------------------------------------------------------


def test_a_busca_limpa_passa_nas_quatro_classes() -> None:
    """As duas séries por sinal, com as posições e os coeficientes que o teste derivou."""
    figura = figura_da_busca_limpa()
    eixo = figura.axes[0]

    # Ordem esperada, derivada AQUI: seleção por |r| e ordenação por r com sinal.
    # (1,0, 0,5, -0,2, -0,9) → todas entram → ordem 1,0 / 0,5 / -0,2 / -0,9.
    assert (
        confere_series(
            eixo,
            {
                # `hlines` produz um segmento por feature: dois vértices cada, em (0, y) e (r, y).
                ROTULO_DAS_SEGUIDORAS: ([0.0, 1.0, 0.0, 0.5], [0.0, 0.0, 1.0, 1.0]),
                ROTULO_DAS_CONTRARIAS: ([0.0, -0.2, 0.0, -0.9], [2.0, 2.0, 3.0, 3.0]),
            },
        )
        == []
    )
    assert (
        confere_rotulos(
            eixo,
            x=ROTULO_DA_ESCALA_DA_CORRELACAO,
            y="feature",
            titulo=TITULO_DA_BUSCA,
            legenda=(ROTULO_DAS_SEGUIDORAS, ROTULO_DAS_CONTRARIAS),
        )
        == []
    )
    assert confere_vista(eixo) == []


def test_a_ORDEM_desenhada_e_a_do_coeficiente_com_sinal() -> None:  # noqa: N802
    """O maior coeficiente em cima: `invert_yaxis` é o que põe a leitura de cima para baixo."""
    figura = figura_da_busca_limpa()
    eixo = figura.axes[0]
    nomes = nomes_de(4)
    assert [t.get_text() for t in eixo.get_yticklabels()] == list(nomes)
    cima, baixo = eixo.get_ylim()
    assert cima > baixo, "o eixo y não foi invertido: o maior coeficiente sai em baixo"


def test_a_SELECAO_e_por_modulo_e_a_ordem_continua_com_SINAL() -> None:  # noqa: N802
    """Com um teto de duas, entram `1,0` e `-0,9` — as duas de maior módulo, em ordem com sinal.

    **É a metade que separa a seleção da ordenação.** Uma seleção pelo coeficiente com sinal
    escolheria `1,0` e `0,5` e perderia a feature que mais contraria o perfil, que é metade do
    achado; uma ordenação por módulo poria `-0,9` acima de `0,5` e o leitor leria as duas juntas.
    """
    figura = figura_da_busca_limpa(n_features=2)
    nomes = nomes_de(4)
    assert [t.get_text() for t in figura.axes[0].get_yticklabels()] == [nomes[0], nomes[3]]


def test_par_causal_da_serie_removida_na_busca() -> None:
    """A planta só é aplicável porque há DUAS séries — é a âncora que `planta_serie_removida`
    exige."""
    figura = figura_da_busca_limpa()
    eixo = figura.axes[0]
    assert len(series(eixo)) == 2, "âncora: a figura tem as duas séries por sinal"

    rotulo = planta_serie_removida(eixo)
    assert rotulo == ROTULO_DAS_CONTRARIAS
    achados = confere_series(
        eixo,
        {
            ROTULO_DAS_SEGUIDORAS: ([0.0, 1.0, 0.0, 0.5], [0.0, 0.0, 1.0, 1.0]),
            ROTULO_DAS_CONTRARIAS: ([0.0, -0.2, 0.0, -0.9], [2.0, 2.0, 3.0, 3.0]),
        },
    )
    assert achados != [], "a planta da série removida não separou"


def test_par_causal_do_corte_de_eixo_na_busca() -> None:
    """A classe (c): dado nenhum muda, e a tela fica em branco."""
    figura = figura_da_busca_limpa()
    eixo = figura.axes[0]
    assert confere_vista(eixo) == [], "âncora: a figura limpa passa a classe (c)"
    planta_corte_de_eixo(eixo)
    assert confere_vista(eixo) != [], "a planta do corte não separou"


def test_so_um_LADO_desenha_so_uma_serie() -> None:  # noqa: N802
    """Entrada de legenda com zero prometeria segmentos que não estão lá."""
    eixo = figura_da_busca_limpa(coeficiente=(0.9, 0.4, 0.1)).axes[0]
    assert [s.rotulo for s in series(eixo)] == [ROTULO_DAS_SEGUIDORAS]


def test_a_busca_SEM_coeficiente_definido_declara_o_vazio_em_texto() -> None:  # noqa: N802
    """R12b: neste corpus o vazio é o caso NORMAL, e uma figura vazia sem explicação lê-se como
    defeito."""
    figura = figura_da_busca_limpa(coeficiente=(float("nan"), float("nan")))
    eixo = figura.axes[0]
    assert series(eixo) == (), "âncora: não há série desenhada"
    escritos = " ".join(t.get_text() for t in eixo.texts)
    assert "nenhuma feature tem coeficiente definido" in " ".join(escritos.split())
    assert "o vazio é um achado" in " ".join(escritos.split())


def test_a_busca_declara_a_ORDEM_DE_GRUPOS_que_assumiu() -> None:  # noqa: N802
    """**A resposta a OQ2 na imagem.** Sem ela, um perfil `0,1,2,3` não significa nada."""
    texto = rodape_de(figura_da_busca_limpa(tipo="profile"))
    assert "a ordem assumida entre os grupos é" in texto
    assert "ATLL" in texto and "NoHTLV" in texto
    assert "por ponto de código" in texto
    assert "contra o perfil por grupo «0,1,2,3»" in texto


def test_a_busca_contra_uma_FEATURE_nomeia_a_feature_e_o_perfil_dela() -> None:  # noqa: N802
    """As duas leituras são diferentes e a figura escreve qual foi."""
    texto = rodape_de(figura_da_busca_limpa(tipo="feature"))
    assert f"contra a feature «{nomes_de(1)[0]}»" in texto
    assert "a ordem assumida entre os grupos é" in texto


def test_a_frase_da_ordem_SEM_grupos_diz_isso_por_extenso() -> None:  # noqa: N802
    """Silêncio leria-se como *"a ordem não importa"*, que é outra afirmação."""
    busca = busca_do_fio(
        features=["a", "b"],
        coeficiente=[0.5, -0.5],
        referencia="a",
        tipo_da_referencia="feature",
        metodo="pearson",
    )
    assert "nenhuma ordem foi assumida" in frase_da_ordem(busca)


def test_a_busca_declara_os_numeros_que_o_teste_derivou() -> None:
    """Classe (d): o número sai do artista, e o esperado é derivado por caminho independente.

    **Os quatro números do perfil saem com a pontuação colada** — `'0,'`, `'3.'` — e isso é do
    arnês e não da figura: `arnes_de_figura.NUMERO` casa `\\d[\\d.,]*`, e a vírgula que separa
    *"ATLL = 0, Asymptomatic = 1"* entra no casamento. Escrevê-los aqui **como o arnês os lê** é o
    que mantém a asserção nas duas direções; normalizá-los seria o teste a reescrever o que o
    instrumento mediu.
    """
    figura = figura_da_busca_limpa()
    nome_da_referencia = nomes_de(1)[0]
    assert (
        confere_numeros(
            figura,
            [
                # `4` desenhadas de `6.035`; `4` de maior módulo entre as `4` medidas.
                "4",
                milhar_do_teste(POPULACAO),
                "4",
                "4",
                # Os dois números do NOME da feature de referência, que o rodapé cita verbatim.
                *nome_da_referencia.removesuffix("m/z").split("_"),
                str(AMOSTRAS),
                # O perfil, na ordem canônica, com a pontuação com que o arnês o lê.
                "0,",
                "1,",
                "2,",
                "3.",
            ],
        )
        == []
    )


# ---------------------------------------------------------------------------
# 5. As guardas de fronteira da busca
# ---------------------------------------------------------------------------


def test_vetores_PARALELOS_de_comprimento_diferente_recusam() -> None:  # noqa: N802
    with pytest.raises(BuscaDesalinhadaError, match="coeficiente"):
        busca_do_fio(
            features=["a", "b", "c"],
            coeficiente=[0.5, 0.1],
            referencia="a",
            tipo_da_referencia="feature",
            metodo="pearson",
        )


def test_perfil_com_MENOS_numeros_que_grupos_recusa() -> None:  # noqa: N802
    """O perfil é por grupo, e a ordem é o que lhe dá significado (OQ2)."""
    with pytest.raises(BuscaDesalinhadaError, match="perfil"):
        busca_do_fio(
            features=["a"],
            coeficiente=[0.5],
            referencia="0,1,2",
            tipo_da_referencia="profile",
            perfil=[0.0, 1.0],
            ordem_dos_grupos=["A", "B", "C"],
            metodo="pearson",
        )


def test_busca_SEM_feature_nenhuma_recusa_por_populacao_vazia() -> None:  # noqa: N802
    with pytest.raises(MatrizVaziaError):
        busca_do_fio(
            features=[],
            coeficiente=[],
            referencia="a",
            tipo_da_referencia="feature",
            metodo="pearson",
        )


def test_coeficiente_que_NAO_e_numero_vira_nan_em_vez_de_levantar() -> None:  # noqa: N802
    """A política do pacote: a posição sobrevive, e quem conta o indefinido é o construtor.

    `True` é excluído antes de `int`, e a ordem é o conteúdo: em Python `True` **é** um `int`, e um
    campo que chegasse `true` viraria um coeficiente de `1,0` — o valor máximo possível, na posição
    de uma feature que ninguém mediu.
    """
    busca = busca_do_fio(
        features=["a", "b", "c"],
        coeficiente=[0.5, None, True],
        referencia="a",
        tipo_da_referencia="feature",
        metodo="pearson",
    )
    assert busca.coeficiente[0] == pytest.approx(0.5)
    assert np.isnan(busca.coeficiente[1])
    assert np.isnan(busca.coeficiente[2])


def test_o_vetor_da_BUSCA_e_imutavel() -> None:  # noqa: N802
    with pytest.raises(ValueError, match="read-only"):
        busca_limpa().coeficiente[0] = 5.0


# ---------------------------------------------------------------------------
# 6. Os nomes que NÃO cabem — a legibilidade que só a imagem montada mostra
# ---------------------------------------------------------------------------


def test_o_predicado_das_marcas_e_a_ALTURA_de_uma_linha_e_nao_um_numero() -> None:  # noqa: N802
    """O limiar é `TAMANHO_DA_FONTE * ENTRELINHA` — derivado, não escolhido.

    As duas direções na mesma função: no limiar exacto cabe, um passo abaixo não.
    """
    linha = TAMANHO_DA_FONTE * ENTRELINHA
    assert marcas_que_cabem(linha * 10.0, 10) is True
    assert marcas_que_cabem(linha * 10.0 - 0.01, 10) is False
    assert marcas_que_cabem(1000.0, 0) is False, "sem marca nenhuma, a resposta é «não cabem»"


def test_com_POUCAS_features_os_nomes_saem_nos_dois_eixos() -> None:  # noqa: N802
    """A âncora do par: com cinco features os nomes cabem, e o título NÃO tem a segunda linha."""
    figura = figura_limpa(5)
    eixo = figura.axes[0]
    assert [t.get_text() for t in eixo.get_yticklabels()] == list(nomes_de(5))
    assert [t.get_text() for t in eixo.get_xticklabels()] == list(nomes_de(5))
    assert texto_corrido(sem_a_matriz(eixo.get_title())) == TITULO_DO_MAPA_DE_CORRELACAO
    assert SEM_NOMES_NUM_EIXO not in texto_corrido(eixo.get_title())


def test_no_TETO_os_nomes_somem_dos_dois_eixos_e_a_figura_DECLARA_isso() -> None:  # noqa: N802
    """**O par causal da legibilidade, e o defeito que ele fecha foi visto ao renderizar.**

    Com as 60 features do teto herdado os rótulos do eixo y saem a um passo de 3,1 px, com 11,4 px
    de altura cada: uma mancha onde deviam estar sessenta identificadores. Nenhuma das quatro
    classes do arnês o vê — o dado está desenhado, os rótulos «estão lá» e são os certos, os limites
    contêm as células, e o número declarado está certo.

    **As duas metades:** os nomes somem, **e** o título diz porquê. Sem a segunda, um eixo de
    sessenta traços sem nome lê-se como figura por acabar, que é R12b aplicado à marca.
    """
    figura = figura_limpa(TETO_DE_FEATURES_PARA_DESENHAR)
    eixo = figura.axes[0]
    assert [t.get_text() for t in eixo.get_yticklabels()] == [""] * TETO_DE_FEATURES_PARA_DESENHAR
    assert [t.get_text() for t in eixo.get_xticklabels()] == [""] * TETO_DE_FEATURES_PARA_DESENHAR
    assert SEM_NOMES_EM_EIXO_NENHUM in texto_corrido(eixo.get_title())
    # E as marcas continuam lá: é a grelha que diz quantas células há.
    assert len(eixo.get_yticks()) == TETO_DE_FEATURES_PARA_DESENHAR


def test_o_estado_do_MEIO_existe_e_a_frase_dele_e_OUTRA() -> None:  # noqa: N802
    """**TRÊS estados e não dois, e o terceiro foi o defeito.**

    A primeira escrita tinha uma frase só, e ela dizia *"o eixo oposto nomeia as mesmas"* — falso
    quando os dois eixos perdem os nomes, que é o caso do corpus real. Este cenário afirma que as
    duas frases são distintas e que a do meio é alcançável: com um número de features entre os dois
    limiares, o eixo y perde os nomes e o x fica com eles.
    """
    assert SEM_NOMES_NUM_EIXO != SEM_NOMES_EM_EIXO_NENHUM, "as duas frases colapsaram numa"

    meio = next(
        (
            n
            for n in range(2, TETO_DE_FEATURES_PARA_DESENHAR + 1)
            if [t.get_text() for t in figura_limpa(n).axes[0].get_xticklabels()] != [""] * n
            and [t.get_text() for t in figura_limpa(n).axes[0].get_yticklabels()] == [""] * n
        ),
        None,
    )
    assert meio is not None, (
        "não há N em que só um dos eixos perca os nomes: o estado do meio é inalcançável, e a "
        "frase dele seria código morto"
    )
    titulo = texto_corrido(figura_limpa(meio).axes[0].get_title())
    assert SEM_NOMES_NUM_EIXO in titulo, titulo
    assert SEM_NOMES_EM_EIXO_NENHUM not in titulo


def test_a_declaracao_vai_no_TITULO_e_nao_no_rotulo_do_eixo() -> None:  # noqa: N802
    """A segunda leitura da mesma imagem: no `ylabel` a frase saía CORTADA pelo topo.

    Um rótulo de eixo y é desenhado rodado 90°, então ocupa a **altura** do seu comprimento — e
    `recorte.requebra_o_que_couber` alcança o título e os textos marcados, **não** os rótulos de
    eixo. A asserção é que os rótulos continuam curtos e que a frase está no título.
    """
    eixo = figura_limpa(TETO_DE_FEATURES_PARA_DESENHAR).axes[0]
    assert eixo.get_xlabel() == ROTULO_DO_EIXO_DO_MAPA
    assert eixo.get_ylabel() == ROTULO_DO_EIXO_DO_MAPA
    assert SEM_NOMES_EM_EIXO_NENHUM in texto_corrido(eixo.get_title())


def test_matriz_IRREGULAR_do_fio_recusa_com_a_frase_da_conversao() -> None:  # noqa: N802
    """Guarda de fronteira: uma lista de listas irregular chega de JSON como objeto.

    **Não é `MatrizVaziaError` nem o desalinhamento de nomes**, e as três mensagens dizem coisas
    diferentes: *"não veio população"*, *"veio com a forma errada"* e *"não é uma matriz de
    números"* pedem consertos distintos do lado de quem produz o corpo.
    """
    with pytest.raises(CorrelacaoDesalinhadaError, match="não converte"):
        mapa_de_correlacao_do_fio(
            valores=[[1.0, 0.2], [0.3]], features=["a", "b"], metodo="pearson"
        )
