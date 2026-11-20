"""O violino de RSD por feature.

> **Saiu de `test_plots_correcao.py` em 17/09/2026**, com `plots/dispersao.py`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, get_args

import numpy as np
import pytest

from refino.domain.labels import Correction
from refino.plots.base import (
    CORRETOR_NAO_DECLARADO,
    NAO_DECLARADA,
    REGUA_AUSENTE,
    MatrizVaziaError,
    nova_figura,
)
from refino.plots.catalogo import CONSTRUTORES, Vinculo
from refino.plots.dispersao import (
    MATRIZ_DA_LEITURA,
    PAPEL_DA_DENSIDADE,
    PAPEL_DA_MEDIANA,
    PAPEL_DOS_QUARTIS,
    ROTULO_DAS_AMOSTRAS,
    ROTULO_DO_EIXO_DO_RSD,
    ROTULO_DOS_QCS,
    TITULO_DO_VIOLINO,
    LadoDoRsd,
    ParDoRsd,
    figura_do_violino_do_rsd_por_feature,
    frase_do_lado_ausente,
    frase_do_lado_do_rsd,
    lado_do_rsd,
    par_do_rsd,
    recorte_do_violino_do_rsd_por_feature,
    rotulo_do_rsd,
)
from refino.plots.recorte import (
    em_percentagem,
    linhas_do_rodape,
)
from refino.reporting.model import ITENS_OBRIGATORIOS
from tests.arnes_de_figura import (
    confere_a_matriz_do_titulo,
    confere_series,
    confere_vista,
    milhar_do_teste,
    numeros_declarados,
    planta_corte_de_eixo,
    planta_eixos_trocados,
    sem_a_matriz,
    series,
    texto_corrido,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes
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
# 8. O VIOLINO de RSD por feature
# ---------------------------------------------------------------------------
#
# **As populações são pequenas de propósito, e os números reais ficam na prosa.** A tabela medida em
# 17/09/2026 vive na docstring de `RsdPorFeature`: no `pos_hilic` são 6.035 features, 5.801 finitas
# nos QCs e 5.795 nas amostras. Aqui bastam duas contagens que **diferem** e um denominador que as
# comporta — o que se mede é que a figura escreve a de cada lado, não qual é ela.

N_FEATURES_DO_RSD = 300
N_FINITAS_DOS_QCS = 240
N_FINITAS_DAS_AMOSTRAS = 197
N_INJECOES_DOS_QCS = 7
N_INJECOES_DAS_AMOSTRAS = 61


def valores_do_rsd(quantos: int, *, centro: float, semente: int) -> list[float]:
    """Um vetor de RSD **em fração**, assimétrico à direita como o do corpus real.

    **Semeado — regra 3 do README.** Dois desenhos da mesma fixtura têm de dar a mesma figura, e um
    sorteio sem semente faria a classe (a) reprovar de forma intermitente, que é pior que reprovar
    sempre.
    """
    gerador = np.random.default_rng(semente)
    bruto = gerador.lognormal(mean=float(np.log(centro)), sigma=0.45, size=quantos)
    return [float(v) for v in bruto]


def lado_do_rsd_de_prova(
    rotulo: str = ROTULO_DOS_QCS,
    *,
    quantos: int = N_FINITAS_DOS_QCS,
    centro: float = 0.12,
    semente: int = 11,
    valores: Any = None,
    n_features_finitas: int | None = None,
    n_injecoes: int = N_INJECOES_DOS_QCS,
) -> LadoDoRsd:
    brutos = (
        valores if valores is not None else valores_do_rsd(quantos, centro=centro, semente=semente)
    )
    return lado_do_rsd(
        rotulo=rotulo,
        valores=brutos,
        n_features_finitas=(
            n_features_finitas if n_features_finitas is not None else len(list(brutos))
        ),
        n_injecoes=n_injecoes,
    )


def lado_das_amostras_de_prova(**troca: Any) -> LadoDoRsd:
    campos: dict[str, Any] = {
        "quantos": N_FINITAS_DAS_AMOSTRAS,
        "centro": 0.35,
        "semente": 23,
        "n_injecoes": N_INJECOES_DAS_AMOSTRAS,
    }
    campos.update(troca)
    return lado_do_rsd_de_prova(ROTULO_DAS_AMOSTRAS, **campos)


def par_do_rsd_de_prova(**troca: Any) -> ParDoRsd:
    """Os dois lados com contagens e centros DIFERENTES — é o que torna as plantas honestas.

    Os QCs dispersam menos que as amostras: é a leitura que a figura existe para dar. Dois lados
    com o mesmo vetor fariam a troca de um pelo outro ser a identidade.
    """
    return par_do_rsd(
        qcs=troca.pop("qcs", lado_do_rsd_de_prova()),
        amostras=troca.pop("amostras", lado_das_amostras_de_prova()),
        matriz=troca.pop("matriz", "corrigida"),
        corretor=troca.pop("corretor", "loess"),
        variante=troca.pop("variante", "outlier_removed"),
        estimador=troca.pop("estimador", "in_sample"),
        n_features=troca.pop("n_features", N_FEATURES_DO_RSD),
    )


def violino_de_prova(par: ParDoRsd | None = None) -> Figure:
    par = par if par is not None else par_do_rsd_de_prova()
    return figura_do_violino_do_rsd_por_feature(
        par=par, recorte=recorte_do_violino_do_rsd_por_feature(par=par)
    )


def eixo_do_violino(figura: Figure) -> Axes:
    return figura.axes[0]


def declarados_no_eixo(figura: Figure) -> list[str]:
    """As frases que o construtor pôs DENTRO do eixo, com o espaço colapsado."""
    return [texto_corrido(t.get_text()) for t in eixo_do_violino(figura).texts]


def test_o_n_de_cada_lado_e_de_FEATURES_e_os_dois_DIFEREM() -> None:  # noqa: N802
    """R3: *"com o `n` de cada lado declarado como **features**"* — e nunca de injeções.

    **O número errado está ao alcance da mão de propósito**: `n_injecoes_*` viaja no mesmo objeto, e
    é o número errado mais barato de usar. A âncora que separa as duas leituras é que **nenhuma**
    das duas contagens de injeção coincide com a contagem de features de lado nenhum.
    """
    par = par_do_rsd_de_prova()
    assert par.qcs.n != par.amostras.n, (
        f"âncora: os dois lados têm {par.qcs.n} features cada, e «o `n` de cada lado» passaria com "
        "a figura a escrever o mesmo número duas vezes"
    )
    assert par.qcs.n not in (N_INJECOES_DOS_QCS, N_INJECOES_DAS_AMOSTRAS)
    assert par.amostras.n not in (N_INJECOES_DOS_QCS, N_INJECOES_DAS_AMOSTRAS)

    titulo = texto_corrido(eixo_do_violino(violino_de_prova(par)).get_title())
    assert f"{milhar_do_teste(par.qcs.n)} features" in titulo, titulo
    assert f"{milhar_do_teste(par.amostras.n)} features" in titulo, titulo
    assert "injeç" not in titulo, (
        f"o título fala de injeções: {titulo!r}. O `n` de R3 é de FEATURES, e o corpo traz a "
        "contagem de injeções ao lado justamente para o número errado estar à mão"
    )


def test_o_titulo_deriva_do_VETOR_e_o_rodape_do_campo_de_contagem() -> None:  # noqa: N802
    """As duas fontes do mesmo número, e o par causal mostra que a figura não escolhe em silêncio.

    O título deriva do vetor **desenhado** — é a população da curva que está ali — e o rodapé do
    campo `n_features_finitas_*`, que é o que o servidor diz ter calculado. Num corpo em que as duas
    divergem, a imagem mostra **as duas**, e é o desfecho certo: escolher uma faria a figura afirmar
    sobre uma população o que mediu de outra.
    """
    par = par_do_rsd_de_prova()
    assert par.qcs.n == par.qcs.n_features_finitas, (
        "âncora: a fixtura já nasce com as duas fontes divergentes, e o par abaixo não separaria"
    )
    rodape = texto_corrido(violino_de_prova(par).texts[-1].get_text())
    assert (
        f"{milhar_do_teste(par.qcs.n_features_finitas)} de "
        f"{milhar_do_teste(N_FEATURES_DO_RSD)} features"
    ) in rodape, rodape
    assert f"{milhar_do_teste(par.amostras.n_features_finitas)} de " in rodape, rodape

    divergente = par_do_rsd_de_prova(qcs=lado_do_rsd_de_prova(n_features_finitas=99))
    titulo = texto_corrido(eixo_do_violino(violino_de_prova(divergente)).get_title())
    outro_rodape = texto_corrido(violino_de_prova(divergente).texts[-1].get_text())
    assert f"{milhar_do_teste(N_FINITAS_DOS_QCS)} features" in titulo, titulo
    assert "99 de " in outro_rodape, outro_rodape
    assert rodape != outro_rodape


def test_o_contorno_e_LINE2D_legivel_e_ax_violinplot_seria_CEGO() -> None:  # noqa: N802
    """**A cegueira do arnês ao `ax.violinplot`, reproduzida como asserção.**

    A metade de baixo é a própria medição a correr: com `ax.violinplot`, trocar o conteúdo
    **inteiro** deixa a leitura do arnês idêntica, porque `FillBetweenPolyCollection` não tem
    `get_segments` e `series` cai em `get_offsets()`, que devolve o par nulo da origem. A metade de
    cima afirma que o contorno em `Line2D` **não** tem essa cegueira.

    Sem a metade de baixo, *"o contorno é legível"* seria uma afirmação sobre a nossa figura e não
    uma razão para não usar a função da biblioteca.
    """
    eixo = eixo_do_violino(violino_de_prova())
    desenhadas = {s.rotulo for s in series(eixo)}
    assert rotulo_do_rsd(ROTULO_DOS_QCS, PAPEL_DA_DENSIDADE) in desenhadas, sorted(desenhadas)
    assert not eixo.collections, (
        f"o violino desenhou {len(eixo.collections)} coleção(ões): `ax.violinplot` produz "
        "`FillBetweenPolyCollection`, que o arnês lê como um ponto na origem (KTD6)"
    )

    def lido_pelo_violinplot(dados: list[float]) -> list[tuple[str, list[float], list[float]]]:
        nua = nova_figura()
        eixo_nu = nua.add_subplot(111)
        eixo_nu.violinplot([dados], positions=[0], showextrema=False)
        return [(s.rotulo, s.x.tolist(), s.y.tolist()) for s in series(eixo_nu)]

    dos_qcs = valores_do_rsd(N_FINITAS_DOS_QCS, centro=0.12, semente=11)
    das_amostras = valores_do_rsd(N_FINITAS_DAS_AMOSTRAS, centro=0.35, semente=23)
    assert abs(float(np.mean(dos_qcs)) - float(np.mean(das_amostras))) > 0.1, (
        "âncora: os dois vetores têm médias parecidas, e «trocar o conteúdo inteiro» não seria o "
        "que a medição de KTD6 trocou"
    )
    assert lido_pelo_violinplot(dos_qcs) == lido_pelo_violinplot(das_amostras), (
        "`ax.violinplot` passou a devolver leituras distintas para conteúdos distintos: a medição "
        "de KTD6 caducou neste `matplotlib`, e a decisão precisa de ser remedida"
    )


def test_trocar_o_vetor_de_um_lado_MUDA_o_que_o_arnes_le() -> None:  # noqa: N802
    """O par causal que a medição acima torna necessário.

    Trocar o vetor de um lado muda o dado que o arnês lê. A âncora de não-vacuidade vem antes: sem
    série nenhuma, *"as leituras divergem"* seria uma comparação entre dois dicionários vazios.
    """
    direto = eixo_do_violino(violino_de_prova())
    trocado = eixo_do_violino(
        violino_de_prova(
            par_do_rsd_de_prova(
                qcs=lado_do_rsd_de_prova(
                    ROTULO_DOS_QCS, quantos=N_FINITAS_DAS_AMOSTRAS, centro=0.35, semente=23
                )
            )
        )
    )
    contorno = rotulo_do_rsd(ROTULO_DOS_QCS, PAPEL_DA_DENSIDADE)
    do_direto = {s.rotulo: s.y.tolist() for s in series(direto)}
    do_trocado = {s.rotulo: s.y.tolist() for s in series(trocado)}
    assert do_direto and contorno in do_direto, sorted(do_direto)
    assert do_direto[contorno] != do_trocado[contorno], (
        "trocar o vetor inteiro de um lado não mudou o contorno que o arnês lê — é exactamente a "
        "cegueira de KTD6, e o teste estaria a medir nada"
    )


def test_a_caixa_de_quartis_sai_do_MESMO_vetor_que_o_contorno() -> None:  # noqa: N802
    """Os quartis são derivados no teste, por caminho independente do construtor.

    **A caixa e o contorno descrevem a mesma população por construção**, e é isso que se afirma: um
    construtor que estimasse a densidade de um vetor e os quartis de outro produziria uma figura bem
    formada com a caixa fora do corpo — o rótulo certo sobre a quantidade errada, dentro da mesma
    imagem.
    """
    par = par_do_rsd_de_prova()
    eixo = eixo_do_violino(violino_de_prova(par))
    desenhadas = {s.rotulo: s for s in series(eixo)}

    esperados = np.quantile(np.asarray(par.qcs.valores, dtype=float) * 100.0, [0.25, 0.5, 0.75])
    caixa = desenhadas[rotulo_do_rsd(ROTULO_DOS_QCS, PAPEL_DOS_QUARTIS)]
    mediana = desenhadas[rotulo_do_rsd(ROTULO_DOS_QCS, PAPEL_DA_MEDIANA)]
    assert float(esperados[0]) < float(esperados[2]), "âncora: os quartis coincidem"

    assert np.isclose(caixa.y.min(), esperados[0]), (caixa.y.min(), esperados[0])
    assert np.isclose(caixa.y.max(), esperados[2]), (caixa.y.max(), esperados[2])
    assert np.allclose(mediana.y, esperados[1]), (mediana.y, esperados[1])

    contorno = desenhadas[rotulo_do_rsd(ROTULO_DOS_QCS, PAPEL_DA_DENSIDADE)]
    assert contorno.y.min() <= esperados[0] and contorno.y.max() >= esperados[2], (
        "a caixa de quartis caiu FORA do contorno: os dois não descrevem a mesma população"
    )


def test_o_eixo_do_rsd_esta_em_PERCENTAGEM_e_o_dado_e_a_fracao_vezes_cem() -> None:  # noqa: N802
    """`metrics/rsd.py` devolve `sd/mean`, que é **fração** — escrever `0,1399` como `13,99%` é
    rótulo igual com quantidade diferente, e a conversão acontece **uma** vez, aqui no desenho.

    **O par causal é entre o vetor do fio e o que está desenhado**, e é ele que separa «converteu»
    de «rotulou como se tivesse convertido»: a mediana desenhada tem de ser cem vezes a mediana do
    vetor, e o eixo tem de dizer `%`.
    """
    par = par_do_rsd_de_prova()
    eixo = eixo_do_violino(violino_de_prova(par))
    assert eixo.get_ylabel() == ROTULO_DO_EIXO_DO_RSD
    assert "%" in ROTULO_DO_EIXO_DO_RSD, ROTULO_DO_EIXO_DO_RSD

    do_fio = float(np.median(np.asarray(par.qcs.valores, dtype=float)))
    assert 0.0 < do_fio < 1.0, f"âncora: a mediana do fio é {do_fio} e já não é uma fração"
    desenhada = {s.rotulo: s for s in series(eixo)}[rotulo_do_rsd(ROTULO_DOS_QCS, PAPEL_DA_MEDIANA)]
    assert np.allclose(desenhada.y, do_fio * 100.0), (desenhada.y, do_fio * 100.0)
    assert em_percentagem(0.1399) == "13,99%"
    assert em_percentagem(1.9e-15) == "1,9e-13%"


def test_com_um_dos_lados_VAZIO_a_figura_declara_a_ausencia_e_desenha_o_OUTRO() -> None:  # noqa: N802
    """R7: insumo ausente é achado válido, **nunca** espaço em branco.

    **O par causal separa**: com os dois lados não há frase de ausência e há seis artistas; com um
    vazio há a frase e só os artistas do lado que sobrou. Sem a metade de cima, *"a figura declara a
    ausência"* passaria com um construtor que a escrevesse sempre.
    """
    cheio = violino_de_prova()
    vazio = violino_de_prova(
        par_do_rsd_de_prova(qcs=lado_do_rsd_de_prova(valores=[], n_injecoes=0))
    )
    do_cheio = {s.rotulo for s in series(eixo_do_violino(cheio))}
    do_vazio = {s.rotulo for s in series(eixo_do_violino(vazio))}
    assert len(do_cheio) == 6, sorted(do_cheio)
    assert len(do_vazio) == 3, sorted(do_vazio)
    assert all(r.startswith(ROTULO_DAS_AMOSTRAS) for r in do_vazio), sorted(do_vazio)

    assert declarados_no_eixo(cheio) == [], (
        f"a figura cheia já escreve texto declarado: {declarados_no_eixo(cheio)}. A planta não "
        "separa nada"
    )
    assert declarados_no_eixo(vazio), (
        "o lado vazio não foi declarado: a figura fica com metade em branco, e branco lê-se como "
        "figura quebrada"
    )


def test_zero_injecoes_e_zero_FEATURES_definidas_sao_fatos_diferentes() -> None:  # noqa: N802
    """É a distinção que `RsdPorFeature` carrega `n_injecoes_*` para poder fazer.

    *"Não há QC nenhum nesta matriz"* e *"há sete QCs e nenhuma feature com RSD definido"* dão os
    dois `n == 0` e pedem frases diferentes. Um motivo único tornaria-os indistinguíveis em
    contagem — a distinção que fez `unassessable` existir, uma camada acima.
    """
    sem_injecao = lado_do_rsd_de_prova(valores=[], n_injecoes=0)
    sem_feature = lado_do_rsd_de_prova(valores=[], n_injecoes=N_INJECOES_DOS_QCS)
    assert not sem_injecao.desenhavel and not sem_feature.desenhavel
    assert frase_do_lado_ausente(sem_injecao) != frase_do_lado_ausente(sem_feature)
    assert "nenhuma injeção" in frase_do_lado_ausente(sem_injecao)
    assert str(N_INJECOES_DOS_QCS) in frase_do_lado_ausente(sem_feature)

    das_figuras = [
        declarados_no_eixo(violino_de_prova(par_do_rsd_de_prova(qcs=lado)))
        for lado in (sem_injecao, sem_feature)
    ]
    assert das_figuras[0] and das_figuras[1], das_figuras
    assert das_figuras[0] != das_figuras[1], das_figuras


def test_os_DOIS_lados_vazios_recusam_por_populacao_vazia() -> None:  # noqa: N802
    """`MatrizVaziaError` é a família que `PainelDeFigura._troca_a_figura` apanha.

    Um eixo montado com duas frases e nenhum artista lê-se como figura quebrada, e o texto que o
    painel põe no lugar diz o que é verdade: não houve população nenhuma a desenhar.
    """
    par = par_do_rsd_de_prova(
        qcs=lado_do_rsd_de_prova(valores=[]),
        amostras=lado_das_amostras_de_prova(valores=[]),
    )
    with pytest.raises(MatrizVaziaError, match="nenhum dos dois lados"):
        violino_de_prova(par)


def test_um_lado_SEM_DISPERSAO_perde_o_contorno_e_mantem_a_caixa() -> None:  # noqa: N802
    """O terceiro desfecho por lado, e a verificação pede as três direções e não duas.

    `gaussian_kde` levanta `LinAlgError` de dentro da álgebra linear quando todos os valores são
    iguais — a matriz de covariância é singular. Recusar a figura inteira por causa de um lado
    degenerado apagaria o outro, que está bem; desenhar uma reta afirmaria distribuição uniforme
    sobre uma população que não tem distribuição nenhuma. O que fica é a caixa, que a mediana e os
    quartis de um valor só descrevem sem inventar nada.
    """
    constante = lado_do_rsd_de_prova(valores=[0.2] * 12)
    assert constante.desenhavel and not constante.tem_densidade
    figura = violino_de_prova(par_do_rsd_de_prova(qcs=constante))
    desenhadas = {s.rotulo for s in series(eixo_do_violino(figura))}

    assert rotulo_do_rsd(ROTULO_DOS_QCS, PAPEL_DOS_QUARTIS) in desenhadas, sorted(desenhadas)
    assert rotulo_do_rsd(ROTULO_DOS_QCS, PAPEL_DA_DENSIDADE) not in desenhadas, sorted(desenhadas)
    assert rotulo_do_rsd(ROTULO_DAS_AMOSTRAS, PAPEL_DA_DENSIDADE) in desenhadas, (
        "o lado bom perdeu o contorno por causa do lado degenerado"
    )
    assert any("valor(es) distinto(s)" in t for t in declarados_no_eixo(figura)), (
        declarados_no_eixo(figura)
    )


def test_a_legenda_distingue_os_dois_lados_por_rotulos_UNICOS() -> None:  # noqa: N802
    """`confere_series` indexa por rótulo, e o último ganha.

    **A âncora de não-vacuidade vem primeiro**: um conjunto vazio de rótulos satisfaz
    trivialmente *"não há repetidos"*.
    """
    eixo = eixo_do_violino(violino_de_prova())
    rotulos_dos_artistas = [s.rotulo for s in series(eixo)]
    assert len(rotulos_dos_artistas) == 6, rotulos_dos_artistas
    assert len(set(rotulos_dos_artistas)) == len(rotulos_dos_artistas), rotulos_dos_artistas

    legenda = eixo.get_legend()
    assert legenda is not None, "o violino não tem legenda: os dois lados só se distinguem por cor"
    textos = [t.get_text() for t in legenda.get_texts()]
    assert textos == [ROTULO_DOS_QCS, ROTULO_DAS_AMOSTRAS], textos
    assert len(set(textos)) == len(textos), textos


def test_o_rodape_declara_o_ESTIMADOR_a_variante_e_o_corretor() -> None:  # noqa: N802
    """R7 e a ressalva da docstring do modelo.

    O lado dos QCs é medido **in-sample** sobre os próprios pontos que um corretor que ajusta modelo
    usou — a mediana in-sample do `batch_loess` é 1,9e-15 no `pos_c18`. Esta figura não ordena
    corretores, e é por isso que o rodapé tem de dizer com que régua o lado dos QCs foi medido: lida
    ao lado da fila de barras, a imagem pode ser lida como *"a correção funcionou"*.

    **O par causal separa**: trocar o estimador muda o rodapé.
    """
    rodape = recorte_do_violino_do_rsd_por_feature(par=par_do_rsd_de_prova()).texto()
    assert "in_sample" in rodape, rodape
    assert "outlier_removed" in rodape, rodape
    assert "loess" in rodape and "corrigida" in rodape, rodape
    assert "NÃO ordena corretores" in rodape, rodape

    outro = recorte_do_violino_do_rsd_por_feature(
        par=par_do_rsd_de_prova(estimador="leave_one_qc_out")
    ).texto()
    assert "in_sample" not in outro and "leave_one_qc_out" in outro, outro
    assert rodape != outro


def test_d567_na_INGERIDA_o_rodape_diz_antes_de_qualquer_correcao_e_nao_nomeia_corretor() -> None:  # noqa: N802
    """Antes do primeiro corretor a janela desenha o violino da matriz ingerida, e o corpo do
    diagnóstico traz `corretor` nulo — `none` afirmaria uma correção que não houve.

    **Três casos, e o terceiro é um caso à parte:** na corrigida o rodapé nomeia o corretor; na
    ingerida diz *«antes de qualquer correção»* e não escreve o recuo; e uma corrigida SEM corretor
    continua a escrever o recuo — ali a ausência é defeito do corpo, e a imagem tem de o mostrar.
    """
    corrigida = recorte_do_violino_do_rsd_por_feature(par=par_do_rsd_de_prova()).matriz
    ingerida = recorte_do_violino_do_rsd_por_feature(
        par=par_do_rsd_de_prova(matriz=MATRIZ_DA_LEITURA, corretor=None)
    ).matriz
    corrigida_sem_corretor = recorte_do_violino_do_rsd_por_feature(
        par=par_do_rsd_de_prova(corretor=None)
    ).matriz

    assert corrigida == "uma — «corrigida», do corretor «loess»", corrigida
    assert ingerida == "uma — «ingerida», antes de qualquer correção", ingerida
    assert "corretor" not in ingerida, ingerida
    assert CORRETOR_NAO_DECLARADO in corrigida_sem_corretor, corrigida_sem_corretor


def test_nenhum_numero_da_figura_vem_das_INJECOES() -> None:  # noqa: N802
    """**O número errado ao alcance da mão**, e é por isso que o corpo o traz.

    A planta são dois pares idênticos **menos** as duas contagens de injeção, e a asserção é que o
    que a figura declara sai igual. A âncora de não-vacuidade vem primeiro: uma figura que não
    declarasse número nenhum satisfaria a igualdade sobre duas listas vazias.
    """
    poucas = par_do_rsd_de_prova()
    muitas = par_do_rsd_de_prova(
        qcs=lado_do_rsd_de_prova(n_injecoes=999),
        amostras=lado_das_amostras_de_prova(n_injecoes=888),
    )
    assert poucas.qcs.n_injecoes != muitas.qcs.n_injecoes, "âncora: a planta não aterrou"

    das_poucas = numeros_declarados(violino_de_prova(poucas))
    das_muitas = numeros_declarados(violino_de_prova(muitas))
    assert das_poucas, "âncora: a figura não declara número nenhum"
    assert das_poucas == das_muitas, (
        f"trocar a contagem de injeções mudou o que a figura declara: {das_poucas} contra "
        f"{das_muitas}. O `n` de R3 é de FEATURES"
    )
    assert "999" not in das_poucas and "888" not in das_muitas


def test_o_violino_limpo_passa_na_classe_c_e_as_plantas_acusam() -> None:
    """A classe (c) sobre o eixo, e as duas plantas que ela e a (a) apanham.

    **A planta entra como argumento**, então o negativo percorre exactamente o caminho que o portão
    percorre — negativo em processo não consegue ficar inerte.
    """
    eixo = eixo_do_violino(violino_de_prova())
    assert series(eixo), "o violino não tem artista de dado nenhum, e a asserção abaixo seria vácua"
    assert confere_vista(eixo) == [], eixo.get_title()

    cortado = eixo_do_violino(violino_de_prova())
    planta_corte_de_eixo(cortado)
    assert confere_vista(cortado) != []

    trocado = eixo_do_violino(violino_de_prova())
    esperadas = {s.rotulo: (s.x.tolist(), s.y.tolist()) for s in series(trocado)}
    planta_eixos_trocados(trocado)
    assert confere_series(trocado, esperadas) != []


def test_nenhum_eixo_do_violino_e_logaritmico() -> None:
    """A tentação é real — o RSD in-sample do `batch_loess` vai a 1,9e-15 —, e a resposta é o
    contorno sobre a população inteira, não a escala."""
    figura = violino_de_prova()
    assert len(figura.axes) == 1, [a.get_title() for a in figura.axes]
    assert (figura.axes[0].get_xscale(), figura.axes[0].get_yscale()) == ("linear", "linear")
    assert texto_corrido(sem_a_matriz(figura.texts[0].get_text())) == TITULO_DO_VIOLINO
    assert confere_a_matriz_do_titulo(figura) == []


@pytest.mark.parametrize(
    ("largura", "altura", "ppp"),
    [(10.0, 6.0, 100.0), (8.0, 4.5, 96.0), (6.5, 3.4, 96.0), (12.0, 7.0, 150.0)],
)
def test_o_rodape_do_violino_cabe_nos_tamanhos_EXTREMOS_sem_levantar(  # noqa: N802
    largura: float, altura: float, ppp: float
) -> None:
    """Os quatro tamanhos, com o rodapé que esta figura de facto produz."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure as FiguraNua

    from refino.plots.recorte import ENTRELINHA, MARGEM, RESERVA_MAXIMA, TAMANHO_DA_FONTE

    texto = recorte_do_violino_do_rsd_por_feature(par=par_do_rsd_de_prova()).texto()
    figura = FiguraNua(figsize=(largura, altura), dpi=ppp)
    FigureCanvasAgg(figura)
    linhas = linhas_do_rodape(figura, texto)
    reservado = len(linhas) * TAMANHO_DA_FONTE * ENTRELINHA / (altura * 72.0) + 2.0 * MARGEM
    assert reservado <= RESERVA_MAXIMA, (
        f"a {largura}x{altura}@{ppp} o rodapé quebra em {len(linhas)} linhas e precisaria de "
        f"{reservado:.1%} da altura, acima do teto de {RESERVA_MAXIMA:.0%}"
    )


def test_um_denominador_ausente_no_violino_RECUSA_por_populacao_vazia() -> None:  # noqa: N802
    """Sem execução não há população, e `MatrizVaziaError` é o que o painel apanha."""
    with pytest.raises(MatrizVaziaError, match="não declara de quantas features"):
        recorte_do_violino_do_rsd_por_feature(par=par_do_rsd_de_prova(n_features=0))


def test_um_NAO_FINITO_no_vetor_sai_da_curva_e_e_CONTADO_na_imagem() -> None:  # noqa: N802
    """`RsdPorFeature` promete *"só com os valores finitos"*, e o construtor não confia na promessa.

    **As duas direções**: com o contrato cumprido a frase não tem a cláusula; com ele
    quebrado a cláusula aparece e o `n` cai. Sem a metade de cima, *"a cláusula aparece"* passaria
    com um construtor que a escrevesse sempre; sem a de baixo, o ramo ficava construído e nunca
    exercitado — e um servidor que mandasse `null` faria a curva descrever uma população menor do
    que a que a figura declara, com a imagem bem formada.
    """
    valores = valores_do_rsd(20, centro=0.12, semente=5)
    limpo = lado_do_rsd_de_prova(valores=valores)
    sujo = lado_do_rsd_de_prova(valores=[*valores, None, float("nan")])

    assert limpo.n_nao_finitos == 0 and sujo.n_nao_finitos == 2, (limpo, sujo)
    assert limpo.n == sujo.n == 20, "âncora: o não finito entrou na curva em vez de sair dela"

    sem_clausula = frase_do_lado_do_rsd(limpo)
    com_clausula = frase_do_lado_do_rsd(sujo)
    assert "não finito" not in sem_clausula, sem_clausula
    assert "2 não finito(s) fora da estimativa" in com_clausula, com_clausula
    assert sem_clausula != com_clausula

    titulo = texto_corrido(
        eixo_do_violino(violino_de_prova(par_do_rsd_de_prova(qcs=sujo))).get_title()
    )
    assert "não finito(s) fora da estimativa" in titulo, titulo


def test_os_QUATRO_rotulos_ausentes_do_corpo_nao_viram_nome_de_coisa_nenhuma() -> None:  # noqa: N802
    """Na ponta do desenho, campo que não veio não pode sair como um valor do vocabulário.

    **`none` é corretor legítimo**, e `in_sample` é estimador legítimo: um recuo escolhido entre os
    plausíveis faria a figura afirmar procedência que ninguém declarou. `REGUA_AUSENTE` também não
    é `NAO_DECLARADA`, que é a cadeia *"matriz não declarada"* — escrita dentro de
    `estimador «...»` ela poria uma frase sobre a **matriz** no lugar do rótulo da régua.

    **O par causal separa**: com os campos o rodapé nomeia-os; sem eles nomeia a ausência.
    """
    com = recorte_do_violino_do_rsd_por_feature(par=par_do_rsd_de_prova()).texto()
    sem = recorte_do_violino_do_rsd_por_feature(
        par=par_do_rsd(
            qcs=lado_do_rsd_de_prova(),
            amostras=lado_das_amostras_de_prova(),
            matriz=None,
            corretor=None,
            variante=None,
            estimador=None,
            n_features=N_FEATURES_DO_RSD,
        )
    ).texto()
    assert com != sem
    assert REGUA_AUSENTE in sem and REGUA_AUSENTE not in com, sem
    assert CORRETOR_NAO_DECLARADO in sem, sem
    assert NAO_DECLARADA in sem, sem
    # A âncora de não-vacuidade vem ANTES da asserção de ausência, e na mesma função:
    # um vocabulário vazio satisfaz *"o recuo não é um nome de corretor"* sobre coisa nenhuma.
    assert VOCABULARIO_DOS_CORRETORES, "o vocabulário de corretores está vazio"
    assert CORRETOR_NAO_DECLARADO not in VOCABULARIO_DOS_CORRETORES, CORRETOR_NAO_DECLARADO
    assert REGUA_AUSENTE not in {"in_sample", "leave_one_qc_out", "plain", "outlier_removed"}, (
        f"o recuo {REGUA_AUSENTE!r} é um rótulo de régua legítimo: a ausência do campo passa a ser "
        "indistinguível de uma medição real com aquela régua"
    )
    assert NAO_DECLARADA != REGUA_AUSENTE, (
        "o recuo da matriz e o da régua voltaram a ser a mesma cadeia: um campo de régua que não "
        "veio sairia da figura como uma frase sobre a matriz"
    )


def test_o_violino_e_o_UNICO_dono_da_chave_item_10() -> None:  # noqa: N802
    """A figura C entra como `Vinculo.RELATORIO` com a chave do item que a emenda criou.

    **Contar e não perguntar se existe**: `"item-10" in {...}` devolve o desfecho de sucesso
    com duas entradas iguais, e duas fontes para o mesmo desenho é o que se recusa.
    """
    donos = [c for c in CONSTRUTORES if c.chave == "item-10"]
    assert len(donos) == 1, [c.funcao for c in donos]
    assert donos[0].modulo == "dispersao"
    assert donos[0].funcao == "figura_do_violino_do_rsd_por_feature"
    assert donos[0].vinculo is Vinculo.RELATORIO

    item = next(i for i in ITENS_OBRIGATORIOS if i.numero == 10)
    assert "RSD por feature" in item.titulo, item.titulo
    assert "n de cada lado" in item.titulo, item.titulo
