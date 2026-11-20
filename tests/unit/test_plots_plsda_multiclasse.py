"""As três figuras do PLS-DA multiclasse, e a regra 6 do README na GRELHA.

**Não é `test_plsda_multiclasse.py`, que já existe em `tests/unit/` e é do cálculo.** Aquele prova
que `OneVsRestModel` não admite construção parcial; este prova que o **desenho** herda a mesma
propriedade, e que a coleção que o desenho recusa colapsar continua sem colapsar na imagem.

## O que este arquivo prova, e por que a primeira seção é a mais importante

A regra 6 do README diz que não pode existir caminho de código que retorne scores de PLS-DA sem
`q2` e `permutation_pvalue`. Com K modelos ela ganha uma exigência a mais: **o par tem de ser por
modelo**. Um `Q²` único no alto de uma grelha de quatro painéis seria um agregado que ninguém
calculou, e o leitor atribuí-lo-ia ao painel que estivesse a olhar.

As três metades vivem no mesmo arquivo de propósito: um teste vizinho que ligue as pontas
pode ser removido sem que nada ligue os dois.

1. as três cargas de desenho exigem `Validacao` sem padrão — o portão é sobre o **módulo**, não
   sobre as três que eu escrevi;
2. cada painel **imprime** o par do modelo dele, e o oráculo é o texto DESENHADO;
3. o rodapé lista os **K** pares, um por classe, e a contagem de vereditos não colapsa nenhum.
"""

from __future__ import annotations

import dataclasses
import inspect
import re

import numpy as np
import pytest
from matplotlib.colors import to_hex
from matplotlib.figure import Figure

from refino.domain.matrix import RefinoMatrix
from refino.domain.metadata import SampleMetadata
from refino.plots import plsda_multiclasse as modulo
from refino.plots.base import MatrizVaziaError, cores_do_ciclo
from refino.plots.catalogo import CONSTRUTORES, Vinculo, rotulo_do_construtor
from refino.plots.plsda import PlsdaDesalinhadoError
from refino.plots.plsda_multiclasse import (
    COLUNAS_DA_GRELHA,
    FEATURES_DA_IMPORTANCIA,
    MIN_MODELOS,
    TITULO_DA_IMPORTANCIA_MULTICLASSE,
    TITULO_DAS_CARGAS_MULTICLASSE,
    TITULO_DOS_ESCORES_MULTICLASSE,
    CargasDoModelo,
    EscoresDoModelo,
    ImportanciaDoModelo,
    cargas_do_modelo,
    escores_do_modelo,
    figura_da_importancia_um_contra_resto,
    figura_das_cargas_um_contra_resto,
    figura_dos_escores_um_contra_resto,
    frase_da_importancia_indisponivel,
    frase_dos_pares,
    frase_dos_vereditos,
    importancia_do_modelo,
    recorte_da_importancia_um_contra_resto,
    recorte_das_cargas_um_contra_resto,
    recorte_dos_escores_um_contra_resto,
    rotulo_do_painel,
    selecao_da_importancia,
    titulo_do_painel,
)
from refino.plots.recorte import decimal, milhar
from refino.plots.series import EstiloDoGrupo, estilos_dos_grupos, ordem_canonica
from refino.plots.validacao import Validacao
from refino.statistics.plsda_multiclasse import MulticlassPLSDAResult, plsda_multiclass
from tests.arnes_de_figura import (
    confere_series,
    numeros_declarados,
    planta_eixos_trocados,
    planta_serie_removida,
    series,
)
from tests.conftest import matrix_from_toserrf

MODO = "pos_hilic"

# As quatro classes da coorte deste estudo. **É OBSERVAÇÃO sobre os quatro arquivos e não
# invariante**: outro estudo traz outros grupos, em qualquer número, e é exactamente isso
# que `plots/series.py` existe para suportar. Aqui servem de forma plausível.
CLASSES = ("ATLL", "Asymptomatic", "HAMTSP", "NoHTLV")


# ---------------------------------------------------------------------------
# As fixturas
# ---------------------------------------------------------------------------


def validacao_de(indice: int) -> Validacao:
    """Uma validação por modelo, com `q2` e `p` DIFERENTES entre eles.

    **Diferentes de propósito, e é a âncora aplicada às frases da coleção.** Com os K
    valores iguais, `frase_dos_pares` sairia a mesma com e sem colapso, e um teste que a lesse
    passaria sobre um agregado. A distinção entre modelos só é observável quando os números diferem.
    """
    return Validacao(
        q2=0.4 - 0.25 * indice,
        permutation_pvalue=0.01 + 0.3 * indice,
        r2y=0.9 - 0.05 * indice,
        n_permutacoes=200,
        cv_folds=5,
        n_componentes=2,
    )


def matriz_sintetica(*, por_grupo: int = 8, n_features: int = 9) -> RefinoMatrix:
    """Quatro classes com um deslocamento CONHECIDO em três features por classe."""
    gerador = np.random.default_rng(20260914)
    linhas = por_grupo * len(CLASSES)
    valores = gerador.normal(10.0, 1.0, size=(linhas, n_features))
    for indice in range(len(CLASSES)):
        fatia = slice(indice * por_grupo, (indice + 1) * por_grupo)
        valores[fatia, indice % n_features] += 4.0
    amostras = tuple(
        SampleMetadata(
            name=f"a{i:02d}",
            injection_order=i + 1,
            sample_type="sample",
            group=CLASSES[i // por_grupo],
        )
        for i in range(linhas)
    )
    return RefinoMatrix(
        values=valores,
        feature_names=tuple(f"f{j:02d}" for j in range(n_features)),
        samples=amostras,
        mode=MODO,
    )


def grupos_por_amostra(matriz: RefinoMatrix, resultado: MulticlassPLSDAResult) -> list[str]:
    """O vetor de grupo das amostras que entraram nos modelos, na ordem das linhas.

    **Pela máscara e não pelo nome** — é a mesma escolha de `statistics.plsda_multiclasse`: o QC é
    excluído por `sample_mask`, e comparar contra a string `"QC"` casaria por acidente de escrita.
    """
    return [
        a.group
        for a, entra in zip(matriz.samples, matriz.sample_mask, strict=True)
        if entra and a.group in resultado.groups
    ]


def cargas_de_escores(
    matriz: RefinoMatrix, resultado: MulticlassPLSDAResult
) -> list[EscoresDoModelo]:
    vetor = grupos_por_amostra(matriz, resultado)
    return [
        escores_do_modelo(
            grupo=m.group,
            comp1=m.scores[:, 0],
            comp2=m.scores[:, 1],
            grupos=list(vetor),
            validacao=validacao_de(i),
        )
        for i, m in enumerate(resultado.models)
    ]


def cargas_de_cargas(
    matriz: RefinoMatrix, resultado: MulticlassPLSDAResult
) -> list[CargasDoModelo]:
    return [
        cargas_do_modelo(
            grupo=m.group,
            comp1=m.loadings[:, 0],
            comp2=m.loadings[:, 1],
            features=list(matriz.feature_names),
            validacao=validacao_de(i),
        )
        for i, m in enumerate(resultado.models)
    ]


def cargas_de_importancia(
    matriz: RefinoMatrix, resultado: MulticlassPLSDAResult
) -> list[ImportanciaDoModelo]:
    return [
        importancia_do_modelo(
            grupo=m.group,
            vip=m.vip,
            features=list(matriz.feature_names),
            validacao=validacao_de(i),
        )
        for i, m in enumerate(resultado.models)
    ]


@pytest.fixture(scope="module")
def sintetico() -> dict[str, object]:
    """Quatro classes sintéticas, com as três coleções de desenho prontas."""
    matriz = matriz_sintetica()
    resultado = plsda_multiclass(
        matriz, n_components=2, cv_folds=4, n_permutations=10, seed=20260914
    )
    return {
        "matriz": matriz,
        "resultado": resultado,
        "escores": cargas_de_escores(matriz, resultado),
        "cargas": cargas_de_cargas(matriz, resultado),
        "importancia": cargas_de_importancia(matriz, resultado),
    }


@pytest.fixture(scope="module")
def real() -> dict[str, object]:
    """O `pos_hilic` do laboratório, com as três coleções.

    **Escopo de módulo, e a regra 1 do README é o que decide a ausência:** sem o arquivo de
    referência `matrix_from_toserrf` faz `skip` com mensagem pelos resolvedores do `conftest`, e
    nada é fabricado no lugar dele.
    """
    matriz = matrix_from_toserrf(MODO)
    resultado = plsda_multiclass(matriz, n_components=2, cv_folds=7, n_permutations=10, seed=42)
    return {
        "matriz": matriz,
        "resultado": resultado,
        "escores": cargas_de_escores(matriz, resultado),
        "cargas": cargas_de_cargas(matriz, resultado),
        "importancia": cargas_de_importancia(matriz, resultado),
    }


def rodape_de(figura: Figure) -> str:
    """O texto do rodapé desenhado, com as quebras colapsadas — é o que o leitor vê."""
    return " ".join(figura.texts[-1].get_text().split())


def titulos_de(figura: Figure) -> list[str]:
    """Os títulos dos eixos que TÊM título, com as quebras colapsadas."""
    return [" ".join(e.get_title().split()) for e in figura.axes if e.get_title().strip()]


# ---------------------------------------------------------------------------
# 1. A REGRA 6 DO README — e o portão é sobre o MÓDULO, não sobre as três que escrevi
# ---------------------------------------------------------------------------


def test_todas_as_cargas_do_desenho_EXIGEM_validacao_sem_padrao() -> None:  # noqa: N802
    """**Total sobre o módulo, e é essa a diferença.**

    Não é *"as três que eu escrevi têm o campo"* — é *"todo dataclass de carga deste módulo tem"*.
    Uma quarta carga escrita amanhã sem `validacao` reprova aqui, que é a distância entre um portão
    sobre a propriedade e um sobre a instância (a segunda pergunta).
    """
    cargas = [
        (nome, cls)
        for nome, cls in inspect.getmembers(modulo, inspect.isclass)
        if dataclasses.is_dataclass(cls) and cls.__module__ == modulo.__name__
    ]
    assert {n for n, _ in cargas} == {
        "CargasDoModelo",
        "EscoresDoModelo",
        "ImportanciaDoModelo",
    }, cargas
    for nome, cls in cargas:
        campos = {c.name: c for c in dataclasses.fields(cls)}
        assert "validacao" in campos, f"{nome} carrega dado de PLS-DA e não exige `Validacao`"
        assert campos["validacao"].default is dataclasses.MISSING, (
            f"{nome}.validacao ganhou valor padrão — passa a existir construção sem o par"
        )
        assert "grupo" in campos, f"{nome} não diz de que modelo é: o painel ficaria sem título"
        assert cls.__dataclass_params__.frozen  # type: ignore[attr-defined]


def test_score_sem_o_par_e_IMPOSSIVEL_e_a_recusa_e_tipada() -> None:  # noqa: N802
    """As duas portas: o construtor de fronteira e o dataclass.

    **Par causal em processo**: a planta é o argumento, e não há intermediário que possa não
    a receber. A metade positiva vem logo a seguir — com o par, os dois constroem.
    """
    with pytest.raises(TypeError):
        escores_do_modelo(  # type: ignore[call-arg]
            grupo="A", comp1=[1.0, 2.0], comp2=[3.0, 4.0], grupos=["A", "B"]
        )
    with pytest.raises(TypeError):
        EscoresDoModelo(  # type: ignore[call-arg]
            grupo="A", x=np.zeros(2), y=np.zeros(2), grupos=("A", "B")
        )

    # A outra direção: com a validação, os dois constroem. Sem esta metade o teste acima passaria
    # com um construtor que recusasse SEMPRE.
    carga = escores_do_modelo(
        grupo="A",
        comp1=[1.0, 2.0],
        comp2=[3.0, 4.0],
        grupos=["A", "B"],
        validacao=validacao_de(0),
    )
    assert carga.validacao.q2 == pytest.approx(0.4)


def test_as_tres_figuras_imprimem_o_par_de_CADA_modelo_na_propria_imagem(  # noqa: N802
    sintetico: dict,
) -> None:
    """**R7 na imagem, e o oráculo é o texto DESENHADO — não a frase que o construtor compôs.**

    Os K pares são derivados aqui, das validações, por caminho independente dos construtores — que
    é a classe (d) do arnês. E a asserção é **por modelo**: um agregado passaria num teste que
    procurasse *um* `Q²`, e é exactamente isso que o desenho recusa.
    """
    escores = sintetico["escores"]
    cargas = sintetico["cargas"]
    importancia = sintetico["importancia"]
    figuras = {
        "escores": figura_dos_escores_um_contra_resto(
            modelos=escores,
            recorte=recorte_dos_escores_um_contra_resto(modelos=escores, n_features_totais=9),
        ),
        "cargas": figura_das_cargas_um_contra_resto(
            modelos=cargas,
            recorte=recorte_das_cargas_um_contra_resto(modelos=cargas, n_features_totais=9),
        ),
        "importancia": figura_da_importancia_um_contra_resto(
            modelos=importancia,
            n_features=4,
            recorte=recorte_da_importancia_um_contra_resto(
                modelos=importancia, n_desenhadas_por_modelo=4
            ),
        ),
    }
    esperados = [
        (m.grupo, decimal(m.validacao.q2), decimal(m.validacao.permutation_pvalue)) for m in escores
    ]
    assert len(esperados) == len(CLASSES), esperados
    assert len({q for _, q, _ in esperados}) == len(esperados), (
        "os Q² esperados repetem-se: a asserção abaixo não distinguiria um agregado de K valores"
    )

    for nome, figura in figuras.items():
        titulos = " | ".join(titulos_de(figura))
        rodape = rodape_de(figura)
        for grupo, q2, p in esperados:
            assert rotulo_do_painel(grupo) in titulos, f"{nome}: falta o painel de {grupo}"
            assert q2 in titulos, f"{nome}: o Q² de {grupo} não está em título nenhum"
            assert p in titulos, f"{nome}: o p de {grupo} não está em título nenhum"
            assert f"{grupo}: Q² = {q2} · p = {p}" in rodape, f"{nome}: o rodapé perdeu {grupo}"


def test_o_par_de_validacao_sobrevive_ao_texto_DESENHADO_no_dado_real(  # noqa: N802
    real: dict,
) -> None:
    """A mesma exigência, sobre os quatro modelos do laboratório (regra 1 do README).

    **O caso que a regra 6 do README existe para tornar visível está aqui:** no `pos_hilic` os
    quatro `Q²` do um-contra-resto são **negativos** — a nuvem separa à vista e o modelo prediz pior
    que a média dos rótulos. A asserção é sobre o sinal e não sobre o valor, porque o valor é
    medição e congelá-lo seria fixar um número antes de saber se é o certo.
    """
    resultado: MulticlassPLSDAResult = real["resultado"]  # type: ignore[assignment]
    escores = real["escores"]
    figura = figura_dos_escores_um_contra_resto(
        modelos=escores,
        recorte=recorte_dos_escores_um_contra_resto(
            modelos=escores,
            n_features_totais=real["matriz"].n_features,  # type: ignore[union-attr]
        ),
    )
    titulos = " | ".join(titulos_de(figura))
    for m in escores:
        assert decimal(m.validacao.q2) in titulos
        assert decimal(m.validacao.permutation_pvalue) in titulos
    assert all(q < 0.0 for q in resultado.q2), (
        f"os Q² do `pos_hilic` cru deixaram de ser todos negativos: {resultado.q2}. Isso é "
        "achado e não defeito — registar antes de mudar o teste"
    )


# ---------------------------------------------------------------------------
# 2. A COLEÇÃO não colapsa — até a imagem
# ---------------------------------------------------------------------------


def test_o_rodape_lista_os_K_pares_e_NENHUM_numero_os_resume(sintetico: dict) -> None:  # noqa: N802
    """**Par causal sobre o colapso**, e a planta é a média que se recusa.

    Antes de plantar, as duas quantidades são calculadas e afirmadas **divergentes**: a média
    dos K `Q²` não pode coincidir com nenhum deles, senão a asserção de ausência passaria por
    coincidência aritmética em vez de por desenho.
    """
    escores = sintetico["escores"]
    valores = [m.validacao.q2 for m in escores]
    media = sum(valores) / len(valores)
    assert all(abs(media - v) > 1e-9 for v in valores), (
        f"a média {media} coincide com um dos Q² {valores}: a planta não separa"
    )

    rodape = rodape_de(
        figura_dos_escores_um_contra_resto(
            modelos=escores,
            recorte=recorte_dos_escores_um_contra_resto(modelos=escores, n_features_totais=9),
        )
    )
    # Presença primeiro — a asserção de AUSÊNCIA abaixo é satisfeita por vacuidade se o rodapé
    # estiver vazio.
    assert len(rodape) > 200, f"o rodapé tem {len(rodape)} caracteres: a ausência seria vácua"
    for m in escores:
        assert decimal(m.validacao.q2) in rodape
    assert decimal(media) not in rodape, (
        f"a média dos Q² ({decimal(media)}) aparece no rodapé — é o agregado que o rodapé recusa"
    )


def test_a_contagem_de_vereditos_preserva_a_particao_e_reporta_a_amplitude() -> None:
    """Os três desfechos de `Q²`, contados — e a amplitude ao lado, sem limiar.

    **Os três são exercitados**: o mecanismo classifica em três baldes, e um que ficasse
    inerte seria indistinguível de um que funciona.
    """
    modelos = [
        escores_do_modelo(
            grupo=nome,
            comp1=[1.0, 2.0],
            comp2=[3.0, 4.0],
            grupos=[nome, "outro"],
            validacao=Validacao(
                q2=q2,
                permutation_pvalue=0.5,
                r2y=0.5,
                n_permutacoes=20,
                cv_folds=3,
                n_componentes=2,
            ),
        )
        for nome, q2 in (("bom", 0.6), ("mau", -0.2), ("sem", float("nan")))
    ]
    frase = frase_dos_vereditos(modelos)
    assert "1 tem Q² <= 0" in frase, frase
    assert "1 tem Q² > 0" in frase, frase
    assert "1 não tem Q² medido" in frase, frase
    assert decimal(0.8) in frase, f"a amplitude entre 0,6 e -0,2 não foi reportada: {frase}"
    assert "não classifica nada" in frase

    # O plural, na outra direção — `frase_contada` já paga esta lição quatro vezes neste pacote.
    dois = [*modelos[:2], modelos[1]]
    dois[2] = dataclasses.replace(modelos[1], grupo="mau2")
    assert "2 têm Q² <= 0" in frase_dos_vereditos(dois), frase_dos_vereditos(dois)


def test_o_regime_viaja_UMA_vez_e_os_pares_K_vezes(sintetico: dict) -> None:  # noqa: N802
    """A frase da coleção repete o par por classe e o regime uma só vez.

    **Repetir o regime K vezes empurraria o rodapé para o teto de `RESERVA_MAXIMA`**, e ele é o
    mesmo nos K modelos por construção: `plsda_multiclass` recebe `cv_folds`, `n_permutations` e
    `n_components` uma vez e passa-os a todos.
    """
    frase = frase_dos_pares(sintetico["escores"])
    assert frase.count("dobras estratificadas") == 1, frase
    assert frase.count("Q² =") == len(CLASSES), frase
    assert f"resolução do p = {decimal(1 / 201)}" in frase, frase


# ---------------------------------------------------------------------------
# 3. A GRELHA — um painel por modelo, e a faixa da legenda
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("quantos", [2, 3, 4, 5])
def test_a_grelha_tem_um_painel_por_modelo_e_NENHUM_eixo_vazio(quantos: int) -> None:  # noqa: N802
    """K painéis mais a faixa da legenda, e nenhum eixo montado sem dado.

    **O caso de K ímpar é o que a leitura natural não cobre:** `subplots(linhas, colunas)` criaria
    `linhas x colunas` eixos e deixaria o último montado e vazio, que se lê como figura quebrada.
    """
    nomes = tuple(f"g{i}" for i in range(quantos))
    modelos = [
        escores_do_modelo(
            grupo=nome,
            comp1=[float(j + i) for j in range(quantos)],
            comp2=[float(-j - i) for j in range(quantos)],
            grupos=list(nomes),
            validacao=validacao_de(i),
        )
        for i, nome in enumerate(nomes)
    ]
    figura = figura_dos_escores_um_contra_resto(
        modelos=modelos,
        recorte=recorte_dos_escores_um_contra_resto(modelos=modelos, n_features_totais=9),
    )
    com_titulo = [e for e in figura.axes if e.get_title().strip()]
    assert len(com_titulo) == quantos, f"{len(com_titulo)} painéis para {quantos} modelos"
    assert len(figura.axes) == quantos + 1, "falta (ou sobra) a faixa da legenda"
    for eixo in com_titulo:
        assert eixo.lines or eixo.collections, f"painel {eixo.get_title()!r} montado e vazio"
    faixa = next(e for e in figura.axes if not e.get_title().strip())
    # `set_axis_off` desliga a moldura e as marcas da CAIXA (`axison`), e não a visibilidade de
    # cada `Axis` — foi a primeira versão deste teste a acusar a diferença.
    assert not faixa.axison, "a faixa da legenda desenhou moldura e marcas"
    assert not faixa.lines and not faixa.collections, "a faixa da legenda recebeu dado"
    assert faixa.get_legend() is not None, "a faixa não recebeu legenda nenhuma"
    # **A altura cresce com as LINHAS e a largura não.** Com K <= 2 a grelha tem uma linha; a
    # partir de 3 são duas, e a figura tem de ser mais alta — senão os painéis achatam-se.
    assert figura.get_figwidth() == pytest.approx(7.0)
    linhas = 1 if quantos <= COLUNAS_DA_GRELHA else (quantos + 1) // COLUNAS_DA_GRELHA
    assert figura.get_figheight() == pytest.approx(4.5 * (linhas + 0.25))


def test_a_legenda_e_UMA_e_nao_uma_por_painel(sintetico: dict) -> None:  # noqa: N802
    """Uma legenda na faixa, e nenhuma dentro dos painéis.

    **K legendas iguais roubariam largura a cada painel para repetir a mesma informação**, e a
    primeira renderização — com `legenda_fora` no último painel — deixou-o mais estreito que os
    outros três.
    """
    escores = sintetico["escores"]
    figura = figura_dos_escores_um_contra_resto(
        modelos=escores,
        recorte=recorte_dos_escores_um_contra_resto(modelos=escores, n_features_totais=9),
    )
    com_legenda = [e for e in figura.axes if e.get_legend() is not None]
    assert len(com_legenda) == 1, f"{len(com_legenda)} eixos com legenda"
    assert not com_legenda[0].get_title().strip(), "a legenda ficou num painel e não na faixa"
    entradas = [t.get_text() for t in com_legenda[0].get_legend().get_texts()]
    assert len(entradas) == len(CLASSES), entradas
    assert len(set(entradas)) == len(entradas), f"a legenda repete entradas: {entradas}"


def test_as_K_classes_recebem_as_K_cores_do_CICLO_e_a_paleta_e_a_mesma_nos_paineis(  # noqa: N802
    sintetico: dict,
) -> None:
    """**Sem literal de cor**, e a mesma classe com a mesma cor em todos os painéis.

    O oráculo é `base.cores_do_ciclo` na ordem canônica — o mesmo caminho que
    `series.cores_dos_grupos` usa —, e a asserção é por painel: duas séries da mesma coorte com
    cores trocadas entre painéis seriam a mesma figura a mentir sobre qual grupo é qual.
    """
    escores = sintetico["escores"]
    figura = figura_dos_escores_um_contra_resto(
        modelos=escores,
        recorte=recorte_dos_escores_um_contra_resto(modelos=escores, n_features_totais=9),
    )
    canonica = ordem_canonica(escores[0].grupos)
    assert len(canonica) == len(CLASSES), canonica
    esperadas = dict(zip(canonica, cores_do_ciclo(len(canonica)), strict=True))

    vistas: dict[str, set[str]] = {}
    for eixo in (e for e in figura.axes if e.get_title().strip()):
        for linha in eixo.lines:
            grupo = str(linha.get_label()).split(" — ")[0]
            vistas.setdefault(grupo, set()).add(str(linha.get_color()))
    assert set(vistas) == set(canonica), f"séries desenhadas: {sorted(vistas)}"
    for grupo, cores in vistas.items():
        assert cores == {str(esperadas[grupo])}, (
            f"{grupo} foi desenhado com {cores} e o ciclo dá {esperadas[grupo]!r} — ou a cor mudou "
            "entre painéis, ou saiu de um literal"
        )


def test_com_ESTILO_cada_classe_tem_a_cor_e_o_simbolo_escolhidos_em_todos_os_paineis(  # noqa: N802
    sintetico: dict,
) -> None:
    """O estilo da coorte vale nos K painéis, com a cor E o marcador de cada classe.

    **A planta é uma escolha só, sobre a primeira classe**, e as outras ficam com o padrão da
    coorte: a asserção cobre os dois ramos de `estilos_dos_grupos` na mesma figura. O teste acima,
    sem estilo, continua a afirmar o ciclo — e passa sem edição.
    """
    escores = sintetico["escores"]
    canonica = ordem_canonica(escores[0].grupos)
    estilos = estilos_dos_grupos(
        canonica, {canonica[0]: EstiloDoGrupo(cor="vermelho", simbolo="quadrado")}
    )
    assert estilos is not None
    figura = figura_dos_escores_um_contra_resto(
        modelos=escores,
        recorte=recorte_dos_escores_um_contra_resto(modelos=escores, n_features_totais=9),
        estilos=estilos,
    )

    vistas: dict[str, set[tuple[str, str]]] = {}
    for eixo in (e for e in figura.axes if e.get_title().strip()):
        for linha in eixo.lines:
            grupo = str(linha.get_label()).split(" — ")[0]
            vistas.setdefault(grupo, set()).add((to_hex(linha.get_color()), linha.get_marker()))
    assert set(vistas) == set(canonica), f"séries desenhadas: {sorted(vistas)}"
    for grupo, tracos in vistas.items():
        estilo = estilos.por_grupo[grupo]
        assert tracos == {(to_hex(estilo.hex), estilo.marcador)}, (grupo, tracos)
    assert vistas[canonica[0]] == {(to_hex("#D55E00"), "s")}


# ---------------------------------------------------------------------------
# 4. O par causal — trocar a matriz de scores muda a figura
# ---------------------------------------------------------------------------


def test_par_causal_trocar_a_matriz_de_scores_MUDA_a_figura(sintetico: dict) -> None:  # noqa: N802
    """Dois modelos trocados de painel produzem uma figura diferente.

    **A âncora vem primeiro:** as duas matrizes de score são comparadas e afirmadas
    divergentes antes de a planta ser aplicada. Sem ela, dois modelos com scores iguais dariam a
    mesma figura e o teste passaria **com a planta aplicada** — que é o par causal que não separa.
    """
    escores: list[EscoresDoModelo] = list(sintetico["escores"])  # type: ignore[arg-type]
    a, b = escores[0], escores[1]
    assert not np.allclose(a.x, b.x) or not np.allclose(a.y, b.y), (
        "os scores dos dois primeiros modelos coincidem: trocá-los é a identidade e a planta não "
        "separa nada"
    )

    def coordenadas(modelos: list[EscoresDoModelo]) -> list[tuple[float, ...]]:
        figura = figura_dos_escores_um_contra_resto(
            modelos=modelos,
            recorte=recorte_dos_escores_um_contra_resto(modelos=modelos, n_features_totais=9),
        )
        return [
            tuple(np.concatenate([s.x, s.y]).tolist())
            for eixo in figura.axes
            if eixo.get_title().strip()
            for s in series(eixo)
        ]

    limpa = coordenadas(escores)
    assert limpa, "a figura limpa não desenhou série nenhuma: a comparação seria vácua"

    trocados = list(escores)
    trocados[0] = dataclasses.replace(a, x=b.x, y=b.y)
    trocados[1] = dataclasses.replace(b, x=a.x, y=a.y)
    assert coordenadas(trocados) != limpa, (
        "trocar a matriz de scores de dois modelos não mudou a figura — o construtor está a "
        "desenhar outra coisa que não os vetores que recebe"
    )


def test_as_plantas_do_arnes_ACUSAM_um_painel_mutilado(sintetico: dict) -> None:  # noqa: N802
    """Série removida e eixos trocados, num painel — as duas direções, na mesma função."""
    escores = sintetico["escores"]

    def nova() -> Figure:
        return figura_dos_escores_um_contra_resto(
            modelos=escores,
            recorte=recorte_dos_escores_um_contra_resto(modelos=escores, n_features_totais=9),
        )

    limpa = nova()
    paineis = [e for e in limpa.axes if e.get_title().strip()]
    esperadas = {s.rotulo: (s.x, s.y) for s in series(paineis[0])}
    assert len(esperadas) == len(CLASSES), esperadas
    assert confere_series(paineis[0], esperadas) == [], "a figura limpa já não bate consigo mesma"

    mutilada = nova()
    alvo = next(e for e in mutilada.axes if e.get_title().strip())
    removida = planta_serie_removida(alvo)
    assert confere_series(alvo, esperadas) != [], f"a remoção de {removida!r} passou despercebida"

    outra = nova()
    alvo2 = next(e for e in outra.axes if e.get_title().strip())
    trocada = planta_eixos_trocados(alvo2)
    assert confere_series(alvo2, esperadas) != [], f"a troca de eixos de {trocada!r} passou"


# ---------------------------------------------------------------------------
# 5. A importância ausente é DECLARADA, nunca desenhada como zero
# ---------------------------------------------------------------------------


def test_par_causal_VIP_zero_contra_VIP_indefinido() -> None:  # noqa: N802
    """Zeros e `nan` são fatos OPOSTOS, e as duas figuras têm de diferir.

    **A âncora vem primeiro:** as duas cargas têm a mesma população e a mesma forma, e o que difere
    é só o conteúdo — se o construtor tratasse `nan` como zero, as duas figuras seriam idênticas e
    esta é a única coisa que as separa.
    """
    nomes = [f"f{i:02d}" for i in range(6)]
    val = validacao_de(0)
    zeros = [
        importancia_do_modelo(grupo=g, vip=[0.0] * 6, features=nomes, validacao=val)
        for g in ("a", "b")
    ]
    indefinidos = [
        importancia_do_modelo(grupo=g, vip=[float("nan")] * 6, features=nomes, validacao=val)
        for g in ("a", "b")
    ]
    assert zeros[0].n == indefinidos[0].n, "as duas cargas têm populações diferentes"
    assert zeros[0].n_indefinidas == 0 and indefinidos[0].n_indefinidas == 6

    figura_zeros = figura_da_importancia_um_contra_resto(
        modelos=zeros,
        n_features=3,
        recorte=recorte_da_importancia_um_contra_resto(modelos=zeros, n_desenhadas_por_modelo=3),
    )
    paineis_zeros = [e for e in figura_zeros.axes if e.get_title().strip()]
    assert all(e.lines for e in paineis_zeros), "o VIP zero deixou de ser desenhado"

    # Com todos indefinidos a figura RECUSA — e a recusa nomeia o fato certo, não «população vazia».
    with pytest.raises(MatrizVaziaError, match="VIP calculável"):
        figura_da_importancia_um_contra_resto(
            modelos=indefinidos,
            n_features=3,
            recorte=recorte_da_importancia_um_contra_resto(
                modelos=indefinidos, n_desenhadas_por_modelo=3
            ),
        )


def test_um_modelo_sem_VIP_calculavel_e_DECLARADO_no_painel_com_a_contagem() -> None:  # noqa: N802
    """Um degenerado ao lado de um que ajustou: a figura desenha um e **escreve** o outro.

    **Derrubar a figura inteira por causa de um modelo apagaria os outros**, e é por isso que a
    recusa só acontece quando nenhum tem o que desenhar. A distinção é `MatrizVaziaError` contra
    R4b, na escala da grelha.
    """
    nomes = [f"f{i:02d}" for i in range(6)]
    val = validacao_de(0)
    modelos = [
        importancia_do_modelo(
            grupo="ajusta", vip=[3.0, 2.0, 1.0, 0.5, 0.2, 0.1], features=nomes, validacao=val
        ),
        importancia_do_modelo(
            grupo="degenera", vip=[float("nan")] * 6, features=nomes, validacao=val
        ),
    ]
    figura = figura_da_importancia_um_contra_resto(
        modelos=modelos,
        n_features=3,
        recorte=recorte_da_importancia_um_contra_resto(modelos=modelos, n_desenhadas_por_modelo=3),
    )
    paineis = {e.get_title().split("\n")[0]: e for e in figura.axes if e.get_title().strip()}
    assert set(paineis) == {rotulo_do_painel("ajusta"), rotulo_do_painel("degenera")}

    ajusta = paineis[rotulo_do_painel("ajusta")]
    degenera = paineis[rotulo_do_painel("degenera")]
    assert ajusta.lines, "o painel que ajustou não desenhou ponto nenhum"
    assert not degenera.lines, "o painel degenerado desenhou pontos — zeros no lugar do indefinido"
    escrito = " ".join(" ".join(t.get_text().split()) for t in degenera.texts)
    assert "não foi possível calcular" in escrito, escrito
    assert "NÃO «importância zero»" in escrito, escrito
    assert not degenera.get_xticks().size and not degenera.get_yticks().size, (
        "o painel sem ponto manteve as marcas, e passa a afirmar um intervalo que ninguém mediu"
    )

    rodape = rodape_de(figura)
    assert "'degenera': 6" in rodape or '"degenera": 6' in rodape, (
        f"o rodapé não declara as 6 indefinidas do modelo degenerado: {rodape}"
    )
    assert "'ajusta': 0" in rodape or '"ajusta": 0' in rodape, rodape


def test_a_frase_do_indisponivel_nomeia_a_POPULACAO_e_nao_a_ausencia_dela() -> None:  # noqa: N802
    """`MatrizVaziaError` diria *"não veio população"*, e a população está lá — é outro fato."""
    carga = importancia_do_modelo(
        grupo="g",
        vip=[float("nan")] * 7,
        features=[f"f{i}" for i in range(7)],
        validacao=validacao_de(0),
    )
    frase = frase_da_importancia_indisponivel(carga)
    assert milhar(7) in frase and "VIP calculável" in frase
    assert "vazia" not in frase.lower()


def test_o_numerador_do_VIP_e_a_UNIAO_e_nao_a_soma(sintetico: dict) -> None:  # noqa: N802
    """`K x n` contaria a mesma feature várias vezes e podia passar da população.

    **A âncora é que há sobreposição entre os modelos** — sem ela a união e a soma coincidem, e o
    teste passaria sobre a implementação errada.
    """
    importancia = sintetico["importancia"]
    por_painel = 4
    conjuntos = [{int(i) for i in selecao_da_importancia(m, por_painel)} for m in importancia]
    uniao = set().union(*conjuntos)
    assert len(uniao) < por_painel * len(conjuntos), (
        f"os {len(conjuntos)} modelos elegem features disjuntas ({sorted(uniao)}): a soma e a "
        "união coincidem, e a planta não separa"
    )
    recorte = recorte_da_importancia_um_contra_resto(
        modelos=importancia, n_desenhadas_por_modelo=por_painel
    )
    assert recorte.desenhadas == len(uniao)
    assert recorte.desenhadas <= recorte.disponiveis


# ---------------------------------------------------------------------------
# 6. A coleção é GUARDADA — as três leituras erradas que ela fecha
# ---------------------------------------------------------------------------


def _carga(nome: str, *, n: int = 3) -> EscoresDoModelo:
    return escores_do_modelo(
        grupo=nome,
        comp1=[float(i) for i in range(n)],
        comp2=[float(-i) for i in range(n)],
        grupos=[nome] * n,
        validacao=validacao_de(0),
    )


def test_um_modelo_so_e_RECUSA_e_nao_grelha_de_um_painel() -> None:  # noqa: N802
    with pytest.raises(PlsdaDesalinhadoError, match="ao menos"):
        recorte_dos_escores_um_contra_resto(modelos=[_carga("a")], n_features_totais=9)
    assert MIN_MODELOS == 2


def test_classe_repetida_na_colecao_e_RECUSA() -> None:  # noqa: N802
    """Dois painéis com o mesmo título dariam dois `Q²` à mesma classe."""
    with pytest.raises(PlsdaDesalinhadoError, match="mais de uma vez"):
        figura_dos_escores_um_contra_resto(
            modelos=[_carga("a"), _carga("a")],
            recorte=recorte_das_cargas_um_contra_resto(
                modelos=[
                    cargas_do_modelo(
                        grupo=g,
                        comp1=[0.1, 0.2],
                        comp2=[0.3, 0.4],
                        features=["f0", "f1"],
                        validacao=validacao_de(0),
                    )
                    for g in ("a", "b")
                ],
                n_features_totais=9,
            ),
        )


def test_populacoes_de_tamanhos_diferentes_sao_RECUSA() -> None:  # noqa: N802
    """Os K modelos saem da MESMA matriz: tamanhos diferentes são duas execuções misturadas."""
    with pytest.raises(PlsdaDesalinhadoError, match="tamanhos diferentes"):
        recorte_dos_escores_um_contra_resto(
            modelos=[_carga("a", n=3), _carga("b", n=4)], n_features_totais=9
        )


@pytest.mark.parametrize(
    ("argumentos", "trecho"),
    [
        ({"grupo": ""}, "classe positiva"),
        ({"grupo": None}, "classe positiva"),
        ({"comp1": []}, "não trouxe score"),
        ({"comp2": [1.0]}, "componente 2"),
        ({"grupos": ["a"]}, "rótulo"),
        ({"grupo": "ausente"}, "não está lá"),
    ],
)
def test_a_fronteira_dos_escores_RECUSA_em_vez_de_recuar(  # noqa: N802
    argumentos: dict, trecho: str
) -> None:
    """Seis formas, uma por execução. **Negativo em processo:** a planta é o argumento."""
    base = {
        "grupo": "a",
        "comp1": [1.0, 2.0],
        "comp2": [3.0, 4.0],
        "grupos": ["a", "b"],
        "validacao": validacao_de(0),
    }
    with pytest.raises((PlsdaDesalinhadoError, MatrizVaziaError), match=trecho):
        escores_do_modelo(**{**base, **argumentos})

    # A outra ponta: sem planta, constrói. Sem esta metade um construtor que recusasse SEMPRE
    # passaria acima.
    assert escores_do_modelo(**base).grupo == "a"  # type: ignore[arg-type]


def test_a_fronteira_das_CARGAS_e_da_IMPORTANCIA_recusa_desalinhamento() -> None:  # noqa: N802
    """As duas irmãs da guarda dos escores — `grep` pelo nome da exceção achou-as."""
    with pytest.raises(PlsdaDesalinhadoError, match="nome"):
        cargas_do_modelo(
            grupo="a",
            comp1=[0.1, 0.2],
            comp2=[0.3, 0.4],
            features=["f0"],
            validacao=validacao_de(0),
        )
    with pytest.raises(PlsdaDesalinhadoError, match="VIP"):
        importancia_do_modelo(grupo="a", vip=[1.0, 2.0], features=["f0"], validacao=validacao_de(0))
    with pytest.raises(MatrizVaziaError, match="carga nenhuma"):
        cargas_do_modelo(grupo="a", comp1=[], comp2=[], features=[], validacao=validacao_de(0))


# ---------------------------------------------------------------------------
# 7. A seleção é determinística, e o rodapé declara o que desenhou
# ---------------------------------------------------------------------------


def test_a_selecao_da_importancia_e_DETERMINISTICA_com_empate_pelo_nome() -> None:  # noqa: N802
    """Regra 3 do README numa seleção em vez de num sorteio. **Não há semente a derivar aqui.**"""
    nomes = ["zz", "aa", "mm", "bb"]
    carga = importancia_do_modelo(
        grupo="g", vip=[1.0, 1.0, 1.0, 1.0], features=nomes, validacao=validacao_de(0)
    )
    escolhidos = [nomes[int(i)] for i in selecao_da_importancia(carga, 3)]
    assert escolhidos == ["aa", "bb", "mm"], escolhidos
    assert escolhidos == [nomes[int(i)] for i in selecao_da_importancia(carga, 3)]
    assert selecao_da_importancia(carga, 0).size == 0
    with pytest.raises(ValueError, match="negativo"):
        selecao_da_importancia(carga, -1)


def test_o_rodape_declara_o_recorte_e_os_numeros_saem_do_ARTISTA(sintetico: dict) -> None:  # noqa: N802
    """A classe (d) do arnês: o número declarado é lido do artista, não recalculado do `Recorte`."""
    cargas = sintetico["cargas"]
    totais = 9
    figura = figura_das_cargas_um_contra_resto(
        modelos=cargas,
        recorte=recorte_das_cargas_um_contra_resto(modelos=cargas, n_features_totais=totais),
    )
    declarados = numeros_declarados(figura)
    assert milhar(totais) in declarados, declarados
    assert str(len(CLASSES)) in declarados or milhar(len(CLASSES)) in declarados


def test_os_tres_titulos_sao_distintos_e_nenhum_e_vazio() -> None:
    """Âncora barata: três títulos iguais fariam o seletor oferecer três vezes o mesmo."""
    titulos = {
        TITULO_DOS_ESCORES_MULTICLASSE,
        TITULO_DAS_CARGAS_MULTICLASSE,
        TITULO_DA_IMPORTANCIA_MULTICLASSE,
    }
    assert len(titulos) == 3 and all(t.strip() for t in titulos)
    assert FEATURES_DA_IMPORTANCIA > 0


def test_o_titulo_do_painel_quebra_o_par_em_DUAS_linhas() -> None:  # noqa: N802
    """Três linhas — a classe, o `Q²` e o `p`. **Veio da imagem montada**.

    Com o par numa linha só, os títulos dos dois painéis de uma linha da grelha tocavam-se, e o da
    direita saía cortado contra a borda.
    """
    titulo = titulo_do_painel("HAMTSP", validacao_de(0))
    linhas = titulo.split("\n")
    assert len(linhas) == 3, linhas
    assert linhas[0] == rotulo_do_painel("HAMTSP")
    assert linhas[1].startswith("Q²") and linhas[2].startswith("p de permutação")
    assert max(len(x) for x in linhas) < 30, (
        f"a linha mais longa do título tem {max(len(x) for x in linhas)} caracteres: numa "
        "grelha de duas colunas ela volta a colidir com o painel vizinho"
    )


# ---------------------------------------------------------------------------
# 8. O registo — as três entram no catálogo com a chave do PLS-DA
# ---------------------------------------------------------------------------


def test_as_tres_estao_no_registo_com_a_chave_do_PLSDA_binario() -> None:  # noqa: N802
    """Uma entrada por FAMÍLIA no `[G-1]`, e a bijeção corre sobre conjuntos de chaves.

    A bijeção completa vive em `tests/unit/test_plots_arnes.py`. Aqui fica a âncora de que as três
    funções deste módulo estão registadas, que é o que apanharia um construtor esquecido.
    """
    do_modulo = {c for c in CONSTRUTORES if c.modulo == "plsda_multiclasse"}
    assert {c.funcao for c in do_modulo} == {
        "figura_dos_escores_um_contra_resto",
        "figura_das_cargas_um_contra_resto",
        "figura_da_importancia_um_contra_resto",
    }
    assert {c.chave for c in do_modulo} == {"PLS-DA"}
    assert {c.vinculo for c in do_modulo} == {Vinculo.CATALOGO}
    for construtor in do_modulo:
        assert hasattr(modulo, construtor.funcao)


def test_os_rotulos_das_tres_NAO_colidem_com_os_do_caminho_binario() -> None:  # noqa: N802
    """Colisão de nome produz rótulos duplicados no combo, e isso já aconteceu com treze entradas.

    **A âncora de não-vacuidade vai primeiro**: com o registo vazio a asserção de distinção
    passaria sobre nada.
    """
    assert CONSTRUTORES, "o registo está vazio: a asserção abaixo seria vácua"
    da_familia = [c for c in CONSTRUTORES if c.chave == "PLS-DA"]
    assert len(da_familia) >= 6, f"a família PLS-DA tem {len(da_familia)} construtores"
    rotulos = [rotulo_do_construtor(c.funcao) for c in da_familia]
    assert len(set(rotulos)) == len(rotulos), f"rótulos repetidos na família PLS-DA: {rotulos}"
    assert all(r.strip() for r in rotulos), rotulos


def test_o_recorte_das_abas_do_catalogo_esta_declarado_no_documento() -> None:
    """A nota do `[G-1]` nomeia o multiclasse, e a linha continua `construída` sem estado novo.

    **É a segunda metade da obrigação de catálogo**, e sem ela o próximo leitor conclui que a
    entrada `PLS-DA` cobre as abas do MetaboAnalyst — e ele não tem aba de um-contra-resto.
    """
    from tests.conftest import REPO

    texto = (REPO / "docs" / "metodos-metaboanalyst.md").read_text(encoding="utf-8")
    assert re.search(r"\|\s*PLS-DA\s*\|\s*Chemometrics\s*\|\s*construída\s*\|", texto), (
        "a linha `PLS-DA` do `[G-1]` deixou de estar `construída`"
    )
    normalizado = " ".join(texto.split())
    assert "A U9 acrescentou TRÊS construtores com a MESMA chave `PLS-DA`" in normalizado
    assert "O MetaboAnalyst não tem aba de um-contra-resto" in normalizado
