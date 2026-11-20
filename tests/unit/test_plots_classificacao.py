"""As figuras dos dois classificadores: a régua do acaso, o `p` no título, e a guarda do nome.

## O que este arquivo afirma, e o que ele deliberadamente NÃO afirma

**Nenhuma asserção sobre um número de acurácia do corpus real.** Ele é resultado do estudo, e o
que aqui se afirma é que a figura **declara** o que R10 obriga — a medida, o esquema de validação e
o `p` de permutação — e que o traço corresponde ao dado.

**Nenhuma asserção sobre legibilidade.** As classes do arnês leem o objeto `Figure`; nenhuma delas
sabe se quatro rótulos de classe cabem na largura do eixo. Isso vê-se na imagem montada.

## A planta que esta família tem e as outras não: o NOME

Os dois construtores desenham o **mesmo** traço e diferem só no rótulo. Sem a guarda de nome, uma
carga trocada produz uma figura perfeitamente bem formada com o método errado no título e os
números de outro modelo ao lado — e nenhuma das quatro classes do arnês a vê, porque tudo o que
elas leem está coerente consigo próprio. A guarda é comparação de texto, e o par causal dela é o
teste mais barato deste arquivo.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Final

import pytest
from matplotlib.colors import to_hex

from refino.plots.base import MatrizVaziaError
from refino.plots.classificacao import (
    NOME_DA_FLORESTA,
    NOME_DO_SVM,
    Classificacao,
    ClassificacaoDesalinhadaError,
    classificacao_do_fio,
    figura_do_random_forest,
    figura_do_svm,
    frase_da_validacao_do_classificador,
    recorte_do_classificador,
    rotulo_do_acaso,
    veredito_do_classificador,
)
from refino.plots.recorte import RecorteInvalidoError
from refino.plots.series import EstiloDoGrupo, estilos_dos_grupos
from tests.arnes_de_figura import (
    confere_rotulos,
    confere_series,
    confere_vista,
    numeros_declarados,
    planta_serie_removida,
    series,
)

if TYPE_CHECKING:  # pragma: no cover - só para o mypy
    from matplotlib.figure import Figure

# A população do `pos_hilic`, literal aqui de propósito: é a fonte independente da classe (d).
POPULACAO: Final = 6035
USADAS: Final = 6031
MATRIZ: Final = "a matriz processada"
AMOSTRAGEM: Final = "todas as 6.031 features utilizáveis, sem sorteio"

# As quatro classes reais e as contagens das amostras de estudo do `pos_hilic`, medidas em
# 18/09/2026. São **observação** sobre aqueles quatro arquivos e não invariante — aqui
# servem de forma plausível, e a figura não depende delas.
GRUPOS: Final = ("ATLL", "Asymptomatic", "HAMTSP", "NoHTLV")
CONTAGENS: Final = (13, 18, 14, 16)


def carga(
    *,
    nome: str = NOME_DA_FLORESTA,
    recall: tuple[float, ...] = (0.4, 0.6, 0.3, 0.5),
    p: float = 0.0199,
    balanceada: float = 0.45,
) -> Classificacao:
    """Uma `Classificacao` plausível. **Os quatro `recall` diferem entre si de propósito.**

    Com quatro barras iguais, trocar duas de lugar seria a identidade e a planta de série removida
    não separaria nada — é a âncora que `planta_eixos_trocados` exige, aqui aplicada ao traço.
    """
    return classificacao_do_fio(
        nome=nome,
        grupos=list(GRUPOS),
        contagens=list(CONTAGENS),
        recall=list(recall),
        balanceada=balanceada,
        simples=0.4754,
        macro_f1=0.4321,
        p=p,
        acaso=0.25,
        cv_folds=5,
        n_permutacoes=200,
        semente=1921827941,
    )


def recorte_de(classificacao: Classificacao):
    return recorte_do_classificador(
        classificacao=classificacao,
        n_features_usadas=USADAS,
        n_features_totais=POPULACAO,
        matriz=MATRIZ,
        amostragem=AMOSTRAGEM,
    )


def figura_de(classificacao: Classificacao) -> Figure:
    construtor = (
        figura_do_random_forest if classificacao.nome == NOME_DA_FLORESTA else figura_do_svm
    )
    return construtor(classificacao=classificacao, recorte=recorte_de(classificacao))


# ---------------------------------------------------------------------------
# 1. A guarda do NOME — a planta que só esta família tem
# ---------------------------------------------------------------------------


def test_o_construtor_da_FLORESTA_recusa_uma_carga_de_svm() -> None:  # noqa: N802
    """A metade que acusa. Sem ela a troca sai bem formada e com o método errado no título."""
    with pytest.raises(ClassificacaoDesalinhadaError, match="Random Forest"):
        figura_do_random_forest(
            classificacao=carga(nome=NOME_DO_SVM),
            recorte=recorte_de(carga(nome=NOME_DO_SVM)),
        )


def test_o_construtor_do_SVM_recusa_uma_carga_da_floresta() -> None:  # noqa: N802
    """A direção oposta: uma guarda que só olhasse para um dos nomes passaria metade das trocas."""
    with pytest.raises(ClassificacaoDesalinhadaError, match="SVM"):
        figura_do_svm(
            classificacao=carga(nome=NOME_DA_FLORESTA),
            recorte=recorte_de(carga(nome=NOME_DA_FLORESTA)),
        )


def test_cada_construtor_ACEITA_a_carga_dele() -> None:  # noqa: N802
    """A terceira direção, e sem ela uma guarda que recusasse tudo passaria nas duas acima.

    **É o par causal completo:** duas plantas que acusam e a árvore limpa que não acusa.
    """
    assert figura_de(carga(nome=NOME_DA_FLORESTA)).axes
    assert figura_de(carga(nome=NOME_DO_SVM)).axes


def test_os_dois_nomes_DIFEREM_e_sao_as_entradas_do_catalogo() -> None:  # noqa: N802
    """Âncora das três acima: com os dois nomes iguais, nenhuma das plantas separaria nada."""
    assert NOME_DA_FLORESTA != NOME_DO_SVM
    assert (NOME_DA_FLORESTA, NOME_DO_SVM) == ("Random Forest", "SVM")


# ---------------------------------------------------------------------------
# 2. A figura declara a MEDIDA, o esquema de VALIDAÇÃO e o `p`
# ---------------------------------------------------------------------------


def test_a_figura_declara_a_medida_o_esquema_de_validacao_e_o_p() -> None:
    """R10 na imagem: os três no título e no rodapé, e o `n` de sorteios ao lado do `p`.

    **O `n` viaja com o `p` sempre**: sem ele, `0,00498` lê-se
    como medida quando é o **mínimo possível** de 200 sorteios.
    """
    classificacao = carga()
    figura = figura_de(classificacao)
    titulo = figura.axes[0].get_title()
    assert NOME_DA_FLORESTA in titulo
    assert "acurácia balanceada" in titulo
    assert "p de permutação" in titulo

    # **O texto CORRIDO e não o cru, e a razão é `linhas_do_rodape`.** O rodapé quebra em linhas
    # para caber na largura, e *"F1 macro"* chega partido por uma quebra de linha — um `in` sobre
    # o cru devolveria `False` sobre um rodapé perfeitamente correto. É a mesma razão pela qual
    # `arnes_de_figura.texto_corrido` existe: o veredito seria *"a frase não está lá"* sobre uma
    # frase que está.
    rodape = " ".join(" ".join(t.get_text().split()) for t in figura.texts)
    for obrigatorio in (
        "acurácia balanceada",
        "acurácia simples",
        "F1 macro",
        "dobras estratificadas",
        "p de permutação",
        "sorteios de rótulo",
    ):
        assert obrigatorio in rodape, f"o rodapé não declara {obrigatorio!r}: {rodape}"


def test_os_numeros_declarados_SAEM_do_dado_e_nao_de_um_literal() -> None:  # noqa: N802
    """Classe (d): cada número escrito na figura é derivado da entrada por caminho independente.

    **A âncora de não-vacuidade está na mesma função**: uma figura sem número nenhum
    satisfaria a asserção de pertinência abaixo por conjunto vazio.
    """
    classificacao = carga()
    escritos = numeros_declarados(figura_de(classificacao))
    assert escritos, "a figura não declara número nenhum — a asserção abaixo seria vácua"

    for esperado in ("0,45", "0,0199", "6.031", "6.035", "200", "5"):
        assert esperado in escritos, f"{esperado!r} não está entre os números escritos: {escritos}"


def test_a_frase_da_validacao_é_UMA_e_as_duas_figuras_a_partilham() -> None:  # noqa: N802
    """Uma função e não um literal por figura — uma fonte só, aplicada a texto."""
    floresta = carga(nome=NOME_DA_FLORESTA)
    maquina = carga(nome=NOME_DO_SVM)
    assert frase_da_validacao_do_classificador(floresta) == frase_da_validacao_do_classificador(
        maquina
    ), "as duas cargas têm os mesmos números e a frase divergiu: há duas redações"

    rodape_da_floresta = " ".join(t.get_text() for t in figura_de(floresta).texts)
    rodape_do_svm = " ".join(t.get_text() for t in figura_de(maquina).texts)
    frase = frase_da_validacao_do_classificador(floresta)
    # O rodapé quebra em linhas, então a comparação é sobre o texto corrido de cada fragmento.
    for pedaco in frase.split(" · "):
        alvo = " ".join(pedaco.split())
        assert alvo in " ".join(rodape_da_floresta.split()), pedaco
        assert alvo in " ".join(rodape_do_svm.split()), pedaco


# ---------------------------------------------------------------------------
# 3. O traço: uma barra por classe, e a régua do acaso
# ---------------------------------------------------------------------------


def test_cada_classe_com_recall_vira_UMA_barra_da_altura_certa() -> None:  # noqa: N802
    """Classe (a): o dado desenhado é o dado que entrou, série a série.

    A planta entra como **argumento** de `confere_series` — negativo em processo, que não consegue
    ficar inerte.
    """
    classificacao = carga()
    figura = figura_de(classificacao)
    eixo = figura.axes[0]

    esperadas = {
        f"{nome} — {quantas} amostras": ([float(i), float(i)], [0.0, valor])
        for i, (nome, quantas, valor) in enumerate(
            zip(GRUPOS, CONTAGENS, classificacao.recall, strict=True)
        )
    }
    esperadas[rotulo_do_acaso(classificacao)] = ([0.0, 1.0], [0.25, 0.25])
    assert confere_series(eixo, esperadas) == []


def test_par_causal_serie_removida() -> None:
    """Tirar uma classe do traço tem de ser acusado — a metade que prova que a classe (a) age."""
    classificacao = carga()
    figura = figura_de(classificacao)
    eixo = figura.axes[0]
    esperadas = {
        f"{nome} — {quantas} amostras": ([float(i), float(i)], [0.0, valor])
        for i, (nome, quantas, valor) in enumerate(
            zip(GRUPOS, CONTAGENS, classificacao.recall, strict=True)
        )
    }
    esperadas[rotulo_do_acaso(classificacao)] = ([0.0, 1.0], [0.25, 0.25])

    apagada = planta_serie_removida(eixo)
    achados = confere_series(eixo, esperadas)
    assert any(apagada in a for a in achados), achados


def test_a_classe_sem_RECALL_nao_vira_barra_de_altura_zero() -> None:  # noqa: N802
    """R12b: o vazio é achado, e a ausência da barra é como ele se declara.

    **Um segmento nulo entra em `get_segments` como dois vértices coincidentes**, e a classe (a)
    passaria a contar pontos que ninguém vê — é a escolha do K-means, pelo mesmo motivo.

    **A planta cai sobre dado em que as duas coisas DIFEREM:** a mesma carga com a classe a `0,3`
    desenha a barra, e a `0,0` não desenha. As duas contagens estão na mesma função.
    """
    com_barra = carga(recall=(0.4, 0.6, 0.3, 0.5))
    sem_barra = carga(recall=(0.4, 0.6, 0.0, 0.5))

    rotulos_com = {s.rotulo for s in series(figura_de(com_barra).axes[0])}
    rotulos_sem = {s.rotulo for s in series(figura_de(sem_barra).axes[0])}
    alvo = "HAMTSP — 14 amostras"
    assert alvo in rotulos_com
    assert alvo not in rotulos_sem, (
        f"a classe de recall zero continuou a desenhar uma barra: {sorted(rotulos_sem)}"
    )
    assert len(rotulos_com) == len(rotulos_sem) + 1, (
        "as duas figuras têm o mesmo número de séries: a planta não separa"
    )


def test_a_REGUA_do_acaso_e_desenhada_SEMPRE_inclusive_no_resultado_negativo() -> None:  # noqa: N802
    """É ela que torna o resultado negativo legível como resultado (R12b).

    Quatro barras baixas sem régua lêem-se como figura quebrada; com régua lêem-se como *"o modelo
    não separa"*. O par é sobre a **presença** dela nos dois desfechos, não sobre a altura.

    **A `balanceada` acompanha o `recall` nos dois casos, e isso é o que faz a planta separar.** Com
    ela fixa em `0,45`, `acima_do_acaso` seria `True` nas duas iterações e uma régua desenhada
    **só** quando o modelo separa passaria neste teste — que é a planta que a primeira redação não
    apanhou, medida em 18/09/2026.
    """
    for recall, balanceada in (((0.9, 0.9, 0.9, 0.9), 0.9), ((0.1, 0.1, 0.1, 0.1), 0.1)):
        classificacao = carga(recall=recall, balanceada=balanceada)
        assert (classificacao.balanceada > classificacao.acaso) == (balanceada > 0.25), (
            "âncora: as duas iterações caem do mesmo lado da régua, e a planta não separa"
        )
        rotulos_desenhados = {s.rotulo for s in series(figura_de(classificacao).axes[0])}
        assert rotulo_do_acaso(classificacao) in rotulos_desenhados, sorted(rotulos_desenhados)


def test_o_eixo_do_recall_vai_de_ZERO_a_UM_e_nao_se_adapta_ao_dado() -> None:  # noqa: N802
    """Deixar o `matplotlib` escolher a extensão faria desempenhos opostos ter a mesma imagem.

    **A planta cai sobre dado em que as duas coisas diferem:** com barras a `0,1` e a `0,9`, um
    eixo adaptativo daria duas vistas idênticas com números diferentes nas marcas.
    """
    baixo = figura_de(carga(recall=(0.1, 0.1, 0.1, 0.1))).axes[0]
    alto = figura_de(carga(recall=(0.9, 0.9, 0.9, 0.9))).axes[0]
    assert baixo.get_ylim() == (0.0, 1.0)
    assert alto.get_ylim() == baixo.get_ylim()


def test_os_rotulos_do_eixo_nomeiam_a_CLASSE_e_a_medida() -> None:  # noqa: N802
    """Classe (b): o eixo diz o que devia dizer, e a legenda nomeia o que está desenhado."""
    classificacao = carga()
    figura = figura_de(classificacao)
    esperada = [
        f"{nome} — {quantas} amostras" for nome, quantas in zip(GRUPOS, CONTAGENS, strict=True)
    ]
    esperada.append(rotulo_do_acaso(classificacao))
    achados = confere_rotulos(
        figura.axes[0],
        x="classe",
        y="recall fora da amostra",
        titulo=figura.axes[0].get_title(),
        legenda=esperada,
    )
    assert achados == []


def test_a_vista_esta_sa() -> None:
    """Classe (c): nada invisível, nada fora do enquadramento, nada em escala trocada."""
    assert confere_vista(figura_de(carga()).axes[0]) == []


# ---------------------------------------------------------------------------
# 4. O veredito — QUATRO desfechos, e todos vistos
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("balanceada", "p", "trecho"),
    [
        (0.45, float("nan"), "NÃO FOI MEDIDO"),
        (0.25, 0.5, "NÃO SEPARA"),
        (0.10, 0.5, "NÃO SEPARA"),
        (0.45, 0.30, "NÃO SUSTENTA"),
        (0.45, 0.01, "raramente"),
    ],
)
def test_o_veredito_tem_os_QUATRO_desfechos_e_cada_um_e_alcancavel(  # noqa: N802
    balanceada: float, p: float, trecho: str
) -> None:
    """A verificação pede que se tenham visto todos, e «dois» é o piso e não o número.

    O terceiro é o deste corpus: separação acima do acaso que o `p` não sustenta.
    """
    assert trecho in veredito_do_classificador(carga(balanceada=balanceada, p=p))


def test_os_QUATRO_vereditos_sao_TEXTOS_DISTINTOS() -> None:  # noqa: N802
    """Âncora do teste acima: com duas frases iguais, um dos `trecho` casaria por acaso."""
    vereditos = {
        veredito_do_classificador(carga(balanceada=b, p=p))
        for b, p in ((0.45, float("nan")), (0.25, 0.5), (0.45, 0.30), (0.45, 0.01))
    }
    assert len(vereditos) == 4, vereditos


def test_o_rodape_conta_as_classes_acima_do_acaso_nos_TRES_ramos() -> None:  # noqa: N802
    """`frase_contada` tem três ramos, e `uma` é obrigatório — os três são alcançáveis aqui."""
    nenhuma = recorte_de(carga(recall=(0.1, 0.2, 0.2, 0.1))).metodo
    uma = recorte_de(carga(recall=(0.1, 0.9, 0.2, 0.1))).metodo
    algumas = recorte_de(carga(recall=(0.9, 0.9, 0.2, 0.1))).metodo
    assert "nenhuma classe ficou acima do acaso" in nenhuma
    assert "1 classe ficou acima do nível do acaso" in uma
    assert "as 2 classes ficaram acima do nível do acaso" in algumas


# ---------------------------------------------------------------------------
# 5. A fronteira: `classificacao_do_fio` e as recusas tipadas
# ---------------------------------------------------------------------------


def test_o_p_e_campo_OBRIGATORIO_da_carga_de_desenho() -> None:  # noqa: N802
    """A regra 6 do README atravessa o fio aqui, e o mecanismo é a ausência de valor padrão.

    **A âncora está na mesma função**: uma classe sem campos satisfaria o laço por vazio.
    """
    from dataclasses import MISSING, fields

    campos = fields(Classificacao)
    assert len(campos) >= 10, f"`Classificacao` tem {len(campos)} campos — a asserção seria vácua"
    assert "p" in {c.name for c in campos}
    com_padrao = sorted(
        c.name for c in campos if c.default is not MISSING or c.default_factory is not MISSING
    )
    assert com_padrao == [], (
        f"estes campos ganharam valor padrão: {com_padrao}. Com um padrão em `p`, uma carga com a "
        "medida e sem o `p` passa a ser construível, e a regra 6 do README deixa de ser mecanismo"
    )
    assert Classificacao.__dataclass_params__.frozen


@pytest.mark.parametrize(
    ("campos", "excecao", "trecho"),
    [
        ({"grupos": []}, MatrizVaziaError, "sem classe nenhuma"),
        ({"recall": [0.4, 0.6]}, ClassificacaoDesalinhadaError, "são lidos alinhados"),
        ({"contagens": [13, 18]}, ClassificacaoDesalinhadaError, "são lidos alinhados"),
    ],
)
def test_a_carga_DESALINHADA_e_recusa_tipada(  # noqa: N802
    campos: dict[str, object], excecao: type[Exception], trecho: str
) -> None:
    """Um vetor mais curto pintaria cada barra com o nome da vizinha, e a figura sairia bem formada.

    A planta entra como **argumento** — negativo em processo.
    """
    base = {
        "nome": NOME_DA_FLORESTA,
        "grupos": list(GRUPOS),
        "contagens": list(CONTAGENS),
        "recall": [0.4, 0.6, 0.3, 0.5],
        "balanceada": 0.45,
        "simples": 0.4754,
        "macro_f1": 0.4321,
        "p": 0.0199,
        "acaso": 0.25,
        "cv_folds": 5,
        "n_permutacoes": 200,
        "semente": 1,
    }
    with pytest.raises(excecao, match=trecho):
        classificacao_do_fio(**{**base, **campos})  # type: ignore[arg-type]


def test_a_carga_SA_nao_levanta() -> None:  # noqa: N802
    """A outra ponta: sem isto, um construtor que recusasse tudo passaria nas três plantas."""
    assert carga().n_classes == 4
    assert carga().n_amostras == sum(CONTAGENS)


def test_o_p_AUSENTE_vira_nan_e_a_figura_escreve_nao_declarado() -> None:  # noqa: N802
    """`nan` e não `0.0`, e a distinção chega à imagem.

    Um zero em `p` afirmaria que **nenhum** sorteio de rótulos alcança o modelo — a afirmação mais
    forte que esta família pode fazer —, e fazê-la a partir de um campo ausente é inventar o
    resultado.
    """
    sem_p = carga(p=float("nan"))
    assert math.isnan(sem_p.p)
    assert "não declarado" in figura_de(sem_p).axes[0].get_title()


def test_o_recorte_recusa_denominador_impossivel() -> None:
    """`desenhadas > disponiveis` é numerador e denominador de populações diferentes."""
    with pytest.raises(RecorteInvalidoError):
        recorte_do_classificador(
            classificacao=carga(),
            n_features_usadas=POPULACAO + 1,
            n_features_totais=POPULACAO,
            matriz=MATRIZ,
            amostragem=AMOSTRAGEM,
        )


@pytest.mark.parametrize("nome", [NOME_DA_FLORESTA, NOME_DO_SVM])
def test_com_ESTILO_cada_barra_tem_a_cor_do_GRUPO_nos_dois_classificadores(nome: str) -> None:  # noqa: N802
    """As classes são os grupos clínicos, e a barra de cada uma leva a cor escolhida.

    **Nos dois construtores**, porque os dois públicos passam o estilo ao desenho partilhado — um
    que o esquecesse desenharia as cores do ciclo com a figura bem formada. **Só a cor**: a barra
    não tem marcador. Sem estilo, os testes acima continuam a afirmar o ciclo.
    """
    estilos = estilos_dos_grupos(GRUPOS, {"HAMTSP": EstiloDoGrupo(cor="vermelho", simbolo="cruz")})
    assert estilos is not None
    classificacao = carga(nome=nome)
    construtor = figura_do_random_forest if nome == NOME_DA_FLORESTA else figura_do_svm
    figura = construtor(
        classificacao=classificacao, recorte=recorte_de(classificacao), estilos=estilos
    )
    cores = {
        str(colecao.get_label()).split(" — ")[0]: to_hex(colecao.get_colors()[0])
        for colecao in figura.axes[0].collections
        if " — " in str(colecao.get_label())
    }
    assert cores == {grupo: to_hex(estilos.por_grupo[grupo].hex) for grupo in GRUPOS}
    assert cores["HAMTSP"] == to_hex("#D55E00")
