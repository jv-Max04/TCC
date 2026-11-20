"""Figuras dos dois classificadores, Random Forest e SVM.

São dois construtores porque o MetaboAnalyst lista os dois métodos como entradas separadas, e
um desenho só, porque a figura é a mesma sobre dados diferentes. O que não é partilhado é o
nome: cada construtor recusa a `Classificacao` do outro, senão uma carga trocada sairia bem
formada com o método errado no título.

A figura tem uma barra por classe, com altura igual ao recall fora da amostra, e uma linha
horizontal no nível do acaso. A média das barras é a acurácia balanceada do título. O número
sozinho não distingue quatro classes a 0,25 (o modelo não aprendeu nada) de uma a 1,0 e três a
0 (o modelo aprendeu a classe grande); as barras distinguem.

Resultado sem sinal é achado, e a figura o diz: a régua do acaso é desenhada sempre, e o
veredito do rodapé lê o `p` de permutação, não a acurácia. `Classificacao` é um dataclass
congelado sem valor padrão e `p` é campo dele, então não há carga de desenho com a medida de
desempenho e sem o `p`. As cores das classes saem de `series.series_por_grupo`, para a mesma
coorte ter a mesma paleta em qualquer aba.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final

import numpy as np

from refino.plots.base import (
    MatrizVaziaError,
    RecusaDoConstrutorError,
    denominador_ou_recusa,
    inteiro_do_fio,
    nova_figura,
    numero_do_fio,
    vetor_do_fio,
)
from refino.plots.recorte import (
    TAMANHO_DA_FONTE,
    Recorte,
    aplica,
    decimal,
    frase_contada,
    legenda_fora,
    milhar,
)
from refino.plots.series import series_por_grupo

if TYPE_CHECKING:  # pragma: no cover - só para o mypy
    from matplotlib.artist import Artist
    from matplotlib.figure import Figure

    from refino.plots.series import EstilosDosGrupos

__all__ = [
    "LARGURA_DA_BARRA_DA_CLASSE",
    "NOME_DA_FLORESTA",
    "NOME_DO_SVM",
    "Classificacao",
    "ClassificacaoDesalinhadaError",
    "classificacao_do_fio",
    "figura_do_random_forest",
    "figura_do_svm",
    "frase_da_validacao_do_classificador",
    "recorte_do_classificador",
    "rotulo_do_acaso",
    "veredito_do_classificador",
]

NOME_DA_FLORESTA: Final = "Random Forest"
NOME_DO_SVM: Final = "SVM"
"""Os dois nomes, iguais às entradas do MetaboAnalyst.

Repetem `statistics.classificadores.NOME_DO_CLASSIFICADOR` de propósito. O nome chega aqui como
texto vindo de JSON, e importar a constante do módulo de cálculo faria a guarda comparar um
valor lido do fio contra uma constante do processo que não o produziu. O que liga os dois lados
é o teste de contrato.
"""

LARGURA_DA_BARRA_DA_CLASSE: Final = 26.0
"""Espessura da barra de recall, em pontos. O mesmo valor da figura do K-means."""


class ClassificacaoDesalinhadaError(RecusaDoConstrutorError):
    """Os vetores por classe não têm o mesmo comprimento, ou o nome não bate.

    Herda de `RecusaDoConstrutorError` porque é a única família que o painel apanha; outra
    exceção levantada dentro de um slot do Qt vai para o `stderr` e o painel fica com a figura
    anterior.

    Cobre os dois casos de figura bem formada e errada: um vetor de recalls mais curto que o de
    classes pintaria cada barra com o nome da vizinha, e uma `Classificacao` de SVM passada ao
    construtor da floresta poria no título um método que não produziu aqueles números.
    """


@dataclass(frozen=True)
class Classificacao:
    """O desempenho desenhável de um classificador, com o `p` de permutação como campo.

    Os doze campos são obrigatórios e nenhum tem valor padrão: não existe `Classificacao` com
    `balanceada` e sem `p`.

    Não é `ClassifierResult`, que é o resultado do núcleo, com o digest das colunas e as
    importâncias por feature. Esta é a carga que atravessa o fio para desenhar uma figura. A
    matriz de confusão fica fora porque a figura só desenha a diagonal normalizada dela, que é
    o vetor de recall.
    """

    nome: str
    """`Random Forest` ou `SVM`. É o que a guarda do nome compara."""

    grupos: tuple[str, ...]
    contagens: tuple[int, ...]
    recall: tuple[float, ...]
    """O recall de cada classe, na ordem de `grupos`. A média deles é `balanceada`."""

    balanceada: float
    simples: float
    macro_f1: float
    p: float
    """O `p` de permutação."""

    acaso: float
    """`1/K`. Vem calculado do núcleo e não é recalculado aqui."""

    cv_folds: int
    n_permutacoes: int
    semente: int

    @property
    def n_classes(self) -> int:
        return len(self.grupos)

    @property
    def n_amostras(self) -> int:
        return sum(self.contagens)

    @property
    def n_acima_do_acaso(self) -> int:
        """Quantas classes têm recall acima do nível do acaso."""
        return sum(1 for v in self.recall if math.isfinite(v) and v > self.acaso)


def classificacao_do_fio(
    *,
    nome: Any,
    grupos: Any,
    contagens: Any,
    recall: Any,
    balanceada: Any,
    simples: Any,
    macro_f1: Any,
    p: Any,
    acaso: Any,
    cv_folds: Any,
    n_permutacoes: Any,
    semente: Any,
) -> Classificacao:
    """Monta uma `Classificacao` a partir do que o fio ou o núcleo trazem, com as guardas.

    Recebe os campos um a um, e não o dicionário, porque o teste de contrato entre a janela e a
    API só confere as chaves lidas no código da janela. Se o dicionário fosse lido aqui, uma
    chave mal escrita devolveria `nan` para sempre sem nada acusar.

    O que falta vira `nan`, não `0.0`: um zero em `p` afirmaria que nenhum sorteio de rótulos
    alcança o modelo, a partir de um campo ausente.
    """
    nomes = tuple(str(g) for g in grupos) if isinstance(grupos, (list, tuple)) else ()
    if not nomes:
        raise MatrizVaziaError(
            "a classificação chegou sem classe nenhuma: não há barra para desenhar, e figura em "
            "branco seria indistinguível de figura quebrada"
        )

    vetor_recall = vetor_do_fio(recall)
    vetor_contagens = (
        [inteiro_do_fio(c) for c in contagens] if isinstance(contagens, (list, tuple)) else []
    )
    if len(vetor_recall) != len(nomes) or len(vetor_contagens) != len(nomes):
        raise ClassificacaoDesalinhadaError(
            f"chegaram {len(nomes)} classe(s), {len(vetor_recall)} recall(s) e "
            f"{len(vetor_contagens)} contagem(ns). Os três são por classe e são lidos "
            "alinhados: um vetor mais curto pinta cada barra com o nome da vizinha, e a figura "
            "sai bem formada"
        )

    return Classificacao(
        nome=str(nome),
        grupos=nomes,
        contagens=tuple(vetor_contagens),
        recall=tuple(float(v) for v in vetor_recall.tolist()),
        balanceada=numero_do_fio(balanceada),
        simples=numero_do_fio(simples),
        macro_f1=numero_do_fio(macro_f1),
        p=numero_do_fio(p),
        acaso=numero_do_fio(acaso),
        cv_folds=inteiro_do_fio(cv_folds),
        n_permutacoes=inteiro_do_fio(n_permutacoes),
        semente=inteiro_do_fio(semente),
    )


def _numero_ou_ausente(valor: float) -> str:
    """`0,2371` ou `"não declarado"`. Nunca um número no lugar da ausência dele.

    Repete `validacao._numero_ou_ausente`: são três linhas, e este módulo não consome
    `Validacao`.
    """
    return decimal(valor) if math.isfinite(valor) else "não declarado"


def rotulo_do_acaso(classificacao: Classificacao) -> str:
    """A legenda da régua: `"acaso — 0,2500 (1 de 4 classes)"`.

    O `1/K` vai por extenso ao lado do número porque '0,25' sozinho parece um limiar escolhido.
    A linha marca o valor que a acurácia balanceada assume por definição quando a predição é
    independente do rótulo.
    """
    return (
        f"acaso — {_numero_ou_ausente(classificacao.acaso)} "
        f"(1 de {milhar(classificacao.n_classes)} classes)"
    )


def frase_da_validacao_do_classificador(classificacao: Classificacao) -> str:
    """A linha do rodapé: as três medidas, o esquema de validação e o `p` com o `n` de sorteios.

    É uma função para que as duas figuras escrevam a mesma frase.

    As três medidas vão juntas. A balanceada é a que o título destaca; a simples vai ao lado
    porque sob desbalanço ela premia quem responde sempre a classe maior, e a distância entre as
    duas é o sintoma.

    O `n` de sorteios acompanha o `p`: a convenção é `(k+1)/(n+1)`, a resolução do teste é
    `1/(n+1)`, e sem o `n` um `p = 0,00498` parece medida quando é o mínimo de 200 sorteios.
    """
    return (
        f"acurácia balanceada = {_numero_ou_ausente(classificacao.balanceada)} "
        f"(validação cruzada de {milhar(classificacao.cv_folds)} dobras estratificadas) · "
        f"acurácia simples = {_numero_ou_ausente(classificacao.simples)} · "
        f"F1 macro = {_numero_ou_ausente(classificacao.macro_f1)} · "
        f"p de permutação = {_numero_ou_ausente(classificacao.p)} "
        f"em {milhar(classificacao.n_permutacoes)} sorteios de rótulo"
    )


def veredito_do_classificador(classificacao: Classificacao) -> str:
    """O que o par `(balanceada, p)` significa, em quatro desfechos.

    - `p` não finito: não foi medido; não há nula, e a acurácia ao lado não é interpretável.
    - balanceada `<=` acaso: o modelo não separa.
    - acima do acaso com `p >= 0,05`: separa dentro da amostra, e um sorteio de rótulos o alcança.
    - acima do acaso com `p < 0,05`: separa, e os sorteios de rótulo raramente o alcançam.

    O corte de `0,05` é convenção da área, como `anova.FDR_ALPHA`, e não é calibrado. O nível do
    acaso `1/K` é definição.
    """
    if not math.isfinite(classificacao.p):
        return (
            "O p de permutação NÃO FOI MEDIDO: sem a nula, a acurácia ao lado não distingue "
            "separação de sorte, e esta figura não tem como dizer se o modelo aprendeu alguma coisa"
        )
    if (
        not math.isfinite(classificacao.balanceada)
        or classificacao.balanceada <= classificacao.acaso
    ):
        return (
            "O modelo NÃO SEPARA os grupos: a acurácia balanceada fica no nível do acaso ou abaixo "
            "dele. É achado e não defeito — com 61 amostras e milhares de features, é o desfecho "
            "que a correção de múltiplos testes já antecipa nas univariadas"
        )
    if classificacao.p >= 0.05:
        return (
            "O modelo fica ACIMA DO ACASO e o p NÃO SUSTENTA a separação: um sorteio de rótulos "
            "alcança este desempenho com frequência. É o falso positivo que a validação existe "
            "para tornar visível — separação convincente na tela com validação que não a confirma"
        )
    return (
        "O modelo fica acima do acaso e os sorteios de rótulo raramente o alcançam. QUANTO acima é "
        "a acurácia balanceada, e o p diz com que frequência o acaso a iguala — o par lê-se junto, "
        "porque um p pequeno sobre um ganho minúsculo continua a ser um ganho minúsculo"
    )


def recorte_do_classificador(
    *,
    classificacao: Classificacao,
    n_features_usadas: int,
    n_features_totais: Any,
    matriz: str,
    amostragem: str,
) -> Recorte:
    """O rodapé: quantas features entraram, com que método, com que validação e com que `p`.

    O método nomeia o classificador, o esquema de validação e o `p`, que a figura tem de declarar
    na própria imagem. A semente entra porque as duas famílias são estocásticas (a floresta
    sorteia features em cada divisão, o `p` sorteia rótulos) e sem ela ninguém as reproduz.
    """
    acima = frase_contada(
        classificacao.n_acima_do_acaso,
        nenhuma="nenhuma classe ficou acima do acaso",
        uma="1 classe ficou acima do nível do acaso",
        algumas="classes ficaram acima do nível do acaso",
    )
    return Recorte(
        desenhadas=n_features_usadas,
        disponiveis=denominador_ou_recusa(n_features_totais, de="o classificador"),
        amostragem=amostragem,
        matriz=matriz,
        metodo=(
            f"{classificacao.nome} sobre as {milhar(classificacao.n_amostras)} amostras de estudo, "
            f"em {milhar(classificacao.n_classes)} classes, com semente "
            f"{milhar(classificacao.semente)}. "
            f"{frase_da_validacao_do_classificador(classificacao)}. "
            f"{acima}. {veredito_do_classificador(classificacao)}"
        ),
    )


def _exige_o_nome(classificacao: Classificacao, esperado: str) -> None:
    """Recusa a carga de outro classificador.

    Os dois construtores recebem o mesmo tipo e desenham o mesmo traço; só o nome do título os
    distingue. Uma carga trocada sairia bem formada com o método errado e os números de outro
    modelo.
    """
    if classificacao.nome != esperado:
        raise ClassificacaoDesalinhadaError(
            f"esta figura é do {esperado!r} e a classificação chegou com nome "
            f"{classificacao.nome!r}. "
            "Os dois construtores desenham o mesmo traço, então uma carga trocada sairia bem "
            "formada com o método errado no título e os números de outro modelo ao lado"
        )


def _figura_do_classificador(
    *,
    classificacao: Classificacao,
    recorte: Recorte,
    estilos: EstilosDosGrupos | None = None,
) -> Figure:
    """Uma barra de recall por classe, com a régua do acaso. É o desenho dos dois construtores.

    Desenha a população inteira que recebe. A cor de cada classe vem de
    `series.series_por_grupo`.

    A régua do acaso é desenhada sempre, mesmo quando nenhuma barra a alcança: barras baixas sem
    régua parecem figura quebrada, e com régua leem-se como 'o modelo não separa'.

    O eixo vai de 0 a 1 fixo. Recall é fração, e a extensão automática faria barras baixas
    parecerem altas.
    """
    posicoes = np.arange(classificacao.n_classes, dtype=float)
    alturas = np.asarray(classificacao.recall, dtype=float)
    figura = nova_figura()
    eixo = figura.add_subplot(1, 1, 1)

    # `series_por_grupo` recebe o vetor por amostra, então ele é reconstruído das contagens.
    por_amostra = [
        nome
        for nome, quantas in zip(classificacao.grupos, classificacao.contagens, strict=True)
        for _ in range(quantas)
    ]
    # Com `estilos`, a barra de cada classe leva a cor escolhida para o grupo. Só a cor: barra
    # não tem marcador.
    series = series_por_grupo(por_amostra, estilos=estilos) if por_amostra else ()
    cores = {serie.grupo: serie.cor for serie in series}

    maos: list[Artist] = []
    for indice, nome in enumerate(classificacao.grupos):
        altura = alturas[indice]
        if not math.isfinite(altura) or altura <= 0.0:
            # Classe sem recall desenhável não vira barra de altura zero: o segmento nulo teria
            # dois vértices coincidentes, que contam como pontos desenhados e ninguém vê. O
            # rodapé conta quantas classes ficaram acima do acaso.
            continue
        maos.append(
            eixo.vlines(
                posicoes[indice],
                0.0,
                float(altura),
                colors=cores.get(nome),
                linewidth=LARGURA_DA_BARRA_DA_CLASSE,
                label=f"{nome} — {milhar(classificacao.contagens[indice])} amostras",
            )
        )

    regua = eixo.axhline(
        classificacao.acaso,
        linestyle="--",
        linewidth=1.0,
        label=rotulo_do_acaso(classificacao),
    )
    maos.append(regua)

    eixo.set_xticks(posicoes.tolist())
    eixo.set_xticklabels(list(classificacao.grupos), fontsize=TAMANHO_DA_FONTE)
    eixo.set_xlabel("classe")
    eixo.set_ylabel("recall fora da amostra")
    eixo.set_title(
        f"{classificacao.nome} · acurácia balanceada = "
        f"{_numero_ou_ausente(classificacao.balanceada)} · "
        f"p de permutação = {_numero_ou_ausente(classificacao.p)}",
        fontsize=TAMANHO_DA_FONTE,
    )
    # `vlines` não ajusta os limites dos eixos; sem isto a figura sai em branco.
    eixo.set_xlim(-0.6, float(classificacao.n_classes) - 0.4)
    eixo.set_ylim(0.0, 1.0)
    eixo.grid(visible=True, axis="y", linewidth=0.4, alpha=0.4)
    eixo.set_axisbelow(True)
    legenda_fora(eixo, maos)
    aplica(figura, recorte)
    return figura


def figura_do_random_forest(
    *,
    classificacao: Classificacao,
    recorte: Recorte,
    estilos: EstilosDosGrupos | None = None,
) -> Figure:
    """O recall por classe da floresta aleatória, com o `p` de permutação no título.

    Recusa uma carga de SVM.
    """
    _exige_o_nome(classificacao, NOME_DA_FLORESTA)
    return _figura_do_classificador(classificacao=classificacao, recorte=recorte, estilos=estilos)


def figura_do_svm(
    *,
    classificacao: Classificacao,
    recorte: Recorte,
    estilos: EstilosDosGrupos | None = None,
) -> Figure:
    """O recall por classe da máquina de vetores de suporte, com o `p` de permutação no título.

    Recusa uma carga da floresta.
    """
    _exige_o_nome(classificacao, NOME_DO_SVM)
    return _figura_do_classificador(classificacao=classificacao, recorte=recorte, estilos=estilos)
