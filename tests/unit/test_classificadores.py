"""Os dois classificadores: o par que impede o número de enganar, e o `p` que o sustenta.

## O que este módulo afirma, e o que ele deliberadamente NÃO afirma

**Nenhuma asserção de tempo de parede.** O custo está declarado em
`AnalysisSpec.classifier_permutations`, medido pelo arnês de `out/` sobre o corpus real, e
comparação de tempo de parede escrita à mão reprova sob contenção sem distinguir regressão de
máquina ocupada. O que fica aqui é a âncora de que o **padrão** não derivou sem ninguém
o remedir.

**Os dois números do classificador são CAMPOS do contrato desde 18/09/2026 — item 80.** Eram
constantes deste módulo, e o que mudou para estes testes é que o número deixa de ser passado à
função e passa a viajar no `AnalysisSpec` que ela já recebia: uma fonte só.

**Nenhuma asserção sobre um número absoluto de acurácia no corpus real.** Ele é resultado do
estudo, não contrato: os números medidos vivem na marca do commit e na sonda do congelado.

## O par que dá sentido a tudo o resto

*"Dado separado por construção: desempenho alto"* é verdade sobre quase qualquer implementação —
inclusive sobre uma que devolvesse a acurácia **dentro** da amostra, que é o defeito que esta
família existe para não ter. O que separa é o par: a **mesma** chamada sobre dado sem sinal tem de
cair no nível do acaso, e o `p` tem de subir com ele. As duas metades estão no mesmo teste, e é a
segunda que faz a primeira significar alguma coisa.

## A planta da medida cai sobre dado em que as DUAS acurácias diferem

*"A medida reportada não é acurácia simples"* seria vácuo sobre classes equilibradas, onde as duas
coincidem por construção. A planta é um preditor **degenerado** — responde sempre a classe maior —
sobre `30/10`: a simples dá `0,75` e a balanceada `0,50`. As duas divergem, e a asserção separa.
"""

from __future__ import annotations

import ast
import re
from collections.abc import Sequence
from dataclasses import MISSING, fields, replace
from pathlib import Path

import numpy as np
import pytest
from joblib import effective_n_jobs

from refino.domain.analysis import PADRAO_DA_ANALISE, AnalysisSpec
from refino.domain.cancellation import CancelCheck, ExecutionCancelledError
from refino.domain.labels import InvalidLabelError
from refino.domain.matrix import RefinoMatrix
from refino.domain.metadata import SampleMetadata, SampleType
from refino.statistics.classificadores import (
    IDENTIFICADOR_DO_CLASSIFICADOR,
    MIN_CLASSES,
    NOME_DO_CLASSIFICADOR,
    SERIE_DO_CLASSIFICADOR,
    TIPO_DA_IMPORTANCIA,
    ClassificacaoImpossivelError,
    ClassifierResult,
    _desempenho,
    classify,
    estimador_do_classificador,
)

REPO = Path(__file__).resolve().parents[2]
FONTE = REPO / "src" / "refino" / "statistics" / "classificadores.py"

# Poucas permutações de propósito: o `p` que estes testes leem é da forma `(k+1)/(n+1)`, e o que se
# afirma é a ORDENAÇÃO entre dois desfechos — não o valor. Duzentas permutações aqui trocariam
# segundos por nada.
#
# **O número viaja no CONTRATO e já não num `n_permutations=` da chamada — item 80.** A diferença
# não é de estilo: o valor passa pela guarda de `AnalysisSpec.__post_init__` em vez de ao lado dela,
# e deixa de haver duas fontes para a mesma quantidade.
PERMUTACOES_DO_TESTE = 9


# ---------------------------------------------------------------------------
# As fixturas sintéticas
# ---------------------------------------------------------------------------


def matriz_de(
    valores: np.ndarray,
    *,
    grupos: Sequence[str],
    tipos: Sequence[SampleType] | None = None,
) -> RefinoMatrix:
    """Uma `RefinoMatrix` sintética, com grupo e tipo por linha.

    **`tipos` é parâmetro e não constante: o valor decide quanto do caminho corre.** A população
    desta família é a `sample_mask`, e um arnês que passasse sempre `sample` nunca exercitaria a
    metade que separa *"o QC é a régua"* de *"o QC é um grupo"*.
    """
    array = np.ascontiguousarray(valores, dtype=np.float64)
    array.flags.writeable = False
    n_amostras, n_features = array.shape
    rotulos = list(grupos)
    kinds: list[SampleType] = list(tipos) if tipos is not None else ["sample"] * n_amostras
    amostras = tuple(
        SampleMetadata(
            name=f"s{i:03d}",
            injection_order=i + 1,
            sample_type=kinds[i],
            group=rotulos[i],
        )
        for i in range(n_amostras)
    )
    return RefinoMatrix(
        values=array,
        feature_names=tuple(f"f{j:03d}" for j in range(n_features)),
        samples=amostras,
        mode="sintetico",
    )


def separadas(*, por_classe: int = 8, n_features: int = 12, seed: int = 20260918) -> RefinoMatrix:
    """Duas classes separadas por construção — `A` em torno de 0, `B` em torno de 4."""
    rng = np.random.default_rng(seed)
    valores = np.vstack(
        [
            rng.normal(0.0, 1.0, (por_classe, n_features)),
            rng.normal(4.0, 1.0, (por_classe, n_features)),
        ]
    )
    return matriz_de(valores, grupos=["A"] * por_classe + ["B"] * por_classe)


def sem_sinal(*, por_classe: int = 8, n_features: int = 12, seed: int = 20260918) -> RefinoMatrix:
    """As MESMAS dimensões e os mesmos rótulos, sobre ruído puro. A metade que separa."""
    rng = np.random.default_rng(seed)
    valores = rng.normal(0.0, 1.0, (2 * por_classe, n_features))
    return matriz_de(valores, grupos=["A"] * por_classe + ["B"] * por_classe)


CONTRATO = AnalysisSpec(classifier_folds=4, classifier_permutations=PERMUTACOES_DO_TESTE)
"""O contrato destes testes: quatro dobras e nove sorteios, os dois pedidos e não herdados."""

UM_SORTEIO = replace(CONTRATO, classifier_permutations=1)
"""Para os testes que NÃO leem o `p` — o laço das permutações corre uma vez e não nove.

O `p` continua a ser calculado, porque a regra 6 do README não admite caminho que o salte; o que
muda é a resolução, que estes testes não afirmam.
"""


# ---------------------------------------------------------------------------
# 1. A regra 6 do README — a bicondicional, lida por introspecção
# ---------------------------------------------------------------------------


def test_o_resultado_NAO_admite_a_medida_sem_o_p_de_permutacao() -> None:  # noqa: N802
    """Não existe `ClassifierResult` construído com a medida presente e o `p` ausente.

    **É o mecanismo de `PLSDAResult` e de `OneVsRestModel`, e a verificação é a mesma:** um
    dataclass congelado **sem valor padrão nenhum** não admite construção parcial, e ler isso é ler
    a declaração. O teste lê-a por introspecção em vez de por confiança — um padrão acrescentado a
    `permutation_pvalue` faria a bicondicional cair sem que nada mais na suíte mudasse.

    **A âncora de não-vacuidade está NA MESMA função e ANTES**: uma classe sem campos
    satisfaria o laço abaixo por conjunto vazio.
    """
    campos = fields(ClassifierResult)
    assert len(campos) >= 10, (
        f"`ClassifierResult` tem {len(campos)} campos — a asserção seria vácua"
    )

    nomes = {c.name for c in campos}
    assert {"permutation_pvalue", "balanced_accuracy", "accuracy", "cv_folds"} <= nomes, nomes

    com_padrao = sorted(
        c.name for c in campos if c.default is not MISSING or c.default_factory is not MISSING
    )
    assert com_padrao == [], (
        f"estes campos de `ClassifierResult` ganharam valor padrão: {com_padrao}. Com um padrão em "
        "qualquer um deles a construção parcial passa a ser possível, e a regra 6 do README "
        "deixa de ser propriedade da declaração para ser cuidado de quem chama"
    )
    assert ClassifierResult.__dataclass_params__.frozen, (
        "`ClassifierResult` deixou de ser congelado: um `p` mutável depois de lido não é o número "
        "que a figura declara"
    )


def test_o_p_e_a_medida_SAEM_JUNTOS_de_uma_execucao_real() -> None:  # noqa: N802
    """A metade de runtime da bicondicional: o caminho de produção devolve os dois preenchidos.

    A introspecção acima prova que a construção parcial é impossível; esta prova que o caminho que
    de facto corre **constrói** o objeto — as duas juntas exercitam a verificação nas duas direções.
    """
    resultado = classify(separadas(), CONTRATO, kind="svm", seed=7)
    assert np.isfinite(resultado.balanced_accuracy)
    assert np.isfinite(resultado.permutation_pvalue)
    assert resultado.n_permutations == PERMUTACOES_DO_TESTE
    assert resultado.permutation_pvalue >= 1.0 / (PERMUTACOES_DO_TESTE + 1), (
        "o `p` ficou abaixo de `1/(n+1)`, que é o piso da convenção `(k+1)/(n+1)` — ou a fórmula "
        "mudou, ou o `n` declarado não é o que correu"
    )


# ---------------------------------------------------------------------------
# 2. O par que impede o classificador de enganar
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["random_forest", "svm"])
def test_dado_separado_sobe_e_dado_SEM_SINAL_cai_ao_nivel_do_acaso(kind: str) -> None:  # noqa: N802
    """O par causal da família inteira, com as duas metades na mesma função.

    **A segunda metade é a que dá sentido à primeira.** Uma implementação que medisse **dentro** da
    amostra passaria na primeira com folga — com `p >> n` sempre há um hiperplano que separa — e
    reprovaria aqui, porque acertaria o ruído também. As duas matrizes têm as **mesmas** dimensões e
    os **mesmos** rótulos: o que muda é só haver sinal.

    **O `p` é a segunda coordenada do par**, e ele é o que a regra 6 do README exige ao lado da
    medida: separação alta com `p` alto é o falso positivo que esta família existe para tornar
    visível.
    """
    com_sinal = classify(separadas(), CONTRATO, kind=kind, seed=11)
    sem = classify(sem_sinal(), CONTRATO, kind=kind, seed=11)

    assert com_sinal.chance_level == pytest.approx(0.5), com_sinal.chance_level
    assert com_sinal.balanced_accuracy > 0.9, (
        f"{kind}: dado separado por construção deu {com_sinal.balanced_accuracy:.4f} de acurácia "
        "balanceada. Duas nuvens a quatro desvios não se confundem fora da amostra"
    )
    assert sem.balanced_accuracy < 0.75, (
        f"{kind}: ruído puro deu {sem.balanced_accuracy:.4f} de acurácia balanceada. Acima disto "
        "a medida está a ler dentro da amostra, que é o defeito que esta família existe para não "
        "ter"
    )
    assert com_sinal.balanced_accuracy > sem.balanced_accuracy, (
        f"{kind}: a matriz com sinal não se distingue da sem sinal — as duas têm as mesmas "
        "dimensões e os mesmos rótulos, então este par não separa nada e as asserções acima são "
        "coincidência"
    )
    assert com_sinal.permutation_pvalue < sem.permutation_pvalue, (
        f"{kind}: o `p` do dado separado ({com_sinal.permutation_pvalue:.4f}) não é menor que o do "
        f"ruído ({sem.permutation_pvalue:.4f}). O `p` é a coordenada que sustenta a medida, e se "
        "ele não distingue os dois casos, ele não está a medir a nula"
    )


# ---------------------------------------------------------------------------
# 3. A medida reportada NÃO é acurácia simples
# ---------------------------------------------------------------------------


def test_a_planta_do_DESBALANCO_separa_as_duas_acuracias() -> None:  # noqa: N802
    """O preditor degenerado: a simples premeia, a balanceada não. **As duas DIFEREM.**

    Esta planta cai sobre dado em que as duas coisas que ela separa diferem, e é construída
    em vez de sorteada: sobre classes equilibradas
    as duas medidas coincidem por definição, e um teste ali seria vácuo.

    As proporções são as do corpus, encolhidas: o desbalanço real das amostras de estudo do
    `pos_hilic` é `18/16/14/13`, e um classificador que responda sempre a maior delas dá `0,295` de
    simples contra `0,250` de balanceada.
    """
    verdadeiro = np.array(["A"] * 30 + ["B"] * 10)
    # O degenerado: responde sempre a classe maior.
    predito = np.array(["A"] * 40)

    medido = _desempenho(verdadeiro, predito, ("A", "B"))
    assert medido.simples == pytest.approx(0.75), medido.simples
    assert medido.balanceada == pytest.approx(0.50), medido.balanceada
    assert medido.simples != pytest.approx(medido.balanceada), (
        "as duas acurácias coincidem nesta planta: ela não separa, e a asserção acima não prova "
        "que a medida reportada é a balanceada"
    )
    # E `0,50` é **exactamente** o nível do acaso de duas classes — a propriedade que torna a
    # balanceada legível, e que a simples não tem.
    assert medido.balanceada == pytest.approx(1.0 / 2)
    assert medido.recall == pytest.approx((1.0, 0.0))


def test_o_oraculo_do_desempenho_reproduzido_A_MAO(  # noqa: N802
) -> None:
    """Reproduzir o oráculo à mão sobre um caso cuja resposta já se sabe.

    Quatro amostras, três classes, uma confusão escrita à mão. Se `_desempenho` estiver a ler a
    matriz transposta — o defeito clássico de uma matriz de confusão —, o *recall* e a precisão
    trocam de lugar e esta tabela acusa.
    """
    verdadeiro = np.array(["A", "A", "B", "C"])
    predito = np.array(["A", "B", "B", "A"])

    medido = _desempenho(verdadeiro, predito, ("A", "B", "C"))
    # linha = verdadeiro, coluna = predito
    assert medido.confusao == ((1, 1, 0), (0, 1, 0), (1, 0, 0))
    assert medido.recall == pytest.approx((0.5, 1.0, 0.0))
    assert medido.balanceada == pytest.approx((0.5 + 1.0 + 0.0) / 3)
    assert medido.simples == pytest.approx(2 / 4)


def test_o_nivel_do_acaso_e_UM_SOBRE_K_e_nao_um_limiar(  # noqa: N802
) -> None:
    """`chance_level` é derivado da contagem de classes, e nada o calibra.

    **Par causal sobre a POPULAÇÃO e não sobre o número:** duas matrizes com número de classes
    diferente têm de dar níveis diferentes. Um `0.5` fixo passaria no caso de duas classes e
    reprovaria aqui.
    """
    duas = classify(separadas(), UM_SORTEIO, kind="svm", seed=3)
    rng = np.random.default_rng(5)
    tres = matriz_de(rng.normal(0.0, 1.0, (18, 10)), grupos=["A"] * 6 + ["B"] * 6 + ["C"] * 6)
    resultado = classify(tres, UM_SORTEIO, kind="svm", seed=3)

    assert duas.chance_level == pytest.approx(1 / 2)
    assert resultado.chance_level == pytest.approx(1 / 3)
    assert duas.chance_level != pytest.approx(resultado.chance_level), (
        "os dois níveis do acaso coincidem: `chance_level` não está a derivar do número de classes"
    )


# ---------------------------------------------------------------------------
# 4. Determinismo — regra 3 do README
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["random_forest", "svm"])
def test_a_mesma_semente_reproduz_e_a_semente_VIAJA(kind: str) -> None:  # noqa: N802
    """Duas execuções com a mesma entrada e semente dão saída idêntica, e a semente sai no objeto.

    **A semente que viaja é a DERIVADA e não a global**, e o teste afirma-o: é ela que reproduz, e é
    ela que a figura declara. Publicar a global faria quem recebe a figura reproduzir outra coisa.
    """
    primeira = classify(separadas(), CONTRATO, kind=kind, seed=42)
    segunda = classify(separadas(), CONTRATO, kind=kind, seed=42)
    assert primeira == segunda

    assert primeira.seed != 42, (
        "a semente publicada é a global: ela tem de ser a derivada `(seed, unidade)`, que é a que "
        "de facto correu — sem isso quem recebe a figura reproduz outra coisa"
    )


def test_os_DOIS_classificadores_consomem_fluxos_DIFERENTES() -> None:  # noqa: N802
    """Os dois correm na mesma execução com a mesma `seed`, e não podem partilhar a sequência.

    **Na escala de duas famílias:** com a semente global crua, os dois consumiriam os
    mesmos embaralhamentos e os dois p-valores deixariam de ser sorteios independentes — sem nada a
    acusar, porque as duas execuções seriam consistentes entre si.

    A âncora de não-vacuidade está na mesma função: os dois identificadores têm de ser distintos
    **antes** de a derivação poder diferir.
    """
    assert len(set(IDENTIFICADOR_DO_CLASSIFICADOR.values())) == 2, IDENTIFICADOR_DO_CLASSIFICADOR

    floresta = classify(separadas(), UM_SORTEIO, kind="random_forest", seed=42)
    maquina = classify(separadas(), UM_SORTEIO, kind="svm", seed=42)
    assert floresta.seed != maquina.seed, (
        f"os dois derivaram a mesma semente ({floresta.seed}) da mesma global: os fluxos colidem, "
        "e um passa a ser previsível a partir do outro"
    )


# ---------------------------------------------------------------------------
# 5. Cancelamento na assinatura do DOMÍNIO
# ---------------------------------------------------------------------------


class _CancelaNaChamada:
    """Devolve `True` a partir da n-ésima consulta.

    **Conta as consultas, e é isso que ela mede.**
    """

    def __init__(self, a_partir_de: int) -> None:
        self.consultas = 0
        self._a_partir_de = a_partir_de

    def __call__(self) -> bool:
        self.consultas += 1
        return self.consultas >= self._a_partir_de


@pytest.mark.parametrize("kind", ["random_forest", "svm"])
def test_o_cancelamento_interrompe_DURANTE_o_treino(kind: str) -> None:  # noqa: N802
    """O verificador está na assinatura do domínio e é consultado **dentro** do laço.

    **O par causal é sobre ONDE a consulta acontece, e não sobre haver consulta.** Um `cancelled`
    consultado uma vez, antes de tudo, passaria num teste que só cancelasse na primeira chamada —
    e deixaria o profissional a esperar minutos depois de clicar «Cancelar». A planta cancela na
    **quinta** consulta, que é depois do início, e a asserção afirma que o levantamento aconteceu
    mesmo assim.
    """
    vigia = _CancelaNaChamada(a_partir_de=5)
    cancelado: CancelCheck = vigia
    with pytest.raises(ExecutionCancelledError):
        classify(
            separadas(),
            CONTRATO,
            kind=kind,
            seed=1,
            cancelled=cancelado,
        )
    assert vigia.consultas >= 5, (
        f"o cancelamento foi consultado {vigia.consultas} vez(es) antes de levantar: com menos de "
        "cinco, ele não chegou ao laço e a planta mediu a guarda de entrada"
    )


def test_sem_cancelamento_a_execucao_COMPLETA() -> None:  # noqa: N802
    """A outra direção do par: o mesmo caminho, sem a planta, chega ao fim.

    Sem isto, um `classify` que levantasse sempre passaria no teste acima.
    """
    resultado = classify(
        separadas(), replace(CONTRATO, classifier_permutations=2), kind="svm", seed=1
    )
    assert resultado.n_samples == 16


# ---------------------------------------------------------------------------
# 6. O PARALELISMO, afirmado em RUNTIME
# ---------------------------------------------------------------------------


def test_os_estimadores_NAO_pedem_mais_de_UM_trabalhador_em_runtime() -> None:  # noqa: N802
    """A afirmação é sobre o OBJETO que o módulo constrói, não sobre o texto do diff.

    **`JOBLIB_MULTIPROCESSING=0` já não existe nesta árvore**, e perdeu-se
    com ela isto: o interruptor global protegia por abrangência, e a rede que o substituiu protege
    por
    **enumeração**. O que este módulo faz é não pedir o que não pode ser servido, e é isso que aqui
    se mede.

    **A âncora de não-vacuidade está NA MESMA função e é a segunda asserção**: numa máquina
    de um núcleo `effective_n_jobs(-1)` devolve `1`, e a primeira asserção passaria sem separar
    nada. Com a âncora, essa máquina reprova em vez de certificar.
    """
    floresta = estimador_do_classificador("random_forest", CONTRATO, seed=1)
    assert floresta.n_jobs == SERIE_DO_CLASSIFICADOR
    assert effective_n_jobs(floresta.n_jobs) == 1, (
        f"o `joblib` resolve `n_jobs={floresta.n_jobs}` para {effective_n_jobs(floresta.n_jobs)} "
        "trabalhadores. Acima de um, o `loky` gera processo — e num congelado cada processo "
        "relança o `refino.exe`, que são as catorze janelas de 26/08/2026"
    )
    assert effective_n_jobs(-1) > 1, (
        "âncora: nesta máquina `effective_n_jobs(-1)` já devolve 1, então a asserção acima passa "
        "sem separar nada. Num núcleo só este portão é vácuo, e reprovar é o desfecho certo"
    )

    # O SVM **não tem** eixo de paralelismo — `SVC` não aceita `n_jobs`. Afirmar a ausência é o que
    # impede alguém de lhe acrescentar um sem passar por aqui.
    maquina = estimador_do_classificador("svm", CONTRATO, seed=1)
    assert not hasattr(maquina, "n_jobs"), (
        "o `SVC` ganhou um eixo `n_jobs`: ele passa a poder gerar processo, e a conjunção de "
        "`n_jobs_efetivo` tem de o alcançar antes de esta linha sair"
    )


def test_o_portao_do_paralelismo_ACUSA_um_menos_um_plantado() -> None:  # noqa: N802
    """O par causal do portão de forma, **em processo**: a planta é o texto, e o detector muda.

    Sem esta função, o detector abaixo seria indistinguível de um que não consegue ver a violação:
    a árvore limpa não a tem, por construção.
    """
    fonte = FONTE.read_text(encoding="utf-8")
    assert _pedidos_de_paralelismo(fonte) == [], (
        f"`classificadores.py` pede mais de um trabalhador em {_pedidos_de_paralelismo(fonte)}"
    )

    plantado = fonte.replace("n_jobs=SERIE_DO_CLASSIFICADOR", "n_jobs=-1")
    assert plantado != fonte, "a planta não aterrou: o texto que ela substitui mudou de forma"
    assert _pedidos_de_paralelismo(plantado) == ["-1"], _pedidos_de_paralelismo(plantado)


def _pedidos_de_paralelismo(fonte: str) -> list[str]:
    """Todo `n_jobs=` deste módulo cujo valor não seja a constante de série, lido por AST.

    Por AST e não por `regex`: `n_jobs` aparece em prosa nesta fonte — o cabeçalho explica por que
    ele é `1` —, e um padrão de texto acusaria a docstring. É o gesto de
    contar antes de escolher, e escolher a forma que não acusa o legítimo.
    """
    achados: list[str] = []
    for no in ast.walk(ast.parse(fonte)):
        if not isinstance(no, ast.Call):
            continue
        for palavra in no.keywords:
            if palavra.arg != "n_jobs":
                continue
            if isinstance(palavra.value, ast.Name) and palavra.value.id == "SERIE_DO_CLASSIFICADOR":
                continue
            achados.append(ast.unparse(palavra.value))
    return achados


# ---------------------------------------------------------------------------
# 7. A população é a MÁSCARA, e as recusas nomeiam os dois números
# ---------------------------------------------------------------------------


def test_o_QC_fica_de_FORA_e_a_planta_mede_o_TIPO_e_nao_o_rotulo() -> None:  # noqa: N802
    """O par causal: tipar as mesmas linhas como `qc` muda o resultado.

    **A planta cai sobre dado em que as duas coisas DIFEREM**: as linhas de QC levam um rótulo de
    classe — `"A"` —, então uma implementação que lesse o **rótulo** em vez da **máscara** as
    contaria na classe `A` e o resultado sairia igual. Tipá-las como `qc` tem de mudar a contagem.
    """
    rng = np.random.default_rng(1)
    valores = rng.normal(0.0, 1.0, (20, 8))
    grupos = ["A"] * 10 + ["B"] * 10

    todas_estudo = matriz_de(valores, grupos=grupos)
    com_qc = matriz_de(valores, grupos=grupos, tipos=["qc"] * 3 + ["sample"] * 17)

    a = classify(todas_estudo, UM_SORTEIO, kind="svm", seed=1)
    b = classify(com_qc, UM_SORTEIO, kind="svm", seed=1)
    assert a.class_counts == (10, 10)
    assert b.class_counts == (7, 10), (
        f"com três linhas tipadas `qc` as contagens saíram {b.class_counts}: a seleção está a ler "
        "o RÓTULO e não a `sample_mask`, e o QC entrou como classe de estudo"
    )
    assert a.n_samples != b.n_samples, "a planta não separa: as duas populações têm o mesmo tamanho"


@pytest.mark.parametrize(
    ("construtor", "trecho"),
    [
        (
            lambda: matriz_de(np.zeros((6, 4)), grupos=["A"] * 6, tipos=["qc"] * 6),
            "nenhuma das 6 injeções é amostra de estudo",
        ),
        (
            lambda: matriz_de(np.arange(24.0).reshape(6, 4), grupos=["A"] * 6),
            f"Abaixo de {MIN_CLASSES} não há contraste",
        ),
        (
            lambda: matriz_de(np.arange(24.0).reshape(6, 4), grupos=["A"] * 3 + ["B"] * 3),
            "Dobras estratificadas exigem",
        ),
        (
            lambda: matriz_de(np.ones((10, 4)), grupos=["A"] * 5 + ["B"] * 5),
            "nenhuma das 4 features é utilizável",
        ),
    ],
)
def test_a_populacao_insuficiente_e_RECUSA_TIPADA_com_os_dois_numeros(  # noqa: N802
    construtor: object, trecho: str
) -> None:
    """As quatro recusas, cada uma nomeando a quantidade contra que o pedido bateu.

    **`ClassificacaoImpossivelError` e não `ValueError` nu**, e a distinção é o que permite à rota
    devolver 422 sem absorver o `ValueError` de um defeito de programação.
    """
    with pytest.raises(ClassificacaoImpossivelError, match=re.escape(trecho)):
        classify(construtor(), UM_SORTEIO, kind="svm", seed=1)  # type: ignore[operator]


def test_a_populacao_SA_nao_levanta() -> None:  # noqa: N802
    """A outra ponta: sem isto, um `classify` que recusasse tudo passaria nas quatro plantas."""
    resultado = classify(separadas(), UM_SORTEIO, kind="svm", seed=1)
    assert resultado.n_samples == 16
    assert resultado.groups == ("A", "B")


def test_um_rotulo_de_classificador_INVALIDO_levanta_na_fronteira() -> None:  # noqa: N802
    """`Literal` é análise estática, e este valor chega da CLI e da janela.

    **A guarda é necessária porque o despacho TEM um ramo final sem condição:** sem ela um rótulo
    inválido construiria o SVM em silêncio e `ClassifierResult.kind` declararia o nome que o
    chamador pediu — o modo de falha insidioso, o que não levanta.
    """
    with pytest.raises(InvalidLabelError):
        classify(separadas(), UM_SORTEIO, kind="xgboost", seed=1)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 8. Os dois números do CONTRATO, e as âncoras contra deriva silenciosa — item 80
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("campo", ["classifier_permutations", "classifier_trees"])
@pytest.mark.parametrize("valor", [0, -1])
def test_o_numero_NAO_POSITIVO_e_recusado_na_FRONTEIRA(campo: str, valor: int) -> None:  # noqa: N802
    """A guarda é do contrato, e `classify` **não** tem uma segunda cópia dela.

    **É o desenho de `test_ward_com_distancia_nao_euclidiana_e_recusado`, e a razão é a mesma:** um
    contrato que não se pode construir inválido é mais barato que duas cópias da mesma verificação,
    e os dois números chegam de `analise_json` do formulário e das caixas da janela — que é
    uma fronteira de confiança.

    **O par causal está na função irmã abaixo**, e sem ele isto seria a metade que nunca vê a
    guarda deixar de acusar.
    """
    with pytest.raises(ValueError) as capturado:
        AnalysisSpec(**{campo: valor})
    mensagem = str(capturado.value)
    assert campo in mensagem and "o mínimo é 1" in mensagem, mensagem


def test_o_numero_NO_MINIMO_e_aceite_e_o_PADRAO_tambem() -> None:  # noqa: N802
    """A outra direção: a guarda não está sempre a levantar.

    Sem esta função, um `__post_init__` que recusasse **todo** valor passaria no teste acima — o
    portão que sempre reprova.
    """
    minimo = AnalysisSpec(classifier_permutations=1, classifier_trees=1)
    assert minimo.classifier_permutations == 1 and minimo.classifier_trees == 1
    assert PADRAO_DA_ANALISE.classifier_permutations == 200
    assert PADRAO_DA_ANALISE.classifier_trees == 300


def test_os_numeros_do_CONTRATO_sao_os_que_os_estimadores_USAM() -> None:  # noqa: N802
    """O campo do contrato é lido pelo objeto construído — não por uma segunda cópia do valor.

    Um número declarado e não lido é prosa com aparência de contrato, e o sintoma
    seria a docstring do campo a descrever uma floresta que ninguém constrói.

    **A segunda metade é o valor que exercita o caminho, e ela é a que a versão anterior deste teste
    não podia ter.** Enquanto o número era constante de módulo, *"a floresta tem 300 árvores"* era a
    única afirmação possível. Agora o eixo tem valor, e o valor natural — o padrão — é o que executa
    menos do caminho: um arnês que só passasse `PADRAO_DA_ANALISE` deixaria por exercitar que o
    campo **chega** ao construtor, e um `n_estimators=300` literal passaria igual.
    """
    floresta = estimador_do_classificador("random_forest", PADRAO_DA_ANALISE, seed=99)
    assert floresta.n_estimators == PADRAO_DA_ANALISE.classifier_trees
    assert floresta.random_state == 99
    assert floresta.class_weight == "balanced"
    assert estimador_do_classificador("svm", PADRAO_DA_ANALISE, seed=99).kernel == "linear"

    # **O valor NÃO-PADRÃO, e a âncora de que ele difere do padrão vem antes**: com
    # `pedidas == PADRAO_DA_ANALISE.classifier_trees` a asserção seguinte passaria sobre um
    # construtor que ignorasse o contrato por completo.
    pedidas = 7
    assert pedidas != PADRAO_DA_ANALISE.classifier_trees, (
        "âncora: o valor plantado é igual ao padrão, então a asserção abaixo não separa um "
        "construtor que leia o contrato de um que devolva sempre o padrão"
    )
    magra = estimador_do_classificador(
        "random_forest", replace(PADRAO_DA_ANALISE, classifier_trees=pedidas), seed=99
    )
    assert magra.n_estimators == pedidas, (
        f"pedi {pedidas} árvores no contrato e a floresta veio com {magra.n_estimators}: o campo "
        "não chega ao construtor, e quem pedir menos vai continuar a pagar o padrão"
    )

    assert PADRAO_DA_ANALISE.classifier_permutations >= 100, (
        f"o padrão de `classifier_permutations` é {PADRAO_DA_ANALISE.classifier_permutations}: "
        "abaixo de cem o menor `p` obtenível passa de 0,0099, e um `p < 0,01` deixa de ser "
        "defensável para quem não pede nada"
    )
    assert (
        set(IDENTIFICADOR_DO_CLASSIFICADOR)
        == set(NOME_DO_CLASSIFICADOR)
        == set(TIPO_DA_IMPORTANCIA)
    ), "os três mapas por classificador deixaram de cobrir o mesmo conjunto de rótulos"


def test_o_numero_de_sorteios_DO_CONTRATO_e_o_que_o_laco_de_facto_CORRE() -> None:  # noqa: N802
    """O par causal do item 80: mudar o campo muda quantas vezes o laço corre.

    **Sem esta função, `classifier_permutations` podia estar a ser copiado para o resultado sem
    alcançar o laço.** `ClassifierResult.n_permutations` diria nove e a execução faria duzentas — o
    rótulo certo sobre outra quantidade, e o sintoma seria o portão de cada
    commit continuar a custar o mesmo depois desta mudança inteira.

    **A contagem é de EXECUÇÕES do código sob teste e não de testes que passam**:
    `_p_de_permutacao` consulta o cancelamento uma vez por sorteio e `_predicoes_fora_da_dobra` uma
    vez por dobra, e nenhuma das duas é privada ao ponto de exigir monkeypatch — a contagem
    observa-se de fora, pelo verificador que a assinatura já aceita.

    A forma fechada é `1 + dobras + n x (1 + dobras)`, então a diferença entre dois `n` é
    `(n2 - n1) x (1 + dobras)` — afirmada exactamente, e não como desigualdade.
    """

    def consultas_com(n: int) -> int:
        vistas = 0

        def vigia() -> bool:
            nonlocal vistas
            vistas += 1
            return False

        resultado = classify(
            separadas(),
            replace(CONTRATO, classifier_permutations=n),
            kind="svm",
            seed=1,
            cancelled=vigia,
        )
        assert resultado.n_permutations == n, (
            f"o resultado declara {resultado.n_permutations} sorteios e o contrato pediu {n}"
        )
        return vistas

    uma, nove = consultas_com(1), consultas_com(9)
    dobras = CONTRATO.classifier_folds
    assert uma == 1 + dobras + 1 * (1 + dobras), (
        f"com um sorteio o cancelamento foi consultado {uma} vezes e a forma fechada dá "
        f"{1 + dobras + 1 * (1 + dobras)}: ou o laço não corre, ou há consulta a mais"
    )
    assert nove - uma == 8 * (1 + dobras), (
        f"oito sorteios a mais custaram {nove - uma} consultas e deviam custar "
        f"{8 * (1 + dobras)}. O campo do contrato não governa o laço"
    )


@pytest.mark.parametrize("kind", ["random_forest", "svm"])
def test_a_importancia_VIAJA_com_o_tipo_dela(kind: str) -> None:  # noqa: N802
    """As importâncias vão para o resumo tipado — e nunca sob a palavra «importância» só.

    **A da floresta e a do SVM não são a mesma quantidade**, e um vetor sem o tipo ao lado convida
    à comparação que não se pode fazer. O teste afirma o alinhamento com as features e
    que o tipo é o do classificador que correu.
    """
    resultado = classify(separadas(), UM_SORTEIO, kind=kind, seed=1)
    assert len(resultado.importances) == len(resultado.feature_names)
    assert resultado.importance_kind == TIPO_DA_IMPORTANCIA[kind]
    assert TIPO_DA_IMPORTANCIA["random_forest"] != TIPO_DA_IMPORTANCIA["svm"], (
        "os dois tipos de importância têm a mesma frase: a asserção acima não distingue os dois "
        "classificadores"
    )
