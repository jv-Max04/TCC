"""A tabela de recuperação dos dois `knn`, e ela corre com TRÊS braços.

## O que se mede, e por que a tabela é o desfecho

Verdade fundamental construída, ausência injetada com política declarada, e o erro medido
**apenas nas células que foram escondidas** — que é a quantidade que a pergunta pede. Medir
sobre a matriz inteira diluiria o erro da imputação nas células que ninguém tocou, e o
número resultante mediria sobretudo quantas células ficaram intactas.

**Nenhum braço precisa de vencer.** O critério foi pré-registrado antes de a implementação
existir, e o desfecho é a tabela: qual método recupera melhor sob MNAR e sob MCAR é achado, e
declarar de antemão quem deve ganhar transformaria a medição em ajuste.

## Por que TRÊS braços e não dois

`knn` e `knn_features` diferem em **duas** coisas ao mesmo tempo: quem é o vizinho, e quantos
vizinhos há — porque a regra `round(sqrt(n_colunas))` conta colunas, e transpor troca o
que são as colunas. Com 61 amostras e milhares de features os dois `k` ficam a uma ordem de
grandeza de distância.

> **Com diferença de implementação e de parâmetro ao mesmo tempo, nenhuma divergência é
> atribuível**.

Então:

| Braço | Vizinho | `k` |
|---|---|---|
| `feature_k_igual` | outra feature | o mesmo dos dois primeiros |
| `amostra_k_igual` | outra amostra | o mesmo dos dois primeiros |
| `amostra_k_producao` | outra amostra | a regra do imputador, que é o que o usuário obtém |

Os dois primeiros isolam a transposição. O terceiro é o que de facto corre em produção, e sem
ele a tabela descreveria uma configuração que ninguém usa.
"""

from __future__ import annotations

import numpy as np
import pytest

from refino.domain.labels import VOCABULARY
from refino.domain.matrix import RefinoMatrix
from refino.domain.metadata import SampleMetadata
from refino.preprocessing.missing import _imputa_por_modelo, impute, is_missing

pytestmark = pytest.mark.synthetic

N_AMOSTRAS = 24
N_FEATURES = 40
SEMENTE = 20260910


def _verdade(seed: int = SEMENTE) -> np.ndarray:
    """Matriz com estrutura em ambos os eixos, porque os dois braços exploram um eixo cada.

    Um dos métodos procura amostras parecidas e o outro features parecidas. Uma matriz de
    ruído puro não teria nem uma coisa nem outra, e os dois braços recuperariam igualmente
    mal — a tabela sairia plana por construção do fixture e não por propriedade do método,
    que é a planta que não separa.

    A construção é de posto baixo mais ruído: quatro fatores latentes ligam as features entre
    si, e as cargas por amostra ligam as amostras entre si.
    """
    rng = np.random.default_rng(seed)
    cargas = rng.normal(loc=0.0, scale=1.0, size=(N_AMOSTRAS, 4))
    fatores = rng.normal(loc=0.0, scale=1.0, size=(4, N_FEATURES))
    base = cargas @ fatores
    escala = rng.uniform(500.0, 5000.0, size=N_FEATURES)
    valores = escala * (1.0 + 0.25 * base) + rng.normal(0.0, 20.0, size=(N_AMOSTRAS, N_FEATURES))
    return np.clip(valores, 1.0, None)


def _matriz(valores: np.ndarray) -> RefinoMatrix:
    amostras = tuple(
        SampleMetadata(
            name=f"S{i}",
            injection_order=i + 1,
            sample_type="qc" if i % 8 == 0 else "sample",
            group="QC" if i % 8 == 0 else "G",
        )
        for i in range(valores.shape[0])
    )
    return RefinoMatrix(
        values=valores,
        feature_names=tuple(f"f{j}" for j in range(valores.shape[1])),
        samples=amostras,
        mode="sintetico",
    )


def _mascara_mnar(verdade: np.ndarray, fracao: float, seed: int) -> np.ndarray:
    """Ausência no **lado baixo de cada feature** — o caso dominante nestes dados.

    Zeros não são zeros: o equipamento deixa a célula vazia quando não detecta, e o não
    detectado está abaixo do limite. Uma máscara sorteada ao acaso mediria outra pergunta, e é
    por isso que a de MCAR existe **ao lado** e não no lugar.

    **O limiar acompanha a fração pedida, e a primeira versão desta função não o fazia.** Ela
    cortava no decil fixo, então pedir 25% escondia 12,7% — o rótulo dizia uma população e a
    medida era sobre outra. Com o corte no quantil da
    própria fração, o número de candidatas por feature é o pedido, e o que a tabela reporta
    passa a ser o que o nome dela diz.
    """
    rng = np.random.default_rng(seed)
    mascara = np.zeros(verdade.shape, dtype=bool)
    alvo = round(fracao * verdade.size)
    limiar = np.quantile(verdade, fracao, axis=0)
    candidatas = np.argwhere(verdade <= limiar)
    if candidatas.shape[0] > alvo:
        escolhidas = rng.choice(candidatas.shape[0], size=alvo, replace=False)
        candidatas = candidatas[escolhidas]
    mascara[candidatas[:, 0], candidatas[:, 1]] = True
    return mascara


def _mascara_mcar(verdade: np.ndarray, fracao: float, seed: int) -> np.ndarray:
    """Ausência completamente ao acaso — o controle.

    Sem ela, uma vantagem medida sob MNAR seria indistinguível de uma vantagem geral do
    método, e a conclusão publicada teria escopo maior que a medição.
    """
    rng = np.random.default_rng(seed)
    return rng.random(verdade.shape) < fracao


def _com_ausencia(verdade: np.ndarray, mascara: np.ndarray) -> np.ndarray:
    """Aplica a máscara na convenção do projeto: a célula não detectada fica **zero**."""
    valores = verdade.copy()
    valores[mascara] = 0.0
    return valores


def _erro_nas_escondidas(imputada: np.ndarray, verdade: np.ndarray, mascara: np.ndarray) -> float:
    """Erro relativo mediano **só nas células escondidas**, que é o que a pergunta pede."""
    assert mascara.any(), "máscara vazia: o erro seria medido sobre nada"
    relativo = np.abs(imputada[mascara] - verdade[mascara]) / verdade[mascara]
    return float(np.median(relativo))


def _tres_bracos(verdade: np.ndarray, mascara: np.ndarray) -> dict[str, float]:
    """A tabela. Os dois primeiros braços diferem SÓ pela transposição."""
    valores = _com_ausencia(verdade, mascara)
    ausente = is_missing(valores)
    k_comum = max(1, round(float(np.sqrt(verdade.shape[0]))))

    por_feature = _imputa_por_modelo(
        valores, ausente, method="knn_features", seed=SEMENTE, n_neighbors=k_comum
    )
    por_amostra_igual = _imputa_por_modelo(
        valores, ausente, method="knn", seed=SEMENTE, n_neighbors=k_comum
    )
    por_amostra_producao = _imputa_por_modelo(valores, ausente, method="knn", seed=SEMENTE)

    return {
        "feature_k_igual": _erro_nas_escondidas(por_feature, verdade, mascara),
        "amostra_k_igual": _erro_nas_escondidas(por_amostra_igual, verdade, mascara),
        "amostra_k_producao": _erro_nas_escondidas(por_amostra_producao, verdade, mascara),
    }


# ---------------------------------------------------------------------------
# As âncoras do fixture, e elas vêm ANTES da tabela
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fracao", [0.10, 0.25])
def test_ancora_a_mascara_esconde_o_que_promete_esconder(fracao: float) -> None:
    """Fixture que promete conter um caso é verificado quanto a contê-lo.

    Se a máscara ficasse vazia, ou se cobrisse a matriz inteira, o erro medido abaixo seria
    respectivamente indefinido ou sobre nada de comparável — e nos dois casos os testes
    passariam sem medir a propriedade.
    """
    verdade = _verdade()
    for nome, mascara in (
        ("MNAR", _mascara_mnar(verdade, fracao, SEMENTE)),
        ("MCAR", _mascara_mcar(verdade, fracao, SEMENTE)),
    ):
        proporcao = mascara.mean()
        # **A fração medida bate a pedida, e esta asserção existe porque ela já não batia.**
        # A primeira versão da máscara MNAR cortava no decil fixo: pedir 25% escondia 12,7%, e
        # a tabela saía rotulada com uma população que não era a dela.
        assert abs(proporcao - fracao) < 0.02, (
            f"{nome} pedido a {fracao:.0%} mascarou {proporcao:.1%}; o rótulo da tabela diria "
            "uma população e a medida seria sobre outra"
        )
        assert (verdade[mascara] > 0).all(), (
            f"{nome}: a verdade tem zero numa célula escondida, e o erro relativo dela não "
            "teria denominador"
        )


def test_ancora_a_mascara_mnar_e_de_facto_o_lado_BAIXO(  # noqa: N802
) -> None:
    """A máscara MNAR tem de esconder valores menores que os que deixa — senão não é MNAR.

    **É a planta que separa**: se o MNAR e o MCAR escondessem a mesma distribuição, os
    dois blocos de medição abaixo estariam a medir a mesma coisa duas vezes, e a comparação
    entre eles não teria conteúdo.
    """
    verdade = _verdade()
    mnar = _mascara_mnar(verdade, 0.10, SEMENTE)
    mcar = _mascara_mcar(verdade, 0.10, SEMENTE)

    relativo_mnar = float(np.median(verdade[mnar] / np.median(verdade, axis=0)[np.where(mnar)[1]]))
    relativo_mcar = float(np.median(verdade[mcar] / np.median(verdade, axis=0)[np.where(mcar)[1]]))
    assert relativo_mnar < relativo_mcar, (
        f"o MNAR escondeu células a {relativo_mnar:.3f} da mediana da feature e o MCAR a "
        f"{relativo_mcar:.3f}: a máscara dita MNAR não é do lado baixo"
    )


# ---------------------------------------------------------------------------
# A tabela — o desfecho pré-registrado
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fracao", [0.10, 0.25])
def test_a_tabela_de_recuperacao_dos_tres_bracos_sob_MNAR(  # noqa: N802
    fracao: float, capsys: pytest.CaptureFixture[str]
) -> None:
    """Mede e REPORTA. Sem valor esperado e sem tolerância — a tabela é o desfecho.

    O que se afirma é que a medição aconteceu sobre uma população nomeada e que nenhum braço
    devolveu um número que não é número. Escolher um vencedor aqui seria fixar um resultado antes de
    saber se é o certo.
    """
    verdade = _verdade()
    mascara = _mascara_mnar(verdade, fracao, SEMENTE)
    tabela = _tres_bracos(verdade, mascara)

    with capsys.disabled():
        print(f"\n  MNAR {fracao:.0%} - {mascara.sum()} celulas escondidas de {verdade.size}")
        for braco, erro in tabela.items():
            print(f"    {braco:<22} erro relativo mediano = {erro:.4f}")

    for braco, erro in tabela.items():
        assert np.isfinite(erro), f"{braco} devolveu {erro!r}"


@pytest.mark.parametrize("fracao", [0.10, 0.25])
def test_a_tabela_de_recuperacao_dos_tres_bracos_sob_MCAR(  # noqa: N802
    fracao: float, capsys: pytest.CaptureFixture[str]
) -> None:
    """O controle. Sem ele, uma vantagem sob MNAR seria lida como vantagem geral."""
    verdade = _verdade()
    mascara = _mascara_mcar(verdade, fracao, SEMENTE)
    tabela = _tres_bracos(verdade, mascara)

    with capsys.disabled():
        print(f"\n  MCAR {fracao:.0%} - {mascara.sum()} celulas escondidas de {verdade.size}")
        for braco, erro in tabela.items():
            print(f"    {braco:<22} erro relativo mediano = {erro:.4f}")

    for braco, erro in tabela.items():
        assert np.isfinite(erro), f"{braco} devolveu {erro!r}"


# ---------------------------------------------------------------------------
# A âncora: com o `k` igual, só a transposição os separa
# ---------------------------------------------------------------------------


def test_ancora_com_k_igual_os_dois_bracos_NAO_dao_o_mesmo_numero(  # noqa: N802
) -> None:
    """A âncora pré-registrada, e é ela que torna a tabela atribuível.

    Com o `k` igual, a **única** diferença entre os dois primeiros braços é a transposição. Se
    os erros batessem a 1e-12, uma de duas coisas seria verdade: ou o ramo novo não transpõe —
    e é o mesmo método com outro nome —, ou o `KNNImputer` é invariante à transposição, e
    então a distinção que o MetaboAnalyst oferece na tela não existe.

    **Uma implementação que não transpusesse passaria em todos os outros testes deste arquivo**
    e reprovaria aqui. É essa a propriedade que o par causal abaixo confirma.
    """
    verdade = _verdade()
    mascara = _mascara_mnar(verdade, 0.10, SEMENTE)
    tabela = _tres_bracos(verdade, mascara)

    assert abs(tabela["feature_k_igual"] - tabela["amostra_k_igual"]) > 1e-12, (
        f"com o mesmo k os dois braços deram {tabela['feature_k_igual']!r} e "
        f"{tabela['amostra_k_igual']!r}: a transposição não está a acontecer, ou não muda nada"
    )


def test_par_causal_a_ancora_ACUSA_uma_implementacao_que_nao_transpoe(  # noqa: N802
) -> None:
    """A planta: um «feature-wise» que não transpõe é indistinguível do sample-wise.

    Este é o desfecho oposto da âncora acima, sobre a mesma entrada e com o mesmo `k`.
    Ver os dois é o que separa a âncora de uma asserção que passaria de qualquer forma.
    """
    verdade = _verdade()
    mascara = _mascara_mnar(verdade, 0.10, SEMENTE)
    valores = _com_ausencia(verdade, mascara)
    ausente = is_missing(valores)
    k_comum = max(1, round(float(np.sqrt(verdade.shape[0]))))

    # A planta: pedir o método por AMOSTRA duas vezes é o que uma implementação sem
    # transposição produziria ao ser chamada com os dois rótulos.
    sem_transpor = _imputa_por_modelo(
        valores, ausente, method="knn", seed=SEMENTE, n_neighbors=k_comum
    )
    de_novo = _imputa_por_modelo(valores, ausente, method="knn", seed=SEMENTE, n_neighbors=k_comum)

    erro_a = _erro_nas_escondidas(sem_transpor, verdade, mascara)
    erro_b = _erro_nas_escondidas(de_novo, verdade, mascara)
    assert abs(erro_a - erro_b) <= 1e-12, (
        "a planta não aterrou: duas chamadas idênticas deram números diferentes, então o "
        "arnês tem não-determinismo e a âncora acima não separa transposição de ruído"
    )


# ---------------------------------------------------------------------------
# O que o método novo NÃO pode ter mudado
# ---------------------------------------------------------------------------


def test_o_metodo_existente_devolve_exatamente_o_que_devolvia() -> None:
    """O `knn` por amostra não se mexeu — R2, e é o que protege execução já gravada.

    A prova é de invariância estrutural e não de número pinado: o caminho existente não
    recebe `keep_empty_features`, e a saída dele sobre a mesma entrada e semente continua
    igual à do imputador chamado com os parâmetros que o imputador declara.
    """
    from sklearn.impute import KNNImputer

    verdade = _verdade()
    mascara = _mascara_mnar(verdade, 0.10, SEMENTE)
    valores = _com_ausencia(verdade, mascara)
    ausente = is_missing(valores)

    nosso = _imputa_por_modelo(valores, ausente, method="knn", seed=SEMENTE)
    entrada = np.where(ausente, np.nan, valores)
    k = max(1, round(float(np.sqrt(entrada.shape[1]))))
    esperado = KNNImputer(n_neighbors=k, weights="uniform").fit_transform(entrada)

    assert np.allclose(nosso, esperado, rtol=0.0, atol=0.0), (
        "o caminho do `knn` por amostra mudou de resultado; execuções já gravadas passariam "
        "a significar outra coisa"
    )


def test_duas_execucoes_do_metodo_novo_dao_saida_identica() -> None:
    """A regra 3 do README, sobre o método novo."""
    verdade = _verdade()
    matriz = _matriz(_com_ausencia(verdade, _mascara_mnar(verdade, 0.10, SEMENTE)))
    primeira = impute(matriz, "knn_features", seed=7)
    segunda = impute(matriz, "knn_features", seed=7)
    assert np.array_equal(np.asarray(primeira.values), np.asarray(segunda.values))


def test_o_metodo_novo_preserva_a_forma_e_nao_deixa_ausencia() -> None:
    """Oito dos nove métodos preservam o conjunto de features; este é um deles."""
    verdade = _verdade()
    matriz = _matriz(_com_ausencia(verdade, _mascara_mnar(verdade, 0.25, SEMENTE)))
    imputada = impute(matriz, "knn_features", seed=SEMENTE)

    assert imputada.feature_names == matriz.feature_names
    assert np.asarray(imputada.values).shape == np.asarray(matriz.values).shape
    assert not is_missing(np.asarray(imputada.values)).any(), (
        "o método novo deixou ausência para trás; só `none` promete isso"
    )
    assert np.isfinite(np.asarray(imputada.values)).all()


def test_o_rotulo_novo_esta_no_vocabulario_e_tem_ramo_proprio() -> None:
    """Medido: o rótulo está no `Literal` **e** não cai no ramo de outro método.

    A segunda metade é a que importa. Um valor que passasse pela guarda e caísse noutra
    política produziria número medido de forma diferente da pedida, com a proveniência a
    registar o nome certo — o modo de falha o mais insidioso.
    """
    assert "knn_features" in VOCABULARY["Imputation"]

    verdade = _verdade()
    matriz = _matriz(_com_ausencia(verdade, _mascara_mnar(verdade, 0.10, SEMENTE)))
    nova = np.asarray(impute(matriz, "knn_features", seed=SEMENTE).values)
    for outro in ("knn", "median", "mean", "one_fifth_min", "col_min"):
        existente = np.asarray(impute(matriz, outro, seed=SEMENTE).values)
        assert not np.allclose(nova, existente, rtol=0.0, atol=0.0), (
            f"`knn_features` devolveu exactamente o mesmo que `{outro}`: o rótulo novo está a "
            "cair no ramo de outro método"
        )


# ---------------------------------------------------------------------------
# A amostra sem NENHUMA detecção — a direção em que `keep_empty_features` age
# ---------------------------------------------------------------------------


def test_a_amostra_SEM_NENHUMA_deteccao_e_imputada_e_nao_devolvida_como_zeros() -> None:  # noqa: N802
    """A bandeira `keep_empty_features=True` estava por testar na direção para que existe.

    ## O defeito, e ele é `impute` a mentir sobre si própria

    A bandeira existe para o `KNNImputer` não **descartar** uma coluna toda vazia — que no eixo
    transposto é uma amostra inteira, e descartá-la mudaria `n_samples`, que `with_values`
    recusa. O que ele faz em vez de descartar é preencher com **zero**.

    E zero, neste projeto, **é ausência**: `is_missing` conta-o como célula não detectada
    (`missing.py`: zeros não são zeros). Medido em 10/09/2026:
    `impute(m, "knn_features")` devolvia `[0, 0, 0, 0]` para a amostra vazia, que `is_missing`
    continuava a reportar como ausente, enquanto `impute(m, "knn")` na mesma matriz imputava
    valores reais.

    **Nenhuma fixture existente tem amostra inteiramente ausente**, e é por isso que a bandeira
    passou por testada: as fixturas de MNAR e MCAR têm ausência parcial, para a qual ela é
    inerte: o valor natural é o que executa menos do caminho.

    ## As três asserções, e nenhuma é redundante

    A forma preservada é o que a bandeira compra; a ausência de ausência é o contrato de
    `impute`; e a comparação com `knn` é o que separa *"preencheu"* de *"preencheu com a mesma
    política do irmão"* — as duas continuam a ser métodos diferentes.
    """
    bruto = np.array(
        [
            [10.0, 20.0, 30.0, 40.0],
            [12.0, 22.0, 32.0, 42.0],
            [0.0, 0.0, 0.0, 0.0],
            [14.0, 24.0, 34.0, 44.0],
        ]
    )
    matriz = _matriz(bruto)
    ausente = is_missing(bruto)
    assert ausente[2].all(), "a planta não aterrou: a amostra 2 devia estar inteiramente ausente"
    assert not ausente[[0, 1, 3]].any(), "as outras três têm de estar completas"

    imputada = np.asarray(impute(matriz, "knn_features", seed=SEMENTE).values)

    assert imputada.shape == bruto.shape, (
        "a forma mudou: `keep_empty_features` existe exactamente para o `KNNImputer` não "
        "descartar a amostra vazia"
    )
    assert not is_missing(imputada).any(), (
        f"`knn_features` devolveu a amostra vazia como {imputada[2].tolist()}, e `is_missing` "
        "ainda a lê como ausente. A função declara ter imputado e não imputou"
    )
    por_amostra = np.asarray(impute(matriz, "knn", seed=SEMENTE).values)
    assert not np.allclose(imputada[2], por_amostra[2], rtol=0.0, atol=0.0), (
        "os dois métodos deram o mesmo para a amostra vazia; o resgate caiu no ramo do irmão"
    )
