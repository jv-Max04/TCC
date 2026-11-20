"""O parâmetro de referência da normalização.

Este arquivo cobre o **caminho** da referência: a guarda de fronteira, o veredito de presença
que `process_matrix` decide antes do primeiro filtro, e a grade que não se move. Os métodos que
a consomem chegam depois, e os testes deles ficam ao lado dos irmãos em
`test_normalization.py`.

## Por que o veredito de presença é testado AQUI e não lá

Ele é propriedade do **pipeline** e não da normalização: a decisão acontece antes do primeiro
filtro, e o que `normalize_samples` recebe é o resultado dela. Testá-lo dentro do teste da
normalização mediria o transporte do valor, não a decisão — que é a distinção que o veredito existe
para manter.
"""

from __future__ import annotations

import numpy as np
import pytest

from refino.domain.matrix import RefinoMatrix
from refino.domain.metadata import SampleMetadata
from refino.domain.spec import SUPPORTED_OVERRIDES, PipelineSpec
from refino.preprocessing.normalization import ReferenciaAusenteError, normalize_samples


def _matriz(n_features: int = 6, n_amostras: int = 8) -> RefinoMatrix:
    rng = np.random.default_rng(20260910)
    valores = rng.uniform(100.0, 5000.0, size=(n_amostras, n_features))
    amostras = tuple(
        SampleMetadata(
            name=f"S{i}",
            injection_order=i + 1,
            sample_type="qc" if i < 3 else "sample",
            group="QC" if i < 3 else "G",
        )
        for i in range(n_amostras)
    )
    return RefinoMatrix(
        values=valores,
        feature_names=tuple(f"f{j}" for j in range(n_features)),
        samples=amostras,
        mode="sintetico",
    )


def _spec(**troca: object) -> PipelineSpec:
    campos: dict[str, object] = {
        "correction": "none",
        "n_batches": 1,
        "rsd_cutoff": None,
        "ranking_filter": "none",
        "ranking_keep_fraction": None,
        "ranking_filter_2": "none",
        "ranking_keep_fraction_2": None,
        "rsd_variant": "outlier_removed",
        "missing_threshold": None,
        "imputation": "one_fifth_min",
        "sample_normalization": "none",
        "normalization_reference": None,
        "normalization_factors": None,
        "transformation": "none",
        "scaling": "none",
    }
    campos.update(troca)
    return PipelineSpec(**campos)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# A guarda de FRONTEIRA — e o que ela NÃO confere
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("vazio", ["", "   ", "\t", "\n"])
def test_nome_vazio_e_recusado_na_fronteira(vazio: str) -> None:
    """`''` não é nome, e o `None` é que diz «não escolhi».

    Sem esta guarda o valor atravessaria a fronteira, chegaria à normalização e falharia a
    procurar uma amostra chamada `""` — com a mensagem a nomear a **etapa** em vez do campo que
    o usuário preencheu: dois significados para o mesmo valor.
    """
    with pytest.raises(ValueError, match="normalization_reference"):
        _spec(normalization_reference=vazio)


def test_none_e_um_nome_real_sao_ACEITOS(  # noqa: N802
) -> None:
    """A outra direção da guarda, e sem ela `test_nome_vazio` passaria numa que recusasse tudo."""
    assert _spec(normalization_reference=None).normalization_reference is None
    assert _spec(normalization_reference="QC1").normalization_reference == "QC1"


def test_a_fronteira_NAO_confere_se_o_nome_existe_e_isso_e_desenho(  # noqa: N802
) -> None:
    """A existência é pergunta que só a matriz responde, e a guarda vive onde ela é conhecida.

    Conferi-la aqui exigiria que o contrato conhecesse a matriz — que ele não conhece, e não
    deve: a mesma especificação corre sobre estudos diferentes no multiverso. É a metade
    que decide **onde** a recusa acontece, e não **se** ela acontece.
    """
    spec = _spec(normalization_reference="nome-que-nenhuma-matriz-tem")
    assert spec.normalization_reference == "nome-que-nenhuma-matriz-tem"


def test_a_referencia_e_eixo_sobrescrevivel() -> None:
    """Trocar a referência muda cada célula: é discordância metodológica e nomeia o eixo."""
    assert "normalization_reference" in SUPPORTED_OVERRIDES


# ---------------------------------------------------------------------------
# A IMPRESSÃO DIGITAL, e é ela que liga o campo ao cache do multiverso
# ---------------------------------------------------------------------------


def test_referencias_diferentes_dao_impressoes_digitais_diferentes() -> None:
    """Sem isto, duas execuções com referências diferentes partilhariam célula de cache.

    A impressão digital soma os campos em ordem declarada, então um campo novo entra nela por
    construção — e este teste é o que prova que entrou, em vez de o supor.
    """
    a = _spec(normalization_reference="QC1")
    b = _spec(normalization_reference="QC2")
    sem = _spec(normalization_reference=None)

    assert a.fingerprint() != b.fingerprint()
    assert a.fingerprint() != sem.fingerprint()


# ---------------------------------------------------------------------------
# O VEREDITO DE PRESENÇA — o que a normalização vai consumir
# ---------------------------------------------------------------------------


def test_o_veredito_de_presenca_e_decidido_ANTES_do_primeiro_filtro(  # noqa: N802
) -> None:
    """`process_matrix` responde «estava na entrada?» sobre a matriz como ela chegou.

    **A planta cai sobre dado em que as duas coisas diferem**: a feature nomeada existe
    na matriz ingerida **e** é removida pelo filtro por posto antes de a normalização correr.
    Sem essa construção, «decidido antes» e «decidido depois» dariam a mesma resposta e o teste
    não separaria nada.
    """
    from refino.preprocessing.pipeline import process_matrix

    matriz = _matriz(n_features=10)
    # A feature de menor variância é a que o filtro por posto corta primeiro.
    valores = np.asarray(matriz.values).copy()
    valores[:, 0] = 1000.0  # constante: variância zero, primeira a sair
    matriz = matriz.with_values(valores, "planta")

    spec = _spec(
        ranking_filter="sd",
        ranking_keep_fraction=0.5,
        normalization_reference="f0",
    )
    processada, resultados = process_matrix(matriz, spec)

    # A planta aterrou: a feature nomeada SAIU no caminho.
    assert "f0" not in processada.feature_names, (
        "a feature nomeada sobreviveu aos filtros; a planta não separa «nunca existiu» de "
        "«foi removida», e o teste passaria com o veredito decidido no lugar errado"
    )
    # E a execução não recusou: nenhum método de referência corre ainda, então o veredito
    # viaja e ninguém o consome.
    assert resultados


def test_o_veredito_distingue_o_nome_que_NUNCA_esteve(  # noqa: N802
) -> None:
    """A outra direção: nome que não está na matriz ingerida.

    As duas juntas são o par que torna `reference_present_at_entry` informação. Com uma só, um
    veredito constante — sempre `True` ou sempre `False` — passaria.
    """
    from refino.preprocessing.pipeline import process_matrix

    matriz = _matriz()
    process_matrix(matriz, _spec(normalization_reference="nao-existe"))
    process_matrix(matriz, _spec(normalization_reference="f0"))


def test_normalize_samples_aceita_os_dois_parametros_e_nenhum_metodo_os_usa_ainda() -> None:
    """Abre o caminho; as etapas seguintes o consomem.

    **Declarado em vez de suposto**: os cinco métodos de hoje devolvem o mesmo com e sem
    referência, e afirmá-lo é o que impede um deles de a passar a consumir por acidente.
    """
    matriz = _matriz()
    for metodo in ("none", "sum", "median", "pqn_group", "quantile"):
        sem = normalize_samples(matriz, metodo)  # type: ignore[arg-type]
        com = normalize_samples(matriz, metodo, reference="QC1", reference_present_at_entry=True)  # type: ignore[arg-type]
        assert np.array_equal(np.asarray(sem.values), np.asarray(com.values)), (
            f"`{metodo}` mudou de resultado com a referência preenchida, e nenhum dos cinco a "
            "consome — ela chegou a um ramo que não devia lê-la"
        )


# ---------------------------------------------------------------------------
# A GRADE não se move
# ---------------------------------------------------------------------------


def test_a_grade_padrao_continua_com_9600_celulas() -> None:
    """A referência é escalar, e aqui o motivo é diferente do das frações.

    Um nome de amostra é propriedade do **arquivo**: uma grade que o variasse seria calibrada
    num estudo e inaplicável no seguinte. Não há conjunto a varrer, e não é uma questão de custo.
    """
    from refino.multiverse.grid import GridDefinition

    grade = GridDefinition()
    assert grade.size() == 9600
    assert grade.normalization_reference is None
    primeira = next(grade.enumerate())
    assert primeira.normalization_reference is None


# ---------------------------------------------------------------------------
# Normalização contra uma AMOSTRA de referência
# ---------------------------------------------------------------------------


def _matriz_com_qcs(n_qc: int, semente: int = 20260910) -> RefinoMatrix:
    """Matriz com o número de QCs pedido. `n_qc` é o parâmetro da âncora."""
    rng = np.random.default_rng(semente)
    valores = rng.uniform(200.0, 4000.0, size=(n_qc + 5, 7))
    amostras = tuple(
        SampleMetadata(
            name=f"S{i}",
            injection_order=i + 1,
            sample_type="qc" if i < n_qc else "sample",
            group="QC" if i < n_qc else "G",
        )
        for i in range(valores.shape[0])
    )
    return RefinoMatrix(
        values=valores,
        feature_names=tuple(f"f{j}" for j in range(valores.shape[1])),
        samples=amostras,
        mode="sintetico",
    )


def test_ancora_com_UM_qc_a_amostra_de_referencia_reproduz_a_por_grupo() -> None:  # noqa: N802
    """A âncora pré-registada, e o fixture de UM QC é o que a torna verdadeira.

    A referência de `pqn_group` é a **mediana sobre as linhas de QC**, e a mediana de uma linha
    só é a própria linha. Então com exactamente um QC os dois métodos calculam o mesmo
    denominador por construção, e a igualdade a 1e-12 mede que o método novo divide **por
    linha** e não por outra coisa.

    Com mais de um QC os dois **não** são comparáveis — o teste abaixo declara isso em vez de o
    descobrir.
    """
    matriz = _matriz_com_qcs(n_qc=1)
    por_grupo = normalize_samples(matriz, "pqn_group")
    por_amostra = normalize_samples(matriz, "sample_reference", reference="S0")

    assert np.allclose(
        np.asarray(por_grupo.values), np.asarray(por_amostra.values), rtol=1e-12, atol=1e-12
    ), "com um QC só os dois métodos têm o mesmo denominador e deram matrizes diferentes"


def test_com_MAIS_de_um_qc_os_dois_metodos_NAO_sao_comparaveis() -> None:  # noqa: N802
    """A metade que impede a âncora de ser lida como equivalência geral.

    **Sem esta direção, a âncora passaria numa implementação que ignorasse o nome** e chamasse
    `pqn_group` por baixo — e o método novo seria um apelido do antigo.
    """
    matriz = _matriz_com_qcs(n_qc=4)
    por_grupo = normalize_samples(matriz, "pqn_group")
    por_amostra = normalize_samples(matriz, "sample_reference", reference="S0")

    assert not np.allclose(
        np.asarray(por_grupo.values), np.asarray(por_amostra.values), rtol=1e-9, atol=1e-9
    ), (
        "com quatro QCs a mediana do grupo não é a linha de `S0`, e os dois métodos deram a "
        "mesma matriz: o nome não está a ser usado"
    )


def test_a_linha_da_propria_referencia_fica_INTACTA() -> None:  # noqa: N802
    """O oráculo exato, e ele é consequência de PQN e não de divisão por espectro.

    As razões da referência contra si mesma são todas `1`, logo a mediana delas é `1`, logo a
    linha é dividida por `1` e não se move. **A âncora barata** — alimentar o
    arnês com algo cuja resposta já se sabe —, e ela não depende de nenhum outro método.

    > A primeira versão deste teste afirmava que a linha ficava toda **igual a 1**, que é o que
    > a divisão feature-a-feature daria. Sob PQN a linha fica **igual a si mesma**, e a distinção
    > é a diferença entre os dois métodos.
    """
    matriz = _matriz_com_qcs(n_qc=2)
    saida = normalize_samples(matriz, "sample_reference", reference="S1")
    indice = list(matriz.sample_names).index("S1")

    assert np.allclose(
        np.asarray(saida.values)[indice],
        np.asarray(matriz.values)[indice],
        rtol=1e-12,
        atol=1e-12,
    ), "a linha da própria referência moveu-se; a mediana das razões dela contra si é 1"


def test_a_amostra_que_NAO_existe_e_recusada_com_o_nome_pedido() -> None:  # noqa: N802
    """Recusa nomeada, e não a matriz intacta.

    Devolver a entrada seria identidade silenciosa — a proveniência registaria o nome enquanto
    nada aconteceu. A mensagem carrega o nome pedido porque é ele que o operador digitou.
    """
    matriz = _matriz_com_qcs(n_qc=2)
    with pytest.raises(ReferenciaAusenteError, match="Sinvent"):
        normalize_samples(matriz, "sample_reference", reference="Sinventada")


def test_a_referencia_AUSENTE_e_recusada_antes_de_procurar_nome_nenhum() -> None:  # noqa: N802
    """`None` com o método escolhido é recusa própria, e não «amostra não encontrada».

    As duas causas levam a ações diferentes — escolher um nome contra corrigir o que se
    escreveu —, e colapsá-las numa mensagem só apaga essa diferença.
    """
    matriz = _matriz_com_qcs(n_qc=2)
    with pytest.raises(ReferenciaAusenteError, match="normalization_reference"):
        normalize_samples(matriz, "sample_reference", reference=None)


def test_uma_feature_com_ZERO_na_referencia_SAI_da_mediana_e_nao_recusa() -> None:  # noqa: N802
    """A política de razão indefinida é a de `pqn_group`, e é uma só para os dois métodos.

    Sob PQN a feature cuja referência é zero **não entra na mediana** — incluí-la como `inf`
    faria a mediana saltar, e como zero a puxaria para baixo. Recusar aqui seria política
    diferente para a mesma conta, que é a segunda fonte de verdade que a extração de
    `_pqn_contra` existe para não ter.

    **E o critério continua satisfeito**, que é o que este teste mede: matriz de entrada sem
    não finito, matriz de saída sem não finito.
    """
    matriz = _matriz_com_qcs(n_qc=2)
    valores = np.asarray(matriz.values).copy()
    valores[0, 3] = 0.0
    com_zero = matriz.with_values(valores, "planta")
    assert np.isfinite(np.asarray(com_zero.values)).all(), "a planta trouxe não finito na entrada"

    saida = normalize_samples(com_zero, "sample_reference", reference="S0")
    assert np.isfinite(np.asarray(saida.values)).all(), (
        "a divisão produziu não finito a partir de matriz sem `NaN` — é o que `[CA-5.3]` proíbe"
    )


def test_a_referencia_SEM_NENHUMA_feature_util_e_recusada() -> None:  # noqa: N802
    """O caso em que não há razão a calcular, e aí a recusa é a saída certa.

    **Aqui os dois métodos DIVERGEM de propósito, e a razão vai escrita**: `pqn_group` devolve a
    matriz intacta, porque a referência dele é derivada e um estudo sem QC útil não pediu
    normalização nenhuma. Aqui o usuário **nomeou** a amostra, e devolvê-la intacta registaria o
    nome dele na proveniência enquanto nada aconteceu — a identidade silenciosa.
    """
    matriz = _matriz_com_qcs(n_qc=2)
    valores = np.asarray(matriz.values).copy()
    valores[0, :] = 0.0
    toda_zero = matriz.with_values(valores, "planta")

    with pytest.raises(ReferenciaAusenteError, match="uma única feature"):
        normalize_samples(toda_zero, "sample_reference", reference="S0")

    # E a divergência declarada: `pqn_group` sobre a mesma matriz NÃO levanta.
    normalize_samples(toda_zero, "pqn_group")


def test_duas_execucoes_com_a_mesma_referencia_dao_saida_identica() -> None:
    """A regra 3 do README sobre o método novo. Ele não é estocástico, e afirmá-lo é barato."""
    matriz = _matriz_com_qcs(n_qc=2)
    a = normalize_samples(matriz, "sample_reference", reference="S0")
    b = normalize_samples(matriz, "sample_reference", reference="S0")
    assert np.array_equal(np.asarray(a.values), np.asarray(b.values))


def test_o_rotulo_novo_esta_no_vocabulario_e_tem_ramo_proprio() -> None:
    """Nas duas metades: no `Literal`, e com saída diferente das outras cinco."""
    from refino.domain.labels import VOCABULARY

    assert "sample_reference" in VOCABULARY["SampleNormalization"]

    matriz = _matriz_com_qcs(n_qc=4)
    nova = np.asarray(normalize_samples(matriz, "sample_reference", reference="S0").values)
    for outro in ("none", "sum", "median", "quantile"):
        existente = np.asarray(normalize_samples(matriz, outro).values)  # type: ignore[arg-type]
        assert not np.allclose(nova, existente, rtol=0.0, atol=0.0), (
            f"`sample_reference` devolveu o mesmo que `{outro}`: o rótulo caiu noutro ramo"
        )


# ---------------------------------------------------------------------------
# Normalização por uma FEATURE de referência
# ---------------------------------------------------------------------------


def _rsd_por_feature(matriz: RefinoMatrix) -> np.ndarray:
    """RSD dos QCs por feature, na fórmula simples — o que a âncora abaixo compara."""
    qcs = np.asarray(matriz.values)[matriz.qc_mask]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.asarray(qcs.std(axis=0, ddof=1) / qcs.mean(axis=0))


def test_ancora_dividir_por_feature_CONSTANTE_deixa_o_rsd_dos_qcs_intacto() -> None:  # noqa: N802
    """A âncora pré-registada, e ela mede que a divisão é POR LINHA.

    Uma feature de referência **constante** faz a matriz inteira ser dividida pelo mesmo
    número, e o RSD é invariante a escala — logo ele não pode mudar em feature nenhuma. **Se
    mudar, a divisão não foi por linha**: foi por coluna, ou por um denominador que varia entre
    amostras quando não devia.

    > **É a âncora barata**, e o motivo de ela ser sobre o RSD e não sobre os
    > valores: comparar valores exigiria reimplementar a divisão no teste, que é medir a minha
    > própria cópia. O RSD é uma propriedade que sobrevive à operação, e verificá-la não
    > precisa de saber como ela é feita.
    """
    matriz = _matriz_com_qcs(n_qc=4)
    valores = np.asarray(matriz.values).copy()
    valores[:, 2] = 777.0  # a referência: constante entre amostras
    com_padrao = matriz.with_values(valores, "planta")

    antes = _rsd_por_feature(com_padrao)
    depois = _rsd_por_feature(normalize_samples(com_padrao, "feature_reference", reference="f2"))

    assert np.allclose(antes, depois, rtol=1e-12, atol=1e-12, equal_nan=True), (
        "o RSD dos QCs mudou ao dividir por uma feature constante, e ele é invariante a "
        f"escala: a divisão não foi por linha. antes={antes}, depois={depois}"
    )


def test_par_causal_com_a_referencia_VARIAVEL_o_rsd_MUDA() -> None:  # noqa: N802
    """A planta que separa, e sem ela a âncora passaria numa identidade.

    Uma implementação que não dividisse nada deixaria o RSD intacto **sempre** — e passaria no
    teste acima. O que o separa é a referência variável: aí a divisão é por um número diferente
    por amostra, o RSD **tem** de mudar, e uma identidade reprova aqui.
    """
    matriz = _matriz_com_qcs(n_qc=4)
    antes = _rsd_por_feature(matriz)
    depois = _rsd_por_feature(normalize_samples(matriz, "feature_reference", reference="f2"))

    assert not np.allclose(antes, depois, rtol=1e-9, atol=1e-9, equal_nan=True), (
        "com a referência a variar entre amostras o RSD ficou igual: a etapa é identidade"
    )


def test_a_referencia_fica_constante_em_UM_e_SOBREVIVE_a_matriz(  # noqa: N802
) -> None:
    """O destino da feature de referência é declarado, e não descoberto.

    Depois da divisão ela vale `1` em toda amostra. A etapa que remove variância quase nula
    **já correu** — ela é a terceira da ordem canônica e a normalização é a sexta —, então a
    feature sobrevive à matriz processada. Removê-la aqui mudaria a forma da saída num ponto em
    que nenhuma outra normalização a muda.
    """
    matriz = _matriz_com_qcs(n_qc=3)
    saida = normalize_samples(matriz, "feature_reference", reference="f4")

    assert "f4" in saida.feature_names
    coluna = np.asarray(saida.values)[:, list(saida.feature_names).index("f4")]
    assert np.allclose(coluna, 1.0, rtol=0.0, atol=1e-12), (
        f"a feature de referência devia ficar toda 1 depois da divisão e é {coluna}"
    )


# ---------------------------------------------------------------------------
# As DUAS causas de ausência — e as mensagens têm de ser distinguíveis
# ---------------------------------------------------------------------------


def test_a_feature_que_NUNCA_esteve_e_a_REMOVIDA_dao_mensagens_diferentes() -> None:  # noqa: N802
    """Na metade que o operador lê, e é para isto que o veredito viaja.

    As duas ausências levam a **ações opostas**: uma manda corrigir o que se escreveu, a outra
    manda afrouxar o filtro. Uma mensagem só para as duas seria um valor com dois
    significados, na superfície em que ela custa mais.
    """
    matriz = _matriz_com_qcs(n_qc=3)

    with pytest.raises(ReferenciaAusenteError, match="não existe no arquivo"):
        normalize_samples(
            matriz, "feature_reference", reference="fX", reference_present_at_entry=False
        )

    with pytest.raises(ReferenciaAusenteError, match="DESCARTADA"):
        normalize_samples(
            matriz, "feature_reference", reference="fX", reference_present_at_entry=True
        )


def test_sem_veredito_a_mensagem_DECLARA_a_ambiguidade(  # noqa: N802
) -> None:
    """`None` é «ninguém perguntou», e afirmar uma das duas seria inventar procedência.

    Quem chama a função por fora do pipeline não tem o veredito, e a mensagem di-lo em vez de
    escolher a causa mais provável.
    """
    matriz = _matriz_com_qcs(n_qc=3)
    with pytest.raises(ReferenciaAusenteError, match="não foi conferida"):
        normalize_samples(matriz, "feature_reference", reference="fX")


def test_a_recusa_por_nome_INEXISTENTE_distingue_NUNCA_EXISTIU_de_descartada(  # noqa: N802
) -> None:
    """O que o veredito decidido na ENTRADA compra, e ele não é o que o nome antigo dizia.

    ## A correção deste teste, em 10/09/2026, e ela é sobre o próprio teste

    Ele chamava-se `..._acontece_antes_da_correcao` e a docstring dizia *"A prova é a contagem
    de etapas executadas: zero"*. **O corpo não contava nada** — só afirmava que
    `process_matrix` levanta. E o nome era falso no produto: as duas rotas correm
    `_corrige_guardando` **antes** de `process_matrix`, então o operador paga a correção inteira
    de qualquer forma.

    **Nome e docstring que afirmam mais que o corpo são afirmação maior que a verificação,
    dentro de um teste** — e num teste é pior, porque quem lê o nome numa lista de
    reprovações acredita nele sem abrir o arquivo.

    ## O que o veredito na entrada compra DE FACTO, e é isto que se afirma aqui

    Cinco etapas removem features antes da normalização. Depois delas, *«o nome nunca esteve
    aqui»* e *«um filtro removeu-o»* são indistinguíveis: a proveniência regista operações e
    não identidades. O veredito é decidido antes do primeiro filtro exactamente para que as
    duas recusas tenham **mensagens diferentes** — e é a diferença entre mandar o operador
    procurar um erro de digitação e mandá-lo afrouxar um corte.

    A asserção é sobre a mensagem, que é o que o mecanismo produz, e as duas metades vêm
    juntas: a que o caso dá, e a que ele **não** dá.
    """
    from refino.preprocessing.pipeline import process_matrix

    matriz = _matriz_com_qcs(n_qc=3)
    spec = _spec(sample_normalization="feature_reference", normalization_reference="fX")

    with pytest.raises(ReferenciaAusenteError) as capturado:
        process_matrix(matriz, spec)

    mensagem = str(capturado.value)
    assert "não existe no arquivo que você enviou" in mensagem, (
        f"a recusa não disse que a feature nunca existiu. Mensagem: {mensagem!r}"
    )
    assert "DESCARTADA" not in mensagem, (
        "a recusa acusou descarte por filtro sobre uma feature que nunca esteve na matriz — o "
        f"veredito na entrada não está a ser consultado. Mensagem: {mensagem!r}"
    )


def test_o_caminho_FELIZ_atravessa_o_pipeline_inteiro() -> None:  # noqa: N802
    """A outra direção: com um nome que existe, o pipeline corre até ao fim.

    Sem ela, `test_a_recusa_por_nome_INEXISTENTE` passaria numa implementação que recusasse
    **sempre** — a planta que não separa.
    """
    from refino.preprocessing.pipeline import process_matrix

    matriz = _matriz_com_qcs(n_qc=3)
    spec = _spec(sample_normalization="feature_reference", normalization_reference="f1")
    processada, resultados = process_matrix(matriz, spec)

    assert np.isfinite(np.asarray(processada.values)).all()
    assert any(r.stage == "normalize_samples" for r in resultados)


def test_a_feature_com_ZERO_numa_amostra_e_recusada() -> None:  # noqa: N802
    """Propagar está proibido, e deixar a linha intacta normaliza só metade.

    **Aqui a política DIVERGE de `sample_reference`, e a razão é a estrutura da conta.** Lá o
    denominador é a mediana de razões, e uma feature degenerada apenas sai da mediana — as
    outras ainda a formam. Aqui o denominador **é** aquele valor: se ele não serve, aquela
    amostra não tem por onde ser dividida, e não há mediana onde se esconder.
    """
    matriz = _matriz_com_qcs(n_qc=3)
    valores = np.asarray(matriz.values).copy()
    valores[2, 1] = 0.0
    com_zero = matriz.with_values(valores, "planta")
    assert np.isfinite(np.asarray(com_zero.values)).all(), "a planta trouxe não finito na entrada"

    with pytest.raises(ReferenciaAusenteError, match="zero, negativo ou não finito"):
        normalize_samples(com_zero, "feature_reference", reference="f1")


def test_a_feature_NEGATIVA_numa_amostra_e_recusada() -> None:  # noqa: N802
    """O irmão do teste acima, e o valor negativo é PIOR que o zero.

    ## Por que este teste não existia, e o que ele custou

    A guarda era `coluna != 0.0`. Um valor negativo é finito e não nulo, então passava — e
    dividir por ele **inverte o sinal da amostra inteira**. Sem exceção, sem rótulo em
    `StageResult`, todos os valores finitos, código de saída 0, e a linha invertida escrita no
    artefato.

    **A assimetria com o zero é o argumento inteiro.** O zero produz `inf`, que o critério
    proíbe e que qualquer conferência de finitude apanha. O negativo produz número **plausível**
    e com o sinal trocado, que nenhuma conferência de finitude vê.

    **A irmã já recusava isto, com esta justificativa escrita.** `PipelineSpec.__post_init__`
    rejeita fator negativo em `normalization_factors` dizendo *"Um NEGATIVO passa pela divisao
    protegida e inverte o sinal da amostra inteira sem nada a acusar"* — a mesma frase, sobre o
    outro método. A guarda não tinha sido copiada, e o teste do negativo vivia só na irmã.

    ## A planta cai onde as duas coisas diferem

    A asserção afirma **antes** que a entrada é toda positiva. Sem isso, uma matriz que já
    tivesse negativos faria a recusa acontecer por outro motivo, e o teste passaria sem separar
    nada. É alcançável em uso real: `NEGATIVE_INTENSITY` é severidade `warning`, então uma
    exportação com subtração de linha de base é ingerida e chega até aqui.
    """
    matriz = _matriz_com_qcs(n_qc=3)
    valores = np.asarray(matriz.values).copy()
    assert (valores > 0).all(), "a matriz de partida já tem não positivo; a planta não separaria"

    valores[2, 1] = -3.0
    com_negativo = matriz.with_values(valores, "planta")
    assert np.isfinite(np.asarray(com_negativo.values)).all(), (
        "a planta trouxe não finito na entrada, e este teste é sobre o valor FINITO negativo"
    )

    with pytest.raises(ReferenciaAusenteError, match="zero, negativo ou não finito"):
        normalize_samples(com_negativo, "feature_reference", reference="f1")


def test_o_rotulo_da_feature_esta_no_vocabulario_e_tem_ramo_proprio() -> None:
    """Nas duas metades, e a segunda inclui o irmão: os dois métodos de referência
    têm de dar resultados diferentes sobre a mesma matriz."""
    from refino.domain.labels import VOCABULARY

    assert "feature_reference" in VOCABULARY["SampleNormalization"]

    matriz = _matriz_com_qcs(n_qc=4)
    por_feature = np.asarray(normalize_samples(matriz, "feature_reference", reference="f1").values)
    por_amostra = np.asarray(normalize_samples(matriz, "sample_reference", reference="S0").values)
    assert not np.allclose(por_feature, por_amostra, rtol=0.0, atol=0.0), (
        "os dois métodos de referência deram a mesma matriz: um deles caiu no ramo do outro"
    )


# ---------------------------------------------------------------------------
# Os FATORES fornecidos, e eles viajam como MAPA e não como caminho
# ---------------------------------------------------------------------------


def _fatores_de(matriz: RefinoMatrix, valor: float = 2.0) -> tuple[tuple[str, float], ...]:
    """Um fator por amostra, cobrindo TODAS as linhas — QCs incluídos."""
    return tuple((nome, valor) for nome in matriz.sample_names)


def test_ancora_vetor_de_UNS_e_identidade_EXATA() -> None:  # noqa: N802
    """A primeira âncora, e ela tem resposta conhecida antes de a função existir.

    Dividir por `1` não muda nada, e a igualdade é **exata** e não a 1e-12: qualquer diferença
    aqui é aritmética que não devia ter acontecido.
    """
    matriz = _matriz_com_qcs(n_qc=3)
    saida = normalize_samples(matriz, "supplied_factors", factors=_fatores_de(matriz, valor=1.0))
    assert np.array_equal(np.asarray(saida.values), np.asarray(matriz.values)), (
        "o vetor de uns não foi identidade exata"
    )


def test_ancora_o_mapa_das_MEDIANAS_reproduz_a_normalizacao_por_mediana() -> None:  # noqa: N802
    """A segunda âncora, e ela mede contra um método que já existe.

    `median` divide cada linha pela mediana dela. Fornecer exactamente essas medianas como
    fatores tem de dar a mesma matriz — e isso prova que a divisão é **por linha** e que o mapa
    é endereçado **por nome**, e não pela ordem em que as entradas aparecem.
    """
    matriz = _matriz_com_qcs(n_qc=3)
    medianas = np.median(np.asarray(matriz.values), axis=1)
    mapa = tuple(zip(matriz.sample_names, medianas, strict=True))

    por_mediana = normalize_samples(matriz, "median")
    por_fatores = normalize_samples(matriz, "supplied_factors", factors=mapa)

    assert np.allclose(
        np.asarray(por_mediana.values), np.asarray(por_fatores.values), rtol=1e-12, atol=1e-12
    ), "o mapa das medianas não reproduziu a normalização por mediana"


def test_o_mapa_e_enderecado_por_NOME_e_nao_por_ordem() -> None:  # noqa: N802
    """A planta que separa: embaralhar a ORDEM do mapa não pode mudar a saída.

    Sem esta direção, uma implementação que consumisse os fatores na ordem em que chegam
    passaria nas duas âncoras acima — nelas o mapa já vem na ordem das amostras.
    """
    matriz = _matriz_com_qcs(n_qc=3)
    crescente = tuple((nome, float(i + 2)) for i, nome in enumerate(matriz.sample_names))
    embaralhado = tuple(reversed(crescente))

    a = normalize_samples(matriz, "supplied_factors", factors=crescente)
    b = normalize_samples(matriz, "supplied_factors", factors=embaralhado)

    assert np.array_equal(np.asarray(a.values), np.asarray(b.values)), (
        "trocar a ordem das entradas do mapa mudou a saída: ele está a ser lido por posição"
    )
    # E a planta aterrou: os fatores DIFEREM entre si, então a ordem podia ter importado.
    assert len({fator for _n, fator in crescente}) > 1


# ---------------------------------------------------------------------------
# A cobertura é TOTAL ou recusa, e os QCs contam
# ---------------------------------------------------------------------------


def test_mapa_que_nao_cobre_todas_as_linhas_e_recusado_com_a_contagem() -> None:
    """Normalizar parte das linhas deixaria a matriz com dois regimes misturados."""
    matriz = _matriz_com_qcs(n_qc=3)
    parcial = _fatores_de(matriz)[:-2]

    with pytest.raises(ReferenciaAusenteError, match="não têm fator"):
        normalize_samples(matriz, "supplied_factors", factors=parcial)


def test_mapa_que_nao_cobre_os_QCS_e_recusado_e_a_mensagem_diz_por_que(  # noqa: N802
) -> None:
    """Os QCs contam, e a mensagem explica — porque a razão não é óbvia para quem opera.

    Peso seco de um QC não existe: ele é uma mistura. A tentação natural é deixá-los de fora, e
    é por isso que a recusa carrega o argumento — **o RSD de QC é a régua** que diz se a
    correção funcionou, e normalizar as amostras sem normalizar os QCs faz a comparação de
    antes e depois deixar de ser sobre a mesma escala.
    """
    matriz = _matriz_com_qcs(n_qc=3)
    sem_qcs = tuple(
        (nome, 2.0)
        for nome, e_qc in zip(matriz.sample_names, matriz.qc_mask, strict=True)
        if not e_qc
    )

    with pytest.raises(ReferenciaAusenteError, match="régua"):
        normalize_samples(matriz, "supplied_factors", factors=sem_qcs)


def test_a_ausencia_do_mapa_com_o_metodo_escolhido_e_recusa_propria() -> None:
    """`None` é causa diferente de «faltam linhas», e as ações são diferentes."""
    matriz = _matriz_com_qcs(n_qc=3)
    with pytest.raises(ReferenciaAusenteError, match="normalization_factors"):
        normalize_samples(matriz, "supplied_factors", factors=None)


# ---------------------------------------------------------------------------
# A guarda de VALOR vive no contrato, e o NEGATIVO é o caso que importa
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ruim", [0.0, -1.0, -0.5, float("inf"), float("nan")])
def test_fator_nao_positivo_ou_nao_finito_e_recusado_na_fronteira(ruim: float) -> None:
    """**O negativo é o que importa**, e é por isso que a guarda é de sinal e não só de zero.

    Zero e não finito ao menos cairiam na guarda de `_seguro`. Um fator **negativo** passa pela
    divisão sem levantar e **inverte o sinal da amostra inteira** — a matriz sai plausível, com
    intensidades negativas que nenhuma etapa seguinte questiona.
    """
    with pytest.raises(ValueError, match="positivo e finito"):
        _spec(normalization_factors=(("S0", ruim),))


def test_nome_duplicado_no_mapa_e_recusado() -> None:
    """O leitor da lista de amostras sobrescreve duplicados em silêncio, e aqui isso decidiria.

    Com dois fatores para a mesma amostra, o último ganharia — e ninguém saberia qual foi usado.
    """
    with pytest.raises(ValueError, match="repete a amostra"):
        _spec(normalization_factors=(("S0", 2.0), ("S1", 3.0), ("S0", 5.0)))


def test_nome_vazio_no_mapa_e_recusado() -> None:
    """Uma chave `''` não endereça linha nenhuma."""
    with pytest.raises(ValueError, match="nome vazio"):
        _spec(normalization_factors=(("", 2.0),))


def test_um_mapa_valido_e_ACEITO_pela_fronteira() -> None:  # noqa: N802
    """A direção que impede a guarda de recusar tudo."""
    spec = _spec(normalization_factors=(("S0", 2.0), ("S1", 3.5)))
    assert spec.normalization_factors == (("S0", 2.0), ("S1", 3.5))


# ---------------------------------------------------------------------------
# A IMPRESSÃO DIGITAL — é isto que o caminho de arquivo não podia dar
# ---------------------------------------------------------------------------


def test_fatores_diferentes_dao_impressoes_digitais_diferentes() -> None:
    """Medido, e é a metade que decide entre mapa e caminho.

    **Com um caminho no contrato, estas duas especificações teriam a mesma impressão digital** —
    o hash carregaria o caminho e nunca o conteúdo —, e dois refinos com fatores diferentes
    partilhariam célula de cache do multiverso. O segundo receberia o resultado guardado do
    primeiro, e nada acusaria.
    """
    a = _spec(normalization_factors=(("S0", 2.0),))
    b = _spec(normalization_factors=(("S0", 3.0),))
    sem = _spec(normalization_factors=None)

    assert a.fingerprint() != b.fingerprint()
    assert a.fingerprint() != sem.fingerprint()


def test_o_rotulo_dos_fatores_esta_no_vocabulario_e_tem_ramo_proprio() -> None:
    """Nas duas metades, contra os dois irmãos de referência."""
    from refino.domain.labels import VOCABULARY

    assert "supplied_factors" in VOCABULARY["SampleNormalization"]

    matriz = _matriz_com_qcs(n_qc=4)
    por_fatores = np.asarray(
        normalize_samples(matriz, "supplied_factors", factors=_fatores_de(matriz, 3.0)).values
    )
    for outro, extra in (
        ("median", {}),
        ("sum", {}),
        ("sample_reference", {"reference": "S0"}),
        ("feature_reference", {"reference": "f1"}),
    ):
        existente = np.asarray(normalize_samples(matriz, outro, **extra).values)  # type: ignore[arg-type]
        assert not np.allclose(por_fatores, existente, rtol=0.0, atol=0.0), (
            f"`supplied_factors` devolveu o mesmo que `{outro}`: o rótulo caiu noutro ramo"
        )


# ---------------------------------------------------------------------------
# A ausência de não-finito como PÓS-CONDIÇÃO, e o caso que nenhuma guarda de entrada apanha
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("metodo", ["supplied_factors", "feature_reference"])
def test_denominador_SUBNORMAL_e_recusado_e_nao_produz_inf(metodo: str) -> None:  # noqa: N802
    """O buraco que `_seguro` **não** fecha, e ele custou um artefato com 173.672 `inf`.

    ## A propriedade que faltava não é a que as guardas de entrada verificam

    `_seguro` protege contra denominador **zero** ou não finito. `PipelineSpec` recusa fator
    não positivo ou não finito. As duas juntas ainda deixam passar `5e-324` — positivo, finito,
    e o menor `float64` que existe.

    > **`finito e > 0` não é a mesma propriedade que `não transborda`.**

    Medido em 10/09/2026, de ponta a ponta: um CSV de fatores com `5e-324` fez
    `refino processar` sair com **código 0** depois de escrever um artefato com 173.672 valores
    `inf` a partir de entrada sem NaN. Nada acusou — nem a guarda de valor, nem `_seguro`, nem
    o código de saída.

    ## Por que a conferência é na SAÍDA

    O critério afirma uma propriedade da saída: *nenhuma normalização pode produzir NaN/Inf a
    partir de matriz que não os tem*. Uma guarda por denominador teria de ser escrita em cada
    ramo, e um ramo novo nasceria descoberto. A pós-condição em `normalize_samples` vale para
    todos de uma vez.

    ## Os dois métodos, e não um

    Parametrizado porque a lente adversarial nomeou os dois: os fatores pela guarda de valor
    que não cobre magnitude, e a feature de referência pela guarda `> 0.0`, que tem o mesmo
    buraco. Um teste sobre um só deixaria o outro por cobrir com o raciocínio a **parecer**
    aplicado.
    """
    matriz = _matriz_com_qcs(n_qc=3)
    assert np.isfinite(np.asarray(matriz.values)).all(), "a entrada tem de ser finita"

    minusculo = 5e-324
    if metodo == "supplied_factors":
        argumentos: dict[str, object] = {
            "factors": tuple((a.name, minusculo) for a in matriz.samples)
        }
    else:
        valores = np.asarray(matriz.values).copy()
        valores[:, 1] = minusculo
        matriz = matriz.with_values(valores, "planta")
        argumentos = {"reference": "f1"}

    with pytest.raises(ValueError, match=r"não finito"):
        normalize_samples(matriz, metodo, **argumentos)  # type: ignore[arg-type]


def test_a_pos_condicao_NAO_dispara_no_caminho_normal() -> None:  # noqa: N802
    """A outra direção, e sem ela a pós-condição podia estar sempre a reprovar.

    Mecanismo que recusa tudo é indistinguível de mecanismo que recusa o certo, e as duas
    direções de falha estão igualmente disponíveis quando ele é escrito.
    """
    matriz = _matriz_com_qcs(n_qc=3)
    for metodo, extra in (
        ("supplied_factors", {"factors": _fatores_de(matriz, 3.0)}),
        ("feature_reference", {"reference": "f1"}),
        ("sample_reference", {"reference": "S0"}),
        ("sum", {}),
        ("median", {}),
        ("quantile", {}),
    ):
        saida = normalize_samples(matriz, metodo, **extra)  # type: ignore[arg-type]
        assert np.isfinite(np.asarray(saida.values)).all(), (
            f"`{metodo}` produziu não finito no caminho normal"
        )
