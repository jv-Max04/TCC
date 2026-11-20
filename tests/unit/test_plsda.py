"""`plsda` — introspecção, orientação em PLS, e o Q² conferido à mão.

O critério **não** está aqui: a medição da nula disparou condição de parada, e nenhum
limiar foi fixado. O que este arquivo verifica são as propriedades da função.

**A matriz com amostra `validate` é SINTÉTICA, e a declaração é requisito (regra 1 do README).**
Medido em 17/09/2026 sobre os quatro `measurements_*.csv` do laboratório: 68 linhas, 61 `sample`, 7
`qc` e **zero** `validate` nos quatro modos. A amostra `validate` de rótulo clínico **não existe no
corpus** — fabricá-la sobre os arquivos reais e apresentá-la como observação seria o que a
regra 1 do README proíbe, e por isso ela é construída à mão e dita como construída. O que os testes
abaixo afirmam é a propriedade de desenho, que não depende do corpus de hoje.
"""

import dataclasses
import math

import numpy as np
import pytest
from sklearn.cross_decomposition import PLSRegression

import refino.statistics.plsda as modulo
from refino.domain.matrix import RefinoMatrix
from refino.domain.metadata import SampleMetadata
from refino.statistics.plsda import PLSDAResult, plsda

pytestmark = pytest.mark.property


def _matriz(valores: np.ndarray, grupos: list[str], tipos: list[str] | None = None) -> RefinoMatrix:
    tipos = tipos if tipos is not None else ["sample"] * len(grupos)
    amostras = tuple(
        SampleMetadata(name=f"S{i}", injection_order=i + 1, sample_type=t, group=g)  # type: ignore[arg-type]
        for i, (g, t) in enumerate(zip(grupos, tipos, strict=True))
    )
    return RefinoMatrix(
        values=valores,
        feature_names=tuple(f"f{j}" for j in range(valores.shape[1])),
        samples=amostras,
        mode="sintetico",
    )


def _separado(n: int, p: int, alteradas: int, fator: float, semente: int) -> RefinoMatrix:
    gerador = np.random.default_rng(semente)
    valores = gerador.normal(10.0, 1.0, size=(2 * n, p))
    valores[n:, :alteradas] += fator
    return _matriz(valores, ["A"] * n + ["B"] * n)


def test_ca_6_6_todos_os_dez_campos_sao_obrigatorios() -> None:
    """Por introspecção: dataclass congelado sem padrão não admite parcial.

    É o mecanismo, não a intenção: não há caminho de código que devolva `scores` sem `q2`, porque
    construir `PLSDAResult` exige os dez.

    **Eram nove**, e o décimo é `cv_scores`, sem padrão pela mesma razão dos outros: um
    escore de validação cruzada servido sem o `q2` ao lado é o mesmo defeito que um de calibração.
    """
    campos = dataclasses.fields(PLSDAResult)
    nomes = {c.name for c in campos}
    assert nomes == {
        "scores",
        "cv_scores",
        "loadings",
        "vip",
        "r2y",
        "q2",
        "permutation_pvalue",
        "n_components",
        "n_permutations",
        "seed",
    }
    for campo in campos:
        assert campo.default is dataclasses.MISSING, (
            f"{campo.name} ganhou valor padrão; com padrão em `q2` ou `permutation_pvalue` "
            "passaria a existir construção sem validação, que é a regra 6 do README"
        )
    assert PLSDAResult.__dataclass_params__.frozen


def test_ca_6_6_a_unica_funcao_publica_devolve_o_resultado_completo() -> None:
    """Nenhum caminho público expõe scores soltos."""
    assert modulo.__all__ == ["PLSDAResult", "plsda"]
    assert plsda.__annotations__["return"] is PLSDAResult


def test_q2_conferido_a_mao_sobre_fixture_pequeno() -> None:
    """`Q² = 1 - PRESS/TSS`, recalculado por fora com a mesma partição.

    Recalcular por fora e não afirmar um número congelado: número congelado não distingue
    "a conta está certa" de "a conta é a mesma de ontem".
    """
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.model_selection import StratifiedKFold

    matriz = _separado(8, 6, 3, 4.0, 7)
    resultado = plsda(matriz, ("A", "B"), n_components=2, cv_folds=4, n_permutations=3, seed=1)

    x = np.asarray(matriz.values)
    y = np.array([0.0] * 8 + [1.0] * 8)
    press = 0.0
    for treino, retido in StratifiedKFold(n_splits=4).split(x, y):
        ajuste = PLSRegression(n_components=2, scale=False).fit(x[treino], y[treino])
        press += float(((y[retido] - np.asarray(ajuste.predict(x[retido])).ravel()) ** 2).sum())
    esperado = 1.0 - press / float(((y - y.mean()) ** 2).sum())

    assert resultado.q2 == pytest.approx(esperado, rel=1e-12)


def test_o_p_valor_de_permutacao_nunca_e_zero_nem_passa_de_um() -> None:
    """A convenção `(k+1)/(n+1)`.

    p-valor zero afirmaria impossibilidade a partir de 200 sorteios.
    """
    resultado = plsda(_separado(9, 8, 8, 8.0, 11), ("A", "B"), n_permutations=20, seed=3)
    assert 0.0 < resultado.permutation_pvalue <= 1.0
    assert resultado.permutation_pvalue >= 1.0 / 21.0


def test_ca_6_7_orientacao_canonica_em_plsda() -> None:
    """Na segunda metade: a matriz negada dá loadings de mesmo sinal."""
    gerador = np.random.default_rng(13)
    valores = gerador.normal(10.0, 1.0, size=(16, 6))
    valores[8:, :3] += 4.0
    grupos = ["A"] * 8 + ["B"] * 8

    direto = plsda(_matriz(valores, grupos), ("A", "B"), n_permutations=2, seed=5)
    negado = plsda(_matriz(-valores, grupos), ("A", "B"), n_permutations=2, seed=5)

    assert np.allclose(np.sign(direto.loadings), np.sign(negado.loadings))


def test_o_vip_e_nao_negativo_e_nao_recebe_orientacao() -> None:
    """Importância não tem sinal, e por isso o helper canônico não a toca."""
    gerador = np.random.default_rng(17)
    valores = gerador.normal(10.0, 1.0, size=(14, 5))
    valores[7:, :2] += 3.0
    grupos = ["A"] * 7 + ["B"] * 7

    direto = plsda(_matriz(valores, grupos), ("A", "B"), n_permutations=2, seed=7)
    negado = plsda(_matriz(-valores, grupos), ("A", "B"), n_permutations=2, seed=7)

    assert np.all(direto.vip >= 0.0)
    assert np.allclose(direto.vip, negado.vip)


def test_cv_folds_acima_do_menor_grupo_levanta_nomeando_as_duas_contagens() -> None:
    """Dobra sem um dos grupos faria o PRESS medir outra coisa."""
    with pytest.raises(ValueError, match="dobras"):
        plsda(_separado(4, 5, 2, 2.0, 19), ("A", "B"), cv_folds=7, n_permutations=2)


def test_grupo_inexistente_e_grupos_iguais_levantam() -> None:
    matriz = _separado(8, 5, 2, 2.0, 23)
    with pytest.raises(ValueError, match="não está entre as amostras de estudo"):
        plsda(matriz, ("A", "Z"), cv_folds=4, n_permutations=2)
    with pytest.raises(ValueError, match="são o mesmo"):
        plsda(matriz, ("A", "A"), cv_folds=4, n_permutations=2)


def test_a_mesma_semente_reproduz_o_p_valor() -> None:
    """Regra 3 do README, e é o fluxo derivado de `(seed, índice)` que a sustenta."""
    matriz = _separado(8, 6, 3, 3.0, 29)
    a = plsda(matriz, ("A", "B"), cv_folds=4, n_permutations=25, seed=41)
    b = plsda(matriz, ("A", "B"), cv_folds=4, n_permutations=25, seed=41)
    assert a.permutation_pvalue == b.permutation_pvalue
    assert a.q2 == b.q2


def test_sementes_diferentes_dao_p_valores_diferentes() -> None:
    """O par causal da semente: sem ele, `seed` poderia ser ignorado e nada acusaria."""
    matriz = _separado(8, 6, 3, 1.0, 31)
    valores = {
        plsda(matriz, ("A", "B"), cv_folds=4, n_permutations=25, seed=s).permutation_pvalue
        for s in (1, 2, 3, 4, 5)
    }
    assert len(valores) > 1, (
        "cinco sementes deram o mesmo p-valor de permutação; então a semente não está sendo "
        "consultada, ou a separação é tão forte que nenhuma permutação a supera"
    )


def test_q2_e_r2y_sao_finitos_no_caso_nao_degenerado() -> None:
    resultado = plsda(_separado(8, 6, 3, 3.0, 37), ("A", "B"), cv_folds=4, n_permutations=5)
    assert np.isfinite(resultado.q2) and np.isfinite(resultado.r2y)
    assert resultado.n_permutations == 5


# ---------------------------------------------------------------------------
# A população é a MÁSCARA e não o rótulo — a irmã de `plsda_multiclasse`
# ---------------------------------------------------------------------------

# O desenho é o do estudo real reduzido — sete controles, e o `ATLL` como o grupo que recebe as
# linhas `validate`. A matriz é SINTÉTICA e dita como tal: nenhum dos quatro arquivos do
# laboratório tem amostra `validate` (medido em 17/09/2026, 0 de 68 nos quatro modos).
N_QC_DO_ESTUDO = 7


def _com_validate(tipo_das_duas: str) -> RefinoMatrix:
    """8 `ATLL` e 8 `NoHTLV` de estudo, DUAS linhas de rótulo `ATLL`, e 7 controles de rótulo `QC`.

    `tipo_das_duas` é a **única** coisa que o par causal move: valores da mesma semente, mesmos
    rótulos, mesma ordem de injeção. Com `"validate"` as duas linhas não são amostras de estudo;
    com `"sample"` são.

    **Sete controles, e o número é requisito e não decoração.** A guarda de `cv_folds` exige ao
    menos 7 membros por grupo, então com menos controles o pedido `groups=("QC", ...)` seria
    recusado pelo **tamanho** — e a recusa não distinguiria as duas guardas, nem o
    teste reprovaria contra a implementação anterior, que aceitava o pedido.
    """
    rotulos = ["ATLL"] * 8 + ["NoHTLV"] * 8 + ["ATLL"] * 2 + ["QC"] * N_QC_DO_ESTUDO
    tipos = ["sample"] * 16 + [tipo_das_duas] * 2 + ["qc"] * N_QC_DO_ESTUDO
    gerador = np.random.default_rng([1212, 7])
    return _matriz(gerador.normal(size=(len(rotulos), 40)), rotulos, tipos)


def test_a_amostra_validate_de_rotulo_clinico_NAO_entra_no_modelo() -> None:  # noqa: N802
    """**`sample_mask` não é o complemento de `qc_mask`**, e é a seleção que tem de o saber.

    Há três valores em `SampleType`. Uma amostra `validate` existe para medir a correção **fora**
    da amostra de treino, e o rótulo clínico dela é o do paciente de onde veio — então ele coincide
    com um grupo de estudo por construção, e não por acidente. Deixá-la entrar contamina a
    população de um modelo que existe para separar grupos clínicos com material que não é dele.

    **A planta cai sobre dado em que as duas seleções DIFEREM, e isso é afirmado ANTES**:
    neste fixture a seleção por rótulo dá **18** linhas e a por máscara dá **16**. Com as duas
    iguais o teste passaria por coincidência — que é o caso dos quatro arquivos reais, onde não há
    `validate` nenhuma e as duas seleções coincidem em 61 linhas.

    **O par causal é o TIPO e nada mais:** as mesmas duas linhas, os mesmos valores e o mesmo
    rótulo, tipadas `validate` numa ponta e `sample` na outra.
    """
    matriz = _com_validate("validate")
    par = ("ATLL", "NoHTLV")
    por_linha = np.array(matriz.groups)
    estudo = np.asarray(matriz.sample_mask)
    por_rotulo = int(np.isin(por_linha, par).sum())
    por_mascara = int((np.isin(por_linha, par) & estudo).sum())
    assert (por_rotulo, por_mascara) == (18, 16), (
        f"a planta não separa: por rótulo entram {por_rotulo} linhas e por máscara {por_mascara}. "
        "Com as duas iguais, as asserções abaixo passariam sobre qualquer das duas seleções"
    )

    resultado = plsda(matriz, par, cv_folds=7, n_permutations=3, seed=42)
    assert resultado.scores.shape[0] == por_mascara, (
        f"o modelo foi ajustado sobre {resultado.scores.shape[0]} linhas e as amostras de estudo "
        f"são {por_mascara}: as duas `validate` de rótulo `ATLL` entraram, e a seleção está a "
        "olhar o rótulo e não a máscara"
    )

    como_estudo = plsda(_com_validate("sample"), par, cv_folds=7, n_permutations=3, seed=42)
    assert como_estudo.scores.shape[0] == por_rotulo, (
        "âncora do par causal: com as duas linhas tipadas `sample` elas TÊM de entrar, senão a "
        "divergência abaixo não seria atribuível ao tipo"
    )
    assert resultado.q2 != como_estudo.q2, (
        f"tipar as duas linhas como `validate` ou como `sample` deu o MESMO Q² ({resultado.q2}): "
        "`sample_type` não está a ser lido de todo"
    )


def test_o_rotulo_dos_QC_pedido_como_GRUPO_e_recusado() -> None:  # noqa: N802
    """**O QC é a régua, e pedi-lo como grupo tem de ser recusado.**

    A guarda de pertinência lia os rótulos de **todas** as linhas, então `groups=("QC", "ATLL")`
    era aceito e devolvia `Q²` e `p` para um contraste entre a mistura de controle e um
    diagnóstico — número que não significa nada, sobre a quantidade que a ferramenta usa como
    régua de tudo o resto. E este caminho **está ligado**: `api/rotas.py` chama `plsda` na
    execução.

    **Medido no `pos_hilic` REAL em 17/09/2026, antes do conserto:** aceito sobre 20 linhas, 7
    controles contra 13 `ATLL`, com `Q² = -0,011996`, `p = 0,15423` e `r2y = 0,40600`. É a mesma
    coincidência que `api/rotas.py` já registou — os QCs são exactamente 7, e `cv_folds=7`
    aceita-os.

    **A âncora vem primeiro e é ela que impede o teste de passar por recusar tudo:** um par de
    grupos de estudo TEM de ser aceito sobre o mesmo fixture.

    **A mensagem é afirmada e não só o tipo.** Sob a máscara o grupo `"QC"` fica com
    zero linhas, e sem a guarda de pertinência a recusa viria da guarda de **tamanho** — um
    `pytest.raises(ValueError)` nu passaria pelo motivo errado, e a mensagem diria ao usuário que
    o rótulo não existe, quando ele existe.
    """
    matriz = _com_validate("validate")

    aceito = plsda(matriz, ("ATLL", "NoHTLV"), cv_folds=7, n_permutations=3, seed=42)
    assert np.isfinite(aceito.q2), "âncora: um par de grupos de estudo é legítimo neste fixture"

    n_controles = int(np.asarray(matriz.qc_mask).sum())
    assert n_controles >= 7, (
        f"há {n_controles} controles e `cv_folds=7`: a implementação anterior teria recusado o "
        "pedido abaixo pelo TAMANHO, e o teste passaria pelo motivo errado"
    )

    with pytest.raises(ValueError) as erro:
        plsda(matriz, ("QC", "ATLL"), cv_folds=7, n_permutations=3, seed=42)
    texto = str(erro.value)
    assert "'QC'" in texto and "não está entre as amostras de estudo" in texto, texto
    assert "dobras" not in texto, (
        f"a recusa veio da guarda de TAMANHO e não da de população: {texto}"
    )
    assert "não existe nesta matriz" not in texto, (
        f"a mensagem afirma que o rótulo não existe, e ele existe — é o rótulo dos controles "
        f"desta matriz. Quem a lê vai procurar um erro de escrita que não há: {texto}"
    )


# ---------------------------------------------------------------------------
# Os escores de VALIDAÇÃO CRUZADA
# ---------------------------------------------------------------------------
#
# Cada amostra projetada pelo modelo da dobra que NÃO a viu, e girada por Procrustes ortogonal
# para o referencial do modelo de calibração. É o que o gráfico de escores passa a mostrar: os de
# calibração tendem a separar os grupos, e com muito mais features que amostras — 200 contra 40
# aqui — separam-nos mesmo sobre rótulos sem informação nenhuma; os de validação cruzada só separam
# quando o modelo prediz. A separação dos de calibração é MEDIDA, e não desenho: em
# 40 amostras de ruído com rótulos alternados, 20 matrizes por contagem (`[99, p, s]`), a mediana do
# `|d|` é 4,33 com 200 features e cai para 0,67 e 1,02 com 5 e 10.
#
# **As matrizes dos dois cenários são as MESMAS, e só os rótulos mudam.** Os números abaixo foram
# medidos antes de escrever os testes, sobre estas 20 matrizes e mais 180 da mesma receita (sementes
# `[20261106, lote, semente]`, dez lotes de vinte): com os rótulos permutados, o `|d|` de calibração
# nunca ficou abaixo de 3,16 e o de validação cruzada teve mediana 0,357 e quantil de 99% de 1,083;
# com os verdadeiros, o `d` de validação cruzada nunca ficou abaixo de 3,44. Nas 20 daqui: 3,77, a
# mediana 0,300, e 3,51. Os limiares — 2 e 0,5 — vêm dessa distribuição, com a folga à vista.

POR_GRUPO_DOS_CENARIOS = 20
FEATURES_DOS_CENARIOS = 200
ALTERADAS_DOS_CENARIOS = 10
EFEITO_FORTE = 2.0
MATRIZES_DOS_CENARIOS = 20


def _separacao(escores: np.ndarray, y: np.ndarray) -> float:
    """O `d` de Cohen no componente 1: centróide de `y = 1` menos o de `y = 0`, em desvios.

    **Calculado aqui e não importado**: é o oráculo destes testes, e um oráculo emprestado do módulo
    sob teste erraria junto com ele.
    """
    a, b = escores[y == 0.0, 0], escores[y == 1.0, 0]
    combinado = math.sqrt(
        ((a.size - 1) * a.var(ddof=1) + (b.size - 1) * b.var(ddof=1)) / (a.size + b.size - 2)
    )
    return float((b.mean() - a.mean()) / combinado)


def _cenario(semente: int, rotulos: list[str]) -> tuple[RefinoMatrix, np.ndarray]:
    """40 amostras e 200 features de ruído, as 10 primeiras deslocadas de 2 desvios nas 20 últimas.

    **O deslocamento é das LINHAS e não dos rótulos**: a matriz é a mesma qualquer que seja
    `rotulos`, e é isso que faz dos dois cenários um par causal — muda só a informação que os
    rótulos carregam sobre o dado.
    """
    gerador = np.random.default_rng([20261106, 0, semente])
    valores = gerador.normal(10.0, 1.0, size=(2 * POR_GRUPO_DOS_CENARIOS, FEATURES_DOS_CENARIOS))
    valores[POR_GRUPO_DOS_CENARIOS:, :ALTERADAS_DOS_CENARIOS] += EFEITO_FORTE
    return _matriz(valores, rotulos), np.array([float(r == "B") for r in rotulos])


def test_com_rotulos_PERMUTADOS_a_calibracao_separa_e_a_validacao_cruzada_NAO() -> None:  # noqa: N802
    """**O falso positivo que a regra 6 do README existe para tornar visível, agora na própria
    figura.**

    Os rótulos permutados alternam dentro de cada metade da matriz: cada um dos dois "grupos" tem
    10 linhas deslocadas e 10 não, então os rótulos não carregam informação NENHUMA sobre o
    deslocamento. Com 200 features e 40 amostras o ajuste acha mesmo assim uma direção que separa
    os rótulos — é o que os escores de calibração mostram —, e a validação cruzada não a reencontra
    nas amostras que o modelo não viu.

    **A primeira asserção é a âncora, e sem ela a segunda passaria por vacuidade:** se a calibração
    também não separasse, "a validação cruzada não separa" seria verdade sobre um dado em que os
    dois escores não diferem.
    """
    permutados = ["B" if i % 2 else "A" for i in range(2 * POR_GRUPO_DOS_CENARIOS)]
    assert permutados.count("B") == POR_GRUPO_DOS_CENARIOS
    calibracao: list[float] = []
    validacao: list[float] = []
    for semente in range(MATRIZES_DOS_CENARIOS):
        matriz, y = _cenario(semente, permutados)
        resultado = plsda(matriz, ("A", "B"), n_permutations=1, seed=semente)
        calibracao.append(abs(_separacao(resultado.scores, y)))
        validacao.append(abs(_separacao(resultado.cv_scores, y)))

    assert min(calibracao) > 2.0, (
        f"âncora: com rótulos sem informação a calibração devia separar por sobreajuste, e o menor "
        f"|d| foi {min(calibracao):.3f}. Sem isto o resto não distingue os dois escores: "
        f"{[round(v, 3) for v in calibracao]}"
    )
    mediana = float(np.median(validacao))
    assert mediana < 0.5, (
        f"os escores de validação cruzada separam rótulos sem informação: mediana de |d| = "
        f"{mediana:.3f} em {MATRIZES_DOS_CENARIOS} matrizes — {[round(v, 3) for v in validacao]}"
    )


def test_com_rotulos_VERDADEIROS_e_efeito_forte_a_validacao_cruzada_SEPARA() -> None:  # noqa: N802
    """A outra ponta do par: os MESMOS dados, com os rótulos que o deslocamento segue.

    **Do mesmo lado que a calibração, e não só longe de zero.** O sinal do `d` de calibração é o do
    referencial canônico; um escore de validação cruzada que separasse com o sinal trocado estaria
    noutro referencial, e a figura poria os dois grupos nos lados opostos dos de calibração.
    """
    verdadeiros = ["A"] * POR_GRUPO_DOS_CENARIOS + ["B"] * POR_GRUPO_DOS_CENARIOS
    alinhadas: list[float] = []
    for semente in range(MATRIZES_DOS_CENARIOS):
        matriz, y = _cenario(semente, verdadeiros)
        resultado = plsda(matriz, ("A", "B"), n_permutations=1, seed=semente)
        d_calibracao = _separacao(resultado.scores, y)
        alinhadas.append(_separacao(resultado.cv_scores, y) * math.copysign(1.0, d_calibracao))

    assert min(alinhadas) > 2.0, (
        f"com efeito forte e rótulos verdadeiros os escores de validação cruzada deviam separar do "
        f"lado da calibração; o menor d foi {min(alinhadas):.3f} — "
        f"{[round(v, 3) for v in alinhadas]}"
    )


def _matriz_que_vira() -> tuple[RefinoMatrix, np.ndarray, np.ndarray]:
    """Uma matriz em que a orientação canônica VIRA o componente 1 que o `sklearn` devolve.

    **O `sklearn` já põe positivo o maior PESO de cada componente, e a casa põe positiva a maior
    CARGA** (`statistics/orientation.py`). As duas convenções só discordam quando features
    diferentes dominam o peso e a carga — o que pede variâncias heterogêneas, como no dado bruto.
    Doze features com escala de 1 a 100, as três primeiras deslocadas de 3 desvios no segundo grupo.

    **A semente é a primeira de uma busca, e não a melhor:** sobre `[20261106, 7, s]` com `s` de 0 a
    93, oito sementes viram o componente 1, e esta é a primeira delas. Com ela, medido antes do
    teste: correlação de 0,832 entre os componentes 1 de calibração e de validação cruzada; -0,832
    sem a virada nas rotações; -0,796 sem o Procrustes.
    """
    gerador = np.random.default_rng([20261106, 7, 16])
    escala = 10.0 ** gerador.uniform(0.0, 2.0, size=12)
    valores = gerador.normal(0.0, 1.0, size=(20, 12)) * escala
    valores[10:, :3] += 3.0 * escala[:3]
    y = np.array([0.0] * 10 + [1.0] * 10)
    return _matriz(valores, ["A"] * 10 + ["B"] * 10), valores, y


def test_a_virada_canonica_vai_as_ROTACOES_antes_do_procrustes(  # noqa: N802
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """**O referencial do Procrustes é o do modelo de calibração JÁ orientado.**

    Os escores de calibração saem de `orient_canonically`; se o alvo do Procrustes fossem as
    rotações cruas do `sklearn`, cada componente virado estaria com o sinal trocado nos escores de
    validação cruzada — os mesmos pontos no lado oposto da figura, sem nada a levantar.

    **Três desfechos sobre a MESMA matriz, e as duas plantas entram pelo módulo sob teste:** sem a
    virada nas rotações, e sem o Procrustes. As duas põem o componente 1 de validação cruzada contra
    o de calibração. **A âncora vem primeiro:** nesta matriz a orientação canônica vira de facto o
    componente 1, senão a primeira planta seria a identidade e não separaria nada.
    """
    matriz, valores, y = _matriz_que_vira()
    cru = PLSRegression(n_components=2, scale=False).fit(valores, y)
    cargas = np.asarray(cru.x_loadings_)[:, 0]
    assert cargas[int(np.argmax(np.abs(cargas)))] < 0.0, (
        "âncora: a orientação canônica não vira o componente 1 nesta matriz, e a planta das "
        "rotações seria a identidade"
    )

    def alinhamento() -> float:
        resultado = plsda(matriz, ("A", "B"), n_permutations=1, seed=3)
        return float(np.corrcoef(resultado.scores[:, 0], resultado.cv_scores[:, 0])[0, 1])

    assert alinhamento() > 0.5

    chamadas: list[str] = []

    def rotacoes_cruas(modelo: PLSRegression) -> np.ndarray:
        chamadas.append("rotações")
        return np.asarray(modelo.x_rotations_)

    with monkeypatch.context() as planta:
        planta.setattr(modulo, "_rotacoes_no_referencial_dos_escores", rotacoes_cruas)
        sem_virar = alinhamento()
    assert chamadas == ["rotações"], "a planta das rotações não aterrou"
    assert sem_virar < 0.0, f"sem a virada nas rotações a correlação ficou {sem_virar:.3f}"

    def sem_giro(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, float]:
        chamadas.append("giro")
        return np.eye(a.shape[1]), 1.0

    with monkeypatch.context() as planta:
        planta.setattr(modulo, "orthogonal_procrustes", sem_giro)
        sem_procrustes = alinhamento()
    assert chamadas.count("giro") == 7, "a planta do Procrustes não aterrou nas sete dobras"
    assert sem_procrustes < 0.0, f"sem o Procrustes a correlação ficou {sem_procrustes:.3f}"


def test_a_mesma_semente_reproduz_os_escores_de_validacao_cruzada() -> None:
    """Regra 3 do README sobre o campo novo — e a semente nem chega a ele.

    **A âncora de não-vacuidade vem primeiro:** dois vetores vazios, ou dois de `nan`, seriam
    iguais por construção.

    **E sementes diferentes dão os MESMOS escores, por desenho e não por acaso:** a partição é a do
    `StratifiedKFold` sem embaralhar, como a do `Q²`, e o PLS não sorteia nada. A semente governa
    só as permutações do `p`. Se a partição passar a ser sorteada, esta asserção é a que reprova —
    e é a que tem de reprovar, porque a docstring de `_validacao_cruzada` passaria a mentir.
    """
    matriz = _separado(8, 6, 3, 3.0, 29)
    a = plsda(matriz, ("A", "B"), cv_folds=4, n_permutations=5, seed=41)
    b = plsda(matriz, ("A", "B"), cv_folds=4, n_permutations=5, seed=41)

    assert a.cv_scores.shape == a.scores.shape == (16, 2)
    assert np.isfinite(a.cv_scores).all()
    assert float(np.ptp(a.cv_scores[:, 0])) > 0.0, "escores constantes seriam iguais por vacuidade"
    assert np.array_equal(a.cv_scores, b.cv_scores)

    outra = plsda(matriz, ("A", "B"), cv_folds=4, n_permutations=5, seed=42)
    assert np.array_equal(a.cv_scores, outra.cv_scores)
