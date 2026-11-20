"""A predição de vias — as quatro correções que a chamada ingênua não faz.

**Quatro destes testes são pares causais, e são eles que sustentam o arquivo:**

| Teste | O que ele separa |
|---|---|
| `test_seeds_diferentes_dao_p_diferente` | impede o de determinismo de passar por vacuidade |
| `test_politica_de_massa_neutra_muda_a_conta` | prova que o ramo da política está LIGADO |
| `test_p_de_permutacao_conta_o_empate_como_pelo_menos_tao_extremo` | `1/101` deles, `1` nosso |
| `test_a_contaminacao_entre_modos_e_real` | prova que o defeito de `_rede_privada` é REAL |

`test_sem_feature_significativa_NENHUMA_via_passa` **não** forma par dentro deste arquivo: o par
dele — o mesmo teste com o p da biblioteca, a reprovar — foi corrido uma vez fora da suíte, por um
script.

O último é o mais importante e o menos óbvio: sem ele, `test_modo_positivo_sobrevive_a_um_negativo`
passaria igualmente numa árvore em que a cópia privada tivesse sido removida — porque
o caminho sequencial recalcula os adutos antes de usar, e só a **partilha** dos objetos
é que morde. Ele exercita a biblioteca crua e afirma que a corrupção acontece.
"""

import numpy as np
import pytest

from refino.domain.labels import InvalidLabelError
from refino.io.readers import load_matrix
from refino.pathways.analise import (
    FAIXA_DE_MASSA,
    MINIMO_COM_CORTE_AUTOMATICO,
    EntradaPequenaDemaisError,
    EntradaRecusadaError,
    EntradaSemFeatureAnalisavelError,
    ViaPredita,
    _chave_de_ordem,
    _confere_faixa,
    p_de_permutacao,
    prediz_vias,
)
from refino.statistics.univariate import univariate_test
from tests.conftest import measurements_path, sample_list_path

# O menor dos quatro arquivos (2.039 features), para manter o portão de commit curto.
# A escolha é de CUSTO e não de conteúdo: os mesmos caminhos correm em qualquer modo.
MODO = "neg_hilic"
GRUPOS = ("ATLL", "NoHTLV")


@pytest.fixture(scope="module")
def entrada() -> tuple[tuple[str, ...], np.ndarray, np.ndarray]:
    """Nomes, p-valores e estatísticas do dado REAL, uma vez por módulo.

    Dado real e não factory (regra 1 do README): a predição de vias depende da massa de
    cada feature casar com um composto do modelo, e massa inventada casa por acaso ou
    não casa nunca — nos dois casos o teste mediria a factory.
    """
    matriz = load_matrix(measurements_path(MODO), sample_list_path(MODO), MODO)
    resultado = univariate_test(matriz, GRUPOS)
    return resultado.feature, resultado.pvalue, resultado.statistic


def _roda(entrada, **kwargs):
    nomes, p, t = entrada
    kwargs.setdefault("modo_de_ionizacao", "negative")
    kwargs.setdefault("seed", 0)
    return prediz_vias(nomes, p, t, **kwargs)


def _assinatura(relatorio) -> list[tuple[str, float]]:
    return [(v.nome, v.pvalue) for v in relatorio.vias]


# ---------------------------------------------------------------------------
# Determinismo — regra 3 do README
# ---------------------------------------------------------------------------


def test_mesma_seed_da_resultado_identico(entrada) -> None:
    assert _assinatura(_roda(entrada, seed=0)) == _assinatura(_roda(entrada, seed=0))


def test_seeds_diferentes_dao_p_diferente(entrada) -> None:
    """O par causal do determinismo: sem ele, o teste acima é vácuo.

    Se a semente não alcançasse o permutador, duas sementes dariam o mesmo resultado
    — e o teste de determinismo passaria por o permutador não estar a ser exercitado,
    e não por a semeadura funcionar. Negativo verde
    é a verificação vazia mais convincente que existe.
    """
    a = dict(_assinatura(_roda(entrada, seed=0)))
    b = dict(_assinatura(_roda(entrada, seed=7)))
    assert any(a[via] != b.get(via) for via in a), (
        "duas sementes deram p idêntico em TODA via: a semeadura não está a alcançar "
        "as permutações, e o teste de determinismo não prova nada"
    )


def test_a_seed_nao_vaza_para_o_processo(entrada) -> None:
    """Semear sem restaurar tornaria determinística toda estocasticidade seguinte.

    O modo de falha é insidioso e não é deste módulo: um teste de outro pacote que
    dependa de `random` passaria a ver a mesma sequência a cada chamada, e o defeito
    apareceria como flutuação inexplicável num arquivo que ninguém tocou.
    """
    import random

    random.seed(12345)
    esperado = [random.random() for _ in range(3)]

    random.seed(12345)
    _roda(entrada, seed=99)
    obtido = [random.random() for _ in range(3)]

    assert obtido == esperado


# ---------------------------------------------------------------------------
# A política de massa neutra
# ---------------------------------------------------------------------------


def test_politica_de_massa_neutra_muda_a_conta(entrada) -> None:
    """Par causal: as duas políticas TÊM de analisar populações diferentes.

    A planta cai sobre dado em que as duas diferem — o `neg_hilic` tem 148 massas
    neutras entre 2.039 features. Num arquivo sem nenhuma, este teste passaria por
    coincidência, e é por isso que a diferença é afirmada em número e não só em
    desigualdade.
    """
    incluindo = _roda(entrada, politica_de_massa_neutra="primary_ion")
    excluindo = _roda(entrada, politica_de_massa_neutra="exclude")

    assert incluindo.recorte.massas_neutras_incluidas == 148
    assert incluindo.recorte.massas_neutras_excluidas == 0
    assert excluindo.recorte.massas_neutras_incluidas == 0
    assert excluindo.recorte.massas_neutras_excluidas == 148
    assert excluindo.recorte.analisadas < incluindo.recorte.analisadas


def test_o_recorte_fecha_a_conta(entrada) -> None:
    """Toda feature recebida tem um destino nomeado — nenhuma desaparece.

    É a asserção que impede saída silenciosa: sem ela, uma feature descartada por
    causa não prevista simplesmente não apareceria, e a tabela de vias no fim não
    denunciaria a ausência.
    """
    relatorio = _roda(entrada)
    r = relatorio.recorte
    contabilizadas = (
        r.analisadas
        + r.massas_neutras_excluidas
        + r.nao_interpretadas
        + r.sem_estatistica
        + r.fora_da_faixa_de_massa
    )
    assert contabilizadas == r.recebidas, (
        f"{r.recebidas} features recebidas e {contabilizadas} contabilizadas: "
        "alguma saiu por caminho sem causa nomeada"
    )


# ---------------------------------------------------------------------------
# Isolamento entre modos de ionização
# ---------------------------------------------------------------------------


def test_a_contaminacao_entre_modos_e_real() -> None:
    """Prova que o defeito que `_rede_privada` corrige EXISTE na biblioteca crua.

    Sem este teste, o seguinte passaria numa árvore em que a cópia privada tivesse
    sido removida — o caminho sequencial recalcula os adutos antes de usar, e a
    partilha só morde quando alguém segura uma referência. Aqui a referência é
    segurada de propósito.

    A sonda cai num composto com adutos NÃO VAZIOS, e isso é requisito: o primeiro
    composto do modelo não tem nenhum, e uma planta sobre ele compararia duas listas
    vazias e não separaria coisa nenhuma.
    """
    from mummichog.JSON_metabolicModels import metabolicModels
    from mummichog.models import metabolicNetwork

    a = metabolicNetwork(metabolicModels["human_model_mfn"])
    a.update_Compounds_adducts(mode="pos_default")
    chave = next(k for k, v in a.Compounds.items() if v.get("adducts"))
    antes = dict(a.Compounds[chave]["adducts"])
    assert antes, "a sonda caiu num composto sem adutos e não separaria nada"

    b = metabolicNetwork(metabolicModels["human_model_mfn"])
    b.update_Compounds_adducts(mode="negative")

    assert dict(a.Compounds[chave]["adducts"]) != antes, (
        "a biblioteca deixou de partilhar os compostos entre modelos. Se isso for "
        "verdade, `_rede_privada` virou cerimônia e a docstring dela mente — "
        "confira antes de remover."
    )


def test_modo_positivo_sobrevive_a_um_negativo() -> None:
    """A correção, do lado de fora: pos → neg → pos devolve o primeiro pos.

    É a propriedade que a janela precisa: quatro modos, trabalho em threads, e uma
    análise não pode mudar o resultado de outra. O desfecho de um defeito aqui não
    seria erro — seria uma tabela de vias plausível e errada.
    """
    matriz = load_matrix(measurements_path("pos_hilic"), sample_list_path("pos_hilic"), "pos_hilic")
    pos = univariate_test(matriz, GRUPOS)
    primeiro = prediz_vias(
        pos.feature, pos.pvalue, pos.statistic, modo_de_ionizacao="positive", seed=0
    )

    matriz_neg = load_matrix(measurements_path(MODO), sample_list_path(MODO), MODO)
    neg = univariate_test(matriz_neg, GRUPOS)
    prediz_vias(neg.feature, neg.pvalue, neg.statistic, modo_de_ionizacao="negative", seed=0)

    segundo = prediz_vias(
        pos.feature, pos.pvalue, pos.statistic, modo_de_ionizacao="positive", seed=0
    )
    assert _assinatura(primeiro) == _assinatura(segundo)


# ---------------------------------------------------------------------------
# Guardas
# ---------------------------------------------------------------------------


def test_vetores_desalinhados_levantam(entrada) -> None:
    nomes, p, t = entrada
    with pytest.raises(ValueError, match="comprimento diferente"):
        prediz_vias(nomes, p[:-1], t, modo_de_ionizacao="negative")


@pytest.mark.parametrize(
    ("campo", "valor"),
    [
        ("modo_de_ionizacao", "pos"),
        ("modo_de_ionizacao", "POSITIVE"),
        ("politica_de_massa_neutra", "incluir"),
        ("busca_de_adutos", "amplo"),
        ("busca_de_adutos", "generic_positive"),
    ],
)
def test_rotulo_invalido_e_recusado_na_fronteira(entrada, campo: str, valor: str) -> None:
    """`Literal` é análise estática, e estes dois vêm da CLI e da janela.

    `"pos"` é o caso caro: é o que o mummichog chama de modo positivo, então é o
    valor que alguém escreveria de memória — e sem a guarda ele cairia no ramo
    negativo, deslocando toda massa neutra em dois prótons sem nada acusar.
    """
    nomes, p, t = entrada
    kwargs = {"modo_de_ionizacao": "negative", campo: valor}
    with pytest.raises(InvalidLabelError):
        prediz_vias(nomes, p, t, **kwargs)


def test_entrada_sem_feature_analisavel_levanta_com_as_causas(entrada) -> None:
    """Lista vazia não vira relatório de zero vias.

    Os dois desfechos pedem ações opostas — "nada entrou" manda conferir a entrada,
    "nada enriqueceu" é resultado —, e colapsá-los num relatório vazio os tornaria
    indistinguíveis.
    """
    nomes, _, _ = entrada
    nan = np.full(len(nomes), np.nan)
    with pytest.raises(ValueError, match="nenhuma feature sobrou"):
        prediz_vias(nomes, nan, nan, modo_de_ionizacao="negative")


def test_confere_faixa_reprova_quando_a_biblioteca_diverge() -> None:
    """O negativo plantado do pino da faixa de massa.

    Sem ele, `_confere_faixa` seria verificação construída e nunca exercitada — e uma
    faixa alargada pela biblioteca faria a nossa contagem de exclusões acusar features
    que ela de facto analisou, mentindo na direção tranquilizadora.
    """

    class FalsoConfig:
        MASS_RANGE = (10, 5000)

    with pytest.raises(RuntimeError, match="faixa de massa do mummichog mudou"):
        _confere_faixa(FalsoConfig)


def test_confere_faixa_aceita_a_biblioteca_instalada() -> None:
    """E a outra direção: com a faixa atual, ele NÃO acusa."""
    from mummichog import config

    _confere_faixa(config)
    assert tuple(float(x) for x in config.MASS_RANGE) == FAIXA_DE_MASSA


# ---------------------------------------------------------------------------
# O conteúdo do relatório
# ---------------------------------------------------------------------------


def test_o_relatorio_carrega_os_parametros_que_valeram(entrada) -> None:
    """Inclusive o corte, que é escolhido pela biblioteca quando não se fixa.

    Guardar só o argumento deixaria `None` no registro, e a pergunta "sob que corte
    esta tabela vale?" ficaria sem resposta meses depois.
    """
    relatorio = _roda(entrada, corte_de_significancia=None)
    assert relatorio.recorte.corte_de_significancia > 0
    assert relatorio.modo_de_ionizacao == "negative"
    assert relatorio.versao_do_modelo == "MFN_1.10.4"
    assert relatorio.permutacoes == 100


def test_corte_fixado_pelo_usuario_e_respeitado(entrada) -> None:
    relatorio = _roda(entrada, corte_de_significancia=0.05)
    assert relatorio.recorte.corte_de_significancia == pytest.approx(0.05)


def test_significativas_e_subconjunto_e_nao_o_resultado(entrada) -> None:
    """A conveniência não corta o resultado.

    Reduzir `vias` ao que passa faria "nenhuma via significativa" — desfecho legítimo
    e informativo — parecer falha da análise.
    """
    relatorio = _roda(entrada)
    assert len(relatorio.vias) > len(relatorio.significativas) or not relatorio.significativas
    assert all(v.pvalue < 0.05 for v in relatorio.significativas)
    assert set(relatorio.significativas) <= set(relatorio.vias)


# ---------------------------------------------------------------------------
# A busca de adutos — o parâmetro que a comparação contra o MetaboAnalystR criou
# ---------------------------------------------------------------------------


def test_a_busca_ampla_procura_mais_formas_ionicas(entrada) -> None:
    """Par causal do parâmetro: `broad` TEM de procurar mais que `default`.

    **Este teste é a amarra de `AdductSearch`**, que é isento de registro de
    implementações porque indexa a tabela de adutos do mummichog — registro de terceiro,
    sem chaves nossas com que bijetar. O que o prende é esta desigualdade medida.

    Se a biblioteca reordenar as listas dele, ou se o nosso mapa apontar as duas entradas
    para a mesma, o teste reprova em vez de a diferença sumir em silêncio.
    """
    estreita = _roda(entrada, modo_de_ionizacao="positive", busca_de_adutos="default")
    ampla = _roda(entrada, modo_de_ionizacao="positive", busca_de_adutos="broad")

    assert estreita.adutos_procurados == 8
    assert ampla.adutos_procurados == 25
    assert ampla.adutos_procurados > estreita.adutos_procurados


def test_a_busca_ampla_MUDA_o_resultado(entrada) -> None:  # noqa: N802
    """E a contagem maior tem de chegar ao RESULTADO, não só ao relatório.

    Sem esta metade, um mapa correto com o valor nunca repassado à biblioteca passaria no
    teste acima: `adutos_procurados` sairia certo e a análise correria sempre com a lista
    estreita — o parâmetro seria decorativo.
    """
    estreita = _roda(entrada, modo_de_ionizacao="positive", busca_de_adutos="default")
    ampla = _roda(entrada, modo_de_ionizacao="positive", busca_de_adutos="broad")

    detectados_estreita = {v.nome: v.detectados_na_via for v in estreita.vias}
    detectados_ampla = {v.nome: v.detectados_na_via for v in ampla.vias}
    assert detectados_estreita != detectados_ampla, (
        "as duas buscas deram exactamente os mesmos compostos detectados por via: "
        "a lista ampla não chegou à biblioteca"
    )
    # E a direção é a medida: procurar mais formas iônicas nunca acha MENOS compostos.
    assert all(
        detectados_ampla[nome] >= detectados_estreita[nome] for nome in detectados_estreita
    ), "a busca ampla detectou MENOS compostos nalguma via, o que inverte a medição de 18/09/2026"


def test_no_modo_negativo_a_busca_ampla_E_A_MESMA(entrada) -> None:  # noqa: N802
    """O não-efeito declarado, AFIRMADO em vez de escondido.

    O mummichog publica uma lista só para a polaridade negativa, então os dois valores de
    `busca_de_adutos` selecionam a mesma. **Isso é fato da biblioteca e não defeito
    nosso** — e fica num teste porque um parâmetro que não faz nada em silêncio é pior que
    parâmetro nenhum: quem o mexe acredita ter mudado alguma coisa.

    Se a biblioteca publicar uma lista ampla negativa, este teste reprova e obriga a
    estender o mapa — que é o gatilho que uma nota em prosa não teria.
    """
    a = _roda(entrada, modo_de_ionizacao="negative", busca_de_adutos="default")
    b = _roda(entrada, modo_de_ionizacao="negative", busca_de_adutos="broad")

    assert a.adutos_procurados == b.adutos_procurados == 16
    assert _assinatura(a) == _assinatura(b)


def test_o_relatorio_carrega_a_busca_que_valeu(entrada) -> None:
    """`adutos_procurados` é campo próprio e não derivado do rótulo.

    No modo negativo o rótulo diz `broad` e a busca é a mesma de `default`. Quem lesse só
    o rótulo concluiria que mudou a busca; este número diz o que aconteceu.
    """
    relatorio = _roda(entrada, modo_de_ionizacao="negative", busca_de_adutos="broad")
    assert relatorio.busca_de_adutos == "broad"
    assert relatorio.adutos_procurados == 16


def test_o_PADRAO_da_busca_de_adutos_e_a_AMPLA(entrada) -> None:  # noqa: N802
    """O padrão é `broad`, e a escolha é do usuário.

    **Sem esta asserção, o padrão reverteria em silêncio.** Quem não passa o argumento é
    quem mais importa: a janela passa-o, mas toda outra chamada — CLI futura, teste,
    script — recebe o padrão, e um `default` que voltasse ali produziria números que não
    conversam com o MetaboAnalyst sem nada acusar.

    A medição que sustenta a escolha: concordância de Spearman 0,36 com `default` contra
    **0,91** com `broad`, e topo-10 em comum 6/10 contra **9/10**.
    """
    relatorio = _roda(entrada, modo_de_ionizacao="positive")
    assert relatorio.busca_de_adutos == "broad"
    assert relatorio.adutos_procurados == 25


def test_rede_privada_ISOLA_de_facto_os_dois_modos() -> None:  # noqa: N802
    """O par causal que faltava: remover a cópia de `_rede_privada` TEM de reprovar aqui.

    **O achado foi medido por mutação:** com a linha da cópia privada removida,
    `test_a_contaminacao_entre_modos_e_real` e `test_modo_positivo_sobrevive_a_um_negativo`
    ficavam os dois **verdes**. O primeiro exercita a biblioteca crua e nunca passa por
    `_rede_privada`; o segundo é sequencial, e o caminho sequencial recalcula os adutos antes de
    usar, então a partilha não morde.

    Os dois juntos não testavam a correção: verificação construída não é verificação
    exercitada, cometida no commit em
    que escrevi sobre ela.

    **O que morde é SEGURAR a referência através da fronteira**, que é o que a janela faz
    quando duas análises coexistem. Aqui as duas redes são construídas pelo caminho de
    produção e a primeira é relida DEPOIS de a segunda existir.
    """
    from mummichog.JSON_metabolicModels import metabolicModels
    from mummichog.models import metabolicNetwork

    from refino.pathways.analise import _rede_privada

    a = _rede_privada(metabolicModels, metabolicNetwork, "pos_default")
    chave = next(k for k, v in a.Compounds.items() if v.get("adducts"))
    antes = dict(a.Compounds[chave]["adducts"])
    assert antes, "a sonda caiu num composto sem adutos e não separaria nada"

    b = _rede_privada(metabolicModels, metabolicNetwork, "negative")

    assert a.Compounds[chave] is not b.Compounds[chave], (
        "as duas redes partilham o objeto do composto: a cópia privada não isolou, e uma "
        "análise concorrente noutro modo reescreveria os adutos desta"
    )
    assert dict(a.Compounds[chave]["adducts"]) == antes, (
        "construir a rede negativa mudou os adutos da positiva. É o defeito que "
        "`_rede_privada` existe para corrigir, e ele está de volta"
    )
    assert dict(b.Compounds[chave]["adducts"]) != antes, (
        "as duas redes têm os mesmos adutos: ou o modo não foi aplicado, ou a sonda caiu "
        "num composto que os dois modos tratam igual — nos dois casos nada foi separado"
    )


def test_o_modelo_GLOBAL_fica_intacto(entrada) -> None:  # noqa: N802
    """E a outra direção: uma predição real não deixa marca no modelo do módulo.

    `test_rede_privada_ISOLA_de_facto_os_dois_modos` prova o isolamento entre duas redes
    nossas. Este prova que o dicionário do MÓDULO — que é processo-global e partilhado com
    qualquer outro consumidor do mummichog — não é tocado por uma chamada completa.
    """
    from mummichog.JSON_metabolicModels import metabolicModels

    compostos = metabolicModels["human_model_mfn"]["Compounds"]
    com_adutos_antes = sum(1 for c in compostos.values() if c.get("adducts"))

    _roda(entrada)

    com_adutos_depois = sum(1 for c in compostos.values() if c.get("adducts"))
    assert com_adutos_depois == com_adutos_antes, (
        f"o modelo global tinha {com_adutos_antes} compostos com adutos e passou a ter "
        f"{com_adutos_depois}: a predição escreveu no dicionário do módulo"
    )


def test_os_candidatos_NAO_SAO_VAZIOS_numa_corrida_real(entrada) -> None:  # noqa: N802
    """A asserção de PRESENÇA que faltava — e sem ela o campo ficou vazio sem nada acusar.

    **Medido antes da correção de 18/09/2026: 119 vias, ZERO candidatos em todas.**
    `analise.py` lia `chosen_compounds`, que nasce `[]` e só é preenchido por
    `collect_hit_Trios()` — que este módulo nunca chama.

    O defeito era caro por ser exactamente onde não se podia errar: estes campos são o que
    sustenta a ressalva de escopo que a aba mostra, *«candidatos compatíveis em massa, não
    identificações»*. Um campo vazio ao lado de uma sobreposição não nula faz quem lê
    concluir o oposto.

    **Nenhum teste o apanhava**, e a razão é a forma da asserção: o único que mencionava
    candidatos construía o `ViaPredita` à mão. Asserção de ausência é satisfeita por
    vacuidade; era a de presença que faltava, e ela tem de correr sobre dado real.
    """
    relatorio = _roda(entrada)
    com_sobreposicao = [v for v in relatorio.vias if v.sobreposicao > 0]
    assert com_sobreposicao, (
        "âncora: nenhuma via com sobreposição nesta corrida, e a asserção abaixo seria vácua"
    )

    sem_candidatos = [v.nome for v in com_sobreposicao if not v.compostos_candidatos]
    assert not sem_candidatos, (
        f"{len(sem_candidatos)} de {len(com_sobreposicao)} vias têm sobreposição > 0 e "
        f"NENHUM candidato: {sem_candidatos[:3]}. O campo que explica a ressalva de escopo "
        "está vazio, e a tela afirmaria o contrário do verdadeiro"
    )

    for via in com_sobreposicao:
        assert len(via.nomes_candidatos) == len(via.compostos_candidatos), (
            f"«{via.nome}» tem {len(via.compostos_candidatos)} identificadores e "
            f"{len(via.nomes_candidatos)} nomes: as duas tuplas deixaram de estar alinhadas, "
            "e o nome na posição i não é o do identificador na posição i"
        )


def test_os_candidatos_vem_ORDENADOS(entrada) -> None:  # noqa: N802
    """A ordenação é a regra 3 do README, não arrumação.

    A fonte é um `set` de objetos sem `__hash__` próprio — a ordem de iteração depende de
    endereços de memória — e `.compounds` vem de `list(set(...))` sobre strings, cuja ordem
    varia com `PYTHONHASHSEED`. Sem ordenar, duas execuções com a mesma entrada e a mesma
    semente devolveriam tuplas em ordem diferente, e o determinismo que o módulo promete
    valeria só para os p-valores.
    """
    relatorio = _roda(entrada)
    com_candidatos = [v for v in relatorio.vias if v.compostos_candidatos]
    assert com_candidatos, "âncora: nenhuma via com candidatos, e a asserção seria vácua"

    for via in com_candidatos:
        ids = via.compostos_candidatos
        assert list(ids) == sorted(ids), f"«{via.nome}» tem candidatos fora de ordem: {ids}"
        assert len(set(ids)) == len(ids), f"«{via.nome}» repete identificador: {ids}"


# ---------------------------------------------------------------------------
# O p de permutação com empate conservador
# ---------------------------------------------------------------------------


def _p_do_mummichog(x: float, nula: list[float]) -> float:
    """O oráculo: a função da própria biblioteca, chamada sem instância (ela não a usa)."""
    from mummichog.functional_analysis import PathwayAnalysis

    return float(PathwayAnalysis.__calculate_p__(None, x, list(nula)))


def test_p_de_permutacao_conta_o_empate_como_pelo_menos_tao_extremo() -> None:
    """O defeito, reproduzido à mão num caso cuja resposta se sabe.

    Nula toda igual a 1 e observado 1: nada na nula é MENOS extremo que o observado, e o
    p honesto é 1. A biblioteca dá ``1 / 101``, porque `list.index` acha a primeira
    ocorrência de um valor empatado e o observado fica com o menor posto.
    """
    nula = [1.0] * 100
    assert _p_do_mummichog(1.0, nula) == pytest.approx(1 / 101), (
        "a biblioteca deixou de dar o menor posto no empate: se isso for verdade, a "
        "correção do empate virou cerimônia e a docstring de `p_de_permutacao` mente"
    )
    assert p_de_permutacao([1.0], nula) == (1.0,)

    # E o empate a meio da nula: 50 valores iguais ao observado contam todos.
    meio = [0.01] * 50 + [1.0] * 50
    assert p_de_permutacao([0.01], meio) == ((50 + 1) / 101,)
    assert _p_do_mummichog(0.01, meio) == pytest.approx(1 / 101)


def test_p_de_permutacao_SEM_empate_e_identico_ao_do_mummichog() -> None:  # noqa: N802
    """Onde não há empate a correção não muda nada — e bit a bit, não por tolerância.

    A nula mistura `int` 1 e `float`, como a da biblioteca (`p_of_pathways.append(1)`),
    e os observados caem entre valores dela, nas duas pontas e fora dela.
    """
    rng = np.random.default_rng(512)
    nula = [float(x) for x in rng.uniform(0.0, 0.99, size=400)] + [1] * 30
    assert len(set(nula)) == 401, "a nula de teste tem empates internos por acaso"
    observados = [0.0005, 0.3333333, 0.5, 0.987654321, 0.995, 1.5]
    assert not set(observados) & set(nula), "um observado caiu em cima da nula"

    nossos = p_de_permutacao(observados, nula)
    deles = tuple(_p_do_mummichog(x, nula) for x in observados)
    assert nossos == deles


def test_p_de_permutacao_nas_pontas() -> None:
    nula = [0.1, 0.2, 0.3, 0.4]
    assert p_de_permutacao([0.9], nula) == (1.0,)
    assert p_de_permutacao([0.01], nula) == (1 / 5,)
    # A ordem da nula recebida não importa: ela é ordenada lá dentro.
    assert p_de_permutacao([0.25], [0.4, 0.1, 0.3, 0.2]) == (3 / 5,)
    # Sem permutação nenhuma o p é 1, como na biblioteca — não há evidência.
    assert p_de_permutacao([0.01], []) == (1.0,)
    assert _p_do_mummichog(0.01, []) == 1.0


def test_p_de_permutacao_recusa_nan() -> None:
    """NaN compara falso com tudo, e `bisect_right` devolveria um p plausível e falso."""
    with pytest.raises(ValueError, match="NaN"):
        p_de_permutacao([float("nan")], [0.5, 1.0])
    # E na nula, nas duas posições: sem a guarda, as duas davam 0,75 e 0,5 sem levantar.
    nan = float("nan")
    with pytest.raises(ValueError, match="NaN na nula"):
        p_de_permutacao([0.5], [nan, 0.1, 0.9])
    with pytest.raises(ValueError, match="NaN na nula"):
        p_de_permutacao([0.5], [0.1, 0.9, nan])


def test_sem_feature_significativa_NENHUMA_via_passa(entrada) -> None:  # noqa: N802
    """O PAR CAUSAL, sobre dado real: nada significativo, nada abaixo de 0,05.

    Todo p finito do `neg_hilic` vira 0,9 e o corte fica em 0,05, então a lista
    significativa é vazia e toda via tem sobreposição zero — p de Fisher 1 por
    atribuição, e toda permutação também 1. **Com o p da biblioteca, este teste reprova:**
    as vias saem todas com ``1 / (len(nula) + 1)``, bem abaixo de 0,05, e o rodapé da aba
    diria que todas são significativas.
    """
    nomes, p, t = entrada
    p09 = np.where(np.isfinite(p), 0.9, p)
    relatorio = prediz_vias(
        nomes, p09, t, modo_de_ionizacao="negative", corte_de_significancia=0.05, seed=0
    )

    # Não-vacuidade, ANTES e na mesma função: sem vias, ou com vias com
    # sobreposição, as asserções de baixo não diriam nada sobre o empate.
    assert relatorio.recorte.features_significativas == 0
    assert len(relatorio.vias) > 0, "nenhuma via: a asserção de ausência seria vácua"
    assert all(v.sobreposicao == 0 for v in relatorio.vias), (
        "há via com sobreposição numa entrada sem feature significativa"
    )

    assert not relatorio.significativas, (
        f"{len(relatorio.significativas)} via(s) abaixo de p = 0,05 sem NENHUMA feature "
        f"significativa; p da primeira = {relatorio.vias[0].pvalue:.3g}. É o empate a "
        "receber o menor posto"
    )
    assert all(v.pvalue == 1.0 for v in relatorio.vias)


def test_via_sem_sobreposicao_tem_p_1_num_contraste_real(entrada) -> None:
    """E o mesmo fato num contraste REAL, onde o p antigo não chegava a 0,05.

    Medido em 24/09/2026 no `neg_hilic`, ATLL x NoHTLV: as vias sem sobreposição saíam com
    p = 0,175 — otimista mas acima do corte, e por isso invisível ao rodapé. Aqui o
    desfecho é exato: toda nula é ≤ 1, então empatar com o 1 é empatar com o máximo.
    """
    relatorio = _roda(entrada)
    sem = [v for v in relatorio.vias if v.sobreposicao == 0]
    com = [v for v in relatorio.vias if v.sobreposicao > 0]
    assert sem, "âncora: nenhuma via sem sobreposição, e a asserção seria vácua"
    assert com, "âncora: nenhuma via com sobreposição, e o contraste não seria real"

    assert all(v.pvalue == 1.0 for v in sem), sorted({v.pvalue for v in sem})
    assert all(v.pvalue_fisher == 1.0 for v in sem)


def test_as_vias_vem_ordenadas_por_p_fisher_e_nome(entrada) -> None:
    """A saída sai ordenada por `_chave_de_ordem`, e o empate tem de existir para contar.

    Com o p novo, todas as vias sem sobreposição empatam em 1; sem desempate explícito a
    ordem entre elas seria a da biblioteca, e a tela mostraria uma ordem que nada nela
    explica (regra 3 do README).

    **O que este dado exercita é o desempate por NOME**, e não o por p de Fisher: medido na
    revisão, este contraste não tem grupo de p empatado com p de Fisher diferente, e
    uma chave `(pvalue, nome)` passaria aqui igual. O segundo componente é exercitado por
    `test_chave_de_ordem_desempata_pelo_p_de_fisher_antes_do_nome`.
    """
    relatorio = _roda(entrada)
    chaves = [_chave_de_ordem(v) for v in relatorio.vias]
    ps = [v.pvalue for v in relatorio.vias]
    assert len(set(ps)) < len(ps), "âncora: nenhum empate de p, e o desempate não é exercitado"
    assert chaves == sorted(chaves)


def _via(nome: str, pvalue: float, pvalue_fisher: float) -> ViaPredita:
    return ViaPredita(
        nome=nome,
        sobreposicao=1,
        detectados_na_via=5,
        pvalue=pvalue,
        pvalue_fisher=pvalue_fisher,
        compostos_candidatos=(),
        nomes_candidatos=(),
    )


def test_chave_de_ordem_desempata_pelo_p_de_fisher_antes_do_nome() -> None:
    """O segundo componente da chave, num caso em que ele e o nome discordam.

    Duas vias com o mesmo p de permutação — o que a convenção produz quando não há
    valor da nula entre os dois p de Fisher — e o nome em ordem OPOSTA à do p de Fisher. Uma
    chave `(pvalue, nome)` poria «A» primeiro, e este teste reprovaria.
    """
    a = _via("A", pvalue=0.08, pvalue_fisher=0.55)
    z = _via("Z", pvalue=0.08, pvalue_fisher=0.54)
    assert a.nome < z.nome and z.pvalue_fisher < a.pvalue_fisher, "âncora: os dois discordam"

    assert [v.nome for v in sorted([a, z], key=_chave_de_ordem)] == ["Z", "A"]
    # E o p de permutação continua a mandar sobre os dois.
    b = _via("B", pvalue=0.01, pvalue_fisher=0.99)
    assert [v.nome for v in sorted([a, z, b], key=_chave_de_ordem)] == ["B", "Z", "A"]


# ---------------------------------------------------------------------------
# A entrada pequena demais — item 88, medido em 25/09/2026
# ---------------------------------------------------------------------------


def _ions_analisaveis(entrada, quantas: int) -> tuple[tuple[str, ...], np.ndarray, np.ndarray]:
    """As primeiras `quantas` features REAIS que chegam à biblioteca: íon, p e t finitos.

    Íon e não massa neutra para a contagem não depender da política; e a massa cai na faixa
    em todos os íons do `neg_hilic` (o recorte do arquivo inteiro dá zero fora da faixa).
    """
    nomes, p, t = entrada
    indices = [
        i
        for i in range(len(nomes))
        if nomes[i].endswith("m/z") and np.isfinite(p[i]) and np.isfinite(t[i])
    ][:quantas]
    assert len(indices) == quantas, "âncora: o arquivo não tem íons analisáveis bastantes"
    return tuple(nomes[i] for i in indices), p[indices], t[indices]


def test_uma_feature_com_corte_automatico_e_RECUSADA_com_motivo(entrada) -> None:  # noqa: N802
    """Uma feature e o corte automático: recusa com causa, e não `IndexError` da biblioteca.

    **Visto reprovar sem a guarda**: o desfecho era `IndexError: list index out of range`,
    levantado em `mummichog/get_user_data.py:418`, que a janela mostrava como *«defeito a
    reportar»*. A classe-base é afirmada porque é ela que o painel apanha.
    """
    nomes, p, t = _ions_analisaveis(entrada, MINIMO_COM_CORTE_AUTOMATICO - 1)
    with pytest.raises(EntradaPequenaDemaisError, match=r"só 1 de 1 feature") as recusa:
        prediz_vias(nomes, p, t, modo_de_ionizacao="negative")
    assert isinstance(recusa.value, EntradaRecusadaError)
    # A distinção no tipo: sobrou uma feature, e o nome «sem feature analisável» mentiria.
    assert not isinstance(recusa.value, EntradaSemFeatureAnalisavelError)


def test_corte_ZERO_passado_por_quem_chama_tambem_e_automatico(entrada) -> None:  # noqa: N802
    """`0.0` é falso para a biblioteca, que então escolhe o corte sozinha — e indexa além do fim.

    A guarda espelha o predicado dela (`not cutoff`) em vez de olhar só para `None`; uma guarda
    que testasse `is None` deixaria este caso chegar ao `IndexError`.
    """
    nomes, p, t = _ions_analisaveis(entrada, 1)
    with pytest.raises(EntradaPequenaDemaisError):
        prediz_vias(nomes, p, t, modo_de_ionizacao="negative", corte_de_significancia=0.0)


def test_uma_feature_com_corte_FIXO_corre(entrada) -> None:  # noqa: N802
    """A outra direção: com o corte fixado uma feature basta, e a guarda NÃO a recusa.

    Medido em 25/09/2026: 40 de 40 sorteios de uma feature correram com corte 0,05 e 0,5, nos
    dois modos. Recusá-los seria restringir a ferramenta por uma falha que ali não existe.
    """
    nomes, p, t = _ions_analisaveis(entrada, 1)
    relatorio = prediz_vias(nomes, p, t, modo_de_ionizacao="negative", corte_de_significancia=0.05)
    assert relatorio.recorte.analisadas == 1
    assert relatorio.vias, "âncora: a corrida de uma feature não devolveu via nenhuma"


def test_o_MINIMO_com_corte_automatico_corre(entrada) -> None:  # noqa: N802
    """A fronteira: com `MINIMO_COM_CORTE_AUTOMATICO` features a análise corre inteira.

    Sem esta metade, uma guarda com o número chutado para cima — «pelo menos 3», «pelo menos
    10» — passaria no teste da recusa e recusaria entrada que a biblioteca aceita.
    """
    nomes, p, t = _ions_analisaveis(entrada, MINIMO_COM_CORTE_AUTOMATICO)
    relatorio = prediz_vias(nomes, p, t, modo_de_ionizacao="negative")
    assert relatorio.recorte.analisadas == MINIMO_COM_CORTE_AUTOMATICO
    assert relatorio.vias


def test_a_biblioteca_CRUA_ainda_levanta_abaixo_do_minimo(entrada) -> None:  # noqa: N802
    """Prova que o defeito que a guarda contorna EXISTE — a irmã de `test_a_contaminacao_...`.

    Chama `InputUserData` diretamente, com o corte automático e as linhas REAIS que
    `prediz_vias` montaria, e afirma o `IndexError` com uma linha e a ausência dele com
    `MINIMO_COM_CORTE_AUTOMATICO`. **Se uma versão futura do mummichog consertar o índice, a
    primeira metade reprova** e diz que a guarda virou restrição sem causa — o gatilho que
    uma nota em prosa não teria.
    """
    import contextlib
    import io

    from mummichog import config as mcg_config
    from mummichog.get_user_data import InputUserData

    from refino.pathways.picos import interpreta_todos

    nomes, p, t = _ions_analisaveis(entrada, MINIMO_COM_CORTE_AUTOMATICO)
    picos = interpreta_todos(nomes).ions
    assert len(picos) == MINIMO_COM_CORTE_AUTOMATICO, "âncora: os nomes não viraram íons"

    def paradict(quantas: int) -> dict[str, object]:
        linhas = [
            "\t".join(
                (str(pico.como_mz("negative")), str(pico.rt), str(p[i]), str(t[i]), pico.nome)
            )
            for i, pico in enumerate(picos[:quantas])
        ]
        cabecalho = "m/z\tretention_time\tp-value\tt-score\tcustom_id"
        return {
            "cutoff": 0.0,
            "datatext": "\n".join([cabecalho, *linhas]),
            "wanted_adduct_list": mcg_config.wanted_adduct_list["negative"],
        }

    with contextlib.redirect_stdout(io.StringIO()), pytest.raises(IndexError):
        InputUserData(paradict(MINIMO_COM_CORTE_AUTOMATICO - 1), web=True)

    with contextlib.redirect_stdout(io.StringIO()):
        dados = InputUserData(paradict(MINIMO_COM_CORTE_AUTOMATICO), web=True)
    assert len(dados.ListOfMassFeatures) == MINIMO_COM_CORTE_AUTOMATICO
