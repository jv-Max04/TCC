"""A interface, e o que fica FORA aparece com motivo.

## Como esta unidade é verificada

O que se verifica é a declaração:

| Verificado | Mecanismo |
|---|---|
| a declaração do fluxo tem os passos que o critério nomeia | asserção sobre `DENTRO_DO_FLUXO` |
| cada passo aponta para **rota que existe** | `rotas_da_aplicacao()`, do portão de rotas |
| a tela de qualidade exibe o conjunto **completo** | paridade com `QualityReport`, derivada |
"""

import ast
import inspect
import json
import re
import textwrap
from dataclasses import replace
from typing import Any, Final, cast, get_args

import numpy as np
import pytest

from refino.api.esquemas import ExecucaoResposta, RsdAntesEDepois
from refino.decision.palpite import LIMIAR_DE_EMPATE_DO_PALPITE, N_FEATURES_DO_PALPITE
from refino.domain.analysis import PADRAO_DA_ANALISE, AnalysisSpec
from refino.domain.labels import DefectType
from refino.domain.matrix import RefinoMatrix
from refino.domain.metadata import SampleMetadata
from refino.domain.spec import (
    REQUIRED_METRIC_FIELDS,
    SUPPORTED_OVERRIDES,
)
from refino.interface.fluxo import (
    CAMINHO_DA_METRICA,
    CAUDA_DA_EXECUCAO_S,
    CHAVE_SEM_TEMA,
    CHAVES_DE_ARTEFATO,
    COMO_O_PALPITE_DECIDE,
    CONTROLES_DE_SOBRESCRITA,
    CUSTO_DA_ANALISE_S,
    DENTRO_DO_FLUXO,
    DICA_DA_FAMILIA,
    FAMILIAS_DE_ANALISE,
    METRICAS_DA_TELA_DE_QUALIDADE,
    METRICAS_DE_VALIDACAO_DO_PLSDA,
    METRICAS_DO_PLSDA,
    METRICAS_SOB_PAR_DE_GRUPOS,
    NAO_CALCULADA,
    NENHUMA_ROTA_CALCULA,
    PONTOS_DE_DECISAO_AUTOMATICA,
    PREFIXO_DE_LEITURA,
    PROSA_DA_VARIANTE_DE_RSD,
    PROSA_DO_ESTIMADOR_DE_RSD,
    ROTULOS_DE_DEFEITO,
    SEM_CAMINHO,
    SEM_VALOR_DEFINIDO,
    SOB_PAR_DE_GRUPOS,
    TEMAS_DA_QUALIDADE,
    TRADUCAO_DA_METRICA,
    ControleDaTela,
    FormatoDoValor,
    MetricaTraduzida,
    PopulacaoDaMetrica,
    TemaDaMetrica,
    agrupa_metricas_da_qualidade,
    cauda_da_execucao_s,
    eixos_do_ponto,
    endereco_da_metrica,
    estado_de_ausencia,
    familia_e_barata,
    familias_pedidas,
    justificativa_em_prosa,
    legenda_da_metrica,
    prosa_da_recomendacao,
    prosa_da_variante_de_rsd,
    prosa_das_analises_pedidas,
    prosa_das_etapas_pedidas,
    prosa_do_estimador_de_rsd,
    prosa_dos_rotulos_do_rsd,
    rotulo_da_metrica,
    valor_da_metrica,
)
from refino.metrics.report import build_quality_report
from tests.conftest import REPO
from tests.test_api_contrato import rotas_da_aplicacao


def _matriz_para_qualidade(n_samples: int = 12, n_features: int = 6) -> RefinoMatrix:
    """Matriz mínima com QC suficiente para as métricas obrigatórias saírem.

    Sintética e não de referência **de propósito**: o que se afirma é quais campos a chamada da
    rota põe em `computed`, e isso é propriedade da CHAMADA e não dos dados do laboratório. Usar o
    arquivo real aqui custaria segundos e mediria a mesma coisa.
    """
    rng = np.random.default_rng(11)
    nivel = rng.uniform(1e4, 1e6, size=(1, n_features))
    deriva = np.linspace(1.0, 3.0, n_samples)[:, None]
    ruido = 1.0 + 0.05 * rng.standard_normal(size=(n_samples, n_features))
    amostras = tuple(
        SampleMetadata(
            name=f"S{i}",
            sample_type="qc" if i % 4 == 0 else "sample",
            injection_order=i + 1,
            batch="A",
            group="QC" if i % 4 == 0 else "G1",
        )
        for i in range(n_samples)
    )
    return RefinoMatrix(
        values=nivel * deriva * ruido,
        feature_names=tuple(f"1.0_{100 + j}m/z" for j in range(n_features)),
        samples=amostras,
        mode="pos_hilic",
    )


# ---------------------------------------------------------------------------
# O fluxo declarado, e cada passo com rota que EXISTE
# ---------------------------------------------------------------------------


def test_ca_12_1_o_fluxo_declara_os_passos_que_o_criterio_nomeia() -> None:
    """Os DEZ que o recorte do critério põe DENTRO — seis de análise, três de artefato, e a
    reclamação.

    Ingestão, validação, diagnóstico com pontos de mudança, pipeline com parâmetros explícitos,
    multiverso e o acompanhamento dele. A confirmação humana das fronteiras de lote não é passo
    próprio: ela é o **ponto de decisão** do diagnóstico, e está em `PONTOS_DE_DECISAO_AUTOMATICA`.

    A `reclamacao` é do mesmo gênero que `progresso`. As quatro operações longas devolvem `202`
    com identificador e o resultado é reclamado por `GET /trabalhos/{identificador}`; sem o passo
    declarado, a tela executaria uma rota que o fluxo não nomeia.
    """
    chaves = [passo.chave for passo in DENTRO_DO_FLUXO]

    assert chaves == [
        "validacao",
        "diagnostico",
        "recomendacao",
        "execucao",
        "multiverso",
        "progresso",
        # Os três de artefato. Sem eles a tela oferecia três operações que o fluxo declarado não
        # nomeava.
        "corrigir",
        "processar",
        "exportar",
        "reclamacao",
    ]


def test_ca_12_1_cada_passo_do_fluxo_aponta_para_uma_rota_que_existe() -> None:
    """**É esta cláusula que faz "executável" ser afirmação e não desenho.**

    Sem ela, o fluxo poderia declarar um passo servido por rota que ninguém escreveu, e a interface
    o mostraria na tela até alguém clicar. A verificação usa o mesmo inventário do portão de
    rotas, então rota renomeada reprova aqui e lá.
    """
    servidas = rotas_da_aplicacao()
    faltando = [
        f"{passo.metodo} {passo.rota} ({passo.chave})"
        for passo in DENTRO_DO_FLUXO
        if (passo.metodo, passo.rota) not in servidas
    ]

    assert faltando == [], f"passos declarados sem rota na aplicação: {faltando}"


# ---------------------------------------------------------------------------
# O que está FORA aparece com motivo, e o que está DENTRO com RESSALVA a declara
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Justificativa E controle, em cada ponto
# ---------------------------------------------------------------------------


def test_os_eixos_de_cada_ponto_saem_dos_controles_que_a_tela_expoe() -> None:
    """Esta asserção INVERTEU, e a inversão é o requisito.

    Ela dizia `set(ponto.eixos_sobrescreviveis) == SUPPORTED_OVERRIDES` e estava certa sobre a
    pergunta errada: a derivação era real — os nove vinham da constante e não de lista à mão —, e a
    constante mede o que a **especificação suporta**, não o que a **tela move**. Medido em
    17/08/2026: o `multiselect` oferecia nove eixos e a tela desenhava controle para **três**, então
    marcar um dos outros seis gravava par de concordância sobre eixo que o sistema ignora.

    **A derivação continua a ser derivação, e o que muda é a FONTE**: `CONTROLES_DE_SOBRESCRITA`,
    uma linha por controle que a tela desenha. A relação com `SUPPORTED_OVERRIDES` fica como
    **contenção** e não como igualdade, que é o que o `_eixos_ou_422` da rota exige.
    """
    eixos_por_ponto = {p.chave: set(p.eixos_sobrescreviveis) for p in PONTOS_DE_DECISAO_AUTOMATICA}

    # A LISTA CRESCEU, e o crescimento e o requisito satisfeito.
    # Eram dois no ponto da recomendacao — `correction` e `rsd_cutoff` — porque a tela desenhava
    # controle so para esses. Depois ele ganhou os quatro eixos da grade, o corte de ausentes e
    # a variante do RSD, entao os eixos oferecidos crescem com os controles, que e exactamente o
    # que esta derivacao existe para garantir.
    assert eixos_por_ponto == {
        "diagnostico": {"n_batches"},
        "recomendacao": {
            "correction",
            "rsd_cutoff",
            # O décimo eixo, que entrou com a paridade. A tela desenha
            # o combo e o `multiselect` da recomendação o oferece: as duas metades entraram no
            # mesmo commit, que é o que a direção A e a direção B deste teste exigem.
            "ranking_filter",
            "imputation",
            "sample_normalization",
            "transformation",
            "scaling",
            "missing_threshold",
            "rsd_variant",
            # Os TRES do filtro em dois eixos chegam pela mesma porta que o
            # decimo: a tela ganhou o combo do segundo passe e os dois campos de fracao retida no
            # mesmo commit em que o `multiselect` passou a oferece-los.
            "ranking_keep_fraction",
            "ranking_filter_2",
            "ranking_keep_fraction_2",
            # A referencia da normalizacao, pela mesma porta: a tela ganhou o combo
            # editavel no mesmo commit em que `SUPPORTED_OVERRIDES` a passou a declarar.
            "normalization_reference",
        },
    }, f"os eixos oferecidos deixaram de casar os controles desenhados: {eixos_por_ponto}"

    todos = {eixo for eixos in eixos_por_ponto.values() for eixo in eixos}
    # A relacao passou de subconjunto ESTRITO para CONTENCAO, e a mudanca e por SUCESSO.
    #
    # A estritez documentava um estado, nao um requisito: a tela movia tres dos nove, e o `<`
    # afirmava essa distancia. Com a paridade fechada ela move os nove, e manter o `<` faria o
    # portao reprovar *porque o defeito foi consertado* — que e a forma mais cara de um teste
    # errado, porque ele passa a defender a lacuna.
    #
    # **O que continua a ser requisito e a contencao**: eixo que a tela mova e a API nao aceite e
    # recusado com 422 na fronteira, e isso vale com nove ou com tres.
    assert todos <= SUPPORTED_OVERRIDES, (
        "a tela move eixo que a API nao aceita: ele e recusado com 422 na fronteira. Sobrando: "
        f"{sorted(todos - SUPPORTED_OVERRIDES)}"
    )
    assert len(SUPPORTED_OVERRIDES) == 14 and len(todos) == 14, (
        f"a especificacao suporta {len(SUPPORTED_OVERRIDES)} eixos e a tela move {len(todos)}: "
        "medido em 07/09/2026 a paridade esta fechada, e uma queda aqui e regressao do "
        "R14. Eram NOVE ate a paridade com o MetaboAnalyst, que acrescentou "
        "`ranking_filter` aos dois lados no mesmo commit, e DEZ ate 10/09/2026, quando o "
        "filtro em dois eixos acrescentou os tres do segundo passe, e CATORZE com a "
        "referencia da normalizacao. Os fatores fornecidos NAO entram, e a razao e "
        "que a maquina nunca os propoe — "
        "todos aos dois lados no mesmo commit, que e o que mantem os dois numeros iguais"
    )


def test_cada_controle_declarado_pertence_a_um_ponto_que_existe() -> None:
    """Controle cujo `ponto` não é de ponto nenhum seria eixo declarado e nunca oferecido.

    A direção que ele fecha é silenciosa: `eixos_do_ponto` filtra por igualdade de chave, então um
    `ponto="recomendacoes"` com um `s` a mais **não levanta** — o controle simplesmente desaparece
    do `multiselect`, e o eixo deixa de ser oferecido sem nada reprovar.
    """
    chaves = {p.chave for p in PONTOS_DE_DECISAO_AUTOMATICA}
    orfaos = [c.chave_do_widget for c in CONTROLES_DE_SOBRESCRITA if c.ponto not in chaves]

    assert CONTROLES_DE_SOBRESCRITA, "âncora: há controle declarado"
    assert orfaos == [], f"controles cujo ponto de decisão não existe: {orfaos}"


def test_eixos_do_ponto_acresce_o_eixo_quando_um_controle_novo_e_exposto() -> None:
    """A planta: expor um controle novo acrescenta o eixo SOZINHO.

    Negativo EM PROCESSO — a planta é argumento da função sob teste —, e é por isso que ele não
    consegue ficar inerte. O par causal é sobre a **tabela**: as duas chamadas diferem só pelo
    controle acrescentado, e o eixo aparece sem que `PONTOS_DE_DECISAO_AUTOMATICA` seja tocado.
    """
    # O par causal parte de uma tabela REDUZIDA.
    #
    # Ele plantava `scaling` — um eixo real que a tela ainda nao movia. Depois a paridade fechou-se
    # e a tela passou a mover os NOVE, entao a ancora *"o eixo plantado nao estava la
    # antes"* deixou de valer: o teste reprovava **porque o defeito foi consertado**.
    #
    # **Um nome fabricado nao serve**, e isso foi medido: `eixos_do_ponto` recusa eixo fora de
    # `SUPPORTED_OVERRIDES`, entao a planta levanta `ValueError`
    # antes de medir coisa nenhuma.
    #
    # A forma que sobrevive as duas: TIRAR um controle real da tabela e po-lo de volta. A funcao
    # recebe `controles` por argumento justamente para isto, e a propriedade medida — o eixo
    # aparece SOZINHO, sem `PONTOS_DE_DECISAO_AUTOMATICA` ser tocado — nao depende de existir um
    # eixo por declarar no mundo.
    novo = next(c for c in CONTROLES_DE_SOBRESCRITA if c.ponto == "recomendacao")
    reduzida = tuple(c for c in CONTROLES_DE_SOBRESCRITA if c is not novo)
    antes = eixos_do_ponto("recomendacao", reduzida)
    depois = eixos_do_ponto("recomendacao", (*reduzida, novo))

    assert novo.eixo not in antes, (
        f"âncora: o eixo `{novo.eixo}` já estava na tabela reduzida, e a planta não separa nada"
    )
    assert set(depois) - set(antes) == {novo.eixo}, (
        f"expor um controle novo não acrescentou o eixo: {antes} -> {depois}"
    )


def test_eixos_do_ponto_recusa_ponto_sem_controle_nenhum() -> None:
    """O ramo que dispara por AUSÊNCIA, e é o que fica inerte sem ninguém notar.

    Um ponto sem controle tem justificativa exibida e nada com que agir, e era exactamente o estado
    em que o ponto do diagnóstico ficaria se a derivação entrasse sem o campo numérico. Levantar
    aqui torna a violação impossível por construção.
    """
    with pytest.raises(ValueError, match=r"não tem controle nenhum"):
        eixos_do_ponto("ponto_sem_controle_nenhum", CONTROLES_DE_SOBRESCRITA)


def test_eixos_do_ponto_recusa_eixo_que_a_api_nao_aceita() -> None:
    """Vocabulário fechado: eixo fora de `SUPPORTED_OVERRIDES` levanta em vez de ser oferecido.

    Sem esta recusa o eixo chegaria ao `multiselect`, o usuário marcá-lo-ia, e `_eixos_ou_422`
    devolveria `422` — que a tela relata como *"causa: usuario"*.
    """
    intruso = ControleDaTela(
        chave_do_widget="campo_inventado",
        eixo="eixo_que_a_api_nao_aceita",
        ponto="p",
        onde="decisao",
    )

    with pytest.raises(ValueError, match=r"SUPPORTED_OVERRIDES"):
        eixos_do_ponto("p", (intruso,))


def test_cada_ponto_de_decisao_corresponde_a_um_passo_do_fluxo() -> None:
    """Ponto que citasse passo inexistente seria decisão sobre nada.

    A enumeração dos pontos sai do inventário de rotas — via os passos —, e não de lista paralela.
    """
    chaves = {passo.chave for passo in DENTRO_DO_FLUXO}
    orfaos = [p.chave for p in PONTOS_DE_DECISAO_AUTOMATICA if p.chave not in chaves]

    assert orfaos == [], f"pontos de decisão sem passo correspondente: {orfaos}"


# ---------------------------------------------------------------------------
# Nenhuma métrica sozinha, e PLS-DA nunca sem validação, sobre a APRESENTAÇÃO
# ---------------------------------------------------------------------------


def test_d263_o_caminho_das_metricas_bate_com_o_que_a_rota_produz() -> None:
    """Comportamental: mede o referente em vez de ler a declaração.

    A tabela `CAMINHO_DA_METRICA` é derivada de `REQUIRED_METRIC_FIELDS`, e essa derivação só é
    verdadeira **porque a rota chama `build_quality_report` sem os dois vetores opcionais**. Um
    portão que comparasse a tabela com a mesma constante que a produz seria tautológico; este
    chama a função como `api/rotas.py:436-438` a chama e lê o que saiu.

    Se alguém passar `qc_rsd_cv` na rota, aquelas métricas ganham caminho e esta asserção reprova
    até a tabela mudar junto — que é o par que impede as duas de derivarem.

    > A rota passou a chamar `plsda` quando o par de grupos é declarado, então o conjunto com
    > caminho deixou de ser uma chamada só: é a UNIÃO de duas, a sem par e a com par. Comparar
    > contra uma delas diria que a tabela diverge quando ela está certa.
    """
    matriz = _matriz_para_qualidade()
    sem_par = build_quality_report(
        matriz,
        measurement_point="post_pipeline",
        structural_pretransform="none",
    )
    # A segunda chamada, como a rota a faz com `group_a`/`group_b` declarados. O `PLSDAResult` é
    # construído à mão em vez de calculado: o que se mede aqui é a TABELA DE CAMINHOS, e correr um
    # PLS-DA real acrescentaria minutos e uma superfície estatística que não é a deste portão.
    com_par = build_quality_report(
        matriz,
        measurement_point="post_pipeline",
        structural_pretransform="none",
        plsda=_plsda_de_mentira(),
        plsda_groups=("A", "B"),
        plsda_cv_folds=7,
        # A contagem de FDR entra aqui pela MESMA razão que o `plsda` acima: a rota a
        # passa quando o par de grupos é declarado. O valor é **zero** e não um número qualquer,
        # porque zero é o que o corpus real dá — e um portão que usasse um valor inventado não
        # exercitaria o caso que a produção vai ver.
        n_significant_fdr=0,
    )
    produzidas = (set(sem_par.computed) | set(com_par.computed)) & METRICAS_DA_TELA_DE_QUALIDADE
    com_caminho = {n for n, c in CAMINHO_DA_METRICA.items() if c != "nenhum"}

    assert produzidas == com_caminho, (
        "a tabela de caminhos das métricas divergiu do que as chamadas da rota produzem"
    )
    assert com_caminho, "nenhuma métrica com caminho: a asserção acima passaria por vacuidade"
    # **A âncora que impede a união de esconder o par:** sem esta, um `plsda=` ignorado por
    # `build_quality_report` faria as duas chamadas darem o mesmo `computed`, e a igualdade acima
    # passaria com a tabela a afirmar um caminho que não existe.
    assert set(com_par.computed) - set(sem_par.computed) == {
        "plsda_q2",
        "plsda_permutation_p",
        # A terceira é o que o par de grupos passou a comprar além do
        # PLS-DA: a rota conta features significativas com `univariate_test` sobre a matriz
        # processada quando `group_a`/`group_b` vêm.
        "n_significant_fdr",
    }, (
        "declarar o par de grupos não acrescentou as TRÊS métricas sob argumento a `computed` — "
        "as duas do PLS-DA e a contagem de FDR"
    )


def _plsda_de_mentira() -> Any:
    """Um `PLSDAResult` com os dez campos, sem correr o algoritmo.

    **Construído e não calculado**, porque o que este arquivo mede é a **tabela de caminhos**. Um
    PLS-DA real sobre matriz de teste acrescentaria segundos e uma superfície estatística que não
    pertence a este portão — e `PLSDAResult` é congelado com dez campos obrigatórios, então
    construí-lo à mão não pode produzir um estado que o algoritmo não produziria.
    """
    from refino.statistics.plsda import PLSDAResult

    return PLSDAResult(
        scores=np.zeros((2, 2)),
        cv_scores=np.zeros((2, 2)),
        loadings=np.zeros((2, 2)),
        vip=np.zeros(2),
        r2y=0.5,
        q2=0.24,
        permutation_pvalue=0.3,
        n_components=2,
        n_permutations=200,
        seed=42,
    )


# ---------------------------------------------------------------------------
def _ancora_da_traducao() -> None:
    """Não-vacuidade de `TRADUCAO_DA_METRICA`, DERIVADA dos dois conjuntos que a compõem.

    Era o literal `11` em três sítios, e ele conflacionava duas populações. A tabela passou a ter o
    par do PLS-DA multiclasse, que não é das onze da tela — a própria docstring dele o diz. Com o
    literal, os três laços passaram a reprovar por uma razão que não é a deles: a tabela não
    encolheu, cresceu por outra origem.

    **Trocar `11` por `13` seria adiar a mesma conflação para a família seguinte.** A tabela traduz
    o que a tela escreve, e a tela escreve de duas origens: a grelha das onze escalares e a seção do
    PLS-DA. Derivar da união das duas apanha o encolhimento — que era o propósito — **e** apanha um
    nome que não pertença a nenhuma delas, que o literal nunca apanhou.
    """
    esperado = set(METRICAS_DA_TELA_DE_QUALIDADE) | set(METRICAS_DE_VALIDACAO_DO_PLSDA)
    assert esperado, "âncora da âncora: os dois conjuntos estão vazios e esta guarda seria vácua"
    assert set(TRADUCAO_DA_METRICA) == esperado, (
        "a população de `TRADUCAO_DA_METRICA` divergiu dos dois conjuntos que a definem: "
        f"{sorted(set(TRADUCAO_DA_METRICA).symmetric_difference(esperado))}"
    )


# ---------------------------------------------------------------------------
# As onze métricas em português, com unidade e com sentido
# ---------------------------------------------------------------------------


def test_u8_a_traducao_das_metricas_e_bijetiva_com_o_conjunto_da_tela() -> None:
    """Domínio fechado, nas DUAS direções — e é o portão que impede o nome cru de voltar.

    **Métrica nova sem tradução é o defeito seguinte, e ele é silencioso:**
    `METRICAS_DA_TELA_DE_QUALIDADE` é derivado de `QualityReport`, então um campo novo no relatório
    entra na tela **sozinho** — e apareceria com o nome de campo em inglês, que é exactamente o
    estado que esta unidade veio desfazer. Tradução órfã reprova pela outra direção, que é o que
    impede a tabela de envelhecer a apontar para uma métrica que saiu.

    **Os três campos obrigatórios são afirmados um a um**, e não só a chave: uma entrada com
    `unidade=""` satisfaria a bijeção e devolveria uma legenda que não diz a escala — e a escala é
    o que já custou a classificação de um modo inteiro.
    """
    traduzidas = set(TRADUCAO_DA_METRICA)

    assert METRICAS_DA_TELA_DE_QUALIDADE, "âncora: o conjunto de métricas da tela está vazio"
    assert len(METRICAS_DA_TELA_DE_QUALIDADE) == 11, (
        f"a população mudou: {len(METRICAS_DA_TELA_DE_QUALIDADE)} métricas. Com uma só, as duas "
        "asserções abaixo passariam por vacuidade"
    )
    # O domínio da bijeção são DOIS conjuntos, e o segundo não é das onze.
    # A tela escreve nome de duas origens: a grelha das onze métricas escalares, e a seção do
    # PLS-DA, que é dirigida por `PARES_DE_VALIDACAO_OBRIGATORIA`. A família multiclasse vive só na
    # segunda — é coleção e não escalar —, e sem a incluir aqui ela apareceria com o nome de campo
    # em inglês, marcado, que é exactamente o estado que esta unidade desfez. As onze continuam a
    # ser onze, e é isso que a asserção acima guarda.
    escritas_pela_tela = METRICAS_DA_TELA_DE_QUALIDADE | METRICAS_DE_VALIDACAO_DO_PLSDA
    assert len(escritas_pela_tela) == 13, (
        f"a população mudou: {sorted(escritas_pela_tela)}. São as onze da grelha mais as duas do "
        "PLS-DA multiclasse, que vivem só na seção dedicada ao PLS-DA"
    )
    assert escritas_pela_tela - traduzidas == set(), (
        f"métricas sem tradução: {sorted(escritas_pela_tela - traduzidas)} — a tela "
        "mostra-as com o nome de campo em inglês"
    )
    assert traduzidas - escritas_pela_tela == set(), (
        f"traduções órfãs: {sorted(traduzidas - escritas_pela_tela)}"
    )

    for nome in sorted(escritas_pela_tela):
        traducao = TRADUCAO_DA_METRICA[nome]
        assert traducao.rotulo.strip(), f"`{nome}` tem rótulo vazio"
        assert traducao.unidade.strip(), f"`{nome}` não declara unidade"
        assert traducao.sentido.strip(), f"`{nome}` não declara sentido de leitura"
        assert nome not in traducao.rotulo, (
            f"o rótulo de `{nome}` repete o nome de campo: {traducao.rotulo!r}. O nome técnico é "
            "desenhado pela interface, ao lado do valor, e não colado dentro do rótulo"
        )

    rotulos = {TRADUCAO_DA_METRICA[n].rotulo for n in escritas_pela_tela}
    assert len(rotulos) == len(escritas_pela_tela), (
        "duas métricas partilham o mesmo rótulo em português: quem lê a tela não tem como as "
        f"distinguir. Rótulos: {sorted(rotulos)}"
    )


def test_u8_o_portao_da_bijecao_acusa_traducao_faltando_e_traducao_orfa() -> None:
    """O negativo, em processo: a planta é a tabela passada por argumento.

    As duas direções, porque elas falham por causas diferentes: **faltar** é a métrica nova que
    ninguém traduziu, e **sobrar** é a tradução que ficou depois de a métrica sair. Um portão que
    só olhasse uma delas passaria com a outra.

    **E há a terceira planta, que a bijeção NÃO apanha:** uma entrada com todos os campos vazios
    tem a chave certa. É por isso que a asserção de conteúdo vive no teste acima, e esta função
    mede que a planta de conteúdo de facto passaria pela bijeção — sem isso, alguém concluiria que
    a bijeção cobre o que ela não cobre.
    """

    escritas_pela_tela = METRICAS_DA_TELA_DE_QUALIDADE | METRICAS_DE_VALIDACAO_DO_PLSDA

    def faltas(tabela: dict[str, MetricaTraduzida]) -> tuple[list[str], list[str]]:
        return (
            sorted(escritas_pela_tela - set(tabela)),
            sorted(set(tabela) - escritas_pela_tela),
        )

    assert faltas(dict(TRADUCAO_DA_METRICA)) == ([], []), (
        "a árvore limpa já acusa: as plantas abaixo não distinguiriam nada"
    )

    sem_uma = dict(TRADUCAO_DA_METRICA)
    removida = sorted(sem_uma)[0]
    del sem_uma[removida]
    assert len(sem_uma) < len(TRADUCAO_DA_METRICA), "a planta de remoção NÃO aterrou"
    assert faltas(sem_uma) == ([removida], []), (
        f"o portão não acusou a métrica `{removida}` sem tradução"
    )

    com_orfa = dict(TRADUCAO_DA_METRICA)
    com_orfa["metrica_que_saiu_do_relatorio"] = MetricaTraduzida(
        rotulo="Uma que já não existe",
        unidade="fração",
        # Os dois campos são OBRIGATÓRIOS, e é isso que os torna guarda: sem padrão, a entrada
        # nova não se constrói sem declarar em que unidade o número se escreve e sobre que
        # população foi medido. Um padrão escolheria a política por omissão para a métrica
        # seguinte.
        formato="razao",
        populacao="nenhuma",
        # E o TEMA pelo mesmo argumento: sem padrão, e sem valor «outras».
        tema="estrutura_no_pca",
        sentido="menor é melhor",
    )
    assert len(com_orfa) > len(TRADUCAO_DA_METRICA), "a planta de órfã NÃO aterrou"
    assert faltas(com_orfa) == ([], ["metrica_que_saiu_do_relatorio"]), (
        "o portão não acusou a tradução órfã"
    )

    # A terceira: campos vazios com a chave certa. A bijeção **não** a vê, e dizê-lo é o que
    # impede a afirmação de cobertura mais larga que o mecanismo.
    vazia = dict(TRADUCAO_DA_METRICA)
    vazia[removida] = MetricaTraduzida(
        rotulo="",
        unidade="",
        formato="razao",
        populacao="nenhuma",
        tema="estrutura_no_pca",
        sentido="",
    )
    assert faltas(vazia) == ([], []), (
        "a bijeção passou a ver conteúdo vazio; se isso for verdade, a asserção de campos do teste "
        "acima é redundante e este comentário está errado"
    )


def test_u8_toda_metrica_sem_caminho_declara_por_que() -> None:
    """As TRÊS que nenhuma rota calcula dizem o motivo, e as outras oito não têm o que dizer.

    **A partição vem de `CAMINHO_DA_METRICA` e é medida, não escrita:** hoje são
    `n_significant_fdr`, `qc_rsd_cv_median` e `qc_rsd_r_cv_median` — o par do PLS-DA saiu desta
    classe quando `POST /execucoes` passou a calculá-lo.

    A âncora do tamanho vem antes: sobre conjunto vazio, *"toda métrica sem caminho
    declara por quê"* passa sem olhar nenhuma.
    """
    sem_caminho = {n for n, c in CAMINHO_DA_METRICA.items() if c == "nenhum"}
    assert sem_caminho == {"qc_rsd_cv_median", "qc_rsd_r_cv_median"}, (
        f"a medição de 01/09/2026 diz que as sem caminho são as DUAS de RSD fora da amostra, e "
        f"vieram {sorted(sem_caminho)}. **O conjunto e não a contagem**: o número sozinho passaria "
        "se uma saísse e outra entrasse"
    )
    for nome in sorted(sem_caminho):
        assert TRADUCAO_DA_METRICA[nome].por_que_sem_caminho.strip(), (
            f"`{nome}` não tem caminho e a tradução não diz por quê: quem lê fica com «não "
            "disponível» e nenhuma ação"
        )


def test_u8_a_legenda_emite_a_razao_de_ausencia_nos_dois_desfechos() -> None:
    """O par causal da frase de ausência, sobre a MESMA métrica.

    Comparar uma métrica sem caminho com outra que tem caminho mediria a diferença entre as duas
    métricas, e não o ramo. A injeção de `caminhos` é o que permite ver os dois desfechos da mesma:
    com `nenhum`, a frase sai; com `api`, ela desaparece **e o resto da legenda fica**.

    **É a segunda direção que não acontece hoje em produção**, e é por isso que ela precisa de
    injeção: enquanto nenhuma rota calcular estas duas, o ramo *"tem caminho"* delas é inalcançável
    — e ramo inalcançável não tem par causal (o padrão de `texto_da_metrica` e de `eixos_do_ponto`).

    > A métrica é DERIVADA, e a mudança é o achado. Ela estava fixa em `n_significant_fdr`, que
    > ganhou caminho depois — e o teste reprovou pela âncora, não pela propriedade que mede. Um
    > nome fixo codifica a instância em que a propriedade foi vista, e não a propriedade: trocá-lo
    > por outro nome fixo compraria o mesmo defeito para daqui a um mês. Derivar da tabela faz o
    > teste seguir a partição sozinho, e
    > a âncora abaixo passa a acusar o caso em que **nenhuma** métrica sem caminho resta — que é
    > quando este par causal deixa de ter sujeito.
    """
    sem_caminho = sorted(n for n, c in CAMINHO_DA_METRICA.items() if c == "nenhum")
    assert sem_caminho, (
        "âncora: nenhuma métrica sem caminho — o par causal desta função ficou sem sujeito, e ela "
        "deve ser removida em vez de afrouxada"
    )
    nome = sem_caminho[0]

    sem = legenda_da_metrica(nome)
    com = legenda_da_metrica(nome, {**CAMINHO_DA_METRICA, nome: "api"})

    assert NENHUMA_ROTA_CALCULA in sem, f"a legenda não diz que nada a calcula: {sem!r}"
    assert TRADUCAO_DA_METRICA[nome].por_que_sem_caminho in sem, "a legenda não diz o motivo"
    assert NENHUMA_ROTA_CALCULA not in com, (
        f"a legenda continua a negar um caminho que a tabela declara: {com!r} — é o defeito que "
        "teve de ser invertido à mão em três superfícies"
    )
    assert com != sem, "âncora: a injeção não mudou nada, e o par causal é vácuo"
    assert TRADUCAO_DA_METRICA[nome].unidade in com, (
        "com caminho a legenda perdeu a unidade: o ramo apagou mais do que a frase de ausência"
    )


def test_u8_a_legenda_traz_unidade_e_sentido_de_todas_as_onze() -> None:
    """A unidade viaja com o número, e o custo de a omitir já foi medido.

    O RSD deste projeto é **fração** — `metrics/rsd.py::rsd` diz *"Devolve **fração** (0,25 = 25%),
    não percentual"* —, e ler os mesmos 0,155 como pontos percentuais trocava `clean` por
    `step_change` no `neg_c18`. Quem lê a tela não tem o `grep` que responde isso.

    **O nome técnico continua na legenda**, porque é por ele que o profissional fala connosco e
    acha a métrica no documento.
    """
    for nome in sorted(METRICAS_DA_TELA_DE_QUALIDADE):
        legenda = legenda_da_metrica(nome)
        traducao = TRADUCAO_DA_METRICA[nome]
        assert nome in legenda, f"a legenda de `{nome}` perdeu o nome técnico: {legenda!r}"
        assert traducao.unidade in legenda, f"a legenda de `{nome}` não diz a unidade"
        assert traducao.sentido in legenda, f"a legenda de `{nome}` não diz o sentido"


def test_u8_metrica_desconhecida_aparece_marcada_e_nao_crua() -> None:
    """O ramo `else`, que dispara por AUSÊNCIA de condição — não há o que plantar nele.

    A saída natural seria devolver o nome cru, que é o `else` razoável: uma
    métrica nova voltaria a aparecer em inglês e **nada** a denunciaria, porque a tela teria
    renderizado *"algo"*. A marca é o que faz a omissão aparecer na própria tela; o portão de
    bijeção é o que a pega antes.
    """
    rotulo = rotulo_da_metrica("uma_metrica_que_nao_existe")
    legenda = legenda_da_metrica("uma_metrica_que_nao_existe")

    assert "uma_metrica_que_nao_existe" in rotulo, "o rótulo não nomeia a métrica sem tradução"
    assert "sem tradução" in rotulo, f"o rótulo devolveu o nome cru: {rotulo!r}"
    assert "sem tradução" in legenda, f"a legenda devolveu o nome cru: {legenda!r}"
    assert rotulo != rotulo_da_metrica("qc_rsd_median"), "âncora: os dois ramos deram o mesmo"


# ---------------------------------------------------------------------------
# O painel de antes e depois contra o esquema que o alimenta
# ---------------------------------------------------------------------------


def chaves_lidas_da_funcao(fonte: str, nome: str) -> set[str]:
    """As chaves literais que uma função lê de um corpo, por análise sintática.

    AST e não `grep`: o nome de um campo aparece em docstring, e um proxy textual contaria menção
    como leitura — que é o defeito que este portão existe para achar, cometido dentro dele.

    O recorte é o **corpo de UMA função** e não o arquivo: outra função do mesmo arquivo lê outro
    esquema, e uni-los faria a bijeção comparar dois contratos como se fossem um.

    Recebe a FONTE em vez de ler o arquivo. A primeira versão lia o arquivo por dentro, e o
    negativo tinha de reimplementar a extração para plantar a chave errada — *"negativo que
    reimplementa a regra fica verde enquanto a regra apodrece"*. Com a fonte como argumento, o
    negativo percorre **este** código, e negativo em processo não consegue ficar inerte.

    `nome` é parâmetro, e antes a função era uma por alvo. O segundo alvo — o que
    exibe os identificadores gravados — precisa exactamente da mesma extração, e uma cópia dela
    apodreceria em silêncio no dia em que a primeira aprendesse uma forma sintática nova.
    """
    alvo = [
        no
        for no in ast.walk(ast.parse(fonte))
        if isinstance(no, ast.FunctionDef) and no.name == nome
    ]
    assert len(alvo) == 1, f"esperava uma definição de `{nome}` e achei {len(alvo)}"

    lidas: set[str] = set()
    for no in ast.walk(alvo[0]):
        if not (isinstance(no, ast.Call) and isinstance(no.func, ast.Attribute)):
            continue
        if no.func.attr != "get" or not no.args:
            continue
        chave = no.args[0]
        if isinstance(chave, ast.Constant) and isinstance(chave.value, str):
            lidas.add(chave.value)
    return lidas


def chaves_lidas_do_painel(fonte: str) -> set[str]:
    """As chaves de `_painel_de_antes_e_depois`. Fina sobre `chaves_lidas_da_funcao` de propósito.

    O negativo desta função continua a percorrer **a** extração e não uma cópia dela: se ela
    ficasse com o corpo próprio, o outro alvo mediria outra coisa.
    """
    return chaves_lidas_da_funcao(fonte, "_painel_de_antes_e_depois")


def test_o_portao_do_painel_acusa_nas_duas_direcoes() -> None:
    """O negativo, **em processo**: a planta é a fonte que `chaves_lidas_do_painel` lê.

    Invoca a função do portão e não uma cópia dela: negativo que reimplementa a regra fica verde
    enquanto a regra apodrece.

    **As duas direções da bijeção, e nenhuma se manifesta como erro sem este teste:** chave lida que
    o esquema não declara, e campo do esquema que o painel não mostra. Mais a âncora — a fonte
    correta **passa** —, sem a qual um extractor que devolvesse vazio satisfaria as duas negações.
    """
    declarados = set(RsdAntesEDepois.model_fields)
    corpo = "\n".join(f'    x = comparacao.get("{campo}")' for campo in sorted(declarados))
    boa = f"def _painel_de_antes_e_depois(st, comparacao):\n{corpo}\n"

    assert chaves_lidas_do_painel(boa) == declarados, "âncora: a fonte completa satisfaz a bijeção"

    errada = boa.replace('get("mediana_antes")', 'get("mediana_ante")')
    assert chaves_lidas_do_painel(errada) != declarados, "chave inventada passaria pelo portão"

    faltando = "\n".join(linha for linha in boa.splitlines() if 'get("n_features")' not in linha)
    assert chaves_lidas_do_painel(faltando + "\n") != declarados, (
        "campo declarado que o painel NÃO mostra passaria pelo portão: é a direção que o desenho "
        "óbvio esquece"
    )
    # Menção em prosa NÃO é leitura, e é o defeito que o proxy textual cometeria.
    prosa = (
        "def _painel_de_antes_e_depois(st, comparacao):\n"
        '    """Lê comparacao.get("mediana_antes")."""\n'
        "    return None\n"
    )
    assert chaves_lidas_do_painel(prosa) == set(), "docstring foi contada como leitura"


# ---------------------------------------------------------------------------
# O identificador exibido vem de campo que a resposta TEM
# ---------------------------------------------------------------------------


# **As três chaves que a tela exibe da execução gravada, e a lista é snapshot deliberado.** A
# contenção sozinha — *"toda chave lida é declarada"* — é satisfeita por **conjunto vazio**, e
# conjunto vazio é o que um extractor quebrado devolve. Este é o dente de presença.
#
# Não é bijeção, e a diferença é de propósito: `ExecucaoResposta` declara onze campos, e a linha
# *"execução gravada"* mostra o par de ids mais o estado de concordância do par.
CHAVES_DO_RESULTADO_EXIBIDO = frozenset({"matrix_id", "run_id", "estado_concordancia"})


# As chaves que `_exibe_conclusao_do_multiverso` le.
#
# Bijecao e nao contencao: campo que a funcao le e o esquema nao declara reprova, e campo do
# esquema que ela ignora tambem — que e o que impede a lista de crescer sem mecanismo.
CHAVES_DA_CONCLUSAO_EXIBIDA = frozenset(
    {"estabilidade", "grupos", "n_celulas", "n_features", "limiares"}
)


# A DELEGAÇÃO EXIBIDA. `_mostra_delegacao` recebe o dicionário por PARÂMETRO, e o que a confere
# é este par: a bijeção contra o esquema, e o negativo que prova que ela consegue acusar.
CHAVES_DA_DELEGACAO_EXIBIDA = frozenset(
    {"abstains", "assessable", "confidence", "correction", "politica", "rationale"}
)


def test_o_portao_da_delegacao_acusa_a_chave_que_o_esquema_nao_declara() -> None:
    """O negativo, **em processo**: a planta é a fonte que a extração lê.

    Par causal, e as duas leituras diferem só pela chave trocada. Sem ele, uma extração que
    devolvesse vazio satisfaria a contenção do teste acima.
    """
    from refino.api.esquemas import DelegacaoResposta

    declarados = set(DelegacaoResposta.model_fields)
    corpo = "\n".join(
        f'    x = delegacao.get("{chave}")' for chave in sorted(CHAVES_DA_DELEGACAO_EXIBIDA)
    )
    boa = f"def _mostra_delegacao(st, delegacao):\n{corpo}\n"
    nome = "_mostra_delegacao"

    assert chaves_lidas_da_funcao(boa, nome) <= declarados, "ancora: a fonte correta passa"

    errada = boa.replace('get("abstains")', 'get("acerto")')
    lidas = chaves_lidas_da_funcao(errada, nome)
    assert not (lidas <= declarados), (
        "o portao nao ve a chave que o esquema nao declara — e `acerto` e justamente a que NAO "
        f"existe, porque a politica nao mede acerto nesta matriz: {sorted(lidas - declarados)}"
    )
    prosa = (
        "def _mostra_delegacao(st, delegacao):\n"
        '    """Le delegacao.get("acerto")."""\n'
        "    return None\n"
    )
    assert chaves_lidas_da_funcao(prosa, nome) == set(), "docstring foi contada como leitura"


def test_u19_o_portao_da_conclusao_acusa_a_chave_que_o_esquema_nao_declara() -> None:
    """O negativo, **em processo**: a planta e a fonte que a extracao le.

    Par causal, e as duas leituras diferem so pela chave trocada. Sem ele, uma extracao que
    devolvesse vazio satisfaria a contencao do teste acima.
    """
    from refino.api.esquemas import ResultadoDoMultiverso

    declarados = set(ResultadoDoMultiverso.model_fields)
    corpo = "\n".join(
        f'    x = conclusao.get("{chave}")' for chave in sorted(CHAVES_DA_CONCLUSAO_EXIBIDA)
    )
    boa = f"def _exibe_conclusao_do_multiverso(st, conclusao):\n{corpo}\n"
    nome = "_exibe_conclusao_do_multiverso"

    assert chaves_lidas_da_funcao(boa, nome) <= declarados, "ancora: a fonte correta passa"

    errada = boa.replace('get("grupos")', 'get("verdict")')
    lidas = chaves_lidas_da_funcao(errada, nome)
    assert not (lidas <= declarados), (
        "o portao nao ve a chave que o esquema nao declara — e `verdict` e justamente a que NAO se "
        f"serve, porque o limiar do `[CA-12.2]` esta em aberto: {sorted(lidas - declarados)}"
    )
    prosa = (
        "def _exibe_conclusao_do_multiverso(st, conclusao):\n"
        '    """Le conclusao.get("verdict")."""\n'
        "    return None\n"
    )
    assert chaves_lidas_da_funcao(prosa, nome) == set(), "docstring foi contada como leitura"


def test_u6_o_portao_do_resultado_acusa_a_chave_que_o_esquema_nao_declara() -> None:
    """O negativo, **em processo**: a planta é a fonte que a extração lê.

    **Par causal, e as duas leituras diferem só pela chave trocada.** Sem ele, uma extração que
    devolvesse vazio satisfaria a contenção do teste acima — e a âncora de presença o pegaria, mas
    só ela: nada mostraria que o portão **consegue** ver a chave inventada.
    """
    declarados = set(ExecucaoResposta.model_fields)
    corpo = "\n".join(
        f'    x = resultado.get("{chave}")' for chave in sorted(CHAVES_DO_RESULTADO_EXIBIDO)
    )
    boa = f"def _exibe_resultado_da_execucao(st, resultado):\n{corpo}\n"
    nome = "_exibe_resultado_da_execucao"

    assert chaves_lidas_da_funcao(boa, nome) <= declarados, "âncora: a fonte correta passa"

    errada = boa.replace('get("run_id")', 'get("identificador")')
    lidas = chaves_lidas_da_funcao(errada, nome)
    assert not (lidas <= declarados), (
        "o portão não vê a chave que o esquema não declara — é exactamente o defeito de R16, e ele "
        f"passaria: {sorted(lidas - declarados)}"
    )
    prosa = (
        "def _exibe_resultado_da_execucao(st, resultado):\n"
        '    """Lê resultado.get("identificador")."""\n'
        "    return None\n"
    )
    assert chaves_lidas_da_funcao(prosa, nome) == set(), "docstring foi contada como leitura"


# ---------------------------------------------------------------------------
# O botão que COMPUTA não diz «Baixar»
# ---------------------------------------------------------------------------


def test_u6_o_passo_de_artefato_declara_rotulo_de_acao_que_nao_diz_baixar() -> None:
    """A declaração decide os dois rótulos.

    `titulo` é o nome do passo no fluxo, e o que o botão de **download**
    diz. `rotulo_de_acao` é o do botão que **dispara** o cálculo, e o defeito era haver um só: o
    primeiro botão dizia *"Calcular — Baixar a matriz corrigida"*, e quem clicasse esperava arquivo.

    A âncora de presença vem antes das negações: `"Baixar" not in ""` passa, e string
    vazia é exactamente o que um passo que esquecesse o campo novo traria.
    """
    artefatos = [passo for passo in DENTRO_DO_FLUXO if passo.chave in CHAVES_DE_ARTEFATO]

    assert len(artefatos) == len(CHAVES_DE_ARTEFATO) == 3, (
        f"a partição dos artefatos mudou: {[p.chave for p in artefatos]}"
    )
    for passo in artefatos:
        assert passo.rotulo_de_acao.strip(), (
            f"o passo `{passo.chave}` produz arquivo e não declara `rotulo_de_acao`: o botão de "
            "disparo cairia no `titulo`, que é o rótulo do download"
        )
        assert "baixar" not in passo.rotulo_de_acao.lower(), (
            f"o rótulo de AÇÃO de `{passo.chave}` diz «Baixar»: {passo.rotulo_de_acao!r}"
        )
        assert "Baixar" in passo.titulo, (
            f"o `titulo` de `{passo.chave}` deixou de dizer «Baixar»: {passo.titulo!r} — é ele que "
            "vai no botão de download, e o par perde a distinção"
        )


# ---------------------------------------------------------------------------
# O portão que impede a declaração de ser decorativa
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# O fluxo sobre os quatro modos REAIS, pela API que a interface consome
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.parametrize("mode", ["pos_hilic", "pos_c18", "neg_hilic", "neg_c18"])
def test_ca_12_1_os_passos_de_leitura_do_fluxo_completam_nos_quatro_modos(mode: str) -> None:
    """Os passos que a interface executa, contra os arquivos do laboratório.

    **Recorte declarado:** exercita `validar`, `diagnosticar` e `recomendar` — os passos que **não**
    exigem banco. `execucoes` e `multiversos` têm cobertura própria em
    `tests/test_api_sobrescrita.py` e `tests/test_api_progresso.py`. Repeti-los aqui nos quatro
    modos mediria custo e não cobertura.

    A afirmação verificável é essa, e não *"o fluxo completo roda nos quatro modos pela
    interface"*.
    """
    from fastapi.testclient import TestClient

    from refino.api.app import build_app
    from tests.conftest import measurements_path, sample_list_path

    cliente = TestClient(build_app())
    arquivos = {
        "measurements": (
            f"measurements_{mode}.csv",
            measurements_path(mode).read_bytes(),
            "text/csv",
        ),
        "sample_list": (f"Sample_List_{mode}.csv", sample_list_path(mode).read_bytes(), "text/csv"),
    }

    for chave in ("validacao", "diagnostico", "recomendacao"):
        passo = next(p for p in DENTRO_DO_FLUXO if p.chave == chave)
        resposta = cliente.post(passo.rota, files=arquivos, data={"mode": mode})
        assert resposta.status_code == 200, (mode, passo.rota, resposta.text)

    corpo = cliente.post("/matrizes/recomendar", files=arquivos, data={"mode": mode}).json()
    assert corpo["rationale"].strip(), f"{mode}: justificativa vazia"
    # Este teste PREVIU a própria inversão. A mensagem antiga dizia que, se isto mudasse, o
    # recorte do critério precisava de revisão — e foi o que aconteceu. A asserção troca de lado
    # e o par que ela guarda continua: o corretor proposto é executável, e a resposta traz o
    # recorte.
    assert corpo["disponivel_para_execucao"] is True, (
        f"{mode}: a política propôs um corretor que a interface NÃO aceita executar; com "
        "as duas portas abertas, uma proposta inexecutável seria regressão"
    )
    assert any(item["item"] == "serrf" for item in corpo["ressalvas"]), (
        f"{mode}: `serrf` é proposto e executável e a resposta não carrega a ressalva medida"
    )


# ---------------------------------------------------------------------------
# `prosa_dos_lotes` contra `SugestaoDeLotes`
# ---------------------------------------------------------------------------


def campos_lidos_por_prosa_dos_lotes() -> frozenset[str]:
    """As chaves que `fluxo.prosa_dos_lotes` lê do objeto, por AST.

    Função de módulo e não corpo de teste: o negativo tem de invocar ESTA e não uma cópia, senão
    fica verde enquanto a regra apodrece.
    """
    import ast
    import inspect

    from refino.interface import fluxo

    arvore = ast.parse(inspect.getsource(fluxo.prosa_dos_lotes))
    return frozenset(
        no.args[0].value
        for no in ast.walk(arvore)
        if isinstance(no, ast.Call)
        and isinstance(no.func, ast.Attribute)
        and no.func.attr == "get"
        and no.args
        and isinstance(no.args[0], ast.Constant)
        and isinstance(no.args[0].value, str)
    )


# **Os campos que a função IGNORA de propósito, e o motivo de cada um.** Sem esta declaração a
# bijeção só teria uma direção, e a que falta é a que pega o oposto: campo novo no esquema que a
# prosa deixa de mencionar sem ninguém decidir.
CAMPOS_DE_LOTES_IGNORADOS: Final[dict[str, str]] = {
    # É o que motivou `prosa_dos_lotes` a existir: uma entrada por amostra — 68 números no
    # `pos_hilic` — despejadas onde cabiam duas frases.
    "rotulos_por_amostra": (
        "uma entrada por amostra; é o despejo que esta função existe para evitar"
    ),
    # Discriminador de união, não dado a exibir.
    "natureza": "discriminador do esquema, não quantidade a mostrar",
}


def test_prosa_dos_lotes_le_so_campos_que_o_esquema_declara() -> None:
    """A bijeção nas DUAS direções.

    **A âncora vem primeiro:** conjunto vazio de leituras passaria as duas asserções abaixo por
    vacuidade, e conjunto vazio é o que uma extração por AST que deixasse de casar devolve.
    """
    from refino.api.esquemas import SugestaoDeLotes

    lidos = campos_lidos_por_prosa_dos_lotes()
    declarados = set(SugestaoDeLotes.model_fields)

    assert lidos, (
        "nenhuma leitura extraída de `prosa_dos_lotes` — a extração por AST deixou de casar"
    )
    assert declarados, "âncora: `SugestaoDeLotes` não declara campo nenhum"

    assert lidos <= declarados, (
        "`prosa_dos_lotes` lê campo que `SugestaoDeLotes` não declara: "
        f"{sorted(lidos - declarados)}"
    )
    nao_lidos = declarados - lidos
    ignorados = set(CAMPOS_DE_LOTES_IGNORADOS)
    assert nao_lidos == ignorados, (
        f"a mais no esquema e sem declaração: {sorted(nao_lidos - ignorados)}\n"
        f"a mais na declaração e já lido: {sorted(set(CAMPOS_DE_LOTES_IGNORADOS) - nao_lidos)}\n"
        "Campo novo em `SugestaoDeLotes` é decisão: ou a prosa o menciona, ou a omissão vai "
        "declarada com o motivo."
    )


def test_negativo_a_bijecao_dos_lotes_acusa_campo_nao_declarado() -> None:
    """A planta, por argumento: um campo a mais no esquema sem declaração tem de acusar.

    E a segunda metade é a que impede o portão de sempre reprovar: com a planta removida, os
    conjuntos reais coincidem.
    """
    lidos = campos_lidos_por_prosa_dos_lotes()
    declarados_plantado = lidos | {"campo_novo_que_ninguem_declarou"}

    nao_lidos = declarados_plantado - lidos
    assert nao_lidos - set(CAMPOS_DE_LOTES_IGNORADOS) == {"campo_novo_que_ninguem_declarou"}, (
        "a planta não foi isolada — o portão não distinguiria um campo novo de um ignorado"
    )


# Os lotes do `pos_hilic` como a rota os devolve: a detecção corta em 23 e 45, os lotes ficariam
# com dois, dois e três QCs, e a sugestão é um lote único com os sete.
LOTES_QUE_RECUARAM: Final[dict[str, Any]] = {
    "natureza": "sugestao",
    "n_lotes": 1,
    "changepoints": [],
    "qcs_por_lote": [7],
    "cortes_detectados": [23, 45],
    "qcs_por_lote_detectados": [2, 2, 3],
}


def test_prosa_dos_lotes_diz_por_que_os_cortes_detectados_nao_entraram() -> None:
    """A frase nomeia os cortes achados, os QCs que cada lote teria, o limiar e o lote único."""
    from refino.diagnosis.batches import MIN_QC_PER_PROPOSED_BATCH
    from refino.interface import fluxo

    frase = fluxo.prosa_dos_lotes(LOTES_QUE_RECUARAM)

    assert frase.startswith("**Lotes sugeridos:** 1 — a detecção achou ponto de mudança"), frase
    assert "na injeção 23, 45" in frase, frase
    assert "deixariam lotes com 2, 2, 3 QC(s)" in frase, frase
    assert f"ao menos {MIN_QC_PER_PROPOSED_BATCH} QCs em cada lote" in frase, frase
    assert "**lote único**, com 7 QCs" in frase, frase
    assert "aba «Correção»" in frase, frase


def test_prosa_dos_lotes_com_os_cortes_adotados_e_a_frase_de_fronteira() -> None:
    """O par do teste de cima: com os mesmos cortes adotados, a frase é a da fronteira."""
    from refino.interface import fluxo

    adotados = {
        **LOTES_QUE_RECUARAM,
        "n_lotes": 3,
        "changepoints": [23, 45],
        "qcs_por_lote": [2, 2, 3],
    }

    frase = fluxo.prosa_dos_lotes(adotados)

    assert frase.startswith("**Lotes sugeridos:** 3, com fronteira na injeção 23, 45."), frase
    assert "lote único" not in frase, frase


# ---------------------------------------------------------------------------
# AS FAMÍLIAS DE ANÁLISE
# ---------------------------------------------------------------------------


def _campos_que_pediu_alguma_coisa_le() -> set[str]:
    """Os campos de `AnalysisSpec` que `pediu_alguma_coisa` de facto lê, extraídos por AST.

    **O oráculo é o MÉTODO e não uma lista à mão.** `pediu_alguma_coisa` existe exactamente para
    o ponto de chamada não enumerar as bandeiras, e a tabela da tela é a segunda enumeração das
    mesmas: sem bijeção, uma família nova entra no domínio e a janela não a oferece — ou entra na
    janela e a rota não a anuncia na barra. Os dois desfechos são silenciosos.
    """
    fonte = inspect.getsource(AnalysisSpec.pediu_alguma_coisa)
    arvore = ast.parse(textwrap.dedent(fonte))
    return {
        no.attr
        for no in ast.walk(arvore)
        if isinstance(no, ast.Attribute)
        and isinstance(no.value, ast.Name)
        and no.value.id == "self"
    }


def test_u23_a_tabela_de_familias_e_bijetiva_com_o_contrato_de_analise() -> None:
    """Família na tabela sem campo no contrato, e campo pedível fora da tabela — as DUAS direções.

    A âncora de não-vacuidade vem primeiro e na mesma função: a extração por AST pode
    devolver conjunto vazio — assinatura mudada, `dedent` que não casa —, e dois conjuntos vazios
    satisfazem a igualdade sem verificar nada.
    """
    lidos = _campos_que_pediu_alguma_coisa_le()
    assert len(lidos) >= 9, (
        f"ANCORA: a extração por AST achou {len(lidos)} campos em `pediu_alguma_coisa` e havia "
        "nove em 14/09/2026. Ou o método mudou de forma, ou o extrator cegou — e no segundo caso "
        "a igualdade abaixo passaria por vacuidade"
    )
    da_tabela = {familia.campo for familia in FAMILIAS_DE_ANALISE}
    assert da_tabela == lidos, (
        f"na tabela e não no contrato: {sorted(da_tabela - lidos)}\n"
        f"pedível no contrato e fora da tabela: {sorted(lidos - da_tabela)}\n"
        "Família fora da tabela não tem caixa na janela nem etapa na barra, e nada acusa."
    )


def test_u23_o_custo_da_familia_deriva_do_padrao_de_kd11() -> None:
    """Três baratas ligadas e seis caras desligadas — a contagem é sobre o padrão de HOJE.

    Contagem esperada e não invariante: nada no desenho fixa quantas famílias há nem quantas vêm
    ligadas. Este número é observação sobre `PADRAO_DA_ANALISE`, e o que o teste guarda é a
    **correspondência** — ligada por padrão é barata, desligada é cara.
    """
    baratas = tuple(f.campo for f in FAMILIAS_DE_ANALISE if familia_e_barata(f))
    caras = tuple(f.campo for f in FAMILIAS_DE_ANALISE if not familia_e_barata(f))
    # Quatro baratas: a normalidade entrou ligada, porque custa milissegundos. A contagem é
    # observação sobre o padrão de hoje, e mudou com ele.
    assert baratas == ("anova", "correlation", "permanova", "normality"), baratas
    assert caras == ("plsda_multiclass", "clustering", "kmeans", "som", "dspc", "classifiers"), (
        caras
    )
    # A correspondência que torna `familia_e_barata` mais que um sinónimo de `getattr`.
    assert set(baratas) == {f.campo for f in familias_pedidas(PADRAO_DA_ANALISE)}


def test_u23_o_palpite_cresce_na_direcao_quando_uma_familia_cara_e_marcada() -> None:
    """A DIREÇÃO e não o número.

    **O que cresce é a contagem declarada, e não segundos inventados.** `CUSTO_DA_ANALISE_S` está
    vazio enquanto nenhuma família calcula, então somar um valor aqui seria constante não derivada
    no rótulo que o profissional lê antes de clicar. O teste afirma que a frase com
    uma família cara marcada diz **estritamente mais** que a mesma frase sem ela.
    """
    sem = prosa_das_analises_pedidas(PADRAO_DA_ANALISE)
    com = prosa_das_analises_pedidas(replace(PADRAO_DA_ANALISE, clustering=True))
    assert "minutos" not in sem, sem
    assert "minutos" in com and "agrupamento" in com, com
    assert (
        len(familias_pedidas(replace(PADRAO_DA_ANALISE, clustering=True)))
        == len(familias_pedidas(PADRAO_DA_ANALISE)) + 1
    )

    # **E o SEGUNDO desfecho: sem análise declarada a frase é vazia.** `None` é *"não me disseram"*
    # e reproduz o palpite anterior a este bloco; colapsá-lo no padrão faria quem chama sem
    # argumento receber a conta de uma escolha que nunca fez.
    assert prosa_das_analises_pedidas(None) == ""
    assert prosa_das_analises_pedidas(replace(PADRAO_DA_ANALISE, **_TUDO_DESLIGADO)) == ""


_TUDO_DESLIGADO: Final[dict[str, Any]] = {
    "anova": False,
    "correlation": False,
    "permanova": False,
    # A quarta barata, ligada por padrão: sem ela aqui, «tudo desligado» pediria uma.
    "normality": False,
    "classifiers": (),
}


def test_u23_a_cauda_so_soma_o_que_esta_medido_e_a_planta_prova_os_dois_desfechos() -> None:
    """O par causal da cauda: com a tabela de custo vazia não soma; com uma entrada, soma.

    **A planta é a ENTRADA na tabela e o restauro vai num `finally`** — um arnês que morra a meio
    deixaria `CUSTO_DA_ANALISE_S` mutado para o resto da sessão, e os testes seguintes mediriam
    outra coisa sem que nada acusasse.

    Ela cai sobre dado em que as duas coisas DIFEREM: a análise plantada pede `anova`,
    que é exactamente a chave que a planta acrescenta. Sem essa coincidência o par não separaria —
    as duas medições dariam a cauda fixa e o teste passaria por construção.
    """
    analise = PADRAO_DA_ANALISE
    assert "anova" in {f.campo for f in familias_pedidas(analise)}, (
        "ANCORA: a planta só separa se a família plantada for uma das pedidas"
    )
    assert CUSTO_DA_ANALISE_S == {}, (
        "ANCORA: a tabela de custo tem entradas, e este par causal foi escrito para a tabela "
        "vazia. Com custo medido, o número de baixo muda e o teste tem de ser remedido, não "
        "afrouxado"
    )
    sem_planta = cauda_da_execucao_s(analise)
    assert sem_planta == CAUDA_DA_EXECUCAO_S

    CUSTO_DA_ANALISE_S["anova"] = 12.5
    try:
        assert "anova" in CUSTO_DA_ANALISE_S, "a planta NAO aterrou"
        com_planta = cauda_da_execucao_s(analise)
        assert com_planta == pytest.approx(sem_planta + 12.5)
        # E a família medida sai da frase de «não medido», que é o outro desfecho do mecanismo.
        assert "o custo delas ainda NÃO foi medido" not in prosa_das_analises_pedidas(analise)
    finally:
        CUSTO_DA_ANALISE_S.pop("anova", None)
    assert CUSTO_DA_ANALISE_S == {}, "o restauro NAO aconteceu"


def test_u23_a_frase_das_etapas_tem_os_dois_desfechos_e_nao_repete_a_do_palpite() -> None:
    """A frase do bloco fala de PROGRESSO; a do palpite fala de DURAÇÃO.

    Duas frases com o mesmo texto lado a lado é o defeito que este painel já pagou, com dois
    rótulos vizinhos a dizerem a mesma coisa em repouso.
    """
    nenhuma = prosa_das_etapas_pedidas(replace(PADRAO_DA_ANALISE, **_TUDO_DESLIGADO))
    algumas = prosa_das_etapas_pedidas(PADRAO_DA_ANALISE)
    assert "Nenhuma família pedida" in nenhuma and "corrige e processa como sempre" in nenhuma
    assert "4 etapas a mais" in algumas and "anova" in algumas and "normalidade" in algumas
    assert nenhuma != algumas
    assert prosa_das_analises_pedidas(PADRAO_DA_ANALISE) not in algumas


def test_u23_o_negativo_da_bijecao_das_familias_acusa_nas_duas_direcoes() -> None:
    """A planta, por ARGUMENTO: tabela a menos e tabela a mais — e a árvore limpa não acusa.

    **Em processo e não sobre o arquivo**, então a planta alimenta a asserção e o negativo não
    consegue ficar inerte: a comparação é entre dois conjuntos que este teste constrói.
    """
    lidos = _campos_que_pediu_alguma_coisa_le()
    limpa = {familia.campo for familia in FAMILIAS_DE_ANALISE}
    assert limpa == lidos, "a arvore limpa nao pode acusar"

    a_menos = limpa - {"dspc"}
    assert "dspc" in limpa, "a planta NAO aterrou: `dspc` não estava na tabela"
    assert lidos - a_menos == {"dspc"}, "a bijeção não acusaria uma família fora da tabela"

    a_mais = limpa | {"familia_que_o_contrato_nao_tem"}
    assert a_mais - lidos == {"familia_que_o_contrato_nao_tem"}, (
        "a bijeção não acusaria uma tabela que oferece o que a rota não sabe pedir"
    )


def test_u23_toda_familia_tem_dica_e_toda_dica_tem_familia() -> None:
    """A bijeção do texto que explica cada família — as duas direções.

    **Sem dica, a caixa aparece na tela com o rótulo curto e mais nada**, e o rótulo é curto por
    medição: a frase que explica custava 631 px de largura mínima à coluna quando vivia nele. A
    direção de dica órfã é a que apodrece sem sintoma — ninguém a vê deixar de ser usada.
    """
    campos = {familia.campo for familia in FAMILIAS_DE_ANALISE}
    assert campos, "ANCORA: a tabela de famílias está vazia"
    assert set(DICA_DA_FAMILIA) == campos, (
        f"família sem dica: {sorted(campos - set(DICA_DA_FAMILIA))}\n"
        f"dica sem família: {sorted(set(DICA_DA_FAMILIA) - campos)}"
    )
    vazias = [campo for campo, dica in DICA_DA_FAMILIA.items() if not dica.strip()]
    assert vazias == [], f"dica vazia é o mesmo que dica ausente, com o portão verde: {vazias}"


# A LINHA DO VALOR — formato, população e os dois rótulos
# ---------------------------------------------------------------------------
#
# O relatório destes testes é DITADO e nenhum número é medido: o que
# se afirma é como a linha se **escreve**, não quanto vale nada. Os valores têm a forma de uma
# medição para o formatador ter o que formatar.

_RELATORIO_PARA_FORMATAR: dict[str, Any] = {
    "computed": sorted(REQUIRED_METRIC_FIELDS),
    "rsd_variant": "plain",
    "rsd_estimator": "in_sample",
    "n_features_rsd_defined": 5933,
    "n_features_rsd_undefined": 102,
    "qc_rsd_median": 0.1799,
    "qc_rsd_mean": 0.31,
    "frac_features_under_20": 0.5801668211306765,
    "frac_features_under_30": 0.72,
    "qc_dispersion_ratio": 1.8523,
    "pc1_injection_correlation_abs": 0.08,
}


def test_d452_a_fracao_sai_em_percentagem_com_a_populacao_e_os_dois_rotulos() -> None:
    """O defeito que o usuário nomeou, e as três metades do conserto.

    Ele escreveu que `frac_features_under_20` *"faz sentido para dev"*; o que a tela mostrava era
    `0.5801668211306765`. Passa a mostrar a percentagem, o **denominador** e os dois rótulos sem
    os quais o RSD não é comparável.

    **A ÂNCORA é sobre QUAL denominador**, e é ela que torna este teste diferente de uma asserção
    de texto: o corpo carrega também as contagens do painel de antes-e-depois, e elas **divergem**
    do denominador certo. Uma implementação que compusesse a população a partir do número que está
    à mão escreveria 4000 ou 6035 e este teste acusaria.
    """
    corpo = dict(_RELATORIO_PARA_FORMATAR)
    corpo["n_features_comparadas"] = 4000
    corpo["n_features_saida"] = 6035
    assert corpo["n_features_rsd_defined"] not in (4000, 6035), (
        "âncora: o denominador certo coincide com um dos números vizinhos, e o teste não "
        "distinguiria a composição errada da leitura certa"
    )

    linha = valor_da_metrica("frac_features_under_20", corpo)

    # Os dois rótulos estão em português. O do estimador diz a circularidade em vez do nome dela:
    # é ela que faz o LOESS zerar o RSD por construção.
    assert linha == (
        "58,0% — sobre as 5933 features com RSD definido; sem remoção de outliers, "
        "medido nos próprios QCs do ajuste"
    ), linha
    assert "0.5801668211306765" not in linha, "os dezasseis dígitos voltaram à tela"
    # A igualdade acima já ancora esta: os valores crus saíram da linha do valor.
    assert "in_sample" not in linha and "outlier_removed" not in linha, linha


def test_d452_a_razao_sai_em_casas_decimais_e_a_percentagem_nao() -> None:
    """O par que separa os dois formatos — e sobre o valor errado ele seria vácuo.

    `1,8523` e `185,2%` são o **mesmo** número escrito em duas unidades, e é a tabela que decide
    qual sai. Sobre um valor que os dois formatos renderizassem igual — não existe nenhum, e é por
    isso que a âncora abaixo pode ser uma desigualdade — o cenário não distinguiria nada.
    """
    razao = valor_da_metrica("qc_dispersion_ratio", _RELATORIO_PARA_FORMATAR)
    percentagem = valor_da_metrica("qc_rsd_median", _RELATORIO_PARA_FORMATAR)

    assert razao == "1,852", razao
    assert "%" not in razao, "uma razão sem unidade saiu em percentagem"
    assert percentagem.startswith("18,0%"), percentagem

    # A âncora do par: os dois vêm do MESMO formatador e do mesmo relatório, e divergem só pelo
    # campo `formato` da tabela.
    assert TRADUCAO_DA_METRICA["qc_dispersion_ratio"].formato == "razao"
    assert TRADUCAO_DA_METRICA["qc_rsd_median"].formato == "percentagem"


def test_d452_um_valor_ausente_nao_escreve_a_palavra_do_nulo() -> None:
    """A tela mostrava `qc_rsd_median: None`, literalmente.

    `nan` chega ao cliente como `null` — pydantic serializa não finito assim —, e `str(None)` é a
    palavra do nulo: nem português nem inglês. **Par causal:** o mesmo campo com número e sem, e
    afirma-se que os dois desfechos divergem e que nenhum contém *"None"*.
    """
    com = dict(_RELATORIO_PARA_FORMATAR)
    sem = dict(_RELATORIO_PARA_FORMATAR, qc_rsd_median=None)

    texto_com = valor_da_metrica("qc_rsd_median", com)
    texto_sem = valor_da_metrica("qc_rsd_median", sem)

    assert texto_com != texto_sem, "âncora: o valor ausente não mudou nada"
    # As DUAS âncoras de não-vacuidade ficam AQUI: as duas asserções de ausência
    # abaixo passam sobre string vazia, e `texto_com` não estava preso a nada — a desigualdade
    # acima é satisfeita por um formatador que devolvesse `""` num dos lados.
    assert texto_com.startswith("18,0%"), texto_com
    assert texto_sem.startswith(SEM_VALOR_DEFINIDO), texto_sem
    assert "None" not in texto_sem and "None" not in texto_com
    # E a população continua a sair: saber que a métrica é indefinida **sobre 5933 features** é
    # outra coisa que saber que ela é indefinida.
    assert "5933" in texto_sem


def test_d452_denominador_zero_nao_escreve_fracao_de_denominador_zero() -> None:
    """**A borda que o produto SABE produzir**, e o par que a separa de vacuidade.

    `build_quality_report` sobre uma matriz cujas features têm todas RSD indefinido devolve
    `frac_features_under_20 = nan` e `n_features_rsd_defined = 0` — há teste em
    `tests/unit/test_quality_report.py` que o constrói. A linha do valor diz
    *calculada-e-indefinida*, e a da população **não** escreve *"0 de 0"*, que é um convite a
    dividir.

    **O par que separa:** o mesmo relatório com **uma** feature definida. Sem ele, um formatador
    que nunca escrevesse a fração passaria.
    """
    zero = dict(_RELATORIO_PARA_FORMATAR, frac_features_under_20=None, n_features_rsd_defined=0)
    uma = dict(_RELATORIO_PARA_FORMATAR, frac_features_under_20=1.0, n_features_rsd_defined=1)

    texto_zero = valor_da_metrica("frac_features_under_20", zero)
    texto_uma = valor_da_metrica("frac_features_under_20", uma)

    assert texto_zero != texto_uma, "âncora: o denominador zero não mudou nada"
    assert texto_zero == (
        f"{SEM_VALOR_DEFINIDO} — nenhuma feature tem RSD definido; "
        "sem remoção de outliers, medido nos próprios QCs do ajuste"
    ), texto_zero
    assert "0 de 0" not in texto_zero and "as 0 features" not in texto_zero
    # **A concordância muda com o denominador, e a asserção fixa-a:** *"sobre as 1 features"* é a
    # assinatura de texto montado por máquina.
    assert texto_uma == (
        "100,0% — sobre a 1 feature com RSD definido; sem remoção de outliers, "
        "medido nos próprios QCs do ajuste"
    ), texto_uma


def test_d452_sem_os_dois_rotulos_a_linha_declara_a_ausencia_em_vez_do_nulo() -> None:
    """O outro ramo dos rótulos, e ele é alcançável: um corpo de rota anterior não os traz.

    **Par causal:** o mesmo relatório com e sem `rsd_variant`/`rsd_estimator`.
    """
    com = valor_da_metrica("qc_rsd_median", _RELATORIO_PARA_FORMATAR)
    sem_corpo = {
        c: v
        for c, v in _RELATORIO_PARA_FORMATAR.items()
        if c not in ("rsd_variant", "rsd_estimator")
    }
    sem = valor_da_metrica("qc_rsd_median", sem_corpo)

    assert com != sem, "âncora: tirar os rótulos não mudou nada"
    # A agulha é a PROSA e não o valor cru. Ela dizia `"variante plain"`, e a tradução
    # dos valores é metade desta unidade: `plain` é tão *"coisa com underscore"* para quem lê
    # quanto o nome do campo era.
    assert prosa_da_variante_de_rsd("plain") in com, com
    assert "não declarados" in sem
    assert "None" not in sem, f"a palavra do nulo voltou nos rótulos: {sem!r}"


def test_d452_toda_metrica_da_tela_declara_formato_e_populacao() -> None:
    """A bijeção com o campo novo OBRIGATÓRIO — e os dois vocabulários são FECHADOS.

    Métrica nova no `QualityReport` entra em `METRICAS_DA_TELA_DE_QUALIDADE` sozinha e tem de ter
    entrada nesta tabela; sem padrão nos dois campos, a entrada não se constrói sem os declarar. O
    que este teste acrescenta é que o valor declarado é um dos vocabulários e não uma string
    qualquer.

    O DOMÍNIO da bijeção são DOIS conjuntos e não um.
    Esta linha comparava contra `METRICAS_DA_TELA_DE_QUALIDADE` sozinha, e era verdade enquanto a
    tela escrevia nome de uma origem só. Hoje escreve de duas: a grelha das onze escalares e a
    seção do PLS-DA, que inclui o par do multiclasse — e esse par **não é das onze**, porque não é
    escalar. Manter o conjunto único faria a bijeção reprovar por uma família nova ter nascido,
    que é o portão a acusar o alvo certo pelo motivo errado.
    """
    assert TRADUCAO_DA_METRICA, "âncora: a tabela está vazia e o laço abaixo é vácuo"
    _ancora_da_traducao()

    formatos = set(get_args(FormatoDoValor))
    populacoes = set(get_args(PopulacaoDaMetrica))
    assert formatos and populacoes, "âncora: o vocabulário está vazio"

    for nome, traduzida in TRADUCAO_DA_METRICA.items():
        assert traduzida.formato in formatos, f"`{nome}` declara formato `{traduzida.formato}`"
        assert traduzida.populacao in populacoes, f"`{nome}` declara `{traduzida.populacao}`"

    # **A partição por população é MEDIDA e não escrita**: as quatro agregações do vetor de RSD
    # levam denominador e rótulos; as outras sete não — as duas de RSD fora da amostra ficam de
    # fora porque medem por OUTRO estimador, e colar-lhes o do relatório seria rótulo errado.
    do_vetor = {n for n, tr in TRADUCAO_DA_METRICA.items() if tr.populacao == "vetor_de_rsd"}
    assert do_vetor == {
        "qc_rsd_median",
        "qc_rsd_mean",
        "frac_features_under_20",
        "frac_features_under_30",
    }, sorted(do_vetor)


# ---------------------------------------------------------------------------
# Os VALORES também vão para português, e o identificador vira ENDEREÇO
# ---------------------------------------------------------------------------


def test_u13_os_dois_vocabularios_de_rsd_sao_traduzidos_NA_INTEGRA() -> None:  # noqa: N802
    """Bijeção contra `domain/labels.py`, **nas duas direções**, com a planta.

    **Medido em 15/09/2026:** dos 33 identificadores `snake_case` da aba alimentada, **nove** eram
    VALORES — `in_sample`, `outlier_removed`, `leave_one_qc_out` — e não nomes de métrica. O pedido
    do usuário alcança-os: eles são *"coisas com underscore"* para quem lê a tela.

    **As duas direções, e a segunda é a que fica inerte:** rótulo do domínio sem tradução
    apareceria cru; tradução órfã sobrevive a um rótulo que saiu do `Literal`.

    **A planta é EM PROCESSO** — a tabela entra pelo argumento do `dict` local —, e é ela que prova
    que o par de asserções acima consegue reprovar.
    """
    from refino.domain.labels import RsdEstimator, RsdVariant

    do_dominio_variante = set(get_args(RsdVariant))
    do_dominio_estimador = set(get_args(RsdEstimator))
    assert do_dominio_variante and do_dominio_estimador, "âncora: o vocabulário do domínio é vazio"

    assert set(PROSA_DA_VARIANTE_DE_RSD) == do_dominio_variante, (
        f"a tradução da variante não é bijetiva: tabela={sorted(PROSA_DA_VARIANTE_DE_RSD)} "
        f"domínio={sorted(do_dominio_variante)}"
    )
    assert set(PROSA_DO_ESTIMADOR_DE_RSD) == do_dominio_estimador, (
        f"a tradução do estimador não é bijetiva: tabela={sorted(PROSA_DO_ESTIMADOR_DE_RSD)} "
        f"domínio={sorted(do_dominio_estimador)}"
    )

    # E nenhuma tradução é o próprio valor cru disfarçado.
    for tabela in (PROSA_DA_VARIANTE_DE_RSD, PROSA_DO_ESTIMADOR_DE_RSD):
        for rotulo, prosa in tabela.items():
            assert rotulo not in prosa, f"a tradução de `{rotulo}` repete o identificador: {prosa}"
            assert "_" not in prosa, f"a tradução de `{rotulo}` ainda tem sublinhado: {prosa}"

    # **A PLANTA, nas duas direções e em processo.**
    faltando = {k: v for k, v in PROSA_DO_ESTIMADOR_DE_RSD.items() if k != "r_cv"}
    assert set(faltando) != do_dominio_estimador, "a planta da omissão não aterrou"
    orfa = {**PROSA_DA_VARIANTE_DE_RSD, "uma_variante_que_nao_existe": "seja lá o que for"}
    assert set(orfa) != do_dominio_variante, "a planta da tradução órfã não aterrou"


def test_u13_um_rotulo_de_rsd_fora_do_vocabulario_aparece_MARCADO() -> None:  # noqa: N802
    """Valor desconhecido não cai num `else` razoável.

    **Par causal**: o mesmo par de funções sobre um rótulo do domínio e sobre um que não existe. Um
    tradutor que devolvesse o valor cru faria um rótulo novo reaparecer em inglês na tela sem nada
    o denunciar — que é exactamente o defeito que esta unidade conserta, a acontecer outra vez.
    """
    conhecida = prosa_da_variante_de_rsd("plain")
    desconhecida = prosa_da_variante_de_rsd("uma_variante_que_nao_existe")
    assert conhecida != desconhecida, "âncora: os dois desfechos colapsaram"
    assert "sem tradução nesta tela" not in conhecida, conhecida
    assert "sem tradução nesta tela" in desconhecida, desconhecida
    assert "uma_variante_que_nao_existe" in desconhecida, (
        "a marca apagou o valor, e quem lê deixa de saber o que a resposta trouxe"
    )

    conhecido = prosa_do_estimador_de_rsd("in_sample")
    desconhecido = prosa_do_estimador_de_rsd("um_estimador_que_nao_existe")
    assert conhecido != desconhecido, "âncora: os dois desfechos colapsaram"
    assert "sem tradução nesta tela" in desconhecido, desconhecido

    # E `None` — o corpo de uma rota anterior — não escreve a palavra do nulo sozinha na tela.
    assert "sem tradução nesta tela" in prosa_do_estimador_de_rsd(None)


def test_u13_o_par_de_rotulos_sai_JUNTO_e_na_ordem_declarada() -> None:  # noqa: N802
    """Os dois rótulos são um par: sem os dois, o RSD não é comparável.

    A âncora é a ORDEM, e ela não é estilo: variante antes de estimador é a ordem declarada, e a
    mesma nas duas superfícies que a chamam — a linha do valor das
    quatro agregações e a linha de antes-e-depois. Sobre duas prosas iguais o par seria vácuo.
    """
    par = prosa_dos_rotulos_do_rsd("outlier_removed", "leave_one_qc_out")
    variante = prosa_da_variante_de_rsd("outlier_removed")
    estimador = prosa_do_estimador_de_rsd("leave_one_qc_out")

    assert variante != estimador, "âncora: as duas prosas coincidem e a ordem não é observável"
    assert variante in par and estimador in par, par
    assert par.index(variante) < par.index(estimador), (
        f"o estimador saiu antes da variante, e a ordem é a variante e depois o estimador: {par}"
    )


def test_u13_a_legenda_termina_no_endereco_e_o_nome_sai_da_abertura() -> None:
    """O identificador passa de RÓTULO a ENDEREÇO — uma vez, e no fim.

    **As duas direções**: ele **está** na legenda (presença, que ancora) e **não** abre a legenda
    nem aparece duas vezes. Um teste só da ausência passaria sobre uma tabela que perdesse a
    entrada e devolvesse texto vazio.

    **O ramo do desconhecido NÃO muda**, e isso vai afirmado: ali o identificador é a única coisa
    verdadeira que a tela sabe dizer.
    """
    _ancora_da_traducao()
    for nome in TRADUCAO_DA_METRICA:
        legenda = legenda_da_metrica(nome)
        assert legenda.startswith(PREFIXO_DE_LEITURA), legenda
        assert legenda.endswith(endereco_da_metrica(nome)), legenda
        assert legenda.count(nome) == 1, (
            f"`{nome}` aparece {legenda.count(nome)} vezes na legenda — o endereço é uma nota de "
            f"rodapé e não um rótulo: {legenda!r}"
        )

    # **O ramo do desconhecido é o contrário, e o par causal está aqui.**
    orfa = legenda_da_metrica("uma_metrica_que_nao_existe")
    assert orfa.startswith("`uma_metrica_que_nao_existe`"), orfa
    assert "sem tradução nesta tela" in orfa, orfa


def test_u13_os_generos_da_legenda_separam_se_por_LINHA_EM_BRANCO() -> None:  # noqa: N802
    """`\\n\\n` e não `\\n`, e a quebra simples COLAPSA — medido, não suposto.

    A legenda era um parágrafo único de 200 a 500 caracteres com a unidade, o sentido, a ressalva e
    a referência bibliográfica todos colados, onze vezes numa coluna só.

    **A planta é o separador de uma `\\n` só**, sobre exactamente o mesmo texto: é ela que prova que
    a contagem de blocos não passaria com qualquer separador. A medição com `QTextDocument` vive em
    `tests/test_desktop_qualidade.py`, que é onde o Qt está disponível; aqui afirma-se a ESTRUTURA
    do texto, que é o que este módulo produz.
    """
    legenda = legenda_da_metrica("frac_features_under_20")
    blocos = legenda.split("\n\n")
    assert len(blocos) == 4, f"a legenda tem {len(blocos)} gêneros: {blocos}"
    assert blocos[0].startswith(PREFIXO_DE_LEITURA), blocos[0]
    assert "COM RESSALVA DE ESCOPO" in blocos[2], blocos[2]
    assert blocos[-1] == endereco_da_metrica("frac_features_under_20"), blocos[-1]
    # Nenhum bloco é vazio: um separador a mais daria um bloco em branco, que em markdown some e
    # faria a contagem acima medir outra coisa.
    assert all(b.strip() for b in blocos), blocos

    # **E a de «sem caminho» tem QUATRO também, com um gênero diferente** — a frase de ausência
    # entra e a referência não existe. É o par que impede a contagem de ser um retrato de uma só
    # métrica.
    sem_caminho = legenda_da_metrica("qc_rsd_cv_median").split("\n\n")
    assert len(sem_caminho) == 4, sem_caminho
    assert any(NENHUMA_ROTA_CALCULA in b for b in sem_caminho), sem_caminho
    assert not any("Referência da área" in b for b in sem_caminho), sem_caminho


def test_d452_metrica_sem_ancora_na_literatura_nao_inventa_referencia() -> None:
    """A partição é por PROCEDÊNCIA, e a tela não inventa fonte nenhuma.

    **Par causal:** uma métrica com âncora lida e uma sem, e afirma-se que a legenda de uma
    nomeia a fonte e a da outra não fala de referência.

    E as DUAS fontes de segunda mão não aparecem em lado nenhum. O corte de RSD chegou citando
    Sangster 2006 e Dunn 2011 de segunda mão, e citá-las de primeira seria procedência inventada
    por construção.
    """
    com_ancora = legenda_da_metrica("qc_rsd_median")
    sem_ancora = legenda_da_metrica("qc_dispersion_ratio")

    assert "Broadhurst" in com_ancora, com_ancora
    assert "10.1007/s11306-018-1367-3" in com_ancora
    # A âncora de não-vacuidade da linha seguinte fica AQUI: `not in ""` passa,
    # e legenda vazia é o que se obtém de uma tabela que perdeu a entrada.
    #
    # A ABERTURA da legenda mudou. Era o nome técnico entre crases; passou
    # a ser o rótulo `Como ler:`, e o nome reaparece uma vez, no FIM, como endereço. A âncora
    # continua a ser de presença e continua a ficar antes da asserção de ausência.
    assert sem_ancora.startswith(PREFIXO_DE_LEITURA), sem_ancora
    assert sem_ancora.endswith(endereco_da_metrica("qc_dispersion_ratio")), sem_ancora
    assert "Referência da área" not in sem_ancora, sem_ancora

    # A ressalva de ESCOPO sai junto com a âncora de classe (b), e nunca sozinha.
    corte = legenda_da_metrica("frac_features_under_20")
    assert "COM RESSALVA DE ESCOPO" in corte
    assert "coincide" in corte and "três dos quatro modos" in corte

    # **E a âncora do LAÇO, pela mesma cláusula:** sobre uma tabela vazia ele não corre, e a
    # asserção de ausência lá dentro passa por não acontecer.
    _ancora_da_traducao()
    for nome in TRADUCAO_DA_METRICA:
        legenda = legenda_da_metrica(nome)
        assert "Sangster" not in legenda and "Dunn" not in legenda, (
            f"`{nome}` cita de primeira uma fonte que o repositório tem de segunda mão"
        )

    # A partição passou de TRÊS para QUATRO, com decisão. `frac_features_under_30` estava fora
    # citando a regra *«as restantes não têm âncora e ficam sem referência»*; mas a referência vai
    # só nas métricas cuja fonte foi LIDA, e a fonte foi lida e nomeia as duas formas do corte — a
    # frase de Broadhurst está citada verbatim em `REFERENCIA_DO_CORTE_DE_20`, e
    # `src/refino/diagnosis/policy.py:37-39` já a lia assim.
    #
    # Este portão reprovou a mudança, e a decisão foi escrita antes de ele ser tocado. Actualizar
    # o conjunto esperado aqui não é afrouxar: o conjunto é contrato de apresentação, não número
    # medido, e o que mudou foi a partição.
    com_referencia = {n for n, tr in TRADUCAO_DA_METRICA.items() if tr.referencia}
    assert com_referencia == {
        "qc_rsd_median",
        "qc_rsd_mean",
        "frac_features_under_20",
        "frac_features_under_30",
    }, f"a partição de KTD16 mudou sem decisão: {sorted(com_referencia)}"

    # **E as duas de RSD saem com ressalvas INVERSAS, que é o que impede as duas linhas de se
    # lerem como a mesma frase:** 20% é a forma mais estrita e neste corpus é exigente; 30% é a
    # mais frouxa, e nos mesmos quatro arquivos dois dos quatro modos ficam abaixo dela.
    corte_30 = legenda_da_metrica("frac_features_under_30")
    assert "COM RESSALVA DE ESCOPO" in corte_30, corte_30
    assert "mais FROUXA" in corte_30 and "dois dos quatro modos" in corte_30, corte_30
    assert "mais estrita" in corte, corte


# ---------------------------------------------------------------------------
# As onze agrupadas por tema, e as três sob argumento dizem de que dependem
# ---------------------------------------------------------------------------


def test_u6_a_particao_por_tema_cobre_as_onze_em_exactamente_um_grupo() -> None:
    """**QUATRO temas mais um RESIDUAL NOMEADO, e a composição é afirmada membro a membro.**

    **O conjunto e não a contagem**, e essa é a diferença que importa: `4 + 2 + 2 + 2 + 1 = 11`
    continua a valer se uma métrica sair de um tema e outra entrar. O que este teste afirma é
    **quem** está em cada grupo.

    **O residual é NOMEADO e não é um balde «outras».** A diferença é exactamente esta asserção:
    um residual nomeado tem composição afirmada — hoje `n_significant_fdr`, e só ele —, e um balde
    absorveria o erro seguinte em silêncio.

    **O tema do PLS-DA é derivado e não escrito:** o conjunto esperado é `METRICAS_DO_PLSDA`, que
    sai de `PARES_DE_VALIDACAO_OBRIGATORIA`. Escrever os dois nomes à mão aqui criaria a segunda
    fonte que diverge no dia em que `PARES_DE_VALIDACAO_OBRIGATORIA` declarar outro par.

    **O ORÁCULO É A TABELA E NÃO AS MOLDURAS**, e isso vai declarado: o par do PLS-DA é desenhado
    duas vezes na janela — a lista e a seção da validação do PLS-DA —, de propósito. A
    propriedade *"toda métrica tem exactamente um tema"* é sobre `TRADUCAO_DA_METRICA`; ao nível do
    widget ela é falsa e continua a sê-lo.
    """
    grupos = agrupa_metricas_da_qualidade()
    por_chave = {grupo.chave: set(grupo.metricas) for grupo in grupos}

    assert grupos, "âncora: a partição está vazia e todas as asserções abaixo seriam vácuas"
    assert len(METRICAS_DA_TELA_DE_QUALIDADE) == 11, (
        f"a população mudou: {len(METRICAS_DA_TELA_DE_QUALIDADE)} métricas"
    )
    assert CHAVE_SEM_TEMA not in por_chave, (
        f"há métrica sem tema declarado: {sorted(por_chave.get(CHAVE_SEM_TEMA, ()))}"
    )

    assert por_chave == {
        "dispersao_na_amostra": {
            "qc_rsd_median",
            "qc_rsd_mean",
            "frac_features_under_20",
            "frac_features_under_30",
        },
        "dispersao_fora_da_amostra": {"qc_rsd_cv_median", "qc_rsd_r_cv_median"},
        "estrutura_no_pca": {"qc_dispersion_ratio", "pc1_injection_correlation_abs"},
        "validacao_do_plsda": set(METRICAS_DO_PLSDA),
        # O RESIDUAL, nomeado: o campo `sentido` dela diz que não é nota de qualidade, e pô-la
        # sob qualquer um dos quatro seria o agrupamento a afirmar por vizinhança o contrário.
        "nao_e_nota_de_qualidade": {"n_significant_fdr"},
    }, {chave: sorted(nomes) for chave, nomes in por_chave.items()}

    # **Exactamente um grupo por métrica, e a soma é o conjunto** — as duas direções.
    todas = [nome for grupo in grupos for nome in grupo.metricas]
    assert len(todas) == len(set(todas)), (
        f"há métrica em mais de um grupo: {sorted(n for n in todas if todas.count(n) > 1)}"
    )
    assert set(todas) == METRICAS_DA_TELA_DE_QUALIDADE, (
        f"a partição perdeu {sorted(METRICAS_DA_TELA_DE_QUALIDADE - set(todas))} e inventou "
        f"{sorted(set(todas) - METRICAS_DA_TELA_DE_QUALIDADE)}"
    )
    for grupo in grupos:
        assert grupo.metricas, f"o tema `{grupo.chave}` saiu sem métrica nenhuma"


def test_u6_o_vocabulario_de_tema_e_fechado_e_cada_tema_tem_titulo_e_nota() -> None:
    """Tema fora do vocabulário não cai num `else` razoável.

    **`Literal` é análise estática, e esta tabela é escrita à mão** — o que torna a guarda
    necessária é a mesma razão de sempre: o mypy prova os pontos de chamada, e não impede alguém
    de escrever um tema novo sem o declarar.

    **Título e nota são afirmados um a um, e os títulos têm de ser distintos:** dois temas com o
    mesmo cabeçalho seriam dois grupos que quem lê a tela não tem como separar.
    """
    assert TEMAS_DA_QUALIDADE, "âncora: não há tema declarado e o laço abaixo é vácuo"
    vocabulario = set(get_args(TemaDaMetrica))
    assert vocabulario == set(TEMAS_DA_QUALIDADE), (
        "o `Literal` dos temas e a tabela de cabeçalhos divergiram: "
        f"{sorted(vocabulario.symmetric_difference(set(TEMAS_DA_QUALIDADE)))}"
    )

    _ancora_da_traducao()
    for nome, traduzida in TRADUCAO_DA_METRICA.items():
        assert traduzida.tema in vocabulario, f"`{nome}` declara o tema `{traduzida.tema}`"

    for chave, tema in TEMAS_DA_QUALIDADE.items():
        assert tema.titulo.strip(), f"o tema `{chave}` não tem título"
        assert tema.nota.strip(), f"o tema `{chave}` não tem nota"
    titulos = {tema.titulo for tema in TEMAS_DA_QUALIDADE.values()}
    assert len(titulos) == len(TEMAS_DA_QUALIDADE), f"dois temas partilham o título: {titulos}"


def test_u6_uma_metrica_sem_tema_declarado_aparece_marcada_e_nao_some() -> None:
    """O cenário de ERRO, que um `else` razoável esconderia.

    Negativo **em processo** — a planta é a tabela passada por argumento, como a bijeção já faz —,
    então ele não consegue ficar inerte: não há intermediário que possa não receber a mutação.

    **DUAS plantas, porque há dois caminhos para a mesma pergunta:** tema fora do vocabulário, e
    nome sem entrada nenhuma na tabela. As duas têm de acabar no MESMO grupo marcado, e é isso que
    impede um segundo estado de nascer para uma pergunta só.

    A direção de sucesso vem primeiro: sobre a tabela real não há grupo marcado, e sem
    essa âncora a asserção de presença abaixo não distinguiria a planta de nada.
    """

    def marcadas(tabela: dict[str, MetricaTraduzida]) -> tuple[str, ...]:
        for grupo in agrupa_metricas_da_qualidade(tabela):
            if grupo.chave == CHAVE_SEM_TEMA:
                return grupo.metricas
        return ()

    def cobertas(tabela: dict[str, MetricaTraduzida]) -> set[str]:
        return {n for grupo in agrupa_metricas_da_qualidade(tabela) for n in grupo.metricas}

    def no_tema(tabela: dict[str, MetricaTraduzida], chave: str) -> set[str]:
        return {
            n
            for grupo in agrupa_metricas_da_qualidade(tabela)
            for n in grupo.metricas
            if grupo.chave == chave
        }

    assert marcadas(dict(TRADUCAO_DA_METRICA)) == (), (
        "a árvore limpa já tem grupo marcado: as plantas abaixo não distinguiriam nada"
    )

    # Planta 1: tema fora do vocabulário fechado.
    alvo = "qc_dispersion_ratio"
    assert alvo in no_tema(dict(TRADUCAO_DA_METRICA), "estrutura_no_pca"), (
        "âncora: a métrica escolhida já não está no tema de que a planta a tira"
    )
    sem_tema = dict(TRADUCAO_DA_METRICA)
    sem_tema[alvo] = replace(
        sem_tema[alvo], tema=cast("TemaDaMetrica", "tema_que_ninguem_declarou")
    )
    assert sem_tema[alvo].tema != TRADUCAO_DA_METRICA[alvo].tema, "a planta de tema NÃO aterrou"
    assert marcadas(sem_tema) == (alvo,), (
        f"a métrica sem tema não apareceu marcada: {marcadas(sem_tema)}"
    )
    assert alvo not in no_tema(sem_tema, "estrutura_no_pca"), (
        "a métrica plantada continuou no tema antigo — a planta não separou"
    )
    # Ela não some: marcar sem mostrar seria esconder uma métrica de qualidade
    # na única camada em que o usuário olha.
    assert cobertas(sem_tema) == METRICAS_DA_TELA_DE_QUALIDADE, (
        "a métrica sem tema desapareceu da tela em vez de aparecer marcada"
    )

    # Planta 2: nome que a tabela de tradução não conhece de todo.
    sem_entrada = dict(TRADUCAO_DA_METRICA)
    del sem_entrada[alvo]
    assert len(sem_entrada) < len(TRADUCAO_DA_METRICA), "a planta de remoção NÃO aterrou"
    assert marcadas(sem_entrada) == (alvo,), (
        f"a métrica sem tradução não apareceu marcada: {marcadas(sem_entrada)}"
    )
    assert cobertas(sem_entrada) == METRICAS_DA_TELA_DE_QUALIDADE, (
        "a métrica sem tradução desapareceu da tela"
    )


def test_u6_o_par_do_plsda_fica_no_mesmo_grupo() -> None:
    """O Q² e o p de permutação leem-se juntos.

    No dado real o PLS-DA mostrou separação visual convincente com Q² de ~24% — provável falso
    positivo —, e um agrupamento que pusesse o score longe da validação reproduziria esse erro no
    olho de quem lê.

    **A planta separa-os e o cenário acusa**, e ela cai sobre dado em que as duas coisas diferem:
    o membro movido vai para um tema que **existe** e que o outro membro não tem.
    """

    def juntos(tabela: dict[str, MetricaTraduzida]) -> bool:
        return len({tabela[n].tema for n in METRICAS_DO_PLSDA}) == 1

    assert len(METRICAS_DO_PLSDA) == 2, (
        f"o par tem {len(METRICAS_DO_PLSDA)} membros; com um só, «juntos» é verdade por vacuidade"
    )
    assert juntos(dict(TRADUCAO_DA_METRICA)), "o par do PLS-DA já está separado na árvore limpa"

    separada = dict(TRADUCAO_DA_METRICA)
    movida = sorted(METRICAS_DO_PLSDA)[0]
    separada[movida] = replace(separada[movida], tema="estrutura_no_pca")
    assert separada[movida].tema != TRADUCAO_DA_METRICA[movida].tema, "a planta NÃO aterrou"
    assert not juntos(separada), f"o cenário não acusou `{movida}` fora do tema do par"


def test_u6_as_tres_sob_argumento_dizem_de_que_dependem_e_onde_se_escolhe() -> None:
    """*"Não calculada"* sozinho lê-se como falha da ferramenta, e o fato é o oposto.

    As três dependem da mesma coisa — o par de grupos declarado —, e a rota calcula-as quando ele
    vem. O que faltava não era um número: era dizer que a ação é de quem opera e **onde** ela está.

    **Par causal:** uma métrica sob argumento e uma que tem caminho e não depende de argumento
    nenhum. A segunda continua a dizer só *"não calculada"*, que é o texto certo para ela — outra
    execução preenche-a sem que ninguém escolha nada.

    **A ABA é nomeada, e os dois campos também.** Nomear só a aba manda procurar dentro dela, que
    tem dois painéis e uma rolagem; nomear só os campos não diz onde eles estão.
    """
    assert METRICAS_DO_PLSDA | {"n_significant_fdr"} == METRICAS_SOB_PAR_DE_GRUPOS, (
        f"a classe «sob argumento» mudou: {sorted(METRICAS_SOB_PAR_DE_GRUPOS)}"
    )
    assert len(METRICAS_SOB_PAR_DE_GRUPOS) == 3, "âncora: a classe encolheu"

    for nome in sorted(METRICAS_SOB_PAR_DE_GRUPOS):
        assert estado_de_ausencia(nome) == SOB_PAR_DE_GRUPOS, (
            f"`{nome}` depende do par de grupos e a tela não o diz: {estado_de_ausencia(nome)!r}"
        )

    # O texto nomeia a aba e os dois campos, e são estes os literais que a janela desenha.
    assert "Análise estatística" in SOB_PAR_DE_GRUPOS, SOB_PAR_DE_GRUPOS
    assert "Grupo A do contraste" in SOB_PAR_DE_GRUPOS
    assert "Grupo B do contraste" in SOB_PAR_DE_GRUPOS
    assert SOB_PAR_DE_GRUPOS.startswith(NAO_CALCULADA), (
        "o estado do meio deixou de citar o vocabulário dos três estados, e quem lê duas métricas "
        "seguidas perde a taxonomia"
    )

    # **O outro lado do par**, e é ele que impede o texto novo de cobrir tudo: uma métrica com
    # caminho e sem argumento continua em `NAO_CALCULADA`.
    sem_argumento = "frac_features_under_20"
    assert sem_argumento not in METRICAS_SOB_PAR_DE_GRUPOS, "âncora: o par causal colapsou"
    assert estado_de_ausencia(sem_argumento) == NAO_CALCULADA, (
        "uma métrica que não depende de argumento passou a mandar preencher dois campos"
    )


def test_u6_o_estado_de_ausencia_tem_os_tres_desfechos_com_a_tabela_injetada() -> None:
    """Três desfechos, e «dois» seria o piso errado: vi todos os que ele pode reportar.

    **A MESMA métrica nos dois primeiros ramos**, com `caminhos` injetado — é o padrão de
    `eixos_do_ponto`, e é a única forma de os exercitar sem esperar que uma rota mude. Com métricas
    diferentes em cada ramo, o teste não distinguiria *"o ramo funciona"* de *"esta métrica é
    assim"*.

    **A ORDEM é afirmada, e é ela o conteúdo:** uma métrica que estivesse nas duas classes — sob
    argumento **e** sem rota que a produza — tem de sair como *"sem caminho"*. Mandar preencher
    dois campos para obter um número que rota nenhuma calcula é pior que não dizer nada.
    """
    alvo = sorted(METRICAS_SOB_PAR_DE_GRUPOS)[0]
    outra = "frac_features_under_20"

    com_caminho = estado_de_ausencia(alvo, {alvo: "api", outra: "api"})
    sem_caminho = estado_de_ausencia(alvo, {alvo: "nenhum", outra: "api"})
    sem_argumento = estado_de_ausencia(outra, {alvo: "api", outra: "api"})

    assert com_caminho == SOB_PAR_DE_GRUPOS, com_caminho
    assert sem_caminho == SEM_CAMINHO, (
        "a ordem inverteu-se: uma métrica sob argumento que nenhuma rota calcula passou a mandar "
        "preencher dois campos"
    )
    assert sem_argumento == NAO_CALCULADA, sem_argumento
    assert len({com_caminho, sem_caminho, sem_argumento}) == 3, "dois dos três desfechos colapsaram"


@pytest.mark.parametrize(
    ("metrica", "valor", "esperado"),
    [
        # `percentagem`, `razao` e `contagem` — os três do vocabulário fechado.
        # **`qc_rsd_cv_median` e não `qc_rsd_median`, e a escolha é o conteúdo:** este teste mede
        # o NÚMERO sozinho, e a mediana leva população junto. Uma métrica de `populacao="nenhuma"`
        # isola o formatador do composto.
        ("qc_rsd_cv_median", 0.1799, "18,0%"),
        ("qc_dispersion_ratio", 1.8523, "1,852"),
        ("n_significant_fdr", 0, "0"),
        ("n_significant_fdr", 17, "17"),
        # O estado «calculada e indefinida» na célula, pelos DOIS caminhos que o produzem.
        ("qc_dispersion_ratio", None, SEM_VALOR_DEFINIDO),
        ("qc_dispersion_ratio", float("nan"), SEM_VALOR_DEFINIDO),
        # `inf` NÃO colapsa com eles: infinito é um VALOR, não uma ausência.
        # `qc_dispersion_ratio` é uma razão, e dispersão de amostras a zero produz infinito —
        # escrever *"indefinido"* ali apagaria a única coisa que o número está a dizer.
        ("qc_dispersion_ratio", float("inf"), "infinito"),
        ("qc_dispersion_ratio", float("-inf"), "infinito negativo"),
        # Fronteira: o JSON pode trazer texto onde o `dataclass` promete número.
        ("qc_dispersion_ratio", "meia dúzia", "meia dúzia (**valor não numérico**)"),
    ],
)
def test_d452_o_formatador_tem_seis_desfechos_e_os_seis_foram_vistos(
    metrica: str, valor: Any, esperado: str
) -> None:
    """«Dois» é o piso, não o número: este mecanismo classifica em SEIS desfechos.

    A pergunta é *quantos desfechos este mecanismo pode reportar, e eu vi todos?* — e a resposta
    aqui é seis: as três unidades do vocabulário fechado, a ausência de valor, o infinito e a
    fronteira não numérica.

    **Os dois últimos são os que ficariam inertes**, e por motivos opostos: o `inf` porque quase
    nunca acontece, e o não numérico porque é o ramo que dispara por **ausência de condição** —
    nada há a plantar nele, e é por isso que é o mais fácil de ficar sem exercício.

    **A âncora contra vacuidade:** os oito esperados abaixo são todos DISTINTOS entre si, então
    um formatador que devolvesse um texto constante reprovaria em sete dos oito casos.
    """
    relatorio = dict(_RELATORIO_PARA_FORMATAR, **{metrica: valor})
    assert valor_da_metrica(metrica, relatorio) == esperado


def test_d452_os_desfechos_do_formatador_sao_mutuamente_distintos() -> None:
    """A âncora que o teste acima não pode carregar, porque ele é parametrizado.

    **Sem ela, um formatador constante passaria em cada caso isolado** que por acaso esperasse o
    mesmo texto. Aqui os seis correm juntos e afirma-se que nenhum colapsa noutro.
    """
    saidas = [
        valor_da_metrica(
            "qc_dispersion_ratio", dict(_RELATORIO_PARA_FORMATAR, **{"qc_dispersion_ratio": v})
        )
        for v in (1.8523, None, float("nan"), float("inf"), float("-inf"), "meia dúzia")
    ]
    # `None` e `nan` colapsam de PROPÓSITO — os dois são «calculada e indefinida» para quem lê, e
    # o `None` daqui **é** o `nan` que a serialização produziu. Os outros quatro são distintos.
    assert saidas[1] == saidas[2], "o colapso declarado entre `None` e `nan` deixou de valer"
    distintos = {saidas[0], saidas[1], saidas[3], saidas[4], saidas[5]}
    assert len(distintos) == 5, f"dois desfechos do formatador colapsaram: {distintos}"


def test_a_concordancia_tem_prosa_para_os_quatro_estados_e_a_chave_crua_ao_lado() -> None:
    """O estado saía cru na tela; a prosa nasce em bijeção com o `Literal`."""
    from refino.interface.fluxo import PROSA_DA_CONCORDANCIA, prosa_da_concordancia
    from refino.persistence.repositories import EstadoDeConcordancia

    assert set(PROSA_DA_CONCORDANCIA) == set(get_args(EstadoDeConcordancia))
    for estado in get_args(EstadoDeConcordancia):
        texto = prosa_da_concordancia(estado)
        assert texto.endswith(f"({estado})"), texto
        assert PROSA_DA_CONCORDANCIA[estado] in texto
    assert "sem prosa" in prosa_da_concordancia("estado_novo"), "estado novo passou sem marca"


def test_o_numero_da_tela_sai_com_virgula_decimal() -> None:
    """O ponto decimal aparecia ao lado de percentuais com vírgula na mesma aba."""
    from refino.interface.fluxo import prosa_do_numero

    assert prosa_do_numero(0.17990825) == "0,1799"
    assert prosa_do_numero(None).startswith("indefinido")


# ---------------------------------------------------------------------------
# «Como o Palpite decide» — o exemplo medido é o do artefato que o mede
# ---------------------------------------------------------------------------

# O artefato é a F2 da rodada JH, e não o `palpite_nos_reais.txt`. Aquele foi calculado sobre
# vetores de antes das mudanças no SERRF e no LOESS. A F2 guarda os escores do Palpite nos quatro
# arquivos reais com os dois corrigidos, e a âncora J dela os conferiu bit a bit contra
# `mede_o_palpite`. A população de cada escore é a mediana, nas 200
# features sorteadas, do RSD dos QCs fora da amostra; o número de features e de QCs de cada arquivo
# vem da unidade que gravou os vetores.
ARTEFATO_DO_PALPITE: Final = REPO / "data" / "fixtures" / "medicoes" / "jh" / "f2_limiar.json"
UNIDADES_DO_PALPITE: Final = ARTEFATO_DO_PALPITE.parent / "f2_vetores" / "unidades"
MODOS_DO_EXEMPLO_DO_PALPITE: Final = ("pos_c18", "neg_hilic")
NOMES_NO_EXEMPLO_DO_PALPITE: Final = {"none": "sem correção", "loess": "LOESS", "serrf": "SERRF"}


def _por_cento(fracao: float) -> str:
    return f"{fracao * 100:.1f}".replace(".", ",") + "%"


def _milhar(numero: int) -> str:
    return f"{numero:,}".replace(",", ".")


def _divergencias_do_exemplo_do_palpite(texto: str) -> list[str]:
    """O que o texto afirma do exemplo e o artefato não sustenta; vazio quando os dois batem.

    A asserção é a IGUALDADE entre o texto e o artefato, e não o valor: o outro caminho como
    oráculo. E vai nas duas direções: cada número do artefato tem de estar na linha do modo,
    e cada percentagem do texto tem de vir do artefato ou do limiar. A segunda metade é a que acha
    um número caduco deixado ao lado dos novos.
    """
    artefato = json.loads(ARTEFATO_DO_PALPITE.read_text(encoding="utf-8"))
    divergencias: list[str] = []
    esperadas = {_por_cento(LIMIAR_DE_EMPATE_DO_PALPITE)}
    if f"{N_FEATURES_DO_PALPITE} features" not in texto:
        divergencias.append(f"o texto não diz {N_FEATURES_DO_PALPITE} features")
    ano, mes, dia = artefato["data"][:10].split("-")
    if f"medido em {dia}/{mes}/{ano}" not in texto:
        divergencias.append(f"o texto não diz a data do artefato, {dia}/{mes}/{ano}")
    for modo in MODOS_DO_EXEMPLO_DO_PALPITE:
        medido = artefato["por_modo"][modo]
        unidade = json.loads((UNIDADES_DO_PALPITE / f"real-{modo}.json").read_text("utf-8"))
        linhas = [linha for linha in texto.splitlines() if f"`{modo}`" in linha]
        if len(linhas) != 1:
            divergencias.append(f"{modo}: {len(linhas)} linhas do exemplo, e não uma")
            continue
        linha = linhas[0]
        populacao = f"{N_FEATURES_DO_PALPITE} das {_milhar(unidade['n_features'])} features"
        if populacao not in linha:
            divergencias.append(f"{modo}: a linha não diz {populacao!r}")
        if f"{unidade['n_qcs']} QCs" not in texto:
            divergencias.append(f"{modo}: o texto não diz {unidade['n_qcs']} QCs")
        for corretor, valor in medido["amostra"].items():
            par = f"{NOMES_NO_EXEMPLO_DO_PALPITE[corretor]} {_por_cento(valor)}"
            if par not in linha:
                divergencias.append(f"{modo}: a linha não diz {par!r}")
            esperadas.add(_por_cento(valor))
        margem = _por_cento(medido["margem"])
        esperadas.add(margem)
        if f"{margem} acima" not in linha:
            divergencias.append(f"{modo}: a linha não diz a margem, {margem}")
        proposta = f"propõe o {NOMES_NO_EXEMPLO_DO_PALPITE[medido['palpite']]}"
        if proposta not in linha:
            divergencias.append(f"{modo}: a linha não diz {proposta!r}")
        empate = bool(medido["empate_com_o_limiar_do_produto"])
        if ("avisa que é empate" in linha) != empate or ("sem empate" in linha) == empate:
            divergencias.append(f"{modo}: a linha diz o empate ao contrário do artefato ({empate})")
        if not medido["concorda_com_a_populacao"]:
            divergencias.append(f"{modo}: o artefato não concorda com a população, e o texto diz")
    percentagens = set(re.findall(r"\d+,\d%", texto))
    if percentagens != esperadas:
        divergencias.append(f"percentagens sem par no artefato: {sorted(percentagens ^ esperadas)}")
    return divergencias


def test_o_exemplo_do_palpite_e_o_do_artefato_que_o_mede() -> None:
    assert _divergencias_do_exemplo_do_palpite(COMO_O_PALPITE_DECIDE) == []


@pytest.mark.parametrize(
    ("de", "para"),
    [
        ("SERRF 14,9%", "SERRF 14,8%"),
        ("avisa que é empate", "decide"),
        ("2.039", "2.040"),
        ("7,3%", "7,3% (era 4,4%)"),
    ],
)
def test_a_conferencia_do_exemplo_acusa_o_que_o_artefato_nao_sustenta(de: str, para: str) -> None:
    """A planta cai sobre o texto real, e a conferência tem de a acusar."""
    assert COMO_O_PALPITE_DECIDE.count(de) == 1, f"a planta {de!r} não aterra numa instância só"
    assert _divergencias_do_exemplo_do_palpite(COMO_O_PALPITE_DECIDE.replace(de, para))


def test_u7_a_traducao_do_defeito_e_bijetiva_com_o_literal() -> None:
    """Domínio fechado, nas DUAS direções — e é o portão que impede o rótulo cru de voltar.

    Tipo novo sem tradução reprova aqui em vez de aparecer em inglês na tela; tradução órfã reprova
    também, que é o que impede a lista de envelhecer apontando para valor que saiu do `Literal`.
    """
    tipos = set(get_args(DefectType))
    traduzidos = set(ROTULOS_DE_DEFEITO)

    assert tipos, "âncora: `DefectType` não declara valor nenhum"
    assert tipos - traduzidos == set(), (
        f"tipos de defeito sem tradução: {sorted(tipos - traduzidos)}"
    )
    assert traduzidos - tipos == set(), f"traduções órfãs: {sorted(traduzidos - tipos)}"


def test_u7_a_justificativa_sem_formatador_avisa_e_nao_despeja() -> None:
    """O ramo `else`, que dispara por AUSÊNCIA de condição — não há o que plantar nele.

    É o ramo mais fácil de ficar inerte: um ponto novo cujo autor esqueça o formatador cairia
    nele, e a saída natural — `str(objeto)` — voltaria a despejar o objeto na tela, sem nada
    reprovar. Este teste é o que o mantém vivo.
    """
    objeto = {"rotulos_por_amostra": list(range(68)), "n_lotes": 3}
    saida = justificativa_em_prosa("ponto_que_nao_existe", objeto)

    assert "rotulos_por_amostra" not in saida, "o ramo sem formatador despejou a chave do objeto"
    assert "[0, 1, 2" not in saida, "o ramo sem formatador despejou a lista"
    assert "ponto_que_nao_existe" in saida, "o aviso não nomeia o ponto, então ninguém o encontra"
    assert "formatador" in saida, "o aviso não diz o que falta"


def test_u7_a_recomendacao_distingue_abstencao_de_nao_avaliavel() -> None:
    """Os três estados, e os dois primeiros levam à mesma ação e a contagens diferentes.

    A tela imprimia `confiança 0.0` nos dois. **A planta cai sobre dado em que as duas coisas
    diferem**: `avaliavel=False` com `abstem=True` contra `avaliavel=True` com `abstem=True` — o par
    que o esquema separa e a tela colapsava.
    """
    nao_avaliavel = prosa_da_recomendacao(avaliavel=False, abstem=True, confianca=0.0)
    abstem = prosa_da_recomendacao(avaliavel=True, abstem=True, confianca=0.0)
    normal = prosa_da_recomendacao(avaliavel=True, abstem=False, confianca=0.87)

    assert nao_avaliavel != abstem, (
        "os dois estados que o esquema separa produzem o MESMO texto: a distinção morreu na tela"
    )
    assert "não foi avaliável" in nao_avaliavel
    assert "abstém-se" in abstem
    assert "87%" in normal, f"a confiança não saiu em percentagem: {normal}"
    for saida in (nao_avaliavel, abstem, normal):
        assert "confiança 0.0" not in saida
