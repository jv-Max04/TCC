"""As referências dos métodos e o resumo metodológico.

`refino.interface.referencias.REFERENCIAS` carrega o texto que a janela mostra, e é a fonte das
obras. Este módulo confere a coerência entre obras e métodos e o que a tela mostra de cada um.

## O que este módulo NÃO alcança

A conferência de cada obra é procedência declarada, não verificada aqui.
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Any, get_args

import pytest

from refino.api.app import build_app
from refino.domain.labels import (
    ClassifierKind,
    ClusterLinkage,
    Correction,
    CorrelationMethod,
    Imputation,
    RankingFilter,
    RsdVariant,
    SampleNormalization,
    Scaling,
    Transformation,
)
from refino.domain.statistics import AnovaMethod, PostHocMethod
from refino.interface import referencias
from refino.interface.referencias import (
    CONFERENCIAS,
    LEGENDA_DA_CONFERENCIA,
    LINHA_DA_ETAPA,
    METODO_POR_CHAVE,
    METODOS,
    POR_CONFERIR,
    REFERENCIAS,
    SEM_RESUMO,
    ArquivoExportado,
    Referencia,
    chave_do_valor,
    resumo_da_execucao,
    resumo_da_exportacao,
    sem_marcacao,
    texto_das_referencias,
)
from refino.preprocessing.pipeline import CANONICAL_ORDER
from tests.conftest import REPO


def test_ha_obras_e_metodos_a_conferir() -> None:
    """A âncora das asserções de ausência abaixo — com o dicionário vazio, nenhuma diverge."""
    assert len(REFERENCIAS) >= 30, f"só {len(REFERENCIAS)} obras no código"
    assert len(METODOS) >= 60, f"só {len(METODOS)} métodos na lista"


def test_a_conferencia_e_vocabulario_fechado_em_runtime() -> None:
    """Um valor fora das duas classes levanta em vez de cair num estilo neutro."""
    assert set(CONFERENCIAS) == {"conteúdo conferido", "só metadados"}
    assert len(LEGENDA_DA_CONFERENCIA) == len(CONFERENCIAS)
    with pytest.raises(ValueError, match="conferência"):
        Referencia("R1", "X", "conferido")  # type: ignore[arg-type]


def test_welch_mann_whitney_e_shapiro_wilk_aparecem_como_so_metadados() -> None:
    """As três obras ao lado do método, com «só metadados» à vista."""
    lista = texto_das_referencias()
    for chave, rid in (
        ("univariada:welch", "R52"),
        ("univariada:mann_whitney", "R53"),
        ("normalidade:shapiro_wilk", "R54"),
    ):
        metodo = METODO_POR_CHAVE[chave]
        assert metodo.referencias == (rid,), (chave, metodo.referencias)
        assert REFERENCIAS[rid].conferencia == "só metadados"
        linha = next(linha for linha in lista.splitlines() if metodo.nome in linha)
        assert f"{rid} (só metadados)" in linha, linha


def test_toda_obra_citada_existe_e_nenhuma_obra_sobra() -> None:
    """As duas direções: método que cita obra ausente, e obra que nenhum método cita."""
    citadas = {rid for metodo in METODOS for rid in metodo.referencias}
    assert citadas, "nenhum método cita obra — a igualdade abaixo seria vácua"
    assert citadas - set(REFERENCIAS) == set(), "método cita obra que o código não carrega"
    assert set(REFERENCIAS) - citadas == set(), "obra carregada que nenhum método cita"


def test_as_chaves_dos_metodos_sao_unicas() -> None:
    chaves = [metodo.chave for metodo in METODOS]
    assert len(chaves) == len(set(chaves)), sorted(c for c in chaves if chaves.count(c) > 1)
    assert set(METODO_POR_CHAVE) == set(chaves)


# O vocabulário de cada eixo vem do DOMÍNIO e não de lista escrita aqui: um rótulo novo no `Literal`
# reprova até ganhar linha na lista — com obra, ou como «referência por conferir».
VOCABULARIO_POR_EIXO: dict[str, tuple[str, ...]] = {
    "correction": get_args(Correction),
    "rsd_variant": get_args(RsdVariant),
    "ranking_filter": get_args(RankingFilter),
    "imputation": get_args(Imputation),
    "sample_normalization": get_args(SampleNormalization),
    "transformation": get_args(Transformation),
    "scaling": get_args(Scaling),
    "anova": get_args(AnovaMethod),
    "post_hoc": get_args(PostHocMethod),
    "correlacao": get_args(CorrelationMethod),
    "classificador": get_args(ClassifierKind),
    "agrupamento": get_args(ClusterLinkage),
}


def test_todo_valor_do_vocabulario_tem_linha_na_lista() -> None:
    """A lista é a dos métodos IMPLEMENTADOS, e o domínio é quem diz quais são."""
    faltam = [
        f"{eixo}:{valor}"
        for eixo, valores in VOCABULARIO_POR_EIXO.items()
        for valor in valores
        if valor != "none" and chave_do_valor(eixo, valor) not in METODO_POR_CHAVE
    ]
    assert faltam == [], f"valores do domínio sem linha na lista: {faltam}"
    # A outra direção: linha de vocabulário cujo valor saiu do domínio.
    orfas = [
        metodo.chave
        for metodo in METODOS
        if metodo.chave.split(":")[0] in VOCABULARIO_POR_EIXO
        and metodo.chave.split(":")[1] not in VOCABULARIO_POR_EIXO[metodo.chave.split(":")[0]]
    ]
    assert orfas == [], f"linhas cujo valor não existe no domínio: {orfas}"


def test_o_segundo_passe_da_reducao_por_posto_e_o_mesmo_metodo() -> None:
    assert chave_do_valor("ranking_filter_2", "iqr") == chave_do_valor("ranking_filter", "iqr")


def test_metodo_sem_obra_aparece_por_conferir_e_sem_citacao() -> None:
    """Método sem linha na tabela aparece como «referência por conferir», sem citação."""
    sem_obra = [metodo for metodo in METODOS if not metodo.referencias]
    assert sem_obra, "âncora: nenhum método sem obra, e a asserção abaixo seria vácua"
    lista = texto_das_referencias()
    for metodo in sem_obra:
        linha = next(linha for linha in lista.splitlines() if metodo.nome in linha)
        assert POR_CONFERIR in linha, linha
        assert re.search(r"\bR\d+\b", linha) is None, f"citação num método sem obra: {linha}"


def test_toda_obra_aparece_na_lista_com_o_texto_e_a_conferencia() -> None:
    lista = sem_marcacao(texto_das_referencias())
    for referencia in REFERENCIAS.values():
        assert sem_marcacao(referencia.abnt) in lista, referencia.numero
        assert f"{referencia.numero} — {referencia.conferencia}" in lista, referencia.numero


# ---------------------------------------------------------------------------
# O resumo de uma execução
# ---------------------------------------------------------------------------


def _pedido(**mudancas: Any) -> dict[str, Any]:
    """O `data` de `POST /execucoes`, com as chaves que `PainelDePipeline.campos_base` envia.

    As chaves são as do contrato da rota, e `test_toda_chave_lida_pelo_resumo_existe_no_contrato`
    confere que as que o resumo lê existem nele; o pedido montado pela própria janela é exercitado
    em `tests/test_desktop_janela.py`.
    """
    base: dict[str, Any] = {
        "mode": "neg_c18",
        "estudo": "estudo-x",
        "correction": "loess",
        "n_batches": 1,
        "ranking_filter": "none",
        "ranking_filter_2": "none",
        "imputation": "one_fifth_min",
        "sample_normalization": "sum",
        "transformation": "log10",
        "scaling": "none",
        "rsd_variant": "outlier_removed",
        "paralelo": True,
        "analise_json": "{}",
        "seed": 42,
        "apresentacao": "exibida",
        "campos_sobrescritos": "[]",
    }
    base.update(mudancas)
    return base


def _resposta(**mudancas: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "run_id": 12,
        "n_features_saida": 2158,
        "pcas_da_correcao": {"corretor": "loess"},
    }
    base.update(mudancas)
    return base


def test_o_resumo_de_uma_execucao_loess_com_semente_42_cita_o_que_foi_escolhido() -> None:
    """O cenário da unidade: corretor, semente, normalização e transformação, cada um citado."""
    resumo = resumo_da_execucao(_pedido(), _resposta())

    for chave in (
        "correction:loess",
        "sample_normalization:sum",
        "transformation:log10",
        "imputation:one_fifth_min",
    ):
        assert chave in resumo.metodos, (chave, resumo.metodos)
        assert METODO_POR_CHAVE[chave].nome.casefold() in resumo.texto.casefold(), chave
    assert "Semente: 42." in resumo.texto
    assert "execução 12" in resumo.texto
    assert "neg_c18" in resumo.texto
    assert "2.158 features" in resumo.texto
    # Cada método citado tem a sua linha na lista, e as obras dele vão no fim do resumo.
    lista = texto_das_referencias()
    for chave in resumo.metodos:
        metodo = METODO_POR_CHAVE[chave]
        assert metodo.nome in lista, chave
        for rid in metodo.referencias:
            assert f"[{rid}] {sem_marcacao(REFERENCIAS[rid].abnt)}" in resumo.texto, rid


def test_o_resumo_nao_tem_marcacao_e_traz_as_ressalvas_das_obras_citadas() -> None:
    resumo = resumo_da_execucao(_pedido(), _resposta())
    assert "**" not in resumo.texto
    for ressalva in METODO_POR_CHAVE["correction:loess"].ressalvas:
        assert ressalva in resumo.texto, ressalva


def test_uma_execucao_sem_normalizacao_nao_menciona_normalizacao() -> None:
    """O par causal: o mesmo pedido, com e sem normalização — a palavra só aparece num deles."""
    com = resumo_da_execucao(_pedido(sample_normalization="sum"), _resposta())
    sem = resumo_da_execucao(_pedido(sample_normalization="none"), _resposta())
    assert "normaliza" in com.texto.lower(), "âncora: com normalização a palavra tem de aparecer"
    assert "normaliza" not in sem.texto.lower(), sem.texto
    assert not any(chave.startswith("sample_normalization:") for chave in sem.metodos)


def test_o_corretor_e_o_que_a_resposta_registrou_e_a_escolha_automatica_e_dita() -> None:
    """Com a escolha automática, quem corrigiu é o que a resposta diz, e não o campo do pedido."""
    resumo = resumo_da_execucao(
        _pedido(correction="none", auto="medida"), _resposta(pcas_da_correcao={"corretor": "serrf"})
    )
    assert "correction:serrf" in resumo.metodos
    assert "escolha_automatica" in resumo.metodos
    manual = resumo_da_execucao(_pedido(correction="none"), _resposta(pcas_da_correcao={}))
    assert "escolha_automatica" not in manual.metodos
    assert "Correção do sinal: nenhuma." in manual.texto


def test_os_lotes_e_os_cortes_do_pedido_aparecem() -> None:
    um = resumo_da_execucao(_pedido(n_batches=1), _resposta())
    dois = resumo_da_execucao(_pedido(n_batches=2, batch_cuts=json.dumps([35])), _resposta())
    iguais = resumo_da_execucao(_pedido(n_batches=3), _resposta())
    assert "um lote só" in um.texto
    assert "2 lotes, cortados antes das injeções 35" in dois.texto
    assert "3 lotes de tamanho igual pela ordem de injeção" in iguais.texto


def test_cada_etapa_do_pipeline_tem_a_sua_linha_no_resumo() -> None:
    """A ordem do resumo é `CANONICAL_ORDER`, e uma etapa nova sem linha reprova aqui."""
    assert set(LINHA_DA_ETAPA) == set(CANONICAL_ORDER)


def test_o_preprocessamento_completo_sai_na_ordem_do_pipeline() -> None:
    pedido = _pedido(
        rsd_cutoff=0.3,
        ranking_filter="iqr",
        ranking_keep_fraction=0.6,
        ranking_filter_2="mean",
        missing_threshold=0.5,
        sample_normalization="feature_reference",
        normalization_reference="1.0_100m/z",
        scaling="pareto",
    )
    texto = resumo_da_execucao(pedido, _resposta()).texto
    nomes = [
        "Corte por RSD dos QCs",
        METODO_POR_CHAVE["ranking_filter:iqr"].nome,
        METODO_POR_CHAVE["ranking_filter:mean"].nome,
        METODO_POR_CHAVE["filtro:variancia_quase_nula"].nome,
        METODO_POR_CHAVE["filtro:ausentes_por_grupo"].nome,
        METODO_POR_CHAVE["imputation:one_fifth_min"].nome,
        METODO_POR_CHAVE["sample_normalization:feature_reference"].nome,
        METODO_POR_CHAVE["transformation:log10"].nome,
        METODO_POR_CHAVE["scaling:pareto"].nome,
    ]
    posicoes = [texto.index(nome) for nome in nomes]
    assert posicoes == sorted(posicoes), list(zip(nomes, posicoes, strict=True))
    assert "mais de 30%" in texto
    assert "mantendo 60%" in texto
    assert "«1.0_100m/z»" in texto


def test_as_analises_vem_da_resposta() -> None:
    resposta = _resposta(
        volcano={
            "grupo_a": "ATLL",
            "grupo_b": "NoHTLV",
            "n_grupo_a": 13,
            "n_grupo_b": 16,
            "parametrico": False,
            "rota_da_normalidade": "shapiro",
            "shapiro_pvalue": 0.0001,
            "n_features_testadas": 2158,
            "correcao": "Benjamini-Hochberg (FDR)",
        },
        anova={
            "metodo": "kruskal_wallis",
            "post_hoc_metodo": "dunn",
            "grupos": ["a", "b", "c"],
            "correcao": "Benjamini-Hochberg (FDR)",
        },
        correlacao={"metodo": "spearman", "amostragem": "todas as 2158 features utilizáveis"},
    )
    resumo = resumo_da_execucao(_pedido(), resposta)
    for chave in (
        "univariada:mann_whitney",
        "normalidade:shapiro_wilk",
        "multiplos_testes:benjamini_hochberg",
        "anova:kruskal_wallis",
        "post_hoc:dunn",
        "correlacao:spearman",
    ):
        assert chave in resumo.metodos, chave
    assert "univariada:welch" not in resumo.metodos
    assert "ATLL (n = 13) e NoHTLV (n = 16)" in resumo.texto


def test_a_normalidade_por_grupo_entra_no_resumo_so_quando_a_resposta_a_traz() -> None:
    """A análise que a resposta traz é citada, e só ela."""
    frase = "por variável e por grupo de estudo, em 2 grupos"
    com = resumo_da_execucao(
        _pedido(), _resposta(normalidade={"grupos": [{"grupo": "a"}, {"grupo": "b"}]})
    )
    assert "normalidade:shapiro_wilk" in com.metodos
    assert frase in com.texto, com.texto
    sem = resumo_da_execucao(_pedido(), _resposta(normalidade=None))
    assert "normalidade:shapiro_wilk" not in sem.metodos
    assert frase not in sem.texto
    vazia = resumo_da_execucao(_pedido(), _resposta(normalidade={"grupos": []}))
    assert "por variável e por grupo de estudo" not in vazia.texto


# ---------------------------------------------------------------------------
# O resumo de uma exportação
# ---------------------------------------------------------------------------


def _pedido_de_artefato(metodo: str, **mudancas: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "mode": "neg_c18",
        "method": metodo,
        "n_batches": 2,
        "batch_cuts": json.dumps([35]),
        "seed": 42,
        "paralelo": True,
    }
    base.update(mudancas)
    return base


def test_o_resumo_do_lote_traz_o_corretor_a_semente_e_os_lotes_de_cada_arquivo() -> None:
    arquivos = [
        ArquivoExportado(f"neg_c18_corrigir_{metodo}.csv", "corrigir", _pedido_de_artefato(metodo))
        for metodo in ("none", "serrf", "loess")
    ]
    resumo = resumo_da_exportacao(arquivos)
    for arquivo in arquivos:
        assert arquivo.nome in resumo.texto
    assert "correction:serrf" in resumo.metodos
    assert "correction:loess" in resumo.metodos
    assert resumo.texto.count("Semente: 42.") == 3
    assert resumo.texto.count("2 lotes, cortados antes das injeções 35") == 2
    assert "Correção do sinal: nenhuma." in resumo.texto


def test_o_resumo_da_exportacao_para_o_metaboanalyst_diz_o_teto_e_a_variante() -> None:
    pedido = _pedido_de_artefato("loess", max_features=5000, rsd_variant="plain")
    resumo = resumo_da_exportacao([ArquivoExportado("a.csv", "exportar", pedido)])
    assert "5.000 features" in resumo.texto
    assert "rsd_variant:plain" in resumo.metodos


def test_o_resumo_da_matriz_processada_traz_o_preprocessamento() -> None:
    pedido = _pedido_de_artefato(
        "loess", imputation="one_fifth_min", sample_normalization="none", transformation="log2"
    )
    resumo = resumo_da_exportacao([ArquivoExportado("p.csv", "processar", pedido)])
    assert "transformation:log2" in resumo.metodos
    assert "normaliza" not in resumo.texto.lower()


def test_a_escolha_automatica_na_exportacao_le_o_aceite() -> None:
    pedido = _pedido_de_artefato("none", auto="medida")
    com_aceite = resumo_da_exportacao(
        [
            ArquivoExportado(
                "c.csv", "corrigir", pedido, aceite={"delegacao": {"correction": "loess"}}
            )
        ]
    )
    sem_aceite = resumo_da_exportacao([ArquivoExportado("c.csv", "corrigir", pedido)])
    assert "correction:loess" in com_aceite.metodos
    assert "correction:loess" not in sem_aceite.metodos
    assert "a resposta não disse qual" in sem_aceite.texto


def test_sem_execucao_nem_exportacao_a_frase_diz_quando_o_resumo_aparece() -> None:
    """A frase de antes da primeira Análise diz o gesto que faz o resumo aparecer, e onde.

    Ela mora na aba «Qualidade do resultado», ao lado das métricas que o mesmo desfecho enche, e o
    gesto é rodar a análise na aba dela. Os arquivos da Correção têm o resumo deles no Resultado de
    lá (`SEM_RESUMO_DA_ETAPA`).
    """
    assert "análise" in SEM_RESUMO
    assert "«Análise estatística»" in SEM_RESUMO
    for saiu in ("Executar", "Gerar os três", "Levar o resultado"):
        assert saiu not in SEM_RESUMO, f"a frase manda para {saiu!r}, que saiu da janela"
    with pytest.raises(ValueError, match="arquivo"):
        resumo_da_exportacao([])


# ---------------------------------------------------------------------------
# O denominador nulo do LOESS: a política que cada rota usa, conferida no código dela
# ---------------------------------------------------------------------------


def test_a_politica_do_denominador_nulo_de_cada_rota_e_a_da_assinatura_dela() -> None:
    """O resumo afirma a política que produziu o arquivo; ela tem de ser a que a ROTA usa.

    **A tabela é conferida contra a assinatura de cada rota**, e não escrita de memória: a janela
    não manda `null_denominator`, então o que vale é o padrão do campo na rota. E as prévias das
    etapas, que a frase do resumo cita como `r_faithful`, constroem o LOESS pelo padrão do
    `LoessCorrector` — conferido no corretor que elas constroem.
    """
    import inspect

    from refino.api import rotas
    from refino.correction.loess import LoessCorrector

    tabela = referencias.POLITICA_DO_DENOMINADOR_NULO_DA_ROTA
    assert set(tabela) == {"corrigir", "processar", "exportar"}, tabela
    for rota, politica in tabela.items():
        padrao = inspect.signature(getattr(rotas, rota)).parameters["null_denominator"].default
        assert padrao == politica, (rota, padrao, politica)
    previa = rotas._corretor_da_spec("loess", paralelo=False)
    assert isinstance(previa, LoessCorrector)
    assert previa.null_denominator == "r_faithful", "a prévia não usa a política que o resumo cita"
    # A diferença que o resumo declara existe: as duas políticas não são a mesma.
    assert tabela["exportar"] != previa.null_denominator


def test_o_resumo_do_arquivo_com_loess_nomeia_a_politica_do_denominador_nulo() -> None:
    """No MetaboAnalyst, `uncorrected_feature` e a diferença da prévia; no laboratório,
    `r_faithful`. Sem LOESS não há denominador a nomear."""
    metabo = resumo_da_exportacao(
        [ArquivoExportado("m.csv", "exportar", _pedido_de_exportacao("loess"))]
    )
    assert "`uncorrected_feature`" in metabo.texto
    assert "difere" in metabo.texto and "`r_faithful`" in metabo.texto
    laboratorio = resumo_da_exportacao(
        [ArquivoExportado("c.csv", "corrigir", _pedido_de_artefato("loess"))]
    )
    assert "`r_faithful`" in laboratorio.texto
    assert "`uncorrected_feature`" not in laboratorio.texto
    sem_loess = resumo_da_exportacao(
        [ArquivoExportado("s.csv", "exportar", _pedido_de_exportacao("serrf"))]
    )
    assert "Denominador nulo" not in sem_loess.texto, "âncora: só o LOESS tem denominador nulo"
    # O pedido que manda a política vence o padrão da rota.
    explicito = resumo_da_exportacao(
        [
            ArquivoExportado(
                "x.csv", "exportar", _pedido_de_exportacao("loess", null_denominator="r_faithful")
            )
        ]
    )
    assert "`r_faithful`" in explicito.texto and "`uncorrected_feature`" not in explicito.texto


def test_a_frase_de_cada_politica_nomeia_a_feature_de_mediana_zero() -> None:
    """As duas frases dizem o que sai quando a mediana da feature também é zero.

    Com a mediana zero o divisor é `0/0`, a guarda não o vê, e a feature sai com células vazias
    (38 features do `neg_c18`): a frase de `uncorrected_feature` não pode dizer sem exceção que
    *«a feature cuja curva ajustada vale zero sai sem correção»*. O que as duas afirmam é o que
    `tests/unit/test_loess_uncorrected.py` mede sobre matriz construída à mão.
    """
    for politica in ("r_faithful", "uncorrected_feature"):
        texto = resumo_da_exportacao(
            [
                ArquivoExportado(
                    "x.csv", "exportar", _pedido_de_exportacao("loess", null_denominator=politica)
                )
            ]
        ).texto
        assert "mediana" in texto and "vazia" in texto, (politica, texto)
    uncorrected = resumo_da_exportacao(
        [ArquivoExportado("m.csv", "exportar", _pedido_de_exportacao("loess"))]
    ).texto
    assert "a feature cuja curva ajustada vale zero sai sem correção" not in uncorrected, (
        "a frase voltou a afirmar sem exceção o que é falso para as features de mediana zero"
    )


def _pedido_de_exportacao(metodo: str, **mudancas: Any) -> dict[str, Any]:
    """O pedido do formato MetaboAnalyst: o de qualquer artefato, mais o teto e a variante."""
    return _pedido_de_artefato(metodo, max_features=5000, rsd_variant="plain", **mudancas)


def test_a_exportacao_com_os_eixos_traz_o_preprocessamento_e_sem_eles_nao() -> None:
    """O MetaboAnalyst da Normalização lista as etapas; o da Correção, sem eixo nenhum, não as
    menciona — nem o filtro de variância quase nula, que a exportação só corre com algum eixo."""
    com_eixos = _pedido_de_exportacao(
        "none", imputation="one_fifth_min", sample_normalization="sum", scaling="auto"
    )
    normalizada = resumo_da_exportacao([ArquivoExportado("n.csv", "exportar", com_eixos)])
    assert "Pré-processamento" in normalizada.texto
    assert "scaling:auto" in normalizada.metodos
    assert "filtro:variancia_quase_nula" in normalizada.metodos
    corrigida = resumo_da_exportacao(
        [ArquivoExportado("c.csv", "exportar", _pedido_de_exportacao("none"))]
    )
    assert "Pré-processamento" not in corrigida.texto
    assert "filtro:variancia_quase_nula" not in corrigida.metodos, (
        "o resumo citou um filtro que a exportação sem eixos não aplica"
    )


def test_o_que_o_arquivo_e_vem_de_quem_o_pediu_e_cada_aba_tem_a_frase_dela() -> None:
    """O Resultado diz o que o arquivo é — a alternativa à recusa, a entrada sem correção —, e a
    frase de antes do primeiro arquivo é a do gesto daquela aba."""
    arquivo = ArquivoExportado(
        "a.csv", "exportar", _pedido_de_exportacao("none"), o_que_e="a ALTERNATIVA à recusa"
    )
    assert "a.csv — a ALTERNATIVA à recusa." in resumo_da_exportacao([arquivo]).texto
    frases = referencias.SEM_RESUMO_DA_ETAPA
    assert set(frases) == {1, 2}
    assert "matriz corrigida" in frases[1] and "Gerar os três" in frases[1]
    assert "matriz normalizada" in frases[2] and "Gerar os três" not in frases[2]


# ---------------------------------------------------------------------------
# As chaves que o resumo lê existem no contrato — a falha que não levanta
# ---------------------------------------------------------------------------


def _chaves_lidas_pelo_resumo() -> set[str]:
    """As chaves literais que `interface/referencias.py` lê de pedido e de resposta, por AST.

    `.get("x")` e `["x"]` com `x` constante. Uma chave mal escrita devolve `None`, e a linha do
    resumo some sem nada acusar — é a direção que o portão de contrato da janela fecha para os
    módulos dela, aqui para este.
    """
    fonte = (REPO / "src" / "refino" / "interface" / "referencias.py").read_text(encoding="utf-8")
    chaves: set[str] = set()
    for no in ast.walk(ast.parse(fonte)):
        if (
            isinstance(no, ast.Call)
            and isinstance(no.func, ast.Attribute)
            and no.func.attr == "get"
            and no.args
            and isinstance(no.args[0], ast.Constant)
            and isinstance(no.args[0].value, str)
        ):
            chaves.add(no.args[0].value)
        elif (
            isinstance(no, ast.Subscript)
            and isinstance(no.slice, ast.Constant)
            and isinstance(no.slice.value, str)
        ):
            chaves.add(no.slice.value)
    return chaves


def _propriedades_do_esquema() -> set[str]:
    esquema = dict(build_app().openapi())
    nomes: set[str] = set()
    for componente in esquema["components"]["schemas"].values():
        nomes |= set(componente.get("properties", {}))
    for operacoes in esquema["paths"].values():
        for operacao in operacoes.values():
            for parametro in operacao.get("parameters", []):
                nomes.add(parametro["name"])
    return nomes


def test_toda_chave_lida_pelo_resumo_existe_no_contrato() -> None:
    lidas = _chaves_lidas_pelo_resumo()
    assert len(lidas) >= 20, f"a extração achou {len(lidas)} chaves — o extrator mediu o vazio"
    fora = sorted(lidas - _propriedades_do_esquema())
    assert fora == [], f"o resumo lê chaves que nenhum esquema da API declara: {fora}"


def test_o_extrator_de_chaves_acusa_uma_chave_inventada() -> None:
    """A planta do portão acima: uma chave que nenhum esquema tem fica de fora."""
    assert "corretor_aplicado_inventado" not in _propriedades_do_esquema()
    assert "pcas_da_correcao" in _propriedades_do_esquema(), "âncora: a raiz real está no esquema"


def test_o_modulo_nao_conhece_qt() -> None:
    """O resumo e a lista correm sem janela: os testes acima os leem sem `QApplication`.

    Por AST e não por `grep`, para a docstring do módulo poder nomear o painel que o mostra.
    """
    arvore = ast.parse(Path(referencias.__file__).read_text(encoding="utf-8"))
    importados = {no.module or "" for no in ast.walk(arvore) if isinstance(no, ast.ImportFrom)} | {
        alias.name for no in ast.walk(arvore) if isinstance(no, ast.Import) for alias in no.names
    }
    assert importados, "âncora: o módulo importa alguma coisa, e a extração não achou nada"
    assert not any(nome.startswith(("PySide6", "refino.desktop")) for nome in importados), sorted(
        importados
    )


def test_a_correcao_de_multiplos_testes_citada_e_a_que_a_resposta_registrou() -> None:
    """Par causal: o mesmo volcano com Benjamini-Hochberg cita R51; com outra correção, ou sem o
    campo, a citação não é herdada e o texto diz o que a resposta trouxe."""
    volcano = {
        "grupo_a": "A",
        "grupo_b": "B",
        "n_grupo_a": 3,
        "n_grupo_b": 3,
        "parametrico": True,
        "rota_da_normalidade": "forced",
        "n_features_testadas": 10,
    }
    bh = resumo_da_execucao(
        _pedido(), _resposta(volcano={**volcano, "correcao": "Benjamini-Hochberg (FDR)"})
    )
    outra = resumo_da_execucao(_pedido(), _resposta(volcano={**volcano, "correcao": "Bonferroni"}))
    sem = resumo_da_execucao(_pedido(), _resposta(volcano=volcano))
    assert "multiplos_testes:benjamini_hochberg" in bh.metodos
    assert "multiplos_testes:benjamini_hochberg" not in outra.metodos
    assert "q-valores por «Bonferroni»" in outra.texto
    assert "sem correção de múltiplos testes registrada na resposta" in sem.texto
    assert "escolhido no pedido" in bh.texto


def _identificadores_doi(texto: str) -> set[str]:
    """Os DOIs citados num texto, pela forma `10.<registrante>/<sufixo>` sem o ponto final."""
    return {doi.rstrip(".,;") for doi in re.findall(r"10\.\d{4,9}/\S+", texto)}


def test_as_referencias_soltas_das_metricas_citam_a_obra_da_lista() -> None:
    """As três legendas da aba «Qualidade» citam Broadhurst et al. por literal em
    `interface/fluxo.py`; a lista as tem como R6. Duas cópias da mesma citação divergem na primeira
    edição, e é a igualdade dos DOIs, e não um valor escrito aqui, que as amarra — o outro caminho
    como oráculo.
    """
    from refino.interface.fluxo import (
        REFERENCIA_DA_REGUA_DE_RSD,
        REFERENCIA_DO_CORTE_DE_20,
        REFERENCIA_DO_CORTE_DE_30,
    )

    da_lista = _identificadores_doi(REFERENCIAS["R6"].abnt)
    assert len(da_lista) == 1, f"âncora: R6 tem de citar um DOI só, e cita {da_lista}"
    for legenda in (
        REFERENCIA_DA_REGUA_DE_RSD,
        REFERENCIA_DO_CORTE_DE_20,
        REFERENCIA_DO_CORTE_DE_30,
    ):
        assert _identificadores_doi(legenda) == da_lista, (legenda[:80], da_lista)
    # A planta: um dígito trocado no DOI da legenda deixa de casar com a obra da lista.
    plantada = REFERENCIA_DA_REGUA_DE_RSD.replace("1367-3", "1367-4")
    assert plantada != REFERENCIA_DA_REGUA_DE_RSD, "a planta não aterrou"
    assert _identificadores_doi(plantada) != da_lista
