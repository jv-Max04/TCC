"""O estado do passo a passo, sem tela, e as etapas independentes com entrada escolhida."""

from __future__ import annotations

import json
from typing import Any

import pytest

from refino.desktop.etapas import (
    CAMPO_SEM_PROPOSTA,
    CAMPOS_DO_PAR_DE_DECISAO,
    ETAPA_DO_CAMPO,
    ETAPAS,
    CampoSemEtapaError,
    Entrada,
    EntradaDesconhecidaError,
    EntradaIndisponivelError,
    EstadoDasEtapas,
    Fotografia,
    _proposta_da_maquina,
    campos_da_rota,
    etapa_a_refazer,
    fotografia,
    fotografia_da_etapa,
)

BASE: dict[str, Any] = {
    "mode": "neg_c18",
    "correction": "loess",
    "n_batches": 1,
    "rsd_variant": "outlier_removed",
    "seed": 42,
    "imputation": "one_fifth_min",
    "sample_normalization": "none",
    "transformation": "log10",
    "scaling": "pareto",
    "estudo": "e",
    "apresentacao": "exibida",
    "analise_json": "{}",
    "campos_sobrescritos": "[]",
    "paralelo": True,
}
MATRIZ = {"matriz@1": "abc"}

# **A tela mais rica que a janela sabe mandar**, com todo eixo fora do neutro: um campo da tela que
# vazasse para um pedido que devia levá-lo neutro só aparece se o valor da tela for outro.
TELA: dict[str, Any] = {
    "mode": "neg_c18",
    "orientation": "features-in-rows",
    "correction": "serrf",
    "n_batches": 2,
    "batch_cuts": "[30]",
    "rsd_variant": "outlier_removed",
    "seed": 7,
    "auto": "politica",
    "imputation": "one_fifth_min",
    "sample_normalization": "sample_reference",
    "normalization_reference": "QC1",
    "transformation": "log10",
    "scaling": "pareto",
    "rsd_cutoff": 30.0,
    "missing_threshold": 0.5,
    "ranking_filter": "iqr",
    "ranking_keep_fraction": 0.5,
    "ranking_filter_2": "sd",
    "ranking_keep_fraction_2": 0.8,
    "estudo": "e",
    "apresentacao": "exibida",
    "analise_json": "{}",
    "group_a": "ATLL",
    "group_b": "NoHTLV",
    "campos_sobrescritos": '["correction"]',
    "paralelo": True,
}
# **A proposta como a Correção a monta no clique**: a spec da TELA, com os eixos da máquina no lugar
# — o corretor do Palpite, e os lotes e os cortes do diagnóstico. O pré-processamento dela é o da
# tela, e não o neutro: é isso que a Análise sobre a corrigida tem de reancorar.
RAZAO_DA_MAQUINA = "palpite medido: loess"
LOTES_DA_MAQUINA: dict[str, Any] = {"n_batches": 2, "batch_cuts": [30]}
TELA["proposta_json"] = _proposta_da_maquina(
    TELA, "loess", RAZAO_DA_MAQUINA, "modelo", outros=LOTES_DA_MAQUINA
)
OUTRA_MATRIZ = {"matriz@1": "xyz"}

# Os sete eixos do pré-processamento, com o valor neutro de cada um, e os que o neutro é a falta.
PREPROCESSAMENTO_EM_NONE = (
    "imputation",
    "sample_normalization",
    "transformation",
    "scaling",
    "ranking_filter",
    "ranking_filter_2",
)
PREPROCESSAMENTO_AUSENTE = (
    "rsd_cutoff",
    "missing_threshold",
    "ranking_keep_fraction",
    "ranking_keep_fraction_2",
    "normalization_reference",
)


def _fotos(campos: dict[str, Any], identidade: dict[str, str] = MATRIZ) -> dict[int, Any]:
    return {k: fotografia(campos, k, identidade) for k in (1, 2, 3)}


def _feitas_as_tres() -> EstadoDasEtapas:
    estado = EstadoDasEtapas()
    fotos = _fotos(BASE)
    for etapa in (1, 2, 3):
        estado.disparou(etapa, fotos[etapa])
        estado.concluiu(etapa)
    return estado


def _estados(estado: EstadoDasEtapas, campos: dict[str, Any], **kw: Any) -> tuple[str, ...]:
    fotos = _fotos(campos, **kw)
    return tuple(estado.estado(k, fotos[k]) for k in (1, 2, 3))


def _faz(
    estado: EstadoDasEtapas,
    etapa: int,
    entrada: Entrada,
    tela: dict[str, Any] = TELA,
    identidade: dict[str, str] = MATRIZ,
    correcao: str | None = None,
) -> dict[str, Any]:
    """Dispara e conclui uma etapa como a janela o fará: o pedido sai da entrada, a foto da tela."""
    campos = estado.campos_da_entrada(etapa, entrada, tela, identidade["matriz@1"])
    foto = fotografia_da_etapa(tela, etapa, identidade, entrada)
    estado.disparou(etapa, foto, campos=campos, entrada=entrada)
    estado.concluiu(etapa, correcao=correcao)
    return campos


def _corrige(estado: EstadoDasEtapas, tela: dict[str, Any] = TELA, **kw: Any) -> dict[str, Any]:
    return _faz(estado, 1, "original", tela, correcao=kw.pop("correcao", "loess"), **kw)


def _atuais(
    estado: EstadoDasEtapas, tela: dict[str, Any] = TELA, identidade: dict[str, str] = MATRIZ
) -> dict[int, Fotografia]:
    """A foto que a tela daria agora para cada etapa, com a entrada que cada uma tem escolhida."""
    matriz = identidade["matriz@1"]
    return {
        k: fotografia_da_etapa(tela, k, identidade, estado.entrada(k, matriz).escolhida)
        for k in ETAPAS
    }


# ---------------------------------------------------------------------------
# A fotografia do pedido montado inteiro a partir da tela
# ---------------------------------------------------------------------------


def test_mudar_a_imputacao_desatualiza_so_as_etapas_2_e_3() -> None:
    estado = _feitas_as_tres()
    assert _estados(estado, BASE) == ("em_dia", "em_dia", "em_dia")
    outra = {**BASE, "imputation": "median"}
    assert _estados(estado, outra) == ("em_dia", "desatualizada", "desatualizada")


@pytest.mark.parametrize("campo", ["correction", "n_batches", "rsd_variant", "mode", "auto"])
def test_mudar_um_campo_da_correcao_desatualiza_as_tres(campo: str) -> None:
    estado = _feitas_as_tres()
    assert _estados(estado, {**BASE, campo: "outro"}) == ("desatualizada",) * 3


def test_paralelo_nao_desatualiza_nada() -> None:
    estado = _feitas_as_tres()
    assert _estados(estado, {**BASE, "paralelo": False}) == ("em_dia",) * 3


def test_o_par_de_decisao_nao_desatualiza_nada() -> None:
    """O par vem da Correção feita e não é parâmetro de resultado: mudá-lo na tela não pede refazer.

    Par causal: o mesmo dicionário com a imputação mudada desatualiza, então a fotografia lê-o.
    """
    estado = _feitas_as_tres()
    outro_par = {**BASE, "campos_sobrescritos": '["imputation"]', "proposta_json": "{}"}
    assert _estados(estado, outro_par) == ("em_dia",) * 3
    assert _estados(estado, {**outro_par, "imputation": "median"})[1] == "desatualizada"


def test_trocar_os_arquivos_desatualiza_as_tres_e_o_fator_so_a_partir_da_2() -> None:
    estado = _feitas_as_tres()
    assert _estados(estado, BASE, identidade={"matriz@1": "outra"}) == ("desatualizada",) * 3
    com_fator = {**MATRIZ, "fatores@2": "f1"}
    assert _estados(estado, BASE, identidade=com_fator) == (
        "em_dia",
        "desatualizada",
        "desatualizada",
    )


def test_campo_sem_etapa_levanta_em_vez_de_cair_fora_da_fotografia() -> None:
    with pytest.raises(CampoSemEtapaError, match="campo_novo"):
        fotografia({**BASE, "campo_novo": 1}, 3, MATRIZ)
    with pytest.raises(CampoSemEtapaError, match="campo_novo"):
        fotografia_da_etapa({**BASE, "campo_novo": 1}, 3, MATRIZ, "original")


def test_a_rota_da_etapa_recebe_so_os_campos_das_etapas_anteriores_e_paralelo() -> None:
    um = campos_da_rota(BASE, 1)
    assert set(um) == {"mode", "correction", "n_batches", "rsd_variant", "seed", "paralelo"}
    dois = campos_da_rota(BASE, 2)
    assert {"imputation", "scaling"} <= set(dois)
    assert not {"estudo", "analise_json"} & set(dois)


def test_o_par_de_decisao_so_viaja_para_a_execucao() -> None:
    """As prévias não declaram o par, e o FastAPI descarta campo não declarado; só o `/execucoes`
    o grava. Par causal: a etapa 3 recebe os dois, as duas prévias nenhum.

    `sem_proposta` está no par e exclui a proposta — a rota recusa os dois juntos —, então a
    tela daqui os leva os três só para conferir o filtro, que não olha combinação."""
    tela = {**TELA, CAMPO_SEM_PROPOSTA: True}
    assert not CAMPOS_DO_PAR_DE_DECISAO & set(campos_da_rota(tela, 1))
    assert not CAMPOS_DO_PAR_DE_DECISAO & set(campos_da_rota(tela, 2))
    assert set(campos_da_rota(tela, 3)) >= CAMPOS_DO_PAR_DE_DECISAO


def test_toda_chave_da_tabela_tem_etapa_valida() -> None:
    assert ETAPA_DO_CAMPO, "ancora: a tabela nao pode estar vazia"
    assert set(ETAPA_DO_CAMPO.values()) <= {None, 1, 2, 3}


def test_o_par_de_decisao_saiu_da_etapa_3() -> None:
    # `sem_proposta` está no par: é a mesma decisão da Correção feita.
    assert {"campos_sobrescritos", "proposta_json", "sem_proposta"} == CAMPOS_DO_PAR_DE_DECISAO
    assert not CAMPOS_DO_PAR_DE_DECISAO & set(ETAPA_DO_CAMPO)


def test_em_dia_e_da_etapa_feita_mesmo_com_ela_a_ser_refeita() -> None:
    """O frescor das FIGURAS: refazer a etapa 1 com os mesmos parametros nao invalida a figura dela.

    Par causal: com outro corretor na tela, a mesma etapa em curso ja nao esta em dia.
    """
    estado = _feitas_as_tres()
    fotos = _fotos(BASE)
    estado.disparou(1, fotos[1])
    assert estado.estado(1, fotos[1]) == "em_curso"
    assert estado.em_dia(1, fotos[1])
    assert not estado.em_dia(1, _fotos({**BASE, "correction": "outro"})[1])


# ---------------------------------------------------------------------------
# O disparo sem a trava em cadeia
# ---------------------------------------------------------------------------


def test_sem_nada_feito_as_tres_disparam_e_com_uma_em_curso_nenhuma() -> None:
    """A dependência entre etapas é só a da entrada: a Normalização dispara sem a Correção.

    Par causal: a mesma etapa 2 que dispara sem nada feito desliga as três enquanto corre, e a
    falha religa as três e a deixa «não feita».
    """
    estado = EstadoDasEtapas()
    assert [estado.pode_disparar(k) for k in ETAPAS] == [True, True, True]
    foto = fotografia_da_etapa(TELA, 2, MATRIZ, "original")
    estado.disparou(2, foto)
    assert estado.estado(2, foto) == "em_curso"
    assert [estado.pode_disparar(k) for k in ETAPAS] == [False, False, False]
    estado.falhou(2)
    assert estado.estado(2, foto) == "nao_feita", "a etapa que falhou ficou como feita"
    assert [estado.pode_disparar(k) for k in ETAPAS] == [True, True, True]


def test_uma_etapa_fora_do_vocabulario_levanta() -> None:
    with pytest.raises(ValueError, match="4"):
        EstadoDasEtapas().pode_disparar(4)


# ---------------------------------------------------------------------------
# As entradas oferecidas, e a padrão
# ---------------------------------------------------------------------------


def test_sem_correcao_a_normalizacao_so_oferece_a_original() -> None:
    entrada = EstadoDasEtapas().entrada(2, "abc")
    assert entrada.disponiveis == ("original",)
    assert entrada.padrao == entrada.escolhida == "original"
    assert entrada.mais_tratada is None and entrada.motivo_da_volta is None


def test_com_a_correcao_feita_a_normalizacao_oferece_as_duas_e_a_padrao_e_a_corrigida() -> None:
    estado = EstadoDasEtapas()
    _corrige(estado)
    assert estado.estado(1, _atuais(estado)[1]) == "em_dia"
    entrada = estado.entrada(2, "abc")
    assert entrada.disponiveis == ("original", "corrigida")
    assert entrada.padrao == entrada.escolhida == "corrigida"


def test_a_correcao_em_curso_nao_e_entrada_ate_concluir() -> None:
    estado = EstadoDasEtapas()
    estado.disparou(
        1,
        fotografia_da_etapa(TELA, 1, MATRIZ, "original"),
        campos=estado.campos_da_entrada(1, "original", TELA, "abc"),
    )
    assert estado.entrada(2, "abc").disponiveis == ("original",)
    estado.concluiu(1, correcao="loess")
    assert estado.entrada(2, "abc").disponiveis == ("original", "corrigida")


def test_com_correcao_e_normalizacao_a_analise_oferece_as_tres_e_a_padrao_e_a_normalizada() -> None:
    estado = EstadoDasEtapas()
    _corrige(estado)
    assert estado.entrada(3, "abc").disponiveis == ("original", "corrigida")
    _faz(estado, 2, "corrigida")
    entrada = estado.entrada(3, "abc")
    assert entrada.disponiveis == ("original", "corrigida", "normalizada")
    assert entrada.padrao == entrada.escolhida == "normalizada"


def test_a_normalizada_e_nomeada_pelo_que_entrou_nela() -> None:
    """Par causal: a mesma normalização sobre a original e sobre a corrigida dá nomes diferentes."""
    sobre_a_original = EstadoDasEtapas()
    _faz(sobre_a_original, 2, "original")
    assert sobre_a_original.entrada(3, "abc").disponiveis == ("original", "normalizada")
    assert sobre_a_original.nome_da_entrada("normalizada") == "normalizada (sem correção)"

    sobre_a_corrigida = EstadoDasEtapas()
    _corrige(sobre_a_corrigida)
    _faz(sobre_a_corrigida, 2, "corrigida")
    assert sobre_a_corrigida.nome_da_entrada("normalizada") == "normalizada (corrigida — loess)"
    assert sobre_a_corrigida.nome_da_entrada("corrigida") == "corrigida — loess"
    assert sobre_a_corrigida.nome_da_entrada("original") == "original"


def test_depois_de_trocar_o_par_a_normalizacao_so_oferece_a_original() -> None:
    """A Correção feita sobre o par anterior não é entrada da matriz carregada agora.

    Par causal: com a mesma matriz, a mesma Correção é oferecida.
    """
    estado = EstadoDasEtapas()
    _corrige(estado)
    assert estado.entrada(2, "abc").disponiveis == ("original", "corrigida")
    assert estado.entrada(2, "xyz").disponiveis == ("original",)
    assert estado.entrada(3, "xyz").disponiveis == ("original",)
    with pytest.raises(EntradaIndisponivelError, match="outra matriz"):
        estado.campos_da_entrada(2, "corrigida", TELA, "xyz")


# ---------------------------------------------------------------------------
# Os campos que o pedido leva
# ---------------------------------------------------------------------------


def test_a_analise_sobre_a_original_leva_a_correcao_e_os_sete_eixos_neutros() -> None:
    campos = EstadoDasEtapas().campos_da_entrada(3, "original", TELA, "abc")
    assert campos["correction"] == "none"
    assert campos["n_batches"] == 1
    assert not {"auto", "batch_cuts"} & set(campos), sorted(campos)
    for eixo in PREPROCESSAMENTO_EM_NONE:
        assert campos[eixo] == "none", eixo
    assert not set(PREPROCESSAMENTO_AUSENTE) & set(campos), sorted(campos)
    # Os comuns vêm da tela na original; os da própria Análise também.
    assert (campos["mode"], campos["rsd_variant"], campos["seed"]) == (
        "neg_c18",
        "outlier_removed",
        7,
    )
    assert campos["orientation"] == "features-in-rows"
    assert (campos["group_a"], campos["estudo"], campos["paralelo"]) == ("ATLL", "e", True)
    # Nenhum eixo de pré-processamento normaliza, e a combinação proibida não se forma.
    assert campos["imputation"] == "none" and campos["sample_normalization"] == "none"


def test_a_normalizacao_sobre_a_original_leva_a_correcao_neutra_e_os_eixos_da_tela() -> None:
    campos = EstadoDasEtapas().campos_da_entrada(2, "original", TELA, "abc")
    assert (campos["correction"], campos["n_batches"]) == ("none", 1)
    assert not {"auto", "batch_cuts"} & set(campos)
    assert (campos["imputation"], campos["ranking_filter"]) == ("one_fifth_min", "iqr")
    assert campos["normalization_reference"] == "QC1"
    assert not {"estudo", "group_a", "analise_json"} & set(campos), "a Análise vazou na etapa 2"


def test_a_correcao_com_auto_que_devolveu_loess_entra_como_loess() -> None:
    """A entrada «corrigida» leva o corretor efetivo, e não a delegação, nem o corretor da tela.

    Mandar `auto` de novo mediria o Palpite outra vez, e o resultado poderia ser outro corretor.
    Mesma regra para a semente e os lotes: mudá-los na tela depois de corrigir não muda o pedido.
    """
    estado = EstadoDasEtapas()
    enviados = _corrige(estado, correcao="loess")
    assert (enviados["correction"], enviados["auto"]) == ("serrf", "politica")
    feita = estado.feita(1)
    assert feita is not None
    assert (feita.correcao, feita.matriz, feita.entrada) == ("loess", "abc", "original")
    assert feita.campos == enviados

    outra_tela = {**TELA, "correction": "none", "seed": 99, "n_batches": 3, "batch_cuts": None}
    for etapa in (2, 3):
        campos = estado.campos_da_entrada(etapa, "corrigida", outra_tela, "abc")
        assert campos["correction"] == "loess", etapa
        assert "auto" not in campos, etapa
        assert (campos["seed"], campos["n_batches"], campos["batch_cuts"]) == (7, 2, "[30]")


def test_a_analise_sobre_a_corrigida_leva_o_pre_processamento_neutro() -> None:
    estado = EstadoDasEtapas()
    _corrige(estado)
    campos = estado.campos_da_entrada(3, "corrigida", TELA, "abc")
    for eixo in PREPROCESSAMENTO_EM_NONE:
        assert campos[eixo] == "none", eixo
    assert not set(PREPROCESSAMENTO_AUSENTE) & set(campos)


def test_a_analise_sobre_a_normalizada_leva_os_campos_da_normalizacao_feita() -> None:
    """Os campos vêm da etapa FEITA: mudar a imputação na tela depois não muda o pedido."""
    estado = EstadoDasEtapas()
    _corrige(estado)
    _faz(estado, 2, "corrigida")
    outra_tela = {**TELA, "imputation": "median", "scaling": "auto", "correction": "none"}
    campos = estado.campos_da_entrada(3, "normalizada", outra_tela, "abc")
    assert (campos["imputation"], campos["scaling"], campos["correction"]) == (
        "one_fifth_min",
        "pareto",
        "loess",
    )
    assert campos["normalization_reference"] == "QC1"


def test_o_par_de_decisao_vem_da_correcao_feita_e_vai_vazio_na_original() -> None:
    """Na original o par não vai vazio: vai `sem_proposta`, porque a correção neutra não foi
    proposta pela máquina. **Par causal:** a corrigida, de uma Correção que
    levou proposta, não recebe o campo — a base dela é outra constante, e com ele dentro a
    declaração vazaria para uma Correção que seguiu as sugestões."""
    estado = EstadoDasEtapas()
    _corrige(estado)
    tela_sem_par = {**TELA, "campos_sobrescritos": "[]", "proposta_json": "{}"}

    corrigida = estado.campos_da_entrada(3, "corrigida", tela_sem_par, "abc")
    assert corrigida["campos_sobrescritos"] == '["correction"]'
    assert json.loads(corrigida["proposta_json"])["rationale"] == RAZAO_DA_MAQUINA

    _faz(estado, 2, "corrigida", tela_sem_par)
    normalizada = estado.campos_da_entrada(3, "normalizada", tela_sem_par, "abc")
    assert normalizada["campos_sobrescritos"] == '["correction"]', "a normalizada perdeu o par"
    assert json.loads(normalizada["proposta_json"])["rationale"] == RAZAO_DA_MAQUINA
    assert "sem_proposta" not in corrigida and "sem_proposta" not in normalizada, (
        "a declaração de que não houve proposta vazou para uma Correção que levou proposta"
    )

    original = estado.campos_da_entrada(3, "original", TELA, "abc")
    assert original["campos_sobrescritos"] == "[]"
    assert "proposta_json" not in original
    assert original["sem_proposta"] is True, "a Análise sobre a original não declarou sem proposta"


def test_d707_a_correcao_sem_proposta_leva_a_declaracao_as_entradas_derivadas() -> None:
    """A Correção feita sem proposta da máquina é gravada assim também pelas entradas dela.

    A Análise grava a decisão da Correção FEITA (`_herdado`), e não a que a tela mostra agora:
    uma tela que depois seguisse as sugestões não pode transformar a decisão da corrigida em
    proposta seguida. **Par causal:** a mesma cadeia com a tela que levou proposta não declara.
    """
    tela_sem = {**BASE, CAMPO_SEM_PROPOSTA: True}
    estado = EstadoDasEtapas()
    _corrige(estado, tela_sem)
    _faz(estado, 2, "corrigida", tela_sem)
    for entrada in ("corrigida", "normalizada"):
        pedido = estado.campos_da_entrada(3, entrada, TELA, "abc")
        assert pedido.get(CAMPO_SEM_PROPOSTA) is True, f"a {entrada} perdeu a declaração"
        assert "proposta_json" not in pedido, f"a {entrada} ganhou a proposta da tela de agora"

    com_proposta = EstadoDasEtapas()
    _corrige(com_proposta)
    assert CAMPO_SEM_PROPOSTA not in com_proposta.campos_da_entrada(3, "corrigida", TELA, "abc")


def _spec_da_proposta(campos: dict[str, Any]) -> dict[str, Any]:
    spec: dict[str, Any] = json.loads(json.loads(campos["proposta_json"])["spec_json"])
    return spec


@pytest.mark.parametrize("entrada", ["corrigida", "normalizada"])
def test_a_proposta_da_correcao_e_reancorada_no_pedido_da_analise(entrada: Entrada) -> None:
    """A linha da máquina é a spec do humano com os eixos dela no lugar, e o par é comparado por
    texto: a proposta que a Análise leva tem a spec do PEDIDO da Análise, com os eixos que a máquina
    propôs na Correção feita.

    **O oráculo é o outro caminho**: `_proposta_da_maquina` sobre o pedido final, com os mesmos
    eixos da máquina, é o que a execução gravaria se tivesse montado a proposta ela mesma. E a
    planta cai onde as duas diferem: a proposta da Correção leva o pré-processamento da tela, que a
    Análise sobre a corrigida manda neutro e a sobre a normalizada manda com outra imputação.
    """
    estado = EstadoDasEtapas()
    _corrige(estado)
    tela_da_normalizacao = {**TELA, "imputation": "median", "scaling": "auto"}
    _faz(estado, 2, "corrigida", tela_da_normalizacao)

    pedido = estado.campos_da_entrada(3, entrada, TELA, "abc")
    da_correcao = _spec_da_proposta(TELA)
    reancorada = _spec_da_proposta(pedido)
    assert da_correcao["imputation"] == "one_fifth_min", "âncora: a proposta leva a tela"
    assert pedido["imputation"] != da_correcao["imputation"], "âncora: as duas não diferem"
    assert reancorada["imputation"] == pedido["imputation"]
    for eixo in ("correction", "n_batches", "batch_cuts", "seed", "rsd_variant"):
        assert reancorada[eixo] == da_correcao[eixo], eixo
    assert (reancorada["correction"], reancorada["n_batches"]) == ("loess", 2)

    oraculo = _proposta_da_maquina(
        pedido, "loess", RAZAO_DA_MAQUINA, "modelo", outros=LOTES_DA_MAQUINA
    )
    assert pedido["proposta_json"] == oraculo


@pytest.mark.parametrize("entrada", ["corrigida", "normalizada"])
def test_a_analise_com_a_proposta_reancorada_fica_em_dia_logo_depois_de_feita(
    entrada: Entrada,
) -> None:
    """A reancoragem faz a proposta da Análise diferir, de propósito, da guardada na Correção; a
    comparação que decide se a entrada foi refeita não pode lê-la como mudança de entrada.

    **O defeito que este teste prende**: com a proposta da máquina na tela, a Análise sobre a
    corrigida saía «desatualizada», pelo motivo da entrada refeita, no instante em que terminava —
    e sem a proposta ficava em dia. A âncora é as duas propostas diferirem; sem isso o teste
    passaria por coincidência.
    """
    estado = EstadoDasEtapas()
    _corrige(estado)
    if entrada == "normalizada":
        # Com a tela da Correção, a proposta reancorada sobre a normalizada sai igual à guardada; a
        # Normalização com outra imputação é o que as faz diferir.
        _faz(estado, 2, "corrigida", {**TELA, "imputation": "median", "scaling": "auto"})
    pedido = _faz(estado, 3, entrada)
    assert pedido["proposta_json"] != TELA["proposta_json"], "âncora: as propostas não diferem"
    atual = _atuais(estado)[3]
    assert estado.motivos(3, atual) == set()
    assert estado.estado(3, atual) == "em_dia"


def test_a_ausencia_diz_por_que_falta_a_entrada_com_o_nome_da_aba() -> None:
    """A frase de uma entrada que não existe, com «a normalização» — o nome da aba — e não o da
    rota. Par causal: com a etapa feita, a ausência some; com outra matriz, volta com outro motivo.
    """
    estado = EstadoDasEtapas()
    assert estado.ausencia("corrigida", "abc") == "a correção não foi feita"
    assert estado.ausencia("normalizada", "abc") == "a normalização não foi feita"
    assert estado.ausencia("original", "abc") is None
    _corrige(estado)
    _faz(estado, 2, "corrigida")
    assert estado.ausencia("corrigida", "abc") is None
    assert estado.ausencia("normalizada", "abc") is None
    assert estado.ausencia("corrigida", "xyz") == (
        "a correção feita é de outra matriz — o par de arquivos ou o modo mudou depois"
    )


def test_o_nome_da_entrada_usa_o_nome_que_a_tela_da_ao_corretor() -> None:
    estado = EstadoDasEtapas()
    _corrige(estado)
    _faz(estado, 2, "corrigida")
    corretores = {"loess": "LOESS"}
    assert estado.nome_da_entrada("corrigida", corretores=corretores) == "corrigida — LOESS"
    assert estado.nome_da_entrada("normalizada", corretores=corretores) == (
        "normalizada (corrigida — LOESS)"
    )
    assert estado.nome_da_entrada("original", corretores=corretores) == "original"


def test_so_a_normalizada_com_imputacao_passou_pela_imputacao() -> None:
    """O aviso dos zeros da Análise: a original e a corrigida levam o pré-processamento neutro, e a
    normalizada passou pela imputação quando a Normalização feita imputou.

    Par causal: a mesma Normalização, com e sem imputação.
    """
    estado = EstadoDasEtapas()
    _corrige(estado)
    _faz(estado, 2, "corrigida")
    assert [estado.passou_pela_imputacao(e) for e in ("original", "corrigida", "normalizada")] == [
        False,
        False,
        True,
    ]
    sem_imputar = {**TELA, "imputation": "none", "sample_normalization": "none"}
    _faz(estado, 2, "corrigida", sem_imputar)
    assert not estado.passou_pela_imputacao("normalizada")


def test_a_cadeia_feita_aba_a_aba_deixa_cada_etapa_entrada_da_seguinte() -> None:
    """A Correção sobre a original, a Normalização sobre a corrigida e a Análise sobre a
    normalizada — cada uma disparada da aba dela, com a entrada mais tratada que existe, que é o
    padrão —: as três ficam feitas, e cada uma serve de entrada à seguinte.

    Antes, esta cadeia saía de um clique só, no «Executar pipeline», que a deixava feita por
    `concluiu_tudo`. O botão saiu com a aba «Executar e exportar»,
    e a cadeia passou a ser feita aba a aba; o que este teste afirma é o mesmo desfecho, pelo
    caminho que existe.

    Par causal do frescor: mudar o corretor da tela desatualiza só a Correção, e as dependentes
    passam a apontar para ela.
    """
    estado = EstadoDasEtapas()
    _corrige(estado, correcao="loess")
    for etapa in (2, 3):
        _faz(estado, etapa, estado.entrada(etapa, "abc").escolhida)
    atuais = _atuais(estado)
    assert [estado.estado(k, atuais[k]) for k in ETAPAS] == ["em_dia"] * 3
    assert [estado.entrada(k, "abc").escolhida for k in ETAPAS] == list(
        ("original", "corrigida", "normalizada")
    )
    assert estado.nome_da_entrada("normalizada") == "normalizada (corrigida — loess)"
    analise = estado.feita(3)
    assert analise is not None and analise.campos is not None
    assert analise.campos["imputation"] == "one_fifth_min"

    atuais = _atuais(estado, {**TELA, "correction": "none"})
    assert [estado.estado(k, atuais[k]) for k in ETAPAS] == ["desatualizada", "em_dia", "em_dia"]
    assert [estado.etapa_a_refazer(k, atuais) for k in ETAPAS] == [1, 1, 1]


def test_pedir_a_corrigida_sem_correcao_feita_levanta() -> None:
    """Não cai na original em silêncio, que mandaria `correction="none"` sem ninguém saber."""
    estado = EstadoDasEtapas()
    with pytest.raises(EntradaIndisponivelError, match="corrigida"):
        estado.campos_da_entrada(2, "corrigida", TELA, "abc")
    with pytest.raises(EntradaIndisponivelError, match="normalizada"):
        estado.campos_da_entrada(3, "normalizada", TELA, "abc")
    with pytest.raises(EntradaIndisponivelError, match="corrigida"):
        estado.escolhe_entrada(3, "corrigida", "abc")


@pytest.mark.parametrize(("etapa", "entrada"), [(2, "normalizada"), (1, "corrigida"), (3, "bruta")])
def test_uma_entrada_fora_do_vocabulario_da_etapa_levanta(etapa: int, entrada: str) -> None:
    estado = EstadoDasEtapas()
    with pytest.raises(EntradaDesconhecidaError, match=entrada):
        estado.campos_da_entrada(etapa, entrada, TELA, "abc")  # type: ignore[arg-type]
    with pytest.raises(EntradaDesconhecidaError, match=entrada):
        estado.escolhe_entrada(etapa, entrada, "abc")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# A escolha explícita
# ---------------------------------------------------------------------------


def test_a_escolha_explicita_persiste_quando_surge_uma_entrada_mais_tratada() -> None:
    """Par causal: sem escolha, a mesma Normalização concluída muda a seleção para a normalizada."""
    estado = EstadoDasEtapas()
    _corrige(estado)
    estado.escolhe_entrada(3, "corrigida", "abc")
    _faz(estado, 2, "corrigida")
    entrada = estado.entrada(3, "abc")
    assert entrada.escolhida == "corrigida"
    assert entrada.disponiveis == ("original", "corrigida", "normalizada")
    assert entrada.mais_tratada == "normalizada"

    sem_escolha = EstadoDasEtapas()
    _corrige(sem_escolha)
    _faz(sem_escolha, 2, "corrigida")
    assert sem_escolha.entrada(3, "abc").escolhida == "normalizada"
    assert sem_escolha.entrada(3, "abc").mais_tratada is None


def test_a_entrada_com_que_a_etapa_rodou_continua_escolhida_quando_surge_outra() -> None:
    """Rodar com a entrada que a tela mostrava é escolhê-la: a Correção concluída depois não muda a
    seleção da Análise feita sobre a original, e por isso não a desatualiza.

    Par causal: a Análise nunca rodada passa à corrigida, que é a padrão.
    """
    estado = EstadoDasEtapas()
    _faz(estado, 3, "original")
    _corrige(estado)
    entrada = estado.entrada(3, "abc")
    assert (entrada.escolhida, entrada.padrao, entrada.mais_tratada) == (
        "original",
        "corrigida",
        "corrigida",
    )
    assert estado.estado(3, _atuais(estado)[3]) == "em_dia"

    nunca_rodada = EstadoDasEtapas()
    _corrige(nunca_rodada)
    assert nunca_rodada.entrada(3, "abc").escolhida == "corrigida"


def test_a_escolha_que_deixa_de_existir_volta_a_padrao_e_diz_por_que() -> None:
    estado = EstadoDasEtapas()
    _corrige(estado)
    estado.escolhe_entrada(2, "corrigida", "abc")
    entrada = estado.entrada(2, "xyz")
    assert entrada.escolhida == entrada.padrao == "original"
    assert entrada.motivo_da_volta is not None
    assert "corrigida" in entrada.motivo_da_volta and "outra matriz" in entrada.motivo_da_volta
    assert estado.entrada(2, "abc").motivo_da_volta is None, "a escolha valida tambem avisou"


def test_rodar_na_entrada_de_recuo_substitui_a_escolha_que_deixou_de_existir() -> None:
    """A escolha que deixou de existir não volta quando a entrada reaparece, se a etapa rodou na
    de recuo — achado da revisão de código.

    Rodar com a entrada que a tela mostrava é escolhê-la. Antes, a escolha velha («corrigida»)
    ficava guardada por baixo do recuo e voltava com a Correção nova, e a Normalização feita na
    original ficava «desatualizada» sem que a pessoa tivesse mudado nada.
    """
    estado = EstadoDasEtapas()
    _corrige(estado)
    estado.escolhe_entrada(2, "corrigida", "abc")
    assert estado.entrada(2, "xyz").escolhida == "original", "âncora: a matriz nova recua"
    _faz(estado, 2, "original", identidade=OUTRA_MATRIZ)
    _corrige(estado, identidade=OUTRA_MATRIZ)

    entrada = estado.entrada(2, "xyz")
    assert entrada.escolhida == "original", entrada
    atual = fotografia_da_etapa(TELA, 2, OUTRA_MATRIZ, entrada.escolhida)
    assert estado.estado(2, atual) == "em_dia"


# ---------------------------------------------------------------------------
# Os dois motivos de desatualização, e o aviso da dependente
# ---------------------------------------------------------------------------


def test_mudar_o_corretor_desatualiza_a_correcao_e_avisa_a_normalizacao() -> None:
    """Motivo (a) na Correção; a Normalização feita sobre ela fica em dia e aponta para corrigir.

    Par causal: antes de mudar, nenhuma das duas tem aviso.
    """
    estado = EstadoDasEtapas()
    _corrige(estado)
    _faz(estado, 2, "corrigida")
    atuais = _atuais(estado)
    assert [estado.estado(k, atuais[k]) for k in (1, 2)] == ["em_dia", "em_dia"]
    assert [estado.etapa_a_refazer(k, atuais) for k in (1, 2)] == [None, None]

    atuais = _atuais(estado, {**TELA, "correction": "none"})
    assert estado.estado(1, atuais[1]) == "desatualizada"
    assert estado.motivos(1, atuais[1]) == {"campos_mudaram"}
    assert estado.estado(2, atuais[2]) == "em_dia"
    assert estado.motivos(2, atuais[2]) == frozenset()
    assert estado.etapa_a_refazer(2, atuais) == 1
    assert estado.etapa_a_refazer(1, atuais) == 1


def test_refazer_a_correcao_com_outro_corretor_desatualiza_a_normalizacao_pela_entrada() -> None:
    """Motivo (b), e só ele: os campos da própria Normalização não mudaram."""
    estado = EstadoDasEtapas()
    _corrige(estado)
    _faz(estado, 2, "corrigida")
    outra_tela = {**TELA, "correction": "none", "auto": None}
    _corrige(estado, outra_tela, correcao="none")
    atuais = _atuais(estado, outra_tela)
    assert estado.estado(1, atuais[1]) == "em_dia"
    assert estado.estado(2, atuais[2]) == "desatualizada"
    assert estado.motivos(2, atuais[2]) == {"entrada_refeita"}
    assert estado.etapa_a_refazer(2, atuais) == 2


def test_refazer_a_correcao_com_os_mesmos_campos_nao_desatualiza_a_normalizacao() -> None:
    estado = EstadoDasEtapas()
    _corrige(estado)
    _faz(estado, 2, "corrigida")
    _corrige(estado)
    atuais = _atuais(estado)
    assert estado.estado(2, atuais[2]) == "em_dia"


def test_a_correcao_refeita_sem_guardar_os_campos_nao_e_entrada_e_desatualiza_a_dependente() -> (
    None
):
    """Sem os campos, ou sem o corretor efetivo, não há de onde herdar: a Correção sai das entradas,
    e a Normalização feita sobre a anterior pede para refazer, que é o lado seguro.

    Par causal: a mesma Correção refeita guardando os dois volta a ser entrada, e a dependente fica
    em dia.
    """
    estado = EstadoDasEtapas()
    _corrige(estado)
    _faz(estado, 2, "corrigida")
    foto = fotografia_da_etapa(TELA, 1, MATRIZ, "original")
    for campos in (None, estado.campos_da_entrada(1, "original", TELA, "abc")):
        estado.disparou(1, foto, campos=campos)
        estado.concluiu(1)
        assert estado.entrada(2, "abc").disponiveis == ("original",)
        # A seleção da Normalização também volta à original, e isso é o outro motivo.
        assert estado.motivos(2, _atuais(estado)[2]) == {"entrada_refeita", "campos_mudaram"}
        with pytest.raises(EntradaIndisponivelError, match="não guardou"):
            estado.nome_da_entrada("corrigida")
    _corrige(estado)
    assert estado.entrada(2, "abc").disponiveis == ("original", "corrigida")
    assert estado.motivos(2, _atuais(estado)[2]) == frozenset()


def test_o_nome_de_uma_entrada_que_nao_existe_levanta() -> None:
    with pytest.raises(EntradaIndisponivelError, match="corrigida"):
        EstadoDasEtapas().nome_da_entrada("corrigida")
    with pytest.raises(EntradaDesconhecidaError, match="bruta"):
        EstadoDasEtapas().nome_da_entrada("bruta")


def test_na_cadeia_refazer_a_correcao_faz_a_analise_apontar_para_a_normalizacao() -> None:
    """A Análise continua em dia — a entrada dela é a Normalização que existe —, e o aviso dela
    aponta para a etapa mais a montante fora de dia, que é a Normalização e não a própria Análise.
    """
    estado = EstadoDasEtapas()
    _corrige(estado)
    _faz(estado, 2, "corrigida")
    _faz(estado, 3, "normalizada")
    outra_tela = {**TELA, "correction": "none", "auto": None}
    _corrige(estado, outra_tela, correcao="none")
    atuais = _atuais(estado, outra_tela)
    assert [estado.estado(k, atuais[k]) for k in ETAPAS] == ["em_dia", "desatualizada", "em_dia"]
    assert estado.etapa_a_refazer(3, atuais) == 2


def test_na_cadeia_a_correcao_mudada_e_nao_refeita_e_o_alvo_do_aviso_da_analise() -> None:
    estado = EstadoDasEtapas()
    _corrige(estado)
    _faz(estado, 2, "corrigida")
    _faz(estado, 3, "normalizada")
    atuais = _atuais(estado, {**TELA, "correction": "none"})
    assert [estado.estado(k, atuais[k]) for k in ETAPAS] == ["desatualizada", "em_dia", "em_dia"]
    assert estado.etapa_a_refazer(3, atuais) == 1


def test_a_analise_sobre_a_original_nao_depende_da_correcao() -> None:
    """Par causal da cadeia: a mesma Correção mudada não avisa a Análise que não a usou."""
    estado = EstadoDasEtapas()
    _corrige(estado)
    estado.escolhe_entrada(3, "original", "abc")
    _faz(estado, 3, "original")
    atuais = _atuais(estado, {**TELA, "correction": "none"})
    assert estado.estado(1, atuais[1]) == "desatualizada"
    assert estado.estado(3, atuais[3]) == "em_dia"
    assert estado.etapa_a_refazer(3, atuais) is None


def test_com_a_original_mudar_a_variante_desatualiza_a_normalizacao() -> None:
    """Na original os comuns vêm da tela, e entram na fotografia da etapa que os consome.

    Par causal: sobre a corrigida, a mesma variante mudada na tela vem da Correção feita, e a
    Normalização continua em dia — quem fica fora de dia é a Correção.
    """
    sobre_a_original = EstadoDasEtapas()
    _faz(sobre_a_original, 2, "original")
    outra_tela = {**TELA, "rsd_variant": "plain"}
    atuais = _atuais(sobre_a_original, outra_tela)
    assert sobre_a_original.estado(2, atuais[2]) == "desatualizada"
    assert sobre_a_original.motivos(2, atuais[2]) == {"campos_mudaram"}

    sobre_a_corrigida = EstadoDasEtapas()
    _corrige(sobre_a_corrigida)
    _faz(sobre_a_corrigida, 2, "corrigida")
    atuais = _atuais(sobre_a_corrigida, outra_tela)
    assert sobre_a_corrigida.estado(2, atuais[2]) == "em_dia"
    assert sobre_a_corrigida.estado(1, atuais[1]) == "desatualizada"


def test_trocar_a_entrada_escolhida_depois_de_feita_desatualiza_a_etapa() -> None:
    """A entrada é parâmetro da etapa: a mesma Normalização com outra entrada é outro pedido."""
    estado = EstadoDasEtapas()
    _corrige(estado)
    _faz(estado, 2, "corrigida")
    estado.escolhe_entrada(2, "original", "abc")
    atuais = _atuais(estado)
    assert estado.estado(2, atuais[2]) == "desatualizada"
    assert estado.motivos(2, atuais[2]) == {"campos_mudaram"}


def test_trocar_o_par_desatualiza_as_feitas_pela_propria_etapa() -> None:
    estado = EstadoDasEtapas()
    _corrige(estado)
    _faz(estado, 2, "corrigida")
    atuais = _atuais(estado, identidade=OUTRA_MATRIZ)
    assert [estado.estado(k, atuais[k]) for k in (1, 2)] == ["desatualizada", "desatualizada"]
    assert estado.motivos(2, atuais[2]) == {"campos_mudaram"}


# ---------------------------------------------------------------------------
# A etapa a refazer, pela cadeia das entradas
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("etapa", "em_dia", "esperada"),
    [
        (2, {1: True, 2: False}, 2),
        (2, {1: False, 2: False}, 1),
        (3, {1: True, 2: False, 3: False}, 2),
        (1, {}, 1),
        # Todas em dia: não há etapa anterior a refazer, e a da figura é a resposta.
        (2, {1: True, 2: True}, 2),
        # Uma etapa POSTERIOR fora de dia não conta: a figura da 1 não depende da 2.
        (1, {1: True, 2: False}, 1),
    ],
)
def test_d552_a_etapa_a_refazer_e_a_primeira_fora_de_dia_ate_a_da_figura(
    etapa: int, em_dia: dict[int, bool], esperada: int
) -> None:
    assert etapa_a_refazer(etapa, em_dia) == esperada


@pytest.mark.parametrize(
    ("entradas", "esperada"),
    [
        # A Análise sobre a original não depende de nenhuma: a 1 e a 2 fora de dia não contam.
        ({3: "original"}, 3),
        # Sobre a corrigida, a cadeia salta a Normalização.
        ({3: "corrigida"}, 1),
        # Sobre a normalizada feita sobre a original, a Correção fica fora da cadeia.
        ({2: "original", 3: "normalizada"}, 2),
        ({2: "corrigida", 3: "normalizada"}, 1),
    ],
)
def test_a_etapa_a_refazer_segue_a_entrada_usada(
    entradas: dict[int, Entrada], esperada: int
) -> None:
    assert etapa_a_refazer(3, {1: False, 2: False, 3: False}, entradas) == esperada


# ---------------------------------------------------------------------------
# As etapas que o resultado da Análise em curso reproduz — quem escreve as figuras de quem
# ---------------------------------------------------------------------------


def _analisa(estado: EstadoDasEtapas, entrada: Entrada) -> None:
    """Dispara a Análise sobre `entrada`, como a janela o faz, e a deixa em curso."""
    campos = estado.campos_da_entrada(3, entrada, TELA, "abc")
    estado.disparou(
        3, fotografia_da_etapa(TELA, 3, MATRIZ, entrada), campos=campos, entrada=entrada
    )


@pytest.mark.parametrize(
    ("normaliza_sobre", "analisa_sobre", "esperadas"),
    [
        (None, "original", set()),
        (None, "corrigida", {1}),
        ("corrigida", "normalizada", {1, 2}),
        ("original", "normalizada", {2}),
    ],
)
def test_a_analise_reproduz_as_etapas_da_cadeia_da_entrada_dela(
    normaliza_sobre: Entrada | None, analisa_sobre: Entrada, esperadas: set[int]
) -> None:
    """Sobre a original, nenhuma: não há correção a mostrar. Sobre a corrigida, a Correção. Sobre a
    normalizada, a Normalização e, se ela partiu da corrigida, a Correção. **O par de decisão não
    conta**: a proposta viaja reancorada no pedido da Análise sobre a corrigida, e mesmo assim a
    Análise reproduz a Correção — `TELA` tem o pré-processamento fora do neutro, que é o que a
    reancoragem troca."""
    estado = EstadoDasEtapas()
    _corrige(estado)
    if normaliza_sobre is not None:
        _faz(estado, 2, normaliza_sobre)
    assert estado.etapas_reproduzidas(3) == frozenset(), "âncora: sem Análise em curso, nenhuma"
    _analisa(estado, analisa_sobre)
    assert estado.etapas_reproduzidas(3) == esperadas


def test_a_etapa_refeita_depois_do_disparo_deixa_de_ser_reproduzida() -> None:
    """Refeita com outro corretor depois do disparo, a Correção guardada já é outra, e o que a
    Análise traz descreve a anterior. **Par causal:** refeita com os mesmos campos, ela continua
    reproduzida. E na cadeia, a Normalização refeita corta a Correção também — a Análise não passa
    pela que existe agora."""
    for tela, correcao, esperadas in (
        (TELA, "loess", {1}),
        ({**TELA, "correction": "none", "auto": None}, "none", set()),
    ):
        estado = EstadoDasEtapas()
        _corrige(estado)
        _analisa(estado, "corrigida")
        _corrige(estado, tela, correcao=correcao)
        assert estado.etapas_reproduzidas(3) == esperadas, correcao

    estado = EstadoDasEtapas()
    _corrige(estado)
    _faz(estado, 2, "corrigida")
    _analisa(estado, "normalizada")
    assert estado.etapas_reproduzidas(3) == {1, 2}, "âncora: a cadeia inteira reproduzida"
    _faz(estado, 2, "corrigida", {**TELA, "imputation": "half_min"})
    assert estado.etapas_reproduzidas(3) == frozenset()


# O envelope da proposta da máquina, byte a byte, para a TELA acima. A comparação do par de decisão
# é por igualdade de texto, então qualquer mudança de ordem, de escape ou de chave muda o desfecho
# gravado. O segundo caso cobre os cortes que chegam como texto JSON e a origem manual.
_ENVELOPE_DA_TELA = (
    '{"spec_json": "{\\"batch_cuts\\": [30], \\"correction\\": \\"loess\\", \\"imputation\\": '
    '\\"one_fifth_min\\", \\"missing_threshold\\": 0.5, \\"n_batches\\": 2, '
    '\\"normalization_factors\\": null, \\"normalization_reference\\": \\"QC1\\", '
    '\\"ranking_filter\\": \\"iqr\\", \\"ranking_filter_2\\": \\"sd\\", '
    '\\"ranking_keep_fraction\\": 0.5, \\"ranking_keep_fraction_2\\": 0.8, \\"rsd_cutoff\\": 30.0, '
    '\\"rsd_variant\\": \\"outlier_removed\\", \\"sample_normalization\\": \\"sample_reference\\", '
    '\\"scaling\\": \\"pareto\\", \\"seed\\": 7, \\"transformation\\": \\"log10\\"}", '
    '"rationale": "palpite medido: loess", "origem": "modelo"}'
)
_ENVELOPE_MANUAL = (
    '{"spec_json": "{\\"batch_cuts\\": [30], \\"correction\\": \\"none\\", \\"imputation\\": '
    '\\"one_fifth_min\\", \\"missing_threshold\\": 0.5, \\"n_batches\\": 2, '
    '\\"normalization_factors\\": null, \\"normalization_reference\\": \\"QC1\\", '
    '\\"ranking_filter\\": \\"iqr\\", \\"ranking_filter_2\\": \\"sd\\", '
    '\\"ranking_keep_fraction\\": 0.5, \\"ranking_keep_fraction_2\\": 0.8, \\"rsd_cutoff\\": 30.0, '
    '\\"rsd_variant\\": \\"outlier_removed\\", \\"sample_normalization\\": \\"sample_reference\\", '
    '\\"scaling\\": \\"pareto\\", \\"seed\\": 7, \\"transformation\\": \\"log10\\"}", '
    '"rationale": "sem motivo", "origem": "manual"}'
)


def test_o_envelope_da_proposta_da_maquina_e_byte_a_byte_o_esperado() -> None:
    """O envelope que a janela grava como linha da máquina não muda de forma."""
    assert TELA["proposta_json"] == _ENVELOPE_DA_TELA
    assert _proposta_da_maquina(TELA, "none", "sem motivo", "manual") == _ENVELOPE_MANUAL
