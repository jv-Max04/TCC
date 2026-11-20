"""A justificativa, a sobrescrita, e a paridade dos eixos.

**Divisão de trabalho com `tests/integration/test_par_de_concordancia.py`, e ela é
deliberada.** Lá a asserção é sobre o que ficou **gravado** num banco SQLite real, no nível do
repositório. Aqui o alvo é a **rota**: que ela recuse o que tem de recusar na fronteira, que a
justificativa chegue não vazia ao cliente, e que a enumeração dos eixos tenha paridade
mecânica com o que a rota expõe.

Os dois níveis são necessários porque uma rota pode estar correta sobre um repositório errado e
vice-versa — e nenhum dos dois testes acha o defeito do outro.

## Por que a paridade é lida da ASSINATURA

Pede-se paridade *"com os campos que a interface expõe, nas duas direções"*. Uma lista
escrita à mão do que a interface expõe seria uma **terceira** fonte de verdade, e ela
divergiria das outras duas em silêncio — que é exactamente o defeito que a enumeração de
`SUPPORTED_OVERRIDES` existe para impedir.

Então o conjunto exposto sai de `inspect.signature(executar)`, o conjunto de eixos possíveis
sai de `dataclasses.fields(PipelineSpec)`, e as três direções são conferidas:

| Direção | O que pega |
|---|---|
| campo de spec exposto pela rota e **não** enumerado | eixo sobrescrevível por acidente |
| eixo enumerado e **não** exposto pela rota | promessa de sobrescrita que a rota não aceita |
| eixo enumerado que **não é campo** de `PipelineSpec` | renomeação de campo do domínio |

**Escopo negativo.** Estes testes não abrem porta: usam `TestClient`, que fala com a aplicação
em processo. A superfície de rede é de `tests/test_api_loopback.py`.
"""

import inspect
import json
from dataclasses import fields
from pathlib import Path

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from sqlalchemy import text

from refino.api.app import build_app
from refino.api.esquemas import INDISPONIVEIS, RESSALVAS
from refino.api.rotas import executar
from refino.cli import (
    AUTO_APRENDIDA,
    AUTO_CAVEATS,
    METHOD_CAVEATS,
    UNAVAILABLE_METHODS,
    Unavailable,
)
from refino.domain.spec import SUPPORTED_OVERRIDES, PipelineSpec, validate_override_fields
from tests.conftest import measurements_path, sample_list_path
from tests.reclamacao import reclama

MODO = "pos_hilic"

# `seed` fica FORA da sobrescrita de proposito: e configuracao de determinismo,
# nao escolha metodologica. Trocar a semente nao e discordar do metodo, e admiti-lo poluiria o
# conjunto de concordancia com linhas cujo eixo nao carrega juizo de especialista.
#: Campos que a rota expõe e que **não** são eixos sobrescrevíveis, com a razão de cada um.
#:
#: **A isenção não é conveniência do portão; é a resposta a uma pergunta diferente.**
#: `SUPPORTED_OVERRIDES` responde *"o humano pode DISCORDAR da máquina neste eixo?"*, e
#: discordar exige alguém do outro lado que **proponha**. Um campo que a rota aceita e que a
#: máquina nunca propõe não é sobrescrevível — é insumo, na categoria de `measurements` e
#: `sample_list`, que também não estão lá.
#:
#: `seed` — a semente é determinismo, não escolha metodológica.
#:
#: `normalization_factors` — os fatores são medidos FORA do equipamento (peso seco, volume de
#: urina, número de células) e vêm do laboratório. A máquina não tem como os propor, então não
#: há proposta de que discordar, e marcar *«sobrescrevi os fatores»* gravaria par de
#: concordância sobre um eixo que ninguém propôs. A referência da normalização **fica** em
#: `SUPPORTED_OVERRIDES`, e a diferença é exactamente essa: ela é escolha sobre dado que já
#: está na matriz.
#:
#: `batch_cuts` — os cortes não são eixo próprio: eles vão com o número de lotes que o
#: diagnóstico propôs, e saem quando o humano muda esse número. Quem discorda dos lotes discorda
#: no eixo `n_batches`, que está em `SUPPORTED_OVERRIDES`; e a linha da máquina do par leva os
#: cortes propostos, então a discordância fica gravada sem um eixo a mais.
FORA_DA_SOBRESCRITA = frozenset({"seed", "normalization_factors", "batch_cuts"})


@pytest.fixture(scope="module")
def cliente() -> TestClient:
    return TestClient(build_app())


def _reais(mode: str = MODO) -> dict[str, tuple[str, bytes, str]]:
    return {
        "measurements": (
            f"measurements_{mode}.csv",
            measurements_path(mode).read_bytes(),
            "text/csv",
        ),
        "sample_list": (f"Sample_List_{mode}.csv", sample_list_path(mode).read_bytes(), "text/csv"),
    }


def _spec_minima(**kw: object) -> dict[str, object]:
    """A spec explícita mais barata que passa pela guarda de `imputation='none'`.

    `imputation='none'` com as três normalizações em `none` é o caminho que deixa a matriz
    como está — a única combinação em que `none` é legítimo, porque propagar ausência por
    normalização produz matriz que **parece** processada e não é.
    """
    base: dict[str, object] = {
        "mode": MODO,
        "estudo": "ATLL/HAM-TSP",
        "correction": "none",
        "n_batches": 1,
        "rsd_variant": "plain",
        "imputation": "none",
        "sample_normalization": "none",
        "transformation": "none",
        "scaling": "none",
        "apresentacao": "exibida",
    }
    base.update(kw)
    return base


# ---------------------------------------------------------------------------
# A justificativa é EXIBIDA, e o rationale não é vazio
# ---------------------------------------------------------------------------


def test_ca_12_6_a_recomendacao_traz_rationale_nao_vazio(cliente: TestClient) -> None:
    """A metade do "Requisito de interação" que não tinha critério nenhum.

    A persistência da sobrescrita tinha critério; exibir a justificativa não era de
    ninguém. O objetivo fecharia com todos os critérios verdes e a justificativa nunca
    exibida.

    A asserção não é só "não vazio": o `rationale` cita duas características numéricas
    concretas, porque confiança sem origem é defeito.
    """
    resposta = cliente.post("/matrizes/recomendar", files=_reais(), data={"mode": MODO})

    assert resposta.status_code == 200, resposta.text
    corpo = resposta.json()
    assert corpo["rationale"].strip(), "justificativa vazia; `[CA-12.6]` exige não vazia"
    assert "RSD mediano de QC" in corpo["rationale"]
    assert "correlação de PC1" in corpo["rationale"]


def test_a_politica_se_abstem_e_isso_e_resultado_honesto_e_nao_degradacao(
    cliente: TestClient,
) -> None:
    """A margem é PREDITA, e no `pos_hilic` a política não se abstém — o que a ressalva declara.

    ## Este teste já afirmou o CONTRÁRIO, e a inversão é de contrato

    Ele exigia `avaliavel is False`, `abstem is True` e `confianca == 0.0`, e estava certo sobre
    a rota de então: ela corria `AlwaysSERRFPolicy()` — uma **constante** — sem estimador de
    margem, enquanto declarava `origem="modelo"`. **Nenhum modelo era consultado.**

    A rota passou a usar a política **aprendida**, que prediz as duas cabeças das 17
    características do perfil. A margem deixou de ser não-medível e passou a ser **predita**, e
    por isso os três campos invertem. **Não é tolerância afrouxada para o teste passar** — é o
    sujeito que mudou, por decisão registada, e o teste segue-o.

    A propriedade que ele protege NÃO mudou, e continua afirmada abaixo:
    `avaliavel` e `confianca` são campos separados, porque *"a margem foi medida e é minúscula"*
    e *"a margem não pôde ser medida"* pedem a mesma ação e produzem contagens diferentes.

    ## E ele passa a fixar o DEFEITO conhecido, em vez de o deixar implícito

    O critério de abstenção era insatisfazível: a folha mínima da árvore da margem valia
    `0,015269` contra o limiar de abstenção de `0,012`, então nenhuma entrada possível a fazia
    abster-se. O caso agudo era o `neg_c18`, cuja margem real era 0,22% e onde a resposta certa
    seria abster-se. Com a floresta do R, a margem do `neg_c18` é 2,56% e ele não é ambíguo
    (`τ` = 0,010251); o critério passou para as duas matrizes do corpus com margem abaixo do
    limiar, e a asserção abaixo continua a fixar o mesmo defeito por outro caminho: a árvore não
    se abstém.

    > **Afirmar `abstem is False` aqui não é aprovar o defeito: é impedi-lo de mudar em
    > silêncio.** Se alguém consertar a árvore da margem, este teste reprova e obriga a decisão a
    > ser registada — que é o oposto de um teste ausente, onde o conserto e a regressão são
    > indistinguíveis.

    E a árvore já mudou sem que ele reprovasse, e o motivo é o modo. Um modelo retreinado no
    corpus de 7 QCs teve uma folha da margem de `0,008299`, abaixo do limiar: a abstenção passou
    a ser alcançável. Mas o `pos_hilic`, que este arquivo usa, caía na folha de `0,1036` e
    continuava a não se abster, com confiança 0,942; então `abstem is False` continuou verdadeiro
    e o teste ficou verde. O portão que afirma o estado da árvore está em
    `tests/test_politica_aprendida.py`.
    """
    corpo = cliente.post("/matrizes/recomendar", files=_reais(), data={"mode": MODO}).json()

    assert corpo["avaliavel"] is True, (
        "a margem passou a ser PREDITA pela política aprendida; `avaliavel` falso "
        "significaria que a rota voltou à constante sem estimador"
    )
    assert corpo["confianca"] > 0.0, "margem predita e confiança zero são incoerentes"
    assert corpo["abstem"] is False, (
        "o `pos_hilic` cai na folha de margem 0,1036 e não se abstém. Se isto passou a ser "
        "`True`, a árvore da margem mudou de novo"
    )
    # O `rationale`: o caminho de decisão da árvore, com as características e os
    # limiares que de facto decidiram — e não uma atribuição calculada depois.
    assert "Por quê" in corpo["rationale"], (
        f"a justificativa não traz o caminho de decisão: {corpo['rationale']!r}"
    )
    # IGUALDADE e não só `"70,0%" in`: a ressalva da aprendida cita 70,0% contra 55,0% da
    # constante, e a da CONSTANTE também cita os 70,0%; pedir o número não separa as duas.
    assert corpo["ressalva_da_politica"] == AUTO_CAVEATS[AUTO_APRENDIDA].resumo, (
        f"a ressalva da resposta não é a da política aprendida: {corpo['ressalva_da_politica']!r}"
    )
    # 67,5% desde o retreino com a floresta do R; a árvore da margem não se abstém em entrada
    # nenhuma desde então (a menor folha vale 0,1139), e o `abstem is False` acima continua
    # verdadeiro por esse motivo, e não só pelo modo.
    assert "67,5%" in corpo["ressalva_da_politica"], (
        "a ressalva que desautoriza o palpite como atalho não veio na resposta, e ela é o que "
        "impede a confiança de 95% de se ler como conselho"
    )


def test_a_recomendacao_traz_as_duas_opinioes_e_a_segunda_nao_decide(
    cliente: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """As duas opiniões do Palpite: a pelo RSD decide, a pela verdade fica ao lado.

    **Âncora de que as duas são árvores diferentes, no corpo e não só no artefato:** no
    `pos_hilic` as duas propõem `serrf`, então a igualdade de corretor não separaria uma rota que
    servisse a mesma árvore duas vezes — é o CAMINHO que a separa. A primeira desce pela
    correlação de PC1, que é a raiz dela; a segunda desce por `is_gradual_drift`, que é a raiz da
    árvore treinada pela verdade.

    Com o interruptor LIGADO. A segunda está em stand-by, e é este teste que garante que
    religá-la devolve o que ela fazia. O desligado é o teste seguinte.
    """
    from refino.cli import RESSALVA_PELA_VERDADE, ROTULO_PELA_VERDADE, ROTULO_PELO_RSD
    from refino.io import politica

    monkeypatch.setattr(politica, "SEGUNDA_OPINIAO_ATIVA", True)
    corpo = cliente.post("/matrizes/recomendar", files=_reais(), data={"mode": MODO}).json()

    assert corpo["rotulo_da_politica"] == ROTULO_PELO_RSD
    segunda = corpo["segunda_opiniao"]
    assert segunda["rotulo"] == ROTULO_PELA_VERDADE
    assert segunda["correction_proposta"] in {"none", "loess", "serrf"}
    assert segunda["ressalva"] == RESSALVA_PELA_VERDADE.resumo
    assert segunda["caminho"].startswith("`is_gradual_drift`"), segunda["caminho"]
    assert segunda["caminho"] not in corpo["rationale"], (
        "o caminho da segunda opinião é o da primeira: a rota serviu a mesma árvore duas vezes"
    )
    # A ausência da confiança é desenho, justificado em `SegundaOpiniao`.
    assert not {"confianca", "abstem", "avaliavel"} & set(segunda), sorted(segunda)


def test_d578_em_stand_by_a_recomendacao_so_traz_a_opiniao_pelo_rsd(
    cliente: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A segunda opinião em stand-by: o campo vem nulo, e o artefato dela nem é lido.

    **Par causal sobre o interruptor:** com o carregador da segunda a levantar, a rota responde
    `200` desligada e `503` ligada — a diferença prova que, desligada, ela não carrega o artefato,
    e não só que o esconde. A âncora de presença é a primeira opinião no mesmo corpo: sem ela, um
    corpo vazio satisfaria a ausência.
    """
    from refino.api import rotas
    from refino.cli import ROTULO_PELO_RSD
    from refino.decision.artefato import ArtefatoInvalidoError
    from refino.io import politica

    assert politica.SEGUNDA_OPINIAO_ATIVA is False, "o estado vigente é a segunda desligada"
    corpo = cliente.post("/matrizes/recomendar", files=_reais(), data={"mode": MODO}).json()
    assert corpo["rotulo_da_politica"] == ROTULO_PELO_RSD
    assert corpo["correction_proposta"] in {"none", "loess", "serrf"}, corpo
    assert corpo["segunda_opiniao"] is None, corpo["segunda_opiniao"]

    def quebrado() -> None:
        raise ArtefatoInvalidoError("planta: o artefato da segunda opinião não abre")

    monkeypatch.setattr(rotas, "carrega_politica_pela_verdade", quebrado)
    desligada = cliente.post("/matrizes/recomendar", files=_reais(), data={"mode": MODO})
    monkeypatch.setattr(politica, "SEGUNDA_OPINIAO_ATIVA", True)
    ligada = cliente.post("/matrizes/recomendar", files=_reais(), data={"mode": MODO})
    assert desligada.status_code == 200, desligada.text
    assert ligada.status_code == 503, "ligada, a planta não chegou à rota: o par não separa nada"


def test_a_recomendacao_propoe_serrf_disponivel_e_com_ressalva(
    cliente: TestClient,
) -> None:
    """A resposta era ENGANOSA por indisponibilidade; hoje seria por SILÊNCIO.

    A política propõe `serrf` e ele passou a ser executável nas três superfícies. O campo
    `disponivel_para_execucao` vira `True` — e a resposta deixaria de dizer o que o uso não
    afirma, se a ressalva não viesse junto. O recorte não desapareceu com o bloqueio.

    **A asserção sobre `indisponiveis` FICA e mudou de conteúdo**, em vez de sair: uma lista
    que deixasse de ser conferida passaria a aceitar item sem motivo, e é ela que impede a
    ausência silenciosa.

    **E ela mudou outra vez em 01/09/2026, quando o `--auto` foi liberado.** O único item que
    restava era `recomendacao_automatica_delegada`, e a lista ficou VAZIA — o que torna o laço
    abaixo vácuo, e é por isso que a igualdade de conjunto continua a ser afirmada em vez de
    trocada por um `all`. Igualdade com o conjunto vazio **reprova** quando alguém acrescenta um
    indisponível sem tocar aqui, que é o sentido certo da falha; `all` sobre vazio nunca reprova.

    As duas políticas do `--auto` passam a aparecer em `ressalvas`, e o conjunto esperado cresce
    com elas: uma tela que anunciasse a delegação sem o número que a desautoriza é exactamente o
    que o item removido existia para impedir.
    """
    corpo = cliente.post("/matrizes/recomendar", files=_reais(), data={"mode": MODO}).json()

    assert corpo["correction_proposta"] == "serrf"
    assert corpo["disponivel_para_execucao"] is True

    itens = {item["item"]: item for item in corpo["indisponiveis"]}
    assert set(itens) == set(), (
        "`serrf` ou a delegação voltaram à lista de indisponíveis sem que este teste soubesse"
    )
    for nome, item in itens.items():  # pragma: no cover - vazio por estado desde 01/09/2026
        assert item["motivo"].strip(), f"{nome} indisponível SEM motivo — ausência silenciosa"
        assert item["criterio"].strip()

    ressalvas = {item["item"]: item for item in corpo["ressalvas"]}
    # Este portão é de igualdade EXACTA de propósito: ressalva nova obriga a decidir aqui em
    # vez de deslizar. A de `loess` está decidida: a matriz corrigida do LOESS sai com células
    # infinitas sob o padrão `r_faithful`, 109 a 763 por modo, e a janela não expõe a política
    # que as evita.
    # A de `--auto medida` segue a mesma decisão das outras duas políticas: a delegação pela
    # medida é anunciada onde a resposta anuncia as outras, e com o número que a qualifica —
    # 92,5% de concordância nas 88, sem critério que a meça.
    assert set(ressalvas) == {
        "serrf",
        "loess",
        "--auto aprendida",
        "--auto constante",
        "--auto medida",
    }, "a proposta é executável e não traz o recorte medido"
    for nome, ressalva in ressalvas.items():
        for campo in ("resumo", "detalhe", "evidencia", "portao_inerte"):
            assert ressalva[campo].strip(), f"ressalva de `{nome}` com `{campo}` vazio"


def test_a_ressalva_do_serrf_e_o_mesmo_texto_da_cli(cliente: TestClient) -> None:
    """Um segundo texto para a mesma ressalva divergiria do primeiro.

    Numa divergência de recorte, a frase viaja para a interface enquanto o número medido fica no
    registro. A CLI imprime e a API devolve a mesma string, e é isto
    que impede as duas de contarem histórias diferentes.

    **Herda o teste que fazia a mesma pergunta sobre o MOTIVO da indisponibilidade** — a
    propriedade verificada é fonte única, e ela não muda com o estado do método.
    """
    corpo = cliente.post("/matrizes/recomendar", files=_reais(), data={"mode": MODO}).json()
    da_api = next(i for i in corpo["ressalvas"] if i["item"] == "serrf")

    assert da_api["resumo"] == METHOD_CAVEATS["serrf"].resumo
    assert da_api["detalhe"] == METHOD_CAVEATS["serrf"].detalhe


def test_a_recomendacao_oferece_os_eixos_sobrescreviveis(cliente: TestClient) -> None:
    """A outra metade da justificativa exibida: quem devolve decisão automática aceita a
    sobrescrita.

    Sai de `SUPPORTED_OVERRIDES` e não de lista escrita à mão, para que um eixo novo não
    dependa de alguém lembrar de acrescentá-lo em dois lugares.
    """
    corpo = cliente.post("/matrizes/recomendar", files=_reais(), data={"mode": MODO}).json()

    assert corpo["eixos_sobrescreviveis"] == sorted(SUPPORTED_OVERRIDES)


# ---------------------------------------------------------------------------
# Paridade MECÂNICA, nas três direções
# ---------------------------------------------------------------------------


def campos_expostos_pela_rota() -> frozenset[str]:
    """Os campos de `PipelineSpec` que `POST /execucoes` aceita, lidos da ASSINATURA.

    Lida por `inspect` e não escrita à mão: uma lista manual seria a terceira fonte de verdade
    e divergiria das outras duas em silêncio.
    """
    parametros = set(inspect.signature(executar).parameters)
    return frozenset(parametros & {campo.name for campo in fields(PipelineSpec)})


def defeitos_de_paridade(expostos: frozenset[str], suportados: frozenset[str]) -> list[str]:
    """Um defeito por divergência, nas TRÊS direções.

    Função separada devolvendo lista: a planta entra como argumento e o negativo percorre o
    caminho do portão. Negativo em processo não consegue ficar inerte,
    porque a entrada alimenta a asserção e não há intermediário que possa não recebê-la.
    """
    achados: list[str] = []
    do_dominio = {campo.name for campo in fields(PipelineSpec)}

    for campo in sorted(expostos - suportados - FORA_DA_SOBRESCRITA):
        achados.append(
            f"a rota expõe `{campo}` e ele NÃO está em SUPPORTED_OVERRIDES: eixo sobrescrevível "
            "por acidente, e o par gravaria concordância sobre eixo que ninguém declarou"
        )
    for campo in sorted(suportados - expostos):
        achados.append(
            f"SUPPORTED_OVERRIDES enumera `{campo}` e a rota NÃO o expõe: promessa de "
            "sobrescrita que a rota não aceita"
        )
    for campo in sorted(suportados - do_dominio):
        achados.append(
            f"SUPPORTED_OVERRIDES enumera `{campo}` e ele NÃO é campo de PipelineSpec: "
            "sobrescrevê-lo não faria nada, e é o que renomeação de campo do domínio produz"
        )
    return achados


def test_to_12_4_a_arvore_limpa_tem_paridade_nas_tres_direcoes() -> None:
    assert defeitos_de_paridade(campos_expostos_pela_rota(), SUPPORTED_OVERRIDES) == []


def test_to_12_4_campo_exposto_e_nao_enumerado_reprova() -> None:
    """Direção A: eixo que passou a ser sobrescrevível sem entrar na lista."""
    defeitos = defeitos_de_paridade(campos_expostos_pela_rota(), SUPPORTED_OVERRIDES - {"scaling"})

    assert any("expõe `scaling`" in d and "NÃO está em SUPPORTED_OVERRIDES" in d for d in defeitos)


def test_to_12_4_eixo_enumerado_e_nao_exposto_reprova() -> None:
    """Direção B: promessa que a rota não cumpre."""
    defeitos = defeitos_de_paridade(
        campos_expostos_pela_rota() - {"imputation"}, SUPPORTED_OVERRIDES
    )

    assert any("enumera `imputation`" in d and "NÃO o expõe" in d for d in defeitos)


def test_to_12_4_eixo_que_nao_e_campo_do_dominio_reprova() -> None:
    """Direção C: a que pega renomeação de campo de `PipelineSpec`."""
    defeitos = defeitos_de_paridade(
        campos_expostos_pela_rota(), SUPPORTED_OVERRIDES | {"correcao_antiga"}
    )

    assert any("NÃO é campo de PipelineSpec" in d for d in defeitos)


def test_to_12_4_com_a_planta_removida_o_portao_deixa_de_acusar() -> None:
    """A outra metade do par causal: sem violação, nenhum defeito.

    Não é o mesmo que ver o portão reprovar — é vê-lo **deixar de acusar** quando não há o que
    acusar. Um portão que sempre acusasse passaria nos três negativos acima.
    """
    assert defeitos_de_paridade(campos_expostos_pela_rota(), SUPPORTED_OVERRIDES) == []


def test_seed_esta_fora_da_sobrescrita_e_a_exclusao_e_deliberada() -> None:
    """`seed` é configuração de determinismo, não escolha metodológica.

    Sem esta asserção, alguém que o acrescentasse a `SUPPORTED_OVERRIDES` passaria pela
    paridade — ele **é** campo de `PipelineSpec` e **é** exposto pela rota. O que o mantém fora
    é a decisão, e a decisão precisa de teste.
    """
    assert "seed" not in SUPPORTED_OVERRIDES
    assert "seed" in campos_expostos_pela_rota(), (
        "a rota deixou de aceitar `seed`; sem ele a execução não é reprodutível (regra 3 do README)"
    )
    with pytest.raises(ValueError, match="nao sobrescreviv"):
        validate_override_fields(["seed"])


# ---------------------------------------------------------------------------
# A fronteira ACEITA `serrf`, e a guarda de recusa continua a ter dente
# ---------------------------------------------------------------------------


def test_a_fronteira_da_rota_aceita_serrf_e_o_recusa_quando_bloqueado() -> None:
    """Par causal sobre `_corretor_ou_422`.

    O teste que aqui estava afirmava `422` para `--correction serrf`. O SERRF passou a ser
    selecionável, e aquelas asserções passaram a afirmar o contrário do estado.

    **Não é chamado pela rota, e a escolha é de CUSTO declarado.** Um `POST /execucoes` com
    `correction=serrf` sobre o `pos_hilic` executa um SERRF real — medido, **86 s** no
    `neg_c18` de 2.184 features, e o `pos_hilic` tem 6.035. A camada de commit deixaria de ser
    de minutos. A propriedade que este teste afirma é a da **fronteira** — quem ela aceita e
    quem ela recusa —, e ela é da função. O caminho ponta a ponta é o teste `slow` abaixo.

    **As duas direções, porque uma fronteira que aceitasse tudo passaria no primeiro braço.**
    """
    from refino.api.rotas import _corretor_ou_422

    corretor = _corretor_ou_422("serrf", "r_faithful")
    assert corretor.name == "serrf"

    planta = Unavailable(
        reason="planta de teste: bloqueado por [CA-9.9]",
        implemented=True,
        gate="[CA-9.9]",
        gate_unit="U99",
        corrector="refino.correction.serrf:SerrfCorrector",
    )
    with pytest.MonkeyPatch.context() as mp:
        mp.setitem(UNAVAILABLE_METHODS, "serrf", planta)
        with pytest.raises(HTTPException) as capturado:
            _corretor_ou_422("serrf", "r_faithful")
    assert capturado.value.status_code == 422
    assert "não é selecionável" in str(capturado.value.detail)
    assert "[CA-9.9]" in str(capturado.value.detail), "a recusa não cita o portão da entrada"


@pytest.mark.slow
def test_serrf_pela_rota_de_execucao_produz_resultado(cliente: TestClient) -> None:
    """**O caminho ponta a ponta, `slow` por CUSTO MEDIDO e não por precaução.**

    Marcado `slow` porque executa um SERRF real: 86 s no `neg_c18`, e este roda sobre o
    `pos_hilic` de 6.035 features. É a camada certa — `make test-all` o alcança, o portão de
    commit não —, e a alternativa era não ter cobertura ponta a ponta nenhuma do método.

    **O que ele afirma que o teste rápido não pode:** que a rota inteira aceita, executa e
    grava — não só que a fronteira devolve o corretor certo.

    Sem `pytest.skip` local: a guarda de banco vive num sítio só, em `conftest.py`, e
    reintroduzi-la aqui recriaria as duas cópias que já divergiram uma vez no texto. O `503`
    entra na tupla aceita, como os testes vizinhos fazem, e a asserção do conteúdo fica sob a
    condição.
    """
    resposta = reclama(cliente, "/execucoes", files=_reais(), data=_spec_minima(correction="serrf"))

    # `202` e nao `200` no disparo, e `reclama` ja o afirma: o que chega aqui e a resposta da
    # RECLAMACAO quando houve banco, ou o `503` do disparo quando nao houve.
    assert resposta.status_code in (200, 503), resposta.text
    if resposta.status_code == 200:
        # **A assercao anterior era `json()["correction"] == "serrf"`, e aquela chave NUNCA existiu
        # em `ExecucaoResposta`** — ela so nao reprovava porque este ramo exige banco de pe, que
        # e a condicao que a memoria deste projeto mede como ausente. Com a forma da resposta a
        # mudar de todo modo, escrever a chave errada de novo seria conhecer o defeito e mante-lo.
        corpo = resposta.json()
        assert corpo["estado_concordancia"], corpo
        assert corpo["run_id"] > 0, corpo


def test_eixo_fora_de_supported_overrides_e_recusado_pela_rota(cliente: TestClient) -> None:
    """Ignorar o eixo desconhecido gravaria par sobre eixo que a enumeração não declara."""
    resposta = cliente.post(
        "/execucoes",
        files=_reais(),
        data=_spec_minima(campos_sobrescritos=json.dumps(["seed"])),
    )

    assert resposta.status_code == 422, resposta.text
    assert "nao sobrescreviv" in resposta.json()["detail"]


def test_eixos_sem_proposta_e_recusado(cliente: TestClient) -> None:
    """Sem a proposta da máquina não há com que concordar."""
    resposta = cliente.post(
        "/execucoes",
        files=_reais(),
        data=_spec_minima(campos_sobrescritos=json.dumps(["correction"])),
    )

    assert resposta.status_code == 422, resposta.text
    assert "não veio `proposta_json`" in resposta.json()["detail"]


def test_proposta_sem_eixos_e_recusada_e_e_a_direcao_que_o_desenho_obvio_esquece(
    cliente: TestClient,
) -> None:
    """A segunda direção: proposta sem eixo gravaria par cujo eixo ninguém sabe nomear."""
    resposta = cliente.post(
        "/execucoes",
        files=_reais(),
        data=_spec_minima(
            campos_sobrescritos="[]",
            proposta_json=json.dumps({"spec_json": "{}", "rationale": "r", "origem": "modelo"}),
        ),
    )

    assert resposta.status_code == 422, resposta.text
    assert "está vazio" in resposta.json()["detail"]


def test_campos_sobrescritos_que_nao_e_vetor_e_recusado(cliente: TestClient) -> None:
    """`"correction"` sozinho seria iterado como caracteres e produziria eixos absurdos."""
    resposta = cliente.post(
        "/execucoes", files=_reais(), data=_spec_minima(campos_sobrescritos='"correction"')
    )

    assert resposta.status_code == 422, resposta.text
    assert "vetor de nomes" in resposta.json()["detail"]


# ---------------------------------------------------------------------------
# `sem_proposta`, o quarto estado do par, pela rota
# ---------------------------------------------------------------------------

_PROPOSTA_DE_AMOSTRA = json.dumps({"spec_json": "{}", "rationale": "r", "origem": "modelo"})


@pytest.mark.parametrize(
    ("extra", "nomeado"),
    [
        ({"campos_sobrescritos": json.dumps(["correction"])}, "`campos_sobrescritos`"),
        ({"proposta_json": _PROPOSTA_DE_AMOSTRA}, "`proposta_json`"),
        ({"auto": AUTO_APRENDIDA}, "`auto`"),
    ],
    ids=["com-eixo", "com-proposta", "com-delegacao"],
)
def test_d707_sem_proposta_ao_lado_de_uma_proposta_e_recusado(
    cliente: TestClient, extra: dict[str, str], nomeado: str
) -> None:
    """`sem_proposta` com eixo, proposta ou delegação é contradição, e a rota a recusa.

    **Par causal na guarda:** com o campo desligado, a mesma combinação passa por ela — e a rota
    segue para as recusas de antes, ou para a execução com `auto`. A metade desligada chama a
    guarda direto, para não disparar uma delegação de verdade só para ver que ela não recusa.
    """
    from refino.api.rotas import _sem_proposta_ou_422

    com = cliente.post(
        "/execucoes", files=_reais(), data=_spec_minima(sem_proposta="true", **extra)
    )
    assert com.status_code == 422, com.text
    detalhe = com.json()["detail"]
    assert "veio `sem_proposta`" in detalhe and nomeado in detalhe, detalhe

    argumentos = {
        "eixos": tuple(json.loads(extra.get("campos_sobrescritos", "[]"))),
        "proposta_json": extra.get("proposta_json"),
        "auto": extra.get("auto"),
    }
    _sem_proposta_ou_422(False, **argumentos)  # desligado: não levanta
    with pytest.raises(HTTPException):
        _sem_proposta_ou_422(True, **argumentos)


@pytest.mark.integration
def test_d707_sem_proposta_grava_a_linha_manual_e_o_banco_le_o_quarto_estado(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """O mesmo pedido, com e sem `sem_proposta`, contra SQLite real.

    **Par causal sobre o que fica GRAVADO**: sem o campo, uma linha `modelo` e
    `nao_sobrescreveu`; com ele, uma linha `manual`, a justificativa `RAZAO_SEM_PROPOSTA` e
    `sem_proposta`. Antes, o segundo pedido não existia, e a janela gravava toda Correção feita
    sem «Seguir sugestões do app» como o primeiro — a aceitação de uma proposta que ninguém
    seguiu.

    O estado vem da reclamação, que o lê de `decisions` (`classifica_concordancia`), e as linhas
    são lidas do banco à parte: a resposta podia dizer o estado certo sobre linhas erradas.
    """
    from sqlalchemy import create_engine

    from refino.api.rotas import RAZAO_SEM_PROPOSTA
    from refino.persistence.config import VARIAVEL_DA_URL, prepara_sqlite
    from refino.persistence.schema import METADATA

    url = f"sqlite:///{tmp_path / 'refino.db'}"
    monkeypatch.delenv("REFINO_DB_HOST", raising=False)
    monkeypatch.setenv(VARIAVEL_DA_URL, url)
    prepara_sqlite(url)
    motor = create_engine(url)
    METADATA.create_all(motor)
    cliente_real = TestClient(build_app(), raise_server_exceptions=False)
    modo = "neg_c18"

    def executa(estudo: str, **extra: str) -> tuple[str, dict[str, object]]:
        resposta = reclama(
            cliente_real,
            "/execucoes",
            files=_reais(modo),
            data=_spec_minima(mode=modo, estudo=estudo, **extra),
        )
        assert resposta.status_code == 200, resposta.text[:600]
        corpo = resposta.json()
        assert corpo["decision_proposta_id"] is None, corpo
        # Um estudo por pedido, então uma matriz por pedido: as linhas desta matriz são as dele.
        with motor.connect() as c:
            linhas = [
                dict(r._mapping)
                for r in c.execute(
                    text("SELECT origem, rationale FROM decisions WHERE matrix_id = :m"),
                    {"m": corpo["matrix_id"]},
                )
            ]
        assert len(linhas) == 1, f"`{estudo}`: o par sem eixo grava uma linha, e há {linhas}"
        return str(corpo["estado_concordancia"]), linhas[0]

    estado_sem, linha_sem = executa("d707_sem_o_campo")
    estado_com, linha_com = executa("d707_com_o_campo", sem_proposta="true")
    motor.dispose()

    assert (estado_sem, linha_sem["origem"]) == ("nao_sobrescreveu", "modelo"), linha_sem
    assert (estado_com, linha_com["origem"]) == ("sem_proposta", "manual"), (
        f"com `sem_proposta` o banco gravou {linha_com} e leu `{estado_com}`"
    )
    assert linha_com["rationale"] == RAZAO_SEM_PROPOSTA, linha_com
    assert linha_sem["rationale"] != RAZAO_SEM_PROPOSTA, "âncora: as duas frases não diferem"


def test_spec_invalida_e_recusada_na_construcao_e_nada_e_gravado(cliente: TestClient) -> None:
    """A guarda da combinação proibida: `imputation='none'` com normalização ativa produz
    matriz que PARECE processada e não é.

    A recusa acontece na **construção** de `PipelineSpec`, antes de qualquer escrita — então
    não há linha a desfazer, nem a proposta.
    """
    resposta = cliente.post(
        "/execucoes",
        files=_reais(),
        data=_spec_minima(imputation="none", scaling="auto"),
    )

    assert resposta.status_code == 422, resposta.text
    assert "imputation" in resposta.json()["detail"]


def test_apresentacao_fora_do_vocabulario_e_recusada_na_fronteira(cliente: TestClient) -> None:
    """O campo registra **exibida**, nunca **lida**, e o vocabulário é fechado em três.

    **A asserção é 422 e não "algum erro", e é ela que fixa o conserto.** A primeira versão
    validava `apresentacao` dentro de `grava_execucao`: o valor inválido atravessava ingestão,
    correção e pipeline inteiro antes de reprovar, e com o banco ausente reprovava como **503**
    — dizendo *"o serviço não subiu"* sobre uma requisição malformada. Aceitar `(422, 500, 503)`
    aqui teria deixado o defeito passar: reprova pelo motivo errado e o desfecho absorve.
    """
    resposta = cliente.post("/execucoes", files=_reais(), data=_spec_minima(apresentacao="lida"))

    assert resposta.status_code == 422, resposta.text
    assert "apresentacao" in resposta.json()["detail"]


def test_os_catalogos_de_escopo_nao_estao_vazios() -> None:
    """A direção que impede os catálogos de passarem por vacuidade.

    Um catálogo vazio satisfaria trivialmente "todo item tem motivo", e o teste nasceria verde
    e inerte.

    `RESSALVAS` tem a mesma guarda, e não como acréscimo simétrico de cortesia: ela é
    derivada de `METHOD_CAVEATS` por compreensão, então esvaziar o dicionário a esvazia em
    silêncio, e toda asserção sobre o conteúdo dela passaria. É a mesma exposição de
    `INDISPONIVEIS`.
    """
    # A âncora é a UNIÃO, e não `INDISPONIVEIS`. O `--auto` saiu de indisponível para ressalva,
    # e `INDISPONIVEIS` ficou vazio — estado legítimo, pelo mesmo argumento de
    # `UNAVAILABLE_METHODS`. O que NÃO é legítimo é a asserção seguir a ancorar nele: `all(...)`
    # sobre coleção vazia passa, e o teste ficaria inerte sem que a contagem de `passed` mudasse.
    #
    # A união é a quantidade que o catálogo existe para ter: **a tela sempre tem algo a dizer
    # sobre o recorte.** Esvaziar as duas tabelas reprova aqui; esvaziar uma não, e é correto —
    # foi o que aconteceu, e o que se afirma de cada item continua item a item.
    assert INDISPONIVEIS or RESSALVAS, "os DOIS catálogos de recorte estão vazios"
    assert all(item.motivo.strip() and item.criterio.strip() for item in INDISPONIVEIS)

    assert RESSALVAS, "o catálogo de ressalvas está vazio"
    assert all(
        item.resumo.strip()
        and item.detalhe.strip()
        and item.evidencia.strip()
        and item.portao_inerte.strip()
        for item in RESSALVAS
    )


def test_sem_banco_a_rota_responde_503_e_nao_500(
    cliente: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Banco ausente é indisponibilidade do serviço, não erro de quem chamou.

    Colapsá-los em 500 diria ao usuário que ele errou a requisição quando o fato é que o
    banco não subiu — são ações opostas, e é a distinção que `MethodUnavailableError` faz
    na CLI.
    """
    monkeypatch.delenv("REFINO_DB_HOST", raising=False)

    resposta = cliente.post("/execucoes", files=_reais(), data=_spec_minima())

    assert resposta.status_code == 503, resposta.text
    assert "REFINO_DB_URL" in resposta.json()["detail"]


def test_d710_a_justificativa_gravada_e_a_frase_da_tela_dizem_o_mesmo_de_sem_proposta() -> None:
    """A justificativa de `sem_proposta` no banco diz o que a tela diz do mesmo estado.

    Na janela o caso comum é o Palpite guardado e não seguido, em que houve proposta, e por isso a
    tela diz 'nenhuma proposta da máquina foi seguida'. As duas superfícies descrevem o mesmo
    estado, e o teste afirma a concordância entre elas em vez de fixar uma frase.
    """
    from refino.api.rotas import RAZAO_SEM_PROPOSTA
    from refino.interface.fluxo import PROSA_DA_CONCORDANCIA

    nucleo = "nenhuma proposta da máquina foi seguida"
    assert nucleo in PROSA_DA_CONCORDANCIA["sem_proposta"], "âncora: a frase da tela mudou"
    assert nucleo in RAZAO_SEM_PROPOSTA, (
        f"a justificativa gravada diz outra coisa do mesmo estado: {RAZAO_SEM_PROPOSTA!r}"
    )
