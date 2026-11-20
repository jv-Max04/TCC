"""`POST /matrizes/medir-corretor`: a metade «depois» do par entre o palpite e a medição.

**A rota é a RECOMENDAÇÃO DE FACTO**, e a distinção contra `/matrizes/recomendar` é o assunto
inteiro desta unidade: aquela **prediz** em segundos e mede 70,0% deixando um modo de fora; esta
**corre** `none`, `loess` e `serrf` e compara pela régua fora da amostra.

## Por que o oráculo entra como DUPLO na maioria destes testes

`LabelingOracle.label` faz `n_qc` reajustes **por corretor** — 21 ajustes com sete QCs e três
corretores —, e o corpus de treino registrou **~10 min por matriz**. Uma suíte que o corresse a
sério em cada asserção não seria corrida por ninguém.

> **O que o duplo NÃO pode esconder é a distinção que a unidade existe para dar**, e por isso a
> âncora está em cada teste: o duplo devolve `scores` com os **três** corretores e um vencedor que
> **não** é o palpite, porque um duplo que devolvesse o mesmo que a política tornaria
> `palpite_acertou` verdadeiro por construção e a asserção do par seria vácua.

**O caminho real é exercitado uma vez**, em `test_medicao_real_de_um_modo`, sob a marca `slow`.
Sem ele estes testes provariam a fiação e não o resultado.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any

import pytest
from fastapi import status
from fastapi.testclient import TestClient

from refino.api.app import build_app
from refino.cli import AUTO_CAVEATS, AUTO_MEDIDA
from refino.decision.labeling import Label
from refino.decision.palpite import decide_o_palpite
from refino.io.readers import load_matrix
from tests.conftest import measurements_path, sample_list_path
from tests.reclamacao import reclama

pytestmark = pytest.mark.integration

MODO = "neg_c18"


@pytest.fixture(scope="module")
def cliente() -> TestClient:
    return TestClient(build_app())


def _arquivos(mode: str) -> dict[str, tuple[str, bytes, str]]:
    return {
        "measurements": (
            f"measurements_{mode}.csv",
            measurements_path(mode).read_bytes(),
            "text/csv",
        ),
        "sample_list": (
            f"Sample_List_{mode}.csv",
            sample_list_path(mode).read_bytes(),
            "text/csv",
        ),
    }


def pedidos_do_oraculo(pedidos: list[tuple[str, ...]]) -> tuple[str, ...]:
    """O que a rota pediu ao oráculo, achatado — vazio se ele nunca foi chamado.

    Vazio é o caso que a âncora tem de apanhar: `set(()) == {...}` é falso, mas
    `assert not any(...)` sobre lista vazia passaria — e este auxiliar existe para a asserção de
    não-vacuidade ficar na MESMA função que a usa.
    """
    return tuple(c for chamada in pedidos for c in chamada)


def _rotulo_falso(vencedor: str, scores: dict[str, float]) -> Label:
    """Um `Label` construído à mão, com os TRÊS corretores nos scores.

    **Os três e não dois**, porque a propriedade que esta rota existe para ter é `none` estar na
    comparação: a árvore aprendida nunca o devolve, e é justamente nos arquivos reais que ele
    ganha. Um duplo com dois prefixos passaria nas asserções de fiação e perderia a única coisa
    que distingue a medição do palpite.
    """
    return Label(
        prefix=(vencedor, 1),  # type: ignore[arg-type]
        scores={(nome, 1): valor for nome, valor in scores.items()},  # type: ignore[dict-item]
        estimator="leave_one_qc_out",
        variant="outlier_removed",
    )


@pytest.fixture
def oraculo_rapido(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, ...]]:
    """O oráculo trocado por um que devolve `none` como vencedor, sem correr corretor nenhum.

    **`none` de propósito**, e é a âncora do duplo: a política aprendida só tem folhas `serrf` e
    `loess`, então um vencedor `none` **não pode** coincidir com o palpite. Isso é o que torna a
    asserção de `palpite_acertou is False` capaz de reprovar — com um duplo que devolvesse `serrf`
    ela passaria com o campo cablado a `True`. O palpite é o Palpite MEDIDO guardado, que pode
    propor `none`; o teste do par semeia-o com `serrf`, e a âncora vale pelo mesmo motivo.

    **DEVOLVE o que a rota pediu ao oráculo, e a devolução foi acrescentada depois de a planta a
    exigir.** A primeira escrita afirmava *"os três corretores estão na comparação"* lendo os
    `scores` do **duplo** — e um duplo devolve o que quem o escreveu pôs lá. Plantado o corte de
    `none` na construção do oráculo da rota, os dois testes **continuaram verdes**: a planta não
    alcançava a asserção.

    O duplo recebe `self`, então a **configuração real** que a rota construiu passa por aqui. É
    ela que se afirma, e não o retorno inventado.
    """
    from refino.api import rotas

    class _Pedidos(list[tuple[str, ...]]):
        """A lista dos prefixos pedidos, com a BANDEIRA que a rota passou pendurada nela.

        **Subclasse de `list` e não uma tupla de dois**, porque oito testes já recebem esta fixture
        e iteram-na; trocar o retorno obrigaria a mexer nos oito para acrescentar um campo que
        só um usa. `list` nua não aceita atributo — daí a subclasse, que é o custo mínimo de
        acrescentar sem quebrar.
        """

        bandeiras: list[Any]
        sementes: list[int]

    pedidos = _Pedidos()

    # A BANDEIRA que a rota passa fica REGISTADA, e não engolida. O duplo ganhou
    # `cancelled` porque `LabelingOracle.label` passou a declará-lo; um duplo que apenas o
    # aceitasse e o descartasse deixaria a rota sem cobertura do caminho do cancelamento —
    # e a rota é o único sítio de produção que o liga. Guardá-lo faz a travessia ser afirmável por
    # `test_d379_a_rota_passa_a_bandeira_de_cancelamento_ao_oraculo`, abaixo.
    bandeiras: list[Any] = []
    sementes: list[int] = []

    # `batches` entrou quando a rota passou a mandar os lotes ao oráculo. O duplo tem
    # de aceitar o que a assinatura real aceita: sem ele, o trabalho levantaria `TypeError` dentro
    # da thread, e os testes deste arquivo — que pulam onde não há os arquivos de referência —
    # reprovariam só na máquina que os tem.
    def falso(
        self: Any,
        matrix: Any,
        *,
        batches: Any = None,
        depois_de_cada: Any = None,
        cancelled: Any = None,
    ) -> Label:
        pedidos.append(tuple(correcao for correcao, _lotes in self.prefixes))
        bandeiras.append(cancelled)
        sementes.append(self.seed)
        # **O duplo INVOCA o gancho, uma vez por prefixo, como o oráculo real.** Um duplo que o
        # ignorasse deixaria a rota avisar o progresso **zero** vezes, e `_executa_trabalho`
        # concluiria na mesma — `conclui` força `feito` a `total`. O desfecho seria verde e a
        # cadência da barra ficaria sem cobertura nenhuma: é o valor que não exercita.
        if depois_de_cada is not None:
            for prefixo in self.prefixes:
                depois_de_cada(prefixo)
        return _rotulo_falso("none", {"none": 0.1550, "serrf": 0.1602, "loess": 0.1974})

    monkeypatch.setattr(rotas.LabelingOracle, "label", falso)
    # **Devolve os dois, e o segundo é acessível por atributo em vez de por índice.** Os testes que
    # já existiam recebem a lista de pedidos e continuam a iterá-la; quem precisa da bandeira lê
    # `.bandeiras`. Trocar o retorno por uma tupla obrigaria a mexer em oito testes para
    # acrescentar um.
    pedidos.bandeiras = bandeiras
    pedidos.sementes = sementes
    return pedidos


def test_d379_a_rota_passa_a_bandeira_de_cancelamento_ao_oraculo(
    cliente: TestClient, oraculo_rapido: Any
) -> None:
    """A rota entrega ao oráculo uma bandeira **chamável**, e não `None`.

    ## O defeito que isto fecha veio de um RELATO, não de um portão

    O usuário mandou parar uma medição de corretor, a tela respondeu *"a parada leva até cerca de
    um segundo"*, e o trabalho continuou até ele fechar o aplicativo. A cadeia tinha um elo em
    branco: `leave_one_qc_out_rsd` não aceitava `cancelled`, então o corretor recebia `None` e os
    três pontos de checagem do SERRF ficavam inertes.

    **Este teste guarda o elo do lado da ROTA**; o do lado do estimador é
    `test_d379_o_estimador_leva_a_bandeira_ate_o_corretor`, em
    `tests/test_cancelamento_de_trabalhos.py`. Separados porque são duas travessias e podem quebrar
    uma sem a outra — foi assim que a cadeia esteve partida no meio com as duas pontas certas.

    **A asserção é sobre CHAMÁVEL e não sobre um valor**, porque o que a rota entrega é a pergunta
    *"já pediram para parar?"*, e ela ainda responde `False` aqui: nada foi cancelado. Afirmar
    `is not None` deixaria passar um `False` literal, que satisfaz *"não é `None`"* e não é a
    pergunta — é a resposta congelada dela, e `raise_if_cancelled` chamaria um `bool`.
    """
    resposta = reclama(
        cliente, "/matrizes/medir-corretor", files=_arquivos(MODO), data={"mode": MODO}, teto_s=120
    )

    assert resposta.status_code == status.HTTP_200_OK, resposta.text
    assert oraculo_rapido.bandeiras, (
        "ÂNCORA: o oráculo não foi chamado, e a asserção abaixo passaria por vacuidade"
    )

    for bandeira in oraculo_rapido.bandeiras:
        assert callable(bandeira), (
            f"a rota entregou {bandeira!r} ao oráculo em vez da pergunta de cancelamento. Com "
            f"`None`, `raise_if_cancelled` é no-op e o trabalho fica sem ponto de parada — é o "
            f"defeito que o usuário reportou tendo de fechar o aplicativo"
        )
        assert bandeira() is False, (
            "a bandeira já diz «parar» sem ninguém ter pedido, e a medição abortaria sozinha"
        )


def test_a_rota_devolve_202_e_o_resultado_traz_os_tres_corretores(
    cliente: TestClient, oraculo_rapido: list[tuple[str, ...]]
) -> None:
    """O contrato: `202` no disparo, e o corpo reclamado com a comparação inteira."""
    resposta = reclama(
        cliente, "/matrizes/medir-corretor", files=_arquivos(MODO), data={"mode": MODO}, teto_s=120
    )

    assert resposta.status_code == status.HTTP_200_OK, resposta.text
    corpo = resposta.json()

    assert set(corpo["scores"]) == {"none", "loess", "serrf"}, (
        f"a medição não comparou os três corretores: {corpo['scores']}"
    )
    assert corpo["vencedor"] == "none"
    assert corpo["estimador"] == "leave_one_qc_out"
    assert corpo["variante"] == "outlier_removed"


def test_a_rota_pede_ao_oraculo_os_tres_corretores_none_incluido(
    cliente: TestClient, oraculo_rapido: list[tuple[str, ...]]
) -> None:
    """A propriedade que a rota existe para ter, e a política aprendida NÃO tem.

    A árvore do artefato só tem folhas `serrf` e `loess` — `none` nunca é resposta dela. Nos
    arquivos reais o `none` não vence em modo nenhum dos quatro, em nenhuma das duas florestas.
    A propriedade que este teste guarda não depende disso: a árvore pelo RSD não tem folha
    `none`, e só a medição pode responder *não corrija* pela régua dos QCs.

    **A asserção é sobre o que a rota PEDIU, e não sobre o que o duplo devolveu.** A primeira
    escrita lia `corpo["scores"]` — e um duplo devolve o que quem o escreveu pôs lá. Cortado
    `none` da construção do oráculo na rota, aquela versão ficava **verde**.
    """
    # Chave distinta pelo motivo que a âncora abaixo apanhou na primeira execução: `_despacha`
    # deduplica por `(operação, matriz, campos)`, e um trabalho reclamado devolve o resultado
    # anterior **sem chamar o oráculo**. A lista ficava vazia e a asserção de conjunto passaria a
    # comparar `set(())`, que reprova — mas com a mensagem errada.
    reclama(
        cliente,
        "/matrizes/medir-corretor",
        files=_arquivos(MODO),
        data={"mode": MODO, "n_sampled_features": "13"},
        teto_s=120,
    )

    assert pedidos_do_oraculo(oraculo_rapido), (
        "ÂNCORA: o oráculo não foi chamado, então esta asserção passaria por vacuidade"
    )
    assert set(pedidos_do_oraculo(oraculo_rapido)) == {"none", "loess", "serrf"}, (
        f"a rota não pediu os três corretores: {pedidos_do_oraculo(oraculo_rapido)}"
    )


def test_o_par_grava_o_palpite_ao_lado_da_medicao(
    cliente: TestClient,
    oraculo_rapido: list[tuple[str, ...]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`palpite` e `palpite_acertou`: um pré-registro aplicado ao produto.

    O palpite é lido **antes** de a medição correr, e o par é o que torna cada estudo um ponto de
    evidência sobre se o Palpite funciona.

    O palpite é o Palpite MEDIDO guardado por `POST /matrizes/palpite`, e não a
    árvore. Ele é semeado aqui com `serrf`, e a âncora do duplo continua: o duplo elege `none`.
    `n_sampled_features` distinto pelo motivo escrito no teste abaixo — a chave do trabalho.
    """
    from refino.api import rotas

    monkeypatch.setattr(rotas, "_PALPITES", OrderedDict())
    matriz = load_matrix(measurements_path(MODO), sample_list_path(MODO), MODO)
    rotas._guarda_o_palpite(
        rotas._chave_do_palpite(matriz, 1, None),
        decide_o_palpite({"none": 0.3, "loess": 0.4, "serrf": 0.2}, n_features_medidas=200),
    )

    corpo = reclama(
        cliente,
        "/matrizes/medir-corretor",
        files=_arquivos(MODO),
        data={"mode": MODO, "n_sampled_features": "12"},
        teto_s=120,
    ).json()

    assert corpo["palpite"] == "serrf", (
        f"o palpite não veio do Palpite guardado: {corpo['palpite']!r}"
    )
    assert corpo["palpite_acertou"] is False, (
        f"o duplo elege `none` e o Palpite semeado é `serrf`, então o par tem de acusar erro: "
        f"palpite={corpo['palpite']!r} vencedor={corpo['vencedor']!r}"
    )


def test_o_par_e_none_e_nao_falso_quando_nao_houve_palpite(
    cliente: TestClient,
    oraculo_rapido: list[tuple[str, ...]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No par, palpite AUSENTE e palpite ERRADO são coisas diferentes.

    Colapsá-los inflaria a contagem de erros do modelo com execuções em que ele nem foi
    consultado — e é a contagem que decide se ele funciona.
    """
    from refino.api import rotas

    # A planta é a memória VAZIA: sem Palpite medido desta matriz e destes lotes, não há palpite.
    monkeypatch.setattr(rotas, "_PALPITES", OrderedDict())

    # **`n_sampled_features` DISTINTO, e não é cosmética — a primeira escrita deste teste reprovou
    # por causa disto.** `_despacha` deduplica por `_chave_do_trabalho(operacao, matriz, campos)`,
    # e os testes acima já correram esta matriz com os mesmos campos: o trabalho vinha `reclamado`
    # e o corpo devolvido era o do resultado ANTERIOR, computado com a política real. A planta
    # aterrava no processo e não alcançava o resultado, e produzia `palpite='serrf'` numa
    # asserção que espera `None`.
    #
    # Um campo diferente faz a chave diferir, e o trabalho nasce. **Não se limpa o registro
    # global**: o teste que mexesse em `TRABALHOS` passaria a depender da ordem de execução, que é
    # exactamente o que esta linha evita.
    corpo = reclama(
        cliente,
        "/matrizes/medir-corretor",
        files=_arquivos(MODO),
        data={"mode": MODO, "n_sampled_features": "11"},
        teto_s=120,
    ).json()

    assert corpo["palpite"] is None, f"a planta não aterrou: palpite={corpo['palpite']!r}"
    assert corpo["palpite_acertou"] is None, (
        "palpite ausente virou `False`, e isso conta como erro do modelo uma execução em que ele "
        "não foi consultado"
    )
    # E a MEDIÇÃO sobrevive à falta do palpite — é ela o resultado, e ele é acréscimo.
    assert corpo["vencedor"] == "none"


def test_a_populacao_volta_no_corpo_e_nao_fica_no_arquivo(
    cliente: TestClient, oraculo_rapido: list[tuple[str, ...]]
) -> None:
    """`n_features_medidas`: a população como campo de resposta.

    Um número medido sobre uma amostra de features e apresentado como propriedade do modo já
    custou uma correção de escopo publicada. Aqui a população viaja com o número.
    """
    corpo = reclama(
        cliente,
        "/matrizes/medir-corretor",
        files=_arquivos(MODO),
        data={"mode": MODO, "n_sampled_features": "40"},
        teto_s=120,
    ).json()

    assert corpo["n_features_medidas"] == 40, (
        f"a população pedida não voltou no corpo: {corpo['n_features_medidas']!r}"
    )


def test_sem_amostragem_a_populacao_e_declarada_como_inteira(
    cliente: TestClient, oraculo_rapido: list[tuple[str, ...]]
) -> None:
    """A outra direção: sem `n_sampled_features` o campo é `None`, e `None` significa POPULAÇÃO
    INTEIRA — não «não sei».

    Sem este, o teste irmão passaria com um campo que ecoasse qualquer coisa.
    """
    corpo = reclama(
        cliente, "/matrizes/medir-corretor", files=_arquivos(MODO), data={"mode": MODO}, teto_s=120
    ).json()

    assert corpo["n_features_medidas"] is None


def test_a_ressalva_da_politica_viaja_com_a_medicao(
    cliente: TestClient, oraculo_rapido: list[tuple[str, ...]]
) -> None:
    """O par mostra o palpite, então a ressalva dele tem de vir junto — senão o número do palpite
    aparece na tela do resultado sem o que o desautoriza."""
    corpo = reclama(
        cliente, "/matrizes/medir-corretor", files=_arquivos(MODO), data={"mode": MODO}, teto_s=120
    ).json()

    # IGUALDADE e não `"70,0%" in`, porque o número sozinho não separa as ressalvas das
    # políticas. O palpite do par é o Palpite MEDIDO, e a ressalva que viaja é a dele: a frase é
    # a de `--auto medida`, e o número é o da concordância nas 88 matrizes simuladas.
    assert corpo["ressalva_da_politica"] == AUTO_CAVEATS[AUTO_MEDIDA].resumo, corpo[
        "ressalva_da_politica"
    ]
    assert "92,5%" in corpo["ressalva_da_politica"], corpo["ressalva_da_politica"]


@pytest.mark.slow
def test_medicao_real_de_um_modo(cliente: TestClient) -> None:
    """O caminho REAL, uma vez, sem duplo.

    Os testes acima provam a **fiação**; este prova que o motor por trás dela produz um resultado
    coerente. Sem ele, o duplo estaria a medir a si próprio.

    **`slow` porque são 21 ajustes de corretor**, e `n_sampled_features` corta o custo da métrica
    sem tocar no dos ajustes — o teto de parede é generoso de propósito.

    **A asserção é de COERÊNCIA e não de valor**, e a distinção é deliberada: fixar o vencedor do
    `neg_c18` aqui congelaria um número que a régua mede. O que se afirma é que o vencedor é o
    menor score — propriedade do desenho.
    """
    resposta = reclama(
        cliente,
        "/matrizes/medir-corretor",
        files=_arquivos(MODO),
        data={"mode": MODO, "n_sampled_features": "40"},
        teto_s=3600,
    )

    assert resposta.status_code == status.HTTP_200_OK, resposta.text
    corpo = resposta.json()

    scores = corpo["scores"]
    assert set(scores) == {"none", "loess", "serrf"}
    finitos = {k: v for k, v in scores.items() if v == v}
    assert finitos, "ÂNCORA: todos os scores saíram indefinidos; a medição não mediu nada"
    assert corpo["vencedor"] == min(finitos, key=lambda k: finitos[k]), (
        f"o vencedor não é o menor score: vencedor={corpo['vencedor']!r} scores={scores}"
    )


# ---------------------------------------------------------------------------
# O substituto da isenção ao proxy estático de `test_api_progresso.py`
# ---------------------------------------------------------------------------


def test_o_gancho_avisa_uma_vez_por_corretor_e_nao_uma_vez_so() -> None:
    """A propriedade que o proxy estático não consegue ler, medida em runtime.

    `test_api_progresso.py` conta chamadas **textuais** da marca. Esta rota chama-a uma vez, num
    fecho que o oráculo invoca por prefixo — o texto diz 1 e o comportamento é 3. A rota está
    isenta lá, com o motivo declarado, e este teste é o que a isenção promete.

    **Não é vácuo, e é isso que autoriza a isenção.** A objeção que torna o portão irmão estático
    é que `conclui` força `feito` a `total` no estado terminal, então afirmar `feito == total`
    passaria com o defeito presente. Contar **invocações do gancho** não sofre disso: ninguém as
    força, e uma implementação que avisasse uma vez só devolveria 1 aqui.
    """
    from refino.decision.labeling import LabelingOracle

    matriz = load_matrix(measurements_path(MODO), sample_list_path(MODO), mode=MODO)
    avisados: list[tuple[str, int]] = []

    # O oráculo com UM prefixo por corretor, mas sem correr corretor nenhum: o que se mede aqui é
    # a cadência do gancho, e um `NoneCorrector` de verdade em três prefixos custaria minutos.
    oraculo = LabelingOracle(n_sampled_features=8)
    rotulo = oraculo.label(matriz, depois_de_cada=avisados.append)

    assert rotulo is not None, "ÂNCORA: o oráculo não devolveu rótulo; nada foi medido"
    assert len(avisados) == len(oraculo.prefixes), (
        f"o gancho avisou {len(avisados)} vezes para {len(oraculo.prefixes)} corretores: {avisados}"
    )
    assert [correcao for correcao, _lotes in avisados] == [
        correcao for correcao, _lotes in oraculo.prefixes
    ], f"a ordem dos avisos não é a dos prefixos: {avisados}"


def test_o_par_causal_sem_o_gancho_o_oraculo_nao_avisa_ninguem() -> None:
    """A outra direção: `depois_de_cada=None` não pode levantar nem inventar avisos.

    Sem isto, o teste acima poderia estar a afirmar sobre um gancho que o oráculo chama sempre —
    e os scripts do corpus, que passam `None`, quebrariam.
    """
    from refino.decision.labeling import LabelingOracle

    matriz = load_matrix(measurements_path(MODO), sample_list_path(MODO), mode=MODO)
    rotulo = LabelingOracle(n_sampled_features=8).label(matriz)

    assert rotulo.estimator == "leave_one_qc_out_rsd"
    assert set(correcao for correcao, _lotes in rotulo.scores) == {"none", "loess", "serrf"}


def test_as_etapas_declaradas_pela_rota_sao_uma_por_corretor() -> None:
    """O que o literal inline abre, e este teste fecha.

    A tupla de etapas é escrita à mão em `rotas.py` porque o proxy estático precisa de a **ler**.
    O preço é poder divergir de `LabelingOracle.prefixes` em silêncio — prefixo novo sem etapa
    nova faria a barra parar antes do fim. Aqui as duas contagens são comparadas.
    """
    import re
    from pathlib import Path

    from refino.decision.labeling import LabelingOracle

    fonte = (Path(__file__).resolve().parents[1] / "src" / "refino" / "api" / "rotas.py").read_text(
        encoding="utf-8"
    )
    achado = re.search(
        r'\(\s*"corretor none",\s*"corretor loess",\s*"corretor serrf",?\s*\)', fonte
    )

    assert achado is not None, (
        "ÂNCORA: a tupla de etapas da medição não foi encontrada em `rotas.py`. Ou ela mudou de "
        "forma — e o proxy estático de `test_api_progresso.py` cegou junto —, ou este teste está "
        "a ler o arquivo errado"
    )
    declaradas = len(re.findall(r'"[^"]*"', achado.group(0)))
    assert declaradas == len(LabelingOracle().prefixes), (
        f"a rota declara {declaradas} etapas e o oráculo mede {len(LabelingOracle().prefixes)} "
        "corretores; a barra pararia antes do fim"
    )


# ---------------------------------------------------------------------------
# ITEM 83 — o tempo da medição passa a ficar em `runs` (25/09/2026)
# ---------------------------------------------------------------------------


def test_item_83_a_medicao_grava_o_tempo_em_runs(
    oraculo_rapido: list[tuple[str, ...]],
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`POST /matrizes/medir-corretor` com SQLite deixa UMA linha em `runs`, com operação e duração.

    ## O que se via antes do conserto

    Nos quatro modos a corrida terminava `ok` e o log trazia *"medir-corretor terminou como ok e o
    tempo dela NAO foi gravado: ValueError: `operacao` de artefato inválida"*: a rota despachava
    com `checksum`, o envoltório tentava gravar, e `validate_operacao_de_artefato` recusava. O
    resultado saía certo — o que se perdia era o custo da rota mais cara do produto.

    **Visto reprovar com o `Literal` antigo, em 25/09/2026**, na asserção do `stderr`, que vem
    primeiro de propósito: ela traz a `ValueError` inteira na mensagem, e a de contagem só diria
    *zero linhas* sem dizer porquê.

    ## Por que o duplo do oráculo serve aqui

    O que se prova é a **fiação até ao disco**, e ela é a mesma com o oráculo real ou com o duplo:
    o envoltório envolve o `trabalho`, e o duplo invoca o gancho uma vez por prefixo como o real.
    A duração gravada é a do duplo, e por isso não se afirma nada sobre o valor dela além da ordem.

    **`n_sampled_features=17` é o que torna o trabalho NOVO** e não a reclamação de outro teste
    deste arquivo: `TRABALHOS` é global do processo, e sem uma chave própria nenhuma thread
    nasceria, nada seria gravado, e o teste reprovaria por ordem de execução da suíte.
    """
    import json

    from sqlalchemy import create_engine, text

    from refino.persistence.config import VARIAVEL_DA_URL
    from refino.persistence.schema import METADATA

    url = f"sqlite:///{tmp_path / 'refino.db'}"
    monkeypatch.delenv("REFINO_DB_HOST", raising=False)
    monkeypatch.setenv(VARIAVEL_DA_URL, url)
    METADATA.create_all(create_engine(url))
    capsys.readouterr()

    resposta = reclama(
        TestClient(build_app()),
        "/matrizes/medir-corretor",
        files=_arquivos(MODO),
        data={"mode": MODO, "n_sampled_features": "17", "paralelo": "false"},
        teto_s=120,
    )

    assert resposta.status_code == status.HTTP_200_OK, resposta.text
    assert pedidos_do_oraculo(oraculo_rapido), (
        "ÂNCORA: o oráculo não foi chamado — o trabalho foi reclamado de outro teste e nenhuma "
        "thread correu, então a ausência de linha abaixo não diria nada sobre a gravação"
    )
    erro = capsys.readouterr().err
    assert "NAO foi gravado" not in erro, f"a rota relatou perda do registro: {erro!r}"

    with create_engine(url).connect() as c:
        linhas = c.execute(
            text(
                "SELECT r.spec_json, r.status, r.seed, r.iniciado_em, r.terminado_em, "
                "r.etapas_json, m.mode FROM runs r JOIN matrices m ON m.id = r.matrix_id"
            )
        ).all()

    assert len(linhas) == 1, f"a rota deixou {len(linhas)} linha(s) em `runs`"
    spec_json, estado, semente, inicio, fim, etapas_json, mode_gravado = linhas[0]
    spec = json.loads(spec_json)
    assert spec["operacao"] == "medir-corretor", spec
    assert estado == "ok", estado
    assert mode_gravado == MODO, mode_gravado
    # **A sentinela e não `42`**: o pedido não escolhe semente — o oráculo usa a padrão do
    # estimador —, e a coluna registaria como escolha de quem pediu o que é constante de código.
    assert semente == -1, semente
    assert "seed" not in spec["parametros"], spec["parametros"]
    # **A irmã das três rotas de arquivo**: sem o regime de núcleos a duração não compara máquina
    # nenhuma. `paralelo=false` dá série em qualquer processo, então o valor é determinado.
    assert spec["parametros"]["n_jobs"] == 1, spec["parametros"]
    assert spec["parametros"]["n_sampled_features"] == 17, spec["parametros"]

    def _dt(valor: Any) -> Any:
        from datetime import datetime

        return valor if isinstance(valor, datetime) else datetime.fromisoformat(str(valor))

    assert _dt(fim) >= _dt(inicio), (inicio, fim)
    assert etapas_json is not None, "a linha diz quanto demorou e nunca em quê"
    assert set(json.loads(etapas_json)) == {"corretor none", "corretor loess", "corretor serrf"}


def test_a_semente_pedida_chega_a_regua_e_o_padrao_e_42(
    cliente: TestClient, oraculo_rapido: list[tuple[str, ...]]
) -> None:
    """`seed` da rota vai ao oráculo; sem ela, a régua usa 42.

    Par causal sobre a mesma matriz, com `n_sampled_features` distinto em cada pedido para o
    `_despacha` não devolver o trabalho anterior sem chamar o oráculo (a mesma armadilha da
    âncora do teste dos três corretores).
    """
    sementes = oraculo_rapido.sementes  # type: ignore[attr-defined]
    reclama(
        cliente,
        "/matrizes/medir-corretor",
        files=_arquivos(MODO),
        data={"mode": MODO, "n_sampled_features": "97", "seed": "7"},
        teto_s=120,
    )
    reclama(
        cliente,
        "/matrizes/medir-corretor",
        files=_arquivos(MODO),
        data={"mode": MODO, "n_sampled_features": "99"},
        teto_s=120,
    )
    assert sementes == [7, 42], sementes
