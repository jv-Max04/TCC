"""O Palpite MEDIDO na janela.

O botão «Palpite» passou a pedir `POST /matrizes/palpite`, que mede o RSD dos QCs fora da amostra em
200 features e responde `202`; e a caixa «Automático» passou a seguir essa medição. Este arquivo
cobre as duas metades do lado da janela:

- **a leitura longa** — o `202` encadeia a espera, o rótulo mostra o progresso por corretor, o corpo
  chega ao painel e é guardado como proposta, e o corretor é pré-preenchido;
- **a espera das sugestões** — sem Palpite em dia para os arquivos e os lotes da tela, «Seguir
  sugestões do app» mede primeiro e põe o corretor dele no seletor quando ele chega; com o Palpite
  em dia, põe na hora; e a Correção sai no clique, sem esperar nada. **Par causal nas três.**

A segunda metade era a espera antes do disparo delegado: com a caixa «Automático» ticada, a
Correção media o Palpite primeiro e saía sozinha com `auto=medida`. A caixa saiu, a janela
deixou de delegar, e o mesmo mecanismo — a identidade do Palpite, a retomada e o
que a larga — passou a servir o botão das sugestões.

**A medição de verdade não corre aqui** — custa minutos por arquivo. O servidor em processo usa um
duplo de `mede_o_palpite` que decide pela regra do produto e avisa o gancho uma vez por corretor; a
medição em si tem teste contra os vetores gravados em `tests/unit/test_palpite_medido.py`.

## Escopo negativo

**Não é o executável congelado** nem o Windows: a ligação é exercitada no Qt `offscreen`, com a API
em processo. E nada aqui corre um SERRF — o duplo propõe `none`, para a execução retomada terminar
em segundos.
"""

from __future__ import annotations

import json
import time
from collections import OrderedDict
from typing import Any

import pytest
from fastapi.testclient import TestClient

from tests.qt_guard import garante_plataforma_offscreen, membros_ausentes, sonda_o_ambiente

_FALTAM = membros_ausentes(sonda_o_ambiente())
if _FALTAM:  # pragma: no cover — a decisão é exercitada em `tests/test_qt_guard.py`, com planta
    pytest.skip(
        f"grupo opcional `desktop` ausente: falta {', '.join(_FALTAM)}. "
        'Instale com `uv pip install -e ".[desktop]"` para exercitar os testes de janela.',
        allow_module_level=True,
    )

garante_plataforma_offscreen()

import refino.api.rotas as rotas  # noqa: E402
from refino.api.app import build_app  # noqa: E402
from refino.decision.palpite import decide_o_palpite  # noqa: E402
from refino.desktop.etapas import ROTA_DA_ETAPA  # noqa: E402
from refino.desktop.janela.arquivos import ParDeArquivos  # noqa: E402
from refino.desktop.janela.leitura import (  # noqa: E402
    CHAVE_DO_PALPITE,
    ROTA_DO_PALPITE,
    ROTAS_DA_LEITURA,
    texto_da_recomendacao,
)
from refino.desktop.janela.pipeline import VARIANTE_DO_RSD_EM_REPOUSO  # noqa: E402
from refino.desktop.janela.principal import (  # noqa: E402
    PALPITE_EM_CURSO,
    JanelaPrincipal,
)
from refino.interface.cliente import Resposta  # noqa: E402
from tests.conftest import measurements_path, sample_list_path  # noqa: E402

# **Memória fria a cada teste**: o registro de trabalhos retém os concluídos, e o Palpite do teste
# anterior seria RECLAMADO em vez de medido — a contagem de medições mediria o resíduo.
pytestmark = pytest.mark.usefixtures("memoria_de_correcoes_privada")

MODO = "neg_c18"
# `none` à frente, e por margem larga: a execução retomada corre o corretor que não corrige, e
# termina em segundos.
ESCORES = {"none": 0.10, "loess": 0.30, "serrf": 0.20}


class _Transporte:
    """O `TestClient` que GRAVA a rota de cada chamada, na ordem — a mesma forma do arnês irmão."""

    def __init__(self, real: TestClient) -> None:
        self._real = real
        self.rotas: list[str] = []
        self.campos: list[dict[str, Any]] = []

    def post(self, url: str, **kwargs: Any) -> Any:
        self.rotas.append(url)
        self.campos.append(dict(kwargs.get("data") or {}))
        return self._real.post(url, **kwargs)

    def get(self, url: str, **kwargs: Any) -> Any:
        self.rotas.append(url)
        self.campos.append({})
        return self._real.get(url, **kwargs)


def _instala_o_medidor(
    monkeypatch: pytest.MonkeyPatch, escores: dict[str, float] = ESCORES
) -> list[str]:
    """O duplo de `mede_o_palpite` no servidor em processo. Devolve os modos medidos."""
    medidos: list[str] = []

    def medidor(
        matriz: Any,
        *,
        batches: Any = None,
        n_jobs: int = -1,
        depois_de_cada: Any = None,
        cancelled: Any = None,
        seed: int = 42,
    ) -> Any:
        medidos.append(str(matriz.mode))
        for corretor in ("none", "loess", "serrf"):
            if depois_de_cada is not None:
                depois_de_cada((corretor, 1))
        return decide_o_palpite(escores, n_features_medidas=200)

    monkeypatch.setattr(rotas, "mede_o_palpite", medidor)
    monkeypatch.setattr(rotas, "_PALPITES", OrderedDict())
    return medidos


@pytest.fixture
def medicoes(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    return _instala_o_medidor(monkeypatch)


def _janela(qtbot: Any, modo: str = MODO) -> tuple[JanelaPrincipal, _Transporte]:
    transporte = _Transporte(TestClient(build_app(), raise_server_exceptions=False))
    janela = JanelaPrincipal(transporte, escolhedor=lambda: None)
    qtbot.addWidget(janela)
    janela.intervalo_da_espera_ms = 0
    medidas, lista = measurements_path(modo), sample_list_path(modo)
    janela.arquivos._par = ParDeArquivos(
        medidas.name, medidas.read_bytes(), lista.name, lista.read_bytes()
    )
    janela.arquivos.modo.setCurrentText(modo)
    return janela, transporte


def _leituras_em_dia(janela: JanelaPrincipal) -> None:
    """A Validação e o Diagnóstico da tela em dia, entregues por `_recebe`.

    O cartão do Palpite e «Seguir sugestões do app» rodam antes a Validação e o Diagnóstico que
    faltarem. Os testes cujo assunto é o Palpite, e não a cadeia, entregam as duas antes do clique,
    e o Palpite sai no clique como saía. Os corpos vêm das rotas,
    num cliente à parte — o `_Transporte` desta janela grava só o que ela pede. **O do Diagnóstico
    vai sem `lotes`**: o real do `neg_c18` sugere dois lotes, e com ele o Palpite mediria noutra
    partição que a que estes testes afirmam; quando o teste quer uma sugestão, ele a põe.
    """
    cliente = TestClient(build_app())
    for chave in ("validacao", "diagnostico"):
        http = cliente.post(
            dict(ROTAS_DA_LEITURA)[chave],
            files=janela.arquivos.arquivos_para_envio(),
            data=janela.arquivos.campos_para_envio(),
        )
        assert http.status_code == 200, http.text
        corpo = http.json()
        corpo.pop("lotes", None)
        janela._recebe(
            chave,
            Resposta(codigo=200, ok=True, causa=None, mensagem="OK", corpo=corpo, conteudo=None),
        )
        assert janela._leitura_em_dia(chave) is True, f"âncora: `{chave}` não ficou em dia"


def _encerra(janela: JanelaPrincipal) -> None:
    """Espera a chamada em voo voltar e para as esperas — sem isto, ela volta a uma janela morta."""
    assert janela.espera_a_piscina(60_000), "a chamada em voo não voltou em 60 s"
    janela.para_as_esperas()


def _gira_ate(janela: JanelaPrincipal, condicao: Any, ms: int = 60_000) -> None:
    """Gira o laço de eventos até a condição, por `processEvents` — a forma do `_gira_ate` irmão.

    O irmão, em `tests/test_desktop_janela.py`, gira do mesmo jeito; este espera a piscina a cada
    volta. Não passa por `QEventLoop.exec()`, e por isso não dependia da limpeza do `Quit` que o
    laço principal deixava na fila.
    """
    from PySide6.QtWidgets import QApplication

    limite = time.monotonic() + ms / 1000
    while time.monotonic() < limite:
        janela.espera_a_piscina(20)
        QApplication.processEvents()
        if condicao():
            return
    raise AssertionError(f"a condição não ocorreu em {ms} ms")


# ---------------------------------------------------------------------------
# A leitura longa
# ---------------------------------------------------------------------------


def test_o_botao_do_palpite_mede_pela_rota_longa_guarda_e_nao_pre_preenche(
    qtbot: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """O clique vai a `/matrizes/palpite` com os lotes e os núcleos, e o corpo vira a proposta.

    **O duplo propõe `loess` aqui**, e não o `none` dos outros testes: o seletor repousa em `none`,
    e uma proposta igual ao repouso não separaria escrever de não escrever.

    Era `..._guarda_e_pre_preenche`: o corretor do Palpite entrava no seletor ao chegar. A aba
    fica nos valores padrão, e ele fica guardado até «Seguir sugestões do app».

    O clique roda antes a Validação e o Diagnóstico, de verdade, e o Palpite sai na partição que o
    Diagnóstico sugeriu. Antes o clique pedia só o Palpite, num lote: a asserção era
    `n_batches == 1`, e
    passou a ser a da sugestão, com a âncora de que ela não é um lote — senão a comparação não
    separaria nada.
    """
    medicoes = _instala_o_medidor(monkeypatch, {"none": 0.30, "loess": 0.10, "serrf": 0.20})
    janela, transporte = _janela(qtbot)
    assert janela.pipeline.corretor.currentText() != "loess", "âncora: o repouso já era a proposta"
    assert janela.leitura.recomendacao_guardada() is None, "âncora: nasceu com proposta"

    janela.leitura.botoes[CHAVE_DO_PALPITE].click()
    _gira_ate(janela, lambda: janela.leitura.recomendacao_guardada() is not None)

    assert medicoes == [MODO], f"o servidor não mediu uma vez: {medicoes}"
    posts = [r for r in transporte.rotas if not r.startswith("/trabalhos/")]
    assert posts == [r for _chave, r in ROTAS_DA_LEITURA], f"a cadeia não seguiu a ordem: {posts}"
    disparo = transporte.campos[transporte.rotas.index(ROTA_DO_PALPITE)]
    n_lotes, cortes = janela.pipeline.particao_sugerida()
    assert n_lotes > 1 and cortes, "âncora: o Diagnóstico do `neg_c18` não sugeriu lotes"
    assert disparo["n_batches"] == n_lotes and "paralelo" in disparo, disparo
    assert json.loads(disparo["batch_cuts"]) == list(cortes), disparo
    assert any(r.startswith("/trabalhos/") for r in transporte.rotas), "a espera não leu o trabalho"
    guardada = janela.leitura.recomendacao_guardada()
    assert guardada is not None and guardada["correction_proposta"] == "loess"
    assert guardada["n_features_medidas"] == 200
    assert janela.pipeline.corretor.currentText() == "none", "o Palpite escreveu no seletor"
    assert janela.pipeline.corretor_sugerido == "loess", "a sugestão do Palpite não ficou guardada"
    texto = janela.leitura.rotulos[CHAVE_DO_PALPITE].text()
    assert texto.startswith("**Pela estabilidade dos QCs**") and "loess." in texto, texto


def test_o_progresso_do_palpite_aparece_no_rotulo_da_leitura(qtbot: Any) -> None:
    """O mostrador da espera do Palpite é o painel da leitura, e a frase conta corretores."""
    janela, _ = _janela(qtbot)
    texto = janela._mostrador_da_espera(CHAVE_DO_PALPITE).mostra_espera(
        CHAVE_DO_PALPITE, feito=1, total=3, leituras=2
    )
    assert "**1 de 3** corretores" in texto
    assert janela.leitura.rotulos[CHAVE_DO_PALPITE].text() == texto
    # E no que está à vista. O rótulo acima vive em «Detalhes», que o cartão recolhe:
    # o texto dele certo não diz que a pessoa o vê. O selo e a manchete são o que ela vê.
    cartao = janela.leitura.cartoes[CHAVE_DO_PALPITE]
    assert cartao.selo.text() == "Medindo"
    assert cartao.manchete.text() == "1 de 3 corretores"
    janela.show()
    assert cartao.manchete.isVisibleTo(janela), "a manchete do progresso não está à vista"


def test_a_recusa_da_rota_e_frase_e_apaga_a_proposta_anterior(qtbot: Any) -> None:
    """Um `422` não é um Palpite chamado `None` — e não fica guardado como proposta.

    **Pelo painel, e não só pelo formatador:** quem guarda e quem apaga é
    `PainelDeLeitura.mostra`, e a proposta guardada é a que `/execucoes` compara. A recusa que chega
    depois de um Palpite apaga o anterior, porque ele era de outra leitura.
    """
    recusa = {"detail": "com 2 lote(s), os QCs por lote são [5, 2]"}
    assert texto_da_recomendacao(recusa) == f"O Palpite não foi medido: {recusa['detail']}"
    medido = {
        "correction_proposta": "loess",
        "avaliavel": True,
        "abstem": True,
        "confianca": 0.2,
        "margem": 0.0749,
        "limiar_de_empate": 0.167,
        "n_features_medidas": 200,
    }
    texto = texto_da_recomendacao(medido)
    # A segunda frase abre em maiúscula, na convenção da interface.
    assert "**Empate na amostra**" in texto and "7,5%" in texto and "16,7%" in texto, texto
    assert "política" not in texto, "a abstenção do Palpite medido falou como a da árvore"

    janela, _transporte = _janela(qtbot)
    janela.leitura.mostra(CHAVE_DO_PALPITE, medido)
    assert janela.leitura.recomendacao_guardada() == medido, "âncora: o Palpite não foi guardado"
    janela.leitura.mostra(CHAVE_DO_PALPITE, recusa)
    assert janela.leitura.recomendacao_guardada() is None, (
        "a recusa ficou guardada como proposta, ou deixou a anterior no lugar"
    )


def test_o_palpite_com_um_corretor_recusado_diz_que_ele_nao_foi_medido_e_por_que() -> None:
    """Com um lote de três QCs o LOESS fica sem número, e o cartão do Palpite —
    que só mostra a proposta e a confiança — diz que ele não entrou na comparação e o motivo, em vez
    de deixar a proposta parecer escolhida entre os três. Sem `recusas`, o texto é o de antes."""
    motivo = "menos de 4 QCs num lote (0=3 QCs): cada ajuste fora da amostra treina com um a menos"
    medido = {
        "correction_proposta": "serrf",
        "avaliavel": True,
        "abstem": False,
        "confianca": 0.9,
        "margem": 0.5,
        "limiar_de_empate": 0.167,
        "n_features_medidas": 200,
    }
    sem = texto_da_recomendacao(medido)
    com = texto_da_recomendacao({**medido, "recusas": {"loess": motivo}})
    assert "não foi medido" not in sem
    assert com.startswith(sem), "a recusa mudou a primeira linha do Palpite"
    assert f"O `loess` não foi medido: {motivo}" in com, com


def test_o_palpite_nada_avaliavel_com_um_recusado_nao_diz_tres_corretores() -> None:
    """A irmã da justificativa, no cartão: sem escore finito e com o `loess`
    recusado, a confiança dizia *«indefinido nos três corretores»* e a frase seguinte dizia que o
    `loess` não foi medido — as duas no mesmo parágrafo. O par causal: sem `recusas`, a frase dos
    três continua."""
    motivo = "menos de 4 QCs num lote (0=3 QCs): cada ajuste fora da amostra treina com um a menos"
    nada = {
        "correction_proposta": "none",
        "avaliavel": False,
        "abstem": True,
        "confianca": 0.0,
        "margem": None,
        "limiar_de_empate": 0.167,
        "n_features_medidas": 200,
    }
    sem = texto_da_recomendacao(nada)
    com = texto_da_recomendacao({**nada, "recusas": {"loess": motivo}})
    assert "nos três corretores" in sem, sem
    assert "três" not in com, com
    assert "indefinido em todos os corretores medidos" in com, com
    assert f"O `loess` não foi medido: {motivo}" in com, com


# ---------------------------------------------------------------------------
# A espera das sugestões pelo Palpite — par causal
# ---------------------------------------------------------------------------
#
# Antes, a espera era a do disparo delegado: com a caixa «Automático» ticada, o clique da
# Correção media o Palpite antes e saía sozinho quando ele chegava. A caixa saiu, a Correção leva o
# corretor do seletor e sai no clique, e quem espera o Palpite é «Seguir sugestões do app». Os
# testes abaixo eram os do disparo delegado e foram reescritos para o botão, com o mesmo mecanismo:
# a identidade do Palpite, a retomada e o que a larga.


def _bloco(janela: JanelaPrincipal) -> Any:
    bloco = janela.pagina_da_correcao.sugestoes
    assert bloco is not None, "âncora: a Correção ficou sem o botão das sugestões"
    return bloco


def _motivo_do_botao(bloco: Any) -> str:
    """A frase ao lado do botão desligado — o rótulo de `desliga_com_motivo`."""
    from PySide6.QtWidgets import QLabel

    rotulos = [r for r in bloco.findChildren(QLabel, "motivo_do_botao") if not r.isHidden()]
    return rotulos[0].text() if rotulos else ""


def test_seguir_sugestoes_sem_palpite_mede_e_aplica_quando_ele_chega(
    qtbot: Any, medicoes: list[str]
) -> None:
    """Sem Palpite em dia, o clique mede primeiro; quando ele chega, o corretor dele entra no
    seletor por cima da escolha da pessoa — e a Correção não sai.

    Era `test_com_automatico_a_correcao_mede_o_palpite_antes_e_retoma`: o clique da Correção, com
    a caixa ticada, esperava o Palpite e saía sozinho com `auto=medida`. Quem espera agora é o
    botão das sugestões, e o desfecho é o seletor preenchido, e não uma correção.

    O clique roda antes a Validação e o Diagnóstico, de verdade, e a frase ao lado do botão diz a
    etapa. Antes o Palpite saía no clique e a frase era `MEDINDO_A_SUGESTAO`; e sem Diagnóstico
    não havia lotes a pôr — a última asserção dizia que a falta da proposta de lotes era dita.
    Agora o Diagnóstico real do `neg_c18` propõe lotes, eles entram, e a frase os diz.
    """
    janela, transporte = _janela(qtbot)
    bloco = _bloco(janela)
    janela.pipeline.corretor.setCurrentText("serrf")
    assert janela._palpite_em_dia is None, "âncora: o Palpite já estava em dia"

    bloco.botao.click()
    assert not bloco.botao.isEnabled(), "o botão ficou ligado enquanto o Palpite mede"
    assert _motivo_do_botao(bloco).startswith("Etapa 1 de 3: validando os arquivos. ")
    assert janela.pipeline.corretor.currentText() == "serrf", "a sugestão entrou antes do Palpite"

    _gira_ate(janela, lambda: bloco.botao.isEnabled())
    _encerra(janela)

    posts = [r for r in transporte.rotas if not r.startswith("/trabalhos/")]
    assert posts == [r for _chave, r in ROTAS_DA_LEITURA], f"a cadeia não seguiu a ordem: {posts}"
    assert ROTA_DA_ETAPA[1] not in transporte.rotas, "seguir as sugestões disparou a Correção"
    assert medicoes == [MODO]
    assert janela.pipeline.corretor.currentText() == "none", "o corretor do Palpite não entrou"
    n_lotes, _cortes = janela.pipeline.particao_sugerida()
    assert n_lotes > 1, "âncora: o Diagnóstico do `neg_c18` não sugeriu lotes"
    assert janela.pipeline.n_lotes.value() == n_lotes, "os lotes do Diagnóstico não entraram"
    assert bloco.desfecho.text() == (
        f"Sugestões aplicadas: corretor «Sem correção», {n_lotes} lotes."
    ), bloco.desfecho.text()


def test_com_o_palpite_em_dia_as_sugestoes_entram_sem_medir_de_novo(
    qtbot: Any, medicoes: list[str]
) -> None:
    """A outra metade do par: o Palpite que já chegou para estes arquivos e lotes não se repete."""
    janela, transporte = _janela(qtbot)
    janela.leitura.botoes[CHAVE_DO_PALPITE].click()
    _gira_ate(janela, lambda: janela.leitura.recomendacao_guardada() is not None)
    _gira_ate(janela, lambda: not janela.esperas_em_curso())
    antes = len(transporte.rotas)
    janela.pipeline.corretor.setCurrentText("loess")

    _bloco(janela).botao.click()
    _encerra(janela)

    assert ROTA_DO_PALPITE not in transporte.rotas[antes:], "o Palpite em dia foi medido de novo"
    assert medicoes == [MODO], f"o servidor mediu mais de uma vez: {medicoes}"
    assert janela.pipeline.corretor.currentText() == "none"


def test_os_lotes_da_correcao_nao_tiram_o_palpite_de_dia_e_os_sugeridos_tiram(
    qtbot: Any, medicoes: list[str]
) -> None:
    """O Palpite é da partição SUGERIDA: o campo da aba «Correção» não o tira de dia.

    Par causal: mudar «Quantos lotes declarar» deixa a identidade igual; um «Diagnóstico» que
    sugere outro número a muda. Era `test_mudar_os_lotes_pede_outro_palpite`: antes a identidade
    lia o campo da aba, e o Palpite passou a ser independente dela.

    A Validação e o Diagnóstico entram em dia antes do clique
    (`_leituras_em_dia`, sem sugestão de lotes): o Palpite é medido num lote, como era, e o
    «Diagnóstico» que sugere dois é o único que muda a partição. Com o Diagnóstico real em dia, ele
    já sugeriria dois, e a sugestão plantada mudaria só os cortes.
    """
    janela, _ = _janela(qtbot)
    _leituras_em_dia(janela)
    janela.leitura.botoes[CHAVE_DO_PALPITE].click()
    _gira_ate(janela, lambda: janela.leitura.recomendacao_guardada() is not None)
    _gira_ate(janela, lambda: not janela.esperas_em_curso())
    em_dia = janela._identidade_do_palpite()
    assert em_dia is not None and janela._palpite_em_dia == em_dia, "âncora: o Palpite não chegou"

    janela.pipeline.n_lotes.setValue(2)
    assert janela._identidade_do_palpite() == em_dia, "os lotes da aba tiraram o Palpite de dia"

    janela._pre_preenche_com_a_proposta("diagnostico", {"lotes": {"n_lotes": 2}})
    assert janela._identidade_do_palpite() != em_dia, "a identidade ignorou os lotes sugeridos"


def test_a_correcao_sai_no_clique_sem_esperar_o_palpite(qtbot: Any, medicoes: list[str]) -> None:
    """Sem Palpite medido, a Correção sai no clique, com o corretor do seletor e sem `auto`.

    Era `test_com_a_escolha_manual_nada_espera_o_palpite` — o lado manual do par da
    caixa «Automático» —, e absorveu `test_com_automatico_a_etapa_1_mede_o_palpite_antes_e_retoma`:
    sem a caixa, o botão e `_dispara_etapa` são o mesmo caminho, e nenhum dos dois espera. O par
    causal é o desenho de antes, em que este mesmo clique media o Palpite primeiro.
    """
    janela, transporte = _janela(qtbot)
    janela.pipeline.corretor.setCurrentText("none")

    janela.pipeline.botoes_de_etapa[1].click()
    assert janela._depois_do_palpite is None, "a Correção ficou à espera do Palpite"
    _gira_ate(janela, lambda: ROTA_DA_ETAPA[1] in transporte.rotas)
    _encerra(janela)

    assert ROTA_DO_PALPITE not in transporte.rotas
    assert medicoes == []
    correcao = transporte.campos[transporte.rotas.index(ROTA_DA_ETAPA[1])]
    assert correcao.get("correction") == "none", correcao
    assert "auto" not in correcao, f"a Correção delegou: {correcao}"


def test_o_palpite_em_curso_trava_a_correcao_com_frase_propria(qtbot: Any) -> None:
    """Uma correção por vez: o Palpite corre os três corretores nos mesmos núcleos.

    E a espera das sugestões trava também, e é ela que cobre o intervalo entre o `POST` do Palpite
    e o primeiro `202`, em que a espera da rota ainda não nasceu; antes era a do disparo delegado.
    """
    janela, _ = _janela(qtbot)
    janela._esperas[CHAVE_DO_PALPITE] = object()  # type: ignore[assignment]
    assert janela._motivo_para_nao_corrigir("execucao") == PALPITE_EM_CURSO
    del janela._esperas[CHAVE_DO_PALPITE]
    assert janela._motivo_para_nao_corrigir("execucao") is None, "âncora: a trava ficou presa"
    janela._depois_do_palpite = lambda: None
    assert janela._motivo_para_nao_corrigir("execucao") == PALPITE_EM_CURSO, (
        "uma correção, antes do `202` do Palpite, passaria pela trava"
    )
    janela._depois_do_palpite = None
    assert janela._motivo_para_nao_corrigir("execucao") is None


def test_o_palpite_que_falha_nao_aplica_e_diz_por_que(qtbot: Any, medicoes: list[str]) -> None:
    """Um lote com um QC só faz o Palpite recusar no disparo, e as sugestões não entram.

    Os QCs do `neg_c18` estão nas injeções 1, 12, 23, 34, 45, 56 e 68; o corte em 60 deixa o 68
    sozinho no segundo lote. O número de lotes vem como PROPOSTA, porque é a condição para os
    cortes estarem em uso — e é a proposta que o botão põe antes de medir.

    A Validação e o Diagnóstico entram em dia antes da sugestão plantada
    (`_leituras_em_dia`, sem sugestão de lotes): sem isso a cadeia rodaria o Diagnóstico real, e a
    sugestão dele — dois lotes cortados em 34 — tomaria o lugar do corte em 60.
    """
    janela, transporte = _janela(qtbot)
    _leituras_em_dia(janela)
    # A sugestão chega pela leitura do «Diagnóstico», de onde o Palpite tira a partição; antes o
    # teste a punha com `pre_preenche`, e o Palpite lia o campo.
    janela._pre_preenche_com_a_proposta(
        "diagnostico", {"lotes": {"n_lotes": 2, "changepoints": [60]}}
    )
    assert janela.pipeline.particao_sugerida() == (2, (60,)), "âncora: a partição não é a sugerida"
    janela.pipeline.corretor.setCurrentText("serrf")
    bloco = _bloco(janela)

    bloco.botao.click()
    _gira_ate(janela, lambda: "não completou" in bloco.desfecho.text())

    assert bloco.botao.isEnabled(), "o botão ficou desligado depois de o Palpite falhar"
    assert janela.pipeline.corretor.currentText() == "serrf", "uma sugestão entrou sem Palpite"
    assert ROTA_DA_ETAPA[1] not in transporte.rotas
    assert medicoes == [], "o Palpite chegou a medir com um lote de dois QCs"
    assert json.loads(transporte.campos[transporte.rotas.index(ROTA_DO_PALPITE)]["batch_cuts"])


def test_no_pos_hilic_a_cadeia_mede_o_palpite_em_lote_unico(
    qtbot: Any, medicoes: list[str]
) -> None:
    """O Palpite do `pos_hilic` pela cadeia, com a Validação e o Diagnóstico reais.

    A detecção corta em 23 e 45, e os lotes ficariam com dois, dois e três QCs, que o Palpite
    recusa medir. A sugestão do Diagnóstico é um lote único, e o Palpite sai e mede nele. O par
    causal é o teste de cima: uma partição com lote de menos de três QCs que chega ao Palpite faz
    a rota recusar, e nada é medido.
    """
    janela, transporte = _janela(qtbot, modo="pos_hilic")

    janela.leitura.botoes[CHAVE_DO_PALPITE].click()
    _gira_ate(janela, lambda: janela.leitura.recomendacao_guardada() is not None)
    _encerra(janela)

    assert medicoes == ["pos_hilic"], f"o Palpite não mediu uma vez: {medicoes}"
    palpite = transporte.campos[transporte.rotas.index(ROTA_DO_PALPITE)]
    assert int(palpite["n_batches"]) == 1, palpite
    assert json.loads(palpite.get("batch_cuts", "[]")) == [], palpite
    assert janela.pipeline.particao_sugerida()[0] == 1


# ---------------------------------------------------------------------------
# A revisão de 23/10/2026: a partição, o lote e a retomada
# ---------------------------------------------------------------------------


def test_as_sugestoes_medem_o_palpite_na_particao_dos_lotes_propostos(
    qtbot: Any, medicoes: list[str]
) -> None:
    """Os lotes propostos entram ANTES de medir, e o Palpite e a Correção levam a mesma partição.

    **A âncora é a contagem do servidor e a escolha da pessoa:** o número de lotes começa noutro
    valor, escolhido à mão, e o botão o devolve ao proposto antes de pedir o Palpite — senão o
    Palpite seria de outra partição. Os QCs do `neg_c18` estão nas injeções 1, 12, 23, 34, 45, 56 e
    68; o corte em 45 deixa quatro e três.

    Era `test_com_automatico_a_correcao_mede_o_palpite_na_particao_que_manda`, em que a
    Correção delegada esperava o Palpite: com cortes diferentes nos dois pedidos, a delegação não
    acharia o Palpite guardado e mediria de novo dentro do pedido. A Correção já não delega, e o que
    a partição comum garante agora é que o corretor sugerido foi medido nos lotes que vão correr.

    A Validação e o Diagnóstico entram em dia antes da sugestão plantada
    (`_leituras_em_dia`, sem sugestão de lotes), pelo mesmo motivo do teste de cima: o corte em 45
    é o que o teste afirma, e o Diagnóstico real o trocaria pelo dele.
    """
    janela, transporte = _janela(qtbot)
    _leituras_em_dia(janela)
    # A sugestão chega pela leitura do «Diagnóstico», que é de onde o Palpite tira a partição
    # (`particao_sugerida`); antes este teste a punha com `pre_preenche`, e o Palpite lia o campo.
    janela._pre_preenche_com_a_proposta(
        "diagnostico", {"lotes": {"n_lotes": 2, "changepoints": [45]}}
    )
    janela.pipeline.n_lotes.setValue(3)
    assert janela.pipeline.cortes_em_uso(3) is None, "âncora: os cortes valeriam com três lotes"
    bloco = _bloco(janela)

    bloco.botao.click()
    assert janela.pipeline.n_lotes.value() == 2, "os lotes propostos não entraram antes de medir"
    _gira_ate(janela, lambda: bloco.botao.isEnabled())
    janela.pipeline.botoes_de_etapa[1].click()
    _gira_ate(janela, lambda: ROTA_DA_ETAPA[1] in transporte.rotas)
    _encerra(janela)

    palpite = transporte.campos[transporte.rotas.index(ROTA_DO_PALPITE)]
    corrigir = transporte.campos[transporte.rotas.index(ROTA_DA_ETAPA[1])]
    assert "auto" not in corrigir, corrigir
    assert corrigir.get("correction") == "none", corrigir
    assert json.loads(corrigir["batch_cuts"]) == json.loads(palpite["batch_cuts"]) == [45]
    assert int(corrigir["n_batches"]) == int(palpite["n_batches"]) == 2
    assert medicoes == [MODO], f"o servidor mediu o Palpite mais de uma vez: {medicoes}"


def test_as_pernas_do_lote_nao_delegam_nem_esperam_o_palpite(
    qtbot: Any, medicoes: list[str], tmp_path: Any
) -> None:
    """«Gerar os três» gera os TRÊS: cada perna manda o corretor dela, sem `auto` e sem Palpite.

    Antes, com a caixa «Automático» ticada, a perna podia delegar — a rota troca o `method`
    pelo corretor do Palpite — e esperaria o Palpite, que a trava do lote recusa: o
    lote abortava na primeira perna. A caixa saiu e a janela não delega mais; o teste fica, porque
    é ele que acusa uma perna que voltasse a delegar. **Só a primeira perna corre aqui**: as outras
    duas são cortadas da fila para o teste não correr um SERRF, e o lote conclui com um arquivo.
    """
    janela, transporte = _janela(qtbot)
    janela._escolhe_pasta = lambda: tmp_path

    # O lote do formato do laboratório, o «Gerar os três» do Resultado da Correção.
    janela._inicia_lote("corrigida")
    lote = janela._lote
    assert lote is not None, f"o lote abortou: {janela.rotulo_de_desfecho.text()}"
    del lote.restantes[1:]
    _gira_ate(janela, lambda: janela._lote is None)
    _encerra(janela)

    assert ROTA_DO_PALPITE not in transporte.rotas, "a perna do lote pediu o Palpite"
    perna = transporte.campos[transporte.rotas.index("/matrizes/corrigir")]
    assert "auto" not in perna, f"a perna do lote delegou: {perna}"
    assert perna.get("method") == "none", perna
    assert medicoes == []
    # **Um CSV**, e não um arquivo: o lote do Resultado da Correção grava também o resumo dos
    # métodos ao lado (a caixa nasce ticada), e o da aba «Executar e exportar», que este teste
    # usava até 09/11/2026, não gravava.
    assert len(list(tmp_path.glob("*.csv"))) == 1, janela.rotulo_de_desfecho.text()


def test_mudar_os_lotes_durante_o_palpite_nao_o_larga_nem_muda_o_que_ele_mede(
    qtbot: Any, medicoes: list[str]
) -> None:
    """O campo de lotes da aba «Correção» muda durante a medição, e o Palpite chega e entra.

    Alterar algo na tela de correção enquanto o Palpite roda não o para nem afeta o resultado. Era
    `test_mudar_a_entrada_durante_o_palpite_nao_aplica_as_sugestoes`: antes o Palpite era medido
    nos lotes do campo, e mudá-los largava as sugestões — primeiro à chegada, com «a entrada
    mudou», e depois na hora do toque. Agora ele mede na partição que o «Diagnóstico» sugere, uma
    só medição serve, e o corretor dele entra.

    A Validação e o Diagnóstico entram em dia antes do clique
    (`_leituras_em_dia`, sem sugestão de lotes), e o Palpite sai no clique. Com o Diagnóstico real,
    que sugere dois lotes, `n_lotes == 2` no fim seria verdade com o toque e sem ele.
    """
    janela, transporte = _janela(qtbot)
    _leituras_em_dia(janela)
    janela.pipeline.corretor.setCurrentText("serrf")
    bloco = _bloco(janela)
    bloco.botao.click()
    assert janela._depois_do_palpite is not None, "âncora: o clique não ficou à espera"

    janela.pipeline.n_lotes.setValue(2)
    assert janela._depois_do_palpite is not None, "mudar os lotes largou a espera do Palpite"
    _gira_ate(janela, lambda: janela._depois_do_palpite is None)
    _encerra(janela)

    assert janela.pipeline.corretor.currentText() == "none", "o corretor do Palpite não entrou"
    assert janela.pipeline.n_lotes.value() == 2, "as sugestões mexeram nos lotes escolhidos"
    assert bloco.desfecho.text().startswith("Sugestões aplicadas:"), bloco.desfecho.text()
    assert bloco.botao.isEnabled()
    assert ROTA_DA_ETAPA[1] not in transporte.rotas
    assert medicoes == [MODO], "o Palpite foi medido de novo por causa dos lotes da aba"


def test_as_sugestoes_mexem_so_no_corretor_e_nos_lotes(qtbot: Any, medicoes: list[str]) -> None:
    """O que a pessoa muda noutro campo enquanto o Palpite mede fica como ela deixou.

    A variante do RSD não entra na identidade do Palpite, então mudá-la não larga a espera — e as
    sugestões, quando entram, mexem só no corretor e nos lotes, que são os dois eixos que a máquina
    propõe. Era `test_a_retomada_rele_a_tela_como_o_clique`, sobre a Correção
    retomada, que mandava o que o primeiro clique leu; a retomada passou a ser a das sugestões.
    """
    janela, _ = _janela(qtbot)
    variante = janela.pipeline.variante_do_rsd
    # O repouso passou a «Com todos os QCs» (`plain`), e a pessoa muda para a outra variante; antes
    # era o contrário.
    assert variante.currentText() == VARIANTE_DO_RSD_EM_REPOUSO == "plain", (
        "âncora: a variante em repouso"
    )
    janela.pipeline.corretor.setCurrentText("serrf")
    bloco = _bloco(janela)
    bloco.botao.click()
    assert janela._depois_do_palpite is not None, "âncora: o clique não ficou à espera"

    variante.setCurrentText("outlier_removed")
    _gira_ate(janela, lambda: bloco.botao.isEnabled())
    _encerra(janela)

    assert janela.pipeline.corretor.currentText() == "none", "âncora: a sugestão não entrou"
    assert variante.currentText() == "outlier_removed", "as sugestões repuseram a variante"


def test_a_chamada_do_palpite_que_falha_larga_a_espera_das_sugestoes(qtbot: Any) -> None:
    """Sem desfecho de rota, a retomada não fica guardada para o próximo Palpite pedido à mão."""
    janela, _ = _janela(qtbot)
    saiu: list[str] = []
    janela._depois_do_palpite = lambda: saiu.append("clique antigo")
    assert janela._motivo_para_nao_corrigir("execucao") == PALPITE_EM_CURSO, "âncora"

    janela._falha(CHAVE_DO_PALPITE, ConnectionError("recusado"))

    assert janela._depois_do_palpite is None, "a retomada ficou guardada"
    assert "não completou" in _bloco(janela).desfecho.text()
    assert janela._motivo_para_nao_corrigir("execucao") is None, "a trava ficou presa"
    assert saiu == []


# ---------------------------------------------------------------------------
# A junção com a interface do colega: a etiqueta de geração não alcança o Palpite
# ---------------------------------------------------------------------------


def test_trocar_de_arquivo_durante_o_post_do_palpite_nao_prende_a_trava(
    qtbot: Any, medicoes: list[str]
) -> None:
    """A troca de arquivo durante o `POST` do Palpite não deixa a espera das sugestões presa.

    A janela etiqueta as leituras com a geração dos arquivos, e descarta a resposta de uma geração
    velha. Etiquetado, o `202` do Palpite que voltava depois da troca era descartado, a espera
    nunca nascia, e `_depois_do_palpite` ficava guardado: toda correção seguinte era recusada com
    «o Palpite está a ser medido», o próprio Palpite incluído. O Palpite tem identidade e
    supersessão próprias, e por isso fica fora da etiqueta. Quem fica à espera é «Seguir sugestões
    do app», e não mais o clique da Correção.

    A Validação e o Diagnóstico entram em dia antes do clique
    (`_leituras_em_dia`), e o clique pede o Palpite na hora: é o `POST` dele que a troca encontra.
    Sem isso a troca cairia durante a Validação, e a cadeia pararia antes de o Palpite sair.
    """
    janela, _ = _janela(qtbot)
    _leituras_em_dia(janela)
    _bloco(janela).botao.click()
    assert janela._depois_do_palpite is not None, "âncora: o clique não ficou à espera"

    # A troca de arquivo, pelo mesmo método que o sinal do par chama, antes da resposta chegar.
    janela._esquece_as_amostras_dos_lotes(None)
    _gira_ate(janela, lambda: janela._depois_do_palpite is None, ms=30_000)
    _encerra(janela)

    assert janela._motivo_para_nao_corrigir("execucao") is None, "a trava ficou presa"
    assert medicoes == [MODO], "âncora: o Palpite não chegou a medir"
