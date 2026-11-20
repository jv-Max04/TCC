"""O segundo modo da aba de vias — o enriquecimento pelo MetaboAnalystR.

O executor do R, o motor, a origem das bibliotecas e os diálogos são injetados; os padrões são os de
produção, e `test_os_padroes_sao_os_de_producao` o afirma (um painel cujo padrão fosse o duplo
passaria aqui e não calcularia nada na máquina do usuário). O oráculo é o widget.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from refino.desktop.janela import enriquecimento as ej
from refino.desktop.vias_r import MotorDoR
from refino.io import dados_vias
from refino.io import enriquecimento_externo as ext
from refino.io.enriquecimento_r import TEMPO_LIMITE_PADRAO_S, RespostaDoR, corre_roteiro_de_vias
from refino.io.rds_restrito import Lista, Vetor, escreve_rds, le_rds
from refino.io.validators import CANONICAL_GROUPS
from refino.pathways import enriquecimento as en

if TYPE_CHECKING:  # pragma: no cover - só para o mypy
    from pytestqt.qtbot import QtBot

CANON = ["Glycine", "L-Alanine", "Citric acid", "Pyruvic acid", "Urea"]
SHA_DE_TESTE = "ab" * 32


def _tabela(tmp_path: Path) -> Path:
    linhas = ["Compound,Description,Score"] + [f"f{i},{n},1" for i, n in enumerate(CANON)]
    linhas += ["f9,Lactose,1", "fx,Taurine,1"]
    caminho = tmp_path / "identificacao.csv"
    caminho.write_text("\n".join(linhas) + "\n", encoding="utf-8")
    return caminho


def _volcano() -> dict[str, Any]:
    features = [f"f{i}" for i in range(len(CANON))] + ["f9", "f8"]
    pvalue = [0.001, 0.002, 0.003, 0.004, 0.2, 0.5, None]
    return {
        "volcano": {
            "features": features,
            "pvalue": pvalue,
            "statistic": [0.0] * 7,
            "grupo_a": "ATLL",
            "grupo_b": "Controle",
            "n_grupo_a": 12,
            "n_grupo_b": 9,
        }
    }


def _mapeamento(nomes: list[str]) -> list[dict[str, Any]]:
    return [
        {"consulta": n, "casado": n, "hmdb": f"HMDB{i}", "kegg": None} for i, n in enumerate(nomes)
    ]


def _ok_mapear(pedido: dict[str, Any]) -> RespostaDoR:
    # O pedido sem fundo — o do modo externo — recebe `"fundo": null`, como o roteiro responde
    # (`r_service/vias_entrypoint.R`, ação `mapear`).
    fundo = pedido.get("fundo")
    return RespostaDoR(
        "ok",
        "",
        dados={
            "lista": {"mapeamento": _mapeamento(pedido["lista"])},
            "fundo": None if fundo is None else {"mapeamento": _mapeamento(fundo)},
        },
    )


def _ok_ora(pedido: dict[str, Any], *, nome: str = "Urea Cycle") -> RespostaDoR:
    # contagens que batem com o hipergeométrico: 3 acertos, via de 10, lista de 4, universo de 6
    from refino.pathways.enriquecimento import p_hipergeometrico

    universo = len(pedido.get("fundo", [])) or 1002
    na_lista = len(pedido["lista"])
    p = p_hipergeometrico(3, 5, na_lista, universo)
    via = {
        "via": nome,
        "total": 5,
        "esperado": 1.0,
        "acertos": 3,
        "p": float(f"{p:.3g}"),
        "holm": 1.0,
        "fdr": float(f"{p:.3g}"),
        "compostos": pedido["lista"][:3],
    }
    return RespostaDoR(
        "ok",
        "",
        dados={
            "biblioteca": "smpdb_pathway",
            "universo": {
                "tipo": "fundo_do_ensaio" if pedido.get("fundo") else "biblioteca",
                "tamanho": universo,
            },
            "na_lista": na_lista,
            "vias": [via],
        },
    )


class Executor:
    def __init__(self, *, ora: Any = _ok_ora, trava: threading.Event | None = None) -> None:
        self.pedidos: list[dict[str, Any]] = []
        self._ora = ora
        self._trava = trava
        self.viu_cancelamento = False
        self.tetos: list[float | None] = []
        # Consulta → canônico que o `mapear` devolve no lugar do próprio nome (modo externo, G1).
        self.renomeia: dict[str, str] = {}

    def __call__(self, pedido: dict[str, Any], **kw: Any) -> RespostaDoR:
        self.pedidos.append(pedido)
        self.tetos.append(kw["tempo_limite_s"])
        if self._trava is not None:
            # O prazo é o que faz uma regressão do cancelamento REPROVAR em vez de pendurar:
            # sem ele, um `encerra` que não cancelasse prenderia o destrutor da piscina para
            # sempre (medido).
            prazo = time.monotonic() + 15
            while not self._trava.wait(0.05):
                if kw["cancelado"]():
                    self.viu_cancelamento = True
                    return RespostaDoR("cancelado", "cancelado")
                if time.monotonic() > prazo:
                    return RespostaDoR("motor_falhou", "o duplo esperou 15 s sem ver cancelamento")
        if pedido["action"] == "mapear":
            r = _ok_mapear(pedido)
            assert r.dados is not None
            for linha in r.dados["lista"]["mapeamento"]:  # type: ignore[index]
                linha["casado"] = self.renomeia.get(linha["consulta"], linha["casado"])
            return r
        return self._ora(pedido)


def _painel(qtbot: QtBot, tmp_path: Path, executor: Any, **kw: Any) -> ej.PainelDeEnriquecimento:
    motor = MotorDoR(
        tmp_path / "R",
        tmp_path / "R" / "vias_entrypoint.R",
        tmp_path / "vias-dados",
        tmp_path / "baixadas",
    )
    origem = ej.OrigemDasBibliotecas(
        tmp_path / "vias-dados",
        "da pasta do laboratório, conferidas",
        True,
        (
            ej.ArquivoDaBiblioteca(
                "syn_nms.qs", SHA_DE_TESTE, "fixado, conferido contra o manifesto"
            ),
        ),
    )
    opcoes: dict[str, Any] = {
        "executor": executor,
        "motor": lambda: motor,
        "abre_arquivo": lambda _pai: _tabela(tmp_path),
        "salva_arquivo": lambda _pai: tmp_path / "saida.csv",
        "confirma": lambda *_a: True,
        "bibliotecas": lambda _m: origem,
    }
    opcoes.update(kw)
    painel = ej.PainelDeEnriquecimento(**opcoes)
    qtbot.addWidget(painel)
    return painel


def _prepara(painel: ej.PainelDeEnriquecimento) -> None:
    painel.recebe_volcano(_volcano())
    painel._carrega_tabela()
    painel._confirma_colunas()


def test_os_padroes_sao_os_de_producao() -> None:
    import inspect

    assinatura = inspect.signature(ej.PainelDeEnriquecimento.__init__)
    assert assinatura.parameters["executor"].default is corre_roteiro_de_vias
    assert assinatura.parameters["bibliotecas"].default is ej.origem_das_bibliotecas
    # O modo externo: o enviador padrão é o transporte real, e o diálogo padrão é o Qt real.
    assert assinatura.parameters["enviador"].default is ext.envia_ao_servidor
    assert assinatura.parameters["confirma_envio"].default is ej._confirma_envio


def test_enriquecer_fica_desligado_ate_confirmar_as_colunas(qtbot: QtBot, tmp_path: Path) -> None:
    painel = _painel(qtbot, tmp_path, Executor())
    painel.recebe_volcano(_volcano())
    painel._carrega_tabela()
    assert not painel.botao_enriquece.isEnabled(), (
        "com a tabela carregada e as colunas por confirmar"
    )
    assert painel.coluna_feature.currentText() == "Compound"
    assert painel.coluna_composto.currentText() == "Description"
    painel._confirma_colunas()
    assert painel.botao_enriquece.isEnabled()


def test_matriz_toserrf_carregada_no_lugar_da_tabela_e_recusada_na_tela(
    qtbot: QtBot, tmp_path: Path
) -> None:
    """Um caso real: a matriz corrigida enchia as caixas com «NA, batch, A, A…»."""
    matriz = tmp_path / "pos_hilic_corrigir_serrf.csv"
    matriz.write_text(
        'NA,"batch","A","A"\nNA,"sampleType","qc","sample"\nNA,"time","1","2"\n'
        '"No","label","QC1","S1"\n"1","2.25_810.7739m/z","10","20"\n',
        encoding="utf-8",
    )
    painel = _painel(qtbot, tmp_path, Executor(), abre_arquivo=lambda _pai: matriz)
    painel.recebe_volcano(_volcano())
    painel._carrega_tabela()
    assert "formato do SERRF" in painel.estado.text()
    assert painel.coluna_feature.count() == 0, "o cabeçalho da matriz foi oferecido como colunas"
    assert not painel.botao_confirma.isEnabled()
    # A âncora do par: a tabela de identificação, pelo mesmo caminho, enche as caixas.
    painel._abre_arquivo = lambda _pai: _tabela(tmp_path)
    painel._carrega_tabela()
    assert painel.coluna_feature.currentText() == "Compound"
    assert painel.botao_confirma.isEnabled()


def test_fluxo_completo_duas_chamadas_e_tabela_visivel(qtbot: QtBot, tmp_path: Path) -> None:
    ex = Executor()
    painel = _painel(qtbot, tmp_path, ex)
    _prepara(painel)
    painel.botao_enriquece.click()
    qtbot.waitUntil(lambda: painel.tabela.rowCount() == 1, timeout=5000)
    assert [p["action"] for p in ex.pedidos] == ["mapear", "ora_local"]
    assert ex.pedidos[0]["lista"] == CANON[:4], "o corte por p < 0,05 deixa os quatro primeiros"
    # A PRESENÇA do fundo do ensaio na ORA do fluxo normal: sem ela, as ausências do modo
    # biblioteca (`test_lista_igual_ao_fundo_...` e `test_download_aceito_...`) passavam também
    # com o fundo perdido em todo pedido, e o R calcularia contra a biblioteca inteira.
    assert painel._recorte is not None
    assert set(ex.pedidos[1]) == {"action", "dados_dir", "lista", "fundo"}
    assert ex.pedidos[1]["fundo"] == sorted(painel._recorte.fundo), "a ORA perdeu o fundo do ensaio"
    assert set(ex.pedidos[1]["fundo"]) > set(ex.pedidos[1]["lista"]), (
        "âncora: fundo e lista diferem"
    )
    assert painel.linhas()[0][0] == "Urea Cycle"
    assert "SMPDB" in painel.estado.text() and "conferidas" in painel.estado.text()
    assert "Glycine" in (painel.tabela.item(0, 0).toolTip() or "")
    assert painel.botao_exporta.isEnabled()
    assert painel.botao_enriquece.isEnabled(), "recorte aceito: o botão volta ligado"
    assert "Aviso:" not in painel.estado.text(), "quatro compostos: o aviso não se aplica"


def test_mudar_o_corte_refaz_as_contagens_sem_chamar_o_r(qtbot: QtBot, tmp_path: Path) -> None:
    ex = Executor()
    painel = _painel(qtbot, tmp_path, ex)
    _prepara(painel)
    assert painel.botao_enriquece.isEnabled(), "âncora: com quatro compostos o botão liga"
    antes = painel.degraus.text()
    painel.valor_do_corte.setValue(0.0025)
    assert painel.degraus.text() != antes
    assert "**2 compostos na lista**" in painel.degraus.text()
    # Antes, dois compostos eram recusa antecipada; agora calculam, com aviso no resultado.
    assert painel.botao_enriquece.isEnabled(), "dois compostos seguem para o cálculo"
    # A direção de recusa continua: um corte que não deixa composto nenhum desliga o botão.
    painel.valor_do_corte.setValue(0.0005)
    assert "**0 compostos na lista**" in painel.degraus.text()
    assert not painel.botao_enriquece.isEnabled(), "lista vazia é recusa antecipada"
    assert ex.pedidos == []


def test_execucao_nova_durante_o_calculo_nao_pinta_o_resultado_velho(
    qtbot: QtBot, tmp_path: Path
) -> None:
    trava = threading.Event()
    ex = Executor(trava=trava)
    painel = _painel(qtbot, tmp_path, ex)
    _prepara(painel)
    painel.botao_enriquece.click()
    qtbot.waitUntil(lambda: len(ex.pedidos) == 1, timeout=5000)
    painel.recebe_volcano(_volcano())  # avança o token
    trava.set()
    painel._piscina.waitForDone(5000)
    qtbot.wait(100)
    assert painel.tabela.rowCount() == 0, "o resultado de um recorte velho foi pintado"
    # par causal: sem a execução nova no meio, o mesmo resultado é pintado
    painel.botao_enriquece.click()
    qtbot.waitUntil(lambda: painel.tabela.rowCount() == 1, timeout=5000)


def test_cancelar_encerra_e_diz_que_nada_foi_calculado(qtbot: QtBot, tmp_path: Path) -> None:
    ex = Executor(trava=threading.Event())
    painel = _painel(qtbot, tmp_path, ex)
    _prepara(painel)
    painel.botao_enriquece.click()
    qtbot.waitUntil(lambda: painel.botao_cancela.isEnabled(), timeout=5000)
    painel.botao_cancela.click()
    qtbot.waitUntil(lambda: "cancelada" in painel.estado.text(), timeout=5000)
    assert ex.viu_cancelamento and painel.tabela.rowCount() == 0
    assert painel.botao_enriquece.isEnabled(), "depois de cancelar, dá para enriquecer de novo"


@pytest.mark.parametrize(
    ("resposta", "chave"),
    [
        (RespostaDoR("motor_ausente", "o R embarcado não está em X"), "motor_ausente"),
        (RespostaDoR("motor_falhou", "saiu com 3", stderr="Error in x"), "motor_falhou"),
        (RespostaDoR("tempo_esgotado", "passou de 120 s"), "tempo_esgotado"),
        (RespostaDoR("resposta_malformada", "status desconhecido"), "resposta_malformada"),
        (RespostaDoR("recusado", "mais da metade", recusa="fundo_recusado"), "fundo_recusado"),
        (RespostaDoR("recusado", "faltam", recusa="biblioteca_ausente"), "biblioteca_ausente"),
        (RespostaDoR("recusado", "curta", recusa="lista_recusada"), "lista_recusada"),
        (
            RespostaDoR("recusado", "nenhum", recusa="sem_casamento_na_biblioteca"),
            "sem_casamento_na_biblioteca",
        ),
    ],
)
def test_cada_desfecho_mostra_classe_frase_e_acao(
    qtbot: QtBot, tmp_path: Path, resposta: RespostaDoR, chave: str
) -> None:
    ex = Executor(ora=lambda _p: resposta)
    painel = _painel(qtbot, tmp_path, ex)
    _prepara(painel)
    painel.botao_enriquece.click()
    msg = ej.MENSAGENS[chave]
    qtbot.waitUntil(lambda: msg.frase in painel.estado.text(), timeout=5000)
    assert msg.acao.split(":")[0] in painel.estado.text()
    titulo = {
        "entrada": "Sobre a sua entrada",
        "ambiente": "Sobre o ambiente",
        "defeito": "Defeito a reportar",
    }
    assert titulo[msg.classe] in painel.estado.text()
    assert painel.tabela.rowCount() == 0


def test_p_que_nao_bate_com_as_contagens_e_resposta_malformada(
    qtbot: QtBot, tmp_path: Path
) -> None:
    def ora_errada(pedido: dict[str, Any]) -> RespostaDoR:
        r = _ok_ora(pedido)
        assert r.dados is not None
        r.dados["vias"][0]["p"] = 0.5  # type: ignore[index]
        return r

    painel = _painel(qtbot, tmp_path, Executor(ora=ora_errada))
    _prepara(painel)
    painel.botao_enriquece.click()
    qtbot.waitUntil(
        lambda: ej.MENSAGENS["resposta_malformada"].frase in painel.estado.text(), timeout=5000
    )


def test_lista_igual_ao_fundo_oferece_a_biblioteca_e_a_marca_aparece(
    qtbot: QtBot, tmp_path: Path
) -> None:
    ex = Executor()
    painel = _painel(qtbot, tmp_path, ex)
    painel.recebe_volcano(
        {"volcano": {"features": [f"f{i}" for i in range(5)], "pvalue": [0.001] * 5}}
    )
    painel._carrega_tabela()
    painel._confirma_colunas()
    # `isHidden` e não `isVisible`: um filho de um widget que nunca foi mostrado tem `isVisible`
    # falso sempre, e a asserção anterior — `isVisible() or not botao_enriquece.isEnabled()` — era
    # satisfeita pela própria recusa, com o botão escondido (revisão de 01/10/2026).
    assert not painel.botao_fundo_biblioteca.isHidden(), "a recusa promete o fundo e não o oferece"
    assert not painel.botao_enriquece.isEnabled()
    painel.botao_fundo_biblioteca.click()
    qtbot.waitUntil(lambda: painel.tabela.rowCount() == 1, timeout=5000)
    # O conjunto inteiro de chaves e não `"fundo" not in`: a igualdade reprova uma chave a mais
    # seja qual for o nome dela, e a presença do fundo no fluxo normal está em
    # `test_fluxo_completo_...`.
    assert set(ex.pedidos[-1]) == {"action", "dados_dir", "lista"}, (
        "com o fundo da biblioteca, a ORA vai sem fundo do ensaio"
    )
    assert "tende a falsos positivos" in painel.estado.text()
    assert "tende a falsos positivos" in (painel.tabela.item(0, 0).toolTip() or "")


def test_depois_do_fundo_da_biblioteca_o_enriquecer_continua_desligado(
    qtbot: QtBot, tmp_path: Path
) -> None:
    """O recorte recusado com o fundo do ensaio continua recusado depois do cálculo com o outro.

    O par causal é `test_fluxo_completo_...`: lá o recorte é aceito, e o botão volta ligado.
    """
    ex = Executor()
    painel = _painel(qtbot, tmp_path, ex)
    painel.recebe_volcano(
        {"volcano": {"features": [f"f{i}" for i in range(5)], "pvalue": [0.001] * 5}}
    )
    painel._carrega_tabela()
    painel._confirma_colunas()
    painel.botao_fundo_biblioteca.click()
    qtbot.waitUntil(lambda: painel.tabela.rowCount() == 1, timeout=5000)
    assert not painel.botao_enriquece.isEnabled(), "o botão religado só leva à mesma recusa"


def test_falha_que_nao_oferece_o_fundo_esconde_o_botao(qtbot: QtBot, tmp_path: Path) -> None:
    """O par causal do teste acima: o botão segue a bandeira da falha, e não fica sempre à vista."""
    ex = Executor(ora=lambda _p: RespostaDoR("recusado", "x", recusa="fundo_recusado"))
    painel = _painel(qtbot, tmp_path, ex)
    _prepara(painel)
    painel.botao_enriquece.click()
    frase = ej.MENSAGENS["fundo_recusado"].frase
    qtbot.waitUntil(lambda: frase in painel.estado.text(), timeout=5000)
    assert painel.botao_fundo_biblioteca.isHidden()


def test_o_csv_neutraliza_formula_vinda_do_servidor(qtbot: QtBot, tmp_path: Path) -> None:
    painel = _painel(qtbot, tmp_path, Executor(ora=lambda p: _ok_ora(p, nome='=HYPERLINK("x")')))
    _prepara(painel)
    painel.botao_enriquece.click()
    qtbot.waitUntil(lambda: painel.botao_exporta.isEnabled(), timeout=5000)
    painel.botao_exporta.click()
    linhas = list(csv.reader((tmp_path / "saida.csv").open(encoding="utf-8-sig")))
    vias = [ln for ln in linhas if ln and ln[0].startswith("'=")]
    assert vias, "o nome de via que começa por = saiu sem neutralizar"
    assert any(ln[:2] == ["bibliotecas validadas", "sim"] for ln in linhas)
    assert ej.neutraliza("\t=1") == "'\t=1" and ej.neutraliza("Urea") == "Urea"


def test_download_recusado_nao_chama_o_r(qtbot: QtBot, tmp_path: Path) -> None:
    ex = Executor()
    painel = _painel(qtbot, tmp_path, ex, bibliotecas=lambda _m: None, confirma=lambda *_a: False)
    _prepara(painel)
    painel.botao_enriquece.click()
    assert ej.MENSAGENS["download_recusado"].frase in painel.estado.text()
    assert ex.pedidos == []


def test_biblioteca_divergente_e_desfecho_proprio(qtbot: QtBot, tmp_path: Path) -> None:
    ex = Executor()
    painel = _painel(
        qtbot, tmp_path, ex, bibliotecas=lambda _m: ej.Falha("biblioteca_divergente", "syn_nms.qs")
    )
    _prepara(painel)
    painel.botao_enriquece.click()
    assert ej.MENSAGENS["biblioteca_divergente"].frase in painel.estado.text()
    assert ex.pedidos == []


def test_encerra_cancela_o_r_antes_de_drenar(qtbot: QtBot, tmp_path: Path) -> None:
    ex = Executor(trava=threading.Event())
    painel = _painel(qtbot, tmp_path, ex)
    _prepara(painel)
    painel.botao_enriquece.click()
    qtbot.waitUntil(lambda: len(ex.pedidos) == 1, timeout=5000)
    assert painel.encerra(teto_ms=5000), "a piscina não drenou: o R em voo não foi cancelado"
    assert ex.viu_cancelamento


def test_todas_as_chaves_de_mensagem_cobrem_os_desfechos() -> None:
    from typing import get_args

    from refino.domain.labels import ViasRecusa
    from refino.io.enriquecimento_externo import DesfechoDoEnvio
    from refino.io.enriquecimento_r import DesfechoDoLancador
    from refino.pathways.enriquecimento import MotivoDeRecusa

    # Os desfechos do envio entram PELA TRADUÇÃO e não pelo nome cru: dois deles (`sem_rede`,
    # `transporte_recusado`) repetem chaves do download, e o nome cru passaria aqui pela frase do
    # download — que, lida depois de um envio, diria o fato errado. A tradução fechada nas duas
    # direções é o que faz um desfecho novo reprovar em vez de cair no padrão de `.get`.
    envio = set(get_args(DesfechoDoEnvio)) - {"ok"}
    assert envio, "âncora: o Literal do envio tem desfechos além de ok"
    assert set(ej._DESFECHO_DO_ENVIO) == envio, sorted(envio ^ set(ej._DESFECHO_DO_ENVIO))
    precisam = (
        set(get_args(ViasRecusa))
        | set(get_args(DesfechoDoLancador))
        | set(get_args(MotivoDeRecusa))
        | {ej._DESFECHO_DO_ENVIO[d] for d in envio}
        | {"envio_recusado", "carga_recusada", "envio_nao_confirmado"}
    ) - {"ok", "recusado"}
    assert precisam <= set(ej.MENSAGENS), sorted(precisam - set(ej.MENSAGENS))


def test_a_primeira_chamada_do_processo_vai_sem_teto_e_as_seguintes_com_ele(
    qtbot: QtBot, tmp_path: Path
) -> None:
    """Resolução 12: a primeira chamada ao R paga o cache frio e só para pelo cancelamento.

    A mesma análise corrida duas vezes: na primeira, o `mapear` vai sem teto e a ORA com ele; na
    segunda, as duas com ele. Sem a marca de aquecido, a segunda repetiria o `None` (par causal).
    """
    from refino.io.enriquecimento_r import TEMPO_LIMITE_PADRAO_S

    ex = Executor()
    painel = _painel(qtbot, tmp_path, ex)
    _prepara(painel)
    painel.botao_enriquece.click()
    qtbot.waitUntil(lambda: painel.tabela.rowCount() == 1, timeout=5000)
    assert ex.tetos == [None, TEMPO_LIMITE_PADRAO_S]
    painel.botao_enriquece.click()
    qtbot.waitUntil(lambda: len(ex.tetos) == 4, timeout=5000)
    assert ex.tetos[2:] == [TEMPO_LIMITE_PADRAO_S, TEMPO_LIMITE_PADRAO_S]


def test_trocar_o_tipo_de_corte_recorta_uma_vez_com_o_valor_final(
    qtbot: QtBot, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Sem o bloqueio a troca recortava 3 a 4 vezes, a 1.ª com `n_melhores 0.0`, que levanta."""
    from refino.pathways import enriquecimento as en

    painel = _painel(qtbot, tmp_path, Executor())
    _prepara(painel)
    cortes: list[tuple[str, float]] = []
    original = en.recorta

    def espia(linhas: Any, nomes: Any, pvalores: Any, corte: Any) -> Any:
        cortes.append((corte.tipo, corte.valor))
        return original(linhas, nomes, pvalores, corte)

    monkeypatch.setattr(en, "recorta", espia)
    painel.tipo_do_corte.setCurrentIndex(painel.tipo_do_corte.findData("n_melhores"))
    assert cortes == [("n_melhores", 25.0)]
    painel.tipo_do_corte.setCurrentIndex(painel.tipo_do_corte.findData("q"))
    assert cortes[1:] == [("q", 0.05)]


def test_despacha_trabalho_emite_o_resultado_ou_a_excecao_com_o_token(qtbot: QtBot) -> None:
    """Na thread principal, onde o `coverage` vê: os dois desfechos levam o token do trabalho."""
    sinais = ej._Sinais()
    recebidos: list[tuple[object, int]] = []
    fases: list[tuple[str, int]] = []
    sinais.concluido.connect(lambda r, t: recebidos.append((r, t)))
    sinais.fase.connect(lambda f, t: fases.append((f, t)))

    def ok(fase: Any) -> object:
        fase("mapeando")
        return "resultado"

    def levanta(_fase: Any) -> object:
        raise RuntimeError("o R caiu")

    ej.despacha_trabalho(ok, sinais, 7)
    ej.despacha_trabalho(levanta, sinais, 8)
    assert fases == [("mapeando", 7)]
    assert recebidos[0] == ("resultado", 7)
    assert isinstance(recebidos[1][0], RuntimeError) and recebidos[1][1] == 8


# --------------------------------------------------------------------------- revisão de 01/10/2026


@pytest.mark.parametrize(
    ("caixa", "outra"),
    [("coluna_feature", "Description"), ("coluna_composto", "Score"), ("coluna_aceito", "Score")],
)
def test_trocar_uma_coluna_depois_de_confirmar_desfaz_a_confirmacao(
    qtbot: QtBot, tmp_path: Path, caixa: str, outra: str
) -> None:
    """A tabela lida é a das colunas de antes: com a caixa mostrando outra, nada calcula com ela."""
    painel = _painel(qtbot, tmp_path, Executor())
    _prepara(painel)
    assert painel.botao_enriquece.isEnabled(), "âncora: a confirmação ligou o botão"
    getattr(painel, caixa).setCurrentText(outra)
    assert getattr(painel, caixa).currentText() == outra, "âncora: a troca não aterrou"
    assert not painel.botao_enriquece.isEnabled(), "o Enriquecer seguiu com as colunas de antes"
    assert painel._recorte is None
    assert "Confirme as colunas da tabela." in painel.degraus.text()


def test_a_lista_reconfirmada_vem_da_coluna_nova(qtbot: QtBot, tmp_path: Path) -> None:
    painel = _painel(qtbot, tmp_path, Executor())
    _prepara(painel)
    assert painel._recorte is not None and "Glycine" in painel._recorte.lista
    painel.coluna_composto.setCurrentText("Score")
    painel._confirma_colunas()
    assert painel._recorte is not None and set(painel._recorte.lista) == {"1"}
    assert painel._colunas == "feature: Compound; composto: Score; aceito: (nenhuma)"


def test_carregar_a_tabela_refaz_o_recorte_uma_vez(qtbot: QtBot, tmp_path: Path) -> None:
    """Repreencher as caixas não dispara a invalidação de cada uma: o token avança uma vez só."""
    painel = _painel(qtbot, tmp_path, Executor())
    painel.recebe_volcano(_volcano())
    antes = painel._token
    painel._carrega_tabela()
    assert painel._token == antes + 1


@pytest.mark.parametrize(
    ("primeira", "teto_da_segunda"),
    [
        # o lançador nem conseguiu arrancar o R: o antivírus a barrar o `Rscript.exe`
        (RespostaDoR("motor_falhou", "o R não pôde ser lançado: [WinError 5] Acesso negado"), None),
        # o R arrancou e recusou: pagou o cache frio, e a próxima já leva o teto
        (RespostaDoR("recusado", "mais da metade", recusa="fundo_recusado"), TEMPO_LIMITE_PADRAO_S),
    ],
)
def test_so_o_r_que_respondeu_conta_como_aquecido(
    qtbot: QtBot, tmp_path: Path, primeira: RespostaDoR, teto_da_segunda: float | None
) -> None:
    base = Executor()
    pendentes = [primeira]

    def executor(pedido: dict[str, Any], **kw: Any) -> RespostaDoR:
        if pendentes:
            base.tetos.append(kw["tempo_limite_s"])
            return pendentes.pop()
        return base(pedido, **kw)

    painel = _painel(qtbot, tmp_path, executor)
    _prepara(painel)
    painel.botao_enriquece.click()
    chave = primeira.recusa or primeira.desfecho
    frase = ej.MENSAGENS[chave].frase
    qtbot.waitUntil(lambda: frase in painel.estado.text(), timeout=5000)
    painel.botao_enriquece.click()
    qtbot.waitUntil(lambda: painel.tabela.rowCount() == 1, timeout=5000)
    assert base.tetos == [None, teto_da_segunda, TEMPO_LIMITE_PADRAO_S]


def test_o_csv_traz_o_contraste_as_colunas_e_o_hash_de_cada_arquivo(
    qtbot: QtBot, tmp_path: Path
) -> None:
    """Sem estas linhas o CSV não dizia sobre qual par nem sobre que bibliotecas foi feito."""
    painel = _painel(qtbot, tmp_path, Executor())
    _prepara(painel)
    painel.botao_enriquece.click()
    qtbot.waitUntil(lambda: painel.botao_exporta.isEnabled(), timeout=5000)
    painel.botao_exporta.click()
    linhas = list(csv.reader((tmp_path / "saida.csv").open(encoding="utf-8-sig")))
    assert ["contraste", "ATLL (n = 12) contra Controle (n = 9)"] in linhas
    assert ["aviso", "nenhum"] in linhas, "quatro compostos: o aviso não se aplica"
    assert [
        "colunas da tabela",
        "feature: Compound; composto: Description; aceito: (nenhuma)",
    ] in linhas
    assert [
        "arquivo de dados syn_nms.qs",
        f"sha256 {SHA_DE_TESTE}; fixado, conferido contra o manifesto",
    ] in linhas


def _ora_curta(pedido: dict[str, Any]) -> RespostaDoR:
    """Uma ORA coerente para lista curta: 1 acerto numa via de 5, contra o fundo do pedido."""
    from refino.pathways.enriquecimento import p_hipergeometrico

    universo = len(pedido["fundo"])
    na_lista = len(pedido["lista"])
    p = float(f"{p_hipergeometrico(1, 5, na_lista, universo):.3g}")
    via = {
        "via": "Urea Cycle",
        "total": 5,
        "esperado": 1.0,
        "acertos": 1,
        "p": p,
        "holm": 1.0,
        "fdr": p,
        "compostos": pedido["lista"][:1],
    }
    return RespostaDoR(
        "ok",
        "",
        dados={
            "biblioteca": "smpdb_pathway",
            "universo": {"tipo": "fundo_do_ensaio", "tamanho": universo},
            "na_lista": na_lista,
            "vias": [via],
        },
    )


def test_lista_de_dois_compostos_calcula_e_avisa_na_tela_e_no_csv(
    qtbot: QtBot, tmp_path: Path
) -> None:
    """O mínimo de 3 foi retirado, com aviso na tela.

    O par é `test_fluxo_completo_...` e `test_o_csv_traz_o_contraste_...`: com quatro compostos
    não há aviso na tela, e o CSV diz `aviso,nenhum`.
    """
    ex = Executor(ora=_ora_curta)
    painel = _painel(qtbot, tmp_path, ex)
    volcano = _volcano()
    volcano["volcano"]["pvalue"] = [0.001, 0.002, 0.3, 0.4, 0.2, 0.5, None]
    painel.recebe_volcano(volcano)
    painel._carrega_tabela()
    painel._confirma_colunas()
    assert painel.botao_enriquece.isEnabled(), "a lista de dois compostos foi recusada antes do R"
    painel.botao_enriquece.click()
    qtbot.waitUntil(lambda: painel.tabela.rowCount() == 1, timeout=5000)
    assert ex.pedidos[-1]["lista"] == ["Glycine", "L-Alanine"], "âncora: a lista é de dois"
    aviso = en.aviso_de_poucos(2)
    assert aviso is not None
    assert f"**Aviso:** {aviso}" in painel.estado.text()
    painel.botao_exporta.click()
    linhas = list(csv.reader((tmp_path / "saida.csv").open(encoding="utf-8-sig")))
    assert ["aviso", aviso] in linhas


def test_o_contraste_sem_par_declarado_diz_isso() -> None:
    assert ej._contraste(None) == "não declarado na execução"
    assert ej._contraste({"features": []}) == "não declarado na execução"
    assert ej._contraste({"grupo_a": "A", "grupo_b": None}) == "não declarado na execução"
    assert ej._contraste({"grupo_a": "A", "grupo_b": "B", "n_grupo_a": True}) == "A contra B"


@pytest.mark.parametrize("alvo", ["pasta_inexistente", "diretorio"])
def test_exportar_para_destino_que_nao_grava_diz_no_painel(
    qtbot: QtBot, tmp_path: Path, alvo: str
) -> None:
    """O CSV aberto no Excel e a pasta sem permissão são `OSError`; escapar do slot é silêncio."""
    destino = tmp_path / "nao-existe" / "x.csv" if alvo == "pasta_inexistente" else tmp_path
    destinos = [destino]
    painel = _painel(qtbot, tmp_path, Executor(), salva_arquivo=lambda _pai: destinos[-1])
    _prepara(painel)
    painel.botao_enriquece.click()
    qtbot.waitUntil(lambda: painel.botao_exporta.isEnabled(), timeout=5000)
    painel.botao_exporta.click()
    assert f"Não foi possível gravar `{destino.name}`" in painel.estado.text()
    assert "Exportado para" not in painel.estado.text()
    assert painel.tabela.rowCount() == 1, "o resultado continua na tela"
    # Par causal na mesma função: o literal da ausência acima aparece PRESENTE quando a exportação
    # grava. Sem isto, reescrever a frase de sucesso deixava a ausência vácua e nada reprovava.
    destinos.append(tmp_path / "gravavel.csv")
    painel.botao_exporta.click()
    assert "Exportado para `gravavel.csv`" in painel.estado.text()
    assert (tmp_path / "gravavel.csv").is_file()


# --- a origem das bibliotecas, sobre pastas reais e um manifesto pequeno


def _manifesto(pasta: Path) -> tuple[dados_vias.Biblioteca, ...]:
    pasta.mkdir(parents=True, exist_ok=True)
    saida = []
    for nome, corpo in (("a.qs", b"alfa"), ("b.qs", b"beta")):
        (pasta / nome).write_bytes(corpo)
        saida.append(
            dados_vias.Biblioteca(
                nome,
                nome,
                len(corpo),
                hashlib.sha256(corpo).hexdigest(),
                f"https://x.invalid/{nome}",
            )
        )
    return tuple(saida)


def _motor(tmp_path: Path) -> MotorDoR:
    return MotorDoR(tmp_path / "R", tmp_path / "r.R", tmp_path / "fixadas", tmp_path / "baixadas")


def _registro(pasta: Path, *, validada: bool) -> None:
    registro = [
        {
            "arquivo": f.name,
            "sha256": hashlib.sha256(f.read_bytes()).hexdigest(),
            "tamanho": f.stat().st_size,
            "baixada_em": "2026-10-01T12:00:00+00:00",
            "validada": validada,
        }
        for f in sorted(pasta.glob("*.qs"))
    ]
    (pasta / dados_vias.REGISTRO_DE_DOWNLOAD).write_text(json.dumps(registro), encoding="utf-8")


def test_origem_pasta_fixada_conferida(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifesto = _manifesto(tmp_path / "fixadas")
    monkeypatch.setattr(dados_vias, "le_manifesto", lambda *_a: manifesto)
    # Uma data que não é a do manifesto embalado: a asserção abaixo só passa se ela viajar.
    monkeypatch.setattr(dados_vias, "le_data_do_manifesto", lambda *_a: "2031-02-03")
    origem = ej.origem_das_bibliotecas(_motor(tmp_path))
    assert isinstance(origem, ej.OrigemDasBibliotecas)
    assert origem.validada and origem.pasta == tmp_path / "fixadas"
    assert [(a.arquivo, a.sha256) for a in origem.arquivos] == [
        (b.arquivo, b.sha256) for b in manifesto
    ]
    assert origem.arquivos and all("2031-02-03" in a.procedencia for a in origem.arquivos), (
        "R13 pede a data da versão de cada biblioteca, e a fixada saiu sem ela"
    )


def test_origem_pasta_fixada_com_um_byte_trocado_e_divergente(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifesto = _manifesto(tmp_path / "fixadas")
    (tmp_path / "fixadas" / "a.qs").write_bytes(b"alfb")  # mesmo tamanho, outro hash
    (tmp_path / "fixadas" / "b.qs").unlink()  # e um ausente: divergente vence
    monkeypatch.setattr(dados_vias, "le_manifesto", lambda *_a: manifesto)
    assert ej.origem_das_bibliotecas(_motor(tmp_path)) == ej.Falha(
        "biblioteca_divergente", "a.qs, b.qs"
    )


def test_origem_pasta_fixada_com_arquivo_faltando_e_ausente(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifesto = _manifesto(tmp_path / "fixadas")
    (tmp_path / "fixadas" / "b.qs").unlink()
    monkeypatch.setattr(dados_vias, "le_manifesto", lambda *_a: manifesto)
    assert ej.origem_das_bibliotecas(_motor(tmp_path)) == ej.Falha("biblioteca_ausente", "b.qs")


def test_origem_sem_pasta_fixada_e_sem_registro_pede_o_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifesto = _manifesto(tmp_path / "baixadas")  # os arquivos estão lá, o registro não
    monkeypatch.setattr(dados_vias, "le_manifesto", lambda *_a: manifesto)
    assert ej.origem_das_bibliotecas(_motor(tmp_path)) is None


@pytest.mark.parametrize("validada", [True, False])
def test_origem_baixada_diz_a_data_e_se_foi_validada(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, validada: bool
) -> None:
    pasta = tmp_path / "baixadas"
    manifesto = _manifesto(pasta)
    if not validada:
        # o servidor trocou um arquivo: o que veio não tem o hash do manifesto
        (pasta / "a.qs").write_bytes(b"alfb")
    _registro(pasta, validada=validada)
    monkeypatch.setattr(dados_vias, "le_manifesto", lambda *_a: manifesto)
    origem = ej.origem_das_bibliotecas(_motor(tmp_path))
    assert isinstance(origem, ej.OrigemDasBibliotecas)
    assert origem.pasta == pasta and origem.validada is validada
    assert "baixadas em 2026-10-01T12:00:00+00:00" in origem.descricao
    assert ("não validadas" in origem.descricao) is not validada
    # o hash gravado é o do arquivo que o R leu, e não o do manifesto
    em_disco = [hashlib.sha256((pasta / b.arquivo).read_bytes()).hexdigest() for b in manifesto]
    assert [a.sha256 for a in origem.arquivos] == em_disco
    procedencia_de_a = origem.arquivos[0].procedencia
    assert procedencia_de_a.startswith("baixado em 2026-10-01T12:00:00+00:00")
    assert ("não validado" in procedencia_de_a) is not validada


# --- o download da versão pública, pelo painel


def test_download_aceito_baixa_e_segue_com_o_fundo_pedido(
    qtbot: QtBot, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """O clique no fundo da biblioteca pede o download; depois dele, a ORA vai SEM fundo."""
    baixou: list[tuple[Path, bool]] = []

    def baixa(pasta: Path, _bibliotecas: Any, *, confirmado: bool, **_kw: Any) -> Any:
        baixou.append((pasta, confirmado))
        return dados_vias.ResultadoDoDownload("baixado", "", ())

    monkeypatch.setattr(dados_vias, "baixa_bibliotecas", baixa)
    depois = ej.OrigemDasBibliotecas(tmp_path / "baixadas", "baixadas em 2026-10-01", True)
    ex = Executor()
    painel = _painel(qtbot, tmp_path, ex, bibliotecas=lambda _m: depois if baixou else None)
    painel.recebe_volcano(
        {"volcano": {"features": [f"f{i}" for i in range(5)], "pvalue": [0.001] * 5}}
    )
    painel._carrega_tabela()
    painel._confirma_colunas()
    painel.botao_fundo_biblioteca.click()
    qtbot.waitUntil(lambda: painel.tabela.rowCount() == 1, timeout=5000)
    assert baixou == [(tmp_path / "baixadas", True)]
    assert [p["action"] for p in ex.pedidos] == ["mapear", "ora_local"]
    # Igualdade do conjunto de chaves, pelo mesmo motivo do teste do fundo da biblioteca acima.
    assert set(ex.pedidos[-1]) == {"action", "dados_dir", "lista"}, (
        "o fundo da biblioteca se perdeu no download"
    )
    assert "tende a falsos positivos" in painel.estado.text()
    assert "baixadas em 2026-10-01" in painel.estado.text()


@pytest.mark.parametrize(
    ("desfecho", "chave"),
    [
        ("sem_rede", "sem_rede"),
        ("transporte_recusado", "transporte_recusado"),
        ("cancelado", "cancelado"),
        ("recusado", "download_recusado"),
    ],
)
def test_download_que_nao_baixou_mostra_o_desfecho_proprio(
    qtbot: QtBot, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, desfecho: str, chave: str
) -> None:
    monkeypatch.setattr(
        dados_vias,
        "baixa_bibliotecas",
        lambda *_a, **_kw: dados_vias.ResultadoDoDownload(desfecho, "motivo do teste", ()),  # type: ignore[arg-type]
    )
    ex = Executor()
    painel = _painel(qtbot, tmp_path, ex, bibliotecas=lambda _m: None)
    _prepara(painel)
    painel.botao_enriquece.click()
    frase = ej.MENSAGENS[chave].frase
    qtbot.waitUntil(lambda: frase in painel.estado.text(), timeout=5000)
    assert "motivo do teste" in painel.estado.text()
    assert ex.pedidos == [], "o R não corre sem as bibliotecas"


# =========================================================================== o modo externo
#
# **Nenhum teste aqui fala com a rede.** O `enviador` é um duplo, e a resposta que ele devolve é
# SINTÉTICA: montada com `escreve_rds` e lida pelo MESMO `ext.le_resposta` da produção, com um nome
# de via INVENTADO. A resposta real traz dados derivados do KEGG, cuja licença não permite
# redistribuir, e nenhuma é versionada.

# Não é o tamanho da biblioteca KEGG: é um universo sintético, escolhido só para o p ser finito.
UNIVERSO_SINTETICO = 1_000
VIA_INVENTADA = "Via inventada do teste"
INSTANTE = datetime(2026, 10, 2, 12, 0, 0, tzinfo=UTC)
# O recorte padrão de `_volcano` (p < 0,05) deixa os quatro primeiros; o mapeamento falso devolve
# cada nome como o próprio canônico, então os nomes que saem são estes, ordenados.
NOMES_DO_ENVIO = sorted(CANON[:4])


def _texto(*valores: str) -> Vetor:
    return Vetor("texto", tuple(valores))


def _rds_do_servidor(nomes: list[str], *, acertos: int = 2, total: int = 10) -> bytes:
    """A resposta na forma que `le_resposta` lê: `analSet$ora.mat`, `ora.hits`, `uniq.count`.

    O p é o hipergeométrico das contagens a três algarismos — a relação medida numa resposta real
    em 01/10/2026 (40 de 40 vias). A lista que o servidor usa é a dos nomes que estão em alguma
    via, e não a dos enviados (medido em 02/10/2026): aqui, os `acertos` nomes da via única.
    """
    na_lista = len(set(nomes[:acertos]))
    p = float(f"{en.p_hipergeometrico(acertos, total, na_lista, UNIVERSO_SINTETICO):.3g}")
    esperado = na_lista * total / UNIVERSO_SINTETICO
    matriz = Vetor(
        "real",
        (float(total), esperado, float(acertos), p, p, p),
        {
            "dim": Vetor("inteiro", (1, 6)),
            "dimnames": Lista(
                (
                    _texto(VIA_INVENTADA),
                    _texto("total", "expected", "hits", "Raw p", "Holm p", "FDR"),
                )
            ),
        },
    )
    hits = Lista((_texto(*nomes[:acertos]),), {"names": _texto(VIA_INVENTADA)})
    anal = Lista((matriz, hits), {"names": _texto("ora.mat", "ora.hits")})
    conjunto = Lista((Vetor("inteiro", (UNIVERSO_SINTETICO,)),), {"names": _texto("uniq.count")})
    return escreve_rds(Lista((anal, conjunto), {"names": _texto("analSet", "dataSet")}))


def _ok_do_servidor(nomes: list[str]) -> ext.RespostaDoServidor:
    """Todos os nomes enviados estão na via: a lista da ORA é a dos enviados."""
    return ext.RespostaDoServidor(
        "ok",
        dados=ext.le_resposta(_rds_do_servidor(nomes, acertos=len(nomes)), nomes),
        segundos=0.1,
    )


class Enviador:
    """O duplo do transporte: guarda o que sairia e devolve uma resposta sintética."""

    def __init__(self, resposta: Any = _ok_do_servidor) -> None:
        self.chamadas: list[tuple[bytes, list[str]]] = []
        self.argumentos: list[set[str]] = []
        self._resposta = resposta

    def __call__(self, carga: bytes, nomes: list[str], **kw: Any) -> ext.RespostaDoServidor:
        self.chamadas.append((carga, list(nomes)))
        self.argumentos.append(set(kw))
        resposta: ext.RespostaDoServidor = self._resposta(list(nomes))
        return resposta


class Dialogo:
    """O duplo do diálogo de envio: guarda os nomes que mostraria e devolve a escolha."""

    def __init__(self, aceita: bool, *, durante: Any = None) -> None:
        self.aceita = aceita
        self.vistos: list[list[str]] = []
        self._durante = durante

    def __call__(self, _pai: Any, nomes: list[str]) -> bool:
        self.vistos.append(list(nomes))
        if self._durante is not None:
            self._durante()
        return self.aceita


def _painel_externo(
    qtbot: QtBot, tmp_path: Path, executor: Any, *, enviador: Any, dialogo: Any, **kw: Any
) -> ej.PainelDeEnriquecimento:
    kw.setdefault("instante", lambda: INSTANTE)
    painel = _painel(qtbot, tmp_path, executor, enviador=enviador, confirma_envio=dialogo, **kw)
    painel.biblioteca_de_vias.setCurrentIndex(painel.biblioteca_de_vias.findData("externo"))
    assert painel._modo_externo(), "âncora: o seletor não foi para o modo externo"
    return painel


def test_o_seletor_nasce_no_modo_local_com_os_rotulos_do_modo_externo(
    qtbot: QtBot, tmp_path: Path
) -> None:
    """Escolhido a cada execução, e o padrão é o local.

    As frases exactas dos dois itens e do seletor são as que
    `docs/EXPLICAR-cada-tela-e-cada-campo.html` também traz.
    """
    from PySide6.QtWidgets import QLabel

    painel = _painel(qtbot, tmp_path, Executor())
    combo = painel.biblioteca_de_vias
    assert [(combo.itemText(i), combo.itemData(i)) for i in range(combo.count())] == [
        ("SMPDB — no computador (modo local)", "local"),
        ("KEGG — pelo servidor do MetaboAnalyst (modo externo)", "externo"),
    ]
    assert combo.currentData() == "local" and not painel._modo_externo()
    assert "Biblioteca de vias" in [r.text() for r in painel.findChildren(QLabel)]
    nota = (
        "O servidor do MetaboAnalyst não aplica o fundo do ensaio na ORA KEGG: o universo é a "
        "biblioteca KEGG inteira, e a análise tende a falsos positivos."
    )
    assert painel.biblioteca.text() == ej.BIBLIOTECA_LOCAL
    assert nota not in painel.biblioteca.text()
    combo.setCurrentIndex(combo.findData("externo"))
    assert nota in painel.biblioteca.text(), "a nota do modo externo não apareceu ao escolher"
    combo.setCurrentIndex(combo.findData("local"))
    assert painel.biblioteca.text() == ej.BIBLIOTECA_LOCAL


def test_externo_com_envio_confirmado_manda_a_carga_conferida_e_mostra_a_ora(
    qtbot: QtBot, tmp_path: Path
) -> None:
    """O caminho inteiro: mapear no R local, diálogo, envio, conferência, tela e CSV.

    O par causal do envio é `test_externo_com_dialogo_recusado_...`: o mesmo fluxo com «Cancelar»
    não chama o enviador.
    """
    ex, envia, dialogo = Executor(), Enviador(), Dialogo(True)
    instantes = iter([INSTANTE, INSTANTE.replace(hour=13)])
    painel = _painel_externo(
        qtbot, tmp_path, ex, enviador=envia, dialogo=dialogo, instante=lambda: next(instantes)
    )
    _prepara(painel)
    assert "contra a biblioteca KEGG inteira" in painel.degraus.text()
    assert "contra um fundo de" not in painel.degraus.text()
    painel.botao_enriquece.click()
    qtbot.waitUntil(lambda: painel.tabela.rowCount() == 1, timeout=5000)

    # O R local só mapeia, e sem fundo: o servidor não o aplica (medido em 01/10/2026).
    assert [p["action"] for p in ex.pedidos] == ["mapear"]
    assert set(ex.pedidos[0]) == {"action", "dados_dir", "lista"}, "o modo externo mapeou o fundo"
    assert ex.tetos == [None], "a primeira chamada fria vai sem teto, como no modo local"
    # O que o diálogo mostrou é o que saiu, e a carga passa na G2 contra esses nomes.
    assert dialogo.vistos == [NOMES_DO_ENVIO]
    assert len(envia.chamadas) == 1
    carga, nomes = envia.chamadas[0]
    assert nomes == NOMES_DO_ENVIO
    ext.confere_carga(carga, NOMES_DO_ENVIO)
    objeto = le_rds(carga)
    assert isinstance(objeto, Lista)
    ora_vec = objeto.campo("oraVec")
    assert isinstance(ora_vec, Vetor) and list(ora_vec.valores) == NOMES_DO_ENVIO
    assert envia.argumentos == [{"cancelado", "relogio"}]

    estado = painel.estado.text()
    assert painel.linhas()[0][0] == VIA_INVENTADA
    assert "ORA do MSEA pelo servidor do MetaboAnalyst sobre a KEGG" in estado
    assert "Universo: a biblioteca KEGG inteira" in estado
    assert ej.NOTA_DO_MODO_EXTERNO in estado
    assert "4 nomes enviados" in estado
    assert "Aviso:" not in estado, "quatro nomes: o aviso não se aplica"
    assert "tende a falsos positivos" in (painel.tabela.item(0, 0).toolTip() or "")
    assert painel.botao_fundo_biblioteca.isHidden()

    painel.botao_exporta.click()
    linhas = list(csv.reader((tmp_path / "saida.csv").open(encoding="utf-8-sig")))
    assert ["motor", "servidor do MetaboAnalyst (www.xialab.ca), ORA do MSEA"] in linhas
    assert ["modo", "externo"] in linhas
    assert ["biblioteca de vias", "kegg_pathway"] in linhas
    # O instante da CONSULTA (o primeiro), e não o da exportação (o segundo).
    assert [
        "biblioteca do servidor",
        "versão não informada pelo servidor; consultada em 2026-10-02T12:00:00+00:00",
    ] in linhas
    assert ["exportado em", "2026-10-02T13:00:00+00:00"] in linhas
    assert ["nomes enviados", "4"] in linhas
    assert ["nomes enviados em alguma via KEGG", "4"] in linhas
    assert ["universo", f"biblioteca ({UNIVERSO_SINTETICO})"] in linhas
    assert ["compostos no fundo", "não se aplica: o modo externo não envia fundo"] in linhas
    chaves = {ln[0] for ln in linhas if ln}
    # Presença antes da ausência: as linhas das bibliotecas locais estão, com o nome delas.
    assert "bibliotecas de dados do mapeamento local" in chaves
    assert "bibliotecas de dados" not in chaves and ["modo", "local"] not in linhas


def test_externo_com_dialogo_recusado_nada_sai_e_a_tela_diz(qtbot: QtBot, tmp_path: Path) -> None:
    """Par causal na mesma função: recusado não envia; o mesmo painel, aceito, envia."""
    ex, envia, dialogo = Executor(), Enviador(), Dialogo(False)
    painel = _painel_externo(qtbot, tmp_path, ex, enviador=envia, dialogo=dialogo)
    _prepara(painel)
    painel.botao_enriquece.click()
    qtbot.waitUntil(lambda: "Nada foi enviado" in painel.estado.text(), timeout=5000)
    assert dialogo.vistos == [NOMES_DO_ENVIO], "âncora: o diálogo abriu com os nomes"
    assert envia.chamadas == [], "o diálogo foi recusado e o enviador foi chamado"
    assert ej.MENSAGENS["envio_nao_confirmado"].frase in painel.estado.text()
    assert painel.tabela.rowCount() == 0 and not painel.botao_exporta.isEnabled()
    assert painel.botao_enriquece.isEnabled(), "depois de recusar, dá para tentar de novo"
    dialogo.aceita = True
    painel.botao_enriquece.click()
    qtbot.waitUntil(lambda: painel.tabela.rowCount() == 1, timeout=5000)
    assert len(envia.chamadas) == 1
    # O `mapear` da primeira tentativa respondeu, ainda que nada tenha saído: a segunda leva o teto.
    assert ex.tetos == [None, TEMPO_LIMITE_PADRAO_S]


def _um_acerto(nomes: list[str]) -> ext.RespostaDoServidor:
    return ext.RespostaDoServidor(
        "ok", dados=ext.le_resposta(_rds_do_servidor(nomes, acertos=1), nomes)
    )


def test_externo_lista_curta_avisa_na_tela_e_no_csv(qtbot: QtBot, tmp_path: Path) -> None:
    """O aviso de lista curta vale no modo externo, sobre a lista que o SERVIDOR usou: os nomes
    enviados que estão em alguma via KEGG. Aqui dois saem e um está numa via."""
    envia = Enviador(_um_acerto)
    painel = _painel_externo(qtbot, tmp_path, Executor(), enviador=envia, dialogo=Dialogo(True))
    volcano = _volcano()
    volcano["volcano"]["pvalue"] = [0.001, 0.002, 0.3, 0.4, 0.2, 0.5, None]
    painel.recebe_volcano(volcano)
    painel._carrega_tabela()
    painel._confirma_colunas()
    painel.botao_enriquece.click()
    qtbot.waitUntil(lambda: painel.tabela.rowCount() == 1, timeout=5000)
    assert envia.chamadas[0][1] == ["Glycine", "L-Alanine"], "âncora: a lista é de dois"
    aviso = en.aviso_de_poucos(1)
    assert aviso is not None
    assert f"**Aviso:** {aviso}" in painel.estado.text()
    assert "2 nomes enviados, 1 em alguma via KEGG" in painel.estado.text()
    painel.botao_exporta.click()
    linhas = list(csv.reader((tmp_path / "saida.csv").open(encoding="utf-8-sig")))
    assert ["aviso", aviso] in linhas
    assert ["nomes enviados em alguma via KEGG", "1"] in linhas


def _corpo_com(fonte: str, nome: str) -> tuple[dict[str, Any], dict[str, str]]:
    """O corpo da execução e o renomeio do mapeamento que põem `nome` no vocabulário proibido."""
    corpo = _volcano()
    renomeia: dict[str, str] = {}
    if fonte == "grupo_do_volcano":
        corpo["volcano"]["grupo_a"] = nome
    elif fonte == "grupo_da_anova":
        corpo["anova"] = {"grupos": ["ATLL", nome]}
    elif fonte == "amostra_do_kmeans":
        corpo["kmeans"] = {"amostras": ["S01", nome], "grupos": ["ATLL", "ATLL"]}
    elif fonte == "grupo_do_som":
        corpo["som"] = {"amostras": ["S01"], "grupos": [nome]}
    else:  # coluna da tabela: o canônico devolvido pelo mapeamento é um nome de coluna
        renomeia = {"Glycine": nome}
    return corpo, renomeia


@pytest.mark.parametrize(
    ("fonte", "nome"),
    [
        ("grupo_do_volcano", "Glycine"),
        ("grupo_da_anova", "Pyruvic acid"),
        ("amostra_do_kmeans", "Citric acid"),
        ("grupo_do_som", "L-Alanine"),
        ("coluna_da_tabela", "Score"),
    ],
)
def test_g1_recusa_nome_que_repete_rotulo_da_execucao_ou_da_tabela(
    qtbot: QtBot, tmp_path: Path, fonte: str, nome: str
) -> None:
    """A G1 recusa antes do diálogo: nada é mostrado, nada sai. Par causal na mesma função."""
    ex, envia, dialogo = Executor(), Enviador(), Dialogo(False)
    painel = _painel_externo(qtbot, tmp_path, ex, enviador=envia, dialogo=dialogo)
    corpo, ex.renomeia = _corpo_com(fonte, nome)
    painel.recebe_volcano(corpo)
    painel._carrega_tabela()
    painel._confirma_colunas()
    painel.botao_enriquece.click()
    frase = ej.MENSAGENS["envio_recusado"].frase
    qtbot.waitUntil(lambda: frase in painel.estado.text(), timeout=5000)
    assert f"«{nome}»" in painel.estado.text(), "a recusa não diz qual nome"
    assert "Sobre a sua entrada" in painel.estado.text()
    assert dialogo.vistos == [], "a G1 recusou e o diálogo abriu mesmo assim"
    assert envia.chamadas == []
    # Par causal: o mesmo painel, sem a coincidência, chega ao diálogo.
    ex.renomeia = {}
    painel.recebe_volcano(_volcano())
    painel.botao_enriquece.click()
    qtbot.waitUntil(lambda: bool(dialogo.vistos), timeout=5000)
    assert dialogo.vistos == [NOMES_DO_ENVIO]


def _vocabulario_campo_a_campo(corpo: Any, cabecalho: Any = ()) -> frozenset[str]:
    """O colhedor ANTERIOR, campo a campo, copiado aqui só como o lado de controle do par
    causal abaixo: lia `volcano.grupo_a`/`grupo_b`, `anova.grupos` e `amostras`/`grupos` do K-means
    e do SOM, mais os canônicos e o cabeçalho."""

    def textos(valores: Any) -> set[str]:
        valores = [valores] if isinstance(valores, str) else valores
        if not isinstance(valores, list):
            return set()
        return {v for v in valores if isinstance(v, str) and v.strip()}

    corpo = corpo if isinstance(corpo, dict) else {}
    vetados = set(CANONICAL_GROUPS) | textos(list(cabecalho))
    for raiz, chaves in (
        ("volcano", ("grupo_a", "grupo_b")),
        ("anova", ("grupos",)),
        ("kmeans", ("amostras", "grupos")),
        ("som", ("amostras", "grupos")),
    ):
        no = corpo.get(raiz)
        if isinstance(no, dict):
            for chave in chaves:
                vetados |= textos(no.get(chave))
    return frozenset(vetados)


def test_g1_recusa_nome_de_amostra_que_so_esta_nas_caixas_por_amostra(
    qtbot: QtBot, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Um canônico igual a um nome de amostra que SÓ aparece em
    `distribuicoes.caixas_por_amostra_antes` é recusado antes do diálogo, e nada sai.

    O par causal é o MESMO corpo, no mesmo painel, com o colhedor campo a campo de antes: ali o
    nome de amostra passa pela G1, o diálogo o mostra e o enviador o recebe.
    """
    corpo = _volcano()
    corpo["distribuicoes"] = {
        "caixas_por_amostra_antes": [
            {"amostra": "S01", "mediana": 1.0, "n_discrepantes": 0},
            {"amostra": "Glycine", "mediana": 2.0, "n_discrepantes": None},
        ],
        "caixas_por_amostra_depois": [],
    }
    cabecalho = ("Compound", "Description", "Score")
    # Âncora: o nome está SÓ nas caixas por amostra, e é um dos que o mapeamento confirma.
    assert "Glycine" in NOMES_DO_ENVIO
    assert "Glycine" not in _vocabulario_campo_a_campo(corpo, cabecalho)
    assert "Glycine" in ej.vocabulario_proibido(corpo, cabecalho)

    envia, dialogo = Enviador(), Dialogo(True)
    painel = _painel_externo(qtbot, tmp_path, Executor(), enviador=envia, dialogo=dialogo)
    painel.recebe_volcano(corpo)
    painel._carrega_tabela()
    painel._confirma_colunas()
    assert tuple(painel._tabela.cabecalho) == cabecalho, "âncora: o cabeçalho é o da tabela"
    painel.botao_enriquece.click()
    frase = ej.MENSAGENS["envio_recusado"].frase
    qtbot.waitUntil(lambda: frase in painel.estado.text(), timeout=5000)
    assert "«Glycine»" in painel.estado.text()
    assert dialogo.vistos == [] and envia.chamadas == [], "o nome de amostra chegou ao envio"

    # Par causal: o colhedor de antes não lê as caixas, e o nome de amostra SAI.
    monkeypatch.setattr(ej, "vocabulario_proibido", _vocabulario_campo_a_campo)
    painel.recebe_volcano(corpo)
    painel.botao_enriquece.click()
    qtbot.waitUntil(lambda: painel.tabela.rowCount() == 1, timeout=5000)
    assert dialogo.vistos == [NOMES_DO_ENVIO]
    assert len(envia.chamadas) == 1 and "Glycine" in envia.chamadas[0][1]


@pytest.mark.parametrize("planta", ["nome_que_nao_foi_confirmado", "esquema_diferente"])
def test_g2_recusa_a_carga_plantada_e_a_classe_e_defeito(
    qtbot: QtBot, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, planta: str
) -> None:
    """A G2 confere os bytes que vão sair. A planta é em `monta_carga`, que o painel chama."""
    original = ext.monta_carga

    def plantada(nomes: Any) -> bytes:
        if planta == "nome_que_nao_foi_confirmado":
            return original([*nomes, "AMOSTRA_07"])
        return escreve_rds(Lista((_texto(*nomes),), {"names": _texto("oraVec")}))

    monkeypatch.setattr(ext, "monta_carga", plantada)
    envia, dialogo = Enviador(), Dialogo(False)
    painel = _painel_externo(qtbot, tmp_path, Executor(), enviador=envia, dialogo=dialogo)
    _prepara(painel)
    painel.botao_enriquece.click()
    frase = ej.MENSAGENS["carga_recusada"].frase
    qtbot.waitUntil(lambda: frase in painel.estado.text(), timeout=5000)
    assert "Defeito a reportar" in painel.estado.text()
    assert dialogo.vistos == [] and envia.chamadas == []
    # Par causal: sem a planta, o mesmo painel chega ao diálogo.
    monkeypatch.setattr(ext, "monta_carga", original)
    painel.botao_enriquece.click()
    qtbot.waitUntil(lambda: bool(dialogo.vistos), timeout=5000)
    assert dialogo.vistos == [NOMES_DO_ENVIO]


@pytest.mark.parametrize(
    ("desfecho", "chave"),
    [
        ("sem_rede", "envio_sem_rede"),
        ("transporte_recusado", "envio_transporte_recusado"),
        ("servidor_recusou", "servidor_recusou"),
        ("resposta_malformada", "servidor_resposta_malformada"),
        ("cancelado", "envio_cancelado"),
    ],
)
def test_cada_desfecho_do_envio_mostra_a_mensagem_dele(
    qtbot: QtBot, tmp_path: Path, desfecho: str, chave: str
) -> None:
    def responde(_nomes: list[str]) -> ext.RespostaDoServidor:
        return ext.RespostaDoServidor(desfecho, "motivo do teste")  # type: ignore[arg-type]

    envia = Enviador(responde)
    painel = _painel_externo(qtbot, tmp_path, Executor(), enviador=envia, dialogo=Dialogo(True))
    _prepara(painel)
    painel.botao_enriquece.click()
    msg = ej.MENSAGENS[chave]
    qtbot.waitUntil(lambda: msg.frase in painel.estado.text(), timeout=5000)
    assert "motivo do teste" in painel.estado.text()
    titulo = {
        "entrada": "Sobre a sua entrada",
        "ambiente": "Sobre o ambiente",
        "defeito": "Defeito a reportar",
    }
    assert titulo[msg.classe] in painel.estado.text()
    assert len(envia.chamadas) == 1, "âncora: o envio aconteceu"
    assert painel.tabela.rowCount() == 0 and not painel.botao_exporta.isEnabled()
    # As frases do envio não falam do download, que é o motivo de as chaves serem próprias.
    assert "download" not in painel.estado.text()


def test_p_do_servidor_que_nao_bate_com_as_contagens_e_resposta_malformada(
    qtbot: QtBot, tmp_path: Path
) -> None:
    """A MESMA conferência do modo local. O par causal é o fluxo confirmado acima."""

    def p_trocado(nomes: list[str]) -> ext.RespostaDoServidor:
        r = _ok_do_servidor(nomes)
        assert r.dados is not None
        r.dados["vias"][0]["p"] = 0.5
        return r

    painel = _painel_externo(
        qtbot, tmp_path, Executor(), enviador=Enviador(p_trocado), dialogo=Dialogo(True)
    )
    _prepara(painel)
    painel.botao_enriquece.click()
    frase = ej.MENSAGENS["servidor_resposta_malformada"].frase
    qtbot.waitUntil(lambda: frase in painel.estado.text(), timeout=5000)
    assert "não é o hipergeométrico" in painel.estado.text()


@pytest.mark.parametrize("aceita", [True, False])
def test_execucao_nova_com_o_dialogo_aberto_nao_envia_nem_pinta(
    qtbot: QtBot, tmp_path: Path, aceita: bool
) -> None:
    """A guarda de obsolescência atravessa o diálogo modal: o «Enviar» de um recorte que já não é
    o da tela não sai.

    E o «Cancelar» dele não pinta «Nada foi enviado» sobre o recorte novo, que nada pediu. O par
    causal é `test_externo_com_envio_confirmado_...` e `test_externo_com_dialogo_recusado_...`:
    sem a execução nova no meio, um envia e o outro diz que nada saiu.
    """
    envia = Enviador()
    painel_ref: list[ej.PainelDeEnriquecimento] = []
    dialogo = Dialogo(aceita, durante=lambda: painel_ref[0].recebe_volcano(_volcano()))
    painel = _painel_externo(qtbot, tmp_path, Executor(), enviador=envia, dialogo=dialogo)
    painel_ref.append(painel)
    _prepara(painel)
    painel.botao_enriquece.click()
    qtbot.waitUntil(lambda: bool(dialogo.vistos), timeout=5000)
    painel._piscina.waitForDone(5000)
    qtbot.wait(100)
    assert envia.chamadas == [], "o envio saiu sobre um recorte que já não era o da tela"
    assert painel.tabela.rowCount() == 0
    assert "Nada foi enviado" not in painel.estado.text(), "a recusa velha pintou o recorte novo"
    assert painel.botao_enriquece.isEnabled(), "âncora: o recorte novo está pronto para enriquecer"


def test_voltar_ao_modo_local_restaura_o_fundo_e_a_recusa_da_lista_igual_ao_fundo(
    qtbot: QtBot, tmp_path: Path
) -> None:
    """No externo não há fundo: a lista igual ao fundo não é recusa, e o botão do fundo some."""
    painel = _painel(qtbot, tmp_path, Executor())
    painel.recebe_volcano(
        {"volcano": {"features": [f"f{i}" for i in range(5)], "pvalue": [0.001] * 5}}
    )
    painel._carrega_tabela()
    painel._confirma_colunas()
    igual = ej.MENSAGENS["igual_ao_fundo"].frase

    def como_no_local() -> None:
        assert not painel.botao_fundo_biblioteca.isHidden()
        assert not painel.botao_enriquece.isEnabled()
        assert igual in painel.estado.text()

    como_no_local()
    combo = painel.biblioteca_de_vias
    combo.setCurrentIndex(combo.findData("externo"))
    assert painel.botao_fundo_biblioteca.isHidden(), "o fundo da biblioteca no modo externo"
    assert painel.botao_enriquece.isEnabled(), "a lista igual ao fundo foi recusada no externo"
    assert igual not in painel.estado.text()
    # O que sobra de recusa no externo é a lista vazia.
    painel.valor_do_corte.setValue(0.0001)
    assert "**0 compostos na lista**" in painel.degraus.text()
    assert not painel.botao_enriquece.isEnabled(), "lista vazia é recusa nos dois modos"
    assert ej.MENSAGENS["lista_vazia"].frase in painel.estado.text()
    painel.valor_do_corte.setValue(0.05)
    combo.setCurrentIndex(combo.findData("local"))
    como_no_local()


def test_no_modo_externo_nenhuma_falha_mostra_o_botao_do_fundo(
    qtbot: QtBot, tmp_path: Path
) -> None:
    painel = _painel(qtbot, tmp_path, Executor())
    oferece = ej.Falha("igual_ao_fundo", oferece_fundo_da_biblioteca=True)
    painel._mostra_falha(oferece)
    assert not painel.botao_fundo_biblioteca.isHidden(), "âncora: no local a falha o oferece"
    painel.biblioteca_de_vias.setCurrentIndex(painel.biblioteca_de_vias.findData("externo"))
    painel._mostra_falha(oferece)
    assert painel.botao_fundo_biblioteca.isHidden()


def test_vocabulario_proibido_tira_cada_fonte_do_corpo_e_nao_quebra_com_corpo_torto() -> None:
    """Cada chave que nomeia, na profundidade em que o corpo de `/execucoes` a traz.

    A igualdade exacta é a âncora das ausências: os chamarizes — texto sob chave que não nomeia —
    ficam de fora porque o conjunto inteiro é afirmado, e não por uma asserção de ausência solta.
    """
    corpo = {
        "volcano": {"grupo_a": "ATLL", "grupo_b": "Controle", "features": ["F1"]},
        "anova": {"grupos": ["Ga", "Gb"], "metodo": "welch"},
        "kmeans": {"amostras": ["S01", "S02"], "grupos": ["Gk"], "rotulos": [0, 1]},
        "som": {"amostras": ["S03"], "grupos": ["Gs"], "amostragem": "todas as 61 amostras"},
        "plsda": {"grupos_por_amostra": ["Gp1"], "grupo_positivo": "Gp+", "grupo_negativo": "Gp-"},
        "busca_por_padrao": {"ordem_dos_grupos": ["Gb1"], "referencia": "F2"},
        "random_forest": {"grupos": ["Grf"], "nome": "Random Forest"},
        "svm": {"grupos": ["Gsvm"]},
        "distribuicoes": {
            "caixas_por_amostra_antes": [{"amostra": "S04", "mediana": 1.0}],
            "caixas_por_amostra_depois": [{"amostra": "S05", "mediana": None}],
            "matriz_antes": "ingerida",
        },
        "composicao_por_classe": {"classes": [{"grupo": "Gc", "n_injecoes": 3}]},
        "pcas_da_correcao": {"tipos_de_amostra_antes": ["qc", "sample"]},
        "qualidade": {
            "plsda_groups": ("Gq1", "Gq2"),
            "plsda_multiclass_groups": ["Gq3"],
            "computed": ["qc_rsd_median"],
        },
    }
    vocabulario = ej.vocabulario_proibido(corpo, ("Compound", "Description"))
    fontes = {"ATLL", "Controle", "Ga", "Gb", "S01", "S02", "Gk", "S03", "Gs", "Gp1", "Gp+"}
    fontes |= {"Gp-", "Gb1", "Grf", "Gsvm", "S04", "S05", "Gc", "Gq1", "Gq2", "Gq3"}
    assert fontes | {"Compound", "Description"} | CANONICAL_GROUPS == vocabulario
    # Só o volcano: os dois grupos e os canônicos.
    assert ej.vocabulario_proibido({"volcano": {"grupo_a": "A", "grupo_b": "B"}}) == (
        CANONICAL_GROUPS | {"A", "B"}
    )
    # «Sob» é literal: o texto num objeto ou numa lista aninhada abaixo de uma chave que nomeia
    # conta, e a chave de um objeto não conta.
    aninhado = {"x": {"grupos": [{"nome": "Gn"}, [["Gl"]]], "S06": 1}}
    assert ej.vocabulario_proibido(aninhado) == CANONICAL_GROUPS | {"Gn", "Gl"}
    assert CANONICAL_GROUPS, "âncora: os grupos canônicos não são vazios"
    tortos: list[Any] = [
        None,
        [],
        "corpo",
        42,
        ["ATLL-solto", {"features": ["F1"]}],
        {"volcano": None, "anova": "x", "kmeans": {"amostras": None}},
        {"volcano": {"grupo_a": 5, "grupo_b": ""}, "som": {"grupos": [None, 3, "  "]}},
        {"distribuicoes": {"caixas_por_amostra_antes": [None, 7, {"amostra": 3.5}]}},
    ]
    for torto in tortos:
        assert ej.vocabulario_proibido(torto) == CANONICAL_GROUPS, torto
    # Um texto solto onde viria a lista conta como um nome: para uma guarda, vetar a mais é a
    # direção barata, e descartá-lo deixaria sair o nome que o campo mal formado carregava.
    assert "S09" in ej.vocabulario_proibido({"kmeans": {"amostras": "S09"}})
    # E um corpo que não é dicionário contribui com o que tiver sob uma chave que nomeia.
    assert "S10" in ej.vocabulario_proibido([{"amostra": "S10"}])


# A regra das candidatas do teste abaixo: o NOME da propriedade fala de amostra, grupo, classe ou
# rótulo. É o oráculo dele, e a docstring de `vocabulario_proibido` o declara.
_NOMEIA_AMOSTRA_OU_GRUPO = re.compile(r"amostra|grupo|classe|rotulo|sample|group|class")


def _traz_texto(esquema: Any) -> bool:
    """A propriedade do OpenAPI carrega texto: `string`, lista dela, ou união que a inclua."""
    if not isinstance(esquema, dict):
        return False
    if esquema.get("type") == "string":
        return True
    if esquema.get("type") == "array" and _traz_texto(esquema.get("items")):
        return True
    return any(
        _traz_texto(parte)
        for chave in ("anyOf", "oneOf", "allOf", "prefixItems")
        for parte in esquema.get(chave, ())
    )


def test_as_chaves_que_nomeiam_sao_as_do_corpo_de_execucoes_nas_duas_direcoes() -> None:
    """A lista de chaves da G1 contra o OpenAPI e o `QualityReport`, nas duas direções.

    **Completude:** toda propriedade de texto com nome de amostra ou de grupo, em todo esquema
    alcançável de `ExecucaoResposta`, e todo campo de texto assim do `QualityReport` (que viaja no
    `dict` livre `qualidade`, opaco no OpenAPI), está em `_CHAVES_QUE_NOMEIAM` ou, com motivo, em
    `_CHAVES_FORA_DA_G1`. **Grafia:** toda chave das duas existe nalgum desses esquemas.

    Medido em 02/10/2026: 21 propriedades candidatas em 19 esquemas alcançáveis, mais 2 campos do
    `QualityReport`. O oráculo é o NOME da propriedade, e escapa-lhe o nome de amostra guardado sob
    outro nome — declarado em `vocabulario_proibido`.
    """
    import dataclasses
    import typing

    from refino.api.app import build_app
    from refino.domain.spec import QualityReport

    esquemas = build_app().openapi()["components"]["schemas"]
    alcancaveis: set[str] = set()
    pendentes = ["ExecucaoResposta"]
    while pendentes:
        nome = pendentes.pop()
        if nome in alcancaveis:
            continue
        alcancaveis.add(nome)
        pendentes += re.findall(r"#/components/schemas/(\w+)", json.dumps(esquemas[nome]))
    propriedades = {
        (esquema, chave): tipo
        for esquema in alcancaveis
        for chave, tipo in esquemas[esquema].get("properties", {}).items()
    }
    tipos = typing.get_type_hints(QualityReport)
    campos = {
        ("QualityReport", c.name): "str" in str(tipos[c.name])
        for c in dataclasses.fields(QualityReport)
    }
    candidatas = {
        (esquema, chave)
        for (esquema, chave), tipo in propriedades.items()
        if _NOMEIA_AMOSTRA_OU_GRUPO.search(chave) and _traz_texto(tipo)
    } | {par for par, texto in campos.items() if texto and _NOMEIA_AMOSTRA_OU_GRUPO.search(par[1])}
    # Âncoras: o percurso chegou às caixas por amostra e ao `qualidade`, e as listas não são vazias.
    assert ("CaixaPorAmostra", "amostra") in candidatas
    assert ("QualityReport", "plsda_groups") in candidatas
    assert ej._CHAVES_QUE_NOMEIAM and ej._CHAVES_FORA_DA_G1
    assert all(motivo.strip() for motivo in ej._CHAVES_FORA_DA_G1.values())

    declaradas = ej._CHAVES_QUE_NOMEIAM | set(ej._CHAVES_FORA_DA_G1)
    assert not ej._CHAVES_QUE_NOMEIAM & set(ej._CHAVES_FORA_DA_G1), "chave nas duas listas"
    sem_destino = sorted(par for par in candidatas if par[1] not in declaradas)
    assert not sem_destino, f"propriedades que nomeiam amostra ou grupo sem destino: {sem_destino}"
    existentes = {chave for _, chave in propriedades} | {chave for _, chave in campos}
    inexistentes = sorted(declaradas - existentes)
    assert not inexistentes, f"chaves que nenhum esquema alcançável declara: {inexistentes}"


def _mapeado_ok(pedido: dict[str, Any], **_kw: Any) -> RespostaDoR:
    return _ok_mapear(pedido)


def _recorte_pequeno() -> en.RecorteDaIdentificacao:
    linhas = [(f"f{i}", n, None) for i, n in enumerate(CANON[:3])]
    return en.recorta(linhas, ["f0", "f1", "f2"], [0.001, 0.002, 0.003], en.Corte("p", 0.05))


def _motor_falso(tmp_path: Path) -> MotorDoR:
    return MotorDoR(tmp_path, tmp_path / "r.R", tmp_path, tmp_path / "b")


def test_prepara_envio_sem_vocabulario_proibido_e_defeito_e_nada_mapeado_e_recusa(
    tmp_path: Path,
) -> None:
    """Sem Qt: as duas recusas da etapa 1 que o painel não alcança por construção."""
    comum: dict[str, Any] = {"cancelado": lambda: False, "fase": lambda _f: None}
    recorte, motor = _recorte_pequeno(), _motor_falso(tmp_path)
    pronto = ej.prepara_envio_externo(
        recorte, motor, tmp_path, proibidos=CANONICAL_GROUPS, executor=_mapeado_ok, **comum
    )
    assert isinstance(pronto, ej.EnvioPreparado), "âncora: com vocabulário, a etapa 1 passa"
    assert pronto.nomes == tuple(sorted(CANON[:3]))
    sem_vocabulario = ej.prepara_envio_externo(
        recorte, motor, tmp_path, proibidos=(), executor=_mapeado_ok, **comum
    )
    assert isinstance(sem_vocabulario, ej.Falha) and sem_vocabulario.chave == "carga_recusada"

    def nada(pedido: dict[str, Any], **_kw: Any) -> RespostaDoR:
        linhas = [{"consulta": n, "casado": None, "hmdb": None} for n in pedido["lista"]]
        return RespostaDoR("ok", "", dados={"lista": {"mapeamento": linhas}, "fundo": None})

    nao_mapeou = ej.prepara_envio_externo(
        recorte, motor, tmp_path, proibidos=CANONICAL_GROUPS, executor=nada, **comum
    )
    assert isinstance(nao_mapeou, ej.Falha)
    assert (nao_mapeou.chave, nao_mapeou.o_r_respondeu) == ("nada_mapeou", True)


def test_prepara_envio_traduz_cada_recusa_da_etapa_1_e_diz_se_o_r_respondeu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Sem Qt, onde o `coverage` vê: no painel estas linhas correm no `QThreadPool`.

    `o_r_respondeu` é o fato que decide o teto da próxima chamada: o R que nem arrancou não
    aquece, e todo desfecho depois de um `mapear` com `ok` aquece.
    """
    comum: dict[str, Any] = {"cancelado": lambda: False, "fase": lambda _f: None}
    recorte, motor = _recorte_pequeno(), _motor_falso(tmp_path)

    def com(executor: Any, proibidos: Any = CANONICAL_GROUPS) -> Any:
        return ej.prepara_envio_externo(
            recorte, motor, tmp_path, proibidos=proibidos, executor=executor, **comum
        )

    nao_arrancou = com(lambda _p, **_k: RespostaDoR("motor_falhou", "[WinError 5]"))
    assert isinstance(nao_arrancou, ej.Falha)
    assert (nao_arrancou.chave, nao_arrancou.o_r_respondeu) == ("motor_falhou", False)
    torta = com(lambda _p, **_k: RespostaDoR("ok", "", dados={"lista": {}}))
    assert isinstance(torta, ej.Falha)
    assert (torta.chave, torta.o_r_respondeu) == ("resposta_malformada", True)
    g1 = com(_mapeado_ok, proibidos={"glycine"})  # a G1 compara sem caixa
    assert isinstance(g1, ej.Falha) and (g1.chave, g1.o_r_respondeu) == ("envio_recusado", True)
    assert "«Glycine»" in g1.detalhe

    def recusa(_carga: bytes, _permitidos: Any) -> None:
        raise ext.CargaRecusadaError("planta do teste")

    monkeypatch.setattr(ext, "confere_carga", recusa)
    g2 = com(_mapeado_ok)
    assert isinstance(g2, ej.Falha) and (g2.chave, g2.detalhe) == (
        "carga_recusada",
        "planta do teste",
    )


def test_conclui_envio_recusa_universo_que_nao_e_a_biblioteca_e_ok_sem_dados() -> None:
    """A tela afirma «a biblioteca KEGG inteira»; uma resposta que declara outra coisa não pinta."""
    nomes = NOMES_DO_ENVIO
    envio = ej.EnvioPreparado(tuple(nomes), ext.monta_carga(nomes))

    def conclui(resposta: ext.RespostaDoServidor) -> object:
        return ej.conclui_envio_externo(
            envio,
            enviador=lambda *_a, **_k: resposta,
            cancelado=lambda: False,
            relogio=lambda: 0.0,
            fase=lambda _f: None,
        )

    certo = _ok_do_servidor(nomes)
    assert isinstance(conclui(certo), en.ResultadoDaOra), "âncora: a resposta certa passa"
    assert certo.dados is not None
    outro = dict(certo.dados, universo={"tipo": "fundo_do_ensaio", "tamanho": 6}, vias=[])
    falha = conclui(ext.RespostaDoServidor("ok", dados=outro))
    assert isinstance(falha, ej.Falha) and falha.chave == "servidor_resposta_malformada"
    assert "fundo_do_ensaio" in falha.detalhe
    vazia = conclui(ext.RespostaDoServidor("ok"))
    assert isinstance(vazia, ej.Falha) and vazia.chave == "servidor_resposta_malformada"
    # Sem Qt, onde o `coverage` vê: a tradução, o desfecho fora do Literal e o p trocado.
    recusou = conclui(ext.RespostaDoServidor("servidor_recusou", "503 Service Unavailable"))
    assert isinstance(recusou, ej.Falha)
    assert (recusou.chave, recusou.detalhe) == ("servidor_recusou", "503 Service Unavailable")
    estranho = conclui(ext.RespostaDoServidor("inventado", "x"))  # type: ignore[arg-type]
    assert isinstance(estranho, ej.Falha) and estranho.chave == "servidor_resposta_malformada"
    trocado = dict(certo.dados, vias=[dict(certo.dados["vias"][0], p=0.5)])
    p_errado = conclui(ext.RespostaDoServidor("ok", dados=trocado))
    assert isinstance(p_errado, ej.Falha) and p_errado.chave == "servidor_resposta_malformada"
    assert "hipergeométrico" in p_errado.detalhe


def test_o_dialogo_de_envio_mostra_a_contagem_o_destino_e_o_padrao_e_cancelar(
    qtbot: QtBot,
) -> None:
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QLabel, QPlainTextEdit, QPushButton

    nomes = ["Citric acid", "Glycine", "Urea"]
    dialogo = ej.dialogo_de_envio(None, nomes)
    qtbot.addWidget(dialogo)
    assert dialogo.windowTitle() == "Enviar ao servidor do MetaboAnalyst?"
    texto = dialogo.findChild(QLabel)
    assert texto is not None
    assert texto.text().startswith("3 nomes de compostos serão enviados"), "a contagem no topo"
    assert "www.xialab.ca" in texto.text()
    assert "nem a tabela, nem amostras, nem o fundo" in texto.text()
    assert ej.texto_do_envio(1).startswith("1 nome de composto será enviado")
    area = dialogo.findChild(QPlainTextEdit, "nomes_do_envio")
    assert area is not None and area.isReadOnly()
    assert area.toPlainText().splitlines() == nomes
    enviar = dialogo.findChild(QPushButton, "enviar")
    cancelar = dialogo.findChild(QPushButton, "cancelar")
    assert enviar is not None and cancelar is not None
    assert (enviar.text(), cancelar.text()) == ("Enviar", "Cancelar")
    assert cancelar.isDefault() and not enviar.isDefault() and not enviar.autoDefault()
    assert dialogo.focusWidget() is cancelar, "o foco padrão não está em «Cancelar»"
    # Enter sem tocar em nada recusa — e o par: «Enviar» clicado aceita.
    dialogo.show()
    qtbot.waitExposed(dialogo)
    with qtbot.waitSignal(dialogo.rejected, timeout=2000):
        qtbot.keyClick(dialogo, Qt.Key.Key_Return)
    outro = ej.dialogo_de_envio(None, nomes)
    qtbot.addWidget(outro)
    botao = outro.findChild(QPushButton, "enviar")
    assert botao is not None
    with qtbot.waitSignal(outro.accepted, timeout=2000):
        botao.click()


@pytest.mark.parametrize(("botao", "esperado"), [("enviar", True), ("cancelar", False)])
def test_confirma_envio_padrao_devolve_o_botao_clicado(
    qtbot: QtBot, botao: str, esperado: bool
) -> None:
    """O padrão de produção, com o `exec` real: o diálogo modal é achado e clicado por um timer."""
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication, QDialog, QPushButton, QWidget

    pai = QWidget()
    qtbot.addWidget(pai)
    tentativas: list[int] = []

    def clica() -> None:
        abertos = [
            w
            for w in QApplication.topLevelWidgets()
            if isinstance(w, QDialog) and w.isVisible() and w.windowTitle() == ej.TITULO_DO_ENVIO
        ]
        tentativas.append(len(abertos))
        if abertos:
            alvo = abertos[0].findChild(QPushButton, botao)
            if alvo is not None:
                alvo.click()
                return
        if len(tentativas) < 100:
            QTimer.singleShot(20, clica)
            return
        # Nunca pendurar: sem o diálogo achado, recusa o que houver aberto.
        for w in QApplication.topLevelWidgets():
            if isinstance(w, QDialog) and w.isVisible():
                w.reject()

    QTimer.singleShot(0, clica)
    assert ej._confirma_envio(pai, ["Glycine"]) is esperado
    assert tentativas and tentativas[-1] == 1, "o diálogo não foi achado aberto"


# ---------------------------------------------------------------------------
# O «Cancelar» do modo externo, e o que o modo local preserva (revisão de 02/10/2026)
# ---------------------------------------------------------------------------


class EnviadorQueEspera:
    """O duplo do transporte que espera até ver `cancelado()` e devolve o que o transporte real
    devolve nesse ponto. Tem prazo, como o `Executor` com trava: reprova em vez de pendurar."""

    def __init__(self) -> None:
        self.viu_cancelamento = False
        self.chamadas = 0

    def __call__(self, carga: bytes, nomes: list[str], **kw: Any) -> ext.RespostaDoServidor:
        self.chamadas += 1
        prazo = time.monotonic() + 15
        while time.monotonic() < prazo:
            if kw["cancelado"]():
                self.viu_cancelamento = True
                return ext.RespostaDoServidor(
                    "cancelado", "cancelado durante o envio: o pedido pode ter chegado ao servidor"
                )
            time.sleep(0.02)
        return ext.RespostaDoServidor("sem_rede", "o duplo esperou 15 s sem ver cancelamento")


def test_cancelar_durante_o_envio_diz_que_os_nomes_podem_ter_saido(
    qtbot: QtBot, tmp_path: Path
) -> None:
    """Até a revisão de 02/10/2026 a tela dizia «A análise foi cancelada. Nada foi calculado.»
    depois de o pedido sair: a frase do cancelamento local, que calava o envio."""
    envia = EnviadorQueEspera()
    painel = _painel_externo(qtbot, tmp_path, Executor(), enviador=envia, dialogo=Dialogo(True))
    _prepara(painel)
    painel.botao_enriquece.click()
    qtbot.waitUntil(lambda: envia.chamadas == 1, timeout=5000)
    assert "Enviando ao servidor do MetaboAnalyst" in painel.estado.text()
    assert "antivírus" not in painel.estado.text(), "o aviso do antivírus é sobre o R local"
    painel.botao_cancela.click()
    frase = ej.MENSAGENS["envio_cancelado"].frase
    qtbot.waitUntil(lambda: frase in painel.estado.text(), timeout=5000)
    assert envia.viu_cancelamento, "o cancelamento chegou ao transporte"
    assert "os nomes da lista chegaram ao servidor" in painel.estado.text()
    assert "pode ter chegado ao servidor" in painel.estado.text(), "o detalhe do transporte"
    assert ej.MENSAGENS["cancelado"].acao not in painel.estado.text()


def test_cancelar_durante_o_mapeamento_do_externo_nao_abre_o_dialogo(
    qtbot: QtBot, tmp_path: Path
) -> None:
    """Na etapa 1 nada saiu: a frase é a do cancelamento local, e o diálogo nunca abre."""
    ex = Executor(trava=threading.Event())
    dialogo, envia = Dialogo(True), Enviador()
    painel = _painel_externo(qtbot, tmp_path, ex, enviador=envia, dialogo=dialogo)
    _prepara(painel)
    painel.botao_enriquece.click()
    qtbot.waitUntil(lambda: painel.botao_cancela.isEnabled(), timeout=5000)
    painel.botao_cancela.click()
    frase = ej.MENSAGENS["cancelado"].frase
    qtbot.waitUntil(lambda: frase in painel.estado.text(), timeout=5000)
    assert ex.viu_cancelamento and dialogo.vistos == [] and envia.chamadas == []


def test_cancelado_com_o_mapeamento_ja_concluido_nao_abre_o_dialogo(
    qtbot: QtBot, tmp_path: Path
) -> None:
    """O cancelamento que chega entre o fim do mapeamento e a tela: o resultado da etapa 1 já está
    a caminho, e mesmo assim o diálogo não abre."""
    painel_ref: list[ej.PainelDeEnriquecimento] = []

    class ExecutorQueCancelaDepoisDeMapear(Executor):
        def __call__(self, pedido: dict[str, Any], **kw: Any) -> RespostaDoR:
            r = super().__call__(pedido, **kw)
            painel_ref[0]._cancelar.set()  # o `Event` é seguro entre threads; o botão não é
            return r

    dialogo = Dialogo(True)
    painel = _painel_externo(
        qtbot, tmp_path, ExecutorQueCancelaDepoisDeMapear(), enviador=Enviador(), dialogo=dialogo
    )
    painel_ref.append(painel)
    _prepara(painel)
    painel.botao_enriquece.click()
    frase = ej.MENSAGENS["cancelado"].frase
    qtbot.waitUntil(lambda: frase in painel.estado.text(), timeout=5000)
    assert dialogo.vistos == []


def _csv_do_local(
    qtbot: QtBot, tmp_path: Path, *, fundo_da_biblioteca: bool = False
) -> tuple[ej.PainelDeEnriquecimento, list[list[str]]]:
    painel = _painel(qtbot, tmp_path, Executor())
    _prepara(painel)
    if fundo_da_biblioteca:
        # Todas as features abaixo do corte: a lista é igual ao fundo, e a recusa oferece o botão.
        painel.recebe_volcano(
            {"volcano": {"features": [f"f{i}" for i in range(5)], "pvalue": [0.001] * 5}}
        )
        painel._carrega_tabela()
        painel._confirma_colunas()
        assert painel.botao_fundo_biblioteca.isVisibleTo(painel), "âncora: a recusa o ofereceu"
        painel.botao_fundo_biblioteca.click()
    else:
        painel.botao_enriquece.click()
    qtbot.waitUntil(lambda: painel.tabela.rowCount() == 1, timeout=5000)
    painel.botao_exporta.click()
    with (tmp_path / "saida.csv").open(encoding="utf-8-sig", newline="") as f:
        linhas = list(csv.reader(f))
    return painel, linhas


def test_o_modo_local_preserva_o_cabecalho_e_os_textos(qtbot: QtBot, tmp_path: Path) -> None:
    """O diff do modo externo reestruturou o cabeçalho do CSV e os textos do modo local: estas
    linhas são as que ele tinha antes, e nenhum teste as afirmava (revisão de 02/10/2026)."""
    painel, linhas = _csv_do_local(qtbot, tmp_path)
    assert ["modo", "local"] in linhas
    assert ["motor", "MetaboAnalystR, ORA do MSEA"] in linhas
    assert ["fundo não específico do ensaio", "não"] in linhas
    assert any(ln[:1] == ["bibliotecas de dados"] and ln[1] for ln in linhas)
    assert painel._recorte is not None
    n = len(painel._recorte.fundo)
    assert n > 0, "âncora: o recorte tem fundo"
    assert ["compostos no fundo", str(n)] in linhas
    assert f"contra um fundo de {n}" in painel.degraus.text()
    assert "Universo: o fundo do ensaio" in painel.estado.text()
    assert "SMPDB" in painel.estado.text()


def test_o_modo_local_com_o_fundo_da_biblioteca_diz_no_csv(qtbot: QtBot, tmp_path: Path) -> None:
    painel, linhas = _csv_do_local(qtbot, tmp_path, fundo_da_biblioteca=True)
    assert ["fundo não específico do ensaio", "sim, tende a falsos positivos"] in linhas
    assert "Universo: a biblioteca inteira" in painel.estado.text()


def test_externo_com_nome_fora_de_toda_via_kegg_calcula(qtbot: QtBot, tmp_path: Path) -> None:
    """O servidor usa só os nomes que estão em alguma via, e a conferência usa a mesma conta.
    Medido em 02/10/2026: com 63 nomes enviados e 61 em alguma via, a conferência que contava os
    63 recusava a resposta inteira como «defeito a reportar»."""

    def tres_de_quatro(nomes: list[str]) -> ext.RespostaDoServidor:
        return ext.RespostaDoServidor(
            "ok", dados=ext.le_resposta(_rds_do_servidor(nomes, acertos=3), nomes)
        )

    envia = Enviador(tres_de_quatro)
    painel = _painel_externo(qtbot, tmp_path, Executor(), enviador=envia, dialogo=Dialogo(True))
    _prepara(painel)
    painel.botao_enriquece.click()
    qtbot.waitUntil(lambda: painel.tabela.rowCount() == 1, timeout=5000)
    assert len(envia.chamadas[0][1]) == 4, "âncora: quatro nomes saíram"
    assert "4 nomes enviados, 3 em alguma via KEGG" in painel.estado.text()


@pytest.mark.parametrize(
    ("nomes_no_envio", "status", "chave"),
    [
        (2, 500, "servidor_falhou_lista_curta"),
        (4, 500, "servidor_recusou"),
        (2, 503, "servidor_recusou"),
    ],
    ids=["lista-curta-500", "lista-longa-500", "lista-curta-503"],
)
def test_o_500_da_lista_curta_tem_mensagem_propria(
    qtbot: QtBot, tmp_path: Path, nomes_no_envio: int, status: int, chave: str
) -> None:
    """O 500 de uma lista de um ou dois compostos é do pedido — medido em 02/10/2026 com um
    composto cujo único acerto cai numa só via —, e a frase genérica mandaria tentar mais tarde.
    Os dois vizinhos ficam com a genérica: a lista longa e o outro status."""

    def recusa(_nomes: list[str]) -> ext.RespostaDoServidor:
        return ext.RespostaDoServidor(
            "servidor_recusou", f"o servidor respondeu {status} Erro", status=status
        )

    envia = Enviador(recusa)
    painel = _painel_externo(qtbot, tmp_path, Executor(), enviador=envia, dialogo=Dialogo(True))
    volcano = _volcano()
    if nomes_no_envio == 2:
        volcano["volcano"]["pvalue"] = [0.001, 0.002, 0.3, 0.4, 0.2, 0.5, None]
    painel.recebe_volcano(volcano)
    painel._carrega_tabela()
    painel._confirma_colunas()
    painel.botao_enriquece.click()
    frase = ej.MENSAGENS[chave].frase
    qtbot.waitUntil(lambda: frase in painel.estado.text(), timeout=5000)
    assert len(envia.chamadas[0][1]) == nomes_no_envio, "âncora: o tamanho da lista enviada"
