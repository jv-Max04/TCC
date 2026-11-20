"""Desanexar escolhas preserva os arquivos e invalida somente os dados dependentes.

Os pequenos CSVs sintéticos servem apenas para o seletor e o multipart: nenhum
teste deste módulo os apresenta como matriz científica ou referência do núcleo.
"""

from __future__ import annotations

from pathlib import Path
from threading import Event
from typing import Any

import pytest

from tests.qt_guard import garante_plataforma_offscreen, membros_ausentes, sonda_o_ambiente

_FALTAM = membros_ausentes(sonda_o_ambiente())
if _FALTAM:  # pragma: no cover — a guarda é exercitada em tests/test_qt_guard.py
    pytest.skip(
        f"grupo opcional `desktop` ausente: falta {', '.join(_FALTAM)}. "
        'Instale com `uv pip install -e ".[desktop]"` para exercitar os testes de janela.',
        allow_module_level=True,
    )

garante_plataforma_offscreen()

from refino.desktop.janela.arquivos import (  # noqa: E402, I001
    ROTULO_DA_LISTA,
    ROTULO_DAS_MEDIDAS,
    ROTULO_DOS_FATORES,
    SEM_ARQUIVO,
    PainelDeArquivos,
)

from PySide6.QtTest import QSignalSpy  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402


@pytest.fixture
def anexos(tmp_path: Path) -> dict[str, Path]:
    """Arquivos distintos permitem detectar troca ou perda de bytes no payload."""
    arquivos = {
        "medidas": tmp_path / "measurements_sintetico.csv",
        "lista": tmp_path / "sample_list_sintetica.csv",
        "fatores": tmp_path / "fatores_sinteticos.csv",
    }
    for chave, caminho in arquivos.items():
        caminho.write_bytes(f"tipo,valor\n{chave},1\n".encode())
    return arquivos


def _escolhedor(anexos: dict[str, Path]) -> Any:
    por_titulo = {
        ROTULO_DAS_MEDIDAS: anexos["medidas"],
        ROTULO_DA_LISTA: anexos["lista"],
        ROTULO_DOS_FATORES: anexos["fatores"],
    }
    return lambda titulo: por_titulo[titulo]


def _anexa_todos(painel: PainelDeArquivos) -> None:
    painel.escolhe_medidas()
    painel.escolhe_lista()
    painel.escolhe_fatores()


@pytest.fixture
def painel(qtbot: Any, anexos: dict[str, Path]) -> PainelDeArquivos:
    montado = PainelDeArquivos(_escolhedor(anexos))
    qtbot.addWidget(montado)
    return montado


@pytest.mark.parametrize("chave", ["medidas", "lista"])
def test_remover_obrigatorio_desfaz_o_par_e_preserva_os_outros_anexos(
    chave: str, painel: PainelDeArquivos, anexos: dict[str, Path]
) -> None:
    _anexa_todos(painel)
    antes = painel.arquivos_da_execucao()
    no_disco = {nome: caminho.read_bytes() for nome, caminho in anexos.items()}
    sinal = QSignalSpy(painel.arquivo_removido)
    assert not painel.botoes_de_remocao[chave].isHidden()

    painel.botoes_de_remocao[chave].click()

    assert painel.par is None
    with pytest.raises(ValueError, match="nenhum par"):
        painel.arquivos_para_envio()
    assert sinal.count() == 1 and sinal.at(0) == [chave]
    assert painel.estados[chave].text() == SEM_ARQUIVO
    assert painel.botoes_de_remocao[chave].isHidden()
    assert painel.selo.text() == "1 / 2 obrigatórios"
    assert painel.selo.tipo == "aviso"
    outra = "lista" if chave == "medidas" else "medidas"
    assert painel.estados[outra].text() == anexos[outra].name
    assert painel.estados["fatores"].text() == anexos["fatores"].name

    # Ao recolocar só a metade retirada, os outros dois payloads permanecem intactos.
    (painel.escolhe_medidas if chave == "medidas" else painel.escolhe_lista)()
    assert painel.arquivos_da_execucao() == antes
    for nome, caminho in anexos.items():
        assert caminho.read_bytes() == no_disco[nome], "desanexar alterou o arquivo original"


def test_remover_fatores_preserva_a_identidade_do_par_e_os_bytes_obrigatorios(
    painel: PainelDeArquivos, anexos: dict[str, Path]
) -> None:
    _anexa_todos(painel)
    par = painel.par
    antes = painel.arquivos_para_envio()
    conteudo_original = anexos["fatores"].read_bytes()
    sinal = QSignalSpy(painel.arquivo_removido)
    assert "normalization_factors" in painel.arquivos_da_execucao()

    painel.botoes_de_remocao["fatores"].click()

    assert painel.par is par
    assert painel.arquivos_da_execucao() == antes
    assert sinal.count() == 1 and sinal.at(0) == ["fatores"]
    assert painel.estados["fatores"].text() == SEM_ARQUIVO
    assert painel.botoes_de_remocao["fatores"].isHidden()
    assert painel.selo.text() == "2 / 2 obrigatórios"
    assert anexos["fatores"].read_bytes() == conteudo_original


@pytest.mark.parametrize("chave", ["medidas", "lista", "fatores"])
def test_remover_arquivo_ausente_e_idempotente_sem_sinal_falso(
    chave: str, painel: PainelDeArquivos
) -> None:
    sinal = QSignalSpy(painel.arquivo_removido)
    assert painel.botoes_de_remocao[chave].isHidden()
    painel.remover_arquivo(chave)
    assert sinal.count() == 0
    _anexa_todos(painel)

    painel.remover_arquivo(chave)
    painel.remover_arquivo(chave)

    assert sinal.count() == 1 and sinal.at(0) == [chave]
    assert painel.botoes_de_remocao[chave].isHidden()


def test_tipo_de_arquivo_desconhecido_levanta_sem_alterar_anexos(
    painel: PainelDeArquivos,
) -> None:
    _anexa_todos(painel)
    antes = painel.arquivos_da_execucao()
    par = painel.par
    sinal = QSignalSpy(painel.arquivo_removido)

    with pytest.raises(ValueError, match="desconhecido"):
        painel.remover_arquivo("desconhecido")

    assert painel.par is par
    assert painel.arquivos_da_execucao() == antes
    assert sinal.count() == 0


@pytest.mark.parametrize("chave", ["medidas", "lista"])
def test_reanexar_obrigatorio_le_os_bytes_novos_e_recompoe_o_par(
    chave: str, painel: PainelDeArquivos, anexos: dict[str, Path]
) -> None:
    _anexa_todos(painel)
    anterior = painel.par
    antes = painel.arquivos_da_execucao()
    painel.botoes_de_remocao[chave].click()
    novos = b"tipo,valor\nreanexado,99\n"
    anexos[chave].write_bytes(novos)

    (painel.escolhe_medidas if chave == "medidas" else painel.escolhe_lista)()

    assert painel.par is not None and painel.par is not anterior
    campos = painel.arquivos_da_execucao()
    alterado = "measurements" if chave == "medidas" else "sample_list"
    assert campos[alterado][1] == novos
    assert campos[alterado][1] != antes[alterado][1]
    for nome, valor in antes.items():
        if nome != alterado:
            assert campos[nome] == valor
    assert painel.selo.text() == "2 / 2 obrigatórios"


@pytest.mark.parametrize("chave", ["medidas", "fatores"])
def test_remocao_atualiza_as_etapas_e_metadados_da_janela_no_mesmo_gesto(
    chave: str, qtbot: Any, anexos: dict[str, Path]
) -> None:
    from refino.desktop.janela.principal import JanelaPrincipal

    class SemRede:
        def post(self, url: str, **kwargs: Any) -> Any:
            raise AssertionError("desanexar arquivo não deve chamar o servidor")

        def get(self, url: str, **kwargs: Any) -> Any:
            raise AssertionError("desanexar arquivo não deve chamar o servidor")

    janela = JanelaPrincipal(SemRede(), escolhedor=_escolhedor(anexos))
    qtbot.addWidget(janela)
    _anexa_todos(janela.arquivos)
    assert janela.pipeline.pre_preenche("n_batches", 2)
    janela.pipeline.define_amostras([1, 2, 3, 4, 5, 6], [True] * 6)
    janela.pipeline.define_cortes_propostos([4])
    assert janela.pipeline.qcs_por_lote.text()
    assert janela.pipeline.cortes_em_uso(2) == (4,)
    atual = janela._campos_atuais()
    assert atual is not None
    # A cadeia feita aba a aba, cada etapa sobre a entrada mais tratada que existe — o que o
    # «Executar pipeline» deixava feito de um clique só (`concluiu_tudo`) até sair com a aba
    # «Executar e exportar».
    from refino.desktop.etapas import ETAPAS, IDENTIDADE_DA_MATRIZ, fotografia_da_etapa

    tela, identidade = atual
    matriz = identidade[IDENTIDADE_DA_MATRIZ]
    for etapa in ETAPAS:
        entrada = janela._etapas.entrada(etapa, matriz).escolhida
        janela._etapas.disparou(
            etapa,
            fotografia_da_etapa(tela, etapa, identidade, entrada),
            campos=janela._etapas.campos_da_entrada(etapa, entrada, tela, matriz),
            entrada=entrada,
        )
        janela._etapas.concluiu(etapa, correcao="none" if etapa == ETAPAS[0] else None)
    assert janela._atualiza_as_etapas() == ("em_dia",) * 3

    janela.arquivos.botoes_de_remocao[chave].click()

    # Sem esperar o relógio nem chamar o atualizador de novo: o sinal já atualizou a tela.
    if chave == "medidas":
        assert janela.pipeline._estados_das_etapas == ("desatualizada",) * 3
        assert not janela.pipeline.qcs_por_lote.text()
        assert janela.pipeline.cortes_em_uso(2) is None
    else:
        # Os fatores são da Normalização. A Análise feita sobre ela continua em dia — o resultado
        # dela corresponde à entrada que existe —, e o aviso dela aponta para a Normalização.
        assert janela.pipeline._estados_das_etapas == ("em_dia", "desatualizada", "em_dia")
        assert janela._alvos_dos_avisos == {2: 2, 3: 2}
        assert janela.pipeline.qcs_por_lote.text()
        assert janela.pipeline.cortes_em_uso(2) == (4,)


def test_validacao_atrasada_nao_repoe_qcs_apos_remover_e_reanexar(
    qtbot: Any, anexos: dict[str, Path]
) -> None:
    """A mesma resposta é descartada na geração antiga e aplicada na geração atual."""
    from refino.desktop.janela.principal import JanelaPrincipal
    from refino.interface.cliente import Resposta

    janela = JanelaPrincipal(None, escolhedor=_escolhedor(anexos))
    qtbot.addWidget(janela)
    _anexa_todos(janela.arquivos)
    assert janela.pipeline.pre_preenche("n_batches", 2)
    resposta = Resposta(
        codigo=200,
        ok=True,
        causa=None,
        mensagem="OK",
        corpo={
            "aceita": True,
            "problemas": [],
            "n_features": 2,
            "n_samples": 6,
            "sample_names": ["sintetica_1", "sintetica_2"],
            "feature_names": ["feature_sintetica"],
            "ordens_de_injecao": [1, 2, 3, 4, 5, 6],
            "qc": [True] * 6,
        },
    )
    with qtbot.waitSignal(janela.resposta_recebida, timeout=5_000):
        janela.enfileira(lambda: resposta, chave="validacao")
    assert janela.leitura.cartoes["validacao"].selo.text() == "Aceita"
    assert janela.pipeline.qcs_por_lote.text(), "âncora: a validação precisa repor os QCs"

    comecou = Event()
    liberar = Event()
    recebidas = QSignalSpy(janela.resposta_recebida)

    def atrasada() -> Resposta:
        comecou.set()
        if not liberar.wait(10):
            raise AssertionError("o teste não liberou a resposta atrasada")
        return resposta

    try:
        janela.enfileira(atrasada, chave="validacao")
        assert comecou.wait(5), "a leitura não chegou à piscina"
        janela.arquivos.botoes_de_remocao["medidas"].click()
        janela.arquivos.escolhe_medidas()
        assert janela.arquivos.par is not None, "a guarda precisa agir mesmo com novo par completo"
        assert not janela.pipeline.qcs_por_lote.text()
        assert all(cartao.selo.text() == "Não lido" for cartao in janela.leitura.cartoes.values())
        liberar.set()
        assert janela.espera_a_piscina(5_000)
        QApplication.processEvents()

        assert recebidas.count() == 0, "a leitura da geração removida foi entregue à interface"
        assert not janela.pipeline.qcs_por_lote.text()
        assert all(cartao.selo.text() == "Não lido" for cartao in janela.leitura.cartoes.values())

        with qtbot.waitSignal(janela.resposta_recebida, timeout=5_000):
            janela.enfileira(lambda: resposta, chave="validacao")
        assert recebidas.count() == 1
        assert janela.leitura.cartoes["validacao"].selo.text() == "Aceita"
        assert janela.pipeline.qcs_por_lote.text()
    finally:
        liberar.set()
        assert janela.drena_a_piscina(5_000)
        QApplication.processEvents()
