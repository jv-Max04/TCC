"""A espera dos arneses de captura pelo Palpite medido vê os três desfechos.

`scripts/espera_o_palpite.py` decide, nos dois roteiros de captura, se a janela já tem o Palpite e
pode corrigir. Os testes são em processo: a janela é um duplo cujo estado avança a cada volta do
laço, e o relógio é injetado, então a planta é argumento da função sob teste e nenhum negativo pode
ficar inerte por um intermediário.

O duplo imita só o que a função lê — `esperas_em_curso()` e `_palpite_em_dia` —, e os nomes
são os da janela: `test_o_duplo_le_os_nomes_da_janela` confere que os dois existem em
`JanelaPrincipal`, porque um duplo com o nome errado aprovaria um arnês que, na janela de verdade,
nunca veria o Palpite chegar.

Os três desfechos são vistos: `CHEGOU`, `FALHOU` — pela espera que acaba sem Palpite e
pela que nunca começa — e `TETO`.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "espera_o_palpite.py"
CHAVE = "recomendacao"


def _modulo() -> ModuleType:
    spec = importlib.util.spec_from_file_location("espera_o_palpite", SCRIPT)
    assert spec is not None and spec.loader is not None
    modulo = importlib.util.module_from_spec(spec)
    sys.modules["espera_o_palpite"] = modulo
    spec.loader.exec_module(modulo)
    return modulo


espera = _modulo()


class _Janela:
    """Um duplo cujo estado é um roteiro: cada volta do laço lê o passo seguinte, e o último fica.

    Cada passo é `(em_curso, em_dia)`.
    """

    def __init__(self, passos: list[tuple[bool, bool]]) -> None:
        self.passos = passos
        self.i = 0
        self._palpite_em_dia: object | None = None

    def esperas_em_curso(self) -> tuple[str, ...]:
        em_curso, em_dia = self.passos[min(self.i, len(self.passos) - 1)]
        self._palpite_em_dia = ("identidade",) if em_dia else None
        return (CHAVE,) if em_curso else ()

    def bombeia(self, _ms: int) -> None:
        self.i += 1


class _Relogio:
    """Avança `passo` segundos a cada leitura — o tempo só anda quando a função o pergunta."""

    def __init__(self, passo: float) -> None:
        self.agora = 0.0
        self.passo = passo

    def __call__(self) -> float:
        self.agora += self.passo
        return self.agora


def _espera(janela: _Janela, *, passo: float = 1.0, **kw: Any) -> str:
    desfecho: str = espera.espera_o_palpite(
        janela, janela.bombeia, chave=CHAVE, relogio=_Relogio(passo), **kw
    )
    return desfecho


def test_o_duplo_le_os_nomes_da_janela() -> None:
    """Os nomes que a função lê existem na janela — senão o duplo aprovaria o impossível.

    **Pela FONTE e não por import**, para este arquivo não depender do grupo `desktop`: os arneses
    que usam a espera rodam com ele, mas a regra da espera é testável sem Qt, e um import da janela
    aqui faria o arquivo inteiro errar onde o grupo falta.
    """
    janela = REPO / "src" / "refino" / "desktop" / "janela"
    leitura = (janela / "leitura.py").read_text(encoding="utf-8")
    principal = (janela / "principal.py").read_text(encoding="utf-8")
    assert leitura.count(f'CHAVE_DO_PALPITE: Final = "{CHAVE}"') == 1
    assert principal.count("    def esperas_em_curso(self) -> tuple[str, ...]:") == 1
    # O único sítio que marca o Palpite em dia, e é só com `200` e `correction_proposta`.
    assert principal.count("self._palpite_em_dia = self._palpite_pedido") == 1


def test_chega_depois_de_a_espera_comecar_e_acabar() -> None:
    janela = _Janela([(False, False), (True, False), (True, False), (False, True)])
    assert _espera(janela) == espera.CHEGOU
    assert janela.i == 3


def test_chega_mesmo_sem_ver_a_espera_quando_ela_cabe_numa_volta() -> None:
    """O `202` e o `200` podem chegar na mesma volta do laço — o Palpite guardado."""
    assert _espera(_Janela([(False, False), (False, True)])) == espera.CHEGOU


def test_em_dia_com_a_espera_ainda_viva_continua_a_esperar() -> None:
    janela = _Janela([(True, True), (True, True), (False, True)])
    assert _espera(janela) == espera.CHEGOU
    assert janela.i == 2


def test_a_espera_que_acaba_sem_palpite_falhou() -> None:
    """A rota respondeu mal: a espera saiu e o Palpite não ficou em dia."""
    janela = _Janela([(True, False), (True, False), (False, False)])
    assert _espera(janela) == espera.FALHOU
    assert janela.i == 2


def test_a_espera_que_nunca_comeca_falhou_no_teto_do_inicio() -> None:
    """O disparo recusado: sem `202`, a espera nunca nasce, e o teto curto decide."""
    janela = _Janela([(False, False)])
    assert _espera(janela, teto_do_inicio_s=10.0, teto_s=1000.0) == espera.FALHOU
    # Parou no teto curto, e não no longo.
    assert janela.i < 20


def test_o_palpite_que_nao_termina_bate_no_teto() -> None:
    janela = _Janela([(True, False)])
    assert _espera(janela, teto_do_inicio_s=5.0, teto_s=30.0) == espera.TETO
    assert 25 <= janela.i <= 31


def test_a_frase_do_palpite_le_o_rotulo_e_nao_inventa() -> None:
    class _Rotulo:
        def text(self) -> str:
            return "serrf — a amostra não separa os dois"

    class _Leitura:
        def __init__(self) -> None:
            self.rotulos = {CHAVE: _Rotulo()}

    class _ComRotulo:
        def __init__(self) -> None:
            self.leitura = _Leitura()

    assert espera.frase_do_palpite(_ComRotulo(), CHAVE) == "serrf — a amostra não separa os dois"
    assert espera.frase_do_palpite(_ComRotulo(), "outra") == ""


def test_os_dois_roteiros_de_captura_esperam_o_palpite() -> None:
    """A ligação: os dois arneses que conduzem a janela chamam a espera antes de corrigir.

    Sem ela, cada um voltaria a dar ao Palpite os segundos da árvore, e a captura sairia com a
    correção recusada e `EXIT=0` — o defeito que este arquivo existe para impedir.

    **O que corrige é o botão da Correção desde 09/11/2026**: o «Executar pipeline», que os dois
    roteiros clicavam, saiu com a aba «Executar e exportar», e eles passaram a disparar as etapas
    pelos botões delas. A espera tem de vir antes do primeiro.

    O motivo da espera mudou, e ela fica. Antes, a Correção delegava ao Palpite com a
    caixa «Automático» ticada, e era por isso que ele tinha de chegar antes. A caixa saiu, e a
    Correção leva o corretor do seletor — mas os dois roteiros pedem o Palpite pelo cartão dele, e
    a trava `PALPITE_EM_CURSO` recusa toda correção enquanto ele mede, porque os dois disputariam os
    mesmos núcleos. Sem a espera, a correção dos roteiros seria recusada e a captura sairia com
    `EXIT=0`. **E os roteiros não podem tocar a caixa que saiu**: `janela.pipeline.automatico`
    levantaria `AttributeError` no meio da captura, fora da suíte, que é onde nada o apanharia.

    E nenhum botão de etapa é clicado cru. Chegar o Palpite não
    religa as etapas na hora: quem as religa é o relógio delas, de meio em meio segundo, e um clique
    num botão ainda desligado não faz nada e não diz nada. O roteiro do guia clicou 200 ms depois do
    Palpite e a etapa ficou «não feita» até o teto. Os dois passam por `_clica_quando_ligado`, que
    espera o botão ligar e diz o motivo se ele não ligar.
    """
    for nome in ("captura_telas_do_guia.py",):
        fonte = (REPO / "scripts" / nome).read_text(encoding="utf-8")
        assert fonte.count("espera_o_palpite(janela,") == 1, nome
        assert "botao_disparo" not in fonte, f"{nome} clica o botão que saiu"
        assert "pipeline.automatico" not in fonte, f"{nome} toca a caixa «Automático», que saiu"
        for etapa in (1, 3):
            assert f"botoes_de_etapa[{etapa}].click()" not in fonte, (
                f"{nome} clica cru o botão da etapa {etapa}, sem esperar que ele ligue"
            )
            assert (
                f"_clica_quando_ligado(app, janela.pipeline.botoes_de_etapa[{etapa}])" in fonte
            ), f"{nome} não dispara a etapa {etapa} pela guarda"
        chamada = fonte.index("espera_o_palpite(janela,")
        assert chamada < fonte.index("botoes_de_etapa[1]"), nome


def test_os_dois_roteiros_de_captura_reconhecem_a_conclusao_que_a_tela_escreve() -> None:
    """A tela trocou «concluído» por «Concluído», e os dois roteiros procuravam a forma antiga.

    O laço do guia e o da apresentação só saíam pelo teto, e a captura acusaria «não concluiu» numa
    execução que concluiu. A palavra passou a uma constante de `pipeline.py`, e a igualdade é com o
    que a tela de facto escreve num `200` — o outro caminho, e não um literal copiado.
    """
    from refino.desktop.janela.pipeline import ROTULO_DE_CONCLUSAO, texto_da_leitura

    escrita = texto_da_leitura(200, {}, None)
    assert escrita.startswith(ROTULO_DE_CONCLUSAO), escrita
    for nome in ("captura_telas_do_guia.py",):
        fonte = (REPO / "scripts" / nome).read_text(encoding="utf-8")
        assert "from refino.desktop.janela.pipeline import ROTULO_DE_CONCLUSAO" in fonte, nome
        assert fonte.count("ROTULO_DE_CONCLUSAO") == 2, nome
        assert '"concluído"' not in fonte and '"concluido"' not in fonte, nome
