"""O recorte da entrada do enriquecimento — núcleo puro.

Os dados são feitos à mão, com nomes públicos de compostos (regra 1 do README). O que se testa são
as contagens, o corte por composto e as recusas — propriedades do mecanismo, não de um estudo.
"""

from __future__ import annotations

import ast
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from refino.pathways import enriquecimento as en
from refino.pathways.enriquecimento import Corte, Mapeado

NAN = float("nan")


def _volcano(pares: dict[str, float]) -> tuple[list[str], list[float]]:
    return list(pares), list(pares.values())


def test_os_degraus_contam_linhas_comum_e_sem_p() -> None:
    linhas = [(f"f{i}", f"C{i}", None) for i in range(10)]
    feats, ps = _volcano(
        {"f0": 0.01, "f1": 0.02, "f2": 0.2, "f3": 0.5, "f4": NAN, "f5": NAN, "x": 0.1}
    )
    r = en.recorta(linhas, feats, ps, Corte("p", 0.05))
    d = r.degraus
    assert (d.linhas_lidas, d.features_em_comum, d.features_sem_p) == (10, 6, 2)
    assert d.compostos_fora_da_execucao == 4, "f6 a f9 não estão na execução"
    assert r.fundo == ("C0", "C1", "C2", "C3"), "o fundo são os compostos com p calculado"


def test_dois_sinais_do_mesmo_composto_contam_uma_vez() -> None:
    linhas = [("f1", "Glycine", None), ("f2", "Glycine", None), ("f3", "Urea", None)]
    feats, ps = _volcano({"f1": 0.001, "f2": 0.6, "f3": 0.4})
    r = en.recorta(linhas, feats, ps, Corte("p", 0.05))
    assert r.lista == ("Glycine",)
    gly = next(c for c in r.testados if c.nome == "Glycine")
    assert gly.features == ("f1", "f2")
    assert math.isclose(gly.p, 1 - (1 - 0.001) ** 2), (
        "o p do composto é o mínimo corrigido por Šidák"
    )


def test_sidak_tira_o_composto_de_muitos_aductos_que_passaria_pela_feature_minima() -> None:
    """Par causal da decisão 10: pela feature mínima ele passaria; pelo composto, não."""
    linhas = [(f"a{i}", "Adutos", None) for i in range(5)] + [("b", "Simples", None)]
    feats, ps = _volcano({"a0": 0.03, "a1": 0.9, "a2": 0.9, "a3": 0.9, "a4": 0.9, "b": 0.03})
    r = en.recorta(linhas, feats, ps, Corte("p", 0.05))
    assert "Simples" in r.lista and "Adutos" not in r.lista


def test_corte_por_q_sem_nada_abaixo_da_lista_vazia() -> None:
    linhas = [(f"f{i}", f"C{i}", None) for i in range(4)]
    feats, ps = _volcano({f"f{i}": 0.4 + i * 0.1 for i in range(4)})
    r = en.recorta(linhas, feats, ps, Corte("q", 0.05))
    assert r.lista == ()
    recusa = en.recusa_previa(r)
    assert recusa is not None and recusa.motivo == "lista_vazia"


def test_lista_com_dois_compostos_segue_para_o_calculo() -> None:
    """O usuário retirou o mínimo de 3 — a lista curta calcula, com aviso no resultado.

    O par é o teste acima: a lista VAZIA continua recusada, então `recusa_previa` não ficou muda.
    """
    linhas = [(f"f{i}", f"C{i}", None) for i in range(6)]
    feats, ps = _volcano({"f0": 0.001, "f1": 0.002, "f2": 0.5, "f3": 0.6, "f4": 0.7, "f5": 0.8})
    r = en.recorta(linhas, feats, ps, Corte("p", 0.05))
    assert len(r.lista) == 2, "âncora: o corte deixa exactamente dois compostos"
    assert en.recusa_previa(r) is None


@pytest.mark.parametrize(
    ("na_lista", "avisa"), [(0, False), (1, True), (2, True), (3, False), (40, False)]
)
def test_aviso_de_poucos_vale_para_um_e_dois_compostos(na_lista: int, avisa: bool) -> None:
    """O aviso nas fronteiras: zero não chega ao cálculo (é recusa), e três já não avisa."""
    aviso = en.aviso_de_poucos(na_lista)
    assert (aviso is not None) is avisa, aviso
    if aviso:
        assert ("1 composto " if na_lista == 1 else f"{na_lista} compostos") in aviso
        assert "cautela" in aviso


def test_lista_igual_ao_fundo_e_recusada_e_oferece_a_biblioteca() -> None:
    """AE1: a tabela traz só as features escolhidas, e o corte deixa todas na lista."""
    linhas = [(f"f{i}", f"C{i}", None) for i in range(5)]
    feats, ps = _volcano({f"f{i}": 0.001 for i in range(5)})
    recusa = en.recusa_previa(en.recorta(linhas, feats, ps, Corte("p", 0.05)))
    assert recusa is not None
    assert (recusa.motivo, recusa.oferece_fundo_da_biblioteca) == ("igual_ao_fundo", True)


def test_o_esperado_por_acaso_e_contado_em_compostos() -> None:
    linhas = [(f"f{i}", f"C{i}", None) for i in range(1000)]
    feats, ps = _volcano({f"f{i}": (i + 1) / 1001 for i in range(1000)})
    r = en.recorta(linhas, feats, ps, Corte("p", 0.05))
    assert math.isclose(r.esperado_por_acaso, 50.0)
    q = en.recorta(linhas, feats, ps, Corte("q", 0.05))
    assert math.isclose(q.esperado_por_acaso, 0.05 * len(q.lista))
    n = en.recorta(linhas, feats, ps, Corte("n_melhores", 10))
    assert len(n.lista) == 10
    assert math.isclose(n.esperado_por_acaso, (10 / 1001) * 1000)


def test_candidato_aceito_vence_e_os_outros_sao_contados() -> None:
    linhas = [
        ("f1", "D-Glutamic acid", False),
        ("f1", "L-Glutamic acid", True),
        ("f2", "Urea", None),
    ]
    feats, ps = _volcano({"f1": 0.01, "f2": 0.02})
    r = en.recorta(linhas, feats, ps, Corte("p", 0.05))
    assert set(r.fundo) == {"L-Glutamic acid", "Urea"}
    assert r.degraus.candidatos_descartados == 1


@pytest.mark.parametrize(
    ("tipo", "valor"), [("p", 0.0), ("q", 1.5), ("n_melhores", 2.5), ("x", 0.1)]
)
def test_corte_invalido_e_recusado(tipo: str, valor: float) -> None:
    with pytest.raises(en.EntradaInvalidaError):
        Corte(tipo, valor)  # type: ignore[arg-type]


def _m(nome: str, hmdb: str | None) -> Mapeado:
    return Mapeado(nome, nome if hmdb else None, hmdb)


def test_prepara_ora_conta_depois_da_intersecao_com_o_fundo() -> None:
    fundo = [_m(f"C{i}", f"HMDB{i}") for i in range(8)]
    lista = [_m("C0", "HMDB0"), _m("C1", "HMDB1"), _m("Inventado", None), _m("Fora", "HMDB99")]
    # «Fora» mapeou mas não está no fundo: o núcleo manda só a interseção, que é o que a ORA testa.
    # Com dois compostos na interseção a ORA segue; isto já foi «menos de 3».
    curta = en.prepara_ora(lista, fundo)
    assert isinstance(curta, en.PreparoDaOra)
    assert curta.lista == ("C0", "C1")
    ok = en.prepara_ora([*lista, _m("C2", "HMDB2")], fundo)
    assert isinstance(ok, en.PreparoDaOra)
    assert ok.lista == ("C0", "C1", "C2")
    assert ok.nao_mapeados_da_lista == ("Inventado",)


def test_prepara_ora_deduplica_pelo_identificador_e_acusa_lista_igual_ao_fundo() -> None:
    fundo = [_m("C0", "H0"), _m("C1", "H1"), _m("C2", "H2")]
    lista = [_m("C0", "H0"), Mapeado("Sinônimo de C0", "C0", "H0"), _m("C1", "H1"), _m("C2", "H2")]
    r = en.prepara_ora(lista, fundo)
    assert (
        isinstance(r, en.Recusa) and r.motivo == "igual_ao_fundo" and r.oferece_fundo_da_biblioteca
    )
    biblioteca = en.prepara_ora(lista, fundo, fundo_da_biblioteca=True)
    assert isinstance(biblioteca, en.PreparoDaOra) and biblioteca.fundo is None


def test_nada_mapeado_e_recusa_propria() -> None:
    r = en.prepara_ora([_m("X", None)], [_m("C0", "H0")])
    assert isinstance(r, en.Recusa) and r.motivo == "nada_mapeou"


def test_canonicos_do_envio_deduplica_pelo_identificador_e_deixa_de_fora_o_que_nao_mapeou() -> None:
    """O modo externo envia o canônico confirmado, um por identificador HMDB, sem fundo."""
    mapa = [
        _m("Glycine", "H1"),
        Mapeado("Glicocola", "Glycine", "H1"),  # sinônimo: o mesmo identificador, conta uma vez
        _m("Urea", "H2"),
        _m("Inventado", None),
        Mapeado("Sem identificador", "Algo", None),  # canônico sem HMDB não sai
        Mapeado("Outro id, mesmo nome", "Urea", "H9"),  # o mesmo texto não vai duas vezes
    ]
    assert en.canonicos_do_envio(mapa) == ("Glycine", "Urea")


def test_canonicos_do_envio_segue_a_mesma_regra_da_ora_local_com_o_fundo_da_biblioteca() -> None:
    """A mesma regra de `_canonicos`: sem fundo, a lista é a que a ORA local montaria sem fundo.

    Uma regra própria faria o modo externo e o local divergirem sobre a mesma tabela em silêncio.
    """
    mapa = [
        _m("C2", "H2"),
        _m("C0", "H0"),
        Mapeado("Sinônimo de C0", "C0", "H0"),
        _m("X", None),
        _m("C1", "H1"),
    ]
    local = en.prepara_ora(mapa, [], fundo_da_biblioteca=True)
    assert isinstance(local, en.PreparoDaOra) and local.lista, "âncora: a ORA local aceita"
    assert en.canonicos_do_envio(mapa) == local.lista == ("C0", "C1", "C2")


def test_canonicos_do_envio_sem_nada_mapeado_e_a_recusa_propria() -> None:
    recusa = en.canonicos_do_envio([_m("X", None), Mapeado("Y", "Y", None)])
    assert isinstance(recusa, en.Recusa) and recusa.motivo == "nada_mapeou"
    assert not recusa.oferece_fundo_da_biblioteca, "no modo externo não há fundo a oferecer"
    # par causal: um nome mapeado basta para não recusar
    assert en.canonicos_do_envio([_m("X", None), _m("Y", "HY")]) == ("Y",)


def test_g1_recusa_nome_de_amostra_e_aceita_a_mesma_carga_sem_ele() -> None:
    """AE2, par causal: a mesma carga, com e sem o nome proibido."""
    permitidos = ["Glycine", "Urea", "AMOSTRA_07"]
    proibidos = ["amostra_07", "Caso", "Controle", "Description"]
    with pytest.raises(en.EnvioRecusadoError, match="AMOSTRA_07"):
        en.confere_g1(["Glycine", "AMOSTRA_07"], permitidos=permitidos, proibidos=proibidos)
    en.confere_g1(["Glycine", "Urea"], permitidos=permitidos, proibidos=proibidos)
    with pytest.raises(en.EnvioRecusadoError, match="não é um nome"):
        en.confere_g1(["Glycine", "Lactose"], permitidos=permitidos, proibidos=proibidos)


def test_g1_sem_vocabulario_proibido_levanta_em_vez_de_aprovar() -> None:
    with pytest.raises(ValueError, match="vazio"):
        en.confere_g1(["Glycine"], permitidos=["Glycine"], proibidos=["  "])


def test_o_p_da_via_confere_dentro_do_signif_e_acusa_o_trocado() -> None:
    exato = en.p_hipergeometrico(acertos=10, total=28, na_lista=25, universo=1002)
    assert en.p_da_via_confere(float(f"{exato:.3g}"), 10, 28, 25, 1002)
    assert not en.p_da_via_confere(exato * 1.02, 10, 28, 25, 1002), "um p 2% fora tem de acusar"
    # âncora contra a tabela que o MetaboAnalystR devolveu na árvore embarcada (Urea Cycle, 01/10)
    assert en.p_da_via_confere(1.24e-10, 10, 28, 25, 1002)


def test_o_nucleo_nao_importa_io_nem_desktop() -> None:
    arvore = ast.parse(Path(en.__file__).read_text(encoding="utf-8"))
    importados = {n.module or "" for n in ast.walk(arvore) if isinstance(n, ast.ImportFrom)} | {
        a.name for n in ast.walk(arvore) if isinstance(n, ast.Import) for a in n.names
    }
    assert importados, "âncora: o módulo importa alguma coisa"
    assert not [m for m in importados if m.startswith(("refino.io", "refino.desktop"))]


def test_tipo_do_universo_fora_do_contrato_e_resposta_malformada() -> None:
    """O tipo vem do R; fora dos dois valores, a tela o leria como biblioteca inteira."""
    resposta = {
        "biblioteca": "smpdb_pathway",
        "universo": {"tipo": "fundo_do_ensaio", "tamanho": 6},
        "na_lista": 4,
        "vias": [],
    }
    assert en.ora_da_resposta(resposta).tipo_do_universo == "fundo_do_ensaio"
    resposta["universo"] = {"tipo": "outra coisa", "tamanho": 6}
    with pytest.raises(en.RespostaInvalidaError, match=r"universo.tipo"):
        en.ora_da_resposta(resposta)


# ---------------------------------------------------------------------------
# A fronteira de confiança com o processo R: cada guarda, com o seu par causal
# ---------------------------------------------------------------------------
#
# `ora_da_resposta` e `mapeamento_da_resposta` são onde o que o R devolveu vira estrutura tipada.
# Cada caso abaixo planta UM defeito numa resposta válida e confere duas coisas: que a recusa vem
# **da guarda que o caso nomeia** (o `match` é a mensagem dela) e não de outra que também
# apanharia a planta — contagem negativa, por exemplo, também reprovaria no p recalculado, e
# `hypergeom.sf(-2, ...)` é 1,0, então a guarda de contagem é a única que a recusa pelo motivo
# certo. A resposta sem planta, na mesma função, tem de passar: é o par causal.

Mutador = Callable[[dict[str, Any]], None]


def _planta(*caminho: str | int, valor: object) -> Mutador:
    def aplica(resposta: dict[str, Any]) -> None:
        alvo: Any = resposta
        for passo in caminho[:-1]:
            alvo = alvo[passo]
        alvo[caminho[-1]] = valor

    return aplica


def _remove(*caminho: str | int) -> Mutador:
    def aplica(resposta: dict[str, Any]) -> None:
        alvo: Any = resposta
        for passo in caminho[:-1]:
            alvo = alvo[passo]
        del alvo[caminho[-1]]

    return aplica


def _resposta_da_ora() -> dict[str, Any]:
    """Uma via com o p que o R devolveria: o hipergeométrico das contagens, a 3 algarismos."""
    exato = en.p_hipergeometrico(acertos=10, total=28, na_lista=25, universo=1002)
    return {
        "biblioteca": "smpdb_pathway",
        "universo": {"tipo": "fundo_do_ensaio", "tamanho": 1002},
        "na_lista": 25,
        "vias": [
            {
                "via": "Urea Cycle",
                "total": 28,
                "esperado": 25 * 28 / 1002,
                "acertos": 10,
                "p": float(f"{exato:.3g}"),
                "fdr": 0.01,
                "compostos": [f"C{i}" for i in range(10)],
            }
        ],
    }


@pytest.mark.parametrize(
    ("planta", "guarda"),
    [
        pytest.param(
            _planta("vias", 0, "acertos", valor=9),
            r"9 acertos e 10 compostos",
            id="acertos_diferente_do_numero_de_compostos",
        ),
        pytest.param(
            _planta("vias", 0, "acertos", valor=-1),
            r"«acertos» não é contagem",
            id="acertos_negativo",
        ),
        pytest.param(
            _planta("vias", 0, "acertos", valor=2.5),
            r"«acertos» não é contagem",
            id="acertos_fracionario",
        ),
        pytest.param(
            _planta("na_lista", valor=-1), r"«na_lista» não é contagem", id="na_lista_negativa"
        ),
        pytest.param(
            _planta("universo", "tamanho", valor=1002.5),
            r"«universo.tamanho» não é contagem",
            id="universo_fracionario",
        ),
        pytest.param(_planta("vias", 0, "p", valor=NAN), r"«p» não é número", id="p_nan"),
        pytest.param(_planta("vias", 0, "p", valor=True), r"«p» não é número", id="p_booleano"),
        pytest.param(_planta("vias", 0, "p", valor="0.01"), r"«p» não é número", id="p_texto"),
        pytest.param(
            _planta("vias", 0, "p", valor=0.5),
            r"não é o hipergeométrico das contagens",
            id="p_que_nao_e_o_das_contagens",
        ),
        pytest.param(
            _remove("vias", 0, "compostos"),
            r"não tem o campo «compostos»",
            id="via_sem_compostos",
        ),
        pytest.param(
            _planta("vias", 0, "compostos", valor=list(range(10))),
            r"«compostos» fora do contrato",
            id="compostos_que_nao_sao_texto",
        ),
        pytest.param(
            _planta("vias", 0, "compostos", valor="abcdefghij"),
            r"«compostos» fora do contrato",
            id="compostos_que_nao_e_lista",
        ),
        pytest.param(_planta("vias", valor={}), r"«vias» não é lista", id="vias_que_nao_e_lista"),
        pytest.param(_remove("vias"), r"não tem o campo «vias»", id="resposta_sem_vias"),
        pytest.param(
            _planta("vias", valor=["Urea Cycle"]),
            r"não tem o campo «compostos»",
            id="via_que_nao_e_objeto",
        ),
    ],
)
def test_ora_da_resposta_recusa_cada_defeito_plantado_pela_guarda_que_o_nomeia(
    planta: Mutador, guarda: str
) -> None:
    sem_planta = en.ora_da_resposta(_resposta_da_ora())
    assert [(v.via, v.acertos, len(v.compostos)) for v in sem_planta.vias] == [
        ("Urea Cycle", 10, 10)
    ]
    defeituosa = _resposta_da_ora()
    planta(defeituosa)
    assert defeituosa != _resposta_da_ora(), "a planta tem de aterrar na resposta"
    with pytest.raises(en.RespostaInvalidaError, match=guarda):
        en.ora_da_resposta(defeituosa)


def _resposta_do_mapeamento() -> dict[str, Any]:
    return {
        "lista": {
            "mapeamento": [
                {"consulta": "Glycine", "casado": "Glycine", "hmdb": "HMDB0000123"},
                {"consulta": "Inventado", "casado": None, "hmdb": None},
            ]
        },
        "fundo": {
            "mapeamento": [{"consulta": "Urea", "casado": "Urea", "hmdb": "HMDB0000294"}],
        },
    }


def test_mapeamento_da_resposta_le_a_lista_e_o_fundo_e_aceita_fundo_nulo() -> None:
    lista, fundo = en.mapeamento_da_resposta(_resposta_do_mapeamento())
    assert lista == (Mapeado("Glycine", "Glycine", "HMDB0000123"), Mapeado("Inventado", None, None))
    assert fundo == (Mapeado("Urea", "Urea", "HMDB0000294"),)
    sem_fundo = _resposta_do_mapeamento()
    sem_fundo["fundo"] = None
    assert en.mapeamento_da_resposta(sem_fundo)[1] == ()


@pytest.mark.parametrize(
    ("planta", "guarda"),
    [
        pytest.param(_remove("lista"), r"não tem o campo «lista»", id="resposta_sem_lista"),
        # O par de `test_..._aceita_fundo_nulo`: `null` é «sem fundo», a chave ausente é outro
        # contrato — o roteiro a declara em toda resposta de `mapear`.
        pytest.param(_remove("fundo"), r"não tem o campo «fundo»", id="resposta_sem_a_chave_fundo"),
        pytest.param(
            _remove("lista", "mapeamento"),
            r"não tem o campo «mapeamento»",
            id="bloco_sem_mapeamento",
        ),
        pytest.param(
            _planta("lista", "mapeamento", valor="Glycine"),
            r"«mapeamento» não é lista",
            id="mapeamento_que_nao_e_lista",
        ),
        pytest.param(
            _planta("fundo", "mapeamento", valor="Urea"),
            r"«mapeamento» não é lista",
            id="mapeamento_do_fundo_que_nao_e_lista",
        ),
        pytest.param(
            _remove("lista", "mapeamento", 0, "hmdb"),
            r"não tem o campo «hmdb»",
            id="linha_sem_hmdb",
        ),
        pytest.param(
            _planta("lista", "mapeamento", valor=["Glycine"]),
            r"não tem o campo «consulta»",
            id="linha_que_nao_e_objeto",
        ),
        pytest.param(
            _planta("lista", "mapeamento", 0, "consulta", valor=5),
            r"linha de mapeamento fora do contrato",
            id="consulta_que_nao_e_texto",
        ),
        pytest.param(
            _planta("lista", "mapeamento", 0, "casado", valor=5),
            r"linha de mapeamento fora do contrato",
            id="casado_que_nao_e_texto_nem_nulo",
        ),
        pytest.param(
            _planta("fundo", "mapeamento", 0, "hmdb", valor=7),
            r"linha de mapeamento fora do contrato",
            id="hmdb_do_fundo_que_nao_e_texto_nem_nulo",
        ),
    ],
)
def test_mapeamento_da_resposta_recusa_cada_defeito_plantado_pela_guarda_que_o_nomeia(
    planta: Mutador, guarda: str
) -> None:
    lista, _ = en.mapeamento_da_resposta(_resposta_do_mapeamento())
    assert len(lista) == 2, "sem a planta a resposta passa: é o par causal"
    defeituosa = _resposta_do_mapeamento()
    planta(defeituosa)
    assert defeituosa != _resposta_do_mapeamento(), "a planta tem de aterrar na resposta"
    with pytest.raises(en.RespostaInvalidaError, match=guarda):
        en.mapeamento_da_resposta(defeituosa)
