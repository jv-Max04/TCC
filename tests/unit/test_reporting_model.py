"""Os pares causais do relatório.

**Os quatro testes obrigatórios são NEGATIVOS EM PROCESSO**
: a planta é argumento de `defeitos_do_relatorio`, então o negativo percorre exactamente
o mesmo caminho que o portão e não consegue ficar inerte.
"""

import pytest

from refino.reporting.avisos import AVISOS, carrega_avisos
from refino.reporting.model import (
    ITENS_OBRIGATORIOS,
    Afirmacao,
    ItemApendice,
    ItemCorpo,
    Relatorio,
    defeitos_do_relatorio,
)

APENDICE = (
    ItemApendice("apx-proc-dv18", "Procedência do SERRF", "onze passos verificados na U12"),
    ItemApendice("apx-proc-dv22", "Perfil do gerador", "6 de 20 fora de tolerância"),
    ItemApendice("apx-proc-m8", "Transferência entre modos", "50,0% contra 60,0%, p = 0,5235"),
    ItemApendice("apx-bloco", "Bloco reexecutável", "refino processar --from-log ..."),
)

METRICAS = {
    "qc_rsd_median": 1.4381,
    "qc_rsd_mean": 1.51,
    "frac_features_under_20": 0.07,
    "qc_dispersion_ratio": 0.9302,
    "pc1_injection_correlation_abs": 0.2894,
}


def _relatorio(**kw: object) -> Relatorio:
    avisos = carrega_avisos()
    base: dict[str, object] = {
        "corpo": tuple(
            ItemCorpo(
                numero=c.numero,
                titulo=c.titulo,
                afirmacoes=(Afirmacao(f"conteúdo do item {c.numero}", "apx-bloco"),),
            )
            for c in ITENS_OBRIGATORIOS
        ),
        "apendice": APENDICE,
        "metricas": METRICAS,
        "metricas_computadas": frozenset(METRICAS),
        "avisos_corpo": {a.id: avisos[a.id]["consequencia"] for a in AVISOS},
        "avisos_apendice": {a.id: avisos[a.id]["procedencia"] for a in AVISOS},
    }
    base.update(kw)
    return Relatorio(**base)  # type: ignore[arg-type]


def test_o_relatorio_integro_nao_acusa_defeito() -> None:
    """**O par causal, lado positivo.** Sem ele, um portão que sempre acusa passaria nos
    negativos."""
    assert defeitos_do_relatorio(_relatorio()) == []


@pytest.mark.parametrize("numero", [c.numero for c in ITENS_OBRIGATORIOS])
def test_to_10_1_item_removido_reprova_e_o_integro_passa(numero: int) -> None:
    """Doze execuções, uma por item removido. **Localiza, não conta.**"""
    rel = _relatorio()
    mutilado = _relatorio(corpo=tuple(i for i in rel.corpo if i.numero != numero))
    d = defeitos_do_relatorio(mutilado)
    assert any(f"item {numero} do conteúdo obrigatório ausente" in x for x in d)


# **Os NOVE de antes, escritos à mão e NÃO derivados de `ITENS_OBRIGATORIOS`.** Se os dois
# lados viessem da mesma fonte, o teste mediria a constante contra ela própria e o par causal da
# emenda desapareceria no dia em que ela fosse desfeita.
NUMEROS_DE_ANTES_DE_D469: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 8, 9)

# **Os dois acrescentados, também à mão.** Antes eles eram «tudo o que fica fora dos
# nove», e o teste derivava-os assim; com o item 12 essa derivação passou a contar três, e o par
# Causal reprovaria por uma emenda que não é a dele.
NUMEROS_DE_D469: tuple[int, ...] = (10, 11)

# **Os ONZE de antes**, pelo mesmo motivo dos nove: à mão, e não derivados da constante.
NUMEROS_DE_ANTES_DE_D675: tuple[int, ...] = (*NUMEROS_DE_ANTES_DE_D469, *NUMEROS_DE_D469)


def test_d469_o_corpo_de_antes_da_emenda_acusa_os_dois_itens_NOVOS() -> None:  # noqa: N802
    """**O par causal da emenda**, e a planta é o corpo tal como era antes dela.

    A emenda acrescenta dois itens ao conteúdo obrigatório — a dispersão do RSD por
    feature e a composição por classe. Sem este teste, desfazer a emenda não reprovaria nada: os
    testes acima derivam o corpo de `ITENS_OBRIGATORIOS`, então encolher a constante encolheria a
    planta junto e os dois lados voltariam a concordar em silêncio.

    **A âncora de não-vacuidade vem ANTES da asserção e na mesma função**: com os dois itens
    ausentes da constante, `novos` fica vazio e o laço passaria por vacuidade.
    """
    rel = _relatorio()
    antigo = _relatorio(corpo=tuple(i for i in rel.corpo if i.numero in NUMEROS_DE_ANTES_DE_D469))
    novos = tuple(c for c in ITENS_OBRIGATORIOS if c.numero in NUMEROS_DE_D469)
    assert len(novos) == 2, (
        f"a emenda acrescenta DOIS itens ao conteúdo obrigatório e `ITENS_OBRIGATORIOS` "
        f"tem {len(novos)} dos dois que ela nomeia: {[c.numero for c in novos]}"
    )

    achados = defeitos_do_relatorio(antigo)
    for canonico in novos:
        assert any(
            f"item {canonico.numero} do conteúdo obrigatório ausente" in x for x in achados
        ), f"o item {canonico.numero} ({canonico.titulo!r}) não foi acusado: {achados}"


def test_d675_o_corpo_de_antes_da_emenda_acusa_o_item_12_e_so_ele() -> None:
    """**O par causal da emenda**: o corpo dos onze de antes dela reprova pelo item 12.

    A emenda acrescenta a curva do LOESS ao longo das injeções, com o envelope das curvas refeitas
    sem um QC. Pelo argumento do par: os testes acima derivam o corpo da constante, então
    desfazer a emenda encolheria a planta junto e nada reprovaria.

    **«E só ele» é a metade que impede o teste de passar por um corpo que acuse tudo:** os onze de
    antes continuam presentes no corpo antigo, e nenhum deles pode aparecer como ausente.
    """
    rel = _relatorio()
    antigo = _relatorio(corpo=tuple(i for i in rel.corpo if i.numero in NUMEROS_DE_ANTES_DE_D675))
    novos = tuple(c for c in ITENS_OBRIGATORIOS if c.numero not in NUMEROS_DE_ANTES_DE_D675)
    assert [c.numero for c in novos] == [12], (
        "a emenda acrescenta UM item, o 12, e `ITENS_OBRIGATORIOS` tem "
        f"{[c.numero for c in novos]} fora dos onze de antes dela"
    )
    assert "LOESS" in novos[0].titulo and "não é intervalo" in novos[0].titulo, novos[0].titulo

    ausentes = [x for x in defeitos_do_relatorio(antigo) if "conteúdo obrigatório ausente" in x]
    assert len(ausentes) == 1 and "item 12 " in ausentes[0], ausentes
    assert defeitos_do_relatorio(rel) == [], "o corpo com os doze não pode acusar nada"


def test_to_10_2_metrica_unica_reprova() -> None:
    """A regra 5 do README na superfície do relatório."""
    d = defeitos_do_relatorio(
        _relatorio(metricas={"qc_rsd_median": 1.4381}, metricas_computadas=frozenset())
    )
    assert any("Métrica única é proibida" in x for x in d)


def test_to_10_2_metrica_calculada_e_ausente_da_tabela_reprova() -> None:
    """A metade positiva: o relatório não pode mostrar menos do que mediu."""
    d = defeitos_do_relatorio(_relatorio(metricas_computadas=frozenset({*METRICAS, "q2_plsda"})))
    assert any("calculadas e ausentes da tabela" in x and "q2_plsda" in x for x in d)


@pytest.mark.parametrize("aviso_id", [a.id for a in AVISOS])
def test_to_10_3_cada_aviso_removido_do_corpo_reprova(aviso_id: str) -> None:
    """Três execuções, um aviso removido por vez, **do corpo**."""
    rel = _relatorio()
    d = defeitos_do_relatorio(
        _relatorio(avisos_corpo={k: v for k, v in rel.avisos_corpo.items() if k != aviso_id})
    )
    assert any(f"aviso `{aviso_id}` ausente do CORPO" in x for x in d)


@pytest.mark.parametrize("aviso_id", [a.id for a in AVISOS])
def test_to_10_3_cada_aviso_removido_do_apendice_reprova(aviso_id: str) -> None:
    """A outra metade: sem procedência o revisor perde a justificação."""
    rel = _relatorio()
    d = defeitos_do_relatorio(
        _relatorio(avisos_apendice={k: v for k, v in rel.avisos_apendice.items() if k != aviso_id})
    )
    assert any(f"aviso `{aviso_id}` ausente do APÊNDICE" in x for x in d)


def test_o_mesmo_texto_nas_duas_leituras_reprova_por_ar79() -> None:
    """**A defesa estrutural contra a dupla contagem** — a que impede o portão de medir o corpo duas
    vezes."""
    rel = _relatorio()
    d = defeitos_do_relatorio(_relatorio(avisos_apendice=dict(rel.avisos_corpo)))
    assert len(d) == len(AVISOS)
    assert all("MESMO texto" in x and "um enquadramento só" in x for x in d)


def test_to_10_6_ponteiro_pendente_reprova() -> None:
    """Ponteiro apontando para item de apêndice inexistente."""
    rel = _relatorio()
    corpo = (
        ItemCorpo(1, "x", (Afirmacao("afirma algo", "apx-que-nao-existe"),)),
        *rel.corpo[1:],
    )
    d = defeitos_do_relatorio(_relatorio(corpo=corpo))
    assert any("não resolve no apêndice" in x and "apx-que-nao-existe" in x for x in d)


def test_afirmacao_sem_ponteiro_reprova() -> None:
    """Afirmação sem ponteiro repete o defeito de `best_by_metric` sozinho."""
    rel = _relatorio()
    corpo = (ItemCorpo(1, "x", (Afirmacao("afirma algo", ""),)), *rel.corpo[1:])
    d = defeitos_do_relatorio(_relatorio(corpo=corpo))
    assert any("afirmação sem ponteiro" in x for x in d)


def test_ca_10_7_item_vazio_precisa_declarar_o_vazio() -> None:
    """Seção vazia sem legenda é indistinguível de seção quebrada."""
    rel = _relatorio()
    sem_nada = (ItemCorpo(7, "Curva de especificação"), *rel.corpo[:6], *rel.corpo[7:])
    assert any("SEM conteúdo" in x for x in defeitos_do_relatorio(_relatorio(corpo=sem_nada)))

    com_legenda = (
        ItemCorpo(
            7,
            "Curva de especificação",
            vazio_declarado=(
                "vazia porque não há efeito a dissolver: nula com mediana 103 contra 62"
            ),
        ),
        *rel.corpo[:6],
        *rel.corpo[7:],
    )
    assert defeitos_do_relatorio(_relatorio(corpo=com_legenda)) == []
