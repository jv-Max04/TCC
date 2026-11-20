"""O rótulo do campo vazio contra o que o vazio FAZ — item 9 da passagem humana.

**O defeito que o motivou, medido em 07/09/2026.** Três campos da janela usam
`setSpecialValueText` para representar *«não escolhi»*, e **dois dos três rótulos afirmavam
comportamento que o sistema não tem**:

| Campo | O texto de então | O que acontece de facto |
|---|---|---|
| Quantos lotes | *usa a sugestão do diagnóstico* | **um lote** |
| Corte de RSD | *vazio — salta a etapa* | salta a etapa |
| Corte de ausentes | *vazio — usa o padrão do servidor* | **salta a etapa** |

Não existe padrão do servidor: `missing_threshold` entra em `_spec_ou_422` sem substituição, e
`process_matrix` faz `_saltada("drop_by_group_missingness", …)` — a mesma forma exata do corte de
RSD dez linhas acima. **Quem lia *«usa o padrão do servidor»* concluía que houve um corte por
ausentes com algum valor sensato; não houve corte nenhum.** O rótulo não estava vago, estava a
afirmar comportamento inexistente.

## Por que este portão não é uma comparação de texto

Comparar o rótulo com uma constante escrita aqui seria a segunda fonte de verdade sobre a mesma
frase, e reprovaria por edição legítima. **O que cada eixo FAZ com o vazio é medido** — corre-se
`process_matrix` com ele em `None` e lê-se se a etapa saiu saltada —, e a asserção de rótulo
percorre os eixos assim verificados.

> **A frase aqui dizia que a POPULAÇÃO era medida, e não é — apanhado na revisão adversarial de
> 07/09/2026.** `EIXOS_QUE_SALTAM` é escrita à mão, e a âncora do teste de rótulo comparava-a com
> um segundo dicionário escrito à mão no mesmo arquivo: um oitavo eixo que passasse a saltar seria
> invisível às duas metades.
>
> **O conserto derivou a população da SPEC**, que é a lista fechada de eixos que a rota aceita: o
> portão percorre `PipelineSpec` inteira, corre `process_matrix` com cada eixo `float | None` em
> `None`, e é a MEDIÇÃO que diz quais saltam. Um eixo novo entra sozinho.

O oráculo foi contado antes: acusava 2 dos 3 campos quando nasceu. O corte de
RSD passava; os outros dois reprovavam, cada um pela sua metade. Um oráculo que acusasse 3 de 3
seria cego, e um que acusasse 0 de 3 não seria portão.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from tests.qt_guard import garante_plataforma_offscreen, membros_ausentes, sonda_o_ambiente

_FALTAM = membros_ausentes(sonda_o_ambiente())
if _FALTAM:  # pragma: no cover — a decisão é exercitada em `tests/test_qt_guard.py`, com planta
    pytest.skip(
        f"grupo opcional `desktop` ausente: falta {', '.join(_FALTAM)}. "
        'Instale com `uv pip install -e ".[desktop]"` para exercitar os testes de janela.',
        allow_module_level=True,
    )

garante_plataforma_offscreen()

from refino.desktop.janela.pipeline import (  # noqa: E402
    LOTES_QUANDO_VAZIO,
    PainelDePipeline,
)
from refino.domain.matrix import RefinoMatrix  # noqa: E402
from refino.domain.metadata import SampleMetadata  # noqa: E402
from refino.domain.spec import PipelineSpec  # noqa: E402
from refino.preprocessing.pipeline import process_matrix  # noqa: E402

# **As palavras que PROMETEM um valor.** Um campo cujo vazio salta a etapa não pode usar nenhuma
# delas: todas dizem ao profissional que alguma coisa foi aplicada. A lista é do vocabulário do
# domínio e não das duas frases que estavam erradas — *«recomendação»* e *«sugestão»* nunca
# apareceram num rótulo de corte, e estão aqui porque são o que a próxima sessão escreveria.
PROMESSAS_DE_VALOR = ("padrão", "servidor", "sugestão", "recomendação", "diagnóstico")

N_QC = 4

# **A matriz é CALIBRADA e não escolhida, e a calibração é o conteúdo.** Cada corte tem de
# descartar alguma coisa **sem** esvaziar a matriz: descartar tudo levanta `EmptyMatrixError` e
# descartar nada faz a planta não separar. As seis colunas foram medidas até as duas
# etapas discardarem — RSD 6 → 3, ausentes 6 → 5.
COLUNAS: dict[str, list[float]] = {
    #    QC1   QC2   QC3   QC4  |  A1    A2    B1    B2
    "a": [10.0, 10.2, 9.8, 10.1, 12.0, 8.0, 11.0, 9.0],
    "b": [20.0, 20.4, 19.6, 20.2, 24.0, 16.0, 22.0, 18.0],
    "c": [5.0, 5.1, 4.9, 5.05, 6.0, 4.0, 5.5, 4.5],
    "d": [1.0, 5.0, 12.0, 30.0, 7.0, 3.0, 9.0, 2.0],  # RSD dos QCs alto: o corte de RSD apanha-o
    "e": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 7.0, 9.0],  # ausente em QC e A, PRESENTE em B: FICA
    "f": [0.0, 0.0, 0.0, 7.0, 0.0, 3.0, 0.0, 5.0],  # maioria ausente nos TRÊS grupos: sai
}
GRUPOS = ("QC", "QC", "QC", "QC", "A", "A", "B", "B")

# `eixo da spec -> (etapa que ele controla, valor que a LIGA)`. O valor é o não-padrão porque,
# com o padrão (`None`), só metade do ramo corre, e a metade não exercitada não
# reprova porque não corre.
EIXOS_QUE_SALTAM: dict[str, tuple[str, float]] = {
    "rsd_cutoff": ("filter_by_qc_rsd", 0.2),
    "missing_threshold": ("drop_by_group_missingness", 0.4),
}


def eixos_que_saltam_de_facto() -> set[str]:
    """Os eixos `float | None` da `PipelineSpec` cujo `None` faz uma etapa sair SALTADA.

    **Derivado da spec e MEDIDO, e não uma lista escrita.** A revisão adversarial de 07/09/2026
    apanhou a versão anterior: `EIXOS_QUE_SALTAM` era escrita à mão e a âncora comparava-a com um
    segundo dicionário à mão — dois literais a concordarem um com o outro, e um oitavo eixo que
    passasse a saltar seria invisível às duas metades.

    **Aqui a população vem de `dataclasses.fields(PipelineSpec)`** — a lista fechada que a rota
    aceita —, e quem decide é a execução: corre-se `process_matrix` com aquele eixo em `None` e
    lê-se se alguma etapa trouxe o rótulo `saltada` com o motivo daquele eixo. Um eixo novo entra
    sozinho.
    """
    import dataclasses

    matriz = _matriz()
    saltam: set[str] = set()
    for campo in dataclasses.fields(PipelineSpec):
        if "None" not in str(campo.type):
            continue
        _, etapas = process_matrix(matriz, _spec(**{campo.name: None}))
        if any(r.labels.get("saltada", "").startswith(f"{campo.name}=") for r in etapas):
            saltam.add(campo.name)
    return saltam


def _matriz() -> RefinoMatrix:
    nomes = tuple(COLUNAS)
    valores = np.array([COLUNAS[nome] for nome in nomes], dtype=np.float64).T
    amostras = tuple(
        SampleMetadata(
            name=f"S{i}",
            injection_order=i + 1,
            batch="A",
            sample_type="qc" if i < N_QC else "sample",
            group=GRUPOS[i],
        )
        for i in range(valores.shape[0])
    )
    return RefinoMatrix(values=valores, feature_names=nomes, samples=amostras, mode="pos_hilic")


def _spec(**troca: Any) -> PipelineSpec:
    campos: dict[str, Any] = {
        "correction": "none",
        "n_batches": 1,
        "rsd_cutoff": None,
        "ranking_filter": "none",
        "ranking_keep_fraction": None,
        "ranking_filter_2": "none",
        "ranking_keep_fraction_2": None,
        "rsd_variant": "outlier_removed",
        "missing_threshold": None,
        "imputation": "one_fifth_min",
        "sample_normalization": "none",
        "normalization_reference": None,
        "normalization_factors": None,
        "transformation": "none",
        "scaling": "none",
    }
    campos.update(troca)
    return PipelineSpec(**campos)


@pytest.mark.parametrize(
    ("eixo", "etapa", "ligado"),
    [(eixo, etapa, ligado) for eixo, (etapa, ligado) in EIXOS_QUE_SALTAM.items()],
)
def test_o_eixo_vazio_salta_a_etapa_e_o_valor_a_faz_correr(
    eixo: str, etapa: str, ligado: float
) -> None:
    """O par causal: `None` salta, um valor real não salta. É esta a medição que o rótulo cita.

    **A âncora é a segunda metade e não a primeira.** Afirmar só que `None` salta passaria numa
    implementação que saltasse sempre — a planta que não separa. O que separa é as duas
    execuções divergirem sobre o mesmo dado, e a divergência é conferida nas contagens: uma etapa
    que corra e descarte **zero** é indistinguível de uma etapa saltada para quem lê o número.
    """
    matriz = _matriz()

    _, vazio = process_matrix(matriz, _spec(**{eixo: None}))
    _, cheio = process_matrix(matriz, _spec(**{eixo: ligado}))

    por_etapa_vazio = {r.stage: r for r in vazio}
    por_etapa_cheio = {r.stage: r for r in cheio}
    assert etapa in por_etapa_vazio, f"`{etapa}` não está no relatório: {sorted(por_etapa_vazio)}"
    assert etapa in por_etapa_cheio, f"`{etapa}` não está no relatório: {sorted(por_etapa_cheio)}"

    assert "saltada" in por_etapa_vazio[etapa].labels, (
        f"`{eixo}=None` devia SALTAR `{etapa}`, e a etapa correu: {por_etapa_vazio[etapa].labels}"
    )
    assert "saltada" not in por_etapa_cheio[etapa].labels, (
        f"`{eixo}={ligado}` devia fazer `{etapa}` CORRER, e ela saltou"
    )
    resultado = por_etapa_cheio[etapa]
    assert resultado.n_features_out < resultado.n_features_in, (
        f"`{eixo}={ligado}` correu e descartou ZERO features "
        f"({resultado.n_features_in} → {resultado.n_features_out}) — a planta não separa, e sem "
        f"separação a primeira metade deste teste não prova nada"
    )


def test_o_rotulo_de_quem_salta_anuncia_ausencia_e_nao_promete_valor(qtbot: Any) -> None:
    """O rótulo dos campos que saltam não pode prometer que alguma coisa foi aplicada.

    **A ligação campo↔eixo sai da tela e não de uma tabela nova aqui:** uma tabela local seria a
    segunda fonte de verdade sobre qual widget alimenta qual chave, e divergiria no dia em que
    alguém renomeasse um.
    """
    painel = PainelDePipeline()
    qtbot.addWidget(painel)
    campos = {"rsd_cutoff": painel.corte, "missing_threshold": painel.ausentes}

    # **A POPULAÇÃO É DERIVADA DA SPEC E MEDIDA, e não comparada com uma segunda lista à mão.**
    #
    # A âncora anterior era `set(campos) == set(EIXOS_QUE_SALTAM)` — dois dicionários escritos à
    # mão neste arquivo a concordarem um com o outro. A revisão adversarial apanhou-a: **um oitavo
    # eixo que passasse a saltar era invisível às duas metades.**
    medidos = eixos_que_saltam_de_facto()
    assert medidos, "âncora: nenhum eixo salta a etapa — a medição não separou nada"
    assert set(campos) == medidos, (
        f"a tela dá controle a {sorted(campos)} e os eixos que SALTAM, medidos sobre "
        f"`process_matrix`, são {sorted(medidos)}. Um eixo que salta sem controle na tela não tem "
        f"rótulo a conferir; um com controle e que não salta tem de dizer o que faz"
    )

    for eixo, widget in campos.items():
        texto = widget.specialValueText()
        assert texto, f"o campo de `{eixo}` não tem marcador de vazio"
        prometidas = [p for p in PROMESSAS_DE_VALOR if p in texto.casefold()]
        assert not prometidas, (
            f"o rótulo de `{eixo}` diz {texto!r}, e `{eixo}=None` SALTA a etapa — a palavra "
            f"{prometidas!r} promete um valor que ninguém aplica. Quem lê conclui que houve corte "
            f"com algum número sensato; não houve corte nenhum"
        )
        assert "não" in texto.casefold(), (
            f"o rótulo de `{eixo}` diz {texto!r} e devia anunciar a AUSÊNCIA do corte — "
            f"«salta a etapa» é linguagem de quem implementou, não de quem opera"
        )


def test_o_rotulo_dos_lotes_diz_o_numero_que_o_vazio_envia(qtbot: Any) -> None:
    """O campo dos lotes não salta nada: o vazio envia **um lote**. O rótulo tem de o dizer.

    **O oráculo é exato e não é de forma:** o número citado no rótulo é o mesmo que a janela
    envia, e os dois saem da mesma constante. Um rótulo escrito à mão poderia dizer *«2»* e nada
    reprovaria.
    """
    painel = PainelDePipeline()
    qtbot.addWidget(painel)
    texto = painel.n_lotes.specialValueText()

    assert LOTES_QUANDO_VAZIO >= 1, (
        f"âncora: `PipelineSpec` recusa `n_batches < 1`, e a constante vale {LOTES_QUANDO_VAZIO}"
    )
    assert str(LOTES_QUANDO_VAZIO) in texto, (
        f"o rótulo dos lotes diz {texto!r} e o vazio envia {LOTES_QUANDO_VAZIO} — o rótulo tem de "
        f"nomear o número que viaja"
    )
    prometidas = [p for p in PROMESSAS_DE_VALOR if p in texto.casefold()]
    assert not prometidas, (
        f"o rótulo dos lotes diz {texto!r}. A sugestão do diagnóstico chega por `pre_preenche`, "
        f"que ESCREVE o número no campo — e escrever o número APAGA este marcador. Ele só está "
        f"visível quando a sugestão NÃO vai ser usada: a palavra {prometidas!r} promete o oposto"
    )


def test_o_vazio_do_campo_de_lotes_vira_o_numero_que_a_constante_declara(qtbot: Any) -> None:
    """A ponte: o campo no mínimo devolve `None`, e quem dispara traduz para a constante.

    Sem isto, `LOTES_QUANDO_VAZIO` podia ficar a **1** no rótulo enquanto a janela mandava outra
    coisa — que é exatamente a distância que o defeito original tinha.
    """
    painel = PainelDePipeline()
    qtbot.addWidget(painel)

    painel.n_lotes.setValue(painel.n_lotes.minimum())
    assert painel.escolhas_do_disparo()["n_lotes"] is None, (
        "o mínimo do campo é o vazio, e ele tem de viajar como `None`"
    )

    painel.n_lotes.setValue(3)
    assert painel.escolhas_do_disparo()["n_lotes"] == 3, (
        "âncora: com um valor escolhido é ele que viaja — sem isto a asserção acima passaria numa "
        "implementação que devolvesse `None` sempre"
    )
