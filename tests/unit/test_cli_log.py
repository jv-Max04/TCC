"""O portão do registro reexecutável, e o predicado é POR CAMPO DO CONTRATO.

## O defeito que ele fecha

O critério diz *"realimentar o log produz saída idêntica"*, e o teste que o mede compara os
dois arquivos **bit a bit**. Essa comparação é a propriedade certa e tem um ponto cego
exato: um campo novo do contrato **com valor padrão** seria omitido da escrita do registro,
a segunda execução cairia no mesmo padrão, e os dois arquivos continuariam idênticos. O
portão fica verde afirmando uma reexecutabilidade que não foi exercida — *o valor
natural é o que executa menos do caminho*.

Este bloco acrescenta dois campos de contrato sem valor padrão, e é por isso
que a rede vem antes deles.

## Por que POR CAMPO NOVO e não a bijeção dos quatro conjuntos

A forma óbvia seria exigir bijeção entre os campos de `PipelineSpec`, as chaves da escrita,
os campos obrigatórios da leitura e as bandeiras da linha de comando. **Essa bijeção é falsa
hoje, por razões legítimas**, e exigi-la faria o portão nascer com uma tabela de exceções
escolhida para a árvore de hoje passar — o oráculo que codifica a forma da instância contra
a qual nasceu.

O predicado é o comportamental, e prova as duas superfícies de uma vez:

> **trocar o valor de um campo do contrato no registro tem de trocar o campo do contrato
> reconstruído a partir dele.**

Um campo que a escrita omita não chega; um campo que a leitura ignore não chega. Nenhuma das
duas formas sobrevive, e nenhuma delas é lida do **texto** do módulo — que seria o oráculo
pela forma sintática.

## A tabela de isenção, e por que cada linha carrega uma citação

Uma tabela de isenção sem prova é uma lista de reprovações permitidas com outro nome. Cada
entrada aqui cita **texto que já existe no código** e que explica a assimetria, e um teste
próprio confere que a citação está lá. Isso é o que impede a próxima sessão de fechar uma
reprovação acrescentando uma linha.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from dataclasses import fields, replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from refino.cli import (
    ProcessOptions,
    UsageError,
    _process_options_from_log,
    build_parser,
    build_process_log,
)
from refino.domain.labels import VOCABULARY
from refino.domain.matrix import RefinoMatrix
from refino.domain.metadata import SampleMetadata
from refino.domain.spec import PipelineSpec

# ---------------------------------------------------------------------------
# O mapeamento entre o nome do campo e a chave do registro, DECLARADO UMA VEZ
# ---------------------------------------------------------------------------
#
# `build_process_log` escreve com os nomes das opções de linha de comando — hífen —, e o
# dataclass usa sublinhado. A tradução vive aqui e em nenhum outro lugar deste módulo:
# repetida em cada teste, ela seria a segunda fonte de verdade que diverge no dia em que um
# campo tiver nome composto de três partes.


def chave_do_registro(campo: str) -> str:
    """O nome do campo do contrato como o registro o escreve."""
    return campo.replace("_", "-")


# ---------------------------------------------------------------------------
# A tabela de isenção — cada linha com a citação que a explica
# ---------------------------------------------------------------------------

#: Campos do contrato que o subcomando `processar` FIXA, e por isso não viajam no registro.
#: O valor de cada entrada diz por quê.
ISENCOES_FIXADAS_PELO_SUBCOMANDO: dict[str, str] = {
    "correction": "pipeline completo pós-correção",
    "n_batches": "pipeline completo pós-correção",
    # Os cortes só existem com mais de um lote, e o `processar` fixa um lote só — pelo mesmo
    # motivo das duas linhas de cima, o campo não tem valor a viajar neste subcomando.
    "batch_cuts": "pipeline completo pós-correção",
}

#: Campos lidos com valor de recurso em vez de recusarem quando ausentes. A política deste
#: módulo é recusar; o recurso existe para o registro escrito ANTES de a etapa existir, que
#: está completo para a versão dele. Cada entrada declara o valor que o recurso produz.
RECURSOS_DECLARADOS: dict[str, object] = {
    "ranking_filter": "none",
    # **Os três do filtro em dois eixos seguem o mesmo regime, e pelo mesmo motivo.** Um registro
    # escrito antes de o segundo passe existir está completo para a versão dele: `none` com
    # fração ausente reproduz exactamente o que aquela execução fez, porque a etapa não correu e
    # a regra por tamanho é a que sempre valeu. Exigi-los reprovaria todo registro anterior a
    # 10/09/2026.
    "ranking_keep_fraction": None,
    "ranking_filter_2": "none",
    "ranking_keep_fraction_2": None,
    # **A referencia segue o mesmo regime.** Um registro escrito antes de ela existir esta
    # completo para a versao dele, e `None` reproduz o que aquela execucao fez: nenhum metodo de
    # referencia correu. A classe CONDICIONAL — obrigatoria quando o metodo a exige — chega com
    # as unidades que a consomem.
    "normalization_reference": None,
    # **Os fatores seguem o mesmo regime.** Um registro escrito antes de eles existirem
    # esta completo para a versao dele, e `None` reproduz o que aquela execucao fez.
    "normalization_factors": None,
}


# ---------------------------------------------------------------------------
# O log de referência, construído pela escrita REAL
# ---------------------------------------------------------------------------


def _matriz() -> RefinoMatrix:
    valores = np.array(
        [
            [10.0, 5.0, 7.0],
            [11.0, 5.2, 7.1],
            [9.0, 4.8, 6.9],
            [20.0, 9.0, 3.0],
            [21.0, 9.5, 3.4],
        ]
    )
    amostras = tuple(
        SampleMetadata(
            name=f"S{i}",
            injection_order=i + 1,
            sample_type="qc" if i < 3 else "sample",
            group="QC" if i < 3 else "G",
        )
        for i in range(valores.shape[0])
    )
    return RefinoMatrix(
        values=valores,
        feature_names=("f0", "f1", "f2"),
        samples=amostras,
        mode="sintetico",
    )


def _spec_base() -> PipelineSpec:
    """Uma especificação com **todos** os eixos ativos.

    Ativos de propósito: um eixo em `none` é o valor que executa menos do caminho, e uma
    especificação toda em `none` faria a troca de valor deste portão ser entre dois padrões.
    """
    return PipelineSpec(
        correction="none",
        n_batches=1,
        rsd_cutoff=0.30,
        ranking_filter="iqr",
        ranking_keep_fraction=0.60,
        ranking_filter_2="mean",
        ranking_keep_fraction_2=0.90,
        rsd_variant="outlier_removed",
        missing_threshold=0.50,
        imputation="one_fifth_min",
        sample_normalization="sum",
        normalization_reference=None,
        normalization_factors=None,
        transformation="log10",
        scaling="auto",
        seed=42,
    )


def _opcoes(spec: PipelineSpec, destino: Path) -> ProcessOptions:
    return ProcessOptions(
        measurements=Path("entrada/measurements.csv"),
        sample_list=Path("entrada/Sample_List.csv"),
        mode="sintetico",
        orientation="features-in-rows",
        spec=spec,
        output=destino / "saida.csv",
        log=destino / "log.json",
    )


def _escreve_registro(
    spec: PipelineSpec,
    destino: Path,
    *,
    construtor: Callable[..., dict[str, Any]] = build_process_log,
) -> Path:
    """Escreve o registro pela função REAL e devolve o caminho.

    `construtor` existe para os pares causais: eles passam uma escrita mutilada e observam o
    portão acusar. Nenhum caminho de produção o substitui.
    """
    registro = construtor(_opcoes(spec, destino), _matriz(), [])
    caminho = destino / "execucao.json"
    caminho.write_text(json.dumps(registro, indent=2), encoding="utf-8")
    return caminho


def _reconstroi(caminho: Path, destino: Path) -> ProcessOptions:
    namespace = argparse.Namespace(
        from_log=caminho,
        output=destino / "reexecucao.csv",
        log=None,
    )
    return _process_options_from_log(namespace)


# ---------------------------------------------------------------------------
# O valor alternativo — derivado, nunca escolhido à mão
# ---------------------------------------------------------------------------


#: Sentinela para «este campo não tem recurso». `None` é valor de recurso legítimo — quatro
#: entradas de `RECURSOS_DECLARADOS` usam-no —, então `.get(campo)` não distingue ausência de
#: recurso-`None`, e é exactamente essa confusão que deixaria o filtro abaixo inerte.
_SEM_RECURSO = object()


def _alternativas(campo: str, spec: PipelineSpec) -> object:
    """Um valor VÁLIDO e diferente do atual para `campo`.

    Derivado do vocabulário ou do tipo, e nunca pinado: um valor à mão por campo teria de
    ser revisitado a cada campo novo, e é exatamente essa revisita que ninguém faz.

    Cada candidato é **construído** antes de ser devolvido, porque a combinação proibida
    existe: `imputation='none'` com normalização ativa não é uma especificação, e um
    alternativo que a produzisse faria este portão reprovar por um motivo que não é o dele.

    ## O CANDIDATO NÃO PODE SER O RECURSO DO LEITOR, e isto foi MEDIDO

    A versão anterior devolvia o primeiro valor do vocabulário diferente do atual. Para
    `RankingFilter` esse valor é `"none"` — que é **exactamente** o que
    `spec_do_log.get("ranking-filter", "none")` produz quando a chave falta. Escrita mutilada
    e leitura intacta concordavam **por coincidência**, e apagar `ranking-filter` de
    `build_process_log` deixava `_campos_que_nao_viajam` a devolver `[]`.

    O portão ficava verde sobre os dois campos para que tinha acabado de ser estendido.

    **O oráculo codifica a forma da instância contra a qual nasceu, e a pergunta
    é:** *quais instâncias deste defeito eu já
    consertei, e o meu oráculo acha as que eu não consertei?* Ele não achava, porque a classe
    «campo com recurso declarado» só passou a existir com estes campos, e o oráculo é anterior
    a ela.
    """
    atual = getattr(spec, campo)
    recurso = RECURSOS_DECLARADOS.get(campo, _SEM_RECURSO)
    rotulo = _ROTULO_DO_CAMPO.get(campo)
    if rotulo is not None:
        candidatos: list[object] = [v for v in VOCABULARY[rotulo] if v != atual and v != recurso]
    elif campo == "normalization_factors":
        # Um mapa diferente do atual. O portao mede o TRANSPORTE — que o mapa chegue por valor —
        # e nao se os nomes existem na matriz, que e pergunta de quem tem a matriz.
        candidatos = [(("S0", 2.0),) if atual != (("S0", 2.0),) else (("S0", 3.0),)]
    elif campo == "normalization_reference":
        # Um nome qualquer, diferente do atual. O valor NAO precisa de existir na matriz: a
        # existencia e conferida onde a matriz e conhecida, e este portao mede o transporte.
        candidatos = ["S0" if atual != "S0" else "S1"]
    elif campo == "seed":
        candidatos = [int(atual) + 1]
    elif campo in (
        "rsd_cutoff",
        "missing_threshold",
        "ranking_keep_fraction",
        "ranking_keep_fraction_2",
    ):
        candidatos = [0.25 if atual != 0.25 else 0.35]
    elif campo == "n_batches":
        candidatos = [int(atual) + 1]
    else:  # pragma: no cover - campo novo sem regra de alternativa
        raise AssertionError(
            f"{campo!r} não tem regra de valor alternativo. Acrescente-a em `_alternativas` "
            "— um campo do contrato que este portão não consegue variar é um campo que ele "
            "não verifica."
        )

    for candidato in candidatos:
        try:
            replace(spec, **{campo: candidato})
        except (ValueError, TypeError):
            continue
        # **A guarda vale para TODOS os ramos, e não só para o do vocabulário.** Os ramos por
        # tipo escolhem valores que hoje diferem do recurso — `"S0"` contra `None`, `0.25`
        # contra `None` —, e essa é a razão de o defeito ter aparecido só no vocabulário. Um
        # ramo novo cujo candidato calhe no recurso reprova aqui, em vez de ficar verde.
        assert candidato != recurso, (
            f"{campo!r}: o valor alternativo {candidato!r} é o RECURSO que o leitor usa "
            "quando a chave falta. Com ele, uma escrita que omita o campo e uma leitura que "
            "o recupere pelo recurso concordam por coincidência, e o portão fica cego "
            "exactamente sobre este campo"
        )
        return candidato
    raise AssertionError(  # pragma: no cover - vocabulário de um valor só
        f"{campo!r}: nenhum valor alternativo produz especificação válida"
    )


#: De qual `Literal` cada campo do contrato tira o seu domínio. Enumerado e não inferido do
#: nome: `rsd_variant` e `ranking_filter` não se chamam como os rótulos deles, e adivinhar
#: pelo nome seria o oráculo pela forma.
_ROTULO_DO_CAMPO: dict[str, str] = {
    "correction": "Correction",
    "ranking_filter": "RankingFilter",
    "ranking_filter_2": "RankingFilter",
    "rsd_variant": "RsdVariant",
    "imputation": "Imputation",
    "sample_normalization": "SampleNormalization",
    "transformation": "Transformation",
    "scaling": "Scaling",
}


def campos_do_contrato() -> tuple[str, ...]:
    """Os campos de `PipelineSpec` que este portão exige que viajem no registro."""
    return tuple(
        campo.name
        for campo in fields(PipelineSpec)
        if campo.name not in ISENCOES_FIXADAS_PELO_SUBCOMANDO
    )


# ---------------------------------------------------------------------------
# A âncora de não-vacuidade, e ela vem antes
# ---------------------------------------------------------------------------


def test_ancora_ha_campos_a_verificar_e_a_isencao_nao_engoliu_o_contrato() -> None:
    """Asserção de ausência é satisfeita por vacuidade; esta guarda vai antes.

    Se `campos_do_contrato` devolvesse vazio — porque `fields` mudou de forma, ou porque a
    tabela de isenção cresceu até cobrir tudo —, todos os testes abaixo passariam sem tocar
    nada. O número é um piso e não uma pinagem: ele sobe quando o contrato cresce, e a
    reprovação a que ele leva é *"a isenção comeu o contrato"*, nunca *"acrescentaram um
    campo"*.
    """
    verificados = campos_do_contrato()
    assert len(verificados) >= 8, (
        f"o portão verifica apenas {len(verificados)} campos do contrato: {verificados}. "
        f"`PipelineSpec` tem {len(fields(PipelineSpec))} e a isenção declara "
        f"{sorted(ISENCOES_FIXADAS_PELO_SUBCOMANDO)}"
    )
    assert set(ISENCOES_FIXADAS_PELO_SUBCOMANDO) < {f.name for f in fields(PipelineSpec)}, (
        "a tabela de isenção nomeia campo que `PipelineSpec` não tem — isenção sobre nada"
    )


# ---------------------------------------------------------------------------
# O predicado
# ---------------------------------------------------------------------------


def _campos_que_nao_viajam(
    tmp_path: Path,
    *,
    construtor: Callable[..., dict[str, Any]] = build_process_log,
) -> list[str]:
    """Os campos cujo valor NÃO sobrevive à ida e volta pelo registro.

    Devolve em vez de afirmar, porque os pares causais precisam de chamar a mesma verificação
    com uma escrita mutilada e comparar os dois desfechos.
    """
    base = _spec_base()
    falharam = []
    for campo in campos_do_contrato():
        alternativo = _alternativas(campo, base)
        caminho = _escreve_registro(
            replace(base, **{campo: alternativo}), tmp_path, construtor=construtor
        )
        try:
            reconstruido = _reconstroi(caminho, tmp_path).spec
        except UsageError:
            falharam.append(campo)
            continue
        if getattr(reconstruido, campo) != alternativo:
            falharam.append(campo)
    return falharam


def test_todo_campo_do_contrato_viaja_no_registro(tmp_path: Path) -> None:
    """O predicado por campo: trocar o valor no registro troca o valor reconstruído.

    Um campo omitido da escrita não chega; um campo que a leitura ignore não chega. As duas
    formas reprovam aqui, e a mensagem nomeia o campo.
    """
    falharam = _campos_que_nao_viajam(tmp_path)
    assert not falharam, (
        f"campos do contrato que não sobrevivem à ida e volta pelo registro: {falharam}. "
        "Ou a escrita em `build_process_log` não os emite, ou "
        "`_process_options_from_log` não os lê. Um campo com valor padrão que caia fora "
        "das duas deixa `[CA-5.9]` verde sem exercitar nada"
    )


def test_par_causal_campo_removido_da_escrita_e_ACUSADO(tmp_path: Path) -> None:  # noqa: N802
    """O par causal do predicado, e é o que prova que ele não está inerte.

    A planta cai sobre um campo que **existe e viaja**, e a asserção é que o desfecho
    diverge do da árvore limpa — não apenas que a árvore mutilada reprova.
    """
    sem_planta = tmp_path / "limpo"
    com_planta = tmp_path / "mutilado"
    sem_planta.mkdir()
    com_planta.mkdir()

    limpo = _campos_que_nao_viajam(sem_planta)
    assert not limpo, f"a árvore limpa já acusa {limpo}; a planta não separaria nada"

    def escrita_mutilada(*args: Any, **kwargs: Any) -> dict[str, Any]:
        registro = build_process_log(*args, **kwargs)
        registro["spec"].pop("scaling")
        return registro

    mutilado = _campos_que_nao_viajam(com_planta, construtor=escrita_mutilada)
    assert "scaling" in mutilado, (
        "com `scaling` fora da escrita do registro o portão continuou verde — ele não está "
        "a verificar a superfície da escrita"
    )


@pytest.mark.parametrize(
    ("campo", "chave"),
    [
        ("ranking_filter", "ranking-filter"),
        ("ranking_filter_2", "ranking-filter-2"),
        ("normalization_reference", "normalization-reference"),
        ("normalization_factors", "normalization-factors"),
    ],
)
def test_par_causal_campo_COM_RECURSO_removido_da_escrita_e_ACUSADO(  # noqa: N802
    tmp_path: Path, campo: str, chave: str
) -> None:
    """O par causal da classe que o irmão acima NÃO alcança, e ela é a que ficou cega.

    ## Por que dois pares e não um

    O par do `scaling` planta numa chave **obrigatória**, e a leitura recusa o registro
    inteiro: `_campos_que_nao_viajam` devolve os catorze campos de uma vez. A asserção
    `"scaling" in mutilado` passa — e passa por causa da conferência de chaves obrigatórias,
    **não** do predicado por campo. O veredito acusa o alvo certo pelo motivo errado.

    Um campo **com recurso declarado** não derruba a leitura. A chave falta, o leitor
    devolve o recurso, e a comparação por campo é o único mecanismo que pode acusar. Por
    isso a asserção aqui é de igualdade exacta e não de pertença: se o portão devolvesse
    tudo, o predicado por campo continuaria por exercitar.

    ## O que isto teria apanhado

    Medido em 10/09/2026: com `_alternativas` a escolher o primeiro valor do vocabulário,
    o candidato de `RankingFilter` era `"none"` — o próprio recurso —, e apagar
    `ranking-filter` da escrita deixava o portão a devolver `[]`. Este teste reprovava.
    """
    sem_planta = tmp_path / "limpo"
    com_planta = tmp_path / "mutilado"
    sem_planta.mkdir()
    com_planta.mkdir()

    limpo = _campos_que_nao_viajam(sem_planta)
    assert not limpo, f"a árvore limpa já acusa {limpo}; a planta não separaria nada"

    def escrita_mutilada(*args: Any, **kwargs: Any) -> dict[str, Any]:
        registro = build_process_log(*args, **kwargs)
        # **A planta é CONFERIDA e não afirmada**. Se a
        # chave mudar de nome, `pop` silencioso deixaria a planta a não aterrar e o desfecho
        # seria indistinguível de sucesso.
        assert chave in registro["spec"], (
            f"a planta não aterrou: {chave!r} não está no registro escrito. As chaves são "
            f"{sorted(registro['spec'])}"
        )
        registro["spec"].pop(chave)
        return registro

    mutilado = _campos_que_nao_viajam(com_planta, construtor=escrita_mutilada)
    assert mutilado == [campo], (
        f"com {chave!r} fora da escrita, o portão devolveu {mutilado} e devia devolver "
        f"exactamente ['{campo}']. Vazio significa que o valor alternativo coincide com o "
        f"recurso do leitor e os dois lados concordam por coincidência; a lista inteira "
        f"significa que quem acusou foi a conferência de chaves obrigatórias, e o predicado "
        f"por campo continua por exercitar"
    )


# ---------------------------------------------------------------------------
# A classe de obrigação, nas duas direções
# ---------------------------------------------------------------------------


def test_a_classe_de_obrigacao_de_cada_campo_esta_declarada(tmp_path: Path) -> None:
    """Campo ausente ou RECUSA, ou tem recurso declarado. As duas direções.

    A política do módulo é recusar: preencher em silêncio faria a reexecução divergir da
    original com o registro parecendo íntegro. O recurso é a exceção, e existe para o
    registro escrito **antes** de a etapa existir — completo para a versão dele.

    Um recurso não declarado reprova, e um campo declarado como recurso que passe a recusar
    reprova também. É a segunda direção que impede a tabela de crescer sozinha.
    """
    base = _spec_base()
    caminho = _escreve_registro(base, tmp_path)
    completo = json.loads(caminho.read_text(encoding="utf-8"))

    recursos_observados: dict[str, object] = {}
    for campo in campos_do_contrato():
        mutilado = json.loads(json.dumps(completo))
        mutilado["spec"].pop(chave_do_registro(campo), None)
        alvo = tmp_path / f"sem_{campo}.json"
        alvo.write_text(json.dumps(mutilado), encoding="utf-8")
        try:
            reconstruido = _reconstroi(alvo, tmp_path).spec
        except UsageError:
            continue
        recursos_observados[campo] = getattr(reconstruido, campo)

    assert recursos_observados == RECURSOS_DECLARADOS, (
        f"a leitura aceita ausência em {sorted(recursos_observados)} com os valores "
        f"{recursos_observados}, e a tabela declara {RECURSOS_DECLARADOS}. Recurso não "
        "declarado preenche em silêncio e faz a reexecução divergir da original; recurso "
        "declarado que passou a recusar quebra todo registro anterior à etapa"
    )


def test_registro_antigo_sem_a_chave_de_uma_etapa_nova_continua_aceito(tmp_path: Path) -> None:
    """A direção que protege o registro escrito antes de a etapa existir.

    Sem ela, o portão da classe de obrigação poderia ser satisfeito tornando **tudo**
    obrigatório — e todo registro anterior a 07/09/2026 deixaria de ser reexecutável.
    """
    caminho = _escreve_registro(_spec_base(), tmp_path)
    antigo = json.loads(caminho.read_text(encoding="utf-8"))
    antigo["spec"].pop("ranking-filter")
    alvo = tmp_path / "antigo.json"
    alvo.write_text(json.dumps(antigo), encoding="utf-8")

    reconstruido = _reconstroi(alvo, tmp_path).spec
    assert reconstruido.ranking_filter == "none", (
        "um registro escrito antes de o filtro por posto existir deixou de ser aceito"
    )


# ---------------------------------------------------------------------------
# A bandeira na linha de comando
# ---------------------------------------------------------------------------


def _bandeiras_do_processar() -> frozenset[str]:
    parser = build_parser()
    (sub,) = [acao for acao in parser._actions if isinstance(acao, argparse._SubParsersAction)]
    processar = sub.choices["processar"]
    return frozenset(opcao for acao in processar._actions for opcao in acao.option_strings)


def test_todo_campo_do_contrato_tem_bandeira_na_linha_de_comando() -> None:
    """A terceira superfície. Sem ela o campo existe e ninguém o pode escolher."""
    bandeiras = _bandeiras_do_processar()
    faltando = [
        campo for campo in campos_do_contrato() if f"--{chave_do_registro(campo)}" not in bandeiras
    ]
    assert not faltando, (
        f"campos do contrato sem bandeira em `processar`: {faltando}. As bandeiras são "
        f"{sorted(bandeiras)}"
    )


# ---------------------------------------------------------------------------
# O que NÃO é campo do contrato, e por isso não entra no predicado
# ---------------------------------------------------------------------------


def test_as_chaves_de_processo_nao_sao_do_contrato_e_isso_e_derivado() -> None:
    """`measurements`, `sample-list`, `mode` e `orientation` viajam no registro e não são spec.

    Declarado por **derivação** e não por lista à mão: são campos de `ProcessOptions` que não
    são campos de `PipelineSpec`. Uma lista pinada aqui teria de ser mantida à mão, e é assim
    que uma isenção sobrevive à razão que a criou.
    """
    do_processo = {f.name for f in fields(ProcessOptions)} - {"spec", "output", "log"}
    do_contrato = {f.name for f in fields(PipelineSpec)}
    assert not (do_processo & do_contrato), (
        f"campos que são das opções de processo E do contrato: {sorted(do_processo & do_contrato)}"
    )
    assert do_processo == {"measurements", "sample_list", "mode", "orientation"}, (
        f"as opções de processo mudaram de forma: {sorted(do_processo)}. O predicado deste "
        "portão só cobre o contrato, e esta asserção é o que torna a fronteira visível"
    )


@pytest.mark.parametrize("campo", sorted(RECURSOS_DECLARADOS))
def test_o_campo_com_recurso_ainda_assim_viaja_quando_presente(campo: str, tmp_path: Path) -> None:
    """Ter recurso não dispensa viajar. As duas coisas são independentes.

    Um campo poderia ser lido com recurso **e** ser ignorado quando presente — o recurso
    esconderia a segunda falha, porque o desfecho seria sempre o valor de recurso e nunca um
    erro. Este teste separa as duas.
    """
    base = _spec_base()
    alternativo = _alternativas(campo, base)
    caminho = _escreve_registro(replace(base, **{campo: alternativo}), tmp_path)
    assert getattr(_reconstroi(caminho, tmp_path).spec, campo) == alternativo, (
        f"{campo} tem valor de recurso e o valor PRESENTE no registro foi ignorado"
    )


# ---------------------------------------------------------------------------
# A forma do requisito CONDICIONAL, declarada aqui para ser preenchida depois
# ---------------------------------------------------------------------------
#
# As duas classes de obrigação acima — obrigatório e com recurso — são as que o contrato
# tem hoje, e nenhuma serve para um parâmetro de referência: ele é **obrigatório quando o
# método o exige e ausente quando não**. Torná-lo obrigatório sempre recusaria todo registro
# de uma execução sem referência; lê-lo com recurso preencheria em silêncio o nome de uma
# amostra que ninguém escolheu, e a reexecução divergiria da original com o registro
# parecendo íntegro.
#
# `CONDICIONAIS` fica vazio de propósito, e o teste abaixo mede a ausência em vez de a
# supor. Quando alguém acrescentar a primeira entrada, o teste passa a exercitar as duas
# direções sem que ninguém tenha de se lembrar de escrever o portão.

#: Campo do contrato -> o predicado sobre a especificação que torna a chave obrigatória.
#: Preenchido pela referência por nome e pelos fatores.
CONDICIONAIS: dict[str, Callable[[PipelineSpec], bool]] = {}


def test_a_classe_condicional_esta_declarada_e_e_verificada(tmp_path: Path) -> None:
    """Obrigatório QUANDO o método exige, ausente quando não. As duas direções.

    Enquanto `CONDICIONAIS` estiver vazio o teste afirma isso explicitamente, em vez de
    percorrer zero itens e reportar sucesso — asserção de ausência é satisfeita por
    vacuidade, e um laço vazio é a forma mais convincente dela.
    """
    conhecidos = set(ISENCOES_FIXADAS_PELO_SUBCOMANDO) | set(RECURSOS_DECLARADOS)
    assert not (set(CONDICIONAIS) & conhecidos), (
        f"campo classificado em duas classes de obrigação: "
        f"{sorted(set(CONDICIONAIS) & conhecidos)}. Um condicional lido com recurso preenche "
        "em silêncio o nome de uma referência que ninguém escolheu"
    )
    if not CONDICIONAIS:
        # **Sem `skip`, e a escolha é deliberada.** Um `skip` aqui desapareceria sozinho com a
        # primeira entrada — que é o marcador que absorve, e a única forma de verificação
        # inerte em que exercitar NÃO revela. A asserção afirma o estado de hoje e reprova no
        # dia em que ele mudar sem que o laço abaixo passe a correr.
        assert not CONDICIONAIS, "inalcançável: o `if` acabou de o afirmar"
        return

    base = _spec_base()  # pragma: no cover - alcançado com a primeira entrada
    for campo, exige in CONDICIONAIS.items():  # pragma: no cover - idem
        assert not exige(base), (
            f"{campo}: a especificação de base já exige a chave, então o caso «ausente e "
            "aceito» não é exercitado por nenhum registro deste portão"
        )
        caminho = _escreve_registro(base, tmp_path)
        semi = json.loads(caminho.read_text(encoding="utf-8"))
        assert chave_do_registro(campo) not in semi["spec"], (
            f"{campo} foi escrito no registro de uma especificação que não o exige"
        )
        _reconstroi(caminho, tmp_path)
