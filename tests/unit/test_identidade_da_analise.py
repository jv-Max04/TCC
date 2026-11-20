"""O portão da chave de invalidação de `/execucoes`.

## O que este portão existe para impedir

Um parâmetro que **muda o resultado** e não entra na chave deixa a figura antiga na tela com o
rótulo novo. O profissional troca o tipo de post-hoc, a figura não se mexe, e nada levanta: a
chave é igual, então o painel conclui que a execução é a mesma. É o modo de falha insidioso — o
que **não** levanta.

## O recorte que este portão cobre, e o que ele NÃO cobre

**Cinco cláusulas conferem NOMES; duas conferem VALORES.** A distinção é o recorte honesto desta
afirmação, e ela foi **medida**: com o contrato a entrar na chave com os nomes e valores
constantes, as cinco de nome passam e só `test_trocar_o_post_hoc_muda_a_chave` reprova. A medição é
de 13/09/2026, quando `AnalysisSpec` tinha **vinte e quatro** campos; hoje tem **vinte e seis**, e
o argumento não depende do número — o que ele depende é de nenhuma das cinco olhar para valor.

O que continua fora: que `_chave_do_trabalho` **hase** bem — a colisão de duas entradas diferentes
é propriedade do `sha256` e não deste portão —, e que o cliente **leia** a chave que a resposta
carrega, que é o contrato da janela e vive noutro arquivo.

## Por que o oráculo de nomes é o dicionário INTEIRO, e não só o contrato de análise

Seis campos já entram na chave **à mão** hoje — `estudo`, `apresentacao`, `campos_sobrescritos`,
`proposta_json`, `group_a` e `group_b` — e nenhum tem portão. Um oráculo que conferisse apenas os
campos novos herdaria o buraco inteiro e pareceria rigor.

**E há uma segunda pergunta, que a contagem sozinha não responde:** um portão escrito depois
de o autor acrescentar os parâmetros codifica **os que ele lembrou**. Por isso este arquivo nasceu
**antes** de `AnalysisSpec` existir e foi corrido contra a árvore de então — a medição está na
tabela abaixo.

## A forma: tabela com coluna de MOTIVO, nunca conjunto

É a forma de `FORA_DA_CHAVE_POR_DESENHO` abaixo, copiada de `FORA_DA_IDENTIDADE` em
`tests/unit/test_identidade_do_corretor.py`. O que a torna honesta é a coluna da direita: excluir
um parâmetro é **afirmar** que ele não muda valor nenhum, e essa afirmação precisa de dono. Sem
ela a tabela é lista de isenção com outro nome.
"""

from __future__ import annotations

import inspect
from dataclasses import fields, replace
from typing import Any

import numpy as np
import pytest

from refino.api import rotas
from refino.domain.analysis import AnalysisSpec
from refino.domain.matrix import RefinoMatrix
from refino.domain.metadata import SampleMetadata
from refino.domain.spec import PipelineSpec

pytestmark = pytest.mark.property


# **Os dataclasses de onde a chave DERIVA os campos.** Campo novo num destes entra na chave
# sozinho, sem ninguém se lembrar deste arquivo — que é a propriedade que `_spec_para_json`
# ganhou depois de `ranking_filter` ter derivado por ser lista manual.
DATACLASSES_DA_CHAVE: tuple[type, ...] = (PipelineSpec, AnalysisSpec)


# **Os campos da chave que NÃO vêm de dataclass, cada um com o motivo de estar à mão.**
#
# Medido em 13/09/2026, antes de escrever a tabela: com a tabela **vazia**
# o portão acusa **seis** campos — os seis abaixo. Seis de seis é o número certo; uma tabela que
# isentasse metade dos campos da chave seria o sintoma de o oráculo estar errado.
CAMPOS_A_MAO: dict[str, str] = {
    "estudo": (
        "identificador do estudo, escrito pelo humano na tela. Não vem de `PipelineSpec` porque "
        "não é escolha de pipeline — é rótulo de proveniência —, e entra na chave porque duas "
        "execuções do mesmo dado sob estudos diferentes são trabalhos diferentes para quem lê o "
        "registro"
    ),
    "apresentacao": (
        "como a proposta da máquina foi apresentada ao humano. Entra na chave porque o par de "
        "concordância do `[CA-12.3]` depende dela, e duas apresentações do mesmo dado produzem "
        "linhas de decisão diferentes"
    ),
    "campos_sobrescritos": (
        "quais eixos o humano sobrescreveu sobre a proposta. É derivado de `SUPPORTED_OVERRIDES` "
        "no ponto de chamada e não é campo de dataclass nenhum; entra na chave pelo mesmo motivo "
        "que `apresentacao`"
    ),
    "proposta_json": (
        "a proposta da máquina, serializada. Entra na chave porque a mesma spec final proposta e "
        "aceita não é o mesmo trabalho que a mesma spec final digitada à mão"
    ),
    "group_a": (
        "o primeiro grupo do contraste. Não está em `PipelineSpec` porque é parâmetro de "
        "ANÁLISE e não de pipeline — a mesma distinção que KTD2 fixa —, e é o NUMERADOR do fold "
        "change, então trocá-lo espelha a figura"
    ),
    "group_b": "o segundo grupo do contraste; ver `group_a`",
    # **O sétimo.** Com a tabela sem esta linha, o portão acusou-o — que é o
    # oráculo a fazer o que existe para fazer.
    "sem_proposta": (
        "a declaração de que nenhuma proposta da máquina foi seguida. Entra na chave pelo motivo "
        "de `proposta_json`: o mesmo pedido gravado como proposta seguida e como execução sem "
        "proposta são duas decisões diferentes no registro do `[CA-12.3]`"
    ),
}


# **Os parâmetros de `executar` que NÃO entram na chave, cada um com o motivo.**
#
# Esta é a metade que apanha o defeito que o portão existe para apanhar: parâmetro novo na rota
# que ninguém ligou à chave. A direção oposta — campo na chave que a rota não aceita — é o
# primeiro teste abaixo.
#
# Medido em 13/09/2026, com a tabela vazia: o portão acusa **seis** parâmetros, os seis abaixo.
FORA_DA_CHAVE_POR_DESENHO: dict[str, str] = {
    "measurements": (
        "o arquivo da matriz. Entra na identidade do trabalho pelo DIGEST da matriz ingerida, que "
        "`_chave_do_trabalho` haseia à parte — pôr o `UploadFile` no dicionário haseia o objeto e "
        "não o conteúdo"
    ),
    "sample_list": "a lista de amostras; ver `measurements`",
    # **`normalization_factors` NÃO entra nesta tabela, e a ausência é medida e não esquecimento.**
    # A primeira versão deste arquivo declarou-o aqui, e a âncora
    # `test_a_tabela_de_excecoes_nao_isenta_parametro_que_entra_na_chave` reprovou na primeira
    # execução: o parâmetro da rota e o campo de `PipelineSpec` **partilham o nome**, então o lado
    # derivado já o cobre e a isenção era falsa.
    #
    # A distinção que a isenção queria preservar continua verdadeira e fica dita aqui: o parâmetro
    # é o **CSV de fatores** e o campo é o **mapa resolvido por valor** — a rota lê o
    # arquivo e resolve o mapa antes de construir a especificação. O que o portão confere é o
    # nome; o que viaja na chave é o mapa.
    "mode": (
        "o modo do estudo. Entra na chave por `matriz.mode`, que `_chave_do_trabalho` haseia à "
        "parte junto do digest"
    ),
    "orientation": (
        "como o arquivo está orientado. Não muda valor nenhum da matriz ingerida — muda apenas "
        "como ela é LIDA —, e duas orientações do mesmo dado produzem a mesma `RefinoMatrix`, "
        "logo o mesmo digest"
    ),
    "auto": (
        "a delegação da escolha de corretor. NÃO entra por desenho e a ordem é o argumento: a "
        "delegação corre ANTES da composição da chave e reescreve `spec.correction`, então o que "
        "a chave vê é o corretor que de facto vai correr. Pôr `auto` também tornaria duas "
        "execuções com o mesmo corretor final em trabalhos diferentes"
    ),
    "analise_json": (
        "o contrato de análise, serializado. O que entra na chave são os CAMPOS resolvidos de "
        "`AnalysisSpec` — todos, derivados de `fields()` — e não a string: duas grafias JSON do "
        "mesmo contrato têm de dar a mesma chave, senão reordenar as chaves do objeto "
        "invalidaria figura sem mudar escolha nenhuma. **Sem número aqui de propósito**: eram "
        "vinte e quatro em 13/09/2026 e são vinte e seis desde o item 80, e a contagem escrita à "
        "mão nesta linha caducaria em silêncio enquanto `_campos_derivados` a mede de facto"
    ),
    "paralelo": (
        "a caixa «usar vários núcleos». A saída é bit-idêntica entre um e vários trabalhadores, "
        "por fluxo de semente derivado de `(seed, feature, lote, passo)` — o par causal é "
        "`test_a_saida_e_bit_identica_serial_e_paralela`, e a exclusão aqui é a mesma que "
        "`FORA_DA_IDENTIDADE` já declara para `SerrfCorrector.n_jobs`"
    ),
}


def _campos_derivados() -> set[str]:
    """Os nomes de campo que a chave herda dos dataclasses, sem ninguém os digitar."""
    return {campo.name for classe in DATACLASSES_DA_CHAVE for campo in fields(classe)}


def _chave_de_amostra() -> dict[str, Any]:
    """A chave composta pelo caminho de produção, com valores de amostra.

    **Chama o construtor real e não monta um dicionário à mão** — um oráculo que reconstruísse a
    composição leria a própria cópia, que é exactamente a instância errada que
    `compoe_campos_da_chave` existe para fechar.
    """
    spec = PipelineSpec(
        correction="none",
        n_batches=1,
        rsd_cutoff=None,
        ranking_filter="none",
        ranking_keep_fraction=None,
        ranking_filter_2="none",
        ranking_keep_fraction_2=None,
        rsd_variant="plain",
        missing_threshold=None,
        imputation="none",
        sample_normalization="none",
        normalization_reference=None,
        normalization_factors=None,
        transformation="none",
        scaling="none",
    )
    return rotas.compoe_campos_da_chave(
        spec,
        analise=AnalysisSpec(),
        estudo="estudo",
        apresentacao="exibida",
        campos_sobrescritos=[],
        proposta_json=None,
        sem_proposta=False,
        group_a=None,
        group_b=None,
    )


def _parametros_da_rota() -> set[str]:
    """Os parâmetros nomeados de `executar`, que é a superfície que aceita escolha do usuário."""
    return {
        nome
        for nome, parametro in inspect.signature(rotas.executar).parameters.items()
        if parametro.kind is not inspect.Parameter.VAR_KEYWORD
    }


def test_todo_campo_da_chave_ou_e_derivado_ou_tem_motivo_escrito() -> None:
    """Direção um: nada entra na chave sem dono.

    Um campo à mão que ninguém declarou é um campo que ninguém revisou.
    """
    a_mao = set(_chave_de_amostra()) - _campos_derivados()
    assert a_mao == set(CAMPOS_A_MAO), (
        "a chave de `/execucoes` tem campo à mão sem motivo declarado, ou a tabela declara campo "
        f"que já não existe. Sem motivo: {sorted(a_mao - set(CAMPOS_A_MAO))}; "
        f"declarado e ausente: {sorted(set(CAMPOS_A_MAO) - a_mao)}"
    )


def test_todo_campo_derivado_chega_de_facto_a_chave() -> None:
    """Direção três: o que se declara derivado tem de **estar lá**.

    **Esta cláusula nasceu de o portão ter passado com a planta aplicada.** As duas direções
    acima perguntam *"o que há a mais na chave?"* e *"que parâmetro da rota ficou de fora?"* — e
    nenhuma pergunta se os campos dos dataclasses **chegam** lá. Removido o contrato de análise da
    composição, o primeiro teste via um conjunto MENOR de campos à mão, que é o desfecho de
    sucesso dele; o segundo não olha para campos que não são parâmetros da rota. Os cinco testes
    passaram sobre a chave mutilada.

    É a asserção de **ausência satisfeita por vacuidade** dentro do portão escrito para a
    impedir: declarar um dataclass em `DATACLASSES_DA_CHAVE` sem ele entrar na composição era
    afirmar cobertura que não existia.

    **Presença não é satisfazível por vacuidade**, e é por isso que esta é a cláusula que morde.
    """
    derivados = _campos_derivados()
    # **A âncora vai AQUI e não no teste vizinho.** `assert not faltando` passa sobre conjunto
    # vazio, e um vizinho que a afirme pode ser removido sem que nada ligue os dois.
    assert derivados, "nenhum campo derivado: o oráculo está a ler dataclass nenhum"
    faltando = sorted(derivados - set(_chave_de_amostra()))
    assert not faltando, (
        "campos declarados derivados que NÃO chegam à chave de invalidação: "
        f"{faltando}. Um dataclass em `DATACLASSES_DA_CHAVE` que a composição não espalha é "
        "cobertura afirmada e não exercida — trocar qualquer um destes parâmetros deixaria a "
        "figura da execução anterior na tela, com o rótulo da nova."
    )


def test_nenhum_motivo_esta_em_branco() -> None:
    """A coluna da direita é o que separa tabela honesta de lista de isenção."""
    assert CAMPOS_A_MAO and FORA_DA_CHAVE_POR_DESENHO, (
        "as duas tabelas têm de ter população: sobre tabelas vazias esta asserção passa sem "
        "conferir motivo nenhum"
    )
    vazios = [
        nome
        for tabela in (CAMPOS_A_MAO, FORA_DA_CHAVE_POR_DESENHO)
        for nome, motivo in tabela.items()
        if not motivo.strip()
    ]
    assert not vazios, f"exclusão sem motivo é lista de isenção com outro nome: {vazios}"


def test_todo_parametro_da_rota_ou_entra_na_chave_ou_tem_motivo_escrito() -> None:
    """Direção dois: parâmetro novo na rota que ninguém ligou à chave REPROVA.

    É esta cláusula que apanha o defeito que o portão existe para apanhar. A anterior apanha o
    excesso; esta apanha a **falta**, que é a que deixa a figura velha na tela com o rótulo novo.
    """
    ausentes = _parametros_da_rota() - set(_chave_de_amostra())
    assert ausentes == set(FORA_DA_CHAVE_POR_DESENHO), (
        "parâmetro de `/execucoes` fora da chave de invalidação sem motivo declarado, ou motivo "
        "declarado para parâmetro que já não existe. Trocar um parâmetro que não entra na chave "
        "deixa a figura da execução ANTERIOR na tela, com o rótulo da nova, e nada levanta. "
        f"Sem motivo: {sorted(ausentes - set(FORA_DA_CHAVE_POR_DESENHO))}; "
        f"declarado e ausente: {sorted(set(FORA_DA_CHAVE_POR_DESENHO) - ausentes)}"
    )


def test_a_tabela_de_excecoes_nao_isenta_parametro_que_entra_na_chave() -> None:
    """A âncora de não-vacuidade das duas tabelas.

    Sem ela, declarar um parâmetro em `FORA_DA_CHAVE_POR_DESENHO` que de facto entra na chave
    passaria despercebido — e a tabela cresceria uma linha de cada vez até não restar portão.
    """
    assert FORA_DA_CHAVE_POR_DESENHO, (
        "a tabela de isenções está vazia: a interseção com ela é vazia por construção, e esta "
        "asserção passaria sem conferir nada"
    )
    na_chave = set(_chave_de_amostra())
    isentos_que_entram = sorted(na_chave & set(FORA_DA_CHAVE_POR_DESENHO))
    assert not isentos_que_entram, (
        "estes parâmetros estão declarados FORA da chave e entram nela: "
        f"{isentos_que_entram}. A isenção é falsa"
    )


def test_as_duas_tabelas_nao_estao_vazias() -> None:
    """Âncora de população: um portão sobre conjunto vazio passa sobre nada.

    É a mesma cláusula que as cinco âncoras de `test_plots_arnes.py` fazem para a bijeção do
    catálogo — asserção de ausência é satisfeita por vacuidade.
    """
    assert _campos_derivados(), "nenhum campo derivado: o oráculo está a ler dataclass nenhum"
    assert _chave_de_amostra(), "a chave veio vazia: a composição não correu"
    assert _parametros_da_rota(), "a rota veio sem parâmetros: a introspecção falhou"


# ---------------------------------------------------------------------------
# Os dois cenários de VALOR — e eles são o que separa o portão de nomes do que importa
# ---------------------------------------------------------------------------
#
# **As cinco cláusulas acima conferem NOMES**: que campo entra, que campo falta, que parâmetro tem
# motivo. Nenhuma delas afirma que a chave **muda** quando o parâmetro muda — e é isso que o
# profissional sente. Um `compoe_campos_da_chave` que pusesse os vinte e seis nomes com valores
# constantes passaria nas cinco.
#
# A afirmação honesta do que o portão cobre é, portanto: **os nomes pelas cinco de cima, os valores
# por estes dois**. Dizer «o dicionário inteiro» sem estes dois seria cobertura mais larga que o
# mecanismo.


def _chave_com(**mudancas: Any) -> str:
    """A chave haseada de facto, com o contrato de análise alterado nos campos pedidos."""
    spec = PipelineSpec(
        correction="none",
        n_batches=1,
        rsd_cutoff=None,
        ranking_filter="none",
        ranking_keep_fraction=None,
        ranking_filter_2="none",
        ranking_keep_fraction_2=None,
        rsd_variant="plain",
        missing_threshold=None,
        imputation="none",
        sample_normalization="none",
        normalization_reference=None,
        normalization_factors=None,
        transformation="none",
        scaling="none",
    )
    campos = rotas.compoe_campos_da_chave(
        spec,
        analise=replace(AnalysisSpec(), **mudancas),
        estudo="estudo",
        apresentacao="exibida",
        campos_sobrescritos=[],
        proposta_json=None,
        sem_proposta=False,
        group_a=None,
        group_b=None,
    )
    return rotas._chave_do_trabalho("execucoes", _matriz_minima(), campos)


def _matriz_minima() -> RefinoMatrix:
    """Duas amostras e uma feature — o menor dado sobre o qual a chave é calculável."""
    return RefinoMatrix(
        values=np.array([[1.0], [2.0]]),
        feature_names=("f1",),
        samples=(
            SampleMetadata(name="a", group="A", sample_type="sample", injection_order=1),
            SampleMetadata(name="b", group="B", sample_type="sample", injection_order=2),
        ),
        mode="pos_hilic",
    )


def test_trocar_o_post_hoc_muda_a_chave() -> None:
    """O par causal que prova que o portão de nomes serve para alguma coisa.

    Sem esta asserção, um contrato que entrasse na chave com valores constantes satisfaria as
    cinco cláusulas de nome — e trocar o post-hoc deixaria a figura da execução anterior na tela,
    com o rótulo da nova, que é exactamente o defeito que este arquivo existe para impedir.
    """
    assert _chave_com(post_hoc="tukey") != _chave_com(post_hoc="fisher"), (
        "duas execuções que diferem apenas no post-hoc produziram a MESMA chave. O painel "
        "concluiria que é a mesma execução e deixaria a figura antiga na tela"
    )


def test_o_contrato_no_padrao_da_a_mesma_chave_que_o_padrao_implicito() -> None:
    """A direção oposta, e ela protege o reuso de tudo o que já está guardado.

    Sem ela, a entrada do contrato na chave invalidaria **toda** execução gravada antes deste
    bloco: o reuso pararia de acontecer em silêncio, e o sintoma no produto seria *"ficou lento"* —
    não um erro que alguém investigasse.
    """
    assert _chave_com() == _chave_com(anova=AnalysisSpec().anova), (
        "compor a chave com o contrato padrão explícito e com o implícito deu chaves diferentes: "
        "a igualdade que protege o reuso do que já está guardado quebrou"
    )


def test_d707_declarar_sem_proposta_muda_a_chave() -> None:
    """`sem_proposta` muda a decisão gravada, então muda o trabalho.

    **Par causal de VALOR**, pelo motivo que a seção acima dá: o portão de nomes passaria com o
    campo na chave a valor constante. Com a mesma chave, `_despacha` serviria a execução gravada
    como proposta seguida a quem pediu a execução sem proposta, e o registro ficava com uma só.
    """
    spec = PipelineSpec(
        correction="none",
        n_batches=1,
        rsd_cutoff=None,
        ranking_filter="none",
        ranking_keep_fraction=None,
        ranking_filter_2="none",
        ranking_keep_fraction_2=None,
        rsd_variant="plain",
        missing_threshold=None,
        imputation="none",
        sample_normalization="none",
        normalization_reference=None,
        normalization_factors=None,
        transformation="none",
        scaling="none",
    )

    def chave(sem_proposta: bool) -> str:
        campos = rotas.compoe_campos_da_chave(
            spec,
            analise=AnalysisSpec(),
            estudo="estudo",
            apresentacao="exibida",
            campos_sobrescritos=[],
            proposta_json=None,
            sem_proposta=sem_proposta,
            group_a=None,
            group_b=None,
        )
        return rotas._chave_do_trabalho("execucoes", _matriz_minima(), campos)

    assert chave(True) != chave(False), (
        "declarar `sem_proposta` deu a mesma chave: o trabalho gravado como proposta seguida "
        "seria servido a quem declarou que não houve proposta"
    )
