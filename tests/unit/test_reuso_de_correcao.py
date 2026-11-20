"""O REUSO de correção entre rotas: fiel, e de facto usado.

## As TRÊS perguntas, e são três testes porque falham independentemente

| Pergunta | O que quebra sem ela |
|---|---|
| o reuso devolve o MESMO resultado? | matriz plausível e diferente, sem nada acusar |
| a chave SEPARA o que muda valores? | a matriz de outro parâmetro, com o rótulo certo |
| o reuso de facto ACERTA? | cache que nunca acerta é custo puro, e o sintoma é ausência de ganho |

A terceira é a que a verificação exige e que ninguém escreve: as duas primeiras ficam verdes sobre
uma memória que nunca guarda nada.

## O molde é o do `run_naive`

O executor do multiverso prova a árvore de correções contra um caminho ingénuo que **não** é
oferecido ao usuário e existe só para ser o oráculo, com comparação **bit-idêntica** e não por
tolerância. Aqui o oráculo já existe e não precisa de ser escrito: é `fit_transform` chamado
directamente.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from refino.api.reuso import MemoriaDeCorrecoes, chave_da_correcao, digest_da_matriz
from refino.correction.loess import LoessCorrector
from refino.correction.none import NoneCorrector
from refino.domain.matrix import RefinoMatrix
from refino.domain.metadata import SampleMetadata


def _matriz(n_qc: int = 4, n_amostras: int = 4, n_features: int = 6) -> RefinoMatrix:
    """Uma matriz pequena com QCs suficientes para o LOESS ajustar.

    **Construída à mão e isso NÃO viola a regra 1 do README**: nenhum número deste arquivo é
    observação sobre os arquivos do laboratório. O que se mede é a travessia de uma chave e a
    identidade de um resultado, e as duas são propriedades do código.
    """
    total = n_qc + n_amostras
    gerador = np.random.default_rng(20260906)
    valores = gerador.uniform(10.0, 100.0, size=(total, n_features))
    # **Uma feature com os QCs TODOS a zero, de propósito.** Ela produz denominador indefinido e o
    # LOESS devolve `NaN` naquela coluna — que é o estado que o dado real tem às centenas e que a
    # comparação de bit-identidade precisa de exercitar. Sem ela, `equal_nan=True` fica decorativo.
    valores[:n_qc, 0] = 0.0
    amostras = tuple(
        SampleMetadata(
            name=f"QC{i + 1}" if i < n_qc else f"S{i - n_qc + 1}",
            injection_order=i + 1,
            sample_type="qc" if i < n_qc else "sample",
            group="QC" if i < n_qc else "A",
            batch="A",
        )
        for i in range(total)
    )
    return RefinoMatrix(
        values=valores,
        feature_names=tuple(f"1.0_{100 + j}m/z" for j in range(n_features)),
        samples=amostras,
        mode="pos_hilic",
    )


def _corrige_reusando(memoria: MemoriaDeCorrecoes, corretor: Any, matriz: RefinoMatrix) -> Any:
    """A forma que a rota usa, reduzida ao essencial e com a memória INJETADA.

    A rota tem uma instância única de processo; aqui cada teste tem a sua, senão a suíte passa a
    depender de ordem — o hazard que os testes da rota de medição já nomeiam para o registro de
    trabalhos, e que eles resolvem variando a entrada em vez de limpar o global.
    """
    chave = chave_da_correcao(corretor, matriz, batches=None, seed=42, target_features=None)
    guardado = memoria.busca(chave)
    if guardado is not None:
        return guardado
    resultado = corretor.fit_transform(matriz, batches=None, seed=42)
    memoria.guarda(chave, resultado)
    return resultado


def test_o_reuso_devolve_resultado_bit_identico_ao_calculo_direto() -> None:
    """O ORÁCULO: três braços, e os três têm de dar o mesmo em VALORES, causas e procedência.

    **`array_equal` e não `allclose`.** Reuso de um resultado idêntico não pode mudar dígito
    nenhum, e tolerância esconderia exactamente o defeito que interessa — é a mesma frase que o
    A equivalência usa para comparar a árvore contra o caminho ingénuo.

    **`equal_nan=True`, e sem isso o teste tinha um FALSO NEGATIVO em dado real.** Medido em
    06/09/2026 no `neg_c18`: o LOESS deixa **240** células `NaN`, e `NaN != NaN` — `array_equal`
    devolvia `False` sobre o **mesmo objeto** devolvido pelo cache. O teste passava só porque a
    fixture sintética não produzia `NaN` nenhum, e teria acusado o alvo por um defeito do
    instrumento no primeiro dado do laboratório: o veredito que fala do alvo
    quando a causa está em quem mede.

    **A âncora abaixo é o que impede a correção de ficar decorativa:** a fixture tem de conter
    `NaN`, senão `equal_nan=True` nunca é exercitado e a próxima pessoa tira-o por supérfluo.

    **`provenance` entra na comparação e é o campo que mais importa aqui**: ele é o único que
    denunciaria um resultado diferente sem que um valor mudasse.
    """
    matriz = _matriz()
    corretor = LoessCorrector()
    memoria = MemoriaDeCorrecoes()

    direto = corretor.fit_transform(matriz, batches=None, seed=42)
    frio = _corrige_reusando(memoria, corretor, matriz)
    quente = _corrige_reusando(memoria, corretor, matriz)

    assert memoria.acertos == 1 and memoria.erros == 1, (
        f"âncora: o segundo braço não foi um acerto ({memoria.acertos} acertos, {memoria.erros} "
        f"erros) — os três braços correram todos o corretor, e o teste não mede reuso nenhum"
    )
    assert quente is frio, "o acerto devolveu um objeto diferente do que foi guardado"

    # **Âncora 1: a fixture produz valor NÃO FINITO.** Sob `r_faithful` o denominador zero dá
    # `Inf`, e o dado real do `neg_c18` traz **240** células `NaN` na mesma correção. Se a fixture
    # só produzisse números finitos, esta comparação nunca tocaria o caso que importa.
    assert not np.isfinite(direto.matrix.values).all(), (
        "âncora: a correção desta fixture só produz valores finitos, e a comparação nunca exercita "
        "o caso não finito — que é o que o dado real tem às centenas"
    )

    # **Âncora 2: é `equal_nan=True` que faz a comparação funcionar sobre `NaN`, e isto é
    # DEMONSTRADO em vez de afirmado.** `NaN != NaN`, então sem a bandeira `array_equal` devolve
    # `False` sobre o **mesmo objeto** — medido em 06/09/2026 no `neg_c18`. Seria o
    # veredito a acusar o alvo por um defeito de quem mede.
    com_nan = np.array([1.0, np.nan])
    assert not np.array_equal(com_nan, com_nan), (
        "âncora: `array_equal` deixou de tratar `NaN` como diferente de si próprio, e a bandeira "
        "abaixo passou a ser supérflua — se for isso, ela sai com esta âncora"
    )
    assert np.array_equal(com_nan, com_nan, equal_nan=True)

    for nome, obtido in (("frio", frio), ("quente", quente)):
        assert np.array_equal(obtido.matrix.values, direto.matrix.values, equal_nan=True), (
            f"o braço `{nome}` diverge do cálculo direto em VALORES"
        )
        assert obtido.uncorrected == direto.uncorrected, (
            f"o braço `{nome}` diverge nas causas de não correção"
        )
        assert obtido.matrix.provenance == direto.matrix.provenance, (
            f"o braço `{nome}` diverge na PROCEDÊNCIA — é o campo que denuncia um resultado "
            f"diferente sem que um valor mude"
        )


def test_a_chave_separa_os_dois_denominadores_do_loess() -> None:
    """O PAR CAUSAL, e ele é o caso VIVO — não uma planta construída.

    As rotas de artefato têm padrões diferentes para `null_denominator`: `r_faithful` em `corrigir`
    e `processar`, `uncorrected_feature` em `exportar`. A janela **não manda o campo a nenhuma
    delas**. Sob `loess`, o par que o profissional clica corre denominadores diferentes — e uma
    chave que ignorasse o eixo serviria uma correção pela outra.

    **A âncora é a divergência dos VALORES**: se as duas políticas produzissem a mesma matriz nesta
    fixture, a asserção sobre as chaves passaria e não separaria nada — seria a planta que não cai
    sobre dado em que as duas coisas diferem.
    """
    matriz = _matriz()
    fiel = LoessCorrector(null_denominator="r_faithful")
    outro = LoessCorrector(null_denominator="uncorrected_feature")

    chave_fiel = chave_da_correcao(fiel, matriz, batches=None, seed=42, target_features=None)
    chave_outro = chave_da_correcao(outro, matriz, batches=None, seed=42, target_features=None)

    assert chave_fiel != chave_outro, (
        "as duas políticas de denominador dão a MESMA chave: `corrigir` e `exportar` receberiam a "
        "mesma matriz, e elas correm denominadores diferentes por padrão"
    )

    # E a chave não separa por separar: o mesmo corretor duas vezes dá a mesma chave.
    assert chave_fiel == chave_da_correcao(
        LoessCorrector(null_denominator="r_faithful"),
        matriz,
        batches=None,
        seed=42,
        target_features=None,
    ), "duas instâncias iguais dão chaves diferentes — o reuso nunca acertaria"


@pytest.mark.parametrize(
    ("campo", "mudanca"),
    [
        ("sample_type", {"sample_type": "sample"}),
        ("batch", {"batch": "B"}),
        ("group", {"group": "OUTRO"}),
    ],
)
def test_o_digest_da_matriz_ve_o_metadado_de_amostra(campo: str, mudanca: dict[str, Any]) -> None:
    """Os quatro campos que `values` não vê, e cada um muda o cálculo ou identifica a matriz.

    **`sample_type` é o mais grave e por isso encabeça a lista:** ele vem do rótulo `LABEL` do
    `measurements` e **não entra em `values`**. `qc_mask` é a régua do sistema — decide o treino do
    SERRF e o eixo do LOESS —, então duas matrizes com os mesmos números e uma amostra reetiquetada
    produzem correções diferentes.

    **`injection_order` tem teste PRÓPRIO abaixo**, e a razão é o invariante do domínio: a matriz
    exige ordem de injeção crescente, então trocar o campo numa amostra só levanta na construção.
    O caso real é a reescala do vetor inteiro.
    """
    from dataclasses import replace

    base = _matriz()
    amostras = list(base.samples)
    amostras[0] = replace(amostras[0], **mudanca)
    plantada = replace(base, samples=tuple(amostras))

    assert digest_da_matriz(plantada) != digest_da_matriz(base), (
        f"trocar `{campo}` numa amostra não mudou o digest: a correção da primeira matriz seria "
        f"servida à segunda, com a tela a mostrar o nome certo"
    )


def test_o_digest_ve_a_reescala_de_injection_order_que_preserva_a_ordem() -> None:
    """O campo mais INSIDIOSO da lista, e ele precisa de teste próprio por causa do invariante.

    `RefinoMatrix` exige `injection_order` crescente, então não há como plantá-lo numa amostra só —
    a construção levanta, e é bom que levante. O caso que importa é outro e é legítimo:

    > `1, 2, 3` reescalado para `10, 20, 30` **mantém a ordem das linhas**, logo mantém o hash dos
    > valores e o dos nomes. Nada além deste campo a vê.

    E ele muda o cálculo: `injection_order` é o `x` do LOESS, e é o que `_lotes` ordena para
    derivar as fronteiras de lote.

    **A âncora é a ordem preservada**: se a reescala trocasse a ordem, o teste mediria outra coisa
    — mediria que reordenar linhas muda o digest, que é verdade por `values` e não prova nada sobre
    este campo.
    """
    from dataclasses import replace

    base = _matriz()
    reescalada = replace(
        base,
        samples=tuple(replace(a, injection_order=a.injection_order * 10) for a in base.samples),
    )

    ordens_base = [a.injection_order for a in base.samples]
    ordens_novas = [a.injection_order for a in reescalada.samples]
    assert ordens_novas != ordens_base, "âncora: a reescala não mudou número nenhum"
    assert sorted(ordens_novas) == ordens_novas, (
        "âncora: a reescala trocou a ORDEM, e então o teste mede reordenação e não este campo"
    )
    assert np.array_equal(reescalada.values, base.values), (
        "âncora: a reescala mexeu nos valores, e o digest mudaria por eles"
    )

    assert digest_da_matriz(reescalada) != digest_da_matriz(base), (
        "reescalar a ordem de injeção não mudou o digest — e ela é o `x` do LOESS e o que `_lotes` "
        "ordena. Nada mais na matriz vê esta mudança"
    )


def test_a_memoria_despeja_o_mais_antigo_e_o_acerto_conta_como_uso() -> None:
    """LRU e não FIFO, e a diferença é o que o acerto faz.

    **A política NÃO é a dos trabalhos, e a diferença é o referente.** Lá a poda escolhe «reclamado
    primeiro», porque a pergunta é *quem ainda espera pelo seu artefato*. Aqui a pergunta é *qual
    correção será pedida outra vez* — e copiar aquela política despejaria precisamente a que acabou
    de ser reusada.

    **Par causal:** o acerto no primeiro item tem de ser o que o salva do despejo. Sem a segunda
    metade deste teste, uma memória FIFO passaria pela primeira.
    """
    memoria = MemoriaDeCorrecoes(limite=2)
    corretor = NoneCorrector()
    matriz = _matriz()
    resultado = corretor.fit_transform(matriz, batches=None, seed=42)

    assert memoria.guarda("a", resultado) == (), "despejou com a memória vazia"
    assert memoria.guarda("b", resultado) == (), "despejou antes de o teto encher"
    assert len(memoria) == 2

    # O acerto em `a` move-a para o fim: quem sai a seguir é `b`, e não `a`.
    assert memoria.busca("a") is resultado, "âncora: `a` não estava guardada e o teste não separa"
    assert memoria.guarda("c", resultado) == ("b",), (
        "o despejo não é LRU: saiu o item errado, e sob FIFO sairia justamente o que acabou de ser "
        "reusado"
    )
    assert memoria.busca("b") is None
    assert memoria.busca("a") is resultado


def test_uma_memoria_de_teto_zero_levanta_em_vez_de_nunca_acertar() -> None:
    """Teto de zero seria uma memória que nunca acerta — custo puro, e o sintoma é silêncio.

    Ela guardaria e despejaria na mesma chamada, e todo teste de fidelidade ficaria verde.
    *"Desligado"* e *"ligado e inútil"* têm de ser estados distinguíveis, e o primeiro
    escreve-se não instanciando a classe.
    """
    with pytest.raises(ValueError, match="nunca acerta"):
        MemoriaDeCorrecoes(limite=0)
