"""As rotas até o diagnóstico.

**A API é a fronteira de confiança mais exposta do projeto**, e é isso que estes testes
exercitam além do caminho feliz: um `Literal` vindo do corpo da requisição nunca passou
pelo mypy, e o modo de falha que a guarda impede não levanta por si — cai num `else`
razoável e se comporta como outra política.

**Escopo negativo.** Estes testes não abrem porta: usam `TestClient`, que fala com a
aplicação em processo. Quem verifica a superfície de rede é `tests/test_api_loopback.py`.
"""

import json
import math
import threading
from pathlib import Path
from typing import Any, get_args

import numpy as np
import pytest
from fastapi.testclient import TestClient

from refino.api.app import build_app
from refino.api.rotas import ESTAGIO_DA_INGERIDA
from refino.diagnosis.profile import PRETRANSFORM
from refino.domain.analysis import PADRAO_DA_ANALISE
from refino.domain.matrix import RefinoMatrix
from refino.domain.metadata import SampleMetadata
from refino.domain.statistics import FoldChangeIntervalUndefined
from refino.interface.cliente import Resposta
from refino.interface.fluxo import familias_pedidas
from refino.io.readers import load_matrix
from refino.metrics.rsd import qc_rsd
from refino.metrics.structure import structural_pca
from refino.plots.dispersao import MATRIZ_DA_LEITURA
from tests.conftest import measurements_path, sample_list_path
from tests.factories import inject, render_pair

MODES = ("pos_hilic", "pos_c18", "neg_hilic", "neg_c18")


@pytest.fixture(scope="module")
def cliente() -> TestClient:
    return TestClient(build_app())


def _reais(mode: str) -> dict[str, tuple[str, bytes, str]]:
    return {
        "measurements": (
            f"measurements_{mode}.csv",
            measurements_path(mode).read_bytes(),
            "text/csv",
        ),
        "sample_list": (f"Sample_List_{mode}.csv", sample_list_path(mode).read_bytes(), "text/csv"),
    }


# ---------------------------------------------------------------------------
# `/matrizes/validar`
# ---------------------------------------------------------------------------


def test_validar_aceita_os_quatro_modos_reais(cliente: TestClient) -> None:
    """O caminho feliz é exercitado sobre os arquivos do laboratório."""
    for mode in MODES:
        resposta = cliente.post("/matrizes/validar", files=_reais(mode), data={"mode": mode})

        assert resposta.status_code == 200, (mode, resposta.text)
        corpo = resposta.json()
        assert corpo["aceita"] is True, (mode, corpo["problemas"])
        assert corpo["n_samples"] and corpo["n_features"]


def test_validar_devolve_os_NOMES_e_nao_so_as_contagens(cliente: TestClient) -> None:  # noqa: N802
    """A capacidade que as normalizações contra referência exigem.

    **Contagem não substitui nomes**, e é por isso que os campos convivem: `n_samples` responde
    *"quantas"*, e um controle que peça ao profissional para NOMEAR uma amostra precisa de
    *"quais"*. Sem esta lista o único caminho seria digitar de memória e descobrir o erro depois
    da correção — que no `neg_c18` custa 490 s em série.

    As duas asserções de comprimento são a âncora contra a lista vazia: `assert corpo[...]` sobre
    uma lista vazia é a asserção de ausência que passa por vacuidade.
    """
    for mode in MODES:
        resposta = cliente.post("/matrizes/validar", files=_reais(mode), data={"mode": mode})

        assert resposta.status_code == 200, (mode, resposta.text)
        corpo = resposta.json()
        assert len(corpo["sample_names"]) == corpo["n_samples"], (
            f"{mode}: a rota devolveu {len(corpo['sample_names'])} nomes de amostra e diz ter "
            f"{corpo['n_samples']} amostras"
        )
        assert len(corpo["feature_names"]) == corpo["n_features"], (
            f"{mode}: a rota devolveu {len(corpo['feature_names'])} nomes de feature e diz ter "
            f"{corpo['n_features']} features"
        )
        assert all(isinstance(nome, str) and nome for nome in corpo["sample_names"])


def test_validar_RECUSADA_nao_devolve_nomes(cliente: TestClient, tmp_path: Path) -> None:  # noqa: N802
    """A outra direção, e ela é o que impede o campo de virar vazamento.

    Uma matriz que não passou não tem nomes de que se possa escolher, e devolvê-los faria a tela
    oferecer opções sobre um arquivo rejeitado. Sem este teste, a asserção acima passaria numa
    implementação que devolvesse nomes sempre.
    """
    quebrado = tmp_path / "measurements.csv"
    quebrado.write_text("isto nao e um csv de medicoes\n", encoding="utf-8")
    lista = tmp_path / "Sample_List.csv"
    lista.write_text("label,time\nQC1,1\n", encoding="utf-8")

    with quebrado.open("rb") as m, lista.open("rb") as s:
        resposta = cliente.post(
            "/matrizes/validar",
            files={
                "measurements": ("m.csv", m, "text/csv"),
                "sample_list": ("s.csv", s, "text/csv"),
            },
            data={"mode": "pos_hilic"},
        )

    assert resposta.status_code == 200, resposta.text
    corpo = resposta.json()
    assert corpo["aceita"] is False, "a planta não aterrou: o arquivo quebrado foi aceite"
    assert corpo["sample_names"] == []
    assert corpo["feature_names"] == []
    assert corpo["ordens_de_injecao"] == []
    assert corpo["qc"] == []


def test_validar_devolve_a_ordem_de_injecao_e_a_marca_de_qc_alinhadas(cliente: TestClient) -> None:
    """A tela conta os QCs de cada lote declarado com estas duas listas.

    O oráculo é a matriz carregada pelo outro caminho, o leitor, e não um número: a rota tem de
    devolver exactamente a ordem e a máscara que a correção vai usar. A âncora contra a lista vazia
    é o comprimento, e a de que a máscara não é constante é haver QC e amostra.
    """
    for mode in MODES:
        resposta = cliente.post("/matrizes/validar", files=_reais(mode), data={"mode": mode})
        corpo = resposta.json()
        matriz = load_matrix(measurements_path(mode), sample_list_path(mode), mode)
        assert len(corpo["qc"]) == corpo["n_samples"] == matriz.n_samples, mode
        assert any(corpo["qc"]) and not all(corpo["qc"]), mode
        assert corpo["ordens_de_injecao"] == matriz.injection_orders.tolist(), mode
        assert corpo["qc"] == matriz.qc_mask.tolist(), mode


def test_validar_devolve_todos_os_problemas_e_nao_o_primeiro(
    cliente: TestClient, tmp_path: Path
) -> None:
    """A exigência que a API não pode desfazer.

    Quem tem um arquivo com vários defeitos precisa de várias linhas e não de várias
    requisições. A ingestão cumpre isso por desenho — coletar e abortar são passos
    distintos — e truncar aqui reintroduziria o defeito na camada de cima.
    """
    especificacao = inject("RAGGED_ROWS")
    medidas, lista = render_pair(especificacao)

    resposta = cliente.post(
        "/matrizes/validar",
        files={
            "measurements": ("m.csv", medidas.encode(), "text/csv"),
            "sample_list": ("s.csv", lista.encode(), "text/csv"),
        },
        data={"mode": "tiny"},
    )

    assert resposta.status_code == 200, "dado ruim é resposta legítima, não erro de protocolo"
    corpo = resposta.json()
    assert corpo["aceita"] is False
    assert corpo["problemas"], "a lista de problemas está vazia numa matriz recusada"
    for problema in corpo["problemas"]:
        assert problema["codigo"] and problema["severidade"] and problema["mensagem"]


def test_validar_rejeita_orientation_invalida_na_fronteira(cliente: TestClient) -> None:
    """`Literal` vindo do corpo da requisição não passou pelo mypy.

    O modo de falha que a guarda impede **não levanta por si**: `features_in_rows` com
    sublinhado cairia no ramo da outra orientação e transporia a matriz ao contrário,
    produzindo número medido sobre o eixo errado e `provenance` com o nome errado.
    """
    resposta = cliente.post(
        "/matrizes/validar",
        files=_reais("pos_c18"),
        data={"mode": "pos_c18", "orientation": "features_in_rows"},
    )

    assert resposta.status_code == 422
    assert "orientation" in resposta.text.lower() or "Orientation" in resposta.text


def test_validar_aceita_a_orientation_canonica(cliente: TestClient) -> None:
    """A direção que impede a guarda de recusar tudo.

    Sem ela, um validador que rejeitasse qualquer valor passaria o teste acima e tornaria
    a rota inutilizável.
    """
    resposta = cliente.post(
        "/matrizes/validar",
        files=_reais("pos_c18"),
        data={"mode": "pos_c18", "orientation": "features-in-rows"},
    )

    assert resposta.status_code == 200
    assert resposta.json()["aceita"] is True


# ---------------------------------------------------------------------------
# `/matrizes/diagnosticar`
# ---------------------------------------------------------------------------


def test_diagnosticar_marca_os_lotes_como_sugestao(cliente: TestClient) -> None:
    """Os lotes são sugestão, não fato.

    A marca vem num objeto próprio e é literal constante. Um campo perdido no meio de doze
    métricas é campo que ninguém lê, e a confirmação humana perderia o objeto.
    """
    resposta = cliente.post(
        "/matrizes/diagnosticar", files=_reais("pos_c18"), data={"mode": "pos_c18"}
    )

    assert resposta.status_code == 200, resposta.text
    lotes = resposta.json()["lotes"]
    assert lotes["natureza"] == "sugestao"
    assert lotes["n_lotes"] >= 1
    assert len(lotes["rotulos_por_amostra"]) == resposta.json()["n_samples"]


def test_diagnosticar_devolve_o_perfil_completo(cliente: TestClient) -> None:
    """Nenhuma métrica única, na camada de transporte.

    A rota devolve o conjunto que o `DefectProfile` carrega. Reduzir a resposta a um
    número — "qualidade: 0,72" — quebraria essa exigência na serialização, onde
    nenhum teste de núcleo a alcança.
    """
    corpo = cliente.post(
        "/matrizes/diagnosticar", files=_reais("neg_c18"), data={"mode": "neg_c18"}
    ).json()

    for campo in (
        "defect_type",
        "confidence",
        "qc_rsd_median",
        "qc_rsd_p75",
        "zero_fraction",
        "missing_fraction",
        "pc1_injection_correlation",
        "qc_dispersion_ratio",
    ):
        assert campo in corpo, f"o perfil não traz {campo}"


def test_u13_diagnosticar_publica_o_par_da_permanova_sobre_os_scores_que_devolve(
    cliente: TestClient,
) -> None:
    """Os dois `p` viajam, e a população é a de ESTUDO e não a da nuvem.

    A asserção de população é o centro deste teste. Os `scores_pc1` do mesmo
    objeto descrevem **todas** as injeções — 68 nos quatro arquivos, com os 7 QCs entre elas —, e o
    teste corre sobre as 61 de estudo. Se `n_por_grupo` somasse 68, a figura publicaria um `n` que
    inclui a régua como se fosse um grupo clínico.

    As contagens são observação sobre os quatro arquivos do laboratório e não invariante: o que
    se afirma é a IGUALDADE entre a soma e o número de amostras de estudo que a mesma resposta
    declara, derivada do corpo — não os quatro números escritos à mão.
    """
    corpo = cliente.post(
        "/matrizes/diagnosticar", files=_reais("pos_hilic"), data={"mode": "pos_hilic"}
    ).json()

    pca = corpo["pca"]
    par = pca["permanova"]
    assert par is not None, "a PERMANOVA não correu sobre uma matriz com quatro grupos clínicos"

    # O par é publicado inteiro, e nunca metade.
    for chave in ("estatistica", "pvalue", "estatistica_da_dispersao", "pvalue_da_dispersao"):
        assert isinstance(par[chave], float), f"{chave} não veio como número: {par[chave]!r}"
    assert 0.0 < par["pvalue"] <= 1.0
    assert 0.0 < par["pvalue_da_dispersao"] <= 1.0

    # **A população do teste é MENOR que a da nuvem, e a diferença é o QC.** Derivada do corpo.
    de_estudo = sum(1 for tipo in pca["tipos_de_amostra"] if tipo == "sample")
    assert sum(par["n_por_grupo"]) == de_estudo, (
        f"o teste declara {sum(par['n_por_grupo'])} amostras e a matriz tem {de_estudo} de estudo"
    )
    assert de_estudo < len(pca["scores_pc1"]), (
        "a planta não separa: nesta matriz não há QC, então a população do teste coincide com a da "
        "nuvem por acidente e esta asserção não afirma nada"
    )
    assert len(par["grupos"]) == len(par["n_por_grupo"]) >= 2
    assert len(par["distancia_media_ao_centroide"]) == len(par["grupos"])
    assert par["n_componentes"] == pca["n_componentes"]
    assert par["permutacoes"] == PADRAO_DA_ANALISE.permanova_permutations
    assert par["metodo"], "o método não viaja, e a figura teria de o inventar"


def test_u13_a_semente_da_rota_de_diagnostico_MOVE_o_p_e_repete_com_o_mesmo_valor(  # noqa: N802
    cliente: TestClient,
) -> None:
    """Determinismo na fronteira: a rota ficou estocástica, e a semente é do chamador.

    **As duas direções.** Duas chamadas com a mesma semente dão o mesmo par; uma com outra semente
    move ao menos um dos dois `p`. Sem a segunda metade, um `seed` que a rota aceitasse e
    descartasse passaria — que é o defeito que `rsd_mediana_max` já foi nesta mesma rota.

    **A estatística OBSERVADA não depende da semente**, e afirmá-lo é o que separa *"a semente
    chegou"* de *"a semente moveu o cálculo todo"*: só a distribuição nula é sorteada.
    """

    def par(semente: int) -> dict[str, Any]:
        corpo = cliente.post(
            "/matrizes/diagnosticar",
            files=_reais("pos_hilic"),
            data={"mode": "pos_hilic", "seed": semente},
        ).json()
        return dict(corpo["pca"]["permanova"])

    um = par(42)
    repetido = par(42)
    assert (um["pvalue"], um["pvalue_da_dispersao"]) == (
        repetido["pvalue"],
        repetido["pvalue_da_dispersao"],
    )

    outro = par(20260918)
    assert (outro["estatistica"], outro["estatistica_da_dispersao"]) == (
        um["estatistica"],
        um["estatistica_da_dispersao"],
    ), "a semente moveu a estatística observada, e ela é determinística por construção"
    assert (outro["pvalue"], outro["pvalue_da_dispersao"]) != (
        um["pvalue"],
        um["pvalue_da_dispersao"],
    ), "duas sementes deram o mesmo par de `p`: a rota aceita `seed` e descarta-o"


def _matriz_de_grupos(rotulos: tuple[str, ...]) -> RefinoMatrix:
    """Uma matriz com um rótulo clínico por amostra de estudo, e nada mais de interessante.

    Os valores são ruído positivo: o que estes testes exercitam é a GUARDA de população, e uma
    estrutura plantada tornaria a leitura ambígua entre *"a população não sustenta"* e *"não há
    separação"*, que são os dois fatos que se está a separar.
    """
    gerador = np.random.default_rng(20260918)
    valores = 100.0 + gerador.standard_normal((len(rotulos), 5))
    amostras = tuple(
        SampleMetadata(name=f"a{i:02d}", injection_order=i + 1, sample_type="sample", group=rotulo)
        for i, rotulo in enumerate(rotulos)
    )
    return RefinoMatrix(
        values=valores,
        feature_names=tuple(f"f{j}" for j in range(5)),
        samples=amostras,
        mode="pos_hilic",
    )


def _sem_imputadas(matriz: Any) -> Any:
    """A máscara de imputadas de uma matriz que não passou por imputação: vazia.

    `corre_as_analises` passou a exigi-la, pela normalidade. As matrizes destes testes são
    construídas à mão e nenhuma célula delas foi preenchida pela imputação, então a máscara
    verdadeira é esta — e não um padrão que a rota aceitasse sem a pedir.
    """
    return np.zeros(np.asarray(matriz.values).shape, dtype=bool)


@pytest.mark.parametrize(
    ("rotulos", "porque"),
    [
        (("a",) * 8, "um grupo só: não há partição a testar"),
        (
            ("a",) * 7 + ("b",),
            "o segundo grupo tem uma amostra: a dispersão dele é zero por"
            " construção, e o teste dela não mede nada",
        ),
    ],
)
def test_u13_a_permanova_devolve_None_quando_a_populacao_nao_sustenta(  # noqa: N802
    rotulos: tuple[str, ...], porque: str
) -> None:
    """A metade que NÃO derruba a rota, e é ela que faz do campo um `None` em vez de um `500`.

    **Derrubar `/matrizes/diagnosticar` inteira seria a alternativa, e é errada:** o diagnóstico é
    o que o profissional corre **antes** de decidir qualquer coisa, e uma matriz de um grupo
    clínico só é dado legítimo dele. Uma família opcional não pode custar a leitura toda.

    **A excepção que a rota apanha é TIPADA**, e é isso que este par de casos exercita sem o dizer:
    um `except ValueError` largo apanharia também o erro de rótulos desalinhados dos scores, que é
    defeito de programação, e a resposta diria *"a PERMANOVA não correu"* sobre um bug.
    """
    from refino.api.rotas import _permanova_dos_grupos

    matriz = _matriz_de_grupos(rotulos)
    pca = structural_pca(matriz, pretransform=PRETRANSFORM, n_components=2)

    assert _permanova_dos_grupos(pca, matriz, seed=42) is None, porque


def test_u13_a_permanova_da_rota_corre_sobre_os_scores_do_MESMO_objeto() -> None:  # noqa: N802
    """A âncora positiva do par acima, e a asserção de PROCEDÊNCIA dos scores.

    **A planta é um `StructuralPCA` com os scores TROCADOS.** Se `_permanova_dos_grupos` calculasse
    a PCA por si em vez de ler o objeto que recebe, o resultado seria o mesmo nas duas chamadas — e
    a asserção de divergência abaixo é o que prova que ele lê o argumento.
    """
    import dataclasses

    from refino.api.rotas import _permanova_dos_grupos

    matriz = _matriz_de_grupos(("a",) * 6 + ("b",) * 6)
    pca = structural_pca(matriz, pretransform=PRETRANSFORM, n_components=2)

    verdadeiro = _permanova_dos_grupos(pca, matriz, seed=42)
    assert verdadeiro is not None
    assert list(verdadeiro.n_por_grupo) == [6, 6]

    # Os scores multiplicados por um fator diferente em cada componente: outra decomposição, mesma
    # forma. É o que uma fonte trocada produziria.
    trocado = dataclasses.replace(pca, scores=pca.scores * np.array([1.0, 25.0]))
    outro = _permanova_dos_grupos(trocado, matriz, seed=42)
    assert outro is not None
    assert outro.estatistica != pytest.approx(verdadeiro.estatistica, rel=1e-6), (
        "a estatística não mudou com os scores trocados: a função recalcula a PCA em vez de ler o "
        "objeto que a resposta publica, e o número do rodapé seria de outra decomposição"
    )


def test_u13_a_bandeira_permanova_NAO_controla_o_calculo_e_isto_e_um_portao() -> None:  # noqa: N802
    """**A caixa «PERMANOVA nos componentes» não liga nem desliga o cálculo — declarado e MEDIDO.**

    O par vive onde ele descreve a nuvem que a figura desenha: em `/matrizes/diagnosticar`, do
    **mesmo** `StructuralPCA` de que saem `scores_pc1` e `scores_pc2`. Essa rota não recebe
    `AnalysisSpec`, então corre o par sempre.

    **Por que não em `corre_as_analises`, que é onde as famílias aterram.** Aquela rota entrega a
    matriz **processada**, e o `p` dela descreveria outra decomposição que a desenhada — as duas
    PCAs deste repositório correlacionam `-0,2324` nos segundos componentes. Pior: o número viajaria
    sob a chave de invalidação de `/execucoes` enquanto os pontos da figura **não** viajam sob
    nenhuma, e a sequência *diagnosticar A → executar → diagnosticar B* poria o `p` de A no rodapé
    da nuvem de B **sem nada a acusar** — que é o defeito exacto que
    `GraficosDaJanela.recebe_pca_do_par` já documenta para o par de PCAs.

    ## O que a bandeira AINDA faz, e o que ela não faz

    `interface.fluxo.familias_pedidas` lê-a, então desmarcar a caixa retira a **etapa anunciada**
    da barra de progresso. O cálculo continua a acontecer na rota de diagnóstico. As duas coisas são
    diferentes e o registro tem de as separar: um valor que significa duas coisas.

    **Este portão mede o desfecho em vez de o afirmar.** Duas execuções de `corre_as_analises` que
    diferem **só** na bandeira produzem resultado idêntico: é isso que torna *"a bandeira não
    controla o cálculo"* uma medição e não uma frase. O par causal é a metade que faltaria a uma
    asserção de ausência.

    **Ele reprova no dia em que o par ganhar casa em `corre_as_analises`**, e a resposta certa nesse
    dia é APAGÁ-LO no mesmo commit — junto com uma decisão registrada sobre de qual matriz as
    figuras de PCA falam, que é a pergunta que `test_ktd7_a_pca_orfa_continua_SEM_CHAMADOR_DE_
    PRODUCAO` já nomeia. Afrouxá-lo seria o portão a sobreviver ao referente.
    """
    from dataclasses import fields, replace

    from refino.api.rotas import ResultadosDasAnalises, corre_as_analises

    matriz = _matriz_de_grupos(("a",) * 6 + ("b",) * 6)

    # Âncora de não-vacuidade: a bandeira EXISTE e a família é conhecida do fluxo.
    assert PADRAO_DA_ANALISE.permanova is True
    assert "permanova" in {f.name for f in fields(PADRAO_DA_ANALISE)}

    campos = {f.name for f in fields(ResultadosDasAnalises)}
    # A PROPRIEDADE vem primeiro, e o conjunto exacto vem a seguir. O oráculo original
    # era `campos == {"anova"}`, escrito quando havia um campo só: ele codificava a **forma** da
    # instância vista, e as duas raízes da correlação reprovaram-no ao entrar — que nada têm
    # que ver com o que este portão guarda. A propriedade é *"a PERMANOVA não tem casa aqui"*, e é
    # ela que decide se o teste deve morrer. O conjunto exacto fica como âncora de que a leitura
    # não é vácua, e atualizá-lo é trabalho de quem acrescenta uma família.
    assert not any("permanova" in nome for nome in campos), (
        f"`ResultadosDasAnalises` tem {sorted(campos)} e algum campo nomeia a PERMANOVA: o par "
        "ganhou casa aqui. APAGUE este teste no mesmo commit e registe de qual matriz as figuras "
        "de PCA falam."
    )
    # `random_forest`, `svm` e `rede` entraram depois, e a âncora foi atualizada a cada vez — que
    # é o que a mensagem abaixo manda fazer. Ela é uma bijeção
    # exata de propósito: um conjunto de pertinência deixaria uma família nova entrar sem que
    # ninguém abrisse este teste, e é aqui que a pergunta *"a PERMANOVA ganhou casa?"* tem de voltar
    # a ser feita. Com a rede ela foi feita e a resposta foi não: a rede entrou com campo próprio e
    # a asserção acima continuou verde, o que é o desfecho que este portão existe para distinguir.
    # E a normalidade entrou com campo próprio, e a pergunta foi feita outra vez: ela
    # não nomeia a PERMANOVA, e a asserção acima continuou verde.
    assert campos == {
        "anova",
        "correlacao",
        "busca_por_padrao",
        "kmeans",
        "som",
        "rede",
        "random_forest",
        "svm",
        "normalidade",
    }, (
        f"`ResultadosDasAnalises` tem {sorted(campos)}. A asserção acima é a que guarda a "
        "PERMANOVA; esta é a âncora de que a leitura dos campos não é vácua, e uma família nova "
        "atualiza-a."
    )

    # **O par causal: a bandeira MOVIDA, e o resultado não se move.**
    ligada = corre_as_analises(
        replace(PADRAO_DA_ANALISE, permanova=True),
        matriz=matriz,
        imputadas=_sem_imputadas(matriz),
        seed=42,
        marca=lambda: None,
        cancelado=lambda: False,
    )
    desligada = corre_as_analises(
        replace(PADRAO_DA_ANALISE, permanova=False),
        matriz=matriz,
        imputadas=_sem_imputadas(matriz),
        seed=42,
        marca=lambda: None,
        cancelado=lambda: False,
    )
    assert ligada == desligada, (
        "a bandeira `permanova` passou a mover o resultado de `corre_as_analises`: o par ganhou "
        "casa na rota de execução, e este portão tem de ser apagado com a decisão ao lado."
    )

    # E a âncora do que ela AINDA faz: a etapa anunciada depende dela.
    de_familias = {f.campo for f in familias_pedidas(replace(PADRAO_DA_ANALISE, permanova=True))}
    sem_familias = {f.campo for f in familias_pedidas(replace(PADRAO_DA_ANALISE, permanova=False))}
    assert "permanova" in de_familias and "permanova" not in sem_familias, (
        "a bandeira deixou de controlar a etapa anunciada também: aí ela não controla NADA, e a "
        "caixa da tela é um controle morto"
    )


# As features sem RSD de QC sob `outlier_removed`, por modo. Observação sobre estes
# quatro arquivos e não invariante, e é por isso que o teste corre sobre eles e nunca
# sobre uma factory.
INDEFINIDAS_POR_MODO: dict[str, int] = {
    "pos_hilic": 234,
    "neg_hilic": 212,
    "pos_c18": 60,
    "neg_c18": 33,
}


def test_d567_o_diagnostico_traz_o_violino_da_INGERIDA_e_as_indefinidas_saem_dele(  # noqa: N802
    cliente: TestClient,
) -> None:
    """A rota devolve o RSD por feature da matriz ingerida — o violino antes da correção —, e
    a contagem de indefinidas deixou de chamar `qc_rsd` por si.

    O comportamento que a troca declara preservar é reafirmado por teste e não pela leitura do
    diff, de dois lados: as quatro contagens de `INDEFINIDAS_POR_MODO`, e a conta antiga refeita
    aqui sobre a matriz lida do disco — o outro caminho como oráculo. **E a carga declara-se:**
    estágio «ingerida», corretor nulo, a variante da contagem e o total da resposta.
    """
    assert MATRIZ_DA_LEITURA == ESTAGIO_DA_INGERIDA, (
        "o rodapé do violino reconhece um estágio e a rota publica outro: a ingerida sairia com o "
        "recuo de corretor não declarado"
    )
    for mode in MODES:
        corpo = cliente.post(
            "/matrizes/diagnosticar", files=_reais(mode), data={"mode": mode}
        ).json()
        carga = corpo["rsd_por_feature"]
        indefinidas = corpo["n_features_rsd_undefined"]
        assert indefinidas == INDEFINIDAS_POR_MODO[mode], (mode, indefinidas)

        matriz = load_matrix(measurements_path(mode), sample_list_path(mode), mode=mode)
        pela_conta_antiga = int((~np.isfinite(qc_rsd(matriz, variant=corpo["rsd_variant"]))).sum())
        assert indefinidas == pela_conta_antiga, (mode, indefinidas, pela_conta_antiga)

        assert (carga["matriz"], carga["corretor"]) == (ESTAGIO_DA_INGERIDA, None), mode
        assert carga["rsd_variant"] == corpo["rsd_variant"], mode
        assert carga["n_features"] == corpo["n_features"], mode
        assert len(carga["rsd_dos_qcs"]) == carga["n_features_finitas_dos_qcs"], mode
        assert len(carga["rsd_das_amostras"]) == carga["n_features_finitas_das_amostras"], mode
        assert carga["n_injecoes_dos_qcs"] == int(matriz.qc_mask.sum()), mode


def test_diagnosticar_os_limiares_viajam_na_requisicao(cliente: TestClient) -> None:
    """`DiagnosisPolicy` não tem padrão, e quem tem os dados calibra.

    A asserção é que mudar um limiar **muda o resultado** — se não mudasse, o parâmetro
    seria decorativo e a calibração não teria canal por onde chegar.

    O modo é `neg_c18` por MEDIÇÃO, e o recorte vai declarado. Medido sobre os quatro arquivos
    reais: só o `neg_c18` inverte — `clean` com o limiar frouxo e `gradual_drift` com o apertado.
    Nos outros três a cascata é decidida
    por regra **anterior** e o limiar não é observável na saída: `pos_hilic` e `pos_c18`
    saem `erratic` e o `neg_hilic` sai `step_change` sob qualquer valor.

    Isso é **observação sobre estes quatro arquivos**, não invariante — e escolher
    `pos_hilic` "porque é o maior" faria o teste reprovar por motivo que não é defeito. Foi
    o que aconteceu na primeira redação.
    """

    def diagnostica(rsd_max: float) -> str:
        return cliente.post(
            "/matrizes/diagnosticar",
            files=_reais("neg_c18"),
            data={"mode": "neg_c18", "rsd_mediana_max": rsd_max},
        ).json()["defect_type"]

    assert diagnostica(0.99) == "clean"
    assert diagnostica(0.001) == "gradual_drift", (
        "o limiar não alterou o diagnóstico: o parâmetro chegou e foi ignorado"
    )


def test_diagnosticar_recusa_matriz_invalida_com_ponteiro_para_validar(
    cliente: TestClient,
) -> None:
    """Aqui a recusa **é** erro de protocolo, ao contrário de `/validar`.

    A diferença é o contrato da rota: `/validar` existe para reportar problemas, então
    problema é a resposta; `/diagnosticar` promete um diagnóstico, e não há diagnóstico de
    matriz recusada. A mensagem aponta para a rota que dá a lista.
    """
    especificacao = inject("RAGGED_ROWS")
    medidas, lista = render_pair(especificacao)

    resposta = cliente.post(
        "/matrizes/diagnosticar",
        files={
            "measurements": ("m.csv", medidas.encode(), "text/csv"),
            "sample_list": ("s.csv", lista.encode(), "text/csv"),
        },
        data={"mode": "tiny"},
    )

    assert resposta.status_code == 422
    assert "validar" in resposta.text, "a mensagem não aponta para a rota que lista os problemas"


def test_diagnosticar_traduz_limiar_fora_de_faixa_em_422(cliente: TestClient) -> None:
    """A guarda do limiar chega ao cliente como erro de domínio, não como 500.

    RSD neste projeto é **fração**: `20` em vez de `0.20` faria a regra 1 nunca casar, a
    cascata escorregar para a regra 2, e um modo sem degrau sair `step_change`. É o valor
    que um cliente digita naturalmente, e a mensagem que explica isso só serve se ele a
    receber — num 500 ela fica num log que ele não lê.
    """
    resposta = cliente.post(
        "/matrizes/diagnosticar",
        files=_reais("pos_c18"),
        data={"mode": "pos_c18", "rsd_mediana_max": 20},
    )

    assert resposta.status_code == 422, "limiar em percentual deveria virar erro de domínio"
    assert (
        "fração" in resposta.text or "FRAÇÃO" in resposta.text or "fracao" in resposta.text.lower()
    )


# ---------------------------------------------------------------------------
# O inventário de grupos traz a CONTAGEM
# ---------------------------------------------------------------------------

# **As quatro contagens do estudo real, medidas em 14/09/2026 e não copiadas de lugar nenhum.**
#
# População: as amostras **não-QC** dos quatro `measurements_*.csv` do laboratório, pela linha
# `LABEL` — 61 das 68 colunas de cada arquivo, com os 7 QCs fora. Os quatro modos dão o mesmo
# inventário, e isso é observação sobre aqueles quatro arquivos e não invariante:
# nada no desenho fixa quantos grupos há nem o tamanho de cada um. Por isso o teste corre sobre os
# arquivos reais, e não sobre matriz de factory — uma factory que reproduzisse o número estaria a
# testar a factory.
CONTAGENS_DO_ESTUDO: dict[str, int] = {
    "ATLL": 13,
    "Asymptomatic": 18,
    "HAMTSP": 14,
    "NoHTLV": 16,
}
QCS_DO_ESTUDO = 7


def test_o_inventario_traz_a_contagem_de_cada_grupo(cliente: TestClient) -> None:
    """O inventário diz **quantas amostras** cada grupo tem, nos quatro modos reais.

    **Por que a contagem e não só o nome.** O profissional escolhe o par do PLS-DA — e escolherá
    as famílias de análise — sem nada na interface a dizer o tamanho dos grupos, e é o tamanho que
    diz se a pergunta tem poder: um contraste contra `ATLL` (13) não é o mesmo que contra
    `Asymptomatic` (18).

    **Sobre os arquivos REAIS, com pulo declarado quando faltam.** `_reais` passa pelos
    resolvedores do `conftest`, que fazem `pytest.skip` nomeando o arquivo ausente. Dado
    fabricado aqui destruiria o valor da comparação.

    A não-vacuidade vai nesta função e ANTES: `set(...) == set(...)` sobre dois vazios
    passa, e a comparação de contagens abaixo herdaria a vacuidade sem nada a acusar.

    **E o QC entra na asserção pelo TOTAL e não só pela ausência do nome.** Afirmar *"«QC» não
    está na lista"* é asserção de ausência; somar 61 e conferir contra `n_samples - 7` é asserção
    de presença sobre a mesma exclusão, e essa não passa por vacuidade.
    """
    for mode in MODES:
        resposta = cliente.post("/matrizes/diagnosticar", files=_reais(mode), data={"mode": mode})

        assert resposta.status_code == 200, (mode, resposta.text)
        corpo = resposta.json()
        inventario = corpo["grupos_disponiveis"]
        assert inventario, f"{mode}: o inventário veio vazio — as asserções abaixo seriam vácuas"

        medido = {entrada["nome"]: entrada["n"] for entrada in inventario}
        assert medido == CONTAGENS_DO_ESTUDO, (
            f"{mode}: o inventário devolveu {medido} e o medido sobre os arquivos do laboratório "
            f"é {CONTAGENS_DO_ESTUDO}. Se a implementação está certa e o número não bate, pare e "
            "relate — é achado, não bug (regra 2 do README)"
        )
        assert sum(medido.values()) == corpo["n_samples"] - QCS_DO_ESTUDO, (
            f"{mode}: o inventário soma {sum(medido.values())} amostras, e a matriz tem "
            f"{corpo['n_samples']} com {QCS_DO_ESTUDO} QCs. O QC é régua e não grupo: ou ele "
            "entrou no inventário, ou alguma outra amostra ficou de fora dele"
        )


def test_a_recusa_da_rota_NOMEIA_a_contagem_e_nao_so_o_nome(cliente: TestClient) -> None:  # noqa: N802
    """A mensagem de `_grupos_ou_422` é o segundo consumidor da lista.

    **O par causal é o texto, e ele separa:** a mensagem antes desta unidade dizia
    `Disponíveis: ['ATLL', 'Asymptomatic', ...]` e nenhum dígito de contagem aparecia nela. A
    asserção afirma que cada nome aparece **com o seu `n` ao lado**, na grafia `Nome (n)` — que é
    a mesma que a linha do painel do pipeline usa.

    **A âncora de não-vacuidade é a própria recusa**: sem o `422`, `detalhe` seria o corpo de
    sucesso e as asserções de pertença não diriam nada sobre a mensagem que interessa.

    **Sobre os arquivos REAIS e não sobre a `tiny_clean`, e a escolha é a planta que separa.** Na
    `tiny_clean` os quatro grupos têm **1** amostra cada, então uma implementação que escrevesse o
    NÚMERO DE GRUPOS, ou o índice, ou qualquer constante `1` ao lado do nome passaria. Aqui as
    quatro contagens são distintas — 13, 18, 14 e 16 —, e só a contagem certa casa.

    **`/execucoes` e não `/matrizes/diagnosticar`, porque é ali que a guarda vive**: quem chama
    `_grupos_ou_422` são `POST /execucoes` e `POST /multiversos`, e a recusa acontece **no
    disparo** — nenhuma thread longa nasce.
    """
    resposta = cliente.post(
        "/execucoes",
        files=_reais("pos_c18"),
        data={
            "mode": "pos_c18",
            "estudo": "e",
            "correction": "none",
            "n_batches": 1,
            "rsd_variant": "plain",
            "imputation": "median",
            "sample_normalization": "none",
            "transformation": "none",
            "scaling": "none",
            "apresentacao": "exibida",
            "group_a": "Inexistente",
            "group_b": "ATLL",
        },
    )

    assert resposta.status_code == 422, resposta.text
    detalhe = resposta.json()["detail"]
    assert "não existe nesta matriz" in detalhe, detalhe
    for nome, n in CONTAGENS_DO_ESTUDO.items():
        assert f"{nome} ({n})" in detalhe, (
            f"a recusa não nomeia {nome!r} com a contagem {n}: {detalhe!r}"
        )


# ---------------------------------------------------------------------------
# A chave de invalidação e o resumo das distribuições atravessam o fio
# ---------------------------------------------------------------------------
#
# **`/execucoes` é a única rota desta árvore que devolve `ExecucaoResposta`, e ela exige banco.**
# O SQLite de arquivo com `prepara_sqlite` mais `METADATA.create_all` dispensa o subprocesso do
# `alembic` — é o mesmo arranjo de
# `tests/test_desktop_qualidade.py::test_u7_o_antes_e_depois_da_rota_real_com_um_duplo_que_separa`,
# e é por isso que os pares causais abaixo correm a rota de verdade em vez de afirmar sobre o
# helper: a superfície que importa é a resposta, não a função que compõe a chave.
#
# **O modo é `neg_c18` porque é o mais barato dos quatro** — 2.155 features contra as 6.035 do
# `pos_hilic` —, e a correção é `none` porque é O(1) e determinística. As duas escolhas são sobre
# custo e não sobre o que se mede: a chave não olha para o valor da correção.

CAMPOS_MINIMOS_DA_EXECUCAO: dict[str, object] = {
    "correction": "none",
    "n_batches": 1,
    "rsd_variant": "plain",
    "imputation": "none",
    "sample_normalization": "none",
    "transformation": "none",
    "scaling": "none",
    "apresentacao": "exibida",
}


def _matriz_sintetica(n_features: int, n_samples: int = 8) -> Any:
    """Matriz sintética para as propriedades da FUNÇÃO — nunca para contagem observada.

    O que os testes em processo afirmam é propriedade de `_resumo_das_distribuicoes` — o teto, o
    determinismo do sorteio, a coluna toda-`nan` — e não número medido nos arquivos do laboratório.
    Usar os 6,08 MB reais aqui mediria a mesma coisa mais devagar.
    """
    import numpy as np

    from refino.domain.matrix import RefinoMatrix, SampleMetadata

    rng = np.random.default_rng(7)
    return RefinoMatrix(
        values=rng.uniform(1e3, 1e6, size=(n_samples, n_features)),
        feature_names=tuple(f"1.0_{100 + j}m/z" for j in range(n_features)),
        samples=tuple(
            SampleMetadata(
                name=f"S{i}",
                sample_type="qc" if i % 4 == 0 else "sample",
                injection_order=i + 1,
                batch="A",
                group="QC" if i % 4 == 0 else "G1",
            )
            for i in range(n_samples)
        ),
        mode="tiny",
    )


def _banco_de_arquivo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Um SQLite de arquivo pronto para `/execucoes`, sem `alembic`."""
    from sqlalchemy import create_engine

    from refino.persistence.config import VARIAVEL_DA_URL, prepara_sqlite
    from refino.persistence.schema import METADATA

    url = f"sqlite:///{tmp_path / 'refino.db'}"
    monkeypatch.delenv("REFINO_DB_HOST", raising=False)
    monkeypatch.setenv(VARIAVEL_DA_URL, url)
    prepara_sqlite(url)
    METADATA.create_all(create_engine(url))


def _executa_ate_o_fim(cliente: TestClient, modo: str, **campos: object) -> dict[str, Any]:
    """Dispara `/execucoes`, espera o `202` virar terminal, e devolve o corpo da reclamação.

    Laço com teto e não `sleep` calibrado, no espelho de `_reclama_ate_terminar` de
    `tests/test_api_execucoes.py`: o teto existe para o teste reprovar em vez de pendurar quando o
    trabalho morre sem estado terminal.
    """
    disparo = cliente.post(
        "/execucoes",
        files=_reais(modo),
        data={"mode": modo, **CAMPOS_MINIMOS_DA_EXECUCAO, **campos},
    )
    assert disparo.status_code == 202, disparo.text[:600]
    identificador = disparo.json()["identificador"]
    for _ in range(2000):
        reclamada = cliente.get(f"/trabalhos/{identificador}")
        if reclamada.status_code != 202:
            assert reclamada.status_code == 200, reclamada.text[:600]
            corpo: dict[str, Any] = reclamada.json()
            return corpo
        threading.Event().wait(0.05)
    raise AssertionError(f"o trabalho {identificador} não terminou")


@pytest.mark.integration
def test_u11_a_chave_de_invalidacao_separa_duas_execucoes_que_so_diferem_na_semente(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """O par causal da chave, plantado no eixo que o atalho de três eixos NÃO cobre.

    O atalho recusado em `src/refino/api/reuso.py:174-176` é `{modo, método, bytes}`. Plantar no
    modo ou no método passaria por coincidência — os dois estão *dentro* daquele atalho, e o teste
    diria «separa» sobre uma chave que não separa nada do que interessa.

    **A semente é a planta mais forte que este corpus admite, e a razão é aritmética:** com
    `correction=none` as duas execuções produzem matriz **byte a byte idêntica**, então o digest da
    matriz é o mesmo, o método é o mesmo e o modo é o mesmo. Se as chaves divergirem, foi a
    semente — e mais nada — que as separou.

    **A direção da igualdade é a metade que impede o portão de ser satisfeito por um `uuid4()`.**
    Uma chave nova a cada resposta separaria as duas execuções acima e destruiria o mecanismo: toda
    figura ficaria obsoleta no instante seguinte.
    """
    _banco_de_arquivo(tmp_path, monkeypatch)
    cliente = TestClient(build_app())
    modo = "neg_c18"

    a = _executa_ate_o_fim(cliente, modo, estudo="u11_semente", seed=20260910)
    b = _executa_ate_o_fim(cliente, modo, estudo="u11_semente", seed=20260911)
    repetida = _executa_ate_o_fim(cliente, modo, estudo="u11_semente", seed=20260910)

    assert a["chave_de_invalidacao"], "âncora: a chave veio vazia, e as comparações seriam vácuas"
    assert a["chave_de_invalidacao"] != b["chave_de_invalidacao"], (
        "duas execuções que diferem SÓ na semente receberam a mesma chave de invalidação — a "
        "chave não cobre a semente, e a figura de uma sobreviveria à outra"
    )
    assert a["chave_de_invalidacao"] == repetida["chave_de_invalidacao"], (
        "duas execuções idênticas receberam chaves diferentes — a chave não é função dos "
        "argumentos, e toda figura ficaria obsoleta no instante seguinte"
    )


def test_u11_o_par_causal_da_ausencia_duas_respostas_sem_chave_LEEM_COMO_DIVERGENCIA() -> None:  # noqa: N802
    """A segunda metade da chave de invalidação, e ela não é verificável do lado do servidor.

    `PainelDeArtefatos.define_entrada` (`artefatos.py:491`) recebe `str` não opcional e abre com
    **retorno antecipado na igualdade**. Um marcador constante para *"não veio chave"* faria duas
    ausências compararem iguais, o retorno antecipado dispararia, e a figura obsoleta sobreviveria —
    o desfecho exacto que a chave existe para evitar.

    **As três direções, e a terceira é a que a torna par e não asserção solta:**

    1. sem campo, duas vezes → dois marcadores **diferentes**;
    2. erro depois de a figura ter sido desenhada → marcador **diferente** da chave desenhada;
    3. com chave, duas vezes → o **mesmo** valor. Sem esta, um `chave_de_invalidacao` que
       devolvesse sempre `uuid4()` passaria nas duas primeiras e destruiria o mecanismo.

    **A âncora de forma existe porque um marcador que colidisse com uma chave real seria pior que
    nenhum:** `_chave_do_trabalho` devolve 64 caracteres de hexadecimal, e `:` não é um deles.
    """
    from refino.interface.cliente import PREFIXO_DE_DIVERGENCIA, chave_de_invalidacao

    def resposta(codigo: int, corpo: object) -> Resposta:
        return Resposta(codigo=codigo, ok=codigo == 200, causa=None, mensagem="", corpo=corpo)

    chave_real = "0" * 64
    desenhada = chave_de_invalidacao(resposta(200, {"chave_de_invalidacao": chave_real}))
    ausente_1 = chave_de_invalidacao(resposta(200, {"matrix_id": 1}))
    ausente_2 = chave_de_invalidacao(resposta(200, {"matrix_id": 1}))
    depois_do_erro = chave_de_invalidacao(resposta(500, {"detail": "boom"}))
    em_curso = chave_de_invalidacao(resposta(202, {"estado": "em_curso"}))

    assert desenhada == chave_real, "a chave presente não atravessou intacta"
    assert desenhada == chave_de_invalidacao(resposta(200, {"chave_de_invalidacao": chave_real})), (
        "duas leituras da MESMA chave divergiram — o extractor está a inventar identidade, e toda "
        "figura ficaria obsoleta no instante seguinte"
    )
    assert ausente_1 != ausente_2, (
        "duas ausências de chave compararam IGUAIS — o retorno antecipado de `define_entrada` "
        "dispararia e a figura obsoleta sobreviveria"
    )
    assert depois_do_erro != desenhada and em_curso != desenhada, (
        "um erro e um `202` leram como a chave da figura desenhada — a figura sobreviveria ao "
        "erro, que é o desfecho que KTD13 nomeia"
    )

    # **O MOTIVO e não o marcador inteiro, e a distinção é o que salva esta asserção da vacuidade.**
    # `depois_do_erro != em_curso` seria satisfeito pelo *nonce* sozinho: dois marcadores diferentes
    # com o mesmo motivo passariam, e a planta que colapsa os quatro desfechos num só motivo ficaria
    # invisível. O que se exige é que os FATOS se distingam, e o fato é o motivo.
    def motivo(marcador: str) -> str:
        return marcador.rsplit(":", 1)[0]

    motivos = {
        motivo(ausente_1),
        motivo(depois_do_erro),
        motivo(em_curso),
    }
    assert len(motivos) == 3, (
        f"os três desfechos de ausência colapsaram em {sorted(motivos)} — *«não veio campo»*, "
        "*«houve erro»* e *«ainda está a correr»* são fatos diferentes, e o painel não consegue "
        "escrever a frase de ausência declarada de R4b a partir de um motivo só"
    )
    assert motivo(ausente_1) == motivo(ausente_2), (
        "duas ausências do mesmo tipo receberam motivos diferentes — o motivo passou a carregar "
        "ruído, e deixou de ser legível"
    )
    for marcador in (ausente_1, ausente_2, depois_do_erro, em_curso):
        assert marcador.startswith(PREFIXO_DE_DIVERGENCIA), marcador
        assert ":" in marcador, (
            f"{marcador!r} não tem `:` — um marcador sem ele poderia colidir com um sha256, e "
            "colisão aqui é a figura obsoleta a passar por corrente"
        )


@pytest.mark.integration
def test_u11_o_resumo_das_distribuicoes_sai_da_rota_sobre_a_populacao_COMUM(  # noqa: N802
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """O resumo antes/depois na rota real, com a população declarada.

    **As caixas são pareadas por NOME e não por posição**, e é a asserção que separa esta
    implementação da natural: o pipeline remove features, então a coluna `i` do «depois» não é a
    coluna `i` do «antes». Casar por posição daria caixas plausíveis com o conteúdo de outra
    feature.

    **A âncora de não-vacuidade vem primeiro.** Sem ela, um resumo com zero caixas passaria em
    todas as comparações abaixo por vacuidade.

    **`missing_threshold` planta a REMOÇÃO, e sem ela a asserção da interseção seria vácua.** Com o
    pipeline todo em `none` nada é removido, `n_features_comparadas == n_features_antes`, e a
    distinção entre «a população comum» e «a população inteira» não teria sobre o que cair.
    """
    _banco_de_arquivo(tmp_path, monkeypatch)
    cliente = TestClient(build_app())
    modo = "neg_c18"

    corpo = _executa_ate_o_fim(
        cliente, modo, estudo="u11_distribuicoes", missing_threshold=0.2, seed=4242
    )
    resumo = corpo["distribuicoes"]

    assert resumo["n_caixas"] > 0, "âncora: zero caixas tornaria vácuo tudo o que vem abaixo"
    assert resumo["matriz_antes"] == "ingerida" and resumo["matriz_depois"] == "processada", (
        f"os dois eixos não declaram a matriz de onde vieram (R5): {resumo['matriz_antes']!r} e "
        f"{resumo['matriz_depois']!r}"
    )

    # O recorte: quantas de quantas, e com que semente.
    assert resumo["n_caixas"] == resumo["teto_de_caixas"] == 50, resumo["n_caixas"]
    assert resumo["amostrada"] is True, "50 de milhares e o resumo diz que não amostrou"
    assert resumo["seed"] == 4242, "a semente do recorte não é a da execução"

    # A planta da remoção: sem ela a interseção seria a matriz inteira e a asserção seria vácua.
    assert resumo["n_features_depois"] < resumo["n_features_antes"], (
        "o filtro de ausentes não removeu feature nenhuma — a asserção da interseção abaixo cai "
        "sobre dado em que as duas quantidades NÃO diferem"
    )
    assert resumo["n_features_comparadas"] == resumo["n_features_depois"], (
        f"a população comum ({resumo['n_features_comparadas']}) não é a do «depois» "
        f"({resumo['n_features_depois']}) — o pipeline remove features e não amostras, então o "
        "«depois» é subconjunto do «antes»"
    )

    # As duas metades são pareadas POR NOME, nas duas direções.
    nomes_antes = [c["feature"] for c in resumo["caixas_antes"]]
    nomes_depois = [c["feature"] for c in resumo["caixas_depois"]]
    assert nomes_antes == nomes_depois, (
        "as caixas de antes e depois não descrevem as mesmas features — a figura poria lado a "
        "lado duas amostras de populações diferentes"
    )
    assert len(set(nomes_antes)) == len(nomes_antes), "uma feature foi sorteada duas vezes"

    # A caixa de Tukey, ordenada — e não a coluna inteira.
    for caixa in resumo["caixas_antes"]:
        assert (
            caixa["bigode_inferior"]
            <= caixa["q1"]
            <= caixa["mediana"]
            <= caixa["q3"]
            <= caixa["bigode_superior"]
        ), caixa

    # E o traço dos pontos que o bigode esconde EXISTE no dado real, em vez de ser um campo que
    # só se enche em fixture. Sem esta âncora, `n_discrepantes` podia sair zero em toda parte e
    # nada acusaria — é a asserção de presença ao lado da de ausência.
    contagens = [c["n_discrepantes"] for c in resumo["caixas_antes"]]
    assert any(n is not None and n > 0 for n in contagens), (
        "nenhuma das 50 features do `neg_c18` tem ponto fora dos bigodes de Tukey — ou o dado "
        f"real não tem discrepante nenhum, ou a contagem não está a ser feita: {contagens}"
    )

    # A legenda — «N de M features, as que sobreviveram ao corte» — tem as DUAS pontas aqui.
    assert 0 < resumo["n_features_comparadas"] < resumo["n_features_antes"], (
        f"a legenda do corte não tem as duas pontas: N={resumo['n_features_comparadas']}, "
        f"M={resumo['n_features_antes']}. Derivar `M` de outro lugar do corpo seria a segunda "
        "fonte de uma verdade só"
    )

    # A densidade é sobre MÉDIAS de eixo, e o comprimento de cada vetor o prova.
    assert len(resumo["medias_por_feature_antes"]) == resumo["n_features_comparadas"]
    assert len(resumo["medias_por_feature_depois"]) == resumo["n_features_comparadas"]
    assert len(resumo["medias_por_amostra_antes"]) == resumo["n_amostras"]
    assert len(resumo["medias_por_amostra_depois"]) == resumo["n_amostras"]
    assert resumo["n_amostras"] == len(resumo["medias_por_amostra_depois"]), (
        "o pipeline removeu AMOSTRAS — cinco etapas removem features e nenhuma remove amostras, e "
        "esta asserção é o que fixa essa premissa do helper"
    )

    # ------------------------------------------------------------------
    # O SEGUNDO EIXO de caixas — a `Sample View`, que é a TRANSPOSTA
    # ------------------------------------------------------------------
    #
    # **A âncora da transposta vem primeiro, e sem ela tudo o que segue é vácuo.** As duas metades
    # saem da mesma matriz por eixos opostos; se dessem os mesmos números, nenhuma asserção
    # distinguiria a figura nova de uma segunda cópia da irmã.
    por_amostra = resumo["caixas_por_amostra_antes"]
    assert por_amostra, "âncora: zero caixas por amostra tornaria vácuo tudo o que vem abaixo"
    quinteto = ("bigode_inferior", "q1", "mediana", "q3", "bigode_superior")
    das_amostras = {tuple(c[k] for k in quinteto) for c in por_amostra}
    das_features = {tuple(c[k] for k in quinteto) for c in resumo["caixas_antes"]}
    assert not (das_amostras & das_features), (
        "há caixas numericamente idênticas nos dois eixos do `neg_c18` — a `Sample View` não se "
        "distinguiria de uma segunda `Feature View`"
    )

    # O teto vale nos DOIS eixos, e é o mesmo número — o do MetaboAnalyst, que diz
    # «at most 50 features/samples». Nos quatro arquivos o eixo das amostras tem 68.
    assert resumo["n_amostras_comparadas"] == resumo["n_amostras"], (
        "as amostras comuns às duas matrizes não são todas as da matriz ingerida — o pipeline "
        "removeu amostra, e a premissa de `_medias` deixou de valer"
    )
    assert resumo["n_amostras_comparadas"] > resumo["teto_de_caixas"], (
        "âncora: este corpus tem de ter mais amostras que o teto, ou o par causal abaixo não cai "
        f"sobre dado que separe ({resumo['n_amostras_comparadas']} amostras)"
    )
    assert resumo["n_caixas_por_amostra"] == resumo["teto_de_caixas"] == 50
    assert resumo["amostrada_por_amostra"] is True, "68 amostras e o resumo diz que não amostrou"
    assert len(por_amostra) == resumo["n_caixas_por_amostra"]

    # As duas metades são pareadas POR NOME também neste eixo, e sem repetição.
    nomes_a = [c["amostra"] for c in por_amostra]
    nomes_d = [c["amostra"] for c in resumo["caixas_por_amostra_depois"]]
    assert nomes_a == nomes_d, (
        "as caixas por amostra de antes e depois não descrevem as mesmas amostras — a figura poria "
        "lado a lado o rótulo de uma injeção sobre a distribuição de outra"
    )
    assert len(set(nomes_a)) == len(nomes_a), "uma amostra foi sorteada duas vezes"
    # **E o nome é de AMOSTRA e não de feature**, que é a metade que apanharia o eixo trocado: os
    # dois espaços de nome deste corpus são disjuntos — `RT_mz` contra o rótulo da injeção —, então
    # um `_caixas_por_amostra` que lesse `feature_names` cairia aqui.
    assert not (set(nomes_a) & set(nomes_antes)), (
        f"uma caixa por amostra leva nome de feature: {sorted(set(nomes_a) & set(nomes_antes))[:3]}"
    )
    assert all(nome.strip() for nome in nomes_a), "caixa por amostra com nome em branco"

    # A caixa de Tukey, ordenada — o mesmo contrato da irmã, porque é a mesma redução.
    for caixa in por_amostra:
        assert (
            caixa["bigode_inferior"]
            <= caixa["q1"]
            <= caixa["mediana"]
            <= caixa["q3"]
            <= caixa["bigode_superior"]
        ), caixa
    contagens_de_amostra = [c["n_discrepantes"] for c in por_amostra]
    assert any(n is not None and n > 0 for n in contagens_de_amostra), (
        "nenhuma das 50 amostras do `neg_c18` tem ponto fora dos bigodes — ou a contagem não está "
        f"a ser feita neste eixo: {contagens_de_amostra}"
    )


def test_u11_o_recorte_declara_QUE_NAO_AMOSTROU_quando_cabe_tudo() -> None:  # noqa: N802
    """**A borda de menos de 50 features, e o campo `amostrada` é o que a torna legível.**

    O proxy óbvio — `n_caixas < n_features_comparadas` — mente exactamente no caso de **50**: não
    houve sorteio, e ele diria o mesmo que diz para 6.035. A figura escreve *"as 12 features"* ou
    *"50 de 2.155, sorteadas com a semente 42"*, e são frases diferentes.

    Em processo e não pela rota, porque a planta é argumento da função sob teste: não há
    invocação intermediária que possa não a receber.
    """
    import numpy as np

    from refino.api.rotas import _resumo_das_distribuicoes

    pequena = _matriz_sintetica(n_features=12)
    resumo = _resumo_das_distribuicoes(pequena, pequena, seed=99)

    assert resumo.amostrada is False, "12 features cabem inteiras e o resumo diz que amostrou"
    assert resumo.n_caixas == 12 == resumo.n_features_comparadas
    assert [c.feature for c in resumo.caixas_antes] == list(pequena.feature_names)

    # O par causal do teto: a MESMA função, com features que passam de 50, amostra e diz que sim.
    grande = _matriz_sintetica(n_features=137)
    amostrado = _resumo_das_distribuicoes(grande, grande, seed=99)
    assert amostrado.amostrada is True and amostrado.n_caixas == 50, (
        f"137 features deveriam ser amostradas em 50: {amostrado.n_caixas}, "
        f"amostrada={amostrado.amostrada}"
    )

    # E o sorteio é DETERMINÍSTICO na semente.
    assert [c.feature for c in amostrado.caixas_antes] == [
        c.feature for c in _resumo_das_distribuicoes(grande, grande, seed=99).caixas_antes
    ], "duas chamadas com a mesma semente sortearam features diferentes"
    assert [c.feature for c in amostrado.caixas_antes] != [
        c.feature for c in _resumo_das_distribuicoes(grande, grande, seed=100).caixas_antes
    ], "duas sementes diferentes sortearam as MESMAS features — a semente não chega ao sorteio"

    # E a coluna toda-`nan` continua a existir como caixa de `nan`, em vez de sumir.
    valores = grande.values.copy()
    valores[:, 0] = np.nan
    com_buraco = grande.with_values(valores, "planta do teste da U11")
    resumo_com_buraco = _resumo_das_distribuicoes(com_buraco, com_buraco, seed=99)
    assert resumo_com_buraco.n_caixas == 50, "a coluna toda-`nan` fez a caixa desaparecer"

    # E ela sai como «não avaliável» e não como «zero discrepantes». A planta vai na
    # matriz PEQUENA de propósito: com 137 features e teto de 50 a coluna 0 pode não ser sorteada,
    # e a asserção cairia sobre a feature errada ou sobre lista vazia, calada nas duas.
    valores_pequena = pequena.values.copy()
    valores_pequena[:, 0] = np.nan
    pequena_com_buraco = pequena.with_values(valores_pequena, "planta do teste da U11")
    caixas = _resumo_das_distribuicoes(pequena_com_buraco, pequena_com_buraco, seed=99).caixas_antes
    assert caixas[0].feature == pequena.feature_names[0], "a caixa 0 não é a coluna plantada"
    assert caixas[0].n_discrepantes is None, (
        f"a coluna sem valor nenhum reportou {caixas[0].n_discrepantes} discrepantes — *«nada "
        "ficou fora»* e *«não havia sobre o que decidir»* colapsaram num número só. "
        "É a divergência deliberada com o `boxplot_stats`, que devolve `fliers=[]` nos dois"
    )
    assert caixas[1].n_discrepantes is not None, (
        "a coluna VIZINHA, que tem os oito valores, também devolveu `None` — sem esta metade a "
        "asserção de cima passaria numa implementação que devolvesse `None` para toda a gente. "
        "O que se afirma aqui é «é avaliável», e não o VALOR: a contagem de uma coluna uniforme "
        "de oito pontos não é propriedade do desenho, e fixá-la seria número sem origem"
    )


def test_u11_o_teto_e_a_semente_valem_nos_DOIS_eixos_e_decidem_SEPARADO() -> None:  # noqa: N802
    """**O teto e o determinismo no eixo das amostras, e a independência entre os dois eixos.**

    O teto é o do MetaboAnalyst — *"at most 50 features/samples"* — e vale para os dois lados; a
    semente é a mesma. O que **não** é o mesmo é a decisão: uma matriz larga e rasa amostra num
    eixo e não no outro, e uma bandeira só faria o rodapé declarar um recorte que não se aplicou.

    Em processo e não pela rota: a planta é argumento da função sob teste.
    """
    from refino.api.rotas import _resumo_das_distribuicoes

    # Larga e rasa: 137 features amostram, 8 amostras não. É a planta que separa as duas bandeiras.
    larga = _matriz_sintetica(n_features=137, n_samples=8)
    resumo = _resumo_das_distribuicoes(larga, larga, seed=99)
    assert resumo.amostrada is True and resumo.n_caixas == 50
    assert resumo.amostrada_por_amostra is False, "8 amostras cabem inteiras e o resumo amostrou"
    assert resumo.n_caixas_por_amostra == 8 == resumo.n_amostras_comparadas
    assert [c.amostra for c in resumo.caixas_por_amostra_antes] == list(larga.sample_names)

    # O par causal do teto NESTE eixo: a MESMA função, com amostras que passam de 50.
    alta = _matriz_sintetica(n_features=12, n_samples=137)
    fundo = _resumo_das_distribuicoes(alta, alta, seed=99)
    assert fundo.amostrada is False, "12 features cabem inteiras e o resumo amostrou"
    assert fundo.amostrada_por_amostra is True and fundo.n_caixas_por_amostra == 50, (
        f"137 amostras deviam ser amostradas em 50: {fundo.n_caixas_por_amostra}, "
        f"amostrada_por_amostra={fundo.amostrada_por_amostra}"
    )

    # E o sorteio deste eixo é DETERMINÍSTICO na semente, nas duas direções.
    assert [c.amostra for c in fundo.caixas_por_amostra_antes] == [
        c.amostra for c in _resumo_das_distribuicoes(alta, alta, seed=99).caixas_por_amostra_antes
    ], "duas chamadas com a mesma semente sortearam amostras diferentes"
    assert [c.amostra for c in fundo.caixas_por_amostra_antes] != [
        c.amostra for c in _resumo_das_distribuicoes(alta, alta, seed=100).caixas_por_amostra_antes
    ], "duas sementes diferentes sortearam as MESMAS amostras — a semente não chega ao sorteio"


def test_u11_a_caixa_por_amostra_e_a_TRANSPOSTA_e_nao_a_caixa_por_feature() -> None:  # noqa: N802
    """A âncora da transposta, sobre uma matriz cuja assimetria é PLANTADA.

    A redução é uma só — `_tukey` — e o que muda é o eixo que se lhe dá. Uma implementação que
    passasse a matriz sem transpor devolveria caixas bem formadas, com nome de amostra e conteúdo
    de feature: o rótulo certo sobre a quantidade errada, e a figura sairia bem formada.

    **A planta é uma matriz em que as duas reduções NÃO PODEM coincidir**: cada linha é constante e
    as linhas diferem entre si, então a caixa de cada amostra é um ponto e a de cada feature cobre
    a variação toda. Sobre uma matriz uniforme os dois eixos dariam o mesmo, e o teste mediria nada.
    """
    import numpy as np

    from refino.api.rotas import _caixas, _caixas_por_amostra

    base = _matriz_sintetica(n_features=6, n_samples=4)
    valores = np.array([[10.0] * 6, [20.0] * 6, [30.0] * 6, [40.0] * 6])
    matriz = base.with_values(valores, "planta da transposta")

    colunas = np.arange(matriz.n_features, dtype=np.int64)
    linhas = np.arange(matriz.n_samples, dtype=np.int64)
    por_feature = _caixas(matriz, colunas)
    por_amostra = _caixas_por_amostra(matriz, linhas)

    assert len(por_feature) == 6 and len(por_amostra) == 4, "âncora: uma caixa por unidade do eixo"
    assert [c.amostra for c in por_amostra] == list(matriz.sample_names)
    assert [c.feature for c in por_feature] == list(matriz.feature_names)

    # Cada AMOSTRA é constante: a caixa dela colapsa nos cinco números iguais.
    for caixa, esperado in zip(por_amostra, (10.0, 20.0, 30.0, 40.0), strict=True):
        assert (
            caixa.bigode_inferior
            == caixa.q1
            == caixa.mediana
            == caixa.q3
            == caixa.bigode_superior
            == esperado
        ), caixa
        assert caixa.n_discrepantes == 0, "linha constante não tem ponto fora dos bigodes"

    # Cada FEATURE varia de 10 a 40: nada colapsa, e é isso que separa os dois eixos.
    for caixa in por_feature:
        assert caixa.bigode_inferior == 10.0 and caixa.bigode_superior == 40.0, caixa
        assert caixa.mediana == 25.0, caixa

    # E a lista vazia é «calculado e vazio» nos dois, e não um 500.
    assert _caixas_por_amostra(matriz, np.array([], dtype=np.int64)) == []


def test_u11_a_amostra_toda_nan_e_NAO_AVALIAVEL_e_nao_zero_discrepantes() -> None:  # noqa: N802
    """No eixo novo, `None` e `0` continuam a ser fatos diferentes.

    A irmã por feature já o afirma sobre a coluna toda-`nan`; esta é a mesma distinção sobre a
    **linha**, e ela precisa da sua própria asserção porque a redução recebe a transposta — um
    `_caixas_por_amostra` que esquecesse a transposição marcaria a *feature* 0 como não avaliável e
    a asserção continuaria verde se ela vivesse só do outro lado.
    """
    import numpy as np

    from refino.api.rotas import _caixas_por_amostra

    matriz = _matriz_sintetica(n_features=6, n_samples=4)
    valores = matriz.values.copy()
    valores[0, :] = np.nan
    com_buraco = matriz.with_values(valores, "planta da linha toda-nan")

    caixas = _caixas_por_amostra(com_buraco, np.arange(4, dtype=np.int64))
    assert caixas[0].amostra == matriz.sample_names[0], "a caixa 0 não é a linha plantada"
    assert caixas[0].n_discrepantes is None, (
        f"a amostra sem valor nenhum reportou {caixas[0].n_discrepantes} discrepantes — *«nada "
        "ficou fora»* e *«não havia sobre o que decidir»* colapsaram num número só"
    )
    assert caixas[1].n_discrepantes is not None, (
        "a amostra VIZINHA, que tem as seis features, também devolveu `None` — sem esta metade a "
        "asserção de cima passaria numa implementação que devolvesse `None` para toda a gente"
    )


# **As colunas da planta de Tukey, e cada uma existe para separar UMA coisa.** A escolha delas é o
# que torna o teste abaixo capaz de reprovar: numa coluna sem ponto extremo, os bigodes de Tukey
# coincidem com o mínimo e o máximo, e uma planta só sobre ela não separaria nada.
COLUNAS_DA_PLANTA_DE_TUKEY: dict[str, list[float]] = {
    # Discrepantes dos DOIS lados: aqui min/max e bigode divergem, e é a coluna que separa.
    "dois_lados": [-900.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 900.0],
    # A MESMA forma sem os dois extremos: bigode == min/max, e zero discrepantes.
    "sem_fora": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0],
    # IQR zero com um ponto fora: o ponto continua a ser discrepante.
    "iqr_zero_com_fora": [0.0] * 10 + [100.0],
    # Constante: IQR zero, todo ponto está NO bigode, e nada fica fora.
    "constante": [7.0] * 11,
    # O ramo `max(dentro) < q3`: o `q3` interpolado é 25 e o maior ponto dentro do limite é 0.
    "colapsa_alto": [0.0, 0.0, 0.0, 100.0] + [float("nan")] * 7,
    # O simétrico, `min(dentro) > q1`: o `q1` é 75 e o menor ponto dentro do limite é 100.
    "colapsa_baixo": [0.0, 100.0, 100.0, 100.0] + [float("nan")] * 7,
    # Menos de quatro valores — três, dois e um.
    "tres": [1.0, 2.0, 50.0] + [float("nan")] * 8,
    "dois": [1.0, 50.0] + [float("nan")] * 9,
    "um": [42.0] + [float("nan")] * 10,
    # Nenhum valor: o caso em que não há sobre o que decidir.
    "vazia": [float("nan")] * 11,
}


def _matriz_da_planta_de_tukey() -> Any:
    """As colunas acima como matriz, uma feature por coluna — propriedade da FUNÇÃO, não contagem.

    O que se afirma aqui é a convenção de bigode de `_caixas`, e ela não é
    número observado nos arquivos do laboratório. A âncora sobre dado real está no teste de rota.
    """
    import numpy as np

    from refino.domain.matrix import RefinoMatrix, SampleMetadata

    valores = np.array([COLUNAS_DA_PLANTA_DE_TUKEY[n] for n in COLUNAS_DA_PLANTA_DE_TUKEY]).T
    return RefinoMatrix(
        values=valores,
        feature_names=tuple(COLUNAS_DA_PLANTA_DE_TUKEY),
        samples=tuple(
            SampleMetadata(
                name=f"S{i}",
                sample_type="qc" if i % 4 == 0 else "sample",
                injection_order=i + 1,
                batch="A",
                group="QC" if i % 4 == 0 else "G1",
            )
            for i in range(valores.shape[0])
        ),
        mode="tiny",
    )


def test_u11_a_caixa_e_a_de_TUKEY_e_bate_com_o_boxplot_stats_do_matplotlib() -> None:  # noqa: N802
    """**A caixa segue a convenção que o lado do DESENHO consome, e o oráculo é ele mesmo.**

    A figura do MetaboAnalyst e o `boxplot` padrão do `matplotlib` desenham bigodes de Tukey. Com
    as duas telas abertas lado a lado, uma caixa de mínimo-a-máximo lê-se como *"a ferramenta nova
    está errada"* sobre uma diferença que é só de convenção — e é por isso que o extremo aqui é o
    bigode.

    O arnês é alimentado com algo cuja resposta já se conhece. `Axes.bxp` recebe
    `whislo/q1/med/q3/whishi` prontos, e `matplotlib.cbook.boxplot_stats` é a função que os produz:
    comparar com ela é reproduzir o oráculo, e não afirmar o nosso próprio número.

    O oráculo recebe a coluna SEM os `nan`, e isso é escopo declarado. O
    `boxplot_stats` não remove ausente — uma célula vazia levaria os números da coluna inteira a
    `nan` —, e o contrato de `_caixas` é «`nan` ignorado». Alimentá-lo com a coluna crua mediria a
    diferença de política de ausente, e não a convenção de bigode.

    A planta separa, e a coluna `sem_fora` é a prova. Numa coluna sem ponto extremo, o
    bigode **é** o mínimo e **é** o máximo: uma planta só sobre ela passaria com a implementação
    antiga intacta. A asserção de divergência vem primeiro, e é ela que torna as outras capazes de
    reprovar.
    """
    import numpy as np
    from matplotlib.cbook import boxplot_stats

    from refino.api.rotas import _caixas

    matriz = _matriz_da_planta_de_tukey()
    caixas = {c.feature: c for c in _caixas(matriz, np.arange(matriz.n_features, dtype=np.int64))}
    assert set(caixas) == set(COLUNAS_DA_PLANTA_DE_TUKEY), "âncora: faltam caixas para comparar"

    # 1. A planta SEPARA: na coluna com discrepantes, Tukey e min/max dão números diferentes.
    com_fora, sem_fora = caixas["dois_lados"], caixas["sem_fora"]
    brutos = np.array(COLUNAS_DA_PLANTA_DE_TUKEY["dois_lados"])
    assert (com_fora.bigode_inferior, com_fora.bigode_superior) != (
        float(np.nanmin(brutos)),
        float(np.nanmax(brutos)),
    ), (
        f"os bigodes de `dois_lados` são o mínimo e o máximo da coluna ({brutos.min()}, "
        f"{brutos.max()}) — a planta caiu sobre dado em que as duas convenções NÃO diferem, e "
        "tudo o que vem abaixo passaria com a implementação antiga intacta"
    )
    limpos = np.array(COLUNAS_DA_PLANTA_DE_TUKEY["sem_fora"])
    assert (sem_fora.bigode_inferior, sem_fora.bigode_superior) == (
        float(limpos.min()),
        float(limpos.max()),
    ), "na coluna sem ponto extremo o bigode TEM de ser o mínimo e o máximo — é a outra direção"

    # 2. Os números batem com o oráculo, coluna a coluna, incluindo as degeneradas.
    for nome, coluna in COLUNAS_DA_PLANTA_DE_TUKEY.items():
        caixa, bruto = caixas[nome], np.array(coluna, dtype=float)
        finito = bruto[~np.isnan(bruto)]
        nosso = (caixa.bigode_inferior, caixa.q1, caixa.mediana, caixa.q3, caixa.bigode_superior)
        if finito.size == 0:
            # **`None` e não `nan`, desde a revisão de 11/09/2026.** A asserção era
            # `all(np.isnan(x) ...)`, e ela media a representação de então. O modelo passou a
            # declarar os cinco como `float | None` porque `nan` chega ao cliente como `null` e o
            # OpenAPI prometia um número — o mesmo argumento que `_opcionais` já fazia para os
            # vetores do volcano. A intenção do teste não mudou, e a asserção ficou mais forte:
            # `None` é o único valor que passa, enquanto `np.isnan` aceitaria qualquer não-número
            # que o `numpy` soubesse coagir.
            assert all(x is None for x in nosso), (
                f"{nome}: coluna sem valor nenhum devia dar `None` nos cinco, e deu {nosso}"
            )
            continue
        (oraculo,) = boxplot_stats(finito, whis=1.5)
        esperado = (
            oraculo["whislo"],
            oraculo["q1"],
            oraculo["med"],
            oraculo["q3"],
            oraculo["whishi"],
        )
        assert nosso == pytest.approx(esperado), (
            f"{nome}: a caixa é {nosso} e o `boxplot_stats` diz {esperado}. A figura seria "
            "desenhada com números que o lado do desenho não produziria dos mesmos dados"
        )
        assert caixa.n_discrepantes == len(oraculo["fliers"]), (
            f"{nome}: contámos {caixa.n_discrepantes} discrepantes e o oráculo marca "
            f"{len(oraculo['fliers'])} — o traço dos pontos que o bigode esconde está errado"
        )

    # 3. E os fatos diferentes saem separados.
    assert caixas["vazia"].n_discrepantes is None, (
        "a coluna sem valor nenhum devolveu um número — *«nada ficou fora»* e *«não havia sobre o "
        "que decidir»* colapsaram, que é o valor de retorno com dois significados"
    )
    assert caixas["constante"].n_discrepantes == 0, (
        "a coluna constante devolveu `None` ou um número — com IQR zero todo ponto está NO bigode "
        "e nada fica fora, e este é o zero que tem de continuar a ser zero"
    )
    assert caixas["iqr_zero_com_fora"].n_discrepantes == 1, (
        "IQR zero não significa «nada é discrepante»: com o limite colapsado no quartil, o ponto "
        "afastado fica fora do bigode e é o `boxplot_stats` que o diz"
    )
    assert [caixas[n].n_discrepantes for n in ("tres", "dois", "um")] == [0, 0, 0], (
        "menos de quatro valores tem de dar caixa avaliável e sem discrepante, e não `None`: há "
        "sobre o que decidir, e a decisão é «nenhum»"
    )


def test_u11_a_intersecao_VAZIA_e_calculada_e_vazia_em_vez_de_levantar() -> None:  # noqa: N802
    """**Nenhuma feature comum é «calculado e vazio», e `_caixas` quase o transformou em 500.**

    O caso é declarado em `ResumoDasDistribuicoes`: *"o caso vazio — nenhuma feature comum — é
    **calculado e vazio**, com `n_features_comparadas` a dizê-lo, e não ausência"*.

    **O defeito que este teste fixa foi introduzido pela mudança para Tukey, e nenhum dos outros
    testes o apanhou.** A versão de mínimo-e-máximo lia `cortes[0, i]` dentro de uma compreensão,
    que sobre zero colunas nunca corre; a de Tukey **desempacota** `q1, mediana, q3`, e o `numpy`
    2.5 colapsa a forma `(3, 0)` em `(0,)` quando a matriz tem zero colunas — `ValueError: not
    enough values to unpack`, e a rota devolveria 500 em vez de um resumo vazio. O comportamento
    que a mudança declarava preservar tinha de ser reafirmado por teste, e não por leitura do
    *diff*.

    **O par é causal:** as duas matrizes são a MESMA função, e só os nomes das features diferem.
    """
    from dataclasses import replace

    import numpy as np

    from refino.api.rotas import _caixas, _resumo_das_distribuicoes

    esquerda = _matriz_sintetica(n_features=6)
    direita = replace(esquerda, feature_names=tuple(f"OUTRO_{n}" for n in esquerda.feature_names))

    vazio = _resumo_das_distribuicoes(esquerda, direita, seed=7)
    assert vazio.n_features_comparadas == 0 and vazio.caixas_antes == [] == vazio.caixas_depois, (
        f"interseção vazia devia dar resumo vazio e deu {vazio.n_features_comparadas} features "
        f"com {len(vazio.caixas_antes)} caixas"
    )
    # E as duas pontas da legenda continuam a existir e a dizer a verdade sobre o corte.
    assert (vazio.n_features_antes, vazio.n_features_depois) == (6, 6), vazio

    # A outra direção, com a MESMA função: nomes que casam dão caixas.
    cheio = _resumo_das_distribuicoes(esquerda, esquerda, seed=7)
    assert len(cheio.caixas_antes) == 6, (
        "âncora: a metade que separa. Sem ela, uma implementação que devolvesse sempre lista "
        "vazia passaria na asserção de cima"
    )
    assert _caixas(esquerda, np.array([], dtype=np.int64)) == [], (
        "`_caixas` com zero colunas tem de devolver lista vazia — é o caminho que o resumo de "
        "interseção vazia percorre, e ele levantava `ValueError` no desempacotamento"
    )


# ---------------------------------------------------------------------------
# A ORDEM entre o par de PCAs e a gravação — 17/09/2026
# ---------------------------------------------------------------------------
#
# `structural_pca` LEVANTA quando `column_usability` não deixa coluna nenhuma, e a matriz
# corrigida é justamente onde o corretor cria não finitos. Enquanto `_par_de_pcas` corria
# dentro do `ExecucaoResposta` — depois de `grava_execucao` —, o desfecho era o pior par possível: a
# linha no banco **e** o trabalho a reportar falha. É o mesmo argumento que a última checagem de
# cancelamento faz, na ordem oposta.


@pytest.mark.integration
def test_a_pca_que_LEVANTA_depois_da_correcao_nao_deixa_linha_gravada(  # noqa: N802
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, memoria_de_correcoes_privada: None
) -> None:
    """**O par causal é sobre a ORDEM, e as duas metades divergem no BANCO e no CÓDIGO.**

    A planta degrada só a matriz **corrigida** — uma célula não finita por coluna —, que é o que um
    corretor de facto faz. Ela atravessa o pipeline sem o esvaziar, e mata `column_usability`: é o
    único ponto em que as duas coisas separam. Uma planta que esvaziasse a matriz daria `409` pelo
    **outro** ramo, o do `EmptyMatrixError`, e o teste passaria a medir o vizinho.

    **Três asserções e nenhuma basta sozinha:** o código é `409` e não `500`, porque o dado é do
    profissional; o detalhe nomeia a PCA, que é o que torna o desfecho **atribuível**; e a contagem
    de `runs` **não se move**, que é a propriedade que a ordem existe para garantir.

    A contagem começa por uma execução LIMPA, e é ela que impede a vacuidade. *"Zero
    linhas gravadas"* é asserção de ausência, e ela passa sobre um banco que ninguém alcançou —
    URL errada, tabela noutro esquema, gravação desligada. A execução limpa deixa **uma** linha e
    prova que esta contagem vê gravação; a degradada deixa a contagem onde estava.

    A memória de correções é privada porque `_corrige_reusando` é global do processo:
    sem isso a planta pode nem correr, e o teste reprovaria a acusar o produto.
    """
    import os

    import numpy as np
    from sqlalchemy import create_engine, func, select

    from refino.api import rotas
    from refino.persistence.config import VARIAVEL_DA_URL
    from refino.persistence.schema import runs

    _banco_de_arquivo(tmp_path, monkeypatch)
    motor = create_engine(os.environ[VARIAVEL_DA_URL])
    cliente = TestClient(build_app())
    modo = "neg_c18"

    def gravadas() -> int:
        with motor.connect() as conexao:
            return int(conexao.execute(select(func.count()).select_from(runs)).scalar_one())

    def executa(**campos: object) -> Any:
        disparo = cliente.post(
            "/execucoes",
            files=_reais(modo),
            data={
                "mode": modo,
                **CAMPOS_MINIMOS_DA_EXECUCAO,
                "estudo": "pca-degenerada",
                # **`median` e não `none`, e isto é o que torna o alvo ATRIBUÍVEL.** O relatório de
                # qualidade corre sobre a matriz **processada** e chama a mesma
                # `structural_metrics`: com `imputation=none` a planta mata os dois, e o trabalho
                # morre no relatório — antes de o par de PCAs sequer correr. Imputar repara a
                # **processada** e deixa `corrigida.matrix` degenerada, que é exactamente a
                # divergência de escopo que faz `_par_de_pcas` ser o único a levantar.
                "imputation": "median",
                **campos,
            },
        )
        assert disparo.status_code == 202, disparo.text[:400]
        identificador = disparo.json()["identificador"]
        for _ in range(2000):
            reclamada = cliente.get(f"/trabalhos/{identificador}")
            if reclamada.status_code != 202:
                return reclamada
            threading.Event().wait(0.05)
        raise AssertionError(f"o trabalho {identificador} não terminou")

    # A ÂNCORA: a mesma spec sem a planta grava, e é ela que torna a contagem abaixo não-vácua.
    limpa = executa(seed=8)
    assert limpa.status_code == 200, limpa.text[:400]
    antes = gravadas()
    assert antes == 1, f"a execução limpa gravou {antes} linhas: a contagem abaixo seria vácua"

    real = rotas._corrige_reusando

    def degradando(corretor: Any, matriz: Any, **kwds: Any) -> Any:
        from dataclasses import replace as _replace

        resultado = real(corretor, matriz, **kwds)
        estragados = np.array(resultado.matrix.values, dtype=np.float64, copy=True)
        # Uma célula por COLUNA, e não a matriz inteira: `column_usability` descarta a coluna com
        # QUALQUER célula não finita, e o resto da matriz continua a ser dado de verdade — que é o
        # que faz isto atravessar o pipeline em vez de o esvaziar.
        estragados[0, :] = np.nan
        return _replace(
            resultado, matrix=resultado.matrix.with_values(estragados, "planta do par causal")
        )

    monkeypatch.setattr(rotas, "_corrige_reusando", degradando)
    reclamada = executa(seed=9)

    detalhe = str(reclamada.json().get("detail", reclamada.text))
    assert reclamada.status_code == 409, (
        f"a PCA degenerada deu {reclamada.status_code} e não `409` — `ui/cliente.py` traduz `500` "
        f"como «Defeito da ferramenta, não do seu arquivo»: {detalhe[:300]}"
    )
    assert "PCA" in detalhe and "coluna utilizável" in detalhe, (
        f"o `409` não é o da PCA: {detalhe[:300]}. Sem isto o teste passaria pelo ramo do "
        "`EmptyMatrixError`, que é o vizinho e não o alvo"
    )
    assert gravadas() == antes, (
        f"a contagem de execuções passou de {antes} para {gravadas()} depois de o trabalho FALHAR: "
        "a linha gravada e a falha reportada são dois fatos contraditórios, e o do banco sobrevive "
        "ao processo"
    )


# ---------------------------------------------------------------------------
# Os vetores do volcano atravessam o fio, e o `q` NÃO se move
# ---------------------------------------------------------------------------
#
# **A raiz `volcano` é o primeiro campo da resposta que depende do PAR DE GRUPOS**, e por isso o
# primeiro que tem `None` legítimo. `distribuicoes` vem sempre — toda execução tem uma matriz antes
# e uma depois —, então o par causal da presença/ausência é novo nesta unidade.


@pytest.mark.integration
def test_u6_sem_par_de_grupos_a_raiz_volcano_e_NULA_e_com_par_ela_VEM(  # noqa: N802
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """O par causal na rota: *"não calculado"* e *"calculado e vazio"* são opostos.

    O teste univariado é pareado, e sem `group_a`/`group_b` a rota não o corre. Um objeto vazio ali
    diria que ele correu e não achou nada — que é a leitura oposta, e é justamente a que **este**
    corpus produz quando o par existe.

    **A âncora de não-vacuidade vem primeiro**: com par, os quatro vetores têm de vir com
    população, senão a comparação de comprimentos abaixo passaria sobre listas vazias.
    """
    _banco_de_arquivo(tmp_path, monkeypatch)
    cliente = TestClient(build_app())
    modo = "neg_c18"

    # (a) sem par: a raiz é `None`, e o resto da resposta continua inteiro.
    sem_par = _executa_ate_o_fim(cliente, modo, estudo="u6_sem_par")
    assert sem_par["volcano"] is None, sem_par["volcano"]
    assert sem_par["distribuicoes"] is not None, (
        "âncora: `distribuicoes` também sumiu, e a ausência do volcano deixaria de ser atribuível "
        "ao par de grupos"
    )

    # (b) com par: a raiz vem, com os quatro vetores alinhados.
    com_par = _executa_ate_o_fim(
        cliente, modo, estudo="u6_com_par", group_a="Asymptomatic", group_b="HAMTSP"
    )
    volcano = com_par["volcano"]
    assert volcano is not None, "a raiz não veio com o par declarado"

    testadas = volcano["n_features_testadas"]
    assert testadas > 0, "âncora: zero features testadas tornaria vácuo tudo o que vem abaixo"
    for campo in ("features", "log2fc", "pvalue", "qvalue_fdr"):
        assert len(volcano[campo]) == testadas, (
            f"`{campo}` tem {len(volcano[campo])} entradas e a rota declara {testadas} features "
            "testadas — os quatro vetores indexam as MESMAS features, e um desalinhamento põe o "
            "`p` de uma ao lado do fold change de outra"
        )
    assert len(set(volcano["features"])) == testadas, "um nome de feature repetiu-se"

    # O `n` de cada grupo, medido na matriz e não inventado.
    assert volcano["n_grupo_a"] > 0 and volcano["n_grupo_b"] > 0, volcano
    assert (volcano["grupo_a"], volcano["grupo_b"]) == ("Asymptomatic", "HAMTSP"), (
        "a ordem do par inverteu-se no fio: `grupo_a` é o NUMERADOR do fold change, e trocá-lo "
        "espelha a figura sem nada acusar"
    )
    assert volcano["correcao"] and volcano["alfa_fdr"] > 0, volcano

    # As duas procedências, e elas são DIFERENTES — o `p` da processada, o FC da anterior.
    assert volcano["matriz_do_pvalue"] != volcano["matriz_do_fold_change"], volcano

    # O `null` atravessa em vez de a feature ser descartada, e há pelo menos um neste corpus.
    assert any(v is None for v in volcano["log2fc"]), (
        "nenhum fold change chegou `null` — ou este corpus não tem mediana zero em grupo nenhum, "
        "ou o vetor está a ser filtrado, que é o que R6 proíbe"
    )


@pytest.mark.integration
def test_u6_a_contagem_de_FDR_da_resposta_e_a_dos_vetores_do_volcano(  # noqa: N802
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Na fronteira, a figura desenha o MESMO número que a aba de qualidade mostra.

    O resultado esperado neste corpus é ZERO, e isso é o achado e não um defeito. O que este teste
    afirma é a **igualdade** entre as duas superfícies, e ela vale seja qual for o número.

    > **E ESTE TESTE NÃO SEPARA DUAS CHAMADAS a `univariate_test`, medido em 11/09/2026.** A planta
    > — uma segunda chamada, com `parametric=True` forçado, alimentando o volcano enquanto a
    > contagem continuava a sair da primeira — deixou o teste **verde**. A razão é o corpus: os dois
    > lados dão `0`, e `0 == 0` é verdade sob qualquer par de cálculos. É a asserção de ausência
    > satisfeita por vacuidade um nível acima, sobre uma igualdade em vez de sobre um
    > conjunto.
    >
    > **Quem separa é `test_u6_a_rota_chama_univariate_test_UMA_vez_so`**, que conta as chamadas na
    > fonte. Este fica porque afirma o **desfecho** — e a distância entre os dois vai
    > declarada em vez de subentendida.
    """
    _banco_de_arquivo(tmp_path, monkeypatch)
    cliente = TestClient(build_app())

    corpo = _executa_ate_o_fim(
        cliente, "neg_c18", estudo="u6_fdr", group_a="Asymptomatic", group_b="HAMTSP"
    )
    volcano = corpo["volcano"]
    da_qualidade = corpo["qualidade"]["n_significant_fdr"]
    assert da_qualidade is not None, (
        "âncora: a aba de qualidade não calculou a contagem, e a comparação abaixo seria vácua"
    )

    alfa = volcano["alfa_fdr"]
    dos_vetores = sum(1 for q in volcano["qvalue_fdr"] if q is not None and q < alfa)
    assert dos_vetores == da_qualidade, (
        f"a aba de qualidade mostra {da_qualidade} significativas e os vetores que a figura "
        f"desenha dão {dos_vetores} — as duas superfícies do mesmo produto contam quantidades "
        "diferentes com o mesmo nome"
    )

    # E a outra metade, que é o que o corpus diz: com 24 pares nos quatro modos, mediu-se zero.
    menor = min((q for q in volcano["qvalue_fdr"] if q is not None), default=None)
    assert menor is not None, "âncora: nenhum `q` é definido, e a leitura abaixo seria vácua"
    assert (dos_vetores == 0) == (menor >= alfa), (
        f"a contagem diz {dos_vetores} e o menor q é {menor} contra o corte {alfa} — as duas "
        "leituras do mesmo vetor discordam"
    )


def test_u6_a_rota_chama_univariate_test_UMA_vez_so() -> None:  # noqa: N802
    """**O mecanismo que a igualdade acima NÃO consegue afirmar — e a distância foi MEDIDA.**

    A figura tem de desenhar o mesmo número que a aba de qualidade mostra. A forma que o
    garante é **uma** chamada a `univariate_test` alimentando as duas coisas: com duas, a igualdade
    passa a ser coincidência de dois cálculos, e `parametric=None` decide por Shapiro-Wilk — duas
    decisões independentes podem legitimamente divergir.

    **A igualdade sobre a resposta não apanha isso neste corpus, e a planta prova-o:** uma segunda
    chamada com `parametric=True` forçado deixou
    `test_u6_a_contagem_de_FDR_da_resposta_e_a_dos_vetores_do_volcano` **verde**, porque os dois
    lados dão `0` e `0 == 0` vale sob qualquer par de cálculos.

    Proxy de FORMA, e a estreiteza vai declarada: ele conta ocorrências do nome no
    módulo. Não vê uma chamada indirecta, nem distingue as rotas — o que compra é a direção que
    importa, que é alguém acrescentar uma segunda chamada por conveniência.

    A contagem, e não `in`: `x in texto` devolve o desfecho de sucesso quando
    há duas ocorrências, que é exactamente o defeito procurado.
    """
    import ast

    fonte = (Path(__file__).resolve().parents[1] / "src" / "refino" / "api" / "rotas.py").read_text(
        encoding="utf-8"
    )
    chamadas = [
        no
        for no in ast.walk(ast.parse(fonte))
        if isinstance(no, ast.Call)
        and isinstance(no.func, ast.Name)
        and no.func.id == "univariate_test"
    ]
    assert len(chamadas) == 1, (
        f"`rotas.py` chama `univariate_test` {len(chamadas)} vez(es), nas linhas "
        f"{[c.lineno for c in chamadas]}. A contagem de FDR da aba de qualidade e os vetores do "
        "volcano têm de sair da MESMA chamada (R8b); com duas, a igualdade entre as duas "
        "superfícies passa a depender de duas decisões de normalidade concordarem"
    )


# ---------------------------------------------------------------------------
# O intervalo do fold change atravessa o fio ao lado do ponto
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_o_intervalo_do_fold_change_atravessa_o_fio_alinhado_e_com_motivo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Os cinco vetores do intervalo, um por feature testada, com o motivo onde ele não existe.

    **`CAMPOS_MINIMOS_DA_EXECUCAO` corre com `imputation="none"`**, então os zeros do arquivo chegam
    à matriz do fold change sem ser trocados: é o caso em que o valor de ordem é zero, e ele existe
    neste corpus — a mesma observação que o teste da raiz `volcano` acima faz sobre `log2fc` nulo.

    **As asserções de forma são de desenho e não de corpus:** limites e motivo são complementares
    (um existe exactamente quando o outro não), o ponto desenhado fica dentro do intervalo em log2,
    e a razão é `2 ** log2`. Sem a âncora de não-vacuidade, um vetor de motivos todo preenchido as
    satisfaria todas.
    """
    _banco_de_arquivo(tmp_path, monkeypatch)
    cliente = TestClient(build_app())

    volcano = _executa_ate_o_fim(
        cliente, "neg_c18", estudo="ic_fc", group_a="Asymptomatic", group_b="HAMTSP"
    )["volcano"]

    testadas = volcano["n_features_testadas"]
    limites_por_campo = (
        "log2fc_ci_lower",
        "log2fc_ci_upper",
        "fold_change_ci_lower",
        "fold_change_ci_upper",
    )
    for campo in (*limites_por_campo, "fold_change_ci_undefined"):
        assert len(volcano[campo]) == testadas, (campo, len(volcano[campo]), testadas)
    assert volcano["nivel_do_intervalo"] == 0.95

    vocabulario = set(get_args(FoldChangeIntervalUndefined))
    definidos = 0
    for i in range(testadas):
        limites = [volcano[campo][i] for campo in limites_por_campo]
        motivo = volcano["fold_change_ci_undefined"][i]
        if motivo is not None:
            assert motivo in vocabulario, motivo
            assert limites == [None] * 4, (i, motivo, limites)
            continue
        definidos += 1
        inferior2, superior2, inferior, superior = limites
        assert None not in limites, (i, limites)
        # `1e-12` cobre o empate em que a meia-largura é zero: o centro sai de `ln(razão) / ln 2` e
        # o ponto de `log2(razão)`, e os dois podem diferir no último bit.
        ponto = volcano["log2fc"][i]
        assert inferior2 - 1e-12 <= ponto <= superior2 + 1e-12, (i, limites, ponto)
        assert inferior == pytest.approx(2.0**inferior2, rel=1e-9)
        assert superior == pytest.approx(2.0**superior2, rel=1e-9)
    assert definidos > 0, "âncora: nenhuma feature teve intervalo, e o laço seria vácuo"

    # Sem ponto não há intervalo, e isso é desenho: mediana ≤ 0 põe o valor de ordem em ≤ 0.
    sem_ponto = [i for i, v in enumerate(volcano["log2fc"]) if v is None]
    assert sem_ponto, "âncora: este corpus tem fold change nulo sob `imputation='none'`"
    assert all(volcano["fold_change_ci_undefined"][i] is not None for i in sem_ponto)


# ---------------------------------------------------------------------------
# A escolha do teste do contraste, e a normalidade por variável e por grupo no fio
# ---------------------------------------------------------------------------

# O par da tela e o estudo de cada corrida. **O estudo é próprio de cada escolha**, para que a chave
# do trabalho seja nova e a rota corra em vez de reclamar um resultado guardado por outro teste.
_PAR_D663: dict[str, object] = {"group_a": "Asymptomatic", "group_b": "HAMTSP"}


def _corre_com_o_teste(cliente: TestClient, escolha: str) -> dict[str, Any]:
    return _executa_ate_o_fim(
        cliente,
        "neg_c18",
        estudo=f"d663_{escolha}",
        analise_json=json.dumps({"univariate_test": escolha}),
        **_PAR_D663,
    )


@pytest.mark.integration
def test_d663_a_escolha_do_teste_chega_a_univariate_test_e_o_automatico_e_o_de_antes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """«Welch» e «Mann-Whitney» chegam como `parametric=True` e `False`; «Automático» como `None`.

    **O espião embrulha a função real** e guarda os argumentos e o resultado de cada chamada. O
    «Automático» é comparado, BIT A BIT, com a chamada de antes deste campo existir — a mesma
    matriz e o mesmo par, SEM o argumento `parametric` —, que é o que «a saída de hoje» quer dizer.

    A planta cai sobre dado em que as coisas diferem. Neste arquivo a decisão automática
    escolhe Mann-Whitney (observação sobre o `neg_c18` sob `CAMPOS_MINIMOS_DA_EXECUCAO`, e é a
    âncora): então o `p` de «Welch» tem de diferir do automático, e o de «Mann-Whitney» forçado tem
    de ser o MESMO vetor, com a rota da normalidade a dizer `forced` em vez de `shapiro`.
    """
    import refino.api.rotas as rotas

    original = rotas.univariate_test
    chamadas: list[dict[str, Any]] = []

    def espiao(matriz: Any, grupos: Any, parametric: Any = None, **kw: Any) -> Any:
        resultado = original(matriz, grupos, parametric, **kw)
        chamadas.append(
            {"matriz": matriz, "grupos": grupos, "parametric": parametric, "kw": kw, "r": resultado}
        )
        return resultado

    monkeypatch.setattr(rotas, "univariate_test", espiao)
    _banco_de_arquivo(tmp_path, monkeypatch)
    cliente = TestClient(build_app())

    volcanos: dict[str, dict[str, Any]] = {}
    for escolha, esperado in (("automatic", None), ("welch", True), ("mann_whitney", False)):
        antes = len(chamadas)
        volcanos[escolha] = _corre_com_o_teste(cliente, escolha)["volcano"]
        assert len(chamadas) == antes + 1, f"«{escolha}» não chamou `univariate_test` uma vez"
        assert chamadas[-1]["parametric"] is esperado, (escolha, chamadas[-1]["parametric"])

    automatico = chamadas[0]
    de_antes = original(automatico["matriz"], automatico["grupos"], **automatico["kw"])
    for campo in ("statistic", "pvalue", "qvalue_fdr", "fold_change", "log2fc"):
        assert getattr(automatico["r"], campo).tobytes() == getattr(de_antes, campo).tobytes(), (
            f"«Automático» mudou `{campo}` em relação à chamada sem `parametric`"
        )
    assert automatico["r"].normality == de_antes.normality

    assert volcanos["automatic"]["rota_da_normalidade"] == "shapiro"
    assert volcanos["automatic"]["parametrico"] is False, (
        "âncora: neste arquivo a decisão automática escolhia Mann-Whitney, e o par abaixo foi "
        "escrito para isso"
    )
    for escolha, parametrico in (("welch", True), ("mann_whitney", False)):
        assert volcanos[escolha]["rota_da_normalidade"] == "forced", escolha
        assert volcanos[escolha]["parametrico"] is parametrico, escolha
    assert volcanos["mann_whitney"]["pvalue"] == volcanos["automatic"]["pvalue"]
    assert volcanos["welch"]["pvalue"] != volcanos["automatic"]["pvalue"], (
        "«Welch» devolveu os mesmos `p` do automático: a escolha não alcançou o teste"
    )


def test_d663_teste_fora_das_tres_opcoes_e_recusado_na_fronteira(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Um valor fora de «Automático», «Welch» e «Mann-Whitney» é `422` antes de correr nada.

    **O par causal é o mesmo leitor com o valor válido**: `_analise_ou_422` é quem a rota chama
    antes de despachar o trabalho, e com `welch` ele devolve o contrato. Disparar a rota com o
    valor válido também provaria isso, ao preço de uma execução inteira em segundo plano.
    """
    from refino.api.rotas import _analise_ou_422

    _banco_de_arquivo(tmp_path, monkeypatch)
    recusa = TestClient(build_app()).post(
        "/execucoes",
        files=_reais("neg_c18"),
        data={
            "mode": "neg_c18",
            **CAMPOS_MINIMOS_DA_EXECUCAO,
            "estudo": "d663_recusa",
            "analise_json": json.dumps({"univariate_test": "t"}),
        },
    )
    assert recusa.status_code == 422, recusa.text[:300]
    assert "UnivariateTest" in recusa.text and "'t'" in recusa.text, recusa.text[:300]
    assert _analise_ou_422(json.dumps({"univariate_test": "welch"})).univariate_test == "welch"


@pytest.mark.integration
def test_d663_a_normalidade_atravessa_o_fio_por_grupo_de_estudo_com_a_particao_e_o_resumo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A raiz `normalidade` de uma execução real: os grupos de estudo, a partição e o resumo.

    Os quatro grupos são observação sobre o `neg_c18`, e o QC não está entre eles. O resumo é
    recontado aqui dos vetores do fio — o outro caminho como oráculo —, e a âncora de
    não-vacuidade vem antes.
    """
    _banco_de_arquivo(tmp_path, monkeypatch)
    raiz = _corre_com_o_teste(TestClient(build_app()), "automatic")["normalidade"]

    assert raiz is not None, "a família vem ligada por padrão e não correu"
    assert raiz["teste"] == "Shapiro-Wilk" and raiz["matriz"] == "processada"
    grupos = {g["grupo"]: g for g in raiz["grupos"]}
    assert sorted(grupos) == ["ATLL", "Asymptomatic", "HAMTSP", "NoHTLV"], sorted(grupos)
    assert any(g["n_testadas"] > 0 for g in grupos.values()), "âncora: nenhuma variável testada"

    total = len(raiz["features"])
    for nome, grupo in grupos.items():
        for vetor in ("w", "pvalue", "qvalue", "n_valores", "nao_avaliavel", "valor_imputado"):
            assert len(grupo[vetor]) == total, (nome, vetor)
        partes = (
            grupo["n_menos_de_tres"]
            + grupo["n_constantes"]
            + grupo["n_com_imputado"]
            + grupo["n_testadas"]
        )
        assert partes == grupo["n_variaveis"] == total, (nome, partes, total)
        testadas = [
            p
            for p, motivo, imputado in zip(
                grupo["pvalue"], grupo["nao_avaliavel"], grupo["valor_imputado"], strict=True
            )
            if motivo is None and not imputado
        ]
        assert len(testadas) == grupo["n_testadas"], nome
        assert all(p is not None for p in testadas), f"{nome}: variável testada sem `p`"
        assert sum(p < raiz["alfa"] for p in testadas) == grupo["n_rejeitadas"], nome


# ---------------------------------------------------------------------------
# As duas cargas novas do fio, e o SVD que corre uma vez
# ---------------------------------------------------------------------------


def test_u8_a_rota_do_diagnostico_SERVE_a_pca_e_ela_bate_com_a_trajetoria() -> None:  # noqa: N802
    """A carga da PCA, medida na saída da rota.

    O oráculo é a coincidência: `trajetoria_dos_qcs.scores` e as posições de QC de
    `pca.scores_pc1` saem do **mesmo** `StructuralPCA`, e o teste afirma que são o mesmo vetor. Se
    alguém acrescentar uma segunda chamada de `structural_pca`, os dois continuam iguais (o SVD é
    determinístico) — o que este teste apanha é a carga **ausente** ou de outra população.
    """
    import numpy as np

    from refino.api.app import build_app

    with TestClient(build_app()) as cliente:
        resposta = cliente.post(
            "/matrizes/diagnosticar", files=_reais("pos_hilic"), data={"mode": "pos_hilic"}
        )
    assert resposta.status_code == 200, resposta.text
    corpo = resposta.json()
    pca = corpo["pca"]

    # (a) a carga existe, e os vetores por AMOSTRA têm a cardinalidade das amostras.
    assert len(pca["scores_pc1"]) == corpo["n_samples"] == len(pca["tipos_de_amostra"])
    assert len(pca["scores_pc2"]) == corpo["n_samples"]

    # (b) os vetores por FEATURE são os das RETIDAS, e o par de denominadores é um par.
    assert len(pca["features"]) == pca["n_features_usadas"] == len(pca["loadings_pc1"])
    assert pca["n_features_totais"] == corpo["n_features"]
    # OBSERVAÇÃO e não invariante: no `pos_hilic` o passo 3 não descarta coluna
    # nenhuma — 6.035 de 6.035. Então este par NÃO exercita a diferença
    # entre os dois denominadores, e quem a exercita é `test_u8_as_quatro_de_pca_DESENHAM_...` em
    # `tests/test_desktop_figuras.py`, cuja carga sintética tem 3 de 6.035. Dizê-lo é o que impede
    # a leitura de que a igualdade abaixo prova alinhamento.
    assert pca["n_features_usadas"] <= pca["n_features_totais"]

    # (c) os tipos são o domínio fechado, e há QC.
    assert set(pca["tipos_de_amostra"]) <= {"qc", "sample", "validate"}
    assert pca["tipos_de_amostra"].count("qc") == 7

    # (d) **a mesma execução do SVD**: os scores de QC batem com a trajetória, elemento a elemento.
    do_qc = [
        s for s, t in zip(pca["scores_pc1"], pca["tipos_de_amostra"], strict=True) if t == "qc"
    ]
    assert np.allclose(do_qc, corpo["trajetoria_dos_qcs"]["scores"])
    assert pca["n_features_usadas"] == corpo["trajetoria_dos_qcs"]["n_features_usadas"]
    assert len(pca["variancia_explicada"]) == pca["n_componentes"] == 2


def test_o_grupo_por_amostra_da_pca_sai_de_SAMPLE_MASK_e_o_QC_vai_nulo() -> None:  # noqa: N802
    """O grupo de cada amostra no fio da PCA, e o QC FORA dele.

    A planta cai sobre dado em que as duas coisas diferem: o QC desta matriz tem o rótulo
    `"QC"` no campo de grupo, como nos arquivos do laboratório, e a validação tem o nome de um grupo
    de estudo. Servir `matriz.groups` inteiro poria os dois na figura como grupos; servir por
    `sample_mask` põe `None` nas duas posições, e é isso que se afirma.
    """
    from refino.api.rotas import _resumo_da_pca

    tipos = ["qc", "sample", "sample", "validate", "sample", "qc", "sample", "sample", "sample"]
    grupos = ["QC", "A", "B", "A", "A", "QC", "B", "A", "B"]
    amostras = tuple(
        SampleMetadata(name=f"a{i:02d}", injection_order=i + 1, sample_type=tipo, group=grupo)
        for i, (tipo, grupo) in enumerate(zip(tipos, grupos, strict=True))
    )
    valores = 100.0 + np.random.default_rng(20261106).standard_normal((len(tipos), 6)) * 5.0
    matriz = RefinoMatrix(
        values=valores,
        feature_names=tuple(f"f{j}" for j in range(6)),
        samples=amostras,
        mode="pos_hilic",
    )
    resumo = _resumo_da_pca(
        structural_pca(matriz, pretransform=PRETRANSFORM, n_components=2), matriz, seed=42
    )

    assert "QC" in grupos and grupos[tipos.index("validate")] == "A", "âncora: a planta não separa"
    assert resumo.grupos_por_amostra == [
        grupo if tipo == "sample" else None for tipo, grupo in zip(tipos, grupos, strict=True)
    ]
    assert "QC" not in resumo.grupos_por_amostra
    assert len(resumo.grupos_por_amostra) == len(resumo.tipos_de_amostra) == len(tipos)


def test_u8_o_resumo_do_plsda_sai_do_resultado_que_a_rota_JA_calculou() -> None:  # noqa: N802
    """No fio, os três vetores e o par, em processo — porque `/execucoes` exige banco.

    O caminho ponta a ponta de `/execucoes` não corre nesta máquina (ver
    `tests/test_api_execucoes.py`), então o que se afirma aqui é a propriedade do **construtor da
    carga**: ele consome o `PLSDAResult` que a rota já calculou e não faz uma segunda chamada.

    A construção do modelo é a garantia de que o PLS-DA nunca sai sem validação, e este teste é a
    evidência de que ela é ALCANÇADA. Um `ResumoDoPlsda` que nunca fosse construído tornaria a
    garantia verdadeira por vacuidade no fio: o mecanismo correto sobre um alvo que não existe.
    """
    import numpy as np

    from refino.api.rotas import PLSDA_CV_FOLDS, _resumo_do_plsda
    from refino.statistics.plsda import plsda
    from tests.conftest import matrix_from_toserrf

    matriz = matrix_from_toserrf("pos_hilic")
    par = ("Asymptomatic", "HAMTSP")
    resultado = plsda(matriz, par, n_components=2, cv_folds=7, n_permutations=20, seed=42)
    resumo = _resumo_do_plsda(resultado, matriz, par)

    # (a) o par de validação do PLS-DA é o do resultado, e não um número recalculado.
    assert resumo.q2 == resultado.q2
    assert resumo.permutation_pvalue == resultado.permutation_pvalue
    assert resumo.cv_folds == PLSDA_CV_FOLDS and resumo.n_permutations == 20

    # (b) os vetores por amostra descrevem as amostras DO CONTRASTE, e mais nenhuma.
    esperadas = [a.group for a in matriz.samples if a.group in par]
    assert resumo.grupos_por_amostra == esperadas
    assert len(resumo.scores_componente_1) == len(esperadas) == resultado.scores.shape[0]
    assert len(esperadas) < matriz.n_samples, (
        "o contraste tem tantas amostras quanto a matriz: o recorte não recortou nada, e as "
        "asserções sobre a seleção seriam vácuas"
    )

    # (c) `grupos[1]` é a classe positiva, e é o que `y = (rotulos == segundo)` monta.
    assert (resumo.grupo_negativo, resumo.grupo_positivo) == par
    assert resumo.n_negativo == esperadas.count(par[0]) == 18
    assert resumo.n_positivo == esperadas.count(par[1]) == 14

    # (d) os vetores por feature descrevem a matriz que correu, e o `n` é o dela.
    assert resumo.n_features == matriz.n_features == len(resumo.features)
    assert len(resumo.vip) == len(resumo.loadings_componente_1) == resumo.n_features

    # (e) os escores de validação cruzada são os do resultado, alinhados às mesmas
    #     amostras — e não os de calibração com outro nome. A âncora é que os dois diferem neste
    #     contraste; sem ela, servir `scores` no campo novo passaria.
    assert not np.allclose(resultado.cv_scores, resultado.scores), (
        "âncora: os escores de validação cruzada coincidem com os de calibração neste contraste, "
        "e a asserção abaixo não distinguiria um campo do outro"
    )
    for componente, campo in (
        (0, resumo.scores_cv_componente_1),
        (1, resumo.scores_cv_componente_2),
    ):
        assert len(campo) == len(esperadas)
        assert np.allclose(np.asarray(campo, dtype=float), resultado.cv_scores[:, componente])
    assert all(v is not None for v in resumo.vip), "o VIP deste corpus é todo finito"
    assert np.allclose([float(v) for v in resumo.vip if v is not None], resultado.vip)


def test_u8_o_resumo_do_plsda_RECUSA_quando_a_selecao_e_o_modelo_discordam() -> None:  # noqa: N802
    """**O par causal da guarda que substitui a irmã que não se pôde partilhar.**

    `statistics/plsda._matriz_e_rotulos` seleciona por `np.isin(rotulos, groups)` e é privada;
    `_resumo_do_plsda` refaz a seleção por `a.group in grupos`. Tornar a máscara pública reprovaria
    `test_ca_6_6_a_unica_funcao_publica_devolve_o_resultado_completo`, que fixa `__all__` como o
    mecanismo de *"nenhum caminho público expõe scores soltos"* — e afrouxar um portão que protege
    a validação obrigatória do PLS-DA é a resposta errada.

    Então a divergência é feita **levantar**. Ponta A: com a matriz certa, constrói. Ponta B: com
    uma matriz de outra população — uma amostra do contraste a menos —, recusa com `500` em vez de
    servir um vetor de grupos deslocado.
    """
    import numpy as np
    from fastapi import HTTPException

    from refino.api.rotas import _resumo_do_plsda
    from refino.domain.matrix import RefinoMatrix
    from refino.statistics.plsda import plsda
    from tests.conftest import matrix_from_toserrf

    matriz = matrix_from_toserrf("pos_hilic")
    par = ("Asymptomatic", "HAMTSP")
    resultado = plsda(matriz, par, n_components=2, cv_folds=7, n_permutations=5, seed=42)
    assert _resumo_do_plsda(resultado, matriz, par) is not None

    # A planta: uma amostra do contraste a MENOS do que o modelo viu. As duas seleções passam a
    # devolver contagens diferentes, que é exactamente o que a guarda existe para acusar.
    fora = next(i for i, a in enumerate(matriz.samples) if a.group == "HAMTSP")
    encolhida = RefinoMatrix(
        values=np.delete(matriz.values, fora, axis=0),
        feature_names=matriz.feature_names,
        samples=tuple(a for i, a in enumerate(matriz.samples) if i != fora),
        mode=matriz.mode,
    )
    with pytest.raises(HTTPException) as erro:
        _resumo_do_plsda(resultado, encolhida, par)
    assert erro.value.status_code == 500
    assert "seleção de amostras do contraste" in str(erro.value.detail)


def test_o_resumo_do_plsda_serve_os_escores_de_VALIDACAO_CRUZADA_do_resultado() -> None:  # noqa: N802
    """O campo novo do fio, em processo e sem o `pos_hilic`.

    A asserção (e) do teste acima diz o mesmo sobre o corpus, e pula onde ele não está; sem esta, a
    linha da rota que serve `cv_scores` não correria em máquina nenhuma sem os dados do laboratório.
    O que se afirma não depende do dado: o corpo leva os escores de validação cruzada DO resultado
    que a rota calculou, nas linhas do contraste, ao lado do `q2` e do `p` do mesmo objeto.

    **Um terceiro grupo fica fora do contraste**, senão a seleção não recortaria nada; **e a âncora
    vem antes**: neste contraste os dois pares de escores diferem, senão servir os de calibração no
    campo novo passaria.
    """
    from refino.api.rotas import PLSDA_CV_FOLDS, _resumo_do_plsda
    from refino.statistics.plsda import plsda

    gerador = np.random.default_rng([20261106, 21])
    grupos = ["A"] * 8 + ["B"] * 8 + ["C"] * 4
    valores = gerador.normal(10.0, 1.0, size=(len(grupos), 20))
    valores[8:16, :3] += 2.0
    matriz = RefinoMatrix(
        values=valores,
        feature_names=tuple(f"f{j}" for j in range(20)),
        samples=tuple(
            SampleMetadata(name=f"s{i:02d}", injection_order=i + 1, sample_type="sample", group=g)
            for i, g in enumerate(grupos)
        ),
        mode="pos_hilic",
    )
    par = ("A", "B")
    resultado = plsda(matriz, par, cv_folds=PLSDA_CV_FOLDS, n_permutations=5, seed=42)
    resumo = _resumo_do_plsda(resultado, matriz, par)

    assert not np.allclose(resultado.cv_scores, resultado.scores), (
        "âncora: os escores de validação cruzada coincidem com os de calibração neste contraste"
    )
    assert resumo.grupos_por_amostra == ["A"] * 8 + ["B"] * 8
    for componente, campo in (
        (0, resumo.scores_cv_componente_1),
        (1, resumo.scores_cv_componente_2),
    ):
        assert len(campo) == 16 < matriz.n_samples
        assert np.allclose(np.asarray(campo, dtype=float), resultado.cv_scores[:, componente])
    assert (resumo.q2, resumo.permutation_pvalue) == (resultado.q2, resultado.permutation_pvalue)


# ---------------------------------------------------------------------------------------
# A ANOVA de N grupos no fio
# ---------------------------------------------------------------------------------------


@pytest.mark.integration
def test_u6_analises_a_familia_DESLIGADA_deixa_a_raiz_nula_e_a_ligada_TRAZ_a_anova(  # noqa: N802
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """O par causal na rota, e aqui ele NÃO depende do par de grupos.

    A ANOVA deriva o conjunto de grupos da matriz, então a raiz `anova` vem **sem**
    `group_a`/`group_b` — ao contrário de `volcano` e `plsda`, que são pareados. O que a apaga é a
    família desligada, e é essa a planta.

    **A âncora de não-vacuidade vem primeiro:** com a família ligada os três vetores têm de vir com
    população, senão a comparação de comprimentos abaixo passaria sobre listas vazias.
    """
    _banco_de_arquivo(tmp_path, monkeypatch)
    cliente = TestClient(build_app())
    modo = "neg_c18"

    # (a) família desligada: a raiz é `None`, e o resto da resposta continua inteiro.
    desligada = _executa_ate_o_fim(
        cliente,
        modo,
        estudo="u6a_desligada",
        analise_json=json.dumps({"anova": False}),
    )
    assert desligada["anova"] is None, desligada["anova"]
    assert desligada["distribuicoes"] is not None, (
        "âncora: `distribuicoes` também sumiu, e a ausência da ANOVA deixaria de ser atribuível à "
        "família desligada"
    )

    # (b) família ligada (o padrão), e SEM par de grupos: a raiz vem na mesma.
    ligada = _executa_ate_o_fim(cliente, modo, estudo="u6a_ligada")
    anova = ligada["anova"]
    assert anova is not None, "a raiz não veio com a família ligada por padrão"
    assert ligada["volcano"] is None, (
        "âncora: o volcano veio sem par de grupos, e a independência da ANOVA em relação ao par "
        "deixaria de ser o que este teste separa"
    )

    testadas = anova["n_features_testadas"]
    assert testadas > 0, "âncora: zero features testadas tornaria vácuo tudo o que vem abaixo"
    for campo in ("features", "statistic", "pvalue", "qvalue_fdr"):
        assert len(anova[campo]) == testadas, (
            f"`{campo}` tem {len(anova[campo])} entradas e a rota declara {testadas} features "
            "testadas — os três vetores indexam as MESMAS features"
        )
    assert len(set(anova["features"])) == testadas, "um nome de feature repetiu-se"

    # Os pares `(grupo, n)`, medidos na matriz e não inventados.
    assert len(anova["grupos"]) == len(anova["n_por_grupo"]) >= 2, anova["grupos"]
    assert all(n > 0 for n in anova["n_por_grupo"]), anova["n_por_grupo"]
    assert "QC" not in anova["grupos"], (
        "o QC entrou na população da ANOVA: ele é a régua do sistema e não um grupo de estudo"
    )

    # Os rótulos do método, e eles são o vocabulário fechado do domínio.
    assert anova["metodo"] == "anova", anova["metodo"]
    assert anova["post_hoc_metodo"] in {"tukey", "fisher", "dunn"}, anova["post_hoc_metodo"]
    assert anova["post_hoc_rota"] in {"ran", "no_survivors", "nothing_testable"}, anova
    assert anova["correcao"] and anova["alfa_fdr"] > 0 and anova["min_por_grupo"] >= 2, anova


@pytest.mark.integration
def test_u6_analises_o_post_hoc_PEDIDO_e_o_que_a_raiz_declara(  # noqa: N802
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """**O par causal do parâmetro: `fisher` pedido tem de chegar ao resultado.**

    `AnalysisSpec.post_hoc` atravessa a rota até `one_way_anova`, e um caminho que o ignorasse
    correria o padrão `tukey` com o resultado a declarar o que o usuário pediu — o modo de falha
    insidioso, o que não levanta.

    **A âncora é a assimetria dos dois valores:** os dois têm de chegar, e o padrão tem de ser
    `tukey`. Com um só, um caminho que escrevesse a constante passaria.
    """
    _banco_de_arquivo(tmp_path, monkeypatch)
    cliente = TestClient(build_app())
    modo = "neg_c18"

    padrao = _executa_ate_o_fim(cliente, modo, estudo="u6a_tukey")
    pedido = _executa_ate_o_fim(
        cliente,
        modo,
        estudo="u6a_fisher",
        analise_json=json.dumps({"post_hoc": "fisher"}),
    )
    assert padrao["anova"]["post_hoc_metodo"] == "tukey"
    assert pedido["anova"]["post_hoc_metodo"] == "fisher", pedido["anova"]


def _matriz_em_rampa(grupos: list[str]) -> Any:
    """Matriz sintética com um rótulo clínico por linha — para a GUARDA, nunca para contagem.

    Chamava-se `_matriz_de_grupos`, como a de ruído lá em cima, e a sombreava: os testes da
    PERMANOVA e os da correlação, escritos para o ruído, corriam sobre esta rampa, que é a
    estrutura plantada que a docstring de lá diz que eles não podem ter. O nome novo diz o que
    ela é, e só as três chamadas de lista — as da guarda — a usam.

    Contagem observada não se testa sobre matriz de factory; o que se mede aqui é o
    comportamento da guarda perante uma população que ela recusa, e isso é propriedade da função.
    """
    from refino.domain.matrix import RefinoMatrix
    from refino.domain.metadata import SampleMetadata

    valores = np.arange(len(grupos) * 2, dtype=np.float64).reshape(len(grupos), 2)
    return RefinoMatrix(
        values=valores,
        feature_names=("f0", "f1"),
        samples=tuple(
            SampleMetadata(name=f"S{i}", injection_order=i + 1, sample_type="sample", group=g)
            for i, g in enumerate(grupos)
        ),
        mode="sintetico",
    )


def test_u6_analises_a_matriz_de_UM_grupo_nao_derruba_a_execucao() -> None:  # noqa: N802
    """**A ANOVA é família OPCIONAL, e uma matriz que não a sustenta não é erro da execução.**

    `statistics.anova._populacao` levanta `ValueError` com menos de dois grupos de estudo. Deixá-lo
    subir derrubaria correção, pipeline e relatório de qualidade por causa de um pedido que vem
    ligado por padrão — o que restringe o que a ferramenta pode fazer.

    As duas direções na mesma função: com dois grupos a raiz vem, com um ela não vem,
    e a etapa é anunciada nas duas.
    """
    from refino.api.rotas import corre_as_analises
    from refino.domain.analysis import AnalysisSpec

    avisos: list[int] = []
    de_dois_grupos = _matriz_em_rampa(["A", "A", "A", "B", "B", "B"])
    dois = corre_as_analises(
        AnalysisSpec(),
        matriz=de_dois_grupos,
        imputadas=_sem_imputadas(de_dois_grupos),
        seed=42,
        marca=lambda: avisos.append(1),
        cancelado=lambda: False,
    )
    assert dois.anova is not None, "âncora: com dois grupos a ANOVA TEM de correr"
    de_dois = len(avisos)

    de_um_grupo = _matriz_em_rampa(["A", "A", "A", "A", "A", "A"])
    um = corre_as_analises(
        AnalysisSpec(),
        matriz=de_um_grupo,
        imputadas=_sem_imputadas(de_um_grupo),
        seed=42,
        marca=lambda: avisos.append(1),
        cancelado=lambda: False,
    )
    assert um.anova is None, "com um grupo só a ANOVA não corre, e a raiz fica `None`"
    assert len(avisos) - de_dois == de_dois, (
        "a barra deixou de andar quando a ANOVA não correu: a etapa continua anunciada, e etapa "
        "anunciada sem aviso é a barra parada"
    )


def test_u6_analises_um_grupo_COM_POUCAS_AMOSTRAS_tambem_nao_derruba() -> None:  # noqa: N802
    """A segunda metade da guarda: o grupo abaixo do mínimo de observações.

    **Ela é irmã da primeira e não a mesma**, e é por isso que tem cenário próprio:
    `_populacao` recusa por DOIS motivos — grupos a menos e grupo pequeno de mais — e um predicado
    que só olhasse o primeiro deixaria o segundo levantar em produção.
    """
    from refino.api.rotas import corre_as_analises
    from refino.domain.analysis import AnalysisSpec

    m = _matriz_em_rampa(["A", "A", "A", "B"])  # `B` tem UMA amostra, e o mínimo são duas
    assert len({a.group for a in m.samples}) >= 2, "âncora: há dois grupos, e a recusa é do tamanho"
    resultado = corre_as_analises(
        AnalysisSpec(),
        matriz=m,
        imputadas=_sem_imputadas(m),
        seed=42,
        marca=lambda: None,
        cancelado=lambda: False,
    )
    assert resultado.anova is None


# ---------------------------------------------------------------------------
# A correlação aterra em `corre_as_analises`, e as DUAS raízes são separadas
# ---------------------------------------------------------------------------


def test_u12_a_familia_da_correlacao_CALCULA_e_a_bandeira_controla_o_calculo() -> None:  # noqa: N802
    """**O par causal do fio: com `correlation` ligado a raiz vem, desligado não vem.**

    É o que separa esta família da PERMANOVA, cuja bandeira só move a etapa anunciada: aqui a rota
    recebe `AnalysisSpec` e a bandeira decide se o cálculo acontece.

    **E a âncora de que o par não é vácuo vem antes:** a matriz tem features com variação, senão as
    duas execuções devolveriam `None` pelo mesmo motivo e o teste passaria sobre o ramo errado.
    """
    from dataclasses import replace

    from refino.api.rotas import corre_as_analises

    matriz = _matriz_de_grupos(("a",) * 6 + ("b",) * 6)
    assert matriz.n_features >= 2, "âncora: há features que cheguem para uma matriz de correlação"

    ligada = corre_as_analises(
        replace(PADRAO_DA_ANALISE, correlation=True),
        matriz=matriz,
        imputadas=_sem_imputadas(matriz),
        seed=42,
        marca=lambda: None,
        cancelado=lambda: False,
    )
    desligada = corre_as_analises(
        replace(PADRAO_DA_ANALISE, correlation=False),
        matriz=matriz,
        imputadas=_sem_imputadas(matriz),
        seed=42,
        marca=lambda: None,
        cancelado=lambda: False,
    )
    assert ligada.correlacao is not None, "a família vem ligada por padrão (KD11) e não calculou"
    assert desligada.correlacao is None, "a bandeira não controla o cálculo"
    assert ligada.correlacao.metodo == "pearson"
    assert len(ligada.correlacao.valores) == len(ligada.correlacao.features)


def test_u12_a_busca_por_padrao_SALTA_sem_referencia_e_NAO_desliga_o_mapa() -> None:  # noqa: N802
    """**`pattern_reference` a `None` salta a busca sem desligar o mapa — o par que o prova.**

    As duas análises partilham o coeficiente e não o alvo. Uma bandeira só para as duas tornaria
    impossível pedir o mapa sem a busca, que é o caso comum — e é por isso que as raízes são duas.
    """
    from dataclasses import replace

    from refino.api.rotas import corre_as_analises

    matriz = _matriz_de_grupos(("a",) * 6 + ("b",) * 6)
    correr = lambda analise: corre_as_analises(  # noqa: E731
        analise,
        matriz=matriz,
        imputadas=_sem_imputadas(matriz),
        seed=42,
        marca=lambda: None,
        cancelado=lambda: False,
    )

    sem = correr(replace(PADRAO_DA_ANALISE, correlation=True, pattern_reference=None))
    assert sem.correlacao is not None, "o mapa tem de correr sem a busca"
    assert sem.busca_por_padrao is None, "sem referência a busca não corre"

    com = correr(
        replace(PADRAO_DA_ANALISE, correlation=True, pattern_reference=matriz.feature_names[0])
    )
    assert com.correlacao is not None, "a busca não pode desligar o mapa"
    assert com.busca_por_padrao is not None
    assert com.busca_por_padrao.features[0] == matriz.feature_names[0], (
        "a feature de referência não saiu no topo: a correlação dela consigo própria é 1"
    )
    assert com.busca_por_padrao.ordem_dos_grupos == ["a", "b"], "a ordem canônica não viajou (OQ2)"


def test_u12_referencia_INEXISTENTE_devolve_422_e_nao_silencio() -> None:  # noqa: N802
    """*"Não pedi"* e *"pedi e escrevi o nome errado"* têm consertos opostos.

    `None` para os dois deixaria o profissional à espera de uma figura que nunca vai aparecer, sem
    nada a dizer porquê. A recusa nomeia as features disponíveis.
    """
    from dataclasses import replace

    from fastapi import HTTPException

    from refino.api.rotas import corre_as_analises

    matriz = _matriz_de_grupos(("a",) * 6 + ("b",) * 6)
    with pytest.raises(HTTPException) as erro:
        corre_as_analises(
            replace(PADRAO_DA_ANALISE, correlation=True, pattern_reference="nao_existe"),
            matriz=matriz,
            imputadas=_sem_imputadas(matriz),
            seed=42,
            marca=lambda: None,
            cancelado=lambda: False,
        )
    assert erro.value.status_code == 422
    assert "nao_existe" in str(erro.value.detail)
    assert matriz.feature_names[0] in str(erro.value.detail), (
        "a recusa não nomeia nenhuma feature disponível, e quem a lê fica a adivinhar"
    )


def test_u12_a_matriz_que_NAO_sustenta_a_correlacao_nao_derruba_a_execucao() -> None:  # noqa: N802
    """A irmã de `_a_populacao_sustenta_a_anova`: família opcional não derruba a execução inteira.

    **As duas direções na mesma função:** com features utilizáveis a raiz vem, com a matriz toda
    constante ela não vem — e a etapa é anunciada nas duas.
    """
    from dataclasses import replace

    from refino.api.rotas import corre_as_analises
    from refino.domain.matrix import RefinoMatrix

    boa = _matriz_de_grupos(("a",) * 6 + ("b",) * 6)
    constante = RefinoMatrix(
        values=np.full_like(boa.values, 3.0),
        feature_names=boa.feature_names,
        samples=boa.samples,
        mode=boa.mode,
    )
    avisos: list[int] = []
    pedida = replace(PADRAO_DA_ANALISE, correlation=True)
    correr = lambda m: corre_as_analises(  # noqa: E731
        pedida,
        matriz=m,
        imputadas=_sem_imputadas(m),
        seed=42,
        marca=lambda: avisos.append(1),
        cancelado=lambda: False,
    )

    assert correr(boa).correlacao is not None, "âncora: a matriz boa sustenta a correlação"
    de_uma = len(avisos)
    assert correr(constante).correlacao is None, "matriz constante não tem correlação definida"
    assert len(avisos) - de_uma == de_uma, (
        "a barra deixou de andar quando a correlação não correu: etapa anunciada sem aviso é a "
        "barra parada"
    )


# ---------------------------------------------------------------------------
# As duas famílias particionais aterram em `corre_as_analises`
# ---------------------------------------------------------------------------


def _matriz_particional(n_amostras: int = 8, n_features: int = 6) -> Any:
    """Uma matriz com VARIAÇÃO em todas as colunas — e é essa a âncora do par causal.

    `_matriz_em_rampa` acima constrói um `arange`, que serve à guarda de população; o que se
    precisa aqui é de uma matriz que o K-means e o SOM **aceitem**, para a ausência das raízes medir
    a **bandeira** e não a guarda. Sem esta âncora, as duas execuções do par devolveriam `None` pelo
    mesmo motivo e o teste passaria sobre o ramo errado.
    """
    from refino.domain.matrix import RefinoMatrix
    from refino.domain.metadata import SampleMetadata

    gerador = np.random.default_rng(20260918)
    valores = gerador.standard_normal((n_amostras, n_features))
    valores[: n_amostras // 2] += 5.0
    return RefinoMatrix(
        values=valores,
        feature_names=tuple(f"f{j}" for j in range(n_features)),
        samples=tuple(
            SampleMetadata(
                name=f"S{i}",
                injection_order=i + 1,
                sample_type="sample",
                group="A" if i < n_amostras // 2 else "B",
            )
            for i in range(n_amostras)
        ),
        mode="sintetico",
    )


def _corre(analise: Any, matriz: Any, *, seed: int = 42) -> Any:
    from refino.api.rotas import corre_as_analises

    return corre_as_analises(
        analise,
        matriz=matriz,
        imputadas=_sem_imputadas(matriz),
        seed=seed,
        marca=lambda: None,
        cancelado=lambda: False,
    )


def test_u11_as_DUAS_bandeiras_particionais_controlam_o_calculo_SEPARADAMENTE() -> None:  # noqa: N802
    """**O par causal do fio, e ele é DUPLO: cada bandeira move a sua raiz e só a sua.**

    É a metade que a separação de campos existe para permitir — pedir o K-means sem o SOM é o caso
    comum, porque as duas vêm desligadas por padrão e custam coisas diferentes. Um campo só faria a
    ausência de uma parecer implicar a da outra, e este teste não conseguiria distinguir as duas.

    A âncora de não-vacuidade vem primeiro: a matriz sustenta as duas famílias, senão as
    quatro leituras seriam `None` pela guarda de população e o par mediria outra coisa.
    """
    from dataclasses import replace

    matriz = _matriz_particional()
    de_referencia = _corre(replace(PADRAO_DA_ANALISE, kmeans=True, som=True), matriz)
    assert de_referencia.kmeans is not None and de_referencia.som is not None, (
        "âncora: a matriz não sustenta as duas famílias, e o par mediria a guarda de população"
    )

    so_kmeans = _corre(replace(PADRAO_DA_ANALISE, kmeans=True, som=False), matriz)
    assert so_kmeans.kmeans is not None and so_kmeans.som is None

    so_som = _corre(replace(PADRAO_DA_ANALISE, kmeans=False, som=True), matriz)
    assert so_som.kmeans is None and so_som.som is not None

    nenhuma = _corre(replace(PADRAO_DA_ANALISE, kmeans=False, som=False), matriz)
    assert nenhuma.kmeans is None and nenhuma.som is None

    assert PADRAO_DA_ANALISE.kmeans is False and PADRAO_DA_ANALISE.som is False, (
        "as duas vêm desligadas por padrão (KD11) porque custam — o padrão não se muda aqui"
    )


def test_u11_as_duas_raizes_atravessam_o_fio_ALINHADAS_com_as_amostras() -> None:  # noqa: N802
    """O que viaja é um inteiro por amostra, e ele indexa o conjunto que o mesmo objeto declara."""
    from dataclasses import replace

    matriz = _matriz_particional()
    resultado = _corre(replace(PADRAO_DA_ANALISE, kmeans=True, kmeans_k=2, som=True), matriz)
    assert resultado.kmeans is not None and resultado.som is not None

    kmeans = resultado.kmeans
    assert len(kmeans.rotulos) == len(kmeans.grupos) == len(kmeans.amostras) == matriz.n_samples
    assert set(kmeans.rotulos) <= set(range(kmeans.k))
    assert kmeans.semente != 42, "a semente do corpo é a GLOBAL e não a derivada"

    som = resultado.som
    assert len(som.unidades) == len(som.grupos) == len(som.amostras) == matriz.n_samples
    assert set(som.unidades) <= set(range(som.linhas * som.colunas))
    assert som.semente != kmeans.semente, "as duas famílias derivaram o MESMO fluxo"


def test_u11_o_k_ACIMA_do_numero_de_amostras_volta_como_422_e_nao_como_None() -> None:  # noqa: N802
    """*"Não pedi"* e *"pedi e o pedido não cabe"* têm consertos opostos.

    É o desenho de `_busca_ou_nada` com a referência desconhecida. `None` para os dois deixaria
    alguém à espera de uma figura que nunca vai aparecer, sem nada a dizer porquê.
    """
    from dataclasses import replace

    from fastapi import HTTPException

    matriz = _matriz_particional(n_amostras=6)
    with pytest.raises(HTTPException) as capturado:
        _corre(replace(PADRAO_DA_ANALISE, kmeans=True, kmeans_k=matriz.n_samples + 1), matriz)
    assert capturado.value.status_code == 422
    assert str(matriz.n_samples) in str(capturado.value.detail)

    # A outra ponta do par: `k == n` é legítimo e atravessa.
    no_limite = _corre(replace(PADRAO_DA_ANALISE, kmeans=True, kmeans_k=matriz.n_samples), matriz)
    assert no_limite.kmeans is not None
    assert sorted(no_limite.kmeans.rotulos) == list(range(matriz.n_samples))


def test_u15_a_TUPLA_de_classificadores_controla_cada_raiz_SEPARADAMENTE() -> None:  # noqa: N802
    """**O par causal do fio, e ele é DUPLO: cada membro da tupla move a sua raiz e só a sua.**

    É a metade que a separação de campos existe para permitir — pedir só o SVM é o caso comum,
    porque cada um custa minutos. Uma lista faria a ausência de um parecer implicar a do outro, e
    este teste não conseguiria distinguir as duas.

    A âncora de não-vacuidade vem primeiro: a matriz sustenta a família, senão as
    leituras seriam recusa e o par mediria a guarda de população em vez da tupla.

    **Correm 200 permutações, que é o padrão de produção.** Não se encolhe aqui: o eixo que muda
    **maquinaria** é `kind`, e este teste exercita os dois — reduzir o número de sorteios mediria o
    mesmo caminho mais depressa, e é o `kind` que decide qual estimador se constrói.
    """
    from dataclasses import replace

    matriz = _matriz_particional(n_amostras=12, n_features=6)

    so_svm = _corre(replace(PADRAO_DA_ANALISE, classifiers=("svm",), classifier_folds=3), matriz)
    assert so_svm.svm is not None, (
        "âncora: nem o SVM correu, então a ausência abaixo mede a guarda de população e não a tupla"
    )
    assert so_svm.random_forest is None, (
        "pedi só o SVM e a floresta correu: a tupla não controla as raízes separadamente"
    )

    nenhum = _corre(replace(PADRAO_DA_ANALISE, classifiers=(), classifier_folds=3), matriz)
    assert nenhum.random_forest is None and nenhum.svm is None, (
        "a tupla vazia é o PADRÃO de KD11 — nenhum classificador corre, e o custo é o argumento"
    )


def test_u15_a_FLORESTA_e_despachada_para_a_raiz_dela_e_nao_para_a_do_svm(  # noqa: N802
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A metade da floresta, com o modelo substituído por um espião — e o motivo é o CUSTO.

    **A pergunta deste teste é sobre o DESPACHO e não sobre o modelo:** cada membro da tupla cai na
    raiz dele, e os dois pedidos juntos preenchem as duas. Correr a floresta de verdade custa **~100
    s** aqui — 201 validações cruzadas de três dobras, cada uma com trezentas árvores —, e o que
    isso compraria é uma medida de qualidade que `tests/unit/test_classificadores.py` já faz sobre
    dado com e sem sinal.

    O espião não descarta argumentos, e isso é afirmado e não suposto. Um duplo que
    engolisse `kind` faria as duas asserções abaixo passarem sobre um despacho que nunca aconteceu —
    a asserção de ausência satisfeita por vacuidade. Ele **regista** o que recebeu, e a primeira
    asserção é sobre esse registo.
    """
    from dataclasses import replace

    from refino.statistics.classificadores import ClassifierResult

    pedidos: list[str] = []

    def espia(matriz: Any, analise: Any, **campos: Any) -> ClassifierResult:
        pedidos.append(campos["kind"])
        return ClassifierResult(
            kind=campos["kind"],
            groups=("A", "B"),
            class_counts=(6, 6),
            balanced_accuracy=0.5,
            accuracy=0.5,
            macro_f1=0.5,
            recall=(0.5, 0.5),
            confusion=((3, 3), (3, 3)),
            permutation_pvalue=0.5,
            chance_level=0.5,
            importances=(0.5, 0.5),
            importance_kind="espião",
            feature_names=("f0", "f1"),
            cv_folds=analise.classifier_folds,
            n_permutations=1,
            seed=campos["seed"],
            n_samples=12,
            n_features_used=2,
            n_features_total=6,
            column_digest="0" * 64,
        )

    monkeypatch.setattr("refino.api.rotas.classify", espia)
    matriz = _matriz_particional(n_amostras=12, n_features=6)
    ambos = _corre(
        replace(PADRAO_DA_ANALISE, classifiers=("random_forest", "svm"), classifier_folds=3), matriz
    )

    assert pedidos == ["random_forest", "svm"], (
        f"o espião recebeu {pedidos}: ou o despacho não passou o `kind`, ou não correu os dois — e "
        "nos dois casos as asserções abaixo passariam sobre um despacho que não aconteceu"
    )
    assert ambos.random_forest is not None and ambos.random_forest.kind == "random_forest"
    assert ambos.svm is not None and ambos.svm.kind == "svm"


def test_u15_o_p_de_permutacao_ATRAVESSA_o_fio_ao_lado_da_medida() -> None:  # noqa: N802
    """Na resposta, não há caminho que publique a medida sem o `p`.

    **O esquema recusa a metade na fronteira** — `permutation_pvalue` é campo obrigatório de
    `ResumoDoClassificador` —, e esta é a metade de runtime: o que a rota de facto monta traz os
    dois preenchidos, mais o regime que os torna comparáveis.
    """
    from dataclasses import replace

    matriz = _matriz_particional(n_amostras=12, n_features=6)
    resumo = _corre(
        replace(PADRAO_DA_ANALISE, classifiers=("svm",), classifier_folds=3), matriz
    ).svm
    assert resumo is not None

    assert math.isfinite(resumo.balanced_accuracy)
    assert math.isfinite(resumo.permutation_pvalue)
    assert resumo.n_permutations >= 1 and resumo.cv_folds == 3
    assert resumo.chance_level == pytest.approx(1 / len(resumo.grupos))
    assert resumo.nome == "SVM" and resumo.kind == "svm"
    # A procedência do recorte, que a figura escreve.
    assert resumo.matriz == "processada"
    assert resumo.n_features_usadas <= resumo.n_features_totais
    assert str(resumo.n_features_usadas) in resumo.amostragem


def test_u15_o_pedido_INSATISFAZIVEL_volta_como_422_e_nao_como_None() -> None:  # noqa: N802
    """*"Não pedi"* e *"pedi e o pedido não cabe"* têm consertos opostos.

    É o desenho de `_kmeans_ou_422` e de `_busca_ou_nada`. `None` para os dois deixaria alguém à
    espera de minutos por uma figura que nunca vai aparecer, sem nada a dizer porquê — e aqui isso
    é pior, porque a família custa minutos.

    **A outra ponta do par está na mesma função:** com dobras que cabem, a mesma matriz atravessa.
    """
    from dataclasses import replace

    from fastapi import HTTPException

    matriz = _matriz_particional(n_amostras=12, n_features=6)
    with pytest.raises(HTTPException) as capturado:
        _corre(replace(PADRAO_DA_ANALISE, classifiers=("svm",), classifier_folds=9), matriz)
    assert capturado.value.status_code == 422
    assert "cv_folds=9" in str(capturado.value.detail)

    cabe = _corre(replace(PADRAO_DA_ANALISE, classifiers=("svm",), classifier_folds=3), matriz)
    assert cabe.svm is not None


def test_u11_a_matriz_que_NAO_sustenta_as_duas_familias_nao_derruba_a_execucao() -> None:  # noqa: N802
    """Famílias OPCIONAIS: a guarda de população devolve `None` em vez de derrubar tudo.

    **As duas direções na mesma função**, e a etapa continua a ser anunciada nas duas — etapa
    anunciada sem aviso é a barra parada.
    """
    from dataclasses import replace

    from refino.api.rotas import corre_as_analises
    from refino.domain.matrix import RefinoMatrix
    from refino.domain.metadata import SampleMetadata

    pedido = replace(PADRAO_DA_ANALISE, kmeans=True, kmeans_k=1, som=True)
    avisos: list[int] = []

    def correr(matriz: Any) -> Any:
        return corre_as_analises(
            pedido,
            matriz=matriz,
            imputadas=_sem_imputadas(matriz),
            seed=42,
            marca=lambda: avisos.append(1),
            cancelado=lambda: False,
        )

    boa = correr(_matriz_particional())
    assert boa.kmeans is not None and boa.som is not None
    da_boa = len(avisos)

    uma_amostra = RefinoMatrix(
        values=np.arange(4, dtype=np.float64).reshape(1, 4),
        feature_names=("f0", "f1", "f2", "f3"),
        samples=(SampleMetadata(name="S0", injection_order=1, sample_type="sample", group="A"),),
        mode="sintetico",
    )
    magra = correr(uma_amostra)
    assert magra.kmeans is None and magra.som is None
    assert len(avisos) - da_boa == da_boa, (
        "a barra deixou de andar quando as duas famílias não correram: etapa anunciada sem aviso "
        "é a barra parada"
    )


def test_u11_a_MESMA_semente_reproduz_as_duas_raizes_atraves_do_fio() -> None:  # noqa: N802
    """Determinismo do lado da rota: a semente que chega é a `PipelineSpec.seed` da execução.

    **E a segunda metade é o par: com outra semente o SOM diverge.** O K-means fica fora dessa
    metade de propósito — sobre dado separado por construção ele converge para a mesma partição a
    partir de qualquer arranque, e afirmá-lo aqui seria uma planta que não separa. Essa
    metade está em `tests/unit/test_clustering.py`, sobre ruído.
    """
    from dataclasses import replace

    matriz = _matriz_particional()
    pedido = replace(PADRAO_DA_ANALISE, kmeans=True, kmeans_k=2, som=True)

    primeira = _corre(pedido, matriz, seed=42)
    segunda = _corre(pedido, matriz, seed=42)
    assert primeira.kmeans is not None and segunda.kmeans is not None
    assert primeira.som is not None and segunda.som is not None
    assert primeira.kmeans.rotulos == segunda.kmeans.rotulos
    assert primeira.som.unidades == segunda.som.unidades
    assert primeira.som.erro_de_quantizacao == segunda.som.erro_de_quantizacao

    outra = _corre(pedido, matriz, seed=20260918)
    assert outra.som is not None
    assert outra.som.semente != primeira.som.semente, "a semente global não chegou ao derivador"
