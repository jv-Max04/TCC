"""As rotas de artefato.

**A asserção é byte-identidade contra a CLI**, e é a mais forte disponível: não há tolerância a
negociar. Ela é verdadeira **por construção** porque a rota serializa pelo mesmo escritor —
`write_toserrf`, `write_metaboanalyst` —, e o teste é o que impede que uma segunda serialização
apareça sem ninguém notar.

**O que estes testes NÃO alcançam.** Divergência que os **dois lados compartilhem**: se o escritor
tiver defeito, a rota e a CLI o têm igual. É critério de **paridade**, não de correção — e dizê-lo
é o que impede a conclusão errada de que a saída está certa porque os dois lados concordam.
"""

from __future__ import annotations

import os
import subprocess
import sysconfig
import tempfile
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pytest
from fastapi.testclient import TestClient

from refino.api.app import build_app
from refino.cli import EXIT_UNAVAILABLE, EXIT_USAGE
from refino.cli import main as cli_main
from refino.correction.none import NoneCorrector
from refino.domain.matrix import RefinoMatrix
from refino.domain.spec import PipelineSpec
from refino.io.readers import load_matrix
from refino.io.writers import write_metaboanalyst
from refino.preprocessing.pipeline import (
    EmptyMatrixError,
    process_matrix,
    requests_preprocessing,
)
from tests.conftest import measurements_path, sample_list_path
from tests.reclamacao import reclama

pytestmark = pytest.mark.integration

# O ponto de entrada onde o ambiente o instala, e não o caminho do Linux. Este teste dizia
# `.venv/bin/refino`, e no Windows o executável é `.venv/Scripts/refino.exe`: o arquivo não
# existia, o teste PULAVA pelo regime `refino_ausente`, e a byte-identidade entre CLI e API não
# era conferida nesta máquina — com o `exportar` a sair com código 1 por baixo do pulo.
REFINO = Path(sysconfig.get_path("scripts")) / ("refino.exe" if os.name == "nt" else "refino")
MODOS = ("pos_hilic", "pos_c18", "neg_hilic", "neg_c18")


@pytest.fixture(scope="module")
def cliente() -> TestClient:
    return TestClient(build_app())


def _arquivos(mode: str) -> dict[str, tuple[str, bytes, str]]:
    return {
        "measurements": (
            f"measurements_{mode}.csv",
            measurements_path(mode).read_bytes(),
            "text/csv",
        ),
        "sample_list": (
            f"Sample_List_{mode}.csv",
            sample_list_path(mode).read_bytes(),
            "text/csv",
        ),
    }


def _pela_cli(mode: str, *argumentos: str) -> bytes:
    """Roda a CLI instalada e devolve os bytes que ela escreveu.

    **Pelo ponto de entrada instalado e não por `main(argv)`**, porque o que se compara é o que o
    usuário obtém — e `[project.scripts]` é o caminho dele.
    """
    if not REFINO.is_file():
        pytest.skip("refino_ausente: o ponto de entrada instalado não existe neste ambiente")
    with tempfile.TemporaryDirectory() as temporario:
        alvo = Path(temporario) / "cli.csv"
        subprocess.run(
            [
                str(REFINO),
                *argumentos,
                "--measurements",
                str(measurements_path(mode)),
                "--sample-list",
                str(sample_list_path(mode)),
                "--mode",
                mode,
                "--output",
                str(alvo),
            ],
            check=True,
            capture_output=True,
        )
        return alvo.read_bytes()


@pytest.mark.parametrize("mode", MODOS)
def test_ca_12_12_corrigir_e_byte_identico_a_cli(cliente: TestClient, mode: str) -> None:
    """`none` nos quatro modos: a rota devolve exactamente o que `refino corrigir` escreve."""
    resposta = reclama(
        cliente, "/matrizes/corrigir", files=_arquivos(mode), data={"mode": mode, "method": "none"}
    )

    assert resposta.status_code == 200, resposta.text
    assert resposta.headers["content-type"].startswith("text/csv")
    assert resposta.content == _pela_cli(mode, "corrigir", "--method", "none")


@pytest.mark.slow
def test_ca_12_12_corrigir_com_loess_e_byte_identico_a_cli(cliente: TestClient) -> None:
    """O caminho que de facto corrige, com a mesma semente. `slow` porque leva ~12 s."""
    resposta = reclama(
        cliente,
        "/matrizes/corrigir",
        files=_arquivos("neg_c18"),
        data={"mode": "neg_c18", "method": "loess", "seed": 42},
    )

    assert resposta.status_code == 200, resposta.text
    assert resposta.content == _pela_cli("neg_c18", "corrigir", "--method", "loess", "--seed", "42")


@pytest.mark.slow
def test_ca_12_12_corrigir_com_serrf_e_byte_identico_a_cli(cliente: TestClient) -> None:
    """O caso do SERRF liberado, e ele é o mais exigente dos três.

    Enquanto o `serrf` era bloqueado, este caso não existia: a byte-identidade cobria `none` nos
    quatro modos e `loess` num. Liberado o método nos dois lados, a paridade byte a byte passa a
    ter uma terceira coluna — e ela é a única **estocástica**.

    **É por isso que este caso vale mais que os outros dois.** `none` e `loess` são
    determinísticos por construção, então a igualdade deles mede serialização. Aqui ela mede
    três coisas que podiam divergir sem nada acusar:

    1. a **semente** chegar igual pelos dois caminhos;
    2. o `n_jobs` ser o mesmo — `build_corrector` e `_corretor_ou_422` passam `-1` os dois, e a
       docstring do segundo **afirma** essa paridade, e só um teste a confere;
    3. a semente ser derivada **por unidade** e não consumida na ordem em que o `joblib`
       entregar as features. Se fosse global, `n_jobs=-1` tornaria a saída dependente do
       escalonamento e este teste reprovaria **de forma intermitente** — que é pior que reprovar
       sempre.

    **`slow` por custo medido:** ~86 s por ajuste no `neg_c18` de 2.184 features, e são dois —
    a rota e o subprocesso da CLI. Medido antes de o teste existir: `sha256` idêntico,
    2.367.156 bytes dos dois lados.
    """
    resposta = reclama(
        cliente,
        "/matrizes/corrigir",
        files=_arquivos("neg_c18"),
        data={"mode": "neg_c18", "method": "serrf", "seed": 42},
    )

    assert resposta.status_code == 200, resposta.text
    assert resposta.content == _pela_cli("neg_c18", "corrigir", "--method", "serrf", "--seed", "42")


def test_ca_12_12_processar_e_byte_identico_a_cli(cliente: TestClient) -> None:
    """O pipeline pós-correção, com os mesmos parâmetros explícitos dos dois lados."""
    campos = {
        "mode": "neg_c18",
        "rsd_cutoff": 0.30,
        "missing_threshold": 0.5,
        "imputation": "one_fifth_min",
        "transformation": "log10",
        "scaling": "pareto",
    }
    resposta = reclama(cliente, "/matrizes/processar", files=_arquivos("neg_c18"), data=campos)

    assert resposta.status_code == 200, resposta.text
    assert resposta.content == _pela_cli(
        "neg_c18",
        "processar",
        "--rsd-cutoff",
        "0.30",
        "--missing-threshold",
        "0.5",
        "--imputation",
        "one_fifth_min",
        "--transformation",
        "log10",
        "--scaling",
        "pareto",
    )


# ---------------------------------------------------------------------------
# O corte passa a agir sobre matriz CORRIGIDA
# ---------------------------------------------------------------------------
#
# **O vão que isto fecha:** nenhum caminho produzia matriz corrigida *e* processada.
# `/execucoes` corrigia e gravava **sem corte**; `/matrizes/processar` cortava e **não** corrigia.
#
# **MEDIDO EM 17/08/2026, pela rota, sobre o `pos_hilic` completo** — 6.035 features e 68 amostras,
# corte `0,30`, `imputation=one_fifth_min`, `seed=42`, `rsd_variant=outlier_removed` (o padrão) e
# `missing_threshold` ausente:
#
# | `method` | Features sobreviventes | Custo |
# |---|---|---|
# | `none` | **758** | 0,6 s |
# | `serrf` | **1107** | 199,8 s |
# | `loess` | **2293** | 66,7 s |
#
# Os três batem com uma medição independente. **O download processado passa a devolver matriz
# diferente da de antes**, e é por isso que a mudança exigiu decisão: quem baixava 758 features
# passa a baixar 1107 com o mesmo corte e o corretor que a tela já mostrava.
#
# Remedido em 03/11/2026, pela mesma rota e com os mesmos parâmetros, no Windows com 12
# núcleos, no commit `e039743`:
#
# | `method` | Features sobreviventes | Custo |
# |---|---|---|
# | `none` | **758** | 0,5 s |
# | `serrf` | **2957** | 198,4 s |
# | `loess` | **2353** | 38,2 s |
#
# O `none` não mudou, porque a matriz crua não mudou. O `serrf` mudou com a floresta do `ranger`,
# causa atribuída por intervenção, e deu 2957 em três corridas, em duas plataformas. O `loess`
# também mudou. **A ordem entre os dois corretores inverteu**, e isso não diz nada sobre qual é
# melhor, pelo motivo do parágrafo seguinte.
#
# **A comparação é entre CORTES DO MESMO CORRETOR quando o assunto é qualidade**:
# o LOESS zera o RSD in-sample por construção, então o `2293` NÃO diz que ele é melhor. O que a
# asserção abaixo usa a contagem para provar é outra coisa, e ela é válida: que a correção
# **alcançou a etapa do corte**. Sem alcançar, os três números seriam o mesmo.


@pytest.mark.slow
def test_a_correcao_alcanca_o_corte_no_download_processado(cliente: TestClient) -> None:
    """Com `serrf` sobrevivem mais features que com `none`.

    **`slow` e sobre o `pos_hilic` COMPLETO, e as duas escolhas são medidas.** O custo é 199,8 s de
    SERRF, que é o mesmo regime dos dois casos `slow` vizinhos da byte-identidade. E os números 758
    e 2957 são propriedade **daquela** população: uma fixture pequena não os reproduz, e uma matriz
    com poucas features tornaria a desigualdade dependente de quais features entraram.

    **A desigualdade é a propriedade; os dois números exatos são a âncora.** Só a desigualdade
    passaria com um corte que não corta e uma correção que não corrige, desde que a segunda mexesse
    em qualquer coisa. Só os números exatos reprovariam por deriva legítima do SERRF sem dizer que a
    propriedade se manteve. As duas juntas dizem qual das duas coisas quebrou.

    **Divergência aqui é ACHADO e não conserto**: o número foi medido duas vezes, por
    dois caminhos, com a população declarada. Se ele mudar, o que se faz é remedir e relatar.

    E foi o que aconteceu com o 1107 da primeira medição. Ele virou 2957 com a floresta do
    `ranger`, a causa foi atribuída por intervenção, e o 2957 foi remedido em três corridas, em
    duas plataformas. A âncora passou a 2957 com a causa já conhecida. A
    troca não foi feita para o teste passar.
    """
    base = {
        "mode": "pos_hilic",
        "rsd_cutoff": 0.30,
        "imputation": "one_fifth_min",
        "seed": 42,
    }
    sem = reclama(cliente, "/matrizes/processar", files=_arquivos("pos_hilic"), data=base)
    com = reclama(
        cliente,
        "/matrizes/processar",
        files=_arquivos("pos_hilic"),
        data={**base, "method": "serrf"},
    )

    assert (sem.status_code, com.status_code) == (200, 200), (sem.text[:200], com.text[:200])

    # `write_toserrf` escreve quatro linhas de cabeçalho — `batch`, `sampleType`, `time`, `label` —
    # e depois uma linha por feature. A contagem é de FEATURES e não de linhas.
    n_sem = len(sem.content.decode("utf-8").splitlines()) - 4
    n_com = len(com.content.decode("utf-8").splitlines()) - 4

    assert n_com > n_sem, (
        f"a correção não alcançou o corte: {n_com} features com `serrf` contra {n_sem} com `none`, "
        "no mesmo corte 0,30 do `pos_hilic`"
    )
    assert (n_sem, n_com) == (758, 2957), (
        f"remedido em 03/11/2026 no `pos_hilic` (6.035 features, 68 amostras) a 0,30: 758 com "
        f"`none` e 2957 com `serrf` (era 1107 antes da floresta do `ranger`). Veio "
        f"{n_sem} e {n_com}. Isto é ACHADO: remedir e relatar, nunca "
        "ajustar o número esperado (regra 2 do README)"
    )


def test_sem_method_a_rota_processa_como_antes(cliente: TestClient) -> None:
    """**A compatibilidade de quem chama a rota direto**, e ela é par causal e não afirmação.

    O padrão `none` foi escolhido para não quebrar quem já usa a rota, e a asserção que o prova é a
    igualdade entre **omitir** `method` e passar `method=none` — não a existência de um padrão na
    assinatura, que é o que uma leitura do código diria.

    **`NoneCorrector` devolve a matriz inalterada em VALOR e nova em identidade**, então a etapa de
    correção acrescentada nesta rota é invisível na serialização. A âncora de que a comparação não é
    entre dois vazios está antes: o artefato começa com o cabeçalho do formato `toserrf`.
    """
    base = {"mode": "neg_c18", "rsd_cutoff": 0.30, "imputation": "one_fifth_min", "seed": 42}
    omitido = reclama(cliente, "/matrizes/processar", files=_arquivos("neg_c18"), data=base)
    explicito = reclama(
        cliente, "/matrizes/processar", files=_arquivos("neg_c18"), data={**base, "method": "none"}
    )

    assert (omitido.status_code, explicito.status_code) == (200, 200), omitido.text[:200]
    assert omitido.content.startswith(b"NA,batch,"), (
        "âncora: o artefato não tem a forma do `toserrf`"
    )
    assert len(omitido.content) > 1000, "âncora: o artefato é vazio e a igualdade seria vácua"
    assert omitido.content == explicito.content, (
        "omitir `method` deixou de ser o mesmo que `method=none`: o padrão da rota mudou de "
        "comportamento e quem a chama direto recebe outra matriz"
    )


def test_o_method_muda_o_artefato_processado_e_a_igualdade_acima_nao_e_cega(
    cliente: TestClient,
) -> None:
    """**O par causal do teste anterior**, e ele é o que impede a igualdade de ser vácua.

    Uma rota que ignorasse `method` produziria bytes iguais para os três valores, e
    `test_sem_method_a_rota_processa_como_antes` passaria **por o campo não fazer nada**. Aqui a
    asserção é que `loess` **diverge** de `none`, sobre a mesma entrada e o mesmo corte.

    **Sem marcador `slow` por custo medido:** o LOESS no `neg_c18` de 2.184 features é o caso barato
    dos três corretores — o teste vizinho de byte-identidade com `loess` no mesmo modo é `slow` por
    ~12 s, e esta função corre duas chamadas do mesmo porte. Se ela passar dos segundos, o marcador
    é a resposta certa; o que não é aceitável é a propriedade ficar sem par causal.
    """
    base = {"mode": "neg_c18", "rsd_cutoff": 0.30, "imputation": "one_fifth_min", "seed": 42}
    sem = reclama(
        cliente, "/matrizes/processar", files=_arquivos("neg_c18"), data={**base, "method": "none"}
    )
    com = reclama(
        cliente, "/matrizes/processar", files=_arquivos("neg_c18"), data={**base, "method": "loess"}
    )

    assert (sem.status_code, com.status_code) == (200, 200), (sem.text[:200], com.text[:200])
    assert sem.content != com.content, (
        "`method` não muda o artefato processado: a rota o aceita e o ignora, que é pior que o "
        "recusar — o cliente pede correção, recebe 200, e leva a matriz crua"
    )


def test_o_method_invalido_e_recusado_no_disparo_da_rota_processada(cliente: TestClient) -> None:
    """A fronteira de vocabulário do `method` vale nas DUAS rotas que o aceitam.

    `_corretor_ou_422` corre antes de a thread nascer, então rótulo inválido é `422` no disparo e
    não um erro que só aparece na reclamação — quem escreveu `loes` não espera por um cálculo para
    o saber. Verificar isto só em `/matrizes/corrigir` seria olhar só uma rota: a guarda é
    compartilhada, e uma rota que a contornasse não apareceria em teste nenhum.
    """
    resposta = cliente.post(
        "/matrizes/processar",
        files=_arquivos("neg_c18"),
        data={"mode": "neg_c18", "method": "loes"},
    )

    assert resposta.status_code == 422, f"`method=loes` foi aceito: {resposta.status_code}"
    assert "loes" in str(resposta.json()["detail"])


def test_ca_12_12_exportar_e_byte_identico_a_cli(cliente: TestClient) -> None:
    """O formato do MetaboAnalyst, sem `--enriquecer` dos dois lados."""
    resposta = reclama(
        cliente,
        "/matrizes/exportar",
        files=_arquivos("neg_c18"),
        data={"mode": "neg_c18", "max_features": 5000},
    )

    assert resposta.status_code == 200, resposta.text
    assert resposta.content == _pela_cli("neg_c18", "exportar", "--max-features", "5000")


def test_sem_method_a_exportacao_e_a_de_sempre(cliente: TestClient) -> None:
    """O padrão `none` preserva o comportamento anterior, byte a byte.

    A correção entrou nesta rota em 30/08/2026, e a mudança é **aditiva**: quem chama sem
    `method` — os testes de byte-identidade inclusive — tem de receber
    exactamente o que recebia. `NoneCorrector` devolve a matriz inalterada em valor, e a
    serialização não vê a diferença.
    """
    omitido = reclama(
        cliente, "/matrizes/exportar", files=_arquivos("neg_c18"), data={"mode": "neg_c18"}
    )
    explicito = reclama(
        cliente,
        "/matrizes/exportar",
        files=_arquivos("neg_c18"),
        data={"mode": "neg_c18", "method": "none"},
    )

    assert (omitido.status_code, explicito.status_code) == (200, 200)
    assert len(omitido.content) > 1000, "âncora: o artefato é vazio e a igualdade seria vácua"
    assert omitido.content == explicito.content, (
        "omitir `method` deixou de ser o mesmo que `method=none` na exportação"
    )


def test_o_method_muda_o_arquivo_do_metaboanalyst(cliente: TestClient) -> None:
    """**O par causal**, e ele é o teste que esta unidade inteira existe para ter.

    Sem ele, uma rota que aceitasse `method` e o **ignorasse** passaria em tudo o mais: o
    artefato sai com `200`, tem o formato conforme, e é a matriz **crua**. É o defeito exacto
    que já foi medido na janela para os outros dois artefatos — *"o profissional escolhia `serrf`,
    clicava «Gerar», e levava a matriz por corrigir a acreditar que estava corrigida"*.

    **A planta cai sobre dado em que as duas coisas diferem**: `loess` no `neg_c18` deixa
    32 features sem correção de 2.184, então 2.152 mudam de valor — a divergência não depende de
    coincidência. Medido em 30/08/2026: a primeira célula vai de `300392.570088058` para
    `252983.48430757`.

    **`loess` e não `serrf`** porque o SERRF custa 111 s neste modo contra 24 s do LOESS, e a
    propriedade medida é a mesma: o corretor alcança o escritor.
    """
    crua = reclama(
        cliente,
        "/matrizes/exportar",
        files=_arquivos("neg_c18"),
        data={"mode": "neg_c18", "method": "none"},
    )
    corrigida = reclama(
        cliente,
        "/matrizes/exportar",
        files=_arquivos("neg_c18"),
        data={"mode": "neg_c18", "method": "loess"},
    )

    assert (crua.status_code, corrigida.status_code) == (200, 200), (
        crua.text[:200],
        corrigida.text[:200],
    )
    assert len(crua.content) > 1000, "âncora: o artefato é vazio e a diferença seria acidental"
    assert crua.content != corrigida.content, (
        "`method` não muda o arquivo do MetaboAnalyst: a rota o aceita e o ignora, que é pior que "
        "o recusar — o cliente pede correção, recebe 200, e leva a matriz crua para o upload"
    )


@pytest.mark.slow
def test_o_n_batches_alcanca_o_corretor_na_exportacao(cliente: TestClient) -> None:
    """O arnês corre com o valor que EXERCITA, não com o que estava lá.

    Os testes vizinhos passam `n_batches` no padrão `1` — e com um lote `_lotes` devolve `None`, o
    que curto-circuita o ramo que constrói o vetor de lotes. Trinta asserções verdes sobre a metade
    do caminho que nunca corre não provam nada sobre ela, e o gesto é este: passar **um valor
    não-padrão no eixo que muda maquinaria**.

    Medido em 30/08/2026 no `neg_c18` com `loess`: `n_batches=1` dá 2.303.483 bytes e `n_batches=2`
    dá 2.316.156. **A planta cai sobre dado em que as duas coisas diferem** — com dois lotes as
    âncoras de mediana do passo 5 do LOESS são calculadas sobre metades, e o resultado tem de mudar.

    **`slow` por custo declarado:** são duas correções LOESS completas, ~50 s. O eixo que sustenta a
    unidade é o `method`, e esse fica no portão de commit sem marcador.
    """
    base = {"mode": "neg_c18", "method": "loess"}
    um = reclama(
        cliente, "/matrizes/exportar", files=_arquivos("neg_c18"), data={**base, "n_batches": 1}
    )
    dois = reclama(
        cliente, "/matrizes/exportar", files=_arquivos("neg_c18"), data={**base, "n_batches": 2}
    )

    assert (um.status_code, dois.status_code) == (200, 200), (um.text[:200], dois.text[:200])
    assert len(um.content) > 1000, "âncora: o artefato é vazio e a diferença seria acidental"
    assert um.content != dois.content, (
        "`n_batches` não alcança o corretor na exportação: a rota o aceita e o ignora, e quem "
        "declarou dois lotes recebe a correção de um só"
    )


def test_o_method_invalido_e_recusado_no_disparo_da_exportacao(cliente: TestClient) -> None:
    """A fronteira de vocabulário do `method` vale nas TRÊS rotas que o aceitam.

    `_corretor_ou_422` corre antes de a thread nascer. Verificar isto só nas outras duas seria
    olhar só parte das rotas: a guarda é compartilhada, e uma rota que a contornasse não
    apareceria em teste nenhum.
    """
    resposta = cliente.post(
        "/matrizes/exportar",
        files=_arquivos("neg_c18"),
        data={"mode": "neg_c18", "method": "loes"},
    )

    assert resposta.status_code == 422, f"`method=loes` foi aceito: {resposta.status_code}"
    assert "loes" in str(resposta.json()["detail"])


def test_r_faithful_na_exportacao_reprova_e_a_mensagem_nomeia_a_saida(cliente: TestClient) -> None:
    """O padrão do `null_denominator` DIVERGE do `corrigir`, e a divergência é medida.

    Sob `r_faithful` o `batch_loess` produz `Inf` — 109 células no `neg_c18`, medido em
    11/11/2026 nos quatro modos (109, 552, 448, 763) —, e `write_metaboanalyst` recusa
    infinito porque o `[F-2]` do formato pede *"only numeric and positive values"*. Daí o padrão
    desta rota ser `uncorrected_feature`.

    **A asserção sobre a MENSAGEM é o conteúdo do teste, e não decoração.** Um `422` sozinho
    deixaria o profissional sem saber que existe uma saída, e aqui o detalhe é a única coisa
    acionável.
    """
    resposta = reclama(
        cliente,
        "/matrizes/exportar",
        files=_arquivos("neg_c18"),
        data={"mode": "neg_c18", "method": "loess", "null_denominator": "r_faithful"},
    )

    assert resposta.status_code == 422, f"esperava 422, veio {resposta.status_code}"
    detalhe = resposta.text
    assert "infinitos" in detalhe, detalhe[:300]
    assert "uncorrected_feature" in detalhe, (
        "a mensagem não nomeia a saída; sem ela o profissional só sabe que falhou"
    )


def test_ca_12_12_o_comparador_consegue_acusar_diferenca(cliente: TestClient) -> None:
    """**O par causal das asserções acima.**

    Byte-identidade que passasse sempre seria indistinguível de comparação vazia. Aqui as duas
    saídas são de MODOS diferentes e a asserção é que elas **divergem** — sem isto, um bug no
    comparador deixaria os quatro testes acima verdes por construção.
    """
    um = reclama(
        cliente,
        "/matrizes/corrigir",
        files=_arquivos("neg_c18"),
        data={"mode": "neg_c18", "method": "none"},
    )
    outro = reclama(
        cliente,
        "/matrizes/corrigir",
        files=_arquivos("neg_hilic"),
        data={"mode": "neg_hilic", "method": "none"},
    )

    assert (um.status_code, outro.status_code) == (200, 200), (um.text[:200], outro.text[:200])
    assert um.content != outro.content, "o comparador não distingue modos diferentes"


def test_ca_12_12_a_rota_de_artefato_aceita_serrf_na_fronteira() -> None:
    """Invertido: a decisão fechou para o outro lado, nas três superfícies.

    Este teste afirmava `422`. A decisão de escopo foi tomada, e o `serrf` passou a ser aceite em
    `/matrizes/corrigir`, `/execucoes` e `/multiversos`.

    **Afirma a FRONTEIRA e não o `POST` inteiro, por custo medido.** A versão anterior era
    barata porque o `422` acontecia antes de qualquer trabalho; com a aceitação, o mesmo
    `POST` executa um SERRF real sobre o `neg_c18` — **86 s medidos**. A propriedade cobrada
    aqui é a paridade da fronteira com a CLI, e ela é da função. O caminho
    ponta a ponta está em `test_api_sobrescrita.py`, marcado `slow`.
    """
    from refino.api.rotas import _corretor_ou_422

    assert _corretor_ou_422("serrf", "r_faithful").name == "serrf"


@pytest.mark.parametrize("rota", ["corrigir", "processar", "exportar"])
def test_ca_12_11_as_rotas_de_artefato_nao_exigem_banco(
    cliente: TestClient, rota: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """O par causal das rotas sem banco, e é ele que impede a regressão contra a CLI.

    Sem esta direção, acoplar uma das três ao repositório passaria despercebido — e o banco
    viraria pré-requisito para corrigir uma matriz, que nenhum dos subcomandos equivalentes exige.
    """
    monkeypatch.delenv("REFINO_DB_HOST", raising=False)

    resposta = reclama(
        cliente, f"/matrizes/{rota}", files=_arquivos("neg_c18"), data={"mode": "neg_c18"}
    )

    assert resposta.status_code == 200, f"{rota} devolveu {resposta.status_code} sem banco"


def test_ca_12_13_o_artefato_nao_persiste_no_servico(cliente: TestClient) -> None:
    """Nada novo fica em caminho durável depois da chamada.

    A comparação é sobre o diretório de temporários do sistema, que é onde `_carrega` e as rotas
    de artefato escrevem. `TemporaryDirectory` apaga ao sair; este teste é o que afirma que o
    apagar acontece, em vez de o supor pelo nome da classe.
    """
    raiz = Path(tempfile.gettempdir())
    antes = {p.name for p in raiz.glob("refino-*")}

    # A espera e obrigatoria, e sem ela este teste mediria o instante errado. O `POST` volta em
    # `202` **antes** de o escritor correr, entao um `glob` logo depois
    # olharia um diretorio temporario que ainda vai nascer — e o teste passaria por chegar cedo, nao
    # por o temporario ser apagado. Reclamar e o que garante que a escrita aconteceu E terminou.
    resposta = reclama(
        cliente,
        "/matrizes/corrigir",
        files=_arquivos("neg_c18"),
        data={"mode": "neg_c18", "method": "none", "seed": 13},
    )
    assert resposta.status_code == 200, "ancora: sem artefato produzido a ausencia abaixo e vacua"

    depois = {p.name for p in raiz.glob("refino-*")}
    assert depois <= antes, f"a chamada deixou temporários para trás: {sorted(depois - antes)}"


def test_ca_12_13_o_detector_de_residuo_consegue_acusar() -> None:
    """Sem esta direção o teste acima passa numa árvore onde nada é criado nem apagado."""
    raiz = Path(tempfile.gettempdir())
    antes = {p.name for p in raiz.glob("refino-*")}
    with tempfile.TemporaryDirectory(prefix="refino-planta-") as plantado:
        durante = {p.name for p in raiz.glob("refino-*")}
        assert durante - antes == {Path(plantado).name}

    assert {p.name for p in raiz.glob("refino-*")} <= antes


def test_ca_12_9_arquivo_malformado_nas_rotas_de_artefato(cliente: TestClient) -> None:
    """O desfecho é o MESMO das rotas existentes — as novas não inventam classificação própria."""
    ruim = {
        "measurements": ("m.csv", b"a,b,c\n1,2,3\n", "text/csv"),
        "sample_list": ("s.csv", sample_list_path("neg_c18").read_bytes(), "text/csv"),
    }

    resposta = cliente.post("/matrizes/corrigir", files=ruim, data={"mode": "neg_c18"})

    assert resposta.status_code == 422


def test_ca_12_11_o_aviso_de_modo_divergente_aparece_e_nao_recusa(cliente: TestClient) -> None:
    """Modo divergente é aviso, não recusa."""
    resposta = cliente.post(
        "/matrizes/validar", files=_arquivos("pos_hilic"), data={"mode": "neg_c18"}
    )

    corpo = resposta.json()
    assert resposta.status_code == 200
    assert corpo["aceita"] is True, "o aviso virou recusa"
    codigos = [p["codigo"] for p in corpo["problemas"]]
    assert codigos.count("MODE_CONTENT_MISMATCH") == 2, "os dois eixos divergem e ambos têm de sair"
    assert all(p["severidade"] == "warning" for p in corpo["problemas"])


@pytest.mark.parametrize("mode", MODOS)
def test_ca_12_11_o_modo_certo_nao_produz_aviso(cliente: TestClient, mode: str) -> None:
    """**O par causal do aviso** — sem ele, um aviso constante passaria por guarda."""
    resposta = cliente.post("/matrizes/validar", files=_arquivos(mode), data={"mode": mode})

    codigos = [p["codigo"] for p in resposta.json()["problemas"]]
    assert "MODE_CONTENT_MISMATCH" not in codigos, f"{mode}: aviso sobre o modo correto"


# ---------------------------------------------------------------------------
# O modo é rótulo ABERTO, e a fronteira o restringe a nome de arquivo seguro
# ---------------------------------------------------------------------------
#
# **A guarda nasce com a abertura do seletor.** Enquanto a tela oferecia quatro nomes fechados,
# nenhum modo hostil chegava aqui por aquela porta; com o seletor aberto, o que era improvável
# passa a ser digitável.
#
# **MEDIDO EM 16/08/2026, ANTES da guarda existir**, contra `POST /matrizes/corrigir` com os
# arquivos reais do `neg_c18` — quatro chamadas, e nenhuma delas recusou:
#
# | `mode` | Antes |
# |---|---|
# | `/tmp/sonda-fora-do-temporario` | **200**, e o CSV de 2.367.696 bytes ficou em `/tmp` DEPOIS |
# | | de o `TemporaryDirectory` sair — estava FORA do diretório apagado |
# | `../fora-do-temporario` | **200** |
# | `a/b` | **200** — `write_toserrf` faz `parent.mkdir(parents=True)`, e nem 500 dava |
# | `lipidomica_neg` | **200**, e este é o que tem de continuar a valer |
#
# A causa é uma linha de `pathlib`: `Path('/tmp/x') / '/home/u/.bashrc_corrigido.csv'` devolve o
# operando da direita. `corrigir`, `processar` e `exportar` montam
# `Path(temporario) / f"{mode}_<sufixo>.csv"` — nome de função e não número de linha, que caduca —,
# então modo absoluto **substitui o diretório inteiro** e a escrita acontece onde o cliente mandar,
# com o utilizador do serviço.
#
# **O conjunto seguro continua ABERTO:** `[A-Za-z0-9_-]{1,32}` não é enumeração —
# `lipidomica_neg`, `hilic2`, `RP_POS` passam. O que ele fecha é a classe de caracteres que faz um
# rótulo virar caminho.

MODO_DE_OUTRO_ESTUDO = "lipidomica_neg"

# Cada entrada é uma FORMA diferente de o rótulo deixar de ser rótulo: absoluto, relativo,
# separador no meio, longo demais, espaço e ponto. Não são sinónimos.
#
# **O vazio NÃO está aqui, e a omissão é medida.** Ele é recusado por outro mecanismo — a
# obrigatoriedade do `Form()` no FastAPI —, e pô-lo nesta lista faria a asserção sobre o texto da
# nossa regra reprovar contra um `Field required` que não é dela. Tem teste próprio, que afirma os
# dois lados.
MODOS_RECUSADOS = (
    "/tmp/fora",
    "../fora",
    "a/b",
    "x" * 33,
    "modo com espaco",
    "modo.csv",
)

# As sete rotas que recebem `mode`. A guarda é UMA, em `_carrega`, e esta lista é o que impede
# que ela seja verificada só onde foi fácil de testar.
ROTAS_COM_MODO = (
    "/matrizes/validar",
    "/matrizes/diagnosticar",
    "/matrizes/recomendar",
    "/matrizes/corrigir",
    "/matrizes/processar",
    "/matrizes/exportar",
    "/multiversos",
)


@pytest.mark.parametrize("rota", ["corrigir", "processar", "exportar"])
def test_o_modo_de_outro_estudo_e_aceito_pelas_tres_rotas(cliente: TestClient, rota: str) -> None:
    """**R3 na API: o modo não é enumeração fechada**, e o rótulo chega ao artefato.

    A asserção é sobre o `Content-Disposition` e não sobre o `200`: o nome do arquivo é onde o
    rótulo de facto aparece para quem baixa, e é ele que estaria errado se o modo fosse
    substituído por um dos quatro no caminho.
    """
    resposta = reclama(
        cliente,
        f"/matrizes/{rota}",
        files=_arquivos("neg_c18"),
        data={"mode": MODO_DE_OUTRO_ESTUDO},
    )

    assert resposta.status_code == 200, resposta.text
    assert MODO_DE_OUTRO_ESTUDO in resposta.headers["content-disposition"], (
        f"o modo não chegou ao nome do artefato: {resposta.headers['content-disposition']}"
    )


@pytest.mark.parametrize("modo", MODOS_RECUSADOS)
def test_modo_que_nao_e_nome_seguro_e_recusado_com_422_que_nomeia_a_regra(
    cliente: TestClient, modo: str
) -> None:
    """**422 e não 500, e o texto nomeia a regra** — a forma de `_corretor_ou_422`.

    Um `500` diria ao profissional *"defeito da ferramenta, não do seu arquivo"* — é o que
    `interface/cliente.py` traduz —, mandando-o ignorar exactamente a coisa que ele consegue
    corrigir.
    E a regra vai no texto porque um `422` que só diz *"modo inválido"* deixa o utilizador a
    adivinhar qual caractere ofende.
    """
    resposta = cliente.post(
        "/matrizes/corrigir", files=_arquivos("neg_c18"), data={"mode": modo, "method": "none"}
    )

    assert resposta.status_code == 422, f"{modo!r} devolveu {resposta.status_code}"
    detalhe = str(resposta.json()["detail"])
    assert "A-Za-z0-9_-" in detalhe, f"o 422 não nomeia a regra: {detalhe}"
    assert "32" in detalhe, f"o 422 não diz o limite de comprimento: {detalhe}"


def test_o_modo_vazio_e_recusado_e_o_mecanismo_e_outro(cliente: TestClient) -> None:
    """**Dois mecanismos recusam o vazio, e dizê-lo é o que impede a conclusão errada.**

    Medido: pela rota, quem recusa é o `Form()` obrigatório do FastAPI — `Field required`, com
    `input: None` —, e a nossa regra nunca chega a correr. Afirmar aqui que o `422` "nomeia a
    regra" seria ler o desfecho certo como prova do mecanismo errado.

    A segunda asserção é sobre a **função**, e é ela que prende a nossa metade: se alguém trocar
    o `Form()` por um com padrão, a rota deixa de recusar e `_modo_ou_422` continua a fazê-lo.
    """
    from fastapi import HTTPException

    from refino.api.rotas import _modo_ou_422

    resposta = cliente.post(
        "/matrizes/corrigir", files=_arquivos("neg_c18"), data={"mode": "", "method": "none"}
    )
    assert resposta.status_code == 422, resposta.text
    assert "Field required" in str(resposta.json()["detail"]), (
        "o mecanismo mudou: o vazio deixou de ser recusado pela obrigatoriedade do campo"
    )

    with pytest.raises(HTTPException) as erro:
        _modo_ou_422("")
    assert erro.value.status_code == 422
    assert "A-Za-z0-9_-" in str(erro.value.detail)


def test_o_par_causal_da_guarda_o_desfecho_diverge_so_pela_forma_do_valor(
    cliente: TestClient,
) -> None:
    """**Par causal**: duas chamadas idênticas em tudo menos na forma do rótulo.

    Sem ele, uma guarda que recusasse **sempre** passaria no teste acima, e uma que recusasse
    **nunca** passaria no teste de aceitação. As duas direções
    numa função só, sobre o mesmo arquivo e a mesma rota.
    """
    aceito = cliente.post(
        "/matrizes/corrigir",
        files=_arquivos("neg_c18"),
        data={"mode": MODO_DE_OUTRO_ESTUDO, "method": "none"},
    )
    recusado = cliente.post(
        "/matrizes/corrigir",
        files=_arquivos("neg_c18"),
        data={"mode": f"../{MODO_DE_OUTRO_ESTUDO}", "method": "none"},
    )

    # O lado aceito e `202`, e a propriedade guardada nao mudou: o desfecho
    # diverge pela FORMA do valor. A assercao fica no **disparo** de proposito — e la que a guarda
    # age, e reclamar o artefato aqui mediria o calculo em vez da fronteira.
    assert (aceito.status_code, recusado.status_code) == (202, 422), (
        f"os desfechos não divergem pela forma do valor: {aceito.status_code} e "
        f"{recusado.status_code}"
    )


@pytest.mark.parametrize("rota", ROTAS_COM_MODO)
def test_a_guarda_esta_na_fronteira_compartilhada_e_nao_rota_a_rota(
    cliente: TestClient, rota: str
) -> None:
    """**A superfície é toda rota que recebe `mode`, e não as três que montam caminho.**

    Verificar só as três de artefato seria olhar só parte delas: a guarda vive em
    `_carrega`, por onde as sete passam, e uma rota nova que a contornasse não apareceria em
    teste nenhum. `/matrizes/validar` entra apesar de recusa de **dado** ali ser 200 — rótulo
    malformado é requisição malformada, como `orientation`, e não defeito do arquivo.

    **`/multiversos` recusa ANTES de a thread nascer**, e é por isso que este caso é barato:
    `_ingere_ou_422` corre na primeira linha da rota, e a grade de 4.320 células só é montada
    depois.
    """
    resposta = cliente.post(rota, files=_arquivos("neg_c18"), data={"mode": "../fora"})

    assert resposta.status_code == 422, f"{rota} aceitou modo com caminho: {resposta.status_code}"


def test_o_modo_absoluto_nao_escreve_fora_do_temporario(cliente: TestClient) -> None:
    """**A propriedade que a guarda existe para dar**, e não o código de estado que a acompanha.

    Medido antes da guarda: a mesma chamada devolvia **200** e deixava
    `fora-do-temporario_corrigido.csv` em `/tmp` depois de `TemporaryDirectory` sair — o arquivo
    estava **fora** do diretório apagado porque o operando absoluto o substituiu.

    **São DUAS âncoras e provam coisas diferentes**, e é por isso que nenhuma substitui a outra:

    1. **O DETECTOR acusa.** Planta-se em `/tmp` o arquivo que a travessia produziria e vê-se o
       `glob` achá-lo. Sem esta, padrão ou diretório errados fariam `escritos == []` passar **para
       sempre**: seria asserção de ausência sobre um detector cego, que é a segunda linha da
       taxonomia de negativo plantado — a invocação não alcança a mutação.
    2. **A ROTA escreve.** A chamada com modo seguro devolve artefato. Sem esta, uma rota que
       respondesse 422 a tudo — ou que não escrevesse nada — satisfaria a ausência sem nada provar.

    A primeira foi acrescentada na revisão da unidade; a versão do worker tinha só a segunda, e a
    pergunta que a achou é *quantos desfechos este mecanismo pode reportar, e eu vi todos?*

    **O alvo é único por execução, e a unicidade é conserto de defeito meu.** Com nome fixo, a
    execução que mediu o defeito **escreveu o arquivo** — 2.367.696 bytes em `/tmp`, que sobrevivem
    ao processo — e a âncora *"não existia antes"* ficava permanentemente vermelha na máquina que
    tivesse rodado o teste uma vez sem a guarda.
    """
    raiz = Path(tempfile.gettempdir())
    alvo = raiz / f"u2-fora-do-temporario-{uuid4().hex[:8]}"
    padrao = f"{alvo.name}*"
    assert not list(raiz.glob(padrao)), "âncora: o alvo já existia antes da chamada"

    plantado = raiz / f"{alvo.name}_corrigido.csv"
    plantado.write_text("planta do detector", encoding="utf-8")
    try:
        assert [p.name for p in raiz.glob(padrao)] == [plantado.name], (
            "âncora do detector: o glob não acha nem o arquivo plantado com o nome exato que a "
            "travessia produziria — a asserção de ausência no fim seria cega"
        )
    finally:
        plantado.unlink()
    assert not list(raiz.glob(padrao)), "a planta do detector não foi removida"

    seguro = reclama(
        cliente,
        "/matrizes/corrigir",
        files=_arquivos("neg_c18"),
        data={"mode": "u2_dentro_do_temporario", "method": "none"},
    )
    assert seguro.status_code == 200 and len(seguro.content) > 1000, (
        "âncora: a rota não produz artefato nem com modo seguro; a ausência abaixo seria vácua"
    )

    hostil = cliente.post(
        "/matrizes/corrigir",
        files=_arquivos("neg_c18"),
        data={"mode": str(alvo), "method": "none"},
    )

    assert hostil.status_code == 422, f"o modo absoluto foi aceite: {hostil.status_code}"
    escritos = sorted(p.name for p in raiz.glob(padrao))
    assert escritos == [], f"a chamada escreveu fora do diretório temporário: {escritos}"


def test_a_gravacao_do_tempo_e_melhor_esforco_e_o_artefato_sai_identico(
    cliente: TestClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """**O par causal do melhor esforço** — a mesma correção com e sem o banco a recusar.

    ## O que ele impede

    Desde 30/08/2026 as três rotas de artefato gravam quanto custaram. A gravação é **melhor
    esforço** pelo motivo que o multiverso já registou em `_abre_no_disco`, e aqui a assimetria é
    maior: lá a escrita é no disparo, aqui é no **fim**. Um `serrf` no `pos_hilic` custa
    **1.334,3 s** medidos no congelado, e derrubar o artefato porque o banco não respondeu faria o
    usuário **perder 22 minutos de cálculo por causa de um registro que é nosso e não dele**.

    ## Por que não basta afirmar o `200`

    `test_ca_12_11_as_rotas_de_artefato_nao_exigem_banco` já afirma que a rota responde sem banco
    nenhum — e ele passaria por **vacuidade** numa implementação que nunca tentasse gravar. A
    planta aqui é uma escrita que **levanta**, e a contagem de chamadas é a âncora que prova que
    ela aterrou: sem essa linha, um `_repositorio()` que falhasse antes de alcançar
    `grava_artefato` daria exactamente este desfecho de sucesso.

    E a comparação é entre **duas execuções idênticas menos a planta**: afirmar só que a resposta
    tem bytes deixaria passar uma gravação que alterasse o artefato — que é a regressão de que
    ninguém suspeita, porque o CSV continuaria a parecer um CSV.
    """
    from refino.persistence.config import VARIAVEL_DA_URL
    from refino.persistence.repositories import RunRepository

    # **Sementes diferentes nas duas voltas, e não é decoração.** `_chave_do_trabalho` inclui os
    # campos da rota: com a mesma semente, a segunda chamada seria a **reclamação** do resultado da
    # primeira, nenhuma thread nasceria, e o teste compararia uns bytes consigo mesmos. Com
    # `method=none` a semente não alcança valor nenhum — `NoneCorrector` devolve a matriz
    # inalterada —, então as duas saídas continuam a ter de ser idênticas.
    sem_planta = reclama(
        cliente,
        "/matrizes/corrigir",
        files=_arquivos("neg_c18"),
        data={"mode": "neg_c18", "method": "none", "seed": 987654},
    )
    assert sem_planta.status_code == 200, sem_planta.text[:300]
    assert sem_planta.content, "âncora: sem artefato na volta limpa a comparação abaixo é vácua"

    # **A captura é DRENADA aqui, e sem isto a última asserção deste teste seria vácua.** A volta
    # limpa também tenta gravar, e o ambiente da suíte quase nunca tem banco: ela já deixou o mesmo
    # texto no `stderr`. Ler no fim sem drenar faria o relato da volta limpa satisfazer a asserção
    # sobre a volta plantada — verificação sã sobre o alvo errado.
    capsys.readouterr()

    chamadas: list[str] = []

    def explode(self: RunRepository, **kw: object) -> object:
        chamadas.append(str(kw["operacao"]))
        raise RuntimeError("planta: o banco recusou a escrita")

    monkeypatch.setattr(RunRepository, "grava_artefato", explode)
    # A configuração tem de ser VÁLIDA, senão `_repositorio()` levanta antes e a planta nunca é
    # alcançada — o teste passaria a medir o caminho do banco ausente, que é outro.
    monkeypatch.delenv("REFINO_DB_HOST", raising=False)
    monkeypatch.setenv(VARIAVEL_DA_URL, f"sqlite:///{tmp_path / 'refino.db'}")

    com_planta = reclama(
        cliente,
        "/matrizes/corrigir",
        files=_arquivos("neg_c18"),
        data={"mode": "neg_c18", "method": "none", "seed": 987655},
    )

    assert chamadas == ["corrigir"], (
        f"a planta não aterrou: a rota chamou `grava_artefato` {len(chamadas)} vez(es). Sem a "
        "chamada, tudo o que este teste afirma abaixo é verdade por a gravação não existir"
    )
    assert com_planta.status_code == 200, com_planta.text[:300]
    assert com_planta.headers["content-type"].startswith("text/csv")
    assert com_planta.content == sem_planta.content, (
        "o artefato mudou quando a gravação do tempo falhou: o registro deixou de ser melhor "
        "esforço e passou a decidir o que o profissional leva"
    )

    # **O silêncio é recusado junto.** Gravação que falha sem dizer é a métrica a parar de
    # medir: o profissional fecharia o programa a contar com um tempo que nunca foi para o disco.
    erro = capsys.readouterr().err
    assert "NAO foi gravado" in erro, f"a perda do registro não foi relatada: {erro!r}"
    # O motivo é o da PLANTA e não uma frase genérica: sem esta linha, qualquer outra falha de
    # gravação satisfaria a asserção acima, e ela deixaria de medir esta.
    assert "planta: o banco recusou a escrita" in erro, erro


# ---------------------------------------------------------------------------
# O reuso de correção entre as três rotas, medido no CAMINHO REAL
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_d381_a_correcao_e_reusada_entre_corrigir_e_exportar(
    cliente: TestClient, memoria_de_correcoes_privada: None
) -> None:
    """A TERCEIRA pergunta do reuso: ele de facto acerta?

    ## Por que este teste é o que impede os outros de passarem por vacuidade

    `tests/unit/test_reuso_de_correcao.py` prova que o reuso é **fiel** e que a chave **separa**.
    Nenhum dos dois distingue uma memória que funciona de uma que nunca guarda nada: os dois ficam
    verdes sobre um cache morto, e o sintoma de um cache morto é ausência de ganho — silêncio, que
    é o desfecho que ninguém investiga.

    **A asserção é sobre o CONTADOR e não sobre o tempo.** Cronometrar duas chamadas mediria a
    máquina; contar acertos mede o mecanismo. É o mesmo argumento de `_EscolhedorQueConta` no arnês
    da janela: *"o painel parece guardar os bytes"* não é verificável, e um número é.

    ## Por que `corrigir` e `exportar`, e por que com `method=none`

    As duas correm `fit_transform` com os mesmos argumentos e são rotas **diferentes**, então a
    deduplicação de trabalhos não as liga — é exactamente o par que o reuso existe para juntar.

    **`none` e não `serrf`**, porque o que se mede é o acerto e não a economia: `serrf` no
    `pos_hilic` custa 209,9 s por chamada, e este teste corre no portão de cada commit.

    **E `null_denominator` não interfere com `none`:** ele só alcança o LOESS. Sob `loess` as duas
    rotas têm padrões diferentes e **não** partilhariam correção — o que é o comportamento certo, e
    o que o par causal da unidade afirma.
    """
    from refino.api import rotas

    antes = rotas.CORRECOES.acertos

    primeira = reclama(
        cliente,
        "/matrizes/corrigir",
        files=_arquivos("neg_c18"),
        data={"mode": "neg_c18", "method": "none"},
        teto_s=120,
    )
    assert primeira.status_code == 200, primeira.text
    depois_da_primeira = rotas.CORRECOES.acertos
    assert depois_da_primeira == antes, (
        f"a PRIMEIRA chamada já acertou na memória ({depois_da_primeira - antes} acerto(s)) — "
        f"outro teste deixou a correção lá, e este mediria o resíduo em vez do reuso"
    )

    segunda = reclama(
        cliente,
        "/matrizes/exportar",
        files=_arquivos("neg_c18"),
        data={"mode": "neg_c18", "method": "none"},
        teto_s=120,
    )
    assert segunda.status_code == 200, segunda.text

    assert rotas.CORRECOES.acertos == depois_da_primeira + 1, (
        f"a segunda rota NÃO reusou a correção da primeira: {rotas.CORRECOES.acertos} acertos "
        f"contra {depois_da_primeira + 1} esperados. As duas fazem o mesmo `fit_transform`, e sem "
        f"o reuso quem gera os dois artefatos paga o corretor duas vezes — 209,9 s por repetição "
        f"no `pos_hilic` sob `serrf`"
    )


@pytest.mark.integration
def test_d381_o_reuso_nao_cruza_metodos_diferentes(cliente: TestClient) -> None:
    """O par causal no CAMINHO REAL: métodos diferentes não partilham correção.

    É a metade que impede o teste acima de passar sobre uma memória que devolve sempre a mesma
    coisa. E é o defeito medido que este projeto já registrou duas vezes: *"o profissional levaria
    a matriz por corrigir a acreditar que estava corrigida"*.

    **A asserção é sobre os BYTES do artefato e não sobre o contador**, porque é a consequência que
    o profissional vê: duas correções diferentes têm de produzir dois arquivos diferentes.
    """
    corrigida_none = reclama(
        cliente,
        "/matrizes/corrigir",
        files=_arquivos("neg_c18"),
        data={"mode": "neg_c18", "method": "none"},
        teto_s=120,
    )
    corrigida_loess = reclama(
        cliente,
        "/matrizes/corrigir",
        files=_arquivos("neg_c18"),
        data={"mode": "neg_c18", "method": "loess"},
        teto_s=600,
    )

    assert corrigida_none.status_code == 200, corrigida_none.text
    assert corrigida_loess.status_code == 200, corrigida_loess.text
    assert corrigida_none.content != corrigida_loess.content, (
        "`none` e `loess` produziram o MESMO arquivo: a chave do reuso não separa por corretor, e "
        "o profissional levaria a matriz por corrigir a acreditar que estava corrigida"
    )


# ---------------------------------------------------------------------------
# Item 6 da passagem humana — `/execucoes` GUARDA a correção e NUNCA a lê
# ---------------------------------------------------------------------------
#
# **O motivo:** executar o pipeline devia deixar os três artefatos prontos para baixar sem
# reprocessar — e não deixava. `_corrige_reusando` era chamado por `corrigir`,
# `processar` e `exportar`; `/execucoes` não o usava, e executar com `serrf` e depois clicar em
# «matriz corrigida» pagava os 209,9 s do `pos_hilic` duas vezes.
#
# **A forma é assimétrica de propósito**: guarda e nunca lê. O escopo negativo do reuso recusava
# esta rota porque ela
# **escreve**, e chave errada ali custa linha persistida errada. Uma rota que guarda e não consulta
# nunca decide nada com a chave: o risco desaparece **por construção** e não por cuidado.
#
# **Os dois testes são um par causal e não dois testes.** Sozinho, o primeiro passaria numa rota
# que também LESSE; sozinho, o segundo passaria numa rota que não fizesse nada. É a conjunção que
# descreve a forma escolhida.


@pytest.fixture
def banco_de_mesa(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Um SQLite proprio para esta prova.

    **`POST /execucoes` GRAVA, e sem banco recusa no disparo.**

    A recusa e deliberada e esta medida noutro teste: descobrir que o banco nao subiu **depois**
    de 209,9 s de SERRF e o desfecho a evitar. Aqui o banco e um arquivo temporario, porque o que
    se mede e a memoria de correcoes e nao a persistencia.
    """
    from refino.desktop.banco import migra
    from refino.persistence.config import VARIAVEL_DA_URL

    url = f"sqlite:///{tmp_path / 'refino.db'}"
    monkeypatch.delenv("REFINO_DB_HOST", raising=False)
    monkeypatch.setenv(VARIAVEL_DA_URL, url)
    # **`migra` e nao so a URL, e a diferenca foi MEDIDA.** Sem ela a rota sobe o motor, grava, e
    # morre em `no such table: studies` — com `500` na resposta e a mensagem no `stderr`. Um
    # arquivo de SQLite vazio nao e um banco.
    migra(url)


def _dados_da_execucao(mode: str) -> dict[str, object]:
    """O `data` mínimo de `POST /execucoes` com o corretor `none`.

    **`none` e não `serrf`**, pelo mesmo motivo do teste irmão do reuso: o que se mede é o acerto e
    não a economia, e este teste corre no portão de cada commit.
    """
    return {
        "mode": mode,
        "estudo": "item-6",
        "correction": "none",
        "n_batches": 1,
        "rsd_variant": "plain",
        "imputation": "none",
        "sample_normalization": "none",
        "transformation": "none",
        "scaling": "none",
        "apresentacao": "exibida",
    }


@pytest.mark.integration
def test_item6_a_execucao_guarda_a_correcao_para_os_botoes_de_download(
    cliente: TestClient,
    memoria_de_correcoes_privada: None,
    banco_de_mesa: None,
) -> None:
    """Depois de executar o pipeline, o download da matriz corrigida ACERTA na memória."""
    from refino.api import rotas

    antes = rotas.CORRECOES.acertos

    execucao = reclama(
        cliente,
        "/execucoes",
        files=_arquivos("neg_c18"),
        data=_dados_da_execucao("neg_c18"),
        teto_s=120,
    )
    assert execucao.status_code == 200, execucao.text
    depois_da_execucao = rotas.CORRECOES.acertos
    assert depois_da_execucao == antes, (
        f"a EXECUÇÃO acertou na memória ({depois_da_execucao - antes} acerto(s)) com a memória "
        f"VAZIA. Desde U24 ela consulta, mas não havia o que acertar: um acerto aqui só "
        f"pode ser resíduo de outro teste, e a fixture que isola a memória falhou"
    )

    download = reclama(
        cliente,
        "/matrizes/corrigir",
        files=_arquivos("neg_c18"),
        data={"mode": "neg_c18", "method": "none"},
        teto_s=120,
    )
    assert download.status_code == 200, download.text
    assert rotas.CORRECOES.acertos == depois_da_execucao + 1, (
        f"o download NÃO encontrou a correção que a execução acabou de calcular: "
        f"{rotas.CORRECOES.acertos} acertos contra {depois_da_execucao + 1} esperados. É a "
        f"expectativa declarada, e sem isto o usuário paga o corretor duas vezes"
    )


@pytest.mark.integration
def test_u24_a_execucao_le_a_memoria_e_salta_o_corretor(
    cliente: TestClient,
    memoria_de_correcoes_privada: None,
    banco_de_mesa: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A outra metade, e ela INVERTEU.

    Ela já afirmou o contrário, e a inversão é emenda e não conserto de defeito. A
    escolha anterior era *"guarda e nunca lê"* entre três saídas, e o argumento era bom: esta
    rota grava no banco, e uma chave errada aqui custaria linha persistida errada — guardando sem
    ler, o pior desfecho era uma entrada que ninguém acerta.

    **O que mudou foi a premissa, não a preferência.** A premissa era *"não há como saber se a
    chave está errada"*; depois veio o portão que a confere em três direções mais dois cenários
    de valor, e ele foi visto acusar em quatro plantas — uma delas uma isenção falsa de quem o
    escreveu.

    **Sem esta metade o par não separa:** o teste irmão só afirma o que acontece com a memória
    VAZIA, e uma rota que voltasse a ignorá-la passaria nele.
    """
    from refino.api import rotas

    semeia = reclama(
        cliente,
        "/matrizes/corrigir",
        files=_arquivos("neg_c18"),
        data={"mode": "neg_c18", "method": "none"},
        teto_s=120,
    )
    assert semeia.status_code == 200, semeia.text
    com_a_correcao_guardada = rotas.CORRECOES.acertos

    # Âncora na MESMA função, e ela usa OUTRA rota, por medição. A primeira
    # versão repetia `/matrizes/corrigir` e media **zero** acertos: dois `POST` idênticos à mesma
    # rota são deduplicados pelo TRABALHO, e o segundo devolve o resultado retido sem sequer
    # perguntar à memória. É a razão de o teste irmão do reuso usar `corrigir` e `exportar`, e a
    # mesma razão vale aqui.
    repete = reclama(
        cliente,
        "/matrizes/exportar",
        files=_arquivos("neg_c18"),
        data={"mode": "neg_c18", "method": "none"},
        teto_s=120,
    )
    assert repete.status_code == 200, repete.text
    assert rotas.CORRECOES.acertos == com_a_correcao_guardada + 1, (
        "âncora: a correção NÃO ficou guardada, então «a execução não a lê» não separa nada"
    )
    antes_da_execucao = rotas.CORRECOES.acertos

    # **O espião entra DEPOIS de semear**: o que se quer medir é o corretor da EXECUÇÃO, e não os
    # das duas rotas que encheram a memória.
    corridas: list[str] = []
    original = NoneCorrector.fit_transform

    def espia(self: NoneCorrector, *args: object, **kwargs: object) -> object:
        corridas.append("fit_transform")
        return original(self, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(NoneCorrector, "fit_transform", espia)

    execucao = reclama(
        cliente,
        "/execucoes",
        files=_arquivos("neg_c18"),
        data=_dados_da_execucao("neg_c18"),
        teto_s=120,
    )
    assert execucao.status_code == 200, execucao.text
    assert rotas.CORRECOES.acertos == antes_da_execucao + 1, (
        f"a execução não CONSULTOU a memória: {rotas.CORRECOES.acertos} acertos contra "
        f"{antes_da_execucao + 1} esperados"
    )
    # **E a asserção que de facto separa: o corretor NÃO correu.**
    #
    # `acertos` incrementa dentro de `busca()` quando a chave é achada — **independentemente de o
    # chamador usar o resultado**. Medido em 13/09/2026: com `if False and guardado is not None`
    # plantado em `_corrige_reusando`, a rota ignorava a memória e as duas metades deste par
    # **passavam**. A asserção estava sobre o alvo errado: media a memória, não a rota.
    #
    # Contar invocações do corretor é a única quantidade que muda quando — e só quando — o
    # trabalho é de facto saltado.
    assert corridas == [], (
        f"a execução CORREU o corretor {len(corridas)} vez(es) mesmo com a correção já guardada. "
        f"Desde U24 ela reusa — sem isso, cada ajuste de parâmetro de análise re-paga os "
        f"209,9 s do `pos_hilic` sob `serrf`, e explorar parâmetros fica impraticável"
    )


# ---------------------------------------------------------------------------
# A exportação com os eixos do pré-processamento — a matriz normalizada no formato do MetaboAnalyst
# ---------------------------------------------------------------------------
#
# **Sobre a `tiny_clean` e não sobre os quatro modos, e isto NÃO é dado fabricado no lugar
# do real.** O que estes testes medem é a TRAVESSIA de um argumento — o eixo chega ao pipeline, o
# pipeline chega ao escritor, a recusa nomeia a causa — e nenhum número afirmado vem dos arquivos
# do laboratório. Trocar o dado real é proibido quando a quantidade medida é dele; aqui
# a quantidade é a própria matriz construída à mão, e o esperado é calculado sobre ela pelas
# funções do núcleo. O que depende de um modo real — o `pos_c18` com soma e `log10` — vive no fim
# desta seção, sobre o arquivo do laboratório, e PULA quando ele não está.

FIXTURES = Path(__file__).resolve().parent.parent / "data" / "fixtures"
MODO_DA_FIXTURE = "tiny"

# O pedido de normalização que a aba «Normalização» manda, menos o que cada teste troca. A
# imputação vai EXPLÍCITA porque o padrão da exportação é `none`, e `none` com normalização ativa é
# a combinação que `PipelineSpec` recusa.
NORMALIZA_POR_SOMA: dict[str, str] = {
    "imputation": "one_fifth_min",
    "sample_normalization": "sum",
}


def _arquivos_da_fixture(medidas: Path | None = None) -> dict[str, tuple[str, bytes, str]]:
    medidas = FIXTURES / "tiny_clean.csv" if medidas is None else medidas
    return {
        "measurements": (medidas.name, medidas.read_bytes(), "text/csv"),
        "sample_list": (
            "tiny_clean_sample_list.csv",
            (FIXTURES / "tiny_clean_sample_list.csv").read_bytes(),
            "text/csv",
        ),
    }


def _matriz_da_fixture(medidas: Path | None = None) -> RefinoMatrix:
    medidas = FIXTURES / "tiny_clean.csv" if medidas is None else medidas
    return load_matrix(medidas, FIXTURES / "tiny_clean_sample_list.csv", MODO_DA_FIXTURE)


def _medidas_com_feature_constante(pasta: Path) -> Path:
    """A `tiny_clean` com a última feature zerada em todas as amostras — nunca detectada.

    **Existe para a planta cair onde as duas coisas diferem.** Na `tiny_clean` intacta toda
    feature varia, e o executor com todos os eixos neutros devolve a mesma matriz: um pedido sem os
    campos novos que passasse pelo executor daria o mesmo arquivo, e a byte-identidade passaria sem
    provar que ele foi saltado. A feature constante é a que o filtro de variância quase nula
    descarta — e o arquivo de hoje a tem.
    """
    linhas = (FIXTURES / "tiny_clean.csv").read_text(encoding="utf-8").splitlines()
    nome, *valores = linhas[-1].split(",")
    linhas[-1] = ",".join([nome, *("0" for _ in valores)])
    alvo = pasta / "tiny_clean_com_constante.csv"
    alvo.write_text("\n".join(linhas) + "\n", encoding="utf-8")
    return alvo


def _spec_neutra(**eixos: Any) -> PipelineSpec:
    """A especificação que a rota monta, com os eixos que o teste pede e o resto NEUTRO."""
    neutra: dict[str, Any] = {
        "correction": "none",
        "n_batches": 1,
        "rsd_cutoff": None,
        "ranking_filter": "none",
        "ranking_keep_fraction": None,
        "ranking_filter_2": "none",
        "ranking_keep_fraction_2": None,
        "rsd_variant": "outlier_removed",
        "missing_threshold": None,
        "imputation": "none",
        "sample_normalization": "none",
        "normalization_reference": None,
        "normalization_factors": None,
        "transformation": "none",
        "scaling": "none",
        "seed": 42,
    }
    return PipelineSpec(**{**neutra, **eixos})


def _bytes_do_escritor(matriz: RefinoMatrix) -> bytes:
    """O que `write_metaboanalyst` escreve para esta matriz — o esperado, por outro caminho."""
    with tempfile.TemporaryDirectory() as temporario:
        destino = Path(temporario) / "esperado.csv"
        write_metaboanalyst(matriz, destino)
        return destino.read_bytes()


def _pela_cli_da_fixture(
    tmp_path: Path, *argumentos: str, medidas: Path | None = None
) -> tuple[int, bytes | None]:
    """`refino exportar` sobre a fixture, EM PROCESSO: o código e os bytes, se houver arquivo.

    Em processo e não pelo ponto de entrada instalado, ao contrário de `_pela_cli`: o instalado
    importa o pacote do ambiente, e o que se confere aqui é o código desta árvore.
    """
    alvo = tmp_path / f"cli-{uuid4().hex}.csv"
    codigo = cli_main(
        [
            "exportar",
            "--measurements",
            str(FIXTURES / "tiny_clean.csv" if medidas is None else medidas),
            "--sample-list",
            str(FIXTURES / "tiny_clean_sample_list.csv"),
            "--mode",
            MODO_DA_FIXTURE,
            "--output",
            str(alvo),
            *argumentos,
        ]
    )
    return codigo, (alvo.read_bytes() if alvo.is_file() else None)


def test_sem_os_eixos_novos_a_exportacao_e_byte_a_byte_a_de_hoje(
    cliente: TestClient, tmp_path: Path
) -> None:
    """**A metade de compatibilidade do par causal da byte-identidade, pela API e pela CLI.**

    O esperado vem do ESCRITOR sobre a matriz ingerida — que é o que a rota fazia antes de os eixos
    existirem: `none` devolve a matriz inalterada em valor, e a escrita é a única transformação.
    Este teste foi corrido sobre a árvore anterior aos eixos e passou; é isso que autoriza chamar o
    esperado de *«o arquivo de hoje»*.

    **Os campos neutros EXPLÍCITOS dão o mesmo arquivo que a ausência**, e é a metade que impede um
    padrão diferente do neutro de se esconder atrás da omissão: se a rota imputasse por padrão, o
    pedido sem campos e o pedido com `imputation=none` divergiriam.

    **A matriz tem uma feature constante, e a âncora afirma porquê:** o executor com tudo neutro a
    descarta. Um pedido sem eixo que passasse pelo executor perderia essa feature, e esta igualdade
    acusaria — na `tiny_clean` intacta ela passaria do mesmo jeito, e não separaria nada.
    """
    medidas = _medidas_com_feature_constante(tmp_path)
    ingerida = _matriz_da_fixture(medidas)
    pelo_executor, _ = process_matrix(ingerida, _spec_neutra())
    assert pelo_executor.n_features == ingerida.n_features - 1, (
        "âncora: o executor com tudo neutro não descartou a feature constante, então correr ou não "
        "correr o executor dá o mesmo arquivo e a igualdade abaixo não prova que ele foi saltado"
    )
    esperado = _bytes_do_escritor(ingerida)
    neutros = {
        "imputation": "none",
        "sample_normalization": "none",
        "transformation": "none",
        "scaling": "none",
        "ranking_filter": "none",
        "ranking_filter_2": "none",
    }

    omitido = reclama(
        cliente,
        "/matrizes/exportar",
        files=_arquivos_da_fixture(medidas),
        data={"mode": MODO_DA_FIXTURE},
    )
    explicito = reclama(
        cliente,
        "/matrizes/exportar",
        files=_arquivos_da_fixture(medidas),
        data={"mode": MODO_DA_FIXTURE, **neutros},
    )
    codigo, pela_cli = _pela_cli_da_fixture(tmp_path, medidas=medidas)

    assert len(esperado) > 100, "âncora: o esperado é vazio e as igualdades seriam vácuas"
    assert (omitido.status_code, explicito.status_code, codigo) == (200, 200, 0), (
        omitido.text[:200],
        explicito.text[:200],
    )
    assert omitido.content == esperado, "sem os campos novos, a API mudou o arquivo de hoje"
    assert explicito.content == esperado, "os campos neutros explícitos mudaram o arquivo"
    assert pela_cli == esperado, "sem as opções novas, a CLI mudou o arquivo de hoje"


@pytest.mark.parametrize(
    ("eixos", "liga"),
    [
        ({}, False),
        ({"rsd_cutoff": 0.3}, True),
        ({"missing_threshold": 0.5}, True),
        ({"ranking_filter": "iqr"}, True),
        ({"ranking_filter_2": "mean"}, True),
        ({"imputation": "one_fifth_min"}, True),
        ({"imputation": "one_fifth_min", "sample_normalization": "sum"}, True),
        ({"imputation": "one_fifth_min", "transformation": "log10"}, True),
        ({"imputation": "one_fifth_min", "scaling": "pareto"}, True),
        # Os campos que só são LIDOS por uma etapa ligada não ligam nada sozinhos.
        ({"ranking_keep_fraction": 0.6, "ranking_keep_fraction_2": 0.9}, False),
        ({"normalization_reference": "TINY_QC1"}, False),
        ({"normalization_factors": (("TINY_QC1", 2.0),)}, False),
    ],
)
def test_quais_eixos_ligam_o_executor_na_exportacao(eixos: dict[str, Any], liga: bool) -> None:
    """`requests_preprocessing` decide se a exportação corre o executor, e tem os dois desfechos.

    Cada campo que liga uma etapa liga sozinho, e os que só a parametrizam não ligam. É a função
    que a rota e a CLI leem, e o teste de byte-identidade acima só exercita o desfecho `False`.
    """
    assert requests_preprocessing(_spec_neutra(**eixos)) is liga


def test_com_os_eixos_o_arquivo_e_o_da_matriz_processada_e_api_e_cli_concordam(
    cliente: TestClient, tmp_path: Path
) -> None:
    """**A outra metade do par causal: com os eixos, o arquivo MUDA, e muda para o certo.**

    **A planta cai sobre dado em que as duas coisas diferem**, e a âncora o afirma antes: a
    soma leva cada linha a somar 1000, então a matriz processada não é a ingerida. Sem a âncora, um
    pedido cujos eixos a rota ignorasse daria o arquivo de hoje e a desigualdade abaixo passaria por
    acaso de um dado em que normalizar não muda nada.

    O esperado é o escritor sobre `process_matrix` — o mesmo executor do `processar` —, então a
    igualdade afirma que a exportação descarrega a MESMA matriz que a aba mostra.
    """
    ingerida = _matriz_da_fixture()
    processada, _ = process_matrix(ingerida, _spec_neutra(**NORMALIZA_POR_SOMA))
    assert not np.allclose(processada.values, ingerida.values), (
        "âncora: a normalização por soma não mudou a matriz, e a diferença abaixo seria acaso"
    )
    esperado = _bytes_do_escritor(processada)

    pela_api = reclama(
        cliente,
        "/matrizes/exportar",
        files=_arquivos_da_fixture(),
        data={"mode": MODO_DA_FIXTURE, **NORMALIZA_POR_SOMA},
    )
    codigo, pela_cli = _pela_cli_da_fixture(
        tmp_path, "--imputation", "one_fifth_min", "--sample-normalization", "sum"
    )

    assert pela_api.status_code == 200, pela_api.text[:300]
    assert pela_api.content == esperado, (
        "o arquivo não é o da matriz processada: a rota aceitou os eixos e escreveu outra matriz"
    )
    assert pela_api.content != _bytes_do_escritor(ingerida), "os eixos não mudaram o arquivo"
    assert codigo == 0
    assert pela_cli == pela_api.content, "a CLI e a API divergem com os mesmos eixos"


def test_o_escalonamento_e_recusado_com_422_e_a_frase_nomeia_o_eixo_e_a_alternativa(
    cliente: TestClient,
) -> None:
    """O escalonamento centra cada feature na média, então SEMPRE produz negativos — e o `[F-2]`.

    **A âncora afirma que a causa é o escalonamento e não o dado:** a matriz ingerida não tem
    negativo, e a processada tem. Sem ela, a recusa podia vir de um negativo que já estava no
    arquivo, e a frase a nomear o escalonamento estaria certa por coincidência.

    A recusa chega na RECLAMAÇÃO e não no disparo, porque depende do cálculo — é a fronteira que
    `exportar` já declara para as recusas do escritor —, e o código é `422`.
    """
    eixos = {**NORMALIZA_POR_SOMA, "transformation": "log10", "scaling": "auto"}
    ingerida = _matriz_da_fixture()
    processada, _ = process_matrix(ingerida, _spec_neutra(**eixos))
    assert not (np.asarray(ingerida.values) < 0).any(), "âncora: a fixture já tinha negativo"
    assert (np.asarray(processada.values) < 0).any(), "âncora: o escalonamento não gerou negativo"

    resposta = reclama(
        cliente,
        "/matrizes/exportar",
        files=_arquivos_da_fixture(),
        data={"mode": MODO_DA_FIXTURE, **eixos},
    )

    assert resposta.status_code == 422, f"esperava 422, veio {resposta.status_code}"
    detalhe = str(resposta.json()["detail"])
    assert "escalonamento" in detalhe and "`auto`" in detalhe, (
        f"a recusa não nomeia o eixo que gerou os negativos: {detalhe[:400]}"
    )
    assert "`none`" in detalhe and "MetaboAnalyst" in detalhe, (
        f"a recusa não oferece a alternativa — transformação e escalonamento em `none`: "
        f"{detalhe[:400]}"
    )


def test_a_alternativa_sem_transformacao_e_escalonamento_sai_sem_recusa(
    cliente: TestClient,
) -> None:
    """O MESMO pedido do teste de cima, com transformação e escalonamento em `none` — a alternativa.

    Âncora: a matriz que sai sem as duas não tem negativo. É a condição sob a qual «sai sem recusa»
    é a afirmação certa, e não a sorte de um dado.
    """
    eixos = {**NORMALIZA_POR_SOMA, "transformation": "none", "scaling": "none"}
    processada, _ = process_matrix(_matriz_da_fixture(), _spec_neutra(**eixos))
    assert not (np.asarray(processada.values) < 0).any(), "âncora: a alternativa tem negativo"

    resposta = reclama(
        cliente,
        "/matrizes/exportar",
        files=_arquivos_da_fixture(),
        data={"mode": MODO_DA_FIXTURE, **eixos},
    )

    assert resposta.status_code == 200, resposta.text[:300]
    assert resposta.content == _bytes_do_escritor(processada)


def test_soma_e_log10_sem_negativo_saem_sem_recusa(cliente: TestClient) -> None:
    """Soma, `log10` e escalonamento `none`: sai no formato, **se a matriz não tiver negativo**.

    A condição é afirmada ANTES, sobre a matriz que a rota vai escrever. Na `tiny_clean` a soma
    leva os valores a 67 a 333, e o `log10` deles é positivo. A recusa do `log` não é incondicional
    como a do escalonamento: depende de haver intensidade abaixo de 1 depois da normalização.
    """
    eixos = {**NORMALIZA_POR_SOMA, "transformation": "log10", "scaling": "none"}
    processada, _ = process_matrix(_matriz_da_fixture(), _spec_neutra(**eixos))
    assert not (np.asarray(processada.values) < 0).any(), (
        "âncora: a soma com `log10` gerou negativo nesta matriz, e «sai sem recusa» não se aplica"
    )

    resposta = reclama(
        cliente,
        "/matrizes/exportar",
        files=_arquivos_da_fixture(),
        data={"mode": MODO_DA_FIXTURE, **eixos},
    )

    assert resposta.status_code == 200, resposta.text[:300]
    assert resposta.content == _bytes_do_escritor(processada)


def test_dois_pedidos_que_so_diferem_no_escalonamento_sao_trabalhos_diferentes(
    cliente: TestClient,
) -> None:
    """**O defeito da chave incompleta na exportação**: o escalonamento fora da chave serviria o
    arquivo errado.

    A chave decide se um disparo é reclamação de um trabalho existente. Dois pedidos que diferem só
    no escalonamento com a mesma chave receberiam o MESMO artefato — e o segundo levaria para o
    MetaboAnalyst a matriz escalonada à maneira do primeiro, com `200`.

    **A âncora é a deduplicação funcionar:** o mesmo pedido repetido é reclamado, com o mesmo
    identificador. Sem ela, identificadores diferentes podiam vir de uma chave que mudasse a cada
    chamada, e a desigualdade de baixo não provaria nada sobre o escalonamento.
    """
    base = {"mode": MODO_DA_FIXTURE, **NORMALIZA_POR_SOMA, "seed": 7}

    def dispara(escalonamento: str) -> dict[str, Any]:
        resposta = cliente.post(
            "/matrizes/exportar",
            files=_arquivos_da_fixture(),
            data={**base, "scaling": escalonamento},
        )
        assert resposta.status_code == 202, resposta.text[:300]
        corpo: dict[str, Any] = resposta.json()
        return corpo

    primeiro = dispara("none")
    repetido = dispara("none")
    pareto = dispara("pareto")

    assert repetido["identificador"] == primeiro["identificador"] and repetido["reclamado"], (
        "âncora: o mesmo pedido não foi reclamado, então a deduplicação não está a operar"
    )
    assert pareto["identificador"] != primeiro["identificador"], (
        "dois pedidos que só diferem no escalonamento têm a MESMA chave de trabalho: o segundo "
        "receberia o arquivo do primeiro"
    )
    assert not pareto["reclamado"]


@pytest.mark.parametrize(
    ("campo", "valor"),
    [
        ("scaling", "autoo"),
        ("transformation", "log"),
        ("imputation", "quinto"),
        ("sample_normalization", "soma"),
        ("ranking_filter", "iqrr"),
        ("ranking_filter_2", "sdd"),
    ],
)
def test_valor_fora_do_vocabulario_e_422_no_disparo_e_nao_cai_num_padrao(
    cliente: TestClient, campo: str, valor: str
) -> None:
    """O `Literal` atravessa a fronteira com guarda, e a recusa é no DISPARO.

    O modo de falha que ela evita é o que não levanta: um campo que a rota não conhecesse seria
    descartado pelo `FastAPI`, e o pedido sairia com `202` e o padrão — o cliente pediria um
    escalonamento e receberia a matriz sem ele.
    """
    resposta = cliente.post(
        "/matrizes/exportar",
        files=_arquivos_da_fixture(),
        data={"mode": MODO_DA_FIXTURE, **NORMALIZA_POR_SOMA, campo: valor},
    )

    assert resposta.status_code == 422, (
        f"`{campo}={valor}` foi aceito com {resposta.status_code}: caiu num padrão"
    )
    assert valor in str(resposta.json()["detail"]), "a recusa não nomeia o valor recusado"


def test_normalizar_sem_imputacao_e_recusado_e_a_rota_nao_imputa_por_omissao(
    cliente: TestClient,
) -> None:
    """O padrão da imputação na exportação é `none`, e a consequência vai afirmada.

    Normalizar sem imputar é a combinação que `PipelineSpec` recusa. Com `one_fifth_min` por
    padrão, como no `processar`, este pedido passaria — e a imputação seria escolha tomada por
    omissão, num arquivo que até aqui saía com o ausente como célula vazia.
    """
    resposta = cliente.post(
        "/matrizes/exportar",
        files=_arquivos_da_fixture(),
        data={"mode": MODO_DA_FIXTURE, "sample_normalization": "sum"},
    )

    assert resposta.status_code == 422, f"esperava 422, veio {resposta.status_code}"
    assert "imputation" in str(resposta.json()["detail"])


def test_os_fatores_enviados_chegam_ao_pipeline_da_exportacao(cliente: TestClient) -> None:
    """O terceiro arquivo do pedido — o CSV de fatores — é lido e resolvido como no `processar`.

    Âncora: os fatores não são todos iguais, então a matriz dividida por eles não é a ingerida. Um
    upload que a rota não lesse chegaria como `None`, e `supplied_factors` sem fatores é recusado —
    a igualdade abaixo só passa se o mapa atravessou.
    """
    ingerida = _matriz_da_fixture()
    fatores = tuple((nome, float(i + 1)) for i, nome in enumerate(ingerida.sample_names))
    eixos = {"imputation": "one_fifth_min", "sample_normalization": "supplied_factors"}
    processada, _ = process_matrix(ingerida, _spec_neutra(**eixos, normalization_factors=fatores))
    assert not np.allclose(processada.values, ingerida.values), (
        "âncora: os fatores não mudaram nada"
    )
    csv_dos_fatores = "".join(f"{nome},{fator:g}\n" for nome, fator in fatores).encode("utf-8")

    resposta = reclama(
        cliente,
        "/matrizes/exportar",
        files={
            **_arquivos_da_fixture(),
            "normalization_factors": ("fatores.csv", csv_dos_fatores, "text/csv"),
        },
        data={"mode": MODO_DA_FIXTURE, **eixos},
    )

    assert resposta.status_code == 200, resposta.text[:300]
    assert resposta.content == _bytes_do_escritor(processada)


def test_o_corte_que_esvazia_a_matriz_e_409_na_reclamacao_e_3_na_cli(
    cliente: TestClient, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """O pré-processamento que descarta todas as features recusa como no `processar`.

    `409` e não `500`: o pedido estava bem formado, e o que falhou foi o encontro dele com esta
    matriz — o corte que o profissional escolheu. A CLI dá o código 3 que o `processar` dá. A
    âncora afirma que o executor de facto esvazia esta matriz com este corte.
    """
    with pytest.raises(EmptyMatrixError):
        process_matrix(_matriz_da_fixture(), _spec_neutra(rsd_cutoff=1e-6))

    resposta = reclama(
        cliente,
        "/matrizes/exportar",
        files=_arquivos_da_fixture(),
        data={"mode": MODO_DA_FIXTURE, "rsd_cutoff": 1e-6},
    )
    codigo, arquivo = _pela_cli_da_fixture(tmp_path, "--rsd-cutoff", "1e-6")

    assert resposta.status_code == 409, f"esperava 409, veio {resposta.status_code}"
    assert "filter_by_qc_rsd" in str(resposta.json()["detail"]), (
        "a etapa que esvaziou não é nomeada"
    )
    assert codigo == EXIT_UNAVAILABLE, capsys.readouterr().err
    assert arquivo is None


def test_cli_escalonamento_recusado_sai_com_2_e_a_frase_nomeia_o_eixo(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A CLI espelha a rota: o mesmo escritor recusa, com a mesma frase, e o código é o de uso."""
    codigo, arquivo = _pela_cli_da_fixture(
        tmp_path,
        "--imputation",
        "one_fifth_min",
        "--sample-normalization",
        "sum",
        "--scaling",
        "auto",
    )

    erro = capsys.readouterr().err
    assert codigo == EXIT_USAGE, erro
    assert arquivo is None, "a CLI escreveu um arquivo que o formato recusa"
    assert "escalonamento" in erro and "MetaboAnalyst" in erro, erro


def test_cli_valor_fora_do_vocabulario_sai_com_2_antes_de_ler_a_matriz(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A guarda na outra fronteira: `--scaling autoo` é erro de uso, e não o padrão."""
    codigo, arquivo = _pela_cli_da_fixture(tmp_path, "--scaling", "autoo")

    assert codigo == EXIT_USAGE
    assert arquivo is None
    assert "autoo" in capsys.readouterr().err


def test_pos_c18_soma_e_log10_gera_negativo_e_a_recusa_nomeia_o_log(cliente: TestClient) -> None:
    """**No `pos_c18`, soma e `log10` NÃO sai no formato, e o motivo é derivado e não medido.**

    O `SumNorm` leva cada linha a somar 1000. Com mais de mil features a média da linha fica abaixo
    de 1, então há intensidade abaixo de 1 em toda linha, e o `log10` dela é negativo. A âncora
    afirma a premissa sobre o arquivo em vez de a supor: se ela falhar, o desfecho a esperar é o
    outro, e o teste diz qual.

    E a alternativa da recusa, no mesmo modo: sem transformação e sem escalonamento, sai.
    """
    modo = "pos_c18"
    ingerida = load_matrix(measurements_path(modo), sample_list_path(modo), modo)
    com_log = {**NORMALIZA_POR_SOMA, "transformation": "log10", "scaling": "none"}
    processada, _ = process_matrix(ingerida, _spec_neutra(**com_log))
    negativos = int((np.asarray(processada.values) < 0).sum())
    assert negativos > 0, (
        f"premissa: soma e `log10` no {modo} não geraram negativo ({processada.n_features} "
        "features). Então o desfecho certo é sair sem recusa, e este teste está a afirmar o outro"
    )

    recusada = reclama(
        cliente, "/matrizes/exportar", files=_arquivos(modo), data={"mode": modo, **com_log}
    )
    alternativa = reclama(
        cliente,
        "/matrizes/exportar",
        files=_arquivos(modo),
        data={"mode": modo, **NORMALIZA_POR_SOMA},
    )

    assert recusada.status_code == 422, f"esperava 422, veio {recusada.status_code}"
    detalhe = str(recusada.json()["detail"])
    assert "log10" in detalhe and "`none`" in detalhe, detalhe[:400]
    assert alternativa.status_code == 200, alternativa.text[:300]
