"""Os DOIS formatos da matriz corrigida, de uma correção só.

## A medição que decidiu o desenho

As duas formas são **duas serializações da mesma matriz corrigida**, não dois cálculos. Medido no
`neg_c18` com `loess`: corrigir custa **13,11 s**, escrever custa **0,10 s** — **0,73%**. Servir os
dois por dois disparos faria a correção correr duas vezes; servi-los do mesmo trabalho é o que
torna a espera igual.

## Por que não um ZIP, e por que não uma rota nova

As duas alternativas foram medidas e recusadas: o ZIP põe três artefatos numa vaga cuja derivação é
**bytes por vaga**, e uma rota nova dispara **~11** portões contra **0** de repetir a chamada.
Aqui são dois arquivos numa vaga — 2,00 vezes, medido — e o `formato` entra como query na rota que
já existe.

## O que cada forma CARREGA, e é isto que faz «ambas» não ser redundância

| | linhas de metadados | coluna de índice | carrega lote/tipo/ordem |
|---|---|---|---|
| `toserrf` — alimenta o motor | 3 | sim | **sim** |
| `normalizado` — o que o motor devolve | nenhuma | não | **não** |
"""

from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from refino.api.app import build_app
from refino.api.rotas import FORMATO_ALTERNATIVO, FORMATO_PADRAO, FORMATOS_DO_ARTEFATO
from refino.io.readers import load_matrix
from refino.io.writers import write_normalizado
from tests.conftest import measurements_path, normalized_path, sample_list_path

MODO = "neg_c18"


@pytest.fixture(scope="module")
def cliente() -> TestClient:
    return TestClient(build_app())


def _arquivos(mode: str = MODO) -> dict[str, tuple[str, bytes, str]]:
    return {
        "measurements": (
            f"measurements_{mode}.csv",
            measurements_path(mode).read_bytes(),
            "text/csv",
        ),
        "sample_list": (f"Sample_List_{mode}.csv", sample_list_path(mode).read_bytes(), "text/csv"),
    }


def _ate_concluir(cliente: TestClient, identificador: str, tentativas: int = 300) -> None:
    """Espera o trabalho sair de `202`. **Teto para REPROVAR em vez de pendurar.**"""
    for _ in range(tentativas):
        if cliente.get(f"/trabalhos/{identificador}").status_code != 202:
            return
        time.sleep(0.2)
    raise AssertionError(f"o trabalho {identificador} não concluiu em {tentativas} tentativas")


# ---------------------------------------------------------------------------
# O ESCRITOR: byte-idêntico ao oráculo, que é o caminho `none` do R
# ---------------------------------------------------------------------------


@pytest.mark.golden
@pytest.mark.parametrize("modo", ["neg_c18", "pos_hilic", "pos_c18", "neg_hilic"])
def test_o_normalizado_e_byte_identico_ao_golden_do_r(modo: str, tmp_path) -> None:  # type: ignore[no-untyped-def]
    """**O caminho `none` é oráculo EXATO** (fato de domínio), então a comparação é por igualdade.

    Não é «parecido» nem «a 1e-9»: são os mesmos bytes. O formato foi medido no golden antes de o
    escritor existir — CRLF, **zero aspas**, cabeçalho `label,<amostras>`, e os 15 dígitos
    significativos que `_format` já dava ao `toserrf`.

    **Reusar `_format` não é economia:** dois formatadores divergiriam na terceira casa entre as
    duas saídas, num sítio onde ninguém compara.
    """
    matriz = load_matrix(measurements_path(modo), sample_list_path(modo), modo)
    saida = tmp_path / f"{modo}.csv"
    write_normalizado(matriz, saida)

    nosso, ouro = saida.read_bytes(), normalized_path(modo, "none").read_bytes()

    assert len(ouro) > 1_000_000, "âncora: o golden é pequeno demais e a igualdade seria vácua"
    # **`strict=False` é deliberado e não descuido.** Os dois podem ter tamanhos diferentes — é
    # justamente o caso em que a mensagem precisa de sair —, e `strict=True` levantaria
    # `ValueError` no meio da f-string, trocando a reprovação informativa por um erro do arnês.
    divergentes = sum(a != b for a, b in zip(nosso, ouro, strict=False))
    assert nosso == ouro, (
        f"{modo}: a saída difere do golden do R em {divergentes} bytes de {len(ouro)}"
    )


def test_as_duas_formas_carregam_coisas_diferentes(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """**A planta cai onde as duas diferem**, e é o que torna «ambas» não redundante.

    Um teste que só comparasse tamanhos passaria com duas cópias do mesmo arquivo. O que separa é
    que o `toserrf` abre com `NA,batch` — a primeira das três linhas de metadados — e o
    `normalizado` abre com `label`, e **não tem** nenhuma delas.
    """
    from refino.io.writers import write_toserrf

    matriz = load_matrix(measurements_path(MODO), sample_list_path(MODO), MODO)
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    write_toserrf(matriz, a)
    write_normalizado(matriz, b)

    forma_de_entrada, forma_de_saida = a.read_bytes(), b.read_bytes()

    assert forma_de_entrada.startswith(b"NA,batch"), "o `toserrf` perdeu as linhas de metadados"
    assert forma_de_saida.startswith(b"label,"), "o `normalizado` não abre com o cabeçalho único"
    for metadado in (b"batch", b"sampleType", b"time"):
        assert metadado not in forma_de_saida.split(b"\r\n")[0], (
            f"o `normalizado` carrega `{metadado.decode()}`, e ele DESCARTA os três metadados"
        )

    # **A COLUNA DE ÍNDICE, e esta asserção existe porque a planta dela ficou INERTE.**
    #
    # As três acima olham o **cabeçalho** — a primeira linha —, e a coluna de índice vive nas
    # linhas de **feature**. Plantar `writerow([str(index), feature, ...])` no escritor não
    # derrubava nada: o teste passava com o `normalizado` a ganhar exactamente a coluna que ele
    # existe para não ter. Foi o par causal que o mostrou, e não a leitura do teste.
    #
    # A propriedade é dupla, e por isso as duas asserções: a linha do `toserrf` **abre com o
    # índice** e a do `normalizado` **abre com o nome da feature**.
    primeira_do_toserrf = forma_de_entrada.split(b"\r\n")[4]
    primeira_do_normalizado = forma_de_saida.split(b"\r\n")[1]

    assert primeira_do_toserrf.startswith(b"1,"), (
        f"o `toserrf` perdeu a coluna de índice: {primeira_do_toserrf[:40]!r}"
    )
    assert not primeira_do_normalizado.split(b",")[0].isdigit(), (
        f"o `normalizado` GANHOU coluna de índice — a primeira célula da linha de feature é um "
        f"número: {primeira_do_normalizado[:40]!r}"
    )
    assert primeira_do_normalizado.startswith(matriz.feature_names[0].encode()), (
        "a linha de feature do `normalizado` não abre com o nome da feature"
    )


# ---------------------------------------------------------------------------
# A ROTA: quatro desfechos do `formato`, e o padrão é o de sempre
# ---------------------------------------------------------------------------


def test_o_vocabulario_do_formato_tem_os_dois_e_o_padrao_e_o_de_sempre() -> None:
    """O domínio é fechado, e o primeiro é o que a ausência devolve."""
    assert FORMATOS_DO_ARTEFATO == (FORMATO_PADRAO, FORMATO_ALTERNATIVO)
    assert FORMATO_PADRAO == "toserrf"


@pytest.mark.slow
def test_um_trabalho_serve_os_dois_e_sem_formato_nada_muda(cliente: TestClient) -> None:
    """**Par causal sobre a compatibilidade**, e é a metade que uma implementação desatenta perde.

    Sem `formato`, a resposta tem de ser **byte a byte** a de antes — é o critério, que
    afirma identidade com o que a CLI escreve. Afirmar só que o alternativo existe passaria com uma
    implementação que tivesse trocado o padrão.
    """
    aceito = cliente.post(
        "/matrizes/corrigir", files=_arquivos(), data={"mode": MODO, "method": "none"}
    )
    assert aceito.status_code == 202, aceito.text[:300]
    identificador = aceito.json()["identificador"]
    _ate_concluir(cliente, identificador)

    sem = cliente.get(f"/trabalhos/{identificador}")
    explicito = cliente.get(f"/trabalhos/{identificador}", params={"formato": FORMATO_PADRAO})
    outro = cliente.get(f"/trabalhos/{identificador}", params={"formato": FORMATO_ALTERNATIVO})

    assert (sem.status_code, explicito.status_code, outro.status_code) == (200, 200, 200)
    assert sem.content == explicito.content, "`formato=toserrf` não é o mesmo que a ausência"
    assert sem.content.startswith(b"NA,batch"), "o padrão deixou de ser o `toserrf`"
    assert outro.content.startswith(b"label,"), "o alternativo não é a forma do resultado"
    assert sem.content != outro.content, (
        "os dois formatos vieram iguais: o trabalho pode estar a servir o mesmo arquivo duas vezes"
    )


@pytest.mark.slow
def test_formato_fora_do_vocabulario_e_recusado_com_422(cliente: TestClient) -> None:
    """A guarda da regra 2 num domínio que não é `Literal`.

    **O modo de falha que ela evita é o que não levanta:** `formato=normalisado` — com `s` — cairia
    num `else` razoável e devolveria o `toserrf`. O cliente pediria uma forma, receberia a outra, e
    o nome no `Content-Disposition` até pareceria certo.
    """
    aceito = cliente.post(
        "/matrizes/corrigir", files=_arquivos(), data={"mode": MODO, "method": "none"}
    )
    identificador = aceito.json()["identificador"]
    _ate_concluir(cliente, identificador)

    resposta = cliente.get(f"/trabalhos/{identificador}", params={"formato": "normalisado"})

    assert resposta.status_code == 422, resposta.text[:200]
    for rotulo in FORMATOS_DO_ARTEFATO:
        assert rotulo in resposta.text, f"a recusa não diz que `{rotulo}` é aceito"


@pytest.mark.slow
def test_pedir_o_alternativo_a_quem_nao_o_tem_e_404_e_nao_422(cliente: TestClient) -> None:
    """**As duas causas pedem ações opostas, e por isso têm códigos diferentes**.

    `422` diz *"corrija o parâmetro"*; `404` diz *"esta operação produz um arquivo só"*. O
    `processar` e o `exportar` não têm segunda forma, e isso é propriedade deles — não erro de quem
    pede. Colapsá-los num código faria o cliente procurar um erro de digitação que não existe.
    """
    aceito = cliente.post(
        "/matrizes/processar", files=_arquivos(), data={"mode": MODO, "imputation": "one_fifth_min"}
    )
    identificador = aceito.json()["identificador"]
    _ate_concluir(cliente, identificador)

    resposta = cliente.get(f"/trabalhos/{identificador}", params={"formato": FORMATO_ALTERNATIVO})

    assert resposta.status_code == 404, resposta.text[:200]
    assert "corrigir" in resposta.text, "o `404` não diz qual operação produz a forma pedida"


# ---------------------------------------------------------------------------
# A MATRIZ NORMALIZADA nos dois formatos: `processar` e `exportar` com os mesmos eixos
# ---------------------------------------------------------------------------


def test_a_matriz_normalizada_sai_nos_dois_formatos_e_e_a_mesma(cliente: TestClient) -> None:
    """O `toserrf` do `processar` e o MetaboAnalyst do `exportar` descarregam a MESMA matriz.

    A aba «Normalização» baixa a matriz normalizada nos dois formatos, e eles vêm de duas rotas. O
    que torna os dois arquivos o mesmo resultado é as duas rotas correrem o mesmo executor sobre a
    mesma especificação — e é isso que se afirma, com cada arquivo comparado ao seu escritor sobre
    UMA matriz processada aqui, pelo núcleo.

    **Sobre a `tiny_clean` da especificação, e não é dado no lugar do real:** o que se mede é a
    travessia dos eixos pelas duas rotas, e nenhum número vem dos arquivos do laboratório. **A
    âncora é a matriz processada não ser a ingerida** — sem ela, duas rotas que ignorassem os eixos
    concordariam por escreverem as duas a matriz crua.
    """
    import tempfile
    from pathlib import Path

    import numpy as np

    from refino.domain.spec import PipelineSpec
    from refino.io.writers import write_metaboanalyst, write_toserrf
    from refino.preprocessing.pipeline import process_matrix
    from tests.reclamacao import reclama

    fixtures = Path(__file__).resolve().parent.parent / "data" / "fixtures"
    medidas, lista = fixtures / "tiny_clean.csv", fixtures / "tiny_clean_sample_list.csv"
    arquivos = {
        "measurements": ("tiny_clean.csv", medidas.read_bytes(), "text/csv"),
        "sample_list": ("tiny_clean_sample_list.csv", lista.read_bytes(), "text/csv"),
    }
    eixos = {
        "imputation": "one_fifth_min",
        "sample_normalization": "sum",
        "transformation": "log10",
        "scaling": "none",
    }
    ingerida = load_matrix(medidas, lista, "tiny")
    spec = PipelineSpec(
        correction="none",
        n_batches=1,
        rsd_cutoff=None,
        ranking_filter="none",
        ranking_keep_fraction=None,
        ranking_filter_2="none",
        ranking_keep_fraction_2=None,
        rsd_variant="outlier_removed",
        missing_threshold=None,
        normalization_reference=None,
        normalization_factors=None,
        seed=42,
        **eixos,
    )
    processada, _ = process_matrix(ingerida, spec)
    assert not np.allclose(processada.values, ingerida.values), (
        "âncora: os eixos não mudaram a matriz, e as duas rotas concordariam pela matriz crua"
    )
    with tempfile.TemporaryDirectory() as temporario:
        destino_toserrf = Path(temporario) / "toserrf.csv"
        destino_metaboanalyst = Path(temporario) / "metaboanalyst.csv"
        write_toserrf(processada, destino_toserrf)
        write_metaboanalyst(processada, destino_metaboanalyst)
        esperado_toserrf = destino_toserrf.read_bytes()
        esperado_metaboanalyst = destino_metaboanalyst.read_bytes()

    toserrf = reclama(
        cliente, "/matrizes/processar", files=arquivos, data={"mode": "tiny", **eixos}
    )
    metaboanalyst = reclama(
        cliente, "/matrizes/exportar", files=arquivos, data={"mode": "tiny", **eixos}
    )

    assert (toserrf.status_code, metaboanalyst.status_code) == (200, 200), (
        toserrf.text[:200],
        metaboanalyst.text[:200],
    )
    assert toserrf.content == esperado_toserrf, "o `processar` não escreveu a matriz processada"
    assert metaboanalyst.content == esperado_metaboanalyst, (
        "o `exportar` não escreveu a mesma matriz que o `processar`: a aba baixaria dois arquivos "
        "que não são o mesmo resultado"
    )
