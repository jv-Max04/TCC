"""As bordas que a revisão de 24/10/2026 achou — C01, C02, C05, C06, C07, C09, C10, C14, C15, C19,
C20 e C21 do `oo.md`, as doze confirmadas por uma conferência independente.

**Um arquivo para o bloco, e não um teste em cada arquivo do assunto**, porque a evidência que
importa aqui é o PAR CAUSAL: cada teste que acusa um defeito foi visto REPROVAR na árvore anterior
ao conserto (`b1a53a1`) e passar nesta, e manter os testes juntos é o que deixa repetir as duas
corridas com um comando só. Os que passam nas duas árvores dizem porquê na docstring — são a
direção oposta, a que impede o conserto de absorver o caso legítimo, e um deles achou o defeito da
primeira versão do conserto do C07 antes do commit.

**Regra 1 do README.** O C01 é exercitado sobre o arquivo do laboratório, onde foi medido, e
faz `skip` com o motivo quando os dados faltam; o resto usa `tests/factories.py`, cujos pares são
construídos
para conter exatamente a borda que o teste nomeia.

**Escopo negativo.** Nenhum teste aqui mede desempenho nem abre porta: `TestClient` fala com a
aplicação em processo. E o C10 do multiverso é afirmado no DISPARO — o `422` —, não pela espera do
trabalho de fundo: o que impede o trabalho de ficar `pendente` para sempre é a rota recusar antes
de ele nascer, e a segunda rede, `_celula_na_posicao`, tem teste próprio.
"""

import csv
import inspect
import io
import json
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fastapi import UploadFile
from fastapi.testclient import TestClient
from sqlalchemy import event, text

from refino.api.app import build_app
from refino.api.progresso import _celula_na_posicao
from refino.api.rotas import _carrega, _motor_ou_503
from refino.cli import EXIT_INPUT_REJECTED, EXIT_OK, EXIT_USAGE, main
from refino.domain.matrix import RefinoMatrix
from refino.domain.metadata import SampleMetadata
from refino.domain.synthetic import SyntheticSpec
from refino.interface import cliente
from refino.io.readers import IngestionError, load_matrix, load_matrix_issues
from refino.io.sniffer import sniff_dialect
from refino.io.validators import check_mode_matches_content
from refino.io.writers import write_normalizado
from refino.multiverse.grid import GridDefinition
from refino.statistics.permanova import PermanovaSemPopulacaoError, permanova
from refino.synthetic.generator import generate
from tests.conftest import measurements_path, sample_list_path
from tests.factories import base_spec, render_pair, tiny_clean_spec

# Um modo que não tem a forma `<polaridade>_<coluna>`: `check_mode_matches_content` não acusa nada
# sobre ele, e os avisos que estes testes contam são só os da ingestão.
MODO = "sintetico"

# A grade de UMA célula, válida: o disparo que a aceita corre em centésimos, e cada teste do C10
# troca um eixo dela pelo valor que a borda exige.
GRADE_MINIMA: dict[str, str] = {
    "corrections": '["none"]',
    "batch_options": "[1]",
    "rsd_cutoffs": "[0.2]",
    "missing_thresholds": "[0.5]",
    "imputations": '["one_fifth_min"]',
    "sample_norms": '["none"]',
    "transformations": '["none"]',
    "scalings": '["none"]',
}


@pytest.fixture(scope="module")
def cliente_http() -> TestClient:
    """`raise_server_exceptions=False`, porque o `500` é o desfecho que o par causal compara.

    Com o padrão, o `TestClient` levantaria no teste a exceção do servidor, e a árvore anterior ao
    conserto reprovaria por exceção em vez de pelo código — a mesma reprovação, lida pior.
    """
    return TestClient(build_app(), raise_server_exceptions=False)


def _arquivos(
    medicoes: str | bytes,
    lista: str | bytes,
    *,
    nome_m: str = "m.csv",
    nome_s: str = "s.csv",
) -> dict[str, tuple[str, bytes, str]]:
    def em_bytes(conteudo: str | bytes) -> bytes:
        return conteudo.encode("utf-8") if isinstance(conteudo, str) else conteudo

    return {
        "measurements": (nome_m, em_bytes(medicoes), "text/csv"),
        "sample_list": (nome_s, em_bytes(lista), "text/csv"),
    }


def _par(n_features: int = 5, n_samples: int = 10, n_qcs: int = 5) -> dict[str, Any]:
    medicoes, lista = render_pair(base_spec(n_features, n_samples, n_qcs))
    return _arquivos(medicoes, lista)


def _grava(pasta: Path, medicoes: str | bytes, lista: str) -> tuple[Path, Path]:
    caminho_m, caminho_s = pasta / "measurements.csv", pasta / "Sample_List.csv"
    if isinstance(medicoes, str):
        caminho_m.write_text(medicoes, encoding="utf-8", newline="")
    else:
        caminho_m.write_bytes(medicoes)
    caminho_s.write_text(lista, encoding="utf-8", newline="")
    return caminho_m, caminho_s


def _grava_par(pasta: Path) -> tuple[Path, Path]:
    return _grava(pasta, *render_pair(base_spec()))


def _valores_da_chave(corpo: Any, chave: str) -> list[Any]:
    """Todo valor de `chave` no JSON, a qualquer profundidade — para não pinar onde ele mora."""
    if isinstance(corpo, dict):
        achados = [corpo[chave]] if chave in corpo else []
        return achados + [v for valor in corpo.values() for v in _valores_da_chave(valor, chave)]
    if isinstance(corpo, list):
        return [v for item in corpo for v in _valores_da_chave(item, chave)]
    return []


# ---------------------------------------------------------------------------
# C01 — `/matrizes/validar` descartava os avisos de uma matriz ACEITA
# ---------------------------------------------------------------------------


def _avisos_pelo_outro_caminho(medicoes: Path, lista: Path, mode: str) -> Counter[tuple[str, str]]:
    """O oráculo do C01 é a ingestão direta, e a asserção é a IGUALDADE.

    `load_matrix_issues` conta os problemas sem passar pela rota, e `check_mode_matches_content` é
    o aviso que a rota sempre devolveu. Um literal com a contagem esperada seria oráculo de si
    próprio no dia em que a validação ganhasse um código.
    """
    _, _, problemas = load_matrix_issues(medicoes, lista)
    nomes = load_matrix(medicoes, lista, mode).sample_names
    esperados = Counter((p.severity, p.code) for p in problemas if not p.is_error)
    esperados.update((p.severity, p.code) for p in check_mode_matches_content(mode, nomes))
    return esperados


def test_c01_validar_devolve_os_avisos_de_uma_matriz_aceita(
    cliente_http: TestClient, tmp_path: Path
) -> None:
    """O par de dois QCs da especificação dispara `INSUFFICIENT_QCS`, e a rota tem de devolvê-lo.

    A primeira asserção é a âncora contra a igualdade de duas listas vazias: sem aviso
    nenhum, a rota antiga e a nova devolveriam o mesmo.
    """
    medicoes, lista = render_pair(tiny_clean_spec())
    esperados = _avisos_pelo_outro_caminho(*_grava(tmp_path, medicoes, lista), MODO)
    assert esperados[("warning", "INSUFFICIENT_QCS")] == 1, (
        f"âncora: o par de dois QCs devia avisar `INSUFFICIENT_QCS`; a ingestão contou {esperados}"
    )

    resposta = cliente_http.post(
        "/matrizes/validar", files=_arquivos(medicoes, lista), data={"mode": MODO}
    )

    assert resposta.status_code == 200, resposta.text
    corpo = resposta.json()
    assert corpo["aceita"] is True, corpo["problemas"]
    devolvidos = Counter((p["severidade"], p["codigo"]) for p in corpo["problemas"])
    assert devolvidos == esperados, (
        f"a rota devolveu {dict(devolvidos)} e a ingestão contou {dict(esperados)}"
    )


def test_c01_validar_devolve_os_avisos_do_neg_c18(cliente_http: TestClient) -> None:
    """O mesmo, sobre o arquivo do laboratório em que o achado foi medido.

    Ali a janela escrevia «nenhum problema» sobre um arquivo que a ingestão aceita COM avisos — e
    os avisos são a informação que o profissional precisa antes de corrigir.
    """
    mode = "neg_c18"
    medicoes, lista = measurements_path(mode), sample_list_path(mode)
    esperados = _avisos_pelo_outro_caminho(medicoes, lista, mode)
    assert sum(esperados.values()) > 0, "âncora: o neg_c18 tem avisos de ingestão"

    resposta = cliente_http.post(
        "/matrizes/validar",
        files=_arquivos(medicoes.read_bytes(), lista.read_text(encoding="utf-8")),
        data={"mode": mode},
    )

    corpo = resposta.json()
    assert corpo["aceita"] is True, corpo["problemas"]
    devolvidos = Counter((p["severidade"], p["codigo"]) for p in corpo["problemas"])
    assert devolvidos == esperados, (
        f"a rota devolveu {dict(devolvidos)} e a ingestão contou {dict(esperados)}"
    )


# ---------------------------------------------------------------------------
# C02 — o `filename` do multipart virava componente de caminho
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("forma", ["relativo", "absoluto"])
def test_c02_o_nome_enviado_nao_vira_caminho(
    cliente_http: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, forma: str
) -> None:
    """`../x.csv` e um caminho absoluto gravavam o upload FORA do temporário, onde ele ficava.

    O temporário da rota nasce dentro de `tmp_path`, então o `../` aterra em `tmp_path`, onde o
    teste olha. A âncora é a aceitação: sem ela, um par recusado antes de ser gravado passaria
    pelo motivo errado.
    """
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    alvo = tmp_path / "escapou.csv"
    nome = "../escapou.csv" if forma == "relativo" else str(alvo)
    medicoes, lista = render_pair(base_spec())

    resposta = cliente_http.post(
        "/matrizes/validar", files=_arquivos(medicoes, lista, nome_m=nome), data={"mode": MODO}
    )

    assert resposta.status_code == 200, resposta.text
    assert resposta.json()["aceita"] is True, resposta.json()["problemas"]
    assert not alvo.exists(), f"o upload foi gravado fora do temporário da rota, em {alvo}"


@pytest.mark.parametrize(
    ("enviado", "exibido"),
    [
        pytest.param("measurements_neg_c18.csv", "measurements_neg_c18.csv", id="nome_comum"),
        pytest.param("../escapou.csv", "escapou.csv", id="relativo"),
        pytest.param("C:\\dados\\lab\\m.csv", "m.csv", id="windows"),
        pytest.param("..", "measurements.csv", id="so_pontos"),
        pytest.param(None, "measurements.csv", id="sem_nome"),
    ],
)
def test_c02_a_procedencia_guarda_o_nome_do_usuario_como_texto(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, enviado: str | None, exibido: str
) -> None:
    """O nome sai do CAMINHO e fica na PROCEDÊNCIA, que existe para guardá-lo.

    A primeira versão do conserto do C02 trocava a procedência por `measurements.csv` em todo
    upload, e o caso `nome_comum` é o que ela reprovaria — ele passa na árvore anterior, que é a
    direção que o conserto tem de preservar. Os outros quatro dizem o que sobra de um nome que não
    é só nome.
    """
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    medicoes, lista = render_pair(base_spec())
    envio_m = UploadFile(file=io.BytesIO(medicoes.encode("utf-8")), filename=enviado)
    envio_s = UploadFile(file=io.BytesIO(lista.encode("utf-8")), filename="Sample_List_x.csv")

    matriz = _carrega(envio_m, envio_s, MODO, "features-in-rows")

    assert matriz.provenance[0] == f"ingestão de {exibido} + Sample_List_x.csv"


def test_c02_dois_arquivos_com_o_mesmo_nome_ficam_cada_um_no_seu_papel(
    cliente_http: TestClient,
) -> None:
    """A janela manda os nomes de disco, e dois `dados.csv` de pastas diferentes são legítimos.

    Com o nome no caminho, o Sample_List sobrescrevia o measurements e a matriz era recusada.
    """
    medicoes, lista = render_pair(base_spec(n_samples=10, n_qcs=5))

    resposta = cliente_http.post(
        "/matrizes/validar",
        files=_arquivos(medicoes, lista, nome_m="dados.csv", nome_s="dados.csv"),
        data={"mode": MODO},
    )

    corpo = resposta.json()
    assert corpo["aceita"] is True, corpo["problemas"]
    assert corpo["n_samples"] == 10


# ---------------------------------------------------------------------------
# C05 — a matriz de um QC, aceita pela validação, derrubava as rotas com 500
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("rota", ["/matrizes/diagnosticar", "/matrizes/recomendar"])
def test_c05_um_qc_e_recusa_do_dado_e_nao_defeito_da_ferramenta(
    cliente_http: TestClient, rota: str
) -> None:
    """`422` que o CLIENTE lê como causa do dado — a pergunta é feita a ele, e não à frase.

    `ui/cliente.py` só diz que a causa é o arquivo quando o `detail` traz uma das
    `MARCAS_DE_CONTEUDO`; a primeira versão deste conserto devolvia `422` sem marca, e a tela
    diria «Faltou um campo». Perguntar a causa ao próprio cliente é o que o docstring dele pede:
    o teste dirige a rota real e afirma a classificação, em vez de reconstruir a frase.
    """
    arquivos = _par(n_features=5, n_samples=10, n_qcs=1)
    validacao = cliente_http.post("/matrizes/validar", files=arquivos, data={"mode": MODO})
    assert validacao.json()["aceita"] is True, (
        "âncora: a premissa do achado é a validação ACEITAR a matriz de um QC"
    )

    resposta = cliente_http.post(rota, files=arquivos, data={"mode": MODO})

    assert resposta.status_code == 422, (resposta.status_code, resposta.text[:300])
    assert cliente.causa_de(resposta.status_code, resposta.json()) == "dado", resposta.json()


# ---------------------------------------------------------------------------
# C06 — nome de feature com vírgula derrubava o escritor do normalizado
# ---------------------------------------------------------------------------


def test_c06_o_normalizado_escreve_e_rele_um_nome_com_virgula(tmp_path: Path) -> None:
    """Nome IUPAC com vírgula: a ingestão o aceita citado, e o escritor tinha de o citar também.

    A releitura pelo `csv` é o oráculo, e a largura igual de todas as linhas é o que distingue
    «citou» de «escreveu a vírgula crua», que partiria a célula em duas.
    """
    amostras = tuple(
        SampleMetadata(
            name=f"S{i}",
            injection_order=i + 1,
            sample_type="qc" if i < 2 else "sample",
            group="QC" if i < 2 else "G1",
        )
        for i in range(4)
    )
    nomes = ("1,3-Diaminopropane", "2.25_810.7739m/z")
    matriz = RefinoMatrix(
        values=np.arange(1.0, 9.0).reshape(4, 2),
        feature_names=nomes,
        samples=amostras,
        mode=MODO,
    )
    destino = tmp_path / "normalizado.csv"

    write_normalizado(matriz, destino)

    with destino.open(newline="", encoding="utf-8") as arquivo:
        linhas = list(csv.reader(arquivo))
    assert tuple(linha[0] for linha in linhas[1:]) == nomes
    assert {len(linha) for linha in linhas} == {len(amostras) + 1}


# ---------------------------------------------------------------------------
# C07 — o leitor do measurements: BOM, linhas vazias, UTF-8 cortado na amostra
# ---------------------------------------------------------------------------


def test_c07_o_bom_do_excel_no_measurements_e_aceito(tmp_path: Path) -> None:
    """O «CSV UTF-8» do Excel começa por um BOM, que ficava colado ao `NAME` da primeira célula."""
    medicoes, lista = render_pair(base_spec())
    caminho_m, caminho_s = _grava(tmp_path, b"\xef\xbb\xbf" + medicoes.encode("utf-8"), lista)

    assert load_matrix(caminho_m, caminho_s, MODO).n_samples == 10


def _transposta(medicoes: str) -> str:
    """O mesmo `measurements` com amostras nas linhas — a orientação `samples-in-rows`."""
    linhas = [linha.split(",") for linha in medicoes.split("\r\n") if linha]
    return "\r\n".join(",".join(coluna) for coluna in zip(*linhas, strict=True)) + "\r\n"


@pytest.mark.parametrize(
    ("orientacao", "codigo"),
    [("features-in-rows", "MISSING_NAME_ROW"), ("samples-in-rows", "RAGGED_ROWS")],
)
def test_c07_linha_vazia_no_topo_e_recusada_pelo_codigo_declarado(
    tmp_path: Path, orientacao: str, codigo: str
) -> None:
    """`rows[0][0]` sobre `[]` levantava `IndexError`; a recusa é por código declarado.

    Na transposta a linha vazia deixa a grade irregular antes de haver linha 1, e o código é o do
    transpositor — que a árvore anterior já dava, e por isso esse parâmetro passa nas duas.
    """
    medicoes, lista = render_pair(base_spec())
    if orientacao == "samples-in-rows":
        medicoes = _transposta(medicoes)
    caminho_m, caminho_s = _grava(tmp_path, "\r\n" + medicoes, lista)

    with pytest.raises(IngestionError) as recusa:
        load_matrix(caminho_m, caminho_s, MODO, orientation=orientacao)  # type: ignore[arg-type]

    assert [p.code for p in recusa.value.issues if p.is_error] == [codigo]


def test_c07_linha_vazia_no_topo_pela_rota_e_lista_de_problemas(cliente_http: TestClient) -> None:
    """Pela rota, o `IndexError` era `500` — «defeito da ferramenta» sobre um arquivo do usuário."""
    medicoes, lista = render_pair(base_spec())

    resposta = cliente_http.post(
        "/matrizes/validar", files=_arquivos("\r\n" + medicoes, lista), data={"mode": MODO}
    )

    assert resposta.status_code == 200, resposta.text[:300]
    corpo = resposta.json()
    assert corpo["aceita"] is False
    assert "MISSING_NAME_ROW" in {p["codigo"] for p in corpo["problemas"]}


@pytest.mark.parametrize("orientacao", ["features-in-rows", "samples-in-rows"])
def test_c07_linhas_vazias_no_fim_sao_ignoradas(tmp_path: Path, orientacao: str) -> None:
    """Uma linha em branco e uma só de vírgulas depois da última linha: formatação, não dado.

    Nas duas orientações, porque as linhas vazias saem ANTES de o transpositor ver a grade.
    """
    medicoes, lista = render_pair(base_spec())
    if orientacao == "samples-in-rows":
        medicoes = _transposta(medicoes)
    largura = medicoes.split("\r\n")[0].count(",")
    caminho_m, caminho_s = _grava(tmp_path, medicoes + "\r\n" + "," * largura + "\r\n", lista)

    matriz = load_matrix(caminho_m, caminho_s, MODO, orientation=orientacao)  # type: ignore[arg-type]

    assert (matriz.n_samples, matriz.n_features) == (10, 5)


def test_c07_linha_vazia_no_MEIO_continua_a_acusar(tmp_path: Path) -> None:  # noqa: N802
    """A direção oposta, e passa nas duas árvores: a linha vazia do meio é dado faltante.

    O conserto tira as linhas vazias do FIM; se tirasse todas, este arquivo seria aceito com uma
    feature a menos e nada acusaria.
    """
    medicoes, lista = render_pair(base_spec())
    linhas = medicoes.split("\r\n")
    linhas.insert(4, "")
    caminho_m, caminho_s = _grava(tmp_path, "\r\n".join(linhas), lista)

    with pytest.raises(IngestionError) as recusa:
        load_matrix(caminho_m, caminho_s, MODO)

    assert "RAGGED_ROWS" in {p.code for p in recusa.value.issues if p.is_error}


# O tamanho da amostra é DERIVADO do padrão de `sniff_dialect`, e não pinado: se o padrão mudar, a
# fronteira que estes dois testes plantam muda com ele (derivar em vez de pinar).
TAMANHO_DA_AMOSTRA: int = inspect.signature(sniff_dialect).parameters["sample_bytes"].default


def _prefixo_ascii(tamanho: int) -> bytes:
    cabecalho = "NAME,a,b\r\nLABEL,x,y\r\n"
    linha = "1.00_100.0000m/z,1,2\r\n"
    return (cabecalho + linha * (tamanho // len(linha) + 1)).encode("ascii")[:tamanho]


def test_c07_utf8_cujo_caractere_atravessa_o_fim_da_amostra_e_utf8(tmp_path: Path) -> None:
    """O `é` ocupa o último byte da amostra e o primeiro de fora: a FATIA é inválida, o arquivo não.

    Sem o conserto o arquivo inteiro caía para `latin-1`, e todo nome acentuado virava `Ã©`.
    """
    caminho = tmp_path / "measurements.csv"
    caminho.write_bytes(_prefixo_ascii(TAMANHO_DA_AMOSTRA - 1) + "é,3,4\r\n".encode())

    assert sniff_dialect(caminho).encoding == "utf-8"


def test_c07_latin1_com_o_ultimo_byte_da_amostra_acentuado_continua_latin1(
    tmp_path: Path,
) -> None:
    """A direção oposta, e foi ELA que achou o defeito da primeira versão do conserto.

    `0xE9` é o `é` do `latin-1` e também o primeiro byte de um caractere UTF-8 de três. A primeira
    versão aceitava qualquer cauda que fosse prefixo válido de UTF-8, e chamava este arquivo de
    `utf-8` — a leitura inteira levantaria depois. Passa nas duas árvores, e reprovou na primeira
    versão do conserto, antes do commit.
    """
    caminho = tmp_path / "measurements.csv"
    caminho.write_bytes(_prefixo_ascii(TAMANHO_DA_AMOSTRA - 1) + "é,3,4\r\n".encode("latin-1"))

    assert sniff_dialect(caminho).encoding == "latin-1"


# ---------------------------------------------------------------------------
# C09 — a PERMANOVA com todos os grupos de duas amostras dividia por zero
# ---------------------------------------------------------------------------


def test_c09_grupos_de_duas_amostras_recusam_o_par_com_o_motivo() -> None:
    """Duas amostras por grupo têm distâncias ao centroide iguais por geometria: o teste de
    dispersão não tem variância dentro, e o `F` seria `inf` sobre um artefato."""
    scores = np.random.default_rng(0).normal(size=(4, 3))

    with pytest.raises(PermanovaSemPopulacaoError, match="todos os grupos têm duas amostras"):
        permanova(scores, list("aabb"), permutations=99, seed=1)


def test_c09_grupos_constantes_recusam_o_par() -> None:
    """Cada grupo num ponto só: a pseudo-F é `inf` e a dispersão é 0/0.

    A árvore anterior levantava `ZeroDivisionError`. É a segunda rede da recusa, a do `isfinite`:
    três amostras por grupo, então a recusa pela contagem não a alcança.
    """
    scores = np.array([[0.0, 0.0]] * 3 + [[5.0, 5.0]] * 3)

    with pytest.raises(PermanovaSemPopulacaoError):
        permanova(scores, list("aaabbb"), permutations=99, seed=1)


def test_c09_tres_amostras_por_grupo_continuam_a_medir() -> None:
    """A direção oposta, e passa nas duas árvores: a recusa nova não alcança a população que
    sustenta o teste."""
    scores = np.random.default_rng(0).normal(size=(6, 3))

    resultado = permanova(scores, list("aaabbb"), permutations=99, seed=1)

    assert np.isfinite(resultado.statistic)
    assert 0 < resultado.pvalue <= 1


def test_c09_o_diagnostico_com_grupos_de_duas_amostras_responde_sem_a_permanova(
    cliente_http: TestClient,
) -> None:
    """Pela rota: oito amostras de estudo em quatro grupos são duas por grupo, e o diagnóstico
    respondia `500`. O campo fica `None` — «a população não sustentou o teste»."""
    resposta = cliente_http.post(
        "/matrizes/diagnosticar", files=_par(n_samples=13, n_qcs=5), data={"mode": MODO}
    )

    assert resposta.status_code == 200, (resposta.status_code, resposta.text[:300])
    permanovas = _valores_da_chave(resposta.json(), "permanova")
    assert permanovas, "âncora: o diagnóstico tem de trazer o campo `permanova`"
    assert all(valor is None for valor in permanovas), permanovas


# ---------------------------------------------------------------------------
# C10 — o multiverso aceitava no disparo a grade que não enumera
# ---------------------------------------------------------------------------


def test_c10_a_grade_minima_continua_aceita(cliente_http: TestClient) -> None:
    """A âncora dos três seguintes: a enumeração no disparo não recusa a grade válida."""
    resposta = cliente_http.post("/multiversos", files=_par(), data={"mode": MODO, **GRADE_MINIMA})

    assert resposta.status_code == 202, resposta.text[:300]


@pytest.mark.parametrize(
    "troca",
    [
        pytest.param(
            {"imputations": '["none"]', "transformations": '["log10"]'}, id="combinacao_recusada"
        ),
        pytest.param({"corrections": '[["none"]]'}, id="vetor_aninhado"),
        pytest.param({"batch_options": "[2.5]"}, id="lotes_fracionarios"),
        pytest.param({"n_jobs": "0"}, id="n_jobs_zero"),
        pytest.param({"corrector_jobs": "0"}, id="corrector_jobs_zero"),
    ],
)
def test_c10_a_grade_que_nao_corre_e_recusada_no_disparo(
    cliente_http: TestClient, troca: dict[str, str]
) -> None:
    """`422` no disparo, onde a docstring da rota diz que as recusas acontecem.

    Na árvore anterior: a combinação recusada respondia `202` e o trabalho ficava `pendente` para
    sempre; o vetor aninhado dava `500`; os lotes fracionários viravam dois lotes em silêncio; e
    `n_jobs=0` respondia `202` e falhava na thread.
    """
    resposta = cliente_http.post(
        "/multiversos", files=_par(), data={"mode": MODO, **GRADE_MINIMA, **troca}
    )

    assert resposta.status_code == 422, (resposta.status_code, resposta.text[:300])


def test_c10_o_tratador_da_falha_nomeia_a_posicao_sem_levantar() -> None:
    """A segunda rede: o tratador corre dentro do `except` do trabalho, e levantar ali deixava o
    trabalho `pendente` para sempre. A primeira asserção é a âncora de que a grade NÃO enumera."""
    grade = GridDefinition(
        corrections=("none",),
        batch_options=(1,),
        rsd_cutoffs=(0.2,),
        missing_thresholds=(0.5,),
        imputations=("none",),
        sample_norms=("none",),
        transformations=("log10",),
        scalings=("none",),
    )
    with pytest.raises(ValueError, match="imputation='none'"):
        list(grade.enumerate())

    assert "nao pode ser enumerada" in _celula_na_posicao(grade, 0)


# ---------------------------------------------------------------------------
# C14 — a CLI saía com 1 e traceback, fora da tabela de códigos
# ---------------------------------------------------------------------------


def _corrigir(medicoes: Path, lista: Path, saida: Path, *extra: str) -> int:
    return main(
        [
            "corrigir",
            "--measurements",
            str(medicoes),
            "--sample-list",
            str(lista),
            "--mode",
            MODO,
            "--output",
            str(saida),
            *extra,
        ]
    )


def test_c14_arquivo_inexistente_sai_com_4(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _, lista = _grava_par(tmp_path)

    codigo = _corrigir(tmp_path / "nao_existe.csv", lista, tmp_path / "saida.csv")

    assert codigo == EXIT_INPUT_REJECTED
    assert "não pôde ser lida" in capsys.readouterr().err


def test_c14_os_dois_csv_trocados_saem_com_4(tmp_path: Path) -> None:
    medicoes, lista = _grava_par(tmp_path)

    assert _corrigir(lista, medicoes, tmp_path / "saida.csv") == EXIT_INPUT_REJECTED


def test_c14_output_que_e_pasta_sai_com_2(tmp_path: Path) -> None:
    medicoes, lista = _grava_par(tmp_path)
    pasta = tmp_path / "saida"
    pasta.mkdir()

    assert _corrigir(medicoes, lista, pasta) == EXIT_USAGE


def test_c14_log_que_e_pasta_sai_com_2_e_a_saida_fica(tmp_path: Path) -> None:
    """A irmã de saída que a revisão não listou: o log é escrito DEPOIS da saída principal.

    O arquivo de saída existir é a metade que a mensagem afirma — «a saída foi escrita; o log não».
    """
    medicoes, lista = _grava_par(tmp_path)
    pasta = tmp_path / "log"
    pasta.mkdir()
    saida = tmp_path / "saida.csv"

    assert _corrigir(medicoes, lista, saida, "--log", str(pasta)) == EXIT_USAGE
    assert saida.exists()


def test_c14_processar_com_log_que_e_pasta_sai_com_2(tmp_path: Path) -> None:
    medicoes, lista = _grava_par(tmp_path)
    pasta = tmp_path / "log"
    pasta.mkdir()

    codigo = main(
        [
            "processar",
            "--measurements",
            str(medicoes),
            "--sample-list",
            str(lista),
            "--mode",
            MODO,
            "--output",
            str(tmp_path / "saida.csv"),
            "--log",
            str(pasta),
        ]
    )

    assert codigo == EXIT_USAGE


def test_c14_exportar_para_uma_pasta_sai_com_2(tmp_path: Path) -> None:
    medicoes, lista = _grava_par(tmp_path)
    pasta = tmp_path / "exportado"
    pasta.mkdir()

    codigo = main(
        [
            "exportar",
            "--measurements",
            str(medicoes),
            "--sample-list",
            str(lista),
            "--mode",
            MODO,
            "--output",
            str(pasta),
        ]
    )

    assert codigo == EXIT_USAGE


def _abre_como_no_windows(monkeypatch: pytest.MonkeyPatch, alvo: Path) -> list[Path]:
    """Faz `Path.open` para escrita em `alvo` levantar o que o WINDOWS levanta, e conta os disparos.

    A forma foi lida de uma corrida no Windows, e não suposta: abrir uma pasta para
    escrita dá `PermissionError: [Errno 13] Permission denied: '<a pasta>'`, de `pathlib`, onde o
    Linux dá `IsADirectoryError`. Sem este duplo, os testes de C14 acima só medem o caminho do
    Linux, e foi assim que as quatro saídas passaram aqui e reprovaram lá. A contagem é a âncora
    de que o duplo aterrou: sem ela, um teste de pasta passaria pelo caminho do Linux sem tocar
    no do Windows.
    """
    original = Path.open
    disparos: list[Path] = []

    def _open(self: Path, mode: str = "r", *args: Any, **kwargs: Any) -> Any:
        if self == alvo and any(letra in mode for letra in "wax"):
            disparos.append(self)
            raise PermissionError(13, "Permission denied", str(self))
        return original(self, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", _open)
    return disparos


@pytest.mark.parametrize(
    "caso",
    ["corrigir_output", "corrigir_log", "processar_output", "processar_log", "exportar_output"],
)
def test_c14_no_windows_a_pasta_levanta_permissionerror_e_sai_com_2(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    caso: str,
) -> None:
    """As quatro saídas que traduzem a pasta, com a exceção do Windows.

    A classe tem quatro tratadores — o `--output` de `corrigir`, de `processar` e de `exportar`, e o
    `--log` que `corrigir` e `processar` gravam pelo mesmo `_grava_log` —, e os cinco casos passam
    pelos quatro.
    """
    medicoes, lista = _grava_par(tmp_path)
    pasta = tmp_path / "alvo"
    pasta.mkdir()
    saida = tmp_path / "saida.csv"
    entrada = [
        "--measurements",
        str(medicoes),
        "--sample-list",
        str(lista),
        "--mode",
        MODO,
    ]
    disparos = _abre_como_no_windows(monkeypatch, pasta)

    if caso == "corrigir_output":
        codigo = _corrigir(medicoes, lista, pasta)
    elif caso == "corrigir_log":
        codigo = _corrigir(medicoes, lista, saida, "--log", str(pasta))
    elif caso == "processar_output":
        codigo = main(["processar", *entrada, "--output", str(pasta)])
    elif caso == "processar_log":
        codigo = main(["processar", *entrada, "--output", str(saida), "--log", str(pasta)])
    else:
        codigo = main(["exportar", *entrada, "--output", str(pasta)])

    assert disparos, "o duplo não disparou: o teste mediu o caminho do Linux, e não o do Windows"
    assert codigo == EXIT_USAGE
    assert "aponta para uma pasta" in capsys.readouterr().err


def test_c14_arquivo_de_saida_bloqueado_nao_e_pasta(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """O par do teste acima: o mesmo `PermissionError`, sobre um ARQUIVO, não vira «pasta».

    No Windows é o CSV de saída aberto no Excel. Dizer que ele aponta para uma pasta mentiria sobre
    a causa, então o erro sobe como subia antes do conserto.
    """
    medicoes, lista = _grava_par(tmp_path)
    saida = tmp_path / "saida.csv"
    disparos = _abre_como_no_windows(monkeypatch, saida)

    with pytest.raises(PermissionError):
        _corrigir(medicoes, lista, saida)
    assert disparos


def test_c14_log_bloqueado_nao_e_pasta(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """O mesmo par para o `--log`, que tem tratador próprio em `_grava_log`.

    Sem este teste, tirar a verificação de lá não reprovava nada (achado da revisão do conserto): um
    log aberto noutro programa sairia como «aponta para uma pasta».
    """
    medicoes, lista = _grava_par(tmp_path)
    saida = tmp_path / "saida.csv"
    log = tmp_path / "execucao.json"
    disparos = _abre_como_no_windows(monkeypatch, log)

    with pytest.raises(PermissionError):
        _corrigir(medicoes, lista, saida, "--log", str(log))
    assert disparos
    assert saida.exists()


@pytest.mark.parametrize(
    "conteudo",
    [
        pytest.param(None, id="ausente"),
        pytest.param("[1, 2]", id="nao_e_objeto"),
        pytest.param("{nao e json", id="json_quebrado"),
    ],
)
def test_c14_from_log_ilegivel_sai_com_2(tmp_path: Path, conteudo: str | None) -> None:
    """O JSON quebrado já saía com 2 na árvore anterior — `JSONDecodeError` é `ValueError` — e fica
    aqui como guarda da mesma fronteira; os outros dois saíam com traceback."""
    log = tmp_path / "log.json"
    if conteudo is not None:
        log.write_text(conteudo, encoding="utf-8")

    codigo = main(["processar", "--from-log", str(log), "--output", str(tmp_path / "s.csv")])

    assert codigo == EXIT_USAGE


@pytest.mark.parametrize(
    ("campo", "valor"),
    [
        pytest.param("seed", "quarenta", id="semente_texto"),
        pytest.param("seed", True, id="semente_bool"),
        pytest.param("missing-threshold", [0.5], id="limiar_lista"),
        pytest.param("measurements", 5, id="caminho_numero"),
    ],
)
def test_c14_from_log_com_campo_de_tipo_errado_sai_com_2(
    tmp_path: Path, campo: str, valor: object
) -> None:
    """O log é ENTRADA, e nada nele passou pelo mypy.

    Na árvore anterior, a semente de outro tipo REEXECUTAVA com código 0 — a irmã silenciosa do
    `TypeError` que a revisão listou —, e os outros dois saíam com traceback. A primeira asserção é
    a âncora: o log que o próprio `processar` escreveu reexecuta, e só o campo trocado o estraga.
    """
    medicoes, lista = _grava_par(tmp_path)
    log = tmp_path / "log.json"
    original = [
        "processar",
        "--measurements",
        str(medicoes),
        "--sample-list",
        str(lista),
        "--mode",
        MODO,
        "--output",
        str(tmp_path / "a.csv"),
        "--log",
        str(log),
    ]
    assert main(original) == EXIT_OK
    bruto = json.loads(log.read_text(encoding="utf-8"))
    assert main(["processar", "--from-log", str(log), "--output", str(tmp_path / "b.csv")]) == (
        EXIT_OK
    ), "âncora: o log intacto tem de reexecutar"

    bruto["spec"][campo] = valor
    log.write_text(json.dumps(bruto), encoding="utf-8")

    codigo = main(["processar", "--from-log", str(log), "--output", str(tmp_path / "c.csv")])

    assert codigo == EXIT_USAGE


# ---------------------------------------------------------------------------
# C15 — os inteiros sem faixa nas rotas
# ---------------------------------------------------------------------------


def test_c15_semente_negativa_e_recusada_no_disparo(cliente_http: TestClient) -> None:
    """`numpy` recusa semente negativa com `ValueError`, que chegava como `500`.

    Doze amostras de estudo em quatro grupos são três por grupo: a PERMANOVA corre e consome a
    semente — com menos, ela recusaria antes de sortear, e o `-1` passaria pela árvore anterior
    sem nunca chegar ao `numpy`. A âncora é a semente válida responder `200`.
    """
    arquivos = _par(n_features=6, n_samples=17, n_qcs=5)
    valida = cliente_http.post(
        "/matrizes/diagnosticar", files=arquivos, data={"mode": MODO, "seed": "42"}
    )
    assert valida.status_code == 200, valida.text[:300]
    assert any(v is not None for v in _valores_da_chave(valida.json(), "permanova")), (
        "âncora: a PERMANOVA tem de correr, senão a semente nunca chega ao `numpy`"
    )

    resposta = cliente_http.post(
        "/matrizes/diagnosticar", files=arquivos, data={"mode": MODO, "seed": "-1"}
    )

    assert resposta.status_code == 422, (resposta.status_code, resposta.text[:300])
    assert cliente.causa_de(422, resposta.json()) == "usuario"


def test_c15_zero_lotes_em_corrigir_e_recusado_no_disparo(cliente_http: TestClient) -> None:
    """`n_batches=0` virava um lote em silêncio, enquanto `/execucoes` respondia `422`."""
    resposta = cliente_http.post(
        "/matrizes/corrigir", files=_par(), data={"mode": MODO, "n_batches": "0"}
    )

    assert resposta.status_code == 422, (resposta.status_code, resposta.text[:300])


# ---------------------------------------------------------------------------
# C19 — o gerador sintético e a fonte com menos features do que o pedido
# ---------------------------------------------------------------------------


def _fonte(n_features: int) -> RefinoMatrix:
    amostras = tuple(
        SampleMetadata(
            name=f"X{i:02d}",
            injection_order=i + 1,
            sample_type="qc" if i % 5 == 0 else "sample",
            group="QC" if i % 5 == 0 else "G1",
        )
        for i in range(30)
    )
    return RefinoMatrix(
        values=np.random.default_rng(0).lognormal(5, 1, size=(30, n_features)),
        feature_names=tuple(f"{i}.0_{100 + i}.0m/z" for i in range(n_features)),
        samples=amostras,
        mode="neg_c18",
    )


def test_c19_o_gerador_recusa_mais_features_do_que_a_fonte_tem() -> None:
    """Levantava `IndexError` três funções abaixo, sem dizer porquê.

    A âncora é o pedido que cabe: a recusa nova não alcança a fonte que tem features de sobra.
    """
    fonte = _fonte(40)

    def spec(n_features: int) -> SyntheticSpec:
        return SyntheticSpec(
            imitated_mode="neg_c18",
            drift_to_noise_ratio=1.0,
            n_features=n_features,
            n_feature_blocks=3,
        )

    assert generate(spec(30), fonte).matrix.n_features == 30
    with pytest.raises(ValueError, match="a fonte tem 40"):
        generate(spec(60), fonte)


# ---------------------------------------------------------------------------
# C20 — o motor por requisição guardava conexões abertas
# ---------------------------------------------------------------------------


def test_c20_o_motor_por_requisicao_fecha_cada_conexao_que_abre(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Contado por eventos do pool, e não por `/proc`: a propriedade é a mesma em toda plataforma.

    Com o pool padrão, a conexão voltava ao pool e ficava aberta até o coletor de lixo — medido:
    um `connect` e nenhum `close` em três usos. O motor nasce por requisição e nunca é reusado,
    então cada conexão aberta tem de fechar ao ser devolvida. A âncora é ter havido conexão.
    """
    monkeypatch.setenv("REFINO_DB_URL", f"sqlite:///{tmp_path / 'banco.db'}")
    motor = _motor_ou_503()
    contagem: Counter[str] = Counter()
    event.listen(motor, "connect", lambda *_: contagem.update(["connect"]))
    event.listen(motor, "close", lambda *_: contagem.update(["close"]))
    try:
        for _ in range(3):
            with motor.connect() as conexao:
                conexao.execute(text("select 1"))

        assert contagem["connect"] >= 1, "âncora: o motor tem de ter aberto conexão"
        assert contagem["close"] == contagem["connect"], dict(contagem)
    finally:
        motor.dispose()
