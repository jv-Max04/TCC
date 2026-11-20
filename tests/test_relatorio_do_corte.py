"""O registro do corte da exportação.

## O que esta unidade fecha

`write_metaboanalyst` separa as três causas de remoção e o `escreve()` da rota
**descartava** o retorno: pela janela de Windows a exportação produzia o CSV cortado e mais nada.
O `stdout` de `refino exportar` era a única superfície que mostrava as contagens, e ninguém vê
`stdout` num executável `--windowed`.

## Por que o par de CSV é CONSTRUÍDO aqui e não vem de `data/reference/`

**Não é fabricação de dado no lugar de dado real (regra 1 do README), e a distinção é o que se
afirma.** Nada aqui compara contra número observado nos arquivos do laboratório: o que se afirma
é o **mecanismo** — que a causa certa cai no balde certo, que o RSD viaja alinhado com o nome, e
que a rota distingue os quatro desfechos. Para isso o dado precisa de ter as três causas presentes
**e conhecidas de antemão**, que é exactamente o que os quatro arquivos reais não dão de graça:
só o `pos_hilic` excede 5.000 features, e por isso o corte é exercitável em **1 de 4** modos.

O contraste que calibra: `tests/test_api_artefatos.py` compara byte a byte contra a CLI **sobre os
arquivos reais** e faz `skip` quando eles faltam. Aquilo é paridade; isto é mecanismo.

## A planta cai sobre dado em que as três causas DIFEREM

Sem isso, este arquivo seria vácuo. `max_features` é escolhido para que o
corte **tenha** de acontecer, e as três causas são plantadas em features distintas:

| Features | Como | Causa esperada |
|---|---|---|
| `F00`, `F01` | QCs **todos** zero | `qcs_todos_zero` |
| `F02`, `F03` | **um** QC não nulo, os outros zero | `outliers_apagaram_qcs` |
| as restantes | dispersão crescente entre QCs | `rsd_acima_do_corte` |

A segunda linha é a que exige cuidado: ela só produz a causa B porque a remoção de valores
atípicos **apaga** o único QC não nulo, e o vetor que sobra é de zeros. Sob `plain` ela continua
definida — e é dessa **diferença entre as duas variantes** que `_classify_features` deriva a
causa, sem inspecionar feature nenhuma.
"""

from __future__ import annotations

import csv
import io
from pathlib import Path

import pytest
from fastapi import status
from fastapi.testclient import TestClient

from refino.api.app import build_app
from refino.io.writers import (
    COLUNAS_DAS_REMOVIDAS,
    FeatureRemovida,
    MetaboAnalystExport,
    write_metaboanalyst,
    write_removidas,
)
from tests.reclamacao import espera

pytestmark = pytest.mark.integration

N_QC = 7
N_AMOSTRA = 5
N_FEATURE = 14
MODO = "pos_hilic"

# O valor do único QC não nulo das features de causa B. **Alto de propósito**: ele tem de ser
# atípico o bastante para a remoção de outliers o apagar, senão a feature cai na causa C e a
# planta separa a coisa errada.
QC_SOLITARIO = 5_000.0


def _nomes() -> tuple[list[str], list[str]]:
    """Os nomes de amostra e os grupos, com os QCs primeiro — como em três dos quatro arquivos.

    A ordem das colunas do `measurements` **não** é a de injeção: os arquivos do laboratório
    agrupam por classe com os QCs à frente, e a ordem temporal vive só no `Sample_List`. Reproduzir
    isso aqui é o que impede o teste de passar por um caminho que o dado real não tem.
    """
    amostras = [f"QC{i + 1}" for i in range(N_QC)] + [f"AM{i + 1}" for i in range(N_AMOSTRA)]
    grupos = ["QC"] * N_QC + ["G1"] * N_AMOSTRA
    return amostras, grupos


def _valores() -> list[list[float]]:
    """A matriz plantada, com as três causas em features conhecidas de antemão."""
    linhas: list[list[float]] = []
    for f in range(N_FEATURE):
        if f < 2:
            qcs = [0.0] * N_QC
        elif f < 4:
            qcs = [QC_SOLITARIO] + [0.0] * (N_QC - 1)
        else:
            # Dispersão crescente com o índice: `F04` é a mais estável e `F13` a menos. É o que
            # torna a ORDEM da causa C afirmável — da pior para a melhor.
            passo = (f - 3) * 3.0
            qcs = [100.0 + passo * (i - N_QC // 2) for i in range(N_QC)]
        linhas.append([*qcs, *[100.0 + f for _ in range(N_AMOSTRA)]])
    return linhas


def _par_de_arquivos() -> dict[str, tuple[str, bytes, str]]:
    """O `measurements` e o `Sample_List` no formato da especificação, prontos para o
    `multipart`."""
    amostras, grupos = _nomes()

    medicoes = io.StringIO()
    escritor = csv.writer(medicoes, lineterminator="\n")
    escritor.writerow(["NAME", *amostras])
    escritor.writerow(["LABEL", *grupos])
    for f, linha in enumerate(_valores()):
        escritor.writerow([f"{f + 1}.00_{100 + f}.0000m/z", *linha])

    # A ordem de injeção **intercala** QC e amostra, que é o que o laboratório faz — e é diferente
    # da ordem das colunas acima. Se as duas coincidissem, um defeito de join passaria.
    ordem = []
    fila_qc = [n for n in amostras if n.startswith("QC")]
    fila_am = [n for n in amostras if n.startswith("AM")]
    while fila_qc or fila_am:
        if fila_qc:
            ordem.append(fila_qc.pop(0))
        if fila_am:
            ordem.append(fila_am.pop(0))

    lista = io.StringIO()
    escritor_lista = csv.writer(lista, lineterminator="\n")
    escritor_lista.writerow(["ID", "inj_order_name"])
    for i, nome in enumerate(ordem, start=1):
        escritor_lista.writerow([i, nome])

    return {
        "measurements": (
            f"measurements_{MODO}.csv",
            medicoes.getvalue().encode("utf-8"),
            "text/csv",
        ),
        "sample_list": (
            f"Sample_List_{MODO}.csv",
            lista.getvalue().encode("utf-8"),
            "text/csv",
        ),
    }


@pytest.fixture(scope="module")
def cliente() -> TestClient:
    return TestClient(build_app())


def _exporta(cliente: TestClient, **extra: object) -> tuple[str, object]:
    """Dispara a exportação, espera, e devolve `(identificador, resposta da reclamação)`.

    Devolve o identificador porque é ele que a rota do relatório precisa — `reclamacao.reclama`
    o consome e não o expõe, e reescrevê-lo aqui daria uma segunda cópia do laço de espera.
    """
    disparo = cliente.post(
        "/matrizes/exportar",
        files=_par_de_arquivos(),
        data={"mode": MODO, **extra},
    )
    assert disparo.status_code == status.HTTP_202_ACCEPTED, disparo.text
    identificador = disparo.json()["identificador"]
    return identificador, espera(cliente, identificador)


# ---------------------------------------------------------------------------
# O escritor do arquivo irmão
# ---------------------------------------------------------------------------


def test_o_relatorio_separa_as_tres_causas_e_o_rsd_acompanha_o_nome(tmp_path: Path) -> None:
    """A planta cai nas três causas, e cada uma no balde certo.

    **A âncora vem primeiro:** sem corte não há o que separar, e um teste que só contasse os
    baldes passaria sobre uma matriz que não excedeu o teto.
    """
    from refino.io.readers import load_matrix

    par = _par_de_arquivos()
    medicoes = tmp_path / "m.csv"
    lista = tmp_path / "s.csv"
    medicoes.write_bytes(par["measurements"][1])
    lista.write_bytes(par["sample_list"][1])
    matriz = load_matrix(medicoes, lista, MODO)

    relatorio = write_metaboanalyst(matriz, tmp_path / "x.csv", max_features=6)

    assert len(relatorio.kept) == 6, "a âncora do corte falhou: não sobraram 6 mantidas"
    assert len(relatorio.removed) == N_FEATURE - 6

    assert len(relatorio.removed_all_zero_qcs) == 2, relatorio.removed_all_zero_qcs
    assert len(relatorio.removed_outliers_erased) == 2, relatorio.removed_outliers_erased
    assert len(relatorio.removed_high_rsd) == N_FEATURE - 6 - 4

    detalhe = relatorio.detalhe_das_removidas
    assert len(detalhe) == len(relatorio.removed)
    assert [item.nome for item in detalhe] == list(relatorio.removed), (
        "`detalhe_das_removidas` saiu em ordem diferente de `removed`, e as duas descrevem o "
        "mesmo corte — a ordem é critério e não formatação"
    )

    # **O RSD existe onde e SÓ onde há medida.** É a metade que o balde vazio não prova.
    por_causa = {item.nome: item for item in detalhe}
    for nome in relatorio.removed_undefined:
        assert por_causa[nome].rsd is None, f"{nome} saiu por indefinição e trouxe RSD"
    for nome in relatorio.removed_high_rsd:
        assert por_causa[nome].rsd is not None, f"{nome} saiu por RSD alto e não trouxe o número"


def test_a_causa_c_sai_da_pior_para_a_melhor(tmp_path: Path) -> None:
    """A ordem do balde C é decrescente em RSD, e ela é o que o leitor usa para julgar o corte.

    Num relatório, a primeira linha da lista de removidas é a que se lê primeiro, e ela tem de ser
    a mais instável — não a que passou raspando. `write_metaboanalyst` já o declara; o que faltava
    era alguém afirmar sobre o **número**, e não sobre a lista de nomes.
    """
    from refino.io.readers import load_matrix

    par = _par_de_arquivos()
    (tmp_path / "m.csv").write_bytes(par["measurements"][1])
    (tmp_path / "s.csv").write_bytes(par["sample_list"][1])
    matriz = load_matrix(tmp_path / "m.csv", tmp_path / "s.csv", MODO)

    relatorio = write_metaboanalyst(matriz, tmp_path / "x.csv", max_features=6)
    rsds = [
        item.rsd
        for item in relatorio.detalhe_das_removidas
        if item.causa == "rsd_acima_do_corte" and item.rsd is not None
    ]

    assert len(rsds) >= 3, f"amostra pequena demais para haver ordem: {rsds}"
    assert rsds == sorted(rsds, reverse=True), f"a causa C não saiu decrescente: {rsds}"


def test_o_arquivo_irmao_tem_uma_linha_por_removida_e_rsd_vazio_no_indefinido(
    tmp_path: Path,
) -> None:
    """`write_removidas` escreve o cabeçalho, uma linha por feature, e vazio onde não há medida."""
    removidas = (
        FeatureRemovida("1.00_100.0000m/z", "qcs_todos_zero", None),
        FeatureRemovida("2.00_200.0000m/z", "outliers_apagaram_qcs", None),
        FeatureRemovida("3.00_300.0000m/z", "rsd_acima_do_corte", 1.8342),
    )
    destino = tmp_path / "removidas.csv"

    write_removidas(removidas, destino)

    linhas = list(csv.reader(destino.read_text(encoding="utf-8").splitlines()))
    assert linhas[0] == list(COLUNAS_DAS_REMOVIDAS)
    assert len(linhas) == 1 + len(removidas)
    assert linhas[1] == ["1.00_100.0000m/z", "qcs_todos_zero", ""]
    assert linhas[2] == ["2.00_200.0000m/z", "outliers_apagaram_qcs", ""]
    assert linhas[3][:2] == ["3.00_300.0000m/z", "rsd_acima_do_corte"]
    assert float(linhas[3][2]) == pytest.approx(1.8342)


def test_o_arquivo_irmao_sai_com_crlf_como_o_artefato(tmp_path: Path) -> None:
    """Mesma terminação de linha que `write_metaboanalyst` e `write_toserrf`.

    **Não é estética: é o Excel.** O arquivo irmão é aberto em planilha no Windows pela mesma
    pessoa que abre o CSV cortado, e dois arquivos irmãos com terminações diferentes é a classe de
    diferença que aparece como *"um abre e o outro não"* sem nada acusar.
    """
    destino = tmp_path / "removidas.csv"

    write_removidas((FeatureRemovida("F", "qcs_todos_zero", None),), destino)

    bruto = destino.read_bytes()
    assert bruto.count(b"\r\n") == 2, f"esperava cabeçalho e uma linha em CRLF: {bruto!r}"
    assert b"\n\n" not in bruto


def test_rsd_desalinhado_do_nome_levanta_na_construcao() -> None:
    """A guarda do array paralelo — o negativo que prova que ela consegue acusar.

    Sem ela, `detalhe_das_removidas` casaria cada nome com o RSD de **outra** feature, e o número
    sairia certo no rótulo errado — na única coluna que justifica o corte.
    """
    with pytest.raises(ValueError, match="posicionalmente alinhados"):
        MetaboAnalystExport(
            kept=("A",),
            zeros_written_as_empty=0,
            renamed=0,
            removed_high_rsd=("B", "C"),
            rsd_of_removed_high_rsd=(1.0,),
        )


def test_com_o_alinhamento_certo_a_construcao_nao_acusa() -> None:
    """O par causal da guarda acima: com os comprimentos iguais, ela deixa de acusar.

    É a exigência da taxonomia de negativo plantado — ter sido visto passar com a planta
    **removida** não é o mesmo que ver o portão reprovar.
    """
    relatorio = MetaboAnalystExport(
        kept=("A",),
        zeros_written_as_empty=0,
        renamed=0,
        removed_high_rsd=("B", "C"),
        rsd_of_removed_high_rsd=(1.0, 0.5),
    )

    assert [item.rsd for item in relatorio.detalhe_das_removidas] == [1.0, 0.5]


# ---------------------------------------------------------------------------
# A rota que serve o relatório
# ---------------------------------------------------------------------------


def test_d347_o_relatorio_do_corte_e_servido_com_causa_e_rsd(cliente: TestClient) -> None:
    """`GET /trabalhos/{id}/relatorio` devolve o que o corte removeu, por causa.

    **É o teste que a lacuna inteira existe para ter.** Sem ele, uma rota que servisse o relatório
    **vazio** passaria em tudo o mais: o artefato sai com `200`, é conforme, e a janela mostraria
    *"0 removidas"* sobre um arquivo em que 8 features saíram.
    """
    identificador, artefato = _exporta(cliente, max_features=6)
    assert artefato.status_code == status.HTTP_200_OK, artefato.text

    resposta = cliente.get(f"/trabalhos/{identificador}/relatorio")

    assert resposta.status_code == status.HTTP_200_OK, resposta.text
    corpo = resposta.json()
    assert corpo["mantidas"] == 6
    assert len(corpo["removidas"]) == N_FEATURE - 6, corpo["removidas"]

    causas = [item["causa"] for item in corpo["removidas"]]
    assert causas.count("qcs_todos_zero") == 2
    assert causas.count("outliers_apagaram_qcs") == 2
    assert causas.count("rsd_acima_do_corte") == N_FEATURE - 6 - 4

    # A soma fecha com o total ingerido: partição exaustiva, não amostra.
    assert corpo["mantidas"] + len(corpo["removidas"]) == N_FEATURE


def test_o_relatorio_e_reclamavel_duas_vezes(cliente: TestClient) -> None:
    """Reclamar não consome, como não consome no artefato.

    Quem salvou o CSV duas vezes tem de poder escrever o arquivo irmão as duas — e é o mesmo
    argumento que faz `reclama_trabalho` não apagar o resultado na primeira leitura.
    """
    identificador, _ = _exporta(cliente, max_features=6)

    primeira = cliente.get(f"/trabalhos/{identificador}/relatorio")
    segunda = cliente.get(f"/trabalhos/{identificador}/relatorio")

    assert (primeira.status_code, segunda.status_code) == (200, 200)
    assert primeira.json() == segunda.json()


def test_sem_corte_o_relatorio_existe_e_a_lista_e_vazia(cliente: TestClient) -> None:
    """Matriz que não excede o teto: `200` com `removidas` vazia — e **não** `409`.

    A distinção mora na célula que decide o que a janela escreve: *"a exportação cortou e
    não removeu nada"* e *"esta operação não corta"* pedem frases opostas. O critério já exige
    a lista vazia do lado do escritor; aqui exige-se que ela **chegue** assim.
    """
    identificador, artefato = _exporta(cliente, max_features=5000)
    assert artefato.status_code == status.HTTP_200_OK

    resposta = cliente.get(f"/trabalhos/{identificador}/relatorio")

    assert resposta.status_code == status.HTTP_200_OK, resposta.text
    assert resposta.json()["removidas"] == []
    assert resposta.json()["mantidas"] == N_FEATURE


def test_corrigir_nao_tem_relatorio_e_o_409_diz_por_que(cliente: TestClient) -> None:
    """Operação que não corta responde `409`, e a mensagem nomeia a causa.

    **É o desfecho que a janela usa para NÃO mostrar o painel do corte.** Um `200` com lista vazia
    aqui faria `corrigir` — que devolve a matriz inteira — parecer uma exportação que não removeu
    nada, e o profissional leria *"0 removidas"* como garantia sobre um arquivo que nunca passou
    por corte nenhum.
    """
    disparo = cliente.post("/matrizes/corrigir", files=_par_de_arquivos(), data={"mode": MODO})
    assert disparo.status_code == status.HTTP_202_ACCEPTED, disparo.text
    identificador = disparo.json()["identificador"]
    assert espera(cliente, identificador).status_code == status.HTTP_200_OK

    resposta = cliente.get(f"/trabalhos/{identificador}/relatorio")

    assert resposta.status_code == status.HTTP_409_CONFLICT, resposta.text
    assert "não tem relatório de corte" in resposta.json()["detail"]


def test_identificador_desconhecido_devolve_404_e_nao_um_relatorio_vazio(
    cliente: TestClient,
) -> None:
    """`404` e não `200` com lista vazia — as duas pedem ações opostas.

    *"Não conheço este identificador"* manda corrigir o identificador; *"conheço e não há corte"*
    manda seguir. Colapsá-los faria a janela escrever um arquivo irmão vazio ao lado de um CSV
    que ela nem sabe se existe.
    """
    resposta = cliente.get("/trabalhos/nao-existe-de-todo/relatorio")

    assert resposta.status_code == status.HTTP_404_NOT_FOUND
    assert "nunca existiu" in resposta.json()["detail"]
