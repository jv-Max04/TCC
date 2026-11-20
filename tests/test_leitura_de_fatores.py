"""O CSV de fatores nas TRÊS fronteiras, e a tradução de erro em cada uma.

## Por que este arquivo existe

`normalization_factors` entrou com dois leitores independentes e **zero** cobertura de
fronteira. A revisão de código de 10/09/2026 mediu a consequência: `grep` exaustivo sobre
`tests/` não achou um único teste que passasse `--normalization-factors`, importasse
`_le_fatores` ou postasse um upload de fatores. Os únicos acertos eram `normalization_factors=None`
como argumento nomeado.

**A ausência de teste não foi o defeito; foi o que deixou três defeitos passarem.** Os três — o
`csv.Error` não apanhado, o `OSError` não apanhado, e a detecção de cabeçalho quebrada por linha
em branco — estavam nos **dois** leitores ao mesmo tempo.

## O que este arquivo verifica, e por que a lista é esta

O corpo da análise vive em `refino.io.fatores` e é testado diretamente. Mas o corpo estar certo
não diz nada sobre as fronteiras: **a pergunta que ficou sem resposta durante todo o bloco foi se
o erro do leitor chega ao operador como recusa dele ou como falha da ferramenta.**

Cada fronteira tem o seu teste, e o oráculo de cada um é o desfecho que o contrato daquela
fronteira declara — `422` na rota, um código da tabela do `cli.py` na linha de comando.

## E `ReferenciaAusenteError` está aqui pelo mesmo motivo

O docstring dela prometia *"a rota devolve `409`"* desde que foi escrita, e nada o implementava.
Uma promessa de tradução sem verificador é uma afirmação sem referente — e a forma de a fechar
não é apagar a promessa, é construir o referente.
Estes testes são o referente.
"""

from __future__ import annotations

import csv
import threading
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from refino.api.app import build_app
from refino.cli import EXIT_UNAVAILABLE, EXIT_USAGE, main
from refino.io.fatores import (
    FatoresMalformadosError,
    le_fatores_de_bytes,
    le_fatores_de_texto,
)
from tests.conftest import measurements_path, sample_list_path

MODO = "pos_hilic"


@pytest.fixture(scope="module")
def cliente() -> TestClient:
    return TestClient(build_app())


def _reais() -> dict[str, tuple[str, bytes, str]]:
    return {
        "measurements": (
            f"measurements_{MODO}.csv",
            measurements_path(MODO).read_bytes(),
            "text/csv",
        ),
        "sample_list": (
            f"Sample_List_{MODO}.csv",
            sample_list_path(MODO).read_bytes(),
            "text/csv",
        ),
    }


def _obrigatorios(**extra: str) -> dict[str, str]:
    """Os campos que as DUAS rotas exigem, num lugar só.

    **Escrever o subconjunto em cada teste foi um defeito medido nesta mesma sessão.** A
    primeira versão de `test_csv_de_fatores_ilegivel_da_422_e_NAO_500` mandava cinco campos a
    `/execucoes`, que exige doze — e recebia `422` por **campo em falta**. A asserção lia
    `422` e dizia *"a rota recusou o CSV"*.

    O veredito estava certo e o motivo era outro — *este veredito acusa o alvo, ou
    acusa o instrumento?* — dentro do teste que eu tinha acabado de escrever para fechar um
    defeito da mesma família.
    """
    campos = {
        "mode": MODO,
        "estudo": "revisao",
        "correction": "none",
        "method": "none",
        "n_batches": "1",
        "rsd_variant": "outlier_removed",
        "imputation": "one_fifth_min",
        "sample_normalization": "none",
        "transformation": "none",
        "scaling": "none",
        "apresentacao": "sem_apresentacao",
    }
    campos.update(extra)
    return campos


def _campo_gigante() -> str:
    """Um CSV que faz o `csv` da biblioteca padrão levantar `csv.Error`.

    **O gatilho é o limite de tamanho de campo e não a sintaxe**, porque o `csv` é
    deliberadamente permissivo: ele não recusa vírgula a mais, aspas desemparelhadas nem linha
    ragged. `field_size_limit` é o caminho de `csv.Error` que um arquivo real alcança — um
    despejo com uma célula enorme —, e é o que a revisão usou para reproduzir os dois 500.
    """
    return "amostra,fator\nS0," + "9" * (csv.field_size_limit() + 10) + "\n"


# ---------------------------------------------------------------------------
# O corpo — `refino.io.fatores`
# ---------------------------------------------------------------------------


def test_o_cabecalho_e_saltado_e_os_pares_chegam() -> None:
    assert le_fatores_de_texto("amostra,fator\nS0,2.0\nS1,0.5\n", origem="t") == (
        ("S0", 2.0),
        ("S1", 0.5),
    )


def test_sem_cabecalho_a_primeira_linha_JA_E_dado() -> None:  # noqa: N802
    """O cabeçalho é **opcional**, e a heurística é o segundo campo ser número.

    Se a primeira linha fosse sempre tratada como cabeçalho, um arquivo sem cabeçalho perderia
    a primeira amostra em silêncio — e o sintoma seria *"uma amostra não tem fator"*, que aponta
    para o dado quando o defeito é do leitor.
    """
    assert le_fatores_de_texto("S0,2.0\nS1,0.5\n", origem="t") == (("S0", 2.0), ("S1", 0.5))


def test_LINHA_EM_BRANCO_no_topo_nao_estraga_o_cabecalho() -> None:  # noqa: N802
    """O par causal da correção de 10/09/2026, e ele reprova na versão anterior.

    As duas versões antigas perguntavam `numero == 1`, com `numero` a contar **todas** as
    linhas. Uma linha vazia no topo — que o Excel produz ao exportar uma seleção — punha o
    cabeçalho na linha 2, onde a regra já não se aplicava, e ele era recusado como
    *"o fator de 'amostra' é 'fator', que não é número"*.

    A asserção afirma o desfecho **e** a igualdade ao caso sem linha em branco: só isso separa
    *"não levantou"* de *"leu a mesma coisa"*.
    """
    com_branco = le_fatores_de_texto("\namostra,fator\nS0,2.0\n", origem="t")
    sem_branco = le_fatores_de_texto("amostra,fator\nS0,2.0\n", origem="t")

    assert com_branco == sem_branco == (("S0", 2.0),)


def test_csv_ilegivel_levanta_a_classe_e_NAO_csv_Error() -> None:  # noqa: N802
    """`csv.Error` não deriva de `ValueError`, e é por isso que escapava.

    A MRO é `Error` -> `Exception`. Todo `except ValueError` à volta destes leitores — e as duas
    fronteiras tinham um — deixava-o passar. `io/sniffer.py` já o apanhava desde o início, na
    mesma biblioteca.
    """
    with pytest.raises(FatoresMalformadosError, match="não é um CSV legível"):
        le_fatores_de_texto(_campo_gigante(), origem="t")


def test_bytes_que_nao_sao_utf8_levantam_a_classe() -> None:
    with pytest.raises(FatoresMalformadosError, match="não é UTF-8"):
        le_fatores_de_bytes(b"amostra,fator\nS\xff0,2.0\n", origem="t")


def test_o_BOM_do_excel_nao_entra_no_nome_da_amostra() -> None:  # noqa: N802
    """`utf-8-sig`, e o defeito que ele evita aponta para o lugar errado.

    Com `utf-8` puro o BOM vira parte do nome da primeira amostra, que depois não casa com
    nenhuma linha da matriz. A recusa seria *"1 de 68 amostras não têm fator"* — verdadeira,
    inútil, e a apontar para o dado do laboratório.
    """
    assert le_fatores_de_bytes("﻿S0,2.0\n".encode(), origem="t") == (("S0", 2.0),)


@pytest.mark.parametrize(
    ("texto", "trecho"),
    [
        ("S0\n", "esperava duas colunas"),
        ("amostra,fator\nS0,dois\n", "que não é número"),
        ("amostra,fator\n", "nenhum fator lido"),
        ("\n\n", "nenhum fator lido"),
    ],
)
def test_as_formas_malformadas_levantam_com_a_causa_NOMEADA(texto: str, trecho: str) -> None:  # noqa: N802
    """A mensagem nomeia a causa, e isso é contrato e não estilo.

    Quem recebe isto é o operador do laboratório, na tela ou no terminal. *"Erro ao ler o
    arquivo"* obriga-o a adivinhar; *"linha 2: o fator de 'S0' é 'dois', que não é número"*
    diz-lhe o que corrigir.
    """
    with pytest.raises(FatoresMalformadosError, match=trecho):
        le_fatores_de_texto(texto, origem="t")


# ---------------------------------------------------------------------------
# O ponto e vírgula do Excel em português
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("texto", "esperado"),
    [
        ("amostra;fator\nS0;1,5\nS1;2\n", (("S0", 1.5), ("S1", 2.0))),
        ("amostra,fator\nS0,1.5\n", (("S0", 1.5),)),
        ("S0;1.5\n", (("S0", 1.5),)),
        ("S0;0.050\n", (("S0", 0.05),)),
        ("\namostra;fator\n\nS0;1,25\n", (("S0", 1.25),)),
    ],
)
def test_o_ponto_e_virgula_le_a_virgula_decimal_e_a_virgula_fica_como_era(
    texto: str, esperado: tuple[tuple[str, float], ...]
) -> None:
    """O modelo editado no Excel em português volta com `;` e vírgula decimal.

    Antes, o primeiro caso era recusado com *«esperava duas colunas»*. Com `,` nada muda, e
    `0.050` com `;` é decimal: zero mil não se escreve assim.
    """
    assert le_fatores_de_texto(texto, origem="t") == esperado


def test_o_BOM_e_o_fim_de_linha_do_windows_passam_com_ponto_e_virgula() -> None:  # noqa: N802
    """O arquivo como o Excel o salva: BOM, `;`, vírgula decimal e `\\r\\n`."""
    bruto = "\ufeffamostra;fator\r\nS0;1,5\r\n".encode()
    assert le_fatores_de_bytes(bruto, origem="t") == (("S0", 1.5),)


def test_o_ponto_de_milhar_com_ponto_e_virgula_e_recusado_com_a_linha() -> None:
    """Lido como decimal, `1.050` daria 1,05: um fator mil vezes menor, positivo e finito.

    A guarda de valor de `PipelineSpec` não acusaria nada, que é o modo de falha mais
    insidioso. A recusa nomeia a linha e o valor, e diz como escrever.
    """
    texto = "amostra;fator\nS0;850\nS1;1.050\n"
    with pytest.raises(FatoresMalformadosError, match=r"t:3: o fator de 'S1' é '1\.050'") as erro:
        le_fatores_de_texto(texto, origem="t")
    # **As duas leituras, e não uma.** A mensagem dizia só «(1050)», e quem a seguisse num
    # arquivo de ponto decimal gravava um fator mil vezes maior; ver o teste de baixo.
    assert "1050 se o ponto separa milhares" in str(erro.value), str(erro.value)
    assert "1,050 se ele é a vírgula decimal" in str(erro.value), str(erro.value)


def test_d674_o_ponto_de_tres_casas_num_arquivo_de_ponto_decimal_da_as_duas_leituras() -> None:
    """O caso da revisão: `;` com ponto decimal, como `pandas.to_csv(sep=";")` o escreve.

    `1.5` passa como 1,5 e `1.125` é recusado — a recusa é a, e fica. O que muda é o
    remédio: a mensagem mandava escrever `1125`, e quem a seguisse gravava para `C` um fator mil
    vezes maior, positivo e finito, que a guarda de valor deixa passar. **Ela não escolhe entre as
    duas leituras, porque o leitor não sabe qual é**; dá as duas, e quem escreveu o arquivo sabe.
    """
    texto = "amostra;fator\nA;1.5\nC;1.125\n"
    with pytest.raises(FatoresMalformadosError, match=r"t:3: o fator de 'C' é '1\.125'") as erro:
        le_fatores_de_texto(texto, origem="t")
    mensagem = str(erro.value)
    assert "1125 se o ponto separa milhares" in mensagem, mensagem
    assert "1,125 se ele é a vírgula decimal" in mensagem, mensagem


@pytest.mark.parametrize(("valor", "milhar"), [("1.234.567", "1234567"), ("1.050,5", "1050,5")])
def test_d674_com_dois_pontos_ou_com_virgula_so_a_leitura_de_milhar_existe(
    valor: str, milhar: str
) -> None:
    """A outra metade: com dois pontos, ou com a vírgula decimal ao lado, o ponto só é milhar.

    Oferecer `1,234,567` ou `1,050,5` como leitura decimal mandaria escrever um número que o próprio
    leitor recusa. A recusa continua — ela é — e a mensagem dá a leitura que existe.
    """
    with pytest.raises(FatoresMalformadosError) as erro:
        le_fatores_de_texto(f"amostra;fator\nS0;{valor}\n", origem="t")
    mensagem = str(erro.value)
    assert f"({milhar})" in mensagem, mensagem
    assert "se ele é a vírgula decimal" not in mensagem, mensagem


@pytest.mark.parametrize("valor", ["+1.050", "-1.050"])
def test_d674_o_ponto_de_milhar_com_sinal_tambem_e_recusado(valor: str) -> None:
    """A irmã da guarda, achada pela pergunta da classe: o sinal explícito a contornava.

    A expressão do ponto de milhar não aceitava sinal. Medido em 07/11/2026 na árvore anterior a
    este conserto: `+1.050` passava como 1,05 — a mesma leitura mil vezes menor que `1.050` sem
    sinal já recusava, positiva e finita, e a guarda de valor deixava passar —, e `-1.050` chegava a
    ela como -1,05, recusado por ser negativo e com a leitura errada. A guarda irmã do arquivo de
    vírgula, `_DIGITOS`, já aceitava o sinal. **Par causal:** sem o ponto, o fator com sinal é lido.
    """
    with pytest.raises(FatoresMalformadosError) as erro:
        le_fatores_de_texto(f"amostra;fator\nS0;{valor}\n", origem="t")
    mensagem = str(erro.value)
    assert f"{valor!r}" in mensagem, mensagem
    assert valor.replace(".", "") in mensagem, mensagem
    assert le_fatores_de_texto("amostra;fator\nS0;+1050\n", origem="t") == (("S0", 1050.0),)


def test_a_virgula_decimal_SEM_aspas_num_csv_de_virgula_e_recusada() -> None:  # noqa: N802
    """O defeito medido em 06/11/2026, antes do conserto: `A,1,5` era lido como 1,0.

    O `csv` parte a linha em três colunas, e o leitor ficava com a segunda e descartava a
    terceira sem nada acusar. A recusa diz a quem salvou como corrigir.
    """
    with pytest.raises(FatoresMalformadosError, match=r"t:2: a linha de 'S0' tem '1' e '5'"):
        le_fatores_de_texto("amostra,fator\nS0,1,5\n", origem="t")


def test_a_terceira_coluna_de_TEXTO_continua_a_ser_ignorada() -> None:  # noqa: N802
    """A direção que a guarda acima não pode tomar: uma coluna de observação não é decimal partido.

    Sem este caso, uma guarda que recusasse toda terceira coluna passaria no teste de cima e
    quebraria o arquivo de quem anota a amostra ao lado do fator.
    """
    texto = "amostra,fator,obs\nS0,1.5,peso seco\n"
    assert le_fatores_de_texto(texto, origem="t") == (("S0", 1.5),)


def test_a_falta_de_coluna_diz_o_delimitador_que_foi_lido() -> None:
    """Quem salvou com o separador do outro idioma precisa de saber qual foi lido."""
    with pytest.raises(FatoresMalformadosError, match="separadas por vírgula"):
        le_fatores_de_texto("S0 1.5\n", origem="t")
    with pytest.raises(FatoresMalformadosError, match="separadas por ponto e vírgula"):
        le_fatores_de_texto("amostra;fator\nS0\n", origem="t")


# ---------------------------------------------------------------------------
# A fronteira da ROTA — o oráculo é o código HTTP
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("rota", ["/execucoes", "/matrizes/processar"])
def test_csv_de_fatores_ilegivel_da_422_e_NAO_500(  # noqa: N802
    cliente: TestClient, rota: str
) -> None:
    """O par que a revisão reproduziu, agora com verificador.

    **422 e 500 são afirmações diferentes sobre de quem é a culpa.** `ui/cliente.py` renderiza
    500 como *"Defeito da ferramenta, não do seu arquivo"* — e o arquivo era dele.

    As duas rotas de propósito: elas foram editadas à mão em paralelo, e a revisão nomeou
    exactamente essa duplicação como o risco.
    """
    arquivos = _reais()
    arquivos["normalization_factors"] = ("fatores.csv", _campo_gigante().encode(), "text/csv")

    resposta = cliente.post(
        rota,
        files=arquivos,
        data=_obrigatorios(sample_normalization="supplied_factors"),
    )

    assert resposta.status_code == 422, (
        f"{rota} devolveu {resposta.status_code} para um CSV de fatores ilegível. "
        f"Corpo: {resposta.text[:400]}"
    )


def test_referencia_INEXISTENTE_da_409_e_nao_500(cliente: TestClient) -> None:  # noqa: N802
    """A promessa que o docstring de `ReferenciaAusenteError` fazia desde que foi escrita.

    Ela dizia *"as camadas de cima a traduzem: a rota devolve `409`"* e **nada o
    implementava** — a classe não era importada fora de `normalization.py`. A revisão
    reproduziu 500.

    O nome pedido existe na matriz? Não, e é esse o ponto: esta é a única das duas ausências
    que `PipelineSpec` **não** pode apanhar, porque responder exige a matriz. A outra — método
    de referência sem nome nenhum — passou a ser recusada na construção da spec, e chega a 422
    antes de o corretor correr.
    """
    # **`/matrizes/processar` e nao `/execucoes`, e a razao vai declarada.** As duas correm
    # `process_matrix` com o mesmo `except`, mas `/execucoes` GRAVA a execucao e recusa com
    # `503` quando nao ha banco — que e a condicao normal desta suite. Testar o 409 ali daria
    # um teste que nunca alcanca o codigo que veio conferir.
    resposta = cliente.post(
        "/matrizes/processar",
        files=_reais(),
        data=_obrigatorios(
            sample_normalization="feature_reference",
            normalization_reference="NAO_EXISTE_EM_MODO_NENHUM",
        ),
    )

    assert resposta.status_code == 202, f"o trabalho nem arrancou: {resposta.text[:300]}"
    identificador = resposta.json()["identificador"]

    # **A sondagem e obrigatoria e nao cerimonia:** enquanto o trabalho corre, `/trabalhos/`
    # devolve `202` com o progresso, e ler o primeiro `202` como desfecho faria este teste
    # afirmar «nao deu 409» sobre um trabalho que ainda nao tinha chegado ao pipeline. E a
    # veredito errado — seria sobre o instante da leitura, nao sobre a rota.
    for _ in range(600):
        desfecho = cliente.get(f"/trabalhos/{identificador}")
        if desfecho.status_code != 202:
            break
        threading.Event().wait(0.2)

    assert desfecho.status_code == 409, (
        f"a referência inexistente devolveu {desfecho.status_code} e o docstring de "
        f"`ReferenciaAusenteError` promete 409. Corpo: {desfecho.text[:400]}"
    )


def test_metodo_de_referencia_SEM_nome_e_recusado_na_construcao(cliente: TestClient) -> None:  # noqa: N802
    """422 e **antes** do corretor, que no `neg_c18` mede 490 s em série.

    Esta combinação é conhecível sem a matriz, então recusá-la em `PipelineSpec` custa nada e
    poupa ao operador a espera inteira. O `422` distingue-a do `409` acima: ali a spec é
    válida e o encontro com a matriz é que falha.
    """
    resposta = cliente.post(
        "/execucoes",
        files=_reais(),
        data=_obrigatorios(sample_normalization="sample_reference"),
    )

    assert resposta.status_code == 422, (
        f"`sample_reference` sem `normalization_reference` devolveu {resposta.status_code}; "
        f"é combinação inválida por construção. Corpo: {resposta.text[:400]}"
    )


# ---------------------------------------------------------------------------
# A fronteira da LINHA DE COMANDO — o oráculo é a tabela de códigos de saída
# ---------------------------------------------------------------------------


def _argv(saida: Path, **extra: str) -> list[str]:
    argv = [
        "processar",
        "--measurements",
        str(measurements_path(MODO)),
        "--sample-list",
        str(sample_list_path(MODO)),
        "--mode",
        MODO,
        "--output",
        str(saida),
        "--imputation",
        "one_fifth_min",
    ]
    for chave, valor in extra.items():
        argv += [f"--{chave.replace('_', '-')}", valor]
    return argv


def test_o_caminho_dos_fatores_ILEGIVEL_sai_com_codigo_2(  # noqa: N802
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A tabela do topo de `cli.py` declara `{0, 2, 3, 4}`, e o traceback saía com **1**.

    Medido na revisão: apontar `--normalization-factors` a um diretório deixava
    `PermissionError` escapar até fora de `main`. Script que leia o código para distinguir
    *"entrada má"* de *"ferramenta quebrada"* recebia a resposta errada — e o operador recebia
    um traceback em vez da linha `erro:` que todo o resto deste CLI imprime.
    """
    pasta = tmp_path / "isto_e_uma_pasta"
    pasta.mkdir()

    codigo = main(
        _argv(
            tmp_path / "saida.csv",
            sample_normalization="supplied_factors",
            normalization_factors=str(pasta),
        )
    )

    assert codigo == EXIT_USAGE, f"esperava {EXIT_USAGE} e recebi {codigo}"
    assert "erro:" in capsys.readouterr().err


def test_o_csv_de_fatores_ilegivel_sai_com_codigo_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    caminho = tmp_path / "fatores.csv"
    caminho.write_text(_campo_gigante(), encoding="utf-8")

    codigo = main(
        _argv(
            tmp_path / "saida.csv",
            sample_normalization="supplied_factors",
            normalization_factors=str(caminho),
        )
    )

    assert codigo == EXIT_USAGE, f"esperava {EXIT_USAGE} e recebi {codigo}"
    assert "não é um CSV legível" in capsys.readouterr().err


def test_referencia_INEXISTENTE_sai_com_codigo_da_tabela(  # noqa: N802
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """O irmão do teste de 409, na outra fronteira.

    `EXIT_UNAVAILABLE` e não `EXIT_USAGE`, pela mesma distinção que a irmã `EmptyMatrixError`
    já fazia: o argumento estava bem formado, e o que falhou foi o encontro dele com **esta**
    matriz.
    """
    codigo = main(
        _argv(
            tmp_path / "saida.csv",
            sample_normalization="feature_reference",
            normalization_reference="NAO_EXISTE_EM_MODO_NENHUM",
        )
    )

    assert codigo == EXIT_UNAVAILABLE, f"esperava {EXIT_UNAVAILABLE} e recebi {codigo}"
    assert "erro:" in capsys.readouterr().err


def test_metodo_de_referencia_SEM_nome_sai_com_codigo_2(  # noqa: N802
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    codigo = main(_argv(tmp_path / "saida.csv", sample_normalization="sample_reference"))

    assert codigo == EXIT_USAGE, f"esperava {EXIT_USAGE} e recebi {codigo}"
    assert "erro:" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# A ida e volta pelo registro, COM fatores
# ---------------------------------------------------------------------------


def test_os_fatores_atravessam_o_registro_e_a_reexecucao_bate(tmp_path: Path) -> None:
    """O log realimentado com o eixo que muda MAQUINARIA, e não apenas um número.

    O portão de `test_cli_log.py` prova que o **campo** viaja. Este prova que a execução
    inteira se repete com ele — que é a propriedade que o critério afirma, e a única que
    apanharia um leitor que resolvesse o mapa de forma dependente de ordem.
    """
    fatores = tmp_path / "fatores.csv"
    nomes = _nomes_das_amostras()
    linhas = ["amostra,fator"] + [f"{nome},{1.0 + i / 100:.4f}" for i, nome in enumerate(nomes)]
    fatores.write_text("\n".join(linhas) + "\n", encoding="utf-8")

    primeira = tmp_path / "primeira.csv"
    log = tmp_path / "log.json"
    codigo = main(
        _argv(
            primeira,
            log=str(log),
            sample_normalization="supplied_factors",
            normalization_factors=str(fatores),
        )
    )
    assert codigo == 0, "a execução com fatores fornecidos não correu"

    segunda = tmp_path / "segunda.csv"
    assert main(["processar", "--from-log", str(log), "--output", str(segunda)]) == 0
    assert primeira.read_bytes() == segunda.read_bytes(), (
        "a reexecução a partir do registro deu saída diferente: o mapa de fatores não "
        "atravessa o log de forma a reproduzir a execução"
    )


def _nomes_das_amostras() -> list[str]:
    from refino.io.readers import load_matrix

    matriz = load_matrix(
        measurements_path(MODO),
        sample_list_path(MODO),
        mode=MODO,
    )
    return [amostra.name for amostra in matriz.samples]
