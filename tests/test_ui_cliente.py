"""O cliente HTTP da janela: cada desfecho de uma chamada à API vira causa e mensagem distintas.

Testes de `refino.interface.cliente`, sem a janela: o mapa de desfechos, o vocabulário de causa, as
mensagens, e a tradução de falhas de transporte.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from refino.api.app import build_app
from refino.interface import cliente
from tests.conftest import measurements_path, sample_list_path


@pytest.fixture(scope="module")
def transporte() -> TestClient:
    return TestClient(build_app(), raise_server_exceptions=False)


def test_ca_12_9_o_mapeamento_cobre_todos_os_desfechos() -> None:
    """Bijeção: todo desfecho tem classificação, e toda classificação tem desfecho.

    A direção *"todo desfecho está no mapa"* pega o desfecho novo; a
    inversa pega a entrada órfã, que é o que sobrevive a uma rota deixar de existir.
    """
    classificados = {c.split(":")[0] for c in cliente.CAUSA_DE_FALHA}
    esperados = {str(c) for c in cliente.DESFECHOS} - {str(c) for c in cliente.SEM_FALHA}

    assert classificados == esperados, (
        f"desfecho sem causa: {sorted(esperados - classificados)}; "
        f"causa órfã: {sorted(classificados - esperados)}"
    )


def test_ca_12_9_o_vocabulario_de_causa_tem_tres_valores() -> None:
    """Fechado, e nenhuma entrada escapa dele."""
    from typing import get_args

    vocabulario = set(get_args(cliente.Causa))

    assert vocabulario == {"usuario", "dado", "ambiente"}
    assert set(cliente.CAUSA_DE_FALHA.values()) <= vocabulario
    assert set(cliente.CAUSA_DE_FALHA.values()) == vocabulario, (
        "há valor do vocabulário que nenhum desfecho usa: ou sobra valor, ou falta desfecho"
    )


def test_ca_12_9_as_mensagens_sao_duas_a_duas_distintas() -> None:
    """Mensagem repetida colapsa dois fatos numa tela — é o mínimo do critério.

    **Itera por CHAVE e não por `(código, corpo)`.** A versão anterior
    chamava `mensagem_de(int(chave.split(":")[0]), ...)`, o que faz as três chaves de `503`
    colapsarem numa só — não há corpo que as separe, porque duas delas são fabricadas sem resposta.
    Esta é a asserção que reprovaria sob a busca por código, então é ela que prova que a busca
    endereçável por chave era necessária.
    """
    chaves = sorted(cliente.CAUSA_DE_FALHA)
    mensagens = [cliente.mensagem_por_chave(c) for c in chaves]

    assert len(chaves) == len(cliente.CAUSA_DE_FALHA), "a lista de chaves não cobre o mapa"
    assert len(set(mensagens)) == len(mensagens), (
        f"mensagens repetidas entre desfechos: {mensagens}"
    )


def _corpo_para(chave: str) -> Any:
    """O corpo que faz `chave_de` devolver aquela chave — o discriminador do `422`."""
    if chave == "422:conteudo":
        return {"detail": "measurements.csv recusado pela validação: ..."}
    if chave == "422:forma":
        return {"detail": [{"type": "missing", "loc": ["body", "mode"]}]}
    return None


@pytest.mark.parametrize("chave", sorted(cliente.CAUSA_DE_FALHA))
def test_ca_12_9_cada_desfecho_declara_a_causa_certa(chave: str) -> None:
    """Um caso por desfecho, **DERIVADO do mapa e não redigitado**.

    A versão anterior era uma lista de seis pares escrita à mão, e nada conferia que ela cobrisse
    `CAUSA_DE_FALHA`. Com a chave nova de `503` aquele teste continuaria verde sem nunca exercitar
    o desfecho que a unidade existe para criar — segunda fonte da mesma lista dentro do arnês que
    a modifica.
    """
    codigo = int(chave.split(":")[0])

    assert cliente.causa_de(codigo, _corpo_para(chave)) == cliente.CAUSA_DE_FALHA[chave]


def test_ca_12_9_a_parametrizacao_cobre_o_mapa_inteiro() -> None:
    """A asserção de PRESENÇA que impede a lista de ficar curta em silêncio.

    `parametrize` sobre coleção derivada não pode ficar curta — mas pode ficar **vazia** se o mapa
    esvaziar, e aí todo caso acima desaparece reportando `passed`. Esta função tem alvo próprio.
    """
    assert len(cliente.CAUSA_DE_FALHA) >= 6, (
        "o mapa encolheu; os casos derivados acima cobrem menos"
    )
    assert sum(1 for c in cliente.CAUSA_DE_FALHA if c.startswith("503:")) == 3, (
        "as três chaves de 503 são banco, demora e rede (KTD-1)"
    )


def test_ca_12_9_as_tres_falhas_de_transporte_dizem_acoes_diferentes() -> None:
    """Excedeu o tempo, nunca completou, e banco ausente pedem ações opostas.

    Antes, as três caíam numa mensagem só — a do banco —, então o usuário cujo pedido excedeu os
    300 s do cliente era mandado subir um banco que já estava de pé.
    """
    banco = cliente.mensagem_por_chave("503:banco")
    demora = cliente.mensagem_por_chave("503:demora")
    rede = cliente.mensagem_por_chave("503:rede")

    assert len({banco, demora, rede}) == 3, "duas das três falhas de transporte dizem o mesmo"
    assert all(m.strip() for m in (banco, demora, rede)), "mensagem vazia"

    # A âncora antiga procurava o nome de um produto de banco na mensagem, e caiu com a mensagem
    # que a satisfazia. Ela codificava a FORMA daquela instância e não a propriedade, que é *esta
    # mensagem é sobre o banco e as outras duas não são*. E a forma era justamente o defeito: um
    # aplicativo de mesa que usa SQLite num arquivo não tem servidor de banco para subir.
    assert "banco" in banco, (
        "âncora: a mensagem do banco não fala de banco, então a exclusão abaixo é vácua"
    )
    assert "banco" not in demora and "banco" not in rede, (
        "falha de transporte manda subir o banco — é a atribuição errada que a U1 conserta"
    )
    assert "em curso" in demora, "a mensagem de demora não diz que a operação pode continuar"


# **A ferramenta de infraestrutura que o aplicativo de mesa NÃO usa, e o vocabulário é fechado.**
#
# O executável de mesa fala com SQLite num arquivo sob `%LOCALAPPDATA%` — `desktop/banco.py` —, e
# `docs/GUIA-do-aplicativo.html` declara Docker fora do produto. Uma frase de erro que nomeia estas
# peças manda o profissional operar algo que ele não tem, para consertar algo que não é isso.
INFRAESTRUTURA_QUE_A_MESA_NAO_TEM = ("docker", "compose", "postgres", "psql", "contêiner")


def test_nenhuma_mensagem_manda_o_profissional_operar_infraestrutura() -> None:
    """Nenhuma frase que o usuário lê prescreve Docker, compose ou Postgres.

    A frase do `503:banco` já mandou subir o banco com `docker compose`, e ela chega à janela Qt,
    que a exibe **sozinha** — medido, `principal.py` mostra `[causa: X] mensagem` e não lê
    `detail`.

    ## O oráculo é um vocabulário fechado, e ele foi CONTADO antes de ser escrito

    A propriedade é *"a mensagem não manda operar infraestrutura que o produto de mesa não usa"*, e
    o predicado é a lista acima. Medido em 30/08/2026 sobre `_MENSAGEM`: **1** acusado antes da
    correção — o `503:banco` — e **0** depois. A contagem é o que separa portão de lista de
    isenção: com muitos acusados, a tabela de exceções viraria o portão.

    **Onde ele é cego, e vai declarado:** ele não alcança o `detail` que o servidor produz, e nem
    deve — `_motor_ou_503` nomeia a configuração concreta, e é lá que o operador precisa do nome do
    produto. O que este portão protege é a frase **genérica**, que é a única coisa que a janela de
    mesa mostra.
    """
    acusadas = {
        chave: palavra
        for chave, texto in cliente._MENSAGEM.items()
        for palavra in INFRAESTRUTURA_QUE_A_MESA_NAO_TEM
        if palavra in texto.lower()
    }
    assert acusadas == {}, (
        f"mensagem de usuário manda operar infraestrutura que o aplicativo de mesa não tem: "
        f"{acusadas}"
    )


def test_o_portao_da_infraestrutura_acusa_a_frase_que_o_fez_nascer() -> None:
    """A direção que prova que o portão acima CONSEGUE reprovar.

    **A planta é a frase real que vigorou até 30/08/2026**, e não um texto inventado: um portão
    exercitado contra `"docker"` solto provaria que a busca por substring funciona, e não que ela
    apanha a redação que de facto existiu.

    **A planta cai sobre dado em que as duas coisas diferem:** a frase de hoje passa e a de
    ontem reprova, na mesma função, com a mesma lista.
    """
    de_ontem = (
        "O banco de dados não está de pé. Isto é do ambiente, não do seu arquivo: suba o Postgres "
        "com `docker compose up -d app db` e aplique o esquema."
    )
    plantado = {**cliente._MENSAGEM, "503:banco": de_ontem}

    def acusa(mapa: dict[str, str]) -> dict[str, str]:
        return {
            chave: palavra
            for chave, texto in mapa.items()
            for palavra in INFRAESTRUTURA_QUE_A_MESA_NAO_TEM
            if palavra in texto.lower()
        }

    assert acusa(plantado), "a frase de ontem não foi acusada: o portão não consegue reprovar"
    assert acusa(dict(cliente._MENSAGEM)) == {}, (
        "âncora: sem a planta o portão também acusa, então ele não separa as duas frases"
    )


def test_ca_12_9_sucesso_nao_recebe_causa() -> None:
    """A bicondicional: `causa is None` **se e somente se** não houve falha.

    Sem ela, `None` cobriria *"não falhou"* e *"falhou e não sei classificar"* — o colapso que o
    vocabulário existe para impedir.
    """
    for codigo in cliente.SEM_FALHA:
        assert cliente.causa_de(codigo) is None
    for chave in cliente.CAUSA_DE_FALHA:
        assert cliente.causa_de(int(chave.split(":")[0]), _corpo_para(chave)) is not None


def test_ca_12_9_desfecho_imprevisto_nao_passa_em_silencio() -> None:
    """O ramo `else`, que dispara por AUSÊNCIA de condição e por isso fica inerte.

    Um código que ninguém previu tem de produzir causa e mensagem, não `None` e vazio.
    """
    assert cliente.causa_de(418) == "ambiente"
    assert "não previsto" in cliente.mensagem_de(418)


def test_ca_12_9_as_duas_causas_do_422_sao_distinguidas_de_verdade(
    transporte: TestClient,
) -> None:
    """**Contra a API real, e não contra corpo escrito à mão.**

    O discriminador é a forma do `detail`, e afirmá-lo sobre um dicionário que o teste inventou
    seria verificar a minha suposição em vez do comportamento.
    """
    arquivos = {
        "measurements": ("m.csv", measurements_path("neg_c18").read_bytes(), "text/csv"),
        "sample_list": ("s.csv", sample_list_path("neg_c18").read_bytes(), "text/csv"),
    }
    sem_campo = cliente.envia(transporte, "POST", "/matrizes/validar", arquivos=arquivos)
    rotulo_ruim = cliente.envia(
        transporte,
        "POST",
        "/matrizes/validar",
        arquivos=arquivos,
        campos={"mode": "neg_c18", "orientation": "xxx"},
    )

    assert sem_campo.codigo == 422 and sem_campo.causa == "usuario"
    assert rotulo_ruim.codigo == 422 and rotulo_ruim.causa == "usuario"
    assert sem_campo.mensagem == rotulo_ruim.mensagem, "as duas são de FORMA e têm a mesma ação"


def test_ca_12_9_o_csv_malformado_vira_causa_de_dado_e_nao_de_ambiente(
    transporte: TestClient,
) -> None:
    """O CSV com uma linha curta é problema do ARQUIVO, e a tela tem de o dizer.

    **Este teste já afirmou o DEFEITO, e o defeito foi consertado.** Ele exigia `codigo == 500` e
    `causa == "ambiente"` com a mensagem *"não adianta conferir o dado"*. A rota passou a traduzir
    o `ValueError` do leitor em `422`; o teste ficou vermelho, como previsto, e é esta a
    remediação.

    **A cadeia inteira é afirmada, e não só o código**, porque foi no meio dela que o conserto
    ficou pela metade: a rota devolvia o `422` com um `detail` que nomeia o arquivo e a linha, e
    `_forma_do_422` classificava-o como `forma` — a tela dizia *"Faltou um campo"* sobre um CSV
    malformado, que é o mesmo defeito de origem com outra roupa.
    """
    ruim = {
        "measurements": ("m.csv", b"nao,e,csv,valido\n1,2\n", "text/csv"),
        "sample_list": ("s.csv", sample_list_path("neg_c18").read_bytes(), "text/csv"),
    }
    resposta = cliente.envia(
        transporte, "POST", "/matrizes/diagnosticar", arquivos=ruim, campos={"mode": "neg_c18"}
    )

    assert resposta.codigo == 422, (
        "a rota voltou a não traduzir o erro do leitor; um `500` faz a tela dizer «defeito da "
        "ferramenta, não do seu arquivo» sobre um arquivo que de facto está malformado"
    )
    assert resposta.causa == "dado", (
        f"a causa é {resposta.causa!r} e não `dado`: o `detail` da rota não casa nenhuma "
        "MARCA_DE_CONTEUDO, então a tela manda conferir o formulário em vez do arquivo"
    )
    assert "arquivo" in resposta.mensagem.lower(), (
        f"a mensagem não fala do arquivo: {resposta.mensagem!r}"
    )


def test_ca_12_9_falha_de_rede_vira_ambiente_e_nao_sobe_stack_trace() -> None:
    """Exceção que sobe à tela vira *stack trace* — a pior classificação possível.

    O duplo levanta na chamada. Se `envia` não traduzisse, o teste veria a exceção subir.

    **A ASSERÇÃO ERA FRACA, e é por isso que um defeito passou por ela.** Ela exigia que a
    mensagem contivesse *"não respondeu"* — propriedade que a mensagem do BANCO satisfazia, porque
    era ela que o ramo `except` reutilizava. O que importa não é o que a frase contém, é o que ela
    **não** manda fazer: subir um banco que não é a causa. Par causal com o desfecho de demora
    logo abaixo.
    """

    class Morto:
        def post(self, url: str, **kwargs: Any) -> Any:
            raise ConnectionError("recusada")

        def get(self, url: str, **kwargs: Any) -> Any:
            raise ConnectionError("recusada")

    resposta = cliente.envia(Morto(), "POST", "/matrizes/validar")

    assert resposta.codigo == 503
    assert resposta.causa == "ambiente"
    assert resposta.mensagem == cliente.mensagem_por_chave("503:rede")
    assert "Postgres" not in resposta.mensagem, "conexão recusada manda subir o banco"
    # O texto da exceção fica em `detail`, separado da frase de ação — `_relata` exibe-o à parte.
    assert "ConnectionError" in str(resposta.corpo["detail"])
    assert "ConnectionError" not in resposta.mensagem


def test_ca_12_9_tempo_esgotado_e_conexao_recusada_dizem_coisas_opostas() -> None:
    """**O par causal.** As duas caem no mesmo `except` e pedem ações contrárias.

    Excedeu o tempo: a operação pode estar viva, e repetir dispara o trabalho duas vezes. Conexão
    recusada: nada está a correr, e a ação é subir o serviço. Antes, as duas recebiam a
    mensagem do banco — e o caso da demora não é hipotético, `POST /execucoes` já excedeu os 300 s
    do `timeout` deste cliente.
    """

    class Demorado:
        def post(self, url: str, **kwargs: Any) -> Any:
            raise _ReadTimeoutError("passou de 300 s")

        def get(self, url: str, **kwargs: Any) -> Any:
            raise _ReadTimeoutError("passou de 300 s")

    class Recusado:
        def post(self, url: str, **kwargs: Any) -> Any:
            raise ConnectionError("recusada")

        def get(self, url: str, **kwargs: Any) -> Any:
            raise ConnectionError("recusada")

    demora = cliente.envia(Demorado(), "POST", "/execucoes")
    rede = cliente.envia(Recusado(), "POST", "/execucoes")

    assert demora.mensagem != rede.mensagem, "os dois desfechos são indistinguíveis"
    assert "em curso" in demora.mensagem
    assert "em curso" not in rede.mensagem
    assert demora.causa == rede.causa == "ambiente", "o vocabulário tem de ficar em três valores"


class _ReadTimeoutError(Exception):
    """Imita a família de tempo esgotado do `httpx` pelo NOME, que é o discriminador.

    O sufixo `Error` é convenção de lint deste repositório, e `Timeout` continua no nome — que é
    a substring que o discriminador de produção procura.

    **Não importa `httpx2` de propósito:** o discriminador de produção compara
    `type(erro).__name__`, e um duplo que herdasse do tipo real esconderia o facto de a comparação
    ser por nome. Se alguém trocar o discriminador por `isinstance`, este teste reprova.
    """
