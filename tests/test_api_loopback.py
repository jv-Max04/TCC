"""A superfície servida é loopback.

Por que este portão existe, e não é sobre segurança em geral: os arquivos de
`data/reference/` são clínicos humanos identificáveis, e o nome de cada amostra liga um
pseudônimo estável a um diagnóstico. A API é a parte que serve esse dado.

## Escopo negativo

Não se abre socket. O que se verifica é o **padrão de host** que a aplicação escolhe, porque
abrir porta em teste o torna dependente de porta livre e de ordem de execução. Logo este
módulo não prova que a API recuse uma conexão externa em tempo de execução.
"""

from refino.api.app import LOOPBACK_HOST, build_app, default_host

# ---------------------------------------------------------------------------
# O padrão de host
# ---------------------------------------------------------------------------


def test_o_padrao_fora_do_conteiner_e_loopback() -> None:
    """O binding é a única fronteira, e ele é loopback."""
    assert default_host() == LOOPBACK_HOST == "127.0.0.1"


# ---------------------------------------------------------------------------
# `httpx2`: sem ele não há teste de contrato por rota
# ---------------------------------------------------------------------------


def test_o_testclient_importa() -> None:
    """Asserção direta, porque sem ele o teste de contrato por rota é inimplementável.

    Medido antes de escrever qualquer rota: `fastapi.testclient.TestClient` não importa
    sem cliente HTTP instalado, e "um teste de contrato por rota" não tinha cliente.
    """
    from fastapi.testclient import TestClient

    assert TestClient is not None


def test_a_aplicacao_sobe_e_o_health_responde() -> None:
    """Fumaça: a aplicação sobe e `/health`, a rota de prontidão, responde."""
    from fastapi.testclient import TestClient

    with TestClient(build_app()) as cliente:
        resposta = cliente.get("/health")

    assert resposta.status_code == 200
    assert resposta.json() == {"status": "ok"}
