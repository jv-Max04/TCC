"""`scripts/verifica_o_app.ps1` manda os campos que as rotas declaram e lê as chaves que elas têm.

**O roteiro nunca corre aqui** — é PowerShell contra o executável, no Windows —, e os dois defeitos
que uma revisão achou nele eram silenciosos no lado de lá:

- mandava `correcoes` ao `/multiversos`, que declara `corrections`. O FastAPI ignora campo de
  formulário que a rota não declara, e o passo disparava a grade padrão e só conferia o id;
- lia o PLS-DA de dentro de `qualidade`, onde ele não está. Sem `Set-StrictMode`, membro
  inexistente é `$null` e não erro, e o passo da validação do PLS-DA dava verde sem olhar score
  nenhum.

**O contrato é derivado, e não pinado**: os campos válidos saem da assinatura do endpoint em
`refino.api.rotas.router`, e as chaves válidas dos modelos de resposta. Um campo renomeado na rota
reprova aqui no dia em que mudar, sem ninguém atualizar lista nenhuma.

**Escopo negativo.** O leitor entende as formas que o roteiro usa hoje — `$comuns`, a função
`Artefato` e o `curl.exe` direto com `-F` — e as leituras do passo das execuções. Forma nova de
chamada que ele não reconheça fica de fora; a âncora de presença diz quantas chamadas ele achou, e
cai se o roteiro mudar de forma.
"""

import dataclasses
import inspect
import re
from pathlib import Path

import pytest

from refino.api.esquemas import ExecucaoResposta, ResumoDoPlsda
from refino.api.rotas import router
from refino.domain.spec import QualityReport

ROTEIRO = Path(__file__).resolve().parents[1] / "scripts" / "verifica_o_app.ps1"

CAMPO_SEPARADO = re.compile(r'"-F"\s+"([A-Za-z_][A-Za-z0-9_]*)=')
CAMPO_EM_LISTA = re.compile(r'"-F",\s*"([A-Za-z_][A-Za-z0-9_]*)=')


def _linhas_logicas(texto: str) -> list[str]:
    """Junta as linhas continuadas pelo acento grave, que é como o PowerShell as lê."""
    return re.sub(r"`\r?\n\s*", " ", texto).splitlines()


def chamadas(texto: str) -> list[tuple[str, set[str]]]:
    """`(rota, campos)` de cada POST de formulário que o roteiro faz."""
    linhas = _linhas_logicas(texto)
    comuns: set[str] = set()
    for linha in linhas:
        if linha.lstrip().startswith("$comuns = @("):
            comuns = set(CAMPO_EM_LISTA.findall(linha))
    achadas: list[tuple[str, set[str]]] = []
    for linha in linhas:
        artefato = re.search(r'Artefato "([a-z-]+)" @\((.*?)\)', linha)
        if artefato:
            achadas.append(
                (f"/matrizes/{artefato.group(1)}", comuns | set(CAMPO_EM_LISTA.findall(linha)))
            )
            continue
        direta = re.search(r'curl\.exe .*"\$base(/[a-z/-]+)"', linha)
        if direta and "@comuns" in linha:
            campos = set(CAMPO_SEPARADO.findall(linha)) | set(CAMPO_EM_LISTA.findall(linha))
            achadas.append((direta.group(1), comuns | campos))
    return achadas


def _parametros_do_post(rota: str) -> set[str] | None:
    for item in router.routes:
        if getattr(item, "path", None) == rota and "POST" in getattr(item, "methods", set()):
            return set(inspect.signature(item.endpoint).parameters)
    return None


def campos_ignorados(texto: str) -> list[tuple[str, str]]:
    """Os `(rota, campo)` que o roteiro manda e a rota não declara — ou rota que não existe."""
    ignorados: list[tuple[str, str]] = []
    for rota, campos in chamadas(texto):
        parametros = _parametros_do_post(rota)
        if parametros is None:
            ignorados.append((rota, "<rota inexistente>"))
            continue
        ignorados.extend((rota, campo) for campo in sorted(campos - parametros))
    return ignorados


def leituras_desconhecidas(texto: str) -> list[str]:
    """As leituras do passo das execuções que não existem no modelo de resposta.

    `$corpo` é a resposta do trabalho de `/execucoes` (`ExecucaoResposta`), `$q` é a `qualidade`
    dela — o `QualityReport` inteiro — e `$pls` é o `plsda`. `PSObject` é membro do PowerShell, e
    não do JSON.
    """
    inicio = texto.index('"$base/execucoes"')
    fim = texto.index("# --- 8.", inicio)
    modelos = {
        "corpo": set(ExecucaoResposta.model_fields),
        "q": {campo.name for campo in dataclasses.fields(QualityReport)},
        "pls": set(ResumoDoPlsda.model_fields),
    }
    desconhecidas: list[str] = []
    for variavel, chave in re.findall(
        r"\$(corpo|q|pls)\.([A-Za-z_][A-Za-z0-9_]*)", texto[inicio:fim]
    ):
        if chave != "PSObject" and chave not in modelos[variavel]:
            desconhecidas.append(f"${variavel}.{chave}")
    return desconhecidas


def test_o_leitor_acha_as_chamadas_do_roteiro() -> None:
    """Âncora de presença: sem chamadas achadas, as duas asserções de ausência passariam."""
    rotas = {rota for rota, _ in chamadas(ROTEIRO.read_text(encoding="utf-8"))}
    assert {"/matrizes/validar", "/matrizes/corrigir", "/execucoes", "/multiversos"} <= rotas, rotas


def test_todo_campo_que_o_roteiro_manda_a_rota_declara() -> None:
    ignorados = campos_ignorados(ROTEIRO.read_text(encoding="utf-8"))
    assert not ignorados, (
        f"o roteiro manda campos que a rota ignora em silêncio: {ignorados}. O FastAPI descarta "
        "campo de formulário não declarado, e o passo mede o padrão em vez do pedido"
    )


def test_toda_chave_que_o_passo_das_execucoes_le_existe() -> None:
    texto = ROTEIRO.read_text(encoding="utf-8")
    assert re.search(r"\$pls\.q2", texto), "âncora: o passo do PLS-DA lê o `q2`"
    desconhecidas = leituras_desconhecidas(texto)
    assert not desconhecidas, (
        f"o roteiro lê chaves que a resposta não tem: {desconhecidas}. Sem `Set-StrictMode` elas "
        "valem `$null`, e o passo que as testa passa sem medir"
    )


@pytest.mark.parametrize(
    ("trecho", "acusado"),
    [
        pytest.param(
            '$comuns = @("-s", "-F", "mode=x")\n'
            '$d = (& curl.exe @comuns "-F" "correcoes=[1]" "$base/multiversos")\n',
            ("/multiversos", "correcoes"),
            id="campo_que_a_rota_ignora",
        ),
        pytest.param(
            '$comuns = @("-s", "-F", "mode=x")\n$r = Artefato "nao-existe" @() $arq\n',
            ("/matrizes/nao-existe", "<rota inexistente>"),
            id="rota_que_nao_existe",
        ),
    ],
)
def test_o_contrato_acusa_a_planta(trecho: str, acusado: tuple[str, str]) -> None:
    """O par causal do contrato de campos: a planta é acusada; o roteiro real, não."""
    assert acusado in campos_ignorados(trecho)


def test_o_contrato_de_leitura_acusa_o_plsda_dentro_da_qualidade() -> None:
    """A planta de C12, na forma em que ele estava: `$q.plsda`."""
    planta = '"$base/execucoes"\n$q = $corpo.qualidade\n$t = $null -ne $q.plsda\n# --- 8. x\n'
    assert leituras_desconhecidas(planta) == ["$q.plsda"]
