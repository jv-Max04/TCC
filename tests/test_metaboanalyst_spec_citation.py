"""A citação da especificação do formato é o REFERENTE do critério.

**Por que este módulo existe, e não é cerimônia sobre um documento.** O critério
confere conformidade contra a especificação publicada. Antes essa citação
não existia, e um critério sem referente é satisfeito por qualquer arquivo,
o mesmo defeito que retirou o critério.

Feito o documento, o modo de falha muda de lugar em vez de desaparecer: ele passa
a ser a **degradação silenciosa da citação**. Um item que perde a frase citada mas
mantém o rótulo `verbatim` deixa o mecanismo do critério conferindo menos coisas
sem nada acusar — saída indistinguível de sucesso.

**A asserção que carrega o peso é `test_verbatim_exige_frase_citada`.** As outras
conferem forma; essa confere que o rótulo não mente.

**Onde este módulo NÃO alcança, e a lacuna vai nomeada:** ele não confere que a
frase citada seja de fato a frase da página. Isso exigiria buscar a página em teste,
o que a tornaria dependência de rede e faria o portão reprovar quando o site cair —
e a página **caduca** de todo modo, que é a razão de a data de acesso existir. O que
este módulo prova é coerência interna da citação; a fidelidade dela ao original é
proveniência declarada, não verificada. É proxy pela forma, com a mesma
força e o mesmo limite.
"""

import re

import pytest

from tests.conftest import REPO

CITATION = REPO / "docs" / "formato-metaboanalyst.md"

# Os seis elementos transcritos. A contagem é explícita e não derivada do
# arquivo: um item que desaparecesse do documento faria uma contagem derivada
# concordar consigo mesma, e o portão passaria vacuamente sobre cinco.
EXPECTED_IDS = ("F-1", "F-2", "F-3", "F-4", "F-5", "F-6")

# `[F-1]` é o único `derivado`, porque as duas passadas de extração discordaram
# sobre a frase do separador. Fixar isso aqui é o que impede que uma edição
# futura o promova a `verbatim` em silêncio — a promoção passaria a autorizar o
# mecanismo a reprovar arquivo delimitado por tabulação, que a especificação aceita.
DERIVED_IDS = frozenset({"F-1"})

_STATUS_VALUES = frozenset({"verbatim", "derivado"})

# Quantas passadas de extração cobriram cada elemento. É dimensão SEPARADA do
# status, e agregar as duas foi defeito real desta suíte: a primeira redação da
# citação marcou os seis como "concordante nas duas passadas", e o `[F-6]` teve
# **uma** — a segunda consulta não perguntou por aquele tópico. Afirmar
# concordância onde houve uma medição é afirmar mais do que se mediu.
_PASSES = {"F-1": "duas", "F-2": "duas", "F-3": "duas", "F-4": "duas", "F-5": "duas", "F-6": "uma"}


def _items() -> dict[str, str]:
    """Mapeia `F-N` para o corpo da sua seção.

    Corta na próxima marca de cabeçalho de qualquer nível para que um item novo
    acrescentado no fim não herde o corpo do anterior.
    """
    text = CITATION.read_text(encoding="utf-8")
    matches = list(re.finditer(r"^### `\[(F-\d+)\]`", text, re.MULTILINE))
    bodies: dict[str, str] = {}
    for index, match in enumerate(matches):
        start = match.end()
        following = text.find("\n## ", start)
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        if following != -1:
            end = min(end, following)
        bodies[match.group(1)] = text[start:end]
    return bodies


def _status(body: str) -> str | None:
    match = re.search(r"\*\*Status\.\*\*\s*`([^`]+)`", body)
    return match.group(1) if match else None


def _passes(body: str) -> str | None:
    match = re.search(r"\*\*Passadas\.\*\*\s*`([^`]+)`", body)
    return match.group(1) if match else None


def test_a_citacao_existe() -> None:
    assert CITATION.is_file(), (
        f"{CITATION} não existe. É o referente do `[CA-11.6]`, e sem ele o critério "
        "é satisfeito por qualquer arquivo (AR66)."
    )


def test_carrega_url_e_data_de_acesso() -> None:
    """Sem os dois a caducidade fica indetectável.

    A data não é metadado decorativo: ela é o que permite a alguém, no futuro,
    perguntar se a página ainda diz isto. Citação sem data não é remedível, porque
    não se sabe em relação a quando ela era verdadeira.
    """
    text = CITATION.read_text(encoding="utf-8")
    assert "metaboanalyst.ca/docs/Format.xhtml" in text, "a citação não nomeia a URL da fonte"
    assert re.search(r"\b2026-\d{2}-\d{2}\b", text), (
        "a citação não traz data de acesso em ISO. Sem ela, ninguém sabe em relação "
        "a que instante a transcrição era verdadeira."
    )


def test_os_seis_elementos_estao_presentes_e_sem_lacuna() -> None:
    found = tuple(_items())
    assert found == EXPECTED_IDS, (
        f"os elementos da citação são {found}, esperava {EXPECTED_IDS}. "
        "Item que desaparece reduz o que o `[CA-11.6]` confere, sem nada acusar."
    )


@pytest.mark.parametrize("item_id", EXPECTED_IDS)
def test_todo_elemento_declara_status_legivel_por_maquina(item_id: str) -> None:
    status = _status(_items()[item_id])
    assert status in _STATUS_VALUES, (
        f"`[{item_id}]` tem status {status!r}, e o mecanismo do `[CA-11.6]` precisa "
        f"distinguir {sorted(_STATUS_VALUES)} para não tratar derivado como citação."
    )


@pytest.mark.parametrize("item_id", EXPECTED_IDS)
def test_todo_elemento_declara_quantas_passadas_o_cobriram(item_id: str) -> None:
    """Concordância é dimensão separada do `verbatim`, e agregá-las foi defeito real.

    A primeira redação declarou os seis como concordantes em duas passadas; o
    `[F-6]` teve **uma**, porque a segunda consulta não perguntou por aquele tópico.
    Fixar a contagem aqui impede que a alegação de concordância cresça sem medição
    nova.
    """
    assert _passes(_items()[item_id]) == _PASSES[item_id], (
        f"`[{item_id}]` declara {_passes(_items()[item_id])!r} passadas, e o medido é "
        f"{_PASSES[item_id]!r}. Aumentar a contagem exige extração nova, não edição de prosa."
    )


def test_o_status_derivado_e_exatamente_o_medido() -> None:
    """`[F-1]` é `derivado` porque as passadas discordaram, e isso não pode derivar.

    Promover `[F-1]` a `verbatim` autorizaria o mecanismo a reprovar arquivo
    delimitado por tabulação — que a especificação **aceita**. O portão passaria a
    reprovar dado conforme.
    """
    derived = {item for item, body in _items().items() if _status(body) == "derivado"}
    assert derived == DERIVED_IDS, (
        f"os elementos `derivado` são {sorted(derived)}, esperava {sorted(DERIVED_IDS)}. "
        "Mudança aqui altera o que o `[CA-11.6]` está autorizado a reprovar, e precisa "
        "de decisão registrada — não de edição de teste."
    )


@pytest.mark.parametrize("item_id", sorted(set(EXPECTED_IDS) - DERIVED_IDS))
def test_verbatim_exige_frase_citada(item_id: str) -> None:
    """A asserção que carrega o peso do módulo.

    Rótulo `verbatim` sem citação de bloco é o rótulo mentindo: o item continua
    parecendo transcrição e não há transcrição nenhuma. O modo de falha: o
    documento fica menor e o portão continua verde.
    """
    body = _items()[item_id]
    quoted = [line for line in body.splitlines() if line.lstrip().startswith(">")]
    assert quoted, (
        f"`[{item_id}]` está marcado `verbatim` e não traz nenhuma linha de citação. "
        "Ou a frase voltou, ou o status vira `derivado` com a discordância registrada."
    )
    assert any(len(line.strip()) > 20 for line in quoted), (
        f"a citação de `[{item_id}]` é curta demais para ser a frase da especificação; "
        "conferir se a transcrição foi truncada."
    )


def test_declara_o_que_nao_estabelece() -> None:
    """O escopo negativo vai escrito, e aqui é verificável.

    O documento é a base de uma afirmação de conformidade, e a distância entre
    "conforme ao formato" e "aceito pelo MetaboAnalyst" é exactamente onde o risco
    está. Ela fica escrita no documento, não subentendida por quem o ler.
    """
    text = CITATION.read_text(encoding="utf-8")
    assert "NÃO estabelece" in text, "a citação não declara o próprio escopo negativo (AR48)"
    assert "[CA-11.7]" in text, (
        "a citação não distingue o formato de arquivo da invocação do MetaboAnalystR; "
        "sem isso, conformidade de formato se lê como integração funcionando."
    )
