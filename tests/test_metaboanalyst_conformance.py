"""Conformidade ao formato do MetaboAnalyst.

Duas coisas diferentes, e a primeira é a que o critério de fato pede.

**1. PARIDADE entre o mecanismo e a citação.** O critério diz "conferidos contra a
especificação **citada** em `docs/`". Se o mecanismo tivesse a própria lista de
regras, ele conferiria o que quem o escreveu lembrava, e a citação seria decoração.
A paridade é por teste e não por leitura em runtime: ler `docs/` de `src/refino/io/`
faria a fronteira depender do layout do repositório, que é o motivo pelo qual os
resolvedores de caminho vivem em `tests/conftest.py` (regra 8 do README). É o padrão de
Flags do estrito contra o que o `mypy` instalado expõe — e.

**2. O par causal do critério**, com uma direção por elemento obrigatório, mais a
direção que prova que o portão não reprova sempre.

**Este negativo é "em processo", e a distinção é.** A mutação é argumento do
conferidor, não artefato entregue a subprocesso, então ele não pode ficar inerte
pelos dois caminhos que a taxonomia tabela: a entrada alimenta a asserção e não há
intermediário que possa não recebê-la. Fica **fora** da exigência de "visto passar
com a planta removida" — que é de outro critério, e lá é externo de verdade.
"""

import re
from pathlib import Path

import pytest

from refino.io.readers import load_matrix
from refino.io.validators import FORMAT_RULES, FormatRule, check_metaboanalyst_conformance
from refino.io.writers import write_metaboanalyst
from tests.conftest import REPO, measurements_path, sample_list_path

CITATION = REPO / "docs" / "formato-metaboanalyst.md"
MODES = ("pos_hilic", "pos_c18", "neg_hilic", "neg_c18")


def _cited() -> dict[str, tuple[str, bool]]:
    """Lê da citação, por elemento, o par `(status, verificado pelo mecanismo)`."""
    text = CITATION.read_text(encoding="utf-8")
    found: dict[str, tuple[str, bool]] = {}
    headings = list(re.finditer(r"^### `\[(F-\d+)\]`", text, re.MULTILINE))
    for index, match in enumerate(headings):
        end = headings[index + 1].start() if index + 1 < len(headings) else len(text)
        body = text[match.end() : end]
        status = re.search(r"\*\*Status\.\*\*\s*`([^`]+)`", body)
        mechanism = re.search(r"\*\*Verificado pelo mecanismo\.\*\*\s*`([^`]+)`", body)
        assert status and mechanism, f"`[{match.group(1)}]` sem Status ou sem flag de mecanismo"
        found[match.group(1)] = (status.group(1), mechanism.group(1) == "sim")
    return found


# ---------------------------------------------------------------------------
# 1. Paridade — o mecanismo confere o que a CITAÇÃO diz, não o que eu lembrava
# ---------------------------------------------------------------------------


def test_os_elementos_do_mecanismo_e_da_citacao_sao_os_mesmos() -> None:
    """Bijeção nas duas direções.

    Elemento na citação e ausente no mecanismo é requisito que ninguém confere.
    Elemento no mecanismo e ausente na citação é **pior**: é o portão reprovando por
    regra que a especificação não tem, e isso parece rigor.
    """
    assert {rule.element for rule in FORMAT_RULES} == set(_cited())


@pytest.mark.parametrize("rule", FORMAT_RULES, ids=lambda rule: rule.element)
def test_o_status_de_cada_elemento_bate_com_a_citacao(rule: FormatRule) -> None:
    """`verbatim` contra `derivado`, e a diferença é o que o portão pode reprovar."""
    status, _ = _cited()[rule.element]
    assert rule.status == status, (
        f"`[{rule.element}]` é {rule.status!r} no mecanismo e {status!r} na citação. "
        "Divergência aqui muda o que o `[CA-11.6]` está autorizado a reprovar."
    )


@pytest.mark.parametrize("rule", FORMAT_RULES, ids=lambda rule: rule.element)
def test_a_flag_de_mecanismo_bate_com_a_citacao(rule: FormatRule) -> None:
    """Item declarado não verificado na citação não pode ser verificado em silêncio.

    E o contrário importa mais: item declarado **verificado** e sem verificação é o
    portão que não protege o que alguém acredita que ele protege.
    """
    _, mechanism = _cited()[rule.element]
    assert rule.mechanism == mechanism


def test_o_f6_e_o_unico_declaradamente_nao_verificado() -> None:
    """Fixado para que "não verificado" não cresça sem decisão registrada."""
    unchecked = {rule.element for rule in FORMAT_RULES if not rule.mechanism}
    assert unchecked == {"F-6"}, (
        f"os não verificados são {sorted(unchecked)}, esperava {{'F-6'}}. Acrescentar "
        "item a essa lista reduz o que o `[CA-11.6]` cobre e precisa de decisão."
    )


# ---------------------------------------------------------------------------
# 2. O par causal, uma direção por elemento
# ---------------------------------------------------------------------------


@pytest.fixture
def exported(tmp_path: Path) -> Path:
    """Um arquivo real e conforme, escrito pelo caminho de produção."""
    matrix = load_matrix(measurements_path("pos_c18"), sample_list_path("pos_c18"), "pos_c18")
    destination = tmp_path / "pos_c18.csv"
    write_metaboanalyst(matrix, destination)
    return destination


def test_o_arquivo_integro_passa(exported: Path) -> None:
    """A direção sem a qual as seis abaixo não provam nada.

    Um conferidor que reprovasse tudo satisfaria os seis negativos e seria
    indistinguível de um correto — é o portão que "sempre reprova".
    """
    assert check_metaboanalyst_conformance(exported) == []


@pytest.mark.parametrize("mode", MODES)
def test_os_quatro_modos_reais_exportam_conforme(mode: str, tmp_path: Path) -> None:
    matrix = load_matrix(measurements_path(mode), sample_list_path(mode), mode)
    destination = tmp_path / f"{mode}.csv"
    write_metaboanalyst(matrix, destination)

    assert check_metaboanalyst_conformance(destination) == []


def _mutate(path: Path, old: str, new: str) -> Path:
    text = path.read_text(encoding="utf-8")
    assert old in text, f"a planta não encontrou {old!r} para substituir — mutação não aplicada"
    path.write_text(text.replace(old, new, 1), encoding="utf-8")
    return path


def test_to_11_5_separador_trocado_reprova(exported: Path) -> None:
    """`[F-1]`. Derivado, então o que se reprova é **este** `.csv` não estar em vírgula."""
    text = exported.read_text(encoding="utf-8")
    exported.write_text(text.replace(",", "\t"), encoding="utf-8")

    violations = check_metaboanalyst_conformance(exported)

    assert any("[F-1]" in v for v in violations), violations


def test_to_11_5_rotulo_fora_da_segunda_coluna_reprova(exported: Path) -> None:
    """`[F-5]`."""
    _mutate(exported, "Sample,Label,", "Sample,Grupo,")

    violations = check_metaboanalyst_conformance(exported)

    assert any("[F-5]" in v for v in violations), violations


def test_to_11_5_nome_nao_conforme_reprova(exported: Path) -> None:
    """`[F-4]`, caractere fora do alfabeto."""
    header = exported.read_text(encoding="utf-8").splitlines()[0]
    first_feature = header.split(",")[2]
    _mutate(exported, f",{first_feature},", f",{first_feature}m/z,")

    violations = check_metaboanalyst_conformance(exported)

    assert any("[F-4]" in v for v in violations), violations


def test_to_11_5_nome_repetido_reprova(exported: Path) -> None:
    """`[F-4]`, unicidade — que é metade do requisito e some se ninguém a testar."""
    lines = exported.read_text(encoding="utf-8").splitlines()
    fields = lines[0].split(",")
    fields[3] = fields[2]
    lines[0] = ",".join(fields)
    exported.write_text("\r\n".join(lines) + "\r\n", encoding="utf-8")

    violations = check_metaboanalyst_conformance(exported)

    assert any("[F-4]" in v and "repetido" in v for v in violations), violations


def test_to_11_5_valor_negativo_reprova(exported: Path) -> None:
    """`[F-2]`."""
    lines = exported.read_text(encoding="utf-8").splitlines()
    fields = lines[1].split(",")
    fields[2] = "-5"
    lines[1] = ",".join(fields)
    exported.write_text("\r\n".join(lines) + "\r\n", encoding="utf-8")

    violations = check_metaboanalyst_conformance(exported)

    assert any("[F-2]" in v for v in violations), violations


def test_to_11_5_numero_com_espaco_reprova(exported: Path) -> None:
    """`[F-3]`, e a especificação declara que este modo de falha é silencioso."""
    lines = exported.read_text(encoding="utf-8").splitlines()
    fields = lines[1].split(",")
    fields[2] = "1 600"
    lines[1] = ",".join(fields)
    exported.write_text("\r\n".join(lines) + "\r\n", encoding="utf-8")

    violations = check_metaboanalyst_conformance(exported)

    assert any("[F-3]" in v for v in violations), violations


def test_to_11_5_celula_nao_numerica_reprova(exported: Path) -> None:
    """`[F-2]`, e `NA` **não** reprova: é a codificação de ausente que o formato pede."""
    lines = exported.read_text(encoding="utf-8").splitlines()
    fields = lines[1].split(",")
    fields[2] = "abc"
    lines[1] = ",".join(fields)
    exported.write_text("\r\n".join(lines) + "\r\n", encoding="utf-8")

    violations = check_metaboanalyst_conformance(exported)

    assert any("[F-2]" in v for v in violations), violations


def test_na_e_vazio_nao_reprovam(exported: Path) -> None:
    """A direção que impede o conferidor de recusar a codificação de ausente.

    `[F-2]` pede "using empty or NA for missing values", então reprovar `NA` seria
    reprovar dado conforme.
    """
    lines = exported.read_text(encoding="utf-8").splitlines()
    fields = lines[1].split(",")
    fields[2] = "NA"
    fields[3] = ""
    lines[1] = ",".join(fields)
    exported.write_text("\r\n".join(lines) + "\r\n", encoding="utf-8")

    assert check_metaboanalyst_conformance(exported) == []


def test_a_mensagem_nomeia_qual_elemento_falhou(exported: Path) -> None:
    """Mensagem genérica reprovaria pelo motivo errado sem ninguém notar.

    Num par causal de seis direções ela tornaria as
    seis indistinguíveis: cada teste acima passaria com qualquer violação presente.
    """
    _mutate(exported, "Sample,Label,", "Sample,Grupo,")

    violations = check_metaboanalyst_conformance(exported)

    assert violations
    for violation in violations:
        assert re.match(r"^\[F-\d+\] ", violation), (
            f"a violação {violation!r} não começa nomeando o elemento"
        )


def test_arquivo_vazio_reprova(tmp_path: Path) -> None:
    """O ramo que dispara por ausência de conteúdo, e não por conteúdo errado."""
    empty = tmp_path / "vazio.csv"
    empty.write_text("", encoding="utf-8")

    assert check_metaboanalyst_conformance(empty)
