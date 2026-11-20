"""Verifica a resolução e a integridade dos dados de referência.

Estes testes NÃO verificam conteúdo numérico: isso é das fases 1, 2 e 3. Aqui
só se prova que os arquivos resolvem por caminho estável, que a integridade é
verificável, e que ausência produz `skip` conforme a especificação.
"""

import csv
import subprocess
from pathlib import Path

import pytest

from tests.conftest import (
    EXPECTED_FEATURES,
    EXPECTED_QC_ORDERS,
    EXPECTED_SAMPLES,
    MODES,
    NORMALIZATION_METHODS,
    measurements_path,
    normalized_path,
    qc_rsd_path,
    read_manifest,
    sample_list_path,
    sha256_of,
    toserrf_path,
)

REPO = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Resolução de caminhos
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", MODES)
def test_measurements_resolves(mode: str) -> None:
    """A exportação primária de cada modo resolve."""
    assert measurements_path(mode).is_file()


@pytest.mark.parametrize("mode", MODES)
def test_sample_list_resolves(mode: str) -> None:
    """A lista de injeção de cada modo resolve."""
    assert sample_list_path(mode).is_file()


@pytest.mark.parametrize("mode", MODES)
def test_toserrf_resolves(mode: str) -> None:
    """O intermediário do SERRF de cada modo resolve."""
    assert toserrf_path(mode).is_file()


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("method", NORMALIZATION_METHODS)
def test_normalized_golden_resolves(mode: str, method: str) -> None:
    """Os golden files normalizados resolvem, com o nome REAL em disco.

    O nome em disco tem espaços — ``normalized by - SERRF.csv`` dentro de
    ``{modo}_toserrf - normalization result/normalized datasets/`` — e não a
    forma achatada com underscores que a especificação declara. Este teste é o que
    trava a divergência num único lugar.
    """
    path = normalized_path(mode, method)
    assert path.is_file()
    assert " " in path.name or " " in path.parent.name


@pytest.mark.parametrize("mode", MODES)
def test_qc_rsd_golden_resolves(mode: str) -> None:
    """O golden de RSD nos QCs resolve — base dos critérios de RSD."""
    assert qc_rsd_path(mode).is_file()


def test_unknown_normalization_method_is_rejected() -> None:
    """Método inexistente é erro de programação, não skip."""
    with pytest.raises(ValueError, match="método desconhecido"):
        normalized_path("neg_c18", "nao_existe")


def test_missing_reference_file_skips_and_names_the_path() -> None:
    """Arquivo ausente vira `skip` legível, não erro nem dado inventado.

    É a metade que de fato prova a regra. Um resolvedor que levantasse
    `FileNotFoundError` faria a suíte ficar vermelha em máquina sem os dados,
    e a tentação seria fabricar substituto — exatamente o que a especificação proíbe.
    """
    from tests.conftest import _require

    with pytest.raises(pytest.skip.Exception) as caught:
        _require(REPO / "data" / "reference" / "nao_existe.csv", "arquivo inventado")

    message = str(caught.value)
    assert "nao_existe.csv" in message, "o skip precisa nomear o caminho esperado"
    assert "§8.4" in message or "fabricado" in message, "o skip precisa citar a regra"


@pytest.mark.parametrize("mode", MODES)
def test_resolvers_skip_when_the_file_is_absent(mode: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Os resolvedores herdam o comportamento de skip, não só `_require`."""
    import tests.conftest as helpers

    monkeypatch.setattr(helpers, "REFERENCE", REPO / "data" / "_reference_ausente")
    for resolver in (helpers.measurements_path, helpers.sample_list_path, helpers.toserrf_path):
        with pytest.raises(pytest.skip.Exception):
            resolver(mode)


# ---------------------------------------------------------------------------
# Sanidade estrutural — barata, e pega troca de arquivo
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", MODES)
def test_measurements_has_expected_shape(mode: str) -> None:
    """Dimensões da tabela da especificação, e as duas linhas de metadado.

    Não é o critério — esse exige `load_matrix`, que é da ingestão. É a
    verificação mais barata que detecta um arquivo de referência trocado.
    """
    with measurements_path(mode).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))

    assert rows[0][0] == "NAME", f"linha 1 deveria começar com NAME, veio {rows[0][0]!r}"
    assert rows[1][0] == "LABEL", f"linha 2 deveria começar com LABEL, veio {rows[1][0]!r}"
    assert len(rows) - 2 == EXPECTED_FEATURES[mode]
    assert len(rows[0]) - 1 == EXPECTED_SAMPLES
    assert len({len(row) for row in rows}) == 1, "linhas com contagem de campos desigual"


@pytest.mark.parametrize("mode", MODES)
def test_sample_list_covers_injection_order_1_to_68(mode: str) -> None:
    """A ordem de injeção é contígua de 1 a 68, sem lacunas."""
    with sample_list_path(mode).open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.reader(handle) if row]

    assert rows[0] == ["ID", "inj_order_name"], f"cabeçalho inesperado: {rows[0]}"
    orders = sorted(int(row[0]) for row in rows[1:])
    assert orders == list(range(1, EXPECTED_SAMPLES + 1))


@pytest.mark.parametrize("mode", MODES)
def test_seven_qcs_at_the_documented_injection_orders(mode: str) -> None:
    """7 QCs nas injeções 1, 12, 23, 34, 45, 56 e 68.

    Precursor do critério, feito por leitura direta em vez de por
    `load_matrix`, que ainda não existe.
    """
    with measurements_path(mode).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    names, labels = rows[0][1:], rows[1][1:]

    with sample_list_path(mode).open(newline="", encoding="utf-8") as handle:
        order = {row[1]: int(row[0]) for row in list(csv.reader(handle))[1:] if row}

    qc_orders = sorted(
        order[name] for name, label in zip(names, labels, strict=True) if label == "QC"
    )
    assert qc_orders == list(EXPECTED_QC_ORDERS)


@pytest.mark.parametrize("mode", MODES)
def test_sample_names_join_cleanly_with_the_injection_list(mode: str) -> None:
    """O join measurements <-> Sample_List fecha nos dois sentidos.

    Se falhar, `SAMPLE_LIST_MISMATCH` dispararia nos dados reais e
    o critério seria inatingível.
    """
    with measurements_path(mode).open(newline="", encoding="utf-8") as handle:
        matrix_names = set(next(csv.reader(handle))[1:])
    with sample_list_path(mode).open(newline="", encoding="utf-8") as handle:
        list_names = {row[1] for row in list(csv.reader(handle))[1:] if row}

    assert matrix_names == list_names, (
        f"só na matriz: {sorted(matrix_names - list_names)}\n"
        f"só na lista: {sorted(list_names - matrix_names)}"
    )


# ---------------------------------------------------------------------------
# Integridade
# ---------------------------------------------------------------------------


def test_manifest_exists_and_is_populated() -> None:
    """O manifesto de checksums existe e não está vazio."""
    manifest = read_manifest()
    assert manifest, "manifesto ausente ou vazio; rode scripts/fetch_reference_data.sh --write"
    assert len(manifest) >= 24, f"manifesto raso demais: {len(manifest)} entradas"


def test_every_manifest_entry_matches_its_file() -> None:
    """Nenhum arquivo de referência divergiu do checksum registrado.

    Ausente não é falha aqui — a especificação manda virar skip, e a decisão é do
    resolvedor. Divergente é falha: o dado mudou sob os golden tests.

    **A asserção de não-vacuidade FICA, e a guarda que a precede é o que a torna
    correta depois.** `checked > 0` existe porque um manifesto lido sobre
    raiz vazia percorre zero arquivos e conclui *"nenhum divergiu"* — a asserção de
    ausência satisfeita por vacuidade que o projeto nomeia. Com `serrf/` fora do git,
    raiz vazia deixou de ser sintoma de symlink quebrado e passou a ser o que um
    clone novo tem, então **pular quando não há raiz** e **reprovar quando há raiz
    e nenhum arquivo** são coisas diferentes, e as duas continuam a ser ditas.
    """
    from tests.conftest import REFERENCE, reference_root_or_none

    if reference_root_or_none() is None:
        pytest.skip(
            "dados de referência ausentes: nenhuma raiz resolve, então não há "
            "checksum a conferir. Esperado `data/reference/input` (symlink) ou "
            "`serrf/data/input`. Ver scripts/fetch_reference_data.sh."
        )

    mismatched: list[str] = []
    checked = 0
    for relative, expected in read_manifest().items():
        path = REFERENCE / relative
        if not path.is_file():
            continue
        checked += 1
        if sha256_of(path) != expected:
            mismatched.append(relative)

    assert checked > 0, "nenhum arquivo do manifesto presente — verifique os symlinks"
    assert not mismatched, f"checksum divergente em: {mismatched}"


def test_verification_script_succeeds() -> None:
    """`fetch_reference_data.sh` roda e sai zero na árvore íntegra.

    **A asserção `"divergentes" in stdout` é o que exige a guarda.** Sem raiz o
    script sai zero de todo modo — por desenho, porque a especificação manda `skip` e não
    falha —, mas imprime *"Nenhuma raiz de dados de referência resolve"* e nunca
    chega à linha do resumo. Afirmar o código de saída sem afirmar a linha deixaria
    o teste verde sobre um script que não conferiu nada; afirmar a linha sem a
    guarda reprova todo clone sem dados, que foi medido.
    """
    from tests.conftest import reference_root_or_none

    if reference_root_or_none() is None:
        pytest.skip(
            "dados de referência ausentes: o verificador não chega ao resumo sem "
            "raiz que resolva. Ver scripts/fetch_reference_data.sh."
        )

    result = subprocess.run(
        ["bash", str(REPO / "scripts" / "fetch_reference_data.sh")],
        cwd=REPO,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    assert result.returncode == 0, f"script falhou:\n{result.stdout}\n{result.stderr}"
    assert "divergentes" in result.stdout


def test_verification_script_is_idempotent() -> None:
    """Duas verificações seguidas produzem a mesma saída."""
    script = str(REPO / "scripts" / "fetch_reference_data.sh")
    runs = [
        subprocess.run(
            ["bash", script],
            cwd=REPO,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        for _ in range(2)
    ]
    assert runs[0].stdout == runs[1].stdout
    assert runs[0].returncode == runs[1].returncode == 0


# ---------------------------------------------------------------------------
# O pipeline R continua funcionando
# ---------------------------------------------------------------------------


def test_r_reference_tree_is_intact() -> None:
    """`serrf.R` tem `serrf/data/output/` hardcoded — o caminho segue válido.

    Prova que a exposição por symlink não moveu nada. Se alguém
    "arrumar" isso convertendo os symlinks numa cópia e apagando `serrf/`, o
    pipeline R de referência que as fases 2 e 3 precisam reexecutar quebra.

    **A premissa deste teste mudou em 02/09/2026, e a distinção é a razão de ele
    continuar a existir.** Ele nasceu para impedir que `serrf/` fosse
    apagado por acidente; hoje a pasta está fora do git POR DECISÃO, e ausência
    deixou de ser sintoma. O que ele ainda protege é o caso que importa: **para
    quem TEM a pasta, o layout que o `serrf.R` exige continua intacto.** Converter
    isto num teste sem guarda reprovaria todo clone novo; apagá-lo perderia a única
    afirmação de que os caminhos embutidos no oráculo seguem válidos.
    """
    if not (REPO / "serrf" / "data" / "input").is_dir():
        pytest.skip(
            "dados de referência ausentes: `serrf/` não está neste clone — ela saiu "
            "do git e é obtida por fora. Sem a pasta não há layout a "
            "conferir. Ver scripts/fetch_reference_data.sh."
        )

    assert (REPO / "serrf" / "data" / "input").is_dir()
    assert (REPO / "serrf" / "data" / "output").is_dir()
    assert (REPO / "serrf" / "script" / "serrf.R").is_file()

    source = (REPO / "serrf" / "script" / "serrf.R").read_text(encoding="utf-8", errors="replace")
    assert "serrf/data/output/" in source, (
        "serrf.R não contém mais o caminho hardcoded; a premissa do KTD-9 mudou "
        "e a decisão de usar symlink deve ser revisitada."
    )


def test_reference_root_resolves_por_symlink_ou_por_queda() -> None:
    """A raiz resolve por symlink OU pela queda para `serrf/data`.

    O symlink continua sendo o layout padrão, porque dá o nome lógico da especificação e
    evita duplicar 69 MB. Deixou de ser REQUISITO porque não materializa em
    checkout Windows sem `core.symlinks=true`, e ali o efeito é o pior possível:
    tudo pula em silêncio e a suíte fica verde sem verificar nada.

    Cópia continua exigindo atualizar a lista em vez de acontecer por acidente —
    e é o que a segunda asserção verifica.
    """
    from tests.conftest import reference_root_or_none

    root = reference_root_or_none()
    if root is None:
        pytest.skip(
            # A frase inicial e a DECLARADA em `reference_root_unresolved`, e alinhar a
            # mensagem ao padrao — em vez de alargar o padrao — e deliberado: o padrao e o
            # contrato, e alarga-lo e como a absorcao entra. Este `skip` NAO casava
            # regime nenhum, entao reprovaria a sessao no dia em que disparasse.
            "dados de referência ausentes: nenhuma raiz resolve. Esperado "
            "`data/reference/input` "
            "(symlink) ou `serrf/data/input`. Ver scripts/fetch_reference_data.sh."
        )
    assert (root / "input").is_dir() and (root / "output").is_dir()

    duplicado = (REPO / "data" / "reference" / "input").is_dir() and not (
        REPO / "data" / "reference" / "input"
    ).is_symlink()
    assert not duplicado, (
        "data/reference/input é diretório real e não symlink: 69 MB duplicados, "
        "e as duas cópias podem divergir em silêncio. Cópia é alternativa "
        "legítima (KTD-9) mas exige decisão registrada."
    )
