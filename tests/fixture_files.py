"""Gera os nove fixtures nomeados na especificação em `data/fixtures/`.

Separado de `factories.py` de propósito: a factory produz casos em memória para
os testes de validação; este módulo materializa os arquivos que a especificação nomeia,
que servem a mais de um propósito — `tiny_drift` e `tiny_step` alimentam as
fases 3 e 4, e os `corrupt_*` documentam por exemplo os defeitos que o
laboratório relata.

Regenerar: `python -m tests.fixture_files`.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from tests.factories import (
    MatrixSpec,
    render_measurements,
    render_sample_list,
    tiny_clean_spec,
)

FIXTURES = Path(__file__).resolve().parent.parent / "data" / "fixtures"

DRIFT_PER_INJECTION = 0.10
STEP_FACTOR = 1.5
STEP_AFTER_INJECTION = 3


def _scaled(spec: MatrixSpec, factor_for_order) -> MatrixSpec:
    """Multiplica cada coluna por um fator função da ordem de injeção."""
    values = tuple(
        tuple(
            spec.values[f][s] * factor_for_order(spec.samples[s].injection_order)
            for s in range(len(spec.samples))
        )
        for f in range(len(spec.feature_names))
    )
    return replace(spec, values=values)


def tiny_clean() -> MatrixSpec:
    """6 amostras (2 QCs), 5 features, sem defeito estrutural."""
    return tiny_clean_spec()


def tiny_drift() -> MatrixSpec:
    """A mesma matriz com deriva linear de 10% por injeção."""
    return _scaled(tiny_clean(), lambda order: (1 + DRIFT_PER_INJECTION) ** (order - 1))


def tiny_step() -> MatrixSpec:
    """A mesma matriz com salto de 50% após a terceira injeção."""
    return _scaled(tiny_clean(), lambda order: STEP_FACTOR if order > STEP_AFTER_INJECTION else 1.0)


# Posições zeradas em `tiny_zeros.csv`, conhecidas e afirmadas por teste.
ZERO_POSITIONS = ((0, 0), (0, 3), (2, 1), (4, 5))


def tiny_zeros() -> MatrixSpec:
    """Zeros em posições conhecidas."""
    spec = tiny_clean()
    values = [list(row) for row in spec.values]
    for f, s in ZERO_POSITIONS:
        values[f][s] = 0.0
    return replace(spec, values=tuple(tuple(r) for r in values))


def tiny_tab() -> MatrixSpec:
    """Delimitador de tabulação, decimal `.`."""
    return replace(tiny_clean(), delimiter="\t")


def tiny_quoted() -> MatrixSpec:
    """Vírgula literal dentro de campo entre aspas.

    Um nome de feature carrega vírgula interna, então a contagem ingênua de
    vírgulas por linha difere da contagem de campos. É o fixture que dá dente à
    cláusula "fora de aspas" do passo 1 do sniffer.
    """
    spec = tiny_clean()
    names = list(spec.feature_names)
    names[0] = "1.00_100.0000m/z, aduto [M+H]+"
    return replace(spec, feature_names=tuple(names))


def corrupt_thousands() -> MatrixSpec:
    """Separador de milhar DETECTÁVEL — pt-BR com ponto de milhar.

    O arquivo é `;`-delimitado com decimal `,`, e algumas células trazem `.`
    agrupando milhares: `1.234` num arquivo onde o decimal é vírgula só pode ser
    mil duzentos e trinta e quatro.

    A primeira versão deste fixture punha `1.234` num arquivo `,`/`.` — e
    aquilo é **indetectável por construção**: com ponto decimal, `1.234` pode
    ser um vírgula duzentos e trinta e quatro ou mil duzentos e trinta e quatro,
    e é justamente a ambiguidade que não se adivinha. O fixture testava
    nada.

    O mecanismo é o real: abrir o CSV numa planilha com locale pt-BR reformata
    números com separador de milhar — o mesmo round-trip por planilha que
    explica as 38 células com expoente `E` maiúsculo nos arquivos de referência.
    """
    spec = replace(tiny_clean(), delimiter=";", decimal=",")
    # Valores fracionários: sem eles o passo 2 não tem evidência para eleger a
    # vírgula como decimal, e `1.234` volta a ser ambíguo. Um pt-BR real tem
    # decimais em abundância — é isso que torna o ponto de milhar detectável.
    values = tuple(tuple(v + 0.5 for v in row) for row in spec.values)
    spec = replace(spec, values=values)
    return replace(spec, raw_overrides={(0, 0): "1.234", (1, 2): "12.345.678"})


def corrupt_ragged() -> MatrixSpec:
    """Linha com contagem de campos diferente do cabeçalho."""
    return replace(tiny_clean(), ragged_row=2)


def corrupt_semicolon() -> MatrixSpec:
    """pt-BR VÁLIDO: delimitador `;`, decimal `,`.

    O nome engana — o arquivo não é corrompido. Ao menos um valor tem
    exatamente três casas decimais depois da vírgula, para provar a segunda
    metade do critério: um decimal legítimo não é corrupção de milhar.
    """
    spec = replace(tiny_clean(), delimiter=";", decimal=",")
    # Frações de tamanho MISTO, como num pt-BR real. As de 3 dígitos (`1,234`)
    # são o caso ambíguo que prova a segunda metade do critério; as de 1 e 2
    # dígitos são a evidência léxica forte que resolve o decimal do arquivo —
    # um grupo de tamanho != 3 não pode ser milhar.
    #
    # Um arquivo com SÓ frações de 3 dígitos é genuinamente irredutível, e a
    # primeira versão deste fixture era exatamente isso: sem evidência, o
    # decimal caía no padrão `.` e `1,234` voltava a ser acusado de corrupção —
    # reintroduzindo o falso positivo que D3 existe para impedir.
    values = [list(row) for row in spec.values]
    values[0][0] = 1.234
    values[1][1] = 12.345
    values[2][2] = 1030.5  # fração de 1 dígito: evidência forte
    values[3][3] = 1040.25  # fração de 2 dígitos: evidência forte
    return replace(spec, values=tuple(tuple(r) for r in values))


# Nome do arquivo -> construtor. Ordem = ordem da especificação.
FIXTURE_BUILDERS = {
    "tiny_clean": tiny_clean,
    "tiny_drift": tiny_drift,
    "tiny_step": tiny_step,
    "tiny_zeros": tiny_zeros,
    "tiny_tab": tiny_tab,
    "tiny_quoted": tiny_quoted,
    "corrupt_thousands": corrupt_thousands,
    "corrupt_ragged": corrupt_ragged,
    "corrupt_semicolon": corrupt_semicolon,
}


def write_all(destination: Path | None = None) -> list[Path]:
    """Escreve os nove `measurements` e os `Sample_List` correspondentes."""
    out = destination or FIXTURES
    out.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name, build in FIXTURE_BUILDERS.items():
        spec = build()
        measurements = out / f"{name}.csv"
        measurements.write_text(render_measurements(spec), encoding="utf-8", newline="")
        written.append(measurements)
        sample_list = out / f"{name}_sample_list.csv"
        sample_list.write_text(render_sample_list(spec), encoding="utf-8", newline="")
        written.append(sample_list)
    return written


if __name__ == "__main__":  # pragma: no cover - utilitário de linha de comando
    for path in write_all():
        print(path)
