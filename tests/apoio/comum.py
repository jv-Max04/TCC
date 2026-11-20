"""As duas políticas de diagnóstico que `tests/test_perfil_da_politica.py` usa como oráculo.

Veio do `comum.py` dos scripts de medição do corpus de 7 QCs: as duas funções abaixo estão como
estavam lá, e o resto daquele arquivo (integridade, procedência, manter a máquina acordada) servia
só à medição e não veio. O teste carrega este arquivo por caminho, compilado da fonte em disco.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
for _caminho in (REPO, REPO / "src"):
    if str(_caminho) not in sys.path:
        sys.path.insert(0, str(_caminho))

GERA_CORPUS_GRAVADO = REPO / "scripts" / "gera_corpus_fase9.py"


def politica_do_corpus_gravado(fonte: Path = GERA_CORPUS_GRAVADO) -> Any:
    """A política com que o corpus gravado foi diagnosticado — LIDA de `gera_corpus_fase9.py`.

    **Derivada e não copiada.** O literal vive dentro do `main()` daquele script, onde não se
    importa, e duas cópias de um limiar divergem no dia em que alguém mexe numa delas. Aqui a única
    chamada a `DiagnosisPolicy` daquele arquivo é lida com `ast`; mais de uma, ou argumento que não
    seja literal, para tudo.
    """
    from refino.diagnosis.policy import DiagnosisPolicy

    arvore = ast.parse(fonte.read_text(encoding="utf-8"))
    chamadas = [
        no
        for no in ast.walk(arvore)
        if isinstance(no, ast.Call)
        and isinstance(no.func, ast.Name)
        and no.func.id == "DiagnosisPolicy"
    ]
    if len(chamadas) != 1 or chamadas[0].args:
        raise RuntimeError(
            f"{fonte.name}: esperava uma chamada a DiagnosisPolicy, só com argumentos nomeados; "
            f"achei {len(chamadas)}"
        )
    argumentos = {k.arg: ast.literal_eval(k.value) for k in chamadas[0].keywords}
    return DiagnosisPolicy(**argumentos)


def politica_do_aplicativo() -> Any:
    """A política com que o aplicativo diagnostica: os limiares iniciais e o RSD por modo.

    **Quem mais a lê:** `tests/test_perfil_da_politica.py`, que confere contra ela a constante do
    perfil da política aprendida. Mudar esta função é mudar o que aquele portão exige.
    """
    from refino.diagnosis.policy import (
        LIMIARES_INICIAIS,
        RSD_MEDIANA_MAX_POR_MODO,
        DiagnosisPolicy,
    )

    return DiagnosisPolicy(**LIMIARES_INICIAIS, rsd_mediana_max_por_modo=RSD_MEDIANA_MAX_POR_MODO)
