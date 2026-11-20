"""Os três avisos que o relatório mostra, cada um com dois textos.

Cada aviso diz ao leitor algo que limita o que ele pode concluir do relatório (`por_que_limita`).
`AVISOS` é a lista de referência e diz quais avisos existem. O texto de cada um fica na fixture
`avisos_relatorio.json`, e a conferência compara a fixture contra `AVISOS`, nunca contra o que o
relatório produziu.

Cada aviso tem dois textos em campos separados: `consequencia`, para o corpo do relatório, e
`procedencia`, para o apêndice. Com um campo só, perguntar se o aviso está no corpo e se está no
apêndice seria a mesma pergunta feita duas vezes.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Final

__all__ = [
    "AVISOS",
    "AvisoCanonico",
    "carrega_avisos",
    "defeitos_de_procedencia",
]

REPO = Path(__file__).resolve().parent.parent.parent.parent
FIXTURE = REPO / "data" / "fixtures" / "avisos_relatorio.json"


@dataclass(frozen=True)
class AvisoCanonico:
    """Um aviso obrigatório: a chave dele na fixture e o que ele limita."""

    id: str
    por_que_limita: str


# A lista de referência. Aviso acrescentado aqui e ausente da fixture é acusado por
# `defeitos_de_procedencia`.
AVISOS: Final[tuple[AvisoCanonico, ...]] = (
    AvisoCanonico(
        id="dv18",
        por_que_limita=(
            "o SERRF desta ferramenta não reproduz o do R no dígito — chega perto, e nem o R "
            "repete o próprio número —, e comparação com resultado obtido no original tem de "
            "contar com a diferença"
        ),
    ),
    AvisoCanonico(
        id="dv22",
        por_que_limita=(
            "os limiares de diagnóstico foram calibrados também sobre matrizes sintéticas, e parte "
            "das propriedades delas diverge das reais"
        ),
    ),
    AvisoCanonico(
        id="m8",
        por_que_limita=(
            "a recomendação automática não está provada melhor que uma regra fixa entre modos, "
            "então é ponto de partida e não veredito"
        ),
    ),
)


def carrega_avisos(caminho: Path | None = None) -> dict[str, dict[str, str]]:
    """Lê a fixture dos avisos. Levanta quando o arquivo falta, em vez de devolver vazio.

    Vazio seria indistinguível de 'nenhum aviso a mostrar', e o relatório reprovaria por falta de
    aviso quando o que falta é a fixture.
    """
    p = caminho or FIXTURE
    if not p.exists():
        raise FileNotFoundError(
            f"{p} ausente. Sem a fixture o relatório reprovaria por falta de aviso, escondendo "
            "que falta o insumo"
        )
    dados: dict[str, dict[str, str]] = json.loads(p.read_text(encoding="utf-8"))
    return dados


def defeitos_de_procedencia(dados: dict[str, dict[str, str]]) -> list[str]:
    """Lista os defeitos da fixture contra `AVISOS`, um por problema.

    Acusa aviso canônico ausente, aviso a mais e enquadramento vazio.
    """
    achados: list[str] = []
    canonicos = {a.id: a for a in AVISOS}

    faltando = sorted(set(canonicos) - set(dados))
    sobrando = sorted(set(dados) - set(canonicos))
    if faltando:
        achados.append(
            f"avisos canônicos ausentes da fixture: {faltando}. `AVISOS` diz quais avisos "
            "existem, e a fixture tem de trazer o texto de cada um"
        )
    if sobrando:
        achados.append(f"avisos na fixture e fora de `AVISOS`: {sobrando}")

    for aid in sorted(set(canonicos) & set(dados)):
        registro = dados[aid]
        for campo in ("consequencia", "procedencia"):
            if not registro.get(campo, "").strip():
                achados.append(
                    f"{aid}: `{campo}` vazio. São exigidos os DOIS enquadramentos — o corpo diz a "
                    "consequência e o apêndice a procedência, e um só campo faria as duas "
                    "perguntas serem a mesma"
                )
    return achados
