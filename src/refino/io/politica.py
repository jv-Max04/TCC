"""Lê do disco os artefatos da política aprendida.

`decision/artefato.py` é núcleo puro e não abre arquivo. O carregador fica aqui para que a CLI, a
rota e o gerador resolvam o caminho num lugar só.

Os artefatos ficam dentro do pacote `decision`, e o caminho é relativo a ele. Assim resolve igual
no clone e no executável congelado, onde `data/fixtures/` não existe. O JSON precisa estar listado
em `datas=` no spec do PyInstaller, que não o colhe sozinho.
"""

from __future__ import annotations

import json
from pathlib import Path

from refino import decision
from refino.decision.artefato import ArtefatoInvalidoError, PoliticaDoArtefato

__all__ = [
    "ARTEFATO_EMBALADO",
    "ARTEFATO_PELA_VERDADE",
    "SEGUNDA_OPINIAO_ATIVA",
    "carrega_politica_aprendida",
    "carrega_politica_pela_verdade",
    "segunda_opiniao_ativa",
]

ARTEFATO_EMBALADO: Path = Path(decision.__file__).resolve().parent / "politica_aprendida.json"

# O segundo artefato, o Palpite pela verdade: a mesma árvore, treinada pelo corretor que mais
# recupera o sinal nas matrizes sintéticas. O automático segue o Palpite pelo RSD; este é a
# segunda opinião.
ARTEFATO_PELA_VERDADE: Path = (
    Path(decision.__file__).resolve().parent / "politica_pela_verdade.json"
)

# A segunda opinião está desligada. O artefato e o carregador ficam, e o congelado continua a
# levá-los, mas nenhuma superfície a consulta: a CLI não a imprime nem a grava no log, e a rota
# devolve `segunda_opiniao` nulo. Religar é trocar este valor; os testes correm as duas posições.
SEGUNDA_OPINIAO_ATIVA = False


def segunda_opiniao_ativa() -> bool:
    """Lê o interruptor na chamada, e não no import, para que um teste possa ligá-lo."""
    return SEGUNDA_OPINIAO_ATIVA


_REGERA_PELA_VERDADE = (
    "python scripts/gera_politica_aprendida.py --corpus "
    "data/fixtures/corpus_pela_verdade_88.jsonl --destino "
    "src/refino/decision/politica_pela_verdade.json"
)


def carrega_politica_aprendida(caminho: Path | None = None) -> PoliticaDoArtefato:
    """Lê e valida o artefato. Levanta `ArtefatoInvalidoError` em qualquer falha.

    Ausente, ilegível como JSON e inválido levam à mesma ação de quem chama, que é recusar o
    `--auto` e dizer por quê; a mensagem distingue os três. Não há política padrão de reserva:
    uma recomendação que ninguém treinou não pode passar por aprendida.
    """
    return _carrega(
        ARTEFATO_EMBALADO if caminho is None else caminho,
        nome="a política aprendida",
        regera="python scripts/gera_politica_aprendida.py",
        artigo="a",
    )


def carrega_politica_pela_verdade(caminho: Path | None = None) -> PoliticaDoArtefato:
    """Lê e valida o artefato do Palpite pela verdade, a segunda opinião, que não decide nada.

    Mesmo contrato do carregador da política aprendida, pelo mesmo leitor (`_carrega`). Mudam o
    caminho e o comando que o regera, que a mensagem de erro nomeia.
    """
    return _carrega(
        ARTEFATO_PELA_VERDADE if caminho is None else caminho,
        nome="o Palpite pela verdade",
        regera=_REGERA_PELA_VERDADE,
        artigo="o",
    )


def _carrega(alvo: Path, *, nome: str, regera: str, artigo: str) -> PoliticaDoArtefato:
    """Leitor comum aos dois artefatos. `artigo` concorda o particípio e o pronome com o nome."""
    lida = "lida" if artigo == "a" else "lido"
    try:
        bruto = alvo.read_text(encoding="utf-8")
    except OSError as erro:
        raise ArtefatoInvalidoError(
            f"{nome} não pôde ser {lida} em {alvo}: {erro}. Regere-{artigo} com `{regera}`."
        ) from erro
    try:
        dados = json.loads(bruto)
    except json.JSONDecodeError as erro:
        raise ArtefatoInvalidoError(
            f"{nome} em {alvo} não é JSON válido: {erro}. Regere-{artigo} com `{regera}`."
        ) from erro
    return PoliticaDoArtefato.de_dict(dados)
