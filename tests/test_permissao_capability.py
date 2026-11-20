"""As duas sondas de permissão, nas duas direções.

## Por que a sonda precisa de teste próprio

Ela decide se **quatro** testes correm ou pulam, e `skip` é a única forma de falha em que exercitar
NÃO revela: o desfecho é `skipped`, que é exactamente o que a política ensina a ler
como legítimo. Uma sonda que devolvesse `False` por engano desligaria os quatro em silêncio, em
toda máquina, e a saída continuaria saudável.

**Nesta máquina as duas devolvem `False`**, e é o único desfecho que a árvore produz sozinha. O
ramo `True` — o que faz os quatro testes correrem — só se vê com planta, e é o que este arquivo
faz. Sem ele a sonda teria **uma** direção exercitada e a outra por acreditar.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from tests.permissao_capability import (
    DETALHE_DO_MODO,
    DETALHE_DO_SEM_ESCRITA,
    modo_de_arquivo_respeitado,
    permissao_de_diretorio_respeitada,
)

if TYPE_CHECKING:  # pragma: no cover - só para o mypy
    import pytest


def test_as_duas_sondas_respondem_booleano_e_nao_levantam() -> None:
    """Âncora de não-vacuidade: a sonda **corre** nesta máquina, seja qual for a resposta.

    Sem isto, uma sonda que levantasse faria os testes guardados erroar em vez de pular — e o
    desfecho seria atribuído ao código sob teste.
    """
    assert isinstance(permissao_de_diretorio_respeitada(), bool)
    assert isinstance(modo_de_arquivo_respeitado(), bool)


def test_a_sonda_do_diretorio_responde_SIM_quando_a_escrita_FALHA(  # noqa: N802
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """O ramo `True`, que esta plataforma não produz sozinha.

    **A planta cai onde a sonda decide** — a escrita dentro do diretório —, e não noutro sítio: é a
    cláusula da planta. Com `write_text` a levantar `PermissionError`, o sistema comporta-se
    como um POSIX que honra `0o500`, e a resposta tem de inverter.
    """
    original = Path.write_text

    def recusa(self: Path, *a: object, **k: object) -> int:
        if self.name == "sonda":
            raise PermissionError(13, "plantado")
        return original(self, *a, **k)  # type: ignore[arg-type]

    monkeypatch.setattr(Path, "write_text", recusa)
    assert permissao_de_diretorio_respeitada() is True


def test_a_sonda_do_diretorio_responde_NAO_com_a_planta_REMOVIDA(  # noqa: N802
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """O par causal: a mesma chamada, sem a planta, devolve o outro valor.

    Sem este, o teste acima seria verde vazio se a sonda devolvesse `True` sempre — e é justamente
    o `True` que faz os quatro testes guardados **correrem**, então uma sonda presa nele reprovaria
    quatro testes por um motivo que não é o deles.
    """
    monkeypatch.undo()
    assert permissao_de_diretorio_respeitada() is False, (
        "esta máquina honra permissão POSIX de diretório — então os quatro testes guardados correm "
        "inteiros aqui, e este par causal deixou de medir o que mede. Não é defeito: é outra "
        "plataforma. Inverta a expectativa deste arquivo ao migrar a suíte"
    )


def test_a_sonda_do_MODO_responde_SIM_quando_o_bit_sobrevive(  # noqa: N802
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """O ramo `True` da segunda sonda, e ela é SEPARADA da primeira por desenho."""
    original = Path.stat

    def finge(self: Path, **k: object) -> object:
        real = original(self, **k)  # type: ignore[arg-type]

        class _Modo:
            st_mode = 0o100600

            def __getattr__(self, nome: str) -> object:
                return getattr(real, nome)

        return _Modo() if self.name == "sonda" else real

    monkeypatch.setattr(Path, "stat", finge)
    assert modo_de_arquivo_respeitado() is True


def test_os_dois_detalhes_sao_DIFERENTES_e_nenhum_traz_o_discriminador() -> None:  # noqa: N802
    """Duas propriedades que o portão de `skip` não confere, e as duas já morderam.

    1. **Os detalhes diferem.** Colapsá-los num só tornaria indistinguíveis dois motivos de `skip`
       que mandam o leitor a sítios diferentes.
    2. **O discriminador NÃO está aqui.** Ele vai literal em cada `pytest.skip(f"...")`, porque
       `test_skip_policy.py` lê o texto estaticamente; repeti-lo na constante produziria
       *"permissão POSIX ignorada: permissão POSIX ignorada: ..."* na mensagem que o humano lê.
    """
    assert DETALHE_DO_SEM_ESCRITA != DETALHE_DO_MODO
    for detalhe in (DETALHE_DO_SEM_ESCRITA, DETALHE_DO_MODO):
        assert "permissão POSIX ignorada" not in detalhe, detalhe
