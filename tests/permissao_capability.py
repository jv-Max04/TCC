"""O sistema de arquivos honra permissão POSIX? — a guarda por CAPACIDADE, não por plataforma.

## O que se mede, e por que não é `sys.platform == "win32"`

`Path.mkdir(mode=0o500)` e `Path.chmod(0o600)` **não levantam** no Windows: eles são aceites e
ignorados. O diretório fica gravável e o arquivo nasce em `0o666`. Um teste que construa o cenário
*"diretório sem permissão de escrita"* por esse caminho não constrói cenário nenhum ali — e o
desfecho não é *"o código está errado"*, é **a medição não aconteceu**.

**A guarda pergunta pela propriedade e não pelo nome do sistema**, que é o gesto de
`docker_capability.py` e dos `shutil.which` deste repositório: um sistema de arquivos montado sem
suporte a permissão — FAT32 num Linux, um volume de contêiner mal montado — falha do mesmo modo, e
um `sys.platform` deixaria esse caso a reprovar sem explicação. **Medir custa um diretório
temporário.**

## As duas capacidades são SEPARADAS de propósito

Um sistema pode honrar o modo de um **arquivo** e não a permissão de um **diretório**, e as duas
respostas mandam o leitor a sítios diferentes. Colapsá-las num só predicado tornaria
indistinguíveis dois motivos de `skip` — que é exactamente o defeito que a política de `skip`
existe para impedir.

## O que ele NÃO afirma

Nada sobre o **código sob teste**. Estas funções medem o sistema de arquivos do hospedeiro, e a
única coisa que autorizam é pular com motivo nomeado. Quando a resposta é `True`, o teste corre
inteiro e a asserção é a de sempre.
"""

from __future__ import annotations

import os
import stat
import tempfile
from pathlib import Path

__all__ = [
    "DETALHE_DO_MODO",
    "DETALHE_DO_SEM_ESCRITA",
    "modo_de_arquivo_respeitado",
    "permissao_de_diretorio_respeitada",
]

# **O DISCRIMINADOR fica literal no sítio e só o DETALHE vive aqui, e a divisão foi imposta por um
# portão.** `tests/test_skip_policy.py` lê o `pytest.skip(...)` **estaticamente** — literais
# e pedaços de f-string — e a primeira redação passava a mensagem inteira por constante importada:
# o extrator não a via, e o portão acusou *"regime declara os sítios [...] e NENHUM emite mensagem
# que case"*. Afrouxar o portão seria a resposta errada; a certa é que quem abre o arquivo de teste
# **leia ali** por que ele pula, e é isso que a f-string com prefixo literal garante.
DETALHE_DO_SEM_ESCRITA = (
    "o sistema de arquivos aceita `mkdir(mode=0o500)` e deixa o diretório gravável, então o "
    "cenário «sem permissão de escrita» não se constrói aqui"
)
DETALHE_DO_MODO = (
    "o sistema de arquivos não guarda os bits de modo de um arquivo, então `0o600` não é "
    "observável aqui"
)


def permissao_de_diretorio_respeitada() -> bool:
    """`True` quando `mkdir(mode=0o500)` de facto impede escrita dentro do diretório.

    **O predicado tenta ESCREVER**, e não lê o modo de volta: `stat` pode devolver `0o500` num
    sistema que mesmo assim deixa criar o arquivo, e é a escrita — não o número — que o cenário sob
    teste precisa que falhe.
    """
    with tempfile.TemporaryDirectory() as raiz:
        alvo = Path(raiz) / "trancado"
        alvo.mkdir(mode=0o500)
        try:
            if os.geteuid() == 0 if hasattr(os, "geteuid") else False:  # pragma: no cover - root
                # Root ignora a permissão, e o cenário também não se constrói — a resposta é a
                # mesma e a causa é outra. Dizê-lo aqui impede que alguém leia o `skip` como
                # «este sistema não tem POSIX» ao correr a suíte como root.
                return False
            try:
                (alvo / "sonda").write_text("x", encoding="utf-8")
            except (PermissionError, OSError):
                return True
            return False
        finally:
            alvo.chmod(stat.S_IRWXU)


def modo_de_arquivo_respeitado() -> bool:
    """`True` quando `chmod(0o600)` fica observável em `stat().st_mode`."""
    with tempfile.TemporaryDirectory() as raiz:
        alvo = Path(raiz) / "sonda"
        alvo.write_text("x", encoding="utf-8")
        alvo.chmod(0o600)
        return stat.S_IMODE(alvo.stat().st_mode) == 0o600
