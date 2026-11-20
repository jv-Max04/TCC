"""De onde vem a URL do banco.

A URL vem de uma variável só, `REFINO_DB_URL`. Sem ela não há banco, e `url_do_banco` levanta
dizendo como declará-la. A API e o alembic leem a URL desta função: duas cópias da regra
deixariam a aplicação gravar num banco e a migração correr noutro.

O arquivo do SQLite guarda dado clínico: `samples.name` e `samples.grupo` são o par de
pseudônimo e diagnóstico. Por isso, dentro do repositório o arquivo só é aceito sob `out/`, que
o `.gitignore` ignora inteiro, e ele nasce com modo 0600. As mensagens de erro deste módulo
nomeiam a configuração e nunca o valor da URL, que pode trazer credencial em linha e chega à
tela pela resposta da API.
"""

from __future__ import annotations

import os
import stat
import sys
from pathlib import Path

__all__ = [
    "DIALETOS_ACEITOS",
    "VARIAVEL_DA_URL",
    "ConfiguracaoDeBancoError",
    "caminho_do_sqlite",
    "prepara_sqlite",
    "url_do_banco",
]

# O nome da configuração, num lugar só: ele aparece em mensagem de erro e em teste.
VARIAVEL_DA_URL = "REFINO_DB_URL"

# A raiz do repositório, derivada do próprio arquivo e não do `git`, para a guarda não depender
# de o `git` existir no ambiente de quem roda a ferramenta.
RAIZ_DO_REPOSITORIO = Path(__file__).resolve().parents[3]

DIRETORIOS_IGNORADOS = ("out",)
"""Os diretórios da árvore em que o banco pode ficar.

Entram só os que o `.gitignore` ignora inteiros e por diretório, o que cobre os arquivos que o
SQLite escreve ao lado do banco. `dist/` também é ignorado assim e fica de fora porque nada
precisa dele: o executável congelado não passa por esta guarda.
"""

# Vocabulário fechado. Um dialeto que o projeto não exercita falharia mais fundo, com mensagem
# do SQLAlchemy sobre um driver que ninguém instalou. Recusar aqui nomeia a configuração e os
# valores aceitos.
DIALETOS_ACEITOS = ("postgresql", "sqlite")


class ConfiguracaoDeBancoError(ValueError):
    """A configuração do banco não serve. A mensagem traz o motivo e nunca a credencial.

    `ValueError` porque é valor de entrada malformado, e é isso que permite a `api/rotas.py`
    traduzi-la para 503 sem inspecionar a mensagem.
    """


def url_do_banco(ambiente: dict[str, str] | None = None) -> str:
    """A URL do banco, lida de `REFINO_DB_URL`. Sem ela, levanta dizendo como declará-la.

    `ambiente` é injetado para o teste não mexer em `os.environ`: uma variável exportada de
    verdade vazaria para os testes que rodam depois.
    """
    env = os.environ if ambiente is None else ambiente

    declarada = (env.get(VARIAVEL_DA_URL) or "").strip()
    if not declarada:
        raise ConfiguracaoDeBancoError(
            f"{VARIAVEL_DA_URL} ausente: não há banco onde gravar a execução. Defina a variável "
            "com a URL de um arquivo SQLite, por exemplo `sqlite:///out/refino.db`"
        )
    _recusa_se_invalida(declarada)
    return declarada


def _recusa_se_invalida(url: str) -> None:
    """Recusa a URL sem esquema ou de dialeto que o projeto não aceita.

    Nenhuma mensagem interpola `url`: ela pode trazer senha em linha, e o texto da exceção chega
    à tela do usuário.
    """
    dialeto = url.split(":", 1)[0].split("+", 1)[0]
    if not dialeto or "://" not in url:
        raise ConfiguracaoDeBancoError(
            f"{VARIAVEL_DA_URL} não é uma URL de banco: falta o esquema `dialeto://`. "
            f"Aceitos: {', '.join(DIALETOS_ACEITOS)}"
        )
    if dialeto not in DIALETOS_ACEITOS:
        raise ConfiguracaoDeBancoError(
            f"{VARIAVEL_DA_URL} declara o dialeto {dialeto!r}, que este projeto não exercita. "
            f"Aceitos: {', '.join(DIALETOS_ACEITOS)}"
        )


def caminho_do_sqlite(url: str) -> Path | None:
    """O arquivo que a URL nomeia, ou `None` quando ela não é SQLite.

    `None` significa que não há arquivo, e quem chama decide se há diretório a preparar.
    """
    if not url.startswith("sqlite"):
        return None
    _, _, resto = url.partition("://")
    # Tira exatamente uma barra, e isso resolve as quatro formas: `/relativo.db` ->
    # `relativo.db`, `//absoluto.db` -> `/absoluto.db`, `/:memory:` -> `:memory:`, e vazio ->
    # sem arquivo.
    if resto.startswith("/"):
        return Path(resto[1:])
    return Path(resto) if resto else None


def _recusa_dentro_da_arvore(caminho: Path, *, congelado: bool | None = None) -> None:
    """Recusa banco em arquivo dentro do repositório e fora de `out/`.

    A guarda é por lugar e não por extensão. Os padrões `*.db`, `*.sqlite` e `*.sqlite3` do
    `.gitignore` não casam os arquivos que o SQLite escreve ao lado do banco (`-wal`, `-journal`
    e `-shm`), que trazem pseudônimo e diagnóstico em bytes crus, nem um banco sem sufixo.
    Dentro da árvore só valem os diretórios de `DIRETORIOS_IGNORADOS`, que o `.gitignore`
    ignora inteiros. Fora da árvore o caminho é livre: `git add` não o alcança, e onde o dado
    fica é decisão do operador.

    No executável congelado a guarda não se aplica.
    """
    if congelado if congelado is not None else getattr(sys, "frozen", False):
        # No executável congelado não há repositório nem `git add` a proteger, e a guarda
        # impediria o programa de abrir. O banco fica ao lado do executável, para que a pasta
        # zipada e devolvida leve junto o registro do que foi feito. Só que ali este módulo vive
        # em `_internal/refino/persistence/`, `RAIZ_DO_REPOSITORIO` cai na pasta do executável,
        # e o caminho relativo `refino.db` não começa por um diretório permitido.
        #
        # `congelado` entra por injeção porque ler só `sys.frozen` deixaria este ramo fora do
        # alcance dos testes.
        return
    try:
        dentro = caminho.resolve().relative_to(RAIZ_DO_REPOSITORIO)
    except ValueError:
        # Fora da árvore: `git add` não o alcança, e onde o dado fica é decisão do operador.
        return
    if dentro.parts and dentro.parts[0] in DIRETORIOS_IGNORADOS:
        return
    # A mensagem concorda em número com a lista, que hoje tem um item.
    permitidos = ", ".join(f"`{nome}/`" for nome in DIRETORIOS_IGNORADOS)
    ignorados = "ignorado" if len(DIRETORIOS_IGNORADOS) == 1 else "ignorados"
    raise ConfiguracaoDeBancoError(
        f"{VARIAVEL_DA_URL} aponta para dentro do repositório e fora de {permitidos}. "
        "As tabelas gravam pseudónimo de paciente e diagnóstico, e o SQLite escreve `-wal`, "
        "`-journal` e `-shm` ao lado do banco — os três ficam fora dos padrões por extensão do "
        f"`.gitignore` e entrariam num `git add -A`. Use {permitidos} ({ignorados} por "
        "diretório, o que cobre os side files) ou um caminho fora da árvore"
    )


def prepara_sqlite(url: str, *, congelado: bool | None = None) -> Path | None:
    """Cria o diretório e o arquivo, e fixa o modo 0600. Devolve o caminho, ou `None`.

    É chamada antes de o motor abrir o banco. O `sqlite3` criaria o arquivo com o `umask` do
    processo, que costuma dar 0644, legível por qualquer conta local. Criar o arquivo vazio e
    fixar o modo antes de qualquer escrita fecha essa janela, e a ordem importa: os arquivos
    auxiliares do SQLite herdam o modo do principal.

    O modo 0600 é propriedade de POSIX. No Windows `Path.chmod` só alterna somente-leitura, e a
    proteção vem da ACL herdada do diretório: restrita sob o perfil do usuário, ampla na raiz
    do disco. Como o executável congelado grava ao lado de si mesmo, `desktop/banco.py` mede
    essa ACL no arranque em vez de a supor.

    Devolve `None` para URL que não é SQLite e para banco em memória.
    """
    caminho = caminho_do_sqlite(url)
    if caminho is None:
        return None
    if str(caminho) == ":memory:":
        return None
    _recusa_dentro_da_arvore(caminho, congelado=congelado)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    caminho.touch(exist_ok=True)
    caminho.chmod(stat.S_IRUSR | stat.S_IWUSR)
    return caminho
