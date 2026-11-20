"""O pino declarado no `pyproject.toml` é a versão instalada.

**Por que este portão existe, e não é sobre `alembic` nem `matplotlib`.** O `pyproject.toml` explica
que as versões são exatas e não faixas, porque "uma faixa deixaria a resolução escorregar entre
majors". O comentário declara a intenção; **nada verificava que ela vale**. Uma dependência
atualizada no ambiente sem o pino acompanhar — ou o inverso — produz suíte verde sobre versão que o
projeto não declara, e é a forma "nunca reprova" numa superfície que ninguém olha.

**A paridade de quatro lugares é do PYTHON**, e essa já tem portão. O
`Dockerfile` e o CI instalam `-e ".[dev]"`, então herdam o `pyproject` por construção e não podem
divergir dele. O que **pode** divergir é o ambiente contra o arquivo, e é isso que se confere aqui.

**Escopo negativo, declarado.** Confere `dependencies` e `optional-dependencies.dev` do
`pyproject.toml` contra o que está instalado **neste** ambiente, mais os **grupos opcionais
atômicos** pela regra abaixo. Não confere que o ambiente do contêiner ou do runner tenha as mesmas
versões — para isso a evidência é o `-e ".[dev]"` mais o `scripts/env_probe.py`, e nenhum dos dois
roda aqui.

## Os grupos OPCIONAIS não podem entrar em `pinos_declarados`, e a razão é o desfecho

**Esta seção entrou em 22/08/2026 com o grupo `desktop`.**
`defeitos` trata *declarado e não instalado* como defeito — o que está certo para o que é
obrigatório e **inverteria o sentido** de um grupo opcional: pôr `desktop` ali faria a suíte
reprovar em toda máquina que não instalasse 649 MB de PySide6, ou seja, tornaria obrigatório
exactamente o que se decidiu deixar opcional.

Então a regra é outra, e é **atômica**:

| O que está instalado | Veredito |
|---|---|
| **nada** do grupo | zero defeitos — o grupo é opcional e não instalá-lo é legítimo |
| **tudo**, nas versões declaradas | zero defeitos |
| **parte**, ou versão divergente | defeito, um por membro |

> **A linha do meio da tabela é o que este portão acrescenta.** Grupo parcialmente instalado é a
> forma de deriva que ninguém vê: `import PySide6` funciona, o `pytest-qt` falta, e o módulo de
> teste da janela **pula** — com aparência de máquina sem o grupo, que é um desfecho legítimo.
> A atomicidade é o que impede essa absorção.
"""

import importlib.metadata as metadata
import re
import tomllib

from tests.conftest import REPO

PYPROJECT = REPO / "pyproject.toml"

# `name==version`, e so essa forma. Requisito com marcador de ambiente ou extra nao entra: o projeto
# nao usa nenhum hoje, e casar o que nao existe criaria ramo sem teste.
PINO = re.compile(r"^(?P<nome>[A-Za-z0-9._-]+)==(?P<versao>[A-Za-z0-9._+-]+)$")


def pinos_declarados() -> dict[str, str]:
    """`distribuição -> versão` de `dependencies` e de `optional-dependencies.dev`."""
    dados = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    projeto = dados["project"]
    requisitos = list(projeto.get("dependencies", []))
    requisitos += list(projeto.get("optional-dependencies", {}).get("dev", []))
    achados: dict[str, str] = {}
    for req in requisitos:
        m = PINO.match(req.strip())
        if m:
            achados[m.group("nome")] = m.group("versao")
    return achados


def pinos_do_grupo(grupo: str) -> dict[str, str]:
    """`distribuição -> versão` de UM grupo de `optional-dependencies`.

    Separada de `pinos_declarados` porque o veredito é outro: ali a ausência é defeito, aqui a
    ausência **de tudo** é legítima. Ver a tabela no topo do módulo.
    """
    dados = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    requisitos = dados["project"].get("optional-dependencies", {}).get(grupo, [])
    achados: dict[str, str] = {}
    for req in requisitos:
        m = PINO.match(str(req).strip())
        if m:
            achados[m.group("nome")] = m.group("versao")
    return achados


def defeitos_de_grupo_atomico(
    grupo: str, declarados: dict[str, str], instalados: dict[str, str | None]
) -> list[str]:
    """Grupo opcional: ou nenhum membro, ou todos na versão declarada.

    **A planta é argumento** — `declarados` e `instalados` entram por parâmetro, então o negativo
    percorre este mesmo caminho e não consegue ficar inerte.

    **TRÊS desfechos, logo três direções**: nada instalado, tudo a bater, e parcial ou
    divergente. A que fica inerte por omissão é sempre a primeira, porque dispara por **ausência**
    de condição — e é a única que a máquina de quem escreveu o portão não produz, já que ela tem o
    grupo instalado.
    """
    presentes = {n for n, v in instalados.items() if v is not None}
    if not presentes:
        return []

    achados: list[str] = []
    for nome, esperado in sorted(declarados.items()):
        obtido = instalados.get(nome)
        if obtido is None:
            achados.append(
                f"[{grupo}] {nome}: declarado `=={esperado}` e NÃO INSTALADO, mas outros "
                f"membros do grupo estão ({', '.join(sorted(presentes))}). Grupo parcial é "
                "deriva invisível: "
                "o módulo de teste pula com aparência de máquina sem o grupo"
            )
        elif obtido != esperado:
            achados.append(
                f"[{grupo}] {nome}: pyproject declara `=={esperado}` e o instalado é `{obtido}`"
            )
    return achados


def defeitos(declarados: dict[str, str], instalados: dict[str, str | None]) -> list[str]:
    """Um defeito por divergência.

    Função separada devolvendo lista porque é o que torna o negativo honesto: a planta é
    **argumento**, e o negativo percorre o mesmo caminho que o portão. Negativo em processo não
    consegue ficar inerte.
    """
    achados: list[str] = []
    for nome, esperado in sorted(declarados.items()):
        obtido = instalados.get(nome)
        if obtido is None:
            achados.append(
                f"{nome}: declarado `=={esperado}` no pyproject e NÃO INSTALADO neste ambiente"
            )
        elif obtido != esperado:
            achados.append(
                f"{nome}: pyproject declara `=={esperado}` e o instalado é `{obtido}`. "
                "Ou o pino não acompanhou a atualização, ou o ambiente escorregou — as duas fazem "
                "a suíte rodar sobre versão que o projeto não declara"
            )
    return achados


def _instalados(nomes: list[str]) -> dict[str, str | None]:
    saida: dict[str, str | None] = {}
    for nome in nomes:
        try:
            saida[nome] = metadata.version(nome)
        except metadata.PackageNotFoundError:
            saida[nome] = None
    return saida


def test_todo_pino_declarado_bate_com_o_instalado() -> None:
    declarados = pinos_declarados()
    assert declarados, "nenhum pino `nome==versao` lido do pyproject — o regex deixou de casar"
    assert not defeitos(declarados, _instalados(list(declarados)))


def test_as_duas_dependencias_da_fase_10_estao_declaradas_e_instaladas() -> None:
    """A persistência precisa de `alembic`; o relatório e a janela precisam de `matplotlib`.

    O critério não é "importa" — é "importa **na versão que o projeto declara**".

    **O que este teste NÃO alcança, e a qualificação importa:** ele afirma que o pino declarado é o
    instalado **nesta venv**. A venv do **alvo** é outra — construída por `congela_windows.ps1` com
    uma lista própria —, e quem afirma que o mesmo pino chegou lá é o passo *"o matplotlib pinado
    está na venv do alvo"* daquele roteiro.
    """
    declarados = pinos_declarados()
    for nome in ("alembic", "matplotlib"):
        assert nome in declarados, f"{nome} não está pinado no pyproject"
    assert not defeitos(
        {k: v for k, v in declarados.items() if k in {"alembic", "matplotlib"}},
        _instalados(["alembic", "matplotlib"]),
    )


def test_negativo_pino_divergente_reprova() -> None:
    """**A planta é argumento** — versão instalada diferente da declarada tem de acusar."""
    d = defeitos({"numpy": "2.5.1"}, {"numpy": "1.99.0"})
    assert len(d) == 1
    assert "pyproject declara `==2.5.1`" in d[0]
    assert "instalado é `1.99.0`" in d[0]


def test_negativo_dependencia_declarada_e_ausente_reprova() -> None:
    """O outro lado: declarada e não instalada é defeito diferente, e a mensagem os separa."""
    d = defeitos({"alembic": "1.18.5"}, {"alembic": None})
    assert len(d) == 1
    assert "NÃO INSTALADO" in d[0]


def test_o_par_causal_a_arvore_limpa_nao_acusa() -> None:
    """**Visto passar com a planta removida**: sem divergência, zero defeitos.

    Sem esta direção o portão poderia estar sempre acusando e os dois negativos acima continuariam
    verdes — é a forma "sempre reprova".
    """
    assert defeitos({"numpy": "2.5.1"}, {"numpy": "2.5.1"}) == []


# ---------------------------------------------------------------------------
# O grupo opcional `desktop`
# ---------------------------------------------------------------------------


def test_o_grupo_desktop_esta_declarado_e_e_atomico_neste_ambiente() -> None:
    """Os três pinos existem, e o que está instalado é tudo ou nada.

    **A asserção de não-vacuidade vem PRIMEIRO e na mesma função**: sem ela,
    `defeitos_de_grupo_atomico` sobre um dicionário vazio devolve `[]` por construção, e o teste
    passaria com o grupo apagado do `pyproject.toml`.
    """
    declarados = pinos_do_grupo("desktop")
    assert declarados, "o grupo `desktop` não foi lido do pyproject — apagado, ou o regex não casa"
    assert set(declarados) == {"PySide6", "pytest-qt", "pyinstaller"}, (
        f"a composição do grupo mudou sem este portão saber: {sorted(declarados)}"
    )
    assert not defeitos_de_grupo_atomico("desktop", declarados, _instalados(list(declarados)))


def test_o_desfecho_que_esta_maquina_nao_produz_grupo_inteiramente_ausente() -> None:
    """A primeira das três direções, e a que ficaria inerte.

    **Esta máquina tem o grupo instalado**, então o ramo "nada instalado" nunca dispara por conta
    própria aqui — e ramo que não dispara não se deixa construído. A planta
    é argumento, então forçá-lo é passar o dicionário de ausências.
    """
    declarados = {"PySide6": "6.11.2", "pytest-qt": "4.5.0", "pyinstaller": "6.22.2"}
    ausentes: dict[str, str | None] = dict.fromkeys(declarados)
    assert defeitos_de_grupo_atomico("desktop", declarados, ausentes) == []


def test_negativo_grupo_parcialmente_instalado_reprova() -> None:
    """A segunda direção, e é a que motiva o portão existir.

    `PySide6` presente e `pytest-qt` ausente faz o módulo da janela **pular** com aparência de
    máquina sem o grupo. A mensagem tem de nomear os dois lados, senão quem a lê conclui "falta
    instalar o grupo" quando o grupo está meio instalado.
    """
    d = defeitos_de_grupo_atomico(
        "desktop",
        {"PySide6": "6.11.2", "pytest-qt": "4.5.0", "pyinstaller": "6.22.2"},
        {"PySide6": "6.11.2", "pytest-qt": None, "pyinstaller": "6.22.2"},
    )
    assert len(d) == 1, f"esperava um defeito, veio {d}"
    assert "pytest-qt" in d[0]
    assert "NÃO INSTALADO" in d[0]
    assert "PySide6" in d[0], "a mensagem não diz quais membros ESTÃO lá, e é isso que a distingue"


def test_negativo_grupo_com_versao_divergente_reprova() -> None:
    """A terceira direção: tudo instalado, uma versão fora do pino."""
    d = defeitos_de_grupo_atomico(
        "desktop",
        {"PySide6": "6.11.2"},
        {"PySide6": "6.10.0"},
    )
    assert len(d) == 1
    assert "declara `==6.11.2`" in d[0]
    assert "instalado é `6.10.0`" in d[0]


def test_o_grupo_opcional_nao_entra_em_pinos_declarados() -> None:
    """A separação é o conteúdo, e este teste é o que a prova em vez de a descrever.

    Se `desktop` escorregasse para `pinos_declarados`, `defeitos` trataria a ausência como defeito e
    o grupo passaria a ser obrigatório — o oposto, e numa direção que só reprova na máquina
    de quem **não** o instalou.
    """
    obrigatorios = pinos_declarados()
    assert obrigatorios, "âncora: `pinos_declarados` não pode estar vazio"
    for membro in pinos_do_grupo("desktop"):
        assert membro not in obrigatorios, (
            f"`{membro}` está nos pinos obrigatórios e no grupo opcional — duas fontes do mesmo "
            "pino, e a ausência passaria a reprovar"
        )
