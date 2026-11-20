"""O `--check` de cada gerador compara `python_version` na granularidade que o projeto pina.

## O defeito, e por que ele é de CLASSE e não de item

Em 16/08/2026, `test_a_regeneracao_reproduz_o_fixture_versionado` — em
`tests/golden/test_insample_medians.py` — reprovou numa máquina com CPython **3.12.14**. Medido
por regeneração e diff, com restauração conferida: **a única linha divergente era
`"python_version": "3.12.13"` → `"3.12.14"`**, e as doze células e todas as quantidades eram
idênticas.

O `--check` comparava o arquivo linha a linha ignorando só `generated_at`, então o **patch** do
CPython o reprovava sem nenhum número mudar — portão que reprova
**pelo motivo errado**. O referente que a docstring daquele teste declara é *"o fixture não foi
editado à mão"*, e o que ele afirmava era mais largo.

**E o patch não é constraint deste projeto.** Os pinos do Python são todos no **minor**:

| Onde | O que pina |
|---|---|
| `pyproject.toml` | `requires-python = ">=3.12,<3.13"` |
| `[tool.mypy]` | `python_version = "3.12"` |

Comparar o patch tornava seis fixtures **mais estritos que os pinos declarados**.

## Por que este arquivo existe, e não só o conserto

Consertar o `gen_insample_medians.py` fecharia **um** item. A varredura da classe achou **seis**
fixtures gravando `python_version`, e
**quatro** geradores que o comparam. Só um tinha reprovado:

| Gerador | Como apareceu |
|---|---|
| `gen_insample_medians.py` | **reprovou** no `make check` |
| `gen_r_cv_prereg.py` | **reprova**, e o teste é `slow`: só no `make test-all` |
| `gen_loess_golden.py` | **não apareceria** — o `--check` exige `--runs-dir` |
| `build_serrf_intermediates.py` | **não apareceria** — o `--check` exige `--dump-dir` |

O `make test-all` é o sinal de merge, então o segundo esperaria até lá.

Os dois últimos esperariam a próxima regeneração para reprovar, e ali a leitura natural seria
*"a regeneração divergiu"* — sobre um artefato do oráculo, que é a pior direção possível para
uma falsa divergência.

**Este portão fecha a classe em vez de os quatro itens.** As quatro cópias da normalização são
duplicação deliberada — os geradores são autónomos por desenho, e o `gen_insample_medians.py` é
uma **segunda transcrição** da fórmula —, e é este arquivo que impede as quatro de
divergirem. Duplicação com portão é duplicação segura; sem portão é a segunda fonte de verdade.

## O que ele afirma, nas três direções

Para cada membro, sobre o texto do fixture **real**:

- mudar o **patch** (`3.12.13` → `3.12.99`) ⇒ compara **igual**
- mudar o **minor** (`3.12.13` → `3.11.13`) ⇒ compara **diferente**
- mudar o `numpy_version` ⇒ compara **diferente**

A terceira é a que impede o conserto de virar afrouxamento: se alguém trocasse a normalização
por *"apague a linha da procedência"*, as três primeiras passariam e esta reprovaria. E o
`numpy` está pinado exato em `pyproject.toml`, então divergência ali é mudança deliberada e
tem de continuar a acusar.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any, Final

import pytest

from tests.conftest import REPO

FIXTURES: Final[Path] = REPO / "data" / "fixtures"
SCRIPTS: Final[Path] = REPO / "scripts"

# **Os membros da classe: gerador, a função que compara, e como invocá-la.**
#
# O adaptador existe porque as assinaturas divergem por motivo legítimo — três comparam TEXTO
# linha a linha e `build_serrf_intermediates` compara o JSON já parseado, porque ele também
# remove uma chave de dentro de `step1.params`. Uniformizar as assinaturas seria reescrever
# quatro geradores para caber num teste.
MEMBROS: Final[tuple[tuple[str, str], ...]] = (
    ("gen_insample_medians.py", "comparavel"),
    ("gen_r_cv_prereg.py", "comparavel"),
    ("gen_loess_golden.py", "comparavel"),
    ("build_serrf_intermediates.py", "sem_ruido"),
)

# **Geradores que gravam `python_version` e NÃO têm `--check`.** Sem comparação não há o que
# afrouxar nem o que apertar. A entrada aqui é uma **afirmação verificável**, não uma isenção:
# `test_a_isencao_por_ausencia_de_check_continua_verdadeira` reprova se algum deles ganhar um
# `--check`, que é o dia em que ele passa a ser membro da classe.
SEM_CHECK: Final[frozenset[str]] = frozenset({"gen_serrf_dispersion.py"})

# Geradores que as fixtures congeladas citam e que saíram do repositório junto com o contêiner.
# A fixture continua a ser a referência; não há script a comparar.
SEM_GERADOR: Final[frozenset[str]] = frozenset({"gen_r_divergence.sh"})

_SCRIPT_NO_COMANDO = re.compile(r"(?P<nome>[\w.]+\.(?:py|sh))")


def _carrega(nome: str) -> Any:
    """Importa um gerador por caminho, **compilando a fonte do disco** e não o `__pycache__`.

    A forma natural — `spec_from_file_location` mais `exec_module`, como
    `tests/golden/test_serrf_transcription.py` faz — usa o `SourceFileLoader`, que valida
    o `.pyc` por **mtime e tamanho da fonte**. Isso é um buraco de referente para este portão, e
    ele foi medido e não suposto:

    > Ao exercitar o portão, a planta trocou `[:2]` por `[:3]` — **mesmo tamanho** —, e a
    > restauração por `git checkout` caiu no **mesmo segundo** que o `.pyc` gravado. O Python
    > considerou o cache válido e o teste passou a ler o bytecode **plantado** com a fonte já
    > restaurada: o portão reprovava sobre uma versão que não estava no disco.

    Com `compile` sobre o texto lido, o referente é inequívoco — **o arquivo como ele está
    agora**. É a caducidade em forma de cache: o mecanismo estava são sobre o alvo certo, e o
    alvo tinha envelhecido por baixo dele.
    """
    caminho = SCRIPTS / nome
    fonte = caminho.read_text(encoding="utf-8")
    modulo = ModuleType(f"_gerador_{caminho.stem}")
    modulo.__file__ = str(caminho)
    exec(compile(fonte, str(caminho), "exec"), modulo.__dict__)
    return modulo


def _comparador(gerador: str, funcao: str) -> Callable[[str], Any]:
    """A função de comparação do gerador, sempre recebendo TEXTO.

    O adaptador de `sem_ruido` faz o `json.loads` aqui, para que os três pares causais abaixo
    sejam escritos uma vez só e não três vezes por membro.
    """
    modulo = _carrega(gerador)
    alvo = getattr(modulo, funcao, None)
    assert alvo is not None, (
        f"{gerador} não expõe `{funcao}` no módulo. Enquanto a comparação vive dentro de "
        "`main`, nenhum teste a alcança e a única forma de a exercitar é correr o `--check` "
        "inteiro — que para dois destes exige os despejos do R."
    )
    if funcao == "sem_ruido":
        return lambda texto: alvo(json.loads(texto))
    return alvo


def _fixtures_com_versao() -> dict[str, str]:
    """`nome do fixture -> gerador declarado`, para todo fixture que grava `python_version`.

    **Derivado do campo `command` de cada fixture, e não de um mapa escrito aqui.** Um mapa à
    mão seria a segunda fonte de verdade sobre qual gerador produz qual artefato, e divergiria
    no dia em que um deles fosse renomeado.
    """
    achados: dict[str, str] = {}
    for caminho in sorted(FIXTURES.glob("*.json")):
        dados = json.loads(caminho.read_text(encoding="utf-8"))
        procedencia = dados.get("provenance")
        if not isinstance(procedencia, dict) or "python_version" not in procedencia:
            continue
        comando = str(procedencia.get("command", ""))
        nomes = _SCRIPT_NO_COMANDO.findall(comando)
        assert nomes, f"{caminho.name} grava python_version e não declara `command` legível"
        achados[caminho.name] = nomes[-1]
    return achados


# ---------------------------------------------------------------------------
# As âncoras de não-vacuidade — antes de qualquer asserção sobre a classe
# ---------------------------------------------------------------------------


def test_a_varredura_acha_os_fixtures_com_procedencia() -> None:
    """Conjunto vazio passaria em toda asserção sobre "todo membro da classe".

        A partição abaixo é uma asserção de ausência — *"nenhum fixture fora das duas listas"* —, e
        ela é satisfeita por vacuidade quando a varredura não acha nada. Esta vem antes por isso
    , e fica **na mesma função** que o número que ela protege.
    """
    achados = _fixtures_com_versao()

    assert len(achados) >= 6, f"a varredura achou {len(achados)} fixtures com procedência de Python"
    assert "insample_rsd_medians.json" in achados, sorted(achados)
    assert achados["insample_rsd_medians.json"] == "gen_insample_medians.py"

    # **Um gerador por fixture, e a colisão reprova em vez de ser resolvida pela ordem do glob.**
    # `texto_por_gerador` indexa pelo GERADOR, então dois fixtures do mesmo
    # gerador colapsariam num só: o segundo sobrescreve o primeiro e um deles deixa de ser
    # exercitado, em silêncio. Nenhum caso hoje; o que se prende é que reprove quando houver.
    valores = list(achados.values())
    duplicados = sorted({g for g in valores if valores.count(g) > 1})
    assert not duplicados, (
        f"mais de um fixture declara o mesmo gerador: {duplicados}. `texto_por_gerador` indexa "
        "por gerador, então um deles sairia da parametrização sem nada acusar — parametrize por "
        "FIXTURE antes de acrescentar o segundo"
    )


def test_todo_fixture_com_versao_cai_numa_das_duas_listas() -> None:
    """A partição, e é ela que faz o portão fechar a CLASSE e não os quatro itens.

    **Enumeração incompleta erra sempre para MAIS**: o que falta de uma partição vira residual,
    e residual lê-se como coberto. Gerador novo que grave `python_version` e compare o arquivo
    reprova aqui até alguém decidir de que lado ele fica.
    """
    membros = {nome for nome, _ in MEMBROS}
    achados = _fixtures_com_versao()

    # **A âncora fica NA MESMA FUNÇÃO, e a repetição é deliberada.** Há um teste vizinho que
    # afirma o mesmo, e é exactamente isso que a torna insuficiente aqui: `assert not fora` sobre
    # varredura vazia passa, e remover o vizinho não deixaria sinal nenhum neste. Este
    # arquivo nasceu com o defeito e ele foi achado por revisão, não por execução.
    assert achados, (
        "a varredura não achou fixture nenhum com `python_version` — sem isto, `fora` vazio "
        "significa 'não olhei' e não 'está tudo particionado'"
    )

    fora = {
        fixture: gerador
        for fixture, gerador in achados.items()
        if gerador not in membros and gerador not in SEM_CHECK and gerador not in SEM_GERADOR
    }

    assert not fora, (
        f"fixture grava `python_version` e o gerador não está em MEMBROS, SEM_CHECK nem "
        f"SEM_GERADOR: {fora}. Se ele compara o arquivo, normalize o patch e some-o a MEMBROS; "
        f"se não compara, declare-o em SEM_CHECK."
    )


def test_os_geradores_declarados_como_ausentes_nao_existem() -> None:
    """Gerador que voltar ao repositório sai de SEM_GERADOR e entra numa das outras listas."""
    assert SEM_GERADOR, "âncora: a lista está vazia e o teste não confere nada"
    presentes = [nome for nome in sorted(SEM_GERADOR) if (SCRIPTS / nome).exists()]
    assert not presentes, f"{presentes} existe(m) em scripts/ e continua(m) em SEM_GERADOR"


def test_a_isencao_por_ausencia_de_check_continua_verdadeira() -> None:
    """`SEM_CHECK` é afirmação sobre o código, e o código muda.

    Isenção que sobrevive ao que a justificava é o defeito que a direção B do portão de rotas
    existe para pegar. Aqui a justificativa é *"não há `--check`"*, e ela é conferível.
    """
    ganharam_check = [
        nome for nome in sorted(SEM_CHECK) if "--check" in (SCRIPTS / nome).read_text("utf-8")
    ]

    assert not ganharam_check, (
        f"{ganharam_check} ganhou(ram) `--check` e continua(m) em SEM_CHECK. A isenção era "
        "'não há comparação a afrouxar'; agora há, e o membro tem de normalizar o patch."
    )


# ---------------------------------------------------------------------------
# Os três pares causais, por membro
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def texto_por_gerador() -> dict[str, str]:
    """O texto do fixture REAL de cada membro, indexado pelo gerador que o produziu."""
    return {
        gerador: (FIXTURES / fixture).read_text(encoding="utf-8")
        for fixture, gerador in _fixtures_com_versao().items()
    }


@pytest.mark.parametrize(("gerador", "funcao"), MEMBROS, ids=[m[0] for m in MEMBROS])
def test_o_patch_do_cpython_nao_separa(
    gerador: str, funcao: str, texto_por_gerador: dict[str, str]
) -> None:
    """Mudar só o patch compara IGUAL — é o conserto, e sobre o fixture de verdade."""
    original = texto_por_gerador[gerador]
    comparar = _comparador(gerador, funcao)

    # Âncora: a substituição de facto aconteceu. Sem ela, os dois lados seriam o mesmo texto e
    # a igualdade passaria sem provar nada — o negativo que não toca a planta.
    mutado = original.replace('"python_version": "3.12.13"', '"python_version": "3.12.99"')
    assert mutado != original, "a planta do patch não pegou; o fixture mudou de formato?"

    assert comparar(original) == comparar(mutado), (
        "o patch do CPython ainda separa, e é ele que reprovava sem nenhum número mudar"
    )


@pytest.mark.parametrize(("gerador", "funcao"), MEMBROS, ids=[m[0] for m in MEMBROS])
def test_o_minor_do_cpython_separa(
    gerador: str, funcao: str, texto_por_gerador: dict[str, str]
) -> None:
    """Mudar o minor compara DIFERENTE — o conserto não pode virar "apague a linha"."""
    original = texto_por_gerador[gerador]
    comparar = _comparador(gerador, funcao)

    mutado = original.replace('"python_version": "3.12.13"', '"python_version": "3.11.13"')
    assert mutado != original, "a planta do minor não pegou"

    assert comparar(original) != comparar(mutado), (
        "3.11 e 3.12 comparam igual — a normalização apagou a linha em vez de a truncar, e o "
        "fixture deixou de afirmar em que interpretador foi produzido"
    )


@pytest.mark.parametrize(("gerador", "funcao"), MEMBROS, ids=[m[0] for m in MEMBROS])
def test_a_versao_do_numpy_continua_a_separar(
    gerador: str, funcao: str, texto_por_gerador: dict[str, str]
) -> None:
    """O resto da procedência segue comparado exatamente.

    **É o par que mede se a métrica melhorou ou parou de medir.** `numpy` está pinado exato em
    `pyproject.toml`, então divergência ali é mudança deliberada, e um `--check` que a deixasse
    passar teria trocado um portão por silêncio.
    """
    original = texto_por_gerador[gerador]
    comparar = _comparador(gerador, funcao)

    mutado = re.sub(r'("numpy_version": "\d+\.\d+\.)\d+"', r'\g<1>99"', original, count=1)
    assert mutado != original, f"{gerador} não grava `numpy_version`; a planta não pegou"

    assert comparar(original) != comparar(mutado), (
        "a versão do numpy deixou de separar — a normalização alcançou mais que o patch do CPython"
    )
