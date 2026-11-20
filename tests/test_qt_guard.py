"""A guarda do grupo `desktop` do aplicativo nativo.

**Este módulo NÃO pula, e a separação é o conteúdo.** Se os testes que provam que o `skip` está
correto vivessem em `test_desktop_janela.py`, eles seriam os primeiros a desaparecer na máquina sem
o grupo — exactamente onde a correção do `skip` importa. É circular: o mecanismo que valida o
desfecho não pode compartilhar o desfecho.

**A repartição entre a guarda e o sítio é medida e não estética.** `tests/skip_policy.py` extrai a
mensagem do `skip` por AST, colhendo constantes de string; `pytest.skip(variavel)` não entrega
literal nenhum, e o regime declarado reprova por *"padrão sem emissão é autorização larga que
ninguém exerce"*. A primeira versão desta guarda devolvia a frase e levou essa reprovação. Então:
`membros_ausentes` **decide**, o sítio **fraseia**, e este módulo cobre os dois lados.
"""

from __future__ import annotations

import ast
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from tests.conftest import REPO
from tests.qt_guard import (
    MEMBROS_EXIGIDOS,
    PLATAFORMA_SEM_TELA,
    garante_plataforma_offscreen,
    membros_ausentes,
    sonda_o_ambiente,
)
from tests.skip_policy import DECLARED_SKIP_REGIMES, _skip_messages

REGIME = next(r for r in DECLARED_SKIP_REGIMES if r.name == "grupo_desktop_ausente")
# **O sítio CANÔNICO, e a lista DERIVADA do regime.** Pinar a tupla à mão foi o que reprovou
# `make check` quando entrou o segundo arquivo de janela: a asserção media a lista de
# ontem, e o desfecho era vermelho por acréscimo legítimo. Derivar troca a pergunta de *"a tupla é
# esta?"* para *"todo sítio declarado existe, frase-a, e pula antes do primeiro import de
# Qt?"* — que é o que a âncora precisa que seja verdade, e é estritamente mais forte: antes
# só um arquivo tinha a existência e a frase conferidas.
SITIO = Path(__file__).with_name("test_desktop_janela.py")
SITIOS = tuple(Path(__file__).with_name(nome) for nome in REGIME.sites)


def _linha_do_skip(arvore: ast.Module) -> int | None:
    """A primeira chamada de `.skip` do módulo, por número de linha da AST.

    Por linha da AST e não por ordem de texto: um import dentro de função não conta, e é
    exactamente essa a distinção que uma varredura por linha perderia.
    """
    return min(
        (
            no.lineno
            for no in ast.walk(arvore)
            if isinstance(no, ast.Call)
            and isinstance(no.func, ast.Attribute)
            and no.func.attr == "skip"
        ),
        default=None,
    )


def test_o_regime_existe_e_nomeia_os_sitios() -> None:
    """Âncora. Sem ela, os testes abaixo passariam com o regime apagado da política.

    A não-vacuidade fica nesta função e antes das asserções que dependem dela: uma lista
    de sítios vazia faria os quatro laços estáticos abaixo passarem sem olhar arquivo nenhum.
    """
    assert REGIME.pattern, "o regime existe mas sem padrão — casaria qualquer motivo"
    assert SITIOS, "o regime não declara sítio nenhum — os laços abaixo mediriam o vazio"
    assert SITIO.name in REGIME.sites, (
        f"o sítio canônico saiu do regime: {REGIME.sites}. O par causal em subprocesso lá abaixo "
        "roda sobre ele, e sem ele no regime aquela medição fica sem referente"
    )
    for caminho in SITIOS:
        assert caminho.is_file(), f"o sítio declarado não existe: {caminho}"


# ---------------------------------------------------------------------------
# A DECISÃO — os três desfechos de `membros_ausentes`
# ---------------------------------------------------------------------------


def test_com_tudo_instalado_nao_falta_ninguem() -> None:
    """O desfecho vazio, e é o que esta máquina produz.

    **A planta é argumento**, então este caminho é o mesmo que a máquina real percorre — não é uma
    simulação dele.
    """
    presentes: dict[str, str | None] = dict.fromkeys(MEMBROS_EXIGIDOS, "6.11.2")
    assert membros_ausentes(presentes) == ()


def test_com_um_ausente_a_lista_nomeia_so_ele() -> None:
    """O segundo desfecho, forçado — e é o que esta máquina **não** produz, porque tem o grupo.

    A segunda asserção é a que importa: nomear quem **não** falta seria causa afirmada, que foi o
    defeito da primeira versão do regime `sem_endereco_externo`.
    """
    faltando: dict[str, str | None] = {"PySide6": None, "pytest-qt": "4.5.0"}
    assert membros_ausentes(faltando) == ("PySide6",)


def test_com_os_dois_ausentes_a_lista_nomeia_os_dois_em_ordem() -> None:
    """O terceiro desfecho. Três estados possíveis, três direções."""
    assert membros_ausentes(dict.fromkeys(MEMBROS_EXIGIDOS)) == tuple(sorted(MEMBROS_EXIGIDOS))


def test_a_sonda_do_ambiente_devolve_um_valor_por_membro() -> None:
    """A única função que toca o ambiente, e o que se afirma dela é a FORMA e não o conteúdo.

    Afirmar que PySide6 está instalado aqui tornaria o teste dependente de quem instalou o quê —
    e o valor desta sonda é justamente poder responder `None`.
    """
    assert set(sonda_o_ambiente()) == set(MEMBROS_EXIGIDOS)


def test_pyinstaller_nao_e_exigido_para_a_janela_correr() -> None:
    """O recorte, afirmado em vez de descrito.

    `pyinstaller` está no grupo `desktop` porque o congelamento precisa dele, e o congelamento não
    corre no WSL. Exigi-lo aqui faria os testes de janela pularem por falta de uma ferramenta que a
    janela não usa — `skip` com motivo verdadeiro e irrelevante, que é pior que nenhum.
    """
    assert "pyinstaller" not in MEMBROS_EXIGIDOS


# ---------------------------------------------------------------------------
# A FRASE — o literal do sítio contra o regime declarado
# ---------------------------------------------------------------------------


def test_a_frase_do_sitio_casa_o_padrao_do_regime() -> None:
    """O que a política de `skip` confere, conferido aqui também — e de propósito.

    A direção de runtime daquele portão só dispara **quando o `skip` acontece**, e nesta máquina ele
    não acontece: o grupo está instalado. Este teste lê o literal do sítio **estaticamente**, então
    uma frase alterada sem o padrão acompanhar reprova aqui, na máquina que nunca vê o `skip`.
    """
    for caminho in SITIOS:
        mensagens = _skip_messages(caminho.read_text(encoding="utf-8"))
        assert mensagens, (
            f"nenhum `pytest.skip` com literal em {caminho.name} — a asserção seria vácua"
        )

        casam = [m for m in mensagens if re.search(REGIME.pattern, m)]
        assert len(casam) == 1, (
            f"em {caminho.name}: esperava exactamente uma mensagem a casar "
            f"{REGIME.pattern!r}, achei {len(casam)} entre {mensagens!r}"
        )


def test_o_skip_do_sitio_vem_antes_do_primeiro_import_de_qt() -> None:
    """A propriedade que o comentário do sítio afirma, aqui afirmada por mecanismo.

    **Comentário não reprova nada.** Se `from PySide6 import …` subisse para o topo, a máquina sem o
    grupo levantaria `ModuleNotFoundError` na **coleta** — que é erro e não `skip`, não casa regime
    nenhum, e faria a política reprovar a sessão pelo motivo errado. O defeito só apareceria numa
    máquina que ninguém usa para desenvolver.

    Por número de linha da AST, e não por ordem de texto: um import dentro de função não conta, e é
    exactamente essa a distinção que uma varredura por linha perderia.
    """
    for caminho in SITIOS:
        arvore = ast.parse(caminho.read_text(encoding="utf-8"))
        linha_do_skip = _linha_do_skip(arvore)
        linhas_de_qt = [
            no.lineno
            for no in ast.walk(arvore)
            if isinstance(no, ast.ImportFrom) and (no.module or "").startswith("PySide6")
        ] + [
            no.lineno
            for no in ast.walk(arvore)
            if isinstance(no, ast.Import)
            for a in no.names
            if a.name.startswith("PySide6")
        ]

        # **Zero imports diretos de PySide6 é legítimo aqui**, e é o caso dos dois sítios de
        # hoje: eles importam o pacote da janela, que os faz. Por isso a não-vacuidade deste
        # teste NÃO é `linhas_de_qt` não vazia — é o teste irmão abaixo, que cobre o caminho
        # indireto e afirma o seu próprio conjunto não vazio. Sem ele, este passaria por
        # vacuidade e afirmaria cobertura larga.
        assert linha_do_skip is not None, (
            f"{caminho.name} não chama `pytest.skip` — o regime ficou órfão nele"
        )
        for linha in linhas_de_qt:
            assert linha_do_skip < linha, (
                f"em {caminho.name}: `pytest.skip` está na linha {linha_do_skip} e um import "
                f"de PySide6 na {linha}: numa máquina sem o grupo isto é erro de coleta, "
                "não `skip`"
            )


def test_o_import_do_pacote_da_janela_tambem_vem_depois_do_skip() -> None:
    """A irmã do teste acima, e sem ela ele cobre o degrau mais baixo.

    O sítio **não** importa `PySide6` direto: importa `refino.desktop.janela.principal`, que o
    importa. Verificar só o import literal de `PySide6` afirmaria, por omissão, que o caminho
    indireto está coberto.
    """
    for caminho in SITIOS:
        arvore = ast.parse(caminho.read_text(encoding="utf-8"))
        linha_do_skip = _linha_do_skip(arvore)
        assert linha_do_skip is not None, f"{caminho.name} não chama `pytest.skip`"

        indiretos = [
            no.lineno
            for no in ast.walk(arvore)
            if isinstance(no, ast.ImportFrom)
            and (no.module or "").startswith("refino.desktop.janela")
        ]
        assert indiretos, (
            f"{caminho.name} não importa o pacote da janela — este teste mediria o vazio nele"
        )
        for linha in indiretos:
            assert linha_do_skip < linha, (
                f"em {caminho.name}: `pytest.skip` na linha {linha_do_skip} e o import da "
                f"janela na {linha}: o segundo arrasta PySide6 e produz erro de coleta onde o "
                "grupo falta"
            )


def test_a_frase_do_sitio_diz_como_instalar() -> None:
    """`skip` que diz o que falta e não como resolver deixa o leitor no mesmo lugar.

    O grupo é opcional, então esta é a mensagem que alguém vai ler numa árvore limpa — é a primeira
    coisa que ela tem de responder.
    """
    for caminho in SITIOS:
        frase = " ".join(_skip_messages(caminho.read_text(encoding="utf-8")))
        assert frase, f"nenhum literal de `skip` em {caminho.name} — a asserção seria vácua"
        assert "[desktop]" in frase, f"a mensagem de {caminho.name} não nomeia o extra a instalar"


# ---------------------------------------------------------------------------
# O `skip` RODADO SEM O MARCADOR, com par causal em subprocesso
# ---------------------------------------------------------------------------


_PLUGIN_QUE_ESCONDE = '''\
"""Faz `PySide6` parecer NÃO INSTALADO, antes da coleta.

Não desinstala nada: substitui a consulta de metadados, que é exactamente o que
`tests/qt_guard.sonda_o_ambiente` usa. É o único jeito de exercitar o `skip` numa máquina que tem o
grupo instalado — e desinstalar 649 MB para o medir seria o custo que este arnês existe para evitar.
"""

import importlib.metadata as metadata

_original = metadata.version


def _finge(nome, *args, **kwargs):
    if nome == "PySide6":
        raise metadata.PackageNotFoundError(nome)
    return _original(nome, *args, **kwargs)


metadata.version = _finge
'''


def test_o_skip_dispara_de_verdade_quando_o_grupo_falta(tmp_path: Path) -> None:
    """O `skip` rodado SEM o marcador, que o resto deste módulo não faz.

    Os outros testes daqui medem a **decisão** por planta e a **frase** por AST. Nenhum deles vê o
    `skip` acontecer: nesta máquina o grupo está instalado, então o ramo nunca dispara por conta
    própria.

    ## Por que em subprocesso, e por que o par causal está na MESMA função

    É negativo **externo** — constrói um artefato e invoca um processo —, e a taxonomia diz que essa
    forma pode ficar inerte por dois caminhos: a invocação não alcançar a mutação, ou o desfecho ser
    o mesmo com e sem ela. As duas execuções ficam aqui juntas por isso: sem a segunda, um comando
    que pulasse por qualquer outro motivo — arquivo inexistente, erro de coleta — passaria por
    *"pulou, então a guarda funciona"*.

    Medido: com o plugin, `1 skipped` e a razão é a do regime; sem ele, `21 passed`.

    **ESCOPO: o sítio canônico, e só ele.** Os quatro testes estáticos acima percorrem todos os
    sítios do regime; este roda um subprocesso e fica em `test_desktop_janela.py`. A razão é que a
    propriedade medida — *a guarda emite `skip` quando o grupo falta* — é da **guarda**, e ela é a
    mesma função em todo sítio: duplicar o subprocesso dobraria o custo sem mudar o que se afirma.
    """
    plugin = tmp_path / "sem_pyside.py"
    plugin.write_text(_PLUGIN_QUE_ESCONDE, encoding="utf-8")

    def roda(*extra: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                *extra,
                str(SITIO),
                "-q",
                "--no-cov",
                "-p",
                "no:randomly",
                "-rs",
            ],
            cwd=REPO,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            # **`os.pathsep` e nao `:` literal.** O separador de `PYTHONPATH` e `;` no Windows,
            # e com o dois-pontos a variavel virava UMA entrada malformada: o plugin nao era
            # importavel e o subprocesso morria com `ImportError: No module named 'sem_pyside'`.
            # O teste reprovava com *"o modulo nao pulou com PySide6 escondido"* — uma frase sobre
            # a GUARDA quando o defeito era do arnes, e a linha de base registava-a como
            # reprovacao permanente.
            env={
                **os.environ,
                "PYTHONPATH": os.pathsep.join([str(tmp_path), os.environ.get("PYTHONPATH", "")]),
            },
        )

    escondido = roda("-p", "sem_pyside")
    visivel = roda()

    # Quebra não é desfecho, e sem o braço de controlo o par causal não se forma.
    # O sítio tem 256 testes de janela, e correr o arquivo INTEIRO provoca a corrupção de heap do
    # Qt que `docs/linha-de-base-windows.md` mede como taxa — o processo morre a meio, sem linha de
    # sumário. Ler isso como *"a guarda do Qt disparou sem a planta"* seria atribuir ao alvo um
    # veredito sobre o ambiente, e foi assim que esta reprovação entrou na linha de base.
    #
    # **A condição é estreita de propósito:** exige AUSÊNCIA de qualquer sumário do pytest. Uma
    # reprovação real imprime `failed`, um sucesso imprime `passed`, e os dois seguem para as
    # asserções abaixo — só o processo que morre sem dizer nada pula.
    if not re.search(r"\d+ (passed|failed|error)", visivel.stdout):
        pytest.skip(
            "sítio de janela QUEBROU no braço de controlo: o processo morreu sem sumário do "
            f"pytest (código {visivel.returncode}), então o par causal não se forma. É a quebra "
            "de heap do Qt que `docs/linha-de-base-windows.md` mede como taxa — quebra não é "
            "desfecho, e o mesmo arquivo passa inteiro quando corrido em fatias"
        )

    # A planta ATERROU: o módulo pulou, e pela razão que o regime declara.
    assert "1 skipped" in escondido.stdout, (
        f"o módulo não pulou com PySide6 escondido:\n{escondido.stdout[-2000:]}"
    )
    assert re.search(REGIME.pattern, escondido.stdout), (
        "pulou, mas por outro motivo que o regime declara — o marcador está a absorver outra "
        f"causa:\n{escondido.stdout[-2000:]}"
    )

    # E o par: sem a planta, o mesmo comando NÃO pula PELO REGIME DO QT. Sem isto, "pulou" não
    # prova nada.
    #
    # Isto dizia `"skipped" not in visivel.stdout`, e era proxy da forma e não da
    # propriedade. A propriedade é *"sem a planta, a guarda do Qt não dispara"*; o
    # proxy afirmava *"nada pula, por motivo nenhum"*, e as duas coincidiam só enquanto este sítio
    # tivesse um único regime capaz de disparar. Com `serrf/` fora do git, `test_desktop_janela.py`
    # passou a pular três testes por `reference_data_absent`, e o proxy acusou o legítimo — o portão
    # a exigir o defeito oposto.
    #
    # Casar o padrão do regime é mais estreito E mais correto: continua a reprovar se a guarda do Qt
    # pular com o grupo instalado, e deixa de opinar sobre regimes que não são o dela.
    assert not re.search(REGIME.pattern, visivel.stdout), (
        "o módulo pula pelo regime do grupo desktop mesmo com o grupo instalado — o `skip` não "
        f"mede a ausência do grupo, mede outra coisa:\n{visivel.stdout[-2000:]}"
    )
    assert " passed" in visivel.stdout, (
        f"sem a planta o módulo não correu — o par não separa:\n{visivel.stdout[-2000:]}"
    )


# ---------------------------------------------------------------------------
# A plataforma sem tela
# ---------------------------------------------------------------------------


def test_a_plataforma_offscreen_fica_valendo_e_a_funcao_devolve_o_efetivo() -> None:
    """`setdefault` e o valor devolvido, para o chamador poder afirmar sem ler o ambiente por fora.

    **Não afirma que o resultado é `offscreen`**: quem tem servidor gráfico pode ter exportado outra
    coisa para ver a janela, e essa é a razão de ser `setdefault`. O que se afirma é que **há** uma
    plataforma fixada — que é a propriedade sem a qual o Qt aborta.
    """
    efetiva = garante_plataforma_offscreen()
    assert efetiva, "nenhuma plataforma ficou valendo — o Qt abortaria sem display"
    assert garante_plataforma_offscreen() == efetiva, (
        "a segunda chamada mudou o que a primeira fixou"
    )


def test_a_plataforma_padrao_e_offscreen_e_nao_minimal() -> None:
    """A escolha entre as duas é medida: `minimal` não entrega evento de rato.

    `qtbot.mouseClick` é metade do que os testes de janela exercitam, então `minimal` tornaria os
    testes de painel impossíveis — e o defeito apareceria como clique que não faz nada, não como
    erro.
    """
    assert PLATAFORMA_SEM_TELA == "offscreen"
