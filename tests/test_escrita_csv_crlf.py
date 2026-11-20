"""O portão que impede a suíte de voltar a quebrar no Windows.

## O defeito que ele fecha, e ele era invisível em Linux

`tests/factories.py` emite CSV com **CRLF de propósito** — a docstring de `render_measurements` diz
*"com terminador CRLF como os arquivos reais"*. `Path.write_text` sem `newline=""` abre o arquivo em
modo texto e traduz cada `\\n` para `os.linesep`: em Linux é `\\n`, e nada acontece; **no Windows é
`\\r\\n`, aplicado a um texto que já tem `\\r`**, e o que fica em disco é `\\r\\r\\n`.

Medido nesta máquina:

    texto = "a,b\\r\\nc,d\\r\\n"
    write_text(texto, encoding="utf-8")              -> b'a,b\\r\\r\\nc,d\\r\\r\\n'
    write_text(texto, encoding="utf-8", newline="")  -> b'a,b\\r\\nc,d\\r\\n'

O leitor então vê uma linha só, e a ingestão morre com `IndexError: list index out of range` em
`readers.py`, num `rows[1][0]` — erro que não nomeia a causa e aponta para o lugar errado.

**Nove dos ONZE sítios já passavam `newline=""`.** Faltavam dois, em
`tests/unit/test_serrf_step3_termination.py`, e o custo foi **quatro testes inteiros mortos** —
os do critério sobre terminação do passo 3 — invisíveis para quem corre a suíte em WSL, que é
onde este projeto foi desenvolvido.

> **A primeira versão desta frase dizia «sete dos nove», e o numerador e o denominador estavam
> ambos baixos.** Quem a corrigiu foi a revisão adversarial, medindo com o próprio oráculo deste
> arquivo. É um número escrito à mão ao lado de um mecanismo que sabe contar — e a
> resposta não é corrigi-lo, é **derivá-lo**, que é o que `test_a_populacao_do_portao_e_derivada`
> passa a fazer.

## O ORÁCULO FOI MEDIDO ANTES DE SER ESCOLHIDO

*"Este oráculo é a propriedade, ou é a forma que a propriedade tinha na instância em que eu a vi?"*
Duas versões, contadas sobre a árvore antes de decidir:

| Oráculo | Acusa hoje | Veredito |
|---|---|---|
| todo `write_text` sem `newline=""` | **51** | morto no primeiro dia |
| só o que escreve conteúdo vindo de `render_*` | **0** de **11**; eram 2 | é a propriedade |

O amplo acusa 51 porque a maioria escreve fonte Python, JSON e script R, onde a tradução de `\n`
é **correta** — um portão com 51 isenções é uma lista de isenção com outro nome. O estreito acusa
o que carrega CRLF **porque** o renderizador o põe.

A diferença entre os dois é a pergunta que separa portão de cerimônia, e os dois números custaram
dois minutos.

## Por que o negativo é EM PROCESSO

A planta é **argumento** da função sob teste (`sitios_sem_newline` recebe o texto-fonte), então a
entrada alimenta a asserção e não há intermediário que possa não a receber. Por triagem essa
forma **não consegue ficar inerte** — ao contrário de um negativo externo, que constrói arquivo
mutado e invoca subprocesso.
"""

from __future__ import annotations

import ast
from pathlib import Path

RAIZ_DOS_TESTES = Path(__file__).resolve().parent

RENDERIZADORES = frozenset({"render_measurements", "render_sample_list", "render_pair"})
"""As três funções de `tests/factories.py` cujo retorno carrega CRLF.

**Lista derivada e não adivinhada:** são as que terminam com `CRLF.join(...) + CRLF`. Um
renderizador novo que emita CRLF e não entre aqui deixa o portão cego — e é por isso que a
constante fica ao lado do portão, e não escondida numa função."""


def _nomes_vindos_de_render(arvore: ast.Module) -> set[str]:
    """As variáveis que recebem, algures no módulo, o retorno de um renderizador.

    Cobre as duas escritas naturais — `m = render_measurements(spec)` e
    `medidas, lista = render_pair(spec)` —, porque um portão que só casasse a primeira estaria a
    codificar a forma em vez da propriedade, que é exactamente o defeito.

    **Escopo de MÓDULO e não de função**, e a folga é deliberada: alarga o que o portão vê, nunca
    estreita. Uma variável chamada `medidas` numa função e escrita noutra continua a ser apanhada.
    """
    nomes: set[str] = set()
    for no in ast.walk(arvore):
        if not isinstance(no, ast.Assign):
            continue
        valor = no.value
        if not (
            isinstance(valor, ast.Call)
            and isinstance(valor.func, ast.Name)
            and valor.func.id in RENDERIZADORES
        ):
            continue
        for alvo in no.targets:
            if isinstance(alvo, ast.Name):
                nomes.add(alvo.id)
            elif isinstance(alvo, ast.Tuple):
                nomes.update(e.id for e in alvo.elts if isinstance(e, ast.Name))
    return nomes


def sitios_sem_newline(fonte: str) -> list[int]:
    """As linhas que escrevem CSV renderizado **sem** `newline=""`. Vazio é o estado bom.

    Devolve linhas e não um `bool`: *"há defeito"* e *"onde"* são perguntas
    diferentes, e um portão que só respondesse à primeira obrigaria quem o vê reprovar a procurar
    à mão.
    """
    arvore = ast.parse(fonte)
    de_render = _nomes_vindos_de_render(arvore)
    achados: list[int] = []
    for no in ast.walk(arvore):
        if not (
            isinstance(no, ast.Call)
            and isinstance(no.func, ast.Attribute)
            and no.func.attr == "write_text"
        ):
            continue
        if any(k.arg == "newline" for k in no.keywords) or not no.args:
            continue
        conteudo = no.args[0]
        veio_de_render = (
            isinstance(conteudo, ast.Call)
            and isinstance(conteudo.func, ast.Name)
            and conteudo.func.id in RENDERIZADORES
        ) or (isinstance(conteudo, ast.Name) and conteudo.id in de_render)
        if veio_de_render:
            achados.append(no.lineno)
    return achados


def test_nenhum_teste_escreve_csv_renderizado_sem_newline_vazio() -> None:
    """A direção positiva: a árvore está limpa.

    Sozinha ela é satisfeita por vacuidade — um `sitios_sem_newline` que devolvesse sempre `[]`
    passaria. É o teste de baixo que a torna significativa.
    """
    culpados: list[str] = []
    for arquivo in sorted(RAIZ_DOS_TESTES.rglob("*.py")):
        for linha in sitios_sem_newline(arquivo.read_text(encoding="utf-8")):
            culpados.append(f"{arquivo.relative_to(RAIZ_DOS_TESTES.parent)}:{linha}")

    assert not culpados, (
        'estes sítios escrevem CSV com CRLF sem `newline=""` e produzem `\\r\\r\\n` no Windows, '
        f"o que mata a ingestão com IndexError: {culpados}"
    )


def test_o_portao_acusa_a_planta_e_absolve_a_versao_corrigida() -> None:
    """**O par causal, e as duas metades são obrigatórias.**

    Duas fontes idênticas em tudo menos no `newline=""`. A primeira asserção prova que o portão
    **consegue** acusar; a segunda prova que ele **deixa de acusar** quando não há o que acusar —
    e é essa que separa portão de mecanismo que reprova sempre.
    """
    plantado = (
        "from tests.factories import render_pair\n"
        "def f(tmp_path, spec):\n"
        "    medidas, lista = render_pair(spec)\n"
        '    (tmp_path / "m.csv").write_text(medidas, encoding="utf-8")\n'
    )
    corrigido = plantado.replace('encoding="utf-8"', 'encoding="utf-8", newline=""')

    assert sitios_sem_newline(plantado) == [4], "o portão não viu a planta"
    assert sitios_sem_newline(corrigido) == [], "o portão acusa a versão correta"


def test_o_portao_ignora_escrita_que_nao_e_csv_renderizado() -> None:
    """A outra direção do falso positivo, e ela é o que mantém o oráculo estreito.

    Medido antes de escolher: o oráculo amplo — *todo* `write_text` sem `newline=""` — acusava
    **51** sítios desta árvore, quase todos legítimos: fonte Python plantada, JSON, script R. Um
    portão com 51 isenções não é portão. Este teste fixa a fronteira que o mantém em zero.
    """
    fonte = (
        "def f(tmp_path):\n"
        '    (tmp_path / "x.py").write_text("import os\\n", encoding="utf-8")\n'
        '    (tmp_path / "y.json").write_text(json.dumps({}), encoding="utf-8")\n'
    )

    assert sitios_sem_newline(fonte) == []


def test_a_escrita_por_chamada_direta_tambem_e_apanhada() -> None:
    """`write_text(render_measurements(spec))` sem variável intermediária.

    É a segunda escrita natural, e um portão que só casasse a variável seria cego a ela — o
    defeito exacto do funil do carimbo, onde quatro escritas naturais escapavam de
    um oráculo que codificava uma.
    """
    fonte = (
        "from tests.factories import render_measurements\n"
        "def f(tmp_path, spec):\n"
        '    (tmp_path / "m.csv").write_text(render_measurements(spec), encoding="utf-8")\n'
    )

    assert sitios_sem_newline(fonte) == [3]


def test_a_populacao_do_portao_e_derivada_e_nao_escrita() -> None:
    """**O denominador é contado, não afirmado** — a correção que a revisão exigiu.

    A docstring deste arquivo dizia *"sete dos nove sítios"*, e medido com o próprio oráculo daqui
    são **onze**. Um número escrito à mão ao lado de um mecanismo que sabe contar é o erro na
    forma mais evitável que existe.

    **A âncora não é o valor 11 e sim a ORDEM DE GRANDEZA da população**: um oráculo que passasse a
    reconhecer zero sítios faria `test_nenhum_teste_escreve_csv_renderizado_sem_newline_vazio`
    passar por vacuidade, e é isso que esta asserção impede. O número exato entra na mensagem para
    quem o vir mudar saber que mudou.
    """
    total = 0
    for arquivo in sorted(RAIZ_DOS_TESTES.rglob("*.py")):
        fonte = arquivo.read_text(encoding="utf-8")
        arvore = ast.parse(fonte)
        de_render = _nomes_vindos_de_render(arvore)
        for no in ast.walk(arvore):
            if not (
                isinstance(no, ast.Call)
                and isinstance(no.func, ast.Attribute)
                and no.func.attr == "write_text"
                and no.args
            ):
                continue
            conteudo = no.args[0]
            if (
                isinstance(conteudo, ast.Call)
                and isinstance(conteudo.func, ast.Name)
                and conteudo.func.id in RENDERIZADORES
            ) or (isinstance(conteudo, ast.Name) and conteudo.id in de_render):
                total += 1

    assert total >= 9, (
        f"o oráculo reconhece {total} sítios que escrevem CSV renderizado, e eram 11 em "
        "27/08/2026. Se caiu, ele deixou de ver uma forma de escrita e o portão de cima passa a "
        "ser satisfeito por vacuidade"
    )
