"""Todo carimbo de disparo do decorrido passa pelo funil, em cada arquivo que toca o depósito.

O decorrido congelado de uma corrida anterior não pode sobreviver a um disparo novo, e por isso
toda escrita no depósito do decorrido acontece em `carimba_disparo` (o funil) ou numa função
declarada como legítima. O portão percorre por AST os arquivos de `src/` que importam
`refino.interface.decorrido`, confere que cada um tem dialeto declarado e planta escritas por
fora para ver cada uma ser acusada.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

from tests.conftest import REPO

DECORRIDO = REPO / "src" / "refino" / "interface" / "decorrido.py"
PIPELINE = REPO / "src" / "refino" / "desktop" / "janela" / "pipeline.py"
PRINCIPAL_DA_JANELA = REPO / "src" / "refino" / "desktop" / "janela" / "principal.py"
ENRIQUECIMENTO_DA_JANELA = REPO / "src" / "refino" / "desktop" / "janela" / "enriquecimento.py"


@dataclass(frozen=True)
class _Dialeto:
    """Como um arquivo escreve no depósito do decorrido, e quais funções dele podem escrever.

    ## O portão é FAIL-CLOSED, e a mudança é de uma revisão adversarial

    A primeira versão discriminava carimbo de disparo pela substring `inicio` no texto da chave.
    Medido com plantas em memória, **quatro escritas naturais escapavam**:

    | Planta | Por que escapava |
    |---|---|
    | uma **segunda** constante, `CHAVE_DO_DISPARO = "inicio_multiverso"`, escrita crua | o
    detector lia o **identificador**, não o valor, e `"inicio" not in "chave_do_disparo"` |
    | `k = CHAVE_DO_INICIO` e depois `deposito.guarda(k, …)` | o identificador é `k` |
    | `deposito.guarda(self._chave_do_inicio, …)` | `ast.Attribute` devolvia texto vazio |
    | `self._inicio = agora()` sem depósito nenhum | não há chamada de escrita a casar |

    A terceira é a mais provável de nascer, e a primeira é a mais aguda: renomear a única chave
    era apanhado pela âncora, mas acrescentar uma segunda não era, porque a âncora só exige que
    alguma chave seja encontrável.

    > Então o portão deixou de olhar a chave. Ele acusa toda chamada ao escritor do depósito
    > que não esteja numa função da lista `legitimas`, e a lista é curta, explícita, e cada membro
    > tem motivo escrito. Uma heurística de nome depende de uma convenção que nada impõe; uma lista
    > de permissão falha para o lado seguro.

    ## O conjunto de arquivos é DERIVADO, e a âncora anterior não o cobria

    `assert len(_DIALETOS) == 3` restringe a tabela, não o disco. Um
    `desktop/janela/multiverso.py` novo que importasse `interface/decorrido` e escrevesse cru
    deixava a contagem em 3, os três caminhos intocados, e o portão verde.

    O conserto é `_arquivos_com_deposito()`, que varre `src/` por quem importa o módulo, e a
    bijeção contra a tabela. O padrão vem do irmão que já existia neste repositório:
    `FONTES_DA_JANELA` em `tests/test_desktop_janela.py` é um `glob` com âncora de não-vacuidade.
    """

    rotulo: str
    caminho: Path
    escritor: str
    por_atributo: bool
    legitimas: frozenset[str]

    # **`sem_chave_propria` é DECLARAÇÃO e não isenção, e o portão a confere nas duas direções.**
    #
    # O desfecho 2 planta a chave **do próprio arquivo** fora do funil, e existe para impedir que o
    # desfecho 1 seja vácuo: um extractor que devolvesse vazio produziria uma planta com chave vazia
    # e o desfecho reprovaria pelo motivo errado. Mas um arquivo pode importar este módulo **sem
    # participar da maquinaria de chaves** — `desktop/janela/principal.py` importa só `agora()`, o
    # seam do relógio, e não toca depósito nenhum. Ali não há chave a plantar.
    #
    # Saltar em silêncio seria isenção. Com o campo, o portão exige **coerência**: quem declara ter
    # chave e não tem reprova (era o comportamento), e quem declara não ter e **tem** também reprova
    # (é novo). Os desfechos 1, 3 e 4 correm para os dois casos — e o 1 é o que importa aqui,
    # porque com `legitimas` vazia ele afirma que **nada** naquele arquivo escreve no depósito.
    sem_chave_propria: bool = False


_DIALETOS = (
    _Dialeto(
        rotulo="interface/decorrido.py",
        caminho=DECORRIDO,
        escritor="guarda",
        por_atributo=True,
        # Duas escritoras legítimas, e a segunda é o motivo de a lista existir:
        #   `carimba_disparo`      — o funil;
        #   `decorrido_da_leitura` — grava o **congelado**, que é o oposto de um carimbo de disparo.
        legitimas=frozenset({"carimba_disparo", "decorrido_da_leitura"}),
    ),
    _Dialeto(
        rotulo="desktop/janela/pipeline.py",
        caminho=PIPELINE,
        escritor="guarda",
        por_atributo=True,
        # **Nenhuma.** Este arquivo é consumidor: ele chama `carimba_disparo` e nunca escreve no
        # depósito. Lista vazia é a política mais forte que existe, e é a correta aqui.
        legitimas=frozenset(),
    ),
    _Dialeto(
        rotulo="desktop/janela/principal.py",
        caminho=PRINCIPAL_DA_JANELA,
        escritor="guarda",
        por_atributo=True,
        # **Nenhuma, e por uma razão diferente da do `pipeline.py`.** Aquele chama
        # `carimba_disparo`; este não toca depósito nenhum. Ele importa só `agora()`, o seam do
        # relógio, e guarda o instante no próprio registro da espera do artefato,
        # `_EsperaDoArtefato.inicio`.
        #
        # A bijeção deste portão foi o que exigiu esta entrada. Ele reprovou por acréscimo
        # legítimo, que é o desfecho correto de um portão derivado: `principal.py` apareceu no disco
        # e não na tabela.
        #
        # **Por que o seam e não `time.monotonic()` direto:** o teste do decorrido da desistência
        # substitui a sequência de instantes com `monkeypatch.setattr(…principal.agora, …)`, que é
        # exactamente o uso do seam. Chamar `time.monotonic()` aqui obrigaria a
        # monkeypatchar o módulo `time` inteiro, e criaria a segunda fonte do relógio.
        legitimas=frozenset(),
        # **Nenhuma chave de disparo, e é isto que o desfecho 2 confere.** Este arquivo não escreve
        # no depósito, então não tem chave própria a plantar. A declaração é verificada nas duas
        # direções: se alguém acrescentar uma chave aqui, o portão reprova por incoerência.
        sem_chave_propria=True,
    ),
    _Dialeto(
        rotulo="desktop/janela/enriquecimento.py",
        caminho=ENRIQUECIMENTO_DA_JANELA,
        escritor="guarda",
        por_atributo=True,
        # O caso do `principal.py`, e entrou pela mesma bijeção. O painel do
        # enriquecimento importa `agora()` para o decorrido da espera do R e `instante_utc()` para
        # carimbar o CSV, os dois por injeção; não toca depósito nenhum nem tem chave de disparo.
        legitimas=frozenset(),
        sem_chave_propria=True,
    ),
)

# Os prefixos de uma chave de disparo. A forma `f`-string, `f"inicio_{chave}"`, é coberta pelo
# prefixo.
PREFIXOS_DE_DISPARO: tuple[str, ...] = ("inicio", "multiverso_inicio")


def _texto_da_chave(no: ast.expr) -> str:
    """O texto de uma chave de depósito: literal, `f`-string, nome de variável, ou atributo.

    Cobre o literal e a `f`-string, mais nome e atributo, que a revisão adversarial mediu como
    escapes.
    """
    if isinstance(no, ast.Constant) and isinstance(no.value, str):
        return no.value
    if isinstance(no, ast.JoinedStr):
        return "".join(
            p.value for p in no.values if isinstance(p, ast.Constant) and isinstance(p.value, str)
        )
    if isinstance(no, ast.Name):
        return no.id
    if isinstance(no, ast.Attribute):
        return no.attr
    return ""


def _e_chave_de_disparo(no: ast.expr) -> bool:
    """Se esta expressão nomeia uma chave de carimbo de disparo. **Insensível a maiúsculas.**"""
    texto = _texto_da_chave(no).lower()
    return any(p in texto for p in PREFIXOS_DE_DISPARO)


def _casa_o_escritor(func: ast.expr, dialeto: _Dialeto) -> bool:
    """Se este nó de chamada é a operação de escrita do dialeto — atributo ou nome."""
    if dialeto.por_atributo:
        return isinstance(func, ast.Attribute) and func.attr == dialeto.escritor
    return isinstance(func, ast.Name) and func.id == dialeto.escritor


def _escopos_por_no(arvore: ast.AST) -> dict[int, str]:
    """Para cada nó, o nome da função que o envolve — ou `"<modulo>"`.

    **Percorre o módulo INTEIRO, e é isso que fecha quatro escapes medidos.** A versão anterior
    filtrava `ast.FunctionDef` e descia só nele, então escrita ao nível de módulo, em corpo de
    classe, dentro de `lambda` de módulo, ou num `async def` era **invisível** — as quatro medidas
    pela revisão adversarial.

    O escopo é o `def` mais **próximo**, e uma função aninhada dentro do funil conta como o
    aninhado — não como o funil. É deliberado: quem escreve uma função nova dentro do funil não
    herda a permissão dele.
    """
    escopos: dict[int, str] = {}

    def _desce(no: ast.AST, atual: str) -> None:
        for filho in ast.iter_child_nodes(no):
            proximo = (
                filho.name
                if isinstance(filho, ast.FunctionDef | ast.AsyncFunctionDef)
                else "<lambda>"
                if isinstance(filho, ast.Lambda)
                else atual
            )
            escopos[id(filho)] = proximo
            _desce(filho, proximo)

    escopos[id(arvore)] = "<modulo>"
    _desce(arvore, "<modulo>")
    return escopos


def _carimbos_fora_do_funil(fonte: str, dialeto: _Dialeto) -> list[str]:
    """As escritas no depósito que NÃO estão numa função legítima do dialeto, por AST.

    Recebe a **fonte** e não o caminho, de propósito: é o que torna o negativo *em processo* —
    a planta é argumento da função sob teste, então não há intermediário que possa não recebê-la,
    e a forma não pode ficar inerte.

    **Duas formas de escrita são detectadas**, e a segunda entrou pela revisão adversarial:

    - **chamada** ao escritor — `deposito.guarda(...)`;
    - **subscrito** — `self._estado[CHAVE] = valor`, que não é chamada nenhuma. A própria docstring
      de `pipeline.py` avisa contra um dicionário local, e um autor que fizesse exactamente o que
      ela proíbe não era acusado.

    **`legitimas` vazio significa «este arquivo não escreve no depósito»**, e é a política de
    `pipeline.py`.
    """
    arvore = ast.parse(fonte)
    escopos = _escopos_por_no(arvore)
    fora: list[str] = []

    for no in ast.walk(arvore):
        escopo = escopos.get(id(no), "<modulo>")

        if isinstance(no, ast.Call) and _casa_o_escritor(no.func, dialeto):
            # Nos dialetos de atributo o portão não olha a chave: acusa toda chamada fora de
            # `legitimas`.
            if dialeto.por_atributo:
                if escopo not in dialeto.legitimas:
                    fora.append(f"{escopo}:{no.lineno}")
                continue
            if escopo in dialeto.legitimas:
                continue
            alvos = [a for i, a in enumerate(no.args) if i >= 1]
            if any(_e_chave_de_disparo(a) for a in alvos):
                fora.append(f"{escopo}:{no.lineno}")
            continue

        # O braço do subscrito: `<algo>[<chave de disparo>] = <valor>`.
        if isinstance(no, ast.Assign):
            for alvo in no.targets:
                if (
                    isinstance(alvo, ast.Subscript)
                    and _e_chave_de_disparo(alvo.slice)
                    and escopo not in dialeto.legitimas
                ):
                    fora.append(f"{escopo}:{no.lineno}")

    return sorted(set(fora))


def _escrita_crua(dialeto: _Dialeto, chave: str) -> str:
    """Uma escrita de depósito na forma do dialeto, para plantar. **A forma vem da tabela.**

    Escrevê-la à mão em cada desfecho duplicaria a decisão de qual é a forma de cada arquivo, e
    é essa duplicação que deixa a planta de um dialeto calibrada no outro.
    """
    if dialeto.por_atributo:
        return f"deposito.{dialeto.escritor}({chave}, 1.0)"
    return f"{dialeto.escritor}(st, {chave}, 1.0)"


def _indice_da_chave(dialeto: _Dialeto) -> int:
    """Em que argumento da escrita a chave vive. **DERIVADO da forma, não declarado.**

    `deposito.guarda(chave, valor)` põe-na no **0**; `_guarda(st, chave, valor)` no **1**. A versão
    anterior da tabela carregava dois campos de índice, e dois campos que se computam de
    `por_atributo` são duas fontes de verdade sobre a mesma coisa — a classe que este arquivo passa
    a unidade inteira a fechar.
    """
    return 0 if dialeto.por_atributo else 1


def _chave_real_do_dialeto(fonte: str, dialeto: _Dialeto) -> str:
    """A expressão de chave de disparo VERBATIM que este arquivo usa, como texto de fonte.

    **É a âncora de não-vacuidade, e ela vem do arquivo e não do meu teclado.** Uma chave
    redigitada aqui provaria que o detector casa o que eu escrevi; a chave tirada da fonte prova
    que ele casa o que o arquivo tem, e muda com o arquivo no dia em que a forma mudar.

    Dois lugares onde procurar, porque os arquivos têm o carimbo em posições diferentes:

    | Arquivo | Onde está a chave real | Por quê |
    |---|---|---|
    | `interface/decorrido.py` | na **escrita dentro do funil** | é ele o funil |
    | `pipeline.py` | nas **chamadas ao funil** | é consumidor: importa e chama |

    Procurar só no primeiro lugar deixaria `interface/decorrido.py` sem âncora; só no segundo
    deixaria `pipeline.py` sem âncora.

    **A chamada ao funil põe a chave no argumento 1** — `carimba_disparo(deposito, chave, instante)`
    —, e a escrita crua no índice que `_indice_da_chave` deriva.
    """
    achados: list[tuple[int, str]] = []
    arvore = ast.parse(fonte)
    escopos = _escopos_por_no(arvore)

    def _chave_do_no(no: ast.Call, indice: int) -> str | None:
        if len(no.args) <= indice:
            return None
        alvo = no.args[indice]
        if not _e_chave_de_disparo(alvo):
            return None
        return ast.get_source_segment(fonte, alvo)

    for no in ast.walk(arvore):
        if not isinstance(no, ast.Call):
            continue
        alvo = no.func
        chamado = (
            alvo.id
            if isinstance(alvo, ast.Name)
            else alvo.attr
            if isinstance(alvo, ast.Attribute)
            else ""
        )
        # (a) chamada ao FUNIL, em qualquer lugar do arquivo. O nome é o mesmo nos três, com ou sem
        # sublinhado à frente.
        if chamado.lstrip("_") == "carimba_disparo" and (texto := _chave_do_no(no, 1)):
            achados.append((no.lineno, texto))
            continue
        # (b) a escrita crua dentro de uma função LEGÍTIMA — é lá que `decorrido.py` a tem.
        if (
            _casa_o_escritor(alvo, dialeto)
            and escopos.get(id(no), "<modulo>") in dialeto.legitimas
            and (texto := _chave_do_no(no, _indice_da_chave(dialeto)))
        ):
            achados.append((no.lineno, texto))

    return sorted(achados)[0][1] if achados else ""


def _afirma_a_regressao_verbatim(fonte: str, dialeto: _Dialeto, chave: str, rotulo: str) -> None:
    """Desfecho 2: a chave REAL do arquivo, plantada fora do funil, tem de ser acusada.

    Função e não bloco inline porque o percurso tem dois ramos: um arquivo
    que declara `sem_chave_propria` não tem chave própria a plantar. Extrair torna o salto uma
    chamada que não acontece, em vez de um `if` a envolver vinte linhas.

    É este desfecho que impede o desfecho 1 de ser vácuo: sem ele, um detector que não casasse a
    forma de escrita **daquele arquivo** deixaria a árvore limpa a passar por não achar nada.
    """
    regressao = (
        f"{fonte}\n\ndef _regressao_verbatim(*a: object, **k: object) -> None:\n"
        f"    {_escrita_crua(dialeto, chave)}\n"
    )
    assert regressao != fonte, "a planta da regressão NÃO aterrou"
    assert any(
        a.startswith("_regressao_verbatim:") for a in _carimbos_fora_do_funil(regressao, dialeto)
    ), (
        f"{rotulo}: o detector não acusou a chave REAL deste arquivo escrita por fora do "
        f"funil — ele não casa a forma daqui, e a asserção da árvore limpa é vácua. "
        f"Chave: {chave}"
    )


def _arquivos_com_deposito() -> set[Path]:
    """Os arquivos de `src/` que importam `interface/decorrido`, o conjunto que a tabela cobre.

    **DERIVADO do disco e não escrito à mão, e a razão é medida.** A âncora anterior era
    `len(_DIALETOS) == 3`, que restringe a **tabela**: um arquivo novo importando o depósito e
    escrevendo cru deixava a contagem correta e o portão cego. O irmão que já fazia isto certo neste
    repositório é `FONTES_DA_JANELA`, um `glob` com âncora.

    O próprio `interface/decorrido.py` entra: ele **define** o depósito e é onde o funil vive.

    ## A varredura é por AST e não por substring

    Substring perde a forma `from refino.interface import decorrido`, em que a string com ponto
    nunca aparece. E erra na direção oposta também: qualquer arquivo cuja prosa mencionasse o
    módulo entraria no conjunto. `ast` casa importação, e nada mais.
    """

    def _importa_o_deposito(caminho: Path) -> bool:
        try:
            arvore = ast.parse(caminho.read_text(encoding="utf-8"))
        except SyntaxError:  # pragma: no cover - não há arquivo assim em `src/`
            return False
        for no in ast.walk(arvore):
            if isinstance(no, ast.ImportFrom):
                # `from refino.interface.decorrido import …` e
                # `from refino.interface import decorrido`
                if (no.module or "").endswith("interface.decorrido"):
                    return True
                if (no.module or "").endswith("refino.interface") and any(
                    a.name == "decorrido" for a in no.names
                ):
                    return True
            elif isinstance(no, ast.Import) and any(
                a.name.endswith("interface.decorrido") for a in no.names
            ):
                return True
        return False

    raiz = REPO / "src"
    achados = {caminho for caminho in raiz.rglob("*.py") if _importa_o_deposito(caminho)}
    return achados | {DECORRIDO}


def test_todo_carimbo_de_disparo_passa_pelo_funil_de_cada_arquivo() -> None:
    """O portão fail-closed do funil, sobre um conjunto derivado.

    O funil torna um quarto lugar de escrita fácil de evitar, não impossível. Este é o mecanismo
    que fecha a distância.

    **Quatro desfechos por arquivo**, porque dois é o piso e não o número:

    | Desfecho | Como é produzido |
    |---|---|
    | **nada fora do funil** | a fonte real do arquivo |
    | **acusa a chave REAL** posta fora do funil | planta com a chave extraída do próprio arquivo |
    | **acusa uma escrita nova** | planta: função que escreve no depósito por fora |
    | **acusa fora de `def`** | planta ao nível de **módulo** — a forma que a versão anterior não
    via |

    ## O que a revisão adversarial derrubou, e o que ficou no lugar

    Ela executou dez plantas e mediu **oito** escapes: quatro pela heurística de substring — segunda
    constante, chave por variável local, chave por atributo, e um par de atributos sem depósito — e
    quatro pela forma de AST — nível de módulo, `lambda` de módulo, corpo de classe, e `async def`.
    Mais um nono, `self._estado[CHAVE] = 1.0`, que não é chamada nenhuma.

    O conserto tem três partes, e cada uma fecha um grupo:

    1. **fail-closed por FUNÇÃO** em vez de por chave, para os dialetos de atributo. `legitimas` é
       uma lista de permissão curta com motivo escrito por membro. Fecha os quatro primeiros.
    2. **percurso do módulo inteiro** com o `def` envolvente rastreado, aceitando
       `AsyncFunctionDef`, `Lambda`, corpo de classe e nível de módulo. Fecha os quatro seguintes.
    3. **braço de `Subscript`**, para a atribuição que não é chamada. Fecha o nono.

    E a quarta parte é sobre o **conjunto de arquivos**: `_arquivos_com_deposito()` varre `src/` por
    quem importa `refino.interface.decorrido` e a bijeção contra a tabela reprova quando nasce um
    arquivo novo. A âncora de contagem anterior restringia a tabela, não o disco.

    Todas as plantas afirmam que aterraram antes de o desfecho ser lido.
    """
    # ── a bijeção do CONJUNTO DE ARQUIVOS, e ela vem antes de tudo ──
    no_disco = _arquivos_com_deposito()
    na_tabela = {d.caminho for d in _DIALETOS}
    assert no_disco, "âncora: a varredura não achou arquivo nenhum a importar `ui/decorrido`"
    assert no_disco == na_tabela, (
        f"o conjunto de arquivos que tocam o depósito do decorrido divergiu da tabela de dialetos. "
        f"Só no disco: {sorted(p.name for p in no_disco - na_tabela)}. "
        f"Só na tabela: {sorted(p.name for p in na_tabela - no_disco)}. "
        f"Arquivo novo que importe `ui/decorrido` precisa de um `_Dialeto`, senão a árvore limpa "
        f"dele nunca é conferida"
    )

    for dialeto in _DIALETOS:
        fonte = dialeto.caminho.read_text(encoding="utf-8")
        rotulo = dialeto.rotulo

        # ── desfecho 1: a árvore limpa ──
        assert _carimbos_fora_do_funil(fonte, dialeto) == [], (
            f"{rotulo}: estes lugares escrevem no depósito do decorrido fora de "
            f"{sorted(dialeto.legitimas) or 'qualquer função legítima'}, e neles o decorrido "
            f"congelado da corrida anterior sobrevive"
        )

        # ── desfecho 2: a chave REAL do arquivo, posta fora do funil ──
        # A asserção da chave é o que impede a âncora de ser vácua: sem ela, um extrator que
        # devolvesse vazio produziria uma planta com chave vazia, e o desfecho reprovaria por um
        # motivo que não é o que se está a medir.
        chave = _chave_real_do_dialeto(fonte, dialeto)
        if dialeto.sem_chave_propria:
            # **A direção NOVA da declaração.** Quem declara não ter chave e tem, reprova: a
            # declaração deixaria de ser verdade e o desfecho 2 continuaria saltado em silêncio.
            assert not chave, (
                f"{rotulo}: o dialeto declara `sem_chave_propria` e há uma chave de disparo no "
                f"arquivo — {chave!r}. Declaração e disco divergem, e o desfecho 2 está a ser "
                f"saltado sobre um arquivo que já participa da maquinaria de chaves"
            )
            # Os desfechos 3 e 4 abaixo não dependem da chave do arquivo: eles plantam um literal.
            # Então o percurso continua, e só a regressão verbatim é que não se aplica.
        else:
            assert chave, (
                f"{rotulo}: `_chave_real_do_dialeto` não achou chave de disparo nenhuma — o "
                f"dialeto está errado, e o desfecho 1 seria vácuo"
            )
            _afirma_a_regressao_verbatim(fonte, dialeto, chave, rotulo)

        # ── desfecho 3: uma escrita nova com chave literal que NÃO vem do arquivo ──
        quarto = (
            f"{fonte}\n\ndef _quarto_lugar(*a: object, **k: object) -> None:\n"
            f"    {_escrita_crua(dialeto, chr(34) + 'inicio_novo' + chr(34))}\n"
        )
        assert quarto != fonte, "a planta do quarto lugar NÃO aterrou"
        acusados = _carimbos_fora_do_funil(quarto, dialeto)
        assert any(a.startswith("_quarto_lugar:") for a in acusados), (
            f"{rotulo}: o quarto lugar não foi acusado: {acusados}"
        )

        # ── desfecho 4: FORA de qualquer `def` — a forma que a versão anterior não via ──
        no_modulo = f"{fonte}\n\n{_escrita_crua(dialeto, chr(34) + 'inicio_novo' + chr(34))}\n"
        assert no_modulo != fonte, "a planta de nível de módulo NÃO aterrou"
        acusados_m = _carimbos_fora_do_funil(no_modulo, dialeto)
        assert any(a.startswith("<modulo>:") for a in acusados_m), (
            f"{rotulo}: escrita ao nível de MÓDULO não foi acusada: {acusados_m}. A versão "
            f"anterior deste portão descia só em `ast.FunctionDef` e era cega a isto"
        )
