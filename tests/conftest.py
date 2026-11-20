"""Resolvedor dos dados de referência.

Por que a resolução vive aqui e não em `src/refino/`. Duas razões
independentes, ambas suficientes:

1. Os nomes em disco divergem dos da especificação. O diretório real é
   ``{modo}_toserrf - normalization result/`` com espaços, subdiretório
   ``normalized datasets/``, arquivos ``normalized by - SERRF.csv`` e
   ``QC - RSD.csv``. A especificação os declara achatados com underscores. Nenhuma
   fase deve hardcodear nenhuma das duas formas.
2. Qualquer código em ``src/refino/`` quebraria o invariante de zero
   statements e reprovaria o portão de cobertura. ``tests/`` está
   fora de ``--cov=src/refino``, então o resolvedor não carrega dívida.

Regra: arquivo de referência ausente produz ``skip`` com mensagem
clara — nunca dado fabricado, nunca erro genérico.
"""

import functools
import hashlib
import subprocess
import sys
import threading
import tomllib
import warnings
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    # pragma: no cover - só para anotação; evita importar domínio na coleta
    import numpy as np

    from refino.domain.matrix import RefinoMatrix

REPO = Path(__file__).resolve().parent.parent
MANIFEST = REPO / "scripts" / "reference_data.sha256"


# **Quanto a teardown espera por uma thread de trabalho que sobrou.** Dez segundos: o cancelamento
# age no ponto de verificação seguinte, e o mais espaçado deles é uma feature do SERRF —
# 362,6 ms medidos. Dez segundos são quase trinta vezes isso, e um teto maior transformaria a
# limpeza num custo que ninguém notaria a acumular.
TETO_DA_JUNCAO_S = 10.0


def encerra_trabalhos_vivos(modulo: Any, *, teto_s: float = TETO_DA_JUNCAO_S) -> tuple[str, ...]:
    """Cancela, junta e esquece os trabalhos ainda vivos. Devolve os identificadores que tratou.

    **Extraída da fixture em 07/09/2026 para que a PLANTA caia sobre ela, e é o mesmo
    conserto que `tests/unit/test_identidade_do_corretor.py` levou no mesmo dia.** Enquanto o corpo
    vivia dentro de um `yield`, nenhum teste conseguia alcançá-lo: uma fixture de teardown não é
    chamável, e um teste não afirma sobre a própria teardown. O mecanismo ficou **construído e não
    exercitado** — medido com sonda, zero threads encontradas em dois arquivos inteiros —, e a
    sessão que o escreveu chegou a atribuir-lhe um conserto que ele não fez.

    **Devolve os identificadores e não `None`**, pelo motivo de `pre_preenche` devolver `bool`: o
    arnês afirma **que** o encerramento aconteceu, em vez de o inferir do estado do registro.

    **`teto_s` é parâmetro e não a constante**, e é isso que torna o ramo do aviso testável: um
    teto de fração de segundo faz uma thread que ignora o cancelamento ser apanhada sem o teste
    esperar dez segundos.
    """
    prefixo = modulo.PREFIXO_DA_THREAD_DE_TRABALHO
    vivas = [t for t in threading.enumerate() if t.name.startswith(prefixo) and t.is_alive()]
    if not vivas:
        return ()

    identificadores = tuple(thread.name[len(prefixo) :] for thread in vivas)
    for identificador in identificadores:
        modulo.TRABALHOS.pede_cancelamento(identificador)
    for thread in vivas:
        thread.join(timeout=teto_s)

    # **E ESQUECER, que e a metade sem a qual esta limpeza abriria um modo de falha novo.**
    #
    # Um trabalho que nao parou dentro do teto fica no registro em `pendente`, sem thread viva a
    # move-lo; um que parou fica terminal em `cancelado`. Nos dois casos ele continua RECLAMAVEL: o
    # teste seguinte que dispare a mesma entrada recebe `reclamado=True` — desenho de R32, e
    # correcto —, `_despacha` nao lanca thread nenhuma, e a rota consulta um cadaver ou recebe um
    # cancelamento que ninguem pediu.
    for identificador in identificadores:
        modulo.TRABALHOS.esquece(identificador)

    teimosas = [t.name for t in vivas if t.is_alive()]
    if teimosas:
        warnings.warn(
            f"trabalho(s) ainda a correr {teto_s:.0f}s depois do pedido de cancelamento: "
            f"{teimosas}. Eles vão correr POR DENTRO dos testes seguintes, e o sintoma "
            f"aparece longe da causa",
            stacklevel=2,
        )
    return identificadores


@pytest.fixture(autouse=True)
def _sem_trabalho_vivo_no_fim_do_teste() -> "Iterator[None]":
    """Nenhuma thread de trabalho atravessa a fronteira do teste.

    ## O que ela impede, e foi MEDIDO

    `_despacha` lança a thread com `daemon=True` e ninguém a junta. Um teste que dispara uma rota
    longa e confere só o `202` deixa o corretor a correr — e o SERRF custa **209,9 s** no
    `pos_hilic`, mais do que a suíte leva a chegar a `tests/unit`.

    **O desfecho não é lentidão, é reprovação com a mensagem errada:**
    `tests/unit/test_forest_speedup.py` reprovava **cinco** vezes na execução única, sempre em
    `assert not is_installed()`. Aquele cache é global do processo e quem o instala é o
    `fit_transform` do SERRF — ele estar instalado prova que uma thread de outro teste ainda está
    dentro do bloco. A asserção acusava o alvo e a causa estava a três mil testes de distância.

    **Pede cancelamento antes de juntar**, e é o que torna a espera curta: sem isso, juntar seria
    esperar o corretor terminar. É o cancelamento a ser usado por quem ele foi feito para
    servir.

    **E AVISA quando alguma não para.** Sem o aviso, uma thread que ignorasse o cancelamento seria
    indistinguível de não haver thread nenhuma — silêncio é o desfecho padrão de um mecanismo novo
    defeituoso.

    ## O RAMO ATIVO DESTA FIXTURE NÃO FOI EXERCITADO, e isso vai declarado

    **Medido em 07/09/2026, com sonda:** `tests/test_api_execucoes.py` e
    `tests/test_cancelamento_de_trabalhos.py` — os dois arquivos que mais disparam rotas longas —
    correram inteiros e a fixture achou **zero** threads vivas. Ou seja: a lista `vivas` esteve
    sempre vazia, e nem o cancelamento, nem a junção, nem o `esquece`, nem o aviso alguma vez
    correram.

    > **Isto é seguro, não é conserto.** As cinco reprovações de
    > `tests/unit/test_forest_speedup.py` desapareceram na execução seguinte, e a atribuição delas a
    > esta fixture **não foi feita**: o par causal — mesma árvore, só sem a teardown — não foi
    > corrido. O que a fixture garante é que uma thread que sobre **não** atravessa a fronteira do
    > teste; que alguma tenha sobrado, ninguém mediu.

    **Quem for exercitá-la tem um caminho:** disparar `POST /matrizes/corrigir` com `method=serrf`
    num modo grande e terminar o teste sem reclamar. O corretor custa 209,9 s no `pos_hilic`, e a
    thread estará viva na teardown com folga.
    """
    yield

    modulo = sys.modules.get("refino.api.rotas")
    if modulo is not None:
        encerra_trabalhos_vivos(modulo)


@pytest.fixture
def memoria_de_correcoes_privada(monkeypatch: pytest.MonkeyPatch) -> None:
    """Uma memória de correções só deste teste. **Pedida por nome, e não `autouse`.**

    ## Por que NÃO é `autouse`, e o número é o argumento

    A primeira escrita era `autouse=True`, e dava memória vazia a todo teste. Isso conserta os
    instrumentos que dependem de o corretor de facto correr — e apaga o reuso **entre** testes, que
    é trabalho real: `test_api_artefatos.py` dispara a mesma matriz muitas vezes.

    **Medido:** com `autouse`, aquele arquivo passou de **1** reprovação para **2**, e a nova era um
    teste sem relação com cache, a estourar o teto de espera de `reclama`. Conserto que troca uma
    reprovação por outra é deslocamento, não conserto.

    Fica, portanto, pedida pelos poucos testes cujo **instrumento** exige memória fria.

    ## O que estava a acontecer, pela porta do estado partilhado

    `rotas.CORRECOES` é um global do processo. Numa suíte, isso faz o resultado de um teste
    ser **insumo** do seguinte, e dois testes de `test_api_execucoes.py` reprovavam por isso com uma
    mensagem que acusava o produto: *"o corretor nunca começou a correr"*. Ele não começou porque
    **não precisou** — um teste anterior, no mesmo arquivo, já deixara aquela correção na memória, e
    o espião que eles instalam no corretor nunca disparava.

    **O par causal está medido:** os dois reprovam sozinhos no arquivo deles, e passam com a memória
    limpa. O que eles verificam — R30, o `202` antes de o cálculo acabar, e R32, o disparo repetido
    reclamado — continua verdadeiro; o que tinha deixado de funcionar era o **instrumento**.

    **Por que `monkeypatch.setattr` e não um método de limpeza.** Uma thread de trabalho que sobrou
    de um teste anterior lê `CORRECOES` do módulo no instante em que corre; trocar o objeto deixa-a
    a escrever no antigo, que ninguém mais lê. Limpar o mesmo objeto deixaria a escrita dela cair
    dentro da medição do teste corrente.

    **`sys.modules.get` e não `import`**: importar `refino.api.rotas` em toda a coleta arrastaria a
    API inteira para dentro de testes que não a tocam. Se o módulo não foi importado, não há memória
    a isolar — e ela nasce vazia de qualquer forma.
    """
    modulo = sys.modules.get("refino.api.rotas")
    if modulo is None:
        return
    monkeypatch.setattr(modulo, "CORRECOES", type(modulo.CORRECOES)())
    # **E os Palpites medidos, pelo mesmo motivo.** `_PALPITES` é global do processo como
    # `CORRECOES`, e o Palpite que um teste mediu viraria o que a delegação do seguinte lê.
    monkeypatch.setattr(modulo, "_PALPITES", type(modulo._PALPITES)())
    # **E o REGISTRO DE TRABALHOS junto, porque memória fria sozinha não basta.**
    #
    # Medido: com a memória privada e o registro sujo, `test_d381_...` dava **0 acertos**. A causa é
    # R32 a funcionar — um teste anterior do mesmo arquivo já correu `corrigir` sobre `neg_c18` com
    # `method=none`, e a chave do trabalho é a mesma. O disparo é então **reclamado**, a thread não
    # nasce, `_corrige_reusando` não corre, e nada é guardado. A segunda rota não tem o que reusar.
    #
    # > **Antes deste conserto o teste passava, e passava por RESÍDUO:** a memória partilhada já
    # > trazia a correção de um teste anterior, o segundo `acertos` subia, e a asserção via reuso
    # > onde não houve nenhum. É a planta que não separa — o desfecho é o mesmo com e sem o
    # > mecanismo que se quer medir.
    modulo.TRABALHOS.esquece_terminados_alem_de(0)


# ---------------------------------------------------------------------------------------------
# **A PILHA DAS THREADS DE TRABALHO — 06/09/2026. Sem isto a suíte QUEBRA no Windows.**
#
# ## O que se media, e o desfecho não era reprovação: era `EXIT=139`
#
# A suíte completa deste clone morria com `Windows fatal exception: access violation`, e a forma
# era sempre a mesma: uma thread de fundo — a de `_executa_trabalho`, que é onde as rotas longas
# correm corretor — dentro de código numérico, enquanto a principal estava no `_process_events` do
# `pytest-qt`.
#
# **A biblioteca acusada MUDAVA a cada quebra**, e é isso que aponta para fora de todas elas:
#
# | Execução | Onde a thread de fundo estava |
# |---|---|
# | suíte completa | `sklearn.base.__sklearn_tags__`, com `File ???, line ??? in ???` por baixo |
# | segunda metade | `sklearn.tree._classes._prune_tree` |
# | controle sem as mudanças da sessão | `numpy.linalg.lstsq` |
# | mais antiga | `joblib` a serializar com o `cloudpickle` |
#
# Quatro bibliotecas, sempre na thread de fundo, nunca na principal. Defeito de uma delas não se
# comporta assim; **pilha esgotada** comporta-se exactamente assim — quem aparece no relato é quem
# calhou estar mais fundo no instante em que a pilha acabou. O `File ???` é o `faulthandler` a
# desistir de percorrer o quadro seguinte, que é o sintoma próprio.
#
# ## O par causal, três células, e a do meio é a que fecha a atribuição
#
# | Árvore | Pilha da thread | Desfecho |
# |---|---|---|
# | com o trabalho da sessão | padrão | `EXIT=139` |
# | **com o trabalho GUARDADO no `stash`** | padrão | **`EXIT=139`** |
# | com o trabalho da sessão | **64 MB** | **completa** — 266 testes, 2 min 48 s |
#
# A segunda linha é o que impede a conclusão errada: a quebra está em `main` e não veio da sessão.
# A terceira é o par causal — mesma árvore, muda só a pilha.
#
# ## Por que aqui e não em produção, e isto é escolha declarada
#
# A thread nasce em `_despacha`, em `src/refino/api/rotas.py`. Pôr o tamanho **lá** consertaria
# também o executável congelado — e ninguém mediu que o executável quebra. Mudar produção por uma
# quebra observada só sob o `pytest-qt` seria tratar o arnês como se fosse o alvo. Fica no arnês,
# com a medição ao lado, e a pergunta *"e o exe?"* fica ABERTA em vez de respondida por dedução.
#
# 64 MB é endereçamento virtual reservado, não memória residente; num processo de 64 bits não custa.
_PILHA_DA_THREAD_BYTES = 64 * 1024 * 1024
threading.stack_size(_PILHA_DA_THREAD_BYTES)

# Candidatos à raiz dos dados de referência, em ordem de preferência. O primeiro
# cujo `input/` resolva vence.
#
# Por que dois e não um symlink: `data/reference/{input,output}` são symlinks para
# `serrf/data/`, e symlink NÃO materializa em checkout Windows sem
# `core.symlinks=true` — vira arquivo de texto com o caminho dentro. O efeito
# seria o pior possível: todos os golden e integration pulariam em silêncio, e a
# suíte ficaria verde sem verificar nada.
#
# A queda para `serrf/data` resolve sem exigir configuração de ninguém. O symlink
# continua sendo o layout padrão — ele dá o nome lógico — e deixou de ser
# requisito.
_REFERENCE_CANDIDATES = (REPO / "data" / "reference", REPO / "serrf" / "data")


def reference_root_or_none() -> Path | None:
    """Primeira raiz de referência que resolve, ou `None` se nenhuma resolve."""
    for candidate in _REFERENCE_CANDIDATES:
        if (candidate / "input").is_dir():
            return candidate
    return None


def _resolved_reference() -> Path:
    """Raiz de referência, ou o primeiro candidato quando nenhuma resolve.

    Devolver o primeiro candidato na ausência é deliberado: o caminho inexistente
    é o que `_require` nomeia na mensagem de `skip`, e nomear
    `data/reference/input/...` é mais legível para quem precisa buscar os dados.
    """
    return reference_root_or_none() or _REFERENCE_CANDIDATES[0]


REFERENCE = _resolved_reference()


# ---------------------------------------------------------------------------
# Utilitários compartilhados
# ---------------------------------------------------------------------------


@functools.cache
def pyproject() -> dict[str, Any]:
    """`pyproject.toml` parseado, uma vez por sessão.

    Cinco arquivos de teste inspecionam a configuração — o pino do
    interpretador, o limiar de cobertura, os marcadores, os
    flags de estrito. Ler em cada um deles multiplicava o mesmo bloco de
    tomllib sete vezes.
    """
    return tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))


def pinned_python_minor() -> str:
    """O minor declarado em `requires-python`, ex. "3.12".

    Fonte única para os testes que conferem a versão do Python. O README é
    comparado contra isto, nunca contra uma constante literal repetida.
    """
    import re

    requires = pyproject()["project"]["requires-python"]
    match = re.search(r">=(\d+\.\d+)", requires)
    if not match:
        raise AssertionError(f"não consegui extrair o minor de {requires!r}")
    return match.group(1)


def run_module(*args: str, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    """Roda `python -m <args>` num subprocesso.

    Subprocesso e não chamada em processo porque vários testes precisam do
    código de saída de uma invocação limpa: `pytest` rodando dentro de
    `pytest` herdaria `addopts` e mediria a cobertura errada.
    """
    return subprocess.run(
        [sys.executable, "-m", *args],
        cwd=cwd or REPO,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )


# Os quatro modos analíticos, com as contagens de features da tabela.
MODES = ("pos_hilic", "pos_c18", "neg_c18", "neg_hilic")
EXPECTED_FEATURES = {"pos_hilic": 6035, "pos_c18": 2572, "neg_c18": 2184, "neg_hilic": 2039}
EXPECTED_SAMPLES = 68
EXPECTED_QC_ORDERS = (1, 12, 23, 34, 45, 56, 68)

# Métodos de normalização cujos golden files o pipeline R produz.
NORMALIZATION_METHODS = ("none", "batch_loess", "SERRF")


def _require(path: Path, what: str) -> Path:
    """Devolve o caminho, ou faz skip nomeando o que falta."""
    if not path.exists():
        pytest.skip(
            f"Arquivo de referência ausente: {what}\n"
            f"  esperado em: {path}\n"
            f"  Rode scripts/fetch_reference_data.sh para ver a procedência.\n"
            f"  Substituir por dado fabricado é proibido pela §8.4."
        )
    return path


def measurements_path(mode: str) -> Path:
    """Exportação primária do Progenesis QI para um modo."""
    return _require(REFERENCE / "input" / f"measurements_{mode}.csv", f"measurements de {mode}")


def sample_list_path(mode: str) -> Path:
    """Lista de injeção — único lugar onde a ordem temporal existe."""
    return _require(REFERENCE / "input" / f"Sample_List_{mode}.csv", f"Sample_List de {mode}")


def toserrf_path(mode: str) -> Path:
    """Formato intermediário exigido pelo motor SERRF."""
    return _require(REFERENCE / "output" / f"{mode}_toserrf.csv", f"toserrf de {mode}")


def _result_dir(mode: str) -> Path:
    """Diretório de saída do pipeline R. Nome com espaços, não underscores."""
    return REFERENCE / "output" / f"{mode}_toserrf - normalization result"


def normalized_path(mode: str, method: str) -> Path:
    """Golden file de uma matriz normalizada por um dos três métodos."""
    if method not in NORMALIZATION_METHODS:
        raise ValueError(f"método desconhecido: {method!r}; esperado um de {NORMALIZATION_METHODS}")
    return _require(
        _result_dir(mode) / "normalized datasets" / f"normalized by - {method}.csv",
        f"golden normalizado de {mode} por {method}",
    )


def qc_rsd_path(mode: str) -> Path:
    """Golden file de RSD nos QCs."""
    return _require(_result_dir(mode) / "QC - RSD.csv", f"QC - RSD de {mode}")


def tools_r_path() -> Path:
    """O `tools.R` do laboratório — código-fonte do oráculo, não dado.

    Fica fora de `REFERENCE` de propósito: as duas raízes resolvem
    `data/`, e o script vive em `serrf/script/`, que não tem symlink nem queda.
    Resolver aqui mantém a regra de que teste não monta caminho de referência à
    mão, e o `skip` reusa o regime `reference_data_absent`.

    Usado por quem afirma propriedade do TEXTO do oráculo — trocam-se duas
    linhas dele na cópia do runner, e a justificativa da troca é uma
    contagem sobre este arquivo.
    """
    return _require(REPO / "serrf" / "script" / "tools.R", "tools.R do pipeline de referência")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def repo_root() -> Path:
    """Raiz do repositório."""
    return REPO


@pytest.fixture(scope="session")
def reference_root() -> Path:
    """data/reference/, faltando ou não. Use os resolvedores para exigir."""
    return REFERENCE


@pytest.fixture(scope="session")
def reference_files() -> dict[str, Callable[..., Path]]:
    """Resolvedores por tipo de arquivo de referência.

    Cada valor faz `skip` quando o arquivo falta.
    """
    return {
        "measurements": measurements_path,
        "sample_list": sample_list_path,
        "toserrf": toserrf_path,
        "normalized": normalized_path,
        "qc_rsd": qc_rsd_path,
    }


@pytest.fixture(params=MODES)
def mode(request: pytest.FixtureRequest) -> str:
    """Parametriza um teste sobre os quatro modos analíticos."""
    return str(request.param)


# ---------------------------------------------------------------------------
# Integridade
# ---------------------------------------------------------------------------


def sha256_of(path: Path) -> str:
    """Digest de um arquivo, em blocos — os arquivos chegam a 6 MB."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_manifest() -> dict[str, str]:
    """Manifesto de checksums: caminho relativo a data/reference/ -> sha256."""
    if not MANIFEST.exists():
        return {}
    entries: dict[str, str] = {}
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        digest, _, relative = stripped.partition("  ")
        if digest and relative:
            entries[relative] = digest
    return entries


# ---------------------------------------------------------------------------
# A âncora da coluna `none` e o que ela autoriza
# ---------------------------------------------------------------------------


def matrix_from_toserrf(mode: str) -> "RefinoMatrix":
    """Matriz de domínio a partir do `{modo}_toserrf.csv`.

    Os grupos vêm do nome da amostra, pelo mesmo extrator que alimenta os códigos
    de rótulo. `to_matrix` exige o mapa e levanta nomeando as ausentes: nos
    quatro arquivos reais todo nome não-QC expõe o grupo, e **isso é observação
    sobre estes arquivos**, não propriedade do formato. Se um arquivo novo não
    expuser, o erro é alto e nomeia quem faltou — que é o desfecho correto.
    """
    from refino.io.readers import read_toserrf, to_matrix
    from refino.io.validators import group_from_name

    doc = read_toserrf(toserrf_path(mode))
    groups = {
        name: grupo for name in doc.sample_names if (grupo := group_from_name(name)) is not None
    }
    return to_matrix(doc, groups, mode=mode)


def matrix_from_normalized(mode: str, method: str) -> "RefinoMatrix":
    """Matriz de domínio com valores de uma golden normalizada e metadados do `toserrf`.

    As matrizes `normalized by - {método}.csv` só trazem nomes de amostra no
    cabeçalho: `sampleType`, `batch` e `time` existem apenas no `{modo}_toserrf.csv`.
    O metadado vem de lá e os valores de cá, casados **por nome de amostra** e não
    por posição — casar por posição suporia que os dois arquivos têm a mesma ordem
    de colunas, o que é observação sobre estes arquivos e não propriedade do
    formato.

    Reordena as colunas para a ordem de `samples`, que `RefinoMatrix` garante ser
    a ordem de injeção. Sem isso `qc_pc1_trajectory` devolveria os QCs na ordem do
    arquivo, que é agrupada por classe.
    """
    import csv
    from dataclasses import replace

    import numpy as np

    base = matrix_from_toserrf(mode)
    with normalized_path(mode, method).open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.reader(handle))
    header = rows[0]
    faltando = [name for name in base.sample_names if name not in header]
    if faltando:
        raise AssertionError(
            f"golden {method} de {mode} não traz as colunas {faltando[:3]}; "
            "o casamento por nome é o que impede supor ordem de coluna igual"
        )
    columns = [header.index(name) for name in base.sample_names]

    features: list[str] = []
    data: list[list[float]] = []
    for row in rows[1:]:
        if not row or not row[0]:
            continue
        features.append(row[0])
        data.append(
            [np.nan if row[c].strip() in ("", "NA", "NaN") else float(row[c]) for c in columns]
        )
    # O arquivo tem features em linhas; a orientação canônica interna é o contrário.
    values = np.asarray(data, dtype=np.float64).T
    return replace(
        base,
        values=values,
        feature_names=tuple(features),
        provenance=(*base.provenance, f"golden:normalized by - {method}"),
    )


def read_qc_rsd_column(mode: str, method: str) -> tuple[tuple[str, ...], "np.ndarray"]:
    """Uma coluna do `QC - RSD.csv`, com os nomes de feature na ordem do arquivo."""
    import csv

    import numpy as np

    with qc_rsd_path(mode).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    header = rows[0]
    if method not in header:
        raise KeyError(f"coluna {method!r} ausente de {qc_rsd_path(mode).name}: {header}")
    coluna = header.index(method)
    features = tuple(row[0] for row in rows[1:])
    valores = np.array(
        [float(row[coluna]) if row[coluna] not in ("", "NA") else np.nan for row in rows[1:]],
        dtype=np.float64,
    )
    return features, valores


@dataclass(frozen=True)
class AnchorOutcome:
    """Resultado do portão da coluna `none`, por modo. **Expõe, não afirma.**

    Fixture que afirma faria os dependentes darem `error` em vez de `skip`, que é
    o oposto do desejado: doze falhas causadas por um instrumento quebrado
    escondem qual é o defeito.
    """

    ok: bool
    detalhe: str
    por_modo: dict[str, str]


@pytest.fixture(scope="session")
def rsd_instrument_anchor() -> AnchorOutcome:
    """Compara `qc_rsd(outlier_removed)` com a coluna `none` do R, uma vez.

    É o que **valida o instrumento** contra artefato externo, e é o que autoriza
    as doze medianas in-sample a valerem como regressão. Sem instrumento
    validado, doze números medidos por ele não informam nada.
    """
    import numpy as np

    from refino.metrics.rsd import qc_rsd

    por_modo: dict[str, str] = {}
    for mode in MODES:
        try:
            matrix = matrix_from_toserrf(mode)
            _, esperado = read_qc_rsd_column(mode, "none")
        except Exception as erro:  # a âncora reporta o motivo; não propaga
            por_modo[mode] = f"indisponível: {erro}"
            continue

        obtido = qc_rsd(matrix, variant="outlier_removed")
        if obtido.size != esperado.size:
            por_modo[mode] = f"tamanhos diferentes: {obtido.size} vs {esperado.size}"
            continue

        indefinidos_iguais = bool(np.array_equal(np.isnan(obtido), np.isnan(esperado)))
        definidos = ~np.isnan(obtido) & ~np.isnan(esperado)
        fracao = float(np.mean(np.abs(obtido[definidos] - esperado[definidos]) < 1e-6))
        if indefinidos_iguais and fracao >= 0.99:
            por_modo[mode] = "ok"
        else:
            por_modo[mode] = (
                f"conjuntos de indefinidos {'iguais' if indefinidos_iguais else 'DIFERENTES'}, "
                f"{100 * fracao:.2f}% das definidas dentro de 1e-6"
            )

    ok = all(estado == "ok" for estado in por_modo.values())
    return AnchorOutcome(
        ok=ok,
        detalhe="instrumento validado contra a coluna `none` do R"
        if ok
        else "portão do [CA-2.1] reprovado: " + "; ".join(f"{m}: {e}" for m, e in por_modo.items()),
        por_modo=por_modo,
    )


# ---------------------------------------------------------------------------
# A guarda de `skip` declarado
# ---------------------------------------------------------------------------
#
# Os ganchos vivem em `tests/skip_guard.py` e são registrados aqui por importação.
# Ficam num módulo próprio para que o teste que planta a violação possa carregá-los
# com `-p tests.skip_guard` num arquivo fora de `tests/` — um `conftest` só alcança
# o que está abaixo dele, e uma guarda que não pode ser testada em isolamento é uma
# guarda cuja reprovação ninguém verificou.
from tests.skip_guard import (  # noqa: E402  (registro de gancho, depois das fixtures)
    pytest_collectreport,
    pytest_runtest_logreport,
    pytest_sessionfinish,
)

__all__ = ["pytest_collectreport", "pytest_runtest_logreport", "pytest_sessionfinish"]
