"""O contrato de orçamento de tempo — a regra da especificação mecanizada para a CLASSE (E16).

**O defeito que ela fecha, medido em 08/08/2026.** A especificação declara, na linha de *Orçamento
de tempo de parede*, que a medição registra `load_per_core` nas duas pontas e produz **`skip`
declarado acima de 0,5 no início**; o critério repete com *"nunca `pass`, nunca `fail`"*.
Medido: `load_per_core` não aparecia em linha nenhuma da árvore, e o critério — o **único**
portão de orçamento que existe — gravava `load_1min`, imprimia, escrevia o aviso em prosa na
mensagem de falha e **assertava de todo modo**. Com `make test-docker` em paralelo, ele
reprovou com `load_per_core` de **0,78**, e as duas mensagens — contenção e regressão — eram a
mesma.

**Por que um portão e não só a guarda.** A especificação declara a regra para a **classe**, não
para um item. Consertar só o critério deixaria o próximo portão de orçamento nascer sem
guarda, e ele nasceria exactamente como este nasceu: com a intenção certa e a chamada ausente.
É a escolha de projeto — *toda vez que a escolha for entre "lembrar de
fazer" e "um mecanismo que faça", a segunda vence*.

## A forma: um FUNIL, e o portão confere que ninguém o contorna

`tests/orcamento.py::afirma_orcamento` é o único sítio em que um teto de orçamento vira
asserção, e `abre_medicao_de_orcamento` é o único que lê a carga de abertura. Daí saem duas
direções conferíveis por análise sintática:

| Direção | O que acusa | O que pega |
|---|---|---|
| **A** | teto dentro de um `ast.Compare` fora do funil | `assert medido < TETO_X` à mão |
| **B** | quem chama `afirma_orcamento` sem `abre_medicao_de_orcamento` | a metade inicial |

**A direção A é sobre COMPARAÇÃO e não sobre menção**, e a escolha é medida: o critério
interpola o teto num `print` de relato, e um portão que acusasse qualquer menção reprovaria o
relatório — portão que sempre reprova. Comparar é o verbo
que decide.

## O que ele NÃO confere, declarado (proxy pela forma)

Que `abre_medicao_de_orcamento` seja chamada **antes** do trabalho caro. A ordem é o que a
especificação pede — *"no início"* — e verificá-la exigiria análise de fluxo que custa mais que o
defeito que ela pegaria: chamar a guarda depois tem o preço da execução sem o benefício, e isso
aparece como lentidão, não como silêncio. O que fica coberto é a ausência, que é o modo de falha
mudo.

**E não confere ARNÊS.** `scripts/remede_teto_to_8_4.py` e `scripts/rederiva_teto_entre_sessoes.py`
medem a mesma quantidade e **não** pulam sob carga, de propósito: eles estimam a distribuição
de que o teto sai, e censurar os pontos de carga alta encolheria o desvio — modo de falha na
direção que parece melhor. A guarda é do que **decide**, não do que **descreve**.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from tests.orcamento import (
    DETALHE_DO_LOAD,
    LIMITE_LOAD_PER_CORE,
    abre_medicao_de_orcamento,
    afirma_orcamento,
    contaminado,
    load_medivel,
    load_per_core,
)
from tests.skip_policy import undeclared_reason

RAIZ = Path(__file__).resolve().parent.parent
TESTS = RAIZ / "tests"

# O prefixo que nomeia constante de teto de orçamento. É convenção declarada e não descoberta:
# um portão que tentasse inferir "isto é um teto" de um float qualquer erraria nas duas direções,
# e proxy que erra nas duas morre no primeiro falso positivo.
PREFIXO_DE_TETO = "TETO_"

# O funil. Ele é o único isento da direção A — é onde a comparação DEVE estar.
FUNIL = "orcamento.py"

_ABRE = "abre_medicao_de_orcamento"
_AFIRMA = "afirma_orcamento"


def _fontes() -> list[Path]:
    return sorted(p for p in TESTS.rglob("*.py") if "__pycache__" not in p.parts)


def constantes_de_teto(fonte: str) -> set[str]:
    """Nomes de módulo atribuídos com o prefixo de teto — função **pura sobre texto**."""
    achados: set[str] = set()
    for no in ast.parse(fonte).body:
        alvos: list[ast.expr] = []
        if isinstance(no, ast.Assign):
            alvos = list(no.targets)
        elif isinstance(no, ast.AnnAssign):
            alvos = [no.target]
        achados |= {
            a.id for a in alvos if isinstance(a, ast.Name) and a.id.startswith(PREFIXO_DE_TETO)
        }
    return achados


def comparacoes_fora_do_funil(fonte: str) -> set[str]:
    """Constantes de teto que aparecem dentro de uma comparação — a direção A.

    Pura sobre texto para que a planta entre como **argumento** e o negativo não consiga ficar
    inerte.
    """
    tetos = constantes_de_teto(fonte)
    if not tetos:
        return set()
    arvore = ast.parse(fonte)
    achados: set[str] = set()
    for no in ast.walk(arvore):
        if not isinstance(no, ast.Compare):
            continue
        for parte in [no.left, *no.comparators]:
            for interno in ast.walk(parte):
                if isinstance(interno, ast.Name) and interno.id in tetos:
                    achados.add(interno.id)
    return achados


def _nomes_chamados(fonte: str) -> set[str]:
    chamados: set[str] = set()
    for no in ast.walk(ast.parse(fonte)):
        if isinstance(no, ast.Call):
            if isinstance(no.func, ast.Name):
                chamados.add(no.func.id)
            elif isinstance(no.func, ast.Attribute):
                chamados.add(no.func.attr)
    return chamados


def afirma_sem_abrir(fonte: str) -> bool:
    """`True` quando o módulo fecha um orçamento sem ter aberto — a direção B."""
    chamados = _nomes_chamados(fonte)
    return _AFIRMA in chamados and _ABRE not in chamados


# ---------------------------------------------------------------------------
# Direção A — nenhum teto vira comparação fora do funil
# ---------------------------------------------------------------------------


def test_nenhuma_constante_de_teto_e_comparada_fora_do_funil() -> None:
    """O estado exacto em que o critério viveu: `assert mili_por_celula < TETO_...`."""
    acusacoes = {
        caminho.relative_to(RAIZ).as_posix(): sorted(fora)
        for caminho in _fontes()
        if caminho.name != FUNIL
        and (fora := comparacoes_fora_do_funil(caminho.read_text(encoding="utf-8")))
    }
    assert not acusacoes, (
        "teto de orçamento comparado fora de `tests/orcamento.py`:\n  "
        + "\n  ".join(f"{arquivo}: {nomes}" for arquivo, nomes in sorted(acusacoes.items()))
        + "\nA §8.2 exige `load_per_core` nas duas pontas e `skip` acima de 0,5 no início. "
        "Comparação escrita à mão pula a guarda, e sob contenção a reprovação não distingue "
        "regressão de máquina ocupada (E16). Use `afirma_orcamento`."
    )


def test_a_varredura_encontra_ao_menos_uma_constante_de_teto() -> None:
    """Guarda de vacuidade, e ela vai **na mesma superfície** e não num teste vizinho.

    A direção A é asserção de **ausência** e passa com conjunto vazio: se o prefixo mudar, se
    a extração deixar de casar, ou se o único portão de orçamento for removido, ela fica verde
    **sem olhar uma constante**. É a forma de defeito que este arquivo existe para não repetir.
    """
    todas = {
        caminho.relative_to(RAIZ).as_posix(): sorted(tetos)
        for caminho in _fontes()
        if (tetos := constantes_de_teto(caminho.read_text(encoding="utf-8")))
    }
    assert todas, (
        f"nenhuma constante `{PREFIXO_DE_TETO}*` casou na suíte: a direção A passaria por "
        "vacuidade. Renomear o prefixo ou remover o único portão de orçamento produz este "
        "estado, e ele é indistinguível de contrato satisfeito"
    )


# ---------------------------------------------------------------------------
# Direção B — quem fecha um orçamento abriu-o
# ---------------------------------------------------------------------------


def test_todo_modulo_que_afirma_orcamento_abre_a_medicao() -> None:
    """A metade *"no início"* da especificação, que é a que tem o custo."""
    acusados = sorted(
        caminho.relative_to(RAIZ).as_posix()
        for caminho in _fontes()
        if caminho.name != FUNIL and afirma_sem_abrir(caminho.read_text(encoding="utf-8"))
    )
    assert not acusados, (
        f"módulo que chama `{_AFIRMA}` sem `{_ABRE}`: {acusados}. A guarda de carga tem de "
        "correr ANTES da medição — pular depois de gastar a execução tem o preço sem o "
        "benefício, e a §8.2 diz `no início`"
    )


def test_ao_menos_um_modulo_usa_o_funil() -> None:
    """Vacuidade da direção B, pelo mesmo argumento da direção A."""
    usuarios = sorted(
        caminho.relative_to(RAIZ).as_posix()
        for caminho in _fontes()
        if caminho.name != FUNIL and _AFIRMA in _nomes_chamados(caminho.read_text(encoding="utf-8"))
    )
    assert usuarios, (
        f"nenhum módulo chama `{_AFIRMA}`: as duas direções deste portão estão inertes. "
        "O `[TO-8.4]` é o portão de orçamento que existe hoje, e é ele que a varredura espera"
    )


# ---------------------------------------------------------------------------
# Os negativos, com a planta como ARGUMENTO das funções puras
# ---------------------------------------------------------------------------


def test_negativo_a_comparacao_escrita_a_mao_e_acusada() -> None:
    """A planta é o defeito **real**, com a forma que ele tinha no critério."""
    plantada = (
        "TETO_MILINUCLEO_MIN_POR_CELULA = 8.56\n"
        "def test_x():\n"
        "    assert mili_por_celula < TETO_MILINUCLEO_MIN_POR_CELULA, 'estourou'\n"
    )
    assert constantes_de_teto(plantada) == {"TETO_MILINUCLEO_MIN_POR_CELULA"}
    assert comparacoes_fora_do_funil(plantada) == {"TETO_MILINUCLEO_MIN_POR_CELULA"}


def test_negativo_com_a_planta_removida_o_portao_deixa_de_acusar() -> None:
    """O par causal: as duas entradas diferem **só** pela planta, e os desfechos divergem.

    O relato do critério interpola o teto num `print`, e é essa a forma que tem de **passar**
    — um portão que acusasse menção reprovaria o relatório, e sempre reprovaria.
    """
    limpa = (
        "TETO_MILINUCLEO_MIN_POR_CELULA = 8.56\n"
        "def test_x():\n"
        "    print(f'teto {TETO_MILINUCLEO_MIN_POR_CELULA}')\n"
        "    afirma_orcamento(a, m, TETO_MILINUCLEO_MIN_POR_CELULA, unidade='u', contexto='c')\n"
    )
    assert constantes_de_teto(limpa), "o extrator não casou nada; o negativo mediria vacuidade"
    assert comparacoes_fora_do_funil(limpa) == set()


def test_negativo_afirmar_sem_abrir_e_acusado() -> None:
    """A direção B nas duas pontas, com a planta como argumento."""
    sem = "def test_x():\n    afirma_orcamento(a, m, T, unidade='u', contexto='c')\n"
    com = (
        "def test_x():\n"
        "    a = abre_medicao_de_orcamento()\n"
        "    afirma_orcamento(a, m, T, unidade='u', contexto='c')\n"
    )
    assert afirma_sem_abrir(sem)
    assert not afirma_sem_abrir(com)


# ---------------------------------------------------------------------------
# Os QUATRO desfechos da guarda, e todos são vistos
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("medido", "esperado"),
    [(0.78, True), (LIMITE_LOAD_PER_CORE, False), (0.20, False)],
)
def test_o_discriminante_de_contaminacao_separa_nas_duas_direcoes(
    medido: float, esperado: bool
) -> None:
    """`0,78` é o valor real da execução que motivou E16; o limiar exacto NÃO contamina."""
    assert contaminado(medido, LIMITE_LOAD_PER_CORE) is esperado


def test_desfecho_1_a_abertura_pula_sob_carga_e_a_mensagem_nomeia_o_numero() -> None:
    """Primeiro desfecho: `skip`. A mensagem é lida do que foi EMITIDO, não recomposta.

    A política de `skip` deste projeto ensina a ler `skipped` como legítimo,
    então um regime novo cuja mensagem não diga **o número, o limiar e o que fazer** vira o
    esconderijo da próxima medição que não aconteceu.

    **A asserção é sobre `Skipped.msg` e não sobre um recompositor**: recompor a mensagem
    provaria que duas funções concordam entre si, e o que precisa ser
    verdade é que o `skip` **realmente emitido** case o padrão que o regime declara.
    """
    with pytest.raises(BaseException) as capturado:
        abre_medicao_de_orcamento(medido=0.78)
    assert capturado.typename == "Skipped", capturado.typename

    texto = str(getattr(capturado.value, "msg", ""))
    assert "orçamento de tempo NÃO medido: load_per_core inicial" in texto
    assert "0.78" in texto and "0.50" in texto
    assert "pedem investigações opostas" in texto
    assert not undeclared_reason(texto), (
        "a mensagem emitida não casa nenhum regime de `skip` declarado — a sessão inteira "
        "reprovaria por AR13, e o teste que a exercita passaria"
    )


def test_desfecho_2_a_abertura_segue_com_a_maquina_livre() -> None:
    """Segundo desfecho: a guarda deixa passar, e devolve a abertura que a asserção exige."""
    abertura = abre_medicao_de_orcamento(medido=0.10)
    assert abertura.load_per_core == 0.10
    assert abertura.limite == LIMITE_LOAD_PER_CORE
    assert abertura.nucleos > 0


def test_desfecho_3_o_orcamento_dentro_do_teto_passa_e_devolve_o_fechamento() -> None:
    """Terceiro desfecho: dentro do teto."""
    abertura = abre_medicao_de_orcamento(medido=0.10)
    fechamento = afirma_orcamento(abertura, 8.0, 8.56, unidade="u", contexto="ctx")
    assert fechamento >= 0.0


def test_desfecho_4_o_estouro_reprova_e_a_mensagem_declara_que_a_guarda_deixou_passar() -> None:
    """Quarto desfecho: estouro — e a mensagem tem de **fechar** a leitura de contenção.

    É o conteúdo da correção: antes, o teste reprovava e a mensagem oferecia a contenção como
    explicação disponível, num caso em que ela era a explicação certa e ninguém podia saber.
    Com a guarda, chegar aqui **prova** que a carga estava dentro do limite, e a mensagem diz
    isso — o que transforma o vermelho em sinal de regressão em vez de ambiguidade.
    """
    abertura = abre_medicao_de_orcamento(medido=0.10)
    with pytest.raises(AssertionError) as erro:
        afirma_orcamento(abertura, 13.508, 8.56, unidade="milinúcleo-min/célula", contexto="ctx")
    texto = str(erro.value)
    assert "13.508" in texto and "8.56" in texto
    assert "a guarda da §8.2 deixou passar" in texto
    assert "contenção NÃO é a explicação disponível" in texto


def test_o_load_per_core_e_a_quantidade_da_secao_8_2_e_nao_o_load_nu() -> None:
    """`load_1min` sozinho não substitui: load 9 é contenção em 4 núcleos e folga em 64.

    Instrumentação, e rotulada como tal — o que ela pega é a divisão ter sido esquecida, que é
    barato e é exactamente o que faltava na árvore inteira até 08/08/2026.
    """
    if not load_medivel():
        pytest.skip(f"load average indisponível: {DETALHE_DO_LOAD}")

    import os

    nu = os.getloadavg()[0]
    assert load_per_core(4) == pytest.approx(nu / 4)
    assert load_per_core(64) == pytest.approx(nu / 64)
    assert load_per_core(4) > load_per_core(64)
