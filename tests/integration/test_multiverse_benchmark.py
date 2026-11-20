"""Benchmark de tempo da grade do multiverso, `slow`, para o noturno.

**O que ele mede, e o critério declara a forma:** tempo de parede da grade padrão sobre o
`pos_hilic`, com o orçamento também em **núcleo-minutos**, **máquina de referência registrada** e
**load average no início e no fim** — pela forma do critério.

**O REGIME ENTRA NO NÚMERO E NÃO COMO RESSALVA.** O critério declara *"menos de 30 minutos em
**8 núcleos**"*, e a máquina de referência tem **12** com load não nulo. Um tempo de parede sem o
regime ao lado **não é comparável ao teto**: mais núcleos empurram para baixo, load concorrente para
cima, e as duas não se cancelam de forma conhecida. Por isso o relato traz núcleo-minutos, que é a
quantidade que atravessa máquinas, e o tempo de parede vem sempre com a contagem de núcleos colada.

**O CUSTO É QUASE TODO FIXO, e é isso que o benchmark tem de mostrar.** Medido na
decomposição:

| Correção | Custo **fixo**, uma vez por prefixo | Marginal por célula |
|---|---|---|
| `none` | 0,7 s | 0,01 s |
| `loess` | 32,9 s | 0,03 s |
| **`serrf`** | **1.076,6 s** | 0,47 s |

**O `serrf` é ~97% do custo total, e está LIBERADO:** é selecionável na CLI, na API e na tela,
e a rota do multiverso não o recusa — a grade que o usuário dispara tem **4.320** células.

**O que NÃO mudou é o que este módulo mede, e a distinção é o conteúdo.** A composição medida aqui
continua **sem** `serrf`, e a razão trocou junto com o estado: não é mais *"cobrar teto sobre
configuração indisponível"* — é que o teto de 8,56 foi **derivado sobre esta composição**, e medir
outra enquanto o teto está em remediação quebraria a comparabilidade com a amostra de treze
repetições de que ele saiu, e com a série diária. Ver a docstring do teste.

**A reavaliação tem condição, não data:** acontece
quando o critério sair de remediação. Gatilho por data orfana quando a data passa.

**O custo continua a saltar ~30x para quem dispara a grade completa**, e é o motivo de este módulo
reportar os dois números em vez de um — hoje mais do que antes, porque agora há quem possa pagá-lo
pela porta da frente.

**Por que reportar e não só cobrar teto.** Um `assert tempo < 30 min` sobre a composição de
derivação passaria por não incluir `serrf`, e passaria **pelo motivo errado**. O relato traz a
composição para que o número diga de onde vem.

**A GUARDA DE CARGA ENTROU EM 08/08/2026, e ela estava DECLARADA desde o início.** A especificação
exige, para orçamento de tempo de parede, *"`load_per_core` registrado nas duas pontas; `skip`
declarado acima de 0,5 no início"*, e o critério repete *"nunca `pass`, nunca `fail`"*. Este
módulo gravava `load_1min`, imprimia, escrevia o aviso em prosa na mensagem de falha — e
**assertava de todo modo**. Medido: com `make test-docker` em paralelo, `load_per_core` inicial de
**0,78**, ele reprovou com **13,508** contra o teto de 8,56, e a mensagem de contenção era
indistinguível da de regressão. A guarda vive em `tests/orcamento.py` porque a regra é da **classe**
e não deste item, e `tests/test_orcamento_contrato.py` é o que impede o próximo portão de orçamento
de nascer sem ela.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from refino.io.readers import load_matrix
from refino.multiverse.executor import run_multiverse
from refino.multiverse.grid import GridDefinition
from tests.conftest import measurements_path, sample_list_path
from tests.orcamento import (
    abre_medicao_de_orcamento,
    afirma_orcamento,
    load_medivel,
    load_per_core,
)

pytestmark = [pytest.mark.slow, pytest.mark.integration]

MODO = "pos_hilic"

# **O teto do critério, na unidade que ele declara desde sempre: NÚCLEO-MINUTOS POR CÉLULA.**
#
# Derivado e não redondo, pela forma do critério: **13 repetições** na máquina de referência,
# com o load de 2,31 a 13,00, deram média **8,497**, desvio **0,021** (0,2%) e máximo **8,530**
# milinúcleo-min/célula. O teto é **média + 3 desvios** = 8,56.
#
# **A PRIMEIRA DERIVAÇÃO USOU n = 4 E REPROVOU NA QUINTA MEDIÇÃO.** Ela deu média 8,472 e
# desvio 0,011, teto 8,51 — e a execução seguinte mediu **8,521**. As quatro repetições eram
# consecutivas, na mesma sessão e sob load parecido, então o desvio media **ruído de relógio** e
# não a dispersão do regime. Os cinco pontos originais **entram** na amostra de 13: descartá-los
# seria escolher a amostra depois de ver o resultado.
#
# **A unidade tem duas partes e cada uma fecha um buraco da outra:** *por célula* sobrevive à
# grade mudar de tamanho — o defeito que aposentou os 30 min, escritos para 1.296 células numa
# grade de 4.320 —, e *núcleo-minutos* sobrevive à máquina mudar. Nenhuma das duas sozinha basta.
#
# **O que ele NÃO cobre:** núcleo-minutos normaliza contagem de núcleos, **não frequência**. Noutra
# máquina, remedir — mesma cláusula do critério.
TETO_MILINUCLEO_MIN_POR_CELULA = 8.56

# **ESTA STRING PARECIA CARREGAR ESTADO E CARREGAVA MODELO.** Ela dizia
# `"AMD Ryzen 5 7600, 6 núcleos físicos / 12 lógicos, 5171,85 MHz máx"`, e o `5171,85` é
# `cpuinfo_max_freq` — o **máximo de fábrica**, lido de `/sys`. Ele é **constante por construção**:
# idêntico e hoje, e por isso incapaz de distinguir "a máquina mudou de estado" de "a
# máquina é a mesma". Quem lesse o campo concluiria que a frequência estava registrada. Ela nunca
# esteve.
#
# **A CPU não roda nesse número:** ociosa, `/proc/cpuinfo` dá 3.401 a 4.242 MHz; sob a carga desta
# grade, 4.387 a 4.460. O campo passa a declarar o que é **fixo** e o `_regime()` a medir o que
# **varia** — que é a distinção que o campo confundia.
MODELO_DA_MAQUINA = (
    "AMD Ryzen 5 7600, 6 núcleos físicos / 12 lógicos "
    "(nominal de fábrica: 5171,85 MHz máx, NUNCA observado)"
)


def _mhz_medio() -> float:
    """Frequência média dos núcleos **agora**, de `/proc/cpuinfo`.

    É a quantidade que a `MAQUINA_DE_REFERENCIA` parecia registrar e não registrava. Barata — uma
    leitura de arquivo — e é a única das três que muda dentro de uma execução.
    """
    try:
        linhas = Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines()
    except OSError:  # pragma: no cover - fora de Linux não há o arquivo, e o regime degrada
        return 0.0
    valores = [float(linha.split(":")[1]) for linha in linhas if linha.startswith("cpu MHz")]
    return round(sum(valores) / len(valores), 1) if valores else 0.0


def _governador() -> str:
    """O governador de frequência, que decide se a máquina persegue o boost.

    `powersave` com `amd-pstate-epp` e `performance` produzem regimes diferentes na **mesma**
    máquina, e núcleo-minutos não normaliza a diferença.
    """
    caminho = Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
    try:
        return caminho.read_text(encoding="utf-8").strip()
    except OSError:  # pragma: no cover - nem todo kernel expõe cpufreq
        return "indisponível"


def _boot() -> tuple[str, float]:
    """`(identificador do boot, segundos ligada)`.

    **Entrou em 08/08/2026 porque DIA e BOOT estavam perfeitamente confundidos.** Todas as
    medições de 08/08 — cinco sessões limpas, o dia 1 da diária e o teste térmico — são do **mesmo**
    boot, de 07/08 20:29; a medição do teto é de um boot anterior. A componente de 0,197 que rompe
    o teto pode ser de calendário ou de reinício, e nenhuma medição feita até aqui as separa.

    **Custa duas leituras de arquivo e é o que permite a separação acontecer depois**, sem exigir
    que se decida agora qual dos dois é o mecanismo — que é a pergunta que a saída 2 precisa e a
    derivação do teto não.
    """
    try:
        identificador = Path("/proc/sys/kernel/random/boot_id").read_text(encoding="utf-8").strip()
    except OSError:  # pragma: no cover - fora de Linux
        identificador = "indisponível"
    try:
        ligada = float(Path("/proc/uptime").read_text(encoding="utf-8").split()[0])
    except (OSError, IndexError, ValueError):  # pragma: no cover
        ligada = 0.0
    return identificador, ligada


LOAD_INDISPONIVEL = "INDISPONIVEL - `os.getloadavg` e POSIX e nao existe neste sistema"


def _regime() -> dict[str, float | int | str]:
    """A máquina e o estado dela, lidos no instante da medição.

    **Três quantidades passaram a ser medidas em 08/08/2026, e as três variam** — ao
    contrário do `5171,85` que o campo antigo carregava:

    | Quantidade | Por que entra |
    |---|---|
    | `mhz_medio` | a unidade normaliza **contagem** e não **frequência**, e esta era a hipótese
      que não pôde ser testada por não haver leitura |
    | `governador` | muda o regime na mesma máquina, e não aparecia em número nenhum |
    | `load_1min` | já existia — núcleos disponíveis não é núcleos livres |
    | `load_per_core` | **08/08/2026 (E16)** — é a quantidade que a especificação nomeia, e
    `load_1min`   sozinho não a substitui: load 9 é contenção severa em 4 núcleos e folga em 64 |

    **O que ainda NÃO é medido, declarado:** `os.cpu_count()` é a contagem **nominal**, e a
    execução obtém menos — medido, **8,46** de 12 com a máquina ociosa. Os núcleos
    *efetivos* são saída da execução e não propriedade da máquina, então **não** entram como
    normalizador: dividir por eles tornaria o critério cego a exatamente o que ele mede.

    **`_regime()` DESCREVE e não DECIDE, e a distinção é o que mantém a guarda fora daqui.** Este
    dicionário é lido pelos dois papéis — o portão do critério e os arnêses de derivação, que
    incluem a sessão diária. Pôr o `skip` aqui censuraria a amostra do arnês e encolheria
    o desvio de que o teto sai, que é o modo de falha na direção que parece melhor.
    """
    boot_id, ligada_s = _boot()
    nucleos = os.cpu_count() or 0
    return {
        "nucleos_nominais": nucleos,
        # **Os dois DESCREVEM a indisponibilidade em vez de levantar, e a forma da cadeia é a que
        # `scripts/mede_etapas_do_pipeline.py` e `scripts/mede_passos_do_serrf.py` já gravavam.**
        # `os.getloadavg` é POSIX; num sistema sem ele, `round(...)` rebentava com `AttributeError`
        # e levava quatro testes de contrato consigo. Um `skip` aqui está fora de questão pelo
        # motivo escrito acima — este dicionário **descreve e não decide** —, e um `0.0` seria
        # pior: afirmaria ociosidade que ninguém mediu. As chaves FICAM, e é isso que mantém o
        # portão de bijeção de `test_regime_contrato.py` a bijetar.
        "load_1min": (round(os.getloadavg()[0], 2) if load_medivel() else LOAD_INDISPONIVEL),
        "load_per_core": (
            round(load_per_core(nucleos), 3) if load_medivel() else LOAD_INDISPONIVEL
        ),
        "mhz_medio": _mhz_medio(),
        "governador": _governador(),
        "boot_id": boot_id,
        "ligada_s": round(ligada_s, 1),
        "modelo_da_maquina": MODELO_DA_MAQUINA,
    }


@pytest.mark.golden
def test_to_8_4_benchmark_da_grade_reporta_tempo_e_composicao() -> None:
    """**Reporta, e o único portão é o teto do critério sobre a COMPOSIÇÃO DE DERIVAÇÃO.**

    O que se afirma com `assert` é o custo da grade sem `serrf` — 2.880 células. A grade
    completa, 4.320, é reportada e **não** cobrada.

    **A grade completa tem 9.600 células, e sem o `serrf` são
    6.400.** O teto de 8,56 continua o de 2.880, e o número por célula sobrevive ao tamanho da
    grade, e não à composição. Em 05/11/2026, na máquina de referência, este teste passou com
    6,764 sobre as 6.400; a última medição sobre as 2.880 tinha dado 8,736. As duas não são
    comparáveis, e a causa da diferença não foi medida.

    **A RAZÃO DESTA EXCLUSÃO MUDOU em 09/08/2026, e o comportamento não.** Ela dizia
    *"`serrf` não é selecionável, e cobrar teto sobre configuração indisponível mediria coisa
    que não roda"* — e o `serrf` foi liberado nas três superfícies de usuário, o que tornaria
    essa frase **falsa** com o filtro intacto. A razão que fica é outra e sobrevive à
    liberação:

    > **O teto de 8,56 foi derivado de treze repetições SOBRE ESTA COMPOSIÇÃO. Medir outra
    > enquanto o teto está em remediação quebra a comparabilidade da série diária —
    > o número deixaria de ser comparável com aquele de que o teto saiu.**

    **A condição é *"enquanto o critério estiver em remediação"*, e NÃO uma data.**
    Medido o custo do gatilho por data: *"CI parado até 2026-09-01"* descrevia um estado
    que não ocorreria, e ninguém olhou quando a data passou. Condição tem referente que se
    pode consultar; data tem referente que expira sozinho.

    **A reavaliação tem condição** — é
    disparada quando o critério sair de remediação (teto rederivado, ou critério
    emendado). Sem isso o filtro fica para sempre com uma razão que já expirou, que é
    exactamente o defeito deste mesmo `if`.

    **E o filtro NÃO consulta `UNAVAILABLE_METHODS` — é comparação com literal, de propósito
    e agora declarado.** Essa foi a razão de a liberação não o tocar, e é o que o tornava
    silencioso. Mantê-lo literal passou a ser a escolha certa pelo motivo NOVO: a
    exclusão aqui é sobre **comparabilidade de medição**, não sobre disponibilidade, e ligá-la
    ao registro de bloqueio faria o benchmark mudar de composição no dia em que alguém
    bloqueasse ou liberasse um método — que é precisamente o que não pode acontecer no meio
    de uma remediação.
    """
    # A guarda ANTES de qualquer custo: a especificação diz "no início", e pular depois de gastar a
    # execução tem o preço sem o benefício. Ela pode terminar o teste aqui, com `skip`.
    abertura = abre_medicao_de_orcamento()

    matriz = load_matrix(measurements_path(MODO), sample_list_path(MODO), MODO)
    grade = GridDefinition()
    regime = _regime()

    # Literal e não `UNAVAILABLE_METHODS`, e a docstring diz por quê: a exclusão é sobre
    # COMPARABILIDADE com a amostra de que o teto saiu, não sobre disponibilidade. Desde
    # o `serrf` É selecionável, e este filtro tem de continuar a excluí-lo enquanto o
    # critério estiver em remediação.
    da_derivacao = tuple(c for c in grade.corrections if c != "serrf")
    assert da_derivacao, "composição de derivação vazia; o benchmark mediria zero"
    assert "serrf" not in da_derivacao, (
        "o `serrf` entrou na composição medida: o teto de 8,56 foi derivado sem ele, e "
        "compará-los mede coisas diferentes"
    )

    from dataclasses import replace

    grade_medida = replace(grade, corrections=da_derivacao)

    load_inicial = os.getloadavg()[0]
    inicio = time.perf_counter()
    execucao = run_multiverse(matriz, grade_medida, n_jobs=-1)
    parede = time.perf_counter() - inicio
    load_final = os.getloadavg()[0]

    minutos = parede / 60.0
    # **`nucleos_nominais` e não `nucleos`, e a distinção custou 3h20 de registro sobre estado não
    # medido.** A renomeação da chave em `_regime()` e atualizou os dois `print`; esta linha,
    # a única que faz CONTA com ela, ficou. O teste passou a morrer com `KeyError` **antes** da
    # asserção do teto — reprova pelo motivo errado —, e o portão dos
    # `[TO-N.M]` continuou verde porque confere existência de nó e ausência de marcador, não
    # desfecho. `test_regime_contrato.py` é a guarda que fecha a classe, e roda no portão de commit
    # porque este módulo é `slow`.
    nucleo_minutos = minutos * float(regime["nucleos_nominais"])
    mili_por_celula = 1000.0 * nucleo_minutos / grade_medida.size()

    # `-rA` no alvo `test-docker` existe porque o pytest engole `stdout` de teste que passa; aqui o
    # noturno roda com `-rA` pela mesma razão. Relatório que só aparece quando falha é o
    # defeito — e este teste existe para RELATAR.
    print(
        f"\n[TO-8.4] composição DE DERIVAÇÃO ({', '.join(da_derivacao)}): "
        f"{grade_medida.size()} células em {minutos:.2f} min "
        f"({nucleo_minutos:.1f} núcleo-min) | células/s: {grade_medida.size() / parede:.1f}\n"
        f"  {mili_por_celula:.3f} milinúcleo-min/célula, teto "
        f"{TETO_MILINUCLEO_MIN_POR_CELULA}\n"
        f"  regime: {regime['nucleos_nominais']} núcleos nominais, load "
        f"{load_inicial:.2f} -> {load_final:.2f} "
        f"(load_per_core {abertura.load_per_core:.2f}, limite da guarda {abertura.limite:.2f}), "
        f"{regime['mhz_medio']} MHz médios, "
        f"governador {regime['governador']}\n"
        f"  modelo: {MODELO_DA_MAQUINA}\n"
        f"  grade COMPLETA: {grade.size()} células, e o `serrf` a torna ~30x mais cara — "
        f"ele é ~97% do custo e está LIBERADO nas três superfícies. "
        f"Fica fora DESTA medição por comparabilidade com a amostra de derivação do teto, "
        f"enquanto o [CA-8.5] estiver em remediação\n"
    )

    assert execucao.cells, "a execução não produziu célula nenhuma"
    assert len(execucao.cells) == grade_medida.size()
    # O executor mede o próprio tempo; conferir os dois é o que impede o relato de divergir da
    # execução — número de fora e número de dentro têm de bater dentro do ruído do relógio.
    assert abs(execucao.elapsed_seconds - parede) < max(2.0, 0.05 * parede), (
        f"o tempo do executor ({execucao.elapsed_seconds:.1f} s) diverge do medido aqui "
        f"({parede:.1f} s); um dos dois não está a medir a execução"
    )
    # **A asserção do teto passa pelo funil de `tests/orcamento.py` e não é escrita aqui (E16).**
    # Não é indireção decorativa: é o que permite a `test_orcamento_contrato.py` conferir, por
    # análise sintática, que nenhuma constante de teto vira comparação sem a guarda de carga antes
    # — a regra da especificação mecanizada para a CLASSE, em vez de lembrada para este item.
    afirma_orcamento(
        abertura,
        mili_por_celula,
        TETO_MILINUCLEO_MIN_POR_CELULA,
        unidade="milinúcleo-min/célula",
        contexto=(
            f"  (media + 3 desvios de TREZE repeticoes). Regime: "
            f"{regime['nucleos_nominais']} núcleos nominais, load "
            f"{load_inicial:.2f}->{load_final:.2f}, "
            f"{regime['mhz_medio']} MHz médios, governador {regime['governador']}.\n"
            "  ANTES DE TRATAR COMO REGRESSÃO: núcleo-minutos normaliza contagem de núcleos e "
            "NÃO frequência, nem a SESSÃO — a dispersão ENTRE sessões é 27x a de dentro de uma "
            f"(AR110). O modelo é {MODELO_DA_MAQUINA}; noutra máquina o teto precisa ser "
            "REMEDIDO, e não afrouxado (`[CA-3.8]`, regra 2 do README)"
        ),
    )


@pytest.mark.golden
def test_o_serrf_domina_o_custo_e_a_consequencia_de_reabri_lo_esta_declarada() -> None:
    """**O par causal do benchmark: com e sem `serrf`, e os desfechos divergem por ~30x.**

    Sem esta direção, o número da composição de derivação seria lido como *"a grade custa 1 min"*.
    Com o `serrf`, a mesma grade custa ~30x mais, e quem tiver lido só o primeiro número não
    saberia.

    **A frase *"e é o que ele diz hoje, com o `serrf` bloqueado"* saiu em 09/08/2026:** o
    `serrf` foi liberado, e a grade que a rota do multiverso dispara **é** a de 4.320 células. O
    par causal ficou mais necessário e não menos — antes o custo maior era hipotético, e agora é o
    que o usuário paga ao apertar o botão. **Terceira irmã deste arquivo na mesma varredura**, com
    o cabeçalho do módulo e a docstring do critério.

    A atribuição do critério é a floresta, e corrigir o passo 6 não destrava o critério nem,
    portanto, o método: **o que liberou o `serrf` foi a decisão de escopo do item 14**, não o
    item 22.

    O teste usa uma grade **mínima** de propósito: o que se mede é a razão entre os dois custos
    fixos, não o custo absoluto — e o fixo é o mesmo com 4 células ou com 4.320, que é justamente a
    propriedade que a projeção de 55 h errou.
    """
    from dataclasses import replace

    matriz = load_matrix(measurements_path(MODO), sample_list_path(MODO), MODO)
    base = GridDefinition()

    def minima(correcao: str) -> GridDefinition:
        return replace(
            base,
            corrections=(correcao,),  # type: ignore[arg-type]
            imputations=base.imputations[:1],
            sample_norms=base.sample_norms[:1],
            transformations=base.transformations[:1],
            scalings=base.scalings[:1],
            rsd_cutoffs=base.rsd_cutoffs[:1],
            missing_thresholds=base.missing_thresholds[:1],
            batch_options=base.batch_options[:1],
        )

    custos: dict[str, float] = {}
    for correcao in ("none", "loess", "serrf"):
        inicio = time.perf_counter()
        run_multiverse(matriz, minima(correcao), n_jobs=-1)
        custos[correcao] = time.perf_counter() - inicio

    sem_serrf = custos["none"] + custos["loess"]
    razao = (sem_serrf + custos["serrf"]) / sem_serrf

    print(
        f"\n[TO-8.4] custo FIXO por correção, grade mínima: "
        f"none {custos['none']:.1f} s | loess {custos['loess']:.1f} s | "
        f"serrf {custos['serrf']:.1f} s\n"
        f"  o `serrf` é {100 * custos['serrf'] / (sem_serrf + custos['serrf']):.0f}% do custo "
        f"total, e destravá-lo multiplica a grade por {razao:.0f}x — destrava pelo item 14, "
        f"decisão de escopo, e não pelo item 22\n"
    )

    assert custos["serrf"] > sem_serrf, (
        "o `serrf` deixou de dominar o custo. Se isso for real, a consequência de o desbloquear "
        "mudou "
        "e a declaração deste módulo precisa ser remedida — não é falha, é caducidade"
    )
    assert razao > 5.0, (
        f"a razão medida é {razao:.1f}x, e a declaração fala em ~30x. Divergência desta ordem "
        "significa que o custo do SERRF mudou; remeça e atualize a declaração"
    )
