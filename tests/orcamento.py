"""A guarda de carga para orçamento de tempo.

**A regra estava DECLARADA e não IMPLEMENTADA, e a demonstração é de 08/08/2026.** A especificação
declara, na linha de *Orçamento de tempo de parede*: *"tempo **e** núcleo-minutos **e**
`load_per_core` registrado nas duas pontas; `skip` declarado acima de 0,5 no início"*.
O critério de aceitação repete com mais força: *"a medição produz **`skip` com motivo
declarado** — nunca `pass`, nunca `fail` — quando `load_per_core` inicial excede 0,5"*.

Medido: `load_per_core` não existia em lugar nenhum da árvore. O benchmark do multiverso gravava
`load_1min` nas duas pontas, imprimia, escrevia o aviso em prosa na mensagem de falha — e
**assertava de todo modo**. Numa execução com `make test-docker` em paralelo, `load_per_core`
inicial de **0,78**, ele reprovou com 13,508 contra o teto de 8,56. *"O código está lento"* e
*"a máquina está ocupada"* saíram **indistinguíveis**.

**O que a guarda compra, e não é o teste passar.** É a medição diária ser **interpretável**:
enquanto o portão puder reprovar por contenção, nenhum ponto da série diária responde à
pergunta que ela existe para responder, porque o desfecho não separa regime de regressão.

## A distinção que torna a regra precisa: PORTÃO contra ARNÊS

A guarda **não** vale para toda medição de tempo. Vale para a que **decide**:

| Papel | Carga contaminada faz o quê |
|---|---|
| **portão de orçamento** — afirma um teto | **`skip` declarado**. Reprovar mediria a máquina |
| **arnês de derivação** — estima a distribuição | **grava e segue**. Pular censura a amostra |

**Pular no arnês seria censurar a amostra.** O teto é `média + 3
desvios` sobre uma amostra que precisa **cobrir o regime**; descartar os pontos de carga
alta encolhe o desvio e produz teto mais apertado — que é falhar na direção que já custou
duas remedições. O arnês
**registra** `load_per_core` para que a dispersão seja atribuível; ele nunca a censura.

O corolário é que a guarda mora aqui e não dentro de `_regime()`: `_regime()` **descreve** a
máquina e é lido pelos dois papéis; a guarda **decide** e é chamada só pelo primeiro.

## Os quatro desfechos, e todos são exercitados

`abre_medicao_de_orcamento` tem dois — segue, ou `skip` — e `afirma_orcamento` tem dois —
dentro do teto, ou estoura. São **quatro**, e o portão de contrato exige que os quatro
tenham sido vistos. As funções puras existem para isso: a planta entra como **argumento**, e
negativo cuja planta é argumento não consegue ficar inerte.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import pytest

# **O limiar é 0,5 e ele é CITADO e não escolhido.** Vem da especificação.
# Mudá-lo aqui seria emendar a especificação por dentro de um
# módulo de teste, e a regra 2 do README é sobre exatamente isso — o número não é uma alavanca.
LIMITE_LOAD_PER_CORE = 0.5

# O detalhe do `skip`; o discriminador vai literal em cada sítio, pelo motivo de `skip_policy.py`
# ler o texto estaticamente — ver `tests/permissao_capability.py`.
DETALHE_DO_LOAD = (
    "`os.getloadavg` é POSIX e não existe neste sistema, então a guarda da §8.2 não distingue "
    "regressão de máquina ocupada e o orçamento não é portão aqui"
)


def load_medivel() -> bool:
    """`True` quando este sistema sabe responder o load average de 1 minuto.

    **`os.getloadavg` é POSIX e não existe no Windows**, e a ausência não é opinião: o atributo não
    está no módulo. Dois scripts deste repositório já a declaravam por extenso —
    `scripts/mede_etapas_do_pipeline.py` e `scripts/mede_passos_do_serrf.py` escrevem
    *"INDISPONIVEL - `os.getloadavg` e POSIX e nao existe no Windows"* no regime que gravam —, e o
    arnês de orçamento não. Oito testes de contrato reprovavam com `AttributeError`, e
    `AttributeError` **não é desfecho**: é medição que não aconteceu.

    **A pergunta é pela CAPACIDADE e não pela plataforma**, pelo mesmo motivo de
    `tests/permissao_capability.py`: `hasattr` responde por qualquer sistema, e um `sys.platform`
    deixaria de fora o que ainda não existe.
    """
    return hasattr(os, "getloadavg")


def load_per_core(nucleos: int | None = None) -> float:
    """Load average de 1 min dividido pela contagem nominal de núcleos.

    **É a quantidade que a regra nomeia, e o `load_1min` sozinho não a substitui.** Load 9 é
    contenção severa em 4 núcleos e folga em 64; núcleo-minutos normaliza a **contagem** de
    núcleos e não a **disputa** por eles, e é justamente esse vão que a guarda cobre.
    """
    # **Levanta em vez de devolver um número inventado.** Um `0.0` aqui diria *"a máquina está
    # ociosa"* a quem pergunta *"a medição é confiável?"* — e essa é a resposta que faz a guarda
    # aprovar uma medição contaminada. Quem chama pergunta por `load_medivel()` antes.
    if not load_medivel():
        raise RuntimeError(
            "load average indisponível: `os.getloadavg` é POSIX e não existe neste sistema. "
            "Pergunte por `load_medivel()` antes — a guarda de orçamento não sabe distinguir "
            "regressão de máquina ocupada sem ele"
        )
    n = nucleos if nucleos is not None else (os.cpu_count() or 0)
    if n <= 0:  # pragma: no cover - máquina sem contagem de núcleos legível
        return 0.0
    return os.getloadavg()[0] / n


def contaminado(medido: float, limite: float = LIMITE_LOAD_PER_CORE) -> bool:
    """`True` quando a carga inicial desqualifica a medição como portão.

    Função pura sobre dois floats **de propósito**: é ela que os negativos plantados
    exercitam, e a planta entra como argumento em vez de exigir que a máquina esteja
    ocupada, e assim o negativo não consegue ficar inerte.
    """
    return medido > limite


@dataclass(frozen=True)
class AberturaDeOrcamento:
    """A leitura de carga do início, guardada para viajar até a asserção.

    Existe como objeto, e não como float solto, porque a regra exige o registro **nas duas
    pontas**: quem abre tem de conseguir provar que abriu, e a assinatura de
    `afirma_orcamento` exigir esta instância é o que impede a segunda ponta de ser escrita
    sem a primeira: exige-se na assinatura o que não se pode adivinhar.
    """

    load_per_core: float
    nucleos: int
    limite: float


def detalhe_da_contaminacao(abertura: AberturaDeOrcamento) -> str:
    """A parte VARIÁVEL da mensagem do `skip` — pura, para ser conferível sem ocupar a máquina.

    **O cabeçalho fica no sítio do `skip` e não aqui, e isso foi FORÇADO pelo portão (E16).**
    `test_skip_policy.py` exige que algum sítio declarado emita, como **literal**, o padrão do
    regime; a primeira versão passava `motivo_da_contaminacao(abertura)` — chamada de função —
    e o extrator por AST não via constante nenhuma. Ele reprovou com *"padrão sem emissão é
    autorização larga que ninguém exerce"*, que é exactamente o defeito: um regime declarado
    que nenhum sítio pode disparar autoriza sem nunca ser usado.

    O corte fica aqui e não numa constante compartilhada de propósito — literal no sítio é o
    que o portão sabe ler, e duas cópias do cabeçalho divergiriam em silêncio.
    """
    return (
        f"de {abertura.load_per_core:.2f} excede o limite de {abertura.limite:.2f} "
        # **O load NU é derivado da abertura e não relido da máquina.** Relê-lo aqui quebrava com
        # `AttributeError` em sistema sem `os.getloadavg` — e, pior, tornava esta função IMPURA
        # depois de a docstring acima prometer que ela é pura «para ser conferível sem ocupar a
        # máquina». O número derivado é o da abertura, que é o que a mensagem diz ter medido.
        f"({abertura.load_per_core * abertura.nucleos:.2f} de load em "
        f"{abertura.nucleos} núcleos nominais). "
        "A §8.2 e o `[CA-3.8]` exigem `skip` com motivo declarado — nunca `pass`, nunca "
        "`fail` — porque núcleo-minutos normaliza a CONTAGEM de núcleos e não a DISPUTA "
        "por eles: sob contenção, 'o código está lento' e 'a máquina está ocupada' "
        "produzem o mesmo vermelho e pedem investigações opostas. "
        "Rode com a máquina ociosa e leia de novo."
    )


def abre_medicao_de_orcamento(
    limite: float = LIMITE_LOAD_PER_CORE,
    *,
    medido: float | None = None,
) -> AberturaDeOrcamento:
    """Lê a carga, **pula** se ela desqualificar a medição, e devolve a abertura.

    Chamada **antes** de a medição começar, e o "antes" é o conteúdo: pular depois de gastar
    a execução tem o custo sem o benefício, e a regra diz *"no início"*.

    `medido` existe para o par causal: a planta entra como argumento e as duas direções são
    exercitáveis sem ocupar a máquina de quem roda a suíte.
    """
    nucleos = os.cpu_count() or 0
    # **Sem load average, a guarda não tem como fazer o trabalho dela**, e o trabalho é
    # distinguir *"o código está lento"* de *"a máquina está ocupada"*. Aprovar a
    # medição aqui seria afirmar ociosidade que ninguém mediu; reprovar seria chamar de regressão
    # o que pode ser contenção. **A terceira saída é a honesta:** pular, com o motivo nomeado.
    # O `medido` explícito continua a passar — é a planta do par causal, e ela não precisa da
    # máquina para nada.
    if medido is None and not load_medivel():
        pytest.skip(f"load average indisponível: {DETALHE_DO_LOAD}")
    lido = medido if medido is not None else load_per_core(nucleos)
    abertura = AberturaDeOrcamento(load_per_core=lido, nucleos=nucleos, limite=limite)
    if contaminado(lido, limite):
        pytest.skip(
            "orçamento de tempo NÃO medido: load_per_core inicial "
            + detalhe_da_contaminacao(abertura)
        )
    return abertura


def mensagem_de_estouro(
    medido: float,
    teto: float,
    *,
    unidade: str,
    abertura: AberturaDeOrcamento,
    fechamento: float,
    contexto: str,
) -> str:
    """A mensagem da reprovação, pura para que o portão de contrato a possa conferir."""
    return (
        f"{medido:.3f} {unidade} contra o teto de {teto} {unidade}.\n"
        f"  carga: load_per_core {abertura.load_per_core:.2f} -> {fechamento:.2f}, "
        f"limite da guarda {abertura.limite:.2f} em {abertura.nucleos} núcleos nominais — "
        "a guarda da §8.2 deixou passar, então a contenção NÃO é a explicação disponível.\n"
        f"{contexto}"
    )


def afirma_orcamento(
    abertura: AberturaDeOrcamento,
    medido: float,
    teto: float,
    *,
    unidade: str,
    contexto: str,
) -> float:
    """A ponta de fechamento: lê a carga outra vez, afirma o teto, devolve o fechamento.

    **É o ÚNICO sítio em que um teto de orçamento vira asserção**, e isso é o que torna a
    regra mecanizável em vez de lembrável: `tests/test_orcamento_contrato.py` confere
    por análise sintática que nenhuma constante de teto é comparada fora daqui, e que quem
    chama esta função chamou `abre_medicao_de_orcamento` antes. Sem o funil, a regra voltaria
    a valer para o item que alguém lembrou e não para a classe — que é o defeito que E16
    registra.
    """
    # A abertura já pulou se o load fosse indisponível, então chegar aqui sem ele só acontece com
    # `medido` plantado — e nesse caso o fechamento repete a planta em vez de tocar a máquina.
    fechamento = load_per_core(abertura.nucleos) if load_medivel() else abertura.load_per_core
    assert medido < teto, mensagem_de_estouro(
        medido,
        teto,
        unidade=unidade,
        abertura=abertura,
        fechamento=fechamento,
        contexto=contexto,
    )
    return fechamento
