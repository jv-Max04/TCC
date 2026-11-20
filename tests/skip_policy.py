"""Política de `skip` declarada.

**O problema.** A suíte tem dois regimes de `skip` legítimo: dado de referência
ausente e ferramenta externa indisponível. Isso significa que uma linha de
`797 passed, 3 skipped` é indistinguível de `797 passed, 2 skipped` mais um teste
que sumiu — e contar `passed` não distingue teste que passou de teste que deixou de
existir. O caso concreto: um teste escrito para provar que a âncora produz
`skip` estava sendo **pulado** em silêncio, e a saída parecia saudável.

**A guarda, em duas direções.** É a mesma família do teste de exaustividade da
partição de códigos de validação e do de paridade dos treze flags do `strict`:
um conjunto declarado num lugar, derivado de outro, e divergência reprovando
em qualquer direção.

1. **Em tempo de execução** (`pytest_sessionfinish` no `conftest`): todo `skip`
   observado tem de casar com um regime declarado aqui. `skip` fora da lista
   reprova a sessão, com o `nodeid` e o motivo na saída.
2. **Estaticamente** (`test_skip_policy.py`): todo arquivo de teste que contém
   chamada de `skip` tem de estar nomeado por algum regime. Sítio de `skip` novo
   sem declaração reprova no momento em que é escrito, e não só quando dispara.

**Por que uma lista e não "zero skips permitidos".** Os dois regimes são
legítimos e a regra do produto os exige — substituir dado de referência ausente por dado
fabricado é proibido, então `skip` é o desfecho correto. O que não é legítimo é
`skip` que ninguém declarou, porque é dele que se esconde atrás dos esperados.

**Por que não é lista de isenção disfarçada.** Isenção
suspende uma verificação; isto a **estreita**: cada regime nomeia a condição
externa que o justifica e o sítio que o emite. Regime cujo sítio desaparecer é
pego pela direção 2.
"""

import ast
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SkipRegime:
    """Um motivo declarado de `skip`, com a condição externa que o legitima."""

    name: str
    # Regex casado contra o motivo do `skip`, com `re.search`.
    pattern: str
    # Condição externa que torna este `skip` o desfecho correto.
    condition: str
    # Arquivos de teste que podem emitir este `skip`, relativos a `tests/`.
    sites: tuple[str, ...]


DECLARED_SKIP_REGIMES: tuple[SkipRegime, ...] = (
    SkipRegime(
        name="refino_nao_instalado",
        pattern=r"refino_ausente:",
        condition=(
            "O ponto de entrada `[project.scripts] refino` não está no ambiente. O "
            "`[CA-12.12]` compara byte a byte a saída da ROTA contra a saída da CLI, e a "
            "comparação exige a CLI instalada — `pip install -e .` a põe lá, e um ambiente "
            "que só tenha o pacote importável não a tem. **Regime próprio e não "
            "`reference_data_absent`**: ali falta DADO e aqui falta FERRAMENTA, e quem lê o "
            "`skip` procura em lugares diferentes (AR88)."
        ),
        sites=("test_api_artefatos.py",),
    ),
    SkipRegime(
        name="reference_data_absent",
        pattern=r"Arquivo de referência ausente:",
        condition=(
            "Os dados de referência não estão no disco. A §8.4 proíbe substituir por "
            "dado fabricado, então `skip` com o caminho esperado é o desfecho correto."
        ),
        sites=("conftest.py",),
    ),
    SkipRegime(
        name="reference_root_unresolved",
        pattern=r"dados de referência indisponíveis|dados de referência ausentes",
        condition=(
            "Nenhuma das duas raízes resolve. Mesmo regime do anterior, por outra "
            "porta: quem precisa dos dados para regenerar um fixture, não para ler um."
        ),
        sites=("golden/test_insample_medians.py", "test_reference_data.py"),
    ),
    SkipRegime(
        name="frozen_fixture_absent",
        pattern=r"ausente; (a fixture é versionada|gere por)",
        condition=(
            "A fixture congelada do oráculo do R ou da expectativa in-sample não está no "
            "disco. As que vieram do R são versionadas e não se regeneram aqui: o `skip` "
            "manda restaurá-las do repositório. As que têm gerador local nomeiam o script."
        ),
        sites=(
            "unit/test_boxplot_mask.py",
            "unit/test_boxplot_fixture.py",
            "golden/test_insample_medians.py",
            "golden/test_r_reproducibility.py",
            "golden/test_serrf_transcription.py",
            "golden/test_derived_gate_pipeline.py",
            "golden/test_loess_golden.py",
            # O cruzamento da régua fora-da-amostra contra o in-sample lê duas fixtures: a
            # régua, versionada, e a mediana in-sample, que tem gerador local. A razão do
            # `skip` é por fixture.
            "test_regua_fora_da_amostra.py",
        ),
    ),
    SkipRegime(
        name="integrity_manifest_absent",
        pattern=r"manifesto de integridade indisponível",
        condition=(
            "`scripts/reference_data.sha256` não está legível. Sem manifesto não há com "
            "o que comparar digest, e afirmar integridade sem referência seria pior."
        ),
        sites=(
            "golden/test_insample_medians.py",
            "golden/test_r_reproducibility.py",
            # A dispersao das 20 sementes custa ~147 min e nao entra na
            # selecao de commit. Ausente, o skip nomeia o script que a gera.
            "golden/test_serrf_agreement.py",
        ),
    ),
    SkipRegime(
        name="paralelismo_desligado",
        pattern=r"paralelismo indisponível: `effective_n_jobs\(-1\)` devolve 1",
        condition=(
            "O par causal que compara `n_jobs=1` com `n_jobs=-1` precisa de os dois valores "
            "resolverem para números DIFERENTES. Com `LOKY_MAX_CPU_COUNT=1` no ambiente, "
            "`joblib.effective_n_jobs(-1)` devolve **1** — medido em 06/09/2026 neste clone: "
            "12 sem a variável, 1 com ela — e o teste passa a comparar 1 contra 1. Ele PASSA, "
            "e não mede nada. "
            "A variável não é exotismo: é o contorno que torna a suíte completa executável "
            "neste clone, onde o arranque de workers do `loky` trava a execução. Ou seja, o "
            "regime dispara justamente na configuração em que o portão de facto corre — que é "
            "a pior altura para uma asserção ficar vácua em silêncio. "
            "O que este `skip` compra é a diferença entre «passou» e «não pôde ser medido», e "
            "essa diferença é a que `SerrfCorrector.identidade()` invoca ao EXCLUIR `n_jobs` "
            "da chave de reuso de correção: a exclusão é uma afirmação de "
            "bit-identidade, e ela precisa de um teste que a tenha de facto medido."
        ),
        # **O segundo sítio entrou em 23/10/2026**: o mesmo par causal,
        # para a floresta que conta como o `ranger`, com a mesma mensagem literal.
        sites=("unit/test_forest_speedup.py", "unit/test_serrf_floresta_ranger.py"),
    ),
    SkipRegime(
        name="openmp_com_uma_thread",
        pattern=r"sem OpenMP com mais de uma thread fora do ajuste",
        condition=(
            "O teste afirma que o `KMeans` corre com o OpenMP a UMA thread dentro do "
            "ajuste, e a planta — o `with threadpool_limits` removido — só separa onde o OpenMP "
            "tem mais de uma thread FORA dele. Numa máquina de um núcleo, ou num processo sem "
            "runtime de OpenMP carregado, a asserção passaria sem o mecanismo (asserção de "
            "ausência), e o `skip` é a diferença entre «passou» e «não pôde ser medido». "
            "Medido em 15/10/2026: 12 threads fora do ajuste neste clone e no WSL."
        ),
        sites=("unit/test_clustering.py",),
    ),
    SkipRegime(
        name="shallow_clone",
        pattern=r"clone raso",
        condition=(
            "Clone `--depth 1` não tem histórico. O conferidor do «Palpite» no congelado lê "
            "do histórico o artefato da versão anterior (`a488add`) para a âncora de poder dos "
            "testes dele; sem o histórico não há modelo velho com que comparar, e a âncora não "
            "se mede. O CI usa `fetch-depth: 0`, então este `skip` não dispara lá; é para clone "
            "raso de quem só quer rodar a suíte. Mesma família de AR32."
        ),
        sites=("test_declared_artifacts.py", "test_confere_palpite_no_congelado.py"),
    ),
    SkipRegime(
        name="document_absent",
        pattern=r"não existe$",
        condition=(
            "Um documento que o teste de coerência do Makefile inspeciona não existe. "
            "Verificar citação de alvo em documento ausente não é possível."
        ),
        sites=("test_makefile.py",),
    ),
    # **Regime NOVO em 08/08/2026 (E16), e ele é o único da lista cuja condição externa é a
    # MÁQUINA e não um artefato ausente.** Os outros dez pulam porque falta alguma coisa —
    # arquivo, contêiner, banco, manifesto. Este pula porque a coisa está lá e o **estado** dela
    # desqualifica a medição, e é por isso que a mensagem precisa dizer o número: sem ele,
    # `skipped` seria indistinguível de "faltou insumo", que é a leitura errada.
    SkipRegime(
        name="time_budget_load_contaminated",
        pattern=r"orçamento de tempo NÃO medido: load_per_core inicial",
        condition=(
            "A máquina está sob contenção acima do limite que a §8.2 declara — `load_per_core` "
            "inicial acima de 0,5 —, e o `[CA-3.8]` exige `skip` com motivo declarado ali, "
            "**nunca `pass`, nunca `fail`**. Núcleo-minutos normaliza a CONTAGEM de núcleos e "
            "não a DISPUTA por eles, então sob contenção a reprovação do teto não distingue "
            "regressão de máquina ocupada — e as duas pedem investigações opostas. "
            "A guarda estava DECLARADA e não implementada até 08/08/2026: medido, o `[TO-8.4]` "
            "reprovou com load_per_core de 0,78 e a mensagem de contenção era a mesma da de "
            "regressão. Vale para PORTÃO de orçamento e não para ARNÊS de derivação, que grava "
            "a carga e segue — censurar a amostra encolheria o desvio de que o teto sai "
            "(AR98, AR110)."
        ),
        sites=("orcamento.py",),
    ),
    SkipRegime(
        name="sem_endereco_externo",
        pattern=r"sem IPv4 não-loopback: nem pelo resolvedor de nomes nem pela rota",
        condition=(
            "Nenhuma das duas vias da sonda devolveu IPv4 não-loopback: nem o resolvedor de "
            "nomes nem o truque da rota. Sem `externo`, ligar um segundo socket a "
            "`(externo, mesma_porta)` não distingue `127.0.0.1` de `0.0.0.0`, e não há o que "
            "medir. **A mensagem nomeia a DISJUNÇÃO de propósito** — a primeira versão dizia "
            "*«máquina sem endereço IPv4 não-loopback»* e isso era causa **afirmada**: numa "
            "máquina com endereço "
            "mas sem rota, o `skip` disparava com motivo falso, que é o marcador a absorver outra "
            "causa (AR72). **O portão de AST sobre a fonte NÃO pula**, e é ele que carrega a "
            "afirmação nas duas direções: este regime cobre apenas a prova de runtime. Fase 12, U1."
        ),
        sites=("test_desktop_servidor.py",),
    ),
    # O grupo `desktop` e OPCIONAL: PySide6 custa
    # 649 MB numa venv Linux (medido em 22/08/2026), e po-lo em `dev` faria toda instalacao de
    # desenvolvimento pagar por uma interface que a maioria nao toca. Sem PySide6 e `pytest-qt` nao
    # ha janela a exercitar, e `skip` e o desfecho correto.
    #
    # **A guarda e `tests/qt_guard.membros_ausentes`, e ela e PURA** — o ambiente entra por
    # argumento. E isso que permite exercitar a DECISAO nesta maquina, que tem o grupo instalado.
    #
    # **A frase do `skip` NAO nasce na guarda, e a divisao e mecanica:** o extractor deste arquivo
    # colhe constantes de string, e `pytest.skip(variavel)` nao entrega literal nenhum — a primeira
    # versao levou a reprovacao de "padrao sem emissao". A guarda decide QUEM FALTA; o sitio e que
    # frase-a.
    #
    # **E o `skip` foi rodado SEM o marcador**, o que nesta maquina exige um plugin de `pytest` que
    # esconde os metadados de PySide6 antes da coleta: `1 skipped` com esta razao contra `21 passed`
    # sem o plugin, as duas execucoes na mesma funcao de teste (`test_qt_guard.py`).
    #
    # **O motivo nomeia QUEM FALTA e nao uma causa suposta**, que e a licao do irmao acima.
    #
    # E este `skip` NAO absorve instalacao quebrada, porque `test_dependency_pins.py` passou a
    # tratar grupo parcialmente instalado como DEFEITO. Sem essa atomicidade, meio grupo produziria
    # `skipped` — o desfecho que a politica ensina a ler como legitimo.
    SkipRegime(
        name="grupo_desktop_ausente",
        pattern=r"grupo opcional `desktop` ausente: falta ",
        condition=(
            "PySide6 ou `pytest-qt` não estão instalados, então não existe janela Qt a exercitar. "
            "O grupo é opcional por decisão (KTD6): 649 MB medidos numa venv Linux, e quem só quer "
            "a biblioteca ou a CLI não os paga. **O motivo enumera os membros ausentes medidos**, "
            "nunca uma hipótese sobre por que faltam. Fase 12, U3 do aplicativo nativo."
        ),
        # **Um sítio por arquivo de janela, e a lista cresceu duas vezes.** O regime é o
        # mesmo — o grupo opcional ausente — e a mensagem é literalmente a mesma; o que muda é
        # haver mais um arquivo. O portão apanhou uma omissão em `make check`, que é o desfecho
        # certo: `skip` sem regime declarado é falha à espera da condição externa, e passaria em
        # toda máquina onde o grupo está instalado. **O sítio seguinte entrou ANTES de o portão o
        # cobrar**, e a diferença é um ciclo de 18 minutos.
        # **Mais TRES sitios em 07/09/2026, e os tres entraram ANTES de o portao os cobrar** —
        # os modulos dos itens 3, 9 e 11 da passagem humana pela janela. Mesmo regime, mesma
        # mensagem literal; o que muda e haver mais tres arquivos com janela a exercitar.
        # **Mais DOIS sitios em 17/09/2026, e os dois estavam em falta ha tempo.** Eles nao
        # foram apanhados porque este portao estava **sempre vermelho** nesta plataforma: comparava
        # `golden/x.py` declarado contra `golden\x.py` medido, acusava quinze arquivos legitimos, e
        # um portao que acusa sempre deixa de ser lido. **O portao que sempre reprova escondeu duas
        # lacunas reais** — e foi consertar o separador que as tornou visiveis.
        # **Mais UM sitio em 25/09/2026**, `test_desktop_execucao_em_curso.py`, e este foi
        # apanhado pelo portao na linha de base antes do merge, que e o desfecho certo.
        # **E mais UM em 25/09/2026**, `test_desktop_tela.py`, que entrou junto com o
        # conserto da busca da tela.
        # **E mais UM em 23/10/2026**, `test_desktop_palpite.py`, apanhado por este portão
        # na régua de quatro seleções, antes do commit.
        # **E mais SEIS em 06/11/2026**, os testes da interface do colega. O clone dele não
        # tinha `serrf/`, e o portão não corria lá; entram na junção, antes de o portão os cobrar.
        # **E mais UM em 07/11/2026**, `test_desktop_curva_do_loess.py`, a figura da curva
        # do LOESS com o envelope, apanhado por este portão na integração.
        # **E mais UM em 07/11/2026**, `test_desktop_intervalo_das_barras.py`, a caixa do
        # intervalo por barra e o traço no painel da correção, apanhado por este portão antes do
        # commit.
        # **E mais UM em 07/11/2026**, `test_desktop_carimbo_da_matriz.py`, a resposta de
        # uma matriz que chega depois da troca para outra, declarado antes do commit.
        sites=(
            "test_cancelamento_de_trabalhos.py",
            "test_contrato_janela_api.py",
            "test_desktop_aparencia.py",
            "test_desktop_avisos.py",
            "test_desktop_cabecalhos.py",
            "test_desktop_carimbo_da_matriz.py",
            "test_desktop_curva_do_loess.py",
            "test_desktop_execucao_em_curso.py",
            "test_desktop_expansao.py",
            "test_desktop_figuras.py",
            "test_desktop_intervalo_das_barras.py",
            "test_desktop_janela.py",
            "test_desktop_lovable.py",
            "test_desktop_marcadores.py",
            "test_desktop_palpite.py",
            "test_desktop_qualidade.py",
            "test_desktop_recorte.py",
            "test_desktop_remocao.py",
            "test_desktop_roda.py",
            "test_desktop_sondagem.py",
            "test_desktop_tela.py",
            "test_desktop_tema.py",
        ),
    ),
    # **Regime de 17/09/2026, e ele DESFAZ quatro reprovacoes permanentes da linha de base.**
    # As quatro tinham UMA causa so — `mkdir(mode=0o500)` e no-op neste sistema de arquivos —, e a
    # linha de base atribuia duas delas a «`subprocess` que nao resolve aqui», o que era falso: nao
    # ha `subprocess` naqueles dois testes. O veredito acusava o alvo errado.
    #
    # **A guarda pergunta pela CAPACIDADE e nao pela plataforma** (`tests/permissao_capability.py`):
    # um volume montado sem suporte a permissao falha igual num Linux, e um `sys.platform` deixaria
    # esse caso a reprovar sem explicacao. E os dois motivos sao SEPARADOS porque um sistema pode
    # honrar o modo de um arquivo e nao a permissao de um diretorio — colapsa-los tornaria
    # indistinguiveis dois motivos de `skip`, que e o defeito que esta politica existe para impedir.
    # **Regime irmao do de baixo, e da mesma varredura.** `os.getloadavg` e POSIX e nao
    # existe no Windows, e OITO testes de contrato reprovavam com `AttributeError` — que nao e
    # desfecho, e medicao que nao aconteceu. Sem load average a guarda de carga nao consegue
    # fazer o trabalho dela, que e distinguir *"o codigo esta lento"* de *"a maquina esta ocupada"*:
    # aprovar afirmaria ociosidade que ninguem mediu, e reprovar chamaria de regressao
    # o que pode ser contencao. **A terceira saida e pular com o motivo nomeado.**
    #
    # **NAO cobre `_regime()` do benchmark**, e a distincao e do proprio docstring dele: aquele
    # dicionario DESCREVE e nao DECIDE, entao ali a indisponibilidade entra como VALOR — a mesma
    # cadeia que `scripts/mede_etapas_do_pipeline.py` ja gravava — e nao como `skip`.
    # **Terceiro regime da varredura de 17/09/2026, e o unico que pula por QUEBRA.**
    # O braco de CONTROLO do par causal corre `test_desktop_janela.py` inteiro — 256 testes — e o
    # processo morre a meio com a corrupcao de heap do Qt que `docs/linha-de-base-windows.md` mede
    # como taxa. Sem controlo nao ha par, e ler a morte como *"a guarda disparou sem a planta"*
    # seria atribuir ao alvo um veredito sobre o ambiente.
    SkipRegime(
        name="sitio_de_janela_quebrou",
        pattern=r"sítio de janela QUEBROU no braço de controlo",
        condition=(
            "O subprocesso do braço de controlo morreu sem imprimir sumário do pytest. **A "
            "condição exige AUSÊNCIA de sumário**, e não um código de saída: uma reprovação real "
            "imprime `failed` e segue para as asserções, um sucesso imprime `passed` e idem — só "
            "o processo que morre calado pula. O mesmo arquivo passa inteiro em fatias, e é assim "
            "que a suíte o mede. Quebra não é desfecho."
        ),
        sites=("test_qt_guard.py",),
    ),
    SkipRegime(
        name="load_average_indisponivel",
        pattern=r"load average indisponível:",
        condition=(
            "O sistema não expõe `os.getloadavg`. A guarda de orçamento da §8.2 não distingue "
            "regressão de máquina ocupada sem ele, então o teto de tempo deixa de ser portão "
            "aqui. **O par causal continua exercitável**: `abre_medicao_de_orcamento(medido=...)` "
            "passa a carga como argumento e não toca a máquina, então as duas direções da guarda "
            "correm em qualquer sistema — o que pula é só o caminho que LÊ a carga real."
        ),
        sites=("orcamento.py", "test_orcamento_contrato.py"),
    ),
    SkipRegime(
        name="permissao_posix_ignorada",
        pattern=r"permissão POSIX ignorada:",
        condition=(
            "O sistema de arquivos aceita e IGNORA `mkdir(mode=0o500)` ou os bits de modo de um "
            "arquivo. Os cenários «diretório sem permissão de escrita» e «o banco nasce em 0600» "
            "não se constroem ali, e o desfecho não é reprovação — é medição que não aconteceu. "
            "Os quatro testes correm inteiros em qualquer sistema POSIX, e o predicado "
            "tenta ESCREVER em vez de ler o modo de volta, porque é a escrita que o cenário "
            "precisa que falhe."
        ),
        sites=(
            "test_desktop_arranque.py",
            "test_desktop_banco.py",
            "test_persistencia_sqlite.py",
        ),
    ),
    SkipRegime(
        name="rsd_anchor_failed",
        pattern=r"âncora do \[CA-2\.1\] reprovou",
        condition=(
            "O instrumento de RSD não fechou contra o `QC - RSD.csv`, e os dependentes "
            "reportam `skip` e não `fail`: doze reprovações causadas por um "
            "instrumento quebrado escondem qual é o defeito."
        ),
        sites=("golden/test_insample_medians.py",),
    ),
    SkipRegime(
        name="fora_do_windows",
        pattern=r"só no Windows:",
        condition=(
            "O lançador do R das vias põe o filho num Job Object do Windows, e é ele que "
            "encerra a árvore de processos no cancelamento. **O regime cobre só os dois testes do "
            "NETO** — `test_cancelar_encerra_o_filho_e_o_neto_e_nao_sobra_temporario` e "
            "`test_par_causal_sem_o_job_o_neto_sobrevive_ao_kill_do_filho` —, que contam processo "
            "vivo por `OpenProcess` e matam o neto com `taskkill`; o `skipif` está neles e não no "
            "módulo, e o resto do arquivo corre em qualquer sistema. O executável é Windows, e é "
            "lá que isto se mede. Em outra plataforma o lançador encerra só o filho, e o par "
            "causal do neto não tem como ser medido. "
            "**E cobre também os dois testes do Palpite que comparam o LOESS "
            "e o SERRF com os vetores da F2 da rodada JH** — "
            "`test_nos_reais_os_escores_do_serrf_e_do_loess_sao_os_dos_vetores_gravados` e "
            "`test_os_escores_do_serrf_e_do_loess_da_rota_local_sao_os_dos_vetores_gravados` —, "
            "que afirmam igualdade exacta. Os vetores foram gravados no Windows, e num contêiner "
            "Linux, com o mesmo código, o LOESS fica a 1 ULP da mediana gravada: a "
            "igualdade vale numa plataforma, e é lá que ela se mede. Os dois testes do `none`, "
            "nos mesmos arquivos, correm em qualquer sistema."
        ),
        sites=(
            "test_enriquecimento_r.py",
            "unit/test_palpite_medido.py",
            "test_confere_palpite_no_congelado.py",
        ),
    ),
)


def undeclared_reason(reason: str) -> bool:
    """`True` quando nenhum regime declarado explica este motivo de `skip`."""
    return not any(re.search(regime.pattern, reason) for regime in DECLARED_SKIP_REGIMES)


# ---------------------------------------------------------------------------
# O buraco que as duas direções não fechavam
# ---------------------------------------------------------------------------
#
# **As duas direções acima validam a FORMA do motivo, não a existência do que ele
# promete.** O regime `frozen_fixture_absent` casa "ausente; gere por X" e declara
# o sítio; nada conferia que `X` existe. E houve exatamente esse caso:
# `test_loess_golden.py` pulava citando `scripts/gen_loess_golden.py`, que não
# havia sido escrito. A política aceitou sem reclamar, porque o motivo casava.
#
# **Por que isso é pior que um `skip` não declarado.** `skip` que nomeia mecanismo
# ausente é `skip` que **nunca vira execução**: dezoito testes que parecem existir
# e nunca rodam, com a linha de saída dizendo `skipped` — que é o desfecho que a
# própria política ensina a ler como legítimo. É a forma do `make test-golden`,
# que existia desde o início e nunca pôde passar.
#
# O escopo é `pytest.skip`, e não toda mensagem de teste, porque é aqui que o modo
# de falha mora: mensagem de `assert` citando script inexistente aparece quando o
# teste reprova, e alguém está lendo. `skip` não tem leitor.

_SCRIPT_CITATION = re.compile(r"(scripts/[A-Za-z0-9_./-]+\.(?:py|sh|R))")


def _skip_messages(source: str) -> list[str]:
    """Textos passados a `pytest.skip(...)`, incluindo os pedaços literais de f-string.

    Por AST e não por linha: mensagem de `skip` quebrada em duas linhas escaparia
    de uma varredura por linha, e é o formato mais comum aqui — a mensagem carrega
    caminho e explicação e não cabe em 88 colunas.
    """
    out: list[str] = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Attribute) and func.attr == "skip"):
            continue
        pieces: list[str] = []
        for argument in [*node.args, *(kw.value for kw in node.keywords)]:
            for inner in ast.walk(argument):
                if isinstance(inner, ast.Constant) and isinstance(inner.value, str):
                    pieces.append(inner.value)
        if pieces:
            out.append(" ".join(pieces))
    return out


def missing_script_citations(repo: Path, sources: list[Path]) -> list[tuple[str, str]]:
    """Pares `(arquivo, caminho citado)` em que o `skip` promete script que não existe.

    Devolve a lista e não um booleano de propósito: `False` não distingue "nenhuma
    citação" de "todas as citações válidas", e as duas pedem ação oposta — a
    primeira é varredura que não achou nada e merece suspeita.
    """
    out: list[tuple[str, str]] = []
    for path in sources:
        for message in _skip_messages(path.read_text(encoding="utf-8")):
            for cited in _SCRIPT_CITATION.findall(message):
                if not (repo / cited).is_file():
                    out.append((path.name, cited))
    return out


_XFAIL_STRICT_ERRO = (
    "`xfail` sem `strict=True` em {sitio}: {alvo}. `xfail` não estrito **passa em "
    "silêncio** quando o teste começa a passar, e é a família dos dezoito testes "
    "de golden pulando por citar gerador inexistente, onde `skipped` era o desfecho que "
    "a política ensina a ler como legítimo. Com `strict=True`, `XPASS` reprova, e é ele "
    "o sinal de que a dependência chegou e o marcador tem de ser REMOVIDO."
)

_XFAIL_IMPERATIVO_ERRO = (
    "`pytest.xfail()` IMPERATIVO em {sitio}: {alvo}. A forma imperativa **não aceita "
    "`strict`**, então nunca reporta `XPASS` - e a compensação que autorizou tirar "
    "`xfail` do registro de regimes foi justamente *todo `xfail` é estrito*. Pior: ela "
    "interrompe o teste onde é chamada, então uma chamada ANTES da medição faz a "
    "asserção **nunca executar**, e a razão passa a descrever um número que o teste não "
    "confere - que é AR72. Use `pytest.param(..., marks=pytest.mark.xfail(strict=True, "
    "reason=...))` quando o marcador vale para um subconjunto dos parâmetros."
)


def non_strict_xfail_sites(sources: list[Path]) -> list[tuple[str, str]]:
    """Marcadores `xfail` sem `strict=True`, um por par (arquivo, teste).

    **Existe porque `xfail` foi EXCLUÍDO do registro de regimes de `skip`**, e a
    exclusão abriria um buraco sem esta guarda. O pytest reporta `xfail` com
    `report.skipped` verdadeiro, então o gancho da guarda o via como `skip` não
    declarado; mas `xfail` não é `skip`, e forçá-lo no registro faz a direção estática
    reprovar - o sítio declarado não emite `skip` nenhum.

    A troca é honesta só com a compensação: `xfail` sai do registro, e em troca todo
    `xfail` tem de ser estrito. Estrito é o que o torna não silencioso.

    **ESCOPO: DECORADOR, e dizê-lo é o conserto de um buraco medido.** Esta
    função percorre `decorator_list` de cada função, então ela **não** vê
    `pytest.xfail()` chamado no CORPO do teste - a forma imperativa, que não aceita
    `strict` e por isso nunca reporta `XPASS`. O docstring anterior afirmava exigir
    `strict=True` em **todo** `xfail`, e havia um sítio imperativo escapando desde que
    foi escrito. `imperative_xfail_sites` fecha a outra metade, e as duas são
    separadas porque as ações são diferentes: acrescentar o argumento, contra
    reestruturar em decorador.
    """
    achados: list[tuple[str, str]] = []
    for caminho in sources:
        arvore = ast.parse(caminho.read_text(encoding="utf-8"))
        for no in ast.walk(arvore):
            if not isinstance(no, ast.FunctionDef | ast.AsyncFunctionDef):
                continue
            for decorador in no.decorator_list:
                if not isinstance(decorador, ast.Call):
                    continue
                if not _e_marcador_xfail(decorador.func):
                    continue
                estrito = any(
                    palavra.arg == "strict"
                    and isinstance(palavra.value, ast.Constant)
                    and palavra.value.value is True
                    for palavra in decorador.keywords
                )
                if not estrito:
                    achados.append((caminho.name, no.name))
    return achados


def imperative_xfail_sites(sources: list[Path]) -> list[tuple[str, str]]:
    """`pytest.xfail()` chamado no CORPO de um teste, um por par (arquivo, teste).

    **A metade que `non_strict_xfail_sites` não vê, e ela custou um sítio inerte.**
    Aquela função percorre decoradores; esta percorre o corpo. O defeito medido:
    `pytest.xfail()` chamado **antes** de `generate()` num teste do gerador sintético, então
    para aquele modo a asserção **nunca rodava** — e a razão do marcador citava um
    desvio de 33,3% que o teste não media. O número estava certo, e estava certo por
    sorte.

    **Não há caso legítimo a permitir, e é por isso que a guarda é incondicional.**
    Marcador que vale para um subconjunto de parâmetros se escreve com
    `pytest.param(..., marks=...)`, que é a forma já em uso. A forma imperativa não tem
    como ser estrita, então não tem como ser lida.
    """
    achados: list[tuple[str, str]] = []
    for caminho in sources:
        arvore = ast.parse(caminho.read_text(encoding="utf-8"))
        for no in ast.walk(arvore):
            if not isinstance(no, ast.FunctionDef | ast.AsyncFunctionDef):
                continue
            for interno in ast.walk(no):
                if not isinstance(interno, ast.Call):
                    continue
                if _e_xfail_imperativo(interno.func):
                    achados.append((caminho.name, no.name))
                    break
    return achados


def _e_xfail_imperativo(func: ast.expr) -> bool:
    """`pytest.xfail` — e **não** `pytest.mark.xfail`, que é o decorador legítimo.

    A distinção é o `value` ser `mark`: sem ela a guarda acusaria todo decorador, e um
    portão que sempre reprova não informa nada.
    """
    if not (isinstance(func, ast.Attribute) and func.attr == "xfail"):
        return False
    dono = func.value
    return not (isinstance(dono, ast.Attribute) and dono.attr == "mark") and not (
        isinstance(dono, ast.Name) and dono.id == "mark"
    )


def _e_marcador_xfail(func: ast.expr) -> bool:
    """`pytest.mark.xfail` ou `mark.xfail`, nas duas formas de import."""
    return isinstance(func, ast.Attribute) and func.attr == "xfail"
