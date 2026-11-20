"""O contrato da JANELA contra a API.

O portão confere o fio entre a janela Qt e a API interna nas duas direções. Na de saída, todo campo
que a janela envia é aceito pela rota, e todo campo que a rota aceita e a janela não envia está
declarado com o motivo. Na de entrada, toda chave que a janela lê de uma resposta é declarada pelo
esquema de onde ela vem. O lado do esquema é `contratos_das_rotas`, em `tests/contrato_de_rotas.py`:
ele lê o OpenAPI e devolve `(método, rota) -> ContratoDeRota(entrada, saida, devolve_json)`.

## Na direção de saída, os construtores são CHAMADOS

A janela tem métodos, e um método pode ser chamado. Então a direção de saída não extrai nada: ela
constrói os painéis e chama `campos_base` — com a cadeia de `EstadoDasEtapas` que a Análise usa —,
`arquivos_para_envio` e `campos_para_envio`, e compara o que eles **de facto devolvem** com o que a
rota aceita.

Isso não é conveniência. Um dicionário construído à mão pode conter o que a produção nunca envia, e
deixar um caminho de produção inalcançável com a suíte verde. Chamar o construtor real fecha essa
distância por construção.

## A partição das LEITURAS é total e FECHADA, e é isso que a torna um portão

Um portão que verificasse *"as chaves que eu reconheço estão no esquema"* seria satisfeito por um
extrator que reconhecesse **zero** chaves. Então a forma é a inversa: **toda** base de leitura dos
módulos de produção da janela cai em exactamente uma classe declarada, e base nova **reprova** até
ser classificada. É uma lista de permissão aplicada a nomes de variável em vez de a formas de
sintaxe.
"""

from __future__ import annotations

import ast
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from refino.api.app import build_app
from refino.metrics.report import QualityReport
from tests.pedido_da_tela import campos_da_tela
from tests.qt_guard import (
    garante_plataforma_offscreen,
    membros_ausentes,
    sonda_o_ambiente,
)

_FALTAM = membros_ausentes(sonda_o_ambiente())
if _FALTAM:  # pragma: no cover — a decisão é exercitada em `tests/test_qt_guard.py`, com planta
    pytest.skip(
        f"grupo opcional `desktop` ausente: falta {', '.join(_FALTAM)}. "
        'Instale com `uv pip install -e ".[desktop]"` para exercitar os testes de janela.',
        allow_module_level=True,
    )

garante_plataforma_offscreen()

from refino.desktop.etapas import (  # noqa: E402
    ENTRADAS,
    ETAPAS,
    IDENTIDADE_DA_MATRIZ,
    ROTA_DA_ETAPA,
    EstadoDasEtapas,
    campos_da_rota,
    fotografia_da_etapa,
)
from refino.desktop.janela.arquivos import PainelDeArquivos, ParDeArquivos  # noqa: E402
from refino.desktop.janela.leitura import ROTAS_DA_LEITURA  # noqa: E402
from refino.desktop.janela.multiverso import CHAVE_DO_MULTIVERSO  # noqa: E402
from refino.desktop.janela.pipeline import (  # noqa: E402
    FIXOS_DECLARADOS,
    ROTA_DA_EXECUCAO,
    PainelDePipeline,
    rota_de_cancelamento,
    rota_do_progresso,
)
from refino.desktop.janela.principal import (  # noqa: E402
    ROTA_DA_CURVA_DO_LOESS,
    JanelaPrincipal,
)
from refino.desktop.janela.trabalhos import CHAVE_DOS_TRABALHOS  # noqa: E402
from refino.desktop.recorte import (  # noqa: E402
    PARA_QUEM_OPERA,
    PREFIXO_COBERTO,
    SECOES_DO_RECORTE,
)
from tests.contrato_de_rotas import contratos_das_rotas  # noqa: E402

if TYPE_CHECKING:  # pragma: no cover - só para o mypy
    from pytestqt.qtbot import QtBot

REPO: Final = Path(__file__).resolve().parents[1]
JANELA: Final = REPO / "src" / "refino" / "desktop" / "janela"


# ---------------------------------------------------------------------------
# AS CLASSES DA PARTIÇÃO — cada base de leitura cai em exactamente uma
# ---------------------------------------------------------------------------
#
# **`corpo` é o corpo de resposta de uma rota**, e as chaves lidas dele têm de ser declaradas por
# alguma rota que devolva JSON.
CORPO_DE_RESPOSTA: Final = frozenset({"corpo"})

# **Raízes ANINHADAS: a variável vem de um campo do corpo, e o esquema dela é o do
# campo.** O nome do esquema é conferido contra `components.schemas`, então nome
# inventado aqui reprova.
#
# `<BoolOp>` é a base sintética de `(recomendacao or {}).get(...)`: o `or` é a guarda de `None`
# que a recomendação exige, e o que se lê depois dele é o corpo do Palpite que a leitura guardou —
# o de `POST /matrizes/palpite`, reclamado em `GET /trabalhos/{id}`. Era o de
# `POST /matrizes/recomendar`, e as três chaves lidas existem nos dois; o esquema declarado é o da
# rota que de facto a produz.
RAIZES_ANINHADAS: Final[dict[str, str]] = {
    "comparacao": "RsdAntesEDepois",
    "p": "ProblemaResposta",
    "<BoolOp>": "PalpiteResposta",
    # A segunda opinião do Palpite. O painel da leitura desce um nível no corpo de
    # `POST /matrizes/recomendar` para tirar o rótulo e o corretor de dentro de `segunda_opiniao`.
    # O corpo passou a ser o de `POST /matrizes/palpite`, que traz o mesmo campo.
    # Bijeta contra `SegundaOpiniao`, que é modelo do projeto e está no OpenAPI — o caso forte.
    # Nasce COM o leitor, que é `leitura.texto_da_recomendacao`.
    "pela_verdade": "SegundaOpiniao",
    # `ConclusaoDoHistorico` traz a execução e a tabela em campos separados, e o painel lê a
    # primeira para escrever o cabeçalho que diz **de onde veio** a conclusão.
    "execucao": "MultiversoNoHistorico",
    # O relatório do corte da exportação. A base chama-se `corte` e **não**
    # `relatorio`, e o nome é escolha do contrato e não gosto: `relatorio` é `RAIZ_DO_RELATORIO`,
    # que é o `QualityReport` isento por o campo `qualidade` ser um `dict` livre. Reusar o nome
    # poria a leitura do corte a bijetar contra o esquema errado — e a primeira versão desta
    # unidade fez exactamente isso, com o portão a reprovar por três chaves em `artefatos.py`.
    #
    # Aqui **não** há isenção a declarar: `RelatorioDeExportacao` é modelo do projeto e o OpenAPI
    # o descreve, então a bijeção é contra o esquema, que é o caso forte.
    "corte": "RelatorioDeExportacao",
    # O corte do veredito. `ResultadoDoMultiverso` e `ConclusaoDoHistorico` trazem os
    # limiares num campo proprio, e o cabecalho do painel os le para dizer contra o que o veredito
    # foi decidido. Bijeta contra `LimiaresDoVeredito`, que e modelo do projeto e esta no OpenAPI —
    # entao e o caso forte, sem isencao a declarar.
    "limiares": "LimiaresDoVeredito",
    # A sugestão de lotes, lida para PRÉ-PREENCHER o controle de valor. A janela
    # desce um nível no corpo de `/matrizes/diagnosticar` para tirar `n_lotes` de dentro de
    # `lotes`, que é o mesmo objeto que `prosa_dos_lotes` formata para a justificativa. Bijeta
    # contra `SugestaoDeLotes`, que é modelo do projeto e está no OpenAPI — o caso forte, sem
    # isenção a declarar.
    "lotes": "SugestaoDeLotes",
    # A trajetória do QC, lida para DESENHAR a figura da deriva. O painel da qualidade
    # desce um nível no corpo de `/matrizes/diagnosticar` para tirar os quatro vetores de dentro de
    # `trajetoria_dos_qcs`, que só significam alguma coisa alinhados. Bijeta contra
    # `TrajetoriaDosQcs`, que é modelo do projeto e está no OpenAPI — o caso forte, sem isenção.
    #
    # A entrada só pôde nascer com o leitor, e isso foi MEDIDO (ver o bloco das distribuições mais
    # abaixo): uma base declarada que nenhum módulo de `desktop/janela/` lê reprova em DOIS
    # portões, porque classe sem uso é a lista de isenção que mata um portão. O leitor é
    # `PainelDeQualidade._constroi_figura`.
    "trajetoria": "TrajetoriaDosQcs",
    # O resumo das distribuições, lido para DESENHAR as três figuras de antes-e-depois. O
    # painel da qualidade desce um nível no corpo de `/execucoes` para tirar as
    # caixas, os vetores de médias e as duas pontas da legenda de dentro de
    # `distribuicoes`. Bijeta contra `ResumoDasDistribuicoes`, que é modelo do projeto e
    # está no OpenAPI — o caso forte, sem isenção a declarar.
    #
    # A entrada esperou o leitor, e o bloco das distribuições mais abaixo mediu por quê:
    # plantada, uma entrada que nenhum módulo de `desktop/janela/` lê
    # reprova em DOIS portões — `test_toda_base_de_leitura_esta_classificada` por classe
    # sem uso, e `test_as_chaves_aninhadas_...` por base declarada e não lida. Ela nasce
    # **com** o leitor, que é `PainelDeQualidade._caixas_da_distribuicao` e as duas irmãs.
    "distribuicoes": "ResumoDasDistribuicoes",
    # Os vetores por feature do volcano. O painel da qualidade desce um nível no corpo de
    # `/execucoes` para tirar os quatro vetores, o par de grupos, os dois `n` e a procedência de
    # cada eixo de dentro de `volcano`. Bijeta contra `ResumoDoVolcano`, que é modelo do projeto e
    # está no OpenAPI — o caso forte, sem isenção a declarar.
    #
    # **A entrada nasce COM o leitor**, como a de `distribuicoes` e pelo mesmo motivo medido: uma
    # base declarada que nenhum módulo de `desktop/janela/` lê reprova em DOIS portões. O leitor é
    # `PainelDeQualidade._volcano_do_par`, por `volcano_do_fio`.
    "volcano": "ResumoDoVolcano",
    # A escolha da política, lida para dizer QUAL corretor a execução corre.
    # O painel do pipeline desce um nível no aceite de `POST /execucoes` para tirar `correction`,
    # `politica`, `abstains` e `assessable` de dentro de `delegacao`. Bijeta contra
    # `DelegacaoResposta`, que é modelo do projeto e está no OpenAPI — o caso forte. Nasce COM o
    # leitor, que é `pipeline.texto_do_corretor`.
    "delegacao": "DelegacaoResposta",
    # Os vetores por feature da ANOVA de N grupos. O painel da
    # qualidade desce um nível no corpo de `/execucoes` para tirar os três vetores, os pares
    # `(grupo, n)`, o método, o post-hoc com a rota e a procedência de dentro de `anova`. Bijeta
    # contra `ResumoDaAnova`, que é modelo do projeto e está no OpenAPI — o caso forte, sem
    # isenção a declarar.
    #
    # **Raiz PRÓPRIA e não uma leitura a mais de `volcano`, e a razão é a população.** Aquela
    # descreve o par de grupos declarado; esta descreve TODOS os grupos de estudo da matriz,
    # derivados dela. Servir as duas da mesma raiz poria o `n` de dois grupos no rodapé de
    # uma figura sobre quatro.
    #
    # **A entrada nasce COM o leitor**, como as de `distribuicoes` e `volcano` e pelo mesmo motivo
    # medido: uma base declarada que nenhum módulo de `desktop/janela/` lê reprova em DOIS
    # portões. O leitor é `PainelDeQualidade._anova_dos_grupos`, por `contexto_do_fio`.
    "anova": "ResumoDaAnova",
    # A PCA que a janela desenha.
    #
    # As quatro figuras foram construídas antes de ligadas, e o motivo é este portão: uma base
    # declarada que nenhum módulo de `desktop/janela/` lê reprova em DOIS testes deste arquivo — por
    # classe sem uso e por base não lida. O par modelo/leitor nasce junto ou não nasce, e o leitor
    # é `PainelDeQualidade._scree` e as três irmãs.
    #
    # Bijeta contra `ResumoDaPca`, que é modelo do projeto e está no OpenAPI — o caso forte, sem
    # isenção a declarar.
    "pca": "ResumoDaPca",
    # Os vetores do PLS-DA, e é a raiz onde a validação do PLS-DA atravessa o fio.
    #
    # **A base chama-se `vetores_do_plsda` e NÃO `plsda`, e o nome é contrato.** `self.plsda` é o
    # dicionário de rótulos que põe o score ao lado da validação, e o portão de AST de
    # `test_desktop_qualidade.py` identifica o dono pelo NOME: uma local chamada `plsda` faria
    # `plsda.get("q2")` ser indistinguível do escape que ele existe para acusar. Ele reprovou com
    # dezoito leituras na primeira versão desta ligação. O painel desce
    # um nível no corpo de `/execucoes` para tirar os scores, as cargas, o VIP e — **no mesmo
    # objeto** — o `q2` e o `permutation_pvalue` que a regra exige ao lado deles. Bijeta contra
    # `ResumoDoPlsda`, que é modelo do projeto e está no OpenAPI.
    "vetores_do_plsda": "ResumoDoPlsda",
    # O inventário de grupos, lido para ESCREVER a linha do painel do pipeline.
    #
    # `pipeline.inventario_do_corpo` desce um nível no corpo de `/matrizes/diagnosticar` para tirar
    # o `nome` e o `n` de cada entrada de `grupos_disponiveis`, que só significam alguma coisa
    # **em par**. Bijeta contra `GrupoComContagem`, que é modelo do projeto e está no OpenAPI — o
    # caso forte, sem isenção a declarar.
    #
    # **Este portão ACUSOU a mudança e o desfecho é o certo:** a partição é fail-closed, e a
    # primeira execução depois de o tipo do campo mudar reprovou nomeando `grupo`. Classificar é a
    # resposta, e não afrouxar.
    "grupo": "GrupoComContagem",
    # O PAR de PCAs do painel da correção. O painel
    # dos gráficos desce um nível no corpo de `/execucoes` para tirar os dois conjuntos de scores,
    # os dois vetores de tipo, as duas variâncias, os dois `qc_dispersion_ratio` e os dois pares
    # contagem/digest de dentro de `pcas_da_correcao`. Bijeta contra `ParDePcas`, que é modelo do
    # projeto e está no OpenAPI — o caso forte, sem isenção a declarar.
    #
    # A entrada não podia nascer antes do leitor: uma base declarada que nenhum módulo de
    # `desktop/janela/` lê reprova em DOIS portões — por classe sem uso e por base declarada e não
    # lida. Guardar o par não é desenhá-lo, e o leitor aninhado é
    # `PainelDeGraficos._painel_da_correcao`.
    #
    # **A armadilha simétrica é apontar para `ResumoDaPca`**, e `test_u2_o_par_de_pcas_esta_no_
    # OPENAPI_...` guarda essa metade: aquele esquema descreve UMA matriz e este descreve duas.
    "par": "ParDePcas",
    # Os dois vetores de RSD por feature. O painel dos
    # gráficos desce um nível no corpo de `/execucoes` para tirar as ONZE chaves de dentro de
    # `rsd_por_feature` — os dois vetores, as duas contagens de finitas, as duas de injeções, os
    # quatro rótulos e o denominador `n_features`, que é o «de quantas» do rodapé. Bijeta
    # contra `RsdPorFeature`, que é modelo do projeto e está no OpenAPI — o caso forte, sem isenção.
    #
    # A entrada não podia nascer antes do leitor, pelo mesmo motivo de `par`: uma base declarada
    # que nenhum módulo de `desktop/janela/` lê reprova em DOIS portões — por classe sem uso e por
    # base declarada e não lida. O leitor aninhado é `PainelDeGraficos._violino_do_rsd`.
    #
    # **A armadilha simétrica é apontar para `RsdAntesEDepois`**, e `test_u6_o_rsd_por_feature_esta_
    # no_OPENAPI_...` guarda essa metade: aquele esquema traz duas MEDIANAS sobre antes e depois, e
    # este traz dois VETORES por feature sobre a matriz corrigida.
    "rsd_por_feature": "RsdPorFeature",
    # A composição por classe. O painel dos gráficos desce um nível no corpo de `/execucoes` para
    # tirar as SEIS chaves de dentro de `composicao_por_classe` — a lista de classes, a
    # procedência, a população, o denominador de features, a contagem de injeções fora da população
    # e os dois tokens de definição. Bijeta contra `ComposicaoPorClasse`, que é modelo do projeto e
    # está no OpenAPI.
    #
    # A base nasce com o leitor, que é `PainelDeGraficos._composicao_por_classe`: uma base
    # declarada que nenhum módulo de `desktop/janela/` lê reprova em DOIS portões.
    #
    # **As leituras POR CLASSE ficam em `plots/`, e isso é desenho e não omissão.** `grupo`,
    # `n_celulas` e `n_detectadas` são lidas em `correcao.classes_do_fio`, como as das duas famílias
    # de caixas em `distribuicao.caixas_do_fio`: este portão parte da **base**, e o elemento de uma
    # lista não é base. O que o guarda do outro lado é `ComposicaoPorClasse.classes`, que o pydantic
    # valida item a item na fronteira.
    #
    # **A armadilha simétrica é apontar para `ClasseDaComposicao`**, e
    # `test_u8_a_composicao_esta_no_OPENAPI_...` guarda essa metade: aquele esquema descreve UMA
    # classe e este descreve o conjunto mais os rótulos que o tornam interpretável.
    "composicao": "ComposicaoPorClasse",
    # O corpo da MEDIÇÃO, e a entrada é uma emenda ao bloco da medição mais abaixo. Aquele bloco
    # diz que a medição «não entra em `RAIZES_ANINHADAS`, e a ausência é estrutural e não dívida»,
    # com o argumento de que *"as chaves saem de `corpo` direto"*. Isso era verdade enquanto o
    # único leitor era `recebe_medicao(corpo)`, e deixou de ser quando o montador do painel da
    # correção passou a ler o corpo **guardado**: ali a base é uma variável local, e uma base nova
    # sem classe reprova por partição fail-closed.
    #
    # **Classificá-la como raiz aninhada é o portão MAIS forte dos dois disponíveis.** Chamar a
    # local `corpo` poria as seis chaves a bijetar contra a UNIÃO das respostas de todas as rotas;
    # aqui elas bijetam contra `MedicaoDoCorretorResposta`, que é exactamente a rota de onde vêm.
    # O argumento daquele bloco — *"declarar uma raiz aqui reprovaria por classe sem uso"* —
    # caducou pelo motivo que ele próprio nomeia: agora há uso.
    "medicao": "MedicaoDoCorretorResposta",
    # A curva do LOESS de uma feature. O painel dos gráficos lê o corpo de
    # `POST /matrizes/curva-do-loess` pela base `curva_do_loess`, e ele bijeta contra o modelo da
    # rota que o produz. Nasce COM o leitor, `graficos._curva_do_loess_da_feature`.
    "curva_do_loess": "CurvaDoLoessResposta",
    # O intervalo de cada barra do painel da correção. `graficos._intervalos_da_medicao`
    # desce um nível em `medicao["intervalos"]` e lê, de cada corretor, os campos de
    # `IntervaloDaBarra` — modelo do projeto, no OpenAPI pelo `dict[str, IntervaloDaBarra]` de
    # `MedicaoDoCorretorResposta`. O caso forte, sem isenção; nasce COM o leitor.
    "intervalo": "IntervaloDaBarra",
    # As duas da correlação. O painel dos gráficos desce um nível no
    # corpo de `/execucoes` para tirar a matriz quadrada, os nomes, o método, os dois denominadores
    # e a contagem de indefinidas de dentro de `correlacao`; e a lista ordenada, os coeficientes, a
    # referência, o perfil e a ordem dos grupos de dentro de `busca_por_padrao`. As duas bijetam
    # contra modelos do projeto que o OpenAPI descreve — o caso forte, sem isenção a declarar.
    #
    # As bases nascem com os leitores, que são `PainelDeGraficos._mapa_de_correlacao` e
    # `._busca_de_padrao`, pelo mesmo motivo medido: uma base declarada que nenhum módulo de
    # `desktop/janela/` lê reprova em DOIS portões.
    #
    # **DUAS entradas e não uma, e a separação é a mesma do lado do esquema:** `pattern_reference` a
    # `None` salta a busca **sem** desligar o mapa, então as duas ausentam-se por motivos que não se
    # sobrepõem. Uma raiz só faria a ausência de uma parecer implicar a da outra — é o argumento com
    # que `anova` não é uma leitura a mais de `volcano`.
    #
    # **A armadilha simétrica é apontar as duas para o mesmo esquema**, e
    # `test_u12_as_duas_raizes_da_correlacao_...` guarda essa metade: `ResumoDaCorrelacao` descreve
    # uma matriz `k x k` e `ResumoDaBuscaPorPadrao` descreve um vetor por feature com a ordem que a
    # busca produziu.
    "correlacao": "ResumoDaCorrelacao",
    "busca_por_padrao": "ResumoDaBuscaPorPadrao",
    # As duas do agrupamento. O painel dos gráficos desce um nível no
    # corpo de `/execucoes` para tirar os rótulos por amostra, os grupos, o `k`, a inércia e a
    # semente **derivada** de dentro de `kmeans`; e as unidades, a forma da grade, as épocas e o
    # erro de quantização de dentro de `som`. As duas bijetam contra modelos do projeto que o
    # OpenAPI descreve — o caso forte, sem isenção a declarar.
    #
    # **As bases nascem COM os leitores**, como as da correlação e pelo mesmo motivo medido: uma
    # base declarada que nenhum módulo de `desktop/janela/` lê reprova em DOIS portões. Os leitores
    # são `PainelDeGraficos._kmeans_das_amostras` e `._som_das_amostras`.
    #
    # **DUAS entradas e não uma, e a separação é a mesma do lado do esquema:** as duas bandeiras são
    # independentes em `AnalysisSpec` e as duas vêm desligadas por padrão, então as duas ausentam-se
    # por motivos que não se sobrepõem — pedir o K-means sem o SOM é o caso comum. Uma raiz só faria
    # a ausência de uma parecer implicar a da outra.
    "kmeans": "ResumoDoKmeans",
    "som": "ResumoDoSom",
    # As duas da classificação. O painel dos gráficos desce um nível no
    # corpo de `/execucoes` para tirar o nome, as classes, as contagens, o vetor de recall, as três
    # medidas, o `p` de permutação, o nível do acaso, o regime de validação e a semente **derivada**
    # de dentro de `random_forest` e de dentro de `svm`. As duas bijetam contra um modelo do projeto
    # que o OpenAPI descreve — o caso forte, sem isenção a declarar.
    #
    # **As bases nascem COM o leitor**, como as do agrupamento e pelo mesmo motivo medido: uma base
    # declarada que nenhum módulo de `desktop/janela/` lê reprova em DOIS portões. O leitor é
    # `PainelDeGraficos._classificacao_de`, chamado por `._floresta_das_amostras` e
    # `._svm_das_amostras`.
    #
    # **DUAS entradas e não uma, e a separação é a mesma do lado do esquema:**
    # `AnalysisSpec.classifiers` é uma **tupla**, e `("random_forest",)` é o caso comum porque cada
    # um custa minutos — as duas ausentam-se por motivos que não se sobrepõem.
    #
    # **E as duas apontam para o MESMO esquema, o que aqui é correcto e não a armadilha simétrica.**
    # Nos pares `correlacao`/`busca_por_padrao` e `kmeans`/`som` os dois esquemas descrevem coisas
    # diferentes — uma matriz `k x k` contra um vetor por feature; uma partição contra uma grade —,
    # e apontá-los ao mesmo seria perder a distinção. Aqui a quantidade é **a mesma** sobre dois
    # modelos, e dois esquemas idênticos seriam a cópia que diverge no dia em que um ganhar um
    # campo.
    # **A base chama-se `classificador` e NÃO `random_forest`/`svm`, e a diferença é medida.**
    # Este portão classifica pelo **nome da variável local** que recebe a raiz, e os dois
    # classificadores são lidos por **um** leitor partilhado — `PainelDeGraficos._classificacao_de`,
    # chamado por `._floresta_das_amostras` e `._svm_das_amostras`. Duas cópias do leitor, uma com
    # o local chamado `random_forest` e outra `svm`, dariam ao portão os dois nomes de raiz e
    # custariam **dezasseis `.get` duplicados** — a irmã que diverge no dia em que um lado ganhar
    # uma chave, com o sintoma a ser uma aba a imprimir «não declarado» onde a outra imprime o
    # número.
    #
    # **O que se perde com o nome partilhado é NADA, e a razão é que a outra metade já está
    # coberta:** `recebe_classificadores` faz `corpo.get("random_forest")` e `corpo.get("svm")`, e
    # `corpo` é corpo de resposta — essas duas chaves bijetam contra a UNIÃO das chaves de resposta,
    # pela direção de entrada. Então os **nomes das raízes** são conferidos lá, e as **chaves de
    # dentro delas** são conferidas aqui, contra o esquema que as duas partilham.
    "classificador": "ResumoDoClassificador",
    # A da rede. O painel dos gráficos desce um nível no corpo de
    # `/execucoes` para tirar as features, a lista de arestas, as duas contagens, o alfa, a
    # penalidade, o `n` de amostras, o método por extenso, a amostragem e o denominador de features
    # de dentro de `rede`. Bijeta contra `ResumoDaRede`, que é modelo do projeto e está no OpenAPI —
    # o caso forte, sem isenção a declarar.
    #
    # **A base nasce COM o leitor**, como as da classificação e pelo mesmo motivo medido: uma base
    # declarada que nenhum módulo de `desktop/janela/` lê reprova em DOIS portões. O leitor é
    # `PainelDeGraficos._rede_das_features`.
    #
    # **Raiz PRÓPRIA e não uma leitura a mais de `correlacao`, e a razão é a bandeira.**
    # `AnalysisSpec.dspc` e `AnalysisSpec.correlation` são separadas, e as duas famílias medem
    # coisas diferentes — a rede estima a **precisão** e o mapa estima a covariância. Uma raiz só
    # faria a ausência de uma parecer implicar a da outra, que é o argumento com que a ANOVA não é
    # uma leitura a mais de `volcano`.
    #
    # **E o campo `matriz` NÃO é lido daqui, porque `ResumoDaRede` não o declara** — é a única das
    # raízes desta aba sem ele. O montador escreve `NAO_DECLARADA` no rodapé em vez de inventar a
    # procedência, e o conserto é acrescentar o campo ao esquema; enquanto ele não existir, ler a
    # chave reprovaria aqui, que é o desfecho certo.
    "rede": "ResumoDaRede",
    # A da normalidade, e ela nasce COM o leitor, pelo motivo medido das outras: uma
    # base declarada que nenhum módulo de `desktop/janela/` lê reprova em DOIS portões. O leitor é
    # `PainelDeQualidade.mostra_normalidade`, que tira a lista de grupos, os três níveis e os nomes
    # das variáveis de dentro de `normalidade`; bijeta contra `ResumoDaNormalidade`, modelo do
    # projeto no OpenAPI — o caso forte, sem isenção.
    #
    # **As leituras POR GRUPO ficam em `interface/fluxo.py`**, como as por classe da composição
    # ficam em `plots/`: este portão parte da base, e o elemento de uma lista não é base. Quem as
    # confere é `tests/unit/test_normalidade.py`, que passa um `ResumoDaNormalidade` real pelo
    # texto e pelo CSV e afirma que os números chegam.
    "normalidade": "ResumoDaNormalidade",
}

# **`relatorio` é o campo `qualidade`, e ele é um `dict` LIVRE no esquema** —
# `{"additionalProperties": true, "type": "object"}`, medido em 23/08/2026. O OpenAPI
# é **opaco** ali, então o oráculo daquela subárvore não pode ser o esquema: é
# `QualityReport`, que é quem produz o
# dicionário. A isenção vai declarada com o motivo.
RAIZ_DO_RELATORIO: Final = "relatorio"

# **`item` percorre o `detail` de um 422, e o `detail` é gerado pelo FastAPI** — não é modelo nosso,
# e `msg` é chave da forma de erro do pydantic. Não há esquema do projeto contra o que bijetar.
GERADAS_PELO_FRAMEWORK: Final[dict[str, frozenset[str]]] = {"item": frozenset({"msg"})}

# **E as CHAVES que o framework gera dentro do próprio `corpo`, com o motivo de cada uma.** Esta
# lista é separada da de bases porque `detail` é lido de `corpo` — que é corpo de
# resposta de facto —, e não de uma raiz própria. A primeira versão deste portão
# reprovou por isto, em `leitura.py:136` e
# `pipeline.py:186`: `corpo.get("detail")` é o caminho de erro das duas, e nenhum modelo nosso o
# declara. **A reprovação foi o desfecho certo** — ela nomeou os dois sítios exactos, e classificar
# é a resposta, não afrouxar.
CHAVES_DO_FRAMEWORK: Final[dict[str, str]] = {
    "detail": (
        "o corpo de erro do FastAPI — 404, 409, 422 e 500 —, gerado pelo framework a partir de "
        "`HTTPException` e da validação do pydantic. Não é modelo do projeto"
    ),
    "msg": "cada item da lista `detail` de um 422 do pydantic",
}

# **Bases que NÃO são resposta**, cada uma com o motivo. `campos` é o corpo do
# PEDIDO, conferido pela direção de SAÍDA e não por esta; `visivel` é o dicionário
# de apresentação do painel de qualidade, e as duas chaves dele são marcadores
# internos com `__` nas pontas.
NAO_SAO_RESPOSTA: Final[dict[str, str]] = {
    # **AS DUAS SEM CAMINHO DE FIO — merge do bloco das análises, 17/09/2026.**
    #
    # `PainelDeGraficos` tem montador para os cinco construtores de `plsda_multiclasse` e
    # `agrupamento` porque o portão de bijeção do painel o exige, e porque sem montador a entrada
    # aparece no combo e levanta `FonteAusenteError` — que o painel descreve como defeito de ligação
    # do programa, e não é isso que se passa. Os montadores leem estas duas bases, que hoje são
    # sempre `None`: **nenhuma rota devolve o que elas consomem**, e `plsda_multiclasse` não tem
    # sequer função `*_do_fio`.
    #
    # **Por que aqui e não em `RAIZES_ANINHADAS`:** aquela tabela bijeta contra um esquema do
    # OpenAPI, e exigir a entrada obrigaria a declarar um modelo que rota nenhuma serve — o
    # instrumento a certificar-se. A pré-condição está escrita em `test_u11_...`: modelo que rota
    # nenhuma alcança não entra no OpenAPI. **Quando a rota existir, estas duas MUDAM de tabela**,
    # e é essa migração que o portão torna obrigatória em vez de opcional.
    "plsda_multiclasse": (
        "raiz por construir: os três montadores do PLS-DA multiclasse leem-na para recusar com "
        "erro TIPADO em vez de quebrar. Rota nenhuma a devolve — migra para `RAIZES_ANINHADAS` "
        "no commit que a ligar"
    ),
    "agrupamento": (
        "raiz por construir: os dois montadores do dendrograma e do mapa de calor leem-na pela "
        "mesma razão da irmã acima. Rota nenhuma a devolve"
    ),
    "campos": "corpo do pedido; conferido pela direção de SAÍDA, abaixo",
    # As duas leituras do esquema claro, que o portão viu ao entrar o tema.
    "_COR_DO_ICONE": "constante de `avisos.py`: o token de cor do ícone de cada tipo de aviso",
    "receita": (
        "dicionário que `poe_icone` grava no widget para a troca de esquema redesenhar o ícone"
    ),
    # O índice da aba para onde «Exportar esta figura em Gráficos» leva sai da tupla
    # `ABAS` — até 09/11/2026 também os atalhos «Ver gráfico», que saíram com a aba deles.
    "paineis": "atributo de `Aba`, da tupla `ABAS` da própria janela — não vem de resposta",
    # A do lote, e ela é do lado do PEDIDO como `campos`. O lote dos três corretores lê o
    # `mode` do pedido para compor o nome dos arquivos que escreve na pasta. É a MESMA tupla que
    # `campos` acima, com nome próprio porque `_inicia_lote` a desempacota sem a enviar — e o
    # portão classifica por NOME da base, não por origem.
    "campos_do_lote": (
        "corpo do pedido, desempacotado em `_inicia_lote` só para ler o `mode` que nomeia os três "
        "arquivos do lote. Quem envia o pedido é `_gera_artefato`, por `campos`"
    ),
    "visivel": "dicionário de apresentação do painel de qualidade, com marcadores internos",
    # `campos_do_artefato` saiu, e quem a acusou foi este portão:
    # `_gera_artefato` lia nela se o artefato delegava, e o único que delegava era o `corrigir` da
    # aba «Executar e exportar», que saiu. Nenhum download da janela espera o Palpite.
    # A da remontagem da aba de preparo. `PainelDeArquivos` guarda o campo de
    # cada arquivo por chave (`medidas`, `lista`, `fatores`), que são nomes de LAYOUT e não campos
    # de resposta: o portão a viu pela forma `x["literal"]`, a mesma de uma leitura de corpo.
    # `fichas` saiu com os cartões que ela nomeava — o portão acusou a classificação
    # caduca.
    "estados": "campos de arquivo do painel de arquivos, por chave — layout, não resposta",
    # A dos nomes das opções. O portão viu-a pela forma `x["literal"]` quando os textos
    # dos fatores e a recusa local passaram a nomear a opção como o seletor a mostra.
    "NOMES_DAS_OPCOES": (
        "constante de `nomes.py`: o nome que a tela mostra para cada valor de eixo, lido "
        "pelos textos que citam a opção — não vem de resposta"
    ),
    "botoes_de_remocao": "controles de desanexar arquivos por chave, sem conteúdo de resposta",
    # As duas da faixa legível. O portão viu-as pela forma `x["literal"]`: a faixa lê
    # o rótulo e o tom de um estado TERMINAL pelo nome dele (`ROTULOS_DOS_ESTADOS["concluido"]`),
    # porque o desfecho chega pelo código HTTP e não por um `estado` no corpo. As chaves são as de
    # `api.progresso.ESTADOS`, e a bijeção com elas tem teste próprio, em
    # `test_todo_estado_que_o_registro_produz_tem_rotulo_legivel`.
    "ROTULOS_DOS_ESTADOS": (
        "o rótulo legível de cada estado do registro, em `janela/trabalhos.py` — vocabulário da "
        "tela, em bijeção com `ESTADOS`, e não corpo de resposta"
    ),
    "TONS_DOS_ESTADOS": (
        "a cor do selo de cada estado do registro, em `janela/trabalhos.py` — aparência, em "
        "bijeção com `ESTADOS`, e não corpo de resposta"
    ),
    # As duas da explicação do Multiverso. `EXPLICACAO_DOS_VEREDITOS` lê, pela chave,
    # o rótulo traduzido de cada veredito e os três limiares padrão, para a frase dizer o corte que
    # o núcleo aplica. O portão as viu pela forma `x["literal"]`, a de uma leitura de corpo.
    "VEREDITOS": "a tradução dos três rótulos do veredito, constante do painel do multiverso",
    "PADRAO_DO_LIMIAR": (
        "os limiares padrão do veredito, que o painel do multiverso lê de `StabilityThresholds` — "
        "constante do domínio, e não resposta"
    ),
    # O estado vazio da figura desenha o ícone da aba «Gráficos», lido do tema.
    "ICONES": "os desenhos SVG das abas, por título, em `tema.py` — aparência, não resposta",
    # A do terceiro arquivo, e ela é do lado do PEDIDO como `campos`.
    # `arquivos_da_execucao` parte do dicionario que `arquivos_para_envio` devolve e acrescenta
    # o CSV de fatores; `_dispara_etapa` faz o mesmo do outro lado. E `files` do multipart, e
    # quem o confere e a direcao de SAIDA — nao ha resposta nenhuma a bijetar contra esquema.
    "arquivos": (
        "o `files` do multipart do PEDIDO; conferido pela direcao de SAIDA, que compara o que a "
        "janela envia com o que a rota declara"
    ),
    # A dos artefatos, e ela é do lado do PEDIDO como `escolhas`. Apareceu quando
    # `campos_do_artefato` passou a compor o payload dos três artefatos, e o portão reprovou
    # nomeando-a — que é o desfecho certo de uma partição fail-closed.
    "disponiveis": (
        "o dicionário LOCAL que `PainelDePipeline.campos_do_artefato` monta com tudo o que este "
        "painel sabe, antes de filtrar pelo que a rota daquele artefato aceita. É pedido e não "
        "resposta, e quem confere o filtro é `test_u23_cada_artefato_leva_so_o_que_a_rota_dele_"
        "aceita`, por bijeção contra a OpenAPI de cada uma das três rotas"
    ),
    # As duas dos slots, e as duas são do lado do PEDIDO. Elas apareceram quando os quatro slots
    # de `principal.py` passaram a montar payload, e o portão reprovou nomeando-as — que é o
    # desfecho certo de uma partição fail-closed: base nova reprova até ser classificada.
    # **`campos_dos_arquivos` saiu em 09/11/2026**, com `_dispara_execucao`, o único que a lia: o
    # botão «Executar pipeline» saiu com a aba «Executar e exportar», e o portão acusou a
    # classificação caduca.
    "escolhas": (
        "o que `PainelDePipeline.escolhas_do_disparo` devolve — `correcao`, `n_lotes` e "
        "`rsd_cutoff` —, lido por `_campos_atuais` para a tela de que as três etapas partem. "
        "Pedido, e os três estão na bijeção de saída"
    ),
    # A dos eixos, e ela entrou na árvore com este portão VERMELHO. Medido:
    # `test_toda_base_de_leitura_esta_classificada` reprovava sobre `main` limpo em `11b38e1`,
    # nomeando `eixos`, e o commit que a introduziu não a classificou. É a partição fail-closed a
    # funcionar — ela acusou — e ninguém leu; a marca do commit declara `make check` verde, e este
    # teste corre nele.
    # A dos títulos, e ela é APRESENTAÇÃO. O agrupamento das abas fez os composers
    # desenharem uma moldura por painel, e o rótulo de cada moldura sai de `PAINEIS` em vez de ser
    # escrito no composer — duas frases para o mesmo painel divergiriam, e a que o portão do
    # LEIA-ME confere é a de `PAINEIS`, então a que caducaria em silêncio seria a da tela.
    "titulos": (
        "`{chave de painel: título}` derivado de `PAINEIS` e entregue aos composers de aba, para "
        "a moldura de cada painel mostrar o nome que a estrutura declara. É texto de "
        "apresentação e nunca sai da janela: não é corpo de pedido nem de resposta"
    ),
    # A do lugar do CSV escrito, e ela é ESTADO INTERNO — nem pedido nem resposta. Nasceu quando
    # o botão «Baixar X» saiu e o registro do corte passou a ser escrito pela janela: os bytes do
    # CSV chegam numa resposta e o relatório do corte noutra, DEPOIS, então é preciso lembrar ONDE
    # o CSV foi escrito para pôr o arquivo irmão ao lado dele.
    #
    # **O portão reprovou nomeando-a, e o desfecho é o certo:** a partição é fail-closed, e base
    # nova reprova até ser classificada. Foi assim que ela apareceu, e não por alguém se lembrar.
    # A identidade dos arquivos do passo a passo. O portão reprovou nomeando-a, como
    # com `_escritos`: é a janela a escrever a chave do CSV de fatores no dicionário que ela própria
    # monta para a fotografia das etapas.
    "identidade": (
        "`JanelaPrincipal._identidade_dos_arquivos`, o conteúdo dos arquivos do passo a passo — a "
        "matriz e o CSV de fatores, por `sha256` — que entra na fotografia de cada etapa. É "
        "estado da janela e nunca sai dela: não é corpo de pedido nem de resposta"
    ),
    # **`_escritos` SAIU em 08/11/2026, e quem o tirou foi a direção das órfãs.** O ramo do
    # relatório do corte lia `self._escritos.get("exportar")`; desde que a Correção e a
    # Normalização baixam cada uma o seu arquivo MetaboAnalyst, ele lê pela chave do artefato que
    # pediu o relatório, que não é literal — e uma leitura sem chave literal não é forma que este
    # portão extraia. A classe ficou sem leitura, e classe sem leitura é isenção que sobrevive ao
    # código que a justificava.
    "eixos": (
        "`PainelDePipeline.ressalvas_dos_eixos`, o dicionário LOCAL que liga cada eixo da grade à "
        "ressalva medida que o painel exibe antes da escolha. Nasce de `RESSALVAS_POR_EIXO` "
        "em `janela/pipeline.py` e nunca sai da janela: não é corpo de resposta nem de pedido, é "
        "texto de apresentação. Quem o confere é `test_u20_todo_eixo_com_ressalva_a_exibe`"
    ),
    # A tabela das rotas que propõem eixo, lida por chave em
    # `janela/leitura.py`.
    "PROPOSTAS_POR_ROTA": (
        "tabela do código, e não corpo de resposta: `_onde_o_valor_proposto_entra` "
        "consulta por ela o eixo que o «Palpite» propõe e o que o «Diagnóstico» propõe, para dizer "
        "onde cada um entra na aba «Correção». Nasce em `ui/fluxo.py` e nunca vem do fio"
    ),
}

# **Os campos que a rota aceita e a janela NUNCA envia**, com o motivo de cada um. A lista existe
# para o complemento ser **declarado** em vez de residual: o que sobra de uma partição vira
# residual, e residual lê-se como pronto.
ACEITOS_E_NUNCA_ENVIADOS: Final[dict[str, str]] = {
    # `missing_threshold` saiu desta lista. A isenção dizia *"a janela não expõe o limiar de
    # ausentes; a rota usa o padrão dela. É um dos sete parâmetros fixos que o painel do recorte
    # declara"*, e era falsa desde que o widget entrou: o widget existe em `pipeline.py`, o valor é
    # lido em `escolhas_do_disparo`, e `recorte.py` diz ao usuário que *"dos sete eixos, os SETE
    # são do usuário nesta janela"*. A isenção era o retrato do mundo anterior ao widget, e enquanto
    # ela viveu o portão certificou como deliberada uma perda silenciosa.
    #
    # > **Isenção carrega afirmação sobre o produto, e afirmação caduca.** Na própria tabela de
    # > exceções, o mecanismo estava exercitado nas duas direções e verde, e o que estava errado era
    # > a premissa que ele dispensava de conferir.
    "orientation": (
        "a janela não expõe a orientação, e o padrão da rota é `features-in-rows` — o dos arquivos "
        "do laboratório"
    ),
    # **`auto` ENTROU nesta lista em 09/11/2026**, com a saída do botão «Executar pipeline», que o
    # mandava ao `/execucoes` com a caixa «Automático» ticada. Quem manda o `/execucoes` agora é a
    # Análise, e ela nunca delega — a isenção tem gatilho: um caminho que voltasse a mandar `auto`
    # ao `/execucoes` faria a igualdade abaixo reprovar, e a linha sairia. E a prévia da
    # Correção, que delegava com a caixa, também deixou de o mandar: a caixa saiu.
    "auto": (
        "a Análise leva o corretor da entrada que a pessoa escolheu — o efetivo da Correção "
        "feita, ou nenhum — e nunca a delegação: mandar `auto` de novo mediria o Palpite "
        "outra vez, e a execução poderia correr outro corretor sem a tela saber. A "
        "janela não delega em pedido nenhum; quem delega é a linha de comando, com `--auto`"
    ),
    # `analise_json` saiu desta lista, e o gatilho FUNCIONOU.
    #
    # A isenção nasceu com a data de morte escrita: *"no commit em que eles
    # entrarem esta linha SAI: o portão reprova sozinho, porque a janela passará a enviar o
    # campo"*. Foi exactamente o que aconteceu — na primeira execução depois de o painel ganhar as
    # caixas por família, este teste reprovou com *"a rota aceita ['orientation'] e a janela não os
    # envia; a lista declarada é ['analise_json', 'orientation']"*.
    #
    # > **A diferença entre esta isenção e a de `missing_threshold` não foi o cuidado de quem as
    # > escreveu — foi ter GATILHO.** Aquela sobreviveu onze dias a ter deixado de ser verdade,
    # > porque nada ligava a afirmação ao estado; esta durou um dia, porque a direção da igualdade
    # > exata a mata sozinha.
}


@pytest.fixture(scope="module")
def contratos() -> dict[tuple[str, str], Any]:
    """Os contratos das rotas, lidos do OpenAPI por `contratos_das_rotas`."""
    return contratos_das_rotas(dict(build_app().openapi()))


def _paineis_mais_ricos(qtbot: QtBot) -> tuple[PainelDePipeline, PainelDeArquivos]:
    """Os dois painéis na configuração MAIS RICA que a produção admite — ver `campos_enviados`.

    Extraído para o portão das prévias `/etapas/*` ler a MESMA tela que o
    portão do `/execucoes`: duas cópias desta montagem divergiriam no primeiro campo novo.
    """
    arquivos = PainelDeArquivos(lambda: None)
    qtbot.addWidget(arquivos)
    arquivos._par = ParDeArquivos("m.csv", b"feature,QC1\n1,2\n", "s.csv", b"label,time\nQC1,1\n")
    # O par entra direto, sem a escolha que deduz o modo: o modo vai à mão, senão a chamada é
    # recusada por falta dele.
    arquivos.modo.setEditText("pos_hilic")

    pipeline = PainelDePipeline()
    qtbot.addWidget(pipeline)
    # O par de grupos do PLS-DA entra AQUI, e não em `ACEITOS_E_NUNCA_ENVIADOS`.
    #
    # `campos_base` só o inclui quando os dois campos estão preenchidos, e este fixture é
    # *"a chamada MAIS RICA que a produção admite"*: deixá-los vazios poria `group_a`/`group_b` no
    # lado dos «aceitos e nunca enviados», que é uma afirmação **falsa** — a janela envia-os.
    pipeline.grupo_a.setText("ATLL")
    pipeline.grupo_b.setText("NoHTLV")
    # A DELEGAÇÃO entrava aqui pelo mesmo argumento do par de grupos: a caixa
    # «Automático» era ticada para a chamada ser a mais rica que a produção admitia. A caixa saiu, e
    # a produção já não admite delegar — `auto` está entre os nunca enviados, com o motivo.
    # **As duas frações de posto entram AQUI pelo mesmo argumento do par de grupos — 10/09/2026.**
    #
    # `campos_base` só as inclui quando não estão no mínimo, e este fixture é *"a chamada
    # MAIS RICA que a produção admite"*: deixá-las vazias poria as duas do lado dos «aceitos e
    # nunca enviados», que é uma afirmação **falsa** — a janela envia-as, e a produção admite
    # escolhê-las.
    pipeline.fracao_posto.setValue(0.60)
    pipeline.fracao_posto_2.setValue(0.90)
    # O CORTE DE AUSENTES entra AQUI, e a ausência dele era o que cegava o portão.
    #
    # Mesmo argumento das duas frações acima, e foi a **única** das seis condicionais que ninguém
    # veio pôr: `ACEITOS_E_NUNCA_ENVIADOS` declarava-o como nunca-enviado, então parecia não
    # haver nada a exercitar. A isenção era falsa desde que o widget entrou, e enquanto ela viveu o
    # portão correu com o widget no mínimo — o valor que **não** exercita o ramo — e certificou a
    # omissão como deliberada.
    pipeline.ausentes.setValue(0.05)
    # A referência entra AQUI pelo mesmo argumento. A janela envia-a quando o combo
    # tem texto, e este fixture é *"a chamada MAIS RICA que a produção admite"*: deixá-lo vazio
    # poria a referência do lado dos «aceitos e nunca enviados», que seria afirmação falsa.
    pipeline.referencia.setCurrentText("QC1")
    # O terceiro arquivo entra AQUI pelo mesmo argumento. A janela envia-o quando
    # escolhido, e este fixture e *"a chamada MAIS RICA que a producao admite"*.
    arquivos._fatores = ("fatores.csv", b"QC1,2.0\n")
    # Os cortes de lote entram AQUI pelo mesmo argumento do par de grupos. A janela
    # envia `batch_cuts` quando o número de lotes é o que o diagnóstico propôs, e este fixture é
    # *"a chamada MAIS RICA que a produção admite"*: sem a proposta, o campo ficaria do lado dos
    # «aceitos e nunca enviados», que seria afirmação falsa. Por isso os dois construtores abaixo
    # pedem dois lotes.
    pipeline.pre_preenche("n_batches", 2)
    pipeline.define_cortes_propostos([35])
    return pipeline, arquivos


@pytest.fixture
def campos_enviados(qtbot: QtBot) -> frozenset[str]:
    """Todo campo que a janela põe num pedido, CHAMANDO os construtores de produção.

    Chamados e não extraídos: um dicionário construído à mão pelo teste pode conter o que a produção
    nunca produz, e foi exactamente isso que já deixou um caminho de congelamento inalcançável com
    a suíte verde.

    **A chamada é a MAIS RICA que a produção admite** — com eixo sobrescrito, recomendação e corte
    de RSD —, porque `proposta_json` e `rsd_cutoff` só entram nesses ramos. Uma chamada mínima
    deixaria dois campos fora da bijeção sem que nada acusasse.

    > **O `PainelDeLeitura` e `leitura.campos_de_sobrescrita()` SAÍRAM desta união, e não por
    > conveniência: o método foi REMOVIDO.** Ele devolvia sempre `{}` aqui — nenhum checkbox está
    > marcado neste fixture —, então era um termo que **não podia** contribuir para a união. Os
    > quatro eixos da grade entram agora pelo lado que os produz: `campos_da_execucao` lê os widgets
    > do painel do pipeline, e `set(execucao)` já os traz.

    ## Quem manda o `/execucoes` mudou em 09/11/2026, e o pedido é o da ANÁLISE

    Até ali o pedido era o do botão «Executar pipeline» (`campos_da_execucao`), com os controles
    das três abas. O botão saiu com a aba «Executar e exportar», e quem manda o
    `/execucoes` é a Análise, a partir da entrada escolhida (`JanelaPrincipal._dispara_etapa`): os
    campos dela vêm da tela, e os das etapas anteriores vêm da etapa FEITA que a entrada é. **A
    entrada mais rica é a normalizada, feita sobre a corrigida** — é a única em que os eixos do
    pré-processamento e a linha da máquina chegam juntos —, e a cadeia é montada aqui pelo mesmo
    `EstadoDasEtapas` que a janela usa.

    ## E a Análise sobre a ORIGINAL, porque um campo não cabe na mais rica

    `sem_proposta` e `proposta_json` se excluem por construção — a rota recusa os dois juntos —,
    então nenhuma chamada única leva os dois. A união soma a segunda chamada que a produção faz,
    e é a mais barata que o leva: a Análise sobre a original, que declara que não houve proposta.
    """
    pipeline, arquivos = _paineis_mais_ricos(qtbot)

    tela = campos_da_tela(
        pipeline,
        modo="pos_hilic",
        correcao="none",
        n_lotes=2,
        eixos=("correction",),
        recomendacao={
            "correction_proposta": "batch_loess",
            "rationale": "o diagnóstico apontou deriva",
            "origem": "modelo",
        },
        rsd_cutoff=30.0,
    )
    matriz = "matriz-do-contrato"
    identidade = {IDENTIDADE_DA_MATRIZ: matriz}
    estado = EstadoDasEtapas()
    for etapa, entrada in zip(ETAPAS, ENTRADAS, strict=True):
        campos = estado.campos_da_entrada(etapa, entrada, tela, matriz)
        if etapa == ETAPAS[-1]:
            execucao = campos_da_rota(campos, etapa)
            break
        estado.disparou(
            etapa,
            fotografia_da_etapa(tela, etapa, identidade, entrada),
            campos=campos,
            entrada=entrada,
        )
        estado.concluiu(etapa, correcao="batch_loess" if etapa == ETAPAS[0] else None)
    sobre_a_original = campos_da_rota(
        EstadoDasEtapas().campos_da_entrada(ETAPAS[-1], "original", tela, matriz), ETAPAS[-1]
    )
    return frozenset(
        set(execucao)
        | set(sobre_a_original)
        | set(FIXOS_DECLARADOS)
        | set(arquivos.arquivos_da_execucao())
        | set(arquivos.campos_para_envio())
    )


# ---------------------------------------------------------------------------
# A DIREÇÃO DE SAÍDA — o que a janela envia contra o que a rota aceita
# ---------------------------------------------------------------------------


def test_todo_campo_que_a_janela_envia_e_aceito_pela_rota(
    campos_enviados: frozenset[str], contratos: dict[tuple[str, str], Any]
) -> None:
    """A direção que uma falha de contrato quebra em produção, com `422` e sem diagnóstico útil.

    A não-vacuidade vai **nesta função e antes**: um construtor que devolvesse `{}` faria a
    diferença ser vazia e a asserção afirmar cobertura sobre nada.
    """
    contrato = contratos[("POST", ROTA_DA_EXECUCAO)]
    assert len(campos_enviados) >= 10, (
        f"a janela envia {len(campos_enviados)} campos: {sorted(campos_enviados)}. Com menos de "
        "dez, os construtores devolveram quase nada e a asserção abaixo seria vácua"
    )
    assert contrato.entrada, "a rota não declara campo de entrada nenhum — o esquema mediu o vazio"

    nao_aceitos = sorted(campos_enviados - contrato.entrada)
    assert not nao_aceitos, (
        f"a janela envia campos que `POST {ROTA_DA_EXECUCAO}` não declara: {nao_aceitos}. "
        f"A rota aceita {sorted(contrato.entrada)}"
    )


def test_os_campos_aceitos_e_nunca_enviados_estao_declarados(
    campos_enviados: frozenset[str], contratos: dict[tuple[str, str], Any]
) -> None:
    """A direção que apodrece sem sintoma: campo novo na rota que a janela nunca preenche.

    **Igualdade exacta e não contenção**, e é o que separa este portão de uma lista de isenção que
    cresce: um campo novo na rota reprova até alguém decidir se a janela o envia ou por que não.
    """
    contrato = contratos[("POST", ROTA_DA_EXECUCAO)]
    sobram = contrato.entrada - campos_enviados
    assert sobram == frozenset(ACEITOS_E_NUNCA_ENVIADOS), (
        f"a rota aceita {sorted(sobram)} e a janela não os envia; a lista declarada é "
        f"{sorted(ACEITOS_E_NUNCA_ENVIADOS)}. Se um campo novo entrou na rota, decida: a janela "
        "passa a enviá-lo, ou ele entra em `ACEITOS_E_NUNCA_ENVIADOS` com o motivo"
    )
    sem_motivo = [c for c, motivo in ACEITOS_E_NUNCA_ENVIADOS.items() if not motivo.strip()]
    assert not sem_motivo, f"campos declarados sem motivo: {sem_motivo}"


def _gabarito_de(constroi: Any, contratos: dict[tuple[str, str], Any]) -> str:
    """O gabarito do esquema que corresponde ao caminho que `constroi` produz.

    **O casamento é por SUBSTITUIÇÃO e não por prefixo**, e a diferença é a que fez a versão
    anterior reprovar: `/trabalhos/{identificador}` e `/trabalhos/{identificador}/cancelar`
    partilham o prefixo, e só um deles reproduz `/trabalhos/IDENTIFICADOR`.

    Levanta quando o casamento não é exactamente um — zero significa rota que a API não declara,
    e mais de um significa que este oráculo voltou a ser ambíguo. As duas merecem reprovar.
    """
    import re

    concreta = constroi("IDENTIFICADOR")
    casados = {
        rota
        for _, rota in contratos
        if "{" in rota and re.sub(r"\{[^}]+\}", "IDENTIFICADOR", rota) == concreta
    }
    assert len(casados) == 1, (
        f"{concreta!r} casa {len(casados)} gabaritos do esquema: {sorted(casados)}"
    )
    return next(iter(casados))


def test_toda_rota_que_a_janela_nomeia_existe_no_esquema(
    contratos: dict[tuple[str, str], Any],
) -> None:
    """Rota escrita à mão é typo à espera de um `404`, e o `404` chega como *"desconhecido"*.

    `rota_do_progresso` devolve um caminho concreto; o esquema declara o gabarito. A comparação é
    feita **pelo gabarito**, derivado da função com um identificador conhecido — não por uma string
    de gabarito escrita aqui, que seria a segunda fonte de verdade.

    > **O ORÁCULO CASAVA POR PREFIXO, e isso era a FORMA e não a propriedade.**
    > Ele comparava `rota.split("{")[0]` com o começo do caminho concreto; com
    > `/trabalhos/{identificador}/cancelar` no esquema, o mesmo prefixo passou a casar **dois**
    > gabaritos e o teste reprovou com *"casa 2 gabaritos"*. O caminho concreto é o mesmo e o
    > esquema está certo — o que estava errado era o casamento.
    >
    > A propriedade é: **substituir o marcador pelo identificador reproduz o caminho concreto,
    > exactamente**. É o que `_gabarito_de` faz agora, e ela distingue as duas rotas sem lista de
    > exceção.
    """
    declaradas = {("POST", ROTA_DA_EXECUCAO)}
    declaradas |= {("POST", rota) for _, rota in ROTAS_DA_LEITURA}
    declaradas |= {("GET", _gabarito_de(rota_do_progresso, contratos))}
    declaradas |= {("POST", _gabarito_de(rota_de_cancelamento, contratos))}
    # As duas prévias do passo a passo; uma revisão achou-as fora daqui.
    declaradas |= {("POST", rota) for rota in ROTA_DA_ETAPA.values()}
    # A curva do LOESS de uma feature. Nasce aqui em vez de ficar fora como a da medição.
    declaradas |= {("POST", ROTA_DA_CURVA_DO_LOESS)}

    assert len(declaradas) == 9, f"esperava nove rotas declaradas pela janela: {sorted(declaradas)}"
    ausentes = sorted(par for par in declaradas if par not in contratos)
    assert not ausentes, (
        f"a janela nomeia rotas que a API não declara: {ausentes}. O esquema tem "
        f"{sorted(contratos)}"
    )


# As prévias do passo a passo. O portão acima só olhava o
# `POST /execucoes`, e o FastAPI DESCARTA campo de formulário que a rota não declara: um eixo novo
# que entrasse no formulário e no `/execucoes` e faltasse a `/etapas/pre-processamento` faria a
# prévia desenhar as distribuições de outra especificação com a etapa marcada «em dia».
#
# `auto` ENTROU nas duas listas, e foi o portão que o pediu: as duas prévias declaram o
# campo, e até ali a janela o mandava com a caixa «Automático» ticada — a lista vazia de `auto` era
# verdadeira porque ele era ENVIADO. A caixa saiu, e a igualdade exacta abaixo
# reprovou com *«aceita ['auto', 'orientation'] e a janela não os envia»*. A isenção tem gatilho:
# um caminho que voltasse a mandar `auto` a uma prévia faria a igualdade reprovar outra vez.
_AUTO_NUNCA_ENVIADO_PELAS_PREVIAS: Final = (
    "a janela não delega: a Correção leva o corretor do seletor, e «Seguir sugestões do "
    "app» põe nele o do Palpite, à vista e editável, em vez de deixar a rota escolher. A linha de "
    "comando continua a delegar com `--auto`, e é ela que o `[CA-9.8]` mede"
)
ACEITOS_E_NUNCA_ENVIADOS_PELA_ETAPA: Final[dict[int, dict[str, str]]] = {
    etapa: {
        "orientation": ACEITOS_E_NUNCA_ENVIADOS["orientation"],
        "auto": _AUTO_NUNCA_ENVIADO_PELAS_PREVIAS,
    }
    for etapa in (1, 2)
}


def _enviados_pela_etapa(qtbot: QtBot, etapa: int) -> frozenset[str]:
    """O que `_dispara_etapa` manda à prévia `etapa`: `campos_base` filtrado, e os arquivos."""
    pipeline, arquivos = _paineis_mais_ricos(qtbot)
    campos = pipeline.campos_base("pos_hilic", "none", 2, eixos=("correction",), rsd_cutoff=30.0)
    enviados = arquivos.arquivos_da_execucao() if etapa == 2 else arquivos.arquivos_para_envio()
    return frozenset(set(campos_da_rota(campos, etapa)) | set(enviados))


@pytest.mark.parametrize("etapa", sorted(ROTA_DA_ETAPA))
def test_os_campos_de_cada_previa_batem_com_a_rota_dela(
    qtbot: QtBot, contratos: dict[tuple[str, str], Any], etapa: int
) -> None:
    """As duas direções do portão do `/execucoes`, para cada prévia `/etapas/*`.

    A não-vacuidade vai nesta função e antes: sem ela, um `campos_base` vazio faria as duas
    diferenças vazias e as asserções afirmariam cobertura sobre nada.
    """
    rota = ROTA_DA_ETAPA[etapa]
    contrato = contratos[("POST", rota)]
    enviados = _enviados_pela_etapa(qtbot, etapa)
    assert len(enviados) >= 6, f"a etapa {etapa} envia {sorted(enviados)}: quase nada"
    assert contrato.entrada, f"`POST {rota}` não declara campo de entrada nenhum"

    nao_aceitos = sorted(enviados - contrato.entrada)
    assert not nao_aceitos, (
        f"a janela envia a `POST {rota}` campos que ela não declara: {nao_aceitos}"
    )
    declarados = ACEITOS_E_NUNCA_ENVIADOS_PELA_ETAPA[etapa]
    sobram = contrato.entrada - enviados
    assert sobram == frozenset(declarados), (
        f"`POST {rota}` aceita {sorted(sobram)} e a janela não os envia; a lista declarada é "
        f"{sorted(declarados)}"
    )
    assert all(motivo.strip() for motivo in declarados.values()), "isenção sem motivo"


def test_a_bijecao_de_saida_acusa_um_campo_inventado(
    campos_enviados: frozenset[str], contratos: dict[tuple[str, str], Any]
) -> None:
    """O par causal, **em processo**: os conjuntos são argumento, então não pode ficar inerte."""
    contrato = contratos[("POST", ROTA_DA_EXECUCAO)]
    assert not (campos_enviados - contrato.entrada), "a árvore limpa já acusava"
    plantado = campos_enviados | {"corretor_favorito"}
    assert plantado - contrato.entrada == {"corretor_favorito"}, (
        "a verificação não acusou um campo que a rota não declara"
    )


# ---------------------------------------------------------------------------
# A DIREÇÃO DE ENTRADA — o que a janela lê contra o que a rota devolve
# ---------------------------------------------------------------------------


def _leituras_da_janela() -> tuple[tuple[str, int, str, str], ...]:
    """`(arquivo, linha, base, chave)` para toda referência a chave nos módulos de produção.

    **TRÊS formas e não duas, e a terceira apareceu por um mutante que não aterrou.** Ao plantar um
    erro de digitação em `comparacao.get("mediana_antes")` a âncora não casou, e a leitura do
    arquivo mostrou por quê: `qualidade.py::texto_da_comparacao` faz
    `if "mediana_antes" not in comparacao`. É
    referência a chave com o **mesmo** modo de falha silencioso — `"mediana_ante" not in comparacao`
    é sempre verdadeiro, e o painel mostraria o estado vazio para sempre — e o extrator não a via.

    | Forma | Nó | Exemplo |
    |---|---|---|
    | acesso tolerante | `Call` de `.get` | `corpo.get("run_id")` |
    | acesso estrito | `Subscript` | `corpo["qc_rsd_median"]` |
    | teste de pertença | `Compare` com `In`/`NotIn` | `"mediana_antes" not in comparacao` |

    Por AST e não por `grep`: `corpo.get` dentro de um comentário ou de uma docstring não conta, uma
    chamada quebrada em duas linhas pelo formatador continua a ser uma chamada, e uma chave escrita
    com aspas simples dentro de uma f-string é o mesmo literal que com aspas duplas — que foi o
    outro motivo de a âncora do mutante não casar.
    """
    achados: list[tuple[str, int, str, str]] = []
    for caminho in sorted(JANELA.glob("*.py")):
        if caminho.name == "__init__.py":
            continue
        arvore = ast.parse(caminho.read_text(encoding="utf-8"))
        for no in ast.walk(arvore):
            base: ast.expr | None = None
            chave: str | None = None
            if (
                isinstance(no, ast.Call)
                and isinstance(no.func, ast.Attribute)
                and no.func.attr == "get"
                and no.args
                and isinstance(no.args[0], ast.Constant)
                and isinstance(no.args[0].value, str)
            ):
                base, chave = no.func.value, no.args[0].value
            elif (
                isinstance(no, ast.Subscript)
                and isinstance(no.slice, ast.Constant)
                and isinstance(no.slice.value, str)
            ):
                base, chave = no.value, no.slice.value
            elif (
                isinstance(no, ast.Compare)
                and len(no.ops) == 1
                and isinstance(no.ops[0], ast.In | ast.NotIn)
                and isinstance(no.left, ast.Constant)
                and isinstance(no.left.value, str)
            ):
                base, chave = no.comparators[0], no.left.value
            if base is None or chave is None:
                continue
            nome = (
                base.id
                if isinstance(base, ast.Name)
                else base.attr
                if isinstance(base, ast.Attribute)
                else f"<{type(base).__name__}>"
            )
            achados.append((caminho.name, no.lineno, nome, chave))
    return tuple(sorted(achados))


def test_a_extracao_de_leituras_nao_mede_o_vazio() -> None:
    """A âncora de todos os testes de entrada, e ela é o teste inteiro para eles.

    Medido em 23/08/2026: **9** bases distintas nos módulos da janela. Uma extração que devolvesse
    vazio faria as partições abaixo passarem sem olhar leitura nenhuma, e as três juntas afirmariam
    contrato completo sobre nada.
    """
    leituras = _leituras_da_janela()
    assert len(leituras) >= 30, (
        f"a extração achou {len(leituras)} leituras nos módulos da janela; uma queda desta ordem é "
        "defeito do extractor, não da janela"
    )
    bases = {base for _, _, base, _ in leituras}
    assert len(bases) >= 5, f"a extração achou {len(bases)} bases distintas: {sorted(bases)}"


def test_toda_base_de_leitura_esta_classificada() -> None:
    """A partição é TOTAL e fechada — base nova reprova até ser classificada.

    É a forma dos portões de AST deste projeto: **permitir uma forma** em vez de
    proibir as conhecidas. Um portão que só conferisse as bases que reconhece seria satisfeito por
    um extrator que não reconhecesse nenhuma.
    """
    classificadas = (
        CORPO_DE_RESPOSTA
        | set(RAIZES_ANINHADAS)
        | {RAIZ_DO_RELATORIO}
        | set(GERADAS_PELO_FRAMEWORK)
        | set(NAO_SAO_RESPOSTA)
    )
    assert classificadas, "não há classe declarada — a asserção abaixo seria vácua"
    lidas = {base for _, _, base, _ in _leituras_da_janela()}
    assert lidas, "a extração não achou base nenhuma"

    sem_classe = sorted(lidas - classificadas)
    assert not sem_classe, (
        f"bases de leitura sem classe declarada: {sem_classe}. Classifique cada uma: corpo de "
        "resposta, raiz aninhada com o esquema, o relatório de qualidade, gerada pelo framework, "
        "ou não-é-resposta com o motivo"
    )
    # E a direção oposta: classe declarada que já não é lida em lugar nenhum apodrece em silêncio.
    orfas = sorted(classificadas - lidas)
    assert not orfas, (
        f"classes declaradas que nenhuma leitura usa: {orfas}. Classe sem uso é isenção que "
        "sobrevive ao código que a justificava"
    )


def _chaves_de_resposta(contratos: dict[tuple[str, str], Any]) -> frozenset[str]:
    """A união das propriedades de resposta de toda rota que devolve JSON."""
    return frozenset().union(*(c.saida for c in contratos.values() if c.devolve_json))


def test_toda_chave_lida_de_corpo_e_declarada_por_alguma_rota(
    contratos: dict[tuple[str, str], Any],
) -> None:
    """A direção que falha em SILÊNCIO, e é a mais cara das duas.

    Uma chave mal escrita — `qc_rsd_medain` — devolve `None`, o painel mostra *"não calculada"* para
    sempre, e nada levanta. É o modo de falha que este portão existe para tornar impossível.

    **A união e não a rota exacta**, e o recorte vai declarado: mapear cada leitura à rota que a
    produziu exigiria uma análise de fluxo de dados. O que
    este portão afirma é *"a chave existe em ALGUMA resposta"*, que é o que apanha o erro de
    digitação; não afirma *"a chave existe na resposta desta rota"*.
    """
    declaradas = _chaves_de_resposta(contratos)
    assert len(declaradas) >= 20, (
        f"a união das respostas tem {len(declaradas)} chaves; com poucas, a asserção abaixo seria "
        "quase vácua"
    )
    do_framework = frozenset(CHAVES_DO_FRAMEWORK)
    assert do_framework, "não há chave de framework declarada — a isenção abaixo seria vácua"
    sem_motivo = [c for c, motivo in CHAVES_DO_FRAMEWORK.items() if not motivo.strip()]
    assert not sem_motivo, f"chaves de framework isentas sem motivo: {sem_motivo}"
    lidas = {
        (arquivo, linha, chave)
        for arquivo, linha, base, chave in _leituras_da_janela()
        if base in CORPO_DE_RESPOSTA
    }
    assert lidas, "nenhuma leitura de `corpo` — a asserção abaixo seria vácua"

    desconhecidas = sorted(
        (arquivo, linha, chave)
        for arquivo, linha, chave in lidas
        if chave not in declaradas and chave not in do_framework
    )
    assert not desconhecidas, (
        f"a janela lê de `corpo` chaves que rota nenhuma declara: {desconhecidas}. Uma chave "
        "errada devolve `None` e o painel mostra o estado vazio para sempre"
    )


def test_as_chaves_aninhadas_estao_no_esquema_do_campo_de_onde_vem() -> None:
    """As raízes aninhadas, cada uma contra o esquema que a produz — e não contra a união.

    Aqui a rota **é** conhecida, porque a variável vem de um campo nomeado, então o portão é mais
    estreito que o da união acima: `comparacao` só pode ter chaves de `RsdAntesEDepois`.
    """
    comp = dict(build_app().openapi()).get("components", {}).get("schemas", {})
    assert comp, "o esquema não tem componentes — as asserções abaixo seriam vácuas"

    for base, nome_do_esquema in RAIZES_ANINHADAS.items():
        propriedades = frozenset(comp.get(nome_do_esquema, {}).get("properties", {}))
        assert propriedades, (
            f"o esquema {nome_do_esquema!r}, declarado para a base {base!r}, não existe ou não tem "
            f"propriedades. O esquema tem {sorted(comp)}"
        )
        lidas = {
            (arquivo, linha, chave)
            for arquivo, linha, b, chave in _leituras_da_janela()
            if b == base
        }
        assert lidas, f"a base {base!r} está declarada e não é lida em lugar nenhum"
        fora = sorted(
            (arquivo, linha, chave) for arquivo, linha, chave in lidas if chave not in propriedades
        )
        assert not fora, (
            f"a base {base!r} lê chaves que {nome_do_esquema} não declara: {fora}. "
            f"O esquema tem {sorted(propriedades)}"
        )


def test_o_campo_qualidade_e_um_dict_livre_e_a_isencao_esta_declarada() -> None:
    """A isenção com motivo — e ela é **medida**, não suposta.

    O `qualidade` de `ExecucaoResposta` é `{"additionalProperties": true, "type": "object"}`: o
    OpenAPI não declara propriedade nenhuma ali, então o esquema **não pode** ser o oráculo daquela
    subárvore. Este teste afirma as duas metades: que o campo é de facto livre, e que o oráculo
    substituto — `QualityReport` — cobre o que a janela lê.
    """
    comp = dict(build_app().openapi()).get("components", {}).get("schemas", {})
    execucao = comp.get("ExecucaoResposta", {}).get("properties", {})
    assert "qualidade" in execucao, "`ExecucaoResposta` não tem o campo `qualidade`"
    qualidade = execucao["qualidade"]
    assert qualidade.get("additionalProperties") is True, (
        f"o campo `qualidade` deixou de ser um `dict` livre: {qualidade}. Se ele ganhou esquema, a "
        "isenção deste teste caducou e a bijeção passa a ser contra o esquema"
    )
    assert not qualidade.get("properties"), (
        "o campo `qualidade` passou a declarar propriedades — a isenção caducou"
    )

    campos_do_relatorio = frozenset(QualityReport.__dataclass_fields__)
    assert campos_do_relatorio, "`QualityReport` não tem campo nenhum — a asserção seria vácua"
    lidas = {
        (arquivo, linha, chave)
        for arquivo, linha, base, chave in _leituras_da_janela()
        if base == RAIZ_DO_RELATORIO
    }
    assert lidas, f"a base {RAIZ_DO_RELATORIO!r} está declarada e não é lida"
    fora = sorted(
        (arquivo, linha, chave)
        for arquivo, linha, chave in lidas
        if chave not in campos_do_relatorio
    )
    assert not fora, (
        f"a janela lê do relatório de qualidade chaves que `QualityReport` não tem: {fora}. "
        f"Os campos são {sorted(campos_do_relatorio)}"
    )


def test_a_bijecao_de_entrada_acusa_uma_chave_inventada(
    contratos: dict[tuple[str, str], Any],
) -> None:
    """O par causal da direção de entrada, **em processo**."""
    declaradas = _chaves_de_resposta(contratos)
    assert "qc_rsd_median" in declaradas, "a âncora saiu do esquema — o par abaixo mede outra coisa"
    assert "qc_rsd_medain" not in declaradas, (
        "a chave com o erro de digitação está declarada, então o par não separa nada"
    )


# ---------------------------------------------------------------------------
# O ACOPLAMENTO com o painel do recorte: `operaveis` vazio é MEDIÇÃO
# ---------------------------------------------------------------------------


def _chamadores_de_disparo(fonte: str) -> list[tuple[int, str]]:
    """Chamadas a `chama`/`enfileira` que **não** são a definição nem a chamada interna.

    A janela tem um funil: `chama` monta o `lambda` e delega a `enfileira`. Um chamador de PRODUÇÃO
    é qualquer outro sítio que invoque um dos dois — e hoje não há nenhum, que é o fato que
    sustenta `operaveis=frozenset()` no painel do recorte.
    """
    arvore = ast.parse(fonte)
    # `ast.walk` devolve nós SEM `lineno` — `arguments`, `Load`, `arg` —, e a primeira versão desta
    # varredura levantou `AttributeError` neles. O `getattr` não é defensivo por hábito: é o que
    # torna a varredura sobre uma árvore inteira possível sem enumerar os tipos que têm linha.
    dentro_do_funil = {
        linha
        for definicao in ast.walk(arvore)
        if isinstance(definicao, ast.FunctionDef) and definicao.name in {"chama", "enfileira"}
        for no in ast.walk(definicao)
        if (linha := getattr(no, "lineno", None)) is not None
    }
    return [
        (no.lineno, no.func.attr)
        for no in ast.walk(arvore)
        if isinstance(no, ast.Call)
        and isinstance(no.func, ast.Attribute)
        and no.func.attr in {"chama", "enfileira"}
        and no.lineno not in dentro_do_funil
    ]


def test_quem_dispara_rota_e_exactamente_quem_operaveis_declara(qtbot: QtBot) -> None:
    """A bijeção entre os painéis LIGADOS e o que o recorte declara operável.

    ## Este teste foi REESCRITO, e a reescrita era o plano

    A versão anterior afirmava que **nenhum** chamador de produção de `chama`/`enfileira` existia, e
    a docstring dela dizia: *"este teste REPROVA quando alguém ligar o primeiro botão, e é de
    propósito … este teste tem de ser reescrito para afirmar QUAL painel dispara"*. Ligaram-se
    quatro, ele reprovou, e esta é a reescrita.

    > **A propriedade guardada não mudou; o que mudou foi de que lado ela é medida.** Antes: o
    > conjunto dos que disparam é vazio, e `operaveis` também. Agora: os dois são o **mesmo**
    > conjunto — e é a igualdade que impede a declaração de envelhecer nas DUAS direções.

    **Ligar um painel sem o declarar** faz o recorte dizer *"sem disparo"* sobre algo que dispara,
    e **declarar sem ligar** faz o recorte dizer que há botão sobre botão morto. As duas reprovam
    aqui.

    ## O LADO ESQUERDO DA IGUALDADE JÁ FICOU CEGO, e ninguém viu

    Este teste lia a chave do painel do TEXTO exibido, procurando cabeçalhos `NO PAINEL «x»`. O
    recorte foi reescrito na língua de quem opera, e a chave saiu do texto de propósito —
    `recorte.py:142`, verbatim: *"`NO PAINEL «{s.onde}»` mostrava a chave do painel — `leitura`,
    `artefatos` — e quem opera não tem o que fazer com nenhum dos dois"*. O prefixo passou a ser
    `NESTE PROGRAMA — <título na língua de quem opera>`.

    A partir dali `declarados` era **sempre vazio**, e a igualdade reprovava com
    `[] != {seis painéis}`. Medido em 27/08/2026: o teste falha em `main` limpo desde `4e61ab3`.

    > **O instrumento lia a instância errada do referente certo.** O referente —
    > *"o que o recorte declara operável"* — continua a existir; o que sumiu foi a forma em que ele
    > estava escrito. E o desfecho foi **reprovação**, não silêncio, o que devia tê-lo tornado
    > visível — mas a marca do commit declarava `make check` verde, e este teste corre nele.

    **A reescrita mede as duas metades separadamente**, porque elas são propriedades diferentes:

    1. **A bijeção** compara `janela.operaveis` — a fonte derivada — contra quem de facto dispara.
    2. **A visibilidade** exige que cada painel operável apareça no texto sob `NESTE PROGRAMA`,
       traduzido. Sem ela, `operaveis` podia crescer sem que o usuário lesse nada a respeito, que é
       exactamente o que a versão antiga protegia ao ler do texto.
    """
    chamadores: list[tuple[str, int, str]] = []
    for caminho in sorted(JANELA.glob("*.py")):
        fonte = caminho.read_text(encoding="utf-8")
        chamadores += [(caminho.name, linha, nome) for linha, nome in _chamadores_de_disparo(fonte)]

    # A não-vacuidade vai antes: sem chamador nenhum, a comparação abaixo passaria com
    # `operaveis` vazio — e esse já foi o estado.
    assert chamadores, (
        "não há chamador de produção de `chama`/`enfileira`. Se o fluxo foi desligado, `operaveis` "
        "em `principal.py` tem de voltar a ser vazio, e este teste volta a afirmar a ausência"
    )

    # O que a janela DECLARA operável — a fonte derivada, que é o que o painel consome.
    janela = JanelaPrincipal(_TransporteEspia(), escolhedor=lambda: None)
    qtbot.addWidget(janela)
    declarados = set(janela.operaveis)

    # **E a metade da VISIBILIDADE**: cada operável tem de aparecer no texto que o usuário lê, sob
    # o prefixo dos cobertos. Sem esta, `operaveis` podia crescer sem que nada disso chegasse à
    # tela — que é a propriedade que a versão antiga guardava ao ler a chave do texto.
    cabecas = {t.split("\n")[0] for t in janela.recorte.textos()}
    for chave in sorted(declarados):
        secao = next((s for s in SECOES_DO_RECORTE if s.onde == chave), None)
        assert secao is not None, (
            f"o painel {chave!r} é declarado operável e não tem seção em `SECOES_DO_RECORTE`; o "
            "usuário não lê nada sobre ele"
        )
        titulo = PARA_QUEM_OPERA[secao.titulo][0]
        esperado = f"{PREFIXO_COBERTO} — {titulo}"
        assert esperado in cabecas, (
            f"o painel {chave!r} dispara rota e o recorte não o mostra como coberto. Esperava o "
            f"cabeçalho {esperado!r} entre {sorted(cabecas)}"
        )

    # E o que de facto DISPARA, medido por EMISSÃO e não por introspecção.
    #
    # **A primeira versão deste bloco perguntava ao Qt com `isSignalConnected`, e era a pergunta
    # errada.** Um sinal ligado a um slot que não chama rota nenhuma responderia *sim* — é a mesma
    # distância entre «composto» e «operável», uma camada acima. Emitir e contar as
    # chamadas ao transporte mede a ligação **de ponta a ponta**: se o slot não existe, não existe
    # chamada; se existe e não chama, também não.
    janela.arquivos._par = ParDeArquivos("m.csv", b"feature,QC1\n1,2\n", "s.csv", b"l,t\nQC1,1\n")
    # O par entra direto, sem a escolha que deduz o modo: o modo vai à mão, senão a chamada é
    # recusada por falta dele.
    janela.arquivos.modo.setEditText("pos_hilic")
    espia = _TransporteEspia()
    janela._transporte = espia

    dispara: dict[str, tuple[object, ...]] = {
        "leitura": (lambda: janela.leitura.pedir.emit("validacao"),),
        # **`ler_progresso` SAIU em 08/09/2026, e a segunda emissão saiu com ele.** O botão
        # «Ler progresso» ficava ao lado de uma barra que já se atualiza sozinha.
        # O painel continua a ser operável pelo sinal que sobra — a asserção é sobre a chave e não
        # sobre a contagem de sinais.
        # **O pipeline dispara pelas etapas desde 09/11/2026**: o sinal `disparar`, do botão
        # «Executar pipeline», saiu com a aba «Executar e exportar». A Análise é a que manda o
        # `/execucoes`; as outras duas etapas têm a prévia delas, pelo mesmo sinal. **E o painel
        # `artefatos` saiu daqui com a aba**: os downloads moram no Resultado das etapas, que são
        # parte do que a composição das páginas do pipeline mostra.
        "pipeline": (lambda: janela.pipeline.disparar_etapa.emit(3),),
        # **O painel 5 entrou depois, e este portão reprovou por acréscimo legítimo** — o
        # desfecho correto de um portão derivado: `multiverso` apareceu em `operaveis` e não aqui.
        # O sinal leva o identificador porque é o painel que tem o campo.
        CHAVE_DO_MULTIVERSO: (lambda: janela.multiverso.pedir.emit("abc123"),),
        # **O painel 3, e a emissão precisa de um trabalho REGISTRADO.** O sinal não leva
        # argumento — o painel pede *"atualiza todos"* —, e sem trabalho conhecido o slot escreve a
        # frase de repouso e **não chama rota**, que é o desenho certo. Registrar um primeiro é o
        # que
        # faz esta emissão medir a ligação em vez de o ramo vazio.
        CHAVE_DOS_TRABALHOS: (
            lambda: janela.registra_trabalho("execucao", "t-do-portao"),
            lambda: janela.trabalhos.atualizar.emit(),
        ),
    }
    ligados: set[str] = set()
    for chave, emissoes in dispara.items():
        antes = len(espia.chamadas)
        for emitir in emissoes:
            emitir()  # type: ignore[operator]
        janela.drena_a_piscina(5_000)
        if len(espia.chamadas) > antes:
            ligados.add(chave)
        # Antes, cada painel era medido com a espera do Palpite apagada aqui: com a
        # caixa «Automático» ticada, a emissão do `pipeline` pedia o Palpite e ficava à espera dele,
        # e a espera travava a correção seguinte. A caixa saiu, e a emissão vai direto à rota da
        # etapa.

    # O painel 1 não tem sinal que dispare rota — `par_escolhido` só avisa que houve escolha —, e
    # ele é operável porque os seus dados são o que TODA chamada leva: as quatro acima só
    # aconteceram porque ele tinha um par. Isso vai declarado em vez de derivado de um `connect`
    # que não existe.
    assert espia.chamadas, "nenhuma emissão produziu chamada — o painel 1 não forneceu o par"
    ligados.add("arquivos")

    assert declarados == ligados, (
        f"o recorte declara operáveis {sorted(declarados)} e os painéis que de facto disparam são "
        f"{sorted(ligados)}. Ligar sem declarar faz o painel dizer «sem disparo» sobre algo que "
        "dispara; declarar sem ligar faz dizer «NO PAINEL» sobre botão morto"
    )


class _TransporteEspia:
    """Um `Transporte` que REGISTRA e não fala. Devolve o mínimo que `cliente.envia` sabe ler."""

    def __init__(self) -> None:
        self.chamadas: list[tuple[str, str]] = []

    def _resposta(self, url: str, metodo: str) -> object:
        self.chamadas.append((metodo, url))

        # `SimpleNamespace` e não uma classe com atributos mutáveis de classe: `headers`
        # como `dict` de classe é `RUF012`, e a regra está certa — um duplo partilhado entre
        # chamadas é estado partilhado que ninguém declarou.
        from types import SimpleNamespace

        return SimpleNamespace(
            status_code=200,
            headers={"content-type": "application/json"},
            content=b"{}",
            text="{}",
            json=lambda: {},
        )

    def post(self, url: str, **kwargs: object) -> object:
        return self._resposta(url, "POST")

    def get(self, url: str, **kwargs: object) -> object:
        return self._resposta(url, "GET")


def test_a_varredura_de_chamadores_acusa_e_deixa_de_acusar() -> None:
    """O par causal, **em processo**: a fonte é argumento, então não pode ficar inerte.

    **As duas direções trocaram de lado com os slots ligados.** Antes a fonte limpa não acusava e
    a plantada acusava; hoje a fonte limpa **acusa quatro** — os quatro slots — e o que prova que a
    varredura não é «sempre acusa» é ela ficar **vazia** sobre uma fonte sem chamador. É a
    exigência da taxonomia de negativo plantado: ver o negativo **deixar de acusar** quando não há
    o que acusar.
    """
    limpa = (JANELA / "principal.py").read_text(encoding="utf-8")
    achados = _chamadores_de_disparo(limpa)

    # **NÃO se pina a contagem, e a primeira versão pinava.** Ela dizia `== 4` — os quatro slots —,
    # e a segunda perna do artefato fez **cinco**: o `202` encadeia um `GET`. O teste reprovou por
    # acréscimo legítimo.
    #
    # A contagem não é a propriedade. As duas direções são: **há** chamador — senão o teste vizinho
    # compara conjuntos vazios —, e a varredura **deixa** de acusar sobre fonte sem chamador. O
    # semântico — quem dispara é exactamente quem `operaveis` declara — é o teste acima.
    assert achados, f"a varredura não achou chamador de produção nenhum em principal.py: {achados}"

    # A direção que impede «sempre acusa»: uma fonte que só define o funil e não o chama.
    sem_chamador = (
        "class Janela:\n"
        "    def chama(self, metodo, rota, *, chave):\n"
        "        self.enfileira(lambda: None, chave=chave)\n"
        "\n"
        "    def enfileira(self, chamada, *, chave):\n"
        "        pass\n"
    )
    assert _chamadores_de_disparo(sem_chamador) == [], (
        "a varredura acusou uma fonte cujo único `enfileira` está DENTRO do funil"
    )

    plantada = limpa + (
        "\n\nclass _PlantaDaU9:\n"
        "    def liga(self, janela: object) -> None:\n"
        '        janela.chama("POST", "/execucoes", chave="x")\n'
    )
    achados = _chamadores_de_disparo(plantada)
    assert achados, "a varredura não acusou um chamador de produção plantado"
    assert achados[0][1] == "chama", f"acusou a chamada errada: {achados}"


def test_d336_cada_sinal_novo_produz_a_chamada_dele(qtbot: QtBot) -> None:
    """**A COSTURA, e uma revisão adversarial provou que ela podia ser cortada em silêncio.**

    Planta medida: um plugin que faz `disconnect()` nas quatro ligações novas depois de a janela
    ser construída deixava **307 testes verdes**. O painel era testado isolado, a rota era testada
    isolada, e o fio entre os dois — que é o que faz o botão fazer alguma coisa — não era.

    Os portões existentes não acusavam, e a razão é dupla:

    | Portão | Por que não vê |
    |---|---|
    | o da bijeção de operáveis | mede por PAINEL, e `multiverso` já contava como ligado |
    | o de método público de painel | ignora nomes com `_`, e os quatro slots são privados |

    > **A asserção passa de «o painel dispara» para «cada SINAL produz a chamada dele».** É a
    > diferença entre medir o painel e medir o fio.

    **Método E caminho**, e não só «houve chamada»: um slot ligado à rota errada produziria chamada
    e passaria numa asserção de contagem.
    """
    janela = JanelaPrincipal(_TransporteEspia(), escolhedor=lambda: None)
    qtbot.addWidget(janela)
    janela.arquivos._par = ParDeArquivos("m.csv", b"feature,QC1\n1,2\n", "s.csv", b"l,t\nQC1,1\n")
    # O par entra direto, sem a escolha que deduz o modo: o modo vai à mão, senão a chamada é
    # recusada por falta dele.
    janela.arquivos.modo.setEditText("pos_hilic")
    espia = _TransporteEspia()
    janela._transporte = espia

    esperado: tuple[tuple[str, object, tuple[str, str]], ...] = (
        (
            "multiverso.disparar",
            lambda: janela.multiverso.disparar.emit(janela.multiverso.escolhas_do_disparo()),
            ("POST", "/multiversos"),
        ),
        (
            "multiverso.ler_progresso",
            lambda: janela.multiverso.ler_progresso.emit("abc123"),
            ("GET", "/multiversos/abc123"),
        ),
        (
            "multiverso.cancelar",
            lambda: janela.multiverso.cancelar.emit("abc123"),
            ("POST", "/multiversos/abc123/cancelar"),
        ),
        (
            "trabalhos.cancelar",
            lambda: (
                janela.registra_trabalho("corrigir", "t-1"),
                janela.trabalhos.cancelar.emit("corrigir"),
            ),
            ("POST", "/trabalhos/t-1/cancelar"),
        ),
        # A curva do LOESS. O sinal leva o nome da feature, que é o que o painel tem.
        (
            "pipeline.ver_curva_do_loess",
            lambda: janela.pipeline.ver_curva_do_loess.emit("F1"),
            ("POST", ROTA_DA_CURVA_DO_LOESS),
        ),
    )

    for nome, emitir, alvo in esperado:
        antes = len(espia.chamadas)
        emitir()  # type: ignore[operator]
        janela.drena_a_piscina(5_000)
        novas = espia.chamadas[antes:]
        assert novas, f"{nome} não produziu chamada nenhuma: o sinal está desligado"
        metodos_e_rotas = {(m, u.split("://", 1)[-1].split("/", 1)[-1]) for m, u in novas}
        casou = any(m == alvo[0] and u.endswith(alvo[1].lstrip("/")) for m, u in metodos_e_rotas)
        assert casou, f"{nome} chamou {sorted(metodos_e_rotas)} e não {alvo}"


# ---------------------------------------------------------------------------
# As cargas das figuras entram na partição, e a entrada de `RAIZES_ANINHADAS` é DEVEDORA
# ---------------------------------------------------------------------------
#
# ## A entrada em `RAIZES_ANINHADAS` NÃO PODE nascer antes do leitor, e isto foi MEDIDO
#
# O requisito pede *"um modelo declarado em `esquemas.py` e uma entrada em `RAIZES_ANINHADAS`"*,
# e as duas metades têm pré-condições diferentes. A segunda é **bidirecional**: plantada uma
# entrada `{"zzz_planta": "RsdAntesEDepois"}` nesta árvore, DOIS portões reprovam —
#
#     test_toda_base_de_leitura_esta_classificada
#         AssertionError: classes declaradas que nenhuma leitura usa: ['zzz_planta']
#     test_as_chaves_aninhadas_estao_no_esquema_do_campo_de_onde_vem
#         AssertionError: a base 'zzz_planta' está declarada e não é lida em lugar nenhum
#
# — porque nenhum módulo de `desktop/janela/` lia `distribuicoes.<chave>` antes de existir o
# `PainelDeFigura`. A entrada esperou o leitor, e os portões estão certos: uma classe declarada sem
# leitor é a lista de isenção que mata um portão.
#
# **O que fica AQUI é a pré-condição da entrada**, e ela não é automática: um `BaseModel` que rota
# nenhuma alcança **não aparece** em `components.schemas` — foi o que se mediu com
# `MedicaoDoCorretorResposta` (`rotas.py:1286`), cujos oito campos estavam ausentes da união porque
# o `POST` declara `response_model=TrabalhoAceito`. Sem o nome no OpenAPI, a entrada de
# `RAIZES_ANINHADAS` reprova com *"o esquema não existe"*, e o painel só o descobriria depois de
# escrito.

NOME_DO_ESQUEMA_DAS_DISTRIBUICOES = "ResumoDasDistribuicoes"


def test_u11_as_cargas_das_figuras_estao_no_OPENAPI_com_o_nome_que_a_entrada_vai_citar() -> None:  # noqa: N802
    """**A pré-condição da entrada em `RAIZES_ANINHADAS`.**

    **O nome da raiz é parte do contrato e não gosto**, e o comentário de `RAIZES_ANINHADAS` mede o
    custo de o tratar como gosto: o relatório do corte chama-se `corte` e **não** `relatorio`, e a
    primeira versão daquela unidade reusou o nome errado — o portão reprovou por três chaves.

    Aqui o par é `distribuicoes` → `ResumoDasDistribuicoes`, e a armadilha simétrica seria apontar
    para `RsdAntesEDepois`: os dois são «o antes e o depois», os dois são campo de
    `ExecucaoResposta`, e um deles tem oito propriedades que não são as desta carga.

    **As duas asserções de não-vacuidade vêm primeiro**, porque `frozenset()` é subconjunto de tudo
    e a bijeção abaixo passaria sobre um esquema inexistente.
    """
    comp = dict(build_app().openapi()).get("components", {}).get("schemas", {})
    assert comp, "o esquema não tem componentes — as asserções abaixo seriam vácuas"

    for nome in (NOME_DO_ESQUEMA_DAS_DISTRIBUICOES, "CaixaDaDistribuicao", "CaixaPorAmostra"):
        propriedades = frozenset(comp.get(nome, {}).get("properties", {}))
        assert propriedades, (
            f"{nome!r} não está em `components.schemas`, ou não tem propriedades. Um `BaseModel` "
            f"que rota nenhuma alcança não entra no OpenAPI, e a entrada de `RAIZES_ANINHADAS` da "
            f"U3 reprovaria por «o esquema não existe». O esquema tem {sorted(comp)}"
        )

    resumo = frozenset(comp[NOME_DO_ESQUEMA_DAS_DISTRIBUICOES]["properties"])
    par_de_rsd = frozenset(comp["RsdAntesEDepois"]["properties"])
    assert not resumo <= par_de_rsd, (
        "as propriedades do resumo cabem todas em `RsdAntesEDepois` — o par de esquemas ficou "
        "ambíguo, e uma entrada apontada para o irmão errado passaria (é a armadilha de `corte` "
        "contra `relatorio`)"
    )

    # O que a figura consome tem de estar lá: o recorte, a procedência e as duas metades.
    #
    # **`n_features_antes` está na lista por decisão, e não por simetria.** A interseção
    # fica, e a figura é obrigada a declará-la: a legenda lê-se *"N de M features, as que
    # sobreviveram ao corte"*, e as duas pontas viajam aqui — `N` é `n_features_comparadas` e `M` é
    # `n_features_antes`. Sem `M` no corpo, a figura teria de o derivar da contagem de
    # `StageResult`, e duas fontes de uma verdade só divergem no dia em que uma etapa nova entrar.
    exigidas = {
        "caixas_antes",
        "caixas_depois",
        "caixas_por_amostra_antes",
        "caixas_por_amostra_depois",
        "medias_por_feature_antes",
        "medias_por_amostra_antes",
        "n_caixas",
        "n_caixas_por_amostra",
        "n_features_comparadas",
        "n_features_antes",
        "n_amostras_comparadas",
        "amostrada",
        "amostrada_por_amostra",
        "seed",
        "matriz_antes",
        "matriz_depois",
    }
    assert exigidas <= resumo, f"faltam ao resumo: {sorted(exigidas - resumo)}"

    # **A caixa é a de Tukey, e a bijeção é com o que `Axes.bxp` consome.** O lado do desenho passa
    # `whislo/q1/med/q3/whishi` prontos, e o par de nomes é o contrato: um `minimo` que voltasse a
    # aparecer aqui seria a convenção antiga a atravessar sem nada acusar, e a figura divergiria da
    # do MetaboAnalyst aberta ao lado. `n_discrepantes` é o traço dos pontos que o bigode esconde.
    caixa = frozenset(comp["CaixaDaDistribuicao"]["properties"])
    assert caixa == {
        "feature",
        "bigode_inferior",
        "q1",
        "mediana",
        "q3",
        "bigode_superior",
        "n_discrepantes",
    }, (
        f"a caixa deixou de ser a de Tukey: {sorted(caixa)}. O extremo é o bigode — o ponto mais "
        "afastado ainda dentro de 1,5xIQR do quartil —, e o que fica fora dele sai do corpo com "
        "`n_discrepantes` a contá-lo em vez de desaparecer sem traço"
    )

    # **A caixa por AMOSTRA é a mesma forma com OUTRO nome no primeiro campo, e as duas metades
    # desta asserção são precisas.** A primeira fixa o campo que a distingue: um `feature` que
    # aparecesse aqui poria nome de amostra num campo com o nome da outra unidade. A segunda fixa
    # que os outros seis **não** divergiram da irmã — as duas saem da mesma redução, e um campo a
    # mais de um lado seria a segunda fonte a nascer.
    por_amostra = frozenset(comp["CaixaPorAmostra"]["properties"])
    assert "amostra" in por_amostra and "feature" not in por_amostra, (
        f"a caixa por amostra não é keyed por amostra: {sorted(por_amostra)}"
    )
    assert por_amostra - {"amostra"} == caixa - {"feature"}, (
        f"os seis números divergiram entre os dois eixos: por amostra {sorted(por_amostra)}, por "
        f"feature {sorted(caixa)}. A redução é uma só (`rotas._tukey`), e formas diferentes aqui "
        "significam que uma das duas ganhou campo que a outra não tem"
    )


# ---------------------------------------------------------------------------
# O PAR de PCAs, e a entrada de `RAIZES_ANINHADAS` é DEVEDORA do painel da correção
# ---------------------------------------------------------------------------
#
# A entrada em `RAIZES_ANINHADAS` não pode nascer antes do leitor, e o motivo é o MEDIDO no bloco
# das distribuições acima: uma entrada declarada que nenhum módulo de `desktop/janela/` lê reprova
# em DOIS portões — `test_toda_base_de_leitura_esta_classificada` por classe sem uso, e
# `test_as_chaves_aninhadas_...` por base declarada e não lida. Guardar o par não é desenhá-lo, e
# nenhum módulo da janela lê `<par>.<chave>` antes de existir o painel da correção.
#
# **O que fica AQUI são as duas pré-condições dela**, e nenhuma é automática:
#
# 1. o esquema tem de estar em `components.schemas` com o nome que a entrada vai citar — um
#    `BaseModel` que rota nenhuma alcança **não aparece** lá, e o painel descobriria isso só
#    depois de escrito;
# 2. a raiz tem de ser **lida** do corpo por algum módulo da janela, senão o corpo cresce e a
#    janela nunca vê a carga — que é o defeito que `test_toda_chave_lida_de_corpo_...` não apanha,
#    porque ele mede a direção oposta.
NOME_DO_ESQUEMA_DO_PAR_DE_PCAS: Final = "ParDePcas"
RAIZ_DO_PAR_DE_PCAS: Final = "pcas_da_correcao"


def test_u2_o_par_de_pcas_esta_no_OPENAPI_com_o_nome_que_a_entrada_vai_citar() -> None:  # noqa: N802
    """**A pré-condição da entrada em `RAIZES_ANINHADAS`.**

    **A armadilha simétrica é apontar para `ResumoDaPca`**, e ela é mais convidativa aqui do que a
    de `corte` contra `relatorio` foi lá: os dois esquemas são «a PCA», os dois são campo de um
    corpo de resposta, e um deles descreve **uma** matriz enquanto o outro descreve **duas**. Um
    consumidor que bijetasse contra o esquema errado leria `scores_pc1` — que este não tem — e
    encontraria o estado vazio para sempre.

    **As duas asserções de não-vacuidade vêm primeiro**: `frozenset()` é subconjunto de tudo, e a
    bijeção abaixo passaria sobre um esquema inexistente.
    """
    comp = dict(build_app().openapi()).get("components", {}).get("schemas", {})
    assert comp, "o esquema não tem componentes — as asserções abaixo seriam vácuas"

    propriedades = frozenset(comp.get(NOME_DO_ESQUEMA_DO_PAR_DE_PCAS, {}).get("properties", {}))
    assert propriedades, (
        f"{NOME_DO_ESQUEMA_DO_PAR_DE_PCAS!r} não está em `components.schemas`, ou não tem "
        "propriedades. Um `BaseModel` que rota nenhuma alcança não entra no OpenAPI, e a entrada "
        f"de `RAIZES_ANINHADAS` da U3 reprovaria por «o esquema não existe». O esquema tem "
        f"{sorted(comp)}"
    )

    uma_matriz = frozenset(comp.get("ResumoDaPca", {}).get("properties", {}))
    assert uma_matriz, "âncora: `ResumoDaPca` sumiu do esquema, e a distinção abaixo seria vácua"
    assert not propriedades <= uma_matriz, (
        "as propriedades do par cabem todas em `ResumoDaPca` — o par de esquemas ficou ambíguo, e "
        "uma entrada apontada para o irmão errado passaria. O par descreve DUAS matrizes e aquele "
        "descreve uma"
    )

    # **Os dois lados são SIMÉTRICOS, e a asserção é derivada e não uma lista redigitada.** Um
    # campo que existisse só de um lado faria a figura declarar de um painel o que mediu do
    # outro.
    antes = {p.removesuffix("_antes") for p in propriedades if p.endswith("_antes")}
    depois = {p.removesuffix("_depois") for p in propriedades if p.endswith("_depois")}
    assert antes, "nenhum campo com sufixo `_antes` — a igualdade abaixo seria entre dois vazios"
    assert antes == depois, (
        f"os dois lados do par divergiram: só no «antes» {sorted(antes - depois)}, só no «depois» "
        f"{sorted(depois - antes)}"
    )

    # O que a figura consome de cada lado, e os dois rótulos que tornam os números comparáveis.
    # `digest_das_colunas_retidas` está na lista porque a figura tem de dizer **quando os dois
    # espaços divergem**, e contagem igual não é conjunto igual.
    exigidas_por_lado = {
        "scores_pc1",
        "scores_pc2",
        "tipos_de_amostra",
        "variancia_explicada",
        "qc_dispersion_ratio",
        "n_features_usadas",
        "n_features_totais",
        "digest_das_colunas_retidas",
        "matriz",
    }
    assert exigidas_por_lado <= antes, (
        f"faltam a cada lado do par: {sorted(exigidas_por_lado - antes)}"
    )
    assert {"pretransformacao", "n_componentes"} <= propriedades, (
        "os dois rótulos que tornam o número comparável não viajam: sem eles, "
        "`qc_dispersion_ratio` daqui e o do `QualityReport` têm o mesmo nome e medem sobre "
        "pré-transformações diferentes"
    )


def test_u2_a_raiz_do_par_de_pcas_e_declarada_pela_rota_e_lida_pela_janela(
    contratos: dict[tuple[str, str], Any],
) -> None:
    """A segunda pré-condição: a raiz atravessa **e** alguém do outro lado a apanha.

    **É a direção que `test_toda_chave_lida_de_corpo_e_declarada_por_alguma_rota` NÃO mede.**
    Aquele portão parte das leituras e pergunta se a rota as declara; este parte da carga nova e
    pergunta se a janela a lê. Sem ele, o corpo pode crescer com uma raiz que nenhum painel guarda,
    e o desfecho é o silêncio: a figura nasceria a mostrar o estado vazio para sempre.

    **Remover `PainelDeGraficos.recebe_pca_do_par` faz este teste acusar**, que é o par causal —
    e é ele que substitui, enquanto não há leitura aninhada, o portão de raiz aninhada.
    """
    declaradas = _chaves_de_resposta(contratos)
    assert len(declaradas) >= 20, (
        f"a união das respostas tem {len(declaradas)} chaves; com poucas, a asserção abaixo seria "
        "quase vácua"
    )
    assert RAIZ_DO_PAR_DE_PCAS in declaradas, (
        f"rota nenhuma declara {RAIZ_DO_PAR_DE_PCAS!r} na resposta: o par de PCAs não atravessa o "
        "fio, e a figura da U3 não teria de onde sair"
    )

    # **E ela é OBRIGATÓRIA**, pelo argumento de `distribuicoes`: o par não depende do par de
    # grupos nem de escolha nenhuma — toda execução tem uma matriz ingerida e uma corrigida —,
    # então não há aqui o *"não calculado"*, distinto de *"calculado e vazio"*.
    execucao = dict(build_app().openapi()).get("components", {}).get("schemas", {})
    obrigatorios = execucao.get("ExecucaoResposta", {}).get("required", [])
    assert obrigatorios, "`ExecucaoResposta` não declara campo obrigatório — a asserção seria vácua"
    assert RAIZ_DO_PAR_DE_PCAS in obrigatorios, (
        f"{RAIZ_DO_PAR_DE_PCAS!r} é opcional no esquema, e o servidor passa a poder omiti-la: a "
        "ausência volta a ser um estado que o painel tem de representar, sem haver o fato "
        "«não calculado» que a justificasse"
    )

    leituras = _leituras_da_janela()
    assert leituras, "a extração não achou leitura nenhuma — a asserção abaixo seria vácua"
    lida_por = sorted(
        (arquivo, linha)
        for arquivo, linha, base, chave in leituras
        if base in CORPO_DE_RESPOSTA and chave == RAIZ_DO_PAR_DE_PCAS
    )
    assert lida_por, (
        f"a janela não lê {RAIZ_DO_PAR_DE_PCAS!r} do corpo em módulo nenhum de `desktop/janela/`. "
        "A raiz atravessa o fio e ninguém a apanha: quem a consome é "
        "`PainelDeGraficos.recebe_pca_do_par`"
    )


# ---------------------------------------------------------------------------
# Os dois vetores de RSD, e a entrada de `RAIZES_ANINHADAS` é do violino
# ---------------------------------------------------------------------------
#
# A entrada em `RAIZES_ANINHADAS` não nasce aqui, e o motivo é o MEDIDO no bloco das
# distribuições: uma base declarada que nenhum módulo de `desktop/janela/` lê reprova em DOIS
# portões. Guardar não é desenhar, e quem lê `<base>.<chave>` é o construtor do violino. A entrada
# é dívida dele, como `"par"` foi dívida do painel da correção.
#
# **O que fica AQUI são as duas mesmas pré-condições**, e nenhuma é automática: o esquema no
# OpenAPI com o nome que a entrada vai citar, e a raiz declarada pela rota **e** lida pela janela.
NOME_DO_ESQUEMA_DO_RSD_POR_FEATURE: Final = "RsdPorFeature"
RAIZ_DO_RSD_POR_FEATURE: Final = "rsd_por_feature"


def test_u6_o_rsd_por_feature_esta_no_OPENAPI_com_o_nome_que_a_entrada_vai_citar() -> None:  # noqa: N802
    """**A pré-condição da entrada em `RAIZES_ANINHADAS`.**

    **A armadilha simétrica é apontar para `RsdAntesEDepois`**, e ela é a mais convidativa das três
    que este arquivo já registrou: os dois esquemas são «o RSD», os dois são campo de
    `ExecucaoResposta`, os dois declaram `rsd_variant` e `rsd_estimator`. E medem coisas
    **diferentes** — aquele é um par de **medianas** sobre os QCs antes e depois da correção; este
    são dois **vetores por feature** sobre a matriz corrigida, um dos QCs e outro das amostras. Um
    consumidor que bijetasse contra o irmão errado procuraria `mediana_antes`, não a encontraria, e
    ficaria no estado vazio para sempre.

    **As asserções de não-vacuidade vêm primeiro**: `frozenset()` é subconjunto de tudo, e a
    bijeção abaixo passaria sobre um esquema inexistente.
    """
    comp = dict(build_app().openapi()).get("components", {}).get("schemas", {})
    assert comp, "o esquema não tem componentes — as asserções abaixo seriam vácuas"

    propriedades = frozenset(comp.get(NOME_DO_ESQUEMA_DO_RSD_POR_FEATURE, {}).get("properties", {}))
    assert propriedades, (
        f"{NOME_DO_ESQUEMA_DO_RSD_POR_FEATURE!r} não está em `components.schemas`, ou não tem "
        "propriedades. Um `BaseModel` que rota nenhuma alcança não entra no OpenAPI, e a entrada "
        f"de `RAIZES_ANINHADAS` da U7 reprovaria por «o esquema não existe». O esquema tem "
        f"{sorted(comp)}"
    )

    medianas = frozenset(comp.get("RsdAntesEDepois", {}).get("properties", {}))
    assert medianas, "âncora: `RsdAntesEDepois` sumiu do esquema, e a distinção abaixo seria vácua"
    assert not propriedades <= medianas, (
        "as propriedades dos vetores cabem todas em `RsdAntesEDepois` — o par de esquemas ficou "
        "ambíguo, e uma entrada apontada para o irmão errado passaria. Aquele traz duas MEDIANAS "
        "sobre antes e depois; este traz dois VETORES por feature sobre a matriz corrigida"
    )

    # **Os dois lados são SIMÉTRICOS, e a asserção é derivada e não uma lista redigitada** — a
    # mesma forma do portão do par de PCAs. Um campo que existisse só de um lado faria a figura
    # declarar de uma população o que mediu da outra.
    dos_qcs = {p.removesuffix("_dos_qcs") for p in propriedades if p.endswith("_dos_qcs")}
    das_amostras = {
        p.removesuffix("_das_amostras") for p in propriedades if p.endswith("_das_amostras")
    }
    assert dos_qcs, (
        "nenhum campo com sufixo `_dos_qcs` — a igualdade abaixo seria entre dois vazios"
    )
    assert dos_qcs == das_amostras, (
        f"os dois lados divergiram: só nos QCs {sorted(dos_qcs - das_amostras)}, só nas amostras "
        f"{sorted(das_amostras - dos_qcs)}"
    )

    # O que o violino consome de cada lado. `n_features_finitas_*` está aqui porque é o `n` que a
    # imagem escreve — **de features, não de injeções** —, e `n_injecoes_*` está porque *"não há QC
    # nenhum"* e *"há sete QCs e nenhuma feature com RSD definido"* são fatos diferentes que a
    # contagem de finitos sozinha colapsaria.
    exigidas_por_lado = {"rsd", "n_features_finitas", "n_injecoes"}
    assert exigidas_por_lado <= dos_qcs, (
        f"faltam a cada lado: {sorted(exigidas_por_lado - dos_qcs)}"
    )
    assert {"rsd_variant", "rsd_estimator", "n_features", "matriz", "corretor"} <= propriedades, (
        "faltam os rótulos que tornam os dois vetores comparáveis: sem `rsd_variant` e "
        "`rsd_estimator` o número não é comparável com nenhum outro RSD do projeto, e sem "
        "`n_features` a figura não tem o denominador de «N de M features»"
    )


def test_u6_a_raiz_do_rsd_por_feature_e_declarada_pela_rota_e_lida_pela_janela(
    contratos: dict[tuple[str, str], Any],
) -> None:
    """A segunda pré-condição: a carga atravessa **e** alguém do outro lado a apanha.

    **É a direção que `test_toda_chave_lida_de_corpo_e_declarada_por_alguma_rota` NÃO mede.**
    Aquele portão parte das leituras e pergunta se a rota as declara; este parte da carga nova e
    pergunta se a janela a lê.

    **Remover `PainelDeGraficos.recebe_rsd_por_feature` faz este teste acusar**, que é o par causal
    — e é ele que substitui, enquanto não há leitura aninhada, o portão de raiz aninhada.
    """
    declaradas = _chaves_de_resposta(contratos)
    assert len(declaradas) >= 20, (
        f"a união das respostas tem {len(declaradas)} chaves; com poucas, a asserção abaixo seria "
        "quase vácua"
    )
    assert RAIZ_DO_RSD_POR_FEATURE in declaradas, (
        f"rota nenhuma declara {RAIZ_DO_RSD_POR_FEATURE!r} na resposta: os dois vetores não "
        "atravessam o fio, e o violino da U7 não teria de onde sair"
    )

    # **E ela é OBRIGATÓRIA**, pelo argumento de `distribuicoes` e do par de PCAs: os dois vetores
    # não dependem do par de grupos nem de escolha nenhuma — toda execução tem uma matriz
    # corrigida —, então não há aqui o *"não calculado"*, distinto de *"calculado e vazio"*. O
    # lado dos QCs **vazio** é «calculado e vazio», e é o que a matriz sem QC produz.
    esquemas = dict(build_app().openapi()).get("components", {}).get("schemas", {})
    obrigatorios = esquemas.get("ExecucaoResposta", {}).get("required", [])
    assert obrigatorios, "`ExecucaoResposta` não declara campo obrigatório — a asserção seria vácua"
    assert RAIZ_DO_RSD_POR_FEATURE in obrigatorios, (
        f"{RAIZ_DO_RSD_POR_FEATURE!r} é opcional no esquema, e o servidor passa a poder omiti-la: "
        "a ausência volta a ser um estado que o painel tem de representar, sem haver o fato "
        "«não calculado» que a justificasse"
    )

    leituras = _leituras_da_janela()
    assert leituras, "a extração não achou leitura nenhuma — a asserção abaixo seria vácua"
    lida_por = sorted(
        (arquivo, linha)
        for arquivo, linha, base, chave in leituras
        if base in CORPO_DE_RESPOSTA and chave == RAIZ_DO_RSD_POR_FEATURE
    )
    assert lida_por, (
        f"a janela não lê {RAIZ_DO_RSD_POR_FEATURE!r} do corpo em módulo nenhum de "
        "`desktop/janela/`. A raiz atravessa o fio e ninguém a apanha: quem a consome é "
        "`PainelDeGraficos.recebe_rsd_por_feature`"
    )


# ---------------------------------------------------------------------------
# A composição por classe, com a entrada de `RAIZES_ANINHADAS`
# ---------------------------------------------------------------------------
#
# A entrada NASCE aqui. As de `par` e de `rsd_por_feature` esperaram o leitor pelo motivo medido
# no bloco das distribuições: uma base declarada que nenhum módulo de `desktop/janela/` lê reprova
# em DOIS portões. Aqui o fio e o desenho entraram juntos, então a base e o leitor nascem juntos.
NOME_DO_ESQUEMA_DA_COMPOSICAO: Final = "ComposicaoPorClasse"
RAIZ_DA_COMPOSICAO: Final = "composicao_por_classe"


def test_u8_a_composicao_esta_no_OPENAPI_com_o_nome_que_a_entrada_cita() -> None:  # noqa: N802
    """**A pré-condição da entrada em `RAIZES_ANINHADAS`.**

    **A armadilha simétrica é apontar para `ClasseDaComposicao`**, que é o esquema do **elemento**
    da lista: os dois nomes começam igual, os dois estão no OpenAPI, e são «a composição». Um
    consumidor que bijetasse contra o irmão errado procuraria `grupo` na raiz, não o encontraria, e
    ficaria no estado vazio para sempre.

    **As asserções de não-vacuidade vêm primeiro**: `frozenset()` é subconjunto de tudo, e a
    bijeção abaixo passaria sobre um esquema inexistente.
    """
    comp = dict(build_app().openapi()).get("components", {}).get("schemas", {})
    assert comp, "o esquema não tem componentes — as asserções abaixo seriam vácuas"

    propriedades = frozenset(comp.get(NOME_DO_ESQUEMA_DA_COMPOSICAO, {}).get("properties", {}))
    assert propriedades, (
        f"{NOME_DO_ESQUEMA_DA_COMPOSICAO!r} não está em `components.schemas`, ou não tem "
        "propriedades. Um `BaseModel` que rota nenhuma alcança não entra no OpenAPI, e a entrada "
        f"de `RAIZES_ANINHADAS` reprovaria por «o esquema não existe». O esquema tem {sorted(comp)}"
    )

    da_classe = frozenset(comp.get("ClasseDaComposicao", {}).get("properties", {}))
    assert da_classe, "âncora: `ClasseDaComposicao` sumiu do esquema, e a distinção seria vácua"
    assert not propriedades & da_classe, (
        f"o conjunto e o elemento partilham propriedades ({sorted(propriedades & da_classe)}): "
        "uma entrada apontada para o irmão errado passaria. Aquele descreve UMA classe; este "
        "descreve o conjunto mais os rótulos que o tornam interpretável"
    )

    # O que a figura consome da raiz. **`n_features` está aqui porque é o denominador do
    # rodapé**, e `n_injecoes_fora_da_populacao` porque sem ele quem some as injeções das classes e
    # compare com o total do arquivo conclui que a figura perdeu amostras.
    exigidas = {
        "classes",
        "matriz",
        "populacao",
        "n_features",
        "n_injecoes_fora_da_populacao",
        "definicao_de_faltante",
        "definicao_de_abundancia",
    }
    assert exigidas <= propriedades, f"faltam à raiz: {sorted(exigidas - propriedades)}"

    # E o que cada CLASSE traz. As três contagens de ausência viajam juntas porque coincidem nestes
    # arquivos **por acidente**, e é a coexistência delas que torna o rótulo conferível na imagem.
    por_classe = {"grupo", "n_injecoes", "n_celulas", "n_zeros", "n_nao_finitas", "n_faltantes"}
    assert por_classe <= da_classe, f"faltam a cada classe: {sorted(por_classe - da_classe)}"
    assert {"n_detectadas", "log10_mediana_das_detectadas"} <= da_classe, (
        "sem a contagem de detectadas ao lado da mediana, a abundância viaja sem a população em "
        "que foi medida"
    )


def test_u8_a_raiz_da_composicao_e_declarada_pela_rota_e_lida_pela_janela(
    contratos: dict[tuple[str, str], Any],
) -> None:
    """A segunda pré-condição: a carga atravessa **e** alguém do outro lado a apanha.

    **Remover `PainelDeGraficos.recebe_composicao` faz este teste acusar**, que é o par causal da
    unidade — e ele mede a direção que `test_toda_chave_lida_de_corpo_e_declarada_por_alguma_rota`
    não mede: aquele parte das leituras e pergunta se a rota as declara; este parte da carga nova e
    pergunta se a janela a lê.
    """
    declaradas = _chaves_de_resposta(contratos)
    assert len(declaradas) >= 20, (
        f"a união das respostas tem {len(declaradas)} chaves; com poucas, a asserção abaixo seria "
        "quase vácua"
    )
    assert RAIZ_DA_COMPOSICAO in declaradas, (
        f"rota nenhuma declara {RAIZ_DA_COMPOSICAO!r} na resposta: a composição não atravessa o "
        "fio, e a figura da U8 não teria de onde sair"
    )

    # **E ela é OBRIGATÓRIA**, pelo argumento das três irmãs: a composição não depende do par de
    # grupos nem de escolha nenhuma — toda execução tem uma matriz ingerida —, então não há aqui o
    # *"não calculado"*, distinto de *"calculado e vazio"*. A lista **vazia** é o
    # segundo, e é o que uma matriz só de QCs produz.
    esquemas = dict(build_app().openapi()).get("components", {}).get("schemas", {})
    obrigatorios = esquemas.get("ExecucaoResposta", {}).get("required", [])
    assert obrigatorios, "`ExecucaoResposta` não declara campo obrigatório — a asserção seria vácua"
    assert RAIZ_DA_COMPOSICAO in obrigatorios, (
        f"{RAIZ_DA_COMPOSICAO!r} é opcional no esquema, e o servidor passa a poder omiti-la: a "
        "ausência volta a ser um estado que o painel tem de representar, sem haver o fato «não "
        "calculado» que a justificasse"
    )

    leituras = _leituras_da_janela()
    assert leituras, "a extração não achou leitura nenhuma — a asserção abaixo seria vácua"
    lida_por = sorted(
        (arquivo, linha)
        for arquivo, linha, base, chave in leituras
        if base in CORPO_DE_RESPOSTA and chave == RAIZ_DA_COMPOSICAO
    )
    assert lida_por, (
        f"a janela não lê {RAIZ_DA_COMPOSICAO!r} do corpo em módulo nenhum de `desktop/janela/`. "
        "A raiz atravessa o fio e ninguém a apanha: quem a consome é "
        "`PainelDeGraficos.recebe_composicao`"
    )


# ---------------------------------------------------------------------------
# O corpo da MEDIÇÃO, e por que ele não é raiz aninhada
# ---------------------------------------------------------------------------
#
# **A medição não entra em `RAIZES_ANINHADAS`, e a ausência é estrutural e não dívida.** As nove
# entradas daquela tabela são variáveis que vêm de um **campo** do corpo — `corpo["distribuicoes"]`,
# `corpo["pca"]` —, e o esquema delas é o do campo. `MedicaoDoCorretorResposta` **é** o corpo de
# `POST /matrizes/medir-corretor`: as chaves saem de `corpo` direto, e a classe delas é
# `CORPO_DE_RESPOSTA`, que já existe. Declarar uma raiz aqui reprovaria por classe sem uso.
#
# > **AS TRÊS FRASES ACIMA CADUCARAM, e ficam com a emenda ao lado.** Elas foram escritas quando
# > o único leitor era `recebe_medicao(corpo)`; depois `PainelDeGraficos._painel_da_correcao`
# > passou a ler o corpo **guardado**, e ali a base é uma variável local que a partição
# > fail-closed obriga a classificar. A premissa *"as chaves saem de `corpo`
# > direto"* deixou de valer, e com ela caiu *"reprovaria por classe sem uso"* — agora há uso.
# > A entrada `"medicao": "MedicaoDoCorretorResposta"` está em `RAIZES_ANINHADAS`, com o argumento
# > de que bijetar contra a rota de origem é mais estreito do que bijetar contra a união de todas
# > as respostas, que é o que `corpo` daria. **O que NÃO mudou é a leitura de `recebe_medicao`**:
# > lá a base continua a ser `corpo`, e continua a ser a classe certa.
NOME_DO_ESQUEMA_DA_MEDICAO: Final = "MedicaoDoCorretorResposta"


def test_u4_o_corpo_da_medicao_e_declarado_pela_rota_e_lido_pelo_painel_dos_graficos(
    contratos: dict[tuple[str, str], Any],
) -> None:
    """A pré-condição da fila de barras: o que ela pede existe no corpo **e** é lido.

    **Remover `PainelDeGraficos.recebe_medicao` faz este teste acusar**, que é o par causal: a
    carga atravessaria o fio e nenhum painel a apanharia, e a figura nasceria a mostrar o texto de
    ausência para sempre.

    ## A população por barra, que o corpo NÃO tem — a decisão, com a medição ao lado

    O pedido é a população **por barra**. Ela é calculada e **descartada** em
    `decision/labeling.py:213`: `np.nanmedian(valores)` reduz o vetor por feature a um número e o
    resto do vetor morre ali. Medido em 17/09/2026 no `pos_hilic` real (68 x 6035), com as **mesmas
    200 features amostradas** nos dois corretores (mesma semente, logo a diferença é atribuível ao
    corretor):

    | corretor | features com RSD finito | nanmediana |
    |---|---|---|
    | `none` | **193** de 200 | 1,521903 |
    | `loess` | **172** de 200 | 1,553443 |

    **As populações diferem entre barras — 21 features, 10,9% da amostra.** Então rotular as três
    barras com um número só seria o número certo sobre a população errada. E o único número que o
    corpo traz, `n_features_medidas`, **não é** nenhuma das duas: ele é `n_sampled_features` tal
    como o pedido o mandou (`rotas.py`), ou seja o **teto pedido ao amostrador** e não a população
    em que a mediana correu — e a janela nunca o manda, então ele chega `None`.

    > **Servi-la exigiria um campo novo em `Label`, que é contrato de `decision/labeling.py`.** A
    > saída escolhida é a outra que o plano autoriza: **a barra declara «população não servida»**, e
    > este teste fixa a ausência para que a figura não invente um número a partir de
    > `n_features_medidas`.

    **As asserções de não-vacuidade vêm primeiro e na mesma função**: `frozenset()` é
    subconjunto de tudo, e a bijeção passaria sobre um esquema inexistente.
    """
    comp = dict(build_app().openapi()).get("components", {}).get("schemas", {})
    assert comp, "o esquema não tem componentes — as asserções abaixo seriam vácuas"

    propriedades = frozenset(comp.get(NOME_DO_ESQUEMA_DA_MEDICAO, {}).get("properties", {}))
    assert propriedades, (
        f"{NOME_DO_ESQUEMA_DA_MEDICAO!r} não está em `components.schemas`, ou não tem "
        f"propriedades. O esquema tem {sorted(comp)}"
    )

    # O que a fila de barras consome: os scores, os dois rótulos que os tornam comparáveis e as
    # duas metades da política de eleição.
    exigidas = {"scores", "vencedor", "estimador", "variante", "avaliavel", "abstem"}
    assert exigidas <= propriedades, (
        f"faltam ao corpo da medição: {sorted(exigidas - propriedades)}"
    )

    # **A ausência da população por barra, fixada.** Um campo por corretor apareceria como uma
    # propriedade cujo nome fala de população; enquanto não existir, a barra declara «população
    # não servida».
    por_barra = sorted(p for p in propriedades if "features" in p and p != "n_features_medidas")
    assert not por_barra, (
        f"o corpo passou a trazer {por_barra}: se for a população POR CORRETOR, R6 deixa de "
        "precisar da frase «população não servida» e a U3 tem de a substituir pelo número"
    )

    declaradas = _chaves_de_resposta(contratos)
    assert len(declaradas) >= 20, (
        f"a união das respostas tem {len(declaradas)} chaves; com poucas, a asserção abaixo seria "
        "quase vácua"
    )
    assert exigidas <= declaradas, (
        f"rota nenhuma declara {sorted(exigidas - declaradas)} na resposta — o corpo da medição "
        "não atravessa o fio, e a fila de barras não teria de onde sair"
    )

    leituras = _leituras_da_janela()
    assert leituras, "a extração não achou leitura nenhuma — a asserção abaixo seria vácua"
    lidas_pelos_graficos = {
        chave
        for arquivo, _linha, base, chave in leituras
        if arquivo == "graficos.py" and base in CORPO_DE_RESPOSTA
    }
    assert lidas_pelos_graficos, (
        "`graficos.py` não lê chave nenhuma de `corpo` — a asserção abaixo seria vácua"
    )
    assert {"scores", "avaliavel"} <= lidas_pelos_graficos, (
        "o painel dos gráficos não lê o corpo da medição: a carga atravessa o fio e ninguém a "
        "apanha. Quem a consome é `PainelDeGraficos.recebe_medicao`, e o que ela lê hoje é "
        f"{sorted(lidas_pelos_graficos)}"
    )


def test_u11_a_chave_de_invalidacao_e_ESCALAR_na_raiz_e_nao_uma_raiz_nova() -> None:  # noqa: N802
    """A chave é campo de `ExecucaoResposta` e é uma `str` — o consumo de `artefatos.py`.

    **Se ela virasse objeto, o padrão de consumo quebrava em silêncio.**
    `PainelDeArtefatos.define_entrada` compara `str` com `==`; dois `dict` iguais também comparam
    iguais, então o painel não daria erro — passaria a exigir que cada consumidor compusesse a
    comparação por conta própria, com a construção duplicada em cada um.

    **E ela é OBRIGATÓRIA no esquema**, que é a metade que o `required` do OpenAPI fixa: um campo
    opcional deixaria o servidor livre para a omitir, e a ausência voltaria a ter de ser
    representada — que é o problema que `interface.cliente.chave_de_invalidacao` existe para
    resolver **do lado do cliente**, sobre respostas que não são `200`.
    """
    comp = dict(build_app().openapi()).get("components", {}).get("schemas", {})
    execucao = comp.get("ExecucaoResposta", {})
    propriedades = execucao.get("properties", {})
    assert propriedades, "`ExecucaoResposta` não tem propriedades — a asserção seria vácua"

    campo = propriedades.get("chave_de_invalidacao")
    assert campo is not None, (
        f"`ExecucaoResposta` não declara `chave_de_invalidacao`. Tem {sorted(propriedades)}"
    )
    assert campo.get("type") == "string", (
        f"a chave deixou de ser escalar: {campo}. `define_entrada` recebe `str` e compara com `==`"
    )
    assert "chave_de_invalidacao" in execucao.get("required", []), (
        "a chave virou opcional no esquema — o servidor passa a poder omiti-la, e a ausência "
        "volta a ser um estado que o corpo tem de representar (KTD13)"
    )
    assert "distribuicoes" in execucao.get("required", []), (
        "o resumo das distribuições virou opcional — distribuição não depende do par de grupos, "
        "então não há aqui o «não calculado» a distinguir"
    )


# ---------------------------------------------------------------------------
# As DUAS raízes da correlação, e a armadilha de as apontar ao mesmo esquema
# ---------------------------------------------------------------------------


def test_u12_as_duas_raizes_da_correlacao_sao_esquemas_DISTINTOS_no_openapi() -> None:  # noqa: N802
    """**A pré-condição das duas entradas de `RAIZES_ANINHADAS`.**

    **A armadilha simétrica é apontar as duas para o mesmo esquema.** Os dois nomes começam por
    «correlação», os dois estão no OpenAPI, e os dois têm `metodo`, `matriz` e os dois
    denominadores. Um consumidor que bijetasse contra o irmão errado procuraria `valores` na busca,
    não o encontraria, e ficaria no estado vazio para sempre — que é o mesmo defeito medido entre
    a composição e a classe dela.

    **O que os separa é o que cada um É:** `ResumoDaCorrelacao` descreve uma matriz `k x k` sobre um
    recorte sorteado; `ResumoDaBuscaPorPadrao` descreve um vetor por feature sobre **todas** elas,
    na ordem que a busca produziu. A asserção abaixo é sobre os campos que **só** um deles tem.

    **As asserções de não-vacuidade vêm primeiro**: `frozenset()` é subconjunto de tudo, e a
    bijeção passaria sobre um esquema inexistente.
    """
    comp = dict(build_app().openapi()).get("components", {}).get("schemas", {})
    assert comp, "o esquema não tem componentes — as asserções abaixo seriam vácuas"

    do_mapa = frozenset(comp.get("ResumoDaCorrelacao", {}).get("properties", {}))
    da_busca = frozenset(comp.get("ResumoDaBuscaPorPadrao", {}).get("properties", {}))
    for nome, propriedades in (
        ("ResumoDaCorrelacao", do_mapa),
        ("ResumoDaBuscaPorPadrao", da_busca),
    ):
        assert propriedades, (
            f"{nome!r} não está em `components.schemas`, ou não tem propriedades. Um `BaseModel` "
            "que rota nenhuma alcança não entra no OpenAPI, e a entrada de `RAIZES_ANINHADAS` "
            f"reprovaria por «o esquema não existe». O esquema tem {sorted(comp)}"
        )

    assert do_mapa != da_busca, (
        "os dois esquemas têm o MESMO conjunto de propriedades: uma entrada apontada para o irmão "
        "errado passaria, e a bijeção deixaria de distinguir uma matriz de um vetor"
    )
    assert {"valores", "amostragem"} <= do_mapa - da_busca, (
        f"o mapa deixou de ter campo exclusivo: {sorted(do_mapa - da_busca)}. `valores` é a matriz "
        "e `amostragem` é a frase do sorteio — nenhum dos dois faz sentido na busca, que corre "
        "sobre todas as features"
    )
    assert {"coeficiente", "referencia", "ordem_dos_grupos"} <= da_busca - do_mapa, (
        f"a busca deixou de ter campo exclusivo: {sorted(da_busca - do_mapa)}. "
        "`ordem_dos_grupos` é a resposta a OQ2, e sem ela um perfil «0,1,2,3» não significa nada"
    )

    # **E as duas são OPCIONAIS no esquema, ao contrário de `distribuicoes`.** A família pode não
    # ter sido pedida, e a busca pode não ter referência: as duas ausências são *"não calculado"*
    # e têm de continuar a ser representáveis.
    execucao = comp.get("ExecucaoResposta", {})
    obrigatorios = execucao.get("required", [])
    assert obrigatorios, "`ExecucaoResposta` não declara campo obrigatório — a asserção seria vácua"
    assert "correlacao" not in obrigatorios and "busca_por_padrao" not in obrigatorios, (
        "uma das raízes da correlação virou obrigatória: a ausência deixa de ser representável, e "
        "«a família não foi pedida» passa a precisar de um objeto vazio que se lê como «correu e "
        "não achou nada»"
    )


def test_u12_as_duas_raizes_sao_declaradas_pela_rota_e_LIDAS_pela_janela(  # noqa: N802
    contratos: dict[tuple[str, str], Any],
) -> None:
    """A segunda pré-condição: a carga atravessa **e** alguém do outro lado a apanha.

    **Remover `PainelDeGraficos.recebe_correlacao` faz este teste acusar**, que é o par causal da
    unidade — e ele mede a direção que `test_toda_chave_lida_de_corpo_e_declarada_por_alguma_rota`
    não mede: aquele parte das leituras e pergunta se a rota as declara; este parte da carga nova e
    pergunta se a janela a lê.
    """
    declaradas = _chaves_de_resposta(contratos)
    assert len(declaradas) >= 20, (
        f"a união das respostas tem {len(declaradas)} chaves; com poucas, a asserção abaixo seria "
        "quase vácua"
    )
    leituras = _leituras_da_janela()
    assert leituras, "a extração não achou leitura nenhuma — a asserção abaixo seria vácua"

    for raiz in ("correlacao", "busca_por_padrao"):
        assert raiz in declaradas, (
            f"rota nenhuma declara {raiz!r} na resposta: a carga não atravessa o fio, e a figura "
            "da U12 não teria de onde sair"
        )
        lida_por = sorted(
            (arquivo, linha)
            for arquivo, linha, base, chave in leituras
            if base in CORPO_DE_RESPOSTA and chave == raiz
        )
        assert lida_por, (
            f"a janela não lê {raiz!r} do corpo em módulo nenhum de `desktop/janela/`. A raiz "
            "atravessa o fio e ninguém a apanha: quem a consome é "
            "`PainelDeGraficos.recebe_correlacao`"
        )


# ---------------------------------------------------------------------------
# O grupo de cada amostra na PCA do diagnóstico
# ---------------------------------------------------------------------------


def test_o_grupo_por_amostra_da_pca_esta_no_esquema_e_a_JANELA_o_le() -> None:  # noqa: N802
    """A chave nova atravessa o fio, é obrigatória, e a janela a lê da raiz `pca`.

    **A direção que os portões genéricos não medem é a segunda.** `test_as_chaves_aninhadas_...`
    parte das leituras e pergunta se o esquema as declara; este parte da chave nova e pergunta se
    alguém a lê. E a falta dela não levanta nada: sem a leitura, `escores_da_pca` recebe `None` e a
    figura cai, bem formada, na forma por tipo — a PCA por grupo sumiria em silêncio.
    """
    comp = dict(build_app().openapi()).get("components", {}).get("schemas", {})
    esquema = comp.get("ResumoDaPca", {})
    propriedades = frozenset(esquema.get("properties", {}))
    assert propriedades, "`ResumoDaPca` não está no esquema, e as asserções abaixo seriam vácuas"
    assert "grupos_por_amostra" in propriedades, sorted(propriedades)
    assert "grupos_por_amostra" in esquema.get("required", []), (
        "o grupo por amostra ficou opcional, e uma rota que o esquecesse serviria a figura por "
        "tipo sem nada a acusar"
    )

    leituras = _leituras_da_janela()
    assert leituras, "a extração não achou leitura nenhuma — a asserção abaixo seria vácua"
    lida_por = sorted(
        (arquivo, linha)
        for arquivo, linha, base, chave in leituras
        if base == "pca" and chave == "grupos_por_amostra"
    )
    assert lida_por, (
        "a janela não lê `grupos_por_amostra` da raiz `pca`: o grupo atravessa o fio e a figura "
        "continua por tipo — quem o consome é `PainelDeGraficos._escores_da_pca`"
    )
