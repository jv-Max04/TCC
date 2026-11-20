"""As rotas da API.

As rotas são finas e não decidem nada: traduzem HTTP para chamada do núcleo e de volta. `api/` é
fronteira como `io/`, e o núcleo não sabe que existe HTTP.

Toda rota está declarada em `docs/rotas-api.md`, e `tests/test_api_contrato.py` confere a
bijeção nas duas direções: rota nova sem linha no manifesto reprova.

O upload grava em diretório temporário e o apaga. Os arquivos são clínicos e identificáveis, e
uma cópia num diretório de trabalho persistente ficaria fora do lugar onde a procedência deles
está declarada.

As quatro rotas que correm corretor são assíncronas: `/execucoes`, `/matrizes/corrigir`,
`/matrizes/processar` e `/matrizes/exportar` devolvem `202` com um identificador, e o resultado
é reclamado em `GET /trabalhos/{identificador}`. O mecanismo está descrito antes de `/execucoes`.
"""

import hashlib
import json
import math
import re
import sys
import tempfile
import threading
import warnings
from collections import Counter, OrderedDict
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, fields, replace
from datetime import UTC, datetime
from pathlib import Path, PureWindowsPath
from typing import Annotated, Any, Final, NamedTuple, cast

import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, Response, UploadFile, status
from numpy.random import default_rng
from pydantic import BaseModel
from sqlalchemy import Engine, create_engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.pool import NullPool

from refino.api.esquemas import (
    COEFICIENTE_DE_TUKEY,
    DEFINICAO_DE_ABUNDANCIA,
    DEFINICAO_DE_FALTANTE,
    TETO_DE_CAIXAS,
    ArestaDaRede,
    CaixaDaDistribuicao,
    CaixaPorAmostra,
    ClasseDaComposicao,
    ComposicaoPorClasse,
    ConclusaoDoHistorico,
    CurvaDoLoessResposta,
    DelegacaoResposta,
    DiagnosticoResposta,
    EstabilidadeDeFeature,
    EtapaDaCorrecaoResposta,
    EtapaDoPreprocessamentoResposta,
    ExecucaoResposta,
    FeatureRemovidaResposta,
    GrupoComContagem,
    HistoricoDeMultiversos,
    IntervaloDaBarra,
    ItemComRessalva,
    LimiaresDoVeredito,
    MedicaoDoCorretorResposta,
    MultiversoAceito,
    MultiversoNoHistorico,
    NormalidadeDoGrupo,
    PalpiteResposta,
    ParDePcas,
    ProblemaResposta,
    ProgressoResposta,
    RecomendacaoResposta,
    RelatorioDeExportacao,
    ResultadoDoMultiverso,
    ResumoDaAnova,
    ResumoDaBuscaPorPadrao,
    ResumoDaCorrelacao,
    ResumoDaNormalidade,
    ResumoDaPca,
    ResumoDaPermanova,
    ResumoDaRede,
    ResumoDasDistribuicoes,
    ResumoDoClassificador,
    ResumoDoKmeans,
    ResumoDoPlsda,
    ResumoDoSom,
    ResumoDoVolcano,
    RsdAntesEDepois,
    RsdPorFeature,
    SegundaOpiniao,
    SugestaoDeLotes,
    TrabalhoAceito,
    TrajetoriaDosQcs,
    ValidacaoResposta,
)
from refino.api.progresso import ProgressoDaExecucao, RegistroDeProgresso, executa_registrando
from refino.api.reuso import MemoriaDeCorrecoes, chave_da_correcao
from refino.cli import (
    AUTO_APRENDIDA,
    AUTO_CAVEATS,
    AUTO_MEDIDA,
    AUTO_VALIDOS,
    AVAILABLE_METHODS,
    UNAVAILABLE_METHODS,
    recomenda,
)
from refino.correction.base import Corrector
from refino.correction.loess import LoessCorrector, curve_sensitivity
from refino.correction.serrf import IntensidadeNegativaError
from refino.correction.serrf_na_vez import SerrfNaVez
from refino.decision.artefato import ArtefatoInvalidoError
from refino.decision.features import perfil_para_a_politica
from refino.decision.labeling import LabelingOracle, com_lotes
from refino.decision.palpite import (
    LIMIAR_DE_EMPATE_DO_PALPITE,
    N_FEATURES_DO_PALPITE,
    PalpiteMedido,
    mede_o_palpite,
)
from refino.decision.policy import ABSTENTION_MARGIN

# `api` importa de `desktop` de propósito: as duas são fronteiras, e o núcleo continua sem saber
# delas. A rota precisa saber se este executável consegue lançar workers antes de construir o
# corretor, e só a camada do congelado sabe responder. `desktop/paralelismo.py` só importa `sys`
# e `pathlib`, então o custo para a CLI é um import da biblioteca padrão.
from refino.desktop.paralelismo import SERIE, n_jobs_efetivo, regime_do_processo
from refino.diagnosis.batches import batch_without_training, propose_supported_batches
from refino.diagnosis.policy import DiagnosisPolicy, InvalidPolicyError
from refino.diagnosis.profile import PRETRANSFORM, diagnose
from refino.domain.analysis import (
    PADRAO_DA_ANALISE,
    PARAMETRIC_BY_UNIVARIATE_TEST,
    AnalysisSpec,
    analise_para_json,
)
from refino.domain.cancellation import (
    CancelCheck,
    ExecutionCancelledError,
    raise_if_cancelled,
)
from refino.domain.correction import CorrectionResult
from refino.domain.labels import (
    ClassifierKind,
    Correction,
    InvalidLabelError,
    PreTransform,
    RsdEstimator,
    RsdVariant,
    validate_label,
)
from refino.domain.lotes import confere_cortes, lotes_declarados
from refino.domain.matrix import RefinoMatrix
from refino.domain.metadata import SampleType
from refino.domain.spec import PipelineSpec, QualityReport, validate_override_fields
from refino.domain.statistics import AnovaResult, UnivariateResult

# A API importa de `interface/fluxo`, que é a declaração pura do fluxo e só importa de `domain` e
# de `multiverse`. A tabela de famílias vive lá porque é a mesma que a janela desenha e que a
# barra de progresso anuncia; duas cópias divergiriam no dia em que entrasse uma família nova.
from refino.interface.fluxo import etapas_das_analises, familias_pedidas
from refino.io.fatores import FatoresMalformadosError, le_fatores_de_bytes
from refino.io.issues import ValidationIssue
from refino.io.politica import (
    carrega_politica_aprendida,
    carrega_politica_pela_verdade,
    segunda_opiniao_ativa,
)
from refino.io.readers import IngestionError, load_matrix_with_issues
from refino.io.validators import check_mode_matches_content
from refino.io.writers import (
    MetaboAnalystExport,
    write_metaboanalyst,
    write_normalizado,
    write_toserrf,
)
from refino.metrics.crossval import (
    JACKKNIFE_METHOD,
    MIN_TRAIN_QC_PER_BATCH,
    DegenerateCrossValidationError,
    JackknifeInterval,
)
from refino.metrics.report import build_quality_report
from refino.metrics.rsd import qc_rsd, rsd
from refino.metrics.structure import (
    EstruturaSemPopulacaoError,
    StructuralPCA,
    column_usability,
    structural_metrics,
    structural_pca,
)
from refino.multiverse.grid import GridDefinition
from refino.multiverse.stability import StabilityThresholds, classifica_veredito
from refino.persistence.config import (
    VARIAVEL_DA_URL,
    ConfiguracaoDeBancoError,
    prepara_sqlite,
    url_do_banco,
)
from refino.persistence.repositories import (
    EstabilidadeParaGravar,
    MultiversoGravado,
    MultiversoRepository,
    RunRepository,
    SobrescritaExercida,
)
from refino.persistence.schema import validate_apresentacao
from refino.preprocessing.missing import is_missing
from refino.preprocessing.normalization import NormalizacaoInvalidaError
from refino.preprocessing.pipeline import (
    EmptyMatrixError,
    ProcessedMatrix,
    process_matrix,
    process_matrix_detailed,
    requests_preprocessing,
)
from refino.statistics.anova import MIN_GRUPOS, one_way_anova
from refino.statistics.classificadores import (
    NOME_DO_CLASSIFICADOR,
    ClassificacaoImpossivelError,
    ClassifierResult,
    classify,
)
from refino.statistics.clustering import (
    TETO_DE_FEATURES_PARA_DESENHAR,
    KMeansResult,
    ParticaoImpossivelError,
    SomResult,
    kmeans_clustering,
    self_organizing_map,
)
from refino.statistics.correlacao import (
    MIN_AMOSTRAS_PARA_CORRELACAO,
    CorrelationResult,
    PatternSearchResult,
    ReferenciaDesconhecidaError,
    feature_correlation,
    pattern_search,
)
from refino.statistics.dspc import (
    MIN_AMOSTRAS_PARA_DSPC,
    DspcResult,
    RedeAcimaDoTetoError,
    RedeSemPopulacaoError,
    dspc_network,
)
from refino.statistics.normalidade import NormalityResult, normality_by_group
from refino.statistics.permanova import (
    PermanovaResult,
    PermanovaSemPopulacaoError,
    permanova,
)
from refino.statistics.plsda import PLSDAResult, plsda
from refino.statistics.univariate import (
    MIN_AMOSTRAS_POR_GRUPO,
    fold_change_interval,
    univariate_test,
)

__all__ = ["ResultadosDasAnalises", "corre_as_analises", "router"]

router = APIRouter()

# O `--auto` precisa de um `DefectProfile`, e `diagnose` exige uma `DiagnosisPolicy`, que não tem
# padrão embutido. Os limiares iniciais vivem em `diagnosis/policy.py` porque `cli.py` também os
# usa e não pode importá-los daqui: `api/esquemas.py` já importa de `cli.py`, e seria ciclo. As
# quatro superfícies do `--auto` partem do mesmo ponto, senão a recomendação mudaria conforme a
# porta de entrada.
from refino.diagnosis.policy import (  # noqa: E402
    LIMIARES_INICIAIS,
    # Reexportado e não usado aqui: `tests/integration/test_gerador_sintetico.py` o importa deste
    # módulo.
    RSD_MEDIANA_MAX_POR_MODO,  # noqa: F401
    mapa_calibrado_para,
)

# O modo é rótulo aberto: `lipidomica_neg`, `RP_POS` e `hilic2` passam, porque um estudo cujos
# modos não são os quatro deste laboratório tem de entrar. O que se recusa são os caracteres que
# fazem o rótulo virar caminho. As rotas de artefato montam `Path(temporario) / f"{mode}_*.csv"`,
# e um operando absoluto à direita de `/` substitui o diretório inteiro: com `/`, `..` ou caminho
# absoluto no modo, a matriz clínica seria escrita fora do temporário, onde o cliente mandasse.
# 32 é limite de nome de arquivo, não de domínio: os modos do laboratório têm de 7 a 9 caracteres.
MODO_SEGURO: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z0-9_-]{1,32}$")


def _modo_ou_422(mode: str) -> None:
    """Recusa com 422 um `mode` que não serve para nome de arquivo.

    O domínio do modo é aberto, mas o valor vira componente de caminho, então a guarda fica na
    fronteira. Fica em `_carrega` porque toda rota que ingere passa por ali. Vale também
    para as que não escrevem artefato: o modo entra em `_chave_do_trabalho` e no `file_name` do
    download, e recusar num lugar só impede as fronteiras de divergirem.

    A mensagem nomeia a regra, e o valor recusado vai truncado para não ecoar entrada grande.
    """
    if MODO_SEGURO.match(mode) is None:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            f"`mode={mode[:64]!r}` não é um rótulo aceitável: esperava de 1 a 32 caracteres de "
            "`[A-Za-z0-9_-]`. O modo NÃO é enumeração fechada — um estudo novo declara o nome que "
            "quiser —, e o conjunto é restrito porque ele entra no nome do arquivo do artefato: "
            "`/` e `..` fariam a escrita sair do diretório temporário do serviço.",
        )


def _nome_de_exibicao(enviado: str | None, padrao: str) -> str:
    """O nome que o cliente mandou, como texto para a procedência, nunca como caminho.

    `PureWindowsPath` corta nos dois separadores, `/` e `\\`, então `../x.csv`, `/tmp/x.csv` e
    `C:\\dados\\x.csv` dão todos `x.csv`. Caractere de controle sai, porque a procedência vai
    para relatório e para a tela. Sem nome útil (ausente, vazio, `.`, `..`), fica o nome fixo do
    disco.
    """
    if not enviado:
        return padrao
    nome = "".join(c for c in PureWindowsPath(enviado).name if c.isprintable())
    return padrao if nome in {"", ".", ".."} else nome


def _carrega(
    measurements: UploadFile,
    sample_list: UploadFile,
    mode: str,
    orientation: str,
    conteudo_m: bytes | None = None,
    avisos: list[ValidationIssue] | None = None,
) -> RefinoMatrix:
    """Escreve o upload em temporário, ingere, e apaga o temporário.

    Os nomes no disco são fixos. O `filename` do multipart vem do cliente: no caminho, `../x.csv`
    ou um caminho absoluto gravariam fora do temporário, e dois arquivos com o mesmo nome (vindos
    de pastas diferentes) fariam o Sample_List sobrescrever o measurements. O nome do usuário
    entra só na procedência, como texto, por `_nome_de_exibicao`.

    `avisos`, quando passado, recebe os problemas que não recusam a matriz; é o que
    `/matrizes/validar` precisa para devolver a lista completa.

    `orientation` e `mode` vêm do corpo da requisição e são validados aqui, antes de qualquer
    uso. Uma orientação inválida cairia no ramo da outra e transporia a matriz ao contrário sem
    levantar.

    `conteudo_m` existe porque um `UploadFile` só se lê uma vez. A rota de execução precisa dos
    bytes para o `checksum_entrada`, e uma segunda leitura devolveria vazio: a matriz seria de um
    CSV sem linhas, e a recusa apareceria como erro de validação.
    """
    _modo_ou_422(mode)
    try:
        validate_label("Orientation", orientation)
    except InvalidLabelError as erro:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(erro)) from erro

    with tempfile.TemporaryDirectory(prefix="refino-api-") as temporario:
        raiz = Path(temporario)
        caminho_m = raiz / "measurements.csv"
        caminho_s = raiz / "Sample_List.csv"
        caminho_m.write_bytes(measurements.file.read() if conteudo_m is None else conteudo_m)
        caminho_s.write_bytes(sample_list.file.read())
        try:
            matriz, nao_recusam = load_matrix_with_issues(
                caminho_m,
                caminho_s,
                mode,
                orientation=orientation,  # type: ignore[arg-type]
                rotulo=(
                    f"{_nome_de_exibicao(measurements.filename, caminho_m.name)} + "
                    f"{_nome_de_exibicao(sample_list.filename, caminho_s.name)}"
                ),
            )
            if avisos is not None:
                avisos.extend(nao_recusam)
            return matriz
        except IngestionError:
            # A recusa de validação tem tradutor próprio: `/matrizes/validar` lista os problemas e
            # `_ingere_ou_422` devolve 422 apontando para ela.
            raise
        except ValueError as erro:
            # `IngestionError` herda de `ValueError`, então a ordem dos `except` é o mecanismo: a
            # filha passa, a mãe vira 422. CSV trocado no seletor ou linha mais curta que o
            # cabeçalho é erro do arquivo; com 500, a janela diria que o defeito é da ferramenta.
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_CONTENT,
                f"o arquivo não pôde ser lido: {erro}",
            ) from erro


# O nome de estudo com que uma execução sem identificação declarada é gravada.
#
# As rotas de artefato não têm campo `estudo` e gravam sob este nome, que é também o padrão do
# campo em `POST /multiversos`. As execuções sem estudo ficam juntas, e as matrizes continuam
# separadas por `checksum_entrada`, como `uq_matrices_origem` manda.
ESTUDO_SEM_IDENTIFICACAO: Final = "sem_identificacao"


def checksum_do_par(medidas: bytes, lista: bytes) -> str:
    """O `checksum_entrada` de uma matriz, calculado sobre os dois arquivos.

    `uq_matrices_origem` é `(study_id, mode, checksum_entrada)`, e a matriz é reaproveitada
    quando ele bate. Se o checksum cobrisse só o `measurements`, dois disparos com o mesmo
    `measurements` e `Sample_List` diferentes reaproveitariam a matriz, e as amostras, a
    `injection_order` e o `batch` do primeiro ficariam ligados às execuções do segundo.

    O separador impede que mover bytes de um arquivo para o outro produza o mesmo fluxo.
    `b"\x00"` não aparece em CSV de texto.
    """
    return hashlib.sha256(medidas + b"\x00" + lista).hexdigest()


def _amostras_para_gravar(matriz: RefinoMatrix) -> list[dict[str, Any]]:
    """As amostras no formato que `_garante_estudo_e_matriz` insere em `samples`.

    Fonte única para as rotas que gravam matriz. As chaves são as colunas de `samples`, e uma
    cópia que esquecesse `batch` ou `injection_order` gravaria amostras sem o único registro da
    ordem temporal, com o `INSERT` a passar e a correção de deriva irreconstruível.
    """
    return [
        {
            "name": a.name,
            "injection_order": a.injection_order,
            "sample_type": a.sample_type,
            "grupo": a.group,
            "batch": a.batch,
        }
        for a in matriz.samples
    ]


def _ingere_ou_422(
    measurements: UploadFile,
    sample_list: UploadFile,
    mode: str,
    orientation: str,
    conteudo_m: bytes | None = None,
) -> RefinoMatrix:
    """`_carrega` com a recusa de validação traduzida em 422 e a lista apontada.

    Aqui a rejeição é erro de protocolo, ao contrário de `/matrizes/validar`: quem pede
    recomendação ou execução sobre matriz recusada não tem resposta possível.
    """
    try:
        return _carrega(measurements, sample_list, mode, orientation, conteudo_m)
    except IngestionError as erro:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            f"a matriz foi recusada pela validação: {len(erro.issues)} problema(s). "
            "Use /matrizes/validar para a lista completa",
        ) from erro


@router.post("/matrizes/validar", response_model=ValidacaoResposta)
def validar(
    measurements: Annotated[UploadFile, File()],
    sample_list: Annotated[UploadFile, File()],
    mode: Annotated[str, Form()],
    orientation: Annotated[str, Form()] = "features-in-rows",
) -> ValidacaoResposta:
    """Ingere a matriz e devolve a lista completa de problemas de validação.

    Recebe os dois CSV, `mode` e `orientation`. Responde 200 também quando a matriz é recusada,
    com `aceita=False` e os problemas enumerados: arquivo com defeito é resposta legítima, e um
    4xx esconderia a lista no corpo de um erro. Aceita, a resposta traz a forma, os nomes de
    amostras e features, a ordem de injeção, a marca de QC e os avisos que não recusam.
    422 para `mode` ou `orientation` inválidos, ou para arquivo ilegível.
    """
    avisos: list[ValidationIssue] = []
    try:
        matriz = _carrega(measurements, sample_list, mode, orientation, avisos=avisos)
    except IngestionError as erro:
        return ValidacaoResposta(
            aceita=False,
            problemas=[
                ProblemaResposta(
                    codigo=problema.code,
                    severidade=problema.severity,
                    mensagem=problema.message,
                )
                for problema in erro.issues
            ],
        )
    # O aviso de modo divergente entra aqui, e não na ingestão: esta é a rota que enumera
    # problemas, no primeiro passo do fluxo, onde ainda dá para corrigir o seletor.
    return ValidacaoResposta(
        aceita=True,
        n_samples=matriz.n_samples,
        n_features=matriz.n_features,
        # Os nomes que as normalizações contra referência pedem. Só no caminho aceito: matriz
        # recusada não tem nomes a escolher.
        sample_names=list(matriz.sample_names),
        feature_names=list(matriz.feature_names),
        # A ordem de injeção e a marca de QC de cada amostra, para a tela contar os QCs de cada
        # lote declarado.
        ordens_de_injecao=[int(v) for v in matriz.injection_orders],
        qc=[bool(v) for v in matriz.qc_mask],
        problemas=[
            ProblemaResposta(
                codigo=problema.code, severidade=problema.severity, mensagem=problema.message
            )
            # Os avisos da ingestão vêm antes do aviso de modo, na mesma lista. Sem eles, uma
            # matriz aceita com rótulos grafados de outro jeito apareceria como 'nenhum problema'.
            for problema in (*avisos, *check_mode_matches_content(mode, matriz.sample_names))
        ],
    )


# Quantos componentes a PCA do diagnóstico produz. Dois é o padrão de `structural_metrics` e o
# que `diagnose` corre; mudá-lo moveria `qc_dispersion_ratio`, a distância média ao centroide
# sobre todos os componentes. A PCA e as métricas têm de declarar o mesmo número, e o guarda de
# `structural_metrics` reprova se divergirem.
COMPONENTES_DA_PCA: Final = 2


# O que a figura de scores escreve como método do par. Viaja no corpo, em vez de ser montado do
# lado do desenho, para que a imagem declare o que a execução fez.
METODO_DA_PERMANOVA: Final = (
    "PERMANOVA de Anderson sobre distância euclidiana entre os componentes, com permutação dos "
    "rótulos; o teste de dispersão compara a distância de cada amostra ao centroide do próprio "
    "grupo, com as distâncias fixas e os mesmos sorteios"
)


def _permanova_dos_grupos(
    pca: StructuralPCA, matriz: RefinoMatrix, *, seed: int
) -> ResumoDaPermanova | None:
    """O par PERMANOVA e teste de dispersão sobre os componentes que esta resposta já publica.

    Recebe o `StructuralPCA` que a rota já calculou, o mesmo de que saem `scores_pc1` e
    `scores_pc2` em `_resumo_da_pca`, e nunca refaz a PCA. Há duas PCAs no projeto, e os PC2
    delas são decomposições diferentes: um `p` calculado da outra ficaria no rodapé de uma nuvem
    que ele não descreve.

    A população é a de estudo (`sample_mask`): o QC sai porque é a régua e não um grupo clínico,
    e `validate` sai porque não é amostra de estudo. Ela é menor que a nuvem desenhada, e o rodapé
    declara os dois denominadores.

    Devolve `None` quando a população não sustenta o teste. A exceção capturada é a tipada: um
    `except ValueError` largo absorveria também rótulos desalinhados dos scores, que é defeito
    desta função. Derrubar a rota seria pior: o diagnóstico vem antes de qualquer decisão, e uma
    matriz de um grupo clínico só é dado legítimo.

    `cancelled=None` porque a rota é síncrona e não passa por `api/progresso.py`. O custo é de
    dezenas de milissegundos e depende do número de amostras, não de features: as permutações
    correm sobre a matriz de distâncias entre amostras.
    """
    estudo = matriz.sample_mask
    try:
        resultado = permanova(
            pca.scores[estudo],
            [nome for nome, entra in zip(matriz.groups, estudo.tolist(), strict=True) if entra],
            # O número vem de `AnalysisSpec`, o dono dele; uma cópia aqui derivaria no dia em
            # que a tela o oferecer.
            permutations=PADRAO_DA_ANALISE.permanova_permutations,
            seed=seed,
            cancelled=None,
        )
    except PermanovaSemPopulacaoError:
        return None
    return _resumo_da_permanova(resultado)


def _resumo_da_permanova(resultado: PermanovaResult) -> ResumoDaPermanova:
    """O resultado da PERMANOVA pronto para desenhar, a partir do que já correu.

    Como em `_resumo_da_anova` e `_resumo_do_volcano`, o número que a figura imprime e o de
    qualquer outra superfície saem do mesmo objeto.
    """
    return ResumoDaPermanova(
        grupos=list(resultado.groups),
        n_por_grupo=list(resultado.n_per_group),
        n_componentes=resultado.n_components,
        permutacoes=resultado.permutations,
        semente=resultado.seed,
        estatistica=resultado.statistic,
        pvalue=resultado.pvalue,
        estatistica_da_dispersao=resultado.dispersion_statistic,
        pvalue_da_dispersao=resultado.dispersion_pvalue,
        distancia_media_ao_centroide=list(resultado.mean_distance_to_centroid),
        n_descartadas_nao_finitas=resultado.n_dropped_non_finite,
        metodo=METODO_DA_PERMANOVA,
    )


def _resumo_da_pca(pca: StructuralPCA, matriz: RefinoMatrix, *, seed: int) -> ResumoDaPca:
    """Os vetores que as quatro figuras de PCA consomem, todos de um só SVD.

    Recebe o `StructuralPCA` que a rota já calculou e nunca o recalcula: a mesma resposta publica
    `pc1_injection_correlation` e a trajetória do QC, e dois SVD dariam duas quantidades com o
    mesmo nome no mesmo corpo.

    O tipo de cada amostra vem da matriz: `qc`, `sample` e `validate` são séries distintas, e o
    agrupamento do QC no meio da nuvem é o que se olha. A ordem das linhas é a de injeção, a
    mesma de `scores`.

    As cargas usam `feature_names_used`, não `matriz.feature_names`: a PCA descarta coluna
    constante e não finita, e o vetor inteiro rotularia cada ponto com o nome de outra feature.

    O grupo de cada amostra sai de `sample_mask`, com `None` fora dele, pela mesma razão de
    `_permanova_dos_grupos`. Excluir o QC pelo nome falharia no primeiro estudo que chamasse o
    pool de outra coisa.
    """
    estudo = matriz.sample_mask.tolist()
    return ResumoDaPca(
        scores_pc1=_opcionais(pca.scores[:, 0]),
        scores_pc2=_opcionais(pca.scores[:, 1]),
        tipos_de_amostra=[amostra.sample_type for amostra in matriz.samples],
        grupos_por_amostra=[
            amostra.group if de_estudo else None
            for amostra, de_estudo in zip(matriz.samples, estudo, strict=True)
        ],
        features=list(pca.feature_names_used),
        loadings_pc1=_opcionais(pca.loadings[:, 0]),
        loadings_pc2=_opcionais(pca.loadings[:, 1]),
        variancia_explicada=list(pca.explained_variance_ratio),
        n_features_usadas=pca.n_features_used,
        n_features_totais=matriz.n_features,
        n_componentes=pca.n_components,
        # O par sai do mesmo objeto que os scores acima, então o número do rodapé descreve a
        # nuvem desenhada.
        permanova=_permanova_dos_grupos(pca, matriz, seed=seed),
    )


@router.post("/matrizes/diagnosticar", response_model=DiagnosticoResposta)
def diagnosticar(
    measurements: Annotated[UploadFile, File()],
    sample_list: Annotated[UploadFile, File()],
    mode: Annotated[str, Form()],
    orientation: Annotated[str, Form()] = "features-in-rows",
    max_spread: Annotated[float, Form()] = LIMIARES_INICIAIS["max_spread"],
    # `None` quer dizer 'não mandou' e dá o mapa calibrado por modo; um valor mandado quer dizer
    # 'eu calibrei' e vale. Com um padrão fixo as duas coisas seriam indistinguíveis. É o mesmo
    # padrão de `processar` em `rsd_cutoff` e `missing_threshold`.
    rsd_mediana_max: Annotated[float | None, Form()] = None,
    pc1_injection_correlation_max: Annotated[float, Form()] = LIMIARES_INICIAIS[
        "pc1_injection_correlation_max"
    ],
    step_sustain_min: Annotated[float, Form()] = LIMIARES_INICIAIS["step_sustain_min"],
    qc_monotonicity_min: Annotated[float, Form()] = LIMIARES_INICIAIS["qc_monotonicity_min"],
    trajectory_movement_min: Annotated[float, Form()] = LIMIARES_INICIAIS[
        "trajectory_movement_min"
    ],
    # A PERMANOVA sorteia rótulos, então a semente vem do chamador. `42` é o padrão de
    # `POST /execucoes`, para a mesma matriz dar o mesmo `p` nas duas rotas. O nome é genérico
    # porque este é o único sorteio da rota.
    # `ge=0` na semente, e `ge=1` no número de lotes e de features amostradas, em todas as rotas
    # que os recebem: o `numpy` recusa semente negativa com `ValueError`, que chegaria como 500,
    # e `n_batches=0` viraria um lote em silêncio. A faixa do `Form` recusa no disparo, com 422.
    seed: Annotated[int, Form(ge=0)] = 42,
) -> DiagnosticoResposta:
    """Diagnostica o defeito da matriz, com as fronteiras de lote como sugestão explícita.

    Recebe os dois CSV, `mode`, `orientation`, os seis limiares da `DiagnosisPolicy` e a semente
    da PERMANOVA. Os limiares viajam na requisição porque a política não tem padrão: quem tem os
    dados calibra, e os valores iniciais valem quando o cliente não manda outra coisa. Devolve o
    perfil do defeito, os lotes sugeridos, o RSD por feature antes da correção, a trajetória do
    QC no PC1, a PCA com a PERMANOVA e os grupos disponíveis com a contagem. 422 para matriz
    recusada pela validação, limiar inválido ou campo fora da faixa.
    """
    # Os bytes são lidos antes de ingerir porque o limiar calibrado depende do `sha256` das
    # medições, e um `UploadFile` só se lê uma vez.
    bruto_m = measurements.file.read()
    try:
        matriz = _carrega(measurements, sample_list, mode, orientation, conteudo_m=bruto_m)
    except IngestionError as erro:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            f"a matriz foi recusada pela validação: {len(erro.issues)} problema(s). "
            "Use /matrizes/validar para a lista completa",
        ) from erro

    # `InvalidPolicyError` vira 422 aqui. Os limiares vêm do corpo, e RSD é fração, não
    # percentual: `20` em vez de `0.20` faria a primeira regra da cascata nunca casar e um modo
    # sem degrau sair `step_change`. Sem a tradução, o cliente receberia 500.
    try:
        # O limiar de RSD mediano tem três fontes: o valor mandado vence o mapa, e o mapa vence a
        # reserva. A reserva (`LIMIARES_INICIAIS`) vale para modo que o mapa não conhece; o mapa é
        # o corte calibrado dos arquivos de referência; o valor mandado é a calibração de quem tem
        # os dados. Sem o mapa, a rota aplicaria a reserva aos modos conhecidos e mediria uma
        # cascata que a produção não roda. Passar o mapa como `None` o desliga, e o valor mandado
        # alcança todos os modos.
        politica = DiagnosisPolicy(
            max_spread=max_spread,
            rsd_mediana_max=(
                LIMIARES_INICIAIS["rsd_mediana_max"] if rsd_mediana_max is None else rsd_mediana_max
            ),
            pc1_injection_correlation_max=pc1_injection_correlation_max,
            step_sustain_min=step_sustain_min,
            qc_monotonicity_min=qc_monotonicity_min,
            trajectory_movement_min=trajectory_movement_min,
            # O mapa entra só para um dos arquivos calibrados, reconhecido pelo `sha256` das
            # medições e não pelo rótulo, que é a condição analítica e vale para qualquer estudo.
            # Todo outro conteúdo recebe a reserva, e o valor mandado continua a vencer.
            rsd_mediana_max_por_modo=(
                mapa_calibrado_para(hashlib.sha256(bruto_m).hexdigest(), mode)
                if rsd_mediana_max is None
                else None
            ),
        )
    except InvalidPolicyError as erro:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(erro)) from erro
    perfil = diagnose(matriz, policy=politica)

    # A trajetória do QC e a PCA inteira saem do mesmo SVD que as métricas: `structural_pca`
    # calcula as cargas e a variância explicada que as figuras pedem, e o objeto é passado a
    # `structural_metrics`, que confere os rótulos e usa o que recebeu. Dois SVD poriam duas
    # quantidades com o mesmo nome no mesmo corpo. O mesmo objeto dá `n_features_used`, o
    # numerador do rodapé da figura.
    #
    # A requisição ainda corre outros SVD: `diagnose`, `classify` e `extract_features`, em
    # `profile.py`, chamam `structural_metrics` cada um por si. Partilhar com eles pediria um
    # parâmetro `pca` nas três funções.
    #
    # O sinal do PC1 não é normalizado aqui: orientação canônica é obrigação de quem produz o
    # componente, e a figura declara a ressalva.
    completa = structural_pca(matriz, pretransform=PRETRANSFORM, n_components=COMPONENTES_DA_PCA)
    estrutura = structural_metrics(
        matriz, pretransform=PRETRANSFORM, n_components=COMPONENTES_DA_PCA, pca=completa
    )
    qc = matriz.qc_mask
    # O violino antes da correção. A contagem de indefinidas sai do mesmo vetor, para os dois
    # números da tela virem de um cálculo só.
    rsd_da_ingerida = _rsd_por_feature_da_ingerida(matriz, variante=VARIANTE_DA_LEITURA)
    # A proposta de lotes, e não os cortes crus: quando a detecção deixaria um lote com QC de
    # menos para medir fora da amostra, a proposta é um lote único, e os cortes detectados vão
    # junto como observação. O perfil continua com os cortes detectados, que a classificação usa.
    proposta = propose_supported_batches(matriz, list(perfil.changepoints))
    return DiagnosticoResposta(
        defect_type=perfil.defect_type,
        confidence=perfil.confidence,
        qc_rsd_median=perfil.qc_rsd_median,
        qc_rsd_p75=perfil.qc_rsd_p75,
        zero_fraction=perfil.zero_fraction,
        missing_fraction=perfil.missing_fraction,
        pc1_injection_correlation=perfil.pc1_injection_correlation,
        qc_dispersion_ratio=perfil.qc_dispersion_ratio,
        n_features=perfil.n_features,
        n_samples=perfil.n_samples,
        lotes=SugestaoDeLotes(
            changepoints=list(proposta.changepoints),
            rotulos_por_amostra=list(proposta.labels),
            n_lotes=len(set(proposta.labels)),
            qcs_por_lote=_qcs_por_lote(matriz, proposta.labels),
            cortes_detectados=list(proposta.detected_changepoints),
            qcs_por_lote_detectados=list(proposta.detected_qcs_per_batch),
        ),
        n_features_rsd_undefined=_features_sem_rsd(rsd_da_ingerida),
        rsd_variant=VARIANTE_DA_LEITURA,
        rsd_por_feature=rsd_da_ingerida,
        trajetoria_dos_qcs=TrajetoriaDosQcs(
            scores=estrutura.qc_pc1_trajectory,
            # A ordem de injeção dos QCs vem do `Sample_List`, nunca da posição da coluna:
            # `RefinoMatrix` ordena as linhas por `injection_order`, e as colunas do
            # `measurements` estão agrupadas por classe.
            ordens_de_injecao=tuple(int(v) for v in matriz.injection_orders[qc]),
            # O lote de cada QC é a sugestão do diagnóstico, não o rótulo do arquivo: a figura
            # desenha a proposta que o profissional avalia. O rótulo continua em `matriz.batches`.
            lotes=tuple(str(proposta.labels[i]) for i in np.nonzero(qc)[0]),
            # A PCA descarta coluna de variância zero e não finita, então o número de features
            # que produziu os scores não é `n_features`; o rodapé precisa dos dois.
            n_features_usadas=estrutura.n_features_used,
            n_features_totais=matriz.n_features,
        ),
        pca=_resumo_da_pca(completa, matriz, seed=seed),
        # Ordenado como na mensagem de recusa de `_grupos_ou_422`, para a interface e o erro não
        # listarem sequências diferentes. O QC sai: é uma mistura de todas as amostras, a régua
        # técnica, e o oposto de um grupo a contrastar. A exclusão usa `qc_mask` e não o nome,
        # porque o rótulo do QC vem do arquivo e pode ser escrito de outra forma. Cada grupo vai
        # com a contagem, para quem escolhe o contraste ver se o menor tem amostras que cheguem.
        grupos_disponiveis=tuple(inventario_de_grupos(matriz)),
    )


# A variante de RSD sob a qual a leitura reporta indefinição. É a variante das contagens de
# referência do domínio (234 features no `pos_hilic`, 212 no `neg_hilic`, 60 no `pos_c18` e 33 no
# `neg_c18`); sob `plain` o número seria menor e não comparável. Ela vai no corpo da resposta, ao
# lado da contagem, para o número não perder a população. Não é parâmetro da requisição porque a
# tela não a oferece como controle.
VARIANTE_DA_LEITURA: Final[RsdVariant] = "outlier_removed"


def _qcs_por_lote(matriz: RefinoMatrix, rotulos: Sequence[int]) -> list[int]:
    """Quantos QCs caem em cada lote sugerido, na ordem dos lotes.

    Conta sobre a máscara de QC, não sobre todas as amostras: a correção por lote se apoia nos
    QCs daquele lote, e um lote com um QC não tem dispersão a medir. `suggested_batches` tem uma
    entrada por amostra, na ordem da matriz, e a contagem é feita aqui porque a resposta não diz
    à tela quais posições são QC.

    Devolve uma entrada por lote existente, ordenada pelo rótulo. Lote sem QC aparece como zero
    em vez de sumir, porque é o caso que precisa ser visto.
    """
    etiquetas = np.asarray(rotulos)
    if etiquetas.size == 0:
        return []
    dos_qcs = etiquetas[matriz.qc_mask]
    return [int(np.count_nonzero(dos_qcs == lote)) for lote in sorted(set(etiquetas.tolist()))]


def _algum_lote_sem_treino(qcs: Sequence[int]) -> bool:
    """Diz se algum lote tem QC, mas poucos para treinar com um deles retido.

    Lote sem QC não conta, como em `_guard_train_size`: reter um QC não o afeta, e o corretor o
    salta. Com `min(qcs) < 3`, os cortes 24 e 34 no `neg_c18` (QCs por lote `[3, 0, 4]`) seriam
    recusados no disparo, embora a medição os aceite.

    O predicado é o do núcleo, o mesmo com que o diagnóstico decide se adota os cortes que
    detectou: uma proposta de lotes do diagnóstico nunca é recusada aqui.
    """
    return batch_without_training(qcs)


def _features_sem_rsd(rsd_por_feature: RsdPorFeature) -> int:
    """Quantas features não têm RSD de QC mensurável, sob a variante da carga.

    Indefinido não é zero nem grande: uma feature detectada num único QC não tem dispersão a
    calcular, e `qc_rsd` devolve `nan` para ela de propósito. A variante importa, porque a
    remoção de outliers cria indefinição ao apagar os QCs não nulos. A conta é a diferença entre
    o total e os finitos do lado dos QCs na carga do violino, que a mesma rota publica, para os
    dois números da tela saírem do mesmo cálculo.
    """
    return rsd_por_feature.n_features - rsd_por_feature.n_features_finitas_dos_qcs


# ---------------------------------------------------------------------------
# A curva do LOESS de uma feature, com o envelope sem um QC
# ---------------------------------------------------------------------------


@router.post("/matrizes/curva-do-loess", response_model=CurvaDoLoessResposta)
def curva_do_loess(
    measurements: Annotated[UploadFile, File()],
    sample_list: Annotated[UploadFile, File()],
    mode: Annotated[str, Form()],
    feature: Annotated[str, Form()],
    n_batches: Annotated[int, Form(ge=1)] = 1,
    seed: Annotated[int, Form(ge=0)] = 42,
    batch_cuts: Annotated[str | None, Form()] = None,
    orientation: Annotated[str, Form()] = "features-in-rows",
) -> CurvaDoLoessResposta:
    """A curva com que o LOESS corrige `feature`, e as curvas refeitas sem cada QC.

    Recebe os dois CSV, `mode`, a `feature`, os lotes (`n_batches` ou `batch_cuts`) e a semente
    da correção. Os lotes passam por `_lotes_ou_422`, como em `/etapas/correcao`, e entram nas
    amostras por `com_lotes`, como na medição dos corretores: a curva é a que a correção com
    esses lotes usa. Devolve os valores da feature, a curva ajustada, a predição de cada QC pela
    curva sem ele e o envelope dessas curvas. É síncrona porque são 1 + n_QC ajustes de uma
    feature, uma fração de segundo. 422 para feature que a matriz não tem, cortes inválidos ou
    matriz recusada.
    """
    cortes = _cortes_ou_422(batch_cuts, n_batches)
    matriz = _ingere_ou_422(measurements, sample_list, mode, orientation)
    if feature not in matriz.feature_names:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            f"`feature={feature}` não existe nesta matriz: o nome tem de ser um dos que a "
            "«Validação» devolve em `feature_names`",
        )
    com_os_lotes = com_lotes(matriz, _lotes_ou_422(matriz, n_batches, cortes))
    sensibilidade = curve_sensitivity(com_os_lotes, feature, seed=seed)
    coluna = matriz.feature_names.index(feature)
    predicao = np.full(matriz.n_samples, np.nan)
    motivos: list[str | None] = [None] * matriz.n_samples
    for linha, curva in zip(sensibilidade.held_out_rows, sensibilidade.held_out, strict=True):
        if curva.cause is None:
            predicao[linha] = curva.curve[linha]
        else:
            motivos[linha] = curva.reason
    ajuste = sensibilidade.fitted
    return CurvaDoLoessResposta(
        feature=feature,
        n_features=matriz.n_features,
        semente=seed,
        ordens_de_injecao=[int(o) for o in com_os_lotes.injection_orders],
        lotes=[str(b) for b in com_os_lotes.batches],
        qc=[bool(q) for q in com_os_lotes.qc_mask],
        valores=_opcionais(np.asarray(matriz.values[:, coluna], dtype=np.float64)),
        curva=_opcionais(ajuste.curve),
        curva_preenchida=[bool(x) for x in ajuste.filled],
        curva_sorteada=[bool(x) for x in ajuste.drawn],
        qc_no_ajuste=[bool(x) for x in ajuste.used_qc],
        motivo_sem_curva=ajuste.reason or None,
        spans_por_lote=[float(s) for s in ajuste.spans],
        predicao_sem_o_qc=_opcionais(predicao),
        motivo_sem_predicao=motivos,
        envelope_inferior=_opcionais(sensibilidade.lower),
        envelope_superior=_opcionais(sensibilidade.upper),
        envelope_nao_ajustado=[bool(x) for x in sensibilidade.not_fitted],
        n_curvas_do_envelope=sensibilidade.n_curves,
        motivo_sem_envelope=sensibilidade.degenerate,
    )


# ---------------------------------------------------------------------------
# O ponto de decisão automática, com a justificativa exibida
# ---------------------------------------------------------------------------


@router.post("/matrizes/recomendar", response_model=RecomendacaoResposta)
def recomendar(
    measurements: Annotated[UploadFile, File()],
    sample_list: Annotated[UploadFile, File()],
    mode: Annotated[str, Form()],
    orientation: Annotated[str, Form()] = "features-in-rows",
) -> RecomendacaoResposta:
    """O palpite da árvore aprendida: prediz o corretor a partir do perfil, sem rodar nenhum.

    As telas usam `POST /matrizes/palpite`, que mede os corretores em 200 features; esta rota
    serve a política aprendida a quem a pedir, a mesma que `auto=aprendida` segue. É a metade
    'antes' de um par, e a recomendação de fato é a medição de `POST /matrizes/medir-corretor`.
    Uma predição mostrada primeiro e com autoridade faz o profissional rodar o que ela disse sem
    olhar a medição, por isso a ressalva da política viaja no ponto de exibição.

    A política prediz o corretor e a margem a partir das características do perfil, e o
    `rationale` traz o caminho de decisão da árvore. Sem artefato carregável, responde 503 em
    vez de cair para uma constante que ninguém treinou. `disponivel_para_execucao` e `ressalvas`
    dizem se o método proposto pode ser executado e com que recorte; o texto das ressalvas é o
    mesmo que a CLI imprime. 422 para matriz recusada.
    """
    matriz = _ingere_ou_422(measurements, sample_list, mode, orientation)
    # O perfil sai com a política com que o corpus de treino foi diagnosticado, porque a árvore
    # foi ajustada sobre esse diagnóstico. Ver
    # `decision/features.py::POLITICA_DE_DIAGNOSTICO_DO_CORPUS`.
    perfil = perfil_para_a_politica(matriz)
    try:
        politica = carrega_politica_aprendida()
        # As duas árvores carregam no mesmo `try`: viajam juntas, e uma sem a outra é instalação
        # quebrada. Com a segunda opinião desligada (`segunda_opiniao_ativa`), ela nem é
        # carregada, e a resposta leva `segunda_opiniao` nulo.
        pela_verdade = carrega_politica_pela_verdade() if segunda_opiniao_ativa() else None
    except ArtefatoInvalidoError as erro:
        # 503, não 500: artefato ausente ou corrompido é indisponibilidade de uma dependência
        # declarada, e a mensagem já traz o comando que o regenera.
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(erro)
        ) from erro
    recomendacao = politica.recommend(perfil)
    segunda = (
        None
        if pela_verdade is None
        else SegundaOpiniao(
            correction_proposta=pela_verdade.recommend(perfil).correction,
            caminho=pela_verdade.caminho_do_corretor(perfil),
        )
    )

    return RecomendacaoResposta(
        segunda_opiniao=segunda,
        correction_proposta=recomendacao.correction,
        confianca=recomendacao.confidence,
        abstem=recomendacao.abstains,
        avaliavel=recomendacao.assessable,
        rationale=recomendacao.rationale,
        # `modelo`, não `heuristica`: não há política heurística no código, e `defect_type` não
        # separa os rótulos. Declarar `heuristica` gravaria no banco um rótulo que nada produz.
        origem="modelo",
        disponivel_para_execucao=recomendacao.correction in AVAILABLE_METHODS,
    )


# ---------------------------------------------------------------------------
# A execução, e o par de concordância
# ---------------------------------------------------------------------------


def _repositorio() -> RunRepository:
    """O repositório de execuções de pipeline; o 503 vem de `_motor_ou_503`.

    Função própria para as rotas que gravam não repetirem a composição em cada chamada.
    """
    return RunRepository(_motor_ou_503())


def _motor_ou_503() -> Engine:
    """O motor do banco, ou 503 com o que falta.

    É função própria porque a execução e o multiverso usam o mesmo motor com repositórios
    diferentes, e as traduções abaixo têm de ser as mesmas nos dois caminhos.

    503, e não 500: banco ausente ou mal configurado é indisponibilidade do serviço, não erro de
    quem chamou, e a tela traduz 500 como defeito da ferramenta. A URL sai de
    `persistence.config`, a mesma que `migrations/env.py` usa, para a aplicação e o `alembic`
    falarem com o mesmo banco; quem decide se falta configuração é `url_do_banco`.

    As três etapas ficam protegidas porque as duas últimas falham por motivo de ambiente:
    `prepara_sqlite` cria e restringe o arquivo num caminho escolhido pelo operador, e
    `create_engine` levanta quando o dialeto é conhecido e o driver não está instalado
    (`postgresql+asyncpg://`), porque a validação da URL descarta o sufixo `+driver`.

    A mensagem de `SQLAlchemyError` não é interpolada: ela carrega a URL, e a URL de Postgres
    carrega a senha. O texto daqui vai para `detail` e chega à tela.
    """
    try:
        url = url_do_banco()
    except ConfiguracaoDeBancoError as erro:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(erro)) from erro

    try:
        # O arquivo do SQLite nasce com modo 0600 antes de o motor abri-lo: ele guarda o par
        # pseudônimo/diagnóstico, e o `umask` comum daria 0644.
        prepara_sqlite(url)
    except ConfiguracaoDeBancoError as erro:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(erro)) from erro
    except OSError as erro:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            f"não foi possível preparar o arquivo que {VARIAVEL_DA_URL} nomeia: "
            f"{type(erro).__name__}. Confira permissão de escrita e espaço em disco no diretório "
            "de destino",
        ) from erro

    try:
        # `NullPool`: cada conexão fecha ao voltar. O motor nasce por requisição e nunca é
        # reaproveitado, e o pool padrão guardaria conexões abertas até o coletor de lixo passar.
        motor: Engine = create_engine(url, poolclass=NullPool)
    except (SQLAlchemyError, ImportError) as erro:
        # `ImportError` entra porque o caso comum é dialeto conhecido com driver ausente
        # (`postgresql+asyncpg://`), e o SQLAlchemy propaga o `ModuleNotFoundError` cru.
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            f"{VARIAVEL_DA_URL} foi aceite pelo vocabulário e recusada ao abrir o motor "
            f"({type(erro).__name__}). O caso comum é o sufixo de driver: o dialeto é conferido "
            "e o driver não, então `postgresql+asyncpg://` passa aqui e falha ali",
        ) from erro
    return motor


def _spec_ou_422(campos: dict[str, Any]) -> PipelineSpec:
    """`PipelineSpec` construída na fronteira, com a recusa traduzida em 422.

    A construção é a guarda: os seis `Literal` passam por `validate_label`, e a combinação
    proibida (`imputation='none'` com normalização ativa) é recusada ali. Deixar `ValueError`
    subir daria 500, e a explicação ficaria num log que o cliente não lê.
    """
    try:
        return PipelineSpec(**campos)
    except (InvalidLabelError, ValueError) as erro:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(erro)) from erro


# ---------------------------------------------------------------------------
# As quatro rotas longas: disparo com identificador, resultado reclamado depois
# ---------------------------------------------------------------------------
#
# `/execucoes`, `/matrizes/corrigir`, `/matrizes/processar` e `/matrizes/exportar` seguem o padrão
# de `POST /multiversos`: devolvem `202` com identificador e servem o progresso por `GET`, com o
# mesmo `RegistroDeProgresso`. Subir o `timeout` do cliente esconderia a espera sem resolvê-la:
# nenhum sinal de vida por minutos, e trabalho que continua no servidor depois de o cliente
# desistir. A espera passa a ser observável (progresso por etapa) e recuperável (o resultado fica
# retido e é reclamado por identificador).
#
# A fronteira não se moveu: o que produz `4xx` (rótulo fora do vocabulário, `mode` inseguro,
# `PipelineSpec` impossível, ingestão recusada) e `503` (banco ausente) continua a acontecer no
# disparo, em modo síncrono, e só o que é caro corre na thread. A falha que depende do cálculo
# (`EmptyMatrixError` do pipeline, recusa do escritor do MetaboAnalyst) chega na reclamação, com
# o mesmo código, que `codigo_http` preserva.
#
# Python não interrompe thread de fora: o cancelamento é cooperativo, pela bandeira que o trabalho
# consulta. O que se garante é que o trabalho não fique órfão: o identificador é devolvido no
# disparo e exibido na tela, o resultado fica retido, e um segundo disparo da mesma entrada é
# reclamado em vez de duplicado.

# Registro separado do multiverso: com uma instância só, `GET /multiversos/{id}` responderia
# progresso para um identificador de `corrigir`, e a mensagem do 404 daquela rota mentiria.
TRABALHOS = RegistroDeProgresso()

# Teto de trabalhos terminados retidos. O registro guarda o artefato de cada trabalho, e um
# serviço de vida longa que nunca poda acumularia CSV clínico em memória sem limite.
#
# A API não tem sessão nem identificador de cliente, então a retenção é uma contagem, e os
# reclamados são podados primeiro. `POST /matrizes/corrigir` retém dois arquivos por trabalho
# (a forma que alimenta o SERRF e a que ele devolve), cerca de 12 MB no `pos_hilic`; com 32 vagas
# o pior caso fica perto de 390 MB, teto escolhido pelo usuário. Sem teto, cem trabalhos de
# `pos_hilic` passariam de 600 MB de dado de paciente em memória.
#
# O trabalho em curso nunca é podado, só os terminais. O custo de errar para baixo é um 404 num
# identificador antigo, e a mensagem do 404 o declara.
LIMITE_DE_TRABALHOS_RETIDOS: Final[int] = 32

# O que uma etapa concluída avisa. Sem argumento de propósito: o nome da etapa sai da posição em
# `etapas`, e um nome passado à mão poderia divergir da tupla declarada.
Marca = Callable[[], None]

# O tipo do trabalho longo, que recebe a marca de etapa e a bandeira de cancelamento. Com o
# alias, as rotas mudam de assinatura juntas, e esquecer uma é erro de `mypy` em vez de uma rota
# que aceita a bandeira e a ignora.
Trabalho = Callable[[Marca, CancelCheck], object]


@dataclass(frozen=True)
class ArtefatoRetido:
    """Os bytes de um artefato e o nome do arquivo, retidos até serem reclamados.

    Tipo próprio, e não tupla, porque a rota de reclamação o distingue por `isinstance` para
    escolher entre `text/csv` e `application/json`; com uma tupla `(nome, bytes)`, o
    `Content-Disposition` sairia com o elemento errado no dia em que a ordem mudasse.

    `relatorio` é `None` quando a operação não corta features, e só `exportar` corta. Um
    relatório vazio no lugar de `None` confundiria 'esta operação não corta' com 'cortou e não
    removeu nada', e é esse campo que decide se a janela mostra o painel do corte.
    """

    nome: str
    conteudo: bytes
    relatorio: RelatorioDeExportacao | None = None
    # O segundo formato, quando a operação produz dois. É o próprio `ArtefatoRetido` aninhado,
    # pelo mesmo motivo do tipo, e o `relatorio` dele é sempre `None`: `exportar`, que corta, não
    # tem segundo formato. `None` quer dizer 'esta operação produz um arquivo', não 'o segundo
    # falhou'.
    alternativo: "ArtefatoRetido | None" = None


def _chave_do_trabalho(operacao: str, matriz: RefinoMatrix, campos: dict[str, Any]) -> str:
    """A identidade do trabalho: a operação, os parâmetros e a matriz ingerida.

    Derivada da matriz ingerida e não dos bytes do upload: um `UploadFile` só se lê uma vez, e
    dois arquivos com bytes diferentes que ingerem para a mesma matriz são o mesmo trabalho.

    Entra tudo o que muda o resultado: a operação, os campos da rota, o modo, os nomes de
    feature, os valores e os metadados de cada amostra. Deixar um de fora faria o resultado de
    uma entrada ser servido sob outra. O `sample_type` vem da linha `LABEL` e não entra em
    `values`, mas decide `qc_mask`, o treino do SERRF e o eixo do LOESS. `injection_order` é o
    `x` do LOESS: uma reescala monótona mantém a ordem das linhas e muda a curva. `batch` é
    constante nesta camada (`load_matrix` não o preenche) e entra para o dia em que deixar de
    ser. `group` não alcança corretor, mas decide o contraste do PLS-DA em `/execucoes`.
    """
    resumo = hashlib.sha256()
    resumo.update(operacao.encode("utf-8"))
    resumo.update(json.dumps(campos, sort_keys=True, default=str).encode("utf-8"))
    resumo.update(matriz.mode.encode("utf-8"))
    resumo.update("\n".join(matriz.feature_names).encode("utf-8"))
    # Cada amostra vira uma linha com os cinco campos, separados por `|`, que nenhum deles admite:
    # `name` e `group` vêm do cabeçalho do CSV, `sample_type` é rótulo fechado, e os outros dois
    # são números. Sem separador, `("AB", "C")` e `("A", "BC")` colidiriam.
    resumo.update(
        "\n".join(
            f"{a.name}|{a.group}|{a.sample_type}|{a.injection_order}|{a.batch}"
            for a in matriz.samples
        ).encode("utf-8")
    )
    resumo.update(np.ascontiguousarray(matriz.values).tobytes())
    return resumo.hexdigest()


# A memória de correções, instância única do processo, ao lado de `TRABALHOS`: as rotas são
# funções, e o estado entre chamadas precisa de um lugar. O teto e o despejo estão em
# `api/reuso.py`.
CORRECOES = MemoriaDeCorrecoes()

# O prefixo do nome da thread de trabalho. Com o identificador do trabalho no nome, quem está de
# fora acha uma thread que sobrou e pede o cancelamento, que `pede_cancelamento` recebe por
# identificador. A limpeza entre testes usa isso, e um `py-spy` sobre o executável mostra o mesmo.
PREFIXO_DA_THREAD_DE_TRABALHO = "refino-trabalho-"


def _corrige_reusando(
    corretor: Any,
    matriz: RefinoMatrix,
    *,
    batches: np.ndarray | None,
    seed: int,
    cancelled: CancelCheck,
) -> CorrectionResult:
    """`fit_transform`, ou o resultado que uma rota irmã já calculou.

    `corrigir`, `processar` e `exportar` fazem esta chamada com os mesmos argumentos, e a
    deduplicação de trabalhos não as liga, porque a chave dela leva a operação. Sob `serrf`, no
    `pos_hilic`, cada repetição custa minutos.

    O reuso é fiel porque a mesma entrada e a mesma semente produzem saída idêntica. O que
    importa é a chave: `chave_da_correcao` a monta dos argumentos que de fato viajam, e
    `tests/unit/test_identidade_do_corretor.py` mantém exaustiva a metade do construtor.

    O acerto não pula o cancelamento: a checagem antes dele continua a correr.
    """
    raise_if_cancelled(cancelled, "reuso: antes de procurar a correção guardada")
    chave = chave_da_correcao(corretor, matriz, batches=batches, seed=seed, target_features=None)
    guardado = CORRECOES.busca(chave)
    if guardado is not None:
        return guardado
    resultado = corretor.fit_transform(matriz, batches=batches, seed=seed, cancelled=cancelled)
    CORRECOES.guarda(chave, resultado)
    return resultado


def _etapa_na_posicao(etapas: tuple[str, ...], indice: int) -> str:
    """O nome da etapa naquela posição, ou um rótulo que diz que não se sabe.

    Nunca inventa, como `_celula_na_posicao`: nomear a etapa errada é pior que não nomear,
    porque quem reexecutasse a errada veria sucesso e concluiria que o defeito era intermitente.
    """
    if 0 <= indice < len(etapas):
        return etapas[indice]
    return f"depois da ultima das {len(etapas)} etapas"


def _etapa_em_curso(etapas: tuple[str, ...], feitas: int) -> str | None:
    """A etapa que corre depois de `feitas` marcas, ou `None` quando todas já terminaram.

    Difere de `_etapa_na_posicao` fora da tupla: lá, 'depois da última etapa' diz onde a
    execução morreu; aqui, depois da última marca nenhuma etapa corre, e um rótulo inventado
    seria lido pela faixa como nome de etapa.
    """
    return etapas[feitas] if 0 <= feitas < len(etapas) else None


def _executa_trabalho(identificador: str, etapas: tuple[str, ...], trabalho: Trabalho) -> None:
    """Corre o trabalho na thread de fundo, alimentando o registro e nomeando a etapa que falhar.

    Não relança, como `executa_registrando`: exceção em thread de fundo morre no
    `sys.excepthook` sem que ninguém a veja, e o progresso ficaria em `executando` para sempre.

    `HTTPException` é capturada à parte para preservar a causa: o trabalho traduz as recusas
    para o código que a rota síncrona devolveria (`409` para `EmptyMatrixError`, `422` para a
    recusa do escritor), e `codigo_http` leva esse código até a reclamação. Tudo em 500 diria
    'defeito da ferramenta' sobre um filtro que o profissional escolheu.

    A etapa que falhou vem da contagem: `marca` dispara depois de cada etapa, então quando a de
    índice `k` levanta, houve `k` avisos.

    `ExecutionCancelledError` é capturada primeiro; depois do `except Exception`, um
    cancelamento pedido apareceria como falha com código 500. A etapa onde parou é gravada também
    no cancelamento, porque decide se vale redisparar, e a rota de reclamação a usa no 410.
    """
    feitas = 0

    # A etapa que corre vai com cada aviso: depois de `k` marcas, corre a de índice `k`.
    def marca() -> None:
        nonlocal feitas
        feitas += 1
        TRABALHOS.avanca(
            identificador, feitas, len(etapas), etapa_em_curso=_etapa_em_curso(etapas, feitas)
        )

    # A ponte para o núcleo, que consulta o pedido de cancelamento a cada checagem.
    def cancelado() -> bool:
        return TRABALHOS.quer_cancelar(identificador)

    # `executando` antes da primeira etapa. `marca` dispara depois de cada etapa, e sem este aviso
    # o registro diria `pendente` ('ainda não começou') durante toda a correção, a etapa mais
    # cara, e quem espera concluiria que travou. `avanca` com zero não move o contador, só o
    # estado; `pendente` continua a querer dizer só 'registrada, thread por pegar'.
    TRABALHOS.avanca(
        identificador, feitas, len(etapas), etapa_em_curso=_etapa_em_curso(etapas, feitas)
    )

    try:
        resultado = trabalho(marca, cancelado)
    except ExecutionCancelledError:
        TRABALHOS.cancela(identificador, etapa=_etapa_na_posicao(etapas, feitas))
        return
    except HTTPException as erro:
        TRABALHOS.falha(
            identificador,
            _etapa_na_posicao(etapas, feitas),
            f"{erro.status_code}: {erro.detail}",
            codigo_http=erro.status_code,
        )
        return
    except EstruturaSemPopulacaoError as erro:
        # O mesmo 422 que o tradutor de `api/app.py` dá às rotas síncronas: o trabalho de fundo
        # muda o momento da execução, não o significado da falha.
        TRABALHOS.falha(
            identificador,
            _etapa_na_posicao(etapas, feitas),
            f"a matriz não sustenta a medida estrutural: {erro}",
            codigo_http=status.HTTP_422_UNPROCESSABLE_CONTENT,
        )
        return
    except IntensidadeNegativaError as erro:
        # O mesmo, para o SERRF: o tradutor de `api/app.py` cobre as rotas síncronas, e o SERRF
        # corre quase sempre aqui, na execução, no multiverso e no Palpite medido.
        TRABALHOS.falha(
            identificador,
            _etapa_na_posicao(etapas, feitas),
            f"a matriz não sustenta o SERRF: {erro}",
            codigo_http=status.HTTP_422_UNPROCESSABLE_CONTENT,
        )
        return
    except Exception as erro:
        TRABALHOS.falha(
            identificador,
            _etapa_na_posicao(etapas, feitas),
            f"{type(erro).__name__}: {erro}",
            codigo_http=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )
        return
    TRABALHOS.conclui(identificador, resultado=resultado)


def _despacha(
    operacao: str,
    matriz: RefinoMatrix,
    campos: dict[str, Any],
    etapas: tuple[str, ...],
    trabalho: Trabalho,
    *,
    checksum: str | None = None,
    delegacao: DelegacaoResposta | None = None,
    parametros_extra: dict[str, Any] | None = None,
) -> TrabalhoAceito:
    """Abre o trabalho, lança a thread se ele é novo, e devolve o identificador na hora.

    A thread só nasce quando o trabalho é novo: `reclamado` verdadeiro quer dizer que já há uma
    a correr para esta entrada, e uma segunda faria o mesmo cálculo competir consigo pelos
    núcleos. A poda vem antes de abrir; depois, poderia esquecer o trabalho recém-aberto se ele
    fosse o mais antigo de uma lista cheia de terminados.

    `campos` forma a chave e é gravado em `runs`; `parametros_extra` só é gravado. A caixa 'usar
    vários núcleos' vai neste segundo grupo: dois pedidos que só diferem nela produzem o mesmo
    artefato e não devem recalcular, mas a duração gravada precisa dizer em que regime o relógio
    correu.
    """
    if checksum is not None:
        trabalho = _com_registro_de_tempo(
            operacao,
            mode=matriz.mode,
            matriz=matriz,
            checksum=checksum,
            parametros={**campos, **(parametros_extra or {})},
            etapas=etapas,
            trabalho=trabalho,
        )

    TRABALHOS.esquece_terminados_alem_de(LIMITE_DE_TRABALHOS_RETIDOS)
    identificador, reclamado = TRABALHOS.abre_ou_reclama(
        len(etapas), chave=_chave_do_trabalho(operacao, matriz, campos)
    )
    if not reclamado:
        threading.Thread(
            target=_executa_trabalho,
            args=(identificador, etapas, trabalho),
            name=f"{PREFIXO_DA_THREAD_DE_TRABALHO}{identificador}",
            daemon=True,
        ).start()
    # `delegacao` é montada aqui, e não remendada no chamador: a resposta das rotas longas nasce
    # num lugar só, e o portão de progresso acha cada rota longa pelo retorno direto de
    # `_despacha`.
    return TrabalhoAceito(
        identificador=identificador,
        total=len(etapas),
        reclamado=reclamado,
        delegacao=delegacao,
    )


def _progresso_resposta(leitura: ProgressoDaExecucao) -> ProgressoResposta:
    """A leitura do registro no esquema da resposta.

    Função compartilhada entre `GET /multiversos/{id}` e `GET /trabalhos/{id}` porque a tradução é
    a mesma: duas cópias divergiriam num campo, e a tela lê os dois pela mesma função.
    """
    return ProgressoResposta(
        identificador=leitura.identificador,
        estado=leitura.estado,
        feito=leitura.feito,
        total=leitura.total,
        fracao=leitura.fracao,
        terminou=leitura.terminou,
        celula_que_falhou=leitura.celula_que_falhou,
        erro=leitura.erro,
        cancelamento_pedido=leitura.cancelamento_pedido,
        etapa_em_curso=leitura.etapa_em_curso,
    )


@router.get(
    "/trabalhos/{identificador}",
    responses={
        200: {
            # A união dos corpos que esta rota pode servir, e não só `ExecucaoResposta`. Os
            # `POST` dos trabalhos longos declaram `TrabalhoAceito`, então os outros corpos só
            # entram no OpenAPI por aqui, e o portão de contrato da janela precisa deles para
            # conhecer os campos que ela lê. `reclama_trabalho` devolve qualquer `BaseModel`
            # retido como JSON.
            "model": ExecucaoResposta
            | MedicaoDoCorretorResposta
            | PalpiteResposta
            | EtapaDoPreprocessamentoResposta
            | EtapaDaCorrecaoResposta,
            "content": {"text/csv": {}, "application/json": {}},
            "description": (
                "O resultado: os bytes do artefato, o corpo da execução, a medição dos "
                "corretores, ou o Palpite medido"
            ),
        },
        202: {"model": ProgressoResposta, "description": "Aceito e em curso — sinal de vida"},
    },
)
def reclama_trabalho(identificador: str, formato: str | None = None) -> Response:
    """Reclama o resultado de um trabalho longo, ou devolve o progresso dele.

    Responde:
    - 404 para identificador desconhecido: nunca existiu ou já foi podado, e a ação é redisparar;
    - 202 com o progresso enquanto o trabalho está `pendente` ou `executando`;
    - 200 com `text/csv` quando o resultado é artefato, ou JSON quando é corpo de execução;
    - 410 quando o trabalho foi cancelado a pedido (409 já quer dizer 'do seu dado');
    - o código que o disparo teria devolvido (409, 422 ou 500) quando falhou, com a etapa.

    `formato` escolhe qual dos dois arquivos servir quando a operação retém dois, como
    `/matrizes/corrigir` (a forma que alimenta o SERRF e a que ele devolve); sem ele, sai o
    `toserrf`. 202, e não 200, para o cliente não guardar o progresso como artefato. Reclamar
    não consome: o resultado fica até ser podado, e quem recarregou a página pode pedir de novo.
    """
    leitura = TRABALHOS.le(identificador)
    if leitura is None:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            f"não há trabalho com identificador {identificador!r}. Ou ele nunca existiu, ou "
            f"terminou há tempo bastante para ser esquecido: o serviço retém os "
            f"{LIMITE_DE_TRABALHOS_RETIDOS} trabalhos terminados mais recentes, e a ação neste "
            "caso é disparar outra vez",
        )

    if not leitura.terminou:
        return Response(
            content=_progresso_resposta(leitura).model_dump_json(),
            media_type="application/json",
            status_code=status.HTTP_202_ACCEPTED,
        )

    # Antes do ramo de falha: um trabalho cancelado tem `erro is None` e cairia ali, e o servidor
    # diria que ele falhou a quem pediu para parar.
    if leitura.estado == "cancelado":
        raise HTTPException(
            status.HTTP_410_GONE,
            f"o trabalho foi CANCELADO a seu pedido, na etapa «{leitura.celula_que_falhou}». Não "
            "há resultado a reclamar: a parada aconteceu antes de o artefato existir. Dispare "
            "outra vez se ainda o quiser — será um trabalho novo",
        )

    if leitura.estado == "falhou":
        raise HTTPException(
            leitura.codigo_http or status.HTTP_500_INTERNAL_SERVER_ERROR,
            f"o trabalho falhou na etapa «{leitura.celula_que_falhou}»: {leitura.erro}",
        )

    resultado = leitura.resultado
    # A marca de reclamado fica aqui, e não no topo: ela diz que o resultado foi servido, e a poda
    # usa isso para esquecer primeiro quem já levou o seu. No topo, marcaria também a consulta de
    # progresso e a leitura de uma falha. Marcar não consome, porque um download pode ser clicado
    # duas vezes.
    if isinstance(resultado, ArtefatoRetido):
        # A escolha do formato vem antes da marca: `_formato_ou_422` levanta 422 ou 404, e
        # marcar antes contaria como levado um arquivo que o cliente não recebeu.
        escolhido = _formato_ou_422(resultado, formato)
        TRABALHOS.marca_reclamado(identificador)
        return _resposta_csv_de_bytes(escolhido.conteudo, escolhido.nome)
    if isinstance(resultado, BaseModel):
        TRABALHOS.marca_reclamado(identificador)
        return Response(content=resultado.model_dump_json(), media_type="application/json")
    # Concluir sem resultado retido é defeito nosso. Dizê-lo é melhor que devolver 200 com corpo
    # vazio, que o cliente leria como artefato de zero bytes.
    raise HTTPException(
        status.HTTP_500_INTERNAL_SERVER_ERROR,
        f"o trabalho {identificador!r} concluiu e não retém resultado reclamável "
        f"({type(resultado).__name__}). É defeito da ferramenta, não do seu arquivo",
    )


@router.get("/trabalhos/{identificador}/relatorio", response_model=RelatorioDeExportacao)
def relatorio_do_trabalho(identificador: str) -> RelatorioDeExportacao:
    """O que o corte da exportação manteve e removeu, por causa.

    Rota irmã, e não campo da que serve o artefato: `GET /trabalhos/{id}` responde `text/csv`,
    e não há onde pôr mil nomes ali, nem no corpo, que é o CSV que sobe para o MetaboAnalyst, nem
    num cabeçalho. Segue o molde de `GET /multiversos/{id}/resultado`.

    Responde 404 para identificador desconhecido ou podado; 409 quando o trabalho ainda corre ou
    quando a operação não corta (só a exportação corta), com mensagens diferentes para quem lê o
    log e o mesmo código porque o cliente faz a mesma coisa nos dois casos; 200 com as contagens
    quando terminou uma exportação. Reclamar não consome.
    """
    leitura = TRABALHOS.le(identificador)
    if leitura is None:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            f"não há trabalho com identificador {identificador!r}. Ou ele nunca existiu, ou "
            f"terminou há tempo bastante para ser esquecido: o serviço retém os "
            f"{LIMITE_DE_TRABALHOS_RETIDOS} trabalhos terminados mais recentes, e a ação neste "
            "caso é disparar outra vez",
        )
    if not leitura.terminou:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            f"o trabalho {identificador!r} ainda corre. O relatório do corte só existe depois de "
            "o arquivo estar escrito — peça-o depois de reclamar o artefato",
        )

    resultado = leitura.resultado
    relatorio = resultado.relatorio if isinstance(resultado, ArtefatoRetido) else None
    if relatorio is None:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            f"o trabalho {identificador!r} não tem relatório de corte. Só a exportação para o "
            "MetaboAnalyst corta features — `corrigir` e `processar` devolvem a matriz inteira, e "
            "para elas a ausência é o desfecho correto e não uma falha",
        )
    return relatorio


@router.post(
    "/trabalhos/{identificador}/cancelar",
    response_model=ProgressoResposta,
    status_code=status.HTTP_202_ACCEPTED,
)
def cancela_trabalho(identificador: str) -> ProgressoResposta:
    """Pede que um trabalho longo pare; a parada é assíncrona.

    Cancelar difere de desistir: desistir encerra a espera do cliente e deixa o trabalho a
    correr; cancelar manda o trabalho parar, e o estado terminal passa a ser `cancelado`. A
    bandeira chega até o laço por feature do corretor, e a parada leva menos de um segundo no
    caminho serial.

    Responde 202 com o progresso quando o trabalho corre e o pedido ficou gravado, 404 para
    identificador desconhecido ou podado, e 409 quando o trabalho já terminou. A partição é a de
    `POST /multiversos/{id}/cancelar`, para o cliente aprender uma só.
    """
    anterior = TRABALHOS.pede_cancelamento(identificador)
    if anterior is None:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            f"não há trabalho com identificador {identificador!r}. Ou ele nunca existiu, ou "
            "terminou há tempo bastante para ser esquecido",
        )
    if anterior.terminou:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            f"o trabalho {identificador!r} já terminou — o estado é «{anterior.estado}» — e não há "
            "o que cancelar. Se quer o resultado, reclame-o",
        )
    depois = TRABALHOS.le(identificador)
    return _progresso_resposta(depois if depois is not None else anterior)


def _qualidade_para_json(relatorio: QualityReport) -> dict[str, Any]:
    """O `QualityReport` inteiro como dicionário, derivado e não redigitado.

    Nenhuma métrica única de qualidade: `asdict` percorre todos os campos do `dataclass`, então
    métrica nova chega ao cliente sozinha. Escrever os nomes à mão criaria uma segunda lista, e a
    tela deriva a dela do mesmo `QualityReport`.

    `computed` vira lista ordenada porque `frozenset` não é JSON, e a ordem fixa mantém a
    serialização determinística.
    """
    bruto = asdict(relatorio)
    bruto["computed"] = sorted(bruto["computed"])
    return bruto


def _rsd_antes_e_depois(
    crua: RefinoMatrix, corrigida: RefinoMatrix, *, correcao: str, variant: RsdVariant
) -> RsdAntesEDepois:
    """As duas medianas de RSD dos QCs, sobre a mesma população.

    Corrigir muda quais features têm RSD definido, e medianas com o denominador de antes e o de
    depois seriam de populações diferentes apresentadas como comparação. A população comum é a
    das features com RSD finito nos dois lados.

    O estimador é `in_sample` porque é o que `qc_rsd` implementa. A ressalva viaja ao lado, com
    o estimador alternativo, porque o RSD dentro da amostra favorece o LOESS por construção.

    Sem feature comparável, devolve `nan` e não zero, que diria 'RSD perfeito' sobre uma matriz
    em que ele não pôde ser medido.
    """
    antes, depois = qc_rsd(crua, variant=variant), qc_rsd(corrigida, variant=variant)
    comum = np.isfinite(antes) & np.isfinite(depois)
    return RsdAntesEDepois(
        correcao=correcao,
        rsd_variant=variant,
        rsd_estimator="in_sample",
        estimador_alternativo="leave_one_qc_out",
        mediana_antes=float(np.median(antes[comum])) if comum.any() else float("nan"),
        mediana_depois=float(np.median(depois[comum])) if comum.any() else float("nan"),
        n_features_comparadas=int(comum.sum()),
        n_features=int(antes.size),
    )


class _CaixasDeTukey(NamedTuple):
    """Os cinco números de cada coluna de uma matriz, mais a contagem do que ficou fora.

    Um retorno nomeado em vez de uma tupla posicional que cada chamador desempacotaria à sua
    maneira: trocar `q1` por `q3` numa delas não levantaria nada.
    """

    q1: np.ndarray
    mediana: np.ndarray
    q3: np.ndarray
    bigode_inferior: np.ndarray
    bigode_superior: np.ndarray
    n_discrepantes: list[int | None]


def _tukey(valores: np.ndarray) -> _CaixasDeTukey:
    """A caixa de Tukey de cada coluna de `valores`, com `nan` ignorado.

    Uma implementação para os dois eixos: as figuras de caixas resumem a mesma matriz por
    colunas e por linhas, e quem quer o outro eixo passa a transposta (`_caixas_por_amostra`).
    Duas cópias divergiriam no dia em que uma ganhasse uma correção.

    `nanpercentile`, porque o equipamento deixa a célula vazia quando não detecta, e uma célula
    não finita apagaria a caixa da coluna inteira sem contagem.

    A conta segue `matplotlib.cbook.boxplot_stats`, porque o desenho passa estes números a
    `Axes.bxp`:

    - `q1, med, q3 = percentile(x, [25, 50, 75])`, com interpolação linear, e não as
      dobradiças de `fivenum` que `metrics.rsd.boxplot_outlier_mask` replica do R; as duas dão
      números diferentes em `n` pequeno.
    - O bigode superior é `max(x[x <= q3 + coef*iqr])`, o ponto de dado, não o limiar.
    - E é `q3` quando não há ponto acima de `q3` dentro do limiar: com `[0, 0, 0, 100]`, `q3`
      é 25 e o maior ponto dentro é 0, e sem a guarda o bigode ficaria abaixo da caixa.
      Simétrico em baixo.
    - Discrepante é o que fica estritamente fora dos bigodes já resolvidos.

    `n_discrepantes` é `None` na coluna sem valor nenhum, a única divergência deliberada:
    `boxplot_stats` devolve `fliers=[]` tanto para 'nada ficou fora' quanto para 'não havia
    sobre o que decidir', e aqui a contagem vira frase na figura.

    O aviso de fatia toda `nan` é silenciado porque é o desfecho esperado de feature não
    detectada, e encheria a saída com um alerta por feature.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        q1, mediana, q3 = np.nanpercentile(valores, [25, 50, 75], axis=0)
        iqr = q3 - q1
        limite_baixo = q1 - COEFICIENTE_DE_TUKEY * iqr
        limite_alto = q3 + COEFICIENTE_DE_TUKEY * iqr

        # `np.where` sobre a própria matriz, e não uma máscara: o `nan` de uma célula ausente
        # falha as duas comparações e sai da conta pelo mesmo caminho do ponto fora do limite.
        acima = np.nanmax(np.where(valores <= limite_alto, valores, np.nan), axis=0)
        abaixo = np.nanmin(np.where(valores >= limite_baixo, valores, np.nan), axis=0)
        bigode_alto = np.where(np.isnan(acima) | (acima < q3), q3, acima)
        bigode_baixo = np.where(np.isnan(abaixo) | (abaixo > q1), q1, abaixo)

        fora = (valores < bigode_baixo).sum(axis=0) + (valores > bigode_alto).sum(axis=0)
        vazias = (~np.isnan(valores)).sum(axis=0) == 0

    return _CaixasDeTukey(
        q1=q1,
        mediana=mediana,
        q3=q3,
        bigode_inferior=bigode_baixo,
        bigode_superior=bigode_alto,
        n_discrepantes=[
            None if bool(vazia) else int(quantos)
            for vazia, quantos in zip(vazias, fora, strict=True)
        ],
    )


def _sorteia(quantos: int, *, teto: int, seed: int) -> tuple[list[int], bool]:
    """Os índices que entram na figura, e se houve sorteio.

    Uma implementação para os três chamadores (o eixo das features das caixas, o das amostras e
    o da matriz de correlação), que aplicam a mesma política com a mesma semente.

    O `teto` é argumento, sem padrão, porque cada figura tem o seu: os das caixas são os do
    MetaboAnalyst ('at most 50 features/samples'), e o do mapa de correlação é
    `TETO_DE_FEATURES_PARA_DESENHAR`. Com 68 amostras nos arquivos do laboratório, o eixo das
    amostras sorteia de fato.

    `replace=False` para a mesma unidade não aparecer duas vezes, e ordenado para a ordem das
    caixas não depender do sorteio e as duas metades não desalinharem. A semente viaja no corpo
    para a figura declará-la.
    """
    if quantos <= teto:
        return list(range(quantos)), False
    escolhidos = default_rng(seed).choice(quantos, size=teto, replace=False)
    return sorted(escolhidos.tolist()), True


def _pares_por_nome(antes: Sequence[str], depois: Sequence[str]) -> list[tuple[int, int]]:
    """Os índices das unidades que existem nos dois lados, na ordem do 'antes'.

    O alinhamento é por nome, nos dois eixos: os índices não sobrevivem à remoção, e casar por
    posição daria um par de caixas com o rótulo certo e o conteúdo de outra unidade.
    """
    indice_depois = {nome: i for i, nome in enumerate(depois)}
    return [(i, indice_depois[nome]) for i, nome in enumerate(antes) if nome in indice_depois]


def _caixas(matriz: RefinoMatrix, colunas: np.ndarray) -> list[CaixaDaDistribuicao]:
    """A caixa de Tukey de cada coluna escolhida, com `nan` ignorado.

    O nome sai da matriz que se está a medir, não de uma lista passada de fora: carimbar nas
    caixas do 'depois' os nomes do 'antes' faria o par sair idêntico por construção, e o teste
    do pareamento por nome não poderia reprovar. A conta vive em `_tukey`.

    A lista vazia retorna antes porque `np.nanpercentile` sobre zero colunas colapsa a forma
    `(3, 0)` em `(0,)`, e o desempacotamento levantaria `ValueError`. Nenhuma feature comum é
    'calculado e vazio', como `ResumoDasDistribuicoes` declara.
    """
    if colunas.size == 0:
        return []
    caixas = _tukey(matriz.values[:, colunas])
    return [
        CaixaDaDistribuicao(
            feature=matriz.feature_names[coluna],
            bigode_inferior=_numero_ou_none(caixas.bigode_inferior[i]),
            q1=_numero_ou_none(caixas.q1[i]),
            mediana=_numero_ou_none(caixas.mediana[i]),
            q3=_numero_ou_none(caixas.q3[i]),
            bigode_superior=_numero_ou_none(caixas.bigode_superior[i]),
            n_discrepantes=caixas.n_discrepantes[i],
        )
        for i, coluna in enumerate(colunas.tolist())
    ]


def _caixas_por_amostra(matriz: RefinoMatrix, linhas: np.ndarray) -> list[CaixaPorAmostra]:
    """A caixa de Tukey de cada amostra escolhida: a distribuição dela sobre as features.

    `values[linhas, :].T` põe cada amostra numa coluna, e `_tukey` reduz colunas. No `pos_hilic`
    são 68 caixas de uns 5.800 números; construí-las das médias por amostra daria uma caixa de
    68 números com o nome de uma caixa sobre milhares.

    O nome sai da matriz que se está a medir, pela mesma razão de `_caixas`, e o retorno
    antecipado cobre o mesmo caso: zero amostras comuns.
    """
    if linhas.size == 0:
        return []
    caixas = _tukey(matriz.values[linhas, :].T)
    return [
        CaixaPorAmostra(
            amostra=matriz.sample_names[linha],
            bigode_inferior=_numero_ou_none(caixas.bigode_inferior[i]),
            q1=_numero_ou_none(caixas.q1[i]),
            mediana=_numero_ou_none(caixas.mediana[i]),
            q3=_numero_ou_none(caixas.q3[i]),
            bigode_superior=_numero_ou_none(caixas.bigode_superior[i]),
            n_discrepantes=caixas.n_discrepantes[i],
        )
        for i, linha in enumerate(linhas.tolist())
    ]


def _medias(
    matriz: RefinoMatrix, colunas: np.ndarray
) -> tuple[list[float | None], list[float | None]]:
    """As médias por feature e por amostra, sobre as colunas comuns aos dois lados.

    As densidades são sobre médias de eixo e não sobre células: as células do `pos_hilic` passam
    de 180 mil e não são a distribuição que o profissional lê.
    """
    valores = matriz.values[:, colunas]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        por_feature = np.nanmean(valores, axis=0)
        por_amostra = np.nanmean(valores, axis=1)
    # Por `_opcionais`: `np.nanmean` de feature não detectada em amostra nenhuma é `nan`, e com
    # zero colunas comuns o vetor inteiro é `nan`, que não é JSON válido.
    return _opcionais(por_feature), _opcionais(por_amostra)


def _resumo_das_distribuicoes(
    crua: RefinoMatrix, processada: RefinoMatrix, *, seed: int
) -> ResumoDasDistribuicoes:
    """O antes e o depois prontos para desenhar, sobre a população comum.

    Etapas do pipeline removem features e nenhuma remove amostras, então o 'depois' é um
    subconjunto de colunas do 'antes'. Sortear do 'antes' daria caixas sem par no 'depois', e a
    figura poria lado a lado duas populações diferentes. O alinhamento é por nome, porque os
    índices não sobrevivem à remoção.

    O eixo das amostras é a transposta: uma caixa por amostra, sobre as features comuns, com o
    mesmo `_sorteia`, o mesmo `TETO_DE_CAIXAS` e a mesma semente. As amostras também são casadas
    por nome: hoje as listas são iguais, mas isso é observação sobre o pipeline, e uma etapa que
    removesse amostra produziria uma interseção menor em vez de uma caixa com o rótulo de outra.

    O sorteio é determinístico, e a semente viaja no corpo para a figura declarar o recorte
    ('50 de 6.035, sorteadas com a semente 42').
    """
    pares = _pares_por_nome(crua.feature_names, processada.feature_names)
    n_comuns = len(pares)
    escolhidos, amostrada = _sorteia(n_comuns, teto=TETO_DE_CAIXAS, seed=seed)
    antes_todas = np.array([a for a, _ in pares], dtype=np.int64)
    depois_todas = np.array([d for _, d in pares], dtype=np.int64)
    antes_caixas = np.array([pares[i][0] for i in escolhidos], dtype=np.int64)
    depois_caixas = np.array([pares[i][1] for i in escolhidos], dtype=np.int64)

    pares_de_amostra = _pares_por_nome(crua.sample_names, processada.sample_names)
    n_amostras_comuns = len(pares_de_amostra)
    escolhidas, amostrada_por_amostra = _sorteia(n_amostras_comuns, teto=TETO_DE_CAIXAS, seed=seed)
    antes_linhas = np.array([pares_de_amostra[i][0] for i in escolhidas], dtype=np.int64)
    depois_linhas = np.array([pares_de_amostra[i][1] for i in escolhidas], dtype=np.int64)

    medias_f_antes, medias_a_antes = _medias(crua, antes_todas)
    medias_f_depois, medias_a_depois = _medias(processada, depois_todas)
    return ResumoDasDistribuicoes(
        # Os rótulos são os estágios do pipeline, não nomes livres: a figura os imprime.
        matriz_antes="ingerida",
        matriz_depois="processada",
        caixas_antes=_caixas(crua, antes_caixas),
        caixas_depois=_caixas(processada, depois_caixas),
        caixas_por_amostra_antes=_caixas_por_amostra(crua, antes_linhas),
        caixas_por_amostra_depois=_caixas_por_amostra(processada, depois_linhas),
        medias_por_feature_antes=medias_f_antes,
        medias_por_feature_depois=medias_f_depois,
        medias_por_amostra_antes=medias_a_antes,
        medias_por_amostra_depois=medias_a_depois,
        n_caixas=len(escolhidos),
        n_caixas_por_amostra=len(escolhidas),
        teto_de_caixas=TETO_DE_CAIXAS,
        n_features_comparadas=n_comuns,
        n_features_antes=crua.n_features,
        n_features_depois=processada.n_features,
        n_amostras=crua.n_samples,
        n_amostras_comparadas=n_amostras_comuns,
        amostrada=amostrada,
        amostrada_por_amostra=amostrada_por_amostra,
        seed=seed,
    )


# A correção de múltiplos testes, por extenso. A legenda do volcano imprime este texto, e sem a
# correção nomeada 'zero significativas' não distingue 'não há efeito' de 'corrigi por milhares
# de testes'. É o método que `benjamini_hochberg` implementa.
CORRECAO_DE_MULTIPLOS_TESTES: Final = "Benjamini-Hochberg (FDR)"


def _opcionais(valores: np.ndarray) -> list[float | None]:
    """Um vetor `numpy` como lista JSON, com o não finito em `None` e na mesma posição.

    Explícito, e não deixado ao `jsonable_encoder`, por causa do esquema: um campo `list[float]`
    poria o OpenAPI a prometer número onde chega `null`. Nada é descartado, porque os vetores do
    volcano indexam as mesmas features, e um descarte de um lado poria `pvalue[i]` ao lado do
    `log2fc` de outra.
    """
    return [None if not np.isfinite(v) else float(v) for v in np.asarray(valores, dtype=float)]


def _numero_ou_none(valor: Any) -> float | None:
    """Um escalar como número JSON, com o não finito em `None`; o irmão escalar de `_opcionais`.

    O mesmo argumento do esquema vale para os escalares de cada caixa de Tukey, que `_tukey`
    deixa em `nan` quando a coluna não tem valor nenhum.
    """
    convertido = float(valor)
    return None if not np.isfinite(convertido) else convertido


# A pré-transformação do par de PCAs reproduz o `tools.R`, que faz `prcomp(t(...), scale.=TRUE)`
# sobre a matriz crua, sem log (linhas 1577 e 1580 no painel 'Before', 1591 e 1595 no 'After').
# A padronização que `structural_pca` aplica é esse `scale.=TRUE`, e logar mudaria a forma da
# nuvem que o laboratório reconhece no PNG.
#
# Não é `PRETRANSFORM`, a do diagnóstico, que é `log10p1`: o `qc_dispersion_ratio` daqui e o da
# aba de qualidade são quantidades diferentes com o mesmo nome, e por isso o rótulo viaja no
# corpo.
PRETRANSFORM_DO_PAR: Final[PreTransform] = "none"


def _pca_e_dispersao(matriz: RefinoMatrix) -> tuple[StructuralPCA, float]:
    """A PCA de um painel e o `qc_dispersion_ratio` do mesmo SVD.

    `structural_pca` dá os pontos e `structural_metrics` dá a medida de agrupamento, recebendo o
    objeto já calculado por `pca=`; dois SVD dariam duas quantidades com o mesmo nome no mesmo
    painel. O par de rótulos é conferido lá dentro por `_pca_ou_recalcula`. O resto de
    `StructuralMetrics` é descartado: a alternativa seria copiar a conta privada de
    `metrics.structure`.
    """
    completa = structural_pca(
        matriz, pretransform=PRETRANSFORM_DO_PAR, n_components=COMPONENTES_DA_PCA
    )
    estrutura = structural_metrics(
        matriz,
        pretransform=PRETRANSFORM_DO_PAR,
        n_components=COMPONENTES_DA_PCA,
        pca=completa,
    )
    return completa, estrutura.qc_dispersion_ratio


def _par_de_pcas(crua: RefinoMatrix, corrigida: RefinoMatrix, *, correcao: Correction) -> ParDePcas:
    """As duas PCAs do painel 'Before'/'After' que o laboratório reconhece.

    Duas PCAs independentes, e não uma projeção do 'depois' no espaço do 'antes', como em
    `tools.R:1527-1600`. A projeção mostraria a nuvem do 'depois' no espaço de uma matriz que já
    não existe. Em troca, os eixos dos dois painéis não são comparáveis, nem em escala nem em
    sinal, e o rodapé da figura diz isso.

    O 'depois' é `corrigida.matrix`, não a processada: o escopo é a correção, como em
    `_rsd_antes_e_depois`. Com a processada, imputação, normalização, transformação e
    escalonamento apertariam os QCs mesmo com `correction=none`. `_resumo_das_distribuicoes`
    mede o pipeline e por isso recebe a processada.

    Cada lado leva o seu número de colunas usadas e o digest delas. A PCA descarta coluna
    constante e não finita, e não finito é feature que o corretor estragou; sem os dois, a
    figura não diria se a queda de dispersão veio do agrupamento ou de outro espaço retido.
    Contagem igual não é conjunto igual, e só o digest denuncia a troca.

    O descarte diverge do `tools.R`, que calcula uma vez, sobre a matriz crua, as linhas com
    todos os valores infinitos (`tools.R:450-452`) e aplica o mesmo índice aos dois painéis.
    Aqui `column_usability` corre por lado e descarta a coluna com qualquer célula não finita.
    A comparação com o R não é contrato deste número, e o par contagem e digest põe a diferença
    na figura em vez de a esconder.

    `correcao` é `spec.correction`, o corretor que produziu `corrigida`; o painel direito o
    escreve no eixo, como `tools.R:1597`.
    """
    antes, dispersao_antes = _pca_e_dispersao(crua)
    depois, dispersao_depois = _pca_e_dispersao(corrigida)
    return ParDePcas(
        # Os rótulos são os estágios do pipeline, como em `_resumo_das_distribuicoes`;
        # 'corrigida' e não 'processada', pelo escopo descrito acima.
        matriz_antes="ingerida",
        matriz_depois="corrigida",
        pretransformacao=PRETRANSFORM_DO_PAR,
        n_componentes=COMPONENTES_DA_PCA,
        # O corretor que correu, que o painel direito escreve no eixo (`tools.R:1597`).
        corretor=correcao,
        scores_pc1_antes=_opcionais(antes.scores[:, 0]),
        scores_pc2_antes=_opcionais(antes.scores[:, 1]),
        # O tipo vem da matriz daquele lado: `qc` e `sample` são séries distintas, e o
        # agrupamento do QC no meio da nuvem é o que se olha.
        tipos_de_amostra_antes=[amostra.sample_type for amostra in crua.samples],
        variancia_explicada_antes=list(antes.explained_variance_ratio),
        # `qc_dispersion_ratio` é `nan` quando as amostras estão todas no mesmo ponto (`0/0` não
        # é 'sem dispersão'), e o painel escreve 'indefinida', nunca um zero que se leria como
        # agrupamento perfeito.
        qc_dispersion_ratio_antes=_numero_ou_none(dispersao_antes),
        n_features_usadas_antes=antes.n_features_used,
        n_features_totais_antes=crua.n_features,
        digest_das_colunas_retidas_antes=antes.retained_column_digest,
        scores_pc1_depois=_opcionais(depois.scores[:, 0]),
        scores_pc2_depois=_opcionais(depois.scores[:, 1]),
        tipos_de_amostra_depois=[amostra.sample_type for amostra in corrigida.samples],
        variancia_explicada_depois=list(depois.explained_variance_ratio),
        qc_dispersion_ratio_depois=_numero_ou_none(dispersao_depois),
        n_features_usadas_depois=depois.n_features_used,
        n_features_totais_depois=corrigida.n_features,
        digest_das_colunas_retidas_depois=depois.retained_column_digest,
    )


# O estimador dos dois vetores de RSD por feature: `qc_rsd` e `rsd` medem `sd/mean` sobre os
# próprios pontos, dentro da amostra. O rótulo viaja no corpo para a figura não o supor.
ESTIMADOR_DO_RSD_POR_FEATURE: Final[RsdEstimator] = "in_sample"


def _finitos(vetor: np.ndarray) -> list[float]:
    """Os valores finitos de um vetor de RSD, como `list[float]`.

    Filtrar é o que o tipo obriga: `nan` não é JSON válido, e o OpenAPI prometeria número onde
    chega `null`. `isfinite`, e não `~isnan`, porque `sd/mean` com média quase zero dá infinito.
    """
    valores = np.asarray(vetor, dtype=np.float64)
    return [float(v) for v in valores[np.isfinite(valores)]]


def _rsd_por_feature(
    corrigida: RefinoMatrix, *, correcao: Correction, variante: RsdVariant
) -> RsdPorFeature:
    """O RSD por feature dos QCs e o das amostras, sobre a matriz corrigida.

    Compõe duas chamadas já conferidas contra o `RSD()` do `tools.R`: `qc_rsd` faz o lado dos
    QCs, incluindo a guarda de zero linhas, e `rsd` sobre `sample_mask` faz o outro.

    A população das amostras é `sample_mask`, não o complemento de `qc_mask`: `SampleType` tem
    três valores, e uma injeção `validate` não entra em nenhum dos lados. Não haver nenhuma nos
    arquivos do laboratório é observação, e a escrita por complemento mediria outra coisa no
    primeiro dado novo.

    As duas contagens viajam porque as populações diferem já na matriz ingerida, e o sinal da
    diferença muda de modo para modo; uma não serve de proxy da outra. As duas causas de RSD
    indefinido ficam juntas aqui (ver `RsdPorFeature`).

    O 'depois' é a corrigida, não a processada, como em `_par_de_pcas`: o corte por RSD de QC do
    pipeline, com a mesma `rsd_variant`, já teria removido as features que esta distribuição
    existe para mostrar, e a figura desenharia o filtro em vez do efeito do corretor.
    """
    dos_qcs = qc_rsd(corrigida, variant=variante)
    das_amostras = rsd(corrigida.values[corrigida.sample_mask], axis=0, variant=variante)
    finitos_dos_qcs = _finitos(dos_qcs)
    finitos_das_amostras = _finitos(das_amostras)
    return RsdPorFeature(
        # O rótulo é o estágio, como em `_par_de_pcas`: a figura o imprime.
        matriz="corrigida",
        corretor=correcao,
        rsd_variant=variante,
        rsd_estimator=ESTIMADOR_DO_RSD_POR_FEATURE,
        rsd_dos_qcs=finitos_dos_qcs,
        rsd_das_amostras=finitos_das_amostras,
        n_features=corrigida.n_features,
        # Contadas do vetor filtrado, e não por um segundo predicado, para o `n` da figura não
        # divergir do próprio vetor.
        n_features_finitas_dos_qcs=len(finitos_dos_qcs),
        n_features_finitas_das_amostras=len(finitos_das_amostras),
        # As injeções, com a unidade no nome: separa 'não há QC nenhum' de 'há QCs e nenhuma
        # feature com RSD definido'.
        n_injecoes_dos_qcs=int(corrigida.qc_mask.sum()),
        n_injecoes_das_amostras=int(corrigida.sample_mask.sum()),
    )


# O estágio do violino antes da correção. A rota do diagnóstico corre antes de qualquer corretor,
# e `recorte.aplica` escreve o estágio no título da imagem.
ESTAGIO_DA_INGERIDA: Final = "ingerida"


def _rsd_por_feature_da_ingerida(ingerida: RefinoMatrix, *, variante: RsdVariant) -> RsdPorFeature:
    """O mesmo par de vetores sobre a matriz ingerida, para o violino antes da correção.

    Chama `_rsd_por_feature` em vez de copiá-la. Mudam os dois rótulos: `matriz` deixa de dizer
    'corrigida', e `corretor` é `None` porque nenhum correu. O `none` da chamada é só o argumento
    obrigatório, e sai na cópia, porque na imagem afirmaria uma correção que não houve.
    """
    return _rsd_por_feature(ingerida, correcao="none", variante=variante).model_copy(
        update={"matriz": ESTAGIO_DA_INGERIDA, "corretor": None}
    )


# A população da composição por classe: `sample_mask`, e não o complemento de `qc_mask`, porque
# `SampleType` tem três valores e uma injeção `validate` não entra em nenhuma das duas.
POPULACAO_DA_COMPOSICAO: Final[SampleType] = "sample"


def _classe_da_composicao(grupo: str, bloco: np.ndarray) -> ClasseDaComposicao:
    """As seis contagens e a abundância típica de um grupo.

    As três candidatas de 'faltante' (`n_zeros`, `n_nao_finitas`, `n_faltantes`) são contadas
    todas. Nos arquivos do laboratório elas coincidem porque não há célula não finita, mas o
    leitor aceita `nan` como número, e servir as três deixa quem lê a figura conferir o rótulo.
    A união vem de `preprocessing.missing.is_missing`, a mesma definição que o pipeline usa para
    recusar `imputation="none"` seguido de normalização.

    'Detectada' é célula finita e estritamente positiva, por predicado próprio. Não é
    `n_celulas - n_faltantes`, porque a célula finita negativa não é ausente para `is_missing`
    e não é aceita por `log10`.

    A abundância é a mediana do log das detectadas, e não a média das células: a média é
    governada pela cauda, e no `pos_hilic` as médias dos quatro grupos se leem como barras
    iguais, enquanto a mediana das detectadas fica dezenas de vezes abaixo delas. É a redução de
    `diagnosis/profile.py::_intensidade`, por classe.

    Sem nenhuma detectada, a abundância é `None`, nunca `nan`: a figura desenha a ausência em
    vez de uma barra de altura zero, que se leria como abundância mínima.
    """
    detectadas = bloco[np.isfinite(bloco) & (bloco > 0.0)]
    return ClasseDaComposicao(
        grupo=grupo,
        n_injecoes=int(bloco.shape[0]),
        n_celulas=int(bloco.size),
        n_zeros=int(np.count_nonzero(bloco == 0.0)),
        n_nao_finitas=int(np.count_nonzero(~np.isfinite(bloco))),
        n_faltantes=int(np.count_nonzero(is_missing(bloco))),
        n_detectadas=int(detectadas.size),
        log10_mediana_das_detectadas=(
            float(np.median(np.log10(detectadas))) if detectadas.size else None
        ),
    )


def _composicao_por_classe(matriz: RefinoMatrix) -> ComposicaoPorClasse:
    """A composição por classe clínica da matriz ingerida.

    A agregação acontece aqui, e não em `plots/`: a figura recebe contagens prontas e só divide
    e desenha. A matriz é a ingerida, e não a corrigida, porque esta figura mede o estudo e não
    a correção: o corretor cria zeros e valores não finitos (milhares de células no `pos_hilic`
    sob `batch_loess`), e a figura atribuiria à aquisição o que foi do corretor.

    As classes saem em ordem alfabética (`np.unique` ordena), para a mesma entrada dar a mesma
    figura qualquer que seja a ordem das linhas do arquivo.
    """
    valores = np.asarray(matriz.values)
    da_populacao = matriz.sample_mask
    grupos = np.asarray(matriz.groups)[da_populacao]
    bloco_da_populacao = valores[da_populacao]
    return ComposicaoPorClasse(
        # O rótulo é o estágio, como em `_par_de_pcas` e `_rsd_por_feature`.
        matriz="ingerida",
        populacao=POPULACAO_DA_COMPOSICAO,
        # Contado da máscara, e não somando `qc_mask` com `validate_mask`: a soma deixaria de
        # fora um quarto valor de `SampleType`, se ele existisse.
        n_injecoes_fora_da_populacao=int(matriz.n_samples - da_populacao.sum()),
        n_features=matriz.n_features,
        definicao_de_faltante=DEFINICAO_DE_FALTANTE,
        definicao_de_abundancia=DEFINICAO_DE_ABUNDANCIA,
        classes=[
            _classe_da_composicao(str(grupo), bloco_da_populacao[grupos == grupo])
            for grupo in np.unique(grupos)
        ],
    )


def _resumo_do_volcano(
    resultado: UnivariateResult, detalhada: ProcessedMatrix, grupos: tuple[str, str]
) -> ResumoDoVolcano:
    """Os vetores por feature do volcano, prontos para desenhar.

    Consome o resultado que a rota já calculou, nunca uma segunda chamada: a figura desenha o
    mesmo número que a aba de qualidade mostra, e `parametric=None` decide por Shapiro-Wilk,
    então duas chamadas poderiam legitimamente divergir.

    As procedências são rótulos curtos de estágio, e não as tuplas de `provenance`: a da
    processada tem onze entradas de prosa no `pos_hilic`, que estourariam a reserva do rodapé.

    `matrizes_diferem` vem de `transform_moved_the_matrix`, e não dos rótulos: com
    `transformation` e `scaling` em `none` os rótulos são dois e a matriz é uma só. A pergunta
    da figura é se os dois eixos vêm de números diferentes.

    As contagens de `UnivariateCounts` não viajam: quem decide o que é desenhável é a figura, a
    partir dos vetores, e uma contagem pronta poderia discordar da tela.

    O intervalo do fold change é calculado aqui, da mesma `detalhada`: sai da pré-transformação,
    com a máscara de imputadas que o executor marcou. Fica fora de `univariate_test` porque o
    multiverso chama aquela função sem esta matriz.
    """
    a, b = grupos
    rotulos = [amostra.group for amostra in detalhada.matrix.samples]
    intervalo = fold_change_interval(detalhada.pre_transform, grupos, detalhada.imputed_cells)
    return ResumoDoVolcano(
        features=list(resultado.feature),
        log2fc=_opcionais(resultado.log2fc),
        pvalue=_opcionais(resultado.pvalue),
        qvalue_fdr=_opcionais(resultado.qvalue_fdr),
        statistic=_opcionais(resultado.statistic),
        log2fc_ci_lower=_opcionais(intervalo.log2_lower),
        log2fc_ci_upper=_opcionais(intervalo.log2_upper),
        fold_change_ci_lower=_opcionais(intervalo.lower),
        fold_change_ci_upper=_opcionais(intervalo.upper),
        fold_change_ci_undefined=list(intervalo.undefined),
        nivel_do_intervalo=intervalo.confidence,
        grupo_a=a,
        grupo_b=b,
        n_grupo_a=rotulos.count(a),
        n_grupo_b=rotulos.count(b),
        parametrico=resultado.parametric,
        rota_da_normalidade=resultado.normality.route,
        shapiro_pvalue=resultado.normality.shapiro_pvalue,
        n_residuos=resultado.normality.n_residuals,
        alfa_fdr=FDR_ALPHA,
        correcao=CORRECAO_DE_MULTIPLOS_TESTES,
        matriz_do_pvalue=MATRIZ_DO_PVALUE,
        matriz_do_fold_change=MATRIZ_DO_FOLD_CHANGE,
        matrizes_diferem=detalhada.transform_moved_the_matrix,
        n_features_testadas=len(resultado.feature),
    )


# Os dois rótulos de matriz do volcano, que são estágios do pipeline e que a figura imprime.
MATRIZ_DO_PVALUE: Final = "processada"
MATRIZ_DO_FOLD_CHANGE: Final = "pré-transformação"


def _resumo_do_plsda(
    resultado: PLSDAResult, processada: RefinoMatrix, grupos: tuple[str, str]
) -> ResumoDoPlsda:
    """Os três vetores do PLS-DA com a validação ao lado.

    PLS-DA nunca sai sem `q2` e o p-valor da permutação. Consome o `PLSDAResult` que a rota já
    calculou: o `q2` da aba de qualidade e o da figura saem do mesmo objeto, e duas chamadas
    dariam dois modelos, com permutações que nem coincidiriam.

    A seleção das amostras abaixo repete a de `statistics/plsda._matriz_e_rotulos`, que é
    privada; torná-la pública furaria o `__all__` que garante que nenhum caminho público expõe
    scores soltos. Por isso a divergência levanta: se as seleções discordarem, o comprimento não
    bate com `resultado.scores` e a rota recusa com 500, em vez de servir um vetor de grupos
    deslocado.

    `grupos[1]` é a classe positiva, porque `_matriz_e_rotulos` monta `y = (rotulos == segundo)`.
    A ordem não muda número nenhum (`orient_canonically` desfaz a troca de sinal), só qual
    grupo a figura chama `y = 1`; os dois nomes viajam com o papel de cada um.
    """
    negativo, positivo = grupos
    selecionadas = tuple(a for a in processada.samples if a.group in grupos)
    if len(selecionadas) != int(resultado.scores.shape[0]):
        raise HTTPException(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            f"a seleção de amostras do contraste devolveu {len(selecionadas)} e o PLS-DA correu "
            f"sobre {resultado.scores.shape[0]}. As duas descrevem o mesmo conjunto, e servir o "
            "vetor de grupos desalinhado poria amostras de um grupo no lado do outro",
        )
    rotulos = [a.group for a in selecionadas]
    return ResumoDoPlsda(
        q2=resultado.q2,
        permutation_pvalue=resultado.permutation_pvalue,
        r2y=resultado.r2y,
        n_permutations=resultado.n_permutations,
        cv_folds=PLSDA_CV_FOLDS,
        n_components=resultado.n_components,
        seed=resultado.seed,
        scores_componente_1=_opcionais(resultado.scores[:, 0]),
        scores_componente_2=_opcionais(resultado.scores[:, 1]),
        # Os scores de validação cruzada saem do mesmo laço que dá o `q2` deste corpo, então o
        # par e os pontos que a figura desenha não podem vir de modelos diferentes.
        scores_cv_componente_1=_opcionais(resultado.cv_scores[:, 0]),
        scores_cv_componente_2=_opcionais(resultado.cv_scores[:, 1]),
        grupos_por_amostra=rotulos,
        grupo_positivo=positivo,
        grupo_negativo=negativo,
        n_positivo=rotulos.count(positivo),
        n_negativo=rotulos.count(negativo),
        features=list(processada.feature_names),
        loadings_componente_1=_opcionais(resultado.loadings[:, 0]),
        loadings_componente_2=_opcionais(resultado.loadings[:, 1]),
        # `None` onde o VIP não é calculável: `_vip` devolve `nan` quando a soma de quadrados
        # total é zero, e um `0.0` diria 'sem importância' sobre uma feature cuja importância
        # ninguém conseguiu calcular.
        vip=_opcionais(resultado.vip),
        n_features=processada.n_features,
    )


# Os três parâmetros do PLS-DA, nomeados porque viajam para o `QualityReport` como procedência
# (`plsda_cv_folds` e `plsda_n_permutations` ficam ao lado do `p`). `cv_folds = 7` é o padrão de
# `statistics/plsda.py`, e coincidir com o número de QCs destes arquivos é acaso: a estratificação
# é por grupo, não por QC.
PLSDA_N_COMPONENTS: Final = 2
PLSDA_CV_FOLDS: Final = 7
PLSDA_N_PERMUTATIONS: Final = 200

# O corte de `q` para contar feature significativa. É o mesmo do multiverso
# (`multiverse/stability.py`), senão a tela e a grade contariam quantidades diferentes com o
# mesmo nome.
FDR_ALPHA: Final = 0.05

# A justificativa gravada quando o cliente declara `sem_proposta`. Ela vale nos dois casos do
# estado: a máquina não propôs nada, ou propôs (o Palpite guardado) e a pessoa não seguiu.
RAZAO_SEM_PROPOSTA: Final = "Parâmetros do pesquisador; nenhuma proposta da máquina foi seguida."


def _plsda_viavel_ou_422(matriz: RefinoMatrix, grupos: tuple[str, str] | None) -> None:
    """Recusa no disparo um par que a validação cruzada não consegue estratificar.

    `StratifiedKFold` com menos membros que dobras não põe um de cada grupo em cada dobra: o
    `Q²` mediria outra coisa, ou a chamada levantaria minutos depois, dentro da thread, na etapa
    errada. A mensagem nomeia as duas contagens, para quem a lê saber se o problema é o par ou o
    desenho do estudo. Nos arquivos do laboratório o menor grupo tem 13 amostras, e o ramo tem
    teste próprio.
    """
    if grupos is None:
        return
    contagens = {nome: int(sum(1 for g in matriz.groups if g == nome)) for nome in grupos}
    pequenos = sorted(nome for nome, n in contagens.items() if n < PLSDA_CV_FOLDS)
    if pequenos:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            f"o PLS-DA usa validação cruzada de {PLSDA_CV_FOLDS} dobras estratificadas, e "
            f"{pequenos} não tem amostras que cheguem. Contagens deste par: {contagens}. "
            "Escolha outro par, ou execute sem declarar grupos — a execução corre igual e o "
            "PLS-DA fica como «não calculado» em vez de sair medido sobre dobras degeneradas",
        )


@router.post("/execucoes", status_code=status.HTTP_202_ACCEPTED, response_model=TrabalhoAceito)
def executar(
    measurements: Annotated[UploadFile, File()],
    sample_list: Annotated[UploadFile, File()],
    mode: Annotated[str, Form()],
    estudo: Annotated[str, Form()],
    correction: Annotated[str, Form()],
    n_batches: Annotated[int, Form(ge=1)],
    rsd_variant: Annotated[str, Form()],
    imputation: Annotated[str, Form()],
    sample_normalization: Annotated[str, Form()],
    transformation: Annotated[str, Form()],
    scaling: Annotated[str, Form()],
    apresentacao: Annotated[str, Form()],
    # Tem padrão enquanto os irmãos não têm: exigir este quebraria os clientes escritos antes de
    # a etapa existir, e `none` reproduz o comportamento anterior, então o padrão não esconde
    # escolha nenhuma.
    ranking_filter: Annotated[str, Form()] = "none",
    # Os três do segundo passe têm padrão pelo mesmo motivo.
    ranking_keep_fraction: Annotated[float | None, Form()] = None,
    normalization_reference: Annotated[str | None, Form()] = None,
    # O terceiro arquivo: o CSV de fatores de normalização. A rota o lê e resolve o mapa antes de
    # construir a especificação, para o registro carregar os fatores por valor.
    normalization_factors: Annotated[UploadFile | None, File()] = None,
    ranking_filter_2: Annotated[str, Form()] = "none",
    ranking_keep_fraction_2: Annotated[float | None, Form()] = None,
    orientation: Annotated[str, Form()] = "features-in-rows",
    rsd_cutoff: Annotated[float | None, Form()] = None,
    missing_threshold: Annotated[float | None, Form()] = None,
    seed: Annotated[int, Form(ge=0)] = 42,
    # Os cortes de lote: lista JSON de ordens de injeção, cada uma a primeira de um lote novo. É
    # a proposta do diagnóstico, quando o usuário mantém o número de lotes que ela propôs.
    # Ausente, os `n_batches` lotes são blocos iguais.
    batch_cuts: Annotated[str | None, Form()] = None,
    campos_sobrescritos: Annotated[str, Form()] = "[]",
    proposta_json: Annotated[str | None, Form()] = None,
    # Ligado, o cliente declara que a máquina não propôs nada para esta execução, ou que a pessoa
    # não seguiu a proposta, e a linha única sai com `origem="manual"`. A rota não sabe isso
    # sozinha: sem eixos e sem proposta, o pedido é igual quando a proposta foi seguida e quando
    # ela nunca existiu. Desligado por padrão, para quem chama a rota sem o campo.
    sem_proposta: Annotated[bool, Form()] = False,
    group_a: Annotated[str | None, Form()] = None,
    group_b: Annotated[str | None, Form()] = None,
    auto: Annotated[str | None, Form()] = None,
    # A caixa 'usar vários núcleos' da janela, ligada por padrão. O `serrf` e o `loess` a leem;
    # o `none` não calcula.
    paralelo: Annotated[bool, Form()] = True,
    # O contrato de análise, como um campo JSON. Não é campo de `PipelineSpec` de propósito: a
    # impressão digital da spec é a chave de cache do multiverso, e um tipo de post-hoc lá dentro
    # invalidaria matriz processada por mudança que não a toca. Ausente, vale o padrão: as
    # análises baratas ligadas e as caras desligadas.
    analise_json: Annotated[str | None, Form()] = None,
) -> TrabalhoAceito:
    """Dispara o pipeline com a spec do humano e devolve `202` com identificador.

    Recebe os dois CSV, os eixos da spec, o estudo, a apresentação, o par de grupos, os eixos
    sobrescritos com a proposta da máquina, o contrato de análise e, opcionalmente, `auto`, que
    delega o corretor à política e grava a justificativa dela na linha de decisão. O resultado
    (`ExecucaoResposta`, com os ids, o estado de concordância lido do banco, o `QualityReport`
    inteiro e o par antes e depois) é reclamado em `GET /trabalhos/{identificador}`.

    A segunda linha de decisão é gravada quando `campos_sobrescritos` não é vazio, e não quando
    a spec difere: abrir o controle, avaliar e manter é concordância medida, e seria
    indistinguível de nunca ter aberto.

    Recusa no disparo: 422 para rótulo, apresentação, par de grupos ou spec inválidos, matriz
    recusada ou método indisponível; 503 para banco ausente, porque `_repositorio()` corre antes
    da thread, e descobrir isso só na reclamação faria esperar a correção inteira.
    """
    # A guarda de `apresentacao` fica aqui, e não só em `grava_execucao`: lá, um valor inválido
    # atravessaria o pipeline inteiro antes de reprovar, e com o banco ausente sairia como 503 em
    # vez de 422.
    try:
        validate_apresentacao(apresentacao)
    except ValueError as erro:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(erro)) from erro

    eixos = _eixos_ou_422(campos_sobrescritos)
    if correction in UNAVAILABLE_METHODS:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            f"`correction={correction}` não é selecionável: "
            f"{UNAVAILABLE_METHODS[correction].reason}",
        )

    spec = _spec_ou_422(
        {
            "correction": correction,
            "n_batches": n_batches,
            "rsd_cutoff": rsd_cutoff,
            "ranking_filter": ranking_filter,
            "ranking_keep_fraction": ranking_keep_fraction,
            "ranking_filter_2": ranking_filter_2,
            "ranking_keep_fraction_2": ranking_keep_fraction_2,
            "rsd_variant": rsd_variant,
            "missing_threshold": missing_threshold,
            "imputation": imputation,
            "sample_normalization": sample_normalization,
            "normalization_reference": normalization_reference,
            "normalization_factors": _fatores_do_upload(normalization_factors),
            "transformation": transformation,
            "scaling": scaling,
            "seed": seed,
            "batch_cuts": _cortes_ou_422(batch_cuts, n_batches),
        }
    )
    # Antes de `_sobrescrita_ou_422`: `sem_proposta` com eixo e sem proposta cairia lá como 'não
    # veio `proposta_json`', o contrário do que o cliente declarou.
    _sem_proposta_ou_422(sem_proposta, eixos=eixos, proposta_json=proposta_json, auto=auto)
    sobrescrita = _sobrescrita_ou_422(eixos, proposta_json)

    bruto_m = measurements.file.read()
    bruto_l = sample_list.file.read()
    sample_list.file.seek(0)
    matriz = _ingere_ou_422(measurements, sample_list, mode, orientation, conteudo_m=bruto_m)
    lotes = _lotes_ou_422(matriz, spec.n_batches, spec.batch_cuts)
    # O diagnóstico gravado com a execução usa o mesmo critério de `/matrizes/diagnosticar`: o
    # mapa calibrado só para os arquivos de referência.
    perfil = diagnose(
        matriz,
        policy=DiagnosisPolicy(
            **LIMIARES_INICIAIS,
            rsd_mediana_max_por_modo=mapa_calibrado_para(hashlib.sha256(bruto_m).hexdigest(), mode),
        ),
    )

    # A guarda do par acontece no disparo, como todo 4xx das rotas longas: um nome de grupo
    # errado descoberto pela thread viraria `falhou` minutos depois, com a correção já paga. Reusa
    # `_grupos_ou_422`, a mesma de `POST /multiversos`, para a mesma falta dar a mesma mensagem.
    grupos_plsda = _grupos_ou_422(matriz, group_a, group_b)
    _plsda_viavel_ou_422(matriz, grupos_plsda)

    # A delegação vem depois do diagnóstico e reescreve só o `correction`: a política decide o
    # corretor e nada diz sobre os outros eixos. `replace`, e não construção nova à mão, para a
    # validação da spec ficar num lugar só.
    delegacao = (
        None
        if auto is None
        else _delegacao_ou_422(
            auto,
            matriz,
            n_batches=spec.n_batches,
            cortes=spec.batch_cuts,
            lotes=lotes,
            paralelo=paralelo,
            seed=spec.seed,
        )
    )
    if delegacao is not None:
        spec = replace(spec, correction=cast(Correction, delegacao.correction))

    corretor = _corretor_da_spec(spec.correction, paralelo=paralelo)
    repo = _repositorio()

    # Os campos da chave são compostos uma vez, aqui, e o mesmo objeto vai para `_despacha` e
    # para a chave de invalidação que a resposta leva; duas cópias poderiam divergir, e a rota
    # devolveria uma chave e reusaria por outra. A delegação já correu, então a chave vê o
    # corretor que de fato vai correr. A composição vive em `compoe_campos_da_chave` para o teste
    # da bijeção importá-la em vez de reconstruí-la.
    analise = _analise_ou_422(analise_json)
    # As etapas das análises são resolvidas uma vez e passadas às duas pontas.
    etapas_de_analise = etapas_das_analises(analise)
    campos_da_chave: dict[str, Any] = compoe_campos_da_chave(
        spec,
        analise=analise,
        estudo=estudo,
        apresentacao=apresentacao,
        campos_sobrescritos=list(eixos),
        proposta_json=proposta_json,
        sem_proposta=sem_proposta,
        group_a=group_a,
        group_b=group_b,
    )
    chave_de_invalidacao = _chave_do_trabalho("execucoes", matriz, campos_da_chave)

    def trabalho(marca: Marca, cancelado: CancelCheck) -> object:
        # O instante do início é capturado aqui e viaja até a gravação; sem ele, as duas colunas
        # de tempo de `runs` receberiam o mesmo `datetime.now()`.
        comecou_em = datetime.now(UTC)
        # O corretor recebe a semente da spec; sem ela, `fit_transform` usaria o padrão 42 e a
        # semente do cliente não alcançaria a correção. `_corrige_reusando` devolve a correção
        # que uma rota irmã já calculou com a mesma chave, e
        # `tests/unit/test_identidade_do_corretor.py` confere essa chave.
        corrigida = _corrige_reusando(
            corretor,
            matriz,
            batches=lotes,
            seed=spec.seed,
            cancelled=cancelado,
        )
        marca()
        raise_if_cancelled(cancelado, "executar: antes do pipeline")
        try:
            # `process_matrix_detailed`, e não `process_matrix`, porque o volcano precisa das
            # duas matrizes: o `p` sai da processada, que é a que o profissional vai analisar, e o
            # fold change sai da pré-transformação, onde o MetaboAnalyst o calcula. `.matrix` é o
            # que `process_matrix` devolve.
            detalhada = process_matrix_detailed(corrigida.matrix, spec)
            processada = detalhada.matrix
        except (EmptyMatrixError, NormalizacaoInvalidaError) as erro:
            # Normalização inválida (como uma referência com erro de digitação) ou matriz
            # esvaziada é erro do dado: 409, e não o 500 que a tela mostra como defeito da
            # ferramenta.
            raise HTTPException(status.HTTP_409_CONFLICT, str(erro)) from erro
        marca()
        raise_if_cancelled(cancelado, "executar: antes do relatorio de qualidade")
        # O PLS-DA é calculado quando o par de grupos é declarado, depois de `process_matrix`: a
        # pergunta é sobre a matriz que o profissional vai analisar, o mesmo ponto de medição do
        # relatório.
        resultado_plsda = (
            None
            if grupos_plsda is None
            else plsda(
                processada,
                grupos_plsda,
                n_components=PLSDA_N_COMPONENTS,
                cv_folds=PLSDA_CV_FOLDS,
                n_permutations=PLSDA_N_PERMUTATIONS,
                seed=spec.seed,
                # A bandeira chega ao PLS-DA, que sem ela seria um bloco de alguns segundos sem
                # ponto de parada.
                cancelled=cancelado,
            )
        )
        # A contagem de features significativas também depende do par de grupos: sem par, a
        # métrica fica 'não calculada', que é outro fato que 'nenhuma rota calcula isto'. Corre
        # sobre a matriz processada, pelo mesmo motivo do PLS-DA. Neste corpus o valor esperado é
        # zero: nenhuma feature sobrevive ao FDR depois da correção de múltiplos testes, e mostrar
        # isso é mais honesto que declarar a métrica indisponível.
        raise_if_cancelled(cancelado, "executar: antes da contagem de FDR")
        n_significativas = None
        volcano = None
        if grupos_plsda is not None:
            # Uma chamada de `univariate_test` serve a contagem e o volcano: a figura desenha o
            # mesmo número que a aba de qualidade mostra, e com `parametric=None` a decisão por
            # Shapiro sobre resíduos poderia divergir entre duas chamadas. `fold_change_matrix`
            # só alcança `fold_change` e `log2fc`; `pvalue` e `qvalue_fdr` saem de `processada`.
            #
            # O `q` não finito (feature sem variância nos dois grupos) fica fora da contagem, e
            # `contagens.qvalue_undefined` diz quantas ficaram de fora.
            #
            # O `parametric` é a escolha da tela: 'Automático' é `None`, o padrão de
            # `AnalysisSpec`; 'Welch' e 'Mann-Whitney' forçam o teste, e a rota da normalidade
            # passa a ser `forced`, que o rodapé do volcano declara.
            univariado = univariate_test(
                processada,
                grupos_plsda,
                PARAMETRIC_BY_UNIVARIATE_TEST[analise.univariate_test],
                fold_change_matrix=detalhada.pre_transform,
            )
            q = np.asarray(univariado.qvalue_fdr, dtype=np.float64)
            definidos = np.isfinite(q)
            n_significativas = int((q[definidos] < FDR_ALPHA).sum())
            volcano = _resumo_do_volcano(univariado, detalhada, grupos_plsda)
        relatorio = build_quality_report(
            processada,
            measurement_point="post_pipeline",
            structural_pretransform="none",
            plsda=resultado_plsda,
            plsda_groups=grupos_plsda,
            plsda_cv_folds=None if resultado_plsda is None else PLSDA_CV_FOLDS,
            n_significant_fdr=n_significativas,
        )
        # O par antes e depois mede a correção, não o pipeline: sai de `matriz` (como veio) e de
        # `corrigida.matrix` (só o corretor). Os resumos da correção vêm da mesma função das
        # prévias do passo a passo, para a figura da etapa 1 ser igual à desta execução.
        resumos_da_correcao = _resumos_da_correcao(matriz, corrigida.matrix, spec)
        comparacao = resumos_da_correcao["rsd_antes_e_depois"]
        # O par de PCAs corre antes de gravar: `structural_pca` levanta quando não sobra coluna
        # usável, e a matriz corrigida pode chegar a isso, porque o corretor cria não finitos.
        # Depois da gravação, o banco teria a linha e o trabalho reportaria falha. A falha vira
        # 409, como `EmptyMatrixError`: é fato sobre o dado, não defeito da ferramenta.
        pcas = resumos_da_correcao["pcas_da_correcao"]
        marca()
        # As famílias de análise pedidas: uma etapa anunciada e um aviso por família, na ordem da
        # tupla de etapas (ver `corre_as_analises`). Correm sobre `processada`, o mesmo ponto de
        # medição do PLS-DA e do teste univariado. `imputadas` vem do mesmo executor: a
        # normalidade conta à parte as variáveis com valor imputado no grupo, e depois de uma
        # normalização por amostra só o executor sabe quais foram.
        resultados_das_analises = corre_as_analises(
            analise,
            matriz=processada,
            imputadas=detalhada.imputed_cells,
            seed=spec.seed,
            marca=marca,
            cancelado=cancelado,
        )
        # A última checagem de cancelamento. Cancelar depois de a linha estar no banco deixaria
        # uma execução gravada e um estado `cancelado` a dizer que ela não aconteceu.
        raise_if_cancelled(cancelado, "executar: antes de gravar no banco")
        gravada = repo.grava_execucao(
            estudo=estudo,
            mode=mode,
            checksum_entrada=checksum_do_par(bruto_m, bruto_l),
            n_samples=matriz.n_samples,
            n_features=matriz.n_features,
            amostras=_amostras_para_gravar(matriz),
            perfil=perfil,
            spec_json=json.dumps(_spec_para_json(spec), ensure_ascii=False, sort_keys=True),
            seed=spec.seed,
            # A justificativa tem quatro formas, e distingui-las é o que torna a concordância
            # legível: com sobrescrita, o registro é do gesto do humano; sob delegação, é o
            # argumento da máquina; sem proposta, a escolha é do pesquisador; sem nada disso, é a
            # ausência de gesto. `origem="manual"` numa linha só é o que `classifica_concordancia`
            # lê como `sem_proposta`.
            rationale=(
                f"Escolha do pesquisador. Eixos sobrescritos: {list(eixos) or 'nenhum'}."
                if eixos
                else delegacao.rationale
                if delegacao is not None
                else RAZAO_SEM_PROPOSTA
                if sem_proposta
                else "Proposta aceita sem que o controle de sobrescrita fosse exercido."
            ),
            origem="manual" if eixos or sem_proposta else "modelo",
            # O relatório inteiro, nenhuma métrica escolhida à mão. `_qualidade_para_json` é a
            # mesma função da resposta, derivada por `asdict`: métrica nova é gravada sozinha.
            metrics_json=json.dumps(_qualidade_para_json(relatorio), ensure_ascii=False),
            apresentacao=apresentacao,
            sobrescrita=sobrescrita,
            iniciado_em=comecou_em,
        )
        marca()
        return ExecucaoResposta(
            matrix_id=gravada.matrix_id,
            run_id=gravada.run_id,
            decision_id=gravada.decision_id,
            decision_proposta_id=gravada.decision_proposta_id,
            # Lido do banco, e não composto aqui.
            estado_concordancia=repo.classifica_concordancia(
                gravada.matrix_id, gravada.diagnosis_id
            ),
            campos_sobrescritos=list(eixos),
            apresentacao=apresentacao,
            qc_rsd_median=relatorio.qc_rsd_median,
            n_features_saida=processada.n_features,
            qualidade=_qualidade_para_json(relatorio),
            rsd_antes_e_depois=comparacao,
            chave_de_invalidacao=chave_de_invalidacao,
            # `matriz` e `processada`, e não `corrigida.matrix`: a figura de normalização mede o
            # pipeline, enquanto o par de RSD acima mede a correção.
            distribuicoes=_resumo_das_distribuicoes(matriz, processada, seed=spec.seed),
            # O par de PCAs mede a correção, e por isso o 'depois' é `corrigida.matrix`. O 'antes'
            # nasce aqui, e não do diagnóstico que o painel já guarda: a sequência arquivo A,
            # diagnosticar, executar, arquivo B, diagnosticar desenharia o 'antes' de B ao lado do
            # 'depois' de A, porque o diagnóstico não viaja sob a chave de invalidação da
            # execução. Calculado acima, antes da gravação; aqui só se transporta.
            pcas_da_correcao=pcas,
            # Os dois vetores de RSD por feature, sobre `corrigida.matrix`: entre ela e a
            # processada está o corte por RSD de QC, que remove as features cuja dispersão esta
            # figura mostra. A variante é a da spec, escolha do pesquisador.
            rsd_por_feature=resumos_da_correcao["rsd_por_feature"],
            # A composição por classe mede o estudo, e por isso usa a matriz ingerida: na
            # corrigida, as células faltantes criadas pelo corretor apareceriam como não detecção
            # do equipamento.
            composicao_por_classe=resumos_da_correcao["composicao_por_classe"],
            # Os três vetores do PLS-DA vão com a validação: `q2` e `permutation_pvalue` são
            # campos obrigatórios do mesmo modelo que carrega os scores. `None` sem par de grupos
            # quer dizer 'não calculado', não 'vazio'.
            plsda=(
                None
                if resultado_plsda is None or grupos_plsda is None
                else _resumo_do_plsda(resultado_plsda, processada, grupos_plsda)
            ),
            # `None` quando não há par: o teste univariado é pareado, e um objeto vazio diria que
            # correu e não achou nada.
            volcano=volcano,
            # `None` quando a família não foi pedida ou a matriz não a sustenta ('não calculado');
            # 'calculado e vazio' é o objeto com zero sobreviventes, o desfecho normal deste
            # corpus. Ver `ExecucaoResposta.anova`.
            anova=resultados_das_analises.anova,
            # Duas raízes: o mapa e a busca partilham o coeficiente e não o alvo, e
            # `pattern_reference` a `None` salta a busca sem desligar o mapa.
            correlacao=resultados_das_analises.correlacao,
            busca_por_padrao=resultados_das_analises.busca_por_padrao,
            # Duas raízes pelo mesmo motivo: `kmeans` e `som` têm bandeiras separadas, ambas
            # desligadas por padrão porque custam, e pedir uma sem a outra é o caso comum.
            kmeans=resultados_das_analises.kmeans,
            som=resultados_das_analises.som,
            # Uma raiz para a rede: as arestas viajam dentro dela, e `arestas` vazia é achado.
            # Uma raiz separada confundiria 'não correu' com 'correu e não achou'.
            rede=resultados_das_analises.rede,
            # Duas raízes porque `classifiers` é uma tupla: `("random_forest",)` e
            # `("random_forest", "svm")` são pedidos diferentes.
            random_forest=resultados_das_analises.random_forest,
            svm=resultados_das_analises.svm,
            # Uma raiz, que não depende do par de grupos: os grupos saem da matriz pela máscara
            # de estudo, como na ANOVA, e `None` quer dizer só que a família não foi pedida.
            normalidade=resultados_das_analises.normalidade,
        )

    return _despacha(
        "execucoes",
        matriz,
        # O `estudo` e a `apresentacao` entram na chave: duas execuções da mesma matriz e spec
        # em estudos diferentes são trabalhos diferentes, e cada uma grava a sua linha. Duas
        # execuções idênticas em tudo produzem uma linha só, porque a segunda reclama o mesmo
        # trabalho; quem quisesse registrar a mesma execução duas vezes não consegue por esta
        # porta.
        #
        # O par de grupos também entra: o `PLSDAResult` vai para o `QualityReport` e para o
        # banco, e fora da chave dois pedidos com pares diferentes reclamariam o mesmo trabalho,
        # e o profissional receberia o `Q²` de outro contraste.
        #
        # O dicionário é o mesmo objeto que a resposta hasheou (ver `campos_da_chave` acima).
        campos_da_chave,
        # A tupla cresce com as famílias pedidas, e as quatro etapas fixas ficam literais: o
        # portão estático de `tests/test_api_progresso.py` conta os literais da tupla contra os
        # avisos escritos neste corpo. O `*` ele não consegue contar, e por isso exige que a rota
        # declare um substituto de runtime (`DINAMICAS_COM_SUBSTITUTO`). Este comentário não
        # nomeia a função do aviso, porque o portão conta ocorrências textuais.
        (
            "correcao",
            "pipeline",
            "relatorio de qualidade",
            *etapas_de_analise,
            "gravacao no banco",
        ),
        trabalho,
    )


# ---------------------------------------------------------------------------
# O passo a passo: duas prévias sem estado, e o `/execucoes` como etapa final
# ---------------------------------------------------------------------------
#
# Cada prévia recebe os arquivos de novo, e nenhuma rota aceita identificador de sessão. O que
# liga uma etapa à seguinte é o cache de correção por conteúdo (`api/reuso.py`), que o
# `/execucoes` também lê: a correção cara corre uma vez, e o pré-processamento, de segundos, é
# refeito. A etapa 3 é o `/execucoes`, a única que grava no banco.
#
# Nenhuma das duas passa `checksum=` a `_despacha`: com ele o trabalho gravaria uma linha de
# tempo no banco, e uma prévia não registra nada.


def _resumos_da_correcao(
    matriz: RefinoMatrix, corrigida: RefinoMatrix, spec: PipelineSpec
) -> dict[str, Any]:
    """Os quatro resumos das figuras da correção, comuns ao `/execucoes` e às prévias.

    A composição por classe mede a matriz ingerida; os outros três, a corrigida. Um par de PCAs
    que não se pode medir vira `409`, porque é fato do dado corrigido e não defeito do pedido.
    """
    try:
        pcas = _par_de_pcas(matriz, corrigida, correcao=spec.correction)
    except ValueError as erro:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            f"o par de PCAs não pôde ser medido depois da correção «{spec.correction}»: {erro}",
        ) from erro
    return {
        "rsd_antes_e_depois": _rsd_antes_e_depois(
            matriz, corrigida, correcao=spec.correction, variant=spec.rsd_variant
        ),
        "pcas_da_correcao": pcas,
        "rsd_por_feature": _rsd_por_feature(
            corrigida, correcao=spec.correction, variante=spec.rsd_variant
        ),
        "composicao_por_classe": _composicao_por_classe(matriz),
    }


def _spec_da_correcao_ou_422(
    correction: str,
    n_batches: int,
    rsd_variant: str,
    seed: int,
    batch_cuts: tuple[int, ...] | None = None,
) -> PipelineSpec:
    """A especificação da etapa 1: os eixos da correção e um pré-processamento neutro.

    Passa pelo mesmo `_spec_ou_422` do `/execucoes` e por isso tem as mesmas recusas de
    vocabulário. O pré-processamento neutro não corre: existe porque `PipelineSpec` só tem padrão
    para a semente, e a etapa 1 não deve recusar por uma escolha que só a etapa 2 usa.
    """
    if correction in UNAVAILABLE_METHODS:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            f"`correction={correction}` não é selecionável: "
            f"{UNAVAILABLE_METHODS[correction].reason}",
        )
    return _spec_ou_422(
        {
            "correction": correction,
            "n_batches": n_batches,
            "rsd_cutoff": None,
            "ranking_filter": "none",
            "ranking_keep_fraction": None,
            "ranking_filter_2": "none",
            "ranking_keep_fraction_2": None,
            "rsd_variant": rsd_variant,
            "missing_threshold": None,
            "imputation": "none",
            "sample_normalization": "none",
            "normalization_reference": None,
            "normalization_factors": None,
            "transformation": "none",
            "scaling": "none",
            "seed": seed,
            "batch_cuts": batch_cuts,
        }
    )


def _delega_na_spec(
    spec: PipelineSpec,
    auto: str | None,
    matriz: RefinoMatrix,
    *,
    lotes: np.ndarray | None = None,
    paralelo: bool = True,
) -> tuple[PipelineSpec, DelegacaoResposta | None]:
    """Aplica a delegação do `--auto`, que reescreve só o corretor, como no `/execucoes`."""
    delegacao = (
        None
        if auto is None
        else _delegacao_ou_422(
            auto,
            matriz,
            n_batches=spec.n_batches,
            cortes=spec.batch_cuts,
            lotes=lotes,
            paralelo=paralelo,
            seed=spec.seed,
        )
    )
    if delegacao is not None:
        spec = replace(spec, correction=cast(Correction, delegacao.correction))
    return spec, delegacao


@router.post(
    "/etapas/correcao", status_code=status.HTTP_202_ACCEPTED, response_model=TrabalhoAceito
)
def etapa_da_correcao(
    measurements: Annotated[UploadFile, File()],
    sample_list: Annotated[UploadFile, File()],
    mode: Annotated[str, Form()],
    correction: Annotated[str, Form()],
    n_batches: Annotated[int, Form(ge=1)],
    rsd_variant: Annotated[str, Form()],
    seed: Annotated[int, Form(ge=0)] = 42,
    batch_cuts: Annotated[str | None, Form()] = None,
    orientation: Annotated[str, Form()] = "features-in-rows",
    auto: Annotated[str | None, Form()] = None,
    paralelo: Annotated[bool, Form()] = True,
) -> TrabalhoAceito:
    """Etapa 1 do passo a passo: corrige a matriz e devolve os resumos das figuras da correção.

    Recebe os dois arquivos e os eixos da correção e responde `202` com o trabalho aceito. Recusa
    com `422` um corretor indisponível ou um valor fora do vocabulário. O corretor, a semente e os
    lotes são os do `/execucoes`, o que deixa a etapa 3 achar esta correção no cache. `paralelo`
    muda o tempo e não o resultado, e por isso fica fora da chave.
    """
    spec = _spec_da_correcao_ou_422(
        correction, n_batches, rsd_variant, seed, _cortes_ou_422(batch_cuts, n_batches)
    )
    matriz = _ingere_ou_422(measurements, sample_list, mode, orientation)
    lotes = _lotes_ou_422(matriz, spec.n_batches, spec.batch_cuts)
    spec, delegacao = _delega_na_spec(spec, auto, matriz, lotes=lotes, paralelo=paralelo)
    corretor = _corretor_da_spec(spec.correction, paralelo=paralelo)
    campos = {
        "correction": spec.correction,
        "n_batches": spec.n_batches,
        "rsd_variant": spec.rsd_variant,
        "seed": spec.seed,
        # Dois pedidos com o mesmo `n_batches` e cortes diferentes não são o mesmo trabalho.
        "batch_cuts": None if spec.batch_cuts is None else list(spec.batch_cuts),
    }
    chave_da_etapa = _chave_do_trabalho("etapa_correcao", matriz, campos)

    def trabalho(marca: Marca, cancelado: CancelCheck) -> object:
        corrigida = _corrige_reusando(
            corretor,
            matriz,
            batches=lotes,
            seed=spec.seed,
            cancelled=cancelado,
        )
        marca()
        raise_if_cancelled(cancelado, "etapa da correção: antes dos resumos")
        resumos = _resumos_da_correcao(matriz, corrigida.matrix, spec)
        marca()
        return EtapaDaCorrecaoResposta(
            correcao=spec.correction, chave_da_etapa=chave_da_etapa, **resumos
        )

    return _despacha(
        "etapa_correcao",
        matriz,
        campos,
        ("correcao", "resumos da correcao"),
        trabalho,
        delegacao=delegacao,
    )


@router.post(
    "/etapas/pre-processamento", status_code=status.HTTP_202_ACCEPTED, response_model=TrabalhoAceito
)
def etapa_do_preprocessamento(
    measurements: Annotated[UploadFile, File()],
    sample_list: Annotated[UploadFile, File()],
    mode: Annotated[str, Form()],
    correction: Annotated[str, Form()],
    n_batches: Annotated[int, Form(ge=1)],
    rsd_variant: Annotated[str, Form()],
    imputation: Annotated[str, Form()],
    sample_normalization: Annotated[str, Form()],
    transformation: Annotated[str, Form()],
    scaling: Annotated[str, Form()],
    rsd_cutoff: Annotated[float | None, Form()] = None,
    missing_threshold: Annotated[float | None, Form()] = None,
    ranking_filter: Annotated[str, Form()] = "none",
    ranking_keep_fraction: Annotated[float | None, Form()] = None,
    ranking_filter_2: Annotated[str, Form()] = "none",
    ranking_keep_fraction_2: Annotated[float | None, Form()] = None,
    normalization_reference: Annotated[str | None, Form()] = None,
    normalization_factors: Annotated[UploadFile | None, File()] = None,
    seed: Annotated[int, Form(ge=0)] = 42,
    batch_cuts: Annotated[str | None, Form()] = None,
    orientation: Annotated[str, Form()] = "features-in-rows",
    auto: Annotated[str | None, Form()] = None,
    paralelo: Annotated[bool, Form()] = True,
) -> TrabalhoAceito:
    """Etapa 2 do passo a passo: corrige pelo cache, pré-processa e devolve os resumos das duas.

    Recebe os dois arquivos e todos os eixos do `/execucoes` e responde `202` com o trabalho
    aceito. Recusa com `422` o que o `/execucoes` recusa: corretor indisponível, vocabulário
    inválido e `imputation='none'` com normalização ativa. Se o pré-processamento esvazia a
    matriz ou a normalização não se aplica a ela, o trabalho termina com `409`. As distribuições
    comparam a matriz ingerida com a processada.
    """
    if correction in UNAVAILABLE_METHODS:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            f"`correction={correction}` não é selecionável: "
            f"{UNAVAILABLE_METHODS[correction].reason}",
        )
    spec = _spec_ou_422(
        {
            "correction": correction,
            "n_batches": n_batches,
            "rsd_cutoff": rsd_cutoff,
            "ranking_filter": ranking_filter,
            "ranking_keep_fraction": ranking_keep_fraction,
            "ranking_filter_2": ranking_filter_2,
            "ranking_keep_fraction_2": ranking_keep_fraction_2,
            "rsd_variant": rsd_variant,
            "missing_threshold": missing_threshold,
            "imputation": imputation,
            "sample_normalization": sample_normalization,
            "normalization_reference": normalization_reference,
            "normalization_factors": _fatores_do_upload(normalization_factors),
            "transformation": transformation,
            "scaling": scaling,
            "seed": seed,
            "batch_cuts": _cortes_ou_422(batch_cuts, n_batches),
        }
    )
    matriz = _ingere_ou_422(measurements, sample_list, mode, orientation)
    lotes = _lotes_ou_422(matriz, spec.n_batches, spec.batch_cuts)
    spec, delegacao = _delega_na_spec(spec, auto, matriz, lotes=lotes, paralelo=paralelo)
    corretor = _corretor_da_spec(spec.correction, paralelo=paralelo)
    campos = asdict(spec)
    chave_da_etapa = _chave_do_trabalho("etapa_preprocessamento", matriz, campos)

    def trabalho(marca: Marca, cancelado: CancelCheck) -> object:
        corrigida = _corrige_reusando(
            corretor,
            matriz,
            batches=lotes,
            seed=spec.seed,
            cancelled=cancelado,
        )
        marca()
        raise_if_cancelled(cancelado, "etapa do pré-processamento: antes do pipeline")
        try:
            processada = process_matrix_detailed(corrigida.matrix, spec).matrix
        except (EmptyMatrixError, NormalizacaoInvalidaError) as erro:
            raise HTTPException(status.HTTP_409_CONFLICT, str(erro)) from erro
        marca()
        raise_if_cancelled(cancelado, "etapa do pré-processamento: antes dos resumos")
        resumos = _resumos_da_correcao(matriz, corrigida.matrix, spec)
        marca()
        return EtapaDoPreprocessamentoResposta(
            correcao=spec.correction,
            chave_da_etapa=chave_da_etapa,
            n_features_saida=processada.n_features,
            distribuicoes=_resumo_das_distribuicoes(matriz, processada, seed=spec.seed),
            **resumos,
        )

    return _despacha(
        "etapa_preprocessamento",
        matriz,
        campos,
        ("correcao", "pipeline", "resumos das etapas"),
        trabalho,
        delegacao=delegacao,
    )


def _eixos_ou_422(campos_sobrescritos: str) -> tuple[str, ...]:
    """Os eixos exercidos, validados contra `SUPPORTED_OVERRIDES` na fronteira.

    Eixo fora da lista levanta. Ignorá-lo gravaria um par de concordância sobre um eixo que a
    enumeração não declara, numa linha cujo significado ninguém reconstrói depois.
    """
    try:
        pedidos = json.loads(campos_sobrescritos)
    except json.JSONDecodeError as erro:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            f"`campos_sobrescritos` não é JSON: {erro}. Esperava um vetor de nomes de eixo",
        ) from erro
    if not isinstance(pedidos, list):
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            f"`campos_sobrescritos` é {type(pedidos).__name__} e tem de ser vetor de nomes",
        )
    try:
        return validate_override_fields(str(campo) for campo in pedidos)
    except ValueError as erro:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(erro)) from erro


def _sobrescrita_ou_422(
    eixos: tuple[str, ...], proposta_json: str | None
) -> SobrescritaExercida | None:
    """A linha A do par, ou `None` quando o controle não foi exercido.

    Recusa as duas metades soltas: eixos sem proposta não têm com que concordar, e proposta sem
    eixos gravaria um par cujo eixo ninguém sabe nomear. As duas dariam linha inclassificável.
    """
    if not eixos:
        if proposta_json:
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_CONTENT,
                "veio `proposta_json` e `campos_sobrescritos` está vazio. O gatilho do par é o "
                "controle ter sido exercido; sem eixo, o par ficaria sem eixo que "
                "alguém soubesse nomear",
            )
        return None
    if not proposta_json:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            f"o controle foi exercido em {list(eixos)} e não veio `proposta_json`. Sem a "
            "proposta da máquina não há com que concordar, e concordância é relação entre dois "
            "valores",
        )
    try:
        proposta = json.loads(proposta_json)
        return SobrescritaExercida(
            spec_proposta_json=str(proposta["spec_json"]),
            rationale_proposta=str(proposta["rationale"]),
            origem_proposta=str(proposta["origem"]),
            campos=eixos,
        )
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as erro:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT, f"`proposta_json` inválido: {erro}"
        ) from erro


def _sem_proposta_ou_422(
    sem_proposta: bool, *, eixos: tuple[str, ...], proposta_json: str | None, auto: str | None
) -> None:
    """Recusa `sem_proposta` ao lado de qualquer coisa que seja proposta da máquina.

    As três recusas são a mesma contradição: eixo sobrescrito diverge de uma proposta,
    `proposta_json` é a proposta, e `auto` é a máquina escolhendo o corretor. Com `sem_proposta`,
    qualquer delas gravaria 'nenhuma proposta foi seguida' sobre uma execução que seguiu ou
    comparou uma. A recusa fica no disparo, antes da ingestão, para o `4xx` não esperar a correção.
    """
    if not sem_proposta:
        return
    contradicoes = [
        nome
        for nome, presente in (
            ("campos_sobrescritos", bool(eixos)),
            ("proposta_json", bool(proposta_json)),
            ("auto", auto is not None),
        )
        if presente
    ]
    if contradicoes:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            f"veio `sem_proposta` e também {', '.join(f'`{c}`' for c in contradicoes)}. "
            "`sem_proposta` declara que nenhuma proposta da máquina foi seguida nesta execução; "
            "sobrescrever, mandar a proposta ou delegar a escolha são formas de haver proposta",
        )


def _fatores_do_upload(arquivo: UploadFile | None) -> tuple[tuple[str, float], ...] | None:
    """Lê o CSV de fatores enviado e devolve o mapa resolvido.

    A `PipelineSpec` guarda o conteúdo, não o arquivo. Um caminho ou identificador de upload
    faria a impressão digital carregar a referência: arquivos diferentes partilhariam célula de
    cache, e reexecutar pelo registro só funcionaria enquanto o arquivo original existisse.

    A análise vive em `refino.io.fatores`, partilhada com a CLI; aqui se decide só de onde vêm os
    bytes e que o erro vira `422`. A guarda de valor (positivo, finito, sem duplicados) fica em
    `PipelineSpec` e vale igual para a CLI, a rota e o registro.
    """
    if arquivo is None:
        return None
    bruto = arquivo.file.read()
    arquivo.file.seek(0)
    try:
        return le_fatores_de_bytes(bruto, origem="o arquivo de fatores")
    except FatoresMalformadosError as erro:
        # `422` e não `500`: o defeito é do arquivo enviado, inclusive um `csv.Error` (que não
        # deriva de `ValueError`), e não da ferramenta.
        raise HTTPException(status_code=422, detail=str(erro)) from erro


def _spec_para_json(spec: PipelineSpec) -> dict[str, Any]:
    """A spec inteira, com chaves ordenadas na serialização.

    Ordenada porque o `spec_json` das duas linhas do par é comparado por igualdade de texto em
    `classifica_concordancia`: duas serializações do mesmo objeto em ordem diferente diriam
    'discordando' sobre specs idênticas.
    """
    # Derivado dos campos de `PipelineSpec`: uma lista escrita à mão fica para trás quando entra
    # um campo novo, e o par, comparado por texto, deixa de poder concordar.
    return {campo.name: getattr(spec, campo.name) for campo in fields(spec)}


def _analise_ou_422(analise_json: str | None) -> AnalysisSpec:
    """O contrato de análise vindo do formulário, ou o padrão.

    Vem num campo JSON, como `proposta_json` e `campos_sobrescritos`, e não em vinte campos de
    formulário, que levariam `executar` a perto de cinquenta parâmetros por um objeto que é um
    contrato só. Quem valida os rótulos é o `__post_init__` de `AnalysisSpec`; aqui o `ValueError`
    do domínio vira `422` com a mensagem dele, em vez de `500`.
    """
    if analise_json is None:
        return PADRAO_DA_ANALISE

    try:
        cru = json.loads(analise_json)
    except json.JSONDecodeError as erro:
        raise HTTPException(
            status_code=422,
            detail=(
                f"`analise_json` não é JSON válido: {erro}. Esperava um objeto com os campos de "
                "`AnalysisSpec`, ou o campo ausente para receber os padrões."
            ),
        ) from erro

    if not isinstance(cru, dict):
        raise HTTPException(
            status_code=422,
            detail=(
                f"`analise_json` é {type(cru).__name__} e esperava um objeto. Um vetor ou um "
                "escalar não nomeiam campo nenhum."
            ),
        )

    conhecidos = {campo.name for campo in fields(AnalysisSpec)}
    desconhecidos = sorted(set(cru) - conhecidos)
    if desconhecidos:
        # Campo desconhecido é recusado: ignorar um nome mal escrito devolveria o padrão em
        # silêncio, no lugar do que a pessoa pediu.
        raise HTTPException(
            status_code=422,
            detail=(
                f"`analise_json` traz campo que `AnalysisSpec` não tem: {desconhecidos}. "
                f"Os aceitos são {sorted(conhecidos)}."
            ),
        )

    if "classifiers" in cru and isinstance(cru["classifiers"], list):
        cru["classifiers"] = tuple(cru["classifiers"])

    try:
        return AnalysisSpec(**cru)
    except (ValueError, TypeError) as erro:
        raise HTTPException(status_code=422, detail=str(erro)) from erro


@dataclass(frozen=True)
class ResultadosDasAnalises:
    """O que as famílias pedidas produziram, uma por campo; o retorno de `corre_as_analises`.

    Um campo por família deixa uma família nova mudar só este arquivo, sem tocar a assinatura de
    `corre_as_analises` nem os chamadores. Campo `None` significa 'não correu', nunca 'correu e deu
    vazio'; `ExecucaoResposta.anova` preserva a mesma distinção do lado do fio.
    """

    anova: ResumoDaAnova | None = None
    correlacao: ResumoDaCorrelacao | None = None
    busca_por_padrao: ResumoDaBuscaPorPadrao | None = None
    """Campo próprio, e não parte de `correlacao`.

    `AnalysisSpec.pattern_reference` a `None` salta a busca sem desligar o mapa: as duas partilham
    o coeficiente, não o alvo. Aninhada, a ausência de uma pareceria implicar a da outra.
    """
    kmeans: ResumoDoKmeans | None = None
    som: ResumoDoSom | None = None
    """Dois campos para as duas famílias particionais.

    Vêm do mesmo módulo e partilham o escalonamento, mas são pedidas por bandeiras separadas:
    `kmeans` sem `som` é o caso comum, porque o segundo custa mais. Um campo só faria a ausência
    de uma parecer implicar a da outra.
    """
    rede: ResumoDaRede | None = None
    """A rede de correlação parcial esparsa desenviesada.

    `None` significa 'não correu': a bandeira `dspc` vem desligada por padrão, e a população pode
    não sustentar o ajuste. 'Correu e não achou aresta' é `arestas` vazia, um desfecho esperado.
    """
    random_forest: ResumoDoClassificador | None = None
    svm: ResumoDoClassificador | None = None
    """Dois campos para os dois classificadores, e não uma lista.

    Pedir só um é o caso comum, porque cada um custa minutos; uma lista faria a ausência de um
    parecer implicar a do outro. Os dois partilham o esquema porque descrevem a mesma quantidade
    sobre modelos diferentes.
    """
    normalidade: ResumoDaNormalidade | None = None
    """A normalidade por variável e por grupo; `None` só quando a família não foi pedida.

    Não há guarda de população: grupo pequeno e variável constante viram motivos por variável
    dentro do resultado.
    """


# O nome do teste por variável, como o texto e o arquivo o escrevem.
TESTE_DA_NORMALIDADE: Final = "Shapiro-Wilk"


def _resumo_da_normalidade(resultado: NormalityResult) -> ResumoDaNormalidade:
    """O resultado da normalidade pronto para o fio, a partir do que a família já calculou.

    O resumo de cada grupo sai das propriedades de `GroupNormality`, que contam os mesmos vetores
    que atravessam aqui, então o texto e o CSV não discordam. `_opcionais` põe `None` onde não há
    número, na mesma posição, para o `p` de uma variável nunca ficar ao lado do nome de outra.
    """
    return ResumoDaNormalidade(
        features=list(resultado.feature),
        grupos=[
            NormalidadeDoGrupo(
                grupo=grupo.group,
                n_amostras=grupo.n_samples,
                w=_opcionais(grupo.statistic),
                pvalue=_opcionais(grupo.pvalue),
                qvalue=_opcionais(grupo.qvalue),
                n_valores=[int(n) for n in grupo.n_values],
                nao_avaliavel=list(grupo.unassessable),
                valor_imputado=[bool(v) for v in grupo.has_imputed],
                n_variaveis=len(resultado.feature),
                n_menos_de_tres=grupo.n_fewer_than_three,
                n_constantes=grupo.n_constant,
                n_com_imputado=grupo.n_with_imputed,
                n_com_imputado_rejeitadas=grupo.n_with_imputed_rejected,
                n_testadas=grupo.n_tested,
                n_rejeitadas=grupo.n_rejected,
                fracao_rejeitada=grupo.fraction_rejected,
                n_rejeitadas_fdr=grupo.n_rejected_fdr,
                fracao_minima_nao_normal=grupo.min_fraction_non_normal,
            )
            for grupo in resultado.groups
        ],
        teste=TESTE_DA_NORMALIDADE,
        matriz=MATRIZ_DO_PVALUE,
        alfa=resultado.rejection_level,
        nivel_do_fdr=resultado.fdr_level,
        lambda_de_storey=resultado.storey_lambda,
    )


def _a_populacao_sustenta_a_anova(matriz: RefinoMatrix) -> bool:
    """Se a matriz tem grupos de estudo que cheguem para a ANOVA correr.

    Repete o predicado de `statistics.anova._populacao`, que é privado, com os mesmos limiares
    importados do núcleo. Perguntar antes evita um `except ValueError` largo, que esconderia um
    defeito de programação atrás de 'a ANOVA não correu'. Deixar o erro subir também não serve:
    a ANOVA é opcional e vem ligada por padrão, e a execução inteira não deve cair porque a matriz
    tem um só grupo. A família não corre e o campo fica `None`.
    """
    estudo = [a.group for a, e in zip(matriz.samples, matriz.sample_mask, strict=True) if e]
    grupos = set(estudo)
    if len(grupos) < MIN_GRUPOS:
        return False
    return all(estudo.count(nome) >= MIN_AMOSTRAS_POR_GRUPO for nome in grupos)


def _resumo_da_anova(resultado: AnovaResult) -> ResumoDaAnova:
    """O resultado da ANOVA pronto para desenhar.

    Lê o resultado que a rota já calculou, sem segunda chamada, para a figura e as outras
    superfícies mostrarem o mesmo número. Tudo sai do resultado, que conta só as amostras de
    estudo; contar os rótulos da matriz daria números maiores com o mesmo nome. A procedência vai
    como rótulo curto porque a tupla de `AnovaResult.provenance` estouraria o rodapé da figura.

    `n_features_testadas` é a matriz inteira: as features que a ANOVA não testa (grupo pequeno
    demais ou feature constante) ficam na tabela com `nan`. O nome do campo é do esquema; a figura
    escreve 'features da matriz'. Descontá-las aqui exigiria a população total num segundo campo
    e quebraria a conferência de comprimento de `plots.anova.pontos_da_anova`.
    """
    return ResumoDaAnova(
        features=list(resultado.feature),
        statistic=_opcionais(resultado.statistic),
        pvalue=_opcionais(resultado.pvalue),
        qvalue_fdr=_opcionais(resultado.qvalue_fdr),
        grupos=list(resultado.groups),
        n_por_grupo=list(resultado.n_per_group),
        metodo=resultado.method,
        post_hoc_metodo=resultado.post_hoc.method,
        post_hoc_rota=resultado.post_hoc.route,
        alfa_fdr=resultado.fdr_alpha,
        correcao=CORRECAO_DE_MULTIPLOS_TESTES,
        min_por_grupo=resultado.min_per_group,
        matriz=MATRIZ_DA_ANOVA,
        # A matriz inteira; ver a docstring sobre o nome do campo.
        n_features_testadas=len(resultado.feature),
    )


# A matriz sobre a qual a ANOVA corre: a processada, que é a que a pessoa vai analisar, como em
# `MATRIZ_DO_PVALUE`. Constante própria para uma mudança de estágio não mover a outra em silêncio.
MATRIZ_DA_ANOVA: Final = "processada"

# O mesmo para a correlação e a busca por padrão, com constante própria pelo mesmo motivo.
MATRIZ_DA_CORRELACAO: Final = "processada"


def _resumo_da_correlacao(
    resultado: CorrelationResult, *, amostrada: bool, seed: int
) -> ResumoDaCorrelacao:
    """A matriz de correlação pronta para desenhar, lida do resultado que a rota já calculou.

    A frase de amostragem é montada aqui porque só este lado sabe se houve sorteio e com que
    semente, e `plots/` não importa `refino.api`. `n_indefinidas` soma os dois descartes, que o
    núcleo guarda separados (constante e não finita), porque o rodapé conta uma coisa só: quantas
    features ficaram sem coeficiente.
    """
    return ResumoDaCorrelacao(
        features=list(resultado.features),
        valores=[_opcionais(linha) for linha in resultado.matrix],
        metodo=resultado.method,
        matriz=MATRIZ_DA_CORRELACAO,
        n_amostras=resultado.n_samples_used,
        n_features_totais=resultado.n_features_total,
        n_indefinidas=(
            resultado.n_features_dropped_constant + resultado.n_features_dropped_non_finite
        ),
        amostragem=(
            f"as {resultado.n_features_used} sorteadas de {resultado.n_features_total}, com a "
            f"semente {seed}"
            if amostrada
            else f"todas as {resultado.n_features_used} features utilizáveis"
        ),
    )


def _resumo_da_busca_por_padrao(resultado: PatternSearchResult) -> ResumoDaBuscaPorPadrao:
    """A busca por padrão pronta para desenhar.

    `ordem_dos_grupos` viaja porque é ela que dá sentido ao perfil: um vetor `0,1,2,3` sem a ordem
    não diz a que grupo cada posição se refere. Não há recorte, ao contrário do mapa: a busca dá
    um número por feature, e quantas se desenham é decisão do painel.
    """
    return ResumoDaBuscaPorPadrao(
        features=list(resultado.features),
        coeficiente=_opcionais(np.asarray(resultado.coefficient, dtype=float)),
        referencia=resultado.reference,
        tipo_da_referencia=resultado.reference_kind,
        perfil=list(resultado.profile),
        ordem_dos_grupos=list(resultado.group_order),
        metodo=resultado.method,
        matriz=MATRIZ_DA_CORRELACAO,
        n_amostras=resultado.n_samples_used,
        n_features_totais=resultado.n_features_total,
        n_indefinidas=(
            resultado.n_features_dropped_constant
            + resultado.n_features_dropped_non_finite
            + resultado.n_coefficient_undefined
        ),
    )


# A matriz sobre a qual as duas famílias particionais correm, com constante própria pelo motivo
# de `MATRIZ_DA_ANOVA`.
MATRIZ_DO_AGRUPAMENTO: Final = "processada"


def _a_populacao_sustenta_o_agrupamento(matriz: RefinoMatrix) -> bool:
    """Se a matriz tem amostras e features que cheguem para o K-means e para o SOM.

    Pergunta antes pelo motivo de `_a_populacao_sustenta_a_anova`. A população é a matriz inteira,
    e não `sample_mask`: a unidade é a amostra, e um QC que caia junto das amostras de estudo é
    achado técnico legítimo (ver `statistics/clustering.py`). `kmeans_k` não é conferido aqui:
    um `k` acima do número de amostras é pedido inválido, e volta como `422` por
    `_kmeans_ou_422`.
    """
    if matriz.n_samples < 2:
        return False
    utilizaveis, _constantes, _nao_finitas = column_usability(matriz.values)
    return int(utilizaveis.sum()) >= 1


def _resumo_do_kmeans(resultado: KMeansResult) -> ResumoDoKmeans:
    """A partição pronta para desenhar, lida do resultado que a rota já calculou.

    Uma segunda chamada só repetiria a partição com a mesma semente; por isso a figura lê este
    objeto. A frase de amostragem é montada aqui pelo motivo de `_resumo_da_correlacao`. Esta
    família não sorteia features, porque a partição é sobre as amostras, e a frase diz isso.
    """
    return ResumoDoKmeans(
        amostras=list(resultado.sample_names),
        rotulos=list(resultado.labels),
        grupos=list(resultado.groups),
        k=resultado.k,
        inercia=resultado.inertia,
        semente=resultado.seed,
        n_iteracoes=resultado.n_iter,
        matriz=MATRIZ_DO_AGRUPAMENTO,
        n_amostras=resultado.n_samples,
        n_features_usadas=resultado.n_features_used,
        n_features_totais=resultado.n_features_total,
        amostragem=(
            f"todas as {resultado.n_features_used} features utilizáveis, sem sorteio — a partição "
            f"é sobre as {resultado.n_samples} amostras"
        ),
    )


def _resumo_do_som(resultado: SomResult) -> ResumoDoSom:
    """O mapa auto-organizável pronto para desenhar.

    A frase de amostragem é a de `_resumo_do_kmeans`: as duas famílias correm sobre a matriz
    inteira, sem recorte.
    """
    return ResumoDoSom(
        amostras=list(resultado.sample_names),
        unidades=list(resultado.units),
        grupos=list(resultado.groups),
        linhas=resultado.rows,
        colunas=resultado.cols,
        epocas=resultado.epochs,
        semente=resultado.seed,
        erro_de_quantizacao=resultado.quantization_error,
        matriz=MATRIZ_DO_AGRUPAMENTO,
        n_amostras=resultado.n_samples,
        n_features_usadas=resultado.n_features_used,
        n_features_totais=resultado.n_features_total,
        amostragem=(
            f"todas as {resultado.n_features_used} features utilizáveis, sem sorteio — o mapa é "
            f"sobre as {resultado.n_samples} amostras"
        ),
    )


# A matriz sobre a qual os classificadores correm, com constante própria pelo motivo de
# `MATRIZ_DA_ANOVA`.
MATRIZ_DO_CLASSIFICADOR: Final = "processada"


def _resumo_do_classificador(resultado: ClassifierResult) -> ResumoDoClassificador:
    """O desempenho pronto para desenhar, lido do resultado que a rota já calculou.

    Uma segunda chamada custaria minutos. O `p` de permutação e a acurácia saem juntos do mesmo
    objeto, e os dois lados do fio os exigem: nenhum classificador chega à figura sem validação.
    A frase de amostragem é montada aqui pelo motivo de `_resumo_da_correlacao`; esta família usa
    todas as features utilizáveis, sem sorteio.
    """
    descartadas = resultado.n_features_total - resultado.n_features_used
    return ResumoDoClassificador(
        kind=resultado.kind,
        nome=NOME_DO_CLASSIFICADOR[resultado.kind],
        grupos=list(resultado.groups),
        contagens=list(resultado.class_counts),
        recall=list(resultado.recall),
        balanced_accuracy=resultado.balanced_accuracy,
        accuracy=resultado.accuracy,
        macro_f1=resultado.macro_f1,
        permutation_pvalue=resultado.permutation_pvalue,
        chance_level=resultado.chance_level,
        confusao=[list(linha) for linha in resultado.confusion],
        importancias=list(resultado.importances),
        importance_kind=resultado.importance_kind,
        features=list(resultado.feature_names),
        cv_folds=resultado.cv_folds,
        n_permutations=resultado.n_permutations,
        semente=resultado.seed,
        matriz=MATRIZ_DO_CLASSIFICADOR,
        n_amostras=resultado.n_samples,
        n_features_usadas=resultado.n_features_used,
        n_features_totais=resultado.n_features_total,
        amostragem=(
            f"todas as {resultado.n_features_used} features utilizáveis, sem sorteio — "
            f"{descartadas} ficaram de fora por serem constantes ou não finitas nas "
            f"{resultado.n_samples} amostras de estudo"
        ),
    )


def _classificador_ou_422(
    matriz: RefinoMatrix,
    analise: AnalysisSpec,
    *,
    kind: ClassifierKind,
    seed: int,
    cancelado: CancelCheck,
) -> ResumoDoClassificador:
    """O desempenho, ou o `422` que devolve o pedido insatisfazível a quem o fez.

    'Não pedi' é `None`; 'pedi e não cabe' é `422` com os números, como em `_kmeans_ou_422`. A
    causa comum é `classifier_folds` acima da menor classe, e um `None` mudo deixaria a pessoa
    esperando minutos por uma figura que não vem.
    """
    try:
        return _resumo_do_classificador(
            classify(matriz, analise, kind=kind, seed=seed, cancelled=cancelado)
        )
    except ClassificacaoImpossivelError as erro:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(erro)) from erro


def _a_populacao_sustenta_a_rede(matriz: RefinoMatrix) -> bool:
    """Se a matriz tem amostras de estudo e features que cheguem para a rede.

    Pergunta antes pelo motivo de `_a_populacao_sustenta_a_anova`. `MIN_AMOSTRAS_PARA_DSPC`, o
    número de dobras da validação cruzada que escolhe a penalidade, vem importado do núcleo. O piso
    de features é dois porque uma correlação parcial precisa de um par.
    """
    if int(matriz.sample_mask.sum()) < MIN_AMOSTRAS_PARA_DSPC:
        return False
    utilizaveis, _constantes, _nao_finitas = column_usability(matriz.values[matriz.sample_mask, :])
    return int(utilizaveis.sum()) >= 2


def _resumo_da_rede(resultado: DspcResult, *, amostrada: bool, seed: int) -> ResumoDaRede:
    """A rede pronta para desenhar.

    A frase de `metodo` nomeia os dois passos: sem a correção de viés o estimador é só o graphical
    lasso, e o resultado não pode usar o nome do método.
    """
    return ResumoDaRede(
        features=list(resultado.features),
        arestas=[
            ArestaDaRede(
                feature_a=aresta.feature_a,
                feature_b=aresta.feature_b,
                correlacao_parcial=aresta.partial_correlation,
                estatistica=aresta.statistic,
                pvalue=aresta.pvalue,
                qvalue_fdr=aresta.qvalue_fdr,
            )
            for aresta in resultado.edges
        ],
        # `n_edges` do núcleo, e não `len()` da lista acima: o núcleo conta por caminho próprio, e
        # `tests/unit/test_dspc.py` afirma a igualdade entre os dois.
        n_arestas=resultado.n_edges,
        n_pares_testados=resultado.n_pairs_tested,
        alfa=resultado.alpha,
        penalidade=resultado.penalty,
        n_amostras_usadas=resultado.n_samples_used,
        n_features_usadas=resultado.n_features_used,
        n_features_totais=resultado.n_features_total,
        n_features_descartadas_constantes=resultado.n_features_dropped_constant,
        n_features_descartadas_nao_finitas=resultado.n_features_dropped_non_finite,
        metodo=(
            "graphical lasso com penalidade por validação cruzada, mais a correção de viés de "
            "Jankova e van de Geer (2015) — o passo DSPC de Basu et al. (2017)"
        ),
        amostragem=(
            f"{resultado.n_features_used} features sorteadas de {resultado.n_features_total} "
            f"com semente {seed}"
            if amostrada
            else f"todas as {resultado.n_features_used} features utilizáveis, sem sorteio"
        ),
    )


def _rede_ou_422(
    matriz: RefinoMatrix, analise: AnalysisSpec, *, seed: int, cancelado: CancelCheck
) -> ResumoDaRede:
    """A rede, ou o `422` que devolve o pedido insatisfazível a quem o fez.

    Recorte acima do teto é `422`, pelo motivo de `_kmeans_ou_422`. Hoje o teto não dispara por
    esta rota: `TETO_DE_FEATURES_PARA_COMPUTAR` é `None` e o recorte daqui já limita a
    `TETO_DE_FEATURES_PARA_DESENHAR`. A tradução fica pronta para quando o teto do núcleo existir.
    """
    try:
        colunas, amostrada = _sorteia(
            matriz.n_features, teto=TETO_DE_FEATURES_PARA_DESENHAR, seed=seed
        )
        return _resumo_da_rede(
            dspc_network(matriz, analise, colunas=colunas, cancelled=cancelado),
            amostrada=amostrada,
            seed=seed,
        )
    except (RedeAcimaDoTetoError, RedeSemPopulacaoError) as erro:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(erro)) from erro


def _kmeans_ou_422(
    matriz: RefinoMatrix, analise: AnalysisSpec, *, seed: int, cancelado: CancelCheck
) -> ResumoDoKmeans:
    """A partição, ou o `422` que devolve o pedido insatisfazível a quem o fez.

    `k` acima do número de amostras é `422`, não `None`. 'Não pedi' e 'pedi e não cabe' pedem
    respostas opostas: o primeiro é silêncio legítimo, o segundo volta com os dois números, para
    ninguém ficar esperando uma figura que não vem. É o desenho de `_busca_ou_nada`.
    """
    try:
        return _resumo_do_kmeans(kmeans_clustering(matriz, analise, seed=seed, cancelled=cancelado))
    except ParticaoImpossivelError as erro:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(erro)) from erro


def corre_as_analises(
    analise: AnalysisSpec,
    *,
    matriz: RefinoMatrix,
    imputadas: np.ndarray,
    seed: int,
    marca: Marca,
    cancelado: CancelCheck,
) -> ResultadosDasAnalises:
    """Corre as famílias de análise pedidas, marcando o progresso uma vez por família.

    O laço percorre `familias_pedidas`, a mesma função que `etapas_das_analises` usa para compor a
    tupla de etapas, então avisos e etapas não divergem. `matriz` e `imputadas` não têm padrão: um
    `None` deixaria os testes de progresso contando avisos sobre um ramo que não calcula nada.
    `imputadas` é a máscara das células preenchidas pela imputação; sem ela, os empates que a
    imputação cria entrariam no teste de normalidade e o fariam rejeitar variáveis normais.

    Calculam aqui sete das dez famílias de `FAMILIAS_DE_ANALISE`. A correlação e os classificadores
    produzem dois resumos cada, daí os doze campos de `ResultadosDasAnalises`. PERMANOVA, PLS-DA
    multiclasse e agrupamento hierárquico ainda completam a etapa sem calcular; a PERMANOVA é
    calculada em `POST /matrizes/diagnosticar`, e desmarcá-la só tira a etapa da barra. O
    agrupamento hierárquico e o K-means/SOM vêm do mesmo módulo, mas são famílias diferentes.

    A matriz de correlação é quadrática nas features, então é sorteada por `_sorteia` até
    `TETO_DE_FEATURES_PARA_DESENHAR`, com `PipelineSpec.seed`. A busca por padrão corre sobre todas.

    A escolha entre ANOVA e Kruskal-Wallis por normalidade, prevista em `AnalysisSpec`, ainda não
    tem consumidor: falta um decisor para N grupos, e corre sempre a ANOVA paramétrica, como diz
    `ResumoDaAnova.metodo`. A família de normalidade daqui é só diagnóstico. O nome daquele campo
    não aparece nesta docstring de propósito: `tests/unit/test_anova.py` procura o identificador
    por texto em `src/refino/` e tomaria a menção por uso.

    O cancelamento é conferido a cada família, porque cada uma pode custar minutos.
    """
    anova: ResumoDaAnova | None = None
    correlacao: ResumoDaCorrelacao | None = None
    busca: ResumoDaBuscaPorPadrao | None = None
    kmeans: ResumoDoKmeans | None = None
    som: ResumoDoSom | None = None
    rede: ResumoDaRede | None = None
    normalidade: ResumoDaNormalidade | None = None
    classificadores: dict[ClassifierKind, ResumoDoClassificador] = {}
    for familia in familias_pedidas(analise):
        raise_if_cancelled(cancelado, f"executar: antes de {familia.etapa}")
        # O cálculo de cada família vem antes do aviso: o aviso fecha a etapa e a duração dela,
        # e um cálculo depois dele teria o tempo atribuído à etapa seguinte. O despacho é por
        # `familia.campo`, e não pela posição, porque a ordem de `FAMILIAS_DE_ANALISE` é da tela.
        if familia.campo == "anova" and _a_populacao_sustenta_a_anova(matriz):
            anova = _resumo_da_anova(
                one_way_anova(matriz, post_hoc=analise.post_hoc, cancelled=cancelado)
            )
        elif familia.campo == "correlation" and _a_populacao_sustenta_a_correlacao(matriz):
            colunas, amostrada = _sorteia(
                matriz.n_features, teto=TETO_DE_FEATURES_PARA_DESENHAR, seed=seed
            )
            correlacao = _resumo_da_correlacao(
                feature_correlation(matriz, analise, colunas=colunas, cancelled=cancelado),
                amostrada=amostrada,
                seed=seed,
            )
            busca = _busca_ou_nada(matriz, analise, cancelado=cancelado)
        elif familia.campo == "kmeans" and _a_populacao_sustenta_o_agrupamento(matriz):
            kmeans = _kmeans_ou_422(matriz, analise, seed=seed, cancelado=cancelado)
        elif familia.campo == "som" and _a_populacao_sustenta_o_agrupamento(matriz):
            som = _resumo_do_som(
                self_organizing_map(matriz, analise, seed=seed, cancelled=cancelado)
            )
        elif familia.campo == "dspc" and _a_populacao_sustenta_a_rede(matriz):
            rede = _rede_ou_422(matriz, analise, seed=seed, cancelado=cancelado)
        elif familia.campo == "normality":
            # Sem guarda de população: o núcleo não levanta por ela. Grupo pequeno e variável
            # constante viram motivos por variável, e sem amostra de estudo o resultado vem sem
            # grupos.
            normalidade = _resumo_da_normalidade(normality_by_group(matriz, imputadas))
        elif familia.campo == "classifiers":
            # Um aviso para a família e um cálculo por classificador: `classifiers` é uma tupla
            # com uma só entrada em `FAMILIAS_DE_ANALISE`, e a barra anuncia uma etapa.
            #
            # Sem guarda de população: `classify` levanta `ClassificacaoImpossivelError`, uma
            # classe própria que não esconde `ValueError` de defeito, e a recusa volta como `422`.
            for kind in analise.classifiers:
                classificadores[kind] = _classificador_ou_422(
                    matriz, analise, kind=kind, seed=seed, cancelado=cancelado
                )
        marca()
    return ResultadosDasAnalises(
        anova=anova,
        correlacao=correlacao,
        busca_por_padrao=busca,
        kmeans=kmeans,
        som=som,
        rede=rede,
        random_forest=classificadores.get("random_forest"),
        svm=classificadores.get("svm"),
        normalidade=normalidade,
    )


def _a_populacao_sustenta_a_correlacao(matriz: RefinoMatrix) -> bool:
    """Se a matriz tem amostras de estudo e features que cheguem para correlacionar.

    Pergunta antes pelo motivo de `_a_populacao_sustenta_a_anova`, com o limiar de amostras
    importado do núcleo. O predicado repete o de `_observacoes_de_estudo`, que é privado.
    """
    if int(matriz.sample_mask.sum()) < MIN_AMOSTRAS_PARA_CORRELACAO:
        return False
    utilizaveis, _constantes, _nao_finitas = column_usability(matriz.values[matriz.sample_mask, :])
    return int(utilizaveis.sum()) >= 2


def _busca_ou_nada(
    matriz: RefinoMatrix, analise: AnalysisSpec, *, cancelado: CancelCheck
) -> ResumoDaBuscaPorPadrao | None:
    """A busca por padrão, quando a referência foi declarada. `None` quando não foi.

    `pattern_reference` ausente é 'não pedi': pedir o mapa sem a busca é o caso comum. Referência
    inválida é `422`, com a lista do que existe, porque 'escrevi o nome errado' pede outra resposta
    e `None` deixaria a pessoa esperando uma figura que não vem.
    """
    if analise.pattern_reference is None:
        return None
    try:
        return _resumo_da_busca_por_padrao(pattern_search(matriz, analise, cancelled=cancelado))
    except ReferenciaDesconhecidaError as erro:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(erro)) from erro


def compoe_campos_da_chave(
    spec: PipelineSpec,
    *,
    analise: AnalysisSpec,
    estudo: str,
    apresentacao: str,
    campos_sobrescritos: list[str],
    proposta_json: str | None,
    sem_proposta: bool,
    group_a: str | None,
    group_b: str | None,
) -> dict[str, Any]:
    """Os campos que identificam um trabalho de `/execucoes`.

    É função importável para os testes conferirem a chave real, e não uma cópia. O mesmo
    dicionário é a chave que a resposta carrega e a chave pela qual `_despacha` deduplica. Sete
    campos não vêm de dataclass nenhum, e o teste que confere a chave os lista com o motivo de cada
    um. O contrato de análise entra à parte porque não é campo de `PipelineSpec`; fora da chave,
    mudar um parâmetro de análise deixaria na tela a figura da execução anterior com o rótulo da
    nova.
    """
    do_pipeline = _spec_para_json(spec)
    da_analise = analise_para_json(analise)

    # Os dois dicionários são achatados no mesmo espaço de chaves; um nome repetido faria um
    # sobrescrever o outro em silêncio, e a chave ignoraria um dos eixos.
    repetidos = sorted(set(do_pipeline) & set(da_analise))
    if repetidos:  # pragma: no cover - defensivo: hoje os dois conjuntos são disjuntos
        raise RuntimeError(
            f"`PipelineSpec` e `AnalysisSpec` partilham nome de campo: {repetidos}. Achatados na "
            "mesma chave, um sobrescreveria o outro em silêncio e a invalidação passaria a "
            "ignorar um dos eixos. Renomeie um dos lados."
        )

    return {
        **do_pipeline,
        **da_analise,
        "estudo": estudo,
        "apresentacao": apresentacao,
        "campos_sobrescritos": campos_sobrescritos,
        "proposta_json": proposta_json,
        "sem_proposta": sem_proposta,
        "group_a": group_a,
        "group_b": group_b,
    }


def _lotes(
    matrix: RefinoMatrix, n_batches: int, cortes: Sequence[int] | None = None
) -> np.ndarray | None:
    """Vetor de lote por amostra, ou `None` quando é lote único.

    `None` e não vetor de zeros: o corretor distingue 'sem declaração de lote' de 'um lote
    declarado', e passar o segundo onde o pedido é o primeiro muda o caminho de código. Mesma
    forma de `_batches_for` do executor do multiverso.

    Delega em `domain/lotes.py::lotes_declarados`: sem `cortes`, blocos iguais; com `cortes`, as
    fronteiras que o diagnóstico propôs. A janela conta os QCs de cada lote com a mesma função, e
    por isso a partição que ela mostra é a que corrige.
    """
    return lotes_declarados(matrix.injection_orders, n_batches, cortes)


def _cortes_ou_422(texto: str | None, n_batches: int) -> tuple[int, ...] | None:
    """Os cortes de lote enviados no formulário, como lista JSON de ordens de injeção.

    Vazio ou ausente é `None`, que dá blocos iguais. A forma é conferida aqui, na fronteira; se
    cada corte cai dentro da corrida, `_lotes_ou_422` confere depois de a matriz ser lida.
    """
    if texto is None or not texto.strip():
        return None
    try:
        valores = json.loads(texto)
        if not isinstance(valores, list):
            raise ValueError(f"esperava uma lista JSON de inteiros; recebi {texto!r}")
        return confere_cortes(valores, n_batches)
    except (json.JSONDecodeError, ValueError) as erro:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT, f"`batch_cuts` inválido: {erro}"
        ) from erro


def _lotes_ou_422(
    matrix: RefinoMatrix, n_batches: int, cortes: Sequence[int] | None
) -> np.ndarray | None:
    """`_lotes`, com o corte além da última injeção traduzido em `422` no disparo.

    A partição é calculada antes de o trabalho ir para a thread porque tudo o que produz `4xx`
    acontece no disparo: um corte fora da corrida é defeito do pedido, não falha do cálculo.
    """
    try:
        return _lotes(matrix, n_batches, cortes)
    except ValueError as erro:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(erro)) from erro


# ---------------------------------------------------------------------------
# Multiverso com progresso incremental
# ---------------------------------------------------------------------------

# Um registro por processo, e não por requisição: o `POST` dispara e devolve, e o `GET` que lê
# o progresso é outra requisição.
REGISTRO = RegistroDeProgresso()

# Teto de execuções terminadas retidas. Cada uma guarda a conclusão, com uma linha por feature
# (cerca de 160 bytes cada no JSON servido, e o tamanho não cresce com o número de células).
# O valor é o de `LIMITE_DE_TRABALHOS_RETIDOS`, com constante própria porque a população é outra:
# 32 execuções do maior modo somam uns 30 MB. Só execuções terminadas são podadas, e quem pede
# uma podada recebe um `404` que diz isso.
LIMITE_DE_MULTIVERSOS_RETIDOS: Final[int] = 32

# Os padrões de dois eixos derivam de `GridDefinition`. `Form()` exige um padrão constante, e uma
# lista escrita à mão aqui divergiria da grade, que é de onde os outros eixos herdam.
_GRADE_PADRAO = GridDefinition()
PADRAO_CORRECTIONS = json.dumps(list(_GRADE_PADRAO.corrections))
PADRAO_BATCH_OPTIONS = json.dumps(list(_GRADE_PADRAO.batch_options))


@router.post("/multiversos", response_model=MultiversoAceito, status_code=status.HTTP_202_ACCEPTED)
def dispara_multiverso(
    measurements: Annotated[UploadFile, File()],
    sample_list: Annotated[UploadFile, File()],
    mode: Annotated[str, Form()],
    orientation: Annotated[str, Form()] = "features-in-rows",
    corrections: Annotated[str, Form()] = PADRAO_CORRECTIONS,
    batch_options: Annotated[str, Form()] = PADRAO_BATCH_OPTIONS,
    rsd_cutoffs: Annotated[str | None, Form()] = None,
    missing_thresholds: Annotated[str | None, Form()] = None,
    imputations: Annotated[str | None, Form()] = None,
    sample_norms: Annotated[str | None, Form()] = None,
    transformations: Annotated[str | None, Form()] = None,
    scalings: Annotated[str | None, Form()] = None,
    # O padrão de `corrector_jobs` difere do da biblioteca de propósito. Lá ele é 1 porque quem
    # não escolhe é a suíte de testes, com grades pequenas; aqui quem chama está esperando a grade
    # na tela, e `-1` encurta a espera. O `1` continua pedível para medir custo sem paralelismo.
    # `n_jobs` alcança as folhas, e paralelizar só as folhas com a raiz em série rende pouco.
    n_jobs: Annotated[int, Form()] = 1,
    corrector_jobs: Annotated[int, Form()] = -1,
    group_a: Annotated[str | None, Form()] = None,
    group_b: Annotated[str | None, Form()] = None,
    estudo: Annotated[str, Form()] = "sem_identificacao",
) -> MultiversoAceito:
    """Dispara o multiverso numa thread e devolve `202` com o identificador.

    Recebe os dois arquivos, o modo e, por eixo, a lista JSON de valores a cruzar; eixo omitido
    herda `GridDefinition`, e o padrão é a grade completa. O `serrf` domina o custo dela. A rota
    não promete tempo: garante progresso observável em `GET /multiversos/{id}` e estado terminal
    nomeado.

    `group_a` e `group_b` declaram o contraste sobre o qual a conclusão é medida. São opcionais
    porque qual par interessa é pergunta de quem analisa; sem eles a execução só acompanha o
    progresso, e `GET /multiversos/{id}/resultado` responde `409` dizendo que nenhum foi declarado.

    Recusa com `422`, antes de aceitar: arquivo ilegível, eixo inválido, grupo que a matriz não
    tem, e `n_jobs` ou `corrector_jobs` iguais a zero. Assim o erro não aparece minutos depois,
    com a grade já paga.
    """
    bruto_m = measurements.file.read()
    bruto_l = sample_list.file.read()
    sample_list.file.seek(0)
    matriz = _ingere_ou_422(measurements, sample_list, mode, orientation, conteudo_m=bruto_m)
    grade = _grade_ou_422(
        corrections,
        batch_options,
        {
            "rsd_cutoffs": rsd_cutoffs,
            "missing_thresholds": missing_thresholds,
            "imputations": imputations,
            "sample_norms": sample_norms,
            "transformations": transformations,
            "scalings": scalings,
        },
    )
    grupos = _grupos_ou_422(matriz, group_a, group_b)
    # O `joblib` aceita qualquer inteiro diferente de zero (`-1` todos os núcleos, `-2` todos
    # menos um) e recusa o zero com um `ValueError` que só apareceria depois do `202`, na thread.
    # A faixa válida tem um buraco no meio e não cabe em `ge`/`le`, por isso a guarda explícita.
    for nome, valor in (("n_jobs", n_jobs), ("corrector_jobs", corrector_jobs)):
        if valor == 0:
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_CONTENT,
                f"`{nome}` = 0 não tem significado: 1 é série, -1 são todos os núcleos, e qualquer "
                "outro inteiro diferente de zero é aceito",
            )

    # A poda vem antes de abrir: cada execução terminada guarda uma linha por feature, e um
    # serviço de vida longa acumularia sem teto.
    REGISTRO.esquece_terminados_alem_de(LIMITE_DE_MULTIVERSOS_RETIDOS)
    identificador = REGISTRO.abre(grade.size())
    gravacao = _abre_no_disco(
        identificador=identificador,
        estudo=estudo,
        mode=mode,
        bruto_m=bruto_m,
        bruto_l=bruto_l,
        matriz=matriz,
        n_specs=grade.size(),
        grupos=grupos,
    )
    # A capacidade do processo baixa os dois eixos. No executável congelado, se o interpretador
    # que o `loky` usaria não está disponível, `spawn._python_exe` continua a ser o `refino.exe`,
    # e pedir `-1` relança o produto inteiro em processos com janela. A capacidade só baixa:
    # quem pede `n_jobs=1` quer o regime sequencial, e a medição de custo depende disso.
    if regime_do_processo() == SERIE:
        n_jobs = SERIE
        corrector_jobs = SERIE
    threading.Thread(
        target=executa_registrando,
        args=(matriz, grade, REGISTRO, identificador),
        kwargs={
            "n_jobs": n_jobs,
            "corrector_jobs": corrector_jobs,
            "grupos": grupos,
            # O gancho só é ligado quando a abertura gravou: sem linha aberta, `fecha_multiverso`
            # devolveria `False` a cada desfecho, sem ninguém para ler.
            "ao_terminar": None if gravacao.motivo else _fecha_no_disco,
        },
        daemon=True,
    ).start()

    return MultiversoAceito(
        identificador=identificador,
        total=grade.size(),
        persistido=gravacao.motivo is None,
        motivo_nao_gravado=gravacao.motivo,
    )


@dataclass(frozen=True)
class _Gravacao:
    """O desfecho da tentativa de abrir a linha em disco; `motivo is None` significa que gravou.

    É um tipo, e não um `str | None` solto, para a leitura no ponto de uso ser
    `gravacao.motivo`, e não um `if not motivo` em que um `str` vazio também passaria por 'gravou'.
    """

    motivo: str | None = None


def _abre_no_disco(
    *,
    identificador: str,
    estudo: str,
    mode: str,
    bruto_m: bytes,
    bruto_l: bytes,
    matriz: RefinoMatrix,
    n_specs: int,
    grupos: tuple[str, str] | None,
) -> _Gravacao:
    """Grava a linha de abertura. Uma falha aqui não impede a execução.

    Recusar o disparo com `503` quando o banco não sobe trocaria a perda do histórico pela perda
    da grade inteira. Mas a falha não é silenciosa: o motivo volta no corpo do `202` e a janela o
    mostra, para ninguém contar com um histórico que não existe.

    O `except` largo é de propósito. `_repositorio` já traduz as causas conhecidas em `503`; para
    as outras o desfecho certo é o mesmo, correr a grade e dizer que não gravou.
    """
    try:
        repo = MultiversoRepository(_motor_ou_503())
        repo.abre_multiverso(
            identificador=identificador,
            estudo=estudo,
            mode=mode,
            checksum_entrada=checksum_do_par(bruto_m, bruto_l),
            n_samples=matriz.n_samples,
            n_features=matriz.n_features,
            amostras=_amostras_para_gravar(matriz),
            n_specs=n_specs,
            grupos=grupos,
        )
    except HTTPException as erro:
        # `_repositorio` levanta `503` com a mensagem pronta e sem a credencial: a URL do Postgres
        # leva a senha, e este texto chega à tela por `interface/cliente.py`.
        return _Gravacao(motivo=str(erro.detail))
    except Exception as erro:
        return _Gravacao(motivo=f"{type(erro).__name__}: {erro}")
    return _Gravacao()


def _fecha_no_disco(final: ProgressoDaExecucao) -> None:
    """Grava o estado terminal e a conclusão. A thread a chama nos três desfechos.

    A tradução de `ProgressoDaExecucao` para o que o banco guarda fica aqui, na camada que conhece
    os dois lados: `api/progresso.py` e `persistence/` não se importam. `resultado` pode ser `None`
    com estado `concluido`: sem par de grupos não há conclusão, a tabela de estabilidade fica
    vazia, e o histórico diz que nenhum par foi declarado.
    """
    resultado = final.resultado
    estabilidade: tuple[EstabilidadeParaGravar, ...] = ()
    if isinstance(resultado, ResultadoDoMultiverso):
        estabilidade = tuple(
            EstabilidadeParaGravar(
                feature=m.feature,
                frac_significant=m.frac_significant,
                n_cells=m.n_cells,
                median_qvalue=m.median_qvalue,
                median_rank=m.median_rank,
                rank_iqr=m.rank_iqr,
                sign_consistency=m.sign_consistency,
            )
            for m in resultado.estabilidade
        )
    gravou = MultiversoRepository(_motor_ou_503()).fecha_multiverso(
        final.identificador,
        estado=final.estado,
        n_feitas=final.feito,
        celula_que_falhou=final.celula_que_falhou,
        erro=final.erro,
        estabilidade=estabilidade,
    )
    if not gravou:
        # `False` significa que a linha de abertura já não existe, por exemplo podada enquanto a
        # grade corria. `_avisa_que_terminou` só relata quando o gancho levanta, então o aviso
        # sai aqui, pelo mesmo canal; no executável, é o log que `desktop/arranque.py` mantém.
        print(
            f"[refino] a execucao {final.identificador} terminou como {final.estado} e a linha "
            f"dela ja nao existia no banco: {len(estabilidade)} linha(s) de conclusao NAO foram "
            "gravadas. A execucao correu; o que se perdeu foi o registro dela",
            file=sys.stderr,
        )


def inventario_de_grupos(matriz: RefinoMatrix) -> list[GrupoComContagem]:
    """Os grupos que fazem sentido contrastar, com quantas amostras cada um tem.

    Ordenados por nome e sem o QC. A contagem vai ao lado do nome porque é ela que diz se a
    pergunta tem poder: uma ANOVA com um grupo de 13 amostras não é a mesma pergunta que com 30.
    É a fonte única de 'quais grupos existem'; `grupos_contrastaveis` é a projeção dela, para a
    interface e as guardas não divergirem sobre o QC.

    O QC fica de fora porque é a régua, não um grupo: é uma mistura de todas as amostras, e
    contrastá-lo com um diagnóstico produz número sem significado. O filtro usa `qc_mask`, e não a
    string 'QC', porque o rótulo vem da linha `LABEL` do arquivo e o domínio é aberto. Amostras de
    validação não são filtradas: `sample_mask` não é o complemento de `qc_mask`, porque
    `SampleType` tem três valores.
    """
    contagem = Counter(
        grupo for grupo, e_qc in zip(matriz.groups, matriz.qc_mask, strict=True) if not e_qc
    )
    return [GrupoComContagem(nome=nome, n=contagem[nome]) for nome in sorted(contagem)]


def grupos_contrastaveis(matriz: RefinoMatrix) -> list[str]:
    """Só os nomes do inventário, na mesma ordem.

    As guardas perguntam pertinência, não tamanho. Derivar do inventário, em vez de passar de novo
    pela matriz, evita duas expressões para 'quais grupos existem', uma delas sem o filtro do QC.
    """
    return [grupo.nome for grupo in inventario_de_grupos(matriz)]


def _grupos_ou_422(
    matriz: RefinoMatrix, group_a: str | None, group_b: str | None
) -> tuple[str, str] | None:
    """O par de grupos validado, ou `None` quando nenhum foi pedido.

    Três desfechos: nenhum pedido (`None`, e a execução corre sem conclusão), par válido, e par
    recusado com `422`. Pedir só um dos dois é recusado: ignorá-lo executaria horas de grade e
    devolveria progresso a quem pediu conclusão. A guarda é de pertinência, porque o nome do grupo
    vem do arquivo e o domínio é aberto, e a mensagem lista os grupos disponíveis.
    """
    if group_a is None and group_b is None:
        return None
    if not group_a or not group_b:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            "a conclusão do teste de robustez é medida sobre um PAR de grupos, e só um foi "
            "declarado "
            f"(`group_a={group_a!r}`, `group_b={group_b!r}`). Declare os dois, ou nenhum — e sem "
            "par a execução devolve progresso e não conclusão",
        )
    inventario = inventario_de_grupos(matriz)
    disponiveis = [grupo.nome for grupo in inventario]
    # A mensagem traz o tamanho de cada grupo, que decide se o par tem poder, no formato
    # `Nome (n)` da linha do inventário na janela, para as duas listas se lerem como uma só.
    catalogo = ", ".join(f"{grupo.nome} ({grupo.n})" for grupo in inventario)
    if group_a == group_b:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            f"os dois grupos são o mesmo ({group_a!r}); não há contraste a testar. "
            f"Disponíveis nesta matriz, com o número de amostras de cada um: {catalogo}",
        )
    for nome in (group_a, group_b):
        if nome not in disponiveis:
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_CONTENT,
                f"grupo {nome!r} não existe nesta matriz. Disponíveis, com o número de amostras "
                f"de cada um: {catalogo}",
            )
    return (group_a, group_b)


# ---------------------------------------------------------------------------
# O histórico, e a ordem destas rotas importa
# ---------------------------------------------------------------------------
#
# `/multiversos/historico` é declarada antes de `/multiversos/{identificador}`. O FastAPI casa
# por ordem de registro: com o padrão variável primeiro, `GET /multiversos/historico` cairia em
# `le_progresso` e responderia um `404` plausível sobre a execução 'historico'. Há teste para isso.


@router.get("/multiversos/historico", response_model=HistoricoDeMultiversos)
def le_historico(limite: int = 50) -> HistoricoDeMultiversos:
    """Lista as execuções de multiverso gravadas em disco, da mais recente para a mais antiga.

    Devolve o que sobreviveu ao fechamento do programa: estado terminal, quanto correu do que se
    pediu, o par de grupos quando houve e a causa quando falhou. O progresso ao vivo fica em
    `GET /multiversos/{identificador}`, que lê a memória do processo, podada; o disco não é.
    `limite` volta no corpo para o cliente distinguir 'há 50' de 'há mais e foram cortadas'.

    Recusa `limite` menor que 1 com `422`. Banco indisponível é `503`, e não lista vazia, que
    diria 'nunca correu nada' sobre um histórico que só não pôde ser lido.
    """
    if limite < 1:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            f"`limite={limite}` nao devolve linha nenhuma. Um teto de zero e indistinguivel de "
            "«o historico esta vazio» para quem le a resposta; peca ao menos 1",
        )
    gravadas = MultiversoRepository(_motor_ou_503()).lista_multiversos(limite=limite)
    return HistoricoDeMultiversos(
        execucoes=tuple(_para_historico(m) for m in gravadas), limite=limite
    )


@router.get("/multiversos/historico/{identificador}", response_model=ConclusaoDoHistorico)
def le_conclusao_gravada(
    identificador: str,
    robust_frac: float | None = None,
    robust_sign: float | None = None,
    absent_frac: float | None = None,
) -> ConclusaoDoHistorico:
    """Devolve a conclusão de uma execução como ficou gravada em disco.

    Traz a linha da execução, a tabela de estabilidade e os limiares usados no veredito, que
    podem vir na consulta (`robust_frac`, `robust_sign`, `absent_frac`). Responde `404` se o
    identificador não está gravado, `422` para limiar inválido e `503` sem banco. Tabela vazia
    não é `404`: uma execução cancelada antes da conclusão existe, e o cliente precisa distingui-la
    de um identificador desconhecido.
    """
    conclusao = MultiversoRepository(_motor_ou_503()).le_multiverso(identificador)
    if conclusao is None:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            f"nao ha execucao de teste de robustez gravada com identificador {identificador!r}. "
            "Note que "
            "o historico so guarda o que COMECOU a correr, e que uma execucao pode ter sido "
            "disparada num momento em que o banco estava indisponivel — o disparo diz isso no "
            "campo `motivo_nao_gravado`",
        )
    # O veredito é recalculado, e não lido do disco. É função pura de `frac_significant` e
    # `sign_consistency`, os dois gravados, e o veredito é uma leitura: mudar os limiares
    # reclassifica também as execuções antigas. Gravado, ele congelaria o corte.
    #
    # `sign_consistency` nulo entra como `nan` de propósito: `None` é 'não medido', e
    # `nan >= limiar` é `False`, então sem consistência de sinal medida a feature não é `robust`.
    limiares = _limiares_ou_422(robust_frac, robust_sign, absent_frac)
    return ConclusaoDoHistorico(
        execucao=_para_historico(conclusao.execucao),
        estabilidade=tuple(
            EstabilidadeDeFeature(
                feature=m.feature,
                frac_significant=m.frac_significant,
                median_qvalue=m.median_qvalue,
                median_rank=m.median_rank,
                rank_iqr=m.rank_iqr,
                sign_consistency=m.sign_consistency,
                n_cells=m.n_cells,
                verdict=classifica_veredito(
                    m.frac_significant,
                    float("nan") if m.sign_consistency is None else m.sign_consistency,
                    limiares,
                ),
            )
            for m in conclusao.estabilidade
        ),
        limiares=LimiaresDoVeredito(
            robust_frac=limiares.robust_frac,
            robust_sign=limiares.robust_sign,
            absent_frac=limiares.absent_frac,
        ),
    )


def _para_historico(m: MultiversoGravado) -> MultiversoNoHistorico:
    """Converte a linha gravada no corpo da resposta, para as duas rotas do histórico.

    `n_specs` do disco vira `total` na fronteira, o mesmo nome de `MultiversoAceito` e de
    `ProgressoResposta`, para a mesma quantidade não ter dois nomes no fio.
    """
    return MultiversoNoHistorico(
        identificador=m.identificador,
        estado=m.estado,
        estudo=m.estudo,
        mode=m.mode,
        total=m.n_specs,
        n_feitas=m.n_feitas,
        iniciado_em=m.iniciado_em,
        terminado_em=m.terminado_em,
        n_features=m.n_features,
        grupos=m.grupos,
        celula_que_falhou=m.celula_que_falhou,
        erro=m.erro,
    )


@router.get("/multiversos/{identificador}", response_model=ProgressoResposta)
def le_progresso(identificador: str) -> ProgressoResposta:
    """Devolve o progresso de uma execução de multiverso; `404` se o identificador não existe.

    `404`, e não um progresso vazio em `pendente`: 'não conheço' pede corrigir o identificador, e
    'ainda não começou' pede esperar. Confundir os dois deixaria a interface esperando para sempre.
    """
    leitura = REGISTRO.le(identificador)
    if leitura is None:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            f"não há execução de teste de robustez com identificador {identificador!r}",
        )
    return _progresso_resposta(leitura)


@router.post(
    "/multiversos/{identificador}/cancelar",
    response_model=ProgressoResposta,
    status_code=status.HTTP_202_ACCEPTED,
)
def cancela_multiverso(identificador: str) -> ProgressoResposta:
    """Pede que a execução de multiverso pare; a parada é assíncrona.

    Responde `202` com o progresso lido depois do pedido (`cancelamento_pedido` verdadeiro,
    `estado` ainda `executando`), o que basta para a tela mostrar que está parando. Responde `404`
    para identificador desconhecido e `409` se a execução já terminou, com o estado na mensagem.
    Cancelar de novo enquanto ainda corre devolve `202` outra vez, porque é o que a pessoa faz
    quando a parada demora.

    É `POST` num sub-recurso, e não `DELETE`, porque o pedido é uma transição: o registro continua
    existindo para servir o estado `cancelado`. A parada é cooperativa: a rota levanta uma
    bandeira que o núcleo lê nos pontos de checagem de `domain/cancellation.py`. Com `n_jobs > 1`
    a latência é a de um lote em voo, porque a bandeira não alcança os processos do `joblib`.
    """
    anterior = REGISTRO.pede_cancelamento(identificador)
    if anterior is None:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            f"não há execução de teste de robustez com identificador {identificador!r}. "
            "Ou ele nunca "
            f"existiu, ou terminou há tempo bastante para ser esquecido: o serviço retém as "
            f"{LIMITE_DE_MULTIVERSOS_RETIDOS} execuções terminadas mais recentes",
        )
    if anterior.terminou:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            f"a execução {identificador!r} já terminou — o estado é «{anterior.estado}» — e não há "
            "o que cancelar. Se quer o resultado, peça-o; se quer outra execução, dispare-a",
        )
    depois = REGISTRO.le(identificador)
    # `le` devolve `Optional`; a alternativa cobre o mypy e a poda entre as duas leituras.
    return _progresso_resposta(depois if depois is not None else anterior)


@router.get(
    "/multiversos/{identificador}/resultado",
    responses={
        200: {
            "model": ResultadoDoMultiverso,
            "description": "A conclusão: estabilidade por feature",
        },
        202: {"model": ProgressoResposta, "description": "Ainda a correr — sinal de vida"},
    },
)
def le_resultado_do_multiverso(
    identificador: str,
    robust_frac: float | None = None,
    robust_sign: float | None = None,
    absent_frac: float | None = None,
) -> Response:
    """Devolve a conclusão do multiverso: a estabilidade de cada feature ao longo da grade.

    Os limiares do veredito podem vir na consulta (`robust_frac`, `robust_sign`, `absent_frac`).
    Os desfechos:

    - `404`: identificador desconhecido.
    - `202` com o progresso: ainda `pendente` ou `executando`; o cliente pergunta de novo.
    - `200` com o resultado: concluída com par de grupos.
    - `409`: concluída sem par declarado no disparo, ou cancelada a pedido. As duas pedem o mesmo,
      disparar de novo, e a mensagem diz qual foi.
    - `422`: limiar inválido.
    - `500`: `falhou`, ou concluída sem resultado retido; a mensagem nomeia a célula.

    Sem par declarado é `409`, e não `200` com tabela vazia, que se leria como 'nenhuma feature
    sobrevive à grade'. Não marca `reclamado`, que só ordena a poda de artefatos.
    """
    leitura = REGISTRO.le(identificador)
    if leitura is None:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            f"não há execução de teste de robustez com identificador {identificador!r}. "
            "Ou ele nunca "
            f"existiu, ou terminou há tempo bastante para ser esquecido: o serviço retém as "
            f"{LIMITE_DE_MULTIVERSOS_RETIDOS} execuções terminadas mais recentes",
        )

    if not leitura.terminou:
        return Response(
            content=_progresso_resposta(leitura).model_dump_json(),
            media_type="application/json",
            status_code=status.HTTP_202_ACCEPTED,
        )

    if leitura.estado == "falhou":
        raise HTTPException(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            f"a execução falhou na célula «{leitura.celula_que_falhou}»: {leitura.erro}",
        )

    # Antes da leitura do resultado: uma execução cancelada tem `resultado is None`, e sem este
    # ramo cairia no `409` de baixo, que diria que o disparo não declarou os grupos.
    if leitura.estado == "cancelado":
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            f"a execução {identificador!r} foi CANCELADA a seu pedido, depois de {leitura.feito} "
            f"de {leitura.total} células. Não há conclusão a servir: a estabilidade por feature é "
            "medida sobre a grade inteira, e uma grade parcial daria uma fração cujo denominador "
            "não é o que se declarou. Dispare outra vez, com uma grade menor se quiser esperar "
            "menos",
        )

    resultado = leitura.resultado
    if isinstance(resultado, ResultadoDoMultiverso):
        # Sem limiares na consulta, o corpo retido sai como está. Com eles, só o `verdict` é
        # recalculado, e `limiares` sai dos mesmos números, para a resposta não declarar um corte
        # diferente do que aplicou.
        if (robust_frac, robust_sign, absent_frac) != (None, None, None):
            limiares = _limiares_ou_422(robust_frac, robust_sign, absent_frac)
            resultado = resultado.model_copy(
                update={
                    "estabilidade": _reclassifica(resultado.estabilidade, limiares),
                    "limiares": LimiaresDoVeredito(
                        robust_frac=limiares.robust_frac,
                        robust_sign=limiares.robust_sign,
                        absent_frac=limiares.absent_frac,
                    ),
                }
            )
        return Response(content=resultado.model_dump_json(), media_type="application/json")
    if resultado is None:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            f"a execução {identificador!r} terminou e não há conclusão a servir: o disparo não "
            "declarou `group_a` e `group_b`, e a estabilidade por feature é medida sobre um par de "
            "grupos. Dispare outra vez declarando o par — os nomes de grupo estão na linha "
            "`LABEL` do seu `measurements`, que é a segunda linha do arquivo",
        )
    # Um resultado de outro tipo é defeito da ferramenta; dizê-lo é melhor que devolver `200` com
    # um corpo que ninguém sabe ler.
    raise HTTPException(
        status.HTTP_500_INTERNAL_SERVER_ERROR,
        f"a execução {identificador!r} concluiu e retém {type(resultado).__name__}, que não é "
        "conclusão de teste de robustez. É defeito da ferramenta, não do seu arquivo",
    )


def _limiares_ou_422(
    robust_frac: float | None,
    robust_sign: float | None,
    absent_frac: float | None,
) -> StabilityThresholds:
    """Os três limiares do veredito, com guarda de fronteira. Ausentes viram os padrões.

    Chegam pela consulta, e não pelo disparo, porque o veredito é uma leitura: os dois números de
    que ele depende já estão gravados, e mudar de limiar não deve custar uma grade nova.

    São frações. Sem a guarda, `7,5` faria `frac >= 7,5` falso para tudo, e a tabela sairia
    plausível e sem nenhuma feature robusta. A guarda é explícita, e não `Query(ge=0, le=1)`, para
    a mensagem dizer o que o número significa. `absent_frac` não é comparado com os outros dois:
    as condições não se ordenam entre si, e exigir `absent_frac < robust_frac` inventaria uma
    relação que o método não tem.
    """
    padrao = StabilityThresholds()
    escolhidos = {
        "robust_frac": robust_frac,
        "robust_sign": robust_sign,
        "absent_frac": absent_frac,
    }
    for nome, valor in escolhidos.items():
        if valor is None:
            continue
        if not 0.0 <= valor <= 1.0:
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_CONTENT,
                f"`{nome}={valor}` está fora de [0, 1]. Os três limiares do veredito são FRAÇÕES — "
                f"o padrão é robust_frac={padrao.robust_frac}, "
                f"robust_sign={padrao.robust_sign}, absent_frac={padrao.absent_frac}. Um valor "
                "acima de 1 não recusaria nada: ele tornaria a condição impossível, e a tabela "
                "sairia sem features robustas sem que nada dissesse porquê",
            )
    return StabilityThresholds(
        robust_frac=padrao.robust_frac if robust_frac is None else robust_frac,
        robust_sign=padrao.robust_sign if robust_sign is None else robust_sign,
        absent_frac=padrao.absent_frac if absent_frac is None else absent_frac,
    )


def _reclassifica(
    estabilidade: Sequence[EstabilidadeDeFeature], limiares: StabilityThresholds
) -> tuple[EstabilidadeDeFeature, ...]:
    """Reaplica os limiares sobre medidas já calculadas, sem tocar em nenhuma delas.

    Só o `verdict` muda: os números por feature são medições, e o veredito é um corte sobre dois
    deles. `sign_consistency` nulo entra como `nan`, e não `0.0`: `None` é 'não medido', e
    `nan >= robust_sign` é falso, então a feature não é `robust` sem que se finja ter medido zero.
    """
    return tuple(
        m.model_copy(
            update={
                "verdict": classifica_veredito(
                    m.frac_significant,
                    float("nan") if m.sign_consistency is None else m.sign_consistency,
                    limiares,
                )
            }
        )
        for m in estabilidade
    )


def _vetor_ou_422(nome: str, bruto: str) -> list[Any]:
    """Um eixo da grade como vetor JSON, ou `422` nomeando qual eixo falhou.

    O nome entra na mensagem porque a rota aceita oito eixos, e 'não é vetor' sem dizer qual
    obrigaria o cliente a adivinhar.
    """
    try:
        valor = json.loads(bruto)
    except json.JSONDecodeError as erro:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT, f"`{nome}` não é JSON: {erro}"
        ) from erro
    if not isinstance(valor, list):
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            f"`{nome}` é {type(valor).__name__} e tem de ser vetor. Uma string sozinha seria "
            "iterada como caracteres e produziria rótulos absurdos",
        )
    # Cada elemento tem de ser escalar: um vetor aninhado, como `[["none"]]`, levantaria
    # `TypeError` em `nome in UNAVAILABLE_METHODS` (lista não é hashable) e daria `500`.
    aninhados = [v for v in valor if isinstance(v, (list, dict))]
    if aninhados:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            f"`{nome}` tem elemento que não é escalar: {aninhados[0]!r}. Cada eixo é um vetor "
            "de rótulos ou de números, sem vetores dentro",
        )
    return valor


def _grade_ou_422(
    corrections: str, batch_options: str, opcionais: dict[str, str | None]
) -> GridDefinition:
    """A grade pedida, com os rótulos validados na fronteira.

    A guarda de `UNAVAILABLE_METHODS` hoje não tem membro, e fica como na CLI para o caso de algum
    método ser bloqueado. Um método indisponível é recusado, e não filtrado: filtrar executaria uma
    grade menor que a pedida e devolveria um `total` que o cliente não pediu. Eixo omitido herda
    `GridDefinition`, sem padrão local, para a grade da interface não divergir da do método.
    """
    pedidas = _vetor_ou_422("corrections", corrections)
    lotes = _vetor_ou_422("batch_options", batch_options)

    for nome in pedidas:
        if nome in UNAVAILABLE_METHODS:
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_CONTENT,
                f"`{nome}` não é selecionável no teste de robustez: "
                f"{UNAVAILABLE_METHODS[nome].reason}",
            )

    # A conversão para `int` fica dentro do `422`: fora dele, `["as_declared"]` ou `["1","dois"]`
    # dariam `500`, que a janela lê como defeito da ferramenta e não do pedido.
    try:
        lotes_inteiros = tuple(int(n) for n in lotes)
        # `int(2.5)` dá 2 sem levantar, e o número de lotes pedido mudaria em silêncio.
        if any(float(n) != i for n, i in zip(lotes, lotes_inteiros, strict=True)):
            raise ValueError("número de lotes fracionário")
    except (TypeError, ValueError) as erro:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            f"`batch_options` tem de ser uma lista de inteiros, e {lotes!r} nao e: "
            f"{type(erro).__name__}. Cada valor e um NUMERO DE LOTES a declarar — por exemplo "
            "`[1, 2]` para comparar um lote contra dois",
        ) from erro

    argumentos: dict[str, Any] = {
        "corrections": tuple(pedidas),
        "batch_options": lotes_inteiros,
    }
    for nome, bruto in opcionais.items():
        if bruto is not None:
            argumentos[nome] = tuple(_vetor_ou_422(nome, bruto))

    _valida_rotulos_da_grade(argumentos)
    try:
        grade = GridDefinition(**argumentos)
        # A grade é enumerada aqui, no disparo. `GridDefinition` confere os rótulos e pouco mais;
        # quem recusa as combinações (`imputation='none'` com normalização, corte de RSD fora da
        # faixa, zero lotes) é `PipelineSpec`, que só nasce na enumeração. Dentro do trabalho de
        # fundo, o tratador da falha enumeraria de novo para nomear a célula e o trabalho ficaria
        # `pendente` para sempre. Enumerar custa centésimos de segundo.
        for _ in grade.enumerate():
            pass
    except (InvalidLabelError, ValueError, TypeError) as erro:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(erro)) from erro
    return grade


# Eixo da grade -> tipo de vocabulário. Os eixos numéricos ficam fora: quem confere a faixa deles
# é `PipelineSpec`, na enumeração que `_grade_ou_422` faz no disparo.
VOCABULARIO_DOS_EIXOS: Final[dict[str, str]] = {
    "corrections": "Correction",
    "imputations": "Imputation",
    "sample_norms": "SampleNormalization",
    "transformations": "Transformation",
    "scalings": "Scaling",
}


def _valida_rotulos_da_grade(argumentos: dict[str, Any]) -> None:
    """Aplica `validate_label` a cada eixo de vocabulário, na fronteira.

    `GridDefinition` não valida os rótulos: eles só passam por `validate_label` dentro de
    `PipelineSpec`, que a grade constrói ao enumerar. Sem esta guarda, `corrections=["loes"]`
    receberia `202` e falharia depois, do outro lado da thread, onde ninguém vê.
    """
    for eixo, tipo in VOCABULARIO_DOS_EIXOS.items():
        for valor in argumentos.get(eixo, ()):
            try:
                validate_label(tipo, valor)
            except InvalidLabelError as erro:
                raise HTTPException(
                    status.HTTP_422_UNPROCESSABLE_CONTENT, f"`{eixo}`: {erro}"
                ) from erro


# ---------------------------------------------------------------------------
# As rotas de artefato
# ---------------------------------------------------------------------------
#
# Dão à janela os três artefatos de arquivo da CLI: `corrigir`, `processar` e `exportar`. O CSV
# não é gravado em disco no servidor; os bytes esperam no registro de trabalhos, em memória.
#
# Nenhuma delas exige o banco, como os comandos equivalentes da CLI. As três gravam o tempo que
# gastaram, para a pessoa saber quanto cada operação custa na máquina dela, mas em melhor esforço:
# o artefato sai igual com ou sem banco. `test_ca_12_11_as_rotas_de_artefato_nao_exigem_banco`
# garante isso.


# Os dois formatos da matriz corrigida. `toserrf` é a forma que alimenta o motor SERRF: três linhas
# de metadados (lote, tipo de amostra, ordem de injeção) e coluna de índice. `normalizado` é a
# forma que ele devolve: uma linha de cabeçalho, sem os metadados. Um reproduz a execução; o
# outro é o resultado.
FORMATO_PADRAO: Final = "toserrf"
FORMATO_ALTERNATIVO: Final = "normalizado"
FORMATOS_DO_ARTEFATO: Final[tuple[str, ...]] = (FORMATO_PADRAO, FORMATO_ALTERNATIVO)


def _formato_ou_422(retido: ArtefatoRetido, formato: str | None) -> ArtefatoRetido:
    """Escolhe qual dos dois arquivos servir; sem `formato`, o `toserrf`, idêntico ao da CLI.

    Valor fora do vocabulário é `422`: `normalisado`, com `s`, não pode cair num `else` e
    devolver o `toserrf` em silêncio. Pedir o alternativo a quem não o tem é `404`, porque
    `processar` e `exportar` produzem um arquivo só, e isso é propriedade da operação, não erro do
    pedido.
    """
    if formato is None:
        return retido
    if formato not in FORMATOS_DO_ARTEFATO:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            f"`formato={formato}` não é um valor válido. Os dois são: "
            f"{', '.join(FORMATOS_DO_ARTEFATO)} — o primeiro alimenta o motor SERRF e carrega "
            "lote, tipo de amostra e ordem de injeção; o segundo é a forma que o pipeline R "
            "devolve, e descarta os três",
        )
    if formato == FORMATO_PADRAO:
        return retido
    if retido.alternativo is None:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            f"este trabalho não retém a forma `{FORMATO_ALTERNATIVO}`: ela é produzida por "
            "`POST /matrizes/corrigir`, e as outras operações devolvem um arquivo só",
        )
    return retido.alternativo


def _artefato_retido(
    nome: str, escreve: Callable[[Path], RelatorioDeExportacao | None]
) -> ArtefatoRetido:
    """Deixa o escritor produzir o arquivo num temporário e devolve os bytes dele.

    O escritor devolve o relatório do corte quando há um (`write_metaboanalyst`) e `None` quando
    não corta nada; pelo retorno, e não por variável capturada, o tipo impede a rota de perdê-lo.
    Ler o arquivo de volta, em vez de serializar aqui, garante os mesmos bytes que a CLI escreve.
    Nada fica em caminho durável: os bytes esperam no registro em memória, com teto de
    `LIMITE_DE_TRABALHOS_RETIDOS`.
    """
    with tempfile.TemporaryDirectory(prefix="refino-artefato-") as temporario:
        destino = Path(temporario) / nome
        relatorio = escreve(destino)
        return ArtefatoRetido(nome=nome, conteudo=destino.read_bytes(), relatorio=relatorio)


def _relatorio_para_json(relatorio: MetaboAnalystExport) -> RelatorioDeExportacao:
    """O relatório do escritor no esquema da resposta.

    A ordem vem pronta de `detalhe_das_removidas` (as indefinidas primeiro, depois as de RSD alto
    da pior para a melhor) e não é refeita aqui, para não divergir da do núcleo.
    """
    return RelatorioDeExportacao(
        mantidas=len(relatorio.kept),
        removidas=tuple(
            FeatureRemovidaResposta(nome=item.nome, causa=item.causa, rsd=item.rsd)
            for item in relatorio.detalhe_das_removidas
        ),
        zeros_como_vazias=relatorio.zeros_written_as_empty,
        nomes_sanitizados=relatorio.renamed,
    )


def _resposta_csv_de_bytes(conteudo: bytes, nome: str) -> Response:
    """Os bytes retidos como corpo `text/csv`, com o nome do arquivo no cabeçalho.

    O `Content-Disposition` leva o nome, com o `mode` dentro, porque é ele que o navegador usa:
    sem nome, o artefato sairia como `identificador.csv`, com o rótulo do trabalho em vez do
    rótulo do estudo.
    """
    return Response(
        content=conteudo,
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{nome}"'},
    )


def _grava_tempo_do_artefato(
    *,
    operacao: str,
    mode: str,
    matriz: RefinoMatrix,
    checksum: str,
    parametros: dict[str, Any],
    iniciado_em: datetime,
    status: str,
    etapas: dict[str, float],
) -> None:
    """Grava a linha de `runs` do artefato. Uma falha aqui não estraga o artefato.

    A escrita acontece no fim, depois de o cálculo correr, e um `serrf` no maior modo leva mais de
    vinte minutos: perder o artefato porque o banco não respondeu trocaria o resultado pelo
    registro do tempo. A falha não é silenciosa: como o `202` já foi respondido, o motivo vai para
    a saída de erro, que no executável é o log que `desktop/arranque.py` mantém.

    O `except` largo é de propósito, como em `_abre_no_disco`: `_motor_ou_503` já traduz as causas
    conhecidas, sem a credencial, e para as outras o desfecho certo é o mesmo.
    """
    try:
        _repositorio().grava_artefato(
            operacao=operacao,
            estudo=ESTUDO_SEM_IDENTIFICACAO,
            mode=mode,
            checksum_entrada=checksum,
            n_samples=matriz.n_samples,
            n_features=matriz.n_features,
            amostras=_amostras_para_gravar(matriz),
            parametros=parametros,
            iniciado_em=iniciado_em,
            status=status,
            etapas=etapas,
        )
    except HTTPException as erro:
        motivo = str(erro.detail)
    except Exception as erro:
        motivo = f"{type(erro).__name__}: {erro}"
    else:
        return
    print(
        f"[refino] {operacao} terminou como {status} e o tempo dela NAO foi gravado: {motivo}. "
        "O artefato foi produzido; o que se perdeu foi o registro de quanto ele custou",
        file=sys.stderr,
    )


def _com_registro_de_tempo(
    operacao: str,
    *,
    mode: str,
    matriz: RefinoMatrix,
    checksum: str,
    parametros: dict[str, Any],
    etapas: tuple[str, ...],
    trabalho: Trabalho,
) -> Trabalho:
    """Envolve o trabalho de uma rota de artefato para que ele deixe o tempo em disco.

    Um envoltório, e não um `try` em cada rota, porque os três desfechos (concluir, falhar,
    cancelar) têm de gravar, e uma cópia que divergisse deixaria de registrar sem ninguém notar.

    O relógio parte quando a thread de trabalho começa, como em `POST /execucoes`, e deixa de fora
    o upload e a ingestão: a duração mede o custo do cálculo, que depende do processador, e não a
    demora da chamada, que depende do arquivo e do disco.

    Não há linha quando o trabalho é reclamado (a mesma entrada já está sendo calculada e nenhum
    cálculo novo corre) nem quando o processo morre no meio, porque a escrita é uma só, no fim.
    """

    def grava(status: str, iniciado_em: datetime, duracoes: dict[str, float]) -> None:
        _grava_tempo_do_artefato(
            operacao=operacao,
            mode=mode,
            matriz=matriz,
            checksum=checksum,
            parametros=parametros,
            iniciado_em=iniciado_em,
            status=status,
            etapas=duracoes,
        )

    def registrado(marca: Marca, cancelado: CancelCheck) -> object:
        comecou_em = datetime.now(UTC)
        duracoes: dict[str, float] = {}
        anterior = comecou_em

        def marca_e_cronometra() -> None:
            """Fecha o relógio da etapa que acabou, e depois avisa o progresso.

            Envolve a `marca` de `_executa_trabalho`, que alimenta a barra da tela, para o relógio
            andar no mesmo instante que o contador. O nome da etapa sai de `etapas` pela contagem,
            como em `_etapa_na_posicao`: o aviso `k` fecha a etapa `k`. Um aviso além das etapas
            declaradas perde o tempo e não levanta, porque derrubar o cálculo pelo cronômetro
            trocaria o artefato por um registro.
            """
            nonlocal anterior
            agora = datetime.now(UTC)
            indice = len(duracoes)
            if indice < len(etapas):
                duracoes[etapas[indice]] = (agora - anterior).total_seconds()
            anterior = agora
            marca()

        try:
            resultado = trabalho(marca_e_cronometra, cancelado)
        except ExecutionCancelledError:
            # Antes do `except Exception`, como em `_executa_trabalho`: senão o cancelamento seria
            # gravado como falha da ferramenta.
            grava("cancelado", comecou_em, duracoes)
            raise
        except Exception:
            # Inclui as recusas legítimas (o `409` de matriz vazia, o `422` do escritor). O
            # `status` permite tirar a duração de uma falha das comparações de custo.
            grava("falhou", comecou_em, duracoes)
            raise
        grava("ok", comecou_em, duracoes)
        return resultado

    return registrado


# ---------------------------------------------------------------------------
# A memória dos Palpites medidos
# ---------------------------------------------------------------------------

# Cada Palpite guardado é um punhado de floats; o limite só impede a lista de crescer sem fim num
# servidor aberto o dia todo. Um estudo cabe com folga: quatro modos e alguns lotes por modo.
LIMITE_DE_PALPITES_RETIDOS: Final = 32

_PALPITES: OrderedDict[str, PalpiteMedido] = OrderedDict()
_TRAVA_DOS_PALPITES = threading.Lock()


def _chave_do_palpite(
    matriz: RefinoMatrix, n_batches: int, cortes: Sequence[int] | None, seed: int = 42
) -> str:
    """A identidade de um Palpite medido: a matriz ingerida, as 200 features e os lotes.

    É a mesma chave que `_despacha` dá ao trabalho de `POST /matrizes/palpite`, e é isso que deixa
    a delegação do 'Automático' achar o Palpite que a tela mediu. `paralelo` fica de fora porque
    muda o tempo, e não o número.
    """
    return _chave_do_trabalho("palpite", matriz, _campos_do_palpite(n_batches, cortes, seed))


def _campos_do_palpite(
    n_batches: int, cortes: Sequence[int] | None, seed: int = 42
) -> dict[str, Any]:
    """Os campos da identidade do Palpite, num dicionário só, lido pela rota e pela chave.

    A semente entra porque com outra semente as 200 features sorteadas são outras, e o Palpite
    guardado com 42 não responde por uma execução com `seed=7`.
    """
    return {
        "n_features": N_FEATURES_DO_PALPITE,
        "n_batches": n_batches,
        "batch_cuts": None if cortes is None else list(cortes),
        "seed": seed,
    }


def _guarda_o_palpite(chave: str, palpite: PalpiteMedido) -> None:
    with _TRAVA_DOS_PALPITES:
        _PALPITES[chave] = palpite
        _PALPITES.move_to_end(chave)
        while len(_PALPITES) > LIMITE_DE_PALPITES_RETIDOS:
            _PALPITES.popitem(last=False)


def _palpite_guardado(chave: str) -> PalpiteMedido | None:
    with _TRAVA_DOS_PALPITES:
        return _PALPITES.get(chave)


def _palpite_da_matriz(
    matriz: RefinoMatrix,
    *,
    n_batches: int,
    cortes: Sequence[int] | None,
    lotes: np.ndarray | None,
    paralelo: bool,
    seed: int = 42,
) -> PalpiteMedido:
    """O Palpite guardado desta matriz e destes lotes, ou medido agora e guardado.

    Medir aqui é o recurso, e custa o que o Palpite custa, cerca de um minuto, dentro do pedido. A
    janela mede o Palpite na leitura, pela rota assíncrona, e a delegação o acha guardado; quem
    chega sem ele (outro cliente, ou lotes mudados depois) paga a medição e recebe o mesmo número.
    """
    chave = _chave_do_palpite(matriz, n_batches, cortes, seed)
    palpite = _palpite_guardado(chave)
    if palpite is None:
        palpite = mede_o_palpite(matriz, batches=lotes, n_jobs=n_jobs_efetivo(paralelo), seed=seed)
        _guarda_o_palpite(chave, palpite)
    return palpite


def _delegacao_ou_422(
    auto: str,
    matriz: RefinoMatrix,
    *,
    n_batches: int = 1,
    cortes: Sequence[int] | None = None,
    lotes: np.ndarray | None = None,
    paralelo: bool = True,
    seed: int = 42,
) -> DelegacaoResposta:
    """Diagnostica, consulta a política e monta a resposta da delegação do `--auto`.

    `medida` não diagnostica: mede. É o Palpite das telas, o corretor com menor RSD dos QCs fora
    da amostra em 200 features, sobre os lotes da execução; ver `_palpite_da_matriz`. A guarda de
    vocabulário é a da CLI, e o `422` faz o papel do `exit 2`: um rótulo mal escrito não pode cair
    num `else` e correr outra política. Ao contrário do caminho sem delegação, que valida o
    corretor antes da ingestão, aqui a política precisa do perfil de defeitos da matriz já lida.
    """
    if auto not in AUTO_VALIDOS:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            f"`auto={auto}` não é um valor válido. Sem valor usa a política aprendida; "
            f"os rótulos são: {', '.join(AUTO_VALIDOS)}.",
        )
    try:
        recomendacao = (
            _palpite_da_matriz(
                matriz,
                n_batches=n_batches,
                cortes=cortes,
                lotes=lotes,
                paralelo=paralelo,
                seed=seed,
            ).como_recomendacao()
            if auto == AUTO_MEDIDA
            else recomenda(matriz, auto)
        )
    except ArtefatoInvalidoError as erro:
        # `503`, e não `422`: a política embalada não carregou, e o cliente não errou. A ação dele
        # é tentar sem `auto` ou avisar quem mantém, não corrigir o pedido.
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(erro)) from erro
    except DegenerateCrossValidationError as erro:
        # Só `medida` chega aqui: com menos de três QCs num lote o Palpite não tem treino e a
        # validação cruzada levanta. É a recusa de `POST /matrizes/palpite`; sem ela, seria `500`.
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            f"`auto={auto}` não mede esta matriz: {erro} Use `auto={AUTO_APRENDIDA}`, ou "
            "escolha o corretor.",
        ) from erro
    ressalva = AUTO_CAVEATS[auto]
    return DelegacaoResposta(
        rotulo=auto,
        # O rótulo validado é a própria política; derivá-la por `if` gravaria `medida` como
        # `aprendida`.
        politica=auto,
        correction=recomendacao.correction,
        confidence=recomendacao.confidence,
        abstains=recomendacao.abstains,
        assessable=recomendacao.assessable,
        rationale=recomendacao.rationale,
        evidence=dict(recomendacao.evidence),
        caveat=ItemComRessalva(
            item=f"--auto {auto}",
            resumo=ressalva.resumo,
            detalhe=ressalva.detalhe,
            evidencia=ressalva.evidencia,
            portao_inerte=ressalva.portao_inerte,
        ),
    )


def _corretor_ou_422(method: str, null_denominator: str, *, paralelo: bool = True) -> Corrector:
    """O corretor pedido, validado na fronteira.

    Espelha `refino.cli.build_corrector`, inclusive o paralelismo do `serrf`, para a mesma matriz
    não custar tempos diferentes conforme a porta. `paralelo` é a caixa 'usar vários núcleos' da
    janela, e quem decide é `n_jobs_efetivo`, que cruza a escolha com a capacidade do processo: a
    caixa é permissão, não ordem. O `loess` também a lê, com o mesmo resultado bit a bit da série;
    a CLI, que não tem a caixa, o constrói em série. `UNAVAILABLE_METHODS` hoje está vazio, e a
    recusa fica para o caso de algum método ser bloqueado.
    """
    try:
        validate_label("Correction", method)
        validate_label("NullDenominatorPolicy", null_denominator)
    except InvalidLabelError as erro:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(erro)) from erro
    if method in UNAVAILABLE_METHODS:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            f"`correction={method}` não é selecionável: {UNAVAILABLE_METHODS[method].reason}",
        )
    if method == "loess":
        return LoessCorrector(
            null_denominator=null_denominator,  # type: ignore[arg-type]
            n_jobs=n_jobs_efetivo(paralelo),
        )
    if method == "serrf":
        return SerrfNaVez(n_jobs=n_jobs_efetivo(paralelo))
    return AVAILABLE_METHODS[method]()


def _corretor_da_spec(name: str, *, paralelo: bool) -> Corrector:
    """O corretor de uma spec já validada, com o regime de núcleos aplicado a `serrf` e `loess`.

    `AVAILABLE_METHODS` daria o `SerrfCorrector` com `n_jobs=1`, em série mesmo onde o `loky`
    funciona. Não usa `_corretor_ou_422` porque `PipelineSpec` já validou o rótulo, e não haveria
    `422` a dar. `loess` e `none` nascem com os padrões deles (inclusive o `null_denominator`);
    o regime de núcleos muda o tempo, não a matriz.
    """
    if name == "serrf":
        return SerrfNaVez(n_jobs=n_jobs_efetivo(paralelo))
    if name == "loess":
        return LoessCorrector(n_jobs=n_jobs_efetivo(paralelo))
    return AVAILABLE_METHODS[name]()


@router.post(
    "/matrizes/corrigir", status_code=status.HTTP_202_ACCEPTED, response_model=TrabalhoAceito
)
def corrigir(
    measurements: Annotated[UploadFile, File()],
    sample_list: Annotated[UploadFile, File()],
    mode: Annotated[str, Form()],
    method: Annotated[str, Form()] = "none",
    null_denominator: Annotated[str, Form()] = "r_faithful",
    n_batches: Annotated[int, Form(ge=1)] = 1,
    seed: Annotated[int, Form(ge=0)] = 42,
    batch_cuts: Annotated[str | None, Form()] = None,
    orientation: Annotated[str, Form()] = "features-in-rows",
    auto: Annotated[str | None, Form()] = None,
    # A caixa 'usar vários núcleos' da janela, ligada por padrão. O `serrf` e o `loess` a leem;
    # o `none` não calcula.
    paralelo: Annotated[bool, Form()] = True,
) -> TrabalhoAceito:
    """Dispara a correção e devolve `202` com o identificador; espelha `refino corrigir`.

    Recebe os dois arquivos, o modo, o corretor (`method`) e os lotes. `auto` espelha o `--auto`
    da CLI: com ele, `method` é ignorado e a resposta traz `delegacao` com a escolha e a
    justificativa. `n_batches` vale 1 por padrão, e não a sugestão do diagnóstico, porque quem
    chama a rota direto não passou por diagnóstico; a janela passa o número.

    O artefato sai por `GET /trabalhos/{identificador}`: a matriz corrigida no formato `toserrf`,
    idêntica byte a byte à da CLI, e no formato `normalizado` como alternativa. Recusa com `422`,
    antes de aceitar, corretor ou `auto` fora do vocabulário, arquivo ilegível e cortes de lote
    inválidos; com `auto=medida`, também a matriz sem QCs que bastem para medir. Política embalada
    que não carrega é `503`.
    """
    # Sem delegação, o corretor nasce antes da ingestão, para um erro de vocabulário não custar a
    # leitura do arquivo. Com delegação ele nasce depois, porque a política precisa da matriz lida.
    corretor = (
        None if auto is not None else _corretor_ou_422(method, null_denominator, paralelo=paralelo)
    )
    # Os bytes são lidos aqui para o `checksum_entrada`, como em `/execucoes`: um `UploadFile` só
    # se lê uma vez, e é por isso que `_carrega` recebe `conteudo_m`.
    bruto_m = measurements.file.read()
    bruto_l = sample_list.file.read()
    sample_list.file.seek(0)
    matriz = _ingere_ou_422(measurements, sample_list, mode, orientation, conteudo_m=bruto_m)
    cortes = _cortes_ou_422(batch_cuts, n_batches)
    lotes = _lotes_ou_422(matriz, n_batches, cortes)

    delegacao = (
        None
        if auto is None
        else _delegacao_ou_422(
            auto,
            matriz,
            n_batches=n_batches,
            cortes=cortes,
            lotes=lotes,
            paralelo=paralelo,
            seed=seed,
        )
    )
    if delegacao is not None:
        method = delegacao.correction
        corretor = _corretor_ou_422(method, null_denominator, paralelo=paralelo)
    assert corretor is not None  # os dois ramos o constroem; o mypy não vê a exclusividade

    def trabalho(marca: Marca, cancelado: CancelCheck) -> object:
        # O cancelamento entra no corretor: com `serrf` esta chamada leva minutos, e uma checagem
        # só entre etapas esperaria por ela inteira.
        resultado = _corrige_reusando(
            corretor,
            matriz,
            batches=lotes,
            seed=seed,
            cancelled=cancelado,
        )
        marca()
        raise_if_cancelled(cancelado, "corrigir: antes de escrever o artefato")
        # Os dois formatos saem da mesma correção: escrever o segundo é quase de graça perto de
        # corrigir. O custo é memória, o dobro por trabalho retido, e foi aceito: com
        # `LIMITE_DE_TRABALHOS_RETIDOS`, o pior caso fica perto de 390 MB.
        artefato = _artefato_retido(
            f"{mode}_corrigido.csv", lambda destino: write_toserrf(resultado.matrix, destino)
        )
        artefato = replace(
            artefato,
            alternativo=_artefato_retido(
                f"{mode}_normalizado.csv",
                lambda destino: write_normalizado(resultado.matrix, destino),
            ),
        )
        # A última etapa também avisa: sem isso a barra parava antes do fim, e a duração da
        # escrita não teria onde ser fechada.
        marca()
        return artefato

    # O mesmo dicionário é a chave do trabalho e o `spec_json` gravado: tudo o que muda o
    # resultado, e é isso que torna a duração interpretável noutra máquina.
    #
    # `method` é o efetivo, reatribuído acima sob delegação: `--auto` que escolhe `serrf` e
    # `method=serrf` produzem o mesmo artefato e partilham o trabalho. A `delegacao` da resposta é
    # do pedido, não do resultado.
    campos = {
        "method": method,
        "null_denominator": null_denominator,
        "n_batches": n_batches,
        "seed": seed,
        "batch_cuts": None if cortes is None else list(cortes),
    }
    return _despacha(
        "corrigir",
        matriz,
        campos,
        ("correcao", "escrita do artefato"),
        trabalho,
        checksum=checksum_do_par(bruto_m, bruto_l),
        delegacao=delegacao,
        # O efetivo, e não a caixa: quem lê a duração quer saber quantos trabalhadores correram,
        # e a caixa marcada num processo sem interpretador ainda dá série.
        parametros_extra={"n_jobs": n_jobs_efetivo(paralelo)},
    )


@router.post(
    "/matrizes/medir-corretor",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=TrabalhoAceito,
)
def medir_corretor(
    measurements: Annotated[UploadFile, File()],
    sample_list: Annotated[UploadFile, File()],
    mode: Annotated[str, Form()],
    orientation: Annotated[str, Form()] = "features-in-rows",
    n_sampled_features: Annotated[int | None, Form(ge=1)] = None,
    # A caixa 'usar vários núcleos' da janela, ligada por padrão.
    paralelo: Annotated[bool, Form()] = True,
    # Os lotes declarados: a medição corre na mesma partição que a correção vai usar.
    n_batches: Annotated[int, Form(ge=1)] = 1,
    batch_cuts: Annotated[str | None, Form()] = None,
    # A semente da medição, a mesma de `POST /matrizes/palpite`, com a qual o par busca o Palpite
    # guardado. `None` e não `42`: sem semente pedida, `runs.seed` grava a sentinela `-1`, e a
    # coluna não registra como escolha da pessoa uma constante do código.
    seed: Annotated[int | None, Form(ge=0)] = None,
    # O intervalo por barra, desligado por padrão: pedido, a medição faz `n(n+1)/2` ajustes por
    # corretor em vez de `n`. Como `bool` do formulário, um valor inválido é `422` no disparo.
    intervalo_por_barra: Annotated[bool, Form()] = False,
) -> TrabalhoAceito:
    """Mede qual corretor vence: corre `none`, `loess` e `serrf` e compara o RSD dos QCs.

    É a metade 'depois' do par de recomendação: o Palpite (`POST /matrizes/palpite`, a mesma
    medição em 200 features) é a metade 'antes', lido da memória e devolvido em `palpite` ao lado
    do vencedor. O estimador é `leave_one_qc_out`: cada corretor é reajustado uma vez por QC, e
    a medição leva minutos, por isso responde `202` e o resultado sai por
    `GET /trabalhos/{identificador}`, com progresso por corretor e cancelamento.

    O `none` está na comparação de propósito: só com ele a medição pode responder 'não corrija'.
    Quando a margem relativa do vencedor fica abaixo de `ABSTENTION_MARGIN`, a resposta vem com
    `abstem` verdadeiro. `n_sampled_features` ausente mede a matriz inteira; quem passa um número
    troca custo por escopo, e ele volta em `n_features_medidas`.

    Recusa com `422`, antes de aceitar: arquivo ilegível, cortes de lote inválidos e lote com QCs
    de menos para medir fora da amostra.
    """
    bruto_m = measurements.file.read()
    bruto_l = sample_list.file.read()
    sample_list.file.seek(0)
    matriz = _ingere_ou_422(measurements, sample_list, mode, orientation, conteudo_m=bruto_m)
    cortes = _cortes_ou_422(batch_cuts, n_batches)
    lotes = _lotes_ou_422(matriz, n_batches, cortes)
    qcs = _qcs_por_lote(matriz, [0] * matriz.n_samples if lotes is None else lotes.tolist())
    # A recusa vem no disparo: com um QC retido, um lote de dois QCs treina sobre um ponto só, e
    # a medição levantaria depois de minutos de SERRF em vez de devolver número.
    if _algum_lote_sem_treino(qcs):
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            f"com {n_batches} lote(s), os QCs por lote são {qcs}, e medir fora da amostra exige "
            f"ao menos {MIN_TRAIN_QC_PER_BATCH + 1} QCs em cada lote: um retido e "
            f"{MIN_TRAIN_QC_PER_BATCH} para ajustar. Com menos, a medição seria sobre um ponto só.",
        )

    # O Palpite é lido aqui, antes de a medição correr, para o par ser uma previsão feita antes e
    # não uma explicação depois. Vem da memória, sem medir de novo: sem Palpite guardado desta
    # matriz e destes lotes, `palpite` sai `None`, que a resposta distingue de 'errou'.
    semente = 42 if seed is None else seed
    guardado = _palpite_guardado(_chave_do_palpite(matriz, n_batches, cortes, semente))
    palpite: str | None = None if guardado is None else guardado.correction

    # A caixa chega aqui por `n_jobs`, porque o `LabelingOracle` constrói os corretores por dentro,
    # fora de `_corretor_ou_422`.
    oraculo = LabelingOracle(
        prefixes=(("none", n_batches), ("loess", n_batches), ("serrf", n_batches)),
        n_sampled_features=n_sampled_features,
        n_jobs=n_jobs_efetivo(paralelo),
        seed=semente,
    )

    def trabalho(marca: Marca, cancelado: CancelCheck) -> object:
        raise_if_cancelled(cancelado, "medir-corretor: antes do primeiro ajuste")

        def terminou_um(prefixo: tuple[str, int]) -> None:
            """Fecha uma etapa por corretor, e só depois confere o cancelamento.

            O corretor que acabou de correr conta na barra e na duração mesmo se a ordem de parar
            já chegou: cancelar interrompe o que vem, não desfaz o que foi. A chamada da marca não
            aparece escrita nesta docstring porque `test_api_progresso.py` conta as ocorrências
            dela no texto da rota.
            """
            marca()
            raise_if_cancelled(cancelado, f"medir-corretor: depois de `{prefixo[0]}`")

        # O cancelamento desce até o estimador: sem ele, cancelar durante um corretor esperaria
        # todos os ajustes dele.
        intervalos: dict[str, IntervaloDaBarra] | None = None
        if intervalo_por_barra:
            # O rótulo é o mesmo de `label`, bit a bit; o intervalo vem ao lado, sem mudar o
            # contrato de `Label`.
            rotulo, por_prefixo = oraculo.label_com_intervalos(
                matriz, batches=lotes, depois_de_cada=terminou_um, cancelled=cancelado
            )
            intervalos = {
                correcao: _intervalo_da_barra(intervalo)
                for (correcao, _lotes), intervalo in por_prefixo.items()
            }
        else:
            rotulo = oraculo.label(
                matriz, batches=lotes, depois_de_cada=terminou_um, cancelled=cancelado
            )
        vencedor = rotulo.correction
        margem = float(rotulo.margin)
        return MedicaoDoCorretorResposta(
            vencedor=vencedor,
            # As chaves são o corretor, e não o prefixo `(correcao, lotes)`: uma tupla não é chave
            # de JSON, e os três prefixos têm o mesmo número de lotes, então nada colide.
            scores={correcao: float(v) for (correcao, _lotes), v in rotulo.scores.items()},
            # `None` e não `nan`, que nem é JSON válido.
            margem=margem if math.isfinite(margem) else None,
            avaliavel=rotulo.assessable,
            # O limiar de abstenção é o da política, importado de `decision.policy`, para os dois
            # cortes não divergirem.
            abstem=(not math.isfinite(margem)) or margem < ABSTENTION_MARGIN,
            estimador="leave_one_qc_out",
            variante=oraculo.variant,
            n_features_medidas=n_sampled_features,
            palpite=palpite,
            palpite_acertou=None if palpite is None else palpite == vencedor,
            n_lotes=n_batches,
            qcs_por_lote=qcs,
            intervalos=intervalos,
            # Os corretores recusados por falta de QCs no lote, com o motivo.
            recusas={correcao: motivo for (correcao, _lotes), motivo in rotulo.refusals.items()},
        )

    return _despacha(
        "medir-corretor",
        matriz,
        {
            "n_sampled_features": n_sampled_features,
            "variante": oraculo.variant,
            "n_batches": n_batches,
            "batch_cuts": None if cortes is None else list(cortes),
            **({} if seed is None else {"seed": seed}),
            # Na chave só quando pedido: sem ele, o pedido com intervalo reclamaria o trabalho sem
            # intervalo; e, como a semente, os pedidos sem ele mantêm a chave de sempre.
            **({"intervalo_por_barra": True} if intervalo_por_barra else {}),
        },
        # Literal, e não derivada de `oraculo.prefixes`: `test_api_progresso.py` lê as etapas no
        # texto da rota. `test_api_medir_corretor.py` confere que a contagem bate com o oráculo.
        ("corretor none", "corretor loess", "corretor serrf"),
        trabalho,
        checksum=checksum_do_par(bruto_m, bruto_l),
        # O regime de núcleos vai para o registro, como nas rotas de arquivo: a duração de um
        # ajuste de SERRF muda várias vezes com o `n_jobs`, e sem ele não compara máquinas.
        parametros_extra={"n_jobs": n_jobs_efetivo(paralelo)},
    )


def _intervalo_da_barra(intervalo: JackknifeInterval) -> IntervaloDaBarra:
    """O intervalo do núcleo no fio: os três números ou o motivo, nunca `nan`."""
    return IntervaloDaBarra(
        metodo=JACKKNIFE_METHOD,
        nivel=intervalo.level,
        n_qc=intervalo.n_qc,
        inferior=intervalo.lower,
        superior=intervalo.upper,
        erro_padrao=intervalo.standard_error,
        motivo=intervalo.reason,
    )


@router.post(
    "/matrizes/palpite",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=TrabalhoAceito,
)
def palpite(
    measurements: Annotated[UploadFile, File()],
    sample_list: Annotated[UploadFile, File()],
    mode: Annotated[str, Form()],
    orientation: Annotated[str, Form()] = "features-in-rows",
    paralelo: Annotated[bool, Form()] = True,
    n_batches: Annotated[int, Form(ge=1)] = 1,
    batch_cuts: Annotated[str | None, Form()] = None,
    # A semente da execução, a mesma de `POST /matrizes/corrigir`: sorteia as 200 features e
    # semeia os ajustes.
    seed: Annotated[int, Form(ge=0)] = 42,
) -> TrabalhoAceito:
    """Mede o Palpite: o corretor com menor RSD dos QCs fora da amostra, em 200 features.

    É a medição de `POST /matrizes/medir-corretor` numa amostra: o mesmo oráculo, os mesmos lotes
    e o mesmo estimador, com 200 features sorteadas pela semente e a regra de empate de
    `decision/palpite.py` (abaixo de `LIMIAR_DE_EMPATE_DO_PALPITE` de margem, o Palpite se
    abstém, e a proposta continua a ser o vencedor da amostra). Responde `202`; o resultado sai
    por `GET /trabalhos/{identificador}` e fica guardado pela matriz, pelos lotes e pela semente,
    onde a delegação do 'Automático' e a medição completa o encontram.

    Recusa com `422`, antes de aceitar, arquivo ilegível, cortes inválidos e lote com QCs de menos
    para medir fora da amostra; política da segunda opinião que não carrega é `503`. Não grava
    duração no banco: não passa `checksum` a `_despacha`.
    """
    matriz = _ingere_ou_422(measurements, sample_list, mode, orientation)
    cortes = _cortes_ou_422(batch_cuts, n_batches)
    lotes = _lotes_ou_422(matriz, n_batches, cortes)
    qcs = _qcs_por_lote(matriz, [0] * matriz.n_samples if lotes is None else lotes.tolist())
    # A mesma recusa no disparo de `medir-corretor`, e pelo mesmo motivo: com um lote de dois QCs,
    # esconder um deixa um só para ajustar, e a régua levanta depois de minutos de SERRF.
    if _algum_lote_sem_treino(qcs):
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            f"com {n_batches} lote(s), os QCs por lote são {qcs}, e medir fora da amostra exige "
            f"ao menos {MIN_TRAIN_QC_PER_BATCH + 1} QCs em cada lote: um retido e "
            f"{MIN_TRAIN_QC_PER_BATCH} para ajustar.",
        )
    campos = _campos_do_palpite(n_batches, cortes, seed)
    chave = _chave_do_palpite(matriz, n_batches, cortes, seed)
    # A segunda opinião carrega no disparo, como em `/matrizes/recomendar`: artefato ilegível é
    # `503` antes do minuto de medição, e não uma falha depois dele. Desativada, não carrega.
    try:
        pela_verdade = carrega_politica_pela_verdade() if segunda_opiniao_ativa() else None
    except ArtefatoInvalidoError as erro:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(erro)
        ) from erro

    def trabalho(marca: Marca, cancelado: CancelCheck) -> object:
        raise_if_cancelled(cancelado, "palpite: antes do primeiro ajuste")

        def terminou_um(prefixo: tuple[str, int]) -> None:
            """Fecha uma etapa por corretor e depois confere o cancelamento."""
            marca()
            raise_if_cancelled(cancelado, f"palpite: depois de `{prefixo[0]}`")

        # O Palpite guardado vale: medir de novo daria o mesmo número por mais um minuto. A
        # memória dos Palpites é separada do registro de trabalhos, que poda os concluídos.
        medido = _palpite_guardado(chave)
        if medido is None:
            medido = mede_o_palpite(
                matriz,
                batches=lotes,
                n_jobs=n_jobs_efetivo(paralelo),
                depois_de_cada=terminou_um,
                cancelled=cancelado,
                seed=seed,
            )
            _guarda_o_palpite(chave, medido)
        segunda = None
        if pela_verdade is not None:
            perfil = perfil_para_a_politica(matriz)
            segunda = SegundaOpiniao(
                correction_proposta=pela_verdade.recommend(perfil).correction,
                caminho=pela_verdade.caminho_do_corretor(perfil),
            )
        return PalpiteResposta(
            segunda_opiniao=segunda,
            correction_proposta=medido.correction,
            escores=dict(medido.escores),
            margem=medido.margem,
            segundo=medido.segundo,
            confianca=medido.confianca,
            abstem=medido.abstem,
            avaliavel=medido.avaliavel,
            rationale=medido.rationale,
            n_features_medidas=medido.n_features_medidas,
            limiar_de_empate=LIMIAR_DE_EMPATE_DO_PALPITE,
            variante=LabelingOracle().variant,
            n_lotes=n_batches,
            qcs_por_lote=qcs,
            disponivel_para_execucao=medido.correction in AVAILABLE_METHODS,
            recusas=dict(medido.recusas),
        )

    return _despacha(
        "palpite",
        matriz,
        campos,
        # Literal, como em `medir-corretor`: `test_api_progresso.py` lê as etapas no texto da
        # rota. Os avisos, que correm num gancho, são contados em `tests/test_api_palpite.py`.
        ("corretor none", "corretor loess", "corretor serrf"),
        trabalho,
    )


@router.post(
    "/matrizes/processar", status_code=status.HTTP_202_ACCEPTED, response_model=TrabalhoAceito
)
def processar(
    measurements: Annotated[UploadFile, File()],
    sample_list: Annotated[UploadFile, File()],
    mode: Annotated[str, Form()],
    method: Annotated[str, Form()] = "none",
    null_denominator: Annotated[str, Form()] = "r_faithful",
    n_batches: Annotated[int, Form(ge=1)] = 1,
    rsd_cutoff: Annotated[float | None, Form()] = None,
    missing_threshold: Annotated[float | None, Form()] = None,
    rsd_variant: Annotated[str, Form()] = "outlier_removed",
    ranking_filter: Annotated[str, Form()] = "none",
    ranking_keep_fraction: Annotated[float | None, Form()] = None,
    normalization_reference: Annotated[str | None, Form()] = None,
    # O terceiro arquivo, o CSV de fatores: a rota o lê antes de construir a especificação, para o
    # registro carregar os fatores por valor.
    normalization_factors: Annotated[UploadFile | None, File()] = None,
    ranking_filter_2: Annotated[str, Form()] = "none",
    ranking_keep_fraction_2: Annotated[float | None, Form()] = None,
    imputation: Annotated[str, Form()] = "one_fifth_min",
    sample_normalization: Annotated[str, Form()] = "none",
    transformation: Annotated[str, Form()] = "none",
    scaling: Annotated[str, Form()] = "none",
    seed: Annotated[int, Form(ge=0)] = 42,
    batch_cuts: Annotated[str | None, Form()] = None,
    orientation: Annotated[str, Form()] = "features-in-rows",
    # A caixa 'usar vários núcleos' da janela, ligada por padrão.
    paralelo: Annotated[bool, Form()] = True,
) -> TrabalhoAceito:
    """Corrige, pré-processa e devolve `202` com o identificador; espelha `refino processar`.

    Recebe os dois arquivos, o corretor, os eixos do pré-processamento e, opcionalmente, o CSV de
    fatores de normalização. O artefato, a matriz processada no formato `toserrf`, sai por
    `GET /trabalhos/{identificador}`. Não exige banco, como o subcomando que espelha; por isso é
    rota própria, e não um parâmetro de `/execucoes`. O corretor padrão é `none`, que devolve a
    matriz inalterada em valor.

    A correção vem antes do corte por RSD, e é etapa própria para uma falha do corretor não ser
    reportada como falha do pipeline. Recusa com `422`, antes de aceitar, vocabulário inválido,
    `imputation='none'` com normalização ativa, fatores malformados e arquivo ilegível. Se o
    corte esvazia a matriz ou a normalização não se aplica, o trabalho termina com `409`.
    """
    corretor = _corretor_ou_422(method, null_denominator, paralelo=paralelo)
    spec = _spec_ou_422(
        {
            "correction": method,
            "n_batches": n_batches,
            "rsd_cutoff": rsd_cutoff,
            "ranking_filter": ranking_filter,
            "ranking_keep_fraction": ranking_keep_fraction,
            "ranking_filter_2": ranking_filter_2,
            "ranking_keep_fraction_2": ranking_keep_fraction_2,
            "rsd_variant": rsd_variant,
            "missing_threshold": missing_threshold,
            "imputation": imputation,
            "sample_normalization": sample_normalization,
            "normalization_reference": normalization_reference,
            "normalization_factors": _fatores_do_upload(normalization_factors),
            "transformation": transformation,
            "scaling": scaling,
            "seed": seed,
            "batch_cuts": _cortes_ou_422(batch_cuts, n_batches),
        }
    )
    # Os bytes, para o `checksum_entrada`, como em `corrigir`.
    bruto_m = measurements.file.read()
    bruto_l = sample_list.file.read()
    sample_list.file.seek(0)
    matriz = _ingere_ou_422(measurements, sample_list, mode, orientation, conteudo_m=bruto_m)
    lotes = _lotes_ou_422(matriz, spec.n_batches, spec.batch_cuts)

    def trabalho(marca: Marca, cancelado: CancelCheck) -> object:
        # A correção de sinal vem antes do corte: cortar por RSD antes de corrigir descartaria a
        # feature pelo erro sistemático que a correção existe para remover.
        corrigida = _corrige_reusando(
            corretor,
            matriz,
            batches=lotes,
            seed=seed,
            cancelled=cancelado,
        )
        marca()
        raise_if_cancelled(cancelado, "processar: antes do pipeline")
        try:
            processada, _ = process_matrix(corrigida.matrix, spec)
        except (EmptyMatrixError, NormalizacaoInvalidaError) as erro:
            # `NormalizacaoInvalidaError` cobre, por exemplo, um nome de referência mal escrito:
            # é defeito do pedido, e `409`, não um `500` de defeito da ferramenta.
            raise HTTPException(status.HTTP_409_CONFLICT, str(erro)) from erro
        marca()
        raise_if_cancelled(cancelado, "processar: antes de escrever o artefato")
        artefato = _artefato_retido(
            f"{mode}_processado.csv", lambda destino: write_toserrf(processada, destino)
        )
        marca()  # a terceira etapa, como em `corrigir`
        return artefato

    # `null_denominator` entra na chave à mão porque `_spec_para_json` não o carrega; fora dela,
    # dois pedidos que só diferem nele receberiam o mesmo artefato.
    campos = {**_spec_para_json(spec), "null_denominator": null_denominator}
    return _despacha(
        "processar",
        matriz,
        campos,
        ("correcao", "pipeline", "escrita do artefato"),
        trabalho,
        checksum=checksum_do_par(bruto_m, bruto_l),
        # O efetivo, e não a caixa, como em `corrigir`.
        parametros_extra={"n_jobs": n_jobs_efetivo(paralelo)},
    )


@router.post(
    "/matrizes/exportar", status_code=status.HTTP_202_ACCEPTED, response_model=TrabalhoAceito
)
def exportar(
    measurements: Annotated[UploadFile, File()],
    sample_list: Annotated[UploadFile, File()],
    mode: Annotated[str, Form()],
    max_features: Annotated[int, Form()] = 5000,
    rsd_variant: Annotated[str, Form()] = "outlier_removed",
    orientation: Annotated[str, Form()] = "features-in-rows",
    method: Annotated[str, Form()] = "none",
    null_denominator: Annotated[str, Form()] = "uncorrected_feature",
    n_batches: Annotated[int, Form(ge=1)] = 1,
    seed: Annotated[int, Form(ge=0)] = 42,
    batch_cuts: Annotated[str | None, Form()] = None,
    # A caixa 'usar vários núcleos' da janela, ligada por padrão.
    paralelo: Annotated[bool, Form()] = True,
    # Os eixos do pré-processamento, os mesmos de `/matrizes/processar`, com padrão neutro; a
    # docstring explica por que a imputação difere.
    rsd_cutoff: Annotated[float | None, Form()] = None,
    missing_threshold: Annotated[float | None, Form()] = None,
    ranking_filter: Annotated[str, Form()] = "none",
    ranking_keep_fraction: Annotated[float | None, Form()] = None,
    ranking_filter_2: Annotated[str, Form()] = "none",
    ranking_keep_fraction_2: Annotated[float | None, Form()] = None,
    imputation: Annotated[str, Form()] = "none",
    sample_normalization: Annotated[str, Form()] = "none",
    normalization_reference: Annotated[str | None, Form()] = None,
    normalization_factors: Annotated[UploadFile | None, File()] = None,
    transformation: Annotated[str, Form()] = "none",
    scaling: Annotated[str, Form()] = "none",
) -> TrabalhoAceito:
    """Corrige e exporta no formato do MetaboAnalyst; espelha `refino exportar` sem `--enriquecer`.

    Recebe os dois arquivos, o corretor, `max_features` e, opcionalmente, os eixos do
    pré-processamento de `/matrizes/processar`. Responde `202`; o CSV, com o relatório das
    features cortadas, sai por `GET /trabalhos/{identificador}`, como nas outras rotas de
    artefato. O enriquecimento fica de fora: é funcionalidade adicional da CLI.

    A rota corrige antes de exportar, porque o fluxo do laboratório corrige antes do upload. O
    `null_denominator` padrão é `uncorrected_feature`, e não o de `corrigir`: sob `r_faithful` o
    `batch_loess` produz `Inf`, que `write_metaboanalyst` recusa.

    Os eixos do pré-processamento têm padrão neutro, e a imputação padrão é `none`, e não a do
    `processar`: o arquivo de quem não manda os campos continua idêntico ao da CLI, o MetaboAnalyst
    imputa por conta própria (`docs/metodos-metaboanalyst.md`, `[N-1]` e `[N-2]`), e quem pede
    normalização sem imputação recebe `422` com o motivo. Sem eixo ligado o pré-processamento não
    corre, e o arquivo não passa pelo filtro de variância quase nula, que o `processar` aplica.
    Quando corre, fica dentro da etapa `exportacao`, para o `total` de quem não manda os campos
    não mudar. `rsd_variant` serve ao filtro de RSD e ao corte do escritor.

    Recusa com `422`, antes de aceitar, vocabulário inválido, `imputation='none'` com
    normalização ativa e arquivo ilegível. O trabalho termina com `422` quando o escritor recusa
    (`max_features < 1`, valor negativo ou infinito); com valor negativo, a mensagem sugere o
    mesmo pedido com transformação e escalonamento em `none`. Termina com `409` se o
    pré-processamento esvazia a matriz.
    """
    try:
        validate_label("RsdVariant", rsd_variant)
    except InvalidLabelError as erro:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(erro)) from erro
    # O corretor é validado antes do disparo, como em `corrigir` e `processar`, para um erro de
    # digitação não esperar a reclamação.
    corretor = _corretor_ou_422(method, null_denominator, paralelo=paralelo)
    # A especificação nasce antes da ingestão, como no `processar`: rótulos inválidos e
    # `imputation='none'` com normalização são recusados sem ler a matriz.
    spec = _spec_ou_422(
        {
            "correction": method,
            "n_batches": n_batches,
            "rsd_cutoff": rsd_cutoff,
            "ranking_filter": ranking_filter,
            "ranking_keep_fraction": ranking_keep_fraction,
            "ranking_filter_2": ranking_filter_2,
            "ranking_keep_fraction_2": ranking_keep_fraction_2,
            "rsd_variant": rsd_variant,
            "missing_threshold": missing_threshold,
            "imputation": imputation,
            "sample_normalization": sample_normalization,
            "normalization_reference": normalization_reference,
            "normalization_factors": _fatores_do_upload(normalization_factors),
            "transformation": transformation,
            "scaling": scaling,
            "seed": seed,
            "batch_cuts": _cortes_ou_422(batch_cuts, n_batches),
        }
    )
    pre_processa = requests_preprocessing(spec)
    # Os bytes, para o `checksum_entrada`, como em `corrigir`.
    bruto_m = measurements.file.read()
    bruto_l = sample_list.file.read()
    sample_list.file.seek(0)
    matriz = _ingere_ou_422(measurements, sample_list, mode, orientation, conteudo_m=bruto_m)
    lotes = _lotes_ou_422(matriz, spec.n_batches, spec.batch_cuts)

    def escreve(alvo: RefinoMatrix, destino: Path) -> RelatorioDeExportacao:
        """Escreve o CSV e devolve o relatório do corte, com as causas que o escritor separa."""
        try:
            relatorio = write_metaboanalyst(alvo, destino, max_features, rsd_variant=rsd_variant)  # type: ignore[arg-type]
        except (ValueError, InvalidLabelError) as erro:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(erro)) from erro
        return _relatorio_para_json(relatorio)

    def trabalho(marca: Marca, cancelado: CancelCheck) -> object:
        # Duas etapas, com o cancelamento conferido entre elas: com `serrf` a correção custa
        # tanto quanto em `corrigir`.
        corrigida = _corrige_reusando(
            corretor,
            matriz,
            batches=lotes,
            seed=seed,
            cancelled=cancelado,
        )
        marca()
        raise_if_cancelled(cancelado, "exportar: antes de escrever o artefato")
        alvo = corrigida.matrix
        if pre_processa:
            try:
                alvo, _ = process_matrix(alvo, spec)
            except (EmptyMatrixError, NormalizacaoInvalidaError) as erro:
                # O mesmo `409` do `processar`: o pedido estava bem formado e falhou contra esta
                # matriz, num corte que esvaziou ou numa referência ausente.
                raise HTTPException(status.HTTP_409_CONFLICT, str(erro)) from erro
        artefato = _artefato_retido(
            f"{mode}_metaboanalyst.csv", lambda destino: escreve(alvo, destino)
        )
        marca()
        return artefato

    # A especificação inteira entra na chave, mais os dois campos que ela não carrega: sem eles,
    # dois pedidos que só diferem no corretor ou no escalonamento receberiam o mesmo artefato do
    # cache. Derivada de `_spec_para_json`, para um campo novo de `PipelineSpec` entrar sozinho.
    campos = {
        **_spec_para_json(spec),
        "max_features": max_features,
        "null_denominator": null_denominator,
    }
    return _despacha(
        "exportar",
        matriz,
        campos,
        ("correcao", "exportacao"),
        trabalho,
        checksum=checksum_do_par(bruto_m, bruto_l),
        # O efetivo, e não a caixa, como em `corrigir`.
        parametros_extra={"n_jobs": n_jobs_efetivo(paralelo)},
    )
