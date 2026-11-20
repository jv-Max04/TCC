"""Predição de atividade de via a partir dos picos: o condutor do mummichog.

O algoritmo é o do mummichog (Li et al., PLoS Comput Biol, 2013), usado como biblioteca, e
é também o da aba Functional Analysis / MS Peaks to Pathways do MetaboAnalyst, o que torna
o resultado comparável com o que o laboratório conhece. Os identificadores em
`compostos_candidatos` são candidatos por massa, não identificações: uma via com
sobreposição de 3 diz que três compostos compatíveis em massa com ela estão entre os picos
significativos, não que eles estão na amostra.

Este módulo cuida do que a chamada direta à biblioteca não cuida:

1. O modelo metabólico é compartilhado e mutável: `update_Compounds_adducts` reescreve
   `cpd['adducts']` no dicionário do módulo, e um modelo negativo construído depois de um
   positivo troca os adutos do primeiro. `_rede_privada` copia os compostos antes.
2. As permutações usam o `random` global, sem semente. O estado é semeado e depois
   restaurado, e o bloco corre sob `_TRAVA`, porque `random`, `sys.stdout` e o logger raiz
   são globais do processo.
3. `logging.basicConfig` roda no import do pacote e mexe no logger raiz. O import é tardio,
   e o nível e os handlers da raiz são restaurados.
4. O p de permutação da biblioteca é otimista em empate, e é recalculado em
   `p_de_permutacao`.
"""

import bisect
import contextlib
import io
import logging
import math
import random
import threading
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from refino.domain.labels import AdductSearch, validate_label
from refino.pathways.picos import Pico, ResumoDeInterpretacao, interpreta_todos

__all__ = [
    "FAIXA_DE_MASSA",
    "MINIMO_COM_CORTE_AUTOMATICO",
    "EntradaPequenaDemaisError",
    "EntradaRecusadaError",
    "EntradaSemFeatureAnalisavelError",
    "PoliticaDeMassaNeutra",
    "RecorteDaEntrada",
    "RelatorioDeVias",
    "ViaPredita",
    "p_de_permutacao",
    "prediz_vias",
]

# A faixa que o mummichog aceita, em Da. É replicada aqui, e conferida contra a dele em
# `_confere_faixa`, porque é preciso contar quem cai fora antes de enviar: a biblioteca
# descarta em silêncio.
FAIXA_DE_MASSA = (50.0, 2000.0)

PoliticaDeMassaNeutra = Literal["primary_ion", "exclude"]

# Serializa o bloco que mexe em `random`, `sys.stdout` e no logger raiz, que são globais
# do processo. Duas chamadas concorrentes com a mesma semente divergem do resultado
# serial.
_TRAVA = threading.Lock()


class EntradaRecusadaError(ValueError):
    """A entrada foi recusada antes de chegar à biblioteca, com a causa escrita para o usuário.

    É exceção própria, e não `ValueError` nu, porque o painel imprime a mensagem inteira
    como diagnóstico da entrada. Um `ValueError` vindo do `scipy`, do `numpy` ou de
    `InvalidLabelError` não pode aparecer ali como se fosse.

    É classe-base porque as duas recusas são fatos diferentes: não sobrou nenhuma
    feature, ou sobrou uma e o corte automático precisa de duas. O painel apanha a base;
    quem quiser distinguir apanha a filha.
    """


class EntradaSemFeatureAnalisavelError(EntradaRecusadaError):
    """Nenhuma feature sobrou para analisar, com as quatro causas nomeadas na mensagem."""


class EntradaPequenaDemaisError(EntradaRecusadaError):
    """Sobrou feature, mas menos do que a escolha automática do corte consegue usar.

    Com o corte automático, `InputUserData.determine_significant_list`
    (`mummichog/get_user_data.py:418`) lê ``new[int(n / 4) + 1]``, e esse índice só existe
    quando ``int(n / 4) + 1 < n``, isto é, ``n >= 2``. Com uma feature a biblioteca levanta
    `IndexError`. Com corte fixo uma feature corre, e por isso a guarda só dispara com o
    corte automático.
    """


# O menor número de features que o corte automático da biblioteca aceita; a derivação está
# na docstring de `EntradaPequenaDemaisError`. Um teste corre a biblioteca crua com
# `MINIMO_COM_CORTE_AUTOMATICO - 1` e afirma que ela ainda levanta: se uma versão futura
# consertar o índice, o teste reprova e avisa que a guarda sobrou.
MINIMO_COM_CORTE_AUTOMATICO = 2


# De `(polaridade, busca)` para o nome da lista no `wanted_adduct_list` do mummichog.
#
# O padrão é `broad` para que o resultado converse com o do MetaboAnalyst. O
# MetaboAnalystR procura 17,62 formas iônicas por composto no modo positivo, e a lista
# `pos_default` do mummichog procura 7,97; com ela este lado acha menos compostos
# empíricos que o deles. Com `generic_positive`, de 25 adutos, a concordância de Spearman
# contra o MetaboAnalystR vai de 0,36 para 0,91.
#
# `default` continua disponível, e é a fidelidade ao padrão do próprio mummichog.
#
# A linha negativa repete a mesma lista porque o mummichog publica uma só para essa
# polaridade.
LISTA_DE_ADUTOS: dict[tuple[str, str], str] = {
    ("positive", "default"): "pos_default",
    ("positive", "broad"): "generic_positive",
    ("negative", "default"): "negative",
    ("negative", "broad"): "negative",
}


@dataclass(frozen=True)
class ViaPredita:
    """Uma via, com a evidência agregada que o algoritmo produziu para ela.

    `pvalue` é o ajustado por permutação, e não o de Fisher. `pvalue_fisher` testa a
    sobreposição contra a hipótese de sorteio uniforme, e ignora que vias grandes e
    compostos de massa comum casam mais por acaso; `pvalue` desconta isso reamostrando
    listas do mesmo tamanho sobre os mesmos picos, e é por ele que as vias são ordenadas.

    `pvalue` é calculado aqui, e não lido da biblioteca: a mesma nula do mummichog, com
    os empates contados como "pelo menos tão extremo" (ver `p_de_permutacao`). Onde não
    há empate o valor é idêntico ao dele.

    `detectados_na_via` é o tamanho medido da via, quantos compostos empíricos dela
    apareceram nestes dados, e não o tamanho teórico no modelo. A razão entre
    sobreposição e esse tamanho é cobertura do que o experimento viu, não da biologia.
    """

    nome: str
    sobreposicao: int
    detectados_na_via: int
    pvalue: float
    pvalue_fisher: float
    compostos_candidatos: tuple[str, ...]
    nomes_candidatos: tuple[str, ...]


@dataclass(frozen=True)
class RecorteDaEntrada:
    """Quem entrou na análise e quem ficou de fora, com a causa de cada saída.

    Na biblioteca toda saída deste caminho é silenciosa: massa fora de faixa é
    descartada sem aviso, estatística não finita quebra na conversão, e massa neutra
    excluída por política simplesmente não aparece. Uma tabela de vias no fim não
    denuncia nenhuma delas.
    """

    recebidas: int
    ions: int
    massas_neutras_incluidas: int
    massas_neutras_excluidas: int
    nao_interpretadas: int
    sem_estatistica: int
    fora_da_faixa_de_massa: int
    analisadas: int
    features_significativas: int
    """Quantas features entraram na lista significativa da biblioteca.

    O nome leva `features` porque `RelatorioDeVias.significativas` conta vias abaixo de
    0,05, que é outra população."""
    corte_de_significancia: float


@dataclass(frozen=True)
class RelatorioDeVias:
    """O desfecho completo: as vias, o recorte da entrada e os parâmetros que valeram.

    Os parâmetros viajam com o resultado para que ele se explique sozinho.
    `corte_de_significancia` é escolhido pela biblioteca quando o usuário não o fixa,
    então guardar só o argumento não diz qual corte valeu.
    """

    vias: tuple[ViaPredita, ...]
    recorte: RecorteDaEntrada
    modo_de_ionizacao: Literal["positive", "negative"]
    politica_de_massa_neutra: PoliticaDeMassaNeutra
    busca_de_adutos: AdductSearch
    adutos_procurados: int
    """Quantas formas iônicas foram de fato procuradas por composto.

    É campo próprio, e não derivado de `busca_de_adutos`, porque no modo negativo os dois
    valores daquele rótulo selecionam a mesma lista. Como referência, o MetaboAnalystR
    procura 17,62 formas por composto no modo positivo; `default` aqui procura 8 e
    `broad`, 25."""
    ppm: float
    permutacoes: int
    seed: int
    versao_do_modelo: str

    @property
    def significativas(self) -> tuple[ViaPredita, ...]:
        """As vias abaixo de 0,05 no p ajustado por permutação.

        É conveniência de leitura, e não um corte aplicado ao resultado: `vias`
        continua completa e ordenada, para que se veja também quando nenhuma passou.
        """
        return tuple(v for v in self.vias if v.pvalue < 0.05)


def _confere_faixa(modulo_config: Any) -> None:
    """Levanta se a faixa replicada aqui deixou de ser a da biblioteca.

    Se o mummichog alargar a faixa, a contagem de `fora_da_faixa_de_massa` passaria a
    acusar features que ele analisou.
    """
    deles = tuple(float(x) for x in modulo_config.MASS_RANGE)
    if deles != FAIXA_DE_MASSA:
        raise RuntimeError(
            f"a faixa de massa do mummichog mudou para {deles}; "
            f"`FAIXA_DE_MASSA` aqui ainda diz {FAIXA_DE_MASSA}. O recorte da entrada "
            "contaria as exclusões pela faixa errada."
        )


def p_de_permutacao(observados: Sequence[float], nula: Sequence[float]) -> tuple[float, ...]:
    """O p de permutação de cada observado, com empate contado como "pelo menos tão extremo".

    ``p = (#{r ∈ nula : r ≤ x} + 1) / (len(nula) + 1)``, com `x` o p de Fisher da via e
    `nula` o `permutation_record` do mummichog: o p de Fisher de todas as vias em todas
    as permutações, agrupados.

    O mummichog 2.7.0 calcula o mesmo quociente em `__calculate_p__`
    (`functional_analysis.py`) como ``(sorted([x] + nula).index(x) + 1) / (len(nula) + 1)``.
    `list.index` devolve a primeira ocorrência, então num empate o numerador conta só os
    `r < x` estritos e o p sai otimista. O empate é comum: o teste de Fisher é discreto, e
    uma via sem sobreposição tem p de Fisher igual a 1 por atribuição, empatando com toda
    permutação que também deu 1. Um valor da nula igual ao observado não é evidência
    contra a hipótese nula, e por isso entra no numerador.

    Sem empate o valor é idêntico ao do mummichog, e uma via sem sobreposição dá
    exatamente 1. A nula continua agrupada entre vias de tamanhos diferentes, como no
    mummichog, então nenhum dos dois lados do empate dá garantia exata de nível.

    O MetaboAnalyst reporta um p Gamma ajustado sobre a mesma nula, e não este quociente;
    com ele a comparação é de posto.
    """
    ordenada = sorted(float(r) for r in nula)
    if any(math.isnan(r) for r in ordenada):
        # Um NaN na nula deixa `sorted` numa ordem que depende da posição dele na lista
        # recebida, e `bisect_right` conta errado sem levantar.
        raise ValueError("p de Fisher NaN na nula: o p de permutação não é definido")
    denominador = len(ordenada) + 1.0
    resultado = []
    for x in observados:
        x = float(x)
        if math.isnan(x):
            # NaN compara falso com tudo, e `bisect_right` devolveria uma posição
            # arbitrária: um p plausível para uma via sem p de Fisher.
            raise ValueError("p de Fisher NaN numa via: o p de permutação não é definido")
        resultado.append((bisect.bisect_right(ordenada, x) + 1) / denominador)
    return tuple(resultado)


def _via_predita(via: Any, definicoes: Any, pvalue: float) -> ViaPredita:
    """Converte uma via da biblioteca em `ViaPredita`, com os candidatos ordenados.

    O campo lido é `compounds`, e não `chosen_compounds`: este nasce `[]` e o único
    escritor é `collect_hit_Trios()`, que este módulo não chama.

    A ordenação garante o determinismo. `overlap_EmpiricalCompounds` é um `set` de
    objetos sem `__hash__` próprio, cuja ordem de iteração depende de endereços de
    memória, e `.compounds` vem de `list(set(...))` sobre strings, cuja ordem varia com
    `PYTHONHASHSEED`.

    Os nomes saem da mesma lista ordenada de identificadores, e não de uma segunda
    travessia, para que o nome na posição i seja o do identificador na posição i.
    """
    identificadores = tuple(
        sorted(
            {
                composto
                for empirico in via.overlap_EmpiricalCompounds
                for composto in empirico.compounds
            }
        )
    )
    return ViaPredita(
        nome=str(via.name),
        sobreposicao=int(via.overlap_size),
        detectados_na_via=int(via.EmpSize),
        pvalue=pvalue,
        pvalue_fisher=float(via.p_FET),
        compostos_candidatos=identificadores,
        nomes_candidatos=tuple(definicoes.get(c, "") for c in identificadores),
    )


def _chave_de_ordem(via: ViaPredita) -> tuple[float, float, str]:
    """A ordem das vias no relatório: p de permutação, p de Fisher, nome.

    É função com nome, e não `lambda` dentro de `prediz_vias`, para que o segundo
    componente possa ser testado diretamente: no dado do laboratório não há p empatado
    com p de Fisher diferente.
    """
    return (via.pvalue, via.pvalue_fisher, via.nome)


def _rede_privada(modelos: Any, rede_classe: Any, modo: str) -> Any:
    """Constrói uma rede metabólica cujos compostos ninguém mais partilha.

    Sem esta cópia, dois modos de ionização no mesmo processo se sobrescrevem, e o
    resultado errado não tem sintoma. Ver a docstring do módulo.
    """
    rede = rede_classe(modelos["human_model_mfn"])
    rede.Compounds = {chave: dict(composto) for chave, composto in rede.Compounds.items()}
    rede.update_Compounds_adducts(mode=modo)
    return rede


def prediz_vias(
    nomes: tuple[str, ...],
    pvalores: np.ndarray,
    estatisticas: np.ndarray,
    *,
    modo_de_ionizacao: Literal["positive", "negative"],
    politica_de_massa_neutra: PoliticaDeMassaNeutra = "primary_ion",
    busca_de_adutos: AdductSearch = "broad",
    ppm: float = 10.0,
    corte_de_significancia: float | None = None,
    permutacoes: int = 100,
    seed: int = 0,
) -> RelatorioDeVias:
    """Prediz atividade de via a partir dos picos, sem identificação prévia.

    `nomes`, `pvalores` e `estatisticas` são paralelos e vêm de `univariate_test`.

    `estatisticas` pode ser uma de duas quantidades, conforme a rota que
    `univariate_test` escolheu: `t` de Welch, com sinal, ou `U` de Mann-Whitney, que é
    uma contagem sem sinal. A função não depende de qual: a estatística não entra no
    p-valor da via, e o mummichog a usa só para eleger uma feature representante por
    composto empírico. Não se deriva direção daqui: uma feature com `U` alto está com
    ranks altos no grupo A, não "para cima".

    `corte_de_significancia=None` deixa a biblioteca escolher: ela procura um corte que
    produza uma lista significativa de tamanho utilizável, e relaxa quando o sinal é
    fraco. O corte que valeu volta em `recorte.corte_de_significancia`, e é preciso
    lê-lo antes de interpretar o resultado: uma tabela sob `p < 0,2` não afirma o que a
    mesma tabela sob `p < 0,001` afirmaria.

    `seed` torna a saída idêntica entre execuções com a mesma entrada.

    Levanta `ValueError` quando os três vetores não têm o mesmo comprimento ou há nome
    repetido. Levanta `EntradaSemFeatureAnalisavelError` quando não sobra feature para
    analisar: um relatório com zero vias seria indistinguível de "nenhuma via
    enriquecida". Levanta `EntradaPequenaDemaisError` quando sobra menos de
    `MINIMO_COM_CORTE_AUTOMATICO` feature e o corte é automático, caso em que a
    biblioteca levantaria `IndexError`. As duas recusas herdam de
    `EntradaRecusadaError`, que é o que o painel apanha.
    """
    validate_label("IonizationMode", modo_de_ionizacao)
    validate_label("NeutralMassPolicy", politica_de_massa_neutra)
    validate_label("AdductSearch", busca_de_adutos)

    if not (len(nomes) == len(pvalores) == len(estatisticas)):
        raise ValueError(
            f"vetores de comprimento diferente: {len(nomes)} nomes, "
            f"{len(pvalores)} p-valores, {len(estatisticas)} estatísticas. "
            "Alinhá-los por posição é responsabilidade de quem chama, e um "
            "desalinhamento silencioso associaria o p de uma feature à massa de outra."
        )

    # Nome repetido é recusado aqui. `posicao` abaixo é indexado por nome, então dois
    # nomes iguais colapsariam no último índice: as duas features leriam o p-valor e a
    # estatística da segunda, e a biblioteca descartaria uma das linhas idênticas em
    # silêncio.
    #
    # `load_matrix` já reprova com `DUPLICATE_FEATURE_NAME`, mas esta função é pública e
    # não pode depender da guarda de quem chama.
    repetidos = sorted(nome for nome, vezes in Counter(nomes).items() if vezes > 1)
    if repetidos:
        raise ValueError(
            f"{len(repetidos)} nome(s) de feature repetido(s), começando em "
            f"{repetidos[:3]}. Um nome repetido torna a associação nome→p-valor "
            "ambígua, e resolvê-la pela última ocorrência analisaria uma feature sob "
            "a estatística de outra."
        )

    resumo: ResumoDeInterpretacao = interpreta_todos(nomes)
    candidatos: list[Pico] = list(resumo.ions)
    if politica_de_massa_neutra == "primary_ion":
        candidatos.extend(resumo.massas_neutras)
        neutras_fora = 0
    else:
        neutras_fora = len(resumo.massas_neutras)

    posicao = {nome: i for i, nome in enumerate(nomes)}
    linhas: list[str] = []
    sem_estatistica = 0
    fora_da_faixa = 0
    for pico in candidatos:
        i = posicao[pico.nome]
        p, t = float(pvalores[i]), float(estatisticas[i])
        if not (np.isfinite(p) and np.isfinite(t)):
            sem_estatistica += 1
            continue
        mz = pico.como_mz(modo_de_ionizacao)
        if not (FAIXA_DE_MASSA[0] < mz < FAIXA_DE_MASSA[1]):
            fora_da_faixa += 1
            continue
        # O nome vai na quinta coluna porque a biblioteca remove linhas duplicadas:
        # duas features com mesmo m/z, RT, p e t colapsariam numa só sem aviso.
        linhas.append(f"{mz}\t{pico.rt}\t{p}\t{t}\t{pico.nome}")

    if not linhas:
        raise EntradaSemFeatureAnalisavelError(
            "nenhuma feature sobrou para análise de vias. Causas separadas: "
            f"{len(resumo.nao_interpretados)} nomes não interpretados, "
            f"{neutras_fora} massas neutras excluídas por política, "
            f"{sem_estatistica} sem estatística finita, "
            f"{fora_da_faixa} fora da faixa de massa {FAIXA_DE_MASSA}. "
            "Devolver relatório vazio faria isto parecer 'nenhuma via enriquecida'."
        )

    # A guarda de entrada pequena espelha o predicado da biblioteca. O mummichog escolhe o
    # corte sozinho quando `paradict["cutoff"]` é falso (`None` daqui vira `0.0`, e um `0.0`
    # passado por quem chama também), e só nesse caminho ele indexa além do fim da lista.
    corte_enviado = 0.0 if corte_de_significancia is None else corte_de_significancia
    if not corte_enviado and len(linhas) < MINIMO_COM_CORTE_AUTOMATICO:
        raise EntradaPequenaDemaisError(
            f"só {len(linhas)} de {len(nomes)} feature(s) chegou à análise, e a escolha "
            f"automática do corte de significância precisa de pelo menos "
            f"{MINIMO_COM_CORTE_AUTOMATICO}. Das que ficaram de fora: "
            f"{len(resumo.nao_interpretados)} nomes não interpretados, "
            f"{neutras_fora} massas neutras excluídas por política, "
            f"{sem_estatistica} sem estatística finita, "
            f"{fora_da_faixa} fora da faixa de massa {FAIXA_DE_MASSA}."
        )

    # ------------------------------------------------------------------
    # Import tardio: ver o item 3 da docstring do módulo.
    raiz = logging.getLogger()
    nivel_anterior = raiz.level
    try:
        from mummichog import config as mcg_config
        from mummichog.functional_analysis import PathwayAnalysis
        from mummichog.get_user_data import DataMeetModel, InputUserData
        from mummichog.JSON_metabolicModels import metabolicModels
        from mummichog.models import metabolicNetwork
    finally:
        raiz.setLevel(nivel_anterior)

    _confere_faixa(mcg_config)

    modo_mcg = "pos_default" if modo_de_ionizacao == "positive" else "negative"

    # No modo negativo os dois valores de busca dão a mesma lista, porque o mummichog só
    # publica uma para essa polaridade. `RelatorioDeVias.adutos_procurados` reporta o
    # número que valeu. Ver `LISTA_DE_ADUTOS`.
    lista_mcg = LISTA_DE_ADUTOS[(modo_de_ionizacao, busca_de_adutos)]
    paradict: dict[str, Any] = {
        # O mesmo valor que a guarda de entrada pequena leu, para que o predicado que decide
        # se a biblioteca escolhe o corte sozinha exista num lugar só.
        "cutoff": corte_enviado,
        "network": "human_mfn",
        "modeling": None,
        "mode": modo_mcg,
        "instrument": str(int(ppm)),
        "force_primary_ion": True,
        "permutation": int(permutacoes),
        "output": "refino",
        "workdir": "",
        "infile": "",
        # `web=True` lê a entrada daqui em vez de abrir arquivo, e nesse caminho a
        # lista de adutos é explícita em vez de resolvida pelo modo.
        "datatext": "m/z\tretention_time\tp-value\tt-score\tcustom_id\n" + "\n".join(linhas),
        "wanted_adduct_list": mcg_config.wanted_adduct_list[lista_mcg],
    }

    # A biblioteca escreve por `print` contagens intermediárias e um contador de
    # permutação. A saída é desviada para memória e descartada. Isso não esconde falha:
    # falha é exceção, que atravessa o `redirect_stdout`.
    ruido = io.StringIO()
    # O bloco inteiro corre sob um lock de módulo. `random`, `sys.stdout` e o logger raiz
    # são globais do processo: duas chamadas concorrentes se intercalam na semeadura, cada
    # uma restaura o estado da outra, e o resultado diverge do serial. `prediz_vias` é
    # pública e não pode depender de quem chama serializar.
    with _TRAVA:
        estado = random.getstate()
        random.seed(seed)
        nivel_anterior = raiz.level
        # Os handlers também são restaurados, e não só o nível: `logging.basicConfig`
        # acrescenta um `StreamHandler` à raiz quando ela não tem nenhum, e a partir daí
        # todo `basicConfig` da aplicação não faria nada.
        handlers_antes = list(raiz.handlers)
        try:
            with contextlib.redirect_stdout(ruido):
                dados = InputUserData(paradict, web=True)
                rede = _rede_privada(metabolicModels, metabolicNetwork, modo_mcg)
                misto = DataMeetModel(rede, dados)
                analise = PathwayAnalysis(misto.model.metabolic_pathways, misto)
                analise.cpd_enrich_test()
            resultado = list(analise.resultListOfPathways)
            # O p de permutação é recalculado, e não lido de `adjusted_p`: a biblioteca dá
            # o menor posto num empate, e o p sai otimista. A nula é a dela, lida aqui
            # dentro do lock porque é atributo do objeto que só este bloco segura.
            p_perm = p_de_permutacao(
                [float(via.p_EASE) for via in resultado], analise.permutation_record
            )
            definicoes = misto.model.dict_cpds_def
            analisadas = len(dados.ListOfMassFeatures)
            significativas = len(dados.input_featurelist)
            corte_usado = float(dados.paradict["cutoff"])
            versao = str(rede.version)
        finally:
            random.setstate(estado)
            raiz.setLevel(nivel_anterior)
            for surgido in [h for h in raiz.handlers if h not in handlers_antes]:
                raiz.removeHandler(surgido)

    # A conta tem de fechar: a biblioteca analisou exatamente o que foi enviado.
    # `_confere_faixa` compara só a constante `MASS_RANGE`, e quem de fato exclui é o
    # conjunto de filtros da biblioteca (o `<` estrito, a remoção de linhas repetidas, e o
    # que uma versão futura acrescentar). Esta conferência não precisa conhecer a causa.
    if analisadas != len(linhas):
        raise RuntimeError(
            f"enviámos {len(linhas)} picos e a biblioteca analisou {analisadas}. A "
            "diferença não tem causa nomeada, e o recorte reportaria uma conta que não "
            "fecha — o que faria features desaparecerem sem que nada acusasse."
        )

    # A ordem é refeita pelo p recalculado. Vias que a biblioteca separava podem empatar
    # (todas as sem sobreposição passam a 1), e a ordem entre elas tem de sair da própria
    # via. A chave (`_chave_de_ordem`) é `(pvalue, pvalue_fisher, nome)`: o p de Fisher
    # separa primeiro, e o nome fecha a ordem. Os nomes são únicos no modelo `MFN_1.10.4`;
    # se um modelo futuro repetir nomes, `sorted` é estável e o empate fica na ordem em
    # que a biblioteca devolveu as vias.
    vias = tuple(
        sorted(
            (_via_predita(via, definicoes, p) for via, p in zip(resultado, p_perm, strict=True)),
            key=_chave_de_ordem,
        )
    )

    return RelatorioDeVias(
        vias=vias,
        recorte=RecorteDaEntrada(
            recebidas=len(nomes),
            ions=len(resumo.ions),
            massas_neutras_incluidas=(
                len(resumo.massas_neutras) if politica_de_massa_neutra == "primary_ion" else 0
            ),
            massas_neutras_excluidas=neutras_fora,
            nao_interpretadas=len(resumo.nao_interpretados),
            sem_estatistica=sem_estatistica,
            fora_da_faixa_de_massa=fora_da_faixa,
            analisadas=analisadas,
            features_significativas=significativas,
            corte_de_significancia=corte_usado,
        ),
        modo_de_ionizacao=modo_de_ionizacao,
        politica_de_massa_neutra=politica_de_massa_neutra,
        busca_de_adutos=busca_de_adutos,
        adutos_procurados=len(mcg_config.wanted_adduct_list[lista_mcg]),
        ppm=ppm,
        permutacoes=permutacoes,
        seed=seed,
        versao_do_modelo=versao,
    )
