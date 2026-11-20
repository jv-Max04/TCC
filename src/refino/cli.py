"""A linha de comando `refino`: os subcomandos `corrigir`, `processar` e `exportar`.

Numa CLI o valor de vocabulário vem do teclado e nunca passou por análise estática. O modo de
falha que importa é o que não levanta: um `--method` inválido caindo num ramo razoável produz
um CSV com o outro corretor, e isso parece sucesso. Por isso todo rótulo passa por
`validate_label` na fronteira, e nenhuma opção usa `choices=` do `argparse` (ver
`_add_vocabulary_argument`).

Códigos de saída:

- 0: saída escrita;
- 2: uso inválido, inclusive rótulo de vocabulário fora do conjunto;
- 3: pedido bem formado que não pôde ser atendido: método sem implementação nesta versão,
  artefato da política inválido, pré-processamento que a matriz não sustenta, ou arquivo
  exportado não conforme;
- 4: entrada recusada: validação reprovada, com a lista completa de problemas, arquivo
  ilegível, ou matriz que não sustenta a correção ou o diagnóstico pedido.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, TextIO, get_args

import numpy as np

from refino.correction.base import Corrector
from refino.correction.loess import LoessCorrector
from refino.correction.none import NoneCorrector
from refino.correction.serrf import IntensidadeNegativaError
from refino.correction.serrf_na_vez import SerrfNaVez
from refino.decision.artefato import ArtefatoInvalidoError
from refino.decision.features import perfil_para_a_politica
from refino.decision.palpite import mede_o_palpite
from refino.decision.policy import AlwaysSERRFPolicy, DecisionPolicy, Recommendation
from refino.domain.correction import CorrectionResult
from refino.domain.labels import (
    VOCABULARY,
    Correction,
    InvalidLabelError,
    NullDenominatorPolicy,
    Orientation,
    RsdVariant,
    validate_label,
)
from refino.domain.matrix import RefinoMatrix
from refino.domain.spec import PipelineSpec
from refino.io.fatores import FatoresMalformadosError, le_fatores_de_bytes
from refino.io.politica import (
    carrega_politica_aprendida,
    carrega_politica_pela_verdade,
    segunda_opiniao_ativa,
)
from refino.io.readers import IngestionError, load_matrix
from refino.io.saida import garante_utf8
from refino.io.writers import write_toserrf
from refino.metrics.crossval import DegenerateCrossValidationError
from refino.metrics.rsd import rsd
from refino.metrics.structure import EstruturaSemPopulacaoError
from refino.preprocessing.normalization import NormalizacaoInvalidaError
from refino.preprocessing.pipeline import (
    EmptyMatrixError,
    chain_is_consistent,
    process_matrix,
    requests_preprocessing,
    stage_counts,
)
from refino.preprocessing.stage import StageResult

EXIT_OK = 0
EXIT_USAGE = 2
EXIT_UNAVAILABLE = 3
EXIT_INPUT_REJECTED = 4


class EntradaIlegivelError(Exception):
    """O arquivo de entrada não pôde ser lido como measurements e Sample_List.

    Existe para que a entrada má saia com código 4. Sem ela, um arquivo inexistente, uma pasta
    no lugar do arquivo ou os dois CSV trocados despejariam traceback e sairiam com 1, que não
    é código desta CLI e que um script lê como ferramenta quebrada.
    """


def _le_matriz(measurements: Path, sample_list: Path, mode: str, orientation: str) -> RefinoMatrix:
    """`load_matrix` com a falha de leitura traduzida e a recusa da validação intacta.

    `IngestionError` passa como está, porque os chamadores a imprimem problema a problema.
    `OSError` de arquivo ausente ou ilegível e `ValueError` de conteúdo fora do formato viram
    `EntradaIlegivelError`.
    """
    try:
        return load_matrix(measurements, sample_list, mode, orientation=orientation)  # type: ignore[arg-type]
    except IngestionError:
        raise
    except (OSError, ValueError) as erro:
        raise EntradaIlegivelError(f"{type(erro).__name__}: {erro}") from erro


# ---------------------------------------------------------------------------
# Os métodos de correção que a CLI expõe
# ---------------------------------------------------------------------------

# O que esta versão implementa. A união com `UNAVAILABLE_METHODS` tem de ser igual ao
# vocabulário de `Correction`, e `tests/unit/test_cli_corrigir.py` reprova se divergir:
# ninguém acrescenta método ao `Literal` sem dizer se está disponível.
AVAILABLE_METHODS: dict[str, type[Corrector]] = {
    "none": NoneCorrector,
    "loess": LoessCorrector,
    "serrf": SerrfNaVez,
}


@dataclass(frozen=True)
class Unavailable:
    """Método do vocabulário que a CLI não expõe, e por que não expõe.

    Registro e não string porque há dois estados, que pedem ação diferente:

    - `implemented=False`: não existe corretor, e falta código;
    - `implemented=True`: o corretor existe, e o critério que autorizaria expô-lo ainda não
      passou.

    `corrector` existe para que a transição entre os dois seja conferida por máquina.
    """

    reason: str
    implemented: bool
    gate: str
    gate_unit: str
    corrector: str


# Vazio, e fica: `_vocabulary_is_partitioned` continua a exigir que a união das duas tabelas
# cubra o vocabulário, e a estrutura permite bloquear um método de novo.
#
# Não se parametriza teste sobre este dicionário: sobre coleção vazia o `parametrize` coleta
# zero casos, e o relatório diz `passed`.
UNAVAILABLE_METHODS: dict[str, Unavailable] = {}


@dataclass(frozen=True)
class Caveat:
    """Método disponível cujo uso carrega ressalva medida.

    Terceiro estado, ao lado de disponível e indisponível: o método pode ser usado, e há um
    recorte medido que quem escolhe precisa ver.

    `resumo` é o que aparece no ponto de escolha, numa linha de terminal; `detalhe` é o recorte
    inteiro. Os dois saem daqui, nunca reescritos no ponto de uso.
    """

    resumo: str
    detalhe: str
    evidencia: str
    portao_inerte: str


# Os rótulos do `--auto`, o vocabulário fechado dessa opção.
AUTO_APRENDIDA: Final = "aprendida"
AUTO_CONSTANTE: Final = "constante"
# O terceiro rótulo é o Palpite medido: o RSD dos QCs fora da amostra em 200 features da
# própria matriz. É o que a opção automática das telas segue. `--auto` sem valor continua a
# ser a política aprendida.
AUTO_MEDIDA: Final = "medida"
AUTO_VALIDOS: Final[tuple[str, ...]] = (AUTO_APRENDIDA, AUTO_CONSTANTE, AUTO_MEDIDA)

# As ressalvas do modo automático, em dicionário próprio: as chaves de `METHOD_CAVEATS` são de
# `Correction`, e misturar dois domínios faria um `METHOD_CAVEATS.get(...)` casar um rótulo de
# política por acidente de nome.
#
# A ressalva da política aprendida declara que ela não está provada melhor que a constante,
# para ninguém ler automático como melhor.
AUTO_CAVEATS: dict[str, Caveat] = {
    AUTO_APRENDIDA: Caveat(
        resumo=(
            "política APRENDIDA, pela estabilidade dos QCs: mede 67,5% deixando um modo de fora, "
            "contra 75,0% de uma constante que sempre escolhe `serrf` — a diferença NÃO é "
            "significativa, e nos quatro arquivos deste estudo as duas escolhem o mesmo e acertam "
            "3 de 4"
        ),
        detalhe=(
            "**A escolha automática é uma opinião, não uma autoridade: medida, ela não supera a "
            "regra mais simples possível.**\n\n"
            "Medida da única forma que não olha para a resposta — treinando sem nenhuma matriz do "
            "modo que vai julgar —, acerta o corretor em **67,5%** de 40 matrizes simuladas; "
            "escolher sempre `serrf`, sem política nenhuma, acerta **75,0%**. A diferença não se "
            "distingue do acaso (McNemar p = 0,5078), e por modo ela empata em dois e perde nos "
            "outros dois. Deixando uma matriz de fora de cada vez ela acerta 55,0% contra os "
            "mesmos 75,0%, e ali a constante é melhor com significância (p = 0,0078). Nos quatro "
            "arquivos deste estudo as duas escolhem `serrf` nos quatro e acertam 3 de 4: erram o "
            "`neg_hilic`, onde a medição elege o LOESS. Como se chegou a esses números está na aba "
            "«Métodos e limites».\n\n"
            "**O que ela escolheu fica à vista antes de qualquer coisa correr**, no seletor "
            "«Corretor». Para decidir à mão, destique «Automático»: o seletor liberta-se, e "
            "voltar a ticar repõe a proposta. **A decisão continua a ser de quem conhece o "
            "ensaio.**"
        ),
        evidencia=(
            "O modelo retreinado no corpus rotulado com a floresta do SERRF que conta como "
            "a do R, medido "
            "nos três regimes: treinando e avaliando nas MESMAS 40 matrizes (vazamento total) "
            "87,5% contra 75,0%; sob leave-one-out, que ainda vaza o MODO porque cada dobra treina "
            "em 9 matrizes do mesmo modo da que testa, 55,0% contra 75,0% com McNemar p=0,0078; e "
            "deixando um MODO INTEIRO de fora — a única que não vaza — 67,5% contra 75,0%, "
            "p=0,5078, isto é, sem sinal detectável. O modelo da floresta anterior: "
            "90,0%, 72,5% e 70,0% contra 55,0%; o de oito QCs: 95,0%, 67,5% e 50,0% "
            "contra 60,0%. O `--auto` sem valor delega à política aprendida, e ela "
            "é o que se pode COMPARAR: `--auto constante` corre a outra sobre os mesmos dados"
        ),
        portao_inerte=(
            "os dois critérios de aceitação da política aprendida: recomendar o corretor ótimo "
            "fora da amostra nos QUATRO modos reais, cada modo avaliado pela dobra que não o viu; "
            "e reportar confiança abaixo de 0,5 no `neg_c18` sem se abster nos modos com vencedor "
            "claro"
        ),
    ),
    AUTO_CONSTANTE: Caveat(
        resumo=(
            "política CONSTANTE: escolhe sempre `serrf`, e mede 75,0% — acima dos 67,5% da "
            "aprendida, sem diferença significativa, e não é linha de base ingênua neste corpus"
        ),
        detalhe=(
            "`AlwaysSERRFPolicy` acerta 3 dos 4 modos reais e 30 das 40 matrizes simuladas do "
            "corpus da política: sob esta régua e no eixo do vencedor, ela É a melhor política "
            "constante, e coincide com `BestFixedCorrector` porque `serrf` é a classe majoritária. "
            "O que o uso NÃO afirma: que `serrf` é bom para a SUA matriz. No `neg_c18` está "
            "medido que corrigir tem tanta chance de piorar quanto de melhorar, e a confiança "
            "que acompanha a escolha vem da MARGEM relativa entre o melhor e o segundo — 2,56% do "
            "melhor naquele modo e 3,82% no `neg_hilic`, contra mais de 30% nos dois positivos. "
            "Margem pequena é "
            "a política a dizer que não sabe."
        ),
        evidencia=(
            "As 30 de 40 foram medidas no corpus simulado da política, rotulado com a floresta do "
            "SERRF que conta como a do R; as margens foram medidas nos quatro arquivos reais do "
            "estudo, com o SERRF e o LOESS desta versão: a mediana "
            "do RSD de 7 QCs fora da amostra sobre as features de RSD finito de cada arquivo real"
        ),
        portao_inerte=(
            "os dois critérios de aceitação da política aprendida: recomendar o corretor ótimo "
            "fora da amostra nos QUATRO modos reais, cada modo avaliado pela dobra que não o viu; "
            "e reportar confiança abaixo de 0,5 no `neg_c18` sem se abster nos modos com vencedor "
            "claro"
        ),
    ),
    AUTO_MEDIDA: Caveat(
        resumo=(
            "Palpite MEDIDO: o menor RSD de QC fora da amostra em 200 features sorteadas da sua "
            "matriz; em 88 matrizes simuladas concordou com o vencedor da medição completa em "
            "92,5% das vezes, contra 78% de escolher sempre `serrf`, e custa perto de 1 minuto com "
            "12 núcleos numa corrida de 7 QCs (cerca de 95 s com 11 QCs e 215 s com 21)"
        ),
        detalhe=(
            "**O Palpite mede, e não adivinha.** Ele corre os três corretores — não corrigir, "
            "LOESS e SERRF — sobre 200 features sorteadas da sua matriz, esconde um QC de cada "
            "vez, e compara o RSD dos QCs escondidos: o menor vence. A mesma matriz dá sempre as "
            "mesmas 200 features e o mesmo Palpite.\n\n"
            "**Quanto ele acerta, e onde erra.** Em 88 matrizes simuladas a partir dos arquivos "
            "deste estudo, concordou com a medição de todas as features em 92,5% dos sorteios, "
            "contra 78% de escolher sempre `serrf`; quando a diferença relativa entre os dois "
            "melhores corretores passa de 20% do melhor, concorda em 99,8%. Os desacordos ficam "
            "onde a escolha quase "
            "não importa: em média, sobre todos os sorteios, o RSD do corretor escolhido ficou "
            "0,29% acima do RSD do melhor, em diferença relativa.\n\n"
            "**Quando a diferença relativa na amostra fica abaixo de 16,7% do melhor, ele diz que "
            "é empate** e "
            "propõe o vencedor da amostra mesmo assim — trocá-lo pelo `serrf` foi medido e "
            "piorou. Nos quatro arquivos deste estudo isso acontece nos dois modos negativos, "
            "onde a medição completa também dá margens relativas pequenas (2,56% e 3,82% do "
            "melhor). **Para decidir "
            "um empate, meça todas as features** — na janela, o botão «Medir qual corretor "
            "vence».\n\n"
            "**O tempo** foi medido num Ryzen 5 7600 com 12 núcleos: de 52,7 a 59,7 s por arquivo. "
            "Com um núcleo só — na janela, a caixa «Usar vários núcleos» desmarcada —, ou numa "
            "máquina mais lenta, demora mais."
        ),
        evidencia=(
            "A exploração: 8 formas de treinar a árvore, nenhuma acima da constante "
            "retirando um modo; e a simulação da medição em 200 features nas 88 matrizes "
            "simuladas. O tempo foi medido no Ryzen. O"
            " limiar de empate, 0,167027, é derivado do ruído do sorteio;"
            " a regra que trocaria o vencedor pelo `serrf` no empate reprovou: arrependimento de "
            "0,841% contra 0,291%. A execução parcial é exata, conferida bit a bit nos "
            "quatro arquivos reais. As margens da medição completa nos dois modos negativos foram "
            "medidas nos arquivos reais, com o SERRF e o LOESS desta versão"
        ),
        portao_inerte=(
            "nenhum critério de aceitação mede o Palpite medido; os dois que existem medem "
            "a política aprendida, que continua em `--auto aprendida`"
        ),
    ),
}

# Os rótulos das duas opiniões do Palpite. A primeira é a política aprendida, a que o `--auto`
# segue, treinada pelo corretor de menor RSD nos QCs. A segunda é a mesma árvore treinada pelo
# corretor que mais recupera o sinal verdadeiro nas matrizes sintéticas. Ela não decide nada:
# é informação ao lado da primeira.
ROTULO_PELO_RSD: Final = "Pela estabilidade dos QCs"
ROTULO_PELA_VERDADE: Final = "Pela recuperação do sinal"

RESSALVA_PELA_VERDADE: Final = Caveat(
    resumo=(
        "opinião PELA RECUPERAÇÃO DO SINAL: treinada pelo corretor que mais recupera o sinal "
        "verdadeiro em 77 matrizes simuladas; em 170 matrizes simuladas novas acertou 132, contra "
        "69 da opinião pela estabilidade dos QCs (p = 9,6e-10) — e ela não decide o que corre"
    ),
    detalhe=(
        "**As duas opiniões respondem perguntas diferentes, e podem discordar.** A primeira, "
        "pela estabilidade dos QCs, prediz qual corretor deixa os QCs mais estáveis — o critério "
        "da literatura, e o que o automático segue. Esta prediz qual corretor deixa as amostras "
        "mais perto do sinal verdadeiro, e só pode ser treinada onde a verdade é conhecida: em "
        "matrizes simuladas. Nas 77 em que foi treinada, o corretor que mais estabiliza os QCs "
        "foi também o que mais recupera o sinal em só 28, e é por isso que ela existe.\n\n"
        "Avaliada em 192 matrizes simuladas novas, do mesmo gerador — 170 com um melhor corretor "
        "claro —, acertou o corretor de menor erro em 132, contra 69 da opinião pelo RSD, e a "
        "diferença não é acaso (McNemar p = 9,6e-10). Nas 78 em que não corrigir era o melhor, a "
        "opinião pelo RSD nunca o escolheu. **O que isto não diz:** a verdade só existe nas "
        "matrizes simuladas, e o gerador difere dos arquivos reais em parte das propriedades "
        "medidas; nos seus arquivos não há como saber qual das duas acerta. **Ela não muda o que "
        "corre**: para seguir esta opinião, destique «Automático» e escolha o corretor à mão."
    ),
    evidencia=(
        "Avaliação fora da amostra, em matrizes simuladas: 192 matrizes novas, 170 "
        "decidíveis, 132 contra 69 (IC de Wilson 0,708 a 0,833 e 0,335 a 0,481), McNemar 86 a 23, "
        "p=9,59e-10; a opinião pelo RSD escolhe none em 0 das 78 em que none é o melhor. Antes "
        "dela, uma versão treinada em 40 dava 27 de 41 contra 19 "
        "(p=0,1686), e o rótulo pelo RSD era igual ao corretor de menor erro em 28 das 77. "
        "O gerador foi comparado com os arquivos reais, e difere deles em parte das propriedades "
        "medidas"
    ),
    portao_inerte=(
        "nenhum critério de aceitação mede esta opinião; ela foi avaliada fora da "
        "amostra, em matrizes simuladas, e nos arquivos reais não há verdade para a medir"
    ),
)

METHOD_CAVEATS: dict[str, Caveat] = {
    "serrf": Caveat(
        resumo=(
            "SERRF: melhor corretor medido em 3 dos 4 modos reais; com a floresta que conta como "
            "a do R, remove de 91% a 159% do erro que o SERRF original remove, e a concordância "
            "número a número passa em 3 dos 4 modos"
        ),
        detalhe=(
            "**SERRF — o melhor corretor em 3 dos 4 modos deste estudo, com a floresta que conta "
            "como a do SERRF original em R.**\n\n"
            "**Modo a modo**, com um QC escondido de cada vez e o valor corrigido lido nele — RSD "
            "mediano fora da amostra, com remoção de outliers, e a margem relativa para o "
            "segundo colocado, em percentagem do melhor:\n\n"
            "- `pos_c18` — SERRF, 14,4% — margem 67,88%\n"
            "- `pos_hilic` — SERRF, 109,9% — margem 30,84%\n"
            "- `neg_c18` — SERRF, 15,1% — margem 2,56%, sobre não corrigir: quase empate, e ali "
            "corrigir tem tanta chance de piorar quanto de melhorar\n"
            "- `neg_hilic` — **LOESS**, 59,2% contra 61,5% do SERRF — margem 3,82%\n\n"
            "**Margem pequena é a medição a dizer que não sabe** — e esta régua elege sempre um "
            "vencedor, mesmo quando não há diferença que preste. Não corrigir não venceu em "
            "nenhum dos 4 arquivos do laboratório.\n\n"
            "**Estes números são deste estudo, não da sua matriz.** Quem decide pela sua é o "
            "botão «Medir qual corretor vence», logo acima, que refaz a conta com os seus QCs.\n\n"
            "**Comparando com uma saída do SERRF original em R, os números chegam perto e não "
            "batem no dígito.** Sob a mesma régua e na população inteira dos quatro modos, o "
            "nosso remove de 91% a 159% do erro sistemático que o do R remove — um pouco menos no "
            "`neg_hilic`, tanto ou mais nos outros três. A concordância número a número passa em "
            "3 dos 4 modos, todos menos o `pos_c18`, e o recorte inteiro está na aba «Métodos e "
            "limites».\n\n"
            "**Em compensação, este repete e o do R não:** o bloco SERRF do `tools.R` não fixa "
            "semente, então duas execuções dele sobre os mesmos dados não dão a mesma tabela."
        ),
        evidencia=(
            "Medido nos quatro arquivos reais do estudo, com o SERRF e o LOESS desta versão "
            "(no Windows; "
            "leave_one_qc_out_rsd, outlier_removed, a mediana sobre as features de "
            "RSD finito de cada um dos 4 reais): neg_hilic loess 0,5919 contra serrf 0,6145, "
            "margem relativa 3,82% do melhor; "
            "neg_c18 serrf 0,1512, margem relativa 2,56%; pos_c18 serrf 0,1438, margem relativa "
            "67,88%; pos_hilic serrf 1,0992, margem relativa 30,84%; none ótimo em 0 das 4 reais. "
            "Antes da última revisão do SERRF e do LOESS, as margens "
            "relativas eram 4,39%, 2,33%, 68,33% e 30,57%. Os dois SERRF sob a MESMA régua, "
            "população inteira "
            "(data/fixtures/regua_fora_da_amostra.json): 90,7% a 159,2% "
            "reproduzidos. A concordância estatística com o R passa no pos_hilic, no neg_hilic e "
            "no neg_c18 com o SERRF desta versão; antes da revisão, só no pos_hilic e no "
            "neg_c18. Antes ainda, com a "
            "floresta do scikit-learn: 15,1% a 29,3% reproduzidos, e a causa estava "
            "localizada na floresta. O determinismo é do nosso lado: mesma entrada e mesma "
            "semente dão a mesma saída"
        ),
        portao_inerte=(
            "a concordância estatística com o R no pos_c18; e a mediana de RSD pós-SERRF, "
            "in-sample, próxima da medida sobre a saída do R nos quatro modos"
        ),
    ),
    # A ressalva do LOESS é sobre o conteúdo do arquivo. A janela não expõe `null_denominator`,
    # e o padrão de `POST /matrizes/corrigir` e de `POST /matrizes/processar` é `r_faithful`,
    # sob o qual saem células infinitas. As features com a mediana da linha também zero saem
    # com células vazias nas duas políticas, porque o divisor é `0/0` e a guarda de
    # `uncorrected_feature` não o vê.
    "loess": Caveat(
        resumo=(
            "batch_loess: a matriz corrigida sai com células INFINITAS onde a curva ajustada "
            "vale zero, e com células vazias onde a mediana da feature também é zero — 109 a 763 "
            "infinitas e 208 a 2669 vazias por modo, medidas — e a janela não expõe a política "
            "que evita as infinitas"
        ),
        detalhe=(
            "O que está medido: no Windows, com semente 42 e um lote, sob "
            "`null_denominator=r_faithful` — que é o padrão de `refino corrigir`, de "
            "`POST /matrizes/corrigir` e de `POST /matrizes/processar` —, a matriz corrigida sai "
            "com 109 células infinitas no neg_c18, 552 no pos_c18, 448 no neg_hilic e 763 no "
            "pos_hilic. Sob `uncorrected_feature` as infinitas são ZERO nos quatro modos, e as "
            "vazias não mudam: 208, 676, 2076 e 2669 células, das features cuja mediana também é "
            "zero — ali o divisor é 0/0, e a guarda dessa política não o vê. "
            "Por que o padrão é esse: `r_faithful` existe para bater com os goldens do R, e a "
            "fidelidade ao R é decisão do projeto — dividir pela curva nula e produzir `Inf` é o "
            "que o tools.R faz. "
            "O que isso NÃO afeta: o arquivo do MetaboAnalyst. `refino exportar` e "
            "`POST /matrizes/exportar` têm padrão `uncorrected_feature`, e a escolha é medição e "
            "não gosto — o escritor daquele formato RECUSA matriz com infinito pelo `[F-2]`, e "
            "sob `r_faithful` a exportação reprovaria em 4 de 4 modos. "
            "O que fazer: a política é escolhível pela CLI "
            "(`--null-denominator uncorrected_feature`) e não por esta janela. "
            "Uma planilha que abra o CSV corrigido vai "
            "mostrar `inf` nessas células, e toda estatística sobre elas fica `nan` por "
            "`inf - inf`. "
            "O que o uso NÃO afirma: nada sobre a qualidade da correção do LOESS. A ressalva é "
            "só sobre o que o arquivo contém"
        ),
        evidencia=(
            "medido nos quatro modos reais do estudo; a guarda de infinito está em io/writers.py"
        ),
        portao_inerte="nenhum",
    ),
}


# As ressalvas do eixo de imputação. Segundo dicionário, e não entradas em `METHOD_CAVEATS`,
# cujas chaves são de `Correction`: um `METHOD_CAVEATS.get(metodo)` casaria um rótulo de
# imputação por acidente de nome.
#
# A do `missforest` é de custo. O seletor é derivado de `get_args(Imputation)`, então ele
# aparece sem que ninguém decida oferecê-lo, e o tempo dele faz a janela parecer travada.
IMPUTATION_CAVEATS: dict[str, Caveat] = {
    "missforest": Caveat(
        resumo=(
            "missforest: CUSTA DEZENAS DE MINUTOS e a janela parece travada enquanto corre — "
            "prefira `knn`, que custou 1,8 s sobre os mesmos dados"
        ),
        detalhe=(
            "Medido sobre o `neg_c18` real (2.184 features): 7,5 s em 25 "
            "features, 12,0 s em 50, 101 s em 100 e 322 s em 200. "
            "A curva é SUPERLINEAR, então extrapolar linearmente dá o PISO e não a estimativa: "
            "cerca de 59 min no `neg_c18` e 162 min no `pos_hilic` (~6.035 features). O custo real "
            "é maior que isso. "
            "Por que ele é assim: `IterativeImputer` ajusta uma floresta de 100 árvores "
            "POR FEATURE "
            "e repete até convergir — e sobre estes dados ele nem converge, o `sklearn` emite "
            "`ConvergenceWarning: Early stopping criterion not reached`. "
            "É por isso que ele fica FORA da grade padrão do teste de robustez: "
            "ele estourou 10 min sobre 758 features, e acrescentá-lo multiplicaria a grade por "
            "1,33x com todas as células novas a pagar aquele custo. "
            "O que o uso NÃO afirma: nada sobre a qualidade da imputação — ela não foi comparada "
            "com as outras quatro sobre verdade conhecida. A ressalva é só de tempo. "
            "Enquanto ele corre a janela NÃO congela — a API está noutra thread e «Ler progresso» "
            "responde —, mas não há etapa de progresso dentro da imputação, então a barra não anda."
        ),
        evidencia=(
            "medição de tempo sobre o `neg_c18` real; o `missforest` está fora da grade padrão do "
            "teste de robustez"
        ),
        portao_inerte="nenhum",
    ),
    "none": Caveat(
        resumo=(
            "none: não imputa, e por isso EXIGE que os três seletores abaixo fiquem em `none` "
            "— normalizar sobre ausência produz matriz que parece processada e não é"
        ),
        detalhe=(
            "`none` deixa as células ausentes como estão — no `neg_c18` são 13.051 de 148.512. "
            "Propagar ausência por normalização por amostra, transformação ou escalonamento produz "
            "uma matriz que PARECE processada e não é, e é por isso que a combinação é recusada na "
            "construção da especificação e não no executor. "
            "O que fazer: ou deixe `Normalização por amostra`, `Transformação` e `Escalonamento` "
            "em `none` — que é o caminho que entrega a matriz como ela está, corrigida mas não "
            "normalizada —, ou escolha um método de imputação. "
            "`one_fifth_min` é o padrão e é o que a literatura da área usa para MNAR: o não "
            "detectado está abaixo do limite de detecção, então imputar por um valor pequeno é "
            "mais fiel que imputar pela média."
        ),
        evidencia="13.051 ausentes medidos no neg_c18",
        portao_inerte="nenhum",
    ),
}


@dataclass(frozen=True)
class Options:
    """O que a CLI leu, já validado. Espelha um-para-um as opções que afetam a saída.

    É o que torna `--log` realimentável: campo que afete a saída e não esteja aqui não entra
    no log, e o log deixa de reproduzir a execução sem que nada acuse.
    """

    measurements: Path
    sample_list: Path
    mode: str
    method: Correction
    # `None` significa que o automático não foi pedido; um rótulo diz qual política decide.
    auto: str | None
    rsd_variant: RsdVariant
    null_denominator: NullDenominatorPolicy
    orientation: Orientation
    seed: int
    output: Path
    log: Path | None


def _add_vocabulary_argument(
    parser: argparse.ArgumentParser, flag: str, kind: str, default: str, help_text: str
) -> None:
    """Adiciona opção de vocabulário sem `choices=`.

    Com `choices=get_args(...)` o `argparse` recusaria o valor inválido antes de
    `validate_label` ser chamado, e essa guarda ficaria escrita, nunca executada e impossível
    de exercitar por teste. Fica uma guarda só, e o `help` também sai de `VOCABULARY`, para o
    texto e o conjunto não divergirem.
    """
    valores = ", ".join(VOCABULARY[kind])
    # A ressalva entra no `--help`, que é um dos pontos de escolha: quem o lê está a escolher
    # antes de rodar. O outro é a execução, para quem roda sem ler.
    ressalvas = "".join(
        f" ATENÇÃO em `{nome}`: {METHOD_CAVEATS[nome].resumo}."
        for nome in VOCABULARY[kind]
        if nome in METHOD_CAVEATS
    )
    # O `%` vai escapado, porque o `argparse` formata o `help` com `%` e a ressalva do `serrf`
    # traz percentagens.
    texto = f"{help_text} Um de: {valores}. Padrão: {default}.{ressalvas}"
    parser.add_argument(flag, default=default, metavar="VALOR", help=texto.replace("%", "%%"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="refino",
        description="Refino quimiométrico de dados de metabolômica LC-MS.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    corrigir = sub.add_parser(
        "corrigir",
        help="corrige deriva de sinal e escreve a matriz corrigida",
        description=(
            "Lê measurements + Sample_List, corrige a deriva pelo método pedido e "
            "escreve a matriz no formato toserrf."
        ),
    )
    corrigir.add_argument("--measurements", required=True, type=Path)
    corrigir.add_argument("--sample-list", required=True, type=Path)
    corrigir.add_argument("--mode", required=True, help="rótulo do modo analítico, ex.: pos_hilic")
    corrigir.add_argument("--output", required=True, type=Path)
    corrigir.add_argument(
        "--log", type=Path, default=None, help="JSON com especificação, contagens e medições"
    )
    corrigir.add_argument("--seed", type=int, default=42)

    _add_vocabulary_argument(
        corrigir, "--method", "Correction", "none", "Método de correção de sinal."
    )
    # O modo automático fica em `corrigir` porque a decisão da política é qual corretor usar, e
    # o `processar` não aplica corretor nenhum.
    #
    # `nargs="?"` com `const` dá os quatro desfechos: ausente, quem escolhe é o usuário;
    # `--auto` sem valor é a política aprendida; `--auto constante` é a que sempre escolhe
    # `serrf`; `--auto medida` é o Palpite medido em 200 features. Um valor fora dos três é
    # recusado pela guarda de `parse_options`.
    corrigir.add_argument(
        "--auto",
        nargs="?",
        const=AUTO_APRENDIDA,
        default=None,
        metavar="{aprendida,constante,medida}",
        help=(
            "delega a escolha do corretor. Sem valor usa a política aprendida; `constante` usa a "
            "que sempre escolhe `serrf`; `medida` mede o RSD dos QCs fora da amostra em 200 "
            "features e escolhe o menor (perto de 1 minuto com 12 núcleos e 7 QCs; mais com mais "
            "QCs). A ressalva medida vai "
            "para o stderr e para o --log."
        ),
    )
    _add_vocabulary_argument(
        corrigir, "--rsd-variant", "RsdVariant", "outlier_removed", "Fórmula do RSD reportado."
    )
    _add_vocabulary_argument(
        corrigir,
        "--null-denominator",
        "NullDenominatorPolicy",
        "r_faithful",
        "Política do batch_loess diante de denominador nulo.",
    )
    _add_vocabulary_argument(
        corrigir,
        "--orientation",
        "Orientation",
        "features-in-rows",
        "Orientação do arquivo de ENTRADA, não a interna.",
    )

    _build_exportar(sub)

    processar = sub.add_parser(
        "processar",
        help="pipeline completo pós-correção: filtro, imputação e normalização",
        description=(
            "Aplica as seis etapas da ordem canônica e escreve a matriz processada. "
            "TODOS os parâmetros são explícitos: a escolha automática não é deste subcomando, e um "
            "padrão aqui seria decisão tomada por omissão e sem registro."
        ),
    )
    processar.add_argument("--measurements", type=Path)
    processar.add_argument("--sample-list", type=Path)
    processar.add_argument("--mode", help="rótulo do modo analítico, ex.: pos_hilic")
    processar.add_argument("--output", type=Path)
    processar.add_argument(
        "--log", type=Path, default=None, help="JSON reexecutável com contagem por etapa"
    )
    processar.add_argument(
        "--from-log",
        type=Path,
        default=None,
        help=(
            "reexecuta a partir de um log. O log é ENTRADA e não relatório: "
            "realimentá-lo produz saída idêntica."
        ),
    )
    processar.add_argument("--seed", type=int, default=42)
    _add_vocabulary_argument(
        processar, "--rsd-variant", "RsdVariant", "outlier_removed", "Fórmula do RSD do filtro."
    )
    _add_eixos_do_pre_processamento(processar, imputacao_padrao="one_fifth_min")
    _add_vocabulary_argument(
        processar,
        "--orientation",
        "Orientation",
        "features-in-rows",
        "Orientação do arquivo de ENTRADA, não a interna.",
    )
    return parser


def _add_eixos_do_pre_processamento(
    parser: argparse.ArgumentParser, *, imputacao_padrao: str
) -> None:
    """As opções do pré-processamento, partilhadas por `processar` e `exportar`.

    Uma definição para os dois subcomandos, para um eixo novo não entrar num e faltar no outro.
    Só a imputação muda de padrão: `one_fifth_min` no `processar` e `none` na exportação, pelo
    motivo que `_main_exportar` declara.
    """
    parser.add_argument(
        "--rsd-cutoff",
        type=float,
        default=None,
        help="corte de RSD em FRAÇÃO; omitido salta a etapa",
    )
    parser.add_argument(
        "--missing-threshold",
        type=float,
        default=None,
        help="fração máxima de ausentes DENTRO de cada grupo; omitido salta a etapa",
    )
    _add_vocabulary_argument(
        parser,
        "--ranking-filter",
        "RankingFilter",
        "none",
        "Redução de features por posto, primeiro passe. `none` salta a etapa.",
    )
    parser.add_argument(
        "--ranking-keep-fraction",
        type=float,
        default=None,
        metavar="FRACAO",
        help=(
            "Fração RETIDA pelo primeiro passe, em (0, 1] — 0,6 mantém as 60%% de cima. "
            "Omitida, vale a regra por tamanho da matriz, que é o que sempre valeu."
        ),
    )
    _add_vocabulary_argument(
        parser,
        "--ranking-filter-2",
        "RankingFilter",
        "none",
        "Redução por posto, segundo passe, sobre o que o primeiro deixou.",
    )
    parser.add_argument(
        "--normalization-factors",
        type=Path,
        default=None,
        metavar="CSV",
        help=(
            "CSV com duas colunas — nome da amostra e fator — para `supplied_factors`. O arquivo "
            "e INSUMO DE FRONTEIRA: ele e lido aqui e o MAPA resolvido entra no registro, para "
            "que reexecutar reproduza a saida mesmo que o arquivo mude depois."
        ),
    )
    parser.add_argument(
        "--normalization-reference",
        default=None,
        metavar="NOME",
        help=(
            "Nome da amostra ou da feature contra a qual normalizar. Exigido pelos metodos de "
            "referencia e ignorado pelos outros; o nome e conferido contra a matriz ingerida."
        ),
    )
    parser.add_argument(
        "--ranking-keep-fraction-2",
        type=float,
        default=None,
        metavar="FRACAO",
        help="Fração RETIDA pelo segundo passe, em (0, 1]. Ver `--ranking-keep-fraction`.",
    )
    _add_vocabulary_argument(
        parser, "--imputation", "Imputation", imputacao_padrao, "Tratamento de ausentes."
    )
    _add_vocabulary_argument(
        parser,
        "--sample-normalization",
        "SampleNormalization",
        "none",
        "Normalização por amostra (linhas).",
    )
    _add_vocabulary_argument(
        parser, "--transformation", "Transformation", "none", "Transformação célula a célula."
    )
    _add_vocabulary_argument(
        parser, "--scaling", "Scaling", "none", "Escalonamento por feature (colunas)."
    )


def parse_options(namespace: argparse.Namespace) -> Options:
    """Valida os rótulos em runtime e devolve as opções.

    Levanta `InvalidLabelError`, que `main` traduz em código 2. A validação é aqui e não no
    corretor, para recusar um argumento inválido antes de gastar a leitura da matriz.
    """
    validate_label("Correction", namespace.method)
    validate_label("RsdVariant", namespace.rsd_variant)
    validate_label("NullDenominatorPolicy", namespace.null_denominator)
    validate_label("Orientation", namespace.orientation)
    # `AUTO_VALIDOS` é fechado, e o `argparse` com `nargs="?"` aceita qualquer string. Sem esta
    # recusa, `--auto constant` cairia num `else` razoável e correria a política errada, com o
    # log a declarar a que o usuário não pediu. `choices=` não é usado porque a recusa sairia
    # como `usage:` do `argparse`, sem dizer o que cada rótulo significa.
    if namespace.auto is not None and namespace.auto not in AUTO_VALIDOS:
        raise InvalidLabelError(
            f"--auto {namespace.auto} não é um valor válido. "
            f"Sem valor usa a política aprendida; os rótulos são: {', '.join(AUTO_VALIDOS)}."
        )
    return Options(
        measurements=namespace.measurements,
        sample_list=namespace.sample_list,
        mode=namespace.mode,
        method=namespace.method,
        rsd_variant=namespace.rsd_variant,
        null_denominator=namespace.null_denominator,
        orientation=namespace.orientation,
        seed=namespace.seed,
        output=namespace.output,
        log=namespace.log,
        auto=namespace.auto,
    )


def build_corrector(method: str, null_denominator: NullDenominatorPolicy) -> Corrector:
    """Instancia o corretor, ou levanta se o método é válido e não implementado.

    A guarda de `UNAVAILABLE_METHODS` fica mesmo com a tabela vazia: é o ponto de recusa se
    algum método voltar a ser bloqueado.

    O `serrf` recebe `n_jobs=-1` porque esta é a chamada que processa matriz real. O padrão da
    classe é `1`, para as matrizes pequenas dos testes, em que criar processos custa mais que o
    trabalho.

    Recebe os dois valores e não o `Options` inteiro porque o `exportar` também corrige e não
    constrói `Options`.
    """
    if method in UNAVAILABLE_METHODS:
        raise MethodUnavailableError(method, UNAVAILABLE_METHODS[method].reason)
    if method == "loess":
        return LoessCorrector(null_denominator=null_denominator)
    if method == "serrf":
        return SerrfNaVez(n_jobs=-1)
    return AVAILABLE_METHODS[method]()


def emite_ressalva(method: str, stream: TextIO) -> bool:
    """Escreve a ressalva do método no ponto de execução. Devolve se escreveu.

    Vai para `stderr`: o `stdout` de `corrigir` é consumido por quem redireciona, e um método
    com ressalva não é erro, então o código de saída não muda.

    Devolve `bool` para o teste distinguir os dois desfechos.
    """
    ressalva = METHOD_CAVEATS.get(method)
    if ressalva is None:
        return False
    print(f"ATENÇÃO: {ressalva.resumo}", file=stream)
    print(f"  {ressalva.detalhe}", file=stream)
    print(f"  Evidência: {ressalva.evidencia}", file=stream)
    print(f"  Portão ainda inerte: {ressalva.portao_inerte}", file=stream)
    return True


def emite_ressalva_do_auto(auto: str, stream: TextIO) -> bool:
    """Escreve a ressalva da política no ponto de execução. Devolve se escreveu.

    Irmã de `emite_ressalva` e não um ramo dentro dela: os domínios são disjuntos, e uma função
    que recebesse um método ou uma política teria de decidir qual tabela consultar pela forma
    da string.
    """
    ressalva = AUTO_CAVEATS.get(auto)
    if ressalva is None:  # pragma: no cover - `parse_options` recusa o que não está na tabela
        return False
    print(f"ATENÇÃO: {ressalva.resumo}", file=stream)
    print(f"  {ressalva.detalhe}", file=stream)
    print(f"  Evidência: {ressalva.evidencia}", file=stream)
    print(f"  Portão ainda inerte: {ressalva.portao_inerte}", file=stream)
    return True


class MethodUnavailableError(RuntimeError):
    """Método do vocabulário, válido, sem implementação nesta versão.

    Exceção própria porque não é argumento inválido: o usuário não errou de digitação, a
    ferramenta ainda não faz aquilo. O `corrigir` a traduz em código 3.
    """

    def __init__(self, method: str, reason: str) -> None:
        super().__init__(
            f"--method {method} ainda não está disponível: {reason}. "
            f"Disponíveis: {', '.join(sorted(AVAILABLE_METHODS))}."
        )
        self.method = method


def median_qc_rsd(matrix: RefinoMatrix, variant: RsdVariant) -> float:
    """Mediana, sobre as features definidas, do RSD nos QCs. Em pontos percentuais.

    As indefinidas ficam de fora: `np.median` sobre vetor com `nan` devolve `nan`, e uma
    feature sem RSD apagaria a medição de todas as outras.
    """
    values = rsd(matrix.values[matrix.qc_mask], axis=0, variant=variant)
    finite = values[np.isfinite(values)]
    return float(np.median(finite) * 100.0) if finite.size else float("nan")


def build_log(
    options: Options,
    matrix: RefinoMatrix,
    result: CorrectionResult,
    *,
    escolhido: str | None = None,
    recomendacao: Recommendation | None = None,
    pela_verdade: OpiniaoPelaVerdade | None = None,
) -> dict[str, Any]:
    """Monta o log. O critério do que entra é que realimentá-lo reproduz a saída.

    `spec` carrega exatamente as opções que afetam o resultado, com os nomes das opções de
    linha de comando. As contagens por causa entram separadas e não somadas, porque cada causa
    diz algo diferente sobre o dado.

    Sob `--auto`, `spec.method` é o método que correu, e não o que o usuário digitou:
    realimentar `none` porque era o padrão do argumento produziria outra matriz. O que foi
    pedido fica em `auto.rotulo`, e a delegação inteira em `auto`.
    """
    before = median_qc_rsd(matrix, options.rsd_variant)
    after = median_qc_rsd(result.matrix, options.rsd_variant)
    metodo = options.method if escolhido is None else escolhido
    return {
        "spec": {
            "measurements": str(options.measurements),
            "sample-list": str(options.sample_list),
            "mode": options.mode,
            "method": metodo,
            "rsd-variant": options.rsd_variant,
            "null-denominator": options.null_denominator,
            "orientation": options.orientation,
            "seed": options.seed,
        },
        "shape": {
            "n_features": matrix.n_features,
            "n_samples": matrix.n_samples,
            "n_qc": int(matrix.qc_mask.sum()),
            "batches": sorted(set(matrix.batches)),
        },
        "uncorrected": {
            "n_features": int(result.uncorrected_mask.sum()),
            "by_cause": dict(result.counts_by_cause()),
            "digest": result.uncorrected_digest(),
        },
        "median_qc_rsd_pct": {
            "before": before,
            "after": after,
            "variant": options.rsd_variant,
            "estimator": "in_sample",
            "note": (
                "in_sample e nada mais. O in-sample favorece quem ajusta a curva nos "
                "proprios QCs, entao este numero NAO compara metodos — os dois "
                "estimadores fora da amostra nao entram neste campo."
            ),
        },
        # A ressalva entra no log, que é o que sobrevive à sessão: quem lê um resultado meses
        # depois não viu o `stderr` nem o `--help`. Chave ausente quando não há ressalva, e não
        # `null`, que obrigaria quem lê a distinguir a falta de ressalva de um campo não
        # preenchido.
        **(
            {
                "caveat": {
                    "method": metodo,
                    "resumo": METHOD_CAVEATS[metodo].resumo,
                    "detalhe": METHOD_CAVEATS[metodo].detalhe,
                    "evidencia": METHOD_CAVEATS[metodo].evidencia,
                    "portao_inerte": METHOD_CAVEATS[metodo].portao_inerte,
                }
            }
            if metodo in METHOD_CAVEATS
            else {}
        ),
        # A delegação e a justificativa da escolha automática. Chave ausente quando não houve
        # delegação, pelo mesmo motivo da `caveat`.
        **(
            {
                "auto": {
                    "rotulo": options.auto,
                    # O rótulo validado é a política: derivar um do outro por `if` faria
                    # `medida` gravar-se como `aprendida`.
                    "politica": options.auto,
                    "correction": recomendacao.correction,
                    "confidence": recomendacao.confidence,
                    "abstains": recomendacao.abstains,
                    "assessable": recomendacao.assessable,
                    "rationale": recomendacao.rationale,
                    "evidence": dict(recomendacao.evidence),
                    "caveat": {
                        "resumo": AUTO_CAVEATS[options.auto].resumo,
                        "detalhe": AUTO_CAVEATS[options.auto].detalhe,
                        "evidencia": AUTO_CAVEATS[options.auto].evidencia,
                        "portao_inerte": AUTO_CAVEATS[options.auto].portao_inerte,
                    },
                    # A segunda opinião vai ao log ao lado da que decidiu, com o rótulo dela.
                    # Chave ausente quando não foi consultada.
                    **(
                        {
                            "segunda_opiniao": {
                                "rotulo": ROTULO_PELA_VERDADE,
                                "correction": pela_verdade.correction,
                                "caminho": pela_verdade.caminho,
                                "caveat": {
                                    "resumo": RESSALVA_PELA_VERDADE.resumo,
                                    "detalhe": RESSALVA_PELA_VERDADE.detalhe,
                                    "evidencia": RESSALVA_PELA_VERDADE.evidencia,
                                    "portao_inerte": RESSALVA_PELA_VERDADE.portao_inerte,
                                },
                            }
                        }
                        if pela_verdade is not None
                        else {}
                    ),
                }
            }
            if recomendacao is not None and options.auto is not None
            else {}
        ),
        "provenance": list(result.matrix.provenance),
    }


def politica_do_rotulo(auto: str) -> DecisionPolicy:
    """Traduz o rótulo já validado na política que decide.

    Função separada de `run` porque é a única parte do `--auto` que não precisa de matriz, de
    disco nem de corretor, e assim cada rótulo é testável numa linha. Não valida: quem valida
    é `parse_options`, na fronteira.
    """
    if auto == AUTO_CONSTANTE:
        return AlwaysSERRFPolicy()
    if auto == AUTO_MEDIDA:
        # O Palpite medido não é política de perfil: precisa da matriz, e quem o mede é
        # `recomenda`. Devolver a árvore aqui faria `medida` correr outra coisa sem levantar.
        raise ValueError("`medida` não é política de perfil: use `recomenda(matriz, 'medida')`")
    return carrega_politica_aprendida()


def recomenda(
    matrix: RefinoMatrix,
    auto: str,
    *,
    batches: np.ndarray | None = None,
    n_jobs: int = -1,
    seed: int = 42,
) -> Recommendation:
    """Diagnostica a matriz e consulta a política.

    Fica aqui e não em `parse_options` porque a política recebe um `DefectProfile`, e o perfil
    só existe depois de a matriz ser lida. `parse_options` corre antes de qualquer leitura,
    para recusar um rótulo inválido sem gastar o tempo do arquivo.
    """
    # O perfil é calculado com a mesma política de diagnóstico com que o corpus de treino da
    # árvore foi diagnosticado. Esta função serve a CLI e a delegação da API, então as duas
    # mudam juntas.
    #
    # `medida` não passa pelo perfil. Ela mede a matriz: o RSD dos QCs fora da amostra em 200
    # features, na partição `batches` e com `n_jobs` núcleos.
    if auto == AUTO_MEDIDA:
        return mede_o_palpite(matrix, batches=batches, n_jobs=n_jobs, seed=seed).como_recomendacao()
    return politica_do_rotulo(auto).recommend(perfil_para_a_politica(matrix))


@dataclass(frozen=True)
class OpiniaoPelaVerdade:
    """O que a segunda opinião diz: o corretor e o caminho da árvore.

    Sem confiança: a margem desta árvore é de erro contra a verdade, e o limiar que a faria
    confiança foi derivado para o RSD.
    """

    correction: str
    caminho: str


def segunda_opiniao(matrix: RefinoMatrix) -> OpiniaoPelaVerdade:
    """O Palpite pela verdade sobre a mesma matriz: a opinião que não decide.

    O perfil é o mesmo que `recomenda` serve à primeira, e é isso que torna as duas
    comparáveis.
    """
    politica = carrega_politica_pela_verdade()
    perfil = perfil_para_a_politica(matrix)
    return OpiniaoPelaVerdade(
        correction=politica.recommend(perfil).correction,
        caminho=politica.caminho_do_corretor(perfil),
    )


def emite_segunda_opiniao(pela_verdade: OpiniaoPelaVerdade, stream: TextIO) -> None:
    """Uma linha com a segunda opinião e a ressalva dela, no ponto em que a primeira já decidiu."""
    print(
        f"{ROTULO_PELA_VERDADE} (segunda opinião, não decide): {pela_verdade.correction}. "
        f"{RESSALVA_PELA_VERDADE.resumo}",
        file=stream,
    )


def run(options: Options, avisos: TextIO | None = None) -> tuple[int, dict[str, Any] | None]:
    """Executa e devolve `(código, log)`. Separado de `main` para ser testável.

    `main` cuida de `argv`, `stderr` e código de saída; `run` recebe opções já validadas, e o
    teste inspeciona o log sem passar por `subprocess`.

    Com `--auto` a ordem é ler a matriz, diagnosticar, escolher e corrigir: é a matriz que
    decide qual corretor construir.
    """
    matrix = _le_matriz(
        options.measurements, options.sample_list, options.mode, options.orientation
    )
    recomendacao = (
        None if options.auto is None else recomenda(matrix, options.auto, seed=options.seed)
    )
    # A segunda opinião só existe onde houve delegação: sem `--auto` o usuário escolheu o
    # corretor, e não há palpite a pôr ao lado. O interruptor dela fica em `io.politica` e é
    # lido aqui, na chamada.
    pela_verdade = (
        segunda_opiniao(matrix) if recomendacao is not None and segunda_opiniao_ativa() else None
    )
    metodo = options.method if recomendacao is None else recomendacao.correction
    # A ressalva do método efetivo só é conhecível aqui. Sem `--auto` não sai nada, porque
    # `main` já a emitiu antes da leitura. `avisos` é opcional para o teste poder chamar `run`
    # sem stream.
    if avisos is not None and recomendacao is not None:
        emite_ressalva(metodo, avisos)
    if avisos is not None and pela_verdade is not None:
        emite_segunda_opiniao(pela_verdade, avisos)
    corrector = build_corrector(metodo, options.null_denominator)
    result = corrector.fit_transform(matrix, seed=options.seed)
    write_toserrf(result.matrix, options.output)
    return EXIT_OK, build_log(
        options,
        matrix,
        result,
        escolhido=metodo,
        recomendacao=recomendacao,
        pela_verdade=pela_verdade,
    )


def main(argv: Sequence[str] | None = None) -> int:
    # Antes da primeira linha impressa. As mensagens desta CLI são acentuadas, e num Windows
    # em `cp932` a saída em tubo não codifica o cedilha: sem isto, `exportar` grava o arquivo
    # inteiro e sai com código 1. Chamada aqui porque é `main` que `[project.scripts]` aponta;
    # tolera saída ausente, que é o caso do executável sem console.
    garante_utf8()
    parser = build_parser()
    namespace = parser.parse_args(argv)

    if namespace.command == "processar":
        return _main_processar(namespace)
    if namespace.command == "exportar":
        return _main_exportar(namespace)

    try:
        options = parse_options(namespace)
    except InvalidLabelError as error:
        print(f"erro: {error}", file=sys.stderr)
        return EXIT_USAGE

    # Antes de correr: o `serrf` numa matriz real leva minutos, e um aviso que só aparece no
    # fim chega depois de a escolha já ter custado o tempo dela.
    #
    # Sob `--auto` sai a ressalva da política e não a do método, que ainda não foi escolhido.
    # A do método efetivo sai dentro de `run`, depois do diagnóstico.
    if options.auto is None:
        emite_ressalva(options.method, sys.stderr)
    else:
        emite_ressalva_do_auto(options.auto, sys.stderr)

    try:
        code, log = run(options, sys.stderr)
    except ArtefatoInvalidoError as error:
        print(f"erro: {error}", file=sys.stderr)
        return EXIT_UNAVAILABLE
    except MethodUnavailableError as error:
        print(f"erro: {error}", file=sys.stderr)
        return EXIT_UNAVAILABLE
    except IngestionError as error:
        # A lista completa, nunca só o primeiro: quem tem um arquivo com quatro defeitos
        # precisa de quatro linhas, e não de quatro execuções.
        print(f"erro: {options.measurements.name} recusado pela validação:", file=sys.stderr)
        for issue in error.issues:
            print(f"  [{issue.severity}] {issue.code}: {issue.message}", file=sys.stderr)
        return EXIT_INPUT_REJECTED
    except EntradaIlegivelError as error:
        print(f"erro: a entrada não pôde ser lida — {error}", file=sys.stderr)
        return EXIT_INPUT_REJECTED
    except EstruturaSemPopulacaoError as error:
        # `--auto` diagnostica antes de escolher, e o diagnóstico mede estrutura: uma matriz
        # com um QC é aceita pela validação, com aviso, e não sustenta a medida.
        print(f"erro: a matriz não sustenta o diagnóstico — {error}", file=sys.stderr)
        return EXIT_INPUT_REJECTED
    except IntensidadeNegativaError as error:
        # É um fato do arquivo, então o código é o da entrada recusada.
        print(f"erro: a matriz não sustenta o SERRF — {error}", file=sys.stderr)
        return EXIT_INPUT_REJECTED
    except (IsADirectoryError, PermissionError) as error:
        # A entrada já passou por `_le_matriz`; uma pasta aqui é o `--output`, e o erro é de uso.
        if not _alvo_e_pasta(error):
            raise
        print(f"erro: `--output` aponta para uma pasta — {error}", file=sys.stderr)
        return EXIT_USAGE
    except DegenerateCrossValidationError as error:
        # Só `--auto medida` chega aqui. O Palpite medido esconde um QC de cada vez, e com
        # menos de três QCs num lote não sobra treino: a régua levanta em vez de devolver
        # número. A matriz é corrigível, só não admite esta decisão, e a mensagem diz a saída
        # que resta.
        print(
            f"erro: o Palpite medido não mede esta matriz — {error} "
            "Use `--auto aprendida`, ou escolha o corretor com `--method`.",
            file=sys.stderr,
        )
        return EXIT_INPUT_REJECTED

    if options.log is not None and log is not None:
        codigo_do_log = _grava_log(options.log, log)
        if codigo_do_log != EXIT_OK:
            return codigo_do_log
    return code


def _alvo_e_pasta(error: OSError) -> bool:
    """A escrita falhou porque o alvo é uma pasta?

    Ao abrir uma pasta para escrita, o Linux levanta `IsADirectoryError` e o Windows levanta
    `PermissionError`. Só que `PermissionError` também é o que o Windows levanta quando o
    arquivo de saída está aberto noutro programa, como o CSV aberto no Excel. Por isso ele só
    conta quando o caminho do erro é de fato uma pasta; fora disso quem chama relança o erro.
    """
    if isinstance(error, IsADirectoryError):
        return True
    return error.filename is not None and Path(error.filename).is_dir()


def _grava_log(caminho: Path, log: dict[str, Any]) -> int:
    """Grava o log e devolve o código: `EXIT_OK`, ou `EXIT_USAGE` quando `--log` é uma pasta.

    O log é escrito depois da saída principal, fora do `try` que traduz as falhas da execução.
    A mensagem diz o que ficou feito, porque a saída principal já está no disco.
    """
    try:
        caminho.parent.mkdir(parents=True, exist_ok=True)
        caminho.write_text(json.dumps(log, indent=2, ensure_ascii=False) + "\n", "utf-8")
    except (IsADirectoryError, PermissionError) as error:
        if not _alvo_e_pasta(error):
            raise
        print(
            f"erro: `--log` aponta para uma pasta — {error}. A saída foi escrita; o log não.",
            file=sys.stderr,
        )
        return EXIT_USAGE
    return EXIT_OK


def _vocabulary_is_partitioned() -> bool:
    """As duas tabelas de métodos cobrem o vocabulário? Usado pelo teste de paridade."""
    return set(AVAILABLE_METHODS) | set(UNAVAILABLE_METHODS) == set(get_args(Correction))


# ---------------------------------------------------------------------------
# `processar`: o pipeline completo pós-correção
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProcessOptions:
    """O que o `processar` leu, já validado. Espelha um-para-um o que afeta a saída.

    O mesmo critério de `Options`: campo que afete a saída e não esteja aqui não entra no log,
    e o log deixa de reproduzir a execução sem que nada acuse.
    """

    measurements: Path
    sample_list: Path
    mode: str
    orientation: Orientation
    spec: PipelineSpec
    output: Path
    log: Path | None


def parse_process_options(namespace: argparse.Namespace) -> ProcessOptions:
    """Valida na fronteira e constrói a especificação.

    `PipelineSpec` valida os rótulos e recusa a combinação proibida na construção, então esta
    função não repete nenhuma das duas. A validação vem antes de carregar a matriz, para não
    gastar a leitura com um argumento que já era inválido na linha de comando.
    """
    if namespace.from_log is not None:
        return _process_options_from_log(namespace)

    faltando = [
        flag
        for flag, valor in (
            ("--measurements", namespace.measurements),
            ("--sample-list", namespace.sample_list),
            ("--mode", namespace.mode),
            ("--output", namespace.output),
        )
        if valor is None
    ]
    if faltando:
        raise UsageError(
            f"faltam argumentos obrigatórios: {', '.join(faltando)}. "
            "Use `--from-log` para reexecutar a partir de um log."
        )

    validate_label("Orientation", namespace.orientation)
    spec = _spec_dos_eixos(namespace, correction="none")
    return ProcessOptions(
        measurements=namespace.measurements,
        sample_list=namespace.sample_list,
        mode=namespace.mode,
        orientation=namespace.orientation,
        spec=spec,
        output=namespace.output,
        log=namespace.log,
    )


def _spec_dos_eixos(namespace: argparse.Namespace, *, correction: Correction) -> PipelineSpec:
    """A `PipelineSpec` das opções de `_add_eixos_do_pre_processamento`, com o corretor dado.

    Uma construção só, para que `processar` e `exportar` leiam os mesmos campos. Um lote só,
    porque nenhum dos dois expõe `--n-batches`.
    """
    return PipelineSpec(
        correction=correction,
        n_batches=1,
        rsd_cutoff=namespace.rsd_cutoff,
        ranking_filter=namespace.ranking_filter,
        ranking_keep_fraction=namespace.ranking_keep_fraction,
        ranking_filter_2=namespace.ranking_filter_2,
        ranking_keep_fraction_2=namespace.ranking_keep_fraction_2,
        rsd_variant=namespace.rsd_variant,
        missing_threshold=namespace.missing_threshold,
        imputation=namespace.imputation,
        sample_normalization=namespace.sample_normalization,
        normalization_reference=namespace.normalization_reference,
        normalization_factors=_le_fatores(namespace.normalization_factors),
        transformation=namespace.transformation,
        scaling=namespace.scaling,
        seed=namespace.seed,
    )


def _le_fatores(caminho: Path | None) -> tuple[tuple[str, float], ...] | None:
    """Lê o CSV de fatores e devolve o mapa resolvido.

    O que entra em `PipelineSpec` é o conteúdo do arquivo, e não o caminho: com o caminho, a
    reexecução a partir do log só reproduziria a saída enquanto o arquivo não mudasse.

    A análise do texto vive em `refino.io.fatores`. Esta função decide de onde vem o texto e
    em que o erro se traduz, que aqui é `UsageError` e por isso código 2. A guarda de valor
    (positivo, finito, sem duplicados) fica em `PipelineSpec`, para valer também para quem
    chega pela rota ou pelo log.
    """
    if caminho is None:
        return None
    try:
        bruto = caminho.read_bytes()
    except OSError as erro:
        # `OSError` e não `FileNotFoundError`: o caminho pode ser um diretório, pode não ter
        # permissão, pode ser um dispositivo.
        raise UsageError(
            f"nao consegui ler `--normalization-factors` em {caminho}: {erro}"
        ) from erro
    try:
        return le_fatores_de_bytes(bruto, origem=str(caminho))
    except FatoresMalformadosError as erro:
        raise UsageError(str(erro)) from erro


def _fatores_do_log(bruto: object) -> tuple[tuple[str, float], ...] | None:
    """Reconstrói o mapa de fatores a partir do log, que é entrada e não relatório."""
    if bruto is None:
        return None
    if not isinstance(bruto, list):
        raise UsageError(
            f"`normalization-factors` no log tem de ser lista de pares e e {type(bruto).__name__}."
        )
    pares: list[tuple[str, float]] = []
    for item in bruto:
        if not isinstance(item, list | tuple) or len(item) != 2:
            raise UsageError(f"`normalization-factors` tem entrada malformada: {item!r}.")
        pares.append((str(item[0]), float(item[1])))
    return tuple(pares)


class UsageError(ValueError):
    """Argumento faltando ou log malformado. Traduzido em código 2 por `main`."""


_CAMPOS_DO_LOG = (
    "measurements",
    "sample-list",
    "mode",
    "orientation",
    "rsd-cutoff",
    "rsd-variant",
    "missing-threshold",
    "imputation",
    "sample-normalization",
    "transformation",
    "scaling",
    "seed",
)


def _process_options_from_log(namespace: argparse.Namespace) -> ProcessOptions:
    """Reconstrói as opções a partir de um log.

    O log é entrada: editado à mão, é dado vindo de disco, e nada ali passou pelo mypy. Campo
    ausente recusa em vez de assumir padrão, porque preencher em silêncio faria a reexecução
    divergir da original com o log parecendo íntegro.
    """
    # As falhas de leitura do log viram erro de uso: arquivo ausente, JSON que não é objeto e,
    # mais abaixo, campo com o tipo errado.
    try:
        bruto = json.loads(namespace.from_log.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as erro:
        raise UsageError(
            f"{namespace.from_log}: o log não pôde ser lido ({type(erro).__name__}: {erro})"
        ) from erro
    if not isinstance(bruto, dict):
        raise UsageError(
            f"{namespace.from_log}: o log é {type(bruto).__name__} e tem de ser um objeto JSON."
        )
    spec_do_log = bruto.get("spec")
    if not isinstance(spec_do_log, dict):
        raise UsageError(f"{namespace.from_log}: sem objeto `spec`; não é log de `processar`.")

    ausentes = [campo for campo in _CAMPOS_DO_LOG if campo not in spec_do_log]
    if ausentes:
        raise UsageError(
            f"{namespace.from_log}: campos ausentes em `spec`: {', '.join(ausentes)}. "
            "Preencher com padrão faria a reexecução divergir da original com o log "
            "parecendo íntegro."
        )

    validate_label("Orientation", spec_do_log["orientation"])
    # A semente é conferida aqui porque `PipelineSpec` não recusa semente de outro tipo: um log
    # com `"seed": "quarenta"` só falharia no meio da execução, com `TypeError`, e só quando
    # alguma etapa sorteasse. `bool` desce de `int` e fica de fora pelo nome.
    semente = spec_do_log["seed"]
    if not isinstance(semente, int) or isinstance(semente, bool):
        raise UsageError(
            f"{namespace.from_log}: `seed` é {semente!r} ({type(semente).__name__}) e tem de ser "
            "um inteiro."
        )
    saida = namespace.output if namespace.output is not None else None
    if saida is None:
        raise UsageError("`--from-log` exige `--output`: o log diz COMO, não onde escrever.")
    # O `try` cobre as duas construções: `Path` de um caminho que não é texto levanta o mesmo
    # `TypeError` que `PipelineSpec` levanta com um limiar que não é número.
    try:
        spec = _spec_do_log(spec_do_log)
        return _process_options_do_log(namespace, spec_do_log, spec, saida)
    except TypeError as erro:
        raise UsageError(
            f"{namespace.from_log}: um campo de `spec` tem o tipo errado — {erro}"
        ) from erro


def _spec_do_log(spec_do_log: dict[str, Any]) -> PipelineSpec:
    """A `PipelineSpec` que o log descreve. Separada para o `try` de quem a chama ficar curto."""
    return PipelineSpec(
        correction="none",
        n_batches=1,
        rsd_cutoff=spec_do_log["rsd-cutoff"],
        # Lido com padrão, ao contrário dos campos obrigatórios. Um log escrito antes de a
        # etapa existir está completo para a versão dele, e `none` reproduz o que aquela
        # execução fez, porque a etapa não correu. Recusá-lo invalidaria todo log antigo.
        ranking_filter=spec_do_log.get("ranking-filter", "none"),
        # As outras três chaves do filtro por posto seguem o mesmo regime: `none` com fração
        # ausente reproduz a execução em que a etapa não correu.
        ranking_keep_fraction=spec_do_log.get("ranking-keep-fraction"),
        ranking_filter_2=spec_do_log.get("ranking-filter-2", "none"),
        ranking_keep_fraction_2=spec_do_log.get("ranking-keep-fraction-2"),
        rsd_variant=spec_do_log["rsd-variant"],
        missing_threshold=spec_do_log["missing-threshold"],
        imputation=spec_do_log["imputation"],
        sample_normalization=spec_do_log["sample-normalization"],
        # Lida com padrão pelo mesmo motivo: `None` reproduz a execução de antes de o campo
        # existir.
        normalization_reference=spec_do_log.get("normalization-reference"),
        # Lido do log e não do arquivo: o mapa está no log por valor, então a reexecução
        # reproduz a saída mesmo que o CSV de origem tenha mudado ou sumido.
        normalization_factors=_fatores_do_log(spec_do_log.get("normalization-factors")),
        transformation=spec_do_log["transformation"],
        scaling=spec_do_log["scaling"],
        seed=spec_do_log["seed"],
    )


def _process_options_do_log(
    namespace: argparse.Namespace,
    spec_do_log: dict[str, Any],
    spec: PipelineSpec,
    saida: Path,
) -> ProcessOptions:
    """As opções do `processar` montadas do log, depois de a `spec` dele ter sido aceita."""
    return ProcessOptions(
        measurements=Path(spec_do_log["measurements"]),
        sample_list=Path(spec_do_log["sample-list"]),
        mode=spec_do_log["mode"],
        orientation=spec_do_log["orientation"],
        spec=spec,
        output=saida,
        log=namespace.log,
    )


def build_process_log(
    options: ProcessOptions, matrix: RefinoMatrix, resultados: list[StageResult]
) -> dict[str, Any]:
    """Monta o log do `processar`. O critério do que entra é que realimentá-lo reproduz a saída.

    `spec` usa os nomes das opções de linha de comando, um-para-um, para
    `_process_options_from_log` reconstruí-las sem tradução. As contagens saem por etapa e por
    motivo, nunca somadas: um total apagaria a distinção entre instabilidade medida e ausência
    de medida.
    """
    return {
        "spec": {
            "measurements": str(options.measurements),
            "sample-list": str(options.sample_list),
            "mode": options.mode,
            "orientation": options.orientation,
            "rsd-cutoff": options.spec.rsd_cutoff,
            "ranking-filter": options.spec.ranking_filter,
            "ranking-keep-fraction": options.spec.ranking_keep_fraction,
            "ranking-filter-2": options.spec.ranking_filter_2,
            "ranking-keep-fraction-2": options.spec.ranking_keep_fraction_2,
            "rsd-variant": options.spec.rsd_variant,
            "missing-threshold": options.spec.missing_threshold,
            "imputation": options.spec.imputation,
            "sample-normalization": options.spec.sample_normalization,
            "normalization-reference": options.spec.normalization_reference,
            # O mapa por valor, como lista de pares: JSON não tem tupla, e a ordem torna a
            # serialização determinística.
            "normalization-factors": (
                None
                if options.spec.normalization_factors is None
                else [[nome, fator] for nome, fator in options.spec.normalization_factors]
            ),
            "transformation": options.spec.transformation,
            "scaling": options.spec.scaling,
            "seed": options.spec.seed,
        },
        "fingerprint": options.spec.fingerprint(),
        "shape": {
            "n_features_in": matrix.n_features,
            "n_samples": matrix.n_samples,
            "n_qc": int(matrix.qc_mask.sum()),
        },
        "stages": stage_counts(resultados),
        "stage_order": [resultado.stage for resultado in resultados],
        "chain_is_consistent": chain_is_consistent(resultados),
        # `provenance` é preenchido por `run_process`, que tem a matriz de saída.
        "provenance": [],
    }


def run_process(options: ProcessOptions) -> tuple[int, dict[str, Any]]:
    """Executa o pipeline e devolve `(código, log)`. Separado de `main` para ser testável."""
    matrix = _le_matriz(
        options.measurements, options.sample_list, options.mode, options.orientation
    )
    processada, resultados = process_matrix(matrix, options.spec)
    write_toserrf(processada, options.output)
    log = build_process_log(options, matrix, resultados)
    log["provenance"] = list(processada.provenance)
    return EXIT_OK, log


def _main_processar(namespace: argparse.Namespace) -> int:
    """`main` do `processar`. Mesma tradução de erro em código que o `corrigir`."""
    try:
        options = parse_process_options(namespace)
    except (InvalidLabelError, UsageError, ValueError) as error:
        print(f"erro: {error}", file=sys.stderr)
        return EXIT_USAGE

    try:
        code, log = run_process(options)
    except (EmptyMatrixError, NormalizacaoInvalidaError) as error:
        # O argumento estava bem formado, e o que falhou foi o encontro dele com esta matriz,
        # como um nome de referência que ela não tem. Por isso `EXIT_UNAVAILABLE` e não
        # `EXIT_USAGE`: erro de forma sai por `parse_process_options`, com código 2.
        print(f"erro: {error}", file=sys.stderr)
        return EXIT_UNAVAILABLE
    except IngestionError as error:
        print(f"erro: {options.measurements.name} recusado pela validação:", file=sys.stderr)
        for issue in error.issues:
            print(f"  [{issue.severity}] {issue.code}: {issue.message}", file=sys.stderr)
        return EXIT_INPUT_REJECTED
    except EntradaIlegivelError as error:
        print(f"erro: a entrada não pôde ser lida — {error}", file=sys.stderr)
        return EXIT_INPUT_REJECTED
    except (IsADirectoryError, PermissionError) as error:
        if not _alvo_e_pasta(error):
            raise
        print(f"erro: `--output` aponta para uma pasta — {error}", file=sys.stderr)
        return EXIT_USAGE

    if options.log is not None:
        codigo_do_log = _grava_log(options.log, log)
        if codigo_do_log != EXIT_OK:
            return codigo_do_log
    return code


# ---------------------------------------------------------------------------
# `exportar`: o arquivo de entrada do MetaboAnalyst
# ---------------------------------------------------------------------------


def _build_exportar(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Declara o subcomando que corrige a matriz e a escreve no formato do MetaboAnalyst."""
    exportar = sub.add_parser(
        "exportar",
        help="corrige e exporta no formato de entrada do MetaboAnalyst",
        description=(
            "Corrige a deriva de sinal com --method, aplica o pré-processamento pedido — "
            "filtros, imputação, normalização, transformação e escalonamento, todos "
            "neutros por padrão —, escreve a matriz no formato de "
            "entrada do MetaboAnalyst, corta por RSD crescente quando excede o limite de "
            "features, e confere a conformidade do arquivo escrito."
        ),
    )
    exportar.add_argument("--measurements", required=True, type=Path)
    exportar.add_argument("--sample-list", required=True, type=Path)
    exportar.add_argument("--mode", required=True, help="rótulo do modo analítico")
    exportar.add_argument("--output", required=True, type=Path)
    exportar.add_argument(
        "--max-features",
        type=int,
        default=5000,
        help="limite do MetaboAnalyst; acima dele mantém as de menor RSD (padrão: 5000)",
    )
    exportar.add_argument("--seed", type=int, default=42)
    _add_vocabulary_argument(
        exportar,
        "--method",
        "Correction",
        "none",
        "Método de correção aplicado ANTES de exportar.",
    )
    _add_vocabulary_argument(
        exportar,
        "--null-denominator",
        "NullDenominatorPolicy",
        "uncorrected_feature",
        "Política do batch_loess diante de denominador nulo. O padrão AQUI difere do "
        "`corrigir` — ver `_main_exportar`.",
    )
    _add_vocabulary_argument(
        exportar,
        "--rsd-variant",
        "RsdVariant",
        "outlier_removed",
        "Fórmula do RSD do corte e do filtro de RSD.",
    )
    _add_eixos_do_pre_processamento(exportar, imputacao_padrao="none")
    _add_vocabulary_argument(
        exportar,
        "--orientation",
        "Orientation",
        "features-in-rows",
        "Orientação do arquivo de ENTRADA, não a interna.",
    )


def _main_exportar(namespace: argparse.Namespace) -> int:
    """Corrige a matriz, escreve o arquivo do MetaboAnalyst e confere a conformidade dele.

    A correção acontece aqui porque o arquivo que o laboratório sobe ao MetaboAnalyst é o
    corrigido, e não o cru.

    O padrão de `--null-denominator` é `uncorrected_feature`, diferente do `corrigir`. Sob
    `r_faithful` o `batch_loess` divide por uma curva que passa por zero e produz `Inf`, e
    `write_metaboanalyst` levanta em infinito, porque o formato só aceita valores numéricos e
    positivos. `r_faithful` existe para bater com as saídas do R, e o R nunca gerou arquivo de
    MetaboAnalyst. O custo é que as features de denominador nulo saem sem correção e marcadas,
    e a contagem por motivo é impressa. As features cuja mediana também é zero não são
    marcadas, porque o divisor é `0/0`, e saem com células vazias nas duas políticas; o
    escritor as grava como ausentes.

    Os eixos do pré-processamento são os do `processar`, com padrão neutro em todos, inclusive
    a imputação: imputar por padrão mudaria o arquivo de quem não passa as opções, que leva o
    ausente como célula vazia. Sem nenhum eixo ligado o executor não corre
    (`requests_preprocessing`), porque ele aplica sempre o filtro de variância quase nula e o
    arquivo perderia as features quase constantes. Com algum eixo ligado, o arquivo é o da
    matriz que `process_matrix` devolve, e o corte por RSD do escritor mede essa matriz.
    """
    from refino.io.validators import check_metaboanalyst_conformance
    from refino.io.writers import write_metaboanalyst

    try:
        validate_label("Correction", namespace.method)
        validate_label("NullDenominatorPolicy", namespace.null_denominator)
        validate_label("RsdVariant", namespace.rsd_variant)
        validate_label("Orientation", namespace.orientation)
    except InvalidLabelError as error:
        print(f"erro: {error}", file=sys.stderr)
        return EXIT_USAGE

    # A especificação nasce antes da matriz, como em `parse_process_options`: rótulo, fração,
    # arquivo de fatores ilegível e combinação proibida são erro de uso, e recusá-los depois de
    # ler a matriz faria esperar por um erro que já estava na linha de comando.
    try:
        spec = _spec_dos_eixos(namespace, correction=namespace.method)
    except (InvalidLabelError, UsageError, ValueError) as error:
        print(f"erro: {error}", file=sys.stderr)
        return EXIT_USAGE

    # O corretor é construído antes do `load_matrix`, para recusar método indisponível sem ler
    # a matriz.
    try:
        corretor = build_corrector(namespace.method, namespace.null_denominator)
    except MethodUnavailableError as error:
        print(f"erro: {error}", file=sys.stderr)
        return EXIT_USAGE

    try:
        matrix = _le_matriz(
            namespace.measurements, namespace.sample_list, namespace.mode, namespace.orientation
        )
    except IngestionError as error:
        print(f"erro: {namespace.measurements.name} recusado pela validação:", file=sys.stderr)
        for issue in error.issues:
            print(f"  [{issue.severity}] {issue.code}: {issue.message}", file=sys.stderr)
        return EXIT_INPUT_REJECTED
    except EntradaIlegivelError as error:
        print(f"erro: a entrada não pôde ser lida — {error}", file=sys.stderr)
        return EXIT_INPUT_REJECTED

    # A ressalva do método no ponto de execução, igual ao `corrigir`.
    emite_ressalva(namespace.method, sys.stderr)

    try:
        resultado = corretor.fit_transform(matrix, seed=namespace.seed)
    except IntensidadeNegativaError as error:
        print(f"erro: a matriz não sustenta o SERRF — {error}", file=sys.stderr)
        return EXIT_INPUT_REJECTED

    alvo = resultado.matrix
    etapas: list[StageResult] | None = None
    if requests_preprocessing(spec):
        try:
            alvo, etapas = process_matrix(alvo, spec)
        except (EmptyMatrixError, NormalizacaoInvalidaError) as error:
            # O mesmo código do `processar`: o argumento estava bem formado, e o que falhou foi
            # o encontro dele com esta matriz.
            print(f"erro: {error}", file=sys.stderr)
            return EXIT_UNAVAILABLE

    try:
        report = write_metaboanalyst(
            alvo,
            namespace.output,
            namespace.max_features,
            rsd_variant=namespace.rsd_variant,
        )
    except (ValueError, InvalidLabelError) as error:
        print(f"erro: {error}", file=sys.stderr)
        return EXIT_USAGE
    except (IsADirectoryError, PermissionError) as error:
        # A mesma tradução de `--output` em pasta que `corrigir` e `processar` fazem.
        if not _alvo_e_pasta(error):
            raise
        print(f"erro: `--output` aponta para uma pasta — {error}", file=sys.stderr)
        return EXIT_USAGE

    print(f"exportado: {namespace.output}")
    print(f"  correção aplicada: {namespace.method}")
    # Contagem por motivo e não total, porque as causas pedem ações diferentes: curva ajustada
    # negativa é propriedade do dado, denominador nulo é a política acima.
    for causa, features in sorted(resultado.uncorrected.items()):
        print(f"    features sem correção ({causa}): {len(features)}")
    # Só quando o executor correu; sem eixo ligado estas linhas não saem.
    if etapas is not None:
        print(
            f"  pré-processamento: imputação {spec.imputation}, normalização "
            f"{spec.sample_normalization}, transformação {spec.transformation}, escalonamento "
            f"{spec.scaling}"
        )
        print(f"    features: {etapas[0].n_features_in} -> {etapas[-1].n_features_out}")
    print(f"  features mantidas: {len(report.kept)}")
    print(f"  removidas: {len(report.removed)}")
    print(f"    QCs todos zero: {len(report.removed_all_zero_qcs)}")
    print(f"    remoção de outliers apagou os QCs: {len(report.removed_outliers_erased)}")
    print(f"    RSD acima do corte: {len(report.removed_high_rsd)}")
    print(f"  células zero ou ausentes escritas como vazias: {report.zeros_written_as_empty}")
    print(f"  nomes sanitizados para o formato: {report.renamed}")

    violations = check_metaboanalyst_conformance(namespace.output)
    if violations:
        # Arquivo não conforme é defeito nosso, não indisponibilidade: reprova.
        print("erro: o arquivo escrito NÃO é conforme ao formato:", file=sys.stderr)
        for violation in violations:
            print(f"  {violation}", file=sys.stderr)
        return EXIT_UNAVAILABLE
    print("  conformidade ao formato publicado: OK")

    return EXIT_OK


# A guarda fica no fim do arquivo. Sob `python -m refino.cli` o corpo do módulo executa de
# cima para baixo, e uma guarda no meio dispararia `main()` antes de as funções de despacho
# estarem definidas. O ponto de entrada instalado importa o módulo inteiro antes de chamar
# `main`, e por isso não mostraria o defeito.
if __name__ == "__main__":  # pragma: no cover - invólucro de execução
    raise SystemExit(main())
