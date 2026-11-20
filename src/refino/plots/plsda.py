"""As três figuras de PLS-DA binário: scores, cargas e VIP.

Scores de PLS-DA nunca aparecem sem Q² e p de permutação. Aqui a regra é garantida pelo
tipo: `Validacao` é campo obrigatório, sem padrão, das três cargas de desenho deste módulo,
e os construtores só aceitam esses tipos. A validação vai na própria imagem, na segunda
linha do título e no rodapé, porque uma separação visual convincente pode vir com Q² baixo
e a figura circula sem o resto da tela. O módulo não classifica: não há limiar de Q² nem
corte de p, e a única referência desenhada, a linha em `VIP = 1`, é derivada. `Q² <= 0`
quer dizer, pela fórmula `1 - PRESS/TSS`, que o modelo prediz pior que a média dos rótulos.

A ordem do par de grupos decide qual grupo é o `y = 1`, porque `statistics/plsda.py` monta
`y = (rotulos == segundo)`. Trocar a ordem não muda scores, cargas, VIP nem Q², já que a
troca de sinal é desfeita pela orientação canônica; a imagem é a mesma nas duas ordens, e
por isso os dois nomes chegam com o papel de cada um e a legenda o declara. VIP indefinido
não é VIP zero: o núcleo devolve `nan` quando a soma de quadrados do modelo é zero, e a
figura conta e declara as indefinidas em vez de desenhar zeros.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final

import numpy as np

from refino.plots.base import (
    GID_DAS_FEATURES,
    GID_REQUEBRAVEL,
    MatrizVaziaError,
    RecusaDoConstrutorError,
    cores_do_ciclo,
    denominador_ou_recusa,
    nova_figura,
    vetor_do_fio_ou_nucleo,
)
from refino.plots.correcao import ROTULO_DA_REGIAO, TRACO_DA_REGIAO
from refino.plots.pca import (
    MINIMO_PARA_ELIPSE,
    Elipse,
    ElipseAusente,
    elipse_de_hotelling,
)
from refino.plots.recorte import (
    TAMANHO_DA_FONTE,
    Recorte,
    aplica,
    frase_contada,
    legenda_fora,
    milhar,
)
from refino.plots.series import MARCADOR_SEM_ESTILO

# A validação é definida em `plots/validacao.py` e reexportada daqui, de onde a janela e os
# testes a importam. Os nomes estão no `__all__`, o que satisfaz o `no_implicit_reexport` do
# mypy.
from refino.plots.validacao import (
    Validacao,
    frase_curta_da_validacao,
    frase_da_validacao,
    validacao_do_fio,
    veredito_da_validacao,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from refino.plots.series import EstilosDosGrupos

__all__ = [
    "ESCORES_DE_VALIDACAO_CRUZADA",
    "FEATURES_DO_VIP",
    "LIMIAR_DE_REFERENCIA_DO_VIP",
    "MATRIZ_DO_MODELO",
    "PRETRATAMENTO",
    "ROTULO_DA_REFERENCIA",
    "TITULO_DAS_CARGAS",
    "TITULO_DOS_SCORES",
    "TITULO_DO_VIP",
    "CargasDoPlsda",
    "EscoresDoPlsda",
    "ImportanciaVip",
    "PlsdaDesalinhadoError",
    "Validacao",
    "cargas_do_plsda",
    "cores_dos_lados",
    "escores_do_plsda",
    "expoente_fora_do_rotulo",
    "figura_das_cargas_do_plsda",
    "figura_do_vip",
    "figura_dos_scores_do_plsda",
    "frase_curta_da_validacao",
    "frase_da_validacao",
    "frase_do_vip_indisponivel",
    "recorte_das_cargas_do_plsda",
    "recorte_do_vip",
    "recorte_dos_scores_do_plsda",
    "regioes_dos_grupos",
    "rotulo_da_regiao_do_grupo",
    "rotulo_do_componente_do_plsda",
    "selecao_do_vip",
    "validacao_do_fio",
    "veredito_da_validacao",
    "vip_do_plsda",
]

# Os escores desenhados são os de validação cruzada, e o título diz isso. Os de calibração
# tendem a separar os grupos porque o ajuste escolhe a direção que os separa; com muito mais
# features que amostras, separam até rótulos sem informação (ver `statistics/plsda.py`). Com
# uma região por grupo por cima, a figura reforçaria uma separação que o Q² desmente.
TITULO_DOS_SCORES: Final = "Scores de validação cruzada do PLS-DA — componentes 1 e 2"

# O texto de método do rodapé dos escores. Descreve o que os pontos são: cada amostra
# projetada pela dobra que não a viu e girada para o referencial da calibração de sinal
# canônico. Sem o giro, cada dobra ficaria no próprio referencial e os pontos não seriam
# comparáveis.
#
# Entra no rodapé no lugar de `PRETRATAMENTO`, e não ao lado, e não pode ser mais longo que
# ele: com as duas frases o rodapé passa por cima do rótulo do eixo x no painel da janela. A
# matriz e o pré-tratamento já estão no título e no campo `matriz` do rodapé, e o tipo da
# região está na legenda.
ESCORES_DE_VALIDACAO_CRUZADA: Final = (
    "PLS-DA em escores de VALIDAÇÃO CRUZADA: cada amostra pela dobra que não a viu, "
    "girada por Procrustes à calibração de sinal canônico"
)

TITULO_DAS_CARGAS: Final = "Cargas de PLS-DA — componentes 1 e 2"

TITULO_DO_VIP: Final = "VIP — importância na projeção"

# O modelo corre sobre a matriz processada (imputada, normalizada, transformada e
# escalonada), a mesma que o `QualityReport` mede, e não sobre a ingerida.
PRETRATAMENTO: Final = (
    "PLS-DA sobre a matriz PROCESSADA — imputação, normalização, transformação e "
    "escalonamento da spec —, com orientação canônica de sinal"
)

# A referência do VIP é derivada, e não calibrada. `VIP_j² = p · Σ(w² · SS)/ΣSS` soma `p`
# sobre as features, então a média de `VIP²` é 1: a linha diz 'acima da média', e não
# 'importante'. Nada neste módulo classifica por ela.
LIMIAR_DE_REFERENCIA_DO_VIP: Final = 1.0

ROTULO_DA_REFERENCIA: Final = "VIP = 1 (média de VIP² é 1 por construção)"

# A procedência que as três figuras declaram, numa constante para as três usarem a mesma
# redação.
MATRIZ_DO_MODELO: Final = (
    "a matriz processada, depois de imputação, normalização, transformação e escalonamento"
)

# Quantas features o VIP desenha por padrão. O rótulo de cada marca do eixo y é texto, e é o
# texto que domina o custo de redesenho: o teto de 100 ms é atravessado entre 16 e 20
# rótulos, e 12 deixa folga para uma máquina mais lenta. É o padrão do parâmetro, e não um
# teto. Não tem relação com `SETAS_DO_BIPLOT`, que por acaso também é 12.
FEATURES_DO_VIP: Final = 12


class PlsdaDesalinhadoError(RecusaDoConstrutorError):
    """Os vetores que deveriam descrever os mesmos objetos não estão alinhados.

    Fica separada de `MatrizVaziaError` porque o conserto é outro e porque o painel mostra uma
    frase por subtipo: 'população vazia' seria a frase errada para um vetor de grupos com uma
    entrada a menos que os scores.
    """


# ---------------------------------------------------------------------------
# As duas populações: uma por amostra, outra por feature
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EscoresDoPlsda:
    """Os scores dos dois componentes por amostra, com o grupo de cada uma e a validação.

    Uma entrada por amostra do contraste (as dos dois grupos declarados), na ordem das linhas
    da matriz processada. A figura desenha os escores de validação cruzada, mas o objeto não
    distingue: quem decide quais vetores entram é quem o constrói.

    `positivo` é o segundo elemento do par de grupos, o `y = 1` do modelo. Como trocar a
    ordem do par não muda número nenhum, só este campo diz de qual grupo o Q² fala.
    """

    x: np.ndarray
    y: np.ndarray
    grupos: tuple[str, ...]
    positivo: str
    negativo: str
    validacao: Validacao

    @property
    def n(self) -> int:
        return int(self.x.size)

    def mascara(self, grupo: str) -> np.ndarray:
        """Booleano por amostra: `True` onde o grupo casa e as duas coordenadas são finitas.

        As duas condições ficam na mesma máscara para a contagem da legenda ser a dos pontos
        desenhados.
        """
        do_grupo = np.array([g == grupo for g in self.grupos], dtype=bool)
        return do_grupo & np.isfinite(self.x) & np.isfinite(self.y)


@dataclass(frozen=True)
class CargasDoPlsda:
    """A carga de cada feature nos dois componentes, com o nome de cada uma e a validação.

    Uma entrada por feature da matriz processada, alinhada a `features`. Um vetor de nomes de
    outra população rotularia cada ponto com o nome de outra feature.
    """

    x: np.ndarray
    y: np.ndarray
    features: tuple[str, ...]
    validacao: Validacao

    @property
    def n(self) -> int:
        return int(self.x.size)

    @property
    def desenhaveis(self) -> np.ndarray:
        """Booleano por feature: `True` onde as duas cargas são finitas."""
        return np.isfinite(self.x) & np.isfinite(self.y)


@dataclass(frozen=True)
class ImportanciaVip:
    """O VIP por feature, com a validação.

    `nan` quer dizer que o VIP não pôde ser calculado, e `0.0` que a feature não tem
    importância. O núcleo devolve `nan` para todas as features quando a soma de quadrados
    total do modelo é zero. `calculaveis` separa os dois casos e `n_indefinidas` conta o
    primeiro.
    """

    valores: np.ndarray
    features: tuple[str, ...]
    validacao: Validacao

    @property
    def n(self) -> int:
        return int(self.valores.size)

    @property
    def calculaveis(self) -> np.ndarray:
        """Booleano por feature: `True` onde o VIP é um número."""
        return np.isfinite(self.valores)

    @property
    def n_calculaveis(self) -> int:
        return int(self.calculaveis.sum())

    @property
    def n_indefinidas(self) -> int:
        return int(self.n - self.n_calculaveis)


# ---------------------------------------------------------------------------
# A fronteira: o que chega do corpo JSON não passou pelo mypy
# ---------------------------------------------------------------------------


def _textos(valores: Any) -> tuple[str, ...]:
    """Uma tupla de `str` vinda de JSON, com o que não é texto convertido em `""`.

    Não descarta, para a posição de cada nome continuar alinhada aos vetores numéricos.
    """
    if not isinstance(valores, (list, tuple)):
        return ()
    return tuple("" if isinstance(v, bool) or not isinstance(v, str) else v for v in valores)


def _rotulo_ou_recusa(valor: Any, papel: str) -> str:
    """Um nome de grupo não vazio, ou a recusa."""
    if isinstance(valor, str) and valor.strip():
        return valor
    raise PlsdaDesalinhadoError(
        f"o grupo {papel} chegou como {valor!r}. A ordem do par decide qual lado da figura é qual "
        "— `y = (rótulos == segundo)` —, e um nome em branco faria a legenda descrever um "
        "contraste que ninguém consegue reproduzir"
    )


def escores_do_plsda(
    *,
    comp1: Sequence[float] | np.ndarray | Any,
    comp2: Sequence[float] | np.ndarray | Any,
    grupos: Sequence[str] | Any,
    grupo_positivo: Any,
    grupo_negativo: Any,
    validacao: Validacao,
) -> EscoresDoPlsda:
    """Os scores vindos do corpo, com as guardas de fronteira e o domínio fechado de dois grupos.

    Um terceiro nome no vetor de grupos é recusado: fora das duas séries, a amostra sumiria da
    figura e a legenda contaria menos do que o contraste tem.

    `validacao` é obrigatório e não tem padrão: não se obtém `EscoresDoPlsda` sem Q² e p.
    """
    x = vetor_do_fio_ou_nucleo(comp1)
    y = vetor_do_fio_ou_nucleo(comp2)
    if x.size == 0:
        raise MatrizVaziaError(
            "não veio score nenhum: o PLS-DA não correu para esta execução — sem par de grupos "
            "declarado a rota não o calcula. Figura em branco seria indistinguível de quebrada"
        )
    if x.size != y.size:
        raise PlsdaDesalinhadoError(
            f"{x.size} score(s) do componente 1 para {y.size} do componente 2. Os dois vetores "
            "descrevem as mesmas amostras, e comprimentos diferentes significam que um é de outra "
            "população"
        )
    rotulos = _textos(grupos)
    if len(rotulos) != x.size:
        raise PlsdaDesalinhadoError(
            f"{len(rotulos)} rótulo(s) de grupo para {x.size} score(s). O grupo é o que separa as "
            "duas séries, e um vetor desalinhado poria amostras do controle no lado do caso por "
            "acidente de índice"
        )
    positivo = _rotulo_ou_recusa(grupo_positivo, "positivo")
    negativo = _rotulo_ou_recusa(grupo_negativo, "negativo")
    if positivo == negativo:
        raise PlsdaDesalinhadoError(
            f"os dois lados do contraste chegaram com o mesmo nome ({positivo!r}): não há "
            "contraste a desenhar, e as duas séries colapsariam numa"
        )
    desconhecidos = sorted({r for r in rotulos if r not in (positivo, negativo)})
    if desconhecidos:
        raise PlsdaDesalinhadoError(
            f"o vetor de grupos traz {desconhecidos} e o contraste é {negativo!r} contra "
            f"{positivo!r}. Um terceiro grupo aqui não pode cair numa das duas séries por omissão "
            "— a legenda contaria uma população que o modelo não viu"
        )
    return EscoresDoPlsda(
        x=x,
        y=y,
        grupos=rotulos,
        positivo=positivo,
        negativo=negativo,
        validacao=validacao,
    )


def cargas_do_plsda(
    *,
    comp1: Sequence[float] | np.ndarray | Any,
    comp2: Sequence[float] | np.ndarray | Any,
    features: Sequence[str] | Any,
    validacao: Validacao,
) -> CargasDoPlsda:
    """As cargas vindas do corpo, alinhadas ao vetor de nomes das features.

    Os nomes acompanham as cargas porque a figura põe um ponto por feature e a identificação
    por clique precisa dizer qual é.
    """
    x = vetor_do_fio_ou_nucleo(comp1)
    y = vetor_do_fio_ou_nucleo(comp2)
    if x.size == 0:
        raise MatrizVaziaError(
            "não veio carga nenhuma: o PLS-DA não correu para esta execução. Figura em branco "
            "seria indistinguível de figura quebrada"
        )
    if x.size != y.size:
        raise PlsdaDesalinhadoError(
            f"{x.size} carga(s) do componente 1 para {y.size} do componente 2. Os dois vetores "
            "descrevem as mesmas features, e comprimentos diferentes significam que um é de outra "
            "população"
        )
    nomes = _textos(features)
    if len(nomes) != x.size:
        raise PlsdaDesalinhadoError(
            f"{len(nomes)} nome(s) de feature para {x.size} carga(s). O nome é o que a "
            "identificação de R2 mostra, e um vetor desalinhado devolveria o nome do vizinho — com "
            "a figura bem formada"
        )
    return CargasDoPlsda(x=x, y=y, features=nomes, validacao=validacao)


def vip_do_plsda(
    *,
    vip: Sequence[float] | np.ndarray | Any,
    features: Sequence[str] | Any,
    validacao: Validacao,
) -> ImportanciaVip:
    """O VIP vindo do corpo, alinhado aos nomes. `None` vira `nan`, e não zero.

    O FastAPI serializa `nan` como `null`. Ler `null` como `0.0` apagaria a diferença entre
    VIP indefinido e VIP zero.
    """
    valores = vetor_do_fio_ou_nucleo(vip)
    if valores.size == 0:
        raise MatrizVaziaError(
            "não veio VIP nenhum: o PLS-DA não correu para esta execução. Figura em branco seria "
            "indistinguível de figura quebrada"
        )
    nomes = _textos(features)
    if len(nomes) != valores.size:
        raise PlsdaDesalinhadoError(
            f"{len(nomes)} nome(s) de feature para {valores.size} valor(es) de VIP. O nome é o "
            "rótulo da marca do eixo, e um vetor desalinhado atribuiria a importância de uma "
            "feature ao nome de outra"
        )
    return ImportanciaVip(valores=valores, features=nomes, validacao=validacao)


# ---------------------------------------------------------------------------
# Os rótulos, as cores e a seleção
# ---------------------------------------------------------------------------


def rotulo_do_componente_do_plsda(indice: int) -> str:
    """O rótulo de eixo de um componente: `0` vira `"componente 1 de PLS-DA"`.

    Não leva ressalva de sinal, ao contrário do rótulo da PCA: aqui o sinal é fixado pela
    orientação canônica, e o rodapé declara o método.

    Tem uma linha só. Com duas, o expoente que o `matplotlib` desenha no fim do eixo x cairia
    sobre a segunda.
    """
    return f"componente {indice + 1} de PLS-DA"


def _rotulo_da_serie(grupo: str, quantas: int, *, positivo: bool) -> str:
    """A entrada de legenda de um lado: `"HAMTSP — 14 amostras (classe positiva, y = 1)"`.

    Traz a contagem, que é o que permite julgar o poder do teste, e o papel do grupo, que a
    imagem sozinha não mostra.
    """
    plural = "amostra" if quantas == 1 else "amostras"
    papel = "classe positiva, y = 1" if positivo else "classe de referência, y = 0"
    return f"{grupo} — {milhar(quantas)} {plural} ({papel})"


def cores_dos_lados() -> tuple[Any, Any]:
    """As duas cores do contraste, do ciclo dos `rcParams`, na ordem negativa e positiva.

    A ordem é a mesma nas três figuras. Só vale para os scores desenhados sem estilo: com
    estilo, a cor dos scores é a do grupo. As cargas e o VIP não pintam grupo.
    """
    negativa, positiva = cores_do_ciclo(2)
    return negativa, positiva


def selecao_do_vip(importancia: ImportanciaVip, quantas: int) -> np.ndarray:
    """Os índices das `quantas` features de maior VIP, em ordem decrescente.

    O empate é resolvido pelo nome, para a mesma execução desenhar as mesmas features em
    qualquer máquina. As de VIP indefinido ficam fora da seleção; o rodapé declara quantas são.
    """
    if quantas < 0:
        raise ValueError(f"`quantas` não pode ser negativo; recebi {quantas}")
    calculaveis = np.nonzero(importancia.calculaveis)[0]
    ordem = sorted(
        calculaveis.tolist(),
        key=lambda i: (-float(importancia.valores[i]), importancia.features[i]),
    )
    return np.array(ordem[:quantas], dtype=int)


# ---------------------------------------------------------------------------
# A região de predição de cada grupo, sobre os escores de validação cruzada
# ---------------------------------------------------------------------------


def rotulo_da_regiao_do_grupo(grupo: str) -> str:
    """O rótulo da linha e da entrada de legenda: `"Região de predição 95% (T²) — HAMTSP"`.

    A frase de base é `correcao.ROTULO_DA_REGIAO`, porque é a mesma região de Hotelling. O
    rótulo é único por grupo: o arnês de teste indexa as séries por rótulo.
    """
    return f"{ROTULO_DA_REGIAO} — {grupo}"


def regioes_dos_grupos(escores: EscoresDoPlsda) -> dict[str, Elipse | ElipseAusente]:
    """Uma região de predição de Hotelling T² por grupo, sobre os pontos daquele grupo.

    Sobre os dois grupos juntos a região cresceria para conter os dois e não diria se eles se
    separam. Por grupo, ela mostra onde cairia a próxima amostra do grupo vista por um modelo
    que não a conheceu, e o que se lê é a sobreposição das duas.

    A máscara é `EscoresDoPlsda.mascara`, a mesma das séries: o `n` da região é o que a legenda
    conta.

    A figura não mostra o `n` e o `c²` de cada região, nem a ressalva de que ela só é exata sob
    normalidade multivariada dos escores, que a figura não testa. O rodapé não tem espaço para
    isso; a legenda diz o tipo e o nível.
    """
    return {
        grupo: elipse_de_hotelling(
            escores.x[escores.mascara(grupo)], escores.y[escores.mascara(grupo)]
        )
        for grupo in (escores.negativo, escores.positivo)
    }


def _motivo_curto(escores: EscoresDoPlsda, grupo: str, regiao: ElipseAusente) -> str:
    """O motivo resumido da região ausente: `"menos de 3 amostras (2)"`, ou a cabeça do motivo.

    O motivo completo de `ElipseAusente` é longo e divide o eixo com os pontos. Fora do caso
    da contagem, usa o texto do motivo até os dois pontos, sem adivinhar a causa.
    """
    quantas = int(escores.mascara(grupo).sum())
    if quantas < MINIMO_PARA_ELIPSE:
        return f"menos de {MINIMO_PARA_ELIPSE} amostras ({milhar(quantas)})"
    return regiao.motivo.split(":", 1)[0]


# ---------------------------------------------------------------------------
# Os três recortes
# ---------------------------------------------------------------------------


def _frase_do_contraste(escores: EscoresDoPlsda) -> str:
    """A ordem do par por extenso: `"NoHTLV (n = 16, y = 0) contra HAMTSP (n = 14, y = 1)"`."""
    n_neg = int(escores.mascara(escores.negativo).sum())
    n_pos = int(escores.mascara(escores.positivo).sum())
    return (
        f"{escores.negativo} (n = {milhar(n_neg)}, y = 0) contra "
        f"{escores.positivo} (n = {milhar(n_pos)}, y = 1)"
    )


def recorte_dos_scores_do_plsda(*, escores: EscoresDoPlsda, n_features_totais: Any) -> Recorte:
    """O rodapé do gráfico de scores: população, método e validação.

    O numerador são as features do modelo, e não as amostras desenhadas: é o número que se
    confere contra o `n_features_saida` da execução. A contagem de amostras está no título e
    na legenda.

    O método usa `ESCORES_DE_VALIDACAO_CRUZADA` no lugar de `PRETRATAMENTO`, que as cargas e o
    VIP continuam usando.
    """
    totais = denominador_ou_recusa(n_features_totais, de="o PLS-DA")
    return Recorte(
        desenhadas=totais,
        disponiveis=totais,
        amostragem=(
            f"nenhuma — as {milhar(totais)} features da matriz processada entram todas "
            f"no modelo. Contraste: {_frase_do_contraste(escores)} — trocar a ordem do par "
            "não muda número nenhum; muda só qual lado é o y = 1 do Q²"
        ),
        matriz=MATRIZ_DO_MODELO,
        metodo=(
            f"{ESCORES_DE_VALIDACAO_CRUZADA}. {frase_da_validacao(escores.validacao)}. "
            f"{veredito_da_validacao(escores.validacao)}"
        ),
    )


def recorte_das_cargas_do_plsda(*, cargas: CargasDoPlsda, n_features_totais: Any) -> Recorte:
    """O rodapé das cargas: as desenháveis sobre as features da matriz inteira.

    Entre as duas pontas ficam as features de carga não finita, que são contadas e declaradas.
    """
    desenhaveis = int(cargas.desenhaveis.sum())
    fora = cargas.n - desenhaveis
    indefinidas = (
        "nenhuma tem carga não finita"
        if fora == 0
        else f"as {milhar(fora)} de carga não finita são contadas, não descartadas"
    )
    return Recorte(
        desenhadas=desenhaveis,
        disponiveis=denominador_ou_recusa(n_features_totais, de="o PLS-DA"),
        amostragem=(
            f"nenhuma — as {milhar(cargas.n)} features do modelo entram todas, e {indefinidas}"
        ),
        matriz=MATRIZ_DO_MODELO,
        metodo=(
            f"{PRETRATAMENTO}. A carga é a projeção da feature no componente, e NÃO uma medida de "
            f"significância. {frase_da_validacao(cargas.validacao)}. "
            f"{veredito_da_validacao(cargas.validacao)}"
        ),
    )


def _frase_da_selecao(n_desenhadas: int, n_calculaveis: int) -> str:
    """A frase da seleção: `"as 12 features de MAIOR VIP, de 5.942 com VIP calculável"`.

    `frase_contada` cuida do zero e do singular.
    """
    return frase_contada(
        n_desenhadas,
        nenhuma="nenhuma feature foi desenhada, porque nenhuma tem VIP calculável",
        uma=f"1 feature de MAIOR VIP, de {milhar(n_calculaveis)} com VIP calculável",
        algumas=f"features de MAIOR VIP, de {milhar(n_calculaveis)} com VIP calculável",
    )


def recorte_do_vip(*, importancia: ImportanciaVip, n_desenhadas: int) -> Recorte:
    """O rodapé do VIP: quantas desenhou, de quantas, e quantas não têm VIP calculável.

    Sem a contagem de indefinidas, VIP não calculado e VIP zero teriam a mesma aparência.
    """
    indefinidas = (
        "nenhuma feature ficou sem VIP calculável"
        if importancia.n_indefinidas == 0
        else (
            f"{milhar(importancia.n_indefinidas)} de {milhar(importancia.n)} features NÃO têm VIP "
            "calculável — isso é «não foi possível calcular», e não «importância zero»"
        )
    )
    return Recorte(
        desenhadas=n_desenhadas,
        disponiveis=denominador_ou_recusa(importancia.n, de="o PLS-DA"),
        amostragem=(
            f"{_frase_da_selecao(n_desenhadas, importancia.n_calculaveis)} — escolha "
            f"determinística, com empate resolvido pelo nome; nenhuma amostragem aleatória. "
            f"{indefinidas}"
        ),
        matriz=MATRIZ_DO_MODELO,
        metodo=(
            f"{PRETRATAMENTO}. VIP_j² = p · Σ_a(w_ja² · SS_a) / Σ_a SS_a — a média de VIP² sobre "
            f"as features é 1 por construção, e a referência é essa média e não um corte de "
            f"decisão. {frase_da_validacao(importancia.validacao)}. "
            f"{veredito_da_validacao(importancia.validacao)}"
        ),
    )


# ---------------------------------------------------------------------------
# 1. Os scores, com a validação na própria imagem
# ---------------------------------------------------------------------------


def expoente_fora_do_rotulo(eixo: Axes) -> None:
    """Sobe o expoente do eixo x para a faixa das marcas, fora da linha do `xlabel`.

    Com scores de ordem `1e8`, que é o que sai sem transformação nem escalonamento, o
    `matplotlib` desenha o expoente no fim do eixo x, na mesma faixa vertical do `xlabel`. Com
    a legenda fora dos eixos, eles ficam estreitos e o rótulo alcança o canto, então os dois
    textos colidem. A colisão depende do comprimento da legenda: com nomes de grupo curtos ela
    não acontece.

    A âncora passa de `(1, 0)` com `va="top"` para `(1, 1)` com `va="bottom"`, e o expoente
    sobe para a linha das marcas, ainda encostado à direita. Desligar a notação com
    `ticklabel_format(style="plain")` trocaria a colisão por marcas longas que se atropelam.
    """
    expoente = eixo.xaxis.get_offset_text()
    expoente.set_position((1.0, 1.0))
    expoente.set_horizontalalignment("right")
    expoente.set_verticalalignment("bottom")


def figura_dos_scores_do_plsda(
    *,
    escores: EscoresDoPlsda,
    recorte: Recorte,
    estilos: EstilosDosGrupos | None = None,
) -> Figure:
    """Componente 1 contra componente 2, uma série e uma região por grupo, com o Q² no título.

    Os pontos são os escores de validação cruzada, e cada grupo leva a região de predição de
    Hotelling T² sobre eles, tracejada e na cor da série. Duas regiões que se sobrepõem são o
    Q² em forma de figura. As regiões são criadas depois das séries e desenhadas por baixo
    delas, por `zorder`, para a primeira linha dos eixos continuar sendo a primeira série.
    Grupo sem região diz por quê dentro do eixo, no canto de baixo.

    A validação é a segunda linha do título. Uma separação visual convincente pode vir com Q²
    baixo (num outro estudo do laboratório, cerca de 24%), e título e rodapé são o que
    sobrevive a uma captura de tela.

    São duas séries, e não um artista com cor por ponto, porque os pontos são amostras: a
    identificação por clique devolve o rótulo da série. A ordem de desenho é negativa e depois
    positiva, para a classe de interesse ficar por cima.

    Sem `estilos`, a referência leva a primeira cor do ciclo e a classe positiva a segunda, e
    o mesmo grupo muda de cor conforme o contraste. Com `estilos`, a cor e o marcador são os
    do grupo, resolvidos sobre a coorte, e o papel continua dito na legenda.
    """
    figura = nova_figura()
    eixo = figura.add_subplot(1, 1, 1)
    if estilos is None:
        cor_negativa, cor_positiva = cores_dos_lados()
        tracos = {
            escores.negativo: (cor_negativa, MARCADOR_SEM_ESTILO),
            escores.positivo: (cor_positiva, MARCADOR_SEM_ESTILO),
        }
    else:
        do_grupo = estilos.para((escores.negativo, escores.positivo))
        tracos = {grupo: (estilo.hex, estilo.marcador) for grupo, estilo in do_grupo.items()}

    desenhadas = 0
    for grupo, positivo in ((escores.negativo, False), (escores.positivo, True)):
        mascara = escores.mascara(grupo)
        quantas = int(mascara.sum())
        if quantas == 0:
            continue
        desenhadas += quantas
        cor, marcador = tracos[grupo]
        eixo.plot(
            escores.x[mascara],
            escores.y[mascara],
            marker=marcador,
            linestyle="none",
            color=cor,
            label=_rotulo_da_serie(grupo, quantas, positivo=positivo),
        )

    if desenhadas == 0:
        raise MatrizVaziaError(
            f"há {escores.n} amostra(s) no contraste e nenhuma com os dois componentes finitos: "
            "não há ponto a desenhar. Uma figura com o eixo montado e nada dentro lê-se como "
            "figura quebrada"
        )

    _desenha_as_regioes(eixo, escores, {grupo: cor for grupo, (cor, _) in tracos.items()})
    eixo.set_xlabel(rotulo_do_componente_do_plsda(0))
    eixo.set_ylabel(rotulo_do_componente_do_plsda(1))
    eixo.set_title(
        f"{TITULO_DOS_SCORES}\n{frase_curta_da_validacao(escores.validacao)}",
        fontsize=TAMANHO_DA_FONTE,
    )
    legenda_fora(eixo)
    expoente_fora_do_rotulo(eixo)
    aplica(figura, recorte)
    return figura


# Abaixo dos pontos, que têm o `zorder` 2 de toda `Line2D`: a região não pode esconder uma
# amostra. Usa `zorder`, e não a ordem de criação, para a primeira linha dos eixos continuar
# sendo a primeira série.
_ORDEM_DA_REGIAO: Final = 1.5


def _desenha_as_regioes(eixo: Axes, escores: EscoresDoPlsda, cores: dict[str, Any]) -> None:
    """Desenha a região de cada grupo, ou um texto curto no eixo que diz por que ela falta.

    A região é uma `Line2D`, e não um `patch`, para ficar visível ao arnês de teste e à
    identificação por clique. Leva a cor da série do grupo.

    O texto da ausência fica dentro do eixo, no canto de baixo, marcado como requebrável:
    `recorte.aplica` o quebra pela largura final do eixo, depois de a legenda o estreitar.
    """
    regioes = regioes_dos_grupos(escores)
    ausentes = []
    for grupo in (escores.negativo, escores.positivo):
        regiao = regioes[grupo]
        if isinstance(regiao, ElipseAusente):
            ausentes.append(f"sem região de {grupo}: {_motivo_curto(escores, grupo, regiao)}")
            continue
        x, y = regiao.contorno()
        eixo.plot(
            x,
            y,
            linestyle=TRACO_DA_REGIAO,
            color=cores[grupo],
            zorder=_ORDEM_DA_REGIAO,
            label=rotulo_da_regiao_do_grupo(grupo),
        )
    if ausentes:
        texto = eixo.text(
            0.02,
            0.02,
            "\n".join(ausentes),
            transform=eixo.transAxes,
            ha="left",
            va="bottom",
            fontsize=TAMANHO_DA_FONTE,
        )
        texto.set_gid(GID_REQUEBRAVEL)


# ---------------------------------------------------------------------------
# 2. As cargas
# ---------------------------------------------------------------------------


def figura_das_cargas_do_plsda(*, cargas: CargasDoPlsda, recorte: Recorte) -> Figure:
    """Um ponto por feature: a carga no componente 1 contra a carga no componente 2.

    Desenha todas as features de carga finita. Há três artistas de dado, a nuvem e as duas
    cruzes da origem, e só a nuvem leva `GID_DAS_FEATURES`: é ela que a identificação por
    clique indexa pelo vetor de nomes.

    As cruzes da origem são desenhadas em coordenadas de dado, com `plot`, e não com
    `axhline`, cuja transformação mista faria o arnês ler pontos fora do limite.

    As duas cores são pedidas ao ciclo: `scatter` e `plot` consomem cicladores separados que
    começam na mesma cor, e a cruz sairia da cor da nuvem.
    """
    desenhaveis = cargas.desenhaveis
    n_desenhadas = int(desenhaveis.sum())
    if n_desenhadas == 0:
        raise MatrizVaziaError(
            f"há {cargas.n} feature(s) e nenhuma com carga finita nos dois componentes: não há "
            "ponto a desenhar. Uma figura com o eixo montado e nada dentro lê-se como quebrada"
        )

    px = cargas.x[desenhaveis]
    py = cargas.y[desenhaveis]
    figura = nova_figura()
    eixo = figura.add_subplot(1, 1, 1)
    da_nuvem, da_origem = cores_do_ciclo(2)
    # As duas cruzes têm rótulos diferentes porque o arnês de teste indexa as séries por
    # rótulo: com o mesmo texto, uma delas ficaria invisível para ele.
    origem = eixo.plot(
        [float(px.min()), float(px.max())],
        [0.0, 0.0],
        linewidth=0.8,
        color=da_origem,
        label="origem do componente 2",
    )[0]
    eixo.plot(
        [0.0, 0.0],
        [float(py.min()), float(py.max())],
        linewidth=0.8,
        color=da_origem,
        label="origem do componente 1",
    )
    pontos = eixo.scatter(px, py, s=8, color=da_nuvem, label=f"{milhar(n_desenhadas)} features")
    # Dos três artistas de dado, só este tem um ponto por feature.
    pontos.set_gid(GID_DAS_FEATURES)

    eixo.set_xlabel(rotulo_do_componente_do_plsda(0))
    eixo.set_ylabel(rotulo_do_componente_do_plsda(1))
    eixo.set_title(
        f"{TITULO_DAS_CARGAS}\n{frase_curta_da_validacao(cargas.validacao)}",
        fontsize=TAMANHO_DA_FONTE,
    )
    expoente_fora_do_rotulo(eixo)
    origem.set_label("origem")
    legenda_fora(eixo, [pontos, origem])
    aplica(figura, recorte)
    return figura


# ---------------------------------------------------------------------------
# 3. O VIP, e a distinção entre zero e indefinido
# ---------------------------------------------------------------------------


def frase_do_vip_indisponivel(importancia: ImportanciaVip) -> str:
    """O texto escrito no lugar dos pontos quando nenhuma feature tem VIP calculável.

    A população não está vazia: há `n` features, todas com VIP indefinido porque a soma de
    quadrados do modelo é zero. Por isso a figura sai com esta frase, em vez de levantar
    `MatrizVaziaError`.
    """
    return (
        f"nenhuma das {milhar(importancia.n)} features tem VIP calculável:\n"
        "a soma de quadrados explicada pelo modelo é zero, então a razão que define o VIP\n"
        "não existe. Isso é «não foi possível calcular» — e NÃO «importância zero»."
    )


def figura_do_vip(
    *,
    importancia: ImportanciaVip,
    n_features: int = FEATURES_DO_VIP,
    recorte: Recorte,
) -> Figure:
    """As `n_features` de maior VIP, num gráfico de pontos, com a referência em `VIP = 1`.

    Quantas desenhar é decisão do chamador. `selecao_do_vip` é determinística, então o rodapé
    pode refazer a seleção e obter o mesmo resultado.

    Usa pontos, e não barras: `ax.bar` produz `Rectangle` em `ax.patches`, que o arnês de teste
    e a identificação por clique não leem. A referência é desenhada em coordenadas de dado.

    Sem nenhuma feature calculável a figura sai com a frase de `frase_do_vip_indisponivel`.
    """
    figura = nova_figura()
    eixo = figura.add_subplot(1, 1, 1)

    indices = selecao_do_vip(importancia, n_features)
    if indices.size == 0:
        eixo.text(
            0.5,
            0.5,
            frase_do_vip_indisponivel(importancia),
            transform=eixo.transAxes,
            ha="center",
            va="center",
            fontsize=TAMANHO_DA_FONTE,
        )
        # Sem pontos, as marcas saem: o `matplotlib` desenharia os limites padrão, e a figura
        # afirmaria que o VIP vai de zero a um.
        eixo.set_xticks([])
        eixo.set_yticks([])
        eixo.set_xlabel("VIP")
        eixo.set_ylabel("feature")
        eixo.set_title(
            f"{TITULO_DO_VIP}\n{frase_curta_da_validacao(importancia.validacao)}",
            fontsize=TAMANHO_DA_FONTE,
        )
        aplica(figura, recorte)
        return figura

    alturas = np.arange(indices.size, dtype=float)
    valores = importancia.valores[indices]
    da_nuvem, da_referencia = cores_do_ciclo(2)
    eixo.plot(
        [LIMIAR_DE_REFERENCIA_DO_VIP, LIMIAR_DE_REFERENCIA_DO_VIP],
        [float(alturas.min()), float(alturas.max())],
        linestyle="--",
        linewidth=0.8,
        color=da_referencia,
        label=ROTULO_DA_REFERENCIA,
    )
    eixo.plot(
        valores,
        alturas,
        marker="o",
        linestyle="none",
        color=da_nuvem,
        label=f"{milhar(int(indices.size))} features de maior VIP",
    )

    # A seleção é decrescente e o eixo é invertido, para a feature de maior VIP ficar em cima.
    eixo.set_yticks(alturas.tolist())
    eixo.set_yticklabels([importancia.features[int(i)] for i in indices], fontsize=TAMANHO_DA_FONTE)
    eixo.invert_yaxis()
    expoente_fora_do_rotulo(eixo)
    eixo.set_xlabel("VIP")
    eixo.set_ylabel("feature")
    eixo.set_title(
        f"{TITULO_DO_VIP}\n{frase_curta_da_validacao(importancia.validacao)}",
        fontsize=TAMANHO_DA_FONTE,
    )
    legenda_fora(eixo)
    aplica(figura, recorte)
    return figura
