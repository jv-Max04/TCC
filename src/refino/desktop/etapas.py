"""O estado do passo a passo na janela: que etapa foi feita, com que campos e sobre qual entrada.

O servidor não guarda sessão, então é a janela que sabe o que foi feito. Cada etapa guarda, no
disparo, a fotografia dos campos que a tela lhe deu e a identidade da matriz, e está em dia
enquanto a tela der a mesma fotografia. Sinais de widget não servem para isso: a máquina também
escreve nos controles, como no pré-preenchimento da recomendação, e uma regra por sinal
acompanharia gestos e não valores. O módulo não conhece Qt, para essa regra ser testada sem tela.

As etapas não se travam em cadeia: uma etapa dispara quando nenhuma outra está em curso. A
Normalização e a Análise partem de uma entrada escolhida (a matriz original, a corrigida ou a
normalizada), e o pedido leva os campos da etapa feita que a entrada é, e não os controles das
outras abas, para o que se baixa e o que se analisa corresponderem ao que a etapa mostrou. Os eixos
que a entrada não aplicou vão neutros.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field, fields, replace
from typing import Any, Final, Literal

from refino.domain.spec import PipelineSpec

__all__ = [
    "CADEIA_DA_TELA",
    "CAMPOS_COMUNS",
    "CAMPOS_DO_PAR_DE_DECISAO",
    "CAMPO_SEM_PROPOSTA",
    "CHAVES_DA_SPEC",
    "CHAVE_DA_ETAPA",
    "ENTRADAS",
    "ENTRADAS_DA_ETAPA",
    "ETAPAS",
    "ETAPA_DA_ENTRADA",
    "ETAPA_DO_CAMPO",
    "IDENTIDADE_DA_MATRIZ",
    "NOMES_DAS_ETAPAS",
    "ROTA_DA_ETAPA",
    "CampoSemEtapaError",
    "Entrada",
    "EntradaDaEtapa",
    "EntradaDesconhecidaError",
    "EntradaIndisponivelError",
    "EstadoDaEtapa",
    "EstadoDasEtapas",
    "EtapaFeita",
    "Fotografia",
    "Motivo",
    "campos_da_rota",
    "etapa_a_refazer",
    "fotografia",
    "fotografia_da_etapa",
]

ETAPAS: Final = (1, 2, 3)
# As chaves de espera das duas prévias; a etapa 3 é a execução, e usa a chave dela.
CHAVE_DA_ETAPA: Final[dict[int, str]] = {1: "etapa_correcao", 2: "etapa_preprocessamento"}
ROTA_DA_ETAPA: Final[dict[int, str]] = {1: "/etapas/correcao", 2: "/etapas/pre-processamento"}
# O nome de cada etapa como a aba o diz, porque as frases que o citam aparecem na tela. Vive aqui
# e não em `janela/pipeline.py` porque `janela/graficos.py` também o usa e não importa o painel do
# pipeline.
NOMES_DAS_ETAPAS: Final[dict[int, str]] = {
    1: "correção",
    2: "normalização",
    3: "análise estatística",
}

# A etapa a que cada campo do pedido pertence; `None` é nenhuma, para o que muda o tempo e não o
# resultado. Campo fora da tabela levanta: ficaria fora de toda fotografia, e mudá-lo não
# desatualizaria nada.
ETAPA_DO_CAMPO: Final[dict[str, int | None]] = {
    "mode": 1,
    "orientation": 1,
    "correction": 1,
    "n_batches": 1,
    # Os cortes de lote do diagnóstico mudam a partição da correção, e por isso são da etapa 1.
    "batch_cuts": 1,
    "rsd_variant": 1,
    "seed": 1,
    "auto": 1,
    "imputation": 2,
    "sample_normalization": 2,
    "transformation": 2,
    "scaling": 2,
    "rsd_cutoff": 2,
    "missing_threshold": 2,
    "ranking_filter": 2,
    "ranking_keep_fraction": 2,
    "ranking_filter_2": 2,
    "ranking_keep_fraction_2": 2,
    "normalization_reference": 2,
    "estudo": 3,
    "apresentacao": 3,
    "analise_json": 3,
    "group_a": 3,
    "group_b": 3,
    "paralelo": None,
}

# O par humano-máquina que o `/execucoes` grava na tabela `decisions`. Não é de etapa nenhuma: é a
# decisão da Correção, viaja com a entrada que deriva dela, e só a execução o grava. Uma Análise
# sobre a original não tem correção a justificar, e uma sobre a corrigida grava a decisão daquela
# correção, e não a que a tela mostra agora.
#
# `sem_proposta` declara que nenhuma proposta da máquina foi seguida. A aba 'Correção' abre nos
# valores padrão e só recebe as sugestões pelo botão 'Seguir sugestões do app'; sem esta marca, a
# Correção feita sem o clique seria gravada como `nao_sobrescreveu`, isto é, proposta seguida.
CAMPO_SEM_PROPOSTA: Final = "sem_proposta"
CAMPOS_DO_PAR_DE_DECISAO: Final[frozenset[str]] = frozenset(
    {"campos_sobrescritos", "proposta_json", CAMPO_SEM_PROPOSTA}
)
# Os campos que toda etapa precisa e que não são eixo de etapa nenhuma em particular: vêm da etapa
# de entrada quando ela existe, e da tela na original.
CAMPOS_COMUNS: Final[frozenset[str]] = frozenset({"mode", "orientation", "rsd_variant", "seed"})
# A identidade da matriz na fotografia. Trocar o par de arquivos ou o modo a muda.
IDENTIDADE_DA_MATRIZ: Final = "matriz@1"

Entrada = Literal["original", "corrigida", "normalizada"]
# Da menos à mais tratada; a padrão é a última que existir.
ENTRADAS: Final[tuple[Entrada, ...]] = ("original", "corrigida", "normalizada")
ENTRADAS_DA_ETAPA: Final[dict[int, tuple[Entrada, ...]]] = {
    1: ("original",),
    2: ("original", "corrigida"),
    3: ENTRADAS,
}
# A etapa feita que cada entrada é. A original é a matriz carregada, e não é etapa.
ETAPA_DA_ENTRADA: Final[dict[Entrada, int | None]] = {
    "original": None,
    "corrigida": 1,
    "normalizada": 2,
}
# A cadeia em que cada etapa refaz as anteriores com os campos da tela: a figura da 2 depende então
# da 1, e a da 3 das duas.
CADEIA_DA_TELA: Final[dict[int, Entrada]] = {2: "corrigida", 3: "normalizada"}

# Os eixos que uma entrada não aplicou. A correção neutra leva um lote só porque a rota exige o
# campo e `none` não o lê; os cortes e a delegação ficam fora, que é como a rota lê 'nenhum'.
_CORRECAO_NEUTRA: Final[dict[str, Any]] = {"correction": "none", "n_batches": 1}
# Os sete eixos do pré-processamento: os filtros, a imputação, a normalização por amostra com a
# referência e os fatores, a transformação e o escalonamento. Os que não têm `none` no vocabulário
# (o corte de RSD, o de ausentes, as frações de posto, a referência) são neutros por ausência, que
# é como as rotas leem 'sem filtro'; os fatores são arquivo, que só viaja com `supplied_factors`.
# Nenhum normaliza, então não se forma a combinação recusada de imputação `none` com normalização.
_PREPROCESSAMENTO_NEUTRO: Final[dict[str, Any]] = {
    "imputation": "none",
    "sample_normalization": "none",
    "transformation": "none",
    "scaling": "none",
    "ranking_filter": "none",
    "ranking_filter_2": "none",
}
# O par vazio é o padrão do `/execucoes`: nenhum eixo sobrescrito, e nenhuma proposta.
_PAR_VAZIO: Final[dict[str, Any]] = {"campos_sobrescritos": "[]"}
# O par da entrada original: a correção neutra não foi proposta pela máquina nem escolhida na aba
# 'Correção', então a decisão gravada é `sem_proposta`. Fica separado de `_PAR_VAZIO`, que é a base
# da entrada corrigida: com `sem_proposta` dentro, ele vazaria para uma Correção que seguiu as
# sugestões.
_PAR_SEM_PROPOSTA: Final[dict[str, Any]] = {**_PAR_VAZIO, CAMPO_SEM_PROPOSTA: True}
_PROPOSTA: Final = "proposta_json"
# O nome de cada etapa que é entrada, com o artigo, como a aba o diz: a etapa 2 é o
# pré-processamento na rota e a Normalização na aba, e a frase da ausência aparece na aba.
_ETAPA_NA_FRASE: Final[dict[int, str]] = {1: "a correção", 2: "a normalização"}
# O nome da entrada na fotografia da etapa. Não tem `@`, então não se confunde com identidade.
_NOME_DA_ENTRADA_NA_FOTO: Final = "entrada"

EstadoDaEtapa = Literal["nao_feita", "em_curso", "em_dia", "desatualizada"]
# Os dois motivos de desatualização, e as frases da tela são diferentes para cada um: os campos da
# própria etapa mudaram depois de feita, ou a etapa de entrada foi refeita com outros campos.
Motivo = Literal["campos_mudaram", "entrada_refeita"]
Fotografia = tuple[tuple[str, str], ...]


class CampoSemEtapaError(ValueError):
    """Um campo do pedido que `ETAPA_DO_CAMPO` não classifica."""


class EntradaDesconhecidaError(ValueError):
    """Uma entrada fora do vocabulário da etapa, inclusive uma que existe mas não para ela."""


class EntradaIndisponivelError(ValueError):
    """Uma entrada do vocabulário que não existe agora para a matriz carregada.

    A etapa dela não foi feita, ou foi feita sobre outra matriz. Levanta em vez de cair na
    original: o pedido sairia com a correção neutra, e a tela diria 'corrigida' sobre uma matriz
    sem correção.
    """


def _etapa(campo: str) -> int | None:
    try:
        return ETAPA_DO_CAMPO[campo]
    except KeyError:
        raise CampoSemEtapaError(
            f"o campo «{campo}» não tem etapa em `ETAPA_DO_CAMPO`; sem ela, mudá-lo não "
            "desatualizaria nenhuma etapa do passo a passo"
        ) from None


def _confere_etapa(etapa: int) -> None:
    if etapa not in ETAPAS:
        raise ValueError(f"a etapa {etapa!r} não existe; as do passo a passo são {ETAPAS}")


def _confere_entrada(etapa: int, entrada: str) -> Entrada:
    """A entrada como `Entrada`, ou levanta: o valor chega de um seletor, e o mypy não o prova."""
    _confere_etapa(etapa)
    for valida in ENTRADAS_DA_ETAPA[etapa]:
        if entrada == valida:
            return valida
    raise EntradaDesconhecidaError(
        f"«{entrada}» não é entrada da etapa {etapa}, {NOMES_DAS_ETAPAS[etapa]}; as dela são "
        f"{', '.join(ENTRADAS_DA_ETAPA[etapa])}"
    )


def _comparavel(valor: Any) -> str:
    return json.dumps(valor, sort_keys=True, ensure_ascii=False)


def _etapa_da_identidade(nome: str) -> int:
    return int(nome.rpartition("@")[2])


def _herdados(campos: Mapping[str, Any], etapa: int) -> dict[str, Any]:
    """Os campos que `etapa` herda das anteriores, sem o par de decisão.

    O par não muda dado nenhum, e na Análise a proposta viaja reancorada no pedido dela, então
    nunca é igual à guardada na Correção: compará-lo faria toda Análise com proposta da máquina
    parecer feita sobre outra entrada.
    """
    return {
        nome: valor
        for nome, valor in campos.items()
        if nome not in CAMPOS_DO_PAR_DE_DECISAO
        and (dono := _etapa(nome)) is not None
        and dono < etapa
    }


def fotografia(campos: Mapping[str, Any], etapa: int, identidade: Mapping[str, str]) -> Fotografia:
    """Os campos das etapas `<= etapa` e a identidade dos arquivos, numa forma comparável.

    É a fotografia do pedido montado inteiro a partir da tela. `identidade` leva a matriz (etapa 1)
    e o arquivo de fatores (etapa 2); cada chave é `"nome@etapa"`, e só entram as das etapas
    `<= etapa`. O par de decisão fica fora, porque não muda resultado. A janela compara
    `fotografia_da_etapa`; esta continua pelos testes da pertença dos campos.
    """
    itens = [
        (nome, _comparavel(valor))
        for nome, valor in campos.items()
        if nome not in CAMPOS_DO_PAR_DE_DECISAO
        and (dono := _etapa(nome)) is not None
        and dono <= etapa
    ]
    for nome, valor in identidade.items():
        if _etapa_da_identidade(nome) <= etapa:
            itens.append((nome, valor))
    return tuple(sorted(itens))


def fotografia_da_etapa(
    campos: Mapping[str, Any], etapa: int, identidade: Mapping[str, str], entrada: str
) -> Fotografia:
    """O que a tela contribui para `etapa` com `entrada`, numa forma comparável.

    É a fotografia que a janela compara. `campos` são os da tela. Entram os campos da própria etapa;
    os comuns só na entrada original, porque nas outras eles vêm da etapa feita; a matriz e a
    identidade da própria etapa (os fatores na etapa 2); e a entrada, porque trocá-la depois de
    feita é mudar um parâmetro da etapa. Os campos herdados ficam fora: se mudam, a etapa de entrada
    foi refeita, e o estado mede isso sem a tela.
    """
    escolhida = _confere_entrada(etapa, entrada)
    itens = []
    for nome, valor in campos.items():
        if nome in CAMPOS_DO_PAR_DE_DECISAO:
            continue
        dono = _etapa(nome)
        if dono == etapa or (escolhida == "original" and nome in CAMPOS_COMUNS):
            itens.append((nome, _comparavel(valor)))
    for nome, valor in identidade.items():
        if nome == IDENTIDADE_DA_MATRIZ or _etapa_da_identidade(nome) == etapa:
            itens.append((nome, valor))
    if etapa != ETAPAS[0]:
        itens.append((_NOME_DA_ENTRADA_NA_FOTO, escolhida))
    return tuple(sorted(itens))


def campos_da_rota(campos: Mapping[str, Any], etapa: int) -> dict[str, Any]:
    """O que a rota da `etapa` aceita: os campos das etapas `<= etapa`, e `paralelo`.

    O par de decisão só vai à execução: as duas prévias não o declaram, e o FastAPI descarta campo
    não declarado sem dizer nada.
    """
    aceitos: dict[str, Any] = {}
    for nome, valor in campos.items():
        if nome in CAMPOS_DO_PAR_DE_DECISAO:
            if etapa == ETAPAS[-1]:
                aceitos[nome] = valor
        elif (dono := _etapa(nome)) is None or dono <= etapa:
            aceitos[nome] = valor
    return aceitos


# As chaves da spec, derivadas de `PipelineSpec`. A rota serializa exatamente os campos desse
# dataclass, e o par de decisão compara os dois textos por igualdade: redigitar os nomes aqui
# criaria uma segunda lista que, ao divergir, transformaria toda concordância em discordância.
CHAVES_DA_SPEC: Final[tuple[str, ...]] = tuple(campo.name for campo in fields(PipelineSpec))


def _proposta_da_maquina(
    campos: dict[str, Any],
    correction: str,
    rationale: str,
    origem: str,
    *,
    outros: Mapping[str, Any] | None = None,
) -> str:
    """A linha da máquina do par de decisão, no envelope que `/execucoes` espera.

    A spec proposta é a spec do humano com os eixos da máquina no lugar: o corretor e, em `outros`,
    o número de lotes e os cortes do diagnóstico. Assim, igualdade de texto entre as duas linhas
    significa que o usuário manteve o que a máquina propôs.

    `sort_keys=True` e `ensure_ascii=False` copiam a serialização da rota, porque a comparação é
    entre dois textos e uma ordem diferente diria 'discordando' sobre specs iguais.
    """
    spec: dict[str, Any] = {chave: campos.get(chave) for chave in CHAVES_DA_SPEC}
    spec["correction"] = correction
    spec.update(outros or {})
    # Os cortes viajam no formulário como texto JSON e a rota grava a lista; a linha da máquina
    # precisa da mesma forma para a comparação por texto.
    if isinstance(spec.get("batch_cuts"), str):
        spec["batch_cuts"] = json.loads(spec["batch_cuts"])
    return json.dumps(
        {
            "spec_json": json.dumps(spec, ensure_ascii=False, sort_keys=True),
            "rationale": rationale,
            "origem": origem,
        },
        ensure_ascii=False,
    )


def _proposta_reancorada(proposta_json: str, pedido: Mapping[str, Any]) -> str:
    """A proposta da Correção feita, com a spec reancorada no `pedido` que a leva à execução.

    A tabela `decisions` compara a linha da máquina com a do humano por texto, e as chaves que
    diferem são os eixos sobrescritos. Os eixos que a máquina propõe são todos da Correção (o
    corretor, os lotes e os cortes), e numa entrada derivada eles são os da Correção feita. O resto
    da spec é o pré-processamento, que muda de dono: a Correção montou a proposta com o da tela, e a
    Análise manda o da entrada, neutro sobre a corrigida e o da Normalização feita sobre a
    normalizada. Sem reancorar, o par gravaria como sobrescritos eixos que ninguém tocou.

    A serialização é a de `_proposta_da_maquina`, porque a comparação é por texto.
    """
    externa = json.loads(proposta_json)
    spec = json.loads(externa["spec_json"])
    for nome in spec:
        if ETAPA_DO_CAMPO.get(nome) != ETAPAS[0]:
            spec[nome] = pedido.get(nome)
    externa["spec_json"] = json.dumps(spec, ensure_ascii=False, sort_keys=True)
    return json.dumps(externa, ensure_ascii=False)


def etapa_a_refazer(
    etapa: int, em_dia: Mapping[int, bool], entradas: Mapping[int, str] = CADEIA_DA_TELA
) -> int:
    """A etapa mais a montante fora de dia na cadeia das entradas de `etapa`.

    Devolve `etapa` quando todas estão em dia. É a que a frase de figura desatualizada manda
    refazer. A cadeia segue a entrada usada, e não a ordem 1-2-3: uma Análise sobre a original não
    depende da Correção, e uma sobre a corrigida salta a Normalização. O padrão é a cadeia em que
    cada etapa refaz as anteriores com os campos da tela. Vive aqui, e não no painel dos gráficos,
    porque é conta, e `janela/graficos.py` tem portão contra aritmética.
    """
    cadeia = [etapa]
    while (bruta := entradas.get(cadeia[-1])) is not None:
        anterior = ETAPA_DA_ENTRADA[_confere_entrada(cadeia[-1], bruta)]
        if anterior is None:
            break
        cadeia.append(anterior)
    for montante in reversed(cadeia):
        if not em_dia.get(montante, False):
            return montante
    return etapa


@dataclass(frozen=True)
class EtapaFeita:
    """Uma etapa disparada ou concluída: o que ela comparou, de onde partiu e o que mandou.

    `campos` são os da etapa: o pedido inteiro e o par de decisão que ela passa adiante; a prévia
    recebe deles o recorte de `campos_da_rota`. São `None` quando o disparo não os guardou, e uma
    etapa assim não serve de entrada, porque não há de onde herdar. `correcao` é o corretor efetivo
    que a Correção devolveu, depois da delegação.

    `fatores` é o arquivo de fatores que a etapa enviou, `(nome, bytes, tipo)` como o multipart o
    leva, e só a Normalização o guarda. Com `supplied_factors`, o resultado depende dos bytes do
    arquivo, e a tela pode ter trocado de arquivo depois; por isso quem parte da Normalização feita
    (os downloads dela e a Análise sobre a normalizada) reenvia estes bytes, e não os de agora.
    """

    foto: Fotografia
    entrada: Entrada = "original"
    campos: Mapping[str, Any] | None = None
    correcao: str | None = None
    fatores: tuple[str, bytes, str] | None = None

    @property
    def matriz(self) -> str | None:
        """A identidade da matriz sobre a qual a etapa rodou, lida da fotografia."""
        return dict(self.foto).get(IDENTIDADE_DA_MATRIZ)


@dataclass(frozen=True)
class EntradaDaEtapa:
    """O que a seção 'Entrada' de uma etapa mostra.

    `mais_tratada` é a padrão quando ela não é a escolhida: a pessoa escolheu ou rodou com outra, e
    a tela avisa que existe uma mais tratada sem trocar a seleção. `motivo_da_volta` diz por que a
    entrada escolhida deixou de existir, quando a seleção voltou à padrão.
    """

    escolhida: Entrada
    padrao: Entrada
    disponiveis: tuple[Entrada, ...]
    mais_tratada: Entrada | None
    motivo_da_volta: str | None


@dataclass
class EstadoDasEtapas:
    """O que foi disparado e o que terminou, com a fotografia e os campos de cada etapa.

    Uma etapa por vez: enquanto uma corre, nenhuma outra dispara. É a única trava; a Normalização e
    a Análise disparam sem a Correção, sobre a entrada que a pessoa escolheu.
    """

    _feitas: dict[int, EtapaFeita] = field(default_factory=dict)
    _em_curso: dict[int, EtapaFeita] = field(default_factory=dict)
    # A escolha de entrada é por matriz, (etapa, identidade da matriz): rodar na entrada de recuo
    # sob a matriz B é escolhê-la para B, e não apaga a escolha feita sob A, que volta quando a
    # pessoa volta a A. A última escolha explícita, de qualquer matriz, fica à parte: é ela que diz
    # que a entrada X deixou de existir sob uma matriz em que nada foi escolhido.
    _escolhas: dict[tuple[int, str], Entrada] = field(default_factory=dict)
    _ultima_escolha: dict[int, Entrada] = field(default_factory=dict)

    def disparou(
        self,
        etapa: int,
        foto: Fotografia,
        *,
        campos: Mapping[str, Any] | None = None,
        entrada: str = "original",
        fatores: tuple[str, bytes, str] | None = None,
    ) -> None:
        """Marca a etapa em curso com o que ela leva.

        `campos` é o que `campos_da_entrada` deu, e `fatores` o arquivo de fatores que ela enviou
        (ver `EtapaFeita`). Rodar com a entrada que a tela mostrava é escolhê-la para esta matriz:
        sem isso, uma escolha que deixou de existir ficaria guardada por baixo do recuo e voltaria
        quando a entrada reaparecesse, deixando a etapa recém-feita desatualizada sem que nada
        mudasse. A escolha feita sob outra matriz continua a valer lá. A Correção não tem escolha:
        parte sempre da matriz carregada.
        """
        escolhida = _confere_entrada(etapa, entrada)
        matriz = dict(foto).get(IDENTIDADE_DA_MATRIZ)
        if etapa != ETAPAS[0] and matriz is not None:
            self._escolhas[(etapa, matriz)] = escolhida
        self._em_curso[etapa] = EtapaFeita(
            foto=foto,
            entrada=escolhida,
            campos=None if campos is None else dict(campos),
            fatores=fatores,
        )

    def concluiu(self, etapa: int, *, correcao: str | None = None) -> None:
        """A etapa voltou com `200`; `correcao` é o corretor efetivo que a Correção devolveu."""
        disparo = self._em_curso.pop(etapa, None)
        if disparo is not None:
            self._feitas[etapa] = replace(disparo, correcao=correcao)

    def falhou(self, etapa: int) -> None:
        self._em_curso.pop(etapa, None)

    def em_curso(self) -> tuple[int, ...]:
        return tuple(sorted(self._em_curso))

    def feita(self, etapa: int) -> EtapaFeita | None:
        """A última vez que a etapa terminou com `200`, ou `None`. É dela que saem os downloads."""
        return self._feitas.get(etapa)

    def motivos(self, etapa: int, atual: Fotografia) -> frozenset[Motivo]:
        """Por que a etapa feita não está em dia; vazio quando está, ou quando não foi feita."""
        feita = self._feitas.get(etapa)
        if feita is None:
            return frozenset()
        motivos: set[Motivo] = set()
        if feita.foto != atual:
            motivos.add("campos_mudaram")
        if self._entrada_refeita(etapa, feita):
            motivos.add("entrada_refeita")
        return frozenset(motivos)

    def em_dia(self, etapa: int, atual: Fotografia) -> bool:
        """Se a etapa feita por último é a da tela, mesmo com ela a ser refeita agora."""
        return etapa in self._feitas and not self.motivos(etapa, atual)

    def estado(self, etapa: int, atual: Fotografia) -> EstadoDaEtapa:
        if etapa in self._em_curso:
            return "em_curso"
        if etapa not in self._feitas:
            return "nao_feita"
        return "em_dia" if self.em_dia(etapa, atual) else "desatualizada"

    def pode_disparar(self, etapa: int) -> bool:
        """Se a etapa pode disparar: só quando nenhuma está em curso.

        A etapa k não pede a k-1 em dia, porque a dependência é a da entrada.
        """
        _confere_etapa(etapa)
        return not self._em_curso

    def etapa_a_refazer(self, etapa: int, atuais: Mapping[int, Fotografia]) -> int | None:
        """A etapa que o aviso desta manda refazer, ou `None` com ela e a cadeia dela em dia.

        A etapa em dia cuja entrada ficou fora de dia também avisa: o resultado dela corresponde à
        entrada que existe, mas essa entrada já não é a da tela. O aviso aponta sempre para a mais a
        montante fora de dia, porque refazer a de baixo antes repetiria o trabalho. `atuais` são as
        fotos de `fotografia_da_etapa` com a entrada que cada etapa tem escolhida.
        """
        _confere_etapa(etapa)
        if etapa not in self._feitas:
            return None
        em_dia = {feita: self.em_dia(feita, atuais[feita]) for feita in self._feitas}
        entradas = {feita: registro.entrada for feita, registro in self._feitas.items()}
        alvo = etapa_a_refazer(etapa, em_dia, entradas)
        return None if alvo == etapa and em_dia[etapa] else alvo

    # ------------------------------------------------------------------
    # As entradas
    # ------------------------------------------------------------------

    def entrada(self, etapa: int, matriz: str) -> EntradaDaEtapa:
        """As entradas que existem para a matriz carregada, a padrão e a selecionada.

        A seleção é a escolha explícita da pessoa. Sem ela, é a entrada com que a etapa rodou da
        última vez, porque rodar com a entrada que a tela mostrava é escolhê-la, e uma Correção
        concluída depois não deve trocar a entrada de uma Análise já feita nem a desatualizar. Sem
        as duas, é a padrão, a mais tratada que existir.
        """
        _confere_etapa(etapa)
        disponiveis = tuple(
            entrada
            for entrada in ENTRADAS_DA_ETAPA[etapa]
            if self._ausencia(entrada, matriz) is None
        )
        padrao = disponiveis[-1]
        feita = self._feitas.get(etapa)
        pedida = (
            self._escolhas.get((etapa, matriz))
            or (None if feita is None else feita.entrada)
            or self._ultima_escolha.get(etapa)
        )
        if pedida is None:
            return EntradaDaEtapa(padrao, padrao, disponiveis, None, None)
        ausencia = self._ausencia(pedida, matriz)
        if ausencia is None:
            mais_tratada = None if padrao == pedida else padrao
            return EntradaDaEtapa(pedida, padrao, disponiveis, mais_tratada, None)
        motivo = (
            f"a entrada «{pedida}» deixou de existir ({ausencia}), e a seleção voltou à «{padrao}»"
        )
        return EntradaDaEtapa(padrao, padrao, disponiveis, None, motivo)

    def escolhe_entrada(self, etapa: int, entrada: str, matriz: str) -> None:
        """A escolha da pessoa para esta matriz, que persiste enquanto a entrada existir."""
        escolhida = _confere_entrada(etapa, entrada)
        self._levanta_se_ausente(etapa, escolhida, matriz)
        self._escolhas[(etapa, matriz)] = escolhida
        self._ultima_escolha[etapa] = escolhida

    def ausencia(self, entrada: str, matriz: str) -> str | None:
        """Por que a entrada não existe para a matriz carregada, numa frase que a tela diz.

        `None` quando ela existe. É o que a seção 'Entrada' diz quando a oferta é só a original.
        """
        return self._ausencia(_confere_entrada(ETAPAS[-1], entrada), matriz)

    def nome_da_entrada(self, entrada: str, *, corretores: Mapping[str, str] | None = None) -> str:
        """O nome da entrada pelo que ela é: o corretor da corrigida, a origem da normalizada.

        `corretores` dá o nome que a tela mostra para cada corretor (`serrf` é 'SERRF'); um corretor
        fora dele aparece como veio.
        """
        nomes = corretores or {}
        conferida = _confere_entrada(ETAPAS[-1], entrada)
        if conferida == "original":
            return "original"
        _, feita, campos = self._fonte(conferida)
        if conferida == "corrigida":
            corretor = str(feita.correcao)
            return f"corrigida — {nomes.get(corretor, corretor)}"
        if feita.entrada == "original":
            return "normalizada (sem correção)"
        corretor = str(campos["correction"])
        return f"normalizada (corrigida — {nomes.get(corretor, corretor)})"

    def passou_pela_imputacao(self, entrada: str) -> bool:
        """Se a matriz da `entrada` passou pela imputação, para o aviso dos zeros da Análise.

        A original e a corrigida não passam: a Análise sobre elas leva o pré-processamento neutro. A
        normalizada passou quando a Normalização feita imputou; `exclude` conta, porque tira as
        features com zero em vez de deixá-los entrar como valor.
        """
        conferida = _confere_entrada(ETAPAS[-1], entrada)
        if conferida != "normalizada":
            return False
        _, _, campos = self._fonte(conferida)
        return campos.get("imputation", "none") != "none"

    def campos_da_entrada(
        self, etapa: int, entrada: str, campos_da_tela: Mapping[str, Any], matriz: str
    ) -> dict[str, Any]:
        """Os campos que a etapa leva com `entrada`: os dela, da tela, e os herdados da entrada.

        - na original, a correção e, na Análise, o pré-processamento vão neutros, os comuns vêm da
          tela, e o par de decisão diz que não houve proposta (`sem_proposta`);
        - na corrigida, os campos da Correção feita, com o corretor efetivo no lugar do pedido e
          sem a delegação: mandar `auto` de novo mediria o Palpite outra vez, e o resultado poderia
          ser outro corretor sem a tela saber. A semente e os lotes seguem a mesma regra;
        - na normalizada, os campos das etapas 1 e 2 da Normalização feita.

        Na Correção, o par de decisão da tela vai junto, porque é ele que as entradas derivadas dela
        levam à execução, onde a proposta é reancorada no pedido da Análise
        (`_proposta_reancorada`). A prévia recebe disto o recorte de `campos_da_rota`.
        """
        conferida = _confere_entrada(etapa, entrada)
        self._levanta_se_ausente(etapa, conferida, matriz)
        proprios: dict[str, Any] = {}
        comuns: dict[str, Any] = {}
        par_da_tela: dict[str, Any] = {}
        for nome, valor in campos_da_tela.items():
            if nome in CAMPOS_DO_PAR_DE_DECISAO:
                par_da_tela[nome] = valor
            elif (dono := _etapa(nome)) is None or dono == etapa:
                proprios[nome] = valor
            elif nome in CAMPOS_COMUNS:
                comuns[nome] = valor
        if etapa == ETAPAS[0]:
            return {**proprios, **par_da_tela}
        if conferida == "original":
            herdado = {**_CORRECAO_NEUTRA, **comuns, **_PAR_SEM_PROPOSTA}
            if etapa == ETAPAS[-1]:
                herdado.update(_PREPROCESSAMENTO_NEUTRO)
        else:
            herdado = self._herdado(etapa, conferida)
        pedido = {**herdado, **proprios}
        proposta = pedido.get(_PROPOSTA)
        if etapa == ETAPAS[-1] and proposta:
            pedido[_PROPOSTA] = _proposta_reancorada(proposta, pedido)
        return pedido

    # ------------------------------------------------------------------
    # O que as entradas leem das etapas feitas
    # ------------------------------------------------------------------

    def _ausencia(self, entrada: Entrada, matriz: str | None) -> str | None:
        """Por que a entrada não existe (para a matriz, quando ela é dada), ou `None`."""
        anterior = ETAPA_DA_ENTRADA[entrada]
        if anterior is None:
            return None
        feita = self._feitas.get(anterior)
        nome = _ETAPA_NA_FRASE[anterior]
        if feita is None:
            return f"{nome} não foi feita"
        if feita.campos is None or (entrada == "corrigida" and feita.correcao is None):
            return f"{nome} feita não guardou os campos que mandou e o corretor efetivo"
        if matriz is not None and feita.matriz != matriz:
            return f"{nome} feita é de outra matriz — o par de arquivos ou o modo mudou depois"
        return None

    def _levanta_se_ausente(self, etapa: int, entrada: Entrada, matriz: str) -> None:
        ausencia = self._ausencia(entrada, matriz)
        if ausencia is not None:
            raise EntradaIndisponivelError(
                f"a entrada «{entrada}» da etapa {etapa}, {NOMES_DAS_ETAPAS[etapa]}, não existe: "
                f"{ausencia}"
            )

    def _fonte(self, entrada: Entrada) -> tuple[int, EtapaFeita, Mapping[str, Any]]:
        """A etapa feita que uma entrada derivada é, com o número e os campos dela.

        Não olha a matriz. Levanta quando a etapa não serve de entrada.
        """
        ausencia = self._ausencia(entrada, None)
        anterior = ETAPA_DA_ENTRADA[entrada]
        feita = None if anterior is None else self._feitas.get(anterior)
        if ausencia is not None or anterior is None or feita is None or feita.campos is None:
            raise EntradaIndisponivelError(f"a entrada «{entrada}» não existe: {ausencia}")
        return anterior, feita, feita.campos

    def _herdado(self, etapa: int, entrada: Entrada) -> dict[str, Any]:
        """Os campos das etapas antes de `etapa` e o par de decisão, como a etapa feita os dá."""
        anterior, fonte, campos = self._fonte(entrada)
        herdado = {
            nome: valor
            for nome, valor in campos.items()
            if nome in CAMPOS_DO_PAR_DE_DECISAO
            or ((dono := _etapa(nome)) is not None and dono <= anterior)
        }
        if entrada == "corrigida":
            herdado = {**_PAR_VAZIO, **herdado, "correction": fonte.correcao}
            herdado.pop("auto", None)
            if etapa == ETAPAS[-1]:
                herdado.update(_PREPROCESSAMENTO_NEUTRO)
        return herdado

    def etapas_reproduzidas(self, etapa: int) -> frozenset[int]:
        """As etapas guardadas cujos dados o resultado da `etapa` em curso reproduz.

        São as da cadeia da entrada dela (a Correção para a corrigida; a Normalização e, se ela
        partiu da corrigida, a Correção para a normalizada), enquanto os campos herdados forem os
        que essas etapas guardam. Se a etapa guardada foi refeita depois do disparo, o que volta
        descreve a anterior. Diz se o desfecho de uma etapa pode escrever as figuras de outra: cada
        etapa é dona dos dados das próprias figuras, e só escreve nos de outra quem a reproduz. Sem
        a etapa em curso, ou com ela sobre a original, nenhuma.
        """
        registro = self._em_curso.get(etapa)
        reproduzidas: set[int] = set()
        corrente = etapa
        while registro is not None and (anterior := ETAPA_DA_ENTRADA[registro.entrada]):
            if not self._herdou_o_guardado(corrente, registro):
                break
            reproduzidas.add(anterior)
            registro, corrente = self._feitas.get(anterior), anterior
        return frozenset(reproduzidas)

    def _herdou_o_guardado(self, etapa: int, registro: EtapaFeita) -> bool:
        """Se os campos que `registro` herdou da entrada dele são os que a etapa guardada dá agora.

        O par de decisão fica fora da comparação: ele não muda dado, e na Análise a proposta viaja
        reancorada no pedido dela (`_proposta_reancorada`), então nunca é igual à guardada. Sem os
        campos do disparo não há como dizer, e a resposta é não.
        """
        if registro.campos is None:
            return False
        try:
            agora = self._herdado(etapa, registro.entrada)
        except EntradaIndisponivelError:
            return False

        return _herdados(registro.campos, etapa) == _herdados(agora, etapa)

    def _entrada_refeita(self, etapa: int, feita: EtapaFeita) -> bool:
        """Se os campos que a etapa herdou já não são os que a etapa de entrada daria agora.

        A etapa de entrada refeita com os mesmos campos não desatualiza, porque o resultado seria o
        mesmo. Refeita por um caminho que não guarda os campos, não há como dizer que é a mesma, e o
        lado seguro é pedir para refazer. O par de decisão fica fora da comparação pelo motivo de
        `_herdou_o_guardado`; comparado, ele poria a Análise sobre a corrigida como desatualizada no
        instante em que ela terminasse, sempre que houvesse proposta da máquina.
        """
        if feita.entrada == "original" or feita.campos is None:
            return False
        try:
            agora = self._herdado(etapa, feita.entrada)
        except EntradaIndisponivelError:
            return True

        return _herdados(feita.campos, etapa) != _herdados(agora, etapa)
