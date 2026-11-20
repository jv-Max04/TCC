"""Limiares da classificação heurística do diagnóstico.

A política é parâmetro: `DiagnosisPolicy` não tem valor padrão nos seis limiares
escalares, para que quem classifica declare os valores que usou. O núcleo não lê
arquivo de configuração; quem carrega os valores os passa para dentro.

Origem dos valores iniciais. O corte de RSD mediano de QC em 0,20 coincide com o
critério de aceitação de RSD < 20% de Broadhurst et al. (2018, Metabolomics), que
também recomenda inspecionar a resposta contra a ordem de injeção, sem dar limiar; o
0,30 da correlação é corte deste trabalho. Os cortes 2,0 (degrau) e 0,487 (deriva)
foram medidos nos quatro arquivos de referência (ver `trajectory.py`). A separação
entre deriva contínua e degrau discreto tem contraparte publicada: Čuklina et al.
(2021, Mol. Syst. Biol. 17(8):e10240) e Westgard et al. (1977, Clin. Chem.
23(10):1857). Nos quatro arquivos de referência a regra 1 usa o limiar calibrado por
arquivo, e três deles ficam acima de 0,20.
"""

from collections.abc import Mapping
from dataclasses import dataclass

__all__ = [
    "LIMIARES_INICIAIS",
    "MEDICOES_DE_REFERENCIA_SHA256",
    "RSD_MEDIANA_MAX_POR_MODO",
    "DiagnosisPolicy",
    "InvalidPolicyError",
    "mapa_calibrado_para",
]


class InvalidPolicyError(ValueError):
    """Limiar fora do domínio em que ele significa o que o nome diz.

    Separada de `ValueError` genérico para que o traceback diga que é configuração
    errada, e não defeito de programação.
    """


# Teto dos limiares do mapa por modo. É 2 e não 1 porque RSD é `sd/mean` e passa de 1:
# o `pos_hilic` mede 1,4381 de RSD mediano de QC, e o maior valor visto em matriz
# sintética é 1,8013. O valor 2 ainda separa fração de percentual, já que o menor RSD
# mediano real em percentual seria 15,50. O limiar escalar mantém teto 1 porque nenhum
# uso dele precisa de mais.
_TETO_FRACAO_POR_MODO = 2.0


@dataclass(frozen=True)
class DiagnosisPolicy:
    """Os limiares das regras 0 a 3 da classificação. Nenhum dos seis escalares tem padrão.

    Um padrão embutido faria a calibração parecer opcional; quem constrói a política
    declara os valores. Os valores iniciais estão em `LIMIARES_INICIAIS`.

    - `max_spread`: regra 0, instabilidade máxima das métricas estruturais avaliáveis.
    - `rsd_mediana_max`: regra 1, RSD mediano de QC em fração; vale para modo fora do mapa.
    - `rsd_mediana_max_por_modo`: regra 1, o limiar por modo quando há mapa.
    - `pc1_injection_correlation_max`: regra 1, correlação de PC1 com a ordem de injeção.
    - `step_sustain_min`: regra 2, mínimo de `sustained_step_ratio`.
    - `qc_monotonicity_min`: regra 3, mínimo de `monotonicity_net_over_gross`.
    - `trajectory_movement_min`: piso de `Σ|Δv|` da trajetória, em unidades de score.
    """

    max_spread: float
    rsd_mediana_max: float
    pc1_injection_correlation_max: float
    step_sustain_min: float
    qc_monotonicity_min: float
    trajectory_movement_min: float

    # Limiar da regra 1 por modo. A escala do RSD mediano de QC varia quase dez vezes
    # entre modos (0,1550 no `neg_c18`, 1,4381 no `pos_hilic`), e um corte único não
    # serve a todos. `None`, o padrão, usa só o escalar. Modo fora do mapa também cai
    # no escalar, em vez de levantar `KeyError` no meio da classificação.
    rsd_mediana_max_por_modo: Mapping[str, float] | None = None

    def rsd_mediana_max_de(self, mode: str) -> float:
        """O limiar da regra 1 para o modo; modo fora do mapa recebe o escalar.

        Modo desconhecido é o caso de um estudo novo, e ele deve ser diagnosticado
        com o limiar geral em vez de recusado.
        """
        mapa = self.rsd_mediana_max_por_modo
        if mapa is None:
            return self.rsd_mediana_max
        return mapa.get(mode, self.rsd_mediana_max)

    def __post_init__(self) -> None:
        """Recusa limiar fora do domínio, com atenção à unidade do RSD.

        `rsd()` devolve fração. Um limiar em percentual faria a regra 1 nunca casar,
        e uma matriz limpa seria classificada pelas regras seguintes sem aviso. A
        guarda pega a confusão na construção.
        """
        if not self.rsd_mediana_max > 0 or self.rsd_mediana_max > 1:
            raise InvalidPolicyError(
                f"`rsd_mediana_max` tem de estar em (0, 1]; recebi {self.rsd_mediana_max!r}. "
                "O RSD deste projeto é FRAÇÃO e não percentual - 0,20 significa 20% -, então "
                "limiar acima de 1 nunca é alcançado por um RSD em fração e todo limiar em "
                "percentual o excede. Valor em percentual aqui faria a regra 1 nunca casar, a "
                "cascata escorregar para a regra 2 e um modo sem degrau sair `step_change`."
            )
        for modo, valor in (self.rsd_mediana_max_por_modo or {}).items():
            # A mesma guarda de unidade do escalar: valor em percentual faria a regra 1
            # nunca casar naquele modo.
            if not valor > 0 or valor > _TETO_FRACAO_POR_MODO:
                raise InvalidPolicyError(
                    f"`rsd_mediana_max_por_modo[{modo!r}]` tem de estar em "
                    f"(0, {_TETO_FRACAO_POR_MODO}]; recebi {valor!r}. O RSD deste projeto é "
                    "FRAÇÃO e não percentual."
                )
        for nome, valor in (
            ("pc1_injection_correlation_max", self.pc1_injection_correlation_max),
            ("qc_monotonicity_min", self.qc_monotonicity_min),
        ):
            if not valor > 0 or valor > 1:
                raise InvalidPolicyError(
                    f"`{nome}` tem de estar em (0, 1]; recebi {valor!r}. É comparado contra "
                    "uma quantidade normalizada - correlação absoluta e razão de líquido "
                    "sobre bruto -, e limiar fora do intervalo torna a regra inalcançável ou "
                    "sempre verdadeira, que são as duas formas de portão que não decide."
                )
        for nome, valor in (
            ("max_spread", self.max_spread),
            ("step_sustain_min", self.step_sustain_min),
            ("trajectory_movement_min", self.trajectory_movement_min),
        ):
            if not valor > 0:
                raise InvalidPolicyError(
                    f"`{nome}` tem de ser estritamente positivo; recebi {valor!r}. Zero torna "
                    "a comparação vácua: com `max_spread = 0` nenhuma métrica é avaliável e "
                    "tudo sai `unassessable`; com os outros dois em zero a regra casa sempre."
                )


# Valores iniciais dos limiares. São o padrão da requisição, não de `DiagnosisPolicy`:
# quem constrói a política continua a passá-los, e o valor usado viaja com a
# requisição. Ficam neste módulo porque a API e a política de decisão partem deles,
# e `cli.py` não pode importar da API sem ciclo.
LIMIARES_INICIAIS = {
    "max_spread": 3.0,
    "rsd_mediana_max": 0.20,
    "pc1_injection_correlation_max": 0.30,
    "step_sustain_min": 2.0,
    "qc_monotonicity_min": 0.487,
    "trajectory_movement_min": 1.0,
}

# Limiar da regra 1 por modo, calibrado nos quatro arquivos de referência. Cada número
# é o de maior acurácia sobre matrizes sintéticas do modo, sujeito a manter a matriz
# real na classe esperada. Fica separado de `LIMIARES_INICIAIS` porque aqueles são
# escalares que a requisição pode sobrescrever, e este é um mapa.
#
# As chaves `sintetico_<modo>` existem porque o gerador rotula assim as matrizes que
# produz; sem elas o mapa cairia no escalar sem avisar.
RSD_MEDIANA_MAX_POR_MODO: dict[str, float] = {
    "pos_hilic": 1.0466,
    "pos_c18": 0.2346,
    "neg_hilic": 0.4915,
    "neg_c18": 0.1639,
    "sintetico_pos_hilic": 1.0466,
    "sintetico_pos_c18": 0.2346,
    "sintetico_neg_hilic": 0.4915,
    "sintetico_neg_c18": 0.1639,
}

# Os quatro números acima valem para quatro arquivos, não para quatro rótulos de modo:
# outro estudo pode usar o rótulo `pos_hilic` e não deve herdar o limiar de 1,0466. A
# identidade é o `sha256` do conteúdo de cada `measurements_*.csv` de referência,
# copiado de `scripts/reference_data.sha256` porque o núcleo não lê arquivo.
# `tests/test_limiar_por_arquivo.py` confere a cópia contra o manifesto. Fora desses
# quatro arquivos vale o escalar.
MEDICOES_DE_REFERENCIA_SHA256: dict[str, str] = {
    "6275b148dafc3c02360d0cce91898a853b5776f7a356928545d3c9b93e34577c": "neg_c18",
    "5805f01f5201c3ca82e089f8c96d09469ee5233be52b0757da8f5781be8eb19c": "neg_hilic",
    "d50ec62120aaac52d403bfd680798d84aef708fffa4badd70bae70cdc464b025": "pos_c18",
    "3600c7e82d76f7b869738a10b7f038dd52aacb757b8a541d2305d5e083895720": "pos_hilic",
}


def mapa_calibrado_para(sha256_das_medicoes: str, mode: str) -> dict[str, float] | None:
    """O mapa de RSD por modo para esta entrada, ou `None` quando vale o escalar.

    Só os quatro arquivos calibrados recebem o limiar calibrado, cada um o seu, sob o
    rótulo de modo que a requisição usar: o limiar pertence ao arquivo, não ao nome.
    Qualquer outro conteúdo, inclusive o mesmo arquivo regravado por uma planilha,
    devolve `None`, que em `DiagnosisPolicy.rsd_mediana_max_por_modo` significa sem mapa.
    """
    modo_de_referencia = MEDICOES_DE_REFERENCIA_SHA256.get(sha256_das_medicoes)
    if modo_de_referencia is None:
        return None
    return {mode: RSD_MEDIANA_MAX_POR_MODO[modo_de_referencia]}
