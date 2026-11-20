"""O artefato da política aprendida: exporta as árvores para um dicionário e as lê de volta.

O modelo é guardado como JSON, e não como `joblib` ou `pickle`: o JSON se lê num diff e não
depende da versão do `scikit-learn`, e treinar ao arranque exigiria levar o corpus rotulado junto
com o produto. As duas árvores têm profundidade 3, no máximo 8 folhas cada, então o arquivo é
pequeno o bastante para ser lido por uma pessoa.

Este módulo não abre arquivo: `exporta` devolve um dicionário e `de_dict` recebe um. Quem lê o
disco é a fronteira (a CLI, a rota, o script gerador). A ordem das características é o contrato:
um artefato gerado sob outra ordem não daria erro ao recomendar, só uma resposta plausível e
errada. Por isso `de_dict` confere a ordem contra `FEATURE_NAMES` e levanta.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, cast, get_args

from refino.decision.features import FEATURE_NAMES, feature_vector
from refino.decision.model import descreve_caminho, monta_recomendacao
from refino.decision.policy import ABSTENTION_MARGIN, Recommendation
from refino.domain.diagnosis import DefectProfile
from refino.domain.labels import Correction

__all__ = [
    "VERSAO_DO_ARTEFATO",
    "ArtefatoInvalidoError",
    "PoliticaDoArtefato",
    "exporta",
]

# Versão do formato, e não do modelo. Sobe quando a forma do dicionário muda de um jeito que um
# carregador antigo leria errado. Corpus e semente ficam em `procedencia`.
VERSAO_DO_ARTEFATO = 1


class ArtefatoInvalidoError(ValueError):
    """O artefato não pode ser lido como política.

    Levanta em vez de cair num padrão: uma política substituta tornaria 'carregou' e 'não carregou'
    indistinguíveis, e o usuário receberia uma recomendação que ninguém treinou.
    """


def _arvore_para_nos(arvore: Any, classes: Any | None) -> list[dict[str, Any]]:
    """Os nós da árvore, na indexação do próprio `scikit-learn`.

    `classes` não nulo marca a árvore de classificação: a folha grava o rótulo previsto, já
    resolvido, em vez do vetor de contagens. Gravar as contagens obrigaria o carregador a
    reimplementar o desempate do `argmax` do `scikit-learn`.
    """
    nos: list[dict[str, Any]] = []
    for indice in range(int(arvore.node_count)):
        esquerda = int(arvore.children_left[indice])
        if esquerda == -1:
            valores = arvore.value[indice][0]
            valor = (
                str(classes[int(valores.argmax())]) if classes is not None else float(valores[0])
            )
            nos.append({"folha": True, "valor": valor})
        else:
            nos.append(
                {
                    "folha": False,
                    "caracteristica": int(arvore.feature[indice]),
                    "limiar": float(arvore.threshold[indice]),
                    "esquerda": esquerda,
                    "direita": int(arvore.children_right[indice]),
                }
            )
    return nos


def exporta(
    politica: Any,
    *,
    corpus: Path | str,
    n_corpus: int,
    gerado_em: str | None = None,
) -> dict[str, Any]:
    """O dicionário versionável de uma `LearnedPolicy` ajustada.

    A árvore de margem ausente sai como `None`, e não como zero: `_margem` é `None` quando nenhuma
    margem do corpus é finita, o que significa 'não medida', e não 'medida e nula'.

    `corpus` entra só pelo nome do arquivo, porque caminho absoluto não é portável entre máquinas.
    """
    if not getattr(politica, "fitted", False):
        raise ArtefatoInvalidoError(
            "a política não está ajustada: exportar agora gravaria um modelo que ninguém treinou"
        )
    return {
        "versao": VERSAO_DO_ARTEFATO,
        "caracteristicas": list(FEATURE_NAMES),
        "limiar_de_abstencao": float(politica.threshold),
        "corretor": {
            "nos": _arvore_para_nos(politica._corretor.tree_, politica._corretor.classes_)
        },
        "margem": (
            {"nos": _arvore_para_nos(politica._margem.tree_, None)}
            if politica._margem is not None
            else None
        ),
        "procedencia": {
            "semente": int(politica.seed),
            "profundidade_maxima": int(politica.max_depth),
            "corpus": Path(corpus).name,
            "n_corpus": int(n_corpus),
            "gerado_em": gerado_em or date.today().isoformat(),
        },
    }


@dataclass(frozen=True)
class PoliticaDoArtefato:
    """A política aprendida, reconstruída do artefato, sem `scikit-learn` no caminho de leitura.

    Recomenda o mesmo que a `LearnedPolicy` ajustada de que foi exportada: o corretor e também a
    confiança.
    """

    corretor: tuple[dict[str, Any], ...]
    margem: tuple[dict[str, Any], ...] | None
    threshold: float = ABSTENTION_MARGIN
    name: str = "learned"

    @classmethod
    def de_dict(cls, dados: Any) -> PoliticaDoArtefato:
        """Valida o dicionário e reconstrói a política.

        Cada recusa nomeia o que está errado, porque a mensagem é lida por quem tem de decidir
        entre regerar o artefato e corrigir o código.
        """
        if not isinstance(dados, dict) or not dados:
            raise ArtefatoInvalidoError("o artefato está vazio ou não é um objeto JSON")

        caracteristicas = dados.get("caracteristicas")
        if list(caracteristicas or []) != list(FEATURE_NAMES):
            raise ArtefatoInvalidoError(
                "a ordem das características do artefato diverge de `FEATURE_NAMES`. Um artefato "
                "sob outra ordem NÃO levanta ao recomendar: cada nó compara um número com um "
                "limiar e a saída fica plausível e errada. Regere o artefato"
            )

        corretor = (dados.get("corretor") or {}).get("nos")
        if not corretor:
            raise ArtefatoInvalidoError("o artefato não traz a árvore do corretor")

        # `Correction` é `Literal`, que só o mypy confere, e um rótulo lido de disco nunca passou
        # por ele. Sem esta guarda, uma folha com valor inválido viraria uma correção que nenhum
        # corretor conhece, e o erro só apareceria mais adiante.
        validas = set(get_args(Correction))
        invalidas = sorted(
            {
                str(no.get("valor"))
                for no in corretor
                if no.get("folha") and str(no.get("valor")) not in validas
            }
        )
        if invalidas:
            raise ArtefatoInvalidoError(
                f"folha(s) do corretor com rótulo fora do domínio: {invalidas}. "
                f"Os válidos são {sorted(validas)}"
            )

        bruto_margem = dados.get("margem")
        nos_margem = (bruto_margem or {}).get("nos") if bruto_margem is not None else None

        return cls(
            corretor=tuple(corretor),
            margem=tuple(nos_margem) if nos_margem else None,
            threshold=float(dados.get("limiar_de_abstencao", ABSTENTION_MARGIN)),
        )

    def recommend(self, profile: DefectProfile) -> Recommendation:
        x = feature_vector(profile)
        correcao, passos = self._percorre(self.corretor, x)
        if self.margem is None:
            margem = float("nan")
        else:
            valor, _ = self._percorre(self.margem, x)
            margem = float(valor)
        # `cast` porque `Correction` é `Literal` e não se instancia. `de_dict` já recusou qualquer
        # folha fora do domínio.
        return monta_recomendacao(
            correcao=cast(Correction, correcao),
            margem=margem,
            threshold=self.threshold,
            caminho=descreve_caminho(passos),
            profile=profile,
        )

    def caminho_do_corretor(self, profile: DefectProfile) -> str:
        """Só o caminho de decisão da árvore do corretor, sem a margem nem a confiança.

        Serve à segunda opinião, dada pela política treinada contra a verdade. Nela a árvore de
        margem prediz a distância relativa de erro contra a verdade, e não de RSD, enquanto
        `ABSTENTION_MARGIN` foi derivado na escala do RSD. A confiança teria o mesmo nome e
        mediria outra coisa, então a segunda opinião leva o corretor e o caminho, sem ela.
        """
        _, passos = self._percorre(self.corretor, feature_vector(profile))
        return descreve_caminho(passos)

    def predict_margin(self, profile: DefectProfile) -> float:
        """A margem predita, como em `LearnedPolicy.predict_margin`; `nan` sem árvore de margem."""
        if self.margem is None:
            return float("nan")
        valor, _ = self._percorre(self.margem, feature_vector(profile))
        return float(valor)

    @staticmethod
    def _percorre(
        nos: tuple[dict[str, Any], ...], x: Any
    ) -> tuple[Any, list[tuple[int, float, float]]]:
        """Desce a árvore e devolve `(valor da folha, passos visitados)`.

        O laço é limitado pelo número de nós: um artefato com ciclo, por edição à mão ou por
        corrupção, travaria o processo em vez de levantar.
        """
        indice = 0
        passos: list[tuple[int, float, float]] = []
        for _ in range(len(nos) + 1):
            no = nos[indice]
            if no.get("folha"):
                return no["valor"], passos
            caracteristica = int(no["caracteristica"])
            limiar = float(no["limiar"])
            valor = float(x[caracteristica])
            passos.append((caracteristica, valor, limiar))
            indice = int(no["esquerda"]) if valor <= limiar else int(no["direita"])
        raise ArtefatoInvalidoError(
            "a árvore do artefato não termina numa folha: há ciclo ou índice de filho inválido"
        )
