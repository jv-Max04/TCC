"""O pedido que os controles do painel do pipeline montam, pelas três partes que a produção chama.

**Ele existe porque `PainelDePipeline.campos_da_execucao` saiu em 09/11/2026**, com o botão
«Executar pipeline», que era o único a chamá-lo. Aquele método juntava três passos num pedido só
de `/execucoes`; desde a saída do
botão, cada etapa monta o seu pedido a partir deles, na janela (`JanelaPrincipal._dispara_etapa`):

| Passo | Quem o chama em produção |
|---|---|
| `campos_base` | `_campos_atuais`, a tela de que as três etapas partem |
| `proposta_do_pedido` | a Correção, no clique dela: é a linha da máquina |
| `confere_a_combinacao` | a Normalização, no clique dela: a recusa local |

**Os testes que liam o pedido inteiro passam a lê-lo por aqui**, e o que eles exercitam continua a
ser código de produção: as três partes, na ordem em que o método antigo as chamava — a recusa
primeiro, o pedido depois, e a proposta por último, montada sobre o pedido COMPLETO (a ordem é a do
docstring que o método tinha: com o corte acrescentado depois, a linha da máquina sairia sem ele).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - só para o mypy
    from refino.desktop.janela.pipeline import PainelDePipeline


def campos_da_tela(
    pipeline: PainelDePipeline,
    *,
    modo: str,
    correcao: str,
    n_lotes: int,
    eixos: list[str] | tuple[str, ...] = (),
    recomendacao: dict[str, Any] | None = None,
    rsd_cutoff: float | None = None,
) -> dict[str, Any]:
    """O pedido dos controles: a recusa da combinação proibida, os campos e a linha da máquina."""
    pipeline.confere_a_combinacao()
    campos = pipeline.campos_base(modo, correcao, n_lotes, eixos=eixos, rsd_cutoff=rsd_cutoff)
    campos.update(pipeline.proposta_do_pedido(campos, eixos, recomendacao))
    return campos
