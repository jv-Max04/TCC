"""Progresso em memória das execuções longas: o multiverso e os trabalhos que correm corretor.

O progresso avança em incrementos observáveis, uma leitura feita durante a execução devolve valor
intermediário, e uma falha no meio produz estado terminal nomeado em vez de contador preso. Por
isso o estado não é um par de inteiros: `7 de 12` para sempre não se distingue de `7 de 12` agora.

- `pendente`: registrada, ainda não começou.
- `executando`: correndo, com `feito` de `total`.
- `concluido`: `feito == total`, com o resultado retido.
- `falhou`: morreu no meio, com a célula ou a etapa nomeada.
- `cancelado`: parou a pedido.

O contador é o `progress` que `run_multiverse` já expõe, para não haver duas fontes do mesmo
número. Há duas instâncias do registro, `rotas.REGISTRO` para o multiverso e `rotas.TRABALHOS`
para as rotas que correm corretor, para que `GET /multiversos/{id}` não responda por um trabalho
de outra rota.
"""

import math
import sys
import threading
import uuid
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Final, Literal

from refino.domain.cancellation import ExecutionCancelledError

if TYPE_CHECKING:  # pragma: no cover - so para o mypy; importar em runtime acoplaria o modulo
    from refino.api.esquemas import ResultadoDoMultiverso
    from refino.domain.matrix import RefinoMatrix
    from refino.multiverse.grid import GridDefinition
    from refino.multiverse.stability import FeatureStability

__all__ = [
    "ESTADOS",
    "TERMINAIS",
    "EstadoDeExecucao",
    "ProgressoDaExecucao",
    "RegistroDeProgresso",
    "executa_registrando",
    "ordem_de_estabilidade",
    "resume_o_multiverso",
]

EstadoDeExecucao = Literal["pendente", "executando", "concluido", "falhou", "cancelado"]
ESTADOS: Final[tuple[str, ...]] = (
    "pendente",
    "executando",
    "concluido",
    "falhou",
    "cancelado",
)

# Os estados de que não se sai, nomeados como conjunto para que a interface pergunte 'terminou?'
# sem enumerar rótulos: um estado novo seria lido como 'ainda executando' por omissão. `cancelado`
# entra: sem ele a interface consultaria para sempre, e a poda nunca o esqueceria.
TERMINAIS: Final[frozenset[str]] = frozenset({"concluido", "falhou", "cancelado"})


@dataclass(frozen=True)
class ProgressoDaExecucao:
    """Uma leitura do progresso, imutável.

    `celula_que_falhou` e `erro` só existem em `falhou`. `resultado` é o que a execução longa
    devolve, retido para ser reclamado por identificador; vive na mesma leitura que o estado para
    que, sob o `Lock`, um `concluido` nunca apareça sem o resultado. `codigo_http` é o código que
    o disparo teria devolvido se o trabalho fosse síncrono: sem ele, toda falha depois do `202`
    viraria `500`, e a tela trata `409` (do dado) e `500` (da ferramenta) de formas opostas.

    `reclamado` existe para a poda: o resultado servido marca o trabalho, que passa a ser
    esquecido antes dos que ninguém buscou. Marcar não consome, porque um download pode ser
    clicado duas vezes.
    """

    identificador: str
    estado: EstadoDeExecucao
    feito: int
    total: int
    celula_que_falhou: str | None = None
    erro: str | None = None
    resultado: object | None = None
    codigo_http: int | None = None
    reclamado: bool = False
    cancelamento_pedido: bool = False
    """Alguém pediu para parar, o que não é o mesmo que ter parado.

    Entre o pedido e a parada há um intervalo real, e nele a interface diz 'a parar'. O campo
    sobrevive à conclusão natural: um pedido que chega depois do fim fica gravado ao lado de
    `concluido`, e o registro mostra que houve corrida."""
    etapa_em_curso: str | None = None
    """O nome da etapa que a rota anunciou por último, a que está a correr.

    A barra sozinha mostra `0 de 4 etapas` durante a correção inteira; o nome diz qual corre.
    `None` quando a rota ainda não anunciou nada ou não nomeia etapas, como o multiverso, que conta
    células. Não confundir com `celula_que_falhou`, que diz onde a execução parou."""

    @property
    def terminou(self) -> bool:
        """Terminou de qualquer maneira: concluído, falhado ou cancelado.

        A interface pergunta isto, e não `estado == "concluido"`, para não esperar para sempre por
        uma execução que morreu.
        """
        return self.estado in TERMINAIS

    @property
    def fracao(self) -> float:
        """`feito / total`, e zero quando o total é zero.

        `total == 0` só acontece antes de a execução abrir; `nan` faria a barra sumir em vez de
        ficar em zero.
        """
        return 0.0 if self.total <= 0 else self.feito / self.total


class RegistroDeProgresso:
    """Registro em memória das execuções em curso, seguro entre threads.

    Em memória por desenho: se o processo morre, a execução morreu com ele, e um registro
    persistido diria `executando` para sempre. O que terminou vai para o disco por
    `MultiversoRepository`; isto guarda o que está a correr.

    O `Lock` é requisito: a thread do cálculo escreve e a da requisição HTTP lê, e sem ele uma
    leitura poderia ver o `feito` de uma atualização e o `total` de outra.
    """

    def __init__(self) -> None:
        self._trava = threading.Lock()
        self._execucoes: dict[str, ProgressoDaExecucao] = {}
        # Chave do trabalho para identificador: o disparo repetido é reclamado em vez de
        # duplicado. Vazio para o multiverso, que abre por `abre` e não tem chave.
        self._por_chave: dict[str, str] = {}

    def abre(self, total: int, *, identificador: str | None = None) -> str:
        """Registra uma execução em `pendente` e devolve o identificador.

        `identificador` é injetável para o teste de determinismo comparar duas execuções sem o
        `uuid` aleatório no meio.
        """
        chave = identificador or uuid.uuid4().hex
        with self._trava:
            self._execucoes[chave] = ProgressoDaExecucao(
                identificador=chave, estado="pendente", feito=0, total=total
            )
        return chave

    def abre_ou_reclama(self, total: int, *, chave: str) -> tuple[str, bool]:
        """Abre um trabalho novo, ou devolve o que já corre para a mesma entrada.

        Devolve `(identificador, False)` para trabalho novo e `(identificador, True)` quando já
        havia um para a mesma `chave`. Reclamar em vez de recusar com `409` reúne com o trabalho o
        cliente que perdeu o identificador; mesma entrada e mesma semente dão a mesma saída, então o
        resultado é o que uma segunda execução produziria.

        A consulta e a abertura são atômicas, senão dois disparos simultâneos lançariam duas
        threads. Chave que aponta para trabalho já esquecido conta como ausente.
        """
        with self._trava:
            existente = self._por_chave.get(chave)
            anterior = None if existente is None else self._execucoes.get(existente)
            # Trabalho cancelado conta como ausente: a mensagem do `410` promete que redisparar
            # cria um trabalho novo, e sem isto o redisparo devolveria o mesmo identificador
            # terminal. `falhou` não entra: reclamá-lo entrega a causa, e redisparar daria a mesma
            # falha.
            if anterior is not None and anterior.estado != "cancelado":
                return existente, True  # type: ignore[return-value]
            novo = uuid.uuid4().hex
            self._execucoes[novo] = ProgressoDaExecucao(
                identificador=novo, estado="pendente", feito=0, total=total
            )
            self._por_chave[chave] = novo
            return novo, False

    def avanca(
        self,
        identificador: str,
        feito: int,
        total: int,
        *,
        etapa_em_curso: str | None = None,
    ) -> None:
        """Avança o contador; é o que `run_multiverse` chama por célula terminada.

        Monotônico: `max` com o valor anterior, para a barra nunca recuar, mesmo com avisos fora de
        ordem. A etapa só troca quando o contador não recua, para a leitura não juntar o contador de
        um aviso com o nome da etapa de outro. Quem não nomeia etapa deixa o padrão `None`.
        """
        with self._trava:
            atual = self._execucoes[identificador]
            self._execucoes[identificador] = replace(
                atual,
                estado="executando",
                feito=max(atual.feito, feito),
                total=total,
                etapa_em_curso=etapa_em_curso if feito >= atual.feito else atual.etapa_em_curso,
            )

    def conclui(self, identificador: str, *, resultado: object | None = None) -> None:
        """Estado terminal de sucesso, com `feito` forçado a `total`.

        Forçar evita a barra parada perto do fim, que se lê como contador preso. `resultado` é
        opcional porque o multiverso disparado sem par de grupos não tem estabilidade a medir, e o
        `None` deixa a rota de resultado dizê-lo.
        """
        with self._trava:
            atual = self._execucoes[identificador]
            self._execucoes[identificador] = replace(
                atual, estado="concluido", feito=atual.total, resultado=resultado
            )

    def falha(
        self, identificador: str, celula: str, erro: str, *, codigo_http: int | None = None
    ) -> None:
        """Estado terminal de falha, com a célula nomeada.

        A célula diz onde reexecutar em vez de refazer a grade inteira; nos trabalhos que correm
        corretor ela é o nome da etapa. `codigo_http` guarda o código que o disparo teria devolvido,
        para `409` não virar `500` só por a execução ser diferida; `None` é 'não classificado'.
        """
        with self._trava:
            atual = self._execucoes[identificador]
            self._execucoes[identificador] = replace(
                atual,
                estado="falhou",
                celula_que_falhou=celula,
                erro=erro,
                codigo_http=codigo_http,
            )

    def pede_cancelamento(self, identificador: str) -> ProgressoDaExecucao | None:
        """Registra a ordem de parar e devolve a leitura de antes do pedido, ou `None`.

        A leitura, e não um `bool`, separa 'não existe' (`404`) de 'já terminou' (`409`). É a de
        antes para que um cancelamento que chega depois da conclusão receba o estado terminal.
        Marca mesmo quando já terminou, e é idempotente.
        """
        with self._trava:
            atual = self._execucoes.get(identificador)
            if atual is None:
                return None
            self._execucoes[identificador] = replace(atual, cancelamento_pedido=True)
            return atual

    def quer_cancelar(self, identificador: str) -> bool:
        """Diz se pediram para parar: a pergunta que o núcleo faz pelo `CancelCheck`.

        O núcleo recebe só um `Callable[[], bool]` e nunca vê o registro. Identificador desconhecido
        dá `False`, para que uma poda não mate o cálculo com `KeyError`. Corre sob a trava, porque
        quem pergunta é a thread do cálculo e quem escreveu foi a da requisição.
        """
        with self._trava:
            atual = self._execucoes.get(identificador)
            return atual is not None and atual.cancelamento_pedido

    def cancela(self, identificador: str, *, etapa: str | None = None) -> None:
        """Estado terminal de parada a pedido, distinto de `falha`.

        Não preenche `erro`, para que uma contagem de falhas não some os cliques no botão. `etapa`
        reusa `celula_que_falhou` e é opcional: as rotas longas a passam, porque onde parou decide
        se vale redisparar; o multiverso já o diz por `feito`. `feito` não é forçado a `total`,
        porque o número parcial diz quanto correu antes de parar.
        """
        with self._trava:
            atual = self._execucoes[identificador]
            self._execucoes[identificador] = replace(
                atual,
                estado="cancelado",
                cancelamento_pedido=True,
                celula_que_falhou=etapa if etapa is not None else atual.celula_que_falhou,
            )

    def le(self, identificador: str) -> ProgressoDaExecucao | None:
        """A leitura, ou `None` quando o identificador não existe.

        `None` separa 'não conheço' (`404`) de 'conheço e não começou' (`200`).
        """
        with self._trava:
            return self._execucoes.get(identificador)

    def marca_reclamado(self, identificador: str) -> bool:
        """Anota que o resultado deste trabalho foi servido, e devolve se anotou.

        Não consome: o resultado fica até ser podado, porque um download pode ser clicado duas
        vezes; só muda a ordem da poda. `False` quer dizer que o identificador não existe.
        Idempotente.
        """
        with self._trava:
            atual = self._execucoes.get(identificador)
            if atual is None:
                return False
            self._execucoes[identificador] = replace(atual, reclamado=True)
            return True

    def esquece(self, identificador: str) -> None:
        """Remove uma execução terminada do registro, e as chaves que apontam para ela.

        Não é chamado ao concluir: quem lê tem de poder ler o estado terminal ao menos uma vez.
        """
        with self._trava:
            self._execucoes.pop(identificador, None)
            for chave in [c for c, ident in self._por_chave.items() if ident == identificador]:
                self._por_chave.pop(chave, None)

    def esquece_terminados_alem_de(self, limite: int) -> tuple[str, ...]:
        """Poda os trabalhos terminados além do limite e devolve quais esqueceu.

        O registro retém artefatos, CSV clínico identificável, e um serviço de vida longa não pode
        acumulá-los sem teto; o teto e a sua derivação estão em `LIMITE_DE_TRABALHOS_RETIDOS`, em
        `api/rotas.py`. Só o que terminou é candidato, para a poda não deixar órfã uma thread viva.

        A ordem: primeiro os já reclamados, depois os mais antigos. Um trabalho cujo resultado o
        cliente já levou não tem quem o espere. Sem nenhum reclamado, a ordem é a de inserção.
        """
        with self._trava:
            terminados = [i for i, p in self._execucoes.items() if p.estado in TERMINAIS]
            # A chave é negada: para um reclamado, `not True` dá `False`, que ordena antes, e a
            # poda leva o começo da lista. A ordenação estável mantém a inserção em cada grupo.
            terminados.sort(key=lambda i: not self._execucoes[i].reclamado)
            excedente = len(terminados) - max(limite, 0)
            esquecidos = tuple(terminados[:excedente]) if excedente > 0 else ()
            for identificador in esquecidos:
                self._execucoes.pop(identificador, None)
                for chave in [c for c, i in self._por_chave.items() if i == identificador]:
                    self._por_chave.pop(chave, None)
            return esquecidos


def resume_o_multiverso(
    saida: object, grupos: tuple[str, str], identificador: str
) -> "ResultadoDoMultiverso":
    """Resume a execução do multiverso: a estabilidade de cada feature, ordenada.

    Levanta `TypeError` quando a saída não é um `MultiverseRun`, em vez de devolver resumo vazio:
    'a grade não concluiu nada' e 'o executor devolveu outra coisa' não podem sair iguais. `nan`
    vira `None` aqui, onde ainda se sabe que quer dizer 'não medido'.
    """
    from refino.api.esquemas import (
        EstabilidadeDeFeature,
        LimiaresDoVeredito,
        ResultadoDoMultiverso,
    )
    from refino.multiverse.executor import MultiverseRun
    from refino.multiverse.stability import StabilityThresholds, feature_stability

    if not isinstance(saida, MultiverseRun):
        raise TypeError(
            f"a execução do teste de robustez devolveu {type(saida).__name__} e não "
            "`MultiverseRun`: "
            "não há células de onde tirar a estabilidade. Um resumo vazio faria «a grade não "
            "produziu conclusão» e «o executor não devolveu nada» ficarem indistinguíveis"
        )

    # Uma só instância de limiares, passada à medição e à resposta, para a resposta nunca declarar
    # um corte diferente do que aplicou.
    limiares = StabilityThresholds()
    medidas = feature_stability(saida.cells, grupos, thresholds=limiares)
    ordenadas = sorted(medidas, key=ordem_de_estabilidade)
    return ResultadoDoMultiverso(
        identificador=identificador,
        grupos=grupos,
        n_celulas=len(saida.cells),
        n_features=len(medidas),
        estabilidade=tuple(
            EstabilidadeDeFeature(
                feature=m.feature,
                frac_significant=m.frac_significant,
                median_qvalue=_ou_nulo(m.median_qvalue),
                median_rank=_ou_nulo(m.median_rank),
                rank_iqr=_ou_nulo(m.rank_iqr),
                sign_consistency=_ou_nulo(m.sign_consistency),
                n_cells=m.n_cells,
                # Vem do núcleo, medido com estes limiares; reclassificar aqui poderia discordar.
                verdict=m.verdict,
            )
            for m in ordenadas
        ),
        limiares=LimiaresDoVeredito(
            robust_frac=limiares.robust_frac,
            robust_sign=limiares.robust_sign,
            absent_frac=limiares.absent_frac,
        ),
    )


def ordem_de_estabilidade(medida: "FeatureStability") -> tuple[float, str]:
    """A chave de ordenação da tabela: fração decrescente, nome como desempate.

    Função e não `lambda`, para a ordem ser testável com frações que divergem: nos arquivos de
    referência nenhuma feature sobrevive ao FDR, e todas as frações valem zero. O desempate pelo
    nome torna a tabela determinística.
    """
    return (-medida.frac_significant, medida.feature)


def _ou_nulo(valor: float) -> float | None:
    """`None` para `nan`, o valor para o resto: a tradução de 'não medido' para JSON.

    `inf` passa de propósito: infinito num agregado é achado, e `None` o esconderia.
    """
    return None if math.isnan(valor) else valor


def executa_registrando(
    matrix: "RefinoMatrix",
    grid: "GridDefinition",
    registro: RegistroDeProgresso,
    identificador: str,
    *,
    n_jobs: int = 1,
    corrector_jobs: int = 1,
    executor: "Callable[..., object] | None" = None,
    grupos: tuple[str, str] | None = None,
    ao_terminar: "Callable[[ProgressoDaExecucao], None] | None" = None,
) -> None:
    """Executa o multiverso alimentando o registro, e nomeia a célula que falhar.

    Não relança: corre numa thread de fundo, onde a exceção morreria no `sys.excepthook` e o
    progresso ficaria em `executando` para sempre. O `except` largo traduz qualquer falha para
    `falhou`. A célula é a da posição: o `progress` dispara depois de cada célula, então quando a
    de índice `k` levanta, o último aviso foi `k`. `executor` é injetável para os testes.

    `grupos` é o par do contraste em que a estabilidade é medida. Sem par, a execução corre e não
    há resultado: escolher um par por conta própria daria uma tabela certa sobre a pergunta errada.
    O resumo corre dentro do mesmo `try`, e falhar nele é falhar a execução.

    `ao_terminar` recebe a leitura final nos três desfechos, depois de registrada, e é por onde a
    execução vai para o disco sem este módulo conhecer o banco. Entre o registro e a gravação a
    memória já diz `concluido` e o disco ainda não; a ordem inversa exigiria montar a leitura à
    mão. Exceção no gancho não derruba a execução, e a causa vai para a saída de erro.
    """
    from refino.multiverse.executor import run_multiverse

    alvo = run_multiverse if executor is None else executor
    feitas = 0

    def avisa(feito: int, total: int) -> None:
        nonlocal feitas
        feitas = feito
        registro.avanca(identificador, feito, total)

    try:
        saida = alvo(
            matrix,
            grid,
            n_jobs=n_jobs,
            corrector_jobs=corrector_jobs,
            progress=avisa,
            # Um `lambda` de propósito: atravessa a pergunta, e o núcleo nunca vê o registro.
            cancelled=lambda: registro.quer_cancelar(identificador),
        )
        resultado = None if grupos is None else resume_o_multiverso(saida, grupos, identificador)
    # Antes do `except Exception`: `ExecutionCancelledError` herda de `Exception`, e depois dele o
    # cancelamento apareceria como `falhou`.
    except ExecutionCancelledError:
        registro.cancela(identificador)
        _avisa_que_terminou(registro, identificador, ao_terminar)
        return
    except Exception as erro:
        registro.falha(
            identificador, _celula_na_posicao(grid, feitas), f"{type(erro).__name__}: {erro}"
        )
        _avisa_que_terminou(registro, identificador, ao_terminar)
        return
    # Um pedido de cancelamento que chega depois da última célula não acha ponto de checagem, e a
    # execução conclui; `cancelamento_pedido` fica gravado ao lado de `concluido`.
    registro.conclui(identificador, resultado=resultado)
    _avisa_que_terminou(registro, identificador, ao_terminar)


def _avisa_que_terminou(
    registro: RegistroDeProgresso,
    identificador: str,
    ao_terminar: "Callable[[ProgressoDaExecucao], None] | None",
) -> None:
    """Entrega a leitura final ao gancho, sem levantar e sem calar.

    A leitura pode ser `None`: a poda pode ter esquecido a execução entre o `conclui` e esta linha.
    """
    if ao_terminar is None:
        return
    final = registro.le(identificador)
    if final is None:
        return
    try:
        ao_terminar(final)
    except Exception as erro:  # pragma: no cover - exercitado por teste com gancho que levanta
        # A execução terminou e o estado está certo na memória; relançar numa thread de fundo só
        # mataria a thread, e a exceção sumiria no `sys.excepthook`.
        print(
            f"[refino] a execucao {identificador} terminou como {final.estado} e NAO pode ser "
            f"gravada em disco: {type(erro).__name__}: {erro}",
            file=sys.stderr,
        )


def _celula_na_posicao(grid: "GridDefinition", indice: int) -> str:
    """O `fingerprint` da especificação naquela posição, ou um rótulo que diz que não se sabe.

    Nunca inventa: nomear a célula errada levaria a reexecutá-la, ver sucesso e concluir que o
    defeito era intermitente.
    """
    # Este tratador não pode levantar: corre dentro do `except` do trabalho, e se a falha veio da
    # própria enumeração, enumerar de novo levantaria e o trabalho ficaria `pendente` para sempre.
    try:
        for posicao, spec in enumerate(grid.enumerate()):
            if posicao == indice:
                return spec.fingerprint()
    except Exception as erro:
        return f"posicao {indice}: a grade nao pode ser enumerada ({type(erro).__name__})"
    return f"posicao {indice} fora da grade de {grid.size()} celulas"
