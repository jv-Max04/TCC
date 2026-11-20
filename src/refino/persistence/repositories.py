"""Gravação e consulta de execuções.

Os repositórios recebem e devolvem as estruturas do domínio, sem modelo ORM paralelo:
`DefectProfile`, `PipelineSpec` e `QualityReport` já são imutáveis e já têm guarda de
fronteira, e classes ORM seriam uma segunda fonte de verdade para o mesmo contrato.

Coluna tipada guarda o que se consulta. Cada `*_json` guarda o objeto inteiro, para que a
reexecução não dependa de o esquema ter previsto cada campo.
"""

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Final, Literal, get_args

from sqlalchemy import Connection, Engine, delete, func, insert, select, update
from sqlalchemy.engine import NestedTransaction
from sqlalchemy.exc import IntegrityError

from refino.domain.diagnosis import DefectProfile
from refino.domain.spec import validate_override_fields
from refino.persistence.schema import (
    EstadoDoMultiversoGravado,
    decisions,
    diagnoses,
    feature_stability,
    matrices,
    multiverse_runs,
    quality_reports,
    runs,
    samples,
    studies,
    validate_apresentacao,
    validate_estado_do_multiverso,
    validate_origem,
)

__all__ = [
    "OPERACOES_DE_ARTEFATO",
    "SEM_SEMENTE",
    "STATUS_DE_ARTEFATO",
    "ArtefatoGravado",
    "ConclusaoGravada",
    "ConcordanciaIndeterminadaError",
    "EstabilidadeParaGravar",
    "EstadoDeConcordancia",
    "ExecucaoGravada",
    "MultiversoGravado",
    "MultiversoRepository",
    "OperacaoDeArtefato",
    "RunRepository",
    "SobrescritaExercida",
    "StatusDeArtefato",
    "validate_operacao_de_artefato",
    "validate_status_de_artefato",
]

# Os quatro desfechos do par de decisões. `sem_proposta` é o caminho em que a máquina nunca
# propôs nada e a execução foi inteiramente manual. Não é erro, e não é `nao_sobrescreveu`,
# que contaria execução manual como aceitação tácita de uma proposta que nunca existiu.
#
# O único chamador de `grava_execucao` é `POST /execucoes`, que chega a este estado quando
# nenhuma sugestão da máquina foi seguida.
EstadoDeConcordancia = Literal[
    "nao_sobrescreveu",
    "sobrescreveu_concordando",
    "sobrescreveu_discordando",
    "sem_proposta",
]


class ConcordanciaIndeterminadaError(RuntimeError):
    """O par gravado não corresponde a nenhum dos quatro estados.

    Zero linhas, duas linhas sem exatamente uma `manual`, ou mais de duas são defeito de
    escrita. Levanta em vez de devolver um quinto rótulo, que entraria na contagem do conjunto
    de treino.
    """


@dataclass(frozen=True)
class SobrescritaExercida:
    """O controle de sobrescrita foi exercido, e isto é o que a máquina havia proposto.

    A segunda linha é gravada porque o controle foi exercido, e não porque a spec difere. Assim
    `sobrescreveu_concordando`, que é concordância medida, não se confunde com
    `nao_sobrescreveu`, que é ausência de sinal.

    `campos` são os eixos que o controle tocou, e não os que mudaram de valor.
    """

    spec_proposta_json: str
    rationale_proposta: str
    origem_proposta: str
    campos: tuple[str, ...] = field(default=())

    def __post_init__(self) -> None:
        """Valida a origem da proposta e os eixos. `campos` vazio levanta.

        A origem da proposta não pode ser `manual`, que é a da escolha do humano. E controle
        exercido sobre nenhum eixo não é fato possível: gravaria um par de concordância cujo
        eixo ninguém sabe nomear.
        """
        if self.origem_proposta == "manual":
            raise ValueError(
                "`origem_proposta='manual'` na linha da PROPOSTA: a proposta e da maquina, e "
                "`manual` e a origem da escolha do humano. Um par com as duas linhas `manual` "
                "nao tem proposta com que concordar"
            )
        validate_origem(self.origem_proposta)
        if not self.campos:
            raise ValueError(
                "`campos` vazio com sobrescrita exercida. O gatilho da segunda linha e o "
                "controle ter sido exercido, e exercer controle sobre nenhum eixo "
                "nao e fato possivel — o par ficaria sem eixo que alguem soubesse nomear"
            )
        object.__setattr__(self, "campos", validate_override_fields(self.campos))


# ---------------------------------------------------------------------------
# O registro das rotas longas: quanto tempo cada uma gastou
# ---------------------------------------------------------------------------
#
# As rotas que gravam o próprio tempo em `runs`. `medir-corretor` devolve JSON e não um
# arquivo, e está aqui porque passa pelo mesmo envoltório de tempo e é a rota mais cara.
#
# O vocabulário é fechado, e o validador é a única guarda: a operação vai para dentro do
# `spec_json`, que é `Text`, e `runs.status` não tem `CHECK` no esquema.
OperacaoDeArtefato = Literal["corrigir", "processar", "exportar", "medir-corretor"]
OPERACOES_DE_ARTEFATO: Final[tuple[str, ...]] = get_args(OperacaoDeArtefato)

# Três desfechos. `cancelado` é separado de `falhou` porque cancelar é um botão que o
# profissional carregou de propósito, e não defeito da ferramenta.
#
# `ok` e não `concluido` porque é o rótulo de sucesso que `grava_execucao` já escreve em
# `runs.status`. Uma segunda palavra para o mesmo fato obrigaria toda consulta de execuções
# bem-sucedidas a conhecer as duas.
StatusDeArtefato = Literal["ok", "falhou", "cancelado"]
STATUS_DE_ARTEFATO: Final[tuple[str, ...]] = get_args(StatusDeArtefato)

# O valor de `runs.seed` quando os parâmetros da operação não trazem semente. A coluna é
# `Integer` e `nullable=False`, e nem todo pedido traz semente: o de `medir-corretor` pode vir
# sem ela, e aí o oráculo corre com a semente padrão do código. Gravar esse padrão na coluna
# diria que o pedido o escolheu.
#
# `-1` e não `0`, porque `0` é uma semente plausível. A autoridade é o `spec_json`, que nesse
# caso não tem a chave `seed`. Um pedido com `seed=-1` produz a mesma coluna, e só o par de
# coluna e `spec_json` os separa.
SEM_SEMENTE: Final = -1


def validate_operacao_de_artefato(valor: str) -> OperacaoDeArtefato:
    """Guarda de fronteira para a operação gravada.

    Aqui ela é a única guarda: a operação vai para dentro do `spec_json`, que é `Text` livre,
    então nada no banco a confere. Um rótulo fora do vocabulário passaria, e a leitura por
    operação perderia as linhas sem nada acusar.
    """
    if valor not in OPERACOES_DE_ARTEFATO:
        raise ValueError(
            f"`operacao` de artefato inválida: {valor!r}. O domínio é fechado em "
            f"{list(OPERACOES_DE_ARTEFATO)} — são as rotas longas que gravam o próprio tempo, e o "
            "rótulo é o que permite comparar o custo de `corrigir` com o de `exportar` noutra "
            "máquina"
        )
    return valor  # type: ignore[return-value]


def validate_status_de_artefato(valor: str) -> StatusDeArtefato:
    """Guarda de fronteira para `runs.status` numa linha de artefato.

    `runs.status` não tem `CHECK` no esquema, ao contrário de `decisions.origem` e de
    `multiverse_runs.estado`. Este validador é a guarda inteira: sem ele, um desfecho mal
    soletrado entraria em silêncio.
    """
    if valor not in STATUS_DE_ARTEFATO:
        raise ValueError(
            f"`status` de artefato inválido: {valor!r}. O domínio é fechado em "
            f"{list(STATUS_DE_ARTEFATO)}. `ok` é o rótulo de sucesso que `runs.status` já usa; "
            "`cancelado` existe porque cancelar não é falhar"
        )
    return valor  # type: ignore[return-value]


@dataclass(frozen=True)
class ArtefatoGravado:
    """O que uma rota de artefato deixou no banco: a matriz e a linha de `runs`.

    Campos nomeados e não uma tupla: dois `int` posicionais trocam de lugar sem que nada
    levante.
    """

    matrix_id: int
    run_id: int


@dataclass(frozen=True)
class ExecucaoGravada:
    """Os identificadores do que uma execução deixou no banco, consultável por `matrix_id`.

    Campos próprios e não um dicionário solto, em que consultar e não achar seria
    indistinguível de não ter pedido.
    """

    matrix_id: int
    run_id: int
    diagnosis_id: int
    decision_id: int
    quality_report_id: int
    # A linha da proposta, quando há par. `None` significa que não houve sobrescrita, e não
    # que a gravação falhou. `decision_id` é sempre a decisão efetiva, a que o pipeline
    # executou.
    decision_proposta_id: int | None = None


def _ponto_de_retorno(c: Connection) -> NestedTransaction:
    """Abre um SAVEPOINT dentro de uma transação de verdade.

    O driver padrão do SQLite só abre a transação no primeiro INSERT. Um SAVEPOINT emitido
    antes disso vira ele mesmo a transação, e liberá-lo confirma o que foi gravado: se um
    passo seguinte falha, o que entrou até ali não é mais desfeito. Por isso o BEGIN sai
    explícito aqui quando a conexão ainda não está em transação.
    """
    if c.dialect.name == "sqlite":
        bruta = c.connection.dbapi_connection
        if bruta is not None and not bruta.in_transaction:
            c.exec_driver_sql("BEGIN")
    return c.begin_nested()


def _estudo_ou_cria(c: Connection, estudo: str, agora: datetime) -> int:
    """O `id` do estudo, criando-o se faltar. Sobrevive a dois chamadores simultâneos.

    `SELECT` seguido de `INSERT` não é atômico. Dois disparos ao mesmo tempo com o mesmo nome
    de estudo, e o padrão de `POST /multiversos` é `"sem_identificacao"`, passam os dois pelo
    `SELECT` vazio, e o segundo colide em `uq_studies_nome`. Sem tratamento, a execução que
    perde não deixa linha nenhuma. O caso é alcançável: as rotas correm na piscina de threads,
    e duas janelas abertas da mesma pasta partilham o mesmo `refino.db`.

    Quem perde a corrida relê o que o vencedor escreveu. A alternativa,
    `ON CONFLICT DO NOTHING ... RETURNING`, devolve zero linhas quando o conflito acontece e
    exigiria a mesma releitura por baixo.

    A inserção corre num SAVEPOINT para que o `IntegrityError` desfaça só a tentativa. A
    transação do chamador continua de pé para a releitura e para o resto da gravação.
    """
    achado = c.execute(select(studies.c.id).where(studies.c.nome == estudo)).scalar_one_or_none()
    if achado is not None:
        return int(achado)
    try:
        with _ponto_de_retorno(c):
            return int(
                c.execute(
                    insert(studies).values(nome=estudo, criado_em=agora).returning(studies.c.id)
                ).scalar_one()
            )
    except IntegrityError:
        # Outro chamador criou o estudo entre o `SELECT` e o `INSERT`. O nome é único, então o
        # que ele criou é o que este queria.
        segunda = c.execute(
            select(studies.c.id).where(studies.c.nome == estudo)
        ).scalar_one_or_none()
        if segunda is None:  # pragma: no cover - só se a restrição for removida
            raise
        return int(segunda)


def _garante_estudo_e_matriz(
    c: Connection,
    *,
    estudo: str,
    mode: str,
    checksum_entrada: str,
    n_samples: int,
    n_features: int,
    amostras: list[dict[str, Any]],
    agora: datetime,
) -> int:
    """Acha ou cria o estudo e a matriz, e devolve o `matrix_id`.

    Uma regra só para todos os chamadores: a matriz é procurada antes de ser inserida, e a
    forma da que já existe é conferida. Duas cópias divergiriam em silêncio, com um caminho a
    reaproveitar a matriz e o outro a violar `uq_matrices_origem`.

    Recebe a `Connection` e não o motor, para o chamador decidir a transação: `grava_execucao`
    grava tudo numa transação ou nada.
    """
    study_id = _estudo_ou_cria(c, estudo, agora)

    # A matriz é procurada antes de ser inserida. `uq_matrices_origem` declara que a mesma
    # entrada, no mesmo modo e no mesmo estudo, é a mesma matriz. Uma execução nova sobre ela,
    # como ao comparar dois métodos sobre os mesmos dados, é um `run` novo e não uma `matrix`
    # nova.
    existente = c.execute(
        select(matrices.c.id, matrices.c.n_samples, matrices.c.n_features).where(
            matrices.c.study_id == study_id,
            matrices.c.mode == mode,
            matrices.c.checksum_entrada == checksum_entrada,
        )
    ).one_or_none()

    if existente is None:
        # A mesma corrida de `_estudo_ou_cria`: dois disparos simultâneos sobre os mesmos
        # arquivos passam ambos pelo `SELECT` vazio. Quem perde devolve o `id` do vencedor e
        # não insere amostras, que colidiriam em `uq_samples_nome` sobre a matriz do vencedor.
        try:
            with _ponto_de_retorno(c):
                matrix_id = c.execute(
                    insert(matrices)
                    .values(
                        study_id=study_id,
                        mode=mode,
                        n_samples=n_samples,
                        n_features=n_features,
                        checksum_entrada=checksum_entrada,
                    )
                    .returning(matrices.c.id)
                ).scalar_one()
        except IntegrityError:
            venceu = c.execute(
                select(matrices.c.id).where(
                    matrices.c.study_id == study_id,
                    matrices.c.mode == mode,
                    matrices.c.checksum_entrada == checksum_entrada,
                )
            ).scalar_one_or_none()
            if venceu is None:  # pragma: no cover - só se a restrição for removida
                raise
            return int(venceu)

        # As amostras entram só com a matriz nova: `uq_samples_nome` é `(matrix_id, name)`, e
        # reinseri-las sobre uma matriz reaproveitada colidiria.
        if amostras:
            c.execute(
                insert(samples),
                [{"matrix_id": matrix_id, **a} for a in amostras],
            )
    else:
        matrix_id = existente.id
        # A forma é conferida. Os mesmos bytes de entrada dão a mesma forma sob a mesma versão
        # do leitor, mas não sob versões diferentes, e reaproveitar em silêncio uma matriz lida
        # por outro código ligaria as amostras antigas a um `run` novo.
        if (existente.n_samples, existente.n_features) != (n_samples, n_features):
            raise ValueError(
                f"a matriz {matrix_id} já gravada para este estudo, modo e entrada tem "
                f"{existente.n_samples}x{existente.n_features} e a leitura de agora deu "
                f"{n_samples}x{n_features}. Os mesmos bytes deram formas diferentes: ou o "
                "leitor mudou entre as duas execuções, ou o checksum não identifica a "
                "entrada. Use um estudo novo em vez de misturar as duas leituras"
            )
    return matrix_id


class RunRepository:
    """Grava e lê uma execução completa, numa transação por execução.

    Gravação parcial responderia à consulta com metade da execução, por exemplo uma matriz com
    `run` e sem `quality_report`, sem nada acusar. Tudo numa transação, ou nada.
    """

    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    def grava_execucao(
        self,
        *,
        estudo: str,
        mode: str,
        checksum_entrada: str,
        n_samples: int,
        n_features: int,
        amostras: list[dict[str, Any]],
        perfil: DefectProfile,
        spec_json: str,
        seed: int,
        rationale: str,
        origem: str,
        metrics_json: str,
        apresentacao: str,
        sobrescrita: SobrescritaExercida | None = None,
        status: str = "ok",
        iniciado_em: datetime | None = None,
    ) -> ExecucaoGravada:
        """Grava tudo numa transação e devolve os identificadores.

        `origem` e `apresentacao` passam pelos validadores antes do banco, para a recusa sair
        com mensagem de domínio e não como `IntegrityError` cru. `apresentacao` é obrigatório e
        sem padrão: quem grava decisão declara se a justificativa foi exibida, ou que não havia
        superfície onde exibi-la.

        `sobrescrita` não nulo grava duas linhas em `decisions`, na mesma transação: a proposta
        da máquina, com a `origem` da política, e a escolha do humano, com `origem="manual"`.
        Duas chamadas separadas deixariam uma janela em que existe proposta sem escolha.
        """
        origem_valida = validate_origem(origem)
        apresentacao_valida = validate_apresentacao(apresentacao)
        if sobrescrita is not None and origem_valida != "manual":
            raise ValueError(
                f"há sobrescrita exercida e `origem={origem_valida!r}`. A linha B do par é a "
                "escolha do humano, e a origem dela é `manual` por definição. "
                "Gravar de outro modo tornaria o par inclassificável pela consulta"
            )
        agora = datetime.now(UTC)
        with self._engine.begin() as c:
            matrix_id = _garante_estudo_e_matriz(
                c,
                estudo=estudo,
                mode=mode,
                checksum_entrada=checksum_entrada,
                n_samples=n_samples,
                n_features=n_features,
                amostras=amostras,
                agora=agora,
            )

            run_id = c.execute(
                insert(runs)
                .values(
                    matrix_id=matrix_id,
                    spec_json=spec_json,
                    seed=seed,
                    # `iniciado_em` vem de quem disparou: este método corre no fim do pipeline
                    # e só conhece o instante do fim. Ausente, início e fim ficam iguais, para
                    # o chamador que grava um desfecho sem ter cronometrado nada.
                    iniciado_em=iniciado_em or agora,
                    terminado_em=agora,
                    status=status,
                )
                .returning(runs.c.id)
            ).scalar_one()

            diagnosis_id = c.execute(
                insert(diagnoses)
                .values(
                    matrix_id=matrix_id,
                    defect_type=perfil.defect_type,
                    confidence=perfil.confidence,
                    profile_json=json.dumps(_perfil_para_json(perfil), ensure_ascii=False),
                    criado_em=agora,
                )
                .returning(diagnoses.c.id)
            ).scalar_one()

            # A linha da proposta, quando houve sobrescrita, é gravada antes da escolha.
            # `criado_em` empata, e é o `id` crescente que deixa a consulta saber qual veio
            # primeiro.
            proposta_id: int | None = None
            if sobrescrita is not None:
                proposta_id = c.execute(
                    insert(decisions)
                    .values(
                        matrix_id=matrix_id,
                        diagnosis_id=diagnosis_id,
                        spec_json=sobrescrita.spec_proposta_json,
                        rationale=sobrescrita.rationale_proposta,
                        origem=validate_origem(sobrescrita.origem_proposta),
                        criado_em=agora,
                        # A apresentação é um fato da interação e vale para as duas linhas.
                        # Repeti-la permite filtrar o conjunto de treino sem junção.
                        apresentacao=apresentacao_valida,
                        # A máquina não exerceu controle nenhum: o vetor dela é vazio, e é
                        # isso que distingue as duas linhas numa consulta.
                        campos_sobrescritos="[]",
                    )
                    .returning(decisions.c.id)
                ).scalar_one()

            decision_id = c.execute(
                insert(decisions)
                .values(
                    matrix_id=matrix_id,
                    diagnosis_id=diagnosis_id,
                    spec_json=spec_json,
                    rationale=rationale,
                    origem=origem_valida,
                    criado_em=agora,
                    apresentacao=apresentacao_valida,
                    campos_sobrescritos=json.dumps(
                        list(sobrescrita.campos) if sobrescrita is not None else [],
                        ensure_ascii=False,
                    ),
                )
                .returning(decisions.c.id)
            ).scalar_one()

            qr_id = c.execute(
                insert(quality_reports)
                .values(run_id=run_id, metrics_json=metrics_json)
                .returning(quality_reports.c.id)
            ).scalar_one()

        return ExecucaoGravada(
            matrix_id=matrix_id,
            run_id=run_id,
            diagnosis_id=diagnosis_id,
            decision_id=decision_id,
            quality_report_id=qr_id,
            decision_proposta_id=proposta_id,
        )

    def grava_artefato(
        self,
        *,
        operacao: str,
        estudo: str,
        mode: str,
        checksum_entrada: str,
        n_samples: int,
        n_features: int,
        amostras: list[dict[str, Any]],
        parametros: Mapping[str, Any],
        iniciado_em: datetime,
        status: str,
        terminado_em: datetime | None = None,
        etapas: Mapping[str, float] | None = None,
    ) -> ArtefatoGravado:
        """Grava uma linha de `runs` para uma rota de artefato: o tempo e o que o torna legível.

        É método próprio, e não `grava_execucao` com campos ausentes, porque a rota de artefato
        não diagnostica, não decide e não constrói `QualityReport`. Reusar `grava_execucao`
        obrigaria a fabricar um diagnóstico e uma decisão por artefato baixado, e as decisões
        fabricadas entrariam na contagem de concordância.

        A linha carrega o que torna o tempo comparável noutra máquina:

        - início e fim, com `iniciado_em` vindo de quem disparou;
        - a rota e os parâmetros, em `spec_json`;
        - a forma da matriz, por `matrices`, sem repeti-la aqui;
        - o desfecho, em `status`;
        - a duração por etapa, em `etapas_json`, um mapa de nome da etapa para segundos.

        `etapas=None` significa que ninguém cronometrou, e a coluna fica nula. `etapas={}` diria
        que a execução correu e nenhuma etapa custou nada.

        A mensagem de erro de um `falhou` não é gravada: `runs` não tem coluna para ela, e o
        `status` já diz que aquela duração não é a de um trabalho completo.

        A escrita é uma só, no fim, com o desfecho já conhecido. Não há linha aberta que possa
        ficar órfã, e em troca um processo que morre no meio da correção não deixa linha
        nenhuma.
        """
        operacao_valida = validate_operacao_de_artefato(operacao)
        status_valido = validate_status_de_artefato(status)
        agora = terminado_em or datetime.now(UTC)
        # A semente da coluna sai dos parâmetros e não de um argumento próprio, que poderia
        # divergir do `spec_json` gravado na mesma linha.
        bruta = parametros.get("seed")
        semente = int(bruta) if isinstance(bruta, int) else SEM_SEMENTE
        spec_json = json.dumps(
            {"operacao": operacao_valida, "parametros": dict(parametros)},
            ensure_ascii=False,
            sort_keys=True,
        )
        # `None` e `{}` ficam diferentes: `None` diz que ninguém cronometrou, e `{}` diria que
        # a execução correu e nenhuma etapa custou nada.
        etapas_json = (
            None if etapas is None else json.dumps(dict(etapas), ensure_ascii=False, sort_keys=True)
        )
        with self._engine.begin() as c:
            matrix_id = _garante_estudo_e_matriz(
                c,
                estudo=estudo,
                mode=mode,
                checksum_entrada=checksum_entrada,
                n_samples=n_samples,
                n_features=n_features,
                amostras=amostras,
                agora=agora,
            )
            run_id = int(
                c.execute(
                    insert(runs)
                    .values(
                        matrix_id=matrix_id,
                        spec_json=spec_json,
                        seed=semente,
                        iniciado_em=iniciado_em,
                        terminado_em=agora,
                        status=status_valido,
                        etapas_json=etapas_json,
                    )
                    .returning(runs.c.id)
                ).scalar_one()
            )
        return ArtefatoGravado(matrix_id=matrix_id, run_id=run_id)

    def classifica_concordancia(self, matrix_id: int, diagnosis_id: int) -> EstadoDeConcordancia:
        """O estado do par, reconstruído por consulta ao que ficou gravado.

        Lê o banco em vez de devolver o que `grava_execucao` acabou de compor: o que se promete
        é que a concordância entre a máquina e o especialista é recuperável do banco.

        Os quatro estados saem das linhas do par e da `origem` de cada uma. Não há coluna de
        estado, que seria outra fonte de verdade para um fato que as duas linhas já determinam.
        """
        with self._engine.connect() as c:
            linhas = [
                dict(r._mapping)
                for r in c.execute(
                    select(decisions.c.id, decisions.c.origem, decisions.c.spec_json)
                    .where(decisions.c.matrix_id == matrix_id)
                    .where(decisions.c.diagnosis_id == diagnosis_id)
                    .order_by(decisions.c.id)
                )
            ]

        manuais = [linha for linha in linhas if linha["origem"] == "manual"]

        if not linhas:
            raise ConcordanciaIndeterminadaError(
                f"nenhuma decisão gravada para (matrix_id={matrix_id}, "
                f"diagnosis_id={diagnosis_id}). Ausência de linha não é ausência de "
                "sobrescrita: é execução que não gravou, e devolvê-la como estado a faria "
                "entrar na contagem do conjunto"
            )
        if len(linhas) == 1:
            # Uma linha só tem dois desfechos, e separá-los impede que execução puramente
            # manual conte como aceitação tácita de uma proposta que nunca existiu.
            return "sem_proposta" if manuais else "nao_sobrescreveu"
        if len(linhas) > 2:
            raise ConcordanciaIndeterminadaError(
                f"{len(linhas)} decisões no mesmo (matrix_id={matrix_id}, "
                f"diagnosis_id={diagnosis_id}); o par tem duas. Ids: "
                f"{[linha['id'] for linha in linhas]}"
            )
        if len(manuais) != 1:
            raise ConcordanciaIndeterminadaError(
                f"duas decisões e {len(manuais)} com `origem='manual'`; o par exige "
                "exactamente uma. Sem ela não há como saber qual linha é a proposta da "
                f"máquina e qual é a escolha do humano. Ids: {[linha['id'] for linha in linhas]}"
            )

        proposta = next(linha for linha in linhas if linha["origem"] != "manual")
        escolha = manuais[0]
        # Chegar aqui significa que o controle foi exercido. `spec_json` igual nas duas linhas
        # é concordância medida: o humano abriu, avaliou e manteve.
        return (
            "sobrescreveu_concordando"
            if proposta["spec_json"] == escolha["spec_json"]
            else "sobrescreveu_discordando"
        )

    def consulta_por_matriz(self, matrix_id: int) -> dict[str, list[dict[str, Any]]]:
        """As linhas de `runs`, `diagnoses`, `decisions` e `quality_reports` de uma matriz.

        `quality_reports` liga-se à execução por `run_id`, e não à matriz por `matrix_id`,
        então a consulta faz o salto pelos `runs` achados.
        """
        with self._engine.connect() as c:
            achados: dict[str, list[dict[str, Any]]] = {}
            for nome, tabela in (
                ("runs", runs),
                ("diagnoses", diagnoses),
                ("decisions", decisions),
            ):
                achados[nome] = [
                    dict(r._mapping)
                    for r in c.execute(select(tabela).where(tabela.c.matrix_id == matrix_id))
                ]
            ids_run = [r["id"] for r in achados["runs"]]
            achados["quality_reports"] = (
                [
                    dict(r._mapping)
                    for r in c.execute(
                        select(quality_reports).where(quality_reports.c.run_id.in_(ids_run))
                    )
                ]
                if ids_run
                else []
            )
            return achados


def _perfil_para_json(perfil: DefectProfile) -> dict[str, Any]:
    """O perfil inteiro, e não um subconjunto escolhido à mão.

    Guardar tudo custa bytes. Um campo deixado de fora só se recuperaria refazendo o
    diagnóstico de cada matriz.
    """
    return {
        "defect_type": perfil.defect_type,
        "confidence": perfil.confidence,
        "suggested_batches": list(perfil.suggested_batches),
        "changepoints": list(perfil.changepoints),
        "qc_rsd_median": perfil.qc_rsd_median,
        "qc_rsd_p75": perfil.qc_rsd_p75,
        "zero_fraction": perfil.zero_fraction,
        "missing_fraction": perfil.missing_fraction,
        "pc1_injection_correlation": perfil.pc1_injection_correlation,
        "qc_dispersion_ratio": perfil.qc_dispersion_ratio,
        "n_features": perfil.n_features,
        "n_samples": perfil.n_samples,
        "n_qcs": perfil.n_qcs,
    }


# ---------------------------------------------------------------------------
# O multiverso em disco
# ---------------------------------------------------------------------------
#
# Grava-se o estado final e a conclusão, e não célula a célula: numa grade só de `none`, que é
# a mais barata, a gravação por célula custaria perto de um quinto do tempo total. O que se
# perde é a retomada. Uma execução interrompida não continua de onde parou, mas fica gravada
# como `interrompido` em vez de desaparecer.


# O teto de execuções terminadas guardadas em disco. É maior que o teto em memória porque
# protege outra coisa: lá é a RAM de um processo vivo, aqui é o tamanho do arquivo, e o
# histórico existe para durar. No pior caso medido, 200 execuções com conclusão ocupam perto
# de 48 MiB; execução sem par de grupos declarado não grava linha de estabilidade.
#
# A poda nunca toca `executando`: apagaria a linha que o fecho vai atualizar.
LIMITE_DE_EXECUCOES_EM_DISCO: Final = 200


@dataclass(frozen=True)
class EstabilidadeParaGravar:
    """Uma linha da conclusão, já com `nan` traduzido para `None`.

    Tipo próprio em vez de `FeatureStability`: o núcleo devolve `nan` para não medido, e quem
    converte é `api/progresso._ou_nulo`. Aceitar o tipo do núcleo aqui obrigaria a persistência
    a repetir a regra do `nan`.

    As quatro medidas são anuláveis porque podem não existir para a feature. `n_cells`, a
    contagem de células sobre a qual a fração foi medida, existe sempre.
    """

    feature: str
    frac_significant: float
    n_cells: int
    median_qvalue: float | None = None
    median_rank: float | None = None
    rank_iqr: float | None = None
    sign_consistency: float | None = None


@dataclass(frozen=True)
class MultiversoGravado:
    """Uma execução de multiverso como ela fica em disco: a linha do histórico."""

    identificador: str
    estado: EstadoDoMultiversoGravado
    estudo: str
    mode: str
    n_specs: int
    n_feitas: int
    """Quantas células terminaram. Num `interrompido`, `0` significa que não se sabe.

    A gravação acontece na abertura e no fecho, e não por célula: durante a execução a coluna
    fica no `0` com que nasceu. `fecha_multiverso` escreve o valor real nos estados terminais
    que passam por ela; `fecha_orfas` não o toca, porque não tem de onde o tirar. Por isso a
    interface mostra `—` para `interrompido`, em vez de um zero que se leria como medida.
    """

    iniciado_em: datetime
    terminado_em: datetime | None
    n_features: int
    grupos: tuple[str, str] | None = None
    celula_que_falhou: str | None = None
    erro: str | None = None

    @property
    def terminou(self) -> bool:
        """Terminou de qualquer maneira: `executando` é o único estado que não terminou.

        Propriedade para a regra ficar num lugar só. Uma enumeração repetida na interface leria
        um estado novo como ainda a correr.
        """
        return self.estado != "executando"


@dataclass(frozen=True)
class ConclusaoGravada:
    """A execução e a tabela de estabilidade dela.

    Num valor só porque a execução existir e a conclusão dela existir são fatos diferentes: uma
    execução `cancelado` existe e não tem conclusão, e `estabilidade` fica vazia.
    """

    execucao: MultiversoGravado
    estabilidade: tuple[EstabilidadeParaGravar, ...] = ()


class MultiversoRepository:
    """Grava e lê execuções de multiverso: uma linha ao abrir, um `UPDATE` ao fechar.

    Separada de `RunRepository`, que grava tudo numa transação por execução. Aqui a gravação
    acontece em dois momentos separados por minutos, e as duas classes só partilham o motor e
    `_garante_estudo_e_matriz`.

    Uma execução não sobrevive ao processo que a abriu, e um processo que morre deixaria a
    linha em `executando` para sempre. `fecha_orfas`, chamada no arranque, traduz cada uma
    para `interrompido`.
    """

    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    def abre_multiverso(
        self,
        *,
        identificador: str,
        estudo: str,
        mode: str,
        checksum_entrada: str,
        n_samples: int,
        n_features: int,
        amostras: list[dict[str, Any]],
        n_specs: int,
        grupos: tuple[str, str] | None = None,
        iniciado_em: datetime | None = None,
        limite: int = LIMITE_DE_EXECUCOES_EM_DISCO,
    ) -> int:
        """Grava a linha de abertura, com `estado='executando'`. Devolve o `id` da linha.

        `iniciado_em` vem de fora porque quem dispara sabe o instante melhor que quem grava. A
        abertura também poda o histórico além de `limite`.
        """
        agora = iniciado_em or datetime.now(UTC)
        with self._engine.begin() as c:
            matrix_id = _garante_estudo_e_matriz(
                c,
                estudo=estudo,
                mode=mode,
                checksum_entrada=checksum_entrada,
                n_samples=n_samples,
                n_features=n_features,
                amostras=amostras,
                agora=agora,
            )
            novo = int(
                c.execute(
                    insert(multiverse_runs)
                    .values(
                        matrix_id=matrix_id,
                        identificador=identificador,
                        estado="executando",
                        n_feitas=0,
                        n_specs=n_specs,
                        group_a=None if grupos is None else grupos[0],
                        group_b=None if grupos is None else grupos[1],
                        iniciado_em=agora,
                        terminado_em=None,
                    )
                    .returning(multiverse_runs.c.id)
                ).scalar_one()
            )
            # A poda corre depois do `INSERT` e dentro da mesma transação. Assim uma abertura
            # que a guarda de forma da matriz recusa não apaga o excedente.
            _esquece_alem_de(c, limite)
            return novo

    def fecha_multiverso(
        self,
        identificador: str,
        *,
        estado: str,
        n_feitas: int,
        celula_que_falhou: str | None = None,
        erro: str | None = None,
        estabilidade: Sequence[EstabilidadeParaGravar] = (),
        terminado_em: datetime | None = None,
    ) -> bool:
        """Grava o estado terminal e a conclusão, numa transação. `False` se a linha não existe.

        O estado passa por `validate_estado_do_multiverso` antes do banco, e `executando` é
        recusado: fechar para um estado não terminal deixaria a linha indistinguível de uma
        órfã, e o arranque seguinte a reescreveria como `interrompido`.

        Devolve `False` em vez de levantar quando a linha falta. O caminho que chega aqui sem
        linha é o da abertura que não pôde gravar, e isso já foi relatado a quem disparou.
        Levantar transformaria uma execução bem-sucedida em falha.
        """
        validado = validate_estado_do_multiverso(estado)
        if validado == "executando":
            raise ValueError(
                "`fecha_multiverso` recebeu `executando`, que não é um estado terminal. Fechar "
                "para ele deixaria a linha indistinguível de uma órfã, e o arranque seguinte a "
                "reescreveria como `interrompido`"
            )
        agora = terminado_em or datetime.now(UTC)
        with self._engine.begin() as c:
            # O veredito sai do `rowcount` do `UPDATE` e não de um `SELECT` anterior. Entre as
            # duas instruções a linha pode ser podada, e a tabela de estabilidade seria
            # inserida a apontar para uma execução que não existe mais, fora do alcance de
            # qualquer poda.
            resultado = c.execute(
                update(multiverse_runs)
                .where(multiverse_runs.c.identificador == identificador)
                .values(
                    estado=validado,
                    n_feitas=n_feitas,
                    celula_que_falhou=celula_que_falhou,
                    erro=erro,
                    terminado_em=agora,
                )
            )
            if resultado.rowcount != 1:
                return False
            linha = c.execute(
                select(multiverse_runs.c.id).where(multiverse_runs.c.identificador == identificador)
            ).scalar_one()
            if estabilidade:
                c.execute(
                    insert(feature_stability),
                    [
                        {
                            "multiverse_run_id": linha,
                            "feature_name": m.feature,
                            "frac_significant": m.frac_significant,
                            "median_qvalue": m.median_qvalue,
                            "median_rank": m.median_rank,
                            "rank_iqr": m.rank_iqr,
                            "sign_consistency": m.sign_consistency,
                            "n_cells": m.n_cells,
                        }
                        for m in estabilidade
                    ],
                )
            return True

    def esquece_alem_de(self, limite: int = LIMITE_DE_EXECUCOES_EM_DISCO) -> tuple[str, ...]:
        """Apaga as execuções terminadas mais antigas além do teto. Devolve as esquecidas.

        A poda corre na abertura de cada execução, que é quando o histórico cresce.
        `executando` nunca é podada: isso apagaria a linha que `fecha_multiverso` vai
        atualizar, e um trabalho que correu bem sumiria do histórico.

        As linhas de estabilidade são apagadas explicitamente. A chave estrangeira declara
        `ondelete="CASCADE"`, mas o SQLite só o aplica com `PRAGMA foreign_keys=ON`, que este
        projeto não liga.
        """
        if limite < 0:
            return ()
        with self._engine.begin() as c:
            return _esquece_alem_de(c, limite)

    def fecha_orfas(self, *, agora: datetime | None = None) -> tuple[str, ...]:
        """Toda linha `executando` vira `interrompido`. Devolve os identificadores fechados.

        Chamada só no arranque. Durante uma execução ela marcaria como interrompido um trabalho
        que continua a correr.

        Supõe uma instância por banco, e nada o garante: duas janelas abertas da mesma pasta
        partilham o mesmo `refino.db`, e o arranque da segunda marca como `interrompido` a
        execução em curso da primeira. O rótulo errado dura até essa execução terminar, quando
        `fecha_multiverso` grava o desfecho real por cima.

        Devolve os identificadores e não a contagem, para quem chama poder dizer quais
        execuções ficaram por terminar.
        """
        instante = agora or datetime.now(UTC)
        with self._engine.begin() as c:
            abertas = tuple(
                str(i)
                for i in c.execute(
                    select(multiverse_runs.c.identificador).where(
                        multiverse_runs.c.estado == "executando"
                    )
                )
                .scalars()
                .all()
            )
            if abertas:
                c.execute(
                    update(multiverse_runs)
                    .where(multiverse_runs.c.estado == "executando")
                    .values(estado="interrompido", terminado_em=instante)
                )
            return abertas

    def lista_multiversos(self, *, limite: int = 50) -> tuple[MultiversoGravado, ...]:
        """As execuções gravadas, a mais recente primeiro, com a contagem de features.

        `limite` tem padrão porque o histórico é lido para ser mostrado numa tabela. A contagem
        de features vem por subconsulta: a conclusão tem uma linha por feature, e carregá-las
        todas para exibir um número por execução leria megabytes.
        """
        contagem = (
            select(
                feature_stability.c.multiverse_run_id.label("run_id"),
                func.count().label("n"),
            )
            .group_by(feature_stability.c.multiverse_run_id)
            .subquery()
        )
        consulta = (
            select(
                multiverse_runs,
                matrices.c.mode,
                studies.c.nome.label("estudo"),
                func.coalesce(contagem.c.n, 0).label("n_features"),
            )
            .select_from(multiverse_runs)
            .join(matrices, matrices.c.id == multiverse_runs.c.matrix_id)
            .join(studies, studies.c.id == matrices.c.study_id)
            .join(contagem, contagem.c.run_id == multiverse_runs.c.id, isouter=True)
            .order_by(multiverse_runs.c.iniciado_em.desc(), multiverse_runs.c.id.desc())
            .limit(limite)
        )
        with self._engine.connect() as c:
            return tuple(_para_gravado(dict(linha)) for linha in c.execute(consulta).mappings())

    def le_multiverso(self, identificador: str) -> ConclusaoGravada | None:
        """Uma execução com a tabela de estabilidade dela, ou `None` quando não existe.

        `None` significa só que o identificador é desconhecido. Execução que existe e não tem
        conclusão devolve `ConclusaoGravada` com `estabilidade` vazia, e as duas situações
        pedem mensagens diferentes ao usuário.
        """
        consulta = (
            select(
                multiverse_runs,
                matrices.c.mode,
                studies.c.nome.label("estudo"),
            )
            .select_from(multiverse_runs)
            .join(matrices, matrices.c.id == multiverse_runs.c.matrix_id)
            .join(studies, studies.c.id == matrices.c.study_id)
            .where(multiverse_runs.c.identificador == identificador)
        )
        with self._engine.connect() as c:
            linha = c.execute(consulta).mappings().one_or_none()
            if linha is None:
                return None
            medidas = c.execute(
                select(feature_stability)
                .where(feature_stability.c.multiverse_run_id == linha["id"])
                # A mesma ordem de `ordem_de_estabilidade`: fração decrescente, nome como
                # desempate. O banco não promete ordem sem `ORDER BY`, e nos dados reais as
                # frações empatam em zero, então sem o desempate duas leituras da mesma
                # execução poderiam vir em ordens diferentes.
                .order_by(
                    feature_stability.c.frac_significant.desc(),
                    feature_stability.c.feature_name,
                )
            ).mappings()
            estabilidade = tuple(
                EstabilidadeParaGravar(
                    feature=m["feature_name"],
                    frac_significant=m["frac_significant"],
                    n_cells=m["n_cells"],
                    median_qvalue=m["median_qvalue"],
                    median_rank=m["median_rank"],
                    rank_iqr=m["rank_iqr"],
                    sign_consistency=m["sign_consistency"],
                )
                for m in medidas
            )
            # A contagem sai das linhas já carregadas, e tem de dar o mesmo número que a
            # subconsulta de `lista_multiversos` dá para a mesma execução.
            return ConclusaoGravada(
                execucao=_para_gravado(dict(linha), n_features=len(estabilidade)),
                estabilidade=estabilidade,
            )


def _com_fuso_obrigatorio(valor: datetime) -> datetime:
    """`_com_fuso` para a coluna `NOT NULL`, com o retorno sem `None` no tipo."""
    marcado = _com_fuso(valor)
    # A coluna é `NOT NULL`; um `None` aqui viria de um esquema que a migração não produz.
    assert marcado is not None
    return marcado


def _com_fuso(valor: datetime | None) -> datetime | None:
    """Devolve o instante marcado como UTC, que é o fuso em que ele foi gravado.

    As colunas são `DateTime(timezone=True)` e o valor entra com `datetime.now(UTC)`, mas o
    SQLite não guarda fuso: o que volta é ingênuo. Sem a marca, a janela mostraria um horário
    em UTC sem nada a dizer que é UTC, e `datetime.now(UTC) - lido` levantaria `TypeError`.

    Marca em vez de converter para a hora local, que é decisão de apresentação e faria o
    mesmo valor sair diferente conforme a máquina que lê o banco. Um valor que já traz fuso
    passa como está.
    """
    if valor is None or valor.tzinfo is not None:
        return valor
    return valor.replace(tzinfo=UTC)


def _esquece_alem_de(c: Connection, limite: int) -> tuple[str, ...]:
    """A poda, na transação de quem chama. Ver `MultiversoRepository.esquece_alem_de`.

    Recebe a `Connection` para a abertura poder inserir e podar na mesma transação.
    """
    if limite < 0:
        return ()
    terminadas = [
        int(i)
        for i in c.execute(
            select(multiverse_runs.c.id)
            .where(multiverse_runs.c.estado != "executando")
            .order_by(multiverse_runs.c.iniciado_em.desc(), multiverse_runs.c.id.desc())
        )
        .scalars()
        .all()
    ]
    excedente = terminadas[limite:]
    if not excedente:
        return ()
    identificadores = tuple(
        str(i)
        for i in c.execute(
            select(multiverse_runs.c.identificador).where(multiverse_runs.c.id.in_(excedente))
        )
        .scalars()
        .all()
    )
    c.execute(delete(feature_stability).where(feature_stability.c.multiverse_run_id.in_(excedente)))
    c.execute(delete(multiverse_runs).where(multiverse_runs.c.id.in_(excedente)))
    return identificadores


def _para_gravado(linha: Mapping[str, Any], *, n_features: int | None = -1) -> MultiversoGravado:
    """Converte uma linha do `SELECT` em `MultiversoGravado`. O `estado` passa pelo validador.

    Leitura também é fronteira: um rótulo que o `CHECK` não cobrisse, de uma linha escrita por
    um cliente de SQL, cairia num ramo razoável da interface e se comportaria como outro
    estado.

    `n_features=-1` é o sentinela de que a contagem vem da subconsulta, na própria linha.
    `le_multiverso` passa a contagem das linhas que carregou, e as duas formas têm de dar o
    mesmo número para a mesma execução.
    """
    if n_features == -1:
        n_features = int(linha["n_features"])
    grupos = (
        None
        if linha["group_a"] is None or linha["group_b"] is None
        else (str(linha["group_a"]), str(linha["group_b"]))
    )
    return MultiversoGravado(
        identificador=str(linha["identificador"]),
        estado=validate_estado_do_multiverso(str(linha["estado"])),
        estudo=str(linha["estudo"]),
        mode=str(linha["mode"]),
        n_specs=int(linha["n_specs"]),
        n_feitas=int(linha["n_feitas"]),
        iniciado_em=_com_fuso_obrigatorio(linha["iniciado_em"]),
        terminado_em=_com_fuso(linha["terminado_em"]),
        n_features=0 if n_features is None else n_features,
        grupos=grupos,
        celula_que_falhou=linha["celula_que_falhou"],
        erro=linha["erro"],
    )
