"""As nove tabelas do banco.

SQLAlchemy Core e não ORM: os repositórios recebem e devolvem as estruturas do domínio, que já
são imutáveis e já têm guarda de fronteira. Modelos ORM seriam uma segunda fonte de verdade
para `DefectProfile` e `PipelineSpec`.

Os vocabulários fechados de `origem`, `apresentacao` e `estado` têm duas guardas: o `CHECK` no
banco e um validador em Python. O validador existe porque a leitura também é fronteira: valor
lido do banco nunca passou pelo mypy, e um rótulo inválido cairia num ramo razoável e se
comportaria como outra classe sem levantar.
"""

from typing import Final, Literal, get_args

from sqlalchemy import (
    CheckConstraint,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    UniqueConstraint,
)

__all__ = [
    "APRESENTACOES",
    "ESTADOS_DO_MULTIVERSO",
    "METADATA",
    "ORIGENS",
    "TABELAS",
    "DecisionOrigin",
    "DecisionPresentation",
    "EstadoDoMultiversoGravado",
    "validate_apresentacao",
    "validate_estado_do_multiverso",
    "validate_origem",
]

DecisionOrigin = Literal["heuristica", "modelo", "manual"]
ORIGENS: Final[tuple[str, ...]] = get_args(DecisionOrigin)

# O campo registra que a justificativa foi exibida, e não que foi lida: a interface controla o
# que renderiza, não o que o pesquisador leu. Sem ele, aceitar a proposta cobriria dois casos
# opostos, o de quem viu a justificativa e concordou e o de quem nunca a abriu.
#
# Três valores e não um booleano: uma decisão gravada sem tela nenhuma não foi exibida nem
# deixou de ser, e `sem_apresentacao` evita contar ausência de tela como omissão da tela.
DecisionPresentation = Literal["exibida", "nao_exibida", "sem_apresentacao"]
APRESENTACOES: Final[tuple[str, ...]] = get_args(DecisionPresentation)


def validate_apresentacao(valor: str) -> DecisionPresentation:
    """Guarda de fronteira para `decisions.apresentacao`.

    O valor chega do corpo de uma requisição ou de uma coluna, e nada nele passou pelo mypy.
    """
    if valor not in APRESENTACOES:
        raise ValueError(
            f"`apresentacao` invalida: {valor!r}. O dominio e fechado em {list(APRESENTACOES)}. "
            "O campo registra que a justificativa foi EXIBIDA, nunca "
            "que foi lida — a interface controla o que renderiza, nao o que se leu"
        )
    return valor  # type: ignore[return-value]


def validate_origem(valor: str) -> DecisionOrigin:
    """Guarda de fronteira para `decisions.origem`.

    `Literal` é análise estática: o mypy prova os pontos de chamada internos, e valor vindo do
    banco nunca passou por ele. Levanta em vez de cair num `else` razoável.
    """
    if valor not in ORIGENS:
        raise ValueError(
            f"`origem` inválida: {valor!r}. O domínio é fechado em {list(ORIGENS)}. "
            "Valor vindo do banco é fronteira de confiança e não passou pelo mypy"
        )
    return valor  # type: ignore[return-value]


# O estado com que uma execução de multiverso fica em disco. Difere do vocabulário em memória
# de `api/progresso` em dois rótulos: `pendente` não existe aqui, porque a linha só é gravada
# quando a execução começa, e `interrompido` só existe aqui.
#
# `interrompido` é o que uma linha `executando` vira no arranque seguinte. A execução não
# sobrevive ao processo que a abriu, e sem o rótulo a linha diria `executando` para sempre.
EstadoDoMultiversoGravado = Literal[
    "executando", "concluido", "falhou", "cancelado", "interrompido"
]
ESTADOS_DO_MULTIVERSO: Final[tuple[str, ...]] = get_args(EstadoDoMultiversoGravado)


def validate_estado_do_multiverso(valor: str) -> EstadoDoMultiversoGravado:
    """Guarda de fronteira para `multiverse_runs.estado`.

    O `CHECK` do banco é a metade que vive no servidor. Esta dá mensagem de domínio em vez de
    `IntegrityError` cru, e vale também na leitura: o rótulo volta do disco sem ter passado
    pelo mypy.
    """
    if valor not in ESTADOS_DO_MULTIVERSO:
        raise ValueError(
            f"`estado` de multiverso inválido: {valor!r}. O domínio é fechado em "
            f"{list(ESTADOS_DO_MULTIVERSO)}. Note que `pendente` NÃO está aqui — a linha só é "
            "gravada quando a execução começa — e que `interrompido` não existe em memória: ele é "
            "o que uma execução aberta vira quando o processo que a abriu morreu"
        )
    return valor  # type: ignore[return-value]


# Convenção de nome para restrições e índices. Sem ela o alembic gera nomes automáticos, e o
# `downgrade` não acha o que derrubar.
METADATA = MetaData(
    naming_convention={
        "ix": "ix_%(column_0_label)s",
        "uq": "uq_%(table_name)s_%(column_0_name)s",
        "ck": "ck_%(table_name)s_%(constraint_name)s",
        "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
        "pk": "pk_%(table_name)s",
    }
)

studies = Table(
    "studies",
    METADATA,
    Column("id", Integer, primary_key=True),
    Column("nome", String(200), nullable=False, unique=True),
    Column("descricao", Text, nullable=True),
    Column("criado_em", DateTime(timezone=True), nullable=False),
)

matrices = Table(
    "matrices",
    METADATA,
    Column("id", Integer, primary_key=True),
    Column("study_id", Integer, ForeignKey("studies.id", ondelete="CASCADE"), nullable=False),
    Column("mode", String(32), nullable=False),
    Column("n_samples", Integer, nullable=False),
    Column("n_features", Integer, nullable=False),
    # O checksum liga a linha ao arquivo que a produziu. Sem ele, duas ingestões do mesmo modo
    # seriam indistinguíveis.
    Column("checksum_entrada", String(64), nullable=False),
    UniqueConstraint("study_id", "mode", "checksum_entrada", name="uq_matrices_origem"),
)

samples = Table(
    "samples",
    METADATA,
    Column("id", Integer, primary_key=True),
    Column("matrix_id", Integer, ForeignKey("matrices.id", ondelete="CASCADE"), nullable=False),
    Column("name", String(200), nullable=False),
    # A ordem de injeção é o único lugar onde o tempo existe: as colunas de `measurements_*.csv`
    # estão agrupadas por classe, não em ordem temporal. Sem ela não há correção de deriva.
    Column("injection_order", Integer, nullable=False),
    Column("sample_type", String(16), nullable=False),
    Column("grupo", String(100), nullable=True),
    Column("batch", String(32), nullable=True),
    UniqueConstraint("matrix_id", "name", name="uq_samples_nome"),
)

runs = Table(
    "runs",
    METADATA,
    Column("id", Integer, primary_key=True),
    Column("matrix_id", Integer, ForeignKey("matrices.id", ondelete="CASCADE"), nullable=False),
    Column("spec_json", Text, nullable=False),
    Column("seed", Integer, nullable=False),
    Column("iniciado_em", DateTime(timezone=True), nullable=False),
    Column("terminado_em", DateTime(timezone=True), nullable=True),
    Column("status", String(32), nullable=False),
    # A duração por etapa. Coluna própria e não um campo do `spec_json`, que guarda o que foi
    # pedido, `operacao` e `parametros`, e não onde o tempo foi gasto.
    #
    # Anulável porque a ausência é um estado distinto de zero: `POST /execucoes` não cronometra
    # etapas, e uma linha gravada por versão anterior também não tem. `{}` diria que a execução
    # correu e nenhuma etapa custou nada.
    Column("etapas_json", Text, nullable=True),
)

diagnoses = Table(
    "diagnoses",
    METADATA,
    Column("id", Integer, primary_key=True),
    Column("matrix_id", Integer, ForeignKey("matrices.id", ondelete="CASCADE"), nullable=False),
    Column("defect_type", String(32), nullable=False),
    Column("confidence", Float, nullable=False),
    Column("profile_json", Text, nullable=False),
    Column("criado_em", DateTime(timezone=True), nullable=False),
)

decisions = Table(
    "decisions",
    METADATA,
    Column("id", Integer, primary_key=True),
    Column("matrix_id", Integer, ForeignKey("matrices.id", ondelete="CASCADE"), nullable=False),
    Column("diagnosis_id", Integer, ForeignKey("diagnoses.id", ondelete="SET NULL"), nullable=True),
    Column("spec_json", Text, nullable=False),
    Column("rationale", Text, nullable=False),
    Column("origem", String(16), nullable=False),
    Column("criado_em", DateTime(timezone=True), nullable=False),
    # As duas colunas do par de concordância. Não há unicidade em `(matrix_id, diagnosis_id)`:
    # a proposta da máquina e a escolha do humano são duas linhas com o mesmo par de
    # identificadores.
    Column("apresentacao", String(32), nullable=False),
    # Os eixos que o controle de sobrescrita tocou, como vetor JSON ordenado. Não se deriva da
    # diferença entre as duas linhas: quando o humano abre o controle e mantém o valor, as duas
    # são iguais, e sem a coluna não se saberia qual eixo ele avaliou.
    Column("campos_sobrescritos", Text, nullable=False),
    CheckConstraint(
        "origem IN ('heuristica', 'modelo', 'manual')",
        name="origem_no_vocabulario",
    ),
    CheckConstraint(
        "apresentacao IN ('exibida', 'nao_exibida', 'sem_apresentacao')",
        name="apresentacao_no_vocabulario",
    ),
)

quality_reports = Table(
    "quality_reports",
    METADATA,
    Column("id", Integer, primary_key=True),
    Column("run_id", Integer, ForeignKey("runs.id", ondelete="CASCADE"), nullable=False),
    Column("metrics_json", Text, nullable=False),
)

# Uma linha por execução de multiverso, aberta como `executando` e fechada com o estado
# terminal.
multiverse_runs = Table(
    "multiverse_runs",
    METADATA,
    Column("id", Integer, primary_key=True),
    Column("matrix_id", Integer, ForeignKey("matrices.id", ondelete="CASCADE"), nullable=False),
    # O identificador que a janela mostra. O registro em memória dá um UUID por execução, e é
    # esta coluna que liga a linha gravada à barra de progresso.
    #
    # Único porque o UUID é a chave natural: duas linhas para o mesmo identificador
    # significariam que a abertura correu duas vezes.
    Column("identificador", String(64), nullable=False),
    Column("estado", String(16), nullable=False),
    # `n_specs` é o tamanho da grade, quanto se pediu; `n_feitas` é quantas células terminaram.
    # Numa execução `concluido` os dois batem; em `cancelado` ou `falhou` a diferença diz até
    # onde ela chegou.
    Column("n_feitas", Integer, nullable=False),
    Column("celula_que_falhou", String(500), nullable=True),
    Column("erro", Text, nullable=True),
    # O par de grupos do contraste, quando houve. `NULL` significa que nenhum foi declarado:
    # sem par a execução corre e não produz conclusão.
    Column("group_a", String(100), nullable=True),
    Column("group_b", String(100), nullable=True),
    Column("n_specs", Integer, nullable=False),
    Column("iniciado_em", DateTime(timezone=True), nullable=False),
    Column("terminado_em", DateTime(timezone=True), nullable=True),
    UniqueConstraint("identificador", name="uq_multiverse_runs_identificador"),
    CheckConstraint(
        "estado IN ('executando', 'concluido', 'falhou', 'cancelado', 'interrompido')",
        name="estado_no_vocabulario",
    ),
)

feature_stability = Table(
    "feature_stability",
    METADATA,
    Column("id", Integer, primary_key=True),
    Column(
        "multiverse_run_id",
        Integer,
        ForeignKey("multiverse_runs.id", ondelete="CASCADE"),
        nullable=False,
    ),
    Column("feature_name", String(200), nullable=False),
    Column("frac_significant", Float, nullable=False),
    # `median_qvalue` guarda um q-valor, já corrigido por FDR, e não um p-valor. Os dois nomes
    # são os de `FeatureStability`.
    Column("median_qvalue", Float, nullable=True),
    Column("median_rank", Float, nullable=True),
    # As outras duas medidas que `FeatureStability` produz, para a conclusão gravada não ser
    # mais pobre que a servida pela rota.
    Column("rank_iqr", Float, nullable=True),
    Column("sign_consistency", Float, nullable=True),
    # Sobre quantas células a fração foi medida. A mesma `frac_significant` sobre duas células
    # e sobre milhares é evidência incomparável.
    #
    # As quatro medidas acima são anuláveis e esta não. Uma medida pode não existir para a
    # feature, o que nos dados reais é frequente, e não medido é diferente de medido e nulo. A
    # contagem de células existe sempre.
    Column("n_cells", Integer, nullable=False),
    UniqueConstraint("multiverse_run_id", "feature_name", name="uq_stability_feature"),
)

TABELAS: Final[tuple[str, ...]] = (
    "studies",
    "matrices",
    "samples",
    "runs",
    "diagnoses",
    "decisions",
    "quality_reports",
    "multiverse_runs",
    "feature_stability",
)
