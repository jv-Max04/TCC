"""`DiagnosisPolicy` - os seis limiares sem padrão, e a guarda de unidade.

**A guarda de `rsd_mediana_max` é a que já custou defeito real**, e por isso o teste
afirma o TEXTO da mensagem e não só o tipo do erro: o valor da mensagem é dizer o que
o valor errado faria, e mensagem que só diz "inválido" não impede a próxima pessoa de
pôr 20 ali de novo.
"""

import dataclasses

import pytest

from refino.diagnosis.policy import DiagnosisPolicy, InvalidPolicyError

# Os valores iniciais que a especificação declara, mais os dois que a especificação deixa
# para quem tem os dados. Estes DOIS são escolha deste teste e não da especificação -
# ele precisa de algum valor positivo para construir, e isso é diferente de um padrão.
VALIDA = {
    "max_spread": 0.10,
    "rsd_mediana_max": 0.20,
    "pc1_injection_correlation_max": 0.30,
    "step_sustain_min": 2.0,
    "qc_monotonicity_min": 0.487,
    "trajectory_movement_min": 1e-9,
}

CAMPOS = tuple(VALIDA)


def test_a_politica_dos_valores_iniciais_constroi() -> None:
    """Os valores iniciais passam pelas guardas."""
    politica = DiagnosisPolicy(**VALIDA)
    assert politica.rsd_mediana_max == 0.20
    assert politica.qc_monotonicity_min == 0.487


def test_a_politica_e_congelada() -> None:
    """Imutável, como todo objeto de configuração deste projeto."""
    politica = DiagnosisPolicy(**VALIDA)
    with pytest.raises(dataclasses.FrozenInstanceError):
        politica.max_spread = 0.5  # type: ignore[misc]


@pytest.mark.parametrize("ausente", CAMPOS)
def test_nenhum_campo_tem_padrao(ausente: str) -> None:
    """**Nenhum padrão, afirmado campo a campo.**

    Não basta confiar no dataclass: a asserção é que a ausência de padrão é
    exigível pelo tipo, e ela vale para os SEIS. Um default acrescentado por
    conveniência em qualquer um deles fez a prosa "não tem padrão" deixar de ser
    verdade sem que nada reprovasse - foi assim que o contrato da fase ficou sem
    canal para `max_spread`.
    """
    incompleta = {campo: valor for campo, valor in VALIDA.items() if campo != ausente}
    with pytest.raises(TypeError, match=ausente):
        DiagnosisPolicy(**incompleta)


@pytest.mark.parametrize("valor", [20.0, 15.5, 1.0001, 100.0])
def test_limiar_de_rsd_em_percentual_reprova(valor: float) -> None:
    """**A guarda de unidade**, e o `15.5` é literalmente o `neg_c18` em percentual.

    Se esta guarda não existisse, `rsd_mediana_max = 15.5` faria a regra 1 casar
    para todo modo real - porque todo RSD em fração é menor que 15,5 - e tudo sairia
    `clean`. O modo de falha é o insidioso: não levanta, e produz a outra classe.
    """
    with pytest.raises(InvalidPolicyError) as erro:
        DiagnosisPolicy(**{**VALIDA, "rsd_mediana_max": valor})

    mensagem = str(erro.value)
    assert "FRAÇÃO" in mensagem, "a mensagem tem de nomear a confusão, não só o intervalo"
    assert "percentual" in mensagem
    assert "step_change" in mensagem, "a mensagem tem de dizer o que o valor errado FARIA"


def test_um_e_a_borda_inclusiva_do_limiar_de_rsd() -> None:
    """`1.0` é fração válida - RSD de 100% - e `1.0001` não é.

    A borda entra porque limiar de 1,0 é legítimo: features com RSD acima de 100%
    existem nos dados reais, e o `pos_hilic` tem mediana de 1,4381.
    """
    DiagnosisPolicy(**{**VALIDA, "rsd_mediana_max": 1.0})
    with pytest.raises(InvalidPolicyError):
        DiagnosisPolicy(**{**VALIDA, "rsd_mediana_max": 1.0001})


@pytest.mark.parametrize("campo", ["pc1_injection_correlation_max", "qc_monotonicity_min"])
@pytest.mark.parametrize("valor", [1.5, 2.0, 0.0, -0.1])
def test_limiar_normalizado_fora_de_zero_um_reprova(campo: str, valor: float) -> None:
    """Correlação absoluta e líquido/bruto vivem em [0, 1]; limiar fora torna a regra
    inalcançável ou sempre verdadeira."""
    with pytest.raises(InvalidPolicyError, match=campo):
        DiagnosisPolicy(**{**VALIDA, campo: valor})


@pytest.mark.parametrize("campo", ["max_spread", "step_sustain_min", "trajectory_movement_min"])
@pytest.mark.parametrize("valor", [0.0, -1.0])
def test_limiar_nao_normalizado_precisa_ser_positivo(campo: str, valor: float) -> None:
    """Zero torna a comparação vácua, e as três direções são diferentes.

    `max_spread = 0` faz nenhuma métrica ser avaliável e tudo sair `unassessable`;
    `step_sustain_min = 0` faz toda série com ponto de mudança sair `step_change`.
    """
    with pytest.raises(InvalidPolicyError, match=campo):
        DiagnosisPolicy(**{**VALIDA, campo: valor})


def test_piso_de_movimento_aceita_valor_pequeno() -> None:
    """`trajectory_movement_min` é soma de deltas em unidades de score, NÃO normalizada.

    Sem este teste a guarda de (0, 1] poderia ser copiada para cá por simetria, e o
    piso legítimo é da ordem do épsilon de ponto flutuante - o caso que ele existe
    para pegar é trajetória exactamente constante.
    """
    politica = DiagnosisPolicy(**{**VALIDA, "trajectory_movement_min": 1e-12})
    assert politica.trajectory_movement_min == 1e-12
