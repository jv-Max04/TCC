"""O limiar de RSD calibrado vale para QUATRO ARQUIVOS, e não para quatro rótulos.

Até 03/11/2026 `RSD_MEDIANA_MAX_POR_MODO` era procurado pelo rótulo do modo. A combinação de modos
varia por estudo e o modo se nomeia pela condição analítica, então
`pos_hilic` é o nome de uma coluna e de uma polaridade — e a primeira sugestão do campo de modo. Um
estudo novo com esse rótulo herdava 1,0466, calibrado sobre a matriz DESTE estudo.

Os testes daqui cobrem as três peças, e cada uma com a direção que reprova:

| Peça | O que pode falhar sem nada acusar |
|---|---|
| a cópia dos `sha256` no núcleo | divergir do manifesto no dia em que o dado mudar |
| `mapa_calibrado_para` | devolver o mapa a um conteúdo que não é de referência |
| as duas rotas que diagnosticam | continuar a usar o rótulo, e o teste da função passar sozinho |
"""

from __future__ import annotations

import dataclasses
import hashlib
import re

import pytest
from fastapi.testclient import TestClient

from refino.api.app import build_app
from refino.diagnosis.policy import (
    LIMIARES_INICIAIS,
    MEDICOES_DE_REFERENCIA_SHA256,
    RSD_MEDIANA_MAX_POR_MODO,
    DiagnosisPolicy,
    mapa_calibrado_para,
)
from refino.diagnosis.profile import diagnose
from refino.domain.synthetic import SyntheticSpec
from refino.io.readers import load_matrix
from refino.synthetic.generator import generate
from tests.conftest import MANIFEST, measurements_path, sample_list_path

MODOS = ("pos_hilic", "pos_c18", "neg_hilic", "neg_c18")


def _digests_do_manifesto(texto: str) -> dict[str, str]:
    """`{sha256: modo}` das linhas `input/measurements_<modo>.csv` do manifesto de integridade."""
    saida = {}
    for linha in texto.splitlines():
        achado = re.fullmatch(r"([0-9a-f]{64})\s+input/measurements_(\w+)\.csv", linha.strip())
        if achado:
            saida[achado.group(1)] = achado.group(2)
    return saida


# ---------------------------------------------------------------------------
# A cópia no núcleo contra o manifesto
# ---------------------------------------------------------------------------


def test_a_copia_dos_sha256_no_nucleo_e_a_do_manifesto() -> None:
    """Bijeção nas duas direções: nenhum digest a mais no núcleo, nenhum a menos.

    **Âncora de não-vacuidade antes da igualdade**: um manifesto que o extrator não lesse
    daria `{}` dos dois lados se a constante também estivesse vazia.
    """
    do_manifesto = _digests_do_manifesto(MANIFEST.read_text(encoding="utf-8"))
    assert sorted(do_manifesto.values()) == sorted(MODOS), do_manifesto
    assert do_manifesto == MEDICOES_DE_REFERENCIA_SHA256


def test_negativo_a_conferencia_acusa_um_digest_trocado() -> None:
    """A comparação de cima consegue reprovar: um digest trocado num manifesto plantado diverge."""
    texto = MANIFEST.read_text(encoding="utf-8")
    alvo = next(d for d, m in MEDICOES_DE_REFERENCIA_SHA256.items() if m == "neg_c18")
    plantado = texto.replace(alvo, "0" * 64)
    assert plantado != texto, "a planta não aterrou: o digest do neg_c18 não está no manifesto"
    assert _digests_do_manifesto(plantado) != MEDICOES_DE_REFERENCIA_SHA256


# ---------------------------------------------------------------------------
# A função pura
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("digest", sorted(MEDICOES_DE_REFERENCIA_SHA256))
def test_cada_arquivo_de_referencia_recebe_o_seu_limiar_sob_qualquer_rotulo(digest: str) -> None:
    modo = MEDICOES_DE_REFERENCIA_SHA256[digest]
    assert mapa_calibrado_para(digest, "rotulo_que_o_usuario_escolheu") == {
        "rotulo_que_o_usuario_escolheu": RSD_MEDIANA_MAX_POR_MODO[modo]
    }


@pytest.mark.parametrize("rotulo", MODOS)
def test_conteudo_que_nao_e_de_referencia_cai_na_reserva_mesmo_com_rotulo_conhecido(
    rotulo: str,
) -> None:
    assert mapa_calibrado_para(hashlib.sha256(b"outro estudo").hexdigest(), rotulo) is None


# ---------------------------------------------------------------------------
# O defeito, no núcleo, sobre dado em que as duas regras DIFEREM
# ---------------------------------------------------------------------------


def test_o_rotulo_pos_hilic_ja_nao_declara_limpa_uma_matriz_com_rsd_de_40_por_cento() -> None:
    """O par causal do defeito: a mesma matriz, a regra antiga e a.

    **A matriz é sintética, e a escolha é medida (a planta tem de cair onde as duas
    coisas diferem).** Nos quatro arquivos reais trocar o rótulo só muda a confiança do `neg_c18`:
    nos outros três a correlação do PC1 com a injeção passa de 0,30 e a regra 1 não decide. Uma
    `clean` que imita o `neg_hilic`, semente 1, tem RSD mediano de 0,4031 e correlação de 0,172 —
    medido em 03/11/2026 —, e é o caso que a regra antiga chamava de limpo por causa do rótulo.
    """
    fonte = load_matrix(measurements_path("neg_hilic"), sample_list_path("neg_hilic"), "neg_hilic")
    dado = generate(
        SyntheticSpec(
            imitated_mode="neg_hilic",
            drift_to_noise_ratio=0.0,
            defect_type="clean",
            n_features=400,
            seed=1,
        ),
        fonte,
    )
    matriz = dataclasses.replace(dado.matrix, mode="pos_hilic")
    antiga = diagnose(
        matriz,
        policy=DiagnosisPolicy(
            **LIMIARES_INICIAIS, rsd_mediana_max_por_modo=RSD_MEDIANA_MAX_POR_MODO
        ),
    )
    assert 0.20 < antiga.qc_rsd_median < RSD_MEDIANA_MAX_POR_MODO["pos_hilic"], (
        "a âncora mudou: o RSD desta matriz já não cai entre a reserva e o limiar do pos_hilic, e "
        "o par deixa de separar as duas regras"
    )
    assert antiga.defect_type == "clean", "a regra antiga devia chamar esta matriz de limpa"

    conteudo_novo = hashlib.sha256(b"um estudo que o laboratorio ainda vai mandar").hexdigest()
    nova = diagnose(
        matriz,
        policy=DiagnosisPolicy(
            **LIMIARES_INICIAIS,
            rsd_mediana_max_por_modo=mapa_calibrado_para(conteudo_novo, "pos_hilic"),
        ),
    )
    assert nova.defect_type != "clean", (
        "com o limiar por arquivo, um estudo novo rotulado pos_hilic não é limpo com RSD de 40%"
    )


# ---------------------------------------------------------------------------
# As rotas
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def cliente() -> TestClient:
    return TestClient(build_app())


def _confianca_do_neg_c18(cliente: TestClient, medidas: bytes, rotulo: str) -> tuple[str, float]:
    corpo = cliente.post(
        "/matrizes/diagnosticar",
        files={
            "measurements": ("m.csv", medidas, "text/csv"),
            "sample_list": ("s.csv", sample_list_path("neg_c18").read_bytes(), "text/csv"),
        },
        data={"mode": rotulo},
    ).json()
    return corpo["defect_type"], corpo["confidence"]


def test_a_rota_escolhe_o_limiar_pelo_arquivo_e_nao_pelo_rotulo(cliente: TestClient) -> None:
    """Três envios do `neg_c18`, e é o único modo real em que a regra 1 decide.

    | Envio | Limiar | Confiança medida em 03/11/2026 |
    |---|---|---|
    | o arquivo, rotulado `neg_c18` | o calibrado, 0,1639 | 0,0541 — a |
    | o arquivo, rotulado `outro_rotulo` | o calibrado: o limiar é do arquivo | 0,0541 |
    | os mesmos valores com um `\\n` a mais no fim | a reserva, 0,20 | 0,1951 |

    **O terceiro é o par causal:** os mesmos números, os mesmos QCs, o mesmo RSD mediano — e outros
    bytes. É o que separa «a rota lê o conteúdo» de «a rota lê o rótulo».
    """
    original = measurements_path("neg_c18").read_bytes()

    tipo, de_referencia = _confianca_do_neg_c18(cliente, original, "neg_c18")
    _, rotulo_trocado = _confianca_do_neg_c18(cliente, original, "outro_rotulo")
    tipo_outro, outro_conteudo = _confianca_do_neg_c18(cliente, original + b"\n", "neg_c18")

    assert tipo == tipo_outro == "clean"
    assert rotulo_trocado == de_referencia, "o rótulo voltou a escolher o limiar"
    assert outro_conteudo > de_referencia, (
        "o mesmo rótulo com outro conteúdo recebeu o limiar calibrado: a rota não lê o arquivo"
    )
