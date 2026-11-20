"""`scripts/prova_enriquecimento_congelado.py`: o que a carga acusa, visto nos dois lados.

A carga corre dentro da pasta congelada, pelo `pythonw.exe` dela, e só o congelado mede o resto do
que ela afirma. **O que não precisa do congelado virou função pura e é visto aqui:**

- **a diferença de conjuntos do item 5.** O desfecho de reprovação de *"não sobra diretório
  temporário da corrida"* nunca tinha sido visto — a prova saiu APROVADA em `7f89f0f` olhando só
  `refino-vias-*`, quando, pelo que a mensagem de `16ad9ca` registra, cancelar deixava um `Rtmp*`
  do R no `%TEMP%`;
- **o item 6, o modo externo sem rede.** O `.ps1` faz seis afirmações sobre ele, e cada uma tem
  aqui ao menos uma planta que a vira:

  1. a G2 que recusa a carga da prova vira `aprovada`, e uma `monta_carga` que perde um nome vira
     a contagem lida dos bytes;
  2. a G2 desligada vira `recusada`, a G2 que recusa por outro texto vira `pelo_motivo_certo`, e a
     planta que não muda os bytes vira `aterrou`;
  3. a carga trocada depois da G2 e o host trocado viram `leva_a_carga` e `destino_constante`;
  4. um universo declarado diferente do usado vira `ora_confere`, e uma via trocada com os p
     coerentes vira `vias_conferem`;
  5. uma resposta de dado puro não dá o motivo da função;
  6. uma tentativa de rede durante o item aparece na contagem.

  **«Ao menos uma planta por afirmação» não é «uma planta por campo».** Medido em 02/10/2026, entre
  as mutações que o arnês daquele dia plantou numa cópia do script contra os testes deste arquivo
  — o que não diz que estas sejam todas: dois campos que as condições do
  `.ps1` leem continuam sem planta aqui — `rede.guarda_posta` trocado por `True` constante, e a
  metade *«exactamente uma vez»* de `leva_a_carga` trocada por `>= 1`, deixam a bateria verde. Fora
  das condições do `.ps1`, o mesmo vale para `na_lista` e para o bloqueio de `connect_ex`. E o
  contrato entre os dois arquivos — toda chave que o `.ps1` lê existe no registro — é conferido,
  porque o `.ps1` não corre aqui.

Os testes são **em processo**: a planta é argumento ou atributo que a função sob teste lê na
chamada, então não há intermediário que possa não a receber.
Nada sai da máquina: o transporte é a conexão falsa da carga, e as plantas de rede usam endereço
numérico de laço local — `getaddrinfo` sem DNS e `connect` de UDP, que não envia pacote.
"""

from __future__ import annotations

import functools
import importlib.util
import re
import socket
import sys
from collections.abc import Iterable
from pathlib import Path
from types import ModuleType

import pytest

from refino.io import enriquecimento_externo as ext
from refino.io.rds_restrito import Lista, Vetor, escreve_rds
from refino.pathways.enriquecimento import p_da_via_confere, p_hipergeometrico

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "prova_enriquecimento_congelado.py"
PS1 = REPO / "scripts" / "prova_enriquecimento_congelado.ps1"
FIXTURE_COM_FUNCAO = REPO / "data" / "fixtures" / "rds" / "com_funcao.ser"


def _modulo() -> ModuleType:
    spec = importlib.util.spec_from_file_location("prova_enriquecimento_congelado", SCRIPT)
    assert spec is not None and spec.loader is not None
    modulo = importlib.util.module_from_spec(spec)
    sys.modules["prova_enriquecimento_congelado"] = modulo
    spec.loader.exec_module(modulo)
    return modulo


carga = _modulo()


def test_sobra_de_rtmp_e_da_pasta_do_lancador_e_acusada(tmp_path: Path) -> None:
    """Os dois rastros nascidos entre as fotografias são acusados; os anteriores e os alheios não.

    A âncora de não-vacuidade vai na mesma função: a fotografia de ANTES tem de ver ao menos a pasta
    antiga do lançador, senão um padrão que não casasse nada daria lista vazia nas duas pontas e
    *"nada sobrou"* passaria por vacuidade.
    """
    (tmp_path / "Rtmp_de_antes").mkdir()
    (tmp_path / "refino-vias-de-antes").mkdir()
    antes = carga.temporarios(tmp_path)
    assert "refino-vias-de-antes" in antes, antes

    assert carga.sobras(antes, carga.temporarios(tmp_path)) == []

    (tmp_path / "Rtmp4x2kq9").mkdir()
    (tmp_path / "refino-vias-abc123").mkdir()
    (tmp_path / "tmp_alheio").mkdir()
    assert carga.sobras(antes, carga.temporarios(tmp_path)) == [
        "Rtmp4x2kq9",
        "refino-vias-abc123",
    ]


# ---------------------------------------------------------------------------
# Item 6 — o modo externo, sem rede
# ---------------------------------------------------------------------------


def _externo() -> dict[str, object]:
    resultado: dict[str, object] = {}
    carga.prova_externa(resultado)
    return resultado


def test_o_literal_com_funcao_e_o_da_fixture() -> None:
    """`data/` não viaja, e por isso os bytes vão embutidos na carga: eles não podem divergir.

    Sem `skip`: a fixture é versionada, e a ausência dela é defeito da árvore, não condição
    externa — `tests/skip_policy.py` não tem, e não deve ter, regime para isto.
    """
    assert FIXTURE_COM_FUNCAO.read_bytes() == carga.COM_FUNCAO


def test_modo_externo_sem_rede_aprova_recusa_e_le() -> None:
    """O caminho que o congelado tem de reproduzir: cada afirmação do `.ps1` no desfecho de sucesso.

    As âncoras de não-vacuidade vão aqui, e não num teste vizinho: a carga leva os onze nomes lidos
    de volta dos bytes, há duas plantas e as duas aterraram, a ORA leu as três vias inventadas, e a
    guarda de rede estava posta enquanto *"nenhuma tentativa"* era contada.
    """
    r = _externo()
    assert "erro" not in r, r

    c = r["carga"]
    assert isinstance(c, dict)
    assert (c["aprovada"], c["motivo"]) == (True, ""), c
    assert c["nomes"] == c["nomes_esperados"] == len(carga.NOMES_DA_PROVA) == 11, c
    assert c["bytes"] > 0

    plantas = r["g2_plantas"]
    assert isinstance(plantas, list) and len(plantas) == 2, plantas
    for p in plantas:
        assert (p["aterrou"], p["recusada"], p["pelo_motivo_certo"]) == (True, True, True), p

    resposta = r["resposta"]
    assert isinstance(resposta, dict)
    assert resposta["desfecho"] == "ok", resposta
    assert (resposta["leva_a_carga"], resposta["destino_constante"]) == (True, True), resposta
    assert resposta["destino"] == ext.DESTINO
    assert (resposta["ora_confere"], resposta["vias_conferem"]) == (True, True), resposta
    assert len(resposta["vias"]) == len(carga.VIAS_INVENTADAS) == 3
    # A lista da ORA é a dos nomes que estão em alguma via (4), e não a dos 11 enviados: é a regra
    # do servidor, medida em 02/10/2026.
    assert carga.NA_LISTA_SINTETICA == 4 < 11
    assert (resposta["universo"], resposta["na_lista"]) == (carga.UNIVERSO_SINTETICO, 4)

    com_funcao = r["com_funcao"]
    assert isinstance(com_funcao, dict)
    assert com_funcao["desfecho"] == "resposta_malformada", com_funcao
    assert com_funcao["pelo_motivo_certo"] is True, com_funcao

    assert r["rede"] == {"tentativas": [], "guarda_posta": True}


def test_sem_a_g2_as_plantas_passam(monkeypatch: pytest.MonkeyPatch) -> None:
    """Par causal da recusa: com a G2 desligada, o registro diz «não recusada» para cada planta.

    Sem isto, `recusada: true` podia vir de outra coisa que não a G2 — de `monta_carga` recusar,
    por exemplo —, e o `.ps1` leria uma recusa que a G2 nunca fez.
    """
    monkeypatch.setattr(ext, "confere_carga", lambda carga, permitidos: None)
    plantas = _externo()["g2_plantas"]
    assert isinstance(plantas, list) and len(plantas) == 2
    assert [(p["aterrou"], p["recusada"]) for p in plantas] == [(True, False), (True, False)]


def test_a_g2_que_recusa_a_carga_da_prova_vira_aprovada(monkeypatch: pytest.MonkeyPatch) -> None:
    """Par causal da primeira afirmação do `.ps1`: `aprovada` é o desfecho da G2, com o motivo dela.

    Sem isto, `aprovada: true` podia ser constante do registro, e a afirmação leria uma aprovação
    que a G2 nunca deu. A contagem de nomes fica a do desfecho de sucesso: só a G2 mudou.
    """
    motivo = "a G2 plantada recusa toda carga"

    def recusa_tudo(c: bytes, permitidos: Iterable[str]) -> None:
        raise ext.CargaRecusadaError(motivo)

    monkeypatch.setattr(ext, "confere_carga", recusa_tudo)
    c = _externo()["carga"]
    assert isinstance(c, dict)
    assert (c["aprovada"], c["motivo"]) == (False, motivo), c
    assert c["nomes"] == c["nomes_esperados"] == 11, c


def test_a_contagem_de_nomes_e_lida_dos_bytes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Par causal de `nomes`: uma `monta_carga` que perde um nome é acusada pela contagem.

    A planta cai sobre dado em que as duas quantidades diferem — a lista passada tem onze nomes, os
    bytes ficam com dez — e a G2 continua a aprovar, porque os dez são permitidos e a carga é a
    forma canônica deles: só a contagem lida de volta dos bytes separa esta carga da certa. Contar
    a lista daria onze nas duas pontas.

    Só a chamada com a lista da prova perde o nome: a G2 remonta a forma canônica pela mesma
    `ext.monta_carga`, com os nomes lidos da carga, e essa chamada tem de ser a real — senão a
    recusa viria da planta e não da carga.
    """
    real = ext.monta_carga
    da_prova = list(carga.NOMES_DA_PROVA)

    def perde_um_nome_da_prova(nomes: list[str]) -> bytes:
        return real(list(nomes)[1:] if list(nomes) == da_prova else nomes)

    monkeypatch.setattr(ext, "monta_carga", perde_um_nome_da_prova)
    c = _externo()["carga"]
    assert isinstance(c, dict)
    assert c["aprovada"] is True, c
    assert (c["nomes"], c["nomes_esperados"]) == (10, 11), c


def test_a_g2_que_recusa_por_outro_texto_nao_da_o_motivo_certo(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Par causal de *«pelo motivo dela»*: a recusa sozinha não distingue, o motivo distingue.

    A G2 plantada recusa exactamente o que a real recusa, com outro texto: a carga da prova continua
    aprovada e as duas plantas continuam recusadas. A planta cai sobre dado em que as duas coisas
    diferem — o texto plantado não contém nenhum dos dois trechos que o registro procura.
    """
    real = ext.confere_carga
    outro = "recusada por outro motivo qualquer"
    assert carga.AMOSTRA_PLANTADA not in outro and "vocabulário fixo" not in outro

    def recusa_com_outro_texto(c: bytes, permitidos: Iterable[str]) -> None:
        try:
            real(c, permitidos)
        except ext.CargaRecusadaError:
            raise ext.CargaRecusadaError(outro) from None

    monkeypatch.setattr(ext, "confere_carga", recusa_com_outro_texto)
    r = _externo()
    c, plantas = r["carga"], r["g2_plantas"]
    assert isinstance(c, dict) and c["aprovada"] is True, c
    assert isinstance(plantas, list) and len(plantas) == 2, plantas
    assert [p["motivo"] for p in plantas] == [outro, outro], plantas
    assert [(p["recusada"], p["pelo_motivo_certo"]) for p in plantas] == [
        (True, False),
        (True, False),
    ]


def test_a_planta_que_nao_muda_os_bytes_so_e_acusada_por_aterrou(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Par causal de `aterrou`: sem ele, a planta que não existiu lê-se recusada pelo motivo certo.

    Uma `monta_carga` cujo vocabulário não tem `msetora` — o texto que a segunda planta troca —
    deixa a segunda planta igual à carga da prova. A G2 recusa a própria carga por *«vocabulário
    fixo»*, e a planta herda essa recusa com o motivo certo: `recusada` e `pelo_motivo_certo` dizem
    o mesmo que no desfecho de sucesso, e só `aterrou` separa os dois casos.
    """
    real = ext.monta_carga

    def sem_msetora(nomes: list[str]) -> bytes:
        montada = real(nomes)
        assert montada.count(b"msetora") == 1  # a troca abaixo aterra
        return montada.replace(b"msetora", b"msetorY")

    monkeypatch.setattr(ext, "monta_carga", sem_msetora)
    r = _externo()
    c, plantas = r["carga"], r["g2_plantas"]
    assert isinstance(c, dict) and c["aprovada"] is False, c
    assert isinstance(plantas, list) and len(plantas) == 2, plantas
    nome_de_amostra, vocabulario = plantas
    assert nome_de_amostra["aterrou"] is True, nome_de_amostra
    trio = (vocabulario["aterrou"], vocabulario["recusada"], vocabulario["pelo_motivo_certo"])
    assert trio == (False, True, True), vocabulario


def test_universo_declarado_diferente_do_usado_faz_a_ora_nao_conferir(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Par causal da conferência dos p: a mesma resposta, com o universo declarado trocado.

    A planta cai sobre dado em que as duas coisas diferem — conferido antes: o p da primeira via
    sobre 1.000 e sobre 2.000 não está dentro da tolerância da conferência.
    """
    _, total, acertos = carga.VIAS_INVENTADAS[0]
    n = len(carga.NOMES_DA_PROVA)
    p_usado = p_hipergeometrico(acertos, total, n, carga.UNIVERSO_SINTETICO)
    assert not p_da_via_confere(p_usado, acertos, total, n, 2 * carga.UNIVERSO_SINTETICO)

    plantada = functools.partial(
        carga.resposta_sintetica, universo_declarado=2 * carga.UNIVERSO_SINTETICO
    )
    monkeypatch.setattr(carga, "resposta_sintetica", plantada)
    resposta = _externo()["resposta"]
    assert isinstance(resposta, dict)
    # o leitor aceitou (a forma está certa); quem recusa é a conferência dos p
    assert resposta["desfecho"] == "ok", resposta
    assert (resposta["ora_confere"], resposta["vias"]) == (False, []), resposta
    assert "hipergeométrico" in str(resposta["motivo_da_ora"]), resposta


def test_uma_via_trocada_com_os_p_coerentes_faz_as_vias_nao_conferirem(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Par causal de `vias_conferem`: o ramo que calcula o campo, com nomes que divergem.

    O par do universo acima desvia para o `except`, onde o campo fica no padrão. Aqui as contagens
    — e portanto os p — são as mesmas, a ORA confere, e só a comparação dos nomes acusa. A planta
    cai sobre dado em que as duas coisas diferem: os nomes declarados não são os inventados.
    """
    declaradas = ("Via que a prova nao inventou", *(v for v, _, _ in carga.VIAS_INVENTADAS[1:]))
    assert sorted(declaradas) != sorted(v for v, _, _ in carga.VIAS_INVENTADAS)

    plantada = functools.partial(carga.resposta_sintetica, vias_declaradas=declaradas)
    monkeypatch.setattr(carga, "resposta_sintetica", plantada)
    resposta = _externo()["resposta"]
    assert isinstance(resposta, dict)
    assert resposta["desfecho"] == "ok", resposta
    assert (resposta["ora_confere"], resposta["vias_conferem"]) == (True, False), resposta
    assert resposta["vias"] == sorted(declaradas), resposta


def test_resposta_de_dado_puro_no_lugar_da_funcao_nao_da_o_motivo_certo(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Par causal do item d: o desfecho sozinho não distingue, o motivo distingue.

    Uma resposta de dado puro sem a forma da ORA também sai `resposta_malformada` — e é por isso
    que o `.ps1` exige o tipo recusado no motivo, e não só o desfecho.
    """
    puro = escreve_rds(Lista((Vetor("real", (1.0,)),), {"names": Vetor("texto", ("ok",))}))
    monkeypatch.setattr(carga, "COM_FUNCAO", puro)
    com_funcao = _externo()["com_funcao"]
    assert isinstance(com_funcao, dict)
    assert com_funcao["desfecho"] == "resposta_malformada", com_funcao
    assert com_funcao["pelo_motivo_certo"] is False, com_funcao


def test_envio_com_outros_bytes_ou_outro_host_e_acusado(monkeypatch: pytest.MonkeyPatch) -> None:
    """Par causal de `leva_a_carga` e `destino_constante`: o envio que troca a carga ou o host.

    As duas plantas ficam do lado do módulo — a carga alterada depois da G2, e o `HOST` trocado sob
    o `DESTINO` constante —, que é onde o defeito nasceria; a conexão continua a falsa.
    """
    real = ext.envia_ao_servidor

    def troca_a_carga(c: bytes, nomes: list[str], **kw: object) -> ext.RespostaDoServidor:
        alterada = c.replace(b"Glycine", b"Glycinf")
        assert alterada != c  # a planta aterrou
        return real(alterada, nomes, **kw)  # type: ignore[arg-type]

    monkeypatch.setattr(ext, "envia_ao_servidor", troca_a_carga)
    monkeypatch.setattr(ext, "HOST", "outro.exemplo.invalid")
    resposta = _externo()["resposta"]
    assert isinstance(resposta, dict)
    assert (resposta["leva_a_carga"], resposta["destino_constante"]) == (False, False), resposta


def test_sem_rede_bloqueia_anota_e_restaura() -> None:
    """A guarda nos dois passos de saída, e a restauração exacta ao sair do bloco.

    As plantas não saem da máquina com a guarda desligada: `getaddrinfo` de endereço numérico não
    consulta DNS, e `connect` de UDP só fixa o par, sem enviar pacote.
    """
    original = socket.getaddrinfo
    assert "connect" not in vars(socket.socket)
    tentativas: list[str] = []
    with carga.sem_rede(tentativas):
        with pytest.raises(OSError, match="a prova bloqueia a rede: getaddrinfo"):
            socket.getaddrinfo("127.0.0.1", 9)
        with (
            socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s,
            pytest.raises(OSError, match="a prova bloqueia a rede: connect"),
        ):
            s.connect(("127.0.0.1", 9))
    assert tentativas == [
        "getaddrinfo ('127.0.0.1', 9)",
        "connect ('127.0.0.1', 9)",
    ]
    assert socket.getaddrinfo is original
    assert "connect" not in vars(socket.socket)
    assert "connect_ex" not in vars(socket.socket)


def test_a_guarda_cobre_o_item_inteiro(monkeypatch: pytest.MonkeyPatch) -> None:
    """Par causal da ausência: uma tentativa de rede no meio do item aparece na contagem.

    *"Nenhuma tentativa"* no desfecho de sucesso só significa algo se a guarda estiver posta durante
    o envio; aqui a montagem da resposta tenta resolver um nome, e o registro tem de o mostrar.
    """
    original = carga.resposta_sintetica

    def tenta_a_rede(nomes: list[str]) -> bytes:
        with pytest.raises(OSError):
            socket.getaddrinfo("127.0.0.1", 9)
        return original(nomes)

    monkeypatch.setattr(carga, "resposta_sintetica", tenta_a_rede)
    rede = _externo()["rede"]
    assert rede == {"tentativas": ["getaddrinfo ('127.0.0.1', 9)"], "guarda_posta": True}


SPEC = REPO / "refino-desktop.spec"
_MODULOS_DO_ITEM_6 = ("refino.io.enriquecimento_externo", "refino.io.rds_restrito")


def _ocultos_do_spec(texto: str) -> set[str]:
    """Os nomes entre aspas da lista `hiddenimports=[...]`, sem os comentários."""
    bloco = re.search(r"hiddenimports=\[(.*?)\n    \],", texto, flags=re.DOTALL)
    assert bloco is not None, "o spec não tem a lista hiddenimports na forma esperada"
    sem_comentarios = re.sub(r"#[^\n]*", "", bloco.group(1))
    return set(re.findall(r'"([\w.]+)"', sem_comentarios))


def test_o_spec_declara_os_modulos_que_a_prova_importa() -> None:
    """A prova importa os dois por nome, de fora do grafo que a análise do PyInstaller percorre.

    A âncora é um nome que já estava na lista; o par causal tira um dos dois e vê a ausência.
    """
    texto = SPEC.read_text(encoding="utf-8")
    ocultos = _ocultos_do_spec(texto)
    assert "httpx2" in ocultos, ocultos
    assert set(_MODULOS_DO_ITEM_6) <= ocultos
    plantado = texto.replace('"refino.io.rds_restrito",', "")
    assert plantado != texto
    assert "refino.io.rds_restrito" not in _ocultos_do_spec(plantado)


# O `.ps1` lê o registro por `$x.<grupo>.<chave>` e cada planta por `$planta.<chave>`. `erro` é a
# única chave que só existe no desfecho de exceção, posta por `main`.
_LEITURA_DO_GRUPO = re.compile(r"\$x\.(\w+)(?:\.(\w+))?")
_LEITURA_DA_PLANTA = re.compile(r"\$planta\.(\w+)")


def _chaves_que_faltam(texto: str, registro: dict[str, object]) -> list[str]:
    faltam: list[str] = []
    for grupo, chave in _LEITURA_DO_GRUPO.findall(texto):
        if grupo == "erro":
            continue
        valor = registro.get(grupo)
        if valor is None or (chave and not (isinstance(valor, dict) and chave in valor)):
            faltam.append(f"{grupo}.{chave}" if chave else grupo)
    plantas = registro.get("g2_plantas")
    assert isinstance(plantas, list) and plantas
    for chave in _LEITURA_DA_PLANTA.findall(texto):
        faltam.extend(f"g2_plantas[].{chave}" for p in plantas if chave not in p)
    return sorted(set(faltam))


def test_toda_chave_que_o_ps1_le_existe_no_registro() -> None:
    """O contrato entre os dois arquivos: uma chave renomeada de um lado reprovaria no congelado.

    Ali o `.ps1` leria `$null`, e cada afirmação foi escrita para reprovar com `$null` — mas o
    motivo impresso seria a ausência e não o defeito. Aqui a divergência aparece com o nome dela.
    """
    texto = PS1.read_text(encoding="utf-8")
    lidas = _LEITURA_DO_GRUPO.findall(texto)
    # âncora: o `.ps1` lê os cinco grupos, e as plantas por chave
    assert {g for g, _ in lidas} >= {"carga", "g2_plantas", "resposta", "com_funcao", "rede"}
    assert _LEITURA_DA_PLANTA.findall(texto)

    registro = _externo()
    assert _chaves_que_faltam(texto, registro) == []
    # par causal: a mesma leitura com uma chave trocada é acusada pelo nome
    plantado = texto.replace("$x.rede.guarda_posta", "$x.rede.guarda_pusta")
    assert plantado != texto
    assert _chaves_que_faltam(plantado, registro) == ["rede.guarda_pusta"]
