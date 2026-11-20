"""Gera e rotula o corpus de diagnóstico — 20 matrizes por modo.

**Escreve INCREMENTAL, uma linha JSON por matriz.** São ~14 h de rotulagem, e um script que só
grava no fim perde tudo numa interrupção. Cada linha é autocontida e o arquivo é retomável: quem
reinicia lê o que existe e continua das sementes que faltam.

`drift_to_noise_ratio` log-uniforme em `[0,1; 3,0]`, a faixa que contém o ponto de virada nos
quatro modos. **A escolha é declarada** — corpus com deriva toda alta daria rótulo constante e
mediria a escolha em vez do dado.

**Este script gerou o corpus GRAVADO, de oito QCs** (`data/fixtures/corpus_fase9.jsonl`), com a
política de diagnóstico do `main()`. **O corpus do modelo distribuído é outro**,
`data/fixtures/corpus_fase9_7qcs.jsonl`: o mesmo plano, com sete QCs e o diagnóstico do
aplicativo. **Não mude a chamada a `DiagnosisPolicy` daqui**: `tests/test_perfil_da_politica.py`
planta a política antiga no lugar da constante.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

MODOS = ("pos_hilic", "pos_c18", "neg_hilic", "neg_c18")


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    ap = argparse.ArgumentParser()
    ap.add_argument("--por-modo", type=int, default=20)
    ap.add_argument("--features", type=int, default=2000)
    ap.add_argument("--saida", type=Path, default=Path("data/fixtures/corpus_fase9.jsonl"))
    args = ap.parse_args()

    from refino.decision.labeling import LabelingOracle
    from refino.diagnosis.policy import DiagnosisPolicy
    from refino.diagnosis.profile import diagnose
    from refino.domain.synthetic import SyntheticSpec
    from refino.io.readers import load_matrix
    from refino.synthetic.generator import generate
    from tests.conftest import measurements_path, sample_list_path

    politica = DiagnosisPolicy(
        max_spread=0.10,
        rsd_mediana_max=0.20,
        pc1_injection_correlation_max=0.30,
        step_sustain_min=2.0,
        qc_monotonicity_min=0.487,
        trajectory_movement_min=1e-9,
    )
    oraculo = LabelingOracle()

    # Semente DERIVADA de `(modo, k)`, e nao de um gerador global consumido na ordem do laco.
    #
    # Com gerador global, mudar `--por-modo` de 20 para 10 mudaria TODAS as sementes, o corpus
    # misturaria dois planos, e estender de 10 para 20 depois nao reaproveitaria nada. Com
    # fluxo por unidade a ordem de iteracao deixa de importar e estender preserva o que existe.
    plano = []
    for modo in MODOS:
        for k in range(args.por_modo):
            fluxo = np.random.default_rng([20260803, sum(ord(c) for c in modo), len(modo), k])
            deriva = float(10 ** fluxo.uniform(np.log10(0.1), np.log10(3.0)))
            plano.append((modo, k, int(fluxo.integers(1, 10**6)), round(deriva, 5)))

    args.saida.parent.mkdir(parents=True, exist_ok=True)
    feitos = set()
    if args.saida.exists():
        for linha in args.saida.read_text(encoding="utf-8").splitlines():
            if linha.strip():
                r = json.loads(linha)
                feitos.add((r["modo"], r["indice"]))
        print(f"retomando: {len(feitos)} de {len(plano)} já feitos", flush=True)

    fontes: dict[str, object] = {}
    for modo, k, semente, deriva in plano:
        if (modo, k) in feitos:
            continue
        if modo not in fontes:
            fontes[modo] = load_matrix(measurements_path(modo), sample_list_path(modo), mode=modo)
        t0 = time.perf_counter()
        ds = generate(
            SyntheticSpec(
                imitated_mode=modo,  # type: ignore[arg-type]
                drift_to_noise_ratio=deriva,
                n_features=args.features,
                seed=semente,
            ),
            fontes[modo],  # type: ignore[arg-type]
        )
        perfil = diagnose(ds.matrix, policy=politica)
        rot = oraculo.label(ds.matrix)
        registro = {
            "modo": modo,
            "indice": k,
            "seed": semente,
            "deriva_pedida": deriva,
            "deriva_efetiva": round(ds.effective_drift_to_noise, 5),
            "rotulo": rot.correction,
            "margem": round(float(rot.margin), 6),
            "assessable": rot.assessable,
            "scores": {a: round(v, 6) for (a, _), v in rot.scores.items()},
            # O PERFIL INTEIRO, e nao um subconjunto escolhido a mao.
            #
            # Guardar so `defect_type`, `diag_confidence` e uma chave constante nao basta:
            # as nove observaveis que `features.py` exige tambem entram, senao treinar
            # exigiria REGERAR as 40 matrizes.
            "profile": {
                "defect_type": perfil.defect_type,
                "confidence": round(perfil.confidence, 6),
                "suggested_batches": list(perfil.suggested_batches),
                "changepoints": list(perfil.changepoints),
                **{
                    nome: round(float(getattr(perfil, nome)), 6)
                    for nome in (
                        "qc_rsd_median",
                        "qc_rsd_p75",
                        "zero_fraction",
                        "missing_fraction",
                        "pc1_injection_correlation",
                        "qc_dispersion_ratio",
                    )
                },
                "n_features": perfil.n_features,
                "n_samples": perfil.n_samples,
                "n_qcs": perfil.n_qcs,
            },
            "min": round((time.perf_counter() - t0) / 60, 2),
        }
        with args.saida.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(registro, ensure_ascii=False) + "\n")
        print(
            f"{modo:<11}{k:>3} deriva={deriva:>6.3f} {perfil.defect_type:<14}"
            f"{rot.correction:<7}margem={rot.margin:>8.4f} {registro['min']:>6.2f} min",
            flush=True,
        )
    print("FIM", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
