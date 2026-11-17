"""Narrative of deep-dive report D3 (representation study); METHODS lists the variants present in runs/d3."""
from pathlib import Path

VARIANT_LABELS = {"strided_only": "strided history only (10 x 10 ms)", "dense32_strided10": "dense 32 ms + strided 100 ms (D2 default)", "dense64_only": "dense 64 ms only",
                  "dense32_strided5ms": "dense 32 ms + strided 10 x 5 ms", "dense8_only": "dense 8 ms only", "no_history": "no history",
                  "dense32_strided10_fourier": "default + Fourier features", "strided_only_fourier": "strided only + Fourier features"}

METHODS = [("pd_shared", "PD, shared gains (v2)")]
for run in sorted(Path("runs/d3").glob("*-s*")) if Path("runs/d3").exists() else []:
    name = run.name.split("-s")[0]
    kind, _, variant = name.partition("_" if name.startswith("pd_residual") else "_")
    METHODS.append((name, VARIANT_LABELS.get(name.replace("pd_residual_", "").replace("neural_torque_", ""), name)))

NARRATIVE = {
    "question": """<p>Which representation of the proprioceptive history serves the learned controller best at 1 kHz: a strided 100 ms window, a dense window of the last 8 to 64 ms, both, or none; and do random Fourier features on the inputs help the network represent the high-frequency structure of contact onset? The advisor raised the encoding question; this study answers it under one fixed training protocol and one parameterization, with one seed per cell and the D2 default as the anchor.</p>""",
    "setup": """<p>Every cell trains with the D2 protocol (ADR 0005), identical budget, seed and loss configuration; only the history windows and the input encoding change (see the architecture section of report D2). Evaluation on all protocol v2 groups, paired against the protocol v2 PD.</p>""",
    "summary_intro": "<p>Pending.</p>",
    "findings": "<p>Pending.</p>",
    "caveats": "<p>Pending.</p>",
    "next": "<p>Pending.</p>",
}
