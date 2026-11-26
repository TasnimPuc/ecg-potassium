"""GitHub-ready REPORT.md (aggregate only)."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


def _t(df: pd.DataFrame) -> str:
    f = lambda v: "–" if isinstance(v, float) and np.isnan(v) else (f"{v:.3f}" if isinstance(v, float) else str(v))
    return "\n".join(["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)] +
                     ["| " + " | ".join(f(v) for v in r) + " |" for r in df.itertuples(index=False)])


def build(work_dir) -> str:
    out = Path(work_dir)
    rep = out / "report"
    (rep / "figures").mkdir(parents=True, exist_ok=True)
    for f in (out / "figures").glob("*.png"):
        shutil.copy(f, rep / "figures" / f.name)
    m = json.loads((out / "metrics.json").read_text())
    c, cfg, reg, base = m["cohort"], m["config"], m["regression"], m["baseline_machine_measurements"]
    cls = pd.DataFrame(m["classification"])
    small = cls["n positive"] < cfg["min_cell"]
    cls["AUROC (95% CI)"] = ["–" if sm or np.isnan(r.AUROC) else f"{r.AUROC:.3f} ({r['AUROC lo']:.3f}–{r['AUROC hi']:.3f})"
                             for sm, (_, r) in zip(small, cls.iterrows())]
    for col in ("prevalence", "AUPRC", "Sens @ 90% spec"):      # prevalence x n would reveal a small count
        cls[col] = cls[col].astype(object).where(~small, np.nan)
    cls["n positive"] = [f"<{cfg['min_cell']}" if sm else v for sm, v in zip(small, cls["n positive"])]
    cls = cls[["Target", "n positive", "prevalence", "AUROC (95% CI)", "AUPRC", "Sens @ 90% spec"]]
    sub = pd.read_csv(out / "tables" / "subgroups.csv")
    hyper = m["classification"][0]
    comp = pd.DataFrame([
        {"Model": "1D ResNet on raw 12-lead ECG (this work)", "AUROC (K ≥ %.1f)" % cfg["hyper_thr"]:
         f"{hyper['AUROC']:.3f} ({hyper['AUROC lo']:.3f}–{hyper['AUROC hi']:.3f})"},
        {"Model": "XGBoost on ECG-machine intervals/axes (baseline)", "AUROC (K ≥ %.1f)" % cfg["hyper_thr"]:
         f"{base['auroc']:.3f} ({base['auroc_lo']:.3f}–{base['auroc_hi']:.3f})" if base else "–"}])
    gc = m.get("gradcam", {})
    gct = pd.DataFrame([{"Group": g, "beats": v["n_beats"], "Grad-CAM share in ST–T (120–450 ms)": v["cam_share_st_t"],
                         "share in QRS (−50–80 ms)": v["cam_share_qrs"]} for g, v in gc.items()])
    md = f"""# Results — Potassium (Hyperkalemia) from the 12-lead ECG

*MIMIC-IV-ECG waveforms + MIMIC-IV potassium within ±{cfg['window_min']} min (hemolyzed samples excluded). Patient-level split.
Aggregate results only; cells with fewer than {cfg['min_cell']} are suppressed.*

## Cohort
| | |
|---|---|
| ECGs with a matched K+ | {c['n_ecgs']:,} |
| Patients | {c['n_patients']:,} |
| Waveforms used (train / val / test) | {c['n_waveforms_used']['train']:,} / {c['n_waveforms_used']['val']:,} / {c['n_waveforms_used']['test']:,} |
| K ≥ {cfg['hyper_thr']} / K < {cfg['hypo_thr']} prevalence | {c['hyper_prevalence']:.2%} / {c['hypo_prevalence']:.2%} |
| Median K+ · median |ECG–lab time gap| | {c['k_median']:.1f} mmol/L · {c['abs_dt_median_min']:.0f} min |

## Test performance (ECG only — no labs, no demographics)
{_t(cls)}

**K+ regression:** MAE {reg['mae']:.3f} mmol/L · RMSE {reg['rmse']:.3f} · Pearson r {reg['pearson_r']:.3f} · within ±0.5 mmol/L {reg['within_0_5']:.1%}

![ROC](figures/roc.png)
![Predicted vs measured](figures/pred_vs_true.png)

## Deep model vs classical ECG measurements
{_t(comp)}

## Where does the model look? (population Grad-CAM, R-peak aligned)
{_t(gct) if len(gct) else '–'}

![Grad-CAM beat](figures/gradcam_beat.png)

## Subgroups (K ≥ {cfg['hyper_thr']})
{_t(sub)}

## Training
Best epoch {m['training']['best_epoch']} of {m['training']['epochs_run']} · best val AUROC {m['training']['best_val_auroc']:.3f}

![Training](figures/training.png)
"""
    (rep / "REPORT.md").write_text(md, encoding="utf-8")
    return md
