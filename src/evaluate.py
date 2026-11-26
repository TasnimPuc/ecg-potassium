"""Test metrics with bootstrap CIs, subgroups, machine-measurement baseline, population Grad-CAM beat."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import torch
from scipy.signal import find_peaks
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

from .model import gradcam
from .train import device

log = logging.getLogger(__name__)


def _boot(y, p, fn, n, seed):
    y, p = np.asarray(y), np.asarray(p)
    if not 0 < y.sum() < len(y):
        return np.nan, np.nan, np.nan
    rng, v = np.random.default_rng(seed), []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if 0 < y[i].sum() < len(i):
            v.append(fn(y[i], p[i]))
    return float(fn(y, p)), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def classification(pred: pd.DataFrame, cfg) -> pd.DataFrame:
    rows = []
    for name, y, p in ((f"K ≥ {cfg.hyper_thr}", pred.k >= cfg.hyper_thr, pred.p_hyper),
                       (f"K ≥ {cfg.severe_thr}", pred.k >= cfg.severe_thr, pred.p_hyper),
                       (f"K < {cfg.hypo_thr}", pred.k < cfg.hypo_thr, pred.p_hypo)):
        y = y.astype(int).to_numpy()
        auc = _boot(y, p, roc_auc_score, cfg.n_boot, cfg.seed)
        ap = _boot(y, p, average_precision_score, cfg.n_boot, cfg.seed)
        sens90 = np.nan
        if 0 < y.sum() < len(y):
            fpr, tpr, _ = roc_curve(y, p)
            sens90 = float(tpr[np.searchsorted(fpr, 0.10, side="right") - 1])   # sensitivity at 90% specificity
        rows.append({"Target": name, "n positive": int(y.sum()), "prevalence": y.mean(),
                     "AUROC": auc[0], "AUROC lo": auc[1], "AUROC hi": auc[2],
                     "AUPRC": ap[0], "Sens @ 90% spec": sens90})
    return pd.DataFrame(rows)


def regression(pred: pd.DataFrame) -> dict:
    e = pred.pred_k - pred.k
    return dict(mae=float(e.abs().mean()), rmse=float(np.sqrt((e ** 2).mean())),
                pearson_r=float(np.corrcoef(pred.pred_k, pred.k)[0, 1]),
                within_0_5=float((e.abs() <= 0.5).mean()))


def subgroups(pred: pd.DataFrame, cfg) -> pd.DataFrame:
    p = pred.copy()
    p["age_group"] = pd.cut(p.age, [0, 39, 64, 79, 200], labels=["18-39", "40-64", "65-79", "80+"]).astype(str)
    p["sex"] = p.gender.fillna("Unknown")
    rows = []
    for a in ("sex", "age_group"):
        for g, d in p.groupby(a):
            y = (d.k >= cfg.hyper_thr).astype(int)
            ok = y.sum() >= cfg.min_cell and (len(y) - y.sum()) >= cfg.min_cell
            rows.append(dict(attribute=a, group=g, n=len(d) if len(d) >= cfg.min_cell else f"<{cfg.min_cell}",
                             n_positive=int(y.sum()) if y.sum() >= cfg.min_cell else f"<{cfg.min_cell}",
                             auroc=float(roc_auc_score(y, d.p_hyper)) if ok else np.nan,
                             mae=float((d.pred_k - d.k).abs().mean()) if len(d) >= cfg.min_cell else np.nan))
    return pd.DataFrame(rows)


MM_COLS = ["rr_interval", "p_onset", "p_end", "qrs_onset", "qrs_end", "t_end", "p_axis", "qrs_axis", "t_axis"]


def mm_features(mm: pd.DataFrame) -> pd.DataFrame:
    f = mm[["study_id"] + [c for c in MM_COLS if c in mm]].copy()
    for c in MM_COLS:
        if c in f:
            f[c] = pd.to_numeric(f[c], errors="coerce")
            f.loc[f[c].abs() >= 29999, c] = np.nan            # machine "not measured" sentinel
    f["hr"] = 60000 / f["rr_interval"]
    f["qrs_dur"] = f["qrs_end"] - f["qrs_onset"]
    f["pr"] = f["qrs_onset"] - f["p_onset"]
    f["qt"] = f["t_end"] - f["qrs_onset"]
    f["qtc"] = f["qt"] / np.sqrt(f["rr_interval"] / 1000)
    return f


def baseline_machine(labels: pd.DataFrame, mm: pd.DataFrame, cfg) -> dict:
    """XGBoost on the ECG machine's own interval/axis measurements (the classical approach)."""
    import xgboost as xgb
    f = mm_features(mm)
    d = labels.merge(f, on="study_id", how="inner")
    cols = [c for c in f.columns if c != "study_id"]
    tr, te = d[d.split == "train"], d[d.split == "test"]
    if len(tr) < 100 or len(te) < 50:
        return {}
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = xgb.XGBClassifier(n_estimators=600, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                          tree_method="hist", device=dev, eval_metric="auc")
    m.fit(tr[cols], (tr.k >= cfg.hyper_thr).astype(int))
    p = m.predict_proba(te[cols])[:, 1]
    y = (te.k >= cfg.hyper_thr).astype(int)
    auc = _boot(y, p, roc_auc_score, cfg.n_boot, cfg.seed)
    return dict(auroc=auc[0], auroc_lo=auc[1], auroc_hi=auc[2], auprc=float(average_precision_score(y, p)),
                n_test=int(len(te)), features=cols)


def gradcam_beats(model, shardset, pred: pd.DataFrame, cfg, fs=None) -> dict:
    """Average lead-II beat and Grad-CAM (hyperkalemia logit) aligned on R-peaks, per group (aggregate only)."""
    fs = fs or cfg.fs_out
    dev = device()
    pre, post = int(0.25 * fs), int(0.60 * fs)
    want = {"hyper (true K ≥ thr)": pred.k >= cfg.hyper_thr, "normal (3.5–5.0)": pred.k.between(3.5, 5.0)}
    ids = {g: set(pred.loc[m, "study_id"].sample(min(cfg.gradcam_n, int(m.sum())), random_state=cfg.seed))
           for g, m in want.items()}
    acc = {g: {"ecg": [], "cam": []} for g in want}
    for _, _, X, meta in shardset.iterate(range(len(shardset.entries))):
        for g in want:
            idx = np.where(meta.study_id.isin(ids[g]).to_numpy())[0]
            for b in range(0, len(idx), 128):
                j = idx[b:b + 128]
                xb = torch.from_numpy(np.asarray(X[j], dtype=np.float32)).to(dev)
                cam = gradcam(model, xb).cpu().numpy()
                lead2 = xb[:, 1].cpu().numpy()
                for s, c in zip(lead2, cam):
                    a = np.abs(s)
                    pk, _ = find_peaks(a, distance=int(0.3 * fs), height=0.5 * np.percentile(a, 99))
                    for r in pk:
                        if r - pre >= 0 and r + post <= len(s):
                            acc[g]["ecg"].append(s[r - pre:r + post] * np.sign(s[r]))   # R upright
                            acc[g]["cam"].append(c[r - pre:r + post] * len(s))
    t = (np.arange(-pre, post) / fs) * 1000
    res = {"t_ms": t.tolist()}
    for g, a in acc.items():
        if len(a["ecg"]) < cfg.min_cell:
            continue
        e, c = np.array(a["ecg"]), np.array(a["cam"])
        qrs = (t >= -50) & (t <= 80)
        stt = (t >= 120) & (t <= 450)
        res[g] = dict(n_beats=len(e), ecg=np.median(e, 0).tolist(), cam=c.mean(0).tolist(),
                      cam_share_qrs=float(c[:, qrs].sum() / c.sum()), cam_share_st_t=float(c[:, stt].sum() / c.sum()))
    return res


def plots(pred, cls, beats, hist, base, cfg, fdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # ROC
    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    for name, y, p in ((f"K≥{cfg.hyper_thr}", pred.k >= cfg.hyper_thr, pred.p_hyper),
                       (f"K≥{cfg.severe_thr}", pred.k >= cfg.severe_thr, pred.p_hyper),
                       (f"K<{cfg.hypo_thr}", pred.k < cfg.hypo_thr, pred.p_hypo)):
        if 0 < y.sum() < len(y):
            fpr, tpr, _ = roc_curve(y, p)
            ax.plot(fpr, tpr, label=f"{name} (AUROC {roc_auc_score(y, p):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=.8); ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.legend(fontsize=8)
    ax.set_title("ECG-only model, test set"); fig.tight_layout(); fig.savefig(fdir / "roc.png", dpi=150); plt.close(fig)
    # predicted vs true (2D histogram = aggregate, no individual points)
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    h = ax.hist2d(pred.k, pred.pred_k, bins=[np.arange(2, 8.01, .1), np.arange(2, 8.01, .1)], cmin=cfg.min_cell, cmap="viridis")
    fig.colorbar(h[3], ax=ax, label=f"ECGs per bin (bins <{cfg.min_cell} hidden)")
    ax.plot([2, 8], [2, 8], "w--", lw=.8); ax.set_xlabel("measured K+ (mmol/L)"); ax.set_ylabel("ECG-predicted K+")
    fig.tight_layout(); fig.savefig(fdir / "pred_vs_true.png", dpi=150); plt.close(fig)
    # Grad-CAM beat
    groups = [g for g in beats if g != "t_ms"]
    if groups:
        t = np.array(beats["t_ms"])
        fig, axes = plt.subplots(1, len(groups), figsize=(5.2 * len(groups), 3.6), squeeze=False)
        for ax, g in zip(axes[0], groups):
            b = beats[g]
            ax.plot(t, b["ecg"], color="k", lw=1.2, label="median beat (lead II)")
            ax2 = ax.twinx()
            ax2.fill_between(t, 0, b["cam"], color="tab:red", alpha=.35, label="Grad-CAM")
            ax.axvspan(120, 450, color="tab:blue", alpha=.07)
            ax.set_title(f"{g}\nST–T share {b['cam_share_st_t']:.0%} · QRS share {b['cam_share_qrs']:.0%}", fontsize=9)
            ax.set_xlabel("ms from R peak"); ax2.set_yticks([])
        fig.tight_layout(); fig.savefig(fdir / "gradcam_beat.png", dpi=150); plt.close(fig)
    # training curve
    if hist:
        h = pd.DataFrame(hist)
        fig, ax = plt.subplots(figsize=(5, 3.2))
        ax.plot(h.epoch, h.val_auroc_hyper, "o-", label="val AUROC")
        ax2 = ax.twinx(); ax2.plot(h.epoch, h.val_mae, "s--", color="tab:orange", label="val MAE")
        ax.set_xlabel("epoch"); ax.set_ylabel("AUROC"); ax2.set_ylabel("MAE")
        fig.tight_layout(); fig.savefig(fdir / "training.png", dpi=150); plt.close(fig)
