"""End-to-end: labels -> waveform shards -> resumable training -> test evaluation -> .pth + report."""
from __future__ import annotations

import json
import logging
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from . import evaluate, labels as lab_mod, waveforms
from .config import Config
from .data import ShardSet
from .store import make_store
from .train import Paused, history_table, label_stats, predict, train

log = logging.getLogger("ecgk")


def run(cfg: Config, fetcher=None, labels=None, mm=None) -> dict:
    t0 = time.time()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", force=True)
    out, priv = Path(cfg.work_dir), Path(cfg.private_dir)
    for d in (out / "model", out / "tables", out / "figures", priv):
        d.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg.seed)
    store = make_store(cfg)

    # %% Labels: K⁺ near each ECG | BigQuery, cached in the bucket; split by patient.
    # show: lab.split.value_counts()
    lab = labels if labels is not None else lab_mod.build_labels(cfg, store)
    log.info("Labels: %d ECGs / %d patients | split %s", len(lab), lab.subject_id.nunique(),
             lab.split.value_counts().to_dict())
    # %% Waveforms → shards (resumable) | Range requests on the PhysioNet zip; every finished pack is committed to the bucket.
    if fetcher is None and cfg.wave_source in ("zip", "auto"):
        from . import zipsource
        try:
            manifest = zipsource.build_packs(cfg, store, lab)
        except Exception as e:
            if cfg.wave_source == "zip":
                raise
            log.warning("ZIP route failed (%s) -> per-file downloads", e)
            manifest = waveforms.build_shards(cfg, store, lab)
    else:
        manifest = waveforms.build_shards(cfg, store, lab, fetcher)
    if cfg.train_needs_gpu and not torch.cuda.is_available():
        log.warning("All waveform shards are ready in the bucket, but this session has no GPU. "
                    "Switch the accelerator to GPU T4 x2 and Run All again to train (shards are reused).")
        return {"status": "shards_ready_no_gpu", "n_shards": sum(len(v) for v in manifest.values())}
    sets = {s: ShardSet(cfg, store, manifest[s]) for s in ("train", "val", "test")}
    stats = label_stats(lab, cfg)
    # %% Train ResNet-1D (resumable) | Full training state is saved every 10 minutes. The table shows every epoch, also the ones loaded from the checkpoint (★ = best).
    # show: history_table(st)
    try:
        model, st = train(cfg, store, sets, stats)
    except Paused as e:
        log.warning("PAUSED: %s -> Run All again (same run_name) to continue.", e)
        return {"status": "paused", "detail": str(e)}
    # %% Evaluate on test | Bootstrap CIs, subgroups, machine-measurement baseline, Grad-CAM beats.
    # show: cls
    # show: reg
    pred = predict(model, sets["test"], cfg, stats)
    cls = evaluate.classification(pred, cfg)
    reg = evaluate.regression(pred)
    sub = evaluate.subgroups(pred, cfg)
    try:
        mmdf = mm if mm is not None else lab_mod.machine_measurements(cfg, store)
        base = evaluate.baseline_machine(lab, mmdf, cfg)
    except Exception as e:
        log.warning("machine-measurement baseline skipped: %s", e)
        base = {}
    beats = evaluate.gradcam_beats(model, sets["test"], pred, cfg)
    evaluate.plots(pred, cls, beats, st["history"], base, cfg, out / "figures")

    # %% Save model + report | `.pth` bundle with everything inference needs.
    bundle = {"state_dict": model.state_dict(), "arch": "ResNet1D",
              "arch_kwargs": dict(width=cfg.width, kernel=cfg.kernel, dropout=cfg.dropout),
              "stats": stats, "fs": cfg.fs_out, "seconds": 10, "leads": waveforms.LEADS,
              "thresholds": dict(hyper=cfg.hyper_thr, severe=cfg.severe_thr, hypo=cfg.hypo_thr)}
    torch.save(bundle, out / "model" / "ecg_k_resnet1d.pth")
    cohort = dict(n_ecgs=int(len(lab)), n_patients=int(lab.subject_id.nunique()),
                  by_split={s: int(n) for s, n in lab.split.value_counts().items()},
                  n_waveforms_used={s: int(sets[s].n) for s in sets},
                  hyper_prevalence=float((lab.k >= cfg.hyper_thr).mean()),
                  hypo_prevalence=float((lab.k < cfg.hypo_thr).mean()),
                  k_median=float(lab.k.median()), abs_dt_median_min=float(lab.k_dt_min.abs().median()))
    metrics = dict(cohort=cohort, classification=cls.to_dict("records"), regression=reg,
                   baseline_machine_measurements=base, training=dict(best_epoch=st["best_epoch"] + 1,
                   best_val_auroc=st["best_auc"], epochs_run=len(st["history"])),
                   gradcam={g: {k: v for k, v in b.items() if k.startswith(("cam_share", "n_beats"))}
                            for g, b in beats.items() if g != "t_ms"},
                   config=cfg.to_dict(), runtime_min=round((time.time() - t0) / 60, 2))
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1, default=float))
    (out / "model" / "model_card.json").write_text(json.dumps(
        {k: bundle[k] for k in ("arch", "arch_kwargs", "stats", "fs", "seconds", "leads", "thresholds")}
        | {"test": metrics["classification"], "regression": reg}, indent=1, default=float))
    cls.to_csv(out / "tables" / "classification.csv", index=False)
    sub.to_csv(out / "tables" / "subgroups.csv", index=False)
    (out / "tables" / "training_history.json").write_text(json.dumps(st["history"], indent=1))
    from .report import build as build_report
    build_report(out)

    # patient-level predictions -> private only
    (priv / "preds").mkdir(exist_ok=True)
    pred[["study_id", "k", "pred_k", "p_hyper", "p_hypo"]].to_parquet(priv / "preds" / "test_predictions.parquet")
    store.put_dir(priv / "preds", f"runs/{cfg.run_name}/private")
    store.put_dir(out, f"runs/{cfg.run_name}/results")
    if cfg.cleanup_private:
        shutil.rmtree(priv, ignore_errors=True)
    a = cls.iloc[0]
    log.info("DONE | test AUROC(%s)=%.3f (%.3f-%.3f) | MAE=%.3f | model -> %s", a.Target, a.AUROC, a["AUROC lo"],
             a["AUROC hi"], reg["mae"], store.uri(f"runs/{cfg.run_name}/results/model/ecg_k_resnet1d.pth"))
    return {"status": "complete", **metrics}
