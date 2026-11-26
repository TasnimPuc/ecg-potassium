"""Resumable training: full state (model/optimizer/scheduler/AMP scaler/epoch/shard position/early stopping)
is checkpointed to the store every `ckpt_minutes` and at every epoch end. Re-run = continue exactly there."""
from __future__ import annotations

import json
import logging
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from .model import ResNet1D

log = logging.getLogger(__name__)


class Paused(Exception):
    """Session time budget used up; state is checkpointed, re-run to continue."""


def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_model(cfg):
    return ResNet1D(cfg.width, cfg.kernel, cfg.dropout)


def label_stats(labels: pd.DataFrame, cfg) -> dict:
    tr = labels[labels.split == "train"]
    hyper, hypo = (tr.k >= cfg.hyper_thr).mean(), (tr.k < cfg.hypo_thr).mean()
    return dict(k_mean=float(tr.k.mean()), k_std=float(tr.k.std() or 1.0),
                pw_hyper=float(np.clip((1 - hyper) / max(hyper, 1e-6), 1, 30)),
                pw_hypo=float(np.clip((1 - hypo) / max(hypo, 1e-6), 1, 30)))


def _augment(x):
    b = x.shape[0]
    x = x * torch.empty(b, 12, 1, device=x.device).uniform_(0.85, 1.15)
    x = x + torch.randn_like(x) * torch.empty(b, 1, 1, device=x.device).uniform_(0.0, 0.03)
    keep = (torch.rand(b, 12, 1, device=x.device) > 0.1).float()
    x = x * keep
    shift = int(torch.randint(-250, 251, (1,)))
    return torch.roll(x, shift, dims=-1)


def _loss(out, k, stats, cfg):
    kz = (k - stats["k_mean"]) / stats["k_std"]
    hyper, hypo = (k >= cfg.hyper_thr).float(), (k < cfg.hypo_thr).float()
    dev = out.device
    return (F.huber_loss(out[:, 0].float(), kz, delta=1.0)
            + F.binary_cross_entropy_with_logits(out[:, 1].float(), hyper, pos_weight=torch.tensor(stats["pw_hyper"], device=dev))
            + 0.5 * F.binary_cross_entropy_with_logits(out[:, 2].float(), hypo, pos_weight=torch.tensor(stats["pw_hypo"], device=dev)))


@torch.no_grad()
def predict(model, shardset, cfg, stats, bs=512) -> pd.DataFrame:
    model.eval()
    dev = device()
    rows = []
    for _, _, X, meta in shardset.iterate(range(len(shardset.entries))):
        outs = []
        for b in range(0, len(X), bs):
            xb = torch.from_numpy(np.asarray(X[b:b + bs], dtype=np.float32)).to(dev)
            with torch.autocast(dev.type, dtype=torch.float16, enabled=cfg.amp and dev.type == "cuda"):
                outs.append(model(xb).float().cpu())
        o = torch.cat(outs).numpy() if outs else np.zeros((0, 3))
        m = meta.copy()
        m["pred_k"] = o[:, 0] * stats["k_std"] + stats["k_mean"]
        m["p_hyper"] = 1 / (1 + np.exp(-o[:, 1]))
        m["p_hypo"] = 1 / (1 + np.exp(-o[:, 2]))
        rows.append(m)
    return pd.concat(rows, ignore_index=True)


def train(cfg, store, sets: dict, stats: dict):
    dev = device()
    ck = f"ckpt/{cfg.run_name}"
    loc = Path(cfg.private_dir) / "ckpt"
    loc.mkdir(parents=True, exist_ok=True)
    model = make_model(cfg).to(dev)
    net = torch.nn.DataParallel(model) if torch.cuda.device_count() > 1 else model
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    tr = sets["train"]
    spe = sum(math.ceil(n / cfg.batch_size) for _, n in tr.entries)
    total, warm = cfg.epochs * spe, max(1, int(cfg.warmup_frac * cfg.epochs * spe))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total))))
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp and dev.type == "cuda")
    st = dict(epoch=0, shards_done=0, step=0, best_auc=-1.0, best_epoch=-1, bad_epochs=0, history=[],
              finished=False, stats=stats)

    if store.get(f"{ck}/last.pt", loc / "last.pt"):
        s = torch.load(loc / "last.pt", map_location=dev, weights_only=False)
        model.load_state_dict(s["model"]); opt.load_state_dict(s["opt"]); sched.load_state_dict(s["sched"])
        scaler.load_state_dict(s["scaler"]); st = s["state"]
        log.info("loaded checkpoint: %d of %d epochs already done%s", len(st["history"]), cfg.epochs,
                 " (training finished)" if st["finished"] else "")
        log.info("RESUME training: epoch %d, shard %d/%d, step %d, best val AUROC %.4f (epoch %d)",
                 st["epoch"] + 1, st["shards_done"], len(tr.entries), st["step"], st["best_auc"], st["best_epoch"] + 1)

    loaded_epochs = len(st["history"])

    def save(tag="last"):
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "scaler": scaler.state_dict(), "state": st}, loc / "last.pt")
        store.put(loc / "last.pt", f"{ck}/last.pt")
        log.info("checkpoint (%s): epoch %d shard %d step %d", tag, st["epoch"] + 1, st["shards_done"], st["step"])

    def save_best():
        torch.save(model.state_dict(), loc / "best.pth")
        store.put(loc / "best.pth", f"{ck}/best.pth")

    if st["finished"]:
        store.get(f"{ck}/best.pth", loc / "best.pth")
        model.load_state_dict(torch.load(loc / "best.pth", map_location=dev))
        return model, {**st, "loaded_epochs": loaded_epochs}

    t0 = last_ck = time.time()
    deadline = t0 + cfg.max_runtime_hours * 3600
    shards_session = 0
    log.info("Training on %s x%d | %d train ECGs, %d steps/epoch", dev, max(torch.cuda.device_count(), 1), tr.n, spe)
    while st["epoch"] < cfg.epochs:
        ep = st["epoch"]
        order = np.random.default_rng(cfg.seed + ep).permutation(len(tr.entries))
        net.train()
        losses = []
        for i, key, X, meta in tr.iterate(order[st["shards_done"]:]):
            perm = np.random.default_rng(cfg.seed + 100003 * ep + int(i)).permutation(len(X))
            kk = meta["k"].to_numpy(np.float32)
            for b in range(0, len(perm), cfg.batch_size):
                idx = np.sort(perm[b:b + cfg.batch_size])
                xb = _augment(torch.from_numpy(X[idx]).to(dev, non_blocking=True).float())   # fp16 over PCIe
                kb = torch.from_numpy(kk[idx]).to(dev)
                with torch.autocast(dev.type, dtype=torch.float16, enabled=cfg.amp and dev.type == "cuda"):
                    out = net(xb)
                loss = _loss(out, kb, stats, cfg)
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(opt); scaler.update(); sched.step()
                st["step"] += 1
                losses.append(float(loss.detach()))
            st["shards_done"] += 1
            shards_session += 1
            if time.time() - last_ck >= cfg.ckpt_minutes * 60:
                save("periodic"); last_ck = time.time()
            if cfg.debug_crash_after_shards is not None and shards_session >= cfg.debug_crash_after_shards:
                raise RuntimeError("simulated session death")
            if time.time() > deadline:
                save("time budget")
                raise Paused(f"paused at epoch {ep + 1}, shard {st['shards_done']}")
        # ---- epoch end: validate ----
        pv = predict(model, sets["val"], cfg, stats)
        y = (pv.k >= cfg.hyper_thr).astype(int)
        auc = float(roc_auc_score(y, pv.p_hyper)) if 0 < y.sum() < len(y) else float("nan")
        mae = float(np.mean(np.abs(pv.pred_k - pv.k)))
        st["history"].append(dict(epoch=ep + 1, train_loss=float(np.mean(losses)) if losses else None,
                                  val_auroc_hyper=auc, val_mae=mae, lr=float(sched.get_last_lr()[0])))
        log.info("epoch %d/%d | train loss %.4f | val AUROC(K>=%.1f) %.4f | val MAE %.3f",
                 ep + 1, cfg.epochs, st["history"][-1]["train_loss"] or float("nan"), cfg.hyper_thr, auc, mae)
        if auc > st["best_auc"]:
            st.update(best_auc=auc, best_epoch=ep, bad_epochs=0)
            save_best()
        else:
            st["bad_epochs"] += 1
        st["epoch"], st["shards_done"] = ep + 1, 0
        stop = st["bad_epochs"] >= cfg.patience
        if stop or st["epoch"] >= cfg.epochs:
            st["finished"] = True
        save("epoch end"); last_ck = time.time()
        if stop:
            log.info("early stopping (no improvement for %d epochs)", cfg.patience)
            break
    st["finished"] = True
    save("finished")
    model.load_state_dict(torch.load(loc / "best.pth", map_location=dev))
    (loc / "history.json").write_text(json.dumps(st["history"], indent=1))
    return model, {**st, "loaded_epochs": loaded_epochs}


def history_table(st) -> pd.DataFrame:
    """Every epoch (also those restored from the checkpoint); ★ = best val AUROC, used for the final model."""
    h = pd.DataFrame(st.get("history", []))
    if not len(h):
        return pd.DataFrame({"info": ["no epochs finished yet"]})
    h["source"] = np.where(h.epoch <= st.get("loaded_epochs", 0), "checkpoint", "this session")
    h["best"] = np.where(h.epoch == st.get("best_epoch", -1) + 1, "★", "")
    return h
