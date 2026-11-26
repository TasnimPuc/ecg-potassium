"""Inference on any 12-lead WFDB ECG with a trained .pth bundle.

    python -m src.predict --weights ecg_k_resnet1d.pth --record path/to/40689238      # .hea/.dat pair
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .model import ResNet1D
from .waveforms import clean_resample, parse_wfdb


def load(weights: str, device="cpu"):
    b = torch.load(weights, map_location=device, weights_only=False)
    m = ResNet1D(**b["arch_kwargs"])
    m.load_state_dict(b["state_dict"])
    return m.eval().to(device), b


@torch.no_grad()
def predict_record(model, bundle, hea: bytes, dat: bytes) -> dict:
    x, fs = parse_wfdb(hea, dat)
    x = clean_resample(x, fs, bundle["fs"], bundle["fs"] * bundle["seconds"])
    if x is None:
        raise ValueError("ECG unusable (mostly missing or flat leads)")
    o = model(torch.from_numpy(x.astype(np.float32))[None])[0].numpy()
    s, t = bundle["stats"], bundle["thresholds"]
    return {"predicted_k_mmol_l": round(float(o[0] * s["k_std"] + s["k_mean"]), 2),
            f"p_hyperkalemia_k_ge_{t['hyper']}": round(float(1 / (1 + np.exp(-o[1]))), 4),
            f"p_hypokalemia_k_lt_{t['hypo']}": round(float(1 / (1 + np.exp(-o[2]))), 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--record", required=True, help="WFDB record path without extension")
    a = ap.parse_args()
    model, bundle = load(a.weights)
    r = Path(a.record)
    print(json.dumps(predict_record(model, bundle, r.with_suffix(".hea").read_bytes(), r.with_suffix(".dat").read_bytes()), indent=1))


if __name__ == "__main__":
    main()
