"""ECG list (open MIMIC-IV-ECG) + potassium labels (credentialed MIMIC-IV labevents via BigQuery)."""
from __future__ import annotations

import io
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from . import gcp

log = logging.getLogger(__name__)

K_SQL = """
SELECT subject_id, charttime, itemid, valuenum,
       REGEXP_CONTAINS(LOWER(IFNULL(comments, '')), r'hemoly') AS hemolyzed
FROM `{HOSP}.labevents`
WHERE itemid IN ({ITEMS}) AND valuenum IS NOT NULL AND valuenum BETWEEN {KMIN} AND {KMAX}
"""
PAT_SQL = "SELECT subject_id, gender, anchor_age, anchor_year FROM `{HOSP}.patients`"


def _cached(cfg, store, key, build):
    loc = Path(cfg.private_dir) / key
    if loc.exists() or store.get(key, loc):
        return pd.read_parquet(loc)
    df = build()
    loc.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(loc, index=False)
    store.put(loc, key)
    return df


def _http_csv(cfg, name: str) -> pd.DataFrame:
    import requests
    last = None
    for base in cfg.wave_base_urls:
        try:
            r = requests.get(base + name, timeout=300)
            r.raise_for_status()
            log.info("Downloaded %s from %s (%.0f MB)", name, base, len(r.content) / 1e6)
            return pd.read_csv(io.BytesIO(r.content), low_memory=False)
        except Exception as e:
            last = e
            log.warning("%s from %s failed: %s", name, base, e)
    raise RuntimeError(f"Could not download {name}: {last}")


def record_list(cfg, store) -> pd.DataFrame:
    return _cached(cfg, store, "cache/record_list.parquet", lambda: _http_csv(cfg, "record_list.csv"))


def machine_measurements(cfg, store) -> pd.DataFrame:
    return _cached(cfg, store, "cache/machine_measurements.parquet",
                   lambda: _http_csv(cfg, "machine_measurements.csv"))


def _hosp(cfg):
    client = gcp.bq_client(cfg.project_id)
    return client, gcp.resolve_dataset(client, cfg.hosp_dataset, cfg.hosp_candidates, "labevents")


def potassium(cfg, store) -> pd.DataFrame:
    def build():
        client, hosp = _hosp(cfg)
        return gcp.query_df(client, K_SQL.format(HOSP=hosp, ITEMS=", ".join(map(str, cfg.k_itemids)),
                                                 KMIN=cfg.k_min, KMAX=cfg.k_max))
    return _cached(cfg, store, "cache/potassium.parquet", build)


def patients(cfg, store) -> pd.DataFrame:
    def build():
        client, hosp = _hosp(cfg)
        return gcp.query_df(client, PAT_SQL.format(HOSP=hosp))
    return _cached(cfg, store, "cache/patients.parquet", build)


def match(ecg: pd.DataFrame, k: pd.DataFrame, pat: pd.DataFrame, cfg) -> pd.DataFrame:
    """Nearest (non-hemolyzed) potassium within +-window of each ECG; patient-level split."""
    ecg = ecg.copy()
    ecg["ecg_time"] = pd.to_datetime(ecg["ecg_time"])
    k = k.copy()
    k["charttime"] = pd.to_datetime(k["charttime"])
    if cfg.exclude_hemolyzed and "hemolyzed" in k:
        n0 = len(k)
        k = k[~k["hemolyzed"].astype(bool)]
        log.info("Removed %d hemolyzed K+ results", n0 - len(k))
    ecg = ecg.dropna(subset=["ecg_time"]).sort_values("ecg_time")
    k = k.dropna(subset=["charttime"]).sort_values("charttime")
    m = pd.merge_asof(ecg, k[["subject_id", "charttime", "valuenum", "itemid"]].rename(columns={"valuenum": "k"}),
                      left_on="ecg_time", right_on="charttime", by="subject_id", direction="nearest",
                      tolerance=pd.Timedelta(minutes=cfg.window_min))
    m = m.dropna(subset=["k"]).copy()
    m["k_dt_min"] = (m["charttime"] - m["ecg_time"]).dt.total_seconds() / 60
    m = m.merge(pat, on="subject_id", how="left")
    m["age"] = m["anchor_age"] + (m["ecg_time"].dt.year - m["anchor_year"])
    rng = np.random.default_rng(cfg.seed)
    subs = m["subject_id"].unique()
    u = pd.Series(rng.random(len(subs)), index=subs)
    r = m["subject_id"].map(u)
    tr, va, _ = cfg.split
    m["split"] = np.where(r < tr, "train", np.where(r < tr + va, "val", "test"))
    keep = ["subject_id", "study_id", "path", "ecg_time", "k", "k_dt_min", "itemid", "age", "gender", "split"]
    m = m[keep].reset_index(drop=True)
    if cfg.max_ecgs:
        m = m.sample(min(cfg.max_ecgs, len(m)), random_state=cfg.seed).reset_index(drop=True)
    return m


def build_labels(cfg, store) -> pd.DataFrame:
    key = f"cache/labels_w{cfg.window_min}_n{cfg.max_ecgs}.parquet"

    def build():
        rl = record_list(cfg, store)
        if "ecg_time" not in rl:
            mm = machine_measurements(cfg, store)[["study_id", "ecg_time"]]
            rl = rl.merge(mm, on="study_id", how="left")
        k, pat = potassium(cfg, store), patients(cfg, store)
        lab = match(rl, k, pat, cfg)
        log.info("Labeled ECGs: %d (%d patients) | K>=%.1f: %.2f%% | K<%.1f: %.2f%%", len(lab),
                 lab.subject_id.nunique(), cfg.hyper_thr, 100 * (lab.k >= cfg.hyper_thr).mean(),
                 cfg.hypo_thr, 100 * (lab.k < cfg.hypo_thr).mean())
        return lab
    return _cached(cfg, store, key, build)
