import json

import numpy as np
import pandas as pd
import pytest
import torch

from src.config import Config
from src.model import ResNet1D, gradcam
from src.pipeline import run
from src.waveforms import LEADS, clean_resample, parse_wfdb, synthetic_record


def _labels(n=360, seed=0):
    r = np.random.default_rng(seed)
    k = np.clip(r.normal(4.3, 0.8, n), 2.5, 7.5)
    subj = r.integers(0, n // 2, n)
    split = np.where(subj % 10 < 6, "train", np.where(subj % 10 < 8, "val", "test"))
    return pd.DataFrame({"subject_id": subj, "study_id": np.arange(n), "path": [f"files/x/{i}" for i in range(n)],
                         "ecg_time": pd.Timestamp("2150-01-01"), "k": k, "k_dt_min": r.normal(0, 20, n),
                         "itemid": 50971, "age": r.integers(20, 90, n), "gender": r.choice(["F", "M"], n),
                         "split": split})


def _fetcher(lab):
    kmap = dict(zip(lab.path, lab.k))
    return lambda path: synthetic_record(kmap[path], seed=hash(path) % 1000)


def _mm(lab):
    r = np.random.default_rng(1)
    return pd.DataFrame({"study_id": lab.study_id, "rr_interval": r.normal(850, 100, len(lab)),
                         "qrs_onset": 40, "qrs_end": 130, "p_onset": -100, "p_end": -10,
                         "t_end": 400 + 30 * (lab.k - 4) + r.normal(0, 20, len(lab)),
                         "p_axis": 50, "qrs_axis": 30, "t_axis": 40 + 10 * (lab.k - 4)})


def _cfg(tmp, **kw):
    return Config(work_dir=str(tmp / "o"), private_dir=str(tmp / "p"), cache_dir=str(tmp / "cache"),
                  local_store=str(tmp / "bucket"), shard_size=40, width=8, epochs=3, batch_size=16,
                  n_boot=20, gradcam_n=30, download_workers=4, min_free_gb=0.0, ckpt_minutes=0,
                  train_needs_gpu=False, **kw)


def test_parse_roundtrip():
    x, fs = parse_wfdb(*synthetic_record(5.0))
    assert x.shape == (12, 5000) and fs == 500
    y = clean_resample(x, fs, 250, 2500)
    assert y.shape == (12, 2500) and y.dtype == np.float16 and len(LEADS) == 12


def test_model_and_gradcam():
    m = ResNet1D(width=8)
    x = torch.randn(3, 12, 2500)
    assert m(x).shape == (3, 3)
    cam = gradcam(m, x)
    assert cam.shape == (3, 2500) and torch.allclose(cam.sum(1), torch.ones(3), atol=1e-4)


def test_end_to_end_and_resume(tmp_path):
    lab = _labels()
    with pytest.raises(RuntimeError, match="simulated"):
        run(_cfg(tmp_path, debug_crash_after_shards=4), fetcher=_fetcher(lab), labels=lab, mm=_mm(lab))
    ck = torch.load(tmp_path / "bucket" / "ecg-potassium" / "ckpt" / "main" / "last.pt", weights_only=False)
    assert ck["state"]["step"] > 0 and not ck["state"]["finished"]
    calls = {"n": 0}
    f0 = _fetcher(lab)

    def counting(p):
        calls["n"] += 1
        return f0(p)
    s = run(_cfg(tmp_path), fetcher=counting, labels=lab, mm=_mm(lab))
    assert calls["n"] == 0                      # shards reused from the store, nothing re-downloaded
    assert s["status"] == "complete" and len(s["training"]) and s["training"]["epochs_run"] >= 1
    assert (tmp_path / "o" / "model" / "ecg_k_resnet1d.pth").exists()
    assert "Grad-CAM" in (tmp_path / "o" / "report" / "REPORT.md").read_text()
    # inference CLI path
    from src.predict import load, predict_record
    m, b = load(str(tmp_path / "o" / "model" / "ecg_k_resnet1d.pth"))
    out = predict_record(m, b, *synthetic_record(6.5))
    assert 1 < out["predicted_k_mmol_l"] < 12


def test_match_handles_bigquery_dtypes():
    from src.labels import match
    ecg = pd.DataFrame({"subject_id": np.array([1, 1, 2], dtype="int64"), "study_id": [10, 11, 12],
                        "path": ["a", "b", "c"], "ecg_time": ["2150-01-01 10:00", "2150-01-02 10:00", "2150-01-01 08:00"]})
    k = pd.DataFrame({"subject_id": pd.array([1, 2, 2, None], dtype="Int64"),
                      "charttime": pd.to_datetime(["2150-01-01 10:20", "2150-01-01 09:30", "2150-01-01 06:00", "2150-01-01 08:00"]).astype("datetime64[us]"),
                      "itemid": pd.array([50971] * 4, dtype="Int64"), "valuenum": [5.9, 4.1, 3.0, 4.0],
                      "hemolyzed": [False, False, False, False]})
    pat = pd.DataFrame({"subject_id": pd.array([1, 2], dtype="Int64"), "gender": ["F", "M"],
                        "anchor_age": pd.array([60, 70], dtype="Int64"), "anchor_year": pd.array([2150, 2150], dtype="Int64")})
    m = match(ecg, k, pat, Config())
    assert sorted(m.study_id.tolist()) == [10]            # 12 is 90 min away (> 60), 11 has no lab nearby
    assert m.k.iloc[0] == 5.9 and m.age.iloc[0] == 60


def test_failed_downloads_do_not_commit_shard(tmp_path):
    from src.store import LocalStore
    from src.waveforms import build_shards
    lab = _labels(80)
    cfg = _cfg(tmp_path)
    store = LocalStore(cfg.local_store, cfg.gcs_prefix)
    good = _fetcher(lab)

    def flaky(p):
        if int(p.split("/")[-1]) % 2:
            raise ConnectionError("timeout")
        return good(p)
    with pytest.raises(RuntimeError, match="NOT saved"):
        build_shards(cfg, store, lab, flaky)
    assert not [k for k in store.list("shards") if k.endswith(".parquet")]
    man = build_shards(cfg, store, lab, good)              # next run retries and completes
    assert sum(n for s in man.values() for _, n in s) == len(lab)


def test_zip_packs(tmp_path, monkeypatch):
    """Build a real ZIP of synthetic WFDB records and read it through the range-request code path."""
    import io as _io
    import zipfile as _zf

    from src import zipsource
    from src.store import LocalStore
    lab = _labels(90)
    zp = tmp_path / "ecg.zip"
    lab = lab.assign(path=[f"files/p1000/p1000{i:04d}/s{i}/{i}" for i in range(len(lab))])
    with _zf.ZipFile(zp, "w", compression=_zf.ZIP_DEFLATED) as z:
        for r in lab.itertuples():
            h, d = synthetic_record(r.k, seed=r.study_id)
            z.writestr(f"mimic-iv-ecg-1.0/{r.path}.hea", h)
            z.writestr(f"mimic-iv-ecg-1.0/{r.path}.dat", d)
    blob = zp.read_bytes()
    calls = {"n": 0}

    def fake_range(s, url, a, b, tries=8):
        calls["n"] += 1
        return blob[a:b + 1]

    class FakeFile(_io.BytesIO):
        def __init__(self, url, session=None):
            super().__init__(blob)
            self.size = len(blob)
    monkeypatch.setattr(zipsource, "fetch_range", fake_range)
    monkeypatch.setattr(zipsource, "HTTPRangeFile", FakeFile)
    monkeypatch.setattr(zipsource, "remote_size", lambda url: len(blob))
    monkeypatch.setattr(zipsource, "_resolve", lambda url: url)
    cfg = _cfg(tmp_path, zip_url="https://example/ecg.zip", zip_chunk_mb=0.2)
    store = LocalStore(cfg.local_store, cfg.gcs_prefix)
    man = zipsource.build_packs(cfg, store, lab)
    assert sum(n for s in man.values() for _, n in s) == len(lab)
    assert calls["n"] > 1
    # resume: remove manifest, packs are reused -> no new downloads
    for f in (tmp_path / "bucket").rglob("manifest_zip*"):
        f.unlink()
    import shutil
    shutil.rmtree(tmp_path / "p", ignore_errors=True)
    n0 = calls["n"]
    man2 = zipsource.build_packs(cfg, store, lab)
    assert calls["n"] == n0 and man2 == man


def _steps(cfg, **kw):
    """Runs the notebook's step cells (built from src/pipeline.py::run) instead of run()."""
    import sys
    from pathlib import Path as _P
    root = _P(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "tools"))
    from nbsteps import exec_steps
    return exec_steps(root / "src" / "pipeline.py", cfg, **kw)["result"]


def test_notebook_steps(tmp_path):
    lab = _labels()
    s = _steps(_cfg(tmp_path), fetcher=_fetcher(lab), labels=lab, mm=_mm(lab))
    assert s["status"] == "complete" and (tmp_path / "o" / "model" / "ecg_k_resnet1d.pth").exists()


def test_notebook_stops_without_gpu(tmp_path):
    if torch.cuda.is_available():
        pytest.skip("needs a CPU-only session")
    lab = _labels()
    with pytest.raises(Exception, match="shards_ready_no_gpu"):
        c = _cfg(tmp_path)
        c.train_needs_gpu = True
        _steps(c, fetcher=_fetcher(lab), labels=lab, mm=_mm(lab))


def test_rerun_shows_saved_epochs(tmp_path):
    from src.train import history_table
    lab = _labels()
    s1 = _steps(_cfg(tmp_path), fetcher=_fetcher(lab), labels=lab, mm=_mm(lab))
    import sys
    from pathlib import Path as _P
    sys.path.insert(0, str(_P(__file__).resolve().parents[1] / "tools"))
    from nbsteps import exec_steps
    ns = exec_steps(_P(__file__).resolve().parents[1] / "src" / "pipeline.py", _cfg(tmp_path),
                    fetcher=_fetcher(lab), labels=lab, mm=_mm(lab))
    h = history_table(ns["st"])
    assert s1["status"] == "complete" and len(h) == len(ns["st"]["history"]) > 0 and set(h.source) == {"checkpoint"}
    assert (h.best == "★").sum() == 1
