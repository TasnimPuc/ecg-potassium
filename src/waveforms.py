"""Download MIMIC-IV-ECG WFDB records in parallel, parse, resample 500->250 Hz, write float16 shards.
Shards go to the checkpoint store; finished shards are skipped on re-run (resumable)."""
from __future__ import annotations

import io
import json
import logging
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)
LEADS = ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]
_GAIN = re.compile(r"^([-\d.eE+]+)(?:\(([-\d]+)\))?(?:/(\w+))?$")


def parse_wfdb(hea: bytes, dat: bytes) -> tuple[np.ndarray, float]:
    """Return (12, n) float32 signal in mV in standard lead order, and fs. Supports format 16 natively."""
    lines = [l for l in hea.decode("latin-1").splitlines() if l.strip() and not l.startswith("#")]
    head = lines[0].split()
    nsig, fs = int(head[1]), float(head[2].split("/")[0])
    nsamp = int(head[3]) if len(head) > 3 else None
    sig = [l.split() for l in lines[1:1 + nsig]]
    fmts = {s[1].split("x")[0].split(":")[0] for s in sig}
    if fmts != {"16"}:
        return _parse_with_wfdb(hea, dat)
    raw = np.frombuffer(dat, dtype="<i2")
    if nsamp is None:
        nsamp = len(raw) // nsig
    raw = raw[: nsamp * nsig].reshape(nsamp, nsig).astype(np.float32)
    out = np.zeros((12, nsamp), np.float32)
    for j, s in enumerate(sig):
        m = _GAIN.match(s[2])
        gain = float(m.group(1)) if m else 200.0
        gain = gain if gain != 0 else 200.0
        adczero = int(s[4]) if len(s) > 4 and s[4].lstrip("-").isdigit() else 0
        base = int(m.group(2)) if m and m.group(2) is not None else adczero
        unit = (m.group(3) or "mV") if m else "mV"
        x = raw[:, j]
        x[x == -32768] = np.nan
        x = (x - base) / gain
        if unit.lower() in ("uv", "µv"):
            x = x / 1000.0
        name = s[-1].upper() if len(s) >= 9 else LEADS[j] if j < 12 else ""
        if name in LEADS:
            out[LEADS.index(name)] = x
    return out, fs


def _parse_with_wfdb(hea: bytes, dat: bytes):
    import tempfile

    import wfdb
    with tempfile.TemporaryDirectory() as d:
        rec = hea.decode("latin-1").split()[0]
        (Path(d) / f"{rec}.hea").write_bytes(hea)
        (Path(d) / f"{rec}.dat").write_bytes(dat)
        sig, f = wfdb.rdsamp(str(Path(d) / rec))
    names = [n.upper() for n in f["sig_name"]]
    out = np.zeros((12, sig.shape[0]), np.float32)
    for j, n in enumerate(names):
        if n in LEADS:
            out[LEADS.index(n)] = sig[:, j]
    return out, float(f["fs"])


def clean_resample(x: np.ndarray, fs: float, fs_out: int, n_out: int) -> np.ndarray | None:
    """(12, n) mV -> (12, n_out) float16, median-centred, NaN->0, clipped. None if unusable."""
    from scipy.signal import resample_poly
    if np.isnan(x).mean() > 0.5:
        return None
    x = np.nan_to_num(x, nan=0.0)
    if int(fs) != fs_out:
        from math import gcd
        g = gcd(int(fs), fs_out)
        x = resample_poly(x, fs_out // g, int(fs) // g, axis=1)
    x = x[:, :n_out]
    if x.shape[1] < n_out:
        x = np.pad(x, ((0, 0), (0, n_out - x.shape[1])))
    x = x - np.median(x, axis=1, keepdims=True)
    if (np.abs(x).max(axis=1) < 1e-3).sum() > 6:          # >6 flat leads
        return None
    return np.clip(x, -10, 10).astype(np.float16)


def s3_list(bucket_url: str, prefix: str, timeout=30, max_pages=20) -> tuple[list[str], list[str]]:
    """Anonymous S3 ListObjectsV2 (paginated) -> (sub-prefixes, keys) directly under `prefix`."""
    import xml.etree.ElementTree as ET

    import requests
    ns = {"s": "http://s3.amazonaws.com/doc/2006-03-01/"}
    pre, keys, token = [], [], None
    for _ in range(max_pages):
        params = {"list-type": "2", "prefix": prefix, "delimiter": "/", "max-keys": "1000"}
        if token:
            params["continuation-token"] = token
        r = requests.get(bucket_url, params=params, timeout=timeout)
        r.raise_for_status()
        root = ET.fromstring(r.content)
        pre += [e.text for e in root.findall("s:CommonPrefixes/s:Prefix", ns)]
        keys += [e.text for e in root.findall("s:Contents/s:Key", ns)]
        t = root.find("s:NextContinuationToken", ns)
        if t is None or not t.text:
            break
        token = t.text
    return pre, keys


def discover_s3_base(cfg, record_path: str) -> str | None:
    """Find the S3 prefix under which `record_path` (files/pNNNN/...) exists. Returns a base URL or None.
    Searches the bucket's top level for folders whose name contains the dataset name (or 'ecg')."""
    import requests
    seen = []
    try:
        top, _ = s3_list(cfg.s3_bucket_url, "")
    except Exception as e:
        log.warning("S3 top-level listing failed: %s", e)
        top = []
    name = cfg.s3_dataset.lower()
    starts = [p for p in top if name in p.lower()] + [p for p in top if "ecg" in p.lower() and name not in p.lower()]
    seen.append(f"top-level matches: {starts[:15]} (of {len(top)} folders)")
    frontier = starts[:10] or [name + "/"]
    for _ in range(3):
        nxt = []
        for p in frontier:
            try:
                if requests.head(cfg.s3_bucket_url + p + record_path + ".hea", timeout=15).status_code == 200:
                    log.info("S3 prefix found: %s", p)
                    return cfg.s3_bucket_url + p
            except Exception:
                pass
            try:
                sub, _ = s3_list(cfg.s3_bucket_url, p)
                seen.append(f"{p} -> {sub[:8]}")
                nxt += [x for x in sub if not x.endswith("files/")][:20]
            except Exception as e:
                seen.append(f"{p} -> list failed: {type(e).__name__}")
        frontier = nxt
        if not frontier:
            break
    log.warning("S3 prefix for %s not found. %s", record_path, seen)
    return None


class HttpFetcher:
    """Thread-safe downloader. Probes the mirrors once, uses the fastest working one; physionet.org (throttled)
    is only a fallback with limited concurrency. Errors are counted so failures are visible, never silent."""

    def __init__(self, bases, workers, org_workers=6, probe_path=None):
        import collections
        self.bases = list(bases)
        self.local = threading.local()
        self.workers = workers
        self.org_sem = threading.Semaphore(org_workers)
        self.errors = collections.Counter()
        self.lock = threading.Lock()
        if probe_path:
            self.bases = self.probe(probe_path)

    def _session(self):
        if not hasattr(self.local, "s"):
            import requests
            from requests.adapters import HTTPAdapter
            from urllib3.util.retry import Retry
            s = requests.Session()
            s.mount("https://", HTTPAdapter(pool_connections=8, pool_maxsize=self.workers, max_retries=Retry(
                total=8, connect=6, read=3, backoff_factor=1.5, status_forcelist=(429, 500, 502, 503, 504))))
            self.local.s = s
        return self.local.s

    def probe(self, path):
        ok, report = [], []
        for b in self.bases:
            t = time.time()
            try:
                r = self._session().get(b + path + ".hea", timeout=(10, 30))
                report.append(f"{b} -> HTTP {r.status_code} in {time.time() - t:.2f}s")
                if r.ok:
                    ok.append((time.time() - t, b))
            except Exception as e:
                report.append(f"{b} -> {type(e).__name__}: {str(e)[:120]}")
        for line in report:
            log.info("mirror probe: %s", line)
        if not ok:
            raise RuntimeError("No ECG mirror reachable from this machine:\n  " + "\n  ".join(report) +
                               "\nCheck that Internet is ON in the Kaggle notebook settings.")
        fast = [b for _, b in sorted(ok)]
        org = [b for b in fast if "physionet.org" in b]
        chosen = [b for b in fast if "physionet.org" not in b][:1] + org[:1]
        log.info("Using mirror(s): %s", chosen)
        return chosen

    def _get(self, base, url):
        if "physionet.org" in base:
            with self.org_sem:
                return self._session().get(url, timeout=(30, 90))
        return self._session().get(url, timeout=(10, 60))

    def __call__(self, path: str) -> tuple[bytes, bytes]:
        last = None
        for b in self.bases:
            try:
                h = self._get(b, b + path + ".hea")
                h.raise_for_status()
                d = self._get(b, b + path + ".dat")
                d.raise_for_status()
                return h.content, d.content
            except Exception as e:
                last = e
                code = getattr(getattr(e, "response", None), "status_code", None)
                with self.lock:
                    self.errors[f"{b.split('/')[2]}: {code or type(e).__name__}"] += 1
        raise RuntimeError(f"{path}: {last}")


def _eta(labels, split, ci, size, rate):
    order = ["train", "val", "test"]
    left = sum(int((labels.split == s).sum()) for s in order[order.index(split) + 1:])
    left += max(int((labels.split == split).sum()) - (ci + 1) * size, 0)
    m = left / max(rate, 1e-6) / 60
    return f"{m:.0f} min" if m < 120 else f"{m / 60:.1f} h"


def build_shards(cfg, store, labels: pd.DataFrame, fetcher=None) -> dict:
    """Writes shards/<split>/shard_XXXXX.{npy,parquet}; returns manifest {split: [(key, n), ...]}."""
    man_key = f"shards/manifest_n{len(labels)}.json"
    man_loc = Path(cfg.private_dir) / man_key
    if store.get(man_key, man_loc):
        return json.loads(man_loc.read_text())
    if fetcher is None:
        bases = list(cfg.wave_base_urls)
        found = discover_s3_base(cfg, labels.path.iloc[0])
        if found:
            bases = [found] + [b for b in bases if b != found]
        fetcher = HttpFetcher(bases, cfg.download_workers, cfg.physionet_org_workers, probe_path=labels.path.iloc[0])
        if all("physionet.org" in b for b in fetcher.bases):
            log.warning("Only physionet.org is reachable: downloads will be slow (throttled). "
                        "Progress is saved per shard, so re-running continues where it stopped.")
    n_out = 10 * cfg.fs_out
    tmp = Path(cfg.private_dir) / "shard_tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    manifest, t0, done_ecgs = {}, time.time(), 0
    existing = set(store.list("shards"))
    for split in ("train", "val", "test"):
        d = labels[labels.split == split].sample(frac=1.0, random_state=cfg.seed).reset_index(drop=True)
        chunks = [d.iloc[i:i + cfg.shard_size] for i in range(0, len(d), cfg.shard_size)]
        manifest[split] = []
        for ci, ch in enumerate(chunks):
            key = f"shards/{split}/shard_{ci:05d}"
            if f"{key}.parquet" in existing and f"{key}.npy" in existing:
                loc = tmp / f"{split}_{ci}.parquet"
                store.get(f"{key}.parquet", loc)
                n_ok = int(len(pd.read_parquet(loc)))
                # shards written by this version carry a .json receipt; older ones must be ~complete
                if f"{key}.json" in existing or n_ok >= 0.95 * len(ch):
                    manifest[split].append([key, n_ok])
                    continue
                log.warning("%s has only %d/%d ECGs (written by an older run) -> rebuilding", key, n_ok, len(ch))

            failed = []

            def work(row):
                try:
                    raw = fetcher(row.path)
                except Exception as e:                 # download problem -> counts against the shard
                    failed.append(str(e)[:200])
                    return None
                try:
                    x, fs = parse_wfdb(*raw)
                    return clean_resample(x, fs, cfg.fs_out, n_out)   # None = unusable signal (expected, rare)
                except Exception:
                    return None
            with ThreadPoolExecutor(cfg.download_workers) as ex:
                res = list(ex.map(work, ch.itertuples(index=False)))
            if len(failed) > cfg.max_fail_frac * len(ch):
                errs = dict(getattr(fetcher, "errors", {}))
                raise RuntimeError(f"{key}: {len(failed)}/{len(ch)} downloads failed -> shard NOT saved "
                                   f"(re-run to retry). Error counts: {errs}. Example: {failed[0]}")
            ok = [i for i, r in enumerate(res) if r is not None]
            X = np.stack([res[i] for i in ok]) if ok else np.zeros((0, 12, n_out), np.float16)
            meta = ch.iloc[ok].drop(columns=["path"]).reset_index(drop=True)
            np.save(tmp / "x.npy", X)
            meta.to_parquet(tmp / "m.parquet", index=False)
            store.put(tmp / "x.npy", f"{key}.npy")
            (tmp / "r.json").write_text(json.dumps({"n": len(ch), "ok": len(ok), "download_failed": len(failed)}))
            store.put(tmp / "r.json", f"{key}.json")
            store.put(tmp / "m.parquet", f"{key}.parquet")      # parquet last = commit marker
            manifest[split].append([key, len(ok)])
            done_ecgs += len(ch)
            el = time.time() - t0
            log.info("%s shard %d/%d: %d/%d ok (%d download errors) | %.0f ECG/s | ETA %s", split, ci + 1, len(chunks),
                     len(ok), len(ch), len(failed), done_ecgs / max(el, 1e-6),
                     _eta(labels, split, ci, cfg.shard_size, done_ecgs / max(el, 1e-6)))
    man_loc.parent.mkdir(parents=True, exist_ok=True)
    man_loc.write_text(json.dumps(manifest))
    store.put(man_loc, man_key)
    return manifest


def synthetic_record(k: float, seed: int = 0, fs=500, n=5000) -> tuple[bytes, bytes]:
    """WFDB bytes for tests/demo: ECG-like signal whose T-wave grows with K+ (no real patient data)."""
    r = np.random.default_rng(seed)
    t = np.arange(n) / fs
    hr = r.uniform(55, 95)
    beat = (t * hr / 60) % 1.0
    qrs = np.exp(-((beat - 0.2) / 0.012) ** 2)
    tw = np.exp(-((beat - 0.45) / (0.05 - 0.004 * (k - 4))) ** 2) * (0.15 + 0.12 * (k - 4))
    sig = np.stack([(qrs * r.uniform(0.6, 1.4) + tw + r.normal(0, 0.02, n)) for _ in range(12)], 1)
    dig = np.round(sig * 200).astype("<i2")
    hdr = [f"rec 12 {fs} {n}"] + [f"rec.dat 16 200/mV 16 0 {dig[0, j]} 0 0 {LEADS[j].replace('AV', 'aV')}" for j in range(12)]
    return ("\n".join(hdr) + "\n").encode(), dig.tobytes()


def check_source(cfg, path="files/p1000/p10000032/s40689238/40689238") -> None:
    """Pre-flight: which open MIMIC-IV-ECG mirrors are reachable from this machine (and how fast)?"""
    import requests
    if cfg.wave_source in ("zip", "auto"):
        from . import zipsource
        try:
            url = zipsource.find_zip_url(cfg)
            size = zipsource.remote_size(url)
            t = time.time()
            b = zipsource.fetch_range(zipsource._session(), url, size - 8 * 2 ** 20, size - 1)
            print(f"  ECG ZIP: {url}\n           {size / 1e9:.1f} GB, Range OK, {len(b) / 1e6 / (time.time() - t):.1f} MB/s on one connection")
        except Exception as e:
            print(f"  ECG ZIP: FAILED -> {type(e).__name__}: {str(e)[:200]}")
        return
    try:
        top, _ = s3_list(cfg.s3_bucket_url, "")
        hits = [p for p in top if "ecg" in p.lower() or "mimic" in p.lower()]
        print(f"  S3 bucket top level: {len(top)} folders; mimic/ecg-like: {hits[:20]}")
    except Exception as e:
        print(f"  S3 listing failed: {type(e).__name__}: {str(e)[:150]}")
    b = discover_s3_base(cfg, path)
    print(f"  S3 auto-discovered base: {b or 'NOT FOUND'}")
    for b in cfg.wave_base_urls:
        t = time.time()
        try:
            r = requests.get(b + path + ".hea", timeout=(10, 30))
            print(f"  ECG mirror {b}: HTTP {r.status_code} ({time.time() - t:.2f}s)")
        except Exception as e:
            print(f"  ECG mirror {b}: FAILED -> {type(e).__name__}: {str(e)[:150]}")
