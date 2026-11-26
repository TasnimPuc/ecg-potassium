"""Fast path for MIMIC-IV-ECG: read the single PhysioNet ZIP with HTTP Range requests.

Instead of ~470k small-file requests (physionet.org throttles those), we
  1. read only the ZIP central directory (index of all members),
  2. group the needed ECGs into ~256 MB contiguous byte ranges,
  3. download each range with a few parallel connections and extract the members from memory.
Every finished range becomes a 'pack' in the checkpoint store -> fully resumable."""
from __future__ import annotations

import io
import json
import logging
import re
import struct
import threading
import time
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from .waveforms import clean_resample, parse_wfdb

log = logging.getLogger(__name__)
PROJECT_PAGE = "https://physionet.org/content/mimic-iv-ecg/1.0/"


def _session():
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
    s = requests.Session()
    s.mount("https://", HTTPAdapter(pool_maxsize=16, max_retries=Retry(
        total=10, connect=8, read=5, backoff_factor=2, status_forcelist=(429, 500, 502, 503, 504))))
    return s


def remote_size(url) -> int:
    r = _session().head(url, timeout=(30, 60), allow_redirects=True)
    r.raise_for_status()
    return int(r.headers["Content-Length"])


ZIP_CANDIDATES = [
    "https://physionet.org/content/mimic-iv-ecg/get-zip/1.0/",
    "https://physionet.org/static/published-projects/mimic-iv-ecg/"
    "mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0.zip",
]


def _resolve(url) -> str | None:
    """Follow redirects; return the final URL if it is a big file that supports Range requests."""
    s = _session()
    try:
        r = s.get(url, headers={"Range": "bytes=0-3"}, timeout=(30, 60), allow_redirects=True, stream=True)
        ok = r.status_code == 206 and r.raw.read(4) == b"PK\x03\x04"
        total = int(r.headers.get("Content-Range", "/0").split("/")[-1] or 0)
        r.close()
        if ok and total > 1e9:
            return r.url
        log.info("zip candidate %s -> HTTP %s, %.1f GB", url, r.status_code, total / 1e9)
    except Exception as e:
        log.info("zip candidate %s -> %s", url, type(e).__name__)
    return None


def find_zip_url(cfg) -> str:
    if cfg.zip_url:
        return _resolve(cfg.zip_url) or cfg.zip_url
    cands = []
    try:
        html = _session().get(PROJECT_PAGE, timeout=(30, 60)).text
        for h in re.findall(r"""href=["']([^"']+)["']""", html):
            if "get-zip" in h or h.lower().endswith(".zip"):
                cands.append(h if h.startswith("http") else "https://physionet.org" + h)
    except Exception as e:
        log.warning("could not read %s: %s", PROJECT_PAGE, e)
    for u in dict.fromkeys(cands + ZIP_CANDIDATES):
        final = _resolve(u)
        if final:
            log.info("ZIP: %s", final)
            return final
    raise RuntimeError(f"No downloadable ZIP found (tried {list(dict.fromkeys(cands + ZIP_CANDIDATES))}). "
                       "Open the PhysioNet page, copy the 'Download the ZIP file' link and set cfg.zip_url.")


class HTTPRangeFile(io.RawIOBase):
    """Seekable read-only file over HTTP Range requests (enough for zipfile to read the index)."""

    def __init__(self, url, session=None):
        self.url, self.s, self.pos = url, session or _session(), 0
        self.size = remote_size(url)

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.size + off
        return self.pos

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        if n == 0 or self.pos >= self.size:
            return b""
        end = min(self.pos + n, self.size) - 1
        data = fetch_range(self.s, self.url, self.pos, end)
        self.pos += len(data)
        return data

    def readinto(self, b):
        d = self.read(len(b))
        b[:len(d)] = d
        return len(d)


_LIVE = [0]                                    # bytes received by all range fetches (progress heartbeat)


def fetch_range(s, url, start, end, tries=8) -> bytes:
    """Bytes [start, end] inclusive; resumes a broken transfer from where it stopped."""
    buf, pos, attempt = bytearray(), start, 0
    while pos <= end:
        try:
            r = s.get(url, headers={"Range": f"bytes={pos}-{end}"}, stream=True, timeout=(30, 120))
            if r.status_code != 206:
                raise RuntimeError(f"expected HTTP 206, got {r.status_code}")
            for chunk in r.iter_content(1 << 20):
                buf += chunk
                pos += len(chunk)
                _LIVE[0] += len(chunk)
            if pos <= end:
                raise IOError("short read")
        except Exception as e:
            attempt += 1
            if attempt >= tries:
                raise
            log.warning("range %d-%d interrupted at %d (%s), retrying", start, end, pos, type(e).__name__)
            time.sleep(min(60, 2 ** attempt))
    return bytes(buf)


def zip_index(cfg, store) -> pd.DataFrame:
    """Central directory -> DataFrame(path, ext, header_offset, compress_size, compress_type), cached."""
    key = "cache/zip_index.parquet"
    loc = Path(cfg.private_dir) / key
    if loc.exists() or store.get(key, loc):
        return pd.read_parquet(loc)
    url = find_zip_url(cfg)
    log.info("Reading ZIP index of %s ...", url)
    f = HTTPRangeFile(url)
    z = zipfile.ZipFile(io.BufferedReader(f, buffer_size=8 << 20))
    rows = []
    for zi in z.infolist():
        m = re.search(r"(files/p\d+/p\d+/s\d+/\d+)\.(hea|dat)$", zi.filename)
        if m:
            rows.append((m.group(1), m.group(2), zi.header_offset, zi.compress_size, zi.compress_type))
    df = pd.DataFrame(rows, columns=["path", "ext", "header_offset", "compress_size", "compress_type"])
    df.attrs["url"] = url
    log.info("ZIP index: %d members (%.1f GB zip)", len(df), f.size / 1e9)
    loc.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(loc, index=False)
    (loc.parent / "zip_url.txt").write_text(url)
    store.put(loc, key)
    store.put(loc.parent / "zip_url.txt", "cache/zip_url.txt")
    return df


def _member(buf: bytes, base: int, off: int, csize: int, ctype: int) -> bytes:
    p = off - base
    sig, = struct.unpack_from("<I", buf, p)
    if sig != 0x04034B50:
        raise ValueError("bad local header")
    n, m = struct.unpack_from("<HH", buf, p + 26)
    data = buf[p + 30 + n + m: p + 30 + n + m + csize]
    if ctype == zipfile.ZIP_STORED:
        return data
    if ctype == zipfile.ZIP_DEFLATED:
        return zlib.decompressobj(-15).decompress(data)
    raise ValueError(f"compression {ctype} not supported")


def build_packs(cfg, store, labels: pd.DataFrame) -> dict:
    """Writes shards/<split>/pack_XXXXX.{npy,parquet,json}; returns manifest {split: [(key, n)]}."""
    man_key = f"shards/manifest_zip_n{len(labels)}.json"
    man_loc = Path(cfg.private_dir) / man_key
    if store.get(man_key, man_loc):
        return json.loads(man_loc.read_text())
    idx = zip_index(cfg, store)
    url = (Path(cfg.private_dir) / "cache" / "zip_url.txt")
    url = url.read_text().strip() if url.exists() or store.get("cache/zip_url.txt", url) else find_zip_url(cfg)
    piv = idx.pivot_table(index="path", columns="ext", values=["header_offset", "compress_size", "compress_type"],
                          aggfunc="first")
    piv.columns = [f"{a}_{b}" for a, b in piv.columns]
    need = labels.merge(piv.dropna().reset_index(), on="path", how="inner")
    log.info("ECGs found in ZIP: %d / %d labelled", len(need), len(labels))
    need["lo"] = need[["header_offset_hea", "header_offset_dat"]].min(axis=1).astype("int64")
    need["hi"] = (need[["header_offset_hea", "header_offset_dat"]].max(axis=1) + 30 + 512 +
                  need[["compress_size_hea", "compress_size_dat"]].max(axis=1)).astype("int64")
    need["pack"] = (need["lo"] // int(cfg.zip_chunk_mb * 1e6)).astype(int)
    packs = sorted(need["pack"].unique())
    existing = set(store.list("shards"))
    done = {int(re.search(r"pack_(\d+)\.json", k).group(1)) for k in existing if re.search(r"pack_\d+\.json$", k)}
    tmp = Path(cfg.private_dir) / "pack_tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    want = need.groupby("pack").size().to_dict()

    def _receipt_ok(p):                       # a pack from an older label set must be rebuilt
        loc = tmp / f"r{p}.json"
        return p, store.get(f"shards/pack_{p:05d}.json", loc) and json.loads(loc.read_text()).get("n") == want[p]

    with ThreadPoolExecutor(8) as ex:
        valid = {p for p, ok in ex.map(_receipt_ok, sorted(done & set(packs))) if ok}
    stale = len(done & set(packs)) - len(valid)
    todo = [p for p in packs if p not in valid]
    log.info("ZIP packs: %d total, %d already done, %d stale (rebuilt), %d to download (~%.0f MB each)",
             len(packs), len(valid), stale, len(todo), cfg.zip_chunk_mb)
    n_out = 10 * cfg.fs_out
    size = remote_size(url) if todo else 0
    t0, nbytes, lock = time.time(), [0], threading.Lock()

    def do_pack(p):
        g = need[need["pack"] == p]
        lo, hi = int(g["lo"].min()), int(g["hi"].max())
        hi = min(hi, size - 1)
        buf = fetch_range(_session(), url, lo, hi)
        X = {s: [] for s in ("train", "val", "test")}
        M = {s: [] for s in ("train", "val", "test")}
        for r in g.itertuples(index=False):
            try:
                hea = _member(buf, lo, int(r.header_offset_hea), int(r.compress_size_hea), int(r.compress_type_hea))
                dat = _member(buf, lo, int(r.header_offset_dat), int(r.compress_size_dat), int(r.compress_type_dat))
                x, fs = parse_wfdb(hea, dat)
                y = clean_resample(x, fs, cfg.fs_out, n_out)
            except Exception as e:
                log.debug("skip %s: %s", r.path, e)
                y = None
            if y is not None:
                X[r.split].append(y)
                M[r.split].append(r)
        keys = []
        d = tmp / f"p{p}"
        d.mkdir(exist_ok=True)
        for s in X:
            if not X[s]:
                continue
            key = f"shards/{s}/pack_{p:05d}"
            np.save(d / f"{s}.npy", np.stack(X[s]))
            meta = pd.DataFrame(M[s])[[c for c in labels.columns if c != "path"]]
            meta.to_parquet(d / f"{s}.parquet", index=False)
            store.put(d / f"{s}.npy", f"{key}.npy")
            store.put(d / f"{s}.parquet", f"{key}.parquet")
            keys.append([s, key, len(X[s])])
        (d / "r.json").write_text(json.dumps({"keys": keys, "n": len(g)}))
        store.put(d / "r.json", f"shards/pack_{p:05d}.json")             # receipt last = commit
        with lock:
            nbytes[0] += hi - lo
        return p, keys

    stop = threading.Event()

    def heartbeat():                          # a 256 MB pack takes minutes; show that bytes are flowing
        last, t_last = _LIVE[0], time.time()
        while not stop.wait(60):
            now, t = _LIVE[0], time.time()
            log.info("  downloading... %.0f MB/s now, %.1f GB received so far",
                     (now - last) / 1e6 / (t - t_last), now / 1e9)
            last, t_last = now, t

    if todo:
        log.info("Downloading %d packs, %d in parallel; first 'pack done' line comes after one full pack "
                 "(a few minutes), progress every 60 s", len(todo), cfg.zip_parallel)
        threading.Thread(target=heartbeat, daemon=True).start()
        with ThreadPoolExecutor(cfg.zip_parallel) as ex:
            futs = [ex.submit(do_pack, p) for p in todo]
            for i, f in enumerate(as_completed(futs), 1):
                p, keys = f.result()
                el = time.time() - t0
                mbps = nbytes[0] / 1e6 / max(el, 1e-6)
                left = (len(todo) - i) * cfg.zip_chunk_mb / max(mbps, 1e-6) / 60
                log.info("pack %d done (%d/%d) | %d ECGs | %.1f MB/s | ETA %.0f min", p, i, len(todo),
                         sum(k[2] for k in keys), mbps, left)
    stop.set()
    manifest = {"train": [], "val": [], "test": []}
    for p in packs:
        loc = tmp / f"r{p}.json"
        store.get(f"shards/pack_{p:05d}.json", loc)
        for s, key, n in json.loads(loc.read_text())["keys"]:
            manifest[s].append([key, n])
    man_loc.parent.mkdir(parents=True, exist_ok=True)
    man_loc.write_text(json.dumps(manifest))
    store.put(man_loc, man_key)
    return manifest
