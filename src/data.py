"""Shard loader: local disk cache (auto-evicted when space is low) + background prefetch from the store."""
from __future__ import annotations

import logging
import queue
import shutil
import threading
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


class ShardSet:
    def __init__(self, cfg, store, entries):
        self.cfg, self.store = cfg, store
        self.entries = [(k, n) for k, n in entries if n > 0]
        self.cache = Path(cfg.cache_dir)
        self.cache.mkdir(parents=True, exist_ok=True)
        self.n = sum(n for _, n in self.entries)

    def _free_gb(self):
        return shutil.disk_usage(self.cache).free / 1e9

    def _evict(self, keep):
        files = sorted(self.cache.rglob("*.npy"), key=lambda p: p.stat().st_atime)
        for f in files:
            if self._free_gb() > self.cfg.min_free_gb:
                break
            if f.with_suffix("").as_posix().endswith(keep):
                continue
            f.unlink(missing_ok=True)
            f.with_suffix(".parquet").unlink(missing_ok=True)

    def load(self, key):
        x, m = self.cache / f"{key}.npy", self.cache / f"{key}.parquet"
        if not (x.exists() and m.exists()):
            self._evict(keep=key)
            self.store.get(f"{key}.parquet", m)
            self.store.get(f"{key}.npy", x)
        return np.load(x, mmap_mode="r"), pd.read_parquet(m)

    def iterate(self, order, prefetch=2):
        """Yield (idx, key, X, meta) in the given shard order, loading the next shards in background."""
        q: queue.Queue = queue.Queue(maxsize=prefetch)

        def producer():
            try:
                for i in order:
                    k = self.entries[i][0]
                    X, m = self.load(k)
                    q.put((i, k, np.ascontiguousarray(X), m))
            except Exception as e:      # surface errors in the consumer
                q.put(e)
            q.put(None)
        threading.Thread(target=producer, daemon=True).start()
        while True:
            item = q.get()
            if item is None:
                return
            if isinstance(item, Exception):
                raise item
            yield item
