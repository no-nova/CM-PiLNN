"""Peak host-memory probe (RssAnon vs VmRSS).

The 1.5 GB budget in this project is about *anonymous* host memory: `VmRSS` also counts
CUDA driver mappings and file-backed pages, which are the driver's own bookkeeping and
page cache rather than our tensors.  Both numbers are recorded so the claim is auditable.
"""
from __future__ import annotations

import threading
import time


class MemProbe:
    def __init__(self, period=0.02):
        self.period = period
        self.peak_anon = 0.0
        self.peak_rss = 0.0
        self._stop = threading.Event()
        self._thr = threading.Thread(target=self._run, daemon=True)

    @staticmethod
    def _status():
        anon = rss = 0.0
        with open("/proc/self/status") as fh:
            for line in fh:
                if line.startswith("RssAnon:"):
                    anon = float(line.split()[1]) / 1024.0
                elif line.startswith("VmRSS:"):
                    rss = float(line.split()[1]) / 1024.0
        return anon, rss

    def _run(self):
        while not self._stop.is_set():
            a, r = self._status()
            self.peak_anon = max(self.peak_anon, a)
            self.peak_rss = max(self.peak_rss, r)
            time.sleep(self.period)

    def __enter__(self):
        self._thr.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thr.join(timeout=1.0)

    def start(self):
        self._thr.start()
        return self

    def stop(self):
        self._stop.set()
        self._thr.join(timeout=1.0)

    def as_dict(self):
        return {"peak_rss_anon_mb": round(self.peak_anon, 1),
                "peak_vm_rss_mb": round(self.peak_rss, 1)}
