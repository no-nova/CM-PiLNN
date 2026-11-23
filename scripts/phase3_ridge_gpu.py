"""GPU cost of the closed-form readout recalibration, measured next to the CPU solve.

Phase 3.2 reports a *training-free* refit of (W_out, b_out).  What matters for deployment is
how long the refit takes on the machine that just ran the forward pass, so this script times
the two halves separately:

  * `forward_ms`  -- the calibration rollouts themselves (128 sequences x T steps, batched);
  * `gram_ms`     -- streaming accumulation A = sum_b H_b^T H_b, r = sum_b H_b^T y_b on the
                     RTX 5070 in fp32 (cudaEvent timing, warm-up 5 / 20 reps);
  * `solve_ms`    -- the fp64 Cholesky/lstsq solve on the CPU (A is tiny: D x D).

Output: results/phase3_ridge_gpu.json (also merged into results/phase3_expressivity.json as
`ridge_gpu`), plus a wall-clock total per (model, method, rho) row.

Run:  python scripts/phase3_ridge_gpu.py
"""
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models import CfC                      # noqa: E402
from src.probe import MemProbe                  # noqa: E402

RES = ROOT / "results"
CKPT = RES / "checkpoints"
TASKS = {"A_oscillator": "oscillator", "B_parity": "parity", "C_largescale": "multiscale"}
LAM = 1e-3
N_CAL, B_CAL, T_CAL = 128, 16, 256
ROWS = [("native-prefix", 0.0), ("ours-jacobian-pi", 0.5)]


def load(name, dev):
    ck = torch.load(CKPT / f"{name}.pt", map_location=dev, weights_only=False)
    m = CfC(ck["cfg"]["d_in"], ck["cfg"]["D"], ck["cfg"]["d_out"]).to(dev)
    m.load_state_dict(ck["state"])
    m.eval()
    return m


def task_batch(name, n, T, dev, seed):
    from src.tasks import TASKS as TT
    fn = TT[TASKS[name]][0]
    g = torch.Generator().manual_seed(seed)
    x, dt, y = fn(n, T, device=dev, gen=g)
    return x, dt, y


def r2(pred, y):
    sse = ((pred - y) ** 2).sum()
    sst = ((y - y.mean(0, keepdim=True)) ** 2).sum()
    return float(1.0 - sse / sst)


def order_for(name, m, method, dev):
    if method == "native-prefix":
        return torch.arange(m.D, device=dev)
    from src.ordering import residual_scores
    x, dt, _ = task_batch(name, 32, 128, dev, 7)
    sc = residual_scores(m, x, dt)          # takes inputs, not states
    J = sc["J"]
    return torch.argsort(J, descending=True)


def forward_h(m, x, dt, order, rho):
    if rho >= 1.0:
        with torch.no_grad():
            _, H = m.forward_exact(x, dt, keep_h=True)
    else:
        hbar = torch.zeros(m.D, device=x.device)
        with torch.no_grad():
            _, H = m.forward_approx(x, dt, hbar=hbar, order=order, rho=rho, keep_h=True)
    return H


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = False
    out = {"lam": LAM, "n_cal": N_CAL, "B_cal": B_CAL, "T_cal": T_CAL, "rows": {}}
    with MemProbe() as probe:
        for name in ("A_oscillator", "B_parity", "C_largescale"):
            m = load(name, dev)
            D = m.D
            # evaluation batch (held out from the calibration stream by seed)
            xe, dte, ye = task_batch(name, 64, 256, dev, 99)
            for method, rho in ROWS:
                order = order_for(name, m, method, dev)
                # ---- calibration rollouts + features
                t0 = time.perf_counter()
                Hb, Yb = [], []
                for k in range(N_CAL // B_CAL):
                    x, dt, y = task_batch(name, B_CAL, T_CAL, dev, 10_000 + k)
                    H = forward_h(m, x, dt, order, rho)
                    Hb.append(H.reshape(-1, D))
                    Yb.append(y.reshape(-1, y.shape[-1]))
                torch.cuda.synchronize()
                forward_ms = 1e3 * (time.perf_counter() - t0)
                Hcat = torch.cat(Hb, 0)
                Ycat = torch.cat(Yb, 0)
                # ---- GPU Gram: time only the accumulation
                for _ in range(5):
                    A = Hcat.t() @ Hcat
                    r = Hcat.t() @ Ycat
                torch.cuda.synchronize()
                ev0, ev1 = torch.cuda.Event(True), torch.cuda.Event(True)
                ts = []
                for _ in range(20):
                    ev0.record()
                    A = Hcat.t() @ Hcat
                    r = Hcat.t() @ Ycat
                    ev1.record()
                    torch.cuda.synchronize()
                    ts.append(ev0.elapsed_time(ev1))
                ts.sort()
                gram_ms = ts[len(ts) // 2]
                # ---- CPU fp64 solve
                t0 = time.perf_counter()
                Ah = A.detach().double().cpu()
                rh = r.detach().double().cpu()
                lam = LAM * float(torch.diagonal(Ah).mean())
                Ah = Ah + lam * torch.eye(Ah.shape[0], dtype=torch.float64)
                L = torch.linalg.cholesky(Ah)
                W = torch.cholesky_solve(rh, L).float()
                solve_ms = 1e3 * (time.perf_counter() - t0)
                # ---- score on the held-out batch
                He = forward_h(m, xe, dte, order, rho)
                pred = He.reshape(-1, D) @ W.to(He.device)
                # bias: mean residual over the calibration rows
                b = (Ycat - Hcat @ W.to(Hcat.device)).mean(0)
                y2 = ye.reshape(-1, ye.shape[-1])
                score = r2(pred + b, y2)
                key = f"{name}|{method}|{rho:g}"
                out["rows"][key] = dict(D=D, forward_ms=round(forward_ms, 1),
                                        gram_ms=round(gram_ms, 3),
                                        solve_ms=round(solve_ms, 1),
                                        total_ms=round(forward_ms + gram_ms + solve_ms, 1),
                                        r2=round(score, 4))
                print(f"{key:34s} fwd {forward_ms:7.1f} ms  gram {gram_ms:6.3f} ms  "
                      f"solve {solve_ms:6.1f} ms  R2 {score:+.3f}", flush=True)
            del m
            torch.cuda.empty_cache()
    out["_memory"] = probe.as_dict()
    json.dump(out, open(RES / "phase3_ridge_gpu.json", "w"), indent=1)
    # merge into the phase-3 log so the paper can cite one file
    p3 = RES / "phase3_expressivity.json"
    d = json.load(open(p3))
    d["ridge_gpu"] = out["rows"]
    json.dump(d, open(p3, "w"), indent=1)
    print("\npeak RssAnon MB:", out["_memory"]["peak_rss_anon_mb"])


if __name__ == "__main__":
    main()
