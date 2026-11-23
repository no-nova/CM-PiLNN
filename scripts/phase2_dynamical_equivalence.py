#!/usr/bin/env python3
"""Phase 2: dynamical-equivalence ablation of the two-stage decoupled pi-CfC.

For each trained model and each rho in RHO_GRID we compare six execution strategies against
the exact recurrence on the same trajectories:

  1 pure-linear        rho = 0, Mamba-style LTV scan (no hot channels at all)
  2 native-prefix      first rho*D channels of the native order kept nonlinear
  3 random-pi          mean +- std over 5 random permutations
  4 variance-pi        channels ranked by state variance Var(h_t,i)
  5 ours-jacobian-pi   channels ranked by the output-weighted residual J_i
  6 ours-jacobian-picard1  as 5, plus one Picard refinement of the cold scan

Reported per cell: output-trajectory relative L2 error, task loss (MSE), and task metric
(R^2 for the dynamical tasks, sign accuracy for the parity automaton).  The associative
scan is also verified against a sequential scan of the same cold channels.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models import CfC, assoc_scan, rel_l2, scan_sequential  # noqa: E402
from src.ordering import (gini, order_by_score, order_identity, order_random,  # noqa: E402
                          residual_scores)
from src.probe import MemProbe  # noqa: E402
from src.tasks import task_metric  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT = os.path.join(ROOT, "results", "checkpoints")
RHO_GRID = [0.0, 0.0625, 0.125, 0.1875, 0.25, 0.375, 0.5, 1.0]
SCAN_GATE_FP32 = 1e-5
SCAN_GATE_FP64 = 1e-13


def load(name, device):
    ck = torch.load(os.path.join(CKPT, f"{name}.pt"), weights_only=False)
    model = CfC(**ck["cfg"]).to(device)
    model.load_state_dict({k: v.to(device) for k, v in ck["state"].items()})
    ev = torch.load(os.path.join(CKPT, f"{name}_eval.pt"), weights_only=False)
    return ck, model, ev


def scan_check(model, x, dt, order, rho, dtype):
    """Associative parallel scan vs sequential scan on the cold channels."""
    m = model.to(dtype)
    xx, dtt = x.to(dtype), dt.to(dtype)
    D = m.D
    n_cold = int(round((1.0 - rho) * D))
    cold = order[:n_cold]
    with torch.no_grad():
        _, H = m.forward_exact(xx, dtt, keep_h=True)
        hbar = H.reshape(-1, D).mean(0)
        if n_cold == 0:
            return 0.0
        a, b = m._cold_gates(xx, dtt, hbar, cold)          # (B,T,n_cold)
        h0 = xx.new_zeros(xx.shape[0], n_cold)
        hp = assoc_scan(a, b, h0.unsqueeze(1))
        hs = scan_sequential(a, b, h0.unsqueeze(1))
        return rel_l2(hp, hs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=["A_oscillator", "B_parity", "C_largescale"])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--n-random", type=int, default=5)
    args = ap.parse_args()
    dev = args.device
    probe = MemProbe().__enter__()

    out, scan_out = {}, {}
    for name in args.models:
        t0 = time.time()
        ck, model, ev = load(name, dev)
        x, dt, y = ev["x"].to(dev), ev["dt"].to(dev), ev["y"].to(dev)
        task = ck["task"]
        D = model.D
        with torch.no_grad():
            y_exact, H = model.forward_exact(x, dt, keep_h=True)
        hbar = H.reshape(-1, D).mean(0)
        sc = residual_scores(model, x, dt, hbar=hbar)
        J, var = sc["J"], H[:, 1:].reshape(-1, D).var(0)
        o_nat = order_identity(D, x.device)
        o_var = order_by_score(var)
        o_jac = order_by_score(J)
        o_rand = [order_random(D, seed=s, device=x.device) for s in range(args.n_random)]
        print(f"\n=== {name} (D={D}, T={x.shape[1]}, task={task}) ===", flush=True)
        print(f"  J gini {gini(J):.3f}   rho grid {RHO_GRID}", flush=True)

        rows = {}
        for rho in RHO_GRID:
            cell = {}
            # 1 pure-linear: identical to native-prefix at rho=0, reported once
            cell["pure-linear"] = [("identity", 0)]
            cell["native-prefix"] = [("identity", 0)]
            cell["random-pi"] = [("random", s) for s in range(len(o_rand))]
            cell["variance-pi"] = [("variance", 0)]
            cell["ours-jacobian-pi"] = [("jacobian", 0)]
            cell["ours-jacobian-picard1"] = [("jacobian", 1)]
            for meth, specs in cell.items():
                errs, mses, mets = [], [], []
                for kind, aux in specs:
                    if kind == "identity":
                        order, pic = o_nat, aux
                    elif kind == "variance":
                        order, pic = o_var, aux
                    elif kind == "jacobian":
                        order, pic = o_jac, aux
                    elif kind == "random":
                        order, pic = o_rand[aux], 0
                    else:
                        raise ValueError(kind)
                    r_eff = 0.0 if meth == "pure-linear" else rho
                    with torch.no_grad():
                        ya, _ = model.forward_approx(x, dt, h0=None, hbar=hbar, order=order,
                                                     rho=r_eff, picard=pic)
                    errs.append(rel_l2(ya, y_exact))
                    mses.append(float(torch.nn.functional.mse_loss(ya, y)))
                    mets.append(task_metric(task, ya, y))
                rows.setdefault(meth, {})[str(rho)] = {
                    "rel_l2": sum(errs) / len(errs),
                    "rel_l2_std": (float(torch.tensor(errs).std()) if len(errs) > 1 else 0.0),
                    "mse": sum(mses) / len(mses),
                    "metric": sum(mets) / len(mets),
                    "metric_std": (float(torch.tensor(mets).std()) if len(mets) > 1 else 0.0),
                }
            tags = {"pure-linear": "pure", "native-prefix": "native", "random-pi": "rand",
                    "variance-pi": "var", "ours-jacobian-pi": "jac", "ours-jacobian-picard1": "jac+P"}
            line = "  ".join(f"{tags[m]}:{rows[m][str(rho)]['rel_l2']:.2e}" for m in rows)
            print(f"  rho={rho:<7} {line}", flush=True)

        # scan gate
        sc_res = {}
        for r in (0.25, 0.5):
            e32 = scan_check(model, x, dt, o_jac, r, torch.float32)
            e64 = scan_check(model, x, dt, o_jac, r, torch.float64)
            sc_res[str(r)] = {"fp32": e32, "fp64": e64,
                              "pass": bool(e32 < SCAN_GATE_FP32 and e64 < SCAN_GATE_FP64)}
            print(f"  scan check rho={r}: fp32 {e32:.2e}  fp64 {e64:.2e}  "
                  f"{'ok' if sc_res[str(r)]['pass'] else 'FAIL'}", flush=True)
        model.to(torch.float32)
        scan_out[name] = {**sc_res, "gate_fp32": SCAN_GATE_FP32, "gate_fp64": SCAN_GATE_FP64}
        out[name] = {"task": task, "D": D, "T": int(x.shape[1]), "rho_grid": RHO_GRID,
                     "rows": rows, "n_random": args.n_random,
                     "exact_metric": task_metric(task, y_exact, y),
                     "exact_mse": float(torch.nn.functional.mse_loss(y_exact, y)),
                     "gini_J": gini(J), "wall_s": round(time.time() - t0, 1)}

    probe.__exit__()
    mem = probe.as_dict()
    print(f"peak host memory: RssAnon {mem['peak_rss_anon_mb']:.0f} MB, "
          f"VmRSS {mem['peak_vm_rss_mb']:.0f} MB", flush=True)
    json.dump({"_memory": mem, "models": out},
              open(os.path.join(ROOT, "results", "phase2_equivalence.json"), "w"), indent=1)
    json.dump(scan_out, open(os.path.join(ROOT, "results", "phase2_scan_check.json"), "w"), indent=1)

    # ---- summary tables ---------------------------------------------------------------
    for name, r in out.items():
        print(f"\n--- {name}: relative L2 of the output trajectory (exact = 0) ---")
        hdr = "method".ljust(26) + "".join(f"{g:>9.3f}" for g in RHO_GRID)
        print(hdr)
        for meth, cells in r["rows"].items():
            print(meth.ljust(26) + "".join(f"{cells[str(g)]['rel_l2']:>9.2e}" for g in RHO_GRID))
        print(f"--- {name}: task metric ({r['task']}; exact = {r['exact_metric']:.4f}) ---")
        print(hdr)
        for meth, cells in r["rows"].items():
            print(meth.ljust(26) + "".join(f"{cells[str(g)]['metric']:>9.4f}" for g in RHO_GRID))
    bad = [k for k, v in scan_out.items() if not all(v[str(r)]["pass"] for r in (0.25, 0.5))]
    if bad:
        raise SystemExit(f"scan gate failed for {bad}")


if __name__ == "__main__":
    main()
