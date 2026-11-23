#!/usr/bin/env python3
"""Phase 1: exact isomorphism gate (rho = 1) + Jacobian / residual-spectrum audit.

Theorem 1 (bilateral permutation isomorphism).  For any permutation matrix P,
  W^pi = P W,  U^pi = P U P^T,  b^pi = P b,  h0^pi = P h0,  W_out^pi = W_out P^T
leaves y_t unchanged for every t.  We certify this in FP64 (< 1e-13 relative L2) and FP32
(< 1e-6), and show that a *one-sided* permutation (rows only, as a naive weight-reordering
pass would produce) breaks the recurrence badly (relative L2 > 0.50).

Outputs:
  results/phase1_isomorphism.json   gate verdicts + per-permutation errors
  results/phase1_spectrum.json      residual scores, R(rho), Gini/Pareto, Jacobian spectra
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models import CfC, permute, rel_l2, unpermute_params_one_sided  # noqa: E402
from src.probe import MemProbe  # noqa: E402
from src.ordering import (gini, order_by_score, order_coact, order_identity,  # noqa: E402
                          order_random, pareto_ratio, residual_scores, retained_curve,
                          step_jacobian)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT = os.path.join(ROOT, "results", "checkpoints")
RHO_GRID = [0.0, 0.0625, 0.125, 0.1875, 0.25, 0.375, 0.5, 0.75, 1.0]

GATE_FP64 = 1e-13
GATE_FP32 = 1e-6
GATE_ONESHOT = 0.50


def load(name, device):
    ck = torch.load(os.path.join(CKPT, f"{name}.pt"), weights_only=False)
    model = CfC(**ck["cfg"]).to(device)
    model.load_state_dict({k: v.to(device) for k, v in ck["state"].items()})
    ev = torch.load(os.path.join(CKPT, f"{name}_eval.pt"), weights_only=False)
    return ck, model, ev


def build_orders(model, x, dt, h, J, var):
    D = model.D
    wn = model.W_out.norm(dim=0)
    orders = {
        "identity": order_identity(D, x.device),
        "variance": order_by_score(var),
        "weight-norm": order_by_score(wn),
        "jacobian": order_by_score(J),
        "jacobian-coact": order_coact(J, h),
    }
    for s in range(5):
        orders[f"random{s}"] = order_random(D, seed=s, device=x.device)
    return orders


def iso_check(model, x, dt, orders, dtype):
    """Forward the exact model, then the bilaterally permuted model, in `dtype`."""
    m = model.to(dtype)
    xx, dtt = x.to(dtype), dt.to(dtype)
    h0 = torch.zeros(xx.shape[0], m.D, dtype=dtype, device=xx.device)
    with torch.no_grad():
        y_ref, _ = m.forward_exact(xx, dtt, h0=h0)
        res = {}
        for name, order in orders.items():
            mp = permute(m, order)
            y, _ = mp.forward_exact(xx, dtt, h0=h0)
            res[name] = {"rel_l2": rel_l2(y, y_ref)}
        # one-sided control: rows permuted, U columns and W_out columns untouched
        one = unpermute_params_one_sided(m, orders["jacobian"])
        y1, _ = one.forward_exact(xx, dtt, h0=h0)
        res["one-sided"] = {"rel_l2": rel_l2(y1, y_ref)}
    return res


def spectrum(model, x, dt, orders, h_all, J, n_jac=4):
    D = model.D
    xf = x[:, :-1].reshape(-1, x.shape[-1])
    dtf = dt[:, :-1].reshape(-1, 1)
    hprev = h_all[:, :-1].reshape(-1, D)
    hcur = h_all[:, 1:].reshape(-1, D)
    var = hcur.var(dim=0)
    L = hprev.shape[0]
    idx = torch.linspace(0, L - 1, n_jac).long()
    rad, rad_hot = [], {r: [] for r in RHO_GRID}
    for k_, i in enumerate(idx.tolist()):
        print(f"    jacobian sample {k_+1}/{len(idx)}", flush=True)
        Jm = step_jacobian(model, xf[i:i + 1], dtf[i:i + 1], hprev[i:i + 1])[0]
        ev = torch.linalg.eigvals(Jm.float().cpu())   # fp64 eig on the 8 GB laptop GPU
                                                      # is pathologically slow
        rad.append(float(ev.abs().max()))
        for r in RHO_GRID:
            k = int(round(r * D))
            if k == 0:
                rad_hot[r].append(0.0)
                continue
            hot = torch.topk(J, k).indices
            blk = Jm[hot][:, hot]
            rad_hot[r].append(float(torch.linalg.eigvals(blk.float().cpu()).abs().max()))
    return {
        "spectral_radius_mean": sum(rad) / len(rad),
        "spectral_radius_max": max(rad),
        "spectral_radius_hot_by_rho": {str(r): sum(v) / len(v) for r, v in rad_hot.items()},
        "n_jacobian_samples": len(rad),
        "variance_score": var.detach().cpu(),  # returned separately, stripped before dump
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=["A_oscillator", "B_parity", "C_largescale"])
    ap.add_argument("--T-stats", type=int, default=256, help="timesteps used for the spectrum audit")
    ap.add_argument("--T-fp64", type=int, default=96, help="timesteps used for the FP64 gate")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    dev = args.device
    probe = MemProbe().__enter__()

    iso_out, spec_out = {}, {}
    for name in args.models:
        t0 = time.time()
        ck, model, ev = load(name, dev)
        x, dt = ev["x"].to(dev), ev["dt"].to(dev)
        T = x.shape[1]
        print(f"\n=== {name}  (D={model.D}, T={T}) ===", flush=True)

        with torch.no_grad():
            _, h_all = model.forward_exact(x, dt, keep_h=True)
        hs = h_all[:, : args.T_stats]
        xs, dts = x[:, : args.T_stats], dt[:, : args.T_stats]
        hbar = hs.reshape(-1, model.D).mean(0)
        sc = residual_scores(model, xs, dts, hbar=hbar)
        J, resid, wn, amean = sc["J"], sc["resid"], sc["wnorm"], sc["alpha_mean"]
        var = hs[:, 1:].reshape(-1, model.D).var(0)
        orders = build_orders(model, xs, dts, hs, J, var)
        print("  orderings built: " + ", ".join(orders), flush=True)

        # ---- spectrum audit -------------------------------------------------------
        sp = spectrum(model, xs, dts, orders, hs, J)
        var_cpu = sp.pop("variance_score")
        sp.update({
            "residual_J": J.detach().cpu(),
            "residual_step": resid.detach().cpu(),
            "wnorm": wn.detach().cpu(),
            "alpha_mean": amean.detach().cpu(),
            "variance": var_cpu,
            "R_of_rho": retained_curve(J, RHO_GRID), "rho_grid": RHO_GRID,
            "gini_J": gini(J), "gini_variance": gini(var), "gini_wnorm": gini(wn),
            "pareto_top10pct_J": pareto_ratio(J, 0.10),
            "pareto_top25pct_J": pareto_ratio(J, 0.25),
            "pearson_log_J_vs_log_var": float(torch.corrcoef(torch.stack(
                [J.clamp_min(1e-30).log(), var.clamp_min(1e-30).log()]))[0, 1]),
            "pearson_log_J_vs_log_wn": float(torch.corrcoef(torch.stack(
                [J.clamp_min(1e-30).log(), wn.clamp_min(1e-30).log()]))[0, 1]),
            "top_hot_share_J": {
                str(r): float(torch.topk(J, max(1, int(round(r * model.D)))).values.sum()
                              / J.sum()) for r in [0.25, 0.5]},
        })
        print(f"  J: gini {sp['gini_J']:.3f}  top25% share {sp['pareto_top25pct_J']:.3f}  "
              f"R(0.25)={sp['R_of_rho'][4]:.3f}  R(0.5)={sp['R_of_rho'][6]:.3f}", flush=True)
        print(f"  spectral radius mean {sp['spectral_radius_mean']:.3f} "
              f"max {sp['spectral_radius_max']:.3f}", flush=True)

        # ---- isomorphism gates ----------------------------------------------------
        xa, dta = xs, dts
        res32 = iso_check(model, xa, dta, orders, torch.float32)
        x64, dt64 = x[:, : args.T_fp64], dt[:, : args.T_fp64]
        res64 = iso_check(model, x64, dt64, orders, torch.float64)
        worst32 = max(v["rel_l2"] for k, v in res32.items() if k != "one-sided")
        worst64 = max(v["rel_l2"] for k, v in res64.items() if k != "one-sided")
        ok = (worst64 < GATE_FP64) and (worst32 < GATE_FP32) \
            and res64["one-sided"]["rel_l2"] > GATE_ONESHOT
        iso_out[name] = {"D": model.D, "T_fp32": int(xa.shape[1]), "T_fp64": int(x64.shape[1]),
                         "fp32": res32, "fp64": res64,
                         "worst_bilateral_fp32": worst32, "worst_bilateral_fp64": worst64,
                         "gate_fp64": GATE_FP64, "gate_fp32": GATE_FP32,
                         "gate_one_sided_min": GATE_ONESHOT,
                         "one_sided_fp64": res64["one-sided"]["rel_l2"],
                         "pass": bool(ok), "wall_s": round(time.time() - t0, 1)}
        print(f"  rho=1 gates: fp64 worst {worst64:.2e} (<{GATE_FP64:.0e})  "
              f"fp32 worst {worst32:.2e} (<{GATE_FP32:.0e})  "
              f"one-sided {res64['one-sided']['rel_l2']:.3f} -> "
              f"{'PASS' if ok else 'FAIL'}", flush=True)
        spec_out[name] = sp
        model.to(torch.float32)

    probe.__exit__()
    mem = probe.as_dict()
    print(f"peak host memory: RssAnon {mem['peak_rss_anon_mb']:.0f} MB, "
          f"VmRSS {mem['peak_vm_rss_mb']:.0f} MB", flush=True)
    json.dump({**iso_out, "_memory": mem},
              open(os.path.join(ROOT, "results", "phase1_isomorphism.json"), "w"), indent=1)
    torch.save(spec_out, os.path.join(ROOT, "results", "phase1_spectrum.pt"))
    slim = {k: {kk: (vv.tolist() if torch.is_tensor(vv) else vv)
                for kk, vv in v.items() if kk not in ("residual_J", "residual_step", "wnorm",
                                                      "alpha_mean", "variance")}
            for k, v in spec_out.items()}
    json.dump(slim, open(os.path.join(ROOT, "results", "phase1_spectrum.json"), "w"), indent=1)

    print("\n=== Phase-1 gate summary ===")
    print(f"{'model':<16}{'fp64 rel L2':>13}{'fp32 rel L2':>13}{'one-sided':>11}  gate")
    for k, v in iso_out.items():
        print(f"{k:<16}{v['worst_bilateral_fp64']:>13.2e}{v['worst_bilateral_fp32']:>13.2e}"
              f"{v['one_sided_fp64']:>11.2f}  {'PASS' if v['pass'] else 'FAIL'}")
    if not all(v["pass"] for v in iso_out.values()):
        raise SystemExit("ISOMORPHISM GATE FAILED")


if __name__ == "__main__":
    main()
