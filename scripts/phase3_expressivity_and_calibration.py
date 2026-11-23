#!/usr/bin/env python3
"""Phase 3: spectral amplification law, training-free readout recalibration, and
non-TC0 expressivity preservation under decoupling (Pillar 3).

Three experiments, each fully measured and logged to results/phase3_expressivity.json:

  A  amplification law   eps_traj(rho,T) ~ eps_step(rho) * (1 - abar^T)/(1 - abar)
  B  ridge readout       closed-form (W_out, b_out) recalibration on N_cal = 128 sequences,
                         0 backward passes, timed (target < 50 ms), evaluated on a held-out split
  C  expressivity        Parity-Stream / Modulo-3 Counter / FlipFlop-Reset at D = 256:
                         pure-linear (rho = 0) vs pi-CfC at rho in {1/16, 1/8, 1/4} for three
                         channel orders, zero-shot and after 5 epochs of decoupling-aware
                         fine-tuning (DAFT); plus DAFT recovery of A_oscillator.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import src.neurtasks as NT  # noqa: E402
from src.models import CfC, rel_l2  # noqa: E402
from src.ordering import (order_by_score, order_identity, order_random,  # noqa: E402
                          residual_scores)
from src.probe import MemProbe  # noqa: E402
from src.tasks import TASKS, task_metric  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT = os.path.join(ROOT, "results", "checkpoints")
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figures")

RHO_READ = [0.0, 0.0625, 0.125, 0.25, 0.375, 0.5]
RHO_AMP = [0.0, 0.0625, 0.125, 0.25, 0.5]
HORIZONS = [16, 32, 64, 128, 256, 512, 1024]
RHO_EXPR = [0.0625, 0.125, 0.25, 0.5]
LAMBDA_RIDGE = 1e-3
N_CAL = 128
TASKMAP = {"A_oscillator": "oscillator", "B_parity": "parity", "C_largescale": "multiscale"}


# --------------------------------------------------------------------------------------
# model / data plumbing
# --------------------------------------------------------------------------------------
def load(name, dev):
    ck = torch.load(os.path.join(CKPT, f"{name}.pt"), weights_only=False)
    cfg = ck["cfg"]
    model = CfC(cfg["d_in"], cfg["D"], cfg["d_out"]).to(dev)
    model.load_state_dict(ck["state"])
    model.eval()
    ev = torch.load(os.path.join(CKPT, f"{name}_eval.pt"), weights_only=False)
    x = ev["x"].to(dev)
    dt = ev["dt"].to(dev)
    y = ev["y"].to(dev)
    return ck, model, x, dt, y


def mean_state(model, x, dt):
    with torch.no_grad():
        _, H = model.forward_exact(x, dt, keep_h=True)
    return H.reshape(-1, model.D).mean(0)


def orders_for(model, x, dt, hbar, dev, n_random=3):
    sc = residual_scores(model, x, dt, hbar=hbar)
    with torch.no_grad():
        _, H = model.forward_exact(x, dt, keep_h=True)
    sc["variance"] = H.var(dim=(0, 1))
    out = {"native-prefix": order_identity(model.D, dev),
           "ours-jacobian-pi": order_by_score(sc["J"]),
           "variance-pi": order_by_score(sc["variance"])}
    rand = [order_random(model.D, seed=1000 + s, device=dev) for s in range(n_random)]
    out["random-pi-list"] = rand
    out["random-pi"] = rand[0]
    out["scores"] = sc
    return out


def run_method(model, x, dt, rho, order, picard=0, hbar=None, keep_h=True):
    """One forward in the requested execution mode (rho = 0 -> pure linear scan)."""
    if rho >= 1.0:
        return model.forward_exact(x, dt, keep_h=keep_h)
    return model.forward_approx(x, dt, hbar=hbar, order=order, rho=rho, picard=picard,
                                keep_h=keep_h)


# --------------------------------------------------------------------------------------
# A. amplification law
# --------------------------------------------------------------------------------------
@torch.no_grad()
def amplification(model, x, dt, hbar, order, rho_list=RHO_AMP, horizons=HORIZONS):
    """One-step residual vs multi-step drift.

    Two residual definitions are reported:

    * ``eps_step_exact`` -- linearised step evaluated at the *exact* state h_{t-1}: how good
      the fixed linearisation point hbar is as a model of the gates.
    * ``eps_step_local`` -- linearised step evaluated along the *approximate* trajectory
      (the state the decoupled rollout actually visits).  This is the injection term of the
      error recursion e_t = a_t e_{t-1} + eta_t, so it is the quantity the analytic law
      eps_traj ~ eps_step (1 - abar^T)/(1 - abar) refers to.

    The law is additionally capped at the decorrelation limit eps_traj <= 1.
    """
    D = model.D
    B, T, _ = x.shape
    h0 = torch.zeros(B, D, device=x.device, dtype=x.dtype)
    _, H_exact = model.forward_exact(x, dt, keep_h=True)
    xf = x @ model.W_f.t() + model.b_f
    xg = x @ model.W_g.t() + model.b_g
    xu = x @ model.W_u.t() + model.b_u

    def gates_at(h_prev):
        f = F.softplus(xf + h_prev @ model.U_f.t())
        g = torch.tanh(xg + h_prev @ model.U_g.t())
        u = torch.sigmoid(xu + h_prev @ model.U_u.t())
        a = torch.exp(-f * dt)
        return a, (1 - a) * g * u

    rows = []
    for rho in rho_list:
        if rho >= 1.0:
            continue
        n_cold = int(round((1.0 - rho) * D))
        cold = order[:n_cold] if n_cold else order[:0]
        _, H_ap = run_method(model, x, dt, rho, order, hbar=hbar, keep_h=True)
        if n_cold == 0:
            rows.append({"rho": rho, "n_cold": 0, "eps_step_exact": 0.0, "eps_step_local": 0.0,
                         "abar": 0.0, "G_inf": 1.0, "horizons": [
                             {"T": Th, "eps_traj": 0.0, "eps_traj_pred": 0.0, "T_sat": None}
                             for Th in horizons if Th <= T]})
            continue

        # exact gates along both trajectories, linearised gates from hbar
        h_prev_e = torch.cat([h0.unsqueeze(1), H_exact[:, :-1]], 1)
        h_prev_a = torch.cat([h0.unsqueeze(1), H_ap[:, :-1]], 1)
        a_e, b_e = gates_at(h_prev_e)
        a_a, b_a = gates_at(h_prev_a)
        fl = F.softplus(x @ model.W_f[cold].t() + (model.U_f[cold] @ hbar) + model.b_f[cold])
        gl = torch.tanh(x @ model.W_g[cold].t() + (model.U_g[cold] @ hbar) + model.b_g[cold])
        ul = torch.sigmoid(x @ model.W_u[cold].t() + (model.U_u[cold] @ hbar) + model.b_u[cold])
        al = torch.exp(-fl * dt)
        bl = (1.0 - al) * gl * ul
        scale = H_exact.norm(dim=-1).mean().clamp_min(1e-12)

        def resid(h_prev, a_t, b_t):
            pred_l = al * h_prev[:, :, cold] + bl
            pred_t = a_t[:, :, cold] * h_prev[:, :, cold] + b_t[:, :, cold]
            return float((pred_l - pred_t).norm(dim=-1).mean() / scale)

        eps_exact = resid(h_prev_e, a_e, b_e)
        eps_local = resid(h_prev_a, a_a, b_a)
        abar = float(al.mean())
        G_inf = 1.0 / max(1.0 - abar, 1e-6)
        traj = []
        for Th in horizons:
            if Th > T:
                continue
            e = float((H_ap[:, :Th] - H_exact[:, :Th]).norm() / H_exact[:, :Th].norm())
            gain_lin = (1.0 - abar ** Th) / (1.0 - abar)            # coherent summation
            gain_quad = ((1.0 - (abar ** 2) ** Th) / (1.0 - abar ** 2)) ** 0.5  # incoherent
            t_sat = (float(torch.log(torch.tensor(max(eps_local, 1e-12)))
                           / torch.log(torch.tensor(max(abar, 1e-6))))) if abar < 1 else None
            lin = eps_local * gain_lin
            quad = min(eps_local * gain_quad, 1.0)
            traj.append({"T": Th, "eps_traj": e, "eps_step_local": eps_local,
                         "eps_step_exact": eps_exact, "gain_analytic": gain_lin,
                         "eps_traj_pred_linear": lin,
                         "eps_traj_pred_quad": quad,
                         "ratio_measured": e / max(eps_local, 1e-12),
                         "in_bounds": bool(quad <= e + 1e-12 and e <= lin + 1e-12),
                         "log_ratio_to_linear": (float(torch.log(torch.tensor(e / max(lin, 1e-12))))),
                         "T_sat": t_sat})
        rows.append({"rho": rho, "n_cold": n_cold, "eps_step_exact": eps_exact,
                     "eps_step_local": eps_local, "abar": abar, "G_inf": G_inf,
                     "horizons": traj})
    return rows


# --------------------------------------------------------------------------------------
# B. training-free closed-form readout recalibration
# --------------------------------------------------------------------------------------
class RidgeAccumulator:
    """Streaming closed-form ridge regression over trajectory rows.

    Only `A = sum H_a^T H_a` (D+1)^2 and `rhs = sum H_a^T Y` are kept, so the calibration
    set can be much larger than GPU memory would allow to materialise.  The solve is done in
    FP64 on the small (D+1) system; the ridge strength is scaled by the mean diagonal of the
    Gram matrix, which makes it invariant to the (arbitrary) scale of `h`.
    """

    def __init__(self, D, d_out, device, lam=LAMBDA_RIDGE):
        self.D, self.d_out, self.dev, self.lam = D, d_out, device, lam
        self.A = torch.zeros(D + 1, D + 1, device=device)
        self.rhs = torch.zeros(D + 1, d_out, device=device)
        self.n = 0
        self.t_regress = 0.0

    def add(self, H, Y):
        t0 = time.perf_counter()
        Hf = H.reshape(-1, self.D).float()
        Ha = torch.cat([Hf, torch.ones(Hf.shape[0], 1, device=Hf.device)], 1)
        self.A += Ha.t() @ Ha
        self.rhs += Ha.t() @ Y.reshape(-1, self.d_out).float()
        self.n += Hf.shape[0]
        self.t_regress += time.perf_counter() - t0

    def solve(self):
        t0 = time.perf_counter()
        A = self.A.double()
        scale = float(A.diagonal()[:self.D].mean())
        A.diagonal()[:self.D].add_(self.lam * max(scale, 1e-12))
        sol = torch.linalg.solve(A, self.rhs.double())
        self.t_regress += time.perf_counter() - t0
        return sol[:-1].t().float().contiguous(), sol[-1].float().contiguous()


@torch.no_grad()
def ridge_study(model, cal_run, cal_targets, x_ev, dt_ev, y_ev, task, dev, rho_list=RHO_READ):
    """Training-free recalibration of (W_out, b_out) on an independent calibration split."""
    out = {}
    d_out = y_ev.shape[-1]
    for meth in ("pure-linear", "native-prefix", "random-pi", "variance-pi",
                 "ours-jacobian-pi", "ours-jacobian-picard1"):
        for rho in rho_list:
            if meth == "pure-linear" and rho != 0.0:
                continue
            r, pic = (0.0, 0) if meth == "pure-linear" else (rho, int(meth.endswith("picard1")))
            order = cal_run["order"](meth)
            acc = RidgeAccumulator(model.D, d_out, dev)
            for xb, dtb, yb in cal_run["batches"](r, order, pic):
                _, H = run_method(model, xb, dtb, r, order, picard=pic, hbar=cal_run["hbar"],
                                  keep_h=True)
                acc.add(H, yb)
            W, b = acc.solve()
            _, He = run_method(model, x_ev, dt_ev, r, order, picard=pic, hbar=cal_run["hbar"],
                               keep_h=True)
            He_flat = He.reshape(-1, model.D).float()
            Ye = y_ev.reshape(-1, d_out).float()
            m_new = task_metric(task, He_flat @ W.t() + b, Ye)
            m_old = task_metric(task, He_flat @ model.W_out.t().float() + model.b_out.float(), Ye)
            out[f"{meth}|{rho}"] = {"rho": r, "picard": pic, "metric_default": m_old,
                                    "metric_ridge": m_new, "delta": m_new - m_old,
                                    "ridge_ms": acc.t_regress * 1e3, "n_rows": acc.n,
                                    "ridge_rel_change": float((W - model.W_out.float()).norm()
                                                              / model.W_out.float().norm())}
        print(f"  ridge @{meth:24s}: " + " ".join(
            f"{out[k]['metric_default']:.3f}->{out[k]['metric_ridge']:.3f}"
            for k in out if k.startswith(meth + "|")), flush=True)
    return out


# --------------------------------------------------------------------------------------
# C. non-TC0 expressivity
# --------------------------------------------------------------------------------------
def train_full(task, dev, D=256, T=128, steps=3000, B=64, lr=3e-3, seed=0):
    torch.manual_seed(seed)
    NT.DT = 0.05
    m = CfC(1, D, NT.SUITE[task]["d_out"]).to(dev)
    with torch.no_grad():
        m.b_f.fill_(2.0)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    X, Dt, S = NT.make_batch(task, 1024, T, dev, seed=3)
    t0 = time.time()
    for step in range(steps):
        idx = torch.randint(0, X.shape[0], (B,), device=dev)
        y, _ = m.forward_exact(X[idx], Dt[idx])
        loss = F.cross_entropy(y.reshape(-1, y.shape[-1]), S[idx].reshape(-1))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        sch.step()
    m.eval()
    return m, (X, Dt, S), time.time() - t0, loss.item()


def hot_mask_grads(model, hot):
    for nm in ("W_f", "W_g", "W_u", "b_f", "b_g", "b_u"):
        g = getattr(model, nm).grad
        if g is not None:
            if nm.startswith("b"):
                g[~hot] = 0
            else:
                g[~hot] = 0
    for nm in ("U_f", "U_g", "U_u"):
        g = getattr(model, nm).grad
        if g is not None:
            g[~hot, :] = 0
            g[:, ~hot] = 0


def daft(model, data, task, rho, order, dev, epochs=5, steps_per_epoch=60, B=64, lr=1e-3):
    """5 epochs of decoupling-aware fine-tuning: hot rows + readout only, cold path frozen."""
    X, Dt, S = data
    D = model.D
    n_cold = int(round((1 - rho) * D))
    hot = torch.zeros(D, dtype=torch.bool, device=dev)
    hot[order[n_cold:]] = True
    hbar = mean_state(model, X, Dt)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    steps = epochs * steps_per_epoch
    t0 = time.time()
    for _ in range(steps):
        idx = torch.randint(0, X.shape[0], (B,), device=dev)
        y, _ = run_method(model, X[idx], Dt[idx], rho, order, hbar=hbar, keep_h=False)
        loss = F.cross_entropy(y.reshape(-1, y.shape[-1]), S[idx].reshape(-1))
        opt.zero_grad()
        loss.backward()
        hot_mask_grads(model, hot)
        opt.step()
    model.eval()
    return loss.item(), time.time() - t0


def expr_study(dev, out):
    res = {"tasks": {}, "blocked_initial_state": (
        "Parity/Mod-3 with an *unobserved random* initial state is unlearnable (accuracy "
        "saturates at 1/nclasses for every architecture): the target is only determined up "
        "to a global bit/offset.  The suite therefore uses the standard protocol, a known "
        "zero initial state.")}
    for task in ("parity_stream", "mod3_counter", "flipflop_reset"):
        m, data, t_train, loss = train_full(task, dev)
        X, Dt, S = data
        xe, dte, se = NT.make_batch(task, 256, 256, dev, seed=99)
        acc_full, bal_full = NT.accuracy(m, task, xe, dte, se)
        hbar = mean_state(m, X, Dt)
        ords = orders_for(m, xe, dte, hbar, dev)
        cell = {"label": NT.SUITE[task]["label"], "train_s": round(t_train, 1),
                "final_train_loss": loss, "full_cfc": {"acc": acc_full, "bal": bal_full},
                "rho": {}, "pure_linear": {}}

        # pure linear (rho = 0) -- before and after the same DAFT budget
        acc0, bal0 = NT.accuracy(m, task, xe, dte, se, rho=0.0, order=ords["native-prefix"],
                                 hbar=hbar)
        m0 = _clone(m)
        loss0, dt0 = daft(m0, data, task, 0.0, ords["native-prefix"], dev)
        acc0d, bal0d = NT.accuracy(m0, task, xe, dte, se, rho=0.0,
                                   order=ords["native-prefix"], hbar=hbar)
        cell["pure_linear"] = {"acc_zero_shot": acc0, "bal_zero_shot": bal0,
                               "acc_daft": acc0d, "bal_daft": bal0d,
                               "daft_loss": loss0, "daft_s": round(dt0, 1)}
        print(f"{task}: full {acc_full:.3f} | pure-linear {acc0:.3f} -> daft {acc0d:.3f}",
              flush=True)

        for rho in RHO_EXPR:
            cell["rho"][str(rho)] = {}
            for meth in ("ours-jacobian-pi", "variance-pi", "random-pi"):
                order = ords[meth]
                acc, bal = NT.accuracy(m, task, xe, dte, se, rho=rho, order=order, hbar=hbar)
                mm = _clone(m)
                loss_d, dt_d = daft(mm, data, task, rho, order, dev)
                acc_d, bal_d = NT.accuracy(mm, task, xe, dte, se, rho=rho, order=order, hbar=hbar)
                cell["rho"][str(rho)][meth] = {
                    "acc_zero_shot": acc, "bal_zero_shot": bal, "acc_daft": acc_d,
                    "bal_daft": bal_d, "daft_loss": loss_d, "daft_s": round(dt_d, 1)}
                print(f"  rho={rho:.3f} {meth:18s} {acc:.3f} -> {acc_d:.3f} "
                      f"(bal {bal:.3f} -> {bal_d:.3f})", flush=True)
        res["tasks"][task] = cell
        del m, m0
        torch.cuda.empty_cache(); gc.collect()
    return res


def _clone(m):
    c = CfC(m.d_in, m.D, m.d_out).to(next(m.parameters()).device)
    c.load_state_dict(m.state_dict())
    c.train()
    return c


def oscillator_daft(dev, out):
    ck, model, x, dt, y = load("A_oscillator", dev)
    hbar = mean_state(model, x, dt)
    ords = orders_for(model, x, dt, hbar, dev)
    # fine-tune on FRESH sequences from the task generator: fitting on the evaluation
    # batch would be training on the test set
    T_ev = int(x.shape[1])
    xf, dtf, yf = TASKS["oscillator"][0](16, T_ev, dev)
    xf, dtf, yf = xf.to(dev), dtf.to(dev), yf.to(dev)
    res = {}
    for rho in (0.0, 0.125, 0.25):
        res[str(rho)] = {}
        for meth in ("ours-jacobian-pi", "variance-pi"):
            if rho == 0.0 and meth != "ours-jacobian-pi":
                continue
            order = ords[meth]
            y0, _ = run_method(model, x, dt, rho, order, hbar=hbar, keep_h=False)
            r2_0 = task_metric("oscillator", y0, y)
            mm = _clone(model)
            hot = torch.zeros(mm.D, dtype=torch.bool, device=dev)
            n_cold = int(round((1 - rho) * mm.D))
            hot[order[n_cold:]] = True
            opt = torch.optim.Adam(mm.parameters(), lr=5e-4)
            steps = 5 * 60
            t0 = time.time()
            for _ in range(steps):
                yp, _ = run_method(mm, xf, dtf, rho, order, hbar=hbar, keep_h=False)
                loss = F.mse_loss(yp, yf)
                opt.zero_grad()
                loss.backward()
                hot_mask_grads(mm, hot)
                opt.step()
            y1, _ = run_method(mm, x, dt, rho, order, hbar=hbar, keep_h=False)
            res[str(rho)][meth] = {"r2_before": r2_0, "r2_after": task_metric("oscillator", y1, y),
                                   "daft_s": round(time.time() - t0, 1),
                                   "final_loss": float(loss.detach())}
            print(f"A_oscillator rho={rho} {meth}: R2 {r2_0:+.3f} -> "
                  f"{res[str(rho)][meth]['r2_after']:+.3f}", flush=True)
            del mm
    torch.cuda.empty_cache(); gc.collect()
    return res


# --------------------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------------------
def figures(amp, ridge, expr, osc):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.25))

    ax = axes[0]
    mcol = {"A_oscillator": "#1f77b4", "B_parity": "#2ca02c", "C_largescale": "#ff7f0e"}
    mlab = {"A_oscillator": "A (oscillator)", "B_parity": "B (parity)",
            "C_largescale": "C (large-scale)"}
    for name, rows in amp.items():
        first = True
        for r in rows:
            if r["rho"] not in (0.25, 0.5):
                continue
            xs = [h["T"] for h in r["horizons"]]
            col = mcol.get(name, "#1f77b4")
            ax.plot(xs, [h["eps_traj"] for h in r["horizons"]], marker="o", ms=3, lw=1.4,
                    ls="-" if r["rho"] == 0.5 else "--", color=col,
                    label=mlab.get(name, name) if first else None)
            ax.plot(xs, [h["eps_traj_pred_linear"] for h in r["horizons"]],
                    ls=":", lw=0.9, color="#d62728", alpha=0.8)
            first = False
    ax.plot([], [], ls=":", lw=1.0, color="#d62728", label="law: $\\epsilon_{step}G_\\infty$")
    ax.plot([], [], ls="-", lw=1.2, color="0.35", label=r"$\rho=1/2$")
    ax.plot([], [], ls="--", lw=1.2, color="0.35", label=r"$\rho=1/4$")
    ax.set_ylim(0.06, 4.2)
    ax.legend(frameon=False, fontsize=5.4, ncol=2, loc="upper left")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("horizon $T$")
    ax.set_ylabel(r"$\epsilon_{\rm traj}$ (measured) and law")
    ax.set_title("(a) amplification law", fontsize=9)
    ax.grid(alpha=0.25, which="both")

    ax = axes[1]
    styles = {"pure-linear": ("0.5", ":"), "native-prefix": ("#1f77b4", "--"),
              "random-pi": ("#7f7f7f", "-."), "variance-pi": ("#2ca02c", "--"),
              "ours-jacobian-pi": ("#d62728", "-"), "ours-jacobian-picard1": ("#9467bd", "-")}
    for name, block in ridge.items():
        for meth, (c, ls) in styles.items():
            keys = [k for k in block if k.startswith(meth + "|")]
            keys.sort(key=lambda k: block[k]["rho"])
            xs = [block[k]["rho"] for k in keys]
            gain = [block[k]["metric_ridge"] - block[k]["metric_default"] for k in keys]
            ax.plot(xs, gain, color=c, ls=ls, marker="o", ms=2.5, lw=1.3,
                    label=f"{name.split('_')[0]} {meth}")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel(r"hot fraction $\rho$")
    ax.set_ylabel("task-metric gain from ridge readout")
    ax.set_title("(b) training-free recalibration", fontsize=9)
    ax.grid(alpha=0.25)

    ax = axes[2]
    tasks = list(expr["tasks"].items())
    labels = [v["label"] for _, v in tasks]
    width = 0.26
    for i, (meth, color) in enumerate((("variance-pi", "#2ca02c"),
                                       ("ours-jacobian-pi", "#d62728"),
                                       ("random-pi", "#7f7f7f"))):
        vals = [v["rho"]["0.125"][meth]["acc_daft"] for _, v in tasks]
        ax.bar([j + (i - 1) * width for j in range(len(tasks))], vals, width,
               color=color, label=meth)
    ax.bar([j + width for j in range(len(tasks))],
           [v["pure_linear"]["acc_daft"] for _, v in tasks], width, color="0.35",
           label="pure-linear (DAFT)")
    ax.axhline(1.0, color="k", lw=0.8, ls=":")
    ax.set_xticks(range(len(tasks)))
    ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_ylabel("held-out state accuracy")
    ax.set_ylim(0, 1.52)
    ax.set_title(r"(c) non-$TC^0$ tasks, $\rho=1/8$ after DAFT", fontsize=9)
    ax.legend(frameon=False, fontsize=5.6, ncol=2, loc="upper center")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_expressivity_and_calibration.pdf"))
    fig.savefig(os.path.join(FIG, "fig_expressivity_and_calibration.png"), dpi=160)
    plt.close(fig)


# --------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--parts", nargs="*", default=["amp", "ridge", "expr", "osc"])
    args = ap.parse_args()
    dev = args.device
    torch.manual_seed(0)
    probe = MemProbe().__enter__()
    out = {"rho_read": RHO_READ, "lambda_ridge": LAMBDA_RIDGE, "n_cal": N_CAL}
    _prev = os.path.join(RES, "phase3_expressivity.json")
    if os.path.exists(_prev):                      # merge: keep parts measured earlier
        try:
            old = json.load(open(_prev))
            out.update({k: v for k, v in old.items() if not k.startswith("_")})
        except Exception:
            pass

    if "amp" in args.parts or "ridge" in args.parts:
        amp, ridge = {}, {}
        for name in ("A_oscillator", "B_parity", "C_largescale"):
            ck, model, x, dt, y = load(name, dev)
            hbar = mean_state(model, x, dt)
            ords = orders_for(model, x, dt, hbar, dev)
            task = TASKMAP[name]
            if "amp" in args.parts:
                t0 = time.time()
                amp[name] = amplification(model, x, dt, hbar, ords["ours-jacobian-pi"])
                print(f"{name}: amplification law ({time.time() - t0:.0f}s)", flush=True)
                for r in amp[name]:
                    h = r["horizons"][-1]
                    print(f"  rho={r['rho']:.3f} n_cold={r['n_cold']:4d} "
                          f"eps_step local/exact={r['eps_step_local']:.2e}/{r['eps_step_exact']:.2e} "
                          f"abar={r['abar']:.3f} G_inf={r['G_inf']:.1f} "
                          f"eps_traj(T={h['T']})={h['eps_traj']:.3f} "
                          f"bounds=[{h['eps_traj_pred_quad']:.3f},{h['eps_traj_pred_linear']:.3f}] "
                          f"{'in' if h['in_bounds'] else 'OUT'}",
                          flush=True)
            if "ridge" in args.parts:
                # independent calibration split: freshly generated sequences from the task
                # generator (never the evaluation batch -- sharing it leaks the trajectory)
                T_ev = int(x.shape[1])
                cal_x, cal_dt, cal_y = TASKS[task][0](min(N_CAL, 128), T_ev, dev)
                cal_x, cal_dt, cal_y = cal_x.to(dev), cal_dt.to(dev), cal_y.to(dev)
                bundle = {"hbar": hbar,
                          "order": lambda m: ords["random-pi"] if m == "random-pi" else
                                             ords.get(m, ords["ours-jacobian-pi"]),
                          "batches": (lambda r, o, p, xs=cal_x, ds=cal_dt, ys=cal_y:
                                      [(xs[i:i + 16], ds[i:i + 16], ys[i:i + 16])
                                       for i in range(0, xs.shape[0], 16)])}
                t0 = time.time()
                ridge[name] = ridge_study(model, bundle, None, x, dt, y, task, dev)
                print(f"{name}: ridge study ({time.time() - t0:.0f}s)", flush=True)
        if amp:
            out["amplification"] = amp
        if ridge:
            out["ridge"] = ridge
            ms = [v["ridge_ms"] for b in ridge.values() for v in b.values()]
            out["ridge_ms"] = {"max": max(ms), "mean": sum(ms) / len(ms), "budget": 50.0}

    if "expr" in args.parts:
        out["expressivity"] = expr_study(dev, out)
    if "osc" in args.parts:
        out["oscillator_daft"] = oscillator_daft(dev, out)

    probe.__exit__()
    out["_memory"] = probe.as_dict()
    json.dump(out, open(os.path.join(RES, "phase3_expressivity.json"), "w"), indent=1)
    if "amp" in out and "ridge" in out and "expressivity" in out:
        figures(out["amplification"], out["ridge"], out["expressivity"],
                out.get("oscillator_daft", {}))
        print("wrote figures/fig_expressivity_and_calibration.{pdf,png}")
    print("memory:", out["_memory"], "| results -> results/phase3_expressivity.json")


if __name__ == "__main__":
    main()
