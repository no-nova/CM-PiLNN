#!/usr/bin/env python3
"""Phase 1a: train three real CfC checkpoints on irregular-time tasks.

Models (as specified for CM-PiLNN):
  A  CfC-Oscillator     D=256,  L=512,  coupled Duffing + Van der Pol + linear modes
  B  CfC-ParityAutomaton D=256, L=512,  cumulative event-parity + autoregressive filter
  C  CfC-LargeScale     D=1024, L=1024, multi-regime continuous-time system

The learned dynamical spectrum is what the later phases audit, so the checkpoints are
trained to convergence on their tasks (loss curves are logged, not just the final value).
A separately seeded eval batch is saved next to each checkpoint so every downstream script
audits exactly the same trajectories.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models import CfC  # noqa: E402
from src.tasks import TASKS, make_batch, task_metric  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT = os.path.join(ROOT, "results", "checkpoints")

SPECS = {
    # T is the *evaluated* trajectory length; training uses short chunks (T_train) because a
    # 512-step BPTT dilutes the per-step gradient by ~8x and stalls at a mean predictor
    # (measured: 0.72 with T=512 vs 0.24 with T=64 at equal wall time).
    "A_oscillator": dict(task="oscillator", D=256, T=512, T_train=192, B=24, steps=1200, lr=8e-3),
    "B_parity": dict(task="parity", D=256, T=512, T_train=64, B=32, steps=1200, lr=5e-3, bf=0.5),
    "C_largescale": dict(task="multiscale", D=1024, T=1024, T_train=192, B=12, steps=300, lr=5e-3),
}


def eval_batch(task, T, device, seed=1234):
    g = torch.Generator().manual_seed(seed)
    return make_batch(task, 8, T, device, gen=g)


def train_one(name, spec, device):
    torch.manual_seed(hash(name) % (2 ** 31))
    task, D, T, B, steps = spec["task"], spec["D"], spec["T"], spec["B"], spec["steps"]
    T_train = spec["T_train"]
    _, d_in, d_out = TASKS[task]
    cfg = {"d_in": d_in, "D": D, "d_out": d_out}
    model = CfC(**cfg).to(device)
    if spec.get("bf") is not None:                     # per-task forget-bias init: a slow
        with torch.no_grad():                          # initial time constant helps tasks with
            model.b_f.fill_(spec["bf"])                # long memory (parity), hurts none
    opt = torch.optim.Adam(model.parameters(), lr=spec["lr"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=spec["lr"] / 10)
    gen = torch.Generator().manual_seed(abs(hash(name)) % (2 ** 31))
    curve, t0 = [], time.time()
    for it in range(steps):
        x, dt, y = make_batch(task, B, T_train, device, gen=gen)
        pred, _ = model.forward_exact(x, dt)
        loss = torch.nn.functional.mse_loss(pred, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if it % 50 == 0 or it == steps - 1:
            curve.append({"step": it, "loss": float(loss.item()),
                          "elapsed_s": round(time.time() - t0, 1)})
            print(f"  [{name}] step {it:4d}  mse {loss.item():.5f}  "
                  f"({time.time() - t0:.0f}s)", flush=True)
    xe, dte, ye = eval_batch(task, T, device)
    with torch.no_grad():
        pred, _ = model.forward_exact(xe, dte)
        ev_loss = float(torch.nn.functional.mse_loss(pred, ye).item())
        ev_metric = task_metric(task, pred, ye)
        Tlen = pred.shape[1]
        horizon = []
        for a, b in [(0, 32), (32, 64), (64, 128), (128, 256), (256, Tlen)]:
            if a >= Tlen:
                continue
            pa, ya = pred[:, a:b], ye[:, a:b]
            horizon.append({"span": f"{a}-{b}",
                            "r2": float(1 - ((pa - ya) ** 2).sum()
                                        / ((ya - ya.mean()) ** 2).sum().clamp_min(1e-12)),
                            "mse": float(((pa - ya) ** 2).mean())})
        zero_mse = float((ye ** 2).mean().item())
    torch.save({"name": name, "cfg": cfg, "task": task, "T": T,
                "state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "train": {"curve": curve, "eval_loss": ev_loss, "eval_metric": ev_metric,
                          "eval_len": int(Tlen), "zero_mse": zero_mse, "horizon": horizon,
                          "wall_s": round(time.time() - t0, 1)}},
               os.path.join(CKPT, f"{name}.pt"))
    torch.save({"x": xe.cpu(), "dt": dte.cpu(), "y": ye.cpu()},
               os.path.join(CKPT, f"{name}_eval.pt"))
    print(f"  [{name}] eval mse {ev_loss:.5f}  metric {ev_metric:.4f}  "
          f"train {time.time() - t0:.0f}s", flush=True)
    return {"name": name, "task": task, "D": D, "T": T, "T_train": T_train,
            "params": sum(p.numel() for p in model.parameters()),
            "eval_loss": ev_loss, "eval_metric": ev_metric, "zero_mse": zero_mse,
            "horizon": horizon, "curve": curve, "wall_s": round(time.time() - t0, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=list(SPECS))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    os.makedirs(CKPT, exist_ok=True)
    out = []
    for name in args.models:
        print(f"training {name} ...", flush=True)
        out.append(train_one(name, SPECS[name], args.device))
    path = os.path.join(ROOT, "results", "phase1_training.json")
    prev = []
    if os.path.exists(path):
        prev = [r for r in json.load(open(path)) if r["name"] not in {o["name"] for o in out}]
    json.dump(prev + out, open(path, "w"), indent=1)
    print("\nmodel                task          D     params     eval_mse  metric   time")
    for r in out:
        print(f"{r['name']:<20} {r['task']:<12} {r['D']:<5} {r['params']:>9} "
              f"{r['eval_loss']:>10.5f} {r['eval_metric']:>7.4f} {r['wall_s']:>6.0f}s")


if __name__ == "__main__":
    main()
