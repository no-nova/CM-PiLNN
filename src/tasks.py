"""Three irregular-time sequence tasks used to train authentic CfC checkpoints.

Each generator returns x (B,T,d_in), dt (B,T,1) and the *next-step* target (B,T,d_out),
sampled with a per-step dt ~ U(0.01, 0.20) exactly as specified for CM-PiLNN.
All dynamics are integrated with RK4 and a fixed number of substeps per sample, so the
targets are numerically honest ground truth rather than a coarse Euler rollout.
"""
from __future__ import annotations

import math

import torch

SUB = 4  # RK4 substeps per sampled dt


def _rk4(f, s, t, dt, substeps=SUB):
    h = dt / substeps
    for _ in range(substeps):
        k1 = f(s, t)
        k2 = f(s + 0.5 * h * k1, t + 0.5 * h)
        k3 = f(s + 0.5 * h * k2, t + 0.5 * h)
        k4 = f(s + h * k3, t + h)
        s = s + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        t = t + h
    return s, t


# --------------------------------------------------------------------------------------
# A: coupled Duffing + Van der Pol + linear harmonic modes
# --------------------------------------------------------------------------------------
def oscillator(B, T, device, gen=None):
    g = torch.Generator(device="cpu").manual_seed(int(torch.randint(0, 10**6, (1,), generator=gen)))
    dt = (0.01 + 0.19 * torch.rand(B, T, 1, generator=g)).to(device)
    w1 = 1.1 + 0.3 * torch.rand(B, 1, generator=g)
    w2 = 2.0 + 0.5 * torch.rand(B, 1, generator=g)
    F1 = 0.8 + 0.4 * torch.rand(B, 1, generator=g)
    F2 = 0.6 + 0.4 * torch.rand(B, 1, generator=g)
    w1, w2, F1, F2 = (z.to(device) for z in (w1, w2, F1, F2))
    s = torch.randn(B, 6, generator=g).to(device) * 0.3
    t = torch.rand(B, 1, generator=g).to(device) * 5.0

    def deriv(s, t):
        u, du, v, dv, w, dw = (s[:, i:i + 1] for i in range(6))
        ddu = -0.30 * du - 1.00 * u - 0.50 * u ** 3 + F1 * torch.cos(w1 * t)
        ddv = 0.80 * (1.0 - v ** 2) * dv - 1.20 * v + F2 * torch.cos(w2 * t)
        ddw = -2.00 * w - 0.05 * dw
        return torch.cat([du, ddu, dv, ddv, dw, ddw], dim=1)

    xs, ys = [], []
    for k in range(T):
        xk = torch.cat([torch.cos(w1 * t), torch.cos(w2 * t),
                        0.5 * torch.sin(0.1 * t)], dim=1)
        xs.append(xk)
        s, t = _rk4(deriv, s, t, dt[:, k])
        ys.append(s.clone())
    return torch.stack(xs, 1), dt, torch.stack(ys, 1)


# --------------------------------------------------------------------------------------
# B: cumulative event-parity automaton + continuous autoregressive filtering
# --------------------------------------------------------------------------------------
def parity(B, T, device, gen=None):
    g = torch.Generator(device="cpu").manual_seed(int(torch.randint(0, 10**6, (1,), generator=gen)))
    dt = (0.01 + 0.19 * torch.rand(B, T, 1, generator=g)).to(device)
    ev = (torch.rand(B, T, 1, generator=g) < 0.15).float().to(device)
    bit = (torch.rand(B, T, 1, generator=g) < 0.5).float().to(device)
    # continuous autoregressive signal, integrated in continuous time
    v = torch.zeros(B, T, 1, device=device)
    x0 = torch.randn(B, 1, generator=g).to(device)
    for k in range(T):
        a = torch.exp(-dt[:, k] / 0.5)
        x0 = a * x0 + (1 - a) * 0.7 * math.sin(0.37 * k) \
            + 0.25 * torch.randn(B, 1, generator=g).to(device) * dt[:, k]
        v[:, k:k + 1] = x0.unsqueeze(-1)
    p = torch.ones(B, 1, device=device)          # parity state in {-1, +1}
    filt = torch.zeros(B, 1, device=device)      # continuous filter state
    xs, ys = [], []
    for k in range(T):
        p = p * torch.where(ev[:, k] * bit[:, k] > 0.5,
                            -torch.ones_like(p), torch.ones_like(p))
        a = torch.exp(-dt[:, k] / 0.5)
        filt = a * filt + (1 - a) * v[:, k]
        xs.append(torch.cat([bit[:, k] * ev[:, k], v[:, k], ev[:, k]], dim=1))
        ys.append(torch.cat([p, filt], dim=1))
    return torch.stack(xs, 1), dt, torch.stack(ys, 1)


# --------------------------------------------------------------------------------------
# C: multi-regime continuous-time system (switch between damped, driven and decaying)
# --------------------------------------------------------------------------------------
def multiscale(B, T, device, gen=None):
    """Four damped harmonic modes spanning a 50x timescale range, driven at the fast end.

    Multi-timescale dynamics is what the spectrum audit (and the cold/hot split) is about:
    the fast mode is driven, the slower modes respond through the shared drive, so channels
    have genuinely different relaxation rates.  States: (u_k, du_k), k = 0..3, d_out = 8.
    """
    g = torch.Generator().manual_seed(int(torch.randint(0, 10 ** 6, (1,), generator=gen)))
    dt = (0.01 + 0.19 * torch.rand(B, T, 1, generator=g)).to(device)
    w = torch.tensor([3.0, 1.2, 0.5, 0.12], device=device)
    zeta = torch.tensor([0.08, 0.05, 0.03, 0.25], device=device)
    amp = (0.5 + 0.5 * torch.rand(B, 4, generator=g)).to(device)
    phase = (torch.rand(B, 1, generator=g) * 6.2832).to(device)
    s = torch.randn(B, 8, generator=g).to(device) * 0.1
    t = (torch.rand(B, 1, generator=g) * 4).to(device)

    def deriv(s, t):
        u, du = s[:, 0::2], s[:, 1::2]
        force = 0.35 * torch.cos(w[1] * t + phase) * amp[:, 1:]
        force = torch.cat([1.2 * torch.cos(w[0] * t + phase) * amp[:, :1], force], dim=1)
        ddu = -2 * zeta * w * du - (w ** 2) * u + force
        # state layout is interleaved (u_0, du_0, u_1, du_1, ...): re-interleave the
        # derivatives, otherwise the ODE solved is a different (unstable) system
        return torch.stack([du, ddu], dim=2).reshape(s.shape)

    xs, ys = [], []
    for k in range(T):
        xs.append(torch.stack([torch.cos(w[0] * t + phase)[:, 0],
                               torch.sin(w[0] * t + phase)[:, 0],
                               0.5 * torch.cos(0.3 * t)[:, 0],
                               torch.cos(w[1] * t + phase)[:, 0]], dim=1))
        s, t = _rk4(deriv, s, t, dt[:, k])
        ys.append(s.clone())
    return torch.stack(xs, 1), dt, torch.stack(ys, 1)


def multiregime(B, T, device, gen=None):
    g = torch.Generator(device="cpu").manual_seed(int(torch.randint(0, 10**6, (1,), generator=gen)))
    dt = (0.01 + 0.19 * torch.rand(B, T, 1, generator=g)).to(device)
    w = 1.3 + 0.4 * torch.rand(B, 1, generator=g)
    w, Fw = w.to(device), (0.9 + 0.3 * torch.rand(B, 1, generator=g)).to(device)
    s = torch.randn(B, 4, generator=g).to(device) * 0.3
    t = torch.rand(B, 1, generator=g).to(device) * 3.0
    reg = torch.zeros(B, 1, dtype=torch.long, device=device)
    xs, ys = [], []
    dwell = torch.randint(60, 140, (B, 1), generator=g).to(device)
    age = torch.zeros(B, 1, dtype=torch.long, device=device)

    def deriv(s, t, reg):
        a, da, b, db = (s[:, i:i + 1] for i in range(4))
        r0 = (reg == 0).float()
        r1 = (reg == 1).float()
        r2 = (reg == 2).float()
        dda = r0 * (-1.4 * a - 0.08 * da) + r1 * (-1.0 * a - 0.40 * a ** 3 - 0.20 * da
                                                   + Fw * torch.cos(w * t)) + r2 * (-0.6 * a)
        ddb = r0 * (-0.8 * b - 0.12 * db) + r1 * (0.5 * (1 - b ** 2) * db - 1.1 * b) \
            + r2 * (-0.6 * b)
        return torch.cat([da, dda, db, ddb], dim=1)

    for k in range(T):
        xs.append(torch.cat([torch.cos(w * t), torch.sin(w * t), reg.float() / 2.0], dim=1))
        s, t = _rk4(lambda st, tt: deriv(st, tt, reg), s, t, dt[:, k])
        ys.append(s.clone())
        age = age + 1
        flip = age >= dwell
        if flip.any():
            reg = torch.where(flip, (reg + 1) % 3, reg)
            age = torch.where(flip, torch.zeros_like(age), age)
            dwell = torch.where(flip, torch.randint(60, 140, (B, 1), generator=g).to(device), dwell)
    return torch.stack(xs, 1), dt, torch.stack(ys, 1)


TASKS = {"oscillator": (oscillator, 3, 6), "parity": (parity, 3, 2), "multiregime": (multiregime, 3, 4),
         "multiscale": (multiscale, 4, 8)}


def make_batch(name, B, T, device, gen=None):
    fn, _, _ = TASKS[name]
    return fn(B, T, device, gen=gen)


def task_metric(name, pred, target):
    """Task-level accuracy: parity sign accuracy for `parity`, R^2 for the state tasks."""
    if name == "parity":
        acc = ((pred[..., 0] > 0).float() == (target[..., 0] > 0).float()).float().mean()
        return float(acc.item())
    num = ((pred - target) ** 2).sum()
    den = ((target - target.mean()) ** 2).sum().clamp_min(1e-12)
    return float((1.0 - num / den).item())
