"""Non-TC0 state-tracking benchmarks (Phase 3, Pillar 3).

Three tasks that a purely linear time-varying (diagonal, positive-decay) recurrence cannot
solve, because the required update is either a state-dependent sign flip or a cyclic state
permutation:

  parity_stream   s_t = s_{t-1} XOR spike_t                       -> 2 classes
  mod3_counter    s_t = (s_{t-1} + trigger_t) mod 3               -> 3 classes
  flipflop_reset  s_t = set/reset/hold automaton (+1 / -1 / hold)  -> 3 classes

All three are classification over the *state* at every timestep, so accuracy is not diluted
by a decoder.  The initial state is zero and known (as in the standard state-tracking
protocol); with a *random* unobserved s_0 the parity target is unpredictable up to a global
flip and no architecture can beat 0.5 -- that variant is unlearnable by construction.
"""
from __future__ import annotations

import torch

DT = 0.02
D_IN = 1


def _spikes(B, T, p, device):
    return (torch.rand(B, T, device=device) < p).float()


def parity_stream(B, T, device, p=0.18):
    x = _spikes(B, T, p, device)                      # 0/1 events
    s = torch.zeros(B, T, dtype=torch.long, device=device)
    cur = torch.zeros(B, dtype=torch.long, device=device)   # known initial state

    for t in range(T):
        cur = torch.remainder(cur + x[:, t].long(), 2)
        s[:, t] = cur
    return x.unsqueeze(-1), s


def mod3_counter(B, T, device, p=0.18):
    trig = _spikes(B, T, p, device).long()
    s = torch.zeros(B, T, dtype=torch.long, device=device)
    cur = torch.zeros(B, dtype=torch.long, device=device)   # known initial state

    for t in range(T):
        cur = torch.remainder(cur + trig[:, t], 3)
        s[:, t] = cur
    x = (trig.float() - 0.18).unsqueeze(-1)           # centred scalar input
    return x, s


def flipflop_reset(B, T, device, p_set=0.10, p_reset=0.10):
    r = torch.rand(B, T, device=device)
    cmd = torch.zeros(B, T, device=device)            # 0 hold
    cmd[r < p_set] = 1.0
    cmd[(r >= p_set) & (r < p_set + p_reset)] = -1.0
    s0 = torch.zeros(B, dtype=torch.long, device=device)    # 0,1,2 <-> state -1,0,+1
    cur = s0.clone()
    s = torch.zeros(B, T, dtype=torch.long, device=device)
    for t in range(T):
        c = cmd[:, t]
        cur = torch.where(c > 0, torch.full_like(cur, 2),
                          torch.where(c < 0, torch.zeros_like(cur), cur))
        s[:, t] = cur
    return cmd.unsqueeze(-1), s


SUITE = {
    "parity_stream": dict(fn=parity_stream, d_out=2, label="Parity-Stream"),
    "mod3_counter": dict(fn=mod3_counter, d_out=3, label="Modulo-3 Counter"),
    "flipflop_reset": dict(fn=flipflop_reset, d_out=3, label="FlipFlop-Reset"),
}


def make_batch(task, B, T, device, seed=None):
    spec = SUITE[task]
    if seed is not None:
        st, stc = torch.get_rng_state(), torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        torch.manual_seed(seed)
        try:
            x, s = spec["fn"](B, T, device)
        finally:
            torch.set_rng_state(st)
            if stc is not None:
                torch.cuda.set_rng_state_all(stc)
    else:
        x, s = spec["fn"](B, T, device)
    dt = torch.full((B, T, 1), DT, device=device)
    return x, dt, s


@torch.no_grad()
def accuracy(model, task, x, dt, s, rho=1.0, order=None, picard=0, hbar=None):
    """Per-timestep classification accuracy and balanced accuracy (mean per-class recall)."""
    if order is None and rho >= 1.0:
        y, _ = model.forward_exact(x, dt, keep_h=True)
    else:
        y, _ = model.forward_approx(x, dt, hbar=hbar, order=order, rho=rho,
                                    picard=picard, keep_h=True)
    pred = y.argmax(-1)
    acc = (pred == s).float().mean().item()
    recalls = []
    for c in range(SUITE[task]["d_out"]):
        m = s == c
        if m.any():
            recalls.append((pred[m] == c).float().mean().item())
    return acc, sum(recalls) / max(len(recalls), 1)
