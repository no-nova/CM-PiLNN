"""Channel orderings used by the pi-CfC audit and ablation.

Every ordering is returned as a *permutation* `order` of length D where the cold prefix
`order[:k]` holds the (1-rho)D channels that will be run as a linear scan and the hot
suffix `order[k:]` holds the rho*D channels kept nonlinear.  All of them are pure index
vectors: the model parameters are permuted bilaterally by `models.permute`, so with
rho = 1 the computation is mathematically identical for every ordering.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


# --------------------------------------------------------------------------------------
# orderings
# --------------------------------------------------------------------------------------
def order_identity(D, device="cpu"):
    return torch.arange(D, device=device)


def order_random(D, seed=0, device="cpu"):
    g = torch.Generator().manual_seed(seed)
    return torch.randperm(D, generator=g).to(device)


def order_by_score(score):
    """Ascending score -> coldest first."""
    return torch.argsort(score)


def order_coact(score, h, group=32):
    """Co-activation grouping with the same greedy rule as the CM-LFM3 CATS blocks.

    Channels are clustered by their empirical activation correlation (seeded at the
    hottest unassigned channel), clusters are then placed cold-first by their mean score.
    """
    D = score.shape[0]
    hc = h.reshape(-1, D)
    hc = hc - hc.mean(0)
    hc = hc / hc.norm(dim=0).clamp_min(1e-8)
    corr = (hc.t() @ hc) / hc.shape[0]
    group = min(group, D)
    assigned = torch.zeros(D, dtype=torch.bool, device=score.device)
    clusters = []
    for _ in range(D // group):
        cand = torch.where(~assigned)[0]
        seed = cand[torch.argmax(score[cand])]
        members = [seed]
        assigned[seed] = True
        while len(members) < group:
            cand = torch.where(~assigned)[0]
            if cand.numel() == 0:
                break
            sim = corr[cand][:, members].mean(1)
            j = int(cand[torch.argmax(sim)])
            members.append(j)
            assigned[j] = True
        clusters.append(members)
    rest = torch.where(~assigned)[0].tolist()
    if rest:
        clusters.append(rest)
    clusters.sort(key=lambda m: float(score[torch.tensor(m, device=score.device)].mean()))
    out = [i for m in clusters for i in sorted(m, key=lambda j: float(score[j]))]
    return torch.tensor(out, device=score.device, dtype=torch.long)


# --------------------------------------------------------------------------------------
# residual score  J_i = E[(h_t - h~_t)^2] * ||W_out[:, i]||^2
# --------------------------------------------------------------------------------------
@torch.no_grad()
def residual_scores(model, x, dt, hbar=None):
    """Output-weighted nonlinear state-modulation residual, per channel.

    `h~` uses the *cold-linearised* gates: the recurrent contribution `U h_{t-1}` is
    replaced by its calibration mean `U hbar`, so the residual measures exactly the part of
    the step that only the state-dependent recurrent term can explain.
    """
    B, T, _ = x.shape
    D = model.D
    Wf, Wg, Wu = model.W_f, model.W_g, model.W_u
    xf = x @ Wf.t() + model.b_f
    xg = x @ Wg.t() + model.b_g
    xu = x @ Wu.t() + model.b_u              # (B, T, D)
    if hbar is None:
        _, H = model.forward_exact(x, dt, h0=None, keep_h=True)
        hbar = H.reshape(-1, D).mean(0)
    of, og, ou = model.U_f @ hbar, model.U_g @ hbar, model.U_u @ hbar

    h = x.new_zeros(B, D)
    acc = x.new_zeros(D)
    a_sum = x.new_zeros(D)
    for t in range(T):
        dtt = dt[:, t]
        ft = F.softplus(xf[:, t] + of)
        gt = torch.tanh(xg[:, t] + og)
        ut = torch.sigmoid(xu[:, t] + ou)
        at = torch.exp(-ft * dtt)
        h_tilde = at * h + (1.0 - at) * gt * ut
        # exact step (dynamic recurrent contribution)
        f = F.softplus(xf[:, t] + h @ model.U_f.t())
        g = torch.tanh(xg[:, t] + h @ model.U_g.t())
        u = torch.sigmoid(xu[:, t] + h @ model.U_u.t())
        a = torch.exp(-f * dtt)
        h = a * h + (1.0 - a) * g * u
        acc += (h - h_tilde).pow(2).mean(0)
        a_sum += a.mean(0)
    acc /= T
    wn = model.W_out.norm(dim=0)
    return {"J": acc * wn ** 2, "resid": acc, "wnorm": wn, "alpha_mean": a_sum / T, "hbar": hbar}


# --------------------------------------------------------------------------------------
# closed-form step Jacobian  dh_t / dh_{t-1}
# --------------------------------------------------------------------------------------
@torch.no_grad()
def step_jacobian(model, x, dt, h):
    """d h_t / d h_{t-1} at a given (x_t, dt_t, h_{t-1}); returns (B, D, D)."""
    D = model.D
    zf = x @ model.W_f.t() + h @ model.U_f.t() + model.b_f
    zg = x @ model.W_g.t() + h @ model.U_g.t() + model.b_g
    zu = x @ model.W_u.t() + h @ model.U_u.t() + model.b_u
    a = torch.exp(-F.softplus(zf) * dt)
    g = torch.tanh(zg)
    u = torch.sigmoid(zu)
    sf = torch.sigmoid(zf)
    c_f = dt * a * sf * (g * u - h)
    c_g = (1.0 - a) * (1.0 - g ** 2) * u
    c_u = (1.0 - a) * g * u * (1.0 - u)
    J = torch.diag_embed(a)
    J = J + c_f.unsqueeze(-1) * model.U_f.unsqueeze(0) \
          + c_g.unsqueeze(-1) * model.U_g.unsqueeze(0) \
          + c_u.unsqueeze(-1) * model.U_u.unsqueeze(0)
    return J


def gini(values):
    """Gini coefficient of a non-negative vector (0 = uniform, 1 = all mass on one item)."""
    v = values.flatten().double().clamp_min(0)
    n = v.numel()
    sv, _ = torch.sort(v)
    idx = torch.arange(1, n + 1, dtype=torch.float64, device=v.device)
    return float(((2 * idx - n - 1) * sv).sum() / (n * sv.sum().clamp_min(1e-30)))


def pareto_ratio(values, frac):
    """Share of total mass held by the largest `frac` fraction of entries."""
    v = values.flatten().float()
    k = max(1, int(round(frac * v.numel())))
    top = torch.topk(v, k).values.sum()
    return float(top / v.sum().clamp_min(1e-30))


def retained_curve(J, grid):
    """R(rho) = sum_{i in hot(rho)} J_i / sum_i J_i, for the top-rho fraction by J."""
    tot = J.sum().clamp_min(1e-30)
    out = []
    for r in grid:
        k = int(round(r * J.numel()))
        hot = torch.topk(J, k).values.sum() if k > 0 else J.new_zeros(())
        out.append(float(hot / tot))
    return out
