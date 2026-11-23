"""CM-PiLNN: Closed-form continuous-time (CfC) layer, bilateral permutation and the
two-stage decoupled execution used by pi-CfC.

The recurrence implemented here is exactly the gated CfC of Hasani et al. (2022):

    f_t = softplus(W_f x_t + U_f h_{t-1} + b_f)      alpha_t = exp(-f_t * dt_t)
    g_t = tanh    (W_g x_t + U_g h_{t-1} + b_g)
    u_t = sigmoid (W_u x_t + U_u h_{t-1} + b_u)
    h_t = alpha_t * h_{t-1} + (1 - alpha_t) * (g_t * u_t)
    y_t = W_out h_t + b_out

Everything is written so that the same code path serves
  * `forward_exact`  -- the reference recurrence (all channels nonlinear),
  * `forward_approx` -- the decoupled two-stage (cold associative scan + hot compact
    recurrence) execution with an optional one-step Picard refinement.

A permutation pi is represented as a *channel order* (a LongTensor of length D): the
permuted model has channel order[0] in position 0, etc.  Cold channels are always the
prefix, hot channels the suffix, so every method in the ablation differs only in the
order it supplies.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------------------
# associative scan
# --------------------------------------------------------------------------------------
def assoc_scan(a: torch.Tensor, b: torch.Tensor, h0: torch.Tensor) -> torch.Tensor:
    """Inclusive parallel scan of h_t = a_t * h_{t-1} + b_t with h_{-1} = h0.

    a, b: (B, T, C);  h0: (B, 1, C);  returns (B, T, C).
    Hillis-Steele: log2(T) rounds of the associative op
        (a2, b2) o (a1, b1) = (a2 a1, b2 + a2 b1).
    """
    B, T, C = a.shape
    A = a.clone()
    Bb = b.clone()
    # identity element for the shift-in
    eye_a = torch.ones_like(A)
    zero_b = torch.zeros_like(Bb)
    step = 1
    while step < T:
        A_sh = torch.cat([eye_a[:, :step], A[:, :-step]], dim=1)
        B_sh = torch.cat([zero_b[:, :step], Bb[:, :-step]], dim=1)
        Bb = Bb + A * B_sh
        A = A * A_sh
        step *= 2
    return A * h0 + Bb


def scan_sequential(a: torch.Tensor, b: torch.Tensor, h0: torch.Tensor) -> torch.Tensor:
    """Reference sequential recurrence (used only to validate assoc_scan)."""
    B, T, C = a.shape
    h = h0[:, 0]
    out = []
    for t in range(T):
        h = a[:, t] * h + b[:, t]
        out.append(h)
    return torch.stack(out, dim=1)


# --------------------------------------------------------------------------------------
# the layer
# --------------------------------------------------------------------------------------
class CfC(nn.Module):
    """Single CfC cell unrolled over a sequence; parameters are the full D x D blocks."""

    def __init__(self, d_in: int, D: int, d_out: int, bias: bool = True):
        super().__init__()
        self.d_in, self.D, self.d_out = d_in, D, d_out
        g_in = 1.0 / math.sqrt(d_in)
        g_rec = 1.0 / math.sqrt(D)
        for nm in ("f", "g", "u"):
            self.register_parameter(f"W_{nm}", nn.Parameter(torch.empty(D, d_in)))
            self.register_parameter(f"U_{nm}", nn.Parameter(torch.empty(D, D)))
            self.register_parameter(f"b_{nm}",
                                    nn.Parameter(torch.zeros(D) if bias else torch.zeros(D)))
        self.W_out = nn.Parameter(torch.empty(d_out, D))
        self.b_out = nn.Parameter(torch.zeros(d_out))
        with torch.no_grad():
            # input drive at O(1), recurrent block at the usual 1/sqrt(D) scale, and a
            # zero forget bias so the learned time constants are not biased slow a priori
            for nm in ("f", "g", "u"):
                getattr(self, f"W_{nm}").uniform_(-g_in, g_in)
                getattr(self, f"U_{nm}").uniform_(-g_rec, g_rec)
            self.b_f.fill_(0.0)
            self.W_out.uniform_(-1.0 / math.sqrt(D), 1.0 / math.sqrt(D))

    # -- convenience -------------------------------------------------------------------
    def names(self):
        return tuple(f"{p}_{n}" for p in ("W", "U", "b") for n in ("f", "g", "u")) + ("W_out", "b_out")

    def tensors(self):
        return {n: getattr(self, n) for n in self.names()}

    # -- the two forward modes ---------------------------------------------------------
    def _cold_gates(self, x, dt, hbar, sl):
        """Input-only (cold-linearised) gates for channel slice `sl` around mean state hbar."""
        fs = F.softplus(x @ self.W_f[sl].t() + hbar @ self.U_f[sl].t() + self.b_f[sl])
        gs = torch.tanh(x @ self.W_g[sl].t() + hbar @ self.U_g[sl].t() + self.b_g[sl])
        us = torch.sigmoid(x @ self.W_u[sl].t() + hbar @ self.U_u[sl].t() + self.b_u[sl])
        a = torch.exp(-fs * dt)
        b = (1.0 - a) * gs * us
        return a, b

    def forward_exact(self, x, dt, h0=None, keep_h=False):
        """Reference recurrence. x (B,T,d_in), dt (B,T,1), h0 (B,D) or None."""
        B, T, _ = x.shape
        D = self.D
        h = torch.zeros(B, D, dtype=x.dtype, device=x.device) if h0 is None else h0.clone()
        xf = x @ self.W_f.t() + self.b_f
        xg = x @ self.W_g.t() + self.b_g
        xu = x @ self.W_u.t() + self.b_u
        hs = []
        for t in range(T):
            ht = h
            f = F.softplus(xf[:, t] + ht @ self.U_f.t())
            g = torch.tanh(xg[:, t] + ht @ self.U_g.t())
            u = torch.sigmoid(xu[:, t] + ht @ self.U_u.t())
            a = torch.exp(-f * dt[:, t])
            h = a * ht + (1.0 - a) * g * u
            hs.append(h)
        H = torch.stack(hs, 1)                       # (B, T, D)
        y = H @ self.W_out.t() + self.b_out          # (B, T, d_out)
        return (y, H) if keep_h else (y, H[:, -1])

    def forward_approx(self, x, dt, h0=None, hbar=None, order=None, rho=1.0, picard=0,
                       keep_h=False):
        """Two-stage decoupled execution in the channel order `order` (cold prefix)."""
        B, T, _ = x.shape
        D = self.D
        if order is None:
            order = torch.arange(D, device=x.device)
        n_cold = int(round((1.0 - rho) * D))
        cold, hot = order[:n_cold], order[n_cold:]
        if h0 is None:
            h0 = torch.zeros(B, D, dtype=x.dtype, device=x.device)
        h0c, h0h = h0[:, cold], h0[:, hot]
        if hbar is None:
            hbar = torch.zeros(D, dtype=x.dtype, device=x.device)
        hb_c, hb_h = hbar[cold], hbar[hot]

        # ---- stage 1: cold linear scan (x-only gates) --------------------------------
        if n_cold:
            a_c, b_c = self._cold_gates(x, dt, hbar, cold)
            h_c = assoc_scan(a_c, b_c, h0c.unsqueeze(1))
        else:
            h_c = x.new_zeros(B, T, 0)

        # ---- stage 2: hot compact recurrence -----------------------------------------
        if len(hot):
            zf = x @ self.W_f[hot].t() + self.b_f[hot]
            zg = x @ self.W_g[hot].t() + self.b_g[hot]
            zu = x @ self.W_u[hot].t() + self.b_u[hot]
            if n_cold:
                # the hot gates at step t need h_{t-1}, i.e. the cold trajectory shifted by
                # one step (h0 for t = 0) -- feeding h_c[:, t] is a one-step lookahead bug
                hc_prev = torch.cat([h0c.unsqueeze(1), h_c[:, :-1]], dim=1)
                zf = zf + torch.einsum("btd,ed->bte", hc_prev, _sub(self.U_f, hot, cold))
                zg = zg + torch.einsum("btd,ed->bte", hc_prev, _sub(self.U_g, hot, cold))
                zu = zu + torch.einsum("btd,ed->bte", hc_prev, _sub(self.U_u, hot, cold))
            Uh_f = _sub(self.U_f, hot, hot)
            Uh_g = _sub(self.U_g, hot, hot)
            Uh_u = _sub(self.U_u, hot, hot)
            hh = h0h.clone()
            outs = []
            for t in range(T):
                f = F.softplus(zf[:, t] + hh @ Uh_f.t())
                g = torch.tanh(zg[:, t] + hh @ Uh_g.t())
                u = torch.sigmoid(zu[:, t] + hh @ Uh_u.t())
                a = torch.exp(-f * dt[:, t])
                hh = a * hh + (1.0 - a) * g * u
                outs.append(hh)
            h_h = torch.stack(outs, 1)
            if picard and n_cold:
                h_c = self._picard_cold(x, dt, hbar, cold, hot, h_c, h_h, h0c)
        else:
            h_h = x.new_zeros(B, T, 0)

        # ---- reassemble in permuted space, then un-permute for the readout -----------
        h_perm = torch.cat([h_c, h_h], dim=2)
        h = torch.empty(B, T, D, dtype=x.dtype, device=x.device)
        h[:, :, order] = h_perm
        y = h @ self.W_out.t() + self.b_out
        return (y, h) if keep_h else (y, h[:, -1])

    def _picard_cold(self, x, dt, hbar, cold, hot, h_c, h_h, h0c):
        """One Picard step: cold gates re-evaluated with the exact hot->cold coupling."""
        Uh_prev = torch.cat([h0c.unsqueeze(1), h_c[:, :-1]], dim=1)
        hot_prev = torch.cat([torch.zeros_like(h_h[:, :1]), h_h[:, :-1]], dim=1)
        add_f = (torch.einsum("btd,ed->bte", hot_prev, _sub(self.U_f, cold, hot))
                 + torch.einsum("btd,ed->bte", Uh_prev, _sub(self.U_f, cold, cold)))
        add_g = (torch.einsum("btd,ed->bte", hot_prev, _sub(self.U_g, cold, hot))
                 + torch.einsum("btd,ed->bte", Uh_prev, _sub(self.U_g, cold, cold)))
        add_u = (torch.einsum("btd,ed->bte", hot_prev, _sub(self.U_u, cold, hot))
                 + torch.einsum("btd,ed->bte", Uh_prev, _sub(self.U_u, cold, cold)))
        # the exact recurrent contribution of both halves replaces the hbar linearisation
        # entirely; adding `hbar @ U[cold].T` on top would double count the cold-cold term
        fs = F.softplus(x @ self.W_f[cold].t() + self.b_f[cold] + add_f)
        gs = torch.tanh(x @ self.W_g[cold].t() + self.b_g[cold] + add_g)
        us = torch.sigmoid(x @ self.W_u[cold].t() + self.b_u[cold] + add_u)
        a = torch.exp(-fs * dt)
        b = (1.0 - a) * gs * us
        return assoc_scan(a, b, h0c.unsqueeze(1))


# --------------------------------------------------------------------------------------
# permutation plumbing
# --------------------------------------------------------------------------------------
def _sub(M: torch.Tensor, rows: torch.Tensor, cols: torch.Tensor) -> torch.Tensor:
    """M[rows][:, cols] as one advanced-indexing op (torch has no `ix_`)."""
    return M[rows.unsqueeze(1), cols.unsqueeze(0)]


def permute(cfc: CfC, order: torch.Tensor) -> CfC:
    """Return a new CfC whose channel `i` is the original channel `order[i]`."""
    new = CfC.__new__(CfC)
    nn.Module.__init__(new)
    new.d_in, new.D, new.d_out = cfc.d_in, cfc.D, cfc.d_out
    order = order.to(cfc.W_out.device)
    for nm in ("f", "g", "u"):
        W = getattr(cfc, f"W_{nm}").detach()[order].clone()
        U = getattr(cfc, f"U_{nm}").detach()[order][:, order].clone()
        b = getattr(cfc, f"b_{nm}").detach()[order].clone()
        new.register_parameter(f"W_{nm}", nn.Parameter(W))
        new.register_parameter(f"U_{nm}", nn.Parameter(U))
        new.register_parameter(f"b_{nm}", nn.Parameter(b))
    new.W_out = nn.Parameter(cfc.W_out.detach()[:, order].clone())
    new.b_out = nn.Parameter(cfc.b_out.detach().clone())
    for p in new.parameters():
        p.requires_grad_(False)
    return new


def unpermute_params_one_sided(cfc: CfC, order: torch.Tensor) -> CfC:
    """Negative control: permute the *rows* of W/U/b and the state, but neither the U
    columns nor the W_out columns."""
    new = CfC.__new__(CfC)
    nn.Module.__init__(new)
    new.d_in, new.D, new.d_out = cfc.d_in, cfc.D, cfc.d_out
    for nm in ("f", "g", "u"):
        new.register_parameter(f"W_{nm}", nn.Parameter(getattr(cfc, f"W_{nm}").detach()[order].clone()))
        new.register_parameter(f"U_{nm}", nn.Parameter(getattr(cfc, f"U_{nm}").detach()[order].clone()))
        new.register_parameter(f"b_{nm}", nn.Parameter(getattr(cfc, f"b_{nm}").detach()[order].clone()))
    new.W_out = nn.Parameter(cfc.W_out.detach().clone())
    new.b_out = nn.Parameter(cfc.b_out.detach().clone())
    for p in new.parameters():
        p.requires_grad_(False)
    return new


def rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    num = (a - b).double().pow(2).sum().sqrt()
    den = b.double().pow(2).sum().sqrt().clamp_min(1e-30)
    return float((num / den).item())


def mean_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().max().item())
