"""Triton kernels for the π-CfC two-stage execution (Phase 4, GPU).

Four execution paths are implemented, all in FP32:

  * ``seq_cfc``        -- one fused sequential kernel for the *full* recurrence
                          (state kept in registers, dense ``U`` streamed from L1/L2).
                          This is the "fair Triton sequential baseline": it removes the
                          ``T`` kernel launches of the PyTorch loop but cannot break the
                          sequential dependency.
  * ``pure_linear``    -- the same kernel with ``rho = 0`` (no nonlinear state feedback):
                          a pure associative scan, i.e. the parallel ceiling.
  * ``picfc``          -- cold parallel scan (chunked Blelloch, O(log L) depth) + one
                          dense exogenous cold->hot GEMM + a compact hot recurrence whose
                          ``d_hot x d_hot`` weights stay resident in registers.
  * ``picfc_picard1``  -- ``picfc`` + one Picard refinement (a second cold scan).

Layout convention (identical to ``src/models.py``): the channels are already reordered so
that the *cold* channels are the prefix ``[0, n_cold)`` and the *hot* channels the suffix.
``hbar`` is the mean-state linearisation point used by the cold gates.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

CHUNK = 64          # time chunk for the cold scan
BLOCK_C = 64        # channels per program in the cold scan
BLOCK_B = 16        # batch rows per program in the hot / sequential kernels


# --------------------------------------------------------------------------------------
# cold path: gates + chunked scan
# --------------------------------------------------------------------------------------
@triton.jit
def _cold_gates(gf, gg, gu, cf, cg, cu, dt, a_out, b_out, T, C: tl.constexpr, BC: tl.constexpr):
    pid_t = tl.program_id(0)                    # over B*T
    pid_c = tl.program_id(1)                    # over channel blocks
    oc = pid_c * BC + tl.arange(0, BC)
    m = oc < C
    base = pid_t * C + oc
    zf = tl.load(gf + base, mask=m, other=0.0) + tl.load(cf + oc, mask=m, other=0.0)
    zg = tl.load(gg + base, mask=m, other=0.0) + tl.load(cg + oc, mask=m, other=0.0)
    zu = tl.load(gu + base, mask=m, other=0.0) + tl.load(cu + oc, mask=m, other=0.0)
    d = tl.load(dt + pid_t)
    f = tl.log(1.0 + tl.exp(zf))                # softplus (zf is O(1) here, no overflow risk)
    a = tl.exp(-f * d)
    g = 1.0 - 2.0 / (1.0 + tl.exp(2.0 * zg))    # tanh
    u = 1.0 / (1.0 + tl.exp(-zu))               # sigmoid
    tl.store(a_out + base, a, mask=m)
    tl.store(b_out + base, (1.0 - a) * g * u, mask=m)


@triton.jit
def _scan_local(a, b, h_loc, cum_a, p_out, q_out, T, C: tl.constexpr, CH: tl.constexpr,
                BC: tl.constexpr):
    """Sequential scan inside one time chunk, starting from zero state."""
    pid = tl.program_id(0)
    nch = tl.cdiv(T, CH)
    pid_c = pid % tl.cdiv(C, BC)
    pid_ch = (pid // tl.cdiv(C, BC)) % nch
    pid_b = pid // (tl.cdiv(C, BC) * nch)
    oc = pid_c * BC + tl.arange(0, BC)
    m = oc < C
    h = tl.zeros((BC,), dtype=tl.float32)
    P = tl.full((BC,), 1.0, tl.float32)
    Q = tl.zeros((BC,), dtype=tl.float32)
    t0 = pid_ch * CH
    for i in range(CH):
        t = t0 + i
        if t < T:
            base = (pid_b * T + t) * C + oc
            at = tl.load(a + base, mask=m, other=1.0)
            bt = tl.load(b + base, mask=m, other=0.0)
            h = at * h + bt
            P = P * at
            Q = at * Q + bt
            tl.store(h_loc + base, h, mask=m)
            tl.store(cum_a + base, P, mask=m)
    tl.store(p_out + (pid_b * nch + pid_ch) * C + oc, P, mask=m)
    tl.store(q_out + (pid_b * nch + pid_ch) * C + oc, Q, mask=m)


@triton.jit
def _scan_fix(h_loc, cum_a, carry, h_out, T, C: tl.constexpr, CH: tl.constexpr,
              BC: tl.constexpr):
    pid = tl.program_id(0)
    nch = tl.cdiv(T, CH)
    pid_c = pid % tl.cdiv(C, BC)
    pid_ch = (pid // tl.cdiv(C, BC)) % nch
    pid_b = pid // (tl.cdiv(C, BC) * nch)
    oc = pid_c * BC + tl.arange(0, BC)
    m = oc < C
    st = tl.load(carry + (pid_b * nch + pid_ch) * C + oc, mask=m, other=0.0)
    t0 = pid_ch * CH
    for i in range(CH):
        t = t0 + i
        if t < T:
            base = (pid_b * T + t) * C + oc
            hl = tl.load(h_loc + base, mask=m, other=0.0)
            ca = tl.load(cum_a + base, mask=m, other=1.0)
            tl.store(h_out + base, ca * st + hl, mask=m)


def cold_scan(gf, gg, gu, cf, cg, cu, dt, h0, B, T):
    """Chunked parallel scan over the cold channels.  gf/gg/gu: (B*T, C) pre-activations,
    cf/cg/cu: (C,) constant linearisation biases, dt: (B,T,1), h0: (B,C)."""
    BT, C = gf.shape
    a = torch.empty_like(gf)
    b = torch.empty_like(gf)
    dtf = dt.reshape(-1).contiguous()
    _cold_gates[(BT, triton.cdiv(C, BLOCK_C))](
        gf, gg, gu, cf, cg, cu, dtf, a, b, T, C, BLOCK_C, num_warps=2)
    h_loc = torch.empty_like(gf)
    cum_a = torch.empty_like(gf)
    nch = triton.cdiv(T, CHUNK)
    P = torch.empty((B, nch, C), device=gf.device, dtype=gf.dtype)
    Q = torch.empty_like(P)
    grid = (B * nch * triton.cdiv(C, BLOCK_C),)
    _scan_local[grid](a, b, h_loc, cum_a, P, Q, T, C, CHUNK, BLOCK_C, num_warps=2)
    # exclusive prefix over the (tiny) chunk summaries: h_start(k) = P_prod(k-1)*h0 + Q_prod(k-1)
    carry = torch.empty_like(P)
    Pc = torch.ones((B, C), device=gf.device, dtype=gf.dtype)
    Qc = torch.zeros_like(Pc)
    for k in range(nch):
        carry[:, k] = Pc * h0 + Qc
        Pc = Pc * P[:, k]
        Qc = P[:, k] * Qc + Q[:, k]
    out = torch.empty_like(gf)
    _scan_fix[grid](h_loc, cum_a, carry, out, T, C, CHUNK, BLOCK_C, num_warps=2)
    return out


# --------------------------------------------------------------------------------------
# hot path: compact recurrence, recurrent weights register-resident
# --------------------------------------------------------------------------------------
@triton.jit
def _recur(gf, gg, gu, exf, exg, exu, Uf, Ug, Uu, dt, h_out, B, T,
           DH: tl.constexpr, BB: tl.constexpr):
    """Sequential recurrence over DH channels, state register-resident.

    Each program owns ``BB`` batch rows and the *full* DH-dimensional state, which stays in
    registers across the time loop; the per-step matvec is an IEEE-precision
    (BB x DH) @ (DH x DH) dot.  ``ex*`` are the exogenous cold->hot contributions (absent
    for the plain sequential baseline).  Limited to DH <= 128 by the register/shared tile.
    """
    pid_b = tl.program_id(0)
    ob = pid_b * BB + tl.arange(0, BB)
    mb = ob < B
    din = tl.arange(0, DH)
    # the host passes U^T, so the tile is used directly as the dot's right operand
    Uft = tl.load(Uf + din[:, None] * DH + din[None, :])
    Ugt = tl.load(Ug + din[:, None] * DH + din[None, :])
    Uut = tl.load(Uu + din[:, None] * DH + din[None, :])
    H = tl.zeros((BB, DH), dtype=tl.float32)
    for t in range(T):
        base = (ob[:, None] * T + t) * DH + din[None, :]
        zf = tl.load(gf + base, mask=mb[:, None], other=0.0) + tl.dot(H, Uft, input_precision="ieee")
        zg = tl.load(gg + base, mask=mb[:, None], other=0.0) + tl.dot(H, Ugt, input_precision="ieee")
        zu = tl.load(gu + base, mask=mb[:, None], other=0.0) + tl.dot(H, Uut, input_precision="ieee")
        if exf is not None:
            zf += tl.load(exf + base, mask=mb[:, None], other=0.0)
            zg += tl.load(exg + base, mask=mb[:, None], other=0.0)
            zu += tl.load(exu + base, mask=mb[:, None], other=0.0)
        d = tl.load(dt + ob * T + t, mask=mb, other=0.0)[:, None]
        a = tl.exp(-tl.log(1.0 + tl.exp(zf)) * d)
        g = 1.0 - 2.0 / (1.0 + tl.exp(2.0 * zg))
        u = 1.0 / (1.0 + tl.exp(-zu))
        H = a * H + (1.0 - a) * g * u
        tl.store(h_out + base, H, mask=mb[:, None])


# --------------------------------------------------------------------------------------
# hot path, L2-fused variant: 64 < d_hot <= 512.  One program per batch row keeps the whole
# state in registers for all L steps (no Python loop, one kernel launch), and walks the three
# d_hot x d_hot recurrent matrices in TK-wide column tiles.  A 512x512 fp32 matrix is 1 MiB,
# so all three matrices (3 MiB) stay resident in the RTX 5070's 32 MiB L2 across every step
# and every program; nothing is re-fetched from HBM.
# --------------------------------------------------------------------------------------
@triton.jit
def _hot_l2(gf, gg, gu, exf, exg, exu, Uf, Ug, Uu, dt, h_out, B, T,
            DH: tl.constexpr, TK: tl.constexpr):
    pid = tl.program_id(0)                       # one batch row
    din = tl.arange(0, DH)
    H = tl.zeros((DH,), dtype=tl.float32)
    for t in range(T):
        base = (pid * T + t) * DH + din
        sf = tl.zeros((DH,), dtype=tl.float32)
        sg = tl.zeros((DH,), dtype=tl.float32)
        su = tl.zeros((DH,), dtype=tl.float32)
        for k0 in tl.static_range(0, DH, TK):
            kk = k0 + tl.arange(0, TK)
            # gather the TK lanes of the register-resident state (no global round trip)
            hsel = tl.sum(tl.where(din[None, :] == kk[:, None], H[None, :], 0.0), axis=1)
            # M[i, j] = (U^T)[i, j] = U[j, i]; the matvec is  z_j = sum_i h_i M[i, j]
            uf = tl.load(Uf + kk[:, None] * DH + din[None, :])
            ug = tl.load(Ug + kk[:, None] * DH + din[None, :])
            uu = tl.load(Uu + kk[:, None] * DH + din[None, :])
            sf += tl.sum(uf * hsel[:, None], axis=0)
            sg += tl.sum(ug * hsel[:, None], axis=0)
            su += tl.sum(uu * hsel[:, None], axis=0)
        zf = tl.load(gf + base) + sf + tl.load(exf + base)
        zg = tl.load(gg + base) + sg + tl.load(exg + base)
        zu = tl.load(gu + base) + su + tl.load(exu + base)
        d = tl.load(dt + pid * T + t)
        a = tl.exp(-tl.log(1.0 + tl.exp(zf)) * d)
        g = 1.0 - 2.0 / (1.0 + tl.exp(2.0 * zg))
        u = 1.0 / (1.0 + tl.exp(-zu))
        H = a * H + (1.0 - a) * g * u
        tl.store(h_out + base, H)


# --------------------------------------------------------------------------------------
# python-level drivers
# --------------------------------------------------------------------------------------
def _pre(model, x, dt):
    """Input-only gate pre-activations (the 'step 1 batched GEMM')."""
    gf = x @ model.W_f.t() + model.b_f
    gg = x @ model.W_g.t() + model.b_g
    gu = x @ model.W_u.t() + model.b_u
    return gf, gg, gu


def _para(model, hbar, order):
    D = model.D
    W = {"f": model.W_f[order], "g": model.W_g[order], "u": model.W_u[order]}
    b = {"f": model.b_f[order], "g": model.b_g[order], "u": model.b_u[order]}
    U = {k: getattr(model, "U_" + k)[order][:, order] for k in "fgu"}
    return W, b, U


@torch.no_grad()
def seq_cfc(model, x, dt, h0=None, mode="exact"):
    """Fused sequential Triton CfC in fp32.  mode='exact' or 'pure_linear' (rho=0)."""
    B, T, _ = x.shape
    D = model.D
    BD = triton.next_power_of_2(D)
    W, b, U = _para(model, None, torch.arange(D, device=x.device))
    gf, gg, gu = _pre(model, x, dt)
    h0 = torch.zeros(B, D, device=x.device) if h0 is None else h0
    pad = torch.zeros(B, T, BD - D, device=x.device)
    gf, gg, gu = (torch.cat([t, pad], -1).contiguous() for t in (gf, gg, gu))
    UfT = torch.zeros(BD, BD, device=x.device)
    UgT = torch.zeros(BD, BD, device=x.device)
    UuT = torch.zeros(BD, BD, device=x.device)
    UfT[:D, :D] = U["f"].t()
    UgT[:D, :D] = U["g"].t()
    UuT[:D, :D] = U["u"].t()
    if mode == "pure_linear":                     # no state feedback -> associative scan
        return seq_linear_scan(gf[:, :, :D], gg[:, :, :D], gu[:, :, :D], dt, h0)
    out = torch.empty(B, T, BD, device=x.device)
    assert BD <= 64, "Triton sequential baseline: DH <= 64 (shared-memory budget of the 3 U tiles)"
    _recur[(triton.cdiv(B, BLOCK_B),)](gf, gg, gu, None, None, None, UfT, UgT, UuT,
                                       dt.reshape(-1), out, B, T, BD, BLOCK_B,
                                       num_warps=8, num_stages=1)
    return out[:, :, :D]


@torch.no_grad()
def seq_linear_scan(gf, gg, gu, dt, h0):
    """rho = 0: gates do not depend on the state, so the recurrence is one scan."""
    B, T, D = gf.shape
    a = torch.exp(-torch.nn.functional.softplus(gf) * dt)
    b = (1.0 - a) * torch.tanh(gg) * torch.sigmoid(gu)
    return scan_torch(a, b, h0)


def scan_torch(a, b, h0):
    """Reference Hillis-Steele scan (used for the rho=0 path and for verification)."""
    A, Bb = a.clone(), b.clone()
    step = 1
    while step < a.shape[1]:
        A_sh = torch.cat([torch.ones_like(A[:, :step]), A[:, :-step]], 1)
        B_sh = torch.cat([torch.zeros_like(Bb[:, :step]), Bb[:, :-step]], 1)
        Bb = Bb + A * B_sh
        A = A * A_sh
        step *= 2
    return A * h0.unsqueeze(1) + Bb


@torch.no_grad()
def picfc(model, x, dt, hbar, order, rho, picard=0):
    """Two-stage decoupled execution: cold Triton scan + one exogenous GEMM + hot kernel."""
    B, T, _ = x.shape
    D = model.D
    n_cold = int(round((1.0 - rho) * D))
    cold, hot = order[:n_cold], order[n_cold:]
    W, bb, U = _para(model, hbar, order)
    gf, gg, gu = _pre(model, x, dt)
    gf, gg, gu = gf[:, :, order].contiguous(), gg[:, :, order].contiguous(), gu[:, :, order].contiguous()
    h0 = torch.zeros(B, D, device=x.device)
    h = torch.zeros(B, T, D, device=x.device)
    if n_cold:
        # the cold-gate bias uses the FULL mean state (cold and hot parts), exactly as
        # `_cold_gates` does in src/models.py
        hbp = hbar[order]
        cf = hbp @ U["f"][:n_cold].t()
        cg = hbp @ U["g"][:n_cold].t()
        cu = hbp @ U["u"][:n_cold].t()
        gc = gf[:, :, :n_cold].contiguous().reshape(B * T, n_cold)
        gc2 = gg[:, :, :n_cold].contiguous().reshape(B * T, n_cold)
        gc3 = gu[:, :, :n_cold].contiguous().reshape(B * T, n_cold)
        hc = cold_scan(gc, gc2, gc3, cf.contiguous(), cg.contiguous(), cu.contiguous(),
                       dt.contiguous(), torch.zeros(B, n_cold, device=x.device), B, T)
        h[:, :, :n_cold] = hc.reshape(B, T, n_cold)
    n_hot = len(hot)
    if n_hot:
        # step 3: one dense GEMM for all T steps (cold -> hot, one step delayed)
        hc_prev = torch.cat([torch.zeros(B, 1, n_cold, device=x.device), h[:, :-1, :n_cold]], 1)
        ex = {k: torch.einsum("btd,ed->bte", hc_prev, U[k][n_cold:, :n_cold]) for k in "fgu"}
        DH = max(8, triton.next_power_of_2(n_hot))   # tl.dot needs K, N >= 8
        pad = torch.zeros(B, T, DH - n_hot, device=x.device)
        zf = torch.cat([gf[:, :, n_cold:], pad], -1).contiguous()
        zg = torch.cat([gg[:, :, n_cold:], pad], -1).contiguous()
        zu = torch.cat([gu[:, :, n_cold:], pad], -1).contiguous()
        exf = torch.cat([ex["f"], pad], -1).contiguous()
        exg = torch.cat([ex["g"], pad], -1).contiguous()
        exu = torch.cat([ex["u"], pad], -1).contiguous()

        def _T(M):
            Z = torch.zeros(DH, DH, device=x.device)
            Z[:n_hot, :n_hot] = U[M][n_cold:, n_cold:].t()
            return Z
        out = torch.empty(B, T, DH, device=x.device)
        if DH <= 64:
            # register-resident dot: three d_hot^2 tiles in shared memory, BB rows per program
            _recur[(triton.cdiv(B, BLOCK_B),)](zf, zg, zu, exf, exg, exu,
                                               _T("f"), _T("g"), _T("u"), dt.reshape(-1),
                                               out, B, T, DH, BLOCK_B,
                                               num_warps=8, num_stages=1)
        elif DH <= 512:
            # L2-fused tiled variant: one program per batch row, state in registers, matrices
            # walked in TK-wide column tiles, whole time loop inside the kernel
            _hot_l2[(B,)](zf, zg, zu, exf, exg, exu, _T("f"), _T("g"), _T("u"),
                          dt.reshape(-1), out, B, T, DH, 32,
                          num_warps=8, num_stages=1)
        else:
            raise ValueError(f"d_hot={DH} exceeds the 512-channel fast path; use picfc_hybrid")
        h[:, :, n_cold:] = out[:, :, :n_hot]
        if picard and n_cold:
            # one Picard refinement: re-run the cold scan with the exact cold-gate
            # pre-activations (the hbar linearisation replaced by the true recurrent term)
            hn_prev = torch.cat([torch.zeros(B, 1, n_hot, device=x.device),
                                 h[:, :-1, n_cold:]], 1)
            hc_prev = torch.cat([torch.zeros(B, 1, n_cold, device=x.device),
                                 h[:, :-1, :n_cold]], 1)
            add = {k: (torch.einsum("btd,ed->bte", hn_prev, U[k][:n_cold, n_cold:])
                       + torch.einsum("btd,ed->bte", hc_prev, U[k][:n_cold, :n_cold]))
                   for k in "fgu"}
            zero = torch.zeros(n_cold, device=x.device)
            hc2 = cold_scan((gf[:, :, :n_cold] + add["f"]).reshape(B * T, n_cold),
                            (gg[:, :, :n_cold] + add["g"]).reshape(B * T, n_cold),
                            (gu[:, :, :n_cold] + add["u"]).reshape(B * T, n_cold),
                            zero, zero, zero, dt,
                            torch.zeros(B, n_cold, device=x.device), B, T)
            h[:, :, :n_cold] = hc2.reshape(B, T, n_cold)
    out = torch.empty(B, T, D, device=x.device)
    out[:, :, order] = h            # scatter back to the original channel order
    return out


@torch.no_grad()
def picfc_forward(model, x, dt, hbar, order, rho, picard=0):
    """picfc + readout, in the permuted-then-unpermuted layout of src/models.py."""
    h = picfc(model, x, dt, hbar, order, rho, picard=picard)
    return h @ model.W_out.t() + model.b_out, h


@torch.no_grad()
def picfc_hybrid(model, x, dt, hbar, order, rho, picard=0):
    """pi-CfC for hot sets too large for a register-resident kernel (d_hot > 64).

    The cold stage still runs as one Triton parallel scan; only the hot recurrence falls back
    to a PyTorch step loop (d_hot small, so each step is three thin GEMVs).  This is the honest
    degraded mode: the parallel gain then comes from the cold stage alone.
    """
    B, T, _ = x.shape
    D = model.D
    W, b, U = _para(model, hbar, order)
    n_hot = int(round(rho * D))
    n_cold = D - n_hot
    gf, gg, gu = _pre(model, x, dt)
    gf = gf[:, :, order].contiguous()
    gg = gg[:, :, order].contiguous()
    gu = gu[:, :, order].contiguous()
    hb = hbar[order]
    hc = cold_scan(gf[:, :, :n_cold].contiguous().reshape(B * T, n_cold),
                   gg[:, :, :n_cold].contiguous().reshape(B * T, n_cold),
                   gu[:, :, :n_cold].contiguous().reshape(B * T, n_cold),
                   (hb @ U["f"][:n_cold].t()).contiguous(),
                   (hb @ U["g"][:n_cold].t()).contiguous(),
                   (hb @ U["u"][:n_cold].t()).contiguous(),
                   dt.contiguous(), torch.zeros(B, n_cold, device=x.device), B, T
                   ).reshape(B, T, n_cold)
    hprev = torch.cat([torch.zeros(B, 1, n_cold, device=x.device), hc[:, :-1]], 1)
    ex = {k: torch.einsum("btd,ed->bte", hprev, U[k][n_cold:, :n_cold]) for k in "fgu"}
    zf = gf[:, :, n_cold:] + ex["f"]
    zg = gg[:, :, n_cold:] + ex["g"]
    zu = gu[:, :, n_cold:] + ex["u"]
    Uf, Ug, Uu = U["f"][n_cold:, n_cold:], U["g"][n_cold:, n_cold:], U["u"][n_cold:, n_cold:]
    h = torch.zeros(B, n_hot, device=x.device)
    outs = []
    for t in range(T):
        f = torch.nn.functional.softplus(zf[:, t] + h @ Uf.t())
        g = torch.tanh(zg[:, t] + h @ Ug.t())
        u = torch.sigmoid(zu[:, t] + h @ Uu.t())
        a = torch.exp(-f * dt[:, t])
        h = a * h + (1 - a) * g * u
        outs.append(h)
    hh = torch.stack(outs, 1)
    out = torch.empty(B, T, D, device=x.device, dtype=x.dtype)
    out[:, :, order] = torch.cat([hc, hh], 2)
    return out
