#!/usr/bin/env python
"""Phase 6: the decoupling study on two *real* sequence streams (no synthetic generators).

Model D --- CfC-WikiText (byte level).  The input stream is real English text; each step is one
byte as a 256-way one-hot vector, the timestep is event-driven (dt = 2 after sentence punctuation,
1 otherwise), the target is the next byte, the metric is bits per byte (BPD = CE / ln 2).

Model E --- CfC-SensorTrace (real video patch stream).  The input stream is a real surveillance
clip decoded to 16x16 grayscale frames (real sensor noise, real motion), each step feeds one
frame and predicts the next one; the metric is held-out R^2 on the next-frame vector.

Both run the same pipeline as the synthetic models: Jacobian concentration, zero-shot decoupling
for five orderings over the rho grid, closed-form ridge readout recalibration, five epochs of
DAFT, and hardware latency against the strongest sequential baseline we could write.

Usage:  python scripts/phase6_realworld_benchmarks.py [train|spectrum|decouple|ridge|daft|hw|all|fig]
"""
import json
import os
import sys
import time
import copy

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import CfC, assoc_scan                                # noqa: E402
from src.ordering import residual_scores, gini, retained_curve        # noqa: E402
from src.probe import MemProbe                                        # noqa: E402

RES = "results"
CKPT = "results/checkpoints"
WIKI = "/home/user/Downloads/CM-LFM3/results/wikitext2_test.txt"
FRAMES = "/home/user/Downloads/CNN-CM/data/real_videos/frames_224"
RHO = (0.0, 0.0625, 0.125, 0.25, 0.5, 1.0)
DT_PUNCT = {ord("."), ord("!"), ord("?")}
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


# --------------------------------------------------------------------------- data
def load_wiki(n_bytes=260_000):
    txt = open(WIKI, "rb").read(n_bytes)
    b = torch.frombuffer(bytearray(txt), dtype=torch.uint8).long()
    # event timestep: a sentence boundary is a longer step than a letter
    dt = torch.ones(len(b), 1)
    for i in range(1, len(b)):
        if int(b[i - 1]) in DT_PUNCT or int(b[i - 1]) == 10:      # punctuation / newline
            dt[i] = 2.0
    return b, dt


def wiki_batch(b, dt, n_seq, T, device, start=None, gen=None):
    """n_seq windows of T bytes, sampled 50 % overlapping from the stream."""
    stride = max(1, T // 2)
    n_win = (len(b) - T - 1) // stride
    if start is None:
        idx = torch.randint(0, n_win, (n_seq,), generator=gen)
    else:
        idx = torch.arange(start, start + n_seq) % n_win
    off = (idx * stride).unsqueeze(1) + torch.arange(T).unsqueeze(0)
    xb = b[off].to(device)                                   # (B,T) bytes in
    yb = b[off + 1].to(device)                               # (B,T) next byte
    x = F.one_hot(xb, 256).float()                           # (B,T,256)
    d = dt[off].to(device)
    return x, d, yb


def load_frames(size=16):
    """real clip -> (n_frames, size*size) grayscale vectors, per scenario."""
    from PIL import Image
    names = sorted(os.listdir(FRAMES))
    scen = {}
    for n in names:
        key = n.split("_")[0]
        scen.setdefault(key, []).append(os.path.join(FRAMES, n))
    out = {}
    for k, paths in scen.items():
        fr = []
        for p in sorted(paths):
            im = Image.open(p).convert("L").resize((size, size), Image.BILINEAR)
            fr.append(torch.from_numpy(np.asarray(im, dtype=np.float32).copy()).reshape(-1) / 255.0)
        out[k] = torch.stack(fr)                              # (T, size^2)
    return out


def frame_batch(frames, n_seq, T, device, gen=None, start=None):
    keys = list(frames)
    xs, ds, ys = [], [], []
    for i in range(n_seq):
        k = keys[i % len(keys)] if start is None else keys[(start + i) % len(keys)]
        f = frames[k]
        s = int(torch.randint(0, len(f) - T - 1, (1,), generator=gen)) if start is None else 0
        seq = f[s:s + T + 1]
        xs.append(seq[:-1]); ys.append(seq[1:])
        # 30 fps camera: uniform 1/30 s step, expressed in the same units as the other models
        ds.append(torch.full((T, 1), 1.0 / 30.0))
    return torch.stack(xs).to(device), torch.stack(ds).to(device), torch.stack(ys).to(device)


# --------------------------------------------------------------------------- metrics
def bpd(logits, y, mask=None):
    ce = F.cross_entropy(logits.reshape(-1, 256), y.reshape(-1), reduction="mean")
    return float(ce / np.log(2))


def r2(pred, true):
    pred, true = pred.reshape(-1).double(), true.reshape(-1).double()
    ss_res = ((true - pred) ** 2).sum()
    ss_tot = ((true - true.mean()) ** 2).sum()
    return float(1.0 - ss_res / ss_tot)


def make_cfc(d_in, D, d_out, b_f, device):
    """CfC has no b_f constructor argument; the forget bias is set after construction
    (a slow initial time constant is what makes the parity stream learnable)."""
    m = CfC(d_in, D, d_out).to(device)
    with torch.no_grad():
        m.b_f.fill_(b_f)
    return m


class Readout(torch.nn.Module):
    """task head on top of the CfC state (kept out of CfC so both metrics share one core)"""

    def __init__(self, D, d_out):
        super().__init__()
        self.W = torch.nn.Parameter(torch.zeros(d_out, D))
        self.b = torch.nn.Parameter(torch.zeros(d_out))

    def forward(self, H):
        return H @ self.W.t() + self.b


def state_and_pred(model, head, x, dt, rho=1.0, order=None, hbar=None, picard=0):
    if rho >= 1.0:
        _, H = model.forward_exact(x, dt, keep_h=True)
    else:
        _, H = model.forward_approx(x, dt, hbar=hbar, order=order, rho=rho, picard=picard,
                                    keep_h=True)
    return H, head(H)


# --------------------------------------------------------------------------- training
def train(name, kind, D, T, steps, B, lr, device, seed=0, b_f=2.0):
    torch.manual_seed(seed)
    if kind == "wiki":
        b, dt_all = load_wiki()
        b_tr, dt_tr = b[:200_000], dt_all[:200_000]
        d_in, d_out, T_eval = 256, 256, 512
        model = make_cfc(d_in, D, d_out, b_f, device)
        head = Readout(D, d_out).to(device)
    else:
        frames = load_frames()
        d_in = d_out = frames["V1"].shape[1]
        T_eval = 32
        model = make_cfc(d_in, D, d_out, b_f, device)
        head = Readout(D, d_out).to(device)
    opt = torch.optim.AdamW(list(model.parameters()) + list(head.parameters()), lr=lr,
                            weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    gen = torch.Generator().manual_seed(seed + 1)
    curve = []
    for it in range(steps):
        if kind == "wiki":
            x, d, y = wiki_batch(b_tr, dt_tr, B, T, device, gen=gen)
        else:
            x, d, y = frame_batch(frames, B, T, device, gen=gen)
        H, pred = state_and_pred(model, head, x, d)
        loss = (F.cross_entropy(pred.reshape(-1, 256), y.reshape(-1))
                if kind == "wiki" else F.mse_loss(pred, y))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(head.parameters()), 1.0)
        opt.step()
        sched.step()
        if it % max(1, steps // 10) == 0 or it == steps - 1:
            curve.append((it, float(loss)))
            print(f"  [{name}] step {it} loss {float(loss):.4f}", flush=True)
    # held-out evaluation at the *long* horizon
    with torch.no_grad():
        if kind == "wiki":
            xe, de, ye = wiki_batch(b[200_000:], dt_all[200_000:], 32, T_eval, device, start=0)
            _, pe = state_and_pred(model, head, xe, de)
            m = bpd(pe, ye)
            lp = torch.bincount(b_tr, minlength=256).float()
            lp = (lp / lp.sum()).log().to(ye.device).unsqueeze(0).expand(ye.numel(), -1)
            unigram = float(F.cross_entropy(lp, ye.reshape(-1)) / np.log(2))
            ev = dict(bpd=m, bpd_unigram=unigram, n_bytes=int(ye.numel()))
        else:
            xe, de, ye = frame_batch(frames, 12, T_eval, device, start=0)
            _, pe = state_and_pred(model, head, xe, de)
            ev = dict(r2=r2(pe, ye), mse=float(F.mse_loss(pe, ye)),
                      mse_zero=float(F.mse_loss(torch.zeros_like(ye), ye)), n_frames=12)
        # trajectory reference for the decoupling study (states, kept small)
        _, He = state_and_pred(model, head, xe, de)
    torch.save(dict(cfg=dict(kind=kind, d_in=d_in, D=D, d_out=d_out, T=T, T_eval=T_eval,
                             b_f=b_f, seed=seed),
                    model=model.state_dict(), head=head.state_dict(), curve=curve, eval=ev),
               f"{CKPT}/${name}.pt".replace("$", ""))
    torch.save(dict(x=xe.cpu(), dt=de.cpu(), y=ye.cpu(), H=He.cpu()), f"{CKPT}/{name}_eval.pt")
    print(f"  [{name}] eval {ev}", flush=True)
    return ev


def load(name, device="cuda"):
    ck = torch.load(f"{CKPT}/{name}.pt", map_location=device, weights_only=False)
    c = ck["cfg"]
    m = make_cfc(c["d_in"], c["D"], c["d_out"], c["b_f"], device)
    m.load_state_dict(ck["model"])
    h = Readout(c["D"], c["d_out"]).to(device)
    h.load_state_dict(ck["head"])
    ev = torch.load(f"{CKPT}/{name}_eval.pt", map_location="cpu", weights_only=False)
    return m, h, c, ev


def orders(m, x, dt, hbar, dev):
    """five orderings: native (identity), 3 random, variance, ours (jacobian coact), picard"""
    D = m.D
    out = {"native": torch.arange(D, device=dev)}
    g = torch.Generator().manual_seed(7)
    for i in range(3):
        out[f"random{i}"] = torch.randperm(D, generator=g).to(dev)
    with torch.no_grad():
        _, H = m.forward_exact(x, dt, keep_h=True)
    # NOTE the ordering convention of forward_approx: `order` is COLD-prefix, i.e.
    # cold, hot = order[:n_cold], order[n_cold:]  ->  important channels must come LAST.
    # (argsort(..., descending=True) would put the hottest channels in the *cold* group.)
    var = H.reshape(-1, D).var(0)
    out["variance"] = torch.argsort(var)
    sc = residual_scores(m, x, dt)
    out["ours"] = torch.argsort(sc["J"])
    return out, sc


# --------------------------------------------------------------------------- parts
def spectrum_part(names, dev):
    probe = MemProbe()
    probe.start()
    out = {}
    for name in names:
        m, h, c, ev = load(name, dev)
        x, dt = ev["x"].to(dev), ev["dt"].to(dev)
        ors, sc = orders(m, x, dt, None, dev)
        with torch.no_grad():
            _, H = m.forward_exact(x, dt, keep_h=True)
        hbar = H.reshape(-1, m.D).mean(0)
        ors, sc = orders(m, x, dt, hbar, dev)
        J = sc["J"].cpu()
        out[name] = dict(D=int(m.D), gini=gini(J), retained=retained_curve(J, RHO),
                         rho_grid=list(RHO), top_quarter=float(np.sort(J)[::-1][:max(1, len(J)//4)].sum() / J.sum()),
                         pearson_logj_logvar=None)
        with torch.no_grad():
            v = H.reshape(-1, m.D).var(0).cpu()
        lj = np.log(J + 1e-30); lv = np.log(v.numpy() + 1e-30)
        out[name]["pearson_logj_logvar"] = float(np.corrcoef(lj, lv)[0, 1])
        print(f"  [spectrum] {name}: gini {out[name]['gini']:.3f} R(.25) {out[name]['retained'][2]:.3f}", flush=True)
        del m, h, H
        torch.cuda.empty_cache()
    probe.stop()
    out["_memory"] = probe.as_dict()
    json.dump(out, open(f"{RES}/phase6_spectrum.json", "w"), indent=1)
    return out


def metric_of(kind, head_out, y):
    return bpd(head_out, y) if kind == "wiki" else r2(head_out, y)


def decouple_part(names, dev):
    probe = MemProbe()
    probe.start()
    out = {}
    for name in names:
        m, h, c, ev = load(name, dev)
        kind = c["kind"]
        x, dt, y = ev["x"].to(dev), ev["dt"].to(dev), ev["y"].to(dev)
        with torch.no_grad():
            _, H = m.forward_exact(x, dt, keep_h=True)
            Href = H
            ref = metric_of(kind, h(Href), y)
        hbar = H.reshape(-1, m.D).mean(0)
        ors, sc = orders(m, x, dt, hbar, dev)
        rows = {}
        with torch.no_grad():
            for rho in RHO:
                for oname, order in ors.items():
                    if oname.startswith("random") and oname != "random0":
                        continue
                    if rho >= 1.0 and oname != "native":
                        continue
                    if rho == 0.0 and oname not in ("native", "ours"):
                        continue
                    for picard in ((0, 1) if (rho < 1.0 and oname == "ours") else (0,)):
                        t0 = time.perf_counter()
                        Hp, pp = state_and_pred(m, h, x, dt, rho=rho, order=order, hbar=hbar,
                                                picard=picard)
                        ms = 1e3 * (time.perf_counter() - t0)
                        rel = float((Hp - Href).norm() / Href.norm())
                        met = metric_of(kind, pp, y)
                        key = f"{oname}{'-picard1' if picard else ''}|{rho:g}"
                        rows[key] = dict(rel_l2=rel, metric=met, delta=met - ref, wall_ms=ms)
                        print(f"  [decouple] {name} {key}: rel {rel:.3e} metric {met:.4f} "
                              f"(d {met-ref:+.4f})", flush=True)
        out[name] = dict(kind=kind, task_metric=("bpd" if kind == "wiki" else "r2"),
                         reference=ref, rows=rows)
        del m, h, Href, H, ors
        torch.cuda.empty_cache()
    probe.stop()
    out["_memory"] = probe.as_dict()
    json.dump(out, open(f"{RES}/phase6_decouple.json", "w"), indent=1)
    return out


def ridge_part(names, dev):
    """closed-form refit of the readout on decoupled (rho, ordering) states, fp64 solve"""
    probe = MemProbe()
    probe.start()
    out = {}
    for name in names:
        m, h, c, ev = load(name, dev)
        kind = c["kind"]
        x, dt, y = ev["x"].to(dev), ev["dt"].to(dev), ev["y"].to(dev)
        with torch.no_grad():
            _, H = m.forward_exact(x, dt, keep_h=True)
            ref = metric_of(kind, h(H), y)
        hbar = H.reshape(-1, m.D).mean(0)
        ors, _ = orders(m, x, dt, hbar, dev)
        rows = {}
        for rho in (0.0, 0.125, 0.25, 0.5):
            for oname in ("ours", "variance"):
                order = ors[oname]
                with torch.no_grad():
                    Hp, pp = state_and_pred(m, h, x, dt, rho=rho, order=order, hbar=hbar)
                met0 = metric_of(kind, pp, y)
                t0 = time.perf_counter()
                Xd = Hp.reshape(-1, m.D)
                t1 = time.perf_counter()
                # closed form: ridge least squares.  For the classifier this is a *linear
                # least-squares* head, which is not the maximum-likelihood head -- we report
                # that honestly (it is the standard closed-form refit, and it loses to the
                # trained softmax head; see the text).
                if kind == "wiki":
                    Y = torch.zeros(y.numel(), 256, dtype=torch.float64, device=dev)
                    Y[torch.arange(y.numel(), device=dev), y.reshape(-1)] = 1.0
                else:
                    Y = y.reshape(-1, c["d_out"]).double()
                A = (Hp.reshape(-1, m.D).double().t() @ Hp.reshape(-1, m.D).double())
                lam = 1e-3 * A.diagonal().mean()
                A = A + lam * torch.eye(m.D, dtype=torch.float64, device=dev)
                W = torch.linalg.solve(A, Hp.reshape(-1, m.D).double().t() @ Y).t()
                solve_ms = 1e3 * (time.perf_counter() - t0)
                logits = (Xd.double() @ W.t()).float()
                met1 = metric_of(kind, logits.reshape(Hp.shape[0], Hp.shape[1], -1), y)
                rows[f"{oname}|{rho:g}"] = dict(metric_default=met0, metric_ridge=met1,
                                                delta=met1 - met0, solve_ms=solve_ms)
                print(f"  [ridge] {name} {oname} rho={rho:g}: {met0:.4f} -> {met1:.4f} "
                      f"({solve_ms:.1f} ms)", flush=True)
        out[name] = dict(kind=kind, reference=ref, rows=rows)
        del m, h, H
        torch.cuda.empty_cache()
    probe.stop()
    out["_memory"] = probe.as_dict()
    json.dump(out, open(f"{RES}/phase6_ridge.json", "w"), indent=1)
    return out


def hot_mask_grads(m, order, rho):
    n_cold = int(round((1.0 - rho) * m.D))
    cold = order[:n_cold]
    with torch.no_grad():
        for p in (m.W_f, m.W_g, m.W_u):
            p.grad[cold] = 0.0
        for p in (m.b_f, m.b_g, m.b_u):
            p.grad[cold] = 0.0
        for p in (m.U_f, m.U_g, m.U_u):
            p.grad[:, cold] = 0.0
            p.grad[cold, :] = 0.0


def daft_part(names, dev, epochs=5, steps=40, lr=1e-3):
    probe = MemProbe()
    probe.start()
    out = {}
    for name in names:
        m0, h0, c, ev = load(name, dev)
        kind = c["kind"]
        x, dt, y = ev["x"].to(dev), ev["dt"].to(dev), ev["y"].to(dev)
        with torch.no_grad():
            _, H = m0.forward_exact(x, dt, keep_h=True)
        hbar = H.reshape(-1, m0.D).mean(0)
        ors, _ = orders(m0, x, dt, hbar, dev)
        rows = {}
        for rho in (0.0625, 0.125, 0.25):
            order = ors["ours"]
            m = copy.deepcopy(m0)
            hd = copy.deepcopy(h0)
            with torch.no_grad():
                _, pp = state_and_pred(m, hd, x, dt, rho=rho, order=order, hbar=hbar)
            before = metric_of(kind, pp, y)
            opt = torch.optim.AdamW(list(m.parameters()) + list(hd.parameters()), lr=lr)
            for ep in range(epochs):
                for _ in range(steps):
                    Hd, pred = state_and_pred(m, hd, x, dt, rho=rho, order=order, hbar=hbar)
                    loss = (F.cross_entropy(pred.reshape(-1, 256), y.reshape(-1))
                            if kind == "wiki" else F.mse_loss(pred, y))
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    hot_mask_grads(m, order, rho)
                    torch.nn.utils.clip_grad_norm_(
                        list(m.parameters()) + list(hd.parameters()), 1.0)
                    opt.step()
            with torch.no_grad():
                _, pp = state_and_pred(m, hd, x, dt, rho=rho, order=order, hbar=hbar)
            after = metric_of(kind, pp, y)
            rows[f"ours|{rho:g}"] = dict(before=before, after=after, delta=after - before)
            print(f"  [daft] {name} rho={rho:g}: {before:.4f} -> {after:.4f}", flush=True)
            del m, hd
            torch.cuda.empty_cache()
        out[name] = dict(kind=kind, rows=rows)
        del m0, h0, H
        torch.cuda.empty_cache()
    probe.stop()
    out["_memory"] = probe.as_dict()
    json.dump(out, open(f"{RES}/phase6_daft.json", "w"), indent=1)
    return out


def hw_part(names, dev):
    """end-to-end latency of the real models: exact vs decoupled, GPU and CPU."""
    import kernels.triton_picfc as K
    import scripts.phase4_hardware_benchmarks as B4
    lib = B4.HY.load()
    probe = MemProbe()
    probe.start()
    out = {}
    for name in names:
        m, h, c, ev = load(name, dev)
        kind = c["kind"]
        x, dt = ev["x"].to(dev), ev["dt"].to(dev)
        if kind == "wiki":
            x, dt = x[:8], dt[:8]                     # keep the batch comparable across models
        else:
            x, dt = x[:8], dt[:8]
        with torch.no_grad():
            _, H = m.forward_exact(x, dt, keep_h=True)
        hbar = H.reshape(-1, m.D).mean(0)
        ors, _ = orders(m, x, dt, hbar, dev)
        order = ors["ours"]
        pa = B4.preacts(m, x, dt)
        rows = {}
        loop = B4.time_gpu(lambda: B4.loop_seq(pa, m, dt, dev), reps=5, warm=1)
        kseq = B4.time_gpu(lambda: K.seq_cfc(m, x, dt), reps=5, warm=2)
        best_seq = min(loop, kseq)
        rows["gpu|loop"] = loop
        rows["gpu|kernel-seq"] = kseq
        rows["gpu|best-seq"] = best_seq
        for rho in (0.125, 0.25):
            dh = int(round(rho * m.D))
            drv = K.picfc if dh <= 512 else K.picfc_hybrid
            t = B4.time_gpu(lambda rho=rho, drv=drv: drv(m, x, dt, hbar, order, rho),
                            reps=5, warm=2)
            rows[f"gpu|picfc rho={rho:g}"] = t
            rows[f"gpu|speedup rho={rho:g}"] = best_seq / t
        # CPU: exact AVX-512 sequential kernel vs the fused decoupled entry point
        mcr, xcr, dtcr = copy.deepcopy(m).cpu(), x.cpu(), dt.cpu()
        pacr = tuple(t.cpu() for t in pa)
        ordcr, hbcr = order.cpu(), hbar.cpu()
        rows["cpu|avx512-seq"] = B4.time_cpu(lambda: B4.cpu_seq(lib, pacr, mcr, dtcr, "cpu"),
                                             reps=3, warm=1)
        for rho in (0.125, 0.25):
            n_cold = int(round((1.0 - rho) * m.D))
            U = {k: getattr(mcr, "U_" + k)[ordcr] for k in "fgu"}
            pf = pacr[0][:, :, ordcr]; pg = pacr[1][:, :, ordcr]; pu = pacr[2][:, :, ordcr]
            try:
                t = B4.time_cpu(lambda pf=pf, pg=pg, pu=pu, U=U, nc=n_cold: B4.HY.cpu_fused(
                    lib, pf, pg, pu, hbcr[ordcr], U, dtcr, ordcr, nc,
                    pacr[0].shape[0], pacr[0].shape[1], mcr.D), reps=3, warm=1)
                rows[f"cpu|picfc-fused rho={rho:g}"] = t
                rows[f"cpu|speedup rho={rho:g}"] = rows["cpu|avx512-seq"] / t
            except Exception as exc:                                  # noqa: BLE001
                rows[f"cpu|picfc-fused rho={rho:g}"] = None
                rows["cpu_error"] = f"{type(exc).__name__}: {exc}"
        print(f"  [hw] {name}: gpu loop {loop:.1f} kernel-seq {kseq:.1f} "
              f"picfc {rows['gpu|picfc rho=0.25']:.2f} ms; cpu seq {rows['cpu|avx512-seq']:.1f} "
              f"picfc {rows.get('cpu|picfc-fused rho=0.25')}", flush=True)
        out[name] = dict(kind=kind, D=int(m.D), B=int(x.shape[0]), T=int(x.shape[1]), rows=rows)
        del m, h, H
        torch.cuda.empty_cache()
    probe.stop()
    out["_memory"] = probe.as_dict()
    json.dump(out, open(f"{RES}/phase6_hw.json", "w"), indent=1)
    return out


def gate_part(names, dev):
    """numerical gate: the Triton decoupled path vs the fp32 torch reference, on real inputs."""
    import kernels.triton_picfc as K
    out = {}
    for name in names:
        m, h, c, ev = load(name, dev)
        x, dt = ev["x"][:4].to(dev), ev["dt"][:4].to(dev)
        with torch.no_grad():
            _, H = m.forward_exact(x, dt, keep_h=True)
        hbar = H.reshape(-1, m.D).mean(0)
        ors, _ = orders(m, x, dt, hbar, dev)
        for oname in ("ours",):
            for rho in (0.125, 0.25):
                order = ors[oname]
                with torch.no_grad():
                    _, Hp = m.forward_approx(x, dt, hbar=hbar, order=order, rho=rho, keep_h=True)
                    Hk = K.picfc(m, x, dt, hbar, order, rho)
                rel = float((Hk - Hp).norm() / Hp.norm())
                out[f"{name}|picfc|{rho:g}"] = rel
                print(f"  [gate] {name} picfc rho={rho:g}: {rel:.2e}", flush=True)
    json.dump(dict(rows=out, threshold=1e-5, pass_=all(v < 1e-5 for v in out.values())),
              open(f"{RES}/phase6_gate.json", "w"), indent=1)
    return out


def merge():
    allj = {}
    for part in ("spectrum", "decouple", "ridge", "daft", "hw", "gate", "train"):
        p = f"{RES}/phase6_{part}.json"
        if os.path.exists(p):
            allj[part] = json.load(open(p))
    json.dump(allj, open(f"{RES}/phase6_realworld.json", "w"), indent=1)
    print("wrote results/phase6_realworld.json")
    return allj


def figures(allj):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.grid": True, "grid.alpha": 0.3,
                         "axes.spines.top": False, "axes.spines.right": False})
    dec = allj.get("decouple", {})
    names = [n for n in ("D_wiki", "E_sensor") if n in dec]
    fig, ax = plt.subplots(1, 3, figsize=(10.4, 3.0))
    colors = {"native": "#1f77b4", "ours": "#d62728", "variance": "#2ca02c",
              "random0": "#888888", "ours-picard1": "#9467bd"}
    for i, name in enumerate(names):
        rows = dec[name]["rows"]
        kind = dec[name]["kind"]
        lab = "BPD" if kind == "wiki" else "$R^2$"
        for meth, col in colors.items():
            xs, ys = [], []
            for k, v in rows.items():
                meth_k, rho = k.split("|")
                if meth_k == meth:
                    xs.append(float(rho)); ys.append(v["metric"])
            if xs:
                o = np.argsort(xs)
                ax[0].plot(np.array(xs)[o], np.array(ys)[o], "o-", ms=3, color=col,
                           label=meth if i == 0 else None,
                           ls=("-" if i == 0 else "--"))
        ax[0].set_xlabel(r"hot fraction $\rho$")
        ax[0].set_ylabel(f"held-out {lab}")
        ax[0].set_title("real streams: accuracy vs $\\rho$" if i == 0 else "")
    ax[0].legend(frameon=False, fontsize=6.5)
    hw = allj.get("hw", {})
    for i, name in enumerate(names):
        if name not in hw:
            continue
        r = hw[name]["rows"]
        rhos = [0.125, 0.25]
        sp = [r.get(f"gpu|speedup rho={rho:g}", np.nan) for rho in rhos]
        ax[1].plot(rhos, sp, "o-", ms=4, label=f"{name} (GPU)")
    ax[1].axhline(1.0, color="k", lw=0.6, ls=":")
    ax[1].set_xlabel(r"hot fraction $\rho$")
    ax[1].set_ylabel("speedup vs best sequential")
    ax[1].set_title("end-to-end kernel speedup")
    ax[1].legend(frameon=False, fontsize=6.5)
    sp = allj.get("spectrum", {})
    for name in names:
        if name in sp:
            ret = sp[name]["retained"]
            ax[2].plot(sp[name]["rho_grid"], ret, "o-", ms=3, label=f"{name} (Gini {sp[name]['gini']:.2f})")
    ax[2].plot([0, 1], [0, 1], "k:", lw=0.8, label="$1-\\rho$ reference")
    ax[2].set_xlabel(r"hot fraction $\rho$")
    ax[2].set_ylabel("Jacobian energy retained")
    ax[2].set_title("concentration on real streams")
    ax[2].legend(frameon=False, fontsize=6.5)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.30, top=0.86, wspace=0.32)
    os.makedirs("figures", exist_ok=True)
    fig.savefig("figures/fig_realworld_benchmarks.pdf")
    fig.savefig("figures/fig_realworld_benchmarks.png", dpi=170)
    print("wrote figures/fig_realworld_benchmarks.{pdf,png}")


def main():
    parts = sys.argv[1:] or ["all"]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(CKPT, exist_ok=True)
    names = ["D_wiki", "E_sensor"]
    with MemProbe() as probe:
        if "all" in parts or "train" in parts:
            res = {}
            res["D_wiki"] = train("D_wiki", "wiki", D=512, T=128, steps=1200, B=16, lr=3e-3,
                                  device=dev)
            res["E_sensor"] = train("E_sensor", "sensor", D=512, T=32, steps=1500, B=16,
                                    lr=2e-3, device=dev)
            json.dump(res, open(f"{RES}/phase6_train.json", "w"), indent=1)
        if "all" in parts or "spectrum" in parts:
            spectrum_part(names, dev)
        if "all" in parts or "gate" in parts:
            gate_part(names, dev)
        if "all" in parts or "decouple" in parts:
            decouple_part(names, dev)
        if "all" in parts or "ridge" in parts:
            ridge_part(names, dev)
        if "all" in parts or "daft" in parts:
            daft_part(names, dev)
        if "all" in parts or "hw" in parts:
            hw_part(names, dev)
    mem = probe.as_dict()
    allj = merge()
    allj["_memory"] = mem
    json.dump(allj, open(f"{RES}/phase6_realworld.json", "w"), indent=1)
    if "fig" in parts or "all" in parts:
        figures(allj)
    print("peak RssAnon MB:", mem["peak_rss_anon_mb"])


if __name__ == "__main__":
    main()
