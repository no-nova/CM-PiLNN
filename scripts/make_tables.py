#!/usr/bin/env python
"""Generate the LaTeX tables of the Phase 1-4 report from the result JSONs."""
import json
import os

RES = "results"
OUT = os.environ.get("TAB_OUT", "paper")


def w(name, text):
    # a table emitted inside a font-size group must close that group again
    if text.lstrip().startswith("{") and text.rstrip().endswith(r"\end{tabular}"):
        text = text.rstrip() + "}"
    os.makedirs(OUT, exist_ok=True)
    open(os.path.join(OUT, name), "w").write(text)
    print("wrote", name, len(text), "chars")


def f(x, n=3):
    return f"{x:.{n}f}"


# ------------------------------------------------------------------ training
def tab_training():
    rows = json.load(open(f"{RES}/phase1_training.json"))
    body = []
    for r in rows:
        hz = "/".join(f(h["r2"], 2) for h in r["horizon"])
        body.append(f"{r['name'].replace('_','-')} & {r['task']} & {r['D']} & {r['T']} & "
                    f"{r['params']/1e3:.0f}k & {f(r['eval_loss'])} & {f(r['eval_metric'])} & {hz} \\\\")
    w("tab_training.tex", "\n".join([
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{llrrrrrc}",
        r"\toprule",
        r"checkpoint & task & $D$ & $T$ & params & eval MSE & metric & $R^2$ by horizon \\\\",
        r"\midrule", *body, r"\bottomrule", r"\end{tabular}", ""]))


# ------------------------------------------------------------------ isomorphism
def tab_iso():
    d = json.load(open(f"{RES}/phase1_isomorphism.json"))
    body = []
    for k in ("A_oscillator", "B_parity", "C_largescale"):
        m = d[k]
        fp32 = m["fp32"]; fp64 = m["fp64"]
        bil32 = max(fp32[x]["rel_l2"] for x in fp32 if "one-sided" not in x)
        bil64 = max(fp64[x]["rel_l2"] for x in fp64 if "one-sided" not in x)
        one32 = max(fp32[x]["rel_l2"] for x in fp32 if "one-sided" in x)
        one64 = max(fp64[x]["rel_l2"] for x in fp64 if "one-sided" in x)
        body.append(f"{k.replace('_','-')} & {m['D']} & {m['T_fp32']} & {bil64:.1e} & "
                    f"{bil32:.1e} & {one64:.3f} & {one32:.3f} \\\\")
    w("tab_iso.tex", "\n".join([
        r"\begin{tabular}{lrrrrrr}", r"\toprule",
        r"checkpoint & $D$ & $T$ & fp64 worst & fp32 worst & one-sided fp64 & one-sided fp32 \\",
        r"\midrule", *body, r"\bottomrule", r"\end{tabular}", ""]))


# ------------------------------------------------------------------ phase 2
def tab_phase2():
    d = json.load(open(f"{RES}/phase2_equivalence.json"))
    order = ["pure-linear", "native-prefix", "random-pi", "variance-pi",
             "ours-jacobian-pi", "ours-jacobian-picard1"]
    for mk in ("A_oscillator", "B_parity", "C_largescale"):
        m = d["models"][mk]
        rhos = [r for r in m["rho_grid"] if r < 1.0]
        hdr = " & ".join(f"{r:g}" for r in rhos)
        body = []
        for meth in order:
            if meth not in m["rows"]:
                continue
            cells = []
            for r in rhos:
                v = m["rows"][meth].get(str(r)) or m["rows"][meth].get(f"{r:g}")
                cells.append("--" if v is None else f"{v['rel_l2']:.2f}")
            body.append(f"{meth.replace('-','-')} & " + " & ".join(cells) + r" \\")
        w(f"tab_p2_{mk}.tex", "\n".join([
            r"\begin{tabular}{l" + "r" * len(rhos) + "}", r"\toprule",
            r"method & \multicolumn{" + str(len(rhos)) + r"}{c}{$\rho$ (hot fraction)} \\",
            r"\cmidrule(lr){2-" + str(len(rhos) + 1) + "}",
            r" & " + hdr + r" \\", r"\midrule", *body, r"\bottomrule", r"\end{tabular}", ""]))


# ------------------------------------------------------------------ phase 3
def tab_ridge():
    """Pivoted: one row per (checkpoint, method), one column per hot fraction."""
    d = json.load(open(f"{RES}/phase3_expressivity.json"))
    RHO = ("0", "0.0625", "0.125", "0.25", "0.5")
    METH = ("pure-linear", "native-prefix", "variance-pi", "ours-jacobian-pi",
            "ours-jacobian-picard1")
    body = []
    for mk in ("A_oscillator", "B_parity", "C_largescale"):
        tab = d["ridge"][mk]
        body.append(r"\midrule")
        for meth in METH:
            cells = []
            for rho in RHO:
                v = tab.get(f"{meth}|{rho}")
                if not v:
                    cells.append("--")
                else:
                    cells.append(f"{v['metric_default']:.3f}$\\to${v['metric_ridge']:.3f}")
            if all(c == "--" for c in cells):
                continue
            body.append(f"{mk.replace('_','-')} & \\code{{{meth}}} & " + " & ".join(cells) + r" \\")
    w("tab_ridge.tex", "\n".join([
        r"{", r"\begin{tabular}{ll" + "c" * len(RHO) + "}", r"\toprule",
        r"checkpoint & method & " + " & ".join(f"$\\rho={r}$" if r != "0" else "$\\rho=0$"
                                               for r in RHO) + r" \\",
        r"\midrule", *[b for b in body if b], r"\bottomrule", r"\end{tabular}}", ""]))


def tab_ridge_gpu():
    """Cost of the refit on the accelerator that ran the forward pass."""
    d = json.load(open(f"{RES}/phase3_ridge_gpu.json"))
    body = []
    last = None
    for key, v in d["rows"].items():
        mk, meth, rho = key.split("|")
        if mk != last:
            body.append(r"\midrule" if last else "")
            last = mk
        body.append(f"{mk.replace('_','-')} & {meth} & {float(rho):g} & {v['forward_ms']:.0f} & "
                    f"{v['gram_ms']:.2f} & {v['solve_ms']:.1f} & {v['refit_ms']:.1f} \\\\")
    w("tab_ridge_gpu.tex", "\n".join([
        r"{\scriptsize\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{llrrrrr}", r"\toprule",
        r"checkpoint & method & $\rho$ & rollouts (ms) & Gram (ms) & solve (ms) & refit (ms) \\",
        r"\midrule", *[b for b in body if b], r"\bottomrule", r"\end{tabular}", ""]))


def tab_expr():
    d = json.load(open(f"{RES}/phase3_expressivity.json"))["expressivity"]["tasks"]
    body = []
    for task, t in d.items():
        label = t["label"]
        body.append(r"\midrule")
        body.append("\\multicolumn{6}{l}{\\emph{" + label + "} ($L=256$)} $\\quad$ full CfC: \\textbf{" + f"{t['full_cfc']['acc']:.3f}" + "} \\\\")
        for rho in ("0.0625", "0.125", "0.25"):
            for meth in ("ours-jacobian-pi", "variance-pi", "random-pi"):
                v = t["rho"].get(rho, {}).get(meth)
                if v is None:
                    continue
                body.append(f"{label} & {meth.replace('-pi','')} & {float(rho):g} & "
                            f"{v['acc_zero_shot']:.3f} & {v['acc_daft']:.3f} & "
                            f"{v['daft_s']:.1f} \\\\")
    w("tab_expr.tex", "\n".join([
        r"\begin{tabular}{llrrrr}", r"\toprule",
        r"task & ordering & $\rho$ & acc.\ zero-shot & acc.\ after DAFT & DAFT (s) \\",
        r"\midrule", *body, r"\bottomrule", r"\end{tabular}", ""]))


def tab_osc():
    d = json.load(open(f"{RES}/phase3_expressivity.json"))["oscillator_daft"]
    body = []
    for rho, meths in d.items():
        for meth, v in meths.items():
            body.append(f"{float(rho):g} & {meth.replace('-pi','')} & {f(v['r2_before'])} & "
                        f"{f(v['r2_after'])} & {v['daft_s']:.1f} \\\\")
    w("tab_osc.tex", "\n".join([
        r"\begin{tabular}{lrrrr}", r"\toprule",
        r"$\rho$ & ordering & $R^2$ before DAFT & $R^2$ after DAFT & DAFT (s) \\",
        r"\midrule", *body, r"\bottomrule", r"\end{tabular}", ""]))


def tab_gate():
    d = json.load(open(f"{RES}/phase4_kernel_gate.json"))
    body = [f"\\texttt{{{k.replace('_','\\_')}}} & {v:.2e} \\\\" for k, v in d["rows"].items()]
    w("tab_gate.tex", "\n".join([
        r"\begin{tabular}{lr}", r"\toprule", r"path & rel.\ $L_2$ vs.\ reference \\",
        r"\midrule", *body, r"\bottomrule",
        r"\multicolumn{2}{l}{threshold $10^{-5}$: \textbf{" +
        ("PASS" if d["pass"] else "FAIL") + r"}} \\", r"\end{tabular}", ""]))


def tab_p4():
    d = json.load(open(f"{RES}/phase4_hardware.json"))
    rows = d["rows"]

    def mark(r):
        return "$^{*}$" if r.get("micro_batched") else ""

    # GPU table: fixed D, vary L, B=8
    gpu = []
    for r in rows:
        if r.get("skipped") or r["B"] != 8 or "ms" not in r:
            continue
        ms = r["ms"]
        base = ms["gpu|pytorch-seq-cfc"]
        kern = ms.get("gpu|triton-seq-cfc", ms.get("gpu|fused-seq-cfc"))
        best = None
        best_s = 0
        for k, v in ms.items():
            if ("triton-picfc" in k) and "picard1" not in k and v > 0:
                s_ = base / v
                if s_ > best_s:
                    best_s, best = s_, (k, v)
        if best is None:
            continue
        kseq = ms.get("gpu|triton-seq-kernel")
        bseq = ms.get("gpu|best-seq", min(v for v in (base, kern, kseq) if v))
        gpu.append(f"{r['D']}{mark(r)} & {r['L']} & {base:.2f} & "
                   + (f"{kseq:.2f}" if kseq else "--")
                   + f" & {best[1]:.2f} & {base/best[1]:.1f}$\\times$ & "
                   f"{bseq/best[1]:.1f}$\\times$ \\\\")
    w("tab_p4_gpu.tex", "\n".join([
        r"\begin{tabular}{rrrrrrr}", r"\toprule",
        r"$D$ & $L$ & loop & 1-launch seq. & $\pi$-CfC & vs loop & vs best seq. \\",
        r"\midrule",
        *gpu, r"\bottomrule", r"\end{tabular}", ""]))

    cpu = []
    for r in rows:
        if r.get("skipped") or r["B"] != 8 or "ms" not in r:
            continue
        ms = r["ms"]
        if "cpu|avx512-seq-cfc" not in ms:
            continue
        base = ms["cpu|avx512-seq-cfc"]
        py = min((v for k, v in ms.items()
                  if k.startswith("cpu|avx512-picfc") and "fused" not in k), default=None)
        fu = min((v for k, v in ms.items() if "picfc-fused" in k), default=None)
        cand = [v for v in (py, fu) if v]
        best = min(cand) if cand else float("nan")
        cpu.append(f"{r['D']}{mark(r)} & {r['L']} & {base:.1f} & "
                   f"{(f'{py:.2f}' if py else '--')} & {(f'{fu:.2f}' if fu else '--')} & "
                   f"{base/best:.2f}$\\times$ \\\\")
    w("tab_p4_cpu.tex", "\n".join([
        r"\begin{tabular}{rrrrrr}", r"\toprule",
        r"$D$ & $L$ & AVX-512 seq. & $\pi$-CfC (py-driven) & $\pi$-CfC (fused C) & best speedup \\",
        r"\midrule", *cpu, r"\bottomrule", r"\end{tabular}", ""]))
    print("peak RssAnon MB:", d.get("peak_rss_anon_mb"))


def tab_amp(path=f"{RES}/phase3_expressivity.json"):
    d = json.load(open(path))["amplification"]
    body = []
    for mk in ("A_oscillator", "B_parity", "C_largescale"):
        rows = d[mk]
        for want in (0.0, 0.5):
            row = next((r for r in rows if abs(r["rho"] - want) < 1e-9), None)
            if row is None or not row["horizons"]:
                continue
            h = row["horizons"][-1]
            body.append(f"{mk.replace('_','-')} & {row['rho']:g} & "
                        f"{row['eps_step_local']:.3f} / {row['eps_step_exact']:.3f} & "
                        f"{row['abar']:.3f} & {row['G_inf']:.1f} & {h['eps_traj']:.3f} & "
                        f"[{h['eps_traj_pred_quad']:.3f}, " + "\\," + f" {h['eps_traj_pred_linear']:.3f}] & "
                        + ("in" if h["in_bounds"] else "out") + r" \\")
    w("tab_amp.tex", "\n".join([
        r"\begin{tabular}{lrrrrrrl}", r"\toprule",
        r"model & $\rho$ & $\varepsilon_{\text{step}}$ loc./exact & $\abar$ & $G_\infty$ & "
        r"$\varepsilon_{\text{traj}}$ & bounds & \\", r"\midrule", *body,
        r"\bottomrule", r"\end{tabular}", ""]))


if __name__ == "__main__":
    tab_training(); tab_iso(); tab_phase2(); tab_ridge(); tab_ridge_gpu(); tab_expr(); tab_osc(); tab_gate()
    try:
        tab_amp()
    except KeyError:
        print("(no amplification section yet)")
    if os.path.exists(f"{RES}/phase4_hardware.json"):
        tab_p4()
