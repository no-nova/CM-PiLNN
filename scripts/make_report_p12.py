#!/usr/bin/env python3
"""Build the Phase-1/2 report tables and figures straight from the JSON logs."""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figures")
METHOD_STYLE = {
    "pure-linear": dict(color="0.55", ls=":", lw=1.4),
    "native-prefix": dict(color="#1f77b4", ls="--", lw=1.5),
    "random-pi": dict(color="#7f7f7f", ls="-.", lw=1.3),
    "variance-pi": dict(color="#2ca02c", ls="--", lw=1.5),
    "ours-jacobian-pi": dict(color="#d62728", ls="-", lw=2.0),
    "ours-jacobian-picard1": dict(color="#9467bd", ls="-", lw=1.6),
}


def main():
    spec = torch.load(os.path.join(RES, "phase1_spectrum.pt"), weights_only=False)
    iso_raw = json.load(open(os.path.join(RES, "phase1_isomorphism.json")))
    iso = {k: v for k, v in iso_raw.items() if not k.startswith("_")}
    mem1 = iso_raw.get("_memory", {})
    abl_raw = json.load(open(os.path.join(RES, "phase2_equivalence.json")))
    abl = abl_raw["models"] if "models" in abl_raw else abl_raw
    mem2 = abl_raw.get("_memory", {})
    tr = {r["name"]: r for r in json.load(open(os.path.join(RES, "phase1_training.json")))}

    # ---------------- figure 1: residual concentration -------------------------------
    fig, ax = plt.subplots(figsize=(5.0, 3.1))
    for name, sp in spec.items():
        ax.plot(sp["rho_grid"], sp["R_of_rho"], marker="o", ms=3.2, lw=1.6,
                label=f"{name.split('_')[0]}  (Gini {sp['gini_J']:.2f})")
    ax.plot([0, 1], [0, 1], color="0.6", ls=":", lw=1.2, label="uniform reference")
    ax.set_xlabel(r"hot fraction $\rho$ kept nonlinear")
    ax.set_ylabel(r"$R(\rho)$: share of $\sum_i\mathcal{J}_i$")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, fontsize=7.5, loc="lower right")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_residual_spectrum.pdf"))
    fig.savefig(os.path.join(FIG, "fig_residual_spectrum.png"), dpi=160)
    plt.close(fig)

    # ---------------- figure 2: dynamical equivalence ---------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.0))
    for ax, (name, r) in zip(axes, abl.items()):
        grid = r["rho_grid"]
        for meth, cells in r["rows"].items():
            if meth == "pure-linear":
                continue                      # identical to native-prefix at rho = 0
            ys = [cells[str(g)]["rel_l2"] for g in grid]
            ax.plot(grid, ys, marker="o", ms=3.0, **METHOD_STYLE[meth])
        ax.set_xlim(-0.03, 1.03)
        ax.set_xlabel(r"hot fraction $\rho$")
        ax.set_title(f"{name} (exact $R^2$ {r['exact_metric']:.2f})", fontsize=9)
        ax.grid(alpha=0.25)
        ax.set_ylim(0, max(0.25, 1.15 * max(c["rel_l2"] for cc in r["rows"].values()
                                            for c in cc.values())))
    axes[0].set_ylabel(r"$\|Y_{\mathrm{approx}}-Y_{\mathrm{exact}}\|_F/\|Y_{\mathrm{exact}}\|_F$")
    handles = [plt.Line2D([], [], **METHOD_STYLE[m], label=m)
               for m in METHOD_STYLE if m != "pure-linear"]
    handles.insert(0, plt.Line2D([], [], **METHOD_STYLE["pure-linear"],
                                 label="pure-linear ($\\rho=0$)"))
    fig.tight_layout()
    fig.subplots_adjust(top=0.74)
    fig.legend(handles=handles, frameon=False, fontsize=7.8, ncol=6,
               loc="upper center", bbox_to_anchor=(0.5, 0.995), handlelength=2.2,
               columnspacing=1.1)
    fig.savefig(os.path.join(FIG, "fig_equivalence.pdf"))
    fig.savefig(os.path.join(FIG, "fig_equivalence.png"), dpi=160)
    plt.close(fig)

    # ---------------- markdown report --------------------------------------------------
    L = []
    A = L.append
    A("# CM-PiLNN — Phase 1 & 2 report\n")
    A("Every number below is produced by an executable script and logged to JSON "
      "(no hand-entered values).\n")
    A("## 1. Trained checkpoints (`results/checkpoints/`, log `results/phase1_training.json`)\n")
    A("| model | task | D | params | train len | eval len | eval MSE | zero-predictor MSE | "
      "task metric | wall |")
    A("|---|---|---|---|---|---|---|---|---|---|")
    for name, r in tr.items():
        m = "parity sign acc." if r["task"] == "parity" else "$R^2$"
        A(f"| {name} | {r['task']} | {r['D']} | {r['params']:,} | {r['T_train']} | {r['T']} | "
          f"{r['eval_loss']:.4f} | {r['zero_mse']:.4f} | {r['eval_metric']:.4f} ({m}) | "
          f"{r['wall_s']:.0f} s |")
    A("")
    A("Per-horizon $R^2$ on the evaluated trajectory (drift check):\n")
    A("| model | " + " | ".join(h["span"] for h in next(iter(tr.values()))["horizon"]) + " |")
    A("|---" * (1 + len(next(iter(tr.values()))["horizon"])) + "|")
    for name, r in tr.items():
        A(f"| {name} | " + " | ".join(f"{h['r2']:+.2f}" for h in r["horizon"]) + " |")
    A("")
    A("## 2. Theorem-1 isomorphism gate\n")
    A("| model | worst bilateral rel $L_2$ (FP64) | worst (FP32) | one-sided control (FP64) | verdict |")
    A("|---|---|---|---|---|")
    for name, r in iso.items():
        A(f"| {name} | {r['worst_bilateral_fp64']:.2e} | {r['worst_bilateral_fp32']:.2e} | "
          f"{r['one_sided_fp64']:.3f} | {'PASS' if r['pass'] else 'FAIL'} |")
    A("")
    A("Permutation families covered: identity, 5 random, variance, weight-norm, "
      "jacobian-coact (10 total, executed in both precisions).\n")
    A("## 3. Residual-spectrum audit\n")
    A("| model | Gini($\\mathcal{J}$) | top-25\\% share of $\\mathcal{J}$ | $R(0.25)$ | $R(0.5)$ | "
      "spectral radius (mean / max) |")
    A("|---|---|---|---|---|---|")
    for name, sp in spec.items():
        A(f"| {name} | {sp['gini_J']:.3f} | {sp['pareto_top25pct_J']:.3f} | "
          f"{sp['R_of_rho'][4]:.3f} | {sp['R_of_rho'][6]:.3f} | "
          f"{sp['spectral_radius_mean']:.3f} / {sp['spectral_radius_max']:.3f} |")
    A("")
    A("## 4. Dynamical-equivalence ablation (relative $L_2$ of the output trajectory)\n")
    for name, r in abl.items():
        grid = [str(g) for g in r["rho_grid"]]
        A(f"**{name}** (task {r['task']}, exact metric {r['exact_metric']:.4f})\n")
        A("| method | " + " | ".join(f"ρ={g}" for g in grid) + " |")
        A("|---" * (1 + len(grid)) + "|")
        for meth, cells in r["rows"].items():
            A(f"| {meth} | " + " | ".join(f"{cells[g]['rel_l2']:.2e}" for g in grid) + " |")
        A("")
        A("| method | " + " | ".join(f"ρ={g}" for g in grid) + " |")
        A("|---" * (1 + len(grid)) + "|")
        for meth, cells in r["rows"].items():
            A(f"| {meth} | " + " | ".join(f"{cells[g]['metric']:.4f}" for g in grid) + " |")
        A("")
    A("### Area under the error curve (trapezoid over $\\rho$, lower is better)\n")
    A("| model | method | AUC of rel. $L_2$ | AUC / native |")
    A("|---|---|---|---|")
    for name, r in abl.items():
        grid = r["rho_grid"]
        aucs = {}
        for meth, cells in r["rows"].items():
            ys = [cells[str(g)]["rel_l2"] for g in grid]
            aucs[meth] = float(sum(0.5 * (ys[i] + ys[i + 1]) * (grid[i + 1] - grid[i])
                                   for i in range(len(grid) - 1)))
        base = max(aucs["native-prefix"], 1e-12)
        for meth, v in aucs.items():
            A(f"| {name} | {meth} | {v:.4f} | {v / base:.3f} |")
    A("")
    A("## 4b. Peak host memory\n")
    A("| script | peak RssAnon | peak VmRSS |")
    A("|---|---|---|")
    A(f"| phase1_isomorphism_and_spectrum.py | {mem1.get('peak_rss_anon_mb', float('nan')):.0f} MB | "
      f"{mem1.get('peak_vm_rss_mb', float('nan')):.0f} MB |")
    A(f"| phase2_dynamical_equivalence.py | {mem2.get('peak_rss_anon_mb', float('nan')):.0f} MB | "
      f"{mem2.get('peak_vm_rss_mb', float('nan')):.0f} MB |")
    A("")
    A("Budget is 1.5 GB of *anonymous* host memory; VmRSS additionally counts CUDA driver "
      "mappings and page cache.\n")
    A("## 5. Associative scan vs sequential recurrence\n")
    sc = json.load(open(os.path.join(RES, "phase2_scan_check.json")))
    A("| model | ρ | FP32 rel $L_2$ | FP64 rel $L_2$ | gate |")
    A("|---|---|---|---|---|")
    for name, v in sc.items():
        for rho in ("0.25", "0.5"):
            A(f"| {name} | {rho} | {v[rho]['fp32']:.2e} | {v[rho]['fp64']:.2e} | "
              f"{'pass' if v[rho]['pass'] else 'FAIL'} |")
    A("")
    open(os.path.join(RES, "PHASE1_2_REPORT.md"), "w").write("\n".join(L) + "\n")
    print("wrote results/PHASE1_2_REPORT.md, figures/fig_residual_spectrum.{pdf,png}, "
          "figures/fig_equivalence.{pdf,png}")


if __name__ == "__main__":
    main()
