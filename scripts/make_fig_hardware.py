#!/usr/bin/env python
"""Hardware figures for the report: speedup vs sequence length (GPU/CPU) and the gate."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 8, "axes.grid": True, "grid.alpha": 0.3,
                     "axes.spines.top": False, "axes.spines.right": False})


def series(rows, D, B=8):
    out = {"L": [], "loop": [], "kern": [], "pic": [], "pic_rho": [], "hyb": []}
    for r in rows:
        if r.get("skipped") or r["D"] != D or r["B"] != B or "ms" not in r:
            continue
        ms = r["ms"]
        loop = ms["gpu|pytorch-seq-cfc"]
        kern = ms.get("gpu|triton-seq-cfc", ms.get("gpu|fused-seq-cfc"))
        best, bl, brho = 1e9, None, None
        for k, v in ms.items():
            if "triton-picfc" in k and "picard1" not in k and 0 < v < best:
                best, bl = v, k
                brho = float(k.split("rho=")[1])
        if bl is None:
            continue
        out["L"].append(r["L"]); out["loop"].append(loop); out["kern"].append(kern)
        out["pic"].append(best); out["pic_rho"].append(brho)
        out["hyb"].append("hybrid" in bl)
    return out


def cpu_series(rows, D, B=8):
    out = {"L": [], "loop": [], "kern": [], "pic": []}
    for r in rows:
        if r.get("skipped") or r["D"] != D or r["B"] != B or "ms" not in r:
            continue
        ms = r["ms"]
        best = min(v for k, v in ms.items() if k.startswith("cpu|avx512-picfc"))
        out["L"].append(r["L"])
        out["loop"].append(ms.get("cpu|pytorch-seq-cfc", ms["cpu|avx512-seq-cfc"]))
        out["kern"].append(ms["cpu|avx512-seq-cfc"]); out["pic"].append(best)
    return out


def main():
    d = json.load(open("results/phase4_hardware.json"))
    rows = d["rows"]
    fig, ax = plt.subplots(1, 3, figsize=(10.4, 3.1))

    for D, c in ((256, "#1f77b4"), (512, "#ff7f0e"), (1024, "#2ca02c"), (2048, "#d62728")):
        s = series(rows, D)
        if not s["L"]:
            continue
        sp = [l / p for l, p in zip(s["loop"], s["pic"])]
        ax[0].plot(s["L"], sp, "-", lw=1.0, color=c, label=f"$D={D}$")
        for L, y, hyb in zip(s["L"], sp, s["hyb"]):
            ax[0].plot([L], [y], "o", ms=3.4, color=c,
                       mfc=("none" if hyb else c), mew=(0.9 if hyb else 0))
    ax[0].set_xscale("log", base=2); ax[0].set_yscale("log")
    ax[0].axhline(1.0, color="k", lw=0.6, ls=":")
    ax[0].set_xlabel("sequence length $L$"); ax[0].set_ylabel("speedup vs PyTorch loop")
    ax[0].set_title("GPU: $\\pi$-CfC vs loop ($B=8$)")
    ax[0].plot([], [], "o", ms=3.4, color="k", label="reg./tiled kernel")
    ax[0].plot([], [], "o", ms=3.4, color="k", mfc="none", mew=0.9, label="hybrid fallback")

    s = series(rows, 256)
    if s["L"]:
        ax[1].plot(s["L"], s["loop"], "s--", color="#888888", ms=3, label="PyTorch loop")
        ax[1].plot(s["L"], s["kern"], "^-", color="#1f77b4", ms=3, label="Triton fused seq")
        ax[1].plot(s["L"], s["pic"], "o-", color="#d62728", ms=3, label="$\\pi$-CfC ($\\rho=1/4$)")
    ax[1].set_xscale("log", base=2); ax[1].set_yscale("log")
    ax[1].set_xlabel("sequence length $L$"); ax[1].set_ylabel("latency (ms)")
    ax[1].set_title("GPU absolute, $D=256$")
    ax[1].legend(frameon=False, fontsize=7)

    c = cpu_series(rows, 2048)
    if c["L"]:
        ax[2].plot(c["L"], c["kern"], "^-", color="#1f77b4", ms=3, label="AVX-512 seq.")
        ax[2].plot(c["L"], c["pic"], "o-", color="#d62728", ms=3, label="$\\pi$-CfC")
    ax[2].set_xscale("log", base=2); ax[2].set_yscale("log")
    ax[2].set_xlabel("sequence length $L$"); ax[2].set_ylabel("latency (ms)")
    ax[2].set_title("CPU, 16 cores, $D=2048$")
    ax[2].legend(frameon=False, fontsize=7)

    for a in ax:
        a.grid(True, which="both", alpha=0.25)
    ax[0].set_ylim(7.0, 260.0)
    ax[0].legend(frameon=False, fontsize=6.2, loc="upper left", ncol=2, handlelength=1.1,
                 columnspacing=0.9, labelspacing=0.3)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.30, top=0.85, wspace=0.30)
    fig.savefig("figures/fig_hardware_speedup.pdf")
    fig.savefig("figures/fig_hardware_speedup.png", dpi=170)
    print("wrote figures/fig_hardware_speedup.{pdf,png}")


if __name__ == "__main__":
    main()
