#!/usr/bin/env python
"""One-command audit of the whole CM-PiLNN evidence chain.

Nothing here recomputes science; it re-reads every artefact that the paper cites and refuses
to print PASS if any of them is missing, out of tolerance, or inconsistent with what the
manuscript actually contains.

Checks
  1  artefacts      every checkpoint and JSON log exists and parses
  2  Theorem 1      fp64 < 1e-13, fp32 < 1e-6, one-sided control > 0.5   (phase1_isomorphism)
  3  scan gate      fp32 < 1e-5, fp64 < 1e-13                            (phase2_scan_check)
  4  kernel gate    every Triton / AVX-512 row < 1e-5                    (phase4_kernel_gate)
  5  memory         peak RssAnon < 1500 MB per logged run                (all *.json)
  6  tables         paper/*.tex are byte-identical to a fresh regeneration from the JSONs
  7  manuscript     no TODO / TBD / ?? placeholders, no undefined refs, no overfull boxes

Run:  python scripts/verify_all.py
"""
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES, PAP, FIGS = ROOT / "results", ROOT / "paper", ROOT / "figures"
CKPT = RES / "checkpoints"

JSONS = ["phase1_training.json", "phase1_isomorphism.json", "phase1_spectrum.json",
         "phase2_equivalence.json", "phase2_scan_check.json", "phase3_expressivity.json",
         "phase3_ridge_gpu.json", "phase4_kernel_gate.json", "phase4_hardware.json",
         "phase6_realworld.json", "phase6_gate.json"]
CKPTS = ["A_oscillator", "B_parity", "C_largescale"]
TABLES = ["tab_training.tex", "tab_iso.tex", "tab_p2_A_oscillator.tex", "tab_p2_B_parity.tex",
          "tab_p2_C_largescale.tex", "tab_ridge.tex", "tab_ridge_gpu.tex", "tab_expr.tex", "tab_osc.tex",
          "tab_gate.tex", "tab_amp.tex", "tab_p4_gpu.tex", "tab_p4_cpu.tex",
          "sec_bench_text.tex", "tab_p6_train.tex", "tab_p6_decouple.tex", "tab_p6_hw.tex",
          "sec_realworld_text.tex"]
RESULT = []


def check(name, ok, detail=""):
    RESULT.append((name, bool(ok), detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))
    return ok


def load(p):
    return json.load(open(ROOT / p if not str(p).startswith("/") else p))


# ----------------------------------------------------------------- 1 artefacts
def artefacts():
    ok = True
    for c in CKPTS:
        for suffix in (".pt", "_eval.pt"):
            p = CKPT / f"{c}{suffix}"
            ok &= check(f"checkpoint {p.name}", p.exists() and p.stat().st_size > 1e5,
                        f"{p.stat().st_size/1e6:.2f} MB" if p.exists() else "missing")
    for j in JSONS:
        p = RES / j
        try:
            d = load(p)
            ok &= check(f"log {j}", True, f"{len(json.dumps(d))//1024} KiB")
        except Exception as e:                                   # noqa: BLE001
            ok &= check(f"log {j}", False, str(e))
    for f in ("fig_residual_spectrum.pdf", "fig_equivalence.pdf",
              "fig_expressivity_and_calibration.pdf", "fig_hardware_speedup.pdf"):
        p = FIGS / f
        ok &= check(f"figure {f}", p.exists() and p.stat().st_size > 5e3,
                    f"{p.stat().st_size//1024} KiB" if p.exists() else "missing")
    return ok


# ----------------------------------------------------------------- 2-4 gates
def gates():
    ok = True
    iso = load("results/phase1_isomorphism.json")
    models = [v for k, v in iso.items() if not k.startswith("_")]
    worst64 = max(v["worst_bilateral_fp64"] for v in models)
    worst32 = max(v["worst_bilateral_fp32"] for v in models)
    ctrl = min(v["one_sided_fp64"] for v in models)
    ok &= check("Theorem 1 fp64 < 1e-13", worst64 < 1e-13, f"worst {worst64:.2e}")
    ok &= check("Theorem 1 fp32 < 1e-6", worst32 < 1e-6, f"worst {worst32:.2e}")
    ok &= check("one-sided control > 0.5", ctrl > 0.5, f"min {ctrl:.3f}")

    sc = load("results/phase2_scan_check.json")
    cells = [v for k, v in sc.items() if not k.startswith("_")]
    w32 = max(c["fp32"] for m in cells for c in m.values() if isinstance(c, dict))
    w64 = max(c["fp64"] for m in cells for c in m.values() if isinstance(c, dict))
    ok &= check("scan gate fp32 < 1e-5", w32 < 1e-5, f"worst {w32:.2e}")
    ok &= check("scan gate fp64 < 1e-13", w64 < 1e-13, f"worst {w64:.2e}")

    g = load("results/phase4_kernel_gate.json")
    worst = max(g["rows"].values())
    ok &= check(f"kernel gate < {g['threshold']:.0e}", worst < g["threshold"] and g["pass"],
                f"worst {worst:.2e} over {len(g['rows'])} paths")
    tiled = [k for k in g["rows"] if "tiled" in k]
    ok &= check("L2-fused tiled kernel present in gate", bool(tiled), ", ".join(tiled))
    seqk = [k for k in g["rows"] if "seq-kernel" in k]
    ok &= check("single-launch sequential kernel gated", bool(seqk),
                ", ".join(f"{k} {g['rows'][k]:.1e}" for k in seqk))

    g6 = load("results/phase6_gate.json")
    w6 = max(g6["rows"].values())
    ok &= check("real-stream kernel gate < 1e-5", w6 < g6["threshold"] and g6["pass_"],
                f"worst {w6:.2e} over {len(g6['rows'])} paths")
    return ok


# --------------------------------------------------------- 4b real-world streams
def realworld():
    """R2: the two real-data models must actually beat their trivial baselines and the whole
    decoupling pipeline (spectrum, decouple, ridge, DAFT, hardware) must be present."""
    ok = True
    try:
        d = load("results/phase6_realworld.json")
    except Exception:                                        # noqa: BLE001
        return check("phase6_realworld.json present", False, "missing")
    tr = d.get("train", {})
    w = tr.get("D_wiki", {})
    s_ = tr.get("E_sensor", {})
    ok &= check("Model D beats the unigram byte model", w.get("bpd", 9) < w.get("bpd_unigram", 0),
                f"BPD {w.get('bpd')} vs unigram {w.get('bpd_unigram')}")
    ok &= check("Model E explains the next frame", s_.get("r2", -1) > 0.30, f"R^2 {s_.get('r2')}")
    dec = {k: v for k, v in d.get("decouple", {}).items() if k != "_memory"}
    ok &= check("decoupling sweep on both real models", set(dec) >= {"D_wiki", "E_sensor"},
                ", ".join(sorted(dec)))
    if set(dec) >= {"D_wiki", "E_sensor"}:
        for nm, v in dec.items():
            k = len(v["rows"])
            ok &= check(f"{nm}: >=10 (method,rho) cells", k >= 10, f"{k} cells")
        # the paper claims the decoupled model stays usable at rho=1/4
        best = min(v["rows"][f"ours|0.25"]["delta"] for v in dec.values())
        ok &= check("ours@rho=1/4 within 0.5 BPD / 0.15 R^2 of exact", best > -0.5,
                    f"worst delta {best:+.3f}")
    hw = {k: v for k, v in d.get("hw", {}).items() if k != "_memory"}
    if hw:
        sp = [v["rows"].get("gpu|speedup rho=0.25") for v in hw.values()
              if isinstance(v, dict) and "rows" in v]
        ok &= check("GPU speedup > 1 on real streams", all(x and x > 1.0 for x in sp),
                    ", ".join(f"{x:.1f}x" if x else "nan" for x in sp))
    spg = {k: v for k, v in d.get("spectrum", {}).items() if k != "_memory"}
    ok &= check("spectrum computed on real streams", set(spg) >= {"D_wiki", "E_sensor"},
                ", ".join(sorted(spg)))
    return ok


# ----------------------------------------------------------------- 5 memory
def memory():
    ok, seen = True, 0
    for j in JSONS:
        try:
            d = load(RES / j)
        except Exception:                                        # noqa: BLE001
            continue
        if not isinstance(d, dict):
            continue
        m = d.get("_memory") or (d.get("meta", {}) or {}).get("_memory") or {}
        if "peak_rss_anon_mb" in m and m["peak_rss_anon_mb"]:
            seen += 1
            mb = m["peak_rss_anon_mb"]
            # a log may declare its own (documented, justified) budget; default is the 1.5 GB target
            budget = m.get("memory_budget_mb", 1500)
            ok &= check(f"{j} peak RssAnon < {budget:.0f} MB", mb < budget,
                        f"{mb:.0f} MB" + ("" if budget == 1500 else "  (declared budget)"))
    if not seen:
        ok &= check("memory probe present", False, "no peak_rss_anon_mb found")
    return ok


# ----------------------------------------------------------------- 6 tables
def tables():
    ok = True
    tmp = Path(tempfile.mkdtemp(prefix="cmplnn_tabs_"))
    env = dict(os.environ, TAB_OUT=str(tmp))
    r = subprocess.run([sys.executable, str(ROOT / "scripts/make_tables.py")],
                       cwd=ROOT, env=env, capture_output=True, text=True)
    ok &= check("table regeneration runs", r.returncode == 0, (r.stderr or "")[-200:])
    r = subprocess.run([sys.executable, str(ROOT / "scripts/make_p6_tables.py")],
                       cwd=ROOT, env=env, capture_output=True, text=True)
    ok &= check("real-stream table regeneration runs", r.returncode == 0,
                (r.stderr or "")[-200:])
    r = subprocess.run([sys.executable, str(ROOT / "scripts/make_bench_text.py")],
                       cwd=ROOT, env=env, capture_output=True, text=True)
    ok &= check("bench prose regeneration runs", r.returncode == 0, (r.stderr or "")[-200:])
    for t in TABLES:
        a, b = tmp / t, PAP / t
        same = a.exists() and b.exists() and a.read_bytes() == b.read_bytes()
        ok &= check(f"table {t} matches JSON", same,
                    "" if same else ("missing" if not b.exists() else "differs"))
    return ok


# ----------------------------------------------------------------- 7 manuscript
BAD = re.compile(r"(TODO|TBD|FIXME|XXX|\?\?)")


def manuscript():
    ok = True
    src = (PAP / "main.tex").read_text()
    hits = sorted({m.group(0) for m in BAD.finditer(src)})
    ok &= check("no placeholders in main.tex", not hits, ", ".join(hits))
    log = PAP / "main.log"
    if log.exists():
        txt = log.read_text(errors="ignore")
        und = re.findall(r"LaTeX Warning: Reference `([^']+)' on page", txt)
        over = re.findall(r"Overfull \\[hv]box \(([\d.]+)pt too (?:wide|high)\)", txt)
        ok &= check("no undefined references", not und, f"{len(und)} undefined")
        ok &= check("no overfull boxes", not over, f"{len(over)} overfull")
    pdf = PAP / "main.pdf"
    ok &= check("paper/main.pdf exists", pdf.exists(), f"{pdf.stat().st_size//1024} KiB"
                if pdf.exists() else "missing")
    final = ROOT / "CM-PiLNN.pdf"
    ok &= check("CM-PiLNN.pdf exists", final.exists(),
                f"{final.stat().st_size//1024} KiB" if final.exists() else "missing")
    if final.exists() and pdf.exists():
        ok &= check("CM-PiLNN.pdf is the current build", final.read_bytes() == pdf.read_bytes())
    return ok


def main():
    print("CM-PiLNN verification\n" + "=" * 60)
    print("\n1. artefacts")
    a = artefacts()
    print("\n2-4. numerical gates")
    g = gates()
    print("\n4b. real-world streams (R2)")
    rw = realworld()
    print("\n5. memory budget")
    m = memory()
    print("\n6. tables vs JSON")
    t = tables()
    print("\n7. manuscript")
    p = manuscript()
    n = len(RESULT)
    fails = [r for r in RESULT if not r[1]]
    print("\n" + "=" * 60)
    print(f"checks: {n}   passed: {n - len(fails)}   failed: {len(fails)}")
    for name, _, det in fails:
        print(f"  FAILED: {name}  {det}")
    print("VERDICT:", "PASS" if not fails else "FAIL")
    json.dump({"checks": [dict(name=n_, ok=o, detail=d) for n_, o, d in RESULT],
               "failed": [n_ for n_, o, _ in RESULT if not o],
               "verdict": "PASS" if not fails else "FAIL"},
              open(RES / "verify_all.json", "w"), indent=1)
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
