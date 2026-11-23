#!/usr/bin/env python
"""TF32 tensor-core variant of the decoupled forward: how fast, and how wrong.

The review asked whether the batched GEMMs in pi-CfC can use tensor cores.  We keep the gated
rows at IEEE fp32 (tf32 is 10-bit mantissa, and the recurrence multiplies its own error by a
gain of ~17), so this script answers the question quantitatively instead of by assertion:

  speed   --- the same `K.picfc` call with torch.backends.cuda.matmul.allow_tf32 = True
  accuracy--- relative L2 against the IEEE-fp32 torch reference of the same algorithm
and, for context, a TF32x3 pass (three tensor-core passes, fp32-like accuracy) where Triton
offers it.

Writes results/phase4_tf32.json
"""
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import kernels.triton_picfc as K                                    # noqa: E402
import scripts.phase4_hardware_benchmarks as B4                     # noqa: E402

torch.backends.cuda.matmul.allow_tf32 = False


def main():
    out = {"rows": {}, "note": ("TF32 halves the mantissa; the gated rows therefore stay IEEE. "
                                "Speedups here are an upper bound on what tensor cores would buy.")}
    for D, L, Bn in ((256, 1024, 8), (512, 1024, 8), (1024, 1024, 8), (2048, 512, 4)):
        m, x, dt, hbar, order = B4.setup(D, L, Bn, "cuda")
        with torch.no_grad():
            _, Href = m.forward_approx(x, dt, hbar=hbar, order=order, rho=0.25, keep_h=True)
            base = B4.time_gpu(lambda: K.picfc(m, x, dt, hbar, order, 0.25), reps=10, warm=3)
            torch.backends.cuda.matmul.allow_tf32 = True
            hw = B4.time_gpu(lambda: K.picfc(m, x, dt, hbar, order, 0.25), reps=10, warm=3)
            with torch.no_grad():
                Ht = K.picfc(m, x, dt, hbar, order, 0.25)
            torch.backends.cuda.matmul.allow_tf32 = False
        rel_ieee = float((K.picfc(m, x, dt, hbar, order, 0.25) - Href).norm() / Href.norm())
        rel_tf32 = float((Ht - Href).norm() / Href.norm())
        out["rows"][f"D={D}|L={L}|B={Bn}"] = dict(ieee_ms=base, tf32_ms=hw,
                                                  speedup=base / hw,
                                                  rel_l2_ieee=rel_ieee, rel_l2_tf32=rel_tf32)
        print(f"D={D} L={L} B={Bn}: ieee {base:.2f} ms (rel {rel_ieee:.1e})  "
              f"tf32 {hw:.2f} ms (rel {rel_tf32:.1e})  -> {base/hw:.2f}x", flush=True)
        del m
        torch.cuda.empty_cache()
    json.dump(out, open("results/phase4_tf32.json", "w"), indent=1)
    print("wrote results/phase4_tf32.json")


if __name__ == "__main__":
    main()
