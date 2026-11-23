"""ctypes bindings for kernels/libpicfc.so (AVX-512 + OpenMP, Zen 5).

The three primitives are deliberately narrow: the batched input projection and the
exogenous cold->hot projection stay in torch (they are dense GEMMs, and the point of the
paper is that they *can* stay dense because they are parallel), while the two recurrences
are the parts that must be hand-written.

    from kernels.hybrid import load
    lib = load()
    lib.cold_scan(out, gf, gg, gu, cf, cg, cu, dt, B, T, C, CH)
"""
import ctypes
import os

import numpy as np

LIB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "libpicfc.so")
_F = ctypes.POINTER(ctypes.c_float)


def load(path=LIB_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} missing; build it with\n"
            "  g++ -O3 -march=znver5 -mavx512f -mavx512bw -mavx512dq -mavx512vl -mfma "
            "-fopenmp -shared -fPIC -o kernels/libpicfc.so kernels/avx512_picfc.cpp")
    lib = ctypes.CDLL(path)
    lib.seq_cfc.restype = None
    lib.seq_cfc.argtypes = [_F] * 8 + [ctypes.c_int] * 3    # out + 7 inputs + B,T,D
    lib.cold_scan.restype = None
    lib.cold_scan.argtypes = [_F] * 8 + [ctypes.c_int] * 4
    lib.cold_coeff.restype = None
    lib.cold_coeff.argtypes = [_F] * 6 + [ctypes.c_int] * 3
    lib.prefix_scan.restype = None
    lib.prefix_scan.argtypes = [_F] * 3 + [ctypes.c_int] * 4
    lib.hot_recur.restype = None
    lib.hot_recur.argtypes = [_F] * 8 + [ctypes.c_int] * 3
    # one-shot fused pipeline: out, gf, gg, gu, hbar, Uf, Ug, Uu, dt, order, B, T, D, n_cold, CH
    lib.picfc_forward_fused.restype = ctypes.c_int
    lib.picfc_forward_fused.argtypes = ([_F] * 9
                                        + [ctypes.POINTER(ctypes.c_int)]
                                        + [ctypes.c_int] * 5)
    return lib


def ptr(t):
    """float32 tensor or array -> ctypes float pointer (no copy for contiguous input)."""
    if hasattr(t, "detach"):
        t = t.detach().cpu().contiguous().numpy()
    a = np.ascontiguousarray(t, dtype=np.float32)
    return a.ctypes.data_as(_F), a      # keep `a` alive for the duration of the call


def cpu_fused(lib, gf, gg, gu, hbar, U, dt, order, n_cold, B, T, D, CH=64):
    """Run the whole pi-CfC forward in one C call (cold coefficients + scan + exogenous
    cold->hot GEMM + hot recurrence + scatter).  Inputs are the *permuted* tensors, exactly
    as `picfc` uses them; `order` maps permuted index -> original index."""
    import torch
    out = torch.empty(B, T, D, dtype=torch.float32)
    po, ko = ptr(out)
    p1, k1 = ptr(gf)
    p2, k2 = ptr(gg)
    p3, k3 = ptr(gu)
    ph, kh = ptr(hbar)
    pu1, k4 = ptr(U["f"])
    pu2, k5 = ptr(U["g"])
    pu3, k6 = ptr(U["u"])
    pdt, k7 = ptr(dt.reshape(-1))
    ord_np = np.ascontiguousarray(order.detach().cpu().numpy() if hasattr(order, "detach")
                                  else order, dtype=np.int32)
    pord = ord_np.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    rc = lib.picfc_forward_fused(po, p1, p2, p3, ph, pu1, pu2, pu3, pdt, pord,
                                 B, T, D, n_cold, CH)
    if rc != 0:
        raise RuntimeError(f"picfc_forward_fused returned {rc} (n_cold or n_hot <= 0)")
    keep = (ko, k1, k2, k3, kh, k4, k5, k6, k7, ord_np)   # keep ptr targets alive
    del keep
    return out.clone()
