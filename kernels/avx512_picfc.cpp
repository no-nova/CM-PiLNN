// kernels/avx512_picfc.cpp -- π-CfC execution primitives for the Ryzen 9 9955HX (Zen 5).
//
// Three primitives, all FP32, all OpenMP-parallel, exported through a C ABI so the Python
// benchmark can call them with ctypes:
//
//   seq_cfc       full sequential CfC (state in registers, dense U)
//   cold_scan     chunked two-pass parallel scan of the cold channels (h_t = a_t h + b_t)
//   hot_recur     compact sequential recurrence over the hot channels
//
// Build:
//   g++ -O3 -march=znver5 -mavx512f -mavx512bw -mavx512dq -mavx512vl -mfma -fopenmp \
//       -shared -fPIC -o kernels/libpicfc.so kernels/avx512_picfc.cpp
#include <immintrin.h>
#include <omp.h>
#include <cmath>
#include <cstring>
#include <vector>

static inline float sigmoidf_(float z) { return 1.0f / (1.0f + std::exp(-z)); }
static inline float softplusf_(float z) { return z > 20.0f ? z : std::log1p(std::exp(z)); }
static inline float tanhf_(float z) { return std::tanh(z); }

// horizontal sum of a __m512 register
static inline float hsum512(__m512 v) {
    return _mm512_reduce_add_ps(v);
}

// dot product of two length-n float arrays (AVX-512, masked tail)
static inline float dot_f32(const float* a, const float* b, int n) {
    __m512 acc = _mm512_setzero_ps();
    int i = 0;
    for (; i + 16 <= n; i += 16)
        acc = _mm512_fmadd_ps(_mm512_loadu_ps(a + i), _mm512_loadu_ps(b + i), acc);
    if (i < n) {
        __mmask16 m = (__mmask16)((1u << (n - i)) - 1u);
        acc = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(m, a + i),
                              _mm512_maskz_loadu_ps(m, b + i), acc);
    }
    return hsum512(acc);
}

// ---------------------------------------------------------------------------------------
// full sequential CfC:  h_t = a_t h_{t-1} + (1 - a_t) tanh(g_t) sigmoid(u_t)
// gates: z = gx[t] + U h_{t-1}
// ---------------------------------------------------------------------------------------
extern "C" void seq_cfc(float* out, const float* gf, const float* gg, const float* gu,
                        const float* Uf, const float* Ug, const float* Uu,
                        const float* dt, int B, int T, int D) {
#pragma omp parallel for schedule(static)
    for (int b = 0; b < B; ++b) {
        std::vector<float> h(D, 0.0f), zf(D), zg(D), zu(D);
        const float* gfb = gf + (size_t)b * T * D;
        const float* ggb = gg + (size_t)b * T * D;
        const float* gub = gu + (size_t)b * T * D;
        float* ob = out + (size_t)b * T * D;
        for (int t = 0; t < T; ++t) {
            const float* rf = gfb + (size_t)t * D;
            const float* rg = ggb + (size_t)t * D;
            const float* ru = gub + (size_t)t * D;
            for (int i = 0; i < D; ++i) {
                zf[i] = rf[i] + dot_f32(Uf + (size_t)i * D, h.data(), D);
                zg[i] = rg[i] + dot_f32(Ug + (size_t)i * D, h.data(), D);
                zu[i] = ru[i] + dot_f32(Uu + (size_t)i * D, h.data(), D);
            }
            const float d = dt[(size_t)b * T + t];
#pragma omp simd
            for (int i = 0; i < D; ++i) {
                float a = std::exp(-softplusf_(zf[i]) * d);
                h[i] = a * h[i] + (1.0f - a) * tanhf_(zg[i]) * sigmoidf_(zu[i]);
                ob[(size_t)t * D + i] = h[i];
            }
        }
    }
}

// ---------------------------------------------------------------------------------------
// cold path: gates from x only (+ constant bias), then a chunked parallel scan.
// ---------------------------------------------------------------------------------------
static inline void cold_gate(float g1, float g2, float g3, float c1,
                             float c2, float c3, float d, float& a, float& b) {
    const float f = softplusf_(g1 + c1);
    a = std::exp(-f * d);
    b = (1.0f - a) * tanhf_(g2 + c2) * sigmoidf_(g3 + c3);
}

extern "C" void cold_scan(float* out, const float* gf, const float* gg, const float* gu,
                          const float* cf, const float* cg, const float* cu,
                          const float* dt, int B, int T, int C, int CH) {
    const int nch = (T + CH - 1) / CH;
    std::vector<float> P((size_t)B * nch * C), Q((size_t)B * nch * C);
    std::vector<float> hloc((size_t)B * T * C), cuma((size_t)B * T * C);

    // phase 1: per-chunk local scan (parallel over (b, chunk, channel))
#pragma omp parallel for collapse(3) schedule(static)
    for (int b = 0; b < B; ++b)
        for (int k = 0; k < nch; ++k)
            for (int c = 0; c < C; ++c) {
                const int t0 = k * CH, t1 = std::min(T, t0 + CH);
                float h = 0.0f, Pc = 1.0f, Qc = 0.0f;
                for (int t = t0; t < t1; ++t) {
                    const size_t idx = ((size_t)b * T + t) * C + c;
                    float a, bb;
                    cold_gate(gf[idx], gg[idx], gu[idx], cf[c], cg[c], cu[c],
                              dt[(size_t)b * T + t], a, bb);
                    h = a * h + bb;
                    Pc = Pc * a;
                    Qc = a * Qc + bb;
                    hloc[idx] = h;
                    cuma[idx] = Pc;
                }
                P[((size_t)b * nch + k) * C + c] = Pc;
                Q[((size_t)b * nch + k) * C + c] = Qc;
            }

    // phase 2 + 3: exclusive scan of chunk summaries, then carry fix-up
#pragma omp parallel for schedule(static)
    for (int b = 0; b < B; ++b) {
        std::vector<float> Pc(C, 1.0f), Qc(C, 0.0f), carry(C);
        for (int k = 0; k < nch; ++k) {
            const int t0 = k * CH, t1 = std::min(T, t0 + CH);
            for (int c = 0; c < C; ++c) carry[c] = Pc[c] * 0.0f + Qc[c];   // h0 = 0
            for (int t = t0; t < t1; ++t)
                for (int c = 0; c < C; ++c) {
                    const size_t idx = ((size_t)b * T + t) * C + c;
                    out[idx] = cuma[idx] * carry[c] + hloc[idx];
                }
            for (int c = 0; c < C; ++c) {
                const float p = P[((size_t)b * nch + k) * C + c];
                const float q = Q[((size_t)b * nch + k) * C + c];
                const float qn = p * Qc[c] + q;
                Pc[c] = Pc[c] * p;
                Qc[c] = qn;
            }
        }
    }
}

// ---------------------------------------------------------------------------------------
// cold coefficients a = exp(-softplus(z_f) dt), b = (1-a) tanh(z_g) sigmoid(z_u)
// g1..g3 are already full pre-activations (input projection + constant bias), dt is per step.
// Vectorised with #pragma omp simd so gcc emits libmvec calls.
// ---------------------------------------------------------------------------------------
extern "C" void cold_coeff(float* oa, float* ob, const float* g1, const float* g2,
                           const float* g3, const float* dtv, int B, int T, int C) {
#pragma omp parallel for schedule(static)
    for (int bb = 0; bb < B; ++bb) {
        const size_t base = (size_t)bb * T * C;
        for (int t = 0; t < T; ++t) {
            const float d = dtv[(size_t)bb * T + t];      // dt is per (batch, step)
            const size_t off = base + (size_t)t * C;
#pragma omp simd
            for (int c = 0; c < C; ++c) {
                const float a = std::exp(-softplusf_(g1[off + c]) * d);
                oa[off + c] = a;
                ob[off + c] = (1.0f - a) * tanhf_(g2[off + c]) * sigmoidf_(g3[off + c]);
            }
        }
    }
}

// ---------------------------------------------------------------------------------------
// Work-optimal parallel prefix scan of the affine monoid (a,b): (a1,b1)o(a2,b2)=(a1a2, a2b1+b2).
// Pass 1 walks CH-long chunks in parallel (vectorised 16 lanes) and records each chunk's
// composite (P,Q); pass 2 scans those nch summaries per (batch, lane-block); pass 3 rewrites
// each chunk with h_t = P_t * carry + hloc_t, recomputing P_t from `a` in cache.  Total work is
// 2x the sequential recurrence, but the free (1-rho)D lanes are spread over B*nch*(C/16) tasks.
// ---------------------------------------------------------------------------------------
extern "C" void prefix_scan(float* out, const float* a_in, const float* b_in, int B, int T,
                            int C, int CH) {
    const int nch = (T + CH - 1) / CH;
    const int cblk = (C + 15) / 16;
    std::vector<float> P((size_t)B * nch * C, 1.0f), Q((size_t)B * nch * C, 0.0f);
    std::vector<float> hloc((size_t)B * T * C, 0.0f);

#pragma omp parallel for collapse(3) schedule(static)
    for (int bb = 0; bb < B; ++bb)
        for (int k = 0; k < nch; ++k)
            for (int cb = 0; cb < cblk; ++cb) {
                const int c0 = cb * 16;
                const __mmask16 m =
                    (__mmask16)(c0 + 16 <= C ? 0xFFFF : ((1u << (C - c0)) - 1u));
                const int t0 = k * CH, t1 = std::min(T, t0 + CH);
                __m512 h = _mm512_setzero_ps(), pc = _mm512_set1_ps(1.0f), qc = _mm512_setzero_ps();
                for (int t = t0; t < t1; ++t) {
                    const size_t idx = ((size_t)bb * T + t) * C + c0;
                    const __m512 av = _mm512_maskz_loadu_ps(m, a_in + idx);
                    const __m512 bv = _mm512_maskz_loadu_ps(m, b_in + idx);
                    h = _mm512_fmadd_ps(av, h, bv);
                    pc = _mm512_mul_ps(pc, av);
                    qc = _mm512_fmadd_ps(av, qc, bv);
                    _mm512_mask_storeu_ps(hloc.data() + idx, m, h);
                }
                const size_t pi = ((size_t)bb * nch + k) * C + c0;
                _mm512_mask_storeu_ps(P.data() + pi, m, pc);
                _mm512_mask_storeu_ps(Q.data() + pi, m, qc);
            }

    // pass 2: exclusive scan of the chunk summaries (serial over <= T/CH chunks, parallel
    // over batch and lane blocks)
#pragma omp parallel for collapse(2) schedule(static)
    for (int bb = 0; bb < B; ++bb)
        for (int cb = 0; cb < cblk; ++cb) {
            const int c0 = cb * 16;
            const __mmask16 m = (__mmask16)(c0 + 16 <= C ? 0xFFFF : ((1u << (C - c0)) - 1u));
            __m512 carry = _mm512_setzero_ps();
            for (int k = 0; k < nch; ++k) {
                const size_t pi = ((size_t)bb * nch + k) * C + c0;
                const __m512 p = _mm512_maskz_loadu_ps(m, P.data() + pi);
                const __m512 q = _mm512_maskz_loadu_ps(m, Q.data() + pi);
                // store this chunk's exclusive carry, then fold in the chunk summary
                _mm512_mask_storeu_ps(P.data() + pi, m, carry);
                carry = _mm512_fmadd_ps(carry, p, q);
            }
        }

    // pass 3: apply the carry
#pragma omp parallel for collapse(3) schedule(static)
    for (int bb = 0; bb < B; ++bb)
        for (int k = 0; k < nch; ++k)
            for (int cb = 0; cb < cblk; ++cb) {
                const int c0 = cb * 16;
                const __mmask16 m =
                    (__mmask16)(c0 + 16 <= C ? 0xFFFF : ((1u << (C - c0)) - 1u));
                const size_t pi = ((size_t)bb * nch + k) * C + c0;
                const __m512 carry = _mm512_maskz_loadu_ps(m, P.data() + pi);
                const int t0 = k * CH, t1 = std::min(T, t0 + CH);
                __m512 pc = _mm512_set1_ps(1.0f);
                for (int t = t0; t < t1; ++t) {
                    const size_t idx = ((size_t)bb * T + t) * C + c0;
                    const __m512 av = _mm512_maskz_loadu_ps(m, a_in + idx);
                    pc = _mm512_mul_ps(pc, av);
                    const __m512 loc = _mm512_maskz_loadu_ps(m, hloc.data() + idx);
                    _mm512_mask_storeu_ps(out + idx, m, _mm512_fmadd_ps(pc, carry, loc));
                }
            }
}

// ---------------------------------------------------------------------------------------
// hot path: compact sequential recurrence over H channels (U: H x H, row major)
// ---------------------------------------------------------------------------------------
extern "C" void hot_recur(float* out, const float* zf, const float* zg, const float* zu,
                          const float* Uf, const float* Ug, const float* Uu,
                          const float* dt, int B, int T, int H) {
#pragma omp parallel for schedule(static)
    for (int b = 0; b < B; ++b) {
        std::vector<float> h(H, 0.0f), zfv(H), zgv(H), zuv(H);
        float* ob = out + (size_t)b * T * H;
        for (int t = 0; t < T; ++t) {
            const size_t base = ((size_t)b * T + t) * H;
            const float d = dt[(size_t)b * T + t];
            // the three gate pre-activations all use h_{t-1}, so they must be complete
            // before the state is advanced
            for (int i = 0; i < H; ++i) {
                zfv[i] = zf[base + i] + dot_f32(Uf + (size_t)i * H, h.data(), H);
                zgv[i] = zg[base + i] + dot_f32(Ug + (size_t)i * H, h.data(), H);
                zuv[i] = zu[base + i] + dot_f32(Uu + (size_t)i * H, h.data(), H);
            }
            // the elementwise update is vectorised over the hot channels (SVML); the serial
            // dependence is across t, not across channels
#pragma omp simd
            for (int i = 0; i < H; ++i) {
                const float a = std::exp(-softplusf_(zfv[i]) * d);
                h[i] = a * h[i] + (1.0f - a) * tanhf_(zgv[i]) * sigmoidf_(zuv[i]);
                ob[(size_t)t * H + i] = h[i];
            }
        }
    }
}

// ---------------------------------------------------------------------------------------
// single-entry fused π-CfC:  cold_coeff -> prefix_scan -> exogenous cold->hot GEMM ->
// hot_recur -> scatter back to the original channel order.  One C call per forward pass:
// no Python round trips between the stages (the benchmark measures the whole pipeline).
//   gf/gg/gu : B*T*D pre-activations, already in permuted channel order
//   U*       : D*D row-major, U[out*D + in]
//   order    : permutation, permuted index -> original index
// ---------------------------------------------------------------------------------------
extern "C" int picfc_forward_fused(float* out, const float* gf, const float* gg, const float* gu,
                                   const float* hbar, const float* Uf, const float* Ug,
                                   const float* Uu, const float* dtv, const int* order,
                                   int B, int T, int D, int n_cold, int CH) {
    const int n_hot = D - n_cold;
    const size_t NT = (size_t)B * T;
    if (n_cold <= 0 || n_hot <= 0) return -1;

    std::vector<float> gc(NT * n_cold), gc2(NT * n_cold), gc3(NT * n_cold);
    std::vector<float> a(NT * n_cold), b(NT * n_cold), hc(NT * n_cold);
    std::vector<float> cf(n_cold), cg(n_cold), cu(n_cold);
    std::vector<float> zh(NT * n_hot * 3), hh(NT * n_hot);

    // cold channel slices + the constant (mean-state) gate bias
#pragma omp parallel for schedule(static)
    for (int c = 0; c < n_cold; ++c) {
        float s1 = 0.f, s2 = 0.f, s3 = 0.f;
        for (int j = 0; j < D; ++j) {
            const float hj = hbar[j];
            s1 += hj * Uf[(size_t)c * D + j];
            s2 += hj * Ug[(size_t)c * D + j];
            s3 += hj * Uu[(size_t)c * D + j];
        }
        cf[c] = s1; cg[c] = s2; cu[c] = s3;
    }
#pragma omp parallel for schedule(static)
    for (size_t r = 0; r < NT; ++r) {
        const float* s1 = gf + r * D; const float* s2 = gg + r * D; const float* s3 = gu + r * D;
        float* d1 = gc.data() + r * n_cold; float* d2 = gc2.data() + r * n_cold;
        float* d3 = gc3.data() + r * n_cold;
        for (int c = 0; c < n_cold; ++c) { d1[c] = s1[c] + cf[c]; d2[c] = s2[c] + cg[c]; d3[c] = s3[c] + cu[c]; }
    }
    cold_coeff(a.data(), b.data(), gc.data(), gc2.data(), gc3.data(), dtv, B, T, n_cold);
    prefix_scan(hc.data(), a.data(), b.data(), B, T, n_cold, CH);

    // exogenous cold -> hot contribution, one step delayed (h_{-1} = 0)
    const size_t per = n_cold;
#pragma omp parallel for schedule(static)
    for (size_t r = 0; r < NT; ++r) {
        const int bb = (int)(r / T), t = (int)(r % T);
        float* z1 = zh.data() + (r * n_hot) * 3;
        float* z2 = z1 + n_hot;
        float* z3 = z2 + n_hot;
        const float* sh = gf + r * D + n_cold;
        const float* sh2 = gg + r * D + n_cold;
        const float* sh3 = gu + r * D + n_cold;
        const float* prev = (t == 0) ? nullptr : hc.data() + (((size_t)bb * T + t - 1) * per);
        for (int e = 0; e < n_hot; ++e) {
            float s1 = 0.f, s2 = 0.f, s3 = 0.f;
            if (prev) {
                const float* w1 = Uf + (size_t)(n_cold + e) * D;
                const float* w2 = Ug + (size_t)(n_cold + e) * D;
                const float* w3 = Uu + (size_t)(n_cold + e) * D;
                s1 = dot_f32(w1, prev, n_cold);
                s2 = dot_f32(w2, prev, n_cold);
                s3 = dot_f32(w3, prev, n_cold);
            }
            z1[e] = sh[e] + s1;
            z2[e] = sh2[e] + s2;
            z3[e] = sh3[e] + s3;
        }
    }
    // hot recurrence: rows of zh are (z_f, z_g, z_u) triplets, so pass strided planes
    {
        std::vector<float> zf(NT * n_hot), zg(NT * n_hot), zu(NT * n_hot);
#pragma omp parallel for schedule(static)
        for (size_t r = 0; r < NT; ++r) {
            std::memcpy(zf.data() + r * n_hot, zh.data() + (r * n_hot) * 3, sizeof(float) * n_hot);
            std::memcpy(zg.data() + r * n_hot, zh.data() + (r * n_hot) * 3 + n_hot, sizeof(float) * n_hot);
            std::memcpy(zu.data() + r * n_hot, zh.data() + (r * n_hot) * 3 + 2 * n_hot, sizeof(float) * n_hot);
        }
        // compact hot-time-hot blocks: hot_recur expects dense H x H row-major
        std::vector<float> hUf((size_t)n_hot * n_hot), hUg((size_t)n_hot * n_hot),
            hUu((size_t)n_hot * n_hot);
#pragma omp parallel for collapse(2) schedule(static)
        for (int e = 0; e < n_hot; ++e)
            for (int f2 = 0; f2 < n_hot; ++f2) {
                const size_t src = (size_t)(n_cold + e) * D + (n_cold + f2);
                const size_t dst = (size_t)e * n_hot + f2;
                hUf[dst] = Uf[src]; hUg[dst] = Ug[src]; hUu[dst] = Uu[src];
            }
        hot_recur(hh.data(), zf.data(), zg.data(), zu.data(),
                  hUf.data(), hUg.data(), hUu.data(), dtv, B, T, n_hot);
    }

    // scatter to the original channel order
#pragma omp parallel for schedule(static)
    for (size_t r = 0; r < NT; ++r) {
        float* o = out + r * D;
        const float* ch = hc.data() + r * n_cold;
        const float* ho = hh.data() + r * n_hot;
        for (int i = 0; i < n_cold; ++i) o[order[i]] = ch[i];
        for (int e = 0; e < n_hot; ++e) o[order[n_cold + e]] = ho[e];
    }
    return 0;
}
