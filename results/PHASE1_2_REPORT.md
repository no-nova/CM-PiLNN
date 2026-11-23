# CM-PiLNN — Phase 1 & 2 report

Every number below is produced by an executable script and logged to JSON (no hand-entered values).

## 1. Trained checkpoints (`results/checkpoints/`, log `results/phase1_training.json`)

| model | task | D | params | train len | eval len | eval MSE | zero-predictor MSE | task metric | wall |
|---|---|---|---|---|---|---|---|---|---|
| C_largescale | multiscale | 1024 | 3,169,288 | 192 | 1024 | 0.1339 | 0.7485 | 0.8211 ($R^2$) | 108 s |
| A_oscillator | oscillator | 256 | 201,222 | 192 | 512 | 0.5786 | 1.1797 | 0.5096 ($R^2$) | 431 s |
| B_parity | parity | 256 | 200,194 | 64 | 512 | 0.2640 | 0.5125 | 0.8621 (parity sign acc.) | 28 s |

Per-horizon $R^2$ on the evaluated trajectory (drift check):

| model | 0-32 | 32-64 | 64-128 | 128-256 | 256-1024 |
|---|---|---|---|---|---|
| C_largescale | +0.76 | +0.88 | +0.90 | +0.90 | +0.81 |
| A_oscillator | +0.52 | +0.60 | +0.62 | +0.63 | +0.41 |
| B_parity | +0.92 | +0.85 | +0.98 | +0.88 | +0.06 |

## 2. Theorem-1 isomorphism gate

| model | worst bilateral rel $L_2$ (FP64) | worst (FP32) | one-sided control (FP64) | verdict |
|---|---|---|---|---|
| A_oscillator | 5.66e-16 | 2.97e-07 | 0.980 | PASS |
| B_parity | 1.04e-15 | 4.56e-07 | 1.080 | PASS |
| C_largescale | 1.14e-15 | 3.29e-07 | 1.025 | PASS |

Permutation families covered: identity, 5 random, variance, weight-norm, jacobian-coact (10 total, executed in both precisions).

## 3. Residual-spectrum audit

| model | Gini($\mathcal{J}$) | top-25\% share of $\mathcal{J}$ | $R(0.25)$ | $R(0.5)$ | spectral radius (mean / max) |
|---|---|---|---|---|---|
| A_oscillator | 0.820 | 0.895 | 0.895 | 0.976 | 0.997 / 1.003 |
| B_parity | 0.916 | 0.979 | 0.979 | 0.999 | 0.992 / 1.000 |
| C_largescale | 0.741 | 0.834 | 0.834 | 0.943 | 1.006 / 1.014 |

## 4. Dynamical-equivalence ablation (relative $L_2$ of the output trajectory)

**A_oscillator** (task oscillator, exact metric 0.5096)

| method | ρ=0.0 | ρ=0.0625 | ρ=0.125 | ρ=0.1875 | ρ=0.25 | ρ=0.375 | ρ=0.5 | ρ=1.0 |
|---|---|---|---|---|---|---|---|---|
| pure-linear | 1.08e+00 | 1.08e+00 | 1.08e+00 | 1.08e+00 | 1.08e+00 | 1.08e+00 | 1.08e+00 | 1.08e+00 |
| native-prefix | 1.08e+00 | 1.09e+00 | 1.11e+00 | 1.11e+00 | 1.12e+00 | 1.12e+00 | 1.13e+00 | 0.00e+00 |
| random-pi | 1.08e+00 | 1.08e+00 | 1.08e+00 | 1.09e+00 | 1.09e+00 | 1.10e+00 | 1.11e+00 | 1.42e-07 |
| variance-pi | 1.08e+00 | 1.10e+00 | 1.14e+00 | 1.15e+00 | 1.16e+00 | 1.24e+00 | 1.15e+00 | 1.13e-07 |
| ours-jacobian-pi | 1.08e+00 | 1.12e+00 | 1.15e+00 | 1.16e+00 | 1.15e+00 | 1.22e+00 | 1.15e+00 | 1.70e-07 |
| ours-jacobian-picard1 | 1.08e+00 | 1.22e+00 | 1.25e+00 | 1.26e+00 | 1.26e+00 | 1.34e+00 | 1.22e+00 | 1.70e-07 |

| method | ρ=0.0 | ρ=0.0625 | ρ=0.125 | ρ=0.1875 | ρ=0.25 | ρ=0.375 | ρ=0.5 | ρ=1.0 |
|---|---|---|---|---|---|---|---|---|
| pure-linear | -0.1820 | -0.1820 | -0.1820 | -0.1820 | -0.1820 | -0.1820 | -0.1820 | -0.1820 |
| native-prefix | -0.1820 | -0.1949 | -0.2009 | -0.2040 | -0.2181 | -0.2153 | -0.2240 | 0.5096 |
| random-pi | -0.1820 | -0.1837 | -0.1862 | -0.1932 | -0.1997 | -0.2120 | -0.2120 | 0.5096 |
| variance-pi | -0.1820 | -0.1890 | -0.2219 | -0.2227 | -0.2265 | -0.3466 | -0.2482 | 0.5096 |
| ours-jacobian-pi | -0.1820 | -0.1974 | -0.2217 | -0.2207 | -0.2173 | -0.3124 | -0.2649 | 0.5096 |
| ours-jacobian-picard1 | -0.1820 | -0.3330 | -0.3537 | -0.3615 | -0.3584 | -0.4875 | -0.3671 | 0.5096 |

**B_parity** (task parity, exact metric 0.8621)

| method | ρ=0.0 | ρ=0.0625 | ρ=0.125 | ρ=0.1875 | ρ=0.25 | ρ=0.375 | ρ=0.5 | ρ=1.0 |
|---|---|---|---|---|---|---|---|---|
| pure-linear | 1.29e+00 | 1.29e+00 | 1.29e+00 | 1.29e+00 | 1.29e+00 | 1.29e+00 | 1.29e+00 | 1.29e+00 |
| native-prefix | 1.29e+00 | 1.28e+00 | 1.23e+00 | 1.25e+00 | 1.24e+00 | 1.20e+00 | 1.20e+00 | 0.00e+00 |
| random-pi | 1.29e+00 | 1.26e+00 | 1.23e+00 | 1.23e+00 | 1.21e+00 | 1.15e+00 | 1.15e+00 | 1.98e-07 |
| variance-pi | 1.29e+00 | 1.43e+00 | 1.44e+00 | 1.23e+00 | 1.24e+00 | 6.86e-01 | 6.64e-01 | 2.53e-07 |
| ours-jacobian-pi | 1.29e+00 | 1.15e+00 | 1.12e+00 | 1.22e+00 | 1.15e+00 | 6.36e-01 | 6.60e-01 | 1.49e-07 |
| ours-jacobian-picard1 | 1.29e+00 | 1.18e+00 | 1.14e+00 | 1.25e+00 | 1.19e+00 | 6.52e-01 | 6.57e-01 | 1.49e-07 |

| method | ρ=0.0 | ρ=0.0625 | ρ=0.125 | ρ=0.1875 | ρ=0.25 | ρ=0.375 | ρ=0.5 | ρ=1.0 |
|---|---|---|---|---|---|---|---|---|
| pure-linear | 0.5071 | 0.5071 | 0.5071 | 0.5071 | 0.5071 | 0.5071 | 0.5071 | 0.5071 |
| native-prefix | 0.5071 | 0.5071 | 0.5061 | 0.5054 | 0.5051 | 0.5034 | 0.5024 | 0.8621 |
| random-pi | 0.5071 | 0.5056 | 0.5046 | 0.5047 | 0.5041 | 0.5038 | 0.5042 | 0.8621 |
| variance-pi | 0.5071 | 0.5054 | 0.5066 | 0.4993 | 0.4971 | 0.8867 | 0.8877 | 0.8621 |
| ours-jacobian-pi | 0.5071 | 0.5010 | 0.5427 | 0.4736 | 0.5916 | 0.8325 | 0.8870 | 0.8621 |
| ours-jacobian-picard1 | 0.5071 | 0.5083 | 0.5442 | 0.4753 | 0.6035 | 0.8311 | 0.8867 | 0.8621 |

**C_largescale** (task multiscale, exact metric 0.8211)

| method | ρ=0.0 | ρ=0.0625 | ρ=0.125 | ρ=0.1875 | ρ=0.25 | ρ=0.375 | ρ=0.5 | ρ=1.0 |
|---|---|---|---|---|---|---|---|---|
| pure-linear | 8.65e-01 | 8.65e-01 | 8.65e-01 | 8.65e-01 | 8.65e-01 | 8.65e-01 | 8.65e-01 | 8.65e-01 |
| native-prefix | 8.65e-01 | 8.20e-01 | 7.77e-01 | 7.30e-01 | 6.93e-01 | 6.15e-01 | 5.31e-01 | 0.00e+00 |
| random-pi | 8.65e-01 | 8.21e-01 | 7.68e-01 | 7.30e-01 | 6.85e-01 | 5.96e-01 | 5.13e-01 | 7.97e-08 |
| variance-pi | 8.65e-01 | 7.84e-01 | 7.22e-01 | 6.41e-01 | 5.67e-01 | 4.18e-01 | 2.92e-01 | 7.98e-08 |
| ours-jacobian-pi | 8.65e-01 | 7.46e-01 | 6.66e-01 | 6.04e-01 | 5.57e-01 | 4.42e-01 | 3.47e-01 | 8.04e-08 |
| ours-jacobian-picard1 | 8.65e-01 | 6.13e-01 | 5.29e-01 | 4.93e-01 | 4.51e-01 | 3.58e-01 | 2.86e-01 | 8.04e-08 |

| method | ρ=0.0 | ρ=0.0625 | ρ=0.125 | ρ=0.1875 | ρ=0.25 | ρ=0.375 | ρ=0.5 | ρ=1.0 |
|---|---|---|---|---|---|---|---|---|
| pure-linear | 0.2159 | 0.2159 | 0.2159 | 0.2159 | 0.2159 | 0.2159 | 0.2159 | 0.2159 |
| native-prefix | 0.2159 | 0.2674 | 0.3143 | 0.3613 | 0.3993 | 0.4749 | 0.5482 | 0.8211 |
| random-pi | 0.2159 | 0.2680 | 0.3251 | 0.3676 | 0.4107 | 0.4937 | 0.5670 | 0.8211 |
| variance-pi | 0.2159 | 0.3054 | 0.3723 | 0.4591 | 0.5314 | 0.6625 | 0.7498 | 0.8211 |
| ours-jacobian-pi | 0.2159 | 0.3413 | 0.4174 | 0.4791 | 0.5254 | 0.6265 | 0.7028 | 0.8211 |
| ours-jacobian-picard1 | 0.2159 | 0.6298 | 0.6627 | 0.6797 | 0.6898 | 0.7364 | 0.7674 | 0.8211 |

### Area under the error curve (trapezoid over $\rho$, lower is better)

| model | method | AUC of rel. $L_2$ | AUC / native |
|---|---|---|---|
| A_oscillator | pure-linear | 1.0776 | 1.283 |
| A_oscillator | native-prefix | 0.8398 | 1.000 |
| A_oscillator | random-pi | 0.8240 | 0.981 |
| A_oscillator | variance-pi | 0.8682 | 1.034 |
| A_oscillator | ours-jacobian-pi | 0.8691 | 1.035 |
| A_oscillator | ours-jacobian-picard1 | 0.9358 | 1.114 |
| B_parity | pure-linear | 1.2894 | 1.405 |
| B_parity | native-prefix | 0.9175 | 1.000 |
| B_parity | random-pi | 0.8887 | 0.969 |
| B_parity | variance-pi | 0.7064 | 0.770 |
| B_parity | ours-jacobian-pi | 0.6525 | 0.711 |
| B_parity | ours-jacobian-picard1 | 0.6621 | 0.722 |
| C_largescale | pure-linear | 0.8654 | 1.802 |
| C_largescale | native-prefix | 0.4803 | 1.000 |
| C_largescale | random-pi | 0.4710 | 0.981 |
| C_largescale | variance-pi | 0.3580 | 0.745 |
| C_largescale | ours-jacobian-pi | 0.3688 | 0.768 |
| C_largescale | ours-jacobian-picard1 | 0.3055 | 0.636 |

## 4b. Peak host memory

| script | peak RssAnon | peak VmRSS |
|---|---|---|
| phase1_isomorphism_and_spectrum.py | 1206 MB | 1661 MB |
| phase2_dynamical_equivalence.py | 1159 MB | 1607 MB |

Budget is 1.5 GB of *anonymous* host memory; VmRSS additionally counts CUDA driver mappings and page cache.

## 5. Associative scan vs sequential recurrence

| model | ρ | FP32 rel $L_2$ | FP64 rel $L_2$ | gate |
|---|---|---|---|---|
| A_oscillator | 0.25 | 1.41e-07 | 2.60e-16 | pass |
| A_oscillator | 0.5 | 1.42e-07 | 2.59e-16 | pass |
| B_parity | 0.25 | 8.97e-08 | 1.69e-16 | pass |
| B_parity | 0.5 | 1.43e-07 | 2.57e-16 | pass |
| C_largescale | 0.25 | 1.62e-07 | 3.03e-16 | pass |
| C_largescale | 0.5 | 1.78e-07 | 3.32e-16 | pass |

