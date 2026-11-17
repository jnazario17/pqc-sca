# Side-Channel Leakage Ranking: ML-KEM-768 Secret-Dependent Intermediates
**Target Algorithm:** `ML_KEM_768`  
**Leakage Model:** `hamming_weight` (Hamming Weight)  
**Simulation Parameters:** SNR = `20.0 dB`, N = `1000` traces (`500` fixed, `500` random), Random Seed = `42`, TVLA Threshold = `4.5`  
**Total Trace Samples per Execution:** `11264`  
**Evaluation Runtime:** `22.79 s`  

## Executive Summary
This report analyzes and ranks the five critical secret-dependent intermediates and their corresponding variables in the ML-KEM-768 decapsulation pipeline according to their physical side-channel vulnerability.

Two complementary metrics are evaluated:
1. **TVLA (Test Vector Leakage Assessment - Welch's t-test):** Evaluates statistical distinguishability between fixed and random secret inputs. Ranked by **max |t|** and leaking sample count ($|t| > 4.5$).
2. **CPA (Correlation Power Analysis):** Evaluates attack exploitability by correlating the true secret-dependent intermediate values against power traces. Ranked by **peak Pearson correlation |ρ|** and Traces to Disclosure (TTD).

## 1. Intermediate Ranking by TVLA Leakage (Max |t| Descending)

| TVLA Rank | Intermediate | Location | Max |t| | Leaking Samples (>4.5) | Leaking % | Mean |t| | Status |
|:---------:|:-------------|:---------|:-------:|:-----------------------:|:---------:|:--------:|:------:|
| **#1** | **MultiplyNTTs** | `_inner_product_ntt` | **79.74** | 1303/1536 | 84.8% | 18.52 | `CRITICAL` |
| **#2** | **ByteDecode_12** | `_kpke_decrypt` | **73.26** | 1650/1920 | 85.9% | 17.67 | `CRITICAL` |
| **#3** | **SHA3-512** | `decaps_internal: _G` | **67.35** | 90/128 | 70.3% | 16.63 | `CRITICAL` |
| **#4** | **NTT^-1** | `_kpke_decrypt: intt_kem` | **66.37** | 989/1152 | 85.9% | 18.33 | `CRITICAL` |
| **#5** | **SHAKE256** | `decaps_internal: _J` | **64.47** | 837/1152 | 72.7% | 16.81 | `CRITICAL` |

## 2. Intermediate Ranking by CPA Attack Exploitability (Peak |ρ| Descending)

| CPA Rank | Intermediate | Location | Peak |ρ| | Mean |ρ| | TTD (Traces to Disclosure) | Attack Viability |
|:--------:|:-------------|:---------|:--------:|:---------:|:--------------------------:|:----------------:|
| **#1** | **ByteDecode_12** | `_kpke_decrypt` | **0.9962** | 0.9926 | **50** | `VERY HIGH` |
| **#2** | **MultiplyNTTs** | `_inner_product_ntt` | **0.9959** | 0.9943 | **50** | `VERY HIGH` |
| **#3** | **NTT^-1** | `_kpke_decrypt: intt_kem` | **0.9959** | 0.9943 | **50** | `VERY HIGH` |
| **#4** | **SHAKE256** | `decaps_internal: _J` | **0.9942** | 0.9925 | **50** | `VERY HIGH` |
| **#5** | **SHA3-512** | `decaps_internal: _G` | **0.9939** | 0.9925 | **50** | `VERY HIGH` |

## 3. Fine-Grained Variable Ranking (Input vs. Intermediate vs. Output)

| Rank (CPA) | Variable | Intermediate | Role & Data Type | Samples | Max |t| | Peak |ρ| | TTD |
|:----------:|:---------|:-------------|:-----------------|:-------:|:-------:|:--------:|:---:|
| #1 | `s_hat (output)` | **ByteDecode_12** | Output: s_hat[i] coefficients (uint12 (coeffs mod q=3329)) | 768 | 73.26 | **0.9962** | 50 |
| #2 | `prod (per i)` | **MultiplyNTTs** | Intermediate: base-case product h[2j], h[2j+1] (uint12 (coeffs mod q=3329)) | 768 | 68.70 | **0.9959** | 50 |
| #3 | `kem_intt (butterflies)` | **NTT^-1** | Intermediate: 7-layer GS butterfly transformations (uint12 (coeffs mod q=3329)) | 896 | 66.37 | **0.9959** | 50 |
| #4 | `sv (output)` | **NTT^-1** | Output: scaled inverse NTT polynomial sv (uint12 (coeffs mod q=3329)) | 256 | 61.26 | **0.9957** | 50 |
| #5 | `acc` | **MultiplyNTTs** | Accumulator across polynomials (sum s_hat[i] * u_hat[i]) (uint12 (coeffs mod q=3329)) | 768 | 79.74 | **0.9956** | 50 |
| #6 | `dk_bytes (input)` | **ByteDecode_12** | Input: dk[i*384:(i+1)*384] (uint8 (bytes 0..255)) | 1152 | 63.04 | **0.9943** | 50 |
| #7 | `z + c (input)` | **SHAKE256** | Input: 32-byte secret rejection key z + 1088-byte ciphertext c (uint8 (bytes 0..255)) | 1120 | 64.47 | **0.9942** | 50 |
| #8 | `m_prime + h (input)` | **SHA3-512** | Input: 32-byte decrypted message m' + 32-byte hash h (uint8 (bytes 0..255)) | 64 | 67.35 | **0.9939** | 50 |
| #9 | `K_bar (output)` | **SHAKE256** | Output: 32-byte rejected shared secret K_bar (uint8 (bytes 0..255)) | 32 | 44.45 | **0.9938** | 50 |
| #10 | `K_prime, r_prime (output)` | **SHA3-512** | Output: 32-byte K' + 32-byte r' (uint8 (bytes 0..255)) | 64 | 50.61 | **0.9937** | 50 |

## 4. Architectural Vulnerability Insights & Technical Discussion

### Key Findings:
- **ByteDecode_12:** Processes the raw secret key bytes `dk[i*384:(i+1)*384]` directly from memory into 12-bit NTT coefficients `s_hat[i]`. Because these values have zero diffusion and directly represent secret key material, they exhibit extremely high signal-to-noise ratio in Hamming weight leakage, leading to rapid key recovery.
- **MultiplyNTTs:** Computes the polynomial multiplication $s\_hat[i] \circ u\_hat[i]$ in the NTT domain. The products `prod` and running accumulations `acc` combine secret key polynomials with ciphertext polynomials, making them prime targets for correlation power analysis attacks targeting secret coefficients.
- **NTT^-1:** Performs the 7 GS butterfly stages and final inverse modular scaling to produce `sv`. Across the 896 butterfly operations, leakage accumulates across layers, yielding sustained TVLA threshold violations across a wide time window.
- **SHA3-512 ($G$):** Absorbs $m' \parallel h$ and squeezes $K' \parallel r'$. Both the input message $m'$ and output keys $K', r'$ produce localized, intense leakage spikes.
- **SHAKE256 ($J$):** Absorbs $z \parallel c$ to produce the implicit rejection key $K\_bar$. The secret seed $z$ leaks during absorption.

### Recommended Mitigations:
1. **First-order Boolean & Arithmetic Masking:** Split `s_hat` into two additive shares before `ByteDecode_12` and convert shares algebraically for `MultiplyNTTs`.
2. **Loop Shuffling & Jitter:** Randomize the evaluation order of polynomial indices $i \in \{0, 1, 2\}$ and NTT butterflies to decorrelate power traces from time samples.
3. **Masked Keccak / SHA-3:** Implement masked sponge absorption for $G$ and $J$ to protect against plain text and rejection seed leakage.
