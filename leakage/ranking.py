"""
Leakage ranking and comparative analysis for ML-KEM-768 secret-dependent intermediates.

Analyzes and ranks the 5 secret-dependent intermediates specified in instructions:
  1. ByteDecode_12 (location: _kpke_decrypt, input: dk[i*384:(i+1)*384], output: s_hat[i])
  2. MultiplyNTTs  (location: _inner_product_ntt, input: s_hat[i], u_hat[i], prod then acc)
  3. NTT^-1        (location: _kpke_decrypt: intt_kem, input: acc, output: sv)
  4. SHA3-512      (location: decaps_internal: _G(m_prime + h), input: m_prime + h, output: K_prime, r_prime)
  5. SHAKE256      (location: decaps_internal: _J(z + c), input: z + c, output: K_bar)

Computes:
  - TVLA (fixed vs random): max |t|, leaking samples count (|t| > 4.5), leaking %, mean |t|
  - CPA against true secret values: peak Pearson |corr|, mean |corr|, Traces to Disclosure (TTD)
  - Tabulated rankings (by max |t| and by CPA correlation descending)
"""

from __future__ import annotations
import os
import json
import csv
import time
import numpy as np
from typing import Optional, Dict, List, Any

from algorithms.mlkem import MLKEM
from algorithms.ntt import Tracer
from leakage.models import HammingWeightModel, _hw
from attacks.tvla import TVLA
from attacks.metrics import traces_to_disclosure
from config import DEFAULT_SNR_DB, RANDOM_SEED, TVLA_THRESHOLD


# ─── Intermediate & Variable Definitions ─────────────────────────────────────

INTERMEDIATE_SPECS = {
    "ByteDecode_12": {
        "name": "ByteDecode_12",
        "location": "_kpke_decrypt",
        "description": "Decodes 384-byte polynomial secret key segments into 256 12-bit NTT coefficients",
        "op_names": ["bytedecode_12_in", "bytedecode_12_out"],
        "variables": {
            "dk_bytes (input)": {
                "op_name": "bytedecode_12_in",
                "role": "Input: dk[i*384:(i+1)*384]",
                "data_type": "uint8 (bytes 0..255)",
                "bits": 8,
            },
            "s_hat (output)": {
                "op_name": "bytedecode_12_out",
                "role": "Output: s_hat[i] coefficients",
                "data_type": "uint12 (coeffs mod q=3329)",
                "bits": 12,
            },
        },
    },
    "MultiplyNTTs": {
        "name": "MultiplyNTTs",
        "location": "_inner_product_ntt",
        "description": "Base-multiplication and vector accumulation in NTT domain (s_hat[i] * u_hat[i])",
        "op_names": ["multiply_ntts_prod", "multiply_ntts_acc"],
        "variables": {
            "prod (per i)": {
                "op_name": "multiply_ntts_prod",
                "role": "Intermediate: base-case product h[2j], h[2j+1]",
                "data_type": "uint12 (coeffs mod q=3329)",
                "bits": 12,
            },
            "acc": {
                "op_name": "multiply_ntts_acc",
                "role": "Accumulator across polynomials (sum s_hat[i] * u_hat[i])",
                "data_type": "uint12 (coeffs mod q=3329)",
                "bits": 12,
            },
        },
    },
    "NTT^-1": {
        "name": "NTT^-1",
        "location": "_kpke_decrypt: intt_kem",
        "description": "Inverse NTT transforming inner product accumulator back to polynomial domain",
        "op_names": ["kem_intt", "kem_intt_sv"],
        "variables": {
            "kem_intt (butterflies)": {
                "op_name": "kem_intt",
                "role": "Intermediate: 7-layer GS butterfly transformations",
                "data_type": "uint12 (coeffs mod q=3329)",
                "bits": 12,
            },
            "sv (output)": {
                "op_name": "kem_intt_sv",
                "role": "Output: scaled inverse NTT polynomial sv",
                "data_type": "uint12 (coeffs mod q=3329)",
                "bits": 12,
            },
        },
    },
    "SHA3-512": {
        "name": "SHA3-512",
        "location": "decaps_internal: _G",
        "description": "Hash function G(m' || h) generating shared secret K' and pseudorandom coin r'",
        "op_names": ["sha3_512_in", "sha3_512_out"],
        "variables": {
            "m_prime + h (input)": {
                "op_name": "sha3_512_in",
                "role": "Input: 32-byte decrypted message m' + 32-byte hash h",
                "data_type": "uint8 (bytes 0..255)",
                "bits": 8,
            },
            "K_prime, r_prime (output)": {
                "op_name": "sha3_512_out",
                "role": "Output: 32-byte K' + 32-byte r'",
                "data_type": "uint8 (bytes 0..255)",
                "bits": 8,
            },
        },
    },
    "SHAKE256": {
        "name": "SHAKE256",
        "location": "decaps_internal: _J",
        "description": "XOF function J(z || c) generating fallback rejection shared key K_bar",
        "op_names": ["shake256_in", "shake256_out"],
        "variables": {
            "z + c (input)": {
                "op_name": "shake256_in",
                "role": "Input: 32-byte secret rejection key z + 1088-byte ciphertext c",
                "data_type": "uint8 (bytes 0..255)",
                "bits": 8,
            },
            "K_bar (output)": {
                "op_name": "shake256_out",
                "role": "Output: 32-byte rejected shared secret K_bar",
                "data_type": "uint8 (bytes 0..255)",
                "bits": 8,
            },
        },
    },
}


# ─── Core Analysis Runner ────────────────────────────────────────────────────

class LeakageRankingAnalyzer:
    """Orchestrates trace acquisition, TVLA, CPA, and ranking across intermediates."""

    def __init__(
        self,
        algorithm: str = "ML_KEM_768",
        model_name: str = "hamming_weight",
        n_traces: int = 1000,
        snr_db: float = DEFAULT_SNR_DB,
        seed: int = RANDOM_SEED,
        tvla_threshold: float = TVLA_THRESHOLD,
    ) -> None:
        self.algorithm = algorithm
        self.model_name = model_name
        self.n_traces = n_traces
        self.snr_db = snr_db
        self.seed = seed
        self.tvla_threshold = tvla_threshold
        self.kem = MLKEM(algorithm)

    def run(self) -> dict:
        """Run full evaluation and return complete ranking results."""
        t_start = time.time()

        # 1. Trace op map (sample index -> op_name)
        tracer = Tracer()
        ek, dk = self.kem.keygen()
        K, ct = self.kem.encaps(ek)
        self.kem.decaps_internal(dk, ct, tracer=tracer)
        sample_ops = [entry[0] for entry in tracer.log]
        total_samples = len(sample_ops)

        # Build op_name -> list of sample indices
        op_indices: Dict[str, list[int]] = {}
        for idx, op in enumerate(sample_ops):
            op_indices.setdefault(op, []).append(idx)

        # 2. TVLA: Fixed vs Random traces
        half = self.n_traces // 2
        rng_fixed = np.random.default_rng(self.seed)
        rng_rand = np.random.default_rng(self.seed + 1)
        model_fixed = HammingWeightModel(snr_db=self.snr_db, seed=self.seed)
        model_rand = HammingWeightModel(snr_db=self.snr_db, seed=self.seed + 1)

        # Fixed key and fixed ciphertext
        fixed_ek, fixed_dk = self.kem.keygen()
        fixed_m = os.urandom(32)
        fixed_K, fixed_ct = self.kem.encaps_internal(fixed_ek, fixed_m)

        fixed_traces = np.zeros((half, total_samples), dtype=np.float64)
        for i in range(half):
            t = Tracer()
            self.kem.decaps_internal(fixed_dk, fixed_ct, tracer=t)
            fixed_traces[i] = model_fixed.compute_trace(t.log)

        # Random key and random ciphertext (also used for CPA true secret values)
        rand_traces = np.zeros((half, total_samples), dtype=np.float64)
        rand_raw_values = np.zeros((half, total_samples), dtype=np.int64)

        for i in range(half):
            ek_i, dk_i = self.kem.keygen()
            m_i = os.urandom(32)
            K_i, ct_i = self.kem.encaps_internal(ek_i, m_i)
            t = Tracer()
            self.kem.decaps_internal(dk_i, ct_i, tracer=t)
            rand_traces[i] = model_rand.compute_trace(t.log)
            rand_raw_values[i] = [entry[2] for entry in t.log]

        # TVLA t-statistic
        tvla = TVLA(fixed_traces, rand_traces, threshold=self.tvla_threshold)
        tvla_res = tvla.run()
        t_stat = tvla_res["t_statistic"]
        abs_t_stat = np.abs(t_stat)

        # 3. CPA against true secret-dependent values
        # For each sample t, H[i, t] = HW(rand_raw_values[i, t])
        hw_hypotheses = np.array([[_hw(int(v)) for v in row] for row in rand_raw_values], dtype=np.float64)

        # Vectorized Pearson correlation per sample t between hw_hypotheses[:, t] and rand_traces[:, t]
        h_mean = hw_hypotheses.mean(axis=0)
        h_std = hw_hypotheses.std(axis=0) + 1e-12
        h_centered = hw_hypotheses - h_mean

        t_mean = rand_traces.mean(axis=0)
        t_std = rand_traces.std(axis=0) + 1e-12
        t_centered = rand_traces - t_mean

        # Per-sample correlation (sample t against hypothesis t)
        cov_sample = np.mean(h_centered * t_centered, axis=0)
        sample_corr = cov_sample / (h_std * t_std)
        abs_sample_corr = np.abs(sample_corr)

        # 4. Compute metrics per intermediate and per variable
        intermediates_result = []
        variables_result = []

        step_ns = list(range(50, half + 1, 50))

        for int_key, spec in INTERMEDIATE_SPECS.items():
            int_indices: list[int] = []
            for op in spec["op_names"]:
                int_indices.extend(op_indices.get(op, []))
            int_indices = sorted(int_indices)

            # TVLA metrics for intermediate
            int_abs_t = abs_t_stat[int_indices]
            max_t = float(np.max(int_abs_t))
            mean_t = float(np.mean(int_abs_t))
            n_leaking = int(np.sum(int_abs_t > self.tvla_threshold))
            total_int_samples = len(int_indices)
            leaking_pct = float(100.0 * n_leaking / max(total_int_samples, 1))

            # CPA metrics for intermediate
            int_abs_corr = abs_sample_corr[int_indices]
            peak_corr = float(np.max(int_abs_corr))
            mean_corr = float(np.mean(int_abs_corr))

            # TTD (traces to disclosure at threshold 0.8) for the top-leaking sample in intermediate
            best_idx = int_indices[int(np.argmax(int_abs_corr))]
            best_h = hw_hypotheses[:, best_idx]
            best_tr = rand_traces[:, best_idx]
            n_corrs = []
            for n_sub in step_ns:
                bh = best_h[:n_sub]
                btr = best_tr[:n_sub]
                c = float(np.abs(np.corrcoef(bh, btr)[0, 1])) if bh.std() > 0 and btr.std() > 0 else 0.0
                n_corrs.append(c)
            ttd = traces_to_disclosure(step_ns, n_corrs, threshold=0.8)

            intermediates_result.append({
                "intermediate": spec["name"],
                "location": spec["location"],
                "description": spec["description"],
                "total_samples": total_int_samples,
                "max_t": round(max_t, 2),
                "n_leaking": n_leaking,
                "leaking_pct": round(leaking_pct, 1),
                "mean_t": round(mean_t, 2),
                "peak_corr": round(peak_corr, 4),
                "mean_corr": round(mean_corr, 4),
                "ttd": ttd if ttd is not None else ">500",
            })

            # Process individual variables
            for var_key, v_info in spec["variables"].items():
                v_op = v_info["op_name"]
                v_idx = sorted(op_indices.get(v_op, []))
                if not v_idx:
                    continue

                v_abs_t = abs_t_stat[v_idx]
                v_max_t = float(np.max(v_abs_t))
                v_mean_t = float(np.mean(v_abs_t))
                v_leaking = int(np.sum(v_abs_t > self.tvla_threshold))
                v_samples = len(v_idx)
                v_leaking_pct = float(100.0 * v_leaking / max(v_samples, 1))

                v_abs_corr = abs_sample_corr[v_idx]
                v_peak_corr = float(np.max(v_abs_corr))
                v_mean_corr = float(np.mean(v_abs_corr))

                v_best_idx = v_idx[int(np.argmax(v_abs_corr))]
                v_best_h = hw_hypotheses[:, v_best_idx]
                v_best_tr = rand_traces[:, v_best_idx]
                v_n_corrs = []
                for n_sub in step_ns:
                    bh = v_best_h[:n_sub]
                    btr = v_best_tr[:n_sub]
                    c = float(np.abs(np.corrcoef(bh, btr)[0, 1])) if bh.std() > 0 and btr.std() > 0 else 0.0
                    v_n_corrs.append(c)
                v_ttd = traces_to_disclosure(step_ns, v_n_corrs, threshold=0.8)

                variables_result.append({
                    "intermediate": spec["name"],
                    "variable": var_key,
                    "location": spec["location"],
                    "role": v_info["role"],
                    "data_type": v_info["data_type"],
                    "samples": v_samples,
                    "max_t": round(v_max_t, 2),
                    "n_leaking": v_leaking,
                    "leaking_pct": round(v_leaking_pct, 1),
                    "mean_t": round(v_mean_t, 2),
                    "peak_corr": round(v_peak_corr, 4),
                    "mean_corr": round(v_mean_corr, 4),
                    "ttd": v_ttd if v_ttd is not None else ">500",
                })

        # 5. Rankings
        # By max |t| descending
        ranked_by_tvla = sorted(intermediates_result, key=lambda x: x["max_t"], reverse=True)
        for i, item in enumerate(ranked_by_tvla, 1):
            item["tvla_rank"] = i

        # By CPA correlation descending
        ranked_by_cpa = sorted(intermediates_result, key=lambda x: x["peak_corr"], reverse=True)
        for i, item in enumerate(ranked_by_cpa, 1):
            item["cpa_rank"] = i

        # Variables ranked by max |t|
        var_ranked_by_tvla = sorted(variables_result, key=lambda x: x["max_t"], reverse=True)
        for i, item in enumerate(var_ranked_by_tvla, 1):
            item["tvla_rank"] = i

        # Variables ranked by CPA correlation
        var_ranked_by_cpa = sorted(variables_result, key=lambda x: x["peak_corr"], reverse=True)
        for i, item in enumerate(var_ranked_by_cpa, 1):
            item["cpa_rank"] = i

        elapsed = round(time.time() - t_start, 2)

        return {
            "algorithm": self.algorithm,
            "leakage_model": self.model_name,
            "n_traces": self.n_traces,
            "snr_db": self.snr_db,
            "seed": self.seed,
            "tvla_threshold": self.tvla_threshold,
            "total_trace_samples": total_samples,
            "execution_time_seconds": elapsed,
            "intermediates_by_tvla": ranked_by_tvla,
            "intermediates_by_cpa": ranked_by_cpa,
            "variables_by_tvla": var_ranked_by_tvla,
            "variables_by_cpa": var_ranked_by_cpa,
        }


# ─── File Generators ─────────────────────────────────────────────────────────

def generate_markdown_report(data: dict) -> str:
    """Generate professional GitHub Flavored Markdown report."""
    md = []
    md.append("# Side-Channel Leakage Ranking: ML-KEM-768 Secret-Dependent Intermediates\n")
    md.append(f"**Target Algorithm:** `{data['algorithm']}`  \n")
    md.append(f"**Leakage Model:** `{data['leakage_model']}` (Hamming Weight)  \n")
    md.append(f"**Simulation Parameters:** SNR = `{data['snr_db']} dB`, N = `{data['n_traces']}` traces (`{data['n_traces']//2}` fixed, `{data['n_traces']//2}` random), Random Seed = `{data['seed']}`, TVLA Threshold = `{data['tvla_threshold']}`  \n")
    md.append(f"**Total Trace Samples per Execution:** `{data['total_trace_samples']}`  \n")
    md.append(f"**Evaluation Runtime:** `{data['execution_time_seconds']} s`  \n\n")

    md.append("## Executive Summary\n")
    md.append("This report analyzes and ranks the five critical secret-dependent intermediates and their corresponding variables in the ML-KEM-768 decapsulation pipeline according to their physical side-channel vulnerability.\n\n")
    md.append("Two complementary metrics are evaluated:\n")
    md.append("1. **TVLA (Test Vector Leakage Assessment - Welch's t-test):** Evaluates statistical distinguishability between fixed and random secret inputs. Ranked by **max |t|** and leaking sample count ($|t| > 4.5$).\n")
    md.append("2. **CPA (Correlation Power Analysis):** Evaluates attack exploitability by correlating the true secret-dependent intermediate values against power traces. Ranked by **peak Pearson correlation |ρ|** and Traces to Disclosure (TTD).\n\n")

    # Table 1: Intermediates Ranked by TVLA (max |t|)
    md.append("## 1. Intermediate Ranking by TVLA Leakage (Max |t| Descending)\n\n")
    md.append("| TVLA Rank | Intermediate | Location | Max |t| | Leaking Samples (>4.5) | Leaking % | Mean |t| | Status |\n")
    md.append("|:---------:|:-------------|:---------|:-------:|:-----------------------:|:---------:|:--------:|:------:|\n")
    for r in data["intermediates_by_tvla"]:
        status = "CRITICAL" if r["max_t"] >= 15.0 else ("HIGH" if r["max_t"] >= 4.5 else "LOW")
        md.append(f"| **#{r['tvla_rank']}** | **{r['intermediate']}** | `{r['location']}` | **{r['max_t']:.2f}** | {r['n_leaking']}/{r['total_samples']} | {r['leaking_pct']:.1f}% | {r['mean_t']:.2f} | `{status}` |\n")
    md.append("\n")

    # Table 2: Intermediates Ranked by CPA Correlation (|ρ| Descending)
    md.append("## 2. Intermediate Ranking by CPA Attack Exploitability (Peak |ρ| Descending)\n\n")
    md.append("| CPA Rank | Intermediate | Location | Peak |ρ| | Mean |ρ| | TTD (Traces to Disclosure) | Attack Viability |\n")
    md.append("|:--------:|:-------------|:---------|:--------:|:---------:|:--------------------------:|:----------------:|\n")
    for r in data["intermediates_by_cpa"]:
        viab = "VERY HIGH" if r["peak_corr"] >= 0.90 else ("HIGH" if r["peak_corr"] >= 0.75 else "MEDIUM")
        md.append(f"| **#{r['cpa_rank']}** | **{r['intermediate']}** | `{r['location']}` | **{r['peak_corr']:.4f}** | {r['mean_corr']:.4f} | **{r['ttd']}** | `{viab}` |\n")
    md.append("\n")

    # Table 3: Fine-Grained Variable Leakage Breakdown
    md.append("## 3. Fine-Grained Variable Ranking (Input vs. Intermediate vs. Output)\n\n")
    md.append("| Rank (CPA) | Variable | Intermediate | Role & Data Type | Samples | Max |t| | Peak |ρ| | TTD |\n")
    md.append("|:----------:|:---------|:-------------|:-----------------|:-------:|:-------:|:--------:|:---:|\n")
    for r in data["variables_by_cpa"]:
        md.append(f"| #{r['cpa_rank']} | `{r['variable']}` | **{r['intermediate']}** | {r['role']} ({r['data_type']}) | {r['samples']} | {r['max_t']:.2f} | **{r['peak_corr']:.4f}** | {r['ttd']} |\n")
    md.append("\n")

    # Analysis & Discussion
    md.append("## 4. Architectural Vulnerability Insights & Technical Discussion\n\n")
    md.append("### Key Findings:\n")
    md.append("- **ByteDecode_12:** Processes the raw secret key bytes `dk[i*384:(i+1)*384]` directly from memory into 12-bit NTT coefficients `s_hat[i]`. Because these values have zero diffusion and directly represent secret key material, they exhibit extremely high signal-to-noise ratio in Hamming weight leakage, leading to rapid key recovery.\n")
    md.append("- **MultiplyNTTs:** Computes the polynomial multiplication $s_hat[i] * u_hat[i]$ in the NTT domain. The products `prod` and running accumulations `acc` combine secret key polynomials with ciphertext polynomials, making them prime targets for correlation power analysis attacks targeting secret coefficients.\n")
    md.append("- **NTT^-1:** Performs the 7 GS butterfly stages and final inverse modular scaling to produce `sv`. Across the 896 butterfly operations, leakage accumulates across layers, yielding sustained TVLA threshold violations across a wide time window.\n")
    md.append("- **SHA3-512 ($G$):** Absorbs $m' || h$ and squeezes $K' || r'$. Both the input message $m'$ and output keys $K', r'$ produce localized, intense leakage spikes.\n")
    md.append("- **SHAKE256 ($J$):** Absorbs $z || c$ to produce the implicit rejection key $K_bar$. The secret seed $z$ leaks during absorption.\n\n")

    md.append("### Recommended Mitigations:\n")
    md.append("1. **First-order Boolean & Arithmetic Masking:** Split `s_hat` into two additive shares before `ByteDecode_12` and convert shares algebraically for `MultiplyNTTs`.\n")
    md.append("2. **Loop Shuffling & Jitter:** Randomize the evaluation order of polynomial indices $i \\in \\{0, 1, 2\\}$ and NTT butterflies to decorrelate power traces from time samples.\n")
    md.append("3. **Masked Keccak / SHA-3:** Implement masked sponge absorption for $G$ and $J$ to protect against plain text and rejection seed leakage.\n")

    return "".join(md)


def save_leakage_ranking_files(data: dict, output_dir: str = ".") -> dict:
    """Save leakage ranking results to .md, .json, and .csv files."""
    os.makedirs(output_dir, exist_ok=True)

    md_path = os.path.join(output_dir, "leakage_ranking.md")
    json_path = os.path.join(output_dir, "leakage_ranking.json")
    csv_path = os.path.join(output_dir, "leakage_ranking.csv")

    # 1. Markdown
    md_content = generate_markdown_report(data)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    # 2. JSON
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    # 3. CSV (Intermediates)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "tvla_rank", "cpa_rank", "intermediate", "location", "total_samples",
            "max_abs_t", "n_leaking_samples", "leaking_pct", "mean_abs_t",
            "peak_cpa_corr", "mean_cpa_corr", "ttd"
        ])
        for r in data["intermediates_by_tvla"]:
            # find corresponding cpa rank
            cpa_rank = next(item["cpa_rank"] for item in data["intermediates_by_cpa"] if item["intermediate"] == r["intermediate"])
            writer.writerow([
                r["tvla_rank"], cpa_rank, r["intermediate"], r["location"], r["total_samples"],
                r["max_t"], r["n_leaking"], r["leaking_pct"], r["mean_t"],
                r["peak_corr"], r["mean_corr"], r["ttd"]
            ])

    return {"md": md_path, "json": json_path, "csv": csv_path}


if __name__ == "__main__":
    analyzer = LeakageRankingAnalyzer(
        algorithm="ML_KEM_768",
        model_name="hamming_weight",
        n_traces=1000,
        snr_db=DEFAULT_SNR_DB,
        seed=RANDOM_SEED,
    )
    print("Running leakage ranking analysis on ML-KEM-768...")
    results = analyzer.run()
    paths = save_leakage_ranking_files(results, output_dir=".")
    print("Ranking complete!")
    print(f"Saved reports:\n  {paths['md']}\n  {paths['json']}\n  {paths['csv']}")
