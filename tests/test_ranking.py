"""
Unit tests for leakage ranking of ML-KEM-768 secret-dependent intermediates.
"""

import pytest
from leakage.ranking import LeakageRankingAnalyzer, INTERMEDIATE_SPECS


def test_intermediate_specs():
    """Verify all 5 requested intermediates exist in specifications."""
    expected = ["ByteDecode_12", "MultiplyNTTs", "NTT^-1", "SHA3-512", "SHAKE256"]
    for name in expected:
        assert name in INTERMEDIATE_SPECS, f"Missing intermediate specification: {name}"
        spec = INTERMEDIATE_SPECS[name]
        assert "location" in spec
        assert "op_names" in spec
        assert "variables" in spec
        assert len(spec["variables"]) >= 2


def test_ranking_analyzer_fast():
    """Run a quick ranking analysis with small N to test correctness and ranking order."""
    analyzer = LeakageRankingAnalyzer(
        algorithm="ML_KEM_768",
        model_name="hamming_weight",
        n_traces=40,
        snr_db=20.0,
        seed=42,
    )
    res = analyzer.run()

    # Check top-level metadata
    assert res["algorithm"] == "ML_KEM_768"
    assert res["leakage_model"] == "hamming_weight"
    assert res["total_trace_samples"] == 11264
    assert len(res["intermediates_by_tvla"]) == 5
    assert len(res["intermediates_by_cpa"]) == 5
    assert len(res["variables_by_tvla"]) == 10
    assert len(res["variables_by_cpa"]) == 10

    # Check sorting order
    tvla_vals = [item["max_t"] for item in res["intermediates_by_tvla"]]
    assert tvla_vals == sorted(tvla_vals, reverse=True), "intermediates_by_tvla not sorted descending"

    cpa_vals = [item["peak_corr"] for item in res["intermediates_by_cpa"]]
    assert cpa_vals == sorted(cpa_vals, reverse=True), "intermediates_by_cpa not sorted descending"

    # Verify each intermediate has positive max |t| and peak correlation > 0.5
    for item in res["intermediates_by_tvla"]:
        assert item["max_t"] > 0
        assert item["peak_corr"] > 0.5
        assert "ttd" in item
