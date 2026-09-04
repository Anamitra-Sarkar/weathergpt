"""Tests for the standalone model package.

These run without any trained artifact, because the properties they check are
the ones that failed silently before: a label that contradicts its own unit, a
feature vector that means something different at serving time than it did at
training time, and a checkpoint that ships a metrics file nobody validates.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from weathergpt_models.features import (MODELS, VARIABLES, assemble_features, ensemble_summary,
                                        feature_names, members_from_mapping)
from weathergpt_models.registry import GATES, ModelRegistry
from weathergpt_models.taxonomy import (CANONICAL_VARIABLES, LABEL_DESCRIPTIONS,
                                        classify_native_field, parse_accumulation_hours,
                                        unit_family)


# --- taxonomy ----------------------------------------------------------------
def test_temperature_in_depth_units_is_refused():
    """The exact defect that put 64 `TMAX (mm)` rows in the previous corpus."""
    good = classify_native_field("TMAX", description="Maximum temperature", unit="K",
                                 level_text="2 m above ground")
    assert good is not None and good.canonical_variable == "temperature_max"

    bad = classify_native_field("TMAX", description="Maximum temperature", unit="mm",
                                level_text="2 m above ground")
    assert bad is None, "a temperature labelled in millimetres must abstain, not guess"


def test_precipitation_family_members_are_not_interchangeable():
    amount = classify_native_field("APCP", description="Total precipitation", unit="kg m-2",
                                   time_range_text="0-3 hour acc fcst",
                                   grib_statistical_processing=1)
    rate = classify_native_field("prate", description="Precipitation rate",
                                 unit="kg m-2 s-1")
    probability = classify_native_field("pop", description="Probability of precipitation",
                                        unit="%")
    assert amount.canonical_variable == "precipitation_amount"
    assert rate.canonical_variable == "precipitation_rate"
    assert probability.canonical_variable == "precipitation_probability"
    assert len({amount.canonical_variable, rate.canonical_variable,
                probability.canonical_variable}) == 3


def test_accumulation_window_comes_from_the_time_range_not_a_guess():
    label = classify_native_field("APCP", description="Total precipitation", unit="kg m-2",
                                  time_range_text="0-3 hour acc fcst",
                                  grib_statistical_processing=1)
    assert label.accumulation_hours == 3.0

    unwindowed = classify_native_field("tp", description="Total precipitation", unit="m",
                                       grib_statistical_processing=1)
    assert unwindowed.accumulation_hours is None, (
        "an accumulation with no declared window must stay unset so the semantic "
        "gate refuses to compare it")


def test_parse_accumulation_hours_handles_real_inventory_strings():
    assert parse_accumulation_hours("0-6 hour acc fcst") == 6.0
    assert parse_accumulation_hours("precip (mm/24h)") == 24.0
    assert parse_accumulation_hours("daily rainfall total") == 24.0
    assert parse_accumulation_hours("instant") is None


def test_non_meteorological_fields_abstain():
    for name, description, unit in (("XLAT", "LATITUDE, SOUTH IS NEGATIVE", "degree_north"),
                                    ("LU_INDEX", "LAND USE CATEGORY", None),
                                    ("tableAEntry", "Table A: Entry", "CCITT IA5")):
        assert classify_native_field(name, description=description, unit=unit) is None


def test_warning_events_stay_categorical():
    label = classify_native_field("Heavy Rainfall",
                                  description="heavy rainfall warning for the district",
                                  unit=None, evidence_class_hint="warning")
    assert label.canonical_variable == "heavy_rain_warning"
    assert label.statistic == "categorical"
    assert label.evidence_class == "warning"


def test_every_canonical_variable_has_label_text():
    missing = set(CANONICAL_VARIABLES) - set(LABEL_DESCRIPTIONS)
    assert not missing, f"the M1 label space would have no text for {missing}"


def test_unit_families_are_unambiguous():
    assert unit_family("K") == "temperature"
    assert unit_family("m s-1") == "speed"
    assert unit_family("kg m-2") == unit_family("mm") == "length"
    assert unit_family(None) == "categorical"


# --- features ----------------------------------------------------------------
def _context(n: int = 1) -> dict:
    return {"lead_hours": np.full(n, 30.0), "lead_age_days": np.full(n, 1.0),
            "hour_utc": np.full(n, 6.0), "doy": np.full(n, 245.0),
            "elevation_m": np.full(n, 303.0), "lat": np.full(n, 21.15),
            "lon": np.full(n, 79.09)}


def test_feature_vector_matches_its_declared_names():
    members = np.array([[31.2, 30.4, 30.9, 31.8]])
    X = assemble_features("temperature_2m", members, {}, _context())
    assert X.shape == (1, len(feature_names("temperature_2m")))


def test_missing_members_are_imputed_and_flagged():
    members = np.array([[31.2, np.nan, 30.8, 31.0]])
    X = assemble_features("temperature_2m", members, {}, _context())
    names = feature_names("temperature_2m")

    imputed = X[0, names.index("fc_temperature_2m_ecmwf_ifs025")]
    assert np.isfinite(imputed), "a missing member must not reach the model as NaN"
    assert imputed == pytest.approx(np.nanmean(members))
    assert X[0, names.index("missing_ecmwf_ifs025")] == 1.0
    assert X[0, names.index("missing_gfs_seamless")] == 0.0


def test_ensemble_summary_is_nan_aware():
    values = np.array([[1.0, np.nan, 3.0, 5.0]])
    summary = ensemble_summary(values)
    assert summary[0, 0] == pytest.approx(3.0)     # mean over live members only
    assert summary[0, 2] == pytest.approx(1.0)     # min
    assert summary[0, 3] == pytest.approx(5.0)     # max


def test_wet_fraction_only_applies_to_precipitation():
    members = np.array([[0.0, 0.0, 4.0, 6.0]])
    names = feature_names("precipitation")
    wet = assemble_features("precipitation", members, {}, _context())[
        0, names.index("ens_wet_fraction")]
    assert wet == pytest.approx(0.5)

    names_t = feature_names("temperature_2m")
    wet_t = assemble_features("temperature_2m", np.array([[30.0, 31.0, 32.0, 33.0]]),
                              {}, _context())[0, names_t.index("ens_wet_fraction")]
    assert wet_t == 0.0


def test_members_from_mapping_preserves_model_order():
    row = members_from_mapping({"gfs_seamless": 1.0, "icon_seamless": 3.0})
    assert row.shape == (1, len(MODELS))
    assert row[0, MODELS.index("gfs_seamless")] == 1.0
    assert np.isnan(row[0, MODELS.index("ecmwf_ifs025")])


def test_cross_variable_block_is_zero_when_absent():
    members = np.array([[1.0, 2.0, 3.0, 4.0]])
    names = feature_names("precipitation")
    X = assemble_features("precipitation", members, {}, _context())
    for other in VARIABLES:
        if other == "precipitation":
            continue
        assert X[0, names.index(f"x_{other}_mean")] == 0.0


# --- registry gate -----------------------------------------------------------
def _write(tmp_path, model: str, metrics: dict):
    directory = tmp_path / GATES[model].directory
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "metrics.json").write_text(json.dumps(metrics))
    return directory


PROVENANCE = {"algorithm_version": "m1_field_mapper_v1",
              "dataset_kind": "d3_authoritative_parameter_tables",
              "dataset_sha256": "deadbeef" * 8, "split": "by_source_table",
              "trained_at": "2026-09-03T00:00:00Z"}


def test_gate_refuses_an_artifact_with_no_provenance(tmp_path):
    _write(tmp_path, "field_mapper", {"test_zeroshot_macro_f1": 0.99})
    registry = ModelRegistry(tmp_path)
    row = next(r for r in registry.status() if r["model"] == "field_mapper")
    assert row["loaded"] is False
    assert "provenance" in row["reason"]
    assert registry.field_mapper is None


def test_gate_refuses_an_artifact_that_does_not_beat_its_baseline(tmp_path):
    _write(tmp_path, "field_mapper", {**PROVENANCE, "test_zeroshot_macro_f1": 0.12,
                                      "test_zeroshot_misassignment_rate": 0.01,
                                      "baselines": {"dict_registry_zeroshot": {"macro_f1": 0.16}}})
    row = next(r for r in ModelRegistry(tmp_path).status() if r["model"] == "field_mapper")
    assert row["loaded"] is False
    assert "does not beat" in row["reason"]


def test_gate_refuses_a_model_that_confidently_misassigns(tmp_path):
    _write(tmp_path, "field_mapper", {**PROVENANCE, "test_zeroshot_macro_f1": 0.90,
                                      "test_zeroshot_misassignment_rate": 0.31,
                                      "baselines": {"dict_registry_zeroshot": {"macro_f1": 0.16}}})
    row = next(r for r in ModelRegistry(tmp_path).status() if r["model"] == "field_mapper")
    assert row["loaded"] is False
    assert "misassignment" in row["reason"]


def test_gate_admits_a_model_that_proves_itself(tmp_path):
    _write(tmp_path, "field_mapper", {**PROVENANCE, "test_zeroshot_macro_f1": 0.73,
                                      "test_zeroshot_misassignment_rate": 0.003,
                                      "baselines": {"dict_registry_zeroshot": {"macro_f1": 0.16}}})
    row = next(r for r in ModelRegistry(tmp_path).status() if r["model"] == "field_mapper")
    assert row["loaded"] is True


def test_gate_refuses_post_processing_with_no_skill(tmp_path):
    """A corrector that does not beat the raw ensemble is not a corrector."""
    _write(tmp_path, "mos", {**PROVENANCE, "algorithm_version": "m2_mos_v1",
                             "results": {"temperature_2m": {
                                 "test_spatial_holdout": {"crpss_vs_raw_ensemble": -0.02}}}})
    row = next(r for r in ModelRegistry(tmp_path).status() if r["model"] == "mos")
    assert row["loaded"] is False
    assert "no better than the raw ensemble" in row["reason"]


def test_gate_refuses_calibration_worse_than_counting_members(tmp_path):
    _write(tmp_path, "calibration", {**PROVENANCE, "algorithm_version": "m4_calibration_v1",
                                     "results": {"precipitation": {"exceedance": {
                                         "1.0": {"brier_csgd_isotonic": 0.09,
                                                 "brier_raw_ensemble_frequency": 0.07}}}}})
    row = next(r for r in ModelRegistry(tmp_path).status() if r["model"] == "calibration")
    assert row["loaded"] is False
    assert "worse than" in row["reason"]


def test_missing_artifacts_report_cleanly(tmp_path):
    status = ModelRegistry(tmp_path).status()
    assert len(status) == len(GATES)
    assert all(row["loaded"] is False for row in status)
    assert all("no artifact" in row["reason"] for row in status)


def test_strict_mode_raises_instead_of_degrading(tmp_path):
    _write(tmp_path, "field_mapper", {"test_zeroshot_macro_f1": 0.99})
    with pytest.raises(RuntimeError, match="failed its metrics gate"):
        ModelRegistry(tmp_path, strict=True)


# --- CSGD numerics -----------------------------------------------------------
# The censored shifted gamma is the one piece of non-standard maths in the
# model layer, and the first implementation was wrong in two ways that produced
# plausible-looking losses: it integrated the density from the censoring point
# instead of from zero, so it computed F(t-shift) - F(-shift) and silently
# dropped the dry-probability mass; and the quadrature itself recovered only 8%
# of the total mass at the small shape parameters a mostly-dry precipitation fit
# actually converges to.  These tests pin the corrected behaviour.
def _csgd_cdf(shape, scale, shift, t):
    import torch

    return torch.special.gammainc(shape, ((t - shift) / scale).clamp(min=0.0))


def _csgd_crps(shape, scale, shift, y, samples: int = 96, seed: int = 0):
    import torch

    torch.manual_seed(seed)
    gamma = torch.distributions.Gamma(shape, 1.0 / scale)
    draws = (gamma.rsample((samples,)).transpose(0, 1) + shift.unsqueeze(-1)).clamp(min=0.0)
    ordered, _ = torch.sort(draws, dim=1)
    first = (ordered - y.unsqueeze(-1)).abs().mean(1)
    weights = 2 * torch.arange(1, samples + 1, dtype=y.dtype) - samples - 1
    return first - (weights * ordered).sum(1) / (samples * (samples - 1))


def test_csgd_cdf_is_monotone_and_saturates_at_one():
    import torch

    shape = torch.tensor([0.05, 0.5, 3.0, 1.0])
    scale = torch.tensor([0.05, 2.0, 5.0, 1.0])
    shift = torch.tensor([-0.001, -1.0, -10.0, 0.0])
    previous = None
    for t in (0.0, 0.1, 1.0, 5.0, 20.0, 100.0, 1000.0):
        current = _csgd_cdf(shape, scale, shift, torch.full_like(shape, t))
        assert (current >= -1e-6).all() and (current <= 1 + 1e-6).all()
        if previous is not None:
            assert (current >= previous - 1e-6).all(), f"CDF decreased at t={t}"
        previous = current
    assert previous.min() > 0.999, "the CDF must reach 1; a quadrature that loses mass will not"


def test_csgd_cdf_at_zero_is_the_dry_probability():
    """A precipitation distribution must be able to say 'it will not rain'."""
    import torch

    dry = _csgd_cdf(torch.tensor([0.05]), torch.tensor([0.05]),
                    torch.tensor([-0.001]), torch.tensor([0.0]))
    assert 0.5 < float(dry[0]) < 1.0


def test_sampled_crps_matches_a_large_sample_reference():
    import torch

    shape = torch.tensor([0.5, 3.0, 1.0])
    scale = torch.tensor([2.0, 5.0, 1.0])
    shift = torch.tensor([-1.0, -10.0, 0.0])
    y = torch.tensor([2.0, 30.0, 1.0])

    estimate = _csgd_crps(shape, scale, shift, y)

    torch.manual_seed(1234)
    big = (torch.distributions.Gamma(shape, 1.0 / scale).sample((100_000,))
           + shift).clamp(min=0.0)
    ordered, _ = torch.sort(big, dim=0)
    m = big.shape[0]
    weights = (2 * torch.arange(1, m + 1, dtype=torch.float32) - m - 1).unsqueeze(-1)
    reference = (big - y).abs().mean(0) - (weights * ordered).sum(0) / (m * (m - 1))

    relative = ((estimate - reference).abs() / reference.clamp(min=1e-6))
    assert float(relative.max()) < 0.15, f"sampled CRPS is off by {relative.tolist()}"


def test_csgd_gradients_reach_all_three_parameters():
    """`gammainc` has no derivative in the shape, which is why the loss samples."""
    import torch
    import torch.nn.functional as F

    raw = torch.zeros(4, 3, requires_grad=True)
    shape = F.softplus(raw[:, 0]) + 0.05
    scale = F.softplus(raw[:, 1]) + 0.05
    shift = -F.softplus(raw[:, 2]).clamp(max=25.0)
    _csgd_crps(shape, scale, shift, torch.tensor([0.0, 2.0, 30.0, 1.0])).mean().backward()

    assert torch.isfinite(raw.grad).all()
    assert (raw.grad.abs().sum(0) > 0).all(), "a parameter received no gradient"


def test_shift_stays_non_positive():
    """A positive shift would mean rain is guaranteed, and removes the dry mass."""
    import torch
    import torch.nn.functional as F

    raw = torch.linspace(-50, 50, 101)
    shift = -F.softplus(raw).clamp(max=25.0)
    assert float(shift.max()) <= 0.0
    assert float(shift.min()) >= -25.0


def test_cube_root_transform_preserves_quantiles():
    """Why M2 can fit precipitation in cube-root space and cube the answer.

    Quantiles are equivariant under a monotone transform, so no Jensen
    correction is needed on the way back — unlike a mean, which would be biased.
    """
    rng = np.random.default_rng(0)
    draws = np.abs(rng.gamma(0.4, 6.0, size=200_000))
    levels = np.array([0.05, 0.25, 0.5, 0.75, 0.95])

    direct = np.quantile(draws, levels)
    via_transform = np.quantile(np.cbrt(draws), levels) ** 3

    assert np.allclose(direct, via_transform, rtol=1e-9), (
        "cubing the quantiles of the cube root must reproduce the original quantiles")

    # a mean does not survive the round trip, which is the reason this only
    # works for a quantile model
    assert not np.isclose(np.cbrt(draws).mean() ** 3, draws.mean(), rtol=0.05)


# --- slot text hygiene -------------------------------------------------------
def test_slot_text_trims_edge_punctuation():
    """BIO tags land on whitespace tokens, so the last token of a span carries
    whatever punctuation followed it.  A TIME slot that comes back as
    "tomorrow afternoon?" makes the downstream time parser deal with a question
    mark that is not part of the time expression."""
    import re

    source = open("weathergpt_models/intent.py").read()
    start = source.index("_EDGE_PUNCTUATION")
    end = source.index("class IntentParser")
    namespace = {"re": re}
    exec(compile(source[start:end], "intent_trim", "exec"), namespace)  # noqa: S102
    trim = namespace["_trim"]

    assert trim("tomorrow afternoon?") == "tomorrow afternoon"
    assert trim('"Kolhapur",') == "Kolhapur"
    assert trim("नागपुर में।") == "नागपुर में"
    assert trim("Bhandara district") == "Bhandara district"
    # never trim a slot down to nothing
    assert trim("???") == "???"


def test_bio_spans_align_to_whitespace_tokens():
    """The D4 builder computes spans from substrings it injected, so a span that
    does not land on token boundaries means the row is mislabelled and must be
    dropped rather than rounded to the nearest token."""
    import re

    source = open("modal_jobs/build_queries.py").read()
    start = source.index("def _spans_to_bio")
    end = source.index("def _build_base")
    namespace = {"re": re}
    exec(compile(source[start:end], "bio", "exec"), namespace)  # noqa: S102
    spans_to_bio = namespace["_spans_to_bio"]

    text = "Will it rain in Bhandara district tomorrow afternoon?"
    tagged = spans_to_bio(text, [(16, 33, "LOC"), (34, 52, "TIME")])
    assert [token for token, tag in tagged if tag.endswith("LOC")] == ["Bhandara", "district"]
    assert [tag for _, tag in tagged if tag.endswith("LOC")] == ["B-LOC", "I-LOC"]
    assert [token for token, tag in tagged if tag.endswith("TIME")] == ["tomorrow", "afternoon?"]

    # a span outside the text produces no covered tokens, and the row is rejected
    assert spans_to_bio("rain tomorrow", [(400, 410, "LOC")]) is None


def test_precipitation_interval_low_cannot_be_negative():
    """quantiles[0.1] is clamped to >= 0 for precipitation, but the conformal
    widening happens after that clamp and can still push interval_low negative
    if it is not clamped again -- a rain-rate lower bound below zero is not a
    valid interval, whatever the model's raw arithmetic says."""
    import re

    source = open("weathergpt_models/mos.py").read()
    assert re.search(
        r'if variable in \("precipitation",\):\s*\n\s*#.*\n\s*interval_low = max\(0\.0, interval_low\)',
        source), "the conformal interval_low for precipitation must be re-clamped to zero"
