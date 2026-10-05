"""Rain products: coherence and edge cases of deriving answers from an exceedance curve."""
import numpy as np
import pytest

from event_models import rain_products as rp

TH = (0.1, 0.5, 1.0, 2.5, 5.0, 7.5, 10.0, 15.6, 25.0, 35.5, 50.0, 64.5, 90.0, 115.6, 150.0, 204.5)


def _exp_curve(scale, p_wet=1.0):
    """Survival of an exponential amount (mean `scale` mm) times a wet probability."""
    return p_wet * np.exp(-np.asarray(TH) / scale)


def test_enforce_monotone_never_raises_a_lower_threshold():
    raw = np.array([[0.5, 0.6, 0.3, 0.4, 0.1]])
    fixed = rp.enforce_monotone(raw)
    assert fixed.tolist() == [[0.5, 0.5, 0.3, 0.3, 0.1]]


def test_class_probabilities_sum_to_probability_of_any_measurable_rain():
    S = np.stack([_exp_curve(8.0, 0.7), _exp_curve(30.0, 0.9), _exp_curve(2.0, 0.2)])
    prod = rp.rain_products(S, TH)
    total = sum(prod["class_probs"].values())
    assert np.allclose(total, rp.survival_at(S, TH, 0.1), atol=1e-9)
    assert (prod["p_any_rain"] >= prod["p_rainy_day"]).all() and (prod["p_rainy_day"] >= prod["p_heavy"]).all()


def test_conditional_amount_quantiles_match_the_known_distribution():
    scale, p_wet = 12.0, 0.8
    S = _exp_curve(scale, p_wet)[None, :]
    amounts, capped = rp.amount_quantiles(S, TH, levels=(0.1, 0.5, 0.9))
    # given wet (>=1 mm) an exponential(12) amount has quantile 1 + (-12 ln(1-q)) -- interpolation between
    # thresholds is coarse, so require agreement within ~15%
    want = 1.0 - scale * np.log(1 - np.array([0.1, 0.5, 0.9]))
    assert np.allclose(amounts[0], want, rtol=0.15), (amounts, want)
    assert (np.diff(amounts[0]) > 0).all() and not capped.any()


def test_dry_cell_has_no_conditional_amount_instead_of_a_made_up_one():
    S = _exp_curve(5.0, p_wet=0.005)[None, :]          # 0.5% chance of rain
    amounts, _ = rp.amount_quantiles(S, TH)
    assert np.isnan(amounts).all()
    prod = rp.rain_products(S, TH)
    assert prod["p_any_rain"][0] < 0.01


def test_tail_beyond_the_highest_threshold_is_flagged_as_a_lower_bound():
    S = np.full((1, len(TH)), 0.95)                    # almost everything exceeds even 204.5 mm
    amounts, capped = rp.amount_quantiles(S, TH, levels=(0.5, 0.9))
    assert amounts[0].tolist() == [TH[-1], TH[-1]] and capped.all()


def test_non_monotone_model_output_still_gives_coherent_products():
    rng = np.random.default_rng(0)
    S = np.clip(_exp_curve(10.0, 0.6) + rng.normal(0, 0.03, len(TH)), 0, 1)[None, :]   # noisy, can be non-monotone
    prod = rp.rain_products(S, TH)
    assert all((v >= 0).all() for v in prod["class_probs"].values())
    amounts = prod["amount_if_wet"][0]
    assert np.all(np.diff(amounts) >= 0)               # quantiles are ordered whatever the input noise


def test_survival_at_interpolates_in_log_threshold_and_clips():
    S = _exp_curve(10.0)[None, :]
    mid = rp.survival_at(S, TH, 3.5)[0]
    assert S[0, 4] < mid < S[0, 3]                     # between the 2.5 and 5.0 values
    assert rp.survival_at(S, TH, 0.01)[0] == S[0, 0] and rp.survival_at(S, TH, 999)[0] == S[0, -1]
