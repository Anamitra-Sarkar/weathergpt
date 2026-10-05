"""Turn a rain exceedance curve S(t) = P(rain >= t) into the answers people ask for.

"Will it rain, and if so how much?" is one question about one distribution.  The curve model
gives S at fixed thresholds; everything here is derived from it, so the answers are coherent
by construction (probabilities of classes sum to P(any rain), amounts never contradict the
probabilities):

  p_any_rain        P(rain >= 1 mm)       light rain or more
  p_rainy_day       P(rain >= 2.5 mm)     IMD's definition of a rainy day
  class_probs       IMD categories (very light ... extremely heavy), summing to P(>= 0.1 mm)
  amount_if_wet     conditional amount quantiles GIVEN rain >= 1 mm  (e.g. 10th / 50th / 90th)

The probability is that of a random point in the forecast cell seeing rain, which matches the
operational PoP definition (confidence x area coverage), not "share of area" alone.
"""
from __future__ import annotations

import numpy as np

IMD_CLASSES = (("very_light", 0.1, 2.5), ("light", 2.5, 15.6), ("moderate", 15.6, 64.5),
               ("heavy", 64.5, 115.6), ("very_heavy", 115.6, 204.5), ("extremely_heavy", 204.5, np.inf))
WET = 1.0           # "it rained" for the conditional-amount distribution
MIN_WET_PROB = 0.02  # below this the conditional amount is not meaningful


def enforce_monotone(S: np.ndarray) -> np.ndarray:
    """Make S non-increasing along thresholds without ever raising a lower threshold's probability."""
    return np.clip(np.minimum.accumulate(np.asarray(S, float), axis=1), 0.0, 1.0)


def survival_at(S: np.ndarray, thresholds, t: float) -> np.ndarray:
    """S at an arbitrary threshold: linear in log(threshold) between the modelled ones; clipped to range."""
    S = np.asarray(S, float)
    x = np.log(np.asarray(thresholds, float))
    xt = np.log(t)
    if xt <= x[0]:
        return S[:, 0]
    if xt >= x[-1]:
        return S[:, -1]
    j = int(np.searchsorted(x, xt) - 1)
    w = (xt - x[j]) / (x[j + 1] - x[j])
    return S[:, j] * (1 - w) + S[:, j + 1] * w


def amount_quantiles(S: np.ndarray, thresholds, levels=(0.1, 0.5, 0.9), wet: float = WET):
    """Conditional quantiles of the amount GIVEN rain >= `wet`.  -> (amounts [n, len(levels)], capped [n, len(levels)]).

    NaN where P(wet) is too small for a conditional amount to mean anything.  `capped` marks quantiles that
    lie beyond the highest modelled threshold (reported as that threshold, i.e. a lower bound).
    """
    S = enforce_monotone(S)
    th = np.asarray(thresholds, float)
    x = np.log(th)
    s0 = survival_at(S, th, wet)
    start = int(np.searchsorted(th, wet))          # first threshold >= wet
    out = np.full((len(S), len(levels)), np.nan)
    capped = np.zeros_like(out, dtype=bool)
    for i in range(len(S)):
        if s0[i] < MIN_WET_PROB:
            continue
        xs = np.concatenate([[np.log(wet)], x[start:]])
        cs = np.concatenate([[1.0], S[i, start:] / s0[i]])          # conditional survival, starts at 1
        cs = np.minimum.accumulate(np.clip(cs, 0, 1))
        for k, level in enumerate(levels):
            target = 1.0 - level                                     # P(Y > a | wet) = 1 - level
            below = np.nonzero(cs <= target)[0]
            if len(below) == 0:
                out[i, k], capped[i, k] = th[-1], True
                continue
            j = below[0]
            if j == 0:
                out[i, k] = wet
                continue
            c0, c1 = cs[j - 1], cs[j]
            frac = 0.0 if c0 == c1 else (c0 - target) / (c0 - c1)
            out[i, k] = float(np.exp(xs[j - 1] + frac * (xs[j] - xs[j - 1])))
    return out, capped


def rain_products(S: np.ndarray, thresholds, levels=(0.1, 0.5, 0.9)) -> dict:
    S = enforce_monotone(S)
    classes = {}
    for name, lo, hi in IMD_CLASSES:
        upper = np.zeros(len(S)) if np.isinf(hi) else survival_at(S, thresholds, hi)
        classes[name] = np.clip(survival_at(S, thresholds, lo) - upper, 0.0, 1.0)
    amounts, capped = amount_quantiles(S, thresholds, levels)
    return {"p_any_rain": survival_at(S, thresholds, 1.0), "p_rainy_day": survival_at(S, thresholds, 2.5),
            "p_heavy": survival_at(S, thresholds, 64.5), "class_probs": classes,
            "amount_if_wet": amounts, "amount_levels": tuple(levels), "amount_capped": capped}
