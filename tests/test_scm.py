from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cpf import scm


def panel(effect: float = 0.0, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    years = np.arange(2000, 2020)
    donors = {f"D{i}": 30 + rng.normal(0, 1, len(years)).cumsum() + i for i in range(8)}
    df = pd.DataFrame(donors, index=years)
    df["T"] = 0.6 * df["D1"] + 0.4 * df["D5"]
    df.loc[2013:, "T"] += effect
    return df


def test_recovers_true_weights() -> None:
    df = panel()
    sc = scm.fit(df, "T", 2013, 2000, 2019, donors=[f"D{i}" for i in range(8)])
    assert sc.weights["D1"] == pytest.approx(0.6, abs=0.02)
    assert sc.weights["D5"] == pytest.approx(0.4, abs=0.02)
    assert sc.weights.sum() == pytest.approx(1.0)
    assert (sc.weights >= 0).all()
    assert sc.pre_rmspe < 1e-3


def test_recovers_effect_and_placebo_rank() -> None:
    df = panel(effect=-5.0)
    donors = [f"D{i}" for i in range(8)]
    sc = scm.fit(df, "T", 2013, 2000, 2019, donors=donors)
    assert sc.average_effect(2013, 2019) == pytest.approx(-5.0, abs=0.05)
    stats, p = scm.placebo_in_space(df, "T", 2013, 2000, 2019, donors)
    assert stats.index[0] == "T"
    assert p == pytest.approx(1 / 9)


def test_simplex_constraints_hold() -> None:
    rng = np.random.default_rng(3)
    w = scm.simplex_weights(rng.normal(size=12), rng.normal(size=(12, 6)))
    assert w.min() >= 0 and w.sum() == pytest.approx(1.0)


def test_augmentation_corrects_extrapolation_bias() -> None:
    """Treated unit sits above every donor: convex weights cannot reach it, ridge can."""
    df = panel()
    df["T"] = df[[f"D{i}" for i in range(8)]].max(axis=1) + 5
    donors = [f"D{i}" for i in range(8)]
    sc = scm.fit(df, "T", 2013, 2000, 2019, donors=donors, ridge=1.0)
    assert sc.augmented is not None
    plain_bias = abs((sc.actual - sc.synthetic).loc[2013:].mean())
    aug_bias = abs((sc.actual - sc.augmented).loc[2013:].mean())
    assert aug_bias < plain_bias


def test_placebo_gaps_shape() -> None:
    df = panel(effect=-3)
    donors = [f"D{i}" for i in range(8)]
    gaps = scm.placebo_gaps(df, "T", 2013, 2000, 2019, donors)
    assert gaps.shape == (20, 9)
