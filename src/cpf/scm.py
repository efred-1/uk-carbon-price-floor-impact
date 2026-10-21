"""Synthetic control (Abadie, Diamond & Hainmueller 2010) and a ridge-augmented variant.

The synthetic unit is a weighted average of donor countries, with
non-negative weights summing to one, chosen so its pre-treatment path
matches the treated unit's. After treatment, the gap between the two is
the estimated effect.

Inference is by placebo: run the same procedure pretending each donor
was treated. If the treated unit's post/pre RMSPE ratio is extreme
relative to the placebos, the effect is unlikely to be noise. With N
donors the smallest attainable p-value is 1/(N+1).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.optimize import minimize

Array = npt.NDArray[np.float64]


def simplex_weights(target: Array, donors: Array) -> Array:
    """Weights w >= 0, sum(w) = 1 minimising ||target - donors @ w||².

    ``donors`` has one column per donor and one row per pre-treatment year.
    """
    n = donors.shape[1]
    gram = donors.T @ donors
    lin = donors.T @ target

    def objective(w: Array) -> tuple[float, Array]:
        resid = target - donors @ w
        return float(resid @ resid), 2 * (gram @ w - lin)

    res = minimize(
        objective,
        np.full(n, 1.0 / n),
        jac=True,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * n,
        constraints=[
            {"type": "eq", "fun": lambda w: w.sum() - 1.0, "jac": lambda w: np.ones_like(w)}
        ],
        options={"maxiter": 1000, "ftol": 1e-12},
    )
    w = np.clip(res.x, 0, None)
    out: Array = w / w.sum()
    return out


@dataclass
class SyntheticControl:
    """Fitted synthetic control for one treated unit."""

    treated: str
    treatment_year: int
    weights: pd.Series
    actual: pd.Series
    synthetic: pd.Series
    augmented: pd.Series | None = None
    meta: dict[str, float] = field(default_factory=dict)

    @property
    def gap(self) -> pd.Series:
        return self.actual - self.synthetic

    @property
    def pre(self) -> pd.Series:
        return self.gap[self.gap.index < self.treatment_year]

    @property
    def post(self) -> pd.Series:
        return self.gap[self.gap.index >= self.treatment_year]

    @property
    def pre_rmspe(self) -> float:
        return float(np.sqrt(np.mean(self.pre**2)))

    @property
    def post_rmspe(self) -> float:
        return float(np.sqrt(np.mean(self.post**2)))

    @property
    def rmspe_ratio(self) -> float:
        return self.post_rmspe / max(self.pre_rmspe, 1e-9)

    def average_effect(self, start: int, end: int, augmented: bool = False) -> float:
        series = (
            (self.actual - self.augmented) if augmented and self.augmented is not None else self.gap
        )
        return float(series.loc[start:end].mean())


def fit(
    panel: pd.DataFrame,
    treated: str,
    treatment_year: int,
    pre_start: int,
    post_end: int,
    donors: list[str] | None = None,
    ridge: float | None = None,
) -> SyntheticControl:
    """Fit on a wide panel (index = year, columns = units).

    ``ridge`` adds the augmented SCM bias correction (Ben-Michael, Feller &
    Rothstein 2021): a ridge regression of each post-period outcome on the
    pre-period path, fitted across donors, corrects for whatever pre-period
    imbalance the convex weights could not remove.
    """
    donors = [c for c in (donors or list(panel.columns)) if c != treated]
    window = panel.loc[pre_start:post_end, [treated, *donors]].dropna(axis=1)
    donors = [c for c in donors if c in window.columns]
    pre = window.loc[: treatment_year - 1]
    w = simplex_weights(pre[treated].to_numpy(float), pre[donors].to_numpy(float))
    weights = pd.Series(w, index=donors).sort_values(ascending=False)
    synthetic = window[donors] @ weights.reindex(donors)
    augmented = None
    if ridge is not None:
        x0 = pre[donors].to_numpy(float).T  # donors × pre-years
        x1 = pre[treated].to_numpy(float)
        mu = x0.mean(axis=0)
        xc = x0 - mu
        k = xc.T @ np.linalg.solve(xc @ xc.T + ridge * np.eye(len(donors)), np.eye(len(donors)))
        # Correction for each year: beta_t' (x1 - x0' w), beta_t from ridge of Y_t on pre path.
        imbalance = x1 - x0.T @ w
        y_all = window[donors].to_numpy(float).T  # donors × all years
        beta = k @ (y_all - y_all.mean(axis=0))  # pre-years × all years
        augmented = synthetic + pd.Series(imbalance @ beta, index=window.index)
    return SyntheticControl(
        treated=treated,
        treatment_year=treatment_year,
        weights=weights,
        actual=window[treated],
        synthetic=synthetic,
        augmented=augmented,
    )


def placebo_in_space(
    panel: pd.DataFrame,
    treated: str,
    treatment_year: int,
    pre_start: int,
    post_end: int,
    donors: list[str],
    max_pre_rmspe_multiple: float | None = None,
) -> tuple[pd.DataFrame, float]:
    """Refit treating each donor as if treated. Returns per-unit stats and the permutation p-value.

    ``max_pre_rmspe_multiple`` drops placebos whose pre-fit is much worse than
    the treated unit's (Abadie et al. 2010 use 2, 5 and 20).
    """
    rows = []
    units = [treated, *donors]
    for unit in units:
        pool = [u for u in units if u != unit and u != treated] if unit != treated else donors
        sc = fit(panel, unit, treatment_year, pre_start, post_end, donors=pool)
        rows.append(
            {
                "unit": unit,
                "pre_rmspe": sc.pre_rmspe,
                "post_rmspe": sc.post_rmspe,
                "ratio": sc.rmspe_ratio,
                "mean_post_gap": float(sc.post.mean()),
            }
        )
    stats = pd.DataFrame(rows).set_index("unit")
    if max_pre_rmspe_multiple is not None:
        limit = max_pre_rmspe_multiple * stats.loc[treated, "pre_rmspe"]
        stats = stats[(stats["pre_rmspe"] <= limit) | (stats.index == treated)]
    rank = int((stats["ratio"] >= stats.loc[treated, "ratio"]).sum())
    return stats.sort_values("ratio", ascending=False), rank / len(stats)


def placebo_gaps(
    panel: pd.DataFrame,
    treated: str,
    treatment_year: int,
    pre_start: int,
    post_end: int,
    donors: list[str],
) -> pd.DataFrame:
    """Gap paths for every placebo unit (for the classic spaghetti plot)."""
    units = [treated, *donors]
    gaps = {}
    for unit in units:
        pool = [u for u in units if u not in (unit, treated)] if unit != treated else donors
        gaps[unit] = fit(panel, unit, treatment_year, pre_start, post_end, donors=pool).gap
    return pd.DataFrame(gaps)
