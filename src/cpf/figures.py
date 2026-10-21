"""README figures for the Carbon Price Floor study."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator

from cpf import vizstyle as vs

T0 = 2013


def _series(d: dict[Any, float]) -> pd.Series:
    return pd.Series(d).rename(index=int).sort_index()


def _policy_line(ax: plt.Axes) -> None:
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.axvline(T0 - 0.5, color=vs.AXIS, lw=1)
    ax.text(
        T0 - 0.35,
        0.97,
        "Carbon Price Floor\nfrom April 2013",
        transform=ax.get_xaxis_transform(),
        va="top",
        ha="left",
        fontsize=8,
        color=vs.INK_2,
    )


def path_plot(res: dict[str, Any], ylabel: str, title: str, out: Path) -> None:
    s = res["_series"]
    actual, synthetic = _series(s["actual"]), _series(s["synthetic"])
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(actual.index, actual, color=vs.BLUE, label="United Kingdom")
    ax.plot(synthetic.index, synthetic, color=vs.ORANGE, label="Synthetic UK (weighted donors)")
    _policy_line(ax)
    ax.axvline(2014.5, color=vs.GRID, lw=1)
    ax.text(
        2014.65,
        0.16,
        "£18/t\nfrom 2015",
        transform=ax.get_xaxis_transform(),
        va="top",
        ha="left",
        fontsize=8,
        color=vs.INK_2,
    )
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(bottom=0)
    ax.legend(loc="lower left")
    vs.save(fig, out)


def placebo_plot(res: dict[str, Any], ylabel: str, title: str, out: Path) -> None:
    gaps = {k: _series(v) for k, v in res["_series"]["placebo_gaps"].items()}

    def pre_rmspe(g: pd.Series) -> float:
        return float(np.sqrt(np.mean(g[g.index < T0] ** 2)))

    limit = 5 * pre_rmspe(gaps["GBR"])
    shown = {u: g for u, g in gaps.items() if u == "GBR" or pre_rmspe(g) <= limit}
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.axhline(0, color=vs.AXIS, lw=1)
    for unit, gap in shown.items():
        if unit != "GBR":
            ax.plot(gap.index, gap, color=vs.MUTED, alpha=0.35, lw=1.1, zorder=1)
    uk = gaps["GBR"]
    ax.plot(uk.index, uk, color=vs.BLUE, lw=2.2, zorder=3, label="United Kingdom")
    ax.plot(
        [],
        [],
        color=vs.MUTED,
        alpha=0.35,
        lw=1.1,
        label=f"Placebos: {len(shown) - 1} donors with pre-2013 fit within 5× the UK's",
    )
    _policy_line(ax)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="lower left")
    vs.save(fig, out)


def make_all(out: dict[str, Any], figures: Path) -> None:
    vs.apply()
    coal = out["coal_share_elec"]
    ci = out["carbon_intensity_elec_g_per_kwh"]
    path_plot(
        coal,
        "Coal share of electricity (%)",
        "Coal share: UK vs synthetic UK",
        figures / "coal_share_synthetic.png",
    )
    placebo_plot(
        coal,
        "Gap: actual − synthetic (pp)",
        "Coal share gap, UK vs placebo countries",
        figures / "coal_share_placebos.png",
    )
    path_plot(
        ci,
        "Carbon intensity (gCO₂/kWh)",
        "Carbon intensity of electricity: UK vs synthetic UK",
        figures / "carbon_intensity_synthetic.png",
    )
