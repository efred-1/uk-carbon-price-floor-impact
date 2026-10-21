"""Did the UK Carbon Price Floor push coal out of British power generation?

    python -m cpf.run        # writes results/*.json and figures/*.png

Design, fixed before looking at post-2013 results:

* Treated unit: United Kingdom. Treatment: Carbon Price Support, from April 2013
  (£4.94/tCO₂, £9.55 from April 2014, £18.08 from April 2015, then frozen).
* Main outcome: coal's share of electricity generation (%).
  Second outcome: carbon intensity of electricity (gCO₂/kWh).
* Pre-period 2000–2012. The 1990s "dash for gas" after privatisation was a
  UK-specific structural break, so matching it adds little; 1990–2012 is run as
  a robustness check.
* Post-period 2013–2019 (stops before COVID-19 and the UK's exit from the EU ETS).
* Donors: European countries with complete data whose coal share averaged at
  least 5% in the pre-period.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from cpf import scm
from cpf.data import ROOT, donor_pool, load_mix, wide

RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
TREATED = "GBR"
T0 = 2013
PRE_START = 2000
POST_END = 2019
MIN_COAL_SHARE = 5.0
OUTCOMES = {
    "coal_share_elec": "coal share of generation (%)",
    "carbon_intensity_elec_g_per_kwh": "carbon intensity of electricity (gCO₂/kWh)",
}


def _save(name: str, payload: Any) -> None:
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"{name}.json").write_text(json.dumps(payload, indent=2, default=float) + "\n")
    print(f"wrote results/{name}.json")


def choose_ridge(panel: pd.DataFrame, donors: list[str]) -> float:
    """Pick the ASCM ridge penalty by forecasting the last three pre-years for each donor."""
    grid = [0.1, 1, 10, 100, 1000, 10000]
    errors = {}
    for lam in grid:
        errs = []
        for unit in donors:
            pool = [d for d in donors if d != unit]
            sc = scm.fit(panel, unit, T0 - 3, PRE_START, T0 - 1, donors=pool, ridge=lam)
            assert sc.augmented is not None
            errs.append(np.mean((sc.actual.loc[T0 - 3 :] - sc.augmented.loc[T0 - 3 :]) ** 2))
        errors[lam] = float(np.mean(errs))
    return min(errors, key=errors.get)


def analyse_outcome(mix: pd.DataFrame, column: str, donors: list[str]) -> dict[str, Any]:
    panel = wide(mix, column, [TREATED, *donors])
    lam = choose_ridge(panel, donors)
    main = scm.fit(panel, TREATED, T0, PRE_START, POST_END, donors=donors, ridge=lam)
    stats, p_value = scm.placebo_in_space(panel, TREATED, T0, PRE_START, POST_END, donors)
    _, p_value_fit5 = scm.placebo_in_space(
        panel, TREATED, T0, PRE_START, POST_END, donors, max_pre_rmspe_multiple=5
    )
    gaps = scm.placebo_gaps(panel, TREATED, T0, PRE_START, POST_END, donors)

    # Leave-one-out: drop each donor with weight > 1% in turn.
    loo = {}
    for d in main.weights[main.weights > 0.01].index:
        sc = scm.fit(panel, TREATED, T0, PRE_START, POST_END, donors=[x for x in donors if x != d])
        loo[d] = sc.average_effect(T0, 2017)

    # Placebo in time: pretend the policy started in 2009, using only pre-2013 data.
    fake = scm.fit(panel, TREATED, 2009, PRE_START, T0 - 1, donors=donors)

    # Longer pre-period.
    long = scm.fit(panel, TREATED, T0, 1990, POST_END, donors=donors)

    # Naive comparison: change in UK minus change in the donor average.
    did = float(
        (panel[TREATED].loc[T0:2017].mean() - panel[TREATED].loc[PRE_START : T0 - 1].mean())
        - (
            panel[donors].loc[T0:2017].mean().mean()
            - panel[donors].loc[PRE_START : T0 - 1].mean().mean()
        )
    )

    return {
        "label": OUTCOMES[column],
        "donors": donors,
        "weights": {k: round(v, 4) for k, v in main.weights.items() if v > 0.001},
        "pre_rmspe": main.pre_rmspe,
        "effect_2013_2017": main.average_effect(T0, 2017),
        "effect_2013_2019": main.average_effect(T0, POST_END),
        "effect_2015_2019_full_rate": main.average_effect(2015, POST_END),
        "effect_by_year": main.post.round(3).to_dict(),
        "augmented_ridge_lambda": lam,
        "augmented_effect_2013_2017": main.average_effect(T0, 2017, augmented=True),
        "uk_mean_2013_2017": float(main.actual.loc[T0:2017].mean()),
        "synthetic_mean_2013_2017": float(main.synthetic.loc[T0:2017].mean()),
        "placebo_p_value": p_value,
        "placebo_p_value_good_fit_only": p_value_fit5,
        "placebo_rank_table": stats.round(3).reset_index().to_dict(orient="records"),
        "leave_one_out_effect_2013_2017": loo,
        "placebo_in_time_2009_effect_2009_2012": fake.average_effect(2009, T0 - 1),
        "pre_period_1990_effect_2013_2017": long.average_effect(T0, 2017),
        "pre_period_1990_pre_rmspe": long.pre_rmspe,
        "naive_did_2013_2017": did,
        "_series": {
            "actual": main.actual.to_dict(),
            "synthetic": main.synthetic.to_dict(),
            "augmented": main.augmented.to_dict() if main.augmented is not None else None,
            "placebo_gaps": {k: v.to_dict() for k, v in gaps.items()},
        },
    }


def emissions_avoided(mix: pd.DataFrame, intensity: dict[str, Any]) -> dict[str, float]:
    """Turn the carbon-intensity gap into MtCO₂ using actual UK generation."""
    gen = mix[mix["iso_code"] == TREATED].set_index("year")["generation_twh"]
    gap = pd.Series(intensity["effect_by_year"]).rename(index=int)
    yearly = -gap * gen.reindex(gap.index) / 1000  # gCO2/kWh × TWh = ktCO2 ... / 1000 → MtCO2
    synthetic = pd.Series(intensity["_series"]["synthetic"]).rename(index=int)
    reduction_share = -gap / synthetic.reindex(gap.index)
    return {
        "mt_co2_avoided_by_year": yearly.round(2).to_dict(),
        "mt_co2_avoided_2015_2019": float(yearly.loc[2015:POST_END].sum()),
        "share_below_synthetic_2015_2019": float(reduction_share.loc[2015:POST_END].mean()),
        "mt_co2_avoided_2013_2017": float(yearly.loc[T0:2017].sum()),
        "mt_co2_avoided_per_year_2013_2017": float(yearly.loc[T0:2017].mean()),
        "share_below_synthetic_2013_2017": float(reduction_share.loc[T0:2017].mean()),
    }


def main() -> None:
    mix = load_mix()
    donors = donor_pool(mix, PRE_START, T0 - 1, MIN_COAL_SHARE)
    out = {column: analyse_outcome(mix, column, donors) for column in OUTCOMES}
    out["emissions"] = emissions_avoided(mix, out["carbon_intensity_elec_g_per_kwh"])
    _save("results", out)
    from cpf.figures import make_all

    make_all(out, FIGURES)


if __name__ == "__main__":
    main()
