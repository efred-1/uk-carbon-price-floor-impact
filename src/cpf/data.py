"""Panel data from the energy warehouse (marts.electricity_mix)."""

from __future__ import annotations

import os
from pathlib import Path

import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
WAREHOUSE = os.environ.get(
    "WAREHOUSE_PATH",
    str(ROOT.parent / "energy-emissions-data-pipeline" / "warehouse" / "energy.duckdb"),
)

# European countries considered as donors before applying the coal rule below.
EUROPE = [
    "AUT",
    "BEL",
    "BGR",
    "CHE",
    "CZE",
    "DEU",
    "DNK",
    "ESP",
    "EST",
    "FIN",
    "FRA",
    "GRC",
    "HRV",
    "HUN",
    "IRL",
    "ITA",
    "LTU",
    "LUX",
    "LVA",
    "NLD",
    "NOR",
    "POL",
    "PRT",
    "ROU",
    "SVK",
    "SVN",
    "SWE",
]


def load_mix() -> pd.DataFrame:
    with duckdb.connect(WAREHOUSE, read_only=True) as con:
        return con.execute("SELECT * FROM marts.electricity_mix").df()


def wide(mix: pd.DataFrame, column: str, units: list[str]) -> pd.DataFrame:
    sub = mix[mix["iso_code"].isin(units)]
    return sub.pivot(index="year", columns="iso_code", values=column).sort_index()


def donor_pool(mix: pd.DataFrame, pre_start: int, pre_end: int, min_coal_share: float) -> list[str]:
    """European countries with complete data and a meaningful coal share before treatment.

    A country that burns almost no coal cannot show the coal-to-gas switch the
    policy targets, so it cannot help build a counterfactual for it.
    """
    shares = wide(mix, "coal_share_elec", EUROPE).loc[pre_start:pre_end]
    complete = shares.columns[shares.notna().all()]
    return sorted(c for c in complete if shares[c].mean() >= min_coal_share)
