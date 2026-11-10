"""Synthetic stand-in for the Kaggle file - used ONLY for tests and smoke runs.

It mimics the real file's schema and its known quirks (tab-separated, dd-mm-YYYY dates, missing
income, impossible birth years, an extreme income, odd marital statuses, duplicate customers,
constant Z_* columns). Never report results produced from this data.

    python -m tests.make_synthetic_data data/raw/marketing_campaign.csv
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

COLUMNS = [
    "ID", "Year_Birth", "Education", "Marital_Status", "Income", "Kidhome", "Teenhome", "Dt_Customer",
    "Recency", "MntWines", "MntFruits", "MntMeatProducts", "MntFishProducts", "MntSweetProducts",
    "MntGoldProds", "NumDealsPurchases", "NumWebPurchases", "NumCatalogPurchases", "NumStorePurchases",
    "NumWebVisitsMonth", "AcceptedCmp3", "AcceptedCmp4", "AcceptedCmp5", "AcceptedCmp1", "AcceptedCmp2",
    "Complain", "Z_CostContact", "Z_Revenue", "Response",
]


def make_synthetic(n: int = 2200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    seg = rng.choice(4, size=n, p=[0.27, 0.30, 0.30, 0.13])
    inc_mu = np.array([72_000, 52_000, 31_000, 92_000])[seg]
    inc_sd = np.array([10_000, 9_000, 7_000, 12_000])[seg]
    income = np.clip(rng.normal(inc_mu, inc_sd), 5_000, None).round()
    kid = np.where(seg == 1, rng.binomial(1, 0.5, n), np.where(seg == 2, rng.binomial(2, 0.55, n), 0))
    teen = np.where(seg == 1, rng.binomial(2, 0.5, n), np.where(seg == 0, rng.binomial(1, 0.2, n), 0))
    kids = kid + teen
    k = income / 1000 * np.clip(1 - 0.22 * kids, 0.25, 1)

    def spend(scale):
        return np.round(rng.gamma(1.4, scale * k / 1.4)).astype(int)

    wines, meat = spend(4.0), spend(2.2)
    fruits, fish, sweets, gold = spend(0.35), spend(0.55), spend(0.35), spend(0.7)
    web = np.clip(rng.poisson(1.5 + income / 40_000), 0, 27)
    catalog = np.clip(rng.poisson(income / 30_000 * (kids == 0) + 0.2), 0, 28)
    store = np.clip(rng.poisson(2.5 + income / 18_000), 0, 13)
    deals = np.clip(rng.poisson(1 + kids), 0, 15)
    visits = np.clip(rng.poisson(np.clip(7 - income / 20_000 + kids, 1, None)), 0, 20)
    recency = rng.integers(0, 100, n)

    total = wines + meat + fruits + fish + sweets + gold
    z = (total - total.mean()) / total.std()
    cmp_p = 1 / (1 + np.exp(-(-3.0 + 0.7 * z)))
    cmps = {c: rng.binomial(1, cmp_p * f) for c, f in
            zip(["AcceptedCmp3", "AcceptedCmp4", "AcceptedCmp5", "AcceptedCmp1", "AcceptedCmp2"],
                [0.5, 0.6, 0.6, 0.6, 0.15])}
    n_acc = sum(cmps.values())
    logit = -2.7 + 0.6 * z + 1.1 * (n_acc > 0) - 0.015 * recency - 0.3 * kids
    response = rng.binomial(1, 1 / (1 + np.exp(-logit)))

    dates = pd.Timestamp("2012-07-30") + pd.to_timedelta(rng.integers(0, 700, n), unit="D")
    age = np.clip(rng.normal(46, 11, n), 20, 80).round().astype(int)
    df = pd.DataFrame({
        "ID": rng.permutation(np.arange(1000, 1000 + n)),
        "Year_Birth": 2014 - age,
        "Education": rng.choice(["Graduation", "PhD", "Master", "2n Cycle", "Basic"], n,
                                p=[0.50, 0.22, 0.17, 0.09, 0.02]),
        "Marital_Status": rng.choice(["Married", "Together", "Single", "Divorced", "Widowed"], n,
                                     p=[0.39, 0.26, 0.22, 0.10, 0.03]),
        "Income": income, "Kidhome": kid, "Teenhome": teen,
        "Dt_Customer": dates.strftime("%d-%m-%Y"), "Recency": recency,
        "MntWines": wines, "MntFruits": fruits, "MntMeatProducts": meat, "MntFishProducts": fish,
        "MntSweetProducts": sweets, "MntGoldProds": gold, "NumDealsPurchases": deals,
        "NumWebPurchases": web, "NumCatalogPurchases": catalog, "NumStorePurchases": store,
        "NumWebVisitsMonth": visits, **cmps, "Complain": rng.binomial(1, 0.01, n),
        "Z_CostContact": 3, "Z_Revenue": 11, "Response": response,
    })

    # --- known data-quality problems in the real file
    df.loc[rng.choice(n, 24, replace=False), "Income"] = np.nan
    df.loc[[5, 6, 7], "Year_Birth"] = [1893, 1899, 1900]
    df.loc[10, "Income"] = 666_666
    df.loc[[20, 21], "Marital_Status"] = "Absurd"
    df.loc[[22, 23], "Marital_Status"] = "YOLO"
    df.loc[[24, 25, 26], "Marital_Status"] = "Alone"
    dupes = df.iloc[100:130].copy()
    dupes["ID"] = np.arange(90_000, 90_000 + len(dupes))
    df = pd.concat([df, dupes], ignore_index=True)
    return df[COLUMNS]


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "data/raw/marketing_campaign.csv"
    make_synthetic().to_csv(out, sep="\t", index=False)
    print(f"Wrote synthetic test data to {out}")
