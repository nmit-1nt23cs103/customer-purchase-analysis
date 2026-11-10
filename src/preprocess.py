"""Step 1 - load the raw Kaggle file, clean it, engineer features, save CSV + SQLite table."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from .db import write_table
from .utils import df_to_markdown

RENAME = {
    "ID": "customer_id", "Year_Birth": "year_birth", "Education": "education",
    "Marital_Status": "marital_status", "Income": "income", "Kidhome": "kidhome",
    "Teenhome": "teenhome", "Dt_Customer": "signup_date", "Recency": "recency_days",
    "MntWines": "spend_wines", "MntFruits": "spend_fruits", "MntMeatProducts": "spend_meat",
    "MntFishProducts": "spend_fish", "MntSweetProducts": "spend_sweets", "MntGoldProds": "spend_gold",
    "NumDealsPurchases": "deal_purchases", "NumWebPurchases": "web_purchases",
    "NumCatalogPurchases": "catalog_purchases", "NumStorePurchases": "store_purchases",
    "NumWebVisitsMonth": "web_visits_month",
    "AcceptedCmp1": "accepted_cmp1", "AcceptedCmp2": "accepted_cmp2", "AcceptedCmp3": "accepted_cmp3",
    "AcceptedCmp4": "accepted_cmp4", "AcceptedCmp5": "accepted_cmp5",
    "Complain": "complain", "Response": "response",
}
CONSTANT_COLS = ["Z_CostContact", "Z_Revenue"]  # constant in the Kaggle file -> no information

EDUCATION_MAP = {
    "Basic": "Undergraduate", "2n Cycle": "Undergraduate",
    "Graduation": "Graduate", "Master": "Postgraduate", "PhD": "Postgraduate",
}
MARITAL_MAP = {  # "Alone", "Absurd" and "YOLO" are data-entry oddities in the Kaggle file
    "Alone": "Single", "Absurd": "Single", "YOLO": "Single",
}
PARTNER_STATUSES = {"Married", "Together"}


def load_raw(path=None) -> pd.DataFrame:
    path = path or C.RAW_CSV
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}.\n"
            f"Run `python -m src.download_data`, or download marketing_campaign.csv from\n"
            f"{C.KAGGLE_URL}\nand place it at {C.RAW_CSV}."
        )
    df = pd.read_csv(path, sep="\t")  # the Kaggle file is tab-separated
    if df.shape[1] == 1:  # fall back to comma-separated variants
        df = pd.read_csv(path, sep=",")
    missing = sorted(set(RENAME) - set(df.columns))
    if missing:
        raise ValueError(f"Raw file is missing expected columns: {missing}")
    return df


def _parse_dates(s: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(s, format="%d-%m-%Y", errors="coerce")
    if parsed.isna().any():
        parsed = pd.to_datetime(s, dayfirst=True, errors="coerce", format="mixed")
    if parsed.isna().any():
        raise ValueError("Could not parse some signup dates")
    return parsed


def clean(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Return (clean_df, stats). Pure function - no file access - so it is easy to test."""
    stats: dict = {"rows_raw": len(raw)}
    df = raw.rename(columns=RENAME).drop(columns=CONSTANT_COLS, errors="ignore").copy()

    # --- types & text tidy-up
    df["signup_date"] = _parse_dates(df["signup_date"])
    for col in ("education", "marital_status"):
        df[col] = df[col].astype(str).str.strip()
    stats["marital_recoded"] = int(df["marital_status"].isin(MARITAL_MAP).sum())
    df["marital_status"] = df["marital_status"].replace(MARITAL_MAP)

    # --- duplicates (identical on every attribute except the ID)
    before = len(df)
    df = df.drop_duplicates(subset=[c for c in df.columns if c != "customer_id"])
    stats["duplicates_removed"] = before - len(df)

    # --- sanity checks
    if (df[C.SPEND_COLS + C.CHANNEL_COLS] < 0).any().any():
        raise ValueError("Negative spend or purchase counts found")

    # --- missing income: impute with the median income of the customer's education group
    stats["income_missing"] = int(df["income"].isna().sum())
    group_median = df.groupby("education")["income"].transform("median")
    df["income"] = df["income"].fillna(group_median).fillna(df["income"].median())

    # --- feature engineering
    snapshot = df["signup_date"].max() + pd.Timedelta(days=1)  # dataset "today"
    stats["snapshot_date"] = snapshot.strftime("%Y-%m-%d")
    df["age"] = snapshot.year - df["year_birth"]
    df["tenure_days"] = (snapshot - df["signup_date"]).dt.days
    df["education_level"] = df["education"].map(EDUCATION_MAP).fillna(df["education"])
    df["living_with"] = np.where(df["marital_status"].isin(PARTNER_STATUSES), "Partner", "Alone")
    df["children"] = df["kidhome"] + df["teenhome"]
    df["is_parent"] = (df["children"] > 0).astype(int)
    df["family_size"] = df["children"] + np.where(df["living_with"] == "Partner", 2, 1)
    df["total_spend"] = df[C.SPEND_COLS].sum(axis=1)
    df["total_purchases"] = df[C.CHANNEL_COLS].sum(axis=1)
    df["campaigns_accepted"] = df[C.CAMPAIGN_COLS].sum(axis=1)
    df["avg_spend_per_purchase"] = (
        df["total_spend"] / df["total_purchases"].replace(0, np.nan)
    ).fillna(0).round(2)

    # --- outliers: impossible ages and extreme incomes (Q3 + 3 * IQR rule)
    q1, q3 = df["income"].quantile([0.25, 0.75])
    income_cap = q3 + 3 * (q3 - q1)
    bad_age = df["age"] > 100
    bad_income = df["income"] > income_cap
    stats.update(
        age_outliers_removed=int(bad_age.sum()),
        income_outliers_removed=int(bad_income.sum()),
        income_cap=float(round(income_cap)),
    )
    df = df[~(bad_age | bad_income)].copy()

    # --- final tidy
    df["signup_date"] = df["signup_date"].dt.strftime("%Y-%m-%d")
    df["income"] = df["income"].round(0)
    stats["rows_clean"] = len(df)
    ordered = [
        "customer_id", "year_birth", "age", "education", "education_level", "marital_status",
        "living_with", "income", "kidhome", "teenhome", "children", "is_parent", "family_size",
        "signup_date", "tenure_days", "recency_days", *C.SPEND_COLS, "total_spend",
        "deal_purchases", *C.CHANNEL_COLS, "total_purchases", "avg_spend_per_purchase",
        "web_visits_month", *C.CAMPAIGN_COLS, "campaigns_accepted", "complain", "response",
    ]
    return df[ordered].reset_index(drop=True), stats


def _summary_md(stats: dict, df: pd.DataFrame) -> str:
    table = pd.DataFrame(
        [
            ("Raw rows", stats["rows_raw"]),
            ("Duplicate customers removed (identical except ID)", stats["duplicates_removed"]),
            ("Missing income values imputed (education-group median)", stats["income_missing"]),
            ("Impossible ages removed (> 100 years)", stats["age_outliers_removed"]),
            (f"Income outliers removed (> {stats['income_cap']:,.0f}, Q3 + 3xIQR)", stats["income_outliers_removed"]),
            ("Marital-status oddities recoded to 'Single' (Alone / Absurd / YOLO)", stats["marital_recoded"]),
            ("**Clean rows**", stats["rows_clean"]),
        ],
        columns=["Step", "Rows"],
    )
    return (
        "### Data cleaning\n\n"
        f"{df_to_markdown(table)}\n\n"
        "Engineered features: `age`, `tenure_days`, `total_spend`, `total_purchases`, `children`, "
        "`family_size`, `is_parent`, `campaigns_accepted`, `avg_spend_per_purchase`, "
        "`education_level`, `living_with`. "
        f"Age and tenure are measured at the dataset snapshot date ({stats['snapshot_date']}).\n"
    )


def run() -> dict:
    raw = load_raw()
    df, stats = clean(raw)
    C.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    C.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(C.CLEAN_CSV, index=False)
    write_table(df, "customers")
    summary = _summary_md(stats, df)
    (C.REPORT_DIR / "data_cleaning_summary.md").write_text(summary, encoding="utf-8")
    print(f"Cleaned {stats['rows_raw']} -> {stats['rows_clean']} rows; saved {C.CLEAN_CSV.name} and SQLite table 'customers'")
    return {"stats": stats, "summary_md": summary}


if __name__ == "__main__":
    run()
