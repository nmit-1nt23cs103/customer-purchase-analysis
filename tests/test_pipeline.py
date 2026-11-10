"""Unit tests on synthetic data: cleaning rules, SQL files and segment naming."""
import sqlite3

import numpy as np
import pandas as pd

from src import config as C
from src.db import load_named_queries
from src.preprocess import clean
from src.segmentation import name_segments
from src.utils import df_to_markdown
from tests.make_synthetic_data import make_synthetic


def _clean():
    return clean(make_synthetic(n=900, seed=1))


def test_cleaning_rules():
    df, stats = _clean()
    assert df["income"].notna().all()
    assert df["age"].max() <= 100
    assert df["income"].max() < 200_000          # the 666,666 outlier is gone
    assert df["customer_id"].is_unique
    assert not {"Z_CostContact", "Z_Revenue"} & set(df.columns)
    assert stats["duplicates_removed"] >= 25
    assert stats["income_missing"] > 0
    assert set(df["marital_status"]) <= {"Married", "Together", "Single", "Divorced", "Widowed"}
    assert set(df["living_with"]) <= {"Partner", "Alone"}


def test_engineered_features_are_consistent():
    df, _ = _clean()
    assert (df["total_spend"] == df[C.SPEND_COLS].sum(axis=1)).all()
    assert (df["total_purchases"] == df[C.CHANNEL_COLS].sum(axis=1)).all()
    assert (df["children"] == df["kidhome"] + df["teenhome"]).all()
    assert (df["campaigns_accepted"] == df[C.CAMPAIGN_COLS].sum(axis=1)).all()
    assert (df["tenure_days"] > 0).all()


def test_all_sql_queries_run():
    df, _ = _clean()
    conn = sqlite3.connect(":memory:")
    df.to_sql("customers", conn, index=False)
    seg = pd.DataFrame({"customer_id": df["customer_id"], "segment_id": df["customer_id"] % 3})
    seg["segment_name"] = "Segment " + seg["segment_id"].astype(str)
    seg.to_sql("segments", conn, index=False)
    for file in ("extract.sql", "analysis.sql"):
        queries = load_named_queries(C.SQL_DIR / file)
        assert queries, f"no queries parsed from {file}"
        for name, (sql, _) in queries.items():
            out = pd.read_sql_query(sql, conn)
            assert len(out) > 0, f"{name} returned no rows"


def test_decile_query_adds_up():
    df, _ = _clean()
    conn = sqlite3.connect(":memory:")
    df.to_sql("customers", conn, index=False)
    sql, _ = load_named_queries(C.SQL_DIR / "analysis.sql")["spend_concentration_by_decile"]
    out = pd.read_sql_query(sql, conn)
    assert len(out) == 10
    assert abs(out["pct_of_total_spend"].sum() - 100) < 1
    assert abs(out["cumulative_pct"].iloc[-1] - 100) < 0.5


def test_segment_naming_is_unique_and_ordered():
    profile = pd.DataFrame({"children": [0.2, 1.4, 1.0, 0.3]}, index=[0, 1, 2, 3])
    names = name_segments(profile, overall_children=0.9)
    assert len(set(names.values())) == 4
    assert "Premium" in names[0] and "Budget" in names[3]


def test_markdown_table():
    md = df_to_markdown(pd.DataFrame({"a": [1, 2], "b": [0.5, np.nan]}))
    assert md.splitlines()[0] == "| a | b |"
    assert len(md.splitlines()) == 4
