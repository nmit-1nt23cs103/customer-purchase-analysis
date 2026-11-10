"""Central configuration: paths, feature lists and constants."""
from __future__ import annotations

import os
from pathlib import Path

# ----------------------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
RAW_CSV = Path(os.environ.get("RAW_DATA_PATH", RAW_DIR / "marketing_campaign.csv"))
PROCESSED_DIR = ROOT / "data" / "processed"
CLEAN_CSV = PROCESSED_DIR / "customers_clean.csv"
SEGMENTS_CSV = PROCESSED_DIR / "customer_segments.csv"
DB_PATH = PROCESSED_DIR / "customers.db"
SQL_DIR = ROOT / "sql"
REPORT_DIR = ROOT / "reports"
FIG_DIR = REPORT_DIR / "figures"
SQL_RESULTS_DIR = REPORT_DIR / "sql_results"
MODEL_DIR = ROOT / "models"

KAGGLE_DATASET = "imakash3011/customer-personality-analysis"
KAGGLE_URL = "https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis"

RANDOM_STATE = 42

# ----------------------------------------------------------------------------- schema
SPEND_COLS = ["spend_wines", "spend_fruits", "spend_meat", "spend_fish", "spend_sweets", "spend_gold"]
CHANNEL_COLS = ["web_purchases", "catalog_purchases", "store_purchases"]
CAMPAIGN_COLS = [f"accepted_cmp{i}" for i in range(1, 6)]

# ----------------------------------------------------------------------------- segmentation
# These columns must also appear in the `clustering_input` query in sql/extract.sql.
CLUSTER_FEATURES = [
    "income",
    "age",
    "children",
    "total_spend",
    "total_purchases",
    "web_visits_month",
    "deal_purchases",
    "catalog_purchases",
]
LOG_FEATURES = ["total_spend", "deal_purchases", "catalog_purchases"]  # right-skewed -> log1p
K_SEARCH_RANGE = range(2, 10)      # evaluated and plotted
K_SELECTION_RANGE = range(3, 7)    # k is picked from here (k=2 is rarely actionable)

# ----------------------------------------------------------------------------- classification
TARGET = "response"  # accepted the last marketing campaign
NUMERIC_FEATURES = [
    "income", "age", "recency_days", "tenure_days",
    "kidhome", "teenhome", "children", "family_size",
    *SPEND_COLS, "total_spend", "avg_spend_per_purchase",
    "deal_purchases", *CHANNEL_COLS, "total_purchases", "web_visits_month",
    *CAMPAIGN_COLS, "complain",
]
CATEGORICAL_FEATURES = ["education_level", "living_with", "segment_id"]
TEST_SIZE = 0.2
CV_FOLDS = 5

# ----------------------------------------------------------------------------- labels for charts
LABELS = {
    "income": "Income", "age": "Age", "children": "Children", "total_spend": "Total spend",
    "total_purchases": "Total purchases", "web_visits_month": "Web visits / month",
    "deal_purchases": "Deal purchases", "catalog_purchases": "Catalog purchases",
    "web_purchases": "Web purchases", "store_purchases": "Store purchases",
    "recency_days": "Days since last purchase", "tenure_days": "Customer tenure (days)",
    "kidhome": "Kids at home", "teenhome": "Teens at home", "family_size": "Family size",
    "spend_wines": "Spend: wines", "spend_fruits": "Spend: fruits", "spend_meat": "Spend: meat",
    "spend_fish": "Spend: fish", "spend_sweets": "Spend: sweets", "spend_gold": "Spend: gold",
    "avg_spend_per_purchase": "Avg spend / purchase", "complain": "Complained",
    "campaigns_accepted": "Past campaigns accepted", "response": "Responded to last campaign",
    "education_level": "Education level", "living_with": "Living with", "segment_id": "Segment",
    **{f"accepted_cmp{i}": f"Accepted campaign {i}" for i in range(1, 6)},
}


def label(col: str) -> str:
    return LABELS.get(col, col.replace("_", " ").capitalize())
