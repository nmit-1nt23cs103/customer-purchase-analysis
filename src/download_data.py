"""Download the Kaggle 'Customer Personality Analysis' dataset into data/raw/.

Usage:
    pip install kagglehub
    python -m src.download_data

Kaggle may ask for credentials. Set KAGGLE_USERNAME / KAGGLE_KEY, or place kaggle.json in ~/.kaggle/
(create an API token at https://www.kaggle.com/settings). You can also download
`marketing_campaign.csv` by hand from the dataset page and copy it to data/raw/.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

from . import config as C


def main() -> None:
    C.RAW_DIR.mkdir(parents=True, exist_ok=True)
    if C.RAW_CSV.exists():
        print(f"Dataset already present: {C.RAW_CSV}")
        return
    try:
        import kagglehub
    except ImportError:
        sys.exit(
            "kagglehub is not installed. Run `pip install kagglehub`, or download "
            f"marketing_campaign.csv manually from {C.KAGGLE_URL} and place it at {C.RAW_CSV}."
        )
    folder = Path(kagglehub.dataset_download(C.KAGGLE_DATASET))
    matches = list(folder.rglob("marketing_campaign.csv"))
    if not matches:
        found = [p.name for p in folder.rglob("*") if p.is_file()]
        sys.exit(f"marketing_campaign.csv not found in the download. Files received: {found}")
    shutil.copy(matches[0], C.RAW_CSV)
    print(f"Saved dataset to {C.RAW_CSV}")


if __name__ == "__main__":
    main()
