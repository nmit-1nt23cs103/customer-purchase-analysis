"""Small helpers shared by the pipeline steps."""
from __future__ import annotations

import time
from contextlib import contextmanager

import numpy as np
import pandas as pd


def df_to_markdown(df: pd.DataFrame, float_digits: int = 2, index: bool = False) -> str:
    """Render a DataFrame as a GitHub-flavoured markdown table (no extra dependencies)."""
    d = df.reset_index() if index else df

    def fmt(v) -> str:
        if v is None or (not isinstance(v, str) and pd.isna(v)):
            return ""
        if isinstance(v, (bool, np.bool_)):
            return str(v)
        if isinstance(v, (int, np.integer)):
            return f"{v:,}"
        if isinstance(v, (float, np.floating)):
            return f"{v:,.{float_digits}f}"
        return str(v).replace("|", "\\|")

    header = "| " + " | ".join(str(c) for c in d.columns) + " |"
    sep = "| " + " | ".join("---" for _ in d.columns) + " |"
    rows = ["| " + " | ".join(fmt(v) for v in row) + " |" for row in d.itertuples(index=False, name=None)]
    return "\n".join([header, sep, *rows])


@contextmanager
def step(name: str):
    """Print a banner and the elapsed time around a pipeline step."""
    print(f"\n=== {name} ===")
    start = time.time()
    yield
    print(f"--- done in {time.time() - start:.1f}s")
