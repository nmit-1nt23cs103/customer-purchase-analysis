"""Step 2 - exploratory charts of customer behaviour and purchasing trends."""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from . import config as C
from .db import query
from .plotting import AXIS, CATEGORICAL, DIV_CMAP, INK, INK2, SURFACE, save_fig, style_axes


def _distributions(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
    for ax, col in zip(axes, ["income", "age", "total_spend"]):
        ax.hist(df[col], bins=30, color=CATEGORICAL[0], edgecolor=SURFACE, linewidth=0.8)
        ax.set_title(C.label(col))
        ax.set_ylabel("Customers" if col == "income" else "")
        style_axes(ax, "y")
    return save_fig(fig, "eda_distributions.png")


def _spend_by_category(df: pd.DataFrame):
    totals = df[C.SPEND_COLS].sum().sort_values()
    share = 100 * totals / totals.sum()
    fig, ax = plt.subplots(figsize=(7, 3.6))
    bars = ax.barh([C.label(c).replace("Spend: ", "").capitalize() for c in share.index], share.values,
                   color=CATEGORICAL[0], edgecolor=SURFACE, linewidth=1.5, height=0.62)
    ax.bar_label(bars, fmt="%.1f%%", padding=4, color=INK2, fontsize=9)
    ax.set_xlim(0, share.max() * 1.15)
    ax.set_xlabel("Share of total spend (%)")
    ax.set_title("Spend by product category")
    style_axes(ax, "x")
    return save_fig(fig, "eda_spend_by_category.png")


def _spend_by_signup_quarter(df: pd.DataFrame):
    d = df.assign(q=pd.to_datetime(df["signup_date"]).dt.to_period("Q").astype(str))
    g = d.groupby("q").agg(avg_spend=("total_spend", "mean"), n=("customer_id", "size"))
    g = g[g["n"] >= 30]  # drop thin cohorts at the edges
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.plot(g.index, g["avg_spend"], color=CATEGORICAL[0], linewidth=2, marker="o", markersize=6,
            markeredgecolor=SURFACE, markeredgewidth=1.5)
    ax.set_ylabel("Average total spend")
    ax.set_xlabel("Signup quarter (cohorts with 30+ customers)")
    ax.set_title("Average spend by signup quarter")
    ax.set_ylim(bottom=0)
    ax.tick_params(axis="x", rotation=45)
    style_axes(ax, "y")
    return save_fig(fig, "eda_spend_by_signup_quarter.png")


def _correlation(df: pd.DataFrame):
    cols = ["income", "age", "children", "recency_days", "tenure_days", "total_spend", "total_purchases",
            "web_visits_month", "deal_purchases", "campaigns_accepted", "response"]
    corr = df[cols].corr()
    fig, ax = plt.subplots(figsize=(8.2, 6.6))
    im = ax.imshow(corr.values, cmap=DIV_CMAP, vmin=-1, vmax=1)
    labels = [C.label(c) for c in cols]
    ax.set_xticks(range(len(cols)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(cols)), labels)
    for i in range(len(cols)):
        for j in range(len(cols)):
            v = corr.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if abs(v) > 0.6 else INK)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title("Correlation between key customer metrics")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.outline.set_edgecolor(AXIS)
    cbar.ax.tick_params(colors=INK2)
    return save_fig(fig, "eda_correlation.png")


def run() -> dict:
    df = query("SELECT * FROM customers")
    figs = [_distributions(df), _spend_by_category(df), _spend_by_signup_quarter(df), _correlation(df)]
    print(f"Saved {len(figs)} EDA figures to {C.FIG_DIR}")
    return {"figures": [f.name for f in figs]}


if __name__ == "__main__":
    run()
