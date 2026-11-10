"""Step 3 - customer segmentation with K-Means (features are extracted with SQL)."""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (adjusted_rand_score, calinski_harabasz_score, davies_bouldin_score,
                             silhouette_score)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from . import config as C
from .db import load_named_queries, query, write_table
from .plotting import (AXIS, CATEGORICAL, DIV_CMAP, GRID, INK, INK2, MARKERS, SURFACE, save_fig,
                       style_axes)
from .utils import df_to_markdown


# ----------------------------------------------------------------------------- data
def extract() -> pd.DataFrame:
    """Pull the clustering input with the SQL query in sql/extract.sql."""
    sql, _ = load_named_queries(C.SQL_DIR / "extract.sql")["clustering_input"]
    df = query(sql)
    missing = [c for c in C.CLUSTER_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Columns {missing} are in config.CLUSTER_FEATURES but not in sql/extract.sql")
    return df


def build_matrix(df: pd.DataFrame) -> np.ndarray:
    """log1p the skewed features, then standardise so every feature has equal weight."""
    X = df[C.CLUSTER_FEATURES].astype(float).copy()
    for col in C.LOG_FEATURES:
        X[col] = np.log1p(X[col])
    return StandardScaler().fit_transform(X)


# ----------------------------------------------------------------------------- model selection
def evaluate_k(Xs: np.ndarray) -> pd.DataFrame:
    rows = []
    for k in C.K_SEARCH_RANGE:
        km = KMeans(n_clusters=k, n_init=20, random_state=C.RANDOM_STATE).fit(Xs)
        rows.append({
            "k": k,
            "inertia": km.inertia_,
            "silhouette": silhouette_score(Xs, km.labels_),
            "davies_bouldin": davies_bouldin_score(Xs, km.labels_),
            "calinski_harabasz": calinski_harabasz_score(Xs, km.labels_),
        })
    return pd.DataFrame(rows)


def choose_k(scores: pd.DataFrame) -> int:
    cand = scores[scores["k"].isin(list(C.K_SELECTION_RANGE))]
    return int(cand.loc[cand["silhouette"].idxmax(), "k"])


def stability(Xs: np.ndarray, k: int, n_runs: int = 8) -> float:
    """Mean pairwise adjusted Rand index across K-Means runs with different seeds (1.0 = identical)."""
    runs = [KMeans(n_clusters=k, n_init=5, random_state=s).fit_predict(Xs) for s in range(n_runs)]
    aris = [adjusted_rand_score(runs[i], runs[j]) for i in range(n_runs) for j in range(i + 1, n_runs)]
    return float(np.mean(aris))


def compare_algorithms(Xs: np.ndarray, k: int, km_labels: np.ndarray) -> pd.DataFrame:
    agg = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(Xs)
    gmm = GaussianMixture(n_components=k, n_init=5, random_state=C.RANDOM_STATE).fit(Xs).predict(Xs)
    rows = [
        ("K-Means", silhouette_score(Xs, km_labels), 1.0),
        ("Agglomerative (Ward)", silhouette_score(Xs, agg), adjusted_rand_score(km_labels, agg)),
        ("Gaussian Mixture", silhouette_score(Xs, gmm), adjusted_rand_score(km_labels, gmm)),
    ]
    return pd.DataFrame(rows, columns=["algorithm", "silhouette", "agreement_with_kmeans (ARI)"])


# ----------------------------------------------------------------------------- naming
def name_segments(profile: pd.DataFrame, overall_children: float) -> dict[int, str]:
    """Descriptive names from spend rank and household type. Segment 0 = highest average spend."""
    n = len(profile)
    names = {}
    for sid in profile.index:
        tier = "Premium" if sid == 0 else "Budget" if sid == n - 1 else "Mid-range"
        family = "parents" if profile.loc[sid, "children"] > overall_children else "few kids"
        names[int(sid)] = f"Segment {sid} - {tier}, {family}"
    return names


# ----------------------------------------------------------------------------- charts
def _plot_k_selection(scores: pd.DataFrame, k: int):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    ax = axes[0]
    ax.plot(scores["k"], scores["inertia"], color=CATEGORICAL[0], linewidth=2, marker="o", markersize=6,
            markeredgecolor=SURFACE, markeredgewidth=1.5)
    ax.set_title("Elbow: within-cluster inertia")
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Inertia")
    style_axes(ax, "y")

    ax = axes[1]
    lo, hi = min(C.K_SELECTION_RANGE), max(C.K_SELECTION_RANGE)
    ax.axvspan(lo - 0.5, hi + 0.5, color=GRID, alpha=0.6, linewidth=0)
    ax.plot(scores["k"], scores["silhouette"], color=CATEGORICAL[0], linewidth=2, marker="o", markersize=6,
            markeredgecolor=SURFACE, markeredgewidth=1.5)
    best = scores.loc[scores["k"] == k].iloc[0]
    ax.scatter([k], [best["silhouette"]], s=110, color=CATEGORICAL[1], zorder=5,
               edgecolor=SURFACE, linewidth=1.5)
    ax.annotate(f"chosen k = {k}", (k, best["silhouette"]), textcoords="offset points", xytext=(10, 8),
                color=INK, fontsize=10)
    ax.text(0.98, 0.04, "shaded: k considered", transform=ax.transAxes, ha="right", color=INK2, fontsize=9)
    ax.set_title("Silhouette score")
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Silhouette")
    style_axes(ax, "y")
    return save_fig(fig, "segments_k_selection.png")


def _plot_pca(Xs: np.ndarray, labels: np.ndarray, names: dict[int, str]):
    pca = PCA(n_components=2, random_state=C.RANDOM_STATE)
    Z = pca.fit_transform(Xs)
    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    for sid in sorted(names):
        m = labels == sid
        ax.scatter(Z[m, 0], Z[m, 1], s=16, alpha=0.7, color=CATEGORICAL[sid % len(CATEGORICAL)],
                   marker=MARKERS[sid % len(MARKERS)], edgecolor=SURFACE, linewidth=0.3, label=names[sid])
        cx, cy = Z[m, 0].mean(), Z[m, 1].mean()
        ax.text(cx, cy, f"S{sid}", ha="center", va="center", fontsize=11, fontweight="bold", color=INK,
                path_effects=[pe.withStroke(linewidth=3, foreground=SURFACE)])
    ev = pca.explained_variance_ratio_ * 100
    ax.set_xlabel(f"Principal component 1 ({ev[0]:.0f}% of variance)")
    ax.set_ylabel(f"Principal component 2 ({ev[1]:.0f}% of variance)")
    ax.set_title("Customer segments in PCA space")
    style_axes(ax, "y")
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), markerscale=1.8, labelcolor=INK2)
    return save_fig(fig, "segments_pca.png")


def _plot_profile_heatmap(df: pd.DataFrame, names: dict[int, str]):
    feats = C.CLUSTER_FEATURES
    means = df.groupby("segment_id")[feats].mean()
    z = (means - df[feats].mean()) / df[feats].std()
    vmax = max(1.0, float(np.abs(z.values).max()))
    fig, ax = plt.subplots(figsize=(10, 0.8 * len(z) + 2.2))
    im = ax.imshow(z.values, cmap=DIV_CMAP, vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(feats)), [C.label(f) for f in feats], rotation=35, ha="right")
    ax.set_yticks(range(len(z)), [names[i] for i in z.index])
    for i in range(z.shape[0]):
        for j in range(z.shape[1]):
            v = z.values[i, j]
            ax.text(j, i, f"{v:+.1f}", ha="center", va="center", fontsize=9,
                    color="white" if abs(v) > 0.55 * vmax else INK)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title("Segment profile: standard deviations above / below the overall average")
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.outline.set_edgecolor(AXIS)
    cbar.ax.tick_params(colors=INK2)
    return save_fig(fig, "segments_profile_heatmap.png")


# ----------------------------------------------------------------------------- pipeline step
def run(k: int | None = None) -> dict:
    df = extract()
    Xs = build_matrix(df)
    scores = evaluate_k(Xs)
    k = k or choose_k(scores)

    km = KMeans(n_clusters=k, n_init=50, random_state=C.RANDOM_STATE).fit(Xs)
    # Renumber clusters so Segment 0 has the highest average spend, Segment k-1 the lowest
    order = df.assign(c=km.labels_).groupby("c")["total_spend"].mean().sort_values(ascending=False).index
    remap = {int(old): new for new, old in enumerate(order)}
    labels = np.array([remap[int(c)] for c in km.labels_])
    df["segment_id"] = labels

    profile = df.groupby("segment_id")[C.CLUSTER_FEATURES].mean()
    names = name_segments(profile, df["children"].mean())
    df["segment_name"] = df["segment_id"].map(names)

    segments = df[["customer_id", "segment_id", "segment_name"]]
    segments.to_csv(C.SEGMENTS_CSV, index=False)
    write_table(segments, "segments")

    metrics = {
        "k": k,
        "silhouette": float(silhouette_score(Xs, labels)),
        "davies_bouldin": float(davies_bouldin_score(Xs, labels)),
        "stability_ari": stability(Xs, k),
        "n_customers": int(len(df)),
    }
    algos = compare_algorithms(Xs, k, labels)

    C.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    scores.to_csv(C.REPORT_DIR / "kmeans_k_scores.csv", index=False)
    figs = [_plot_k_selection(scores, k), _plot_pca(Xs, labels, names), _plot_profile_heatmap(df, names)]

    table = profile.round(1).assign(customers=df["segment_id"].value_counts().sort_index())
    table.insert(0, "segment", [names[i] for i in table.index])
    table = table.reset_index(drop=True)
    table["customers"] = table.pop("customers")
    table = table.rename(columns={c: C.label(c) for c in C.CLUSTER_FEATURES})
    table["Income"] = table["Income"].round(0).astype(int)

    summary = (
        "### Customer segmentation (K-Means)\n\n"
        f"- **Features:** {', '.join(C.label(c).lower() for c in C.CLUSTER_FEATURES)} "
        f"(skewed spend features log-transformed, then standardised)\n"
        f"- **k = {k}**, chosen by silhouette score among k = {min(C.K_SELECTION_RANGE)}-{max(C.K_SELECTION_RANGE)} "
        f"(elbow chart below)\n"
        f"- **Silhouette {metrics['silhouette']:.3f}**, Davies-Bouldin {metrics['davies_bouldin']:.3f}, "
        f"run-to-run stability (adjusted Rand index over 8 seeds) {metrics['stability_ari']:.3f}\n\n"
        f"{df_to_markdown(table, float_digits=1)}\n\n"
        "Algorithm cross-check at the same k:\n\n"
        f"{df_to_markdown(algos, float_digits=3)}\n\n"
        "![Segments in PCA space]({FIG}/segments_pca.png)\n\n"
        "![Segment profile heatmap]({FIG}/segments_profile_heatmap.png)\n\n"
        "![Choosing k]({FIG}/segments_k_selection.png)\n"
    )
    (C.REPORT_DIR / "segmentation_summary.md").write_text(summary.replace("{FIG}", "figures"), encoding="utf-8")
    print(f"k={k}  silhouette={metrics['silhouette']:.3f}  stability ARI={metrics['stability_ari']:.3f}")
    print(table.to_string(index=False))
    return {"metrics": metrics, "summary_md": summary, "figures": [f.name for f in figs]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run customer segmentation")
    parser.add_argument("--k", type=int, default=None, help="force the number of clusters")
    run(parser.parse_args().k)
