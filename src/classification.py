"""Step 5 - predict which customers accept the latest marketing campaign.

Target: `response` (1 = accepted the last campaign). About 1 in 7 customers respond, so
the classes are imbalanced: models use class weights, models are compared on PR-AUC and
ROC-AUC (not accuracy), and the decision threshold is tuned on cross-validated predictions
from the training set only - the test set is touched once for the final numbers.
"""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score, recall_score, roc_auc_score,
                             roc_curve)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config as C
from .db import load_named_queries, query
from .plotting import AXIS, CATEGORICAL, INK, INK2, SEQ_CMAP, SURFACE, save_fig, style_axes
from .utils import df_to_markdown


# ----------------------------------------------------------------------------- data & models
def load_dataset() -> pd.DataFrame:
    sql, _ = load_named_queries(C.SQL_DIR / "extract.sql")["classification_dataset"]
    df = query(sql)
    df["segment_id"] = df["segment_id"].astype(str)
    return df


def make_preprocessor() -> ColumnTransformer:
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    return ColumnTransformer([("num", numeric, C.NUMERIC_FEATURES), ("cat", categorical, C.CATEGORICAL_FEATURES)])


def make_models() -> dict[str, Pipeline]:
    rs = C.RANDOM_STATE
    estimators = {
        "Logistic Regression": LogisticRegression(max_iter=3000, C=0.5, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(n_estimators=400, min_samples_leaf=3, n_jobs=-1,
                                                class_weight="balanced_subsample", random_state=rs),
        "Gradient Boosting": HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=300,
                                                            class_weight="balanced", random_state=rs),
    }
    return {name: Pipeline([("prep", make_preprocessor()), ("model", est)]) for name, est in estimators.items()}


def best_f1_threshold(y_true: np.ndarray, proba: np.ndarray) -> float:
    p, r, thr = precision_recall_curve(y_true, proba)
    f1 = 2 * p[:-1] * r[:-1] / np.clip(p[:-1] + r[:-1], 1e-12, None)
    return float(thr[int(np.argmax(f1))])


def evaluate(y_true, proba, threshold: float) -> dict:
    pred = (proba >= threshold).astype(int)
    return {
        "threshold": threshold,
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred),
        "f1": f1_score(y_true, pred),
        "roc_auc": roc_auc_score(y_true, proba),
        "pr_auc": average_precision_score(y_true, proba),
    }


# ----------------------------------------------------------------------------- charts
def _plot_curves(y_test, probas: dict[str, np.ndarray]):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    ax = axes[0]
    ax.plot([0, 1], [0, 1], color=AXIS, linestyle="--", linewidth=1.2)
    for i, (name, p) in enumerate(probas.items()):
        fpr, tpr, _ = roc_curve(y_test, p)
        ax.plot(fpr, tpr, color=CATEGORICAL[i], linewidth=2, label=f"{name} (AUC {roc_auc_score(y_test, p):.2f})")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve (test set)")
    ax.legend(loc="lower right")
    style_axes(ax, "y")

    ax = axes[1]
    prevalence = float(np.mean(y_test))
    ax.axhline(prevalence, color=AXIS, linestyle="--", linewidth=1.2)
    ax.text(0.02, prevalence + 0.015, f"no-skill ({prevalence:.2f})", color=INK2, fontsize=9)
    for i, (name, p) in enumerate(probas.items()):
        prec, rec, _ = precision_recall_curve(y_test, p)
        ax.plot(rec, prec, color=CATEGORICAL[i], linewidth=2,
                label=f"{name} (AP {average_precision_score(y_test, p):.2f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_ylim(0, 1.02)
    ax.set_title("Precision-recall curve (test set)")
    ax.legend(loc="upper right")
    style_axes(ax, "y")
    return save_fig(fig, "clf_roc_pr_curves.png")


def _plot_confusion(y_test, proba, threshold: float, model_name: str):
    cm = confusion_matrix(y_test, (proba >= threshold).astype(int))
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ax.imshow(cm, cmap=SEQ_CMAP)
    row_tot = cm.sum(axis=1, keepdims=True)
    for i in range(2):
        for j in range(2):
            dark = cm[i, j] > cm.max() * 0.5
            ax.text(j, i, f"{cm[i, j]}\n({100 * cm[i, j] / row_tot[i, 0]:.0f}% of row)", ha="center",
                    va="center", color="white" if dark else INK, fontsize=10)
    ax.set_xticks([0, 1], ["No response", "Response"])
    ax.set_yticks([0, 1], ["No response", "Response"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(f"{model_name} (threshold {threshold:.2f})", fontsize=11)
    return save_fig(fig, "clf_confusion_matrix.png")


def _plot_importance(imp: pd.DataFrame):
    top = imp.head(12).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.barh([C.label(c) for c in top.index], top["mean"], xerr=top["std"], color=CATEGORICAL[0],
            edgecolor=SURFACE, linewidth=1.5, height=0.65, error_kw={"ecolor": INK2, "elinewidth": 1, "capsize": 2})
    ax.set_xlabel("Drop in test PR-AUC when the feature is shuffled")
    ax.set_title("Which features drive campaign response?")
    style_axes(ax, "x")
    return save_fig(fig, "clf_feature_importance.png")


# ----------------------------------------------------------------------------- pipeline step
def run() -> dict:
    df = load_dataset()
    missing = [c for c in C.NUMERIC_FEATURES + C.CATEGORICAL_FEATURES + [C.TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns for classification: {missing}")
    X, y = df[C.NUMERIC_FEATURES + C.CATEGORICAL_FEATURES], df[C.TARGET].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=C.TEST_SIZE, stratify=y, random_state=C.RANDOM_STATE)

    cv = StratifiedKFold(n_splits=C.CV_FOLDS, shuffle=True, random_state=C.RANDOM_STATE)
    models = make_models()
    cv_rows, test_probas = [], {}
    for name, pipe in models.items():
        res = cross_validate(pipe, X_train, y_train, cv=cv, n_jobs=1,
                             scoring={"roc_auc": "roc_auc", "pr_auc": "average_precision", "f1": "f1"})
        cv_rows.append({
            "model": name,
            "cv_roc_auc": res["test_roc_auc"].mean(), "cv_roc_auc_std": res["test_roc_auc"].std(),
            "cv_pr_auc": res["test_pr_auc"].mean(), "cv_pr_auc_std": res["test_pr_auc"].std(),
            "cv_f1": res["test_f1"].mean(),
        })
        pipe.fit(X_train, y_train)
        test_probas[name] = pipe.predict_proba(X_test)[:, 1]
        print(f"{name:<20} CV ROC-AUC {cv_rows[-1]['cv_roc_auc']:.3f}  CV PR-AUC {cv_rows[-1]['cv_pr_auc']:.3f}")
    cv_df = pd.DataFrame(cv_rows).sort_values("cv_pr_auc", ascending=False).reset_index(drop=True)
    best_name = cv_df.loc[0, "model"]
    best = models[best_name]

    # Tune the decision threshold on out-of-fold predictions from the TRAIN set only
    oof = cross_val_predict(make_models()[best_name], X_train, y_train, cv=cv, method="predict_proba")[:, 1]
    threshold = best_f1_threshold(y_train.values, oof)
    proba = test_probas[best_name]
    at_default, at_tuned = evaluate(y_test, proba, 0.5), evaluate(y_test, proba, threshold)

    imp = permutation_importance(best, X_test, y_test, scoring="average_precision", n_repeats=15,
                                 random_state=C.RANDOM_STATE, n_jobs=1)
    importance = pd.DataFrame({"mean": imp.importances_mean, "std": imp.importances_std},
                              index=X_test.columns).sort_values("mean", ascending=False)

    figs = [_plot_curves(y_test, test_probas), _plot_confusion(y_test, proba, threshold, best_name),
            _plot_importance(importance)]

    C.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": best, "threshold": threshold, "features": list(X.columns), "model": best_name},
                C.MODEL_DIR / "best_model.joblib")
    cv_df.to_csv(C.REPORT_DIR / "classification_cv_results.csv", index=False)
    importance.to_csv(C.REPORT_DIR / "feature_importance.csv")

    baseline_acc = 1 - float(y_test.mean())
    metrics = {
        "best_model": best_name, "prevalence": float(y.mean()), "baseline_accuracy": baseline_acc,
        "n_train": len(X_train), "n_test": len(X_test), "default": at_default, "tuned": at_tuned,
        "top_features": importance.head(5).index.tolist(),
    }

    cv_show = cv_df.rename(columns={
        "model": "Model", "cv_roc_auc": "CV ROC-AUC", "cv_roc_auc_std": "+/-", "cv_pr_auc": "CV PR-AUC",
        "cv_pr_auc_std": "+/- ", "cv_f1": "CV F1"})
    test_show = pd.DataFrame([
        {"Threshold": "0.50 (default)", **{k: v for k, v in at_default.items() if k != "threshold"}},
        {"Threshold": f"{threshold:.2f} (F1-tuned on train CV)", **{k: v for k, v in at_tuned.items() if k != "threshold"}},
    ]).rename(columns={"accuracy": "Accuracy", "precision": "Precision", "recall": "Recall", "f1": "F1",
                       "roc_auc": "ROC-AUC", "pr_auc": "PR-AUC"})
    top5 = ", ".join(C.label(c) for c in metrics["top_features"])
    summary = (
        "### Campaign-response classification\n\n"
        f"- **Target:** accepted the latest campaign ({100 * metrics['prevalence']:.1f}% of customers), "
        f"so always predicting 'no' already scores {100 * baseline_acc:.1f}% accuracy - models are compared on "
        "ROC-AUC / PR-AUC instead.\n"
        f"- **Split:** {metrics['n_train']} train / {metrics['n_test']} test (stratified), "
        f"{C.CV_FOLDS}-fold stratified cross-validation on the training set.\n"
        f"- **Best model (by CV PR-AUC): {best_name}**\n\n"
        f"{df_to_markdown(cv_show, float_digits=3)}\n\n"
        f"Held-out test performance of {best_name}:\n\n"
        f"{df_to_markdown(test_show, float_digits=3)}\n\n"
        f"Most influential features (permutation importance): {top5}.\n\n"
        "![ROC and precision-recall curves]({FIG}/clf_roc_pr_curves.png)\n\n"
        "![Confusion matrix]({FIG}/clf_confusion_matrix.png)\n\n"
        "![Feature importance]({FIG}/clf_feature_importance.png)\n"
    )
    (C.REPORT_DIR / "classification_summary.md").write_text(summary.replace("{FIG}", "figures"), encoding="utf-8")
    print(f"Best: {best_name}  test ROC-AUC {at_tuned['roc_auc']:.3f}  PR-AUC {at_tuned['pr_auc']:.3f}  "
          f"F1 {at_tuned['f1']:.3f} @ {threshold:.2f}")
    return {"metrics": metrics, "summary_md": summary, "figures": [f.name for f in figs]}


if __name__ == "__main__":
    run()
