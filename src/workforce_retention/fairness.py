"""Subgroup performance auditing and sensitive-attribute ablation for responsible HR ML."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split

from workforce_retention.data_prep import prepare_feature_matrix
from workforce_retention.modeling import build_preprocessor, make_logistic_regression
from imblearn.pipeline import Pipeline as ImbPipeline


def evaluate_subgroup_performance(
    clean_df: pd.DataFrame,
    test_indices: pd.Index,
    y_test: pd.Series,
    y_prob: np.ndarray,
    threshold: float,
    *,
    metrics_dir: Path | None = None,
    figures_dir: Path | None = None,
) -> tuple[pd.DataFrame, Path | None]:
    """Audit held-out test set error rates and selection rates across demographic subgroups."""
    sub_df = clean_df.loc[test_indices, ["Gender", "Age", "MaritalStatus"]].copy()
    sub_df["AgeGroup"] = pd.Categorical(
        pd.cut(
            sub_df["Age"],
            bins=[17, 29, 44, 100],
            labels=["Under 30", "30-44", "45+"],
        ),
        categories=["Under 30", "30-44", "45+"],
        ordered=True,
    )
    sub_df["AgeADEA40Plus"] = np.where(sub_df["Age"] >= 40, "40+ (ADEA Protected)", "Under 40")
    sub_df["y_true"] = y_test.to_numpy(dtype=int)
    sub_df["y_prob"] = y_prob
    sub_df["y_pred"] = (y_prob >= threshold).astype(int)

    audit_dimensions = ["Gender", "AgeGroup", "AgeADEA40Plus", "MaritalStatus"]
    rows: list[dict[str, Any]] = []

    for dim in audit_dimensions:
        for group_val, grp in sub_df.groupby(dim, observed=False):
            yt = grp["y_true"].to_numpy()
            yp = grp["y_pred"].to_numpy()
            ypr = grp["y_prob"].to_numpy()
            tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
            n_grp = len(grp)
            pos_cnt = int(yt.sum())
            fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
            pr_auc_val = (
                float(average_precision_score(yt, ypr))
                if 0 < pos_cnt < n_grp
                else float("nan")
            )
            rows.append(
                {
                    "dimension": dim,
                    "subgroup": str(group_val),
                    "n_test": int(n_grp),
                    "attrition_count": pos_cnt,
                    "base_attrition_rate": round(float(yt.mean()), 4),
                    "flagged_selection_rate": round(float(yp.mean()), 4),
                    "precision": round(float(precision_score(yt, yp, zero_division=0)), 4),
                    "recall_tpr": round(float(recall_score(yt, yp, zero_division=0)), 4),
                    "false_positive_rate": round(fpr, 4),
                    "f1": round(float(f1_score(yt, yp, zero_division=0)), 4),
                    "pr_auc": round(pr_auc_val, 4) if not np.isnan(pr_auc_val) else None,
                }
            )

    audit_df = pd.DataFrame(rows)

    if metrics_dir is not None:
        metrics_dir.mkdir(parents=True, exist_ok=True)
        audit_df.to_csv(metrics_dir / "subgroup_audit.csv", index=False)

    fig_path: Path | None = None
    if figures_dir is not None:
        figures_dir.mkdir(parents=True, exist_ok=True)
        sns.set_theme(style="whitegrid", context="notebook", font_scale=0.98)
        plot_dims = ["Gender", "AgeGroup", "MaritalStatus"]
        fig, axes = plt.subplots(1, 3, figsize=(15.2, 5.0), sharey=True)

        for ax, dim in zip(axes, plot_dims):
            d_sub = audit_df[audit_df["dimension"] == dim].reset_index(drop=True)
            x = np.arange(len(d_sub))
            width = 0.26
            ax.bar(x - width, d_sub["flagged_selection_rate"] * 100.0, width, label="Flagged Rate", color="#6c757d", edgecolor="black", linewidth=0.5)
            ax.bar(x, d_sub["recall_tpr"] * 100.0, width, label="Recall (TPR)", color="#2b5c8f", edgecolor="black", linewidth=0.5)
            ax.bar(x + width, d_sub["false_positive_rate"] * 100.0, width, label="False Positive Rate", color="#d95f02", edgecolor="black", linewidth=0.5)
            ax.set_xticks(x)
            ax.set_xticklabels([f"{row.subgroup}\n(n={row.n_test})" for row in d_sub.itertuples()])
            ax.set_title(f"Subgroup Audit: {dim}", fontweight="bold")
            ax.set_ylim(0, 105)

        axes[0].set_ylabel("Rate (%) on Held-Out Test Set")
        axes[2].legend(loc="upper right", fontsize=8.8)
        fig.suptitle(
            f"Responsible-AI Subgroup Error & Selection Rate Audit (Threshold t* = {threshold:.2f})",
            fontsize=12.8,
            fontweight="bold",
            y=1.03,
        )
        fig.tight_layout()
        fig_path = figures_dir / "fig11_subgroup_audit.png"
        fig.savefig(fig_path, dpi=220, bbox_inches="tight")
        plt.close(fig)

    return audit_df, fig_path


def run_demographic_ablation(
    clean_df: pd.DataFrame,
    *,
    test_size: float = 0.20,
    cv_folds: int = 5,
    random_seed: int = 42,
    metrics_dir: Path | None = None,
) -> dict[str, Any]:
    """Compare L1 Logistic Regression performance with vs. without Age, Gender, MaritalStatus."""
    results: dict[str, Any] = {}
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_seed)

    for label, drop_sens in [
        ("with_demographics", False),
        ("without_age_gender_marital", True),
    ]:
        X_curr, y_curr = prepare_feature_matrix(clean_df, drop_sensitive=drop_sens)
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_curr, y_curr, test_size=test_size, stratify=y_curr, random_state=random_seed
        )
        pipe = ImbPipeline(
            steps=[
                ("preprocessor", build_preprocessor(X_tr)),
                (
                    "classifier",
                    make_logistic_regression(
                        l1_regularized=True,
                        C=0.5,
                        class_weight="balanced",
                        random_state=random_seed,
                    ),
                ),
            ]
        )
        cv_res = cross_validate(
            pipe,
            X_tr,
            y_tr,
            cv=skf,
            scoring={"pr_auc": "average_precision", "roc_auc": "roc_auc"},
        )
        pipe.fit(X_tr, y_tr)
        prob_te = pipe.predict_proba(X_te)[:, 1]
        results[label] = {
            "feature_count": int(X_curr.shape[1]),
            "cv_pr_auc_mean": round(float(np.mean(cv_res["test_pr_auc"])), 4),
            "test_pr_auc": round(float(average_precision_score(y_te, prob_te)), 4),
            "test_roc_auc": round(float(roc_auc_score(y_te, prob_te)), 4),
        }

    if metrics_dir is not None:
        metrics_dir.mkdir(parents=True, exist_ok=True)
        (metrics_dir / "fairness_ablation.json").write_text(
            json.dumps(results, indent=2), encoding="utf-8"
        )

    return results
