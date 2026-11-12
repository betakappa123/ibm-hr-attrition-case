"""Leak-free model training, within-fold imbalance handling, threshold selection, and evaluation."""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from workforce_retention.config import CATEGORICAL_FEATURES, NUMERIC_FEATURES


def make_logistic_regression(
    *,
    l1_regularized: bool,
    C: float = 1.0,
    class_weight: str | dict[int, float] | None = None,
    random_state: int = 42,
    max_iter: int = 5000,
) -> LogisticRegression:
    """Instantiate LogisticRegression cleanly across scikit-learn versions without warnings."""
    sig = inspect.signature(LogisticRegression)
    uses_l1_ratio_api = sig.parameters["penalty"].default == "deprecated"

    if l1_regularized:
        if uses_l1_ratio_api:
            return LogisticRegression(
                l1_ratio=1.0,
                C=C,
                solver="saga",
                class_weight=class_weight,
                random_state=random_state,
                max_iter=max_iter,
            )
        return LogisticRegression(
            penalty="l1",
            C=C,
            solver="saga",
            class_weight=class_weight,
            random_state=random_state,
            max_iter=max_iter,
        )

    if uses_l1_ratio_api:
        return LogisticRegression(
            l1_ratio=0.0,
            C=C,
            solver="lbfgs",
            class_weight=class_weight,
            random_state=random_state,
            max_iter=max_iter,
        )
    return LogisticRegression(
        penalty="l2",
        C=C,
        solver="lbfgs",
        class_weight=class_weight,
        random_state=random_state,
        max_iter=max_iter,
    )


class FoldBalancedXGBClassifier(ClassifierMixin, BaseEstimator):
    """XGBClassifier wrapper that computes scale_pos_weight strictly on each training fold."""

    def __init__(
        self,
        n_estimators: int = 250,
        max_depth: int = 3,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_lambda: float = 1.0,
        min_child_weight: int = 3,
        balance_positive_weight: bool = True,
        random_state: int = 42,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_lambda = reg_lambda
        self.min_child_weight = min_child_weight
        self.balance_positive_weight = balance_positive_weight
        self.random_state = random_state

    def fit(self, X: np.ndarray | pd.DataFrame, y: np.ndarray | pd.Series) -> FoldBalancedXGBClassifier:
        y_arr = np.asarray(y, dtype=int)
        self.classes_ = np.unique(y_arr)
        n_pos = int((y_arr == 1).sum())
        n_neg = int((y_arr == 0).sum())
        if self.balance_positive_weight and n_pos > 0:
            spw = float(n_neg / n_pos)
        else:
            spw = 1.0
        self.computed_scale_pos_weight_ = spw

        self.model_ = XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            reg_lambda=self.reg_lambda,
            min_child_weight=self.min_child_weight,
            scale_pos_weight=spw,
            eval_metric="logloss",
            random_state=self.random_state,
            n_jobs=1,
        )
        self.model_.fit(X, y_arr)
        return self

    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        return self.model_.predict(X)

    def predict_proba(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        return self.model_.predict_proba(X)


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Create leak-free ColumnTransformer fitted strictly on training folds."""
    num_cols = [c for c in X.columns if c in NUMERIC_FEATURES]
    cat_cols = [c for c in X.columns if c in CATEGORICAL_FEATURES]

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    drop="if_binary",
                    sparse_output=False,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, num_cols),
            ("cat", categorical_pipe, cat_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_model_specs(
    X_train: pd.DataFrame,
    random_seed: int = 42,
) -> dict[str, dict[str, Any]]:
    """Define the four candidate model pipelines and training-only hyperparameter grids."""
    specs: dict[str, dict[str, Any]] = {}

    # 1. Baseline Logistic Regression (untuned L2 benchmark with fold-balanced class weights)
    specs["Baseline Logistic Regression"] = {
        "pipeline": ImbPipeline(
            steps=[
                ("preprocessor", build_preprocessor(X_train)),
                ("sampler", "passthrough"),
                (
                    "classifier",
                    make_logistic_regression(
                        l1_regularized=False,
                        C=1.0,
                        class_weight="balanced",
                        random_state=random_seed,
                    ),
                ),
            ]
        ),
        "param_grid": None,
    }

    # 2. L1-Regularized Logistic Regression (sparse binomial log-likelihood, NOT ordinary Lasso)
    specs["L1 Logistic Regression"] = {
        "pipeline": ImbPipeline(
            steps=[
                ("preprocessor", build_preprocessor(X_train)),
                ("sampler", "passthrough"),
                (
                    "classifier",
                    make_logistic_regression(
                        l1_regularized=True,
                        C=1.0,
                        class_weight="balanced",
                        random_state=random_seed,
                    ),
                ),
            ]
        ),
        "param_grid": [
            {
                "sampler": ["passthrough"],
                "classifier__class_weight": ["balanced"],
                "classifier__C": [0.05, 0.1, 0.25, 0.5, 1.0, 2.0],
            },
            {
                "sampler": [SMOTE(random_state=random_seed, k_neighbors=5)],
                "classifier__class_weight": [None],
                "classifier__C": [0.05, 0.1, 0.25, 0.5, 1.0, 2.0],
            },
        ],
    }

    # 3. Random Forest
    specs["Random Forest"] = {
        "pipeline": ImbPipeline(
            steps=[
                ("preprocessor", build_preprocessor(X_train)),
                ("sampler", "passthrough"),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=300,
                        random_state=random_seed,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "param_grid": [
            {
                "sampler": ["passthrough"],
                "classifier__class_weight": ["balanced_subsample"],
                "classifier__max_depth": [5, 8, None],
                "classifier__min_samples_leaf": [3, 6],
                "classifier__max_features": ["sqrt", 0.4],
            },
            {
                "sampler": [SMOTE(random_state=random_seed, k_neighbors=5)],
                "classifier__class_weight": [None],
                "classifier__max_depth": [5, 8],
                "classifier__min_samples_leaf": [3, 6],
                "classifier__max_features": ["sqrt"],
            },
        ],
    }

    # 4. XGBoost
    specs["XGBoost"] = {
        "pipeline": ImbPipeline(
            steps=[
                ("preprocessor", build_preprocessor(X_train)),
                ("sampler", "passthrough"),
                ("classifier", FoldBalancedXGBClassifier(random_state=random_seed)),
            ]
        ),
        "param_grid": [
            {
                "sampler": ["passthrough"],
                "classifier__balance_positive_weight": [True],
                "classifier__n_estimators": [150, 250],
                "classifier__max_depth": [2, 3, 4],
                "classifier__learning_rate": [0.03, 0.06],
                "classifier__reg_lambda": [1.0, 3.0],
            },
            {
                "sampler": [SMOTE(random_state=random_seed, k_neighbors=5)],
                "classifier__balance_positive_weight": [False],
                "classifier__n_estimators": [150, 250],
                "classifier__max_depth": [2, 3],
                "classifier__learning_rate": [0.03, 0.06],
                "classifier__reg_lambda": [1.0],
            },
        ],
    }

    return specs


def evaluate_threshold_grid(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: np.ndarray | None = None,
) -> pd.DataFrame:
    """Sweep candidate probability thresholds on out-of-fold validation predictions."""
    if thresholds is None:
        thresholds = np.round(np.arange(0.10, 0.90, 0.01), 2)

    rows: list[dict[str, float]] = []
    n_total = len(y_true)
    for thr in thresholds:
        y_pred = (y_prob >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        prec = float(precision_score(y_true, y_pred, zero_division=0))
        rec = float(recall_score(y_true, y_pred, zero_division=0))
        f1 = float(f1_score(y_true, y_pred, zero_division=0))
        # F2 weights recall twice as much as precision for comparison
        denom_f2 = 4.0 * prec + rec
        f2 = float(5.0 * prec * rec / denom_f2) if denom_f2 > 0 else 0.0
        fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
        flagged_pct = 100.0 * float((tp + fp) / n_total)
        rows.append(
            {
                "threshold": float(thr),
                "precision": prec,
                "recall": rec,
                "f1": f1,
                "f2": f2,
                "fpr": fpr,
                "specificity": 1.0 - fpr,
                "flagged_pct": flagged_pct,
                "tp": int(tp),
                "fp": int(fp),
                "tn": int(tn),
                "fn": int(fn),
            }
        )
    return pd.DataFrame(rows)


def select_validation_threshold(
    y_val_true: np.ndarray,
    y_val_prob: np.ndarray,
    criterion: str = "f1",
) -> tuple[float, pd.DataFrame]:
    """Select optimal decision threshold strictly from validation results."""
    sweep_df = evaluate_threshold_grid(y_val_true, y_val_prob)
    if criterion not in {"f1", "f2"}:
        raise ValueError(f"Unsupported threshold criterion '{criterion}'.")
    best_idx = int(sweep_df[criterion].idxmax())
    best_threshold = float(sweep_df.loc[best_idx, "threshold"])
    return best_threshold, sweep_df


def compute_classification_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> dict[str, float | int]:
    """Compute threshold-independent and threshold-specific classification metrics."""
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    return {
        "threshold": round(float(threshold), 4),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "brier_score": float(brier_score_loss(y_true, y_prob)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "specificity": float(1.0 - fpr),
        "fpr": fpr,
        "flagged_rate": float((tp + fp) / len(y_true)),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


def extract_l1_coefficients(l1_pipeline: ImbPipeline) -> pd.DataFrame:
    """Extract standardized log-odds coefficients and odds ratios from L1 Logistic Regression."""
    pre = l1_pipeline.named_steps["preprocessor"]
    clf = l1_pipeline.named_steps["classifier"]
    feature_names = list(pre.get_feature_names_out())
    coefs = clf.coef_.ravel()
    coef_df = pd.DataFrame(
        {
            "encoded_feature": feature_names,
            "coefficient_log_odds": coefs,
            "abs_coefficient": np.abs(coefs),
            "odds_ratio": np.exp(coefs),
            "non_zero": np.abs(coefs) > 1e-6,
        }
    ).sort_values("abs_coefficient", ascending=False).reset_index(drop=True)
    return coef_df


def train_and_evaluate_models(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    test_size: float = 0.20,
    cv_folds: int = 5,
    random_seed: int = 42,
    threshold_criterion: str = "f1",
    permutation_repeats: int = 15,
    metrics_dir: Path | None = None,
    models_dir: Path | None = None,
) -> dict[str, Any]:
    """Execute stratified split, training-only tuning, validation thresholding, and test evaluation."""
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_seed,
    )

    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_seed)
    specs = build_model_specs(X_train, random_seed=random_seed)

    fitted_models: dict[str, ImbPipeline] = {}
    oof_probabilities: dict[str, np.ndarray] = {}
    test_probabilities: dict[str, np.ndarray] = {}
    threshold_sweeps: dict[str, pd.DataFrame] = {}
    comparison_rows: list[dict[str, Any]] = []
    best_params_by_model: dict[str, Any] = {}

    for model_name, spec in specs.items():
        base_pipe = spec["pipeline"]
        param_grid = spec["param_grid"]

        if param_grid is not None:
            search = GridSearchCV(
                estimator=base_pipe,
                param_grid=param_grid,
                scoring="average_precision",
                cv=skf,
                n_jobs=-1,
                refit=True,
            )
            search.fit(X_train, y_train)
            best_pipe: ImbPipeline = search.best_estimator_
            best_params_repr = {
                k: (type(v).__name__ if not isinstance(v, (str, int, float, bool, type(None))) else v)
                for k, v in search.best_params_.items()
            }
        else:
            best_pipe = base_pipe
            best_pipe.fit(X_train, y_train)
            best_params_repr = {"tuning": "none (untuned L2 baseline)"}

        # Compute out-of-fold CV metrics & probabilities strictly on training data
        cv_scores = cross_validate(
            best_pipe,
            X_train,
            y_train,
            cv=skf,
            scoring={"pr_auc": "average_precision", "roc_auc": "roc_auc"},
            n_jobs=-1,
        )
        oof_prob = cross_val_predict(
            best_pipe,
            X_train,
            y_train,
            cv=skf,
            method="predict_proba",
            n_jobs=-1,
        )[:, 1]

        val_threshold, sweep_df = select_validation_threshold(
            y_train.to_numpy(), oof_prob, criterion=threshold_criterion
        )
        oof_metrics_opt = compute_classification_metrics(
            y_train.to_numpy(), oof_prob, threshold=val_threshold
        )

        # Evaluate ONCE on held-out test set
        test_prob = best_pipe.predict_proba(X_test)[:, 1]
        test_metrics_default = compute_classification_metrics(
            y_test.to_numpy(), test_prob, threshold=0.50
        )
        test_metrics_opt = compute_classification_metrics(
            y_test.to_numpy(), test_prob, threshold=val_threshold
        )

        fitted_models[model_name] = best_pipe
        oof_probabilities[model_name] = oof_prob
        test_probabilities[model_name] = test_prob
        threshold_sweeps[model_name] = sweep_df
        best_params_by_model[model_name] = best_params_repr

        comparison_rows.append(
            {
                "model": model_name,
                "cv_pr_auc_mean": float(np.mean(cv_scores["test_pr_auc"])),
                "cv_pr_auc_std": float(np.std(cv_scores["test_pr_auc"])),
                "cv_roc_auc_mean": float(np.mean(cv_scores["test_roc_auc"])),
                "cv_roc_auc_std": float(np.std(cv_scores["test_roc_auc"])),
                "oof_f1_at_val_threshold": float(oof_metrics_opt["f1"]),
                "selected_val_threshold": val_threshold,
                "test_pr_auc": float(test_metrics_opt["pr_auc"]),
                "test_roc_auc": float(test_metrics_opt["roc_auc"]),
                "test_precision_default_050": float(test_metrics_default["precision"]),
                "test_recall_default_050": float(test_metrics_default["recall"]),
                "test_f1_default_050": float(test_metrics_default["f1"]),
                "test_precision_val_thr": float(test_metrics_opt["precision"]),
                "test_recall_val_thr": float(test_metrics_opt["recall"]),
                "test_f1_val_thr": float(test_metrics_opt["f1"]),
                "test_tp": int(test_metrics_opt["tp"]),
                "test_fp": int(test_metrics_opt["fp"]),
                "test_tn": int(test_metrics_opt["tn"]),
                "test_fn": int(test_metrics_opt["fn"]),
            }
        )

    comparison_df = pd.DataFrame(comparison_rows).sort_values(
        "cv_pr_auc_mean", ascending=False
    ).reset_index(drop=True)

    # Select primary champion model strictly by training CV PR-AUC (never test score!)
    primary_model_name = str(comparison_df.loc[0, "model"])
    primary_model = fitted_models[primary_model_name]
    primary_threshold = float(comparison_df.loc[0, "selected_val_threshold"])

    # Permutation feature importance on held-out test set (measuring PR-AUC drop per input column)
    perm = permutation_importance(
        primary_model,
        X_test,
        y_test,
        scoring="average_precision",
        n_repeats=permutation_repeats,
        random_state=random_seed,
        n_jobs=-1,
    )
    perm_df = (
        pd.DataFrame(
            {
                "feature": list(X_test.columns),
                "pr_auc_drop_mean": perm.importances_mean,
                "pr_auc_drop_std": perm.importances_std,
            }
        )
        .sort_values("pr_auc_drop_mean", ascending=False)
        .reset_index(drop=True)
    )

    l1_coef_df = extract_l1_coefficients(fitted_models["L1 Logistic Regression"])

    details_payload: dict[str, Any] = {
        "split": {
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            "train_positive_count": int(y_train.sum()),
            "train_positive_rate": float(y_train.mean()),
            "test_positive_count": int(y_test.sum()),
            "test_positive_rate": float(y_test.mean()),
            "feature_count": int(X.shape[1]),
        },
        "primary_model_selected_by_cv": primary_model_name,
        "primary_validation_threshold": primary_threshold,
        "best_hyperparameters": best_params_by_model,
        "l1_sparsity_summary": {
            "total_encoded_features": int(len(l1_coef_df)),
            "non_zero_features": int(l1_coef_df["non_zero"].sum()),
            "zeroed_out_features": int((~l1_coef_df["non_zero"]).sum()),
        },
        "models": comparison_df.to_dict(orient="records"),
    }

    if metrics_dir is not None:
        metrics_dir.mkdir(parents=True, exist_ok=True)
        comparison_df.to_csv(metrics_dir / "model_comparison.csv", index=False)
        threshold_sweeps[primary_model_name].to_csv(
            metrics_dir / "threshold_sweep_validation.csv", index=False
        )
        perm_df.to_csv(metrics_dir / "feature_importance.csv", index=False)
        l1_coef_df.to_csv(metrics_dir / "l1_logistic_coefficients.csv", index=False)
        (metrics_dir / "model_evaluation_details.json").write_text(
            json.dumps(details_payload, indent=2), encoding="utf-8"
        )

    if models_dir is not None:
        models_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model_name": primary_model_name,
                "pipeline": primary_model,
                "threshold": primary_threshold,
                "features": list(X.columns),
            },
            models_dir / "champion_model.joblib",
        )

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "fitted_models": fitted_models,
        "oof_probabilities": oof_probabilities,
        "test_probabilities": test_probabilities,
        "threshold_sweeps": threshold_sweeps,
        "comparison_df": comparison_df,
        "primary_model_name": primary_model_name,
        "primary_threshold": primary_threshold,
        "permutation_importance_df": perm_df,
        "l1_coefficients_df": l1_coef_df,
        "details": details_payload,
    }


def generate_modeling_figures(
    modeling_results: dict[str, Any],
    figures_dir: Path,
) -> list[Path]:
    """Generate PR/ROC comparison curves, threshold trade-off chart, confusion matrices, and feature importances."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook", font_scale=1.0)
    saved_paths: list[Path] = []

    y_test = modeling_results["y_test"].to_numpy()
    test_probs: dict[str, np.ndarray] = modeling_results["test_probabilities"]
    comp_df: pd.DataFrame = modeling_results["comparison_df"]
    primary_name: str = modeling_results["primary_model_name"]
    primary_thr: float = modeling_results["primary_threshold"]
    sweep_df: pd.DataFrame = modeling_results["threshold_sweeps"][primary_name]
    perm_df: pd.DataFrame = modeling_results["permutation_importance_df"]
    l1_coef_df: pd.DataFrame = modeling_results["l1_coefficients_df"]

    model_palette = {
        "Baseline Logistic Regression": "#6c757d",
        "L1 Logistic Regression": "#1f77b4",
        "Random Forest": "#2ca02c",
        "XGBoost": "#d95f02",
    }

    # -------------------------------------------------------------------------
    # Figure 6: Held-Out Test PR Curves and ROC Curves
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.5))
    prevalence = float(y_test.mean())

    for m_name, probs in test_probs.items():
        prec, rec, _ = precision_recall_curve(y_test, probs)
        ap = average_precision_score(y_test, probs)
        axes[0].plot(
            rec,
            prec,
            linewidth=2.2,
            color=model_palette.get(m_name, "#333333"),
            label=f"{m_name} (PR-AUC = {ap:.3f})",
        )

        fpr, tpr, _ = roc_curve(y_test, probs)
        roc_val = roc_auc_score(y_test, probs)
        axes[1].plot(
            fpr,
            tpr,
            linewidth=2.2,
            color=model_palette.get(m_name, "#333333"),
            label=f"{m_name} (ROC-AUC = {roc_val:.3f})",
        )

    axes[0].axhline(
        prevalence,
        linestyle="--",
        color="#444444",
        linewidth=1.3,
        label=f"No-Skill Baseline ({prevalence:.3f})",
    )
    axes[0].set_xlabel("Recall (Sensitivity)")
    axes[0].set_ylabel("Precision (Positive Predictive Value)")
    axes[0].set_title("A. Precision-Recall Curves (Held-Out Test Set, n = 294)", fontweight="bold")
    axes[0].legend(loc="upper right", fontsize=9)
    axes[0].set_xlim(0.0, 1.02)
    axes[0].set_ylim(0.0, 1.05)

    axes[1].plot([0, 1], [0, 1], linestyle="--", color="#444444", linewidth=1.3, label="Random Guess (0.500)")
    axes[1].set_xlabel("False Positive Rate (1 - Specificity)")
    axes[1].set_ylabel("True Positive Rate (Recall)")
    axes[1].set_title("B. ROC Curves (Held-Out Test Set, n = 294)", fontweight="bold")
    axes[1].legend(loc="lower right", fontsize=9)

    fig.suptitle("Held-Out Test Discrimination Across Four Candidate Classifiers", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    p6 = figures_dir / "fig06_model_comparison_curves.png"
    fig.savefig(p6, dpi=220, bbox_inches="tight")
    plt.close(fig)
    saved_paths.append(p6)

    # -------------------------------------------------------------------------
    # Figure 7: Validation Decision Threshold Selection & Trade-Offs
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))
    axes[0].plot(sweep_df["threshold"], sweep_df["precision"], label="Precision", linewidth=2.1, color="#1f77b4")
    axes[0].plot(sweep_df["threshold"], sweep_df["recall"], label="Recall", linewidth=2.1, color="#d95f02")
    axes[0].plot(sweep_df["threshold"], sweep_df["f1"], label="F1 Score", linewidth=2.5, color="#2ca02c")
    axes[0].plot(
        sweep_df["threshold"],
        sweep_df["flagged_pct"] / 100.0,
        label="Flagged Workforce Share",
        linestyle=":",
        linewidth=2.0,
        color="#555555",
    )
    axes[0].axvline(
        primary_thr,
        linestyle="--",
        color="#b22222",
        linewidth=1.8,
        label=f"Selected Val Threshold (t* = {primary_thr:.2f})",
    )
    axes[0].set_xlabel("Predicted Probability Threshold (t)")
    axes[0].set_ylabel("Metric Value / Proportion")
    axes[0].set_title(f"A. Out-of-Fold Validation Threshold Sweep ({primary_name})", fontweight="bold")
    axes[0].legend(loc="best", fontsize=8.8)

    axes[1].plot(sweep_df["threshold"], sweep_df["fp"], label="False Positives (Unneeded Outreach)", linewidth=2.2, color="#1f77b4")
    axes[1].plot(sweep_df["threshold"], sweep_df["fn"], label="False Negatives (Missed Leavers)", linewidth=2.2, color="#d95f02")
    axes[1].axvline(
        primary_thr,
        linestyle="--",
        color="#b22222",
        linewidth=1.8,
        label=f"t* = {primary_thr:.2f}",
    )
    axes[1].set_xlabel("Predicted Probability Threshold (t)")
    axes[1].set_ylabel("Out-of-Fold Validation Employee Count (n = 1,176)")
    axes[1].set_title("B. Operational Error Trade-Off on Training CV Folds", fontweight="bold")
    axes[1].legend(loc="best", fontsize=9)

    fig.tight_layout()
    p7 = figures_dir / "fig07_threshold_selection.png"
    fig.savefig(p7, dpi=220, bbox_inches="tight")
    plt.close(fig)
    saved_paths.append(p7)

    # -------------------------------------------------------------------------
    # Figure 8: Held-Out Test Confusion Matrices Across All 4 Models
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 9.2))
    model_order = [
        "Baseline Logistic Regression",
        "L1 Logistic Regression",
        "Random Forest",
        "XGBoost",
    ]
    thr_lookup = dict(zip(comp_df["model"], comp_df["selected_val_threshold"]))

    for ax, m_name in zip(axes.ravel(), model_order):
        thr = float(thr_lookup[m_name])
        preds = (test_probs[m_name] >= thr).astype(int)
        cm = confusion_matrix(y_test, preds, labels=[0, 1])
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            xticklabels=["Pred: Stay (0)", "Pred: Leave (1)"],
            yticklabels=["Actual: Stay (0)", "Actual: Leave (1)"],
            annot_kws={"size": 12, "weight": "bold"},
            ax=ax,
        )
        prec_m = precision_score(y_test, preds, zero_division=0)
        rec_m = recall_score(y_test, preds, zero_division=0)
        f1_m = f1_score(y_test, preds, zero_division=0)
        tag = " [Primary]" if m_name == primary_name else ""
        ax.set_title(
            f"{m_name}{tag}\nVal t*={thr:.2f} | P={prec_m:.2f}, R={rec_m:.2f}, F1={f1_m:.2f}",
            fontweight="bold",
            fontsize=10.5,
        )

    fig.suptitle("Held-Out Test Confusion Matrices at Validation-Selected Thresholds (n = 294)", fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    p8 = figures_dir / "fig08_confusion_matrices.png"
    fig.savefig(p8, dpi=220, bbox_inches="tight")
    plt.close(fig)
    saved_paths.append(p8)

    # -------------------------------------------------------------------------
    # Figure 9: Permutation Importance & Directional L1 Coefficients
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 6.2))

    top_perm = perm_df.head(12).iloc[::-1].reset_index(drop=True)
    y_p = np.arange(len(top_perm))
    axes[0].barh(
        y_p,
        top_perm["pr_auc_drop_mean"],
        xerr=top_perm["pr_auc_drop_std"],
        color="#2b5c8f",
        height=0.62,
        capsize=3,
        edgecolor="black",
        linewidth=0.5,
    )
    axes[0].axvline(0.0, color="#333333", linewidth=1.0)
    axes[0].set_yticks(y_p)
    axes[0].set_yticklabels(top_perm["feature"])
    axes[0].set_xlabel("Mean Decrease in Held-Out PR-AUC (±1 SD)")
    axes[0].set_title(f"A. Permutation Importance ({primary_name})", fontweight="bold")

    top_l1 = l1_coef_df[l1_coef_df["non_zero"]].head(12).iloc[::-1].reset_index(drop=True)
    y_l = np.arange(len(top_l1))
    l1_colors = ["#d95f02" if c > 0 else "#2b5c8f" for c in top_l1["coefficient_log_odds"]]
    axes[1].barh(
        y_l,
        top_l1["coefficient_log_odds"],
        color=l1_colors,
        height=0.62,
        edgecolor="black",
        linewidth=0.5,
    )
    axes[1].axvline(0.0, color="#333333", linewidth=1.0)
    axes[1].set_yticks(y_l)
    axes[1].set_yticklabels(top_l1["encoded_feature"])
    axes[1].set_xlabel("Standardized Log-Odds Coefficient (+ Higher Risk / - Lower Risk)")
    axes[1].set_title("B. Top Non-Zero L1 Logistic Regression Coefficients", fontweight="bold")

    fig.text(
        0.5,
        -0.02,
        "Caution: Feature importances capture predictive associations among correlated variables, not causal intervention effects.",
        ha="center",
        fontsize=9.5,
        style="italic",
        color="#444444",
    )
    fig.tight_layout()
    p9 = figures_dir / "fig09_feature_importance.png"
    fig.savefig(p9, dpi=220, bbox_inches="tight")
    plt.close(fig)
    saved_paths.append(p9)

    return saved_paths

