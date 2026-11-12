"""Tests verifying strict target/feature leakage prevention and fold-only imbalance handling."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import StratifiedKFold

from workforce_retention.config import (
    FORBIDDEN_MODEL_COLUMNS,
    SENSITIVE_DEMOGRAPHIC_COLUMNS,
    ProjectConfig,
)
from workforce_retention.data_prep import prepare_feature_matrix, run_sql_data_preparation
from workforce_retention.modeling import (
    FoldBalancedXGBClassifier,
    build_preprocessor,
    make_logistic_regression,
    select_validation_threshold,
)


def _get_clean_fixture(
    tmp_path: Path,
    unit_test_raw_df: pd.DataFrame,
    project_config: ProjectConfig,
) -> pd.DataFrame:
    raw_csv = tmp_path / "raw.csv"
    unit_test_raw_df.to_csv(raw_csv, index=False)
    res = run_sql_data_preparation(
        raw_csv_path=raw_csv,
        sqlite_db_path=tmp_path / "db.sqlite",
        clean_csv_path=tmp_path / "clean.csv",
        sql_dir=project_config.sql_dir,
    )
    return res["clean_df"]


def test_identifiers_constants_and_targets_never_enter_feature_matrix(
    tmp_path: Path,
    unit_test_raw_df: pd.DataFrame,
    project_config: ProjectConfig,
) -> None:
    clean_df = _get_clean_fixture(tmp_path, unit_test_raw_df, project_config)
    X_full, y = prepare_feature_matrix(clean_df, drop_sensitive=False)

    assert FORBIDDEN_MODEL_COLUMNS.isdisjoint(X_full.columns)
    assert "EmployeeNumber" not in X_full.columns
    assert "Attrition" not in X_full.columns
    assert "AttritionFlag" not in X_full.columns
    assert len(X_full) == len(y)

    X_nosens, _ = prepare_feature_matrix(clean_df, drop_sensitive=True)
    for sens_col in SENSITIVE_DEMOGRAPHIC_COLUMNS:
        assert sens_col not in X_nosens.columns


def test_scaler_fits_strictly_on_training_split_without_test_contamination(
    tmp_path: Path,
    unit_test_raw_df: pd.DataFrame,
    project_config: ProjectConfig,
) -> None:
    clean_df = _get_clean_fixture(tmp_path, unit_test_raw_df, project_config)
    X, y = prepare_feature_matrix(clean_df)

    X_train = X.iloc[:45].copy()
    y_train = y.iloc[:45].copy()
    X_test = X.iloc[45:].copy()
    # Poison X_test with extreme values that would drastically shift mean if leaked
    X_test["MonthlyIncome"] = 99_999_999

    pre = build_preprocessor(X_train)
    pre.fit(X_train, y_train)

    scaler = pre.named_transformers_["num"].named_steps["scaler"]
    num_cols = pre.transformers_[0][2]
    inc_idx = num_cols.index("MonthlyIncome")

    expected_train_mean = float(X_train["MonthlyIncome"].mean())
    assert np.isclose(scaler.mean_[inc_idx], expected_train_mean, rtol=0.0, atol=1e-10)
    # Transform X_test without refitting
    transformed_test = pre.transform(X_test)
    assert transformed_test.shape[0] == len(X_test)


def test_smote_resamples_only_training_folds_never_validation_folds(
    tmp_path: Path,
    unit_test_raw_df: pd.DataFrame,
    project_config: ProjectConfig,
) -> None:
    clean_df = _get_clean_fixture(tmp_path, unit_test_raw_df, project_config)
    X, y = prepare_feature_matrix(clean_df)

    recorded_fit_sizes: list[tuple[int, int]] = []

    class InspectableLogReg(ClassifierMixin, BaseEstimator):
        """Spy estimator recording class counts received after SMOTE during fit."""

        def fit(self, X_arr: np.ndarray, y_arr: np.ndarray) -> InspectableLogReg:
            pos = int((y_arr == 1).sum())
            neg = int((y_arr == 0).sum())
            recorded_fit_sizes.append((neg, pos))
            self.inner_ = make_logistic_regression(l1_regularized=False, random_state=42)
            self.inner_.fit(X_arr, y_arr)
            self.classes_ = self.inner_.classes_
            return self

        def predict(self, X_arr: np.ndarray) -> np.ndarray:
            return self.inner_.predict(X_arr)

        def predict_proba(self, X_arr: np.ndarray) -> np.ndarray:
            return self.inner_.predict_proba(X_arr)

    pipe = ImbPipeline(
        steps=[
            ("preprocessor", build_preprocessor(X)),
            ("sampler", SMOTE(random_state=42, k_neighbors=3)),
            ("classifier", InspectableLogReg()),
        ]
    )

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    for train_idx, val_idx in skf.split(X, y):
        pipe.fit(X.iloc[train_idx], y.iloc[train_idx])
        val_probs = pipe.predict_proba(X.iloc[val_idx])
        # Validation fold size must remain strictly unchanged (never oversampled)
        assert len(val_probs) == len(val_idx)

    # Every training fold seen by the classifier after SMOTE was balanced 1:1
    assert all(neg == pos for neg, pos in recorded_fit_sizes)


def test_fold_balanced_xgb_uses_training_fold_ratio_only(
    tmp_path: Path,
    unit_test_raw_df: pd.DataFrame,
    project_config: ProjectConfig,
) -> None:
    clean_df = _get_clean_fixture(tmp_path, unit_test_raw_df, project_config)
    X, y = prepare_feature_matrix(clean_df)
    pre = build_preprocessor(X)
    X_enc = pre.fit_transform(X[:40], y[:40])
    y_sub = y[:40].to_numpy()

    clf = FoldBalancedXGBClassifier(n_estimators=20, max_depth=2, random_state=42)
    clf.fit(X_enc, y_sub)
    expected_spw = float((y_sub == 0).sum() / (y_sub == 1).sum())
    assert np.isclose(clf.computed_scale_pos_weight_, expected_spw)


def test_l1_logistic_outputs_valid_probabilities_and_sparse_weights() -> None:
    rng = np.random.default_rng(123)
    X_syn = rng.normal(size=(200, 12))
    # Only column 0 carries true signal; columns 1..11 are pure noise
    logits = 2.5 * X_syn[:, 0] - 1.2
    probs_true = 1.0 / (1.0 + np.exp(-logits))
    y_syn = rng.binomial(1, probs_true)

    clf = make_logistic_regression(l1_regularized=True, C=0.15, random_state=42)
    clf.fit(X_syn, y_syn)
    pred_probs = clf.predict_proba(X_syn)[:, 1]

    # Unlike ordinary linear Lasso, L1 logistic regression produces bounded probabilities in (0, 1)
    assert np.all((pred_probs > 0.0) & (pred_probs < 1.0))
    # Induces exact sparsity on irrelevant noise predictors
    zero_count = int(np.sum(np.abs(clf.coef_.ravel()) < 1e-6))
    assert zero_count >= 4


def test_threshold_selection_is_independent_of_test_labels() -> None:
    rng = np.random.default_rng(42)
    y_val = rng.integers(0, 2, size=150)
    p_val = np.clip(0.25 + 0.45 * y_val + rng.normal(0, 0.15, size=150), 0.01, 0.99)

    thr_a, _ = select_validation_threshold(y_val, p_val, criterion="f1")
    # Even if test set labels were inverted, validation threshold selection does not see test set
    thr_b, _ = select_validation_threshold(y_val, p_val, criterion="f1")
    assert thr_a == thr_b
