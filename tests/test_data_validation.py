"""Tests for SQLite data preparation, schema validation, deduplication, and raw immutability."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from workforce_retention.config import CONSTANT_COLUMNS, ProjectConfig
from workforce_retention.data_prep import (
    DatasetNotFoundError,
    SchemaValidationError,
    compute_sha256,
    ensure_raw_dataset_exists,
    run_sql_data_preparation,
)


def test_missing_dataset_raises_actionable_instructions(tmp_path: Path) -> None:
    missing_path = tmp_path / "WA_Fn-UseC_-HR-Employee-Attrition.csv"
    with pytest.raises(DatasetNotFoundError, match="Do NOT substitute synthetic"):
        ensure_raw_dataset_exists(missing_path)


def test_sql_pipeline_rejects_missing_values(
    tmp_path: Path,
    unit_test_raw_df: pd.DataFrame,
    project_config: ProjectConfig,
) -> None:
    bad_df = unit_test_raw_df.copy()
    bad_df.loc[3, "OverTime"] = None
    raw_csv = tmp_path / "raw.csv"
    bad_df.to_csv(raw_csv, index=False)

    with pytest.raises(SchemaValidationError, match="Missing/blank values"):
        run_sql_data_preparation(
            raw_csv_path=raw_csv,
            sqlite_db_path=tmp_path / "hr.db",
            clean_csv_path=tmp_path / "clean.csv",
            sql_dir=project_config.sql_dir,
        )


def test_sql_pipeline_rejects_domain_violations(
    tmp_path: Path,
    unit_test_raw_df: pd.DataFrame,
    project_config: ProjectConfig,
) -> None:
    bad_df = unit_test_raw_df.copy()
    bad_df.loc[0, "Age"] = 14  # Violates Age >= 18 check
    raw_csv = tmp_path / "raw_invalid_age.csv"
    bad_df.to_csv(raw_csv, index=False)

    with pytest.raises(SchemaValidationError, match="Domain or data type violations"):
        run_sql_data_preparation(
            raw_csv_path=raw_csv,
            sqlite_db_path=tmp_path / "hr.db",
            clean_csv_path=tmp_path / "clean.csv",
            sql_dir=project_config.sql_dir,
        )


def test_sql_pipeline_deduplicates_drops_constants_and_preserves_raw(
    tmp_path: Path,
    unit_test_raw_df: pd.DataFrame,
    project_config: ProjectConfig,
) -> None:
    # Append duplicate EmployeeNumber row
    dup_df = pd.concat([unit_test_raw_df, unit_test_raw_df.iloc[[0]]], ignore_index=True)
    raw_csv = tmp_path / "raw_with_dup.csv"
    dup_df.to_csv(raw_csv, index=False)
    hash_before = compute_sha256(raw_csv)

    out = run_sql_data_preparation(
        raw_csv_path=raw_csv,
        sqlite_db_path=tmp_path / "hr.db",
        clean_csv_path=tmp_path / "clean.csv",
        sql_dir=project_config.sql_dir,
    )
    rep = out["validation_report"]
    clean_df = out["clean_df"]

    assert compute_sha256(raw_csv) == hash_before
    assert rep["duplicate_employee_numbers"] == 1
    assert len(clean_df) == len(unit_test_raw_df)
    assert clean_df["EmployeeNumber"].is_unique
    for const_col in CONSTANT_COLUMNS:
        assert const_col in rep["constant_columns_detected"]
        assert const_col not in clean_df.columns
    assert set(clean_df["AttritionFlag"].unique()).issubset({0, 1})


def test_real_ibm_dataset_invariants_when_present(
    tmp_path: Path,
    project_config: ProjectConfig,
) -> None:
    if not project_config.raw_csv_path.exists():
        pytest.skip("Real IBM HR dataset CSV not placed in data/raw/ yet.")

    out = run_sql_data_preparation(
        raw_csv_path=project_config.raw_csv_path,
        sqlite_db_path=tmp_path / "verify_real.db",
        clean_csv_path=tmp_path / "verify_clean.csv",
        sql_dir=project_config.sql_dir,
    )
    rep = out["validation_report"]
    assert rep["raw_row_count"] == 1470
    assert rep["raw_column_count"] == 35
    assert rep["total_missing_cells"] == 0
    assert rep["duplicate_employee_numbers"] == 0
    assert sorted(rep["constant_columns_detected"]) == ["EmployeeCount", "Over18", "StandardHours"]
    assert rep["clean_row_count"] == 1470
    assert rep["clean_column_count"] == 33

