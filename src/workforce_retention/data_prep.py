"""SQLite data ingestion, validation, cleaning, and feature-matrix preparation."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd

from workforce_retention.config import (
    CATEGORICAL_FEATURES,
    CONSTANT_COLUMNS,
    FORBIDDEN_MODEL_COLUMNS,
    IDENTIFIER_COLUMNS,
    NUMERIC_FEATURES,
    RAW_EXPECTED_COLUMNS,
    SENSITIVE_DEMOGRAPHIC_COLUMNS,
    TARGET_BINARY_COLUMN,
)


class DatasetNotFoundError(FileNotFoundError):
    """Raised when the raw IBM HR dataset CSV is not present in data/raw/."""


class SchemaValidationError(ValueError):
    """Raised when raw input fails schema, domain, or integrity validation."""


def compute_sha256(file_path: Path) -> str:
    """Compute SHA-256 hex digest of a file to verify raw-data immutability."""
    hasher = hashlib.sha256()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def ensure_raw_dataset_exists(raw_csv_path: Path) -> Path:
    """Verify the IBM HR dataset exists; provide setup instructions if absent."""
    if not raw_csv_path.exists():
        raise DatasetNotFoundError(
            f"Raw dataset not found at '{raw_csv_path}'.\n"
            "Do NOT substitute synthetic or unrelated data. To obtain the dataset:\n"
            "  1. Download 'IBM HR Analytics Employee Attrition & Performance' from Kaggle:\n"
            "     https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset\n"
            "  2. Place 'WA_Fn-UseC_-HR-Employee-Attrition.csv' in 'data/raw/'.\n"
            "  3. Re-run: python -m workforce_retention.cli run-all"
        )
    return raw_csv_path


def parse_named_sql_queries(sql_text: str) -> dict[str, str]:
    """Parse '-- @name: <query_name>' blocks from a SQL script."""
    queries: dict[str, str] = {}
    current_name: str | None = None
    buffer: list[str] = []

    for line in sql_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("-- @name:"):
            if current_name is not None:
                stmt = "\n".join(buffer).strip()
                if stmt:
                    queries[current_name] = stmt
            current_name = stripped.split("-- @name:", 1)[1].strip()
            buffer = []
        elif current_name is not None:
            buffer.append(line)

    if current_name is not None:
        stmt = "\n".join(buffer).strip()
        if stmt:
            queries[current_name] = stmt

    return queries


def validate_raw_dataframe_schema(raw_df: pd.DataFrame) -> None:
    """Verify column presence before loading into SQLite staging."""
    expected_set = set(RAW_EXPECTED_COLUMNS)
    actual_set = set(raw_df.columns)
    missing = sorted(expected_set - actual_set)
    unexpected = sorted(actual_set - expected_set)
    if missing or unexpected:
        raise SchemaValidationError(
            f"Raw CSV schema mismatch. Missing columns: {missing}; "
            f"Unexpected columns: {unexpected}."
        )
    if raw_df.empty:
        raise SchemaValidationError("Raw CSV contains 0 rows.")


def run_sql_data_preparation(
    raw_csv_path: Path,
    sqlite_db_path: Path,
    clean_csv_path: Path,
    sql_dir: Path,
    metrics_dir: Path | None = None,
) -> dict[str, Any]:
    """Execute end-to-end SQLite ingestion, validation, cleaning, and cohort queries."""
    ensure_raw_dataset_exists(raw_csv_path)
    raw_hash_before = compute_sha256(raw_csv_path)

    raw_df = pd.read_csv(raw_csv_path)
    validate_raw_dataframe_schema(raw_df)
    # Enforce canonical column order for staging insert
    raw_df = raw_df[list(RAW_EXPECTED_COLUMNS)]

    sqlite_db_path.parent.mkdir(parents=True, exist_ok=True)
    clean_csv_path.parent.mkdir(parents=True, exist_ok=True)

    schema_sql = (sql_dir / "01_schema.sql").read_text(encoding="utf-8")
    validate_sql = (sql_dir / "02_inspect_and_validate.sql").read_text(encoding="utf-8")
    clean_sql = (sql_dir / "03_clean_data.sql").read_text(encoding="utf-8")
    analytical_sql = (sql_dir / "04_analytical_queries.sql").read_text(encoding="utf-8")

    validation_queries = parse_named_sql_queries(validate_sql)
    analytical_queries = parse_named_sql_queries(analytical_sql)

    with sqlite3.connect(sqlite_db_path) as conn:
        # 1. Create staging table and load raw CSV records
        conn.executescript(schema_sql)
        placeholders = ", ".join(["?"] * len(RAW_EXPECTED_COLUMNS))
        insert_sql = (
            f"INSERT INTO raw_employees ({', '.join(RAW_EXPECTED_COLUMNS)}) "
            f"VALUES ({placeholders})"
        )
        # Convert pandas NA/NaN to Python None for accurate SQL NULL auditing
        records = [
            tuple(None if pd.isna(val) else val for val in row)
            for row in raw_df.itertuples(index=False, name=None)
        ]
        conn.executemany(insert_sql, records)
        conn.commit()

        # 2. Execute SQL inspection & validation queries
        row_summary = pd.read_sql_query(
            validation_queries["row_and_id_summary"], conn
        ).iloc[0].to_dict()
        full_dup_summary = pd.read_sql_query(
            validation_queries["full_row_duplicates"], conn
        ).iloc[0].to_dict()
        missing_row = pd.read_sql_query(
            validation_queries["missing_values_audit"], conn
        ).iloc[0].to_dict()
        domain_row = pd.read_sql_query(
            validation_queries["type_and_domain_audit"], conn
        ).iloc[0].to_dict()
        constant_df = pd.read_sql_query(
            validation_queries["constant_columns_audit"], conn
        )
        target_df = pd.read_sql_query(
            validation_queries["target_distribution"], conn
        )

        missing_by_col = {
            k.replace("missing_", ""): int(v)
            for k, v in missing_row.items()
            if v is not None and int(v) > 0
        }
        domain_violations = {
            k: int(v)
            for k, v in domain_row.items()
            if v is not None and int(v) > 0
        }
        detected_constant_cols = constant_df["column_name"].tolist()

        if missing_by_col:
            raise SchemaValidationError(
                f"Missing/blank values detected in raw_employees: {missing_by_col}"
            )
        if domain_violations:
            raise SchemaValidationError(
                f"Domain or data type violations detected in raw_employees: {domain_violations}"
            )

        # 3. Execute SQL cleaning script -> clean_employees
        conn.executescript(clean_sql)
        clean_df = pd.read_sql_query(
            "SELECT * FROM clean_employees ORDER BY EmployeeNumber", conn
        )

        # 4. Run SQL analytical cohort queries over clean_employees
        cohort_summaries: dict[str, list[dict[str, Any]]] = {}
        for q_name, q_sql in analytical_queries.items():
            q_df = pd.read_sql_query(q_sql, conn)
            cohort_summaries[q_name] = q_df.to_dict(orient="records")

    clean_df.to_csv(clean_csv_path, index=False)

    raw_hash_after = compute_sha256(raw_csv_path)
    if raw_hash_before != raw_hash_after:
        raise RuntimeError("Raw dataset file hash changed during data preparation!")

    validation_report: dict[str, Any] = {
        "raw_file_name": raw_csv_path.name,
        "raw_sha256": raw_hash_after,
        "raw_file_unchanged": raw_hash_before == raw_hash_after,
        "raw_row_count": int(row_summary["total_rows"]),
        "raw_column_count": len(RAW_EXPECTED_COLUMNS),
        "distinct_employee_numbers": int(row_summary["distinct_employee_numbers"]),
        "duplicate_employee_numbers": int(row_summary["duplicate_employee_numbers"]),
        "duplicate_feature_groups": int(full_dup_summary["duplicate_feature_groups"]),
        "total_missing_cells": int(sum(int(v) for v in missing_row.values())),
        "missing_columns_nonzero": missing_by_col,
        "domain_violations_nonzero": domain_violations,
        "constant_columns_detected": detected_constant_cols,
        "constant_columns_dropped": list(CONSTANT_COLUMNS),
        "identifier_columns_retained_as_pk_only": list(IDENTIFIER_COLUMNS),
        "clean_row_count": int(len(clean_df)),
        "clean_column_count": int(clean_df.shape[1]),
        "target_distribution": target_df.to_dict(orient="records"),
    }

    if metrics_dir is not None:
        metrics_dir.mkdir(parents=True, exist_ok=True)
        (metrics_dir / "data_validation_report.json").write_text(
            json.dumps(validation_report, indent=2), encoding="utf-8"
        )
        (metrics_dir / "sql_cohort_summaries.json").write_text(
            json.dumps(cohort_summaries, indent=2), encoding="utf-8"
        )

    return {
        "validation_report": validation_report,
        "cohort_summaries": cohort_summaries,
        "clean_df": clean_df,
    }


def load_clean_dataset(clean_csv_path: Path) -> pd.DataFrame:
    """Load the cleaned dataset produced by the SQLite preparation stage."""
    if not clean_csv_path.exists():
        raise FileNotFoundError(
            f"Cleaned dataset not found at '{clean_csv_path}'. "
            "Run 'python -m workforce_retention.cli prepare-data' first."
        )
    return pd.read_csv(clean_csv_path)


def prepare_feature_matrix(
    clean_df: pd.DataFrame,
    drop_sensitive: bool = False,
) -> tuple[pd.DataFrame, pd.Series]:
    """Construct leak-free feature matrix X and binary target vector y.

    Strictly excludes:
      - Employee identifiers (EmployeeNumber)
      - Constant columns (EmployeeCount, Over18, StandardHours)
      - Target labels (Attrition, AttritionFlag)
    Optionally excludes sensitive demographic attributes (Age, Gender, MaritalStatus)
    for responsible-AI ablation comparisons.
    """
    if TARGET_BINARY_COLUMN not in clean_df.columns:
        raise SchemaValidationError(f"Missing binary target '{TARGET_BINARY_COLUMN}'.")

    y = clean_df[TARGET_BINARY_COLUMN].astype(int).copy()
    if not set(y.unique()).issubset({0, 1}):
        raise SchemaValidationError("Binary target must contain only 0 and 1.")

    excluded = set(FORBIDDEN_MODEL_COLUMNS)
    if drop_sensitive:
        excluded.update(SENSITIVE_DEMOGRAPHIC_COLUMNS)

    expected_features = [
        col
        for col in (*NUMERIC_FEATURES, *CATEGORICAL_FEATURES)
        if col not in excluded
    ]
    missing_features = [col for col in expected_features if col not in clean_df.columns]
    if missing_features:
        raise SchemaValidationError(f"Missing expected model features: {missing_features}")

    X = clean_df[expected_features].copy()

    leaked = FORBIDDEN_MODEL_COLUMNS.intersection(X.columns)
    if leaked:
        raise AssertionError(f"Forbidden columns leaked into feature matrix: {leaked}")

    zero_var = [col for col in X.columns if X[col].nunique(dropna=False) <= 1]
    if zero_var:
        raise SchemaValidationError(
            f"Zero-variance constant columns found in feature matrix: {zero_var}"
        )

    return X, y

