"""Centralized configuration, relative paths, and schema governance constants."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Relative to repository root (two levels above src/workforce_retention/)
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH: Path = PROJECT_ROOT / "configs" / "default_config.json"

RAW_EXPECTED_COLUMNS: tuple[str, ...] = (
    "Age",
    "Attrition",
    "BusinessTravel",
    "DailyRate",
    "Department",
    "DistanceFromHome",
    "Education",
    "EducationField",
    "EmployeeCount",
    "EmployeeNumber",
    "EnvironmentSatisfaction",
    "Gender",
    "HourlyRate",
    "JobInvolvement",
    "JobLevel",
    "JobRole",
    "JobSatisfaction",
    "MaritalStatus",
    "MonthlyIncome",
    "MonthlyRate",
    "NumCompaniesWorked",
    "Over18",
    "OverTime",
    "PercentSalaryHike",
    "PerformanceRating",
    "RelationshipSatisfaction",
    "StandardHours",
    "StockOptionLevel",
    "TotalWorkingYears",
    "TrainingTimesLastYear",
    "WorkLifeBalance",
    "YearsAtCompany",
    "YearsInCurrentRole",
    "YearsSinceLastPromotion",
    "YearsWithCurrManager",
)

IDENTIFIER_COLUMNS: tuple[str, ...] = ("EmployeeNumber",)
CONSTANT_COLUMNS: tuple[str, ...] = ("EmployeeCount", "Over18", "StandardHours")
TARGET_RAW_COLUMN: str = "Attrition"
TARGET_BINARY_COLUMN: str = "AttritionFlag"
SENSITIVE_DEMOGRAPHIC_COLUMNS: tuple[str, ...] = ("Age", "Gender", "MaritalStatus")

FORBIDDEN_MODEL_COLUMNS: frozenset[str] = frozenset(
    {
        *IDENTIFIER_COLUMNS,
        *CONSTANT_COLUMNS,
        TARGET_RAW_COLUMN,
        TARGET_BINARY_COLUMN,
    }
)

NUMERIC_FEATURES: tuple[str, ...] = (
    "Age",
    "DailyRate",
    "DistanceFromHome",
    "Education",
    "EnvironmentSatisfaction",
    "HourlyRate",
    "JobInvolvement",
    "JobLevel",
    "JobSatisfaction",
    "MonthlyIncome",
    "MonthlyRate",
    "NumCompaniesWorked",
    "PercentSalaryHike",
    "PerformanceRating",
    "RelationshipSatisfaction",
    "StockOptionLevel",
    "TotalWorkingYears",
    "TrainingTimesLastYear",
    "WorkLifeBalance",
    "YearsAtCompany",
    "YearsInCurrentRole",
    "YearsSinceLastPromotion",
    "YearsWithCurrManager",
)

CATEGORICAL_FEATURES: tuple[str, ...] = (
    "BusinessTravel",
    "Department",
    "EducationField",
    "Gender",
    "JobRole",
    "MaritalStatus",
    "OverTime",
)


@dataclass(frozen=True)
class ProjectConfig:
    """Typed project configuration loaded from relative JSON config."""

    random_seed: int
    raw_csv_path: Path
    sqlite_db_path: Path
    clean_csv_path: Path
    sql_dir: Path
    figures_dir: Path
    metrics_dir: Path
    models_dir: Path
    test_size: float
    cv_folds: int
    scoring_metric: str
    threshold_selection_metric: str
    permutation_repeats: int
    default_workforce_size: int
    default_baseline_attrition: float
    default_intervention_effectiveness: float
    default_replacement_cost: float
    default_intervention_cost: float

    @classmethod
    def load(
        cls,
        config_path: Path | None = None,
        root_dir: Path | None = None,
    ) -> ProjectConfig:
        base_root = root_dir if root_dir is not None else PROJECT_ROOT
        cfg_file = config_path if config_path is not None else DEFAULT_CONFIG_PATH
        raw_cfg: dict[str, Any] = json.loads(cfg_file.read_text(encoding="utf-8"))

        paths = raw_cfg["paths"]
        modeling = raw_cfg["modeling"]
        scenario = raw_cfg["business_scenario_defaults"]

        return cls(
            random_seed=int(raw_cfg["random_seed"]),
            raw_csv_path=base_root / paths["raw_csv"],
            sqlite_db_path=base_root / paths["sqlite_db"],
            clean_csv_path=base_root / paths["clean_csv"],
            sql_dir=base_root / paths["sql_dir"],
            figures_dir=base_root / paths["figures_dir"],
            metrics_dir=base_root / paths["metrics_dir"],
            models_dir=base_root / paths["models_dir"],
            test_size=float(modeling["test_size"]),
            cv_folds=int(modeling["cv_folds"]),
            scoring_metric=str(modeling["scoring_metric"]),
            threshold_selection_metric=str(modeling["threshold_selection_metric"]),
            permutation_repeats=int(modeling["permutation_repeats"]),
            default_workforce_size=int(scenario["workforce_size"]),
            default_baseline_attrition=float(scenario["baseline_attrition_rate"]),
            default_intervention_effectiveness=float(scenario["intervention_effectiveness"]),
            default_replacement_cost=float(scenario["replacement_cost_per_departure"]),
            default_intervention_cost=float(scenario["intervention_cost_per_flagged_employee"]),
        )

