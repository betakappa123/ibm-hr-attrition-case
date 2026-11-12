"""Shared pytest fixtures for schema validation, anti-leakage, and scenario unit tests.

Note: `unit_test_raw_df` is a small deterministic fixture used strictly to test
software validation logic, error handling, and pipeline isolation in unit tests.
It is NEVER used for analytical findings or reported metrics.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from workforce_retention.config import PROJECT_ROOT, ProjectConfig


@pytest.fixture(scope="session")
def project_config() -> ProjectConfig:
    return ProjectConfig.load(root_dir=PROJECT_ROOT)


@pytest.fixture()
def unit_test_raw_df() -> pd.DataFrame:
    """Deterministic 60-row schema-compliant fixture for testing SQL & pipeline code paths."""
    rng = np.random.default_rng(42)
    n = 60
    attrition = np.array(["Yes"] * 15 + ["No"] * 45)
    rng.shuffle(attrition)
    total_yrs = rng.integers(2, 25, size=n)
    company_yrs = np.minimum(total_yrs, rng.integers(0, 15, size=n))

    return pd.DataFrame(
        {
            "Age": rng.integers(22, 58, size=n),
            "Attrition": attrition,
            "BusinessTravel": rng.choice(["Travel_Rarely", "Travel_Frequently", "Non-Travel"], size=n),
            "DailyRate": rng.integers(200, 1400, size=n),
            "Department": rng.choice(["Sales", "Research & Development", "Human Resources"], size=n),
            "DistanceFromHome": rng.integers(1, 28, size=n),
            "Education": rng.integers(1, 6, size=n),
            "EducationField": rng.choice(["Life Sciences", "Medical", "Marketing", "Technical Degree"], size=n),
            "EmployeeCount": np.ones(n, dtype=int),
            "EmployeeNumber": np.arange(1001, 1001 + n, dtype=int),
            "EnvironmentSatisfaction": rng.integers(1, 5, size=n),
            "Gender": rng.choice(["Female", "Male"], size=n),
            "HourlyRate": rng.integers(35, 98, size=n),
            "JobInvolvement": rng.integers(1, 5, size=n),
            "JobLevel": rng.integers(1, 6, size=n),
            "JobRole": rng.choice(
                ["Sales Executive", "Research Scientist", "Laboratory Technician", "Manager"],
                size=n,
            ),
            "JobSatisfaction": rng.integers(1, 5, size=n),
            "MaritalStatus": rng.choice(["Single", "Married", "Divorced"], size=n),
            "MonthlyIncome": rng.integers(2200, 16000, size=n),
            "MonthlyRate": rng.integers(3000, 25000, size=n),
            "NumCompaniesWorked": rng.integers(0, 8, size=n),
            "Over18": ["Y"] * n,
            "OverTime": rng.choice(["Yes", "No"], size=n),
            "PercentSalaryHike": rng.integers(11, 24, size=n),
            "PerformanceRating": rng.choice([3, 4], size=n),
            "RelationshipSatisfaction": rng.integers(1, 5, size=n),
            "StandardHours": np.full(n, 80, dtype=int),
            "StockOptionLevel": rng.integers(0, 4, size=n),
            "TotalWorkingYears": total_yrs,
            "TrainingTimesLastYear": rng.integers(0, 6, size=n),
            "WorkLifeBalance": rng.integers(1, 5, size=n),
            "YearsAtCompany": company_yrs,
            "YearsInCurrentRole": np.minimum(company_yrs, rng.integers(0, 10, size=n)),
            "YearsSinceLastPromotion": np.minimum(company_yrs, rng.integers(0, 7, size=n)),
            "YearsWithCurrManager": np.minimum(company_yrs, rng.integers(0, 10, size=n)),
        }
    )

