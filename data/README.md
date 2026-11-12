# Data Directory & Acquisition Guide

This project uses the **IBM HR Analytics Employee Attrition & Performance** dataset (`WA_Fn-UseC_-HR-Employee-Attrition.csv`). Raw source files and generated processed artifacts are strictly separated and excluded from Git version control via `.gitignore`.

## Directory Layout

```text
data/
├── README.md                                  # Dataset instructions & schema overview
├── raw/
│   └── WA_Fn-UseC_-HR-Employee-Attrition.csv  # Immutable raw source CSV (1,470 rows × 35 columns)
└── processed/
    ├── hr_attrition.db                        # SQLite database (raw_employees + clean_employees)
    └── employees_clean.csv                    # Validated & cleaned analytical dataset (1,470 rows × 32 columns)
```

## How to Obtain the Dataset

1. **Via Kaggle Web UI**:
   - Visit the **IBM HR Analytics Employee Attrition & Performance** page on Kaggle:
     `https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset`
   - Download the archive and extract `WA_Fn-UseC_-HR-Employee-Attrition.csv` into `data/raw/WA_Fn-UseC_-HR-Employee-Attrition.csv`.

2. **Via Kaggle CLI** (requires `~/.kaggle/kaggle.json` credentials):
   ```bash
   kaggle datasets download -d pavansubhasht/ibm-hr-analytics-attrition-dataset -p data/raw --unzip
   ```

## Expected Schema & Integrity Check

When running `python -m workforce_retention.cli prepare-data`, the SQLite pipeline verifies:
- **Row count**: `1,470` employee records
- **Raw column count**: `35` columns
- **Missing values**: `0` NULL or blank cells across all 35 columns
- **Primary key uniqueness**: `1,470` unique `EmployeeNumber` values (`0` duplicates)
- **Zero-variance (constant) columns removed during cleaning**:
  - `EmployeeCount` (constant `1`)
  - `Over18` (constant `'Y'`)
  - `StandardHours` (constant `80`)
- **Identifier governance**: `EmployeeNumber` is retained in `clean_employees` strictly as a relational primary key and is **excluded** from all machine learning feature matrices.

## Dataset Provenance & Limitations

The IBM HR Analytics dataset was created by IBM data scientists as a **fictional / illustrative benchmark dataset** originally showcased with IBM Watson Analytics. It does **not** represent actual confidential HR records of IBM employees, lacks longitudinal timestamps (preventing survival or censoring analysis), and does not separate voluntary resignations from involuntary departures. Do not substitute synthetic or unrelated datasets when reproducing results in this repository.

