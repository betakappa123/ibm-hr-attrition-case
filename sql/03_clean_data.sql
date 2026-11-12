-- ============================================================================
-- 03_clean_data.sql
-- Produces the validated analytical table `clean_employees` from `raw_employees`:
--   1. Deduplicates on EmployeeNumber (retaining first occurrence if any exist)
--   2. Removes constant zero-variance columns (EmployeeCount, Over18, StandardHours)
--   3. Trims categorical strings and enforces strict SQLite type/domain CHECKs
--   4. Creates binary target `AttritionFlag` (1 = Yes, 0 = No)
--   5. Keeps `EmployeeNumber` strictly as relational PRIMARY KEY (excluded in ML)
-- ============================================================================

DROP TABLE IF EXISTS clean_employees;

CREATE TABLE clean_employees (
    EmployeeNumber           INTEGER PRIMARY KEY,
    Attrition                TEXT    NOT NULL CHECK (Attrition IN ('Yes', 'No')),
    AttritionFlag            INTEGER NOT NULL CHECK (AttritionFlag IN (0, 1)),
    Age                      INTEGER NOT NULL CHECK (Age BETWEEN 18 AND 100),
    BusinessTravel           TEXT    NOT NULL,
    DailyRate                INTEGER NOT NULL CHECK (DailyRate > 0),
    Department               TEXT    NOT NULL,
    DistanceFromHome         INTEGER NOT NULL CHECK (DistanceFromHome >= 0),
    Education                INTEGER NOT NULL CHECK (Education BETWEEN 1 AND 5),
    EducationField           TEXT    NOT NULL,
    EnvironmentSatisfaction  INTEGER NOT NULL CHECK (EnvironmentSatisfaction BETWEEN 1 AND 4),
    Gender                   TEXT    NOT NULL,
    HourlyRate               INTEGER NOT NULL CHECK (HourlyRate > 0),
    JobInvolvement           INTEGER NOT NULL CHECK (JobInvolvement BETWEEN 1 AND 4),
    JobLevel                 INTEGER NOT NULL CHECK (JobLevel BETWEEN 1 AND 5),
    JobRole                  TEXT    NOT NULL,
    JobSatisfaction          INTEGER NOT NULL CHECK (JobSatisfaction BETWEEN 1 AND 4),
    MaritalStatus            TEXT    NOT NULL,
    MonthlyIncome            INTEGER NOT NULL CHECK (MonthlyIncome > 0),
    MonthlyRate              INTEGER NOT NULL CHECK (MonthlyRate > 0),
    NumCompaniesWorked       INTEGER NOT NULL CHECK (NumCompaniesWorked >= 0),
    OverTime                 TEXT    NOT NULL CHECK (OverTime IN ('Yes', 'No')),
    PercentSalaryHike        INTEGER NOT NULL CHECK (PercentSalaryHike >= 0),
    PerformanceRating        INTEGER NOT NULL CHECK (PerformanceRating BETWEEN 1 AND 4),
    RelationshipSatisfaction INTEGER NOT NULL CHECK (RelationshipSatisfaction BETWEEN 1 AND 4),
    StockOptionLevel         INTEGER NOT NULL CHECK (StockOptionLevel BETWEEN 0 AND 3),
    TotalWorkingYears        INTEGER NOT NULL CHECK (TotalWorkingYears >= 0),
    TrainingTimesLastYear    INTEGER NOT NULL CHECK (TrainingTimesLastYear >= 0),
    WorkLifeBalance          INTEGER NOT NULL CHECK (WorkLifeBalance BETWEEN 1 AND 4),
    YearsAtCompany           INTEGER NOT NULL CHECK (YearsAtCompany >= 0),
    YearsInCurrentRole       INTEGER NOT NULL CHECK (YearsInCurrentRole >= 0),
    YearsSinceLastPromotion  INTEGER NOT NULL CHECK (YearsSinceLastPromotion >= 0),
    YearsWithCurrManager     INTEGER NOT NULL CHECK (YearsWithCurrManager >= 0)
);

WITH deduplicated AS (
    SELECT
        *,
        ROW_NUMBER() OVER (PARTITION BY EmployeeNumber ORDER BY rowid) AS rn
    FROM raw_employees
    WHERE EmployeeNumber IS NOT NULL
)
INSERT INTO clean_employees (
    EmployeeNumber,
    Attrition,
    AttritionFlag,
    Age,
    BusinessTravel,
    DailyRate,
    Department,
    DistanceFromHome,
    Education,
    EducationField,
    EnvironmentSatisfaction,
    Gender,
    HourlyRate,
    JobInvolvement,
    JobLevel,
    JobRole,
    JobSatisfaction,
    MaritalStatus,
    MonthlyIncome,
    MonthlyRate,
    NumCompaniesWorked,
    OverTime,
    PercentSalaryHike,
    PerformanceRating,
    RelationshipSatisfaction,
    StockOptionLevel,
    TotalWorkingYears,
    TrainingTimesLastYear,
    WorkLifeBalance,
    YearsAtCompany,
    YearsInCurrentRole,
    YearsSinceLastPromotion,
    YearsWithCurrManager
)
SELECT
    CAST(EmployeeNumber AS INTEGER),
    TRIM(Attrition),
    CASE WHEN TRIM(Attrition) = 'Yes' THEN 1 ELSE 0 END AS AttritionFlag,
    CAST(Age AS INTEGER),
    TRIM(BusinessTravel),
    CAST(DailyRate AS INTEGER),
    TRIM(Department),
    CAST(DistanceFromHome AS INTEGER),
    CAST(Education AS INTEGER),
    TRIM(EducationField),
    CAST(EnvironmentSatisfaction AS INTEGER),
    TRIM(Gender),
    CAST(HourlyRate AS INTEGER),
    CAST(JobInvolvement AS INTEGER),
    CAST(JobLevel AS INTEGER),
    TRIM(JobRole),
    CAST(JobSatisfaction AS INTEGER),
    TRIM(MaritalStatus),
    CAST(MonthlyIncome AS INTEGER),
    CAST(MonthlyRate AS INTEGER),
    CAST(NumCompaniesWorked AS INTEGER),
    TRIM(OverTime),
    CAST(PercentSalaryHike AS INTEGER),
    CAST(PerformanceRating AS INTEGER),
    CAST(RelationshipSatisfaction AS INTEGER),
    CAST(StockOptionLevel AS INTEGER),
    CAST(TotalWorkingYears AS INTEGER),
    CAST(TrainingTimesLastYear AS INTEGER),
    CAST(WorkLifeBalance AS INTEGER),
    CAST(YearsAtCompany AS INTEGER),
    CAST(YearsInCurrentRole AS INTEGER),
    CAST(YearsSinceLastPromotion AS INTEGER),
    CAST(YearsWithCurrManager AS INTEGER)
FROM deduplicated
WHERE rn = 1;

