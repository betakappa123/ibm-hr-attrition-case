-- ============================================================================
-- 01_schema.sql
-- Staging schema for the raw IBM HR Analytics Employee Attrition dataset.
-- Columns are kept nullable in raw_employees without restrictive constraints
-- so that 02_inspect_and_validate.sql can audit raw data quality directly.
-- ============================================================================

DROP TABLE IF EXISTS raw_employees;

CREATE TABLE raw_employees (
    Age                      INTEGER,
    Attrition                TEXT,
    BusinessTravel           TEXT,
    DailyRate                INTEGER,
    Department               TEXT,
    DistanceFromHome         INTEGER,
    Education                INTEGER,
    EducationField           TEXT,
    EmployeeCount            INTEGER,
    EmployeeNumber           INTEGER,
    EnvironmentSatisfaction  INTEGER,
    Gender                   TEXT,
    HourlyRate               INTEGER,
    JobInvolvement           INTEGER,
    JobLevel                 INTEGER,
    JobRole                  TEXT,
    JobSatisfaction          INTEGER,
    MaritalStatus            TEXT,
    MonthlyIncome            INTEGER,
    MonthlyRate              INTEGER,
    NumCompaniesWorked       INTEGER,
    Over18                   TEXT,
    OverTime                 TEXT,
    PercentSalaryHike        INTEGER,
    PerformanceRating        INTEGER,
    RelationshipSatisfaction INTEGER,
    StandardHours            INTEGER,
    StockOptionLevel         INTEGER,
    TotalWorkingYears        INTEGER,
    TrainingTimesLastYear    INTEGER,
    WorkLifeBalance          INTEGER,
    YearsAtCompany           INTEGER,
    YearsInCurrentRole       INTEGER,
    YearsSinceLastPromotion  INTEGER,
    YearsWithCurrManager     INTEGER
);

