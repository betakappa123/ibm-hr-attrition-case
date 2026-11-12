-- ============================================================================
-- 02_inspect_and_validate.sql
-- SQL data quality inspection for raw_employees:
--   1. Row count & primary-key duplicate check
--   2. Full-record (non-ID) duplicate check
--   3. Missing / NULL / blank string audit across all 35 columns
--   4. Data type & domain range validation
--   5. Constant (zero-variance) column detection
--   6. Target variable (Attrition) class balance summary
-- ============================================================================

-- @name: row_and_id_summary
SELECT
    COUNT(*) AS total_rows,
    COUNT(DISTINCT EmployeeNumber) AS distinct_employee_numbers,
    COUNT(*) - COUNT(DISTINCT EmployeeNumber) AS duplicate_employee_numbers
FROM raw_employees;

-- @name: full_row_duplicates
SELECT
    COUNT(*) AS duplicate_feature_groups
FROM (
    SELECT
        Age, Attrition, BusinessTravel, DailyRate, Department,
        DistanceFromHome, Education, EducationField, EnvironmentSatisfaction,
        Gender, HourlyRate, JobInvolvement, JobLevel, JobRole,
        JobSatisfaction, MaritalStatus, MonthlyIncome, MonthlyRate,
        NumCompaniesWorked, OverTime, PercentSalaryHike, PerformanceRating,
        RelationshipSatisfaction, StockOptionLevel, TotalWorkingYears,
        TrainingTimesLastYear, WorkLifeBalance, YearsAtCompany,
        YearsInCurrentRole, YearsSinceLastPromotion, YearsWithCurrManager,
        COUNT(*) AS grp_count
    FROM raw_employees
    GROUP BY
        Age, Attrition, BusinessTravel, DailyRate, Department,
        DistanceFromHome, Education, EducationField, EnvironmentSatisfaction,
        Gender, HourlyRate, JobInvolvement, JobLevel, JobRole,
        JobSatisfaction, MaritalStatus, MonthlyIncome, MonthlyRate,
        NumCompaniesWorked, OverTime, PercentSalaryHike, PerformanceRating,
        RelationshipSatisfaction, StockOptionLevel, TotalWorkingYears,
        TrainingTimesLastYear, WorkLifeBalance, YearsAtCompany,
        YearsInCurrentRole, YearsSinceLastPromotion, YearsWithCurrManager
    HAVING COUNT(*) > 1
);

-- @name: missing_values_audit
SELECT
    SUM(CASE WHEN Age IS NULL THEN 1 ELSE 0 END) AS missing_Age,
    SUM(CASE WHEN Attrition IS NULL OR TRIM(Attrition) = '' THEN 1 ELSE 0 END) AS missing_Attrition,
    SUM(CASE WHEN BusinessTravel IS NULL OR TRIM(BusinessTravel) = '' THEN 1 ELSE 0 END) AS missing_BusinessTravel,
    SUM(CASE WHEN DailyRate IS NULL THEN 1 ELSE 0 END) AS missing_DailyRate,
    SUM(CASE WHEN Department IS NULL OR TRIM(Department) = '' THEN 1 ELSE 0 END) AS missing_Department,
    SUM(CASE WHEN DistanceFromHome IS NULL THEN 1 ELSE 0 END) AS missing_DistanceFromHome,
    SUM(CASE WHEN Education IS NULL THEN 1 ELSE 0 END) AS missing_Education,
    SUM(CASE WHEN EducationField IS NULL OR TRIM(EducationField) = '' THEN 1 ELSE 0 END) AS missing_EducationField,
    SUM(CASE WHEN EmployeeCount IS NULL THEN 1 ELSE 0 END) AS missing_EmployeeCount,
    SUM(CASE WHEN EmployeeNumber IS NULL THEN 1 ELSE 0 END) AS missing_EmployeeNumber,
    SUM(CASE WHEN EnvironmentSatisfaction IS NULL THEN 1 ELSE 0 END) AS missing_EnvironmentSatisfaction,
    SUM(CASE WHEN Gender IS NULL OR TRIM(Gender) = '' THEN 1 ELSE 0 END) AS missing_Gender,
    SUM(CASE WHEN HourlyRate IS NULL THEN 1 ELSE 0 END) AS missing_HourlyRate,
    SUM(CASE WHEN JobInvolvement IS NULL THEN 1 ELSE 0 END) AS missing_JobInvolvement,
    SUM(CASE WHEN JobLevel IS NULL THEN 1 ELSE 0 END) AS missing_JobLevel,
    SUM(CASE WHEN JobRole IS NULL OR TRIM(JobRole) = '' THEN 1 ELSE 0 END) AS missing_JobRole,
    SUM(CASE WHEN JobSatisfaction IS NULL THEN 1 ELSE 0 END) AS missing_JobSatisfaction,
    SUM(CASE WHEN MaritalStatus IS NULL OR TRIM(MaritalStatus) = '' THEN 1 ELSE 0 END) AS missing_MaritalStatus,
    SUM(CASE WHEN MonthlyIncome IS NULL THEN 1 ELSE 0 END) AS missing_MonthlyIncome,
    SUM(CASE WHEN MonthlyRate IS NULL THEN 1 ELSE 0 END) AS missing_MonthlyRate,
    SUM(CASE WHEN NumCompaniesWorked IS NULL THEN 1 ELSE 0 END) AS missing_NumCompaniesWorked,
    SUM(CASE WHEN Over18 IS NULL OR TRIM(Over18) = '' THEN 1 ELSE 0 END) AS missing_Over18,
    SUM(CASE WHEN OverTime IS NULL OR TRIM(OverTime) = '' THEN 1 ELSE 0 END) AS missing_OverTime,
    SUM(CASE WHEN PercentSalaryHike IS NULL THEN 1 ELSE 0 END) AS missing_PercentSalaryHike,
    SUM(CASE WHEN PerformanceRating IS NULL THEN 1 ELSE 0 END) AS missing_PerformanceRating,
    SUM(CASE WHEN RelationshipSatisfaction IS NULL THEN 1 ELSE 0 END) AS missing_RelationshipSatisfaction,
    SUM(CASE WHEN StandardHours IS NULL THEN 1 ELSE 0 END) AS missing_StandardHours,
    SUM(CASE WHEN StockOptionLevel IS NULL THEN 1 ELSE 0 END) AS missing_StockOptionLevel,
    SUM(CASE WHEN TotalWorkingYears IS NULL THEN 1 ELSE 0 END) AS missing_TotalWorkingYears,
    SUM(CASE WHEN TrainingTimesLastYear IS NULL THEN 1 ELSE 0 END) AS missing_TrainingTimesLastYear,
    SUM(CASE WHEN WorkLifeBalance IS NULL THEN 1 ELSE 0 END) AS missing_WorkLifeBalance,
    SUM(CASE WHEN YearsAtCompany IS NULL THEN 1 ELSE 0 END) AS missing_YearsAtCompany,
    SUM(CASE WHEN YearsInCurrentRole IS NULL THEN 1 ELSE 0 END) AS missing_YearsInCurrentRole,
    SUM(CASE WHEN YearsSinceLastPromotion IS NULL THEN 1 ELSE 0 END) AS missing_YearsSinceLastPromotion,
    SUM(CASE WHEN YearsWithCurrManager IS NULL THEN 1 ELSE 0 END) AS missing_YearsWithCurrManager
FROM raw_employees;

-- @name: type_and_domain_audit
SELECT
    SUM(CASE WHEN typeof(Age) != 'integer' OR Age < 18 OR Age > 100 THEN 1 ELSE 0 END) AS invalid_Age,
    SUM(CASE WHEN TRIM(Attrition) NOT IN ('Yes', 'No') THEN 1 ELSE 0 END) AS invalid_Attrition,
    SUM(CASE WHEN typeof(MonthlyIncome) != 'integer' OR MonthlyIncome <= 0 THEN 1 ELSE 0 END) AS invalid_MonthlyIncome,
    SUM(CASE WHEN Education NOT BETWEEN 1 AND 5 THEN 1 ELSE 0 END) AS invalid_Education,
    SUM(CASE WHEN JobLevel NOT BETWEEN 1 AND 5 THEN 1 ELSE 0 END) AS invalid_JobLevel,
    SUM(CASE WHEN EnvironmentSatisfaction NOT BETWEEN 1 AND 4 THEN 1 ELSE 0 END) AS invalid_EnvironmentSatisfaction,
    SUM(CASE WHEN JobInvolvement NOT BETWEEN 1 AND 4 THEN 1 ELSE 0 END) AS invalid_JobInvolvement,
    SUM(CASE WHEN JobSatisfaction NOT BETWEEN 1 AND 4 THEN 1 ELSE 0 END) AS invalid_JobSatisfaction,
    SUM(CASE WHEN RelationshipSatisfaction NOT BETWEEN 1 AND 4 THEN 1 ELSE 0 END) AS invalid_RelationshipSatisfaction,
    SUM(CASE WHEN WorkLifeBalance NOT BETWEEN 1 AND 4 THEN 1 ELSE 0 END) AS invalid_WorkLifeBalance,
    SUM(CASE WHEN StockOptionLevel NOT BETWEEN 0 AND 3 THEN 1 ELSE 0 END) AS invalid_StockOptionLevel,
    SUM(CASE WHEN TRIM(OverTime) NOT IN ('Yes', 'No') THEN 1 ELSE 0 END) AS invalid_OverTime,
    SUM(CASE WHEN TotalWorkingYears < 0 OR YearsAtCompany < 0 OR YearsInCurrentRole < 0 THEN 1 ELSE 0 END) AS negative_tenure,
    SUM(CASE WHEN YearsAtCompany > TotalWorkingYears THEN 1 ELSE 0 END) AS company_exceeds_total_years
FROM raw_employees;

-- @name: constant_columns_audit
SELECT column_name, distinct_count
FROM (
    SELECT 'Age' AS column_name, COUNT(DISTINCT Age) AS distinct_count FROM raw_employees
    UNION ALL SELECT 'Attrition', COUNT(DISTINCT Attrition) FROM raw_employees
    UNION ALL SELECT 'BusinessTravel', COUNT(DISTINCT BusinessTravel) FROM raw_employees
    UNION ALL SELECT 'DailyRate', COUNT(DISTINCT DailyRate) FROM raw_employees
    UNION ALL SELECT 'Department', COUNT(DISTINCT Department) FROM raw_employees
    UNION ALL SELECT 'DistanceFromHome', COUNT(DISTINCT DistanceFromHome) FROM raw_employees
    UNION ALL SELECT 'Education', COUNT(DISTINCT Education) FROM raw_employees
    UNION ALL SELECT 'EducationField', COUNT(DISTINCT EducationField) FROM raw_employees
    UNION ALL SELECT 'EmployeeCount', COUNT(DISTINCT EmployeeCount) FROM raw_employees
    UNION ALL SELECT 'EmployeeNumber', COUNT(DISTINCT EmployeeNumber) FROM raw_employees
    UNION ALL SELECT 'EnvironmentSatisfaction', COUNT(DISTINCT EnvironmentSatisfaction) FROM raw_employees
    UNION ALL SELECT 'Gender', COUNT(DISTINCT Gender) FROM raw_employees
    UNION ALL SELECT 'HourlyRate', COUNT(DISTINCT HourlyRate) FROM raw_employees
    UNION ALL SELECT 'JobInvolvement', COUNT(DISTINCT JobInvolvement) FROM raw_employees
    UNION ALL SELECT 'JobLevel', COUNT(DISTINCT JobLevel) FROM raw_employees
    UNION ALL SELECT 'JobRole', COUNT(DISTINCT JobRole) FROM raw_employees
    UNION ALL SELECT 'JobSatisfaction', COUNT(DISTINCT JobSatisfaction) FROM raw_employees
    UNION ALL SELECT 'MaritalStatus', COUNT(DISTINCT MaritalStatus) FROM raw_employees
    UNION ALL SELECT 'MonthlyIncome', COUNT(DISTINCT MonthlyIncome) FROM raw_employees
    UNION ALL SELECT 'MonthlyRate', COUNT(DISTINCT MonthlyRate) FROM raw_employees
    UNION ALL SELECT 'NumCompaniesWorked', COUNT(DISTINCT NumCompaniesWorked) FROM raw_employees
    UNION ALL SELECT 'Over18', COUNT(DISTINCT Over18) FROM raw_employees
    UNION ALL SELECT 'OverTime', COUNT(DISTINCT OverTime) FROM raw_employees
    UNION ALL SELECT 'PercentSalaryHike', COUNT(DISTINCT PercentSalaryHike) FROM raw_employees
    UNION ALL SELECT 'PerformanceRating', COUNT(DISTINCT PerformanceRating) FROM raw_employees
    UNION ALL SELECT 'RelationshipSatisfaction', COUNT(DISTINCT RelationshipSatisfaction) FROM raw_employees
    UNION ALL SELECT 'StandardHours', COUNT(DISTINCT StandardHours) FROM raw_employees
    UNION ALL SELECT 'StockOptionLevel', COUNT(DISTINCT StockOptionLevel) FROM raw_employees
    UNION ALL SELECT 'TotalWorkingYears', COUNT(DISTINCT TotalWorkingYears) FROM raw_employees
    UNION ALL SELECT 'TrainingTimesLastYear', COUNT(DISTINCT TrainingTimesLastYear) FROM raw_employees
    UNION ALL SELECT 'WorkLifeBalance', COUNT(DISTINCT WorkLifeBalance) FROM raw_employees
    UNION ALL SELECT 'YearsAtCompany', COUNT(DISTINCT YearsAtCompany) FROM raw_employees
    UNION ALL SELECT 'YearsInCurrentRole', COUNT(DISTINCT YearsInCurrentRole) FROM raw_employees
    UNION ALL SELECT 'YearsSinceLastPromotion', COUNT(DISTINCT YearsSinceLastPromotion) FROM raw_employees
    UNION ALL SELECT 'YearsWithCurrManager', COUNT(DISTINCT YearsWithCurrManager) FROM raw_employees
)
WHERE distinct_count <= 1
ORDER BY column_name;

-- @name: target_distribution
SELECT
    TRIM(Attrition) AS attrition_label,
    COUNT(*) AS employee_count,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) AS pct_of_workforce
FROM raw_employees
GROUP BY TRIM(Attrition)
ORDER BY employee_count DESC;

