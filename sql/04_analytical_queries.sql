-- ============================================================================
-- 04_analytical_queries.sql
-- Exploratory SQL cohort summaries over `clean_employees` covering:
--   - OverTime
--   - JobRole
--   - MonthlyIncome quartiles
--   - YearsAtCompany tenure bands
--   - JobSatisfaction
-- ============================================================================

-- @name: attrition_by_overtime
SELECT
    OverTime,
    COUNT(*) AS headcount,
    SUM(AttritionFlag) AS attrition_count,
    ROUND(100.0 * AVG(AttritionFlag), 2) AS attrition_rate_pct,
    ROUND(AVG(MonthlyIncome), 0) AS avg_monthly_income,
    ROUND(AVG(YearsAtCompany), 1) AS avg_years_at_company
FROM clean_employees
GROUP BY OverTime
ORDER BY attrition_rate_pct DESC;

-- @name: attrition_by_job_role
SELECT
    JobRole,
    Department,
    COUNT(*) AS headcount,
    SUM(AttritionFlag) AS attrition_count,
    ROUND(100.0 * AVG(AttritionFlag), 2) AS attrition_rate_pct,
    ROUND(AVG(MonthlyIncome), 0) AS avg_monthly_income,
    ROUND(100.0 * AVG(CASE WHEN OverTime = 'Yes' THEN 1.0 ELSE 0.0 END), 1) AS overtime_pct
FROM clean_employees
GROUP BY JobRole, Department
ORDER BY attrition_rate_pct DESC;

-- @name: attrition_by_income_quartile
WITH income_quartiles AS (
    SELECT
        EmployeeNumber,
        AttritionFlag,
        MonthlyIncome,
        OverTime,
        NTILE(4) OVER (ORDER BY MonthlyIncome) AS income_quartile
    FROM clean_employees
)
SELECT
    'Q' || income_quartile AS income_quartile_label,
    MIN(MonthlyIncome) AS min_income,
    MAX(MonthlyIncome) AS max_income,
    COUNT(*) AS headcount,
    SUM(AttritionFlag) AS attrition_count,
    ROUND(100.0 * AVG(AttritionFlag), 2) AS attrition_rate_pct
FROM income_quartiles
GROUP BY income_quartile
ORDER BY income_quartile;

-- @name: attrition_by_tenure_band
WITH tenure_bands AS (
    SELECT
        AttritionFlag,
        MonthlyIncome,
        YearsAtCompany,
        CASE
            WHEN YearsAtCompany < 2 THEN '0-1 yrs (New Hire)'
            WHEN YearsAtCompany BETWEEN 2 AND 4 THEN '2-4 yrs (Early)'
            WHEN YearsAtCompany BETWEEN 5 AND 9 THEN '5-9 yrs (Established)'
            ELSE '10+ yrs (Tenured)'
        END AS tenure_band,
        CASE
            WHEN YearsAtCompany < 2 THEN 1
            WHEN YearsAtCompany BETWEEN 2 AND 4 THEN 2
            WHEN YearsAtCompany BETWEEN 5 AND 9 THEN 3
            ELSE 4
        END AS sort_order
    FROM clean_employees
)
SELECT
    tenure_band,
    COUNT(*) AS headcount,
    SUM(AttritionFlag) AS attrition_count,
    ROUND(100.0 * AVG(AttritionFlag), 2) AS attrition_rate_pct,
    ROUND(AVG(MonthlyIncome), 0) AS avg_monthly_income
FROM tenure_bands
GROUP BY tenure_band, sort_order
ORDER BY sort_order;

-- @name: attrition_by_job_satisfaction
SELECT
    JobSatisfaction,
    CASE JobSatisfaction
        WHEN 1 THEN '1 - Low'
        WHEN 2 THEN '2 - Medium'
        WHEN 3 THEN '3 - High'
        WHEN 4 THEN '4 - Very High'
    END AS satisfaction_label,
    COUNT(*) AS headcount,
    SUM(AttritionFlag) AS attrition_count,
    ROUND(100.0 * AVG(AttritionFlag), 2) AS attrition_rate_pct
FROM clean_employees
GROUP BY JobSatisfaction
ORDER BY JobSatisfaction;

