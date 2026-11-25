-- =====================================================================
-- Saudi Fair Hiring Audit - SQL (SQLite; standard SQL, portable to
-- PostgreSQL / SQL Server with minor type changes)
--
-- Three parts, read by pipeline.py and app.py:
--   @@ SCHEMA   tables, constraints, indexes, log tables
--   @@ VIEWS    reporting views used by the app and Power BI
--   @@ AUDIT    named audit queries ("-- name: ..."), results saved to output/
-- =====================================================================

-- @@ SCHEMA
DROP VIEW  IF EXISTS vw_fairness_scorecard;
DROP VIEW  IF EXISTS vw_selection_by_group;
DROP VIEW  IF EXISTS vw_applicant_predictions;
DROP TABLE IF EXISTS data_quality_profile;
DROP TABLE IF EXISTS data_quality_log;
DROP TABLE IF EXISTS agent_logs;
DROP TABLE IF EXISTS scoring_requests;
DROP TABLE IF EXISTS feature_importance;
DROP TABLE IF EXISTS proxy_scores;
DROP TABLE IF EXISTS group_metrics;
DROP TABLE IF EXISTS model_metrics;
DROP TABLE IF EXISTS predictions;
DROP TABLE IF EXISTS models;
DROP TABLE IF EXISTS applicants;

-- One row per (fictional) job application
CREATE TABLE applicants (
    applicant_id        TEXT PRIMARY KEY,
    application_year    INTEGER NOT NULL,
    job_family          TEXT    NOT NULL,
    gender              TEXT    NOT NULL CHECK (gender IN ('Male', 'Female')),
    nationality         TEXT    NOT NULL CHECK (nationality IN ('Saudi', 'Non-Saudi')),
    region              TEXT    NOT NULL,
    region_group        TEXT    NOT NULL,
    age                 INTEGER NOT NULL CHECK (age BETWEEN 18 AND 70),
    degree_field        TEXT    NOT NULL,
    university_tier     INTEGER NOT NULL CHECK (university_tier IN (1, 2, 3)),
    gpa                 REAL    NOT NULL CHECK (gpa BETWEEN 0 AND 5),
    years_experience    INTEGER NOT NULL,
    english_score       REAL    NOT NULL,
    certifications      INTEGER NOT NULL,
    internship          INTEGER NOT NULL CHECK (internship IN (0, 1)),
    career_gap_months   INTEGER NOT NULL,   -- potential proxy for gender
    commute_km          INTEGER NOT NULL,   -- potential proxy for region
    qualification_score REAL    NOT NULL,   -- synthetic ground-truth ability (0-100)
    truly_qualified     INTEGER NOT NULL CHECK (truly_qualified IN (0, 1)),
    hired_historical    INTEGER NOT NULL CHECK (hired_historical IN (0, 1))
);

-- Models compared in the study
CREATE TABLE models (
    model_id        TEXT PRIMARY KEY,
    model_name      TEXT NOT NULL,
    algorithm       TEXT NOT NULL,
    uses_sensitive  INTEGER NOT NULL,
    uses_proxies    INTEGER NOT NULL,
    mitigation      TEXT NOT NULL,
    description     TEXT
);

-- Test-set predictions: one row per applicant per model
CREATE TABLE predictions (
    applicant_id    TEXT NOT NULL REFERENCES applicants(applicant_id),
    model_id        TEXT NOT NULL REFERENCES models(model_id),
    score           REAL NOT NULL,
    predicted_hire  INTEGER NOT NULL,
    PRIMARY KEY (applicant_id, model_id)
);

CREATE TABLE model_metrics (
    model_id                TEXT PRIMARY KEY REFERENCES models(model_id),
    accuracy_vs_history     REAL,
    f1_vs_history           REAL,
    auc_vs_history          REAL,
    accuracy_vs_truth       REAL,
    recall_qualified        REAL,
    selection_rate          REAL,
    di_gender               REAL,
    di_nationality          REAL,
    di_region_group         REAL,
    eod_gender              REAL,
    eod_region_group        REAL,
    passes_four_fifths      INTEGER
);

CREATE TABLE group_metrics (
    model_id                TEXT NOT NULL REFERENCES models(model_id),
    attribute               TEXT NOT NULL,
    group_name              TEXT NOT NULL,
    n                       INTEGER,
    qualified_share         REAL,
    selection_rate          REAL,
    impact_ratio_vs_best    REAL,
    tpr                     REAL,
    fpr                     REAL,
    mean_score              REAL,
    PRIMARY KEY (model_id, attribute, group_name)
);

CREATE TABLE proxy_scores (
    sensitive_attribute     TEXT NOT NULL,
    feature                 TEXT NOT NULL,
    mutual_information      REAL,
    auc_predicting_group    REAL,
    risk_level              TEXT,
    PRIMARY KEY (sensitive_attribute, feature)
);

CREATE TABLE feature_importance (
    model_id        TEXT NOT NULL REFERENCES models(model_id),
    feature         TEXT NOT NULL,
    importance      REAL,
    direction       TEXT,
    PRIMARY KEY (model_id, feature)
);

-- Written by the cleaning step: every cleaning action and the before/after profile
CREATE TABLE data_quality_log (
    step_no         INTEGER PRIMARY KEY,
    category        TEXT NOT NULL,
    step            TEXT NOT NULL,
    description     TEXT,
    rows_affected   INTEGER,
    rows_after      INTEGER
);

CREATE TABLE data_quality_profile (
    metric          TEXT PRIMARY KEY,
    raw             INTEGER,
    clean           INTEGER
);

-- Written by the web app: every applicant scored through the API (audit trail)
CREATE TABLE scoring_requests (
    request_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at      TEXT NOT NULL DEFAULT (datetime('now')),
    model_id        TEXT NOT NULL REFERENCES models(model_id),
    inputs_json     TEXT NOT NULL,
    score           REAL NOT NULL,
    decision        TEXT NOT NULL,
    percentile      REAL
);

-- Written by the AI assistant: every question, answer and tool call (accountability)
CREATE TABLE agent_logs (
    log_id          INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at      TEXT NOT NULL DEFAULT (datetime('now')),
    mode            TEXT NOT NULL,           -- 'claude' or 'offline'
    question        TEXT NOT NULL,
    answer          TEXT NOT NULL,
    tools_used      TEXT,
    sql_executed    TEXT
);

CREATE INDEX ix_applicants_gender  ON applicants(gender);
CREATE INDEX ix_applicants_region  ON applicants(region_group);
CREATE INDEX ix_predictions_model  ON predictions(model_id);

-- @@ VIEWS
-- Predictions joined with applicant attributes
CREATE VIEW vw_applicant_predictions AS
SELECT
    p.model_id,
    m.model_name,
    a.applicant_id,
    a.application_year,
    a.job_family,
    a.gender,
    a.nationality,
    a.region,
    a.region_group,
    a.qualification_score,
    a.truly_qualified,
    a.hired_historical,
    p.score,
    p.predicted_hire,
    CASE
        WHEN a.truly_qualified = 1 AND p.predicted_hire = 1 THEN 'Qualified - selected'
        WHEN a.truly_qualified = 1 AND p.predicted_hire = 0 THEN 'Qualified - rejected (missed talent)'
        WHEN a.truly_qualified = 0 AND p.predicted_hire = 1 THEN 'Not qualified - selected'
        ELSE 'Not qualified - rejected'
    END AS outcome_type
FROM predictions p
JOIN applicants a ON a.applicant_id = p.applicant_id
JOIN models     m ON m.model_id     = p.model_id;

-- Selection rate per model and gender, with the impact ratio vs the best group
CREATE VIEW vw_selection_by_group AS
WITH rates AS (
    SELECT model_id, gender AS group_name,
           COUNT(*)                AS applicants,
           SUM(predicted_hire)     AS selected,
           AVG(predicted_hire*1.0) AS selection_rate
    FROM vw_applicant_predictions
    GROUP BY model_id, gender
)
SELECT r.*,
       r.selection_rate / MAX(r.selection_rate) OVER (PARTITION BY r.model_id) AS impact_ratio
FROM rates r;

-- One-line verdict per model
CREATE VIEW vw_fairness_scorecard AS
SELECT
    m.model_id,
    m.model_name,
    m.mitigation,
    ROUND(mm.auc_vs_history, 3)   AS auc,
    ROUND(mm.recall_qualified, 3) AS recall_qualified,
    ROUND(mm.di_gender, 3)        AS di_gender,
    ROUND(mm.di_region_group, 3)  AS di_region_group,
    CASE WHEN mm.passes_four_fifths = 1 THEN 'PASS' ELSE 'FAIL' END AS four_fifths_rule
FROM models m
JOIN model_metrics mm ON mm.model_id = m.model_id;

-- @@ AUDIT
-- name: q01_dataset_overview
-- How many applicants, and what share were hired in the past?
SELECT COUNT(*)                              AS applicants,
       ROUND(AVG(hired_historical) * 100, 1) AS historical_hire_rate_pct,
       ROUND(AVG(truly_qualified) * 100, 1)  AS truly_qualified_pct
FROM applicants;

-- name: q02_historical_selection_by_gender
-- Four-fifths rule on PAST decisions: impact ratio below 0.80 signals adverse impact
WITH g AS (
    SELECT gender, COUNT(*) AS n, AVG(hired_historical * 1.0) AS hire_rate
    FROM applicants GROUP BY gender
)
SELECT gender, n,
       ROUND(hire_rate, 3)                                 AS hire_rate,
       ROUND(hire_rate / (SELECT MAX(hire_rate) FROM g), 3) AS impact_ratio,
       CASE WHEN hire_rate / (SELECT MAX(hire_rate) FROM g) < 0.8 THEN 'ADVERSE IMPACT' ELSE 'OK' END AS verdict
FROM g ORDER BY hire_rate DESC;

-- name: q03_historical_selection_by_region
WITH r AS (
    SELECT region_group, COUNT(*) AS n, AVG(hired_historical * 1.0) AS hire_rate
    FROM applicants GROUP BY region_group
)
SELECT region_group, n,
       ROUND(hire_rate, 3)                                 AS hire_rate,
       ROUND(hire_rate / (SELECT MAX(hire_rate) FROM r), 3) AS impact_ratio
FROM r ORDER BY hire_rate DESC;

-- name: q04_equally_qualified_comparison
-- The key test: among applicants who are TRULY qualified, were hire rates equal?
SELECT gender,
       COUNT(*)                                  AS qualified_applicants,
       ROUND(AVG(hired_historical) * 100, 1)     AS hired_pct,
       ROUND(AVG(qualification_score), 1)        AS avg_qualification
FROM applicants
WHERE truly_qualified = 1
GROUP BY gender;

-- name: q05_qualification_band_gap
-- Hire rate by qualification band and gender (gap should be ~0 if fair)
SELECT CASE
           WHEN qualification_score < 40 THEN '1: <40'
           WHEN qualification_score < 50 THEN '2: 40-49'
           WHEN qualification_score < 60 THEN '3: 50-59'
           WHEN qualification_score < 70 THEN '4: 60-69'
           ELSE '5: 70+'
       END AS qualification_band,
       ROUND(AVG(CASE WHEN gender = 'Male'   THEN hired_historical END) * 100, 1) AS male_hired_pct,
       ROUND(AVG(CASE WHEN gender = 'Female' THEN hired_historical END) * 100, 1) AS female_hired_pct,
       ROUND((AVG(CASE WHEN gender = 'Male'   THEN hired_historical END)
            - AVG(CASE WHEN gender = 'Female' THEN hired_historical END)) * 100, 1) AS gap_pct_points
FROM applicants
GROUP BY qualification_band
ORDER BY qualification_band;

-- name: q06_proxy_check_career_gap
-- Does the "neutral" career-gap feature differ by gender? (proxy warning)
SELECT gender,
       ROUND(AVG(career_gap_months), 1) AS avg_career_gap_months,
       ROUND(AVG(qualification_score), 1) AS avg_qualification
FROM applicants GROUP BY gender;

-- name: q07_proxy_check_commute
SELECT region_group,
       ROUND(AVG(commute_km), 0)          AS avg_commute_km,
       ROUND(AVG(qualification_score), 1) AS avg_qualification
FROM applicants GROUP BY region_group ORDER BY avg_commute_km;

-- name: q08_model_scorecard
SELECT * FROM vw_fairness_scorecard ORDER BY di_gender DESC;

-- name: q09_missed_talent_by_model
-- Qualified applicants each model rejects, split by gender
SELECT model_name, gender,
       SUM(CASE WHEN outcome_type = 'Qualified - rejected (missed talent)' THEN 1 ELSE 0 END) AS missed_talent,
       SUM(truly_qualified) AS qualified,
       ROUND(1.0 * SUM(CASE WHEN outcome_type = 'Qualified - rejected (missed talent)' THEN 1 ELSE 0 END)
             / SUM(truly_qualified), 3) AS miss_rate
FROM vw_applicant_predictions
GROUP BY model_name, gender
ORDER BY model_name, gender;

-- name: q10_top_proxies
SELECT sensitive_attribute, feature, ROUND(auc_predicting_group, 3) AS auc, risk_level
FROM proxy_scores
WHERE feature <> 'ALL FEATURES COMBINED'
ORDER BY auc_predicting_group DESC
LIMIT 6;

-- name: q11_data_cleaning_summary
-- What the cleaning step fixed, by category
SELECT category,
       COUNT(*)           AS steps,
       SUM(rows_affected) AS rows_affected
FROM data_quality_log
WHERE category <> 'Load'
GROUP BY category
ORDER BY rows_affected DESC;
