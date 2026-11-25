"""Saudi Fair Hiring Audit - data pipeline (one file).

Steps:  synthetic data -> messy HR export -> cleaning (24 logged steps) -> SQLite database
        -> 5 models -> fairness metrics, proxy detection, explainability
        -> CSV files for Power BI, Excel workbooks, SQL audit results

Run:    python pipeline.py              (add --regenerate for new synthetic data)
Output: everything is written to the output/ folder.
All data is synthetic. No real person is represented.
"""
from __future__ import annotations

import argparse
import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.chart import BarChart, Reference
from openpyxl.formatting.rule import CellIsRule
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.table import Table, TableStyleInfo
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "output"
SQL_FILE = ROOT / "database.sql"
SEED = 42
RECOMMENDED_MODEL = "M5"



# ==================================================================================================
# 1. SYNTHETIC DATA (with documented, planted bias)
# ==================================================================================================
REGIONS = {
    # region: (share, region_group, mean commute to a Riyadh-based employer in km)
    "Riyadh": (0.30, "Central", 25),
    "Qassim": (0.06, "Central", 330),
    "Makkah": (0.22, "Western", 870),
    "Madinah": (0.07, "Western", 850),
    "Eastern Province": (0.17, "Eastern", 400),
    "Asir": (0.06, "Southern", 900),
    "Jazan": (0.04, "Southern", 1050),
    "Tabuk": (0.04, "Northern", 1250),
    "Hail": (0.04, "Northern", 640),
}
JOB_FAMILIES = ["Data Analyst", "Software Engineer", "Cybersecurity Specialist", "Business Analyst", "HR Specialist"]
DEGREES = ["Computer Science", "Information Systems", "Engineering", "Statistics", "Business Administration"]

# Bias planted in historical decisions (log-odds). Documented in docs/ethics_analysis.md.
BIAS_FEMALE = -1.3
BIAS_REGION_GROUP = {"Central": 0.0, "Western": 0.0, "Eastern": 0.0, "Southern": -0.9, "Northern": -0.8}
SAUDIZATION_PREFERENCE = 0.35  # lawful national-employment policy, analysed separately


def generate(n: int = 10000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    gender = rng.choice(["Male", "Female"], size=n, p=[0.58, 0.42])
    nationality = rng.choice(["Saudi", "Non-Saudi"], size=n, p=[0.76, 0.24])
    names = list(REGIONS)
    region = rng.choice(names, size=n, p=[REGIONS[r][0] for r in names])
    region_group = np.array([REGIONS[r][1] for r in region])
    age = rng.integers(21, 46, size=n)

    # University tier (1 = top). Same distribution in every region, so true ability
    # does not differ by group and any gap in outcomes is caused by bias.
    university_tier = rng.choice([1, 2, 3], size=n, p=[0.25, 0.45, 0.30])
    degree_field = rng.choice(DEGREES, size=n, p=[0.28, 0.18, 0.2, 0.1, 0.24])
    job_family = rng.choice(JOB_FAMILIES, size=n, p=[0.26, 0.24, 0.16, 0.2, 0.14])

    gpa = np.clip(rng.normal(3.85, 0.55, n), 2.0, 5.0).round(2)  # Saudi 5-point scale
    years_experience = np.clip(rng.poisson(np.maximum(age - 23, 0) * 0.55), 0, 20)
    english_score = np.clip(rng.normal(6.2, 0.9, n), 4.0, 9.0).round(1)  # IELTS-like
    certifications = rng.poisson(1.2, n).clip(0, 6)
    internship = rng.binomial(1, 0.45, n)

    # Proxies: carry group information, but no information about ability.
    career_gap_months = np.where(
        gender == "Female", rng.gamma(3.0, 7.0, n), rng.gamma(1.0, 3.0, n)
    ).round(0).astype(int)
    commute_km = np.array([max(5, rng.normal(REGIONS[r][2], 60)) for r in region]).round(0).astype(int)

    # True qualification depends ONLY on job-relevant attributes.
    z = (
        1.10 * (gpa - 3.85) / 0.55
        + 0.55 * (years_experience - years_experience.mean()) / (years_experience.std() + 1e-9)
        + 0.60 * (english_score - 6.2) / 0.9
        + 0.35 * (certifications - 1.2)
        + 0.45 * internship
        + 0.40 * (2 - university_tier)
        + rng.normal(0, 0.8, n)
    )
    qualification_score = (50 + 12 * z).clip(0, 100).round(1)
    truly_qualified = (z > np.quantile(z, 0.62)).astype(int)

    # Historical decisions = qualification + planted bias + policy + noise.
    logit = (
        2.2 * (z - np.quantile(z, 0.62))
        + np.where(gender == "Female", BIAS_FEMALE, 0.0)
        + np.array([BIAS_REGION_GROUP[g] for g in region_group])
        + np.where(nationality == "Saudi", SAUDIZATION_PREFERENCE, 0.0)
        + rng.normal(0, 0.6, n)
    )
    hired_historical = rng.binomial(1, 1 / (1 + np.exp(-logit)))

    df = pd.DataFrame({
        "applicant_id": [f"APP-{i:05d}" for i in range(1, n + 1)],
        "application_year": rng.choice([2023, 2024, 2025], size=n, p=[0.3, 0.35, 0.35]),
        "job_family": job_family,
        "gender": gender,
        "nationality": nationality,
        "region": region,
        "region_group": region_group,
        "age": age,
        "degree_field": degree_field,
        "university_tier": university_tier,
        "gpa": gpa,
        "years_experience": years_experience,
        "english_score": english_score,
        "certifications": certifications,
        "internship": internship,
        "career_gap_months": career_gap_months,
        "commute_km": commute_km,
        "qualification_score": qualification_score,
        "truly_qualified": truly_qualified,
        "hired_historical": hired_historical,
    })
    return df




# ==================================================================================================
# 2. MESSY HR EXPORT (duplicates, mixed Arabic/English labels, numbers as text, personal identifiers)
# ==================================================================================================
RAW_COLUMNS = {
    "applicant_id": "App ID", "application_year": "Application Date", "job_family": "Job Title",
    "gender": "Gender", "nationality": "Nationality", "region": "City/Region", "age": "Age",
    "degree_field": "Major", "university_tier": "University Tier", "gpa": "GPA",
    "years_experience": "Experience", "english_score": "English (IELTS)", "certifications": "Certifications",
    "internship": "Internship", "career_gap_months": "Career Gap (months)", "commute_km": "Commute (km)",
    "qualification_score": "Qualification Score [SIM]", "truly_qualified": "Truly Qualified [SIM]",
    "hired_historical": "Hired",
}

GENDER_VARIANTS = {"Male": ["Male", "male", "M", "ذكر", "MALE"], "Female": ["Female", "female", "F", "أنثى", "FEMALE"]}
NATIONALITY_VARIANTS = {"Saudi": ["Saudi", "SA", "سعودي", "saudi", "KSA"],
                        "Non-Saudi": ["Non-Saudi", "non saudi", "غير سعودي", "Expat", "NON-SAUDI"]}
REGION_VARIANTS = {
    "Riyadh": ["Riyadh", "الرياض", "riyadh", "Riyadh Region"], "Qassim": ["Qassim", "القصيم", "Al-Qassim"],
    "Makkah": ["Makkah", "مكة المكرمة", "Mecca", "Makkah Region"], "Madinah": ["Madinah", "المدينة المنورة", "Medina"],
    "Eastern Province": ["Eastern Province", "الشرقية", "Eastern", "Ash Sharqiyah"], "Asir": ["Asir", "عسير", "Aseer"],
    "Jazan": ["Jazan", "جازان", "Jizan"], "Tabuk": ["Tabuk", "تبوك"], "Hail": ["Hail", "حائل", "Ha'il"],
}
YES = ["Yes", "yes", "Y", "نعم", "1"]
NO = ["No", "no", "N", "لا", "0"]
HIRED = ["Hired", "Yes", "1", "Offer accepted"]
NOT_HIRED = ["Rejected", "No", "0", "Not selected"]

FIRST_M = ["Fahad", "Abdullah", "Mohammed", "Khalid", "Saud", "Faisal", "Omar", "Turki", "Nasser", "Yousef", "Ali", "Majed"]
FIRST_F = ["Noura", "Sara", "Reem", "Lama", "Hessa", "Abeer", "Maha", "Ghada", "Dana", "Rawan", "Lina", "Amal"]
FAMILY = ["Al-Qahtani", "Al-Otaibi", "Al-Harbi", "Al-Shehri", "Al-Ghamdi", "Al-Dosari", "Al-Mutairi", "Al-Zahrani",
          "Al-Anazi", "Al-Shammari", "Al-Malki", "Al-Subaie", "Haddad", "Khan", "Hassan", "Nasser"]


def _pick(rng, options, n):
    return np.array(options)[rng.integers(0, len(options), n)]


def make_raw_export(df: pd.DataFrame, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = len(df)
    raw = pd.DataFrame(index=df.index)

    # --- direct identifiers (PII) that an HR system would hold ---
    female = df.gender.eq("Female").to_numpy()
    first = np.where(female, _pick(rng, FIRST_F, n), _pick(rng, FIRST_M, n))
    family = _pick(rng, FAMILY, n)
    names = pd.Series([f"{a} {b}" for a, b in zip(first, family)], index=df.index)
    saudi = df.nationality.eq("Saudi").to_numpy()
    nat_id = [("1" if s else "2") + "".join(map(str, rng.integers(0, 10, 9))) for s in saudi]
    mobile = ["05" + "".join(map(str, rng.integers(0, 10, 8))) for _ in range(n)]
    email = [f"{a.lower()}.{b.lower().replace('al-', '')}{rng.integers(10, 999)}@example.com" for a, b in zip(first, family)]

    raw["App ID"] = df.applicant_id
    raw["Full Name"] = names
    raw["National ID"] = nat_id
    raw["Mobile"] = mobile
    raw["Email"] = email

    # --- dates in three formats (only the year is needed later) ---
    day = rng.integers(1, 29, n)
    month = rng.integers(1, 13, n)
    fmt = rng.choice(3, n, p=[0.6, 0.3, 0.1])
    months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
              "November", "December"]
    raw["Application Date"] = [
        f"{y}-{m:02d}-{d:02d}" if f == 0 else f"{d:02d}/{m:02d}/{y}" if f == 1 else f"{months[m - 1]} {d}, {y}"
        for y, m, d, f in zip(df.application_year, month, day, fmt)]

    def messy_text(values: pd.Series, share=0.15):
        out = values.astype(str).copy()
        idx = rng.random(n) < share
        style = rng.integers(0, 3, n)
        out[idx & (style == 0)] = out[idx & (style == 0)].str.lower()
        out[idx & (style == 1)] = "  " + out[idx & (style == 1)] + " "
        out[idx & (style == 2)] = out[idx & (style == 2)].str.upper()
        return out

    raw["Job Title"] = messy_text(df.job_family)
    raw["Gender"] = [GENDER_VARIANTS[g][k] for g, k in zip(df.gender, rng.choice(5, n, p=[0.6, 0.15, 0.1, 0.1, 0.05]))]
    raw["Nationality"] = [NATIONALITY_VARIANTS[x][k] for x, k in zip(df.nationality, rng.choice(5, n, p=[0.65, 0.1, 0.1, 0.1, 0.05]))]
    raw["City/Region"] = [REGION_VARIANTS[r][min(k, len(REGION_VARIANTS[r]) - 1)]
                          for r, k in zip(df.region, rng.choice(4, n, p=[0.6, 0.2, 0.12, 0.08]))]
    age = df.age.astype(str)
    as_float = rng.random(n) < 0.1
    age[as_float] = df.age[as_float].astype(float).astype(str)  # "29.0"
    raw["Age"] = age
    raw["Major"] = messy_text(df.degree_field, 0.1)
    raw["University Tier"] = np.where(rng.random(n) < 0.08, "Tier " + df.university_tier.astype(str),
                                      df.university_tier.astype(str))

    gpa = df.gpa.map(lambda v: f"{v:.2f}")
    comma = rng.random(n) < 0.04
    x10 = (rng.random(n) < 0.01) & ~comma
    gpa[comma] = gpa[comma].str.replace(".", ",", regex=False)
    gpa[x10] = (df.gpa[x10] * 10).map(lambda v: f"{v:.1f}")
    raw["GPA"] = gpa

    exp = df.years_experience.astype(str)
    w = rng.random(n) < 0.05
    exp[w] = exp[w] + np.where(rng.random(w.sum()) < 0.5, " years", " سنوات")
    raw["Experience"] = exp

    eng = df.english_score.map(lambda v: f"{v:.1f}")
    b = rng.random(n) < 0.03
    eng[b] = "Band " + eng[b]
    raw["English (IELTS)"] = eng
    raw["Certifications"] = df.certifications
    raw["Internship"] = [(_pick(rng, YES, 1) if v else _pick(rng, NO, 1))[0] for v in df.internship]
    raw["Career Gap (months)"] = df.career_gap_months
    raw["Commute (km)"] = df.commute_km
    raw["Qualification Score [SIM]"] = df.qualification_score
    raw["Truly Qualified [SIM]"] = df.truly_qualified
    raw["Hired"] = [(_pick(rng, HIRED, 1) if v else _pick(rng, NOT_HIRED, 1))[0] for v in df.hired_historical]

    # --- missing contact details (not needed for modelling) ---
    raw.loc[rng.random(n) < 0.05, "Email"] = np.nan
    raw.loc[rng.random(n) < 0.03, "Mobile"] = np.nan

    # --- duplicates: 250 exact, 80 near-duplicates (extra spaces / case in text fields) ---
    exact = raw.sample(250, random_state=seed)
    near = raw.sample(80, random_state=seed + 1).copy()
    near["Gender"] = " " + near["Gender"].astype(str).str.upper() + " "
    near["Job Title"] = near["Job Title"].astype(str).str.lower()

    # --- junk rows: test records and rows with no ID ---
    junk = raw.sample(65, random_state=seed + 2).copy()
    junk.iloc[:40, junk.columns.get_loc("App ID")] = [f"TEST-{i:03d}" for i in range(40)]
    junk.iloc[:40, junk.columns.get_loc("Full Name")] = "test user"
    junk.iloc[40:, junk.columns.get_loc("App ID")] = np.nan

    out = pd.concat([raw, exact, near, junk], ignore_index=True)
    return out.sample(frac=1, random_state=seed).reset_index(drop=True)


# ==================================================================================================
# 3. DATA CLEANING (24 logged steps)
# ==================================================================================================
PII_COLUMNS = ["Full Name", "National ID", "Mobile", "Email"]
CLEAN_ORDER = ["applicant_id", "application_year", "job_family", "gender", "nationality", "region", "region_group",
               "age", "degree_field", "university_tier", "gpa", "years_experience", "english_score", "certifications",
               "internship", "career_gap_months", "commute_km", "qualification_score", "truly_qualified",
               "hired_historical"]


def _lookup(variants: dict[str, list[str]]) -> dict[str, str]:
    """Build {normalised variant -> canonical value}."""
    return {v.strip().lower(): k for k, vs in variants.items() for v in vs + [k]}


@dataclass
class CleaningLog:
    steps: list[dict] = field(default_factory=list)

    def add(self, step: str, description: str, affected: int, rows_after: int, category: str) -> None:
        self.steps.append({"step_no": len(self.steps) + 1, "category": category, "step": step,
                           "description": description, "rows_affected": int(affected), "rows_after": int(rows_after)})

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.steps)


def _standardise(series: pd.Series, mapping: dict[str, str]) -> tuple[pd.Series, int]:
    key = series.astype(str).str.strip().str.lower()
    out = key.map(mapping)
    changed = int((out.astype(str) != series.astype(str).str.strip()).sum())
    if out.isna().any():
        bad = series[out.isna()].unique()[:5]
        raise ValueError(f"Unknown category values: {list(bad)}")
    return out, changed


def _to_number(series: pd.Series) -> tuple[pd.Series, int]:
    """'3,85' -> 3.85 · '5 years' -> 5 · 'Band 6.5' -> 6.5 · 'Tier 2' -> 2 · '29.0' -> 29."""
    s = series.astype(str).str.strip()
    cleaned = s.str.replace(",", ".", regex=False).str.extract(r"(-?\d+(?:\.\d+)?)", expand=False)
    out = pd.to_numeric(cleaned, errors="coerce")
    changed = int((~s.str.fullmatch(r"-?\d+(\.\d+)?")).sum())
    return out, changed


def _year(value: str) -> int:
    m = re.search(r"(19|20)\d{2}", str(value))
    return int(m.group(0)) if m else np.nan


def clean(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    log = CleaningLog()
    df = raw.copy()
    log.add("Load raw export", "Rows read from the HR system export", len(df), len(df), "Load")

    # 1. remove rows with no ID and test records
    no_id = df["App ID"].isna() | df["App ID"].astype(str).str.strip().eq("")
    df = df[~no_id]
    log.add("Drop rows without ID", "Rows with an empty App ID cannot be linked to a person", no_id.sum(), len(df), "Validity")
    test = df["App ID"].astype(str).str.upper().str.startswith("TEST") | df["Full Name"].astype(str).str.lower().eq("test user")
    df = df[~test]
    log.add("Drop test records", "Records created while testing the HR system (TEST-xxx)", test.sum(), len(df), "Validity")

    # 2. trim spaces in every text column
    obj = df.select_dtypes(include=["object", "string"]).columns
    before = df[obj].copy()
    for c in obj:
        df[c] = df[c].str.strip()
    log.add("Trim whitespace", "Leading/trailing spaces removed from text cells",
            int((before.fillna("") != df[obj].fillna("")).any(axis=1).sum()), len(df), "Consistency")

    # 3. standardise categories (English / Arabic / abbreviations -> one label)
    for col, variants, label in [("Gender", GENDER_VARIANTS, "gender"), ("Nationality", NATIONALITY_VARIANTS, "nationality"),
                                 ("City/Region", REGION_VARIANTS, "region")]:
        df[col], n = _standardise(df[col], _lookup(variants))
        log.add(f"Standardise {label}", f"Variants such as Arabic labels and abbreviations mapped to one value", n, len(df),
                "Consistency")
    for col, values, label in [("Job Title", JOB_FAMILIES, "job title"), ("Major", DEGREES, "major")]:
        df[col], n = _standardise(df[col], {v.lower(): v for v in values})
        log.add(f"Standardise {label}", "Case differences fixed (e.g. 'data analyst' -> 'Data Analyst')", n, len(df),
                "Consistency")
    df["Internship"], n = _standardise(df["Internship"], {**{v.lower(): 1 for v in YES}, **{v.lower(): 0 for v in NO}})
    log.add("Standardise internship flag", "Yes/No/نعم/لا/1/0 converted to 1/0", n, len(df), "Consistency")
    df["Hired"], n = _standardise(df["Hired"], {**{v.lower(): 1 for v in HIRED}, **{v.lower(): 0 for v in NOT_HIRED}})
    log.add("Standardise hiring outcome", "Hired/Rejected/Yes/No converted to 1/0", n, len(df), "Consistency")

    # 4. remove duplicates (after standardising, so near-duplicates become exact duplicates)
    dup = df.duplicated(subset=[c for c in df.columns if c not in PII_COLUMNS])
    df = df[~dup]
    log.add("Remove duplicates", "Exact and near-duplicate rows (same applicant entered twice)", dup.sum(), len(df),
            "Uniqueness")
    dup_id = df["App ID"].duplicated()
    df = df[~dup_id]
    log.add("Check unique IDs", "Remaining rows sharing an App ID", dup_id.sum(), len(df), "Uniqueness")

    # 5. convert text to numbers
    for col, label in [("GPA", "GPA"), ("Experience", "experience"), ("English (IELTS)", "English score"),
                       ("Age", "age"), ("University Tier", "university tier")]:
        df[col], n = _to_number(df[col])
        log.add(f"Parse {label}", "Numbers stored as text converted ('3,85', '5 years', 'Band 6.5', 'Tier 2')", n,
                len(df), "Format")
    df["application_year"] = df["Application Date"].map(_year)
    non_iso = ~df["Application Date"].astype(str).str.fullmatch(r"\d{4}-\d{2}-\d{2}")
    log.add("Parse application date", "Three date formats (ISO, DD/MM/YYYY, 'March 5, 2024') reduced to the year",
            non_iso.sum(), len(df), "Format")

    # 6. fix data-entry errors
    typo = df["GPA"].between(10, 50)
    df.loc[typo, "GPA"] = (df.loc[typo, "GPA"] / 10).round(2)
    log.add("Fix GPA entry errors", "GPA typed without the decimal point (38.5 -> 3.85)", typo.sum(), len(df), "Validity")

    # 7. validate ranges (should all pass after the fixes above)
    rules = {"GPA": (0, 5), "Experience": (0, 45), "English (IELTS)": (0, 9), "Age": (18, 70), "University Tier": (1, 3)}
    bad = pd.Series(False, index=df.index)
    for col, (lo, hi) in rules.items():
        bad |= ~df[col].between(lo, hi)
    df = df[~bad]
    log.add("Validate ranges", "Rows with impossible values removed (GPA 0-5, age 18-70, ...)", bad.sum(), len(df), "Validity")

    # 8. anonymise: remove direct identifiers (PDPL data minimisation)
    df = df.drop(columns=PII_COLUMNS + ["Application Date"])
    log.add("Remove personal identifiers", "Name, national ID, mobile and e-mail dropped; App ID kept as a pseudonym",
            len(PII_COLUMNS), len(df), "Privacy")

    # 9. rename, derive, type
    df = df.rename(columns={v: k for k, v in RAW_COLUMNS.items() if k != "application_year"})
    df["region_group"] = df.region.map(lambda r: REGIONS[r][1])
    log.add("Derive region group", "Nine regions grouped into Central, Western, Eastern, Southern, Northern", len(df),
            len(df), "Enrichment")
    ints = ["application_year", "age", "university_tier", "years_experience", "certifications", "internship",
            "career_gap_months", "commute_km", "truly_qualified", "hired_historical"]
    df[ints] = df[ints].astype(int)
    df["qualification_score"] = df.qualification_score.astype(float)
    df["gpa"] = df.gpa.round(2)
    df["english_score"] = df.english_score.round(1)
    df = df[CLEAN_ORDER].sort_values("applicant_id").reset_index(drop=True)
    missing = int(df.isna().sum().sum())
    log.add("Final check", f"Clean table: {df.shape[1]} columns, {missing} missing values", 0, len(df), "Validity")
    return df, log.frame()


def quality_profile(raw: pd.DataFrame, clean_df: pd.DataFrame) -> pd.DataFrame:
    """Before/after summary used in the Excel workbook and the web app."""
    return pd.DataFrame([
        {"metric": "Rows", "raw": len(raw), "clean": len(clean_df)},
        {"metric": "Columns", "raw": raw.shape[1], "clean": clean_df.shape[1]},
        {"metric": "Missing cells", "raw": int(raw.isna().sum().sum()), "clean": int(clean_df.isna().sum().sum())},
        {"metric": "Duplicate rows", "raw": int(raw.drop(columns=PII_COLUMNS).duplicated().sum()),
         "clean": int(clean_df.duplicated().sum())},
        {"metric": "Distinct gender labels", "raw": raw["Gender"].nunique(), "clean": clean_df.gender.nunique()},
        {"metric": "Distinct region labels", "raw": raw["City/Region"].nunique(), "clean": clean_df.region.nunique()},
        {"metric": "Columns with personal identifiers", "raw": len(PII_COLUMNS), "clean": 0},
    ])




# ==================================================================================================
# 4. FAIRNESS METRICS
# ==================================================================================================
def selection_rate(y_pred: np.ndarray) -> float:
    y_pred = np.asarray(y_pred)
    return float(y_pred.mean()) if len(y_pred) else float("nan")


def true_positive_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    pos = y_true == 1
    return float(y_pred[pos].mean()) if pos.any() else float("nan")


def false_positive_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    neg = y_true == 0
    return float(y_pred[neg].mean()) if neg.any() else float("nan")


def disparate_impact_ratio(y_pred, groups) -> float:
    """Lowest group selection rate divided by the highest (four-fifths rule: >= 0.8)."""
    rates = pd.Series(np.asarray(y_pred)).groupby(np.asarray(groups)).mean()
    return float(rates.min() / rates.max()) if rates.max() > 0 else float("nan")


def statistical_parity_difference(y_pred, groups) -> float:
    """Highest minus lowest group selection rate (0 = parity)."""
    rates = pd.Series(np.asarray(y_pred)).groupby(np.asarray(groups)).mean()
    return float(rates.max() - rates.min())


def equal_opportunity_difference(y_true, y_pred, groups) -> float:
    """Gap in true-positive rate between groups (0 = qualified people treated alike)."""
    df = pd.DataFrame({"t": np.asarray(y_true), "p": np.asarray(y_pred), "g": np.asarray(groups)})
    tpr = df[df.t == 1].groupby("g").p.mean()
    return float(tpr.max() - tpr.min())


def group_table(y_true, y_pred, y_prob, groups, attribute: str) -> pd.DataFrame:
    """Per-group metrics for one sensitive attribute."""
    df = pd.DataFrame({"t": np.asarray(y_true), "p": np.asarray(y_pred), "s": np.asarray(y_prob), "g": np.asarray(groups)})
    rows = []
    best = df.groupby("g").p.mean().max()
    for g, d in df.groupby("g"):
        sr = selection_rate(d.p)
        rows.append({
            "attribute": attribute,
            "group": g,
            "n": len(d),
            "qualified_share": float(d.t.mean()),
            "selection_rate": sr,
            "impact_ratio_vs_best": sr / best if best else np.nan,
            "tpr": true_positive_rate(d.t, d.p),
            "fpr": false_positive_rate(d.t, d.p),
            "mean_score": float(d.s.mean()),
        })
    return pd.DataFrame(rows)


# ==================================================================================================
# 5. PROXY DETECTION
# ==================================================================================================
def proxy_scores(df: pd.DataFrame, features: list[str], sensitive: str, positive: str, seed: int = 42) -> pd.DataFrame:
    """For each numeric feature: mutual information and single-feature AUC for predicting `sensitive`.

    AUC 0.5 = no information about the group; above ~0.65 is a warning sign.
    """
    y = (df[sensitive] == positive).astype(int).to_numpy()
    X = df[features].astype(float)
    mi = mutual_info_classif(X, y, random_state=seed)
    rows = []
    for f, m in zip(features, mi):
        prob = cross_val_predict(
            make_pipeline(StandardScaler(), LogisticRegression()), X[[f]], y, cv=5, method="predict_proba"
        )[:, 1]
        auc = roc_auc_score(y, prob)
        rows.append({"sensitive_attribute": sensitive, "feature": f, "mutual_information": float(m),
                     "auc_predicting_group": float(max(auc, 1 - auc))})
    # all features together
    prob = cross_val_predict(make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)), X, y, cv=5,
                             method="predict_proba")[:, 1]
    rows.append({"sensitive_attribute": sensitive, "feature": "ALL FEATURES COMBINED",
                 "mutual_information": float("nan"), "auc_predicting_group": float(roc_auc_score(y, prob))})
    out = pd.DataFrame(rows)
    out["risk_level"] = pd.cut(out.auc_predicting_group, [0, 0.6, 0.7, 1.01], labels=["Low", "Medium", "High"]).astype(str)
    return out.sort_values("auc_predicting_group", ascending=False)


# ==================================================================================================
# 6. MODELS
# ==================================================================================================
JOB_NUMERIC = ["gpa", "years_experience", "english_score", "certifications", "internship", "university_tier"]
JOB_CATEGORICAL = ["degree_field", "job_family"]
PROXIES = ["career_gap_months", "commute_km"]
SENSITIVE = ["gender", "nationality", "region_group"]


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    model_name: str
    algorithm: str
    use_sensitive: bool
    use_proxies: bool
    reweigh: bool
    description: str

    @property
    def mitigation(self) -> str:
        parts = []
        if not self.use_sensitive:
            parts.append("Sensitive attributes removed")
        if not self.use_proxies:
            parts.append("Proxies removed")
        if self.reweigh:
            parts.append("Reweighing")
        return " + ".join(parts) or "None"


MODEL_SPECS = [
    ModelSpec("M1", "Black-box baseline", "Gradient Boosting", True, True, False,
              "High-capacity model trained on every column, including gender, nationality and region."),
    ModelSpec("M2", "Transparent baseline", "Logistic Regression", True, True, False,
              "Interpretable model trained on every column."),
    ModelSpec("M3", "Fairness through unawareness", "Logistic Regression", False, True, False,
              "Sensitive columns dropped, but proxy columns kept - shows that unawareness is not enough."),
    ModelSpec("M4", "Unaware + reweighing", "Logistic Regression", False, True, True,
              "Proxies kept, training rows reweighed so group and outcome are independent (Kamiran & Calders, 2012)."),
    ModelSpec("M5", "Job-relevant only + reweighing", "Logistic Regression", False, False, True,
              "Only job-relevant features, plus reweighing. Recommended model."),
]


def design_matrix(df: pd.DataFrame, spec: ModelSpec) -> pd.DataFrame:
    cols = JOB_NUMERIC.copy()
    cats = JOB_CATEGORICAL.copy()
    if spec.use_proxies:
        cols += PROXIES
    if spec.use_sensitive:
        cats += SENSITIVE
    X = pd.concat([df[cols].astype(float), pd.get_dummies(df[cats], prefix_sep="=", dtype=float)], axis=1)
    return X


def reweighing_weights(groups: pd.Series, y: pd.Series) -> np.ndarray:
    """Kamiran & Calders (2012): w(g, y) = P(g) * P(y) / P(g, y)."""
    df = pd.DataFrame({"g": groups.to_numpy(), "y": y.to_numpy()})
    p_g = df.g.value_counts(normalize=True)
    p_y = df.y.value_counts(normalize=True)
    p_gy = df.groupby(["g", "y"]).size() / len(df)
    return np.array([p_g[g] * p_y[v] / p_gy[(g, v)] for g, v in zip(df.g, df.y)])


def build_estimator(spec: ModelSpec, seed: int = 42):
    if spec.algorithm == "Gradient Boosting":
        return GradientBoostingClassifier(n_estimators=250, max_depth=3, learning_rate=0.05, random_state=seed)
    return Pipeline([("scale", StandardScaler()), ("clf", LogisticRegression(max_iter=2000))])


def fit(spec: ModelSpec, X: pd.DataFrame, y: pd.Series, groups: pd.Series, seed: int = 42):
    est = build_estimator(spec, seed)
    kw = {}
    if spec.reweigh:
        w = reweighing_weights(groups, y)
        kw = {"sample_weight": w} if spec.algorithm == "Gradient Boosting" else {"clf__sample_weight": w}
    est.fit(X, y, **kw)
    return est


def top_k_decisions(scores: np.ndarray, rate: float) -> np.ndarray:
    """Select the top `rate` share of applicants (a fixed number of open positions)."""
    k = int(round(rate * len(scores)))
    order = np.argsort(-scores, kind="stable")
    pred = np.zeros(len(scores), dtype=int)
    pred[order[:k]] = 1
    return pred


# ==================================================================================================
# 7. EXCEL WORKBOOKS
# ==================================================================================================
FONT = "Arial"
NAVY, SOFT, GOOD, BAD = "1F3864", "EDF2FA", "C6EFCE", "FFC7CE"
HEAD = Font(name=FONT, bold=True, color="FFFFFF", size=10)
BODY = Font(name=FONT, size=10)
BOLD = Font(name=FONT, size=10, bold=True)
TITLE = Font(name=FONT, size=16, bold=True, color=NAVY)
SUB = Font(name=FONT, size=10, italic=True, color="666666")
INPUT = Font(name=FONT, size=10, color="0000FF")
LINK = Font(name=FONT, size=10, color="008000")
FILL_HEAD = PatternFill("solid", fgColor=NAVY)
FILL_SOFT = PatternFill("solid", fgColor=SOFT)
FILL_INPUT = PatternFill("solid", fgColor="FFF2CC")
THIN = Border(*(Side(style="thin", color="D0D7E2"),) * 4)
GREEN_RULE = dict(fill=PatternFill("solid", fgColor=GOOD), font=Font(name=FONT, color="006100"))
RED_RULE = dict(fill=PatternFill("solid", fgColor=BAD), font=Font(name=FONT, color="9C0006"))

FORMATS = {"rate": "0.0%", "ratio": "0.00", "score": "0.000", "int": "#,##0", "pp": "0.0"}


def _write_table(ws, df: pd.DataFrame, name: str, start_row: int = 1, formats: dict | None = None,
                 widths: dict | None = None, style: str = "TableStyleMedium2") -> None:
    formats = formats or {}
    for j, col in enumerate(df.columns, 1):
        c = ws.cell(start_row, j, col)
        c.font, c.fill, c.alignment = HEAD, FILL_HEAD, Alignment(vertical="center", wrap_text=True)
    for i, row in enumerate(df.itertuples(index=False), start_row + 1):
        for j, v in enumerate(row, 1):
            if pd.isna(v):
                v = None
            c = ws.cell(i, j, v.item() if hasattr(v, "item") else v)
            c.font = BODY
            if df.columns[j - 1] in formats:
                c.number_format = formats[df.columns[j - 1]]
    end = f"{get_column_letter(df.shape[1])}{start_row + len(df)}"
    tab = Table(displayName=name, ref=f"A{start_row}:{end}")
    tab.tableStyleInfo = TableStyleInfo(name=style, showRowStripes=True)
    ws.add_table(tab)
    ws.freeze_panes = ws.cell(start_row + 1, 1)
    for j, col in enumerate(df.columns, 1):
        width = (widths or {}).get(col) or min(max(len(str(col)), *(len(str(x)) for x in df.iloc[:200, j - 1])) + 3, 45)
        ws.column_dimensions[get_column_letter(j)].width = width


def _title(ws, title: str, subtitle: str) -> None:
    ws["A1"], ws["A2"] = title, subtitle
    ws["A1"].font, ws["A2"].font = TITLE, SUB


def _chart_basics(ch, title: str, y_title: str):
    ch.title, ch.y_axis.title = title, y_title
    ch.roundedCorners = False
    ch.legend.overlay = False
    ch.x_axis.delete = False
    ch.y_axis.delete = False
    ch.height, ch.width = 7.5, 14
    return ch


# --------------------------------------------------------------------------------------- data workbook
def data_workbook(raw: pd.DataFrame, clean_df: pd.DataFrame, log: pd.DataFrame, profile: pd.DataFrame, path: Path) -> None:
    wb = Workbook()
    ws = wb.active
    ws.title = "About"
    _title(ws, "Saudi Fair Hiring Audit – Data workbook", "CS 476 Ethics in Data Science · all records are synthetic")
    rows = [
        ("Sheet", "Contents"),
        ("Raw_Export_Sample", "First 300 rows of the messy HR export (fictional names, IDs and contacts)"),
        ("Cleaning_Log", "Every cleaning step with rows affected and rows remaining"),
        ("Data_Quality", "Before / after comparison of the raw export and the clean table"),
        ("Clean_Data", f"The clean, anonymised table used for modelling ({len(clean_df):,} applicants)"),
        ("Fairness_Calculator", "Four-fifths rule audit built from live Excel formulas on Clean_Data"),
        ("", ""),
        ("Legend", ""),
        ("Blue text, yellow fill", "Input you can change (threshold)"),
        ("Black text", "Formula"),
        ("Green / red fill", "Passes / fails the four-fifths rule"),
    ]
    for i, (a, b) in enumerate(rows, 4):
        ws.cell(i, 1, a).font = HEAD if i == 4 else (BOLD if a else BODY)
        ws.cell(i, 2, b).font = HEAD if i == 4 else BODY
        if i == 4:
            ws.cell(i, 1).fill = ws.cell(i, 2).fill = FILL_HEAD
    ws.cell(12, 1).font = INPUT
    ws.cell(12, 1).fill = FILL_INPUT
    ws.column_dimensions["A"].width, ws.column_dimensions["B"].width = 26, 90

    ws = wb.create_sheet("Raw_Export_Sample")
    _write_table(ws, raw.head(300).astype(object), "RawSample", style="TableStyleLight9")
    ws = wb.create_sheet("Cleaning_Log")
    _write_table(ws, log, "CleaningLog", formats={"rows_affected": FORMATS["int"], "rows_after": FORMATS["int"]},
                 widths={"description": 80})
    ws = wb.create_sheet("Data_Quality")
    _write_table(ws, profile, "DataQuality", formats={"raw": FORMATS["int"], "clean": FORMATS["int"]},
                 widths={"metric": 36, "raw": 12, "clean": 12})
    ws = wb.create_sheet("Clean_Data")
    _write_table(ws, clean_df, "CleanData", formats={"gpa": "0.00", "english_score": "0.0", "qualification_score": "0.0"})

    _fairness_calculator(wb.create_sheet("Fairness_Calculator"), len(clean_df))
    path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(path)


def _fairness_calculator(ws, n: int) -> None:
    col = {c: get_column_letter(i) for i, c in enumerate(CLEAN_ORDER, 1)}
    rng = lambda c: f"Clean_Data!${col[c]}$2:${col[c]}${n + 1}"  # noqa: E731
    _title(ws, "Fairness Calculator – historical hiring decisions",
           "Every number below is an Excel formula on the Clean_Data sheet. Change the threshold to test other rules.")
    ws["A4"], ws["B4"] = "Four-fifths threshold", 0.8
    ws["A4"].font, ws["B4"].font, ws["B4"].fill, ws["B4"].number_format = BOLD, INPUT, FILL_INPUT, "0.00"
    ws["C4"] = "Input: impact ratio below this value = adverse impact (U.S. EEOC four-fifths rule)"
    ws["C4"].font = SUB

    def header(row, labels):
        for j, lab in enumerate(labels, 1):
            c = ws.cell(row, j, lab)
            c.font, c.fill, c.alignment = HEAD, FILL_HEAD, Alignment(wrap_text=True, vertical="center")

    def block(top: int, title: str, attr: str, groups: list[str]) -> int:
        ws.cell(top, 1, title).font = Font(name=FONT, size=12, bold=True, color=NAVY)
        header(top + 1, ["Group", "Applicants", "Hired", "Hire rate", "Impact ratio", "Verdict",
                         "Qualified applicants", "Qualified hired", "Hire rate (qualified)"])
        first, last = top + 2, top + 1 + len(groups)
        for i, g in enumerate(groups):
            r = first + i
            ws.cell(r, 1, g).font = BOLD
            ws.cell(r, 2, f'=COUNTIFS({rng(attr)},$A{r})')
            ws.cell(r, 3, f'=SUMIFS({rng("hired_historical")},{rng(attr)},$A{r})')
            ws.cell(r, 4, f"=IFERROR(C{r}/B{r},0)")
            ws.cell(r, 5, f"=IFERROR(D{r}/MAX($D${first}:$D${last}),0)")
            ws.cell(r, 6, f'=IF(E{r}<$B$4,"ADVERSE IMPACT","OK")')
            ws.cell(r, 7, f'=COUNTIFS({rng(attr)},$A{r},{rng("truly_qualified")},1)')
            ws.cell(r, 8, f'=SUMIFS({rng("hired_historical")},{rng(attr)},$A{r},{rng("truly_qualified")},1)')
            ws.cell(r, 9, f"=IFERROR(H{r}/G{r},0)")
            for j, fmt in zip(range(2, 10), ["#,##0", "#,##0", "0.0%", "0.00", None, "#,##0", "#,##0", "0.0%"]):
                c = ws.cell(r, j)
                c.font, c.border = BODY, THIN
                if fmt:
                    c.number_format = fmt
        ws.conditional_formatting.add(f"E{first}:E{last}", CellIsRule(operator="lessThan", formula=["$B$4"], **RED_RULE))
        ws.conditional_formatting.add(f"E{first}:E{last}", CellIsRule(operator="greaterThanOrEqual", formula=["$B$4"], **GREEN_RULE))
        return last

    g_last = block(6, "1. By gender", "gender", ["Male", "Female"])
    ws.cell(g_last + 1, 1, "Gap among equally qualified (percentage points)").font = BOLD
    ws.cell(g_last + 1, 9, f"=(I{g_last - 1}-I{g_last})*100").number_format = "0.0"
    ws.cell(g_last + 1, 9).font = BOLD
    r_top = g_last + 3
    r_last = block(r_top, "2. By region group", "region_group", ["Central", "Western", "Eastern", "Southern", "Northern"])
    n_top = r_last + 2
    n_last = block(n_top, "3. By nationality (reported only: Saudization is a lawful policy)", "nationality",
                   ["Saudi", "Non-Saudi"])
    note = n_last + 2
    ws.cell(note, 1, "How to read: hire rate = hired / applicants. Impact ratio = group hire rate / best group hire rate. "
                     "'Hire rate (qualified)' compares only truly qualified applicants, so any gap there is bias.").font = SUB
    for j, w in enumerate([24, 12, 10, 11, 12, 17, 14, 13, 14], 1):
        ws.column_dimensions[get_column_letter(j)].width = w

    for (top, last, title), anchor in zip([(8, g_last, "Hire rate by gender"),
                                           (r_top + 2, r_last, "Hire rate by region group")], ["K4", "K21"]):
        ch = _chart_basics(BarChart(), title, "Hire rate")
        ch.add_data(Reference(ws, min_col=4, min_row=top - 1, max_row=last), titles_from_data=True)
        ch.add_data(Reference(ws, min_col=9, min_row=top - 1, max_row=last), titles_from_data=True)
        ch.set_categories(Reference(ws, min_col=1, min_row=top, max_row=last))
        ch.y_axis.number_format = "0%"
        ch.y_axis.scaling.min = 0
        ws.add_chart(ch, anchor)


# --------------------------------------------------------------------------------------- results workbook
def results_workbook(tables: dict[str, pd.DataFrame], clean_df: pd.DataFrame, path: Path) -> None:
    wb = Workbook()
    ws = wb.active
    ws.title = "About"
    _title(ws, "Saudi Fair Hiring Audit – Results workbook", "Test set: 3,000 applicants · every model shortlists 35.4%")
    for i, (a, b) in enumerate([
        ("Sheet", "Contents"),
        ("Model_Scorecard", "Accuracy and fairness for M1–M5, with PASS/FAIL against the four-fifths rule"),
        ("Group_Metrics", "Selection rate, impact ratio and recall of qualified applicants per group"),
        ("Proxy_Scores", "How well each feature reveals gender or region (AUC; > 0.7 = high risk)"),
        ("Feature_Importance", "What drives each model's score"),
        ("M5_Predictions", "Scores and decisions of the recommended model M5 for every test applicant"),
    ], 4):
        ws.cell(i, 1, a).font = HEAD if i == 4 else BOLD
        ws.cell(i, 2, b).font = HEAD if i == 4 else BODY
        if i == 4:
            ws.cell(i, 1).fill = ws.cell(i, 2).fill = FILL_HEAD
    ws.column_dimensions["A"].width, ws.column_dimensions["B"].width = 22, 90

    sc = tables["models"].merge(tables["model_metrics"], on="model_id")
    sc = sc[["model_id", "model_name", "algorithm", "mitigation", "auc_vs_history", "accuracy_vs_truth", "recall_qualified",
             "di_gender", "di_region_group", "di_nationality", "eod_gender", "eod_region_group"]]
    ws = wb.create_sheet("Model_Scorecard")
    fm = {c: FORMATS["score"] for c in ["auc_vs_history", "accuracy_vs_truth", "recall_qualified", "eod_gender", "eod_region_group"]}
    fm.update({c: FORMATS["ratio"] for c in ["di_gender", "di_region_group", "di_nationality"]})
    _write_table(ws, sc, "Scorecard", formats=fm, widths={"mitigation": 48, "model_name": 30})
    last = len(sc) + 1
    ws.cell(1, sc.shape[1] + 1, "four_fifths_verdict").font = HEAD
    ws.cell(1, sc.shape[1] + 1).fill = FILL_HEAD
    for r in range(2, last + 1):
        ws.cell(r, sc.shape[1] + 1, f'=IF(AND(H{r}>=0.8,I{r}>=0.8),"PASS","FAIL")').font = BOLD
    ws.column_dimensions[get_column_letter(sc.shape[1] + 1)].width = 18
    ws.cell(last + 2, 1, "Verdict uses gender and region impact ratios (>= 0.80). Nationality is reported only.").font = SUB
    for rng in [f"H2:I{last}"]:
        ws.conditional_formatting.add(rng, CellIsRule(operator="lessThan", formula=["0.8"], **RED_RULE))
        ws.conditional_formatting.add(rng, CellIsRule(operator="greaterThanOrEqual", formula=["0.8"], **GREEN_RULE))
    vcol = get_column_letter(sc.shape[1] + 1)
    ws.conditional_formatting.add(f"{vcol}2:{vcol}{last}", CellIsRule(operator="equal", formula=['"PASS"'], **GREEN_RULE))
    ws.conditional_formatting.add(f"{vcol}2:{vcol}{last}", CellIsRule(operator="equal", formula=['"FAIL"'], **RED_RULE))
    ch = _chart_basics(BarChart(), "Impact ratio by model (>= 0.80 passes)", "Impact ratio")
    ch.add_data(Reference(ws, min_col=8, max_col=9, min_row=1, max_row=last), titles_from_data=True)
    ch.set_categories(Reference(ws, min_col=1, min_row=2, max_row=last))
    ch.y_axis.scaling.min, ch.y_axis.scaling.max = 0.5, 1.0
    ws.add_chart(ch, f"B{last + 4}")

    ws = wb.create_sheet("Group_Metrics")
    gm = tables["group_metrics"]
    _write_table(ws, gm, "GroupMetrics", formats={"qualified_share": "0.0%", "selection_rate": "0.0%",
                                                  "impact_ratio_vs_best": "0.00", "tpr": "0.0%", "fpr": "0.0%", "mean_score": "0.000"})
    rng = f"G2:G{len(gm) + 1}"
    ws.conditional_formatting.add(rng, CellIsRule(operator="lessThan", formula=["0.8"], **RED_RULE))

    ws = wb.create_sheet("Proxy_Scores")
    px = tables["proxy_scores"]
    _write_table(ws, px, "ProxyScores", formats={"mutual_information": "0.000", "auc_predicting_group": "0.000"})
    ws.conditional_formatting.add(f"D2:D{len(px) + 1}", CellIsRule(operator="greaterThan", formula=["0.7"], **RED_RULE))

    ws = wb.create_sheet("Feature_Importance")
    _write_table(ws, tables["feature_importance"].sort_values(["model_id", "importance"], ascending=[True, False]),
                 "FeatureImportance", formats={"importance": "0.000"})

    ws = wb.create_sheet("M5_Predictions")
    pr = tables["predictions"].query("model_id == 'M5'").merge(
        clean_df[["applicant_id", "job_family", "gender", "nationality", "region", "qualification_score",
                  "truly_qualified", "hired_historical"]], on="applicant_id")
    pr = pr.sort_values("score", ascending=False)
    _write_table(ws, pr, "M5Predictions", formats={"score": "0.000", "qualification_score": "0.0"})

    path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(path)


# ==================================================================================================
# 8. PIPELINE
# ==================================================================================================


def log(msg: str) -> None:
    print(f"[fairhire] {msg}")


def sql_part(name: str) -> str:
    """Return one part (SCHEMA, VIEWS or AUDIT) of database.sql."""
    text = SQL_FILE.read_text(encoding="utf-8")
    parts = dict(re.findall(r"^-- @@ (\w+)\n(.*?)(?=^-- @@ |\Z)", text, flags=re.M | re.S))
    return parts[name]


def run_named_queries(con: sqlite3.Connection, text: str, out_dir: Path) -> dict[str, pd.DataFrame]:
    blocks = re.split(r"^-- name:\s*(\S+)\s*$", text, flags=re.M)[1:]
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for name, sql in zip(blocks[0::2], blocks[1::2]):
        df = pd.read_sql_query(sql.strip().rstrip(";"), con)
        df.to_csv(out_dir / f"{name}.csv", index=False)
        results[name] = df
    return results


def main(n: int = 10000, regenerate: bool = False) -> None:
    OUT.mkdir(exist_ok=True)
    data = out = pbi = OUT
    raw_path = OUT / "hr_export_raw.csv"
    clean_path = OUT / "applicants_clean.csv"

    # 1. Data: messy HR export -> cleaning -> clean table --------------------------
    if regenerate or not raw_path.exists():
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        make_raw_export(generate(n, SEED)).to_csv(raw_path, index=False, encoding="utf-8-sig")
        log(f"generated messy HR export {raw_path.relative_to(ROOT)}")
    raw_df = pd.read_csv(raw_path, dtype=str, encoding="utf-8-sig")
    df, cleaning_log = clean(raw_df)
    profile = quality_profile(raw_df, df)
    clean_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(clean_path, index=False)
    cleaning_log.to_csv(data / "data_quality_log.csv", index=False)
    log(f"cleaned {len(raw_df):,} raw rows -> {len(df):,} applicants ({len(cleaning_log)} steps) -> "
        f"{clean_path.relative_to(ROOT)}")

    # 2. Database ----------------------------------------------------------------
    db_path = out / "fairhire.db"
    con = sqlite3.connect(db_path)
    con.executescript(sql_part("SCHEMA"))
    df.to_sql("applicants", con, if_exists="append", index=False)
    cleaning_log.to_sql("data_quality_log", con, if_exists="append", index=False)
    profile.to_sql("data_quality_profile", con, if_exists="append", index=False)

    # 3. Train / test split ------------------------------------------------------
    strat = df.gender + "|" + df.region_group + "|" + df.hired_historical.astype(str)
    train, test = train_test_split(df, test_size=0.3, random_state=SEED, stratify=strat)
    df["split"] = np.where(df.applicant_id.isin(test.applicant_id), "test", "train")
    hire_rate = train.hired_historical.mean()
    fairness_group = train.gender + "|" + train.region_group  # groups used for reweighing

    model_rows, metric_rows, group_rows, pred_rows, imp_rows = [], [], [], [], []
    for spec in MODEL_SPECS:
        Xtr, Xte = design_matrix(train, spec), design_matrix(test, spec)
        Xte = Xte.reindex(columns=Xtr.columns, fill_value=0.0)
        est = fit(spec, Xtr, train.hired_historical, fairness_group, SEED)
        prob = est.predict_proba(Xte)[:, 1]
        pred = top_k_decisions(prob, hire_rate)

        y_hist, y_true = test.hired_historical.to_numpy(), test.truly_qualified.to_numpy()
        di = {a: disparate_impact_ratio(pred, test[a]) for a in ["gender", "nationality", "region_group"]}
        metric_rows.append({
            "model_id": spec.model_id,
            "accuracy_vs_history": accuracy_score(y_hist, pred),
            "f1_vs_history": f1_score(y_hist, pred),
            "auc_vs_history": roc_auc_score(y_hist, prob),
            "accuracy_vs_truth": accuracy_score(y_true, pred),
            "recall_qualified": true_positive_rate(y_true, pred),
            "selection_rate": selection_rate(pred),
            "di_gender": di["gender"],
            "di_nationality": di["nationality"],
            "di_region_group": di["region_group"],
            "eod_gender": equal_opportunity_difference(y_true, pred, test.gender),
            "eod_region_group": equal_opportunity_difference(y_true, pred, test.region_group),
            # nationality is excluded from the pass/fail rule: Saudization is a lawful policy (see docs)
            "passes_four_fifths": int(di["gender"] >= 0.8 and di["region_group"] >= 0.8),
        })
        model_rows.append({"model_id": spec.model_id, "model_name": spec.model_name, "algorithm": spec.algorithm,
                           "uses_sensitive": int(spec.use_sensitive), "uses_proxies": int(spec.use_proxies),
                           "mitigation": spec.mitigation, "description": spec.description})
        for a in ["gender", "nationality", "region_group"]:
            g = group_table(y_true, pred, prob, test[a], a).rename(columns={"group": "group_name"})
            g.insert(0, "model_id", spec.model_id)
            group_rows.append(g)
        pred_rows.append(pd.DataFrame({"applicant_id": test.applicant_id.to_numpy(), "model_id": spec.model_id,
                                       "score": prob.round(5), "predicted_hire": pred}))

        # Explainability
        if spec.algorithm == "Gradient Boosting":
            pi = permutation_importance(est, Xte, y_hist, scoring="roc_auc", n_repeats=5, random_state=SEED)
            imp = pd.DataFrame({"feature": Xte.columns, "importance": pi.importances_mean, "direction": "n/a"})
        else:
            coef = est.named_steps["clf"].coef_[0]
            imp = pd.DataFrame({"feature": Xtr.columns, "importance": np.abs(coef),
                                "direction": np.where(coef >= 0, "raises score", "lowers score")})
        imp.insert(0, "model_id", spec.model_id)
        imp_rows.append(imp)
        if spec.model_id == RECOMMENDED_MODEL:  # saved for the web app / AI assistant
            joblib.dump({"model_id": spec.model_id, "estimator": est, "columns": list(Xtr.columns),
                         "cutoff": float(np.sort(prob)[::-1][int(round(hire_rate * len(prob))) - 1]),
                         "test_scores": np.sort(prob)}, out / "model_m5.joblib")
        m = metric_rows[-1]
        log(f"{spec.model_id} {spec.model_name:<32} AUC={m['auc_vs_history']:.3f} "
            f"DI(gender)={m['di_gender']:.2f} DI(region)={m['di_region_group']:.2f} "
            f"recall(qualified)={m['recall_qualified']:.2f}")

    tables = {
        "models": pd.DataFrame(model_rows),
        "predictions": pd.concat(pred_rows),
        "model_metrics": pd.DataFrame(metric_rows),
        "group_metrics": pd.concat(group_rows),
        "feature_importance": pd.concat(imp_rows),
    }

    # 4. Proxy detection ---------------------------------------------------------
    numeric = ["gpa", "years_experience", "english_score", "certifications", "internship",
               "university_tier", "age"] + PROXIES
    region_flag = df.assign(far_region=np.where(df.region_group.isin(["Southern", "Northern"]), "Yes", "No"))
    tables["proxy_scores"] = pd.concat([
        proxy_scores(df, numeric, "gender", "Female", SEED),
        proxy_scores(region_flag, numeric, "far_region", "Yes", SEED).assign(sensitive_attribute="region_group"),
    ])

    for name, t in tables.items():
        t.to_sql(name, con, if_exists="append", index=False)
    con.executescript(sql_part("VIEWS"))
    con.commit()

    # 5. SQL audit ---------------------------------------------------------------
    res = run_named_queries(con, sql_part("AUDIT"), data)
    log(f"ran {len(res)} SQL audit queries -> output/q*.csv")
    con.close()

    # 6. Power BI exports (star schema) -----------------------------------------
    df.to_csv(pbi / "dim_applicant.csv", index=False)
    tables["models"].to_csv(pbi / "dim_model.csv", index=False)
    tables["predictions"].to_csv(pbi / "fact_prediction.csv", index=False)
    tables["model_metrics"].round(4).to_csv(pbi / "fact_model_metrics.csv", index=False)
    tables["group_metrics"].round(4).to_csv(pbi / "fact_group_metrics.csv", index=False)
    tables["proxy_scores"].round(4).to_csv(pbi / "fact_proxy_scores.csv", index=False)
    tables["feature_importance"].round(4).to_csv(pbi / "fact_feature_importance.csv", index=False)
    log("Power BI files written -> output/dim_*.csv, output/fact_*.csv")

    # 7. Excel workbooks ---------------------------------------------------------
    data_workbook(raw_df, df.drop(columns="split"), cleaning_log, profile, data / "FairHire_Data.xlsx")
    results_workbook(tables, df, data / "FairHire_Results.xlsx")
    log("Excel workbooks written -> output/FairHire_*.xlsx")
    log(f"database -> {db_path.relative_to(ROOT)}")
    log("done. Start the web app with:  uvicorn app:app --reload")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=10000, help="number of synthetic applicants")
    ap.add_argument("--regenerate", action="store_true", help="re-create the raw data file")
    a = ap.parse_args()
    main(a.n, a.regenerate)
