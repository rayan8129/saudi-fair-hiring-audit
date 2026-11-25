"""Saudi Fair Hiring Audit - web app, REST API and AI assistant (one file).

Start:  uvicorn app:app --reload      then open http://127.0.0.1:8000   (API docs: /docs)
The first start runs pipeline.py automatically if the database does not exist yet.

AI assistant: set the ANTHROPIC_API_KEY environment variable to run it on Claude (tool use).
Without a key it runs in offline mode (rule-based, Arabic + English) with the same tools.
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
from contextlib import asynccontextmanager, contextmanager
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "output"


# ==================================================================================================
# 1. DATABASE ACCESS (SQLite, safe read-only SQL)
# ==================================================================================================
DB_PATH = OUT / "fairhire.db"
MODEL_PATH = OUT / "model_m5.joblib"
SQL_FILE = ROOT / "database.sql"
MAX_ROWS = 200


def ensure_built() -> None:
    """Build the database and the model on first start (runs the full pipeline, ~10 s)."""
    if DB_PATH.exists() and MODEL_PATH.exists():
        return
    from pipeline import main

    main()


@contextmanager
def connect(readonly: bool = True):
    uri = f"file:{DB_PATH.as_posix()}?mode={'ro' if readonly else 'rw'}"
    con = sqlite3.connect(uri, uri=True, check_same_thread=False)
    con.row_factory = sqlite3.Row
    try:
        yield con
        if not readonly:
            con.commit()
    finally:
        con.close()


def query(sql: str, params: tuple | dict = ()) -> list[dict]:
    with connect() as con:
        return [dict(r) for r in con.execute(sql, params).fetchall()]


def execute_write(sql: str, params: tuple) -> int:
    with connect(readonly=False) as con:
        cur = con.execute(sql, params)
        return cur.lastrowid


# ---------------------------------------------------------------- safe SQL for the AI assistant
_FORBIDDEN = re.compile(r"\b(insert|update|delete|drop|alter|create|replace|attach|detach|pragma|vacuum|reindex)\b", re.I)
_ALLOWED_ACTIONS = {sqlite3.SQLITE_READ, sqlite3.SQLITE_SELECT, sqlite3.SQLITE_FUNCTION}
_RECURSIVE = getattr(sqlite3, "SQLITE_RECURSIVE", None)
if _RECURSIVE is not None:
    _ALLOWED_ACTIONS.add(_RECURSIVE)


class UnsafeQueryError(ValueError):
    pass


def safe_select(sql: str, max_rows: int = MAX_ROWS) -> dict:
    """Run one read-only SELECT. Three layers of protection:
    1. text checks (single statement, must start with SELECT/WITH, no write keywords)
    2. the connection is opened read-only (mode=ro)
    3. an SQLite authorizer that denies every action except reading.
    """
    s = re.sub(r"^\s*(--[^\n]*(\n|$)\s*)+", "", sql).strip().rstrip(";").strip()  # drop leading comments
    if ";" in s:
        raise UnsafeQueryError("Only one statement is allowed.")
    if not re.match(r"^(select|with)\b", s, re.I):
        raise UnsafeQueryError("Only SELECT queries are allowed.")
    if _FORBIDDEN.search(s):
        raise UnsafeQueryError("Write or schema-changing keywords are not allowed.")
    with connect() as con:
        con.set_authorizer(lambda action, *_: sqlite3.SQLITE_OK if action in _ALLOWED_ACTIONS else sqlite3.SQLITE_DENY)
        cur = con.execute(s)
        cols = [d[0] for d in cur.description]
        rows = cur.fetchmany(max_rows + 1)
    return {"columns": cols, "rows": [list(r) for r in rows[:max_rows]], "truncated": len(rows) > max_rows}


def schema_description() -> str:
    lines = []
    with connect() as con:
        objs = con.execute("SELECT name, type FROM sqlite_master WHERE type IN ('table','view') "
                           "AND name NOT LIKE 'sqlite_%' ORDER BY type, name").fetchall()
        for name, typ in objs:
            cols = [r[1] for r in con.execute(f'PRAGMA table_info("{name}")')]
            lines.append(f"{typ} {name}({', '.join(cols)})")
    return "\n".join(lines)


def audit_queries() -> dict[str, str]:
    text = SQL_FILE.read_text(encoding="utf-8").split("-- @@ AUDIT", 1)[1]
    parts = re.split(r"^-- name:\s*(\S+)\s*$", text, flags=re.M)[1:]
    return {n: q.strip().rstrip(";") for n, q in zip(parts[0::2], parts[1::2])}


# ==================================================================================================
# 2. SCORING AND EXPLANATION (model M5)
# ==================================================================================================
DEGREES = ["Computer Science", "Information Systems", "Engineering", "Statistics", "Business Administration"]
JOB_FAMILIES = ["Data Analyst", "Software Engineer", "Cybersecurity Specialist", "Business Analyst", "HR Specialist"]

LABELS = {
    "gpa": "GPA", "years_experience": "Years of experience", "english_score": "English score",
    "certifications": "Certifications", "internship": "Internship", "university_tier": "University tier (1 = top)",
}


class ApplicantInput(BaseModel):
    """Only job-relevant information is collected (PDPL data minimisation)."""
    gpa: float = Field(..., ge=0, le=5, description="GPA on the Saudi 5-point scale")
    years_experience: int = Field(..., ge=0, le=40)
    english_score: float = Field(..., ge=0, le=9, description="IELTS-like band 0-9")
    certifications: int = Field(0, ge=0, le=20)
    internship: int = Field(0, ge=0, le=1)
    university_tier: int = Field(2, ge=1, le=3)
    degree_field: str = Field("Computer Science")
    job_family: str = Field("Data Analyst")


@lru_cache(maxsize=1)
def load_model() -> dict:
    return joblib.load(MODEL_PATH)


def _row(inp: dict, columns: list[str]) -> pd.DataFrame:
    x = pd.DataFrame([[0.0] * len(columns)], columns=columns)
    for k in ["gpa", "years_experience", "english_score", "certifications", "internship", "university_tier"]:
        x.loc[0, k] = float(inp[k])
    for prefix, val in [("degree_field", inp["degree_field"]), ("job_family", inp["job_family"])]:
        col = f"{prefix}={val}"
        if col in x.columns:
            x.loc[0, col] = 1.0
    return x


def explain_row(x: pd.DataFrame) -> list[dict]:
    """Contribution of each feature to the log-odds = coefficient x standardised value."""
    m = load_model()
    est = m["estimator"]
    scaler, clf = est.named_steps["scale"], est.named_steps["clf"]
    z = (x.to_numpy()[0] - scaler.mean_) / scaler.scale_
    contrib = clf.coef_[0] * z
    out = []
    for col, c, raw in zip(x.columns, contrib, x.to_numpy()[0]):
        if "=" in col and raw == 0:
            continue
        out.append({"feature": col, "label": LABELS.get(col, col.replace("=", ": ")),
                    "value": float(raw), "contribution": round(float(c), 3),
                    "effect": "raises score" if c >= 0 else "lowers score"})
    return sorted(out, key=lambda d: -abs(d["contribution"]))


def score(inp: ApplicantInput | dict, log: bool = True) -> dict:
    data = inp.model_dump() if isinstance(inp, BaseModel) else dict(ApplicantInput(**inp).model_dump())
    m = load_model()
    x = _row(data, m["columns"])
    s = float(m["estimator"].predict_proba(x)[0, 1])
    pct = float(np.searchsorted(m["test_scores"], s) / len(m["test_scores"]) * 100)
    shortlisted = s >= m["cutoff"]
    result = {
        "model_id": m["model_id"],
        "score": round(s, 4),
        "percentile": round(pct, 1),
        "cutoff": round(m["cutoff"], 4),
        "decision": "Shortlist" if shortlisted else "Not shortlisted",
        "note": "Recommendation only. A human recruiter makes the final decision.",
        "factors": explain_row(x)[:6],
    }
    if log:
        result["request_id"] = execute_write(
            "INSERT INTO scoring_requests (model_id, inputs_json, score, decision, percentile) VALUES (?,?,?,?,?)",
            (m["model_id"], json.dumps(data), s, result["decision"], pct),
        )
    return result


def explain_applicant(applicant_id: str, model_id: str = "M5") -> dict:
    rows = query("SELECT * FROM applicants WHERE applicant_id = ?", (applicant_id,))
    if not rows:
        return {"error": f"Applicant {applicant_id} not found."}
    a = rows[0]
    preds = query("SELECT p.model_id, m.model_name, p.score, p.predicted_hire FROM predictions p "
                  "JOIN models m USING(model_id) WHERE applicant_id = ? ORDER BY p.model_id", (applicant_id,))
    res = {"applicant": a, "predictions": preds, "in_test_set": bool(preds)}
    m = load_model()
    res["m5_explanation"] = explain_row(_row(a, m["columns"]))[:6]
    res["m5_score"] = round(float(m["estimator"].predict_proba(_row(a, m["columns"]))[0, 1]), 4)
    return res


# ==================================================================================================
# 3. AI ASSISTANT - TOOLS (all backed by SQL)
# ==================================================================================================
ATTRIBUTES = ("gender", "nationality", "region_group")


def get_fairness_summary() -> dict:
    hist = query("""
        SELECT COUNT(*) AS applicants, AVG(hired_historical) AS hire_rate,
               AVG(CASE WHEN gender='Female' THEN hired_historical END) AS female_rate,
               AVG(CASE WHEN gender='Male' THEN hired_historical END) AS male_rate,
               AVG(CASE WHEN gender='Female' AND truly_qualified=1 THEN hired_historical END) AS female_qualified_rate,
               AVG(CASE WHEN gender='Male' AND truly_qualified=1 THEN hired_historical END) AS male_qualified_rate
        FROM applicants""")[0]
    hist["gender_impact_ratio"] = min(hist["female_rate"], hist["male_rate"]) / max(hist["female_rate"], hist["male_rate"])
    hist["qualified_gap_pp"] = (hist["male_qualified_rate"] - hist["female_qualified_rate"]) * 100
    return {"historical_decisions": {k: round(v, 4) if isinstance(v, float) else v for k, v in hist.items()},
            "model_scorecard": query("SELECT * FROM vw_fairness_scorecard ORDER BY model_id"),
            "recommended_model": "M5"}


def compare_models() -> list[dict]:
    return query("""
        SELECT m.model_id, m.model_name, m.algorithm, m.mitigation,
               ROUND(mm.auc_vs_history,3) AS auc, ROUND(mm.accuracy_vs_truth,3) AS accuracy_vs_truth,
               ROUND(mm.recall_qualified,3) AS recall_qualified,
               ROUND(mm.di_gender,3) AS di_gender, ROUND(mm.di_region_group,3) AS di_region,
               ROUND(mm.di_nationality,3) AS di_nationality,
               ROUND(mm.eod_gender,3) AS eod_gender, ROUND(mm.eod_region_group,3) AS eod_region,
               CASE WHEN mm.passes_four_fifths=1 THEN 'PASS' ELSE 'FAIL' END AS four_fifths
        FROM models m JOIN model_metrics mm USING(model_id) ORDER BY m.model_id""")


def get_group_metrics(model_id: str = "M5", attribute: str = "gender") -> list[dict]:
    if attribute not in ATTRIBUTES:
        return [{"error": f"attribute must be one of {ATTRIBUTES}"}]
    return query("""
        SELECT model_id, group_name, n, ROUND(qualified_share,3) AS qualified_share,
               ROUND(selection_rate,3) AS selection_rate, ROUND(impact_ratio_vs_best,3) AS impact_ratio,
               ROUND(tpr,3) AS recall_qualified, ROUND(fpr,3) AS fpr
        FROM group_metrics WHERE model_id = ? AND attribute = ? ORDER BY selection_rate DESC""",
                    (model_id.upper(), attribute))


def detect_proxies(attribute: str | None = None) -> list[dict]:
    sql = ("SELECT sensitive_attribute, feature, ROUND(auc_predicting_group,3) AS auc, risk_level "
           "FROM proxy_scores {w} ORDER BY auc_predicting_group DESC")
    if attribute:
        return query(sql.format(w="WHERE sensitive_attribute = ?"), (attribute,))
    return query(sql.format(w=""))


def run_sql_query(sql: str) -> dict:
    try:
        return safe_select(sql)
    except Exception as e:  # noqa: BLE001 - returned to the model so it can fix the query
        return {"error": str(e)}


def score_new_applicant(**fields) -> dict:
    try:
        return score(ApplicantInput(**fields))
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}


def explain_applicant_decision(applicant_id: str) -> dict:
    return explain_applicant(applicant_id.upper())


def get_data_quality() -> dict:
    return {"profile": query("SELECT metric, raw, clean FROM data_quality_profile"),
            "steps": query("SELECT step_no, category, step, rows_affected, rows_after FROM data_quality_log ORDER BY step_no")}


def get_database_schema() -> str:
    return schema_description()


TOOL_FUNCTIONS = {
    "get_fairness_summary": get_fairness_summary,
    "compare_models": compare_models,
    "get_group_metrics": get_group_metrics,
    "detect_proxies": detect_proxies,
    "run_sql_query": run_sql_query,
    "score_new_applicant": score_new_applicant,
    "explain_applicant_decision": explain_applicant_decision,
    "get_database_schema": get_database_schema,
    "get_data_quality": get_data_quality,
}

# JSON schemas in the format used by the Claude Messages API (tool use)
TOOL_SCHEMAS = [
    {"name": "get_fairness_summary",
     "description": "Headline audit results: historical hire rates by gender, the gap among equally qualified "
                    "applicants, and a PASS/FAIL four-fifths scorecard for every model.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "compare_models",
     "description": "Accuracy and fairness metrics for the five models M1-M5 (AUC, recall of qualified "
                    "applicants, impact ratios, equal-opportunity gaps, four-fifths verdict).",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "get_group_metrics",
     "description": "Per-group selection rate, impact ratio and recall of qualified applicants for one model "
                    "and one sensitive attribute.",
     "input_schema": {"type": "object", "properties": {
         "model_id": {"type": "string", "enum": ["M1", "M2", "M3", "M4", "M5"]},
         "attribute": {"type": "string", "enum": list(ATTRIBUTES)}},
         "required": ["model_id", "attribute"]}},
    {"name": "detect_proxies",
     "description": "Features that reveal a sensitive attribute (AUC > 0.7 = high proxy risk).",
     "input_schema": {"type": "object", "properties": {
         "attribute": {"type": "string", "enum": ["gender", "region_group"]}}}},
    {"name": "run_sql_query",
     "description": "Run ONE read-only SQLite SELECT on the audit database (max 200 rows). Call "
                    "get_database_schema first if unsure of column names.",
     "input_schema": {"type": "object", "properties": {"sql": {"type": "string"}}, "required": ["sql"]}},
    {"name": "score_new_applicant",
     "description": "Score a new (hypothetical) applicant with the recommended model M5 and explain the score. "
                    "Uses job-relevant fields only.",
     "input_schema": {"type": "object", "properties": {
         "gpa": {"type": "number", "description": "0-5 scale"},
         "years_experience": {"type": "integer"},
         "english_score": {"type": "number", "description": "IELTS-like 0-9"},
         "certifications": {"type": "integer"},
         "internship": {"type": "integer", "enum": [0, 1]},
         "university_tier": {"type": "integer", "enum": [1, 2, 3]},
         "degree_field": {"type": "string", "enum": ["Computer Science", "Information Systems", "Engineering",
                                                     "Statistics", "Business Administration"]},
         "job_family": {"type": "string", "enum": ["Data Analyst", "Software Engineer", "Cybersecurity Specialist",
                                                   "Business Analyst", "HR Specialist"]}},
         "required": ["gpa", "years_experience", "english_score"]}},
    {"name": "explain_applicant_decision",
     "description": "Look up an applicant by ID (e.g. APP-00042): attributes, every model's decision and the "
                    "factors behind the M5 score.",
     "input_schema": {"type": "object", "properties": {"applicant_id": {"type": "string"}},
                      "required": ["applicant_id"]}},
    {"name": "get_data_quality",
     "description": "How the messy HR export was cleaned: before/after profile (rows, duplicates, missing cells, "
                    "label variants, personal identifiers) and every cleaning step with rows affected.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "get_database_schema",
     "description": "List all tables and views with their columns.",
     "input_schema": {"type": "object", "properties": {}}},
]


# ==================================================================================================
# 4. AI ASSISTANT - SYSTEM PROMPT
# ==================================================================================================
SYSTEM_PROMPT = """You are FairHire Assistant, the AI assistant of the "Saudi Fair Hiring Audit" project
(CS 476 Ethics in Data Science, Majmaah University).

The project audits an AI recruitment system built on 10,000 SYNTHETIC Saudi job applications. Past
hiring decisions contain planted bias against women and against applicants from the Southern and
Northern regions. Five models were trained on those decisions:
- M1 Black-box baseline (gradient boosting, all features)
- M2 Transparent baseline (logistic regression, all features)
- M3 Fairness through unawareness (sensitive columns dropped, proxies kept)
- M4 Unaware + reweighing
- M5 Job-relevant only + reweighing (recommended)
The raw HR export (duplicates, mixed Arabic/English labels, numbers stored as text, test rows and
personal identifiers) was cleaned in 24 logged steps into 10,000 anonymised applicants.
Key metrics: impact ratio (four-fifths rule, >= 0.80 passes), equal-opportunity gap (difference in
recall of truly qualified applicants), AUC.

How to work:
1. Always use the tools to get numbers. Never invent figures.
2. Prefer the specific tools; use run_sql_query for anything else (read-only SELECT on SQLite).
3. Reply in the user's language (Arabic or English). Be concise; use a small markdown table when
   comparing numbers.
4. Explain metrics in plain words for students.

Ethical rules:
- The data is synthetic; say so if asked about real people or companies.
- You support, but never replace, human hiring decisions. When you score an applicant, say that a
  human recruiter makes the final decision.
- Do not help to discriminate. If asked to rank or filter people by gender, nationality or region to
  exclude them, refuse and explain why; analysing outcomes by group for auditing is fine.
- Nationality differences are reported but not counted as a fairness failure, because Saudization
  (Nitaqat) is a lawful national policy.
"""


# ==================================================================================================
# 5. AI ASSISTANT - AGENT (Claude tool use, or offline Arabic/English router)
# ==================================================================================================
DEFAULT_MODEL = os.getenv("FAIRHIRE_MODEL", "claude-sonnet-5-5")
MAX_STEPS = 6
AR = re.compile(r"[؀-ۿ]")


def _ar(text: str) -> bool:
    return bool(AR.search(text))


def _pct(v) -> str:
    return "–" if v is None else f"{v * 100:.1f}%"


def _table(rows: list[dict], cols: list[str], headers: list[str] | None = None) -> str:
    headers = headers or cols
    out = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        out.append("| " + " | ".join("" if r.get(c) is None else str(r.get(c)) for c in cols) + " |")
    return "\n".join(out)


class FairHireAgent:
    def __init__(self, api_key: str | None = None, model: str = DEFAULT_MODEL):
        self.model = model
        self.client = None
        key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if key:
            try:
                import anthropic

                self.client = anthropic.Anthropic(api_key=key)
            except ImportError:
                self.client = None

    @property
    def mode(self) -> str:
        return "claude" if self.client else "offline"

    # ------------------------------------------------------------------ public
    def chat(self, message: str, history: list[dict] | None = None) -> dict:
        trace: list[dict] = []
        mode = self.mode
        if self.client:
            try:
                answer = self._chat_claude(message, history or [], trace)
            except Exception as e:  # noqa: BLE001 - network / auth problems fall back to offline mode
                mode = "offline"
                answer = self._chat_offline(message, trace)
                answer += f"\n\n_(Claude API unavailable: {type(e).__name__}. Answered in offline mode.)_"
        else:
            answer = self._chat_offline(message, trace)
        sql = [t["input"]["sql"] for t in trace if t["tool"] == "run_sql_query"]
        execute_write("INSERT INTO agent_logs (mode, question, answer, tools_used, sql_executed) VALUES (?,?,?,?,?)",
                         (mode, message, answer, json.dumps([t["tool"] for t in trace]), json.dumps(sql)))
        return {"answer": answer, "mode": mode, "tools": trace, "sql": sql}

    def _call(self, name: str, args: dict, trace: list[dict]):
        result = TOOL_FUNCTIONS[name](**args)
        trace.append({"tool": name, "input": args})
        return result

    # ------------------------------------------------------------------ Claude tool-use loop
    def _chat_claude(self, message: str, history: list[dict], trace: list[dict]) -> str:
        messages = [{"role": h["role"], "content": h["content"]} for h in history[-10:]
                    if h.get("role") in ("user", "assistant") and h.get("content")]
        messages.append({"role": "user", "content": message})
        for _ in range(MAX_STEPS):
            resp = self.client.messages.create(model=self.model, max_tokens=1500, system=SYSTEM_PROMPT,
                                               tools=TOOL_SCHEMAS, messages=messages)
            if resp.stop_reason != "tool_use":
                return "".join(b.text for b in resp.content if b.type == "text").strip()
            messages.append({"role": "assistant", "content": resp.content})
            results = []
            for block in resp.content:
                if block.type == "tool_use":
                    try:
                        out = self._call(block.name, dict(block.input), trace)
                    except Exception as e:  # noqa: BLE001
                        out = {"error": str(e)}
                    results.append({"type": "tool_result", "tool_use_id": block.id,
                                    "content": json.dumps(out, default=str)[:20000]})
            messages.append({"role": "user", "content": results})
        return "I could not finish within the step limit. Please ask a narrower question."

    # ------------------------------------------------------------------ offline rule-based agent
    def _chat_offline(self, msg: str, trace: list[dict]) -> str:
        ar, low = _ar(msg), msg.lower()
        has = lambda *w: any(x in low for x in w)  # noqa: E731

        if re.match(r"^\s*(select|with|drop|delete|update|insert|alter|create|pragma|attach)\b", msg, re.I):
            return self._sql(msg, ar, trace)
        if m := re.search(r"app-?\s?(\d{1,5})", low):
            return self._explain(f"APP-{int(m.group(1)):05d}", ar, trace)
        if has("exclude", "reject all", "don't hire", "do not hire", "filter out", "استبعد", "ارفض", "لا توظف",
               "لا تقبل", "ابعد"):
            return self._refuse(ar)
        if has("score", "evaluate", "assess", "قيم", "قيّم", "تقييم", "قيّم", "احسب") and re.search(r"\d", msg):
            return self._score(msg, ar, trace)
        if has("clean", "quality", "duplicate", "missing", "تنظيف", "تنظف", "نظف", "جودة", "مكرر", "ناقص", "البيانات الخام", "raw data"):
            return self._quality(ar, trace)
        if has("proxy", "proxies", "بديل", "البديلة", "تسرب", "leak"):
            return self._proxies(ar, trace)
        if has("compare", "best", "model", "قارن", "مقارنة", "أفضل", "افضل", "النماذج", "نموذج", "موديل"):
            return self._models(ar, trace)
        if has("region", "منطق", "الجنوب", "الشمال", "south", "north"):
            return self._groups("region_group", ar, trace)
        if has("nationality", "saudi", "الجنسية", "سعودي", "سعودة", "نطاقات"):
            return self._groups("nationality", ar, trace)
        if has("gender", "women", "woman", "female", "men ", "النساء", "المرأة", "الجنس", "الرجال", "نساء"):
            return self._groups("gender", ar, trace)
        if has("schema", "table", "column", "جدول", "جداول", "أعمدة", "قاعدة البيانات"):
            s = self._call("get_database_schema", {}, trace)
            return ("**جداول قاعدة البيانات:**\n" if ar else "**Database tables and views:**\n") + f"```\n{s}\n```"
        if has("help", "hello", "hi", "مرحبا", "السلام", "وش تقدر", "ماذا تستطيع", "ساعد"):
            return self._help(ar)
        return self._summary(ar, trace)

    def _help(self, ar: bool) -> str:
        if ar:
            return ("أنا **مساعد FairHire**. أقدر:\n- ألخص نتائج تدقيق العدالة\n- أقارن النماذج الخمسة\n"
                    "- أعرض النتائج حسب الجنس أو المنطقة أو الجنسية\n- أكشف المتغيرات البديلة (proxies)\n- أشرح كيف نظفنا البيانات الخام\n"
                    "- أشرح قرار متقدم مثل `APP-00042`\n- أقيّم متقدم جديد: *قيّم متقدم معدله 4.2 وخبرته 3 سنوات وإنجليزي 7*\n"
                    "- أنفذ استعلام SQL للقراءة فقط يبدأ بـ `SELECT`")
        return ("I'm the **FairHire Assistant**. I can:\n- summarise the fairness audit\n- compare the five models\n"
                "- break results down by gender, region or nationality\n- detect proxy features\n- explain how the raw data was cleaned\n"
                "- explain a decision, e.g. `APP-00042`\n- score a new applicant: *score gpa 4.2, 3 years experience, english 7*\n"
                "- run a read-only SQL query starting with `SELECT`")

    def _summary(self, ar: bool, trace) -> str:
        s = self._call("get_fairness_summary", {}, trace)
        h = s["historical_decisions"]
        rows = [{**r, "auc": r["auc"], "four_fifths_rule": ("✅ " if r["four_fifths_rule"] == "PASS" else "❌ ")
                 + r["four_fifths_rule"]} for r in s["model_scorecard"]]
        tbl = _table(rows, ["model_id", "model_name", "auc", "di_gender", "di_region_group", "four_fifths_rule"],
                     ["Model", "Name", "AUC", "DI gender", "DI region", "4/5 rule"])
        if ar:
            return (f"**ملخص التدقيق** (بيانات مصطنعة، {h['applicants']:,} متقدم):\n\n"
                    f"- نسبة التوظيف التاريخية: **{_pct(h['hire_rate'])}**\n"
                    f"- النساء {_pct(h['female_rate'])} مقابل الرجال {_pct(h['male_rate'])} ← نسبة الأثر "
                    f"**{h['gender_impact_ratio']:.2f}** (أقل من 0.80 = تحيز)\n"
                    f"- بين المؤهلين فعلاً، الرجال توظفوا أكثر بفارق **{h['qualified_gap_pp']:.1f} نقطة مئوية**\n\n"
                    f"{tbl}\n\nالنموذج الموصى به **M5**: يحذف الصفات الحساسة والمتغيرات البديلة ويستخدم إعادة الوزن.")
        return (f"**Audit summary** (synthetic data, {h['applicants']:,} applicants):\n\n"
                f"- Historical hire rate: **{_pct(h['hire_rate'])}**\n"
                f"- Women {_pct(h['female_rate'])} vs men {_pct(h['male_rate'])} → impact ratio "
                f"**{h['gender_impact_ratio']:.2f}** (below 0.80 = adverse impact)\n"
                f"- Among truly qualified applicants, men were hired **{h['qualified_gap_pp']:.1f} pp** more often\n\n"
                f"{tbl}\n\nRecommended model: **M5** (sensitive attributes and proxies removed + reweighing).")

    def _models(self, ar: bool, trace) -> str:
        rows = self._call("compare_models", {}, trace)
        tbl = _table(rows, ["model_id", "mitigation", "auc", "recall_qualified", "di_gender", "di_region", "four_fifths"],
                     ["Model", "Mitigation", "AUC", "Recall (qualified)", "DI gender", "DI region", "4/5"])
        m2 = next(r for r in rows if r["model_id"] == "M2")
        m5 = next(r for r in rows if r["model_id"] == "M5")
        if ar:
            return (f"{tbl}\n\n**الخلاصة:** M5 رفع نسبة الأثر للجنس من {m2['di_gender']} إلى **{m5['di_gender']}** "
                    f"والدقة (AUC) نزلت بس من {m2['auc']} إلى {m5['auc']}، وزاد اكتشاف المؤهلين فعلاً من "
                    f"{m2['recall_qualified']} إلى {m5['recall_qualified']}. يعني العدالة هنا شبه مجانية.\n\n"
                    "ملاحظة: M3 حذف الجنس لكنه ما نجح، لأن المتغيرات البديلة سرّبت التحيز.")
        return (f"{tbl}\n\n**Takeaway:** M5 raises the gender impact ratio from {m2['di_gender']} to "
                f"**{m5['di_gender']}** while AUC only moves from {m2['auc']} to {m5['auc']}, and recall of truly "
                f"qualified applicants improves from {m2['recall_qualified']} to {m5['recall_qualified']}.\n\n"
                "Note that M3 removed gender but still fails: proxy features leak the bias back in.")

    def _groups(self, attr: str, ar: bool, trace) -> str:
        parts = []
        for mid in ("M2", "M5"):
            rows = self._call("get_group_metrics", {"model_id": mid, "attribute": attr}, trace)
            parts.append(f"**{mid}**\n\n" + _table(rows, ["group_name", "n", "selection_rate", "impact_ratio",
                                                          "recall_qualified"],
                                                   ["Group", "n", "Selection rate", "Impact ratio", "Recall (qualified)"]))
        body = "\n\n".join(parts)
        note_ar = {"nationality": "\n\nالفروق حسب الجنسية تُعرض للشفافية فقط، لأن السعودة (نطاقات) سياسة نظامية.",
                   "gender": "\n\nنسبة الأثر أقل من 0.80 تعني تحيز حسب قاعدة الأربعة أخماس.",
                   "region_group": "\n\nالمنطقتان الجنوبية والشمالية عيّنتهما صغيرة، فالنتائج فيها تذبذب أكبر."}
        note_en = {"nationality": "\n\nNationality gaps are shown for transparency only: Saudization (Nitaqat) is a lawful policy.",
                   "gender": "\n\nAn impact ratio below 0.80 signals adverse impact under the four-fifths rule.",
                   "region_group": "\n\nSouthern and Northern test groups are small, so their numbers are noisier."}
        title = {"gender": ("الجنس", "gender"), "region_group": ("المنطقة", "region"),
                 "nationality": ("الجنسية", "nationality")}[attr]
        if ar:
            return f"**النتائج حسب {title[0]}: النموذج الأساسي M2 مقابل M5**\n\n{body}{note_ar[attr]}"
        return f"**Results by {title[1]}: baseline M2 vs recommended M5**\n\n{body}{note_en[attr]}"

    def _quality(self, ar: bool, trace) -> str:
        q = self._call("get_data_quality", {}, trace)
        prof = _table(q["profile"], ["metric", "raw", "clean"], ["Metric", "Raw export", "Clean"])
        steps = [s for s in q["steps"] if s["rows_affected"] and s["category"] not in ("Load",)]
        top = _table(sorted(steps, key=lambda s: -s["rows_affected"])[:6], ["category", "step", "rows_affected"],
                     ["Category", "Step", "Rows affected"])
        if ar:
            return (f"**تنظيف البيانات:** {len(q['steps'])} خطوة حوّلت تصدير نظام الموارد البشرية المتسخ إلى جدول نظيف.\n\n"
                    f"{prof}\n\n**أكثر الخطوات تأثيراً:**\n\n{top}\n\nحذفنا الاسم والهوية والجوال والإيميل "
                    "(تقليل البيانات حسب نظام حماية البيانات الشخصية) وخلينا رقم المتقدم كمعرّف مستعار.")
        return (f"**Data cleaning:** {len(q['steps'])} steps turned the messy HR export into a clean table.\n\n"
                f"{prof}\n\n**Biggest steps:**\n\n{top}\n\nName, national ID, mobile and e-mail were removed "
                "(PDPL data minimisation); the App ID is kept as a pseudonym.")

    def _proxies(self, ar: bool, trace) -> str:
        rows = [r for r in self._call("detect_proxies", {}, trace) if r["feature"] != "ALL FEATURES COMBINED"][:6]
        tbl = _table(rows, ["sensitive_attribute", "feature", "auc", "risk_level"], ["Reveals", "Feature", "AUC", "Risk"])
        if ar:
            return (f"{tbl}\n\nالـAUC يقيس قدرة المتغير على كشف الفئة (0.5 = ما يكشف شي، 1 = يكشفها تماماً). "
                    "`career_gap_months` يكشف الجنس و`commute_km` يكشف المنطقة، وهم ما لهم علاقة بالكفاءة، "
                    "عشان كذا حذفناهم في M5.")
        return (f"{tbl}\n\nAUC shows how well a feature reveals the group (0.5 = nothing, 1 = perfectly). "
                "`career_gap_months` reveals gender and `commute_km` reveals region, and neither measures ability, "
                "so M5 drops both.")

    def _explain(self, app_id: str, ar: bool, trace) -> str:
        r = self._call("explain_applicant_decision", {"applicant_id": app_id}, trace)
        if "error" in r:
            return ("ما لقيت هذا المتقدم. الأرقام من APP-00001 إلى APP-10000." if ar else
                    "Applicant not found. IDs run from APP-00001 to APP-10000.")
        a = r["applicant"]
        preds = _table([{**p, "decision": "✅ Shortlist" if p["predicted_hire"] else "—"} for p in r["predictions"]],
                       ["model_id", "model_name", "score", "decision"], ["Model", "Name", "Score", "Decision"]) \
            if r["predictions"] else ("_(هذا المتقدم من بيانات التدريب، ما عليه توقعات اختبار)_" if ar
                                      else "_(training-set applicant: no test predictions)_")
        factors = _table(r["m5_explanation"], ["label", "value", "contribution", "effect"],
                         ["Factor", "Value", "Contribution", "Effect"])
        head = (f"**{app_id}** · {a['job_family']} · {a['gender']} · {a['region']} · GPA {a['gpa']} · "
                f"{a['years_experience']} yrs · English {a['english_score']} · qualification {a['qualification_score']} "
                f"({'qualified' if a['truly_qualified'] else 'not qualified'}) · past decision: "
                f"{'hired' if a['hired_historical'] else 'not hired'}")
        if ar:
            return f"{head}\n\n**قرارات النماذج:**\n\n{preds}\n\n**أهم العوامل في درجة M5 ({r['m5_score']}):**\n\n{factors}"
        return f"{head}\n\n**Model decisions:**\n\n{preds}\n\n**Main factors behind the M5 score ({r['m5_score']}):**\n\n{factors}"

    def _score(self, msg: str, ar: bool, trace) -> str:
        low = msg.lower()
        num = r"(\d+(?:\.\d+)?)"
        f = {}
        if m := re.search(r"(?:gpa|معدل[هـ]?|المعدل)\D{0,8}" + num, low):
            f["gpa"] = float(m.group(1))
        if m := (re.search(num + r"\s*(?:years?|yrs?|سنة|سنوات|سنين)", low)
                 or re.search(r"(?:experience|خبرة|خبرته)\D{0,8}" + num, low)):
            f["years_experience"] = int(float(m.group(1)))
        if m := re.search(r"(?:english|ielts|انجليزي|إنجليزي|الانجليزي|الإنجليزي)\D{0,8}" + num, low):
            f["english_score"] = float(m.group(1))
        if m := re.search(num + r"\s*(?:cert|شهاد)", low):
            f["certifications"] = int(float(m.group(1)))
        if re.search(r"internship|تدريب|تعاوني", low):
            f["internship"] = 1
        if missing := [k for k in ("gpa", "years_experience", "english_score") if k not in f]:
            return ("أحتاج المعدل وسنوات الخبرة ودرجة الإنجليزي. مثال: *قيّم متقدم معدله 4.2 وخبرته 3 سنوات وإنجليزي 7*"
                    if ar else "I need GPA, years of experience and English score, e.g. "
                               "*score gpa 4.2, 3 years experience, english 7*") + f" (missing: {', '.join(missing)})"
        r = self._call("score_new_applicant", f, trace)
        if "error" in r:
            return f"⚠️ {r['error']}"
        factors = _table(r["factors"], ["label", "value", "contribution", "effect"], ["Factor", "Value", "Contribution", "Effect"])
        if ar:
            dec = "✅ يُرشَّح للمقابلة" if r["decision"] == "Shortlist" else "❌ لا يُرشَّح"
            return (f"**النتيجة (M5):** {r['score']} · أعلى من {r['percentile']}% من المتقدمين · الحد {r['cutoff']}\n\n"
                    f"**التوصية:** {dec}\n\n{factors}\n\n⚖️ هذي توصية فقط، والقرار النهائي لموظف التوظيف. "
                    f"(سُجّل الطلب في جدول `scoring_requests` رقم {r.get('request_id')})")
        dec = "✅ Shortlist" if r["decision"] == "Shortlist" else "❌ Not shortlisted"
        return (f"**Score (M5):** {r['score']} · higher than {r['percentile']}% of applicants · cutoff {r['cutoff']}\n\n"
                f"**Recommendation:** {dec}\n\n{factors}\n\n⚖️ Recommendation only; a human recruiter decides. "
                f"(Logged in `scoring_requests`, id {r.get('request_id')})")

    def _sql(self, sql: str, ar: bool, trace) -> str:
        r = self._call("run_sql_query", {"sql": sql}, trace)
        if "error" in r:
            return ("⚠️ الاستعلام مرفوض: " if ar else "⚠️ Query rejected: ") + r["error"]
        rows = [dict(zip(r["columns"], row)) for row in r["rows"][:25]]
        more = len(r["rows"]) - len(rows)
        tail = (f"\n\n_+{more} صف إضافي_" if ar else f"\n\n_+{more} more rows_") if more > 0 else ""
        return _table(rows, r["columns"]) + tail

    def _refuse(self, ar: bool) -> str:
        if ar:
            return ("ما أقدر أساعد في استبعاد أشخاص بسبب الجنس أو المنطقة أو أي صفة حساسة. هذا تمييز، وهو عكس هدف "
                    "المشروع. أقدر أعرض لك كيف تختلف النتائج بين الفئات عشان نكشف التحيز ونعالجه.")
        return ("I can't help exclude people because of gender, region or another sensitive attribute: that is "
                "discrimination and the opposite of this project's goal. I can show how outcomes differ between "
                "groups so the bias can be found and fixed.")


# ==================================================================================================
# 6. REST API + WEB PAGE
# ==================================================================================================
agent: FairHireAgent | None = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    global agent
    ensure_built()
    agent = FairHireAgent()
    yield


app = FastAPI(title="Saudi Fair Hiring Audit API", version="1.0.0", lifespan=lifespan,
              description="Fairness audit of an AI recruitment model on synthetic Saudi data.")


# ------------------------------------------------------------------ audit data
@app.get("/api/health")
def health():
    return {"status": "ok", "agent_mode": agent.mode if agent else "starting"}


@app.get("/api/summary")
def summary():
    return get_fairness_summary()


@app.get("/api/models")
def models():
    return compare_models()


@app.get("/api/group-metrics")
def group_metrics(model_id: str = "M5", attribute: str = "gender"):
    rows = get_group_metrics(model_id, attribute)
    if rows and "error" in rows[0]:
        raise HTTPException(400, rows[0]["error"])
    return rows


@app.get("/api/history")
def history():
    q = audit_queries()
    return {
        "by_band": query(q["q05_qualification_band_gap"]),
        "by_region": query(q["q03_historical_selection_by_region"]),
        "by_gender": query(q["q02_historical_selection_by_gender"]),
    }


@app.get("/api/data-quality")
def data_quality():
    q = get_data_quality()
    q["clean_sample"] = query("SELECT applicant_id, job_family, gender, nationality, region, gpa, years_experience, "
                                 "english_score, internship, hired_historical FROM applicants ORDER BY applicant_id LIMIT 6")
    return q


@app.get("/api/data-quality/raw-sample")
def raw_sample(limit: int = 6):
    """A few rows of the messy export, with personal identifiers masked."""
    import pandas as pd

    raw = pd.read_csv(OUT / "hr_export_raw.csv", dtype=str, encoding="utf-8-sig")
    raw = raw[raw["App ID"].isin([r["applicant_id"] for r in data_quality()["clean_sample"]])].head(limit)
    for c, keep in [("Full Name", 1), ("National ID", 2), ("Mobile", 2), ("Email", 1)]:
        raw[c] = raw[c].fillna("").map(lambda v, k=keep: (v[:k] + "•" * 6) if v else "")
    cols = ["App ID", "Full Name", "National ID", "Application Date", "Job Title", "Gender", "Nationality", "City/Region",
            "GPA", "Experience", "English (IELTS)", "Internship", "Hired"]
    return raw[cols].fillna("").to_dict(orient="records")


@app.get("/api/proxies")
def proxies():
    return detect_proxies()


@app.get("/api/feature-importance")
def feature_importance(model_id: str = "M5", top: int = 10):
    return query("SELECT feature, ROUND(importance,4) AS importance, direction FROM feature_importance "
                    "WHERE model_id = ? ORDER BY importance DESC LIMIT ?", (model_id.upper(), top))


@app.get("/api/applicants")
def applicants(model_id: str = "M5", gender: str | None = None, region_group: str | None = None,
               outcome: str | None = None, limit: int = Query(25, le=100), offset: int = 0):
    where, params = ["model_id = ?"], [model_id.upper()]
    for col, val in [("gender", gender), ("region_group", region_group), ("outcome_type", outcome)]:
        if val:
            where.append(f"{col} = ?")
            params.append(val)
    w = " AND ".join(where)
    total = query(f"SELECT COUNT(*) AS n FROM vw_applicant_predictions WHERE {w}", tuple(params))[0]["n"]
    rows = query(f"SELECT applicant_id, job_family, gender, nationality, region, qualification_score, "
                    f"truly_qualified, hired_historical, ROUND(score,3) AS score, predicted_hire, outcome_type "
                    f"FROM vw_applicant_predictions WHERE {w} ORDER BY score DESC LIMIT ? OFFSET ?",
                    tuple(params + [limit, offset]))
    return {"total": total, "rows": rows}


@app.get("/api/applicants/{applicant_id}")
def applicant(applicant_id: str):
    r = explain_applicant(applicant_id.upper())
    if "error" in r:
        raise HTTPException(404, r["error"])
    return r


# ------------------------------------------------------------------ scoring (writes to SQL)
@app.get("/api/options")
def options():
    return {"degree_field": DEGREES, "job_family": JOB_FAMILIES}


@app.post("/api/score")
def score_applicant(inp: ApplicantInput):
    return score(inp)


@app.get("/api/scoring-requests")
def scoring_requests(limit: int = 10):
    return query("SELECT request_id, created_at, inputs_json, ROUND(score,3) AS score, decision, "
                    "ROUND(percentile,1) AS percentile FROM scoring_requests ORDER BY request_id DESC LIMIT ?", (limit,))


# ------------------------------------------------------------------ SQL audit
@app.get("/api/audit")
def audit_list():
    return [{"name": n, "sql": s} for n, s in audit_queries().items()]


@app.get("/api/audit/{name}")
def audit_run(name: str):
    q = audit_queries()
    if name not in q:
        raise HTTPException(404, "Unknown audit query")
    return {"name": name, "sql": q[name], **safe_select(q[name])}


# ------------------------------------------------------------------ AI assistant
class ChatIn(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    history: list[dict] = Field(default_factory=list)


@app.post("/api/agent/chat")
def agent_chat(body: ChatIn):
    return agent.chat(body.message, body.history)


@app.get("/api/agent/logs")
def agent_logs(limit: int = 20):
    return query("SELECT log_id, created_at, mode, question, tools_used, sql_executed FROM agent_logs "
                    "ORDER BY log_id DESC LIMIT ?", (limit,))


# ------------------------------------------------------------------ front end
@app.get("/", include_in_schema=False)
def index():
    return FileResponse(ROOT / "index.html")
