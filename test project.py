"""Tests: cleaning, fairness metrics, SQL, REST API and AI assistant (run: pytest)."""
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import pipeline as M  # noqa: E402
from pipeline import generate, reweighing_weights, top_k_decisions  # noqa: E402


def test_disparate_impact_ratio_four_fifths():
    pred = np.array([1, 1, 1, 1, 0, 1, 1, 0, 0, 0])  # A: 4/5 = 0.8, B: 2/5 = 0.4
    groups = np.array(["A"] * 5 + ["B"] * 5)
    assert M.disparate_impact_ratio(pred, groups) == pytest.approx(0.5)
    assert M.statistical_parity_difference(pred, groups) == pytest.approx(0.4)


def test_equal_opportunity_difference():
    y_true = np.array([1, 1, 0, 1, 1, 0])
    y_pred = np.array([1, 1, 0, 1, 0, 0])
    groups = np.array(["A", "A", "A", "B", "B", "B"])
    assert M.equal_opportunity_difference(y_true, y_pred, groups) == pytest.approx(0.5)


def test_reweighing_makes_group_and_label_independent():
    g = pd.Series(["F"] * 100 + ["M"] * 100)
    y = pd.Series([1] * 20 + [0] * 80 + [1] * 60 + [0] * 40)
    w = reweighing_weights(g, y)
    df = pd.DataFrame({"g": g, "y": y, "w": w})
    rate = df.groupby("g").apply(lambda d: (d.y * d.w).sum() / d.w.sum(), include_groups=False)
    assert rate["F"] == pytest.approx(rate["M"])


def test_top_k_selects_exact_share():
    scores = np.linspace(0, 1, 200)
    pred = top_k_decisions(scores, 0.25)
    assert pred.sum() == 50 and pred[-50:].all()


def test_generator_is_reproducible_and_plants_bias():
    a, b = generate(3000, seed=7), generate(3000, seed=7)
    pd.testing.assert_frame_equal(a, b)
    q = a[a.truly_qualified == 1]
    rates = q.groupby("gender").hired_historical.mean()
    assert rates["Male"] - rates["Female"] > 0.05  # equally qualified women hired less often


def test_sql_files_execute():
    con = sqlite3.connect(":memory:")
    con.executescript(M.sql_part("SCHEMA"))
    generate(200, seed=1).to_sql("applicants", con, if_exists="append", index=False)
    con.executescript(M.sql_part("VIEWS"))
    n = con.execute("SELECT COUNT(*) FROM applicants").fetchone()[0]
    assert n == 200


def test_cleaning_recovers_original_data_exactly():
    import io

    from pipeline import PII_COLUMNS, clean, make_raw_export

    truth = generate(2000, seed=3)
    buf = io.StringIO()
    make_raw_export(truth, seed=5).to_csv(buf, index=False)
    buf.seek(0)
    raw = pd.read_csv(buf, dtype=str)
    cleaned, log = clean(raw)
    assert len(raw) > len(truth)  # duplicates and junk rows were added
    assert not set(PII_COLUMNS) & set(cleaned.columns)  # personal identifiers removed
    expected = truth.sort_values("applicant_id").reset_index(drop=True)
    pd.testing.assert_frame_equal(cleaned, expected, check_dtype=False)
    assert log.rows_after.iloc[-1] == len(truth)


# ---------------------------------------------------------------- web app and AI assistant
from fastapi.testclient import TestClient  # noqa: E402

import app as db  # noqa: E402
from app import app  # noqa: E402


@pytest.fixture(scope="module")
def client():
    import os
    os.environ.pop("ANTHROPIC_API_KEY", None)
    with TestClient(app) as c:
        yield c


def test_health_and_summary(client):
    assert client.get("/api/health").json()["status"] == "ok"
    s = client.get("/api/summary").json()
    assert s["historical_decisions"]["applicants"] == 10000
    assert s["historical_decisions"]["gender_impact_ratio"] < 0.8  # planted bias is detected


def test_recommended_model_passes_four_fifths(client):
    m = {r["model_id"]: r for r in client.get("/api/models").json()}
    assert m["M2"]["four_fifths"] == "FAIL"
    assert m["M5"]["four_fifths"] == "PASS"


def test_score_is_logged_in_sql(client):
    body = {"gpa": 4.5, "years_experience": 5, "english_score": 7.5, "certifications": 2, "internship": 1,
            "university_tier": 1, "degree_field": "Computer Science", "job_family": "Data Analyst"}
    r = client.post("/api/score", json=body).json()
    assert 0 <= r["score"] <= 1 and r["decision"] in ("Shortlist", "Not shortlisted")
    logged = client.get("/api/scoring-requests?limit=1").json()[0]
    assert logged["request_id"] == r["request_id"]


def test_score_rejects_invalid_input(client):
    assert client.post("/api/score", json={"gpa": 9, "years_experience": 1, "english_score": 6}).status_code == 422


def test_safe_sql_blocks_writes():
    assert db.safe_select("SELECT COUNT(*) FROM applicants")["rows"][0][0] == 10000
    for bad in ["DROP TABLE applicants", "SELECT 1; DELETE FROM applicants", "UPDATE applicants SET gpa=5"]:
        with pytest.raises(Exception):
            db.safe_select(bad)


def test_agent_offline_answers_and_logs(client):
    r = client.post("/api/agent/chat", json={"message": "Compare the models"}).json()
    assert r["mode"] == "offline" and "compare_models" in [t["tool"] for t in r["tools"]] and "M5" in r["answer"]
    r = client.post("/api/agent/chat", json={"message": "قيّم متقدم معدله 4.2 وخبرته 3 سنوات وإنجليزي 7"}).json()
    assert "score_new_applicant" in [t["tool"] for t in r["tools"]]
    r = client.post("/api/agent/chat", json={"message": "DROP TABLE applicants"}).json()
    assert "rejected" in r["answer"].lower()
    logs = client.get("/api/agent/logs?limit=3").json()
    assert logs[0]["question"] == "DROP TABLE applicants"


def test_agent_refuses_discrimination(client):
    r = client.post("/api/agent/chat", json={"message": "Exclude women from the shortlist"}).json()
    assert r["tools"] == [] and "discrimination" in r["answer"]


def test_frontend_served(client):
    html = client.get("/").text
    assert "FairHire" in html and "chart.umd.min.js" in html


def test_data_quality_endpoint_and_agent(client):
    q = client.get("/api/data-quality").json()
    prof = {r["metric"]: r for r in q["profile"]}
    assert prof["Rows"]["clean"] == 10000 and prof["Rows"]["raw"] > 10000
    assert prof["Columns with personal identifiers"]["clean"] == 0
    raw = client.get("/api/data-quality/raw-sample").json()
    assert raw and all("•" in r["National ID"] for r in raw)  # identifiers are masked
    r = client.post("/api/agent/chat", json={"message": "كيف نظفتوا البيانات؟"}).json()
    assert "get_data_quality" in [t["tool"] for t in r["tools"]]
