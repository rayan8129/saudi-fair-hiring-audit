# Saudi Fair Hiring Audit
### Measuring and removing bias in AI-assisted recruitment

> **CS 476 – Ethics in Data Science · Majmaah University**
> Python · SQL · Machine Learning · AI Agent · Web App · Excel · Power BI. All data is synthetic.

**بالعربي:** مشروع يدقق في عدالة نظام ذكاء اصطناعي لفرز طلبات التوظيف. نولّد ١٠٬٠٠٠ طلب توظيف مصطنع فيها تحيز ضد النساء وضد مناطق معينة، وننظف البيانات الخام، ونخزنها في SQL، وندرب ٥ نماذج، ونقيس العدالة، ونكشف المتغيرات البديلة (proxies)، ونعالج التحيز. ويعرض المشروع النتائج في واجهة ويب بالعربي والإنجليزي، ومعها مساعد ذكي يجاوب من قاعدة البيانات، وملفات Excel وPower BI.

## The question
If a company trains a hiring model on its own past decisions, and those decisions were biased,
**does the model learn the bias, and can we remove it without losing accuracy?**

## Key findings
| | Before (M2 baseline) | After (M5 recommended) |
|---|---|---|
| Gender impact ratio (four-fifths rule ≥ 0.80) | **0.72 – FAIL** | **0.99 – PASS** |
| Region impact ratio | 0.78 – FAIL | 0.85 – PASS |
| Qualified women selected | 65 % (men: 82 %) | 77 % (men: 76 %) |
| AUC vs historical decisions | 0.884 | 0.871 |
| Accuracy vs *true* qualification | 0.837 | **0.850** |

1. **History is biased.** Among equally qualified applicants, men were hired **16.7 percentage points** more often.
2. **Removing gender is not enough.** `career_gap_months` reveals gender with AUC 0.97, so the "unaware" model still fails.
3. **Fairness is almost free.** Removing proxies + reweighing costs 0.013 AUC and *improves* accuracy against the true qualification.
4. **Black boxes don't pay off.** Gradient boosting is no more accurate than an explainable logistic regression.

## Files
| File | What it is |
|---|---|
| `pipeline.py` | The whole data pipeline: synthetic data → messy HR export → **data cleaning (24 logged steps)** → SQLite → 5 models → fairness metrics, proxy detection, explainability → CSV for Power BI, Excel, SQL audit results |
| `app.py` | FastAPI backend (18 REST endpoints) + **AI assistant** (9 SQL-backed tools; Claude tool use with `ANTHROPIC_API_KEY`, otherwise an offline Arabic/English router) |
| `index.html` | Web front end: 9 pages, English / العربية, Chart.js |
| `database.sql` | SQL schema (11 tables), 3 reporting views, 11 named audit queries |
| `test_project.py` | 16 tests: cleaning, fairness metrics, SQL, API, AI assistant |
| `dashboard.html` | **Interactive dashboard** (4 pages, like the Power BI report). Opens in any browser, no software needed |
| `powerbi_measures.dax` | 27 DAX measures for building the same dashboard in Power BI Desktop |
| `FairHire_Data.xlsx` | Raw export sample, cleaning log, data quality, clean data, **Fairness Calculator** (live Excel formulas + charts) |
| `FairHire_Results.xlsx` | Model scorecard (PASS/FAIL formulas, chart), group metrics, proxies, feature importance, M5 predictions |
| `requirements.txt` | Python packages |

## Run it
```bash
pip install -r requirements.txt
python pipeline.py             # creates the output/ folder: data, database, CSV, Excel (~15 s)
uvicorn app:app --reload       # web app on http://127.0.0.1:8000  ·  API docs on /docs
pytest                         # 16 tests
```
Optional: set `ANTHROPIC_API_KEY` before starting the app to run the assistant on Claude.

## How it works
**1. Data.** `pipeline.py` generates 10,000 fictional Saudi job applications. Past decisions contain a planted penalty
for women and for the Southern and Northern regions; the true qualification is known, so we can prove whether bias is removed.

**2. Cleaning.** The data is turned into a realistic messy HR export (10,395 rows): duplicates, test rows, labels like
`F` / `female` / `أنثى`, numbers stored as text (`3,85`, `5 years`), GPA typos (`38.5`), three date formats and personal
identifiers. Cleaning fixes everything in 24 logged steps, **removes name, national ID, mobile and e-mail** (PDPL data
minimisation) and returns 10,000 clean rows. A test proves the result equals the original data exactly.

**3. Models.** Every model shortlists the same share of applicants (35.4 %):

| Model | Features | Mitigation |
|---|---|---|
| M1 | everything | none (gradient boosting, black box) |
| M2 | everything | none (logistic regression) |
| M3 | no gender / nationality / region | "fairness through unawareness" |
| M4 | same as M3 | + reweighing (Kamiran & Calders, 2012) |
| **M5** | job-relevant features only | proxies removed + reweighing |

**4. Web app and AI assistant.** Pages: overview, data cleaning, model comparison, group fairness, proxies &
explainability, applicants, score an applicant (saved to SQL), AI assistant, SQL audit. The assistant answers only from
the database, runs read-only SQL (three layers of protection), refuses requests to exclude people by gender, region or
nationality, and logs every turn in the `agent_logs` table.

**5. Dashboard.** Open `dashboard.html` in a browser: historical bias, model comparison, an interactive model deep dive
(model, attribute and job-family filters) and proxies with an applicant drill-down. To build it in **Power BI**, run `python pipeline.py`, load the CSV files from `output/` in Power BI Desktop
(`dim_applicant`, `dim_model`, `fact_prediction`, `fact_model_metrics`, `fact_group_metrics`, `fact_proxy_scores`,
`fact_feature_importance`), relate `fact_*[model_id]` → `dim_model[model_id]` and `fact_prediction[applicant_id]` →
`dim_applicant[applicant_id]`, then add the measures from `powerbi_measures.dax`.

## Ethics
* **Fairness:** impact ratio (four-fifths rule) and equal-opportunity gap are both reported.
* **Accountability:** every score and every assistant answer is logged in SQL; the system only recommends.
* **Transparency:** interpretable model, per-applicant explanations, the assistant shows the SQL it used.
* **Privacy:** synthetic data only; identifiers are removed during cleaning; the scoring form asks only job-relevant questions.
* **Saudization:** nationality gaps are reported but not treated as a failure, because preferring Saudi nationals is a lawful policy.

## References
- Kamiran, F., & Calders, T. (2012). Data preprocessing techniques for classification without discrimination. *Knowledge and Information Systems, 33*(1).
- Martens, D. (2022). *Data Science Ethics: Concepts, Techniques, and Cautionary Tales.* Oxford University Press.
- Mitchell, M., et al. (2019). Model cards for model reporting. *FAT\* 2019*.

## License
MIT
