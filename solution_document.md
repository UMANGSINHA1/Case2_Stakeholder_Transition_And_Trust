# Reconstructing Medication Persistence — Final Solution Document

**Technical reviewer copy | 3 November 2026 | Analysis baseline 1.1.0**  
**Synthetic demonstration only.** No real patient-level data was used.

## 1. Solution overview
I understood the case as a reproducibility problem: the pharmaceutical client reports 68% six-month medication persistence, but the field team suspects earlier discontinuation and the historical calculation is not fully documented. I built a Python pipeline that generates synthetic pharmacy claims and insurance enrollment, reconstructs an incident-user cohort for `TARGET_BRAND_A`, calculates fixed-cohort persistence and Kaplan–Meier time-to-discontinuation, and produces sensitivity outputs. I also built a standalone HTML dashboard and a FastAPI RAG methodology assistant that retrieves from controlled Markdown documents, cites supporting passages, and abstains on unsupported or unsafe questions. In the synthetic baseline, 766 of 1,099 fixed-cohort patients were persistent at six months (69.70%; Wilson 95% CI 66.92%–72.34%); this does not validate or reproduce the client's historical 68%.

## 2. Approach and methodology
1. **Generate data:** `src/generate_data.py` uses NumPy `default_rng(seed=20261103)` for 1,500 synthetic patients, claims and enrollment spans. It creates `persistent`, `early_stop`, `intermittent`, `coverage_loss` behaviour groups and coverage patterns for continuous coverage, early loss, lookback gaps and lookforward gaps.
2. **Prepare claims:** `src/analysis.py` filters `TARGET_BRAND_A` and `PAID`, deduplicates `claim_id`, validates `days_supply` in [1,365], merges adjacent/overlapping inclusive enrollment spans and derives index dates from eligible observed claims. It does not use `candidate_index_date` as the analytic index.
3. **Define cohort:** first paid target-brand fill after 12 calendar months continuous enrollment and no paid target-brand claim in the 12-month washout. Original fixed cohort requires continuous enrollment through the 12-month index anniversary.
4. **Define persistence:** coverage is inclusive (`fill_date + days_supply - 1`); early refill carry-forward is capped at 30 days; exactly 60 uncovered days are allowed; discontinuation occurs on the 61st uncovered day. Restart does not restore the original episode. Six-month status is assessed at the calendar-month anniversary.
5. **Estimate time-to-event:** fixed-cohort proportion is persistent / original cohort. A separate survival cohort does not require 12-month future enrollment; discontinuation is event, loss of insurance is censoring at last covered day, and administrative censoring is `2026-12-31`. A custom Kaplan–Meier estimator with Greenwood CI is in `src/analysis.py`; Lifelines cross-validation was not run because it was not installed.
6. **Dashboard:** `dashboard/index.html` embeds aggregate outputs and renders KPIs, baseline KM, cohort funnel, sensitivity and assumptions using HTML/CSS/JS/SVG. Controls select precomputed scenarios; they do not recalculate patient-level data or recompute the KM curve.
7. **RAG:** `rag_assistant/api.py` exposes FastAPI `/ask`; `cli.py` is the terminal client; `rag_engine.py` loads approved `methodology.md` and `analysis_results.md`, performs BM25-like retrieval, scope/safety checks and source citation. Optional OpenAI-compatible generation uses `RAG_MODEL` (default `gpt-4.1-mini`) if `OPENAI_API_KEY` is set; otherwise extractive fallback is used. Optional local transformer hooks exist (`all-MiniLM-L6-v2`, `cross-encoder/ms-marco-MiniLM-L-6-v2`) but are not the models evaluated in the reported ablation.

## 3. How I arrived at it
### Alternatives considered
- Fixed proportion only: retained as the business snapshot but paired with Kaplan–Meier to represent event timing and censoring.
- Twelve-month future enrollment for all analyses: applied only to fixed cohort; survival cohort censors loss of coverage.
- Single refill gap: 60 days used provisionally with 30/90-day sensitivity because historical rule was unclear.
- Hybrid/reranker default: rejected; gains were small and bootstrap intervals included zero, so BM25 remains default.
- Vector DB, microservices, React frontend: not added because small controlled corpus and local dashboard did not justify complexity.
- LLM for calculations: rejected; cohort/date/event/CI calculations must be deterministic.
- LLM for methodology Q&A: optional grounded generation because natural-language synthesis helps stakeholders query approved documents.

### Assumptions
- Product is `TARGET_BRAND_A`; switching is not modelled.
- Only `PAID` claims count; `REJECTED`/`REVERSED` excluded. `claim_id` is deduplication key; distinct same-day paid claims remain.
- Valid days supply is 1–365; missing/invalid records need production data-quality reporting.
- Calendar-month anniversaries; inclusive enrollment; adjacent spans merged.
- 30-day stockpiling cap and 60-day gap are proposed assumptions, not confirmed client rules.
- KM assumes non-informative censoring; insurance loss may violate this.
- Synthetic behaviour probabilities are invented, not calibrated to a real population.

### Clarification desk
No clarification-desk transcript, named answers, or client-approved methodology response was included in the supplied artefacts. I did not invent desk answers or treat assumptions as confirmed. Client confirmation is required for product mapping, washout, gap threshold, stockpiling cap, same-day claims, censoring and comparability with the historical 68%.

## 4. Data
`src/generate_data.py` uses seed `20261103`, 1,500 patients, candidate index dates in 2024, four behaviour groups (0.43 persistent, 0.25 early_stop, 0.20 intermittent, 0.12 coverage_loss), coverage patterns, ~8% prior-use washout violations and ~12% rejected/reversed claim noise. Outputs: `data/raw/patients.csv` (1,500 rows), `pharmacy_claims.csv` (12,187 rows), `enrollment.csv` (1,674 spans). These are artificial test records, not actual adherence, payer mix, demographics, adjudication or missingness. They support code testing only, not population inference or validation of 68%. Candidate index date is for audit; analytic index comes from claims.

## 5. Evaluation
### Analytics
- 1,500 patients; 12,187 claims; 1,674 enrollment spans.
- 1,329 lookback eligible; 1,099 original fixed cohort; 766 persistent at six months.
- Six-month persistence 69.70%; Wilson 95% CI 66.92%–72.34%.
- Survival cohort 1,329; 598 observed discontinuations; 731 censored; KM survival at 183 days 69.702%.
- Fixed-cohort exclusions: 230 failed 12-month lookforward, 141 prior target-drug use in washout, 30 no continuous 12-month lookback.
- Sensitivity: 30-day gap 61.24%; 60-day 69.70%; 90-day 75.71%.

### RAG and mid-session update
Positive set: 14 questions (10 answerable, 4 unsupported): Source Recall@5 100%, MRR 93.33%, answerability accuracy 100%, citation validity 100%, evidence-anchor coverage 94.74%.
Negative set: before guardrail update, 24/40 abstentions (60%) and 16 false accepts. After adding clinical/medication, patient-data, unsupported-analysis/external-fact, prompt-injection and domain-scope gates: 40/40 correct abstentions, zero false accepts across six categories. Positive regression was rerun and retained 100% answerability accuracy. These are small hand-authored development sets used during tuning, not unbiased production estimates. Configured live LLM generation was not independently adversarially tested.

### Retrieval ablation
| Configuration | Recall@5 | MRR | Top-1 |
|---|---:|---:|---:|
| BM25 | 83.33% | 74.56% | 64.58% |
| BM25 + TF-IDF/TruncatedSVD (LSA) | 87.50% | 74.77% | 64.58% |
| Hybrid + logistic-regression reranker | 87.50% | 75.55% | 64.58% |

144 hand-authored templated queries (96 train, 48 held-out), 12 topics and 13 paragraphs. Hybrid Recall@5 +4.17 percentage points; reranker MRR +0.78 points over hybrid. Paired bootstrap intervals included zero: hybrid-vs-BM25 Recall@5 [0, 0.1042], MRR [-0.0082, 0.0118]; reranker-vs-hybrid MRR [-0.0004, 0.019]. Topics shared across splits; no unseen-topic or LLM answer factuality evaluation. BM25 remains default.

Tests: `python -m unittest tests.test_analysis -v`; `python -m unittest rag_assistant.tests.test_rag -v`; evaluations `python -m rag_assistant.evaluate`, `python -m rag_assistant.evaluate_negative`, `python -m rag_assistant.benchmark_ablation`. Lifelines validation not run (not installed).

## 6. Tech stack
Python 3; NumPy; Pandas; custom Kaplan–Meier/Greenwood CI; optional Lifelines; HTML/CSS/JavaScript/SVG dashboard; FastAPI and Uvicorn; BM25-like lexical retrieval; scikit-learn TF-IDF, TruncatedSVD and logistic regression for ablation; optional local sentence-transformers and cross-encoder hooks; optional OpenAI-compatible Chat Completions (`gpt-4.1-mini` default). Local CSV and Markdown files; no database, vector DB or microservice deployment. Exact installed versions and successful live LLM call are not recorded.

## 7. Use of LLMs
LLM is not used for claims filtering, cohort logic, date arithmetic, event determination, confidence intervals, KM or dashboard metrics; those are deterministic Python. LLM is optional for natural-language synthesis of retrieved methodology and aggregate results. Context is restricted to approved Markdown passages; raw claims/patient rows are not passed. Evidence-only prompt, source-restricted citations, scope checks, abstention and extractive fallback provide controls. Known failure modes: pattern-based guardrails can miss multilingual/obfuscated/indirect attacks; indirect prompt injection in documents untested; citations do not prove entailment; live model factuality not evaluated; provider outage/latency/context limits can affect output.

## 8. Limitations and next steps
Confirm definitions with client; use governed licensed data; implement minimization, pseudonymization, RBAC, encryption, secrets management, retention/deletion and audit logs; keep identifiers out of LLM context; deploy only in approved environment; complete privacy/security/legal/model-risk review; add CI/CD, dependency scanning, authentication, rate limits, redacted logs and incident response; monitor data quality/drift, metric movement, retrieval, abstention, citations, latency and cost; version prompts/models/knowledge base; cross-check KM with Lifelines and historical client results; expand independent, unseen-topic, multilingual, indirect-injection and LLM factuality evaluation. This is a synthetic POC, not a validated production or clinical system.

## 9. Submission index
- `solution_document.docx` / `.md`: this technical solution and run guide.
- `presentation_beautiful_ai.pdf`: supplied eight-slide Beautiful.ai export, included unchanged.
- `project/README.md`: setup and assumptions. Run from `project/`: `python -m pip install -r requirements.txt`; `python run_project.py`.
- `project/src/generate_data.py`: synthetic data generator.
- `project/src/analysis.py`: cohort, persistence, KM and sensitivity calculations.
- `project/run_project.py`: end-to-end run entry point.
- `project/data/raw/*.csv`: synthetic patient, claims and enrollment files.
- `project/outputs/baseline/*`: baseline cohort, exclusions, KM and summary.
- `project/outputs/sensitivity/scenario_summary.csv`: sensitivity scenarios.
- `project/dashboard/index.html`: open in browser.
- `project/rag_assistant/api.py`: API; install `rag_assistant/requirements.txt`, run `uvicorn rag_assistant.api:app --reload`, open `/docs`.
- `project/rag_assistant/cli.py`: run `python -m rag_assistant.cli`.
- `project/rag_assistant/rag_engine.py`: retrieval, safety, citations, optional LLM and fallback.
- `project/rag_assistant/knowledge_base/*`: approved methodology/results.
- `project/rag_assistant/evaluate.py`, `evaluate_negative.py`, `benchmark_ablation.py`: evaluation scripts.
- `project/rag_assistant/tests/test_rag.py`: run `python -m unittest rag_assistant.tests.test_rag -v`.
- `project/tests/test_analysis.py`: run `python -m unittest tests.test_analysis -v`.
- `project/rag_assistant/*report*`: saved evaluation reports.

Run sequence: create venv; install root requirements; `python run_project.py`; open dashboard; install assistant requirements; run both test commands; launch API or CLI. Set `OPENAI_API_KEY` to enable optional LLM synthesis; without it extractive mode works.

**Source-of-truth files:** `project/outputs/baseline/summary.json`, `project/outputs/sensitivity/scenario_summary.csv`, `project/rag_assistant/evaluation_report.json`, `negative_evaluation_report.json`, and `retrieval_ablation_report.json`.
