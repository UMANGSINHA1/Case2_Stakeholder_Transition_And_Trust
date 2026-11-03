# Medication Persistence Analysis — Synthetic POC

## Run
```bash
python -m pip install -r requirements.txt
python run_project.py
```

The script generates synthetic data under `data/raw/` and writes analysis outputs under `outputs/`.

## Explicit assumptions (v1.0)
1. All records and patient IDs are synthetic; no real patient data is used.
2. The target product is `TARGET_BRAND_A`; brand/active-ingredient switching is not included in the primary analysis.
3. Only `PAID` claims count. `REJECTED` and `REVERSED` claims are excluded.
4. `claim_id` is the deduplication key; distinct paid claims on the same date are retained.
5. Days supply must be an integer from 1 through 365. Missing or invalid days supply is excluded and should be reported in a production data-quality report.
6. A new observed initiator is the first paid target-drug fill after 12 calendar months of continuous enrolment and no paid target-drug claim in the preceding 12 months.
7. Look-back and look-forward periods use calendar-month anniversaries. Enrolment spans are inclusive; adjacent spans are merged.
8. The original cohort requires continuous enrolment from index date through the 12-month index anniversary.
9. Medication supply is inclusive: covered-through = fill date + days supply - 1.
10. Early fills carry forward unused supply, capped at 30 days. This is a proposed assumption, not specified in Exhibit 1.
11. A refill gap of exactly 60 days is allowed; >60 days is discontinuation. The event date is the 61st uncovered day.
12. A restart after a >60-day gap does not restore persistence in the original episode.
13. Six-month persistence is assessed at the calendar-month anniversary. A patient must be observable through that endpoint and must not have had a discontinuation event on or before it.
14. The survival cohort requires look-back eligibility but not 12-month future enrolment. Insurance loss is treated as censoring at the last covered day.
15. Administrative censoring is at 2026-12-31. The synthetic generator's candidate index dates are in 2024.
16. Kaplan–Meier assumes non-informative censoring. This may be violated if insurance loss is related to discontinuation.
17. The synthetic behaviour mixture and dates are invented solely to create varied test scenarios. They are not calibrated to any real drug or population.
18. Synthetic outputs are not evidence that the historical 68% is correct or reproducible.

## Important limitations in this demonstration
- The synthetic data generator includes coverage gaps and prior-use patterns to exercise exclusion logic.
- The demonstration is a transparent POC, not a validated production claims adjudication system.
- Drug mapping, days-supply corrections, reversals, same-day multi-claim handling, payer changes, and real-world stockpiling require client confirmation.
- A strict implementation should additionally report missing/invalid records and all exclusion counts by reason.


## Implementation notes
- The cohort builder searches for the first paid target-drug fill preceded by 12 calendar months of continuous enrolment and no paid target-drug fills in that look-back.
- The survival cohort uses the same look-back/new-observed-initiator rule but does not require 12 months of future enrolment.
- For survival, follow-up is censored at the end of the continuous enrolment interval containing the index date (or the administrative data end).
- This POC uses a custom, inspectable Kaplan–Meier implementation with Greenwood confidence intervals. It should be cross-checked against Lifelines before production sign-off.
- The generator includes a candidate index date in `patients.csv` for audit/ground-truth inspection; the analysis does not use that column to identify the index date.

20. A patient with no subsequent refill is still classified as discontinued if the 61st uncovered day occurs before the end of continuous observable enrolment; otherwise follow-up is censored.


## Validation additions (v1.1)
- Edge-case tests cover early refill stockpiling, exactly 60-day gap, 61-day gap, restart after discontinuation, and insurance loss before/after the discontinuation threshold.
- Custom Kaplan–Meier output is cross-validated against Lifelines when installed. The run summary records `passed`, `failed`, or `not_run_lifelines_not_installed`.
- Fixed six-month persistence now includes a Wilson score 95% confidence interval.
- `cohort_exclusion_summary.csv` reports mutually exclusive dispositions: included, no paid target-drug claim, no continuous 12-month look-back, prior target-drug use during washout, and failure of 12-month look-forward.
- Install `lifelines` from requirements before running to perform the independent Kaplan–Meier cross-check.

## Interactive dashboard
Open `dashboard/index.html` in a browser. It is a standalone HTML explorer using aggregate synthetic outputs. See `dashboard/README.md` for the case-study requirement mapping and limitations.


## RAG methodology assistant
See `rag_assistant/README.md` for setup, API usage, evidence retrieval, citations, and abstention behaviour.
