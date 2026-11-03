# Interactive HTML dashboard

Open `dashboard/index.html` in a modern browser. It is self-contained and embeds only aggregate synthetic outputs; it does not require a backend or external chart library.

## Case-study requirement mapping
- Rebuild six-month persistence under original rules: baseline metric and methodology panel.
- Inspect cohort methodology: cohort construction and exclusion-reason funnel.
- Time-to-discontinuation: Kaplan–Meier step curve, observed events and censoring summary.
- Sensitivity analysis: interactive 30/60/90-day refill gap and original vs six-month-observable cohort selectors.
- Explain assumptions and limitations: methodology and validation panels.
- Traceability: analysis version, synthetic-data disclaimer and cross-validation status.

## Deliberate limitations
- The dashboard only selects among precomputed scenarios; it does not run new patient-level calculations in the browser.
- The Kaplan–Meier curve is the baseline survival cohort and is not recalculated when the scenario controls change.
- The 68% historical number is a labelled reference line, not a synthetic target or verified comparator.
- The LLM assistant is a separate next deliverable and is not embedded here.
