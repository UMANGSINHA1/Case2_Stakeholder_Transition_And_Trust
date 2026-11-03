# Baseline synthetic analysis results

## Headline result
The original fixed cohort contains 1,099 patients. 766 were persistent at six months, for a persistence estimate of 69.70%. The Wilson 95% confidence interval is 66.92% to 72.34%.

## Cohort construction
There are 1,500 candidate patients and 1,329 look-back eligible patients. Original cohort N is 1,099. Mutually exclusive dispositions:
{
  "included_in_original_cohort": 1099,
  "12_month_lookforward_not_met": 230,
  "prior_target_drug_use_in_washout": 141,
  "no_continuous_12m_lookback": 30
}

## Survival results
The survival cohort contains 1,329 patients, with 598 observed discontinuations and 731 censored observations. Kaplan-Meier survival at 183 days is 69.70%. This is the survival cohort estimate, not necessarily identical to the fixed-cohort six-month proportion.

## Sensitivity analysis
The sensitivity analysis compares 30-, 60-, and 90-day allowed refill gaps under two cohort rules. Exact rows from scenario_summary.csv:
scenario,gap_days,lookforward_rule,cohort_n,persistent_n,persistence_pct
gap_30_12_months,30,12_months,1099,673,61.237488626023655
gap_30_six_month_observable,30,six_month_observable,1230,756,61.46341463414634
gap_60_12_months,60,12_months,1099,766,69.69972702456779
gap_60_six_month_observable,60,six_month_observable,1230,862,70.08130081300813
gap_90_12_months,90,12_months,1099,832,75.70518653321201
gap_90_six_month_observable,90,six_month_observable,1230,931,75.6910569105691


## Validation status
Kaplan-Meier cross-validation status: not_run_lifelines_not_installed. Independent Lifelines validation has not yet been completed in this environment. Do not describe it as passed.

## Important interpretation
All results are synthetic demonstration outputs. They do not reproduce or validate the client-reported historical 68% result.
