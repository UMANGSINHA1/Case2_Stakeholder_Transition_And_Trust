from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import numpy as np
import pandas as pd

@dataclass(frozen=True)
class Config:
    lookback_months: int = 12
    lookforward_months: int = 12
    allowed_gap_days: int = 60
    persistence_months: int = 6
    stockpiling_cap_days: int = 30
    drug_id: str = "TARGET_BRAND_A"

def load_inputs(data_dir="data/raw"):
    d = Path(data_dir)
    return (
        pd.read_csv(d/"pharmacy_claims.csv", parse_dates=["fill_date"]),
        pd.read_csv(d/"enrollment.csv", parse_dates=["enrollment_start","enrollment_end"])
    )

def clean_claims(claims, drug_id="TARGET_BRAND_A"):
    x = claims.copy()
    x = x[(x.claim_status == "PAID") & (x.drug_id == drug_id)].copy()
    x = x.dropna(subset=["patient_id","claim_id","fill_date","days_supply"])
    x["days_supply"] = pd.to_numeric(x.days_supply, errors="coerce")
    x = x[x.days_supply.between(1,365)]
    x = x.drop_duplicates(subset=["claim_id"])
    x["days_supply"] = x["days_supply"].astype(int)
    return x.sort_values(["patient_id","fill_date","claim_id"]).reset_index(drop=True)

def merge_intervals(spans):
    spans = sorted(spans)
    merged = []
    for s,e in spans:
        if not merged or s > merged[-1][1] + pd.Timedelta(days=1):
            merged.append([s,e])
        else:
            merged[-1][1] = max(merged[-1][1],e)
    return merged

def interval_covers(intervals, start, end):
    return any(s <= start and e >= end for s,e in intervals)

def continuous_end_from(intervals, start):
    """End of the continuous coverage interval containing start; NaT if uncovered."""
    for s,e in intervals:
        if s <= start <= e:
            return e
    return pd.NaT


def build_cohort(claims, enrollment, config=Config()):
    x = clean_claims(claims, config.drug_id)
    claim_groups = {pid:g for pid,g in x.groupby("patient_id", sort=False)}
    enr_groups = {pid:merge_intervals(list(zip(g.enrollment_start,g.enrollment_end)))
                  for pid,g in enrollment.groupby("patient_id", sort=False)}
    all_ids = sorted(set(enr_groups) | set(claim_groups))
    out = []
    for pid in all_ids:
        g = claim_groups.get(pid, pd.DataFrame(columns=x.columns))
        intervals = enr_groups.get(pid, [])
        if g.empty:
            out.append({"patient_id":pid,"index_date":pd.NaT,"lookback_eligible":False,
                        "lookforward_eligible":False,"eligible":False,
                        "exclusion_reason":"no_paid_target_drug_claim"})
            continue
        chosen = None
        had_continuous_lookback = False
        had_washout = False
        for r in g.itertuples():
            idx = r.fill_date
            lb_start = idx - pd.DateOffset(months=config.lookback_months)
            lb_end = idx - pd.Timedelta(days=1)
            if not interval_covers(intervals, lb_start, lb_end):
                continue
            had_continuous_lookback = True
            prior = g[(g.fill_date >= lb_start) & (g.fill_date < idx)]
            if prior.empty:
                had_washout = True
                chosen = idx
                break
        if chosen is None:
            reason = ("no_continuous_12m_lookback" if not had_continuous_lookback
                      else "prior_target_drug_use_in_washout")
            out.append({"patient_id":pid,"index_date":pd.NaT,"lookback_eligible":False,
                        "lookforward_eligible":False,"eligible":False,
                        "exclusion_reason":reason})
            continue
        lf_end = chosen + pd.DateOffset(months=config.lookforward_months)
        lf_ok = interval_covers(intervals, chosen, lf_end)
        out.append({"patient_id":pid,"index_date":chosen,"lookback_eligible":True,
                    "lookforward_eligible":bool(lf_ok),"eligible":bool(lf_ok),
                    "exclusion_reason":None if lf_ok else "12_month_lookforward_not_met"})
    return pd.DataFrame(out)

def patient_episodes(claims, cohort, enrollment, config=Config(), require_lookforward=True):
    x = clean_claims(claims, config.drug_id)
    claim_groups = {pid:g for pid,g in x.groupby("patient_id", sort=False)}
    enr_groups = {pid:merge_intervals(list(zip(g.enrollment_start,g.enrollment_end)))
                  for pid,g in enrollment.groupby("patient_id", sort=False)}
    c = cohort[cohort.lookback_eligible == True].copy()
    if require_lookforward:
        c = c[c.eligible == True]
    out = []
    for r in c.itertuples():
        pid, idx = r.patient_id, r.index_date
        g = claim_groups.get(pid, pd.DataFrame(columns=x.columns))
        g = g[g.fill_date >= idx].sort_values("fill_date")
        coverage_end = idx - pd.Timedelta(days=1)
        event_date = pd.NaT
        for claim in g.itertuples():
            fill, days = claim.fill_date, int(claim.days_supply)
            next_uncovered = coverage_end + pd.Timedelta(days=1)
            if fill > next_uncovered:
                gap = (fill-next_uncovered).days
                if gap > config.allowed_gap_days:
                    event_date = next_uncovered + pd.Timedelta(days=config.allowed_gap_days)
                    break
                supply_start = fill
            else:
                early_days = max(0,(next_uncovered-fill).days)
                carry = min(early_days,config.stockpiling_cap_days)
                supply_start = max(fill,next_uncovered-pd.Timedelta(days=carry))
            coverage_end = supply_start + pd.Timedelta(days=days-1)

        intervals = enr_groups.get(pid,[])
        continuous_end = continuous_end_from(intervals,idx)
        admin_end = pd.Timestamp("2026-12-31")
        censor_date = min(continuous_end,admin_end) if not pd.isna(continuous_end) else idx

        # If no later paid refill arrives, the gap can still become observable:
        # discontinuation occurs on the 61st uncovered day if coverage remains
        # observable through that day. If coverage ends first, censor instead.
        if pd.isna(event_date):
            threshold_date = coverage_end + pd.Timedelta(days=config.allowed_gap_days + 1)
            if threshold_date <= censor_date:
                event_date = threshold_date

        if not pd.isna(event_date) and event_date <= censor_date:
            event = 1
            follow_end = event_date
        else:
            event = 0
            follow_end = censor_date
        endpoint = idx + pd.DateOffset(months=config.persistence_months)
        endpoint_observable = interval_covers(intervals,idx,endpoint)
        persistent = bool(endpoint_observable and (pd.isna(event_date) or event_date > endpoint))
        out.append({
            "patient_id":pid,"index_date":idx,"event":event,"event_date":event_date,
            "censor_date":censor_date,"followup_end":follow_end,
            "duration_days":max(0,(follow_end-idx).days),
            "six_month_endpoint":endpoint,"endpoint_observable":endpoint_observable,
            "persistent_6m":persistent
        })
    return pd.DataFrame(out)

def kaplan_meier(data, time_col="duration_days", event_col="event"):
    d = data[[time_col,event_col]].dropna().copy()
    d[time_col] = d[time_col].astype(int)
    rows, surv, greenwood = [], 1.0, 0.0
    for t in sorted(d.loc[d[event_col] == 1,time_col].unique()):
        n = int((d[time_col] >= t).sum())
        events = int(((d[event_col] == 1)&(d[time_col] == t)).sum())
        if n <= 0: continue
        surv *= 1-events/n
        if n > events: greenwood += events/(n*(n-events))
        if 0 < surv < 1 and greenwood > 0:
            se = np.sqrt(greenwood)/abs(np.log(surv))
            z = 1.959963984540054
            lower = np.exp(-np.exp(np.log(-np.log(surv))+z*se))
            upper = np.exp(-np.exp(np.log(-np.log(surv))-z*se))
        else:
            lower = upper = surv
        rows.append({"time_days":t,"at_risk":n,"events":events,"survival":surv,
                     "ci_lower":lower,"ci_upper":upper})
    return pd.DataFrame(rows)


def wilson_interval(successes, total, z=1.959963984540054):
    """Wilson score 95% CI for a binomial proportion, returned as percentages."""
    if total <= 0:
        return (np.nan, np.nan)
    p = successes / total
    denom = 1 + z*z/total
    center = (p + z*z/(2*total)) / denom
    half = z * np.sqrt((p*(1-p)/total) + (z*z/(4*total*total))) / denom
    return (100*(center-half), 100*(center+half))

def summarize_exclusions(cohort):
    """Count mutually exclusive cohort disposition / exclusion reasons."""
    if cohort.empty:
        return pd.DataFrame(columns=["reason","patient_count"])
    reason = cohort["exclusion_reason"].fillna("included_in_original_cohort")
    counts = reason.value_counts(dropna=False).rename_axis("reason").reset_index(name="patient_count")
    return counts

def km_estimate_at(km_table, time_days):
    """Return the right-continuous KM estimate at the requested time."""
    if km_table.empty:
        return 1.0
    eligible = km_table[km_table["time_days"] <= time_days]
    return 1.0 if eligible.empty else float(eligible.iloc[-1]["survival"])

def cross_validate_km(data, time_col="duration_days", event_col="event"):
    """
    Cross-check custom KM output against Lifelines when installed.
    Returns a status object; does not silently claim validation if unavailable.
    """
    ours = kaplan_meier(data, time_col, event_col)
    try:
        from lifelines import KaplanMeierFitter
    except ImportError:
        return {
            "status": "not_run_lifelines_not_installed",
            "max_absolute_difference": None,
            "lifelines_version": None
        }
    fitter = KaplanMeierFitter()
    fitter.fit(data[time_col], event_observed=data[event_col])
    lifelines_survival = fitter.survival_function_.reset_index()
    time_name = lifelines_survival.columns[0]
    survival_name = lifelines_survival.columns[1]
    diffs = []
    for row in ours.itertuples():
        eligible = lifelines_survival[lifelines_survival[time_name] <= row.time_days]
        expected = 1.0 if eligible.empty else float(eligible.iloc[-1][survival_name])
        diffs.append(abs(float(row.survival) - expected))
    max_diff = max(diffs) if diffs else 0.0
    return {
        "status": "passed" if max_diff < 1e-10 else "failed",
        "max_absolute_difference": max_diff,
        "lifelines_version": __import__("lifelines").__version__
    }

def run_sensitivity(claims,enrollment,cohort,base=Config()):
    rows=[]
    for gap in [30,60,90]:
        cfg=Config(base.lookback_months,base.lookforward_months,gap,
                   base.persistence_months,base.stockpiling_cap_days,base.drug_id)
        for require_lf,label in [(True,"12_months"),(False,"six_month_observable")]:
            ep=patient_episodes(claims,cohort,enrollment,cfg,require_lf)
            if not require_lf: ep=ep[ep.endpoint_observable]
            n=len(ep); p=int(ep.persistent_6m.sum()) if n else 0
            rows.append({"scenario":f"gap_{gap}_{label}","gap_days":gap,
                         "lookforward_rule":label,"cohort_n":n,"persistent_n":p,
                         "persistence_pct":100*p/n if n else np.nan})
    return pd.DataFrame(rows)


def run_analysis(data_dir="data/raw",output_dir="outputs"):
    claims,enrollment=load_inputs(data_dir)
    cfg=Config()
    cohort=build_cohort(claims,enrollment,cfg)
    original=patient_episodes(claims,cohort,enrollment,cfg,True)
    survival=patient_episodes(claims,cohort,enrollment,cfg,False)
    km=kaplan_meier(survival)
    sensitivity=run_sensitivity(claims,enrollment,cohort,cfg)

    out=Path(output_dir)
    for sub in ["baseline","sensitivity"]: (out/sub).mkdir(parents=True,exist_ok=True)
    cohort.to_csv(out/"baseline"/"cohort_eligibility.csv",index=False)
    original.to_csv(out/"baseline"/"original_patient_results.csv",index=False)
    survival.to_csv(out/"baseline"/"survival_patient_results.csv",index=False)
    km.to_csv(out/"baseline"/"kaplan_meier.csv",index=False)
    sensitivity.to_csv(out/"sensitivity"/"scenario_summary.csv",index=False)

    exclusion_summary=summarize_exclusions(cohort)
    exclusion_summary.to_csv(out/"baseline"/"cohort_exclusion_summary.csv",index=False)

    n=len(original)
    persistent=int(original.persistent_6m.sum()) if n else 0
    ci_low,ci_high=wilson_interval(persistent,n)
    six_month_days=(pd.Timestamp("2025-07-01")-pd.Timestamp("2025-01-01")).days
    # Calculate per-patient endpoint times and obtain KM at each individual's
    # calendar six-month anniversary through a common 183-day approximation only
    # for a cohort-level descriptive comparison; primary fixed endpoint remains calendar months.
    km_at_6m=km_estimate_at(km,183)

    km_validation=cross_validate_km(survival)
    summary={
        "analysis_version":"1.1.0","synthetic_data_only":True,"config":asdict(cfg),
        "patients_with_claims":int(claims.patient_id.nunique()),
        "candidate_patients":int(cohort.patient_id.nunique()),
        "lookback_eligible":int(cohort.lookback_eligible.sum()),
        "original_cohort_n":n,
        "persistent_6m_n":persistent,
        "persistent_6m_pct":float(100*persistent/n) if n else None,
        "persistent_6m_wilson_95ci_pct":[ci_low,ci_high],
        "exclusion_reason_counts":dict(zip(exclusion_summary.reason,exclusion_summary.patient_count)),
        "survival_cohort_n":int(len(survival)),
        "observed_discontinuations":int(survival.event.sum()),
        "censored_n":int((survival.event==0).sum()),
        "km_survival_at_183_days":km_at_6m,
        "lifelines_cross_validation":km_validation,
        "note":"Synthetic demonstration only; does not reproduce or validate historical 68%."
    }
    (out/"baseline"/"summary.json").write_text(json.dumps(summary,indent=2,default=str))
    return summary,sensitivity

if __name__=="__main__":
    s,ss=run_analysis()
    print(json.dumps(s,indent=2))
    print(ss.to_string(index=False))
