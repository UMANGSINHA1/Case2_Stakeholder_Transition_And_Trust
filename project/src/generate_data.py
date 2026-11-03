from __future__ import annotations
import numpy as np
import pandas as pd

SEED = 20261103
N_PATIENTS = 1500
DATA_END = pd.Timestamp("2026-12-31")

def _make_enrollment(pid, index_date, rng, coverage_type):
    """Create one or two enrollment spans; dates are inclusive."""
    start = index_date - pd.DateOffset(months=12)
    end = index_date + pd.DateOffset(months=12)
    if coverage_type == "continuous":
        return [{"patient_id": pid, "enrollment_start": start, "enrollment_end": end,
                 "plan_id": f"PLAN{rng.integers(1, 5)}"}]
    if coverage_type == "early_loss":
        loss = index_date + pd.Timedelta(days=int(rng.integers(90, 300)))
        return [{"patient_id": pid, "enrollment_start": start, "enrollment_end": loss,
                 "plan_id": f"PLAN{rng.integers(1, 5)}"}]
    if coverage_type == "lookback_gap":
        gap_start = index_date - pd.Timedelta(days=int(rng.integers(60, 250)))
        gap_end = gap_start + pd.Timedelta(days=int(rng.integers(5, 25)))
        return [
            {"patient_id": pid, "enrollment_start": start, "enrollment_end": gap_start - pd.Timedelta(days=1),
             "plan_id": f"PLAN{rng.integers(1, 5)}"},
            {"patient_id": pid, "enrollment_start": gap_end + pd.Timedelta(days=1), "enrollment_end": end,
             "plan_id": f"PLAN{rng.integers(1, 5)}"}
        ]
    if coverage_type == "lookforward_gap":
        gap_start = index_date + pd.Timedelta(days=int(rng.integers(90, 300)))
        gap_end = gap_start + pd.Timedelta(days=int(rng.integers(5, 25)))
        return [
            {"patient_id": pid, "enrollment_start": start, "enrollment_end": gap_start - pd.Timedelta(days=1),
             "plan_id": f"PLAN{rng.integers(1, 5)}"},
            {"patient_id": pid, "enrollment_start": gap_end + pd.Timedelta(days=1), "enrollment_end": end,
             "plan_id": f"PLAN{rng.integers(1, 5)}"}
        ]
    return []

def generate_synthetic_data(n_patients=N_PATIENTS, seed=SEED):
    rng = np.random.default_rng(seed)
    # Candidate initiation dates leave enough follow-up before the dataset end.
    start = pd.Timestamp("2024-01-01")
    end = pd.Timestamp("2024-12-31")
    span = (end - start).days
    patients, enrollments, claims = [], [], []
    claim_counter = 0

    for i in range(n_patients):
        pid = f"SYN{i+1:06d}"
        index_date = start + pd.Timedelta(days=int(rng.integers(0, span + 1)))

        # Behaviour mix: persistent, early discontinuation, intermittent, and coverage loss.
        behaviour = rng.choice(
            ["persistent", "early_stop", "intermittent", "coverage_loss"],
            p=[0.43, 0.25, 0.20, 0.12]
        )
        coverage_type = rng.choice(
            ["continuous", "early_loss", "lookback_gap", "lookforward_gap"],
            p=[0.78, 0.10, 0.06, 0.06]
        )
        # Some patients have prior fills inside the 12-month washout.
        prior_use = rng.random() < 0.08

        patients.append({
            "patient_id": pid,
            "synthetic_behaviour_group": behaviour,
            "has_prior_use_in_lookback": bool(prior_use),
            "candidate_index_date": index_date
        })
        enrollments.extend(_make_enrollment(pid, index_date, rng, coverage_type))

        # Generate prior target-drug claims for washout violations.
        if prior_use:
            for d in [index_date - pd.Timedelta(days=210), index_date - pd.Timedelta(days=180)]:
                claim_counter += 1
                claims.append({
                    "patient_id": pid, "claim_id": f"CLM{claim_counter:08d}",
                    "drug_id": "TARGET_BRAND_A", "fill_date": d,
                    "days_supply": 30, "quantity": 30, "claim_status": "PAID",
                    "pharmacy_id": f"PH{int(rng.integers(1, 80)):03d}"
                })

        # Some rejected/reversed noise claims, which should not drive persistence.
        if rng.random() < 0.12:
            claim_counter += 1
            claims.append({
                "patient_id": pid, "claim_id": f"CLM{claim_counter:08d}",
                "drug_id": "TARGET_BRAND_A", "fill_date": index_date + pd.Timedelta(days=30),
                "days_supply": 30, "quantity": 30,
                "claim_status": rng.choice(["REJECTED", "REVERSED"]),
                "pharmacy_id": f"PH{int(rng.integers(1, 80)):03d}"
            })

        # Paid fills. A refill schedule with behaviour-specific gaps.
        fill_dates = [index_date]
        max_days = min(365, max(0, (DATA_END - index_date).days))
        if behaviour == "persistent":
            while (fill_dates[-1] - index_date).days < max_days:
                fill_dates.append(fill_dates[-1] + pd.Timedelta(days=int(rng.integers(27, 34))))
        elif behaviour == "early_stop":
            # Usually one or two fills, then no refill.
            n_extra = int(rng.integers(0, 2))
            for _ in range(n_extra):
                fill_dates.append(fill_dates[-1] + pd.Timedelta(days=int(rng.integers(27, 34))))
        elif behaviour == "intermittent":
            while (fill_dates[-1] - index_date).days < max_days:
                gap = int(rng.choice([35, 45, 55, 75, 95], p=[0.15, 0.25, 0.25, 0.20, 0.15]))
                fill_dates.append(fill_dates[-1] + pd.Timedelta(days=gap))
        elif behaviour == "coverage_loss":
            while (fill_dates[-1] - index_date).days < min(max_days, 180):
                fill_dates.append(fill_dates[-1] + pd.Timedelta(days=int(rng.integers(27, 34))))

        # Do not create paid claims after insurance coverage has ended or in a coverage gap.
        spans = [e for e in enrollments if e["patient_id"] == pid]
        for fd in fill_dates:
            if any(e["enrollment_start"] <= fd <= e["enrollment_end"] for e in spans):
                claim_counter += 1
                claims.append({
                    "patient_id": pid, "claim_id": f"CLM{claim_counter:08d}",
                    "drug_id": "TARGET_BRAND_A", "fill_date": fd,
                    "days_supply": int(rng.choice([28, 30, 30, 30, 31])),
                    "quantity": 30, "claim_status": "PAID",
                    "pharmacy_id": f"PH{int(rng.integers(1, 80)):03d}"
                })

    patients_df = pd.DataFrame(patients)
    enrollment_df = pd.DataFrame(enrollments)
    claims_df = pd.DataFrame(claims)
    for df, cols in [(enrollment_df, ["enrollment_start", "enrollment_end"]),
                     (claims_df, ["fill_date"]),
                     (patients_df, ["candidate_index_date"])]:
        for col in cols:
            df[col] = pd.to_datetime(df[col])
    return patients_df, claims_df, enrollment_df

if __name__ == "__main__":
    p, c, e = generate_synthetic_data()
    p.to_csv("data/raw/patients.csv", index=False)
    c.to_csv("data/raw/pharmacy_claims.csv", index=False, date_format="%Y-%m-%d")
    e.to_csv("data/raw/enrollment.csv", index=False, date_format="%Y-%m-%d")
    print(f"Generated {len(p):,} patients, {len(c):,} claims, {len(e):,} enrollment spans.")
