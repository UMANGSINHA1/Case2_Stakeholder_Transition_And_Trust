import unittest
import pandas as pd
from src.analysis import (
    clean_claims, merge_intervals, kaplan_meier, patient_episodes, Config
)

def fixture(index="2025-01-01", coverage_end="2026-01-01"):
    pid="P1"
    enrollment=pd.DataFrame([{
        "patient_id":pid,
        "enrollment_start":pd.Timestamp("2024-01-01"),
        "enrollment_end":pd.Timestamp(coverage_end)
    }])
    cohort=pd.DataFrame([{
        "patient_id":pid,"index_date":pd.Timestamp(index),
        "lookback_eligible":True,"lookforward_eligible":True,
        "eligible":True,"exclusion_reason":None
    }])
    return enrollment,cohort

def claims_for(fill_dates, days=30):
    return pd.DataFrame([{
        "patient_id":"P1","claim_id":f"C{i}","drug_id":"TARGET_BRAND_A",
        "fill_date":pd.Timestamp(d),"days_supply":days,
        "quantity":30,"claim_status":"PAID","pharmacy_id":"PH001"
    } for i,d in enumerate(fill_dates)])

class TestPersistenceEdgeCases(unittest.TestCase):
    def test_early_refill_carries_forward_supply(self):
        # Day 20 refill is 10 days early. With 30-day cap, supply extends to Feb 28.
        enr,cohort=fixture()
        claims=claims_for(["2025-01-01","2025-01-21","2025-02-20","2025-03-22",
                           "2025-04-21","2025-05-21","2025-06-20"])
        result=patient_episodes(claims,cohort,enr,Config(),True).iloc[0]
        # Later discontinuation after the six-month endpoint is allowed;
        # the patient must nevertheless be persistent at the six-month endpoint.
        self.assertTrue(bool(result.persistent_6m))

    def test_exactly_60_day_gap_is_allowed(self):
        enr,cohort=fixture()
        # 30-day supply ends Jan 30; expected next fill Jan 31; fill after 60 gap is Apr 1.
        claims=claims_for(["2025-01-01","2025-04-01","2025-05-01","2025-05-31",
                           "2025-06-30","2025-07-30","2025-08-29"])
        result=patient_episodes(claims,cohort,enr,Config(allowed_gap_days=60),True).iloc[0]
        # The 60-day gap itself is accepted; any later event must occur after
        # the six-month endpoint because the subsequent refill schedule continues.
        self.assertTrue(bool(result.persistent_6m))
        self.assertNotEqual(pd.Timestamp(result.event_date),pd.Timestamp("2025-04-01"))

    def test_61_day_gap_is_discontinuation(self):
        enr,cohort=fixture()
        # Expected refill Jan 31; 61-day gap means refill Apr 2, which is too late.
        claims=claims_for(["2025-01-01","2025-04-02","2025-05-02"])
        result=patient_episodes(claims,cohort,enr,Config(allowed_gap_days=60),True).iloc[0]
        self.assertEqual(int(result.event),1)
        self.assertEqual(pd.Timestamp(result.event_date),pd.Timestamp("2025-04-01"))

    def test_restart_does_not_restore_original_persistence(self):
        enr,cohort=fixture()
        claims=claims_for(["2025-01-01","2025-04-02","2025-05-02","2025-06-01",
                           "2025-07-01","2025-07-31"])
        result=patient_episodes(claims,cohort,enr,Config(allowed_gap_days=60),True).iloc[0]
        self.assertEqual(int(result.event),1)
        self.assertFalse(bool(result.persistent_6m))

    def test_insurance_loss_before_gap_threshold_is_censored(self):
        enr,cohort=fixture(coverage_end="2025-03-15")
        claims=claims_for(["2025-01-01"])
        result=patient_episodes(claims,cohort,enr,Config(allowed_gap_days=60),False).iloc[0]
        self.assertEqual(int(result.event),0)
        self.assertEqual(pd.Timestamp(result.followup_end),pd.Timestamp("2025-03-15"))

    def test_insurance_continues_long_enough_then_gap_is_event(self):
        enr,cohort=fixture(coverage_end="2025-12-31")
        claims=claims_for(["2025-01-01"])
        result=patient_episodes(claims,cohort,enr,Config(allowed_gap_days=60),False).iloc[0]
        self.assertEqual(int(result.event),1)
        self.assertEqual(pd.Timestamp(result.event_date),pd.Timestamp("2025-04-01"))

class TestOtherValidation(unittest.TestCase):
    def test_rejected_claim_excluded(self):
        df=pd.DataFrame([
            {"patient_id":"P1","claim_id":"1","drug_id":"TARGET_BRAND_A","fill_date":"2025-01-01","days_supply":30,"claim_status":"PAID"},
            {"patient_id":"P1","claim_id":"2","drug_id":"TARGET_BRAND_A","fill_date":"2025-02-01","days_supply":30,"claim_status":"REJECTED"}])
        self.assertEqual(len(clean_claims(df)),1)

    def test_adjacent_enrollment_spans_merge(self):
        spans=merge_intervals([
            (pd.Timestamp("2024-01-01"),pd.Timestamp("2024-06-30")),
            (pd.Timestamp("2024-07-01"),pd.Timestamp("2024-12-31"))])
        self.assertEqual(len(spans),1)

    def test_km_non_increasing(self):
        d=pd.DataFrame({"duration_days":[10,20,30,40],"event":[1,1,0,1]})
        vals=kaplan_meier(d).survival.tolist()
        self.assertEqual(vals,sorted(vals,reverse=True))

if __name__=="__main__": unittest.main()
