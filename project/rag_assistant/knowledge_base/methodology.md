# Approved POC methodology

## Scope and data
This is a synthetic proof of concept. All patient IDs, pharmacy claims, enrollment spans, and generated outcomes are synthetic. The target product identifier is TARGET_BRAND_A. Brand or active-ingredient switching is not included in the primary analysis.

## Claim eligibility and cleaning
Only claims with status PAID count. REJECTED and REVERSED claims are excluded. claim_id is the deduplication key. Distinct paid claims on the same date are retained. Days supply must be an integer from 1 through 365; missing or invalid values are excluded.

## Index date and washout
An observed initiator is assigned the first paid target-drug fill after 12 calendar months of continuous insurance enrollment and no paid target-drug claim during the preceding 12-month washout. Look-back and look-forward use calendar-month anniversaries. Enrollment dates are inclusive, and adjacent or overlapping enrollment spans are merged.

## Original fixed 6-month cohort
The original cohort requires continuous enrollment from index date through the 12-month index anniversary. Six-month persistence is assessed at the calendar-month anniversary. A patient must be observable through that endpoint and must not have had a discontinuation event on or before it.

## Days supply and refill gap
Supply coverage is inclusive: covered-through date equals fill date plus days supply minus one. Early fills carry forward unused supply, capped at 30 days. This 30-day stockpiling cap is a proposed POC assumption and requires client confirmation. A refill gap of exactly 60 days is allowed. A gap greater than 60 days is discontinuation; the event date is the 61st uncovered day. A restart after a greater-than-60-day gap does not restore persistence in the original episode.

## Survival analysis
The survival cohort requires look-back eligibility but does not require 12 months of future enrollment. Insurance loss is treated as censoring at the last covered day. Administrative censoring is 2026-12-31. The Kaplan-Meier estimator assumes non-informative censoring; this may be violated if insurance loss is related to discontinuation.

## Interpretation limits
The synthetic behaviour mixture and dates are invented solely to create varied test scenarios and are not calibrated to a real drug or population. Synthetic outputs cannot prove, reproduce, or validate the client's historically reported 68% persistence. Production use requires client confirmation of drug mapping, reversal handling, payer changes, enrollment rules, stockpiling, and cohort definitions.
