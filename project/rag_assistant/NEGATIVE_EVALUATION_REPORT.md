# Out-of-scope and abstention evaluation

## Test design
The assistant was tested against 40 questions that it should not answer, spanning:
- Clinical advice (10)
- Patient-level data / identification (10)
- Unsupported analytical cuts and causal claims (6)
- External medical facts and regulatory claims (4)
- Prompt injection and attempts to override instructions (5)
- Unrelated questions (5)

## Results
- Correct abstentions: 40/40
- False accepts: 0
- Negative-set abstention rate: 100%

| Category | Correct abstentions | Rate |
|---|---:|---:|
| Clinical | 10/10 | 100% |
| Patient Data | 10/10 | 100% |
| Unsupported Analysis | 6/6 | 100% |
| External Medical | 4/4 | 100% |
| Prompt Injection | 5/5 | 100% |
| Unrelated | 5/5 | 100% |

## Changes made after initial failures
The first run abstained on only 24/40 questions (60%). It incorrectly attempted to answer unrelated requests and some patient-specific or clinical questions because common words overlapped with the methodology corpus. I added:
- Explicit clinical-care and medication recommendation refusal patterns.
- Patient-level data and patient-identification refusal patterns.
- Refusal for unsupported subgroup, provider/payer, causal, efficacy, and external regulatory questions.
- Prompt-injection and instruction-override refusal.
- A methodology-domain gate so unrelated questions do not reach retrieval.

The final run achieved 40/40 abstentions. The 14-case positive/answerability evaluation was rerun and still reports 100% answerability accuracy (10 answerable, 4 unsupported), so the new guardrails did not break those tested supported questions.

## Interpretation and limitations
This is a small hand-authored development benchmark. Guardrail patterns were refined against this same negative set, so 100% is not an unbiased estimate of future performance and should not be presented as production accuracy. The positive set is also small. The configured LLM generation path was not exercised; this tests the local deterministic fallback and pre-retrieval guardrails. Further work should use an independently authored adversarial test set, paraphrases, multilingual questions, indirect prompt injection inside retrieved documents, and human review. A production system should use policy-based scope controls, not rely solely on keyword patterns.
