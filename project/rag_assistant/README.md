# RAG Methodology Assistant

A small, auditable RAG service for the medication-persistence POC. It retrieves from a controlled Markdown knowledge base using hybrid BM25 + optional local sentence embeddings, then optionally reranks the shortlist with a local cross-encoder. An OpenAI-compatible model can synthesize a cited answer. If optional models or API credentials are unavailable, the service degrades to lexical retrieval and extractive evidence snippets. Unsupported questions abstain.

## 1. Install and run
From the project root:

```bash
python -m pip install -r rag_assistant/requirements.txt
uvicorn rag_assistant.api:app --reload
```

Open `http://127.0.0.1:8000/docs` for the API playground.

## 2. Optional LLM generation
Set `OPENAI_API_KEY` and optionally `RAG_MODEL` / `OPENAI_BASE_URL`. The default model name is `gpt-4.1-mini`; use a model available to your account. Never put keys in source control. Without a key, the assistant remains usable in extractive RAG mode.

## 3. Example request
```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"How is a discontinuation event defined?"}'
```

## 4. RAG flow
1. User asks a question.
2. Query is tokenized and ranked against approved knowledge-base paragraphs using a transparent BM25-like lexical score.
3. Low-evidence queries are refused.
4. Retrieved chunks are passed to the LLM with strict evidence-only instructions, or returned directly in extractive mode.
5. Citations are restricted to retrieved source IDs.

## 5. Knowledge-base governance
- `knowledge_base/methodology.md`: explicit POC methodology and assumptions.
- `knowledge_base/analysis_results.md`: current synthetic results and validation status.
- `knowledge_base/source_register.json`: source inventory/status.
- Update `analysis_results.md` after each accepted analysis run. Do not let the assistant read arbitrary raw claims or patient-level files.
- Treat proposed assumptions (such as the 30-day stockpiling cap) as assumptions requiring client sign-off.

## 6. Safety and limitations
- This is a case-study POC, not a validated production pharmaceutical system.
- The knowledge base contains synthetic aggregate outputs only; no patient-level data is indexed.
- The assistant is not a medical device and must not give patient-specific medical, dosing, or treatment advice.
- Lexical retrieval is intentionally simple and inspectable. For a larger corpus, evaluate hybrid BM25 + embeddings and reranking against a labelled question set before adding a vector database.
- The abstention threshold is a basic POC guardrail, not a calibrated confidence score.
- An LLM can still make mistakes. Keep citations, human review, versioned source documents, and evaluation tests.
- Provider failures fall back to extractive evidence snippets. Do not report that an LLM generated the answer unless `mode` is `llm_rag`.

## 7. Terminal chat
```bash
python -m rag_assistant.cli
```

## 8. Suggested demo questions
- How is the index date selected?
- Is a 60-day refill gap considered discontinuation?
- What is the six-month persistence estimate in the synthetic baseline?
- How does the persistence estimate change with a 30- or 90-day gap?
- Does the synthetic result validate the historical 68%?
- What dose should a patient take? (Expected: refusal; this assistant is not for clinical advice.)

## 7. Evaluate the assistant

Run:

```bash
python -m rag_assistant.evaluate
python -m unittest rag_assistant.tests.test_rag -v
```

The labelled cases are in `evaluation_set.json`; metrics and per-question retrieval traces are written to `evaluation_report.json`. The current test set is intentionally small and hand-labelled. It measures source Recall@5, source MRR, answerability accuracy, citation validity, and evidence-anchor coverage. It does not establish LLM factuality or production readiness.

## Retrieval ablation benchmark

Run `python -m rag_assistant.benchmark_ablation` to compare BM25, BM25 + LSA semantic retrieval, and BM25 + LSA + a learned logistic-regression reranker on the same 144 labelled questions. See `RETRIEVAL_ABLATION_REPORT.md` for the metrics and caveats. The benchmark's LSA and reranker are reproducible lightweight proxies, not transformer embeddings or a neural cross-encoder.

See `RETRIEVAL_ABLATION_REPORT.md` for the measured comparison and paired bootstrap uncertainty. The current evidence supports keeping BM25 as default rather than enabling heavier retrieval stages by default.

## Negative / out-of-scope evaluation

Run:
```bash
python -m rag_assistant.evaluate_negative
```
This tests 40 questions across clinical advice, patient-level data, unsupported analyses, external medical facts, prompt injection, and unrelated requests. Results are saved to `negative_evaluation_report.json`. The guardrails include explicit scope checks and a methodology-domain gate. These checks are defence-in-depth, not a substitute for model-based adversarial testing or security review.
