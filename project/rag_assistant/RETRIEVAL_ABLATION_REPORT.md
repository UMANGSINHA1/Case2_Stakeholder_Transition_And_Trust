# Retrieval ablation evaluation

## Dataset and protocol
- 144 hand-authored, templated supported questions across 12 methodology/result topics.
- 96 training wording variants and 48 held-out test variants (four per topic).
- 13 paragraph-sized knowledge-base chunks.
- All systems use the same documents, questions, and held-out split.
- Gold label: the relevant source paragraph.
- Paired bootstrap confidence intervals use 2,000 resamples of the 48 held-out questions.

## Results

| System | Recall@5 | MRR | Top-1 accuracy |
|---|---:|---:|---:|
| BM25 only | 83.33% | 74.56% | 64.58% |
| BM25 + LSA semantic | 87.50% | 74.77% | 64.58% |
| BM25 + LSA + learned reranker | 87.50% | 75.55% | 64.58% |

## Incremental changes

| Added component | Recall@5 change | MRR change | Top-1 change |
|---|---:|---:|---:|
| LSA semantic vs BM25 | +4.17 pp | +0.21 pp | +0.00 pp |
| Reranker vs hybrid | +0.00 pp | +0.78 pp | +0.00 pp |

## Paired bootstrap 95% confidence intervals for changes

| Comparison | Metric | 95% CI |
|---|---|---:|
| Hybrid vs BM25 | Recall@5 | [0.0, 0.1042] |
| Hybrid vs BM25 | MRR | [-0.0082, 0.0118] |
| Reranker vs hybrid | MRR | [-0.0004, 0.019] |

Intervals include zero. Therefore, this small test does not establish a statistically reliable improvement for either added component.

## Interpretation and decision
- LSA hybrid increases Recall@5 by 4.17 percentage points. MRR changes by only 0.21 points and top-1 accuracy does not change.
- The learned reranker increases MRR by 0.78 points over hybrid, with no gain in Recall@5 or top-1 accuracy.
- The paired confidence intervals include zero. Treat the observed gains as directional, not proven.
- **Recommendation: retain BM25-only as the default for this small knowledge base.** Keep hybrid retrieval as an experiment behind a feature flag. Do not add reranking to the production path until an independent, larger evaluation demonstrates a meaningful improvement.

## Important limitations
- Semantic component: corpus-fitted TF-IDF + TruncatedSVD (LSA), not a pretrained neural embedding model.
- Reranker: supervised logistic-regression pairwise relevance model, not a neural cross-encoder.
- Train/test split holds out wording variants within each topic family. It measures paraphrase robustness for the current knowledge base, not generalization to unseen topics.
- Dataset is hand-authored and templated, not a sample of real user traffic.
- LLM answer factuality and citation entailment are not measured.
- Transformer embedding and neural cross-encoder weights were not available locally, so those exact components were not empirically benchmarked.

## Reproduction
From the project root:
```bash
python -m rag_assistant.benchmark_ablation
```
Outputs `large_evaluation_set.json` and `retrieval_ablation_report.json`.
