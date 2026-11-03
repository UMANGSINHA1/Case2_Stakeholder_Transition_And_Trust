"""Offline retrieval benchmark comparing BM25, BM25+LSA, and BM25+LSA+learned reranking.

The reranker is a lightweight supervised pairwise relevance model, NOT a neural cross-encoder.
It is used here because transformer weights are unavailable in this environment.
"""
from __future__ import annotations
import json, re, math, random
from pathlib import Path
from collections import Counter
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression

ROOT=Path(__file__).resolve().parent
KB=ROOT/"knowledge_base"
TOKEN=re.compile(r"[a-zA-Z0-9_]+")
STOP={"how","what","when","which","does","do","is","are","the","a","an","for","of","to","in","on","and","with","was","were","be","it","this","that","from","after","before","my","our","your","please","explain","tell","me","about"}
def tok(s): return [x.lower() for x in TOKEN.findall(s) if x.lower() not in STOP]
def docs():
    out=[]
    for path in sorted(KB.glob("*.md")):
        ps=[p.strip() for p in re.split(r"\n\s*\n",path.read_text()) if p.strip()]
        for i,p in enumerate(ps,1):
            if p.startswith("# "): continue
            out.append({"id":f"{path.name}#p{i}","file":path.name,"text":p})
    return out

# Each family has a gold paragraph and 12 independently worded question variants.
FAMILIES=[
("methodology.md#p2",[
 "What data is synthetic and which medication identifier is analysed?",
 "Are these real patient records or simulated claims?",
 "Which product code is in scope for the analysis?",
 "Does the analysis include switching to another brand?",
 "What is the data scope and target drug definition?",
 "Are patient IDs and pharmacy transactions real?",
 "Which target brand is covered by this proof of concept?",
 "Does the cohort include active ingredient or brand switches?",
 "Clarify whether the dataset represents actual patients.",
 "What medication identifier does the synthetic dataset use?",
 "Is this a real-world evidence dataset?",
 "Are other brands counted as treatment continuation?"]),
("methodology.md#p3",[
 "Which pharmacy claim statuses count in the analysis?",
 "Are rejected or reversed prescriptions included?",
 "How are duplicate claim records handled?",
 "What is the rule for invalid days supply values?",
 "Do multiple paid fills on one date get removed?",
 "Explain pharmacy transaction cleaning and eligibility.",
 "Are only successfully paid claims retained?",
 "What happens to a claim with 400 days supply?",
 "Which field is used as the deduplication key?",
 "Are reversed transactions counted as medication use?",
 "How are same-day distinct paid transactions treated?",
 "What claim status is excluded from the cohort?"]),
("methodology.md#p4",[
 "How do you determine the index date for a patient?",
 "What is the washout period before the first fill?",
 "How much continuous insurance history is required before initiation?",
 "Can a patient with a paid target drug claim in the prior year qualify?",
 "How is an observed new initiator identified?",
 "What enrollment condition applies to the lookback window?",
 "Is the index date the first observed paid target medication fill?",
 "How are calendar month anniversaries used in washout?",
 "What prior medication history disqualifies a patient?",
 "How long must a patient be continuously insured before index?",
 "Does the patient need a clean 12-month drug history?",
 "Describe the new-user cohort entry rule."]),
("methodology.md#p5",[
 "What enrollment is required for the original six month cohort?",
 "How is six-month persistence assessed in the fixed cohort?",
 "Does the original cohort require future insurance coverage?",
 "What is the persistence endpoint date?",
 "Can a patient lost to follow-up before six months be counted persistent?",
 "What does the 12-month lookforward rule mean?",
 "How is the fixed cohort denominator selected?",
 "Is persistence checked on a calendar-month anniversary?",
 "Does a discontinuation before the endpoint count as persistent?",
 "What future enrollment is needed after index?",
 "Explain original cohort inclusion and six month endpoint.",
 "What happens if coverage ends before the persistence endpoint?"]),
("methodology.md#p6",[
 "How are days supply and early refills accumulated?",
 "Is exactly a 60-day refill gap allowed?",
 "When is a patient considered discontinued after a gap?",
 "What is the stockpiling cap for early refills?",
 "Does restarting after discontinuation restore the original episode?",
 "How is the covered-through date calculated?",
 "Is a 61-day uncovered gap an event?",
 "What is the event date after the allowed refill gap?",
 "Can unused medication carry over to the next fill?",
 "What is the rule for overlapping days supply?",
 "How many days of early refill stock can be carried forward?",
 "Explain the refill gap and discontinuation definition."]),
("methodology.md#p7",[
 "How is insurance loss handled in survival analysis?",
 "What is the Kaplan Meier survival cohort eligibility rule?",
 "Does survival analysis require 12 months of future enrollment?",
 "When is a patient censored due to insurance loss?",
 "What is the administrative censoring date?",
 "What assumption does the Kaplan Meier estimator make about censoring?",
 "How does the survival cohort differ from the fixed cohort?",
 "Are patients censored on their last covered day?",
 "Does loss of insurance count as a discontinuation event?",
 "What is the non-informative censoring assumption?",
 "What future enrollment is needed for the survival cohort?",
 "Explain the handling of insurance disenrollment."]),
("analysis_results.md#p2",[
 "What was the six-month persistence estimate in the baseline?",
 "How many patients persisted at six months?",
 "What was the Wilson confidence interval for persistence?",
 "How large was the original fixed cohort?",
 "What percentage of the cohort remained persistent at six months?",
 "How many original cohort patients were included?",
 "What was the baseline persistence rate and its uncertainty?",
 "What was the synthetic six-month result?",
 "Give the six month persistence numerator and denominator.",
 "What was the reported confidence range?",
 "How many patients were not persistent by six months?",
 "What is the main baseline result?"]),
("analysis_results.md#p3",[
 "How many candidate patients were screened?",
 "How many patients passed the lookback eligibility criteria?",
 "What were the original cohort exclusion counts?",
 "How many people failed the 12-month lookforward rule?",
 "How many had prior target drug use in washout?",
 "What was the cohort construction funnel?",
 "How many candidates were eligible after lookback?",
 "Why were patients excluded from the original cohort?",
 "How many had no continuous 12-month lookback?",
 "What were the mutually exclusive patient dispositions?",
 "How many patients entered the original cohort?",
 "Summarize the cohort eligibility counts."]),
("analysis_results.md#p4",[
 "How many discontinuations were observed in survival analysis?",
 "How many survival cohort patients were censored?",
 "What was Kaplan Meier survival at 183 days?",
 "What was the size of the survival cohort?",
 "How many observed events and censorings were there?",
 "What was the 183-day survival probability?",
 "How many patients had an observed discontinuation event?",
 "How many records were censored because of coverage?",
 "What is the survival analysis baseline summary?",
 "What is the KM estimate at about six months?",
 "How does survival cohort size compare with original cohort?",
 "What are the event and censor counts?"]),
("analysis_results.md#p5",[
 "What did the refill gap sensitivity analysis compare?",
 "What happens to persistence with 30, 60 and 90 day gaps?",
 "Which gap thresholds were tested?",
 "What cohort eligibility approaches were compared in sensitivity analysis?",
 "How does the persistence estimate change when gap threshold increases?",
 "What are the sensitivity scenario results?",
 "Were both fixed and observable cohorts evaluated?",
 "Compare persistence at a 30-day allowed gap versus 90 days.",
 "Which refill-gap assumptions were varied?",
 "What did the scenario table measure?",
 "How does a stricter gap definition affect measured persistence?",
 "Summarize the sensitivity analysis design."]),
("analysis_results.md#p6",[
 "Was the Kaplan Meier result independently checked with Lifelines?",
 "Did the external survival library validation run?",
 "What is the current KM cross validation status?",
 "Has independent Lifelines validation passed?",
 "Was the survival estimate cross checked?",
 "Is the independent validation complete?",
 "What does the Lifelines validation status say?",
 "Can we claim that the KM implementation passed validation?",
 "Which validation step has not yet been completed?",
 "Was a second implementation used to verify Kaplan Meier?",
 "What is the status of independent statistical validation?",
 "Has the Lifelines package comparison been performed?"]),
("analysis_results.md#p7",[
 "Does the synthetic analysis validate the historical 68 percent?",
 "Can these synthetic results prove the client's reported persistence?",
 "What are the limitations of the synthetic result?",
 "Is the historical 68 percent reproduced by this analysis?",
 "Can we claim the old client figure is confirmed?",
 "Are the generated patient behaviours calibrated to a real population?",
 "What does the report say about the 68 percent benchmark?",
 "Can the synthetic cohort establish the true persistence rate?",
 "Are these results real world evidence?",
 "What conclusions can be drawn about the client's historical number?",
 "Does the proof of concept validate the historical methodology?",
 "What is the key interpretation caveat?"])
]

def build_dataset():
    rows=[]
    for family,(gold,qs) in enumerate(FAMILIES):
        for j,q in enumerate(qs):
            rows.append({"id":f"Q{family+1:02d}-{j+1:02d}","question":q,"gold_source":gold,
                         "family":family,"split":"test" if j in (2,5,8,11) else "train"})
    return rows

def bm25_scores(queries, documents):
    corpus=[tok(d["text"]) for d in documents]
    df=Counter()
    for ts in corpus: df.update(set(ts))
    n=len(corpus); avgdl=sum(map(len,corpus))/max(1,n)
    results=[]
    for q in queries:
        qt=tok(q); scores=[]
        for d,ts in zip(documents,corpus):
            tf=Counter(ts); dl=len(ts); score=0.
            for term in qt:
                if tf[term]:
                    idf=math.log(1+(n-df[term]+.5)/(df[term]+.5))
                    k1,b=1.5,.75
                    score+=idf*(tf[term]*(k1+1))/(tf[term]+k1*(1-b+b*dl/max(avgdl,1)))
            scores.append(score)
        results.append(np.array(scores))
    return results

def metric_rows(rankings, rows, docs):
    rrs=[]; recall=[]; top1=[]
    for ranking,row in zip(rankings,rows):
        ids=[docs[i]["id"] for i in ranking]
        rank=(ids.index(row["gold_source"])+1) if row["gold_source"] in ids else None
        rrs.append(1/rank if rank else 0)
        recall.append(1 if row["gold_source"] in ids[:5] else 0)
        top1.append(1 if ids and ids[0]==row["gold_source"] else 0)
    return {"recall_at_5":round(float(np.mean(recall)),4),
            "mrr":round(float(np.mean(rrs)),4),
            "top_1_accuracy":round(float(np.mean(top1)),4)}

def paired_bootstrap_ci(base_rank, challenger_rank, test_rows, documents, metric, reps=2000, seed=17):
    rng=np.random.default_rng(seed)
    def per_query(rankings):
        vals=[]
        for ranking,row in zip(rankings,test_rows):
            ids=[documents[i]["id"] for i in ranking]
            rank=(ids.index(row["gold_source"])+1) if row["gold_source"] in ids else None
            if metric=="recall_at_5": vals.append(1.0 if rank is not None and rank<=5 else 0.0)
            elif metric=="mrr": vals.append(1.0/rank if rank else 0.0)
            else: vals.append(1.0 if rank==1 else 0.0)
        return np.array(vals)
    a=per_query(base_rank); b=per_query(challenger_rank); delta=b-a
    samples=[]
    for _ in range(reps):
        idx=rng.integers(0,len(delta),len(delta))
        samples.append(float(delta[idx].mean()))
    return [round(float(np.quantile(samples,.025)),4),round(float(np.quantile(samples,.975)),4)]

def run():
    documents=docs(); rows=build_dataset()
    (ROOT/"large_evaluation_set.json").write_text(json.dumps(rows,indent=2))
    qs=[r["question"] for r in rows]
    bm=bm25_scores(qs,documents)
    # Word+character TF-IDF latent semantic vectors. This is a reproducible local
    # semantic baseline, not a pretrained transformer embedding.
    corpus=[d["text"] for d in documents]
    vectorizer=TfidfVectorizer(ngram_range=(1,2),min_df=1,sublinear_tf=True)
    X=vectorizer.fit_transform(corpus)
    qX=vectorizer.transform(qs)
    ncomp=max(2,min(32,min(X.shape)-1))
    svd=TruncatedSVD(n_components=ncomp,random_state=42)
    DX=svd.fit_transform(X); QX=svd.transform(qX)
    DX=DX/(np.linalg.norm(DX,axis=1,keepdims=True)+1e-9)
    QX=QX/(np.linalg.norm(QX,axis=1,keepdims=True)+1e-9)
    sem=QX@DX.T
    bm_norm=np.array([v/(v.max()+1e-9) for v in bm])
    sem_norm=np.clip((sem+1)/2,0,1)
    hybrid=.55*bm_norm+.45*sem_norm

    # Train a transparent supervised reranker on train-family question/document pairs.
    # Features: BM25 normalized score, LSA cosine, query-term coverage, phrase overlap,
    # and query/document length ratio. Train split is disjoint by paraphrase row.
    train_pairs=[]; train_y=[]
    for qi,row in enumerate(rows):
        if row["split"]!="train": continue
        qterms=set(tok(row["question"]))
        for di,d in enumerate(documents):
            dterms=set(tok(d["text"]))
            overlap=len(qterms & dterms)/max(1,len(qterms))
            phrase=1.0 if any(" ".join(row["question"].lower().split()[i:i+2]) in d["text"].lower()
                              for i in range(max(0,len(row["question"].split())-1))) else 0.0
            length_ratio=min(len(qterms),len(dterms))/max(1,len(qterms),len(dterms))
            train_pairs.append([bm_norm[qi,di],sem_norm[qi,di],overlap,phrase,length_ratio])
            train_y.append(1 if d["id"]==row["gold_source"] else 0)
    model=LogisticRegression(max_iter=1000,class_weight="balanced",random_state=42)
    model.fit(np.array(train_pairs),np.array(train_y))
    rerank_scores=[]
    for qi,row in enumerate(rows):
        qterms=set(tok(row["question"])); feats=[]
        for di,d in enumerate(documents):
            dterms=set(tok(d["text"]))
            overlap=len(qterms & dterms)/max(1,len(qterms))
            phrase=1.0 if any(" ".join(row["question"].lower().split()[i:i+2]) in d["text"].lower()
                              for i in range(max(0,len(row["question"].split())-1))) else 0.0
            length_ratio=min(len(qterms),len(dterms))/max(1,len(qterms),len(dterms))
            feats.append([bm_norm[qi,di],sem_norm[qi,di],overlap,phrase,length_ratio])
        rerank_scores.append(model.predict_proba(np.array(feats))[:,1])
    rerank_scores=np.array(rerank_scores)

    # Use held-out paraphrase questions only for the headline comparison.
    test_idx=[i for i,r in enumerate(rows) if r["split"]=="test"]
    def rankings(mat,idx):
        return [np.argsort(mat[i])[::-1] for i in idx]
    test_rows=[rows[i] for i in test_idx]
    bm_r=rankings(bm_norm,test_idx)
    hy_r=rankings(hybrid,test_idx)
    rr_r=rankings(rerank_scores,test_idx)
    metrics={"test_questions":len(test_rows),"train_questions":len(rows)-len(test_rows),
      "documents":len(documents),
      "BM25_only":metric_rows(bm_r,test_rows,documents),
      "BM25_plus_LSA_semantic":metric_rows(hy_r,test_rows,documents),
      "BM25_plus_LSA_plus_learned_reranker":metric_rows(rr_r,test_rows,documents),
      "ablation_deltas_vs_BM25":{
        "hybrid_recall_at_5":round(metric_rows(hy_r,test_rows,documents)["recall_at_5"]-metric_rows(bm_r,test_rows,documents)["recall_at_5"],4),
        "hybrid_mrr":round(metric_rows(hy_r,test_rows,documents)["mrr"]-metric_rows(bm_r,test_rows,documents)["mrr"],4),
        "hybrid_top1":round(metric_rows(hy_r,test_rows,documents)["top_1_accuracy"]-metric_rows(bm_r,test_rows,documents)["top_1_accuracy"],4),
        "reranked_recall_at_5":round(metric_rows(rr_r,test_rows,documents)["recall_at_5"]-metric_rows(bm_r,test_rows,documents)["recall_at_5"],4),
        "reranked_mrr":round(metric_rows(rr_r,test_rows,documents)["mrr"]-metric_rows(bm_r,test_rows,documents)["mrr"],4),
        "reranked_top1":round(metric_rows(rr_r,test_rows,documents)["top_1_accuracy"]-metric_rows(bm_r,test_rows,documents)["top_1_accuracy"],4)},
      "paired_bootstrap_95ci_for_deltas":{
        "hybrid_vs_bm25":{
          "recall_at_5":paired_bootstrap_ci(bm_r,hy_r,test_rows,documents,"recall_at_5"),
          "mrr":paired_bootstrap_ci(bm_r,hy_r,test_rows,documents,"mrr"),
          "top_1_accuracy":paired_bootstrap_ci(bm_r,hy_r,test_rows,documents,"top_1_accuracy")},
        "reranker_vs_hybrid":{
          "recall_at_5":paired_bootstrap_ci(hy_r,rr_r,test_rows,documents,"recall_at_5"),
          "mrr":paired_bootstrap_ci(hy_r,rr_r,test_rows,documents,"mrr"),
          "top_1_accuracy":paired_bootstrap_ci(hy_r,rr_r,test_rows,documents,"top_1_accuracy")}},
      "important_caveats":[
       "The 144-query set is hand-authored and templated, not an independent real-user benchmark.",
       "The semantic component is corpus-fitted TF-IDF + TruncatedSVD (LSA), not a pretrained neural embedding model.",
       "The reranker is a supervised logistic-regression pairwise relevance model, not a neural cross-encoder.",
       "Train/test split holds out paraphrase rows within each topic family; topic documents are shared, so this tests wording generalization over the current knowledge base, not generalization to unseen topics.",
       "No LLM generation is evaluated here; retrieval metrics do not prove answer factuality or citation entailment."
      ]}
    report={"metrics":metrics,"test_case_details":[]}
    for i,row in zip(test_idx,test_rows):
        report["test_case_details"].append({"id":row["id"],"question":row["question"],"gold_source":row["gold_source"],
          "bm25_top1":documents[int(np.argmax(bm_norm[i]))]["id"],
          "hybrid_top1":documents[int(np.argmax(hybrid[i]))]["id"],
          "reranked_top1":documents[int(np.argmax(rerank_scores[i]))]["id"]})
    (ROOT/"retrieval_ablation_report.json").write_text(json.dumps(report,indent=2))
    print(json.dumps(report["metrics"],indent=2))
if __name__=="__main__": run()
