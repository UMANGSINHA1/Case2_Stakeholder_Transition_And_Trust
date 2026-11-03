"""Run deterministic retrieval/answerability evaluation: python -m rag_assistant.evaluate."""
import json, re
from pathlib import Path
from .rag_engine import retrieve, answer_question, tokenize

ROOT=Path(__file__).resolve().parent
def run():
    dataset=json.loads((ROOT/"evaluation_set.json").read_text())
    rows=[]; rr=[]; recall=[]; abstain_correct=0; unsupported=len([x for x in dataset["cases"] if not x["answerable"]])
    citation_ok=0; citation_total=0; keyword_hits=0; keyword_total=0
    for case in dataset["cases"]:
        hits=retrieve(case["question"],top_k=5)
        files=[h["file"] for h in hits]
        if case["answerable"]:
            relevant_rank=next((i+1 for i,f in enumerate(files) if f==case["expected_file"]),None)
            rr.append(1/relevant_rank if relevant_rank else 0)
            recall.append(1 if case["expected_file"] in files else 0)
        result=answer_question(case["question"])
        predicted_answerable=not result["abstained"]
        if predicted_answerable == case["answerable"]: abstain_correct+=1
        if result["citations"]:
            citation_total+=1
            allowed={h["source_id"] for h in result["retrieved"]}
            if set(result["citations"]).issubset(allowed): citation_ok+=1
        # Check whether evidence returned to the answer path contains expected factual anchors.
        if case["answerable"] and case["must_contain"]:
            evidence_text=" ".join(h["text"].lower() for h in result["retrieved"])
            for term in case["must_contain"]:
                keyword_total+=1
                if term.lower() in evidence_text: keyword_hits+=1
        rows.append({"id":case["id"],"question":case["question"],
          "expected_answerable":case["answerable"],"predicted_answerable":predicted_answerable,
          "retrieved_files":files,"expected_file_retrieved":(case["expected_file"] in files if case["expected_file"] else None),
          "mode":result["mode"],"citations":result["citations"]})
    n=len(dataset["cases"])
    metrics={
      "evaluation_cases":n,"answerable_cases":len(rr),"unsupported_cases":unsupported,
      "source_recall_at_5":round(sum(recall)/len(recall),4) if recall else None,
      "source_mrr":round(sum(rr)/len(rr),4) if rr else None,
      "answerability_accuracy":round(abstain_correct/n,4),
      "citation_validity":round(citation_ok/citation_total,4) if citation_total else None,
      "evidence_anchor_coverage":round(keyword_hits/keyword_total,4) if keyword_total else None,
      "note":"Small hand-labelled POC set; metrics are directional, not a production benchmark. LLM answer factuality is not measured because this run uses deterministic local evaluation."
    }
    out={"metrics":metrics,"cases":rows}
    (ROOT/"evaluation_report.json").write_text(json.dumps(out,indent=2))
    print(json.dumps(out,indent=2))
if __name__=="__main__": run()
