"""Evaluate abstention on out-of-scope questions."""
import json
from pathlib import Path
from collections import defaultdict
from .rag_engine import answer_question
ROOT=Path(__file__).resolve().parent
def run():
    data=json.loads((ROOT/"negative_evaluation_set.json").read_text())
    rows=[]; by=defaultdict(lambda:{"n":0,"correct":0}); false_accept=[]
    for c in data["cases"]:
        r=answer_question(c["question"])
        correct=bool(r["abstained"]) == bool(c["expected_abstain"])
        by[c["category"]]["n"]+=1; by[c["category"]]["correct"]+=int(correct)
        if not r["abstained"]: false_accept.append(c["id"])
        rows.append({"id":c["id"],"category":c["category"],"question":c["question"],
          "abstained":r["abstained"],"mode":r["mode"],"citation_count":len(r["citations"]),
          "correct":correct})
    total=len(rows); correct=sum(x["correct"] for x in rows)
    report={"metrics":{"negative_test_cases":total,"correct_abstentions":sum(x["abstained"] for x in rows),
      "false_accepts":len(false_accept),"negative_abstention_rate":round(sum(x["abstained"] for x in rows)/total,4),
      "negative_class_accuracy":round(correct/total,4),
      "false_accept_ids":false_accept,
      "by_category":{k:{"n":v["n"],"abstained":v["correct"],"rate":round(v["correct"]/v["n"],4)} for k,v in by.items()},
      "note":"Negative-only evaluation. A high abstention rate is necessary but not sufficient; answerable-question recall must be evaluated alongside it."},
      "cases":rows}
    (ROOT/"negative_evaluation_report.json").write_text(json.dumps(report,indent=2))
    print(json.dumps(report["metrics"],indent=2))
if __name__=="__main__":run()
