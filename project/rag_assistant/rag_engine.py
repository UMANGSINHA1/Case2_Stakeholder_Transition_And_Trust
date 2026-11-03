"""Hybrid RAG: BM25 + optional sentence embeddings + optional cross-encoder reranking."""
from __future__ import annotations
import os, re, math
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parent
KB = ROOT / "knowledge_base"
TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+")
ABSTAIN = ("I can't answer that from the documented methodology or saved analysis results. "
           "The knowledge base does not contain enough evidence to support an answer.")

def tokenize(text):
    return [t.lower() for t in TOKEN_RE.findall(text)]

def load_documents():
    docs=[]
    for path in sorted(KB.glob("*.md")):
        raw=path.read_text(encoding="utf-8")
        paragraphs=[p.strip() for p in re.split(r"\n\s*\n",raw) if p.strip()]
        for i,p in enumerate(paragraphs,1):
            if p.startswith("# "): continue
            docs.append({"source_id":f"{path.name}#p{i}","file":path.name,"text":p,"tokens":tokenize(p)})
    return docs

def _bm25(query, docs):
    q=tokenize(query)
    if not q: return {}
    df=Counter()
    for d in docs: df.update(set(d["tokens"]))
    n=max(1,len(docs)); avgdl=sum(len(d["tokens"]) for d in docs)/n
    scores={}
    for d in docs:
        tf=Counter(d["tokens"]); dl=len(d["tokens"]); score=0.0
        for term in q:
            if not tf[term]: continue
            idf=math.log(1+(n-df[term]+0.5)/(df[term]+0.5))
            k1,b=1.5,0.75
            score += idf*(tf[term]*(k1+1))/(tf[term]+k1*(1-b+b*dl/max(avgdl,1)))
        scores[d["source_id"]]=score
    return scores

def _embedding_scores(query, docs):
    """Use a local/cached SentenceTransformer only; never download implicitly."""
    if os.getenv("RAG_DISABLE_EMBEDDINGS","").lower() in {"1","true","yes"}: return None
    model_name=os.getenv("RAG_EMBEDDING_MODEL","sentence-transformers/all-MiniLM-L6-v2")
    try:
        from sentence_transformers import SentenceTransformer
        # local_files_only avoids hidden downloads and makes deployment reproducible.
        model=SentenceTransformer(model_name, local_files_only=True)
        texts=[d["text"] for d in docs]
        vectors=model.encode([query]+texts, normalize_embeddings=True, convert_to_numpy=True)
        qv=vectors[0]
        return {d["source_id"]:float((vectors[i+1]*qv).sum()) for i,d in enumerate(docs)}
    except Exception:
        return None

def _cross_encoder_rerank(query, candidates):
    """Optional local cross-encoder reranker; candidates should already be a small shortlist."""
    if os.getenv("RAG_DISABLE_RERANKER","").lower() in {"1","true","yes"}: return None
    model_name=os.getenv("RAG_RERANKER_MODEL","cross-encoder/ms-marco-MiniLM-L-6-v2")
    try:
        from sentence_transformers import CrossEncoder
        model=CrossEncoder(model_name, automodel_args={"local_files_only":True})
        pairs=[(query,c["text"]) for c in candidates]
        scores=model.predict(pairs)
        return {c["source_id"]:float(scores[i]) for i,c in enumerate(candidates)}
    except Exception:
        return None

def retrieve(query, top_k=5, candidate_k=12):
    docs=load_documents()
    if not docs or not tokenize(query): return []
    bm=_bm25(query,docs)
    em=_embedding_scores(query,docs)
    # Normalize BM25 to [0,1], cosine similarities from embeddings to [0,1].
    max_b=max(bm.values(),default=0.0)
    combined={}
    by_id={d["source_id"]:d for d in docs}
    for sid in by_id:
        b=(bm.get(sid,0)/max_b) if max_b else 0
        if em is not None:
            e=max(0.0,min(1.0,(em[sid]+1)/2))
            combined[sid]=0.55*b+0.45*e
        else:
            combined[sid]=b
    ids=sorted(combined,key=combined.get,reverse=True)[:max(candidate_k,top_k)]
    candidates=[{**by_id[sid],"score":round(combined[sid],5),
                 "bm25_score":round(bm.get(sid,0),5),
                 "embedding_score":round(em[sid],5) if em is not None else None}
                for sid in ids]
    ce=_cross_encoder_rerank(query,candidates)
    if ce is not None:
        lo=min(ce.values(),default=0); hi=max(ce.values(),default=1)
        for c in candidates:
            norm=(ce[c["source_id"]]-lo)/(hi-lo) if hi>lo else 0.5
            c["reranker_score"]=round(ce[c["source_id"]],5)
            c["score"]=round(0.35*c["score"]+0.65*norm,5)
        candidates.sort(key=lambda x:x["score"],reverse=True)
    else:
        candidates.sort(key=lambda x:x["score"],reverse=True)
    for c in candidates:
        c.pop("tokens",None)
    return candidates[:top_k]

def _llm_answer(question,evidence):
    api_key=os.getenv("OPENAI_API_KEY")
    if not api_key: return None
    try:
        from openai import OpenAI
        client=OpenAI(api_key=api_key,base_url=os.getenv("OPENAI_BASE_URL") or None)
        model=os.getenv("RAG_MODEL","gpt-4.1-mini")
        context="\n\n".join(f"[{e['source_id']}]\n{e['text']}" for e in evidence)
        system=("You are a pharmaceutical medication-persistence methodology assistant. "
          "Answer only from supplied evidence; do not use outside knowledge or infer missing rules. "
          "If evidence is insufficient, explicitly abstain. Distinguish documented rules, POC assumptions, "
          "and synthetic results. Cite each material factual claim with the exact source ID in square brackets. "
          "Never fabricate citations. Do not provide patient-specific medical, dosing, or treatment advice.")
        r=client.chat.completions.create(model=model,temperature=0,
          messages=[{"role":"system","content":system},
                    {"role":"user","content":f"Question: {question}\n\nEvidence:\n{context}"}])
        return r.choices[0].message.content
    except Exception: return None

def _scope_abstention_reason(question):
    q=question.lower()
    # Patient-care requests are outside this analytical assistant's remit.
    clinical_patterns=[
      r"\b(dose|dosage|prescribe|prescription|treatment|treat|diagnos(?:e|is)|symptoms?|contraindication|side effects?)\b",
      r"\b(pregnan(?:t|cy)|breastfeeding|drug interaction|allergic reaction|adverse reaction)\b",
      r"\b(recommend|suggest)\b.{0,50}\b(medication|drug|medicine|therapy|dose)\b",
      r"\bshould\s+(i|we|the patient|a patient)\s+(take|start|stop|continue)\b",
      r"\bwhat\s+medication\s+should\b",
      r"\brecommend(?:ation)?\s+(a\s+)?dose\b"
    ]
    if any(re.search(p,q) for p in clinical_patterns):
        return "safety_abstain"
    # Questions requiring data or evidence that this knowledge base does not contain.
    out_of_scope=[
      r"\b(blood pressure|heart rate|lab results?|lab values?|a1c|creatinine|weight|bmi)\b",
      r"\b(fda|ema)\b.*\b(approved|indication|label|warning)\b",
      r"\b(insurance provider|payer|pharmacy|hospital|doctor|physician)\b.*\b(best|worst|rank|compare|performance|outcome)",
      r"\b(by|per|among|across)\s+(age|gender|sex|race|ethnicity|state|country|payer|provider|insurance plan)\b",
      r"\b(subgroup|demographic|geographic|provider[- ]level|payer[- ]level)\b",
      r"\b(by|per|across|among)\s+(state|region|country|age|gender|sex|race|ethnicity)\b",
      r"\b(real[- ]world|actual|real)\s+(patient|clinical|claims?|efficacy|effectiveness)\b",
      r"\b(efficacy|effectiveness|safety|hospitalization|mortality)\b",
      r"\b(patient|member)\s+(name|address|phone|email|date of birth|dob|id|identifier)\b",
      r"\b(names?|phone numbers?|addresses?|date of birth|dob|patient ids?)\b",
      r"\b(identify|identify which|who is|list|show)\b.{0,50}\b(patient|member|individual)\b",
      r"\b(longest|largest|highest|lowest)\b.{0,50}\b(patient|member)\b",
      r"\bwhich\s+patient\b",
      r"\bpatient\b.{0,50}\b(id|identifier|name)\b",
      r"\b(state|region|country)\b.{0,40}\b(highest|lowest|most|least|rate|ranking)\b",
      r"\bwho\s+is\s+patient\b",
      r"\b(predict|forecast)\b.*\b(patient|individual|next refill|next discontinuation)\b",
      r"\b(cause|caused by|causality|causal effect|treatment effect)\b",
      r"\b(clinical efficacy|safety profile|adverse event|mortality|hospitalization)\b",
    ]
    if any(re.search(p,q) for p in out_of_scope):
        return "scope_abstain"
    # Attempts to override instructions or extract hidden prompts are never treated as data questions.
    if re.search(r"\b(write|compose|create)\b.{0,30}\b(poem|story|joke|song|fiction)\b",q):
        return "out_of_domain_abstain"
    injection=[
      r"\b(ignore|disregard|override)\b.{0,50}\b(instruction|rule|system prompt|policy)\b",
      r"\b(reveal|show|print|repeat)\b.{0,40}\b(system prompt|hidden prompt|secret instruction|api key|environment variable)\b",
      r"\bexecute\b.{0,40}\b(sql|python|shell|code)\b",
      r"\b(invent|fabricate|make up)\b.{0,40}\b(result|evidence|answer|data)\b",
      r"\b(use|rely on)\b.{0,40}\b(general knowledge|outside knowledge)\b",
    ]
    if any(re.search(p,q) for p in injection):
        return "security_abstain"
    return None

def answer_question(question,top_k=5):
    reason=_scope_abstention_reason(question)
    if reason:
        return {"answer":ABSTAIN,"abstained":True,"mode":reason,"citations":[],"retrieved":[]}
    domain_anchors={"medication","drug","pharmacy","claim","claims","cohort","persistence","persistent",
      "refill","gap","discontinuation","discontinue","index","washout","enrollment","insurance",
      "survival","kaplan","meier","censor","censored","synthetic","sensitivity","days","supply",
      "paid","reversed","rejected","stockpiling","lookback","lookforward","six","month","months",
      "baseline","confidence","interval","wilson","lifelines","target_brand_a"}
    if not (set(tokenize(question)) & domain_anchors):
        return {"answer":ABSTAIN,"abstained":True,"mode":"out_of_domain_abstain","citations":[],"retrieved":[]}
    evidence=retrieve(question,top_k=max(top_k,8))
    # Evidence threshold plus lexical overlap prevents generic corpus terms from
    # turning unrelated questions into apparently grounded answers.
    q=set(tokenize(question))
    lexical_overlap=bool(q & set(t for e in evidence for t in tokenize(e["text"])))
    if not evidence or not lexical_overlap or evidence[0]["score"]<0.12:
        return {"answer":ABSTAIN,"abstained":True,"mode":"abstain","citations":[],"retrieved":[]}
    llm=_llm_answer(question,evidence)
    if llm:
        allowed={e["source_id"] for e in evidence}
        cited=re.findall(r"\[([^\]]+)\]",llm)
        if cited and all(c in allowed for c in cited):
            return {"answer":llm,"abstained":False,"mode":"llm_rag",
                    "citations":sorted(set(cited)),"retrieved":evidence}
    snippets=[f"{e['text']} [{e['source_id']}]" for e in evidence[:3]]
    return {"answer":"Based on the available documentation:\n\n"+"\n\n".join(snippets),
            "abstained":False,"mode":"extractive_rag",
            "citations":[e["source_id"] for e in evidence[:3]],"retrieved":evidence[:3]}
