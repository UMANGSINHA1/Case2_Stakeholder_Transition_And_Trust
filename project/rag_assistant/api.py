"""FastAPI wrapper. Run: uvicorn rag_assistant.api:app --reload"""
from fastapi import FastAPI
from pydantic import BaseModel, Field
from .rag_engine import answer_question, retrieve, ABSTAIN

app = FastAPI(title="Medication Persistence Methodology Assistant", version="1.0.0")

class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=10)

@app.get("/health")
def health():
    return {"status": "ok", "service": "methodology-rag"}

@app.get("/sources")
def sources():
    from .rag_engine import load_documents
    docs = load_documents()
    seen = {}
    for d in docs:
        seen[d["file"]] = d["file"]
    return {"sources": [{"source_id": k, "title": v} for k, v in seen.items()]}

@app.post("/ask")
def ask(request: AskRequest):
    return answer_question(request.question, request.top_k)
