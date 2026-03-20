from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from db.rag import rag_query_with_sources

app = FastAPI(title="RAG Research Assistant")

class QueryRequest(BaseModel):
    question: str

class SourceItem(BaseModel):
    text: str
    source: str

class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceItem]

@app.get("/")
def serve_frontend():
    return FileResponse("frontend/index.html")

@app.post("/api/query", response_model=QueryResponse)
def query_endpoint(req: QueryRequest):
    if not req.question.strip():
        raise HTTPException(status_code=422, detail="question must not be empty")
    result = rag_query_with_sources(req.question)
    return QueryResponse(answer=result["answer"], sources=[SourceItem(**s) for s in result["sources"]])
