import io
import os
import secrets
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Cookie, Request, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from pydantic import BaseModel
from pypdf import PdfReader
from docx import Document
from db.rag import rag_query_with_sources

load_dotenv()

app = FastAPI(title="RAG Research Assistant")

APP_USERNAME = os.environ["APP_USERNAME"]
APP_PASSWORD = os.environ["APP_PASSWORD"]

# In-memory session store
active_tokens: set[str] = set()

def get_verified_token(session_token: str | None) -> str:
    if not session_token or session_token not in active_tokens:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return session_token

def extract_text(file: UploadFile) -> str:
    data = file.file.read()
    name = (file.filename or "").lower()
    if name.endswith(".pdf"):
        reader = PdfReader(io.BytesIO(data))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    if name.endswith(".docx"):
        doc = Document(io.BytesIO(data))
        return "\n".join(p.text for p in doc.paragraphs)
    return data.decode("utf-8", errors="ignore")

# --- Auth routes (no protection) ---

@app.get("/login")
def serve_login():
    return FileResponse("frontend/login.html")

class LoginRequest(BaseModel):
    username: str
    password: str

@app.post("/api/login")
def login(req: LoginRequest):
    valid_user = secrets.compare_digest(req.username, APP_USERNAME)
    valid_pass = secrets.compare_digest(req.password, APP_PASSWORD)
    if not (valid_user and valid_pass):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = secrets.token_hex(32)
    active_tokens.add(token)
    response = JSONResponse({"ok": True})
    response.set_cookie("session_token", token, httponly=True, samesite="strict")
    return response

@app.post("/api/logout")
def logout(session_token: str | None = Cookie(default=None)):
    if session_token:
        active_tokens.discard(session_token)
    response = JSONResponse({"ok": True})
    response.delete_cookie("session_token")
    return response

# --- Protected routes ---

@app.get("/")
def serve_frontend(request: Request, session_token: str | None = Cookie(default=None)):
    if not session_token or session_token not in active_tokens:
        return RedirectResponse("/login")
    return FileResponse("frontend/index.html")

class SourceItem(BaseModel):
    text: str
    source: str

class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceItem]

@app.post("/api/query", response_model=QueryResponse)
async def query_endpoint(
    question: str = Form(...),
    files: list[UploadFile] = File(default=[]),
    session_token: str | None = Cookie(default=None),
):
    get_verified_token(session_token)
    if not question.strip():
        raise HTTPException(status_code=422, detail="question must not be empty")
    additional_context = "\n\n".join(extract_text(f) for f in files if f.filename).strip()
    result = rag_query_with_sources(question, additional_context)
    return QueryResponse(answer=result["answer"], sources=[SourceItem(**s) for s in result["sources"]])
