# app.py - FastAPI API for OncoSUS RAG
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from rag_pipeline import RAGPipeline

# Paths: app.py is in backend/rag/, repo root is backend/rag/../../ = oncosus-novo
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
VECTOR_DB_PATH = REPO_ROOT / "data" / "vectorstore"

# Fallback: if data is under backend/ (as in main.py)
if not VECTOR_DB_PATH.exists():
    BACKEND_ROOT = Path(__file__).resolve().parent.parent
    VECTOR_DB_PATH = BACKEND_ROOT / "data" / "vectorstore"

rag: RAGPipeline | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag
    try:
        rag = RAGPipeline(persist_directory=str(VECTOR_DB_PATH))
        yield
    finally:
        rag = None


app = FastAPI(
    title="OncoSUS API",
    description="API do assistente de perguntas e respostas sobre oncologia (INCA/PCDT)",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:4200",
        "http://127.0.0.1:4200",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    question: str


class ChatResponse(BaseModel):
    question: str
    answer: str
    sources: list[str]


@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    if rag is None:
        raise HTTPException(status_code=503, detail="RAG pipeline não inicializado")
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Pergunta não pode ser vazia")
    try:
        result = rag.run(question)
        return ChatResponse(
            question=result["question"],
            answer=result["answer"],
            sources=result["sources"],
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erro ao processar pergunta: {str(e)}",
        )


@app.get("/api/health")
def health():
    try:
        import ollama
        ollama.list()
        return {"status": "ok", "ollama": "disponível"}
    except Exception as e:
        return {"status": "degraded", "ollama": str(e)}
