# app.py - FastAPI API for OncoSUS RAG
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from rag_pipeline import RAGPipeline

logger = logging.getLogger("uvicorn.error")

# Paths: app.py is in backend/rag/, repo root is backend/rag/../../ = oncosus-novo
RAG_DIR = Path(__file__).resolve().parent
REPO_ROOT = RAG_DIR.parent.parent
VECTOR_DB_PATH = REPO_ROOT / "data" / "vectorstore"

# Fallback: if data is under backend/ (as in main.py)
if not VECTOR_DB_PATH.exists():
    BACKEND_ROOT = RAG_DIR.parent
    VECTOR_DB_PATH = BACKEND_ROOT / "data" / "vectorstore"

rag: RAGPipeline | None = None
_startup_error: str | None = None


def _load_env() -> None:
    load_dotenv(REPO_ROOT / ".env")
    load_dotenv(RAG_DIR / ".env", override=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag, _startup_error
    _load_env()
    _startup_error = None
    rag = None
    if not VECTOR_DB_PATH.exists():
        _startup_error = (
            f"Vector store não encontrado em {VECTOR_DB_PATH}. "
            "Rode backend/scripts (ex.: step_4_0_embed_chunks.py) para gerar a base."
        )
        logger.error(_startup_error)
    else:
        try:
            rag = RAGPipeline(persist_directory=str(VECTOR_DB_PATH))
        except Exception as e:
            _startup_error = str(e)
            logger.exception("Falha ao inicializar RAGPipeline")
    try:
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
        detail = "RAG pipeline não inicializado."
        if _startup_error:
            detail = f"{detail} {_startup_error}"
        raise HTTPException(status_code=503, detail=detail)
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
    out: dict = {
        "status": "ok" if rag is not None else "unavailable",
        "vectorstore_path": str(VECTOR_DB_PATH),
        "vectorstore_exists": VECTOR_DB_PATH.exists(),
        "ollama_model": os.environ.get("ONCOSUS_OLLAMA_MODEL") or "llama3",
    }
    if _startup_error:
        out["startup_error"] = _startup_error
    try:
        import ollama

        ollama.list()
        out["ollama"] = "disponível"
    except Exception as e:
        out["ollama"] = str(e)
        if out["status"] == "ok":
            out["status"] = "degraded"
    return out
