# app.py - FastAPI API for OncoSUS RAG
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
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

# Dev: ng serve pode cair em outra porta se 4200 estiver ocupada (ex. 64600).
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:4200",
        "http://127.0.0.1:4200",
        "http://localhost:4201",
        "http://127.0.0.1:4201",
        "http://localhost:4300",
        "http://127.0.0.1:4300",
    ],
    allow_origin_regex=r"http://(127\.0\.0\.1|localhost):\d+",
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


def _hint_for_ollama_error(msg: str) -> str:
    m = msg.lower()
    if "more system memory" in m or "requires more system memory" in m:
        return (
            " O modelo no Ollama precisa de mais RAM livre do que o Windows reportou. "
            "Feche navegador (muitas abas), Docker, outros IDEs e o que ocupar RAM; "
            "deixe só API + Ollama + um terminal. Aumente arquivo de paginação. "
            "No .env reduza ainda: ONCOSUS_OLLAMA_NUM_CTX=512 e menos trechos (TOP_K/FINAL_K). "
            "Se continuar sem ~1,5 GiB livres para o llama3.2:1b, só resta um modelo menor no Ollama."
        )
    if "exit status 2" in m or "llama runner" in m or "cpu buffer" in m:
        return (
            " Dicas (Ollama no Windows): (1) PATH: %LOCALAPPDATA%\\Programs\\Ollama\\lib\\ollama. "
            "(2) Atualize o Ollama. "
            "(3) $env:OLLAMA_NUM_GPU='0'; ollama serve (outro terminal: API). "
            "(4) .env: ONCOSUS_OLLAMA_NUM_CTX=512, ONCOSUS_OLLAMA_NUM_BATCH=64, ONCOSUS_TOP_K=4, ONCOSUS_FINAL_K=2."
        )
    if "1455" in m or "paginação" in m:
        return " Ver README: memória virtual / arquivo de paginação ou modelo de embeddings menor."
    return ""


@app.get("/", include_in_schema=False)
def root():
    """Raiz: navegador em / abre a documentação interativa da API."""
    return RedirectResponse(url="/docs")


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
        err = str(e)
        raise HTTPException(
            status_code=500,
            detail=f"Erro ao processar pergunta: {err}.{_hint_for_ollama_error(err)}",
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

        # Evita health travar minutos: o cliente global usa timeout=None.
        ollama.Client(timeout=20.0).list()
        out["ollama"] = "disponível"
    except Exception as e:
        out["ollama"] = str(e)
        if out["status"] == "ok":
            out["status"] = "degraded"
    return out
