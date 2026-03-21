# main.py
from pathlib import Path

from dotenv import load_dotenv

from rag_pipeline import RAGPipeline

RAG_DIR = Path(__file__).resolve().parent
REPO_ROOT = RAG_DIR.parent.parent
load_dotenv(REPO_ROOT / ".env")
load_dotenv(RAG_DIR / ".env", override=True)

# Prefer repo root data/vectorstore; fallback to backend/data/vectorstore
VECTOR_DB_PATH = REPO_ROOT / "data" / "vectorstore"
if not VECTOR_DB_PATH.exists():
    BACKEND_ROOT = Path(__file__).resolve().parent.parent
    VECTOR_DB_PATH = BACKEND_ROOT / "data" / "vectorstore"

if __name__ == "__main__":

    rag = RAGPipeline(persist_directory=str(VECTOR_DB_PATH))

    while True:
        query = input("\nPergunta: ")

        if query.lower() in ["exit", "sair"]:
            break

        result = rag.run(query)

        print("\n================ RESPOSTA ================\n")
        print(result["answer"])

        print("\n=============== METADADOS ===============\n")
        for src in result["sources"]:
            print(src)