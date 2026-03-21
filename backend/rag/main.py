# main.py
from pathlib import Path
from rag_pipeline import RAGPipeline

# Prefer repo root data/vectorstore; fallback to backend/data/vectorstore
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
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