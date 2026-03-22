# main.py
from pathlib import Path
from rag_pipeline import RAGPipeline

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VECTOR_DB_PATH = PROJECT_ROOT / "data/vectorstore"

if __name__ == "__main__":
    print(f"VECTOR_DB_PATH: {VECTOR_DB_PATH}")
    exit()
    rag = RAGPipeline(
        persist_directory=str(VECTOR_DB_PATH)
    )

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