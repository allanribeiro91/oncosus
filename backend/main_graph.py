from pathlib import Path

from rag.rag_pipeline import RAGPipeline
from graph.graph_builder import build_graph

# ----------------------------------
# CONFIG
# ----------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VECTOR_DB_PATH = PROJECT_ROOT / "backend/data/vectorstore"
print(f"VECTOR_DB_PATH: {VECTOR_DB_PATH}")

# ----------------------------------
# INIT
# ----------------------------------

rag = RAGPipeline(
    persist_directory=str(VECTOR_DB_PATH),
    use_reranker=False
)

graph = build_graph(rag_pipeline=rag)

# ----------------------------------
# LOOP
# ----------------------------------

if __name__ == "__main__":
    while True:
        question = input("\nPergunta: ")

        if question.lower() in ["exit", "sair"]:
            break

        result = graph.invoke({
            "question": question
        })

        print("\n================ RESPOSTA ================\n")
        print(result["answer"])

        print("\n================ DEBUG ================\n")
        print("Route:", result.get("route"))
        print("Grounded:", result.get("grounded"))