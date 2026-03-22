import csv
from datetime import datetime
import sys
from pathlib import Path

# adiciona a pasta /rag no path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from rag_pipeline import RAGPipeline
from perguntas_respostas import EVAL_DATA

# ----------------------------------
# CONFIG
# ----------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
VECTOR_DB_PATH = PROJECT_ROOT / "data/vectorstore"
OUTPUT_DIR = PROJECT_ROOT / "data/rag_test"


# ----------------------------------
# MAIN
# ----------------------------------

def gerar_respostas_com_oncosus():

    print("\n🚀 Iniciando avaliação do RAG...\n")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = OUTPUT_DIR / f"rag_{timestamp}.csv"

    rag = RAGPipeline(
        persist_directory=str(VECTOR_DB_PATH),
        llm_model="mistral"
    )

    results = []

    total = len(EVAL_DATA)

    for i, item in enumerate(EVAL_DATA, 1):

        question = item["question"]
        expected = item["expected_answer"]

        print(f"[{i}/{total}] {question}")

        try:
            result = rag.run(question)

            generated = result.get("answer", "")
            sources = result.get("sources", [])

            results.append({
                "question": question,
                "expected_answer": expected,
                "generated_answer": generated,
                "sources": " | ".join(sources)
            })

        except Exception as e:
            results.append({
                "question": question,
                "expected_answer": expected,
                "generated_answer": f"ERROR: {str(e)}",
                "sources": ""
            })

    # salvar CSV
    with open(output_file, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "question",
                "expected_answer",
                "generated_answer",
                "sources"
            ]
        )
        writer.writeheader()
        writer.writerows(results)

    print(f"\n✅ Finalizado!")
    print(f"📁 Arquivo: {output_file}\n")


if __name__ == "__main__":
    gerar_respostas_com_oncosus()