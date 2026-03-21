from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

from rag_pipeline import RAGPipeline

# LangGraph
from graph.graph_builder import build_graph


PROJECT_ROOT = Path(__file__).resolve().parent.parent
VECTOR_DB_PATH = PROJECT_ROOT / "data/vectorstore"


def load_patient_case(case_path: Optional[str]) -> Optional[Dict[str, Any]]:
    """
    Carrega um prontuário/caso clínico a partir de um arquivo JSON.

    Aceita:
    - um objeto JSON único
    - uma lista de objetos JSON (neste caso, usa o primeiro)
    """
    if not case_path:
        return None

    path = Path(case_path)

    if not path.exists():
        raise FileNotFoundError(f"Arquivo de caso clínico não encontrado: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        if not data:
            raise ValueError("O arquivo JSON do caso clínico está vazio.")
        return data[0]

    if isinstance(data, dict):
        return data

    raise ValueError("O arquivo JSON deve conter um objeto ou uma lista de objetos.")


def run_linear_mode(rag: RAGPipeline) -> None:
    """
    Mantém o fluxo original do projeto.
    """
    while True:
        query = input("\nPergunta: ").strip()

        if query.lower() in ["exit", "sair"]:
            break

        result = rag.run(query)

        print("\n================ RESPOSTA ================\n")
        print(result["answer"])

        print("\n=============== METADADOS ===============\n")
        for src in result["sources"]:
            print(src)


def run_langgraph_mode(
    rag: RAGPipeline,
    patient_case: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Executa o modo LangGraph em loop interativo.
    """
    app = build_graph(rag_pipeline=rag)

    while True:
        query = input("\nPergunta: ").strip()

        if query.lower() in ["exit", "sair"]:
            break

        state = {
            "question": query,
        }

        if patient_case is not None:
            state["patient_case"] = patient_case

        result = app.invoke(state)

        print("\n================ RESPOSTA ================\n")
        print(result["answer"])

        print("\n=============== METADADOS ===============\n")
        for src in result.get("retrieved_sources", []):
            print(src)

        print("\n================ DEBUG ==================\n")
        print(f"Rota: {result.get('route')}")
        print(f"Grounded: {result.get('grounded')}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Executa o OncoSUS em modo RAG linear ou LangGraph."
    )

    parser.add_argument(
        "--mode",
        choices=["linear", "graph"],
        default="linear",
        help="Seleciona o modo de execução. Padrão: linear",
    )

    parser.add_argument(
        "--case-json",
        type=str,
        default=None,
        help=(
            "Caminho para um arquivo JSON contendo um caso clínico/prontuário. "
            "Usado apenas no modo graph."
        ),
    )

    parser.add_argument(
        "--llm-model",
        type=str,
        default="llama3",
        help="Nome do modelo no Ollama. Padrão: llama3",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=8,
        help="Quantidade inicial de documentos recuperados. Padrão: 8",
    )

    parser.add_argument(
        "--final-k",
        type=int,
        default=4,
        help="Quantidade final de documentos usados no contexto. Padrão: 4",
    )

    return parser


if __name__ == "__main__":
    parser = build_arg_parser()
    args = parser.parse_args()

    rag = RAGPipeline(
        persist_directory=str(VECTOR_DB_PATH),
        llm_model=args.llm_model,
        top_k=args.top_k,
        final_k=args.final_k,
    )

    if args.mode == "linear":
        run_linear_mode(rag)
    else:
        patient_case = load_patient_case(args.case_json)
        run_langgraph_mode(rag, patient_case=patient_case)