from __future__ import annotations

from typing import Any, Callable, Optional

from langgraph.graph import END, START, StateGraph

from .nodes import (
    build_enriched_query,
    check_sufficiency,
    classify_input,
    extract_case_entities,
    fallback_response,
    finalize_output,
    generate_answer,
    retrieve_protocols,
)
from .state import GraphState


def _route_after_classification(state: GraphState) -> str:
    route = state.get("route", "general_question")

    if route == "case_question":
        return "extract_case_entities"

    return "build_enriched_query"


def _route_after_sufficiency(state: GraphState) -> str:
    if state.get("grounded", False):
        return "generate_answer"

    return "fallback_response"


def build_graph(
    rag_pipeline: Any,
    llm_callable: Optional[Callable[[str], str]] = None,
):
    """
    Monta o LangGraph reaproveitando a interface existente do RAGPipeline.

    Fluxo:
    START
      -> classify_input
        -> (case_question) extract_case_entities
        -> build_enriched_query
        -> retrieve_protocols
        -> check_sufficiency
            -> (grounded) generate_answer
            -> (not grounded) fallback_response
        -> finalize_output
      -> END
    """
    graph = StateGraph(GraphState)

    graph.add_node("classify_input", classify_input)
    graph.add_node("extract_case_entities", extract_case_entities)
    graph.add_node("build_enriched_query", build_enriched_query)
    graph.add_node(
        "retrieve_protocols",
        lambda state: retrieve_protocols(state, rag_pipeline),
    )
    graph.add_node("check_sufficiency", check_sufficiency)
    graph.add_node(
        "generate_answer",
        lambda state: generate_answer(
            state=state,
            rag_pipeline=rag_pipeline,
            llm_callable=llm_callable,
        ),
    )
    graph.add_node("fallback_response", fallback_response)
    graph.add_node("finalize_output", finalize_output)

    graph.add_edge(START, "classify_input")

    graph.add_conditional_edges(
        "classify_input",
        _route_after_classification,
        {
            "extract_case_entities": "extract_case_entities",
            "build_enriched_query": "build_enriched_query",
        },
    )

    graph.add_edge("extract_case_entities", "build_enriched_query")
    graph.add_edge("build_enriched_query", "retrieve_protocols")
    graph.add_edge("retrieve_protocols", "check_sufficiency")

    graph.add_conditional_edges(
        "check_sufficiency",
        _route_after_sufficiency,
        {
            "generate_answer": "generate_answer",
            "fallback_response": "fallback_response",
        },
    )

    graph.add_edge("generate_answer", "finalize_output")
    graph.add_edge("fallback_response", "finalize_output")
    graph.add_edge("finalize_output", END)

    return graph.compile()