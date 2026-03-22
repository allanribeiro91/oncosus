from typing import Any, Dict, List, Optional, TypedDict


class GraphState(TypedDict, total=False):
    # Entrada principal
    question: str

    # Caso clínico opcional
    patient_case: Optional[Dict[str, Any]]

    # Roteamento
    route: str

    # Entidades clínicas extraídas do caso
    extracted_entities: Dict[str, Any]

    # Retrieval / contexto
    retrieved_docs: List[Any]
    retrieved_sources: List[str]
    context: str
    enriched_query: str

    # Controle de qualidade / fallback
    grounded: bool
    fallback_reason: str

    # Saída final
    answer: str