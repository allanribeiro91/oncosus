from __future__ import annotations

import re
from graph.patient_repository import find_patient_by_name


from typing import Any, Callable, Dict, List, Optional

from .state import GraphState


SAFE_FALLBACK_MESSAGE = (
    "Os trechos recuperados não contêm orientação específica suficiente "
    "para responder com segurança."
)

def resolve_patient(state: GraphState) -> GraphState:
    question = state.get("question", "")

    # ----------------------------------
    # 🔢 1. TENTAR POR ID
    # ----------------------------------
    id_match = re.search(r"(?:paciente\s*|id\s*)(\d+)", question.lower())

    if id_match:
        patient_id = int(id_match.group(1))
        patient = find_patient_by_id(patient_id)

        if patient:
            return {"patient_case": patient}

    # ----------------------------------
    # 🔤 2. TENTAR POR NOME
    # ----------------------------------
    name_match = re.search(
        r"\b[A-ZÁÉÍÓÚÂÊÔÃÕÇ][a-záéíóúâêôãõç]+(?:\s[A-ZÁÉÍÓÚÂÊÔÃÕÇ][a-záéíóúâêôãõç]+)+",
        question
    )

    if name_match:
        name = name_match.group(0)
        patient = find_patient_by_name(name)

        if patient:
            return {"patient_case": patient}

    return {}

def classify_input(state: GraphState) -> GraphState:
    """
    Decide se a pergunta é:
    - baseada em caso clínico (há patient_case)
    - pergunta protocolar / SUS / INCA / PCDT
    - pergunta geral
    """
    question = (state.get("question") or "").strip().lower()
    patient_case = state.get("patient_case")

    if patient_case:
        return {"route": "case_question"}

    protocol_keywords = [
        "protocolo",
        "pcdt",
        "sus",
        "diretriz",
        "diretrizes",
        "linha de cuidado",
        "tratamento",
        "quimioterapia",
        "radioterapia",
        "cirurgia",
        "estadiamento",
        "neoplasia",
        "oncologia",
    ]

    if any(keyword in question for keyword in protocol_keywords):
        return {"route": "protocol_question"}

    return {"route": "general_question"}


def extract_case_entities(state: GraphState) -> GraphState:
    """
    Extrai campos clínicos úteis de um prontuário/caso.
    O método é tolerante a chaves ausentes.
    """
    case = state.get("patient_case") or {}

    extracted = {
        "id": case.get("id"),
        "nome": case.get("nome"),
        "idade": case.get("idade"),
        "sexo": case.get("sexo"),
        "peso": case.get("peso"),
        "altura": case.get("altura"),
        "hipertenso": case.get("hipertenso"),
        "diabetico": case.get("diabetico"),
        "obeso": case.get("obeso"),
        "tipo_cancer": (
            case.get("tipo_cancer")
            or case.get("tipo do cancer")
            or case.get("tipo_do_cancer")
        ),
        "estadio": (
            case.get("estadio")
            or case.get("estadio_do_cancer")
            or case.get("estadiamento")
        ),
        "metastase": case.get("metastase"),
        "cirurgia": case.get("cirurgia"),
        "quimioterapia": case.get("quimioterapia"),
        "radioterapia": case.get("radioterapia"),
        "hormonoterapia": case.get("hormonoterapia"),
        "imunoterapia": case.get("imunoterapia"),
        "terapia_alvo": case.get("terapia_alvo"),
        "comorbidades": case.get("comorbidades"),
        "historico_familiar": case.get("historico_familiar"),
        "sintomas": case.get("sintomas"),
        "observacoes": case.get("observacoes"),
        "medicamentos_em_uso": case.get("medicamentos_em_uso"),
        "acompanhamento": case.get("acompanhamento"),
    }

    return {"extracted_entities": extracted}


def build_enriched_query(state: GraphState) -> GraphState:
    """
    Enriquece a query com informações clínicas do caso, quando existirem.
    Isso ajuda o retriever a buscar protocolos mais aderentes.
    """
    question = (state.get("question") or "").strip()
    entities = state.get("extracted_entities") or {}

    if not entities:
        return {"enriched_query": question}

    ordered_fields = [
        "tipo_cancer",
        "estadio",
        "metastase",
        "idade",
        "sexo",
        "cirurgia",
        "quimioterapia",
        "radioterapia",
        "hormonoterapia",
        "imunoterapia",
        "terapia_alvo",
        "comorbidades",
        "sintomas",
    ]

    parts: List[str] = []
    for field in ordered_fields:
        value = entities.get(field)

        if value in (None, "", [], {}, "NÃO", "Nao", "Não informado", "N/A"):
            continue

        if isinstance(value, list):
            value = ", ".join(str(v) for v in value if v not in (None, ""))

        parts.append(f"{field}: {value}")

    if not parts:
        return {"enriched_query": question}

    enriched_query = (
        f"{question}\n"
        f"Contexto clínico relevante:\n" +
        "\n".join(parts)
    )

    return {"enriched_query": enriched_query}


def retrieve_protocols(state: GraphState, rag_pipeline: Any) -> GraphState:
    """
    Usa o RAGPipeline já existente para:
    - recuperar documentos
    - selecionar os melhores
    - montar contexto
    - formatar fontes
    """
    query = (state.get("enriched_query") or state.get("question") or "").strip()

    docs = rag_pipeline.retrieve(query)
    reranked_docs = rag_pipeline.rerank(query, docs)
    selected_docs = reranked_docs[:rag_pipeline.final_k]
    context = rag_pipeline.build_context(selected_docs)
    sources = rag_pipeline.format_sources(selected_docs)

    return {
        "retrieved_docs": selected_docs,
        "retrieved_sources": sources,
        "context": context,
    }


def check_sufficiency(state: GraphState) -> GraphState:
    """
    Regra simples inicial de suficiência:
    - sem docs => insuficiente
    - sem contexto útil => insuficiente
    - poucos docs => ainda pode responder, mas com cautela
    """
    docs = state.get("retrieved_docs") or []
    context = (state.get("context") or "").strip()

    if not docs:
        return {
            "grounded": False,
            "fallback_reason": "Nenhum documento relevante foi recuperado.",
        }

    if not context:
        return {
            "grounded": False,
            "fallback_reason": "Os documentos recuperados não produziram contexto utilizável.",
        }

    # Regra inicial conservadora: ao menos 1 doc com contexto já permite tentativa.
    return {"grounded": True}


def build_graph_prompt(state: GraphState, rag_pipeline: Any) -> str:
    """
    Reaproveita o prompt já existente do projeto.
    Para caso clínico, acrescenta resumo estruturado antes da pergunta.
    """
    question = (state.get("question") or "").strip()
    context = state.get("context") or ""
    entities = state.get("extracted_entities") or {}

    has_case = bool(state.get("patient_case"))

    if not has_case:
        return rag_pipeline.build_prompt(question=question, context=context)

    case_lines = []
    for key, value in entities.items():
        if value in (None, "", [], {}, "N/A"):
            continue

        if isinstance(value, list):
            value = ", ".join(str(v) for v in value if v not in (None, ""))

        case_lines.append(f"- {key}: {value}")

    case_summary = "\n".join(case_lines) if case_lines else "- Caso clínico sem atributos estruturados suficientes."

    enhanced_question = (
        "Considere o caso clínico estruturado abaixo apenas como contexto adicional "
        "para interpretar a pergunta e selecionar a parte relevante dos trechos. "
        "Não utilize informações do caso para inventar condutas que não estejam nos documentos.\n\n"
        f"CASO CLÍNICO:\n{case_summary}\n\n"
        f"PERGUNTA ORIGINAL:\n{question}"
    )

    return rag_pipeline.build_prompt(question=enhanced_question, context=context)


def generate_answer(
    state: GraphState,
    rag_pipeline: Any,
    llm_callable: Optional[Callable[[str], str]] = None,
) -> GraphState:
    """
    Gera resposta com:
    - llm_callable customizado, se informado
    - senão, usa rag_pipeline.generate_answer()
    """
    if not state.get("grounded", False):
        fallback_reason = state.get("fallback_reason") or SAFE_FALLBACK_MESSAGE
        return {
            "answer": f"{SAFE_FALLBACK_MESSAGE}\n\nMotivo: {fallback_reason}"
        }

    prompt = build_graph_prompt(state, rag_pipeline)

    if llm_callable is not None:
        answer = llm_callable(prompt)
    else:
        answer = rag_pipeline.generate_answer(prompt)

    # ----------------------------------
    # 🧹 REMOVE FONTES DO LLM (ROBUSTO)
    # ----------------------------------
    answer_clean = re.split(
        r"\n\s*\*{0,2}Fontes\*{0,2}\s*\n",
        answer,
        flags=re.IGNORECASE
    )[0]

    answer_clean = re.sub(
        r"\*{0,2}Fontes\*{0,2}:?",
        "",
        answer_clean,
        flags=re.IGNORECASE
    ).strip()

    # ----------------------------------
    # 🔗 RECONSTRUIR FONTES CORRETAS
    # ----------------------------------
    selected_docs = state.get("retrieved_docs") or []

    doc_map = {
        f"[DOC_{i+1}]": doc
        for i, doc in enumerate(selected_docs)
    }

    docs_used = set(re.findall(r"\[DOC_\d+\]", answer))

    sources_final = []

    for d in docs_used:
        doc = doc_map.get(d)
        if doc:
            md = doc.metadata or {}
            citation = rag_pipeline.build_citation(md)
            sources_final.append(f"{d} – {citation}")

    # fallback se LLM não citou
    if not sources_final:
        sources_final = state.get("retrieved_sources", [])

    # ----------------------------------
    # 🧾 MONTA RESPOSTA FINAL
    # ----------------------------------
    answer_final = answer_clean

    if sources_final:
        fontes_str = "\n".join([f"- {s}" for s in sources_final])
        answer_final += f"\n\nFontes:\n{fontes_str}"

    return {
        "answer": answer_final,
        "sources": sources_final
    }


def fallback_response(state: GraphState) -> GraphState:
    """
    Resposta padronizada quando o grafo decide que o contexto é insuficiente.
    """
    reason = state.get("fallback_reason") or "Contexto insuficiente."
    return {
        "answer": f"{SAFE_FALLBACK_MESSAGE}\n\nMotivo: {reason}"
    }


def finalize_output(state: GraphState) -> GraphState:
    """
    Nó final opcional. Garante consistência do estado de saída.
    """
    return {
        "question": state.get("question", ""),
        "answer": state.get("answer", SAFE_FALLBACK_MESSAGE),
        "retrieved_sources": state.get("retrieved_sources", []),
        "context": state.get("context", ""),
        "route": state.get("route", ""),
        "grounded": state.get("grounded", False),
    }