import json
from pathlib import Path

from rag_pipeline import RAGPipeline
from avaliacao_rag.perguntas_respostas import EVAL_DATA

PROJECT_ROOT = Path(__file__).resolve().parents[2]

VECTOR_DB_PATH = PROJECT_ROOT / "backend/data/vectorstore"

OUTPUT_PATH = PROJECT_ROOT / "backend" / "data" / "ft_dataset" / "ft_dataset.jsonl"
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

def format_context(docs):
    context_text = ""

    for i, doc in enumerate(docs, 1):
        context_text += f"""
        [DOC_{i}]
        CONTEÚDO:
        {doc.strip()}
        """

    return context_text.strip()


def build_user_prompt(question, context):
    return f"""PERGUNTA:
{question}

CONTEXTO:
{context}
"""


def build_assistant_answer(expected_answer, docs):

    # pega apenas 2 primeiros docs (evita ruído)
    doc_refs = [f"[DOC_{i+1}]" for i in range(min(2, len(docs)))]

    refs_inline = " ".join(doc_refs)

    sources_str = "\n".join([f"- [DOC_{i+1}]" for i in range(len(docs))])

    return f"""### Resposta
            {expected_answer} {refs_inline}

            ### Critérios clínicos
            Não explicitamente detalhados nos trechos fornecidos.

            ### Observações
            Resposta baseada exclusivamente no contexto.

            ### Fontes
            {sources_str}
    """

def build_system_prompt():
    return """Você é o OncoSUS, um assistente especializado em Protocolos Clínicos e Diretrizes Terapêuticas (PCDTs) oncológicos do SUS.

Regras obrigatórias:
1. Responda APENAS com base no contexto fornecido
2. NÃO invente informações
3. Sempre cite as fontes usando [DOC_X]
4. Use a estrutura:
   - Resposta
   - Critérios clínicos
   - Observações
   - Fontes
"""


def gerar_dataset():

    print("Iniciando geração do dataset...")
    print(f"Usando VECTOR_DB_PATH: {VECTOR_DB_PATH}")
    print(f"OUTPUT_PATH: {OUTPUT_PATH}")
    rag = RAGPipeline(
        persist_directory=str(VECTOR_DB_PATH),
        llm_model="mistral"
    )

    dataset = []

    for i, item in enumerate(EVAL_DATA, 1):

        question = item["question"]
        expected = item["expected_answer"]

        print(f"[{i}] Gerando exemplo...")

        result = rag.run(question)

        docs = result.get("documents", [])

        # 🔥 se seu RAG não retorna texto, ajuste aqui
        context = format_context(docs)

        user_prompt = build_user_prompt(question, context)

        assistant_answer = build_assistant_answer(expected, docs)

        example = {
            "messages": [
                {"role": "system", "content": build_system_prompt()},
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": assistant_answer},
            ]
        }

        dataset.append(example)

    # salvar JSONL
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for item in dataset:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\n✅ Dataset salvo em: {OUTPUT_PATH}")


if __name__ == "__main__":
    gerar_dataset()