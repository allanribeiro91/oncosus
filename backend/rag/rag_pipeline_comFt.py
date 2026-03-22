import re
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from sentence_transformers import CrossEncoder
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
LORA_PATH = PROJECT_ROOT / "backend/rag/lora-oncosus"

class RAGPipelineComFt:
    def __init__(
        self,
        persist_directory: str,
        embedding_model: str = "intfloat/multilingual-e5-base",
        top_k: int = 10,
        final_k: int = 5,
    ):
        # ----------------------------------
        # Embeddings
        # ----------------------------------
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model
        )

        # ----------------------------------
        # Vector DB
        # ----------------------------------
        self.db = Chroma(
            persist_directory=persist_directory,
            embedding_function=self.embeddings,
            collection_name="oncology_documents"
        )
        print("Collection count - rag com FT:", self.db._collection.count())

        # ----------------------------------
        # LLM (Mistral + LoRA)
        # ----------------------------------
        base_model_name = "mistralai/Mistral-7B-Instruct-v0.2"

        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            device_map="auto",
            torch_dtype=torch.float16
        )

        self.model = PeftModel.from_pretrained(
            base_model,
            LORA_PATH
        )

        self.model.eval()

        # ----------------------------------
        # Config
        # ----------------------------------
        self.top_k = top_k
        self.final_k = final_k

        # ----------------------------------
        # Re-ranker
        # ----------------------------------
        self.reranker = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )

    # ----------------------------------
    # 1. QUERY EXPANSION
    # ----------------------------------
    def expand_query(self, query: str):
        # 🔹 opcional: manter simples para evitar custo
        return [query]

    # ----------------------------------
    # 2. RETRIEVAL
    # ----------------------------------
    def retrieve(self, query: str):
        queries = self.expand_query(query)

        all_docs = []

        for q in queries:
            docs = self.db.similarity_search(q, k=self.top_k)
            all_docs.extend(docs)

        # remover duplicados
        unique_docs = list({d.page_content: d for d in all_docs}.values())

        # remover chunks ruins
        unique_docs = [
            d for d in unique_docs
            if d.page_content and len(d.page_content.strip()) > 100
        ]

        return unique_docs[:20]

    # ----------------------------------
    # 3. RERANK
    # ----------------------------------
    def rerank(self, query: str, docs):
        if not docs:
            return []

        pairs = [
            (query, d.page_content[:512])
            for d in docs
        ]

        scores = self.reranker.predict(pairs)

        scored_docs = list(zip(docs, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        return [doc for doc, _ in scored_docs]

    # ----------------------------------
    # 4. CONTEXTO
    # ----------------------------------
    def build_context(self, docs):
        context_str = ""

        for i, doc in enumerate(docs, 1):
            context_str += f"""
[DOC_{i}]
CONTEÚDO:
{doc.page_content.strip()}
"""
        return context_str

    # ----------------------------------
    # 5. FONTES
    # ----------------------------------
    def build_citation(self, metadata):
        titulo = metadata.get("titulo") or "Documento"
        secao = metadata.get("secao")
        pagina = metadata.get("pagina")

        parts = [titulo]

        if secao and secao != "N/A":
            parts.append(secao)

        if pagina and pagina != "N/A":
            parts.append(f"pág. {pagina}")

        return " – ".join(parts)

    def format_sources(self, docs):
        sources = []
        seen = set()

        for doc in docs:
            md = doc.metadata or {}
            citation = self.build_citation(md)

            if citation not in seen:
                seen.add(citation)
                sources.append(citation)

        return sources

    def extract_docs_from_answer(self, answer: str):
        docs = set(re.findall(r"\[DOC_\d+\]", answer))
        return sorted(docs)

    # ----------------------------------
    # 6. PROMPT (FORMATO MISTRAL)
    # ----------------------------------
    def build_prompt(self, question: str, context: str):

        system = """
Você é um assistente especializado em oncologia do SUS (PCDTs).

Regras:
- Use apenas o contexto
- Cite como [DOC_X]
- Seja técnico e objetivo
- Não invente informações
- Se não houver evidência suficiente, diga explicitamente
"""

        user = f"""
PERGUNTA:
{question}

CONTEXTO:
{context}
"""

        return f"<s>[INST] {system}\n\n{user} [/INST]"

    # ----------------------------------
    # 7. GERAÇÃO (LLM LOCAL)
    # ----------------------------------
    def generate_answer(self, prompt: str):

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.1,
                do_sample=False,
                repetition_penalty=1.1,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id
            )

        response = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )

        return response

    # ----------------------------------
    # 8. PIPELINE COMPLETO
    # ----------------------------------
    def run(self, question: str):
        docs = self.retrieve(question)
        reranked_docs = self.rerank(question, docs)
        selected_docs = reranked_docs[:self.final_k]

        context = self.build_context(selected_docs)
        sources = self.format_sources(selected_docs)

        prompt = self.build_prompt(question, context)
        answer = self.generate_answer(prompt)

        docs_used = self.extract_docs_from_answer(answer)

        retorno = {
            "question": question,
            "answer": answer,
            "sources": docs_used if docs_used else sources,
            "documents": [
                {
                    "text": doc.page_content,
                    "title": doc.metadata.get("document_title", "Documento")
                }
                for doc in selected_docs
            ]
        }

        print(retorno)

        return retorno