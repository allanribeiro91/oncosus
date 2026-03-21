# rag_pipeline.py

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

import ollama

from prompt import PROMPT_TEMPLATE


class RAGPipeline:
    def __init__(
        self,
        persist_directory: str,
        embedding_model: str = "intfloat/multilingual-e5-base",
        llm_model: str = "llama3",
        top_k: int = 8,
        final_k: int = 4,
    ):
        # Embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model
        )

        # Vector DB
        self.db = Chroma(
            persist_directory=persist_directory,
            embedding_function=self.embeddings,
            collection_name="oncology_documents",
        )

        # LLM local (Ollama)
        self.llm_model = llm_model
        self.top_k = top_k
        self.final_k = final_k

    # ----------------------------------
    # 1. Retrieval
    # ----------------------------------
    def retrieve(self, query: str):
        docs = self.db.similarity_search(query, k=self.top_k)
        return docs

    # ----------------------------------
    # 2. Seleção
    # ----------------------------------
    def select_top_docs(self, docs):
        return [d for d in docs if d.page_content.strip()][:self.final_k]

    # ----------------------------------
    # 3. Contexto estruturado
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
    
    def build_context(self, docs):
        context_str = ""

        for i, doc in enumerate(docs, 1):
            metadata = doc.metadata or {}

            fonte = metadata.get("fonte", "Desconhecido")
            titulo = metadata.get("titulo", "Documento")
            secao = metadata.get("secao", "N/A")
            pagina = metadata.get("pagina", "N/A")

            citation = self.build_citation(metadata)

            context_str += f"""
    [Documento {i}]
    Fonte: {fonte}
    Documento: {titulo}
    Seção: {secao}
    Página: {pagina}
    Citação: {citation}

    Trecho:
    {doc.page_content}
    """
        return context_str
    

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

    # ----------------------------------
    # 4. Prompt
    # ----------------------------------
    def build_prompt(self, question: str, context: str):
        return PROMPT_TEMPLATE.format(
            question=question,
            context=context
        )

    # ----------------------------------
    # 5. LLM (Ollama)
    # ----------------------------------
    def generate_answer(self, prompt: str):

        response = ollama.chat(
            model=self.llm_model,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            options={
                "temperature": 0.0  # anti-hallucination
            }
        )

        return response["message"]["content"]

    # ----------------------------------
    # 6. Pipeline completo
    # ----------------------------------
    def run(self, question: str):
        # 1. Retrieval
        docs = self.retrieve(question)

        # 2. Seleção
        selected_docs = self.select_top_docs(docs)

        # 3. Contexto
        context = self.build_context(selected_docs)

        sources = self.format_sources(selected_docs)

        # 4. Prompt
        prompt = self.build_prompt(question, context)

        # 5. LLM
        answer = self.generate_answer(prompt)

        return {
            "question": question,
            "answer": answer,
            "sources": sources
        }