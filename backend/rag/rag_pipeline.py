import re

from sentence_transformers import CrossEncoder
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
        print("Collection count:", self.db._collection.count())

        # ----------------------------------
        # LLM (Ollama)
        # ----------------------------------
        self.llm_model = llm_model

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
        try:
            response = ollama.chat(
                model=self.llm_model,
                messages=[
                    {
                        "role": "user",
                        "content": f"""
                            Gere até 3 variações da pergunta abaixo usando terminologia clínica equivalente.

                            Regras:
                            - Manter o mesmo significado
                            - Usar termos técnicos médicos
                            - NÃO mudar o escopo

                            Pergunta: {query}
                        """
                    }
                ],
                options={"temperature": 0.0}
            )

            variations = response["message"]["content"].split("\n")
            variations = [
                v.strip("- ").strip()
                for v in variations
                if v.strip() and len(v.strip()) > 10
            ]

            return list(set([query] + variations))

        except:
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

        # remover duplicados reais
        unique_docs = list({d.page_content: d for d in all_docs}.values())

        # remover chunks ruins
        unique_docs = [
            d for d in unique_docs
            if d.page_content and len(d.page_content.strip()) > 100
        ]

        # limitar antes do rerank
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
    # 4. CONTEXTO (SIMPLIFICADO E OTIMIZADO)
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
    # 5. FONTES (PARA LOG/CSV)
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
    # 6. PROMPT
    # ----------------------------------
    def build_prompt(self, question: str, context: str):
        return PROMPT_TEMPLATE.format(
            question=question,
            context=context
        )

    # ----------------------------------
    # 7. LLM
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
                "temperature": 0.0
            }
        )

        return response["message"]["content"]

    # ----------------------------------
    # 8. PIPELINE COMPLETO
    # ----------------------------------
    def run(self, question: str):
        # 1. Retrieval
        docs = self.retrieve(question)

        # 2. Rerank
        reranked_docs = self.rerank(question, docs)

        # 3. Seleção final
        selected_docs = reranked_docs[:self.final_k]

        # 4. Contexto
        context = self.build_context(selected_docs)

        # 5. Fontes
        sources = self.format_sources(selected_docs)

        # 6. Prompt
        prompt = self.build_prompt(question, context)

        # 7. Geração
        answer = self.generate_answer(prompt)

        docs_used = self.extract_docs_from_answer(answer)

        retorno = {
            "question": question,
            "answer": answer,
            "sources": docs_used if docs_used else sources,
            "documents": [doc.page_content for doc in selected_docs]
        }

        print(retorno)

        return retorno