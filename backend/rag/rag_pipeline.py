import re

from sentence_transformers import CrossEncoder
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

import ollama

from rag.prompt import PROMPT_TEMPLATE


class RAGPipeline:
    def __init__(
        self,
        persist_directory: str,
        embedding_model: str = "intfloat/multilingual-e5-base",
        llm_model: str = "llama3",
        top_k: int = 5,
        final_k: int = 3,
        use_reranker: bool = False,
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
        self.use_reranker = use_reranker
        print(f"Use reranker: {self.use_reranker}")

        if self.use_reranker:
            self.reranker = CrossEncoder(
                "cross-encoder/ms-marco-MiniLM-L-6-v2"
            )
        else:
            self.reranker = None

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

        if not self.use_reranker:
            return docs  # 🔥 bypass direto

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
        titulo = (
            metadata.get("document_title")
            or metadata.get("title")
            or "Documento"
        )

        secao = metadata.get("section")
        ano = metadata.get("year")

        parts = [titulo]

        # 🔹 ano (AQUI entra o que você perguntou)
        if ano:
            parts.append(str(ano))

        # 🔹 seção (filtrando lixo)
        if secao and secao not in ["N/A", "other"]:
            parts.append(secao)

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

        # ----------------------------------
        # 🔥 DEBUG DO METADATA (AQUI!)
        # ----------------------------------
        if selected_docs:
            print("\n🔍 DEBUG METADATA:")
            print(selected_docs[0].metadata)
            print("\n")

        # 4. Contexto
        context = self.build_context(selected_docs)

        # 5. Fontes
        sources = self.format_sources(selected_docs)

        # 6. Prompt
        prompt = self.build_prompt(question, context)

        # 7. Geração
        answer = self.generate_answer(prompt)

        # ----------------------------------
        # 🔗 MAPEAMENTO DOC_X → DOCUMENTO
        # ----------------------------------
        doc_map = {
            f"[DOC_{i+1}]": doc
            for i, doc in enumerate(selected_docs)
        }

        docs_used = self.extract_docs_from_answer(answer)

        # ----------------------------------
        # 🧠 CONSTRUIR FONTES CORRETAS
        # ----------------------------------
        sources_final = sources  # default

        if docs_used:
            sources_final = []
            for d in docs_used:
                doc = doc_map.get(d)
                if doc:
                    md = doc.metadata or {}
                    citation = self.build_citation(md)
                    sources_final.append(f"{d} – {citation}")


        # ----------------------------------
        # 🧹 REMOVE "Fontes" do LLM
        # ----------------------------------
        answer_clean = re.split(r"\n\s*Fontes:\s*\n", answer)[0]

        # ----------------------------------
        # 🧾 ADICIONA FONTES CORRETAS
        # ----------------------------------
        answer_final = answer_clean

        if sources_final:
            fontes_str = "\n".join([f"- {s}" for s in sources_final])
            answer_final += f"\n\nFontes:\n{fontes_str}"

        retorno = {
            "question": question,
            "answer": answer_final,
            "sources": sources_final,
            "documents": [
                {
                    "text": doc.page_content,
                    "title": doc.metadata.get("title", "Documento")
                }
                for doc in selected_docs
            ]
        }

        print(retorno)

        return retorno