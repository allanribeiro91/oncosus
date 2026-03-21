# rag_pipeline.py

import os
import warnings

# Desenvolvimento atrás de proxy SSL (ex.: Fortinet): Hub usa httpx com verify padrão.
# A variável HF_HUB_DISABLE_SSL_VERIFICATION não é mais aplicada pelo huggingface_hub 1.7+;
# replicamos o efeito aqui via client_factory.
# Ative com: ONCOSUS_INSECURE_SSL=1 ou HF_HUB_DISABLE_SSL_VERIFICATION=1 (só ambiente local).
def _configure_hf_hub_ssl() -> None:
    insecure = (
        os.environ.get("ONCOSUS_INSECURE_SSL", "").strip().lower(),
        os.environ.get("HF_HUB_DISABLE_SSL_VERIFICATION", "").strip().lower(),
    )
    if not any(x in ("1", "true", "yes", "on") for x in insecure if x):
        return
    try:
        import httpx
        from huggingface_hub.utils import _http as hf_http
    except ImportError:
        return

    def _client_factory():
        return httpx.Client(
            verify=False,
            event_hooks={"request": [hf_http.hf_request_event_hook]},
            follow_redirects=True,
            timeout=None,
        )

    hf_http.set_client_factory(_client_factory)
    warnings.filterwarnings(
        "ignore", message="Unverified HTTPS request", category=Warning
    )


_configure_hf_hub_ssl()

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

import ollama

from prompt import PROMPT_TEMPLATE


class RAGPipeline:
    def __init__(
        self,
        persist_directory: str,
        embedding_model: str | None = None,
        llm_model: str | None = None,
        top_k: int = 8,
        final_k: int = 4,
    ):
        # Modelo de embedding: parâmetro > env ONCOSUS_EMBEDDING_MODEL > padrão (exige ~1,1 GB em HF_HOME)
        # Para disco apertado (~500 MB), exemplo: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
        resolved_model = (
            embedding_model
            or os.environ.get("ONCOSUS_EMBEDDING_MODEL")
            or "intfloat/multilingual-e5-base"
        )
        self.embeddings = HuggingFaceEmbeddings(
            model_name=resolved_model
        )

        # Vector DB
        self.db = Chroma(
            persist_directory=persist_directory,
            embedding_function=self.embeddings
        )

        # LLM local (Ollama): parâmetro > env ONCOSUS_OLLAMA_MODEL > padrão
        # Máquinas com pouca RAM: pull um modelo menor, ex.: ollama pull llama3.2:1b
        # e exporte ONCOSUS_OLLAMA_MODEL=llama3.2:1b
        self.llm_model = (
            llm_model
            or os.environ.get("ONCOSUS_OLLAMA_MODEL")
            or "llama3"
        )

        # Menos trechos = prompt menor = menos RAM no Ollama (rede/PDF grande: use 4 e 2)
        self.top_k = int(os.environ.get("ONCOSUS_TOP_K", str(top_k)))
        self.final_k = int(os.environ.get("ONCOSUS_FINAL_K", str(final_k)))

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
    def _ollama_chat_options(self) -> dict:
        """Opções passadas ao llama.cpp via Ollama — reduzir num_ctx poupa muita RAM."""
        num_ctx = int(os.environ.get("ONCOSUS_OLLAMA_NUM_CTX", "2048"))
        opts: dict = {
            "temperature": 0.0,
            "num_ctx": num_ctx,
        }
        # Opcional: ex. 128 ou 256 se ainda der "unable to allocate CPU buffer"
        nb = os.environ.get("ONCOSUS_OLLAMA_NUM_BATCH")
        if nb:
            opts["num_batch"] = int(nb)
        return opts

    def generate_answer(self, prompt: str):

        response = ollama.chat(
            model=self.llm_model,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            options=self._ollama_chat_options(),
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