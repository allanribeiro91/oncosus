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

# Mesmo nome no step_4_0_embed_chunks.py — trocar embeddings exige reindexar o Chroma.
_DEFAULT_EMBEDDING_MODEL = "intfloat/multilingual-e5-base"
_LIGHT_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
# Padrão: llama3 (o que costuma funcionar no projeto). Máquinas com pouca RAM: ONCOSUS_OLLAMA_MODEL=llama3.2:1b
_DEFAULT_OLLAMA_MODEL = "llama3"


def _ollama_client_timeout() -> float | None:
    raw = (os.environ.get("ONCOSUS_OLLAMA_TIMEOUT_SEC") or "300").strip().lower()
    if raw in ("", "0", "none", "inf", "false"):
        return None
    return float(raw)

_EMBEDDING_LOAD_FAIL_HINT = (
    "Memória virtual insuficiente ao carregar embeddings (erro 1455). Tente nesta ordem: "
    "(0) Feche o Ollama da bandeja, suba só a API até aparecer 'Application startup', depois abra o Ollama. "
    "(1) Aumente o arquivo de paginação do Windows e feche navegadores. "
    "(2) Só se ainda falhar: mesmo embedding do índice é obrigatório — use %s, apague data/vectorstore "
    "e rode backend/scripts/step_4_0_embed_chunks.py com a mesma variável."
    % _LIGHT_EMBEDDING_MODEL
)


class RAGPipeline:
    def __init__(
        self,
        persist_directory: str,
        embedding_model: str | None = None,
        llm_model: str | None = None,
        top_k: int = 6,
        final_k: int = 3,
    ):
        # Modelo de embedding: parâmetro > env ONCOSUS_EMBEDDING_MODEL > padrão (e5-base ~1,1 GB RAM+commit)
        # Vector store foi gerado com um modelo: consultas DEVEM usar o mesmo (ou reindexar).
        resolved_model = (
            embedding_model
            or os.environ.get("ONCOSUS_EMBEDDING_MODEL")
            or _DEFAULT_EMBEDDING_MODEL
        )
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        try:
            # Não usar low_cpu_mem_usage aqui: várias versões do sentence-transformers quebram com TypeError.
            self.embeddings = HuggingFaceEmbeddings(model_name=resolved_model)
        except (OSError, MemoryError) as e:
            winerr = getattr(e, "winerror", None)
            msg = str(e).lower()
            if (
                isinstance(e, MemoryError)
                or winerr == 1455
                or "paginação" in msg
                or "paging" in msg
                or "1455" in msg
            ):
                raise RuntimeError(f"{_EMBEDDING_LOAD_FAIL_HINT}\n\nErro original: {e}") from e
            raise

        # Vector DB (mesmo nome da coleção que step_4_0_embed_chunks.py grava no Chroma)
        self.db = Chroma(
            persist_directory=persist_directory,
            embedding_function=self.embeddings,
            collection_name="oncology_documents",
        )

        # LLM local (Ollama): parâmetro > env ONCOSUS_OLLAMA_MODEL > padrão
        self.llm_model = (
            llm_model
            or os.environ.get("ONCOSUS_OLLAMA_MODEL")
            or _DEFAULT_OLLAMA_MODEL
        )

        # Cliente com timeout — o módulo ollama.chat() usa timeout=None e pode travar a API para sempre.
        self._ollama_timeout_sec = _ollama_client_timeout()
        self._ollama_client = ollama.Client(
            timeout=self._ollama_timeout_sec,
        )

        # Menos trechos = prompt menor = menos RAM no Ollama
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
    @staticmethod
    def _normalize_meta(metadata: dict) -> dict:
        """Chroma pode ter document_title/source (ingestão) ou titulo/fonte (legado)."""
        md = metadata or {}
        titulo = md.get("titulo") or md.get("document_title") or "Documento"
        fonte = md.get("fonte") or md.get("source") or "Desconhecido"
        secao = md.get("secao") or md.get("section")
        pagina = md.get("pagina")
        return {"titulo": titulo, "fonte": fonte, "secao": secao, "pagina": pagina}

    def build_citation(self, metadata):
        m = self._normalize_meta(metadata)
        titulo = m["titulo"]
        secao = m["secao"]
        pagina = m["pagina"]

        parts = [titulo]

        if secao and secao != "N/A":
            parts.append(str(secao))

        if pagina and pagina != "N/A":
            parts.append(f"pág. {pagina}")

        return " – ".join(parts)
    
    def build_context(self, docs):
        context_str = ""

        for i, doc in enumerate(docs, 1):
            metadata = doc.metadata or {}
            m = self._normalize_meta(metadata)

            fonte = m["fonte"]
            titulo = m["titulo"]
            secao = m["secao"] if m["secao"] is not None else "N/A"
            pagina = m["pagina"] if m["pagina"] is not None else "N/A"

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
        """Opções llama.cpp via Ollama — padrões conservadores para evitar falta de RAM no CPU."""
        num_ctx = int(os.environ.get("ONCOSUS_OLLAMA_NUM_CTX", "512"))
        opts: dict = {
            "temperature": 0.0,
            "num_ctx": num_ctx,
        }
        # num_batch menor reduz "unable to allocate CPU buffer" no Windows
        nb = os.environ.get("ONCOSUS_OLLAMA_NUM_BATCH")
        opts["num_batch"] = int(nb) if nb else 128
        return opts

    def generate_answer(self, prompt: str):
        try:
            response = self._ollama_client.chat(
                model=self.llm_model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                options=self._ollama_chat_options(),
            )
        except Exception as e:
            msg = str(e).lower()
            if self._ollama_timeout_sec is not None and (
                "timeout" in msg or "timed out" in msg or "read timeout" in msg
            ):
                raise RuntimeError(
                    f"Ollama não respondeu em {self._ollama_timeout_sec:.0f}s "
                    f"(modelo `{self.llm_model}`). CPU/RAM ocupados ou modelo grande. "
                    "Garanta o app Ollama aberto; confira ONCOSUS_OLLAMA_MODEL ou "
                    "aumente ONCOSUS_OLLAMA_TIMEOUT_SEC."
                ) from e
            raise

        text = response.message.content
        return (text or "").strip()

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