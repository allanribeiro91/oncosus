# pipeline.py
"""
Assistente IA OncoSUS - Pipeline RAG com Router
Conecta o Router (classificação de intenção) ao mecanismo de busca vetorial.
"""

from pathlib import Path
from typing import Optional

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

import ollama

from prompt import PROMPT_TEMPLATE, ROUTER_PROMPT


# ---------------------------------------------------------------------------
# CATEGORIAS DE CÂNCER (mapeamento Router <-> metadata do Chroma)
# ---------------------------------------------------------------------------
CANCER_CATEGORIES = [
    "cancer_mama",
    "cancer_pulmao",
    "cancer_prostata",
    "cancer_esofago",
    "cancer_colo_retal",
    "cancer_melanoma",
    "cancer_estomago",
    "cancer_tireoide",
    "cancer_bexiga",
    "cancer_rim",
    "cancer_hematologico",
    "geral",
]

CATEGORY_DISPLAY = {
    "cancer_mama": "Câncer de Mama",
    "cancer_pulmao": "Câncer de Pulmão",
    "cancer_prostata": "Câncer de Próstata",
    "cancer_esofago": "Carcinoma de Esôfago",
    "cancer_colo_retal": "Câncer Colorretal",
    "cancer_melanoma": "Melanoma",
    "cancer_estomago": "Câncer de Estômago",
    "cancer_tireoide": "Câncer de Tireoide",
    "cancer_bexiga": "Câncer de Bexiga",
    "cancer_rim": "Câncer de Rim",
    "cancer_hematologico": "Câncer Hematológico (mieloma, leucemia, linfoma)",
    "geral": "Geral",
}


# ---------------------------------------------------------------------------
# ROUTER - Reconhecimento de Intenção
# ---------------------------------------------------------------------------
class CancerTypeRouter:
    """
    Router que classifica a pergunta do usuário em uma categoria de câncer
    antes da busca vetorial.
    """

    def __init__(self, llm_model: str = "llama3"):
        self.llm_model = llm_model
        self._chain = None

    def _build_chain(self):
        """Constrói a chain de classificação."""
        llm = Ollama(model=self.llm_model, temperature=0.0)
        prompt = ChatPromptTemplate.from_messages([
            ("system", ROUTER_PROMPT),
            ("human", "{question}"),
        ])
        return prompt | llm | StrOutputParser()

    def classify(self, question: str) -> str:
        """
        Classifica a pergunta e retorna a categoria de câncer.
        Retorna 'geral' se não identificar tipo específico.
        """
        if self._chain is None:
            self._chain = self._build_chain()

        result = self._chain.invoke({"question": question})
        category = result.strip().lower().replace(" ", "_")

        # Normaliza a saída para uma categoria válida
        for valid in CANCER_CATEGORIES:
            if valid in category or category in valid:
                return valid

        # Fallback: verifica termos na pergunta
        q_lower = question.lower()
        if "mama" in q_lower or "mamário" in q_lower:
            return "cancer_mama"
        if "pulmão" in q_lower or "pulmao" in q_lower or "pulmonar" in q_lower:
            return "cancer_pulmao"
        if "próstata" in q_lower or "prostata" in q_lower:
            return "cancer_prostata"
        if "esôfago" in q_lower or "esofago" in q_lower or "esofágico" in q_lower:
            return "cancer_esofago"
        if "colorretal" in q_lower or "cólon" in q_lower or "colo" in q_lower or "retal" in q_lower:
            return "cancer_colo_retal"
        if "melanoma" in q_lower:
            return "cancer_melanoma"
        if "estômago" in q_lower or "estomago" in q_lower or "gástrico" in q_lower:
            return "cancer_estomago"
        if "tireoide" in q_lower or "tireóide" in q_lower:
            return "cancer_tireoide"
        if "bexiga" in q_lower:
            return "cancer_bexiga"
        if "rim" in q_lower and "câncer" in q_lower:
            return "cancer_rim"

        return "geral"


# ---------------------------------------------------------------------------
# RAG PIPELINE COM ROUTER
# ---------------------------------------------------------------------------
class OncoSUSPipeline:
    """
    Pipeline RAG completo: Router -> Retriever (filtrado) -> Geração.
    """

    def __init__(
        self,
        persist_directory: str,
        collection_name: str = "oncology_documents",
        embedding_model: str = "intfloat/multilingual-e5-base",
        llm_model: str = "llama3",
        top_k: int = 8,
        final_k: int = 4,
    ):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.llm_model = llm_model
        self.top_k = top_k
        self.final_k = final_k

        # Embeddings
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)

        # Vector DB (Chroma)
        self.db = Chroma(
            persist_directory=persist_directory,
            embedding_function=self.embeddings,
            collection_name=collection_name,
        )

        # Router
        self.router = CancerTypeRouter(llm_model=llm_model)

    # ----------------------------------
    # 1. Retrieval (com filtro por categoria)
    # ----------------------------------
    def retrieve(self, query: str, category: Optional[str] = None):
        """
        Busca documentos similares. Se category != 'geral', filtra por cancer_type.
        """
        filter_dict = None
        if category and category != "geral":
            filter_dict = {"cancer_type": {"$eq": category}}

        docs = self.db.similarity_search(
            query,
            k=self.top_k,
            filter=filter_dict,
        )
        return docs

    # ----------------------------------
    # 2. Seleção de documentos
    # ----------------------------------
    def select_top_docs(self, docs):
        return [d for d in docs if d.page_content.strip()][: self.final_k]

    # ----------------------------------
    # 3. Contexto estruturado
    # ----------------------------------
    def _build_citation(self, metadata: dict) -> str:
        titulo = metadata.get("document_title") or metadata.get("titulo") or "Documento"
        fonte = metadata.get("source") or metadata.get("fonte", "Desconhecido")
        secao = metadata.get("secao")
        pagina = metadata.get("pagina")

        parts = [titulo]
        if fonte and fonte != "N/A":
            parts.append(f"Fonte: {fonte}")
        if secao and secao != "N/A":
            parts.append(secao)
        if pagina and pagina != "N/A":
            parts.append(f"pág. {pagina}")

        return " – ".join(parts)

    def build_context(self, docs: list[Document]) -> str:
        context_str = ""
        for i, doc in enumerate(docs, 1):
            metadata = doc.metadata or {}
            citation = self._build_citation(metadata)
            context_str += f"""
[Documento {i}]
Fonte: {metadata.get('source', metadata.get('fonte', 'Desconhecido'))}
Documento: {metadata.get('document_title', metadata.get('titulo', 'Documento'))}
Seção: {metadata.get('secao', metadata.get('section', 'N/A'))}
Citação: {citation}

Trecho:
{doc.page_content}
"""
        return context_str

    def format_sources(self, docs: list[Document]) -> list[str]:
        sources = []
        seen = set()
        for doc in docs:
            md = doc.metadata or {}
            citation = self._build_citation(md)
            if citation not in seen:
                seen.add(citation)
                sources.append(citation)
        return sources

    # ----------------------------------
    # 4. Prompt e Geração
    # ----------------------------------
    def build_prompt(self, question: str, context: str) -> str:
        return PROMPT_TEMPLATE.format(question=question, context=context)

    def generate_answer(self, prompt: str) -> str:
        response = ollama.chat(
            model=self.llm_model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.0},
        )
        return response["message"]["content"]

    # ----------------------------------
    # 5. Pipeline completo
    # ----------------------------------
    def run(self, question: str) -> dict:
        """
        Executa o pipeline: Router -> Retriever -> Geração.
        """
        # 1. Router: classifica o tipo de câncer
        category = self.router.classify(question)

        # 2. Retrieval (filtrado pela categoria)
        docs = self.retrieve(question, category=category)

        # Se não encontrou com filtro e categoria != geral, tenta sem filtro
        if len(docs) < 2 and category != "geral":
            docs = self.retrieve(question, category=None)

        # 3. Seleção
        selected_docs = self.select_top_docs(docs)

        # 4. Contexto
        context = self.build_context(selected_docs)
        sources = self.format_sources(selected_docs)

        # 5. Prompt
        prompt = self.build_prompt(question, context)

        # 6. LLM
        answer = self.generate_answer(prompt)

        return {
            "question": question,
            "answer": answer,
            "sources": sources,
            "category": category,
            "category_display": CATEGORY_DISPLAY.get(category, category),
        }


# ---------------------------------------------------------------------------
# PONTO DE ENTRADA
# ---------------------------------------------------------------------------
def create_pipeline(
    persist_directory: Optional[str] = None,
    **kwargs,
) -> OncoSUSPipeline:
    """Factory para criar o pipeline com paths padrão."""
    if persist_directory is None:
        project_root = Path(__file__).resolve().parent.parent
        persist_directory = str(project_root / "data" / "vectorstore")
    return OncoSUSPipeline(persist_directory=persist_directory, **kwargs)


if __name__ == "__main__":
    pipeline = create_pipeline()

    print("=== Assistente IA OncoSUS - Pipeline RAG com Router ===\n")
    print("Digite sua pergunta (ou 'sair' para encerrar).\n")

    while True:
        query = input("Pergunta: ").strip()
        if not query:
            continue
        if query.lower() in ["exit", "sair", "quit"]:
            break

        result = pipeline.run(query)

        print("\n" + "=" * 50)
        print(f"Categoria identificada: {result['category_display']}")
        print("=" * 50)
        print("\nRESPOSTA:\n")
        print(result["answer"])
        print("\nFONTES:")
        for src in result["sources"]:
            print(f"  • {src}")
        print()
