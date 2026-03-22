from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VECTOR_DB_PATH = PROJECT_ROOT / "data/vectorstore"

MODEL_NAME = "intfloat/multilingual-e5-base"

TOP_K = 8

# ---------------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------------

print("Loading embedding model...")

model = SentenceTransformer(MODEL_NAME)

# ---------------------------------------------------------
# LOAD VECTOR DB
# ---------------------------------------------------------

print("Loading vector database...")

client = chromadb.PersistentClient(
    path=str(VECTOR_DB_PATH)
)

collection = client.get_collection("oncology_documents")

# ---------------------------------------------------------
# SEARCH FUNCTION (OTIMIZADA)
# ---------------------------------------------------------

def search(query, top_k=TOP_K, filter_section=None):

    print(f"\nQuery: {query}")

    # 🔥 E5 REQUIREMENT: prefixo query:
    query_input = f"query: {query}"

    query_embedding = model.encode(
        [query_input],
        normalize_embeddings=True  # 🔥 importante
    )

    # 🔥 filtro opcional
    where_filter = None
    if filter_section:
        where_filter = {"section": filter_section}

    results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=top_k,
        where=where_filter
    )

    return results


# ---------------------------------------------------------
# PRINT RESULTS (BONITO E ÚTIL)
# ---------------------------------------------------------

def print_results(results):

    docs = results["documents"][0]
    metas = results["metadatas"][0]

    for i, (doc, meta) in enumerate(zip(docs, metas)):

        print("\n" + "="*60)
        print(f"Result {i+1}")

        print(f"\n📌 Documento: {meta.get('document_title', '')}")
        print(f"📂 Seção: {meta.get('section', '')}")
        print(f"📅 Ano: {meta.get('year', '')}")

        print("\n📄 Texto:")
        print(doc[:600] + ("..." if len(doc) > 600 else ""))


# ---------------------------------------------------------
# TESTES
# ---------------------------------------------------------

if __name__ == "__main__":

    # 🔹 Teste 1 (geral)
    results = search(
        "tratamento para melanoma metastático"
    )
    print_results(results)

    # 🔹 Teste 2 (com filtro)
    results = search(
        "tratamento para melanoma metastático",
        filter_section="treatment"
    )
    print_results(results)