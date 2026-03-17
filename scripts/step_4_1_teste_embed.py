from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VECTOR_DB_PATH = PROJECT_ROOT / "data/vectorstore"

MODEL_NAME = "intfloat/multilingual-e5-base"

# ---------------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------------

model = SentenceTransformer(MODEL_NAME)

# ---------------------------------------------------------
# LOAD DB
# ---------------------------------------------------------

client = chromadb.PersistentClient(
    path=str(VECTOR_DB_PATH)
)

collection = client.get_collection("oncology_documents")

# ---------------------------------------------------------
# QUERY FUNCTION
# ---------------------------------------------------------

def search(query, top_k=5):
    query_embedding = model.encode(
        ["query: " + query]
    )

    results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=top_k
    )

    return results

# ---------------------------------------------------------
# TEST
# ---------------------------------------------------------

query = "tratamento para melanoma metastático"

results = search(query)

for i in range(len(results["documents"][0])):
    print("\n-----------------------------")
    print(f"Result {i+1}")
    print(results["documents"][0][i][:500])
    print("\nMetadata:", results["metadatas"][0][i])