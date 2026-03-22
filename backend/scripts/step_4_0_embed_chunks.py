import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer
import chromadb

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

CHUNKS_PATH = PROJECT_ROOT / "data/chunks/chunks.csv"
VECTOR_DB_PATH = PROJECT_ROOT / "data/vectorstore"

# MODEL_NAME = "BAAI/bge-m3"
MODEL_NAME = "intfloat/multilingual-e5-base"

# ---------------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------------

print("Loading embedding model...")

model = SentenceTransformer(MODEL_NAME)

# ---------------------------------------------------------
# LOAD CHUNKS
# ---------------------------------------------------------

print("Loading chunks...")

df = pd.read_csv(CHUNKS_PATH)

texts = df["text"].tolist()
chunk_ids = df["chunk_id"].tolist()

metadatas = []

for _, row in df.iterrows():
    metadatas.append({
        "document_id": row["document_id"],
        "document_title": row.get("document_title", ""),
        "source": row.get("source", ""),
        "chunk_index": int(row["chunk_index"])
    })

# ---------------------------------------------------------
# ENSURE VECTOR DB PATH EXISTS
# ---------------------------------------------------------

VECTOR_DB_PATH.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------
# CREATE VECTOR DATABASE
# ---------------------------------------------------------

client = chromadb.PersistentClient(
    path=str(VECTOR_DB_PATH)
)

collection = client.get_or_create_collection(
    name="oncology_documents"
)

# ---------------------------------------------------------
# GENERATE EMBEDDINGS
# ---------------------------------------------------------

print("Generating embeddings...")

embeddings = model.encode(
    texts,
    batch_size=32,
    show_progress_bar=True
)

# ---------------------------------------------------------
# STORE IN VECTOR DB
# ---------------------------------------------------------

BATCH_SIZE = 500  # seguro

print("Storing embeddings in batches...")

for i in range(0, len(chunk_ids), BATCH_SIZE):

    batch_ids = chunk_ids[i:i+BATCH_SIZE]
    batch_embeddings = embeddings[i:i+BATCH_SIZE]
    batch_texts = texts[i:i+BATCH_SIZE]
    batch_metadatas = metadatas[i:i+BATCH_SIZE]

    collection.add(
        ids=batch_ids,
        embeddings=batch_embeddings.tolist(),
        documents=batch_texts,
        metadatas=batch_metadatas
    )

    print(f"Inserted {i + len(batch_ids)} / {len(chunk_ids)}")

print("Embeddings stored successfully!")
