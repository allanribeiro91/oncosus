import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer
import chromadb

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CHUNKS_PATH = PROJECT_ROOT / "data/chunks/chunks.csv"
VECTOR_DB_PATH = PROJECT_ROOT / "data/vectorstore"

MODEL_NAME = "intfloat/multilingual-e5-base"

BATCH_SIZE = 96

VECTOR_DB_PATH.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------------

print("Loading embedding model...")

model = SentenceTransformer(MODEL_NAME)

# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------

print("Loading chunks...")

df = pd.read_csv(CHUNKS_PATH)

# 🔥 E5 REQUIREMENT: prefixo passage:
texts_raw = df["text"].tolist()
texts = [f"passage: {t}" for t in texts_raw]

chunk_ids = df["chunk_id"].astype(str).tolist()

# ---------------------------------------------------------
# METADATA (ENRIQUECIDA)
# ---------------------------------------------------------

metadatas = []

for _, row in df.iterrows():
    metadatas.append({
        "document_id": row.get("document_id", ""),
        "document_title": row.get("document_title", ""),
        "source": row.get("source", ""),
        "year": row.get("year", ""),
        "document_type": row.get("document_type", ""),
        "section": row.get("section", ""),
        "tags": row.get("tags", ""),
        "chunk_index": int(row.get("chunk_index", 0))
    })

# ---------------------------------------------------------
# INIT VECTOR DB
# ---------------------------------------------------------

client = chromadb.PersistentClient(
    path=str(VECTOR_DB_PATH)
)

COLLECTION_NAME = "oncology_documents"

# 🔥 evita duplicação
existing_collections = [c.name for c in client.list_collections()]

if COLLECTION_NAME in existing_collections:
    print("Deleting existing collection...")
    client.delete_collection(COLLECTION_NAME)

collection = client.create_collection(
    name=COLLECTION_NAME
)

# ---------------------------------------------------------
# EMBEDDING + INSERT (BATCHED)
# ---------------------------------------------------------

print("Generating embeddings and storing...")

total = len(chunk_ids)

for i in range(0, total, BATCH_SIZE):

    batch_texts = texts[i:i+BATCH_SIZE]
    batch_ids = chunk_ids[i:i+BATCH_SIZE]
    batch_metadatas = metadatas[i:i+BATCH_SIZE]
    batch_documents = texts_raw[i:i+BATCH_SIZE]

    # 🔥 encode por batch (melhor memória)
    batch_embeddings = model.encode(
        batch_texts,
        batch_size=48,
        show_progress_bar=False,
        normalize_embeddings=True
    )

    collection.add(
        ids=batch_ids,
        embeddings=batch_embeddings.tolist(),
        documents=batch_documents,
        metadatas=batch_metadatas
    )

    print(f"Inserted {i + len(batch_ids)} / {total}")

print("Embeddings stored successfully!")