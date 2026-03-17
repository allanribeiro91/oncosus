import pandas as pd
from pathlib import Path
import uuid

import re

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

METADATA_PATH = PROJECT_ROOT / "data/metadata/documents_metadata.csv"

metadata_df = pd.read_csv(METADATA_PATH, sep=';')
metadata_map = {
    row["document_id"]: row
    for _, row in metadata_df.iterrows()
}

INPUT_DIR = PROJECT_ROOT / "data/processed"
OUTPUT_FILE = PROJECT_ROOT / "data/chunks/chunks.csv"

CHUNK_SIZE = 1200
OVERLAP = 200

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------
# CHUNK FUNCTION
# ---------------------------------------------------------

def split_text(text, max_chars=800):

    # 🔥 normalizar quebras de linha
    text = text.replace("\r", "\n")

    # 🔥 criar parágrafos artificiais
    text = re.sub(r"\n(?=[A-Z])", "\n\n", text)

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    chunks = []
    buffer = ""

    for paragraph in paragraphs:

        # ignora lixo muito pequeno
        if len(paragraph) < 50:
            continue

        # se cabe no buffer
        if len(buffer) + len(paragraph) <= max_chars:
            if buffer:
                buffer += " " + paragraph
            else:
                buffer = paragraph

        else:
            if buffer:
                chunks.append(buffer.strip())
            buffer = paragraph

    if buffer:
        chunks.append(buffer.strip())

    return chunks

def is_valid_chunk(text):

    text_lower = text.lower()

    if len(text) < 100:
        return False

    # # remove tabelas SUS / códigos
    if len(re.findall(r"\b\d{2}\.\d{2}\.\d{2}", text)) > 3:
        return False

    # remove referências bibliográficas
    if "doi" in text_lower or "et al" in text_lower:
        return False

    # remove seção de referências
    if "referências" in text_lower:
        return False

    if "secretário de atenção à saúde" in text_lower:
        return False
    
    if "metodologia de busca" in text_lower:
        return False
    
    if "metodologia de busca" in text_lower:
        return False

    return True

def detect_section(text):
    text_lower = text.lower()

    if "tratamento" in text_lower:
        return "treatment"
    if "diagnóstico" in text_lower:
        return "diagnosis"
    if "prognóstico" in text_lower:
        return "prognosis"

    return "general"

# ---------------------------------------------------------
# PROCESS DOCUMENTS
# ---------------------------------------------------------

def process_documents():

    rows = []

    files = list(INPUT_DIR.glob("*.txt"))

    print(f"Processing {len(files)} documents")

    for file in files:

        document_id = file.stem

        with open(file, "r", encoding="utf-8") as f:
            text = f.read()
        
        text = re.sub(r"\s+\n", "\n", text)

        chunks = split_text(text)

        valid_chunks = [c for c in chunks if is_valid_chunk(c)]

        meta = metadata_map.get(document_id)

        if meta is None:
            meta = {}
        
        tags_raw = meta.get("tags", "")

        if isinstance(tags_raw, str):
            tags_list = [t.strip() for t in tags_raw.split(",") if t.strip()]
        else:
            tags_list = []

        tags_str = ", ".join(tags_list)

        for i, chunk in enumerate(valid_chunks):

            rows.append(
                {
                    "chunk_id": str(uuid.uuid4()),
                    "document_id": document_id,
                    "chunk_index": i,
                    "text": chunk,
                    "document_title": meta.get("title", ""),
                    "source": meta.get("source", ""),
                    "year": meta.get("year", ""),
                    "document_type": meta.get("document_type", ""),
                    "tags": tags_str,
                    "section": detect_section(chunk)
                }
            )

    seen = set()
    filtered_rows = []

    for row in rows:
        if row["text"] not in seen:
            seen.add(row["text"])
            filtered_rows.append(row)

    df = pd.DataFrame(filtered_rows)

    df.to_csv(OUTPUT_FILE, index=False)

    print(f"Chunks saved to {OUTPUT_FILE}")
    print(f"Total chunks: {len(df)}")


# ---------------------------------------------------------
# RUN
# ---------------------------------------------------------

if __name__ == "__main__":
    process_documents()