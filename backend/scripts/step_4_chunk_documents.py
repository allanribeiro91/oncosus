import pandas as pd
from pathlib import Path
import uuid
import re

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

METADATA_PATH = PROJECT_ROOT / "data/metadata/documents_metadata.csv"
INPUT_DIR = PROJECT_ROOT / "data/processed/v3_final"
OUTPUT_FILE = PROJECT_ROOT / "data/chunks/chunks.csv"

CHUNK_SIZE = 1300
OVERLAP = 100

metadata_df = pd.read_csv(METADATA_PATH, sep=';')

metadata_map = {
    row["file_name_standardized"]: row
    for _, row in metadata_df.iterrows()
}

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------
# 1. SPLIT POR SEÇÕES (BASE DO PIPELINE)
# ---------------------------------------------------------

def split_sections(text):
    """
    Divide o documento por headings reais (#, ##, ###)
    """
    pattern = r"(#{1,3}\s*\d+.*)"
    parts = re.split(pattern, text)

    sections = []
    current_title = "unknown"

    for part in parts:
        part = part.strip()
        if not part:
            continue

        if part.startswith("#"):
            current_title = part
        else:
            sections.append((current_title, part))

    return sections


# ---------------------------------------------------------
# 2. SPLIT POR BLOCOS INTERNOS
# ---------------------------------------------------------

def split_blocks(text):
    """
    Divide em blocos semânticos:
    - parágrafos
    - listas
    """
    text = text.replace("\r", "\n")

    # separa listas fortes
    text = re.sub(r"\n(?=\s*[•\-]\s)", "\n\n", text)

    # separa parágrafos
    blocks = [b.strip() for b in text.split("\n\n") if b.strip()]

    return blocks

def merge_bullets(blocks):

    merged = []
    buffer = ""

    for b in blocks:
        if re.match(r"^\s*[•\-–\*]", b):
            buffer += " " + b
        else:
            if buffer:
                merged.append(buffer.strip())
                buffer = ""
            merged.append(b)

    if buffer:
        merged.append(buffer.strip())

    return merged


def merge_small_blocks(blocks, min_size=300):

    merged = []
    buffer = ""

    for b in blocks:

        if len(b) < min_size:
            buffer += " " + b
        else:
            if buffer:
                merged.append(buffer.strip())
                buffer = ""
            merged.append(b)

    if buffer:
        merged.append(buffer.strip())

    return merged


# ---------------------------------------------------------
# 3. CHUNKING COM OVERLAP (POR BLOCO)
# ---------------------------------------------------------

def chunk_blocks(blocks, max_chars=1000, overlap=150):

    chunks = []
    current = ""

    for block in blocks:

        # ignora ruído
        if len(block) < 40:
            continue

        # 🔥 quebra blocos muito grandes em sentenças
        if len(block) > 1500:
            sentences = re.split(r'(?<=[.!?]) +', block)
        else:
            sentences = [block]

        for sentence in sentences:

            if len(current) + len(sentence) <= max_chars:
                current += " " + sentence if current else sentence

            else:
                if current:
                    chunks.append(current.strip())

                # 🔥 overlap real
                if overlap > 0 and chunks:
                    tail = current[-overlap:]
                    current = tail + " " + sentence
                else:
                    current = sentence

    if current:
        chunks.append(current.strip())

    return chunks


# ---------------------------------------------------------
# 4. ENRIQUECIMENTO DE CONTEXTO
# ---------------------------------------------------------

def enrich_chunk(chunk, section_title):
    """
    Injeta contexto da seção no chunk
    """
    clean_section = re.sub(r"#", "", section_title).strip()
    return f"[SECTION: {clean_section}]\n{chunk}"


# ---------------------------------------------------------
# 5. DETECÇÃO DE SEÇÃO
# ---------------------------------------------------------

def detect_section(section_title):

    title = section_title.lower()

    if "tratamento" in title:
        return "treatment"

    if "diagnóstico" in title:
        return "diagnosis"

    if "prognóstico" in title:
        return "prognosis"

    if "introdução" in title:
        return "introduction"

    if "classificação" in title:
        return "classification"

    if "critérios" in title:
        return "criteria"

    return "other"


# ---------------------------------------------------------
# 7. PROCESSAMENTO PRINCIPAL
# ---------------------------------------------------------
def is_valid_chunk_light(text):
    return len(text) > 150

def process_documents():

    rows = []
    files = list(INPUT_DIR.glob("*.txt"))

    print(f"Processing {len(files)} documents")

    for file in files:

        document_id = file.stem

        with open(file, "r", encoding="utf-8") as f:
            text = f.read()

        text = re.sub(r"\s+\n", "\n", text)

        sections = split_sections(text)

        meta = metadata_map.get(document_id, {})

        tags_raw = meta.get("tags", "")
        tags_list = [t.strip() for t in tags_raw.split(",") if t.strip()] if isinstance(tags_raw, str) else []
        tags_str = ", ".join(tags_list)

        chunk_index = 0

        for section_title, section_text in sections:

            blocks = split_blocks(section_text)
            blocks = merge_bullets(blocks)
            blocks = merge_small_blocks(blocks)
            chunks = chunk_blocks(blocks, CHUNK_SIZE, OVERLAP)

            for chunk in chunks:
                if not is_valid_chunk_light(chunk):
                    continue

                enriched = enrich_chunk(chunk, section_title)

                rows.append({
                    "chunk_id": str(uuid.uuid4()),
                    "document_id": document_id,
                    "chunk_index": chunk_index,
                    "text": enriched,
                    "document_title": meta.get("title", ""),
                    "source": meta.get("source", ""),
                    "year": meta.get("year", ""),
                    "document_type": meta.get("document_type", ""),
                    "tags": tags_str,
                    "section": detect_section(section_title)
                })

                chunk_index += 1

    # remove duplicados
    seen = set()
    filtered = []

    for row in rows:
        if row["text"] not in seen:
            seen.add(row["text"])
            filtered.append(row)

    df = pd.DataFrame(filtered)
    df.to_csv(OUTPUT_FILE, index=False)

    print(f"Chunks saved to {OUTPUT_FILE}")
    print(f"Total chunks: {len(df)}")


# ---------------------------------------------------------
# RUN
# ---------------------------------------------------------

if __name__ == "__main__":
    process_documents()