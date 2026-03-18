import os
import re
import uuid
import hashlib
from pathlib import Path
from datetime import datetime
import ollama
import json
import unicodedata
import time

import fitz  # PyMuPDF
import pandas as pd

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_PATHS = {
    "inca": PROJECT_ROOT / "data/raw/inca",
    "pcdt": PROJECT_ROOT / "data/raw/pcdt",
}

OUTPUT_METADATA = PROJECT_ROOT / "data/metadata/documents_metadata.csv"

DOC_TYPES = [
    "clinical_guideline",      # PCDT e diretrizes clínicas
    "technical_manual",        # manuais técnicos para profissionais
    "patient_education",       # cartilhas para pacientes
    "institutional_guide",     # guias institucionais / apresentação
]

# ---------------------------------------------------------
# UTILS
# ---------------------------------------------------------

def calculate_file_hash(file_path):
    """Generate SHA256 hash for deduplication."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            sha256.update(chunk)
    return sha256.hexdigest()


def get_pdf_pages(file_path):
    """Return number of pages in PDF."""
    doc = fitz.open(file_path)
    pages = len(doc)
    doc.close()
    return pages


def extract_initial_text(file_path, max_pages=1, max_chars=2500):
    doc = fitz.open(file_path)
    parts = []

    for i in range(min(max_pages, len(doc))):
        parts.append(doc[i].get_text())

    doc.close()

    text = "\n".join(parts)
    text = fix_encoding(text)
    text = text.replace("\x00", "")
    text = re.sub(r"\s+", " ", text).strip()

    return text[:max_chars]


def detect_year(text, filename):
    """Try to detect year."""
    match = re.search(r"(20\d{2})", text)
    if match:
        return int(match.group(1))

    match = re.search(r"(20\d{2})", filename)
    if match:
        return int(match.group(1))

    return None


def detect_title(text):
    """Extract probable title from first page."""
    lines = text.split("\n")

    for line in lines[:10]:
        if len(line) > 20:
            return line.strip()

    return None

def classify_doc_type(text, filename, source):
    """
    Classify document type based on heuristics.
    """

    text_lower = text.lower()
    filename_lower = filename.lower()

    # -------------------------
    # CLINICAL GUIDELINES
    # -------------------------
    if source == "pcdt":
        return "clinical_guideline"

    if "protocolo clínico" in text_lower:
        return "clinical_guideline"

    if "diretrizes diagnósticas" in text_lower:
        return "clinical_guideline"

    if "diretrizes terapêuticas" in text_lower:
        return "clinical_guideline"

    # -------------------------
    # PATIENT EDUCATION
    # -------------------------
    if "orientações ao paciente" in text_lower:
        return "patient_education"

    if "cartilha" in text_lower:
        return "patient_education"

    if "prezado paciente" in text_lower:
        return "patient_education"

    # -------------------------
    # TECHNICAL MANUAL
    # -------------------------
    if "manual" in text_lower:
        return "technical_manual"

    if "bases técnicas" in text_lower:
        return "technical_manual"

    # -------------------------
    # INSTITUTIONAL GUIDE
    # -------------------------
    if "conheça o hospital" in text_lower:
        return "institutional_guide"

    if "instituto nacional de câncer" in text_lower and "serviços" in text_lower:
        return "institutional_guide"

    # -------------------------
    # INSTITUTIONAL DOCUMENT
    # -------------------------
    if "carta de serviços" in text_lower:
        return "institutional_document"

    if "serviços ao usuário" in text_lower:
        return "institutional_document"

    # fallback
    return "unknown"

def slugify(text):
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.lower()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"\s+", "_", text)
    return text[:80]


def standardize_filename(source, year, title, original_name):
    """Create standardized filename."""
    if title:
        title_slug = slugify(title)
    else:
        title_slug = slugify(Path(original_name).stem)

    year_str = str(year) if year else "unknown"

    return f"{source}_{year_str}_{title_slug}.pdf"

def generate_metadata_llm(text):
    """Generate title, summary and tags using local LLM."""

    prompt = f"""
        Você é um assistente especializado em documentos médicos do SUS.

        Analise o texto abaixo e retorne:

        1) title → título curto do documento
        2) summary → resumo em até 240 caracteres
        3) tags → lista de até 10 tags relevantes

        Responda SOMENTE em JSON no formato:

        {{
        "title": "...",
        "summary": "...",
        "tags": ["tag1","tag2"]
        }}

        Texto:
        {text}
    """

    response = ollama.chat(
        model="mistral",
        messages=[{"role": "user", "content": prompt}]
    )

    content = response["message"]["content"]

    content = content.strip()

    if "```" in content:
        content = content.split("```")[1]

    try:
        return json.loads(content)
    except:
        return {
            "title": None,
            "summary": None,
            "tags": []
        }

def rename_file(original_path, new_name):
    """Rename file safely, avoiding name collisions."""
    
    new_path = original_path.parent / new_name

    counter = 1

    while new_path.exists():
        stem = Path(new_name).stem
        suffix = Path(new_name).suffix
        new_path = original_path.parent / f"{stem}_{counter}{suffix}"
        counter += 1

    original_path.rename(new_path)

    return new_path

def fix_encoding(text):
    """Fix common UTF-8 / Latin1 encoding issues."""
    try:
        return text.encode("latin1").decode("utf-8")
    except:
        return text

# ---------------------------------------------------------
# MAIN PROCESS
# ---------------------------------------------------------

def process_documents():
    records = []

    start_time = datetime.now()
    start_ts = time.time()

    print(f"\nStart time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    # coletar todos os PDFs
    all_pdfs = []
    for source, folder in RAW_PATHS.items():
        for pdf_path in folder.glob("*.pdf"):
            all_pdfs.append((source, pdf_path))

    total = len(all_pdfs)

    print(f"Total documents found: {total}\n")

    for idx, (source, pdf_path) in enumerate(all_pdfs, start=1):

        original_name = pdf_path.name

        doc_start = time.time()

        print(f"[{idx}/{total}] Processing {original_name}")

        try:
            file_size_mb = round(os.path.getsize(pdf_path) / (1024 * 1024), 2)

            pages = get_pdf_pages(pdf_path)

            initial_text = extract_initial_text(pdf_path, max_pages=2)

            doc_type = classify_doc_type(
                initial_text,
                pdf_path.name,
                source
            )

            llm_metadata = generate_metadata_llm(initial_text)

            title = llm_metadata.get("title")
            summary = llm_metadata.get("summary")
            tags = llm_metadata.get("tags")

            year = detect_year(initial_text, pdf_path.name)

            standardized_name = standardize_filename(
                source,
                year,
                title,
                pdf_path.name
            )

            new_path = rename_file(pdf_path, standardized_name)
            pdf_path = new_path

            document_id = str(uuid.uuid4())

            file_hash = calculate_file_hash(pdf_path)

            record = {
                "document_id": document_id,
                "source": source,
                "title": title,
                "summary": summary,
                "tags": ", ".join(tags) if isinstance(tags, list) else "",
                "year": year,
                "file_name_original": original_name,
                "file_name_standardized": pdf_path.name,
                "file_path": str(pdf_path),
                "file_size_mb": file_size_mb,
                "pages": pages,
                "language": "pt",
                "document_type": doc_type,
                "hash_sha256": file_hash,
                "url": None,
                "created_at": datetime.utcnow().isoformat()
            }

            print("Record:")
            for key, value in record.items():
                print(f"  {key}: {value}")

            records.append(record)

            doc_elapsed = round(time.time() - doc_start, 2)

            print(f"  ✓ done in {doc_elapsed}s")

        except Exception as e:
            print(f"  ! error processing {pdf_path.name}: {e}")

    df = pd.DataFrame(records)

    OUTPUT_METADATA.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(OUTPUT_METADATA, index=False, encoding="utf-8")

    end_time = datetime.now()
    total_elapsed = round(time.time() - start_ts, 2)

    print(f"\nEnd time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total duration: {total_elapsed} seconds ({round(total_elapsed/60,2)} minutes)")
    print(f"\nMetadata file created: {OUTPUT_METADATA}")

# ---------------------------------------------------------
# RUN
# ---------------------------------------------------------

if __name__ == "__main__":
    process_documents()