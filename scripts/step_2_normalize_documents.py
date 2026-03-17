import re
import fitz
import pandas as pd
from pathlib import Path

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

METADATA_PATH = PROJECT_ROOT / "data/metadata/documents_metadata.csv"

OUTPUT_DIR = PROJECT_ROOT / "data/processed"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------
# TEXT CLEANING FUNCTIONS
# ---------------------------------------------------------

def fix_encoding(text):
    try:
        return text.encode("latin1").decode("utf-8")
    except:
        return text


def remove_page_numbers(text):
    """
    Remove lines that are only numbers (typical page numbers).
    """
    lines = text.split("\n")
    cleaned = []

    for line in lines:
        if re.fullmatch(r"\d{1,4}", line.strip()):
            continue
        cleaned.append(line)

    return "\n".join(cleaned)


def remove_headers_footers(text):

    patterns = [
        r"minist[eé]rio da sa[uú]de",
        r"secretaria de aten[cç][aã]o",
        r"instituto nacional de c[aâ]ncer",
        r"inca",
        r"rio de janeiro",
        r"impresso no brasil",
        r"todos os direitos reservados",
        r"portaria.*",
        r"anexo",
    ]

    lines = text.split("\n")
    cleaned = []

    for line in lines:

        l = line.lower().strip()

        if any(re.search(p, l) for p in patterns):
            continue

        cleaned.append(line)

    return "\n".join(cleaned)

def remove_table_artifacts(text):
    """
    Remove table fragments extracted from PDF.
    """

    lines = text.split("\n")
    cleaned = []

    for line in lines:

        l = line.strip()

        # remove linhas com muitos números
        if re.fullmatch(r"[\d\.,\s]+", l):
            continue

        cleaned.append(line)

    return "\n".join(cleaned)

def remove_short_lines(text):

    lines = text.split("\n")
    cleaned = []

    for line in lines:

        if len(line.strip()) < 25:
            continue

        cleaned.append(line)

    return "\n".join(cleaned)

def fix_line_breaks(text):
    """
    Merge broken sentences caused by PDF extraction.
    """

    text = re.sub(r"\n(?=[a-z])", " ", text)

    return text


def normalize_whitespace(text):
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def remove_editorial_sections(text):
    """
    Remove common editorial sections.
    """

    patterns = [
        r"ficha catalogr[aá]fica.*",
        r"copyright.*",
        r"catalogação na fonte.*",
    ]

    lines = text.split("\n")
    cleaned = []

    for line in lines:

        if any(re.search(p, line.lower()) for p in patterns):
            continue
        cleaned.append(line)

    return "\n".join(cleaned)

def fix_hyphenation(text):
    """
    Fix words broken by PDF hyphenation.
    """

    text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)

    return text


# ---------------------------------------------------------
# PDF EXTRACTION
# ---------------------------------------------------------

def extract_pdf_text(file_path):
    doc = fitz.open(file_path)
    pages = []
    for page in doc:
        pages.append(page.get_text())
    doc.close()
    return "\n".join(pages)


# ---------------------------------------------------------
# NORMALIZATION PIPELINE
# ---------------------------------------------------------

def normalize_document(file_path):
    text = extract_pdf_text(file_path)
    text = fix_encoding(text)
    text = remove_headers_footers(text)
    text = remove_page_numbers(text)
    text = remove_editorial_sections(text)
    text = remove_table_artifacts(text)
    text = remove_short_lines(text)
    text = fix_hyphenation(text)
    text = fix_line_breaks(text)
    text = normalize_whitespace(text)

    return text


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def process_documents():
    df = pd.read_csv(
        METADATA_PATH,
        encoding="utf-8",
        sep=";",
        quotechar='"',
        engine="python"
    )
    total = len(df)
    print(f"Processing {total} documents")
    for idx, row in df.iterrows():
        file_path = Path(row["file_path"])
        document_id = row["document_id"]
        print(f"[{idx+1}/{total}] {file_path.name}")
        try:
            clean_text = normalize_document(file_path)
            output_file = OUTPUT_DIR / f"{document_id}.txt"
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(clean_text)

        except Exception as e:
            print("Error:", e)


# ---------------------------------------------------------
# RUN
# ---------------------------------------------------------

if __name__ == "__main__":
    process_documents()