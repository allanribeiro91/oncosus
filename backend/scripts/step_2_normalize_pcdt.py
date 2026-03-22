import re
import fitz
import pandas as pd
from pathlib import Path

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

METADATA_PATH = PROJECT_ROOT / "data/metadata/documents_metadata.csv"

OUTPUT_DIR = PROJECT_ROOT / "data/processed/v1_clean"

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
        r"secretaria de ciência",
        r"atenção especializada",
        r"comissão nacional",
    ]

    lines = text.split("\n")
    cleaned = []

    for line in lines:

        l = line.lower().strip()

        if any(re.match(rf"^{p}$", l) for p in patterns):
            continue

        cleaned.append(line)

    return "\n".join(cleaned)

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

def fix_broken_sentences(text):
    lines = text.split("\n")
    new_lines = []

    for i in range(len(lines)):
        line = lines[i].strip()

        if not line:
            new_lines.append("")
            continue

        # próxima linha
        if i < len(lines) - 1:
            next_line = lines[i + 1].strip()

            # 🔴 REGRA: só junta se for claramente continuação
            if (
                line and
                not re.search(r'[.!?:]$', line) and   # não termina frase
                next_line and
                next_line[0].islower()               # próxima começa minúscula
            ):
                lines[i + 1] = line + " " + next_line
                continue

        new_lines.append(line)

    return "\n".join(new_lines)

def fix_section_breaks(text):
    # Junta número + título
    text = re.sub(r"(\d+)\.\s*\n\s*([A-ZÇÁÉÍÓÚÂÊÔÃÕ ]+)", r"\1. \2", text)

    return text

def fix_broken_words(text):

    # junta palavras quebradas por newline
    text = re.sub(r"(\w+)\n(\w+)", r"\1\2", text)

    # corrige casos comuns tipo "se\n" ou "lo\n"
    text = re.sub(r"\b(se|lo|la|nos|lhe)\n", r"\1 ", text)

    return text

def fix_broken_lines(text):
    lines = text.split("\n")
    merged = []

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        if not line:
            merged.append("")
            i += 1
            continue

        # se for a última linha, apenas adiciona
        if i == len(lines) - 1:
            merged.append(line)
            break

        next_line = lines[i + 1].strip()

        # não juntar se a próxima estiver vazia
        if not next_line:
            merged.append(line)
            i += 1
            continue

        # não juntar títulos/seções/subseções
        if re.match(r"^\d+[\.\-]?\s+[A-ZÁÉÍÓÚÂÊÔÃÕÇ]", next_line):
            merged.append(line)
            i += 1
            continue

        if re.match(r"^\d+\.\d+", next_line):
            merged.append(line)
            i += 1
            continue

        # não juntar bullets/listas
        if next_line.startswith("-") or line.startswith("-"):
            merged.append(line)
            i += 1
            continue

        # não juntar blocos muito curtos e estruturais
        if len(line) < 3 or len(next_line) < 3:
            merged.append(line)
            i += 1
            continue

        # juntar quando parece continuação natural de parágrafo
        if (
            not re.search(r"[.!?:;]$", line)
            and (
                next_line[0].islower()
                or next_line[0].isdigit()
                or next_line.startswith("(")
            )
        ):
            lines[i + 1] = line + " " + next_line
        else:
            merged.append(line)

        i += 1

    # remove quebras excessivas
    text = "\n".join(merged)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text


def remove_pcdt_frontmatter(text):
    """
    Remove completamente a parte institucional do PCDT,
    mantendo apenas a partir de '1 - INTRODUÇÃO'
    """

    # padrão robusto (variações comuns)
    patterns = [
        r"1\s*-\s*INTRODU[CÇ][ÃA]O",
        r"1\.\s*INTRODU[CÇ][ÃA]O",
        r"INTRODU[CÇ][ÃA]O"
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)

        if match:
            return text[match.start():]

    return text  # fallback

def fix_broken_lines_advanced(text):

    import re

    lines = text.split("\n")
    merged = []

    i = 0
    while i < len(lines):

        line = lines[i].strip()

        if not line:
            merged.append("")
            i += 1
            continue

        # tenta juntar com próxima linha
        if i + 1 < len(lines):
            next_line = lines[i + 1].strip()

            if (
                # linha atual NÃO termina frase
                not re.search(r"[.!?:]$", line)
                # próxima começa minúscula ou número
                and (
                    next_line[:1].islower()
                    or next_line[:1].isdigit()
                    or next_line.startswith("(")
                )
            ):
                merged.append(line + " " + next_line)
                i += 2
                continue

        merged.append(line)
        i += 1

    return "\n".join(merged)

def is_table_block(lines, i):

    count = 0

    for j in range(i, min(i + 5, len(lines))):
        line = lines[j].strip()

        if (
            len(line.split()) <= 4
            and not line.endswith(".")
            and not line.endswith(":")
        ):
            count += 1

    return count >= 3

# ---------------------------------------------------------
# PDF EXTRACTION
# ---------------------------------------------------------

def extract_pdf_text(file_path):
    doc = fitz.open(file_path)
    pages = []

    for page in doc:
        blocks = page.get_text("blocks")

        # ordena por posição (top → bottom)
        blocks = sorted(blocks, key=lambda b: (b[1], b[0]))

        page_text = "\n".join([b[4] for b in blocks])
        pages.append(page_text)

    doc.close()
    return "\n".join(pages)




def process_clinical(text):

    # ---------------------------------------------------------
    # 1. LIMPEZA BASE
    # ---------------------------------------------------------

    text = fix_encoding(text)

    # 🔥 corta tudo antes do início real
    if "PROTOCOLO CLÍNICO E DIRETRIZES TERAPÊUTICAS" in text:
        text = text.split("PROTOCOLO CLÍNICO E DIRETRIZES TERAPÊUTICAS", 1)[1]

    # remove portarias e blocos legais
    text = re.sub(r"PORTARIA.*?Art\. 5º.*?\n", "", text, flags=re.DOTALL)

    # remove números de página
    text = remove_page_numbers(text)

    # remove headers/footers (usa sua função existente)
    text = remove_headers_footers(text)

    # ---------------------------------------------------------
    # 2. NORMALIZAÇÃO
    # ---------------------------------------------------------

    text = fix_hyphenation(text)
    text = fix_broken_words(text)
    text = fix_section_breaks(text)

    # remove múltiplas quebras
    text = re.sub(r"\n{3,}", "\n\n", text)

    # ---------------------------------------------------------
    # 3. PARSE SEMÂNTICO
    # ---------------------------------------------------------

    lines = text.split("\n")

    structured = []

    current_section = None
    current_subsection = None
    buffer = []

    section_pattern = re.compile(r"^\d+\.\s+[A-ZÇÃÉÍÓÚ ]+$")
    subsection_pattern = re.compile(r"^\d+\.\d+(\.\d+)?\.?\s+.+$")

    def flush_buffer():
        nonlocal buffer, current_section, current_subsection

        if not buffer:
            return

        content = " ".join(buffer).strip()

        if content:
            structured.append({
                "section": current_section,
                "subsection": current_subsection,
                "text": content
            })

        buffer = []

    for line in lines:
        line = line.strip()

        if not line:
            continue

        # 🔹 seção principal
        if section_pattern.match(line):
            flush_buffer()
            current_section = line
            current_subsection = None
            continue

        # 🔹 subseção
        if subsection_pattern.match(line):
            flush_buffer()
            current_subsection = line
            continue

        # 🔹 conteúdo normal
        buffer.append(line)

    # flush final
    flush_buffer()

    # ---------------------------------------------------------
    # 4. FORMATAÇÃO FINAL (STRING)
    # ---------------------------------------------------------

    output_blocks = []

    for item in structured:

        section = item["section"] or ""
        subsection = item["subsection"] or ""
        content = item["text"]

        block = []

        if section:
            block.append(f"### {section}")

        if subsection:
            block.append(f"## {subsection}")

        block.append(content)

        output_blocks.append("\n".join(block))

    return "\n\n".join(output_blocks)




def process_pcdt_type_1(text):

    # ---------------------------------------------------------
    # 1. LIMPEZA BASE
    # ---------------------------------------------------------

    text = fix_encoding(text)

    # 🔥 corta início real
    if "DIRETRIZES DIAGNÓSTICAS E TERAPÊUTICAS" in text:
        text = text.split("DIRETRIZES DIAGNÓSTICAS E TERAPÊUTICAS", 1)[1]

    # remove blocos legais grandes
    text = re.sub(r"PORTARIA.*?Art\..*?\n", "", text, flags=re.DOTALL)

    # remove referências (final do documento)
    text = re.split(r"\n\s*\d+\s+REFERÊNCIAS", text, flags=re.IGNORECASE)[0]

    text = remove_page_numbers(text)
    text = remove_headers_footers(text)
    text = remove_editorial_sections(text)

    # ---------------------------------------------------------
    # 2. NORMALIZAÇÃO
    # ---------------------------------------------------------

    text = fix_hyphenation(text)
    text = fix_broken_words(text)
    text = fix_section_breaks(text)
    text = fix_broken_sentences(text)

    # remove múltiplas quebras
    text = re.sub(r"\n{3,}", "\n\n", text)

    # ---------------------------------------------------------
    # 3. PRÉ-ESTRUTURAÇÃO (IMPORTANTE)
    # ---------------------------------------------------------

    # 🔹 garante quebra antes de seções tipo "3 DIAGNÓSTICO"
    text = re.sub(r"\n(\d{1,2}\s+[A-ZÇÁÉÍÓÚÂÊÔÃÕ ]+)", r"\n\n\1", text)

    # 🔹 garante quebra antes de subseções tipo "3.1"
    text = re.sub(r"\n(\d+\.\d+)", r"\n\n\1", text)

    # ---------------------------------------------------------
    # 4. PARSE SEMÂNTICO
    # ---------------------------------------------------------

    lines = text.split("\n")

    structured = []

    current_section = None
    current_subsection = None
    buffer = []

    # 🔥 ajuste crítico aqui
    section_pattern = re.compile(r"^\d{1,2}\s+[A-ZÇÁÉÍÓÚÂÊÔÃÕ ]+$")
    subsection_pattern = re.compile(r"^\d+\.\d+(\.\d+)?\.?\s+.+$")

    def flush_buffer():
        nonlocal buffer, current_section, current_subsection

        if not buffer:
            return

        content = " ".join(buffer).strip()

        if content:
            structured.append({
                "section": current_section,
                "subsection": current_subsection,
                "text": content
            })

        buffer = []

    for line in lines:
        line = line.strip()

        if not line:
            continue

        # 🔹 seção principal (ex: "3 DIAGNÓSTICO E ESTADIAMENTO")
        if section_pattern.match(line):
            flush_buffer()
            current_section = line
            current_subsection = None
            continue

        # 🔹 subseção (ex: "3.1 DIAGNÓSTICO CLÍNICO")
        if subsection_pattern.match(line):
            flush_buffer()
            current_subsection = line
            continue

        # 🔹 ignora linhas muito curtas isoladas (ruído típico)
        if len(line) < 3:
            continue

        buffer.append(line)

    flush_buffer()

    # ---------------------------------------------------------
    # 5. FORMATAÇÃO FINAL
    # ---------------------------------------------------------

    output_blocks = []

    for item in structured:

        section = item["section"] or ""
        subsection = item["subsection"] or ""
        content = item["text"]

        block = []

        if section:
            block.append(f"### {section}")

        if subsection:
            block.append(f"## {subsection}")

        block.append(content)

        output_blocks.append("\n".join(block))

    return "\n\n".join(output_blocks)

def process_pcdt_type_2(text):

    # ---------------------------------------------------------
    # 1. LIMPEZA BASE
    # ---------------------------------------------------------

    text = fix_encoding(text)

    # 🔥 corta início após DIRETRIZES
    if "DIRETRIZES DIAGNÓSTICAS E TERAPÊUTICAS" in text:
        text = text.split("DIRETRIZES DIAGNÓSTICAS E TERAPÊUTICAS", 1)[1]

    # remove bloco da portaria
    text = re.sub(r"PORTARIA.*?Art\..*?\n", "", text, flags=re.DOTALL)

    # remove referências finais
    text = re.split(r"\n\s*12\s*-\s*REFERÊNCIAS", text, flags=re.IGNORECASE)[0]

    text = remove_page_numbers(text)
    text = remove_headers_footers(text)
    text = remove_editorial_sections(text)

    # ---------------------------------------------------------
    # 2. NORMALIZAÇÃO
    # ---------------------------------------------------------

    text = fix_hyphenation(text)
    text = fix_broken_words(text)
    text = fix_broken_sentences(text)

    # 🔥 separa título colado
    text = re.sub(
        r"([A-ZÇÁÉÍÓÚÂÊÔÃÕ ]{10,})\s+(\d+-)",
        r"\1\n\n\2",
        text
    )

    # 🔥 garante quebra antes de seções
    text = re.sub(r"\n(\d+\s*-\s+)", r"\n\n\1", text)

    # 🔥 garante quebra antes de subseções
    text = re.sub(r"\n(\d+\.\d+)", r"\n\n\1", text)

    text = re.sub(r"\n{3,}", "\n\n", text)

    # ---------------------------------------------------------
    # 3. PARSE SEMÂNTICO
    # ---------------------------------------------------------

    lines = text.split("\n")

    structured = []

    current_section = None
    current_subsection = None
    current_subsub = None
    buffer = []

    section_pattern = re.compile(r"^\d+\s*-\s+.+$")
    subsection_pattern = re.compile(r"^\d+\.\d+\.?\s+.+$")

    def flush_buffer():
        nonlocal buffer, current_section, current_subsection, current_subsub

        if not buffer:
            return

        content = " ".join(buffer).strip()

        if content:
            structured.append({
                "section": current_section,
                "subsection": current_subsection,
                "subsub": current_subsub,
                "text": content
            })

        buffer = []

    for line in lines:
        line = line.strip()

        if not line:
            continue

        # 🔹 seção principal (1- INTRODUÇÃO)
        if section_pattern.match(line):
            flush_buffer()
            current_section = line
            current_subsection = None
            current_subsub = None
            continue

        # 🔹 subseção (9.1 ...)
        if subsection_pattern.match(line):
            flush_buffer()
            current_subsection = line
            current_subsub = None
            continue

        # 🔹 subsub (ex: "Terapia de Indução –")
        if line.endswith("–") and len(line) < 80:
            flush_buffer()
            current_subsub = line
            continue

        # 🔹 mantém listas com "-"
        if line.startswith("- "):
            buffer.append(line)
            continue

        buffer.append(line)

    flush_buffer()

    # ---------------------------------------------------------
    # 4. FORMATAÇÃO FINAL
    # ---------------------------------------------------------

    output_blocks = []

    for item in structured:

        block = []

        if item["section"]:
            block.append(f"### {item['section']}")

        if item["subsection"]:
            block.append(f"## {item['subsection']}")

        if item["subsub"]:
            block.append(f"# {item['subsub']}")

        block.append(item["text"])

        output_blocks.append("\n".join(block))

    return "\n\n".join(output_blocks)

def process_pcdt_type_3(text):

    text = fix_encoding(text)

    # -------------------------------------------------
    # 1. REMOVER FRONTMATTER (PORTARIA)
    # -------------------------------------------------
    if "ANEXO" in text:
        text = text.split("ANEXO", 1)[1]

    # -------------------------------------------------
    # 2. REMOVER HEADER/FOOTER
    # -------------------------------------------------
    text = remove_headers_footers(text)
    text = remove_page_numbers(text)

    # -------------------------------------------------
    # 3. NORMALIZAÇÃO
    # -------------------------------------------------
    text = fix_hyphenation(text)
    text = fix_broken_words(text)
    text = fix_broken_lines(text)

    # -------------------------------------------------
    # 4. LINHAS
    # -------------------------------------------------
    lines = text.split("\n")

    result = []

    current_section = None
    current_subsection = None

    for line in lines:
        line = line.strip()

        if not line:
            continue

        # -------------------------------
        # STOP EM REFERÊNCIAS
        # -------------------------------
        if re.match(r"^11\s+Refer", line, re.IGNORECASE):
            break

        # -------------------------------
        # SECTION (1. / 2. / 6 )
        # -------------------------------
        if re.match(r"^\d+\.\s+[A-Z]", line) or re.match(r"^\d+\s+[A-Z]", line):
            current_section = line
            current_subsection = None
            result.append(f"\n# {line}\n")
            continue

        # -------------------------------
        # SUBSECTION (4.1)
        # -------------------------------
        if re.match(r"^\d+\.\d+", line):
            current_subsection = line
            result.append(f"\n## {line}\n")
            continue

        # -------------------------------
        # BLOCO TÉCNICO (TNM)
        # -------------------------------
        if any(keyword in line for keyword in ["TUMOR (T)", "LINFONODO", "METÁSTASE"]):
            result.append(f"\n### BLOCO_TECNICO\n{line}")
            continue

        # -------------------------------
        # TEXTO NORMAL
        # -------------------------------
        result.append(line)

    return "\n".join(result)

def process_pcdt_type_4(text):

    text = fix_encoding(text)

    # -------------------------------------------------
    # 1. REMOVER FRONTMATTER
    # -------------------------------------------------
    if "ANEXO" in text:
        text = text.split("ANEXO", 1)[1]

    # -------------------------------------------------
    # 2. LIMPEZA BASE
    # -------------------------------------------------
    text = remove_headers_footers(text)
    text = remove_page_numbers(text)

    text = fix_hyphenation(text)
    text = fix_broken_words(text)
    text = fix_broken_lines(text)

    lines = text.split("\n")
    result = []

    for line in lines:
        line = line.strip()

        if not line:
            continue

        # -------------------------------
        # STOP REFERÊNCIAS
        # -------------------------------
        if re.search(r"(refer[eê]ncias|bibliogr[aá]ficas)", line, re.IGNORECASE):
            break

        # -------------------------------
        # REMOVER HEADER INSTITUCIONAL
        # -------------------------------
        if "Ministério da Saúde" in line or "Secretaria de Atenção" in line:
            continue

        # -------------------------------
        # REMOVER TABELA QUEBRADA
        # -------------------------------
        if (
            len(line.split()) <= 3 and line.isupper()
        ) or re.match(r"^(Estágio|Tumor|Invasão)", line):
            continue

        # -------------------------------
        # SECTION
        # -------------------------------
        if re.match(r"^\d+\.?\s+[A-ZÁ-Ú]", line):
            result.append(f"\n# {line}\n")
            continue

        # -------------------------------
        # SUBSECTION
        # -------------------------------
        if re.match(r"^\d+\.\d+", line):
            result.append(f"\n## {line}\n")
            continue

        # -------------------------------
        # SUB-SUBSECTION (4.3.1)
        # -------------------------------
        if re.match(r"^\d+\.\d+\.\d+", line):
            result.append(f"\n### {line}\n")
            continue

        # -------------------------------
        # LISTAS (a), b), c))
        # -------------------------------
        if re.match(r"^[a-z]\)", line):
            result.append(f"- {line}")
            continue

        # -------------------------------
        # TNM / CLASSIFICAÇÃO (T1, N0, M1)
        # -------------------------------
        if re.match(r"^[TNM]\d", line):
            line = line.replace("–", ":")
            result.append(f"- {line}")
            continue

        # -------------------------------
        # CID (C00, C01...)
        # -------------------------------
        if re.match(r"^C\d{2}", line):
            result.append(f"- {line}")
            continue

        # -------------------------------
        # ESTÁGIOS (Estágio I, II...)
        # -------------------------------
        if re.match(r"^Estágio", line, re.IGNORECASE):
            result.append(f"\n### {line}\n")
            continue

        # -------------------------------
        # TEXTO NORMAL
        # -------------------------------
        result.append(line)

    return "\n".join(result)

def process_pcdt_type_5(text):

    text = fix_encoding(text)

    text = remove_pcdt_frontmatter(text)

    text = remove_headers_footers(text)
    text = remove_page_numbers(text)

    text = fix_hyphenation(text)
    text = fix_broken_words(text)
    text = fix_broken_lines(text)

    lines = text.split("\n")
    result = []

    current_section = None

    for line in lines:
        line = line.strip()

        if not line:
            continue

        # -------------------------------------------------
        # STOP
        # -------------------------------------------------
        if re.search(r"(refer[eê]ncias|bibliogr[aá]ficas)", line, re.IGNORECASE):
            break

        # -------------------------------------------------
        # SECTION
        # -------------------------------------------------
        if re.match(r"^\d+\.\s+[A-ZÁ-Ú]", line):
            current_section = line
            result.append(f"\n[SECTION]\n{line}\n")
            continue

        # -------------------------------------------------
        # SUBSECTION
        # -------------------------------------------------
        if re.match(r"^\d+\.\d+", line):
            result.append(f"\n[SUBSECTION]\n{line}\n")
            continue

        # -------------------------------------------------
        # LISTA CLÍNICA
        # -------------------------------------------------
        if re.match(r"^[a-z]\)", line):
            clean = re.sub(r"^[a-z]\)\s*", "", line)
            result.append(f"- {clean}")
            continue

        # -------------------------------------------------
        # TNM
        # -------------------------------------------------
        if re.match(r"^[TNM]\d", line):
            code = line.split()[0]
            desc = line.replace(code, "").strip(" –-:")

            tipo = code[0]

            tipo_map = {
                "T": "Tumor primário",
                "N": "Linfonodos",
                "M": "Metástase"
            }

            result.append(
                f"[TNM]\n"
                f"Tipo: {tipo_map.get(tipo, tipo)}\n"
                f"Código: {code}\n"
                f"Descrição: {desc}\n"
            )
            continue

        # -------------------------------------------------
        # CID
        # -------------------------------------------------
        if re.match(r"^C\d{2}", line):
            parts = line.split(" ", 1)
            code = parts[0]
            desc = parts[1] if len(parts) > 1 else ""

            result.append(
                f"[CID]\n"
                f"Código: {code}\n"
                f"Descrição: {desc}\n"
                f"Sistema: CID-10\n"
            )
            continue

        # -------------------------------------------------
        # ESTÁGIO
        # -------------------------------------------------
        if re.match(r"^Estágio", line, re.IGNORECASE):
            result.append(f"\n[STAGING]\n{line}\n")
            continue

        # -------------------------------------------------
        # TEXTO NORMAL
        # -------------------------------------------------
        result.append(line)

    return "\n".join(result)

def process_pcdt_type_6(text):

    import re

    # -------------------------------------------------
    # 1. BASE
    # -------------------------------------------------
    text = fix_encoding(text)
    text = remove_pcdt_frontmatter(text)

    text = remove_headers_footers(text)
    text = remove_page_numbers(text)

    # -------------------------------------------------
    # 2. NORMALIZAÇÃO
    # -------------------------------------------------
    text = fix_hyphenation(text)
    text = fix_broken_words(text)

    # 🔥 reconstrução de linhas MELHORADA
    text = fix_broken_lines_advanced(text)

    # -------------------------------------------------
    # 3. REMOVER PARTES NÃO RELEVANTES
    # -------------------------------------------------
    text = re.split(
        r"\n\s*#?\s*\d*\.?\s*(REFER[ÊE]NCIAS|BIBLIOGRAFIA)\b.*",
        text,
        flags=re.IGNORECASE
    )[0]

    # -------------------------------------------------
    # 4. PROCESSAMENTO
    # -------------------------------------------------
    lines = text.split("\n")

    result = []
    buffer = []

    current_section = None
    current_subsection = None

    def flush_buffer():
        nonlocal buffer
        if buffer:
            paragraph = " ".join(buffer).strip()
            if len(paragraph) > 40:
                result.append(paragraph)
        buffer = []

    i = 0
    while i < len(lines):

        line = lines[i].strip()

        if not line:
            flush_buffer()
            i += 1
            continue

        # -------------------------------------------------
        # 4.1 DETECTAR SEÇÃO
        # -------------------------------------------------
        if re.match(r"^\d+\.?\s+[A-ZÁ-Ú].+", line):
            flush_buffer()
            current_section = line
            current_subsection = None
            result.append(f"\n# {line}\n")
            i += 1
            continue

        # -------------------------------------------------
        # 4.2 SUBSEÇÃO
        # -------------------------------------------------
        if re.match(r"^\d+\.\d+(\.\d+)?\.?\s+.+", line):
            flush_buffer()
            current_subsection = line
            result.append(f"\n## {line}\n")
            i += 1
            continue

        # -------------------------------------------------
        # 4.3 DETECÇÃO REAL DE TABELA (BLOCO)
        # -------------------------------------------------
        if is_table_block(lines, i):
            i += 5  # pula bloco inteiro
            continue

        # -------------------------------------------------
        # 4.4 LISTAS
        # -------------------------------------------------
        if re.match(r"^[-•]\s+", line):
            flush_buffer()
            result.append(line)
            i += 1
            continue

        # -------------------------------------------------
        # 4.5 CID / TNM
        # -------------------------------------------------
        if re.match(r"^C\d{2}", line) or re.match(r"^[TNM]\d", line):
            result.append(f"- {line}")
            i += 1
            continue

        # -------------------------------------------------
        # 4.6 TEXTO NORMAL
        # -------------------------------------------------
        buffer.append(line)

        i += 1

    flush_buffer()

    # -------------------------------------------------
    # 5. FINAL
    # -------------------------------------------------
    final_text = "\n".join(result)
    final_text = re.sub(r"\n{3,}", "\n\n", final_text)

    return final_text

LISTA_DOCUMENTOS_PCDT = [
    {"process_pcdt_type_1": 
        [
            "pcdt_2018_diretrizes_diagnosticas_e_terapeuticas_do_adenocarcinoma_de_estomago.pdf",
            "pcdt_2016_diretrizes_diagnosticas_e_terapeuticas_do_adenocarcinoma_de_prostata.pdf",
         ]
    },
    {"process_pcdt_type_2": 
        [
            "pcdt_2014_diretrizes_diagnosticas_e_terapeuticas_-_leucemia_mieloide_aguda_do_adulto.pdf",
            "pcdt_2014_diretrizes_diagnosticas_e_terapeuticas_da_leucemia_mieloide_aguda_em_criancas_e_.pdf",      
        
        ]},
    {"process_pcdt_type_3": 
        [
            "pcdt_2014_diretrizes_diagnosticas_e_terapeuticas_do_carcinoma_de_esofago.pdf",
            "pcdt_2014_portaria_no_10512014_diretrizes_diagnosticas_e_terapeuticas_do_linfoma_folicular.pdf",
            "pcdt_2014_portaria_no_9572014_diretrizes_diagnosticas_e_terapeuticas_do_cancer_de_pulmao.pdf",
            "pcdt_2014_protocolo_clinico_e_diretrizes_terapeuticas_do_carcinoma_diferenciado_da_tireoid.pdf",
            "pcdt_2014_protocolo_clinico_e_diretrizes_terapeuticas_do_linfoma_difuso_de_grandes_celulas.pdf",         
            "pcdt_2014_protocolo_clinico_e_diretrizes_terapeuticas_do_tumor_do_estroma_gastrointestinal.pdf",
         ]
     },
    {"process_pcdt_type_4":
        [
            "pcdt_2015_portaria_no_5162015_diretrizes_diagnosticas_e_terapeuticas_para_cancer_de_cabeca.pdf",
        ]
    },
    {"process_pcdt_type_5":
        [
            "pcdt_2020_portaria_conjunta_no_7_de_abril_de_2020_-_diretrizes_diagnosticas_e_terapeuticas.pdf",
            "pcdt_2020_protocolo_clinico_e_diretrizes_terapeuticas_-_linfoma_de_hodgkin_no_adulto.pdf",
            "pcdt_2021_aprovacao_das_diretrizes_diagnosticas_e_terapeuticas_de_mesilato_de_imatinibe_pa.pdf",
            "pcdt_2021_diretrizes_diagnosticas_e_terapeuticas_-_mesilato_de_imatinibe_para_leucemia_lin.pdf",
            "pcdt_2021_protocolo_clinico_e_diretrizes_terapeuticas_-_leucemia_mieloide_cronica_de_crian.pdf",
            "pcdt_2021_protocolo_clinico_e_diretrizes_terapeuticas_-_leucemia_mieloide_cronica_do_adult.pdf",
            "pcdt_2022_diretrizes_diagnosticas_e_terapeuticas_-_carcinoma_hepatocelular_no_adulto.pdf",
            "pcdt_2022_diretrizes_diagnosticas_e_terapeuticas_do_carcinoma_de_celulas_renais.pdf",
            "pcdt_2022_portaria_conjunta_no_192022_-_diretrizes_diagnosticas_e_terapeuticas_do_melanoma.pdf",
            "pcdt_2023_diretrizes_diagnosticas_e_terapeuticas_do_mieloma_multiplo.pdf",
            "pcdt_2025_protocolo_clinico_e_diretrizes_terapeuticas_-_adenocarcinoma_de_colon_e_reto.pdf",
        ]
    },
    {"process_pcdt_type_6":
        [
            "pcdt_2024_protocolo_clinico_e_diretrizes_terapeuticas_do_cancer_de_mama_-_2024.pdf",
        ]
    }
]

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
        document_type = row["document_type"]
        file_standardized = row["file_name_standardized"]

        

        try:
            # 🔹 extrai texto bruto primeiro
            raw_text = extract_pdf_text(file_path)

            # 🔹 roteamento por tipo
            if file_standardized in LISTA_DOCUMENTOS_PCDT[0]["process_pcdt_type_1"]:
                print(f"[{idx+1}/{total}] {file_path.name}")
                clean_text = process_pcdt_type_1(raw_text)
            elif file_standardized in LISTA_DOCUMENTOS_PCDT[1]["process_pcdt_type_2"]:
                print(f"[{idx+1}/{total}] {file_path.name}")
                clean_text = process_pcdt_type_2(raw_text)
            elif file_standardized in LISTA_DOCUMENTOS_PCDT[2]["process_pcdt_type_3"]:
                print(f"[{idx+1}/{total}] {file_path.name}")
                clean_text = process_pcdt_type_3(raw_text)
            elif file_standardized in LISTA_DOCUMENTOS_PCDT[3]["process_pcdt_type_4"]:
                print(f"[{idx+1}/{total}] {file_path.name}")
                clean_text = process_pcdt_type_4(raw_text)
            elif file_standardized in LISTA_DOCUMENTOS_PCDT[4]["process_pcdt_type_5"]:
                print(f"[{idx+1}/{total}] {file_path.name}")
                clean_text = process_pcdt_type_5(raw_text)
            elif file_standardized in LISTA_DOCUMENTOS_PCDT[5]["process_pcdt_type_6"]:
                print(f"[{idx+1}/{total}] {file_path.name}")
                clean_text = process_pcdt_type_6(raw_text)
            else:
                # print(f"Skipping unsupported type: {document_type}")
                continue

            # 🔹 salva resultado
            output_file = OUTPUT_DIR / f"{file_standardized}.txt"

            with open(output_file, "w", encoding="utf-8") as f:
                f.write(clean_text)

        except Exception as e:
            print(f"Error processing {file_path.name}: {e}")


# ---------------------------------------------------------
# RUN
# ---------------------------------------------------------

if __name__ == "__main__":
    process_documents()