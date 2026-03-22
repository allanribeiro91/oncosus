import re
from pathlib import Path

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_DIR = PROJECT_ROOT / "data/processed/v2_inspect"
OUTPUT_DIR = PROJECT_ROOT / "data/processed/v3_final"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------
# REMOVE BROKEN LINES
# ---------------------------------------------------------

def remove_broken_lines(text):

    lines = text.split("\n")
    result = []

    i = 0
    while i < len(lines):
        line = lines[i].rstrip()

        # linha vazia → mantém (separa parágrafos)
        if not line.strip():
            result.append("")
            i += 1
            continue

        # mantém headings intactos
        if line.strip().startswith("#"):
            result.append(line.strip())
            i += 1
            continue

        merged_line = line.strip()

        while i + 1 < len(lines):
            next_line = lines[i + 1].strip()

            # condições de parada
            if (
                not next_line or
                next_line.startswith("#") or
                re.match(r"\d+\.", next_line) or
                next_line.startswith("•") or
                next_line.startswith("- ")
            ):
                break

            # junta linha
            merged_line += " " + next_line
            i += 1

        # limpeza leve de espaços
        merged_line = re.sub(r"\s{2,}", " ", merged_line)

        result.append(merged_line)
        i += 1

    return "\n".join(result)


# ---------------------------------------------------------
# MAIN PROCESS FUNCTION
# ---------------------------------------------------------

def process_final_txt(text):

    refined_text = remove_broken_lines(text)

    return refined_text.strip()


# ---------------------------------------------------------
# BATCH PROCESS
# ---------------------------------------------------------

def process_all_txts():

    input_dir = Path(INPUT_DIR)
    output_dir = Path(OUTPUT_DIR)

    files = list(input_dir.glob("*.txt"))
    total = len(files)

    print(f"Processing {total} TXT files...\n")

    for i, file_path in enumerate(files):

        print(f"[{i+1}/{total}] {file_path.name}")

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()

            refined_text = process_final_txt(text)

            output_path = output_dir / file_path.name

            with open(output_path, "w", encoding="utf-8") as f:
                f.write(refined_text)

        except Exception as e:
            print(f"❌ Error in {file_path.name}: {e}")

    print("\n✅ Done.")


# ---------------------------------------------------------
# RUN
# ---------------------------------------------------------

if __name__ == "__main__":
    process_all_txts()