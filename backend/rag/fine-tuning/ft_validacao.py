import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_PATH = PROJECT_ROOT / "backend" / "data" / "ft_dataset" / "ft_dataset.jsonl"


def validar_dataset(path):

    erros = 0

    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):

            try:
                item = json.loads(line)
                messages = item["messages"]

                roles = [m["role"] for m in messages]

                if roles != ["system", "user", "assistant"]:
                    print(f"[{i}] ❌ roles incorretos: {roles}")
                    erros += 1

                assistant = messages[2]["content"]

                if "### Resposta" not in assistant:
                    print(f"[{i}] ❌ sem estrutura")
                    erros += 1

                if "[DOC_" not in assistant:
                    print(f"[{i}] ❌ sem fontes")
                    erros += 1

            except Exception as e:
                print(f"[{i}] ❌ erro parsing: {e}")
                erros += 1

    print(f"\nTotal de erros: {erros}")


if __name__ == "__main__":
    validar_dataset(DATASET_PATH)