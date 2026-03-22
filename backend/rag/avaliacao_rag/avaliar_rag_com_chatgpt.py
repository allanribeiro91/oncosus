import csv
import json
from dotenv import load_dotenv
import os
import sys
from pathlib import Path
from datetime import datetime

# adiciona a pasta /rag no path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from rag_pipeline import RAGPipeline
from perguntas_respostas import EVAL_DATA

from openai import OpenAI

load_dotenv()

# ----------------------------------
# CONFIG
# ----------------------------------

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_DIR = PROJECT_ROOT / "avaliacao_rag/data/rag_test"
OUTPUT_DIR = PROJECT_ROOT / "avaliacao_rag/data/rag_test"
print(f"🔍 Input directory: {INPUT_DIR}")
print(f"📂 Output directory: {OUTPUT_DIR}")

# ----------------------------------
# PROMPT
# ----------------------------------

def build_prompt(question, expected, generated):

    return f"""
Você é um avaliador técnico de um sistema de IA médica baseado em documentos oficiais do SUS.

Sua tarefa é avaliar a qualidade de uma resposta gerada por um sistema RAG.

Critérios:

1. Correção factual (0–4)
2. Aderência ao esperado (0–3)
3. Ausência de alucinação (0–2)
4. Clareza (0–1)

Regras:

- NÃO use conhecimento externo
- Avalie apenas comparando com a resposta esperada
- Penalize qualquer invenção
- Penalize extrapolação

---

Saída obrigatória em JSON:

{{
  "score": número de 0 a 10,
  "criteria": {{
    "factual_correctness": 0-4,
    "coverage": 0-3,
    "no_hallucination": 0-2,
    "clarity": 0-1
  }},
  "justification": "explicação objetiva"
}}

---

Pergunta:
{question}

Resposta esperada:
{expected}

Resposta gerada:
{generated}
"""


# ----------------------------------
# CALL OPENAI
# ----------------------------------

def evaluate_with_openai(prompt):

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    content = response.choices[0].message.content.strip()

    # limpar markdown
    if "```" in content:
        content = content.split("```")[1]

    return json.loads(content)


# ----------------------------------
# MAIN
# ----------------------------------

def avaliar_rag_com_chatgpt():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # pegar último CSV gerado
    files = sorted(INPUT_DIR.glob("rag_*.csv"), reverse=True)

    if not files:
        print("❌ Nenhum arquivo encontrado")
        return

    input_csv = files[0]
    print(f"\n📂 Avaliando: {input_csv}\n")

    results = []

    with open(input_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for i, row in enumerate(reader, 1):

            question = row["question"]
            expected = row["expected_answer"]
            generated = row["generated_answer"]

            print(f"[{i}] Avaliando...")

            prompt = build_prompt(question, expected, generated)

            try:
                data = evaluate_with_openai(prompt)

                results.append({
                    "question": question,
                    "expected_answer": expected,
                    "generated_answer": generated,
                    "score": data.get("score"),
                    "factual": data.get("criteria", {}).get("factual_correctness"),
                    "coverage": data.get("criteria", {}).get("coverage"),
                    "no_hallucination": data.get("criteria", {}).get("no_hallucination"),
                    "clarity": data.get("criteria", {}).get("clarity"),
                    "justification": data.get("justification")
                })

            except Exception as e:
                print(f"❌ Erro: {e}")

                results.append({
                    "question": question,
                    "expected_answer": expected,
                    "generated_answer": generated,
                    "score": "ERROR",
                    "factual": "",
                    "coverage": "",
                    "no_hallucination": "",
                    "clarity": "",
                    "justification": str(e)
                })

    # salvar CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = OUTPUT_DIR / f"evaluation_{timestamp}.csv"

    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "question",
                "expected_answer",
                "generated_answer",
                "score",
                "factual",
                "coverage",
                "no_hallucination",
                "clarity",
                "justification"
            ]
        )
        writer.writeheader()
        writer.writerows(results)

    print(f"\n✅ Avaliação concluída!")
    print(f"📁 Resultado: {output_file}\n")


# ----------------------------------
# RUN
# ----------------------------------

if __name__ == "__main__":
    avaliar_rag_com_chatgpt()