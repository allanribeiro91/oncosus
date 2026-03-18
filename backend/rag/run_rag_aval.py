import csv
from datetime import datetime
from pathlib import Path

from rag_pipeline import RAGPipeline


# ----------------------------------
# CONFIG
# ----------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
VECTOR_DB_PATH = PROJECT_ROOT / "data/vectorstore"
OUTPUT_DIR = PROJECT_ROOT / "data/rag_test"


# ----------------------------------
# DATASET (Q + EXPECTED)
# ----------------------------------

EVAL_DATA = [

    # --- CARCINOMA DE CÉLULAS RENAIS ---
    {
        "question": "Qual é a tríade clássica do carcinoma de células renais e qual sua frequência?",
        "expected_answer": "A tríade clássica inclui dor em flanco, hematúria e massa abdominal palpável, ocorrendo em apenas cerca de 6% a 10% dos casos, geralmente associada a doença avançada."
    },

    {
        "question": "Qual é o exame inicial mais comum na detecção de massas renais?",
        "expected_answer": "A ultrassonografia abdominal é geralmente o primeiro exame utilizado, permitindo diferenciar massas sólidas de císticas."
    },

    # --- OSTEOPOROSE ---
    {
        "question": "Quais são os critérios diagnósticos de osteoporose segundo o T-escore?",
        "expected_answer": "O diagnóstico pode ser feito com T-escore ≤ -2,5, presença de fratura por fragilidade ou T-escore entre -1 e -2,49 com alto risco de fratura pelo FRAX."
    },

    {
        "question": "Quais fatores influenciam o pico de massa óssea durante a vida?",
        "expected_answer": "Fatores genéticos, ingestão de cálcio, níveis de vitamina D, atividade física, hormônios e presença de doenças influenciam o pico de massa óssea."
    },

    # --- MIELOMA MÚLTIPLO ---
    {
        "question": "Quais são os principais critérios clínicos utilizados no diagnóstico do mieloma múltiplo?",
        "expected_answer": "Incluem evidência de proliferação de plasmócitos clonais e presença de danos orgânicos relacionados, como hipercalcemia, insuficiência renal, anemia e lesões ósseas (CRAB)."
    },

    {
        "question": "Qual é a importância da avaliação de cadeia leve livre no mieloma múltiplo?",
        "expected_answer": "A dosagem de cadeias leves livres auxilia no diagnóstico, prognóstico e monitoramento da doença, especialmente em casos não secretantes ou oligosecretantes."
    },

    # --- LINFOMA FOLICULAR ---
    {
        "question": "Qual é o comportamento clínico típico do linfoma folicular?",
        "expected_answer": "O linfoma folicular geralmente apresenta curso indolente, com progressão lenta e períodos prolongados de estabilidade."
    },

    {
        "question": "Quando pode ser indicada a estratégia de 'watch and wait' no linfoma folicular?",
        "expected_answer": "Pode ser indicada em pacientes assintomáticos, com baixa carga tumoral e sem critérios de tratamento imediato."
    },

    # --- ASSISTÊNCIA FARMACÊUTICA EM ONCOLOGIA ---
    {
        "question": "Qual é o papel da assistência farmacêutica em oncologia?",
        "expected_answer": "A assistência farmacêutica garante o uso seguro e eficaz dos medicamentos, incluindo seleção, armazenamento, preparo, dispensação e acompanhamento do paciente."
    },

    {
        "question": "Por que é importante monitorar eventos adversos em tratamentos oncológicos?",
        "expected_answer": "Porque os medicamentos oncológicos possuem alta toxicidade, sendo essencial monitorar eventos adversos para garantir segurança e adesão ao tratamento."
    },

    # --- MASTOLOGIA / PÓS-CIRURGIA ---
    {
        "question": "Quais cuidados são recomendados após cirurgia de mama com esvaziamento axilar?",
        "expected_answer": "Incluem evitar esforços com o braço afetado, manter higiene adequada, observar sinais de infecção e realizar exercícios orientados para prevenir linfedema."
    },

    {
        "question": "Qual é o principal objetivo da fisioterapia em pacientes com cirurgia de mama?",
        "expected_answer": "O objetivo é recuperar a mobilidade do membro superior, prevenir linfedema e melhorar a funcionalidade e qualidade de vida."
    },

    {
        "question": "O que é linfedema e por que ele pode ocorrer após cirurgia de mama?",
        "expected_answer": "Linfedema é o acúmulo de líquido linfático nos tecidos, podendo ocorrer devido à remoção de linfonodos e alteração da drenagem linfática."
    },

    # --- TRAQUEOSTOMIA / TRANSPLANTE ---
    {
        "question": "Quais são os principais cuidados com a traqueostomia após transplante de medula óssea?",
        "expected_answer": "Incluem higiene rigorosa da cânula, aspiração de secreções quando necessário e prevenção de infecções devido à imunossupressão."
    },

    # --- HOSPITAL DO CÂNCER IV ---
    {
        "question": "Qual é o papel do Hospital do Câncer IV no contexto do INCA?",
        "expected_answer": "O Hospital do Câncer IV é voltado principalmente para cuidados paliativos, oferecendo assistência integral a pacientes com câncer avançado."
    },

    # --- INTEGRAÇÃO / SISTEMA DE SAÚDE ---
    {
        "question": "Por que a Atenção Primária é importante no manejo do câncer?",
        "expected_answer": "Porque permite identificação precoce de sinais e sintomas, encaminhamento adequado e melhor prognóstico dos pacientes."
    },

    {
        "question": "Qual é a importância de informar o paciente sobre riscos e efeitos adversos dos tratamentos?",
        "expected_answer": "É obrigatório informar o paciente ou responsável sobre riscos e efeitos adversos para garantir consentimento informado e segurança no tratamento."
    },

    # --- QUESTÕES TRANSVERSAIS (BOAS PARA RAG) ---
    {
        "question": "Quais são exemplos de fatores prognósticos negativos em doenças oncológicas?",
        "expected_answer": "Incluem estado funcional reduzido, alterações laboratoriais como anemia, hipercalcemia e presença de doença metastática."
    },

    {
        "question": "Qual a importância dos exames de imagem no diagnóstico de doenças oncológicas?",
        "expected_answer": "Os exames de imagem são fundamentais para detecção, caracterização de lesões, estadiamento e acompanhamento da resposta ao tratamento."
    },

    {
        "question": "Em quais situações o diagnóstico pode ser feito de forma incidental em oncologia?",
        "expected_answer": "O diagnóstico pode ocorrer incidentalmente durante exames de imagem realizados por outros motivos, como ocorre frequentemente no câncer renal."
    }

]


# ----------------------------------
# MAIN
# ----------------------------------

def main():

    print("\n🚀 Iniciando avaliação do RAG...\n")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = OUTPUT_DIR / f"rag_{timestamp}.csv"

    rag = RAGPipeline(
        persist_directory=str(VECTOR_DB_PATH),
        llm_model="mistral"
    )

    results = []

    total = len(EVAL_DATA)

    for i, item in enumerate(EVAL_DATA, 1):

        question = item["question"]
        expected = item["expected_answer"]

        print(f"[{i}/{total}] {question}")

        try:
            result = rag.run(question)

            generated = result.get("answer", "")
            sources = result.get("sources", [])

            results.append({
                "question": question,
                "expected_answer": expected,
                "generated_answer": generated,
                "sources": " | ".join(sources)
            })

        except Exception as e:
            results.append({
                "question": question,
                "expected_answer": expected,
                "generated_answer": f"ERROR: {str(e)}",
                "sources": ""
            })

    # salvar CSV
    with open(output_file, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "question",
                "expected_answer",
                "generated_answer",
                "sources"
            ]
        )
        writer.writeheader()
        writer.writerows(results)

    print(f"\n✅ Finalizado!")
    print(f"📁 Arquivo: {output_file}\n")


if __name__ == "__main__":
    main()