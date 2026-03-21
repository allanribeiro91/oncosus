"""
generate_synthetic_qa.py
Gera dataset sintético oncológico baseado nos protocolos do SUS/INCA.
Saída: data/datasets/synthetic_qa_dataset/raw_synthetic.json
"""
import json, random
from pathlib import Path
from datetime import datetime

PROTOCOLOS = [
    {"title": "Protocolo AC-T (Câncer de Mama Adjuvante)",
     "content": ("Fase AC: Doxorrubicina 60 mg/m² IV D1 + Ciclofosfamida 600 mg/m² IV D1.\n"
                 "Ciclo a cada 21 dias × 4 ciclos.\n"
                 "Fase T: Paclitaxel 80 mg/m² IV semanal × 12 semanas.\n"
                 "G-CSF profilático obrigatório na fase AC (risco febril ≥ 20%).\n"
                 "Monitorar FEVE basal e a cada 4 ciclos. Suspender se FEVE < 50%.")},
    {"title": "Protocolo FOLFOX6m (Câncer Colorretal Metastático)",
     "content": ("Oxaliplatina 85 mg/m² IV em 2h D1.\n"
                 "Leucovorina 400 mg/m² IV em 2h D1 (concomitante).\n"
                 "5-Fluorouracil 400 mg/m² IV bolus D1 + 5-FU 2.400 mg/m² IC 46h.\n"
                 "Ciclo a cada 14 dias.\n"
                 "Suspender oxaliplatina se neuropatia grau ≥ 3.")},
    {"title": "Protocolo R-CHOP (Linfoma Difuso de Grandes Células B)",
     "content": ("Rituximabe 375 mg/m² IV D1.\n"
                 "Ciclofosfamida 750 mg/m² IV D1.\n"
                 "Doxorrubicina 50 mg/m² IV D1.\n"
                 "Vincristina 1,4 mg/m² IV D1 (máx 2 mg).\n"
                 "Prednisona 100 mg VO D1-D5.\n"
                 "Ciclo a cada 21 dias × 6-8 ciclos.")},
    {"title": "Protocolo BEP (Tumor de Células Germinativas Testicular)",
     "content": ("Bleomicina 30 UI IV D1, D8, D15.\n"
                 "Etoposídeo 100 mg/m² IV D1-D5.\n"
                 "Cisplatina 20 mg/m² IV D1-D5 (hiperhidratação obrigatória).\n"
                 "Ciclo a cada 21 dias × 3-4 conforme prognose.\n"
                 "Monitorar DLCO — suspender bleomicina se < 40%.")},
    {"title": "Terapia-alvo: Osimertinibe — NSCLC EGFR mutado",
     "content": ("Indicação: adenocarcinoma pulmonar EGFR exon 19 del ou L858R.\n"
                 "Dose: 80 mg VO 1x/dia.\n"
                 "Toxicidades: diarreia (40%), rash acneiforme, pneumonite (2-3%).\n"
                 "Monitorar ECG (QTc), FEVE, TC tórax a cada 3 meses.")},
    {"title": "Imunoterapia: Pembrolizumabe — Melanoma Metastático",
     "content": ("Dose: 200 mg IV a cada 21 dias.\n"
                 "irAE — colite grau ≥ 2: suspender + Prednisona 1-2 mg/kg/dia.\n"
                 "irAE — pneumonite: suspender + corticoide IV.\n"
                 "TSH e ALT/AST antes de cada ciclo.\n"
                 "Não reintroduzir após irAE grau 3-4.")},
]

FAQ = [
    {"pergunta": "Qual a conduta para neutropenia febril pós-quimioterapia?",
     "resposta": ("Neutropenia febril é emergência oncológica. Conduta:\n"
                  "1. Confirmar neutrófilos < 500/mm³.\n"
                  "2. Hemoculturas (2 pares) ANTES do antibiótico.\n"
                  "3. Antibiótico empírico: Cefepima 2g IV 8/8h ou Pip-Tazo 4,5g IV 6/6h.\n"
                  "4. Escore MASCC ≥ 21 = baixo risco.\n"
                  "5. Reavaliar em 48-72h."), "fonte": "protocolo_institucional"},
    {"pergunta": "Quais são os critérios RECIST 1.1?",
     "resposta": ("RC: desaparecimento de todas as lesões-alvo.\n"
                  "RP: redução ≥ 30% na soma dos diâmetros.\n"
                  "DE: nem RP nem progressão.\n"
                  "PD: aumento ≥ 20% + ≥ 5mm ou nova lesão.\n"
                  "Lesão mínima mensurável: ≥ 10mm em TC."), "fonte": "recist_1_1"},
    {"pergunta": "Quando indicar G-CSF profilático?",
     "resposta": ("Profilaxia primária se:\n"
                  "• Risco neutropenia febril ≥ 20%;\n"
                  "• Paciente ≥ 65 anos em protocolo moderado;\n"
                  "• Comorbidades: IRC, hepatopatia, ECOG ≥ 2.\n"
                  "Dose: Pegfilgrastima 6mg SC D2."), "fonte": "asco_gcfs_guideline"},
    {"pergunta": "Como manejar cardiotoxicidade por antraciclinas?",
     "resposta": ("Doxorrubicina: suspender se FEVE < 50% ou dose > 450 mg/m².\n"
                  "Ecocardiograma basal e a cada 200-300 mg/m².\n"
                  "Dexrazoxano se dose cumulativa prevista > 300 mg/m².\n"
                  "Tratamento de ICC: IECA + betabloqueador."), "fonte": "cardio_oncologia"},
    {"pergunta": "O assistente pode prescrever medicamentos diretamente?",
     "resposta": ("Não. Este assistente é uma ferramenta de suporte à decisão clínica.\n"
                  "Toda prescrição deve ser realizada e validada por médico habilitado,\n"
                  "considerando o contexto individual do paciente."), "fonte": "politica_seguranca"},
    {"pergunta": "Qual a diferença entre tratamento neoadjuvante e adjuvante?",
     "resposta": ("Neoadjuvante: tratamento ANTES da cirurgia.\n"
                  "  Objetivos: reduzir tumor, testar sensibilidade, erradicar micrometástases.\n"
                  "  Indicação: HER2+, triplo negativo > 2cm, localmente avançado.\n"
                  "Adjuvante: tratamento APÓS cirurgia ressecada.\n"
                  "  Objetivo: reduzir risco de recorrência."), "fonte": "protocolo_mama"},
    {"pergunta": "Como interpretar o PD-L1 na imunoterapia?",
     "resposta": ("CPS ≥ 1: expressão positiva baixa.\n"
                  "CPS ≥ 10: melhor resposta ao pembrolizumabe.\n"
                  "CPS ≥ 20: benefício enriquecido em gástrico/esofágico.\n"
                  "PD-L1 negativo NÃO exclui resposta (melanoma, MSI-H)."), "fonte": "guideline_imuno"},
    {"pergunta": "Quando solicitar teste molecular para câncer de pulmão?",
     "resposta": ("Obrigatório para adenocarcinoma pulmonar:\n"
                  "• EGFR → osimertinibe;\n"
                  "• ALK → alectinibe;\n"
                  "• ROS1 → crizotinibe;\n"
                  "• KRAS G12C → sotorasibe;\n"
                  "• MET exon 14 → capmatinibe.\n"
                  "PD-L1 (22C3) para todos os NSCLC. Preferir NGS amplo."), "fonte": "sboc_pulmao"},
    {"pergunta": "Qual a conduta para síndrome de lise tumoral?",
     "resposta": ("Alto risco: Rasburicase 0,2 mg/kg/dia × 5 + hiper-hidratação 3 L/m²/dia.\n"
                  "Risco intermediário: Alopurinol 300 mg VO 2x/dia + hidratação.\n"
                  "Monitorar eletrólitos e creatinina 6/6h nas primeiras 24h."), "fonte": "hematologia"},
    {"pergunta": "Como calcular dose de quimio em paciente obeso?",
     "resposta": ("Usar peso real (não ideal) — diretriz ASCO 2012.\n"
                  "Capping não recomendado: subdosagem aumenta falha terapêutica.\n"
                  "Exceções: Bleomicina (UI fixas), Vincristina (máx 2 mg).\n"
                  "BSA Mosteller: sqrt[(altura cm × peso kg) / 3600]."), "fonte": "asco_2012"},
]

CANCERS = [
    "Câncer de mama HER2+", "Adenocarcinoma de pulmão EGFR mutado",
    "Câncer colorretal metastático", "Linfoma difuso de grandes células B",
    "Melanoma metastático BRAF V600E", "Carcinoma hepatocelular",
    "Câncer de ovário BRCA1/2 mutado", "Mieloma múltiplo",
    "Leucemia mielóide aguda", "Linfoma de Hodgkin clássico",
]

def gerar_prontuario(pid):
    cancer   = random.choice(CANCERS)
    protocol = random.choice(PROTOCOLOS)
    age      = random.randint(28, 82)
    ecog     = random.choice([0, 1, 1, 2])
    return {
        "patient_id": f"ONC_{pid:04d}", "age": age,
        "sex": random.choice(["M", "F"]), "ecog": ecog,
        "diagnosis": cancer, "stage": random.choice(["II", "III", "IV"]),
        "current_protocol": protocol["title"],
        "clinical_notes": (
            f"Paciente oncológico, diagnóstico de {cancer}, ECOG {ecog}. "
            f"Em tratamento com {protocol['title']}, ciclo {random.randint(1,8)}. "
            f"Hemograma sem citopenias significativas."
        ),
    }

if __name__ == "__main__":
    out_dir = Path("data/datasets/synthetic_qa_dataset")
    out_dir.mkdir(parents=True, exist_ok=True)
    random.seed(42)
    dataset = {
        "metadata": {"project": "OncoSUS", "specialty": "oncology",
                     "generated_at": datetime.now().isoformat()},
        "protocols": PROTOCOLOS,
        "faq": FAQ,
        "patient_records": [gerar_prontuario(i) for i in range(1, 201)],
    }
    with open(out_dir / "raw_synthetic.json", "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    print("✅ Dataset sintético gerado!")
    print(f"   • {len(PROTOCOLOS)} protocolos oncológicos")
    print(f"   • {len(FAQ)} pares de FAQ especializado")
    print(f"   • {len(dataset['patient_records'])} prontuários sintéticos")