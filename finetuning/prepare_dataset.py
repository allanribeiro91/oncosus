"""
prepare_dataset.py
Anonimiza, cura e formata os dados para fine-tuning.
Integra dados sintéticos + PDFs das DDTs (se disponíveis).
Saída: dataset_treino.json + dataset_validacao.json
"""
import re, json, random, hashlib
from pathlib import Path
from datetime import datetime
from collections import Counter
import yaml

with open("finetuning/training_config.yaml", encoding="utf-8") as f:
    CFG = yaml.safe_load(f)

SYSTEM_PROMPT = CFG["system_prompt"].strip()
VAL_RATIO     = CFG["data"]["val_ratio"]

# Padrões de anonimização (PII)
PII = [
    (re.compile(r"\d{3}\.?\d{3}\.?\d{3}-?\d{2}"),              "000.000.000-00"),
    (re.compile(r"\bCRM[-/]?\s*[A-Z]{0,2}\s*\d{4,6}\b", re.I), "CRM 00000"),
    (re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"), "contato@clinica.com"),
    (re.compile(r"\(?\d{2}\)?\s?\d{4,5}-?\d{4}"),              "(00) 0000-0000"),
    (re.compile(
        r"(Paciente|Dr\.?|Dra\.?|médico)[\s:]+([A-ZÁÉÍÓÚ][a-záéíóú]+"
        r"(?:\s+[A-ZÁÉÍÓÚ][a-záéíóú]+){1,3})", re.I
    ), r"\1 Silva"), # Substitui nomes reais por um genérico "Silva" em vez da tag [NOME]
]

def anonimizar(texto):
    for padrao, sub in PII:
        texto = padrao.sub(sub, texto)
    return texto

def format_prompt(pergunta, resposta=""):
    return (
        f"### Sistema:\n{SYSTEM_PROMPT}\n\n"
        f"### Instrução:\n{pergunta}\n\n"
        f"### Resposta:\n{resposta}"
    )

def curar(pergunta, resposta, fonte="interno"):
    pergunta = anonimizar(pergunta.strip())
    resposta = anonimizar(resposta.strip())
    if len(pergunta) < 15 or len(resposta) < 80:
        return None
    if len(resposta) > 2500:
        resposta = resposta[:2500] + "…"
    return {"text": format_prompt(pergunta, resposta),
            "fonte": fonte, "curated_at": datetime.now().isoformat()}

def processar_sinteticos(raw_path):
    with open(raw_path, encoding="utf-8") as f:
        raw = json.load(f)
    exemplos = []
    for p in raw.get("protocols", []):
        ex = curar(f"Descreva o protocolo oncológico: {p['title']}",
                   p["content"], "protocolo_oncologico")
        if ex: exemplos.append(ex)
    for qa in raw.get("faq", []):
        ex = curar(qa["pergunta"], qa["resposta"], qa.get("fonte", "faq_oncologia"))
        if ex: exemplos.append(ex)
    for rec in raw.get("patient_records", []):
        nota = anonimizar(rec.get("clinical_notes", ""))
        resp = (f"Diagnóstico: {rec['diagnosis']} (estadio {rec['stage']}). "
                f"ECOG {rec['ecog']}. Protocolo: {rec['current_protocol']}. {nota}")
        ex = curar("Resuma o quadro oncológico e a conduta terapêutica.", resp, "prontuario_sintetico")
        if ex: exemplos.append(ex)
    print(f"   ✓ Dados sintéticos: {len(exemplos)} exemplos")
    return exemplos

def processar_pdfs():
    try:
        from pdfminer.high_level import extract_text
        from pdfminer.pdfpage import PDFPage
    except ImportError:
        print("   ⚠️  pdfminer não instalado. Pulando PDFs.")
        return []

    pdf_dir = Path("data/raw_documents/pcdt")
    pdfs    = list(pdf_dir.glob("*.pdf"))
    if not pdfs:
        print(f"   ℹ️  Nenhum PDF em {pdf_dir}. Coloque as DDTs do MS lá.")
        return []

    TIPO_MAP = {
        "diagnóstico":   "Quais são os critérios diagnósticos para {cancer}?",
        "tratamento":    "Qual o tratamento recomendado pelo MS para {cancer}?",
        "estadiamento":  "Como é realizado o estadiamento do {cancer}?",
        "quimioterapia": "Qual o protocolo de quimioterapia para {cancer}?",
        "imunoterapia":  "Quais imunoterápicos disponíveis no SUS para {cancer}?",
        "rastreamento":  "Qual o rastreamento recomendado para {cancer}?",
        "monitoramento": "Como monitorar o paciente com {cancer}?",
    }
    SECTION_RE = re.compile(
        r"^(\d{1,2}(?:\.\d{1,2}){0,3})\s+([A-ZÁÉÍÓÚÂÊÔÃÕÇ][^\n]{5,80})$",
        re.MULTILINE
    )

    def gerar_pergunta(titulo, cancer):
        lower = titulo.lower()
        for kw, tmpl in TIPO_MAP.items():
            if kw in lower:
                return tmpl.format(cancer=cancer)
        return f"Segundo o MS, explique '{titulo}' no contexto do {cancer}."

    exemplos = []
    for pdf_path in pdfs:
        nome = pdf_path.stem.replace("_", " ").replace("-", " ")
        try:
            with open(pdf_path, "rb") as f:
                total = len(list(PDFPage.get_pages(f)))
            texto = extract_text(str(pdf_path), page_numbers=list(range(3, total)))
            texto = re.sub(r"\n{3,}", "\n\n", texto).strip()
            matches = list(SECTION_RE.finditer(texto))
            for i, m in enumerate(matches):
                titulo   = m.group(2).strip()
                conteudo = texto[m.end(): matches[i+1].start()
                                 if i+1 < len(matches) else len(texto)].strip()
                if len(conteudo) < 80:
                    continue
                for chunk in [conteudo[j:j+1800] for j in range(0, len(conteudo), 1600)]:
                    perg = gerar_pergunta(titulo, nome)
                    resp = f"Segundo as DDT do Ministério da Saúde:\n\n{anonimizar(chunk)}"
                    ex   = curar(perg, resp, fonte=f"ddt_{pdf_path.stem[:20]}")
                    if ex: exemplos.append(ex)
        except Exception as e:
            print(f"   ⚠️  Erro ao processar {pdf_path.name}: {e}")

    print(f"   ✓ PDFs das DDTs: {len(exemplos)} exemplos")
    return exemplos

if __name__ == "__main__":
    print("=" * 55)
    print("📊 PREPARE DATASET — OncoSUS Fine-Tuning")
    print("=" * 55)

    raw_path = "data/datasets/synthetic_qa_dataset/raw_synthetic.json"
    if not Path(raw_path).exists():
        print("❌ Execute generate_synthetic_qa.py primeiro.")
        exit(1)

    print("\n📂 Processando fontes de dados...")
    todos = processar_sinteticos(raw_path) + processar_pdfs()

    print(f"\n   Total de exemplos: {len(todos)}")
    print("   Por fonte:")
    for fonte, n in Counter(e["fonte"] for e in todos).most_common():
        print(f"     {fonte:<40} {n:>4}")

    random.seed(42)
    random.shuffle(todos)
    split     = int(len(todos) * (1 - VAL_RATIO))
    treino    = todos[:split]
    validacao = todos[split:]

    out_dir = Path("data/datasets/synthetic_qa_dataset")
    with open(out_dir / "dataset_treino.json", "w", encoding="utf-8") as f:
        json.dump(treino, f, ensure_ascii=False, indent=2)
    with open(out_dir / "dataset_validacao.json", "w", encoding="utf-8") as f:
        json.dump(validacao, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Dataset salvo!")
    print(f"   Treino    : {len(treino)} exemplos")
    print(f"   Validação : {len(validacao)} exemplos")