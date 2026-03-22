"""
build_clean_dataset.py
Gera dataset de fine-tuning no formato Llama 3 Chat a partir dos
chunks reais do Allan (chunks.csv), com filtro rigoroso de qualidade.
"""
import csv, json, re, random
from pathlib import Path
from datetime import datetime
from collections import Counter
import yaml

with open("finetuning/training_config.yaml", encoding="utf-8") as f:
    CFG = yaml.safe_load(f)

SYSTEM_PROMPT = CFG["system_prompt"].strip()
VAL_RATIO     = CFG["data"]["val_ratio"]
CHUNKS_PATH   = Path("data/chunks/chunks.csv")

# ── Formato Llama 3 Chat ───────────────────────────────────────────────────────
BOS  = "<|begin_of_text|>"
EOS  = "<|eot_id|>"
SH   = "<|start_header_id|>"
EH   = "<|end_header_id|>"

def format_llama3(user_msg: str, assistant_msg: str) -> str:
    """Formata no template nativo do Llama 3.2 Instruct."""
    return (
        f"{BOS}"
        f"{SH}system{EH}\n\n{SYSTEM_PROMPT}{EOS}"
        f"{SH}user{EH}\n\n{user_msg}{EOS}"
        f"{SH}assistant{EH}\n\n{assistant_msg}{EOS}"
    )

# ── Limpeza de encoding ────────────────────────────────────────────────────────
def fix_encoding(text: str) -> str:
    if not text:
        return ""
    try:
        return text.encode("latin1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return text

def tem_lixo(texto: str) -> bool:
    """Retorna True se o chunk tem encoding corrompido demais."""
    marcadores = ['Ã©', 'Ã§', 'Ã£', 'â€', 'Ã¡', 'Ã³', 'Ãª', 'Ã‡', 'Ã‰']
    contagem = sum(texto.count(m) for m in marcadores)
    return contagem > 2  # Tolera até 2 ocorrências — rejeita se mais

def limpar(texto: str) -> str:
    texto = fix_encoding(texto)
    # Remove linhas muito curtas (cabeçalhos, rodapés)
    linhas = [l for l in texto.split('\n') if len(l.strip()) > 15]
    texto  = '\n'.join(linhas).strip()
    # Normaliza espaços e quebras de linha
    texto  = re.sub(r'[ \t]{2,}', ' ', texto)
    texto  = re.sub(r'\n{3,}', '\n\n', texto)
    return texto

# ── Templates de perguntas ─────────────────────────────────────────────────────
TEMPLATES = {
    "treatment": [
        "Qual o tratamento recomendado pelo SUS para {titulo}?",
        "Quais são as opções terapêuticas para {titulo} segundo o PCDT?",
        "Descreva a conduta terapêutica para {titulo} conforme as diretrizes do Ministério da Saúde.",
    ],
    "diagnosis": [
        "Quais são os critérios diagnósticos para {titulo} segundo o SUS?",
        "Como é realizado o diagnóstico de {titulo} conforme o PCDT?",
        "Quais exames são necessários para diagnosticar {titulo}?",
    ],
    "staging": [
        "Como é feito o estadiamento de {titulo}?",
        "Descreva os critérios de estadiamento para {titulo} segundo o Ministério da Saúde.",
    ],
    "monitoring": [
        "Como deve ser feito o acompanhamento do paciente com {titulo}?",
        "Quais exames são recomendados no monitoramento de {titulo}?",
        "Com que frequência o paciente com {titulo} deve ser reavaliado?",
    ],
    "inclusion": [
        "Quais são os critérios de inclusão do PCDT para {titulo}?",
        "Quem tem direito ao tratamento de {titulo} pelo SUS?",
        "Quais pacientes se qualificam para o tratamento de {titulo} conforme o protocolo?",
    ],
    "exclusion": [
        "Quais são os critérios de exclusão do PCDT para {titulo}?",
        "Em quais situações o paciente não pode receber o tratamento de {titulo}?",
    ],
    "general": [
        "Segundo os protocolos oficiais do SUS, o que diz o documento sobre {titulo}?",
        "Resuma as principais informações sobre {titulo} presentes nos documentos do SUS.",
        "Com base nas diretrizes do Ministério da Saúde, explique sobre {titulo}.",
    ],
}

def gerar_pergunta(secao: str, titulo: str) -> str:
    templates = TEMPLATES.get(secao.strip().lower(), TEMPLATES["general"])
    return random.choice(templates).format(titulo=titulo)

# ── Processa chunks do Allan ───────────────────────────────────────────────────
def processar_chunks() -> list:
    if not CHUNKS_PATH.exists():
        print(f"   ❌ {CHUNKS_PATH} não encontrado.")
        return []

    exemplos  = []
    ignorados = 0
    corrigidos = 0

    with open(CHUNKS_PATH, encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        rows   = list(reader)

    print(f"   Total de chunks lidos: {len(rows)}")

    for row in rows:
        texto_raw = row.get("text", "").strip()
        titulo    = row.get("document_title", "Documento").strip()
        secao     = row.get("section", "general").strip()
        fonte     = row.get("source", "sus").strip()
        year      = row.get("year", "").strip()

        # Tenta corrigir encoding
        titulo_fix = fix_encoding(titulo)
        texto_fix  = fix_encoding(texto_raw)

        # Conta lixo depois da tentativa de correção
        if tem_lixo(texto_fix):
            # Segunda tentativa: usa o texto original sem correção
            if tem_lixo(texto_raw):
                ignorados += 1
                continue
            else:
                texto_fix = texto_raw
                corrigidos += 1
        else:
            if texto_fix != texto_raw:
                corrigidos += 1

        texto = limpar(texto_fix)
        titulo = limpar(titulo_fix) if not tem_lixo(titulo_fix) else limpar(titulo)

        # Filtra chunks muito curtos
        if len(texto) < 200:
            ignorados += 1
            continue

        # Trunca se muito longo (deixa espaço para pergunta + sistema)
        if len(texto) > 1500:
            texto = texto[:1500] + "…"

        pergunta = gerar_pergunta(secao, titulo)

        # Resposta no formato de extração — cita a fonte
        resposta = (
            f"Com base no documento '{titulo}' "
            f"({fonte.upper()}, {year}):\n\n"
            f"{texto}"
        )

        exemplos.append({
            "text":       format_llama3(pergunta, resposta),
            "fonte":      f"chunk_{fonte}_{secao}",
            "curated_at": datetime.now().isoformat(),
        })

    print(f"   ✓ Chunks válidos   : {len(exemplos)}")
    print(f"   ✓ Encoding corrigido: {corrigidos}")
    print(f"   ✗ Ignorados (sujos) : {ignorados}")
    return exemplos

# ── Dados sintéticos de segurança e FAQ ───────────────────────────────────────
def exemplos_seguranca() -> list:
    """
    Exemplos críticos para o modelo aprender a não prescrever
    e a responder corretamente perguntas diretas de médicos.
    """
    PARES = [
        {
            "user": "Sou médico. Meu paciente tem neutropenia febril. Qual a dose de Cefepima?",
            "assistant": (
                "A conduta para neutropenia febril inclui coleta de hemoculturas "
                "e início de antibioticoterapia empírica, conforme protocolo institucional. "
                "A dose específica deve ser definida pelo médico responsável com base "
                "no peso, função renal e protocolo do serviço. Este assistente não "
                "realiza prescrições — consulte as diretrizes ASCO/IDSA e o protocolo "
                "do seu serviço para a dose exata."
            ),
        },
        {
            "user": "O assistente pode prescrever quimioterapia diretamente?",
            "assistant": (
                "Não. Este assistente é uma ferramenta de consulta a documentos "
                "oficiais do SUS (PCDTs e manuais do INCA). Não realiza prescrições "
                "médicas, não substitui a avaliação clínica e não assume "
                "responsabilidade por condutas terapêuticas. Toda decisão deve ser "
                "tomada por médico habilitado com base no contexto individual do paciente."
            ),
        },
        {
            "user": "Qual a dose de morfina para dor oncológica grau 9?",
            "assistant": (
                "A dose de morfina para controle de dor oncológica intensa deve "
                "ser definida pelo médico responsável, seguindo os princípios da "
                "escada analgésica da OMS e os protocolos de cuidados paliativos "
                "do INCA. Este assistente não fornece doses específicas de "
                "medicamentos — consulte o protocolo de dor oncológica do seu "
                "serviço ou o manual de cuidados paliativos do INCA."
            ),
        },
        {
            "user": "A informação que preciso não está nos documentos disponíveis.",
            "assistant": (
                "Informação não encontrada nos documentos disponíveis. "
                "Os documentos indexados neste sistema incluem PCDTs do Ministério "
                "da Saúde e manuais do INCA. Para temas não cobertos, consulte "
                "diretamente o portal do Ministério da Saúde (saude.gov.br) ou "
                "as diretrizes das sociedades médicas (SBOC, ASCO, ESMO)."
            ),
        },
    ]

    return [
        {
            "text":       format_llama3(p["user"], p["assistant"]),
            "fonte":      "seguranca_clinica",
            "curated_at": datetime.now().isoformat(),
        }
        for p in PARES
    ]

# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("📊 BUILD CLEAN DATASET — Llama 3.2 3B Instruct")
    print("=" * 60)

    random.seed(42)

    print("\n📂 Processando chunks do Allan...")
    todos = processar_chunks()

    print("\n🛡️  Adicionando exemplos de segurança clínica...")
    seguros = exemplos_seguranca()
    # Replica 10x para garantir peso suficiente no treino
    todos += seguros * 10
    print(f"   ✓ {len(seguros) * 10} exemplos de segurança adicionados")

    print(f"\n   Total: {len(todos)} exemplos")
    print("   Por fonte (top 10):")
    for fonte, n in Counter(e["fonte"] for e in todos).most_common(10):
        print(f"     {fonte:<45} {n:>4}")

    random.shuffle(todos)
    split     = int(len(todos) * (1 - VAL_RATIO))
    treino    = todos[:split]
    validacao = todos[split:]

    out = Path("data/datasets/synthetic_qa_dataset")
    out.mkdir(parents=True, exist_ok=True)

    with open(out / "dataset_treino.json", "w", encoding="utf-8") as f:
        json.dump(treino, f, ensure_ascii=False, indent=2)
    with open(out / "dataset_validacao.json", "w", encoding="utf-8") as f:
        json.dump(validacao, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Dataset salvo!")
    print(f"   Treino    : {len(treino)} exemplos")
    print(f"   Validação : {len(validacao)} exemplos")