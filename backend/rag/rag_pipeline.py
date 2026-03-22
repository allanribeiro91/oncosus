# rag_pipeline.py — OncoSUS
# Qwen2.5-3B-Instruct fine-tuned + ChromaDB nativo (Allan)

from pathlib import Path
import torch
import chromadb
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

from prompt import PROMPT_TEMPLATE

PROJECT_ROOT   = Path(__file__).resolve().parent.parent.parent
TOKENIZER_PATH = str(PROJECT_ROOT / "finetuning" / "output" / "final_adapter")
ADAPTER_PATH   = str(PROJECT_ROOT / "finetuning" / "output" / "final_adapter")
BASE_MODEL     = "Qwen/Qwen2.5-3B-Instruct"

SYSTEM_PROMPT = (
    "Você é um assistente clínico especializado em oncologia do SUS. "
    "Responda APENAS com base nos trechos fornecidos dos documentos oficiais "
    "(PCDTs e manuais do INCA). Se a informação não estiver nos trechos, "
    "diga 'Informação não encontrada nos documentos disponíveis.' "
    "Nunca faça prescrições diretas ou substitua a avaliação médica."
)


def load_finetuned_model():
    print(f"🤖 Carregando {BASE_MODEL} fine-tuned...")

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb,
        device_map="auto",
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = PeftModel.from_pretrained(base, ADAPTER_PATH)
    model.eval()

    print("✅ Modelo carregado!")
    return model, tokenizer


class RAGPipeline:
    def __init__(
        self,
        persist_directory: str,
        embedding_model: str = "intfloat/multilingual-e5-base",
        llm_model: str = "qwen25-finetuned",
        top_k: int = 8,
        final_k: int = 2,
    ):
        print("📐 Carregando embeddings...")
        self.embedding_model = SentenceTransformer(embedding_model)

        print("🗄️  Conectando ao ChromaDB...")
        chroma_client   = chromadb.PersistentClient(path=persist_directory)
        self.collection = chroma_client.get_or_create_collection(
            name="oncology_documents"
        )
        print(f"   ✅ {self.collection.count()} chunks indexados")

        self.top_k   = top_k
        self.final_k = final_k
        self.model, self.tokenizer = load_finetuned_model()

    # ----------------------------------
    # 1. Retrieval
    # ----------------------------------
    def retrieve(self, query: str):
        query_embedding = self.embedding_model.encode([query]).tolist()
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=self.top_k,
        )

        class Doc:
            def __init__(self, content, metadata):
                self.page_content = content
                self.metadata     = metadata

        docs = []
        for i, text in enumerate(results["documents"][0]):
            meta = results["metadatas"][0][i] if results["metadatas"] else {}
            meta["titulo"] = meta.get("document_title", "Documento")
            meta["fonte"]  = meta.get("source", "")
            meta["secao"]  = "N/A"
            meta["pagina"] = "N/A"
            docs.append(Doc(text, meta))
        return docs

    # ----------------------------------
    # 2. Seleção
    # ----------------------------------
    def select_top_docs(self, docs):
        return [d for d in docs if d.page_content.strip()][:self.final_k]

    # ----------------------------------
    # 3. Contexto
    # ----------------------------------
    def build_citation(self, metadata):
        titulo = metadata.get("titulo", "Documento")
        fonte  = metadata.get("fonte", "")
        year   = metadata.get("year", "")
        parts  = [titulo]
        if fonte and fonte.upper() not in titulo.upper():
            parts.append(fonte.upper())
        if year:
            parts.append(str(year))
        return " – ".join(parts)

    def build_context(self, docs):
        context_str = ""
        for i, doc in enumerate(docs, 1):
            meta     = doc.metadata or {}
            citation = self.build_citation(meta)
            context_str += (
                f"\n[Documento {i}]\n"
                f"Fonte: {citation}\n\n"
                f"{doc.page_content}\n"
            )
        return context_str

    def format_sources(self, docs):
        sources, seen = [], set()
        for doc in docs:
            citation = self.build_citation(doc.metadata or {})
            if citation not in seen:
                seen.add(citation)
                sources.append(citation)
        return sources

    def build_prompt(self, question: str, context: str):
        return PROMPT_TEMPLATE.format(question=question, context=context)

    # ----------------------------------
    # 4. Geração — Qwen2.5 Chat Template
    # ----------------------------------
    def generate_answer(self, question: str, context: str) -> str:
        user_msg = (
            f"Com base nos documentos oficiais do SUS abaixo, responda: {question}\n\n"
            f"Documentos recuperados:\n{context}"
        )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg},
        ]

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        ).to(self.model.device)

        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=False,
                temperature=0.1,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.2,
            )

        novos    = out[0][input_len:]
        resposta = self.tokenizer.decode(novos, skip_special_tokens=True).strip()

        for corte in ["<|im_end|>", "<|im_start|>", "assistant\n"]:
            if corte in resposta:
                resposta = resposta[:resposta.index(corte)].strip()

        return resposta or "Informação não encontrada nos documentos disponíveis."

    # ----------------------------------
    # 5. Pipeline completo
    # ----------------------------------
    def run(self, question: str):
        docs          = self.retrieve(question)
        selected_docs = self.select_top_docs(docs)
        context       = self.build_context(selected_docs)
        sources       = self.format_sources(selected_docs)
        answer        = self.generate_answer(question, context)
        return {
            "question": question,
            "answer":   answer,
            "sources":  sources,
        }