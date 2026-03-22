from avaliar_rag_com_chatgpt import avaliar_rag_com_chatgpt
from gerar_respostas_com_oncosus import gerar_respostas_com_oncosus

def main():
    print("\n🔍 Gerando respostas com o RAG...\n")
    gerar_respostas_com_oncosus()

    print("\n🤖 Avaliando respostas com ChatGPT...\n")
    avaliar_rag_com_chatgpt()

if __name__ == "__main__":
    main()