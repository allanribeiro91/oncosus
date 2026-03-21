# Assistente IA OncoSUS

Assistente Inteligente de Protocolos Oncológicos do SUS

## 1. Contexto

- O Sistema Único de Saúde (SUS) disponibiliza um amplo conjunto de documentos técnicos que orientam o diagnóstico, tratamento e acompanhamento de diversas doenças, incluindo diferentes tipos de câncer. Entre esses documentos destacam-se:
  Protocolos Clínicos e Diretrizes Terapêuticas (PCDTs)
  Manuais clínicos do Instituto Nacional do Câncer (INCA)
  Guias de conduta e recomendações terapêuticas
  Documentos institucionais sobre fluxos de tratamento

- Esses materiais constituem a principal referência normativa para profissionais de saúde no atendimento oncológico dentro do SUS.
- No entanto, esse conjunto de documentos é extenso, distribuído em diversos arquivos e frequentemente apresenta centenas de páginas. A consulta manual a esses materiais pode ser demorada, especialmente em contextos de atendimento clínico onde decisões precisam ser tomadas com rapidez.
- Diante desse cenário, surge a oportunidade de utilizar técnicas de Inteligência Artificial para criar um assistente capaz de consultar esses documentos e retornar orientações baseadas nos protocolos oficiais do SUS.

## 2. Problema

- Profissionais de saúde e gestores frequentemente precisam consultar documentos técnicos extensos para verificar recomendações terapêuticas, critérios de tratamento e diretrizes clínicas.
- Entretanto, existem algumas dificuldades importantes:
  grande volume de documentos técnicos
  informações distribuídas em diversos arquivos
  tempo necessário para localizar trechos relevantes
  dificuldade de navegação em documentos extensos
  necessidade de garantir que as orientações estejam alinhadas com os protocolos oficiais
- Essas dificuldades podem tornar a consulta às diretrizes clínicas mais lenta e menos eficiente.

## 3. Solução Proposta

- Este projeto propõe o desenvolvimento do Assistente IA OncoSUS, um assistente inteligente capaz de responder perguntas relacionadas a tratamentos oncológicos com base em documentos oficiais do SUS e do Instituto Nacional do Câncer (INCA).
- O sistema utilizará técnicas modernas de Large Language Models (LLMs) combinadas com Retrieval-Augmented Generation (RAG) para recuperar informações relevantes diretamente dos documentos clínicos e gerar respostas estruturadas.
- O assistente terá como objetivo principal apoiar a consulta aos protocolos clínicos, retornando orientações baseadas nas diretrizes oficiais.
- É importante destacar que o sistema não tem como objetivo substituir o médico ou realizar diagnósticos, mas sim atuar como uma ferramenta de apoio à consulta de documentos técnicos.

## 4. Arquitetura Geral

- A arquitetura do sistema combina três componentes principais:

Classificação da pergunta
Um módulo baseado em LangChain identifica o tipo de câncer ou o tema principal da pergunta realizada pelo usuário.

Recuperação de conhecimento (RAG)
Um mecanismo de busca semântica consulta uma base vetorial construída a partir de documentos do SUS e do INCA, recuperando trechos relevantes dos protocolos clínicos.

Geração da resposta (LLM)
Um modelo de linguagem ajustado por fine-tuning utiliza o contexto recuperado para gerar uma resposta clara e estruturada, citando a fonte da informação.

Fluxo simplificado do sistema:
Pergunta do usuário
→ Classificação da pergunta
→ Recuperação de trechos relevantes dos documentos
→ Geração da resposta pelo modelo de linguagem
→ Retorno da resposta ao usuário com referência ao documento

## 5. Etapas de Desenvolvimento

- O desenvolvimento do Assistente IA OncoSUS será realizado em diferentes etapas.

### 1. Coleta e organização dos documentos

- Nesta etapa serão reunidos documentos oficiais relacionados ao tratamento de câncer no SUS, incluindo:
- Protocolos Clínicos e Diretrizes Terapêuticas (PCDTs)
- Manuais clínicos do INCA
- Guias institucionais de tratamento
- Os documentos serão organizados por tipo de câncer e área temática.

### 2. Processamento e preparação dos documentos

- Os documentos coletados serão convertidos para texto e submetidos a um processo de limpeza, incluindo:
  - remoção de cabeçalhos e rodapés
  - eliminação de elementos redundantes
  - padronização do texto
- Em seguida, os textos serão divididos em pequenos trechos (chunks) que poderão ser utilizados pelo sistema de busca semântica.

### 3. Vetorização dos documentos

- Cada trecho de texto será transformado em um vetor numérico utilizando modelos de embeddings semânticos.
- Esses vetores serão armazenados em um banco de dados vetorial, permitindo que o sistema realize buscas baseadas em similaridade semântica entre a pergunta do usuário e o conteúdo dos documentos.

### 4. Construção do mecanismo de recuperação (RAG)

- Será implementado um sistema de recuperação de informações baseado em Retrieval-Augmented Generation, capaz de:
  - localizar trechos relevantes dos documentos
    retornar o contexto mais apropriado para responder à pergunta do usuário
- Esse mecanismo permitirá que o modelo de linguagem utilize informações atualizadas diretamente dos documentos clínicos.

### 5. Geração de dataset para fine-tuning

- A partir dos documentos coletados, serão geradas perguntas e respostas sintéticas com o objetivo de criar um dataset de treinamento.
- Esse dataset será utilizado para ajustar o modelo de linguagem, ensinando-o a:
  - responder perguntas médicas
  - utilizar linguagem técnica apropriada
  - citar protocolos clínicos
  - estruturar respostas baseadas em diretrizes institucionais

### 6. Fine-tuning do modelo de linguagem

- O modelo de linguagem será ajustado por meio de fine-tuning supervisionado, utilizando dois tipos complementares de datasets.
- O fine-tuning tem como objetivo melhorar a interpretação das perguntas e a estrutura das respostas médicas.
- O conteúdo factual será recuperado dinamicamente por meio do RAG, que consulta diretamente os documentos clínicos indexados.

#### Dataset clínico geral

- Será utilizado um dataset público de perguntas e respostas médicas. Esse conjunto de dados permite que o modelo aprenda:
  - terminologia médica
  - padrões de perguntas clínicas
  - estrutura de respostas em contexto de saúde
- Esse treinamento fornece ao modelo uma base geral de linguagem médica.

#### Dataset sintético baseado nos documentos do SUS e INCA

- Também será criado um dataset sintético a partir dos documentos utilizados no sistema RAG, incluindo:
  - Protocolos Clínicos e Diretrizes Terapêuticas (PCDTs)
  - manuais e diretrizes do INCA
- A partir desses documentos serão geradas perguntas e respostas sintéticas seguindo um formato padronizado de resposta.
- Esse dataset tem como objetivo ensinar o modelo a:
  - responder com base em protocolos clínicos
  - estruturar respostas claras e objetivas
  - citar a fonte da informação.

### 7. Integração com LangChain

- O pipeline final será construído utilizando LangChain, integrando:
  - roteamento de perguntas
  - recuperação de documentos
  - geração de respostas pelo modelo de linguagem
- Essa arquitetura permitirá organizar o fluxo de decisão do assistente de forma modular e escalável.

### 8. Interface de interação com o usuário

- Por fim, será desenvolvida uma interface simples (com React) para interação com o assistente, permitindo que usuários realizem perguntas relacionadas a protocolos clínicos e recebam respostas contextualizadas.

## 9. Resultado Esperado

- Espera-se que o Assistente IA OncoSUS seja capaz de:
  - responder perguntas sobre tratamento oncológico no SUS
  - localizar rapidamente informações relevantes nos protocolos clínicos
  - apresentar respostas estruturadas e claras
  - citar os documentos utilizados como fonte da informação
- Dessa forma, o sistema poderá atuar como um assistente de consulta inteligente aos protocolos clínicos do SUS, contribuindo para facilitar o acesso às diretrizes terapêuticas oficiais.

---

## 10. Execução e Deploy

### Pré-requisitos

- Python 3.10+
- Node.js 18+
- Ollama instalado e rodando com o modelo `llama3` (ou outro configurado no RAG)
- Vector store populado em `data/vectorstore/` (rodar os scripts de ingestão se necessário)

### Desenvolvimento local

**1. Backend (API FastAPI)**

```bash
cd backend/rag
python -m pip install -r requirements.txt
```

Opcional: copie `backend/rag/.env.example` para `backend/rag/.env` (ou `.env` na raiz `oncosus-novo/`) e defina `ONCOSUS_OLLAMA_MODEL` para o nome do seu modelo Ollama (ex.: `oncosus-llm`). O Ollama precisa estar rodando.

```bash
python -m uvicorn app:app --reload --port 8000
```

A API estará em `http://localhost:8000`. Endpoints: `POST /api/chat`, `GET /api/health`.

**2. Frontend (Angular)**

```bash
cd frontend
npm install
npm start
```

O servidor de **desenvolvimento** sobe com `npm start` (não use só `npm run build` — o build só gera `dist/` e não abre porta). A app fica em `http://127.0.0.1:4200` (ou `http://localhost:4200`) e chama a API em `http://localhost:8000/api`. Se `localhost` não responder, abra explicitamente `http://127.0.0.1:4200`.

**3. CLI (alternativa ao frontend)**

```bash
cd backend/rag
python main.py
```

### Deploy em servidor único (Nginx)

1. Build do frontend: `cd frontend && npm run build`
2. Copiar `frontend/dist/oncosus-frontend` para o servidor (ex.: `/srv/oncosus/frontend/dist/oncosus-frontend`)
3. Copiar `backend/` e `data/` para o servidor
4. Usar `deploy/nginx.conf` como referência para configurar o Nginx (raiz = build Angular, `/api` = proxy para FastAPI na porta 8000)
5. Rodar a API com `deploy/start-api.sh` ou: `cd backend/rag && python -m uvicorn app:app --host 0.0.0.0 --port 8000`

### Path do vector store

O sistema procura o Chroma em `data/vectorstore/` (raiz do repo). Se não existir, tenta `backend/data/vectorstore/`. Garanta que o vector store esteja populado antes de usar a API.
