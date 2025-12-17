# UIT Regulations Chatbot (RAG System)

This project implements a **Retrieval-Augmented Generation (RAG)** pipeline that allows users to ask questions in natural language (Vietnamese) and receive accurate answers based on the official university documents. The system leverages **Hybrid Search, Cross-Encoder Re-ranking, and a Fine-tuned Embedding Model** to ensure high retrieval accuracy.

## Key Features

- **Natural Language Q&A:** Users can query complex academic rules (e.g., graduation requirements, scholarship criteria) in Vietnamese.

- **Advanced RAG Pipeline:**
  - **Hybrid Search:** Combines keyword search (**BM25**) and semantic search (**ChromaDB**) to capture both exact matches and contextual meaning.

  - **Re-ranking:** Utilizes a Cross-Encoder (ms-marco-MiniLM-L-6-v2) to re-rank retrieved documents, ensuring the most relevant context is passed to the LLM.

- **Fine-tuned Embedding Model:** The embedding model ([hiieu/halong_embedding](https://huggingface.co/hiieu/halong_embedding)) was fine-tuned specifically on the UIT regulations dataset to improve domain adaptability.

- **Source Attribution:** Every answer includes citations to the specific articles or documents used.

- **Interactive Web Interface:** Built with **Flask** for easy interaction.

## Tech Stack & Architecture

- **LLM:** Google Gemini 2.5 Flash Lite.

- **Orchestration:** LangChain.

- **Vector Database:** ChromaDB.

- **Retrieval:** Ensemble Retriever (BM25 + Vector Search).

- **Re-ranking:** CrossEncoderReranker.

- **Backend:** Flask (Python).

- **Training:** Sentence Transformers (for fine-tuning).

### System Workflow

1. **Input:** User asks a question via the web interface.

2. **Retrieval:** The system fetches candidate documents using both BM25 (sparse) and ChromaDB (dense).

3. **Re-ranking:** The top candidates are re-scored and re-ordered by the Cross-Encoder to filter out noise.

4. **Generation:** The top 3 relevant chunks are sent to Google Gemini, which synthesizes the final answer.

5. **Output:** The answer is displayed to the user with source references.

## Model Performance

The embedding model was fine-tuned using `MultipleNegativesRankingLoss` on the university's regulation dataset. Evaluation metrics on the test set show significant improvement over the base model:

| Metric | Base Model | Fine-tuned Model | Improvement |
| :--- | :--- | :--- | :--- |
| **MRR@10** | 0.3334 | **0.4110** | +7.76% |
| **Accuracy@10** | 0.7254 | **0.8381** | +11.27% |

*Data based on evaluation results from `fine_tune_model.ipynb`.*
