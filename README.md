# AI-Powered Document Search Engine

An intelligent document question-answering system that combines semantic search, keyword retrieval, hybrid ranking, deep-learning reranking, and Retrieval-Augmented Generation (RAG) to answer questions from PDF documents.

## Overview

Traditional keyword search can miss relevant information when a query uses different wording from a document. This project combines semantic and lexical retrieval to identify relevant passages, reranks them using a CrossEncoder model, and uses Google's Gemini API to generate answers grounded in the retrieved context.

## Key Features

- **PDF text extraction:** Extract text from PDF documents using PyPDF.
- **Text preprocessing:** Clean extracted text and split it into overlapping chunks.
- **Semantic search:** Generate embeddings using Sentence Transformers and retrieve similar passages with FAISS.
- **Keyword search:** Use BM25 to find passages containing relevant terms.
- **Hybrid retrieval:** Combine semantic and keyword rankings using Reciprocal Rank Fusion (RRF).
- **CrossEncoder reranking:** Reorder candidate passages according to query-passage relevance.
- **RAG answer generation:** Generate context-grounded answers using the Gemini API.
- **REST API:** Expose document question-answering functionality through Flask.
- **Source passages:** Return retrieved passages and reranker scores with API responses.

## Architecture

```text
PDF Document
     |
     v
Text Extraction (PyPDF)
     |
     v
Text Cleaning and Chunking
     |
     +------------------------+
     |                        |
     v                        v
Sentence Transformer         BM25
Embeddings                   Keyword Index
     |                        |
     v                        v
FAISS Semantic Search       Keyword Search
     |                        |
     +-----------+------------+
                 |
                 v
       Hybrid Retrieval (RRF)
                 |
                 v
       CrossEncoder Reranking
                 |
                 v
       Relevant Context Passages
                 |
                 v
          Gemini API (RAG)
                 |
                 v
        Grounded Answer + Sources
                 |
                 v
             Flask API
```

## Technology Stack

| Component | Technology |
|---|---|
| Language | Python |
| PDF extraction | PyPDF |
| Text embeddings | Sentence Transformers |
| Embedding model | `all-MiniLM-L6-v2` |
| Vector search | FAISS |
| Keyword retrieval | BM25 (`rank-bm25`) |
| Hybrid ranking | Reciprocal Rank Fusion |
| Reranking | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Answer generation | Google Gemini API |
| REST API | Flask |

## Repository Structure

```text
AI-Powered-document-search-engine/
├── data/
│   ├── sample.txt
│   └── sample.pdf
├── src/
│   ├── __init__.py
│   ├── app.py
│   ├── document_loader.py
│   ├── text_processor.py
│   ├── embeddings.py
│   ├── vector_store.py
│   ├── bm25_search.py
│   ├── search.py
│   └── reranker.py
├── requirements.txt
├── .gitignore
└── README.md
```

The RAG answer-generation component is implemented in `src/rag.py`.

## Prerequisites

- Python 3.10 or later
- Git
- A Google AI Studio API key with access to a supported Gemini model
- Sufficient memory and computing resources to run the embedding and reranking models

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Prag-git-cmd/AI-Powered-document-search-engine.git
cd AI-Powered-document-search-engine
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

Ensure `requirements.txt` includes:

```text
sentence-transformers
numpy
faiss-cpu
pypdf
rank-bm25
google-genai
flask
```

### 3. Configure the Gemini API key

Create an API key through [Google AI Studio](https://aistudio.google.com/).

Configure the key as an environment variable named `DOCUMENT_SEARCH_ENGINE`. In GitHub Codespaces, add it under your repository's **Settings → Secrets and variables → Codespaces**.

For a local development session, set the variable in your terminal instead.

Linux/macOS:

```bash
export DOCUMENT_SEARCH_ENGINE="YOUR_API_KEY"
```

Windows PowerShell:

```powershell
$env:DOCUMENT_SEARCH_ENGINE="YOUR_API_KEY"
```

Replace the placeholder with your actual key. Never commit API keys to Git.

### 4. Prepare your document

Place the PDF you want to search in the `data/` directory. The default example uses `data/sample.pdf`.

## Usage

### Run the command-line search application

From the repository root:

```bash
python src/search.py
```

Enter a question when prompted. The application retrieves relevant passages, reranks them, and generates an answer using Gemini.

### Start the REST API

```bash
python src/app.py
```

The API starts on port `5000`.

### Health check

Send a GET request to:

```text
http://127.0.0.1:5000/
```

Example response:

```json
{
  "status": "running",
  "message": "Document Search Engine API is ready"
}
```

### Ask a question

Send a POST request to `/ask`:

```bash
curl -X POST http://127.0.0.1:5000/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"What is retrieval augmented generation?"}'
```

Example response structure:

```json
{
  "question": "What is retrieval augmented generation?",
  "answer": "The generated answer based on retrieved document passages.",
  "sources": [
    {
      "passage": 1,
      "text": "A retrieved document passage...",
      "rerank_score": 7.95
    }
  ]
}
```

The answer and scores above illustrate the response structure; actual results depend on the document, query, model, and retrieval quality.

## How It Works

### 1. Document preprocessing

PDF text is extracted, cleaned, and divided into overlapping chunks. Chunking makes it possible to retrieve smaller, relevant sections instead of sending an entire document to the language model.

### 2. Semantic retrieval

Sentence Transformers converts document chunks and user queries into dense vector representations. FAISS searches for passages with high vector similarity.

### 3. Keyword retrieval

BM25 scores passages based on query-term relevance. This complements semantic retrieval when exact terms, names, or technical phrases matter.

### 4. Hybrid retrieval with RRF

Reciprocal Rank Fusion combines the rankings produced by FAISS and BM25. It uses ranks rather than directly comparing scores from different retrieval systems.

### 5. CrossEncoder reranking

A CrossEncoder evaluates query-passage pairs and reorders the retrieved candidates based on relevance.

### 6. Retrieval-Augmented Generation

The top-ranked passages are supplied to Gemini as context. The model is instructed to answer using the provided passages and acknowledge when the evidence is insufficient.

### 7. REST API

Flask provides endpoints for checking application health and submitting questions. The question-answer endpoint returns the answer and source passages as JSON.

## Error Handling and Limitations

- Answer quality depends on the quality of the extracted text and retrieved passages.
- Scanned PDFs may require OCR before their text can be searched.
- A relevant passage may be missed if chunking or retrieval parameters are unsuitable.
- Gemini requests depend on model availability, API quotas, rate limits, and network connectivity.
- The current implementation indexes the configured sample PDF at startup. Persistent indexes, document uploads, and multi-document management can be added later.
- The Flask development server is intended for development and testing, not production deployment.

## Future Improvements

- Add retrieval evaluation using Recall@K, Precision@K, and Mean Reciprocal Rank (MRR).
- Compare retrieval configurations and measure answer faithfulness.
- Add a document-upload endpoint and support multiple documents.
- Persist FAISS indexes and document metadata between application restarts.
- Add a frontend for interactive document search.
- Add automated tests for preprocessing, retrieval, reranking, and API endpoints.
- Add production deployment with a WSGI server and appropriate authentication.

## Learning Outcomes

This project demonstrates practical integration of information retrieval and language-model components, including dense embeddings, vector search, BM25, hybrid ranking, neural reranking, RAG, environment-based secret management, and REST API development.

## License

Choose and add an appropriate open-source license before distributing the project.
