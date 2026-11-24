# Explainable Agentic RAG

An academic prototype that answers questions using information retrieved from PDF documents. The system combines document retrieval, evidence-based answer synthesis, explainability metadata, a FastAPI interface, and evaluation.

## Project overview

The system is designed to make answers traceable to their source documents. Instead of returning an answer alone, it can also return supporting evidence, source information, retrieval rankings/scores, and evidence-acceptance details.

### Main workflow

1. **PDF processing** – extract text from PDF documents while retaining page information.
2. **Chunking** – split extracted text into smaller passages for retrieval.
3. **Embeddings and storage** – represent text as vectors and use the project's ChromaDB-backed storage.
4. **Hybrid retrieval** – combine dense retrieval with keyword-based retrieval signals such as TF-IDF and BM25.
5. **Evidence selection** – rank and assess retrieved passages.
6. **Answer synthesis** – produce an extractive answer based on retrieved document evidence.
7. **Trace persistence** – save agent run details and evidence metadata for later inspection.
8. **Evaluation** – run the evaluation script against the configured questions and reference data.

> Note: The answer synthesizer is extractive/rule-based. Do not describe it as an LLM-generated answer unless the implementation is changed.

## Technology

- Python
- FastAPI and Uvicorn
- Sentence Transformers
- ChromaDB
- TF-IDF and BM25 retrieval components
- Reciprocal Rank Fusion (RRF) for combining supported ranking signals

## Repository structure

```text
.
├── api.py
├── app.py
├── requirements.txt
├── .env.example
├── data/
│   ├── evaluation/
│   │   └── ground_truth.json
│   ├── processed/
│   └── raw/
├── chroma_db/
├── src/
│   ├── agent.py
│   ├── answer_synthesizer.py
│   ├── chunker.py
│   ├── chroma_vector_store.py
│   ├── embedding_provider.py
│   ├── evaluation_agent.py
│   ├── evaluation_metrics.py
│   ├── hybrid_retriever.py
│   ├── pdf_loader.py
│   └── retriever.py
└── tests/
```

## Requirements

- Python 3.10 or later (use the Python version supported by the installed dependencies)
- Git
- Windows, macOS, or Linux

## Setup

Clone the repository and enter its directory:

```bash
git clone https://github.com/mgcwijekoon/Explainable-Agentic-RAG.git
cd Explainable-Agentic-RAG
```

Create and activate a virtual environment.

**Windows PowerShell:**

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**macOS / Linux:**

```bash
python -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

If the project uses environment variables, copy `.env.example` to `.env` and fill in only the values required by your local configuration. Do not commit `.env` or API keys.

## Running the API

From the project root, with the virtual environment activated:

```bash
python -m uvicorn api:app --host 127.0.0.1 --port 8003
```

Open the interactive API documentation:

```text
http://127.0.0.1:8003/docs
```

Available endpoints:

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/` | Basic API/root response |
| GET | `/health` | Check API and component health |
| POST | `/retrieve` | Retrieve relevant document passages |
| POST | `/ask` | Retrieve evidence and return an answer |

Use the request schema shown by Swagger (`/docs`) for `/retrieve` and `/ask`. Example question used in the project demonstration:

```json
{
  "question": "What is the maximum number of members in each group?",
  "top_k": 5
}
```

The exact request fields are defined by the current API schema. If Swagger shows a different schema, follow the schema displayed by the running application.

## Evaluation

Run the evaluation module from the repository root:

```bash
python -m src.evaluation_agent
```

The evaluation process uses the project's configured evaluation questions and reference data. Results are written to the processed-data area (for example, `data/processed/evaluation_agent_results.json`, depending on the current implementation).

Metrics used in the reported evaluation include page hit rate, all expected pages hit rate, fallback answers, context relevance, groundedness, answer relevance, and reference-answer similarity.

### Reported evaluation run

One recorded run on 10 questions reported:

| Metric | Result |
|---|---:|
| Page Hit Rate | 100.0% |
| All Expected Pages Hit Rate | 100.0% |
| Fallback Answers | 0 |
| Average Retrieval Iterations | 1 |
| Average Context Relevance | 0.4558 |
| Average Groundedness | 1.0 |
| Average Answer Relevance | 0.5028 |
| Average Reference Answer Similarity | 0.6717 |

These figures describe that specific 10-question run, not a guarantee of performance on other documents or question sets.

## Explainability and trace data

The API response can expose source documents, page numbers, retrieval ranks and scores, and evidence reasoning/acceptance details. Agent execution traces are persisted under `data/processed/` when trace persistence is enabled.

These details help users inspect which passages supported an answer. They should not be interpreted as a complete internal chain-of-thought.

## Limitations

- Evaluation coverage is limited to the configured questions and available source documents.
- High page-hit and groundedness scores do not by themselves guarantee high context or answer relevance.
- The extractive synthesizer may return answers that are less fluent or less focused than a generative system.
- A clean clone may require the source PDF files and a locally built vector index. Do not assume generated indexes or private documents are included in the public repository.
- Model downloads may require an internet connection the first time the embedding model is loaded.

## Academic use

This repository was developed as an academic project for the DS205.3 Data Science with Python coursework. Refer to the coursework brief and project report for assessment-specific details.

## YouTube Demo Video

[Watch the Explainable Agentic RAG Demo](https://youtu.be/-q4d7LgLWbw)

