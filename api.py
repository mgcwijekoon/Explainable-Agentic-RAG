import logging
import math
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.agent import ExplainableRAGAgent
from src.hybrid_retriever import HybridRetriever
from src.answer_synthesizer import ExtractiveAnswerSynthesizer


# =====================================
# LOGGING
# =====================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =====================================
# APPLICATION LIFESPAN
# =====================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.retriever = None
    app.state.synthesizer = None
    app.state.init_error = None

    try:
        logger.info("Initializing Explainable Agentic RAG API...")

        # Shared retrieval and synthesis components are initialized once.
        # A fresh agent instance is created for each /ask request so its
        # mutable state (last_state / retrieval counters) is not shared
        # between concurrent requests.
        app.state.retriever = HybridRetriever(
            dense_weight=0.5,
            rrf_k=10,
        )
        app.state.synthesizer = ExtractiveAnswerSynthesizer()

        logger.info("Explainable Agentic RAG API initialized successfully!")

    except Exception as error:
        app.state.init_error = str(error)
        logger.exception("RAG API initialization failed")

    yield

    logger.info("RAG API shutting down.")


# =====================================
# FASTAPI APPLICATION
# =====================================

app = FastAPI(
    title="Explainable Agentic RAG API",
    description=(
        "API for evidence-based question answering using hybrid retrieval, "
        "agentic reasoning, answer synthesis, and trace persistence."
    ),
    version="2.1.0",
    lifespan=lifespan,
)


# =====================================
# REQUEST MODEL
# =====================================

class QuestionRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=10)


# =====================================
# RESPONSE MODELS
# =====================================

class RetrievalConfig(BaseModel):
    dense_weight: float
    sparse_weight: float
    rrf_k: int


class SourceResponse(BaseModel):
    chunk_id: str
    source: str
    page: str

    dense_rank: int | None = None
    sparse_rank: int | None = None
    bm25_rank: int | None = None

    dense_similarity: float | None = None
    tfidf_similarity: float = 0.0
    bm25_similarity: float | None = None

    dense_score: float = 0.0
    sparse_score: float = 0.0
    hybrid_score: float = 0.0

    reasoning_details: dict | None = None


class RetrievalResponse(BaseModel):
    query: str
    retrieval_count: int
    retrieval_config: RetrievalConfig
    sources: list[SourceResponse]


class AnswerResponse(BaseModel):
    query: str
    answer: str
    retrieval_count: int
    retrieval_config: RetrievalConfig
    sources: list[SourceResponse]
    agent_status: str
    run_id: str
    trace_file: str


# =====================================
# INITIALIZATION CHECK
# =====================================

def check_initialization():
    if app.state.init_error:
        raise HTTPException(
            status_code=503,
            detail=f"Initialization failed: {app.state.init_error}",
        )

    if app.state.retriever is None:
        raise HTTPException(
            status_code=503,
            detail="Retriever is not initialized.",
        )

    if app.state.synthesizer is None:
        raise HTTPException(
            status_code=503,
            detail="Synthesizer is not initialized.",
        )


# =====================================
# SAFE CONVERSION HELPERS
# =====================================

def safe_float(value, default=0.0):
    try:
        number = float(value)
        return number if math.isfinite(number) else default
    except (TypeError, ValueError, OverflowError):
        return default


def safe_optional_float(value):
    if value is None:
        return None

    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError, OverflowError):
        return None


def safe_optional_int(value):
    if value is None:
        return None

    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return None


# =====================================
# RETRIEVAL CONFIGURATION
# =====================================

def get_retrieval_config():
    retriever = app.state.retriever

    return RetrievalConfig(
        dense_weight=safe_float(getattr(retriever, "dense_weight", 0.5)),
        sparse_weight=safe_float(getattr(retriever, "sparse_weight", 0.5)),
        rrf_k=safe_optional_int(getattr(retriever, "rrf_k", 10)) or 10,
    )


# =====================================
# SOURCE FORMATTING
# =====================================

def format_sources(results):
    sources = []

    for result in results or []:
        if not isinstance(result, dict):
            continue

        metadata = result.get("metadata") or {}
        if not isinstance(metadata, dict):
            metadata = {}

        # Raw retriever results use metadata; serialized agent evidence
        # stores source and page directly at the top level.
        source_value = result.get("source") or metadata.get(
            "source",
            metadata.get("file_name", "Unknown"),
        )
        page_value = result.get("page")
        if page_value is None:
            page_value = metadata.get(
                "page",
                metadata.get("page_number", "Unknown"),
            )

        reasoning = result.get("reasoning_details")
        if not isinstance(reasoning, dict):
            reasoning = None

        sources.append(
            SourceResponse(
                chunk_id=str(result.get("chunk_id", "Unknown")),
                source=str(source_value),
                page=str(page_value),
                dense_rank=safe_optional_int(result.get("dense_rank")),
                sparse_rank=safe_optional_int(result.get("sparse_rank")),
                bm25_rank=safe_optional_int(result.get("bm25_rank")),
                dense_similarity=safe_optional_float(
                    result.get("dense_similarity")
                ),
                tfidf_similarity=safe_float(
                    result.get("tfidf_similarity")
                ),
                bm25_similarity=safe_optional_float(
                    result.get("bm25_similarity")
                ),
                dense_score=safe_float(result.get("dense_score")),
                sparse_score=safe_float(result.get("sparse_score")),
                hybrid_score=safe_float(result.get("hybrid_score")),
                reasoning_details=reasoning,
            )
        )

    return sources


# =====================================
# ROOT ENDPOINT
# =====================================

@app.get("/")
def home():
    return {
        "message": "Explainable Agentic RAG API is running!",
        "version": app.version,
        "status": "online",
    }


# =====================================
# HEALTH CHECK
# =====================================

@app.get("/health")
def health():
    check_initialization()

    return {
        "status": "healthy",
        "retriever": "Hybrid Retriever (Dense + TF-IDF + BM25)",
        "agent": "ExplainableRAGAgent",
        "synthesizer": "Extractive Answer Synthesizer",
        "trace_persistence": "enabled",
    }


# =====================================
# RETRIEVAL ENDPOINT
# =====================================

@app.post("/retrieve", response_model=RetrievalResponse)
def retrieve_documents(request: QuestionRequest):
    check_initialization()

    try:
        results = app.state.retriever.search(
            request.query.strip(),
            top_k=request.top_k,
        ) or []

        return RetrievalResponse(
            query=request.query.strip(),
            retrieval_count=len(results),
            retrieval_config=get_retrieval_config(),
            sources=format_sources(results),
        )

    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))
    except Exception as error:
        logger.exception("Retrieval failed")
        raise HTTPException(status_code=500, detail=str(error))


# =====================================
# AGENTIC QUESTION ANSWERING ENDPOINT
# =====================================

@app.post("/ask", response_model=AnswerResponse)
def ask_question(request: QuestionRequest):
    check_initialization()

    try:
        # Reuse the initialized retriever and synthesizer, but create a
        # request-local agent to keep its mutable AgentState isolated.
        agent = ExplainableRAGAgent(
            retriever=app.state.retriever,
            synthesizer=app.state.synthesizer,
            initial_top_k=request.top_k,
            max_top_k=10,
            max_iterations=3,
        )

        answer = agent.run(request.query.strip())
        state = agent.get_last_state()

        if state is None:
            raise RuntimeError("Agent completed without returning an AgentState.")

        sources = format_sources(state.evidence)
        trace_file = str(agent.trace_store.file_path)

        return AnswerResponse(
            query=request.query.strip(),
            answer=str(answer),
            retrieval_count=len(state.evidence),
            retrieval_config=get_retrieval_config(),
            sources=sources,
            agent_status=str(state.status),
            run_id=str(state.run_id),
            trace_file=trace_file,
        )

    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))
    except Exception as error:
        logger.exception("Agentic question answering failed")
        raise HTTPException(status_code=500, detail=str(error))


# =====================================
# MAIN ENTRY POINT
# =====================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
    )
