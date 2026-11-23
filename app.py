from pathlib import Path

import streamlit as st

from src.agent import ExplainableRAGAgent

from src.chunker import main as create_chunks

from src.embeddings import generate_embeddings

from src.vector_store import store_embeddings

from src.evaluation_metrics import RAGEvaluationMetrics


# ==========================================
# PROJECT PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

PDF_FOLDER = BASE_DIR / "data" / "raw"

PDF_PATH = PDF_FOLDER / "sample.pdf"


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(

    page_title="Explainable Agentic RAG",

    page_icon="📚",

    layout="wide"

)


# ==========================================
# SESSION STATE
# ==========================================

if "agent" not in st.session_state:

    st.session_state.agent = None

if "chat_history" not in st.session_state:

    st.session_state.chat_history = []


# ==========================================
# HEADER
# ==========================================

st.title("📚 Explainable Agentic RAG")

st.subheader(
    "Data Science PDF Assistant"
)

st.caption(
    "Ask questions from uploaded PDF documents "
    "using an Explainable Hybrid RAG System."
)

st.divider()


# ==========================================
# SIDEBAR - PDF UPLOAD
# ==========================================

st.sidebar.title("📄 Document Manager")

uploaded_file = st.sidebar.file_uploader(

    "Upload PDF Document",

    type=["pdf"]

)

st.sidebar.caption(
    "Upload a PDF to process and index its contents."
)


# ==========================================
# PROCESS PDF
# ==========================================

if st.sidebar.button(
    "🚀 Process PDF",
    use_container_width=True
):

    if uploaded_file is None:

        st.sidebar.error(
            "Please upload a PDF first!"
        )

    else:

        try:

            PDF_FOLDER.mkdir(
                parents=True,
                exist_ok=True
            )

            with st.spinner(
                "Processing PDF document..."
            ):

                # Save uploaded PDF using the filename
                # expected by the existing pipeline.

                PDF_PATH.write_bytes(
                    uploaded_file.getbuffer()
                )

                # STEP 1: PDF PROCESSING AND CHUNKING

                st.write("Processing PDF...")

                create_chunks()

                # STEP 2: EMBEDDING GENERATION

                st.write("Generating embeddings...")

                generate_embeddings()

                # STEP 3: VECTOR DATABASE

                st.write("Updating ChromaDB...")

                store_embeddings()

                # STEP 4: INITIALIZE AGENT

                st.session_state.agent = (
                    ExplainableRAGAgent()
                )

                # Clear old chat history
                st.session_state.chat_history = []

            st.sidebar.success(
                "PDF processed successfully!"
            )

            st.success(
                f"Successfully processed: {uploaded_file.name}"
            )

            st.rerun()

        except Exception as error:

            st.error(
                f"PDF Processing Error: {error}"
            )


# ==========================================
# LOAD EXISTING DATABASE
# ==========================================

if st.sidebar.button(
    "📂 Load Existing Documents",
    use_container_width=True
):

    try:

        with st.spinner("Loading existing documents..."):

            st.session_state.agent = (
                ExplainableRAGAgent()
            )

        st.sidebar.success(
            "Existing documents loaded!"
        )

        st.rerun()

    except Exception as error:

        st.sidebar.error(
            f"Unable to load database: {error}"
        )


# ==========================================
# SYSTEM STATUS
# ==========================================

st.sidebar.divider()

st.sidebar.subheader("System Status")

if st.session_state.agent is not None:

    st.sidebar.success("RAG System: Online")

    st.sidebar.success("Agent: Active")

    st.sidebar.info(
        f"Documents: "
        f"{st.session_state.agent.retriever.dense_retriever.vector_store.count()}"
    )

else:

    st.sidebar.warning(
        "RAG System: Not Initialized"
    )


# ==========================================
# MAIN CHAT INTERFACE
# ==========================================

st.header("💬 Ask Your Questions")

if st.session_state.agent is None:

    st.info(
        "Upload a PDF and click 'Process PDF' "
        "to start asking questions."
    )

else:

    st.success(
        "Your document is ready! Ask a question below."
    )

    user_question = st.chat_input(
        "Type your question here..."
    )

    # ======================================
    # PROCESS QUESTION
    # ======================================

    if user_question:

        try:

            with st.spinner(
                "Agent is searching and analyzing..."
            ):

                agent = st.session_state.agent

                # STEP 1: Generate answer

                answer = agent.run(
                    user_question
                )

                state = agent.last_state

                # STEP 2: Retrieve evidence

                evidence = state.evidence

                contexts = [
                    item["text"]
                    for item in evidence
                ]

                # STEP 3: Calculate evaluation metrics

                metrics = RAGEvaluationMetrics(

                    embedding_provider=(
                        agent.retriever
                        .dense_retriever
                        .embedding_provider
                    )

                )

                context_relevance = (
                    metrics.context_relevance(
                        user_question,
                        contexts
                    )
                )

                groundedness = (
                    metrics.groundedness(
                        answer,
                        contexts
                    )
                )

                answer_relevance = (
                    metrics.answer_relevance(
                        user_question,
                        answer
                    )
                )

                evaluation_results = {

                    "context_relevance":
                        round(context_relevance, 4),

                    "groundedness":
                        round(groundedness, 4),

                    "answer_relevance":
                        round(answer_relevance, 4)

                }

                # STEP 4: Save chat history

                st.session_state.chat_history.append({

                    "question": user_question,

                    "answer": answer,

                    "evidence": evidence,

                    "metrics": evaluation_results,

                    "status": state.status

                })

        except Exception as error:

            st.error(
                f"Question Processing Error: {error}"
            )


# ==========================================
# DISPLAY CHAT HISTORY
# ==========================================

for chat in st.session_state.chat_history:

    # USER MESSAGE

    with st.chat_message("user"):

        st.markdown(
            chat["question"]
        )

    # ASSISTANT MESSAGE

    with st.chat_message("assistant"):

        st.markdown("### 🤖 AI Answer")

        st.markdown(
            chat["answer"]
        )

        # ==================================
        # RETRIEVED CHUNKS
        # ==================================

        with st.expander(
            "📑 View Retrieved Chunks"
        ):

            evidence = chat["evidence"]

            if not evidence:

                st.warning(
                    "No relevant chunks retrieved."
                )

            for index, item in enumerate(
                evidence,
                start=1
            ):

                st.markdown(
                    f"#### Retrieved Chunk {index}"
                )

                st.write(
                    "Chunk ID:",
                    item["chunk_id"]
                )

                st.write(
                    "Source:",
                    item["source"]
                )

                st.write(
                    "Page:",
                    item["page"]
                )

                st.write(
                    "Dense Similarity:",
                    item["dense_similarity"]
                )

                st.write(
                    "TF-IDF Similarity:",
                    item["tfidf_similarity"]
                )

                st.write(
                    "Hybrid Score:",
                    item["hybrid_score"]
                )

                st.text_area(

                    "Retrieved Text",

                    value=item["text"],

                    height=130,

                    disabled=True,

                    key=(
                        f"{item['chunk_id']}_"
                        f"{index}_"
                        f"{len(st.session_state.chat_history)}"
                    )

                )

                st.divider()

        # ==================================
        # EVALUATION METRICS
        # ==================================

        with st.expander(
            "📊 Evaluation Metrics"
        ):

            metric_data = chat["metrics"]

            col1, col2, col3 = st.columns(3)

            col1.metric(

                "Context Relevance",

                f"{metric_data['context_relevance']:.4f}"

            )

            col2.metric(

                "Groundedness",

                f"{metric_data['groundedness']:.4f}"

            )

            col3.metric(

                "Answer Relevance",

                f"{metric_data['answer_relevance']:.4f}"

            )

            st.write(
                "Agent Status:",
                chat["status"]
            )


# ==========================================
# FOOTER
# ==========================================

st.divider()

st.caption(
    "Explainable Agentic RAG System | "
    "DS205.3 - Data Science with Python"
)