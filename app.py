import html
import streamlit as st

from src.pdf_processor import extract_pdf_pages
from src.retriever import split_into_chunks, retrieve_passages
from src.qa_model import QuestionAnswerer


# 1. PAGE CONFIGURATION
st.set_page_config(
    page_title="StudyDoc AI | Document Intelligence",
    page_icon="📘",
    layout="wide",
    initial_sidebar_state="expanded",
)


# 2. PROFESSIONAL UI STYLING
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background:
            radial-gradient(ellipse at 0% 0%, rgba(45, 91, 160, 0.17), transparent 35%),
            #0b1020;
        color: #e8edf8;
    }

    [data-testid="stHeader"] {
        background: rgba(11, 16, 32, 0.85);
    }

    [data-testid="stSidebar"] {
        background: #10182a;
        border-right: 1px solid rgba(148, 163, 184, 0.17);
    }

    .block-container {
        max-width: 1250px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    .hero {
        padding: 30px;
        border: 1px solid rgba(148, 163, 184, 0.22);
        border-radius: 22px;
        background: linear-gradient(
            125deg, #172746 0%, #111a30 58%, #142239 100%
        );
        margin-bottom: 24px;
    }

    .eyebrow {
        color: #90b8ff;
        text-transform: uppercase;
        letter-spacing: 2px;
        font-size: 11px;
        font-weight: 800;
    }

    .hero h1 {
        margin: 10px 0;
        color: #f5f8ff;
        font-size: clamp(28px, 4vw, 42px);
        font-weight: 800;
        letter-spacing: -1.2px;
    }

    .hero p {
        color: #b7c4da;
        font-size: 14px;
        line-height: 1.8;
        margin-bottom: 0;
    }

    .section-title {
        color: #f2f6ff;
        font-size: 19px;
        font-weight: 750;
        margin: 22px 0 12px 0;
    }

    .metric-card {
        background: rgba(20, 31, 52, 0.9);
        border: 1px solid rgba(148, 163, 184, 0.18);
        border-radius: 15px;
        padding: 17px 19px;
        min-height: 92px;
    }

    .metric-label {
        color: #91a1bb;
        font-size: 12px;
        margin-bottom: 8px;
    }

    .metric-value {
        color: #f4f7ff;
        font-size: 24px;
        font-weight: 800;
    }

    .answer-card {
        background: linear-gradient(
            140deg, rgba(23, 42, 71, 0.98), rgba(17, 27, 46, 0.98)
        );
        border: 1px solid rgba(96, 165, 250, 0.35);
        border-radius: 18px;
        padding: 24px;
        margin: 14px 0;
    }

    .answer-label {
        color: #91bdff;
        font-size: 11px;
        font-weight: 800;
        letter-spacing: 1.6px;
        text-transform: uppercase;
    }

    .answer-text {
        color: #f7f9ff;
        font-size: 21px;
        font-weight: 600;
        line-height: 1.7;
        margin-top: 12px;
        overflow-wrap: anywhere;
    }

    .source-chip {
        display: inline-block;
        padding: 7px 11px;
        border-radius: 8px;
        background: rgba(59, 130, 246, 0.15);
        color: #a9ccff;
        font-size: 12px;
        margin-top: 15px;
    }

    .muted {
        color: #93a3bc;
        font-size: 12px;
    }

    div.stButton > button {
        border-radius: 10px;
        min-height: 44px;
        font-weight: 700;
        border: 1px solid #5a91ff;
        background: linear-gradient(120deg, #3677f5, #5b65ed);
        color: white;
        transition: all 0.2s ease;
    }

    div.stButton > button:hover {
        border-color: #8cb4ff;
        filter: brightness(1.08);
        transform: translateY(-1px);
    }

    div[data-testid="stExpander"] {
        border: 1px solid rgba(148, 163, 184, 0.20);
        border-radius: 12px;
        background: rgba(16, 24, 42, 0.75);
    }

    div[data-testid="stFileUploader"] {
        border: 1px dashed rgba(125, 164, 230, 0.55);
        border-radius: 14px;
        padding: 12px;
        background: rgba(20, 31, 52, 0.45);
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# 3. MODEL LOADING
@st.cache_resource(show_spinner="Loading question-answering model...")
def load_answerer():
    return QuestionAnswerer()


# 4. SIDEBAR
with st.sidebar:
    st.markdown("## 📘 StudyDoc AI")
    st.caption("Document intelligence workspace")

    st.divider()

    st.markdown("### Search settings")

    top_k = st.slider(
        "Passages to retrieve",
        min_value=1,
        max_value=5,
        value=3,
        help="Controls how many relevant passages are checked.",
    )

    st.divider()

    st.markdown("### Processing pipeline")
    st.markdown(
        """
        **01 · Extract**  
        Read PDF text page by page.

        **02 · Retrieve**  
        Find relevant passages.

        **03 · Answer**  
        Extract an answer using DistilBERT.

        **04 · Verify**  
        Inspect the answer's source text.
        """
    )

    st.divider()

    # Load the model once and show the actual source.
    try:
        answerer = load_answerer()
        model_status = getattr(
            answerer,
            "model_source",
            "Loaded QA model",
        )
        st.success("Question-answering model loaded")
        st.caption(f"Model source: {model_status}")
    except Exception as model_error:
        answerer = None
        st.error("Could not load the QA model")
        st.caption(str(model_error))

    st.divider()
    st.caption("StudyDoc AI · Educational prototype")



# 5. HEADER
st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">Document intelligence workspace</div>
        <h1>📘 StudyDoc AI</h1>
        <p>
            Ask questions about your study material. Get an extracted answer,
            inspect the supporting evidence, and trace it to the source page.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# 6. PDF UPLOAD
upload_col, info_col = st.columns([1.3, 0.7], gap="large")

with upload_col:
    st.markdown(
        '<div class="section-title">Your document</div>',
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Upload a text-based PDF",
        type=["pdf"],
        help="Text-based PDFs work best. Scanned PDFs need OCR.",
    )

with info_col:
    st.markdown(
        '<div class="section-title">Ask your document</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="metric-card">
            <div class="metric-label">Example questions</div>
            <div style="color:#e8edf8; line-height:1.9; font-size:13px;">
                How do neural networks work?<br>
                What is self-attention?<br>
                What happens during training?
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# 7. PROCESS DOCUMENT
if uploaded_file is None:
    st.info("Upload a PDF to begin.")
    st.stop()

try:
    pdf_bytes = uploaded_file.getvalue()

    with st.spinner("Extracting text and preparing passages..."):
        pages = extract_pdf_pages(pdf_bytes)
        chunks = split_into_chunks(pages)

    if not pages or not chunks:
        st.error(
            "No readable text was found. The PDF may be scanned or image-only. "
            "OCR is not included in this version."
        )
        st.stop()

    # Document metrics
    metric1, metric2, metric3 = st.columns(3, gap="medium")

    with metric1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Text pages</div>
                <div class="metric-value">{len(pages)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with metric2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Passages indexed</div>
                <div class="metric-value">{len(chunks)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with metric3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">File size</div>
                <div class="metric-value">{uploaded_file.size / 1024:.1f} KB</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # 8. QUESTION INPUT
    st.markdown(
        '<div class="section-title">Ask a question</div>',
        unsafe_allow_html=True,
    )

    with st.form("question_form"):
        question = st.text_input(
            "Question",
            placeholder="e.g. What happens during neural network training?",
            label_visibility="collapsed",
        )

        submitted = st.form_submit_button(
            "✦  Find answer",
            type="primary",
            use_container_width=True,
        )

    # 9. RETRIEVAL AND ANSWERING
    if submitted:
        if not question.strip():
            st.warning("Please enter a question.")
            st.stop()

        if answerer is None:
            st.error(
                "The QA model did not load. Check the model error in the sidebar."
            )
            st.stop()

        with st.spinner("Searching for relevant passages..."):
            passages = retrieve_passages(
                question,
                chunks,
                top_k=top_k,
            )

        if not passages:
            st.warning(
                "No matching passages found. Try using important words "
                "from the document in your question."
            )
            st.stop()

        candidates = []

        with st.spinner("Extracting and comparing candidate answers..."):
            for passage in passages:
                result = answerer.answer(
                    question=question,
                    context=passage["text"],
                )

                candidates.append(
                    {
                        "answer": result.get("answer", ""),
                        "qa_score": float(result.get("score", 0.0)),
                        "retrieval_score": passage["score"],
                        "page": passage["page"],
                        "text": passage["text"],
                    }
                )

        # This is a ranking heuristic, not a calibrated confidence score.
        candidates.sort(
            key=lambda item: (
                item["qa_score"],
                item["retrieval_score"],
            ),
            reverse=True,
        )

        best = candidates[0]


        # 10. ANSWER DISPLAY
        st.markdown(
            '<div class="section-title">Answer</div>',
            unsafe_allow_html=True,
        )

        if best["answer"]:
            safe_answer = html.escape(best["answer"])

            st.markdown(
                f"""
                <div class="answer-card">
                    <div class="answer-label">Extracted answer</div>
                    <div class="answer-text">{safe_answer}</div>
                    <div class="source-chip">
                        Source page · {best["page"]}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.warning(
                "The model could not extract a clear answer. "
                "Review the passages below or rephrase the question."
            )

        st.caption(
            "The QA score is a model span score, not the probability "
            "that the answer is correct."
        )


        # 11. SOURCE EVIDENCE
        with st.expander("📄 Supporting passage", expanded=True):
            st.markdown(f"**Source page: {best['page']}**")
            st.write(best["text"])

        with st.expander("🔎 All retrieved passages"):
            for index, item in enumerate(candidates, start=1):
                st.markdown(
                    f"### Passage {index} · Page {item['page']}"
                )

                if item["answer"]:
                    st.markdown(
                        f"**Candidate answer:** {item['answer']}"
                    )

                score_col1, score_col2 = st.columns(2)

                with score_col1:
                    st.caption(
                        f"QA span score: {item['qa_score']:.4f}"
                    )

                with score_col2:
                    st.caption(
                        f"TF-IDF retrieval score: "
                        f"{item['retrieval_score']:.4f}"
                    )

                st.write(item["text"])
                st.divider()


    # 12. DOCUMENT PREVIEW
    with st.expander("📚 Extracted document text"):
        selected_page = st.selectbox(
            "Choose a page",
            options=[page["page"] for page in pages],
        )

        selected_content = next(
            page["text"]
            for page in pages
            if page["page"] == selected_page
        )

        st.text_area(
            "Extracted text",
            value=selected_content,
            height=250,
            disabled=True,
        )

except Exception as error:
    st.error(f"Unable to process the PDF: {error}")
    st.info(
        "If this is the first run, check your internet connection so "
        "the model can download. Also verify that all four source files "
        "are saved correctly."
    )


# 13. LIMITATIONS
with st.expander("ℹ️ How it works and limitations"):
    st.markdown(
        """
        **Processing pipeline**

        1. PyMuPDF extracts selectable text from each PDF page.
        2. Text is divided into passages.
        3. TF-IDF ranks passages by word overlap with the question.
        4. DistilBERT extracts an answer span from each retrieved passage.
        5. The app displays a candidate answer and its supporting text.

        **Limitations**

        - The QA model extracts text spans; it does not generate a full explanation.
        - TF-IDF may miss relevant passages when the question uses different wording.
        - Scanned PDFs, complicated tables, and multi-column layouts may be read incorrectly.
        - Always verify the answer against the displayed source passage.
        """
    )
