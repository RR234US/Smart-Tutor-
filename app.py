"""Streamlit frontend for PDF Smart Subject Tutor."""

from __future__ import annotations

import time
import streamlit as st
from dotenv import load_dotenv
from pdf_rag import PDFTutor

load_dotenv()

st.set_page_config(
    page_title="Smart Subject Tutor - PDF First",
    page_icon="🎓",
    layout="wide",
)

st.title("🎓 Smart Subject Tutor")
st.caption("Upload PDFs. Ask questions. Get answers with page references.")

DEFAULT_MODELS = {
    "Gemini": "gemini-3.8-flash",
    "OpenAI": "gpt-6-luna",
    "Claude": "claude-haiku-5-5",
}

KEY_NAMES = {
    "Gemini": "GEMINI_API_KEY",
    "OpenAI": "OPENAI_API_KEY",
    "Claude": "ANTHROPIC_API_KEY",
}

# Initialize tutor
if "tutor" not in st.session_state:
    with st.spinner("Loading PDF search model..."):
        st.session_state.tutor = PDFTutor()

if "indexed" not in st.session_state:
    st.session_state.indexed = False

if "xp" not in st.session_state:
    st.session_state.xp = 0

if "api_calls" not in st.session_state:
    st.session_state.api_calls = 0

if "last_api_call" not in st.session_state:
    st.session_state.last_api_call = 0.0

# Sidebar
with st.sidebar:

    st.header("Learning Progress")
    st.metric("Learning XP", st.session_state.xp)

    if st.session_state.xp < 15:
        learner_level = "Beginner"
        st.info("Mode: Beginner")

    elif st.session_state.xp < 40:
        learner_level = "Intermediate"
        st.info("Mode: Intermediate")

    else:
        learner_level = "Advanced"
        st.success("Mode: Advanced")

    st.divider()

    st.header("App Safety Limits")

    st.write(
        f"API calls: {st.session_state.api_calls} / 25"
    )

    st.caption("5-second wait between API requests.")

# PDF Upload
st.subheader("1. Upload Your Subject PDFs")

uploaded_pdfs = st.file_uploader(
    "Choose PDF files",
    type=["pdf"],
    accept_multiple_files=True,
)

if uploaded_pdfs:

    st.write("Selected files:")

    for pdf_file in uploaded_pdfs:
        size_mb = pdf_file.size / (1024 * 1024)

        st.write(
            f"- {pdf_file.name} ({size_mb:.1f} MB)"
        )

index_col, clear_col = st.columns(2)

with index_col:

    if st.button(
        "Index Selected PDFs",
        type="primary",
        disabled=not uploaded_pdfs,
    ):

        too_large = [
            f.name for f in uploaded_pdfs
            if f.size > 25 * 1024 * 1024
        ]

        if too_large:
            st.error(
                "Each PDF must be 25 MB or smaller: "
                + ", ".join(too_large)
            )

        else:
            with st.spinner("Reading PDFs and creating search index..."):

                try:
                    stats = st.session_state.tutor.index_pdfs(
                        uploaded_pdfs
                    )

                    st.session_state.indexed = True

                    st.success(
                        f"Indexed {len(stats['files'])} file(s), "
                        f"{stats['pages_with_text']} pages and "
                        f"{stats['chunks']} searchable passages."
                    )

                    if stats["files_without_text"]:
                        st.warning(
                            "No selectable text found in: "
                            + ", ".join(stats["files_without_text"])
                        )

                except Exception as exc:
                    st.session_state.indexed = False
                    st.error(f"PDF indexing failed: {exc}")

with clear_col:

    if st.button("Clear Current PDF Index"):

        try:
            existing = st.session_state.tutor.collection.get()

            if existing.get("ids"):
                st.session_state.tutor.collection.delete(
                    ids=existing["ids"]
                )

            st.session_state.tutor.indexed_files = []
            st.session_state.tutor.indexed_pages = 0
            st.session_state.indexed = False

            st.success("PDF index cleared.")

        except Exception as exc:
            st.error(f"Could not clear index: {exc}")

st.divider()

# AI Provider
st.subheader("2. Choose Your AI Provider")

provider_col, model_col = st.columns([1, 2])

with provider_col:

    provider = st.selectbox(
        "Select Provider",
        ["Gemini", "OpenAI", "Claude"],
    )

with model_col:

    model = st.text_input(
        "Model ID",
        value=DEFAULT_MODELS[provider],
        key=f"model_{provider}",
    )

st.caption(
    f"Required API key: {KEY_NAMES[provider]}"
)

st.divider()

# Ask Question
st.subheader("3. Ask Your Question")

question = st.text_area(
    "What would you like to learn?",
    placeholder="Explain the four conditions required for deadlock.",
    height=100,
)

cooldown = max(
    0,
    5 - (time.time() - st.session_state.last_api_call)
)

can_ask = (
    st.session_state.indexed
    and bool(question.strip())
    and bool(model.strip())
    and st.session_state.api_calls < 25
    and cooldown <= 0
)

if st.button(
    "🚀 Ask the Tutor",
    type="primary",
    disabled=not can_ask,
):

    st.session_state.api_calls += 1
    st.session_state.last_api_call = time.time()

    with st.spinner(f"Searching PDFs using {provider}..."):

        try:
            answer, sources = st.session_state.tutor.answer(
                question=question.strip(),
                provider=provider,
                model=model.strip(),
                learner_level=learner_level,
                top_k=4,
            )

            st.session_state.xp += 5

            st.markdown("### 💡 Tutor's Answer")
            st.markdown(answer)

            with st.expander("📚 View Retrieved PDF Passages"):

                for idx, item in enumerate(sources, start=1):

                    st.markdown(
                        f"**{idx}. {item['source']} "
                        f"- Page {item['page']}**"
                    )

                    st.write(item["text"])

            st.caption("+5 XP for asking a question.")

        except Exception as exc:

            st.error(f"Request failed: {exc}")

            st.info(
                "Check your API key, model ID, API credits "
                "and provider usage limits."
            )

elif not st.session_state.indexed:

    st.info("Upload PDFs and click Index Selected PDFs first.")

elif cooldown > 0:

    st.warning(
        f"Please wait {cooldown:.1f} seconds."
    )

elif st.session_state.api_calls >= 25:

    st.warning("API request limit reached for this session.")

st.divider()

st.caption(
    "Privacy: Retrieved PDF passages are sent to your selected "
    "AI provider. Do not upload confidential documents without permission."
)