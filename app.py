import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from pypdf import PdfReader

from vectorstore.chroma_db import (
    create_vector_store,
    load_vector_store
)

import os

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Industrial Knowledge Intelligence",
    page_icon="🏭",
    layout="wide"
)

st.title("🏭 Industrial Knowledge Intelligence")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.sidebar.success(
    "Multi-Document Retrieval Active"
)

st.markdown("""
### Unified Asset & Operations Brain

📄 Upload manuals, SOPs, reports, and regulations.

🤖 Ask questions about uploaded documents.

📌 Get AI-generated answers with source references.

📊 Generate document summaries.
""")

# -----------------------------
# Gemini Setup
# -----------------------------
load_dotenv()

genai.configure(
    api_key=os.getenv("GOOGLE_API_KEY")
)

if os.path.exists("db"):

    vector_db = load_vector_store()

    st.sidebar.success(
        "Existing Knowledge Base Loaded"
    )

else:

    vector_db = None

# -----------------------------
# File Upload
# -----------------------------
uploaded_files = st.file_uploader(
    "Upload PDFs",
    type=["pdf"],
    accept_multiple_files=True
)

# -----------------------------
# Process PDFs
# -----------------------------
if uploaded_files:

    all_chunks = []
    all_metadata = []

    total_pages = 0
    all_text = ""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    # Process every PDF
    for uploaded_file in uploaded_files:

        pdf = PdfReader(uploaded_file)

        total_pages += len(pdf.pages)

        text = ""

        for page in pdf.pages:

            extracted = page.extract_text()

            if extracted:
                text += extracted

        all_text += text + "\n\n"

        chunks = splitter.split_text(text)

        all_chunks.extend(chunks)

        metadata = [
            {
                "source": uploaded_file.name
            }
            for _ in chunks
        ]

        all_metadata.extend(metadata)

    st.success("PDFs Loaded Successfully!")

    # -----------------------------
    # Preview
    # -----------------------------
    st.text_area(
        "Document Preview",
        all_text[:5000],
        height=300
    )

    # -----------------------------
    # Create Persistent ChromaDB
    # -----------------------------
    vector_db = create_vector_store(
        all_chunks,
        all_metadata
    )

    st.success(
        "Knowledge Base Saved"
    )

    # -----------------------------
    # Metrics
    # -----------------------------
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Documents",
            len(uploaded_files)
        )

    with col2:
        st.metric(
            "Pages",
            total_pages
        )

    with col3:
        st.metric(
            "Chunks",
            len(all_chunks)
        )

    # -----------------------------
    # First Chunk Preview
    # -----------------------------
    with st.expander(
        "View First Chunk"
    ):

        st.write(
            all_chunks[0]
        )

    # -----------------------------
    # Generate Summary
    # -----------------------------
    if st.button(
        "Generate Document Summary"
    ):

        summary_text = "\n\n".join(
            all_chunks[:10]
        )

        summary_prompt = f"""
Analyze the following industrial documents and provide:

1. Executive Summary
2. Key Equipment/Assets
3. Important Procedures
4. Risks & Safety Concerns
5. Compliance Requirements
6. Recommended Actions

Documents:

{summary_text}
"""

        gemini_model = genai.GenerativeModel(
            "gemini-2.5-flash"
        )

        try:

            summary_response = gemini_model.generate_content(
                summary_prompt
            )

            st.subheader(
                "📊 Document Summary"
            )

            st.write(
                summary_response.text
            )

        except Exception:

            st.warning(
                "Gemini quota exceeded."
            )

            st.subheader(
                "📄 Document Overview"
            )

            st.write(
                f"Documents Uploaded: {len(uploaded_files)}"
            )

            st.write(
                f"Total Pages: {total_pages}"
            )

            st.write(
                f"Total Chunks: {len(all_chunks)}"
            )

    # -----------------------------
    # Preview Knowledge Base
    # -----------------------------
    with st.expander(
        "Preview Knowledge Base"
    ):

        for chunk in all_chunks[:5]:

            st.write(
                chunk[:500]
            )

            st.divider()

# =====================================================
# CHAT SECTION (OUTSIDE uploaded_files BLOCK)
# =====================================================

st.divider()

st.header(
    "💬 Ask Questions"
)

query = st.text_input(
    "Ask a question about the knowledge base"
)

if vector_db and query:

    retrieval_query = query

    if len(st.session_state.chat_history) > 0:

        recent_questions = " ".join(
            [
                item["question"]
                for item in st.session_state.chat_history[-3:]
            ]
        )

        retrieval_query = (
            recent_questions + " " + query
        )

    results = vector_db.similarity_search_with_score(
        retrieval_query,
        k=5
    )

    docs = []

    for doc, score in results:

        if score < 1.0:
            docs.append((doc, score))

    if len(docs) == 0:

        st.warning(
            "No relevant information found in the knowledge base."
        )

        st.stop()

    context = "\n\n".join(
        [doc.page_content for doc, score in docs]
    )

    history_text = ""

    for item in st.session_state.chat_history[-6:]:

        history_text += (
            f"User: {item['question']}\n"
            f"Assistant: {item['answer']}\n\n"
        )

    prompt = f"""
You are an Industrial Knowledge Assistant.

Previous Conversation:
{history_text}

Context:
{context}

Question:
{query}

Rules:
1. Answer only from the provided context.
2. If the answer is not in the context, say:
   "I could not find that information in the uploaded documents."
3. Give concise professional answers.
4. Mention important values, dates, procedures and regulations.

"""

    gemini_model = genai.GenerativeModel(
        "gemini-2.5-flash"
    )

    try:

        response = gemini_model.generate_content(
            prompt
        )

        st.subheader(
            "🤖 AI Answer"
        )

        st.write(
            response.text
        )

        st.session_state.chat_history.append(
            {
               "question": query,
               "answer": response.text
            }
        )

    except Exception:

        st.warning(
            "Gemini quota exceeded. Showing retrieved content instead."
        )

        st.subheader(
            "📚 Retrieved Information"
        )

        for i, (doc, score) in enumerate(docs):

            st.markdown(
                f"### Result {i+1}"
            )

            st.write(
                f"📄 Source: {doc.metadata['source']}"
            )

            st.write(
                f"🎯 Similarity Score: {score:.4f}"
            )

            content = doc.page_content

            if len(content) > 700:
                content = content[:700] + "..."

            st.write(content)

            st.divider()

    # -----------------------------
    # Sources
    # -----------------------------
    with st.expander(
        "Sources Used"
    ):

        for i, (doc, score) in enumerate(docs):

            st.markdown(
                f"### Source Chunk {i+1}"
            )

            st.write(
                f"📄 Document: {doc.metadata['source']}"
            )

            confidence = max(
                0,
                round((1 / (1 + score)) * 100, 2)
            )

            st.write(
                f"🎯 Confidence: {confidence}%"
            )

            st.write(
                doc.page_content
            )

            st.divider()

# -----------------------------
# Placeholder Knowledge Graph
# -----------------------------

st.divider()

st.header("📝 Chat History")

for item in reversed(
    st.session_state.chat_history
):

    with st.expander(
        item["question"]
    ):

        st.write(
            item["answer"]
        )

# if st.button(
#     "Generate Knowledge Graph"
# ):

#                 st.write(
#                     doc.page_content
#                 )

#                 st.divider()

# -----------------------------
# Placeholder Knowledge Graph
# -----------------------------
st.divider()

st.header("📝 Chat History")

for item in reversed(st.session_state.chat_history):

    with st.expander(item["question"]):

        st.write(item["answer"])


if st.button("Generate Knowledge Graph"):

    st.success(
        "Knowledge Graph feature coming in Week 2 🚀"
    )

