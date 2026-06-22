import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
# from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from pypdf import PdfReader
import os

st.title("Industrial Knowledge Intelligence")

st.set_page_config(
    page_title="Industrial Knowledge Intelligence",
    page_icon="🏭",
    layout="wide"
)

st.markdown("""
### Unified Asset & Operations Brain

📄 Upload manuals, SOPs, reports, and regulations.

🤖 Ask questions about uploaded documents.

📌 Get AI-generated answers with source references.

📊 Generate document summaries.
""")

load_dotenv()

genai.configure(
    api_key=os.getenv("GOOGLE_API_KEY")
)

uploaded_file = st.file_uploader(
    "Upload PDF",
    type=["pdf"]
)

if uploaded_file:

    # Create docs folder if it doesn't exist
    os.makedirs("docs", exist_ok=True)

    # Save uploaded PDF
    save_path = os.path.join(
        "docs",
        uploaded_file.name
    )

    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Read PDF
    pdf = PdfReader(uploaded_file)

    text = ""

    for page in pdf.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted

    st.success("PDF Loaded Successfully!")

    st.text_area(
        "Document Content",
        text[:5000],
        height=400
    )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_text(text)

    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_db = FAISS.from_texts(
        chunks,
        embedding_model
    )

    st.success("FAISS Vector Database Created")

    st.write(f"Total Chunks Created: {len(chunks)}")

    with st.expander("View First Chunk"):
        st.write(chunks[0])

    # embedding_model_st = SentenceTransformer(
    #     "all-MiniLM-L6-v2"
    # )  

    # embeddings = model.encode(chunks)

    # st.success(
    #     f"Created {len(embeddings)} embeddings"
    # )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Pages", len(pdf.pages))

    with col2:
        st.metric("Chunks", len(chunks))

    with col3:
        st.metric("Embeddings", len(chunks))

    query = st.text_input(
        "Ask a question about the document"
    )
  
    if query:

        docs = vector_db.similarity_search(
            query,
            k=3
        )

        st.subheader("Retrieved Chunks")

        context = "\n\n".join(
           [doc.page_content for doc in docs]
        )
        
        prompt = f"""
        You are an Industrial Knowledge Assistant.

        Rules:
        1. Answer only from the provided context.
        2. If the answer is not in the context, say:
           "I could not find that information in the uploaded documents."
        3. Give concise professional answers.
        4. Mention important values, dates, procedures and regulations.

        Context:
        {context}

        Question:
        {query}
        """
        
        gemini_model = genai.GenerativeModel(
            "gemini-2.5-flash"
        )

        response = gemini_model.generate_content(
            prompt
        )

        st.subheader("AI Answer")

        st.write(response.text)

        with st.expander("Sources Used"):

            for i, doc in enumerate(docs):

                st.markdown(
                    f"### Source Chunk {i+1}"
                )

                st.write(doc.page_content)

        if st.button("Generate Knowledge Graph"):
            st.success("Knowledge Graph Generated")

if st.button("Generate Document Summary"):

    summary_prompt = f"""
    Analyze the following industrial document and provide:

    1. Executive Summary
    2. Key Equipment/Assets
    3. Important Procedures
    4. Risks & Safety Concerns
    5. Compliance Requirements
    6. Recommended Actions

    Document:
    {text[:15000]}
    """

    summary_response = gemini_model.generate_content(
        summary_prompt
    )

    st.subheader("Document Summary")
    st.write(summary_response.text)