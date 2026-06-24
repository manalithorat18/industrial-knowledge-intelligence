from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

def create_vector_db(chunks):

    db = Chroma.from_texts(
        texts=chunks,
        embedding=embedding_model,
        persist_directory="db"
    )

    return db