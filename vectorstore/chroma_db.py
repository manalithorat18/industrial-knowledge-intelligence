from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

DB_DIR = "db"

def create_vector_store(chunks, metadata):

    db = Chroma.from_texts(
        texts=chunks,
        metadatas=metadata,
        embedding=embedding_model,
        persist_directory=DB_DIR
    )

    return db


def load_vector_store():

    db = Chroma(
        persist_directory=DB_DIR,
        embedding_function=embedding_model
    )

    return db