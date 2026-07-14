# import os
# import shutil

from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

DEFAULT_DB_DIR = "db"

def create_vector_store(chunks, metadata, db_dir=DEFAULT_DB_DIR):

    # Delete previous database
    # if os.path.exists(DB_DIR):
    #     shutil.rmtree(DB_DIR)

    db = Chroma.from_texts(
        texts=chunks,
        metadatas=metadata,
        embedding=embedding_model,
        persist_directory=db_dir
    )

    return db


def load_vector_store(db_dir=DEFAULT_DB_DIR):

    return Chroma(
        persist_directory=db_dir,
        embedding_function=embedding_model
    )