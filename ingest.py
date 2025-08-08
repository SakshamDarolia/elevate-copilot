import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings

# Define the path to your documents and the local database
DOCS_PATH = "documents"
DB_PATH = "vector_db"

def create_vector_database():
    """
    Creates a ChromaDB vector database from documents in the specified directory.
    """
    # Use DirectoryLoader to load all .txt files
    loader = DirectoryLoader(DOCS_PATH, glob="*.txt", loader_cls=TextLoader)
    documents = loader.load()

    # Split the documents into smaller chunks for better processing
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)

    print(f"Loaded {len(documents)} documents and split them into {len(texts)} chunks.")

    # Use a free, high-quality sentence transformer model for creating embeddings
    # This model runs locally on your machine
    embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    # Create the Chroma vector store, persisting it to disk
    db = Chroma.from_documents(
        texts,
        embedding_function,
        persist_directory=DB_PATH
    )
    print(f"Successfully created and saved the vector database to {DB_PATH}")

if __name__ == "__main__":
    create_vector_database()