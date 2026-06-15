import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# 1. Configuration - Pointing exactly to your file structure
FILE_PATH = "./legal_docs/registrationActEnglish.txt"
DB_DIR = "./chroma_db"

def ingest_data():
    print(f"\n--- Starting Ingestion Process ---")
    
    # 2. Load the clean English Text Document
    print(f"Loading document from: {FILE_PATH}")
    try:
        loader = TextLoader(FILE_PATH, encoding="utf-8")
        documents = loader.load()
        print("Document loaded successfully.")
    except Exception as e:
        print(f"[Error] Could not load document. Make sure the file exists! Details: {e}")
        return

    # 3. Split the Text into Chunks
    print("Splitting text into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,       # Smaller chunks are better for the 1B Jetson model
        chunk_overlap=150,    # Overlap ensures sentences aren't cut off awkwardly
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} text chunks.")

    # 4. Initialize Embeddings (Matching your rag_logic.py exactly)
    print("Initializing embedding model...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    # 5. Create and Save the Vector Database
    print("Building Chroma vector database. This may take a moment...")
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_DIR
    )
    
    print(f"--- Success! Database built and safely saved to {DB_DIR} ---")

if __name__ == "__main__":
    ingest_data()