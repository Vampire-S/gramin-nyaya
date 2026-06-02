import os
import shutil
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# --- 1. Setup Paths ---
CHROMA_PATH = "./chroma_db"
DOCUMENT_PATH = "./legal_docs/registrationActEnglish.txt"

# --- 2. Clear Old Database ---
print("🧹 Clearing old database...")
if os.path.exists(CHROMA_PATH):
    shutil.rmtree(CHROMA_PATH)

# --- 3. Load Clean English Text ---
print(f"📄 Loading clean English document: {DOCUMENT_PATH}")
try:
    loader = TextLoader(DOCUMENT_PATH, encoding="utf-8")
    docs = loader.load()
except FileNotFoundError:
    print(f"❌ Error: Could not find the file at {DOCUMENT_PATH}")
    print("Please make sure you created the 'registrationActEnglish.txt' file in the 'legal_docs' folder!")
    exit()

# --- 4. Split Text ---
print("✂️ Splitting text into highly focused chunks...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400, 
    chunk_overlap=50
)
chunks = text_splitter.split_documents(docs)
print(f"✅ Created {len(chunks)} chunks.")

# --- 5. Generate Embeddings & Save ---
print("🧠 Generating embeddings and saving to Chroma DB...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

vector_db = Chroma.from_documents(
    documents=chunks, 
    embedding=embeddings, 
    persist_directory=CHROMA_PATH
)

print("🚀 Ingestion Complete! Clean English vector database is ready.")