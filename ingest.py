#Block 0 - All imports :
#Library  - unstructured.io
#from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_community.document_loaders import PyPDFLoader

from langchain_community.document_loaders import DirectoryLoader

#Breaking in chunks Liabrary - RecursiveCharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter

#User Query Translator to vector Library  - HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

#Vector DB  - Chroma DB
from langchain_chroma import Chroma

#Block - 1 Loading Legal files (md) :
print("---Step 1: Loading Legal -pdfs/text/MD ")

folder_path = "./legal_data_folder"
loader = DirectoryLoader(
    folder_path,
    #glob = "**/*.md",                           #Only Select Markdown files
    #loader_cls = UnstructuredMarkdownLoader     #Only Read as Markdown files
    glob = "**/*.pdf",                           #Only Select PDF files
    loader_cls = PyPDFLoader     #Only Read as PDF files
)
documents = loader.load()

print(f"Success! Total {len(documents)} (MD/PDF/TXT)Documents are prepared. ")

#Block - 2 Smart Chunking :
print("Step 2: Smart chunking started...")

#Text Splitter setup : 
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,         #Approx 1000 characters per chunk
    chunk_overlap = 200,      #Overlap of 200 characters to maintain context
    separators = ["\n\n", "\n", ". ", " ", ""]      #Splits in order  -> Paragraphs -> Lines -> Sentences 
)

chunks = text_splitter.split_documents(documents)

print(f"Success ! Documents are chunked in total {len(chunks)}")

#Block - 3 Embedding and Vectorization into Chroma DB :
print("Step 3 : Loading Embeddings Model...")

# Setup Translator Model - HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(
    model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

print("Step 4: Chunks converted to vectors and stored in Chroma DB...")

vector_db = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db",

    collection_metadata={"hnsw:space": "cosine"}   #Using COSINE SIMILARITY for vector search
)

print("Success ! Chroma DB is ready for querying.")
print("Ingestion pipeline completed ! ")