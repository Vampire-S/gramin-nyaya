import ollama
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# 1. Setup Data Pipeline
print("--- Initializing Legal Database... ---")
loader = PyPDFLoader("./legal_docs/registrationActHindi.pdf") 
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
chunks = text_splitter.split_documents(docs)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
vector_db = Chroma.from_documents(
    documents=chunks, 
    embedding=embeddings, 
    persist_directory="./chroma_db"
)

def ask_gramin_nyaya(user_query):
    # 1. Precise Search
    relevant_chunks = vector_db.similarity_search(user_query, k=2)
    context_text = "\n\n".join([chunk.page_content for chunk in relevant_chunks])
    
    # 2. The "Strict" Prompt
    # We tell the AI NOT to use its own knowledge, only the provided text.
    response = ollama.chat(model='qwen2.5:1.5b', messages=[
        {
            'role': 'system', 
            'content': f"आप एक कानूनी सहायक हैं। केवल दिए गए Context का उपयोग करके जवाब दें। यदि उत्तर Context में नहीं है, तो कहें 'क्षमा करें, यह जानकारी इस दस्तावेज़ में उपलब्ध नहीं है।'\n\nContext:\n{context_text}"
        },
        {'role': 'user', 'content': user_query}
    ], 
    # 3. Maximum Stability Parameters
    options={
        "temperature": 0.0,    # 0.0 makes it predictable and stops 'creative' guessing
        "repeat_penalty": 1.3, # Increased to 1.3 to forcefully break loops
        "top_k": 10,           # Extreme focus on the best words
        "top_p": 0.1,          # Very strict word selection
        "num_predict": 150     # Keeps the answer short and to the point
    })
    
    return response['message']['content']