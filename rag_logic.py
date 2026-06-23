import ollama
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# 1. Setup Data Pipeline
print("--- Connecting to Existing Legal Database... ---")

# We only LOAD the database here, we do not rebuild it!
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
vector_db = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)

def ask_gramin_nyaya(user_query):
    try:
        # 1. Precise Search (k=3 retrieved chunks)
        relevant_chunks = vector_db.similarity_search(user_query, k=3)
        
        if not relevant_chunks:
            return "क्षमा करें, यह जानकारी इस दस्तावेज़ में उपलब्ध नहीं है।"
            
        context_text = "\n\n".join([chunk.page_content for chunk in relevant_chunks])
        
        # 2. Single Step: Generate Answer with Llama 3.2
        print("--- Running Llama 3.2:1b (Legal Assistant) ---")
        
        llama_prompt = f"""You are Gramin-Nyaya, a legal assistant for rural India.
Read the legal context below and answer the question in simple Hindi.

Context:
{context_text}

Question: {user_query}

Hindi Answer:"""
        
        final_response = ollama.chat(model='llama3.2:1b', messages=[
            {'role': 'system', 'content': 'You are a legal assistant. You must answer in Hindi.'},
            {'role': 'user', 'content': llama_prompt}
        ], 
        options={
            "temperature": 0.1,    
            "repeat_penalty": 1.1, # Lowered from 1.25 to prevent grammar breaking
            "top_k": 10,           
            "top_p": 0.1,          
            "keep_alive": 0        # CRITICAL for Edge Devices
        })
        
        return final_response['message']['content']
        
    except Exception as e:
        print(f"\n[Error in RAG pipeline]: {e}")
        return "तकनीकी समस्या के कारण जवाब देने में असमर्थ हूँ।"