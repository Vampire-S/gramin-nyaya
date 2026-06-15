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
        # 1. Precise Search
        relevant_chunks = vector_db.similarity_search(user_query, k=2)
        
        if not relevant_chunks:
            return "क्षमा करें, यह जानकारी इस दस्तावेज़ में उपलब्ध नहीं है।"
            
        context_text = "\n\n".join([chunk.page_content for chunk in relevant_chunks])
        
        # 2. Single Step: Generate Answer with Llama 3.2
        print("--- Running Llama 3.2:1b (Legal Assistant) ---")
        llama_prompt = f"""आप एक कानूनी सहायक हैं जो ग्रामीण लोगों को आसान भाषा में समझाते हैं।
        नीचे दिए गए English 'Context' को पढ़ें और उपयोगकर्ता के 'Query' का सटीक उत्तर Devanagari Hindi में दें।
        
        RULES:
        1. Read the English context carefully, extract the facts, and translate them into simple, clear Hindi.
        2. Do not change or alter any specific numbers, sections, or monetary values (like 100 rupees or 1 year).
        3. If the answer is not in the context, strictly state: "यह जानकारी उपलब्ध नहीं है।"

        Context (In English):
        {context_text}

        Query (In Hindi): {user_query}
        """
        
        final_response = ollama.chat(model='llama3.2:1b', messages=[
            {'role': 'system', 'content': 'आप एक मददगार और सरल भाषा में बोलने वाले कानूनी सहायक हैं।'},
            {'role': 'user', 'content': llama_prompt}
        ], 
        options={
            "temperature": 0.1,    
            "repeat_penalty": 1.2, 
            "top_k": 10,           
            "top_p": 0.1,          
            "keep_alive": 0        # CRITICAL for Edge Devices
        })
        
        return final_response['message']['content']
        
    except Exception as e:
        print(f"\n[Error in RAG pipeline]: {e}")
        return "तकनीकी समस्या के कारण जवाब देने में असमर्थ हूँ।"