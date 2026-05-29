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
    
    # 2. Step One: The Researcher (DeepSeek R1)
    # Analyzes context and extracts strict legal facts.
    print("--- Running DeepSeek-R1 (Researcher) ---")
    deepseek_prompt = f"""You are a precise legal researcher. 
Read the context below and extract the exact legal facts, requirements, and section numbers related to the user's query.
Do NOT use outside knowledge. Do NOT simplify language yet. Just extract the facts.
If the answer is not in the context, output: 'NOT_FOUND'.

Context:
{context_text}

Query: {user_query}
"""
    research_response = ollama.chat(model='deepseek-r1:1.5b', messages=[
        {'role': 'user', 'content': deepseek_prompt}
    ], 
    options={
        "temperature": 0.0,
        "keep_alive": 0  # CRITICAL for Jetson Nano: Unload model immediately after use
    })
    
    extracted_facts = research_response['message']['content']
    
    if "NOT_FOUND" in extracted_facts:
        return "क्षमा करें, यह जानकारी इस दस्तावेज़ में उपलब्ध नहीं है।"

    # 3. Step Two: The Communicator (Qwen 2.5)
    # Translates and simplifies the extracted facts into easy Hindi.
    print("--- Running Qwen 2.5 (Communicator) ---")
    qwen_prompt = f"""आप एक कानूनी सहायक हैं जो ग्रामीण लोगों को आसान भाषा में समझाते हैं।
नीचे दिए गए 'Legal Facts' को बहुत ही सरल, स्पष्ट और संवादात्मक हिंदी (Devanagari) में समझाएं।
यदि कोई धारा (Section) है, तो उसका उल्लेख करें लेकिन आसान शब्दों में।
अपनी तरफ से कोई नई कानूनी जानकारी न जोड़ें।

Legal Facts:
{extracted_facts}
"""
    final_response = ollama.chat(model='qwen2.5:1.5b', messages=[
        {'role': 'system', 'content': 'आप एक मददगार और सरल भाषा में बोलने वाले कानूनी सहायक हैं।'},
        {'role': 'user', 'content': qwen_prompt}
    ], 
    options={
        "temperature": 0.1,    
        "repeat_penalty": 1.2, 
        "top_k": 10,           
        "top_p": 0.1,          
        "keep_alive": 0        # CRITICAL for Jetson Nano: Unload model immediately after use
    })
    
    return final_response['message']['content']
