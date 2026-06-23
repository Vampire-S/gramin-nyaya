#Block 0 : Imports :
import sys
import ollama 
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

#NOTE : retrival file only used vector stored data, no Loader  or Splitter imports are required here.

#Block 1: Loading Chroma DB :
print("Step 1: Opening Chroma DB...")

#NOTE : Using same model as used in ingest.py for embedding, to avoid any mismatch in vector dimensions.
embeddings = HuggingFaceEmbeddings(
    model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

vector_db = Chroma(
    persist_directory="./Chroma_db",
    embedding_function=embeddings,

    collection_metadata={"hnsw:space":"cosine"} #Double check, using same COSINE SIMILARITY for vector search
)
print("Success! Chroma DB perfectly Loaded")

#print("middle man step - Offline Translation...")--------- this is under development
#Hindi to English model:
#model_name_hi_en = "Helsinki-NLP/opus-mt-hi-en"
#tokenizer_hi_en = MarianTokenizer.from_pretrained(model_name_hi_en)
#model_hi_en = MarianMTModel.from_pretrained(model_name_hi_en)

#English to Hindi model:
#model_name_en_hi = "Helsinki-NLP/opus-mt-en-hi"
#tokenizer_en_hi = MarianTokenizer.from_pretrained(model_name_en_hi)
#model_en_hi = MarianMTModel.from_pretrained(model_name_en_hi)
#print("Translators Loaded Successfully!")**/


#Block 2: User Query and Retrieval :
print("Step 2: User Query finding... and Retrieval...")

#NOTE : for testing purpose, we are using a hardcoded query. In real application, this can be taken as user input.
user_query = "1) Tell me the objective of PM KISsan smmaN NIDHI ?, 2) also TELL me objective of NALSA jagriti scheme ?"
print(f"User Query : {user_query}")

# Direct translation logic(this logic is not completed yet) --------- this is under development
#inputs_hi = tokenizer_hi_en(user_query_hindi, return_tensors="pt", padding=True)
#translated_tokens_en = model_hi_en.generate(**inputs_hi)
#user_query_eng = tokenizer_hi_en.decode(translated_tokens_en[0], skip_special_tokens=True)

#Retrieving top 4 relevant chunks from Chroma DB based on user query
retriever = vector_db.as_retriever(
    search_type = "similarity_score_threshold",
    search_kwargs = {
        "k": 4,
        "score_threshold" : 0.3
    }
)

relevant_chunks = retriever.invoke(user_query)

#Safty Check - (No Hallucination Rule) : if no relevant chunks found, we will return a safe message instead of generating an answer.
if not relevant_chunks:
    print("==================================================") 
    print("No relevant data found in documents, Query is out of source!")
    print("==================================================")
    context_text = "" #Empty context will lead to a safe answer from LLM, instead of hallucination.
    sys.exit()
else:
    print(f"Success! {len(relevant_chunks)} Relevant chunks found and retrieved.")

    print("\n"+"*"*40)
    print(" Selected Chunks are : ")
    print("*"*40)
    for i, chunk in enumerate(relevant_chunks):
        print(f"\n--CHUNK {i+1}--")
        print(chunk.page_content)
    print("*"*40 + "\n")

    #Conveting chunks to a single paragraph 
    context_text = "\n\n".join([chunk.page_content for chunk in relevant_chunks])

#Block 3: LLM Output Generation :

#FOR LLM to not make mistakes and not Break Rules :
system_prompt = """Answer the question using only the Context below.
your role is legal advisor for rural peoples.
Find the paragraph or sentence that answers it and state it clearly. 
if you don't know the exact answer then give exactly same paragraph or line which relevent to user query in simple words don't think for it.
unless until no information available say 'sorry nothing found'.
"""

print("Step 3: (Llama 3.2) Generating Answers...")

final_prompt = f"Context:\n{context_text}\n\nQuestion: {user_query}"

#TRY EXCEPT BLOCK for not to crash ...
try : 

    import ollama
    
    response = ollama.chat(
        model='llama3.2:1b', 
        messages=[
            #Forwarding both Context and User Query to LLM in a single message, to maintain the connection between them and avoid any confusion for LLM.
            {'role': 'user', 'content': final_prompt}
            
        ],
        options={
            "temperature": 0.0,         #0.0 for zero creativity, Purely Leagal Facts
            "repeat_penalty": 1.1,      #To avoid repetition in LLM output, especially when context is long and may have similar sentences.
            #"top_k": 10,
            #"top_p":0.1,
            #"keep_alive": 0             #No need to keep the session alive, as we are doing a single query-response interaction. Setting it to 0 will free up resources immediately after response is generated.
        }
    )
    english_answer = response["message"]["content"]

    #Final Output Display :
    print("\n" + "="*50)
    print("According to Gramin Nyaya here is your solution : ")
    print("="*50)
    print(response['message']['content'])
    print("="*50)

except Exception as e:
    #for code not to crash, if ollama closed in background,
    print(f"\n[Error]: There is an issues while connecting with AI Model :  {e}")
    print("Please make sure Ollama is curently running in backgorund or not. ")
    sys.exit()



        


