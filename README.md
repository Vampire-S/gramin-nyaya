# ⚖️ Gramin-Nyaya: AI Legal Assistant

**Gramin-Nyaya** (Rural Justice) is an offline, privacy-first AI assistant designed to bridge the legal information gap for rural populations in India. Built specifically to run on constrained edge hardware (NVIDIA Jetson), the system provides grounded, simplified legal guidance from the **Registration Act, 1908** in high-quality Hindi.

---

## 🚀 Key Features

* **Edge-Optimized Pipeline:** Powered by **Llama 3.2 (1B)** via Ollama. The architecture is strictly optimized to run efficiently on low-memory edge devices without crashing.
* **100% Offline & Private:** Runs entirely on local hardware. No audio or text data ever leaves the device—crucial for maintaining absolute confidentiality in legal matters.
* **"English-in, Hindi-out" RAG:** Uses a highly accurate cross-lingual retrieval strategy. The AI searches clean English text to maintain flawless logical accuracy, but is prompted to translate and deliver its final response in simple Devanagari Hindi to prevent "token explosion."
* **Voice-Ready:** Integrated with **Faster-Whisper (Small)** for highly accurate Hindi speech-to-text, allowing users to ask questions naturally via voice input.
* **Hallucination Control:** Uses LangChain and ChromaDB to ensure every answer is strictly grounded in official legal documents. If the law isn't in the text, the AI safely refuses to answer.

---

## 🤖 The Architecture

To maximize reasoning on a 1B parameter model, Gramin-Nyaya avoids messy PDFs and complex native Hindi tokenization. Instead, it uses a **Cross-Lingual Workflow**:

1.  **Ingestion:** The legal document (`registrationActEnglish.txt`) is chunked and embedded into a local ChromaDB vector store.
2.  **Retrieval:** User queries (in Hindi) are mapped to the highly accurate English legal chunks.
3.  **Generation:** Llama 3.2:1b reads the retrieved English facts, extracts the exact legal metrics (e.g., limits like "100 rupees" or "4 months"), and safely formats the final output into simplified Hindi for the user.

---

## 🛠️ Tech Stack & Libraries

This project uses a lean, bloat-free stack designed for edge deployment:

* **Core Logic:** Python 3.10+
* **LLM Engine:** `ollama` (Running `llama3.2:1b`)
* **RAG Framework:** `langchain`, `langchain-community`
* **Vector Database:** `chromadb`
* **Embeddings:** `langchain-huggingface`, `sentence-transformers` (Model: *paraphrase-multilingual-MiniLM-L12-v2*)
* **Voice / STT:** `faster-whisper`, `pyaudio`
* **Web Server / API:** `fastapi`, `uvicorn`
* **Utilities:** `pypdf` (For legacy document conversion)

---

## 📂 Project Structure

```text
Gramin-Nyaya/
├── legal_docs/             # Clean text source files (registrationActEnglish.txt)
├── chroma_db/              # Local vector database for semantic search
├── api.py                  # FastAPI server connecting backend to frontend
├── rag_logic.py            # LangChain RAG pipeline & LLM prompting logic
├── stt_service.py          # Voice-to-Text transcription service
├── gramin_nyaya_main.py    # Main CLI-based controller and testing loop
├── ingest_docs.py          # Script to chunk and embed documents into ChromaDB
├── requirements.txt        # Exact Python dependencies
└── index.html              # Village-friendly Web Interface
