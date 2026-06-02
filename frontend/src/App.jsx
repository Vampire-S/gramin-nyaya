import { useState } from 'react';
import ReactMarkdown from 'react-markdown';
import './App.css';

function App() {
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState("");
  const [loading, setLoading] = useState(false);

  const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

  const askAI = async () => {
    if (!question.trim()) return;
    
    setLoading(true);
    setAnswer(""); 

    try {
      const response = await fetch(`${API_URL}/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      setAnswer(data.answer);
      
    } catch (error) {
      console.error("Fetch error:", error);
      setAnswer("Sorry, there is a problem connecting to the server. Please ensure the backend is running.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-container">
      {/* Header */}
      <header className="header">
        <h1>ग्रामीण न्याय (Gramin-Nyaya)</h1>
        <p>आपका डिजिटल कानूनी सहायक</p>
      </header>

      {/* Main Card */}
      <main className="main-card">
        <label className="input-label">
          अपना सवाल यहाँ लिखें:
        </label>
        
        <textarea 
          className="text-area"
          placeholder="जैसे: रजिस्ट्री के लिए क्या चाहिए?"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
        />

        <button 
          onClick={askAI}
          disabled={loading || !question.trim()} 
          className="submit-btn"
        >
          {loading ? "सोच रहा हूँ..." : "जवाब जानें"}
        </button>

        {/* Answer Section */}
        {answer && (
          <div className="answer-box" aria-live="polite">
            <h2 className="answer-title">⚖️ कानूनी परामर्श:</h2>
            <div className="answer-content">
              <ReactMarkdown>{answer}</ReactMarkdown>
            </div>
          </div>
        )}
      </main>

      <footer className="footer">
        © 2026 ग्रामीण न्याय परियोजना | कानूनी जानकारी के लिए सुरक्षित
      </footer>
    </div>
  );
}

export default App;