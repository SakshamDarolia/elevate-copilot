import React, { useState, useRef, useEffect } from 'react'; // <-- Import useRef and useEffect
import axios from 'axios';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import './App.css';

function App() {
  const [prompt, setPrompt] = useState('');
  const [messages, setMessages] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  
  // Create a ref that we will attach to the bottom of the chat window
  const bottomOfChatRef = useRef(null);

  // This useEffect hook will run every time the `messages` array changes
  useEffect(() => {
    // Scroll to the bottom of the chat window smoothly
    bottomOfChatRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!prompt.trim() || isLoading) return;

    const userMessage = { role: 'user', content: prompt };
    setMessages((prevMessages) => [...prevMessages, userMessage]);
    setIsLoading(true);
    setPrompt('');

    try {
      const response = await axios.post('http://localhost:8000/ask', { prompt });
      const aiMessage = { role: 'assistant', content: response.data.answer };
      setMessages((prevMessages) => [...prevMessages, aiMessage]);
    } catch (error) {
      const errorMessage = { role: 'assistant', content: 'Sorry, something went wrong. Please try again.' };
      setMessages((prevMessages) => [...prevMessages, errorMessage]);
      console.error("Error fetching response:", error);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="app-container">
      <div className="header">
        <h1>Elevate Aviation Co-Pilot</h1>
      </div>
      <div className="chat-window">
        {messages.map((msg, index) => (
          <div key={index} className={`message ${msg.role}`}>
            {msg.role === 'user' ? (
              <p>{msg.content}</p>
            ) : (
              <ReactMarkdown remarkPlugins={[remarkGfm]}>
                {msg.content}
              </ReactMarkdown>
            )}
          </div>
        ))}
        {isLoading && <div className="message assistant"><p>Thinking...</p></div>}
        {/* This empty div is our target. We will always scroll to it. */}
        <div ref={bottomOfChatRef} />
      </div>
      <form className="chat-input-form" onSubmit={handleSubmit}>
        <input
          type="text"
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          placeholder="Ask about your trip..."
          disabled={isLoading}
        />
        <button type="submit" disabled={isLoading}>Send</button>
      </form>
    </div>
  );
}

export default App;