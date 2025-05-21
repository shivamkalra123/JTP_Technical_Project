import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';

export default function RecommendationPage() {
  const navigate = useNavigate();
  const [userId, setUserId] = useState(localStorage.getItem('userId'));
  const [prompt, setPrompt] = useState('');
  const [recommendations, setRecommendations] = useState([]);
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState('');

  useEffect(() => {
    if (!userId) {
      navigate('/');
    } else {
      fetchInferredRecommendations();
    }
  }, [userId]);

  const fetchInferredRecommendations = async () => {
    setLoading(true);
    setPrompt(''); // Clear the prompt
    try {
      const res = await fetch(`http://localhost:8000/recommend/inferred?user_id=${userId}`);
      const data = await res.json();

      if (Array.isArray(data) && data.length > 0) {
        setRecommendations(data);
        setMessage('Based on your preferences, you might like:');
      } else {
        setRecommendations([]);
        setMessage("Start searching so we can recommend places you'll love!");
      }
    } catch (err) {
      console.error("Error fetching inferred recommendations:", err);
      setMessage("Something went wrong. Try again later.");
    }
    setLoading(false);
  };

  const fetchRecommendations = async () => {
    if (!prompt.trim()) return;
    setLoading(true);
    const res = await fetch(`http://localhost:8000/recommend?user_id=${userId}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ prompt }),
    });
    const data = await res.json();
    setRecommendations(data);
    setMessage("Here’s what we found based on your interest:");
    setLoading(false);
  };

  const handleLogout = () => {
    localStorage.removeItem('userId');
    setUserId(null);
    navigate('/');
  };

  return (
    <div style={{ maxWidth: 600, margin: 'auto', padding: 20 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <h2
          style={{ cursor: 'pointer', margin: 0 }}
          onClick={fetchInferredRecommendations}
        >
          Hi, Wanderer! 🧭
        </h2>
        <button onClick={handleLogout}>Logout</button>
      </div>

      <input
        value={prompt}
        onChange={(e) => setPrompt(e.target.value)}
        placeholder="What type of places do you like?"
        style={{ width: '100%', padding: 10, fontSize: 16, marginTop: 20 }}
      />
      <button onClick={fetchRecommendations} style={{ marginTop: 10, padding: '10px 20px' }}>
        {loading ? 'Thinking...' : 'Get Recommendations'}
      </button>

      {message && <p style={{ marginTop: 20, fontWeight: 'bold' }}>{message}</p>}

      <ul>
        {recommendations.map((r, idx) => (
          <li key={idx} style={{ margin: '20px 0', borderBottom: '1px solid #ccc', paddingBottom: 10 }}>
            <h3>{r.name}</h3>
            <p>{r.city}, {r.state}</p>
            <p>{r.description}</p>
            <p><strong>Rating:</strong> {r.rating}</p>
          </li>
        ))}
      </ul>
    </div>
  );
}
