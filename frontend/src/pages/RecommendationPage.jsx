import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import './Recommendation.css';

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
    setPrompt('');
    try {
      const res = await fetch(`http://localhost:8000/recommend/inferred?user_id=${userId}`);
      const data = await res.json();
      console.log('Inferred recommendations:', data);

      if (Array.isArray(data)) {
        setRecommendations(data);
        setMessage(data.length > 0
          ? 'Based on your preferences, you might like:'
          : "Start searching so we can recommend places you'll love!");
      } else {
        setRecommendations([]);
        setMessage("Start searching so we can recommend places you'll love!");
      }
    } catch (err) {
      console.error("Error fetching inferred recommendations:", err);
      setRecommendations([]);
      setMessage("Something went wrong. Try again later.");
    }
    setLoading(false);
  };

  const fetchRecommendations = async () => {
    if (!prompt.trim()) return;
    setLoading(true);
    try {
      const res = await fetch(`http://localhost:8000/recommend?user_id=${userId}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt }),
      });
      const data = await res.json();
      console.log('Recommendations:', data);

      if (Array.isArray(data)) {
        setRecommendations(data);
        setMessage("Here’s what we found based on your interest:");
      } else {
        setRecommendations([]);
        setMessage("Sorry, no recommendations found.");
      }
    } catch (err) {
      console.error("Error fetching recommendations:", err);
      setRecommendations([]);
      setMessage("Something went wrong. Try again later.");
    }
    setLoading(false);
  };

  const handleLogout = () => {
    localStorage.removeItem('userId');
    setUserId(null);
    navigate('/');
  };

  return (
    <div className="container">
      <div className="header">
        <h2
          className="title"
          onClick={fetchInferredRecommendations}
          tabIndex={0}
          onKeyDown={e => { if (e.key === 'Enter') fetchInferredRecommendations() }}
        >
          Hi, Wanderer! 🧭
        </h2>
        <button
          className="logout-button"
          onClick={handleLogout}
        >
          Logout
        </button>
      </div>

      <input
        className="prompt-input"
        value={prompt}
        onChange={(e) => setPrompt(e.target.value)}
        placeholder="What type of places do you like?"
      />
      <button
        className="recommend-button"
        onClick={fetchRecommendations}
        disabled={loading}
      >
        {loading ? 'Thinking...' : 'Get Recommendations'}
      </button>

      {message && <p className="message">{message}</p>}

      <ul className="recommendations-list">
        {Array.isArray(recommendations) && recommendations.length > 0 ? (
          recommendations.map((r, idx) => (
            <li key={idx} className="recommendation-item">
              <h3>{r.name}</h3>
              <p className="italic">{r.city}, {r.state}</p>
              <p>{r.description}</p>
              <p><strong>Rating:</strong> {r.rating}</p>
            </li>
          ))
        ) : (
          <p>No recommendations available.</p>
        )}
      </ul>
    </div>
  );
}
