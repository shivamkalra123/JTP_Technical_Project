import React, { useState } from 'react';

export default function RecommendationApp() {
  const [userId, setUserId] = useState('');
  const [prompt, setPrompt] = useState('');
  const [recommendations, setRecommendations] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);

    try {
      const res = await fetch(`http://localhost:8000/recommend?user_id=${userId}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ user_id: userId, prompt }),
      });
      if (!res.ok) throw new Error('Failed to fetch recommendations');

      const data = await res.json();
      setRecommendations(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ maxWidth: 600, margin: 'auto', padding: 20 }}>
      <h1>Travel Recommendation</h1>
      <form onSubmit={handleSubmit}>
        <input
          type="text"
          value={userId}
          onChange={(e) => setUserId(e.target.value)}
          placeholder="Enter user ID"
          style={{ width: '100%', padding: 10, fontSize: 16, marginBottom: 10 }}
          required
        />
        <input
          type="text"
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          placeholder="Enter what you like (e.g., historical places)"
          style={{ width: '100%', padding: 10, fontSize: 16 }}
          required
        />
        <button type="submit" disabled={loading} style={{ marginTop: 10, padding: '10px 20px' }}>
          {loading ? 'Loading...' : 'Get Recommendations'}
        </button>
      </form>

      {error && <p style={{ color: 'red' }}>{error}</p>}

      <ul style={{ listStyleType: 'none', padding: 0 }}>
        {recommendations.map((rec) => (
          <li
            key={rec.name}
            style={{
              border: '1px solid #ccc',
              marginTop: 10,
              padding: 10,
              borderRadius: 5,
              backgroundColor: '#f9f9f9',
            }}
          >
            <h3>{rec.name}</h3>
            <p>
              {rec.city}, {rec.state}
            </p>
            <p>{rec.description}</p>
            <p>Rating: {rec.rating}</p>
          </li>
        ))}
      </ul>
    </div>
  );
}
