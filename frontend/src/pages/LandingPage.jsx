import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import './LandingPage.css';

export default function LandingPage() {
  const navigate = useNavigate();
  const [showAuthModal, setShowAuthModal] = useState(false);
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [isLoginMode, setIsLoginMode] = useState(true);
  const [error, setError] = useState('');

  const handleSubmit = async (e) => {
    e.preventDefault();
    const endpoint = isLoginMode ? '/login' : '/signup';

    try {
      const res = await fetch(`http://localhost:8000${endpoint}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email, password }),
      });

      const data = await res.json();

      if (!res.ok) {
        setError(data.detail || 'Authentication failed');
        return;
      }

      localStorage.setItem('userId', data.user_id);
      setShowAuthModal(false);
      navigate('/recommendations');
    } catch (err) {
      console.error(err);
      setError('Something went wrong.');
    }
  };

  return (
    <div className="landing-container">
      <video autoPlay loop muted playsInline className="background-video">
        <source src="/nature.mp4" type="video/mp4" />
        Your browser does not support the video tag.
      </video>

      <div className="video-overlay"></div>

      <div className="landing-content">
        <h1 className="landing-title">
          Welcome to <span className="brand-gradient">WanderMind</span> 🌍
        </h1>

        <p className="landing-subtitle">
          Your passport to the unseen. AI-powered journeys await.
        </p>

        <p className="landing-tagline">
          From Himalayan trails to Tokyo streets,<br />
          tell us who you are —<br />
          and we’ll tell you where to go.
        </p>

        <button
          className="google-signin-btn"
          onClick={() => {
            setShowAuthModal(true);
            setError('');
          }}
        >
          Login / Sign Up
        </button>

        <p className="landing-quote">
          <em>“Not all those who wander are lost — some just haven’t logged in yet.”</em>
        </p>
      </div>

      {/* Modal */}
      {showAuthModal && (
        <div className="modal-overlay" onClick={() => setShowAuthModal(false)}>
          <div className="modal-content" onClick={e => e.stopPropagation()}>
            <h2>{isLoginMode ? 'Login' : 'Sign Up'}</h2>

            <form onSubmit={handleSubmit} className="auth-form">
              <input
                type="email"
                placeholder="Email"
                value={email}
                required
                onChange={(e) => setEmail(e.target.value)}
              />
              <input
                type="password"
                placeholder="Password"
                value={password}
                required
                onChange={(e) => setPassword(e.target.value)}
              />
              {error && <p className="error-text">{error}</p>}
              <button type="submit" className="google-signin-btn">
                {isLoginMode ? 'Login' : 'Sign Up'}
              </button>
            </form>

            <p className="switch-auth-mode">
              {isLoginMode ? "Don't have an account?" : 'Already have an account?'}{' '}
              <span
                onClick={() => {
                  setIsLoginMode(!isLoginMode);
                  setError('');
                }}
                style={{ cursor: 'pointer', textDecoration: 'underline' }}
              >
                {isLoginMode ? 'Sign Up' : 'Login'}
              </span>
            </p>

            <button
              className="close-modal-btn"
              onClick={() => setShowAuthModal(false)}
              aria-label="Close Modal"
            >
              &times;
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
