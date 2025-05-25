import React from 'react';
import { useNavigate } from 'react-router-dom';
import { useGoogleLogin } from '@react-oauth/google';
import './LandingPage.css';
import { FcGoogle } from 'react-icons/fc';

export default function LandingPage() {
  const navigate = useNavigate();

  const login = useGoogleLogin({
    onSuccess: async tokenResponse => {
      const res = await fetch('https://www.googleapis.com/oauth2/v3/userinfo', {
        headers: { Authorization: `Bearer ${tokenResponse.access_token}` },
      });
      const profile = await res.json();

      localStorage.setItem('userId', profile.sub);
      navigate('/recommendations');
    },
    onError: () => alert('Login Failed'),
  });

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

        <button onClick={login} className="google-signin-btn">
  <img
    src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/google/google-original.svg"
    alt="Google"
    className="google-icon"
  />
  <span>Begin Your Journey with Google</span>
</button>





        <p className="landing-quote">
          <em>“Not all those who wander are lost — some just haven’t logged in yet.”</em>
        </p>
      </div>

   
    </div>
  );
}
