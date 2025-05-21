import React from 'react';
import { useNavigate } from 'react-router-dom';
import { useGoogleLogin } from '@react-oauth/google';

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
    <div style={{ textAlign: 'center', paddingTop: 100 }}>
      <h1>Welcome to WanderMind 🌍</h1>
      <p>Personalized travel recommendations for you</p>
      <button onClick={() => login()} style={{ padding: '12px 24px', fontSize: 16 }}>
        Sign in with Google
      </button>
    </div>
  );
}
