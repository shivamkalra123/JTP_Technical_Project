import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import LandingPage from './pages/LandingPage';
import RecommendationPage from './pages/RecommendationPage';
import { GoogleOAuthProvider } from '@react-oauth/google';

const clientId = "823556860370-i7476nscqc44lcv54tvqe90onlhv4jca.apps.googleusercontent.com";

function App() {
  return (
    <GoogleOAuthProvider clientId={clientId}>
      <Router>
        <Routes>
          <Route path="/" element={<LandingPage />} />
          <Route path="/recommendations" element={<RecommendationPage />} />
        </Routes>
      </Router>
    </GoogleOAuthProvider>
  );
}

export default App;
