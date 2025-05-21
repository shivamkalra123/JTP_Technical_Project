import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';
import { GoogleOAuthProvider } from '@react-oauth/google';

ReactDOM.createRoot(document.getElementById('root')).render(
  <GoogleOAuthProvider clientId="823556860370-i7476nscqc44lcv54tvqe90onlhv4jca.apps.googleusercontent.com">
    <App />
  </GoogleOAuthProvider>
);
