// ~/reconbot/reconbot-frontend/src/main.jsx
import { createRoot } from 'react-dom/client';
import App from './App';
import { BrowserRouter } from 'react-router-dom'; // Keep this here for the top-level Router
import { AuthProvider } from './contexts/AuthContext'; // Keep this here for the top-level AuthProvider
import { GoogleOAuthProvider } from '@react-oauth/google';
import './index.css';

createRoot(document.getElementById('root')).render(
  <GoogleOAuthProvider clientId={import.meta.env.VITE_GOOGLE_CLIENT_ID}>
    {/* BrowserRouter should be at the top level to provide routing context */}
    <BrowserRouter>
      {/* AuthProvider should wrap App and any components that need auth context */}
      <AuthProvider>
        <App /> {/* App component will now only define and render routes */}
      </AuthProvider>
    </BrowserRouter>
  </GoogleOAuthProvider>
);