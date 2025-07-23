// src/components/Auth/GoogleLogin.jsx
import { useAuth } from '../../contexts/AuthContext';
import { GoogleLogin } from '@react-oauth/google';
import { useNavigate } from 'react-router-dom';
import { useState } from 'react';
import api from '../../utils/api';

// Helper function to decode JWT (for debugging only)
const decodeJWT = (token) => {
  try {
    const payload = token.split('.')[1];
    const decoded = JSON.parse(atob(payload));
    return decoded;
  } catch (error) {
    console.error('Failed to decode JWT:', error);
    return null;
  }
};

export default function GoogleAuth() {
  const { login } = useAuth();
  const navigate = useNavigate();
  const [isLoading, setIsLoading] = useState(false);

  const handleSuccess = async (credentialResponse) => {
    if (isLoading) return; // Prevent multiple simultaneous requests

    try {
      setIsLoading(true);
      console.log('Google credential response:', credentialResponse);

      if (!credentialResponse.credential) {
        throw new Error('No credential received from Google');
      }

      const decoded = decodeJWT(credentialResponse.credential);
      console.log('Decoded JWT payload:', decoded);
      console.log('Token audience (aud):', decoded?.aud);
      console.log('Token issuer (iss):', decoded?.iss);

      console.log('Sending request to backend...');
      console.log('Request payload:', { token: credentialResponse.credential });

      const response = await api.post('/auth/google', {
        token: credentialResponse.credential
      });

      console.log('Backend response:', response.data);

      if (!response.data.access_token) {
        throw new Error('No access token received from backend');
      }

      // Call login and wait for it to complete
      await new Promise((resolve, reject) => {
        try {
          login(response.data.access_token);
          // Give a small delay to ensure state is updated
          setTimeout(resolve, 100);
        } catch (error) {
          reject(error);
        }
      });

      console.log('Login successful, navigating to dashboard...');

      // Force navigation with replace to avoid back button issues
      navigate('/dashboard', { replace: true });

    } catch (error) {
      console.error('Google auth failed:', error);
      console.error('Error response:', error.response);
      console.error('Error status:', error.response?.status);
      console.error('Error data:', error.response?.data);

      let errorMessage = 'Google login failed';

      if (error.response?.status === 400) {
        errorMessage = error.response?.data?.detail || 'Invalid token. Please try again.';
      } else if (error.response?.status === 403) {
        errorMessage = 'Access denied. Please check your credentials.';
      } else if (error.response?.data?.message) {
        errorMessage = error.response.data.message;
      } else if (error.message) {
        errorMessage = error.message;
      }

      alert(`Google login failed: ${errorMessage}`);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex justify-center my-4">
      {isLoading && (
        <div className="flex items-center justify-center mb-4">
          <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-blue-600"></div>
          <span className="ml-2 text-sm text-gray-600">Signing in...</span>
        </div>
      )}
      <GoogleLogin
        onSuccess={handleSuccess}
        onError={(error) => {
          console.log('Google login error:', error);
          alert('Google login failed. Please try again.');
          setIsLoading(false);
        }}
        useOneTap={false}
        theme="filled_blue"
        size="large"
        text="continue_with"
        shape="rectangular"
        disabled={isLoading}
      />
    </div>
  );
}