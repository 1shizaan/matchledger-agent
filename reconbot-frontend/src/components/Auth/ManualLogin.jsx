// src/components/Auth/ManualLogin.jsx
import React, { useState } from 'react';
import { useAuth } from '../../contexts/AuthContext';
import { useNavigate } from 'react-router-dom';
import api from '../../utils/api';

import { Input } from '../ui/input.tsx';
import { Button } from '../ui/button.tsx';
import { Label } from '../ui/label.tsx';

export default function ManualLogin() {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const { login } = useAuth();
  const navigate = useNavigate();

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    setLoading(true);

    try {
      console.log('Attempting manual login for:', email);
      const response = await api.post('/auth/login', { email, password });

      console.log('Manual login response:', response.data);

      if (response.data.access_token) {
        console.log('Manual login successful, processing token...');

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

        console.log('Manual login processed, navigating to dashboard...');

        // Force navigation with replace to avoid back button issues
        navigate('/dashboard', { replace: true });
      } else {
        setError('Login failed. Please check your credentials.');
      }
    } catch (err) {
      console.error('Manual login error:', err.response?.data || err.message);

      let errorMessage = 'Login failed. Please try again.';

      if (err.response?.status === 401) {
        errorMessage = 'Invalid email or password.';
      } else if (err.response?.status === 400) {
        errorMessage = err.response?.data?.detail || 'Invalid credentials.';
      } else if (err.response?.data?.detail) {
        errorMessage = err.response.data.detail;
      } else if (err.message) {
        errorMessage = err.message;
      }

      setError(errorMessage);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex flex-col gap-4 w-full max-w-sm mx-auto p-6 border rounded-lg shadow-md bg-white">
      <h2 className="text-2xl font-semibold text-center mb-4">Login with Email</h2>
      <form onSubmit={handleSubmit} className="flex flex-col gap-4">
        <div>
          <Label htmlFor="email">Email</Label>
          <Input
            id="email"
            type="email"
            placeholder="you@example.com"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            required
            className="mt-1"
            disabled={loading}
          />
        </div>
        <div>
          <Label htmlFor="password">Password</Label>
          <Input
            id="password"
            type="password"
            placeholder="********"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            required
            className="mt-1"
            disabled={loading}
          />
        </div>
        {error && (
          <div className="bg-red-50 border border-red-200 rounded-md p-3">
            <p className="text-red-700 text-sm text-center">{error}</p>
          </div>
        )}
        <Button type="submit" disabled={loading} className="w-full">
          {loading ? (
            <div className="flex items-center justify-center">
              <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
              Logging in...
            </div>
          ) : (
            'Login'
          )}
        </Button>
      </form>

      {/* Forgot Password Link */}
      <div className="text-center mt-2">
        <a
          href="/forgot-password"
          className="text-sm text-blue-600 hover:underline"
        >
          Forgot your password?
        </a>
      </div>
    </div>
  );
}