// src/components/Auth/Register.jsx
import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import api from '../../utils/api';

import { Input } from '../ui/input.tsx';
import { Button } from '../ui/button.tsx';
import { Label } from '../ui/label.tsx';

export default function Register() {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [fullName, setFullName] = useState('');
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();

  const validateForm = () => {
    if (!email || !password || !confirmPassword) {
      setError('Please fill in all required fields.');
      return false;
    }

    if (password.length < 6) {
      setError('Password must be at least 6 characters long.');
      return false;
    }

    if (password !== confirmPassword) {
      setError('Passwords do not match.');
      return false;
    }

    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!emailRegex.test(email)) {
      setError('Please enter a valid email address.');
      return false;
    }

    return true;
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    setSuccess('');
    setLoading(true);

    if (!validateForm()) {
      setLoading(false);
      return;
    }

    try {
      console.log('Attempting registration for:', email);

      // Send registration data to your backend /auth/register endpoint
      const response = await api.post('/auth/register', {
        email,
        password,
        full_name: fullName || null, // Send null if empty, backend might handle it better
      });

      console.log('Registration response:', response.data);

      if (response.status === 201 || response.status === 200) {
        setSuccess('Registration successful! You can now sign in with your credentials.');

        // Clear form
        setEmail('');
        setPassword('');
        setConfirmPassword('');
        setFullName('');

        // Auto-switch to login mode after 2 seconds
        setTimeout(() => {
          // If you're using the toggle in Login.jsx, you might want to pass a callback
          // For now, we'll just show the success message
          window.location.reload(); // Simple way to refresh and show login
        }, 2000);

      } else {
        setError(response.data?.detail || 'Registration failed. Please try again.');
      }
    } catch (err) {
      console.error('Registration error:', err.response?.data || err.message);

      let errorMessage = 'Registration failed. Please try again.';

      if (err.response?.status === 400) {
        errorMessage = err.response?.data?.detail || 'Invalid registration data.';
      } else if (err.response?.status === 409) {
        errorMessage = 'An account with this email already exists.';
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
      <h2 className="text-2xl font-semibold text-center mb-4">Create Account</h2>

      {success && (
        <div className="bg-green-50 border border-green-200 rounded-md p-3 mb-4">
          <p className="text-green-700 text-sm text-center">{success}</p>
        </div>
      )}

      <form onSubmit={handleSubmit} className="flex flex-col gap-4">
        <div>
          <Label htmlFor="fullName">Full Name (Optional)</Label>
          <Input
            id="fullName"
            type="text"
            placeholder="John Doe"
            value={fullName}
            onChange={(e) => setFullName(e.target.value)}
            className="mt-1"
            disabled={loading}
          />
        </div>
        <div>
          <Label htmlFor="email">Email *</Label>
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
          <Label htmlFor="password">Password *</Label>
          <Input
            id="password"
            type="password"
            placeholder="At least 6 characters"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            required
            className="mt-1"
            disabled={loading}
          />
        </div>
        <div>
          <Label htmlFor="confirmPassword">Confirm Password *</Label>
          <Input
            id="confirmPassword"
            type="password"
            placeholder="Repeat your password"
            value={confirmPassword}
            onChange={(e) => setConfirmPassword(e.target.value)}
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
              Creating Account...
            </div>
          ) : (
            'Create Account'
          )}
        </Button>
      </form>
    </div>
  );
}