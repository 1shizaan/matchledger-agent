// src/contexts/AuthContext.jsx
import { createContext, useContext, useState, useEffect, useCallback } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { jwtDecode } from 'jwt-decode';
import api, { setLogoutCallback } from '../utils/api';

const AuthContext = createContext();

export const AuthProvider = ({ children }) => {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);
  const navigate = useNavigate();
  const location = useLocation();

  const logout = useCallback(() => {
    console.log('Logging out user...');
    localStorage.removeItem('token');
    delete api.defaults.headers.common['Authorization'];
    setUser(null);

    // Only navigate to login if not already there
    if (location.pathname !== '/login') {
      navigate('/login', { replace: true });
    }
  }, [navigate, location.pathname]);

  // Set the logout callback for the API interceptor
  useEffect(() => {
    setLogoutCallback(logout);
  }, [logout]);

  const login = useCallback((token) => {
    return new Promise((resolve, reject) => {
      try {
        console.log('Processing login with token...');
        const decoded = jwtDecode(token);

        if (!decoded.sub || !decoded.user_id) {
          throw new Error("Invalid token structure");
        }

        // Check if token is expired
        if (decoded.exp * 1000 < Date.now()) {
          throw new Error("Token is expired");
        }

        localStorage.setItem('token', token);
        api.defaults.headers.common['Authorization'] = `Bearer ${token}`;

        const userData = {
          email: decoded.sub,
          id: decoded.user_id,
          name: decoded.name || '',
          isAdmin: decoded.is_admin || false
        };

        console.log('Setting user data:', userData);
        setUser(userData);

        // Use a longer timeout to ensure state is properly set
        setTimeout(() => {
          resolve(userData);
        }, 200);

      } catch (error) {
        console.error("Login error:", error);
        reject(error);
      }
    });
  }, []);

  // Initialize auth state
  useEffect(() => {
    console.log('Initializing auth state...');
    const token = localStorage.getItem('token');

    if (token) {
      try {
        const decoded = jwtDecode(token);

        if (decoded.exp * 1000 < Date.now()) {
          console.log('Token expired, logging out...');
          logout();
          setLoading(false);
          return;
        }

        // Set the Authorization header for Axios when the token is valid
        api.defaults.headers.common['Authorization'] = `Bearer ${token}`;

        const userData = {
          email: decoded.sub,
          id: decoded.user_id,
          name: decoded.name || '',
          isAdmin: decoded.is_admin || false
        };

        console.log('Restored user from token:', userData);
        setUser(userData);
      } catch (error) {
        console.error('Error restoring user from token:', error);
        logout();
      }
    } else {
      console.log('No token found');
    }

    setLoading(false);
  }, [logout]);

  const value = {
    user,
    loading,
    login,
    logout,
    isAuthenticated: !!user // Keep this for backward compatibility
  };

  return (
    <AuthContext.Provider value={value}>
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within AuthProvider');
  }
  return context;
};