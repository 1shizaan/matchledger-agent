// src/contexts/AuthContext.jsx - Enhanced Production Version
import { createContext, useContext, useState, useEffect, useCallback } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { jwtDecode } from 'jwt-decode';
import api, { setLogoutCallback, googleAuth, getUserProfile } from '../utils/api';

const AuthContext = createContext();

// Configuration constants
const TOKEN_REFRESH_BUFFER = 300; // 5 minutes in seconds
const TOKEN_EXPIRY_CHECK_BUFFER = 600; // 10 minutes in seconds
const TOKEN_CHECK_INTERVAL = 5 * 60 * 1000; // 5 minutes in milliseconds
const LOGIN_STATE_DELAY = 100; // Small delay for state setting

const PUBLIC_ROUTES = [
  '/login',
  '/register',
  '/forgot-password',
  '/reset-password',
  '/privacy-policy',
  '/terms-of-service'
];

export const AuthProvider = ({ children }) => {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);
  const navigate = useNavigate();
  const location = useLocation();

  // Enhanced logout function with comprehensive cleanup
  const logout = useCallback(() => {
    try {
      // Clear all authentication data from localStorage
      const keysToRemove = ['token', 'google_token', 'user_profile'];
      keysToRemove.forEach(key => localStorage.removeItem(key));

      // Clear API authorization header
      if (api.defaults?.headers?.common) {
        delete api.defaults.headers.common['Authorization'];
      }

      // Clear user state
      setUser(null);

      // Navigate to login if not on public route
      if (!PUBLIC_ROUTES.includes(location.pathname)) {
        navigate('/login', { replace: true });
      }
    } catch (error) {
      console.error('Logout cleanup failed:', error);
      // Force navigation even if cleanup fails
      setUser(null);
      navigate('/login', { replace: true });
    }
  }, [navigate, location.pathname]);

  // Set logout callback for API interceptor
  useEffect(() => {
    setLogoutCallback(logout);
  }, [logout]);

  // Enhanced user profile fetching with fallback handling
  const fetchUserProfile = useCallback(async (jwtToken) => {
  try {
    console.log('🔍 API DEBUG - Fetching user profile from API...');

    // First try to get profile from API
    const profileResponse = await getUserProfile();

    console.log('🔍 API DEBUG - Raw profile response:', profileResponse);

    let userData;
    if (profileResponse?.data) {
      userData = profileResponse.data;
    } else if (profileResponse) {
      userData = profileResponse;
    } else {
      throw new Error('No profile data received');
    }

    console.log('🔍 API DEBUG - Extracted userData:', userData);
    console.log('🔍 API DEBUG - userData.isAdmin:', userData.isAdmin);
    console.log('🔍 API DEBUG - userData.is_admin:', userData.is_admin);

    // ✅ FIXED: Better admin field handling for API response
    const isAdminFromAPI = userData.isAdmin === true || userData.is_admin === true;

    console.log('🔍 API DEBUG - Final isAdminFromAPI:', isAdminFromAPI);

    const result = {
      email: userData.email,
      id: userData.id,
      name: userData.name || userData.full_name || '',
      isAdmin: isAdminFromAPI,
      is_admin: isAdminFromAPI,
      is_beta_user: userData.is_beta_user === true,
      tokenExpiry: userData.tokenExpiry || jwtDecode(jwtToken).exp
    };

    console.log('🔍 API DEBUG - Final user object being returned:', result);
    return result;

  } catch (error) {
    console.warn('Profile fetch failed, using JWT fallback:', error.message);

    // Fallback to JWT data
    try {
      const decoded = jwtDecode(jwtToken);

      // ✅ CRITICAL FIX: Properly extract admin from JWT
      console.log('🔍 JWT DEBUG - Full decoded token:', decoded);
      console.log('🔍 JWT DEBUG - is_admin value:', decoded.is_admin);
      console.log('🔍 JWT DEBUG - typeof is_admin:', typeof decoded.is_admin);

      const isAdminFromJWT = decoded.is_admin === true;

      console.log('🔍 JWT DEBUG - Final isAdmin value:', isAdminFromJWT);

      return {
        email: decoded.sub,
        id: decoded.user_id,
        name: decoded.name || decoded.full_name || '',
        isAdmin: isAdminFromJWT,
        is_admin: isAdminFromJWT,
        is_beta_user: decoded.is_beta_user === true,
        tokenExpiry: decoded.exp
      };
    } catch (jwtError) {
      throw new Error('Failed to decode JWT token');
    }
  }
}, []);

  // Enhanced login function with comprehensive error handling
  const login = useCallback((token, isGoogleToken = false) => {
    return new Promise(async (resolve, reject) => {
      try {
        let jwtToken = token;

        // Handle Google token exchange
        if (isGoogleToken) {
          try {
            const authResult = await googleAuth(token);
            if (!authResult?.access_token) {
              throw new Error('No access token received from Google auth');
            }
            jwtToken = authResult.access_token;
          } catch (error) {
            reject(new Error(`Google authentication failed: ${error.message}`));
            return;
          }
        }

        // Validate JWT token structure and expiry
        let decoded;
        try {
          decoded = jwtDecode(jwtToken);
        } catch (error) {
          throw new Error('Invalid JWT token format');
        }

        if (!decoded.sub || !decoded.user_id) {
          throw new Error('Invalid token structure - missing required fields');
        }

        // Check token expiry with buffer
        const currentTime = Date.now() / 1000;
        if (decoded.exp < (currentTime + TOKEN_REFRESH_BUFFER)) {
          throw new Error('Token is expired or expires too soon');
        }

        // Store tokens
        localStorage.setItem('token', jwtToken);
        if (isGoogleToken) {
          localStorage.setItem('google_token', token);
        }

        // Set API authorization header
        api.defaults.headers.common['Authorization'] = `Bearer ${jwtToken}`;

        // Fetch and set user profile
        const userData = await fetchUserProfile(jwtToken);

        // Cache user profile
        localStorage.setItem('user_profile', JSON.stringify(userData));
        setUser(userData);

        // Small delay to ensure state is set
        setTimeout(() => resolve(userData), LOGIN_STATE_DELAY);

      } catch (error) {
        // Comprehensive cleanup on failure
        const keysToRemove = ['token', 'google_token', 'user_profile'];
        keysToRemove.forEach(key => localStorage.removeItem(key));

        if (api.defaults?.headers?.common) {
          delete api.defaults.headers.common['Authorization'];
        }

        reject(new Error(`Login failed: ${error.message}`));
      }
    });
  }, [fetchUserProfile]);

  // Enhanced token validation
  const isTokenValid = useCallback((token) => {
    if (!token || typeof token !== 'string') return false;

    try {
      const decoded = jwtDecode(token);
      const currentTime = Date.now() / 1000;

      // Check if token has required fields
      if (!decoded.sub || !decoded.user_id || !decoded.exp) {
        return false;
      }

      // Check expiry with buffer
      return decoded.exp > (currentTime + TOKEN_REFRESH_BUFFER);
    } catch (error) {
      return false;
    }
  }, []);

  // Enhanced token refresh with better error handling
  const refreshToken = useCallback(async () => {
    const googleToken = localStorage.getItem('google_token');

    if (!googleToken) {
      return false;
    }

    try {
      const authResult = await googleAuth(googleToken);
      if (!authResult?.access_token) {
        throw new Error('No access token received');
      }

      const newJwtToken = authResult.access_token;

      // Update stored token and API header
      localStorage.setItem('token', newJwtToken);
      api.defaults.headers.common['Authorization'] = `Bearer ${newJwtToken}`;

      // Fetch updated user profile
      const updatedUserData = await fetchUserProfile(newJwtToken);

      // Update cached profile and state
      localStorage.setItem('user_profile', JSON.stringify(updatedUserData));
      setUser(updatedUserData);

      return true;
    } catch (error) {
      console.error('Token refresh failed:', error.message);
      logout();
      return false;
    }
  }, [logout, fetchUserProfile]);

  // Profile refresh function
  const refreshProfile = useCallback(async () => {
    const token = localStorage.getItem('token');
    if (!token || !isTokenValid(token)) {
      return false;
    }

    try {
      const updatedUserData = await fetchUserProfile(token);
      localStorage.setItem('user_profile', JSON.stringify(updatedUserData));
      setUser(updatedUserData);
      return true;
    } catch (error) {
      console.error('Profile refresh failed:', error.message);
      return false;
    }
  }, [fetchUserProfile, isTokenValid]);

  // Initialize authentication state
  useEffect(() => {
    const initializeAuth = async () => {
      try {
        const token = localStorage.getItem('token');
        const cachedProfile = localStorage.getItem('user_profile');

        if (!token) {
          setLoading(false);
          return;
        }

        if (isTokenValid(token)) {
          // Set API header
          api.defaults.headers.common['Authorization'] = `Bearer ${token}`;

          let userData;

          // Try cached profile first
          if (cachedProfile) {
            try {
              userData = JSON.parse(cachedProfile);
              setUser(userData);

              // Refresh profile in background
              setTimeout(() => refreshProfile(), 1000);
            } catch (parseError) {
              // If cached profile is invalid, fetch fresh
              userData = await fetchUserProfile(token);
              localStorage.setItem('user_profile', JSON.stringify(userData));
              setUser(userData);
            }
          } else {
            // No cache, fetch fresh
            userData = await fetchUserProfile(token);
            localStorage.setItem('user_profile', JSON.stringify(userData));
            setUser(userData);
          }
        } else {
          // Try to refresh expired token
          const refreshed = await refreshToken();
          if (!refreshed) {
            logout();
          }
        }
      } catch (error) {
        console.error('Auth initialization failed:', error.message);
        logout();
      } finally {
        setLoading(false);
      }
    };

    initializeAuth();
  }, [logout, isTokenValid, refreshToken, fetchUserProfile, refreshProfile]);

  // Token expiry monitoring
  useEffect(() => {
    if (!user?.tokenExpiry) return;

    const checkTokenExpiry = () => {
      const currentTime = Date.now() / 1000;
      const timeUntilExpiry = user.tokenExpiry - currentTime;

      if (timeUntilExpiry < TOKEN_EXPIRY_CHECK_BUFFER) {
        refreshToken();
      }
    };

    // Set up interval and check immediately
    const interval = setInterval(checkTokenExpiry, TOKEN_CHECK_INTERVAL);
    checkTokenExpiry();

    return () => clearInterval(interval);
  }, [user?.tokenExpiry, refreshToken]);

  // User update function
  const updateUser = useCallback((updates) => {
    setUser(prevUser => {
      if (!prevUser) return null;

      const updatedUser = {
        ...prevUser,
        ...updates,
        // Ensure admin consistency
        isAdmin: updates.isAdmin === true || updates.is_admin === true || prevUser.isAdmin,
        is_admin: updates.isAdmin === true || updates.is_admin === true || prevUser.is_admin
      };

      // Update cache
      localStorage.setItem('user_profile', JSON.stringify(updatedUser));
      return updatedUser;
    });
  }, []);

  // Helper functions for role checking
  const checkAdminStatus = useCallback(() => {
    return user?.isAdmin === true || user?.is_admin === true;
  }, [user]);

  const checkBetaStatus = useCallback(() => {
    return user?.is_beta_user === true;
  }, [user]);

  // Beta access management
  const grantBetaAccess = useCallback(() => {
    updateUser({ is_beta_user: true });
  }, [updateUser]);

  const updateAdminStatus = useCallback((isAdmin) => {
    updateUser({
      isAdmin: isAdmin,
      is_admin: isAdmin
    });
  }, [updateUser]);

  const contextValue = {
    // State
    user,
    loading,
    isAuthenticated: !!user,
    isBetaUser: !!user && checkBetaStatus(),
    isAdmin: !!user && checkAdminStatus(),

    // Functions
    login,
    logout,
    refreshToken,
    refreshProfile,
    updateUser,
    grantBetaAccess,
    updateAdminStatus,
    checkAdminStatus,
    checkBetaStatus,
    isTokenValid
  };

  return (
    <AuthContext.Provider value={contextValue}>
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};