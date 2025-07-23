// src/utils/api.js
import axios from 'axios';

// Create an Axios instance
const api = axios.create({
  // baseURL: 'https://coldemailai.in', // Use HTTPS to match your frontend
  baseURL: import.meta.env.VITE_API_BASE_URL || 'https://coldemailai.in',
  timeout: 30000, // 30s timeout
 // headers: {
   //'Content-Type': 'application/json',
   //'X-Requested-With': 'XMLHttpRequest'
  //}
});

// Variable to hold the logout function reference
let onLogoutCallback = null;

// Function to set the logout callback from AuthContext
export const setLogoutCallback = (callback) => {
  onLogoutCallback = callback;
};

// Request interceptor for auth and errors
api.interceptors.request.use(config => {
  // Inject JWT token if exists
  const token = localStorage.getItem('token');
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }

  // Add security headers
  config.headers['X-CSRF-Protection'] = '1';
  config.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains';

  return config;
}, error => {
  return Promise.reject(error);
});

// Response interceptor for unified error handling
api.interceptors.response.use(
  response => response,
  error => {
    // Handle specific error codes
    if (error.code === 'ECONNABORTED') {
      throw new Error('Request timeout - please try again');
    }

    const status = error.response?.status;
    const data = error.response?.data;

    if (status === 401) {
      // Auto-logout on unauthorized
      if (onLogoutCallback) {
        onLogoutCallback();
      }
    }

    if (status === 413) {
      throw new Error('File too large (max 10MB)');
    }

    // Use server error message or default
    throw new Error(data?.detail || data?.message || 'Request failed');
  }
);

// Enhanced reconcileFiles function with proper FormData handling
// ✅ UPDATED: Now uses /api/reconcile path
export const reconcileFiles = async (ledgerFile, bankFile, email = null) => {
  const MAX_SIZE = 10 * 1024 * 1024; // 10MB

  if (ledgerFile.size > MAX_SIZE || bankFile.size > MAX_SIZE) {
    throw new Error('Each file must be smaller than 10MB');
  }

  const formData = new FormData();
  formData.append('ledger_file', ledgerFile);
  formData.append('bank_file', bankFile);

  if (bankFile.type === 'application/pdf') {
    formData.append('bank_is_pdf', 'true');
  }

  if (email) {
    formData.append('email', email);
  }

  try {
    const { data } = await api.post('/api/reconcile', formData, {
      headers: {
        'Content-Type': 'multipart/form-data', // Override for file uploads
      },
      onUploadProgress: progress => {
        const percent = Math.round((progress.loaded * 100) / progress.total);
        console.log(`Upload: ${percent}%`);
      },
      timeout: 120000 // 2 minutes for large files
    });

    return data;
  } catch (error) {
    console.error('Reconciliation error:', error);
    throw error;
  }
};

// ✅ NEW: Google Auth function with correct /auth/google path
export const googleAuth = async (token) => {
  try {
    const { data } = await api.post('/auth/google', { token });
    return data;
  } catch (error) {
    console.error('Google auth error:', error);
    throw error;
  }
};

// Add these helper functions
export const checkAPIHealth = async () => {
  try {
    await api.get('/health');
    return true;
  } catch {
    return false;
  }
};

export const setNewBaseURL = (newUrl) => {
  api.defaults.baseURL = newUrl;
  localStorage.setItem('api_base_url', newUrl);
};

// Export the configured Axios instance as the default export
export default api;