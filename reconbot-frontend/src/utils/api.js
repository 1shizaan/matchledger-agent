// src/utils/api.js - ENHANCED VERSION WITH ADVANCED SECURITY & ERROR HANDLING
import axios from 'axios';
import securityMonitor from './monitoring';

const api = axios.create({
  baseURL: 'https://matchledger.in',
  timeout: 120000, // Increased timeout for large file uploads
});

let onLogoutCallback = null;
let isRefreshing = false;
let failedQueue = [];

// Enhanced Security Features
class SecurityManager {
  constructor() {
    this.suspiciousActivityCount = 0;
    this.lastFailedAttempt = null;
    this.rateLimitCount = 0;
    this.lastRateLimitReset = Date.now();
  }

  checkSuspiciousActivity() {
    // Reset counters every 5 minutes
    if (Date.now() - this.lastRateLimitReset > 300000) {
      this.rateLimitCount = 0;
      this.suspiciousActivityCount = 0;
      this.lastRateLimitReset = Date.now();
    }

    this.suspiciousActivityCount++;

    if (this.suspiciousActivityCount > 10) {
      securityMonitor.logSecurityEvent(
        'HIGH_ERROR_RATE',
        'high',
        { errorCount: this.suspiciousActivityCount }
      );
      return true;
    }
    return false;
  }

  handleRateLimit() {
    this.rateLimitCount++;
    if (this.rateLimitCount > 3) {
      securityMonitor.logSecurityEvent(
        'RATE_LIMIT_EXCEEDED',
        'high',
        { attempts: this.rateLimitCount }
      );
    }
  }

  validateToken(token) {
    if (!token || typeof token !== 'string') return false;

    try {
      // Basic JWT structure validation
      const parts = token.split('.');
      if (parts.length !== 3) return false;

      const payload = JSON.parse(atob(parts[1]));
      const currentTime = Date.now() / 1000;

      // Check if token is expired or expires very soon
      return payload.exp && payload.exp > (currentTime + 60);
    } catch (error) {
      securityMonitor.logError(error, {}, 'Token validation failed');
      return false;
    }
  }
}

const securityManager = new SecurityManager();

export const setLogoutCallback = (callback) => {
  onLogoutCallback = callback;
};

const processQueue = (error, token = null) => {
  failedQueue.forEach(({ resolve, reject }) => {
    if (error) {
      reject(error);
    } else {
      resolve(token);
    }
  });
  failedQueue = [];
};

const isTokenExpired = (token) => {
  if (!token) return true;
  try {
    const payload = JSON.parse(atob(token.split('.')[1]));
    const currentTime = Date.now() / 1000;
    // Check if token expires in less than 5 minutes
    return payload.exp < (currentTime + 300);
  } catch (error) {
    securityMonitor.logError(error, {}, 'Token expiration check failed');
    return true;
  }
};

// Enhanced Request Interceptor
api.interceptors.request.use(async (config) => {
  try {
    let token = localStorage.getItem('token');

    // Enhanced token validation
    if (token && !securityManager.validateToken(token)) {
      console.log('🔐 Token validation failed, removing invalid token');
      localStorage.removeItem('token');
      token = null;
    }

    // Check if token is expired or about to expire
    if (token && isTokenExpired(token)) {
      console.log('🔄 Token expired, attempting refresh...');

      if (!isRefreshing) {
        isRefreshing = true;

        try {
          const googleToken = localStorage.getItem('google_token');
          if (googleToken) {
            console.log('🔄 Refreshing with Google token...');
            const response = await api.post('/auth/google', { token: googleToken });
            const newToken = response.data.access_token;

            // Validate new token before storing
            if (securityManager.validateToken(newToken)) {
              localStorage.setItem('token', newToken);
              token = newToken;
              processQueue(null, newToken);
              console.log('✅ Token refreshed successfully');
            } else {
              throw new Error('Invalid token received from refresh');
            }
          } else {
            throw new Error('No refresh method available');
          }
        } catch (error) {
          console.error('❌ Token refresh failed:', error);
          processQueue(error, null);
          localStorage.removeItem('token');
          localStorage.removeItem('google_token');

          securityMonitor.logSecurityEvent(
            'TOKEN_REFRESH_FAILED',
            'medium',
            { error: error.message }
          );

          if (onLogoutCallback) onLogoutCallback();
          return Promise.reject(error);
        } finally {
          isRefreshing = false;
        }
      } else {
        // If already refreshing, queue this request
        return new Promise((resolve, reject) => {
          failedQueue.push({ resolve, reject });
        }).then(token => {
          config.headers.Authorization = `Bearer ${token}`;
          return config;
        }).catch(err => {
          return Promise.reject(err);
        });
      }
    }

    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }

    // Add request ID for tracking
    config.headers['X-Request-ID'] = `req_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;

    // Enhanced security headers
    config.headers['X-Content-Type-Options'] = 'nosniff';
    config.headers['X-Frame-Options'] = 'DENY';

    return config;
  } catch (error) {
    securityMonitor.logError(error, {}, 'Request interceptor failed');
    return Promise.reject(error);
  }
}, error => {
  securityMonitor.logError(error, {}, 'Request interceptor error');
  return Promise.reject(error);
});

// Enhanced Response Interceptor with Better Error Handling
api.interceptors.response.use(
  response => {
    // Reset suspicious activity counter on successful requests
    securityManager.suspiciousActivityCount = Math.max(0, securityManager.suspiciousActivityCount - 1);
    return response;
  },
  async error => {
    const status = error.response?.status;
    const data = error.response?.data;
    const originalRequest = error.config;

    // Enhanced error logging
    securityMonitor.logError(error, {
      status,
      url: originalRequest?.url,
      method: originalRequest?.method,
      responseData: data
    }, 'API Response Error');

    // Check for suspicious activity
    if (securityManager.checkSuspiciousActivity()) {
      console.warn('⚠️ High error rate detected - potential security issue');
    }

    // Handle specific error cases
    switch (status) {
      case 401:
        if (!originalRequest._retry) {
          originalRequest._retry = true;
          console.log('🔒 401 Unauthorized - clearing tokens');
          localStorage.removeItem('token');
          localStorage.removeItem('google_token');
          if (onLogoutCallback) onLogoutCallback();
        }
        break;

      case 403:
        securityMonitor.logSecurityEvent(
          'FORBIDDEN_ACCESS',
          'medium',
          { url: originalRequest?.url, method: originalRequest?.method }
        );
        break;

      case 429:
        securityManager.handleRateLimit();
        console.warn('⏱️ Rate limit exceeded');
        break;

      case 500:
      case 502:
      case 503:
      case 504:
        console.error('🚨 Server error detected:', status);
        break;
    }

    // Enhanced error messages
    const errorMessage = data?.detail ||
                        data?.message ||
                        (status === 429 ? 'Too many requests. Please wait and try again.' :
                         status === 401 ? 'Your session has expired. Please log in again.' :
                         status === 403 ? 'Access forbidden. Please check your permissions.' :
                         status >= 500 ? 'Server error. Please try again later.' :
                         'An unexpected error occurred. Please try again.');

    throw new Error(errorMessage);
  }
);

// Enhanced Google Auth with Better Security
export const googleAuth = async (token) => {
  try {
    console.log('🔐 Starting Google authentication...');

    // Validate Google token format (basic check)
    if (!token || typeof token !== 'string' || token.length < 100) {
      throw new Error('Invalid Google token format');
    }

    const { data } = await api.post('/auth/google', { token });

    // Enhanced token validation
    if (!data.access_token || !securityManager.validateToken(data.access_token)) {
      throw new Error('Invalid access token received');
    }

    // Store both tokens securely
    localStorage.setItem('token', data.access_token);
    localStorage.setItem('google_token', token);

    securityMonitor.logSecurityEvent(
      'SUCCESSFUL_LOGIN',
      'info',
      { method: 'google', userEmail: data.user?.email }
    );

    console.log('✅ Google authentication successful');
    return data;
  } catch (error) {
    securityMonitor.logError(error, { method: 'google_auth' }, 'Google authentication failed');
    throw error;
  }
};

// Add this function to your api.js file
export const getUserProfile = async () => {
  try {
    const { data } = await api.get('/api/user/profile');
    return data;
  } catch (error) {
    securityMonitor.logError(error, {}, 'User profile fetch failed');
    throw new Error(error.response?.data?.detail || 'Failed to fetch user profile');
  }
};

// Enhanced File Upload with Progress and Security
export const startReconciliation = async (ledgerFile, bankFile, email = null, onProgress = null) => {
  const MAX_SIZE = 25 * 1024 * 1024; // 25MB limit
  const ALLOWED_TYPES = [
    'text/csv',
    'application/vnd.ms-excel',
    'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
    'application/pdf'
  ];

  // Enhanced file validation
  if (ledgerFile.size > MAX_SIZE || bankFile.size > MAX_SIZE) {
    throw new Error(`Each file must be smaller than ${MAX_SIZE / 1024 / 1024}MB`);
  }

  if (!ALLOWED_TYPES.includes(ledgerFile.type) || !ALLOWED_TYPES.includes(bankFile.type)) {
    throw new Error('Only CSV, Excel, and PDF files are allowed');
  }

  // Check for suspicious file names
  const suspiciousPatterns = [/\.exe$/, /\.bat$/, /\.cmd$/, /\.scr$/, /\.js$/, /\.html$/];
  if (suspiciousPatterns.some(pattern =>
    pattern.test(ledgerFile.name) || pattern.test(bankFile.name)
  )) {
    securityMonitor.logSecurityEvent(
      'SUSPICIOUS_FILE_UPLOAD',
      'high',
      {
        ledgerFile: ledgerFile.name,
        bankFile: bankFile.name,
        ledgerType: ledgerFile.type,
        bankType: bankFile.type
      }
    );
    throw new Error('File type not allowed for security reasons');
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
    const { data } = await api.post('/api/reconcile/start', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      onUploadProgress: (progressEvent) => {
        if (onProgress) {
          const percentCompleted = Math.round((progressEvent.loaded * 100) / progressEvent.total);
          onProgress(percentCompleted);
        }
      },
    });

    securityMonitor.logSecurityEvent(
      'FILE_UPLOAD_SUCCESS',
      'info',
      {
        taskId: data.task_id,
        ledgerSize: ledgerFile.size,
        bankSize: bankFile.size
      }
    );

    return data;
  } catch (error) {
    securityMonitor.logError(error, {
      ledgerFile: ledgerFile.name,
      bankFile: bankFile.name,
      sizes: { ledger: ledgerFile.size, bank: bankFile.size }
    }, 'File upload failed');
    throw error;
  }
};

// Enhanced status checking with retry logic
export const getReconciliationStatus = async (taskId, retryCount = 0) => {
  const MAX_RETRIES = 3;

  try {
    if (!taskId || typeof taskId !== 'string') {
      throw new Error('Invalid task ID');
    }

    const { data } = await api.get(`/api/reconcile/status/${taskId}`);
    return data;
  } catch (error) {
    if (retryCount < MAX_RETRIES && error.response?.status >= 500) {
      console.log(`🔄 Retrying status check (attempt ${retryCount + 1}/${MAX_RETRIES})`);
      await new Promise(resolve => setTimeout(resolve, 1000 * (retryCount + 1)));
      return getReconciliationStatus(taskId, retryCount + 1);
    }

    securityMonitor.logError(error, { taskId, retryCount }, 'Status check failed');
    throw error;
  }
};

// All other functions with enhanced error handling...
export const chatWithData = async (query, ledgerFile, bankFile, reconciliationSummary = null) => {
  try {
    // Input validation
    if (!query || query.trim().length === 0) {
      throw new Error('Query cannot be empty');
    }

    if (query.length > 5000) {
      throw new Error('Query too long. Maximum 5000 characters allowed.');
    }

    const formData = new FormData();
    formData.append('query', query.trim());
    formData.append('ledger_file', ledgerFile);
    formData.append('bank_file', bankFile);

    if (reconciliationSummary) {
      formData.append('reconciliation_summary', JSON.stringify(reconciliationSummary));
    }

    const { data } = await api.post('/api/chat', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
      timeout: 60000, // 60 second timeout for AI queries
    });

    securityMonitor.logSecurityEvent(
      'AI_QUERY_SUCCESS',
      'info',
      { queryLength: query.length, hasReconciliation: !!reconciliationSummary }
    );

    return data;
  } catch (error) {
    securityMonitor.logError(error, { queryLength: query?.length }, 'Chat query failed');
    throw new Error(error.response?.data?.detail || 'Chat failed');
  }
};

export const getReconciliationHistory = async (skip = 0, limit = 100) => {
  try {
    if (skip < 0 || limit <= 0 || limit > 1000) {
      throw new Error('Invalid pagination parameters');
    }

    const { data } = await api.get(`/api/history?skip=${skip}&limit=${limit}`);
    return data;
  } catch (error) {
    securityMonitor.logError(error, { skip, limit }, 'History fetch failed');
    throw new Error(error.response?.data?.detail || 'Failed to fetch history');
  }
};

export const getUsageStats = async () => {
  try {
    const { data } = await api.get('/api/usage/stats');
    return data;
  } catch (error) {
    securityMonitor.logError(error, {}, 'Usage stats failed');
    throw new Error(error.response?.data?.detail || 'Failed to fetch usage stats');
  }
};

export const submitBetaFeedback = async (message) => {
  try {
    if (!message || message.trim().length === 0) {
      throw new Error('Feedback message cannot be empty');
    }

    if (message.length > 5000) {
      throw new Error('Feedback too long. Maximum 5000 characters allowed.');
    }

    const formData = new FormData();
    formData.append('message', message.trim());

    const { data } = await api.post('/api/beta/feedback', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });

    securityMonitor.logSecurityEvent(
      'FEEDBACK_SUBMITTED',
      'info',
      { messageLength: message.length }
    );

    return data;
  } catch (error) {
    securityMonitor.logError(error, { messageLength: message?.length }, 'Feedback submission failed');
    throw new Error(error.response?.data?.detail || 'Failed to submit feedback');
  }
};

export const detectColumns = async (ledgerFile, bankFile) => {
  try {
    const formData = new FormData();
    formData.append('ledger_file', ledgerFile);
    formData.append('bank_file', bankFile);

    console.log('🔍 Sending column detection request...');
    const response = await api.post('/api/detect-columns', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
      timeout: 30000, // 30 second timeout
    });

    console.log('✅ Column detection successful:', response.data);
    return response.data;
  } catch (error) {
    console.error('❌ Column detection failed:', error);
    securityMonitor.logError(error, {
      ledgerFile: ledgerFile?.name,
      bankFile: bankFile?.name
    }, 'Column detection failed');
    throw new Error(error.response?.data?.detail || 'Failed to detect columns');
  }
};

export const startReconciliationWithMapping = async (formData) => {
  try {
    console.log('🚀 Starting enhanced reconciliation with mapping...');
    const response = await api.post('/api/reconcile/start-with-mapping', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
      timeout: 120000, // 2 minute timeout for enhanced processing
    });

    console.log('✅ Enhanced reconciliation started:', response.data);

    securityMonitor.logSecurityEvent(
      'ENHANCED_RECONCILIATION_STARTED',
      'info',
      { taskId: response.data.task_id }
    );

    return response.data;
  } catch (error) {
    console.error('❌ Enhanced reconciliation failed:', error);
    securityMonitor.logError(error, {}, 'Enhanced reconciliation failed');
    throw new Error(error.response?.data?.detail || 'Failed to start enhanced reconciliation');
  }
};

export const getAdminFeedback = async () => {
  try {
    const { data } = await api.get('/api/admin/beta-feedback');
    return data;
  } catch (error) {
    securityMonitor.logError(error, {}, 'Admin feedback fetch failed');
    throw new Error(error.response?.data?.detail || 'Failed to fetch feedback');
  }
};

// Security utility functions
export const validateFileSecurely = (file, maxSize = 25 * 1024 * 1024) => {
  const allowedTypes = [
    'text/csv',
    'application/vnd.ms-excel',
    'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
    'application/pdf'
  ];

  if (!file) return { valid: false, error: 'No file provided' };
  if (file.size > maxSize) return { valid: false, error: `File too large (max ${maxSize / 1024 / 1024}MB)` };
  if (!allowedTypes.includes(file.type)) return { valid: false, error: 'File type not allowed' };

  return { valid: true };
};

export const clearSecureSession = () => {
  try {
    // Clear all auth-related storage
    localStorage.removeItem('token');
    localStorage.removeItem('google_token');
    sessionStorage.clear();

    // Clear any cached data
    if ('caches' in window) {
      caches.keys().then(names => {
        names.forEach(name => caches.delete(name));
      });
    }

    securityMonitor.logSecurityEvent(
      'SECURE_LOGOUT',
      'info',
      { timestamp: new Date().toISOString() }
    );
  } catch (error) {
    console.error('Error during secure session clear:', error);
  }
};

export default api;