// src/components/NetworkErrorHandler.jsx
import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

const NetworkErrorHandler = ({ children }) => {
  const [isOnline, setIsOnline] = useState(navigator.onLine);
  const [showOfflineMessage, setShowOfflineMessage] = useState(false);
  const [networkError, setNetworkError] = useState(null);
  const [retryCount, setRetryCount] = useState(0);

  useEffect(() => {
    const handleOnline = () => {
      setIsOnline(true);
      setShowOfflineMessage(false);
      setNetworkError(null);
      setRetryCount(0);

      // Show a brief "back online" message
      const backOnlineToast = document.createElement('div');
      backOnlineToast.className = 'fixed top-4 right-4 bg-green-500/90 backdrop-blur-xl text-white p-4 rounded-xl shadow-2xl border border-green-400/30 z-50';
      backOnlineToast.innerHTML = `
        <div class="flex items-center space-x-3">
          <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"></path>
          </svg>
          <div>
            <p class="font-semibold">Back Online!</p>
            <p class="text-sm text-green-200">Connection restored</p>
          </div>
        </div>
      `;

      document.body.appendChild(backOnlineToast);

      setTimeout(() => {
        if (backOnlineToast.parentNode) {
          backOnlineToast.remove();
        }
      }, 3000);
    };

    const handleOffline = () => {
      setIsOnline(false);
      setShowOfflineMessage(true);
    };

    // Listen for network state changes
    window.addEventListener('online', handleOnline);
    window.addEventListener('offline', handleOffline);

    // Monitor failed API requests
    const originalFetch = window.fetch;
    window.fetch = async (...args) => {
      try {
        const response = await originalFetch(...args);

        // Reset network error if request succeeds
        if (response.ok) {
          setNetworkError(null);
          setRetryCount(0);
        }

        return response;
      } catch (error) {
        if (error.name === 'TypeError' && error.message.includes('fetch')) {
          setNetworkError({
            message: 'Network request failed',
            timestamp: new Date().toISOString(),
            url: args[0]
          });
          setRetryCount(prev => prev + 1);
        }
        throw error;
      }
    };

    return () => {
      window.removeEventListener('online', handleOnline);
      window.removeEventListener('offline', handleOffline);
      window.fetch = originalFetch;
    };
  }, []);

  const handleRetry = () => {
    setNetworkError(null);
    window.location.reload();
  };

  const getConnectionQuality = () => {
    if ('connection' in navigator) {
      const connection = navigator.connection;
      return {
        effectiveType: connection.effectiveType,
        downlink: connection.downlink,
        rtt: connection.rtt
      };
    }
    return null;
  };

  const connectionQuality = getConnectionQuality();

  return (
    <>
      <AnimatePresence>
        {/* Offline Message */}
        {showOfflineMessage && (
          <motion.div
            className="fixed top-4 left-4 right-4 z-50 bg-red-500/90 backdrop-blur-xl text-white p-4 rounded-xl shadow-2xl border border-red-400/30"
            initial={{ opacity: 0, y: -100 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -100 }}
            transition={{ duration: 0.3 }}
          >
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-3">
                <motion.div
                  animate={{ rotate: 360 }}
                  transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
                >
                  <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M18.364 5.636l-3.536 3.536m0 5.656l3.536 3.536M9.172 9.172L5.636 5.636m3.536 9.192L5.636 18.364M12 2v6m0 8v6m8-12h-6m-8 0h6"></path>
                  </svg>
                </motion.div>
                <div>
                  <p className="font-semibold">Connection Lost</p>
                  <p className="text-sm text-red-200">
                    Please check your internet connection. Your work is automatically saved.
                  </p>
                </div>
              </div>

              <motion.button
                onClick={() => setShowOfflineMessage(false)}
                className="text-red-200 hover:text-white transition-colors"
                whileHover={{ scale: 1.1 }}
                whileTap={{ scale: 0.9 }}
              >
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M6 18L18 6M6 6l12 12"></path>
                </svg>
              </motion.button>
            </div>
          </motion.div>
        )}

        {/* Network Error Message */}
        {networkError && retryCount > 2 && (
          <motion.div
            className="fixed top-20 left-4 right-4 z-50 bg-orange-500/90 backdrop-blur-xl text-white p-4 rounded-xl shadow-2xl border border-orange-400/30"
            initial={{ opacity: 0, y: -100 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -100 }}
            transition={{ duration: 0.3 }}
          >
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-3">
                <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L3.732 6.5c-.77.833.192 2.5 1.732 2.5z"></path>
                </svg>
                <div>
                  <p className="font-semibold">Network Issues Detected</p>
                  <p className="text-sm text-orange-200">
                    Some requests are failing. Please check your connection.
                  </p>
                  {connectionQuality && (
                    <p className="text-xs text-orange-300 mt-1">
                      Connection: {connectionQuality.effectiveType} | Speed: {connectionQuality.downlink}Mbps
                    </p>
                  )}
                </div>
              </div>

              <div className="flex items-center space-x-2">
                <motion.button
                  onClick={handleRetry}
                  className="bg-orange-600 hover:bg-orange-700 text-white px-3 py-1 rounded text-sm transition-colors"
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                >
                  Retry
                </motion.button>

                <motion.button
                  onClick={() => setNetworkError(null)}
                  className="text-orange-200 hover:text-white transition-colors"
                  whileHover={{ scale: 1.1 }}
                  whileTap={{ scale: 0.9 }}
                >
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M6 18L18 6M6 6l12 12"></path>
                  </svg>
                </motion.button>
              </div>
            </div>
          </motion.div>
        )}

        {/* Slow Connection Warning */}
        {isOnline && connectionQuality?.effectiveType === 'slow-2g' && (
          <motion.div
            className="fixed bottom-4 left-4 right-4 z-40 bg-yellow-500/90 backdrop-blur-xl text-yellow-900 p-3 rounded-xl shadow-2xl border border-yellow-400/30"
            initial={{ opacity: 0, y: 100 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 100 }}
            transition={{ duration: 0.3 }}
          >
            <div className="flex items-center space-x-3">
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path>
              </svg>
              <div>
                <p className="font-semibold text-sm">Slow Connection Detected</p>
                <p className="text-xs text-yellow-800">
                  Some features may take longer to load
                </p>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Main Content with Network Status Overlay */}
      <div className={`${!isOnline ? 'opacity-75' : ''} transition-opacity duration-500`}>
        {children}

        {/* Network Status Indicator */}
        <div className="fixed bottom-4 right-4 z-30">
          <motion.div
            className={`w-3 h-3 rounded-full ${
              isOnline
                ? networkError && retryCount > 2
                  ? 'bg-orange-400'
                  : 'bg-green-400'
                : 'bg-red-400'
            } shadow-lg`}
            animate={{
              scale: isOnline ? [1, 1.2, 1] : [1, 0.8, 1],
              opacity: [0.7, 1, 0.7]
            }}
            transition={{
              duration: 2,
              repeat: Infinity,
              ease: "easeInOut"
            }}
            title={
              isOnline
                ? networkError && retryCount > 2
                  ? 'Network issues detected'
                  : 'Connected'
                : 'Offline'
            }
          />
        </div>
      </div>
    </>
  );
};

export default NetworkErrorHandler;