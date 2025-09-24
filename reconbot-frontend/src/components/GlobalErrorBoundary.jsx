// src/components/GlobalErrorBoundary.jsx
import React from 'react';
import { motion } from 'framer-motion';

class GlobalErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      hasError: false,
      error: null,
      errorInfo: null,
      errorId: null,
      showDetails: false
    };
  }

  static getDerivedStateFromError(error) {
    return {
      hasError: true,
      errorId: Math.random().toString(36).substr(2, 9)
    };
  }

  componentDidCatch(error, errorInfo) {
    this.setState({
      error,
      errorInfo
    });

    // Log error to console for development
    console.error('Global Error Boundary caught an error:', error, errorInfo);

    // In production, you can send this to your monitoring service
    // Example: Sentry.captureException(error, { extra: errorInfo });

    // Send error report to your backend
    this.reportError(error, errorInfo);
  }

  reportError = async (error, errorInfo) => {
    try {
      // Only report in production
      if (process.env.NODE_ENV === 'production') {
        await fetch('/api/error-report', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            error: {
              message: error.message,
              stack: error.stack,
              name: error.name
            },
            errorInfo: {
              componentStack: errorInfo.componentStack
            },
            errorId: this.state.errorId,
            userAgent: navigator.userAgent,
            url: window.location.href,
            timestamp: new Date().toISOString()
          })
        });
      }
    } catch (reportError) {
      console.error('Failed to report error:', reportError);
    }
  };

  handleReload = () => {
    // Clear any corrupted state from localStorage
    const keysToKeep = ['token', 'google_token'];
    const itemsToKeep = {};

    keysToKeep.forEach(key => {
      const value = localStorage.getItem(key);
      if (value) itemsToKeep[key] = value;
    });

    localStorage.clear();

    Object.entries(itemsToKeep).forEach(([key, value]) => {
      localStorage.setItem(key, value);
    });

    window.location.reload();
  };

  handleGoHome = () => {
    this.setState({ hasError: false, error: null, errorInfo: null });
    window.history.pushState(null, null, '/');
    window.location.reload();
  };

  render() {
    if (this.state.hasError) {
      return (
        <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 flex items-center justify-center p-4 relative overflow-hidden">
          {/* Animated Background Elements */}
          <div className="absolute inset-0">
            {[...Array(15)].map((_, i) => (
              <motion.div
                key={i}
                className="absolute w-2 h-2 bg-red-400/20 rounded-full"
                animate={{
                  x: [Math.random() * window.innerWidth, Math.random() * window.innerWidth],
                  y: [Math.random() * window.innerHeight, Math.random() * window.innerHeight],
                }}
                transition={{
                  duration: Math.random() * 20 + 10,
                  repeat: Infinity,
                  repeatType: "reverse",
                }}
                initial={{
                  x: Math.random() * window.innerWidth,
                  y: Math.random() * window.innerHeight,
                }}
              />
            ))}
          </div>

          <motion.div
            className="bg-white/10 backdrop-blur-xl border border-white/20 rounded-2xl p-8 max-w-lg w-full text-center relative z-10"
            initial={{ opacity: 0, scale: 0.9, y: 50 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            transition={{ duration: 0.6, ease: "easeOut" }}
          >
            {/* Error Icon */}
            <motion.div
              className="w-20 h-20 bg-red-500/20 rounded-full flex items-center justify-center mx-auto mb-6 border border-red-400/30"
              animate={{
                scale: [1, 1.05, 1],
                rotate: [0, 1, -1, 0]
              }}
              transition={{
                duration: 3,
                repeat: Infinity,
                ease: "easeInOut"
              }}
            >
              <svg className="w-10 h-10 text-red-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L3.732 6.5c-.77.833.192 2.5 1.732 2.5z"></path>
              </svg>
            </motion.div>

            {/* Error Title */}
            <motion.h1
              className="text-3xl font-bold text-white mb-4"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.3 }}
            >
              Oops! Something went wrong
            </motion.h1>

            {/* Error Description */}
            <motion.p
              className="text-white/70 mb-6 leading-relaxed"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.5 }}
            >
              We encountered an unexpected error. Don't worry - your data is safe and our team has been automatically notified.
            </motion.p>

            {/* Error ID */}
            <motion.div
              className="bg-white/5 rounded-lg p-4 mb-6 border border-white/10"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.7 }}
            >
              <p className="text-sm text-white/50 mb-2">Error Reference:</p>
              <div className="flex items-center justify-center space-x-2">
                <code className="text-red-300 text-sm font-mono bg-red-900/20 px-3 py-1 rounded">
                  {this.state.errorId}
                </code>
                <button
                  onClick={() => navigator.clipboard.writeText(this.state.errorId)}
                  className="text-white/50 hover:text-white transition-colors"
                  title="Copy Error ID"
                >
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z"></path>
                  </svg>
                </button>
              </div>
            </motion.div>

            {/* Action Buttons */}
            <motion.div
              className="space-y-3"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.9 }}
            >
              <motion.button
                onClick={this.handleReload}
                className="w-full bg-gradient-to-r from-purple-500 to-pink-500 hover:from-purple-600 hover:to-pink-600 text-white py-3 px-6 rounded-lg font-medium transition-all duration-300 shadow-lg"
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
              >
                Reload Application
              </motion.button>

              <motion.button
                onClick={this.handleGoHome}
                className="w-full bg-white/10 hover:bg-white/20 text-white py-3 px-6 rounded-lg font-medium border border-white/20 transition-all duration-300"
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
              >
                Go to Homepage
              </motion.button>
            </motion.div>

            {/* Developer Details Toggle */}
            {process.env.NODE_ENV === 'development' && this.state.error && (
              <motion.div
                className="mt-6"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 1.1 }}
              >
                <button
                  onClick={() => this.setState({ showDetails: !this.state.showDetails })}
                  className="text-red-400 hover:text-red-300 text-sm transition-colors flex items-center mx-auto space-x-2"
                >
                  <span>Developer Details</span>
                  <svg
                    className={`w-4 h-4 transition-transform ${this.state.showDetails ? 'rotate-180' : ''}`}
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 9l-7 7-7-7"></path>
                  </svg>
                </button>

                {this.state.showDetails && (
                  <motion.div
                    className="mt-4 text-left"
                    initial={{ opacity: 0, height: 0 }}
                    animate={{ opacity: 1, height: 'auto' }}
                    transition={{ duration: 0.3 }}
                  >
                    <div className="bg-red-900/20 border border-red-400/20 rounded-lg p-4 max-h-60 overflow-auto">
                      <h4 className="text-red-300 font-semibold mb-2">Error Details:</h4>
                      <pre className="text-xs text-red-200 whitespace-pre-wrap break-words">
                        <strong>Message:</strong> {this.state.error.message}
                        {'\n\n'}
                        <strong>Stack:</strong> {this.state.error.stack}
                        {this.state.errorInfo?.componentStack && (
                          <>
                            {'\n\n'}
                            <strong>Component Stack:</strong> {this.state.errorInfo.componentStack}
                          </>
                        )}
                      </pre>
                    </div>
                  </motion.div>
                )}
              </motion.div>
            )}

            {/* Help Information */}
            <motion.div
              className="mt-6 p-4 bg-blue-500/10 border border-blue-400/20 rounded-lg"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 1.3 }}
            >
              <p className="text-blue-200 text-sm">
                <strong>Need help?</strong> Contact our support team at{' '}
                <a
                  href="mailto:support@matchledger.in"
                  className="text-blue-300 hover:text-blue-100 underline"
                >
                  support@matchledger.in
                </a>
                {' '}and include the error reference above.
              </p>
            </motion.div>
          </motion.div>
        </div>
      );
    }

    return this.props.children;
  }
}

export default GlobalErrorBoundary;