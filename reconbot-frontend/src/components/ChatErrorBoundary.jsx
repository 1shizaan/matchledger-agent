// src/components/ChatErrorBoundary.jsx
import React from 'react';
import { motion } from 'framer-motion';

class ChatErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error) {
    // Update state so the next render will show the fallback UI
    return { hasError: true, error: error };
  }

  componentDidCatch(error, errorInfo) {
    // You can log the error to an error reporting service
    console.error('Chat Error Boundary caught an error:', error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      // Fallback UI
      return (
        <motion.div
          className="space-y-6"
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.6 }}
        >
          <div className="bg-white/5 rounded-2xl border border-white/10 backdrop-blur-sm overflow-hidden">
            <div className="h-96 overflow-y-auto p-6 flex items-center justify-center">
              <div className="text-center space-y-4">
                <div className="text-6xl">🤖💥</div>
                <h3 className="text-xl font-semibold text-white">Chat Temporarily Unavailable</h3>
                <p className="text-white/70 max-w-md">
                  The chat system encountered an error. This usually happens when the server sends unexpected data.
                </p>
                <div className="space-y-2">
                  <button
                    onClick={() => this.setState({ hasError: false, error: null })}
                    className="bg-gradient-to-r from-purple-500 to-pink-500 hover:from-purple-600 hover:to-pink-600 text-white px-6 py-2 rounded-lg transition-all duration-300 mr-2"
                  >
                    🔄 Try Again
                  </button>
                  <button
                    onClick={() => window.location.reload()}
                    className="bg-white/10 hover:bg-white/20 border border-white/20 text-white/80 px-6 py-2 rounded-lg transition-all duration-200"
                  >
                    🔃 Refresh Page
                  </button>
                </div>
                {this.props.showErrorDetails && this.state.error && (
                  <details className="mt-4 text-left bg-red-500/10 border border-red-500/30 rounded-lg p-3">
                    <summary className="cursor-pointer text-red-300 text-sm">
                      🔍 Technical Details (for developers)
                    </summary>
                    <pre className="text-xs text-red-200/80 mt-2 overflow-auto">
                      {this.state.error.toString()}
                    </pre>
                  </details>
                )}
              </div>
            </div>
          </div>
        </motion.div>
      );
    }

    return this.props.children;
  }
}

export default ChatErrorBoundary;