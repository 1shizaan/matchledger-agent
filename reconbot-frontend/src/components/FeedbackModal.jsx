// src/components/FeedbackModal.jsx
import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import api from '../utils/api';

const FeedbackModal = ({ isOpen, onClose }) => {
  const [feedback, setFeedback] = useState('');
  const [email, setEmail] = useState('');
  const [category, setCategory] = useState('general');
  const [loading, setLoading] = useState(false);
  const [submitted, setSubmitted] = useState(false);
  const [error, setError] = useState(null);

  const categories = [
    { value: 'general', label: '💬 General Feedback' },
    { value: 'bug', label: '🐛 Bug Report' },
    { value: 'feature', label: '✨ Feature Request' },
    { value: 'ui', label: '🎨 UI/UX Improvement' },
    { value: 'performance', label: '⚡ Performance Issue' }
  ];

  const handleSubmit = async (e) => {
    e.preventDefault();

    if (!feedback.trim()) {
      setError('Please enter your feedback');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append('message', feedback.trim());
      formData.append('category', category);
      if (email.trim()) {
        formData.append('contact_email', email.trim());
      }

      await api.post('/api/beta/feedback', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        }
      });

      setSubmitted(true);
      setTimeout(() => {
        handleClose();
      }, 2000);

    } catch (error) {
      setError(error.response?.data?.message || 'Failed to submit feedback. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const handleClose = () => {
    setFeedback('');
    setEmail('');
    setCategory('general');
    setSubmitted(false);
    setError(null);
    setLoading(false);
    onClose();
  };

  if (!isOpen) return null;

  return (
    <AnimatePresence>
      <motion.div
        className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center p-4"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        onClick={(e) => e.target === e.currentTarget && handleClose()}
      >
        <motion.div
          className="bg-white/10 backdrop-blur-xl border border-white/20 rounded-3xl p-8 max-w-2xl w-full max-h-[90vh] overflow-y-auto"
          initial={{ opacity: 0, scale: 0.8, y: 50 }}
          animate={{ opacity: 1, scale: 1, y: 0 }}
          exit={{ opacity: 0, scale: 0.8, y: 50 }}
          transition={{ duration: 0.3 }}
        >
          {/* Header */}
          <div className="flex items-center justify-between mb-6">
            <div>
              <h2 className="text-2xl font-bold text-white mb-2">Share Your Feedback</h2>
              <p className="text-white/70">Help us improve MatchLedger AI</p>
            </div>
            <button
              onClick={handleClose}
              className="text-white/50 hover:text-white transition-colors p-2 rounded-xl hover:bg-white/10"
            >
              <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M6 18L18 6M6 6l12 12"></path>
              </svg>
            </button>
          </div>

          {submitted ? (
            <motion.div
              className="text-center py-12"
              initial={{ opacity: 0, scale: 0.8 }}
              animate={{ opacity: 1, scale: 1 }}
            >
              <div className="text-6xl mb-4">🎉</div>
              <h3 className="text-2xl font-bold text-white mb-2">Thank You!</h3>
              <p className="text-white/70">
                Your feedback has been submitted successfully. We appreciate your input!
              </p>
            </motion.div>
          ) : (
            <form onSubmit={handleSubmit} className="space-y-6">
              {/* Category Selection */}
              <div>
                <label className="block text-sm font-medium text-white/90 mb-3">
                  What type of feedback is this?
                </label>
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                  {categories.map((cat) => (
                    <motion.button
                      key={cat.value}
                      type="button"
                      onClick={() => setCategory(cat.value)}
                      className={`p-3 rounded-xl text-left transition-all duration-200 border ${
                        category === cat.value
                          ? 'bg-purple-500/20 border-purple-400/50 text-white'
                          : 'bg-white/5 border-white/20 text-white/70 hover:bg-white/10'
                      }`}
                      whileHover={{ scale: 1.02 }}
                      whileTap={{ scale: 0.98 }}
                    >
                      <div className="text-sm font-medium">{cat.label}</div>
                    </motion.button>
                  ))}
                </div>
              </div>

              {/* Feedback Text */}
              <div>
                <label className="block text-sm font-medium text-white/90 mb-3">
                  Your Feedback *
                </label>
                <textarea
                  value={feedback}
                  onChange={(e) => setFeedback(e.target.value)}
                  placeholder="Tell us about your experience, suggestions for improvement, or report any issues..."
                  className="w-full bg-white/10 border border-white/20 rounded-xl text-white placeholder-white/50 px-4 py-3 focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-transparent transition-all duration-300 backdrop-blur-sm resize-none"
                  rows="6"
                  disabled={loading}
                  required
                />
                <div className="text-right text-xs text-white/50 mt-1">
                  {feedback.length}/1000 characters
                </div>
              </div>

              {/* Optional Email */}
              <div>
                <label className="block text-sm font-medium text-white/90 mb-3">
                  Contact Email (optional)
                </label>
                <input
                  type="email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  placeholder="your.email@example.com (if you want us to follow up)"
                  className="w-full bg-white/10 border border-white/20 rounded-xl text-white placeholder-white/50 px-4 py-3 focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-transparent transition-all duration-300 backdrop-blur-sm"
                  disabled={loading}
                />
              </div>

              {/* Error Display */}
              {error && (
                <motion.div
                  className="bg-red-500/10 border border-red-500/20 text-red-300 p-4 rounded-xl"
                  initial={{ opacity: 0, scale: 0.95 }}
                  animate={{ opacity: 1, scale: 1 }}
                >
                  <div className="flex items-center space-x-2">
                    <span>⚠️</span>
                    <span>{error}</span>
                  </div>
                </motion.div>
              )}

              {/* Submit Buttons */}
              <div className="flex items-center justify-end space-x-4 pt-6 border-t border-white/10">
                <button
                  type="button"
                  onClick={handleClose}
                  disabled={loading}
                  className="px-6 py-3 text-white/70 hover:text-white transition-colors disabled:opacity-50"
                >
                  Cancel
                </button>
                <motion.button
                  type="submit"
                  disabled={!feedback.trim() || loading}
                  className={`px-8 py-3 rounded-xl font-semibold transition-all duration-300 shadow-lg ${
                    feedback.trim() && !loading
                      ? 'bg-gradient-to-r from-purple-500 to-pink-500 hover:from-purple-600 hover:to-pink-600 text-white'
                      : 'bg-gray-500/50 text-gray-300 cursor-not-allowed'
                  }`}
                  whileHover={feedback.trim() && !loading ? { scale: 1.05 } : {}}
                  whileTap={feedback.trim() && !loading ? { scale: 0.95 } : {}}
                >
                  {loading ? (
                    <div className="flex items-center space-x-2">
                      <motion.div
                        className="w-4 h-4 border-2 border-white border-t-transparent rounded-full"
                        animate={{ rotate: 360 }}
                        transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
                      />
                      <span>Sending...</span>
                    </div>
                  ) : (
                    <div className="flex items-center space-x-2">
                      <span>Send Feedback</span>
                      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8"></path>
                      </svg>
                    </div>
                  )}
                </motion.button>
              </div>
            </form>
          )}
        </motion.div>
      </motion.div>
    </AnimatePresence>
  );
};

export default FeedbackModal;