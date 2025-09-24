// src/pages/BetaUserDashboard.jsx
import { useEffect, useState } from 'react';
import api from '../utils/api';
import { getUsageStats, submitBetaFeedback } from '../utils/api';
import { useNavigate } from 'react-router-dom';

const BetaUserDashboard = () => {
  const [stats, setStats] = useState(null);
  const [feedback, setFeedback] = useState('');
  const [submitted, setSubmitted] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const navigate = useNavigate();

  useEffect(() => {
    const fetchStats = async () => {
      try {
        setError(null);
        const response = await getUsageStats();
        // FIXED: Handle both response.data and direct response
        const data = response.data || response;
        setStats(data);
      } catch (error) {
        console.error('Failed to load usage stats:', error);
        setError('Failed to load usage statistics. Please try again.');
      } finally {
        setLoading(false);
      }
    };
    fetchStats();
  }, []);

  const submitFeedback = async () => {
    try {
      await submitBetaFeedback(feedback);
      setSubmitted(true);
      setFeedback('');
      // Reset submitted state after 3 seconds
      setTimeout(() => setSubmitted(false), 3000);
    } catch (error) {
      console.error('Feedback failed:', error);
      alert('Failed to submit feedback. Please try again.');
    }
  };

  const refreshStats = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await getUsageStats();
      const data = response.data || response;
      setStats(data);
    } catch (error) {
      console.error('Failed to refresh stats:', error);
      setError('Failed to refresh statistics. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-4xl mx-auto py-10 px-4 bg-gray-50 min-h-screen">
      <div className="flex justify-between items-center mb-6">
        <h1 className="text-3xl font-bold text-gray-900">Beta Access Dashboard</h1>
        <div className="flex gap-2">
          <button
            onClick={refreshStats}
            disabled={loading}
            className="bg-blue-500 hover:bg-blue-600 disabled:bg-blue-300 text-white px-4 py-2 rounded-lg font-medium transition-colors"
          >
            {loading ? 'Refreshing...' : 'Refresh'}
          </button>
          <button
            onClick={() => navigate(-1)}
            className="bg-gray-200 hover:bg-gray-300 text-gray-800 px-4 py-2 rounded-lg font-medium transition-colors"
          >
            ← Back
          </button>
        </div>
      </div>

      {loading ? (
        <div className="text-center py-12">
          <div className="inline-block animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500"></div>
          <p className="text-gray-600 mt-4">Loading usage data...</p>
        </div>
      ) : error ? (
        <div className="bg-red-50 border border-red-200 text-red-800 p-6 rounded-lg text-center">
          <p className="font-medium">{error}</p>
          <button
            onClick={refreshStats}
            className="mt-3 bg-red-600 hover:bg-red-700 text-white px-4 py-2 rounded-lg font-medium transition-colors"
          >
            Try Again
          </button>
        </div>
      ) : stats ? (
        <div className="space-y-8">
          {/* Plan Information */}
          <div className="bg-white shadow-lg rounded-xl p-6 border border-gray-200">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-xl font-bold text-gray-900">Your Plan</h2>
              <span className="bg-yellow-100 text-yellow-800 text-sm font-semibold px-3 py-1 rounded-full">
                {stats.plan_type || 'Beta Pro'}
              </span>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <p className="text-sm text-gray-600">Plan Type</p>
                <p className="text-lg font-semibold text-gray-900">{stats.plan_type || 'Beta Pro'}</p>
              </div>
              {stats.plan_expires_at && (
                <div>
                  <p className="text-sm text-gray-600">Expires</p>
                  <p className="text-lg font-semibold text-gray-900">
                    {new Date(stats.plan_expires_at).toLocaleDateString()}
                  </p>
                </div>
              )}
            </div>
          </div>

          {/* Usage Statistics - FIXED: Show actual transaction counts */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {[
              {
                key: 'reconciliations',
                title: 'Reconciliations',
                icon: '📊',
                color: 'blue'
              },
              {
                key: 'transactions',
                title: 'Total Transactions',
                icon: '💳',
                color: 'green'
              },
              {
                key: 'ai_queries',
                title: 'AI Queries',
                icon: '🤖',
                color: 'purple'
              }
            ].map(({ key, title, icon, color }) => {
              const usage = stats[key] || { used: 0, limit: null };
              const colorClasses = {
                blue: 'bg-blue-50 border-blue-200 text-blue-900',
                green: 'bg-green-50 border-green-200 text-green-900',
                purple: 'bg-purple-50 border-purple-200 text-purple-900'
              };

              return (
                <div key={key} className={`${colorClasses[color]} border rounded-xl p-6 shadow-sm`}>
                  <div className="flex items-center justify-between mb-3">
                    <span className="text-2xl">{icon}</span>
                    <span className="text-xs font-medium uppercase tracking-wide opacity-70">
                      {title}
                    </span>
                  </div>
                  <div className="space-y-1">
                    <p className="text-3xl font-bold">
                      {usage.used?.toLocaleString() || 0}
                    </p>
                    <p className="text-sm opacity-70">
                      {usage.limit !== null ? (
                        <>of {usage.limit.toLocaleString()} limit</>
                      ) : (
                        'Unlimited'
                      )}
                    </p>
                    {usage.limit !== null && (
                      <div className="w-full bg-white/50 rounded-full h-2 mt-2">
                        <div
                          className={`h-2 rounded-full ${
                            color === 'blue' ? 'bg-blue-500' :
                            color === 'green' ? 'bg-green-500' : 'bg-purple-500'
                          }`}
                          style={{
                            width: `${Math.min((usage.used / usage.limit) * 100, 100)}%`
                          }}
                        />
                      </div>
                    )}
                  </div>
                </div>
              );
            })}
          </div>

          {/* Additional Statistics */}
          {stats.additional_stats && (
            <div className="bg-white shadow-lg rounded-xl p-6 border border-gray-200">
              <h3 className="text-lg font-bold text-gray-900 mb-4">Detailed Statistics</h3>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                {Object.entries(stats.additional_stats).map(([key, value]) => (
                  <div key={key} className="text-center">
                    <p className="text-2xl font-bold text-gray-900">{value}</p>
                    <p className="text-sm text-gray-600 capitalize">
                      {key.replace(/_/g, ' ')}
                    </p>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Feedback Section */}
          <div className="bg-white shadow-lg rounded-xl p-6 border border-gray-200">
            <h2 className="text-xl font-bold text-gray-900 mb-4 flex items-center gap-2">
              <span>💭</span>
              Send Feedback
            </h2>
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Your feedback helps us improve the platform:
                </label>
                <textarea
                  className="w-full border border-gray-300 rounded-lg p-3 text-sm focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-vertical"
                  rows={5}
                  placeholder="Share your thoughts, suggestions, or report any issues..."
                  value={feedback}
                  onChange={e => setFeedback(e.target.value)}
                />
              </div>
              <div className="flex items-center justify-between">
                <p className="text-xs text-gray-500">
                  Your feedback is anonymous and helps us improve MatchLedger AI
                </p>
                <button
                  onClick={submitFeedback}
                  disabled={!feedback.trim() || submitted}
                  className="bg-blue-600 hover:bg-blue-700 disabled:bg-gray-300 text-white px-6 py-2 rounded-lg font-medium transition-colors disabled:cursor-not-allowed"
                >
                  {submitted ? '✓ Submitted!' : 'Submit Feedback'}
                </button>
              </div>
              {submitted && (
                <div className="bg-green-50 border border-green-200 text-green-800 p-3 rounded-lg">
                  <p className="text-sm font-medium">Thank you! Your feedback was submitted successfully.</p>
                </div>
              )}
            </div>
          </div>

          {/* Beta Program Info */}
          <div className="bg-gradient-to-r from-yellow-50 to-orange-50 border border-yellow-200 rounded-xl p-6">
            <div className="flex items-start gap-4">
              <span className="text-3xl">✨</span>
              <div>
                <h3 className="text-lg font-bold text-gray-900 mb-2">Beta Program Benefits</h3>
                <ul className="text-sm text-gray-700 space-y-1">
                  <li>• Early access to new features</li>
                  <li>• Enhanced AI-powered reconciliation</li>
                  <li>• Priority customer support</li>
                  <li>• Direct influence on product development</li>
                  <li>• Advanced analytics and reporting</li>
                </ul>
                <p className="text-xs text-gray-600 mt-3">
                  Thank you for being part of our beta program! Your usage and feedback help us build a better product.
                </p>
              </div>
            </div>
          </div>
        </div>
      ) : null}
    </div>
  );
};

export default BetaUserDashboard;