// // src/components/HistoryList.jsx
// import React, { useEffect, useState } from 'react';
// import api from '../utils/api';


// const HistoryList = () => {
//   const [history, setHistory] = useState([]);
//   const [loading, setLoading] = useState(true);
//   const [error, setError] = useState(null);

//   useEffect(() => {
//     const fetchHistory = async () => {
//       try {
//         setLoading(true);
//         // Fetch data from the backend
//         const res = await api.get('/api/history');


//         // --- IMPORTANT: Log the raw response to verify ---
//         console.log("Raw API response for history:", res.data);

//         // Process each history entry: parse the summary string into a JSON object
//         const parsedHistory = res.data.history.map(entry => {
//           let processedSummary = null; // Initialize to null

//           try {
//             // *** FIXED: Check if it's already an object first ***
//             if (entry.summary === null || entry.summary === undefined) {
//               // Handle null/undefined summary
//               processedSummary = null;
//               console.log("Summary is null/undefined for entry ID", entry.id);
//             } else if (typeof entry.summary === 'object' && entry.summary !== null) {
//               // If it's already an object, use it directly (this is the most common case)
//               processedSummary = entry.summary;
//               console.log("Using already-parsed summary for entry ID", entry.id, ":", processedSummary);
//             } else if (typeof entry.summary === 'string') {
//               // Only try to parse if it's a valid JSON string (not "[object Object]")
//               if (entry.summary.trim() === '' || entry.summary === '[object Object]') {
//                 console.warn("Invalid summary string for entry ID", entry.id, ":", entry.summary);
//                 processedSummary = null;
//               } else {
//                 processedSummary = JSON.parse(entry.summary);
//                 console.log("Parsed (from string) summary for entry ID", entry.id, ":", processedSummary);
//               }
//             } else {
//               // Handle any other unexpected types
//               console.warn("Unexpected summary type for entry ID", entry.id, ":", typeof entry.summary, entry.summary);
//               processedSummary = null;
//             }
//           } catch (e) {
//             // Log an error if processing fails (e.g., if there are truly malformed records)
//             console.error("Failed to parse summary for entry ID", entry.id, ": Raw summary string:", entry.summary, "Error:", e);
//             // Keep processedSummary as null or fallback to a default structure if needed
//             processedSummary = null;
//           }

//           return {
//             ...entry,
//             summary: processedSummary // Replace the summary (string or object) with the processed object
//           };
//         });

//         // --- IMPORTANT: Log the final state before setting ---
//         console.log("Final parsed history state to be set:", parsedHistory);

//         setHistory(parsedHistory); // Update the state with the parsed data
//         setError(null); // Clear any previous errors
//       } catch (err) {
//         // Handle any errors during the API call itself
//         console.error("Error fetching history from backend:", err);
//         setError("Failed to load reconciliation history. Please try again.");
//       } finally {
//         setLoading(false); // Set loading to false once fetching is complete
//       }
//     };

//     fetchHistory(); // Call the async function
//   }, []); // Empty dependency array means this runs once on component mount

//   // Conditional rendering based on loading and error states
//   if (loading) return <div className="p-4 mt-6 border rounded-2xl bg-white shadow-lg text-center">Loading past reconciliations...</div>;
//   if (error) return <div className="p-4 mt-6 border rounded-2xl bg-white shadow-lg text-center text-red-500">{error}</div>;
//   if (history.length === 0) return <div className="p-4 mt-6 border rounded-2xl bg-white shadow-lg text-center">No past reconciliations found. Perform a new reconciliation!</div>;

//   return (
//     <div className="p-4 mt-6 border rounded-2xl bg-white shadow-lg">
//       <h2 className="text-xl font-bold mb-4">Past Reconciliations</h2>
//       <div className="overflow-x-auto"> {/* Added for better responsiveness on small screens */}
//         <table className="min-w-full text-sm">
//           <thead className="bg-gray-100">
//             <tr>
//               <th className="p-2 text-left">#</th>
//               <th className="p-2 text-left">Date</th>
//               <th className="p-2 text-left">Matched</th>
//               <th className="p-2 text-left">Unmatched</th>
//               <th className="p-2 text-left">Uploader</th> {/* Added Uploader column */}
//               {/* Add more headers if needed */}
//             </tr>
//           </thead>
//           <tbody>
//             {history.map((entry) => ( // Removed idx as it's not strictly necessary for key if entry.id exists
//               <tr key={entry.id} className="even:bg-gray-50">
//                 <td className="p-2">{entry.id}</td> {/* Display actual ID */}
//                 <td className="p-2">{new Date(entry.created_at).toLocaleString()}</td>
//                 {/* Use optional chaining (?) in case summary or its properties are null */}
//                 <td className="p-2">{entry.summary?.matched?.length || 0}</td>
//                 <td className="p-2">
//                   {(entry.summary?.unmatched_ledger?.length || 0) +
//                    (entry.summary?.unmatched_bank?.length || 0)}
//                 </td>
//                 <td className="p-2">{entry.uploaded_by}</td> {/* Display uploader */}
//                 {/* Add more data cells */}
//               </tr>
//             ))}
//           </tbody>
//         </table> {/* Closing </table> tag */}
//       </div>
//     </div>
//   );
// };

// export default HistoryList;
// src/components/HistoryList.jsx - COMPLETE ENHANCED VERSION with No Duplicates

import { useState, useEffect, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { getReconciliationHistory } from '../utils/api';
import { useAuth } from '../contexts/AuthContext';

const HistoryList = ({ refreshTrigger }) => {
  const { user } = useAuth();
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const isBetaUser = user?.is_beta_user === true;

  const fetchHistory = useCallback(async () => {
    if (!user) {
      setLoading(false);
      return;
    }

    try {
      setLoading(true);
      setError(null);

      console.log('✅ HistoryList: Fetching reconciliation history...');

      // ✅ FIXED: Use correct API function with proper error handling
      const response = await getReconciliationHistory(0, 100);

      console.log('✅ HistoryList: Received history response:', response);

      // ✅ FIXED: Handle response structure and remove duplicates
      if (response && response.history) {
        // Remove duplicates by ID and sort by creation date
        const uniqueHistory = response.history.reduce((acc, item) => {
          if (!acc.find(existing => existing.id === item.id)) {
            acc.push(item);
          }
          return acc;
        }, []);

        // Sort by created_at descending
        uniqueHistory.sort((a, b) => new Date(b.created_at) - new Date(a.created_at));

        setHistory(uniqueHistory);
        console.log(`✅ HistoryList: Set ${uniqueHistory.length} unique history items`);
      } else {
        console.warn('⚠️ HistoryList: No history data in response');
        setHistory([]);
      }

    } catch (error) {
      console.error('❌ HistoryList: Error fetching history:', error);
      setError(error.message || 'Failed to load reconciliation history');
      setHistory([]);
    } finally {
      setLoading(false);
    }
  }, [user]);

  // Fetch history when component mounts or refreshTrigger changes
  useEffect(() => {
    fetchHistory();
  }, [fetchHistory, refreshTrigger]);

  const formatDate = (dateString) => {
    try {
      return new Date(dateString).toLocaleDateString('en-US', {
        year: 'numeric',
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit'
      });
    } catch {
      return 'Invalid Date';
    }
  };

  const getMatchRateColor = (rate) => {
    if (rate >= 90) return 'from-green-500 to-emerald-500';
    if (rate >= 70) return 'from-yellow-500 to-orange-500';
    return 'from-red-500 to-rose-500';
  };

  const getMatchRateEmoji = (rate) => {
    if (rate >= 90) return '🎯';
    if (rate >= 70) return '⚡';
    return '⚠️';
  };

  const getBetaBadge = () => {
    if (!isBetaUser) return null;

    return (
      <div className="bg-gradient-to-r from-yellow-400/20 to-yellow-500/20 text-yellow-300 px-3 py-1 rounded-full text-xs font-semibold border border-yellow-400/30 flex items-center gap-1">
        <span>✨</span>
        <span>Pro</span>
      </div>
    );
  };

  const HistoryItem = ({ item, index }) => {
    const matchRate = item.match_rate || 0;
    const totalCount = item.total_count || (item.matched_count + item.unmatched_count);

    return (
      <motion.div
        className="bg-white/5 rounded-xl p-6 border border-white/10 backdrop-blur-sm hover:bg-white/10 transition-all duration-300 group"
        initial={{ opacity: 0, y: 20, scale: 0.95 }}
        animate={{ opacity: 1, y: 0, scale: 1 }}
        transition={{ duration: 0.4, delay: index * 0.1 }}
        whileHover={{ scale: 1.02 }}
        key={`history-${item.id}`}
      >
        {/* Header Section */}
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center space-x-3">
            <div className="w-12 h-12 bg-gradient-to-r from-purple-500 to-pink-500 rounded-xl flex items-center justify-center shadow-lg">
              <span className="text-white text-sm font-bold">#{item.id}</span>
            </div>
            <div>
              <h4 className="text-white font-semibold text-lg group-hover:text-purple-300 transition-colors">
                Reconciliation #{item.id}
              </h4>
              <p className="text-white/60 text-sm flex items-center gap-2">
                <span>📅</span>
                <span>{formatDate(item.created_at)}</span>
              </p>
            </div>
            {getBetaBadge()}
          </div>

          {/* Match Rate Display */}
          <div className="text-right">
            <div className="text-white/60 text-xs mb-1">Match Rate</div>
            <div className="flex items-center gap-2">
              <span className="text-2xl">
                {getMatchRateEmoji(matchRate)}
              </span>
              <span className="text-2xl font-bold text-white">
                {Math.round(matchRate)}%
              </span>
            </div>
          </div>
        </div>

        {/* Stats Grid */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
          <motion.div
            className="flex items-center space-x-2 p-3 bg-green-500/10 rounded-lg border border-green-500/20"
            whileHover={{ scale: 1.05 }}
          >
            <span className="text-green-400 text-xl">✅</span>
            <div>
              <span className="text-white/80 text-xs block">Matched</span>
              <span className="text-green-400 font-bold text-lg">
                {item.matched_count || 0}
              </span>
            </div>
          </motion.div>

          <motion.div
            className="flex items-center space-x-2 p-3 bg-red-500/10 rounded-lg border border-red-500/20"
            whileHover={{ scale: 1.05 }}
          >
            <span className="text-red-400 text-xl">❌</span>
            <div>
              <span className="text-white/80 text-xs block">Unmatched</span>
              <span className="text-red-400 font-bold text-lg">
                {item.unmatched_count || 0}
              </span>
            </div>
          </motion.div>

          <motion.div
            className="flex items-center space-x-2 p-3 bg-blue-500/10 rounded-lg border border-blue-500/20"
            whileHover={{ scale: 1.05 }}
          >
            <span className="text-blue-400 text-xl">📊</span>
            <div>
              <span className="text-white/80 text-xs block">Total</span>
              <span className="text-blue-400 font-bold text-lg">
                {totalCount}
              </span>
            </div>
          </motion.div>

          <motion.div
            className="flex items-center space-x-2 p-3 bg-purple-500/10 rounded-lg border border-purple-500/20"
            whileHover={{ scale: 1.05 }}
          >
            <span className="text-purple-400 text-xl">👤</span>
            <div>
              <span className="text-white/80 text-xs block">Uploaded By</span>
              <span className="text-purple-400 font-bold text-sm truncate max-w-20">
                {(item.uploaded_by || user?.email || 'Unknown').split('@')[0]}
              </span>
            </div>
          </motion.div>
        </div>

        {/* Enhanced Progress Bar */}
        <div className="space-y-2">
          <div className="flex justify-between text-xs text-white/60">
            <span>Reconciliation Progress</span>
            <span>
              {totalCount > 0
                ? `${item.matched_count}/${totalCount} transactions matched`
                : 'No data available'
              }
            </span>
          </div>
          <div className="w-full bg-white/10 rounded-full h-4 overflow-hidden shadow-inner">
            <motion.div
              className={`bg-gradient-to-r ${getMatchRateColor(matchRate)} h-4 rounded-full flex items-center justify-end pr-3 shadow-lg`}
              initial={{ width: 0 }}
              animate={{
                width: totalCount > 0
                  ? `${Math.max(5, (item.matched_count / totalCount) * 100)}%`
                  : '0%'
              }}
              transition={{
                duration: 1.5,
                delay: index * 0.15,
                ease: "easeOut",
                type: "spring",
                stiffness: 100
              }}
            >
              {matchRate >= 30 && totalCount > 0 && (
                <motion.span
                  className="text-white text-xs font-bold drop-shadow-sm"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  transition={{ delay: (index * 0.15) + 0.8 }}
                >
                  {Math.round(matchRate)}%
                </motion.span>
              )}
            </motion.div>
          </div>

          {/* Additional Info Bar */}
          <div className="flex justify-between items-center text-xs text-white/50 pt-1">
            <span>
              {isBetaUser ? '✨ Enhanced with Pro features' : '📊 Standard reconciliation'}
            </span>
            <span>ID: {item.id}</span>
          </div>
        </div>

        {/* Hover Effects */}
        <motion.div
          className="absolute inset-0 bg-gradient-to-r from-purple-500/5 to-pink-500/5 rounded-xl opacity-0 group-hover:opacity-100 transition-opacity duration-300 pointer-events-none"
          initial={false}
        />
      </motion.div>
    );
  };

  // Loading State
  if (loading) {
    return (
      <div className="space-y-6">
        <div className="flex items-center justify-between mb-8">
          <div>
            <div className="h-8 bg-white/20 rounded w-64 animate-pulse mb-2"></div>
            <div className="h-4 bg-white/10 rounded w-48 animate-pulse"></div>
          </div>
          <div className="h-12 bg-white/10 rounded w-24 animate-pulse"></div>
        </div>

        {[...Array(3)].map((_, i) => (
          <motion.div
            key={`loading-${i}`}
            className="bg-white/5 rounded-xl p-6 border border-white/10 backdrop-blur-sm"
            animate={{
              opacity: [0.3, 0.7, 0.3],
            }}
            transition={{
              duration: 2,
              repeat: Infinity,
              delay: i * 0.3,
            }}
          >
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center space-x-4">
                <div className="w-12 h-12 bg-white/20 rounded-xl animate-pulse"></div>
                <div>
                  <div className="h-5 bg-white/20 rounded mb-2 w-48 animate-pulse"></div>
                  <div className="h-3 bg-white/10 rounded w-32 animate-pulse"></div>
                </div>
              </div>
              <div className="h-8 bg-white/10 rounded w-16 animate-pulse"></div>
            </div>
            <div className="grid grid-cols-4 gap-4 mb-6">
              {[...Array(4)].map((_, j) => (
                <div key={j} className="h-16 bg-white/10 rounded-lg animate-pulse"></div>
              ))}
            </div>
            <div className="h-4 bg-white/10 rounded animate-pulse"></div>
          </motion.div>
        ))}
      </div>
    );
  }

  // Error State
  if (error) {
    return (
      <motion.div
        className="bg-red-500/10 border border-red-500/20 text-red-300 p-12 rounded-2xl text-center"
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 0.4 }}
      >
        <motion.div
          className="text-red-400 text-6xl mb-6"
          animate={{
            rotate: [0, -10, 10, -10, 0],
            scale: [1, 1.1, 1]
          }}
          transition={{ duration: 2, repeat: Infinity, repeatDelay: 3 }}
        >
          ⚠️
        </motion.div>
        <h3 className="text-2xl font-bold mb-4">Failed to Load History</h3>
        <p className="mb-8 text-red-300/80 max-w-md mx-auto text-lg leading-relaxed">
          {error}
        </p>
        <motion.button
          onClick={fetchHistory}
          className="bg-red-500/20 hover:bg-red-500/30 border border-red-500/40 px-8 py-4 rounded-xl text-sm font-semibold transition-all duration-200 flex items-center gap-3 mx-auto"
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
        >
          <span>🔄</span>
          Try Again
        </motion.button>
      </motion.div>
    );
  }

  // Empty State
  if (history.length === 0) {
    return (
      <motion.div
        className="text-center py-20"
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 0.5 }}
      >
        <motion.div
          className="text-8xl mb-8"
          animate={{
            y: [0, -10, 0],
            rotate: [0, 5, -5, 0]
          }}
          transition={{
            duration: 4,
            repeat: Infinity,
            ease: "easeInOut"
          }}
        >
          📂
        </motion.div>
        <h3 className="text-3xl font-bold text-white mb-4">No History Yet</h3>
        <p className="text-white/60 mb-10 max-w-lg mx-auto text-lg leading-relaxed">
          {isBetaUser
            ? "🚀 Start your first Pro reconciliation with smart column detection to see your history here!"
            : "📊 Upload your ledger and bank statement files to run your first reconciliation and see the results here."
          }
        </p>
        <div className="flex flex-col sm:flex-row justify-center gap-4">
          <motion.button
            onClick={fetchHistory}
            className="bg-gradient-to-r from-purple-500 to-pink-500 hover:from-purple-600 hover:to-pink-600 text-white px-10 py-4 rounded-xl font-semibold text-lg transition-all duration-300 flex items-center gap-3 mx-auto sm:mx-0"
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
          >
            <span>🔄</span>
            Refresh History
          </motion.button>
          {isBetaUser && (
            <motion.div
              className="bg-yellow-500/10 border border-yellow-500/30 text-yellow-300 px-6 py-4 rounded-xl font-medium flex items-center gap-2 mx-auto sm:mx-0"
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.3 }}
            >
              <span>✨</span>
              <span>Pro features enabled</span>
            </motion.div>
          )}
        </div>
      </motion.div>
    );
  }

  // Main History List
  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between mb-8">
        <div>
          <h3 className="text-3xl font-bold text-white mb-2 flex items-center gap-3">
            <span>📊</span>
            <span>Recent Reconciliations</span>
          </h3>
          <p className="text-white/60 text-lg">
            {history.length} reconciliation{history.length !== 1 ? 's' : ''} found
            {isBetaUser && (
              <span className="ml-3 text-yellow-300 font-medium">✨ Enhanced with Pro features</span>
            )}
          </p>
        </div>
        <motion.button
          onClick={fetchHistory}
          className="bg-white/10 hover:bg-white/20 border border-white/20 text-white/80 hover:text-white px-8 py-4 rounded-xl font-semibold transition-all duration-200 flex items-center gap-3 text-lg"
          whileHover={{ scale: 1.02 }}
          whileTap={{ scale: 0.98 }}
        >
          <motion.span
            animate={{ rotate: loading ? 360 : 0 }}
            transition={{ duration: 1, repeat: loading ? Infinity : 0 }}
          >
            🔄
          </motion.span>
          <span>Refresh</span>
        </motion.button>
      </div>

      {/* History Items */}
      <AnimatePresence mode="wait">
        {history.map((item, index) => (
          <HistoryItem
            key={`history-item-${item.id}-${item.created_at}`}
            item={item}
            index={index}
          />
        ))}
      </AnimatePresence>

      {/* Footer */}
      {history.length >= 100 && (
        <motion.div
          className="text-center py-8 border-t border-white/10"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.5 }}
        >
          <p className="text-white/60">
            📋 Showing latest 100 reconciliations. Contact support for more history.
          </p>
        </motion.div>
      )}

      {/* Pro Features Reminder */}
      {!isBetaUser && history.length > 0 && (
        <motion.div
          className="bg-gradient-to-r from-yellow-500/10 to-orange-500/10 border border-yellow-500/20 rounded-2xl p-6 text-center"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.8 }}
        >
          <div className="text-3xl mb-3">✨</div>
          <h4 className="text-xl font-semibold text-yellow-300 mb-2">
            Upgrade to Pro for Enhanced Features
          </h4>
          <p className="text-yellow-200/80">
            Get smart column detection, higher accuracy, and advanced reconciliation features.
          </p>
        </motion.div>
      )}
    </div>
  );
};

export default HistoryList;