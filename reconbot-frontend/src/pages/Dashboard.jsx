// // src/pages/Dashboard.jsx
// import { useState } from 'react';
// import FileUploader from '../components/FileUploader';
// import ResultsTable from '../components/ResultsTable';
// import { SummaryCard } from '../components/SummaryCard';
// import { reconcileFiles } from '../utils/api';
// import ChatBox from '../components/ChatBox';
// import HistoryList from '../components/HistoryList';
// import { useAuth } from '../contexts/AuthContext';

// const Dashboard = () => {
//   const { logout, user, loading } = useAuth();
//   const [data, setData] = useState(null);
//   const [ledgerFile, setLedgerFile] = useState(null);
//   const [bankFile, setBankFile] = useState(null);

//   const handleUpload = async (uploadedLedgerFile, uploadedBankFile, email) => {
//     setLedgerFile(uploadedLedgerFile);
//     setBankFile(uploadedBankFile);

//     const res = await reconcileFiles(uploadedLedgerFile, uploadedBankFile, email);
//     setData(res);
//   };

//   return (
//     <div className="min-h-screen bg-gray-50">
//       {/* Mobile-friendly header */}
//       <div className="bg-white shadow-sm border-b">
//         <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
//           <div className="flex justify-between items-center py-4">
//             <h1 className="text-2xl md:text-3xl font-bold text-gray-900">ReconBot</h1>
//             <div className="flex items-center space-x-2 md:space-x-4">
//               <span className="text-sm md:text-base text-gray-600 truncate max-w-32 md:max-w-none">
//                 {user?.email}
//               </span>
//               <button
//                 onClick={logout}
//                 className="bg-red-500 hover:bg-red-600 text-white px-3 py-2 md:px-4 rounded-lg text-sm font-medium transition-colors"
//               >
//                 Logout
//               </button>
//             </div>
//           </div>
//         </div>
//       </div>

//       {/* Main content */}
//       <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6 space-y-6">

//         {/* File Uploader - responsive */}
//         <div className="bg-white rounded-lg shadow-sm border">
//           <FileUploader onUpload={handleUpload} />
//         </div>

//         {/* Results section - responsive grid */}
//         {data && data.summary && (
//           <div className="space-y-6">
//             {/* Summary Cards */}
//             <SummaryCard
//               matched={data.summary.matched}
//               unmatchedLedger={data.summary.unmatched_ledger}
//               unmatchedBank={data.summary.unmatched_bank}
//             />

//             {/* Results Tables - stacked on mobile */}
//             <div className="grid grid-cols-1 gap-6">
//               <div className="bg-white rounded-lg shadow-sm border overflow-hidden">
//                 <ResultsTable title="Matched Transactions" data={data.summary.matched} />
//               </div>
//               <div className="bg-white rounded-lg shadow-sm border overflow-hidden">
//                 <ResultsTable title="Unmatched Ledger Entries" data={data.summary.unmatched_ledger} />
//               </div>
//               <div className="bg-white rounded-lg shadow-sm border overflow-hidden">
//                 <ResultsTable title="Unmatched Bank Transactions" data={data.summary.unmatched_bank} />
//               </div>
//             </div>
//           </div>
//         )}

//         {/* Chat section - full width on mobile */}
//         {ledgerFile && bankFile && (
//           <div className="bg-white rounded-lg shadow-sm border">
//             <div className="p-4 border-b">
//               <h2 className="text-xl md:text-2xl font-bold text-gray-900">Chat with your Data</h2>
//             </div>
//             <ChatBox ledgerFile={ledgerFile} bankFile={bankFile} />
//           </div>
//         )}

//         {/* History section */}
//         {!loading && user && (
//           <div className="bg-white rounded-lg shadow-sm border">
//             <HistoryList />
//           </div>
//         )}
//       </div>
//     </div>
//   );
// };

// export default Dashboard;

// // src/pages/Dashboard.jsx
// import React, { useState, useEffect } from 'react';
// import { useDarkMode } from '../App';
// import FileUploader from '../components/FileUploader';
// import ChatBox from '../components/ChatBox';
// import FeedbackForm from '../components/FeedbackForm';
// import Header from '../components/Header';
// import ResultsTable from '../components/ResultsTable';
// import { SummaryCard } from '../components/SummaryCard';
// import { reconcileFiles } from '../utils/api';
// import HistoryList from '../components/HistoryList';
// import { useAuth } from '../contexts/AuthContext';
// import matchledgerLogo from '../assets/favicon1.png';

// const Dashboard = () => {
//   const { logout, user } = useAuth();
//   const { darkMode, toggleDarkMode } = useDarkMode();
//   const [reconciliationData, setReconciliationData] = useState(null);
//   const [ledgerFile, setLedgerFile] = useState(null);
//   const [bankFile, setBankFile] = useState(null);
//   const [isReconciling, setIsReconciling] = useState(false);
//   const [reconciliationError, setReconciliationError] = useState(null);
//   const [isFeedbackFormOpen, setIsFeedbackFormOpen] = useState(false);

//   // Effect to show feedback prompt after successful reconciliation
//   useEffect(() => {
//     if (reconciliationData && !isReconciling && !reconciliationError && !isFeedbackFormOpen) {
//       const timer = setTimeout(() => {
//         setIsFeedbackFormOpen(true);
//       }, 2000);

//       return () => clearTimeout(timer);
//     }
//   }, [reconciliationData, isReconciling, reconciliationError, isFeedbackFormOpen]);

//   // Unified upload and reconciliation logic
//   const handleUpload = async (uploadedLedgerFile, uploadedBankFile) => {
//     setReconciliationError(null);
//     setIsReconciling(true);
//     setReconciliationData(null); // Clear previous data
//     setLedgerFile(uploadedLedgerFile);
//     setBankFile(uploadedBankFile);
//     setIsFeedbackFormOpen(false);

//     try {
//       const res = await reconcileFiles(uploadedLedgerFile, uploadedBankFile, user?.email);
//       console.log("API response from reconcileFiles:", res); // Log the response to inspect its structure
//       setReconciliationData(res);
//     } catch (error) {
//       console.error("Reconciliation error:", error);
//       setReconciliationError(`Failed to reconcile files: ${error.message || 'Unknown error.'}`);
//     } finally {
//       setIsReconciling(false);
//     }
//   };

//   const handleFeedbackSubmit = (feedbackData) => {
//     console.log('Feedback received:', feedbackData);
//   };

//   return (
//     <div className="min-h-screen flex flex-col bg-gray-100 dark:bg-gray-900 text-gray-900 dark:text-gray-100 transition-colors duration-200">
//       {/* Header component with logout and dark mode toggle */}
//       <Header
//         toggleDarkMode={toggleDarkMode}
//         darkMode={darkMode}
//         userEmail={user?.email}
//         onLogout={logout}
//         logoSrc={matchledgerLogo}
//       />

//       <main className="flex-grow container mx-auto p-4 md:p-6 lg:p-8">
//         <div className="space-y-6">
//           {/* File Uploader Section */}
//           <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 transition-colors duration-200">
//             <FileUploader
//               onUpload={handleUpload}
//               isReconciling={isReconciling}
//             />
//           </div>

//           {/* Reconciliation Loading/Error States */}
//           {reconciliationError && (
//             <div className="p-4 mt-6 border rounded-2xl bg-red-100 dark:bg-red-900/20 shadow-lg text-center text-red-700 dark:text-red-300 border-red-200 dark:border-red-700/50 transition-colors duration-200">
//               {reconciliationError}
//             </div>
//           )}

//           {isReconciling && (
//             <div className="p-4 mt-6 border rounded-2xl bg-blue-50 dark:bg-blue-900/20 shadow-lg text-center flex items-center justify-center h-32 border-blue-200 dark:border-blue-700/50 transition-colors duration-200">
//               <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 dark:border-blue-400 mr-3"></div>
//               <span className="text-lg text-blue-700 dark:text-blue-300">Reconciling your files... This may take a moment.</span>
//             </div>
//           )}

//           {/* Reconciliation Results Section */}
//           {reconciliationData && reconciliationData.summary && !isReconciling && (
//             <div className="space-y-6">
//               {/* Summary Cards */}
//               <SummaryCard
//                 matched={reconciliationData.summary.matched || []} // Safeguard
//                 unmatchedLedger={reconciliationData.summary.unmatched_ledger || []} // Safeguard
//                 unmatchedBank={reconciliationData.summary.unmatched_bank || []} // Safeguard
//               />

//               {/* Results Tables */}
//               <div className="grid grid-cols-1 gap-6">
//                 <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 overflow-hidden transition-colors duration-200">
//                   {/* Add || [] to ensure an array is always passed */}
//                   <ResultsTable title="Matched Transactions" data={reconciliationData.matched_transactions || []} />
//                 </div>
//                 <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 overflow-hidden transition-colors duration-200">
//                   {/* Add || [] to ensure an array is always passed */}
//                   <ResultsTable title="Unmatched Ledger Entries" data={reconciliationData.unmatched_ledger_entries || []} />
//                 </div>
//                 <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 overflow-hidden transition-colors duration-200">
//                   {/* Add || [] to ensure an array is always passed */}
//                   <ResultsTable title="Unmatched Bank Transactions" data={reconciliationData.unmatched_bank_transactions || []} />
//                 </div>
//               </div>
//             </div>
//           )}

//           {/* Chat Section */}
//           {ledgerFile && bankFile && reconciliationData && (
//             <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 transition-colors duration-200">
//               <div className="p-4 border-b border-gray-200 dark:border-gray-700">
//                 <h2 className="text-xl md:text-2xl font-bold text-gray-900 dark:text-white">Chat with your Data</h2>
//               </div>
//               <ChatBox
//                 ledgerFile={ledgerFile}
//                 bankFile={bankFile}
//                 reconciliationData={reconciliationData}
//               />
//             </div>
//           )}

//           {/* History Section */}
//           {user && (
//             <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 transition-colors duration-200">
//               <HistoryList />
//             </div>
//           )}
//         </div>
//       </main>

//       {/* Feedback Form Modal */}
//       <FeedbackForm
//         isOpen={isFeedbackFormOpen}
//         onClose={() => setIsFeedbackFormOpen(false)}
//         onSubmit={handleFeedbackSubmit}
//       />
//     </div>
//   );
// };

// export default Dashboard;
//23-0825
import { useState, useRef, useEffect, useCallback } from 'react';
import { Link } from 'react-router-dom';
import { motion, useMotionValue, useSpring } from 'framer-motion';
import FileUploader from '../components/FileUploader';
import ResultsTable from '../components/ResultsTable';
import { SummaryCard } from '../components/SummaryCard';
import { startReconciliation, getReconciliationStatus } from '../utils/api';  // ✅ USING CORRECT API
import ChatBox from '../components/ChatBox';
import ChatErrorBoundary from '../components/ChatErrorBoundary';
import FeedbackModal from '../components/FeedbackModal';
import HistoryList from '../components/HistoryList';
import { useAuth } from '../contexts/AuthContext';
import api from '../utils/api';
import { getUsageStats } from '../utils/api';

// ✅ REMOVED DUPLICATE API FUNCTIONS - Using the working ones from utils/api

// Animated Background Component
const AnimatedBackground = () => {
  const [windowSize, setWindowSize] = useState({ width: 1920, height: 1080 });

  useEffect(() => {
    const updateWindowSize = () => {
      setWindowSize({
        width: window.innerWidth,
        height: window.innerHeight
      });
    };

    updateWindowSize();
    window.addEventListener('resize', updateWindowSize);
    return () => window.removeEventListener('resize', updateWindowSize);
  }, []);

  return (
    <div className="fixed inset-0 -z-50 overflow-hidden">
      {/* Gradient Background */}
      <div className="absolute inset-0 bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900" />

      {/* Animated Particles */}
      {[...Array(20)].map((_, i) => (
        <motion.div
          key={i}
          className="absolute w-1 h-1 bg-white rounded-full opacity-20"
          initial={{
            x: Math.random() * windowSize.width,
            y: Math.random() * windowSize.height,
          }}
          animate={{
            x: Math.random() * windowSize.width,
            y: Math.random() * windowSize.height,
          }}
          transition={{
            duration: Math.random() * 20 + 10,
            repeat: Infinity,
            repeatType: "reverse",
          }}
        />
      ))}

      {/* Gradient Orbs */}
      <div className="absolute top-1/4 left-1/4 w-64 h-64 bg-purple-500/10 rounded-full blur-3xl animate-pulse" />
      <div className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-blue-500/10 rounded-full blur-3xl animate-pulse delay-1000" />
    </div>
  );
};

// Modern Metric Card Component
const MetricCard = ({ title, value, subtitle, icon, color = "purple", loading = false }) => {
  const cardRef = useRef();

  const colorMap = {
    purple: "from-purple-500 to-pink-500",
    blue: "from-blue-500 to-cyan-500",
    green: "from-green-500 to-emerald-500",
    orange: "from-orange-500 to-red-500",
  };

  return (
    <motion.div
      ref={cardRef}
      className={`relative p-6 rounded-2xl bg-gradient-to-br ${colorMap[color]} text-white shadow-2xl backdrop-blur-sm border border-white/10`}
      whileHover={{
        scale: 1.05,
        rotate: 1,
        transition: { duration: 0.3, ease: "easeOut" }
      }}
      whileTap={{ scale: 0.95 }}
      initial={{ opacity: 0, y: 50 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.6, ease: "easeOut" }}
    >
      <div className="absolute inset-0 bg-gradient-to-br from-white/20 to-transparent rounded-2xl" />
      <div className="relative z-10">
        <div className="flex items-center justify-between mb-3">
          <div className="text-sm font-medium opacity-90">{title}</div>
          {icon && <div className="text-2xl">{icon}</div>}
        </div>
        <div className="text-3xl font-bold mb-1">
          {loading ? (
            <div className="flex items-center space-x-2">
              <motion.div
                className="w-6 h-6 border-2 border-white/30 border-t-white rounded-full"
                animate={{ rotate: 360 }}
                transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
              />
              <span className="text-lg">Loading...</span>
            </div>
          ) : (
            value || '0'
          )}
        </div>
        <div className="text-xs opacity-75">{subtitle}</div>
      </div>
    </motion.div>
  );
};

// Enhanced Professional Header Component with User Menu
const Header = ({ user, logout }) => {
  const [showUserMenu, setShowUserMenu] = useState(false);
  const menuRef = useRef(null);

  // Fix the admin check - use proper role checking
  const isActualAdmin = user?.role === 'admin' || user?.is_admin === true;
  const isBetaUser = user?.is_beta_user === true;

  useEffect(() => {
    const handleClickOutside = (event) => {
      if (menuRef.current && !menuRef.current.contains(event.target)) {
        setShowUserMenu(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  return (
    <motion.div
      className="relative backdrop-blur-xl bg-white/5 border-b border-white/10"
      initial={{ y: -100 }}
      animate={{ y: 0 }}
      transition={{ duration: 0.8, ease: "easeOut" }}
      style={{ zIndex: 1000 }}
    >
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center py-6">
          <motion.div
            className="flex items-center space-x-4"
            whileHover={{ scale: 1.02 }}
          >
            <div className="flex items-center space-x-3">
              <div className="w-10 h-10 bg-gradient-to-br from-purple-500 to-pink-500 rounded-xl flex items-center justify-center">
                <span className="text-white font-bold text-lg">M</span>
              </div>
              <h1 className="text-3xl font-bold bg-gradient-to-r from-white to-purple-200 bg-clip-text text-transparent">
                MatchLedger AI
              </h1>
            </div>

            {isBetaUser && (
              <motion.div
                className="bg-gradient-to-r from-yellow-400 to-orange-500 text-white px-4 py-2 rounded-full text-sm font-semibold shadow-lg"
                animate={{ scale: [1, 1.1, 1] }}
                transition={{ duration: 2, repeat: Infinity }}
              >
                ✨ Beta Pro
              </motion.div>
            )}
          </motion.div>

          <div className="relative" ref={menuRef}>
            {/* User Menu Button */}
            <motion.button
              onClick={() => setShowUserMenu(!showUserMenu)}
              className="flex items-center space-x-3 bg-white/5 rounded-full px-4 py-2 backdrop-blur-sm border border-white/10 hover:bg-white/10 transition-all duration-300"
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
            >
              <div className="w-8 h-8 bg-gradient-to-br from-purple-500 to-pink-500 rounded-full flex items-center justify-center">
                <span className="text-white text-sm font-bold">
                  {user?.email?.[0]?.toUpperCase() || 'U'}
                </span>
              </div>
              <span className="text-sm text-white/90 font-medium hidden sm:block">
                {user?.email || 'User'}
              </span>
              {isBetaUser && (
                <span className="bg-yellow-400/20 text-yellow-300 text-xs font-semibold px-2 py-1 rounded-full border border-yellow-400/30">
                  Beta
                </span>
              )}
              {isActualAdmin && (
                <span className="bg-purple-400/20 text-purple-300 text-xs font-semibold px-2 py-1 rounded-full border border-purple-400/30">
                  Admin
                </span>
              )}
              <svg
                className={`w-4 h-4 text-white/70 transition-transform ${showUserMenu ? 'rotate-180' : ''}`}
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 9l-7 7-7-7" />
              </svg>
            </motion.button>

            {/* Dropdown Menu */}
            {showUserMenu && (
              <motion.div
                className="absolute right-0 mt-2 w-72 bg-white/95 backdrop-blur-xl border border-white/20 rounded-2xl shadow-2xl overflow-hidden z-50"
                style={{ zIndex: 1001 }}
                initial={{ opacity: 0, scale: 0.95, y: -10 }}
                animate={{ opacity: 1, scale: 1, y: 0 }}
                exit={{ opacity: 0, scale: 0.95, y: -10 }}
                transition={{ duration: 0.2 }}
              >
                <div className="p-6">
                  {/* User Info */}
                  <div className="border-b border-gray-200/20 pb-4 mb-4">
                    <p className="text-gray-800 font-semibold">{user?.email}</p>
                    <div className="flex flex-wrap gap-2 mt-2">
                      {isBetaUser && (
                        <span className="bg-yellow-100 text-yellow-800 text-xs font-semibold px-2 py-1 rounded-full">
                          Beta Pro
                        </span>
                      )}
                      {isActualAdmin && (
                        <span className="bg-purple-100 text-purple-800 text-xs font-semibold px-2 py-1 rounded-full">
                          Admin
                        </span>
                      )}
                    </div>
                  </div>

                  {/* Menu Items */}
                  <div className="space-y-2">
                    {/* Beta Dashboard */}
                    {isBetaUser && (
                      <Link
                        to="/beta-dashboard"
                        className="flex items-center space-x-3 text-gray-700 hover:text-purple-600 hover:bg-purple-50/50 p-3 rounded-lg transition-colors w-full text-left"
                        onClick={() => setShowUserMenu(false)}
                      >
                        <span>✨</span>
                        <span>Beta Dashboard</span>
                      </Link>
                    )}

                    {/* Privacy Policy */}
                    <Link
                      to="/privacy"
                      className="flex items-center space-x-3 text-gray-700 hover:text-purple-600 hover:bg-purple-50/50 p-3 rounded-lg transition-colors w-full text-left"
                      onClick={() => setShowUserMenu(false)}
                    >
                      <span>🔒</span>
                      <span>Privacy Policy</span>
                    </Link>

                    {/* User Guide */}
                    <Link
                      to="/guide"
                      className="flex items-center space-x-3 text-gray-700 hover:text-purple-600 hover:bg-purple-50/50 p-3 rounded-lg transition-colors w-full text-left"
                      onClick={() => setShowUserMenu(false)}
                    >
                      <span>📖</span>
                      <span>User Guide</span>
                    </Link>

                    {/* Admin Feedback - Only for actual admins */}
                    {isActualAdmin && (
                      <Link
                        to="/admin/feedback"
                        className="flex items-center space-x-3 text-gray-700 hover:text-purple-600 hover:bg-purple-50/50 p-3 rounded-lg transition-colors w-full text-left"
                        onClick={() => setShowUserMenu(false)}
                      >
                        <span>👑</span>
                        <span>Admin Feedback</span>
                      </Link>
                    )}

                    <div className="border-t border-gray-200/20 pt-2 mt-2">
                      <button
                        onClick={() => {
                          setShowUserMenu(false);
                          logout();
                        }}
                        className="flex items-center space-x-3 text-red-600 hover:text-red-700 hover:bg-red-50/50 p-3 rounded-lg transition-colors w-full text-left"
                      >
                        <span>🚪</span>
                        <span>Sign Out</span>
                      </button>
                    </div>
                  </div>
                </div>
              </motion.div>
            )}
          </div>
        </div>
      </div>
    </motion.div>
  );
};

// Status Alert Component
const StatusAlert = ({ jobStatus, jobError }) => {
  if (!jobStatus && !jobError) return null;

  return (
    <motion.div
      className={`text-center p-6 rounded-2xl backdrop-blur-sm border ${
        jobError
          ? 'bg-red-500/10 text-red-300 border-red-500/20'
          : 'bg-blue-500/10 text-blue-300 border-blue-500/20'
      }`}
      initial={{ opacity: 0, scale: 0.8 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.5 }}
    >
      {jobError ? (
        <div>
          <p className="font-semibold mb-1">Error:</p>
          <p>{jobError}</p>
        </div>
      ) : (
        <motion.p
          className="font-semibold"
          animate={{ opacity: [1, 0.5, 1] }}
          transition={{ duration: 1.5, repeat: Infinity }}
        >
          {jobStatus}
        </motion.p>
      )}
    </motion.div>
  );
};

const Dashboard = () => {
  const { logout, user, loading } = useAuth();

  const [data, setData] = useState(null);
  const [ledgerFile, setLedgerFile] = useState(null);
  const [bankFile, setBankFile] = useState(null);
  const [jobId, setJobId] = useState(null);
  const [jobStatus, setJobStatus] = useState('');
  const [jobError, setJobError] = useState(null);

  // Live metrics state
  const [stats, setStats] = useState(null);
  const [statsLoading, setStatsLoading] = useState(true);
  const [statsError, setStatsError] = useState(null);

  // Feedback modal state
  const [showFeedbackModal, setShowFeedbackModal] = useState(false);

  // Fix admin and beta user checks
  const isActualAdmin = user?.role === 'admin' || user?.is_admin === true;
  const isBetaUser = user?.is_beta_user === true;

  // Fetch live statistics for beta users
  const fetchStats = useCallback(async () => {
    if (!user || !isBetaUser) {
      setStatsLoading(false);
      return;
    }

    try {
      setStatsLoading(true);
      setStatsError(null);
      const response = await getUsageStats();
      setStats(response.data || response);
    } catch (error) {
      setStatsError('Failed to load statistics');
      console.error('Error fetching stats:', error);
    } finally {
      setStatsLoading(false);
    }
  }, [user, isBetaUser]);

  // Fetch stats on component mount and when user changes
  useEffect(() => {
    if (user && isBetaUser) {
      fetchStats();
    }
  }, [user, isBetaUser, fetchStats]);

  const pollJobStatus = async (taskId) => {
    try {
      const statusResult = await getReconciliationStatus(taskId);
      const state = statusResult.state;
      const info = statusResult.info;

      if (state === 'SUCCESS') {
        const finalResult = statusResult.result;
        setData(finalResult);
        setJobId(null);
        setJobStatus('');
        // Refresh stats after successful reconciliation
        if (isBetaUser) {
          setTimeout(fetchStats, 1000);
        }
      } else if (state === 'FAILURE') {
        setJobError(info?.error_message || 'An unknown error occurred.');
        setJobId(null);
        setJobStatus('');
      } else {
        setJobStatus(info?.status || state);
        setTimeout(() => pollJobStatus(taskId), 3000);
      }
    } catch (error) {
      setJobError('Failed to get job status.');
      setJobId(null);
      setJobStatus('');
    }
  };

  // ✅ FIXED: Using the correct handleUpload function from working version
  const handleUpload = async (uploadedLedgerFile, uploadedBankFile, email, existingTaskId = null) => {
    setData(null);
    setJobId(null);
    setJobError(null);
    setJobStatus('Starting job...');

    setLedgerFile(uploadedLedgerFile);
    setBankFile(uploadedBankFile);

    try {
      let taskId = existingTaskId;

      if (!taskId) {
        // This is a regular upload - start basic reconciliation
        const response = await startReconciliation(uploadedLedgerFile, uploadedBankFile, email);
        taskId = response.task_id;
      }

      // For both basic and enhanced reconciliation, poll the same way
      setJobId(taskId);
      pollJobStatus(taskId);

    } catch (error) {
      setJobError(error?.message || 'Failed to start reconciliation');
      setJobStatus('');
    }
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 flex items-center justify-center">
        <motion.div
          className="text-center"
          animate={{ scale: [1, 1.1, 1] }}
          transition={{ duration: 2, repeat: Infinity }}
        >
          <div className="w-16 h-16 bg-gradient-to-r from-purple-500 to-pink-500 rounded-full flex items-center justify-center mx-auto mb-4">
            <motion.div
              className="w-8 h-8 border-2 border-white border-t-transparent rounded-full"
              animate={{ rotate: 360 }}
              transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
            />
          </div>
          <p className="text-white font-medium">Loading your workspace...</p>
        </motion.div>
      </div>
    );
  }

  // Don't render if user is not loaded
  if (!user) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 flex items-center justify-center">
        <p className="text-white">Please log in to access the dashboard.</p>
      </div>
    );
  }

  return (
    <div className="min-h-screen text-white relative">
      <AnimatedBackground />

      <Header user={user} logout={logout} />

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-8">

        {/* Welcome Section */}
        <motion.section
          className="text-center mb-12"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
        >
          <h2 className="text-4xl md:text-5xl font-bold text-white mb-4">
            Welcome back, {user.email?.split('@')[0] || 'User'}!
          </h2>
          <p className="text-lg text-white/70 max-w-2xl mx-auto">
            Transform your financial reconciliation with AI-powered matching and insights.
          </p>
        </motion.section>

        {/* Beta Users: Enhanced Stats Dashboard with LIVE DATA */}
        {isBetaUser && (
          <motion.div
            className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8"
            initial={{ opacity: 0, y: 50 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ staggerChildren: 0.1, delayChildren: 0.3 }}
          >
            <MetricCard
              title="Reconciliations"
              value={stats?.reconciliations?.used || 0}
              subtitle={`${stats?.reconciliations?.limit ? `/ ${stats.reconciliations.limit} limit` : 'Unlimited'}`}
              icon="📊"
              color="blue"
              loading={statsLoading}
            />
            <MetricCard
              title="Total Transactions"
              value={stats?.transactions?.used || 0}
              subtitle={`${stats?.transactions?.limit ? `/ ${stats.transactions.limit} limit` : 'All time'}`}
              icon="💳"
              color="green"
              loading={statsLoading}
            />
            <MetricCard
              title="AI Queries"
              value={stats?.ai_queries?.used || 0}
              subtitle={`${stats?.ai_queries?.limit ? `/ ${stats.ai_queries.limit} limit` : 'All time'}`}
              icon="🤖"
              color="purple"
              loading={statsLoading}
            />
            <MetricCard
              title="Plan Status"
              value={stats?.plan_type || "Beta"}
              subtitle={stats?.plan_expires_at ? `Expires ${new Date(stats.plan_expires_at).toLocaleDateString()}` : 'Active'}
              icon="✨"
              color="orange"
              loading={statsLoading}
            />
          </motion.div>
        )}

        {/* Show error if stats failed to load */}
        {isBetaUser && statsError && (
          <motion.div
            className="bg-red-500/10 border border-red-500/20 text-red-300 p-4 rounded-2xl text-center"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
          >
            <p>{statsError}</p>
            <button
              onClick={fetchStats}
              className="mt-2 bg-red-500/20 hover:bg-red-500/30 px-4 py-2 rounded-lg text-sm transition-colors"
            >
              Retry Loading Stats
            </button>
          </motion.div>
        )}

        {/* File Upload Section */}
        <motion.div
          className="bg-white/5 rounded-3xl shadow-2xl border border-white/10 backdrop-blur-xl"
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.6 }}
        >
          <div className="p-8 border-b border-white/10">
            <h2 className="text-2xl font-bold bg-gradient-to-r from-white to-purple-200 bg-clip-text text-transparent mb-2">
              {isBetaUser ? "Pro File Upload" : "Upload Files"}
            </h2>
            {isBetaUser && (
              <p className="text-white/70">
                Enhanced processing with AI-powered matching and advanced analytics
              </p>
            )}
          </div>
          <div className="p-8">
            <FileUploader onUpload={handleUpload} disabled={!!jobId} />
          </div>
        </motion.div>

        <StatusAlert jobStatus={jobStatus} jobError={jobError} />

        {/* Results Section */}
        {data && data.summary && (
          <motion.div
            className="space-y-8"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.8 }}
          >
            <SummaryCard
              matched={data.summary.matched}
              unmatchedLedger={data.summary.unmatched_ledger}
              unmatchedBank={data.summary.unmatched_bank}
            />

            <div className="grid grid-cols-1 gap-8">
              {[
                { title: "Matched Transactions", data: data.summary.matched, icon: "✅" },
                { title: "Unmatched Ledger Entries", data: data.summary.unmatched_ledger, icon: "📋" },
                { title: "Unmatched Bank Transactions", data: data.summary.unmatched_bank, icon: "🏦" }
              ].map((section, index) => (
                <motion.div
                  key={section.title}
                  className="bg-white/5 rounded-3xl shadow-2xl border border-white/10 backdrop-blur-xl overflow-hidden"
                  initial={{ opacity: 0, y: 50 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.6, delay: index * 0.1 }}
                >
                  <div className="p-6 border-b border-white/10">
                    <h3 className="text-xl font-semibold text-white flex items-center gap-3">
                      <span>{section.icon}</span>
                      {section.title}
                    </h3>
                  </div>
                  <ResultsTable title={section.title} data={section.data} />
                </motion.div>
              ))}
            </div>
          </motion.div>
        )}

        {/* AI Chat Section with Error Boundary */}
        {ledgerFile && bankFile && (
          <motion.div
            className="bg-white/5 rounded-3xl shadow-2xl border border-white/10 backdrop-blur-xl"
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.6, delay: 0.4 }}
          >
            <div className="p-8 border-b border-white/10">
              <h2 className="text-2xl font-bold bg-gradient-to-r from-white to-purple-200 bg-clip-text text-transparent mb-2 flex items-center gap-3">
                <span>🤖</span>
                {isBetaUser ? "AI Assistant Pro" : "Chat with your Data"}
              </h2>
              {isBetaUser && (
                <p className="text-white/70">
                  Advanced AI with context-aware responses and smart insights
                </p>
              )}
            </div>
            <div className="p-8">
              <ChatErrorBoundary showErrorDetails={process.env.NODE_ENV === 'development'}>
                <ChatBox
                  ledgerFile={ledgerFile}
                  bankFile={bankFile}
                  reconciliationSummary={data ? data.summary : null}
                />
              </ChatErrorBoundary>
            </div>
          </motion.div>
        )}

        {/* Beta Users: Feedback Section - Only for ACTUAL beta users, not admins */}
        {isBetaUser && (
          <motion.div
            className="bg-gradient-to-br from-yellow-500/10 to-orange-500/10 rounded-3xl shadow-2xl border border-yellow-400/20 backdrop-blur-xl p-8"
            initial={{ opacity: 0, y: 50 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.5 }}
          >
            <div className="flex items-center space-x-4 mb-6">
              <div className="bg-gradient-to-r from-yellow-500 to-orange-500 text-white p-3 rounded-xl">
                <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M7 8h10m0 0l-4-4m4 4l-4 4"></path>
                </svg>
              </div>
              <h3 className="text-2xl font-bold text-white">Help Us Improve</h3>
            </div>
            <p className="text-white/80 mb-6 text-lg">
              Your feedback as a beta user is invaluable! Share your experience to help us make MatchLedger AI even better.
            </p>
            <div className="flex space-x-4">
              <Link
                to="/beta-dashboard"
                className="bg-gradient-to-r from-yellow-500 to-orange-500 hover:from-yellow-600 hover:to-orange-600 text-white px-6 py-3 rounded-xl font-medium transition-all duration-300 shadow-lg transform hover:scale-105"
              >
                Beta Dashboard
              </Link>
              <button
                className="bg-white/10 hover:bg-white/20 text-white px-6 py-3 rounded-xl border border-white/20 font-medium transition-all duration-300 backdrop-blur-sm"
                onClick={() => setShowFeedbackModal(true)}
              >
                Send Feedback
              </button>
            </div>
          </motion.div>
        )}

        {/* History Section */}
        {user && (
          <motion.div
            className="bg-white/5 rounded-3xl shadow-2xl border border-white/10 backdrop-blur-xl"
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.6, delay: 0.6 }}
          >
            <div className="p-8 border-b border-white/10">
              <h2 className="text-2xl font-bold bg-gradient-to-r from-white to-purple-200 bg-clip-text text-transparent flex items-center gap-3">
                <span>📊</span>
                {isBetaUser ? "Reconciliation History Pro" : "Past Reconciliations"}
              </h2>
            </div>
            <div className="p-8">
              <HistoryList key={data ? data.reconciliation_id : jobId} />
            </div>
          </motion.div>
        )}

        {/* Feedback Modal */}
        {showFeedbackModal && (
          <FeedbackModal
            isOpen={showFeedbackModal}
            onClose={() => setShowFeedbackModal(false)}
          />
        )}
      </div>
    </div>
  );
};

export default Dashboard;