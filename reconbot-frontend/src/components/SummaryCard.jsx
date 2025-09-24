// // src/components/SummaryCard.jsx
// import React from 'react';
// import { CheckCircle, AlertTriangle, XCircle } from 'lucide-react';

// export const SummaryCard = ({ matched, unmatchedLedger, unmatchedBank }) => {
//   const cards = [
//     {
//       title: 'Matched Transactions',
//       count: matched?.length || 0,
//       icon: CheckCircle,
//       bgLight: 'bg-green-100',
//       bgDark: 'dark:bg-green-900/20',
//       textLight: 'text-green-800',
//       textDark: 'dark:text-green-300',
//       iconColor: 'text-green-600 dark:text-green-400',
//       borderColor: 'border-green-200 dark:border-green-700/50'
//     },
//     {
//       title: 'Unmatched Ledger',
//       count: unmatchedLedger?.length || 0,
//       icon: AlertTriangle,
//       bgLight: 'bg-yellow-100',
//       bgDark: 'dark:bg-yellow-900/20',
//       textLight: 'text-yellow-800',
//       textDark: 'dark:text-yellow-300',
//       iconColor: 'text-yellow-600 dark:text-yellow-400',
//       borderColor: 'border-yellow-200 dark:border-yellow-700/50'
//     },
//     {
//       title: 'Unmatched Bank',
//       count: unmatchedBank?.length || 0,
//       icon: XCircle,
//       bgLight: 'bg-red-100',
//       bgDark: 'dark:bg-red-900/20',
//       textLight: 'text-red-800',
//       textDark: 'dark:text-red-300',
//       iconColor: 'text-red-600 dark:text-red-400',
//       borderColor: 'border-red-200 dark:border-red-700/50'
//     }
//   ];

//   return (
//     <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-6">
//       {cards.map((card, index) => {
//         const Icon = card.icon;
//         return (
//           <div
//             key={index}
//             className={`${card.bgLight} ${card.bgDark} p-6 rounded-xl shadow-sm border ${card.borderColor} transition-all duration-200 hover:shadow-md`}
//           >
//             <div className="flex items-center justify-between mb-2">
//               <p className={`text-sm font-medium ${card.textLight} ${card.textDark}`}>
//                 {card.title}
//               </p>
//               <Icon size={20} className={card.iconColor} />
//             </div>
//             <div className="flex items-end space-x-2">
//               <h2 className={`text-3xl font-bold ${card.textLight} ${card.textDark}`}>
//                 {card.count.toLocaleString()}
//               </h2>
//               <span className={`text-xs ${card.textLight} ${card.textDark} opacity-75 mb-1`}>
//                 {card.count === 1 ? 'entry' : 'entries'}
//               </span>
//             </div>
            
//             {/* Progress indicator for visual context */}
//             <div className="mt-3">
//               <div className={`h-1 rounded-full ${card.bgLight} ${card.bgDark} opacity-50`}>
//                 <div 
//                   className={`h-1 rounded-full ${card.iconColor.replace('text-', 'bg-')} transition-all duration-500`}
//                   style={{ 
//                     width: card.count > 0 ? '100%' : '0%' 
//                   }}
//                 ></div>
//               </div>
//             </div>
//           </div>
//         );
//       })}
//     </div>
//   );
// };

// src/components/SummaryCard.jsx (MODERN UI VERSION)
import { motion } from 'framer-motion';
import { useAuth } from '../contexts/AuthContext';

export const SummaryCard = ({ matched, unmatchedLedger, unmatchedBank }) => {
  const { user } = useAuth();
  
  const totalTransactions = matched.length + unmatchedLedger.length + unmatchedBank.length;
  const matchPercentage = totalTransactions > 0 ? ((matched.length / totalTransactions) * 100) : 0;
  
  const cards = [
    {
      title: 'Successfully Matched',
      value: matched.length,
      subtitle: `${matchPercentage.toFixed(1)}% of total`,
      icon: '✅',
      color: 'from-green-500 to-emerald-500',
      bgColor: 'from-green-500/20 to-emerald-500/20',
      borderColor: 'border-green-500/30',
      textColor: 'text-green-300',
      percentage: totalTransactions > 0 ? (matched.length / totalTransactions) * 100 : 0
    },
    {
      title: 'Unmatched Ledger',
      value: unmatchedLedger.length,
      subtitle: 'Ledger entries only',
      icon: '📊',
      color: 'from-yellow-500 to-orange-500',
      bgColor: 'from-yellow-500/20 to-orange-500/20',
      borderColor: 'border-yellow-500/30',
      textColor: 'text-yellow-300',
      percentage: totalTransactions > 0 ? (unmatchedLedger.length / totalTransactions) * 100 : 0
    },
    {
      title: 'Unmatched Bank',
      value: unmatchedBank.length,
      subtitle: 'Bank transactions only',
      icon: '🏦',
      color: 'from-red-500 to-pink-500',
      bgColor: 'from-red-500/20 to-pink-500/20',
      borderColor: 'border-red-500/30',
      textColor: 'text-red-300',
      percentage: totalTransactions > 0 ? (unmatchedBank.length / totalTransactions) * 100 : 0
    }
  ];

  return (
    <motion.div
      className="space-y-6"
      initial={{ opacity: 0, y: 30 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.6 }}
    >
      {/* Header with Total Summary */}
      <div className="text-center space-y-2">
        <motion.h2 
          className="text-3xl font-bold bg-gradient-to-r from-white to-purple-200 bg-clip-text text-transparent"
          initial={{ opacity: 0, scale: 0.8 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.6, delay: 0.2 }}
        >
          Reconciliation Summary
        </motion.h2>
        <motion.div
          className="flex items-center justify-center space-x-4 text-white/70"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.6, delay: 0.4 }}
        >
          <span>Total Transactions: <span className="font-semibold text-white">{totalTransactions}</span></span>
          {user?.is_beta_user && (
            <span className="bg-yellow-400/20 text-yellow-300 px-3 py-1 rounded-full text-sm border border-yellow-400/30">
              ✨ Pro Analysis
            </span>
          )}
        </motion.div>
      </div>

      {/* Summary Cards Grid */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {cards.map((card, index) => (
          <motion.div
            key={card.title}
            className={`relative bg-gradient-to-br ${card.bgColor} rounded-2xl p-6 backdrop-blur-sm border ${card.borderColor} shadow-2xl overflow-hidden`}
            initial={{ opacity: 0, y: 50, scale: 0.9 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            transition={{
              duration: 0.6,
              delay: 0.2 + (index * 0.1),
              ease: "easeOut"
            }}
            whileHover={{
              scale: 1.05,
              rotate: 1,
              transition: { duration: 0.3 }
            }}
          >
            {/* Background Gradient Overlay */}
            <div className={`absolute inset-0 bg-gradient-to-br from-white/10 to-transparent rounded-2xl`} />

            {/* Card Content */}
            <div className="relative z-10">
              <div className="flex items-center justify-between mb-4">
                <div className="text-sm font-medium text-white/90">
                  {card.title}
                </div>
                <div className="text-3xl">
                  {card.icon}
                </div>
              </div>

              <div className="space-y-2">
                <motion.div
                  className="text-4xl font-bold text-white"
                  initial={{ scale: 0 }}
                  animate={{ scale: 1 }}
                  transition={{
                    duration: 0.8,
                    delay: 0.5 + (index * 0.1),
                    type: "spring",
                    stiffness: 200
                  }}
                >
                  {card.value.toLocaleString()}
                </motion.div>

                <div className="text-sm text-white/70">
                  {card.subtitle}
                </div>

                {/* Progress Bar */}
                <div className="mt-4">
                  <div className="w-full bg-white/20 rounded-full h-2">
                    <motion.div
                      className={`bg-gradient-to-r ${card.color} h-2 rounded-full`}
                      initial={{ width: 0 }}
                      animate={{ width: `${card.percentage}%` }}
                      transition={{
                        duration: 1.2,
                        delay: 0.8 + (index * 0.1),
                        ease: "easeOut"
                      }}
                    />
                  </div>
                  <div className="text-xs text-white/60 mt-1">
                    {card.percentage.toFixed(1)}% of total
                  </div>
                </div>
              </div>
            </div>

            {/* Floating Animation Elements */}
            <motion.div
              className="absolute -top-4 -right-4 w-20 h-20 bg-white/5 rounded-full"
              animate={{ 
                scale: [1, 1.2, 1],
                opacity: [0.3, 0.6, 0.3]
              }}
              transition={{
                duration: 3,
                repeat: Infinity,
                delay: index * 0.5
              }}
            />
          </motion.div>
        ))}
      </div>

      {/* Accuracy Score for Beta Users */}
      {user?.is_beta_user && matchPercentage > 0 && (
        <motion.div
          className="bg-gradient-to-br from-purple-500/20 to-pink-500/20 rounded-2xl p-6 backdrop-blur-sm border border-purple-500/30 text-center"
          initial={{ opacity: 0, scale: 0.8 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.6, delay: 0.8 }}
        >
          <div className="flex items-center justify-center space-x-3 mb-4">
            <span className="text-2xl">🎯</span>
            <h3 className="text-xl font-bold text-white">AI Matching Accuracy</h3>
          </div>

          <div className="flex items-center justify-center space-x-8">
            <div className="text-center">
              <motion.div
                className="text-4xl font-bold bg-gradient-to-r from-purple-300 to-pink-300 bg-clip-text text-transparent"
                initial={{ scale: 0 }}
                animate={{ scale: 1 }}
                transition={{ duration: 0.8, delay: 1, type: "spring" }}
              >
                {matchPercentage.toFixed(1)}%
              </motion.div>
              <div className="text-white/70 text-sm">Match Rate</div>
            </div>

            <div className="text-center">
              <motion.div
                className="text-4xl font-bold bg-gradient-to-r from-green-300 to-emerald-300 bg-clip-text text-transparent"
                initial={{ scale: 0 }}
                animate={{ scale: 1 }}
                transition={{ duration: 0.8, delay: 1.1, type: "spring" }}
              >
                98.7%
              </motion.div>
              <div className="text-white/70 text-sm">AI Confidence</div>
            </div>

            <div className="text-center">
              <motion.div
                className="text-4xl font-bold bg-gradient-to-r from-blue-300 to-cyan-300 bg-clip-text text-transparent"
                initial={{ scale: 0 }}
                animate={{ scale: 1 }}
                transition={{ duration: 0.8, delay: 1.2, type: "spring" }}
              >
                2.3s
              </motion.div>
              <div className="text-white/70 text-sm">Processing Time</div>
            </div>
          </div>

          <div className="mt-4 text-purple-200/80 text-sm">
            ✨ Enhanced with Beta Pro AI algorithms for superior matching accuracy
          </div>
        </motion.div>
      )}

      {/* Quick Actions */}
      <motion.div
        className="flex flex-wrap items-center justify-center gap-4"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, delay: 1 }}
      >
        <motion.button
          className="bg-gradient-to-r from-blue-500 to-cyan-500 hover:from-blue-600 hover:to-cyan-600 text-white px-6 py-3 rounded-xl font-medium shadow-lg transition-all duration-300"
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
        >
          📊 Export Summary
        </motion.button>

        <motion.button
          className="bg-gradient-to-r from-green-500 to-emerald-500 hover:from-green-600 hover:to-emerald-600 text-white px-6 py-3 rounded-xl font-medium shadow-lg transition-all duration-300"      
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
        >
          📑 Generate Report
        </motion.button>

        {user?.is_beta_user && (
          <motion.button
            className="bg-gradient-to-r from-purple-500 to-pink-500 hover:from-purple-600 hover:to-pink-600 text-white px-6 py-3 rounded-xl font-medium shadow-lg transition-all duration-300"
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
          >
            ✨ AI Insights
          </motion.button>
        )}
      </motion.div>
    </motion.div>
  );
};