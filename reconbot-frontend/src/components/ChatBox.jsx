// import { useState } from 'react'
// import api from '../utils/api';
// import { Button } from '@/components/ui/button'

// const ChatBox = ({ ledgerFile, bankFile }) => {
//   const [query, setQuery] = useState('')
//   const [response, setResponse] = useState('')
//   const [loading, setLoading] = useState(false)

//   const handleSend = async () => {
//   if (!query || !ledgerFile || !bankFile) {
//     alert("Please upload both files and enter a question.")
//     return
//   }

//   const formData = new FormData()
//   formData.append('query', query)
//   formData.append('ledger_file', ledgerFile)
//   formData.append('bank_file', bankFile)

//   try {
//     setLoading(true)

//     // ✅ Check exactly what you're sending
//     for (let pair of formData.entries()) {
//       console.log(`${pair[0]}:`, pair[1])
//     }

//     const res = await api.post('/api/chat', formData, {
//       // ❗ DO NOT manually set headers — let Axios handle it
//     })

//     setResponse(res.data.response)
//   } catch (error) {
//     console.error("Chat error:", error)
//     setResponse(`⚠️ Chat Error: ${error.message || 'Unable to process query.'}`)
//   } finally {
//     setLoading(false)
//   }
// }


//   return (
//     <div className="p-4 border rounded-2xl shadow-lg bg-white mt-6 space-y-4">
//       <h2 className="text-xl font-bold">💬 Ask ReconBot</h2>
//       <input
//         type="text"
//         value={query}
//         onChange={e => setQuery(e.target.value)}
//         placeholder="e.g. Show unmatched for April"
//         className="w-full p-2 border rounded-md"
//       />
//       <Button onClick={handleSend} disabled={loading}>
//         {loading ? 'Thinking...' : 'Ask'}
//       </Button>
//       {response && (
//         <div className="mt-4 p-3 bg-gray-100 rounded-md whitespace-pre-wrap text-sm text-gray-700 border">
//           {response}
//         </div>
//       )}
//     </div>
//   )
// }

// export default ChatBox
// src/components/ChatBox.jsx - Professional Enhanced Version

import { useState, useRef, useEffect, useCallback, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useAuth } from '../contexts/AuthContext';
import { chatWithData } from '../utils/api';
import TableRenderer from './TableRenderer';

const ChatBox = ({ ledgerFile, bankFile, reconciliationSummary }) => {
  const { user } = useAuth();
  const [query, setQuery] = useState('');
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);

  const isActualAdmin = user?.role === 'admin' || user?.is_admin === true;
  const isBetaUser = user?.is_beta_user === true;

  // Memoize reconciliation stats for performance
  const reconciliationStats = useMemo(() => {
    if (!reconciliationSummary) return null;

    return {
      matched: reconciliationSummary.matched?.length || 0,
      unmatchedLedger: reconciliationSummary.unmatched_ledger?.length || 0,
      unmatchedBank: reconciliationSummary.unmatched_bank?.length || 0,
      hasData: (reconciliationSummary.matched?.length || 0) > 0 ||
               (reconciliationSummary.unmatched_ledger?.length || 0) > 0 ||
               (reconciliationSummary.unmatched_bank?.length || 0) > 0
    };
  }, [reconciliationSummary]);

  // Debug logging with better structure
  useEffect(() => {
    if (reconciliationSummary) {
      console.log('🔍 ChatBox: Reconciliation Summary Analysis', {
        keys: Object.keys(reconciliationSummary),
        stats: reconciliationStats,
        sampleData: {
          unmatchedLedgerSample: reconciliationSummary.unmatched_ledger?.slice(0, 2),
          unmatchedBankSample: reconciliationSummary.unmatched_bank?.slice(0, 2)
        }
      });
    }
  }, [reconciliationSummary, reconciliationStats]);

  // Enhanced suggested questions based on data context
  const getSuggestedQuestions = useCallback(() => {
    if (reconciliationStats?.hasData) {
      const baseQuestions = [
        "Show me all unmatched transactions",
        "Why did transactions go unmatched and how do I match them?",
        "Show me all matched transactions",
        "What patterns do you see in the unmatched data?"
      ];

      const advancedQuestions = [
        "Analyze reconciliation discrepancies with AI insights",
        "Run comprehensive analysis on data patterns",
        "Generate reconciliation improvement recommendations"
      ];

      return [
        ...baseQuestions,
        ...(isBetaUser ? advancedQuestions.slice(0, 2) : ["Analyze the reconciliation results"])
      ];
    } else {
      return [
        "Show transactions above $1000",
        "What's the date range of these transactions?",
        "Are there any duplicate entries?",
        "Show me transaction summary by amount",
        "What's the total transaction count?",
        isBetaUser ? "Run AI analysis on the transaction data" : "Analyze the data patterns"
      ];
    }
  }, [reconciliationStats, isBetaUser]);

  const suggestedQuestions = getSuggestedQuestions();

  const scrollToBottom = useCallback(() => {
    try {
      messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    } catch (error) {
      console.warn('Scroll error (non-critical):', error.message);
    }
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [messages, scrollToBottom]);

  // Enhanced welcome message with better formatting
  useEffect(() => {
    if (!user) return;

    let welcomeContent = '';

    if (isBetaUser) {
      welcomeContent = '✨ **Welcome to ReconBot AI Pro!**\n\nI\'m your advanced AI assistant with enhanced analytical capabilities for financial reconciliation.';
    } else {
      welcomeContent = '👋 **Welcome to ReconBot!**\n\nI\'m your intelligent reconciliation assistant, ready to help analyze your financial data.';
    }

    if (reconciliationStats?.hasData) {
      const { matched, unmatchedLedger, unmatchedBank } = reconciliationStats;
      const totalTransactions = matched + unmatchedLedger + unmatchedBank;
      const matchRate = totalTransactions > 0 ? ((matched / totalTransactions) * 100).toFixed(1) : 0;

      welcomeContent += `\n\n📊 **Reconciliation Overview:**\n`;
      welcomeContent += `• ✅ **Matched:** ${matched} transaction pairs\n`;
      welcomeContent += `• 🔴 **Unmatched Ledger:** ${unmatchedLedger} transactions\n`;
      welcomeContent += `• 🔵 **Unmatched Bank:** ${unmatchedBank} transactions\n`;
      welcomeContent += `• 📈 **Match Rate:** ${matchRate}%\n\n`;

      if (unmatchedLedger > 0 || unmatchedBank > 0) {
        welcomeContent += `🎯 **Ready to help you:** Identify patterns, resolve discrepancies, and improve your reconciliation process!`;
      } else {
        welcomeContent += `🎉 **Perfect match!** All transactions are reconciled. Ask me about your data insights!`;
      }
    } else {
      welcomeContent += '\n\n📁 **Ready to analyze** your transaction data! Upload your files and ask me anything.';
    }

    const welcomeMessage = {
      id: `welcome_${Date.now()}`,
      type: 'bot',
      content: welcomeContent,
      timestamp: new Date(),
      isWelcome: true
    };
    setMessages([welcomeMessage]);
  }, [user, isBetaUser, reconciliationStats]);

  const validateInputs = useCallback(() => {
    if (!ledgerFile || !bankFile) {
      setError("Please upload both ledger and bank files before starting the conversation.");
      return false;
    }
    return true;
  }, [ledgerFile, bankFile]);

  // Enhanced response parsing with better error handling
  const parseApiResponse = (responseData) => {
    console.log('🔄 ChatBox: Parsing API response:', responseData);

    try {
      // Handle nested response structure: { response: { response_type: '...', data: {...} } }
      if (responseData?.response) {
        const response = responseData.response;

        if (response.response_type === 'table' && response.data) {
          return {
            content: response,
            responseType: 'table',
            success: true
          };
        } else if (response.response_type === 'text' && response.data) {
          return {
            content: String(response.data).trim(),
            responseType: 'text',
            success: true
          };
        } else if (typeof response === 'string') {
          return {
            content: response.trim(),
            responseType: 'text',
            success: true
          };
        }
      }

      // Handle direct response formats
      if (typeof responseData === 'string') {
        return {
          content: responseData.trim(),
          responseType: 'text',
          success: true
        };
      }

      // Handle direct table format
      if (responseData?.response_type === 'table' && responseData.data) {
        return {
          content: responseData,
          responseType: 'table',
          success: true
        };
      }

      // Fallback
      console.warn('🚨 Unknown response format:', responseData);
      return {
        content: 'I received a response but couldn\'t parse it properly. Please try rephrasing your question.',
        responseType: 'text',
        success: false
      };

    } catch (parseError) {
      console.error('🚨 Response parsing error:', parseError);
      return {
        content: 'I had trouble understanding the response format. Please try again.',
        responseType: 'text',
        success: false
      };
    }
  };

  const handleSend = async (questionText = query) => {
    try {
      const queryToSend = typeof questionText === 'string' ? questionText : query;
      if (!queryToSend?.trim()) {
        setError("Please enter a question.");
        return;
      }

      if (!validateInputs()) {
        return;
      }

      setError(null);

      // Add user message
      const userMessage = {
        id: `user_${Date.now()}`,
        type: 'user',
        content: String(queryToSend.trim()),
        timestamp: new Date()
      };

      setMessages(prev => [...prev, userMessage]);
      setQuery('');
      setLoading(true);

      console.log('🚀 ChatBox: Sending request', {
        query: queryToSend.trim(),
        hasReconciliation: !!reconciliationSummary,
        reconciliationStats
      });

      // Call API
      const responseData = await chatWithData(
        queryToSend.trim(),
        ledgerFile,
        bankFile,
        reconciliationSummary
      );

      if (!responseData) {
        throw new Error('No response received from server');
      }

      // Parse response
      const parsedResponse = parseApiResponse(responseData);

      const botMessage = {
        id: `bot_${Date.now()}`,
        type: 'bot',
        content: parsedResponse.content,
        responseType: parsedResponse.responseType,
        timestamp: new Date(),
        isError: !parsedResponse.success
      };

      setMessages(prev => [...prev, botMessage]);

    } catch (error) {
      console.error('🚨 Chat error:', error);

      let errorMessage = 'I encountered an error processing your request. Please try again.';

      // Enhanced error handling
      if (error.message?.includes('timeout') || error.code === 'ECONNABORTED') {
        errorMessage = 'Request timed out. Please try again with a simpler question or check your connection.';
      } else if (error.message?.includes('413') || error.message?.includes('too large')) {
        errorMessage = 'The files are too large to process. Please try with smaller files or contact support.';
      } else if (error.message?.includes('500')) {
        errorMessage = 'Server error occurred. Please try again in a moment or contact support if the issue persists.';
      } else if (error.message?.includes('422')) {
        errorMessage = 'There was an issue with the data format. Please try a different question or re-upload your files.';
      } else if (error.message?.includes('parsing')) {
        errorMessage = 'I had trouble processing that request. Try: "Show me all unmatched transactions"';
      } else if (error.message) {
        errorMessage = error.message;
      }

      const errorBotMessage = {
        id: `error_${Date.now()}`,
        type: 'error',
        content: errorMessage,
        timestamp: new Date()
      };

      setMessages(prev => [...prev, errorBotMessage]);

    } finally {
      setLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const clearChat = useCallback(() => {
    const welcomeMessage = messages.find(msg => msg.isWelcome);
    setMessages(welcomeMessage ? [welcomeMessage] : []);
    setError(null);
  }, [messages]);

  // Enhanced message bubble with better styling and functionality
  const MessageBubble = ({ message }) => {
    if (!message) return null;

    const isBot = message.type === 'bot';
    const isError = message.type === 'error';
    const isUser = message.type === 'user';
    const isWelcome = message.isWelcome;

    const messageContent = message.content;
    const isTableResponse = isBot && message.responseType === 'table' &&
                           typeof messageContent === 'object' &&
                           messageContent.response_type === 'table';

    return (
      <motion.div
        initial={{ opacity: 0, y: 20, scale: 0.95 }}
        animate={{ opacity: 1, y: 0, scale: 1 }}
        transition={{ duration: 0.3 }}
        className={`flex ${isUser ? 'justify-end' : 'justify-start'} mb-4`}
      >
        <div className={`flex items-start space-x-3 max-w-[85%] ${isUser ? 'flex-row-reverse space-x-reverse' : ''}`}>
          {/* Enhanced Avatar */}
          <div className={`flex-shrink-0 w-10 h-10 rounded-xl flex items-center justify-center text-lg shadow-lg ${
            isUser
              ? 'bg-gradient-to-r from-purple-500 to-pink-500 text-white'
              : isError
                ? 'bg-gradient-to-r from-red-500 to-pink-500 text-white'
                : isWelcome
                  ? 'bg-gradient-to-r from-green-500 to-blue-500 text-white'
                  : 'bg-gradient-to-r from-blue-500 to-cyan-500 text-white'
          }`}>
            {isUser ? '👤' : isError ? '⚠️' : isWelcome ? '🤖' : isBetaUser ? '✨' : '🤖'}
          </div>

          {/* Enhanced Message Content */}
          <div className={`rounded-2xl px-4 py-3 shadow-lg backdrop-blur-sm border ${
            isUser
              ? 'bg-gradient-to-r from-purple-500 to-pink-500 text-white border-purple-400/30'
              : isError
                ? 'bg-gradient-to-r from-red-500/20 to-pink-500/20 text-red-200 border-red-400/30'
                : isWelcome
                  ? 'bg-gradient-to-r from-green-500/10 to-blue-500/10 text-white border-green-400/30'
                  : 'bg-white/10 text-white border-white/20'
          }`}>
            {/* Content Rendering */}
            {isTableResponse ? (
              <div className="space-y-2">
                <TableRenderer data={messageContent.data} />
              </div>
            ) : (
              <div className="space-y-2">
                {/* Handle markdown-style formatting */}
                {typeof messageContent === 'string'
                  ? messageContent.split('\n').map((line, index) => {
                      if (line.trim() === '') return <br key={index} />;

                      // Handle bold text
                      if (line.includes('**')) {
                        const parts = line.split('**');
                        return (
                          <p key={index} className="text-sm leading-relaxed">
                            {parts.map((part, partIndex) =>
                              partIndex % 2 === 1 ? <strong key={partIndex}>{part}</strong> : part
                            )}
                          </p>
                        );
                      }

                      // Handle bullet points
                      if (line.trim().startsWith('•') || line.trim().startsWith('-')) {
                        return (
                          <p key={index} className="text-sm leading-relaxed ml-2">
                            {line.trim()}
                          </p>
                        );
                      }

                      return (
                        <p key={index} className="text-sm leading-relaxed">
                          {line}
                        </p>
                      );
                    })
                  : (messageContent?.data || messageContent?.toString() || 'No content')
                }
              </div>
            )}

            {/* Timestamp */}
            <div className={`text-xs mt-2 flex items-center justify-between ${
              isUser ? 'text-white/70' : isError ? 'text-red-200/70' : 'text-white/50'
            }`}>
              <span>
                {message.timestamp ? message.timestamp.toLocaleTimeString() : 'Unknown time'}
              </span>
              {isTableResponse && (
                <span className="text-xs bg-white/20 px-2 py-1 rounded">
                  📊 Table View
                </span>
              )}
            </div>
          </div>
        </div>
      </motion.div>
    );
  };

  // Don't render if user is not loaded
  if (!user) {
    return (
      <div className="flex items-center justify-center h-96 bg-white/5 rounded-2xl border border-white/10">
        <div className="text-white/50 flex items-center space-x-2">
          <motion.div
            className="w-4 h-4 border-2 border-white/50 border-t-transparent rounded-full"
            animate={{ rotate: 360 }}
            transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
          />
          <span>Loading chat interface...</span>
        </div>
      </div>
    );
  }

  return (
    <motion.div
      className="space-y-6"
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.6 }}
    >
      {/* Enhanced Chat Messages Area */}
      <div className="bg-white/5 rounded-2xl border border-white/10 backdrop-blur-sm overflow-hidden">
        <div className="h-96 overflow-y-auto p-6 scroll-smooth scrollbar-thin scrollbar-thumb-white/20 scrollbar-track-transparent">
          <AnimatePresence>
            {messages.map((message) => (
              <MessageBubble key={message.id} message={message} />
            ))}
          </AnimatePresence>

          {/* Enhanced Loading Animation */}
          {loading && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="flex justify-start mb-4"
            >
              <div className="flex items-start space-x-3 max-w-[85%]">
                <div className="flex-shrink-0 w-10 h-10 rounded-xl bg-gradient-to-r from-blue-500 to-cyan-500 flex items-center justify-center text-lg shadow-lg">
                  <motion.div
                    animate={{ rotate: 360 }}
                    transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
                  >
                    {isBetaUser ? '✨' : '🤖'}
                  </motion.div>
                </div>
                <div className="bg-white/10 rounded-2xl px-4 py-3 border border-white/20 backdrop-blur-sm">
                  <div className="flex space-x-2">
                    {[0, 1, 2].map((i) => (
                      <motion.div
                        key={i}
                        className="w-2 h-2 bg-blue-400 rounded-full"
                        animate={{
                          scale: [1, 1.5, 1],
                          opacity: [0.5, 1, 0.5],
                        }}
                        transition={{
                          duration: 1.5,
                          repeat: Infinity,
                          delay: i * 0.2,
                        }}
                      />
                    ))}
                  </div>
                  <div className="text-xs text-white/50 mt-2">
                    {isBetaUser ? '🧠 AI is analyzing with advanced algorithms...' : '💭 Processing your request...'}
                  </div>
                </div>
              </div>
            </motion.div>
          )}

          <div ref={messagesEndRef} />
        </div>

        {/* Enhanced Input Area */}
        <div className="border-t border-white/10 p-6 bg-white/5">
          {/* Smart Suggested Questions */}
          {messages.length <= 1 && suggestedQuestions.length > 0 && (
            <motion.div
              className="mb-4"
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.4 }}
            >
              <p className="text-sm text-white/70 mb-3 flex items-center">
                <span className="mr-2">💡</span>
                <span>Try these smart suggestions:</span>
              </p>
              <div className="flex flex-wrap gap-2">
                {suggestedQuestions.slice(0, 4).map((question, index) => (
                  <motion.button
                    key={index}
                    onClick={() => handleSend(question)}
                    disabled={loading || !validateInputs()}
                    className={`bg-white/10 hover:bg-white/20 border border-white/20 text-white/80 px-3 py-2 rounded-lg text-xs transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed ${
                      isBetaUser ? 'hover:border-purple-400/50' : 'hover:border-blue-400/50'
                    }`}
                    whileHover={!loading ? { scale: 1.02 } : {}}
                    whileTap={!loading ? { scale: 0.98 } : {}}
                  >
                    {question}
                  </motion.button>
                ))}
              </div>
            </motion.div>
          )}

          {/* Enhanced Input Field */}
          <div className="flex items-center space-x-4">
            <div className="flex-1 relative">
              <textarea
                ref={inputRef}
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                onKeyPress={handleKeyPress}
                placeholder={
                  isBetaUser
                    ? "Ask me anything about your data with AI-powered insights..."
                    : reconciliationStats?.hasData
                      ? "Ask about your reconciliation data..."
                      : "Ask me about your transaction data..."
                }
                className="w-full bg-white/10 border border-white/20 rounded-xl text-white placeholder-white/50 px-4 py-3 pr-12 focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-transparent transition-all duration-300 backdrop-blur-sm resize-none"
                rows="1"
                disabled={loading}
                style={{ minHeight: '48px', maxHeight: '120px' }}
              />
              <div className="absolute right-3 top-1/2 transform -translate-y-1/2 text-white/40">
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z"></path>
                </svg>
              </div>
            </div>

            {/* Enhanced Send Button */}
            <motion.button
              onClick={() => handleSend()}
              disabled={!query.trim() || loading || !ledgerFile || !bankFile}
              className={`px-6 py-3 rounded-xl font-semibold transition-all duration-300 shadow-lg ${
                query.trim() && !loading && ledgerFile && bankFile
                  ? isBetaUser
                    ? 'bg-gradient-to-r from-purple-500 to-pink-500 hover:from-purple-600 hover:to-pink-600 text-white cursor-pointer'
                    : 'bg-gradient-to-r from-blue-500 to-cyan-500 hover:from-blue-600 hover:to-cyan-600 text-white cursor-pointer'
                  : 'bg-gray-500/50 text-gray-300 cursor-not-allowed'
              }`}
              whileHover={query.trim() && !loading && ledgerFile && bankFile ? { scale: 1.05 } : {}}
              whileTap={query.trim() && !loading && ledgerFile && bankFile ? { scale: 0.95 } : {}}
            >
              {loading ? (
                <motion.div
                  className="w-5 h-5 border-2 border-white border-t-transparent rounded-full"
                  animate={{ rotate: 360 }}
                  transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
                />
              ) : (
                <div className="flex items-center space-x-2">
                  <span>Send</span>
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8"></path>
                  </svg>
                </div>
              )}
            </motion.button>
          </div>

          {/* Enhanced Error Display */}
          {error && (
            <motion.div
              className="mt-3 text-red-300 text-sm bg-red-500/20 border border-red-500/30 rounded-lg p-3"
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
            >
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-2">
                  <span>⚠️</span>
                  <span>{error}</span>
                </div>
                <button
                  onClick={() => setError(null)}
                  className="text-red-300 hover:text-white ml-2 transition-colors"
                >
                  ✕
                </button>
              </div>
            </motion.div>
          )}

          {/* Enhanced Status Indicators */}
          <div className="flex items-center justify-between mt-4 text-xs text-white/50">
            <div className="flex items-center space-x-4">
              <span className={`flex items-center space-x-1 transition-colors ${ledgerFile ? 'text-green-300' : 'text-red-300'}`}>
                <span>{ledgerFile ? '✅' : '❌'}</span>
                <span>Ledger File</span>
              </span>
              <span className={`flex items-center space-x-1 transition-colors ${bankFile ? 'text-green-300' : 'text-red-300'}`}>
                <span>{bankFile ? '✅' : '❌'}</span>
                <span>Bank File</span>
              </span>
              {reconciliationStats?.hasData && (
                <span className="flex items-center space-x-1 text-blue-300">
                  <span>📊</span>
                  <span>Reconciliation: {reconciliationStats.matched}M / {reconciliationStats.unmatchedLedger + reconciliationStats.unmatchedBank}U</span>
                </span>
              )}
            </div>
            {isBetaUser && (
              <span className="text-yellow-300 flex items-center space-x-1">
                <motion.span
                  animate={{ rotate: [0, 360] }}
                  transition={{ duration: 3, repeat: Infinity, ease: "linear" }}
                >
                  ✨
                </motion.span>
                <span>AI Pro Active</span>
              </span>
            )}
          </div>
        </div>
      </div>

      {/* Enhanced Quick Actions */}
      {messages.length > 2 && (
        <motion.div
          className="flex flex-wrap gap-2 justify-center"
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.4, delay: 0.2 }}
        >
          <motion.button
            onClick={clearChat}
            disabled={loading}
            className="bg-white/10 hover:bg-white/20 border border-white/20 text-white/80 px-4 py-2 rounded-lg text-sm transition-all duration-200 disabled:opacity-50 flex items-center space-x-2"   
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
          >
            <span>🗑️</span>
            <span>Clear Chat</span>
          </motion.button>

          <motion.button
            onClick={() => handleSend("Summarize our conversation and key insights")}
            disabled={loading || !validateInputs()}
            className="bg-gradient-to-r from-blue-500 to-cyan-500 hover:from-blue-600 hover:to-cyan-600 text-white px-4 py-2 rounded-lg text-sm transition-all duration-300 disabled:opacity-50 flex items-center space-x-2"
            whileHover={!loading ? { scale: 1.02 } : {}}
            whileTap={!loading ? { scale: 0.98 } : {}}
          >
            <span>📝</span>
            <span>Summarize</span>
          </motion.button>

          {reconciliationStats?.hasData && reconciliationStats.unmatchedLedger + reconciliationStats.unmatchedBank > 0 && (
            <motion.button
              onClick={() => handleSend("Provide detailed analysis of why transactions are unmatched and how to resolve them")}
              disabled={loading || !validateInputs()}
              className="bg-gradient-to-r from-orange-500 to-red-500 hover:from-orange-600 hover:to-red-600 text-white px-4 py-2 rounded-lg text-sm transition-all duration-300 disabled:opacity-50 flex items-center space-x-2"
              whileHover={!loading ? { scale: 1.02 } : {}}
              whileTap={!loading ? { scale: 0.98 } : {}}
            >
              <span>🔍</span>
              <span>Analyze Unmatched</span>
            </motion.button>
          )}
        </motion.div>
      )}
    </motion.div>
  );
};

export default ChatBox;