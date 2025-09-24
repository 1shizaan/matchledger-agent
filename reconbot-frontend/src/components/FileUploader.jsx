// import { useState } from 'react'
// import { Button } from '@/components/ui/button'

// const FileUploader = ({ onUpload }) => {
//   const [ledgerFile, setLedgerFile] = useState(null)
//   const [bankFile, setBankFile] = useState(null)
//   const [email, setEmail] = useState('') // 👈 New state for email input

//   const handleSubmit = () => {
//     if (ledgerFile && bankFile) {
//       // Pass email along with the files to the onUpload function
//       onUpload(ledgerFile, bankFile, email) // 👈 Update onUpload call
//     }
//   }

//   return (
//     <div className="space-y-4 p-4 border rounded-2xl shadow-lg bg-white">
//       <h2 className="text-xl font-bold">Upload Files</h2>
//       <input type="file" accept=".csv" onChange={e => setLedgerFile(e.target.files[0])} />
//       <input type="file" accept=".csv" onChange={e => setBankFile(e.target.files[0])} />

//       {/* 👈 New email input field */}
//       <input
//         type="email"
//         placeholder="Optional: Enter your email"
//         className="p-2 border rounded-md w-full"
//         value={email}
//         onChange={e => setEmail(e.target.value)}
//       />

//       <Button onClick={handleSubmit}>Reconcile</Button>
//     </div>
//   )
// }

// export default FileUploader

// src/components/FileUploader.jsx - COMPLETE ENHANCED VERSION
import { useState, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useAuth } from '../contexts/AuthContext';
import { detectColumns, startReconciliationWithMapping, startReconciliation } from '../utils/api';

const FileUploader = ({ onUpload, disabled }) => {
  const { user } = useAuth();
  const [ledgerFile, setLedgerFile] = useState(null);
  const [bankFile, setBankFile] = useState(null);
  const [email, setEmail] = useState('');
  const [dragOver, setDragOver] = useState({ ledger: false, bank: false });
  const [uploading, setUploading] = useState(false);

  // ✅ ENHANCED: Column detection states
  const [showColumnMapping, setShowColumnMapping] = useState(false);
  const [detectedColumns, setDetectedColumns] = useState(null);
  const [columnMappings, setColumnMappings] = useState({
    ledger: { date: '', amount: '', description: '' },
    bank: { date: '', amount: '', description: '' }
  });
  const [detectingColumns, setDetectingColumns] = useState(false);
  const [mappingConfidence, setMappingConfidence] = useState({});

  const ledgerInputRef = useRef();
  const bankInputRef = useRef();

  // ✅ ENHANCED: Column detection with comprehensive error handling
  const handleColumnDetection = async () => {
    if (!ledgerFile || !bankFile) {
      console.error('❌ Both files required for column detection');
      alert('Please select both ledger and bank statement files before detecting columns.');
      return;
    }

    setDetectingColumns(true);
    try {
      console.log('🎯 Starting enhanced column detection...');
      const response = await detectColumns(ledgerFile, bankFile);
      console.log('✅ Column detection response:', response);

      setDetectedColumns(response);

      // ✅ ENHANCED: Handle multiple response structures from backend
      const ledgerHeaders = response.ledger_headers || response.ledger_columns || [];
      const bankHeaders = response.bank_headers || response.bank_columns || [];

      // ✅ ENHANCED: Smart mapping from AI suggestions with fallbacks
      const ledgerSuggestions = response.ledger_suggestions || response.ledger_mappings || response.suggested_mappings?.ledger || {};
      const bankSuggestions = response.bank_suggestions || response.bank_mappings || response.suggested_mappings?.bank || {};

      setColumnMappings({
        ledger: {
          date: ledgerSuggestions.date || findBestMatch(ledgerHeaders, ['date', 'transaction_date', 'dt', 'transaction date', 'txn_date']),
          amount: ledgerSuggestions.amount || findBestMatch(ledgerHeaders, ['amount', 'value', 'total', 'debit', 'credit', 'sum']),
          description: ledgerSuggestions.description || findBestMatch(ledgerHeaders, ['description', 'details', 'memo', 'reference', 'narration', 'particulars', 'remarks'])
        },
        bank: {
          date: bankSuggestions.date || findBestMatch(bankHeaders, ['date', 'transaction_date', 'dt', 'transaction date', 'txn_date']),
          amount: bankSuggestions.amount || findBestMatch(bankHeaders, ['amount', 'debit', 'credit', 'value', 'withdrawal', 'deposit', 'sum']),
          description: bankSuggestions.description || findBestMatch(bankHeaders, ['description', 'details', 'memo', 'reference', 'narration', 'particulars', 'remarks'])
        }
      });

      // ✅ ENHANCED: Set confidence scores from multiple possible response keys
      setMappingConfidence(response.confidence_scores || response.confidence || {});

      setShowColumnMapping(true);
      console.log('✅ Column mapping interface ready');

    } catch (error) {
      console.error('❌ Column detection failed:', error);

      // ✅ ENHANCED: Better error handling with specific error messages
      const errorMessage = error.response?.data?.detail || error.message || 'Unknown error occurred';

      const confirmMessage = `Smart column detection failed: ${errorMessage}\n\n` +
        `Would you like to:\n` +
        `• OK - Try basic reconciliation instead\n` +
        `• Cancel - Select different files`;

      if (window.confirm(confirmMessage)) {
        await handleBasicUpload();
      }
    } finally {
      setDetectingColumns(false);
    }
  };

  // ✅ ENHANCED: Advanced column matching algorithm
  const findBestMatch = (headers, keywords) => {
    if (!headers || headers.length === 0) return '';

    // Normalize headers for better matching
    const normalizedHeaders = headers.map(header => ({
      original: header,
      normalized: header.toLowerCase().replace(/[^a-z0-9]/g, '')
    }));

    // 1. Exact matches first (case-insensitive)
    for (const keyword of keywords) {
      const normalizedKeyword = keyword.toLowerCase().replace(/[^a-z0-9]/g, '');
      const exactMatch = normalizedHeaders.find(header =>
        header.normalized === normalizedKeyword
      );
      if (exactMatch) return exactMatch.original;
    }

    // 2. Starts with matches
    for (const keyword of keywords) {
      const normalizedKeyword = keyword.toLowerCase().replace(/[^a-z0-9]/g, '');
      const startsWithMatch = normalizedHeaders.find(header =>
        header.normalized.startsWith(normalizedKeyword) || normalizedKeyword.startsWith(header.normalized)
      );
      if (startsWithMatch) return startsWithMatch.original;
    }

    // 3. Contains matches
    for (const keyword of keywords) {
      const normalizedKeyword = keyword.toLowerCase().replace(/[^a-z0-9]/g, '');
      const containsMatch = normalizedHeaders.find(header =>
        header.normalized.includes(normalizedKeyword) || normalizedKeyword.includes(header.normalized)
      );
      if (containsMatch) return containsMatch.original;
    }

    // 4. Fallback to first header
    return headers[0] || '';
  };

  // ✅ ENHANCED: Submit with proper validation and error handling
  const handleEnhancedSubmit = async () => {
    if (!ledgerFile || !bankFile || disabled) return;

    // ✅ ENHANCED: Comprehensive validation
    if (!columnMappings.ledger.date || !columnMappings.ledger.amount ||
        !columnMappings.bank.date || !columnMappings.bank.amount) {
      alert('Please map at least the Date and Amount columns for both files.');
      return;
    }

    setUploading(true);
    try {
      console.log('🚀 Starting enhanced reconciliation with mappings:', columnMappings);

      // ✅ ENHANCED: Check if backend expects FormData or JSON
      const backendExpectsFormData = true; // Set based on your backend API

      let result;

      if (backendExpectsFormData) {
        // ✅ Original FormData approach
        const formData = new FormData();
        formData.append('ledger_file', ledgerFile);
        formData.append('bank_file', bankFile);
        formData.append('email', email || user?.email || '');

        // Add column mappings
        formData.append('ledger_date_column', columnMappings.ledger.date);
        formData.append('ledger_amount_column', columnMappings.ledger.amount);
        formData.append('ledger_description_column', columnMappings.ledger.description || '');
        formData.append('bank_date_column', columnMappings.bank.date);
        formData.append('bank_amount_column', columnMappings.bank.amount);
        formData.append('bank_description_column', columnMappings.bank.description || '');

        // Mark as PDF if needed
        if (bankFile.name.toLowerCase().endsWith('.pdf')) {
          formData.append('bank_is_pdf', 'true');
        }

        result = await startReconciliationWithMapping(formData);
      } else {
        // ✅ Enhanced JSON approach with base64 files
        const ledgerBase64 = await fileToBase64(ledgerFile);
        const bankBase64 = await fileToBase64(bankFile);

        const requestPayload = {
          ledger_filename: ledgerFile.name,
          bank_filename: bankFile.name,
          ledger_file_content: ledgerBase64.split(',')[1], // Remove data:... prefix
          bank_file_content: bankBase64.split(',')[1],
          bank_is_pdf: bankFile.name.toLowerCase().endsWith('.pdf'),
          email: email || user?.email || '',
          ledger_column_map: {
            date: columnMappings.ledger.date,
            amount: columnMappings.ledger.amount,
            description: columnMappings.ledger.description || ''
          },
          bank_column_map: {
            date: columnMappings.bank.date,
            amount: columnMappings.bank.amount,
            description: columnMappings.bank.description || ''
          }
        };

        result = await startReconciliationWithMapping(requestPayload);
      }

      console.log('✅ Enhanced reconciliation started:', result);

      // ✅ ENHANCED: Call onUpload with proper parameters
      await onUpload(ledgerFile, bankFile, email, result.task_id);

      // Reset states
      setShowColumnMapping(false);
      setDetectedColumns(null);
      setColumnMappings({
        ledger: { date: '', amount: '', description: '' },
        bank: { date: '', amount: '', description: '' }
      });

      console.log('🎉 Enhanced reconciliation initiated successfully!');

    } catch (error) {
      console.error('❌ Enhanced reconciliation failed:', error);
      const errorMessage = error.response?.data?.detail || error.message || 'Unknown error occurred';
      alert(`Enhanced reconciliation failed: ${errorMessage}\n\nPlease try again or contact support if the issue persists.`);
    } finally {
      setUploading(false);
    }
  };

  // ✅ HELPER: Convert file to base64
  const fileToBase64 = (file) => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.readAsDataURL(file);
      reader.onload = () => resolve(reader.result);
      reader.onerror = error => reject(error);
    });
  };

  // ✅ ENHANCED: Basic upload with better error handling
  const handleBasicUpload = async () => {
    if (!ledgerFile || !bankFile || disabled) {
      alert('Please select both ledger and bank statement files.');
      return;
    }

    setUploading(true);
    try {
      console.log('🔄 Starting basic reconciliation...');
      await onUpload(ledgerFile, bankFile, email);
      console.log('✅ Basic reconciliation initiated successfully!');
    } catch (error) {
      console.error('❌ Basic reconciliation failed:', error);
      const errorMessage = error.response?.data?.detail || error.message || 'Unknown error occurred';
      alert(`Reconciliation failed: ${errorMessage}\n\nPlease try again or contact support if the issue persists.`);
    } finally {
      setUploading(false);
    }
  };

  // ✅ ENHANCED: Main submit handler with proper flow control
  const handleSubmit = async () => {
    if (!ledgerFile || !bankFile || disabled) {
      alert('Please select both ledger and bank statement files.');
      return;
    }

    if (showColumnMapping) {
      // User is in column mapping step - proceed with enhanced reconciliation
      await handleEnhancedSubmit();
    } else if (user?.is_beta_user) {
      // Beta user starting process - begin with column detection
      await handleColumnDetection();
    } else {
      // Regular user - basic upload
      await handleBasicUpload();
    }
  };

  // ✅ ENHANCED: Drag and drop handlers with better file validation
  const handleDragOver = (e, type) => {
    e.preventDefault();
    setDragOver(prev => ({ ...prev, [type]: true }));
  };

  const handleDragLeave = (e, type) => {
    e.preventDefault();
    setDragOver(prev => ({ ...prev, [type]: false }));
  };

  const handleDrop = (e, type) => {
    e.preventDefault();
    setDragOver(prev => ({ ...prev, [type]: false }));

    const files = Array.from(e.dataTransfer.files);
    const acceptedFile = files.find(file => {
      const fileName = file.name.toLowerCase();
      return fileName.endsWith('.csv') ||
             fileName.endsWith('.xlsx') ||
             (type === 'bank' && fileName.endsWith('.pdf'));
    });

    if (acceptedFile) {
      if (type === 'ledger') {
        setLedgerFile(acceptedFile);
      } else {
        setBankFile(acceptedFile);
      }

      // Reset column mapping if files change
      if (showColumnMapping) {
        setShowColumnMapping(false);
        setDetectedColumns(null);
      }
    } else {
      alert(`Please select a valid ${type === 'bank' ? 'CSV, XLSX, or PDF' : 'CSV or XLSX'} file.`);
    }
  };

  // ✅ ENHANCED: File drop zone with better visuals
  const FileDropZone = ({ type, file, onFileSelect, icon, title, description }) => {
    const isDragging = dragOver[type];

    return (
      <motion.div
        className={`relative border-2 border-dashed rounded-2xl p-8 text-center transition-all duration-300 cursor-pointer ${
          isDragging
            ? 'border-purple-400 bg-purple-500/10 scale-105 shadow-lg shadow-purple-500/25'
            : file
            ? 'border-green-400 bg-green-500/10 shadow-lg shadow-green-500/25'
            : 'border-white/30 bg-white/5 hover:border-purple-400 hover:bg-purple-500/5'
        }`}
        onClick={() => type === 'ledger' ? ledgerInputRef.current?.click() : bankInputRef.current?.click()}
        onDragOver={(e) => handleDragOver(e, type)}
        onDragLeave={(e) => handleDragLeave(e, type)}
        onDrop={(e) => handleDrop(e, type)}
        whileHover={{ scale: 1.02 }}
        whileTap={{ scale: 0.98 }}
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4 }}
      >
        <div className="space-y-4">
          <motion.div
            className={`text-5xl mx-auto w-fit ${
              file ? 'text-green-400' : isDragging ? 'text-purple-400' : 'text-white/70'
            }`}
            animate={{
              scale: isDragging ? [1, 1.2, 1] : file ? [1, 1.1, 1] : 1,
              rotate: isDragging ? [0, 5, -5, 0] : 0
            }}
            transition={{ duration: 0.3 }}
          >
            {file ? '✅' : isDragging ? '📁' : icon}
          </motion.div>

          <div>
            <h3 className={`text-xl font-semibold mb-2 ${file ? 'text-green-300' : 'text-white'}`}>
              {file ? `📄 ${file.name}` : title}
            </h3>
            <p className="text-white/70 text-sm">
              {file ? (
                <span className="flex items-center justify-center gap-2">
                  <span>✅ Ready to upload</span>
                  <span>•</span>
                  <span>{(file.size / (1024 * 1024)).toFixed(2)} MB</span>
                </span>
              ) : (
                description
              )}
            </p>
          </div>

          {file && (
            <motion.button
              onClick={(e) => {
                e.stopPropagation();
                onFileSelect(null);
                // Reset column mapping if files are removed
                if (showColumnMapping) {
                  setShowColumnMapping(false);
                  setDetectedColumns(null);
                }
              }}
              className="bg-red-500/20 hover:bg-red-500/30 text-red-300 px-4 py-2 rounded-lg text-sm font-medium transition-colors duration-200 flex items-center gap-2 mx-auto"
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
            >
              <span>🗑️</span>
              <span>Remove File</span>
            </motion.button>
          )}

          <input
            ref={type === 'ledger' ? ledgerInputRef : bankInputRef}
            type="file"
            accept={type === 'bank' ? '.csv,.xlsx,.pdf' : '.csv,.xlsx'}
            onChange={(e) => {
              const selectedFile = e.target.files[0];
              if (selectedFile) {
                onFileSelect(selectedFile);
                // Reset column mapping if files change
                if (showColumnMapping) {
                  setShowColumnMapping(false);
                  setDetectedColumns(null);
                }
              }
            }}
            className="hidden"
          />
        </div>

        {isDragging && (
          <motion.div
            className="absolute inset-0 bg-purple-500/20 rounded-2xl border-2 border-purple-400 flex items-center justify-center backdrop-blur-sm"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
          >
            <div className="text-center">
              <div className="text-4xl mb-2">📁</div>
              <p className="text-purple-200 font-semibold">Drop your file here</p>
            </div>
          </motion.div>
        )}
      </motion.div>
    );
  };

  // ✅ ENHANCED: Column Mapping Modal with premium UX
  const ColumnMappingModal = () => {
    if (!showColumnMapping || !detectedColumns) return null;

    const handleColumnMapChange = (fileType, standardColumn, selectedHeader) => {
      setColumnMappings(prev => ({
        ...prev,
        [fileType]: {
          ...prev[fileType],
          [standardColumn]: selectedHeader
        }
      }));
    };

    const standardColumns = [
      { key: 'date', label: 'Date Column', icon: '📅', required: true, desc: 'Transaction date' },
      { key: 'amount', label: 'Amount Column', icon: '💰', required: true, desc: 'Transaction amount' },
      { key: 'description', label: 'Description Column', icon: '📝', required: false, desc: 'Transaction details' }
    ];

    // Get available headers with fallbacks
    const ledgerHeaders = detectedColumns.ledger_headers || detectedColumns.ledger_columns || [];
    const bankHeaders = detectedColumns.bank_headers || detectedColumns.bank_columns || [];

    const isValidMapping = () => {
      return columnMappings.ledger.date && columnMappings.ledger.amount &&
             columnMappings.bank.date && columnMappings.bank.amount;
    };

    const getConfidenceColor = (fileType, column) => {
      const confidence = mappingConfidence[`${fileType}_${column}`] || 0;
      if (confidence >= 90) return 'text-green-400 bg-green-500/20';
      if (confidence >= 70) return 'text-yellow-400 bg-yellow-500/20';
      if (confidence >= 50) return 'text-orange-400 bg-orange-500/20';
      return 'text-red-400 bg-red-500/20';
    };

    const getConfidenceBadge = (fileType, column) => {
      const confidence = mappingConfidence[`${fileType}_${column}`];
      if (!confidence) return null;

      return (
        <span className={`text-xs px-2 py-1 rounded font-medium ${getConfidenceColor(fileType, column)}`}>
          {confidence >= 90 ? '🎯' : confidence >= 70 ? '⚡' : confidence >= 50 ? '⚠️' : '❌'} {confidence}%
        </span>
      );
    };

    return (
      <motion.div
        className="fixed inset-0 bg-black/80 backdrop-blur-sm flex items-center justify-center z-50 p-4"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
      >
        <motion.div
          className="bg-gradient-to-br from-gray-900 via-gray-800 to-gray-900 rounded-2xl p-8 max-w-6xl w-full max-h-[90vh] overflow-y-auto border border-white/20 shadow-2xl"
          initial={{ scale: 0.9, opacity: 0, y: 50 }}
          animate={{ scale: 1, opacity: 1, y: 0 }}
          exit={{ scale: 0.9, opacity: 0, y: 50 }}
        >
          {/* Header */}
          <div className="text-center mb-8">
            <motion.div
              className="inline-flex items-center gap-3 mb-4"
              initial={{ y: -20 }}
              animate={{ y: 0 }}
            >
              <div className="p-4 bg-gradient-to-r from-purple-500 to-blue-500 rounded-xl shadow-lg">
                <span className="text-3xl">🎯</span>
              </div>
              <div className="text-left">
                <h2 className="text-3xl font-bold text-white">Smart Column Detection</h2>
                <p className="text-purple-300 text-sm">AI-powered column mapping for precise reconciliation</p>
              </div>
            </motion.div>
            <div className="mt-6 inline-flex items-center gap-4 bg-blue-500/20 border border-blue-500/30 rounded-lg px-6 py-3">
              <span className="text-blue-400 text-lg">✨</span>
              <span className="text-blue-300 font-medium">
                Detected {ledgerHeaders.length} ledger columns and {bankHeaders.length} bank columns
              </span>
            </div>
          </div>

          {/* Column Mappings */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            {/* Ledger Mapping */}
            <motion.div
              className="space-y-6"
              initial={{ x: -20, opacity: 0 }}
              animate={{ x: 0, opacity: 1 }}
              transition={{ delay: 0.1 }}
            >
              <div className="flex items-center gap-3 mb-6 p-4 bg-green-500/10 border border-green-500/20 rounded-lg">
                <div className="w-5 h-5 bg-green-500 rounded-full flex-shrink-0"></div>
                <div className="flex-1">
                  <h3 className="text-xl font-semibold text-green-400">📊 Ledger File</h3>
                  <p className="text-green-300/80 text-sm">Your internal accounting records</p>
                </div>
                <span className="text-sm text-green-400 bg-green-500/20 px-3 py-1 rounded font-medium">
                  {ledgerHeaders.length} columns
                </span>
              </div>

              {standardColumns.map((col, index) => (
                <motion.div
                  key={col.key}
                  className="space-y-3"
                  initial={{ y: 20, opacity: 0 }}
                  animate={{ y: 0, opacity: 1 }}
                  transition={{ delay: 0.2 + index * 0.1 }}
                >
                  <div className="flex items-center justify-between">
                    <label className="flex items-center gap-3 text-white/90 font-medium">
                      <span className="text-xl">{col.icon}</span>
                      <div>
                        <span className="block">{col.label}</span>
                        <span className="text-xs text-white/60">{col.desc}</span>
                      </div>
                      {col.required && <span className="text-red-400 font-bold">*</span>}
                    </label>
                    {getConfidenceBadge('ledger', col.key)}
                  </div>

                  <select
                    className="w-full px-4 py-3 bg-white/10 border border-white/20 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-green-500 focus:border-transparent transition-all"
                    value={columnMappings.ledger[col.key] || ''}
                    onChange={(e) => handleColumnMapChange('ledger', col.key, e.target.value)}
                  >
                    <option value="" className="bg-gray-800">
                      {col.required ? '⚠️ Select required column...' : '📝 Select column (optional)...'}
                    </option>
                    {ledgerHeaders.map(header => (
                      <option key={header} value={header} className="bg-gray-800">
                        📄 {header}
                      </option>
                    ))}
                  </select>
                </motion.div>
              ))}
            </motion.div>

            {/* Bank Mapping */}
            <motion.div
              className="space-y-6"
              initial={{ x: 20, opacity: 0 }}
              animate={{ x: 0, opacity: 1 }}
              transition={{ delay: 0.2 }}
            >
              <div className="flex items-center gap-3 mb-6 p-4 bg-blue-500/10 border border-blue-500/20 rounded-lg">
                <div className="w-5 h-5 bg-blue-500 rounded-full flex-shrink-0"></div>
                <div className="flex-1">
                  <h3 className="text-xl font-semibold text-blue-400">🏦 Bank Statement</h3>
                  <p className="text-blue-300/80 text-sm">Your bank transaction records</p>
                </div>
                <span className="text-sm text-blue-400 bg-blue-500/20 px-3 py-1 rounded font-medium">
                  {bankHeaders.length} columns
                </span>
              </div>

              {standardColumns.map((col, index) => (
                <motion.div
                  key={col.key}
                  className="space-y-3"
                  initial={{ y: 20, opacity: 0 }}
                  animate={{ y: 0, opacity: 1 }}
                  transition={{ delay: 0.3 + index * 0.1 }}
                >
                  <div className="flex items-center justify-between">
                    <label className="flex items-center gap-3 text-white/90 font-medium">
                      <span className="text-xl">{col.icon}</span>
                      <div>
                        <span className="block">{col.label}</span>
                        <span className="text-xs text-white/60">{col.desc}</span>
                      </div>
                      {col.required && <span className="text-red-400 font-bold">*</span>}
                    </label>
                    {getConfidenceBadge('bank', col.key)}
                  </div>

                  <select
                    className="w-full px-4 py-3 bg-white/10 border border-white/20 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all" 
                    value={columnMappings.bank[col.key] || ''}
                    onChange={(e) => handleColumnMapChange('bank', col.key, e.target.value)}
                  >
                    <option value="" className="bg-gray-800">
                      {col.required ? '⚠️ Select required column...' : '📝 Select column (optional)...'}
                    </option>
                    {bankHeaders.map(header => (
                      <option key={header} value={header} className="bg-gray-800">
                        🏦 {header}
                      </option>
                    ))}
                  </select>
                </motion.div>
              ))}
            </motion.div>
          </div>

          {/* Validation Message */}
          {!isValidMapping() && (
            <motion.div
              className="mt-8 p-4 bg-yellow-500/10 border border-yellow-500/20 rounded-lg"
              initial={{ scale: 0.95, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
            >
              <div className="flex items-center gap-3 text-yellow-400">
                <span className="text-2xl">⚠️</span>
                <div>
                  <span className="font-semibold block">Required Mappings Missing</span>
                  <span className="text-sm text-yellow-300">
                    Please map the Date and Amount columns for both files to proceed.
                  </span>
                </div>
              </div>
            </motion.div>
          )}

          {/* Action Buttons */}
          <motion.div
            className="flex justify-between items-center mt-10 pt-6 border-t border-white/10"
            initial={{ y: 20, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            transition={{ delay: 0.5 }}
          >
            <motion.button
              onClick={() => {
                setShowColumnMapping(false);
                setDetectedColumns(null);
                setColumnMappings({
                  ledger: { date: '', amount: '', description: '' },
                  bank: { date: '', amount: '', description: '' }
                });
              }}
              className="px-6 py-3 bg-gray-600/50 text-white rounded-lg font-medium hover:bg-gray-600/70 transition-colors"
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
            >
              ❌ Cancel
            </motion.button>

            <div className="flex gap-4">
              <motion.button
                onClick={handleBasicUpload}
                className="px-6 py-3 bg-gray-700/50 text-white rounded-lg font-medium hover:bg-gray-700/70 transition-colors"
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
              >
                🔄 Use Basic Mode
              </motion.button>

              <motion.button
                onClick={handleSubmit}
                disabled={!isValidMapping() || uploading}
                className={`px-8 py-3 rounded-lg font-medium transition-all flex items-center gap-2 ${
                  isValidMapping() && !uploading
                    ? 'bg-gradient-to-r from-purple-500 to-pink-500 text-white hover:from-purple-600 hover:to-pink-600 shadow-lg'
                    : 'bg-gray-500/50 text-gray-300 cursor-not-allowed'
                }`}
                whileHover={isValidMapping() && !uploading ? { scale: 1.05 } : {}}
                whileTap={isValidMapping() && !uploading ? { scale: 0.95 } : {}}
              >
                {uploading ? (
                  <>
                    <motion.div
                      className="w-5 h-5 border-2 border-white border-t-transparent rounded-full"
                      animate={{ rotate: 360 }}
                      transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
                    />
                    Processing...
                  </>
                ) : (
                  <>
                    <span>🚀</span>
                    Start Pro Reconciliation
                  </>
                )}
              </motion.button>
            </div>
          </motion.div>
        </motion.div>
      </motion.div>
    );
  };

  const isReadyToUpload = ledgerFile && bankFile && !disabled;

  return (
    <div className="space-y-8">
      {/* File Upload Areas */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <FileDropZone
          type="ledger"
          file={ledgerFile}
          onFileSelect={setLedgerFile}
          icon="📊"
          title="Ledger File"
          description="Drag & drop your ledger CSV/XLSX file here or click to browse"
        />

        <FileDropZone
          type="bank"
          file={bankFile}
          onFileSelect={setBankFile}
          icon="🏦"
          title="Bank Statement"
          description="Drag & drop your bank CSV/XLSX/PDF file here or click to browse"
        />
      </div>

      {/* Email Input */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4, delay: 0.2 }}
      >
        <label className="block text-sm font-medium text-white/90 mb-3">
          📧 Email Address (Optional)
        </label>
        <div className="relative">
          <input
            type="email"
            placeholder={user?.email || "Enter your email for notifications"}
            className="w-full px-4 py-4 bg-white/10 border border-white/20 rounded-xl text-white placeholder-white/50 focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-transparent transition-all duration-300 backdrop-blur-sm"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
          />
          <div className="absolute inset-y-0 right-0 flex items-center pr-4">
            <svg className="w-5 h-5 text-white/40" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M16 12a4 4 0 10-8 0 4 4 0 008 0zm0 0v1.5a2.5 2.5 0 005 0V12a9 9 0 10-9 9m4.5-1.206a8.959 8.959 0 01-4.5 1.207"></path>
            </svg>
          </div>
        </div>
      </motion.div>

      {/* Upload Button */}
      <motion.div
        className="flex justify-center"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4, delay: 0.3 }}
      >
        <motion.button
          onClick={handleSubmit}
          disabled={!isReadyToUpload || uploading || detectingColumns}
          className={`relative px-8 py-4 rounded-xl font-semibold text-white text-lg shadow-2xl transition-all duration-300 ${
            isReadyToUpload && !uploading && !detectingColumns
              ? 'bg-gradient-to-r from-purple-500 to-pink-500 hover:from-purple-600 hover:to-pink-600 cursor-pointer'
              : 'bg-gray-500/50 cursor-not-allowed'
          }`}
          whileHover={isReadyToUpload && !uploading && !detectingColumns ? { scale: 1.05 } : {}}
          whileTap={isReadyToUpload && !uploading && !detectingColumns ? { scale: 0.95 } : {}}
        >
          <AnimatePresence mode="wait">
            {detectingColumns ? (
              <motion.div
                className="flex items-center space-x-3"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
              >
                <motion.div
                  className="w-5 h-5 border-2 border-white border-t-transparent rounded-full"
                  animate={{ rotate: 360 }}
                  transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
                />
                <span>🎯 Detecting Columns...</span>
              </motion.div>
            ) : uploading ? (
              <motion.div
                className="flex items-center space-x-3"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
              >
                <motion.div
                  className="w-5 h-5 border-2 border-white border-t-transparent rounded-full"
                  animate={{ rotate: 360 }}
                  transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
                />
                <span>🚀 Processing...</span>
              </motion.div>
            ) : showColumnMapping ? (
              <motion.div
                className="flex items-center space-x-3"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
              >
                <span>🎯</span>
                <span>Configure Column Mapping</span>
              </motion.div>
            ) : (
              <motion.div
                className="flex items-center space-x-3"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
              >
                <span>{user?.is_beta_user ? '🎯' : '🚀'}</span>
                <span>
                  {user?.is_beta_user ? 'Start Smart Reconciliation' : 'Start Reconciliation'}
                </span>
              </motion.div>
            )}
          </AnimatePresence>
        </motion.button>
      </motion.div>

      {/* File Requirements & Beta Features */}
      <motion.div
        className="text-center space-y-3"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.4, delay: 0.4 }}
      >
        <p className="flex items-center justify-center gap-2 text-sm text-white/60">
          <span>📋</span>
          <span>Supported formats: CSV, XLSX files (PDF for bank statements)</span>
        </p>

        {user?.is_beta_user ? (
          <div className="bg-gradient-to-r from-yellow-400/10 to-yellow-500/10 border border-yellow-400/20 rounded-lg p-4 mx-auto max-w-2xl">
            <p className="text-yellow-300 font-medium flex items-center justify-center gap-2">
              <span>✨</span>
              <span>Pro Features Enabled: Smart column detection + AI-powered matching + 90%+ accuracy</span>
            </p>
            <p className="text-yellow-200/80 text-xs mt-1">
              Enhanced reconciliation with confidence scoring and fallback options
            </p>
          </div>
        ) : (
          <div className="bg-gradient-to-r from-blue-400/10 to-blue-500/10 border border-blue-400/20 rounded-lg p-3 mx-auto max-w-xl">
            <p className="text-blue-300 text-sm flex items-center justify-center gap-2">
              <span>🔄</span>
              <span>Standard reconciliation • Upgrade to Pro for smart features</span>
            </p>
          </div>
        )}

        <p className="text-xs text-white/40">
          Maximum file size: 10MB per file • Secure processing with enterprise-grade encryption
        </p>
      </motion.div>

      {/* Column Mapping Modal */}
      <AnimatePresence>
        <ColumnMappingModal />
      </AnimatePresence>
    </div>
  );
};

export default FileUploader;