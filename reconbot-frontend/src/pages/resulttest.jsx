// src/components/ResultsTable.jsx (FIXED AMOUNT DISPLAY VERSION)
import { useState, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

const ResultsTable = ({ title, data }) => {
  const [currentPage, setCurrentPage] = useState(1);
  const [itemsPerPage, setItemsPerPage] = useState(10);
  const [sortColumn, setSortColumn] = useState(null);
  const [sortDirection, setSortDirection] = useState('asc');
  const [searchTerm, setSearchTerm] = useState('');

  // FIXED: Enhanced flatten function that properly handles amounts and nested data
  const flattenRow = (row) => {
    const flat = {};

    // First, copy all top-level properties
    for (const key in row) {
      const value = row[key];

      if (typeof value === "object" && value !== null && !Array.isArray(value)) {
        // Handle nested objects by flattening them
        for (const subKey in value) {
          const flatKey = `${key}_${subKey}`;
          flat[flatKey] = value[subKey];
        }
      } else {
        // Direct copy for primitive values and arrays
        flat[key] = value;
      }
    }

    return flat;
  };

  const flattenedData = useMemo(() => {
    return data.map(flattenRow);
  }, [data]);

  const headers = flattenedData.length > 0 ? Object.keys(flattenedData[0]) : [];

  // Search functionality
  const filteredData = useMemo(() => {
    if (!searchTerm) return flattenedData;

    return flattenedData.filter(row =>
      Object.values(row).some(value =>
        String(value).toLowerCase().includes(searchTerm.toLowerCase())
      )
    );
  }, [flattenedData, searchTerm]);

  // Sort functionality
  const sortedData = useMemo(() => {
    if (!sortColumn) return filteredData;

    return [...filteredData].sort((a, b) => {
      const aVal = a[sortColumn];
      const bVal = b[sortColumn];

      // Handle numeric sorting for amounts
      if (typeof aVal === 'number' && typeof bVal === 'number') {
        return sortDirection === 'asc' ? aVal - bVal : bVal - aVal;
      }

      // Handle string sorting
      const aStr = String(aVal);
      const bStr = String(bVal);

      if (aStr < bStr) return sortDirection === 'asc' ? -1 : 1;
      if (aStr > bStr) return sortDirection === 'asc' ? 1 : -1;
      return 0;
    });
  }, [filteredData, sortColumn, sortDirection]);

  // Pagination
  const totalItems = sortedData.length;
  const totalPages = Math.ceil(totalItems / itemsPerPage);
  const startIndex = (currentPage - 1) * itemsPerPage;
  const paginatedData = sortedData.slice(startIndex, startIndex + itemsPerPage);

  const handleSort = (column) => {
    if (sortColumn === column) {
      setSortDirection(sortDirection === 'asc' ? 'desc' : 'asc');
    } else {
      setSortColumn(column);
      setSortDirection('asc');
    }
  };

  // FIXED: Enhanced formatValue function that properly handles amounts
  const formatValue = (value, columnName) => {
    // Handle null/undefined values
    if (value === null || value === undefined) {
      return 'N/A';
    }

    // Handle arrays (like match_reasons)
    if (Array.isArray(value)) {
      return value.join(', ');
    }

    // Handle objects
    if (typeof value === 'object' && value !== null) {
      return JSON.stringify(value);
    }

    // Handle numbers - especially amounts
    if (typeof value === 'number') {
      // Check if this is likely a monetary amount
      if (columnName && (
        columnName.toLowerCase().includes('amount') ||
        columnName.toLowerCase().includes('price') ||
        columnName.toLowerCase().includes('cost') ||
        columnName.toLowerCase().includes('diff')
      )) {
        // Format as currency
        return new Intl.NumberFormat('en-US', {
          minimumFractionDigits: 2,
          maximumFractionDigits: 2
        }).format(value);
      }

      // Format as regular number with commas
      return value.toLocaleString();
    }

    // Handle strings
    const stringValue = String(value);

    // If it's a string that looks like a number, try to format it
    if (!isNaN(stringValue) && stringValue.trim() !== '') {
      const numValue = parseFloat(stringValue);
      if (columnName && columnName.toLowerCase().includes('amount')) {
        return new Intl.NumberFormat('en-US', {
          style: 'currency',
          currency: 'USD'
        }).format(numValue);
      }
      return numValue.toLocaleString();
    }

    return stringValue;
  };

  // FIXED: Enhanced header formatting
  const formatHeaderName = (columnName) => {
    return columnName
      .replace(/_/g, ' ')
      .replace(/\b\w/g, l => l.toUpperCase())
      .replace(/Id\b/g, 'ID')
      .replace(/Ref No/g, 'Reference')
      .replace(/Ledger Amount/g, 'Ledger Amount ($)')
      .replace(/Bank Amount/g, 'Bank Amount ($)');
  };

  const getTableIcon = (title) => {
    if (title.includes('Matched')) return '✅';
    if (title.includes('Ledger')) return '📊';
    if (title.includes('Bank')) return '🏦';
    return '📋';
  };

  const getTableColor = (title) => {
    if (title.includes('Matched')) return 'from-green-500/20 to-emerald-500/20 border-green-500/30';
    if (title.includes('Ledger')) return 'from-blue-500/20 to-cyan-500/20 border-blue-500/30';
    if (title.includes('Bank')) return 'from-purple-500/20 to-pink-500/20 border-purple-500/30';
    return 'from-gray-500/20 to-slate-500/20 border-gray-500/30';
  };

  // FIXED: Better column width handling for amounts
  const getColumnClass = (columnName) => {
    if (columnName.toLowerCase().includes('amount') || columnName.toLowerCase().includes('diff')) {
      return 'text-right font-mono'; // Right-align amounts and use monospace font
    }
    if (columnName.toLowerCase().includes('date')) {
      return 'text-center';
    }
    if (columnName.toLowerCase().includes('ref') || columnName.toLowerCase().includes('id')) {
      return 'text-center font-mono text-xs';
    }
    return '';
  };

  if (!data || data.length === 0) {
    return (
      <motion.div
        className="flex flex-col items-center justify-center py-16 text-center"
        initial={{ opacity: 0, scale: 0.8 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 0.5 }}
      >
        <div className="text-6xl mb-4 opacity-50">📭</div>
        <p className="text-white/70 text-lg font-medium">No data available</p>
        <p className="text-white/50 text-sm">Upload files to see results here</p>
      </motion.div>
    );
  }

  return (
    <motion.div
      className="space-y-6"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.6 }}
    >
      {/* Header with Search */}
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
        <div className="flex items-center space-x-3">
          <span className="text-2xl">{getTableIcon(title)}</span>
          <h3 className="text-xl font-bold text-white">{title}</h3>
          <span className="bg-white/10 text-white/80 text-sm font-medium px-3 py-1 rounded-full">
            {totalItems} items
          </span>
        </div>

        <div className="flex items-center space-x-4">
          <div className="relative">
            <input
              type="text"
              placeholder="Search transactions..."
              value={searchTerm}
              onChange={(e) => {
                setSearchTerm(e.target.value);
                setCurrentPage(1);
              }}
              className="pl-10 pr-4 py-2 bg-white/10 border border-white/20 rounded-lg text-white placeholder-white/50 focus:outline-none focus:ring-2 focus:ring-purple-500 transition-all duration-300 w-64"
            />
            <svg className="absolute left-3 top-2.5 w-4 h-4 text-white/40" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"></path>
            </svg>
          </div>

          <select
            value={itemsPerPage}
            onChange={(e) => {
              setItemsPerPage(Number(e.target.value));
              setCurrentPage(1);
            }}
            className="bg-white/10 border border-white/20 rounded-lg text-white px-3 py-2 focus:outline-none focus:ring-2 focus:ring-purple-500"
          >
            <option value={10}>10 per page</option>
            <option value={25}>25 per page</option>
            <option value={50}>50 per page</option>
            <option value={100}>100 per page</option>
          </select>
        </div>
      </div>

      {/* Modern Table */}
      <div className={`rounded-2xl overflow-hidden shadow-2xl bg-gradient-to-br ${getTableColor(title)} backdrop-blur-sm border`}>
        <div className="overflow-x-auto">
          <table className="min-w-full">
            <thead>
              <tr className="bg-white/10 backdrop-blur-sm">
                {headers.map((column) => (
                  <th
                    key={column}
                    onClick={() => handleSort(column)}
                    className={`px-6 py-4 text-left text-sm font-semibold text-white cursor-pointer hover:bg-white/5 transition-colors duration-200 select-none ${getColumnClass(column)}`}
                  >
                    <div className="flex items-center space-x-2">
                      <span>{formatHeaderName(column)}</span>
                      <motion.div
                        initial={{ opacity: 0 }}
                        animate={{ opacity: sortColumn === column ? 1 : 0.3 }}
                        className="text-xs"
                      >
                        {sortColumn === column && sortDirection === 'asc' ? '↑' : '↓'}
                      </motion.div>
                    </div>
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              <AnimatePresence>
                {paginatedData.map((row, idx) => (
                  <motion.tr
                    key={startIndex + idx}
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    exit={{ opacity: 0, x: 20 }}
                    transition={{ duration: 0.3, delay: idx * 0.05 }}
                    className="bg-white/5 hover:bg-white/10 border-b border-white/10 transition-all duration-200"
                  >
                    {headers.map((column, colIdx) => {
                      const value = row[column];
                      const formattedValue = formatValue(value, column);

                      return (
                        <td
                          key={colIdx}
                          className={`px-6 py-4 text-sm text-white/90 max-w-xs ${getColumnClass(column)}`}
                        >
                          <div
                            className="truncate"
                            title={typeof value === 'object' ? JSON.stringify(value) : String(formattedValue)}
                          >
                            {formattedValue}
                          </div>
                        </td>
                      );
                    })}
                  </motion.tr>
                ))}
              </AnimatePresence>
            </tbody>
          </table>
        </div>

        {/* Pagination */}
        {totalPages > 1 && (
          <div className="flex items-center justify-between px-6 py-4 bg-white/5 backdrop-blur-sm border-t border-white/10">
            <div className="text-sm text-white/70">
              Showing {startIndex + 1}-{Math.min(startIndex + itemsPerPage, totalItems)} of {totalItems} results
            </div>

            <div className="flex items-center space-x-2">
              <motion.button
                onClick={() => setCurrentPage(Math.max(1, currentPage - 1))}
                disabled={currentPage === 1}
                className="px-3 py-1 rounded-lg bg-white/10 text-white disabled:opacity-50 disabled:cursor-not-allowed hover:bg-white/20 transition-colors duration-200"
                whileHover={{ scale: currentPage === 1 ? 1 : 1.05 }}
                whileTap={{ scale: currentPage === 1 ? 1 : 0.95 }}
              >
                ←
              </motion.button>

              <div className="flex space-x-1">
                {[...Array(totalPages)].map((_, i) => {
                  const page = i + 1;
                  const isCurrentPage = page === currentPage;

                  if (
                    page === 1 ||
                    page === totalPages ||
                    (page >= currentPage - 1 && page <= currentPage + 1)
                  ) {
                    return (
                      <motion.button
                        key={page}
                        onClick={() => setCurrentPage(page)}
                        className={`px-3 py-1 rounded-lg text-sm font-medium transition-colors duration-200 ${
                          isCurrentPage
                            ? 'bg-purple-500 text-white'
                            : 'bg-white/10 text-white/70 hover:bg-white/20'
                        }`}
                        whileHover={{ scale: 1.05 }}
                        whileTap={{ scale: 0.95 }}
                      >
                        {page}
                      </motion.button>
                    );
                  } else if (page === currentPage - 2 || page === currentPage + 2) {
                    return <span key={page} className="text-white/50">...</span>;
                  }
                  return null;
                })}
              </div>

              <motion.button
                onClick={() => setCurrentPage(Math.min(totalPages, currentPage + 1))}
                disabled={currentPage === totalPages}
                className="px-3 py-1 rounded-lg bg-white/10 text-white disabled:opacity-50 disabled:cursor-not-allowed hover:bg-white/20 transition-colors duration-200"
                whileHover={{ scale: currentPage === totalPages ? 1 : 1.05 }}
                whileTap={{ scale: currentPage === totalPages ? 1 : 0.95 }}
              >
                →
              </motion.button>
            </div>
          </div>
        )}
      </div>

      {/* Debug Info - Remove this after testing */}
      {process.env.NODE_ENV === 'development' && data.length > 0 && (
        <details className="bg-black/20 rounded p-4 text-xs">
          <summary className="text-white cursor-pointer">Debug: First Row Data</summary>
          <pre className="text-green-400 mt-2 overflow-auto">
            {JSON.stringify(data[0], null, 2)}
          </pre>
        </details>
      )}
    </motion.div>
  );
};

export default ResultsTable;