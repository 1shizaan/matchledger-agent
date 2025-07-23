// src/pages/Dashboard.jsx
import { useState } from 'react';
import FileUploader from '../components/FileUploader';
import ResultsTable from '../components/ResultsTable';
import { SummaryCard } from '../components/SummaryCard';
import { reconcileFiles } from '../utils/api';
import ChatBox from '../components/ChatBox';
import HistoryList from '../components/HistoryList';
import { useAuth } from '../contexts/AuthContext';

const Dashboard = () => {
  const { logout, user, loading } = useAuth();
  const [data, setData] = useState(null);
  const [ledgerFile, setLedgerFile] = useState(null);
  const [bankFile, setBankFile] = useState(null);

  const handleUpload = async (uploadedLedgerFile, uploadedBankFile, email) => {
    setLedgerFile(uploadedLedgerFile);
    setBankFile(uploadedBankFile);

    const res = await reconcileFiles(uploadedLedgerFile, uploadedBankFile, email);
    setData(res);
  };

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Mobile-friendly header */}
      <div className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-4">
            <h1 className="text-2xl md:text-3xl font-bold text-gray-900">ReconBot</h1>
            <div className="flex items-center space-x-2 md:space-x-4">
              <span className="text-sm md:text-base text-gray-600 truncate max-w-32 md:max-w-none">
                {user?.email}
              </span>
              <button
                onClick={logout}
                className="bg-red-500 hover:bg-red-600 text-white px-3 py-2 md:px-4 rounded-lg text-sm font-medium transition-colors"
              >
                Logout
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Main content */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6 space-y-6">

        {/* File Uploader - responsive */}
        <div className="bg-white rounded-lg shadow-sm border">
          <FileUploader onUpload={handleUpload} />
        </div>

        {/* Results section - responsive grid */}
        {data && data.summary && (
          <div className="space-y-6">
            {/* Summary Cards */}
            <SummaryCard
              matched={data.summary.matched}
              unmatchedLedger={data.summary.unmatched_ledger}
              unmatchedBank={data.summary.unmatched_bank}
            />

            {/* Results Tables - stacked on mobile */}
            <div className="grid grid-cols-1 gap-6">
              <div className="bg-white rounded-lg shadow-sm border overflow-hidden">
                <ResultsTable title="Matched Transactions" data={data.summary.matched} />
              </div>
              <div className="bg-white rounded-lg shadow-sm border overflow-hidden">
                <ResultsTable title="Unmatched Ledger Entries" data={data.summary.unmatched_ledger} />
              </div>
              <div className="bg-white rounded-lg shadow-sm border overflow-hidden">
                <ResultsTable title="Unmatched Bank Transactions" data={data.summary.unmatched_bank} />
              </div>
            </div>
          </div>
        )}

        {/* Chat section - full width on mobile */}
        {ledgerFile && bankFile && (
          <div className="bg-white rounded-lg shadow-sm border">
            <div className="p-4 border-b">
              <h2 className="text-xl md:text-2xl font-bold text-gray-900">Chat with your Data</h2>
            </div>
            <ChatBox ledgerFile={ledgerFile} bankFile={bankFile} />
          </div>
        )}

        {/* History section */}
        {!loading && user && (
          <div className="bg-white rounded-lg shadow-sm border">
            <HistoryList />
          </div>
        )}
      </div>
    </div>
  );
};

export default Dashboard;