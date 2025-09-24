// src/pages/AdminFeedbackList.jsx
import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import api from '../utils/api';
import { getAdminFeedback } from '../utils/api';
import { useAuth } from '../contexts/AuthContext';

const AdminFeedbackList = () => {
  const { user, loading } = useAuth();
  const navigate = useNavigate();
  const [feedbacks, setFeedbacks] = useState([]);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (!loading && (!user || !user.is_admin)) {
      navigate('/');
    }
  }, [user, loading, navigate]);

  useEffect(() => {
    const fetchFeedback = async () => {
      try {
        const data = await getAdminFeedback();
        setFeedbacks(data);
      } catch (err) {
        setError('Failed to fetch feedback.');
        console.error(err);
      }
    };

    if (user?.is_admin) {
      fetchFeedback();
    }
  }, [user]);

  const exportToCSV = () => {
    const csvContent = [
      ['User ID', 'Message', 'Created At'],
      ...feedbacks.map(f => [
        f.user_id,
        `"${f.message.replace(/"/g, '""')}`,
        new Date(f.created_at).toLocaleString()
      ])
    ]
      .map(row => row.join(','))
      .join('\n');

    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const url = window.URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.setAttribute('download', `beta_feedback_${new Date().toISOString()}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <div className="max-w-3xl mx-auto py-10 px-4">
      <h1 className="text-2xl font-bold mb-6">Beta Feedback (Admin)</h1>

      {error && <p className="text-red-600 mb-4">{error}</p>}

      <button
        onClick={exportToCSV}
        className="mb-4 bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700"
      >
        Download Feedback as CSV
      </button>

      <div className="bg-white shadow rounded-lg border overflow-x-auto">
        <table className="min-w-full divide-y divide-gray-200">
          <thead className="bg-gray-100">
            <tr>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-600 uppercase tracking-wider">
                User ID
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-600 uppercase tracking-wider">
                Message
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-600 uppercase tracking-wider">
                Created At
              </th>
            </tr>
          </thead>
          <tbody className="bg-white divide-y divide-gray-200">
            {feedbacks.map((f) => (
              <tr key={f.id}>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">{f.user_id}</td>
                <td className="px-6 py-4 whitespace-pre-line text-sm text-gray-800">{f.message}</td>
                <td className="px-6 py-4 text-sm text-gray-500">
                  {new Date(f.created_at).toLocaleString()}
                </td>
              </tr>
            ))}
            {feedbacks.length === 0 && !error && (
              <tr>
                <td colSpan="3" className="px-6 py-4 text-center text-sm text-gray-500">
                  No feedback found.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
};

export default AdminFeedbackList;