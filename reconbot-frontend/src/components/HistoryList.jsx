// src/components/HistoryList.jsx
import React, { useEffect, useState } from 'react';
import api from '../utils/api';


const HistoryList = () => {
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchHistory = async () => {
      try {
        setLoading(true);
        // Fetch data from the backend
        const res = await api.get('/api/history');


        // --- IMPORTANT: Log the raw response to verify ---
        console.log("Raw API response for history:", res.data);

        // Process each history entry: parse the summary string into a JSON object
        const parsedHistory = res.data.history.map(entry => {
          let processedSummary = null; // Initialize to null

          try {
            // *** FIXED: Check if it's already an object first ***
            if (entry.summary === null || entry.summary === undefined) {
              // Handle null/undefined summary
              processedSummary = null;
              console.log("Summary is null/undefined for entry ID", entry.id);
            } else if (typeof entry.summary === 'object' && entry.summary !== null) {
              // If it's already an object, use it directly (this is the most common case)
              processedSummary = entry.summary;
              console.log("Using already-parsed summary for entry ID", entry.id, ":", processedSummary);
            } else if (typeof entry.summary === 'string') {
              // Only try to parse if it's a valid JSON string (not "[object Object]")
              if (entry.summary.trim() === '' || entry.summary === '[object Object]') {
                console.warn("Invalid summary string for entry ID", entry.id, ":", entry.summary);
                processedSummary = null;
              } else {
                processedSummary = JSON.parse(entry.summary);
                console.log("Parsed (from string) summary for entry ID", entry.id, ":", processedSummary);
              }
            } else {
              // Handle any other unexpected types
              console.warn("Unexpected summary type for entry ID", entry.id, ":", typeof entry.summary, entry.summary);
              processedSummary = null;
            }
          } catch (e) {
            // Log an error if processing fails (e.g., if there are truly malformed records)
            console.error("Failed to parse summary for entry ID", entry.id, ": Raw summary string:", entry.summary, "Error:", e);
            // Keep processedSummary as null or fallback to a default structure if needed
            processedSummary = null;
          }

          return {
            ...entry,
            summary: processedSummary // Replace the summary (string or object) with the processed object
          };
        });

        // --- IMPORTANT: Log the final state before setting ---
        console.log("Final parsed history state to be set:", parsedHistory);

        setHistory(parsedHistory); // Update the state with the parsed data
        setError(null); // Clear any previous errors
      } catch (err) {
        // Handle any errors during the API call itself
        console.error("Error fetching history from backend:", err);
        setError("Failed to load reconciliation history. Please try again.");
      } finally {
        setLoading(false); // Set loading to false once fetching is complete
      }
    };

    fetchHistory(); // Call the async function
  }, []); // Empty dependency array means this runs once on component mount

  // Conditional rendering based on loading and error states
  if (loading) return <div className="p-4 mt-6 border rounded-2xl bg-white shadow-lg text-center">Loading past reconciliations...</div>;
  if (error) return <div className="p-4 mt-6 border rounded-2xl bg-white shadow-lg text-center text-red-500">{error}</div>;
  if (history.length === 0) return <div className="p-4 mt-6 border rounded-2xl bg-white shadow-lg text-center">No past reconciliations found. Perform a new reconciliation!</div>;

  return (
    <div className="p-4 mt-6 border rounded-2xl bg-white shadow-lg">
      <h2 className="text-xl font-bold mb-4">Past Reconciliations</h2>
      <div className="overflow-x-auto"> {/* Added for better responsiveness on small screens */}
        <table className="min-w-full text-sm">
          <thead className="bg-gray-100">
            <tr>
              <th className="p-2 text-left">#</th>
              <th className="p-2 text-left">Date</th>
              <th className="p-2 text-left">Matched</th>
              <th className="p-2 text-left">Unmatched</th>
              <th className="p-2 text-left">Uploader</th> {/* Added Uploader column */}
              {/* Add more headers if needed */}
            </tr>
          </thead>
          <tbody>
            {history.map((entry) => ( // Removed idx as it's not strictly necessary for key if entry.id exists
              <tr key={entry.id} className="even:bg-gray-50">
                <td className="p-2">{entry.id}</td> {/* Display actual ID */}
                <td className="p-2">{new Date(entry.created_at).toLocaleString()}</td>
                {/* Use optional chaining (?) in case summary or its properties are null */}
                <td className="p-2">{entry.summary?.matched?.length || 0}</td>
                <td className="p-2">
                  {(entry.summary?.unmatched_ledger?.length || 0) +
                   (entry.summary?.unmatched_bank?.length || 0)}
                </td>
                <td className="p-2">{entry.uploaded_by}</td> {/* Display uploader */}
                {/* Add more data cells */}
              </tr>
            ))}
          </tbody>
        </table> {/* Closing </table> tag */}
      </div>
    </div>
  );
};

export default HistoryList;