// src/components/TableRenderer.jsx
import React from 'react';

const TableRenderer = ({ data }) => {
    if (!data || !data.headers || !data.rows) {
        return <p>No table data available.</p>;
    }

    return (
        <div className="overflow-x-auto my-4">
            {data.title && <h4 className="text-white/80 font-bold mb-2">{data.title}</h4>}
            <table className="min-w-full divide-y divide-white/20 rounded-lg overflow-hidden">
                <thead className="bg-white/10">
                    <tr>
                        {data.headers.map((header, index) => (
                            <th key={index} className="px-4 py-2 text-left text-xs font-semibold text-white/70 uppercase tracking-wider">
                                {header.replace(/_/g, ' ')}
                            </th>
                        ))}
                    </tr>
                </thead>
                <tbody className="divide-y divide-white/10">
                    {data.rows.map((row, rowIndex) => (
                        <tr key={rowIndex} className="hover:bg-white/5 transition-colors duration-200">
                            {data.headers.map((header, colIndex) => (
                                <td key={colIndex} className="px-4 py-2 whitespace-nowrap text-sm text-white/90">
                                    {row[header] !== undefined ? String(row[header]) : 'N/A'}
                                </td>
                            ))}
                        </tr>
                    ))}
                </tbody>
            </table>
        </div>
    );
};

export default TableRenderer;