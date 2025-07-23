const ResultsTable = ({ title, data }) => {
  // 🛠️ Flatten deeply nested rows for uniform rendering
  const flattenRow = (row) => {
    const flat = {};
    for (const key in row) {
      if (typeof row[key] === "object" && row[key] !== null && !Array.isArray(row[key])) {
        // Merge nested object keys with prefix
        for (const subKey in row[key]) {
          flat[`${key}_${subKey}`] = row[key][subKey];
        }
      } else {
        flat[key] = row[key];
      }
    }
    return flat;
  };

  const flattenedData = data.map(flattenRow);
  const headers = flattenedData.length > 0 ? Object.keys(flattenedData[0]) : [];

  return (
    <div className="mt-6">
      <h3 className="text-lg font-semibold mb-2">{title}</h3>
      <div className="overflow-auto border rounded-xl shadow-sm">
        <table className="min-w-full text-sm">
          <thead className="bg-gray-100">
            <tr>
              {headers.map((col) => (
                <th key={col} className="px-4 py-2 text-left">{col}</th>
              ))}
            </tr>
          </thead>
          <tbody>
           {flattenedData.map((row, idx) => (
             <tr key={idx} className="even:bg-gray-50">
               {headers.map((col, i) => {
                 const val = row[col];
                 return (
                   <td key={i} className="px-4 py-2">
                     {typeof val === 'object' && val !== null
                       ? JSON.stringify(val)
                       : val}
                   </td>
                 );
               })}
             </tr>
           ))}
         </tbody>

        </table>
      </div>
    </div>
  );
};

export default ResultsTable;
