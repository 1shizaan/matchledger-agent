export const SummaryCard = ({ matched, unmatchedLedger, unmatchedBank }) => {
  return (
    <div className="grid grid-cols-3 gap-4 mt-6">
      <div className="bg-green-100 p-4 rounded-xl shadow-sm">
        <p className="text-sm">Matched</p>
        <h2 className="text-2xl font-bold">{matched.length}</h2>
      </div>
      <div className="bg-yellow-100 p-4 rounded-xl shadow-sm">
        <p className="text-sm">Unmatched Ledger</p>
        <h2 className="text-2xl font-bold">{unmatchedLedger.length}</h2>
      </div>
      <div className="bg-red-100 p-4 rounded-xl shadow-sm">
        <p className="text-sm">Unmatched Bank</p>
        <h2 className="text-2xl font-bold">{unmatchedBank.length}</h2>
      </div>
    </div>
  )
}