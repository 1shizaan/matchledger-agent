import { useState } from 'react'
import api from '../utils/api';
import { Button } from '@/components/ui/button'

const ChatBox = ({ ledgerFile, bankFile }) => {
  const [query, setQuery] = useState('')
  const [response, setResponse] = useState('')
  const [loading, setLoading] = useState(false)

  const handleSend = async () => {
  if (!query || !ledgerFile || !bankFile) {
    alert("Please upload both files and enter a question.")
    return
  }

  const formData = new FormData()
  formData.append('query', query)
  formData.append('ledger_file', ledgerFile)
  formData.append('bank_file', bankFile)

  try {
    setLoading(true)

    // ✅ Check exactly what you're sending
    for (let pair of formData.entries()) {
      console.log(`${pair[0]}:`, pair[1])
    }

    const res = await api.post('/api/chat', formData, {
      // ❗ DO NOT manually set headers — let Axios handle it
    })

    setResponse(res.data.response)
  } catch (error) {
    console.error("Chat error:", error)
    setResponse(`⚠️ Chat Error: ${error.message || 'Unable to process query.'}`)
  } finally {
    setLoading(false)
  }
}


  return (
    <div className="p-4 border rounded-2xl shadow-lg bg-white mt-6 space-y-4">
      <h2 className="text-xl font-bold">💬 Ask ReconBot</h2>
      <input
        type="text"
        value={query}
        onChange={e => setQuery(e.target.value)}
        placeholder="e.g. Show unmatched for April"
        className="w-full p-2 border rounded-md"
      />
      <Button onClick={handleSend} disabled={loading}>
        {loading ? 'Thinking...' : 'Ask'}
      </Button>
      {response && (
        <div className="mt-4 p-3 bg-gray-100 rounded-md whitespace-pre-wrap text-sm text-gray-700 border">
          {response}
        </div>
      )}
    </div>
  )
}

export default ChatBox