import { useState } from 'react'
import { Button } from '@/components/ui/button'

const FileUploader = ({ onUpload }) => {
  const [ledgerFile, setLedgerFile] = useState(null)
  const [bankFile, setBankFile] = useState(null)
  const [email, setEmail] = useState('') // 👈 New state for email input

  const handleSubmit = () => {
    if (ledgerFile && bankFile) {
      // Pass email along with the files to the onUpload function
      onUpload(ledgerFile, bankFile, email) // 👈 Update onUpload call
    }
  }

  return (
    <div className="space-y-4 p-4 border rounded-2xl shadow-lg bg-white">
      <h2 className="text-xl font-bold">Upload Files</h2>
      <input type="file" accept=".csv" onChange={e => setLedgerFile(e.target.files[0])} />
      <input type="file" accept=".csv" onChange={e => setBankFile(e.target.files[0])} />

      {/* 👈 New email input field */}
      <input
        type="email"
        placeholder="Optional: Enter your email"
        className="p-2 border rounded-md w-full"
        value={email}
        onChange={e => setEmail(e.target.value)}
      />

      <Button onClick={handleSubmit}>Reconcile</Button>
    </div>
  )
}

export default FileUploader