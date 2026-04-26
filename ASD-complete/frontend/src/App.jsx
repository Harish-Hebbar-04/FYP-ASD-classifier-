import { useEffect, useState } from 'react'
import axios from 'axios'
import Upload from './components/Upload'
import Result from './components/Result'
import './App.css'

const API_URL = import.meta.env.VITE_API_URL ?? 'http://localhost:8000'

function App() {
  const [selectedFile, setSelectedFile] = useState(null)
  const [previewUrl, setPreviewUrl] = useState('')
  const [result, setResult] = useState(null)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    if (!selectedFile) {
      setPreviewUrl('')
      return undefined
    }

    const objectUrl = URL.createObjectURL(selectedFile)
    setPreviewUrl(objectUrl)

    return () => URL.revokeObjectURL(objectUrl)
  }, [selectedFile])

  const handleFileSelect = (file) => {
    setSelectedFile(file)
    setResult(null)
    setError('')
  }

  const handleSubmit = async (event) => {
    event.preventDefault()

    if (!selectedFile) {
      setError('Please choose a facial image before submitting.')
      return
    }

    const formData = new FormData()
    formData.append('file', selectedFile)

    try {
      setLoading(true)
      setError('')
      setResult(null)

      const response = await axios.post(`${API_URL}/predict`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      })

      setResult(response.data)
    } catch (err) {
      const detail = err.response?.data?.detail
      setError(
        typeof detail === 'string'
          ? detail
          : 'Prediction request failed. Please check the backend and try again.',
      )
    } finally {
      setLoading(false)
    }
  }

  return (
    <main className="app-shell">
      <section className="hero-card">
        <div className="hero-copy">
          <p className="eyebrow">Computer Vision Screening</p>
          <h1>ASD Detection System</h1>
          <p className="subtitle">
            Upload a clear facial image to run the trained PyTorch model and view the predicted class with
            confidence.
          </p>
        </div>

        <form className="form-panel" onSubmit={handleSubmit}>
          <Upload selectedFile={selectedFile} onFileSelect={handleFileSelect} disabled={loading} />

          {previewUrl && (
            <div className="preview-panel">
              <p className="section-label">Image Preview</p>
              <img className="preview-image" src={previewUrl} alt="Selected preview" />
            </div>
          )}

          <button className="submit-button" type="submit" disabled={loading}>
            {loading ? 'Processing...' : 'Predict'}
          </button>

          {loading && (
            <div className="loading-row" aria-live="polite">
              <span className="spinner" />
              <span>Running inference...</span>
            </div>
          )}

          {error && (
            <p className="message message-error" role="alert">
              {error}
            </p>
          )}

          <Result result={result} />
        </form>
      </section>
    </main>
  )
}

export default App

