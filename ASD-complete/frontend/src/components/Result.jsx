function Result({ result }) {
  if (!result) {
    return null
  }

  return (
    <section className="result-card" aria-live="polite">
      <p className="section-label">Prediction Result</p>
      <div className="result-grid">
        <div>
          <span className="result-key">Class</span>
          <strong className="result-value">{result.prediction}</strong>
        </div>
        <div>
          <span className="result-key">Confidence</span>
          <strong className="result-value">{(result.confidence * 100).toFixed(2)}%</strong>
        </div>
      </div>
    </section>
  )
}

export default Result

