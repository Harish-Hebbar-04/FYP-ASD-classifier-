function Upload({ selectedFile, onFileSelect, disabled }) {
  const handleChange = (event) => {
    const file = event.target.files?.[0] ?? null
    onFileSelect(file)
  }

  return (
    <div className="upload-panel">
      <label className="section-label" htmlFor="image-upload">
        Upload Face Image
      </label>
      <input
        id="image-upload"
        className="file-input"
        type="file"
        accept="image/*"
        onChange={handleChange}
        disabled={disabled}
      />
      <p className="file-hint">{selectedFile ? selectedFile.name : 'Supported formats: JPG, PNG, WEBP, BMP'}</p>
    </div>
  )
}

export default Upload

