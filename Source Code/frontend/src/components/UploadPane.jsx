import { useState } from 'react';
import './UploadPane.css';

function UploadPane() {
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
    setResult(null);
    setError(null);
  };

  const handleValidate = async () => {
    if (!file) {
      setError('Please select a file first');
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch('http://localhost:5000/upload', {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();

      if (response.ok) {
        setResult(data);
      } else {
        setError(data.error || 'An error occurred');
      }
    } catch (err) {
      setError('Failed to connect to backend. Make sure Flask server is running on port 5000.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="upload-pane">
      <div className="upload-section">
        <label htmlFor="csv-file" className="file-label">
          Select CSV File
        </label>
        <input
          type="file"
          id="csv-file"
          accept=".csv"
          onChange={handleFileChange}
          className="file-input"
        />
        {file && <div className="file-name">Selected: {file.name}</div>}
        <button
          className="validate-btn"
          onClick={handleValidate}
          disabled={loading}
        >
          {loading ? 'Processing...' : 'Validate'}
        </button>
      </div>

      {error && (
        <div className="result-box error-box">
          <h3>Error</h3>
          <p>{error}</p>
        </div>
      )}

      {result && (
        <div className={`result-box ${result.risk_level === 'high' ? 'high-risk' : 'low-risk'}`}>
          <h3>Prediction Result</h3>
          <div className="result-content">
            <div className="result-item">
              <strong>Prediction:</strong> {result.prediction}
            </div>
            <div className="result-item">
              <strong>Average Rainfall:</strong> {result.average_rainfall} mm
            </div>
            <div className="result-item">
              <strong>Records Analyzed:</strong> {result.records_analyzed}
            </div>
            <div className="result-item">
              <strong>Model Accuracy:</strong>{' '}
              {result.accuracy !== null && result.accuracy !== undefined
                ? `${result.accuracy}%`
                : (result.accuracy_message || 'Not available')}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default UploadPane;
