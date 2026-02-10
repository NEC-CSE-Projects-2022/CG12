import { useState } from 'react';
import './Tabs.css';
import UploadPane from './UploadPane';

function Tabs() {
  const [activeTab, setActiveTab] = useState('home');

  const renderContent = () => {
    switch (activeTab) {
      case 'home':
        return (
          <div className="tab-content">
            <h2>Welcome to Flood Prediction System</h2>
            <p>
              This project leverages machine learning and deep learning algorithms to predict flood risks based on historical weather data and environmental factors.
              By analyzing patterns in rainfall, water levels, and other meteorological parameters, our system provides accurate predictions
              that can help communities prepare for potential flooding events.
            </p>
            <p>
              The application uses advanced data processing techniques to validate input data and generate risk assessments.
              Users can upload CSV files containing relevant weather data, and our trained models will analyze the information
              to determine whether the conditions indicate a high or low flood risk. This early warning system can be instrumental
              in disaster preparedness and response planning.
            </p>
            <p>
              Our goal is to create a reliable, user-friendly tool that empowers decision-makers with actionable insights.
              Through continuous model training and validation, we aim to improve prediction accuracy and contribute to
              safer, more resilient communities.
            </p>
          </div>
        );
      case 'about':
        return (
          <div className="tab-content">
            <h2>About This Project</h2>
            <p>
              The Flood Prediction System is designed to analyze weather and environmental data to assess flood risks.
              The system consists of a React-based frontend interface and a Flask backend that processes uploaded CSV files
              containing rainfall and other relevant data points.
            </p>
            <p>
              Machine learning models are trained on historical flood data to identify patterns and correlations between
              various factors and flood occurrences. The system provides real-time predictions and risk assessments based
              on the uploaded data.
            </p>
          </div>
        );
      case 'objectives':
        return (
          <div className="tab-content">
            <h2>Project Objectives</h2>
            <ul className="objectives-list">
              <li>Develop a machine learning model capable of predicting flood risks with high accuracy</li>
              <li>Create an intuitive user interface for easy data upload and result visualization</li>
              <li>Implement robust data validation to ensure input quality</li>
              <li>Provide real-time flood risk assessments based on current weather data</li>
              <li>Support multiple prediction models for comparison and validation</li>
              <li>Enable early warning capabilities for disaster preparedness</li>
              <li>Contribute to community safety and resilience against flood events</li>
            </ul>
          </div>
        );
      case 'procedure':
        return (
          <div className="tab-content">
            <h2>Procedure</h2>
            <div className="procedure-steps">
              <div className="step">
                <div className="step-number">1</div>
                <div className="step-content">
                  <h3>Upload CSV File</h3>
                  <p>User uploads a CSV file containing weather data with a 'rainfall' column</p>
                </div>
              </div>
              <div className="step">
                <div className="step-number">2</div>
                <div className="step-content">
                  <h3>Backend Validation</h3>
                  <p>Flask backend validates the file format and checks for required columns</p>
                </div>
              </div>
              <div className="step">
                <div className="step-number">3</div>
                <div className="step-content">
                  <h3>Data Processing</h3>
                  <p>System processes the data and calculates relevant metrics like average rainfall</p>
                </div>
              </div>
              <div className="step">
                <div className="step-number">4</div>
                <div className="step-content">
                  <h3>Risk Prediction</h3>
                  <p>ML model analyzes the data and predicts flood risk level</p>
                </div>
              </div>
              <div className="step">
                <div className="step-number">5</div>
                <div className="step-content">
                  <h3>Display Results</h3>
                  <p>System displays prediction results with risk assessment and supporting metrics</p>
                </div>
              </div>
            </div>
          </div>
        );
      case 'validation':
        return (
          <div className="tab-content">
            <h2>Validation / Results</h2>
            <UploadPane />
          </div>
        );
      default:
        return null;
    }
  };

  return (
    <div className="tabs-container">
      <div className="tabs-header">
        <button
          className={`tab-btn ${activeTab === 'home' ? 'active' : ''}`}
          onClick={() => setActiveTab('home')}
        >
          Home
        </button>
        <button
          className={`tab-btn ${activeTab === 'about' ? 'active' : ''}`}
          onClick={() => setActiveTab('about')}
        >
          About Project
        </button>
        <button
          className={`tab-btn ${activeTab === 'objectives' ? 'active' : ''}`}
          onClick={() => setActiveTab('objectives')}
        >
          Objectives
        </button>
        <button
          className={`tab-btn ${activeTab === 'procedure' ? 'active' : ''}`}
          onClick={() => setActiveTab('procedure')}
        >
          Procedure
        </button>
        <button
          className={`tab-btn ${activeTab === 'validation' ? 'active' : ''}`}
          onClick={() => setActiveTab('validation')}
        >
          Validation / Results
        </button>
      </div>
      <div className="tabs-body">{renderContent()}</div>
    </div>
  );
}

export default Tabs;
