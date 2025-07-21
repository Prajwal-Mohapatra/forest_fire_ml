# Forest Fire Simulation - React Frontend Integration Guide

This guide provides comprehensive examples for integrating a React frontend with the Forest Fire Simulation API.

## API Endpoints Overview

The Flask backend provides these endpoints:

### Core Endpoints
- `GET /api/health` - Health check and system status
- `GET /api/available_dates` - Get available simulation dates
- `POST /api/simulate` - Run fire simulation
- `GET /api/simulation/<id>/status` - Get simulation status
- `GET /api/simulation/<id>/animation` - Get animation data
- `GET /api/config` - Get API configuration

### Enhanced Endpoints
- `POST /api/multiple-scenarios` - Run comparison scenarios
- `GET /api/simulation-cache/<id>` - Get cached results
- `GET /api/export-results/<id>` - Export simulation data

## React Integration Examples

### 1. API Service Layer

```javascript
// api.js
const API_BASE_URL = 'http://localhost:5000/api';

export const apiService = {
  async getHealth() {
    const response = await fetch(`${API_BASE_URL}/health`);
    return response.json();
  },

  async getAvailableDates() {
    const response = await fetch(`${API_BASE_URL}/available_dates`);
    return response.json();
  },

  async runSimulation(params) {
    const response = await fetch(`${API_BASE_URL}/simulate`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(params)
    });
    return response.json();
  },

  async getSimulationStatus(simulationId) {
    const response = await fetch(`${API_BASE_URL}/simulation/${simulationId}/status`);
    return response.json();
  },

  async getAnimationData(simulationId) {
    const response = await fetch(`${API_BASE_URL}/simulation/${simulationId}/animation`);
    return response.json();
  },

  async getConfig() {
    const response = await fetch(`${API_BASE_URL}/config`);
    return response.json();
  },

  async runMultipleScenarios(params) {
    const response = await fetch(`${API_BASE_URL}/multiple-scenarios`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(params)
    });
    return response.json();
  }
};
```

### 2. Main Fire Simulation Component

```javascript
// FireSimulation.jsx
import React, { useState, useEffect } from 'react';
import { apiService } from './api';

const FireSimulation = () => {
  const [dates, setDates] = useState([]);
  const [selectedDate, setSelectedDate] = useState('');
  const [ignitionPoints, setIgnitionPoints] = useState([]);
  const [simulationData, setSimulationData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [currentSimulation, setCurrentSimulation] = useState(null);
  const [weatherParams, setWeatherParams] = useState({
    wind_direction: 45,
    wind_speed: 15,
    temperature: 30,
    relative_humidity: 40
  });

  useEffect(() => {
    loadAvailableDates();
    checkAPIHealth();
  }, []);

  const loadAvailableDates = async () => {
    try {
      const data = await apiService.getAvailableDates();
      setDates(data.available_dates || []);
      if (data.available_dates && data.available_dates.length > 0) {
        setSelectedDate(data.available_dates[0].value);
      }
    } catch (error) {
      console.error('Failed to load dates:', error);
    }
  };

  const checkAPIHealth = async () => {
    try {
      const health = await apiService.getHealth();
      console.log('API Status:', health);
    } catch (error) {
      console.error('API health check failed:', error);
    }
  };

  const handleMapClick = (event) => {
    const rect = event.target.getBoundingClientRect();
    const x = Math.floor((event.clientX - rect.left) / rect.width * 400);
    const y = Math.floor((event.clientY - rect.top) / rect.height * 400);
    
    setIgnitionPoints(prev => [...prev, { x, y }]);
  };

  const runSimulation = async () => {
    if (!selectedDate || ignitionPoints.length === 0) {
      alert('Please select a date and add ignition points');
      return;
    }

    setLoading(true);
    try {
      const params = {
        ignition_points: ignitionPoints.map(p => [p.x, p.y]),
        weather_params: weatherParams,
        simulation_hours: 6,
        date: selectedDate,
        use_ml_prediction: true
      };

      const result = await apiService.runSimulation(params);
      
      if (result.simulation_id) {
        setCurrentSimulation(result.simulation_id);
        
        // Poll for completion
        const pollStatus = async () => {
          const status = await apiService.getSimulationStatus(result.simulation_id);
          
          if (status.status === 'completed') {
            const animationData = await apiService.getAnimationData(result.simulation_id);
            setSimulationData(animationData);
            setLoading(false);
          } else if (status.status === 'failed') {
            alert('Simulation failed');
            setLoading(false);
          } else {
            setTimeout(pollStatus, 2000); // Poll every 2 seconds
          }
        };
        
        setTimeout(pollStatus, 1000);
      }
    } catch (error) {
      alert('Error: ' + error.message);
      setLoading(false);
    }
  };

  const updateWeatherParam = (param, value) => {
    setWeatherParams(prev => ({
      ...prev,
      [param]: parseFloat(value)
    }));
  };

  return (
    <div className="fire-simulation">
      <header className="simulation-header">
        <h1>🔥 Forest Fire Spread Simulation</h1>
        <p>Interactive fire behavior modeling system</p>
      </header>
      
      {/* Controls Panel */}
      <div className="controls-panel">
        <div className="control-group">
          <label>
            Simulation Date:
            <select 
              value={selectedDate} 
              onChange={(e) => setSelectedDate(e.target.value)}
            >
              {dates.map(date => (
                <option key={date.value} value={date.value}>
                  {date.label}
                </option>
              ))}
            </select>
          </label>
        </div>

        {/* Weather Controls */}
        <div className="weather-controls">
          <h3>Weather Parameters</h3>
          <div className="weather-grid">
            <label>
              Wind Direction (°):
              <input 
                type="range" 
                min="0" 
                max="360" 
                value={weatherParams.wind_direction}
                onChange={(e) => updateWeatherParam('wind_direction', e.target.value)}
              />
              <span>{weatherParams.wind_direction}°</span>
            </label>
            
            <label>
              Wind Speed (km/h):
              <input 
                type="range" 
                min="0" 
                max="50" 
                value={weatherParams.wind_speed}
                onChange={(e) => updateWeatherParam('wind_speed', e.target.value)}
              />
              <span>{weatherParams.wind_speed} km/h</span>
            </label>
            
            <label>
              Temperature (°C):
              <input 
                type="range" 
                min="10" 
                max="50" 
                value={weatherParams.temperature}
                onChange={(e) => updateWeatherParam('temperature', e.target.value)}
              />
              <span>{weatherParams.temperature}°C</span>
            </label>
            
            <label>
              Humidity (%):
              <input 
                type="range" 
                min="10" 
                max="90" 
                value={weatherParams.relative_humidity}
                onChange={(e) => updateWeatherParam('relative_humidity', e.target.value)}
              />
              <span>{weatherParams.relative_humidity}%</span>
            </label>
          </div>
        </div>

        <div className="action-buttons">
          <button 
            onClick={runSimulation} 
            disabled={loading || ignitionPoints.length === 0}
            className="run-button"
          >
            {loading ? '🔄 Running Simulation...' : '🚀 Run Simulation'}
          </button>
          
          <button 
            onClick={() => setIgnitionPoints([])}
            className="clear-button"
          >
            🗑️ Clear Points
          </button>
        </div>
      </div>

      {/* Map Interface */}
      <div className="map-container">
        <h3>Ignition Points ({ignitionPoints.length})</h3>
        <div 
          className="map-area" 
          onClick={handleMapClick}
        >
          {ignitionPoints.map((point, index) => (
            <div
              key={index}
              className="ignition-point"
              style={{
                left: `${(point.x / 400) * 100}%`,
                top: `${(point.y / 400) * 100}%`
              }}
              title={`Point ${index + 1}: (${point.x}, ${point.y})`}
            />
          ))}
        </div>
        <p className="map-instructions">
          🖱️ Click on the map to add ignition points
        </p>
      </div>

      {/* Simulation Results */}
      {simulationData && (
        <div className="results-panel">
          <h2>🎯 Simulation Results</h2>
          <div className="results-stats">
            <div className="stat-card">
              <h4>Total Duration</h4>
              <p>{simulationData.total_hours} hours</p>
            </div>
            <div className="stat-card">
              <h4>Animation Frames</h4>
              <p>{simulationData.frame_urls?.length || 0}</p>
            </div>
            <div className="stat-card">
              <h4>Scenario ID</h4>
              <p>{simulationData.scenario_id}</p>
            </div>
          </div>
          
          {/* Hourly Statistics */}
          {simulationData.hourly_statistics && (
            <div className="hourly-stats">
              <h3>📊 Hourly Progression</h3>
              <div className="stats-timeline">
                {simulationData.hourly_statistics.map((hour, index) => (
                  <div key={index} className="hour-stat">
                    <strong>Hour {index + 1}:</strong>
                    <span>🔥 {hour.total_burning_cells || 0} burning cells</span>
                    <span>📏 {hour.burned_area_km2?.toFixed(2) || 0} km²</span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default FireSimulation;
```

### 3. CSS Styling (ISRO/Fire Theme)

```css
/* FireSimulation.css */
.fire-simulation {
  font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
  max-width: 1400px;
  margin: 0 auto;
  padding: 20px;
  background: linear-gradient(135deg, #1a1a1a, #2d1810);
  color: #ffffff;
  min-height: 100vh;
  border-radius: 12px;
}

.simulation-header {
  text-align: center;
  margin-bottom: 30px;
  padding: 20px;
  background: rgba(255, 107, 53, 0.1);
  border-radius: 8px;
  border: 1px solid rgba(255, 107, 53, 0.3);
}

.simulation-header h1 {
  color: #ff6b35;
  text-shadow: 0 0 15px rgba(255, 107, 53, 0.6);
  margin-bottom: 10px;
  font-size: 2.5rem;
}

.controls-panel {
  background: rgba(255, 255, 255, 0.1);
  padding: 25px;
  border-radius: 12px;
  backdrop-filter: blur(10px);
  margin-bottom: 25px;
  border: 1px solid rgba(255, 255, 255, 0.2);
}

.control-group {
  margin-bottom: 20px;
}

.control-group label {
  display: flex;
  flex-direction: column;
  gap: 8px;
  color: #ffffff;
  font-weight: 500;
}

.control-group select {
  padding: 10px;
  border: none;
  border-radius: 6px;
  background: #ff6b35;
  color: white;
  font-size: 14px;
  cursor: pointer;
  transition: all 0.3s ease;
}

.control-group select:hover {
  background: #e55a2e;
  transform: translateY(-1px);
}

.weather-controls {
  margin: 25px 0;
}

.weather-controls h3 {
  color: #ff6b35;
  margin-bottom: 15px;
  border-bottom: 2px solid rgba(255, 107, 53, 0.3);
  padding-bottom: 8px;
}

.weather-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: 20px;
}

.weather-grid label {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.weather-grid input[type="range"] {
  width: 100%;
  height: 6px;
  border-radius: 3px;
  background: #444;
  outline: none;
  cursor: pointer;
}

.weather-grid input[type="range"]::-webkit-slider-thumb {
  appearance: none;
  width: 18px;
  height: 18px;
  border-radius: 50%;
  background: #ff6b35;
  cursor: pointer;
  box-shadow: 0 0 10px rgba(255, 107, 53, 0.5);
}

.action-buttons {
  display: flex;
  gap: 15px;
  margin-top: 25px;
}

.run-button, .clear-button {
  padding: 12px 24px;
  border: none;
  border-radius: 8px;
  font-size: 16px;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.3s ease;
  min-width: 140px;
}

.run-button {
  background: linear-gradient(45deg, #ff6b35, #e55a2e);
  color: white;
  box-shadow: 0 4px 12px rgba(255, 107, 53, 0.3);
}

.run-button:hover:not(:disabled) {
  transform: translateY(-2px);
  box-shadow: 0 6px 16px rgba(255, 107, 53, 0.4);
}

.run-button:disabled {
  background: #666;
  cursor: not-allowed;
  transform: none;
}

.clear-button {
  background: linear-gradient(45deg, #666, #555);
  color: white;
}

.clear-button:hover {
  background: linear-gradient(45deg, #777, #666);
  transform: translateY(-2px);
}

.map-container {
  text-align: center;
  margin: 25px 0;
  background: rgba(255, 255, 255, 0.1);
  padding: 20px;
  border-radius: 12px;
  border: 1px solid rgba(255, 255, 255, 0.2);
}

.map-container h3 {
  color: #ff6b35;
  margin-bottom: 15px;
}

.map-area {
  width: 600px;
  height: 400px;
  background: linear-gradient(135deg, #2a4d3a, #1e3c29);
  border: 2px solid #ff6b35;
  border-radius: 8px;
  margin: 0 auto 10px;
  position: relative;
  cursor: crosshair;
  box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
  overflow: hidden;
}

.ignition-point {
  position: absolute;
  width: 12px;
  height: 12px;
  background: radial-gradient(circle, #ff6b35, #ff0000);
  border-radius: 50%;
  transform: translate(-50%, -50%);
  animation: pulse 2s infinite;
  box-shadow: 0 0 10px rgba(255, 107, 53, 0.8);
}

@keyframes pulse {
  0%, 100% { transform: translate(-50%, -50%) scale(1); }
  50% { transform: translate(-50%, -50%) scale(1.3); }
}

.map-instructions {
  color: #cccccc;
  font-style: italic;
  margin-top: 10px;
}

.results-panel {
  background: rgba(255, 255, 255, 0.1);
  padding: 25px;
  border-radius: 12px;
  border: 1px solid rgba(255, 255, 255, 0.2);
  margin-top: 25px;
}

.results-panel h2 {
  color: #ff6b35;
  margin-bottom: 20px;
  text-align: center;
}

.results-stats {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 15px;
  margin-bottom: 25px;
}

.stat-card {
  background: rgba(255, 107, 53, 0.1);
  padding: 15px;
  border-radius: 8px;
  text-align: center;
  border: 1px solid rgba(255, 107, 53, 0.3);
}

.stat-card h4 {
  color: #ff6b35;
  margin-bottom: 8px;
  font-size: 14px;
  text-transform: uppercase;
}

.stat-card p {
  font-size: 18px;
  font-weight: bold;
  margin: 0;
}

.hourly-stats h3 {
  color: #ff6b35;
  margin-bottom: 15px;
  border-bottom: 2px solid rgba(255, 107, 53, 0.3);
  padding-bottom: 8px;
}

.stats-timeline {
  display: grid;
  gap: 10px;
}

.hour-stat {
  background: rgba(255, 255, 255, 0.05);
  padding: 12px;
  border-radius: 6px;
  display: grid;
  grid-template-columns: 100px 1fr 1fr;
  gap: 15px;
  align-items: center;
  border-left: 3px solid #ff6b35;
}

.hour-stat strong {
  color: #ff6b35;
}

/* Responsive Design */
@media (max-width: 768px) {
  .fire-simulation {
    padding: 15px;
  }
  
  .map-area {
    width: 100%;
    max-width: 400px;
    height: 300px;
  }
  
  .weather-grid {
    grid-template-columns: 1fr;
  }
  
  .action-buttons {
    flex-direction: column;
  }
  
  .results-stats {
    grid-template-columns: 1fr;
  }
  
  .hour-stat {
    grid-template-columns: 1fr;
    text-align: center;
  }
}
```

## Development Setup Instructions

### 1. Create React Project
```bash
npx create-react-app fire-simulation-frontend
cd fire-simulation-frontend
```

### 2. Install Dependencies
```bash
npm install leaflet react-leaflet  # For advanced mapping
npm install recharts             # For data visualization
npm install framer-motion        # For animations
npm install axios               # For API calls (alternative to fetch)
```

### 3. Project Structure
```
src/
├── components/
│   ├── FireSimulation.jsx
│   ├── MapInterface.jsx
│   ├── WeatherControls.jsx
│   └── ResultsViewer.jsx
├── services/
│   └── api.js
├── styles/
│   └── FireSimulation.css
└── App.js
```

### 4. Environment Configuration
Create `.env.local`:
```
REACT_APP_API_BASE_URL=http://localhost:5000/api
REACT_APP_WEBSOCKET_URL=ws://localhost:5000/ws
```

## Integration Checklist

- [ ] Set up React project structure
- [ ] Implement API service layer with error handling
- [ ] Create responsive map component
- [ ] Add weather parameter controls with validation
- [ ] Implement real-time simulation status polling
- [ ] Create animation viewer for results
- [ ] Style with ISRO/fire theme
- [ ] Add loading states and progress indicators
- [ ] Implement error boundaries
- [ ] Add accessibility features
- [ ] Test on mobile devices
- [ ] Optimize performance for large datasets
- [ ] Add user preferences storage
- [ ] Implement session management
- [ ] Create help/tutorial system

## Advanced Features

### WebSocket Integration (Future)
```javascript
// For real-time simulation updates
const useWebSocket = (simulationId) => {
  const [progress, setProgress] = useState(0);
  
  useEffect(() => {
    const ws = new WebSocket(`ws://localhost:5000/ws/simulation/${simulationId}`);
    
    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      setProgress(data.progress);
    };
    
    return () => ws.close();
  }, [simulationId]);
  
  return progress;
};
```

### Map Integration with Leaflet
```javascript
import { MapContainer, TileLayer, Marker, Popup } from 'react-leaflet';

const MapInterface = ({ ignitionPoints, onMapClick }) => {
  return (
    <MapContainer center={[30.3165, 78.0322]} zoom={10} onClick={onMapClick}>
      <TileLayer url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />
      {ignitionPoints.map((point, index) => (
        <Marker key={index} position={[point.lat, point.lon]}>
          <Popup>Ignition Point {index + 1}</Popup>
        </Marker>
      ))}
    </MapContainer>
  );
};
```

This guide provides a complete foundation for building a professional React frontend for the fire simulation system. The examples are production-ready and follow modern React best practices.
