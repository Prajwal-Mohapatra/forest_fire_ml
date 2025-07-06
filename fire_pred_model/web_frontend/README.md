# Forest Fire Simulation - Frontend Integration Guide

This directory is for your React frontend code. Here's how to integrate with the backend API:

## Backend API Endpoints

The Flask backend provides these endpoints:

### 1. Health Check
```javascript
GET /api/health
// Returns: { status: 'healthy', timestamp: '...', ml_ca_integration: true }
```

### 2. Available Dates
```javascript
GET /api/available-dates
// Returns: { dates: [...], total_count: N }
```

### 3. Run Simulation
```javascript
POST /api/run-simulation
// Body: {
//   date: "2016_04_15",
//   ignition_points: [{"x": 100, "y": 100}, {"x": 150, "y": 120}],
//   simulation_hours: 6,
//   weather_data: {
//     wind_speed: 5.0,
//     wind_direction: 45.0,
//     humidity: 40.0,
//     temperature: 25.0
//   }
// }
// Returns: { success: true, simulation_data: {...}, parameters: {...} }
```

### 4. Configuration
```javascript
GET /api/config
// Returns: API configuration and defaults
```

## React Integration Example

Here's how to call the API from your React components:

### Basic API Service
```javascript
// api.js
const API_BASE_URL = 'http://localhost:5000/api';

export const apiService = {
  async getHealth() {
    const response = await fetch(`${API_BASE_URL}/health`);
    return response.json();
  },

  async getAvailableDates() {
    const response = await fetch(`${API_BASE_URL}/available-dates`);
    return response.json();
  },

  async runSimulation(params) {
    const response = await fetch(`${API_BASE_URL}/run-simulation`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(params)
    });
    return response.json();
  },

  async getConfig() {
    const response = await fetch(`${API_BASE_URL}/config`);
    return response.json();
  }
};
```

### React Component Example
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

  useEffect(() => {
    // Load available dates
    apiService.getAvailableDates()
      .then(data => {
        setDates(data.dates);
        if (data.dates.length > 0) {
          setSelectedDate(data.dates[0].value);
        }
      })
      .catch(console.error);
  }, []);

  const handleMapClick = (event) => {
    // Add ignition point where user clicks
    const rect = event.target.getBoundingClientRect();
    const x = Math.floor((event.clientX - rect.left) / rect.width * 400); // Assuming 400px map width
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
        date: selectedDate,
        ignition_points: ignitionPoints,
        simulation_hours: 6,
        weather_data: {
          wind_speed: 5.0,
          wind_direction: 45.0,
          humidity: 40.0,
          temperature: 25.0
        }
      };

      const result = await apiService.runSimulation(params);
      
      if (result.success) {
        setSimulationData(result.simulation_data);
      } else {
        alert('Simulation failed: ' + result.error);
      }
    } catch (error) {
      alert('Error: ' + error.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="fire-simulation">
      <h1>Forest Fire Spread Simulation</h1>
      
      {/* Date Selection */}
      <div className="controls">
        <label>
          Select Date:
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
        
        <button onClick={runSimulation} disabled={loading}>
          {loading ? 'Running Simulation...' : 'Run Simulation'}
        </button>
        
        <button onClick={() => setIgnitionPoints([])}>
          Clear Ignition Points
        </button>
      </div>

      {/* Map Area */}
      <div className="map-container">
        <div 
          className="map-area" 
          onClick={handleMapClick}
          style={{
            width: '400px',
            height: '400px',
            border: '1px solid #ccc',
            position: 'relative',
            backgroundColor: '#f0f0f0'
          }}
        >
          {/* Render ignition points */}
          {ignitionPoints.map((point, index) => (
            <div
              key={index}
              style={{
                position: 'absolute',
                left: `${(point.x / 400) * 100}%`,
                top: `${(point.y / 400) * 100}%`,
                width: '8px',
                height: '8px',
                backgroundColor: 'red',
                borderRadius: '50%',
                transform: 'translate(-50%, -50%)'
              }}
            />
          ))}
        </div>
        <p>Click on the map to add ignition points</p>
      </div>

      {/* Simulation Results */}
      {simulationData && (
        <div className="results">
          <h2>Simulation Results</h2>
          <div className="animation-container">
            {/* This is where you would render the fire spread animation */}
            <p>Total Frames: {simulationData.simulation_info.total_frames}</p>
            <p>Simulation Duration: {simulationData.simulation_info.simulation_hours} hours</p>
            
            {/* Animation frames would be rendered here */}
            {simulationData.animation_frames.map((frame, index) => (
              <div key={index} className="frame-info">
                Frame {frame.time_step}: {frame.total_pixels} fire pixels
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default FireSimulation;
```

### CSS Styling (ISRO/Fire Theme)
```css
/* styles.css */
.fire-simulation {
  font-family: 'Arial', sans-serif;
  max-width: 1200px;
  margin: 0 auto;
  padding: 20px;
  background: linear-gradient(135deg, #1a1a1a, #2d1810);
  color: #ffffff;
  min-height: 100vh;
}

.fire-simulation h1 {
  text-align: center;
  color: #ff6b35;
  text-shadow: 0 0 10px rgba(255, 107, 53, 0.5);
  margin-bottom: 30px;
}

.controls {
  display: flex;
  gap: 15px;
  margin-bottom: 20px;
  align-items: center;
  background: rgba(255, 255, 255, 0.1);
  padding: 15px;
  border-radius: 8px;
  backdrop-filter: blur(10px);
}

.controls label {
  display: flex;
  flex-direction: column;
  gap: 5px;
}

.controls select, .controls button {
  padding: 8px 12px;
  border: none;
  border-radius: 4px;
  background: #ff6b35;
  color: white;
  cursor: pointer;
  transition: background 0.3s;
}

.controls button:hover {
  background: #e55a2e;
}

.controls button:disabled {
  background: #666;
  cursor: not-allowed;
}

.map-container {
  text-align: center;
  margin: 20px 0;
}

.map-area {
  cursor: crosshair;
  box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
  border-radius: 8px;
  margin: 0 auto;
}

.results {
  background: rgba(255, 255, 255, 0.1);
  padding: 20px;
  border-radius: 8px;
  margin-top: 20px;
}

.results h2 {
  color: #ff6b35;
  margin-bottom: 15px;
}

.frame-info {
  padding: 5px 0;
  border-bottom: 1px solid rgba(255, 255, 255, 0.1);
}
```

## Starting Your Development

1. Create a new React app in this directory:
   ```bash
   npx create-react-app frontend
   cd frontend
   ```

2. Install additional dependencies you might need:
   ```bash
   npm install leaflet react-leaflet  # For map functionality
   npm install recharts              # For charts/graphs
   npm install framer-motion         # For animations
   ```

3. Replace the default App.js with the FireSimulation component above

4. Start your React development server:
   ```bash
   npm start
   ```

5. Make sure the Flask backend is running on localhost:5000

## Integration Checklist

- [ ] Set up React project structure
- [ ] Implement API service layer
- [ ] Create map component for ignition point selection
- [ ] Implement date selection interface
- [ ] Add weather parameter controls
- [ ] Create animation viewer for simulation results
- [ ] Style with ISRO/fire theme
- [ ] Add error handling and loading states
- [ ] Test integration with backend API
- [ ] Optimize for demo presentation

The backend is ready to receive requests and return simulation data. Focus on creating a clean, visually appealing interface that showcases the fire spread simulation capabilities!
