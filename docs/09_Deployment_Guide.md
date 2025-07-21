# 🚀 Deployment Guide

## Overview

This comprehensive deployment guide covers all aspects of setting up and running the Forest Fire Spread Simulation system across different environments - from local development to production cloud deployment.

## System Requirements

### Minimum Requirements

- **OS**: Linux (Ubuntu 18.04+), Windows 10+, or macOS 10.15+
- **RAM**: 8GB (16GB recommended for full-state simulations)
- **Storage**: 50GB free space for complete system and datasets
- **GPU**: Optional but recommended (NVIDIA GPU with 4GB+ VRAM)
- **Network**: Broadband internet for data downloads

### Recommended Requirements

- **OS**: Ubuntu 20.04 LTS or Windows 11
- **RAM**: 32GB for optimal performance
- **Storage**: 100GB+ SSD storage
- **GPU**: NVIDIA RTX 3060+ or Tesla T4+ (8GB+ VRAM)
- **CPU**: 8+ cores (Intel i7/AMD Ryzen 7 or better)

### Cloud Requirements

- **AWS**: g4dn.xlarge or larger (GPU instance)
- **Google Cloud**: n1-standard-4 with T4 GPU
- **Azure**: Standard_NC6s_v3 or equivalent

## Installation Instructions

### 1. Repository Setup

```bash
# Clone the repository
git clone https://github.com/Prajwal-Mohapatra/forest_fire_spread.git
cd forest_fire_spread

# Verify repository structure
ls -la
# Should show: cellular_automata/, forest_fire_ml/, docs/, etc.
```

### 2. Python Environment Setup

#### Using Conda (Recommended)

```bash
# Create conda environment
conda create -n fire_prediction python=3.9
conda activate fire_prediction

# Install core dependencies
conda install tensorflow-gpu=2.8.0 -c conda-forge
conda install rasterio geopandas matplotlib seaborn -c conda-forge
conda install jupyter ipywidgets -c conda-forge

# Install additional packages with pip
pip install plotly==5.14.0
pip install opencv-python==4.5.5.64
pip install scikit-learn==1.1.0
```

#### Using pip (Alternative)

```bash
# Create virtual environment
python -m venv fire_prediction_env
source fire_prediction_env/bin/activate  # Linux/macOS
# or
fire_prediction_env\Scripts\activate  # Windows

# Install requirements
pip install -r requirements.txt
```

#### Requirements.txt

```txt
tensorflow==2.8.0
numpy==1.21.6
rasterio==1.3.0
geopandas==0.11.1
matplotlib==3.5.2
seaborn==0.11.2
plotly==5.14.0
jupyter==1.0.0
ipywidgets==7.7.1
opencv-python==4.5.5.64
scikit-learn==1.1.0
pandas==1.4.3
scipy==1.8.1
pyproj==3.3.1
shapely==1.8.2
xarray==0.20.2
netcdf4==1.6.0
```

### 3. GPU Configuration (Optional but Recommended)

#### NVIDIA GPU Setup

```bash
# Check GPU availability
nvidia-smi

# Install CUDA toolkit (if not already installed)
# Ubuntu
sudo apt update
sudo apt install nvidia-cuda-toolkit

# Verify TensorFlow GPU detection
python -c "import tensorflow as tf; print('GPU Available:', tf.config.list_physical_devices('GPU'))"
```

#### TensorFlow GPU Verification

```python
import tensorflow as tf

# Check TensorFlow version
print(f"TensorFlow version: {tf.__version__}")

# Check GPU availability
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"GPU(s) detected: {len(gpus)}")
    for i, gpu in enumerate(gpus):
        print(f"  GPU {i}: {gpu}")
else:
    print("No GPU detected - using CPU")

# Test GPU operation
with tf.device('/GPU:0' if gpus else '/CPU:0'):
    a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
    c = tf.matmul(a, b)
    print(f"Test computation result: {c}")
```

## Data Setup

### 1. Download Required Datasets

#### Option A: Kaggle Datasets (Recommended)

```bash
# Install Kaggle API
pip install kaggle

# Configure Kaggle credentials (kaggle.json in ~/.kaggle/)
# Download stacked dataset
kaggle datasets download -d your-username/stacked-fire-probability-prediction-dataset
unzip stacked-fire-probability-prediction-dataset.zip -d data/

# Download unstacked dataset (optional)
kaggle datasets download -d your-username/datasets
unzip datasets.zip -d data/
```

#### Option B: Manual Data Collection

```bash
# Create data directories
mkdir -p data/raw_datasets/{dem,era5,lulc,ghsl,viirs}
mkdir -p data/stacked_datasets
mkdir -p data/processed

# Download individual datasets (requires Google Earth Engine setup)
# See dataset_collection/ scripts for automated download
```

### 2. Verify Data Integrity

```python
# Run data validation script
python scripts/validate_datasets.py --data-dir data/stacked_datasets

# Expected output:
# ✅ Found 59 stacked dataset files
# ✅ All datasets pass spatial consistency checks
# ✅ Value ranges validated for all bands
# ✅ Missing data within acceptable limits (<5%)
```

### 3. Download Pre-trained Model

```bash
# Download trained ResUNet-A model
wget https://github.com/Prajwal-Mohapatra/forest_fire_spread/releases/download/v1.0/final_model.h5
mv final_model.h5 forest_fire_ml/outputs/

# Verify model
python -c "
import tensorflow as tf
model = tf.keras.models.load_model('forest_fire_ml/outputs/final_model.h5', compile=False)
print(f'Model loaded successfully: {model.input_shape} -> {model.output_shape}')
"
```

## Component Deployment

### 1. ML Model Deployment

#### Test ML Prediction Pipeline

```bash
cd forest_fire_ml/

# Test with sample data
python predict.py --input data/stacked_datasets/stack_2016_05_15.tif \
                  --output outputs/test_prediction \
                  --model outputs/final_model.h5

# Expected output:
# ✅ Model loaded successfully
# ✅ Input image loaded: (500, 400, 9)
# 📊 Processing 156 patches
# ✅ Fire probability map saved to outputs/test_prediction/fire_probability_map.tif
```

#### ML Service Configuration

```python
# config/ml_config.py
ML_CONFIG = {
    'model_path': 'forest_fire_ml/outputs/final_model.h5',
    'patch_size': 256,
    'overlap': 64,
    'batch_size': 4,
    'gpu_memory_fraction': 0.7,
    'output_format': 'GTiff',
    'compression': 'LZW'
}
```

### 2. Cellular Automata Engine Deployment

#### Test CA Engine

```bash
cd cellular_automata

# Test CA engine initialization
python -c "
from ca_engine import ForestFireCA
ca = ForestFireCA(use_gpu=True)
print('✅ CA engine initialized successfully')
print(f'GPU available: {ca.use_gpu}')
"

# Test quick simulation
python test_ca_engine.py --probability-map ../data/test_probability.tif \
                        --ignition-lat 30.3165 --ignition-lon 78.0322 \
                        --simulation-hours 6
```

#### CA Configuration

```python
# config/ca_config.py
CA_CONFIG = {
    'default_simulation_hours': 6,
    'max_simulation_hours': 24,
    'base_spread_rate': 0.1,
    'wind_factor': 0.05,
    'neighbor_weight': 0.4,
    'suppression_threshold': 0.5,
    'output_format': 'GTiff',
    'save_frequency': 1  # Save every hour
}
```

### 3. Integration Bridge Deployment

#### Test ML-CA Bridge

```bash
# Test complete pipeline
python cellular_automata/integration/test_bridge.py \
    --input-data data/stacked_datasets/stack_2016_05_15.tif \
    --ignition-points "[(78.0322, 30.3165)]" \
    --weather "{'wind_speed': 15, 'wind_direction': 225, 'temperature': 30, 'humidity': 40}" \
    --simulation-hours 6

# Expected output:
# 🚀 Running complete ML→CA pipeline
# 📊 Step 1: ML Prediction Generation
# ✅ ML prediction generated: outputs/probability_map_2016_05_15.tif
# 🔥 Step 2: CA Simulation
# ✅ CA simulation completed: ml_ca_sim_20241201_143022
# 📦 Step 3: Results Packaging
# ✅ Complete pipeline results saved
```

## Web Interface Deployment

### 1. Frontend Setup (React)

```bash
cd web_interface/frontend

# Install Node.js dependencies
npm install

# Install specific packages
npm install @mui/material @emotion/react @emotion/styled
npm install @mui/icons-material
npm install leaflet react-leaflet
npm install chart.js react-chartjs-2
npm install socket.io-client
npm install axios

# Create environment file
cat > .env << EOF
REACT_APP_API_BASE_URL=http://localhost:5000
REACT_APP_WEBSOCKET_URL=http://localhost:5000
REACT_APP_MAP_CENTER_LAT=30.0
REACT_APP_MAP_CENTER_LON=79.0
EOF

# Test development server
npm start
```

### 2. Backend Setup (Node.js)

```bash
cd web_interface/backend

# Initialize Node.js project
npm init -y

# Install dependencies
npm install express cors socket.io
npm install multer helmet morgan
npm install child_process fs path

# Create server configuration
cat > config/server.js << EOF
module.exports = {
    port: process.env.PORT || 5000,
    cors: {
        origin: "http://localhost:3000",
        methods: ["GET", "POST"]
    },
    python: {
        executable: "python",
        script_path: "../scripts/"
    },
    upload: {
        dest: "uploads/",
        max_size: "50mb"
    }
};
EOF

# Test backend server
node server.js
```

### 3. Web Interface Integration

#### API Route Setup

```javascript
// routes/simulation.js
const express = require("express");
const { spawn } = require("child_process");
const router = express.Router();

router.post("/start", async (req, res) => {
	const { dateStr, ignitionPoints, weatherParams, simulationHours } = req.body;

	try {
		const pythonProcess = spawn("python", [
			"../scripts/run_simulation.py",
			"--date",
			dateStr,
			"--ignition-points",
			JSON.stringify(ignitionPoints),
			"--weather",
			JSON.stringify(weatherParams),
			"--hours",
			simulationHours.toString(),
		]);

		// Handle process output and send response
		// Implementation continues...
	} catch (error) {
		res.status(500).json({ error: error.message });
	}
});

module.exports = router;
```

### 4. WebSocket Configuration

```javascript
// websocket/simulation.js
const socketIo = require("socket.io");

function setupWebSocket(server) {
	const io = socketIo(server);

	io.on("connection", (socket) => {
		console.log("Client connected:", socket.id);

		socket.on("start-simulation", async (params) => {
			try {
				// Start simulation with progress updates
				await runSimulationWithProgress(params, (progress) => {
					socket.emit("simulation-progress", progress);
				});
			} catch (error) {
				socket.emit("simulation-error", { error: error.message });
			}
		});
	});

	return io;
}

module.exports = setupWebSocket;
```

## Environment-Specific Deployments

### 1. Local Development Environment

#### Complete Local Setup Script

```bash
#!/bin/bash
# deploy_local.sh

echo "🚀 Setting up Forest Fire Simulation - Local Development"

# 1. Environment setup
conda create -n fire_prediction python=3.9 -y
conda activate fire_prediction
pip install -r requirements.txt

# 2. Data verification
python scripts/verify_installation.py

# 3. Start services
echo "Starting backend server..."
cd web_interface/backend && npm start &
BACKEND_PID=$!

echo "Starting frontend development server..."
cd ../frontend && npm start &
FRONTEND_PID=$!

echo "✅ Local deployment complete!"
echo "Frontend: http://localhost:3000"
echo "Backend API: http://localhost:5000"
echo "Press Ctrl+C to stop services"

# Cleanup on exit
trap "kill $BACKEND_PID $FRONTEND_PID" EXIT
wait
```

### 2. Kaggle Environment Deployment

#### Kaggle Setup Script

```python
# kaggle_setup.py
import os
import subprocess
import sys

def setup_kaggle_environment():
    """Setup environment for Kaggle execution"""

    # Install additional packages not in Kaggle default
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'plotly==5.14.0'])

    # Configure paths for Kaggle environment
    os.environ['PYTHONPATH'] = '/kaggle/working/forest_fire_spread'

    # Verify GPU availability
    import tensorflow as tf
    gpus = tf.config.list_physical_devices('GPU')
    print(f"✅ Kaggle GPU setup: {len(gpus)} device(s) available")

    # Configure GPU memory growth
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    return True

if __name__ == "__main__":
    setup_kaggle_environment()
```

### 3. Cloud Deployment (AWS)

#### AWS EC2 Deployment

```bash
#!/bin/bash
# deploy_aws.sh

# Launch EC2 instance (g4dn.xlarge recommended)
aws ec2 run-instances \
    --image-id ami-0c02fb55956c7d316 \
    --instance-type g4dn.xlarge \
    --key-name your-key-pair \
    --security-group-ids sg-xxxxxxxxx \
    --subnet-id subnet-xxxxxxxxx \
    --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=FireSimulation}]'

# Wait for instance to be ready
aws ec2 wait instance-running --instance-ids i-xxxxxxxxx

# Get instance public IP
INSTANCE_IP=$(aws ec2 describe-instances \
    --instance-ids i-xxxxxxxxx \
    --query 'Reservations[0].Instances[0].PublicIpAddress' \
    --output text)

echo "Instance ready at: $INSTANCE_IP"

# Connect and setup
ssh -i your-key.pem ubuntu@$INSTANCE_IP << 'EOF'
# Update system
sudo apt update && sudo apt upgrade -y

# Install NVIDIA drivers
sudo apt install ubuntu-drivers-common -y
sudo ubuntu-drivers autoinstall

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh

# Install NVIDIA Docker runtime
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt update && sudo apt install nvidia-docker2 -y
sudo systemctl restart docker

# Clone and setup application
git clone https://github.com/Prajwal-Mohapatra/forest_fire_spread.git
cd forest_fire_spread
EOF
```

#### Docker Configuration

```dockerfile
# Dockerfile
FROM tensorflow/tensorflow:2.8.0-gpu-jupyter

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gdal-bin \
    libgdal-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set Python path
ENV PYTHONPATH=/app

# Expose ports
EXPOSE 8888 5000 3000

# Start command
CMD ["bash", "scripts/start_services.sh"]
```

#### Docker Compose Configuration

```yaml
# docker-compose.yml
version: "3.8"

services:
  fire-simulation:
    build: .
    ports:
      - "8888:8888" # Jupyter
      - "5000:5000" # Backend API
      - "3000:3000" # Frontend
    volumes:
      - ./data:/app/data
      - ./outputs:/app/outputs
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
    runtime: nvidia
    command: bash scripts/start_services.sh

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - fire-simulation
```

## Configuration Management

### Environment Variables

```bash
# .env file
# Data paths
FIRE_DATA_ROOT=/path/to/data
FIRE_OUTPUT_ROOT=/path/to/outputs
FIRE_MODEL_PATH=/path/to/final_model.h5

# Service configuration
API_PORT=5000
FRONTEND_PORT=3000
WEBSOCKET_PORT=5001

# GPU configuration
CUDA_VISIBLE_DEVICES=0
TF_GPU_MEMORY_GROWTH=true
TF_MIXED_PRECISION=true

# Logging
LOG_LEVEL=INFO
LOG_FILE=/var/log/fire_simulation.log

# Security (production only)
SECRET_KEY=your-secret-key
SSL_CERT_PATH=/etc/ssl/certs/fire-sim.crt
SSL_KEY_PATH=/etc/ssl/private/fire-sim.key
```

### Configuration Files

```python
# config/production.py
import os

class ProductionConfig:
    # Data paths
    DATA_ROOT = os.environ.get('FIRE_DATA_ROOT', '/data/fire_simulation')
    OUTPUT_ROOT = os.environ.get('FIRE_OUTPUT_ROOT', '/outputs/fire_simulation')
    MODEL_PATH = os.environ.get('FIRE_MODEL_PATH', 'models/final_model.h5')

    # Service configuration
    API_HOST = '0.0.0.0'
    API_PORT = int(os.environ.get('API_PORT', 5000))

    # GPU configuration
    GPU_MEMORY_GROWTH = os.environ.get('TF_GPU_MEMORY_GROWTH', 'true').lower() == 'true'
    MIXED_PRECISION = os.environ.get('TF_MIXED_PRECISION', 'true').lower() == 'true'

    # Security
    SECRET_KEY = os.environ.get('SECRET_KEY', 'dev-secret-key')
    SSL_CONTEXT = None

    # Logging
    LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')
    LOG_FILE = os.environ.get('LOG_FILE', 'fire_simulation.log')
```

## Testing and Validation

### Deployment Verification Script

```python
# scripts/verify_deployment.py
import os
import sys
import subprocess
import requests
import tensorflow as tf

def verify_python_environment():
    """Verify Python packages and versions"""
    print("🔍 Verifying Python environment...")

    required_packages = [
        ('tensorflow', '2.8.0'),
        ('rasterio', '1.3.0'),
        ('numpy', '1.21.0'),
        ('matplotlib', '3.5.0')
    ]

    for package, min_version in required_packages:
        try:
            module = __import__(package)
            version = getattr(module, '__version__', 'unknown')
            print(f"  ✅ {package}: {version}")
        except ImportError:
            print(f"  ❌ {package}: Not installed")
            return False

    return True

def verify_gpu_setup():
    """Verify GPU configuration"""
    print("🔍 Verifying GPU setup...")

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"  ✅ GPU detected: {len(gpus)} device(s)")
        for i, gpu in enumerate(gpus):
            print(f"    GPU {i}: {gpu}")
        return True
    else:
        print("  ⚠️ No GPU detected - CPU mode only")
        return True  # Not a failure for CPU deployment

def verify_data_availability():
    """Verify required data files"""
    print("🔍 Verifying data availability...")

    required_paths = [
        'forest_fire_ml/outputs/final_model.h5',
        'data/stacked_datasets/',
        'cellular_automata/ca_engine/',
        'cellular_automata/integration/'
    ]

    for path in required_paths:
        if os.path.exists(path):
            print(f"  ✅ {path}")
        else:
            print(f"  ❌ {path}: Missing")
            return False

    return True

def verify_services():
    """Verify running services"""
    print("🔍 Verifying services...")

    # Test backend API
    try:
        response = requests.get('http://localhost:5000/api/health', timeout=5)
        if response.status_code == 200:
            print("  ✅ Backend API: Running")
        else:
            print(f"  ❌ Backend API: Status {response.status_code}")
            return False
    except requests.exceptions.RequestException:
        print("  ❌ Backend API: Not responding")
        return False

    # Test frontend
    try:
        response = requests.get('http://localhost:3000', timeout=5)
        if response.status_code == 200:
            print("  ✅ Frontend: Running")
        else:
            print(f"  ❌ Frontend: Status {response.status_code}")
            return False
    except requests.exceptions.RequestException:
        print("  ❌ Frontend: Not responding")
        return False

    return True

def main():
    """Run complete deployment verification"""
    print("🚀 Forest Fire Simulation - Deployment Verification")
    print("=" * 60)

    checks = [
        verify_python_environment,
        verify_gpu_setup,
        verify_data_availability,
        verify_services
    ]

    results = []
    for check in checks:
        try:
            result = check()
            results.append(result)
        except Exception as e:
            print(f"  ❌ Error during {check.__name__}: {e}")
            results.append(False)

    print("\n" + "=" * 60)
    if all(results):
        print("✅ All verification checks passed!")
        print("🎉 Deployment is ready for use")
        return 0
    else:
        print("❌ Some verification checks failed")
        print("Please review the errors above")
        return 1

if __name__ == "__main__":
    exit(main())
```

## Monitoring and Maintenance

### Health Check Endpoints

```python
# health_check.py
from flask import Flask, jsonify
import tensorflow as tf
import os
import psutil

app = Flask(__name__)

@app.route('/api/health')
def health_check():
    """System health check endpoint"""

    health_status = {
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'components': {}
    }

    # Check GPU availability
    gpus = tf.config.list_physical_devices('GPU')
    health_status['components']['gpu'] = {
        'available': len(gpus) > 0,
        'count': len(gpus)
    }

    # Check memory usage
    memory = psutil.virtual_memory()
    health_status['components']['memory'] = {
        'used_percent': memory.percent,
        'available_gb': memory.available / (1024**3)
    }

    # Check disk space
    disk = psutil.disk_usage('/')
    health_status['components']['disk'] = {
        'used_percent': (disk.used / disk.total) * 100,
        'free_gb': disk.free / (1024**3)
    }

    # Check model availability
    model_path = 'forest_fire_ml/outputs/final_model.h5'
    health_status['components']['model'] = {
        'available': os.path.exists(model_path),
        'path': model_path
    }

    return jsonify(health_status)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
```

### Log Monitoring

```bash
# monitoring/setup_logging.sh
#!/bin/bash

# Create log directories
sudo mkdir -p /var/log/fire_simulation
sudo chown $USER:$USER /var/log/fire_simulation

# Setup log rotation
sudo cat > /etc/logrotate.d/fire_simulation << EOF
/var/log/fire_simulation/*.log {
    daily
    missingok
    rotate 7
    compress
    delaycompress
    notifempty
    create 644 $USER $USER
}
EOF

# Setup monitoring script
cat > monitoring/check_services.sh << 'EOF'
#!/bin/bash
# Service monitoring script

check_service() {
    local service_name=$1
    local service_url=$2

    if curl -f -s "$service_url" > /dev/null; then
        echo "✅ $service_name is running"
        return 0
    else
        echo "❌ $service_name is down"
        # Add notification logic here
        return 1
    fi
}

echo "🔍 Checking Fire Simulation services..."
check_service "Backend API" "http://localhost:5000/api/health"
check_service "Frontend" "http://localhost:3000"
check_service "Health Monitor" "http://localhost:5001/api/health"

echo "Monitoring complete at $(date)"
EOF

chmod +x monitoring/check_services.sh

# Add to crontab for regular monitoring
(crontab -l 2>/dev/null; echo "*/5 * * * * /path/to/monitoring/check_services.sh >> /var/log/fire_simulation/monitoring.log 2>&1") | crontab -
```

## Troubleshooting Guide

### Common Issues and Solutions

#### 1. GPU Not Detected

```bash
# Check NVIDIA driver
nvidia-smi

# Check CUDA installation
nvcc --version

# Check TensorFlow GPU support
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# Solution: Reinstall GPU drivers
sudo apt purge nvidia-*
sudo ubuntu-drivers autoinstall
sudo reboot
```

#### 2. Memory Issues

```python
# Configure TensorFlow memory growth
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# Use mixed precision
tf.keras.mixed_precision.set_global_policy('mixed_float16')
```

#### 3. Data Loading Errors

```python
# Verify data integrity
from scripts.validate_datasets import validate_stacked_dataset

result = validate_stacked_dataset('data/stacked_datasets/stack_2016_05_15.tif')
if not result['overall_valid']:
    print("Data validation failed:", result)
```

#### 4. Web Interface Connection Issues

```bash
# Check if services are running
ps aux | grep node
ps aux | grep python

# Check port availability
netstat -tulpn | grep :5000
netstat -tulpn | grep :3000

# Restart services
pkill -f "node server.js"
pkill -f "npm start"
./scripts/start_services.sh
```

---

**Deployment Summary**: This guide provides comprehensive instructions for deploying the Forest Fire Spread Simulation system across various environments, from local development to production cloud deployment. The modular approach ensures flexibility while maintaining consistency across deployments.
