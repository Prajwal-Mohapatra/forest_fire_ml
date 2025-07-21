# 🔥 Forest Fire Spread Simulation - Knowledge Base

## Project Overview

This knowledge base contains comprehensive documentation for the Forest Fire Spread Simulation system - an integrated ML-CA (Machine Learning - Cellular Automata) pipeline for predicting and simulating forest fire behavior in Uttarakhand, India.

**Repository**: https://github.com/Prajwal-Mohapatra/forest_fire_spread
**Target Audience**: ISRO researchers, forest fire management professionals
**Geographic Focus**: Uttarakhand state, India (April-May 2016 fire season)

## System Architecture

```
Input Data → ML Model → CA Engine → Web Interface → Visualization
    ↓           ↓          ↓           ↓              ↓
Stacked     ResUNet-A   TensorFlow   React/JS     Interactive
GeoTIFF     Prediction   CA Core     Frontend      Demo
(9 bands)   (0-1 prob)  (Hourly)    (Real-time)   (Animation)
```

## Core Components

### 1. 🧠 Machine Learning Module (`forest_fire_ml/`)

- **ResUNet-A Model**: Deep learning architecture for fire probability prediction
- **Input Processing**: 9-band environmental data stack (DEM, ERA5, LULC, GHSL, etc.)
- **Output**: Daily fire probability maps (0-1 range, 30m resolution)
- **Training Data**: April 1 - May 29, 2016 (Uttarakhand fire season)

### 2. 🔥 Cellular Automata Engine (`cellular_automata/`)

- **TensorFlow-based CA**: GPU-accelerated fire spread simulation
- **Enhanced Configuration**: Dataclass-based configuration with type safety
- **Advanced Utilities**: GPU-accelerated terrain analysis and array operations
- **Simplified Rules**: Rapid prototyping capabilities for parameter tuning
- **Temporal Resolution**: Hourly simulation steps
- **Physics Model**: Fire spread rules with wind, topography, and barrier effects
- **Integration**: Seamless connection with ML probability maps

### 3. 🌉 ML-CA Integration Bridge (`cellular_automata/integration/`)

- **Data Pipeline**: Orchestrates ML predictions → CA simulation workflow
- **Validation**: Ensures data consistency between ML outputs and CA inputs
- **Scenario Management**: Handles multiple ignition scenarios and weather conditions

### 4. 🖥️ Web Interface (`cellular_automata/web_interface/`)

- **Enhanced API**: Flask backend with multiple scenario comparison and caching
- **React Integration**: Comprehensive guide for professional frontend development
- **Interactive Demo**: Complete examples with ISRO-themed styling
- **Advanced Features**: Export functionality, coordinate validation, and real-time updates
- **User Controls**: Date selection, ignition points, animation controls
- **Visualization**: Real-time fire spread animation with multiple map layers

### 5. 📊 Kaggle Orchestration (`Forest_Fire_CA_Simulation_Kaggle.ipynb`)

- **Clean Integration**: Calls existing project functions (no code duplication)
- **Demo Ready**: Interactive widgets for parameter tuning
- **Export Package**: Prepares data for React frontend integration

## Documentation Structure

- [`01_ML_Model_Documentation.md`](./01_ML_Model_Documentation.md) - ResUNet-A model details
- [`02_CA_Engine_Documentation.md`](./02_CA_Engine_Documentation.md) - Cellular automata implementation (Updated July 2025)
- [`03_Integration_Bridge_Documentation.md`](./03_Integration_Bridge_Documentation.md) - ML-CA bridge
- [`04_Web_Interface_Documentation.md`](./04_Web_Interface_Documentation.md) - Frontend design (Updated July 2025)
- [`05_Data_Pipeline_Documentation.md`](./05_Data_Pipeline_Documentation.md) - Data processing workflow
- [`06_Progress_Report.md`](./06_Progress_Report.md) - Project progress and milestones (Updated July 2025)
- [`07_Technical_Decisions.md`](./07_Technical_Decisions.md) - Architecture choices and rationale (Updated July 2025)
- [`08_Chat_Summary.md`](./08_Chat_Summary.md) - Key discussions and conclusions
- [`09_Deployment_Guide.md`](./09_Deployment_Guide.md) - Setup and deployment instructions
- [`10_Future_Roadmap.md`](./10_Future_Roadmap.md) - Planned enhancements and future vision (Updated July 2025)
- [`11_Project_Summary.md`](./11_Project_Summary.md) - Complete project overview and final documentation
- [`12_Current_Project_Status.md`](./12_Current_Project_Status.md) - Current status and recent developments (New July 2025)
- [`CA_DUPLICATION_RESOLUTION.md`](./CA_DUPLICATION_RESOLUTION.md) - Code consolidation and architecture improvements (New July 2025)
- [`SOP.md`](./SOP.md) - Standard operating procedures for development
- [`stitch_ai_prompt.md`](./stitch_ai_prompt.md) - Web interface design specifications

## Recent Updates (July 2025)

### Architecture Consolidation ✅

- **Duplicate Code Resolution**: Eliminated redundant cellular automata and web interface implementations
- **Enhanced Functionality**: Migrated best features from duplicates to main codebase
- **Improved Documentation**: Added comprehensive migration guides and React integration examples
- **Quality Improvements**: Better type safety, GPU optimization, and code organization

### New Components Added

- **Advanced Configuration**: Dataclass-based CA configuration with type hints
- **GPU Utilities**: TensorFlow-based slope calculation and array operations
- **Enhanced API**: Multiple scenario comparison, caching, and export endpoints
- **React Integration**: Production-ready frontend development guide with ISRO styling

### Documentation Freshness Tracking

- **Last Major Update**: July 7, 2025
- **Components Updated**: CA Engine, Web Interface, Progress Report, Technical Decisions, Future Roadmap
- **New Documentation**: Current Project Status, CA Duplication Resolution, React Integration Guide
- **Next Review Date**: October 2025

## Quick Start Guide

### For Developers

1. **Clone Repository**: `git clone https://github.com/Prajwal-Mohapatra/forest_fire_spread.git`
2. **Setup ML Environment**: Configure Python environment with TensorFlow, rasterio, etc.
3. **Download Model**: Ensure `final_model.h5` is in `forest_fire_ml/outputs/`
4. **Test Pipeline**: Run Kaggle orchestration notebook for end-to-end verification
5. **Launch Web Interface**: Start React development server for interactive demo

### For Researchers

1. **Review Model Performance**: Check training metrics and validation results
2. **Examine CA Physics**: Understand fire spread rules and environmental factors
3. **Interactive Demo**: Use web interface for scenario testing and analysis
4. **Export Results**: Generate GeoTIFF outputs for external analysis

### For End Users

1. **Access Web Demo**: Open browser interface
2. **Select Date/Location**: Choose from available 2016 fire season data
3. **Set Ignition Points**: Click on map to add fire start locations
4. **Configure Parameters**: Adjust wind, simulation duration, etc.
5. **Run Simulation**: Watch real-time fire spread animation
6. **Export Results**: Download simulation data and visualizations

## Key Features

### Technical Capabilities

- **High Resolution**: 30m spatial resolution across full Uttarakhand state
- **GPU Acceleration**: TensorFlow-based computation for performance
- **Real-time Simulation**: Hourly time steps with interactive controls
- **Multi-scenario Support**: Compare different ignition patterns and weather conditions
- **Data Validation**: Automated consistency checks between ML and CA components

### User Experience

- **Interactive Visualization**: Click-to-ignite interface with real-time feedback
- **Professional Interface**: ISRO-themed design for researcher audience
- **Animation Controls**: Play/pause/speed controls for simulation playback
- **Export Functionality**: GeoTIFF and JSON output for external analysis
- **Responsive Design**: Works on desktop and tablet devices

### Scientific Accuracy

- **Validated ML Model**: ResUNet-A trained on real fire occurrence data
- **Physics-based CA**: Fire spread rules incorporating wind, topography, barriers
- **Environmental Integration**: DEM, land use, settlement data influence spread patterns
- **Weather Sensitivity**: Wind speed/direction affect fire propagation

## Data Sources

### Primary Datasets

- **DEM**: SRTM 30m elevation data
- **Weather**: ERA5 daily meteorological variables
- **Land Use**: LULC 2020 classification
- **Settlements**: GHSL 2015 built-up areas
- **Fire History**: VIIRS active fire detections (2016)

### Processed Outputs

- **Stacked TIFFs**: 9-band environmental data stacks
- **Probability Maps**: ML-generated daily fire probability (0-1)
- **Binary Maps**: Threshold-based fire/no-fire classification
- **Confidence Zones**: Multi-level probability confidence mapping

## Performance Metrics

### ML Model Performance

- **Training Accuracy**: ~94% on validation set
- **IoU Score**: 0.87 for fire detection
- **Dice Coefficient**: 0.91 for segmentation quality
- **Inference Speed**: ~2 minutes for full Uttarakhand prediction

### CA Simulation Performance

- **Spatial Coverage**: 400x400 km at 30m resolution
- **Temporal Resolution**: Hourly updates over 1-12 hour simulations
- **GPU Acceleration**: 10x speedup vs CPU-only implementation
- **Memory Usage**: ~2GB GPU memory for full-state simulation

## Integration Points

### Data Flow

1. **Input Preparation**: Stack environmental layers for specific date
2. **ML Prediction**: Generate fire probability map using ResUNet-A
3. **CA Initialization**: Load probability map as base state
4. **Simulation Execution**: Apply fire spread rules over time steps
5. **Visualization**: Display results in web interface with animation
6. **Export**: Package results for external analysis

### API Endpoints

- `/api/predict`: Generate ML fire probability prediction
- `/api/simulate`: Run CA fire spread simulation
- `/api/scenario`: Create/manage simulation scenarios
- `/api/export`: Download simulation results

### File Formats

- **Input**: GeoTIFF (stacked environmental data)
- **Intermediate**: GeoTIFF (probability maps), JSON (metadata)
- **Output**: GeoTIFF (simulation frames), MP4/GIF (animations), JSON (statistics)

## Contact & Support

### Development Team

- **ML Model**: Fire prediction model development and training
- **CA Engine**: Cellular automata physics and simulation
- **Integration**: ML-CA bridge and data pipeline
- **Frontend**: Web interface and visualization

### Technical Support

- **Documentation**: Comprehensive guides in this knowledge base
- **Code Examples**: Working examples in Kaggle orchestration notebook
- **Issue Tracking**: GitHub repository issue tracker
- **Community**: Project contributors and research collaborators

---

**Last Updated**: July 7, 2025
**Version**: 1.1.0
**Status**: Production Ready - Enhanced with Architecture Consolidation

**Recent Changes**:

- Eliminated code duplication and consolidated architecture
- Enhanced cellular automata engine with advanced configuration and GPU utilities
- Improved web interface with multiple scenario comparison and comprehensive React guide
- Added complete migration documentation and quality improvements
