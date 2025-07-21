# Mermaid Diagrams for Forest Fire Spread Simulation System

This file contains Mermaid diagram code for visualizing the architecture of the Forest Fire Spread Simulation System.

## Main Process Flow Diagram

```mermaid
---
config:
  layout: dagre
---
flowchart TB
  subgraph DC["Data Collection"]
    direction LR
    SRTM["SRTM DEM\n(30m)"] --> ERA5["ERA5 Weather\n(0.25°→30m)"] --> LULC["LULC 2020\n(10m→30m)"] --> GHSL["GHSL 2015\n(30m)"] --> VIIRS["VIIRS Fire\n(375m→30m)"]
  end

  subgraph DP["Data Pre-processing"]
    direction LR
    Stack["9-band Stacking"] --> Resample["Spatial Resampling"] --> QC["Quality Control"] --> DEM["DEM Derivatives\n(Slope, Aspect)"]
  end

  subgraph ML["ML Training & Prediction"]
    direction LR
    ResUNET["ResUNet-A\nArchitecture\nIoU: 0.82, Dice: 0.82"] --> SlidingWindow["Sliding Window\nInference"] --> FireProb["Fire Probability\nMaps"]
  end

  subgraph CA["Cellular Automata Simulation"]
    direction LR
    GPU["TensorFlow GPU\nAcceleration"] --> Rules["Physics-based\nSpread Rules"] --> TimeSteps["Hourly\nSimulation Steps"]
  end

  subgraph VIS["Visualization"]
    direction LR
    React["React Frontend"] --> Leaflet["Interactive\nLeaflet Maps"] --> API["Flask API\nBackend"]
  end

  DC --> DP
  DP --> ML
  ML --> CA
  CA --> VIS

  DP -. "Stacked GeoTIFF\n(9 bands)" .-> ML
  ML -. "Probability Map\n(0-1 range)" .-> CA
  CA -. "Time Series\nFire Spread" .-> VIS

  User["User Input"] --> VIS
  VIS --> Results["Simulation\nResults"]
  Results --> User
```

## Component Architecture Diagram

```mermaid
flowchart TD
    %% Main Components with Tools
    subgraph DataSources["Data Sources & Processing"]
        direction TB
        GEE["Google Earth Engine\nData Collection"]
        Rasterio["Rasterio\nGeospatial Processing"]
        NumPy["NumPy\nArray Operations"]
        GDAL["GDAL\nFormat Conversion"]
    end

    subgraph MLModel["ML Prediction Model"]
        direction TB
        TF["TensorFlow 2.x"]
        ResUNet["ResUNet-A\nArchitecture"]
        FocalLoss["Focal Loss\nClass Imbalance"]
        Patches["Patch-based\nProcessing (256×256)"]
    end

    subgraph CAEngine["Cellular Automata Engine"]
        direction TB
        TFCA["TensorFlow-based CA"]
        GPU["GPU Acceleration"]
        Moore["8-connected Moore\nNeighborhood"]
        Physics["Physics Rules\n(Wind, Slope, Fuel)"]
        Config["Dataclass-based\nConfiguration"]
    end

    subgraph Bridge["Integration Bridge"]
        direction TB
        Pipeline["Data Pipeline\nOrchestration"]
        Validation["Quality Assurance"]
        Error["Error Handling"]
        Scenarios["Multiple Scenario\nManagement"]
    end

    subgraph WebUI["Web Interface"]
        direction TB
        React["React + Material UI"]
        Leaflet["Leaflet Maps"]
        Flask["Flask API"]
        WebSocket["WebSocket\nReal-time Updates"]
        Charts["Chart.js\nStatistics"]
    end

    %% Connections between components
    DataSources --> MLModel
    MLModel --> Bridge
    Bridge --> CAEngine
    CAEngine --> WebUI
    Bridge --> WebUI
```

## Data Flow Diagram

```mermaid
sequenceDiagram
    participant DS as Data Sources
    participant ML as ML Prediction
    participant CA as CA Simulation
    participant UI as Web Interface
    participant User as User

    Note over DS,User: Primary Data Flow

    DS->>ML: 9-band environmental stack
    ML->>CA: Fire probability map
    CA->>UI: Hourly fire spread frames
    UI->>User: Interactive visualization

    Note over User,CA: Feedback Loop

    User->>UI: Set ignition points & parameters
    UI->>CA: Simulation request
    CA->>UI: Real-time simulation updates
    UI->>User: Updated visualization & statistics
```

## System Architecture Overview

```mermaid
---
config:
  layout: dagre
---
flowchart TB
    %% Main components with performance metrics
    subgraph InputLayer["Input Layer"]
        direction LR
        Satellite["Satellite Data"] --> Derived["Derived Features"]
    end

    subgraph PredictionLayer["Prediction Layer"]
        ML["ResUNet-A Model\n94.2% accuracy\n2-3 min inference"]
    end

    subgraph SimulationLayer["Simulation Layer"]
        CA["GPU-Accelerated CA\n30-sec for 6hr simulation"]
    end

    subgraph InterfaceLayer["Interface Layer"]
        Web["React + Leaflet\n<1 sec response"]
    end

    %% Highlight key infrastructure requirements
    subgraph Infrastructure["Infrastructure Requirements"]
        direction LR
        GPU["NVIDIA GPU\n8GB+ VRAM"] --> RAM["16GB+ RAM"] --> Storage["50GB+ Storage"] --> Python["Python 3.8+\nTensorFlow 2.x"]
    end

    %% Connections
    InputLayer --> PredictionLayer
    PredictionLayer --> SimulationLayer
    SimulationLayer --> InterfaceLayer
    Infrastructure -.-> PredictionLayer
    Infrastructure -.-> SimulationLayer
```

These diagrams provide a visual representation of the Forest Fire Spread Simulation System architecture, focusing on the main process flow, component interactions, data flow, and system requirements.

The diagrams are designed to be clear and concise while highlighting the key technologies, performance characteristics, and data transformations throughout the system.
