# Modified Mermaid Diagram

This file contains a modified version of the main process flow diagram with outer containers flowing top to bottom and sub-containers arranged left to right.

## Main Process Flow Diagram (Modified Layout)

```mermaid
---
config:
  layout: dagre
---
flowchart TB
  subgraph DC["Data Collection"]
    direction LR
    SRTM["SRTM DEM
    (30m)"] --> ERA5["ERA5 Weather
    (0.25°→30m)"] --> LULC["LULC 2020
    (10m→30m)"] --> GHSL["GHSL 2015
    (30m)"] --> VIIRS["VIIRS Fire
    (375m→30m)"]
  end

  subgraph DP["Data Pre-processing"]
    direction LR
    Stack["9-band Stacking"] --> Resample["Spatial Resampling"] --> QC["Quality Control"] --> DEM["DEM Derivatives
    (Slope, Aspect)"]
  end

  subgraph ML["ML Training & Prediction"]
    direction LR
    ResUNET["ResUNet-A
    Architecture"] --> SlidingWindow["Sliding Window
    Inference"] --> FireProb["Fire Probability
    Maps"]
  end

  subgraph CA["Cellular Automata Simulation"]
    direction LR
    GPU["TensorFlow GPU
    Acceleration"] --> Rules["Physics-based
    Spread Rules"] --> TimeSteps["Hourly
    Simulation Steps"]
  end

  subgraph VIS["Visualization"]
    direction LR
    React["React Frontend"] --> Leaflet["Interactive
    Leaflet Maps"] --> API["Flask API
    Backend"]
  end

  DC --> DP
  DP --> ML
  ML --> CA
  CA --> VIS

  DP -. "Stacked GeoTIFF
  (9 bands)" .-> ML
  ML -. "Probability Map
  (0-1 range)" .-> CA
  CA -. "Time Series
  Fire Spread" .-> VIS

  User["User Input"] --> VIS
  VIS --> Results["Simulation
  Results"]
  Results --> User
```

The above diagram shows the Forest Fire Spread Simulation System with:

1. Main process components arranged vertically (top to bottom)
2. Sub-components within each section arranged horizontally (left to right)
3. Data flow connections between major components
4. Special data transformations shown with dotted lines
5. User feedback loop included
