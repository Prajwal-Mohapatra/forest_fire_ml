# Forest Fire Spread Prediction System

_AI-Powered Fire Probability Prediction and Spread Simulation_

## ISRO BAH Hackathon 2025 Submission

**Team:** The Minions  
**Problem Statement:** Simulation/Modelling of Forest Fire Spread using AI/ML techniques  
**Technology Stack:** Geospatial Analysis + Deep Learning + Cellular Automata

### Project Resources
- **Primary Repository:** [Forest Fire Spread System](https://github.com/Prajwal-Mohapatra/forest_fire_spread)
- **Model Development:** [ML Implementation](https://github.com/Prajwal-Mohapatra/forest_fire_ml)
- **Raw Dataset:** [Unstacked Satellite Data](https://kaggle.com/datasets/9f4a7d0bdf49fd695f3028108915bd3594658cb8e8b89e5c082bf018448ca1e8) (Kaggle)
- **Processed Dataset:** [Stacked Training Data](https://kaggle.com/datasets/b8f7d15fad5b34501767b80eac0202ff9d1529e924e3fafcf3d1acf38b2b8590) (Kaggle)
- **Design & Architecture:** [System Wireframes](https://www.figma.com/design/YeS8pwYnDU9ZhLxeAP6ZHH/ISRO-BAH-Hackathon-2025?node-id=0-1&t=EEjAIq96FQ77oQAR-1) (Figma)

---

## What is this project?

### Core System Components

Our system, **dual-phase forest fire prediction and simulation platform**, combines:

1. **ResUNet-A Deep Learning Model** - Predicts fire probability from multi-spectral satellite imagery, stacked into a 10-band raster image
2. **Cellular Automata Engine** - Simulates *realistic* fire spread dynamics *with physics-based rules* (realistic part, to be added in the future)
3. **Interactive Web Interface** - Allows real-time scenario testing, visualization and sending & receiving of alerts
4. **Comprehensive Data Pipeline** - Processes Landsat 8, MODIS, SRTM, Sentinel-2 and auxiliary geospatial datasets

### Technical Architecture

```
Satellite Data      →     ML Prediction      →   Fire Spread Simulation   →   Interactive Visualization
(10-band imagery)   →   (Probability maps)   →     (Cellular Automata)    →         (Web Interface)
```

---

## Why we built this project?

### 1. **Real-World Impact Motivation**

- Forest fires have increased by 75% globally in the last decade
- Our target region (Uttarakhand) faces 2000+ fire incidents annually=
- Protecting critical ecosystems and wildlife habitats

### 2. **Technical Challenge Appeal**

- **Multi-disciplinary Problem:** Combines computer vision, geospatial analysis, and physics simulation
- **ISRO BAH Hackathon Opportunity:** Platform to contribute to national disaster management
- **Cutting-edge Technology Application:** Implementing state-of-the-art ResUNet-A architecture
- **Scalable Solution Development:** Building system for real-world deployment

### 3. **Gap in Existing Solutions**

- Current systems **lack spatial precision** (30m resolution), and are **reactive**, instead of being proactive
- **Missing Dynamic Simulation**, static risk maps don't show fire spread patterns
- **Poor Integration:** Disconnected tools for prediction vs. simulation
- Complex systems not user-friendly for field personnel

---

## How we built this project?

### Phase 1: Data Engineering & Preprocessing

```python
# Multi-source data integration
- Landsat 8 OLI/TIRS (optical + thermal bands)
- GHSL Urban Infrastructure Data
- SRTM Digital Elevation Models
- VIIRS-SNPP Active Fire Products (ground truth)

Custom temporal alignment and spatial registration
```

**Key Innovation:** Fire-focused patch sampling strategy (80% fire-prone areas) dramatically improved model training efficiency.

### Phase 2: Deep Learning Architecture

```python
class ResUNetA:
    # Enhanced U-Net with Res (residual connections) + A (Atrous Convolution)
    - Encoder: 4-stage residual blocks with max pooling
    - Bridge: Atrous Spatial Pyramid Pooling (ASPP)
    - Decoder: Bilinear upsampling with skip connections

    - Output: Sigmoid activation for probability mapping
```

**Key Innovation:** Focal loss implementation (γ=2.0, α=0.25) to handle severe class imbalance (fire pixels <1% of total).

### Phase 3: Cellular Automata Simulation

```python
class FireSpreadEngine:
    # Physics-based spread rules
    - Moore neighborhood (8-cell connectivity)
    - Wind-driven directional bias, Terrain slope effects
    - Infrastructure barriers (roads, water, urban)
    - Hourly time step simulation
```

**Key Innovation:** Integration of ML probability maps as base probability conditions for CA engine, creating seamless prediction-to-simulation pipeline.

### Phase 4: Web Integration & Visualization

```python
# Flask REST API + React Frontend
- Real-time simulation execution
- Interactive ignition point selection
- Multiple scenario comparison
- Animation generation and playback
```

---

## What We Learned

### 1. **Technical Challenges**

- **Geospatial Data Complexity:** Managing multi-temporal, multi-sensor datasets requires sophisticated preprocessing pipelines *(plans automate the pipeline, from data collection to stacking)*
- Focal loss + fire-focused sampling reduced false negatives by 60%
- Sliding window prediction with 64-pixel overlap enables processing of large geographical areas efficiently
- Cellular automata simulation must balance accuracy with computational speed for web deployment

### 2. **Domain Exploration**

- *Wind speed/direction critically affects spread patterns (2x faster in wind direction) (to be implemented in the future)*
- **Satellite Remote Sensing:** NIR and SWIR bands most informative for fire detection
- Fire activity shows strong diurnal patterns requiring time-aware modeling
- **Geospatial Analysis:** 30m resolution optimal balance between detail and computational feasibility

---

## Technical Overview

### 1. **Technical Innovation**

- **Dual-AI Architecture:** First system combining **ResUNet-A + Probabilistic Cellular Automata** for forest fires
- **High Spatial Resolution:** 30m pixel accuracy vs. 1km standard in existing systems
- **Multi-scenario Analysis:** Compare different ignition points and weather conditions simultaneously

### 2. **Practical Deployment Ready**

- **Complete Pipeline:** From raw satellite data to interactive web visualization
- **Production Architecture:** Flask API + React frontend ready for cloud deployment

### 3. **Scientific Metrics**

- **Validation Metrics:** IoU=0.821, Dice=0.857 on validation set demonstrate strong performance
- **Temporal Splitting:** Training on April data, testing on May prevents data leakage
- **Physics-Based Simulation:** CA rules incorporate actual fire behavior principles
- **Uncertainty Quantification:** Confidence zones (high/medium/low/no fire) for decision support

### 4. **UI/UX Focus**

- **Interactive Interface:** Point-and-click ignition selection on map
- **Real-time Feedback:** Immediate simulation results with progress indicators
- **Visualization Quality:** Custom fire-themed colormaps and smooth animations
- **Multi-format Output:** GeoTIFF, JSON, and PNG outputs for different use cases

---

## Impact & Results

### **Technical Milestones**

- **50 Training Epochs** with early stopping and learning rate scheduling
- **9-Band Multi-spectral** satellite imagery processing
- **6-Hour Fire Spread** simulation with realistic physics
- **5 REST API Endpoints** for complete system integration
- **3-Tier Confidence** mapping (high/medium/low fire risk)

---

## Problems Solved

Our system addresses the critical gap between **static fire risk assessment** and **dynamic spread prediction**. Traditional systems show where fires might start, but not how they will spread. We provide:

1. **Precise Fire Probability Maps** (30m resolution)
2. **Dynamic Spread Simulation** (hour-by-hour progression)
3. **Interactive Scenario Testing** (what-if analysis)
4. **Decision Support Tools** (confidence zones, multiple outputs)
5. **Real-time Deployment Capability** (web-based interface)

This enables forest departments, disaster management agencies, and researchers to make data-driven decisions for:

- **Resource Allocation** (where to position firefighting teams)
- **Evacuation Planning** (which areas to evacuate first)
- **Prevention Strategies** (where to create firebreaks)
- **Risk Communication** (clear visualizations for public awareness)

---

## References

1. Diakogiannis, F. I., Waldner, F., Caccetta, P., & Wu, C. (2020). ResUNet-a: A deep learning framework for semantic segmentation of remotely sensed data. *ISPRS Journal of Photogrammetry and Remote Sensing*, 162, 94-114. https://doi.org/10.1016/j.isprsjprs.2020.01.013

2. Huot, F., Hu, R. L., Goyal, N., Sankar, T., Ihme, M., & Chen, Y. F. (2022). Next day wildfire spread: A machine learning dataset to predict wildfire spreading from remote-sensing data. *IEEE Transactions on Geoscience and Remote Sensing*, 60, 1-13. https://doi.org/10.1109/TGRS.2022.3192974

3. Karafyllidis, I., & Thanailakis, A. (1997). A model for predicting forest fire spreading using cellular automata. *Ecological Modelling*, 99(1), 87-97. https://doi.org/10.1016/S0304-3800(96)01942-4

4. United Nations, Department of Economic and Social Affairs - Sustainable Development. (2015). Transforming our world: The 2030 Agenda for Sustainable Development (A/RES/70/1). https://sdgs.un.org/2030agenda

5. Forest Survey of India. (2023). *India State of Forest Report 2023*. Ministry of Environment, Forest and Climate Change, Government of India. https://fsi.nic.in/forest-report-2023

