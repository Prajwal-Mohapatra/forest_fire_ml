# Stitch AI Design Prompt: Forest Fire Spread Simulation Website

## Project Brief
Create a minimalistic, professional website for forest fire spread simulation targeting ISRO researchers. The website demonstrates ML-based fire prediction integrated with cellular automata simulation for Uttarakhand, India.

## Target Audience
- ISRO researchers (10-15 years experience)
- Forest fire management professionals
- Scientific demonstration and hackathon judges

## Design Requirements

### Overall Theme & Style
- **Primary Theme**: ISRO/Space technology aesthetic
- **Secondary Theme**: Forest fire/emergency management
- **Style**: Minimalistic, clean, professional, scientific
- **Atmosphere**: Serious, data-driven, cutting-edge technology

### Color Palette (ISRO/Fire Theme)
- **Primary**: Deep space blue (#0B1426, #1E3A5F)
- **Secondary**: ISRO orange (#FF6B35, #FF8C42)
- **Accent**: Fire red (#D63031, #E17055)
- **Warning**: Bright orange/yellow (#FDCB6E, #F39C12)
- **Success**: Forest green (#00B894, #6C5CE7)
- **Background**: Clean white (#FFFFFF, #F8F9FA)
- **Text**: Dark gray (#2D3436, #636E72)

### Typography
- **Headers**: Modern, sans-serif (Inter, Poppins, or similar)
- **Body**: Clean, readable sans-serif
- **Data/Technical**: Monospace for coordinates, data values
- **Style**: Professional, scientific, not playful

## Page Layout & Structure

### Main Layout (Single Page Application)
```
┌─────────────────────────────────────────────────────────────┐
│ Header: Logo + Title + Navigation                           │
├─────────────────────────────────────────────────────────────┤
│ Control Panel (Left Sidebar, 25% width)                    │
│ ┌─────────────────────────────────────┐ ┌─────────────────┐ │
│ │ Date Selection                      │ │                 │ │
│ │ Ignition Point Controls             │ │                 │ │
│ │ Simulation Parameters               │ │                 │ │
│ │ Animation Controls                  │ │   Main Map      │ │
│ │ Layer Toggles                       │ │   Display       │ │
│ │ Export Options                      │ │   (75% width)   │ │
│ └─────────────────────────────────────┘ │                 │ │
│                                         │                 │ │
│ Status Bar + Progress Indicators        │                 │ │
│                                         └─────────────────┘ │
├─────────────────────────────────────────────────────────────┤
│ Results Panel (Bottom, collapsible)                        │
│ Statistics + Timeline + Export Options                     │
└─────────────────────────────────────────────────────────────┘
```

### Header Design
- **Logo**: ISRO logo (left) + Project title (center) + University/Team (right)
- **Title**: "Forest Fire Spread Prediction System - Uttarakhand"
- **Subtitle**: "ML-Based Fire Probability & Cellular Automata Simulation"
- **Navigation**: Minimal - Home, About, Documentation (optional)

### Control Panel (Left Sidebar)
1. **Date & Time Selection**
   - Date picker: April 1 - May 29, 2016
   - Time controls for simulation duration (1/2/3/6/12 hours)
   - Current simulation status indicator

2. **Ignition Controls**
   - "Click to Ignite" mode toggle
   - Active ignition points list
   - Clear all ignitions button
   - Ignition intensity slider

3. **Simulation Parameters**
   - Wind speed/direction (constant daily)
   - Simulation speed control
   - Resolution settings
   - Physics model selector (simple/advanced)

4. **Layer Toggles**
   - Fire probability layer (ML output)
   - Current fire spread (CA simulation)
   - Terrain/DEM overlay
   - Roads/barriers (GHSL)
   - Weather data overlay

5. **Animation Controls**
   - Play/Pause/Stop buttons
   - Timeline scrubber
   - Speed control (1x, 2x, 4x)
   - Frame-by-frame controls

### Main Map Display (Center)
- **Base Map**: Satellite/terrain view of Uttarakhand
- **Interactive Features**: Zoom, pan, coordinate display
- **Overlays**: 
  - Fire probability heatmap (0-1 gradient: blue → yellow → red)
  - Current fire spread (bright red/orange)
  - Predicted spread zones (orange gradient)
  - Geographic boundaries, roads, rivers
- **UI Elements**:
  - Zoom controls (top-right)
  - Coordinate display (bottom-right)
  - Scale indicator (bottom-left)
  - Layer legend (top-left)

### Results Panel (Bottom)
- **Real-time Statistics**: Area burned, spread rate, risk level
- **Timeline**: Hour-by-hour progression graph
- **Comparison**: ML prediction vs. actual simulation
- **Export Options**: Download maps, data, animation

## Interactive Elements

### User Interactions
1. **Primary**: Click on map to set ignition points
2. **Secondary**: Adjust simulation parameters
3. **Tertiary**: Control animation playback
4. **Navigation**: Zoom/pan map, toggle layers

### Visual Feedback
- **Loading States**: Simulation processing indicators
- **Hover Effects**: Subtle highlighting for interactive elements
- **Active States**: Clear indication of selected tools/modes
- **Progress**: Real-time simulation progress bar

### Animation Features
- **Smooth Transitions**: Between time steps
- **Color Gradients**: Fire intensity visualization
- **Pulse Effects**: Active fire spreading
- **Trail Effects**: Show fire progression path

## Technical Specifications

### Responsive Design
- **Primary**: Desktop-first (1920x1080 optimal)
- **Secondary**: Tablet compatibility (1024x768)
- **Mobile**: Basic functionality only

### Performance Requirements
- **Map Rendering**: Smooth at 30m resolution
- **Animation**: 30 FPS for simulation playback
- **Data Loading**: Progressive loading with indicators
- **Memory**: Efficient handling of large raster data

### Browser Compatibility
- **Primary**: Chrome, Firefox, Edge (latest versions)
- **Secondary**: Safari support
- **Requirements**: Modern ES6+, WebGL support

## Specific UI Components

### Map Legend
```
Fire Probability Scale:
Low Risk    [████████████] High Risk
0.0   0.2   0.4   0.6   0.8   1.0
Blue → Green → Yellow → Orange → Red

Fire Spread:
● Active Fire (Bright Red)
● Predicted Spread (Orange Gradient)
● Burned Area (Dark Red)
● Natural Barriers (Gray)
```

### Control Buttons Style
- **Primary Actions**: ISRO orange background, white text
- **Secondary Actions**: White background, orange border
- **Danger Actions**: Fire red background, white text
- **Disabled State**: Gray background, muted text

### Data Display Cards
- **Style**: Clean white cards with subtle shadows
- **Headers**: ISRO blue background
- **Content**: Organized data with clear typography
- **Icons**: Scientific/technical iconography

## Content & Messaging

### Headlines
- Main: "AI-Powered Forest Fire Spread Prediction"
- Sub: "Real-time simulation for Uttarakhand Forest Management"

### Interface Text
- Use scientific, professional terminology
- Clear, concise instructions
- No casual or playful language
- Technical accuracy in labels

### Help Text
- Contextual tooltips for complex features
- Clear error messages with solutions
- Progress indicators with descriptive text

## Assets Needed

### Icons & Graphics
- ISRO logo (official)
- Fire/flame icons (various intensities)
- Weather icons (wind, temperature)
- Map control icons (zoom, pan, layers)
- Simulation controls (play, pause, stop)
- Scientific/data visualization icons

### Map Imagery
- Uttarakhand state boundary
- Satellite/terrain base layer
- DEM elevation data visualization
- Road/infrastructure overlay (GHSL)

### Loading Graphics
- Simulation processing animation
- Data loading spinners
- Progress bars with scientific aesthetics

## Example Workflows

### Primary Use Case: Fire Spread Simulation
1. User selects date (e.g., May 23, 2016)
2. ML probability map loads automatically
3. User clicks on map to set ignition point
4. User configures simulation duration (e.g., 6 hours)
5. User clicks "Start Simulation"
6. Real-time animation shows fire spread
7. User can pause, adjust speed, or restart

### Secondary Use Case: Comparison Analysis
1. User loads multiple dates for comparison
2. Sets same ignition points on different days
3. Runs parallel simulations
4. Views side-by-side results
5. Exports comparison data

## Technical Integration Notes

### Data Sources
- Input: Daily fire probability maps (.tif, 30m resolution)
- Output: Hourly simulation results (raster + vector)
- Weather: ERA5 daily constants
- Geographic: DEM, LULC, GHSL layers

### Backend Integration
- Real-time communication with CA simulation engine
- Efficient data streaming for large raster files
- WebSocket connections for live updates
- RESTful API for data management

### File Formats
- Raster: GeoTIFF for probability maps
- Vector: GeoJSON for boundaries, points
- Animation: WebM or MP4 for export
- Data: CSV for statistics export

## Accessibility & Usability

### Scientific Standards
- Professional appearance suitable for research presentations
- Clear data visualization following scientific conventions
- Accurate color schemes for data representation
- Proper attribution and methodology notes

### Usability Principles
- Intuitive interface for non-technical users
- Progressive disclosure of advanced features
- Clear visual hierarchy
- Consistent interaction patterns

This prompt provides comprehensive guidance for creating a professional, ISRO-themed forest fire simulation website that balances scientific rigor with user-friendly design for your hackathon submission.
