# Standard Operating Procedures (SOPs) for Forest Fire Prediction Project

This document contains pre-prompts and standard operating procedures to maximize efficiency when working with Claude/LLMs on the forest fire prediction system.

## General SOPs

### **SOP-001: Initial Context Setup**

```
[Your basic information and goals].
Ask me any questions so you can give me the best possible response.
```

### **SOP-002: Architecture Design**

```
Think step-by-step about the design and architecture for [component name]. Consider:
- Data flow
- Key functions or classes
- Potential challenges
- Integration with other components
```

### **SOP-003: Code Generation**

```
Generate [language] code for [specific functionality]. Include:
- Input/output types
- Error handling
- Detailed inline comments explaining the logic
- Any necessary imports or dependencies
```

### **SOP-004: Code Explanation**

```
Please explain how [specific part of the code] works and why it was implemented this way.
```

### **SOP-005: Code Review**

```
Review the following code for potential improvements, bugs, or best practice violations:
[Paste the code here]
```

### **SOP-006: Documentation Generation**

```
Based on the code and our discussions, generate the following documentation:
- README file section for this component
- API documentation (if applicable)
- Usage examples
```

### **SOP-007: Database Schema Design**

```
Generate a database schema for [describe your data model].
Include:
- Table definitions
- Relationships
- Indexes
- Any necessary constraints
```

### **SOP-008: SQL Query Optimization**

```
Generate an optimized SQL query to [describe the query goal].
Consider performance and explain your optimization choices.
```

### **SOP-009: Unit Testing**

```
Generate unit tests for the following code, ensuring comprehensive coverage:
[Paste your code here]
```

### **SOP-010: Debugging**

```
I'm experiencing the following bug: [describe the bug]
Here's the relevant code: [paste code]
What could be causing this, and how can I fix it?
```

### **SOP-011: Project Improvement Analysis**

```
Based on our work so far, what areas of the project could be improved?
Consider code quality, architecture, and potential scalability issues.
```

### **SOP-012: Security Review**

```
Review the following code/architecture for potential security vulnerabilities and suggest improvements:
[Paste relevant information here]
```

### **SOP-013: Best Practices Update**

```
What are the latest best practices for [your tech stack] as of [current date]?
How can I apply these to my current project?
```

### **SOP-014: Git Commit Messages**

```
Analyze the project fully, and based on the following code changes and our previous conversation, generate a very "short", clear and informative git commit message
```

### **SOP-015: Merge Conflict Resolution**

```
I'm facing the following merge conflict. How should I resolve it while maintaining the intended functionality of both changes?
[Paste the conflict details here]
```

### **SOP-016: Detailed Project Documentation**

```
Based on the code and our discussions, analyze the code again, in detail, generate a folder, named `knowledge` in the root folder (if already present, add on to it) and add the following documentation in as much detail as possible:
- README file section for each component and all the component that created or worked on in this session
- progress report for the project, including from where we started, where we are know, and where we are heading
- detailed summarization of the chat messages, important information, conclusions from all the question answer sessions
- any other info related to the project should also be copied into the `knowledge` folder, which will be created in the root directory
- if any particular documentation is already present, then add onto it, in as much detail as possible
```

### **SOP-017: Knowledge Base Update**

```
Read through the existing `knowledge` folder documentation thoroughly. Then analyze the current project state and add/update the following:
- Document any new components, files, or features added since last update
- Update progress report with recent developments and current status
- Add documentation for any code changes, bug fixes, or improvements
- Update integration points and architecture changes
- Record any new decisions, conclusions, or learnings
- Ensure all existing documentation reflects current project state
- Add timestamps to track documentation freshness
```

### **SOP-018: Read files without lazing around**

```
remember, read entire files thoroughly — no truncation. Always process code from the first line to the last, never stopping arbitrarily (e.g., after 50 or 100 lines). Each read operation must cover at least 1000 lines, and continue beyond that if needed to cover the full file. Analyze all code elements: functions, classes, variables, imports/exports, and structure. Avoid phrases like "truncated for brevity" or "rest omitted" — these reflect incomplete work. Your analysis and suggestions must reflect full-file understanding, referencing and connecting code across the entire file to ensure accurate, contextual recommendations. Incomplete reads lead to poor results — thoroughness is non-negotiable.
```

### **SOP-018: Project Analysis and Code Review**

```
I don't understand this project fully and in depth, so help me in that aspect, after analying this project, I will ask quesiton about different parts, and you will have to explain them in detail and depth, and also maintain a code review doc in knowledge, to keep track of all the questions and answer (short and summarize afterwards)

Based on the code and our discussions, analyze the code, read through the existing `knowledge` folder documentation thoroughly in depth, and then we will go through code review
```

### **SOP-018: Project Analysis and Bug Fixing**

```
I am current in the phase of testing this project for any bugs, while running in production. Do the below, and understand the project in as depth as depth, to know how the bug is occuring and what needs to do to fix it, and also maintain a bug report doc in knowledge, to keep track of all the bugs and their fixes (short and summarize afterwards)

Based on the code and our discussions, analyze the code, read through the existing `knowledge` folder documentation thoroughly in depth, and then we will go through bug fixing
```

---

## Project-Specific SOPs for Forest Fire Prediction System

### **SOP-FF-001: ML Model Integration**

```
For the forest fire prediction ResUNet-A model integration:
- Model location: forest_fire_ml/
- Input: 9-band stacked GeoTIFF (DEM, ERA5, LULC, GHSL, etc.)
- Output: Fire probability maps (0-1 range, 30m resolution)
- Framework: TensorFlow/Keras
- Consider: Patch-based inference, sliding window approach, memory optimization

[Paste your specific requirement here]
```

### **SOP-FF-002: Cellular Automata Engine**

```
For the cellular automata fire spread simulation:
- Engine location: forest_fire_ml/cellular_automata/
- Input: ML probability maps + ignition points + weather data
- Framework: TensorFlow for GPU acceleration
- Temporal resolution: Hourly time steps
- Spatial resolution: 30m (maintain consistency)
- Consider: Simplified vs. advanced physics rules, performance optimization

[Paste your CA-specific requirement here]
```

### **SOP-FF-003: GeoTIFF Data Processing**

```
For geospatial data handling in the forest fire system:
- Primary library: rasterio
- Coordinate system: Maintain consistency across all layers
- Resolution: 30m target resolution
- Format: GeoTIFF for all spatial data
- Consider: Reprojection, resampling, data alignment, nodata handling

[Paste your geospatial requirement here]
```

### **SOP-FF-004: Dataset Collection Scripts**

```
For Google Earth Engine data collection scripts:
- Location: dataset collection/
- Data sources: SRTM DEM, ERA5 Daily, LULC 2020, GHSL 2015, VIIRS fire data
- Time range: April 1 - May 29, 2016 (Uttarakhand focus)
- Export format: GeoTIFF, 30m resolution
- Consider: Asset management, batch processing, cloud storage

[Paste your GEE requirement here]
```

### **SOP-FF-005: Web Interface Integration**

```
For the Flask API and React frontend:
- Backend: Flask with CORS support (web_api/app.py)
- Frontend: React with Leaflet for mapping
- API endpoints: /api/run-simulation, /api/health
- Data flow: JSON API responses with animation frames
- Theme: ISRO/fire management aesthetic
- Consider: Real-time updates, animation performance, mobile responsiveness

[Paste your web interface requirement here]
```

### **SOP-FF-006: Model Training Pipeline**

```
For ResUNet-A model training and evaluation:
- Architecture: ResUNet-A with ASPP and attention gates
- Loss function: Focal loss for class imbalance
- Metrics: IoU score, Dice coefficient
- Data augmentation: Albumentations library
- Validation: Patch-based with fire-focused sampling
- Consider: Overfitting prevention, learning rate scheduling, model checkpointing

[Paste your training requirement here]
```

### **SOP-FF-007: Simulation Configuration**

```
For cellular automata simulation parameters:
- Config location: cellular_automata/config.py
- Key parameters: Wind speed/direction, fire spread rates, barrier effects
- Scenarios: 1/2/3/6/12 hour simulations
- Output: Animation frames + statistics + metadata
- LULC mapping: Fire behavior by land use class
- Consider: Realistic vs. simplified physics, visual appeal vs. accuracy

[Paste your simulation config requirement here]
```

### **SOP-FF-008: Performance Optimization**

```
For system performance optimization:
- GPU utilization: TensorFlow GPU configuration
- Memory management: Patch-based processing, data generators
- Computation: Vectorized operations, efficient algorithms
- I/O: Optimized file formats, caching strategies
- Target: Real-time simulation for demo purposes
- Consider: Trade-offs between accuracy and speed

[Paste your performance requirement here]
```

### **SOP-FF-009: Error Handling and Validation**

```
For robust error handling in the fire prediction system:
- Data validation: Check for missing files, corrupted data, projection mismatches
- Model validation: Verify input shapes, output ranges, prediction sanity
- Simulation validation: Check for infinite loops, unrealistic spread rates
- API validation: Input parameter checking, response format validation
- Consider: Graceful degradation, user-friendly error messages

[Paste your error handling requirement here]
```

### **SOP-FF-010: Testing and Quality Assurance**

```
For comprehensive testing of fire prediction components:
- Unit tests: Individual functions and classes
- Integration tests: ML-CA pipeline, API endpoints
- Performance tests: Memory usage, processing speed
- Visual tests: Animation quality, map rendering
- Test data: Mock datasets, synthetic scenarios
- Consider: Automated testing, continuous integration

[Paste your testing requirement here]
```

### **SOP-FF-011: Deployment and Demo Preparation**

```
For deployment and demonstration setup:
- Environment: Local deployment with potential cloud scaling
- Dependencies: requirements.txt, environment setup
- Data preparation: Sample datasets, demo scenarios
- Documentation: Setup guides, API documentation
- Demo flow: Interactive walkthrough, preset scenarios
- Consider: Presentation quality, stakeholder expectations (ISRO researchers)

[Paste your deployment requirement here]
```

### **SOP-FF-012: Data Pipeline Architecture**

```
For the complete data processing pipeline:
- Raw data: GEE collection scripts → GeoTIFF exports
- Preprocessing: Data stacking, alignment, normalization
- ML inference: Sliding window prediction, probability map generation
- CA simulation: Ignition points → hourly fire spread frames
- Visualization: Animation data, statistical summaries
- Consider: Data versioning, pipeline automation, quality control

[Paste your pipeline requirement here]
```

### **SOP-FF-013: Scientific Accuracy vs. Demo Appeal**

```
For balancing scientific rigor with visual demonstration:
- Current priority: Visual appeal and functional completeness
- Future priority: Scientific accuracy and validation
- Physics models: Simplified rules now, Rothermel physics later
- Validation approach: Visual realism vs. ground truth comparison
- Documentation: Clear distinction between demo and research versions
- Consider: Stakeholder expectations, time constraints, future scalability

[Paste your accuracy/demo balance requirement here]
```

### **SOP-FF-014: Multi-Component Integration**

```
For integrating multiple system components:
- ML model outputs → CA engine inputs (probability maps)
- CA simulation results → Web interface (animation data)
- User interactions → Backend processing (ignition points, parameters)
- Real-time updates → Frontend visualization (progress tracking)
- Error propagation → User feedback (graceful failure handling)
- Consider: Data format consistency, API versioning, backward compatibility

[Paste your integration requirement here]
```

### **SOP-FF-015: Context Window Management**

```
When working with Claude on the forest fire project, establish context with:
- Project overview: "This is a forest fire prediction system with ML + CA components"
- Current working directory: Specify the exact folder/component
- Relevant files: Mention key files in the current context
- Integration points: How the current work connects to other components
- Technical constraints: Performance, accuracy, demo requirements
- Use semantic search to provide relevant code context before asking complex questions

[Paste your specific context management need here]
```

---

## Usage Guidelines

1. **Copy and paste the relevant SOP** as a prefix to your specific question
2. **Fill in the bracketed placeholders** with your specific details
3. **Combine multiple SOPs** when working on complex, multi-faceted problems
4. **Customize the SOPs** based on your specific needs and project evolution
5. **Update this document** as you discover new patterns and requirements

## Best Practices

- Always provide relevant code context when using technical SOPs
- Use the project-specific SOPs (SOP-FF-xxx) for domain-specific questions
- Combine general SOPs with project-specific ones for comprehensive guidance
- Keep your SOPs updated as the project evolves and new patterns emerge
- Use semantic search to provide relevant context before applying SOPs to complex problems

## Usage

### For ML model work:

Use SOP-FF-001 + SOP-003 (code generation) + your specific requirement

### For CA simulation:

Use SOP-FF-002 + SOP-FF-007 + SOP-002 (architecture design)

### For web interface:

Use SOP-FF-005 + SOP-006 (documentation) + your frontend needs

### For performance issues:

Use SOP-FF-008 + SOP-011 (improvement analysis)

---

_Last updated: July 7, 2025_
_Project: Forest Fire Spread Prediction System_
_Target: ISRO Demonstration & Research_
