<div align="center">

# 📍 GeoClusterEngine
**Geospatial Clustering & Business Location Recommendation System**

Interactive end‑to‑end platform for discovering high‑potential business locations using OpenStreetMap data, geospatial clustering, multi-factor scoring, and comparative expansion analysis.

</div>

## 1. Core Capabilities

- **POI Data Ingestion**: High‑volume category fetch via Overpass / OSMnx with caching.
- **Geodata Cleaning**: Deduplication, coordinate validation, category grouping, enrichment.
- **Spatial Clustering**: DBSCAN (haversine) for organic density + factor-based KMeans feature clustering.
- **Transport Proximity Annotation**: Nearest public transport (train, bus, station, footway, footpath) for clusters & top picks.
- **Multi-Factor Location Scoring**: Demand, competition, accessibility, infrastructure weighted model.
- **Best Recommendation Mode**: Auto‑suggest top business types for an area when exploring viability.
- **Multi-City Comparison**: Side‑by‑side metrics & per‑city business selection with comparative charting.
- **Scenario Modeling**: Save and compare weight configurations (what‑if scoring).
- **Static & Interactive Maps**: Folium HTML + Matplotlib PNG exports (scatter, clusters, true density heatmap).
- **PDF Reporting**: Structured executive report via ReportLab.
- **Performance Optimizations**: Category consolidation, selective radii, caching, vectorized scoring.

## 2. High-Level Architecture

```
┌──────────────────────────────────────────────────────────┐
│                         Streamlit UI                     │
│  Sidebar (inputs)  |  Tabs (Maps • Clusters • Heatmaps   │
│  Scenarios • Multi-City • Reports                        │
└───────────────┬──────────────────────────────────────────┘
    │ triggers
  ┌───────▼───────┐
  │ Pipeline Orchestrator (run_analysis_pipeline)    │
  └───────┬───────┘
    │
   ┌────────────▼────────────┐
   │   DataFetcher (OSMnx)    │ → raw POIs CSV
   └────────────┬────────────┘
    │ cleaned
  ┌───────▼───────┐
  │ DataCleaner    │ → cleaned_pois
  └───────┬───────┘
    │ coords
  ┌───────▼────────┐
  │ GeoClusterer    │ → cluster stats / labels
  └───────┬────────┘
    │ features
  ┌───────▼──────────────┐
  │ Factor Cluster Engine │ → top feature clusters
  └───────┬──────────────┘
    │ scoring
  ┌───────▼────────┐
  │ LocationScorer │ → top ranked locations
  └───────┬────────┘
    │ annotate
  ┌───────▼─────────────┐
  │ Transport Annotation │
  └───────┬─────────────┘
    │ visualization
  ┌───────▼────────┐   ┌──────────────┐
  │ MapVisualizer  │→  │ ReportGenerator │
  └────────────────┘   └──────────────┘
```

## 3. Module Responsibilities

| Module | Purpose |
|--------|---------|
| `fetch_data.py` | Fetch POIs, center geocoding, category expansion & caching |
| `clean_data.py` | Validation, deduplication, category grouping, dataframe hygiene |
| `clustering.py` | DBSCAN + hotspot/sparse detection + factor-based KMeans feature scoring |
| `visualize.py` | Folium map builders + static PNG export (scatter / clusters / heatmap) |
| `scoring.py` | Candidate grid generation, multi-factor weighting, top ranking extraction |
| `recommendations.py` | Business suggestion mode, transport proximity annotation helpers |
| `report_generator.py` | PDF compilation (summary, metrics, top sites) |
| `business_suggestion.py` | Heuristics / advanced viability profiling (future extension) |
| `simulation.py` | Agent-based simulation utilities (currently optional) |

## 4. Data Workflow

1. Geocode or coordinate input defines center.
2. Fetch categorized POIs within radius (raw → CSV).
3. Clean & normalize data (group categories, drop invalid rows).
4. Cluster (DBSCAN) + compute cluster stats (count, noise, hotspots).
5. Factor-based clustering: build localized density features (competitor / supporting / neutral) and rank clusters.
6. Generate candidate grid points → score with weighted model.
7. Select top N locations → annotate with nearest transport.
8. Render interactive & static maps; persist data artifacts.
9. Provide multi-city comparison & scenario modeling views.
10. Optional PDF report generation.

## 5. Scoring & Recommendation Engine

```
final_score = (w_demand * demand_score
       + w_competition * competition_score
       + w_accessibility * accessibility_score
       + w_infrastructure * infrastructure_score)
```

| Component | Driver | Interpretation |
|-----------|--------|---------------|
| Demand | Supporting POI density | Higher = more potential customer flow |
| Competition | Competitor density (inverse) | Lower competitor presence boosts score |
| Accessibility | Transport / road access | Better access improves viability |
| Infrastructure | Essential support services | Banks / hospitals / schools add robustness |

Weights are normalized automatically to sum to 1 based on user sliders.

## 6. Factor-Based Cluster Recommendations

Build localized density features per cluster: competitor, supporting, neutral. Apply KMeans to these vectors; derive a cluster score (e.g. competitor penalty + supporting boost). Top clusters are annotated with nearest transport nodes for operational planning.

## 7. Transport Proximity Annotation

For each cluster and recommended location, nearest transport POI among filtered keywords: `['train','bus','station','footway','footpath']`. Distance computed via haversine → surfaced in UI tables.

## 8. Multi-City & Scenario Modeling

- **Multi-City**: Add cities with distinct radii & business types; table & Altair chart compare POIs, clusters, competitors, top score.
- **Scenario Modeling**: Save weight configurations; compare top scores; perform quick re-score without refetching POIs.

## 9. Performance & Caching

- Category list minimized & deduplicated.
- Vectorized distance / density calculations.
- Intermediate CSV artifacts allow reuse.
- Lightweight progress feedback & terminal summaries for POI, cleaning, clustering, scoring.

## 10. Directory Structure (Effective)

```
GeoClusterEngine/
  app.py                # Streamlit application
  main.py               # CLI entry
  pyproject.toml        # Project metadata & dependencies
  modules/              # Functional components
  data/                 # Generated CSV outputs
  maps/                 # HTML + PNG maps
  README.md             # Documentation
```

## 11. Local Setup

```bash
git clone <repo-url>
cd GeoClusterEngine
python -m venv .venv
./.venv/Scripts/activate  # Windows PowerShell
pip install -U pip
pip install .             # Uses pyproject dependencies
```

If pip install . is not desired:
```bash
pip install folium geopandas numpy osmnx pandas reportlab requests scikit-learn shapely streamlit streamlit-folium scipy matplotlib seaborn
```

## 12. Running the Dashboard

```bash
streamlit run app.py --server.port 8501
```
Access: http://localhost:8501

## 13. CLI Usage

```bash
python main.py --city "Bangalore" --business "cafe" --radius 5
python main.py --interactive
```

| Flag | Purpose |
|------|---------|
| `--city` | City name or descriptor |
| `--business` | Business type (e.g. cafe, gym, bakery) |
| `--radius` | Search radius in km |
| `--interactive` | Guided prompts mode |

## 14. Key Output Artifacts

| File | Description |
|------|-------------|
| `data/raw_data.csv` | Raw fetched POIs |
| `data/cleaned_data.csv` | Cleaned, normalized POIs |
| `data/clusters.csv` | Cluster labels & metadata |
| `data/location_scores.csv` | Ranked candidate grid locations |
| `maps/*.html` | Interactive Folium maps |
| `maps/*.png` | Static export images |
| `data/analysis_report.pdf` | Generated PDF (on request) |

## 15. Extending the System

| Goal | Approach |
|------|----------|
| Add new POI categories | Extend base list in `run_analysis_pipeline` |
| Change clustering radius | Adjust `eps_km` heuristic in `clustering.py` |
| New scoring factor | Add column computation + integrate in `LocationScorer` |
| Advanced ML clustering | Integrate HDBSCAN / OPTICS under `clustering.py` |
| API service layer | Wrap pipeline in FastAPI endpoints |

## 16. Troubleshooting

| Issue | Resolution |
|-------|------------|
| No POIs found | Increase radius; verify spelling; try coordinates |
| Slow fetch | Reduce category list; ensure stable network |
| Map not rendering | Confirm HTML file exists in `maps/` |
| Score seems off | Re-check weights normalization & business type |
| Transport empty | Area lacks defined transport POIs under current keywords |

## 17. Dependencies (pyproject)

| Package | Purpose |
|---------|---------|
| osmnx | OSM data acquisition & graph operations |
| geopandas | Spatial dataframe operations |
| folium / streamlit-folium | Interactive web mapping |
| scikit-learn | Clustering, KMeans, DBSCAN |
| pandas / numpy / scipy | Data manipulation & numerics |
| shapely | Geometry handling |
| matplotlib / seaborn | Static visualization exports |
| reportlab | PDF report generation |
| streamlit | UI framework |

## 18. License & Attribution

Open source (flexible use). Data © OpenStreetMap contributors. Built with OSMnx, Folium, GeoPandas, Scikit‑learn.

## 19. Quick Start (One-Liner)

```bash
python -m venv .venv && ./.venv/Scripts/activate && pip install . && streamlit run app.py
```

---
**GeoClusterEngine** – Turn raw geospatial POIs into actionable expansion insights.

## Project Structure

```
/project
    /data
        raw_data.csv           # Original fetched POI data
        cleaned_data.csv       # Processed and cleaned data
        clusters.csv           # POIs with cluster labels
        location_scores.csv    # Scored candidate locations
    /maps
        all_pois_map.html      # Interactive map of all POIs
        cluster_map.html       # Cluster visualization map
        heatmap.html           # POI density heatmap
        competition_heatmap.html # Competition analysis map
        recommendations_map.html # Top recommended locations
    /modules
        __init__.py            # Package initialization
        fetch_data.py          # OSM data fetching module
        clean_data.py          # Data cleaning module
        clustering.py          # Clustering algorithms
        visualize.py           # Map visualization module
        scoring.py             # Location scoring module
    main.py                    # Command-line orchestration
    app.py                     # Streamlit dashboard
    README.md                  # This file
```

## Installation

The project uses the following Python packages:

```bash
pip install osmnx geopandas folium scikit-learn pandas numpy shapely streamlit streamlit-folium
```

## How to Run

### Option 1: Streamlit Dashboard (Recommended)

```bash
streamlit run app.py --server.port 5000
```

Then open your browser to the provided URL to access the interactive dashboard.

### Option 2: Command Line Interface

```bash
# Interactive mode
python main.py --interactive

# Direct execution
python main.py --city "Bangalore" --business "cafe" --radius 5
```

### Command Line Arguments

| Argument | Short | Description |
|----------|-------|-------------|
| `--city` | `-c` | City name to analyze |
| `--business` | `-b` | Business type (cafe, restaurant, shop, etc.) |
| `--radius` | `-r` | Search radius in kilometers (default: 5) |
| `--interactive` | `-i` | Run in interactive mode |

## Algorithms Used

### 1. DBSCAN Clustering

- **Purpose**: Identifies dense clusters of POIs and noise points
- **Distance Metric**: Haversine distance (accounts for Earth's curvature)
- **Parameters**:
  - `eps`: Maximum distance between points (in radians)
  - `min_samples`: Minimum points to form a cluster

### 2. K-Means Clustering

- **Purpose**: Alternative clustering for comparison
- **Features**: Finds K cluster centers using iterative optimization
- **Used for**: Optimal cluster count estimation via silhouette score

### 3. Kernel Density Estimation (KDE)

- **Purpose**: Creates smooth density surfaces for heatmaps
- **Implementation**: Gaussian kernel over coordinate space
- **Output**: Interactive Folium heatmap layers

### 4. Location Scoring Model

```
Final Score = w1 * demand_score + 
              w2 * competition_score + 
              w3 * accessibility_score + 
              w4 * infrastructure_score
```

**Default Weights:**
- Demand: 0.4 (40%)
- Competition: 0.3 (30%)
- Accessibility: 0.2 (20%)
- Infrastructure: 0.1 (10%)

**Score Components:**

| Score | Description | Calculation |
|-------|-------------|-------------|
| Demand | Supporting POIs density | Count of shops, restaurants nearby |
| Competition | Inverse competitor density | Lower competition = higher score |
| Accessibility | Transport access | Bus stops, parking, major roads |
| Infrastructure | Essential services | Hospitals, banks, schools nearby |

## POI Categories Supported

- **Retail**: shop, supermarket, convenience, mall
- **Food & Beverage**: restaurant, cafe, fast_food, bakery
- **Finance**: bank, atm
- **Healthcare**: hospital, pharmacy, clinic
- **Education**: school, university, college
- **Hospitality**: hotel, hostel
- **Transport**: bus_station, parking, fuel
- **Leisure**: gym, cinema

## Output Files

### CSV Files (in `/data/`)

1. **raw_data.csv**: Original fetched POI data
2. **cleaned_data.csv**: Processed data with validated coordinates
3. **clusters.csv**: POIs with cluster assignments and hotspot flags
4. **location_scores.csv**: Ranked candidate locations with scores

### HTML Maps (in `/maps/`)

1. **all_pois_map.html**: All POIs with marker clustering
2. **cluster_map.html**: Color-coded cluster visualization
3. **heatmap.html**: POI density heatmap
4. **competition_heatmap.html**: Competition concentration map
5. **recommendations_map.html**: Top 10 recommended locations

## Dashboard Features

The Streamlit dashboard provides:

- **Configuration Panel**: City, business type, radius, and weight adjustments
- **POI Map Tab**: Interactive map of all fetched points of interest
- **Clusters Tab**: Cluster visualization with statistics
- **Heatmaps Tab**: Toggle between density and competition heatmaps
- **Recommendations Tab**: Top 10 locations with detailed scores
- **Data & Reports Tab**: Download CSV files and view analysis summary

## Example Usage

### Dashboard Workflow

1. Enter city name (e.g., "Bangalore", "Mumbai", "New York")
2. Select business type from dropdown
3. Adjust search radius (1-20 km)
4. Optionally modify scoring weights
5. Click "Run Analysis"
6. Explore maps and recommendations in different tabs
7. Download data files for further analysis

### Console Output Example

```
============================================================
GEOSPATIAL CLUSTERING & BUSINESS LOCATION RECOMMENDATION
============================================================

City: Bangalore
Business Type: cafe
Search Radius: 5 km

[1/6] Fetching POI Data from OpenStreetMap...
  City center: 12.97194, 77.59369
  Fetching shop... Found 523 POIs
  Fetching restaurant... Found 312 POIs
  ...

[2/6] Cleaning and Preparing Data...
  Initial records: 1842
  After cleaning: 1756
  Retention rate: 95.3%

[3/6] Performing Clustering Analysis...
  Total clusters found: 15
  Noise points: 127
  Hotspots identified: 4

[4/6] Generating Visualizations...
  Created: all_pois_map.html
  Created: cluster_map.html
  ...

[5/6] Scoring Potential Locations...
  Candidates analyzed: 177
  Top Score: 0.847

[6/6] Creating Recommendations Map...

ANALYSIS COMPLETE!
============================================================
```

## Future Improvements

1. **Real-time Data Integration**
   - Google Places API for ratings and reviews
   - Population density data from census APIs
   - Real-time foot traffic data

2. **Advanced Analytics**
   - Time-series analysis for seasonal patterns
   - Competitor performance prediction
   - Market saturation modeling

3. **Enhanced Visualization**
   - 3D terrain visualization
   - Catchment area analysis
   - Isochrone maps for travel time

4. **Machine Learning Enhancements**
   - HDBSCAN for variable-density clustering
   - Neural network-based scoring
   - Demand prediction models

5. **Export Options**
   - PDF report generation
   - GeoJSON export for GIS software
   - API endpoint for integration

## Technical Notes

- The system uses the WGS84 (EPSG:4326) coordinate reference system
- Haversine distance calculations account for Earth's curvature
- KD-Tree data structure used for efficient nearest-neighbor queries
- Memory-efficient streaming for large datasets

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| osmnx | 2.0+ | OpenStreetMap data fetching |
| geopandas | 0.14+ | Geospatial data handling |
| folium | 0.15+ | Interactive map creation |
| scikit-learn | 1.3+ | Clustering algorithms |
| pandas | 2.0+ | Data manipulation |
| numpy | 1.24+ | Numerical operations |
| shapely | 2.0+ | Geometric operations |
| streamlit | 1.28+ | Web dashboard |
| streamlit-folium | 0.15+ | Folium-Streamlit integration |

<!-- Legacy sections (License & Acknowledgments) consolidated above -->
