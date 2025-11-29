"""
Geospatial Clustering & Business Location Recommendation System
Streamlit Dashboard Application

This dashboard provides an interactive interface for:
- Selecting city and business type
- Configuring analysis parameters
- Viewing POI maps and clusters
- Exploring heatmaps and recommendations
- What-if scenario modeling
- PDF report generation
- Multi-city comparison
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import folium
import altair as alt
from streamlit_folium import st_folium, folium_static
import time
import json
from datetime import datetime

from modules.fetch_data import DataFetcher
from modules.clean_data import DataCleaner
try:
    from modules.clustering import GeoClusterer, compute_factor_cluster_recommendations
except ImportError:
    from modules.clustering import GeoClusterer
    # Fallback stub to avoid runtime failure if function absent
    def compute_factor_cluster_recommendations(*args, **kwargs):
        return {
            'enriched_pois': None,
            'cluster_summary': pd.DataFrame(),
            'top_clusters': pd.DataFrame()
        }
from modules.visualize import MapVisualizer
from modules.scoring import LocationScorer
from modules.report_generator import ReportGenerator
from modules.business_suggestion import BusinessSuggester
from modules.simulation import run_simulation, generate_agents, analyze_simulation_results
from modules.clustering import compute_kde, smooth_heatmap
from modules.scoring import compute_opportunity_score, compute_confidence
from modules.visualize import plot_heatmap


os.makedirs('data', exist_ok=True)
os.makedirs('maps', exist_ok=True)


st.set_page_config(
    page_title="Business Location Recommender",
    page_icon="📍",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 1.5rem;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 0.8rem 1.5rem;
    }
    .scenario-card {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize session state variables."""
    if 'analysis_complete' not in st.session_state:
        st.session_state.analysis_complete = False
    if 'results' not in st.session_state:
        st.session_state.results = {}
    if 'cleaned_pois' not in st.session_state:
        st.session_state.cleaned_pois = None
    if 'clustered_pois' not in st.session_state:
        st.session_state.clustered_pois = None
    if 'top_locations' not in st.session_state:
        st.session_state.top_locations = None
    if 'center' not in st.session_state:
        st.session_state.center = None
    if 'cluster_stats' not in st.session_state:
        st.session_state.cluster_stats = None
    if 'saved_scenarios' not in st.session_state:
        st.session_state.saved_scenarios = []
    if 'multi_city_results' not in st.session_state:
        st.session_state.multi_city_results = {}
    if 'current_city' not in st.session_state:
        st.session_state.current_city = None
    if 'current_business' not in st.session_state:
        st.session_state.current_business = None
    if 'current_weights' not in st.session_state:
        st.session_state.current_weights = None


def run_analysis_pipeline(city: str, business_type: str, radius_km: float,
                          weights: dict, progress_bar, center_coords=None) -> dict:
    """
    Run the complete analysis pipeline with progress updates.
    
    Args:
        city: Name of city or location description
        business_type: Type of business to analyze
        radius_km: Search radius in kilometers
        weights: Dictionary of scoring weights
        progress_bar: Streamlit progress bar object
        center_coords: Optional tuple of (latitude, longitude) to use instead of geocoding city name
    """
    results = {}
    
    # Create fetcher instance
    fetcher = DataFetcher("", radius_km)
    
    if center_coords is not None:
        # Use provided coordinates directly
        progress_bar.progress(10, "Using provided coordinates...")
        center = center_coords
        fetcher.center_point = center
    else:
        # Geocode city name
        progress_bar.progress(10, "Fetching city location...")
        fetcher = DataFetcher(city, radius_km)
        center = fetcher.get_city_center()
        
        if center is None:
            st.error(f"Could not find city: {city}. Please check the spelling.")
            return None
    
    results['center'] = center
    results['city'] = city
    results['business_type'] = business_type
    results['radius_km'] = radius_km
    results['weights'] = weights.copy()
    st.session_state.center = center
    
    progress_bar.progress(20, "Fetching POI data from OpenStreetMap...")

    # Build POI categories to fetch
    base_categories = [
        'shop', 'restaurant', 'cafe', 'supermarket', 'bank',
        'hospital', 'pharmacy', 'school', 'hotel', 'fuel',
        'parking', 'bus_station'
    ]

    is_best_mode = str(business_type).lower().strip() in [
        'best recommendation', 'best_recommendation', 'best'
    ]

    if is_best_mode:
        # Use profiles from recommendations module (no synthetic expansion)
        from modules.recommendations import compute_business_recommendations
        candidate_types = list(BUSINESS_PROFILES.keys()) if 'BUSINESS_PROFILES' in globals() else [
            'cafe','restaurant','bakery','fast_food','shop','supermarket','pharmacy','bank','gym','hotel'
        ]
        categories = list(set(base_categories + candidate_types))
    else:
        categories = list(set(base_categories + [business_type]))
    
    all_pois = fetcher.fetch_all_pois(categories)
    
    if len(all_pois) == 0:
        st.error("No POIs found in the specified area. Try a larger radius or different city.")
        return None
    
    fetcher.save_to_csv(all_pois, 'raw_data.csv')
    results['raw_poi_count'] = len(all_pois)
    
    progress_bar.progress(40, "Cleaning and preparing data...")
    cleaner = DataCleaner()
    cleaned_pois = cleaner.clean_geodataframe(all_pois)
    
    fetcher.save_to_csv(cleaned_pois, 'cleaned_data.csv')
    st.session_state.cleaned_pois = cleaned_pois
    results['cleaned_poi_count'] = len(cleaned_pois)
    results['cleaning_report'] = cleaner.get_cleaning_report()
    
    progress_bar.progress(55, "Performing clustering analysis...")
    coords = cleaner.prepare_for_clustering(cleaned_pois)
    
    clusterer = GeoClusterer()
    eps_km = min(radius_km / 10, 0.5)
    min_samples = max(3, int(len(coords) / 100))
    
    labels = clusterer.dbscan_clustering(coords, eps_km=eps_km, min_samples=min_samples)
    hotspots = clusterer.detect_hotspots(coords)
    sparse_regions = clusterer.detect_sparse_regions(coords, center[0], center[1])
    
    clustered_pois = clusterer.add_cluster_labels(cleaned_pois)
    clusterer.save_clusters(clustered_pois)
    
    st.session_state.clustered_pois = clustered_pois
    st.session_state.cluster_stats = clusterer.cluster_stats
    results['cluster_stats'] = clusterer.cluster_stats
    results['hotspots'] = hotspots
    results['sparse_regions'] = sparse_regions
    
    # Factor-based cluster recommendations (KMeans on competitor/supporting/neutral densities)
    try:
        factor_cluster_res = compute_factor_cluster_recommendations(
            cleaned_pois,
            business_type=business_type,
            competitor_radius_km=min(0.4, radius_km/10),
            supporting_radius_km=min(0.5, radius_km/8),
            neutral_radius_km=min(0.4, radius_km/10)
        )
        # Annotate top clusters with nearest transport
        from modules.recommendations import annotate_clusters_with_transport
        top_clusters = factor_cluster_res.get('top_clusters')
        if top_clusters is not None and len(top_clusters) > 0:
            top_clusters = annotate_clusters_with_transport(top_clusters, cleaned_pois)
        results['factor_cluster_recommendations'] = top_clusters
        results['factor_cluster_summary'] = factor_cluster_res.get('cluster_summary')
    except Exception as e:
        results['factor_cluster_recommendations'] = None
        results['factor_cluster_summary'] = None

    # If "Best Recommendation" mode, compute top business types first
    selected_business = business_type
    best_suggestions_records = None
    if is_best_mode:
        try:
            from modules.recommendations import compute_business_recommendations
            suggestions_df = compute_business_recommendations(
                cleaned_pois, center[0], center[1], radius_km, weights
            )
            best_suggestions_records = suggestions_df.head(3).to_dict(orient='records')
            if len(suggestions_df) > 0:
                selected_business = str(suggestions_df.iloc[0]['Business Type']).lower()
        except Exception as e:
            st.warning(f"Best recommendation computation failed: {e}")
            selected_business = 'cafe'

    progress_bar.progress(70, "Scoring potential locations...")
    scorer = LocationScorer(weights)
    
    supporting_categories = fetcher.get_supporting_pois(selected_business)
    candidates = scorer.generate_candidate_locations(
        center[0], center[1],
        radius_km=radius_km,
        grid_size=15
    )
    
    scores = scorer.score_locations(
        candidates,
        cleaned_pois,
        selected_business,
        supporting_categories
    )
    
    scorer.save_scores()
    top_locations = scorer.get_top_locations(10)
    # Annotate with nearest transport
    try:
        from modules.recommendations import annotate_locations_with_transport
        top_locations_annotated = annotate_locations_with_transport(top_locations, cleaned_pois)
    except Exception:
        top_locations_annotated = top_locations
    st.session_state.top_locations = top_locations_annotated
    results['top_locations'] = top_locations_annotated
    results['top_locations_base'] = top_locations
    results['scoring_report'] = scorer.get_analysis_report()
    # Record best-mode outputs
    results['best_mode'] = is_best_mode
    if best_suggestions_records is not None:
        results['best_suggestions'] = best_suggestions_records
    # Persist final business type used for scoring
    results['business_type_used'] = selected_business
    if is_best_mode:
        # Align primary business type with what was actually used downstream
        results['business_type'] = selected_business
    
    progress_bar.progress(85, "Generating maps...")
    viz = MapVisualizer(center[0], center[1], zoom_start=13)
    
    try:
        all_pois_map = viz.create_all_pois_map(cleaned_pois)
        viz.save_map(all_pois_map, 'all_pois_map.html')
        # Static image
        viz.save_static_scatter(cleaned_pois, 'all_pois_map.png', center_lat=center[0], center_lon=center[1])
    except Exception as e:
        st.warning(f"Could not generate POI map: {e}")
    
    try:
        cluster_map = viz.create_cluster_map(clustered_pois, clusterer.cluster_stats)
        viz.save_map(cluster_map, 'cluster_map.html')
        # Annotated cluster scatter
        viz.save_static_cluster_scatter(clustered_pois, 'cluster_map.png', center_lat=center[0], center_lon=center[1])
    except Exception as e:
        st.warning(f"Could not generate cluster map: {e}")
    
    try:
        heatmap = viz.create_heatmap(cleaned_pois)
        viz.save_map(heatmap, 'heatmap.html')
        # Dedicated density heatmap export
        viz.save_static_heatmap(cleaned_pois, 'heatmap.png', center_lat=center[0], center_lon=center[1])
    except Exception as e:
        st.warning(f"Could not generate heatmap: {e}")
    
    try:
        competition_map = viz.create_competition_heatmap(cleaned_pois, selected_business)
        viz.save_map(competition_map, 'competition_heatmap.html')
    except Exception as e:
        st.warning(f"Could not generate competition heatmap: {e}")
    
    try:
        recommendations_map = viz.create_recommendations_map(
            cleaned_pois, top_locations, selected_business
        )
        viz.save_map(recommendations_map, 'recommendations_map.html')
    except Exception as e:
        st.warning(f"Could not generate recommendations map: {e}")
    
    progress_bar.progress(100, "Analysis complete!")
    time.sleep(0.5)
    
    return results


def run_quick_rescore(weights: dict) -> pd.DataFrame:
    """
    Re-score existing data with new weights without refetching.
    """
    if st.session_state.cleaned_pois is None or st.session_state.center is None:
        return None
    
    cleaned_pois = st.session_state.cleaned_pois
    center = st.session_state.center
    results = st.session_state.results
    
    scorer = LocationScorer(weights)
    
    business_type = results.get('business_type', 'cafe')
    radius_km = results.get('radius_km', 5.0)
    
    fetcher = DataFetcher("", radius_km)
    supporting_categories = fetcher.get_supporting_pois(business_type)
    
    candidates = scorer.generate_candidate_locations(
        center[0], center[1],
        radius_km=radius_km,
        grid_size=15
    )
    
    scores = scorer.score_locations(
        candidates,
        cleaned_pois,
        business_type,
        supporting_categories
    )
    
    return scorer.get_top_locations(10)


def save_scenario(name: str, weights: dict, top_locations: pd.DataFrame):
    """Save current scenario for comparison."""
    scenario = {
        'name': name,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M'),
        'weights': weights.copy(),
        'top_score': float(top_locations['final_score'].max()) if len(top_locations) > 0 else 0,
        'top_location': {
            'lat': float(top_locations.iloc[0]['latitude']) if len(top_locations) > 0 else 0,
            'lon': float(top_locations.iloc[0]['longitude']) if len(top_locations) > 0 else 0,
            'score': float(top_locations.iloc[0]['final_score']) if len(top_locations) > 0 else 0
        },
        'top_5_scores': top_locations.head(5)['final_score'].tolist() if len(top_locations) > 0 else []
    }
    st.session_state.saved_scenarios.append(scenario)


def display_metrics(results: dict):
    """Display key metrics in a row of cards."""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total POIs",
            results.get('cleaned_poi_count', 0),
            f"Raw: {results.get('raw_poi_count', 0)}"
        )
    
    with col2:
        cluster_stats = results.get('cluster_stats', {})
        st.metric(
            "Clusters Found",
            cluster_stats.get('n_clusters', 0),
            f"Noise: {cluster_stats.get('n_noise', 0)}"
        )
    
    with col3:
        st.metric(
            "Hotspots",
            len(results.get('hotspots', [])),
            "High-density areas"
        )
    
    with col4:
        top_score = 0
        if results.get('top_locations') is not None and len(results['top_locations']) > 0:
            top_score = results['top_locations']['final_score'].max()
        st.metric(
            "Top Score",
            f"{top_score:.3f}",
            "Best location"
        )


def render_map_from_file(filepath: str, height: int = 500):
    """Render a saved HTML map file."""
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            html_content = f.read()
        st.components.v1.html(html_content, height=height, scrolling=True)
    else:
        st.warning(f"Map file not found: {filepath}")


def generate_pdf_report(city: str, business_type: str, radius_km: float,
                        results: dict, weights: dict, top_locations: pd.DataFrame) -> str:
    """Generate PDF report and return the file path."""
    report_gen = ReportGenerator('data/analysis_report.pdf')
    return report_gen.generate_report(
        city, business_type, radius_km,
        results, weights, top_locations
    )


def main():
    """Main Streamlit application."""
    initialize_session_state()
    
    st.markdown('<h1 class="main-header">📍 Business Location Recommender</h1>', 
                unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Geospatial Clustering & Location Recommendation System</p>', 
                unsafe_allow_html=True)
    
    with st.sidebar:
        st.header("Configuration")
        
        # Location input method selection
        location_method = st.radio(
            "How would you like to specify the location?",
            options=["City Name", "Coordinates (Lat/Lon)"],
            horizontal=True,
            help="Choose between entering a city name or geographic coordinates"
        )
        
        if location_method == "City Name":
            city = st.text_input(
                "City Name",
                value="Bangalore",
                help="Enter the city name for analysis"
            )
            center_coords = None
        else:
            col1, col2 = st.columns(2)
            with col1:
                latitude = st.number_input(
                    "Latitude",
                    value=12.9716,
                    format="%.6f",
                    help="Enter latitude coordinate (e.g., 12.9716 for Bangalore)"
                )
            with col2:
                longitude = st.number_input(
                    "Longitude",
                    value=77.5946,
                    format="%.6f",
                    help="Enter longitude coordinate (e.g., 77.5946 for Bangalore)"
                )
            city = f"Location ({latitude}, {longitude})"
            center_coords = (latitude, longitude)
        
        business_options = [
            'cafe', 'restaurant', 'shop', 'supermarket', 'pharmacy',
            'gym', 'hotel', 'bakery', 'fast_food', 'bank', 'Best Recommendation'
        ]
        business_type = st.selectbox(
            "Business Type",
            options=business_options,
            index=0,
            help="Select the type of business to analyze"
        )
        
        radius_km = st.slider(
            "Search Radius (km)",
            min_value=1.0,
            max_value=20.0,
            value=5.0,
            step=0.5,
            help="Radius around city center to search for POIs"
        )
        
        st.subheader("Scoring Weights")
        st.caption("Adjust weights to prioritize different factors")
        
        w_demand = st.slider("Demand Weight", 0.0, 1.0, 0.4, 0.05)
        w_competition = st.slider("Competition Weight", 0.0, 1.0, 0.3, 0.05)
        w_accessibility = st.slider("Accessibility Weight", 0.0, 1.0, 0.2, 0.05)
        w_infrastructure = st.slider("Infrastructure Weight", 0.0, 1.0, 0.1, 0.05)
        
        total = w_demand + w_competition + w_accessibility + w_infrastructure
        weights = {
            'demand': w_demand / total,
            'competition': w_competition / total,
            'accessibility': w_accessibility / total,
            'infrastructure': w_infrastructure / total
        }
        
        st.session_state.current_weights = weights
        
        st.divider()
        
        run_analysis = st.button("🚀 Run Analysis", type="primary", use_container_width=True)
        
        if st.session_state.analysis_complete:
            st.success("Analysis Complete!")
            
            if st.button("🔄 Re-score with New Weights", use_container_width=True):
                with st.spinner("Re-scoring..."):
                    new_top = run_quick_rescore(weights)
                    if new_top is not None:
                        st.session_state.top_locations = new_top
                        st.session_state.results['top_locations'] = new_top
                        st.rerun()
    
    if run_analysis:
        st.session_state.analysis_complete = False
        st.session_state.current_city = city
        st.session_state.current_business = business_type
        
        with st.spinner("Running analysis..."):
            progress_bar = st.progress(0, "Initializing...")
            
            results = run_analysis_pipeline(
                city, business_type, radius_km, weights, progress_bar, 
                center_coords=center_coords
            )
            
            if results:
                st.session_state.results = results
                st.session_state.analysis_complete = True
                st.rerun()
    
    if st.session_state.analysis_complete and st.session_state.results:
        results = st.session_state.results
        
        display_metrics(results)
        
        st.divider()
        
        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
            "📍 POI Map",
            "🔵 Clusters",
            "🔥 Heatmaps",
            "⭐ Recommendations",
            "🎯 Scenario Modeling",
            "🌍 Multi-City",
            "📊 Reports"
        ])
        
        with tab1:
            st.subheader("All Points of Interest")
            st.caption("Interactive map showing all fetched POIs with marker clustering")
            render_map_from_file('maps/all_pois_map.html', height=600)
            # Download buttons
            col_dl1, col_dl2 = st.columns([1,1])
            with col_dl1:
                try:
                    with open('maps/all_pois_map.html','r',encoding='utf-8') as f:
                        st.download_button(label="Download HTML", data=f.read(), file_name='all_pois_map.html', mime='text/html')
                except: pass
            with col_dl2:
                try:
                    with open('maps/all_pois_map.png','rb') as f:
                        st.download_button(label="Download PNG", data=f.read(), file_name='all_pois_map.png', mime='image/png')
                except: pass
        
        with tab2:
            st.subheader("Cluster Analysis")
            
            col1, col2 = st.columns([3, 1])
            
            with col1:
                render_map_from_file('maps/cluster_map.html', height=500)
                dlc1, dlc2 = st.columns([1,1])
                with dlc1:
                    try:
                        with open('maps/cluster_map.html','r',encoding='utf-8') as f:
                            st.download_button("HTML", f.read(), file_name='cluster_map.html', mime='text/html')
                    except: pass
                with dlc2:
                    try:
                        with open('maps/cluster_map.png','rb') as f:
                            st.download_button("PNG", f.read(), file_name='cluster_map.png', mime='image/png')
                    except: pass
            
            with col2:
                st.markdown("**Cluster Statistics**")
                cluster_stats = results.get('cluster_stats', {})
                st.write(f"Total Clusters: {cluster_stats.get('n_clusters', 0)}")
                st.write(f"Noise Points: {cluster_stats.get('n_noise', 0)}")
                
                if 'clusters' in cluster_stats:
                    st.markdown("**Top Clusters by Size:**")
                    clusters = cluster_stats['clusters']
                    sorted_clusters = sorted(
                        clusters.items(),
                        key=lambda x: x[1]['size'],
                        reverse=True
                    )[:5]
                    
                    for label, data in sorted_clusters:
                        st.write(f"Cluster {label}: {data['size']} POIs")

                # Nearest public transport for top factor-based clusters
                factor_recs_for_tab2 = results.get('factor_cluster_recommendations')
                if factor_recs_for_tab2 is not None and len(factor_recs_for_tab2) > 0:
                    st.markdown("**Nearest Public Transport (Top Clusters)**")
                    fr = pd.DataFrame(factor_recs_for_tab2).copy()
                    cols = [
                        c for c in [
                            'rank','cluster_id','cluster_score',
                            'nearest_transport_type','nearest_transport_name','nearest_transport_distance_km'
                        ] if c in fr.columns
                    ]
                    if cols:
                        fr_display = fr[cols].copy()
                        if 'cluster_score' in fr_display.columns:
                            fr_display['cluster_score'] = fr_display['cluster_score'].round(3)
                        st.dataframe(fr_display, hide_index=True, use_container_width=True)
        
        with tab3:
            st.subheader("Density & Competition Heatmaps")
            
            heatmap_type = st.radio(
                "Select Heatmap",
                ["POI Density", "Competition"],
                horizontal=True
            )
            
            if heatmap_type == "POI Density":
                render_map_from_file('maps/heatmap.html', height=550)
                dh1, dh2 = st.columns([1,1])
                with dh1:
                    try:
                        with open('maps/heatmap.html','r',encoding='utf-8') as f:
                            st.download_button("HTML", f.read(), file_name='heatmap.html', mime='text/html')
                    except: pass
                with dh2:
                    try:
                        with open('maps/heatmap.png','rb') as f:
                            st.download_button("PNG", f.read(), file_name='heatmap.png', mime='image/png')
                    except: pass
            else:
                render_map_from_file('maps/competition_heatmap.html', height=550)
                ch1, ch2 = st.columns([1,1])
                with ch1:
                    try:
                        with open('maps/competition_heatmap.html','r',encoding='utf-8') as f:
                            st.download_button("HTML", f.read(), file_name='competition_heatmap.html', mime='text/html')
                    except: pass
                with ch2:
                    try:
                        # No dedicated PNG; reuse heatmap if desired
                        if os.path.exists('maps/heatmap.png'):
                            with open('maps/heatmap.png','rb') as f:
                                st.download_button("PNG", f.read(), file_name='competition_heatmap.png', mime='image/png')
                    except: pass
        
        with tab4:
            st.subheader("Top Recommended Locations")

            # If in Best Recommendation mode, show top-3 business types
            if results.get('best_mode') and results.get('best_suggestions'):
                st.markdown("**Top Business Types for This Area**")
                best_df = pd.DataFrame(results['best_suggestions'])
                st.dataframe(best_df, hide_index=True, use_container_width=True)

                # Allow quick re-score by choosing among top-3 types
                suggestion_names = [r.get('Business Type', '') for r in results['best_suggestions']]
                chosen = st.selectbox(
                    "Re-score with one of the top business types",
                    options=suggestion_names,
                    index=0 if suggestion_names else None,
                    key="best_reco_pick"
                )
                if chosen:
                    col_rescore_a, col_rescore_b = st.columns([1,1])
                    with col_rescore_a:
                        if st.button("🔄 Re-score with selected type", use_container_width=True, key="rescore_best_pick"):
                            # Update current selection and re-score quickly
                            st.session_state.results['business_type'] = chosen.lower()
                            st.session_state.current_business = chosen.lower()
                            with st.spinner("Re-scoring for selected business type..."):
                                new_top = run_quick_rescore(st.session_state.current_weights)
                                if new_top is not None:
                                    st.session_state.top_locations = new_top
                                    st.session_state.results['top_locations'] = new_top
                                    st.rerun()
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                render_map_from_file('maps/recommendations_map.html', height=500)
                # Moved Factor-Based Cluster Recommendations directly below the map
                factor_recs_below_map = results.get('factor_cluster_recommendations')
                if factor_recs_below_map is not None and len(factor_recs_below_map) > 0:
                    st.markdown("**Factor-Based Cluster Recommendations**")
                    fr_map_df = pd.DataFrame(factor_recs_below_map)
                    fr_map_df_display = fr_map_df.copy()
                    if 'cluster_score' in fr_map_df_display.columns:
                        fr_map_df_display['cluster_score'] = fr_map_df_display['cluster_score'].round(3)
                    st.dataframe(fr_map_df_display, hide_index=True, use_container_width=True)
                    st.caption("Score = supporting_density_mean - 0.7*competitor_density_mean + 0.2*neutral_density_mean")
            
            with col2:
                st.markdown("**Top 10 Locations**")
                
                top_locations = st.session_state.top_locations
                if top_locations is not None and len(top_locations) > 0:
                    for idx, row in top_locations.head(10).iterrows():
                        rank = int(row['rank'])
                        score = row['final_score']
                        
                        if rank <= 3:
                            st.markdown(f"🥇 **#{rank}** Score: {score:.3f}" if rank == 1 else 
                                       f"🥈 **#{rank}** Score: {score:.3f}" if rank == 2 else
                                       f"🥉 **#{rank}** Score: {score:.3f}")
                        else:
                            st.write(f"#{rank} Score: {score:.3f}")
                        
                        with st.expander(f"Details & Transport"):
                            st.write(f"Latitude: {row['latitude']:.5f}")
                            st.write(f"Longitude: {row['longitude']:.5f}")
                            st.write(f"Demand: {row['demand_score']:.3f}")
                            st.write(f"Competition: {row['competition_score']:.3f}")
                            st.write(f"Accessibility: {row['accessibility_score']:.3f}")
                            st.write(f"Infrastructure: {row['infrastructure_score']:.3f}")
                            if 'nearest_transport_type' in row and pd.notna(row['nearest_transport_type']):
                                st.write(f"Nearest Transport: {row['nearest_transport_type']} ({row.get('nearest_transport_name','')}) @ {row.get('nearest_transport_distance_km','?')} km")

                # Removed factor-based cluster recommendations from this side column (now shown below map)
        
        with tab5:
            st.subheader("What-If Scenario Modeling")
            st.caption("Experiment with different weight configurations and compare results")
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("### Create Scenario")
                
                scenario_name = st.text_input("Scenario Name", value=f"Scenario {len(st.session_state.saved_scenarios) + 1}")
                
                st.markdown("**Adjust Weights:**")
                s_demand = st.slider("Scenario Demand", 0.0, 1.0, 0.4, 0.05, key="s_demand")
                s_competition = st.slider("Scenario Competition", 0.0, 1.0, 0.3, 0.05, key="s_comp")
                s_accessibility = st.slider("Scenario Accessibility", 0.0, 1.0, 0.2, 0.05, key="s_access")
                s_infrastructure = st.slider("Scenario Infrastructure", 0.0, 1.0, 0.1, 0.05, key="s_infra")
                
                s_total = s_demand + s_competition + s_accessibility + s_infrastructure
                scenario_weights = {
                    'demand': s_demand / s_total,
                    'competition': s_competition / s_total,
                    'accessibility': s_accessibility / s_total,
                    'infrastructure': s_infrastructure / s_total
                }
                
                col_a, col_b = st.columns(2)
                with col_a:
                    if st.button("🔍 Preview Scenario", use_container_width=True):
                        with st.spinner("Calculating..."):
                            preview_results = run_quick_rescore(scenario_weights)
                            if preview_results is not None:
                                st.session_state['preview_results'] = preview_results
                
                with col_b:
                    if st.button("💾 Save Scenario", use_container_width=True):
                        preview = st.session_state.get('preview_results')
                        if preview is not None:
                            save_scenario(scenario_name, scenario_weights, preview)
                            st.success(f"Saved: {scenario_name}")
                        else:
                            current_top = st.session_state.top_locations
                            save_scenario(scenario_name, scenario_weights, current_top)
                            st.success(f"Saved: {scenario_name}")
                
                if 'preview_results' in st.session_state and st.session_state['preview_results'] is not None:
                    st.markdown("**Preview Results:**")
                    preview = st.session_state['preview_results']
                    for i, row in preview.head(5).iterrows():
                        st.write(f"#{int(row['rank'])} Score: {row['final_score']:.3f}")
            
            with col2:
                st.markdown("### Saved Scenarios")
                
                if not st.session_state.saved_scenarios:
                    st.info("No scenarios saved yet. Create and save scenarios to compare them.")
                else:
                    for i, scenario in enumerate(st.session_state.saved_scenarios):
                        with st.container():
                            st.markdown(f"""
                            <div class="scenario-card">
                                <b>{scenario['name']}</b><br/>
                                <small>{scenario['timestamp']}</small><br/>
                                Top Score: <b>{scenario['top_score']:.3f}</b><br/>
                                Weights: D:{scenario['weights']['demand']:.0%} 
                                C:{scenario['weights']['competition']:.0%}
                                A:{scenario['weights']['accessibility']:.0%}
                                I:{scenario['weights']['infrastructure']:.0%}
                            </div>
                            """, unsafe_allow_html=True)
                    
                    if len(st.session_state.saved_scenarios) >= 2:
                        st.markdown("### Scenario Comparison")
                        
                        comparison_data = []
                        for s in st.session_state.saved_scenarios:
                            comparison_data.append({
                                'Scenario': s['name'],
                                'Top Score': s['top_score'],
                                'Demand Wt': f"{s['weights']['demand']:.0%}",
                                'Competition Wt': f"{s['weights']['competition']:.0%}"
                            })
                        
                        st.dataframe(pd.DataFrame(comparison_data), hide_index=True)
                    
                    if st.button("🗑️ Clear All Scenarios"):
                        st.session_state.saved_scenarios = []
                        st.rerun()
        
        with tab6:
            st.subheader("Multi-City Comparison")
            st.caption("Compare analysis results across multiple cities for franchise expansion")
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("### Add City to Comparison")
                
                compare_city = st.text_input("City to Add", value="Mumbai", key="compare_city")
                compare_radius = st.slider("Radius (km)", 1.0, 20.0, 5.0, 0.5, key="compare_radius")
                
                if st.button("➕ Add City to Comparison", use_container_width=True):
                    if compare_city:
                        with st.spinner(f"Analyzing {compare_city}..."):
                            progress = st.progress(0)
                            compare_results = run_analysis_pipeline(
                                compare_city, 
                                results.get('business_type', 'cafe'),
                                compare_radius,
                                st.session_state.current_weights or weights,
                                progress
                            )
                            if compare_results:
                                st.session_state.multi_city_results[compare_city] = {
                                    'results': compare_results,
                                    'top_locations': compare_results.get('top_locations'),
                                    'radius': compare_radius
                                }
                                st.success(f"Added {compare_city} to comparison!")
                                st.rerun()
                
                current_city = st.session_state.current_city or results.get('city', 'Current')
                if current_city not in st.session_state.multi_city_results:
                    st.session_state.multi_city_results[current_city] = {
                        'results': results,
                        'top_locations': st.session_state.top_locations,
                        'radius': results.get('radius_km', 5.0)
                    }
            
            with col2:
                st.markdown("### City Comparison Results")
                
                if st.session_state.multi_city_results:
                    comparison_rows = []
                    for city_name, data in st.session_state.multi_city_results.items():
                        city_results = data['results']
                        city_top = data['top_locations']
                        
                        top_score = 0
                        if city_top is not None and len(city_top) > 0:
                            top_score = city_top['final_score'].max()
                        
                        comparison_rows.append({
                            'City': city_name,
                            'POIs': city_results.get('cleaned_poi_count', 0),
                            'Clusters': city_results.get('cluster_stats', {}).get('n_clusters', 0),
                            'Competitors': city_results.get('scoring_report', {}).get('total_competitors', 0),
                            'Top Score': f"{top_score:.3f}",
                            'Radius': f"{data['radius']} km"
                        })
                    
                    comparison_df = pd.DataFrame(comparison_rows)
                    st.dataframe(comparison_df, hide_index=True, use_container_width=True)

                    # Multi-City Comparison Chart
                    numeric_df = comparison_df.copy()
                    # Convert Top Score and Radius to numeric for plotting
                    numeric_df['Top Score'] = numeric_df['Top Score'].astype(float)
                    numeric_df['Radius'] = numeric_df['Radius'].str.replace(' km','', regex=False).astype(float)
                    melted = numeric_df.melt('City', var_name='Metric', value_name='Value')
                    # Exclude radius if not desired in main comparison (optional)
                    focus_metrics = melted[melted['Metric'].isin(['POIs','Clusters','Competitors','Top Score'])]
                    chart = (
                        alt.Chart(focus_metrics)
                        .mark_bar()
                        .encode(
                            x=alt.X('City:N', title='City'),
                            y=alt.Y('Value:Q', title='Value'),
                            color=alt.Color('Metric:N', title='Metric'),
                            tooltip=['City','Metric','Value']
                        )
                        .properties(title='Multi-City Metric Comparison', height=360)
                    )
                    st.altair_chart(chart, use_container_width=True)
                    
                    st.markdown("### Best City Recommendation")
                    if comparison_rows:
                        best_city = max(comparison_rows, key=lambda x: float(x['Top Score']))
                        st.success(f"🏆 **{best_city['City']}** has the highest opportunity score ({best_city['Top Score']})")
                    
                    if st.button("🗑️ Clear Comparison"):
                        st.session_state.multi_city_results = {}
                        st.rerun()
                else:
                    st.info("Add cities above to compare locations across multiple cities.")
        
        with tab7:
            st.subheader("Data Files & Reports")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Download Data Files")
                
                if os.path.exists('data/cleaned_data.csv'):
                    with open('data/cleaned_data.csv', 'r', encoding='utf-8') as f:
                        st.download_button(
                            "📥 Download Cleaned POIs",
                            f.read(),
                            file_name="cleaned_pois.csv",
                            mime="text/csv"
                        )
                
                if os.path.exists('data/clusters.csv'):
                    with open('data/clusters.csv', 'r', encoding='utf-8') as f:
                        st.download_button(
                            "📥 Download Clusters",
                            f.read(),
                            file_name="clusters.csv",
                            mime="text/csv"
                        )
                
                if os.path.exists('data/location_scores.csv'):
                    with open('data/location_scores.csv', 'r', encoding='utf-8') as f:
                        st.download_button(
                            "📥 Download Location Scores",
                            f.read(),
                            file_name="location_scores.csv",
                            mime="text/csv"
                        )
                
                st.markdown("### Generate PDF Report")
                
                if st.button("📄 Generate PDF Report", type="primary", use_container_width=True):
                    with st.spinner("Generating PDF report..."):
                        try:
                            report_path = generate_pdf_report(
                                results.get('city', st.session_state.current_city or 'Unknown'),
                                results.get('business_type', 'cafe'),
                                results.get('radius_km', 5.0),
                                results,
                                st.session_state.current_weights or weights,
                                st.session_state.top_locations
                            )
                            st.success("PDF report generated!")
                            
                            if os.path.exists(report_path):
                                with open(report_path, 'rb') as f:
                                    st.download_button(
                                        "📥 Download PDF Report",
                                        f.read(),
                                        file_name="business_location_analysis.pdf",
                                        mime="application/pdf"
                                    )
                        except Exception as e:
                            st.error(f"Error generating report: {e}")
            
            with col2:
                st.markdown("### Analysis Summary")
                
                scoring_report = results.get('scoring_report', {})
                if scoring_report:
                    st.write(f"Candidates Analyzed: {scoring_report.get('total_candidates', 0)}")
                    st.write(f"Competitors Found: {scoring_report.get('total_competitors', 0)}")
                    st.write(f"Supporting POIs: {scoring_report.get('total_supporting_pois', 0)}")
                    st.write(f"Top Score: {scoring_report.get('top_score', 0):.3f}")
                
                st.markdown("**Weights Used:**")
                current_weights = st.session_state.current_weights or weights
                for key, value in current_weights.items():
                    st.write(f"  {key.title()}: {value:.2f}")
            
            st.divider()
            
            st.markdown("### Location Scores Table")
            if os.path.exists('data/location_scores.csv'):
                scores_df = pd.read_csv('data/location_scores.csv')
                st.dataframe(
                    scores_df.head(20),
                    use_container_width=True,
                    hide_index=True
                )
            
            # ========== NEW ADVANCED FEATURES SECTION ==========
            st.divider()
            st.markdown("## 🚀 Advanced Features")
            
            # Create tabs for advanced features
            adv_tab1, adv_tab2, adv_tab3, adv_tab4 = st.tabs(
                ["Opportunity Scoring", "KDE Heatmap", "Agent Simulation", "Counterfactual Analysis"]
            )
            
            # ===== Tab 1: Opportunity Scoring =====
            with adv_tab1:
                st.markdown("### Weighted Opportunity Score Engine")
                st.write("Compute composite opportunity scores with custom feature weights.")
                
                col_opp1, col_opp2, col_opp3 = st.columns(3)
                with col_opp1:
                    footfall_w = st.slider("Footfall Weight", 0.0, 1.0, 0.25, key="opp_footfall")
                    income_w = st.slider("Income Weight", 0.0, 1.0, 0.20, key="opp_income")
                
                with col_opp2:
                    competition_w = st.slider("Competition Weight", 0.0, 1.0, -0.20, key="opp_comp")
                    transit_w = st.slider("Transit Weight", 0.0, 1.0, 0.15, key="opp_transit")
                
                with col_opp3:
                    safety_w = st.slider("Safety Weight", 0.0, 1.0, 0.10, key="opp_safety")
                    rent_w = st.slider("Rent Weight", 0.0, 1.0, -0.10, key="opp_rent")
                
                if st.button("Calculate Opportunity Score", key="calc_opp"):
                    features = {
                        "footfall": footfall_w * 100,
                        "income": income_w * 100,
                        "competition": competition_w * 100,
                        "transit": transit_w * 100,
                        "safety": safety_w * 100,
                        "rent": rent_w * 100
                    }
                    
                    weights = {
                        "footfall": footfall_w,
                        "income": income_w,
                        "competition": competition_w,
                        "transit": transit_w,
                        "safety": safety_w,
                        "rent": rent_w
                    }
                    
                    opp_score = compute_opportunity_score(features, weights)
                    
                    col_left, col_right = st.columns(2)
                    with col_left:
                        st.metric("Opportunity Score", f"{opp_score:.2f}/100")
                    
                    with col_right:
                        # Compute confidence
                        data_density = 0.8
                        noise_level = 0.2
                        confidence = compute_confidence(data_density, noise_level)
                        st.metric("Confidence Score", f"{confidence:.2f}/1.0")
            
            # ===== Tab 2: KDE Heatmap =====
            with adv_tab2:
                st.markdown("### Kernel Density Estimation Heatmap")
                st.write("Visualize POI density with smooth Gaussian smoothing.")
                
                if st.session_state.cleaned_pois is not None and len(st.session_state.cleaned_pois) > 0:
                    col_kde1, col_kde2 = st.columns(2)
                    
                    with col_kde1:
                        kde_bandwidth = st.slider("KDE Bandwidth", 0.001, 0.1, 0.01, key="kde_bw")
                        grid_size_kde = st.slider("Grid Size", 50, 300, 150, key="grid_kde")
                    
                    with col_kde2:
                        smooth_sigma = st.slider("Smoothing Sigma", 0.5, 10.0, 2.0, key="smooth_sigma")
                    
                    if st.button("Generate KDE Heatmap", key="gen_kde"):
                        with st.spinner("Computing KDE heatmap..."):
                            try:
                                points = st.session_state.cleaned_pois[['latitude', 'longitude']].values
                                xx, yy, zz = compute_kde(points, bandwidth=kde_bandwidth, grid_size=grid_size_kde)
                                
                                if xx is not None and yy is not None and zz is not None:
                                    zz_smooth = smooth_heatmap(zz, sigma=smooth_sigma)
                                    fig = plot_heatmap(xx, yy, zz_smooth)
                                    st.pyplot(fig)
                                    st.success("KDE heatmap generated successfully!")
                            except Exception as e:
                                st.error(f"Error generating heatmap: {e}")
                else:
                    st.info("Run analysis first to enable KDE heatmap generation.")
            
            # ===== Tab 3: Agent-Based Simulation =====
            with adv_tab3:
                st.markdown("### Agent-Based Simulation")
                st.write("Simulate customer attraction and business viability using multi-agent simulation.")
                
                col_sim1, col_sim2 = st.columns(2)
                
                with col_sim1:
                    pop_size = st.slider("Population Size", 10, 500, 100, step=10, key="sim_pop")
                    income_dist = st.selectbox("Income Distribution", 
                                             ["normal", "uniform", "skewed"], key="sim_income")
                
                with col_sim2:
                    sim_steps = st.slider("Simulation Steps", 10, 200, 100, step=10, key="sim_steps")
                
                if st.button("Run Simulation", key="run_sim"):
                    with st.spinner("Running simulation..."):
                        try:
                            agents = generate_agents(pop_size, income_dist)
                            business_types = ['cafe', 'restaurant', 'shop', 'supermarket', 'gym']
                            
                            if st.session_state.center:
                                location = st.session_state.center
                                sim_results = run_simulation(location, agents, business_types, sim_steps)
                                
                                # Display results
                                st.markdown("#### Simulation Results")
                                
                                col_sim_l, col_sim_r = st.columns(2)
                                with col_sim_l:
                                    st.metric("Average Satisfaction", 
                                            f"{sim_results.get('average_satisfaction', 0):.2f}")
                                    st.metric("Predicted Revenue", 
                                            f"${sim_results.get('predicted_revenue', 0):.2f}")
                                
                                with col_sim_r:
                                    st.metric("Total Attraction Score", 
                                            f"{sim_results.get('total_attraction', 0):.2f}")
                                
                                # Business scores
                                st.markdown("#### Business Type Viability")
                                biz_data = []
                                for biz, scores in sim_results.get('business_scores', {}).items():
                                    biz_data.append({
                                        'Business Type': biz.title(),
                                        'Attraction': scores.get('total_attraction', 0),
                                        'Satisfaction': scores.get('satisfaction_score', 0),
                                        'Viability': scores.get('viability', 'unknown')
                                    })
                                
                                if biz_data:
                                    biz_df = pd.DataFrame(biz_data)
                                    st.dataframe(biz_df, use_container_width=True, hide_index=True)
                                
                                st.info(analyze_simulation_results(sim_results))
                        except Exception as e:
                            st.error(f"Simulation error: {e}")
            
            # ===== Tab 4: Counterfactual Analysis =====
            with adv_tab4:
                st.markdown("### Counterfactual Impact Analysis")
                st.write("Predict the impact of opening a business at a specific location.")
                
                if st.session_state.top_locations is not None and len(st.session_state.top_locations) > 0:
                    selected_idx = st.slider("Select Location Rank", 1, 
                                            min(10, len(st.session_state.top_locations)), 
                                            1, key="cf_idx")
                    
                    location_row = st.session_state.top_locations.iloc[selected_idx - 1]
                    
                    st.markdown(f"**Selected Location: #{selected_idx}**")
                    col_cf1, col_cf2, col_cf3 = st.columns(3)
                    with col_cf1:
                        st.metric("Latitude", f"{location_row['latitude']:.4f}")
                    with col_cf2:
                        st.metric("Longitude", f"{location_row['longitude']:.4f}")
                    with col_cf3:
                        st.metric("Location Score", f"{location_row['final_score']:.3f}")
                    
                    st.divider()
                    
                    col_cf_l, col_cf_r = st.columns(2)
                    with col_cf_l:
                        footfall_cf = st.number_input("Footfall Level", 0.0, 100.0, 50.0, key="cf_footfall")
                        competition_cf = st.number_input("Competition Level", 0.0, 100.0, 30.0, key="cf_comp")
                    
                    with col_cf_r:
                        transit_cf = st.number_input("Transit Accessibility", 0.0, 100.0, 70.0, key="cf_transit")
                    
                    if st.button("Calculate Counterfactual Impact", key="calc_cf"):
                        from modules.business_suggestion import predict_counterfactual_effect
                        
                        features = {
                            "footfall": footfall_cf,
                            "competition": competition_cf,
                            "transit": transit_cf
                        }
                        
                        impact = predict_counterfactual_effect(
                            (location_row['latitude'], location_row['longitude']),
                            st.session_state.current_business or 'cafe',
                            features
                        )
                        
                        st.metric("Predicted Impact on Market", f"{impact:+.3f}")
                        
                        if impact > 0:
                            st.success(f"✅ Positive impact: Opening here would increase local market value by {impact:.3f}")
                        else:
                            st.warning(f"⚠️ Negative impact: Opening here would decrease market value by {abs(impact):.3f}")
                else:
                    st.info("Run analysis first to enable counterfactual analysis.")
    
    else:
        st.info("👈 Configure your analysis parameters in the sidebar and click 'Run Analysis' to get started.")
        
        st.markdown("### How It Works")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            **1. Data Collection**
            - Fetches POI data from OpenStreetMap
            - Includes shops, restaurants, banks, hospitals, schools, etc.
            - Configurable search radius
            """)
        
        with col2:
            st.markdown("""
            **2. Analysis**
            - DBSCAN clustering to find hotspots
            - Density and competition heatmaps
            - Multi-factor location scoring
            """)
        
        with col3:
            st.markdown("""
            **3. Recommendations**
            - Ranked list of best locations
            - Interactive map visualization
            - PDF reports & scenario modeling
            """)
        
        st.markdown("---")
        
        st.markdown("### New Features")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            **🎯 Scenario Modeling**
            - Compare different weight configurations
            - Save and compare scenarios
            - What-if analysis
            """)
        
        with col2:
            st.markdown("""
            **🌍 Multi-City Comparison**
            - Analyze multiple cities
            - Compare opportunities
            - Franchise expansion planning
            """)
        
        with col3:
            st.markdown("""
            **📄 PDF Reports**
            - Executive summary
            - Methodology documentation
            - Top recommendations
            """)
        
        st.markdown("---")
        st.markdown("*Powered by OSMnx, Folium, and Scikit-learn*")


if __name__ == "__main__":
    main()
