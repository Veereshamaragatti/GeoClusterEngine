"""
Clustering Module
Implements geospatial clustering algorithms for POI analysis
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.neighbors import KernelDensity
from scipy.ndimage import gaussian_filter
from collections import Counter


class GeoClusterer:
    """
    Implements clustering algorithms for geospatial POI data:
    - DBSCAN with haversine distance metric
    - K-Means for comparison
    - Hotspot and sparse region detection
    """
    
    EARTH_RADIUS_KM = 6371.0
    
    def __init__(self):
        """Initialize the GeoClusterer."""
        self.labels = None
        self.cluster_stats = {}
        self.hotspots = []
        self.sparse_regions = []
    
    def haversine_distance(self, point1: np.ndarray, 
                           point2: np.ndarray) -> float:
        """
        Calculate haversine distance between two points.
        
        Args:
            point1: [lat, lon] array
            point2: [lat, lon] array
            
        Returns:
            Distance in kilometers
        """
        lat1, lon1 = np.radians(point1)
        lat2, lon2 = np.radians(point2)
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        
        return self.EARTH_RADIUS_KM * c
    
    def dbscan_clustering(self, coords: np.ndarray, 
                          eps_km: float = 0.5,
                          min_samples: int = 5) -> np.ndarray:
        """
        Perform DBSCAN clustering with haversine distance.
        
        Args:
            coords: Numpy array of (lat, lon) coordinates
            eps_km: Maximum distance between points in a cluster (km)
            min_samples: Minimum points to form a cluster
            
        Returns:
            Array of cluster labels
        """
        coords_rad = np.radians(coords)
        
        eps_rad = eps_km / self.EARTH_RADIUS_KM
        
        dbscan = DBSCAN(
            eps=eps_rad,
            min_samples=min_samples,
            metric='haversine',
            algorithm='ball_tree'
        )
        
        self.labels = dbscan.fit_predict(coords_rad)
        
        self._compute_cluster_stats(coords)
        
        return self.labels
    
    def kmeans_clustering(self, coords: np.ndarray, 
                          n_clusters: int = 10) -> np.ndarray:
        """
        Perform K-Means clustering for comparison.
        
        Args:
            coords: Numpy array of (lat, lon) coordinates
            n_clusters: Number of clusters
            
        Returns:
            Array of cluster labels
        """
        scaler = StandardScaler()
        coords_scaled = scaler.fit_transform(coords)
        
        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=42,
            n_init=10
        )
        
        labels = kmeans.fit_predict(coords_scaled)
        
        centers_scaled = kmeans.cluster_centers_
        centers = scaler.inverse_transform(centers_scaled)
        
        self.kmeans_centers = centers
        
        return labels
    
    def find_optimal_clusters(self, coords: np.ndarray,
                              max_clusters: int = 15) -> int:
        """
        Find optimal number of clusters using silhouette score.
        
        Args:
            coords: Coordinate array
            max_clusters: Maximum clusters to try
            
        Returns:
            Optimal number of clusters
        """
        if len(coords) < 3:
            return 1
            
        scaler = StandardScaler()
        coords_scaled = scaler.fit_transform(coords)
        
        scores = []
        k_range = range(2, min(max_clusters, len(coords)))
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(coords_scaled)
            
            if len(set(labels)) > 1:
                score = silhouette_score(coords_scaled, labels)
                scores.append((k, score))
        
        if scores:
            optimal_k = max(scores, key=lambda x: x[1])[0]
            return optimal_k
        
        return 5
    
    def _compute_cluster_stats(self, coords: np.ndarray) -> dict:
        """
        Compute statistics for each cluster.
        
        Args:
            coords: Coordinate array
            
        Returns:
            Dictionary of cluster statistics
        """
        if self.labels is None:
            return {}
            
        unique_labels = set(self.labels)
        
        stats = {
            'n_clusters': len([l for l in unique_labels if l >= 0]),
            'n_noise': (self.labels == -1).sum(),
            'clusters': {}
        }
        
        for label in unique_labels:
            if label == -1:
                continue
                
            mask = self.labels == label
            cluster_coords = coords[mask]
            
            center_lat = cluster_coords[:, 0].mean()
            center_lon = cluster_coords[:, 1].mean()
            
            if len(cluster_coords) > 1:
                distances = []
                for point in cluster_coords:
                    d = self.haversine_distance(
                        point, 
                        np.array([center_lat, center_lon])
                    )
                    distances.append(d)
                avg_radius = np.mean(distances)
            else:
                avg_radius = 0
            
            stats['clusters'][label] = {
                'size': int(mask.sum()),
                'center_lat': float(center_lat),
                'center_lon': float(center_lon),
                'avg_radius_km': float(avg_radius),
                'density': float(mask.sum() / max(avg_radius, 0.1))
            }
        
        self.cluster_stats = stats
        return stats
    
    def detect_hotspots(self, coords: np.ndarray,
                        density_threshold: float = 0.75) -> list:
        """
        Detect high-density hotspot areas.
        
        Args:
            coords: Coordinate array
            density_threshold: Percentile threshold for hotspots
            
        Returns:
            List of hotspot cluster IDs
        """
        if not self.cluster_stats or 'clusters' not in self.cluster_stats:
            return []
            
        densities = [
            (label, data['density'])
            for label, data in self.cluster_stats['clusters'].items()
        ]
        
        if not densities:
            return []
            
        density_values = [d[1] for d in densities]
        threshold = np.percentile(density_values, density_threshold * 100)
        
        self.hotspots = [
            {
                'cluster_id': label,
                'density': density,
                'center_lat': self.cluster_stats['clusters'][label]['center_lat'],
                'center_lon': self.cluster_stats['clusters'][label]['center_lon'],
                'size': self.cluster_stats['clusters'][label]['size']
            }
            for label, density in densities
            if density >= threshold
        ]
        
        return self.hotspots
    
    def detect_sparse_regions(self, coords: np.ndarray,
                              center_lat: float,
                              center_lon: float,
                              grid_size: int = 10) -> list:
        """
        Detect sparse/underserved regions in the area.
        
        Args:
            coords: Coordinate array (lat, lon)
            center_lat, center_lon: Center of the search area
            grid_size: Number of grid divisions
            
        Returns:
            List of sparse region coordinates
        """
        if len(coords) == 0:
            return []
        
        if len(coords) < 2:
            return []
            
        lat_range = coords[:, 0].max() - coords[:, 0].min()
        lon_range = coords[:, 1].max() - coords[:, 1].min()
        
        if lat_range < 1e-6 or lon_range < 1e-6:
            return []
        
        lat_step = max(lat_range / grid_size, 1e-10)
        lon_step = max(lon_range / grid_size, 1e-10)
        
        lat_min = coords[:, 0].min()
        lon_min = coords[:, 1].min()
        
        grid_counts = np.zeros((grid_size, grid_size))
        
        for lat, lon in coords:
            lat_idx = min(int((lat - lat_min) / max(lat_step, 1e-10)), grid_size - 1)
            lon_idx = min(int((lon - lon_min) / max(lon_step, 1e-10)), grid_size - 1)
            lat_idx = max(0, lat_idx)
            lon_idx = max(0, lon_idx)
            grid_counts[lat_idx, lon_idx] += 1
        
        threshold = np.percentile(grid_counts, 25)
        
        sparse_regions = []
        for i in range(grid_size):
            for j in range(grid_size):
                if grid_counts[i, j] <= threshold:
                    region_lat = lat_min + (i + 0.5) * lat_step
                    region_lon = lon_min + (j + 0.5) * lon_step
                    sparse_regions.append({
                        'latitude': float(region_lat),
                        'longitude': float(region_lon),
                        'poi_count': int(grid_counts[i, j]),
                        'grid_cell': (i, j)
                    })
        
        self.sparse_regions = sparse_regions
        return sparse_regions
    
    def add_cluster_labels(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Add cluster labels to GeoDataFrame.
        
        Args:
            gdf: Input GeoDataFrame
            
        Returns:
            GeoDataFrame with cluster_label column
        """
        if self.labels is None:
            raise ValueError("Run clustering first before adding labels")
            
        if len(self.labels) != len(gdf):
            raise ValueError("Label count doesn't match GeoDataFrame length")
            
        gdf = gdf.copy()
        gdf['cluster_label'] = self.labels
        
        is_hotspot = np.zeros(len(gdf), dtype=bool)
        hotspot_ids = [h['cluster_id'] for h in self.hotspots]
        for i, label in enumerate(self.labels):
            if label in hotspot_ids:
                is_hotspot[i] = True
        gdf['is_hotspot'] = is_hotspot
        
        return gdf
    
    def get_cluster_summary(self) -> str:
        """
        Get a text summary of clustering results.
        
        Returns:
            Summary string
        """
        if not self.cluster_stats:
            return "No clustering performed yet"
            
        summary = []
        summary.append("=" * 50)
        summary.append("CLUSTERING ANALYSIS SUMMARY")
        summary.append("=" * 50)
        summary.append(f"Total clusters found: {self.cluster_stats['n_clusters']}")
        summary.append(f"Noise points: {self.cluster_stats['n_noise']}")
        summary.append("")
        
        if self.hotspots:
            summary.append(f"Hotspot regions identified: {len(self.hotspots)}")
            for i, hotspot in enumerate(self.hotspots[:5], 1):
                summary.append(
                    f"  {i}. Cluster {hotspot['cluster_id']}: "
                    f"{hotspot['size']} POIs, "
                    f"density: {hotspot['density']:.2f}"
                )
        
        if self.sparse_regions:
            summary.append(f"\nSparse regions identified: {len(self.sparse_regions)}")
        
        summary.append("=" * 50)
        
        return "\n".join(summary)
    
    def save_clusters(self, gdf: gpd.GeoDataFrame, 
                      filepath: str = 'data/clusters.csv') -> str:
        """
        Save clustered data to CSV.
        
        Args:
            gdf: GeoDataFrame with cluster labels
            filepath: Output file path
            
        Returns:
            Path to saved file
        """
        df = gdf.copy()
        
        if 'geometry' in df.columns:
            df = df.drop(columns=['geometry'])
        if 'point_geometry' in df.columns:
            df = df.drop(columns=['point_geometry'])
        
        cols_to_save = ['name', 'category', 'latitude', 'longitude', 
                        'cluster_label', 'is_hotspot']
        available = [c for c in cols_to_save if c in df.columns]
        
        df[available].to_csv(filepath, index=False)
        print(f"Saved clusters to {filepath}")
        
        return filepath


def compute_kde(points: np.ndarray, bandwidth: float = 0.01, grid_size: int = 200) -> tuple:
    """
    Compute Kernel Density Estimation (KDE) heatmap for point data.
    
    Args:
        points: Array of (lat, lon) coordinates
        bandwidth: KDE bandwidth parameter
        grid_size: Resolution of output grid
        
    Returns:
        Tuple of (xx grid, yy grid, z density values)
    """
    if len(points) == 0:
        return None, None, None
    
    kde = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
    kde.fit(points)
    
    x_min, y_min = points.min(axis=0)
    x_max, y_max = points.max(axis=0)
    
    x_grid = np.linspace(x_min, x_max, grid_size)
    y_grid = np.linspace(y_min, y_max, grid_size)
    xx, yy = np.meshgrid(x_grid, y_grid)
    
    sample_grid = np.vstack([xx.ravel(), yy.ravel()]).T
    z = kde.score_samples(sample_grid)
    
    return xx, yy, z.reshape(xx.shape)


def smooth_heatmap(z: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    """
    Apply Gaussian smoothing to heatmap data.
    
    Args:
        z: 2D array of density values
        sigma: Standard deviation for Gaussian kernel
        
    Returns:
        Smoothed 2D array
    """
    if z is None:
        return None
    
    return gaussian_filter(z, sigma=sigma)


def spatio_temporal_hawkes(events: list) -> dict:
    """
    Placeholder for Hawkes Process spatio-temporal modeling.
    
    Args:
        events: List of event records with (lat, lon, timestamp)
        
    Returns:
        Dictionary with predicted hotspots (placeholder)
    """
    # Placeholder implementation - safe, doesn't break existing code
    if not events:
        return {"predicted_hotspots": []}
    
    # Simple aggregation as placeholder
    hotspots = []
    try:
        if len(events) > 0:
            event_lats = [e.get('lat', 0) for e in events if isinstance(e, dict)]
            event_lons = [e.get('lon', 0) for e in events if isinstance(e, dict)]
            
            if event_lats and event_lons:
                hotspots.append({
                    'center_lat': np.mean(event_lats),
                    'center_lon': np.mean(event_lons),
                    'intensity': len(events)
                })
    except Exception:
        pass
    
    return {"predicted_hotspots": hotspots}


def compute_factor_cluster_recommendations(
    gdf: gpd.GeoDataFrame,
    business_type: str,
    competitor_radius_km: float = 0.3,
    supporting_radius_km: float = 0.4,
    neutral_radius_km: float = 0.3,
    n_clusters: int = None,
    random_state: int = 42
) -> dict:
    """Compute K-Means clusters using competitor/supporting/neutral density features.

    Builds feature vectors for each POI consisting of local densities of competitor,
    supporting, and neutral categories (within separate radii). Clusters these features
    with K-Means, then scores each cluster for business suitability.

    Cluster viability score formula (heuristic):
        score = 1.0 * supporting_mean - 0.7 * competitor_mean + 0.2 * neutral_mean
    Higher supporting and lower competitor densities yield better scores.

    Args:
        gdf: Cleaned POI GeoDataFrame with 'category','latitude','longitude'
        business_type: Target business type (used to define competitor/supporting sets)
        competitor_radius_km: Radius for competitor density
        supporting_radius_km: Radius for supporting density
        neutral_radius_km: Radius for neutral density
        n_clusters: If None, a heuristic chooses cluster count
        random_state: Random seed for K-Means

    Returns:
        dict with keys:
            'enriched_pois': GeoDataFrame with density features + cluster_id
            'cluster_summary': DataFrame of cluster metrics
            'top_clusters': DataFrame of top 3 cluster recommendations
    """
    result = {
        'enriched_pois': None,
        'cluster_summary': pd.DataFrame(),
        'top_clusters': pd.DataFrame()
    }
    if gdf is None or len(gdf) == 0 or 'category' not in gdf.columns:
        return result

    df = gdf.copy().reset_index(drop=True)
    # Basic category normalization
    df['category_norm'] = df['category'].astype(str).str.lower()
    target = str(business_type).lower().strip()

    # Supporting category map (simple heuristic list per common business types)
    SUPPORTING_MAP = {
        'restaurant': ['cafe','coffee_shop','bakery','supermarket','parking','bus_station','bank','atm'],
        'cafe': ['restaurant','coffee_shop','bakery','supermarket','parking','bus_station','bank','atm'],
        'bakery': ['cafe','coffee_shop','supermarket','restaurant','parking','bus_station'],
        'fast_food': ['restaurant','cafe','supermarket','parking','bus_station'],
        'shop': ['supermarket','parking','bus_station','bank','atm'],
        'supermarket': ['shop','parking','bus_station','atm','bank'],
        'pharmacy': ['hospital','clinic','supermarket','parking','bus_station'],
        'bank': ['atm','shop','supermarket','parking'],
        'gym': ['parking','bus_station','shop','supermarket','cafe'],
        'hotel': ['restaurant','cafe','parking','bus_station','atm','bank']
    }
    supporting_set = set(SUPPORTING_MAP.get(target, []))
    competitor_set = {target}

    # Classify categories
    def classify(cat: str) -> str:
        if cat in competitor_set:
            return 'competitor'
        if cat in supporting_set:
            return 'supporting'
        return 'neutral'

    df['factor_class'] = df['category_norm'].apply(classify)

    # Coordinate array (lat, lon)
    coords = df[['latitude','longitude']].to_numpy()
    if len(coords) == 0:
        return result

    # Build KDTree for approximate neighbor search (degree approximation)
    # 1 degree lat ~ 111 km; adjust lon by cos(mean_lat)
    mean_lat = np.mean(coords[:,0]) if len(coords) else 0.0
    lat_km_factor = 111.0
    lon_km_factor = 111.321 * np.cos(np.radians(mean_lat))

    # Convert radii km to degree radius (use min of lat/lon scaling for conservative search)
    def km_to_deg(km: float) -> float:
        return km / lat_km_factor

    comp_deg = km_to_deg(competitor_radius_km)
    supp_deg = km_to_deg(supporting_radius_km)
    neut_deg = km_to_deg(neutral_radius_km)

    from sklearn.neighbors import KDTree
    tree = KDTree(coords, metric='euclidean')  # degree-space approximation

    # Pre-sets for vectorized lookups
    categories = df['factor_class'].to_numpy()

    comp_counts = np.zeros(len(df), dtype=int)
    supp_counts = np.zeros(len(df), dtype=int)
    neut_counts = np.zeros(len(df), dtype=int)

    # Query neighbors and count category types per point
    comp_inds = tree.query_radius(coords, r=comp_deg)
    supp_inds = tree.query_radius(coords, r=supp_deg)
    neut_inds = tree.query_radius(coords, r=neut_deg)

    for i, inds in enumerate(comp_inds):
        if len(inds):
            comp_counts[i] = np.sum(categories[inds] == 'competitor') - (categories[i] == 'competitor')
    for i, inds in enumerate(supp_inds):
        if len(inds):
            supp_counts[i] = np.sum(categories[inds] == 'supporting') - (categories[i] == 'supporting')
    for i, inds in enumerate(neut_inds):
        if len(inds):
            neut_counts[i] = np.sum(categories[inds] == 'neutral') - (categories[i] == 'neutral')

    df['competitor_density'] = comp_counts.astype(float)
    df['supporting_density'] = supp_counts.astype(float)
    df['neutral_density'] = neut_counts.astype(float)

    # Feature matrix for clustering
    feature_cols = ['competitor_density','supporting_density','neutral_density']
    X = df[feature_cols].to_numpy()

    # Heuristic clusters if not provided
    if n_clusters is None:
        n_clusters = int(np.clip(np.sqrt(len(df)/50)+3, 4, 12))

    # Scale features
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    df['factor_cluster'] = km.fit_predict(Xs)

    # Aggregate cluster metrics
    agg = df.groupby('factor_cluster')[feature_cols].mean().rename(columns=lambda c: c+'_mean')
    sizes = df.groupby('factor_cluster').size().rename('size')

    cluster_centers = df.groupby('factor_cluster')[['latitude','longitude']].mean()
    summary = pd.concat([agg, sizes, cluster_centers], axis=1).reset_index().rename(columns={'factor_cluster':'cluster_id'})

    # Score clusters
    summary['cluster_score'] = (
        1.0 * summary['supporting_density_mean']
        - 0.7 * summary['competitor_density_mean']
        + 0.2 * summary['neutral_density_mean']
    )

    # Rank clusters
    summary = summary.sort_values('cluster_score', ascending=False).reset_index(drop=True)
    summary['rank'] = summary.index + 1

    top_clusters = summary.head(3)[['rank','cluster_id','latitude','longitude','supporting_density_mean','competitor_density_mean','neutral_density_mean','cluster_score']]

    result['enriched_pois'] = df
    result['cluster_summary'] = summary
    result['top_clusters'] = top_clusters
    return result
