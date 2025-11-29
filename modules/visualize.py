"""
Visualization Module
Creates interactive maps using Folium for geospatial visualization
"""

import os
import numpy as np
import pandas as pd
import geopandas as gpd
import folium
from folium import plugins
from folium.plugins import MarkerCluster, HeatMap
from scipy import stats
import matplotlib.pyplot as plt


class MapVisualizer:
    """
    Creates various interactive maps for POI visualization:
    - All POIs map with marker clustering
    - Cluster visualization map
    - Density heatmaps
    - Competition analysis maps
    - Recommendation maps
    """
    
    CATEGORY_COLORS = {
        'restaurant': 'red',
        'cafe': 'orange',
        'coffee_shop': 'orange',
        'shop': 'blue',
        'retail': 'blue',
        'supermarket': 'purple',
        'bank': 'green',
        'finance': 'green',
        'hospital': 'pink',
        'healthcare': 'pink',
        'pharmacy': 'lightred',
        'school': 'cadetblue',
        'education': 'cadetblue',
        'hotel': 'darkblue',
        'hospitality': 'darkblue',
        'gym': 'darkgreen',
        'fuel': 'gray',
        'parking': 'lightgray',
        'default': 'gray'
    }
    
    CATEGORY_ICONS = {
        'restaurant': 'cutlery',
        'cafe': 'coffee',
        'coffee_shop': 'coffee',
        'shop': 'shopping-cart',
        'retail': 'shopping-cart',
        'supermarket': 'shopping-basket',
        'bank': 'university',
        'finance': 'university',
        'hospital': 'plus-sign',
        'healthcare': 'plus-sign',
        'pharmacy': 'plus',
        'school': 'education',
        'education': 'education',
        'hotel': 'home',
        'hospitality': 'home',
        'gym': 'heart',
        'default': 'map-marker'
    }
    
    def __init__(self, center_lat: float, center_lon: float, 
                 zoom_start: int = 13):
        """
        Initialize the MapVisualizer.
        
        Args:
            center_lat: Center latitude
            center_lon: Center longitude
            zoom_start: Initial zoom level
        """
        self.center_lat = center_lat
        self.center_lon = center_lon
        self.zoom_start = zoom_start
        self.maps_created = []
    
    def create_base_map(self, tiles: str = 'OpenStreetMap') -> folium.Map:
        """
        Create a base Folium map.
        
        Args:
            tiles: Tile layer to use
            
        Returns:
            Folium Map object
        """
        m = folium.Map(
            location=[self.center_lat, self.center_lon],
            zoom_start=self.zoom_start,
            tiles=tiles
        )
        
        folium.TileLayer('cartodbpositron', name='Light Mode').add_to(m)
        folium.TileLayer('cartodbdark_matter', name='Dark Mode').add_to(m)
        
        return m
    
    def create_all_pois_map(self, gdf: gpd.GeoDataFrame,
                            use_clustering: bool = True) -> folium.Map:
        """
        Create a map showing all POIs with marker clustering.
        
        Args:
            gdf: GeoDataFrame with POI data
            use_clustering: Whether to use marker clustering
            
        Returns:
            Folium Map object
        """
        m = self.create_base_map()
        
        if use_clustering:
            marker_cluster = MarkerCluster(name='All POIs')
        
        for idx, row in gdf.iterrows():
            if pd.isna(row.get('latitude')) or pd.isna(row.get('longitude')):
                continue
                
            category = str(row.get('category', 'default')).lower()
            color = self.CATEGORY_COLORS.get(category, self.CATEGORY_COLORS['default'])
            icon = self.CATEGORY_ICONS.get(category, self.CATEGORY_ICONS['default'])
            
            name = row.get('name', 'Unknown')
            popup_html = f"""
            <div style="min-width: 150px">
                <b>{name}</b><br>
                Category: {category}<br>
                Lat: {row['latitude']:.5f}<br>
                Lon: {row['longitude']:.5f}
            </div>
            """
            
            marker = folium.Marker(
                location=[row['latitude'], row['longitude']],
                popup=folium.Popup(popup_html, max_width=300),
                icon=folium.Icon(color=color, icon=icon, prefix='glyphicon'),
                tooltip=name
            )
            
            if use_clustering:
                marker.add_to(marker_cluster)
            else:
                marker.add_to(m)
        
        if use_clustering:
            marker_cluster.add_to(m)
        
        self._add_legend(m, gdf)
        folium.LayerControl().add_to(m)
        
        return m
    
    def create_cluster_map(self, gdf: gpd.GeoDataFrame,
                           cluster_stats: dict = None) -> folium.Map:
        """
        Create a map visualizing cluster assignments.
        
        Args:
            gdf: GeoDataFrame with cluster_label column
            cluster_stats: Optional cluster statistics
            
        Returns:
            Folium Map object
        """
        m = self.create_base_map()
        
        if 'cluster_label' not in gdf.columns:
            return self.create_all_pois_map(gdf)
        
        unique_clusters = gdf['cluster_label'].unique()
        colors = self._generate_cluster_colors(len(unique_clusters))
        cluster_color_map = dict(zip(sorted(unique_clusters), colors))
        
        for cluster_id in unique_clusters:
            cluster_data = gdf[gdf['cluster_label'] == cluster_id]
            
            if cluster_id == -1:
                layer_name = 'Noise Points'
                color = 'gray'
            else:
                layer_name = f'Cluster {cluster_id}'
                color = cluster_color_map[cluster_id]
            
            feature_group = folium.FeatureGroup(name=layer_name)
            
            for idx, row in cluster_data.iterrows():
                if pd.isna(row.get('latitude')) or pd.isna(row.get('longitude')):
                    continue
                    
                name = row.get('name', 'Unknown')
                popup_html = f"""
                <div>
                    <b>{name}</b><br>
                    Cluster: {cluster_id}<br>
                    Category: {row.get('category', 'N/A')}
                </div>
                """
                
                folium.CircleMarker(
                    location=[row['latitude'], row['longitude']],
                    radius=8,
                    color=color,
                    fill=True,
                    fillColor=color,
                    fillOpacity=0.7,
                    popup=folium.Popup(popup_html, max_width=200),
                    tooltip=f"Cluster {cluster_id}: {name}"
                ).add_to(feature_group)
            
            feature_group.add_to(m)
        
        if cluster_stats and 'clusters' in cluster_stats:
            for label, data in cluster_stats['clusters'].items():
                folium.Marker(
                    location=[data['center_lat'], data['center_lon']],
                    icon=folium.DivIcon(
                        html=f'<div style="font-size: 12px; color: black; '
                             f'background-color: white; padding: 3px; '
                             f'border-radius: 5px; border: 1px solid black;">'
                             f'C{label}: {data["size"]}</div>',
                        icon_size=(60, 20),
                        icon_anchor=(30, 10)
                    )
                ).add_to(m)
        
        folium.LayerControl().add_to(m)
        
        return m
    
    def create_heatmap(self, gdf: gpd.GeoDataFrame,
                       intensity_column: str = None,
                       radius: int = 15,
                       blur: int = 10) -> folium.Map:
        """
        Create a density heatmap.
        
        Args:
            gdf: GeoDataFrame with POI data
            intensity_column: Column to use for intensity weights
            radius: Heatmap point radius
            blur: Heatmap blur amount
            
        Returns:
            Folium Map object
        """
        m = self.create_base_map()
        
        heat_data = []
        for idx, row in gdf.iterrows():
            if pd.isna(row.get('latitude')) or pd.isna(row.get('longitude')):
                continue
                
            if intensity_column and intensity_column in row:
                weight = row[intensity_column]
                if pd.isna(weight):
                    weight = 1
            else:
                weight = 1
                
            heat_data.append([row['latitude'], row['longitude'], weight])
        
        if heat_data:
            HeatMap(
                heat_data,
                name='POI Density',
                radius=radius,
                blur=blur,
                max_zoom=18,
                gradient={0.2: 'blue', 0.4: 'lime', 0.6: 'yellow', 1: 'red'}
            ).add_to(m)
        
        folium.LayerControl().add_to(m)
        
        return m
    
    def create_competition_heatmap(self, gdf: gpd.GeoDataFrame,
                                   business_type: str) -> folium.Map:
        """
        Create a heatmap showing competition density.
        
        Args:
            gdf: GeoDataFrame with POI data
            business_type: Type of business to analyze
            
        Returns:
            Folium Map object
        """
        m = self.create_base_map()
        
        competitors = gdf[gdf['category'].str.lower() == business_type.lower()]
        
        heat_data = []
        for idx, row in competitors.iterrows():
            if pd.isna(row.get('latitude')) or pd.isna(row.get('longitude')):
                continue
            heat_data.append([row['latitude'], row['longitude'], 1])
        
        if heat_data:
            HeatMap(
                heat_data,
                name=f'{business_type} Competition',
                radius=20,
                blur=15,
                max_zoom=18,
                gradient={0.2: 'green', 0.5: 'yellow', 0.8: 'orange', 1: 'red'}
            ).add_to(m)
        
        for idx, row in competitors.iterrows():
            if pd.isna(row.get('latitude')) or pd.isna(row.get('longitude')):
                continue
                
            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=5,
                color='red',
                fill=True,
                fillColor='red',
                fillOpacity=0.5,
                tooltip=row.get('name', 'Competitor')
            ).add_to(m)
        
        folium.LayerControl().add_to(m)
        
        return m
    
    def create_recommendations_map(self, gdf: gpd.GeoDataFrame,
                                   recommendations: pd.DataFrame,
                                   business_type: str) -> folium.Map:
        """
        Create a map showing top recommended locations.
        
        Args:
            gdf: GeoDataFrame with POI data
            recommendations: DataFrame with scored locations
            business_type: Type of business
            
        Returns:
            Folium Map object
        """
        m = self.create_base_map()
        
        all_pois = folium.FeatureGroup(name='All POIs', show=False)
        for idx, row in gdf.iterrows():
            if pd.isna(row.get('latitude')) or pd.isna(row.get('longitude')):
                continue
            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=3,
                color='gray',
                fill=True,
                fillColor='gray',
                fillOpacity=0.3
            ).add_to(all_pois)
        all_pois.add_to(m)
        
        competitors = folium.FeatureGroup(name='Competitors')
        competitor_data = gdf[gdf['category'].str.lower() == business_type.lower()]
        for idx, row in competitor_data.iterrows():
            if pd.isna(row.get('latitude')) or pd.isna(row.get('longitude')):
                continue
            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=6,
                color='red',
                fill=True,
                fillColor='red',
                fillOpacity=0.5,
                tooltip=f"Competitor: {row.get('name', 'Unknown')}"
            ).add_to(competitors)
        competitors.add_to(m)
        
        top_locations = folium.FeatureGroup(name='Recommended Locations')
        for rank, (idx, row) in enumerate(recommendations.head(10).iterrows(), 1):
            if pd.isna(row.get('latitude')) or pd.isna(row.get('longitude')):
                continue
                
            popup_html = f"""
            <div style="min-width: 200px">
                <h4>Rank #{rank}</h4>
                <b>Final Score: {row.get('final_score', 0):.2f}</b><br>
                <hr>
                Demand Score: {row.get('demand_score', 0):.2f}<br>
                Competition Score: {row.get('competition_score', 0):.2f}<br>
                Accessibility Score: {row.get('accessibility_score', 0):.2f}<br>
                Infrastructure Score: {row.get('infrastructure_score', 0):.2f}<br>
                <hr>
                Lat: {row['latitude']:.5f}<br>
                Lon: {row['longitude']:.5f}
            </div>
            """
            
            icon_html = f"""
            <div style="
                background-color: {'gold' if rank <= 3 else 'green'};
                border: 2px solid {'#FF6B00' if rank <= 3 else 'darkgreen'};
                border-radius: 50%;
                width: 30px;
                height: 30px;
                display: flex;
                align-items: center;
                justify-content: center;
                font-weight: bold;
                color: {'black' if rank <= 3 else 'white'};
                font-size: 14px;
            ">{rank}</div>
            """
            
            folium.Marker(
                location=[row['latitude'], row['longitude']],
                popup=folium.Popup(popup_html, max_width=300),
                icon=folium.DivIcon(
                    html=icon_html,
                    icon_size=(30, 30),
                    icon_anchor=(15, 15)
                ),
                tooltip=f"Rank #{rank} - Score: {row.get('final_score', 0):.2f}"
            ).add_to(top_locations)
        top_locations.add_to(m)
        
        folium.LayerControl().add_to(m)
        
        return m
    
    def create_kde_heatmap(self, coords: np.ndarray,
                           grid_size: int = 100) -> folium.Map:
        """
        Create a Kernel Density Estimation heatmap.
        
        Args:
            coords: Array of (lat, lon) coordinates
            grid_size: Resolution of the KDE grid
            
        Returns:
            Folium Map object
        """
        m = self.create_base_map()
        
        if len(coords) < 2:
            return m
        
        lat_min, lat_max = coords[:, 0].min(), coords[:, 0].max()
        lon_min, lon_max = coords[:, 1].min(), coords[:, 1].max()
        
        lat_grid = np.linspace(lat_min, lat_max, grid_size)
        lon_grid = np.linspace(lon_min, lon_max, grid_size)
        
        try:
            kde = stats.gaussian_kde(coords.T)
            
            heat_data = []
            for lat in lat_grid[::5]:
                for lon in lon_grid[::5]:
                    density = kde([lat, lon])[0]
                    heat_data.append([lat, lon, float(density)])
            
            if heat_data:
                HeatMap(
                    heat_data,
                    name='KDE Density',
                    radius=15,
                    blur=10,
                    max_zoom=18
                ).add_to(m)
        except Exception as e:
            print(f"KDE computation failed: {e}")
            heat_data = [[lat, lon, 1] for lat, lon in coords]
            HeatMap(heat_data, name='Density', radius=15, blur=10).add_to(m)
        
        folium.LayerControl().add_to(m)
        
        return m
    
    def _add_legend(self, m: folium.Map, gdf: gpd.GeoDataFrame) -> None:
        """
        Add a legend to the map.
        
        Args:
            m: Folium Map object
            gdf: GeoDataFrame with category data
        """
        categories = gdf['category'].unique() if 'category' in gdf.columns else []
        
        legend_html = '''
        <div style="position: fixed; 
                    bottom: 50px; right: 50px; 
                    border: 2px solid grey; 
                    z-index: 1000; 
                    background-color: white;
                    padding: 10px;
                    font-size: 12px;
                    border-radius: 5px;">
        <b>Categories</b><br>
        '''
        
        for cat in categories[:10]:
            color = self.CATEGORY_COLORS.get(str(cat).lower(), 'gray')
            legend_html += f'<i style="background: {color}; width: 15px; height: 15px; display: inline-block; margin-right: 5px;"></i> {cat}<br>'
        
        legend_html += '</div>'
        
        m.get_root().html.add_child(folium.Element(legend_html))
    
    def _generate_cluster_colors(self, n_clusters: int) -> list:
        """
        Generate distinct colors for clusters.
        
        Args:
            n_clusters: Number of clusters
            
        Returns:
            List of color strings
        """
        base_colors = [
            'red', 'blue', 'green', 'purple', 'orange',
            'darkred', 'darkblue', 'darkgreen', 'cadetblue', 'pink',
            'lightred', 'lightblue', 'lightgreen', 'beige', 'gray'
        ]
        
        colors = []
        for i in range(n_clusters):
            colors.append(base_colors[i % len(base_colors)])
        
        return colors
    
    def save_map(self, fmap: folium.Map, filename: str) -> str:
        """Save the folium map to HTML file.

        Args:
            fmap: Folium Map object
            filename: Output filename

        Returns:
            Path to saved file or empty string on failure
        """
        path = os.path.join('maps', filename)
        try:
            fmap.save(path)
            self.maps_created.append(path)
            return path
        except Exception as e:
            print(f"Error saving map: {e}")
            return ""

    def save_static_scatter(self, gdf: gpd.GeoDataFrame, filename: str,
                             color_by: str = None, center_lat: float = None,
                             center_lon: float = None) -> str:
        """Create a static scatter PNG for download (no browser rendering).

        Args:
            gdf: GeoDataFrame with latitude/longitude
            filename: Output PNG name
            color_by: Optional column to color points
            center_lat/center_lon: Optional center marker

        Returns:
            File path to saved PNG or empty string if not generated
        """
        if 'latitude' not in gdf.columns or 'longitude' not in gdf.columns or len(gdf) == 0:
            return ""
        import matplotlib.pyplot as plt
        path = os.path.join('maps', filename)
        try:
            fig, ax = plt.subplots(figsize=(8, 6))
            if color_by and color_by in gdf.columns:
                groups = gdf[color_by].fillna('unknown')
                unique = groups.unique()
                cmap = plt.get_cmap('tab20')
                for i, u in enumerate(unique):
                    subset = gdf[groups == u]
                    ax.scatter(subset['longitude'], subset['latitude'], s=12,
                               color=cmap(i % 20), alpha=0.6, label=str(u))
                ax.legend(loc='upper right', fontsize='x-small', frameon=False)
            else:
                ax.scatter(gdf['longitude'], gdf['latitude'], s=12, alpha=0.6, color='#1f77b4')
            if center_lat is not None and center_lon is not None:
                ax.scatter([center_lon], [center_lat], c='red', s=60, marker='x')
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')
            ax.set_title('POI Distribution' if not color_by else f'POIs by {color_by}')
            plt.tight_layout()
            fig.savefig(path, dpi=150)
            plt.close(fig)
            return path
        except Exception as e:
            print(f"Failed to save static scatter: {e}")
            return ""

    def save_static_cluster_scatter(self, gdf: gpd.GeoDataFrame, filename: str,
                                    center_lat: float = None, center_lon: float = None) -> str:
        """Create a cluster annotated scatter plot.

        Adds cluster labels (id + count) at cluster centroid.

        Args:
            gdf: GeoDataFrame with 'cluster' column
            filename: Output PNG name
            center_lat/center_lon: Optional center marker
        Returns:
            File path or empty string
        """
        if 'cluster' not in gdf.columns or len(gdf) == 0:
            return ""
        import matplotlib.pyplot as plt
        path = os.path.join('maps', filename)
        try:
            fig, ax = plt.subplots(figsize=(8, 6))
            clusters = gdf['cluster'].fillna(-1).astype(int)
            unique = sorted(clusters.unique())
            cmap = plt.get_cmap('tab20')
            for i, cid in enumerate(unique):
                subset = gdf[clusters == cid]
                ax.scatter(subset['longitude'], subset['latitude'], s=14,
                           color=cmap(i % 20), alpha=0.65, label=f"C{cid} ({len(subset)})")
                # Annotate centroid
                cx = subset['longitude'].mean()
                cy = subset['latitude'].mean()
                ax.text(cx, cy, f"C{cid}", fontsize=8, ha='center', va='center',
                        color='black', bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.6))
            if center_lat is not None and center_lon is not None:
                ax.scatter([center_lon], [center_lat], c='red', s=70, marker='x')
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')
            ax.set_title('Cluster Distribution')
            ax.legend(loc='upper right', fontsize='x-small', frameon=False)
            plt.tight_layout()
            fig.savefig(path, dpi=150)
            plt.close(fig)
            return path
        except Exception as e:
            print(f"Failed to save cluster scatter: {e}")
            return ""

    def save_static_heatmap(self, gdf: gpd.GeoDataFrame, filename: str,
                             center_lat: float = None, center_lon: float = None,
                             bins: int = 60, smooth: bool = True) -> str:
        """Create a density heatmap PNG using 2D histogram (optionally smoothed).

        Args:
            gdf: GeoDataFrame with latitude/longitude
            filename: Output PNG name
            center_lat/center_lon: Optional center marker
            bins: Resolution of histogram grid
            smooth: Apply Gaussian smoothing if possible
        Returns:
            File path or empty string
        """
        if 'latitude' not in gdf.columns or 'longitude' not in gdf.columns or len(gdf) == 0:
            return ""
        import matplotlib.pyplot as plt
        path = os.path.join('maps', filename)
        try:
            lats = gdf['latitude'].values
            lons = gdf['longitude'].values
            # Compute histogram
            H, xedges, yedges = np.histogram2d(lons, lats, bins=bins)
            # Optional smoothing
            if smooth:
                try:
                    from scipy.ndimage import gaussian_filter
                    H = gaussian_filter(H, sigma=1.2)
                except Exception:
                    pass
            # Plot
            fig, ax = plt.subplots(figsize=(8, 6))
            im = ax.imshow(H.T, origin='lower', cmap='YlOrRd',
                           extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                           aspect='auto')
            plt.colorbar(im, ax=ax, label='Density')
            if center_lat is not None and center_lon is not None:
                ax.scatter([center_lon], [center_lat], c='blue', s=70, marker='x')
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')
            ax.set_title('POI Density Heatmap')
            plt.tight_layout()
            fig.savefig(path, dpi=150)
            plt.close(fig)
            return path
        except Exception as e:
            print(f"Failed to save heatmap: {e}")
            return ""
    
    def get_map_html(self, m: folium.Map) -> str:
        """
        Get HTML representation of the map.
        
        Args:
            m: Folium Map object
            
        Returns:
            HTML string
        """
        return m._repr_html_()


def visualize_timelapse(kde_frames: list) -> list:
    """
    Generate timelapse animation frames from KDE density heatmaps.
    
    Args:
        kde_frames: List of KDE density arrays over time
        
    Returns:
        List of image arrays for UI display
    """
    if not kde_frames:
        return []
    
    image_list = []
    
    for frame in kde_frames:
        if frame is None:
            continue
        
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.contourf(frame, levels=15, cmap='YlOrRd')
        ax.set_title("Density Heatmap - Time Series")
        plt.colorbar(im, ax=ax)
        
        # Convert to image array (placeholder)
        image_list.append(fig)
        plt.close(fig)
    
    return image_list


def plot_heatmap(xx: np.ndarray, yy: np.ndarray, zz: np.ndarray) -> plt.Figure:
    """
    Plot a 2D heatmap from KDE data.
    
    Args:
        xx: X-axis grid
        yy: Y-axis grid
        zz: Density values
        
    Returns:
        Matplotlib Figure object
    """
    if xx is None or yy is None or zz is None:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No data available for heatmap")
        return fig
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create contourf plot
    contour = ax.contourf(xx, yy, zz, levels=20, cmap='viridis')
    
    # Add contour lines
    ax.contour(xx, yy, zz, levels=10, colors='black', alpha=0.3, linewidths=0.5)
    
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Location Density Heatmap")
    
    cbar = plt.colorbar(contour, ax=ax)
    cbar.set_label("Density")
    
    return fig
