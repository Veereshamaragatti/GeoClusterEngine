"""
Data Fetching Module
Fetches POI data from OpenStreetMap using OSMnx and Overpass API
"""

import os
import time
import hashlib
import json
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import osmnx as ox
import requests


class DataFetcher:
    """
    Handles fetching of Points of Interest (POI) data from OpenStreetMap.
    Supports various POI categories including shops, restaurants, schools,
    banks, hospitals, and custom business types.
    """
    
    POI_CATEGORIES = {
        'shop': {'shop': True},
        'restaurant': {'amenity': 'restaurant'},
        'cafe': {'amenity': 'cafe'},
        'coffee_shop': {'amenity': ['cafe', 'coffee']},
        'school': {'amenity': 'school'},
        'bank': {'amenity': 'bank'},
        'hospital': {'amenity': 'hospital'},
        'pharmacy': {'amenity': 'pharmacy'},
        'supermarket': {'shop': 'supermarket'},
        'convenience': {'shop': 'convenience'},
        'fast_food': {'amenity': 'fast_food'},
        'hotel': {'tourism': 'hotel'},
        'atm': {'amenity': 'atm'},
        'fuel': {'amenity': 'fuel'},
        'parking': {'amenity': 'parking'},
        'bus_station': {'amenity': 'bus_station'},
        'gym': {'leisure': 'fitness_centre'},
        'cinema': {'amenity': 'cinema'},
        'mall': {'shop': 'mall'},
        'bakery': {'shop': 'bakery'}
    }
    
    INFRASTRUCTURE_TAGS = {
        'roads': {'highway': ['primary', 'secondary', 'tertiary', 'residential']},
        'public_transport': {'public_transport': True},
        'bus_stops': {'highway': 'bus_stop'}
    }
    
    def __init__(self, city_name: str, radius_km: float = 5.0, fast_mode: bool = True):
        """
        Initialize the DataFetcher with city and search parameters.
        
        Args:
            city_name: Name of the city to search in
            radius_km: Search radius in kilometers
        """
        self.city_name = city_name
        self.radius_km = radius_km
        self.radius_m = radius_km * 1000
        self.center_point = None
        self.all_pois = None
        self.fast_mode = fast_mode
        # Configure osmnx cache to speed up repeated queries
        try:
            ox.settings.use_cache = True
            cache_dir = os.path.join('cache', 'osmnx')
            os.makedirs(cache_dir, exist_ok=True)
            ox.settings.cache_folder = cache_dir
            ox.settings.log_console = False
        except Exception:
            pass
        
    def get_city_center(self) -> tuple:
        """
        Get the center coordinates of the specified city.
        
        Returns:
            Tuple of (latitude, longitude)
        """
        try:
            geocode_result = ox.geocode(self.city_name)
            self.center_point = geocode_result
            return geocode_result
        except Exception as e:
            print(f"Error geocoding city: {e}")
            return None
    
    def fetch_pois_by_category(self, category: str) -> gpd.GeoDataFrame:
        """
        Fetch POIs for a specific category within the search area.
        
        Args:
            category: POI category name (e.g., 'restaurant', 'school')
            
        Returns:
            GeoDataFrame containing POIs
        """
        if self.center_point is None:
            self.get_city_center()
            
        if self.center_point is None:
            return gpd.GeoDataFrame()
        
        tags = self.POI_CATEGORIES.get(category, {'amenity': category})
        
        try:
            gdf = ox.features_from_point(
                self.center_point,
                tags=tags,
                dist=self.radius_m
            )
            
            if len(gdf) > 0:
                gdf['category'] = category
                gdf['source'] = 'osm'
                
                if 'name' not in gdf.columns:
                    gdf['name'] = f'{category}_unnamed'
                    
            return gdf
            
        except Exception as e:
            print(f"Error fetching {category}: {e}")
            return gpd.GeoDataFrame()
    
    def fetch_all_pois(self, categories: list = None) -> gpd.GeoDataFrame:
        """
        Fetch all POIs for multiple categories.
        
        Args:
            categories: List of categories to fetch. If None, fetches all.
            
        Returns:
            Combined GeoDataFrame of all POIs
        """
        if categories is None:
            categories = list(self.POI_CATEGORIES.keys())
        
        # Fast path: combine tags into a single Overpass request where possible
        try:
            combined = self._build_combined_tags(categories)
        except Exception:
            combined = None

        if combined is not None and self.fast_mode:
            gdf = self._fetch_pois_bulk(combined, categories)
            if len(gdf) > 0:
                self.all_pois = gdf
                return gdf

        # Fallback to per-category sequential fetch
        all_gdfs = []
        for category in categories:
            print(f"Fetching {category}...")
            gdf = self.fetch_pois_by_category(category)
            if len(gdf) > 0:
                all_gdfs.append(gdf)
                print(f"  Found {len(gdf)} {category} POIs")
        if all_gdfs:
            self.all_pois = gpd.GeoDataFrame(pd.concat(all_gdfs, ignore_index=True))
            return self.all_pois
        return gpd.GeoDataFrame()

    def _build_combined_tags(self, categories: list) -> dict:
        """Build a combined tags dict to query multiple categories at once.
        Reduces the number of Overpass requests significantly.
        """
        if self.center_point is None:
            self.get_city_center()
        if self.center_point is None:
            return None

        amenity_vals = []
        shop_vals = []
        leisure_vals = []
        tourism_vals = []
        highway_vals = []

        for cat in categories:
            spec = self.POI_CATEGORIES.get(cat, {'amenity': cat})
            for key, val in spec.items():
                if key == 'amenity':
                    if isinstance(val, list):
                        amenity_vals.extend(val)
                    elif val is True:
                        # Ignore amenity True (too broad)
                        pass
                    else:
                        amenity_vals.append(val)
                elif key == 'shop':
                    if val is True:
                        # Broad 'shop' fetch is expensive; narrow in fast mode
                        if self.fast_mode:
                            shop_vals.extend(['supermarket', 'convenience', 'mall', 'bakery'])
                        else:
                            # If not fast_mode, still avoid True to prevent huge responses
                            shop_vals.extend(['supermarket', 'convenience', 'mall', 'bakery'])
                    elif isinstance(val, list):
                        shop_vals.extend(val)
                    else:
                        shop_vals.append(val)
                elif key == 'leisure':
                    if isinstance(val, list):
                        leisure_vals.extend(val)
                    else:
                        leisure_vals.append(val)
                elif key == 'tourism':
                    if isinstance(val, list):
                        tourism_vals.extend(val)
                    else:
                        tourism_vals.append(val)
                elif key == 'highway':
                    if isinstance(val, list):
                        highway_vals.extend(val)
                    else:
                        highway_vals.append(val)

        tags = {}
        if amenity_vals:
            tags['amenity'] = sorted(list(set(amenity_vals)))
        if shop_vals:
            tags['shop'] = sorted(list(set(shop_vals)))
        if leisure_vals:
            tags['leisure'] = sorted(list(set(leisure_vals)))
        if tourism_vals:
            tags['tourism'] = sorted(list(set(tourism_vals)))
        if highway_vals:
            tags['highway'] = sorted(list(set(highway_vals)))

        return tags if tags else None

    def _bulk_cache_key(self, combined_tags: dict) -> str:
        sig = json.dumps({k: combined_tags[k] for k in sorted(combined_tags)}, sort_keys=True)
        cp = self.center_point or (0, 0)
        key = f"{cp[0]:.5f}_{cp[1]:.5f}_{int(self.radius_m)}_{sig}"
        return hashlib.md5(key.encode('utf-8')).hexdigest()

    def _fetch_pois_bulk(self, combined_tags: dict, categories: list) -> gpd.GeoDataFrame:
        """Single Overpass request for many categories with simple caching."""
        if self.center_point is None:
            self.get_city_center()
        if self.center_point is None:
            return gpd.GeoDataFrame()

        cache_dir = os.path.join('cache', 'bulk')
        os.makedirs(cache_dir, exist_ok=True)
        cache_key = self._bulk_cache_key(combined_tags)
        cache_path = os.path.join(cache_dir, f"pois_{cache_key}.parquet")

        # Try cache
        try:
            if os.path.exists(cache_path):
                df = gpd.read_parquet(cache_path)
                return df
        except Exception:
            pass

        try:
            gdf = ox.features_from_point(self.center_point, tags=combined_tags, dist=self.radius_m)
            if len(gdf) == 0:
                return gpd.GeoDataFrame()

            # Derive our 'category' values from OSM tags
            def derive_category(row):
                # Priority by specificity
                val = str(row.get('amenity', '')).lower()
                if val in ['restaurant', 'cafe', 'fast_food', 'pharmacy', 'bank', 'fuel', 'parking', 'bus_station']:
                    return val
                sval = str(row.get('shop', '')).lower()
                if sval in ['supermarket', 'convenience', 'mall', 'bakery']:
                    return sval
                tval = str(row.get('tourism', '')).lower()
                if tval == 'hotel':
                    return 'hotel'
                lval = str(row.get('leisure', '')).lower()
                if lval == 'fitness_centre':
                    return 'gym'
                return 'other'

            gdf = gdf.copy()
            gdf['category'] = gdf.apply(derive_category, axis=1)
            gdf['source'] = 'osm'
            if 'name' not in gdf.columns:
                gdf['name'] = 'unknown'

            # Save cache
            try:
                gdf.to_parquet(cache_path, index=False)
            except Exception:
                pass

            return gdf
        except Exception as e:
            print(f"Bulk fetch failed, falling back. Reason: {e}")
            return gpd.GeoDataFrame()
    
    def fetch_competitors(self, business_type: str) -> gpd.GeoDataFrame:
        """
        Fetch competitor businesses for a specific business type.
        
        Args:
            business_type: Type of business to find competitors for
            
        Returns:
            GeoDataFrame of competitor locations
        """
        competitors = self.fetch_pois_by_category(business_type)
        if len(competitors) > 0:
            competitors['is_competitor'] = True
        return competitors
    
    def fetch_infrastructure(self) -> dict:
        """
        Fetch infrastructure data (roads, public transport).
        
        Returns:
            Dictionary containing infrastructure GeoDataFrames
        """
        if self.center_point is None:
            self.get_city_center()
            
        infrastructure = {}
        
        try:
            G = ox.graph_from_point(
                self.center_point,
                dist=self.radius_m,
                network_type='drive'
            )
            nodes, edges = ox.graph_to_gdfs(G)
            infrastructure['road_network'] = edges
            infrastructure['road_nodes'] = nodes
        except Exception as e:
            print(f"Error fetching road network: {e}")
            
        try:
            bus_stops = ox.features_from_point(
                self.center_point,
                tags={'highway': 'bus_stop'},
                dist=self.radius_m
            )
            infrastructure['bus_stops'] = bus_stops
        except Exception as e:
            print(f"Error fetching bus stops: {e}")
            
        return infrastructure
    
    def save_to_csv(self, gdf: gpd.GeoDataFrame, filename: str) -> str:
        """
        Save GeoDataFrame to CSV file.
        
        Args:
            gdf: GeoDataFrame to save
            filename: Output filename
            
        Returns:
            Path to saved file
        """
        filepath = os.path.join('data', filename)
        
        df = gdf.copy()
        
        if 'geometry' in df.columns:
            df['latitude'] = df.geometry.apply(
                lambda g: g.centroid.y if g else None
            )
            df['longitude'] = df.geometry.apply(
                lambda g: g.centroid.x if g else None
            )
            df = df.drop(columns=['geometry'])
        
        cols_to_keep = ['name', 'category', 'latitude', 'longitude', 'source']
        available_cols = [c for c in cols_to_keep if c in df.columns]
        
        for col in df.columns:
            if col not in available_cols:
                try:
                    if df[col].apply(lambda x: isinstance(x, (list, dict))).any():
                        continue
                    available_cols.append(col)
                except:
                    pass
        
        df_export = df[available_cols].copy()
        
        for col in df_export.columns:
            try:
                df_export[col] = df_export[col].apply(
                    lambda x: str(x) if isinstance(x, (list, dict)) else x
                )
            except:
                pass
        
        df_export.to_csv(filepath, index=False)
        print(f"Saved CSV to {filepath}")
        return filepath
    
    def save_to_geojson(self, gdf: gpd.GeoDataFrame, filename: str) -> str:
        """
        Save GeoDataFrame to GeoJSON file.
        
        Args:
            gdf: GeoDataFrame to save
            filename: Output filename
            
        Returns:
            Path to saved file
        """
        filepath = os.path.join('data', filename)
        
        gdf_clean = gdf.copy()
        
        for col in gdf_clean.columns:
            if col != 'geometry':
                try:
                    gdf_clean[col] = gdf_clean[col].apply(
                        lambda x: str(x) if isinstance(x, (list, dict)) else x
                    )
                except:
                    gdf_clean[col] = gdf_clean[col].astype(str)
        
        gdf_clean.to_file(filepath, driver='GeoJSON')
        print(f"Saved GeoJSON to {filepath}")
        return filepath
    
    def get_supporting_pois(self, business_type: str) -> list:
        """
        Get list of supporting POI categories for a business type.
        These are POIs that typically drive foot traffic to an area.
        
        Args:
            business_type: Type of business
            
        Returns:
            List of supporting POI categories
        """
        supporting_map = {
            'restaurant': ['shop', 'supermarket', 'mall', 'cinema', 'hotel'],
            'cafe': ['shop', 'bank', 'school', 'gym', 'supermarket'],
            'coffee_shop': ['shop', 'bank', 'school', 'gym', 'supermarket'],
            'pharmacy': ['hospital', 'school', 'supermarket', 'bank'],
            'gym': ['cafe', 'restaurant', 'supermarket', 'mall'],
            'hotel': ['restaurant', 'cafe', 'mall', 'cinema', 'bus_station'],
            'supermarket': ['bank', 'pharmacy', 'fuel', 'parking'],
            'default': ['shop', 'restaurant', 'cafe', 'supermarket', 'mall']
        }
        
        return supporting_map.get(business_type, supporting_map['default'])


def fetch_realtime_footfall(lat: float, lon: float) -> dict:
    """
    Fetch real-time footfall estimate for a location.
    (Placeholder - returns simulated data)
    
    Args:
        lat: Latitude
        lon: Longitude
        
    Returns:
        Dictionary with footfall estimate
    """
    import numpy as np
    
    footfall_estimate = 120 + np.random.randint(-10, 10)
    
    return {
        "footfall_live": footfall_estimate,
        "unit": "people/hour",
        "confidence": 0.75,
        "last_updated": "2025-11-29 21:00:00"
    }


def fetch_events(area: str) -> list:
    """
    Fetch upcoming events in an area.
    (Placeholder - returns sample events)
    
    Args:
        area: Area name or coordinates
        
    Returns:
        List of event dictionaries
    """
    sample_events = [
        {"name": "Local Festival", "impact": 1.3, "date": "2025-12-15"},
        {"name": "Cricket Match", "impact": 1.8, "date": "2025-12-20"},
        {"name": "Farmers Market", "impact": 1.2, "date": "2025-12-08"}
    ]
    
    return sample_events
