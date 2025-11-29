"""
Business Suggestion Module
Recommends suitable business types for a given location
"""

import pandas as pd
from modules.clean_data import DataCleaner
from modules.scoring import LocationScorer


class BusinessSuggester:
    """
    Analyzes a specific location and suggests viable business types.
    """
    
    BUSINESS_TYPES = [
        'cafe', 'restaurant', 'bakery', 'fast_food', 'shop',
        'supermarket', 'pharmacy', 'bank', 'gym', 'hotel'
    ]
    
    def __init__(self, cleaned_pois: pd.DataFrame, center_lat: float, center_lon: float):
        """
        Initialize the BusinessSuggester.
        
        Args:
            cleaned_pois: GeoDataFrame of all POIs in the area
            center_lat: Latitude of city center
            center_lon: Longitude of city center
        """
        self.cleaned_pois = cleaned_pois
        self.center_lat = center_lat
        self.center_lon = center_lon
    
    def suggest_businesses_at_location(self, lat: float, lon: float, 
                                       radius_km: float = 5.0) -> pd.DataFrame:
        """
        Suggest viable business types for a given location.
        
        Args:
            lat: Latitude of the location
            lon: Longitude of the location
            radius_km: Search radius in km
            
        Returns:
            DataFrame with business suggestions ranked by viability score
        """
        suggestions = []
        
        # Use equal weights for comprehensive analysis
        weights = {
            'demand': 0.25,
            'competition': 0.25,
            'accessibility': 0.25,
            'infrastructure': 0.25
        }
        
        for business_type in self.BUSINESS_TYPES:
            try:
                # Create scorer for this business type
                scorer = LocationScorer(weights)
                
                # Get supporting POIs for this business type
                from modules.fetch_data import DataFetcher
                fetcher = DataFetcher("", radius_km)
                supporting_categories = fetcher.get_supporting_pois(business_type)
                
                # Score just this one location
                candidates = pd.DataFrame([{
                    'latitude': lat,
                    'longitude': lon
                }])
                
                scores = scorer.score_locations(
                    candidates,
                    self.cleaned_pois,
                    business_type,
                    supporting_categories,
                    radius_km=radius_km
                )
                
                if len(scores) > 0:
                    score_row = scores.iloc[0]
                    
                    # Calculate local competition density
                    local_pois = self._get_nearby_pois(lat, lon, 0.5)
                    business_pois = local_pois[
                        local_pois['category'].str.contains(business_type, case=False, na=False)
                    ]
                    competition_density = len(business_pois)
                    
                    suggestions.append({
                        'Business Type': business_type.title(),
                        'Viability Score': float(score_row['final_score']),
                        'Demand': float(score_row['demand_score']),
                        'Competition': float(score_row['competition_score']),
                        'Accessibility': float(score_row['accessibility_score']),
                        'Infrastructure': float(score_row['infrastructure_score']),
                        'Local Competitors': competition_density,
                        'Category': business_type
                    })
            except Exception as e:
                # If scoring fails for a business type, assign a default low score
                suggestions.append({
                    'Business Type': business_type.title(),
                    'Viability Score': 0.0,
                    'Demand': 0.0,
                    'Competition': 0.0,
                    'Accessibility': 0.0,
                    'Infrastructure': 0.0,
                    'Local Competitors': 0,
                    'Category': business_type
                })
        
        # Create DataFrame and sort by viability
        results = pd.DataFrame(suggestions)
        results = results.sort_values('Viability Score', ascending=False)
        results['Rank'] = range(1, len(results) + 1)
        
        return results[['Rank', 'Business Type', 'Viability Score', 'Demand', 
                        'Competition', 'Accessibility', 'Infrastructure', 
                        'Local Competitors']]
    
    def _get_nearby_pois(self, lat: float, lon: float, radius_km: float) -> pd.DataFrame:
        """
        Get POIs within a specific radius of a location.
        
        Args:
            lat: Latitude
            lon: Longitude
            radius_km: Radius in kilometers
            
        Returns:
            Filtered GeoDataFrame
        """
        from scipy.spatial.distance import haversine
        
        distances = self.cleaned_pois.apply(
            lambda row: haversine(
                (lat, lon),
                (row['latitude'], row['longitude'])
            ),
            axis=1
        )
        
        return self.cleaned_pois[distances <= radius_km].copy()
    
    def get_recommendation_summary(self, suggestions: pd.DataFrame) -> str:
        """
        Get a text summary of business suggestions.
        
        Args:
            suggestions: DataFrame from suggest_businesses_at_location
            
        Returns:
            Summary string
        """
        if len(suggestions) == 0:
            return "No business suggestions available."
        
        summary = []
        summary.append("=" * 60)
        summary.append("BUSINESS VIABILITY ANALYSIS FOR LOCATION")
        summary.append("=" * 60)
        summary.append("")
        
        # Top 3 recommendations
        summary.append("🏆 TOP RECOMMENDED BUSINESSES:")
        for idx, row in suggestions.head(3).iterrows():
            medal = "🥇" if row['Rank'] == 1 else "🥈" if row['Rank'] == 2 else "🥉"
            summary.append(
                f"{medal} #{row['Rank']}. {row['Business Type']}: "
                f"Score {row['Viability Score']:.3f}"
            )
        
        summary.append("")
        summary.append("📊 ANALYSIS:")
        
        top_business = suggestions.iloc[0]
        summary.append(f"Best option: {top_business['Business Type']}")
        summary.append(f"  - Demand: {top_business['Demand']:.3f}/1.0")
        summary.append(f"  - Competition: {top_business['Competition']:.3f}/1.0")
        summary.append(f"  - Accessibility: {top_business['Accessibility']:.3f}/1.0")
        summary.append(f"  - Infrastructure: {top_business['Infrastructure']:.3f}/1.0")
        summary.append(f"  - Local Competitors: {int(top_business['Local Competitors'])}")
        
        summary.append("")
        summary.append("=" * 60)
        
        return "\n".join(summary)
