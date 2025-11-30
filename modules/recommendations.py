"""Business Recommendation Computation
Computes viability scores for different business types at a given location
based on local POI composition without synthetic/random data.

Metrics per business type:
  demand_score: Relative density of supporting POI groups
  competition_score: Inverse density of competitor POI groups
  accessibility_score: Density of transport-related POIs
  infrastructure_score: Density of essential service groups
  viability_score: Weighted aggregate of the four component scores

All densities are normalized to 0-1 range across the evaluated business types.
"""

from __future__ import annotations
import math
import pandas as pd
from typing import Dict, List

# Implement local haversine (distance in km) to avoid missing import issues
def _haversine(coord1, coord2):
    lat1, lon1 = coord1
    lat2, lon2 = coord2
    # convert to radians
    lat1_r, lon1_r, lat2_r, lon2_r = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r
    a = math.sin(dlat/2)**2 + math.cos(lat1_r) * math.cos(lat2_r) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return 6371.0 * c


# Profile configuration mapping business types to groups and support/competitor sets
BUSINESS_PROFILES: Dict[str, Dict[str, List[str]]] = {
    'cafe': {
        'group': 'cafe',
        'supporting': ['retail', 'education', 'hospitality', 'finance'],
        'competitor': ['cafe', 'restaurant']
    },
    'restaurant': {
        'group': 'restaurant',
        'supporting': ['retail', 'education', 'hospitality', 'finance'],
        'competitor': ['restaurant', 'cafe', 'hospitality']
    },
    'bakery': {
        'group': 'restaurant',  # treat bakery as small food service
        'supporting': ['retail', 'education', 'hospitality', 'finance'],
        'competitor': ['restaurant', 'cafe']
    },
    'fast_food': {
        'group': 'restaurant',
        'supporting': ['retail', 'education', 'hospitality', 'finance'],
        'competitor': ['restaurant', 'cafe']
    },
    'shop': {
        'group': 'retail',
        'supporting': ['hospitality', 'restaurant', 'education', 'finance'],
        'competitor': ['retail']
    },
    'supermarket': {
        'group': 'retail',
        'supporting': ['hospitality', 'restaurant', 'education', 'finance'],
        'competitor': ['retail']
    },
    'pharmacy': {
        'group': 'healthcare',
        'supporting': ['retail', 'education', 'finance'],
        'competitor': ['healthcare']
    },
    'bank': {
        'group': 'finance',
        'supporting': ['retail', 'hospitality', 'education'],
        'competitor': ['finance']
    },
    'gym': {
        'group': 'other',
        'supporting': ['retail', 'hospitality', 'education', 'finance'],
        'competitor': ['other']
    },
    'hotel': {
        'group': 'hospitality',
        'supporting': ['retail', 'restaurant', 'education', 'finance'],
        'competitor': ['hospitality']
    }
}

DEFAULT_WEIGHTS = {
    'demand': 0.4,
    'competition': 0.3,
    'accessibility': 0.2,
    'infrastructure': 0.1
}

TRANSPORT_KEYWORDS = ['train', 'bus', 'station', 'footway', 'footpath']
INFRA_GROUPS = ['healthcare', 'education', 'finance']


def _filter_radius(df: pd.DataFrame, center_lat: float, center_lon: float, radius_km: float) -> pd.DataFrame:
    if len(df) == 0:
        return df.copy()
    dists = df.apply(lambda r: _haversine((center_lat, center_lon), (r['latitude'], r['longitude'])), axis=1)
    return df[dists <= radius_km].copy()


def compute_business_recommendations(cleaned_pois: pd.DataFrame,
                                     center_lat: float,
                                     center_lon: float,
                                     radius_km: float = 5.0,
                                     weights: Dict[str, float] | None = None) -> pd.DataFrame:
    """Compute viability scores for each business type using real POI data.

    Args:
        cleaned_pois: DataFrame with at least ['latitude','longitude','category','category_group']
        center_lat: Center latitude
        center_lon: Center longitude
        radius_km: Radius for local analysis
        weights: Optional custom weights dict

    Returns:
        DataFrame of business recommendations sorted by viability_score
    """
    if cleaned_pois is None or len(cleaned_pois) == 0:
        return pd.DataFrame(columns=[
            'Business Type','Viability Score','Demand','Competition','Accessibility','Infrastructure',
            'Supporting Count','Competitor Count','Transport Count','Infrastructure Count','Total Local'
        ])

    local = _filter_radius(cleaned_pois, center_lat, center_lon, radius_km)
    if len(local) == 0:
        return pd.DataFrame(columns=[
            'Business Type','Viability Score','Demand','Competition','Accessibility','Infrastructure',
            'Supporting Count','Competitor Count','Transport Count','Infrastructure Count','Total Local'
        ])

    area_km2 = math.pi * (radius_km ** 2)
    weights = weights or DEFAULT_WEIGHTS.copy()
    total_local = len(local)

    # Precompute transport & infrastructure counts
    transport_count = 0
    if 'category' in local.columns:
        for kw in TRANSPORT_KEYWORDS:
            transport_count += len(local[local['category'].str.contains(kw, case=False, na=False)])

    infra_count = 0
    if 'category_group' in local.columns:
        infra_count = len(local[local['category_group'].isin(INFRA_GROUPS)])

    records = []
    for b_type, profile in BUSINESS_PROFILES.items():
        group = profile['group']
        supporting_groups = profile['supporting']
        competitor_groups = profile['competitor']

        if 'category_group' in local.columns:
            supporting_count = len(local[local['category_group'].isin(supporting_groups)])
            competitor_count = len(local[local['category_group'].isin(competitor_groups)])
        else:
            supporting_count = 0
            competitor_count = 0

        # Densities
        supporting_density = supporting_count / area_km2
        competitor_density = competitor_count / area_km2
        transport_density = transport_count / area_km2
        infra_density = infra_count / area_km2

        records.append({
            'Business Type': b_type,
            'supporting_density': supporting_density,
            'competitor_density': competitor_density,
            'transport_density': transport_density,
            'infra_density': infra_density,
            'Supporting Count': supporting_count,
            'Competitor Count': competitor_count,
            'Transport Count': transport_count,
            'Infrastructure Count': infra_count,
            'Total Local': total_local
        })

    df = pd.DataFrame(records)
    # Normalization (avoid division by zero)
    for col in ['supporting_density','competitor_density','transport_density','infra_density']:
        max_val = df[col].max()
        if max_val > 0:
            df[col + '_norm'] = df[col] / max_val
        else:
            df[col + '_norm'] = 0.0

    # Component scores
    df['Demand'] = df['supporting_density_norm']
    df['Competition'] = 1 - df['competitor_density_norm']  # inverse
    df['Accessibility'] = df['transport_density_norm']
    df['Infrastructure'] = df['infra_density_norm']

    # Viability
    df['Viability Score'] = (
        weights['demand'] * df['Demand'] +
        weights['competition'] * df['Competition'] +
        weights['accessibility'] * df['Accessibility'] +
        weights['infrastructure'] * df['Infrastructure']
    )

    out_cols = [
        'Business Type','Viability Score','Demand','Competition','Accessibility','Infrastructure',
        'Supporting Count','Competitor Count','Transport Count','Infrastructure Count','Total Local'
    ]
    df = df[out_cols].sort_values('Viability Score', ascending=False).reset_index(drop=True)
    df['Rank'] = df.index + 1
    return df[['Rank'] + out_cols]


__all__ = ['compute_business_recommendations']


def annotate_clusters_with_transport(top_clusters: pd.DataFrame,
                                     cleaned_pois: pd.DataFrame,
                                     transport_keywords: List[str] | None = None) -> pd.DataFrame:
    """Annotate cluster centers with nearest public transport info.

    Args:
        top_clusters: DataFrame with columns ['latitude','longitude'] for cluster centers
        cleaned_pois: POIs DataFrame including 'latitude','longitude','category'
        transport_keywords: Optional list of keywords to match transport POIs

    Returns:
        New DataFrame with added columns:
            - nearest_transport_type
            - nearest_transport_name
            - nearest_transport_distance_km
    """
    if top_clusters is None or len(top_clusters) == 0:
        return top_clusters
    if cleaned_pois is None or len(cleaned_pois) == 0:
        df = top_clusters.copy()
        df['nearest_transport_type'] = None
        df['nearest_transport_name'] = None
        df['nearest_transport_distance_km'] = None
        return df

    transport_keywords = transport_keywords or ['train', 'bus', 'station','footway', 'footpath']

    # Filter transport POIs
    if 'category' in cleaned_pois.columns:
        mask = pd.Series(False, index=cleaned_pois.index)
        for kw in transport_keywords:
            mask = mask | cleaned_pois['category'].str.contains(kw, case=False, na=False)
        transport_pois = cleaned_pois[mask].copy()
    else:
        transport_pois = cleaned_pois.copy()

    df = top_clusters.copy()
    df['nearest_transport_type'] = None
    df['nearest_transport_name'] = None
    df['nearest_transport_distance_km'] = None

    if len(transport_pois) == 0:
        return df

    # Compute nearest transport per cluster center
    for i in range(len(df)):
        clat = float(df.iloc[i]['latitude'])
        clon = float(df.iloc[i]['longitude'])
        best_d = float('inf')
        best_type = None
        best_name = None
        for _, r in transport_pois.iterrows():
            tlat = float(r['latitude'])
            tlon = float(r['longitude'])
            d = _haversine((clat, clon), (tlat, tlon))
            if d < best_d:
                best_d = d
                best_name = str(r.get('name') or '')
                # Infer type from category
                cat = str(r.get('category') or '')
                # Pick first matching keyword
                matched = None
                for kw in transport_keywords:
                    if kw.lower() in cat.lower():
                        matched = kw
                        break
                best_type = matched or ('transport' if cat else None)
        df.at[df.index[i], 'nearest_transport_type'] = best_type
        df.at[df.index[i], 'nearest_transport_name'] = best_name
        df.at[df.index[i], 'nearest_transport_distance_km'] = round(best_d, 3) if best_d != float('inf') else None

    return df


__all__.extend(['annotate_clusters_with_transport'])

def annotate_locations_with_transport(locations_df: pd.DataFrame,
                                      cleaned_pois: pd.DataFrame,
                                      transport_keywords: List[str] | None = None) -> pd.DataFrame:
    """Annotate scored location candidates with nearest public transport.

    Args:
        locations_df: DataFrame containing at least ['latitude','longitude'] for recommended spots
        cleaned_pois: Full POI set including potential transport features
        transport_keywords: Override list of transport category substrings

    Returns:
        DataFrame with added columns:
            nearest_transport_type, nearest_transport_name, nearest_transport_distance_km
    """
    if locations_df is None or len(locations_df) == 0:
        return locations_df
    if cleaned_pois is None or len(cleaned_pois) == 0:
        df = locations_df.copy()
        df['nearest_transport_type'] = None
        df['nearest_transport_name'] = None
        df['nearest_transport_distance_km'] = None
        return df

    transport_keywords = transport_keywords or ['train','bus','station','footway','footpath']
    if 'category' in cleaned_pois.columns:
        mask = pd.Series(False, index=cleaned_pois.index)
        for kw in transport_keywords:
            mask = mask | cleaned_pois['category'].str.contains(kw, case=False, na=False)
        transport_pois = cleaned_pois[mask].copy()
    else:
        transport_pois = cleaned_pois.copy()

    df = locations_df.copy()
    df['nearest_transport_type'] = None
    df['nearest_transport_name'] = None
    df['nearest_transport_distance_km'] = None

    if len(transport_pois) == 0:
        return df

    for i in range(len(df)):
        clat = float(df.iloc[i]['latitude'])
        clon = float(df.iloc[i]['longitude'])
        best_d = float('inf')
        best_type = None
        best_name = None
        for _, r in transport_pois.iterrows():
            tlat = float(r['latitude'])
            tlon = float(r['longitude'])
            d = _haversine((clat, clon), (tlat, tlon))
            if d < best_d:
                best_d = d
                best_name = str(r.get('name') or '')
                cat = str(r.get('category') or '')
                matched = None
                for kw in transport_keywords:
                    if kw.lower() in cat.lower():
                        matched = kw
                        break
                best_type = matched or ('transport' if cat else None)
        df.at[df.index[i], 'nearest_transport_type'] = best_type
        df.at[df.index[i], 'nearest_transport_name'] = best_name
        df.at[df.index[i], 'nearest_transport_distance_km'] = round(best_d, 3) if best_d != float('inf') else None

    return df

__all__.extend(['annotate_locations_with_transport'])
