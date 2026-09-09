"""
data_cleaning.py - Vectorized cleaning for Uber Peru dataset
Handles ; delimiter and European decimals
"""
import pandas as pd
import numpy as np

LIMA_BBOX = {
    "lat_min": -12.5, "lat_max": -11.8,
    
    "lon_min": -77.2, "lon_max": -76.5
}

def load_uber_data(path: str) -> pd.DataFrame:
    """Correct loader for this specific CSV"""
    df = pd.read_csv(path, delimiter=';', decimal=',', low_memory=False)
    # Fix last column name artifact from original export
    df.columns = [c.replace(',','').strip() for c in df.columns]
    if 'rider_score,,,,,,' in df.columns:
        df.rename(columns={'rider_score,,,,,,': 'rider_score'}, inplace=True)
    return df

def clean_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    """Vectorized validation - no apply()"""
    # Keep only necessary columns early to save memory
    cols = ['journey_id','user_id','start_at','end_at','start_lat','start_lon','end_lat','end_lon','end_state','distance','duration']
    cols = [c for c in cols if c in df.columns]
    clean = df[cols].copy()
    
    # Drop rows with null core coords
    clean.dropna(subset=['start_lat','start_lon','end_lat','end_lon'], inplace=True)
    
    # Valid lat/lon range
    valid = (
        clean['start_lat'].between(-90,90) & clean['start_lon'].between(-180,180) &
        clean['end_lat'].between(-90,90) & clean['end_lon'].between(-180,180)
    )
    clean = clean[valid]
    
    # Lima filter - removes GPS outliers like 57.47, 140.48
    lima_mask = (
        clean['start_lat'].between(LIMA_BBOX['lat_min'], LIMA_BBOX['lat_max']) &
        clean['start_lon'].between(LIMA_BBOX['lon_min'], LIMA_BBOX['lon_max']) &
        clean['end_lat'].between(LIMA_BBOX['lat_min'], LIMA_BBOX['lat_max']) &
        clean['end_lon'].between(LIMA_BBOX['lon_min'], LIMA_BBOX['lon_max'])
    )
    clean = clean[lima_mask]
    
    # Parse datetime
    clean['start_at'] = pd.to_datetime(clean['start_at'], format='%d/%m/%Y %H:%M', errors='coerce')
    clean['end_at'] = pd.to_datetime(clean['end_at'], format='%d/%m/%Y %H:%M', errors='coerce')
    
    return clean.reset_index(drop=True)

def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    df['hour'] = df['start_at'].dt.hour
    df['weekday'] = df['start_at'].dt.day_name()
    df['is_weekend'] = df['start_at'].dt.weekday >= 5
    return df
