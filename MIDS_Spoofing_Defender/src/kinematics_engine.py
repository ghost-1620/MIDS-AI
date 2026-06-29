# src/kinematics_engine.py
import numpy as np
import pandas as pd


def calculate_kinematics(df):
    """
    Transforms raw geospatial coordinates into dynamic tracking vectors using the Haversine formula.
    """
    print("Kinematics Engine: Extracting spatial vector features...")
    df = df.sort_values(by='BaseDateTime').reset_index(drop=True)

    # Calculate time differentials (seconds) between pings
    df['dt'] = pd.to_datetime(df['BaseDateTime']).diff().dt.total_seconds().fillna(60)
    df['dt'] = np.where(df['dt'] == 0, 1, df['dt'])  # Avoid divide-by-zero errors

    # Haversine Formula Setup
    R = 6371.0  # Earth's radius in kilometers
    lat1 = np.radians(df['LAT'].shift(1))
    lat2 = np.radians(df['LAT'])
    dlat = np.radians(df['LAT'].diff())
    dlon = np.radians(df['LON'].diff())

    # Spherical Trigonometry to calculate actual physical distance moved
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance_km = R * c
    distance_nm = distance_km * 0.539957  # Convert Kilometers to Nautical Miles

    # Feature 1: Calculated True Physical Velocity (in Knots)
    df['V_calc'] = (distance_nm / (df['dt'] / 3600.0)).fillna(0)

    # Feature 2: Speed Delta (Discrepancy between broadcast SOG and true physics speed)
    df['Speed_Delta'] = np.abs(df['V_calc'] - df['SOG']).fillna(0)

    # Feature 3: Angular Velocity (Heading modifications over time)
    df['Heading_Delta'] = df['COG'].diff().abs().fillna(0)
    df['Angular_Velocity'] = df['Heading_Delta'] / df['dt']

    return df.dropna()