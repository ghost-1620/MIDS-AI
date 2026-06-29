# src/data_simulator.py
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def generate_synthetic_marinecadastre(num_steps=100):
    """
    Generates a realistic, self-contained mock dataset mirroring
    the exact structure and layout of an AccessAIS/MarineCadastre CSV export.
    """
    print(f"Generating self-contained Maritime data stream ({num_steps} positions)...")

    # 1. Initialize anchor variables for a mock vessel track
    mmsi_id = 477220100  # Mock Container Ship Identity Number
    start_time = datetime(2026, 6, 5, 12, 0, 0)

    # Starting coordinates right outside a major shipping channel
    base_lat = 24.8607
    base_lon = 67.0011

    # Baseline cruising speed (Knots) and heading tracking direction (Degrees)
    current_sog = 14.5
    current_cog = 45.0  # Heading North-East

    records = []

    for step in range(num_steps):
        # Move time forward by 1 minute (60 seconds) per ping entry
        current_time = start_time + timedelta(minutes=step)

        # Add slight natural environmental drift/noise to physics parameters
        current_sog += np.random.uniform(-0.1, 0.1)
        current_cog += np.random.uniform(-0.5, 0.5)

        # Convert speed and heading vectors into progressive latitude/longitude changes
        # (Approximate degree variations corresponding to physical movement scale)
        lat_step = (current_sog * 0.0003) * np.cos(np.radians(current_cog))
        lon_step = (current_sog * 0.0003) * np.sin(np.radians(current_cog))

        base_lat += lat_step
        base_lon += lon_step

        records.append({
            'MMSI': mmsi_id,
            'BaseDateTime': current_time.strftime('%Y-%m-%dT%H:%M:%S'),
            'LAT': base_lat,
            'LON': base_lon,
            'SOG': round(max(0.1, current_sog), 1),
            'COG': round(current_cog % 360.0, 1)
        })

    df = pd.DataFrame(records)
    print(f"Success: Engineered {len(df)} tracking records for MMSI: {mmsi_id}")
    return df


if __name__ == "__main__":
    # Local test execution block to verify the file works independently
    mock_data = generate_synthetic_marinecadastre(num_steps=10)
    print("\nSample Rows of the Internal Data Stream Pipeline:")
    print(mock_data.head())