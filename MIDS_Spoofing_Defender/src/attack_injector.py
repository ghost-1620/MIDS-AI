# src/attack_injector.py
import numpy as np


def inject_gradual_drift(df_vessel, drift_rate_lat=0.002, drift_rate_lon=0.002, start_idx=30):
    """
    Simulates a sophisticated GPS spoofing cyber-intrusion by progressively
    shifting coordinates over time on a single vessel's track.
    """
    spoofed_df = df_vessel.copy()
    spoofed_df['is_spoofed'] = 0  # Label 0 = Secure Data

    if len(spoofed_df) <= start_idx:
        return spoofed_df

    # Generate an accumulating delta sequence starting from the injection point
    drift_steps = np.arange(1, len(spoofed_df) - start_idx + 1)

    spoofed_df.iloc[start_idx:, spoofed_df.columns.get_loc('LAT')] += drift_steps * drift_rate_lat
    spoofed_df.iloc[start_idx:, spoofed_df.columns.get_loc('LON')] += drift_steps * drift_rate_lon
    spoofed_df.iloc[start_idx:, spoofed_df.columns.get_loc('is_spoofed')] = 1  # Label 1 = Spoofed

    print(f"Cyber-Attack Simulator: Injected gradual coordinate drift at step index {start_idx}.")
    return spoofed_df