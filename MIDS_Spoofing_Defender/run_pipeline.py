# run_pipeline.py
from src.data_simulator import generate_synthetic_marinecadastre
from src.attack_injector import inject_gradual_drift
from src.kinematics_engine import calculate_kinematics
from src.model_engine import train_gru_model  # <-- Imports your newly fixed AI engine


def main():
    print("=== STARTING MARITIME CYBER DEFENSE DATA PIPELINE ===\n")

    # 1. Generate mock shipping telemetry (Increased steps to 100 for better AI training)
    raw_stream = generate_synthetic_marinecadastre(num_steps=100)
    print("-" * 60)

    # 2. Inject gradual GPS spoofing attack vectors starting at index 45
    spoofed_stream = inject_gradual_drift(raw_stream, start_idx=45)
    print("-" * 60)

    # 3. Process raw coordinate changes into kinematics tracking features
    final_features = calculate_kinematics(spoofed_stream)
    print("-" * 60)

    # 4. Define the training input parameters for the AI
    FEATURES_FOR_AI = ['V_calc', 'Speed_Delta', 'Angular_Velocity']

    # 5. Train your Gated Recurrent Unit (GRU) neural network
    trained_detector = train_gru_model(
        df_features=final_features,
        feature_cols=FEATURES_FOR_AI,
        epochs=25,
        sequence_length=5
    )

    print("\n" + "=" * 60)
    print("MIDS SYSTEM ENGINE FULLY OPERATIONAL!")
    print("The GRU network is now primed to intercept telemetry drift anomalies.")
    print("=" * 60)


if __name__ == "__main__":
    main()