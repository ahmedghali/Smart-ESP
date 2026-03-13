"""
Main Training Script for ESP AI Models
=======================================
Unified training script for all AI components:
- LSTM Predictor (failure prediction)
- LSTM Autoencoder (anomaly detection)
- DRL Optimizer (operational optimization)
- PINN Model (physics-constrained predictions)
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
import json

# Add project root
sys.path.append(str(Path(__file__).parent))

from src.config.config import Config
from src.data.synthetic_generator import SyntheticESPDataGenerator, FailureMode
from src.data.preprocessor import DataPreprocessor
from src.data.data_loader import ESPDataLoader


def setup_paths(config: Config) -> None:
    """Create necessary directories."""
    directories = [
        config.data_dir,
        Path(config.data_dir) / "design",
        Path(config.data_dir) / "raw",
        Path(config.data_dir) / "processed",
        Path(config.data_dir) / "synthetic",
        config.model_dir,
        config.logs_dir,
        Path(config.model_dir) / "checkpoints",
        "outputs",
        "explanations"
    ]
    for d in directories:
        Path(d).mkdir(parents=True, exist_ok=True)


def ensure_design_data(config: Config) -> None:
    """Ensure static design data files exist in data/design/."""
    design_dir = Path(config.data_dir) / "design"
    required_files = ["pump_curves.json", "motor_specs.json", "well_config.json"]
    missing = [f for f in required_files if not (design_dir / f).exists()]
    if missing:
        print(f"[WARNING] Missing design data files: {missing}")
        print(f"  Expected in: {design_dir}")
    else:
        print(f"Design data OK: {design_dir}")


def generate_training_data(config: Config, n_samples: int = 50000) -> str:
    """Generate synthetic training data."""
    print("\n" + "="*60)
    print("STEP 1: Generating Synthetic Training Data")
    print("="*60)
    
    generator = SyntheticESPDataGenerator(config=config, seed=42)
    
    data_path = Path(config.data_dir) / "synthetic_esp_data.csv"
    
    df, failure_events = generator.generate_dataset(
        n_samples=n_samples,
        n_failures=100,
        failure_modes=[
            FailureMode.BEARING_WEAR,
            FailureMode.SHAFT_MISALIGNMENT,
            FailureMode.ELECTRICAL_FAULT,
            FailureMode.GAS_LOCK,
            FailureMode.SCALE_BUILDUP,
            FailureMode.OVERHEATING,
            FailureMode.SAND_PRODUCTION,
            FailureMode.CAVITATION
        ],
        save_path=str(data_path)
    )
    
    # Save failure events
    events_path = Path(config.data_dir) / "failure_events.json"
    with open(events_path, 'w') as f:
        json.dump([{"start_idx": e.start_idx, "failure_idx": e.failure_idx, 
                   "failure_mode": e.failure_mode.value, "severity": e.severity} 
                  for e in failure_events], f, indent=2)
    
    # Generate maintenance history
    maintenance_path = Path(config.data_dir) / "maintenance_history.json"
    generator.generate_maintenance_history(
        failure_events=failure_events,
        start_date="2024-01-01",
        save_path=str(maintenance_path)
    )

    print(f"\nGenerated {len(df)} samples with {len(failure_events)} failure events")
    print(f"Data saved to: {data_path}")

    return str(data_path)


def train_lstm_predictor(config: Config, data_path: str) -> None:
    """Train the LSTM failure predictor."""
    print("\n" + "="*60)
    print("STEP 2: Training LSTM Predictor")
    print("="*60)
    
    import torch
    from src.models.lstm_predictor import LSTMPredictor
    
    # Load and preprocess data
    # Load and preprocess data
    loader = ESPDataLoader(config=config)
    df = loader.load_csv(data_path)
    
    preprocessor = DataPreprocessor(config=config)
    df_processed = preprocessor.fit_transform(df)
    
    # Create sequences
    X, y = preprocessor.create_sequences(
        df_processed,
        sequence_length=config.training.sequence_length,
        prediction_horizon=config.training.prediction_horizon
    )
    
    print(f"Created {len(X)} sequences")

    # Compute class weight for imbalanced dataset (capped to avoid over-correction)
    n_positive = float(y.sum())
    n_negative = float(len(y) - n_positive)
    raw_weight = n_negative / (n_positive + 1e-6)
    pos_weight = min(raw_weight, 3.0)  # Cap to prevent aggressive over-prediction
    print(f"Class balance: {n_positive:.0f}/{len(y)} positive ({100*n_positive/len(y):.1f}%), raw_weight={raw_weight:.2f}, pos_weight(capped)={pos_weight:.2f}")

    # Split data
    train_loader, val_loader, test_loader = preprocessor.create_dataloaders(
        X, y,
        batch_size=config.training.batch_size
    )

    # Initialize model with class weight
    input_dim = X.shape[2]
    model = LSTMPredictor(input_dim=input_dim, config=config, pos_weight=pos_weight)
    
    # Train
    history = model.train(
        train_loader, val_loader,
        epochs=config.training.epochs
    )
    
    # Evaluate
    loss, accuracy = model.evaluate(test_loader)
    print(f"\nTest Results:")
    print(f"  test_loss: {loss:.4f}")
    print(f"  test_accuracy: {accuracy:.4f}")
    
    # Save
    model.save(str(Path(config.model_dir) / "lstm_predictor.pt"))
    
    # Save preprocessor
    preprocessor.save(str(Path(config.model_dir) / "preprocessor.pkl"))


def train_autoencoder(config: Config, data_path: str) -> None:
    """Train the LSTM Autoencoder for anomaly detection."""
    print("\n" + "="*60)
    print("STEP 3: Training LSTM Autoencoder")
    print("="*60)
    
    import torch
    from src.models.lstm_autoencoder import AnomalyDetector
    
    # Load data
    loader = ESPDataLoader(config=config)
    df = loader.load_csv(data_path)

    # CRITICAL: Filter normal data BEFORE fitting the scaler
    # to prevent anomalous data from contaminating normalization statistics
    if 'failure_label' in df.columns:
        normal_df_raw = df[df['failure_label'] == 0].copy()
        print(f"Filtered {len(normal_df_raw)}/{len(df)} normal samples for autoencoder training")
    else:
        normal_df_raw = df.copy()

    preprocessor = DataPreprocessor(config=config)
    normal_df_processed = preprocessor.fit_transform(normal_df_raw)

    # Create sequences from normal data only
    X = preprocessor.create_autoencoder_sequences(
        normal_df_processed,
        sequence_length=config.training.sequence_length
    )

    print(f"Created {len(X)} normal operation sequences")
    
    # Create dummy labels for compatibility
    y = torch.zeros(len(X))
    
    # Split
    train_loader, val_loader, _ = preprocessor.create_dataloaders(
        X, y,
        batch_size=config.training.batch_size
    )
    
    # Initialize model
    input_dim = X.shape[2]
    detector = AnomalyDetector(input_dim=input_dim, config=config)
    
    # Train
    history = detector.train(
        train_loader, val_loader,
        epochs=config.training.epochs
    )
    
    # Save
    detector.save(str(Path(config.model_dir) / "anomaly_detector.pt"))


def train_drl_optimizer(config: Config, total_timesteps: int = 100000) -> None:
    """Train the DRL optimizer."""
    print("\n" + "="*60)
    print("STEP 4: Training DRL Optimizer")
    print("="*60)
    
    try:
        from src.models.drl_optimizer import DRLOptimizer
        
        optimizer = DRLOptimizer(config=config)
        optimizer.create_environment()
        
        optimizer.train(
            total_timesteps=total_timesteps,
            save_path=str(Path(config.model_dir) / "drl_optimizer.zip")
        )
        
        # Evaluate
        metrics = optimizer.evaluate(n_episodes=10)
        print("\nDRL Evaluation:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")
            
    except ImportError as e:
        print(f"Skipping DRL training: {e}")
        print("Install stable-baselines3 to enable DRL training.")


def train_pinn(config: Config, data_path: str) -> None:
    """Train the Physics-Informed Neural Network."""
    print("\n" + "="*60)
    print("STEP 5: Training Physics-Informed Neural Network")
    print("="*60)
    
    import torch
    from src.models.pinn_model import PINN, PINNTrainer, ESPPhysicsLoss
    from src.data.data_loader import ESPDataLoader
    
    # Load data
    loader = ESPDataLoader(config=config)
    df = loader.load_csv(data_path)
    
    # Select input and output columns
    input_cols = config.training.sensor_channels
    output_cols = ['flow_rate', 'discharge_pressure', 'power_consumption']

    # Filter available columns
    input_cols = [c for c in input_cols if c in df.columns]
    output_cols = [c for c in output_cols if c in df.columns]

    if len(output_cols) < 2:
        output_cols = ['flow_rate', 'discharge_pressure', 'power_consumption']
        output_cols = [c for c in output_cols if c in df.columns]

    print(f"PINN inputs: {len(input_cols)} features, outputs: {output_cols}")

    X = df[input_cols].values.astype('float32')
    y = df[output_cols].values.astype('float32')

    # Normalize inputs and outputs for stable training
    from sklearn.preprocessing import StandardScaler
    input_scaler = StandardScaler()
    output_scaler = StandardScaler()

    X_scaled = input_scaler.fit_transform(X)
    y_scaled = output_scaler.fit_transform(y)

    print(f"Output scales - mean: {output_scaler.mean_}, std: {output_scaler.scale_}")

    # Split
    n_train = int(0.7 * len(X_scaled))
    n_val = int(0.15 * len(X_scaled))

    X_train, y_train = X_scaled[:n_train], y_scaled[:n_train]
    X_val, y_val = X_scaled[n_train:n_train+n_val], y_scaled[n_train:n_train+n_val]

    # Create dataloaders
    from torch.utils.data import TensorDataset, DataLoader

    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.FloatTensor(y_train)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val),
        torch.FloatTensor(y_val)
    )

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256)

    # Store scaler info for physics loss denormalization
    output_mean = torch.FloatTensor(output_scaler.mean_)
    output_std = torch.FloatTensor(output_scaler.scale_)
    input_mean = torch.FloatTensor(input_scaler.mean_)
    input_std = torch.FloatTensor(input_scaler.scale_)

    # Frequency column index for physics loss
    freq_idx = input_cols.index('frequency') if 'frequency' in input_cols else 8

    # Initialize model
    model = PINN(
        input_dim=len(input_cols),
        output_dim=len(output_cols),
        hidden_layers=config.model.pinn_hidden_layers
    )

    physics_loss = ESPPhysicsLoss(esp_config=config.esp)
    trainer = PINNTrainer(
        model, physics_loss, config=config,
        output_mean=output_mean, output_std=output_std,
        input_mean=input_mean, input_std=input_std,
        freq_idx=freq_idx
    )

    # Train
    history = trainer.train(train_loader, val_loader, epochs=100)

    # Check physics compliance (on denormalized data)
    compliance = trainer.check_physics_compliance(val_loader)

    # Save (include scalers)
    trainer.save(
        str(Path(config.model_dir) / "pinn_model.pt"),
        extra_data={
            'input_scaler_mean': input_scaler.mean_,
            'input_scaler_scale': input_scaler.scale_,
            'output_scaler_mean': output_scaler.mean_,
            'output_scaler_scale': output_scaler.scale_,
            'output_cols': output_cols,
            'input_cols': input_cols
        }
    )


def main():
    """Main training pipeline."""
    parser = argparse.ArgumentParser(description="ESP AI Model Training")
    parser.add_argument("--skip-datagen", action="store_true",
                       help="Skip data generation (use existing data)")
    parser.add_argument("--skip-lstm", action="store_true",
                       help="Skip LSTM predictor training")
    parser.add_argument("--skip-autoencoder", action="store_true",
                       help="Skip autoencoder training")
    parser.add_argument("--skip-drl", action="store_true",
                       help="Skip DRL training")
    parser.add_argument("--skip-pinn", action="store_true",
                       help="Skip PINN training")
    parser.add_argument("--data-samples", type=int, default=50000,
                       help="Number of training samples")
    parser.add_argument("--drl-timesteps", type=int, default=100000,
                       help="DRL training timesteps")
    
    args = parser.parse_args()
    
    print("="*60)
    print("ESP AI OPTIMIZATION - TRAINING PIPELINE")
    print("="*60)
    print(f"Started at: {datetime.now()}")
    
    # Load config
    config = Config()
    
    # Check for GPU
    import torch
    if torch.cuda.is_available():
        print(f"\n[INFO] CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
        config.training.device = "cuda"
    else:
        print("\n[WARNING] CUDA not available. Using CPU.")
        config.training.device = "cpu"
        
    setup_paths(config)
    ensure_design_data(config)

    # Data generation
    data_path = Path(config.data_dir) / "synthetic_esp_data.csv"
    
    if not args.skip_datagen or not data_path.exists():
        data_path = generate_training_data(config, args.data_samples)
    else:
        print(f"\nUsing existing data: {data_path}")
        data_path = str(data_path)
    
    # Train models
    if not args.skip_lstm:
        train_lstm_predictor(config, data_path)
    
    if not args.skip_autoencoder:
        train_autoencoder(config, data_path)
    
    if not args.skip_drl:
        train_drl_optimizer(config, args.drl_timesteps)
    
    if not args.skip_pinn:
        train_pinn(config, data_path)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Finished at: {datetime.now()}")
    print(f"\nModels saved to: {config.model_dir}")


if __name__ == "__main__":
    main()
