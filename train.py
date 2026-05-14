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


def train_lstm_predictor(config: Config, data_path: str) -> dict:
    """Train the LSTM failure predictor. Returns metrics dict."""
    print("\n" + "="*60)
    print("STEP 2: Training LSTM Predictor")
    print("="*60)

    import torch
    from src.models.lstm_predictor import LSTMPredictor
    from sklearn.metrics import precision_score, recall_score, f1_score

    loader = ESPDataLoader(config=config)
    df = loader.load_csv(data_path)

    preprocessor = DataPreprocessor(config=config)
    df_processed = preprocessor.fit_transform(df)

    X, y = preprocessor.create_sequences(
        df_processed,
        sequence_length=config.training.sequence_length,
        prediction_horizon=config.training.prediction_horizon
    )

    print(f"Created {len(X)} sequences")

    n_positive = float(y.sum())
    n_negative = float(len(y) - n_positive)
    raw_weight = n_negative / (n_positive + 1e-6)
    pos_weight = min(raw_weight, 3.0)
    print(f"Class balance: {n_positive:.0f}/{len(y)} positive ({100*n_positive/len(y):.1f}%), pos_weight(capped)={pos_weight:.2f}")

    train_loader, val_loader, test_loader = preprocessor.create_dataloaders(
        X, y, batch_size=config.training.batch_size
    )

    input_dim = X.shape[2]
    model = LSTMPredictor(input_dim=input_dim, config=config, pos_weight=pos_weight)
    history = model.train(train_loader, val_loader, epochs=config.training.epochs)
    loss, accuracy = model.evaluate(test_loader)

    # Compute precision / recall / F1 / false alarm rate
    # LSTMPredictor is a wrapper class (not nn.Module) — access inner .model
    device = next(model.model.parameters()).device
    model.model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            logits = model.model(xb.to(device))[0].squeeze(-1)
            preds  = (torch.sigmoid(logits) > 0.5).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(yb.numpy().tolist())

    import numpy as np
    all_preds  = [int(p) for p in all_preds]
    all_labels = [int(l) for l in all_labels]
    precision  = precision_score(all_labels, all_preds, zero_division=0)
    recall     = recall_score(all_labels, all_preds, zero_division=0)
    f1         = f1_score(all_labels, all_preds, zero_division=0)
    far        = 1 - precision  # false alarm rate

    metrics = {
        "model":           "LSTM Predictor",
        "test_loss":       round(loss, 4),
        "accuracy":        round(accuracy, 4),
        "precision":       round(precision, 4),
        "recall":          round(recall, 4),
        "f1":              round(f1, 4),
        "false_alarm_rate":round(far, 4),
        "n_sequences":     len(X),
        "input_channels":  input_dim,
    }

    print(f"\n{'─'*40}")
    print(f"  LSTM PREDICTOR — TEST RESULTS")
    print(f"{'─'*40}")
    print(f"  Accuracy        : {accuracy:.4f}  ({accuracy*100:.1f}%)")
    print(f"  Precision       : {precision:.4f}  ({precision*100:.1f}%)")
    print(f"  Recall          : {recall:.4f}  ({recall*100:.1f}%)")
    print(f"  F1 Score        : {f1:.4f}")
    print(f"  False Alarm Rate: {far:.4f}  ({far*100:.1f}%)")
    print(f"  Test Loss       : {loss:.4f}")
    print(f"{'─'*40}")

    model.save(str(Path(config.model_dir) / "lstm_predictor.pt"))
    preprocessor.save(str(Path(config.model_dir) / "preprocessor.pkl"))
    return metrics


def train_autoencoder(config: Config, data_path: str) -> dict:
    """Train the LSTM Autoencoder for anomaly detection. Returns metrics dict."""
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
    
    history = detector.train(train_loader, val_loader, epochs=config.training.epochs)

    # Extract final val loss from history
    val_loss = history.get("val_loss", [None])[-1] if history else None
    train_loss = history.get("train_loss", [None])[-1] if history else None

    metrics = {
        "model":       "LSTM Autoencoder",
        "train_loss":  round(float(train_loss), 4) if train_loss else None,
        "val_loss":    round(float(val_loss), 4)   if val_loss   else None,
        "n_sequences": len(X),
        "input_channels": X.shape[2],
        "latent_dim":  16,
    }

    print(f"\n{'─'*40}")
    print(f"  AUTOENCODER — TRAINING RESULTS")
    print(f"{'─'*40}")
    print(f"  Train Loss (reconstruction): {train_loss:.4f}" if train_loss else "  Train Loss: N/A")
    print(f"  Val   Loss (reconstruction): {val_loss:.4f}"   if val_loss   else "  Val Loss: N/A")
    print(f"  Normal sequences used: {len(X):,}")
    print(f"{'─'*40}")

    detector.save(str(Path(config.model_dir) / "anomaly_detector.pt"))
    return metrics


def train_drl_optimizer(config: Config, total_timesteps: int = 100000) -> dict:
    """Train the DRL optimizer. Returns metrics dict."""
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

        drl_metrics = optimizer.evaluate(n_episodes=10)

        metrics = {"model": "DRL Optimizer", **{k: round(float(v), 4) for k, v in drl_metrics.items()}}

        print(f"\n{'─'*40}")
        print(f"  DRL OPTIMIZER — EVALUATION (10 episodes)")
        print(f"{'─'*40}")
        for k, v in drl_metrics.items():
            print(f"  {k}: {v:.4f}")
        print(f"{'─'*40}")
        return metrics

    except ImportError as e:
        print(f"Skipping DRL training: {e}")
        return {"model": "DRL Optimizer", "status": "skipped"}


def train_pinn(config: Config, data_path: str) -> dict:
    """Train the Physics-Informed Neural Network. Returns metrics dict."""
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
    # PINN predicts the 3 physically constrained outputs from the 8 non-derived inputs
    non_derived = [
        "motor_temperature", "intake_pressure", "motor_current",
        "frequency", "fluid_temperature", "voltage", "vibration", "current_leakage"
    ]
    input_cols  = [c for c in non_derived if c in df.columns]
    # PINN outputs: discharge_pressure, differential_pressure, power_consumption
    # (all three obey affinity laws and energy conservation)
    output_cols = ["discharge_pressure", "differential_pressure", "power_consumption"]
    output_cols = [c for c in output_cols if c in df.columns]

    if len(output_cols) < 2:
        print("[WARN] PINN outputs not found — falling back to available columns")
        output_cols = [c for c in config.training.sensor_channels if c in df.columns][:3]

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

    history = trainer.train(train_loader, val_loader, epochs=100)
    compliance = trainer.check_physics_compliance(val_loader)

    # R² on validation set
    import torch, numpy as np
    pinn_device = next(trainer.model.parameters()).device
    val_preds, val_targets = [], []
    trainer.model.eval()
    with torch.no_grad():
        for xb, yb in val_loader:
            pred = trainer.model(xb.to(pinn_device))
            val_preds.append(pred.cpu().numpy())
            val_targets.append(yb.numpy())
    val_preds   = np.concatenate(val_preds)
    val_targets = np.concatenate(val_targets)
    ss_res = ((val_targets - val_preds) ** 2).sum(axis=0)
    ss_tot = ((val_targets - val_targets.mean(axis=0)) ** 2).sum(axis=0)
    r2_per_output = 1 - ss_res / (ss_tot + 1e-8)
    r2_mean = float(r2_per_output.mean())
    mape = float(np.mean(np.abs((val_targets - val_preds) / (np.abs(val_targets) + 1e-8))) * 100)

    metrics = {
        "model":        "PINN",
        "r2_mean":      round(r2_mean, 4),
        "mape_pct":     round(mape, 4),
        "physics_compliance": round(float(compliance.get("overall_compliance", 0)), 4),
        "outputs":      str(output_cols),
        "n_inputs":     len(input_cols),
    }

    print(f"\n{'─'*40}")
    print(f"  PINN — VALIDATION RESULTS")
    print(f"{'─'*40}")
    print(f"  R² (mean over outputs)  : {r2_mean:.4f}")
    print(f"  MAPE                    : {mape:.2f}%")
    for i, col in enumerate(output_cols):
        print(f"  R² [{col}]: {r2_per_output[i]:.4f}")
    print(f"  Physics compliance      : {compliance}")
    print(f"{'─'*40}")

    trainer.save(
        str(Path(config.model_dir) / "pinn_model.pt"),
        extra_data={
            'input_scaler_mean':  input_scaler.mean_,
            'input_scaler_scale': input_scaler.scale_,
            'output_scaler_mean': output_scaler.mean_,
            'output_scaler_scale':output_scaler.scale_,
            'output_cols': output_cols,
            'input_cols':  input_cols
        }
    )
    return metrics


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
    parser.add_argument("--drl-timesteps", type=int, default=300000,
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
    
    import csv, torch
    all_metrics = []
    start_time  = datetime.now()

    if not args.skip_lstm:
        m = train_lstm_predictor(config, data_path)
        if m: all_metrics.append(m)

    if not args.skip_autoencoder:
        m = train_autoencoder(config, data_path)
        if m: all_metrics.append(m)

    if not args.skip_drl:
        m = train_drl_optimizer(config, args.drl_timesteps)
        if m: all_metrics.append(m)

    if not args.skip_pinn:
        m = train_pinn(config, data_path)
        if m: all_metrics.append(m)

    end_time     = datetime.now()
    elapsed      = str(end_time - start_time).split(".")[0]  # HH:MM:SS

    # ── Save results to CSV ──────────────────────────────────────────────
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    timestamp_str = start_time.strftime("%Y%m%d_%H%M%S")
    csv_path = results_dir / f"training_{timestamp_str}.csv"

    if all_metrics:
        all_keys = sorted({k for m in all_metrics for k in m.keys()})
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["run_timestamp", "elapsed"] + all_keys)
            writer.writeheader()
            for m in all_metrics:
                row = {"run_timestamp": timestamp_str, "elapsed": elapsed}
                row.update(m)
                writer.writerow(row)
        # Also append to master summary
        master_path = results_dir / "all_runs.csv"
        write_header = not master_path.exists()
        with open(master_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["run_timestamp", "elapsed"] + all_keys)
            if write_header:
                writer.writeheader()
            for m in all_metrics:
                row = {"run_timestamp": timestamp_str, "elapsed": elapsed}
                row.update(m)
                writer.writerow(row)

    # ── Final summary table ───────────────────────────────────────────────
    print("\n" + "="*60)
    print("  TRAINING COMPLETE — FINAL SUMMARY")
    print("="*60)
    print(f"  Started  : {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Finished : {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Duration : {elapsed}")
    print(f"  Device   : {config.training.device.upper()}")
    if torch.cuda.is_available():
        mem = torch.cuda.max_memory_allocated() / 1024**3
        print(f"  GPU Mem  : {mem:.2f} GB peak")
    print()
    print(f"  {'MODEL':<25} {'KEY METRIC':<20} {'VALUE'}")
    print(f"  {'─'*55}")
    for m in all_metrics:
        name = m.get("model", "?")
        if name == "LSTM Predictor":
            kv = f"Precision={m.get('precision','?')}  FAR={m.get('false_alarm_rate','?')}"
        elif name == "LSTM Autoencoder":
            kv = f"Val Loss={m.get('val_loss','?')}"
        elif name == "DRL Optimizer":
            kv = f"Avg Reward={m.get('mean_reward', m.get('average_reward','?'))}"
        elif name == "PINN":
            kv = f"R²={m.get('r2_mean','?')}  MAPE={m.get('mape_pct','?')}%"
        else:
            kv = str(m)
        print(f"  {name:<25} {kv}")
    print()
    print(f"  Models saved to : {config.model_dir}/")
    print(f"  Results saved to: {csv_path}")
    print(f"  Master log      : {results_dir}/all_runs.csv")
    print("="*60)


if __name__ == "__main__":
    main()
