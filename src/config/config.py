"""
Configuration Module for ESP AI Optimization Framework
=======================================================
Contains all configuration parameters for models, training, and ESP physics.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
from pathlib import Path
import yaml


@dataclass
class ESPConfig:
    """ESP Physics Configuration Parameters."""
    
    # Pump operating ranges
    min_frequency_hz: float = 20.0
    max_frequency_hz: float = 60.0
    nominal_frequency_hz: float = 50.0
    
    # Pressure parameters (psi)
    min_intake_pressure: float = 100.0
    max_intake_pressure: float = 3000.0
    min_discharge_pressure: float = 500.0
    max_discharge_pressure: float = 5000.0
    
    # Temperature parameters (°C)
    min_motor_temp: float = 50.0
    max_motor_temp: float = 150.0
    critical_temp: float = 140.0
    
    # Flow parameters (bpd - barrels per day)
    min_flow_rate: float = 100.0
    max_flow_rate: float = 10000.0
    
    # Current parameters (Amps)
    min_current: float = 10.0
    max_current: float = 150.0
    
    # Vibration parameters (mm/s RMS)
    normal_vibration: float = 2.5
    warning_vibration: float = 7.0
    critical_vibration: float = 11.0

    # Voltage parameters (V)
    nominal_voltage: float = 480.0
    min_voltage: float = 432.0  # 90% of nominal
    max_voltage: float = 528.0  # 110% of nominal

    # Power factor
    nominal_power_factor: float = 0.85
    min_power_factor: float = 0.65
    max_power_factor: float = 0.95

    # Wellhead pressure (psi)
    min_wellhead_pressure: float = 50.0
    max_wellhead_pressure: float = 800.0

    # Sand rate (%)
    normal_sand_rate: float = 0.05
    warning_sand_rate: float = 0.5
    critical_sand_rate: float = 2.0

    # Physics constants
    pump_efficiency: float = 0.65
    motor_efficiency: float = 0.90
    fluid_density_kg_m3: float = 850.0  # Oil density
    gravity: float = 9.81


@dataclass
class ModelConfig:
    """Model Architecture Configuration."""
    
    # LSTM Predictor
    lstm_hidden_size: int = 256
    lstm_num_layers: int = 3
    lstm_dropout: float = 0.2
    lstm_bidirectional: bool = True
    
    # LSTM-Autoencoder
    autoencoder_latent_dim: int = 16
    autoencoder_hidden_sizes: List[int] = field(default_factory=lambda: [128, 64])
    
    # Attention mechanism
    attention_heads: int = 8
    attention_dim: int = 512
    
    # DRL Agent (PPO)
    drl_policy: str = "MlpPolicy"
    drl_learning_rate: float = 3e-4
    drl_n_steps: int = 2048
    drl_batch_size: int = 64
    drl_n_epochs: int = 10
    drl_gamma: float = 0.99
    drl_gae_lambda: float = 0.95
    drl_clip_range: float = 0.2
    
    # PINN
    pinn_hidden_layers: List[int] = field(default_factory=lambda: [200, 200, 200, 200, 200])
    pinn_activation: str = "tanh"
    physics_loss_weight: float = 1.0
    boundary_loss_weight: float = 0.5


@dataclass
class TrainingConfig:
    """Training Configuration."""
    
    # General training
    batch_size: int = 512
    epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    early_stopping_patience: int = 10
    
    # Data configuration
    sequence_length: int = 168  # 7 days of hourly data
    prediction_horizon: int = 48  # 48 hours ahead
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15
    
    # Sensor channels — 11 real-deployable channels matching DHM standard sensors
    # Removed: vibration_x/y/z, flow_rate, casing_pressure, power_factor,
    #          wellhead_pressure, sand_rate  (not universally available on ESP DHM)
    # Added:   vibration (scalar g), current_leakage (mA), differential_pressure (psi)
    sensor_channels: List[str] = field(default_factory=lambda: [
        "motor_temperature",       # Motor temperature (°C)
        "intake_pressure",         # Pump intake pressure (psi)
        "discharge_pressure",      # Pump discharge pressure (psi)
        "motor_current",           # Motor current draw (A)
        "frequency",               # VSD drive frequency (Hz)
        "fluid_temperature",       # Intake fluid temperature (°C)
        "voltage",                 # Supply voltage (V)
        "vibration",               # Vibration magnitude scalar (g)
        "current_leakage",         # Active current leakage (mA) — insulation health
        "power_consumption",       # Derived: √3 × V × I × 0.85 (kW)
        "differential_pressure",   # Derived: discharge − intake (psi)
    ])
    
    # Anomaly detection
    anomaly_threshold_percentile: float = 95.0
    
    # Device
    device: str = "cuda"  # or "cpu"
    num_workers: int = 4
    
    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    save_every_n_epochs: int = 5


@dataclass
class Config:
    """Main Configuration Class."""
    
    esp: ESPConfig = field(default_factory=ESPConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    
    # Paths
    data_dir: str = "data"
    raw_data_dir: str = "data/raw"
    processed_data_dir: str = "data/processed"
    synthetic_data_dir: str = "data/synthetic"
    model_dir: str = "models"
    logs_dir: str = "logs"
    
    # Project info
    project_name: str = "ESP_AI_Optimization"
    version: str = "1.0.0"
    
    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """Load configuration from YAML file."""
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)
    
    def to_yaml(self, path: str) -> None:
        """Save configuration to YAML file."""
        import dataclasses
        
        def to_dict(obj):
            if dataclasses.is_dataclass(obj):
                return {k: to_dict(v) for k, v in dataclasses.asdict(obj).items()}
            return obj
        
        with open(path, 'w') as f:
            yaml.dump(to_dict(self), f, default_flow_style=False)
    
    def create_directories(self) -> None:
        """Create necessary directories."""
        dirs = [
            self.data_dir, self.raw_data_dir, self.processed_data_dir,
            self.synthetic_data_dir, self.model_dir, self.logs_dir,
            self.training.checkpoint_dir
        ]
        for d in dirs:
            Path(d).mkdir(parents=True, exist_ok=True)


# Default configuration instance
default_config = Config()


if __name__ == "__main__":
    # Create and display default configuration
    config = Config()
    print(f"Project: {config.project_name} v{config.version}")
    print(f"\nESP Configuration:")
    print(f"  Frequency range: {config.esp.min_frequency_hz}-{config.esp.max_frequency_hz} Hz")
    print(f"  Temperature range: {config.esp.min_motor_temp}-{config.esp.max_motor_temp} °C")
    print(f"\nModel Configuration:")
    print(f"  LSTM layers: {config.model.lstm_num_layers}, hidden: {config.model.lstm_hidden_size}")
    print(f"  Attention heads: {config.model.attention_heads}")
    print(f"\nTraining Configuration:")
    print(f"  Batch size: {config.training.batch_size}")
    print(f"  Epochs: {config.training.epochs}")
    print(f"  Sensors: {len(config.training.sensor_channels)}")
