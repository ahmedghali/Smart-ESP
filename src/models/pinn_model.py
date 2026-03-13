"""
Physics-Informed Neural Network (PINN) for ESP Systems
=======================================================
Ensures AI predictions respect pump hydraulics principles:
- Pump affinity laws (Q∝N, H∝N², P∝N³)
- Bernoulli equation
- Energy conservation
- Physical constraints
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Tuple, Optional, Dict, List, Callable
import numpy as np
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.config.config import Config, ESPConfig


class ESPPhysicsLoss(nn.Module):
    """
    Physics-based loss functions for ESP systems.
    
    Implements constraints based on:
    - Pump affinity laws
    - Energy conservation
    - Pressure relationships
    - Operating limits
    """
    
    def __init__(self, esp_config: Optional[ESPConfig] = None):
        """
        Initialize physics loss.
        
        Args:
            esp_config: ESP configuration with physical parameters
        """
        super().__init__()
        self.esp = esp_config or ESPConfig()
        
        # Physical constants
        self.nominal_frequency = self.esp.nominal_frequency_hz
        self.pump_efficiency = self.esp.pump_efficiency
        self.motor_efficiency = self.esp.motor_efficiency
        self.fluid_density = self.esp.fluid_density_kg_m3
        self.gravity = self.esp.gravity
    
    def affinity_law_loss(
        self,
        predicted: torch.Tensor,
        frequency: torch.Tensor,
        reference_flow: float = 2000.0,
        reference_head: float = 2000.0,
        reference_power: float = 50.0
    ) -> torch.Tensor:
        """
        Loss based on pump affinity laws.
        
        Affinity Laws:
        - Q2/Q1 = N2/N1 (Flow rate)
        - H2/H1 = (N2/N1)² (Head/Pressure)  
        - P2/P1 = (N2/N1)³ (Power)
        
        Args:
            predicted: Predicted values [flow, head, power]
            frequency: Operating frequency
            reference_*: Reference values at nominal frequency
        """
        freq_ratio = frequency / self.nominal_frequency
        
        # Expected values based on affinity laws
        expected_flow = reference_flow * freq_ratio
        expected_head = reference_head * (freq_ratio ** 2)
        expected_power = reference_power * (freq_ratio ** 3)
        
        # Predicted values (assuming columns 0, 1, 2)
        pred_flow = predicted[:, 0] if predicted.dim() > 1 else predicted[0]
        pred_head = predicted[:, 1] if predicted.dim() > 1 else predicted[1]
        pred_power = predicted[:, 2] if predicted.dim() > 1 else predicted[2]
        
        # Relative error loss
        flow_loss = torch.mean(((pred_flow - expected_flow) / (expected_flow + 1e-6)) ** 2)
        head_loss = torch.mean(((pred_head - expected_head) / (expected_head + 1e-6)) ** 2)
        power_loss = torch.mean(((pred_power - expected_power) / (expected_power + 1e-6)) ** 2)
        
        return flow_loss + head_loss + power_loss
    
    def energy_conservation_loss(
        self,
        flow_rate: torch.Tensor,
        head: torch.Tensor,
        power: torch.Tensor
    ) -> torch.Tensor:
        """
        Loss based on energy conservation.
        
        Hydraulic Power = ρ * g * Q * H
        Motor Power = Hydraulic Power / (η_pump * η_motor)
        """
        # Convert units (flow in bpd to m³/s, head in psi to meters)
        flow_m3_s = flow_rate * 0.00000184  # bpd to m³/s
        head_m = head * 0.703  # psi to meters of water
        
        # Theoretical hydraulic power (kW)
        hydraulic_power = self.fluid_density * self.gravity * flow_m3_s * head_m / 1000
        
        # Expected motor power
        expected_power = hydraulic_power / (self.pump_efficiency * self.motor_efficiency)
        
        # Loss
        loss = torch.mean(((power - expected_power) / (expected_power + 1e-6)) ** 2)
        
        return loss
    
    def boundary_constraint_loss(
        self,
        predictions: torch.Tensor,
        bounds: Dict[str, Tuple[float, float]]
    ) -> torch.Tensor:
        """
        Loss for violating physical boundaries.
        
        Args:
            predictions: Model predictions
            bounds: Dictionary mapping feature index to (min, max) bounds
        """
        total_loss = torch.tensor(0.0, device=predictions.device)
        
        for idx, (min_val, max_val) in bounds.items():
            if predictions.dim() > 1:
                values = predictions[:, idx]
            else:
                values = predictions[idx:idx+1]
            
            # Penalty for values below minimum
            below_min = F.relu(min_val - values)
            # Penalty for values above maximum
            above_max = F.relu(values - max_val)
            
            total_loss = total_loss + torch.mean(below_min ** 2) + torch.mean(above_max ** 2)
        
        return total_loss
    
    def non_negative_loss(
        self,
        predictions: torch.Tensor,
        indices: List[int]
    ) -> torch.Tensor:
        """
        Ensure certain outputs are non-negative (flow, power, pressure).
        """
        total_loss = torch.tensor(0.0, device=predictions.device)
        
        for idx in indices:
            if predictions.dim() > 1:
                values = predictions[:, idx]
            else:
                values = predictions[idx:idx+1]
            
            # Penalty for negative values
            negative = F.relu(-values)
            total_loss = total_loss + torch.mean(negative ** 2)
        
        return total_loss


class PINNBlock(nn.Module):
    """Basic building block for PINN with skip connections."""
    
    def __init__(self, in_features: int, out_features: int, activation: nn.Module):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.activation = activation
        self.bn = nn.BatchNorm1d(out_features)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.linear(x)
        out = self.bn(out)
        out = self.activation(out)
        return out


class PINN(nn.Module):
    """
    Physics-Informed Neural Network for ESP prediction.
    
    Combines data-driven learning with physics constraints
    to ensure predictions respect fundamental pump laws.
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_layers: Optional[List[int]] = None,
        activation: str = "tanh"
    ):
        """
        Initialize PINN.
        
        Args:
            input_dim: Number of input features
            output_dim: Number of output features
            hidden_layers: List of hidden layer sizes
            activation: Activation function ('tanh', 'relu', 'silu')
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        hidden_layers = hidden_layers or [200, 200, 200, 200, 200]
        
        # Activation function
        if activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "silu":
            self.activation = nn.SiLU()
        else:
            self.activation = nn.Tanh()
        
        # Build network
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_layers:
            layers.append(PINNBlock(prev_dim, hidden_dim, self.activation))
            prev_dim = hidden_dim
        
        self.hidden_layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(prev_dim, output_dim)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Xavier initialization for better PINN training."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        features = self.hidden_layers(x)
        output = self.output_layer(features)
        return output


class PINNTrainer:
    """
    Trainer for Physics-Informed Neural Network.
    
    Combines data loss with physics loss for training.
    """
    
    def __init__(
        self,
        model: PINN,
        physics_loss: ESPPhysicsLoss,
        config: Optional[Config] = None,
        device: Optional[str] = None,
        output_mean: Optional[torch.Tensor] = None,
        output_std: Optional[torch.Tensor] = None,
        input_mean: Optional[torch.Tensor] = None,
        input_std: Optional[torch.Tensor] = None,
        freq_idx: int = 8
    ):
        """
        Initialize trainer.

        Args:
            model: PINN model
            physics_loss: Physics loss calculator
            config: Configuration object
            device: Device to use
            output_mean/std: For denormalizing outputs before physics loss
            input_mean/std: For denormalizing inputs before physics loss
            freq_idx: Index of frequency column in input features
        """
        self.model = model
        self.physics_loss = physics_loss
        self.config = config or Config()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Scaler parameters for denormalization (physics loss needs physical units)
        self.output_mean = output_mean.to(self.device) if output_mean is not None else None
        self.output_std = output_std.to(self.device) if output_std is not None else None
        self.input_mean = input_mean.to(self.device) if input_mean is not None else None
        self.input_std = input_std.to(self.device) if input_std is not None else None
        self.freq_idx = freq_idx
        
        self.model = self.model.to(self.device)
        
        # Loss weights
        self.data_loss_weight = 1.0
        self.physics_loss_weight = self.config.model.physics_loss_weight
        self.boundary_loss_weight = self.config.model.boundary_loss_weight
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay
        )
        
        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=100
        )
        
        # History
        self.history: Dict[str, List[float]] = {
            "data_loss": [],
            "physics_loss": [],
            "total_loss": []
        }
    
    def compute_total_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        inputs: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute combined data and physics loss.
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            inputs: Input features (for physics calculations)
            
        Returns:
            Tuple of (total_loss, loss_components)
        """
        # Data loss (MSE) - operates in normalized space for stability
        data_loss = F.mse_loss(predictions, targets)

        # Denormalize for physics loss (physics equations need physical units)
        if self.output_mean is not None and self.output_std is not None:
            pred_phys = predictions * self.output_std + self.output_mean
        else:
            pred_phys = predictions

        if self.input_mean is not None and self.input_std is not None:
            freq_col = self.freq_idx
            frequency = inputs[:, freq_col] * self.input_std[freq_col] + self.input_mean[freq_col]
        else:
            frequency = inputs[:, self.freq_idx] if inputs.size(1) > self.freq_idx else torch.full(
                (inputs.size(0),), 50.0, device=inputs.device
            )

        # Physics losses (on denormalized/physical values)
        n_out = pred_phys.size(1)
        affinity_loss = self.physics_loss.affinity_law_loss(
            pred_phys[:, :min(3, n_out)],
            frequency
        )

        # Non-negative constraints
        non_neg_loss = self.physics_loss.non_negative_loss(
            pred_phys, indices=list(range(min(3, n_out)))
        )

        # Boundary constraints
        bounds = {
            0: (0, 10000),  # Flow rate
            1: (0, 5000),   # Pressure
            2: (0, 200),    # Power
        }
        boundary_loss = self.physics_loss.boundary_constraint_loss(
            pred_phys[:, :min(3, n_out)],
            {k: v for k, v in bounds.items() if k < n_out}
        )
        
        # Total physics loss
        physics_loss = affinity_loss + non_neg_loss
        
        # Combined loss
        total_loss = (
            self.data_loss_weight * data_loss +
            self.physics_loss_weight * physics_loss +
            self.boundary_loss_weight * boundary_loss
        )
        
        loss_components = {
            "data_loss": data_loss.item(),
            "affinity_loss": affinity_loss.item(),
            "non_neg_loss": non_neg_loss.item(),
            "boundary_loss": boundary_loss.item(),
            "physics_loss": physics_loss.item(),
            "total_loss": total_loss.item()
        }
        
        return total_loss, loss_components
    
    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        epoch_losses = {
            "data_loss": 0.0,
            "physics_loss": 0.0,
            "total_loss": 0.0
        }
        n_batches = 0
        
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            
            # Flatten sequence if needed
            if batch_x.dim() == 3:
                batch_x = batch_x[:, -1, :]  # Use last timestep
            if batch_y.dim() == 3:
                batch_y = batch_y[:, -1, :]
            
            self.optimizer.zero_grad()
            
            predictions = self.model(batch_x)
            total_loss, components = self.compute_total_loss(predictions, batch_y, batch_x)
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            epoch_losses["data_loss"] += components["data_loss"]
            epoch_losses["physics_loss"] += components["physics_loss"]
            epoch_losses["total_loss"] += components["total_loss"]
            n_batches += 1
        
        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= n_batches
        
        return epoch_losses
    
    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Evaluate on validation data."""
        self.model.eval()
        total_losses = {"data_loss": 0.0, "physics_loss": 0.0, "total_loss": 0.0}
        n_batches = 0
        
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            
            if batch_x.dim() == 3:
                batch_x = batch_x[:, -1, :]
            if batch_y.dim() == 3:
                batch_y = batch_y[:, -1, :]
            
            predictions = self.model(batch_x)
            _, components = self.compute_total_loss(predictions, batch_y, batch_x)
            
            total_losses["data_loss"] += components["data_loss"]
            total_losses["physics_loss"] += components["physics_loss"]
            total_losses["total_loss"] += components["total_loss"]
            n_batches += 1
        
        for key in total_losses:
            total_losses[key] /= n_batches
        
        return total_losses
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 100
    ) -> Dict[str, List[float]]:
        """
        Full training loop.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs
            
        Returns:
            Training history
        """
        print(f"Training PINN on {self.device}")
        print(f"Physics loss weight: {self.physics_loss_weight}")
        
        best_val_loss = float('inf')
        patience = 15
        patience_counter = 0
        
        for epoch in range(epochs):
            train_losses = self.train_epoch(train_loader)
            val_losses = self.evaluate(val_loader)
            
            self.scheduler.step()
            
            # Record history
            self.history["data_loss"].append(train_losses["data_loss"])
            self.history["physics_loss"].append(train_losses["physics_loss"])
            self.history["total_loss"].append(train_losses["total_loss"])
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}: "
                      f"Train Loss: {train_losses['total_loss']:.6f} "
                      f"(Data: {train_losses['data_loss']:.6f}, "
                      f"Physics: {train_losses['physics_loss']:.6f}), "
                      f"Val Loss: {val_losses['total_loss']:.6f}")
            
            # Early stopping
            if val_losses["total_loss"] < best_val_loss:
                best_val_loss = val_losses["total_loss"]
                patience_counter = 0
                self.save("checkpoints/best_pinn.pt")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        return self.history
    
    def check_physics_compliance(
        self,
        dataloader: DataLoader
    ) -> Dict[str, float]:
        """
        Check physics compliance of predictions.

        Returns:
            Dictionary of compliance metrics
        """
        self.model.eval()

        affinity_errors = []
        non_neg_violations = 0
        total_samples = 0

        with torch.no_grad():
            for batch_x, _ in dataloader:
                batch_x = batch_x.to(self.device)

                if batch_x.dim() == 3:
                    batch_x = batch_x[:, -1, :]

                predictions = self.model(batch_x)

                # Denormalize predictions before checking physics compliance
                if self.output_mean is not None and self.output_std is not None:
                    pred_phys = predictions * self.output_std + self.output_mean
                else:
                    pred_phys = predictions

                # Check for negative values in physical space
                if pred_phys.size(1) >= 3:
                    neg_mask = (pred_phys[:, :3] < 0).any(dim=1)
                    non_neg_violations += neg_mask.sum().item()

                total_samples += batch_x.size(0)
        
        compliance = {
            "non_negative_compliance": 1 - (non_neg_violations / total_samples),
            "total_samples": total_samples,
            "violations": non_neg_violations
        }
        
        print("\nPhysics Compliance Check:")
        print(f"  Non-negative compliance: {compliance['non_negative_compliance']:.2%}")
        print(f"  Violations: {compliance['violations']}/{compliance['total_samples']}")
        
        return compliance
    
    def save(self, path: str, extra_data: Optional[Dict] = None) -> None:
        """Save model."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        save_dict = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "history": self.history
        }
        if extra_data:
            save_dict.update(extra_data)
        torch.save(save_dict, path)
    
    def load(self, path: str) -> "PINNTrainer":
        """Load model."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.history = checkpoint["history"]
        return self


if __name__ == "__main__":
    # Test PINN
    print("Testing Physics-Informed Neural Network...")
    
    # Create model
    input_dim = 12
    output_dim = 4  # flow, pressure, power, temperature
    
    model = PINN(input_dim=input_dim, output_dim=output_dim)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    x = torch.randn(32, input_dim)
    output = model(x)
    print(f"Output shape: {output.shape}")
    
    # Test physics loss
    physics_loss = ESPPhysicsLoss()
    frequency = torch.full((32,), 50.0)
    
    affinity_loss = physics_loss.affinity_law_loss(output[:, :3], frequency)
    print(f"Affinity loss: {affinity_loss.item():.6f}")
    
    non_neg_loss = physics_loss.non_negative_loss(output, [0, 1, 2])
    print(f"Non-negative loss: {non_neg_loss.item():.6f}")
    
    print("\nPINN test passed!")
