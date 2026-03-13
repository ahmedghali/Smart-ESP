"""
LSTM-Autoencoder for ESP Anomaly Detection
===========================================
Autoencoder-based anomaly detection using reconstruction error.
Reduces false alarm rate by learning normal operation patterns.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Tuple, Optional, Dict, List
import numpy as np
from pathlib import Path
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.config.config import Config


class LSTMEncoder(nn.Module):
    """LSTM Encoder for sequence compression."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        latent_dim: int = 32,
        num_layers: int = 2,
        dropout: float = 0.2
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Project to latent space
        self.fc_latent = nn.Linear(hidden_dim, latent_dim)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Encode input sequence to latent representation.
        
        Args:
            x: Input tensor (batch_size, seq_len, input_dim)
            
        Returns:
            Tuple of (latent, (h_n, c_n))
        """
        # Encode sequence
        outputs, (h_n, c_n) = self.lstm(x)
        
        # Use last hidden state for latent representation
        latent = self.fc_latent(h_n[-1])
        
        return latent, (h_n, c_n)


class LSTMDecoder(nn.Module):
    """LSTM Decoder for sequence reconstruction."""
    
    def __init__(
        self,
        output_dim: int,
        hidden_dim: int = 128,
        latent_dim: int = 32,
        num_layers: int = 2,
        dropout: float = 0.2
    ):
        super().__init__()
        
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        
        # Expand latent to hidden
        self.fc_hidden = nn.Linear(latent_dim, hidden_dim)
        
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Output projection
        self.fc_out = nn.Linear(hidden_dim, output_dim)
    
    def forward(
        self,
        latent: torch.Tensor,
        seq_len: int
    ) -> torch.Tensor:
        """
        Decode latent representation to sequence.
        
        Args:
            latent: Latent tensor (batch_size, latent_dim)
            seq_len: Length of output sequence
            
        Returns:
            Reconstructed sequence (batch_size, seq_len, output_dim)
        """
        batch_size = latent.size(0)
        
        # Expand latent to hidden dimension
        hidden = self.fc_hidden(latent)
        
        # Repeat hidden state for each time step
        decoder_input = hidden.unsqueeze(1).repeat(1, seq_len, 1)
        
        # Initialize hidden states
        h_0 = hidden.unsqueeze(0).repeat(self.num_layers, 1, 1)
        c_0 = torch.zeros_like(h_0)
        
        # Decode
        outputs, _ = self.lstm(decoder_input, (h_0, c_0))
        
        # Project to output dimension
        reconstructed = self.fc_out(outputs)
        
        return reconstructed


class LSTMAutoencoder(nn.Module):
    """
    Complete LSTM Autoencoder for ESP anomaly detection.
    
    Architecture:
    - LSTM Encoder compresses sequence to latent space
    - LSTM Decoder reconstructs sequence from latent
    - Anomaly score based on reconstruction error
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        latent_dim: int = 32,
        num_layers: int = 2,
        dropout: float = 0.2
    ):
        """
        Initialize autoencoder.
        
        Args:
            input_dim: Number of input features
            hidden_dim: LSTM hidden dimension
            latent_dim: Latent space dimension
            num_layers: Number of LSTM layers
            dropout: Dropout rate
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        
        self.encoder = LSTMEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            num_layers=num_layers,
            dropout=dropout
        )
        
        self.decoder = LSTMDecoder(
            output_dim=input_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            num_layers=num_layers,
            dropout=dropout
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch_size, seq_len, input_dim)
            
        Returns:
            Tuple of (reconstructed, latent)
        """
        seq_len = x.size(1)
        
        # Encode
        latent, _ = self.encoder(x)
        
        # Decode
        reconstructed = self.decoder(latent, seq_len)
        
        return reconstructed, latent
    
    def get_reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculate reconstruction error (anomaly score).
        
        Args:
            x: Input tensor
            
        Returns:
            Reconstruction error per sample
        """
        reconstructed, _ = self.forward(x)
        
        # MSE per sample
        error = F.mse_loss(reconstructed, x, reduction='none')
        error = error.mean(dim=[1, 2])  # Mean over sequence and features
        
        return error


class AnomalyDetector:
    """
    Complete anomaly detection system using LSTM-Autoencoder.
    
    Features:
    - Adaptive threshold based on training data
    - Real-time anomaly scoring
    - Explainability through feature-wise reconstruction error
    """
    
    def __init__(
        self,
        input_dim: int,
        config: Optional[Config] = None,
        device: Optional[str] = None
    ):
        """
        Initialize detector.
        
        Args:
            input_dim: Number of input features
            config: Configuration object
            device: Device to use
        """
        self.config = config or Config()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = LSTMAutoencoder(
            input_dim=input_dim,
            hidden_dim=self.config.model.autoencoder_hidden_sizes[0],
            latent_dim=self.config.model.autoencoder_latent_dim,
            num_layers=2,
            dropout=0.2
        ).to(self.device)
        
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.training.learning_rate
        )
        
        # Threshold for anomaly detection (set during training)
        self.threshold: Optional[float] = None
        
        # History
        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": []
        }
    
    def train_epoch(self, dataloader: DataLoader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        n_samples = 0
        
        for batch_x, _ in dataloader:
            batch_x = batch_x.to(self.device)
            
            self.optimizer.zero_grad()
            
            reconstructed, _ = self.model(batch_x)
            loss = self.criterion(reconstructed, batch_x)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item() * batch_x.size(0)
            n_samples += batch_x.size(0)
        
        return total_loss / n_samples
    
    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> Tuple[float, np.ndarray]:
        """Evaluate and return loss and reconstruction errors."""
        self.model.eval()
        total_loss = 0.0
        all_errors = []
        n_samples = 0
        
        for batch_x, _ in dataloader:
            batch_x = batch_x.to(self.device)
            
            reconstructed, _ = self.model(batch_x)
            loss = self.criterion(reconstructed, batch_x)
            
            # Per-sample reconstruction error
            errors = self.model.get_reconstruction_error(batch_x)
            all_errors.append(errors.cpu().numpy())
            
            total_loss += loss.item() * batch_x.size(0)
            n_samples += batch_x.size(0)
        
        avg_loss = total_loss / n_samples
        errors = np.concatenate(all_errors)
        
        return avg_loss, errors
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: Optional[int] = None,
        threshold_percentile: Optional[float] = None
    ) -> Dict[str, List[float]]:
        """
        Train the autoencoder.
        
        Args:
            train_loader: Training data (normal operation only)
            val_loader: Validation data
            epochs: Number of epochs
            threshold_percentile: Percentile for anomaly threshold
            
        Returns:
            Training history
        """
        epochs = epochs or self.config.training.epochs
        threshold_percentile = threshold_percentile or self.config.training.anomaly_threshold_percentile
        
        print(f"Training Anomaly Detector on {self.device}")
        
        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0
        
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss, _ = self.evaluate(val_loader)
            
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            
            print(f"Epoch {epoch+1}/{epochs}: "
                  f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self.save("checkpoints/best_autoencoder.pt")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        # Set threshold based on training data reconstruction errors
        _, train_errors = self.evaluate(train_loader)
        self.threshold = np.percentile(train_errors, threshold_percentile)
        print(f"Anomaly threshold set to {self.threshold:.6f} "
              f"({threshold_percentile}th percentile)")
        
        return self.history
    
    @torch.no_grad()
    def detect_anomalies(
        self,
        x: torch.Tensor,
        threshold: Optional[float] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect anomalies in input sequences.
        
        Args:
            x: Input tensor
            threshold: Optional custom threshold
            
        Returns:
            Tuple of (is_anomaly, anomaly_scores)
        """
        self.model.eval()
        x = x.to(self.device)
        
        threshold = threshold or self.threshold
        if threshold is None:
            raise RuntimeError("Threshold not set. Train the model first.")
        
        errors = self.model.get_reconstruction_error(x).cpu().numpy()
        is_anomaly = errors > threshold
        
        return is_anomaly, errors
    
    @torch.no_grad()
    def get_feature_contributions(
        self,
        x: torch.Tensor
    ) -> np.ndarray:
        """
        Get per-feature reconstruction error for explainability.
        
        Args:
            x: Input tensor
            
        Returns:
            Feature contributions (batch_size, n_features)
        """
        self.model.eval()
        x = x.to(self.device)
        
        reconstructed, _ = self.model(x)
        
        # Per-feature mean squared error
        feature_errors = F.mse_loss(
            reconstructed, x, reduction='none'
        ).mean(dim=1).cpu().numpy()
        
        return feature_errors
    
    def save(self, path: str) -> None:
        """Save model and threshold."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "threshold": self.threshold,
            "history": self.history
        }, path)
        print(f"Anomaly detector saved to {path}")
    
    def load(self, path: str) -> "AnomalyDetector":
        """Load model and threshold."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.threshold = checkpoint["threshold"]
        self.history = checkpoint["history"]
        print(f"Anomaly detector loaded from {path}")
        return self


if __name__ == "__main__":
    # Test autoencoder
    print("Testing LSTM Autoencoder...")
    
    # Create sample data
    batch_size = 32
    seq_len = 168
    input_dim = 12
    
    x = torch.randn(batch_size, seq_len, input_dim)
    
    # Create model
    model = LSTMAutoencoder(input_dim=input_dim)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Forward pass
    reconstructed, latent = model(x)
    print(f"Reconstructed shape: {reconstructed.shape}")
    print(f"Latent shape: {latent.shape}")
    
    # Test anomaly detection
    errors = model.get_reconstruction_error(x)
    print(f"Reconstruction errors shape: {errors.shape}")
    print(f"Mean error: {errors.mean().item():.6f}")
    
    # Test detector
    detector = AnomalyDetector(input_dim=input_dim)
    detector.threshold = 0.5  # Set dummy threshold
    is_anomaly, scores = detector.detect_anomalies(x)
    print(f"Anomalies detected: {is_anomaly.sum()}")
    
    print("LSTM Autoencoder test passed!")
