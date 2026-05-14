"""
LSTM Predictor for ESP Failure Prediction
==========================================
Bidirectional LSTM with attention mechanism for predicting ESP failures
24-48 hours in advance with target 85% accuracy.
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
from src.config.config import Config, ModelConfig


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism."""
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch_size, seq_len, embed_dim)
            mask: Optional attention mask
            
        Returns:
            Tuple of (output, attention_weights)
        """
        batch_size, seq_len, _ = x.shape
        
        # Project to Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        context = torch.matmul(attn_weights, v)
        
        # Reshape back
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        output = self.out_proj(context)
        
        return output, attn_weights


class AttentionLSTM(nn.Module):
    """
    Bidirectional LSTM with Multi-Head Attention for ESP Failure Prediction.
    
    Architecture:
    - Bidirectional LSTM layers for temporal feature extraction
    - Multi-head self-attention for capturing long-range dependencies
    - Fully connected layers for classification
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
        num_heads: int = 8,
        dropout: float = 0.2,
        bidirectional: bool = True,
        num_classes: int = 1
    ):
        """
        Initialize the model.
        
        Args:
            input_dim: Number of input features
            hidden_dim: LSTM hidden dimension
            num_layers: Number of LSTM layers
            num_heads: Number of attention heads
            dropout: Dropout rate
            bidirectional: Whether to use bidirectional LSTM
            num_classes: Number of output classes (1 for binary)
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Attention dimension
        lstm_output_dim = hidden_dim * self.num_directions
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(lstm_output_dim)
        
        # Multi-head attention
        self.attention = MultiHeadAttention(
            embed_dim=lstm_output_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch_size, seq_len, input_dim)
            return_attention: Whether to return attention weights
            
        Returns:
            Tuple of (predictions, attention_weights)
        """
        batch_size = x.size(0)
        
        # Input projection
        x = self.input_proj(x)
        
        # LSTM encoding
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Layer normalization
        lstm_out = self.layer_norm(lstm_out)
        
        # Self-attention
        attn_out, attn_weights = self.attention(lstm_out)
        
        # Residual connection
        combined = lstm_out + attn_out
        
        # Global average pooling over sequence
        pooled = combined.mean(dim=1)
        
        # Classification
        output = self.classifier(pooled)
        
        if return_attention:
            return output, attn_weights
        return output, None


class LSTMPredictor:
    """
    Complete LSTM Predictor for ESP failure prediction.
    
    Includes training, evaluation, and inference capabilities.
    """
    
    def __init__(
        self,
        input_dim: int,
        config: Optional[Config] = None,
        device: Optional[str] = None,
        pos_weight: Optional[float] = None
    ):
        """
        Initialize predictor.

        Args:
            input_dim: Number of input features
            config: Configuration object
            device: Device to use ('cuda' or 'cpu')
            pos_weight: Weight for positive class to handle class imbalance
        """
        self.config = config or Config()
        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        
        # Create model
        self.model = AttentionLSTM(
            input_dim=input_dim,
            hidden_dim=self.config.model.lstm_hidden_size,
            num_layers=self.config.model.lstm_num_layers,
            num_heads=self.config.model.attention_heads,
            dropout=self.config.model.lstm_dropout,
            bidirectional=self.config.model.lstm_bidirectional
        ).to(self.device)
        
        # Loss and optimizer (with class imbalance handling)
        if pos_weight is not None:
            pw = torch.tensor([pos_weight], device=self.device)
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=pw)
            print(f"Using pos_weight={pos_weight:.2f} for class imbalance")
        else:
            self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        
        # Training history
        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": []
        }
    
    def train_epoch(self, dataloader: DataLoader) -> Tuple[float, float]:
        """
        Train for one epoch.
        
        Args:
            dataloader: Training data loader
            
        Returns:
            Tuple of (loss, accuracy)
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_x, batch_y in tqdm(dataloader, desc="Training", leave=False):
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device).unsqueeze(1)
            
            self.optimizer.zero_grad()
            
            output, _ = self.model(batch_x)
            loss = self.criterion(output, batch_y)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item() * batch_x.size(0)
            
            # Calculate accuracy
            preds = (torch.sigmoid(output) > 0.5).float()
            correct += (preds == batch_y).sum().item()
            total += batch_y.size(0)
        
        avg_loss = total_loss / total
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> Tuple[float, float]:
        """
        Evaluate model on validation/test data.
        
        Args:
            dataloader: Evaluation data loader
            
        Returns:
            Tuple of (loss, accuracy)
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device).unsqueeze(1)
            
            output, _ = self.model(batch_x)
            loss = self.criterion(output, batch_y)
            
            total_loss += loss.item() * batch_x.size(0)
            
            preds = (torch.sigmoid(output) > 0.5).float()
            correct += (preds == batch_y).sum().item()
            total += batch_y.size(0)
        
        avg_loss = total_loss / total
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: Optional[int] = None,
        early_stopping_patience: Optional[int] = None
    ) -> Dict[str, List[float]]:
        """
        Train the model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs
            early_stopping_patience: Early stopping patience
            
        Returns:
            Training history
        """
        epochs = epochs or self.config.training.epochs
        patience = early_stopping_patience or self.config.training.early_stopping_patience
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        print(f"Training on {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(epochs):
            # Train
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validate
            val_loss, val_acc = self.evaluate(val_loader)
            
            # Update scheduler
            self.scheduler.step(val_loss)
            
            # Record history
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_acc"].append(val_acc)
            
            # Print progress
            print(f"Epoch {epoch+1}/{epochs}: "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                self.save("checkpoints/best_lstm_predictor.pt")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        return self.history
    
    @torch.no_grad()
    def predict(
        self,
        x: torch.Tensor,
        return_attention: bool = False
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Make predictions.
        
        Args:
            x: Input tensor
            return_attention: Whether to return attention weights
            
        Returns:
            Tuple of (predictions, attention_weights)
        """
        self.model.eval()
        x = x.to(self.device)
        
        output, attn_weights = self.model(x, return_attention=return_attention)
        probs = torch.sigmoid(output).cpu().numpy()
        
        if attn_weights is not None:
            attn_weights = attn_weights.cpu().numpy()
        
        return probs, attn_weights
    
    def save(self, path: str) -> None:
        """Save model to file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "history": self.history,
            "config": self.config
        }, path)
    
    def load(self, path: str) -> "LSTMPredictor":
        """Load model from file."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.history = checkpoint["history"]
        print(f"Model loaded from {path}")
        return self


if __name__ == "__main__":
    # Test model
    print("Testing LSTM Predictor...")
    
    # Create sample data
    batch_size = 32
    seq_len = 168
    input_dim = 12
    
    x = torch.randn(batch_size, seq_len, input_dim)
    y = torch.randint(0, 2, (batch_size,)).float()
    
    # Create model
    model = AttentionLSTM(input_dim=input_dim)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Forward pass
    output, attn = model(x, return_attention=True)
    print(f"Output shape: {output.shape}")
    print(f"Attention shape: {attn.shape}")
    
    # Test predictor
    predictor = LSTMPredictor(input_dim=input_dim)
    probs, _ = predictor.predict(x)
    print(f"Predictions shape: {probs.shape}")
    print("LSTM Predictor test passed!")
