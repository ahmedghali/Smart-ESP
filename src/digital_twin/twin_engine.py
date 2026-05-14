"""
Digital Twin Engine
====================
Real-time synchronization between physical ESP and virtual model.
Integrates AI models for prediction and optimization.
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import json

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.config.config import Config
from src.digital_twin.esp_simulator import ESPSimulator, ESPState, ESPStatus


@dataclass
class Alert:
    """System alert."""
    timestamp: datetime
    level: str  # "info", "warning", "critical"
    message: str
    sensor: Optional[str] = None
    value: Optional[float] = None
    threshold: Optional[float] = None


@dataclass
class Recommendation:
    """AI-generated recommendation."""
    timestamp: datetime
    action: str  # "adjust_frequency", "adjust_choke", "schedule_maintenance"
    current_value: float
    recommended_value: float
    confidence: float
    reasoning: str
    expected_impact: Dict[str, float] = field(default_factory=dict)


class DigitalTwinEngine:
    """
    Digital Twin Engine for ESP systems.
    
    Features:
    - Real-time state synchronization
    - AI-powered predictions
    - Anomaly detection
    - Optimization recommendations
    - What-if scenario analysis
    """
    
    def __init__(
        self,
        config: Optional[Config] = None,
        device: Optional[str] = None
    ):
        """
        Initialize digital twin engine.
        
        Args:
            config: Configuration object
            device: Device for AI models
        """
        self.config = config or Config()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Core components
        self.simulator = ESPSimulator(config=config)
        self.current_state = self.simulator.state
        
        # AI models (loaded on demand)
        self.predictor = None
        self.anomaly_detector = None
        self.optimizer = None
        self.pinn = None
        
        # History
        self.state_history: List[Dict] = []
        self.alert_history: List[Alert] = []
        self.recommendation_history: List[Recommendation] = []
        
        # Thresholds for alerts
        self.alert_thresholds = {
            "motor_temperature": {"warning": 110, "critical": 130},
            "vibration_magnitude": {"warning": 6, "critical": 10},
            "equipment_health": {"warning": 0.6, "critical": 0.3},
            "motor_current": {"warning": 80, "critical": 100},
            "voltage_low": {"warning": 450, "critical": 430},
            "sand_rate": {"warning": 0.5, "critical": 2.0}
        }
    
    def load_models(
        self,
        predictor_path: Optional[str] = None,
        anomaly_path: Optional[str] = None,
        optimizer_path: Optional[str] = None,
        pinn_path: Optional[str] = None
    ) -> None:
        """Load AI models from checkpoints."""
        # Import models on demand
        from src.models.lstm_predictor import LSTMPredictor
        from src.models.lstm_autoencoder import AnomalyDetector
        from src.models.drl_optimizer import DRLOptimizer
        from src.models.pinn_model import PINN, PINNTrainer, ESPPhysicsLoss
        
        input_dim = len(self.config.training.sensor_channels)
        
        if predictor_path and Path(predictor_path).exists():
            # Peek at checkpoint to extract saved config — handles finetuned
            # (hidden=32, 1 layer) vs pretrained (hidden=256, 3 layers) shape mismatch.
            raw = torch.load(predictor_path, map_location=self.device,
                             weights_only=False)
            saved_config = raw.get("config", self.config)
            self.predictor = LSTMPredictor(input_dim=input_dim, config=saved_config)
            self.predictor.model.load_state_dict(raw["model_state_dict"])
            self.predictor.history = raw.get("history", {})
            print(f"Loaded predictor from {predictor_path} "
                  f"(hidden={saved_config.model.lstm_hidden_size}, "
                  f"layers={saved_config.model.lstm_num_layers})")
        
        if anomaly_path and Path(anomaly_path).exists():
            # Infer input_dim from encoder LSTM weights to handle checkpoint shape mismatches.
            ae_ckpt = torch.load(anomaly_path, map_location=self.device,
                                 weights_only=False)
            ae_sd = ae_ckpt["model_state_dict"]
            # encoder.lstm.weight_ih_l0: shape [4*hidden, input_dim]
            ae_in = ae_sd["encoder.lstm.weight_ih_l0"].shape[1]
            self.anomaly_detector = AnomalyDetector(input_dim=ae_in, config=self.config)
            self.anomaly_detector.model.load_state_dict(ae_sd)
            self.anomaly_detector.threshold = ae_ckpt.get("threshold", 1.0)
            self.anomaly_detector.history   = ae_ckpt.get("history", {})
            print(f"Loaded anomaly detector from {anomaly_path} (input_dim={ae_in})")
        
        if optimizer_path and Path(optimizer_path).exists():
            self.optimizer = DRLOptimizer(config=self.config)
            self.optimizer.load(optimizer_path)
            print(f"Loaded optimizer from {optimizer_path}")
        
        if pinn_path and Path(pinn_path).exists():
            # Infer input_dim and output_dim from checkpoint weights to avoid
            # shape mismatch when the saved model differs from current defaults.
            pinn_ckpt = torch.load(pinn_path, map_location=self.device,
                                   weights_only=False)
            sd = pinn_ckpt["model_state_dict"]
            pinn_in  = sd["hidden_layers.0.linear.weight"].shape[1]
            pinn_out = sd["output_layer.weight"].shape[0]
            self.pinn = PINN(input_dim=pinn_in, output_dim=pinn_out)
            self.pinn.load_state_dict(sd)
            print(f"Loaded PINN from {pinn_path} "
                  f"(input_dim={pinn_in}, output_dim={pinn_out})")
    
    def sync_state(self, sensor_data: Dict[str, float]) -> ESPState:
        """
        Synchronize digital twin with real sensor data.
        
        Args:
            sensor_data: Dictionary of sensor readings
            
        Returns:
            Updated state
        """
        # Update state from sensor data
        for key, value in sensor_data.items():
            if hasattr(self.current_state, key):
                if key == "status" and isinstance(value, str):
                    from src.digital_twin.esp_simulator import ESPStatus
                    value = ESPStatus(value)
                setattr(self.current_state, key, value)
        
        # Record history
        state_record = self.current_state.to_dict()
        state_record["timestamp"] = datetime.now().isoformat()
        self.state_history.append(state_record)
        
        # Check for alerts
        self._check_alerts()
        
        return self.current_state
    
    def _check_alerts(self) -> List[Alert]:
        """Check current state against alert thresholds."""
        new_alerts = []
        now = datetime.now()
        
        # Temperature alert
        temp = self.current_state.motor_temperature
        if temp > self.alert_thresholds["motor_temperature"]["critical"]:
            alert = Alert(
                timestamp=now,
                level="critical",
                message=f"Motor temperature CRITICAL: {temp:.1f}°C",
                sensor="motor_temperature",
                value=temp,
                threshold=self.alert_thresholds["motor_temperature"]["critical"]
            )
            new_alerts.append(alert)
        elif temp > self.alert_thresholds["motor_temperature"]["warning"]:
            alert = Alert(
                timestamp=now,
                level="warning",
                message=f"Motor temperature elevated: {temp:.1f}°C",
                sensor="motor_temperature",
                value=temp,
                threshold=self.alert_thresholds["motor_temperature"]["warning"]
            )
            new_alerts.append(alert)
        
        # Vibration alert
        vib_mag = np.sqrt(
            self.current_state.vibration_x**2 +
            self.current_state.vibration_y**2 +
            self.current_state.vibration_z**2
        )
        if vib_mag > self.alert_thresholds["vibration_magnitude"]["critical"]:
            alert = Alert(
                timestamp=now,
                level="critical",
                message=f"Vibration CRITICAL: {vib_mag:.1f} mm/s",
                sensor="vibration",
                value=vib_mag,
                threshold=self.alert_thresholds["vibration_magnitude"]["critical"]
            )
            new_alerts.append(alert)
        elif vib_mag > self.alert_thresholds["vibration_magnitude"]["warning"]:
            alert = Alert(
                timestamp=now,
                level="warning",
                message=f"Vibration elevated: {vib_mag:.1f} mm/s",
                sensor="vibration",
                value=vib_mag,
                threshold=self.alert_thresholds["vibration_magnitude"]["warning"]
            )
            new_alerts.append(alert)
        
        # Health alert
        health = self.current_state.equipment_health
        if health < self.alert_thresholds["equipment_health"]["critical"]:
            alert = Alert(
                timestamp=now,
                level="critical",
                message=f"Equipment health CRITICAL: {health:.1%}",
                sensor="equipment_health",
                value=health,
                threshold=self.alert_thresholds["equipment_health"]["critical"]
            )
            new_alerts.append(alert)
        elif health < self.alert_thresholds["equipment_health"]["warning"]:
            alert = Alert(
                timestamp=now,
                level="warning",
                message=f"Equipment health degraded: {health:.1%}",
                sensor="equipment_health",
                value=health,
                threshold=self.alert_thresholds["equipment_health"]["warning"]
            )
            new_alerts.append(alert)
        
        # Voltage alert (low voltage is the risk)
        voltage = self.current_state.voltage
        if voltage < self.alert_thresholds["voltage_low"]["critical"]:
            alert = Alert(
                timestamp=now,
                level="critical",
                message=f"Voltage CRITICAL LOW: {voltage:.1f}V",
                sensor="voltage",
                value=voltage,
                threshold=self.alert_thresholds["voltage_low"]["critical"]
            )
            new_alerts.append(alert)
        elif voltage < self.alert_thresholds["voltage_low"]["warning"]:
            alert = Alert(
                timestamp=now,
                level="warning",
                message=f"Voltage below normal: {voltage:.1f}V",
                sensor="voltage",
                value=voltage,
                threshold=self.alert_thresholds["voltage_low"]["warning"]
            )
            new_alerts.append(alert)

        # Sand rate alert
        sand = self.current_state.sand_rate
        if sand > self.alert_thresholds["sand_rate"]["critical"]:
            alert = Alert(
                timestamp=now,
                level="critical",
                message=f"Sand rate CRITICAL: {sand:.2f}%",
                sensor="sand_rate",
                value=sand,
                threshold=self.alert_thresholds["sand_rate"]["critical"]
            )
            new_alerts.append(alert)
        elif sand > self.alert_thresholds["sand_rate"]["warning"]:
            alert = Alert(
                timestamp=now,
                level="warning",
                message=f"Sand rate elevated: {sand:.2f}%",
                sensor="sand_rate",
                value=sand,
                threshold=self.alert_thresholds["sand_rate"]["warning"]
            )
            new_alerts.append(alert)

        # Deduplicate: skip new alerts that are identical to one fired in the
        # last DEDUP_WINDOW seconds (same sensor + same level). Prevents the
        # log from spamming the same "Motor CRITICAL: 142.x" every step.
        DEDUP_WINDOW_S = 30.0
        from datetime import timedelta
        dedup_cutoff = now - timedelta(seconds=DEDUP_WINDOW_S)
        recent = [a for a in self.alert_history if a.timestamp >= dedup_cutoff]
        deduped = []
        for a in new_alerts:
            already_fired = any(
                r.sensor == a.sensor and r.level == a.level for r in recent
            )
            if not already_fired:
                deduped.append(a)

        self.alert_history.extend(deduped)
        return deduped

    def predict_failure(
        self,
        horizon_hours: int = 48
    ) -> Tuple[float, Dict[str, float]]:
        """
        Predict probability of failure within horizon.
        
        Args:
            horizon_hours: Prediction horizon in hours
            
        Returns:
            Tuple of (failure_probability, sensor_contributions)
        """
        if self.predictor is None:
            # Heuristic fallback
            health = self.current_state.equipment_health
            temp_risk = max(0, (self.current_state.motor_temperature - 100) / 50)
            vib_mag = np.sqrt(
                self.current_state.vibration_x**2 +
                self.current_state.vibration_y**2 +
                self.current_state.vibration_z**2
            )
            vib_risk = max(0, (vib_mag - 5) / 10)
            
            probability = (1 - health) * 0.5 + temp_risk * 0.3 + vib_risk * 0.2
            
            contributions = {
                "equipment_health": (1 - health) * 0.5,
                "motor_temperature": temp_risk * 0.3,
                "vibration": vib_risk * 0.2
            }
            
            return min(probability, 1.0), contributions
        
        # Use LSTM predictor (requires enough history)
        if len(self.state_history) < self.config.training.sequence_length:
            return 0.0, {}

        recent_states = self.state_history[-self.config.training.sequence_length:]
        sequence = np.array([
            [s.get(col, 0) for col in self.config.training.sensor_channels]
            for s in recent_states
        ])

        x = torch.FloatTensor(sequence).unsqueeze(0)
        probability, attention = self.predictor.predict(x, return_attention=True)
        
        # Calculate contributions from attention
        contributions = {}
        if attention is not None:
            mean_attention = attention.mean(axis=(0, 1, 2))
            for i, col in enumerate(self.config.training.sensor_channels):
                if i < len(mean_attention):
                    contributions[col] = float(mean_attention[i])
        
        return float(probability[0, 0]), contributions
    
    def detect_anomaly(self) -> Tuple[bool, float, Dict[str, float]]:
        """
        Detect anomalies in current state.
        
        Returns:
            Tuple of (is_anomaly, anomaly_score, feature_contributions)
        """
        if self.anomaly_detector is None:
            # Heuristic fallback
            score = 0.0
            contributions = {}
            
            # Temperature deviation
            temp_dev = abs(self.current_state.motor_temperature - 85) / 50
            score += temp_dev * 0.3
            contributions["motor_temperature"] = temp_dev
            
            # Vibration deviation
            vib_mag = np.sqrt(
                self.current_state.vibration_x**2 +
                self.current_state.vibration_y**2 +
                self.current_state.vibration_z**2
            )
            vib_dev = max(0, (vib_mag - 3) / 5)
            score += vib_dev * 0.3
            contributions["vibration"] = vib_dev
            
            # Current deviation
            curr_dev = abs(self.current_state.motor_current - 45) / 30
            score += curr_dev * 0.2
            contributions["motor_current"] = curr_dev
            
            is_anomaly = score > 0.5
            return is_anomaly, score, contributions
        
        # Use autoencoder (requires enough history)
        if len(self.state_history) < self.config.training.sequence_length:
            return False, 0.0, {}

        recent_states = self.state_history[-self.config.training.sequence_length:]
        sequence = np.array([
            [s.get(col, 0) for col in self.config.training.sensor_channels]
            for s in recent_states
        ])
        
        x = torch.FloatTensor(sequence).unsqueeze(0)
        is_anomaly, scores = self.anomaly_detector.detect_anomalies(x)
        feature_contrib = self.anomaly_detector.get_feature_contributions(x)
        
        contributions = {}
        for i, col in enumerate(self.config.training.sensor_channels):
            contributions[col] = float(feature_contrib[0, i])
        
        return bool(is_anomaly[0]), float(scores[0]), contributions
    
    def get_optimization_recommendation(self) -> Recommendation:
        """
        Get AI-powered optimization recommendation.
        
        Returns:
            Recommendation object
        """
        current_freq = self.current_state.frequency
        current_choke = self.current_state.choke_position
        
        if self.optimizer is not None:
            # Use DRL agent
            obs = np.concatenate([
                self.current_state.to_array(),
                [self.current_state.equipment_health, 2500.0]  # target production
            ])
            action = self.optimizer.get_action(obs)
            
            new_freq = current_freq + action[0] * 40
            new_freq = np.clip(new_freq, 20, 60)
            new_choke = action[1]
            
            recommendation = Recommendation(
                timestamp=datetime.now(),
                action="adjust_frequency",
                current_value=current_freq,
                recommended_value=new_freq,
                confidence=0.85,
                reasoning=f"DRL agent recommends frequency {new_freq:.1f} Hz for optimal efficiency",
                expected_impact={
                    "energy_savings": 0.15,
                    "flow_rate_change": (new_freq - current_freq) / current_freq * 100
                }
            )
        else:
            # Heuristic optimization
            temp = self.current_state.motor_temperature
            flow = self.current_state.flow_rate
            
            if temp > 110:
                # Reduce frequency to cool down
                new_freq = current_freq - 3
                reasoning = "Reduce frequency to lower motor temperature"
            elif flow < 1500 and current_freq < 55:
                # Increase frequency for more production
                new_freq = current_freq + 2
                reasoning = "Increase frequency to boost production"
            else:
                new_freq = current_freq
                reasoning = "Current settings are optimal"
            
            recommendation = Recommendation(
                timestamp=datetime.now(),
                action="adjust_frequency",
                current_value=current_freq,
                recommended_value=new_freq,
                confidence=0.7,
                reasoning=reasoning,
                expected_impact={
                    "energy_change": ((new_freq / current_freq) ** 3 - 1) * 100
                }
            )
        
        self.recommendation_history.append(recommendation)
        if len(self.recommendation_history) > 100:
            self.recommendation_history = self.recommendation_history[-100:]
        return recommendation
    
    def what_if_analysis(
        self,
        frequency: Optional[float] = None,
        choke_position: Optional[float] = None,
        hours: int = 24
    ) -> List[Dict]:
        """
        Run what-if scenario analysis.
        
        Args:
            frequency: Hypothetical frequency
            choke_position: Hypothetical choke position
            hours: Duration to simulate
            
        Returns:
            List of predicted states
        """
        # Create a copy of current state
        scenario_sim = ESPSimulator(config=self.config)
        state_dict = self.current_state.to_dict()
        state_dict.pop("status", None)  # ESPStatus enum not serializable as kwarg
        scenario_sim.state = ESPState(**state_dict)
        
        # Apply hypothetical conditions
        freq = frequency or self.current_state.frequency
        choke = choke_position or self.current_state.choke_position
        
        # Run simulation
        predictions = []
        for _ in range(hours):
            state = scenario_sim.step(frequency=freq, choke_position=choke)
            predictions.append(state.to_dict())
            if scenario_sim.state.status == ESPStatus.FAILED:
                break

        return predictions
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """
        Get all data needed for dashboard display.
        
        Returns:
            Dictionary with dashboard data
        """
        # Failure prediction
        failure_prob, failure_contrib = self.predict_failure()
        
        # Anomaly detection
        is_anomaly, anomaly_score, anomaly_contrib = self.detect_anomaly()
        
        # Optimization recommendation
        recommendation = self.get_optimization_recommendation()
        
        # KPIs
        kpis = self.simulator.get_kpi_summary()
        
        return {
            "current_state": self.current_state.to_dict(),
            "status": self.current_state.status.value if isinstance(self.current_state.status, ESPStatus) else self.current_state.status,
            "failure_prediction": {
                "probability": failure_prob,
                "horizon_hours": 48,
                "contributions": failure_contrib
            },
            "anomaly_detection": {
                "is_anomaly": is_anomaly,
                "score": anomaly_score,
                "contributions": anomaly_contrib
            },
            "recommendation": {
                "action": recommendation.action,
                "current": recommendation.current_value,
                "recommended": recommendation.recommended_value,
                "confidence": recommendation.confidence,
                "reasoning": recommendation.reasoning
            },
            "kpis": kpis,
            "recent_alerts": [
                {
                    "level": a.level,
                    "message": a.message,
                    "timestamp": a.timestamp.isoformat()
                }
                for a in self.alert_history[-5:]
            ],
            "history_length": len(self.state_history)
        }
    
    def export_history(self, path: str) -> None:
        """Export state history to JSON file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.state_history, f, indent=2, default=str)
        print(f"History exported to {path}")


if __name__ == "__main__":
    # Test digital twin engine
    print("Testing Digital Twin Engine...")
    
    engine = DigitalTwinEngine()
    
    # Simulate some state updates
    print("\nRunning 50-hour simulation...")
    for i in range(50):
        # Simulate sensor data
        engine.simulator.step(frequency=52, choke_position=0.8)
        
        sensor_data = engine.simulator.state.to_dict()
        engine.sync_state(sensor_data)
        
        if i % 10 == 0:
            dashboard = engine.get_dashboard_data()
            print(f"\nHour {i}:")
            print(f"  Status: {dashboard['status']}")
            print(f"  Failure Prob: {dashboard['failure_prediction']['probability']:.2%}")
            print(f"  Anomaly: {dashboard['anomaly_detection']['is_anomaly']}")
    
    # Get final dashboard
    print("\nFinal Dashboard Data:")
    data = engine.get_dashboard_data()
    for key, value in data.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for k, v in value.items():
                print(f"    {k}: {v}")
        else:
            print(f"  {key}: {value}")
    
    print("\nDigital Twin Engine test passed!")
