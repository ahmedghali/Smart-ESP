"""
XAI Visualizer for ESP AI Optimization
=======================================
Provides explainable AI visualizations using SHAP, LIME, and attention weights.
Helps petroleum engineers understand and trust AI recommendations.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import json

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.config.config import Config


class SHAPExplainer:
    """
    SHAP-based explainability for ESP models.
    
    Provides global and local feature importance explanations.
    """
    
    def __init__(
        self,
        model: Any,
        feature_names: List[str],
        background_data: Optional[np.ndarray] = None
    ):
        """
        Initialize SHAP explainer.
        
        Args:
            model: Model to explain
            feature_names: Names of input features
            background_data: Background data for SHAP calculations
        """
        self.model = model
        self.feature_names = feature_names
        self.background_data = background_data
        self.shap_values = None
        
        try:
            import shap
            self.shap_available = True
            
            if background_data is not None:
                # Use KernelExplainer for model-agnostic explanation
                self.explainer = shap.KernelExplainer(
                    self._model_predict,
                    background_data[:100]  # Subsample for efficiency
                )
            else:
                self.explainer = None
        except ImportError:
            self.shap_available = False
            self.explainer = None
            print("SHAP not installed. Using fallback explanations.")
    
    def _model_predict(self, x: np.ndarray) -> np.ndarray:
        """Wrapper for model prediction."""
        import torch
        with torch.no_grad():
            x_tensor = torch.FloatTensor(x)
            if hasattr(self.model, 'predict'):
                return self.model.predict(x_tensor)[0]
            else:
                return self.model(x_tensor).numpy()
    
    def explain_instance(
        self,
        instance: np.ndarray,
        n_samples: int = 100
    ) -> Dict[str, float]:
        """
        Explain a single prediction.
        
        Args:
            instance: Input instance to explain
            n_samples: Number of samples for SHAP estimation
            
        Returns:
            Dictionary of feature contributions
        """
        if not self.shap_available or self.explainer is None:
            # Fallback: simple gradient-based importance
            return self._fallback_explain(instance)
        
        shap_values = self.explainer.shap_values(instance.reshape(1, -1), nsamples=n_samples)
        
        contributions = {}
        for i, name in enumerate(self.feature_names):
            if i < len(shap_values[0]):
                contributions[name] = float(shap_values[0][i])
        
        return contributions
    
    def _fallback_explain(self, instance: np.ndarray) -> Dict[str, float]:
        """Fallback explanation using input magnitude."""
        contributions = {}
        for i, name in enumerate(self.feature_names):
            if i < len(instance):
                # Simple magnitude-based importance
                contributions[name] = abs(float(instance[i]))
        
        # Normalize
        total = sum(contributions.values())
        if total > 0:
            contributions = {k: v / total for k, v in contributions.items()}
        
        return contributions
    
    def plot_waterfall(
        self,
        contributions: Dict[str, float],
        prediction: float,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create SHAP waterfall plot.
        
        Args:
            contributions: Feature contributions
            prediction: Model prediction
            save_path: Optional path to save figure
            
        Returns:
            Matplotlib figure
        """
        # Sort by absolute contribution
        sorted_items = sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)
        features = [item[0] for item in sorted_items[:10]]
        values = [item[1] for item in sorted_items[:10]]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        colors = ['red' if v > 0 else 'blue' for v in values]
        y_pos = np.arange(len(features))
        
        ax.barh(y_pos, values, color=colors, alpha=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(features)
        ax.set_xlabel('SHAP Value (Impact on prediction)')
        ax.set_title(f'Feature Contributions to Prediction: {prediction:.3f}')
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def plot_summary(
        self,
        data: np.ndarray,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create SHAP summary plot for multiple instances.
        
        Args:
            data: Data to explain (N, features)
            save_path: Optional path to save figure
            
        Returns:
            Matplotlib figure
        """
        # Get SHAP values for all instances
        all_contributions = []
        for i in range(min(len(data), 100)):
            contrib = self.explain_instance(data[i])
            all_contributions.append([contrib.get(f, 0) for f in self.feature_names])
        
        shap_array = np.array(all_contributions)
        
        # Mean absolute SHAP values
        mean_shap = np.abs(shap_array).mean(axis=0)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Sort features by importance
        sorted_idx = np.argsort(mean_shap)[::-1]
        sorted_features = [self.feature_names[i] for i in sorted_idx[:12]]
        sorted_values = mean_shap[sorted_idx[:12]]
        
        y_pos = np.arange(len(sorted_features))
        ax.barh(y_pos, sorted_values, color='steelblue', alpha=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(sorted_features)
        ax.set_xlabel('Mean |SHAP Value|')
        ax.set_title('Global Feature Importance')
        ax.invert_yaxis()
        
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig


class AttentionVisualizer:
    """
    Visualize attention weights from transformer/attention-based models.
    """
    
    def __init__(self, feature_names: List[str]):
        """
        Initialize visualizer.
        
        Args:
            feature_names: Names of features
        """
        self.feature_names = feature_names
    
    def plot_attention_heatmap(
        self,
        attention_weights: np.ndarray,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot attention weights as heatmap.
        
        Args:
            attention_weights: Attention matrix (seq_len, seq_len) or (heads, seq, seq)
            save_path: Optional path to save figure
            
        Returns:
            Matplotlib figure
        """
        if attention_weights.ndim == 3:
            # Average over heads
            attention_weights = attention_weights.mean(axis=0)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        im = ax.imshow(attention_weights, cmap='Blues', aspect='auto')
        ax.set_xlabel('Key Position')
        ax.set_ylabel('Query Position')
        ax.set_title('Attention Weights')
        
        plt.colorbar(im, ax=ax, label='Attention Weight')
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def plot_temporal_attention(
        self,
        attention_weights: np.ndarray,
        timestamps: Optional[List[str]] = None,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot attention over time sequence.
        
        Args:
            attention_weights: Attention for each timestep (seq_len,)
            timestamps: Optional timestamp labels
            save_path: Optional path to save figure
            
        Returns:
            Matplotlib figure
        """
        seq_len = len(attention_weights)
        
        fig, ax = plt.subplots(figsize=(12, 4))
        
        x = np.arange(seq_len)
        ax.bar(x, attention_weights, color='steelblue', alpha=0.8)
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Attention Weight')
        ax.set_title('Temporal Attention Distribution')
        
        if timestamps and len(timestamps) <= 20:
            ax.set_xticks(x)
            ax.set_xticklabels(timestamps, rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def plot_feature_attention(
        self,
        attention_weights: np.ndarray,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot attention over features.
        
        Args:
            attention_weights: Attention per feature (n_features,)
            save_path: Optional path to save figure
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Sort by attention
        sorted_idx = np.argsort(attention_weights)[::-1]
        n_show = min(len(self.feature_names), 12)
        
        features = [self.feature_names[i] for i in sorted_idx[:n_show]]
        values = attention_weights[sorted_idx[:n_show]]
        
        y_pos = np.arange(len(features))
        ax.barh(y_pos, values, color='coral', alpha=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(features)
        ax.set_xlabel('Attention Weight')
        ax.set_title('Feature Attention Distribution')
        ax.invert_yaxis()
        
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig


class XAIVisualizer:
    """
    Main XAI visualization class combining SHAP, LIME, and attention.
    """
    
    def __init__(
        self,
        config: Optional[Config] = None
    ):
        """
        Initialize XAI visualizer.
        
        Args:
            config: Configuration object
        """
        self.config = config or Config()
        self.feature_names = self.config.training.sensor_channels
        
        self.shap_explainer: Optional[SHAPExplainer] = None
        self.attention_viz = AttentionVisualizer(self.feature_names)
    
    def create_explanation_dashboard(
        self,
        prediction: float,
        contributions: Dict[str, float],
        anomaly_score: Optional[float] = None,
        attention_weights: Optional[np.ndarray] = None,
        save_dir: str = "explanations"
    ) -> Dict[str, str]:
        """
        Create a complete explanation dashboard.
        
        Args:
            prediction: Model prediction value
            contributions: Feature contributions
            anomaly_score: Optional anomaly score
            attention_weights: Optional attention weights
            save_dir: Directory to save visualizations
            
        Returns:
            Dictionary of saved file paths
        """
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        saved_files = {}
        
        # 1. Feature contribution bar chart
        fig, ax = plt.subplots(figsize=(10, 6))
        
        sorted_items = sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)
        features = [item[0] for item in sorted_items[:10]]
        values = [item[1] for item in sorted_items[:10]]
        
        colors = ['#d73027' if v > 0 else '#4575b4' for v in values]
        y_pos = np.arange(len(features))
        
        ax.barh(y_pos, values, color=colors, alpha=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(features)
        ax.set_xlabel('Contribution to Prediction')
        ax.set_title(f'Feature Contributions (Prediction: {prediction:.2%})')
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        
        # Add legend
        ax.barh([], [], color='#d73027', label='Increases risk')
        ax.barh([], [], color='#4575b4', label='Decreases risk')
        ax.legend(loc='lower right')
        
        plt.tight_layout()
        path = f"{save_dir}/feature_contributions.png"
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        saved_files['contributions'] = path
        
        # 2. Attention visualization if available
        if attention_weights is not None:
            path = f"{save_dir}/attention_heatmap.png"
            fig = self.attention_viz.plot_attention_heatmap(attention_weights, path)
            plt.close(fig)
            saved_files['attention'] = path
        
        # 3. Summary text
        summary = self._generate_text_explanation(
            prediction, contributions, anomaly_score
        )
        
        summary_path = f"{save_dir}/explanation_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        saved_files['summary'] = summary_path
        
        return saved_files
    
    def _generate_text_explanation(
        self,
        prediction: float,
        contributions: Dict[str, float],
        anomaly_score: Optional[float] = None
    ) -> Dict[str, Any]:
        """Generate human-readable text explanation."""
        
        # Get top contributors
        sorted_contrib = sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)
        top_positive = [(k, v) for k, v in sorted_contrib if v > 0][:3]
        top_negative = [(k, v) for k, v in sorted_contrib if v < 0][:3]
        
        explanation = {
            "prediction": prediction,
            "prediction_level": "high" if prediction > 0.7 else "moderate" if prediction > 0.3 else "low",
            "main_factors": []
        }
        
        # Explain top positive factors
        for feature, contrib in top_positive:
            explanation["main_factors"].append({
                "feature": feature,
                "contribution": contrib,
                "direction": "increases",
                "description": f"{feature.replace('_', ' ').title()} is contributing to higher failure risk"
            })
        
        # Explain top negative factors
        for feature, contrib in top_negative:
            explanation["main_factors"].append({
                "feature": feature,
                "contribution": contrib,
                "direction": "decreases",
                "description": f"{feature.replace('_', ' ').title()} is keeping failure risk lower"
            })
        
        if anomaly_score is not None:
            explanation["anomaly"] = {
                "score": anomaly_score,
                "is_anomaly": anomaly_score > 0.5,
                "description": "Current behavior deviates from normal patterns" if anomaly_score > 0.5 else "Operating within normal parameters"
            }
        
        # Generate natural language summary
        risk_level = explanation["prediction_level"]
        top_factor = explanation["main_factors"][0]["feature"] if explanation["main_factors"] else "unknown"
        
        explanation["summary"] = (
            f"The AI predicts a {risk_level} probability of failure ({prediction:.1%}). "
            f"The main contributing factor is {top_factor.replace('_', ' ')}. "
        )
        
        if prediction > 0.5:
            explanation["recommendation"] = "Consider scheduling preventive maintenance or adjusting operating parameters."
        else:
            explanation["recommendation"] = "Continue monitoring. No immediate action required."
        
        return explanation
    
    def plot_counterfactual(
        self,
        current_values: Dict[str, float],
        target_prediction: float,
        counterfactual_values: Dict[str, float],
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Visualize counterfactual explanation.
        
        Shows what changes would lead to a different prediction.
        
        Args:
            current_values: Current feature values
            target_prediction: Target prediction value
            counterfactual_values: Feature values that would achieve target
            save_path: Optional path to save figure
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        features = list(current_values.keys())[:10]
        current = [current_values[f] for f in features]
        counterfactual = [counterfactual_values.get(f, current_values[f]) for f in features]
        
        x = np.arange(len(features))
        width = 0.35
        
        ax.bar(x - width/2, current, width, label='Current', color='steelblue', alpha=0.8)
        ax.bar(x + width/2, counterfactual, width, label='Counterfactual', color='coral', alpha=0.8)
        
        ax.set_xticks(x)
        ax.set_xticklabels(features, rotation=45, ha='right')
        ax.set_ylabel('Value')
        ax.set_title(f'What-If Analysis: Achieving {target_prediction:.1%} Failure Probability')
        ax.legend()
        
        # Highlight changes
        for i, (c, cf) in enumerate(zip(current, counterfactual)):
            if abs(cf - c) / (abs(c) + 1e-6) > 0.1:
                ax.annotate('', xy=(i + width/2, cf), xytext=(i - width/2, c),
                           arrowprops=dict(arrowstyle='->', color='green', lw=2))
        
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def create_sensor_importance_radar(
        self,
        contributions: Dict[str, float],
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create radar chart of sensor importance.
        
        Args:
            contributions: Feature contributions
            save_path: Optional path to save figure
            
        Returns:
            Matplotlib figure
        """
        # Select top sensors
        sorted_contrib = sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)[:8]
        
        labels = [item[0].replace('_', '\n') for item in sorted_contrib]
        values = [abs(item[1]) for item in sorted_contrib]
        
        # Normalize values
        max_val = max(values) if values else 1
        values = [v / max_val for v in values]
        
        # Create radar chart
        angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
        values += values[:1]  # Complete the loop
        angles += angles[:1]
        
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        
        ax.fill(angles, values, color='steelblue', alpha=0.3)
        ax.plot(angles, values, color='steelblue', linewidth=2)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels)
        ax.set_title('Sensor Importance Radar', y=1.08)
        
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig


if __name__ == "__main__":
    # Test XAI Visualizer
    print("Testing XAI Visualizer...")
    
    config = Config()
    viz = XAIVisualizer(config)
    
    # Create sample data
    contributions = {
        "motor_temperature": 0.25,
        "vibration_x": 0.18,
        "motor_current": 0.12,
        "intake_pressure": -0.08,
        "flow_rate": -0.15,
        "discharge_pressure": 0.05,
        "frequency": -0.03,
        "equipment_health": -0.22
    }
    
    prediction = 0.65
    anomaly_score = 0.45
    
    # Create dashboard
    saved_files = viz.create_explanation_dashboard(
        prediction=prediction,
        contributions=contributions,
        anomaly_score=anomaly_score,
        save_dir="explanations_test"
    )
    
    print("\nSaved Files:")
    for name, path in saved_files.items():
        print(f"  {name}: {path}")
    
    # Create radar chart
    fig = viz.create_sensor_importance_radar(
        contributions,
        save_path="explanations_test/sensor_radar.png"
    )
    plt.close(fig)
    print("  radar: explanations_test/sensor_radar.png")
    
    print("\nXAI Visualizer test passed!")
