"""
Deep Reinforcement Learning Optimizer for ESP Systems
======================================================
PPO-based agent for dynamic optimization of ESP operating parameters
to achieve 15% energy savings while maintaining production targets.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, Optional, Dict, List, Any
from pathlib import Path
import gymnasium as gym
from gymnasium import spaces

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.config.config import Config, ESPConfig


class ESPEnvironment(gym.Env):
    """
    Gymnasium environment for ESP optimization.
    
    Simulates ESP operation with realistic physics:
    - Pump affinity laws
    - Energy consumption modeling
    - Equipment wear tracking
    - Production targets
    """
    
    metadata = {"render_modes": ["human"]}
    
    def __init__(
        self,
        config: Optional[Config] = None,
        render_mode: Optional[str] = None
    ):
        """
        Initialize environment.
        
        Args:
            config: Configuration object
            render_mode: Rendering mode
        """
        super().__init__()
        
        self.config = config or Config()
        self.esp = self.config.esp
        self.render_mode = render_mode
        
        # State space: sensor readings + equipment health
        # [temp, intake_p, discharge_p, current, vib_x, vib_y, vib_z,
        #  flow, freq, power, fluid_temp, casing_p, voltage, power_factor,
        #  wellhead_p, sand_rate, health_score, production_target]
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(18,),
            dtype=np.float32
        )
        
        # Action space: [frequency_adjustment, choke_position]
        # frequency: relative adjustment (-0.1 to +0.1 of range)
        # choke: absolute position (0 to 1)
        self.action_space = spaces.Box(
            low=np.array([-0.1, 0.0]),
            high=np.array([0.1, 1.0]),
            dtype=np.float32
        )
        
        # Episode parameters
        self.max_steps = 720  # 30 days of hourly steps
        self.current_step = 0
        
        # State variables
        self.state = None
        self.frequency = None
        self.choke_position = None
        self.equipment_health = None
        self.cumulative_production = 0.0
        self.cumulative_energy = 0.0
        
        # Target production (bpd)
        self.target_production = 2500.0
        
        # Baseline energy (for comparison)
        self.baseline_energy_per_barrel = 15.0  # kWh per barrel
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation vector."""
        freq_ratio = self.frequency / self.esp.nominal_frequency_hz
        
        # Simulate sensor readings based on current state
        motor_temp = 80 + 20 * freq_ratio + np.random.normal(0, 1)
        intake_pressure = 800 + np.random.normal(0, 10)
        discharge_pressure = 2000 * (freq_ratio ** 2) + np.random.normal(0, 20)
        motor_current = 40 * freq_ratio + np.random.normal(0, 1)
        
        vibration_base = 2.0 * (1 + (1 - self.equipment_health) * 3)
        vibration_x = vibration_base + np.abs(np.random.normal(0, 0.3))
        vibration_y = vibration_base + np.abs(np.random.normal(0, 0.3))
        vibration_z = vibration_base + np.abs(np.random.normal(0, 0.2))
        
        # Flow rate based on affinity law and choke
        base_flow = 2000 * freq_ratio * self.choke_position
        flow_rate = base_flow + np.random.normal(0, 30)
        
        power = 50 * (freq_ratio ** 3) + np.random.normal(0, 1)
        fluid_temp = 70 + np.random.normal(0, 1)
        casing_pressure = 150 + np.random.normal(0, 5)

        # New sensors
        voltage = self.esp.nominal_voltage - 2 * freq_ratio + np.random.normal(0, 2)
        power_factor = np.clip(
            self.esp.nominal_power_factor - 0.05 * (1 - freq_ratio) + np.random.normal(0, 0.01),
            self.esp.min_power_factor, self.esp.max_power_factor
        )
        wellhead_pressure = np.clip(
            250 - 30 * freq_ratio + np.random.normal(0, 5),
            self.esp.min_wellhead_pressure, self.esp.max_wellhead_pressure
        )
        sand_rate = max(0, self.esp.normal_sand_rate + abs(np.random.normal(0, 0.01)))

        obs = np.array([
            motor_temp,
            intake_pressure,
            discharge_pressure,
            motor_current,
            vibration_x,
            vibration_y,
            vibration_z,
            flow_rate,
            self.frequency,
            power,
            fluid_temp,
            casing_pressure,
            voltage,
            power_factor,
            wellhead_pressure,
            sand_rate,
            self.equipment_health,
            self.target_production
        ], dtype=np.float32)
        
        return obs
    
    def _calculate_reward(
        self,
        obs: np.ndarray,
        action: np.ndarray
    ) -> Tuple[float, Dict[str, float]]:
        """
        Calculate reward based on multiple objectives.
        
        Objectives:
        - Maximize production (towards target)
        - Minimize energy consumption
        - Preserve equipment health
        - Avoid constraint violations
        """
        flow_rate = obs[7]
        power = obs[9]
        motor_temp = obs[0]
        vibration_mag = np.sqrt(obs[4]**2 + obs[5]**2 + obs[6]**2)
        sand_rate = obs[15]
        
        # Production reward (meeting target)
        production_fraction = min(flow_rate / self.target_production, 1.5)
        if production_fraction < 0.8:
            production_reward = -2.0 * (0.8 - production_fraction)
        elif production_fraction < 1.0:
            production_reward = 0.5 * production_fraction
        else:
            production_reward = 1.0 * min(production_fraction, 1.2)
        
        # Energy efficiency reward
        if flow_rate > 0:
            energy_per_barrel = (power * 24) / flow_rate  # kWh per barrel
            efficiency_ratio = self.baseline_energy_per_barrel / max(energy_per_barrel, 0.1)
            energy_reward = 0.5 * (efficiency_ratio - 1.0)  # Reward for being better than baseline
        else:
            energy_reward = -1.0
        
        # Equipment health reward
        health_reward = 0.3 * self.equipment_health
        
        # Constraint violation penalties
        constraint_penalty = 0.0
        
        # Temperature constraint
        if motor_temp > self.esp.critical_temp:
            constraint_penalty -= 5.0 * (motor_temp - self.esp.critical_temp) / 10
        
        # Vibration constraint
        if vibration_mag > self.esp.critical_vibration:
            constraint_penalty -= 3.0 * (vibration_mag - self.esp.critical_vibration) / 5
        
        # Frequency bounds
        if self.frequency < self.esp.min_frequency_hz or self.frequency > self.esp.max_frequency_hz:
            constraint_penalty -= 10.0

        # Sand rate constraint
        if sand_rate > self.esp.critical_sand_rate:
            constraint_penalty -= 2.0 * (sand_rate - self.esp.critical_sand_rate)
        
        # Total reward
        total_reward = (
            production_reward +
            energy_reward +
            health_reward +
            constraint_penalty
        )
        
        info = {
            "production_reward": production_reward,
            "energy_reward": energy_reward,
            "health_reward": health_reward,
            "constraint_penalty": constraint_penalty,
            "flow_rate": flow_rate,
            "power": power,
            "efficiency": energy_per_barrel if flow_rate > 0 else float('inf')
        }
        
        return total_reward, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step in the environment.
        
        Args:
            action: [frequency_adjustment, choke_position]
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        # Apply actions
        freq_adjustment = action[0] * (self.esp.max_frequency_hz - self.esp.min_frequency_hz)
        self.frequency = np.clip(
            self.frequency + freq_adjustment,
            self.esp.min_frequency_hz,
            self.esp.max_frequency_hz
        )
        self.choke_position = np.clip(action[1], 0.0, 1.0)
        
        # Update equipment health (gradual degradation)
        degradation_rate = 0.0001 * (1 + (self.frequency / self.esp.nominal_frequency_hz - 1) ** 2)
        self.equipment_health = max(0.0, self.equipment_health - degradation_rate)
        
        # Get observation
        obs = self._get_observation()
        
        # Calculate reward
        reward, info = self._calculate_reward(obs, action)
        
        # Track cumulative metrics
        flow_rate = obs[7]
        power = obs[9]
        self.cumulative_production += flow_rate / 24  # hourly contribution
        self.cumulative_energy += power
        
        self.current_step += 1
        
        # Episode termination
        terminated = self.equipment_health <= 0.0  # Equipment failure
        truncated = self.current_step >= self.max_steps
        
        info.update({
            "step": self.current_step,
            "frequency": self.frequency,
            "choke_position": self.choke_position,
            "equipment_health": self.equipment_health,
            "cumulative_production": self.cumulative_production,
            "cumulative_energy": self.cumulative_energy
        })
        
        return obs, reward, terminated, truncated, info
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """Reset environment to initial state."""
        super().reset(seed=seed)
        
        # Initialize state
        self.frequency = self.esp.nominal_frequency_hz
        self.choke_position = 0.8
        self.equipment_health = 1.0
        self.current_step = 0
        self.cumulative_production = 0.0
        self.cumulative_energy = 0.0
        
        # Random variation in target
        self.target_production = 2500 + np.random.uniform(-500, 500)
        
        obs = self._get_observation()
        info = {"target_production": self.target_production}
        
        return obs, info
    
    def render(self):
        """Render the environment."""
        if self.render_mode == "human":
            print(f"Step: {self.current_step}, "
                  f"Freq: {self.frequency:.1f} Hz, "
                  f"Choke: {self.choke_position:.2f}, "
                  f"Health: {self.equipment_health:.3f}")


class DRLOptimizer:
    """
    Deep Reinforcement Learning optimizer for ESP systems.
    
    Uses PPO algorithm from Stable Baselines3 for training.
    """
    
    def __init__(
        self,
        config: Optional[Config] = None,
        device: Optional[str] = None
    ):
        """
        Initialize optimizer.
        
        Args:
            config: Configuration object
            device: Device to use
        """
        self.config = config or Config()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        self.env = None
        self.model = None
        self.training_stats: List[Dict] = []
    
    def create_environment(self) -> ESPEnvironment:
        """Create and return the ESP environment."""
        self.env = ESPEnvironment(config=self.config)
        return self.env
    
    def train(
        self,
        total_timesteps: int = 100000,
        save_path: Optional[str] = None
    ) -> None:
        """
        Train the DRL agent.
        
        Args:
            total_timesteps: Total training timesteps
            save_path: Path to save trained model
        """
        try:
            from stable_baselines3 import PPO
            from stable_baselines3.common.callbacks import EvalCallback
            from stable_baselines3.common.env_util import make_vec_env
        except ImportError:
            print("stable-baselines3 not installed. Please run: pip install stable-baselines3")
            return
        
        if self.env is None:
            self.create_environment()
        
        print(f"Training DRL Optimizer for {total_timesteps} timesteps")
        
        # Create model
        self.model = PPO(
            policy=self.config.model.drl_policy,
            env=self.env,
            learning_rate=self.config.model.drl_learning_rate,
            n_steps=self.config.model.drl_n_steps,
            batch_size=self.config.model.drl_batch_size,
            n_epochs=self.config.model.drl_n_epochs,
            gamma=self.config.model.drl_gamma,
            gae_lambda=self.config.model.drl_gae_lambda,
            clip_range=self.config.model.drl_clip_range,
            verbose=1,
            device=self.device
        )
        
        # Train
        self.model.learn(total_timesteps=total_timesteps)
        
        # Save model
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            self.model.save(save_path)
            print(f"Model saved to {save_path}")
    
    def evaluate(
        self,
        n_episodes: int = 10
    ) -> Dict[str, float]:
        """
        Evaluate trained agent.
        
        Args:
            n_episodes: Number of evaluation episodes
            
        Returns:
            Evaluation metrics
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")
        
        if self.env is None:
            self.create_environment()
        
        total_rewards = []
        total_production = []
        total_energy = []
        
        for _ in range(n_episodes):
            obs, _ = self.env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.env.step(action)
                episode_reward += reward
                done = terminated or truncated
            
            total_rewards.append(episode_reward)
            total_production.append(info["cumulative_production"])
            total_energy.append(info["cumulative_energy"])
        
        metrics = {
            "mean_reward": np.mean(total_rewards),
            "std_reward": np.std(total_rewards),
            "mean_production": np.mean(total_production),
            "mean_energy": np.mean(total_energy),
            "energy_per_barrel": np.mean(total_energy) / (np.mean(total_production) + 1e-6)
        }
        
        baseline_energy_per_barrel = 15.0
        metrics["energy_savings_pct"] = (
            (baseline_energy_per_barrel - metrics["energy_per_barrel"]) / 
            baseline_energy_per_barrel * 100
        )
        
        print("\nEvaluation Results:")
        print(f"  Mean Reward: {metrics['mean_reward']:.2f} ± {metrics['std_reward']:.2f}")
        print(f"  Mean Production: {metrics['mean_production']:.0f} barrels")
        print(f"  Energy per Barrel: {metrics['energy_per_barrel']:.2f} kWh")
        print(f"  Energy Savings: {metrics['energy_savings_pct']:.1f}%")
        
        return metrics
    
    def get_action(
        self,
        observation: np.ndarray,
        deterministic: bool = True
    ) -> np.ndarray:
        """
        Get optimal action for given observation.
        
        Args:
            observation: Current state observation
            deterministic: Whether to use deterministic policy
            
        Returns:
            Optimal action
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")
        
        action, _ = self.model.predict(observation, deterministic=deterministic)
        return action
    
    def load(self, path: str) -> "DRLOptimizer":
        """Load trained model."""
        try:
            from stable_baselines3 import PPO
        except ImportError:
            print("stable-baselines3 not installed.")
            return self
        
        if self.env is None:
            self.create_environment()
        
        self.model = PPO.load(path, env=self.env, device=self.device)
        print(f"Model loaded from {path}")
        return self


if __name__ == "__main__":
    # Test environment
    print("Testing ESP Environment...")
    
    env = ESPEnvironment()
    obs, info = env.reset()
    print(f"Initial observation shape: {obs.shape}")
    print(f"Initial info: {info}")
    
    # Test random actions
    total_reward = 0
    for i in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if terminated or truncated:
            break
    
    print(f"Total reward after 100 steps: {total_reward:.2f}")
    print(f"Final frequency: {info['frequency']:.1f} Hz")
    print(f"Final health: {info['equipment_health']:.3f}")
    
    print("\nESP Environment test passed!")
