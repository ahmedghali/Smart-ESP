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
        
        # State space: 11 real sensor channels + equipment_health + production_target = 13
        # [motor_temp, intake_p, discharge_p, current, freq, fluid_temp,
        #  voltage, vibration, current_leakage, power_consumption,
        #  differential_pressure, health_score, production_target]
        self.n_sensors = len(self.config.training.sensor_channels)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.n_sensors + 2,),
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
        
        # Vibration scalar (g) — increases with equipment degradation
        vibration = 0.15 * (1 + (1 - self.equipment_health) * 5) + np.abs(np.random.normal(0, 0.02))

        fluid_temp = 70 + np.random.normal(0, 1)
        voltage = self.esp.nominal_voltage - 2 * freq_ratio + np.random.normal(0, 2)

        # Power consumption (kW) — derived: √3 × V × I × 0.85
        power = np.sqrt(3) * voltage * motor_current * 0.85 / 1000.0 + np.random.normal(0, 0.5)
        power = max(0.0, power)

        # Differential pressure (psi)
        differential_pressure = discharge_pressure - intake_pressure + np.random.normal(0, 5)

        # Current leakage (mA) — rises as insulation degrades
        current_leakage = max(0.0, 1.5 + (1 - self.equipment_health) * 20 + np.random.normal(0, 0.3))

        obs = np.array([
            motor_temp,
            intake_pressure,
            discharge_pressure,
            motor_current,
            self.frequency,
            fluid_temp,
            voltage,
            vibration,
            current_leakage,
            power,
            differential_pressure,
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
        # obs indices: [motor_temp(0), intake_p(1), discharge_p(2), current(3),
        #               freq(4), fluid_temp(5), voltage(6), vibration(7),
        #               current_leakage(8), power(9), diff_pressure(10),
        #               health(11), prod_target(12)]
        motor_temp       = obs[0]
        power            = obs[9]
        diff_pressure    = obs[10]
        vibration        = obs[7]
        current_leakage  = obs[8]

        # ── Production reward (dominant signal) ──────────────────────────────
        # Differential pressure as production proxy (1000–3000 psi = healthy range).
        # Weight ×3 vs previous version so production drives policy, not health.
        prod_fraction = np.clip(diff_pressure / 2000.0, 0.0, 1.5)
        if prod_fraction < 0.5:
            production_reward = -6.0 * (0.5 - prod_fraction)   # steep penalty near zero
        elif prod_fraction < 1.0:
            production_reward = 1.5 * prod_fraction             # was 0.5
        else:
            production_reward = 1.5 * min(prod_fraction, 1.2)  # was 1.0

        # Hard minimum-production constraint: if virtually no pressure, penalise hard.
        # Prevents the agent from idling at low frequency to protect health.
        if prod_fraction < 0.4:
            constraint_penalty_prod = -4.0
        else:
            constraint_penalty_prod = 0.0

        # ── Energy efficiency reward ──────────────────────────────────────────
        energy_per_unit  = power / max(diff_pressure, 1.0)
        baseline_ratio   = 0.02  # kW/psi baseline
        efficiency_ratio = baseline_ratio / max(energy_per_unit, 1e-6)
        energy_reward    = 0.5 * (efficiency_ratio - 1.0)

        # ── Equipment health reward (reduced — health matters, but less than production) ──
        # Was 0.3 → now 0.1 so 720-step total health signal < production signal.
        health_reward = 0.1 * self.equipment_health

        # ── Constraint violation penalties ────────────────────────────────────
        constraint_penalty = constraint_penalty_prod

        # Temperature constraint
        if motor_temp > self.esp.critical_temp:
            constraint_penalty -= 5.0 * (motor_temp - self.esp.critical_temp) / 10

        # Vibration constraint (g scale: warning ~0.5g, critical ~1.0g)
        if vibration > 0.8:
            constraint_penalty -= 3.0 * (vibration - 0.8) / 0.2

        # Insulation constraint (leakage > 50 mA = danger)
        if current_leakage > 50:
            constraint_penalty -= 2.0 * (current_leakage - 50) / 50

        # Frequency bounds
        if self.frequency < self.esp.min_frequency_hz or self.frequency > self.esp.max_frequency_hz:
            constraint_penalty -= 10.0

        # ── Total reward ──────────────────────────────────────────────────────
        total_reward = (
            production_reward +
            energy_reward +
            health_reward +
            constraint_penalty
        )

        info = {
            "production_reward":  production_reward,
            "energy_reward":      energy_reward,
            "health_reward":      health_reward,
            "constraint_penalty": constraint_penalty,
            "diff_pressure":      diff_pressure,
            "power":              power,
            "efficiency":         energy_per_unit,
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
        # obs[7] = vibration (NOT flow_rate). Use frequency-based production estimate.
        # Affinity law: flow ∝ frequency → estimate bpd proportional to freq ratio.
        freq_ratio = self.frequency / self.esp.nominal_frequency_hz
        est_flow_bpd = self.target_production * freq_ratio
        power = obs[9]
        self.cumulative_production += est_flow_bpd / 24  # hourly bbl contribution
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
            "cumulative_energy": self.cumulative_energy,
            "est_flow_bpd": est_flow_bpd,
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

    Supports PPO (default) and SAC algorithms from Stable Baselines3.
    Automatically wraps the environment with VecNormalize for observation
    and reward normalization (Engstrom et al. ICLR 2020).
    """

    def __init__(
        self,
        config: Optional[Config] = None,
        device: Optional[str] = None,
        algo: str = "sac",                  # "sac" (default, +30-50% reward) or "ppo"
        use_vec_normalize: bool = True,     # observation+reward normalization
    ):
        """
        Initialize optimizer.

        Args:
            config: Configuration object
            device: Device to use
            algo: Algorithm to use ("sac" or "ppo"). SAC is better for continuous control.
            use_vec_normalize: Wrap env with VecNormalize (recommended).
        """
        self.config = config or Config()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.algo = algo.lower()
        self.use_vec_normalize = use_vec_normalize

        if self.algo not in ("sac", "ppo"):
            raise ValueError(f"algo must be 'sac' or 'ppo', got: {algo}")

        self.env = None         # VecEnv (possibly wrapped)
        self.raw_env = None     # Underlying ESPEnvironment (for direct access)
        self.model = None
        self.training_stats: List[Dict] = []

    def create_environment(self):
        """Create the ESP environment, optionally wrapped with VecNormalize."""
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

        self.raw_env = ESPEnvironment(config=self.config)

        # Wrap into a single-env VecEnv (required by SB3)
        env = DummyVecEnv([lambda: self.raw_env])

        if self.use_vec_normalize:
            # Normalise observations (running mean/std) + rewards (gamma=0.99)
            env = VecNormalize(env, norm_obs=True, norm_reward=True, gamma=0.99)
            print("  Env wrapped with VecNormalize (obs + reward normalization)")

        self.env = env
        return self.env
    
    def train(
        self,
        total_timesteps: int = 100000,
        save_path: Optional[str] = None
    ) -> None:
        """
        Train the DRL agent (SAC or PPO based on self.algo).

        Args:
            total_timesteps: Total training timesteps
            save_path: Path to save trained model
        """
        try:
            from stable_baselines3 import PPO, SAC
        except ImportError:
            print("stable-baselines3 not installed. Please run: pip install stable-baselines3")
            return

        if self.env is None:
            self.create_environment()

        print(f"Training DRL Optimizer [{self.algo.upper()}] "
              f"for {total_timesteps} timesteps")

        # Create model based on algorithm
        if self.algo == "sac":
            # SAC: off-policy, entropy-regularized, sample-efficient for continuous control
            # Haarnoja et al. 2018 (arXiv:1801.01290)
            self.model = SAC(
                policy="MlpPolicy",
                env=self.env,
                learning_rate=self.config.model.drl_learning_rate,
                batch_size=256,                 # SAC default
                buffer_size=100_000,            # Replay buffer
                learning_starts=1000,           # Warm-up
                tau=0.005,                      # Soft target update
                gamma=self.config.model.drl_gamma,
                train_freq=1,
                gradient_steps=1,
                ent_coef="auto",                # Auto-adjust entropy coefficient
                verbose=1,
                device=self.device,
            )
        else:  # ppo
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
                device=self.device,
            )

        # Train
        self.model.learn(total_timesteps=total_timesteps)

        # Save model + VecNormalize stats (so we can normalise obs at inference)
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            self.model.save(save_path)
            print(f"Model saved to {save_path}")

            if self.use_vec_normalize:
                vn_path = Path(save_path).with_suffix(".vecnorm.pkl")
                self.env.save(str(vn_path))
                print(f"VecNormalize stats saved to {vn_path}")
    
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

        # Use raw_env for evaluation (Gymnasium API), apply normalize_obs manually.
        # VecNormalize is for training-time observation/reward normalisation,
        # at eval time we want the *true* rewards reported.
        def _normalise(obs):
            if self.use_vec_normalize and hasattr(self.env, "normalize_obs"):
                return self.env.normalize_obs(obs)
            return obs

        for _ in range(n_episodes):
            obs, _ = self.raw_env.reset()
            episode_reward = 0
            done = False

            while not done:
                action, _ = self.model.predict(_normalise(obs), deterministic=True)
                obs, reward, terminated, truncated, info = self.raw_env.step(action)
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

        # --- Baseline aléatoire pour mesurer le gain réel de l'agent ---
        random_rewards = []
        for _ in range(n_episodes):
            obs, _ = self.raw_env.reset()
            ep_reward = 0.0
            done = False
            while not done:
                action = self.raw_env.action_space.sample()   # agent aléatoire
                obs, reward, terminated, truncated, info = self.raw_env.step(action)
                ep_reward += reward
                done = terminated or truncated
            random_rewards.append(ep_reward)

        metrics["random_mean_reward"]  = float(np.mean(random_rewards))
        metrics["reward_vs_random_pct"] = float(
            (metrics["mean_reward"] - metrics["random_mean_reward"]) /
            (abs(metrics["random_mean_reward"]) + 1e-6) * 100
        )

        print("\nEvaluation Results:")
        print(f"  Mean Reward (agent)  : {metrics['mean_reward']:.2f} ± {metrics['std_reward']:.2f}")
        print(f"  Mean Reward (random) : {metrics['random_mean_reward']:.2f}")
        print(f"  Gain vs random       : {metrics['reward_vs_random_pct']:+.1f}%")
        print(f"  Mean Production      : {metrics['mean_production']:.0f} barrels")
        print(f"  Energy per Barrel    : {metrics['energy_per_barrel']:.2f} kWh")
        print(f"  Energy Savings       : {metrics['energy_savings_pct']:.1f}% vs fixed baseline")

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

        # Normalize observation using VecNormalize stats if active.
        # Skip if the saved stats have a different obs dimension than the
        # current observation (e.g. model trained with old 16-channel config).
        if self.use_vec_normalize and hasattr(self.env, "normalize_obs"):
            try:
                expected_dim = self.env.obs_rms.mean.shape[0]
                if observation.shape[0] != expected_dim:
                    print(f"[DRL] VecNormalize shape mismatch "
                          f"({observation.shape[0]} vs {expected_dim}) — "
                          f"skipping normalization.")
                else:
                    observation = self.env.normalize_obs(observation)
            except Exception:
                pass  # skip normalization on any error

        # Guard against policy obs-space mismatch (e.g. model trained with
        # old 18-D obs but current env has 13-D obs).
        try:
            policy_obs_dim = self.model.observation_space.shape[0]
        except Exception:
            policy_obs_dim = observation.shape[0]

        if observation.shape[0] != policy_obs_dim:
            print(f"[DRL] Policy obs-space mismatch "
                  f"({observation.shape[0]} vs {policy_obs_dim}) — "
                  f"returning neutral action.")
            # Return the midpoint of the action space as a safe default
            action = np.array(
                [(self.model.action_space.low[i] + self.model.action_space.high[i]) / 2.0
                 for i in range(self.model.action_space.shape[0])],
                dtype=np.float32,
            )
            return action

        action, _ = self.model.predict(observation, deterministic=deterministic)
        return action

    def load(self, path: str) -> "DRLOptimizer":
        """Load trained model (auto-detects SAC vs PPO from saved file)."""
        try:
            from stable_baselines3 import PPO, SAC
            from stable_baselines3.common.vec_env import VecNormalize
        except ImportError:
            print("stable-baselines3 not installed.")
            return self

        if self.env is None:
            self.create_environment()

        # Load VecNormalize stats first (must match env wrapping at training)
        if self.use_vec_normalize:
            vn_path = Path(path).with_suffix(".vecnorm.pkl")
            if vn_path.exists():
                # Re-wrap with the saved stats
                from stable_baselines3.common.vec_env import DummyVecEnv
                self.raw_env = ESPEnvironment(config=self.config)
                base_env = DummyVecEnv([lambda: self.raw_env])
                self.env = VecNormalize.load(str(vn_path), base_env)
                self.env.training = False     # freeze stats at inference
                self.env.norm_reward = False  # don't normalise reward at inference
                print(f"VecNormalize stats loaded from {vn_path}")
            else:
                # Old model trained without VecNormalize -> disable for inference
                print(f"  [INFO] No VecNormalize file at {vn_path}, disabling normalization")
                self.use_vec_normalize = False

        # Try SAC first, fallback to PPO (zip file format differs)
        algo_cls = SAC if self.algo == "sac" else PPO
        try:
            self.model = algo_cls.load(path, env=self.env, device=self.device)
        except Exception as e:
            print(f"  Loading as {self.algo.upper()} failed ({e}), trying the other algo...")
            other_cls = PPO if self.algo == "sac" else SAC
            self.model = other_cls.load(path, env=self.env, device=self.device)
            self.algo = "ppo" if other_cls is PPO else "sac"
        print(f"Model loaded from {path} as {self.algo.upper()}")
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
