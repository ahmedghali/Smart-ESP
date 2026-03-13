"""
ESP Physics Simulator
======================
High-fidelity simulation of ESP operation based on pump hydraulics.
Used for digital twin and training data generation.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.config.config import Config, ESPConfig


class ESPStatus(Enum):
    """ESP Operating Status."""
    RUNNING = "running"
    STOPPED = "stopped"
    WARNING = "warning"
    CRITICAL = "critical"
    FAILED = "failed"


@dataclass
class ESPState:
    """Complete state of an ESP system."""
    
    # Operating parameters
    frequency: float = 50.0  # Hz
    choke_position: float = 0.8  # 0-1
    
    # Sensor readings
    motor_temperature: float = 85.0  # °C
    intake_pressure: float = 800.0  # psi
    discharge_pressure: float = 2000.0  # psi
    motor_current: float = 45.0  # Amps
    vibration_x: float = 2.0  # mm/s
    vibration_y: float = 2.0  # mm/s
    vibration_z: float = 1.5  # mm/s
    flow_rate: float = 2000.0  # bpd
    power_consumption: float = 50.0  # kW
    fluid_temperature: float = 70.0  # °C
    casing_pressure: float = 150.0  # psi
    voltage: float = 480.0  # V
    power_factor: float = 0.85
    wellhead_pressure: float = 250.0  # psi
    sand_rate: float = 0.05  # %

    # Health indicators
    equipment_health: float = 1.0  # 0-1
    bearing_health: float = 1.0
    shaft_health: float = 1.0
    motor_health: float = 1.0
    
    # Cumulative metrics
    runtime_hours: float = 0.0
    total_production: float = 0.0  # barrels
    total_energy: float = 0.0  # kWh
    
    # Status
    status: ESPStatus = ESPStatus.RUNNING
    last_maintenance: float = 0.0  # hours ago
    
    def to_dict(self) -> Dict:
        """Convert state to dictionary."""
        return {
            "frequency": self.frequency,
            "choke_position": self.choke_position,
            "motor_temperature": self.motor_temperature,
            "intake_pressure": self.intake_pressure,
            "discharge_pressure": self.discharge_pressure,
            "motor_current": self.motor_current,
            "vibration_x": self.vibration_x,
            "vibration_y": self.vibration_y,
            "vibration_z": self.vibration_z,
            "flow_rate": self.flow_rate,
            "power_consumption": self.power_consumption,
            "fluid_temperature": self.fluid_temperature,
            "casing_pressure": self.casing_pressure,
            "voltage": self.voltage,
            "power_factor": self.power_factor,
            "wellhead_pressure": self.wellhead_pressure,
            "sand_rate": self.sand_rate,
            "equipment_health": self.equipment_health,
            "bearing_health": self.bearing_health,
            "shaft_health": self.shaft_health,
            "motor_health": self.motor_health,
            "runtime_hours": self.runtime_hours,
            "total_production": self.total_production,
            "total_energy": self.total_energy,
            "status": self.status.value if isinstance(self.status, ESPStatus) else self.status,
            "last_maintenance": self.last_maintenance
        }
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array for model input."""
        return np.array([
            self.motor_temperature,
            self.intake_pressure,
            self.discharge_pressure,
            self.motor_current,
            self.vibration_x,
            self.vibration_y,
            self.vibration_z,
            self.flow_rate,
            self.frequency,
            self.power_consumption,
            self.fluid_temperature,
            self.casing_pressure,
            self.voltage,
            self.power_factor,
            self.wellhead_pressure,
            self.sand_rate
        ], dtype=np.float32)


class ESPSimulator:
    """
    Physics-based ESP simulator.
    
    Implements:
    - Pump affinity laws
    - Heat transfer modeling
    - Equipment degradation
    - Sensor noise
    """
    
    def __init__(
        self,
        config: Optional[Config] = None,
        seed: Optional[int] = None
    ):
        """
        Initialize simulator.
        
        Args:
            config: Configuration object
            seed: Random seed
        """
        self.config = config or Config()
        self.esp = self.config.esp
        
        if seed is not None:
            np.random.seed(seed)
        
        # Initialize state
        self.state = ESPState()
        
        # Reservoir properties
        self.reservoir_pressure = 2500.0  # psi
        self.water_cut = 0.3  # fraction
        self.gas_oil_ratio = 150.0  # scf/bbl
        
        # Time tracking
        self.time_step = 1.0  # hours
        self.current_time = 0.0
    
    def reset(self) -> ESPState:
        """Reset simulator to initial state."""
        self.state = ESPState()
        self.current_time = 0.0
        return self.state
    
    def _apply_affinity_laws(self, frequency: float) -> Tuple[float, float, float]:
        """
        Apply pump affinity laws.
        
        Returns:
            Tuple of (flow_rate, head, power)
        """
        freq_ratio = frequency / self.esp.nominal_frequency_hz
        
        # Reference values at nominal frequency
        ref_flow = 2000.0  # bpd
        ref_head = 2000.0  # psi differential
        ref_power = 50.0  # kW
        
        # Affinity laws
        flow = ref_flow * freq_ratio
        head = ref_head * (freq_ratio ** 2)
        power = ref_power * (freq_ratio ** 3)
        
        return flow, head, power
    
    def _compute_flow_rate(self) -> float:
        """Compute actual flow rate considering all factors."""
        freq_ratio = self.state.frequency / self.esp.nominal_frequency_hz
        
        # Base flow from affinity law
        base_flow = 2000.0 * freq_ratio
        
        # Choke effect
        choke_factor = 0.2 + 0.8 * self.state.choke_position
        
        # Health degradation effect
        health_factor = 0.7 + 0.3 * self.state.equipment_health
        
        # Reservoir pressure effect (depletion simulation)
        pressure_factor = self.reservoir_pressure / 2500.0
        
        # Gas effect (reduces efficiency at high GOR)
        gas_factor = 1.0 - 0.1 * (self.gas_oil_ratio / 500.0)
        
        flow = base_flow * choke_factor * health_factor * pressure_factor * gas_factor
        
        # Add noise
        flow += np.random.normal(0, 30)
        
        return max(0, flow)
    
    def _compute_pressures(self) -> Tuple[float, float]:
        """Compute intake and discharge pressures."""
        freq_ratio = self.state.frequency / self.esp.nominal_frequency_hz
        
        # Intake pressure (from reservoir)
        intake = self.reservoir_pressure * 0.35 + np.random.normal(0, 10)
        
        # Discharge pressure (head generated)
        head = 2000.0 * (freq_ratio ** 2) * self.state.equipment_health
        discharge = intake + head + np.random.normal(0, 20)
        
        return intake, discharge
    
    def _compute_motor_temperature(self) -> float:
        """Compute motor temperature based on load and cooling."""
        freq_ratio = self.state.frequency / self.esp.nominal_frequency_hz
        
        # Base temperature from load
        load_temp = 60 + 30 * freq_ratio
        
        # Ambient/fluid cooling
        cooling_effect = 0.3 * (self.state.fluid_temperature - 70)
        
        # Heat from degradation (worn bearings generate more heat)
        degradation_heat = 20 * (1 - self.state.bearing_health)
        
        # Inertia (temperature changes slowly)
        target_temp = load_temp + cooling_effect + degradation_heat
        current_temp = self.state.motor_temperature
        new_temp = current_temp + 0.1 * (target_temp - current_temp)
        
        # Add noise
        new_temp += np.random.normal(0, 1)
        
        return np.clip(new_temp, self.esp.min_motor_temp, self.esp.max_motor_temp + 30)
    
    def _compute_vibration(self) -> Tuple[float, float, float]:
        """Compute vibration levels."""
        freq_ratio = self.state.frequency / self.esp.nominal_frequency_hz
        
        # Base vibration from operation
        base_vib = 1.5 * freq_ratio
        
        # Unbalance from shaft degradation
        shaft_vib = 5 * (1 - self.state.shaft_health)
        
        # Bearing noise
        bearing_vib = 6 * (1 - self.state.bearing_health)
        
        vib_x = base_vib + shaft_vib + np.abs(np.random.normal(0, 0.3))
        vib_y = base_vib + shaft_vib * 0.8 + np.abs(np.random.normal(0, 0.3))
        vib_z = base_vib * 0.7 + bearing_vib + np.abs(np.random.normal(0, 0.2))
        
        return vib_x, vib_y, vib_z
    
    def _compute_power(self) -> float:
        """Compute power consumption."""
        freq_ratio = self.state.frequency / self.esp.nominal_frequency_hz
        
        # Base power from affinity law
        base_power = 50.0 * (freq_ratio ** 3)
        
        # Efficiency loss from degradation
        efficiency_loss = 1 + 0.2 * (1 - self.state.motor_health)
        
        power = base_power * efficiency_loss + np.random.normal(0, 1)
        
        return max(0, power)
    
    def _compute_current(self) -> float:
        """Compute motor current."""
        power = self.state.power_consumption

        # Use state voltage and power factor
        voltage = self.state.voltage if self.state.voltage > 0 else self.esp.nominal_voltage
        power_factor = self.state.power_factor if self.state.power_factor > 0 else self.esp.nominal_power_factor

        current = (power * 1000) / (np.sqrt(3) * voltage * power_factor)
        current += np.random.normal(0, 1)

        return np.clip(current, self.esp.min_current, self.esp.max_current)
    
    def _update_equipment_health(self) -> None:
        """Update equipment health based on operating conditions."""
        # Base degradation rate (per hour)
        base_rate = 0.00001
        
        # Frequency stress (higher frequency = faster wear)
        freq_ratio = self.state.frequency / self.esp.nominal_frequency_hz
        freq_stress = 1 + 2 * max(0, freq_ratio - 1) ** 2
        
        # Temperature stress
        temp_stress = 1 + 0.1 * max(0, self.state.motor_temperature - 100) / 10
        
        # Vibration stress
        vib_mag = np.sqrt(
            self.state.vibration_x**2 + 
            self.state.vibration_y**2 + 
            self.state.vibration_z**2
        )
        vib_stress = 1 + 0.2 * max(0, vib_mag - 5) / 5
        
        # Combined degradation rate
        degradation = base_rate * freq_stress * temp_stress * vib_stress
        
        # Update individual component health
        self.state.bearing_health = max(0, self.state.bearing_health - degradation * 1.2)
        self.state.shaft_health = max(0, self.state.shaft_health - degradation * 0.8)
        self.state.motor_health = max(0, self.state.motor_health - degradation * 1.0)
        
        # Overall health
        self.state.equipment_health = np.mean([
            self.state.bearing_health,
            self.state.shaft_health,
            self.state.motor_health
        ])
    
    def _update_status(self) -> None:
        """Update ESP status based on current conditions."""
        # Check critical conditions
        if self.state.equipment_health <= 0:
            self.state.status = ESPStatus.FAILED
            return
        
        # Temperature check
        if self.state.motor_temperature > self.esp.critical_temp:
            self.state.status = ESPStatus.CRITICAL
            return
        
        # Vibration check
        vib_mag = np.sqrt(
            self.state.vibration_x**2 + 
            self.state.vibration_y**2 + 
            self.state.vibration_z**2
        )
        if vib_mag > self.esp.critical_vibration:
            self.state.status = ESPStatus.CRITICAL
            return
        
        # Warning conditions
        if (self.state.motor_temperature > 120 or 
            vib_mag > self.esp.warning_vibration or
            self.state.equipment_health < 0.5):
            self.state.status = ESPStatus.WARNING
            return
        
        self.state.status = ESPStatus.RUNNING
    
    def step(
        self,
        frequency: Optional[float] = None,
        choke_position: Optional[float] = None,
        dt: float = 1.0
    ) -> ESPState:
        """
        Advance simulation by one time step.
        
        Args:
            frequency: New frequency setpoint (Hz)
            choke_position: New choke position (0-1)
            dt: Time step in hours
            
        Returns:
            Updated ESP state
        """
        # Apply control inputs
        if frequency is not None:
            self.state.frequency = np.clip(
                frequency,
                self.esp.min_frequency_hz,
                self.esp.max_frequency_hz
            )
        
        if choke_position is not None:
            self.state.choke_position = np.clip(choke_position, 0, 1)
        
        # Update physics
        self.state.flow_rate = self._compute_flow_rate()
        self.state.intake_pressure, self.state.discharge_pressure = self._compute_pressures()
        self.state.motor_temperature = self._compute_motor_temperature()
        self.state.vibration_x, self.state.vibration_y, self.state.vibration_z = self._compute_vibration()
        self.state.power_consumption = self._compute_power()
        self.state.motor_current = self._compute_current()
        
        # Update fluid temperature (slight warming from pump)
        self.state.fluid_temperature = 68 + 4 * (self.state.frequency / 50) + np.random.normal(0, 0.5)
        
        # Update casing pressure
        self.state.casing_pressure = 150 + 20 * (self.state.frequency / 50) + np.random.normal(0, 3)

        # Update voltage (slight drop under load)
        freq_ratio_v = self.state.frequency / self.esp.nominal_frequency_hz
        self.state.voltage = self.esp.nominal_voltage - 2 * freq_ratio_v + np.random.normal(0, 2)

        # Update power factor (varies with load)
        self.state.power_factor = np.clip(
            self.esp.nominal_power_factor - 0.05 * (1 - freq_ratio_v) + np.random.normal(0, 0.01),
            self.esp.min_power_factor, self.esp.max_power_factor
        )

        # Update wellhead pressure
        self.state.wellhead_pressure = np.clip(
            250 - 30 * freq_ratio_v + np.random.normal(0, 5),
            self.esp.min_wellhead_pressure, self.esp.max_wellhead_pressure
        )

        # Update sand rate (normally very low, random spikes)
        self.state.sand_rate = max(0, self.esp.normal_sand_rate + np.abs(np.random.normal(0, 0.01)))

        # Update equipment health
        self._update_equipment_health()
        
        # Update cumulative metrics
        self.state.runtime_hours += dt
        self.state.total_production += self.state.flow_rate * dt / 24  # barrels
        self.state.total_energy += self.state.power_consumption * dt  # kWh
        
        # Update status
        self._update_status()
        
        # Update time
        self.current_time += dt
        self.state.last_maintenance += dt
        
        return self.state
    
    def run_simulation(
        self,
        duration_hours: int = 720,
        control_policy: Optional[callable] = None
    ) -> List[Dict]:
        """
        Run simulation for specified duration.
        
        Args:
            duration_hours: Simulation duration in hours
            control_policy: Optional function(state) -> (frequency, choke)
            
        Returns:
            List of state dictionaries
        """
        self.reset()
        history = []
        
        for _ in range(duration_hours):
            # Apply control policy if provided
            if control_policy is not None:
                frequency, choke = control_policy(self.state)
            else:
                frequency = self.state.frequency
                choke = self.state.choke_position
            
            # Step simulation
            self.step(frequency=frequency, choke_position=choke)
            
            # Record state
            history.append(self.state.to_dict())
            
            # Check for failure
            if self.state.status == ESPStatus.FAILED:
                break
        
        return history
    
    def get_kpi_summary(self) -> Dict:
        """Get summary of Key Performance Indicators."""
        avg_efficiency = self.state.total_production / (self.state.total_energy + 1e-6)
        
        return {
            "total_runtime_hours": self.state.runtime_hours,
            "total_production_barrels": self.state.total_production,
            "total_energy_kwh": self.state.total_energy,
            "average_efficiency_bbl_per_kwh": avg_efficiency,
            "final_equipment_health": self.state.equipment_health,
            "final_status": self.state.status.value if isinstance(self.state.status, ESPStatus) else self.state.status
        }


if __name__ == "__main__":
    # Test simulator
    print("Testing ESP Simulator...")
    
    sim = ESPSimulator(seed=42)
    
    # Run short simulation
    print("\nRunning 100-hour simulation...")
    for i in range(100):
        state = sim.step(frequency=52, choke_position=0.85)
        
        if i % 20 == 0:
            print(f"Hour {i}: Flow={state.flow_rate:.0f} bpd, "
                  f"Temp={state.motor_temperature:.1f}°C, "
                  f"Health={state.equipment_health:.3f}")
    
    # Get KPIs
    kpis = sim.get_kpi_summary()
    print("\nKPI Summary:")
    for key, value in kpis.items():
        print(f"  {key}: {value}")
    
    print("\nESP Simulator test passed!")
