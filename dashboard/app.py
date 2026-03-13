"""
ESP AI Optimization Dashboard
==============================
Streamlit-based Digital Twin dashboard for ESP monitoring and optimization.

Features:
- Real-time sensor visualization
- AI prediction displays
- SHAP/LIME explainability charts
- What-if scenario analysis
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import sys
from pathlib import Path
import time

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.config.config import Config
from src.digital_twin.esp_simulator import ESPSimulator, ESPState, ESPStatus
from src.digital_twin.twin_engine import DigitalTwinEngine

# Page configuration
st.set_page_config(
    page_title="ESP AI Optimization",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        padding: 1rem;
        color: white;
    }
    .status-running {
        color: #4CAF50;
        font-weight: bold;
    }
    .status-warning {
        color: #FF9800;
        font-weight: bold;
    }
    .status-critical {
        color: #F44336;
        font-weight: bold;
    }
    .recommendation-box {
        background-color: #E3F2FD;
        border-left: 4px solid #1976D2;
        padding: 1rem;
        border-radius: 0 8px 8px 0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def initialize_engine():
    """Initialize the digital twin engine."""
    config = Config()
    engine = DigitalTwinEngine(config=config)
    return engine


def create_gauge_chart(value: float, title: str, max_val: float,
                       warning_threshold: float, critical_threshold: float,
                       invert: bool = False) -> go.Figure:
    """Create a gauge chart for displaying sensor values.

    Args:
        invert: If True, lower values are worse (e.g. flow rate).
    """
    if invert:
        # Lower is worse: below critical = red, below warning = orange, above = green
        if value < critical_threshold:
            color = "red"
        elif value < warning_threshold:
            color = "orange"
        else:
            color = "green"
        steps = [
            {'range': [0, critical_threshold], 'color': '#FFEBEE'},
            {'range': [critical_threshold, warning_threshold], 'color': '#FFF3E0'},
            {'range': [warning_threshold, max_val], 'color': '#E8F5E9'}
        ]
        threshold_val = critical_threshold
    else:
        # Higher is worse: above critical = red, above warning = orange, below = green
        if value < warning_threshold:
            color = "green"
        elif value < critical_threshold:
            color = "orange"
        else:
            color = "red"
        steps = [
            {'range': [0, warning_threshold], 'color': '#E8F5E9'},
            {'range': [warning_threshold, critical_threshold], 'color': '#FFF3E0'},
            {'range': [critical_threshold, max_val], 'color': '#FFEBEE'}
        ]
        threshold_val = critical_threshold

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 16}},
        gauge={
            'axis': {'range': [None, max_val], 'tickwidth': 1},
            'bar': {'color': color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': steps,
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': threshold_val
            }
        }
    ))
    fig.update_layout(height=200, margin=dict(l=20, r=20, t=40, b=20))
    return fig


def create_time_series_chart(history: list, columns: list, title: str) -> go.Figure:
    """Create time series chart for sensor history."""
    if not history:
        return go.Figure()
    
    df = pd.DataFrame(history)
    
    fig = make_subplots(rows=len(columns), cols=1, shared_xaxes=True,
                        vertical_spacing=0.05)
    
    colors = px.colors.qualitative.Set2
    
    for i, col in enumerate(columns):
        if col in df.columns:
            fig.add_trace(
                go.Scatter(x=list(range(len(df))), y=df[col], 
                          mode='lines', name=col.replace('_', ' ').title(),
                          line=dict(color=colors[i % len(colors)])),
                row=i+1, col=1
            )
            fig.update_yaxes(title_text=col.replace('_', ' ').title(), row=i+1, col=1)
    
    fig.update_layout(height=100 * len(columns), title=title,
                      showlegend=True, legend=dict(orientation="h"))
    fig.update_xaxes(title_text="Time (hours)", row=len(columns), col=1)
    
    return fig


def create_contribution_chart(contributions: dict) -> go.Figure:
    """Create horizontal bar chart for feature contributions."""
    sorted_items = sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
    
    features = [item[0].replace('_', ' ').title() for item in sorted_items]
    values = [item[1] for item in sorted_items]
    colors = ['#EF5350' if v > 0 else '#42A5F5' for v in values]
    
    fig = go.Figure(go.Bar(
        x=values,
        y=features,
        orientation='h',
        marker_color=colors
    ))
    
    fig.update_layout(
        title="Feature Contributions to Prediction",
        xaxis_title="Contribution",
        yaxis_title="",
        height=400,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig


def create_radar_chart(contributions: dict) -> go.Figure:
    """Create radar chart for sensor importance."""
    sorted_items = sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)[:8]
    
    labels = [item[0].replace('_', ' ').title() for item in sorted_items]
    values = [abs(item[1]) for item in sorted_items]
    
    # Normalize
    max_val = max(values) if values else 1
    values = [v / max_val * 100 for v in values]
    
    # Complete the loop
    labels.append(labels[0])
    values.append(values[0])
    
    fig = go.Figure(go.Scatterpolar(
        r=values,
        theta=labels,
        fill='toself',
        name='Sensor Importance'
    ))
    
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=False,
        height=400,
        title="Sensor Importance"
    )
    
    return fig


def main():
    """Main dashboard application."""
    
    # Header
    st.markdown('<p class="main-header">⚙️ ESP AI Optimization Dashboard</p>', 
                unsafe_allow_html=True)
    st.markdown("---")
    
    # Initialize engine
    engine = initialize_engine()
    
    # Sidebar controls
    with st.sidebar:
        st.header("🎛️ Control Panel")
        
        # Simulation controls
        st.subheader("Simulation")
        auto_run = st.checkbox("Auto-run simulation", value=False)
        
        if st.button("Step Simulation", type="primary"):
            engine.simulator.step()
            engine.sync_state(engine.simulator.state.to_dict())
            st.rerun()
        
        if st.button("Reset Simulation"):
            engine.simulator.reset()
            engine.current_state = engine.simulator.state
            engine.state_history.clear()
            engine.alert_history.clear()
            engine.recommendation_history.clear()
            st.rerun()
        
        st.markdown("---")
        
        # Manual controls
        st.subheader("ESP Controls")
        frequency = st.slider(
            "Frequency (Hz)",
            min_value=20,
            max_value=60,
            value=int(engine.current_state.frequency),
            step=1
        )
        
        choke = st.slider(
            "Choke Position",
            min_value=0.0,
            max_value=1.0,
            value=float(engine.current_state.choke_position),
            step=0.05
        )
        
        if st.button("Apply Settings"):
            engine.simulator.step(frequency=frequency, choke_position=choke)
            engine.sync_state(engine.simulator.state.to_dict())
            st.rerun()
        
        st.markdown("---")
        
        # What-if analysis
        st.subheader("What-If Analysis")
        what_if_hours = st.slider("Prediction Hours", 1, 72, 24)
        what_if_freq = st.number_input("Hypothetical Frequency", 20, 60, 50)
        
        if st.button("Run What-If"):
            predictions = engine.what_if_analysis(
                frequency=what_if_freq,
                hours=what_if_hours
            )
            st.session_state['what_if_results'] = predictions
    
    # Main content
    # Get current dashboard data
    dashboard_data = engine.get_dashboard_data()
    
    # Row 1: Status and KPIs
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        status = dashboard_data['status']
        status_class = f"status-{status}"
        st.markdown(f"### Status: <span class='{status_class}'>{status.upper()}</span>", 
                    unsafe_allow_html=True)
    
    with col2:
        failure_prob = dashboard_data['failure_prediction']['probability']
        st.metric("Failure Probability", f"{failure_prob:.1%}",
                  delta=None)
    
    with col3:
        health = dashboard_data['current_state']['equipment_health']
        st.metric("Equipment Health", f"{health:.1%}",
                  delta=f"{(health - 1) * 100:.1f}%")
    
    with col4:
        runtime = dashboard_data['kpis']['total_runtime_hours']
        st.metric("Runtime", f"{runtime:.0f} hours")
    
    st.markdown("---")
    
    # Row 2: Gauges for critical sensors
    st.subheader("📊 Critical Sensors")
    
    gauge_cols = st.columns(4)
    
    with gauge_cols[0]:
        temp = dashboard_data['current_state']['motor_temperature']
        fig = create_gauge_chart(temp, "Motor Temp (°C)", 160, 110, 130)
        st.plotly_chart(fig, use_container_width=True)
    
    with gauge_cols[1]:
        vib_x = dashboard_data['current_state']['vibration_x']
        vib_y = dashboard_data['current_state']['vibration_y']
        vib_z = dashboard_data['current_state']['vibration_z']
        vib_mag = np.sqrt(vib_x**2 + vib_y**2 + vib_z**2)
        fig = create_gauge_chart(vib_mag, "Vibration (mm/s)", 15, 6, 10)
        st.plotly_chart(fig, use_container_width=True)
    
    with gauge_cols[2]:
        current = dashboard_data['current_state']['motor_current']
        fig = create_gauge_chart(current, "Motor Current (A)", 150, 80, 100)
        st.plotly_chart(fig, use_container_width=True)
    
    with gauge_cols[3]:
        flow = dashboard_data['current_state']['flow_rate']
        fig = create_gauge_chart(flow, "Flow Rate (bpd)", 5000, 1500, 800, invert=True)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Row 3: AI Insights
    st.subheader("🤖 AI Insights")
    
    insight_cols = st.columns(2)
    
    with insight_cols[0]:
        st.markdown("#### Failure Prediction")
        
        # Failure probability gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=failure_prob * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "48-Hour Failure Risk (%)"},
            delta={'reference': 30},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 30], 'color': "#E8F5E9"},
                    {'range': [30, 70], 'color': "#FFF3E0"},
                    {'range': [70, 100], 'color': "#FFEBEE"}
                ]
            }
        ))
        fig.update_layout(height=250)
        st.plotly_chart(fig, use_container_width=True)
        
        # Feature contributions
        contributions = dashboard_data['failure_prediction']['contributions']
        if contributions:
            fig = create_contribution_chart(contributions)
            st.plotly_chart(fig, use_container_width=True)
    
    with insight_cols[1]:
        st.markdown("#### Anomaly Detection")
        
        is_anomaly = dashboard_data['anomaly_detection']['is_anomaly']
        anomaly_score = dashboard_data['anomaly_detection']['score']
        
        if is_anomaly:
            st.error(f"⚠️ Anomaly Detected! Score: {anomaly_score:.3f}")
        else:
            st.success(f"✓ Normal Operation. Score: {anomaly_score:.3f}")
        
        # Anomaly contributions radar
        anomaly_contrib = dashboard_data['anomaly_detection']['contributions']
        if anomaly_contrib:
            fig = create_radar_chart(anomaly_contrib)
            st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Row 4: Recommendation
    st.subheader("💡 AI Recommendation")
    
    rec = dashboard_data['recommendation']
    
    st.markdown(f"""
    <div class="recommendation-box">
        <strong>Recommended Action:</strong> {rec['action'].replace('_', ' ').title()}<br>
        <strong>Current Value:</strong> {rec['current']:.1f}<br>
        <strong>Recommended Value:</strong> {rec['recommended']:.1f}<br>
        <strong>Confidence:</strong> {rec['confidence']:.1%}<br>
        <strong>Reasoning:</strong> {rec['reasoning']}
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Row 5: Time Series
    st.subheader("📈 Sensor History")
    
    if engine.state_history:
        # Select which sensors to display
        sensor_options = ['motor_temperature', 'vibration_x', 'motor_current', 
                         'flow_rate', 'discharge_pressure', 'power_consumption']
        selected_sensors = st.multiselect(
            "Select sensors to display",
            sensor_options,
            default=['motor_temperature', 'vibration_x', 'flow_rate']
        )
        
        if selected_sensors:
            fig = create_time_series_chart(engine.state_history, selected_sensors, 
                                          "Sensor Time Series")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Run simulation to see sensor history.")
    
    # Row 6: Alerts
    st.subheader("🚨 Recent Alerts")
    
    alerts = dashboard_data['recent_alerts']
    if alerts:
        for alert in alerts:
            if alert['level'] == 'critical':
                st.error(f"🔴 {alert['message']} - {alert['timestamp']}")
            elif alert['level'] == 'warning':
                st.warning(f"🟡 {alert['message']} - {alert['timestamp']}")
            else:
                st.info(f"🔵 {alert['message']} - {alert['timestamp']}")
    else:
        st.success("No recent alerts.")
    
    # What-if results
    if 'what_if_results' in st.session_state and st.session_state['what_if_results']:
        st.markdown("---")
        st.subheader("🔮 What-If Analysis Results")
        
        results = st.session_state['what_if_results']
        df = pd.DataFrame(results)
        
        # Show key predictions
        col1, col2, col3 = st.columns(3)
        
        with col1:
            final_health = df['equipment_health'].iloc[-1]
            st.metric("Predicted Health", f"{final_health:.1%}")
        
        with col2:
            avg_temp = df['motor_temperature'].mean()
            st.metric("Avg Temperature", f"{avg_temp:.1f}°C")
        
        with col3:
            total_prod = df['flow_rate'].sum() / 24
            st.metric("Total Production", f"{total_prod:.0f} bbl")
        
        # Plot predictions
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True)
        fig.add_trace(go.Scatter(y=df['equipment_health'], mode='lines', 
                                name='Equipment Health'), row=1, col=1)
        fig.add_trace(go.Scatter(y=df['motor_temperature'], mode='lines',
                                name='Motor Temperature'), row=2, col=1)
        fig.update_layout(height=400, title="What-If Scenario Predictions")
        st.plotly_chart(fig, use_container_width=True)
    
    # Auto-run simulation (non-blocking with st.empty placeholder)
    if auto_run:
        placeholder = st.empty()
        placeholder.info("Auto-running simulation... Uncheck 'Auto-run' to stop.")
        engine.simulator.step()
        engine.sync_state(engine.simulator.state.to_dict())
        time.sleep(0.5)
        st.rerun()


if __name__ == "__main__":
    main()
