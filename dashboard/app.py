"""
ESP AI Digital Twin Dashboard
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.config.config import Config
from src.digital_twin.twin_engine import DigitalTwinEngine

try:
    from streamlit_autorefresh import st_autorefresh
    HAS_AUTOREFRESH = True
except ImportError:
    HAS_AUTOREFRESH = False

st.set_page_config(
    page_title="ESP Digital Twin",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* Hide Streamlit chrome that overlaps content */
  #MainMenu { visibility: hidden; }
  header[data-testid="stHeader"] { display: none; }
  footer { visibility: hidden; }

  /* Global */
  .block-container { padding: 0.5rem 1.2rem 0.5rem 1.2rem; max-width: 100%; }
  body { font-family: 'Segoe UI', sans-serif; }

  /* Header */
  .dash-header {
    background: linear-gradient(90deg, #0d1b2a 0%, #1b2838 60%, #0d1b2a 100%);
    border-bottom: 2px solid #00b4d8;
    padding: 0.5rem 1.5rem;
    display: flex; align-items: center; justify-content: space-between;
    border-radius: 8px; margin-bottom: 0.5rem;
  }
  .dash-title { color: #00b4d8; font-size: 1.4rem; font-weight: 700; letter-spacing: 1px; }
  .dash-subtitle { color: #8ecae6; font-size: 0.78rem; margin-top: 2px; }

  /* KPI cards */
  .kpi-row { display: flex; gap: 0.5rem; margin-bottom: 0.5rem; }
  .kpi-card {
    flex: 1; background: #1b2838; border: 1px solid #2a3f5f;
    border-radius: 8px; padding: 0.55rem 0.8rem;
    display: flex; flex-direction: column; align-items: center;
  }
  .kpi-label { color: #8ecae6; font-size: 0.7rem; text-transform: uppercase; letter-spacing: 1px; }
  .kpi-value { color: #ffffff; font-size: 1.4rem; font-weight: 700; line-height: 1.2; }
  .kpi-value.green  { color: #4ade80; }
  .kpi-value.orange { color: #fb923c; }
  .kpi-value.red    { color: #f87171; }

  /* Status badge */
  .badge {
    display: inline-block; padding: 0.15rem 0.6rem;
    border-radius: 20px; font-size: 0.75rem; font-weight: 700; letter-spacing: 0.5px;
  }
  .badge-running  { background: #14532d; color: #4ade80; border: 1px solid #4ade80; }
  .badge-warning  { background: #7c2d12; color: #fb923c; border: 1px solid #fb923c; }
  .badge-critical { background: #450a0a; color: #f87171; border: 1px solid #f87171; }
  .badge-failed   { background: #1c0a0a; color: #dc2626; border: 1px solid #dc2626; }

  /* Alert banners */
  .alert-critical {
    background: #450a0a; border: 2px solid #dc2626; border-left: 6px solid #dc2626;
    color: #fca5a5; border-radius: 6px; padding: 0.6rem 1rem; margin-bottom: 0.3rem;
    font-weight: 600; font-size: 0.88rem;
    animation: pulse-red 1.5s infinite;
  }
  .alert-warning {
    background: #431407; border: 2px solid #f97316; border-left: 6px solid #f97316;
    color: #fdba74; border-radius: 6px; padding: 0.5rem 1rem; margin-bottom: 0.3rem;
    font-size: 0.85rem;
  }
  .alert-info {
    background: #0c1a2e; border: 1px solid #0ea5e9; border-left: 5px solid #0ea5e9;
    color: #7dd3fc; border-radius: 6px; padding: 0.45rem 1rem; margin-bottom: 0.3rem;
    font-size: 0.82rem;
  }
  .no-alerts {
    background: #052e16; border: 1px solid #4ade80;
    color: #4ade80; border-radius: 6px; padding: 0.5rem 1rem;
    font-size: 0.85rem; text-align: center;
  }
  @keyframes pulse-red {
    0%   { box-shadow: 0 0 0 0 rgba(220,38,38,0.7); }
    70%  { box-shadow: 0 0 0 8px rgba(220,38,38,0); }
    100% { box-shadow: 0 0 0 0 rgba(220,38,38,0); }
  }

  /* Fault active banner */
  .fault-banner {
    background: #450a0a; border: 2px solid #dc2626;
    color: #fca5a5; border-radius: 8px; padding: 0.4rem 0.8rem;
    font-size: 0.8rem; font-weight: 700; text-align: center;
    animation: pulse-red 1.5s infinite; margin-top: 0.3rem;
  }

  /* Recommendation card */
  .rec-card {
    background: #0c1a2e; border: 1px solid #0ea5e9; border-left: 5px solid #0ea5e9;
    border-radius: 8px; padding: 0.65rem 1rem; margin-top: 0.4rem;
  }
  .rec-title { color: #7dd3fc; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 1px; }
  .rec-action { color: #ffffff; font-size: 1rem; font-weight: 700; margin: 0.2rem 0; }
  .rec-detail { color: #93c5fd; font-size: 0.8rem; }
  .rec-reason { color: #cbd5e1; font-size: 0.78rem; margin-top: 0.3rem; font-style: italic; }

  /* Section labels */
  .section-label {
    color: #8ecae6; font-size: 0.72rem; text-transform: uppercase;
    letter-spacing: 1.5px; font-weight: 600; margin-bottom: 0.2rem;
    border-bottom: 1px solid #2a3f5f; padding-bottom: 0.15rem;
  }

  /* Tabs override */
  .stTabs [data-baseweb="tab-list"] { background: #0d1b2a; border-radius: 8px 8px 0 0; }
  .stTabs [data-baseweb="tab"] {
    color: #8ecae6; font-size: 0.82rem; font-weight: 600;
    padding: 0.4rem 1.1rem; border-radius: 6px 6px 0 0;
  }
  .stTabs [aria-selected="true"] { color: #00b4d8 !important; border-bottom: 2px solid #00b4d8; }
  .stTabs [data-baseweb="tab-panel"] {
    background: #0d1b2a; border: 1px solid #2a3f5f;
    border-top: none; border-radius: 0 0 8px 8px; padding: 0.6rem;
  }

  /* Sidebar */
  [data-testid="stSidebar"] { background: #0d1b2a; }
  [data-testid="stSidebar"] .stButton button {
    width: 100%; border-radius: 6px; font-weight: 600;
  }

  /* Metric overrides */
  [data-testid="stMetricValue"] { font-size: 1.1rem; color: #e2e8f0; }
  [data-testid="stMetricLabel"] { font-size: 0.72rem; color: #8ecae6; }

  /* Info / success / error override in dark theme */
  .stAlert { border-radius: 6px; }
</style>
""", unsafe_allow_html=True)

# ── Audio alert (JS) ─────────────────────────────────────────────────────────
ALERT_SOUND_JS = """
<script>
function playAlertBeep(type) {
  var ctx = new (window.AudioContext || window.webkitAudioContext)();
  var freqs = type === 'critical' ? [880, 660, 880] : [440, 0, 440];
  var t = ctx.currentTime;
  freqs.forEach(function(f, i) {
    if (f === 0) return;
    var o = ctx.createOscillator();
    var g = ctx.createGain();
    o.connect(g); g.connect(ctx.destination);
    o.type = type === 'critical' ? 'sawtooth' : 'sine';
    o.frequency.value = f;
    g.gain.setValueAtTime(0.25, t + i * 0.22);
    g.gain.exponentialRampToValueAtTime(0.001, t + i * 0.22 + 0.2);
    o.start(t + i * 0.22);
    o.stop(t + i * 0.22 + 0.25);
  });
}
</script>
"""

def inject_sound(alert_type: str):
    """Trigger browser beep for critical or warning alert."""
    st.components.v1.html(
        f'{ALERT_SOUND_JS}<script>playAlertBeep("{alert_type}");</script>',
        height=0,
    )


# ── Engine ────────────────────────────────────────────────────────────────────
def get_engine() -> DigitalTwinEngine:
    """Return the engine stored in session state, creating it once per session."""
    if "engine" not in st.session_state:
        st.session_state["engine"] = DigitalTwinEngine(config=Config())
    return st.session_state["engine"]


# ── Chart helpers ─────────────────────────────────────────────────────────────
def gauge(value, title, max_val, warn, crit, invert=False, unit=""):
    if invert:
        color = "#f87171" if value < crit else "#fb923c" if value < warn else "#4ade80"
        steps = [
            {"range": [0, crit],     "color": "#2d0a0a"},
            {"range": [crit, warn],  "color": "#2d1a0a"},
            {"range": [warn, max_val],"color": "#0a2d1a"},
        ]
    else:
        color = "#4ade80" if value < warn else "#fb923c" if value < crit else "#f87171"
        steps = [
            {"range": [0, warn],     "color": "#0a2d1a"},
            {"range": [warn, crit],  "color": "#2d1a0a"},
            {"range": [crit, max_val],"color": "#2d0a0a"},
        ]
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={"font": {"color": color, "size": 22}, "suffix": unit},
        title={"text": title, "font": {"size": 12, "color": "#8ecae6"}},
        gauge={
            "axis": {"range": [0, max_val], "tickcolor": "#4a6fa5",
                     "tickfont": {"size": 9, "color": "#8ecae6"}},
            "bar": {"color": color, "thickness": 0.25},
            "bgcolor": "#0d1b2a",
            "borderwidth": 1, "bordercolor": "#2a3f5f",
            "steps": steps,
            "threshold": {"line": {"color": "#dc2626", "width": 3},
                          "thickness": 0.8, "value": crit},
        },
    ))
    fig.update_layout(
        height=170, paper_bgcolor="#0d1b2a", plot_bgcolor="#0d1b2a",
        margin=dict(l=12, r=12, t=38, b=8),
    )
    return fig


def risk_gauge(value_pct):
    color = "#4ade80" if value_pct < 30 else "#fb923c" if value_pct < 70 else "#f87171"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value_pct,
        number={"font": {"color": color, "size": 28}, "suffix": "%"},
        title={"text": "48-Hour Failure Risk", "font": {"size": 13, "color": "#8ecae6"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#4a6fa5",
                     "tickfont": {"size": 9, "color": "#8ecae6"}},
            "bar": {"color": color, "thickness": 0.3},
            "bgcolor": "#0d1b2a",
            "borderwidth": 1, "bordercolor": "#2a3f5f",
            "steps": [
                {"range": [0,  30], "color": "#052e16"},
                {"range": [30, 70], "color": "#2d1a0a"},
                {"range": [70, 100],"color": "#2d0a0a"},
            ],
        },
    ))
    fig.update_layout(
        height=200, paper_bgcolor="#0d1b2a", plot_bgcolor="#0d1b2a",
        margin=dict(l=12, r=12, t=42, b=8),
    )
    return fig


def contribution_chart(contributions: dict):
    items = sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)[:8]
    features = [i[0].replace("_", " ").title() for i in items]
    values   = [i[1] for i in items]
    colors   = ["#f87171" if v > 0 else "#60a5fa" for v in values]
    fig = go.Figure(go.Bar(
        x=values, y=features, orientation="h",
        marker_color=colors, marker_line_width=0,
    ))
    fig.update_layout(
        title={"text": "SHAP Feature Contributions", "font": {"size": 11, "color": "#8ecae6"}},
        xaxis={"title": "Contribution", "color": "#8ecae6", "gridcolor": "#1e3a5f"},
        yaxis={"color": "#8ecae6", "gridcolor": "#1e3a5f"},
        height=240, paper_bgcolor="#0d1b2a", plot_bgcolor="#0d1b2a",
        font={"color": "#cbd5e1"}, margin=dict(l=10, r=10, t=35, b=10),
    )
    return fig


def radar_chart(contributions: dict):
    items = sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)[:8]
    if not items:
        return go.Figure()
    labels = [i[0].replace("_", " ").title() for i in items]
    values = [abs(i[1]) for i in items]
    mx = max(values) or 1
    values = [v / mx * 100 for v in values]
    labels.append(labels[0]); values.append(values[0])
    fig = go.Figure(go.Scatterpolar(
        r=values, theta=labels, fill="toself",
        line_color="#f97316", fillcolor="rgba(249,115,22,0.15)",
        name="Anomaly Contribution",
    ))
    fig.update_layout(
        polar=dict(
            bgcolor="#0d1b2a",
            radialaxis=dict(visible=True, range=[0, 100], color="#4a6fa5",
                            gridcolor="#1e3a5f", tickfont={"size": 8}),
            angularaxis=dict(color="#8ecae6"),
        ),
        title={"text": "Anomaly Sensor Contributions", "font": {"size": 11, "color": "#8ecae6"}},
        showlegend=False, height=260,
        paper_bgcolor="#0d1b2a", plot_bgcolor="#0d1b2a",
        font={"color": "#cbd5e1"}, margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig


def time_series_chart(history: list, columns: list):
    if not history:
        return go.Figure()
    df = pd.DataFrame(history)
    palette = ["#60a5fa","#f97316","#4ade80","#fb923c","#a78bfa","#f472b6"]
    fig = make_subplots(rows=len(columns), cols=1, shared_xaxes=True,
                        vertical_spacing=0.04)
    for i, col in enumerate(columns):
        if col in df.columns:
            fig.add_trace(
                go.Scatter(x=list(range(len(df))), y=df[col], mode="lines",
                           name=col.replace("_", " ").title(),
                           line=dict(color=palette[i % len(palette)], width=1.5)),
                row=i+1, col=1,
            )
            fig.update_yaxes(title_text=col.replace("_"," ").title(),
                             title_font={"size": 9, "color": "#8ecae6"},
                             color="#8ecae6", gridcolor="#1e3a5f",
                             row=i+1, col=1)
    fig.update_xaxes(title_text="Simulation Steps", color="#8ecae6",
                     gridcolor="#1e3a5f", row=len(columns), col=1)
    fig.update_layout(
        height=max(220, 72 * len(columns)),
        paper_bgcolor="#0d1b2a", plot_bgcolor="#0d1b2a",
        font={"color": "#cbd5e1"},
        legend=dict(orientation="h", font={"size": 9}, bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=10, r=10, t=20, b=10),
    )
    return fig


# ── Badge helper ──────────────────────────────────────────────────────────────
def status_badge(status: str) -> str:
    cls = {"running": "running", "warning": "warning",
           "critical": "critical", "failed": "failed"}.get(status, "running")
    icon = {"running": "▶", "warning": "⚠", "critical": "🔴", "failed": "💀"}.get(status, "▶")
    return f'<span class="badge badge-{cls}">{icon} {status.upper()}</span>'


# ── Fault scenarios (English) ─────────────────────────────────────────────────
FAULT_SCENARIOS = {
    "motor_overheat": {
        "label": "🌡️ Motor Overheating",
        "desc": (
            "**What happens:** The cooling system fails — motor temperature climbs above 142 °C "
            "(critical threshold is 140 °C).\n\n"
            "**Everyday analogy:** Like a car engine with a blocked radiator. "
            "Continued operation burns the motor windings.\n\n"
            "**What AI should detect:** High failure probability · Thermal anomaly signature."
        ),
    },
    "sand_ingestion": {
        "label": "🏜️ Sand Ingestion",
        "desc": (
            "**What happens:** The well starts producing sand alongside oil. "
            "Sand acts like sandpaper on rotating parts, eroding impellers and the shaft.\n\n"
            "**Everyday analogy:** Running a blender full of gravel.\n\n"
            "**What AI should detect:** Abnormal vibration spike · Accelerated shaft wear."
        ),
    },
    "bearing_wear": {
        "label": "⚙️ Bearing Wear",
        "desc": (
            "**What happens:** End-of-life bearings generate friction, heat, and vibration. "
            "Bearing health drops to 25 %.\n\n"
            "**Everyday analogy:** A bicycle wheel with worn bearings grinding on every turn.\n\n"
            "**What AI should detect:** Elevated Z-axis vibration · Temperature rise."
        ),
    },
    "voltage_drop": {
        "label": "⚡ Voltage Drop",
        "desc": (
            "**What happens:** Grid supply drops to 380 V (nominal 480 V). "
            "The motor draws more current to maintain the same power.\n\n"
            "**Everyday analogy:** Plugging a high-power device through a thin extension cord — "
            "the wire heats up and voltage sags.\n\n"
            "**What AI should detect:** Electrical anomaly · Over-current risk."
        ),
    },
    "gas_lock": {
        "label": "💨 Gas Lock",
        "desc": (
            "**What happens:** Too much free gas enters the pump intake. "
            "The pump churns gas instead of liquid — flow collapses.\n\n"
            "**Everyday analogy:** Drinking with a straw in an almost-empty glass — "
            "you suck air instead of liquid.\n\n"
            "**What AI should detect:** Flow rate collapse · Unstable discharge pressure."
        ),
    },
    "plugged_choke": {
        "label": "🚧 Plugged Choke / Restriction",
        "desc": (
            "**What happens:** The surface choke valve is partially blocked by wax or asphaltene "
            "deposits, restricting flow. Upstream pressure builds.\n\n"
            "**Everyday analogy:** A kinked garden hose — pressure rises before the kink, "
            "flow drops after it.\n\n"
            "**What AI should detect:** High discharge pressure · Reduced production."
        ),
    },
    "shaft_imbalance": {
        "label": "🔩 Shaft Imbalance",
        "desc": (
            "**What happens:** The pump shaft is mechanically unbalanced (deformation or asymmetric "
            "deposits). Severe lateral vibrations develop.\n\n"
            "**Everyday analogy:** A washing machine with laundry bunched on one side during spin — "
            "it shakes violently.\n\n"
            "**What AI should detect:** Very high X/Y vibration · Rapid degradation."
        ),
    },
    "reservoir_depletion": {
        "label": "📉 Reservoir Depletion",
        "desc": (
            "**What happens:** The reservoir pressure has declined (ageing field). "
            "Less fluid reaches the pump intake.\n\n"
            "**Everyday analogy:** A water table that has dropped — the well starts delivering less.\n\n"
            "**What AI should detect:** Low intake pressure · Reduced flow rate."
        ),
    },
}


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    engine = get_engine()

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### 🎛️ Control Panel")

        # Simulation controls
        st.markdown('<div class="section-label">Simulation</div>', unsafe_allow_html=True)
        auto_run = st.checkbox("Auto-run (live)", value=False)
        c1, c2 = st.columns(2)
        with c1:
            if st.button("▶ Step", type="primary", use_container_width=True):
                engine.simulator.step()
                engine.sync_state(engine.simulator.state.to_dict())
                st.rerun()
        with c2:
            if st.button("↺ Reset", use_container_width=True):
                # Drop the engine from session_state so get_engine() rebuilds it fresh
                st.session_state.pop("engine", None)
                st.session_state.pop("last_fault", None)
                st.session_state.pop("sound_played", None)
                st.session_state.pop("what_if_results", None)
                st.rerun()

        st.markdown("---")

        # ESP manual controls
        st.markdown('<div class="section-label">ESP Setpoints</div>', unsafe_allow_html=True)
        frequency = st.slider("Frequency (Hz)", 20, 60,
                               int(engine.current_state.frequency), step=1)
        choke = st.slider("Choke Position", 0.0, 1.0,
                           float(engine.current_state.choke_position), step=0.05)
        if st.button("Apply Setpoints", use_container_width=True):
            engine.simulator.step(frequency=frequency, choke_position=choke)
            engine.sync_state(engine.simulator.state.to_dict())
            st.rerun()

        st.markdown("---")

        # Fault injection
        st.markdown('<div class="section-label">🔥 Fault Injection</div>',
                    unsafe_allow_html=True)
        st.caption("Select a fault scenario, read the explanation, then inject. "
                   "Step the simulation to watch AI models react.")

        scenario_key = st.selectbox(
            "Scenario",
            options=list(FAULT_SCENARIOS.keys()),
            format_func=lambda k: FAULT_SCENARIOS[k]["label"],
        )
        with st.expander("ℹ️ Scenario Details", expanded=True):
            st.markdown(FAULT_SCENARIOS[scenario_key]["desc"])

        fc1, fc2 = st.columns(2)
        with fc1:
            if st.button("💥 Inject", type="secondary", use_container_width=True):
                engine.simulator.inject_fault(scenario_key)
                engine.sync_state(engine.simulator.state.to_dict())
                st.session_state["last_fault"] = FAULT_SCENARIOS[scenario_key]["label"]
                st.session_state.pop("sound_played", None)
                st.rerun()
        with fc2:
            if st.button("✓ Clear Fault", use_container_width=True):
                engine.simulator.clear_fault()
                engine.sync_state(engine.simulator.state.to_dict())
                st.session_state.pop("last_fault", None)
                st.session_state.pop("sound_played", None)
                st.rerun()

        if "last_fault" in st.session_state:
            st.markdown(
                f'<div class="fault-banner">⚠️ ACTIVE FAULT<br>{st.session_state["last_fault"]}</div>',
                unsafe_allow_html=True,
            )

        st.markdown("---")

        # What-if
        st.markdown('<div class="section-label">🔮 What-If Analysis</div>',
                    unsafe_allow_html=True)
        what_if_hours = st.slider("Horizon (hours)", 1, 72, 24)
        what_if_freq  = st.number_input("Test Frequency (Hz)", 20, 60, 50)
        if st.button("Run Simulation", use_container_width=True):
            preds = engine.what_if_analysis(frequency=what_if_freq, hours=what_if_hours)
            st.session_state["what_if_results"] = preds
            st.rerun()

    # ── Fetch live data ───────────────────────────────────────────────────────
    data          = engine.get_dashboard_data()
    status        = data["status"]
    failure_prob  = data["failure_prediction"]["probability"]
    health        = data["current_state"]["equipment_health"]
    runtime       = data["kpis"]["total_runtime_hours"]
    is_anomaly    = data["anomaly_detection"]["is_anomaly"]
    anomaly_score = data["anomaly_detection"]["score"]
    alerts        = data["recent_alerts"]
    rec           = data["recommendation"]

    # ── Sound triggers ────────────────────────────────────────────────────────
    sound_key = f"{status}_{is_anomaly}_{failure_prob > 0.7}"
    if sound_key != st.session_state.get("sound_played"):
        if status == "critical" or failure_prob > 0.7:
            inject_sound("critical")
            st.session_state["sound_played"] = sound_key
        elif status == "warning" or is_anomaly:
            inject_sound("warning")
            st.session_state["sound_played"] = sound_key

    # ── Alert banners (always visible, top of page) ───────────────────────────
    critical_alerts = [a for a in alerts if a["level"] == "critical"]
    warning_alerts  = [a for a in alerts if a["level"] == "warning"]

    if critical_alerts:
        for a in critical_alerts[-2:]:
            st.markdown(
                f'<div class="alert-critical">🔴 CRITICAL — {a["message"]} &nbsp;|&nbsp; {a["timestamp"]}</div>',
                unsafe_allow_html=True,
            )
    if warning_alerts:
        for a in warning_alerts[-2:]:
            st.markdown(
                f'<div class="alert-warning">⚠️ WARNING — {a["message"]} &nbsp;|&nbsp; {a["timestamp"]}</div>',
                unsafe_allow_html=True,
            )

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="dash-header">
      <div>
        <div class="dash-title">🛢️ ESP AI Digital Twin</div>
        <div class="dash-subtitle">Real-time monitoring · Predictive maintenance · DRL optimization</div>
      </div>
      <div>{status_badge(status)}</div>
    </div>
    """, unsafe_allow_html=True)

    # ── KPI row ───────────────────────────────────────────────────────────────
    def risk_color(p): return "red" if p > 0.7 else "orange" if p > 0.3 else "green"
    def health_color(h): return "green" if h > 0.7 else "orange" if h > 0.4 else "red"
    def anomaly_color(a): return "red" if a else "green"

    st.markdown(f"""
    <div class="kpi-row">
      <div class="kpi-card">
        <div class="kpi-label">Failure Risk (48h)</div>
        <div class="kpi-value {risk_color(failure_prob)}">{failure_prob:.1%}</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">Equipment Health</div>
        <div class="kpi-value {health_color(health)}">{health:.1%}</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">Anomaly Status</div>
        <div class="kpi-value {anomaly_color(is_anomaly)}">{"⚠ ANOMALY" if is_anomaly else "✓ Normal"}</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">Anomaly Score</div>
        <div class="kpi-value {anomaly_color(is_anomaly)}">{anomaly_score:.3f}</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">Runtime</div>
        <div class="kpi-value green">{runtime:.0f} h</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">Flow Rate</div>
        <div class="kpi-value">{data["current_state"]["flow_rate"]:.0f} bpd</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab_ov, tab_ai, tab_hist, tab_log = st.tabs([
        "📊  Sensors & Gauges",
        "🤖  AI Insights",
        "📈  Trend History",
        "🚨  Alert Log & What-If",
    ])

    # ── Tab 1: Sensors & Gauges ───────────────────────────────────────────────
    with tab_ov:
        g1, g2, g3, g4 = st.columns(4)
        with g1:
            st.plotly_chart(
                gauge(data["current_state"]["motor_temperature"],
                      "Motor Temperature", 160, 110, 140, unit="°C"),
                use_container_width=True, config={"displayModeBar": False},
            )
        with g2:
            vx = data["current_state"]["vibration_x"]
            vy = data["current_state"]["vibration_y"]
            vz = data["current_state"]["vibration_z"]
            vib = np.sqrt(vx**2 + vy**2 + vz**2)
            st.plotly_chart(
                gauge(vib, "Vibration Magnitude", 18, 6, 10, unit=" mm/s"),
                use_container_width=True, config={"displayModeBar": False},
            )
        with g3:
            st.plotly_chart(
                gauge(data["current_state"]["motor_current"],
                      "Motor Current", 150, 80, 105, unit=" A"),
                use_container_width=True, config={"displayModeBar": False},
            )
        with g4:
            st.plotly_chart(
                gauge(data["current_state"]["flow_rate"],
                      "Flow Rate", 5000, 1500, 800, invert=True, unit=" bpd"),
                use_container_width=True, config={"displayModeBar": False},
            )

        # Second gauge row
        g5, g6, g7, g8 = st.columns(4)
        with g5:
            st.plotly_chart(
                gauge(data["current_state"]["discharge_pressure"],
                      "Discharge Pressure", 5500, 3500, 4500, unit=" psi"),
                use_container_width=True, config={"displayModeBar": False},
            )
        with g6:
            st.plotly_chart(
                gauge(data["current_state"]["voltage"],
                      "Supply Voltage", 520, 460, 430, invert=True, unit=" V"),
                use_container_width=True, config={"displayModeBar": False},
            )
        with g7:
            st.plotly_chart(
                gauge(data["current_state"]["power_consumption"],
                      "Power Consumption", 120, 70, 90, unit=" kW"),
                use_container_width=True, config={"displayModeBar": False},
            )
        with g8:
            st.plotly_chart(
                gauge(data["current_state"]["sand_rate"] * 100,
                      "Sand Rate", 5, 0.5, 1.5, unit="%"),
                use_container_width=True, config={"displayModeBar": False},
            )

        # DRL Recommendation card
        action_str = rec["action"].replace("_", " ").title()
        conf_pct   = f"{rec['confidence']:.0%}"
        st.markdown(f"""
        <div class="rec-card">
          <div class="rec-title">💡 DRL Recommended Action</div>
          <div class="rec-action">{action_str}</div>
          <div class="rec-detail">
            Current: <b>{rec['current']:.1f}</b> &nbsp;→&nbsp;
            Target: <b>{rec['recommended']:.1f}</b> &nbsp;|&nbsp;
            Confidence: <b>{conf_pct}</b>
          </div>
          <div class="rec-reason">{rec['reasoning']}</div>
        </div>
        """, unsafe_allow_html=True)

    # ── Tab 2: AI Insights ────────────────────────────────────────────────────
    with tab_ai:
        ai_l, ai_r = st.columns(2)

        with ai_l:
            st.markdown('<div class="section-label">LSTM Failure Predictor</div>',
                        unsafe_allow_html=True)
            st.plotly_chart(risk_gauge(failure_prob * 100),
                            use_container_width=True, config={"displayModeBar": False})
            contribs = data["failure_prediction"]["contributions"]
            if contribs:
                st.plotly_chart(contribution_chart(contribs),
                                use_container_width=True, config={"displayModeBar": False})

        with ai_r:
            st.markdown('<div class="section-label">Autoencoder Anomaly Detector</div>',
                        unsafe_allow_html=True)
            if is_anomaly:
                st.markdown(
                    f'<div class="alert-critical" style="animation:none">⚠️ ANOMALY DETECTED &nbsp;|&nbsp; '
                    f'Reconstruction Error: <b>{anomaly_score:.4f}</b></div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="no-alerts">✓ Normal Operation &nbsp;|&nbsp; '
                    f'Score: {anomaly_score:.4f}</div>',
                    unsafe_allow_html=True,
                )
            anom_contrib = data["anomaly_detection"]["contributions"]
            if anom_contrib:
                st.plotly_chart(radar_chart(anom_contrib),
                                use_container_width=True, config={"displayModeBar": False})

    # ── Tab 3: Trend History ──────────────────────────────────────────────────
    with tab_hist:
        if engine.state_history:
            sensor_opts = [
                "motor_temperature", "vibration_x", "motor_current",
                "flow_rate", "discharge_pressure", "power_consumption",
                "voltage", "sand_rate",
            ]
            selected = st.multiselect(
                "Select sensors to plot",
                options=sensor_opts,
                default=["motor_temperature", "vibration_x", "flow_rate"],
                format_func=lambda s: s.replace("_", " ").title(),
            )
            if selected:
                st.plotly_chart(
                    time_series_chart(engine.state_history, selected),
                    use_container_width=True, config={"displayModeBar": False},
                )
        else:
            st.info("Step the simulation to populate sensor history.")

    # ── Tab 4: Alert Log & What-If ────────────────────────────────────────────
    with tab_log:
        log_col, wi_col = st.columns([1, 1])

        with log_col:
            st.markdown('<div class="section-label">🚨 Alert Log</div>',
                        unsafe_allow_html=True)
            if alerts:
                for a in reversed(alerts[-10:]):
                    lvl = a["level"]
                    msg = a["message"]
                    ts  = a["timestamp"]
                    if lvl == "critical":
                        st.markdown(
                            f'<div class="alert-critical" style="animation:none">🔴 {msg}<br>'
                            f'<small style="opacity:.7">{ts}</small></div>',
                            unsafe_allow_html=True,
                        )
                    elif lvl == "warning":
                        st.markdown(
                            f'<div class="alert-warning">⚠️ {msg}<br>'
                            f'<small style="opacity:.7">{ts}</small></div>',
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            f'<div class="alert-info">ℹ️ {msg}<br>'
                            f'<small style="opacity:.7">{ts}</small></div>',
                            unsafe_allow_html=True,
                        )
            else:
                st.markdown('<div class="no-alerts">✓ No alerts — system nominal</div>',
                            unsafe_allow_html=True)

        with wi_col:
            st.markdown('<div class="section-label">🔮 What-If Scenario Results</div>',
                        unsafe_allow_html=True)
            if "what_if_results" in st.session_state and st.session_state["what_if_results"]:
                df = pd.DataFrame(st.session_state["what_if_results"])
                m1, m2, m3 = st.columns(3)
                with m1:
                    st.metric("Final Health",  f"{df['equipment_health'].iloc[-1]:.1%}")
                with m2:
                    st.metric("Avg Temp",      f"{df['motor_temperature'].mean():.1f} °C")
                with m3:
                    st.metric("Total Prod.",   f"{df['flow_rate'].sum() / 24:.0f} bbl")

                fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                    vertical_spacing=0.06)
                fig.add_trace(go.Scatter(y=df["equipment_health"], mode="lines",
                                         name="Health", line=dict(color="#4ade80")), row=1, col=1)
                fig.add_trace(go.Scatter(y=df["motor_temperature"], mode="lines",
                                         name="Temp °C", line=dict(color="#f97316")), row=2, col=1)
                fig.update_layout(
                    height=260, paper_bgcolor="#0d1b2a", plot_bgcolor="#0d1b2a",
                    font={"color": "#cbd5e1"}, margin=dict(l=10, r=10, t=20, b=10),
                    legend=dict(orientation="h", font={"size": 9}, bgcolor="rgba(0,0,0,0)"),
                )
                for row in [1, 2]:
                    fig.update_yaxes(color="#8ecae6", gridcolor="#1e3a5f", row=row, col=1)
                fig.update_xaxes(color="#8ecae6", gridcolor="#1e3a5f",
                                 title_text="Hours", row=2, col=1)
                st.plotly_chart(fig, use_container_width=True,
                                config={"displayModeBar": False})
            else:
                st.info("Configure a What-If scenario in the sidebar and click **Run Simulation**.")

    # ── Auto-run ──────────────────────────────────────────────────────────────
    # Use streamlit-autorefresh when available (clean pattern, no sleep+rerun race).
    # Fallback: auto-run disabled with a one-time install hint.
    if auto_run:
        if HAS_AUTOREFRESH:
            # Advance one step on each scheduled refresh
            tick = st_autorefresh(interval=1500, key="esp_autorun_tick")
            last_tick = st.session_state.get("last_autorun_tick", -1)
            if tick != last_tick:
                engine.simulator.step()
                engine.sync_state(engine.simulator.state.to_dict())
                st.session_state["last_autorun_tick"] = tick
        else:
            st.warning(
                "Auto-run requires `streamlit-autorefresh`. "
                "Install it with: `pip install streamlit-autorefresh`"
            )


if __name__ == "__main__":
    main()
