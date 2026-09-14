"""
SamudraVani — Multi-agent Ocean Intelligence Dashboard (hackathon demo)
Run with: streamlit run samudravani_app.py

Mirrors the architecture: Data sources -> 5 specialist agents -> Conversational
& Visualisation orchestrator -> Output/interface -> Notifications.

Mock data throughout — swap the functions marked "# REAL:" for your actual
agent calls / DB / API integrations. The UI and orchestrator logic don't
need to change when you do.
"""

import time
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="SamudraVani — Ocean Intelligence", page_icon="🌊", layout="wide")

# ----------------------------------------------------------------------
# Mock data sources (# REAL: replace with satellite/weather/marine APIs)
# ----------------------------------------------------------------------
NODES = [
    {"name": "Veraval", "lat": 20.9159, "lon": 70.3629, "state": "Gujarat"},
    {"name": "Porbandar", "lat": 21.6417, "lon": 69.6293, "state": "Gujarat"},
    {"name": "Kochi", "lat": 9.9312, "lon": 76.2673, "state": "Kerala"},
    {"name": "Vizag", "lat": 17.6868, "lon": 83.2185, "state": "Andhra Pradesh"},
    {"name": "Chennai", "lat": 13.0827, "lon": 80.2707, "state": "Tamil Nadu"},
    {"name": "Paradip", "lat": 20.3167, "lon": 86.6167, "state": "Odisha"},
]
HAZARDS = ["None", "High Waves", "Cyclone Watch", "Strong Wind", "Rip Current"]


@st.cache_data(ttl=300)
def fetch_raw_ocean_data(seed: int = 7) -> pd.DataFrame:
    """# REAL: pull from satellite (SST/chlorophyll), weather, marine-parameter APIs."""
    rng = np.random.default_rng(seed)
    rows = []
    for n in NODES:
        rows.append({
            **n,
            "sst_c": round(rng.uniform(26, 31), 1),
            "wave_height_m": round(rng.uniform(0.4, 3.2), 1),
            "wind_kmph": round(rng.uniform(5, 38), 1),
            "swell_m": round(rng.uniform(0.2, 2.0), 1),
            "current_kmph": round(rng.uniform(0.5, 4.0), 1),
            "precip_mm": round(rng.uniform(0, 40), 1),
            "chlorophyll_mg_m3": round(rng.uniform(0.1, 2.5), 2),
            "hazard": rng.choice(HAZARDS, p=[0.55, 0.2, 0.05, 0.15, 0.05]),
        })
    return pd.DataFrame(rows)


@st.cache_data(ttl=300)
def fetch_timeseries(node_name: str, days: int = 14, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed + hash(node_name) % 1000)
    dates = pd.date_range(end=pd.Timestamp.today(), periods=days)
    sst = rng.uniform(27, 29) + np.cumsum(rng.normal(0, 0.15, days))
    chl = np.clip(rng.uniform(0.5, 1.5) + np.cumsum(rng.normal(0, 0.05, days)), 0.05, None)
    wave = np.clip(rng.normal(1.5, 0.6, days), 0.2, None)
    return pd.DataFrame({"date": dates, "sst_c": sst, "chlorophyll_mg_m3": chl, "wave_height_m": wave})


# ----------------------------------------------------------------------
# The 5 specialist agents — each takes the raw row and returns a structured
# verdict. # REAL: replace each body with your actual model/tool call.
# ----------------------------------------------------------------------
def agent_marine_weather(row) -> dict:
    return {
        "agent": "Marine & Weather",
        "summary": f"SST {row.sst_c}°C · wind {row.wind_kmph} km/h · swell {row.swell_m} m · current {row.current_kmph} km/h",
        "detail": "Wind, waves, swell, currents, SST pulled from live weather + satellite feeds.",
    }


def agent_pfz(row) -> dict:
    rng = np.random.default_rng(int(row.chlorophyll_mg_m3 * 1000))
    score = round(min(1.0, row.chlorophyll_mg_m3 / 2.5 + rng.uniform(-0.1, 0.1)), 2)
    verdict = "strong" if score > 0.6 else "moderate" if score > 0.35 else "weak"
    return {
        "agent": "PFZ & Ocean Analytics",
        "summary": f"Fishing zone likelihood: {score:.2f}/1.0 ({verdict})",
        "detail": f"Derived from chlorophyll ({row.chlorophyll_mg_m3} mg/m³) and SST gradient patterns.",
        "score": score,
    }


def agent_hazard_risk(row) -> dict:
    risky = row.hazard != "None" or row.wave_height_m > 2.5 or row.wind_kmph > 30
    return {
        "agent": "Hazard & Risk",
        "summary": f"Status: {'⚠️ Caution' if risky else '✅ Normal'} · advisory: {row.hazard}",
        "detail": "Checks cyclone tracks, storm warnings, and threshold breaches on wave/wind.",
        "risky": risky,
    }


def agent_gis_route(row) -> dict:
    return {
        "agent": "GIS & Route",
        "summary": "Suggested route stays within 12 nm safe boundary" if not (row.wave_height_m > 2.5) else "Recommend shortened route, boundary tightened to 6 nm",
        "detail": "Cross-checks maritime boundary lines and known safe-passage corridors.",
    }


def agent_historical(row) -> dict:
    return {
        "agent": "Historical & Pattern Analysis",
        "summary": f"Conditions {'align with' if row.chlorophyll_mg_m3 > 1.0 else 'diverge from'} last season's productive window",
        "detail": "Compares current readings against multi-year catch and condition logs for this node.",
    }


AGENTS = [agent_marine_weather, agent_pfz, agent_hazard_risk, agent_gis_route, agent_historical]


def route_query(query: str) -> list:
    """Very simple keyword router simulating orchestrator agent-selection.
    # REAL: replace with your actual LLM-based router/orchestrator."""
    q = query.lower()
    picks = []
    if any(w in q for w in ["safe", "venture", "wind", "wave", "weather", "temperature", "sst"]):
        picks.append(agent_marine_weather)
    if any(w in q for w in ["fish", "pfz", "zone", "catch"]):
        picks.append(agent_pfz)
    if any(w in q for w in ["safe", "venture", "hazard", "risk", "cyclone", "storm", "warning"]):
        picks.append(agent_hazard_risk)
    if any(w in q for w in ["route", "boundary", "path", "navigate", "gis"]):
        picks.append(agent_gis_route)
    if any(w in q for w in ["history", "season", "trend", "pattern", "usual"]):
        picks.append(agent_historical)
    return picks or [agent_marine_weather, agent_hazard_risk]  # default fallback


def orchestrate(query: str, row) -> tuple:
    invoked = route_query(query)
    outputs = [a(row) for a in invoked]
    lines = [o["summary"] for o in outputs]
    reply = f"Here's what I found for **{row['name']}**:\n\n" + "\n".join(f"- {l}" for l in lines)
    reply += f"\n\n_Sources: {', '.join(o['agent'] for o in outputs)}_"
    return reply, outputs


# ----------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------
st.sidebar.title("🌊 SamudraVani")
st.sidebar.caption("Multi-agent ocean intelligence system")
df_nodes = fetch_raw_ocean_data()
selected_node = st.sidebar.selectbox("Location", df_nodes["name"].tolist())
lang = st.sidebar.selectbox("Language", ["English", "Hindi", "Gujarati", "Tamil", "Telugu"])
st.sidebar.divider()
st.sidebar.markdown("**Active specialists**")
for a in AGENTS:
    st.sidebar.markdown(f"🟢 {a.__name__.replace('agent_', '').replace('_', ' ').title()}")
st.sidebar.divider()
st.sidebar.markdown("**Delivery channels**")
st.sidebar.markdown("💬 Chat · 📱 App push · ✉️ SMS/Email")

row = df_nodes[df_nodes["name"] == selected_node].iloc[0]

# ----------------------------------------------------------------------
# Header + KPIs
# ----------------------------------------------------------------------
st.title("SamudraVani — Ocean Intelligence Dashboard")
st.caption(f"{selected_node}, {row.state} · live snapshot from the knowledge & infrastructure layer")

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Sea Surface Temp", f"{row.sst_c} °C")
k2.metric("Wave Height", f"{row.wave_height_m} m")
k3.metric("Wind Speed", f"{row.wind_kmph} km/h")
k4.metric("Chlorophyll", f"{row.chlorophyll_mg_m3} mg/m³")
hazard_out = agent_hazard_risk(row)
k5.metric("Risk Status", "⚠️ Caution" if hazard_out["risky"] else "✅ Normal")
if row.hazard != "None":
    st.warning(f"⚠️ Active advisory at {selected_node}: **{row.hazard}**")

st.divider()

# ----------------------------------------------------------------------
# Agent orchestration panel — the visual hook
# ----------------------------------------------------------------------
st.subheader("🧠 Specialist agents — live outputs for this node")
cols = st.columns(5)
for col, agent_fn in zip(cols, AGENTS):
    out = agent_fn(row)
    with col:
        st.markdown(f"**{out['agent']}**")
        st.caption(out["summary"])
        with st.expander("details"):
            st.write(out["detail"])

st.divider()

# ----------------------------------------------------------------------
# Map + trends
# ----------------------------------------------------------------------
col_map, col_trend = st.columns([1.1, 1])

with col_map:
    st.subheader("Coastal nodes")
    df_nodes["pfz_score"] = df_nodes.apply(lambda r: agent_pfz(r)["score"], axis=1)
    df_nodes["risk_status"] = df_nodes.apply(
        lambda r: "Caution" if agent_hazard_risk(r)["risky"] else "Normal", axis=1
    )
    fig_map = px.scatter_mapbox(
        df_nodes, lat="lat", lon="lon", color="risk_status",
        color_discrete_map={"Normal": "#2E8B57", "Caution": "#D85A30"},
        size="pfz_score", size_max=22, hover_name="name",
        hover_data={"lat": False, "lon": False, "hazard": True, "sst_c": True},
        zoom=3.6, center={"lat": 15.5, "lon": 78}, height=420,
    )
    fig_map.update_layout(mapbox_style="carto-positron", margin=dict(l=0, r=0, t=0, b=0))
    st.plotly_chart(fig_map, use_container_width=True)

with col_trend:
    st.subheader(f"14-day trend · {selected_node}")
    ts = fetch_timeseries(selected_node)
    metric_choice = st.radio("Metric", ["SST (°C)", "Chlorophyll (mg/m³)", "Wave height (m)"], horizontal=True)
    col_map_metric = {"SST (°C)": "sst_c", "Chlorophyll (mg/m³)": "chlorophyll_mg_m3", "Wave height (m)": "wave_height_m"}[metric_choice]
    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(x=ts["date"], y=ts[col_map_metric], mode="lines+markers", line=dict(width=3)))
    fig_trend.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig_trend, use_container_width=True)

st.divider()

# ----------------------------------------------------------------------
# Predictions strip
# ----------------------------------------------------------------------
st.subheader("📅 Predicted best windows (next 5 days)")
pred_days = pd.date_range(start=pd.Timestamp.today() + pd.Timedelta(days=1), periods=5)
rng = np.random.default_rng(hash(selected_node) % 1000)
pred_scores = np.clip(rng.normal(0.6, 0.2, 5), 0, 1)
pred_cols = st.columns(5)
for c, d, s in zip(pred_cols, pred_days, pred_scores):
    label = "Good" if s > 0.6 else "Fair" if s > 0.35 else "Poor"
    c.metric(d.strftime("%a %d %b"), label, f"{s:.2f}")

st.divider()

# ----------------------------------------------------------------------
# Conversational orchestrator
# ----------------------------------------------------------------------
st.subheader("💬 Ask SamudraVani")
st.caption("Query routes to the relevant specialist agent(s) — swap `route_query`/agents for real models.")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for role, msg in st.session_state.chat_history:
    with st.chat_message(role):
        st.markdown(msg)

user_q = st.chat_input(f"Ask about {selected_node} (e.g. 'Is it safe to venture out today?')")
if user_q:
    st.session_state.chat_history.append(("user", user_q))
    with st.spinner("Orchestrator routing to specialist agents..."):
        time.sleep(0.4)
        reply, invoked = orchestrate(user_q, row)
    st.session_state.chat_history.append(("assistant", reply))
    st.rerun()
