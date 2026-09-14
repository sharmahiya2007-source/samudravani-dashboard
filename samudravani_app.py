"""
SamudraVani — Ocean Intelligence Chatbot (hackathon demo)
Run with: streamlit run samudravani_chatbot.py

Same 5-agent mock architecture as the dashboard version, but the UI is a
pure chat interface — no KPI cards, no map, no charts. Good for a focused
conversational demo.

Mock data throughout — swap the functions marked "# REAL:" for your actual
agent calls / DB / API integrations. The chat UI doesn't need to change
when you do.
"""

import time
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="SamudraVani — Ask the Ocean", page_icon="🌊", layout="centered")

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

EXAMPLE_QUESTIONS = [
    "Is it safe to venture out today?",
    "Where's the best fishing zone right now?",
    "Any cyclone or storm warnings?",
    "What's the safest route to take?",
    "How do conditions compare to last season?",
]


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


# ----------------------------------------------------------------------
# The 5 specialist agents — each takes the raw row and returns a structured
# verdict. # REAL: replace each body with your actual model/tool call.
# ----------------------------------------------------------------------
def agent_marine_weather(row) -> dict:
    return {
        "agent": "Marine & Weather",
        "icon": "🌤️",
        "summary": f"SST {row.sst_c}°C · wind {row.wind_kmph} km/h · swell {row.swell_m} m · current {row.current_kmph} km/h",
        "detail": "Wind, waves, swell, currents, SST pulled from live weather + satellite feeds.",
    }


def agent_pfz(row) -> dict:
    rng = np.random.default_rng(int(row.chlorophyll_mg_m3 * 1000))
    score = round(min(1.0, row.chlorophyll_mg_m3 / 2.5 + rng.uniform(-0.1, 0.1)), 2)
    verdict = "strong" if score > 0.6 else "moderate" if score > 0.35 else "weak"
    return {
        "agent": "PFZ & Ocean Analytics",
        "icon": "🐟",
        "summary": f"Fishing zone likelihood: {score:.2f}/1.0 ({verdict})",
        "detail": f"Derived from chlorophyll ({row.chlorophyll_mg_m3} mg/m³) and SST gradient patterns.",
        "score": score,
    }


def agent_hazard_risk(row) -> dict:
    risky = row.hazard != "None" or row.wave_height_m > 2.5 or row.wind_kmph > 30
    return {
        "agent": "Hazard & Risk",
        "icon": "⚠️",
        "summary": f"Status: {'⚠️ Caution' if risky else '✅ Normal'} · advisory: {row.hazard}",
        "detail": "Checks cyclone tracks, storm warnings, and threshold breaches on wave/wind.",
        "risky": risky,
    }


def agent_gis_route(row) -> dict:
    return {
        "agent": "GIS & Route",
        "icon": "🗺️",
        "summary": "Suggested route stays within 12 nm safe boundary" if not (row.wave_height_m > 2.5) else "Recommend shortened route, boundary tightened to 6 nm",
        "detail": "Cross-checks maritime boundary lines and known safe-passage corridors.",
    }


def agent_historical(row) -> dict:
    return {
        "agent": "Historical & Pattern Analysis",
        "icon": "📊",
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
    lines = [f"{o['icon']} **{o['agent']}** — {o['summary']}" for o in outputs]
    reply = f"Here's what I found for **{row['name']}**:\n\n" + "\n\n".join(lines)
    return reply, outputs


# ----------------------------------------------------------------------
# Sidebar — minimal, just context controls
# ----------------------------------------------------------------------
st.sidebar.title("🌊 SamudraVani")
st.sidebar.caption("Multi-agent ocean intelligence chatbot")
df_nodes = fetch_raw_ocean_data()
selected_node = st.sidebar.selectbox("Location", df_nodes["name"].tolist())
lang = st.sidebar.selectbox("Language", ["English", "Hindi", "Gujarati", "Tamil", "Telugu"])
st.sidebar.divider()
st.sidebar.markdown("**Active specialists**")
for a in AGENTS:
    dummy_row = df_nodes.iloc[0]
    out = a(dummy_row)
    st.sidebar.markdown(f"{out['icon']} {out['agent']}")
st.sidebar.divider()
if st.sidebar.button("🗑️ Clear chat"):
    st.session_state.chat_history = []
    st.rerun()

row = df_nodes[df_nodes["name"] == selected_node].iloc[0]

# ----------------------------------------------------------------------
# Main chat interface
# ----------------------------------------------------------------------
st.title("🌊 SamudraVani")
st.caption(f"Ask anything about ocean conditions near **{selected_node}, {row.state}**")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Welcome message + example question chips (only shown before first message)
if not st.session_state.chat_history:
    with st.chat_message("assistant"):
        st.markdown(
            f"Namaste! 👋 I'm **SamudraVani**, your ocean intelligence assistant for "
            f"**{selected_node}**. Ask me about safety, fishing zones, weather, routes, "
            f"or seasonal patterns — I'll route your question to the right specialist agent."
        )
    st.markdown("**Try asking:**")
    chip_cols = st.columns(len(EXAMPLE_QUESTIONS))
    clicked_example = None
    for c, q in zip(chip_cols, EXAMPLE_QUESTIONS):
        if c.button(q, use_container_width=True):
            clicked_example = q
else:
    clicked_example = None

# Render chat history
for entry in st.session_state.chat_history:
    role, msg = entry[0], entry[1]
    with st.chat_message(role):
        st.markdown(msg)
        if role == "assistant" and len(entry) > 2:
            outputs = entry[2]
            with st.expander("🔍 Agent details & sources"):
                for o in outputs:
                    st.markdown(f"**{o['icon']} {o['agent']}**")
                    st.caption(o["detail"])

# Chat input (typed or via example chip)
typed_q = st.chat_input(f"Ask about {selected_node} (e.g. 'Is it safe to venture out today?')")
user_q = typed_q or clicked_example

if user_q:
    st.session_state.chat_history.append(("user", user_q))
    with st.spinner("Orchestrator routing to specialist agents..."):
        time.sleep(0.4)
        reply, invoked = orchestrate(user_q, row)
    st.session_state.chat_history.append(("assistant", reply, invoked))
    st.rerun()
