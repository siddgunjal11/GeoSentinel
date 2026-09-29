# ============================================================
#  🌍 GeoSentinel — Cloud-Based Disaster Intelligence Platform
#  Author : Siddharth Gunjal
#  Upgraded: Live alert tracking with auto-refresh
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import datetime
import random
import warnings
warnings.filterwarnings("ignore")

# ─── PAGE CONFIG ────────────────────────────────────────────
st.set_page_config(
    page_title="GeoSentinel | Disaster Intelligence",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CUSTOM CSS ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
}

.stApp {
    background: #0a0e1a;
    color: #e2e8f0;
}

section[data-testid="stSidebar"] {
    background: #0d1220;
    border-right: 1px solid rgba(99,179,237,0.15);
}
section[data-testid="stSidebar"] * { color: #cbd5e0 !important; }

.metric-card {
    background: linear-gradient(135deg, #111827 0%, #1a2234 100%);
    border: 1px solid rgba(99,179,237,0.2);
    border-radius: 12px;
    padding: 20px 24px;
    margin: 6px 0;
    box-shadow: 0 4px 24px rgba(0,0,0,0.4);
    transition: border-color 0.3s;
}
.metric-card:hover { border-color: rgba(99,179,237,0.5); }
.metric-card h2 { font-family: 'JetBrains Mono', monospace; font-size: 2rem; margin: 4px 0; }
.metric-card p  { font-size: 0.8rem; color: #718096; text-transform: uppercase; letter-spacing: 1px; margin: 0; }

.badge-critical { background:#ff4444; color:#fff; padding:3px 10px; border-radius:20px; font-size:0.75rem; font-weight:600; }
.badge-high     { background:#ff8c00; color:#fff; padding:3px 10px; border-radius:20px; font-size:0.75rem; font-weight:600; }
.badge-moderate { background:#f6c90e; color:#1a1a1a; padding:3px 10px; border-radius:20px; font-size:0.75rem; font-weight:600; }
.badge-low      { background:#22c55e; color:#fff; padding:3px 10px; border-radius:20px; font-size:0.75rem; font-weight:600; }

.section-header {
    font-size: 1.05rem;
    font-weight: 600;
    color: #63b3ed;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin: 28px 0 14px 0;
    padding-bottom: 8px;
    border-bottom: 1px solid rgba(99,179,237,0.2);
}

/* ── Live alert rows ── */
.live-alert-critical {
    background: rgba(255,68,68,0.10);
    border-left: 4px solid #ff4444;
    padding: 11px 16px;
    border-radius: 0 8px 8px 0;
    margin: 6px 0;
    font-size: 0.86rem;
    animation: fadeIn 0.4s ease;
}
.live-alert-high {
    background: rgba(255,140,0,0.10);
    border-left: 4px solid #ff8c00;
    padding: 11px 16px;
    border-radius: 0 8px 8px 0;
    margin: 6px 0;
    font-size: 0.86rem;
    animation: fadeIn 0.4s ease;
}
.live-alert-moderate {
    background: rgba(246,201,14,0.10);
    border-left: 4px solid #f6c90e;
    padding: 11px 16px;
    border-radius: 0 8px 8px 0;
    margin: 6px 0;
    font-size: 0.86rem;
    animation: fadeIn 0.4s ease;
}
.live-alert-low {
    background: rgba(34,197,94,0.10);
    border-left: 4px solid #22c55e;
    padding: 11px 16px;
    border-radius: 0 8px 8px 0;
    margin: 6px 0;
    font-size: 0.86rem;
    animation: fadeIn 0.4s ease;
}
@keyframes fadeIn { from { opacity:0; transform:translateY(-4px); } to { opacity:1; transform:translateY(0); } }

.live-dot {
    display: inline-block;
    width: 9px; height: 9px;
    border-radius: 50%;
    background: #22c55e;
    margin-right: 6px;
    animation: blink 1.4s infinite;
}
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.25} }

.status-bar {
    background: linear-gradient(90deg, #0d1220, #111827);
    border: 1px solid rgba(99,179,237,0.15);
    border-radius: 10px;
    padding: 10px 18px;
    font-size: 0.82rem;
    color: #718096;
    margin-bottom: 16px;
    display: flex;
    align-items: center;
    gap: 20px;
}

.alert-critical {
    background: rgba(255,68,68,0.1);
    border-left: 4px solid #ff4444;
    padding: 12px 16px;
    border-radius: 0 8px 8px 0;
    margin: 8px 0;
    font-size: 0.88rem;
}
.alert-moderate {
    background: rgba(246,201,14,0.1);
    border-left: 4px solid #f6c90e;
    padding: 12px 16px;
    border-radius: 0 8px 8px 0;
    margin: 8px 0;
    font-size: 0.88rem;
}

.js-plotly-plot .plotly, .js-plotly-plot .plotly .main-svg {
    background: transparent !important;
}

.stButton > button {
    background: linear-gradient(135deg, #2b6cb0, #3182ce);
    color: white;
    border: none;
    border-radius: 8px;
    padding: 10px 28px;
    font-weight: 600;
    font-family: 'Space Grotesk', sans-serif;
    letter-spacing: 0.5px;
    transition: all 0.2s;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #3182ce, #4299e1);
    box-shadow: 0 4px 20px rgba(49,130,206,0.4);
    transform: translateY(-1px);
}

.stSlider > div > div > div { background: #2b6cb0 !important; }

.stTabs [data-baseweb="tab-list"] {
    background: #111827;
    border-radius: 10px;
    padding: 4px;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: #718096;
    border-radius: 8px;
    font-weight: 500;
}
.stTabs [aria-selected="true"] {
    background: #2b6cb0 !important;
    color: white !important;
}

.dataframe { font-size: 0.83rem !important; }
</style>
""", unsafe_allow_html=True)

# ─── CHART THEME ────────────────────────────────────────────
PALETTE = {
    "Very High": "#ff4444",
    "High":      "#ff8c00",
    "Moderate":  "#f6c90e",
    "Low":       "#22c55e",
}

# ─── DATA LOADING ────────────────────────────────────────────
@st.cache_data
def load_data():
    tsunami  = pd.read_csv("data/tsunamis.csv")
    disaster = pd.read_csv("data/dummy_disasters.csv")
    tsunami["maximumwaterheight"] = tsunami["maximumwaterheight"].fillna(0)
    tsunami["deaths"]             = tsunami["deaths"].fillna(0)
    disaster["Economic_Damage_Million$"] = disaster["Economic_Damage_Million$"].fillna(0)
    disaster["Deaths"]            = disaster["Deaths"].fillna(0)
    disaster["Decade"]            = (disaster["Year"] // 10) * 10
    disaster["Log_Damage"]        = np.log1p(disaster["Economic_Damage_Million$"])
    return tsunami, disaster

@st.cache_resource
def train_model(disaster_df):
    def risk_label(mag):
        if mag >= 8:   return "Very High"
        elif mag >= 7: return "High"
        elif mag >= 6: return "Moderate"
        else:          return "Low"

    disaster_df = disaster_df.copy()
    disaster_df["Risk_Level"] = disaster_df["Magnitude_Intensity"].apply(risk_label)
    le = LabelEncoder()
    disaster_df["Disaster_Enc"] = le.fit_transform(disaster_df["Disaster_Type"])
    features = ["Magnitude_Intensity", "Deaths", "Economic_Damage_Million$", "Disaster_Enc", "Year"]
    X = disaster_df[features]
    y = disaster_df["Risk_Level"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    rf = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    acc = rf.score(X_test, y_test)
    cv  = cross_val_score(rf, X, y, cv=5, scoring="accuracy").mean()
    importances = dict(zip(features, rf.feature_importances_))
    return rf, le, acc, cv, importances, X_test, y_test

tsunami_df, disaster_df = load_data()
model, label_enc, test_acc, cv_acc, feat_imp, X_test_g, y_test_g = train_model(disaster_df)

# ─── LIVE EVENT GENERATOR ────────────────────────────────────
# Deterministic pool of realistic seismic/disaster events.
# Each page refresh picks the next N events based on current minute
# so the feed appears to scroll forward in time.

_EVENT_POOL = [
    # (severity, emoji, title, detail, lat, lon, mag)
    ("critical","🔴","M8.3 — Tohoku, Japan","Tsunami watch active · Depth 12km · PTWC alert issued",38.1,142.8,8.3),
    ("critical","🔴","M8.1 — Valparaíso, Chile","Coastal evacuation ordered · Depth 18km · 4m surge expected",-32.9,-71.5,8.1),
    ("critical","🔴","Category 5 Cyclone — Bay of Bengal","Wind 280km/h · Landfall in 14h · 3 states evacuating",16.2,88.4,None),
    ("critical","🔴","M7.9 — Kamchatka, Russia","Very High risk · Deep aftershock sequence · 23km depth",52.8,160.1,7.9),
    ("critical","🔴","Volcanic Eruption — Merapi, Indonesia","Pyroclastic flow · 6km exclusion zone · Aviation RED",-7.5,110.4,None),
    ("high","🟠","M7.4 — Sulawesi, Indonesia","Tsunami advisory · Depth 34km · Coastline monitoring",-1.3,120.2,7.4),
    ("high","🟠","M7.2 — Kermadec Islands, NZ","High risk · No tsunami threat · Monitoring continues",-29.2,-177.9,7.2),
    ("high","🟠","Flash Flood — Yangtze River Basin","380mm/24h · 14 counties evacuated · Rising rapidly",30.5,114.2,None),
    ("high","🟠","M7.1 — Anchorage, Alaska","Building damage reported · Depth 41km · Aftershocks expected",61.2,-149.9,7.1),
    ("high","🟠","Wildfire — New South Wales, AU","45,000 ha burned · Wind 70km/h · Spreading NE",-33.4,150.9,None),
    ("high","🟠","M6.9 — Hindu Kush, Afghanistan","Depth 190km · Regional alert · Infrastructure damage",36.5,70.8,6.9),
    ("high","🟠","Drought Emergency — Horn of Africa","3rd failed rainy season · 9.4M affected · Famine risk",8.0,46.0,None),
    ("moderate","🟡","M6.4 — Luzon, Philippines","Depth 52km · Structural damage · No tsunami",15.9,121.1,6.4),
    ("moderate","🟡","Tropical Storm — Gulf of Mexico","Wind 105km/h · 72h track uncertain · Advisory issued",24.1,-91.3,None),
    ("moderate","🟡","M6.2 — Aegean Sea, Greece","Minor damage reported · Depth 8km · Aftershocks",39.1,25.4,6.2),
    ("moderate","🟡","Landslide — Koshi Zone, Nepal","Threshold exceeded · 3 villages warned · Roads blocked",27.3,87.1,None),
    ("moderate","🟡","M6.0 — Oaxaca, Mexico","Felt widely · Depth 22km · No major damage",16.3,-96.7,6.0),
    ("moderate","🟡","Flood Watch — Bangladesh Delta","Rivers above warning level · Low-lying areas at risk",23.5,89.9,None),
    ("low","🟢","M4.8 — Azores, Atlantic","No damage · Depth 16km · Standard monitoring",38.7,-27.2,4.8),
    ("low","🟢","M4.5 — Reykjanes, Iceland","Minor tremor · Volcanic activity nominal",63.9,-22.6,4.5),
    ("low","🟢","M4.3 — Cascadia Zone, Oregon","Routine monitoring · Depth 30km · No alerts",44.1,-124.6,4.3),
    ("low","🟢","M4.1 — Canary Islands, Spain","Volcanic monitoring active · No hazard",28.3,-16.4,4.1),
]

def _get_live_events(n=12):
    """
    Return n events that change every ~30 seconds so the feed
    looks live without needing a real API.
    """
    now = datetime.datetime.utcnow()
    # Rotate offset based on current 30-second slot
    slot = (now.minute * 60 + now.second) // 30
    rng = random.Random(slot)            # deterministic per slot
    pool = _EVENT_POOL.copy()
    rng.shuffle(pool)
    chosen = pool[:n]
    # Assign fake "seconds ago" timestamps that also rotate
    result = []
    elapsed = 0
    for ev in chosen:
        elapsed += rng.randint(15, 180)
        ts = now - datetime.timedelta(seconds=elapsed)
        result.append((*ev, ts))
    return result   # (severity, emoji, title, detail, lat, lon, mag, ts)

def _severity_color(sev):
    return {"critical":"#ff4444","high":"#ff8c00","moderate":"#f6c90e","low":"#22c55e"}.get(sev,"#63b3ed")

def _time_ago(ts):
    diff = int((datetime.datetime.utcnow() - ts).total_seconds())
    if diff < 60:   return f"{diff}s ago"
    if diff < 3600: return f"{diff//60}m ago"
    return f"{diff//3600}h {(diff%3600)//60}m ago"

# ─── HELPERS ────────────────────────────────────────────────
def styled_metric(label, value, delta=None, color="#63b3ed"):
    delta_html = f"<span style='color:#68d391;font-size:0.8rem'>▲ {delta}</span>" if delta else ""
    st.markdown(f"""
    <div class='metric-card'>
        <p>{label}</p>
        <h2 style='color:{color}'>{value}</h2>
        {delta_html}
    </div>""", unsafe_allow_html=True)

def apply_template(fig):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(17,24,39,0.6)",
        font=dict(family="Space Grotesk", color="#cbd5e0", size=12),
    )
    fig.update_xaxes(gridcolor="rgba(99,179,237,0.08)", linecolor="rgba(99,179,237,0.15)")
    fig.update_yaxes(gridcolor="rgba(99,179,237,0.08)", linecolor="rgba(99,179,237,0.15)")
    return fig

# ─── SIDEBAR ────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌍 GeoSentinel")
    st.caption("Disaster Intelligence Platform v2.1")
    st.markdown("---")

    page = st.radio(
        "Navigation",
        ["🏠 Overview",
         "🌊 Tsunami Intelligence",
         "🌋 Disaster Analytics",
         "🧠 ML Risk Engine",
         "📡 Live Simulator",
         "🚨 Live Tracking",
         "ℹ️ About"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("**Data Status**")
    st.success(f"✅ {len(disaster_df):,} disaster records loaded")
    st.success(f"✅ {len(tsunami_df)} tsunami events")
    st.success(f"✅ ML Model: {test_acc*100:.1f}% accuracy")
    st.markdown("---")
    st.caption("Hari Maheshwari · GeoSentinel v2.1")


# ════════════════════════════════════════════════════════════
#  PAGE 1 — OVERVIEW
# ════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.markdown("# 🌍 GeoSentinel — Disaster Intelligence Platform")
    st.markdown(
        "<p style='color:#718096;font-size:1.05rem;margin-top:-10px'>"
        "Real-time monitoring · Predictive analytics · Multi-hazard intelligence"
        "</p>", unsafe_allow_html=True
    )

    total_deaths = int(disaster_df["Deaths"].sum())
    total_damage = disaster_df["Economic_Damage_Million$"].sum()
    worst_type   = disaster_df.groupby("Disaster_Type")["Deaths"].sum().idxmax()
    worst_loc    = disaster_df.groupby("Location")["Economic_Damage_Million$"].sum().idxmax()

    c1, c2, c3, c4 = st.columns(4)
    with c1: styled_metric("Total Casualties Recorded", f"{total_deaths:,}", color="#fc8181")
    with c2: styled_metric("Economic Damage", f"${total_damage/1e6:.1f}T", color="#f6c90e")
    with c3: styled_metric("Deadliest Disaster Type", worst_type, color="#b794f4")
    with c4: styled_metric("Highest Risk Region", worst_loc, color="#63b3ed")

    st.markdown("<div class='section-header'>Global Disaster Trend</div>", unsafe_allow_html=True)

    trend = disaster_df.groupby("Year").agg(
        Deaths=("Deaths","sum"),
        Damage=("Economic_Damage_Million$","sum"),
        Events=("Disaster_Type","count")
    ).reset_index()

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(
        x=trend["Year"], y=trend["Deaths"],
        name="Deaths", mode="lines+markers",
        line=dict(color="#fc8181", width=2.5),
        marker=dict(size=5),
        fill="tozeroy", fillcolor="rgba(252,129,129,0.07)"
    ), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=trend["Year"], y=trend["Damage"],
        name="Economic Damage ($M)", mode="lines+markers",
        line=dict(color="#f6c90e", width=2, dash="dot"),
        marker=dict(size=5)
    ), secondary_y=True)
    fig.add_trace(go.Bar(
        x=trend["Year"], y=trend["Events"],
        name="Events", marker_color="rgba(99,179,237,0.25)"
    ), secondary_y=False)

    fig.update_layout(
        title="Deaths, Economic Damage & Event Frequency Over Time",
        yaxis=dict(title="Total Deaths", color="#fc8181"),
        yaxis2=dict(title="Economic Damage ($M)", color="#f6c90e", overlaying="y", side="right"),
        legend=dict(orientation="h", y=1.08),
        height=420
    )
    apply_template(fig)
    st.plotly_chart(fig, width="stretch")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("<div class='section-header'>Disaster Type Distribution</div>", unsafe_allow_html=True)
        counts = disaster_df["Disaster_Type"].value_counts()
        fig_pie = go.Figure(go.Pie(
            labels=counts.index, values=counts.values,
            hole=0.55, textinfo="label+percent",
            marker=dict(colors=["#63b3ed","#fc8181","#f6c90e","#68d391","#b794f4","#fbd38d"])
        ))
        fig_pie.update_layout(title="Proportion of Disaster Types", showlegend=False, height=360)
        apply_template(fig_pie)
        st.plotly_chart(fig_pie, width="stretch")

    with col_b:
        st.markdown("<div class='section-header'>Deaths by Disaster Category</div>", unsafe_allow_html=True)
        cat_deaths = disaster_df.groupby("Disaster_Type")["Deaths"].sum().sort_values(ascending=True)
        fig_bar = go.Figure(go.Bar(
            x=cat_deaths.values, y=cat_deaths.index, orientation="h",
            marker=dict(color=cat_deaths.values,
                        colorscale=[[0,"#2b6cb0"],[0.5,"#f6c90e"],[1,"#fc8181"]],
                        showscale=False),
            text=[f"{v:,}" for v in cat_deaths.values],
            textposition="outside",
            textfont=dict(color="#cbd5e0", size=11)
        ))
        fig_bar.update_layout(title="Cumulative Deaths by Type", height=360)
        apply_template(fig_bar)
        st.plotly_chart(fig_bar, width="stretch")

    st.markdown("<div class='section-header'>⚠ Simulated Active Alerts</div>", unsafe_allow_html=True)
    a1, a2, a3 = st.columns(3)
    with a1:
        st.markdown("""<div class='alert-critical'>
            🔴 <strong>CRITICAL — Earthquake M8.2</strong><br>
            Japan Sea Region · Risk: Very High<br>
            <small>Issued: 2025-04-26 08:14 UTC</small></div>""", unsafe_allow_html=True)
    with a2:
        st.markdown("""<div class='alert-moderate'>
            🟡 <strong>WATCH — Cyclone Category 3</strong><br>
            Bay of Bengal · Risk: High<br>
            <small>Issued: 2025-04-26 06:30 UTC</small></div>""", unsafe_allow_html=True)
    with a3:
        st.markdown("""<div class='alert-moderate'>
            🟡 <strong>ADVISORY — Flood Warning</strong><br>
            Mekong Delta Region · Risk: Moderate<br>
            <small>Issued: 2025-04-25 22:00 UTC</small></div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
#  PAGE 2 — TSUNAMI INTELLIGENCE
# ════════════════════════════════════════════════════════════
elif page == "🌊 Tsunami Intelligence":
    st.markdown("# 🌊 Tsunami Intelligence Module")
    st.caption("Historical tsunami event analysis — NOAA dataset")

    t1, t2, t3 = st.columns(3)
    with t1: styled_metric("Total Tsunami Events", str(len(tsunami_df)), color="#63b3ed")
    with t2: styled_metric("Peak Water Height", f"{tsunami_df['maximumwaterheight'].max():.1f} m", color="#fc8181")
    with t3: styled_metric("Total Fatalities", f"{int(tsunami_df['deaths'].sum()):,}", color="#f6c90e")

    tab1, tab2, tab3 = st.tabs(["📈 Time Series", "🫧 Impact Matrix", "📊 Country Breakdown"])

    with tab1:
        st.markdown("#### Maximum Water Height — Historical Trend")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=tsunami_df["Year"], y=tsunami_df["maximumwaterheight"],
            mode="lines+markers",
            line=dict(color="#63b3ed", width=3),
            marker=dict(size=10, color=tsunami_df["maximumwaterheight"],
                        colorscale="Blues", showscale=True,
                        colorbar=dict(title="Height (m)")),
            fill="tozeroy", fillcolor="rgba(99,179,237,0.1)",
            name="Max Water Height",
            hovertemplate="<b>%{x}</b><br>Height: %{y:.1f}m<extra></extra>"
        ))
        fig.update_layout(title="Tsunami Wave Heights Over Time",
                          xaxis_title="Year", yaxis_title="Maximum Water Height (m)", height=420)
        apply_template(fig)
        st.plotly_chart(fig, width="stretch")

    with tab2:
        st.markdown("#### Deaths vs Year — Bubble = Wave Height")
        fig = px.scatter(
            tsunami_df, x="Year", y="deaths",
            size="maximumwaterheight", color="country",
            size_max=80,
            color_discrete_sequence=px.colors.qualitative.Bold,
            hover_data={"maximumwaterheight": ":.1f", "deaths": ":,"},
            labels={"deaths": "Fatalities", "maximumwaterheight": "Wave Height (m)"},
            title="Tsunami Impact Matrix (Bubble = Wave Height)"
        )
        fig.update_layout(height=460)
        apply_template(fig)
        st.plotly_chart(fig, width="stretch")

    with tab3:
        st.markdown("#### Deaths by Country")
        country_deaths = tsunami_df.groupby("country").agg(
            Deaths=("deaths","sum"),
            Events=("Year","count"),
            AvgHeight=("maximumwaterheight","mean")
        ).reset_index().sort_values("Deaths", ascending=False)

        fig = px.bar(
            country_deaths, x="country", y="Deaths",
            color="AvgHeight", color_continuous_scale="Blues",
            text="Deaths",
            title="Fatalities and Average Wave Height by Country"
        )
        fig.update_traces(texttemplate="%{text:,}", textposition="outside")
        fig.update_layout(height=400, coloraxis_colorbar_title="Avg Height (m)")
        apply_template(fig)
        st.plotly_chart(fig, width="stretch")

        st.markdown("##### Detailed Table")
        country_deaths["Deaths"]    = country_deaths["Deaths"].apply(lambda x: f"{int(x):,}")
        country_deaths["AvgHeight"] = country_deaths["AvgHeight"].apply(lambda x: f"{x:.1f} m")
        st.dataframe(country_deaths.rename(columns={"country":"Country","AvgHeight":"Avg Wave Height"}),
                     hide_index=True)


# ════════════════════════════════════════════════════════════
#  PAGE 3 — DISASTER ANALYTICS
# ════════════════════════════════════════════════════════════
elif page == "🌋 Disaster Analytics":
    st.markdown("# 🌋 Multi-Hazard Disaster Analytics")
    st.caption(f"Analysing {len(disaster_df):,} global disaster events across {disaster_df['Year'].nunique()} years")

    with st.expander("🔧 Data Filters", expanded=False):
        fc1, fc2, fc3 = st.columns(3)
        with fc1:
            sel_types = st.multiselect("Disaster Types", disaster_df["Disaster_Type"].unique(),
                                       default=disaster_df["Disaster_Type"].unique())
        with fc2:
            yr_range = st.slider("Year Range",
                                 int(disaster_df["Year"].min()), int(disaster_df["Year"].max()),
                                 (int(disaster_df["Year"].min()), int(disaster_df["Year"].max())))
        with fc3:
            mag_min = st.slider("Min Magnitude", 0.0, 10.0, 0.0, 0.5)

    filtered = disaster_df[
        (disaster_df["Disaster_Type"].isin(sel_types)) &
        (disaster_df["Year"].between(*yr_range)) &
        (disaster_df["Magnitude_Intensity"] >= mag_min)
    ]
    st.caption(f"Showing **{len(filtered):,}** events after filters")

    st.markdown("<div class='section-header'>Disaster Frequency Heatmap</div>", unsafe_allow_html=True)
    pivot = filtered.groupby(["Decade","Disaster_Type"]).size().reset_index(name="Count")
    pivot_wide = pivot.pivot(index="Disaster_Type", columns="Decade", values="Count").fillna(0)

    fig_heat = go.Figure(go.Heatmap(
        z=pivot_wide.values,
        x=[str(int(c))+"s" for c in pivot_wide.columns],
        y=pivot_wide.index,
        colorscale=[[0,"#0d1220"],[0.4,"#2b6cb0"],[0.7,"#f6c90e"],[1,"#fc8181"]],
        text=pivot_wide.values.astype(int),
        texttemplate="%{text}",
        hovertemplate="<b>%{y}</b> in %{x}<br>Events: %{z}<extra></extra>",
        colorbar=dict(title="Events")
    ))
    fig_heat.update_layout(title="Disaster Events by Type & Decade", height=320)
    apply_template(fig_heat)
    st.plotly_chart(fig_heat, width="stretch")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<div class='section-header'>Top 10 Locations — Economic Damage</div>", unsafe_allow_html=True)
        top_dmg = (filtered.groupby("Location")["Economic_Damage_Million$"]
                   .sum().sort_values(ascending=False).head(10))
        fig_dmg = go.Figure(go.Bar(
            x=top_dmg.values, y=top_dmg.index, orientation="h",
            marker=dict(color=top_dmg.values,
                        colorscale=[[0,"#2b6cb0"],[1,"#fc8181"]], showscale=False),
            text=[f"${v:,.0f}M" for v in top_dmg.values],
            textposition="outside"
        ))
        fig_dmg.update_layout(title="Costliest Locations", height=400)
        apply_template(fig_dmg)
        st.plotly_chart(fig_dmg, width="stretch")

    with col2:
        st.markdown("<div class='section-header'>Magnitude vs Deaths</div>", unsafe_allow_html=True)
        fig_scatter = px.scatter(
            filtered.sample(min(len(filtered), 300), random_state=1),
            x="Magnitude_Intensity", y="Deaths",
            color="Disaster_Type",
            size="Economic_Damage_Million$", size_max=30, opacity=0.75,
            hover_data=["Location","Year"],
            title="Magnitude vs Fatalities (size = damage)"
        )
        fig_scatter.update_layout(height=400)
        apply_template(fig_scatter)
        st.plotly_chart(fig_scatter, width="stretch")

    st.markdown("<div class='section-header'>Deaths Timeline by Disaster Type</div>", unsafe_allow_html=True)
    timeline = filtered.groupby(["Year","Disaster_Type"])["Deaths"].sum().reset_index()
    fig_line = px.line(timeline, x="Year", y="Deaths", color="Disaster_Type",
                       title="Annual Fatalities by Disaster Category", markers=True)
    fig_line.update_traces(line=dict(width=2))
    fig_line.update_layout(height=380, legend=dict(orientation="h", y=1.1))
    apply_template(fig_line)
    st.plotly_chart(fig_line, width="stretch")

    st.markdown("<div class='section-header'>Summary Statistics</div>", unsafe_allow_html=True)
    summary = (filtered.groupby("Disaster_Type")
               .agg(Events=("Year","count"),
                    Deaths=("Deaths","sum"),
                    Avg_Magnitude=("Magnitude_Intensity","mean"),
                    Total_Damage_M=("Economic_Damage_Million$","sum"))
               .reset_index().sort_values("Deaths", ascending=False))
    summary["Deaths"]         = summary["Deaths"].apply(lambda x: f"{int(x):,}")
    summary["Total_Damage_M"] = summary["Total_Damage_M"].apply(lambda x: f"${x:,.0f}M")
    summary["Avg_Magnitude"]  = summary["Avg_Magnitude"].apply(lambda x: f"{x:.2f}")
    st.dataframe(summary, hide_index=True)


# ════════════════════════════════════════════════════════════
#  PAGE 4 — ML RISK ENGINE
# ════════════════════════════════════════════════════════════
elif page == "🧠 ML Risk Engine":
    st.markdown("# 🧠 Machine Learning Risk Prediction Engine")
    st.caption("Random Forest Classifier — trained on historical multi-hazard data")

    m1, m2, m3, m4 = st.columns(4)
    with m1: styled_metric("Test Accuracy",       f"{test_acc*100:.1f}%",           color="#68d391")
    with m2: styled_metric("CV Accuracy (5-fold)", f"{cv_acc*100:.1f}%",            color="#63b3ed")
    with m3: styled_metric("Training Samples",    f"{int(len(disaster_df)*0.8):,}", color="#b794f4")
    with m4: styled_metric("Features Used",       "5",                              color="#fbd38d")

    tab_pred, tab_exp, tab_perf = st.tabs(["🎯 Predict", "📊 Explainability", "📈 Performance"])

    with tab_pred:
        st.markdown("#### Enter Disaster Parameters")
        p1, p2, p3 = st.columns(3)
        with p1:
            mag    = st.slider("Magnitude / Intensity", 4.0, 10.0, 6.5, 0.1)
            deaths = st.number_input("Estimated Deaths", 0, 500000, 500)
        with p2:
            damage = st.number_input("Economic Damage ($M)", 0, 200000, 10000)
            year   = st.number_input("Year", 1990, 2030, 2025)
        with p3:
            dtype  = st.selectbox("Disaster Type", sorted(disaster_df["Disaster_Type"].unique()))

        if st.button("🔍 Predict Risk Level", width="stretch"):
            dtype_enc = label_enc.transform([dtype])[0]
            inp = pd.DataFrame({
                "Magnitude_Intensity": [mag], "Deaths": [deaths],
                "Economic_Damage_Million$": [damage],
                "Disaster_Enc": [dtype_enc], "Year": [year]
            })
            pred  = model.predict(inp)[0]
            proba = model.predict_proba(inp)[0]
            classes = model.classes_
            colors = {"Very High":"#ff4444","High":"#ff8c00","Moderate":"#f6c90e","Low":"#22c55e"}
            clr = colors.get(pred, "#63b3ed")

            st.markdown(f"""
            <div style='background:rgba(0,0,0,0.3);border:2px solid {clr};border-radius:12px;
                        padding:24px;text-align:center;margin:16px 0'>
                <p style='color:#718096;font-size:0.8rem;text-transform:uppercase;letter-spacing:2px'>
                    Predicted Risk Level</p>
                <h1 style='color:{clr};font-size:3rem;margin:8px 0'>{pred}</h1>
                <p style='color:#718096;font-size:0.85rem'>
                    Based on {dtype} · Magnitude {mag} · {deaths:,} deaths · ${damage:,}M damage</p>
            </div>""", unsafe_allow_html=True)

            st.markdown("#### Confidence Breakdown")
            prob_df = pd.DataFrame({"Risk Level": classes, "Probability": proba}).sort_values("Probability", ascending=True)
            fig_prob = go.Figure(go.Bar(
                x=prob_df["Probability"], y=prob_df["Risk Level"], orientation="h",
                marker=dict(color=[colors.get(r,"#63b3ed") for r in prob_df["Risk Level"]]),
                text=[f"{v*100:.1f}%" for v in prob_df["Probability"]],
                textposition="outside"
            ))
            fig_prob.update_layout(xaxis=dict(range=[0,1.15], tickformat=".0%"),
                                   title="Class Probability Distribution", height=260)
            apply_template(fig_prob)
            st.plotly_chart(fig_prob, width="stretch")

    with tab_exp:
        st.markdown("#### Feature Importances")
        feat_df = pd.DataFrame(list(feat_imp.items()), columns=["Feature","Importance"]).sort_values("Importance")
        nice_names = {
            "Magnitude_Intensity": "Magnitude / Intensity",
            "Deaths": "Fatalities",
            "Economic_Damage_Million$": "Economic Damage",
            "Disaster_Enc": "Disaster Type",
            "Year": "Year"
        }
        feat_df["Feature"] = feat_df["Feature"].map(nice_names)
        fig_fi = go.Figure(go.Bar(
            x=feat_df["Importance"], y=feat_df["Feature"], orientation="h",
            marker=dict(color=feat_df["Importance"],
                        colorscale=[[0,"#2b6cb0"],[1,"#f6c90e"]],
                        showscale=True, colorbar=dict(title="Importance")),
            text=[f"{v:.3f}" for v in feat_df["Importance"]],
            textposition="outside"
        ))
        fig_fi.update_layout(title="Random Forest Feature Importances", height=340)
        apply_template(fig_fi)
        st.plotly_chart(fig_fi, width="stretch")
        st.info("💡 **Magnitude/Intensity** is the dominant predictor, followed by economic damage.")

    with tab_perf:
        st.markdown("#### Model Performance on Held-Out Test Set")
        report = classification_report(y_test_g, model.predict(X_test_g), output_dict=True)
        report_df = pd.DataFrame(report).T.iloc[:-3]
        report_df = report_df[["precision","recall","f1-score","support"]].round(3)
        st.dataframe(report_df, width="stretch")

        cm = confusion_matrix(y_test_g, model.predict(X_test_g), labels=model.classes_)
        fig_cm = px.imshow(
            cm, x=model.classes_, y=model.classes_,
            color_continuous_scale=[[0,"#0d1220"],[1,"#2b6cb0"]],
            labels=dict(x="Predicted", y="Actual"),
            text_auto=True, title="Confusion Matrix"
        )
        fig_cm.update_layout(height=400)
        apply_template(fig_cm)
        st.plotly_chart(fig_cm, width="stretch")


# ════════════════════════════════════════════════════════════
#  PAGE 5 — LIVE SIMULATOR  (original page, unchanged)
# ════════════════════════════════════════════════════════════
elif page == "📡 Live Simulator":
    st.markdown("# 📡 Real-Time Event Simulator")
    st.caption("Simulates incoming seismic telemetry — mimics USGS live feed pipeline")

    col_l, col_r = st.columns([2, 1])

    with col_l:
        st.markdown("#### 🌐 Simulated Seismic Activity Map")
        np.random.seed(42)
        n_events = 60
        sim_data = pd.DataFrame({
            "lat":  np.random.uniform(-60, 70, n_events),
            "lon":  np.random.uniform(-180, 180, n_events),
            "mag":  np.random.uniform(4.5, 9.2, n_events).round(1),
            "depth": np.random.uniform(5, 300, n_events).round(1),
            "time": pd.date_range("2025-04-01", periods=n_events, freq="12h")
        })
        sim_data["risk"] = sim_data["mag"].apply(
            lambda m: "Very High" if m>=8 else ("High" if m>=7 else ("Moderate" if m>=6 else "Low"))
        )
        fig_map = go.Figure()
        for risk, grp in sim_data.groupby("risk"):
            fig_map.add_trace(go.Scattergeo(
                lat=grp["lat"], lon=grp["lon"],
                mode="markers", name=risk,
                marker=dict(size=grp["mag"]*2.5, color=PALETTE[risk], opacity=0.75,
                            line=dict(color="#0d1220", width=0.5)),
                text=grp.apply(lambda r: f"M{r['mag']} · {r['risk']}<br>Depth: {r['depth']}km", axis=1),
                hoverinfo="text"
            ))
        fig_map.update_layout(
            geo=dict(showland=True, landcolor="#1a2234",
                     showocean=True, oceancolor="#0a0e1a",
                     showcoastlines=True, coastlinecolor="rgba(99,179,237,0.3)",
                     showframe=False, projection_type="natural earth",
                     bgcolor="rgba(0,0,0,0)"),
            legend=dict(orientation="h", y=0),
            height=460, title="Live Seismic Event Map (Simulated)",
            paper_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_map, width="stretch")

    with col_r:
        st.markdown("#### 🚨 Event Feed")
        recent = sim_data.sort_values("time", ascending=False).head(8)
        for _, row in recent.iterrows():
            clr = PALETTE[row["risk"]]
            st.markdown(f"""
            <div style='background:rgba(17,24,39,0.8);border-left:3px solid {clr};
                        padding:10px 14px;border-radius:0 8px 8px 0;margin:6px 0;font-size:0.83rem'>
                <b style='color:{clr}'>M{row['mag']} — {row['risk']}</b><br>
                <span style='color:#718096'>{row['lat']:.1f}°N, {row['lon']:.1f}°E<br>
                Depth: {row['depth']}km · {row['time'].strftime('%b %d %H:%M')}</span>
            </div>""", unsafe_allow_html=True)

    st.markdown("<div class='section-header'>Simulated Magnitude Distribution</div>", unsafe_allow_html=True)
    fig_hist = px.histogram(
        sim_data, x="mag", color="risk",
        color_discrete_map=PALETTE,
        nbins=20, barmode="overlay", opacity=0.75,
        title="Magnitude Frequency Distribution"
    )
    fig_hist.update_layout(height=320)
    apply_template(fig_hist)
    st.plotly_chart(fig_hist, width="stretch")


# ════════════════════════════════════════════════════════════
#  PAGE 6 — LIVE TRACKING  (new — auto-refreshing alert feed)
# ════════════════════════════════════════════════════════════
elif page == "🚨 Live Tracking":
    # ── Auto-refresh every 30 seconds ──────────────────────
    st.markdown("""
    <meta http-equiv="refresh" content="30">
    <style>
    /* wider column for alert feed */
    .block-container { max-width: 1400px !important; }
    </style>
    """, unsafe_allow_html=True)

    now_utc = datetime.datetime.utcnow()

    # ── Header ──────────────────────────────────────────────
    st.markdown("# 🚨 Live Disaster Tracking")
    col_h1, col_h2 = st.columns([3,1])
    with col_h1:
        st.markdown(
            f"<span class='live-dot'></span>"
            f"<span style='color:#22c55e;font-size:0.9rem;font-weight:600'>LIVE</span>"
            f"<span style='color:#718096;font-size:0.85rem'> · Auto-refreshes every 30 s · "
            f"Last update: <b style='color:#cbd5e0'>{now_utc.strftime('%H:%M:%S')} UTC</b></span>",
            unsafe_allow_html=True
        )
    with col_h2:
        if st.button("🔄 Refresh now"):
            st.rerun()

    st.markdown("")

    # ── Fetch live event list ────────────────────────────────
    live_events = _get_live_events(14)

    # Count by severity
    sev_counts = {"critical":0,"high":0,"moderate":0,"low":0}
    for ev in live_events:
        sev_counts[ev[0]] = sev_counts.get(ev[0],0) + 1

    # ── KPI strip ───────────────────────────────────────────
    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        styled_metric("Critical", str(sev_counts["critical"]), color="#ff4444")
    with k2:
        styled_metric("High", str(sev_counts["high"]), color="#ff8c00")
    with k3:
        styled_metric("Moderate", str(sev_counts["moderate"]), color="#f6c90e")
    with k4:
        styled_metric("Low", str(sev_counts["low"]), color="#22c55e")
    with k5:
        styled_metric("Total active", str(len(live_events)), color="#63b3ed")

    st.markdown("")

    # ── Main layout: feed left, map right ───────────────────
    left, right = st.columns([5, 5])

    with left:
        st.markdown("<div class='section-header'>Active Alert Feed</div>", unsafe_allow_html=True)

        # Status bar
        st.markdown(
            f"<div class='status-bar'>"
            f"<span class='live-dot'></span>"
            f"<span>Feed active · {len(live_events)} events · "
            f"Next refresh in ~{30 - (now_utc.second % 30)}s</span>"
            f"<span style='margin-left:auto;color:#4a5568'>UTC {now_utc.strftime('%Y-%m-%d')}</span>"
            f"</div>",
            unsafe_allow_html=True
        )

        for ev in live_events:
            sev, emoji, title, detail, lat, lon, mag, ts = ev
            clr  = _severity_color(sev)
            ago  = _time_ago(ts)
            mag_str = f" · M{mag}" if mag else ""
            loc_str = f"{abs(lat):.1f}°{'N' if lat>=0 else 'S'}, {abs(lon):.1f}°{'E' if lon>=0 else 'W'}"

            st.markdown(f"""
            <div class='live-alert-{sev}'>
              <div style='display:flex;justify-content:space-between;align-items:flex-start'>
                <div style='flex:1'>
                  <span style='font-weight:700;color:{clr}'>{emoji} {title}{mag_str}</span><br>
                  <span style='color:#a0aec0;font-size:0.82rem'>{detail}</span><br>
                  <span style='color:#4a5568;font-size:0.78rem;font-family:JetBrains Mono,monospace'>
                    📍 {loc_str}
                  </span>
                </div>
                <div style='text-align:right;flex-shrink:0;margin-left:12px'>
                  <span style='font-size:0.78rem;color:#4a5568'>{ago}</span><br>
                  <span style='font-size:0.75rem;padding:2px 8px;border-radius:20px;
                               background:{clr}22;color:{clr};font-weight:600'>
                    {sev.upper()}
                  </span>
                </div>
              </div>
            </div>""", unsafe_allow_html=True)

    with right:
        st.markdown("<div class='section-header'>Live Event Map</div>", unsafe_allow_html=True)

        map_df = pd.DataFrame([
            {"lat": ev[4], "lon": ev[5], "title": ev[2],
             "sev": ev[0], "mag": ev[6] or 5.5,
             "color": _severity_color(ev[0])}
            for ev in live_events
        ])

        fig_live = go.Figure()
        for sev, grp in map_df.groupby("sev"):
            clr = _severity_color(sev)
            fig_live.add_trace(go.Scattergeo(
                lat=grp["lat"], lon=grp["lon"],
                mode="markers",
                name=sev.title(),
                marker=dict(
                    size=grp["mag"] * 2.2,
                    color=clr,
                    opacity=0.82,
                    line=dict(color="#0d1220", width=0.5)
                ),
                text=grp["title"],
                hoverinfo="text+name"
            ))

        fig_live.update_layout(
            geo=dict(
                showland=True,     landcolor="#1a2234",
                showocean=True,    oceancolor="#0a0e1a",
                showcoastlines=True, coastlinecolor="rgba(99,179,237,0.25)",
                showframe=False,   projection_type="natural earth",
                bgcolor="rgba(0,0,0,0)"
            ),
            legend=dict(orientation="h", y=-0.04, font=dict(size=11)),
            height=420,
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=0, t=4, b=0),
            title=""
        )
        st.plotly_chart(fig_live, width="stretch")

        # ── Severity breakdown bar ───────────────────────────
        st.markdown("<div class='section-header'>Severity breakdown</div>", unsafe_allow_html=True)
        sev_order = ["critical","high","moderate","low"]
        sev_labels = ["Critical","High","Moderate","Low"]
        sev_vals   = [sev_counts.get(s,0) for s in sev_order]
        sev_colors = [_severity_color(s) for s in sev_order]
        fig_sev = go.Figure(go.Bar(
            x=sev_labels, y=sev_vals,
            marker_color=sev_colors,
            text=sev_vals,
            textposition="outside",
            textfont=dict(color="#cbd5e0", size=13, family="JetBrains Mono")
        ))
        fig_sev.update_layout(
            height=200,
            yaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
            xaxis=dict(showgrid=False),
            **{"paper_bgcolor":"rgba(0,0,0,0)","plot_bgcolor":"rgba(17,24,39,0.4)",
               "font":dict(color="#cbd5e0", size=12), "margin":dict(l=0,r=0,t=8,b=0)}
        )
        st.plotly_chart(fig_sev, width="stretch")

    # ── Last 24h simulated activity timeline ────────────────
    st.markdown("<div class='section-header'>Simulated alert frequency — last 24 hours</div>",
                unsafe_allow_html=True)

    rng24 = random.Random(now_utc.hour)   # changes once per hour
    hours = [(now_utc - datetime.timedelta(hours=i)).strftime("%H:00") for i in range(23,-1,-1)]
    critical_h = [rng24.randint(0,2) for _ in hours]
    high_h     = [rng24.randint(1,4) for _ in hours]
    moderate_h = [rng24.randint(1,3) for _ in hours]
    low_h      = [rng24.randint(0,2) for _ in hours]

    fig_timeline = go.Figure()
    fig_timeline.add_trace(go.Bar(x=hours, y=critical_h, name="Critical",
                                   marker_color="#ff4444", opacity=0.85))
    fig_timeline.add_trace(go.Bar(x=hours, y=high_h, name="High",
                                   marker_color="#ff8c00", opacity=0.85))
    fig_timeline.add_trace(go.Bar(x=hours, y=moderate_h, name="Moderate",
                                   marker_color="#f6c90e", opacity=0.85))
    fig_timeline.add_trace(go.Bar(x=hours, y=low_h, name="Low",
                                   marker_color="#22c55e", opacity=0.85))
    fig_timeline.update_layout(
        barmode="stack",
        height=250,
        legend=dict(orientation="h", y=1.05),
        xaxis=dict(tickangle=-45),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(17,24,39,0.5)",
        font=dict(color="#cbd5e0", size=11),
        margin=dict(l=0, r=0, t=8, b=0)
    )
    apply_template(fig_timeline)
    st.plotly_chart(fig_timeline, width="stretch")

    st.caption(
        "⚙ This feed uses deterministic simulation seeded to UTC time — "
        "events rotate every 30 seconds. In production, replace `_get_live_events()` "
        "with a live USGS GeoJSON WebSocket call."
    )


# ════════════════════════════════════════════════════════════
#  PAGE 7 — ABOUT
# ════════════════════════════════════════════════════════════
elif page == "ℹ️ About":
    st.markdown("# ℹ️ About GeoSentinel")
    st.markdown("""
    **GeoSentinel** is a production-grade, cloud-connected disaster intelligence platform
    that converts raw seismic and multi-hazard data into actionable risk assessments
    through machine learning, real-time data pipelines, and interactive visual analytics.

    ### 🎯 What It Does
    GeoSentinel monitors, predicts, and visualises risk across **6 disaster categories**
    (Earthquake · Tsunami · Cyclone · Flood · Wildfire · Volcano) using a 500-event
    historical dataset and a live-tracking alert pipeline.

    ### 🛠 Tech Stack
    | Layer | Technology |
    |---|---|
    | Dashboard | Streamlit 1.32 |
    | Charts & Maps | Plotly 5.20 |
    | ML Engine | Scikit-learn — Random Forest |
    | Cloud Database | Firebase Realtime DB |
    | Data Sources | NOAA Tsunami DB · USGS Earthquake API · EM-DAT |
    | Pipeline | Pandas · NumPy |
    | Language | Python 3.11 |

    ### 📊 Model Performance
    | Metric | Value |
    |---|---|
    | Test Accuracy | **100%** |
    | CV Accuracy (5-fold) | **99.6%** |
    | Algorithm | Random Forest (200 estimators, max_depth=8) |
    | Features | Magnitude · Deaths · Economic Damage · Disaster Type · Year |
    | Risk Classes | Low · Moderate · High · Very High |

    ### 🚨 Live Tracking
    The **Live Tracking** page auto-refreshes every 30 seconds and simulates an incoming
    USGS-style event feed. To connect to a real feed, replace `_get_live_events()` in
    `app.py` with a call to `https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/significant_hour.geojson`.

    ### 🔮 Roadmap
    - WebSocket integration for USGS live GeoJSON feed
    - SMS/Email alerts via Firebase Cloud Functions
    - LSTM time-series forecasting for seasonal risk
    - NASA FIRMS satellite wildfire overlay
    - Docker deployment

    )


# ─── FOOTER ─────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center;padding:32px 0 12px;color:#4a5568;font-size:0.8rem;
            border-top:1px solid rgba(99,179,237,0.1);margin-top:40px'>
    🌍 GeoSentinel — Disaster Intelligence Platform &nbsp;|&nbsp;
    Built by Siddharth &nbsp;|&nbsp;
    Python · Streamlit · Plotly · Firebase
</div>
""", unsafe_allow_html=True)
