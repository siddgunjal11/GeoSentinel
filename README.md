<div align="center">

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=13&pause=1000&color=63B3ED&center=true&vCenter=true&width=435&lines=Real-time+disaster+monitoring;Predictive+ML+risk+engine;Multi-hazard+intelligence+platform" alt="Typing SVG" />

# 🌍 GeoSentinel

### Cloud-Based Disaster Intelligence Platform

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Plotly](https://img.shields.io/badge/Plotly-5.20-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)](https://plotly.com)
[![Scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Firebase](https://img.shields.io/badge/Firebase-Realtime_DB-FFCA28?style=for-the-badge&logo=firebase&logoColor=black)](https://firebase.google.com)

<br/>

> **Monitor. Predict. Respond.**
> GeoSentinel fuses live seismic feeds, historical disaster archives, and a trained ML risk engine
> into a single dark-mode intelligence dashboard — refreshing every 30 seconds, around the clock.

<br/>

---

</div>

## 📸 Dashboard Preview

| 🏠 Overview & KPIs | 🌊 Tsunami Intelligence |
|:---:|:---:|
| ![Overview](assets/screenshots/02_overview_kpi.png) | ![Tsunami](assets/screenshots/05_tsunami_impact_matrix.png) |

| 🧠 ML Risk Engine | 📡 Live Seismic Map |
|:---:|:---:|
| ![ML](assets/screenshots/11_ml_prediction_result.png) | ![Seismic](assets/screenshots/25_seismic_map.png) |

---

## ✨ What Makes GeoSentinel Different

Unlike static disaster dashboards, GeoSentinel is built around **three pillars**:

| Pillar | What it does |
|---|---|
| 🔴 **Live Intelligence** | Auto-refreshes every 30 s · scrolling alert feed · real-time geo-map · 24h event timeline |
| 📊 **Deep Analytics** | Decade-level heatmaps · scatter plots · stacked fatality timelines · country breakdowns |
| 🤖 **ML Risk Engine** | Random Forest classifier · 99.6% CV accuracy · confidence bars · confusion matrix |

---

## 🗺 Pages at a Glance

```
🏠  Overview          →  Global KPIs + trend chart + disaster distribution + live alert strip
🌊  Tsunami           →  NOAA time-series · bubble impact matrix · country-level breakdown
🌋  Disaster Analytics →  Decade heatmap · scatter · stacked timeline · summary table + filters
🧠  ML Risk Engine    →  RF prediction · confidence bars · feature importance · confusion matrix
📡  Live Simulator    →  Randomly seeded seismic event map + magnitude histogram
🚨  Live Tracking     →  Auto-refresh · scrolling alert feed · live geo-map · 24h timeline
ℹ️  About             →  Tech stack · model metrics · roadmap
```

---

## 🚀 Quick Start

```bash
# 1 — Clone the repo
git clone https://github.com/haryflux/geosentinel.git
cd geosentinel

# 2 — Install dependencies
pip install -r requirements.txt

# 3 — Launch
streamlit run app.py
```

App opens at **http://localhost:8501** 🎉

---

## 🛠 Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **Dashboard** | Streamlit 1.32 | Multi-page app shell & UI |
| **Charts & Maps** | Plotly 5.20 | Interactive visualizations |
| **ML Engine** | Scikit-learn 1.4 | Random Forest risk classifier |
| **Cloud Database** | Firebase Realtime DB | Live event persistence |
| **Data Sources** | NOAA · USGS · EM-DAT | Historical & live disaster data |
| **Language** | Python 3.11 | Core runtime |

---

## 🧠 ML Risk Engine — Under the Hood

```
Algorithm   →  Random Forest Classifier (200 estimators)
Features    →  Magnitude · Deaths · Economic Damage · Disaster Type · Year
Output      →  Low  |  Moderate  |  High  |  Very High

Test Accuracy   ~100%  (on magnitude-derived synthetic labels)
5-Fold CV        99.6%
```

The model learns from 500 multi-hazard events and outputs a **risk class with confidence scores** and **feature importance rankings** — all rendered live in the dashboard.

---

## 🗂 Project Structure

```
geosentinel/
├── app.py                        # 🧠 Main Streamlit application (~50k)
├── requirements.txt
├── README.md
│
├── data/
│   ├── tsunamis.csv              # NOAA historical tsunami events
│   └── dummy_disasters.csv       # 500 synthetic multi-hazard events
│
├── notebooks/
│   ├── 1_Openweather.ipynb       # Weather API exploration
│   ├── 2_Current_cities.ipynb    # City-level risk profiling
│   ├── Firebase_and_Earthquake.ipynb
│   └── Tsunami_and_Plots.ipynb
│
└── assets/
    └── screenshots/              # 25 dashboard screenshots
```

---

## 🔌 Connect a Real USGS Live Feed

The Live Tracking page currently runs on a deterministic seed. To wire it to real USGS data, swap in this function inside `app.py`:

```python
import requests, datetime

def _get_live_events_real():
    url = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/significant_hour.geojson"
    data = requests.get(url, timeout=5).json()
    events = []
    for f in data["features"]:
        p, coords = f["properties"], f["geometry"]["coordinates"]
        events.append((
            "high", "🟠", p["place"], f"M{p['mag']}",
            coords[1], coords[0], p["mag"],
            datetime.datetime.utcfromtimestamp(p["time"] / 1000)
        ))
    return events
```

---

## 📊 Screenshots Gallery

<details>
<summary>Click to expand all screenshots</summary>

| # | Screen | Preview |
|---|---|---|
| 01 | Sidebar Navigation | ![](assets/screenshots/01_sidebar.png) |
| 02 | Overview KPIs | ![](assets/screenshots/02_overview_kpi.png) |
| 03 | Overview Charts | ![](assets/screenshots/03_overview_charts.png) |
| 04 | Tsunami Time-Series | ![](assets/screenshots/04_tsunami_timeseries.png) |
| 05 | Tsunami Impact Matrix | ![](assets/screenshots/05_tsunami_impact_matrix.png) |
| 06 | Tsunami by Country | ![](assets/screenshots/06_tsunami_country.png) |
| 07 | Analytics Heatmap | ![](assets/screenshots/07_analytics_heatmap.png) |
| 08 | Analytics Scatter | ![](assets/screenshots/08_analytics_scatter.png) |
| 09 | Analytics Summary | ![](assets/screenshots/09_analytics_summary.png) |
| 10 | ML Predict Input | ![](assets/screenshots/10_ml_predict.png) |
| 11 | ML Prediction Result | ![](assets/screenshots/11_ml_prediction_result.png) |
| 12 | ML Confidence Bars | ![](assets/screenshots/12_ml_confidence.png) |
| 13 | Simulator Map | ![](assets/screenshots/13_simulator_map.png) |
| 14 | Simulator Histogram | ![](assets/screenshots/14_simulator_histogram.png) |
| 25 | Live Seismic Map | ![](assets/screenshots/25_seismic_map.png) |

</details>

---

## 🏗 System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        GeoSentinel Platform                         │
└─────────────────────────────────────────────────────────────────────┘

  ┌──────────────────────────────────────────────────────────────┐
  │                      DATA SOURCES LAYER                      │
  │                                                              │
  │   🌊 NOAA Tsunami DB    🌍 USGS Earthquake API    📊 EM-DAT  │
  │        (CSV)                 (GeoJSON feed)       (synthetic)│
  └────────────────────┬─────────────────────────────────────────┘
                       │
                       ▼
  ┌──────────────────────────────────────────────────────────────┐
  │                    DATA PIPELINE LAYER                       │
  │                                                              │
  │   pandas (ETL)  →  Feature Engineering  →  Label Encoding   │
  │   • Null imputation        • Magnitude bands                 │
  │   • Type normalization     • Risk class derivation           │
  └────────────────────┬─────────────────────────────────────────┘
                       │
           ┌───────────┴────────────┐
           ▼                        ▼
  ┌─────────────────┐    ┌────────────────────────────────────┐
  │   ML ENGINE     │    │         CLOUD LAYER                │
  │                 │    │                                    │
  │ Random Forest   │    │  Firebase Realtime DB              │
  │ 200 estimators  │    │  • Live event sync                 │
  │ max_depth=8     │    │  • 30-second refresh pipeline      │
  │ 5-fold CV 99.6% │    │  • Deterministic seed fallback     │
  │                 │    │                                    │
  │ Output:         │    └────────────────────────────────────┘
  │ Risk class +    │               │
  │ Confidence %    │               │
  └────────┬────────┘               │
           │                        │
           └───────────┬────────────┘
                       ▼
  ┌──────────────────────────────────────────────────────────────┐
  │                  PRESENTATION LAYER (Streamlit)              │
  │                                                              │
  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────┐   │
  │  │Overview  │ │ Tsunami  │ │Analytics │ │  ML Engine   │   │
  │  │KPI Cards │ │NOAA plots│ │Heatmaps  │ │ Predictions  │   │
  │  └──────────┘ └──────────┘ └──────────┘ └──────────────┘   │
  │  ┌──────────┐ ┌──────────────────────────────────────────┐  │
  │  │Simulator │ │ Live Tracking  (auto-refresh / 30s)      │  │
  │  │Seismic   │ │ Alert Feed · Geo-Map · 24h Timeline      │  │
  │  └──────────┘ └──────────────────────────────────────────┘  │
  │                                                              │
  │                  Plotly · Custom CSS · Dark UI               │
  └──────────────────────────────────────────────────────────────┘
```

### Data Flow Summary

```
Raw CSVs / API feeds
      │
      ▼
pandas ETL & cleaning
      │
      ├──► Random Forest training (sklearn)  ──► Risk predictions + confidence scores
      │
      ├──► Firebase Realtime DB              ──► Live alert feed (30s refresh)
      │
      └──► Plotly figures                    ──► Interactive dashboard pages
```

---

## 🛣 Roadmap

- [ ] Real USGS live feed integration
- [ ] Firebase push notifications for critical events
- [ ] User-defined alert thresholds
- [ ] Export reports as PDF
- [ ] Deploy on Streamlit Cloud

---

<div align="center">

**Built with 🖤 by [Hari Maheshwari](https://github.com/haryflux)**

*GeoSentinel v2.1 — Disaster intelligence, always on.*

</div>
