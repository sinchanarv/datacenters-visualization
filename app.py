"""
app.py — AI Hydro Risk Atlas (ML-Enhanced)
──────────────────────────────────────────
Integrates 5 ML/AI layers:
  1. Water Stress Prediction (Gradient Boosting + Climate Scenarios)
  3. Anomaly Detection (Isolation Forest)
  4. AI Chatbot (Local TF-IDF NLP)
  5. Smart Placement Recommendations (ML Hybrid Scoring)
  7. Time-Series Forecasting (Prophet / Linear Trend)
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import geopandas as gpd
import plotly.express as px

st.set_page_config(
    layout="wide",
    page_title="AI Hydro Risk Atlas",
    page_icon="🌊",
    initial_sidebar_state="expanded"
)

# ── Import ML modules ────────────────────────────────────────────────────────
from ml_prediction      import render_prediction_section
from ml_anomaly         import render_anomaly_section, detect_anomalies
from ml_chatbot         import render_chatbot_section
from ml_recommendations import render_recommendations_section
from ml_forecast        import render_forecast_section

# ── Styles (unchanged from original) ─────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');
:root {
  --bg:#050d1a;--panel:#0a1628;--border:#1a2d4a;--accent:#00d4ff;
  --danger:#ff4b4b;--warn:#ffaa00;--safe:#00ff88;--text:#c8d8e8;--muted:#4a6080;
}
html,body,[data-testid="stAppViewContainer"],[data-testid="stMain"],.main{background:var(--bg)!important;}
*{font-family:'DM Sans',sans-serif;color:var(--text);}
[data-testid="stAppViewContainer"]::before{
  content:'';position:fixed;top:0;left:0;right:0;height:3px;
  background:linear-gradient(90deg,transparent,var(--accent),var(--safe),var(--danger),transparent);
  animation:scan 4s linear infinite;z-index:999;}
@keyframes scan{0%{background-position:0%}100%{background-position:200%}}
.hero{background:linear-gradient(135deg,#051020 0%,#0a1e35 50%,#051020 100%);
  border:1px solid var(--border);border-radius:12px;padding:28px 36px 20px;
  margin-bottom:20px;position:relative;overflow:hidden;}
.hero::after{content:'';position:absolute;top:-60px;right:-60px;width:200px;height:200px;
  background:radial-gradient(circle,rgba(0,212,255,0.08) 0%,transparent 70%);border-radius:50%;}
.hero h1{font-family:'Space Mono',monospace;font-size:1.7rem;font-weight:700;letter-spacing:-0.5px;
  background:linear-gradient(90deg,#ffffff,var(--accent));
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;margin:0 0 6px;}
.hero p{color:var(--muted);font-size:0.85rem;margin:0;letter-spacing:0.3px;}
.metrics-row{display:flex;gap:12px;margin-bottom:20px;flex-wrap:wrap;}
.metric-card{flex:1;min-width:160px;background:var(--panel);border:1px solid var(--border);
  border-radius:10px;padding:16px 20px;position:relative;overflow:hidden;}
.metric-card::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;}
.metric-card.danger::before{background:var(--danger);}
.metric-card.warn::before{background:var(--warn);}
.metric-card.safe::before{background:var(--safe);}
.metric-card.accent::before{background:var(--accent);}
.metric-card.purple::before{background:#a78bfa;}
.metric-label{font-size:0.7rem;color:var(--muted);text-transform:uppercase;letter-spacing:1.5px;margin-bottom:6px;}
.metric-value{font-family:'Space Mono',monospace;font-size:1.6rem;font-weight:700;line-height:1;}
.metric-card.danger .metric-value{color:var(--danger);}
.metric-card.warn .metric-value{color:var(--warn);}
.metric-card.safe .metric-value{color:var(--safe);}
.metric-card.accent .metric-value{color:var(--accent);}
.metric-card.purple .metric-value{color:#a78bfa;}
.metric-sub{font-size:0.72rem;color:var(--muted);margin-top:4px;}
[data-testid="stSidebar"]{background:var(--panel)!important;border-right:1px solid var(--border)!important;}
[data-testid="stSidebar"] *{color:var(--text)!important;}
[data-testid="stSidebar"] label{font-size:0.72rem!important;text-transform:uppercase!important;
  letter-spacing:1.2px!important;color:var(--muted)!important;}
.section-title{font-family:'Space Mono',monospace;font-size:0.75rem;text-transform:uppercase;
  letter-spacing:2px;color:var(--muted);border-bottom:1px solid var(--border);
  padding-bottom:8px;margin:24px 0 16px;}
.stDataFrame{border:1px solid var(--border)!important;border-radius:10px!important;}
#MainMenu,footer,header{visibility:hidden;}

/* Tab styling */
[data-testid="stTab"]{
  font-family:'Space Mono',monospace;font-size:0.72rem;letter-spacing:1.5px;
  text-transform:uppercase;
}
</style>
""", unsafe_allow_html=True)


# ── Load Data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_and_process_data():
    gdf = gpd.read_file("datacenters.geojson")
    gdf['Region']    = gdf['country']
    gdf['Latitude']  = gdf.geometry.y
    gdf['Longitude'] = gdf.geometry.x
    gdf['Hub_Name']  = gdf['name']

    np.random.seed(42)
    gdf['Estimated_MW']        = np.random.randint(50, 500, len(gdf))
    gdf['Daily_Water_Liters']  = gdf['Estimated_MW'] * 20000
    gdf['Equivalent_Human_Lives'] = (gdf['Daily_Water_Liters'] / 135).astype(int)
    gdf['Water_Stress_Score']  = np.random.uniform(0, 5, len(gdf))
    gdf['Risk_Category'] = pd.cut(
        gdf['Water_Stress_Score'],
        bins=[0, 1.5, 2.5, 3.5, 5.01],
        labels=['Low', 'Moderate', 'High', 'Critical']
    )
    return gdf

df = load_and_process_data()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='font-family:Space Mono,monospace;font-size:0.9rem;font-weight:700;
         color:#00d4ff;letter-spacing:1px;padding:12px 0 20px;
         border-bottom:1px solid #1a2d4a;margin-bottom:20px;'>
    ⬡ HYDRO RISK ATLAS
    </div>""", unsafe_allow_html=True)

    st.markdown("<p style='font-size:0.65rem;color:#4a6080;letter-spacing:1px;'>ML FEATURES ACTIVE</p>",
                unsafe_allow_html=True)
    for feat in ["✅ Stress Predictor", "✅ Anomaly Detection",
                 "✅ AI Chatbot", "✅ Recommendations", "✅ Forecasting"]:
        st.markdown(f"<div style='font-size:0.75rem;color:#00ff88;padding:2px 0;'>{feat}</div>",
                    unsafe_allow_html=True)

    st.markdown("<hr style='border-color:#1a2d4a;margin:16px 0;'>", unsafe_allow_html=True)

    all_regions = sorted(df['Region'].unique())
    regions = st.multiselect("Filter by Region", options=all_regions, default=all_regions)
    risk_filter = st.multiselect(
        "Risk Category",
        options=['Low', 'Moderate', 'High', 'Critical'],
        default=['Low', 'Moderate', 'High', 'Critical']
    )
    stress_range = st.slider("Water Stress Score Range", 0.0, 5.0, (0.0, 5.0), 0.1)
    mw_max = int(df['Estimated_MW'].max())
    mw_range = st.slider("Estimated MW Range", 0, mw_max, (0, mw_max), 10)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:0.68rem;color:#4a6080;line-height:1.7;'>
    <b style='color:#00d4ff;'>DATA SOURCES</b><br>
    WRI Aqueduct 4.0 · Custom AI Infrastructure Survey<br><br>
    <b style='color:#00d4ff;'>COVERAGE</b><br>
    500 data centres · 10 regions
    </div>""", unsafe_allow_html=True)


# ── Filters ───────────────────────────────────────────────────────────────────
fdf = df[
    df['Region'].isin(regions) &
    df['Risk_Category'].isin(risk_filter) &
    df['Water_Stress_Score'].between(*stress_range) &
    df['Estimated_MW'].between(*mw_range)
].copy()


# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class='hero'>
  <h1>🌐 Global AI Infrastructure · Hydrological Risk Atlas</h1>
  <p>Water stress analysis · WRI Aqueduct 4.0 · ML-Powered: Prediction · Anomaly Detection · Forecasting · Smart Recommendations</p>
</div>
""", unsafe_allow_html=True)


# ── Metric Cards ──────────────────────────────────────────────────────────────
critical_count = len(fdf[fdf['Risk_Category'] == 'Critical'])
total_water_bl = fdf['Daily_Water_Liters'].sum() / 1e9
avg_stress     = fdf['Water_Stress_Score'].mean()
total_lives_M  = fdf['Equivalent_Human_Lives'].sum() / 1e6
total_mw       = fdf['Estimated_MW'].sum()

st.markdown(f"""
<div class='metrics-row'>
  <div class='metric-card danger'>
    <div class='metric-label'>Critical Risk Hubs</div>
    <div class='metric-value'>{critical_count}</div>
    <div class='metric-sub'>Stress score &gt; 3.5 / 5</div>
  </div>
  <div class='metric-card accent'>
    <div class='metric-label'>Total Daily Consumption</div>
    <div class='metric-value'>{total_water_bl:.2f}B L</div>
    <div class='metric-sub'>Across {len(fdf)} active hubs</div>
  </div>
  <div class='metric-card warn'>
    <div class='metric-label'>Avg Stress Score</div>
    <div class='metric-value'>{avg_stress:.2f}</div>
    <div class='metric-sub'>WRI Aqueduct scale 0–5</div>
  </div>
  <div class='metric-card purple'>
    <div class='metric-label'>Equivalent Human Lives</div>
    <div class='metric-value'>{total_lives_M:.1f}M</div>
    <div class='metric-sub'>Daily water equivalent</div>
  </div>
  <div class='metric-card safe'>
    <div class='metric-label'>Total Capacity</div>
    <div class='metric-value'>{total_mw:,} MW</div>
    <div class='metric-sub'>Estimated power draw</div>
  </div>
</div>
""", unsafe_allow_html=True)


# ── Main Tabs ─────────────────────────────────────────────────────────────────
tab_map, tab_predict, tab_anomaly, tab_recommend, tab_forecast, tab_chat = st.tabs([
    "🗺️  Map & Explorer",
    "🤖  Stress Predictor",
    "🔬  Anomaly Detection",
    "📍  Recommendations",
    "📈  Forecast",
    "💬  AI Chatbot",
])


# ── TAB 1: Original Map + Table ───────────────────────────────────────────────
with tab_map:
    st.markdown("<div class='section-title'>▸ Infrastructure Geospatial View</div>", unsafe_allow_html=True)

    color_map  = {'Low': '#00ff88', 'Moderate': '#ffdc32', 'High': '#ffaa00', 'Critical': '#ff4b4b'}
    size_scale = 200_000

    fig = go.Figure()
    for risk_cat in ['Low', 'Moderate', 'High', 'Critical']:
        sub = fdf[fdf['Risk_Category'] == risk_cat]
        if sub.empty:
            continue
        if risk_cat in ('High', 'Critical'):
            fig.add_trace(go.Scattermap(
                lat=sub.geometry.y, lon=sub.geometry.x, mode='markers', showlegend=False,
                marker=dict(size=np.clip(sub['Daily_Water_Liters'] / size_scale, 6, 40) + 10,
                            color=color_map[risk_cat], opacity=0.22),
                hoverinfo='skip',
            ))
        fig.add_trace(go.Scattermap(
            lat=sub.geometry.y, lon=sub.geometry.x,
            mode='markers+text',
            name=f"{risk_cat} ({len(sub)})",
            text=sub['Hub_Name'],
            textposition='top right',
            textfont=dict(size=10, color='#ffffff', family='DM Sans'),
            marker=dict(size=np.clip(sub['Daily_Water_Liters'] / size_scale, 6, 40),
                        color=color_map[risk_cat], opacity=0.92),
            customdata=np.stack([
                sub['Hub_Name'], sub['Water_Stress_Score'].round(2),
                sub['Daily_Water_Liters'] / 1e6, sub['Estimated_MW'],
                sub['Equivalent_Human_Lives'], sub['Risk_Category'].astype(str), sub['Region'],
            ], axis=-1),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>────────────────────<br>"
                "💧 Stress: <b>%{customdata[1]}/5</b><br>"
                "🌊 Daily Usage: <b>%{customdata[2]:.1f} M L</b><br>"
                "⚡ Capacity: <b>%{customdata[3]} MW</b><br>"
                "👥 Human equiv: <b>%{customdata[4]:,}</b><br>"
                "🌍 Region: <b>%{customdata[6]}</b><extra></extra>"
            ),
        ))

    fig.update_layout(
        map=dict(style="carto-voyager", zoom=1.8, center={"lat": 25, "lon": 60}),
        uirevision='constant', paper_bgcolor='rgba(5,13,26,0)', plot_bgcolor='rgba(5,13,26,0)',
        margin={"r": 0, "t": 0, "l": 0, "b": 0}, height=600,
        legend=dict(title=dict(text="RISK LEVEL", font=dict(size=10, color="#223344", family="Space Mono")),
                    bgcolor="rgba(255,255,255,0.92)", bordercolor="#c8d8e8", borderwidth=1,
                    font=dict(size=11, color="#1a2d4a"), x=0.01, y=0.99),
        font=dict(family="DM Sans", color="#1a2d4a"),
    )

    # Zoom-based label toggle
    zoom_js = """
    <script>
    (function waitForPlotly() {
        const plots = document.querySelectorAll('.js-plotly-plot');
        if (!plots.length) { setTimeout(waitForPlotly, 300); return; }
        const plot = plots[0];
        Plotly.restyle(plot, { mode: 'markers' });
        plot.on('plotly_relayout', function() {
            const zoom = plot._fullLayout?.map?.zoom ?? 1.8;
            if (zoom >= 8) {
                Plotly.restyle(plot, { mode: 'markers+text' });
            } else if (zoom >= 5) {
                const modes = plot.data.map(t =>
                    (t.name?.startsWith('Critical') || t.name?.startsWith('High'))
                        ? 'markers+text' : 'markers');
                Plotly.restyle(plot, { mode: modes });
            } else {
                Plotly.restyle(plot, { mode: 'markers' });
            }
        });
    })();
    </script>"""

    st.plotly_chart(fig, use_container_width=True,
                    config={"scrollZoom": True, "displayModeBar": True,
                            "modeBarButtonsToRemove": ["select2d", "lasso2d"]})
    st.components.v1.html(zoom_js, height=0)

    # Region drill-down table
    st.markdown("<div class='section-title'>▸ Regional Risk Explorer</div>", unsafe_allow_html=True)
    col1, col2 = st.columns([1, 2])
    with col1:
        region_sel = st.selectbox("Region", ["All"] + sorted(df['Region'].unique()),
                                  label_visibility="collapsed")
    with col2:
        sort_col = st.selectbox("Sort by", ["Water_Stress_Score", "Daily_Water_Liters",
                                             "Estimated_MW", "Equivalent_Human_Lives"],
                                label_visibility="collapsed")

    table_df = fdf.copy() if region_sel == "All" else fdf[fdf['Region'] == region_sel]
    show_cols = ['Hub_Name', 'Region', 'Estimated_MW', 'Water_Stress_Score',
                 'Daily_Water_Liters', 'Equivalent_Human_Lives', 'Risk_Category']
    styled = table_df[show_cols].sort_values(sort_col, ascending=False).reset_index(drop=True)
    st.dataframe(
        styled.style
        .background_gradient(subset=['Water_Stress_Score'], cmap='RdYlGn_r', vmin=0, vmax=5)
        .background_gradient(subset=['Daily_Water_Liters'], cmap='Blues')
        .background_gradient(subset=['Estimated_MW'], cmap='Purples')
        .format({'Water_Stress_Score': '{:.3f}', 'Daily_Water_Liters': '{:,.0f}',
                 'Equivalent_Human_Lives': '{:,.0f}', 'Estimated_MW': '{:,}'}),
        use_container_width=True, height=350,
    )

    # Donut
    col1, col2 = st.columns([3, 1])
    with col2:
        st.subheader("Risk Distribution")
        fig_donut = px.pie(df, names='Risk_Category', hole=0.6,
                           color_discrete_map={'Critical':'#ff4b4b','High':'#ffaa00',
                                               'Moderate':'#ffdc32','Low':'#00ff88'})
        fig_donut.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                font=dict(color='#c8d8e8'))
        st.plotly_chart(fig_donut, use_container_width=True)

    # Regional aggregates
    st.markdown("<div class='section-title'>▸ Regional Aggregates</div>", unsafe_allow_html=True)
    agg = (fdf.groupby('Region').agg(
        Hubs=('Hub_Name','count'), Avg_Stress=('Water_Stress_Score','mean'),
        Total_Water_BL=('Daily_Water_Liters', lambda x: x.sum()/1e9),
        Total_MW=('Estimated_MW','sum'),
        Critical_Hubs=('Risk_Category', lambda x: (x=='Critical').sum()),
        Total_Lives_M=('Equivalent_Human_Lives', lambda x: x.sum()/1e6),
    ).sort_values('Avg_Stress', ascending=False).reset_index())
    st.dataframe(
        agg.style
        .background_gradient(subset=['Avg_Stress'], cmap='RdYlGn_r', vmin=0, vmax=5)
        .background_gradient(subset=['Critical_Hubs'], cmap='Reds')
        .background_gradient(subset=['Total_Water_BL'], cmap='Blues')
        .format({'Avg_Stress':'{:.3f}','Total_Water_BL':'{:.2f} B L',
                 'Total_MW':'{:,}','Total_Lives_M':'{:.2f} M'}),
        use_container_width=True, height=360,
    )


# ── TAB 2: ML Prediction ──────────────────────────────────────────────────────
with tab_predict:
    render_prediction_section(fdf)


# ── TAB 3: Anomaly Detection ──────────────────────────────────────────────────
with tab_anomaly:
    render_anomaly_section(fdf)


# ── TAB 4: Smart Recommendations ─────────────────────────────────────────────
with tab_recommend:
   render_recommendations_section(fdf)


# ── TAB 5: Forecasting ────────────────────────────────────────────────────────
with tab_forecast:
    render_forecast_section(fdf)


# ── TAB 6: AI Chatbot ─────────────────────────────────────────────────────────
with tab_chat:
    render_chatbot_section(fdf)


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center;padding:28px 0 10px;font-size:0.7rem;color:#2a4060;letter-spacing:0.5px;'>
HYDRO RISK ATLAS · WRI Aqueduct 4.0 · ML Layer: GBM · IsolationForest · TF-IDF NLP · Prophet
</div>""", unsafe_allow_html=True)