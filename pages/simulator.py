import streamlit as st
import pandas as pd
import geopandas as gpd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(layout="wide", page_title="Policy Simulator")

# --- CSS Styling ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');
    .stApp { background: #050d1a; }
    * { font-family: 'DM Sans', sans-serif; color: #c8d8e8; }
    h1, h2, h3 { color: #00d4ff !important; font-family: 'Space Mono', monospace; }
    .stDataFrame { border: 1px solid #1a2d4a !important; border-radius: 10px !important; }

    .kpi-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; margin: 18px 0; }
    .kpi-card {
        background: rgba(0,212,255,0.04);
        border: 1px solid rgba(0,212,255,0.18);
        border-radius: 12px;
        padding: 18px 16px;
        text-align: center;
    }
    .kpi-card.warn  { border-color: rgba(255,170,0,0.35);  background: rgba(255,170,0,0.04); }
    .kpi-card.crit  { border-color: rgba(255,75,75,0.35);  background: rgba(255,75,75,0.04); }
    .kpi-card.good  { border-color: rgba(0,255,136,0.35);  background: rgba(0,255,136,0.04); }
    .kpi-value { font-size: 1.75rem; font-weight: 700; font-family: 'Space Mono', monospace; color: #00d4ff; }
    .kpi-card.warn  .kpi-value { color: #ffaa00; }
    .kpi-card.crit  .kpi-value { color: #ff4b4b; }
    .kpi-card.good  .kpi-value { color: #00ff88; }
    .kpi-label { font-size: 0.78rem; color: #7a9ab8; margin-top: 4px; letter-spacing: 0.04em; text-transform: uppercase; }
    .kpi-delta { font-size: 0.82rem; margin-top: 6px; }

    .insight-box {
        background: rgba(0,212,255,0.05);
        border-left: 3px solid #00d4ff;
        border-radius: 0 8px 8px 0;
        padding: 12px 16px;
        margin: 10px 0 20px 0;
        font-size: 0.88rem;
        line-height: 1.65;
        color: #a8c8e8;
    }
    .insight-box strong { color: #00d4ff; }

    .policy-lever {
        background: rgba(10,25,50,0.6);
        border: 1px solid #1a2d4a;
        border-radius: 10px;
        padding: 16px 18px;
        margin-bottom: 14px;
    }
    .lever-title { font-size: 0.82rem; text-transform: uppercase; letter-spacing: 0.06em; color: #7a9ab8; margin-bottom: 4px; }

    .roadmap-header {
        background: linear-gradient(90deg, rgba(0,212,255,0.1), transparent);
        border-left: 3px solid #00d4ff;
        padding: 10px 16px;
        border-radius: 0 8px 8px 0;
        margin-bottom: 16px;
        font-family: 'Space Mono', monospace;
        font-size: 0.9rem;
        color: #00d4ff;
    }

    .verdict-box {
        border-radius: 10px;
        padding: 20px 22px;
        margin-top: 20px;
        font-size: 0.95rem;
        line-height: 1.7;
    }
    .verdict-green { background: rgba(0,255,136,0.07); border: 1px solid rgba(0,255,136,0.3); }
    .verdict-amber { background: rgba(255,170,0,0.07);  border: 1px solid rgba(255,170,0,0.3); }
    .verdict-red   { background: rgba(255,75,75,0.07);  border: 1px solid rgba(255,75,75,0.3); }

    hr.sec { border: none; border-top: 1px solid #0e2540; margin: 28px 0; }

    .stSlider > div > div > div { background: #1a2d4a !important; }
    div[data-testid="stMetric"] { background: rgba(0,212,255,0.04); border: 1px solid rgba(0,212,255,0.15); border-radius: 10px; padding: 12px; }
</style>
""", unsafe_allow_html=True)

st.title("🛠️ Strategic Policy Simulator")
st.markdown("Model the impact of water-reduction policies across the global data centre portfolio.")
st.markdown("---")

# ─── DATA ─────────────────────────────────────────────────────────────────────
@st.cache_data
def load_atlas_data():
    np.random.seed(42)
    n = 120
    regions = ['North America', 'Europe', 'Asia Pacific', 'Middle East', 'Latin America', 'Africa']
    region_weights = [0.30, 0.28, 0.25, 0.08, 0.06, 0.03]
    countries = {
        'North America': ['United States', 'Canada', 'Mexico'],
        'Europe': ['Germany', 'United Kingdom', 'Netherlands', 'France', 'Sweden'],
        'Asia Pacific': ['China', 'Japan', 'Singapore', 'Australia', 'India'],
        'Middle East': ['UAE', 'Saudi Arabia', 'Qatar'],
        'Latin America': ['Brazil', 'Chile', 'Colombia'],
        'Africa': ['South Africa', 'Nigeria', 'Kenya'],
    }
    tier_labels = ['Hyperscale', 'Enterprise', 'Edge']
    tier_weights = [0.2, 0.5, 0.3]
    assigned_regions = np.random.choice(regions, size=n, p=region_weights)
    assigned_countries = [np.random.choice(countries[r]) for r in assigned_regions]
    assigned_tiers = np.random.choice(tier_labels, size=n, p=tier_weights)
    mw_ranges = {'Hyperscale': (200, 800), 'Enterprise': (50, 250), 'Edge': (5, 60)}
    mw = np.array([np.random.randint(*mw_ranges[t]) for t in assigned_tiers])
    water_stress = np.random.beta(2, 3, n) * 5
    pue = np.random.normal(1.45, 0.18, n).clip(1.1, 2.2)
    renewable_pct = np.random.beta(2, 2, n) * 100
    cooling_type = np.random.choice(['Air', 'Liquid', 'Evaporative', 'Hybrid'], n, p=[0.35, 0.20, 0.25, 0.20])
    year_built = np.random.randint(2005, 2024, n)
    daily_water = mw * 20000 * (1 + (pue - 1.2) * 0.3)
    carbon_intensity = (1 - renewable_pct / 100) * 450 + 50
    risk_cat = pd.cut(water_stress, bins=[0, 1.5, 2.5, 3.5, 5.01], labels=['Low', 'Moderate', 'High', 'Critical'])
    return pd.DataFrame({
        'Hub_Name': [f"DC-{i:03d}" for i in range(n)],
        'Region': assigned_regions, 'Country': assigned_countries, 'Tier': assigned_tiers,
        'Estimated_MW': mw, 'Water_Stress_Score': water_stress,
        'Daily_Water_Liters': daily_water.astype(int), 'PUE': pue,
        'Renewable_Pct': renewable_pct, 'Cooling_Type': cooling_type,
        'Year_Built': year_built, 'Carbon_Intensity': carbon_intensity,
        'Equivalent_Human_Lives': (daily_water / 135).astype(int),
        'Risk_Category': risk_cat,
    })

df = load_atlas_data()
LAYOUT = dict(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
              font_color='#c8d8e8', margin=dict(t=40, b=40, l=40, r=40))
GRID = dict(gridcolor='#0e2540', zerolinecolor='#1a2d4a')
RISK_COLORS = {'Critical': '#ff4b4b', 'High': '#ffaa00', 'Moderate': '#ffdc32', 'Low': '#00ff88'}

total_water_original = df['Daily_Water_Liters'].sum()
total_lives_original = int(total_water_original / 135)
total_mw = df['Estimated_MW'].sum()
total_carbon = (df['Carbon_Intensity'] * df['Estimated_MW']).sum()  # gCO2/kWh * MW proxy
n_critical = (df['Risk_Category'] == 'Critical').sum()

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — CURRENT BASELINE KPIs
# ══════════════════════════════════════════════════════════════════════════════
st.subheader("📌 Current Baseline — Portfolio Snapshot")
st.markdown(f"""
<div class="kpi-grid">
  <div class="kpi-card crit">
    <div class="kpi-value">{total_water_original/1e9:.2f}B</div>
    <div class="kpi-label">Litres / Day</div>
    <div class="kpi-delta" style="color:#ff4b4b">Total water draw</div>
  </div>
  <div class="kpi-card warn">
    <div class="kpi-value">{total_lives_original/1e6:.1f}M</div>
    <div class="kpi-label">Human Equivalent / Day</div>
    <div class="kpi-delta" style="color:#ffaa00">@ 135 L/person UN standard</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-value">{total_mw:,}</div>
    <div class="kpi-label">Total Installed MW</div>
    <div class="kpi-delta" style="color:#00d4ff">Across {len(df)} facilities</div>
  </div>
  <div class="kpi-card crit">
    <div class="kpi-value">{n_critical}</div>
    <div class="kpi-label">Critical-Risk Hubs</div>
    <div class="kpi-delta" style="color:#ff4b4b">Stress score > 3.5</div>
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown('<hr class="sec">', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — POLICY LEVERS
# ══════════════════════════════════════════════════════════════════════════════
st.subheader("⚙️ Adjust Policy Levers")
st.markdown("Tune each lever independently. The simulation updates all panels in real time.")

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown('<div class="lever-title">💧 Water Recycling</div>', unsafe_allow_html=True)
    recycle = st.slider("Greywater Recycling Mandate (%)", 0, 100, 20, key="recycle",
                        help="% of wastewater recaptured and reused on-site")
    st.markdown('<div class="lever-title" style="margin-top:16px">⚡ PUE Improvement</div>', unsafe_allow_html=True)
    pue_improve = st.slider("Efficiency Upgrade Target — PUE Reduction (%)", 0, 40, 10, key="pue",
                             help="Reduces power overhead, indirectly cuts cooling water demand")

with col2:
    st.markdown('<div class="lever-title">🌱 Renewable Energy Mandate (%)</div>', unsafe_allow_html=True)
    renewable_mandate = st.slider("Renewable Mix Floor (%)", 0, 100, 40, key="renew",
                                   help="Forces facilities below this % to switch energy sources")
    st.markdown('<div class="lever-title" style="margin-top:16px">❄️ Cooling Technology Upgrade</div>', unsafe_allow_html=True)
    cooling_upgrade = st.slider("% of Evaporative Sites Converted to Liquid Cooling", 0, 100, 25, key="cool",
                                 help="Liquid cooling uses ~40% less water than evaporative")

with col3:
    st.markdown('<div class="lever-title">🔄 Workload Redistribution</div>', unsafe_allow_html=True)
    redistribute = st.slider("Shift Workloads from High-Stress to Low-Stress Hubs (%)", 0, 50, 15, key="redist",
                              help="Routes compute away from Critical/High hubs to Low/Moderate peers")
    st.markdown('<div class="lever-title" style="margin-top:16px">🏗️ Relocation Programme</div>', unsafe_allow_html=True)
    relocate = st.toggle("Relocate Critical Hubs (30% Efficiency Gain)", key="reloc",
                          help="Physically moves highest-stress facilities to water-abundant regions")

# ─── SIMULATION MATH ──────────────────────────────────────────────────────────
recycle_factor      = 1 - recycle / 100
pue_factor          = 1 - (pue_improve / 100) * 0.4     # PUE cut → ~40% of that reduces water
cooling_factor      = 1 - (cooling_upgrade / 100) * 0.3  # evap→liquid saves ~30% of evap water share
redis_factor        = 1 - (redistribute / 100) * 0.5     # redistribution reduces peak draw
reloc_factor        = 0.70 if relocate else 1.0

custom_water = total_water_original * recycle_factor * pue_factor * cooling_factor * redis_factor * reloc_factor
custom_lives = int(custom_water / 135)
custom_carbon = total_carbon * (1 - renewable_mandate / 200)  # proxy: partial carbon reduction

water_saved  = total_water_original - custom_water
lives_freed  = total_lives_original - custom_lives
pct_saved    = (water_saved / total_water_original) * 100

# Aggressive baseline
agg_water = total_water_original * (1-0.30) * (1-0.04) * (1-0.075) * (1-0.075) * 0.70
agg_lives = int(agg_water / 135)

st.markdown('<hr class="sec">', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — LIVE IMPACT KPIs
# ══════════════════════════════════════════════════════════════════════════════
st.subheader("📊 Simulated Policy Impact")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Water After Policy",  f"{custom_water/1e9:.2f}B L/day",  f"-{pct_saved:.1f}%")
m2.metric("Water Saved Daily",   f"{water_saved/1e9:.2f}B L",       f"{pct_saved:.1f}% reduction")
m3.metric("Human Lives Freed",   f"{lives_freed/1e6:.2f}M",         "people supplied daily")
m4.metric("Carbon Proxy Reduction", f"{((total_carbon - custom_carbon)/total_carbon*100):.1f}%", "from renewable mandate")

st.markdown('<hr class="sec">', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — WATERFALL CHART: Lever-by-lever breakdown
# ══════════════════════════════════════════════════════════════════════════════
st.subheader("A. Policy Lever Contribution — Waterfall Breakdown")

baseline_bl = total_water_original / 1e9
after_recycle   = total_water_original * recycle_factor / 1e9
after_pue       = after_recycle * pue_factor
after_cooling   = after_pue * cooling_factor
after_redis     = after_cooling * redis_factor
after_reloc     = after_redis * reloc_factor

waterfall_x = ["Baseline", "Greywater\nRecycling", "PUE\nUpgrade", "Cooling\nConversion", "Workload\nRedistribution", "Relocation", "Final State"]
waterfall_y = [baseline_bl, -(baseline_bl - after_recycle), -(after_recycle - after_pue),
               -(after_pue - after_cooling), -(after_cooling - after_redis), -(after_redis - after_reloc), after_reloc]
measure = ["absolute", "relative", "relative", "relative", "relative", "relative", "total"]
text = [f"{v:.3f}B" for v in waterfall_y]
colors = ["#00d4ff", "#00ff88", "#00ff88", "#00ff88", "#00ff88", "#a78bfa", "#ffdc32"]

fig_wf = go.Figure(go.Waterfall(
    x=waterfall_x, y=waterfall_y, measure=measure,
    textposition="outside", text=text,
    increasing_marker_color="#ff4b4b",
    decreasing_marker_color="#00ff88",
    totals_marker_color="#ffdc32",
    connector=dict(line=dict(color="#1a2d4a", width=1.5)),
    hovertemplate='<b>%{x}</b><br>%{y:.3f}B litres/day<extra></extra>'
))
fig_wf.update_layout(**LAYOUT, height=420,
    yaxis=dict(**GRID, title="Daily Water (Billion Litres)"),
    xaxis=dict(gridcolor='#0e2540')
)
st.plotly_chart(fig_wf, use_container_width=True)

st.markdown("""<div class="insight-box">
<strong>How to read this:</strong> Each bar shows the <strong>incremental water savings</strong> from applying one policy lever on top of the previous. Green bars reduce consumption; the gold bar is the final total. This lets policymakers see at a glance which single intervention delivers the most bang — and where diminishing returns set in. A very short bar means that lever has minimal additional impact given the others already applied.
</div>""", unsafe_allow_html=True)

st.markdown('<hr class="sec">', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — GAUGE CHARTS side by side
# ══════════════════════════════════════════════════════════════════════════════
st.subheader("B. Policy Effectiveness Gauges")

col_g1, col_g2, col_g3 = st.columns(3)

def make_gauge(value, title, color, suffix="%", max_val=100):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=value,
        title={'text': title, 'font': {'color': '#c8d8e8', 'size': 13}},
        number={'suffix': suffix, 'font': {'color': color, 'size': 28, 'family': 'Space Mono'}},
        gauge={
            'axis': {'range': [0, max_val], 'tickcolor': '#4a6080', 'tickfont': {'color': '#7a9ab8'}},
            'bar': {'color': color},
            'bgcolor': '#0a1929',
            'bordercolor': '#1a2d4a',
            'steps': [
                {'range': [0, max_val * 0.4], 'color': 'rgba(255,75,75,0.1)'},
                {'range': [max_val * 0.4, max_val * 0.7], 'color': 'rgba(255,220,50,0.1)'},
                {'range': [max_val * 0.7, max_val], 'color': 'rgba(0,255,136,0.1)'},
            ],
            'threshold': {'line': {'color': "#00d4ff", 'width': 3}, 'value': max_val * 0.6}
        }
    ))
    gauge_layout = {k: v for k, v in LAYOUT.items() if k != 'margin'}
    fig.update_layout(**gauge_layout, height=280, margin=dict(t=30, b=10, l=20, r=20))
    return fig

with col_g1:
    st.plotly_chart(make_gauge(pct_saved, "Overall Water Reduction", "#00d4ff"), use_container_width=True)
    st.caption("Total % reduction from baseline across all levers")
with col_g2:
    renew_actual = min(renewable_mandate + np.random.uniform(5, 15), 100)
    st.plotly_chart(make_gauge(renew_actual, "Portfolio Renewable Share", "#00ff88"), use_container_width=True)
    st.caption("Estimated green energy coverage after mandate enforcement")
with col_g3:
    risk_reduction = min((recycle * 0.3 + pue_improve * 0.4 + cooling_upgrade * 0.2 + redistribute * 0.5) / 4, 100)
    st.plotly_chart(make_gauge(risk_reduction, "Critical Site Risk Reduction", "#ffaa00"), use_container_width=True)
    st.caption("Weighted reduction in exposure score across Critical-tier hubs")

st.markdown('<hr class="sec">', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — SCENARIO TIMELINE: Projected water usage over 10 years
# ══════════════════════════════════════════════════════════════════════════════
st.subheader("C. 10-Year Water Usage Projection")

years = list(range(2025, 2036))
growth_rate = 0.07  # 7% annual data centre demand growth

# BAU: no policy, just demand growth
bau = [total_water_original * ((1 + growth_rate) ** i) / 1e9 for i in range(len(years))]

# Custom policy: apply reductions phased in over 3 years, then demand grows on reduced base
policy_base = custom_water / 1e9
custom_proj = []
for i, y in enumerate(years):
    phase_in = min(i / 3, 1.0)   # policies fully in by 2028
    reduced_base = total_water_original / 1e9 - phase_in * (total_water_original / 1e9 - policy_base)
    custom_proj.append(reduced_base * ((1 + growth_rate * 0.5) ** i))  # slower growth with efficiency

# Aggressive scenario
agg_base = agg_water / 1e9
agg_proj = []
for i in range(len(years)):
    phase_in = min(i / 2, 1.0)
    reduced_base = total_water_original / 1e9 - phase_in * (total_water_original / 1e9 - agg_base)
    agg_proj.append(reduced_base * ((1 + growth_rate * 0.3) ** i))

fig_timeline = go.Figure()
fig_timeline.add_trace(go.Scatter(
    x=years, y=bau, mode='lines', name='Business As Usual',
    line=dict(color='#ff4b4b', width=2.5, dash='dash'),
    fill='tozeroy', fillcolor='rgba(255,75,75,0.05)',
    hovertemplate='%{x}: %{y:.2f}B L/day<extra>BAU</extra>'
))
fig_timeline.add_trace(go.Scatter(
    x=years, y=custom_proj, mode='lines+markers', name='Your Custom Policy',
    line=dict(color='#00d4ff', width=2.5),
    marker=dict(size=7, symbol='circle', color='#00d4ff'),
    fill='tozeroy', fillcolor='rgba(0,212,255,0.07)',
    hovertemplate='%{x}: %{y:.2f}B L/day<extra>Custom Policy</extra>'
))
fig_timeline.add_trace(go.Scatter(
    x=years, y=agg_proj, mode='lines+markers', name='Aggressive Roadmap',
    line=dict(color='#00ff88', width=2, dash='dot'),
    marker=dict(size=6, symbol='diamond', color='#00ff88'),
    hovertemplate='%{x}: %{y:.2f}B L/day<extra>Aggressive</extra>'
))
fig_timeline.add_vrect(x0=2025, x1=2028, fillcolor="rgba(167,139,250,0.06)",
                       annotation_text="Phase-In Period", annotation_font_color="#a78bfa",
                       line_width=0, annotation_position="top left")
fig_timeline.update_layout(**LAYOUT, height=460,
    xaxis=dict(**GRID, title="Year", dtick=1),
    yaxis=dict(**GRID, title="Daily Water Consumption (Billion Litres)"),
    legend=dict(bgcolor='rgba(0,0,0,0)', orientation='h', y=-0.15)
)
st.plotly_chart(fig_timeline, use_container_width=True)

st.markdown("""<div class="insight-box">
<strong>How to read this:</strong> The <strong>red dashed line</strong> is the Business-As-Usual trajectory assuming 7% annual demand growth and no policy action. The <strong>blue line</strong> applies your custom policy settings, phased in over 3 years (purple shaded zone), then continues growing on the reduced base at half the BAU growth rate (efficiency compounding). The <strong>green dotted line</strong> shows the aggressive roadmap. The growing gap between BAU and your policy line is the cumulative water saved — every pixel of separation represents millions of litres returned to stressed basins.
</div>""", unsafe_allow_html=True)

st.markdown('<hr class="sec">', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — REGION-LEVEL IMPACT TABLE + BAR
# ══════════════════════════════════════════════════════════════════════════════
st.subheader("D. Regional Impact Breakdown")

region_stats = df.groupby('Region').agg(
    Facilities=('Hub_Name', 'count'),
    Total_MW=('Estimated_MW', 'sum'),
    Baseline_Water_BL=('Daily_Water_Liters', lambda x: x.sum() / 1e9),
    Avg_Stress=('Water_Stress_Score', 'mean'),
    Critical_Hubs=('Risk_Category', lambda x: (x == 'Critical').sum())
).reset_index()

region_stats['Policy_Water_BL'] = region_stats['Baseline_Water_BL'] * (custom_water / total_water_original)
region_stats['Saved_BL'] = region_stats['Baseline_Water_BL'] - region_stats['Policy_Water_BL']
region_stats['Pct_Saved'] = (region_stats['Saved_BL'] / region_stats['Baseline_Water_BL'] * 100).round(1)

fig_reg = go.Figure()
fig_reg.add_trace(go.Bar(
    name='Baseline', x=region_stats['Region'],
    y=region_stats['Baseline_Water_BL'],
    marker_color='rgba(255,75,75,0.6)',
    hovertemplate='<b>%{x}</b><br>Baseline: %{y:.3f}B L/day<extra></extra>'
))
fig_reg.add_trace(go.Bar(
    name='After Policy', x=region_stats['Region'],
    y=region_stats['Policy_Water_BL'],
    marker_color='rgba(0,212,255,0.7)',
    hovertemplate='<b>%{x}</b><br>Post-Policy: %{y:.3f}B L/day<extra></extra>'
))
fig_reg.update_layout(**LAYOUT, height=400, barmode='group',
    xaxis=dict(**GRID, title=""),
    yaxis=dict(**GRID, title="Daily Water (Billion Litres)"),
    legend=dict(bgcolor='rgba(0,0,0,0)'),
    bargap=0.25, bargroupgap=0.1
)
st.plotly_chart(fig_reg, use_container_width=True)

# Styled detail table
display_df = region_stats[['Region','Facilities','Critical_Hubs','Baseline_Water_BL','Policy_Water_BL','Pct_Saved','Avg_Stress']].copy()
display_df.columns = ['Region','Facilities','Critical Hubs','Baseline (BL/day)','Post-Policy (BL/day)','Saved (%)','Avg Stress']
display_df['Baseline (BL/day)'] = display_df['Baseline (BL/day)'].map('{:.3f}'.format)
display_df['Post-Policy (BL/day)'] = display_df['Post-Policy (BL/day)'].map('{:.3f}'.format)
display_df['Avg Stress'] = display_df['Avg Stress'].map('{:.2f}'.format)
st.dataframe(display_df, use_container_width=True, hide_index=True)

st.markdown("""<div class="insight-box">
<strong>How to read this:</strong> Each region pair shows baseline (red) vs post-policy (blue) water draw. Regions with many Critical hubs and high average stress scores are the primary targets — but also where policy friction is highest. The table below quantifies the absolute and percentage savings per region so implementation budgets can be allocated proportionally.
</div>""", unsafe_allow_html=True)

st.markdown('<hr class="sec">', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — WHAT-IF SCENARIO COMPARISON TABLE
# ══════════════════════════════════════════════════════════════════════════════
st.subheader("E. Executive 'What-If' Scenario Comparison")
st.markdown('<div class="roadmap-header">Comparing custom settings against standard policy roadmaps</div>', unsafe_allow_html=True)

agg_water2 = total_water_original * (1 - 0.30) * (1 - 0.04) * (1 - 0.075) * (1 - 0.075) * 0.70
agg_lives2 = int(agg_water2 / 135)
conservative_water = total_water_original * (1 - 0.10)
conservative_lives = int(conservative_water / 135)

roadmap_data = {
    "Scenario": [
        "🔴 Current Baseline (No Action)",
        "🟡 Conservative (10% Recycling Only)",
        "🔵 Your Custom Policy",
        "🟢 Aggressive Roadmap (All Levers Max)"
    ],
    "Recycling": ["0%", "10%", f"{recycle}%", "30%"],
    "Relocation": ["No", "No", "Yes" if relocate else "No", "Yes"],
    "PUE Target": ["—", "—", f"-{pue_improve}%", "-40%"],
    "Renewables": ["—", "—", f"{renewable_mandate}%", "100%"],
    "Daily Water (BL)": [
        f"{total_water_original/1e9:.3f}",
        f"{conservative_water/1e9:.3f}",
        f"{custom_water/1e9:.3f}",
        f"{agg_water2/1e9:.3f}"
    ],
    "vs Baseline": [
        "—",
        f"-{(1-conservative_water/total_water_original)*100:.1f}%",
        f"-{pct_saved:.1f}%",
        f"-{(1-agg_water2/total_water_original)*100:.1f}%"
    ],
    "Lives Freed (M)": [
        "0",
        f"{(total_lives_original - conservative_lives)/1e6:.2f}",
        f"{lives_freed/1e6:.2f}",
        f"{(total_lives_original - agg_lives2)/1e6:.2f}"
    ]
}
st.dataframe(pd.DataFrame(roadmap_data), use_container_width=True, hide_index=True)

st.markdown('<hr class="sec">', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 9 — COST-BENEFIT ANALYSIS CHART
# ══════════════════════════════════════════════════════════════════════════════
st.subheader("F. Estimated Cost vs. Water Saved — Policy ROI")

levers = ['Greywater Recycling', 'PUE Upgrade', 'Cooling Conversion', 'Workload Redistribution', 'Relocation Programme']
est_cost_usd_bn = [0.8, 1.5, 2.2, 0.3, 4.5]
water_save_per_bn = [
    (total_water_original * (recycle/100)) / 1e9,
    (total_water_original * (pue_improve/100) * 0.4) / 1e9,
    (total_water_original * (cooling_upgrade/100) * 0.3) / 1e9,
    (total_water_original * (redistribute/100) * 0.5) / 1e9,
    (total_water_original * 0.30) / 1e9 if relocate else 0
]
roi = [ws / c if c > 0 else 0 for ws, c in zip(water_save_per_bn, est_cost_usd_bn)]

fig_roi = go.Figure()
fig_roi.add_trace(go.Bar(
    x=levers, y=water_save_per_bn,
    name='Water Saved (BL/day)',
    marker_color='#00d4ff',
    yaxis='y',
    hovertemplate='<b>%{x}</b><br>Water Saved: %{y:.3f}B L/day<extra></extra>'
))
fig_roi.add_trace(go.Scatter(
    x=levers, y=est_cost_usd_bn,
    name='Est. Implementation Cost ($Bn)',
    mode='lines+markers',
    line=dict(color='#ffaa00', width=2.5),
    marker=dict(size=10, symbol='diamond'),
    yaxis='y2',
    hovertemplate='<b>%{x}</b><br>Cost: $%{y:.1f}Bn<extra></extra>'
))
fig_roi.update_layout(**LAYOUT, height=440,
    yaxis=dict(**GRID, title="Water Saved (Billion L/day)"),
    yaxis2=dict(title="Est. Cost (USD Billion)", overlaying='y', side='right',
                gridcolor='rgba(0,0,0,0)', tickcolor='#ffaa00', tickfont=dict(color='#ffaa00')),
    xaxis=dict(gridcolor='#0e2540'),
    legend=dict(bgcolor='rgba(0,0,0,0)', orientation='h', y=-0.2)
)
st.plotly_chart(fig_roi, use_container_width=True)

st.markdown("""<div class="insight-box">
<strong>How to read this:</strong> Blue bars show how much water each lever saves (at your current slider values). The amber line plots estimated implementation cost per lever. <strong>Workload Redistribution</strong> typically has the best ROI — low cost, meaningful water shift. <strong>Relocation</strong> has the highest absolute savings but also the highest capital requirement. Use this chart to sequence policy implementation: highest-ROI levers first to build political momentum and free capital for costlier interventions.
</div>""", unsafe_allow_html=True)

st.markdown('<hr class="sec">', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 10 — VERDICT
# ══════════════════════════════════════════════════════════════════════════════
st.subheader("🏁 Policy Verdict")

if pct_saved >= (1 - agg_water2 / total_water_original) * 100:
    verdict_class = "verdict-green"
    verdict_icon = "🌟"
    verdict_title = "EXCEEDS AGGRESSIVE ROADMAP"
    verdict_body = (f"Your custom policy achieves a <strong>{pct_saved:.1f}% reduction</strong> in daily water consumption, "
                    f"surpassing even the most aggressive UN-aligned roadmap. This frees water equivalent to "
                    f"<strong>{lives_freed/1e6:.2f} million people's</strong> daily needs. "
                    f"Your combination of levers is portfolio-ready for immediate regulatory submission.")
elif pct_saved >= (1 - conservative_water / total_water_original) * 100:
    verdict_class = "verdict-amber"
    verdict_icon = "⚠️"
    verdict_title = "MEANINGFUL PROGRESS — ROOM TO GROW"
    verdict_body = (f"Your policy achieves <strong>{pct_saved:.1f}% reduction</strong>, outperforming the conservative baseline "
                    f"but falling short of the aggressive target by "
                    f"{((1-agg_water2/total_water_original)*100 - pct_saved):.1f} percentage points. "
                    f"Consider increasing recycling above 30% or enabling the relocation toggle to close the gap.")
else:
    verdict_class = "verdict-red"
    verdict_icon = "🚨"
    verdict_title = "INSUFFICIENT — BELOW CONSERVATIVE THRESHOLD"
    verdict_body = (f"At <strong>{pct_saved:.1f}% reduction</strong>, your current settings fall below even the conservative "
                    f"10% recycling-only baseline. The portfolio remains critically exposed. "
                    f"Prioritise Workload Redistribution and Greywater Recycling as immediate first steps.")

st.markdown(f"""
<div class="verdict-box {verdict_class}">
  <div style="font-family:'Space Mono',monospace; font-size:1rem; margin-bottom:8px;">
    {verdict_icon} &nbsp;<strong>{verdict_title}</strong>
  </div>
  <div style="font-size:0.92rem; line-height:1.75;">{verdict_body}</div>
  <div style="margin-top:14px; font-size:0.82rem; color:#7a9ab8;">
    Simulation parameters: {recycle}% recycling · {pue_improve}% PUE improvement · {cooling_upgrade}% cooling conversion · 
    {redistribute}% workload shift · {renewable_mandate}% renewable mandate · Relocation: {"ON" if relocate else "OFF"}
  </div>
</div>
""", unsafe_allow_html=True)