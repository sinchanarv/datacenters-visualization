import streamlit as st
import pandas as pd
import geopandas as gpd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(layout="wide", page_title="Advanced Analytics")

# --- CSS Styling ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500&display=swap');
    .stApp { background: #050d1a; }
    * { font-family: 'DM Sans', sans-serif; color: #c8d8e8; }
    h1, h2, h3 { color: #00d4ff !important; font-family: 'Space Mono', monospace; }
    .insight-box {
        background: rgba(0, 212, 255, 0.05);
        border-left: 3px solid #00d4ff;
        border-radius: 0 8px 8px 0;
        padding: 14px 18px;
        margin: 10px 0 24px 0;
        font-size: 0.92rem;
        line-height: 1.7;
        color: #a8c8e8;
    }
    .insight-box strong { color: #00d4ff; }
    .section-divider {
        border: none;
        border-top: 1px solid #0e2540;
        margin: 32px 0;
    }
    .metric-pill {
        display: inline-block;
        background: rgba(0,212,255,0.1);
        border: 1px solid rgba(0,212,255,0.3);
        border-radius: 20px;
        padding: 4px 12px;
        font-size: 0.8rem;
        margin: 0 4px 8px 0;
        color: #00d4ff;
    }
</style>
""", unsafe_allow_html=True)

st.title("📊 Advanced Analytics & Risk Distribution")
st.markdown("---")

# ─── DATA GENERATION ──────────────────────────────────────────────────────────
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

    risk_cat = pd.cut(
        water_stress,
        bins=[0, 1.5, 2.5, 3.5, 5.01],
        labels=['Low', 'Moderate', 'High', 'Critical']
    )

    df = pd.DataFrame({
        'Hub_Name': [f"DC-{i:03d}" for i in range(n)],
        'Region': assigned_regions,
        'Country': assigned_countries,
        'Tier': assigned_tiers,
        'Estimated_MW': mw,
        'Water_Stress_Score': water_stress,
        'Daily_Water_Liters': daily_water.astype(int),
        'PUE': pue,
        'Renewable_Pct': renewable_pct,
        'Cooling_Type': cooling_type,
        'Year_Built': year_built,
        'Carbon_Intensity': carbon_intensity,
        'Equivalent_Human_Lives': (daily_water / 135).astype(int),
        'Risk_Category': risk_cat,
    })
    return df

df = load_atlas_data()
LAYOUT = dict(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
              font_color='#c8d8e8', margin=dict(t=40, b=40, l=40, r=40))
GRID = dict(gridcolor='#0e2540', zerolinecolor='#1a2d4a')
RISK_COLORS = {'Critical': '#ff4b4b', 'High': '#ffaa00', 'Moderate': '#ffdc32', 'Low': '#00ff88'}

# ══════════════════════════════════════════════════════════════════════════════
# SECTION A — Resource Efficiency Quadrant (original)
# ══════════════════════════════════════════════════════════════════════════════
st.subheader("A. Resource Efficiency Quadrant")
st.write("Cross-referencing Data Center Capacity (MW) against Hydrological Stress to identify critical intervention targets.")

fig_quad = px.scatter(
    df, x='Water_Stress_Score', y='Estimated_MW',
    color='Risk_Category', size='Daily_Water_Liters',
    hover_name='Hub_Name', opacity=0.75,
    color_discrete_map=RISK_COLORS
)
fig_quad.add_vline(x=2.5, line_width=2, line_dash="dash", line_color="#4a6080")
fig_quad.add_hline(y=250, line_width=2, line_dash="dash", line_color="#4a6080")
fig_quad.add_annotation(x=4.3, y=750, text="🚨 HIGH RISK (Critical Action Needed)", showarrow=False, font=dict(color="#ff4b4b", size=13))
fig_quad.add_annotation(x=0.6, y=40,  text="✅ OPTIMISED HUBS (Safe Zone)",          showarrow=False, font=dict(color="#00ff88", size=13))
fig_quad.update_layout(**LAYOUT, height=480, xaxis=dict(**GRID, title="Water Stress Score (0–5)"), yaxis=dict(**GRID, title="Power Capacity (MW)"))
st.plotly_chart(fig_quad, use_container_width=True)

st.markdown("""<div class="insight-box">
<strong>How to read this:</strong> Each bubble is a data centre. Its horizontal position shows the local <strong>hydrological stress</strong> (0 = water-abundant, 5 = severe scarcity) and its vertical position shows <strong>installed power capacity in MW</strong>. Bubble size encodes daily water consumption. The dashed crosshairs divide the space into four quadrants — facilities in the <strong>top-right quadrant</strong> (high stress + high capacity) represent the most urgent intervention targets because they consume large volumes of water in already-stressed basins. Facilities in the bottom-left are operating safely.
</div>""", unsafe_allow_html=True)

st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION B — Sankey (original)
# ══════════════════════════════════════════════════════════════════════════════
st.subheader("B. Humanitarian Cost Flow")
st.write("Tracing water consumption from risk tier to aggregate volume and human-equivalent daily usage.")

risk_sums  = df.groupby('Risk_Category', observed=True)['Daily_Water_Liters'].sum()
total_water = risk_sums.sum()
total_lives = df['Equivalent_Human_Lives'].sum()

fig_sankey = go.Figure(data=[go.Sankey(
    node=dict(
        pad=20, thickness=30,
        line=dict(color="black", width=0.5),
        label=["Low Risk Hubs", "Moderate Risk Hubs", "High Risk Hubs", "Critical Risk Hubs",
               f"Total Water ({total_water/1e9:.2f}B L/day)", f"Human Equivalent ({total_lives/1e6:.1f}M people)"],
        color=["#00ff88", "#ffdc32", "#ffaa00", "#ff4b4b", "#00d4ff", "#a78bfa"]
    ),
    link=dict(
        source=[0, 1, 2, 3, 4],
        target=[4, 4, 4, 4, 5],
        value=[risk_sums.get('Low', 0), risk_sums.get('Moderate', 0),
               risk_sums.get('High', 0), risk_sums.get('Critical', 0), total_water],
        color=["rgba(0,255,136,0.3)", "rgba(255,220,50,0.3)", "rgba(255,170,0,0.3)", "rgba(255,75,75,0.3)", "rgba(0,212,255,0.4)"]
    )
)])
fig_sankey.update_layout(**LAYOUT, height=440)
st.plotly_chart(fig_sankey, use_container_width=True)

st.markdown("""<div class="insight-box">
<strong>How to read this:</strong> Flows travel left-to-right. Each coloured band originates from a <strong>risk tier</strong> and merges into the <strong>aggregate daily water pool</strong>, which then converts to a human-equivalent figure (based on 135 L/person/day — the UN minimum standard). The width of each band is proportional to that tier's share of total consumption. A wide band from the <em>Critical</em> tier means a small number of high-stress facilities account for a disproportionate share of total water demand.
</div>""", unsafe_allow_html=True)

st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION C — Grouped Bar: MW capacity by Region & Tier
# ══════════════════════════════════════════════════════════════════════════════
st.subheader("C. Capacity Distribution by Region & Tier")

region_tier = df.groupby(['Region', 'Tier'], observed=True)['Estimated_MW'].sum().reset_index()
fig_bar = px.bar(
    region_tier, x='Region', y='Estimated_MW', color='Tier', barmode='group',
    color_discrete_map={'Hyperscale': '#00d4ff', 'Enterprise': '#a78bfa', 'Edge': '#ffdc32'},
    text_auto='.0f'
)
fig_bar.update_traces(textfont_size=10, textangle=0, cliponaxis=False)
fig_bar.update_layout(**LAYOUT, height=440,
    xaxis=dict(**GRID, title="Region"),
    yaxis=dict(**GRID, title="Total Installed Capacity (MW)"),
    legend=dict(bgcolor='rgba(0,0,0,0)'),
    bargap=0.25, bargroupgap=0.08
)
st.plotly_chart(fig_bar, use_container_width=True)

st.markdown("""<div class="insight-box">
<strong>How to read this:</strong> Each cluster of bars represents a geographic region. Within each cluster, bars are split by <strong>facility tier</strong> — Hyperscale (very large campuses), Enterprise (mid-size), and Edge (small, distributed). Taller bars signal <strong>greater raw power demand</strong> in that region-tier combination. Regions with dominant Hyperscale bars are centralised infrastructure zones where a single outage or drought event could cascade across a large fraction of the network's capacity.
</div>""", unsafe_allow_html=True)

st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION D — Pie / Donut: Risk category share of total water
# ══════════════════════════════════════════════════════════════════════════════
st.subheader("D. Water Consumption by Risk Category")

col1, col2 = st.columns(2)

with col1:
    st.caption("Share of Total Daily Water Consumption")
    fig_pie = go.Figure(go.Pie(
        labels=risk_sums.index.tolist(),
        values=risk_sums.values,
        hole=0.55,
        marker=dict(colors=[RISK_COLORS[r] for r in risk_sums.index],
                    line=dict(color='#050d1a', width=3)),
        textinfo='percent+label',
        hovertemplate='<b>%{label}</b><br>%{value:,.0f} L/day<br>%{percent}<extra></extra>'
    ))
    fig_pie.add_annotation(text=f"<b>{total_water/1e9:.1f}B</b><br>litres/day",
                           x=0.5, y=0.5, showarrow=False, font=dict(size=14, color='#00d4ff'))
    fig_pie.update_layout(**LAYOUT, height=380, showlegend=True,
                          legend=dict(bgcolor='rgba(0,0,0,0)', orientation='h', y=-0.1))
    st.plotly_chart(fig_pie, use_container_width=True)

with col2:
    st.caption("Facility Count by Risk Category")
    risk_count = df['Risk_Category'].value_counts()
    fig_pie2 = go.Figure(go.Pie(
        labels=risk_count.index.tolist(),
        values=risk_count.values,
        hole=0.55,
        marker=dict(colors=[RISK_COLORS[r] for r in risk_count.index],
                    line=dict(color='#050d1a', width=3)),
        textinfo='percent+label',
        hovertemplate='<b>%{label}</b><br>%{value} facilities (%{percent})<extra></extra>'
    ))
    fig_pie2.add_annotation(text=f"<b>{len(df)}</b><br>facilities",
                            x=0.5, y=0.5, showarrow=False, font=dict(size=14, color='#a78bfa'))
    fig_pie2.update_layout(**LAYOUT, height=380, showlegend=True,
                           legend=dict(bgcolor='rgba(0,0,0,0)', orientation='h', y=-0.1))
    st.plotly_chart(fig_pie2, use_container_width=True)

st.markdown("""<div class="insight-box">
<strong>How to read this:</strong> The <strong>left donut</strong> shows what fraction of total water consumption each risk tier is responsible for — a large Critical slice means a small subset of facilities is drinking a disproportionate amount of water in stressed regions. The <strong>right donut</strong> shows how many physical facilities fall into each tier. Comparing the two reveals <em>intensity</em>: if Critical holds a larger share of water than of facility count, those individual sites consume far more per unit than lower-risk peers.
</div>""", unsafe_allow_html=True)

st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION E — Histogram: PUE distribution
# ══════════════════════════════════════════════════════════════════════════════
st.subheader("E. Power Usage Effectiveness (PUE) Distribution")

fig_hist = go.Figure()
for tier, color in [('Hyperscale', '#00d4ff'), ('Enterprise', '#a78bfa'), ('Edge', '#ffdc32')]:
    sub = df[df['Tier'] == tier]['PUE']
    fig_hist.add_trace(go.Histogram(
        x=sub, name=tier, nbinsx=18, opacity=0.72,
        marker_color=color,
        hovertemplate=f'<b>{tier}</b><br>PUE: %{{x:.2f}}<br>Count: %{{y}}<extra></extra>'
    ))
fig_hist.add_vline(x=1.2, line_dash='dot', line_color='#00ff88', line_width=1.5,
                   annotation_text='Best Practice (1.2)', annotation_font_color='#00ff88', annotation_position='top right')
fig_hist.add_vline(x=1.58, line_dash='dash', line_color='#ffaa00', line_width=1.5,
                   annotation_text='Global Avg (1.58)', annotation_font_color='#ffaa00', annotation_position='top left')
fig_hist.update_layout(**LAYOUT, height=440, barmode='overlay',
    xaxis=dict(**GRID, title="PUE Score (lower = more efficient)"),
    yaxis=dict(**GRID, title="Number of Facilities"),
    legend=dict(bgcolor='rgba(0,0,0,0)')
)
st.plotly_chart(fig_hist, use_container_width=True)

st.markdown("""<div class="insight-box">
<strong>How to read this:</strong> PUE (Power Usage Effectiveness) measures how efficiently a data centre uses electricity — a PUE of <strong>1.0</strong> is theoretical perfection (all power goes to compute), while <strong>2.0</strong> means half of all electricity is lost to cooling and overhead. The histogram shows the <em>frequency distribution</em> across all facilities, split by tier. Bars clustered near 1.2 represent best-in-class sites; bars trailing right toward 2.0 flag inefficient ageing infrastructure. Hyperscale facilities typically benefit from purpose-built cooling, while Edge sites often show higher variance.
</div>""", unsafe_allow_html=True)

st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION F — Scatter: PUE vs Renewable % coloured by Carbon Intensity
# ══════════════════════════════════════════════════════════════════════════════
st.subheader("F. Efficiency vs. Renewables — Carbon Footprint Landscape")

fig_scat2 = px.scatter(
    df, x='PUE', y='Renewable_Pct',
    color='Carbon_Intensity', size='Estimated_MW',
    hover_name='Hub_Name', hover_data={'Tier': True, 'Region': True, 'Carbon_Intensity': ':.0f'},
    color_continuous_scale='RdYlGn_r',
    opacity=0.8, size_max=30,
    labels={'PUE': 'PUE Score', 'Renewable_Pct': 'Renewable Energy (%)', 'Carbon_Intensity': 'gCO₂/kWh'}
)
fig_scat2.update_coloraxes(colorbar=dict(title='Carbon<br>Intensity<br>(gCO₂/kWh)', tickfont=dict(color='#c8d8e8')))
fig_scat2.update_layout(**LAYOUT, height=500,
    xaxis=dict(**GRID, title="PUE Score"),
    yaxis=dict(**GRID, title="Renewable Energy Share (%)")
)
st.plotly_chart(fig_scat2, use_container_width=True)

st.markdown("""<div class="insight-box">
<strong>How to read this:</strong> The ideal facility sits in the <strong>top-left corner</strong> — low PUE (efficient) and high renewable share. Colour encodes <strong>carbon intensity</strong> (grams of CO₂ per kWh): green = low-carbon, red = high-carbon. Bubble size reflects power capacity. A red bubble in the bottom-right is doubly exposed: it wastes power AND sources it from fossil fuels. Facilities in the top-left green zone demonstrate that renewable-heavy grids and efficient design correlate strongly — these should serve as architectural templates for future builds.
</div>""", unsafe_allow_html=True)

st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION G — Dot + Line plot: Median water stress by build year
# ══════════════════════════════════════════════════════════════════════════════
st.subheader("G. Water Stress Trend by Build Year")

yearly = df.groupby('Year_Built').agg(
    Median_Stress=('Water_Stress_Score', 'median'),
    Mean_Stress=('Water_Stress_Score', 'mean'),
    Count=('Hub_Name', 'count'),
    Avg_MW=('Estimated_MW', 'mean')
).reset_index()

fig_line = go.Figure()
fig_line.add_trace(go.Scatter(
    x=yearly['Year_Built'], y=yearly['Mean_Stress'],
    mode='lines', name='Mean Stress',
    line=dict(color='#ffaa00', width=2, dash='dot'),
    hovertemplate='Year: %{x}<br>Mean: %{y:.2f}<extra></extra>'
))
fig_line.add_trace(go.Scatter(
    x=yearly['Year_Built'], y=yearly['Median_Stress'],
    mode='lines+markers', name='Median Stress',
    line=dict(color='#00d4ff', width=2.5),
    marker=dict(size=yearly['Count'] * 2.5 + 4, color='#00d4ff',
                line=dict(color='#050d1a', width=2)),
    hovertemplate='Year: %{x}<br>Median: %{y:.2f}<br>Facilities: %{text}<extra></extra>',
    text=yearly['Count']
))
fig_line.add_hrect(y0=3.5, y1=5, fillcolor="rgba(255,75,75,0.07)", line_width=0, annotation_text="Critical Zone", annotation_font_color="#ff4b4b")
fig_line.add_hrect(y0=0, y1=1.5, fillcolor="rgba(0,255,136,0.05)", line_width=0, annotation_text="Safe Zone", annotation_font_color="#00ff88")
fig_line.update_layout(**LAYOUT, height=460,
    xaxis=dict(**GRID, title="Year Built", dtick=2),
    yaxis=dict(**GRID, title="Water Stress Score", range=[0, 5.2]),
    legend=dict(bgcolor='rgba(0,0,0,0)')
)
st.plotly_chart(fig_line, use_container_width=True)

st.markdown("""<div class="insight-box">
<strong>How to read this:</strong> The <strong>blue line with dots</strong> traces the median water stress score of facilities commissioned in each year; dot size grows with the number of builds that year. The <strong>amber dashed line</strong> shows the mean (which is sensitive to outliers). An upward trend over time would indicate that the industry is increasingly siting facilities in water-stressed regions — a systemic planning failure. Conversely, a downward trend suggests improving site selection practices. The shaded bands mark the Critical (red) and Safe (green) thresholds for quick reference.
</div>""", unsafe_allow_html=True)

st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION H — Stacked Bar: Cooling type breakdown by region
# ══════════════════════════════════════════════════════════════════════════════
st.subheader("H. Cooling Technology Mix by Region")

cooling_region = df.groupby(['Region', 'Cooling_Type'], observed=True).size().reset_index(name='Count')
cool_colors = {'Air': '#4a90d9', 'Liquid': '#00d4ff', 'Evaporative': '#ffdc32', 'Hybrid': '#a78bfa'}

fig_stack = px.bar(
    cooling_region, x='Region', y='Count', color='Cooling_Type', barmode='stack',
    color_discrete_map=cool_colors,
    text='Count'
)
fig_stack.update_traces(textfont_size=10)
fig_stack.update_layout(**LAYOUT, height=440,
    xaxis=dict(**GRID, title="Region"),
    yaxis=dict(**GRID, title="Number of Facilities"),
    legend=dict(bgcolor='rgba(0,0,0,0)', title='Cooling Type'),
    bargap=0.3
)
st.plotly_chart(fig_stack, use_container_width=True)

st.markdown("""<div class="insight-box">
<strong>How to read this:</strong> Each stacked bar shows the <strong>cooling technology composition</strong> within a region. <em>Air cooling</em> is cheapest but least efficient; <em>Liquid cooling</em> offers superior PUE but higher capital cost; <em>Evaporative</em> cooling is efficient but consumes large volumes of water — directly worsening stress scores; <em>Hybrid</em> systems balance both. Regions with a high proportion of Evaporative cooling in high-stress basins are compounding their hydrological risk. This chart links technology choices to the water stress outcomes seen in earlier panels.
</div>""", unsafe_allow_html=True)

st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION I — Dot plot: Top 20 facilities by daily water consumption
# ══════════════════════════════════════════════════════════════════════════════
st.subheader("I. Top 20 Facilities by Daily Water Consumption")

top20 = df.nlargest(20, 'Daily_Water_Liters').sort_values('Daily_Water_Liters')

fig_dot = go.Figure()
for rc, color in RISK_COLORS.items():
    sub = top20[top20['Risk_Category'] == rc]
    if sub.empty:
        continue
    fig_dot.add_trace(go.Scatter(
        x=sub['Daily_Water_Liters'] / 1e6,
        y=sub['Hub_Name'],
        mode='markers',
        name=rc,
        marker=dict(size=14, color=color, symbol='circle',
                    line=dict(color='#050d1a', width=1.5)),
        hovertemplate='<b>%{y}</b><br>%{x:.1f}M litres/day<extra></extra>'
    ))
# Add connector lines
for _, row in top20.iterrows():
    fig_dot.add_shape(type='line',
        x0=0, x1=row['Daily_Water_Liters']/1e6,
        y0=row['Hub_Name'], y1=row['Hub_Name'],
        line=dict(color='rgba(74,96,128,0.4)', width=1.5))

fig_dot.update_layout(**LAYOUT, height=540,
    xaxis=dict(**GRID, title="Daily Water Consumption (Million Litres)"),
    yaxis=dict(gridcolor='#0e2540', title=""),
    legend=dict(bgcolor='rgba(0,0,0,0)', title='Risk'),
    showlegend=True
)
st.plotly_chart(fig_dot, use_container_width=True)

st.markdown("""<div class="insight-box">
<strong>How to read this:</strong> A <strong>dot plot (lollipop chart)</strong> ranks the 20 most water-intensive facilities left-to-right. Each dot's colour maps to risk category, letting you immediately see whether the heaviest consumers are also operating in stressed regions (red/orange dots far to the right are the worst-case scenario). The horizontal lines emphasise the <em>gap</em> between facilities — a long jump between two adjacent dots indicates a steep drop-off in consumption and may mark a natural intervention threshold.
</div>""", unsafe_allow_html=True)

st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION J — Multi-line: Cumulative MW and Water vs Risk Score threshold
# ══════════════════════════════════════════════════════════════════════════════
st.subheader("J. Cumulative Exposure Curve — MW & Water vs. Stress Threshold")

df_sorted = df.sort_values('Water_Stress_Score').reset_index(drop=True)
df_sorted['Cum_MW'] = df_sorted['Estimated_MW'].cumsum()
df_sorted['Cum_Water_BL'] = df_sorted['Daily_Water_Liters'].cumsum() / 1e9
total_mw = df_sorted['Cum_MW'].iloc[-1]
total_water_bl = df_sorted['Cum_Water_BL'].iloc[-1]

fig_cum = make_subplots(specs=[[{"secondary_y": True}]])
fig_cum.add_trace(go.Scatter(
    x=df_sorted['Water_Stress_Score'], y=df_sorted['Cum_MW'],
    mode='lines', name='Cumulative MW',
    line=dict(color='#00d4ff', width=2.5),
    fill='tozeroy', fillcolor='rgba(0,212,255,0.07)'
), secondary_y=False)
fig_cum.add_trace(go.Scatter(
    x=df_sorted['Water_Stress_Score'], y=df_sorted['Cum_Water_BL'],
    mode='lines', name='Cumulative Water (B Litres/day)',
    line=dict(color='#a78bfa', width=2.5),
    fill='tozeroy', fillcolor='rgba(167,139,250,0.07)'
), secondary_y=True)

for threshold, color, label in [(1.5, '#00ff88', 'Low→Moderate'), (2.5, '#ffdc32', 'Moderate→High'), (3.5, '#ff4b4b', 'High→Critical')]:
    fig_cum.add_vline(x=threshold, line_dash='dash', line_color=color, line_width=1.5,
                      annotation_text=label, annotation_font_color=color, annotation_position='top left')

fig_cum.update_layout(**LAYOUT, height=460,
    xaxis=dict(**GRID, title="Water Stress Score"),
    legend=dict(bgcolor='rgba(0,0,0,0)')
)
fig_cum.update_yaxes(title_text="Cumulative Capacity (MW)", gridcolor='#0e2540', secondary_y=False)
fig_cum.update_yaxes(title_text="Cumulative Water (Billion L/day)", gridcolor='#0e2540', secondary_y=True)
st.plotly_chart(fig_cum, use_container_width=True)

st.markdown("""<div class="insight-box">
<strong>How to read this:</strong> Reading left-to-right, facilities are added in order of <em>increasing</em> water stress. The <strong>blue area</strong> tracks cumulative installed capacity (MW) and the <strong>purple area</strong> tracks cumulative daily water usage. Where these curves accelerate steeply — particularly after the dashed <em>High→Critical</em> threshold — it reveals that a concentrated block of high-stress facilities accounts for a large jump in exposure. The steeper the curve in the red zone, the more the portfolio's risk is backloaded into its most water-stressed assets. A flat curve through the Critical zone would be ideal.
</div>""", unsafe_allow_html=True)

st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION K — Box plot: Water stress by cooling type
# ══════════════════════════════════════════════════════════════════════════════
st.subheader("K. Water Stress Distribution by Cooling Technology")

fig_box = go.Figure()
order = ['Air', 'Evaporative', 'Hybrid', 'Liquid']
box_colors = {
    'Air':        ('#4a90d9', 'rgba(74,144,217,0.15)'),
    'Evaporative':('#ffdc32', 'rgba(255,220,50,0.15)'),
    'Hybrid':     ('#a78bfa', 'rgba(167,139,250,0.15)'),
    'Liquid':     ('#00d4ff', 'rgba(0,212,255,0.15)'),
}
for ct, (color, fill) in box_colors.items():
    sub = df[df['Cooling_Type'] == ct]['Water_Stress_Score']
    fig_box.add_trace(go.Box(
        y=sub, name=ct,
        marker_color=color, line_color=color,
        fillcolor=fill,
        boxmean='sd',
        hovertemplate=f'<b>{ct}</b><br>Stress: %{{y:.2f}}<extra></extra>'
    ))
fig_box.update_layout(**LAYOUT, height=440,
    xaxis=dict(gridcolor='#0e2540', title="Cooling Technology"),
    yaxis=dict(**GRID, title="Water Stress Score", range=[0, 5.5]),
)
st.plotly_chart(fig_box, use_container_width=True)

st.markdown("""<div class="insight-box">
<strong>How to read this:</strong> Each box shows the <strong>statistical spread</strong> of water stress scores for facilities using that cooling technology. The <em>box</em> spans the interquartile range (25th–75th percentile); the <em>line inside</em> is the median; <em>whiskers</em> extend to the data extremes; and the <em>diamond</em> marks the mean ± one standard deviation. A wide box means variable site selection; a box sitting high on the y-axis means that technology type is disproportionately deployed in water-stressed areas. If Evaporative cooling shows higher median stress than Liquid cooling, it suggests these facilities were sited in arid regions precisely because they prioritised low ambient temperature, inadvertently coupling high water-draw technology with scarce water supply.
</div>""", unsafe_allow_html=True)