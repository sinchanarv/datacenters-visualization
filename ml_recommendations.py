"""
ml_recommendations.py
──────────────────────
Feature 5: Smart Placement Recommendations Engine
- Rule-based + ML hybrid
- Given a target region or requirements, recommends safe locations for new data centres
- Scores candidates on water stress, capacity gap, infrastructure density
- Output: ranked table + map of recommended zones
"""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler


# ── Scoring Weights ───────────────────────────────────────────────────────────
DEFAULT_WEIGHTS = {
    "water_stress":    0.40,   # lower is better
    "capacity_gap":    0.25,   # regions with room for more MW
    "hub_density":     0.15,   # not too overcrowded
    "infrastructure":  0.20,   # existing hubs = proven feasibility
}


@st.cache_data
def compute_region_scores(_df: pd.DataFrame, weights: dict) -> pd.DataFrame:
    """Score each region for suitability of new data centre placement."""
    agg = _df.groupby('Region').agg(
        avg_stress=('Water_Stress_Score', 'mean'),
        total_mw=('Estimated_MW', 'sum'),
        hub_count=('Hub_Name', 'count'),
        critical_pct=('Risk_Category', lambda x: (x == 'Critical').mean()),
        avg_lat=('Latitude', 'mean'),
        avg_lon=('Longitude', 'mean'),
    ).reset_index()

    scaler = MinMaxScaler()

    # Normalize (0 = worst, 1 = best for placement)
    agg['stress_score']   = 1 - scaler.fit_transform(agg[['avg_stress']])        # lower stress = better
    agg['capacity_score'] = 1 - scaler.fit_transform(agg[['total_mw']])          # less capacity = more room
    agg['density_score']  = 1 - scaler.fit_transform(agg[['hub_count']])         # fewer hubs = less crowded
    agg['infra_score']    = scaler.fit_transform(agg[['hub_count']])              # more hubs = proven infra
    agg['risk_penalty']   = 1 - scaler.fit_transform(agg[['critical_pct']])      # fewer critical = better

    agg['placement_score'] = (
        weights["water_stress"]   * agg['stress_score']   +
        weights["capacity_gap"]   * agg['capacity_score'] +
        weights["hub_density"]    * agg['density_score']  +
        weights["infrastructure"] * agg['infra_score']
    ) * agg['risk_penalty']

    agg['placement_score'] = (agg['placement_score'] * 100).round(1)

    agg['recommendation'] = pd.cut(
        agg['placement_score'],
        bins=[0, 30, 50, 70, 100],
        labels=['Not Recommended', 'Feasible', 'Good', 'Optimal']
    )

    return agg.sort_values('placement_score', ascending=False).reset_index(drop=True)


def render_recommendations_section(df: pd.DataFrame):
    """Full Streamlit section for the recommendations engine."""
    st.markdown("<div class='section-title'>▸ Smart Placement Recommendations Engine</div>",
                unsafe_allow_html=True)

    st.markdown("""
    <div style='background:#0a1628;border:1px solid #1a2d4a;border-radius:10px;
         padding:12px 20px;margin-bottom:16px;font-size:0.82rem;color:#4a6080;'>
    🧠 <b style='color:#00d4ff;'>ML Hybrid Scoring</b> — Combines water stress, capacity gap,
    infrastructure feasibility and regional risk penalty to rank regions for new data centre placement.
    </div>""", unsafe_allow_html=True)

    # Weight controls
    with st.expander("⚙️ Adjust Scoring Weights", expanded=False):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            w_stress = st.slider("Water Stress",    0.0, 1.0, 0.40, 0.05, key="w_stress")
        with col2:
            w_cap    = st.slider("Capacity Gap",    0.0, 1.0, 0.25, 0.05, key="w_cap")
        with col3:
            w_dens   = st.slider("Hub Density",     0.0, 1.0, 0.15, 0.05, key="w_dens")
        with col4:
            w_infra  = st.slider("Infrastructure",  0.0, 1.0, 0.20, 0.05, key="w_infra")
        total = w_stress + w_cap + w_dens + w_infra
        if abs(total - 1.0) > 0.01:
            st.warning(f"⚠️ Weights sum to {total:.2f} — ideally they should sum to 1.0")
        else:
            w_stress, w_cap, w_dens, w_infra = 0.40, 0.25, 0.15, 0.20

    weights = {"water_stress": w_stress, "capacity_gap": w_cap,
               "hub_density": w_dens, "infrastructure": w_infra}

    scores_df = compute_region_scores(df, weights)

    col1, col2 = st.columns([2, 1])

    with col1:
        # Map of regions coloured by placement score
        rec_colors = {
            'Optimal':          '#00ff88',
            'Good':             '#00d4ff',
            'Feasible':         '#ffaa00',
            'Not Recommended':  '#ff4b4b',
        }

        fig = go.Figure()

        for rec_level in ['Optimal', 'Good', 'Feasible', 'Not Recommended']:
            sub = scores_df[scores_df['recommendation'] == rec_level]
            if sub.empty:
                continue
            fig.add_trace(go.Scattermap(
                lat=sub['avg_lat'], lon=sub['avg_lon'],
                mode='markers+text',
                name=rec_level,
                text=sub['Region'],
                textposition='top right',
                textfont=dict(size=10, color=rec_colors[rec_level]),
                marker=dict(
                    size=np.clip(sub['placement_score'] / 5, 10, 28),
                    color=rec_colors[rec_level], opacity=0.85,
                ),
                customdata=np.stack([
                    sub['Region'],
                    sub['placement_score'],
                    sub['avg_stress'].round(3),
                    sub['total_mw'],
                    sub['hub_count'],
                    sub['recommendation'].astype(str),
                ], axis=-1),
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "──────────────────────<br>"
                    "🏆 Placement Score : <b>%{customdata[1]:.1f} / 100</b><br>"
                    "💧 Avg Stress      : <b>%{customdata[2]} / 5</b><br>"
                    "⚡ Total Capacity  : <b>%{customdata[3]:,} MW</b><br>"
                    "🏢 Hub Count       : <b>%{customdata[4]:,}</b><br>"
                    "✅ Verdict         : <b>%{customdata[5]}</b>"
                    "<extra></extra>"
                ),
            ))

        fig.update_layout(
            map=dict(style="carto-voyager", zoom=1.5, center={"lat": 25, "lon": 60}),
            paper_bgcolor='rgba(5,13,26,0)',
            plot_bgcolor='rgba(5,13,26,0)',
            margin={"r": 0, "t": 0, "l": 0, "b": 0},
            height=480,
            legend=dict(
                bgcolor="rgba(255,255,255,0.9)", bordercolor="#c8d8e8", borderwidth=1,
                font=dict(size=11, color="#1a2d4a"),
                x=0.01, y=0.99,
            ),
            uirevision='rec',
        )
        st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": True})

    with col2:
        st.markdown("**Region Ranking**")
        display = scores_df[['Region', 'placement_score', 'avg_stress',
                              'hub_count', 'recommendation']].copy()
        display.columns = ['Region', 'Score', 'Avg Stress', 'Hubs', 'Verdict']

        def color_verdict(val):
            c = {'Optimal': '#00ff88', 'Good': '#00d4ff',
                 'Feasible': '#ffaa00', 'Not Recommended': '#ff4b4b'}.get(val, '#c8d8e8')
            return f'color: {c}; font-weight: bold'

        st.dataframe(
            display.style
            .applymap(color_verdict, subset=['Verdict'])
            .background_gradient(subset=['Score'], cmap='RdYlGn', vmin=0, vmax=100)
            .format({'Score': '{:.1f}', 'Avg Stress': '{:.3f}'}),
            use_container_width=True,
            height=450,
        )

    # Summary callout
    optimal = scores_df[scores_df['recommendation'] == 'Optimal']
    if not optimal.empty:
        top = optimal.iloc[0]
        st.markdown(f"""
        <div style='background:linear-gradient(135deg,#041810,#061f12);
             border:1px solid #00ff88;border-radius:10px;padding:16px 20px;margin-top:12px;'>
            <span style='color:#00ff88;font-size:0.7rem;text-transform:uppercase;
                letter-spacing:1.5px;'>Top Recommendation</span><br>
            <span style='font-size:1.1rem;font-weight:600;color:#ffffff;'>
                {top['Region']}
            </span>
            &nbsp;<span style='color:#00ff88;'>— Score: {top['placement_score']:.1f}/100</span><br>
            <span style='color:#4a6080;font-size:0.8rem;'>
                Avg stress {top['avg_stress']:.3f} · {int(top['hub_count'])} existing hubs ·
                {int(top['total_mw']):,} MW total capacity
            </span>
        </div>""", unsafe_allow_html=True)