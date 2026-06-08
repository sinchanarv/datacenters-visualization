"""
ml_anomaly.py
─────────────
Feature 3: Anomaly Detection with Isolation Forest
- Flags data centers that are statistical outliers
- e.g. very high water draw in a low-stress region, or critical stress with tiny capacity
- Highlights anomalies on the map with a distinct marker style
"""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


@st.cache_data
def detect_anomalies(_df: pd.DataFrame, contamination: float = 0.05) -> pd.DataFrame:
    """
    Run Isolation Forest on key numeric features.
    Returns df with added columns: anomaly_score, is_anomaly, anomaly_reason
    """
    df = _df.copy()

    features = ['Water_Stress_Score', 'Estimated_MW', 'Daily_Water_Liters',
                'Latitude', 'Longitude']
    X = df[features].fillna(0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    iso = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=42,
        n_jobs=-1,
    )
    iso.fit(X_scaled)

    df['anomaly_score']  = -iso.score_samples(X_scaled)   # higher = more anomalous
    df['is_anomaly']     = iso.predict(X_scaled) == -1    # True = anomaly

    # Human-readable reason for top anomalies
    def get_reason(row):
        reasons = []
        stress_mean = df['Water_Stress_Score'].mean()
        water_mean  = df['Daily_Water_Liters'].mean()
        mw_mean     = df['Estimated_MW'].mean()

        if row['Water_Stress_Score'] > 4.0 and row['Estimated_MW'] < mw_mean * 0.5:
            reasons.append("Critical stress, very low capacity")
        if row['Daily_Water_Liters'] > water_mean * 3:
            reasons.append("Unusually high water draw")
        if row['Water_Stress_Score'] > stress_mean * 1.8 and row['Daily_Water_Liters'] < water_mean * 0.3:
            reasons.append("High stress, low usage — possible data gap")
        if row['Water_Stress_Score'] < 1.0 and row['Daily_Water_Liters'] > water_mean * 2.5:
            reasons.append("Low-stress region, extreme consumption")
        if not reasons:
            reasons.append("Statistical outlier in feature space")
        return " · ".join(reasons)

    df['anomaly_reason'] = df.apply(get_reason, axis=1)
    return df


def render_anomaly_map(df_with_anomalies: pd.DataFrame) -> go.Figure:
    """Return a Plotly map figure with anomalies highlighted."""
    color_map = {'Low': '#00ff88', 'Moderate': '#ffdc32', 'High': '#ffaa00', 'Critical': '#ff4b4b'}
    normal = df_with_anomalies[~df_with_anomalies['is_anomaly']]
    anom   = df_with_anomalies[df_with_anomalies['is_anomaly']]

    fig = go.Figure()

    # Normal points (faded)
    for risk_cat in ['Low', 'Moderate', 'High', 'Critical']:
        sub = normal[normal['Risk_Category'] == risk_cat]
        if sub.empty:
            continue
        fig.add_trace(go.Scattermap(
            lat=sub.geometry.y, lon=sub.geometry.x,
            mode='markers',
            name=f"{risk_cat}",
            marker=dict(size=6, color=color_map[risk_cat], opacity=0.35),
            hoverinfo='skip',
            showlegend=True,
        ))

    # Anomaly glow ring
    if not anom.empty:
        fig.add_trace(go.Scattermap(
            lat=anom.geometry.y, lon=anom.geometry.x,
            mode='markers',
            showlegend=False,
            marker=dict(size=22, color='#ff00ff', opacity=0.18),
            hoverinfo='skip',
        ))
        # Anomaly core dot
        fig.add_trace(go.Scattermap(
            lat=anom.geometry.y, lon=anom.geometry.x,
            mode='markers+text',
            name=f"⚠ Anomaly ({len(anom)})",
            text=anom['Hub_Name'],
            textposition='top right',
            textfont=dict(size=10, color='#ff00ff'),
            marker=dict(
                size=12, color='#ff00ff', opacity=0.95,
                symbol='circle',
            ),
            customdata=np.stack([
                anom['Hub_Name'],
                anom['Water_Stress_Score'].round(3),
                anom['Daily_Water_Liters'] / 1e6,
                anom['Estimated_MW'],
                anom['anomaly_score'].round(3),
                anom['anomaly_reason'],
                anom['Risk_Category'].astype(str),
            ], axis=-1),
            hovertemplate=(
                "<b>⚠ ANOMALY: %{customdata[0]}</b><br>"
                "─────────────────────────────<br>"
                "💧 Stress Score  : <b>%{customdata[1]} / 5</b><br>"
                "🌊 Daily Usage   : <b>%{customdata[2]:.1f} M litres</b><br>"
                "⚡ Capacity      : <b>%{customdata[3]} MW</b><br>"
                "🔬 Anomaly Score : <b>%{customdata[4]}</b><br>"
                "🔍 Reason        : <b>%{customdata[5]}</b><br>"
                "⚠️ Risk Category : <b>%{customdata[6]}</b>"
                "<extra></extra>"
            ),
        ))

    fig.update_layout(
        map=dict(style="carto-darkmatter", zoom=1.5, center={"lat": 25, "lon": 60}),
        paper_bgcolor='rgba(5,13,26,0)',
        plot_bgcolor='rgba(5,13,26,0)',
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        height=520,
        legend=dict(
            bgcolor="rgba(10,22,40,0.9)", bordercolor="#1a2d4a", borderwidth=1,
            font=dict(size=11, color="#c8d8e8"),
            x=0.01, y=0.99,
        ),
        uirevision='anomaly',
    )
    return fig


def render_anomaly_section(df: pd.DataFrame):
    """Full Streamlit section for anomaly detection."""
    st.markdown("<div class='section-title'>▸ Anomaly Detection — Isolation Forest</div>",
                unsafe_allow_html=True)

    col_ctrl1, col_ctrl2 = st.columns([1, 3])
    with col_ctrl1:
        contamination = st.slider(
            "Anomaly Sensitivity",
            min_value=0.01, max_value=0.15, value=0.05, step=0.01,
            help="Fraction of dataset expected to be anomalous (higher = more flagged)"
        )

    df_anom = detect_anomalies(df, contamination=contamination)
    anom_df = df_anom[df_anom['is_anomaly']].sort_values('anomaly_score', ascending=False)

    with col_ctrl2:
        total_anom = len(anom_df)
        avg_score  = anom_df['anomaly_score'].mean() if total_anom else 0
        st.markdown(f"""
        <div class='metrics-row' style='margin-bottom:0;'>
          <div class='metric-card danger'>
            <div class='metric-label'>Anomalies Detected</div>
            <div class='metric-value'>{total_anom}</div>
            <div class='metric-sub'>{contamination*100:.0f}% contamination threshold</div>
          </div>
          <div class='metric-card purple'>
            <div class='metric-label'>Avg Anomaly Score</div>
            <div class='metric-value'>{avg_score:.3f}</div>
            <div class='metric-sub'>Higher = more anomalous</div>
          </div>
          <div class='metric-card warn'>
            <div class='metric-label'>Critical Anomalies</div>
            <div class='metric-value'>{len(anom_df[anom_df['Risk_Category']=='Critical'])}</div>
            <div class='metric-sub'>Highest priority</div>
          </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Map
    fig = render_anomaly_map(df_anom)
    st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": True})

    # Anomaly table
    if not anom_df.empty:
        st.markdown("**Top Detected Anomalies**")
        show = anom_df[['Hub_Name', 'Region', 'Risk_Category',
                         'Water_Stress_Score', 'Estimated_MW',
                         'Daily_Water_Liters', 'anomaly_score', 'anomaly_reason']
                      ].head(20).reset_index(drop=True)

        st.dataframe(
            show.style
            .background_gradient(subset=['anomaly_score'], cmap='Purples')
            .background_gradient(subset=['Water_Stress_Score'], cmap='RdYlGn_r', vmin=0, vmax=5)
            .format({
                'Water_Stress_Score': '{:.3f}',
                'anomaly_score':      '{:.4f}',
                'Daily_Water_Liters': '{:,.0f}',
                'Estimated_MW':       '{:,}',
            }),
            use_container_width=True,
            height=320,
        )