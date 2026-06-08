"""
ml_prediction.py
────────────────
Feature 1: Water Stress Prediction Model
- Trains XGBoost on existing data to predict stress scores
- Supports climate scenario simulation (2030 / 2050)
- Returns predictions + feature importances
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px


# ── Climate Scenario Multipliers ─────────────────────────────────────────────
CLIMATE_SCENARIOS = {
    "Current (2024)": {"stress_multiplier": 1.0,  "water_multiplier": 1.0},
    "2030 Projection": {"stress_multiplier": 1.18, "water_multiplier": 1.12},
    "2050 Projection": {"stress_multiplier": 1.45, "water_multiplier": 1.35},
}

SCENARIO_COLORS = {
    "Current (2024)":  "#00d4ff",
    "2030 Projection": "#ffaa00",
    "2050 Projection": "#ff4b4b",
}


@st.cache_resource
def train_stress_model(_df: pd.DataFrame):
    """Train a GradientBoosting regressor on the dataset."""
    le = LabelEncoder()
    df = _df.copy()
    df['Region_enc'] = le.fit_transform(df['Region'].astype(str))

    features = ['Latitude', 'Longitude', 'Estimated_MW', 'Region_enc', 'Daily_Water_Liters']
    target   = 'Water_Stress_Score'

    X = df[features].fillna(0)
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = GradientBoostingRegressor(
        n_estimators=200, learning_rate=0.08,
        max_depth=4, random_state=42
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics = {
        "MAE":  round(mean_absolute_error(y_test, y_pred), 4),
        "R2":   round(r2_score(y_test, y_pred), 4),
        "RMSE": round(np.sqrt(np.mean((y_test - y_pred) ** 2)), 4),
    }

    importance = pd.DataFrame({
        "Feature": features,
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=False)

    return model, le, features, metrics, importance


def predict_scenario(df: pd.DataFrame, model, le, features: list, scenario_name: str) -> pd.DataFrame:
    """Apply a climate scenario and return predicted stress scores."""
    df2 = df.copy()
    mult = CLIMATE_SCENARIOS[scenario_name]

    df2['Region_enc']         = le.transform(df2['Region'].astype(str).map(
        lambda x: x if x in le.classes_ else le.classes_[0]
    ))
    df2['Estimated_MW']       = df2['Estimated_MW'] * mult["water_multiplier"]
    df2['Daily_Water_Liters'] = df2['Daily_Water_Liters'] * mult["water_multiplier"]

    X = df2[features].fillna(0)
    df2['Predicted_Stress'] = np.clip(model.predict(X) * mult["stress_multiplier"], 0, 5)

    df2['Predicted_Risk'] = pd.cut(
        df2['Predicted_Stress'],
        bins=[0, 1.5, 2.5, 3.5, 5.01],
        labels=['Low', 'Moderate', 'High', 'Critical']
    )
    return df2


def render_prediction_section(df: pd.DataFrame):
    """Streamlit section for the prediction feature."""
    st.markdown("<div class='section-title'>▸ ML Water Stress Predictor — Climate Scenario Analysis</div>",
                unsafe_allow_html=True)

    model, le, features, metrics, importance = train_stress_model(df)

    # Model performance pills
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        st.markdown(f"""<div class='metric-card accent'>
            <div class='metric-label'>Model Type</div>
            <div class='metric-value' style='font-size:1rem;'>Gradient Boost</div>
        </div>""", unsafe_allow_html=True)
    with col_m2:
        st.markdown(f"""<div class='metric-card safe'>
            <div class='metric-label'>R² Score</div>
            <div class='metric-value'>{metrics['R2']}</div>
            <div class='metric-sub'>Variance explained</div>
        </div>""", unsafe_allow_html=True)
    with col_m3:
        st.markdown(f"""<div class='metric-card warn'>
            <div class='metric-label'>MAE</div>
            <div class='metric-value'>{metrics['MAE']}</div>
            <div class='metric-sub'>Mean absolute error</div>
        </div>""", unsafe_allow_html=True)
    with col_m4:
        st.markdown(f"""<div class='metric-card purple'>
            <div class='metric-label'>RMSE</div>
            <div class='metric-value'>{metrics['RMSE']}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns([1, 2])

    with col1:
        scenario = st.radio(
            "Climate Scenario",
            options=list(CLIMATE_SCENARIOS.keys()),
            index=0,
        )

        # Feature importance mini chart
        fig_imp = go.Figure(go.Bar(
            x=importance['Importance'],
            y=importance['Feature'],
            orientation='h',
            marker_color='#00d4ff',
        ))
        fig_imp.update_layout(
            title=dict(text="Feature Importance", font=dict(size=11, color="#4a6080")),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#c8d8e8', size=10),
            margin=dict(l=10, r=10, t=30, b=10),
            height=200,
            xaxis=dict(gridcolor='#1a2d4a', zeroline=False),
            yaxis=dict(gridcolor='#1a2d4a'),
        )
        st.plotly_chart(fig_imp, use_container_width=True)

    with col2:
        pred_df = predict_scenario(df, model, le, features, scenario)

        # Compare actual vs predicted by region
        compare = pred_df.groupby('Region').agg(
            Actual_Stress=('Water_Stress_Score', 'mean'),
            Predicted_Stress=('Predicted_Stress', 'mean'),
        ).reset_index().sort_values('Predicted_Stress', ascending=False).head(15)

        fig_cmp = go.Figure()
        fig_cmp.add_trace(go.Bar(
            name='Current Stress',
            x=compare['Region'], y=compare['Actual_Stress'].round(3),
            marker_color='#00d4ff', opacity=0.75,
        ))
        fig_cmp.add_trace(go.Bar(
            name=f'Predicted ({scenario})',
            x=compare['Region'], y=compare['Predicted_Stress'].round(3),
            marker_color=SCENARIO_COLORS[scenario], opacity=0.9,
        ))
        fig_cmp.update_layout(
            barmode='group',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#c8d8e8', size=10),
            margin=dict(l=10, r=10, t=10, b=10),
            height=260,
            legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(size=10)),
            xaxis=dict(gridcolor='#1a2d4a', tickangle=-35),
            yaxis=dict(gridcolor='#1a2d4a', title='Stress Score (0–5)'),
        )
        st.plotly_chart(fig_cmp, use_container_width=True)

        # Delta summary
        new_critical = int((pred_df['Predicted_Risk'] == 'Critical').sum())
        old_critical = int((df['Risk_Category'] == 'Critical').sum())
        delta = new_critical - old_critical
        color = "#ff4b4b" if delta > 0 else "#00ff88"
        st.markdown(f"""
        <div style='background:#0a1628;border:1px solid #1a2d4a;border-radius:10px;padding:14px 20px;'>
            <span style='font-size:0.72rem;color:#4a6080;text-transform:uppercase;letter-spacing:1px;'>
                Scenario Impact
            </span><br>
            Critical hubs: <b style='color:{color};'>{new_critical}</b>
            &nbsp;(<span style='color:{color};'>{"+" if delta>=0 else ""}{delta}</span> vs current)
            &nbsp;·&nbsp; Avg predicted stress:
            <b style='color:#ffaa00;'>{pred_df['Predicted_Stress'].mean():.3f}</b>
        </div>""", unsafe_allow_html=True)