"""
ml_forecast.py
──────────────
Feature 7: Time-Series Water Stress Forecasting
- Uses Facebook Prophet (or fallback linear trend if prophet unavailable)
- Generates synthetic monthly stress history from static data
- Forecasts 12 months ahead per region
- Shows trend, uncertainty bands, and seasonality decomposition
"""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime, timedelta


def _generate_synthetic_history(region_df: pd.DataFrame, months: int = 24) -> pd.DataFrame:
    """
    Generate synthetic monthly time-series from static stress data.
    Adds realistic seasonal variation + slight upward trend.
    """
    base_stress = region_df['Water_Stress_Score'].mean()
    base_water  = region_df['Daily_Water_Liters'].sum()
    np.random.seed(int(abs(base_stress * 100)) % 999)

    dates = pd.date_range(end=datetime.today(), periods=months, freq='MS')

    # Seasonal: summer peaks, winter dips (hemisphere-aware via latitude)
    avg_lat = region_df['Latitude'].mean() if 'Latitude' in region_df.columns else 20
    season_sign = 1 if avg_lat >= 0 else -1

    seasonal = np.array([
        np.sin((m - 3) / 12 * 2 * np.pi) * season_sign * 0.25
        for m in range(months)
    ])
    trend    = np.linspace(0, 0.15, months)            # slight upward pressure
    noise    = np.random.normal(0, 0.08, months)

    stress_series = np.clip(base_stress + seasonal + trend + noise, 0, 5)
    water_series  = base_water * (1 + seasonal * 0.1 + trend * 0.05 + noise * 0.03)

    return pd.DataFrame({
        'ds': dates,
        'y':  stress_series,
        'water': water_series,
    })


def _prophet_forecast(history_df: pd.DataFrame, periods: int = 12):
    """Try Prophet forecast; fall back to linear trend if not installed."""
    try:
        from prophet import Prophet
        import logging
        logging.getLogger('prophet').setLevel(logging.WARNING)
        logging.getLogger('cmdstanpy').setLevel(logging.WARNING)

        m = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            changepoint_prior_scale=0.15,
            seasonality_prior_scale=10,
            interval_width=0.90,
        )
        m.fit(history_df[['ds', 'y']])
        future   = m.make_future_dataframe(periods=periods, freq='MS')
        forecast = m.predict(future)
        forecast['yhat']       = np.clip(forecast['yhat'], 0, 5)
        forecast['yhat_lower'] = np.clip(forecast['yhat_lower'], 0, 5)
        forecast['yhat_upper'] = np.clip(forecast['yhat_upper'], 0, 5)
        return forecast, m, "prophet"

    except ImportError:
        # Fallback: simple linear regression extrapolation
        from sklearn.linear_model import LinearRegression
        X = np.arange(len(history_df)).reshape(-1, 1)
        y = history_df['y'].values
        reg = LinearRegression().fit(X, y)

        future_X = np.arange(len(history_df), len(history_df) + periods).reshape(-1, 1)
        future_dates = pd.date_range(
            start=history_df['ds'].iloc[-1] + pd.DateOffset(months=1),
            periods=periods, freq='MS'
        )
        pred   = np.clip(reg.predict(future_X), 0, 5)
        std    = np.std(y) * 1.2
        future_df = pd.DataFrame({
            'ds': future_dates,
            'yhat': pred,
            'yhat_lower': np.clip(pred - std, 0, 5),
            'yhat_upper': np.clip(pred + std, 0, 5),
        })
        full = pd.concat([
            history_df.rename(columns={'y': 'yhat'}).assign(
                yhat_lower=lambda d: np.clip(d['yhat'] - std * 0.5, 0, 5),
                yhat_upper=lambda d: np.clip(d['yhat'] + std * 0.5, 0, 5),
            ),
            future_df
        ], ignore_index=True)
        return full, reg, "linear"


def build_forecast_chart(history_df, forecast_df, region_name: str, model_type: str) -> go.Figure:
    cutoff = history_df['ds'].max()

    fig = go.Figure()

    # Confidence band (future only)
    fut = forecast_df[forecast_df['ds'] > cutoff]
    fig.add_trace(go.Scatter(
        x=pd.concat([fut['ds'], fut['ds'][::-1]]),
        y=pd.concat([fut['yhat_upper'], fut['yhat_lower'][::-1]]),
        fill='toself',
        fillcolor='rgba(0, 212, 255, 0.10)',
        line=dict(color='rgba(0,0,0,0)'),
        showlegend=True, name='90% Confidence Band',
        hoverinfo='skip',
    ))

    # Historical actuals
    fig.add_trace(go.Scatter(
        x=history_df['ds'], y=history_df['y'].round(3),
        mode='lines+markers',
        name='Historical Stress',
        line=dict(color='#00d4ff', width=2),
        marker=dict(size=5),
    ))

    # Forecast line
    fig.add_trace(go.Scatter(
        x=fut['ds'], y=fut['yhat'].round(3),
        mode='lines+markers',
        name='Forecast',
        line=dict(color='#ffaa00', width=2.5, dash='dot'),
        marker=dict(size=6, symbol='diamond'),
    ))

    # Risk zone lines
    for level, color, label in [
        (3.5, '#ff4b4b', 'Critical threshold'),
        (2.5, '#ffaa00', 'High threshold'),
    ]:
        fig.add_hline(y=level, line_dash='dash', line_color=color,
                      line_width=1, opacity=0.5,
                      annotation_text=label,
                      annotation_font_color=color,
                      annotation_font_size=10)

    # Cutoff line (vertical marker using scatter)
    fig.add_trace(go.Scatter(
        x=[cutoff, cutoff], y=[0, 5.2],
        mode='lines',
        line=dict(color='#4a6080', width=1, dash='dash'),
        name='Forecast start',
        showlegend=False,
        hoverinfo='skip',
    ))

    fig.update_layout(
        title=dict(
            text=f"{region_name} — Water Stress Forecast  "
                 f"<span style='font-size:10px;color:#4a6080;'>"
                 f"({model_type.upper()} model)</span>",
            font=dict(size=13, color='#c8d8e8', family='Space Mono'),
        ),
        paper_bgcolor='rgba(5,13,26,0)',
        plot_bgcolor='rgba(5,13,26,0)',
        font=dict(color='#c8d8e8', size=10),
        margin=dict(l=10, r=10, t=50, b=10),
        height=360,
        legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(size=10),
                    orientation='h', yanchor='bottom', y=1.01, xanchor='left', x=0),
        xaxis=dict(gridcolor='#1a2d4a', zeroline=False, title='Month'),
        yaxis=dict(gridcolor='#1a2d4a', zeroline=False, title='Water Stress Score (0–5)',
                   range=[0, 5.2]),
        hovermode='x unified',
    )
    return fig


def render_forecast_section(df_input: pd.DataFrame):
    """Full Streamlit section for time-series forecasting."""
    st.markdown("<div class='section-title'>▸ Time-Series Forecast — 12-Month Water Stress Outlook</div>",
                unsafe_allow_html=True)

    st.markdown("""
    <div style='background:#0a1628;border:1px solid #1a2d4a;border-radius:10px;
         padding:12px 20px;margin-bottom:16px;font-size:0.82rem;color:#4a6080;'>
    📈 <b style='color:#00d4ff;'>Prophet / Linear Trend Forecast</b> —
    Synthetic monthly stress history generated from static data with seasonal patterns.
    Forecasts 12 months ahead with 90% confidence intervals.
    </div>""", unsafe_allow_html=True)

    all_regions = sorted(df_input['Region'].unique())

    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        selected_regions = st.multiselect(
            "Select Regions to Forecast",
            options=all_regions,
            default=all_regions[:3] if len(all_regions) >= 3 else all_regions,
        )
    with col2:
        history_months = st.slider("History (months)", 12, 36, 24, key="hist_months")
    with col3:
        forecast_months = st.slider("Forecast (months)", 6, 24, 12, key="fcast_months")

    if not selected_regions:
        st.info("Select at least one region to generate forecasts.")
        return

    # Summary table across regions
    summaries = []

    for region in selected_regions:
        region_df = df_input[df_input['Region'] == region].copy()
        if len(region_df) < 3:
            continue

        history = _generate_synthetic_history(region_df, months=history_months)
        forecast, model, model_type = _prophet_forecast(history, periods=forecast_months)

        # Chart
        fig = build_forecast_chart(history, forecast, region, model_type)
        st.plotly_chart(fig, use_container_width=True)

        # Forecast summary metrics
        future_preds = forecast[forecast['ds'] > history['ds'].max()]
        if not future_preds.empty:
            current_stress = history['y'].iloc[-1]
            future_stress  = future_preds['yhat'].iloc[-1]
            delta          = future_stress - current_stress
            risk_label     = "Critical" if future_stress > 3.5 else \
                             "High" if future_stress > 2.5 else \
                             "Moderate" if future_stress > 1.5 else "Low"
            arrow = "▲" if delta > 0 else "▼"
            color = "#ff4b4b" if delta > 0.3 else "#ffaa00" if delta > 0 else "#00ff88"

            summaries.append({
                'Region':           region,
                'Current Stress':   round(current_stress, 3),
                f'+{forecast_months}M Stress': round(future_stress, 3),
                'Change':           f"{arrow} {abs(delta):.3f}",
                'Forecast Risk':    risk_label,
            })

            st.markdown(f"""
            <div style='background:#0a1628;border:1px solid #1a2d4a;border-radius:8px;
                 padding:10px 18px;margin:-10px 0 20px;display:flex;gap:32px;
                 font-size:0.82rem;'>
              <span>Current: <b style='color:#00d4ff;'>{current_stress:.3f}</b></span>
              <span>Forecast end: <b style='color:{color};'>{future_stress:.3f}</b>
                ({arrow} {abs(delta):.3f})</span>
              <span>Projected risk: <b style='color:{color};'>{risk_label}</b></span>
              <span style='color:#4a6080;'>Model: {model_type}</span>
            </div>""", unsafe_allow_html=True)

    # Cross-region summary table
    if len(summaries) > 1:
        st.markdown("**Cross-Region Forecast Summary**")
        st.dataframe(
            pd.DataFrame(summaries).style
            .background_gradient(subset=['Current Stress'], cmap='RdYlGn_r', vmin=0, vmax=5)
            .background_gradient(subset=[f'+{forecast_months}M Stress'], cmap='RdYlGn_r', vmin=0, vmax=5),
            use_container_width=True,
            height=200,
        )