import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(layout="wide")
st.title("🌐 Global AI Infrastructure: Water Risk Intelligence")
df = pd.read_csv("Global_AI_Dataset_Pro.csv")

# Pro-Grade Mapbox Map
fig = px.scatter_mapbox(df, lat='Latitude', lon='Longitude', size='Daily_Water_Liters',
                        color='Water_Stress_Score', hover_name='Hub_Name', size_max=25,
                        color_continuous_scale="RdYlGn_r", zoom=1.2, mapbox_style="carto-darkmatter")
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
st.plotly_chart(fig, use_container_width=True)

# Analytics Modules
col1, col2 = st.columns(2)
with col1:
    st.subheader("Efficiency Gap (WUI vs Power)")
    fig_scatter = px.scatter(df, x='Estimated_MW', y='Daily_Water_Liters', color='Water_Stress_Score', template="plotly_dark")
    st.plotly_chart(fig_scatter, use_container_width=True)
with col2:
    st.subheader("Vulnerability Index")
    fig_hist = px.histogram(df, x='Water_Stress_Score', nbins=20, template="plotly_dark")
    st.plotly_chart(fig_hist, use_container_width=True)