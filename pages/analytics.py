import streamlit as st
import pandas as pd
import plotly.express as px

st.title("📊 Advanced Analytics")
df = pd.read_csv("Global_AI_Dataset_Pro.csv")

col1, col2 = st.columns(2)
with col1:
    st.subheader("Efficiency Gap (WUI vs Power)")
    fig_scatter = px.scatter(df, x='Estimated_MW', y='Daily_Water_Liters', size='Water_Stress_Score', 
                             color='Hub_Name', template="plotly_dark")
    st.plotly_chart(fig_scatter, use_container_width=True)

with col2:
    st.subheader("Vulnerability Heatmap")
    fig_hist = px.histogram(df, x='Water_Stress_Score', color='Water_Stress_Score', template="plotly_dark")
    st.plotly_chart(fig_hist, use_container_width=True)