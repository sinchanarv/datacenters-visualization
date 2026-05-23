import streamlit as st
import pandas as pd

st.title("🛠️ Interactive Policy Simulator")
df = pd.read_csv("Global_AI_Dataset_Pro.csv")

recycle_percent = st.slider("Greywater Recycling Rate (%)", 0, 100, 20)
relocate = st.checkbox("Relocate to Low-Stress Zones (30% efficiency gain)")

# Simulation Math
total_water = df['Daily_Water_Liters'].sum()
simulated_water = total_water * (1 - recycle_percent/100)
if relocate: simulated_water *= 0.70
lives_saved = int((total_water - simulated_water) / 135)

st.metric("Total Lives Sustained by Policy", f"{lives_saved:,} citizens")
st.success("This dashboard demonstrates how data-driven policy can mitigate the environmental impact of AI.")