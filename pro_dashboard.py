import pandas as pd
import plotly.express as px
import streamlit as st

# Setup the page
st.set_page_config(layout="wide")
st.title("🌐 Global AI Infrastructure: Water Risk Intelligence")
st.markdown("### Real-time analysis of Hyperscale Data Centers vs. Global Hydrological Stress")

# Load your 500-record dataset
df = pd.read_csv("Global_AI_Dataset_Pro.csv")

# Create a sleek dark-themed map
# Create a high-impact map
fig = px.scatter_mapbox(
    df,
    lat='Latitude', lon='Longitude',
    size='Daily_Water_Liters',
    color='Water_Stress_Score',
    hover_name='Hub_Name',
    size_max=25,
    color_continuous_scale="RdYlGn_r",
    zoom=1.2, # Zoom level: 1 = World view
    mapbox_style="carto-darkmatter" # The professional "Dark Mode"
)

# Force the map to fill the screen
fig.update_layout(
    margin={"r":0,"t":0,"l":0,"b":0},
     mapbox=dict(
        style="carto-darkmatter",
        zoom=1,
        center=dict(lat=20, lon=0)
    ),
    coloraxis_colorbar=dict(title="Water Stress", x=0.9, y=0.5)
)

# Display in Streamlit
with st.container():
    st.plotly_chart(fig, use_container_width=True, theme=None)

# Add "Executive Summary" Metrics
col1, col2, col3 = st.columns(3)
col1.metric("Total AI Hubs", f"{len(df)}")
col2.metric("Total Daily Water Usage", f"{df['Daily_Water_Liters'].sum()/1e9:.2f} Billion Liters")
col3.metric("Total Lives Impacted", f"{df['Equivalent_Human_Lives'].sum():,}")

# Add the Filter
st.sidebar.header("Filter Analytics")
risk_level = st.sidebar.slider("Min. Water Risk Score", 0.0, 5.0, 3.0)
filtered_df = df[df['Water_Stress_Score'] >= risk_level]

st.subheader("High-Risk Data Centers (Critical Action Required)")
st.dataframe(filtered_df.sort_values(by='Water_Stress_Score', ascending=False).head(10))