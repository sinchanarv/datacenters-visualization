import streamlit as st
import pandas as pd
import geopandas as gpd
import folium
from streamlit_folium import st_folium
import plotly.express as px

# --- 1. DASHBOARD CONFIGURATION ---
st.set_page_config(page_title="AI Water Footprint", layout="wide")
st.title("💧 The Hidden Water Footprint of AI in India")
st.markdown("### Analyzing the intersection of Data Center expansion and Groundwater Depletion")
st.markdown("---")

# --- 2. LOAD DATA (Cached for speed) ---
@st.cache_data
def load_map_data():
    # Loads the lightweight file you just created!
    return gpd.read_file("india_water_risk.geojson")

india_map = load_map_data()

# Embedded Data Center Data
data = {
    'City_Name':['Chennai', 'Mumbai', 'Hyderabad', 'Bangalore', 'Pune'],
    'Latitude':[13.0827, 19.0760, 17.3850, 12.9716, 18.5204],
    'Longitude':[80.2707, 72.8777, 78.4867, 77.5946, 73.8567],
    'Estimated_Power_MW':[150, 200, 100, 120, 80]
}
df_centers = pd.DataFrame(data)

# --- 3. THE ANALYTICS (The Math) ---
# Formula based on Li et al. (2023): ~20,000 liters of water per 1 MW of Data Center power per day
df_centers['Water_Liters_Per_Day'] = df_centers['Estimated_Power_MW'] * 20000
# 1 human needs about 135 liters a day in India (Standard municipal supply)
df_centers['Equivalent_Human_Lives'] = (df_centers['Water_Liters_Per_Day'] / 135).astype(int)

# --- 4. THE VISUALIZATION (Danger Map) ---
st.subheader("📍 Geospatial Danger Map: Tech Hubs vs. Water Basins")

col1, col2 = st.columns([2, 1])

with col1:
    # Create the base map of India
    m = folium.Map(location=[20.5937, 78.9629], zoom_start=5, tiles="CartoDB positron")

    # Add the water basins (Just a light blue outline for our first test)
    folium.GeoJson(
        india_map,
        name="Water Risk Basins",
        style_function=lambda feature: {
            'color': 'blue',
            'weight': 0.5,
            'fillOpacity': 0.05
        }
    ).add_to(m)

    # Plot the Data Centers
    for idx, row in df_centers.iterrows():
        folium.Marker(
            location=[row['Latitude'], row['Longitude']],
            popup=f"<b>{row['City_Name']} Hub</b><br>{row['Estimated_Power_MW']} MW",
            tooltip="Click for Data Center Info",
            icon=folium.Icon(color="red", icon="info-sign")
        ).add_to(m)

    # Render map in Streamlit
    st_folium(m, width=700, height=500)

with col2:
    st.subheader("📊 The AI Math")
    st.write("According to recent literature, large-scale AI data centers require massive liquid cooling.")
    st.dataframe(df_centers[['City_Name', 'Water_Liters_Per_Day', 'Equivalent_Human_Lives']], hide_index=True)
    st.warning("⚠️ Notice how the water needed to cool Mumbai's data centers could supply over 29,000 citizens daily!")
    
 