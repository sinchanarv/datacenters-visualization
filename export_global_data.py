import pandas as pd
import geopandas as gpd
import os
import warnings
warnings.filterwarnings('ignore')

print("🌍 Preparing Global AI Water Footprint Data...")

# 1. Global Hyperscale AI Data Center Hubs
data = {
    'Hub_Name':[
        'Ashburn, Virginia (US)', 'Phoenix, Arizona (US)', 'San Jose, California (US)', 
        'Des Moines, Iowa (US)', 'Dublin (Ireland)', 'Frankfurt (Germany)', 
        'London (UK)', 'Amsterdam (Netherlands)', 'Singapore', 
        'Tokyo (Japan)', 'Beijing (China)', 'Sydney (Australia)', 
        'Mumbai (India)', 'Chennai (India)', 'Oslo (Norway)', 'Santiago (Chile)'
    ],
    'Latitude':[
        39.0438, 33.4484, 37.3382, 41.5868, 53.3498, 50.1109, 
        51.5074, 52.3676, 1.3521, 35.6762, 39.9042, -33.8688, 
        19.0760, 13.0827, 59.9139, -33.4489
    ],
    'Longitude':[
        -77.4874, -112.0740, -121.8863, -93.6250, -6.2603, 8.6821, 
        -0.1278, 4.9041, 103.8198, 139.6503, 116.4074, 151.2093, 
        72.8777, 80.2707, 10.7522, -70.6693
    ],
    'Estimated_MW':[
        2500, 1200, 800, 500, 900, 1100, 
        850, 700, 1000, 950, 1500, 600, 
        200, 150, 100, 120
    ]
}
df_hubs = pd.DataFrame(data)

# Convert our hubs into a Geospatial format
gdf_hubs = gpd.GeoDataFrame(
    df_hubs, 
    geometry=gpd.points_from_xy(df_hubs.Longitude, df_hubs.Latitude),
    crs="EPSG:4326"
)

# 2. Load the Massive Global WRI Database
current_directory = os.path.dirname(os.path.abspath(__file__))
gdb_path = os.path.join(current_directory, r"Aqueduct40_waterrisk_download_Y2023M07D05\GDB\Aq40_Y2023D07M05.gdb")

print("⏳ Reading Global Water Database (This takes a minute)...")
global_map = gpd.read_file(gdb_path, engine="pyogrio")

# 3. Perform a Spatial Join (Finding which drought zone each hub is in!)
print("📍 Intersecting AI Hubs with Drought Zones...")
# We do an intersection to grab the exact water risk score (bws_score) for each city's coordinates
joined_data = gpd.sjoin(gdf_hubs, global_map[['bws_score', 'bws_cat', 'geometry']], how="left", predicate="intersects")

# 4. Do the AI Water Math
# Formula: 20,000 liters of water per 1 MW per day
joined_data['Daily_Water_Liters'] = joined_data['Estimated_MW'] * 20000
# 1 Human needs ~135 liters a day
joined_data['Equivalent_Human_Lives'] = (joined_data['Daily_Water_Liters'] / 135).astype(int)

# Clean up the final table
final_df = joined_data[['Hub_Name', 'Latitude', 'Longitude', 'Estimated_MW', 'bws_score', 'bws_cat', 'Daily_Water_Liters', 'Equivalent_Human_Lives']]

# Save it!
final_df.to_csv("Global_AI_Water_Footprint.csv", index=False)
print("✅ SUCCESS! Saved as 'Global_AI_Water_Footprint.csv'")
print("You are ready for Power BI / Tableau!")