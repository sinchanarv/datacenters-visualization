import geopandas as gpd
import warnings
import os

# Ignore annoying warnings
warnings.filterwarnings('ignore')

print("🚀 Starting Map Extraction...")

# 🌟 THE FIX: Automatically find the path based on where this script is saved!
current_directory = os.path.dirname(os.path.abspath(__file__))
gdb_folder_name = r"Aqueduct40_waterrisk_download_Y2023M07D05\GDB\Aq40_Y2023D07M05.gdb"
gdb_path = os.path.join(current_directory, gdb_folder_name)

print(f"📁 Looking for database at: {gdb_path}")
print("⏳ Loading massive global map data... (Please wait 1-2 minutes, it's a huge file!)")

try:
    # We use the pyogrio engine because it is much faster for reading GDB files
    global_map = gpd.read_file(gdb_path, engine="pyogrio")
    print("✅ Global Map Loaded Successfully!")

    # The WRI dataset uses 'gid_0' for Country Codes. India is 'IND'.
    print("✂️ Filtering out the rest of the world and keeping only India...")
    india_map = global_map[global_map['gid_0'] == 'IND']

    if not india_map.empty:
        print(f"✅ Success! Found {len(india_map)} water basin zones in India.")
        
        # Save it as a lightweight GeoJSON file in your project folder
        output_filename = "india_water_risk.geojson"
        india_map.to_file(output_filename, driver="GeoJSON")
        
        print(f"🎉 All done! Saved a new, lightweight file called: '{output_filename}'")
        print("You will never need to load the massive global file again!")
    else:
        print("❌ Uh oh. Couldn't find India. Let me know if you see this error.")
        
except Exception as e:
    print(f"❌ Error: {e}")