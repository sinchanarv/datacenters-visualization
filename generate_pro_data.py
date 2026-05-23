import pandas as pd
import numpy as np

years = np.arange(2020, 2031)
hubs =["Mumbai", "Chennai", "Phoenix", "London", "Frankfurt", "Singapore", "Virginia", "Dublin", "Tokyo"]
data =[]

for year in years:
    for hub in hubs:
        # Simulate growth: Data Centers increase by 10% each year
        count = int(20 * (1.1 ** (year - 2020)))
        for i in range(count):
            power_mw = np.random.randint(50, 500)
            # WUI = Water Usage Intensity (Liters per MWh)
            wui = np.random.uniform(20, 200) 
            data.append({
                'Year': year,
                'Hub': hub,
                'Power_MW': power_mw,
                'WUI': wui,
                'Water_Liters': power_mw * wui * 24, # Daily usage
                'Water_Stress_Score': np.random.uniform(2, 5),
                'Pop_Density': np.random.randint(1000, 50000)
            })

df = pd.DataFrame(data)
# Add the 'Vulnerability Index' (Your Unique Formula)
df['Vulnerability_Index'] = (df['Water_Stress_Score'] * df['Water_Liters']) / df['Pop_Density']
df.to_csv("Big_Analytics_Master.csv", index=False)
print("✅ Created Big_Analytics_Master.csv with 10 years of professional data!")