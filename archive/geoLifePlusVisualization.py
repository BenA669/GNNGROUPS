import folium
import pandas as pd

# Example fix
data = pd.read_csv("Data/staypoints/staypoints.tsv")

# Inspect column names
print(data.columns)

# Replace `Latitude` and `Longitude` with actual column names
m = folium.Map(location=[data['Latitude'].mean(), data['Longitude'].mean()], zoom_start=12)

for user_id, user_data in data.groupby("UserId"):
    coords = list(zip(user_data['Latitude'], user_data['Longitude']))
    folium.PolyLine(coords, color="blue", weight=2, opacity=0.5).add_to(m)

m.save("map.html")
