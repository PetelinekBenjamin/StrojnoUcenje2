import pandas as pd
import json
import os

# Branje JSON datoteke s podatki o postajah za izposojo koles
with open('C:/Users/benja/Desktop/Stuff/Sola/Strojno ucenje2/data/raw/api_data.json', 'r') as f:
    data = json.load(f)

# Pretvorba podatkov v DataFrame
df = pd.DataFrame(data)

# Pretvorba datetime
df['last_update'] = pd.to_datetime(df['last_update'], unit='ms')

# Shranjevanje v mapi 'processed'
output_directory = 'C:/Users/benja/Desktop/Stuff/Sola/Strojno ucenje2/data/processed'
os.makedirs(output_directory, exist_ok=True)  # Ustvari mapo, če še ne obstaja

# Za vsako postajališče, obdelava in shranjevanje v CSV
for name, group in df.groupby('name'):
    # ime datoteke
    filename = os.path.join(output_directory, f"{name.replace(' ', '_').lower()}.csv")

    # Shranimo
    group.to_csv(filename, index=False)
