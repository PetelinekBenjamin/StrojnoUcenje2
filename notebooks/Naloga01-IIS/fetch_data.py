import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import requests
import json


# URL
url = "https://api.jcdecaux.com/vls/v1/stations?contract=maribor&apiKey=5e150537116dbc1786ce5bec6975a8603286526b"

# Pridobivanje podatkov
response = requests.get(url)

# Preverjanje zahtevka
if response.status_code == 200:
    # json format
    data = response.json()

    with open('C:/Users/benja/Desktop/Stuff/Šola/Strojno ucenje2/data/raw/api_data.json', 'w') as f:
        json.dump(data, f)

    # Pretvorba podatkov v pandas DataFrame
    df = pd.DataFrame(data)

    # Izpis prvih nekaj vrstic DataFrame-a
    print(df.head())
else:
    print("Napaka pri pridobivanju podatkov. Koda napake:", response.status_code)

