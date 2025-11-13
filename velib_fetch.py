# velib_fetch.py
import os
import requests
import pandas as pd
from datetime import datetime

def get_data():
    nbrows = 1500
    url = (
        "https://opendata.paris.fr/api/records/1.0/search/"
        "?dataset=velib-disponibilite-en-temps-reel%40parisdata"
        f"&rows={nbrows}"
    )

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] Téléchargement des données...")

    resp = requests.get(url)
    if resp.status_code != 200:
        print(f"Erreur HTTP {resp.status_code}")
        return

    data = resp.json()
    records = data.get("records", [])
    if not records:
        print("Aucune donnée récupérée.")
        return

    df = pd.DataFrame([{
        'Timer': now,
        'ID': rec.get('recordid'),
        'Station': rec['fields'].get('name'),
        'Code Station': rec['fields'].get('stationcode'),
        'Ebikes': rec['fields'].get('ebike'),
        'Mechanical Bikes': rec['fields'].get('mechanical'),
        'Bikes Available': rec['fields'].get('numbikesavailable'),
        'Docks Available': rec['fields'].get('numdocksavailable'),
        'Capacity': rec['fields'].get('capacity'),
        'Coordonnees Geo': rec['fields'].get('coordonnees_geo'),
    } for rec in records])

    # Crée le dossier data si besoin
    os.makedirs("data", exist_ok=True)

    # Nom du fichier du jour
    filename = datetime.now().strftime("data/velib_data_%Y-%m-%d.csv")

    # Append si le fichier existe déjà
    header = not os.path.exists(filename)
    df.to_csv(filename, mode='a', index=False, header=header)
    print(f"[{now}] {len(df)} lignes enregistrées dans {filename}")

if __name__ == "__main__":
    get_data()
