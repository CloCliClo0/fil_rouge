import csv
from pathlib import Path

DOSSIER = Path(__file__).resolve().parent
CSV_METEO = DOSSIER / "demo_meteo.csv"
CSV_RESULTAT = DOSSIER / "resultat.csv"
CSV_STATS = DOSSIER / "stats_meteo.csv"

with open(CSV_METEO, "r", encoding="utf-8") as f:
    lecteur = csv.DictReader(f, delimiter=";")
    for ligne in lecteur:
        station = ligne["station"]
        date = ligne["date"]
        temperature = float(ligne["temperature"])
        print(f"{station} ({date}) : {temperature} °C")

lignes_filtrees = []

with open(CSV_METEO, "r", encoding="utf-8") as f:
    lecteur = csv.DictReader(f, delimiter=";")
    for ligne in lecteur:
        if float(ligne["temperature"]) > 10:
            lignes_filtrees.append(ligne)

with open(CSV_RESULTAT, "w", encoding="utf-8", newline="") as f:
    ecrivain = csv.DictWriter(
        f, fieldnames=["station", "date", "temperature", "humidite"], delimiter=";"
    )
    ecrivain.writeheader()
    ecrivain.writerows(lignes_filtrees)

temperatures_par_station = {}

with open(CSV_METEO, "r", encoding="utf-8") as f:
    lecteur = csv.DictReader(f, delimiter=";")
    for ligne in lecteur:
        station = ligne["station"]
        temperature = float(ligne["temperature"])
        temperatures_par_station.setdefault(station, []).append(temperature)

stats = []
for station, valeurs in temperatures_par_station.items():
    stats.append(
        {
            "station": station,
            "temperature_moyenne": round(sum(valeurs) / len(valeurs), 1),
            "temperature_max": max(valeurs),
            "temperature_min": min(valeurs),
        }
    )

with open(CSV_STATS, "w", encoding="utf-8", newline="") as f:
    ecrivain = csv.DictWriter(
        f,
        fieldnames=["station", "temperature_moyenne", "temperature_max", "temperature_min"],
        delimiter=";",
    )
    ecrivain.writeheader()
    ecrivain.writerows(stats)
