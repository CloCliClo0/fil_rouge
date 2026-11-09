from pathlib import Path

import pandas as pd

CSV_METEO = Path(__file__).resolve().parent.parent / "meteo_lille_2023.csv"

df = pd.read_csv(CSV_METEO, sep=";", na_values=["mq"])
print(f"Lignes : {df.shape[0]}, colonnes : {df.shape[1]}")

df = df[["date", "t", "pres", "u", "ff", "rr3"]].copy()
df["t_celsius"] = df["t"] - 273.15
print(df.dtypes)

df["date"] = pd.to_datetime(df["date"], format="%Y%m%d%H%M%S")
df["mois"] = df["date"].dt.month

print(df.isna().sum())
avant = len(df)
df = df.dropna(subset=["t_celsius", "pres", "u", "ff", "rr3"])
print(f"Lignes supprimees : {avant - len(df)} sur {avant}")

mensuel = df.groupby("mois")["t_celsius"].agg(["mean", "min", "max"])
print(mensuel)
