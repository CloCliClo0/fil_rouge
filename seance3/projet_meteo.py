import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent
CSV_METEO = ROOT.parent / "meteo_lille_2023.csv"
DOSSIER_EXPORTS = ROOT / "exports"
COLONNES_UTILES = ["date", "t", "pres", "u", "ff", "rr3"]
COLONNES_INTERET = ["t_celsius", "pres", "u", "ff", "rr3"]
FORMAT_DATE = "%Y%m%d%H%M%S"
KELVIN = 273.15
TEMPERATURE_MIN = -50
TEMPERATURE_MAX = 55
LIBELLES = {
    "pres": "Pression (Pa)",
    "u": "Humidité (%)",
    "ff": "Vent (m/s)",
    "rr3": "Précipitations (mm / 3 h)",
}


def charger_donnees(chemin):
    return pd.read_csv(chemin, sep=";", na_values=["mq"])


def verifier_colonnes(df, colonnes_requises):
    manquantes = [c for c in colonnes_requises if c not in df.columns]
    if manquantes:
        raise ValueError(f"Colonnes absentes du fichier : {manquantes}")


def preparer_donnees(df):
    df = df[COLONNES_UTILES].copy()
    df["t_celsius"] = df["t"] - KELVIN
    df["date"] = pd.to_datetime(df["date"], format=FORMAT_DATE)
    df["mois"] = df["date"].dt.month
    return df


def verifier_temperatures(df, minimum=TEMPERATURE_MIN, maximum=TEMPERATURE_MAX):
    valeurs = df["t_celsius"].dropna()
    n = (~valeurs.between(minimum, maximum)).sum()
    if n > 0:
        raise ValueError(f"{n} valeur(s) de t_celsius hors de [{minimum}, {maximum}] °C.")


def nettoyer_donnees(df):
    return df.dropna(subset=COLONNES_INTERET)


def compter_releves_absents(df):
    attendues = pd.date_range(df["date"].min(), df["date"].max(), freq="3h")
    return len(attendues.difference(pd.DatetimeIndex(df["date"])))


def agreger_mensuel(df):
    return df.groupby("mois")["t_celsius"].agg(["mean", "min", "max"])


def tracer_temperature(df, mensuel, chemin):
    fig, axes = plt.subplots(2, 1, figsize=(12, 9))

    df.plot(x="date", y="t_celsius", ax=axes[0], linewidth=0.5, legend=False)
    axes[0].set_title("Température à Lille-Lesquin en 2023 : un point toutes les 3 h")
    axes[0].set_xlabel("Date")
    axes[0].set_ylabel("Température (°C)")

    mensuel.plot(ax=axes[1], marker="o")
    axes[1].set_title("Température mensuelle à Lille-Lesquin en 2023")
    axes[1].set_xlabel("Mois")
    axes[1].set_ylabel("Température (°C)")
    axes[1].set_xticks(range(1, 13))
    axes[1].legend(["Moyenne", "Minimum", "Maximum"])

    fig.tight_layout()
    fig.savefig(chemin, dpi=150, bbox_inches="tight")
    plt.close(fig)


def stats_precipitations(df):
    return {
        "minimum": df["rr3"].min(),
        "maximum": df["rr3"].max(),
        "part_zeros": (df["rr3"] == 0).mean(),
        "negatives": int((df["rr3"] < 0).sum()),
    }


def remplacer_traces(df):
    df = df.copy()
    df.loc[df["rr3"] < 0, "rr3"] = 0.0
    return df


def tracer_distribution_precipitations(df, chemin):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    df["rr3"].plot(kind="hist", bins=20, ax=axes[0])
    axes[0].set_title("Précipitations sur 3 h : tous les relevés")

    df.loc[df["rr3"] > 0, "rr3"].plot(kind="hist", bins=20, ax=axes[1])
    axes[1].set_title("Précipitations sur 3 h : relevés pluvieux (> 0 mm)")

    for ax in axes:
        ax.set_xlabel("Précipitations (mm / 3 h)")
        ax.set_ylabel("Nombre de relevés")

    fig.tight_layout()
    fig.savefig(chemin, dpi=150, bbox_inches="tight")
    plt.close(fig)


def cumul_precipitations_mensuel(df):
    return df.groupby("mois")["rr3"].sum()


def tracer_precipitations_mensuelles(cumul, chemin):
    fig, ax = plt.subplots(figsize=(10, 5))

    cumul.plot(kind="bar", ax=ax, rot=0)
    ax.set_title("Cumul mensuel des précipitations à Lille-Lesquin en 2023")
    ax.set_xlabel("Mois")
    ax.set_ylabel("Précipitations (mm)")

    fig.tight_layout()
    fig.savefig(chemin, dpi=150, bbox_inches="tight")
    plt.close(fig)


def calculer_correlations(df):
    return df[COLONNES_INTERET].corr()


def variables_les_plus_correlees(correlations, n=2):
    return correlations["t_celsius"].drop("t_celsius").abs().nlargest(n).index.tolist()


def tracer_nuages(df, variables, chemin):
    fig, axes = plt.subplots(1, len(variables), figsize=(7 * len(variables), 5), squeeze=False)

    for ax, variable in zip(axes[0], variables):
        points = ax.scatter(
            df[variable], df["t_celsius"], c=df["date"].dt.hour, cmap="hsv", s=8
        )
        ax.set_title(f"Température en fonction de : {LIBELLES[variable]}")
        ax.set_xlabel(LIBELLES[variable])
        ax.set_ylabel("Température (°C)")
        fig.colorbar(points, ax=ax, label="Heure du relevé (h)")

    fig.tight_layout()
    fig.savefig(chemin, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    DOSSIER_EXPORTS.mkdir(exist_ok=True)

    try:
        brut = charger_donnees(CSV_METEO)
    except FileNotFoundError:
        sys.exit(f"Erreur : fichier de données introuvable : {CSV_METEO}. Vérifiez le nom et le chemin.")

    try:
        verifier_colonnes(brut, COLONNES_UTILES)
        df = preparer_donnees(brut)
        verifier_temperatures(df)
    except ValueError as exc:
        sys.exit(f"Erreur : {exc}")

    print(f"Relevés absents de la série : {compter_releves_absents(df)}")
    avant = len(df)
    df = nettoyer_donnees(df)
    print(f"Lignes supprimées par le nettoyage : {avant - len(df)} sur {avant}")

    mensuel = agreger_mensuel(df)
    print(mensuel)
    tracer_temperature(df, mensuel, DOSSIER_EXPORTS / "temperature_evolution.png")

    stats = stats_precipitations(df)
    print(f"rr3 : minimum {stats['minimum']} mm, maximum {stats['maximum']} mm")
    print(f"rr3 : {stats['part_zeros']:.1%} de zéros, {stats['negatives']} valeurs négatives")
    df = remplacer_traces(df)
    tracer_distribution_precipitations(df, DOSSIER_EXPORTS / "precipitations_histogramme.png")
    cumul = cumul_precipitations_mensuel(df)
    print(f"Mois le plus pluvieux : {cumul.idxmax()} ({cumul.max():.1f} mm)")
    tracer_precipitations_mensuelles(cumul, DOSSIER_EXPORTS / "precipitations_mensuelles.png")

    correlations = calculer_correlations(df)
    print(correlations.round(2))
    plus_correlees = variables_les_plus_correlees(correlations)
    print(f"Variables les plus corrélées à la température : {plus_correlees}")
    tracer_nuages(df, plus_correlees, DOSSIER_EXPORTS / "correlation_temperature.png")

    for fichier in sorted(DOSSIER_EXPORTS.glob("*.png")):
        print(f"{fichier.name} : {fichier.stat().st_size} octets")


if __name__ == "__main__":
    main()
