import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

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

CIBLE = "t_celsius"
VARIABLES = ["pres", "u", "ff", "mois", "heure"]
TAILLE_TEST = 0.2
GRAINE = 42
MIN_LIGNES = 100
VARIANTES = {
    "référence (découpage aléatoire)": (VARIABLES, False),
    "découpage chronologique": (VARIABLES, True),
    "sans mois ni heure": (["pres", "u", "ff"], False),
    "avec t (fuite d'information)": (VARIABLES + ["t"], False),
}


def charger_donnees(chemin):
    return pd.read_csv(chemin, sep=";", na_values=["mq"])


def verifier_colonnes(df, colonnes_requises):
    manquantes = [c for c in colonnes_requises if c not in df.columns]
    if manquantes:
        raise ValueError(f"Colonnes absentes du fichier : {manquantes}")


def verifier_lignes(df, minimum=MIN_LIGNES):
    if len(df) < minimum:
        raise ValueError(f"{len(df)} ligne(s) disponible(s), minimum requis : {minimum}.")


def preparer_donnees(df):
    df = df[COLONNES_UTILES].copy()
    df["t_celsius"] = df["t"] - KELVIN
    df["date"] = pd.to_datetime(df["date"], format=FORMAT_DATE)
    df["mois"] = df["date"].dt.month
    return df


def ajouter_heure(df):
    df = df.copy()
    df["heure"] = df["date"].dt.hour
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


def preparer_variables(df, variables, cible):
    colonnes = variables + [cible]
    absentes = [c for c in colonnes if c not in df.columns]
    if absentes:
        raise ValueError(f"Colonnes absentes : {absentes}")
    donnees = df[colonnes].dropna()
    verifier_lignes(donnees)
    return donnees[variables], donnees[cible]


def decouper(X, y, chronologique=False):
    return train_test_split(
        X, y, test_size=TAILLE_TEST, random_state=GRAINE, shuffle=not chronologique
    )


def entrainer(X_train, y_train):
    return LinearRegression().fit(X_train, y_train)


def evaluer(modele, X_train, y_train, X_test, y_test):
    y_pred = modele.predict(X_test)
    reference = DummyRegressor(strategy="mean").fit(X_train, y_train)
    return {
        "mae_test": mean_absolute_error(y_test, y_pred),
        "rmse_test": np.sqrt(mean_squared_error(y_test, y_pred)),
        "r2_test": r2_score(y_test, y_pred),
        "mae_train": mean_absolute_error(y_train, modele.predict(X_train)),
        "mae_reference": mean_absolute_error(y_test, reference.predict(X_test)),
    }


def comparer_variantes(df, variantes):
    resultats = {}
    for nom, (variables, chronologique) in variantes.items():
        X, y = preparer_variables(df, variables, CIBLE)
        X_train, X_test, y_train, y_test = decouper(X, y, chronologique)
        modele = entrainer(X_train, y_train)
        resultats[nom] = evaluer(modele, X_train, y_train, X_test, y_test)
    return pd.DataFrame(resultats).T


def tracer_erreurs(y_test, y_pred, dates, chemin):
    residus = y_test - y_pred
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    limites = [y_test.min(), y_test.max()]
    axes[0, 0].scatter(y_pred, y_test, s=8)
    axes[0, 0].plot(limites, limites, color="red", linestyle="--", label="Prédiction parfaite")
    axes[0, 0].set_title("Température réelle contre prédite (jeu de test)")
    axes[0, 0].set_xlabel("Température prédite (°C)")
    axes[0, 0].set_ylabel("Température réelle (°C)")
    axes[0, 0].legend()

    residus.plot(kind="hist", bins=30, ax=axes[0, 1])
    axes[0, 1].set_title("Répartition des résidus (réel - prédit)")
    axes[0, 1].set_xlabel("Résidu (°C)")
    axes[0, 1].set_ylabel("Nombre de relevés")

    residus.groupby(dates.dt.hour).mean().plot(kind="bar", ax=axes[1, 0], rot=0)
    axes[1, 0].set_title("Résidu moyen par heure du relevé")
    axes[1, 0].set_xlabel("Heure (h)")
    axes[1, 0].set_ylabel("Résidu moyen (°C)")

    residus.groupby(dates.dt.month).mean().plot(kind="bar", ax=axes[1, 1], rot=0)
    axes[1, 1].set_title("Résidu moyen par mois")
    axes[1, 1].set_xlabel("Mois")
    axes[1, 1].set_ylabel("Résidu moyen (°C)")

    fig.tight_layout()
    fig.savefig(chemin, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    DOSSIER_EXPORTS.mkdir(exist_ok=True)

    try:
        brut = charger_donnees(CSV_METEO)
    except FileNotFoundError:
        sys.exit(f"Erreur : fichier de données introuvable : {CSV_METEO}. Vérifiez le nom et le chemin.")
    except pd.errors.EmptyDataError:
        sys.exit(f"Erreur : le fichier de données est vide : {CSV_METEO}.")

    try:
        verifier_colonnes(brut, COLONNES_UTILES)
        verifier_lignes(brut)
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

    df = ajouter_heure(df)
    try:
        X, y = preparer_variables(df, VARIABLES, CIBLE)
    except ValueError as exc:
        sys.exit(f"Erreur : {exc}")
    print(f"X : {X.shape}, y : {y.shape}, lignes perdues : {len(df) - len(X)}")

    X_train, X_test, y_train, y_test = decouper(X, y)
    print(f"Entraînement : {len(X_train)} lignes, test : {len(X_test)} lignes")

    modele = entrainer(X_train, y_train)
    y_pred = modele.predict(X_test)
    for nom, coefficient in zip(X.columns, modele.coef_):
        print(f"Coefficient {nom} : {coefficient:+.4f} °C par unité")
    print(f"Ordonnée à l'origine : {modele.intercept_:.2f} °C")

    metriques = evaluer(modele, X_train, y_train, X_test, y_test)
    print(f"Amplitude de la température : {y.max() - y.min():.1f} °C, écart-type : {y.std():.2f} °C")
    print(f"MAE test : {metriques['mae_test']:.2f} °C (référence moyenne : {metriques['mae_reference']:.2f} °C)")
    print(f"RMSE test : {metriques['rmse_test']:.2f} °C, R² test : {metriques['r2_test']:.2f}")
    print(f"MAE entraînement : {metriques['mae_train']:.2f} °C")

    tracer_erreurs(y_test, y_pred, df.loc[y_test.index, "date"], DOSSIER_EXPORTS / "erreurs_modele.png")
    periode = df.loc[decouper(X, y, chronologique=True)[1].index, "date"]
    print(f"Test chronologique : du {periode.min():%d/%m/%Y} au {periode.max():%d/%m/%Y}")
    print(comparer_variantes(df, VARIANTES).round(2).to_string())

    for fichier in sorted(DOSSIER_EXPORTS.glob("*.png")):
        print(f"{fichier.name} : {fichier.stat().st_size} octets")


if __name__ == "__main__":
    main()
