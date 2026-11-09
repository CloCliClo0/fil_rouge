# Fil Rouge

Projet météo : relevés SYNOP 2023 de la station de Lille-Lesquin (`meteo_lille_2023.csv`).

Dépendances : `pandas`, `matplotlib`, `scikit-learn` (séance 4).
Chaque séance s'exécute avec `python seanceN/projet_meteo.py`.

## fil rouge1
- Lecture, filtrage et agrégation par ville d'un CSV avec le module `csv`.
- Fichiers : `demo_meteo.csv` (entrée), `resultat.csv` et `stats_meteo.csv` (sorties).

## fil rouge2
- Chargement du fichier complet avec pandas (séparateur `;`, valeurs manquantes `mq`).
- Température en °C, date convertie, mois extrait, valeurs manquantes supprimées.
- Agrégation mensuelle de la température (moyenne, minimum, maximum).

## fil rouge3
- Figures d'exploration enregistrées dans `seance3/exports/` : température, précipitations, relations entre variables.
- Script découpé en fonctions, enchaînées par `main()`, avec validation des entrées.

## fil rouge4
- Régression linéaire de la température (`pres`, `u`, `ff`, `mois`, `heure`) avec référence naïve et variantes.
- Figure des erreurs dans `seance4/exports/erreurs_modele.png`.
