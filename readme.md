# Prédiction des Prix de l'Électricité pour Habo Plast

Ce projet développe trois modèles d'apprentissage automatique pour prédire les prix quotidiens de l'électricité dans la zone SE3 de Suède, en utilisant des données météorologiques et historiques.

## Structure du Projet
habo_plast_electricity_prediction/
│
├── data/                           # Dossier pour les données prétraitées
│
├── models/                         # Dossier pour les modèles entraînés
│
├── results/                        # Dossier pour les graphiques et résultats
│
├── data_preparation.py             # Script de préparation des données
├── model1_linear_regression.py     # Script pour le modèle de régression linéaire
├── model2_ensemble_methods.py      # Script pour les méthodes d'ensemble
├── model3_neural_networks.py       # Script pour les réseaux de neurones
├── model_comparison.py             # Script de comparaison des modèles
├── prediction_function.py          # Fonction de prédiction
├── main.py                         # Script principal
└── journals/                       # Journaux de projet individuels

## Installation et Prérequis

Pour exécuter ce projet, vous aurez besoin des packages Python suivants:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow joblib
```

## Données Requises
Le projet utilise les données suivantes qui doivent être placées dans le répertoire approprié:

Données de prix d'électricité: Fichiers CSV de prix ENTSO-E pour SE3 (2016-2024) dans ./habo_plast/electricity/
Données météorologiques SMHI: Fichiers CSV des stations dans ./habo_plast/smhi_data_2022-today/

## Exécution du Projet
Pour exécuter le projet complet:

```bash
python main.py
```

Cela exécutera séquentiellement:

- La préparation des données
- L'entraînement des trois modèles
- La comparaison des modèles
- La génération des visualisations et résultats

## Prédiction avec les Modèles Entraînés
Pour utiliser les modèles entraînés pour une prédiction:

```bash
from prediction_function import predict_electricity_price
from datetime import datetime

# Exemple d'utilisation
prediction = predict_electricity_price(
    date=datetime.now(),
    temperature=15.0,          # °C
    precipitation=2.5,         # mm
    snow_depth=0.0,            # cm
    sunshine_hours=8.0,        # heures
    historical_prices=[45.2, 47.8, 50.1, 49.3, 48.7, 51.2, 52.5]  # 7 derniers jours
)

print(prediction)
```

## Résultats
Les modèles développés dans ce projet ont atteint les performances suivantes sur l'ensemble de test:

ModèleR²RMSE (EUR/MWh)MAE (EUR/MWh)Ridge0.99871.72420.9641Random Forest0.815220.919510.5179MLP0.97557.62314.8921Ensemble (moyenne)0.96339.32865.0203

## Contributeurs

Rémi Maigrot
Mathieu Rio
Arthur Bourdin
Deepa Krishnan
Loshma

## Licence
Ce projet est développé dans le cadre académique de Jönköping University. Les données utilisées sont la propriété de leurs fournisseurs respectifs (ENTSO-E et SMHI).