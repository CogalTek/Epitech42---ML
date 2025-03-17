# Manuel d'utilisation - Système de prédiction des prix de l'électricité pour Habo Plast

Ce manuel explique comment utiliser le système de prédiction des prix de l'électricité développé pour Habo Plast. Le système permet de prédire les prix quotidiens de l'électricité dans la zone SE3 de Suède en utilisant des données météorologiques et historiques.

## Installation

1. Assurez-vous que Python 3.8 ou supérieur est installé sur votre système.
2. Installez les dépendances requises:
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn tensorflow joblib
3. Clonez ou téléchargez le dépôt contenant le code source.

## Structure des fichiers
Le système est composé des fichiers suivants:

- data_preparation.py: Module de préparation des données
- model1_linear_regression.py: Implémentation du modèle de régression Ridge
- model2_ensemble_methods.py: Implémentation des méthodes d'ensemble (Random Forest)
- model3_neural_networks.py: Implémentation des réseaux de neurones
- model_comparison.py: Scripts de comparaison des modèles
- prediction_function.py: Fonction unifiée pour effectuer des prédictions
- main.py: Script principal pour exécuter l'ensemble du pipeline

## Exécution du pipeline complet
Pour exécuter l'ensemble du pipeline (préparation des données, entraînement des modèles, évaluation et comparaison):

```bash
python main.py
```

Cette commande:

1) Préparera les données à partir des sources brutes
2) Entraînera les trois modèles de prédiction
3) Évaluera et comparera leurs performances
4) Générera des visualisations et rapports dans le dossier results/

Le processus complet prend généralement environ 10-15 minutes selon les performances de votre ordinateur.

## Faire de nouvelles prédictions
Pour utiliser les modèles entraînés afin de prédire les prix de l'électricité pour une nouvelle journée, utilisez la fonction predict_electricity_price du module prediction_function.py:

```bash
from prediction_function import predict_electricity_price
from datetime import datetime

# Exemple de prédiction pour une journée spécifique
prediction = predict_electricity_price(
    date=datetime(2025, 3, 15),             # Date de la prédiction
    temperature=10.5,                       # Température moyenne (°C)
    precipitation=1.2,                      # Précipitations (mm)
    snow_depth=0.0,                         # Épaisseur de neige (cm)
    sunshine_hours=6.5,                     # Durée d'ensoleillement (heures)
    historical_prices=[45.2, 47.8, 50.1, 49.3, 48.7, 51.2, 52.5]  # Prix des 7 derniers jours
)

print(prediction)
```

Cette fonction retourne un dictionnaire contenant les prédictions de chaque modèle ainsi que la prédiction d'ensemble:

```bash
    'Linear Regression': 53.24,         # Prédiction du modèle Ridge
    'Ensemble Method': 55.12,           # Prédiction du Random Forest
    'Neural Network': 54.18,            # Prédiction du MLP
    'Ensemble (Average)': 54.18         # Moyenne des trois modèles
```

## Interprétation des résultats
Pour interpréter les résultats des prédictions:

1) Le modèle Linear Regression (Ridge) a montré les meilleures performances globales (R² = 0.9987) et devrait être considéré comme le plus fiable dans la plupart des situations.
2) Le modèle Neural Network (MLP) est également très précis (R² = 0.9755) et peut parfois mieux capturer les changements brusques de prix.
3) La prédiction Ensemble (Average) combine les trois modèles et peut offrir une robustesse supplémentaire contre les erreurs importantes d'un modèle individuel.
4) Utilisez la valeur MAE (Mean Absolute Error) pour estimer l'erreur typique des prédictions. Pour le modèle Ridge, l'erreur moyenne est d'environ 0.96 EUR/MWh.

## Mises à jour des modèles
Pour maintenir la précision des prédictions, il est recommandé de réentraîner les modèles régulièrement (idéalement tous les mois) avec les nouvelles données de prix et météorologiques. Pour ce faire:

1) Mettez à jour les fichiers CSV dans les dossiers ./habo_plast/electricity/ et ./habo_plast/smhi_data_2022-today/
2) Exécutez à nouveau le script principal:

```bash
python main.py
```

## Dépannage
Si vous rencontrez des problèmes:

1) Erreur lors du chargement des données: Vérifiez que les fichiers CSV sont présents dans les chemins attendus et ont le format correct.
2) Erreur lors de la prédiction: Assurez-vous que les modèles ont été correctement entraînés et que les fichiers de modèle sont présents dans le dossier ./models/.
3) Prédictions anormales: Si les prédictions semblent irréalistes (trop élevées ou trop basses), vérifiez que les valeurs d'entrée (température, précipitations, etc.) sont dans des plages raisonnables et que les prix historiques sont correctement ordonnés (du plus ancien au plus récent).

Pour toute assistance supplémentaire, contactez l'équipe de développement.