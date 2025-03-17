# main.py
# À placer à la racine de votre repo

import os
import time
from data_preparation import prepare_data
from model1_linear_regression import train_linear_models
from model2_ensemble_methods import train_ensemble_models
from model3_neural_networks import train_neural_networks
from model_comparison import compare_models
import pandas as pd
import matplotlib.pyplot as plt

def main():
    start_time = time.time()
    
    # Création des répertoires nécessaires
    os.makedirs('./data', exist_ok=True)
    os.makedirs('./models', exist_ok=True)
    os.makedirs('./results', exist_ok=True)
    
    print("=== Projet de prédiction des prix de l'électricité pour Habo Plast ===")
    
    # Étape 1: Préparation des données
    print("\n[Étape 1/5] Préparation des données...")
    X_train, X_test, y_train, y_test, feature_names = prepare_data()
    
    # Étape 2: Entraînement du modèle de régression linéaire
    print("\n[Étape 2/5] Entraînement du modèle de régression linéaire...")
    model1, pred1 = train_linear_models()
    
    # Étape 3: Entraînement des modèles d'ensemble
    print("\n[Étape 3/5] Entraînement des modèles d'ensemble...")
    model2, pred2 = train_ensemble_models()
    
    # Étape 4: Entraînement des réseaux de neurones
    print("\n[Étape 4/5] Entraînement des réseaux de neurones...")
    model3, pred3 = train_neural_networks()
    
    # Étape 5: Comparaison des modèles
    print("\n[Étape 5/5] Comparaison des modèles...")
    comparison_df = compare_models()
    
    # Afficher les résultats finaux
    print("\n=== Résultats de la comparaison des modèles ===")
    print(comparison_df[['Model', 'R²', 'RMSE', 'MAE']])
    
    # Identifier le meilleur modèle
    best_model = comparison_df.loc[comparison_df['R²'].idxmax()]
    print(f"\nLe meilleur modèle est {best_model['Model']} avec un R² de {best_model['R²']:.4f}")
    
    # Créer un résumé
    with open('./results/summary.txt', 'w') as f:
        f.write("=== Projet de prédiction des prix de l'électricité pour Habo Plast ===\n\n")
        f.write(f"Nombre de caractéristiques: {len(feature_names)}\n")
        f.write(f"Caractéristiques: {', '.join(feature_names)}\n\n")
        f.write("=== Résultats des modèles ===\n")
        f.write(comparison_df[['Model', 'R²', 'RMSE', 'MAE']].to_string(index=False))
        f.write(f"\n\nLe meilleur modèle est {best_model['Model']} avec un R² de {best_model['R²']:.4f}")
        f.write(f"\n\nTemps d'exécution total: {(time.time() - start_time)/60:.2f} minutes")
    
    print(f"\nProjet terminé en {(time.time() - start_time)/60:.2f} minutes.")
    print(f"Consultez les résultats dans le dossier './results/'")

if __name__ == "__main__":
    main()