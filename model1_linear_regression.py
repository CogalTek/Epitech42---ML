import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def train_linear_models():
    # Chargement des données préparées
    print("Chargement des données...")
    X_train = np.load('./data/X_train.npy')
    X_test = np.load('./data/X_test.npy')
    y_train = np.load('./data/y_train.npy')
    y_test = np.load('./data/y_test.npy')
    
    # Charger les noms des caractéristiques
    with open('./data/feature_names.txt', 'r') as f:
        feature_names = [line.strip() for line in f.readlines()]

    # Liste des modèles à tester
    print("Entraînement des modèles linéaires...")
    models = {
        'Ridge': Ridge(alpha=1.0),
        'Lasso': Lasso(alpha=0.1),
        'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5)
    }

    # Évaluation des modèles
    results = {}
    for name, model in models.items():
        print(f"Entraînement du modèle {name}...")
        # Entraînement du modèle
        model.fit(X_train, y_train)
        
        # Prédictions
        y_pred = model.predict(X_test)
        
        # Métriques d'évaluation
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        results[name] = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R²': r2,
            'predictions': y_pred
        }
        
        print(f"Modèle: {name}")
        print(f"MSE: {mse:.2f}")
        print(f"RMSE: {rmse:.2f}")
        print(f"MAE: {mae:.2f}")
        print(f"R²: {r2:.2f}")
        print("-" * 50)

    # Choisir le meilleur modèle (en fonction du R²)
    best_model_name = max(results, key=lambda x: results[x]['R²'])
    best_model = models[best_model_name]

    print(f"Le meilleur modèle est {best_model_name} avec un R² de {results[best_model_name]['R²']:.2f}")

    # Analyse des coefficients pour comprendre l'importance des caractéristiques
    coefficients = best_model.coef_

    # Créer un DataFrame pour visualiser les coefficients
    coef_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefficients})
    coef_df = coef_df.sort_values(by='Coefficient', ascending=False)

    plt.figure(figsize=(10, 8))
    sns.barplot(x='Coefficient', y='Feature', data=coef_df)
    plt.title(f'Importance des caractéristiques - {best_model_name}')
    plt.tight_layout()
    plt.savefig(f'./results/feature_importance_{best_model_name}.png')
    plt.close()

    # Sauvegarde du modèle
    joblib.dump(best_model, f'./models/model1_{best_model_name}.pkl')
    
    # Sauvegarde des résultats pour comparaison ultérieure
    for name, result in results.items():
        # Supprimer les prédictions du dictionnaire à sauvegarder (trop volumineux)
        result_to_save = {k: v for k, v in result.items() if k != 'predictions'}
        np.save(f'./results/model1_{name}_results.npy', result_to_save)
    
    # Sauvegarde des prédictions du meilleur modèle pour la visualisation finale
    np.save('./results/model1_best_predictions.npy', results[best_model_name]['predictions'])
    
    print(f"Modèle 1 ({best_model_name}) entraîné et sauvegardé.")
    return best_model, results[best_model_name]['predictions']

if __name__ == "__main__":
    train_linear_models()