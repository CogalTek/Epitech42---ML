import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def train_ensemble_models():
    # Chargement des données préparées
    print("Chargement des données...")
    X_train = np.load('./data/X_train.npy')
    X_test = np.load('./data/X_test.npy')
    y_train = np.load('./data/y_train.npy')
    y_test = np.load('./data/y_test.npy')
    
    # Chargement des noms de caractéristiques
    with open('./data/feature_names.txt', 'r') as f:
        feature_names = [line.strip() for line in f.readlines()]

    print("Entraînement des modèles d'ensemble...")
    # Random Forest
    print("Entraînement du modèle Random Forest...")
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    # Prédictions avec Random Forest
    rf_pred = rf_model.predict(X_test)

    # Gradient Boosting
    print("Entraînement du modèle Gradient Boosting...")
    gb_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    gb_model.fit(X_train, y_train)

    # Prédictions avec Gradient Boosting
    gb_pred = gb_model.predict(X_test)

    # Évaluation des modèles
    models = {
        'Random Forest': {'model': rf_model, 'predictions': rf_pred},
        'Gradient Boosting': {'model': gb_model, 'predictions': gb_pred}
    }

    results = {}
    for name, model_info in models.items():
        y_pred = model_info['predictions']
        
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
    best_model = models[best_model_name]['model']

    print(f"Le meilleur modèle est {best_model_name} avec un R² de {results[best_model_name]['R²']:.2f}")

    # Analyse de l'importance des caractéristiques
    feature_importance = best_model.feature_importances_
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)

    plt.figure(figsize=(10, 8))
    sns.barplot(x='Importance', y='Feature', data=importance_df.head(15))
    plt.title(f'Top 15 des caractéristiques les plus importantes - {best_model_name}')
    plt.tight_layout()
    plt.savefig(f'./results/feature_importance_{best_model_name}.png')
    plt.close()

    # Visualisation des prédictions vs valeurs réelles
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, models[best_model_name]['predictions'], alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Valeurs réelles')
    plt.ylabel('Prédictions')
    plt.title(f'Prédictions vs Valeurs réelles - {best_model_name}')
    plt.tight_layout()
    plt.savefig(f'./results/predictions_vs_actual_{best_model_name}.png')
    plt.close()

    # Sauvegarde du modèle
    joblib.dump(best_model, f'./models/model2_{best_model_name}.pkl')
    
    # Sauvegarde des résultats pour comparaison ultérieure
    for name, result in results.items():
        # Supprimer les prédictions du dictionnaire à sauvegarder (trop volumineux)
        result_to_save = {k: v for k, v in result.items() if k != 'predictions'}
        np.save(f'./results/model2_{name}_results.npy', result_to_save)
    
    # Sauvegarde des prédictions du meilleur modèle pour la visualisation finale
    np.save('./results/model2_best_predictions.npy', results[best_model_name]['predictions'])
    
    print(f"Modèle 2 ({best_model_name}) entraîné et sauvegardé.")
    return best_model, results[best_model_name]['predictions']

if __name__ == "__main__":
    train_ensemble_models()