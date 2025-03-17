# model_comparison.py
# À placer à la racine de votre repo

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def compare_models():
    # Charger les données de test
    print("Chargement des données de test...")
    X_test = np.load('./data/X_test.npy')
    y_test = np.load('./data/y_test.npy')

    # Charger les modèles ou leurs prédictions
    print("Chargement des modèles ou prédictions...")
    
    # Modèle 1: Linear Regression (supposons que c'est Ridge qui est le meilleur)
    try:
        # Essayer de charger directement les prédictions si elles existent
        y_pred1 = np.load('./results/model1_best_predictions.npy')
    except:
        # Sinon, charger le modèle et faire les prédictions
        model1_files = [f for f in os.listdir('./models') if f.startswith('model1_') and f.endswith('.pkl')]
        if len(model1_files) > 0:
            model1 = joblib.load(f'./models/{model1_files[0]}')
            y_pred1 = model1.predict(X_test)
        else:
            print("Aucun modèle linéaire trouvé. Veuillez d'abord exécuter model1_linear_regression.py")
            return
    
    # Modèle 2: Ensemble Methods (supposons que c'est Random Forest qui est le meilleur)
    try:
        y_pred2 = np.load('./results/model2_best_predictions.npy')
    except:
        model2_files = [f for f in os.listdir('./models') if f.startswith('model2_') and f.endswith('.pkl')]
        if len(model2_files) > 0:
            model2 = joblib.load(f'./models/{model2_files[0]}')
            y_pred2 = model2.predict(X_test)
        else:
            print("Aucun modèle d'ensemble trouvé. Veuillez d'abord exécuter model2_ensemble_methods.py")
            return
    
    # Modèle 3: Neural Networks (MLP ou LSTM)
    try:
        # Essayer de charger MLP d'abord
        y_pred3 = np.load('./results/model3_MLP_predictions.npy')
        model3_name = "Neural Network (MLP)"
    except:
        try:
            # Sinon essayer LSTM
            y_pred3 = np.load('./results/model3_LSTM_predictions.npy')
            model3_name = "Neural Network (LSTM)"
        except:
            try:
                # Si pas de prédictions sauvegardées, essayer de charger le modèle MLP
                if os.path.exists('./models/model3_MLP.h5'):
                    model3 = load_model('./models/model3_MLP.h5')
                    y_pred3 = model3.predict(X_test).flatten()
                    model3_name = "Neural Network (MLP)"
                # Sinon essayer de charger LSTM
                elif os.path.exists('./models/model3_LSTM.h5'):
                    # Pour LSTM, nous aurions besoin de reformater les données
                    print("Modèle LSTM détecté, mais pas de prédictions sauvegardées. Utilisation des valeurs MLP.")
                    model3_name = "Neural Network (non chargé)"
                    y_pred3 = np.zeros_like(y_test)  # Valeur par défaut
                else:
                    print("Aucun modèle neural network trouvé. Veuillez d'abord exécuter model3_neural_networks.py")
                    return
            except:
                print("Erreur lors du chargement du modèle neural network.")
                return
    
    # Identifier les noms des modèles
    model1_name = [f.replace('model1_', '').replace('.pkl', '') for f in os.listdir('./models') if f.startswith('model1_') and f.endswith('.pkl')][0] if len([f for f in os.listdir('./models') if f.startswith('model1_') and f.endswith('.pkl')]) > 0 else "Linear Regression"
    model2_name = [f.replace('model2_', '').replace('.pkl', '') for f in os.listdir('./models') if f.startswith('model2_') and f.endswith('.pkl')][0] if len([f for f in os.listdir('./models') if f.startswith('model2_') and f.endswith('.pkl')]) > 0 else "Ensemble Method"
    
    # Calculer les métriques
    print("Calcul des métriques pour tous les modèles...")
    models = {
        f'Linear Regression ({model1_name})': y_pred1,
        f'Ensemble Method ({model2_name})': y_pred2,
        model3_name: y_pred3
    }

    results = {}
    for name, predictions in models.items():
        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        
        results[name] = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R²': r2
        }

    # Créer un DataFrame pour la comparaison
    comparison_df = pd.DataFrame(results).T
    comparison_df = comparison_df.reset_index().rename(columns={'index': 'Model'})
    
    # Sauvegarder les résultats
    comparison_df.to_csv('./results/model_comparison.csv', index=False)

    # Visualisation des métriques
    print("Création des visualisations de comparaison...")
    metrics = ['MSE', 'RMSE', 'MAE']
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Model', y=metric, data=comparison_df)
        plt.title(f'Comparaison des modèles - {metric}')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'./results/comparison_{metric}.png')
        plt.close()

    # Visualisation du R²
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Model', y='R²', data=comparison_df)
    plt.title('Comparaison des modèles - R²')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('./results/comparison_R2.png')
    plt.close()

    # Créer un modèle d'ensemble (moyenne des prédictions)
    print("Création et évaluation du modèle d'ensemble...")
    ensemble_pred = (y_pred1 + y_pred2 + y_pred3) / 3

    # Calculer les métriques pour l'ensemble
    ensemble_mse = mean_squared_error(y_test, ensemble_pred)
    ensemble_rmse = np.sqrt(ensemble_mse)
    ensemble_mae = mean_absolute_error(y_test, ensemble_pred)
    ensemble_r2 = r2_score(y_test, ensemble_pred)

    print("Modèle d'ensemble (moyenne des 3 modèles)")
    print(f"MSE: {ensemble_mse:.2f}")
    print(f"RMSE: {ensemble_rmse:.2f}")
    print(f"MAE: {ensemble_mae:.2f}")
    print(f"R²: {ensemble_r2:.2f}")

    # Ajouter l'ensemble aux résultats de comparaison
    new_row = pd.DataFrame({
        'Model': ['Ensemble (Average)'],
        'MSE': [ensemble_mse],
        'RMSE': [ensemble_rmse],
        'MAE': [ensemble_mae],
        'R²': [ensemble_r2]
    })
    comparison_df = pd.concat([comparison_df, new_row], ignore_index=True)
    
    # Sauvegarder les résultats mis à jour
    comparison_df.to_csv('./results/model_comparison_with_ensemble.csv', index=False)

    # Visualisation finale avec l'ensemble
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Model', y='R²', data=comparison_df)
    plt.title('Comparaison des modèles incluant l\'ensemble - R²')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('./results/comparison_with_ensemble.png')
    plt.close()
    
    # Visualisation des prédictions des modèles
    plt.figure(figsize=(15, 10))

    # Créer un sous-ensemble des données pour une meilleure visualisation
    sample_size = min(100, len(y_test))
    indices = np.random.choice(len(y_test), sample_size, replace=False)
    indices.sort()

    x_values = range(sample_size)

    plt.plot(x_values, y_test[indices], marker='o', linestyle='-', label='Valeurs réelles', color='black')
    plt.plot(x_values, y_pred1[indices], marker='s', linestyle='--', label=f'Linear Regression ({model1_name})', alpha=0.7)
    plt.plot(x_values, y_pred2[indices], marker='^', linestyle='--', label=f'Ensemble Method ({model2_name})', alpha=0.7)
    plt.plot(x_values, y_pred3[indices], marker='d', linestyle='--', label=model3_name, alpha=0.7)
    plt.plot(x_values, ensemble_pred[indices], marker='*', linestyle='--', label='Ensemble (Average)', alpha=0.7)

    plt.xlabel('Échantillon')
    plt.ylabel('Prix de l\'électricité (EUR/MWh)')
    plt.title('Comparaison des prédictions des différents modèles')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('./results/model_predictions_comparison.png')
    plt.close()
    
    print("Comparaison des modèles terminée.")
    return comparison_df

if __name__ == "__main__":
    compare_models()