# model3_neural_networks.py
# À placer à la racine de votre repo

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.callbacks import EarlyStopping

def train_neural_networks():
    # Chargement des données préparées
    print("Chargement des données...")
    X_train = np.load('./data/X_train.npy')
    X_test = np.load('./data/X_test.npy')
    y_train = np.load('./data/y_train.npy')
    y_test = np.load('./data/y_test.npy')

    # Modèle 3.1: Perceptron multicouche (MLP)
    def create_mlp_model(input_dim):
        model = Sequential()
        model.add(Dense(64, activation='relu', input_dim=input_dim))
        model.add(Dropout(0.2))
        model.add(Dense(32, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(1))
        
        model.compile(loss='mse', optimizer='adam')
        return model

    # Définir early stopping pour éviter le surapprentissage
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # Créer et entraîner le modèle MLP
    print("Entraînement du modèle MLP...")
    mlp_model = create_mlp_model(X_train.shape[1])
    mlp_history = mlp_model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=1
    )

    # Prédictions avec MLP
    mlp_pred = mlp_model.predict(X_test).flatten()

    # Modèle 3.2: LSTM (pour capturer les dépendances temporelles)
    # Restructurer les données pour LSTM (besoin d'une dimension temporelle)
    def prepare_lstm_data(X, y, time_steps=7):
        X_lstm, y_lstm = [], []
        for i in range(len(X) - time_steps):
            X_lstm.append(X[i:i + time_steps])
            y_lstm.append(y[i + time_steps])
        return np.array(X_lstm), np.array(y_lstm)

    # Si nous avons assez de données pour utiliser LSTM
    lstm_trained = False
    lstm_pred = None
    lstm_model = None
    if len(X_train) > 30:  # Au moins quelques séquences
        # Créer séquences pour LSTM
        time_steps = min(7, X_train.shape[0] // 3)  # Au plus 7 jours, ou 1/3 des données
        X_train_lstm, y_train_lstm = prepare_lstm_data(X_train, y_train, time_steps)
        X_test_lstm, y_test_lstm = prepare_lstm_data(X_test, y_test, time_steps)
        
        if len(X_train_lstm) > 0 and len(X_test_lstm) > 0:
            # Créer modèle LSTM
            def create_lstm_model(time_steps, features):
                model = Sequential()
                model.add(LSTM(50, return_sequences=True, input_shape=(time_steps, features)))
                model.add(Dropout(0.2))
                model.add(LSTM(50))
                model.add(Dropout(0.2))
                model.add(Dense(1))
                model.compile(loss='mse', optimizer='adam')
                return model
            
            # Entraîner LSTM
            print("Entraînement du modèle LSTM...")
            lstm_model = create_lstm_model(time_steps, X_train.shape[1])
            lstm_history = lstm_model.fit(
                X_train_lstm, y_train_lstm,
                epochs=100,
                batch_size=32,
                validation_split=0.2,
                callbacks=[early_stopping],
                verbose=1
            )
            
            # Prédictions avec LSTM
            lstm_pred = lstm_model.predict(X_test_lstm).flatten()
            
            # Évaluation LSTM
            lstm_mse = mean_squared_error(y_test_lstm, lstm_pred)
            lstm_rmse = np.sqrt(lstm_mse)
            lstm_mae = mean_absolute_error(y_test_lstm, lstm_pred)
            lstm_r2 = r2_score(y_test_lstm, lstm_pred)
            
            print("Modèle: LSTM")
            print(f"MSE: {lstm_mse:.2f}")
            print(f"RMSE: {lstm_rmse:.2f}")
            print(f"MAE: {lstm_mae:.2f}")
            print(f"R²: {lstm_r2:.2f}")
            print("-" * 50)
            
            # Sauvegarder modèle LSTM
            lstm_model.save('./models/model3_LSTM.h5')
            
            # Sauvegarder les résultats LSTM
            lstm_results = {
                'MSE': lstm_mse,
                'RMSE': lstm_rmse,
                'MAE': lstm_mae,
                'R²': lstm_r2
            }
            np.save('./results/model3_LSTM_results.npy', lstm_results)
            
            # Sauvegarde des prédictions LSTM
            np.save('./results/model3_LSTM_predictions.npy', lstm_pred)
            
            # Visualisation LSTM
            plt.figure(figsize=(10, 6))
            plt.scatter(y_test_lstm, lstm_pred, alpha=0.5)
            plt.plot([y_test_lstm.min(), y_test_lstm.max()], [y_test_lstm.min(), y_test_lstm.max()], 'r--')
            plt.xlabel('Valeurs réelles')
            plt.ylabel('Prédictions')
            plt.title('Prédictions vs Valeurs réelles - LSTM')
            plt.tight_layout()
            plt.savefig('./results/predictions_vs_actual_LSTM.png')
            plt.close()
            
            lstm_trained = True

    # Évaluation MLP
    mlp_mse = mean_squared_error(y_test, mlp_pred)
    mlp_rmse = np.sqrt(mlp_mse)
    mlp_mae = mean_absolute_error(y_test, mlp_pred)
    mlp_r2 = r2_score(y_test, mlp_pred)

    print("Modèle: MLP")
    print(f"MSE: {mlp_mse:.2f}")
    print(f"RMSE: {mlp_rmse:.2f}")
    print(f"MAE: {mlp_mae:.2f}")
    print(f"R²: {mlp_r2:.2f}")
    print("-" * 50)

    # Sauvegarder modèle MLP
    mlp_model.save('./models/model3_MLP.h5')
    
    # Sauvegarder les résultats MLP
    mlp_results = {
        'MSE': mlp_mse,
        'RMSE': mlp_rmse,
        'MAE': mlp_mae,
        'R²': mlp_r2
    }
    np.save('./results/model3_MLP_results.npy', mlp_results)
    
    # Sauvegarde des prédictions MLP
    np.save('./results/model3_MLP_predictions.npy', mlp_pred)

    # Visualisation de l'apprentissage MLP
    plt.figure(figsize=(10, 6))
    plt.plot(mlp_history.history['loss'], label='Train')
    plt.plot(mlp_history.history['val_loss'], label='Validation')
    # Visualisation de l'apprentissage MLP
    plt.figure(figsize=(10, 6))
    plt.plot(mlp_history.history['loss'], label='Train')
    plt.plot(mlp_history.history['val_loss'], label='Validation')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (MSE)')
    plt.title('Courbe d\'apprentissage - MLP')
    plt.legend()
    plt.savefig('./results/learning_curve_MLP.png')
    plt.close()

    # Visualisation des prédictions MLP
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, mlp_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Valeurs réelles')
    plt.ylabel('Prédictions')
    plt.title('Prédictions vs Valeurs réelles - MLP')
    plt.tight_layout()
    plt.savefig('./results/predictions_vs_actual_MLP.png')
    plt.close()
    
    # Retourner le meilleur modèle neural network et ses prédictions
    if lstm_trained and mlp_results['R²'] < lstm_results['R²']:
        print(f"Modèle 3 (LSTM) entraîné et sauvegardé avec R² de {lstm_results['R²']:.2f}")
        return lstm_model, lstm_pred
    else:
        print(f"Modèle 3 (MLP) entraîné et sauvegardé avec R² de {mlp_results['R²']:.2f}")
        return mlp_model, mlp_pred

if __name__ == "__main__":
    train_neural_networks()